"""Partie 6 : Grad-CAM, segmentation de la feuille et test du biais de fond."""
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gradcam_maps(model, target_layer, x, targets=None):
    """Cartes Grad-CAM (N×H×W dans [0, 1]) pour un batch x déjà normalisé.
    targets : liste d'indices de classe (None = classe prédite)."""
    model.eval()
    model.requires_grad_(True)   # nécessaire si le backbone était gelé : il faut un gradient jusqu'à la couche cible
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        t = None if targets is None else [ClassifierOutputTarget(int(c)) for c in targets]
        return cam(input_tensor=x, targets=t)


def overlay(img_uint8, cam, alpha=0.45):
    """Superpose une carte de chaleur (colormap JET) à l'image RGB."""
    h = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    h = cv2.resize(h, (img_uint8.shape[1], img_uint8.shape[0]))
    return (alpha * h + (1 - alpha) * img_uint8).astype(np.uint8)


def leaf_mask(img_uint8):
    """Masque binaire de la feuille par seuillage HSV + morphologie.

    - la feuille (verte, jaune ou brune) est plus saturée que le fond gris/blanc du labo :
      seuil d'Otsu sur la saturation S
    - fermeture puis ouverture morphologiques pour boucher les taches et retirer le bruit
    - on garde la plus grande composante connexe et on remplit ses trous"""
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == biggest, 255, 0).astype(np.uint8)
    # remplissage des trous : marge de 1 px pour que le remplissage parte toujours de l'extérieur
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    outside = pad.copy()
    cv2.floodFill(outside, ff, (0, 0), 255)
    holes = cv2.bitwise_not(outside)[1:-1, 1:-1]
    return (m | holes) > 0


def soil_texture(h, w, rng, photos=None):
    """Texture de sol : crop d'une photo de sol si disponible, sinon texture procédurale
    (brun de base + bruit multi-échelle flouté + mouchetures)."""
    if photos:
        img = cv2.cvtColor(cv2.imread(str(photos[rng.integers(len(photos))])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (max(w, img.shape[1] * h // img.shape[0]), h))
        x0 = rng.integers(0, img.shape[1] - w + 1)
        return img[:, x0:x0 + w]
    base = np.array([115, 88, 62], np.float32) * rng.uniform(0.8, 1.15)
    tex = np.zeros((h, w), np.float32)
    for scale, amp in ((3, 30), (9, 22), (25, 14)):
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        tex += amp * cv2.GaussianBlur(n, (0, 0), scale) / (cv2.GaussianBlur(n, (0, 0), scale).std() + 1e-6)
    speck = (rng.random((h, w)) < 0.02) * rng.uniform(-60, 60, (h, w))
    out = base[None, None, :] + (tex + speck)[..., None] * np.array([1.0, 0.85, 0.7])
    return np.clip(out, 0, 255).astype(np.uint8)


def replace_background(img_uint8, mask, mode, rng, photos=None):
    """Remplace le fond (hors masque) par : 'noise' (bruit uniforme), 'soil' (sol) ou 'black'."""
    h, w = mask.shape
    if mode == "noise":
        bg = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif mode == "soil":
        bg = soil_texture(h, w, rng, photos)
    elif mode == "black":
        bg = np.zeros((h, w, 3), np.uint8)
    else:
        raise ValueError(mode)
    m = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]   # bord adouci
    return (m * img_uint8 + (1 - m) * bg).astype(np.uint8)


def cam_mass_on_leaf(cam, mask):
    """Part de l'énergie Grad-CAM située sur la feuille (1 = tout sur la feuille)."""
    cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))
    return float((cam * mask).sum() / (cam.sum() + 1e-8))


def to_model_input(batch_uint8, norm, device):
    """Liste/array N×H×W×3 uint8 -> tenseur normalisé N×3×H×W."""
    x = torch.from_numpy(np.stack(batch_uint8)).permute(0, 3, 1, 2).float().div(255)
    m = torch.tensor(norm["mean"]).view(1, 3, 1, 1)
    s = torch.tensor(norm["std"]).view(1, 3, 1, 1)
    return ((x - m) / s).to(device)
