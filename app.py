"""Démo DiagnoCulture (Gradio) : photo de feuille -> maladie, ou « inconnu », avec carte Grad-CAM.

Lancement : python app.py   (ou depuis le notebook 07)
Fichiers requis (produits par les notebooks 02 à 04) :
  results/ckpt/clf_sans_M.pt, results/ckpt/ae_connus.pt, results/hybrid_config.json
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C                      # noqa: E402
from src.data import short                       # noqa: E402
from src.models import load_classifier, build_ae  # noqa: E402
from src import explain as X                     # noqa: E402

CONSEIL_INCONNU = ("Le système ne reconnaît pas cette image avec assez de confiance. "
                   "Ne pas traiter sur cette base : consulter un agent agricole.")


class Diagnostiqueur:
    def __init__(self, device="cpu"):
        self.cfg = C.load_json(C.RESULTS / "hybrid_config.json")
        self.device = torch.device(device)
        self.clf, _ = load_classifier(C.CKPT_DIR / self.cfg["clf_ckpt"], self.device)
        # auto-encodeur des classes connues (notebook 04), même architecture que celui de la partie 3
        self.ae = build_ae(self.cfg["ae_k"]).to(self.device)
        self.ae.load_state_dict(torch.load(C.CKPT_DIR / self.cfg["ae_ckpt"], map_location=self.device))
        self.ae.eval()

    def _square(self, img: Image.Image, size: int) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)

    @torch.inference_mode()
    def scores(self, img: Image.Image):
        x_clf = self._square(img, self.cfg["clf_size"])
        x_ae = self._square(img, self.cfg["ae_size"])
        prob = torch.softmax(self.clf(X.to_model_input([x_clf], self.cfg["norm"], self.device)).double(), 1)[0].cpu().numpy()
        t = torch.from_numpy(x_ae).permute(2, 0, 1)[None].float().div(255).to(self.device)
        err = float((self.ae(t) - t).pow(2).mean())
        return x_clf, prob, err

    def decide(self, prob, err, rule=None):
        rule = rule or self.cfg["default_rule"]
        msp = float(prob.max())
        if rule == "or":
            r = self.cfg["or"]
            reject = msp < r["msp_min"] or err > r["err_max"]
            raison = []
            if msp < r["msp_min"]:
                raison.append(f"confiance du classifieur {msp:.2f} < {r['msp_min']:.2f}")
            if err > r["err_max"]:
                raison.append(f"image atypique (erreur AE {err:.4f} > {r['err_max']:.4f})")
        else:
            s = self.cfg["score"]
            z1 = ((1 - msp) - s["z_msp"][0]) / s["z_msp"][1]
            z2 = (err - s["z_err"][0]) / s["z_err"][1]
            val = s["alpha"] * z1 + (1 - s["alpha"]) * z2
            reject = val > s["tau"]
            raison = [f"score combiné {val:.2f} > {s['tau']:.2f}"] if reject else []
        return reject, raison

    def predict(self, img, rule=None):
        x_clf, prob, err = self.scores(img)
        reject, raison = self.decide(prob, err, rule)
        classes = self.cfg["classes"]
        cam = X.gradcam_maps(self.clf, self.clf.cam_layer, X.to_model_input([x_clf], self.cfg["norm"], self.device))[0]
        overlay = X.overlay(x_clf, cam)
        if reject:
            texte = f"### INCONNU\n{CONSEIL_INCONNU}\n\nRaison : {', '.join(raison)}."
        else:
            texte = (f"### {short(classes[int(prob.argmax())])}\nConfiance {prob.max():.1%} "
                     f"(erreur de reconstruction {err:.4f}). Diagnostic indicatif, à confirmer par un agent agricole.")
        return texte, {short(c): float(p) for c, p in zip(classes, prob)}, overlay


def exemples_demo():
    """Images du dossier demo/ (connue, inconnue, hors-distribution), proposées sous l'interface."""
    dossier = Path(__file__).resolve().parent / "demo"
    exemples = []
    for f in sorted(dossier.glob("*.jpg")):
        exemples.append([str(f), "OU (MSP ou AE)"])
    return exemples or None


def build(device="cpu"):
    import gradio as gr
    d = Diagnostiqueur(device)

    def run(img, rule):
        if img is None:
            return "Ajouter une photo.", {}, None
        return d.predict(img, "or" if rule.startswith("OU") else "score")

    return gr.Interface(
        fn=run,
        inputs=[gr.Image(type="pil", label="Photo de feuille de tomate"),
                gr.Radio(["OU (MSP ou AE)", "Score combiné"], value="OU (MSP ou AE)", label="Règle de rejet")],
        outputs=[gr.Markdown(label="Diagnostic"), gr.Label(num_top_classes=3, label="Probabilités (9 classes connues)"),
                 gr.Image(label="Grad-CAM : zones utilisées par le modèle")],
        title="DiagnoCulture : diagnostic des maladies de la tomate",
        description="Prototype : classifieur MobileNetV3 + auto-encodeur. Répond « inconnu » plutôt que d'inventer un diagnostic.",
        flagging_mode="never",
        examples=exemples_demo(),
    )


if __name__ == "__main__":
    build().launch(share="--share" in sys.argv)
