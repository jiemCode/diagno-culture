"""Données : téléchargement, Dataset lisant splits.csv, transformations, DataLoaders."""
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from . import config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
KNOWN_CLASSES = [c for c in CLASSES if c != C.UNKNOWN_CLASS]


def short(name: str) -> str:
    """Nom court lisible pour les figures."""
    s = name.replace("Tomato___", "").replace("Spider_mites Two-spotted_spider_mite", "Spider_mites")
    return s.replace("Tomato_Yellow_Leaf_Curl_Virus", "Yellow_Leaf_Curl").replace("Tomato_mosaic_virus", "Mosaic_virus")


# --------------------------------------------------------------------------
# Téléchargement
# --------------------------------------------------------------------------
def list_images(folder: Path):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def find_color_dir(base: Path):
    """Cherche le dossier 'color' de PlantVillage dans une arborescence téléchargée."""
    base = Path(base)
    for cand in [base, *base.rglob("color")]:
        if cand.is_dir() and (cand / "Tomato___healthy").is_dir():
            return cand
    return None


def ensure_raw_data(force_download: bool = False) -> Path:
    """Garantit que RAW_DIR contient les 10 dossiers Tomato de la version color.

    1. déjà présents -> rien à faire ;
    2. sinon variable DIAGNO_SOURCE (copie fournie par l'enseignant) ;
    3. sinon téléchargement Kaggle via kagglehub.
    Renvoie le dossier 'color' complet (utile pour les autres espèces, jeu OOD).
    """
    import os
    marker = C.RAW_DIR / "_color_dir.txt"
    if not force_download and all((C.RAW_DIR / c).is_dir() for c in CLASSES) and marker.exists():
        return Path(marker.read_text().strip())
    src = os.environ.get("DIAGNO_SOURCE")
    if src:
        color = find_color_dir(Path(src))
    else:
        import kagglehub
        path = kagglehub.dataset_download(C.KAGGLE_DATASET)
        color = find_color_dir(Path(path))
    if color is None:
        raise FileNotFoundError("Dossier 'color' de PlantVillage introuvable.")
    for c in CLASSES:
        dst = C.RAW_DIR / c
        if not dst.is_dir():
            shutil.copytree(color / c, dst)
    marker.write_text(str(color))
    return color


# --------------------------------------------------------------------------
# Lecture d'images et cache mémoire
# --------------------------------------------------------------------------
def read_rgb(path, size=None) -> np.ndarray:
    """Lit une image en RGB ; si elle n'est pas carrée (photos de téléphone), recadrage central
    carré avant redimensionnement pour ne pas déformer la feuille."""
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if w != h:
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_cache(paths, size: int) -> np.ndarray:
    """Charge (une fois) toutes les images redimensionnées dans un tableau uint8 N×H×W×3,
    sauvegardé sur disque et relu en mémoire partagée (mmap) aux exécutions suivantes."""
    key = hashlib.md5(("|".join(map(str, paths)) + f"@{size}").encode()).hexdigest()[:16]
    f = C.CACHE_DIR / f"cache_{size}_{key}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    arr = np.lib.format.open_memmap(f.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(paths), size, size, 3))
    def _load(i):
        arr[i] = read_rgb(paths[i], size)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_load, range(len(paths))))
    arr.flush()
    del arr
    f.with_suffix(".tmp.npy").rename(f)
    return np.load(f, mmap_mode="r")


class LeafDataset(Dataset):
    """Dataset générique : une ligne du DataFrame = une image.

    df       : DataFrame avec au moins la colonne 'path' (relative, cf. config.resolve)
    classes  : liste ordonnée des classes -> indices ; None = étiquette -1
    size     : côté de l'image après redimensionnement
    preload  : met les images en cache mémoire (recommandé pour 64 et 128 px)
    """

    def __init__(self, df: pd.DataFrame, size: int, transform=None, classes=None,
                 preload: bool = True, label_col: str = "class"):
        self.df = df.reset_index(drop=True)
        self.paths = [C.resolve(p) for p in self.df["path"]]
        self.size = size
        self.transform = transform
        if classes is not None and label_col in self.df:
            idx = {c: i for i, c in enumerate(classes)}
            self.labels = np.array([idx.get(c, -1) for c in self.df[label_col]], dtype=np.int64)
        else:
            self.labels = np.full(len(self.df), -1, dtype=np.int64)
        self.cache = load_cache(self.paths, size) if preload and len(self.paths) else None

    def __len__(self):
        return len(self.paths)

    def get_uint8(self, i) -> np.ndarray:
        if self.cache is not None:
            return np.array(self.cache[i])
        return read_rgb(self.paths[i], self.size)

    def __getitem__(self, i):
        img = torch.from_numpy(self.get_uint8(i)).permute(2, 0, 1).contiguous()   # uint8 C×H×W
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
class RandomRot90(torch.nn.Module):
    """Rotation aléatoire d'un multiple de 90° : sans perte, sans coins noirs ajoutés."""

    def forward(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(img, k, dims=(-2, -1))


def get_transform(kind: str, train: bool, norm=None):
    """kind : 'clf' (normalisation par canal) ou 'ae' (image dans [0, 1], pas de normalisation).
    norm : dict {'mean': [...], 'std': [...]}. Augmentations légères seulement en train."""
    t = []
    if train:
        t += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RandomRot90()]
    t += [v2.ToDtype(torch.float32, scale=True)]
    if kind == "clf":
        t += [v2.Normalize(norm["mean"], norm["std"])]
    return v2.Compose(t)


def norm_stats(kind: str):
    """'scratch' -> statistiques du train (norm_stats.json) ; 'imagenet' -> celles du pré-entraînement."""
    if kind == "imagenet":
        return {"mean": C.IMAGENET_MEAN, "std": C.IMAGENET_STD}
    return C.load_json(C.NORM_JSON)


def make_loader(ds, batch_size=64, shuffle=False, sampler=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=C.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=C.seed_worker, generator=C.make_generator(),
                      persistent_workers=C.NUM_WORKERS > 0)


def load_splits() -> pd.DataFrame:
    return pd.read_csv(C.SPLITS_CSV)


def subset(df, split=None, classes=None, exclude=None):
    out = df
    if split is not None:
        out = out[out["split"].isin([split] if isinstance(split, str) else split)]
    if classes is not None:
        out = out[out["class"].isin(classes)]
    if exclude is not None:
        out = out[~out["class"].isin([exclude] if isinstance(exclude, str) else exclude)]
    return out.reset_index(drop=True)


def class_weights(labels, n_classes) -> torch.Tensor:
    """Poids inverses de la fréquence : w_c = N / (K · n_c)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def clf_loaders(df, classes, size, norm, batch_size=64, preload=True):
    """Trois DataLoaders train/val/test pour un classifieur."""
    out = {}
    for split in ("train", "val", "test"):
        d = subset(df, split=split, classes=classes)
        ds = LeafDataset(d, size, get_transform("clf", split == "train", norm), classes, preload=preload)
        out[split] = make_loader(ds, batch_size, shuffle=(split == "train"))
    return out


def denorm(x: torch.Tensor, norm) -> torch.Tensor:
    """Inverse de Normalize pour l'affichage."""
    m = torch.tensor(norm["mean"], device=x.device).view(-1, 1, 1)
    s = torch.tensor(norm["std"], device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)
