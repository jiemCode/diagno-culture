"""Configuration centrale de DiagnoCulture : chemins, graine, choix figés du projet.

Tous les notebooks importent ce module. Rien n'est gardé en mémoire d'un notebook
à l'autre : tout passe par des fichiers dans data/ et results/.
"""
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Environnement d'exécution
# --------------------------------------------------------------------------
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_WINDOWS = platform.system() == "Windows"
SMOKE = os.environ.get("DIAGNO_SMOKE", "0") == "1"   # test rapide sur mini-jeu


def _find_root() -> Path:
    """Racine du dépôt : variable DIAGNO_ROOT, sinon premier parent contenant src/config.py."""
    env = os.environ.get("DIAGNO_ROOT")
    if env:
        return Path(env)
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "src" / "config.py").exists():
            return cand
    return Path(__file__).resolve().parents[1]


ROOT = _find_root()

# data/ : petits fichiers (splits.csv, photos OOD) versionnés ou sur le Drive.
DATA_DIR = ROOT / "data"
OOD_DIR = DATA_DIR / "ood"
SPLITS_CSV = DATA_DIR / "splits.csv"
OOD_CSV = DATA_DIR / "ood.csv"

# Images brutes : sur Colab/Kaggle on les garde sur le disque local de la VM
# (le Drive est trop lent pour 18 000 petits fichiers).
if "DIAGNO_RAW" in os.environ:
    RAW_DIR = Path(os.environ["DIAGNO_RAW"])
elif IN_COLAB:
    RAW_DIR = Path("/content/data_raw")
elif IN_KAGGLE:
    RAW_DIR = Path("/kaggle/working/data_raw")
else:
    RAW_DIR = DATA_DIR / "raw"

CACHE_DIR = Path("/content/cache") if IN_COLAB else DATA_DIR / "cache"

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
RUNS_DIR = RESULTS / "runs"
CKPT_DIR = RESULTS / "ckpt"
SUMMARY_CSV = RESULTS / "summary.csv"
NORM_JSON = RESULTS / "norm_stats.json"
SCORES_CSV = RESULTS / "scores.csv"

for _d in (DATA_DIR, OOD_DIR, RAW_DIR, CACHE_DIR, RESULTS, FIG_DIR, RUNS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Choix figés du projet (décision d'équipe du lundi 19 octobre)
# --------------------------------------------------------------------------
SEED = 42
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"   # PlantVillage original (Mohanty et al.), dossier color/
UNKNOWN_CLASS = "Tomato___Tomato_mosaic_virus"            # maladie M de la partie 4
HEALTHY_CLASS = "Tomato___healthy"
LIGHT_BACKBONE = "mobilenet_v3_large"
HEAVY_BACKBONE = "resnet50"

SCRATCH_SIZE = 128   # CNN entraînés de zéro
TL_SIZE = int(os.environ.get("DIAGNO_TL_SIZE", 224))   # transfer learning (passer à 160 si le T4 sature)
AE_SIZE = 64         # auto-encodeurs et VAE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if ON_WINDOWS else min(4, os.cpu_count() or 1)
PRETRAINED = not SMOKE   # le test hors-ligne n'a pas accès aux poids ImageNet


def ep(n: int) -> int:
    """Nombre d'époques : n en vrai, 1 en mode SMOKE."""
    return 1 if SMOKE else n


def set_seed(seed: int = SEED) -> None:
    """Fixe toutes les sources d'aléa (random, numpy, torch, cuDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Graine propre à chaque worker du DataLoader (reproductibilité)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve(rel_path: str) -> Path:
    """Chemin relatif stocké dans les CSV -> chemin réel.

    raw/... -> RAW_DIR, ood/... -> OOD_DIR. Les CSV restent ainsi portables
    entre Colab, Kaggle et un PC local.
    """
    rel_path = str(rel_path).replace("\\", "/")
    if rel_path.startswith("raw/"):
        return RAW_DIR / rel_path[4:]
    if rel_path.startswith("ood/"):
        return OOD_DIR / rel_path[4:]
    return DATA_DIR / rel_path


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def describe_env() -> str:
    """Machine utilisée (à recopier dans le rapport)."""
    import torchvision
    lines = [f"Python {sys.version.split()[0]}, torch {torch.__version__}, torchvision {torchvision.__version__}",
             f"Calcul sur : {DEVICE}"]
    if torch.cuda.is_available():
        lines.append(f"GPU : {torch.cuda.get_device_name(0)}")
    lines.append(f"CPU : {cpu_name()} ({os.cpu_count()} coeurs logiques)")
    if SMOKE:
        lines.append("MODE TEST RAPIDE (1 époque)")
    return "\n".join(lines)


def cpu_name() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Windows":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        return platform.processor() or platform.machine()
    except Exception:
        return platform.machine()
