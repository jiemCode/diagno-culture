from pathlib import Path

# Chemins
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = RESULTS / "ckpt"

# Création des dossiers
for p in (DATA, RESULTS, CKPT, RESULTS / "figures"):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = sorted([
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
])
HEALTHY = "Tomato___healthy"

SEED = 42

# Fixer le seed pour la reproducibilité
def set_seed(seed=SEED):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Taille des images
SIZE_AE = 64        # Autoencoder
SIZE_CNN = 128      # CNN
SIZE_TL = 224       # Transfer Learning
