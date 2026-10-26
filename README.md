# DiagnoCulture

Système hybride de diagnostic des maladies de la tomate, fiable et conscient de ses limites.
Projet de fin de module Apprentissage profond (1ADA2301), M2 MISID, École Polytechnique de Diamniadio.

Un classifieur supervisé (MobileNetV3) reconnaît les maladies connues ; un auto-encodeur non supervisé mesure si l'image ressemble à ce que le système a appris. Combinés, ils répondent « inconnu, consulter un agent agricole » face à une maladie jamais vue ou à une image qui n'est pas une feuille de tomate.

## Arborescence

```
diagnoculture/
├── README.md
├── requirements.txt
├── app.py                      # démo Gradio (partie bonus)
├── data/                       # non versionné sauf splits.csv et ood.csv
│   ├── splits.csv              # path, class, leaf_group, split : unique source de vérité
│   ├── ood.csv                 # jeu hors-distribution (partie 4)
│   └── ood/photos/<catégorie>/ # photos personnelles : tomate_fond_naturel, autres_plantes, mains, sol_objets
├── notebooks/
│   ├── 00_eda_split.ipynb
│   ├── 01_cnn_from_scratch.ipynb
│   ├── 02_transfer_learning.ipynb
│   ├── 03_autoencoders.ipynb
│   ├── 04_hybrid_open_set.ipynb
│   ├── 05_generative_extension.ipynb
│   ├── 06_critical_analysis.ipynb
│   └── 07_bonus_onnx_gradio.ipynb
├── src/
│   ├── config.py      # chemins, graine, choix figés (M, backbones, résolutions)
│   ├── data.py        # téléchargement, Dataset lisant splits.csv, transformations
│   ├── dedup.py       # pHash sur 8 variantes + union-find (anti-fuite)
│   ├── models.py      # CNN-Plain, CNN-Res, TLModel (3 stratégies), ConvAE, VAE
│   ├── train.py       # boucles avec journal CSV, checkpoints et reprise
│   ├── metrics.py     # métriques, latence, summary.csv
│   ├── openset.py     # seuils, règle OU, score combiné, couverture / précision
│   ├── explain.py     # Grad-CAM, segmentation de la feuille, biais de fond
│   └── viz.py
├── results/           # CSV et figures versionnés ; ckpt/ et onnx/ non versionnés (Drive)
└── report/
```

## Installation

**PC local avec GPU NVIDIA (ce que nous utilisons).**

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Vérifier que `python -c "import torch; print(torch.cuda.is_available())"` affiche `True`.

**Google Colab.** Choisir un GPU T4. Copier le dossier du projet dans `Mon Drive/diagnoculture`, puis dans la première cellule de chaque notebook :

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/diagnoculture/notebooks
%pip install -q imagehash torchinfo grad-cam onnx onnxruntime gradio kagglehub
```

**Données.** PlantVillage, version originale « color » (Mohanty et al.), téléchargée depuis Kaggle (`abdallahalidev/plantvillage-dataset`) par le notebook 00. Si le téléchargement demande une connexion, télécharger le zip à la main puis, avant la première cellule : `import os; os.environ["DIAGNO_SOURCE"] = r"<dossier décompressé>"`.

Important : `data/splits.csv` est créé une seule fois par le notebook 00 et partagé entre nous. Il ne faut pas relancer le 00 de son côté une fois les entraînements commencés.

## Ordre d'exécution et temps estimés (T4)

| Notebook | Contenu | Produit | Temps |
|---|---|---|---|
| 00 | EDA, doublons (pHash), split par groupe, normalisation, OOD | `data/splits.csv`, `results/norm_stats.json`, `data/ood.csv` | 10-15 min |
| 01 | CNN-Plain, CNN-Res, normes du gradient, étude du LR | runs `cnn_*`, `summary.csv` | 50-70 min |
| 02 | MobileNetV3 (3 stratégies), ResNet-50, latence, Pareto, classifieur sans M | `recommendation.json`, `latency.json`, `ckpt/clf_sans_M.pt` | 40-60 min |
| 03 | AE k ∈ {16, 64, 256}, débruiteur, seuil p95, analyse par maladie | `ckpt/ae_best.pt`, `thresholds_ae.json` | 10-20 min |
| 04 | AE connus, MSP / AE / hybride, couverture / précision, OOD | `ckpt/ae_connus.pt`, `scores.csv`, `hybrid_config.json` | 15-20 min |
| 05 | VAE : espace latent, interpolations, génération, score d'anomalie | runs `vae_*` | 10-15 min |
| 06 | Grad-CAM, test du biais de fond, analyse d'erreurs | figures, `background_test.json` | 10 min |
| 07 | Bonus : ONNX, quantification int8, démo Gradio | `results/onnx/` | 5 min |

03 et 05 ne dépendent que de 00 et peuvent tourner en parallèle de 01-02. 04 nécessite 02 et 03 ; 06 nécessite 02 ; 07 nécessite 04.

Chaque entraînement écrit un historique par époque (`results/runs/<run>/history.csv`) et un checkpoint par époque : si la session coupe, relancer la cellule reprend là où elle s'était arrêtée.

## Contrats de fichiers

| Fichier | Produit par | Utilisé par |
|---|---|---|
| `data/splits.csv` | 00 | tous |
| `results/norm_stats.json` | 00 | 01 |
| `results/ckpt/ae_best.pt`, `results/thresholds_ae.json` | 03 | 04, 05 |
| `results/ckpt/clf_sans_M.pt` | 02 | 04, 07, app |
| `results/ckpt/ae_connus.pt` | 04 | 07, app |
| `results/scores.csv` | 04 | rapport, 07 |
| `results/hybrid_config.json` | 04 | 07, app |
| `results/summary.csv` | tous | rapport (tableau récapitulatif global) |

## Choix méthodologiques figés

- Split 70 / 15 / 15 stratifié **par groupe de feuilles** (`StratifiedGroupKFold`, seed 42) : une même feuille, tournée ou retournée, est entièrement dans un seul split ; preuve « 0 hash commun » dans le notebook 00.
- Le test n'est utilisé qu'une fois par modèle final. Hyperparamètres, stratégie, k, seuils et α sont choisis sur la validation (α sur un proxy d'inconnu : autres espèces PlantVillage, moitié `calib`).
- Maladie inconnue M : virus de la mosaïque (`src/config.py`, `UNKNOWN_CLASS`).
- Backbone léger : MobileNetV3-Large ; lourd : ResNet-50. Résolutions : 128 (de zéro), 224 (transfer learning), 64 (AE, VAE).
- Métrique principale : F1 macro (déséquilibre des classes) ; perte pondérée par l'inverse des fréquences.

## Reproductibilité

`C.set_seed(42)` (random, numpy, torch, cuDNN déterministe), graines des DataLoaders fixées, versions dans `requirements.txt`. Test de non-régression rapide sur un mini-jeu : variables d'environnement `DIAGNO_SMOKE=1` (1 époque par entraînement) et `DIAGNO_SOURCE=<mini-jeu>`.

## Démo

```bash
python app.py            # interface locale
python app.py --share    # lien public temporaire (Colab, soutenance)
```

Pour la soutenance : une maladie connue, une image de M, une photo hors-distribution.
