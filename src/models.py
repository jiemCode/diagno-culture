"""Modèles : auto-encodeur convolutif (partie 3). Le VAE sera ajouté jeudi."""
from torch import nn


def build_ae(k, img_size=64):
    """Auto-encodeur convolutif : image 3x64x64 dans [0, 1] -> code de dimension k -> image.

    Conv F=4, S=2, P=1 : W_out = (W - 4 + 2)/2 + 1 = W/2 (64 -> 32 -> 16 -> 8).
    ConvTranspose F=4, S=2, P=1 : W_out = 2 W. BatchNorm après chaque convolution.
    Sigmoid en sortie (pixels dans [0, 1]).
    """
    s = img_size // 8
    encodeur = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),      # 64 -> 32
        nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),     # 32 -> 16
        nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),   # 16 -> 8
        nn.Flatten(), nn.Linear(128 * s * s, k),                       # goulot : k << 3x64x64
    )
    decodeur = nn.Sequential(
        nn.Linear(k, 128 * s * s), nn.Unflatten(1, (128, s, s)),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 8 -> 16
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),    # 16 -> 32
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),                      # 32 -> 64
    )
    return nn.Sequential(encodeur, decodeur)
