"""Petites fonctions d'affichage communes."""
import matplotlib.pyplot as plt
import numpy as np


def image_grid(images, titles=None, ncols=8, size=1.8, path=None, suptitle=None, title_fs=7, colors=None):
    """Affiche une grille d'images (uint8 H×W×3 ou float dans [0, 1])."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.1))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=title_fs, color="black" if colors is None else colors[i])
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def chw_to_img(x):
    """Tenseur C×H×W dans [0, 1] -> array H×W×3 pour imshow."""
    return x.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
