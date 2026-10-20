"""Détection des doublons (même feuille sous plusieurs formes) par hash perceptuel.

Principe : pour chaque image on calcule le pHash (64 bits) de ses 8 variantes
(4 rotations × avec/sans miroir). Deux images A et B sont considérées comme la même
feuille si min_v hamming(hash(A), hash_v(B)) <= seuil. Les paires sont ensuite
regroupées par union-find : chaque composante connexe devient un leaf_group.
"""
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image

N_VARIANTS = 8


def _variants(img: Image.Image):
    out = []
    for flip in (False, True):
        base = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True))
    return out   # variante 0 = image originale


def _hash_bits(h: imagehash.ImageHash) -> np.uint64:
    bits = h.hash.flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def image_hashes(path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((64, 64), Image.BILINEAR)
    return np.array([_hash_bits(imagehash.phash(v)) for v in _variants(img)], dtype=np.uint64)


def compute_hashes(paths, workers: int = 8) -> np.ndarray:
    """Renvoie un tableau N×8 de hashes (uint64)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.stack(list(ex.map(image_hashes, paths)))


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def min_distances(h_a: np.ndarray, h_b: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Distance minimale (sur les 8 variantes de B) entre chaque original de A et chaque image de B.
    h_a : N_a×8, h_b : N_b×8 -> matrice N_a×N_b (uint8), calculée par blocs."""
    a0 = h_a[:, 0]
    out = np.empty((len(a0), len(h_b)), dtype=np.uint8)
    flat_b = h_b.reshape(-1)
    for s in range(0, len(a0), chunk):
        x = np.bitwise_xor(a0[s:s + chunk, None], flat_b[None, :])
        d = popcount64(x).reshape(len(x), len(h_b), N_VARIANTS).min(-1)
        out[s:s + chunk] = d
    return out


def find_pairs(hashes: np.ndarray, threshold: int = 4, chunk: int = 64):
    """Toutes les paires (i, j, distance) avec i < j et distance <= seuil."""
    n = len(hashes)
    flat = hashes.reshape(-1)
    pairs = []
    for s in range(0, n, chunk):
        a0 = hashes[s:s + chunk, 0]
        d = popcount64(np.bitwise_xor(a0[:, None], flat[None, :])).reshape(len(a0), n, N_VARIANTS).min(-1)
        ii, jj = np.nonzero(d <= threshold)
        ii = ii + s
        keep = ii != jj
        for i, j, dist in zip(ii[keep], jj[keep], d[ii[keep] - s, jj[keep]]):
            pairs.append((min(i, j), max(i, j), int(dist)))
    pairs = sorted(set(pairs))
    return pairs


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def groups_from_pairs(n: int, pairs) -> np.ndarray:
    uf = UnionFind(n)
    for i, j, _ in pairs:
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid
