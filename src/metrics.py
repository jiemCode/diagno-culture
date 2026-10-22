"""Métriques, figures standard, mesure de latence et tableau récapitulatif."""
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score)

from . import config as C


@torch.inference_mode()
def predict(model, loader, device=C.DEVICE):
    """Renvoie y_true, y_pred et les probabilités softmax (N×K)."""
    model.eval().to(device)
    ys, probs = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(x.to(device))
        probs.append(torch.softmax(out.double(), 1).cpu())   # float64 : MSP proches de 1 mieux départagées
        ys.append(y)
    p = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), p.argmax(1), p


def clf_report(y_true, y_pred, class_names):
    """Accuracy, F1 macro, rappel par classe, matrice de confusion."""
    labels = list(range(len(class_names)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "recall": dict(zip(class_names, rec)),
            "cm": confusion_matrix(y_true, y_pred, labels=labels)}


def plot_confusion(cm, names, title, path=None, ax=None):
    """Matrice de confusion normalisée par ligne (= rappel), effectifs bruts en annotation."""
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Proportion de la classe réelle")
    if own:
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_histories(histories: dict, path=None, title=""):
    """Courbes perte et accuracy, train (pointillé) vs validation (plein), une couleur par expérience."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, h) in enumerate(histories.items()):
        col = f"C{i}"
        axes[0].plot(h["epoch"], h["train_loss"], "--", color=col, label=f"{name} train")
        axes[0].plot(h["epoch"], h["val_loss"], "-", color=col, label=f"{name} val")
        axes[1].plot(h["epoch"], h["train_acc"], "--", color=col)
        axes[1].plot(h["epoch"], h["val_acc"], "-", color=col)
    axes[0].set(xlabel="Époque", ylabel="Perte (entropie croisée)", title="Perte")
    axes[1].set(xlabel="Époque", ylabel="Accuracy", title="Accuracy")
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr


def model_size_mb(model) -> float:
    """Taille du state_dict sérialisé (Mo), ce qui serait embarqué sur le téléphone."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.inference_mode()
def measure_latency(model, input_size, device="cpu", n_warmup=20, n_runs=100, threads=1, images=None):
    """Latence d'inférence d'UNE image (batch 1) en millisecondes : moyenne et écart-type.

    - model.eval() + inference_mode : pas de graphe de gradient, BN/Dropout en mode inférence
    - images : liste de tenseurs C×H×W (≥ 100 images du test) ; sinon une image aléatoire
    - n_warmup itérations non comptées (allocations, choix des noyaux)
    - GPU : torch.cuda.synchronize() avant/après chaque mesure, sinon on mesure un lancement asynchrone
    - CPU : nombre de threads fixé (1 par défaut, proche d'un smartphone d'entrée de gamme)"""
    device = torch.device(device)
    prev = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(threads)
    model = model.eval().to(device)
    if images is None:
        images = [torch.randn(3, input_size, input_size)]
    for k in range(n_warmup):
        model(images[k % len(images)][None].to(device))
    times = []
    for k in range(n_runs):
        x = images[k % len(images)][None].to(device)   # une vraie image du test, batch de 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev)
    return float(np.mean(times)), float(np.std(times))


def upsert_summary(row: dict):
    """Ajoute ou remplace (clé 'experiment') une ligne de results/summary.csv."""
    df = pd.read_csv(C.SUMMARY_CSV) if C.SUMMARY_CSV.exists() else pd.DataFrame()
    if len(df) and "experiment" in df:
        df = df[df["experiment"] != row["experiment"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(C.SUMMARY_CSV, index=False)
    return df


def detection_metrics(scores_normal, scores_anomaly, tau):
    """Détection d'anomalies, score élevé = anormal.
    AUROC, rappel des anomalies (score > tau), taux de fausses alarmes (normaux avec score > tau)."""
    y = np.r_[np.zeros(len(scores_normal)), np.ones(len(scores_anomaly))]
    s = np.r_[scores_normal, scores_anomaly]
    return {"auroc": roc_auc_score(y, s),
            "recall_anomaly": float((np.asarray(scores_anomaly) > tau).mean()),
            "false_alarm": float((np.asarray(scores_normal) > tau).mean())}


def savefig(name):
    p = C.FIG_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    return p


def update_summary(experiment: str, **fields):
    """Complète une ligne existante de summary.csv (ex. ajout de la latence dans le notebook 02)."""
    df = pd.read_csv(C.SUMMARY_CSV) if C.SUMMARY_CSV.exists() else pd.DataFrame(columns=["experiment"])
    if experiment not in set(df["experiment"]):
        df = pd.concat([df, pd.DataFrame([{"experiment": experiment}])], ignore_index=True)
    for k, v in fields.items():
        if k not in df:
            df[k] = np.nan
        df[k] = df[k].astype(object)
        df.loc[df["experiment"] == experiment, k] = v
    df.to_csv(C.SUMMARY_CSV, index=False)
    return df
