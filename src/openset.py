"""Partie 4 : rejet de l'inconnu. Convention : pour tout système, un score ÉLEVÉ = rejeter.

Tous les seuils et paramètres (tau, alpha) se fixent sur la VALIDATION des classes connues
(et éventuellement sur un proxy d'inconnu qui ne vient pas du test). Jamais sur M.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def tau_for_acceptance(score_known_val, accept=0.95):
    """Seuil tel qu'une fraction `accept` des connus de validation soit acceptée (score <= tau)."""
    return float(np.quantile(score_known_val, accept))


class ECDF:
    """Rang percentile d'un score par rapport à une référence (connus de validation)."""

    def __init__(self, ref):
        self.ref = np.sort(np.asarray(ref))

    def __call__(self, s):
        return np.searchsorted(self.ref, np.asarray(s), side="right") / len(self.ref)


class ZScore:
    def __init__(self, ref):
        self.m, self.s = float(np.mean(ref)), float(np.std(ref) + 1e-12)

    def __call__(self, x):
        return (np.asarray(x) - self.m) / self.s


def or_rule_score(rank_msp, rank_ae):
    """Règle OU continue : rejet si MSP trop basse OU erreur trop haute.
    Avec des scores convertis en rangs percentiles de la validation connue,
    'rang_1 > q ou rang_2 > q' équivaut à 'max(rang_1, rang_2) > q'."""
    return np.maximum(rank_msp, rank_ae)


def combined_score(z_msp, z_ae, alpha):
    """Score combiné normalisé : s = alpha·z(1 - MSP) + (1 - alpha)·z(erreur)."""
    return alpha * np.asarray(z_msp) + (1 - alpha) * np.asarray(z_ae)


def choose_alpha(z_msp_known, z_ae_known, z_msp_proxy, z_ae_proxy, grid=np.linspace(0, 1, 21)):
    """Choisit alpha qui maximise l'AUROC 'connu (val) vs proxy d'inconnu' (autres espèces, jeu de calibration)."""
    y = np.r_[np.zeros(len(z_msp_known)), np.ones(len(z_msp_proxy))]
    res = []
    for a in grid:
        s = np.r_[combined_score(z_msp_known, z_ae_known, a), combined_score(z_msp_proxy, z_ae_proxy, a)]
        res.append((a, roc_auc_score(y, s)))
    best = max(res, key=lambda t: t[1])
    return best[0], res


def system_metrics(score_known, score_unknown, tau, correct_known):
    """Métriques demandées pour un système de rejet évalué sur le test.

    score_known / score_unknown : scores de rejet des images connues / de M
    correct_known : booléen, le classifieur a-t-il la bonne classe sur chaque image connue"""
    score_known, score_unknown = np.asarray(score_known), np.asarray(score_unknown)
    correct_known = np.asarray(correct_known, bool)
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    acc_known = score_known <= tau
    acc_unknown = score_unknown <= tau
    n_acc = acc_known.sum()
    return {
        "auroc_known_vs_M": roc_auc_score(y, np.r_[score_known, score_unknown]),
        "rejet_M": float(1 - acc_unknown.mean()),
        "rejet_a_tort_connus": float(1 - acc_known.mean()),
        "acc_connus_acceptes": float(correct_known[acc_known].mean()) if n_acc else np.nan,
        "couverture": float((acc_known.sum() + acc_unknown.sum()) / (len(score_known) + len(score_unknown))),
        "acc_systeme_acceptes": float(correct_known[acc_known].sum() / max(1, n_acc + acc_unknown.sum())),
    }


def coverage_accuracy_curve(scores, correct):
    """Courbe couverture / précision : on accepte les images par score de rejet croissant.
    Une image de M acceptée compte comme une erreur (le système invente un diagnostic)."""
    order = np.argsort(scores, kind="stable")
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k
