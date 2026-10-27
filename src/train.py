"""Boucles d'entraînement avec journal CSV par époque, checkpoints et reprise automatique.

Les sessions Colab gratuites s'interrompent : chaque époque écrit
results/runs/<run>/history.csv et results/ckpt/<run>/last.pt. Relancer la même
cellule reprend là où l'entraînement s'était arrêté (ou saute s'il est terminé).
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from . import config as C


def _amp_ctx(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda"))


def _paths(run_name):
    run_dir, ck_dir = C.RUNS_DIR / run_name, C.CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ck_dir


def _resume(ck_dir, model, optimizer, scheduler, scaler, device, resume):
    last = ck_dir / "last.pt"
    if not (resume and last.exists()):
        return 0, -np.inf, 0.0, []
    ck = torch.load(last, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    print(f"Reprise depuis l'époque {ck['epoch'] + 1}")
    return ck["epoch"] + 1, ck["best"], ck["train_time"], ck["history"]


def fit(model, loaders, epochs, optimizer, criterion, run_name, meta, scheduler=None,
        grad_every=None, device=C.DEVICE, resume=True):
    """Entraîne un classifieur et sélectionne le meilleur checkpoint sur le F1 macro de VALIDATION.

    meta       : dict décrivant le modèle (arch, classes, hyperparamètres), stocké dans le checkpoint
    grad_every : si entier, enregistre la norme du gradient de chaque poids conv toutes les N itérations
    Renvoie l'historique (DataFrame)."""
    run_dir, ck_dir = _paths(run_name)
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    start, best, train_time, history = _resume(ck_dir, model, optimizer, scheduler, scaler, device, resume)
    grad_file = run_dir / "grad_norms.csv"
    if start == 0 and grad_file.exists():
        grad_file.unlink()
    conv_names = [n for n, p in model.named_parameters() if p.dim() == 4 and p.shape[-1] == 3]
    it = start * len(loaders["train"])

    for epoch in range(start, epochs):
        model.train()
        t0 = time.time()
        tot, correct, loss_sum, grad_rows = 0, 0, 0.0, []
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx(device):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if grad_every and it % grad_every == 0:
                scaler.unscale_(optimizer)   # gradients réels, pas mis à l'échelle par l'AMP
                params = dict(model.named_parameters())
                for i, n in enumerate(conv_names):
                    g = params[n].grad
                    if g is not None:
                        grad_rows.append((it, epoch, i + 1, n, g.norm().item()))
            scaler.step(optimizer)
            scaler.update()
            it += 1
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        if scheduler is not None:
            scheduler.step()
        if grad_rows:
            pd.DataFrame(grad_rows, columns=["iter", "epoch", "layer_idx", "layer", "grad_norm"]).to_csv(
                grad_file, mode="a", header=not grad_file.exists(), index=False)

        val = evaluate_loss(model, loaders["val"], criterion, device)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / tot, train_acc=correct / tot,
                   val_loss=val["loss"], val_acc=val["acc"], val_f1=val["f1"],
                   lr=optimizer.param_groups[0]["lr"], epoch_time_s=ep_time, train_time_s=train_time)
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val["f1"] > best:
            best = val["f1"]
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch + 1, "val_f1": best},
                       ck_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best,
                    "train_time": train_time, "history": history, "meta": meta}, ck_dir / "last.pt")
        print(f"[{run_name}] ep {epoch + 1}/{epochs} | loss {row['train_loss']:.4f}/{row['val_loss']:.4f} "
              f"| acc {row['train_acc']:.4f}/{row['val_acc']:.4f} | F1 val {row['val_f1']:.4f} | {ep_time:.0f}s")
    return pd.DataFrame(history)


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device=C.DEVICE):
    model.eval()
    tot, loss_sum, ys, ps = 0, 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with _amp_ctx(device):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * len(y)
        tot += len(y)
        ys.append(y.cpu())
        ps.append(out.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"loss": loss_sum / tot, "acc": float((y == p).mean()), "f1": f1_score(y, p, average="macro")}


def load_best(model, run_name, device=C.DEVICE):
    ck = torch.load(C.CKPT_DIR / run_name / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def export_checkpoint(run_name, dest_name):
    """Copie le meilleur checkpoint d'un run sous un nom de contrat (ex. clf_sans_M.pt)."""
    ck = torch.load(C.CKPT_DIR / run_name / "best.pt", map_location="cpu", weights_only=False)
    torch.save(ck, C.CKPT_DIR / dest_name)
    return C.CKPT_DIR / dest_name


# ==========================================================================
# Auto-encodeurs et VAE
# ==========================================================================
def gaussian_noise(sigma=0.1):
    def f(x):
        return (x + sigma * torch.randn_like(x)).clamp(0, 1)
    f.sigma = sigma
    return f


def fit_recon(model, loaders, epochs, optimizer, run_name, meta, noise_fn=None, vae=False,
              beta_max=1.0, warmup_epochs=10, device=C.DEVICE, resume=True):
    """Entraîne un AE (perte MSE) ou un VAE (reconstruction + beta·KL avec montée progressive de beta).

    noise_fn : si fourni (auto-encodeur débruiteur), l'entrée est bruitée et la cible reste propre :
               L = ||x - g(f(x_bruité))||².
    Le meilleur checkpoint est choisi sur la perte de validation."""
    from .models import vae_loss
    run_dir, ck_dir = _paths(run_name)
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start, best, train_time, history = _resume(ck_dir, model, optimizer, None, scaler, device, resume)
    best = -best if np.isfinite(best) else np.inf   # on minimise

    for epoch in range(start, epochs):
        beta = beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs)) if vae else 0.0
        model.train()
        t0, sums, n = time.time(), np.zeros(3), 0
        for x, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            inp = noise_fn(x) if noise_fn else x
            optimizer.zero_grad(set_to_none=True)
            if vae:
                x_hat, mu, logvar = model(inp)
                loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta)
                vals = (loss.item(), rec.item(), kl.item())
            else:
                loss = F.mse_loss(model(inp), x)
                vals = (loss.item(), loss.item(), 0.0)
            loss.backward()
            optimizer.step()
            sums += np.array(vals) * len(x)
            n += len(x)
        ep_time = time.time() - t0
        train_time += ep_time
        v = eval_recon(model, loaders["val"], noise_fn, vae, beta_max if vae else 0.0, device)
        row = dict(epoch=epoch + 1, train_loss=sums[0] / n, train_rec=sums[1] / n, train_kl=sums[2] / n,
                   val_loss=v["loss"], val_mse=v["mse"], val_kl=v["kl"], beta=beta,
                   epoch_time_s=ep_time, train_time_s=train_time)
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        crit = v["loss"]
        if crit < best:
            best = crit
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch + 1, "val_loss": crit},
                       ck_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None,
                    "scaler": None, "epoch": epoch, "best": -best, "train_time": train_time,
                    "history": history, "meta": meta}, ck_dir / "last.pt")
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"[{run_name}] ep {epoch + 1}/{epochs} | train {row['train_loss']:.5f} | "
                  f"val {v['loss']:.5f} (MSE {v['mse']:.5f}) | {ep_time:.0f}s")
    return pd.DataFrame(history)


@torch.inference_mode()
def eval_recon(model, loader, noise_fn=None, vae=False, beta=1.0, device=C.DEVICE):
    from .models import vae_loss
    model.eval()
    s, n = np.zeros(3), 0
    g = torch.Generator(device=device).manual_seed(C.SEED)   # bruit de validation identique d'une époque à l'autre
    for x, _ in loader:
        x = x.to(device)
        inp = (x + noise_fn.sigma * torch.randn(x.shape, generator=g, device=device)).clamp(0, 1) if noise_fn else x
        if vae:
            x_hat, mu, logvar = model(inp)
            loss, _, kl = vae_loss(x_hat, x, mu, logvar, beta)
            mse = F.mse_loss(x_hat, x)
            vals = (loss.item(), mse.item(), kl.item())
        else:
            mse = F.mse_loss(model(inp), x)
            vals = (mse.item(), mse.item(), 0.0)
        s += np.array(vals) * len(x)
        n += len(x)
    return {"loss": s[0] / n, "mse": s[1] / n, "kl": s[2] / n}


@torch.inference_mode()
def recon_errors(model, loader, device=C.DEVICE, vae=False, return_maps=False):
    """Score d'anomalie par image : moyenne des pixels de ||x - x_hat||² (erreur de reconstruction).
    Pour un VAE, renvoie aussi -ELBO (reconstruction + KL) par image."""
    model.eval()
    errs, elbos, maps = [], [], []
    for x, _ in loader:
        x = x.to(device)
        if vae:
            mu, logvar = model.encode(x)
            x_hat = model.decoder(mu)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            rec_sum = (x_hat - x).pow(2).flatten(1).sum(1)
            elbos.append((rec_sum + kl).cpu())
        else:
            x_hat = model(x)
        e = (x_hat - x).pow(2)
        errs.append(e.flatten(1).mean(1).cpu())
        if return_maps:
            maps.append(e.mean(1).cpu())
    out = {"err": torch.cat(errs).numpy()}
    if vae:
        out["neg_elbo"] = torch.cat(elbos).numpy()
    if return_maps:
        out["maps"] = torch.cat(maps).numpy()
    return out
