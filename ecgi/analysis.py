"""ecgi.analysis — latent-space collection, 2-D embedding, and study runner."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ecgi.config import DEVICE as _DEVICE
from ecgi.data import build_loader, num_classes
from ecgi.inference import load_models
from ecgi.viz import _plot_2x1


# ── latent collection ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_branch_latents(
    G,
    D,
    loader,
    *,
    dz: int,
    device: torch.device,
    max_per_class: int = 800,
    seed: int = 125,
):
    rng = np.random.default_rng(seed)
    D.eval(); G.eval()

    buckets = {
        "st":    {"real": [], "fake": []},
        "w":     {"real": [], "fake": []},
        "fused": {"real": [], "fake": []},
        "y":     {"real": [], "fake": []},
    }

    per_class_count = defaultdict(int)

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)

        take_mask = torch.ones_like(y, dtype=torch.bool)
        for i, yi in enumerate(y.tolist()):
            if per_class_count[yi] >= max_per_class:
                take_mask[i] = False
            else:
                per_class_count[yi] += 1

        if not take_mask.any():
            continue

        x = x[take_mask]; y = y[take_mask]
        B = x.size(0)

        real_lat = D.extract_latents(x, y, detach=True)

        z  = torch.randn(B, dz, device=device)
        xf = G(z, y)
        fake_lat = D.extract_latents(xf, y, detach=True)

        for branch, key in (("st", "flat_st"), ("w", "flat_w"), ("fused", "flat_fused")):
            if real_lat[key] is not None:
                buckets[branch]["real"].append(real_lat[key].numpy())
                buckets[branch]["fake"].append(fake_lat[key].numpy())

        buckets["y"]["real"].append(real_lat["label"].numpy())
        buckets["y"]["fake"].append(fake_lat["label"].numpy())

    out = {}
    for branch in ("st", "w", "fused"):
        R  = np.concatenate(buckets[branch]["real"], axis=0) if buckets[branch]["real"] else None
        F  = np.concatenate(buckets[branch]["fake"], axis=0) if buckets[branch]["fake"] else None
        yR = np.concatenate(buckets["y"]["real"],    axis=0) if buckets["y"]["real"]    else None
        yF = np.concatenate(buckets["y"]["fake"],    axis=0) if buckets["y"]["fake"]    else None
        out[branch] = {"X_real": R, "y_real": yR, "X_fake": F, "y_fake": yF}
    return out


# ── 2-D embedding helpers ──────────────────────────────────────────────────────

def _fit_embed_2d(X, method="pca", seed=125, whiten=False):
    if method == "pca":
        scaler = StandardScaler(with_std=True)
        Xs     = scaler.fit_transform(X)
        pca    = PCA(n_components=2, whiten=whiten, random_state=seed)
        Z      = pca.fit_transform(Xs)
        return Z, {"evr_2D": pca.explained_variance_ratio_.sum()}
    elif method == "tsne":
        scaler = StandardScaler(with_std=True)
        Xs     = scaler.fit_transform(X)
        d50    = min(50, Xs.shape[1])
        Xp     = PCA(n_components=d50, random_state=seed).fit_transform(Xs)
        perp   = max(5, min(30, (Xs.shape[0] - 1) // 3))
        tsne   = TSNE(n_components=2, perplexity=perp, n_iter=1500,
                      learning_rate="auto", init="pca", random_state=seed)
        Z = tsne.fit_transform(Xp)
        return Z, {"perplexity": perp}
    else:
        raise ValueError("method = 'pca' or 'tsne'")


def _stack_rf(X_real, y_real, X_fake, y_fake):
    X   = np.concatenate([X_real, X_fake], axis=0)
    y   = np.concatenate([y_real, y_fake], axis=0)
    dom = np.array([0] * len(y_real) + [1] * len(y_fake))  # 0=real, 1=fake
    return X, y, dom


# ── latent-space study runner ─────────────────────────────────────────────────

def latent_space_study(
    *,
    cfg_path: Path,
    data_path: Path,
    fs: int,
    out_dir: Path,
    batch_size: int = 16,
    dz: int | None = None,
    device: str = None,
):
    DEV = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader = build_loader(data_path, batch_size=batch_size, device=str(DEV))
    base_ds = val_loader.dataset.dataset if hasattr(val_loader.dataset, "dataset") else val_loader.dataset
    n_cls   = num_classes(base_ds)
    C, T    = base_ds.x.shape[1:]

    cfg = torch.load(cfg_path, map_location="cpu")
    if dz is None:
        dz = int(cfg["dz"])
    G, D = load_models(cfg_path, C, T, n_cls, DEV, fs)

    pack = collect_branch_latents(G, D, val_loader, dz=dz, device=DEV,
                                   max_per_class=800, seed=125)

    for branch in ("st", "w", "fused"):
        Xr, yr = pack[branch]["X_real"], pack[branch]["y_real"]
        Xf, yf = pack[branch]["X_fake"], pack[branch]["y_fake"]
        if Xr is None or Xf is None:
            print(f"✗ Branch {branch}: insufficient data.")
            continue

        X, y, dom = _stack_rf(Xr, yr, Xf, yf)

        Zp, info_p = _fit_embed_2d(X, method="pca",  seed=125, whiten=True)
        Zt, info_t = _fit_embed_2d(X, method="tsne", seed=125)

        title      = f"Latentes {branch.upper()}  (C={C}, k≈{D.k}, N={len(X)})"
        out_branch = out_dir / f"{branch}_pca_tsne.png"
        _plot_2x1(Zp, Zt, y, dom, title, out_branch)

        np.savez_compressed(
            out_dir / f"{branch}_embeddings.npz",
            Z_pca=Zp, Z_tsne=Zt, y=y, dom=dom,
            pca_info=np.array(list(info_p.items()), dtype=object),
            tsne_info=np.array(list(info_t.items()), dtype=object),
        )
        print(f"✓ {branch}: {out_branch}")
