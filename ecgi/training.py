"""ecgi.training — loss functions, weight initialisation, train loop, and experiment runner.

Opt-2: torch.compile for G and D after .to(DEVICE), before .apply(init_he).
Opt-4: GradScaler explicit device string "cuda".
Opt-5: cudnn.benchmark = True at module level.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as _cudnn
import matplotlib.pyplot as plt

from ecgi.config import DEVICE, DATASETS, VARIANTS, disc_kwargs, n_scales, f_min, f_max
from ecgi.data import build_loader, num_classes
from ecgi.fusion import FiLMFusion
from ecgi.models import Generator, Discriminator, RegressionHead
from ecgi.inference import resolve_mat_path, global_stats

# Opt-5: select fastest conv algorithm for fixed-size inputs
_cudnn.benchmark = True


# ── loss functions ─────────────────────────────────────────────────────────────

def pearson_corr(fake: torch.Tensor, real: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1 - PearsonCorrelation(fake, real) as a loss.
    fake, real: (B, C, T)
    """
    B, C, T = fake.shape
    f = fake.reshape(B, -1)
    r = real.reshape(B, -1)

    f_mean   = f.mean(dim=1, keepdim=True)
    r_mean   = r.mean(dim=1, keepdim=True)
    f_center = f - f_mean
    r_center = r - r_mean

    num  = (f_center * r_center).sum(dim=1)
    den  = torch.sqrt((f_center ** 2).sum(dim=1) * (r_center ** 2).sum(dim=1)) + eps
    corr = num / den
    return (1.0 - corr).mean()


def composite_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    fs: float,                 # kept for API compatibility
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,        # kept for future spectral term
) -> torch.Tensor:
    L_time    = F.l1_loss(fake, real, reduction="mean")
    L_pearson = pearson_corr(fake, real)
    return alpha * L_time + beta * L_pearson


# ── weight initialisation ──────────────────────────────────────────────────────

def init_he(m):
    if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
        try:
            shape = m.weight.shape
        except RuntimeError:
            return
        if len(shape) >= 2:
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
    if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
        try:
            _ = m.bias.shape
        except RuntimeError:
            return
        nn.init.zeros_(m.bias)


# ── parameter counter ──────────────────────────────────────────────────────────

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── training epoch ─────────────────────────────────────────────────────────────

def train_epoch(G, D, R, loader, optG, optD, optR, scaler, device, dz,
                feedback_ratio=0.1, g_steps=1, lambda_fid=0.5):
    """
    Trains one epoch:
      • D with LSGAN + label smoothing.
      • R with MSE (if BSPM is in the batch).
      • G with adversarial + fidelity (L1 + PCC).
    Returns last-batch losses: d_loss, r_loss (or None), g_loss.
    """
    d_last = g_last = None
    r_last = None

    for batch in loader:
        # robust unpacking (2 or 3 elements)
        if len(batch) == 3:
            xr, y, y_bspm = batch
            y_bspm = y_bspm.to(device)
        else:
            xr, y = batch
            y_bspm = None

        xr, y = xr.to(device), y.to(device)
        B = xr.size(0)

        N_feedback = int(B * feedback_ratio)
        z  = torch.randn(B, dz, device=device)
        xf = G(z, y).detach()

        if N_feedback > 0:
            idx     = torch.randperm(B, device=device)[:N_feedback]
            xr[idx] = xf[idx]

        # ==================== Discriminator ====================
        optD.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            d_real   = D(xr, y)                              # (B,1)
            d_fake   = D(xf, y)                              # (B,1)
            real_label = torch.full_like(d_real, 0.9, device=device)
            fake_label = torch.full_like(d_fake, 0.1, device=device)
            d_loss = 0.5 * (F.mse_loss(d_real, real_label) +
                            F.mse_loss(d_fake, fake_label))
        scaler.scale(d_loss).backward()
        scaler.step(optD)
        scaler.update()
        d_last = d_loss.item()

        # ==================== Regression (optional) ====================
        if y_bspm is not None and R is not None and optR is not None:
            optR.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                _, feats = D(xr, y, return_feats=True)
                y_hat    = R(feats)                          # (B, E_t, T)
                r_loss   = F.mse_loss(y_hat, y_bspm)
            scaler.scale(r_loss).backward()
            scaler.step(optR)
            scaler.update()
            r_last = r_loss.item()

        # ==================== Generator ====================
        for _ in range(g_steps):
            z  = torch.randn(B, dz, device=device)
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                gen      = G(z, y)
                d_pred   = D(gen, y)
                adv_label = torch.full_like(d_pred, 0.9, device=device)
                adv_loss  = F.mse_loss(d_pred, adv_label)
                fid_loss  = composite_loss(gen, xr, fs=0.0, alpha=0.5, beta=0.5)
                g_loss    = adv_loss + lambda_fid * fid_loss
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()
            g_last = g_loss.item()

    return d_last, r_last, g_last


# ── experiment runner ──────────────────────────────────────────────────────────

def run_experiment(config: dict, dataset_name: str, dataset_path: Path, dataset_fs: int):
    variant_cfg = config["VARIANT_CFG"]
    print(f"\n🚀 Dataset: {dataset_name.upper()} | Variante: {config['VARIANT_NAME']}")

    # resolve actual .mat path
    dataset_path = resolve_mat_path(Path(dataset_path))

    # loaders
    train_loader, val_loader = build_loader(dataset_path, batch_size=config["B_s"], device=DEVICE)
    mu, sigma = global_stats(train_loader, DEVICE)
    base_ds   = train_loader.dataset.dataset  # Subset -> underlying dataset

    n_cls = num_classes(base_ds)
    C, T  = base_ds.x.shape[1:]

    # models
    G = Generator(C=C, T=T, dz=config["dz"], de=config["de"], n_cls=n_cls).to(DEVICE)

    D = Discriminator(
        C=C, T=T, de=config["de"], n_cls=n_cls,
        use_st=variant_cfg["USE_ST"],
        use_w=variant_cfg["USE_W"],
        block_st=variant_cfg["BLOCK_ST"],
        block_w=variant_cfg["BLOCK_W"],
        fusion=FiLMFusion(C=C),
        tf_mode=variant_cfg["tf_mode"],
        cwt_kwargs=dict(fs=dataset_fs, n_scales=n_scales, f_min=f_min, f_max=f_max, power=0.5),
        stft_kwargs=dict(fs=dataset_fs, n_fft=256, win_length=256, hop_length=64, power=0.5),
        disc_kwargs=disc_kwargs,
    ).to(DEVICE)

    # Opt-2: torch.compile for reduced overhead (before weight init)
    if torch.cuda.is_available():
        G = torch.compile(G, mode="reduce-overhead")
        D = torch.compile(D, mode="reduce-overhead")

    # optional regression head
    has_bspm = hasattr(base_ds, "y_bspm") and base_ds.y_bspm is not None
    if has_bspm:
        E_t    = base_ds.y_bspm.shape[1]
        k      = D.k
        in_dim = C * k + config["de"]
        R = RegressionHead(in_dim=in_dim, E_t=E_t, T=T).to(DEVICE)
    else:
        R = None

    # param counts
    print("Params G:", count_params(G))
    print("Params D:", count_params(D))
    if R is not None:
        print("Params R:", count_params(R))
    print("TOTAL   :", count_params(G) + count_params(D) + (count_params(R) if R else 0))

    # weight init
    G.apply(init_he)
    D.apply(init_he)
    if R is not None:
        R.apply(init_he)

    # optimisers
    optG = torch.optim.RMSprop(G.parameters(), lr=config["LR_G_VALUES"], alpha=config["OPT_ALPHA"])
    optD = torch.optim.RMSprop(D.parameters(), lr=config["LR_D_VALUES"], alpha=config["OPT_ALPHA"])
    optR = torch.optim.RMSprop(R.parameters(), lr=config["LR_G_VALUES"], alpha=config["OPT_ALPHA"]) if R else None

    # Opt-4: explicit device string for GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # output dir
    save_dir = Path("results") / dataset_name / config["VARIANT_NAME"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    start_time = time.time()
    for epoch in range(config["EPOCHS"]):
        d_loss, r_loss, g_loss = train_epoch(
            G, D, R, train_loader, optG, optD, optR, scaler,
            device=DEVICE, dz=config["dz"],
            feedback_ratio=config["FEEDBACK_RATIO"],
            g_steps=config["G_STEPS"],
            lambda_fid=0.5,
        )
        if r_loss is None:
            print(f"[{epoch+1:03d}/{config['EPOCHS']}] D: {d_loss:.4f} | G: {g_loss:.4f}")
        else:
            print(f"[{epoch+1:03d}/{config['EPOCHS']}] D: {d_loss:.4f} | R: {r_loss:.4f} | G: {g_loss:.4f}")

    # save artefacts
    torch.save(G.state_dict(), save_dir / "generator.pt")
    torch.save(D.state_dict(), save_dir / "discriminator.pt")
    if R is not None:
        torch.save(R.state_dict(), save_dir / "regression_head.pt")
    torch.save(
        {**config, "mu": mu, "sigma": sigma, "C": C, "T": T, "n_cls": n_cls, "fs": dataset_fs},
        save_dir / "config.pt",
    )

    print(f"\n✅ Modelos guardados en {save_dir}")
    print(f"⏱️ Tiempo total: {(time.time() - start_time) / 60:.1f} min")

    return dict(G=G, D=D, R=R)
