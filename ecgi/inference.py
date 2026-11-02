"""ecgi.inference — seed control, normalisation helpers, and model I/O utilities."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import savgol_filter

from ecgi.config import DEVICE
from ecgi.models import Generator, Discriminator
from ecgi.fusion import FiLMFusion


# ── seed control ───────────────────────────────────────────────────────────────

def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── normalisation helpers ──────────────────────────────────────────────────────

def zscore_np(x: np.ndarray, mu: float, sigma: float, eps: float = 1e-8) -> np.ndarray:
    return (x - mu) / (sigma + eps)


def zscore_t(x: torch.Tensor, mu: float, sigma: float, eps: float = 1e-8) -> torch.Tensor:
    return (x - mu) / (sigma + eps)


# ── global statistics over a DataLoader ───────────────────────────────────────

@torch.no_grad()
def global_stats(loader, device) -> Tuple[float, float]:
    """
    Supports batches (x, y) or (x, y, ecg).  Computes global μ and σ of x.
    """
    s = s2 = n = 0.0
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device, non_blocking=True)
        s  += xb.sum().item()
        s2 += (xb ** 2).sum().item()
        n  += xb.numel()

    mu  = s / n
    var = max(0.0, s2 / n - mu ** 2)
    sigma = max(1e-8, math.sqrt(var))
    return mu, sigma


# ── example collection ─────────────────────────────────────────────────────────

@torch.no_grad()
def collect_examples(
    G: nn.Module,
    loader,
    *,
    dz: int,
    fs: int,
    device: torch.device,
    mu: float,
    sigma: float,
    seed: int = 0,
    sg_win: int = 11,
    sg_order: int = 3,
    save_path: Optional[Path] = None,
) -> dict[int, dict]:
    """
    For each class (tag) in the loader, returns a dict with:
      'emg':  (N, C, T) real z-scored,
      'fake': (N, C, T) synthetic z-scored,
      'tag':  (N,),
      'fs' :  sampling frequency.
    Accepts loaders that yield (x,y) or (x,y,ecg). The ecg is ignored here.
    """
    if sg_win % 2 == 0 or sg_win <= sg_order:
        raise ValueError("sg_win must be odd and greater than sg_order")

    set_seeds(seed)
    G.eval()

    # 1) Group all real signals by class
    buckets: Dict[int, List[torch.Tensor]] = {}
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Loader must yield (x,y) or (x,y,ecg).")
        xb, yb = batch[0], batch[1]   # ignore ecg if present
        for lbl in torch.unique(yb):
            buckets.setdefault(int(lbl), []).append(xb[yb == lbl])

    out: Dict[int, Dict] = {}
    for cls, stacks in buckets.items():
        # 2) Concatenate and move to device
        real = torch.cat(stacks, dim=0).to(device)  # (N,C,T)

        # 3) Savitzky-Golay filter in numpy
        real_np = real.detach().cpu().numpy()
        for i in range(real_np.shape[0]):
            for c in range(real_np.shape[1]):
                real_np[i, c] = savgol_filter(real_np[i, c], sg_win, sg_order)
        real = torch.from_numpy(real_np).to(device)

        # 4) Z-score normalisation and back to CPU
        real_std = zscore_t(real, mu, sigma).cpu()  # (N,C,T)

        # 5) Generate class-conditioned synthetic samples
        N = real_std.shape[0]
        z = torch.randn(N, dz, device=device)
        y = torch.full((N,), cls, dtype=torch.long, device=device)
        fake = G(z, y)  # (N,C,T)

        # 6) Same SG filter + normalisation
        fake_np = fake.detach().cpu().numpy()
        for i in range(fake_np.shape[0]):
            for c in range(fake_np.shape[1]):
                fake_np[i, c] = savgol_filter(fake_np[i, c], sg_win, sg_order)
        fake = torch.from_numpy(fake_np).to(device)
        fake_std = zscore_t(fake, mu, sigma).cpu()  # (N,C,T)

        # 7) Store
        out[cls] = {
            "emg":  real_std.numpy(),
            "fake": fake_std.numpy(),
            "tag":  np.full((N,), cls, dtype=int),
            "fs":   fs,
        }

    G.train()

    # 8) Optional save to .mat
    if save_path is not None:
        mat_dict = {}
        for cls, data in out.items():
            mat_dict[f"cls{cls}_emg"]  = data["emg"]
            mat_dict[f"cls{cls}_fake"] = data["fake"]
            mat_dict[f"cls{cls}_tag"]  = data["tag"]
            mat_dict[f"cls{cls}_fs"]   = np.array([data["fs"]], dtype=float)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(save_path), mat_dict)
        print(f"✅ Ejemplos guardados en {save_path}")

    return out


# ── path resolver ──────────────────────────────────────────────────────────────

def resolve_mat_path(p: Path, drive_root: Path = Path("/content/drive")) -> Path:
    """
    Returns p if it exists.
    Otherwise, searches by filename inside drive_root and returns the first match.
    """
    p = Path(p)
    if p.exists():
        return p

    matches = list(drive_root.rglob(p.name))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No existe: {p}\n"
            f"Tampoco encontré '{p.name}' dentro de {drive_root}.\n"
            f"Tip: revisa en el panel de archivos de Colab y copia la ruta exacta."
        )

    print(f"⚠️ Ruta no existe: {p}")
    print(f"✅ Encontré '{p.name}' aquí (usando el primero): {matches[0]}")
    return matches[0]


# ── model loader ───────────────────────────────────────────────────────────────

def load_models(
    cfg_path: Path,
    C: int,
    T: int,
    n_cls: int,
    device: torch.device,
    fs: int,
):
    cfg = torch.load(cfg_path, map_location="cpu")
    model_dir = cfg_path.parent

    # Generator
    G = Generator(C=C, T=T, dz=cfg["dz"], de=cfg["de"], n_cls=n_cls).to(device)
    G.load_state_dict(torch.load(model_dir / "generator.pt", map_location=device))
    G.eval()

    # Variant config (new format with fallback for old checkpoints)
    vcfg = cfg.get("VARIANT_CFG", None)
    if vcfg is None:
        vcfg = dict(
            USE_ST=cfg.get("USE_ST", True),
            USE_W=cfg.get("USE_W", True),
            BLOCK_ST=cfg.get("BLOCK_ST", "st"),
            BLOCK_W=cfg.get("BLOCK_W", "w"),
            tf_mode=cfg.get("tf_mode", "cwt"),
        )

    cwt_kwargs  = dict(fs=fs, n_scales=64, f_min=1, f_max=200, power=0.5)
    stft_kwargs = dict(fs=fs, n_fft=256, win_length=256, hop_length=64, power=0.5)

    D = Discriminator(
        C=C, T=T, de=cfg["de"], n_cls=n_cls,
        use_st=vcfg["USE_ST"],
        use_w=vcfg["USE_W"],
        block_st=vcfg["BLOCK_ST"],
        block_w=vcfg["BLOCK_W"],
        fusion=FiLMFusion(C=C),
        tf_mode=vcfg.get("tf_mode", "cwt"),
        cwt_kwargs=cwt_kwargs,
        stft_kwargs=stft_kwargs,
        disc_kwargs=dict(d_model=128, heads_t=8, heads_c=8),
    ).to(device)

    D.load_state_dict(torch.load(model_dir / "discriminator.pt", map_location=device))
    D.eval()

    return G, D
