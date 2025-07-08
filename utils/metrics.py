import torch
import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
from pathlib import Path
import random
import os

# ─────────────────────────  MÉTRICAS  ─────────────────────────
PER_CHANNEL  = ['rmse', 'pearson', 'spectralcos', 'maxxcorr']   # ← NEW
METRIC_NAMES = PER_CHANNEL + ['ssim_mat']                       # lista global

DISPLAY_NAMES = {                      # alias legibles
    'rmse':        'RMSE',
    'pearson':     'Pearson',
    'spectralcos': 'SpectralCos',
    'maxxcorr':    'MaxXcorr',
    'ssim_mat':    'SSIM2D'
}

@torch.no_grad()
def collect_examples(G: torch.nn.Module, loader, cfg: dict, seed=4):
    """Genera tantos ejemplos sintéticos como reales, por clase."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    G.eval()
    device = cfg["device"]
    z_dim = cfg["dz"]

    # Agrupar ejemplos por clase
    buckets = {}
    for x, y in loader:
        for lbl in torch.unique(y):
            buckets.setdefault(int(lbl), []).append(x[y == lbl])

    out = {}
    for cls, samples in buckets.items():
        real_full = torch.cat(samples, dim=0)
        N, C, T = real_full.shape

        real_sel = real_full.to(device)
        z   = torch.randn(N, z_dim, device=device)
        y   = torch.full((N,), cls, dtype=torch.long, device=device)
        fake_sel = G(z, y).cpu()

        out[cls] = {
            "real": real_sel.cpu().numpy(),
            "fake": fake_sel.numpy(),
            "fs":   cfg["fs"]
        }

    G.train()
    return out


# ---------- utilidades ----------
def _pearson(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def _metrics_1d(a, b):
    """Devuelve rmse, pearson, spectralcos y maxxcorr (en ese orden)."""
    rmse        = np.sqrt(np.mean((a - b) ** 2))
    pearson     = _pearson(a, b)
    spectralcos = 1 - cosine(np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(b)))
    xc          = correlate(a, b, mode='full')
    maxxcorr    = np.max(np.abs(xc)) / (len(a) * a.std() * b.std() + 1e-8)
    return rmse, pearson, spectralcos, maxxcorr


def _ssim_2d(a, b, win=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    """SSIM bidimensional con núcleo gaussiano (entradas 2-D)."""
    mu_a = gaussian_filter(a, sigma, mode='reflect', truncate=win/(2*sigma))
    mu_b = gaussian_filter(b, sigma, mode='reflect', truncate=win/(2*sigma))
    sigma_a2 = gaussian_filter(a * a, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_a**2
    sigma_b2 = gaussian_filter(b * b, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_b**2
    sigma_ab = gaussian_filter(a * b, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_a * mu_b
    num   = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denom = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-12
    return float(np.mean(num / denom))


# ---------- evaluación ----------
def evaluate_examples(examples):
    """
    Calcula métricas canal a canal + SSIM matricial global por clase.
    Retorna: out[clase][métrica] -> np.ndarray  o  escalar.
    """
    out = {}
    for cls, data in examples.items():
        real, fake = data['real'], data['fake']         # (R, C, T)
        R, C, T = real.shape

        # Reagrupamos en matrices 2-D (C, R*T)
        real_c = real.transpose(1, 0, 2).reshape(C, -1)
        fake_c = fake.transpose(1, 0, 2).reshape(C, -1)

        # Normalización por canal
        real_c = (real_c - real_c.mean(axis=1, keepdims=True)) / (real_c.std(axis=1, keepdims=True) + 1e-8)
        fake_c = (fake_c - fake_c.mean(axis=1, keepdims=True)) / (fake_c.std(axis=1, keepdims=True) + 1e-8)

        # Métricas canal a canal
        cls_metrics = {m: [] for m in PER_CHANNEL}      # sólo las per-channel
        for ch in range(C):
            vals = _metrics_1d(real_c[ch], fake_c[ch])
            for name, val in zip(PER_CHANNEL, vals):
                cls_metrics[name].append(val)

        # SSIM global
        cls_metrics['ssim_mat'] = _ssim_2d(real_c, fake_c)

        # Empaquetado
        out[cls] = {k: np.array(v) if isinstance(v, list) else v
                    for k, v in cls_metrics.items()}
    return out


# ---------- guardar resultados ----------
def save_results(params, metrics, filename, total_time, dataset_path, variant_name="default"):
    dataset_name = Path(dataset_path).stem
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "VARIANT":       variant_name,
        "DATASET":       dataset_name,
        "TIMESTAMP":     timestamp,
        "LATENT_DIM":    params["dz"],
        "BATCH_SIZE":    params["B_s"],
        "POOL_K":        params["k"],
        "EMB_DIM":       params["de"],
        "LR_G_VALUES":   params["LR_G_VALUES"],
        "LR_D_VALUES":   params["LR_D_VALUES"],
        "EPOCHS":        params["EPOCHS"],
        "G_STEPS":       params["G_STEPS"],
        "FEEDBACK_RATIO":params["FEEDBACK_RATIO"],
        "USE_TEMPORAL":  params["USE_TEMPORAL"],
        "USE_SPATIAL":   params["USE_SPATIAL"],
        "USE_FREQUENCY": params["USE_FREQUENCY"],
        "VAL_SIZE":      params["VAL_SIZE"],
        "N_CLASSES":     params["N_CLASSES"],
        "G_PARAMS":      params["G_PARAMS"],
        "D_PARAMS":      params["D_PARAMS"],
        "DEVICE":        str(params["DEVICE"]),
        "TRAIN_TIME_S":  round(total_time, 2),
        **metrics
    }

    df = pd.DataFrame([row])
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)
    print(f"Resultados guardados en {filename}")
