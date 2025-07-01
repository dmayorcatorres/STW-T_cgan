import torch
import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import time
from pathlib import Path
import random
import os


METRIC_NAMES = ['RMSE', 'MAE', 'Pearson', 'SpectralCos', 'MaxXcorr', 'SSIM1D']

@torch.no_grad()
def collect_examples(G: torch.nn.Module, loader, cfg: dict, seed=4):
    """
    Genera tantos ejemplos sintéticos como reales, por clase.

    Args:
        G: Generador entrenado.
        loader: DataLoader de validación.
        cfg: Configuración con 'device', 'latent_dim' y 'fs'.
        seed: Semilla para reproducibilidad.

    Returns:
        Dict con ejemplos reales y generados por clase.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
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
        z = torch.randn(N, z_dim, device=device)
        y = torch.full((N,), cls, dtype=torch.long, device=device)
        fake_sel = G(z, y).cpu()

        print(f"[Clase {cls}] Reales: {real_sel.shape} | Generadas: {fake_sel.shape}")

        out[cls] = {
            "real": real_sel.cpu().numpy(),
            "fake": fake_sel.numpy(),
            "fs": cfg["fs"]
        }

    G.train()
    return out




def _pearson(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def _ssim_1d(a, b, win=51, sigma=7.0, c1=0.01**2, c2=0.03**2):
    mu_a = gaussian_filter1d(a, sigma, mode='reflect', truncate=win/(2*sigma))
    mu_b = gaussian_filter1d(b, sigma, mode='reflect', truncate=win/(2*sigma))
    sigma_a2 = gaussian_filter1d(a*a, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_a**2
    sigma_b2 = gaussian_filter1d(b*b, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_b**2
    sigma_ab = gaussian_filter1d(a*b, sigma, mode='reflect', truncate=win/(2*sigma)) - mu_a*mu_b

    num = (2*mu_a*mu_b + c1) * (2*sigma_ab + c2)
    denom = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (denom + 1e-12)
    return float(np.mean(ssim_map))


def _metrics(a, b):
    rmse = np.sqrt(np.mean((a - b) ** 2))
    mae = np.mean(np.abs(a - b))
    corr = _pearson(a, b)
    cos_sp = 1 - cosine(np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(b)))
    xc = correlate(a, b, mode='full')
    xcorr = np.max(np.abs(xc)) / (len(a) * a.std() * b.std() + 1e-8)
    ssim = _ssim_1d(a, b)
    return rmse, mae, corr, cos_sp, xcorr, ssim


def evaluate_examples(examples):
    """
    Calcula las métricas para todos los ejemplos por clase.
    """
    out = {}
    for cls, data in examples.items():
        real, fake = data['real'], data['fake']
        R, C, T = real.shape
        real_c = real.transpose(1, 0, 2).reshape(C, -1)
        fake_c = fake.transpose(1, 0, 2).reshape(C, -1)

        cls_metrics = {m: [] for m in METRIC_NAMES}
        for ch in range(C):
            r = real_c[ch]
            f = fake_c[ch]
            r = (r - r.mean()) / (r.std() + 1e-8)
            f = (f - f.mean()) / (f.std() + 1e-8)
            vals = _metrics(r, f)
            for name, val in zip(METRIC_NAMES, vals):
                cls_metrics[name].append(val)

        out[cls] = {k: np.array(v) for k, v in cls_metrics.items()}

    return out


def save_results(params, metrics, filename, total_time, dataset_path, variant_name="default"):
    dataset_name = Path(dataset_path).stem
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "VARIANT": variant_name,
        "DATASET": dataset_name,
        "TIMESTAMP": timestamp,
        "LATENT_DIM": params["dz"],
        "BATCH_SIZE": params["B_s"],
        "POOL_K": params["k"],
        "EMB_DIM": params["de"],
        "LR_G_VALUES": params["LR_G_VALUES"],
        "LR_D_VALUES": params["LR_D_VALUES"],
        "EPOCHS": params["EPOCHS"],
        "G_STEPS": params["G_STEPS"],
        "FEEDBACK_RATIO": params["FEEDBACK_RATIO"],
        "USE_TEMPORAL": params["USE_TEMPORAL"],
        "USE_SPATIAL": params["USE_SPATIAL"],
        "USE_FREQUENCY": params["USE_FREQUENCY"],
        "VAL_SIZE": params["VAL_SIZE"],
        "N_CLASSES": params["N_CLASSES"],
        "G_PARAMS": params["G_PARAMS"],
        "D_PARAMS": params["D_PARAMS"],
        "DEVICE": str(params["DEVICE"]),
        "TRAIN_TIME_S": round(total_time, 2),
        **metrics
    }

    df = pd.DataFrame([row])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    print(f"Resultados guardados en {filename}")
