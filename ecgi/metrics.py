"""ecgi.metrics — signal quality metrics, evaluation runners, and reporting.

No torch dependency at module level; no ecgi imports at module level.
run_for and main_evaluate use lazy imports for torch/ecgi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import correlate, find_peaks
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter


# ── global constants ────────────────────────────────────────────────────────────

FS           = 2048
PER_CHANNEL  = ["rmse", "pearson", "spectralcos", "maxxcorr"]
METRIC_NAMES = PER_CHANNEL + ["ssim_mat"]
DISPLAY_NAMES = {
    "rmse":        "RMSE",
    "pearson":     "Pearson",
    "spectralcos": "SpectralCos",
    "maxxcorr":    "MaxXcorr",
}


# ── 1-D metric helpers ─────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    if min(a.std(), b.std()) < eps:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _metrics_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> tuple[float, float, float, float]:
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    pear = _pearson(a, b, eps=eps)
    spec = 1.0 - cosine(np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(b)))
    xc   = correlate(a, b, mode="full")
    mxc  = float(xc.max() / (len(a) * a.std() * b.std() + eps))
    return rmse, pear, spec, mxc


def _ssim_2d(a: np.ndarray, b: np.ndarray, win: int = 31, sigma: int = 5,
             c1: float = 1e-4, c2: float = 9e-4, eps: float = 1e-12) -> float:
    g = lambda x: gaussian_filter(x, sigma, mode="reflect", truncate=win / (2 * sigma))

    mu_a, mu_b    = g(a), g(b)
    sigma_a2      = g(a * a) - mu_a ** 2
    sigma_b2      = g(b * b) - mu_b ** 2
    sigma_ab      = g(a * b) - mu_a * mu_b

    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a2 + sigma_b2 + c2) + eps
    return float((num / den).mean())


# ── legacy evaluate_examples (uses PER_CHANNEL keys) ──────────────────────────
# (overridden below by the SignalMetrics-based functional wrapper, mirroring
#  the original train.py behaviour where the second definition wins)

def _evaluate_examples_v1(examples: Dict[int, Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for cls, data in examples.items():
        real, fake = data["emg"], data["fake"]   # (N,C,T) arrays
        C = real.shape[1]

        r = real.transpose(1, 0, 2).reshape(C, -1)
        f = fake.transpose(1, 0, 2).reshape(C, -1)
        for m in (r, f):
            m -= m.mean(axis=1, keepdims=True)
            m /= m.std(axis=1, keepdims=True) + 1e-8

        m_cls = {k: [] for k in PER_CHANNEL}
        for ch in range(C):
            vals = _metrics_1d(r[ch], f[ch])
            for name, val in zip(PER_CHANNEL, vals):
                m_cls[name].append(val)
        m_cls["ssim_mat"] = _ssim_2d(r, f)

        out[cls] = {k: np.asarray(v) if isinstance(v, list) else v
                    for k, v in m_cls.items()}
    return out


# ── SignalMetrics — vectorised global metrics per tag ──────────────────────────

@dataclass
class SignalMetrics:
    """
    Compares real vs synthetic blocks per tag.
    For each tag: real,fake ~ (N,C,T). Normalises per-channel (z-score),
    concatenates C×T→vector, and computes a single figure per metric.
    """
    fs: int
    max_lag_ms: float = 50.0
    ssim_win: int = 31
    ssim_sigma: int = 5
    eps: float = 1e-8

    # helpers
    def _to_matrix(self, x: np.ndarray) -> np.ndarray:
        """Returns (C, L) after z-score normalisation per channel. Accepts (N,C,T) or (C,T)."""
        if x.ndim == 3:
            C = x.shape[1]
            X = x.transpose(1, 0, 2).reshape(C, -1)
        elif x.ndim == 2:
            C = x.shape[0]
            X = x.reshape(C, -1)
        else:
            raise ValueError("Expected (N,C,T) or (C,T)")
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + self.eps)
        return X

    def _to_vector(self, x: np.ndarray) -> np.ndarray:
        return self._to_matrix(x).ravel()

    def _pearson(self, a: np.ndarray, b: np.ndarray) -> float:
        if min(a.std(), b.std()) < self.eps:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def _rmse(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _mae(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def _spectral_cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - cosine(np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(b)))

    def _xcorr_delay_ms(self, a: np.ndarray, b: np.ndarray) -> float:
        T       = a.size
        max_lag = int(self.fs * self.max_lag_ms / 1000)
        cc      = correlate(a - a.mean(), b - b.mean(), mode="full")
        lags    = np.arange(-T + 1, T)
        mask    = np.abs(lags) <= max_lag
        lags = lags[mask]; cc = cc[mask]
        tau = lags[np.argmax(cc)]
        return float(tau) / self.fs * 1000.0

    def _peak_rmse(self, a: np.ndarray, b: np.ndarray, *, height=None, distance=None) -> float:
        p_a, _ = find_peaks(a, height=height, distance=distance)
        p_b, _ = find_peaks(b, height=height, distance=distance)
        n = min(len(p_a), len(p_b))
        if n == 0:
            return np.nan
        return float(np.sqrt(np.mean((a[p_a[:n]] - b[p_b[:n]]) ** 2)))

    def _ssim_2d(self, A: np.ndarray, B: np.ndarray, c1: float = 1e-4, c2: float = 9e-4) -> float:
        g = lambda x: gaussian_filter(
            x, self.ssim_sigma, mode="reflect",
            truncate=self.ssim_win / (2 * self.ssim_sigma),
        )
        mu_a, mu_b = g(A), g(B)
        sigma_a2 = g(A * A) - mu_a ** 2
        sigma_b2 = g(B * B) - mu_b ** 2
        sigma_ab = g(A * B) - mu_a * mu_b
        num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a2 + sigma_b2 + c2) + self.eps
        return float((num / den).mean())

    def vector_metrics(self, a_mat: np.ndarray, b_mat: np.ndarray,
                       *, peak_height=None, peak_distance=None) -> Dict[str, float]:
        a = self._to_vector(a_mat); b = self._to_vector(b_mat)
        out = {
            "rmse":           self._rmse(a, b),
            "mae":            self._mae(a, b),
            "pearson":        self._pearson(a, b),
            "spectral_cos":   self._spectral_cos(a, b),
            "xcorr_delay_ms": self._xcorr_delay_ms(a, b),
            "peak_rmse":      self._peak_rmse(a, b, height=peak_height, distance=peak_distance),
        }
        A2 = self._to_matrix(a_mat)
        B2 = self._to_matrix(b_mat)
        ssim_val = self._ssim_2d(A2, B2)
        out["ssim2d"]   = ssim_val
        out["ssim_mat"] = ssim_val  # alias for compatibility
        return out

    def evaluate_examples(self, examples: Dict[int, Dict]) -> Dict[int, Dict]:
        """examples: tag -> {"emg":(N,C,T), "fake":(N,C,T), "fs":int}"""
        return {tag: self.vector_metrics(d["emg"], d["fake"]) for tag, d in examples.items()}


# API funcional de compatibilidad (mirrors train.py: second definition overrides first)
def evaluate_examples(examples: Dict[int, Dict]) -> Dict[int, Dict]:
    first_tag = next(iter(examples))
    fs = int(examples[first_tag]["fs"])
    return SignalMetrics(fs=fs).evaluate_examples(examples)


# ── peak / delay metric functions ─────────────────────────────────────────────

def peak_rmse(
    real: np.ndarray,
    fake: np.ndarray,
    height: float = None,
    distance: int = None,
) -> float:
    """RMSE between peak amplitudes of real and fake."""
    p_real, _ = find_peaks(real, height=height, distance=distance)
    p_fake, _ = find_peaks(fake, height=height, distance=distance)
    n = min(len(p_real), len(p_fake))
    if n == 0:
        return np.nan
    a_real = real[p_real[:n]]
    a_fake = fake[p_fake[:n]]
    return np.sqrt(np.mean((a_real - a_fake) ** 2))


def crosscorr_delay(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    max_lag_ms: float = 50.0,
) -> float:
    """Lag (ms) that maximises cross-correlation between x and y."""
    T       = x.size
    max_lag = int(fs * max_lag_ms / 1000)
    cc      = correlate(x - x.mean(), y - y.mean(), mode="full")
    lags    = np.arange(-T + 1, T)
    mask    = np.abs(lags) <= max_lag
    lags = lags[mask]; cc = cc[mask]
    tau = lags[np.argmax(cc)]
    return tau / fs * 1000.0  # ms


def interchannel_delays(
    sig: np.ndarray,
    fs: float,
    max_lag_ms: float = 50.0,
) -> Dict[Tuple[int, int], float]:
    """Cross-correlation peak lag (ms) for each channel pair."""
    C, T = sig.shape
    delays: Dict[Tuple[int, int], float] = {}
    for i in range(C):
        for j in range(i + 1, C):
            delays[(i, j)] = crosscorr_delay(sig[i], sig[j], fs, max_lag_ms)
    return delays


def peak_rmse_per_channel(
    real: np.ndarray,
    fake: np.ndarray,
    height: float = None,
    distance: int = None,
) -> Dict[int, float]:
    """Peak amplitude RMSE per channel."""
    C, T = real.shape
    return {c: peak_rmse(real[c], fake[c], height=height, distance=distance)
            for c in range(C)}


def crosscorr_delay_per_channel(
    real: np.ndarray,
    fake: np.ndarray,
    fs: float,
    max_lag_ms: float = 50.0,
) -> Dict[int, float]:
    """Alignment lag (ms) per channel."""
    C, T = real.shape
    return {c: crosscorr_delay(real[c], fake[c], fs, max_lag_ms) for c in range(C)}


def summarize_signal_metrics(
    examples: Dict[int, Dict],
    fs: float,
) -> Dict[int, Dict]:
    """
    For each class in examples, returns:
      - peak_rmse:     dict channel→RMSE
      - delay_ch:      dict channel→lag (ms)
      - delay_ic_real: dict channel-pair→inter-channel lag real (ms)
      - delay_ic_fake: dict channel-pair→inter-channel lag fake (ms)
    """
    summary = {}
    for cls, data in examples.items():
        real = data["emg"]   # (N, C, T)
        fake = data["fake"]
        r0 = real[0]
        f0 = fake[0]

        summary[cls] = {
            "peak_rmse":     peak_rmse_per_channel(r0, f0),
            "delay_ch":      crosscorr_delay_per_channel(r0, f0, fs),
            "delay_ic_real": interchannel_delays(r0, fs),
            "delay_ic_fake": interchannel_delays(f0, fs),
        }
    return summary


# ── .mat loader ────────────────────────────────────────────────────────────────

def load_mat_examples(matfile: Path) -> Dict[int, Dict]:
    data     = sio.loadmat(str(matfile))
    examples = {}
    for key, arr in data.items():
        if key.startswith("cls") and key.endswith("_emg"):
            cls = int(key[3:key.find("_")])
            examples[cls] = {
                "emg":  data[f"cls{cls}_emg"],
                "fake": data[f"cls{cls}_fake"],
                "tag":  data[f"cls{cls}_tag"].flatten(),
                "fs":   int(data[f"cls{cls}_fs"].flatten()[0]),
            }
    return examples


# ── TagMatRunner ───────────────────────────────────────────────────────────────

@dataclass
class TagMatRunner:
    patterns: List[str]
    out_dir: Path = Path("results")
    peak_height: Optional[float] = None
    peak_distance: Optional[int] = None

    def _load_mat_examples(self, matfile: Path) -> Dict[int, Dict]:
        """dict[tag] -> {emg, fake, tag, fs}"""
        data     = sio.loadmat(str(matfile))
        examples = {}
        for key in list(data.keys()):
            if key.startswith("cls") and key.endswith("_emg"):
                cls = int(key[3:key.find("_")])
                examples[cls] = {
                    "emg":  data[f"cls{cls}_emg"],
                    "fake": data[f"cls{cls}_fake"],
                    "tag":  data[f"cls{cls}_tag"].flatten(),
                    "fs":   int(np.array(data[f"cls{cls}_fs"]).flatten()[0]),
                }
        return examples

    def _eval_one(self, path: Path) -> List[Dict]:
        if not path.exists():
            print(f"⚠️ No encontrado: {path}")
            return []
        ex      = self._load_mat_examples(path)
        fs      = int(ex[next(iter(ex))]["fs"])
        sm      = SignalMetrics(fs=fs)
        res     = sm.evaluate_examples(ex)

        dataset = path.parent.name
        m       = re.findall(r"seed(\d+)", path.stem)
        seed    = int(m[0]) if m else path.stem

        rows = []
        for tag, mets in res.items():
            for k, v in mets.items():
                rows.append({
                    "dataset": dataset,
                    "file":    path.name,
                    "seed":    seed,
                    "tag":     int(tag),
                    "metric":  k,
                    "value":   float(v) if np.isscalar(v) else v,
                })
        return rows

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict] = []
        for pat in self.patterns:
            all_rows.extend(self._eval_one(Path(pat)))

        if not all_rows:
            raise RuntimeError("No se encontraron resultados para los patrones dados.")

        df = pd.DataFrame(all_rows).sort_values(["dataset", "seed", "tag", "metric"])

        wide_by_seed = (
            df.pivot_table(index=["dataset", "seed", "tag"], columns="metric", values="value", aggfunc="first")
              .reset_index()
              .sort_values(["dataset", "seed", "tag"])
        )

        agg = (
            df.groupby(["dataset", "tag", "metric"])["value"]
              .agg(mean="mean", std="std", n="count")
              .reset_index()
        )
        table_mean_std = (
            agg.assign(**{"mean±std": agg["mean"].round(4).astype(str) + " ± " + agg["std"].round(4).astype(str)})
               .pivot_table(index=["dataset", "tag"], columns="metric", values="mean±std", aggfunc="first")
               .reset_index()
               .sort_values(["dataset", "tag"])
        )
        wide_mean = agg.pivot_table(index=["dataset", "tag"], columns="metric", values="mean", aggfunc="first").reset_index()
        wide_std  = agg.pivot_table(index=["dataset", "tag"], columns="metric", values="std",  aggfunc="first").reset_index()

        (self.out_dir / "summary_global_metrics_per_tag_by_seed.csv").write_text(
            wide_by_seed.to_csv(index=False)
        )
        (self.out_dir / "metrics_mean±std_per_tag.csv").write_text(
            table_mean_std.to_csv(index=False)
        )
        (self.out_dir / "metrics_mean_per_tag.csv").write_text(
            wide_mean.to_csv(index=False)
        )
        (self.out_dir / "metrics_std_per_tag.csv").write_text(
            wide_std.to_csv(index=False)
        )

        expected = df.groupby("dataset")["seed"].nunique().rename("expected_seeds")
        chk      = agg.merge(expected, on="dataset", how="left")
        faltantes = chk[chk["n"] < chk["expected_seeds"]]
        if not faltantes.empty:
            print("⚠️ Hay combinaciones con menos seeds de las esperadas:")
            print(faltantes[["dataset", "tag", "metric", "n", "expected_seeds"]].sort_values(["dataset", "tag", "metric"]))

        return wide_by_seed, table_mean_std, wide_mean, wide_std


# ── main_evaluate — multi-seed evaluation with peak/delay metrics ──────────────

def main_evaluate():
    # lazy imports to keep module torch-free at import time
    from ecgi.config import SEEDS

    RESULTS_DIR = Path("results")
    for ds_dir in RESULTS_DIR.iterdir():
        if not ds_dir.is_dir():
            continue
        print(f"\n🔍 Dataset {ds_dir.name}")

        for seed in SEEDS:
            matfile = ds_dir / f"gen_seed{seed}.mat"
            ex      = load_mat_examples(matfile)

            mets = evaluate_examples(ex)
            if mets is None:
                raise RuntimeError("evaluate_examples devolvió None")

            fs      = ex[next(iter(ex))]["fs"]
            sig_mets = summarize_signal_metrics(ex, fs)

            print(f"\n  — Seed {seed}:")
            for cls in ex.keys():
                print(f"    Clase {cls}:")
                rmse_glob = np.nanmean(mets[cls]["rmse"])
                print(f"      RMSE global:          {rmse_glob:.3f} mV")
                prmse = sig_mets[cls]["peak_rmse"].get(20, np.nan)
                print(f"      RMSE de picos (Ch20):  {prmse:.3f} mV")
                dch   = sig_mets[cls]["delay_ch"].get(20, np.nan)
                print(f"      Desfase (Ch20):        {dch:.2f} ms")
                ic_real = sig_mets[cls]["delay_ic_real"]
                ic_fake = sig_mets[cls]["delay_ic_fake"]
                print(f"      Max desfase intercanal real: {max(ic_real.values()):.2f} ms")
                print(f"      Max desfase intercanal fake: {max(ic_fake.values()):.2f} ms")


# ── run_for — load model + data and produce evaluation + plots ─────────────────

def run_for(
    ds: str,
    *,
    cfg_path: Path,
    data_path: Path,
    fs: int,
    enable_plots: bool = True,
):
    # lazy imports to keep module torch-free at import time
    import torch
    from ecgi.config import DEVICE, SEEDS
    from ecgi.data import build_loader, num_classes
    from ecgi.inference import global_stats, load_models, collect_examples, zscore_np
    from ecgi.viz import plot_comparison

    result_dir = cfg_path.parent / "eval"
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = torch.load(cfg_path, map_location="cpu")
    train_loader, val_loader = build_loader(data_path, batch_size=cfg["B_s"], device=DEVICE)
    base_ds = val_loader.dataset.dataset if hasattr(val_loader.dataset, "dataset") else val_loader.dataset
    n_cls   = num_classes(base_ds)
    C, T    = base_ds.x.shape[1:]

    mu, sigma = global_stats(val_loader, DEVICE)

    G, _ = load_models(cfg_path, C, T, n_cls, DEVICE, fs)

    all_m = {cls: {k: [] for k in METRIC_NAMES} for cls in range(n_cls)}
    for s in SEEDS:
        ex = collect_examples(
            G, val_loader,
            dz=cfg["dz"], fs=fs,
            device=DEVICE, mu=mu, sigma=sigma, seed=s,
            save_path=result_dir / f"gen_seed{s}.mat",
        )
        m = evaluate_examples(ex)
        for cls, mets in m.items():
            for k in METRIC_NAMES:
                all_m[cls][k].append(np.nanmean(mets[k]))

    print(f"\n📊 RESULTS - {ds.upper()} (5 seeds)")
    for cls, mm in all_m.items():
        print(f"\nClass {cls}")
        for k, vals in mm.items():
            arr = np.asarray(vals)
            print(f"  {k:>12}: μ={arr.mean():.4f} ± {arr.std():.4f}")

    if enable_plots:
        ex  = collect_examples(
            G, val_loader,
            dz=cfg["dz"], fs=fs,
            device=DEVICE, mu=mu, sigma=sigma, seed=SEEDS[0],
        )
        z_fn = partial(zscore_np, mu=mu, sigma=sigma)
        plot_comparison(
            ex, channels=[20, 50, 90], mode="time",
            save_dir=result_dir, idx_fake=1, idx_reales=[0, 20, 50],
            zscore_fn=z_fn,
        )
        plot_comparison(
            ex, channels=[20, 50, 90], mode="cwt",
            save_dir=result_dir, idx_fake=1, idx_reales=[0, 20, 50],
            zscore_fn=z_fn,
        )
