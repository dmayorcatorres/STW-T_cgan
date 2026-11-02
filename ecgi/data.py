"""ecgi.data — dataset, filters, z-score normalisation, and DataLoader factory.

Opt-3: DataLoader persistent_workers + prefetch_factor applied in build_loader.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from ecgi.config import fs, LP_CUTS, SEED


# ── 1) configurable Butterworth filter ─────────────────────────────────────────

class ButterFilter:
    """
    • HP 0.5 Hz   • notch 50 Hz   • LP variable (dict por etiqueta)
    Se diseña una vez y se reutiliza en cada llamada.
    """
    def __init__(self,
                 fs: float,
                 lp_cuts: dict,
                 hp_cut: float = 0.5,
                 notch: float = 50.0,
                 notch_Q: float = 35.0):

        self.fs      = fs
        self.lp_cuts = lp_cuts

        # HP 0.5 Hz  (4th order)
        self.b_hp, self.a_hp = sps.butter(4, hp_cut, btype="high", fs=fs)

        # Notch 50 Hz (biquad)
        self.b_no, self.a_no = sps.iirnotch(notch, notch_Q, fs=fs)

        # LP bank per label (6th order)
        self.lp_cache = {}
        for tag, fc in lp_cuts.items():
            self.lp_cache[tag] = sps.butter(6, fc, btype="low", fs=fs)

    def __call__(self, tr: torch.Tensor, tag: int):
        """tr: (C,T) tensor  •  tag: numeric label"""
        b_lp, a_lp = self.lp_cache.get(tag, self.lp_cache[min(self.lp_cache)])

        x_np = tr.cpu().numpy()          # (C,T)
        for c in range(x_np.shape[0]):
            sig = x_np[c]
            sig = sps.filtfilt(self.b_hp, self.a_hp, sig)
            sig = sps.filtfilt(self.b_no, self.a_no, sig)
            sig = sps.filtfilt(b_lp, a_lp, sig)
            x_np[c] = sig
        return torch.from_numpy(x_np).to(tr.dtype)


# ── 2) Z-score normaliser ───────────────────────────────────────────────────────

class ZScore:
    def __init__(self, μ, σ, eps=1e-8):
        self.μ, self.σ, self.eps = μ, σ, eps

    def __call__(self, x):
        return (x - self.μ) / (self.σ + self.eps)


# ── 3) ECG dataset ─────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    def __init__(self,
                 path: Path,
                 key_x="emg",            # input key
                 key_y="tag",
                 key_bspm="ecg",         # regression target key
                 tf_x=None,
                 tf_bspm=None,
                 tf_filt: ButterFilter | None = None):

        mat = sio.loadmat(path, squeeze_me=True)

        # --- input (EMG) ---
        self.x = torch.as_tensor(mat[key_x]).float()   # (N,Cx,T)

        # --- labels ---
        y_raw  = torch.as_tensor(mat[key_y]).long().squeeze()
        uniq   = torch.unique(y_raw)
        remap  = {int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        self.y = torch.tensor([remap[int(v)] for v in y_raw])

        # --- regression target (ECG/BSPM) ---
        self.y_bspm = torch.as_tensor(mat[key_bspm]).float()  # (N,Ce,T*) or (N,T*)
        if self.y_bspm.dim() == 2:         # (N,T) → (N,1,T)
            self.y_bspm = self.y_bspm.unsqueeze(1)

        # Align temporal length with input
        if self.y_bspm.size(-1) != self.x.size(-1):
            self.y_bspm = F.interpolate(self.y_bspm, size=self.x.size(-1),
                                        mode="linear", align_corners=False)

        self.tf_x    = tf_x
        self.tf_bspm = tf_bspm
        self.tf_flt  = tf_filt

    def __len__(self): return self.x.size(0)

    def __getitem__(self, idx):
        emg  = self.x[idx]             # (Cx,T)
        tag  = int(self.y[idx])
        ecg  = self.y_bspm[idx]        # (Ce,T)

        if self.tf_x is not None:     emg = self.tf_x(emg)
        if self.tf_flt is not None:   emg = self.tf_flt(emg, tag)
        if self.tf_bspm is not None:  ecg = self.tf_bspm(ecg)

        return emg, tag, ecg


# ── 4) stratified 70/30 split ──────────────────────────────────────────────────

def _stratified_split_indices(labels, ratio, seed=SEED):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, lbl in enumerate(labels.tolist()):
        buckets[lbl].append(i)

    tr, va = [], []
    for idxs in buckets.values():
        idxs = np.array(idxs); rng.shuffle(idxs)
        k = int(len(idxs) * ratio)
        tr.extend(idxs[:k]); va.extend(idxs[k:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


# ── 5) DataLoader factory (Opt-3: persistent_workers + prefetch_factor) ─────────

def build_loader(path: Path, batch_size: int, device: str,
                 split_ratio: float = 0.7, seed: int = SEED):

    tmp  = ECGDataset(path)                 # reads emg + ecg
    μx, σx = tmp.x.mean().item(),     tmp.x.std().item()
    μb, σb = tmp.y_bspm.mean().item(), tmp.y_bspm.std().item()

    filt = ButterFilter(fs=fs, lp_cuts=LP_CUTS)

    dataset = ECGDataset(
        path,
        tf_x=ZScore(μx, σx),
        tf_bspm=ZScore(μb, σb),
        tf_filt=filt,
    )

    tr_idx, va_idx = _stratified_split_indices(dataset.y, split_ratio, seed)
    train_ds, val_ds = Subset(dataset, tr_idx), Subset(dataset, va_idx)

    pin = device == "cuda"
    nw  = 4 if pin else 0
    pw  = nw > 0          # Opt-3: persistent_workers only when workers > 0

    return (
        DataLoader(train_ds, batch_size, True,  pin_memory=pin, num_workers=nw,
                   persistent_workers=pw, prefetch_factor=2 if pw else None),
        DataLoader(val_ds,   batch_size, False, pin_memory=pin, num_workers=nw,
                   persistent_workers=pw, prefetch_factor=2 if pw else None),
    )


def num_classes(dataset: Dataset) -> int:
    return int(dataset.y.max() + 1)
