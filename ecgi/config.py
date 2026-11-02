"""ecgi.config — project-wide constants and dataset/variant registries.

No ecgi dependencies (leaf node).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml as _yaml


# ── config loader ──────────────────────────────────────────────────────────────

def _load_project_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return _yaml.safe_load(f) or {}


_PROJECT_CFG = _load_project_config()

# ── signal constants ────────────────────────────────────────────────────────────

SEED = 125
fs   = 2048                          # Hz — real sampling rate
LP_CUTS = {                          # low-pass cutoff per label id
    0: 200,    # LV-P / SR
    1: 200,    # LV-FW
    2: 200,    # SR
    3: 250,    # EP — wider
}

n_scales = 64
f_min    = 1
f_max    = 200

# ── device ──────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── dataset registry ────────────────────────────────────────────────────────────

DATASETS = {
    "bordeaux": {
        "path": _PROJECT_CFG.get("datasets", {}).get("bordeaux", {}).get(
            "path", "data/bordeaux/bordeaux_rcm_only.mat"),
        "lr_g": 0.0009, "lr_d": 0.0001, "opt_alpha": 0.99, "fs": 2048,
    },
    "auckland": {
        "path": _PROJECT_CFG.get("datasets", {}).get("auckland", {}).get(
            "path", "data/auckland/auckland_rcm_only.mat"),
        "lr_g": 0.0010, "lr_d": 0.0005, "opt_alpha": 0.99, "fs": 2048,
    },
}

# ── discriminator hyperparameters ───────────────────────────────────────────────

disc_kwargs = dict(
    d_model=128,
    heads_t=8,
    heads_c=8,
)

# ── variant registry ────────────────────────────────────────────────────────────

VARIANTS = {
    "Spatial-only": dict(USE_ST=True,  USE_W=False, BLOCK_ST="spatial",    BLOCK_W="w",          tf_mode="none"),
    "STW-T_FiLM":   dict(USE_ST=True,  USE_W=True,  BLOCK_ST="st",         BLOCK_W="w",          tf_mode="cwt"),
    "STFT-T_FiLM":  dict(USE_ST=True,  USE_W=True,  BLOCK_ST="st",         BLOCK_W="w",          tf_mode="stft"),
    "DilConv-T":    dict(USE_ST=True,  USE_W=True,  BLOCK_ST="st",         BLOCK_W="dilconv",    tf_mode="none"),
    "MultiScale-T": dict(USE_ST=True,  USE_W=True,  BLOCK_ST="st",         BLOCK_W="multiscale", tf_mode="none"),
}

# ── evaluation seeds / viz constants ────────────────────────────────────────────

SEEDS   = [5, 6, 7, 8, 9]
_SCALES = np.arange(1, 64)
_CMAP   = "viridis"
