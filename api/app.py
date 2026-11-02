"""FastAPI wrapper for the ECGi STW-TCGAN model."""

from __future__ import annotations

import sys
import io
import uuid
import tempfile
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import scipy.io as sio
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Make the project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import ecgi as _train

app = FastAPI(title="ECGi STW-TCGAN API", version="0.1.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RESULTS_DIR = Path(_train._PROJECT_CFG.get("results_dir", "results"))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------
@app.get("/models")
def list_models():
    models = []
    if not _RESULTS_DIR.exists():
        return {"models": models}
    for ds_dir in sorted(_RESULTS_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for variant_dir in sorted(ds_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            ready = (variant_dir / "generator.pt").exists() and (variant_dir / "config.pt").exists()
            models.append(
                {
                    "dataset": ds_dir.name,
                    "variant": variant_dir.name,
                    "ready_for_generate": ready,
                }
            )
    return {"models": models}


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------
def _run_training(dataset: str, variant: str):
    """Background task: run run_experiment for the requested dataset/variant."""
    ds_cfg = _train._PROJECT_CFG.get("datasets", {}).get(dataset, {})
    tr_cfg = _train._PROJECT_CFG.get("training", {})

    if dataset not in _train.DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if variant not in _train.VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}")

    config = {
        "VARIANT_NAME": variant,
        "VARIANT_CFG": _train.VARIANTS[variant],
        "dz": tr_cfg.get("dz", 128),
        "de": tr_cfg.get("de", 8),
        "B_s": tr_cfg.get("batch_size", 8),
        "EPOCHS": tr_cfg.get("epochs", 50),
        "G_STEPS": tr_cfg.get("g_steps", 3),
        "FEEDBACK_RATIO": tr_cfg.get("feedback_ratio", 0.1),
        "LR_G_VALUES": ds_cfg.get("lr_g", _train.DATASETS[dataset]["lr_g"]),
        "LR_D_VALUES": ds_cfg.get("lr_d", _train.DATASETS[dataset]["lr_d"]),
        "OPT_ALPHA": ds_cfg.get("opt_alpha", _train.DATASETS[dataset]["opt_alpha"]),
    }

    _train.run_experiment(
        config=config,
        dataset_name=dataset,
        dataset_path=Path(_train.DATASETS[dataset]["path"]),
        dataset_fs=_train.DATASETS[dataset]["fs"],
    )


@app.post("/train")
def train(
    background_tasks: BackgroundTasks,
    dataset: str = Form(...),
    variant: str = Form("STW-T_FiLM"),
):
    if dataset not in _train.DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset!r}. Valid: {list(_train.DATASETS)}")
    if variant not in _train.VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown variant: {variant!r}. Valid: {list(_train.VARIANTS)}")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_training, dataset, variant)
    return {"job_id": job_id, "status": "queued", "dataset": dataset, "variant": variant}


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    dataset: str = Form(...),
    variant: str = Form("STW-T_FiLM"),
    n_samples: Optional[int] = Form(None),
    seed: int = Form(0),
    file: UploadFile = File(...),
):
    if dataset not in _train.DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset!r}")
    if variant not in _train.VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown variant: {variant!r}")

    model_dir = _RESULTS_DIR / dataset / variant
    gen_path = model_dir / "generator.pt"
    cfg_path = model_dir / "config.pt"

    if not gen_path.exists() or not cfg_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for dataset={dataset!r}, variant={variant!r}. Train first.",
        )

    # --- load config ---
    cfg = torch.load(cfg_path, map_location="cpu", weights_only=False)
    C = int(cfg["C"])
    T = int(cfg["T"])
    dz = int(cfg["dz"])
    de = int(cfg["de"])
    n_cls = int(cfg["n_cls"])
    fs = int(cfg.get("fs", _train.DATASETS[dataset]["fs"]))
    mu = float(cfg["mu"])
    sigma = float(cfg["sigma"])

    # --- reconstruct generator ---
    G = _train.Generator(C=C, T=T, dz=dz, de=de, n_cls=n_cls)
    G.load_state_dict(torch.load(gen_path, map_location=DEVICE, weights_only=False))
    G = G.to(DEVICE)
    G.eval()

    # --- parse uploaded .mat file ---
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        mat = sio.loadmat(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    if "emg" not in mat or "tag" not in mat:
        raise HTTPException(status_code=422, detail="Uploaded .mat must contain 'emg' and 'tag' fields.")

    emg = torch.tensor(mat["emg"], dtype=torch.float32)   # (N, C, T) or (N, T)
    tag = torch.tensor(mat["tag"].squeeze(), dtype=torch.long)

    if emg.dim() == 2:
        emg = emg.unsqueeze(1)  # (N, 1, T)

    if n_samples is not None:
        emg = emg[:n_samples]
        tag = tag[:n_samples]

    # build minimal iterable loader from tensors
    from torch.utils.data import TensorDataset, DataLoader
    ds_obj = TensorDataset(emg, tag)
    loader = DataLoader(ds_obj, batch_size=8, shuffle=False)

    # --- generate ---
    results = _train.collect_examples(
        G, loader,
        dz=dz,
        fs=fs,
        device=torch.device(DEVICE),
        mu=mu,
        sigma=sigma,
        seed=seed,
    )

    # --- serialise to JSON-safe dict ---
    output: dict = {"dataset": dataset, "variant": variant, "classes": {}}
    for cls_id, data in results.items():
        output["classes"][str(cls_id)] = {
            "fake": data["fake"].tolist() if hasattr(data["fake"], "tolist") else data["fake"],
            "emg": data["emg"].tolist() if hasattr(data["emg"], "tolist") else data["emg"],
            "fs": fs,
        }

    return JSONResponse(content=output)
