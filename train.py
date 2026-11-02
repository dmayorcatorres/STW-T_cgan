"""train.py — thin CLI entrypoint for ECGi STW-TCGAN training."""
from __future__ import annotations

import argparse
from pathlib import Path

from ecgi.config import _PROJECT_CFG, DATASETS, VARIANTS
from ecgi.training import run_experiment


def _parse_args():
    p = argparse.ArgumentParser(description="Train ECGi STW-TCGAN")
    p.add_argument("--dataset", choices=list(DATASETS), default=None,
                   help="Dataset to train on (default: all)")
    p.add_argument("--variant", choices=list(VARIANTS), default=None,
                   help="Model variant to train (default: all)")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override number of epochs from config.yaml")
    return p.parse_args()


def main():
    args = _parse_args()
    tr   = _PROJECT_CFG.get("training", {})

    ds = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    vs = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS

    for dsname, dsinfo in ds.items():
        for vname, vcfg in vs.items():
            run_experiment(
                config=dict(
                    VARIANT_NAME=vname,
                    VARIANT_CFG=vcfg,
                    dz=tr.get("dz", 128),
                    de=tr.get("de", 8),
                    B_s=tr.get("batch_size", 8),
                    EPOCHS=args.epochs or tr.get("epochs", 50),
                    G_STEPS=tr.get("g_steps", 3),
                    FEEDBACK_RATIO=tr.get("feedback_ratio", 0.1),
                    LR_G_VALUES=dsinfo["lr_g"],
                    LR_D_VALUES=dsinfo["lr_d"],
                    OPT_ALPHA=dsinfo["opt_alpha"],
                ),
                dataset_name=dsname,
                dataset_path=Path(dsinfo["path"]),
                dataset_fs=dsinfo["fs"],
            )


if __name__ == "__main__":
    main()
