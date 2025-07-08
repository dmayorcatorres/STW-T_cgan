import os
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from utils.dataset import build_loader, num_classes
from models.generator import Generator
from models.discriminator import Discriminator
#from models.generator2D import Generator2D
#from models.discriminator2D import Discriminator2D
from utils.training import init_he, train_epoch
from utils.visualization import plot_comparison, visualize_breakdowns
import pandas as pd
from tabulate import tabulate
from utils.metrics   import (
    collect_examples, evaluate_examples, save_results,
    METRIC_NAMES, DISPLAY_NAMES                    # ← lista única y alias
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    #"utah": {
    #    "path": "/workspace/ecgi-integrated-framework-gan/data/utah2002/utah2002.mat",
    #    "lr_g": 0.0005,
    #    "lr_d": 0.00005,
    #    "fs": 1000
    #},
    "bordeaux": {
        "path": "/workspace/ecgi-integrated-framework-gan/data/bordeaux/bordeaux_rcm_only.mat",
        "lr_g": 0.0009,
        "lr_d": 0.0001,
        "opt_alpha": 0.99,   # el extra_hp que favorizó RMSprop
        "fs": 2048  # Ejemplo, confirma el real
    },
    "auckland": {
        "path": "/workspace/ecgi-integrated-framework-gan/data/auckland/auckland.mat",
        "lr_g": 0.0010,
        "lr_d": 0.0005,
        "opt_alpha": 0.99,   # el extra_hp que favorizó RMSprop
        "fs": 2048  # Ejemplo, confirma el real
    }
}



def run_experiment(config: dict, dataset_name: str, dataset_path: Path, dataset_fs: int):
    print(f"\n🚀 Ejecutando en dataset: {dataset_name.upper()}")

    train_loader, val_loader = build_loader(dataset_path, batch_size=config["B_s"], device=DEVICE)
    n_cls = num_classes(train_loader.dataset.dataset)
    C, T = train_loader.dataset.dataset.x.shape[1:]

    #G = Generator2D(C, T, config["dz"], config["de"], n_cls).to(DEVICE)
    #D = Discriminator2D(C, T, config["de"], n_cls,
                        #use_st=config["USE_ST"],
                        #use_f=config["USE_FREQUENCY"]).to(DEVICE)

    G = Generator(
        C=train_loader.dataset.dataset.x.shape[1],
        T=train_loader.dataset.dataset.x.shape[2],
        dz=config["dz"],
        de=config["de"],
        n_cls=num_classes(train_loader.dataset.dataset)
    ).to(DEVICE)

    D = Discriminator(
            C=train_loader.dataset.dataset.x.shape[1],
            T=train_loader.dataset.dataset.x.shape[2],
            de=config["de"],                # ←   solo estos cuatro
            n_cls=n_cls,
            use_t=config["USE_TEMPORAL"],
            use_s=config["USE_SPATIAL"],
            use_f=config["USE_FREQUENCY"],
            block_s =config["SPATIAL_BLOCK"],
            block_t=config["TEMPORAL_BLOCK"],
    ).to(DEVICE)


    G.apply(init_he)
    D.apply(init_he)

    optG = torch.optim.RMSprop( G.parameters(), lr=config["LR_G_VALUES"], alpha=config["OPT_ALPHA"])
    optD = torch.optim.RMSprop( D.parameters(), lr=config["LR_D_VALUES"], alpha=config["OPT_ALPHA"])

    scaler = torch.amp.GradScaler()

    start_time = time.time()
    for epoch in range(config["EPOCHS"]):
        d_loss, g_loss = train_epoch(G, D, train_loader, optG, optD, scaler,
                                     device=DEVICE, dz=config["dz"],
                                     feedback_ratio=config["FEEDBACK_RATIO"], g_steps=config["G_STEPS"])
        print(f"[{epoch+1:03d}/{config['EPOCHS']}] D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")

    total_time = time.time() - start_time

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path("results") / dataset_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluación
    examples = collect_examples(
        G=G,
        loader=val_loader,
        cfg=dict(device=DEVICE, dz=config["dz"], fs=dataset_fs),
        seed=42
    )
    metrics_per_class = evaluate_examples(examples)

    all_rows = []   # ← lista donde iremos guardando cada fila
    # ───────────────────────────────────────────────────────────────
    for cls, metrics in metrics_per_class.items():
        class_metrics = {f"mean_{m}": np.nanmean(metrics[m]) for m in METRIC_NAMES}
        class_metrics.update({f"std_{m}": np.nanstd(metrics[m]) for m in METRIC_NAMES})

        full_params = {**config,
                    "DATASET": dataset_name,
                    "CLASE": cls,
                    "TIMESTAMP": timestamp,
                    "VAL_SIZE": len(val_loader.dataset),
                    "N_CLASSES": n_cls,
                    "G_PARAMS": sum(p.numel() for p in G.parameters()),
                    "D_PARAMS": sum(p.numel() for p in D.parameters()),
                    "DEVICE": DEVICE}

        # 1)  guardar en CSV
        save_results(full_params, class_metrics,
                    filename="resultados_experimentos.csv",
                    total_time=total_time,
                    dataset_path=dataset_path,
                    variant_name="Rama_Temporal_Solo")

        # 2)  añadir a la lista para impresión
        row = {**full_params, **class_metrics}
        all_rows.append(row)

    # ───────────────────── impresión en consola ─────────────────────
    df = pd.DataFrame(all_rows)

# ───────────── BLOQUE DE IMPRESIÓN EN CONSOLA ─────────────
    cols_show = ["DATASET", "CLASE"] + [f"mean_{m}" for m in METRIC_NAMES]
    renamer   = {f"mean_{m}": f"mean_{DISPLAY_NAMES[m]}" for m in METRIC_NAMES}

    avg_rows = (df.groupby("DATASET")[cols_show[2:]]
                  .mean()
                  .reset_index()
                  .assign(CLASE="AVG"))

    df_print = (pd.concat([df[cols_show], avg_rows[cols_show]], ignore_index=True)
                  .rename(columns=renamer)
                  .sort_values(["DATASET", "CLASE"]))

    print("\n======  Métricas de validación  ======")
    print(tabulate(df_print, headers="keys", tablefmt="pretty", floatfmt=".4f"))


    # Visualizaciones
    plot_comparison(
        examples=examples,
        channels=[15, 30, 45],
        mode="time",
        save_dir= save_dir,
        idx_fake=1,
        idx_reales=[0, 1, 2]
    )

    # Visualizaciones
    plot_comparison(
        examples=examples,
        channels=[15, 30, 45],
        mode="cwt",
        save_dir= save_dir,
        idx_fake=1,
        idx_reales=[0, 1, 2]
    )
    #plot_comparison(examples=examples, channels=[15, 30, 45], mode="cwt", save_dir=save_dir)
    #visualize_breakdowns(G, cfg=dict(device=DEVICE, dz=config["dz"]), save_dir=save_dir)

    return dict(G=G, D=D, metrics=metrics_per_class)


# Configuración base
if __name__ == "__main__":
 

    for dataset_name, dataset_info in DATASETS.items():

        config = dict(
            dz=128,
            B_s=8,
            k=16,
            de=8,
            LR_G_VALUES=dataset_info["lr_g"],
            LR_D_VALUES=dataset_info["lr_d"],
            OPT_ALPHA=dataset_info["opt_alpha"],  # Extra HP para RMSprop
            EPOCHS=30,
            G_STEPS=3,
            FEEDBACK_RATIO=0.15,
            USE_TEMPORAL=True,
            USE_SPATIAL= False,
            USE_FREQUENCY=False,
            USE_ST = True,  # Rama 2D spatiotemporal
            SPATIAL_BLOCK=   "spatial_axial",
            TEMPORAL_BLOCK=  "transformer_temporal",
        )
        
        run_experiment(config, dataset_name, Path(dataset_info["path"]), dataset_info["fs"])

