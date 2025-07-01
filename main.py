import os
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from utils.dataset import build_loader, num_classes
from models.generator import Generator
from models.discriminator import Discriminator
from utils.training import init_he, train_epoch
from utils.metrics import collect_examples, evaluate_examples, save_results
from utils.visualization import plot_comparison, visualize_breakdowns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    #"utah": {
    #    "path": "/workspace/ecgi-integrated-framework-gan/data/utah2002/utah2002.mat",
    #    "lr_g": 0.0005,
    #    "lr_d": 0.00005,
    #    "fs": 1000
    #},
    "bordeaux": {
        "path": "/workspace/ecgi-integrated-framework-gan/data/bordeaux/bordeaux_all.mat",
        "lr_g": 0.00018,
        "lr_d": 0.0001,
        "fs": 2048  # Ejemplo, confirma el real
    },
    "auckland": {
        "path": "/workspace/ecgi-integrated-framework-gan/data/auckland/auckland.mat",
        "lr_g": 0.0002,
        "lr_d": 0.0001,
        "fs": 2048  # Ejemplo, confirma el real
    }
}


METRIC_NAMES = ['RMSE', 'MAE', 'Pearson', 'SpectralCos', 'MaxXcorr', 'SSIM1D']


def run_experiment(config: dict, dataset_name: str, dataset_path: Path, dataset_fs: int):
    print(f"\n🚀 Ejecutando en dataset: {dataset_name.upper()}")

    train_loader, val_loader = build_loader(dataset_path, batch_size=config["B_s"], device=DEVICE)
    n_cls = num_classes(train_loader.dataset.dataset)
    C, T = train_loader.dataset.dataset.x.shape[1:]

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
        k=config["k"],
        de=config["de"],
        n_cls=num_classes(train_loader.dataset.dataset),
        use_t=config["USE_TEMPORAL"],
        use_s=config["USE_SPATIAL"],
        use_f=config["USE_FREQUENCY"]
    ).to(DEVICE)

    G.apply(init_he)
    D.apply(init_he)

    optG = torch.optim.RMSprop(G.parameters(), lr=config["LR_G_VALUES"])
    optD = torch.optim.RMSprop(D.parameters(), lr=config["LR_D_VALUES"])
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

    # Guardar métricas por clase
    for cls, metrics in metrics_per_class.items():
        class_metrics = {f"mean_{m}": np.nanmean(metrics[m]) for m in METRIC_NAMES}
        class_metrics.update({f"std_{m}": np.nanstd(metrics[m]) for m in METRIC_NAMES})

        full_params = {**config,
                    "DATASET": dataset_name,
                    "VAL_SIZE": len(val_loader.dataset),
                    "N_CLASSES": n_cls,
                    "G_PARAMS": sum(p.numel() for p in G.parameters()),
                    "D_PARAMS": sum(p.numel() for p in D.parameters()),
                    "DEVICE": str(DEVICE),
                    "TIMESTAMP": timestamp,
                    "CLASE": cls}

        save_results(full_params, class_metrics, filename="resultados_experimentos.csv",
                    total_time=total_time, dataset_path=dataset_path, variant_name="Rama_Temporal_Solo")

    # Guardar métricas globales
    summary_metrics = {}
    for metric in METRIC_NAMES:
        all_vals = np.concatenate([metrics_per_class[c][metric] for c in metrics_per_class])
        summary_metrics[f"mean_{metric}"] = np.nanmean(all_vals)
        summary_metrics[f"std_{metric}"] = np.nanstd(all_vals)

    full_params_global = {**config,
                        "DATASET": dataset_name,
                        "VAL_SIZE": len(val_loader.dataset),
                        "N_CLASSES": n_cls,
                        "G_PARAMS": sum(p.numel() for p in G.parameters()),
                        "D_PARAMS": sum(p.numel() for p in D.parameters()),
                        "DEVICE": str(DEVICE),
                        "TIMESTAMP": timestamp,
                        "CLASE": "GLOBAL"}

    save_results(full_params_global, summary_metrics, filename="resultados_experimentos.csv",
                total_time=total_time, dataset_path=dataset_path, variant_name="Rama_Temporal_Solo")
    
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
            EPOCHS=30,
            G_STEPS=2,
            FEEDBACK_RATIO=0.1,
            USE_TEMPORAL=True,
            USE_SPATIAL=False,
            USE_FREQUENCY=False
        )
        
        run_experiment(config, dataset_name, Path(dataset_info["path"]), dataset_info["fs"])

