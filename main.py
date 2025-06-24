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

#DATA_PATH = Path("/workspace/ecgi-integrated-framework-gan/data/utah2002/utah2002.mat")
DATA_PATH = Path("/workspace/ecgi-integrated-framework-gan/data/bordeaux/bordeaux_all.mat")
#DATA_PATH = Path("/workspace/ecgi-integrated-framework-gan/data/auckland/auckland.mat")

METRIC_NAMES = ['RMSE', 'MAE', 'Pearson', 'SpectralCos', 'MaxXcorr', 'SSIM1D']


def run_experiment(config: dict):
    # Carga de datos
    train_loader, val_loader = build_loader(DATA_PATH, batch_size=config["BATCH_SIZE"], device=DEVICE)
    n_cls = num_classes(train_loader.dataset.dataset)
    C, T = train_loader.dataset.dataset.x.shape[1:]

    # Inicialización de modelos
    G = Generator(config["LATENT_DIM"], n_cls, C, T, config["POOL_K"], config["EMB_DIM"],
                  use_t=config["USE_TEMPORAL"], use_s=config["USE_SPATIAL"], use_f=config["USE_FREQUENCY"]).to(DEVICE)
    D = Discriminator(n_cls, C, T, config["POOL_K"], config["EMB_DIM"],
                      use_t=config["USE_TEMPORAL"], use_s=config["USE_SPATIAL"], use_f=config["USE_FREQUENCY"]).to(DEVICE)

    G.apply(init_he)
    D.apply(init_he)

    optG = torch.optim.RMSprop(G.parameters(), lr=config["LR_G_VALUES"])
    optD = torch.optim.RMSprop(D.parameters(), lr=config["LR_D_VALUES"])
    scaler = torch.amp.GradScaler()

    # Entrenamiento
    start_time = time.time()
    for epoch in range(config["EPOCHS"]):
        d_loss, g_loss = train_epoch(G, D, train_loader, optG, optD, scaler,
                                     device=DEVICE, latent_dim=config["LATENT_DIM"],
                                     feedback_ratio=config["FEEDBACK_RATIO"], g_steps=config["G_STEPS"])
        print(f"[{epoch+1:03d}/{config['EPOCHS']}] D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")

    total_time = time.time() - start_time

    # Preparar carpeta de resultados
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path("results") / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluación
    examples = collect_examples(
        G=G,
        loader=val_loader,
        cfg=dict(device=DEVICE, latent_dim=config["LATENT_DIM"], fs=2048),
        channels=[15, 30, 45],
        classes=None,
        examples_per_cls=3
    )
    metrics_per_class = evaluate_examples(examples)

    # Resumen de métricas
    summary_metrics = {}
    for metric in METRIC_NAMES:
        all_vals = np.concatenate([metrics_per_class[c][metric] for c in metrics_per_class])
        summary_metrics[f"mean_{metric}"] = np.nanmean(all_vals)
        summary_metrics[f"std_{metric}"] = np.nanstd(all_vals)

    # Guardar resultados
    full_params = {**config,
                   "VAL_SIZE": len(val_loader.dataset),
                   "N_CLASSES": n_cls,
                   "G_PARAMS": sum(p.numel() for p in G.parameters()),
                   "D_PARAMS": sum(p.numel() for p in D.parameters()),
                   "DEVICE": str(DEVICE),
                   "TIMESTAMP": timestamp}
    
    save_results(full_params, summary_metrics, filename="resultados_experimentos.csv",
                 total_time=total_time, dataset_path=DATA_PATH, variant_name="Rama_Temporal_Solo")

    # Visualizaciones
    #plot_comparison(examples=examples, channels=[15, 30, 45], mode="time", save_dir=save_dir)
    #plot_comparison(examples=examples, channels=[15, 30, 45], mode="cwt", save_dir=save_dir)
    #visualize_breakdowns(G, cfg=dict(device=DEVICE, latent_dim=config["LATENT_DIM"]), save_dir=save_dir)

    return dict(G=G, D=D, metrics=metrics_per_class)


# Configuración base
if __name__ == "__main__":
    config = dict(
        LATENT_DIM=80,
        BATCH_SIZE=8,
        POOL_K=16,
        EMB_DIM=64,
        LR_G_VALUES=0.0005, # Utah
        LR_D_VALUES=0.00005, # Utah
        #LR_G_VALUES=0.00018, # Bordeaux
        #LR_D_VALUES=0.0001, # Bordeaux
        #LR_G_VALUES= 0.0002, # Auckland
        #LR_D_VALUES=0.0001, # Auckland
        EPOCHS=30,
        G_STEPS=2,
        FEEDBACK_RATIO=0.1,
        USE_TEMPORAL=True,
        USE_SPATIAL=False,
        USE_FREQUENCY=False
    )

    out = run_experiment(config)
