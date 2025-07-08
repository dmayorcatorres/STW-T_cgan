import torch
import time
import numpy as np
from pathlib import Path

from utils.dataset import build_loader, num_classes
from models.generator import Generator
from models.discriminator import Discriminator
from utils.training import init_he, train_epoch
from utils.metrics import collect_examples, evaluate_examples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "bordeaux": {"path": "/workspace/ecgi-integrated-framework-gan/data/bordeaux/bordeaux_all.mat",
                 "fs":   2048,
                 "lr_g": 0.00018,
                 "lr_d": 0.0001},
    "auckland":{"path": "/workspace/ecgi-integrated-framework-gan/data/auckland/auckland.mat",
                 "fs":   2048,
                 "lr_g": 0.0002,
                 "lr_d": 0.0001},
}

OPTIMIZERS = {
    "RMSprop": torch.optim.RMSprop,
    "Adam":    torch.optim.Adam,
    "AdamW":   torch.optim.AdamW,
    "SGD":     torch.optim.SGD,
}

def make_grid(base_g, base_d):
    factors = [0.2, 1.0, 5.0]
    return [(base_g * fg, base_d * fd) for fg in factors for fd in factors]

# Arma la grilla de configuraciones
OPT_GRID = []
for ds_name, info in DATASETS.items():
    for opt_name, OptCls in OPTIMIZERS.items():
        for lr_g, lr_d in make_grid(info["lr_g"], info["lr_d"]):
            if opt_name in ("Adam", "AdamW"):
                for betas in [(0.5,0.999),(0.9,0.999)]:
                    OPT_GRID.append((ds_name, opt_name, OptCls, lr_g, lr_d, {"betas": betas}))
            elif opt_name=="RMSprop":
                for alpha in [0.9,0.99]:
                    OPT_GRID.append((ds_name, opt_name, OptCls, lr_g, lr_d, {"alpha": alpha}))
            else:
                for momentum in [0.0,0.9]:
                    OPT_GRID.append((ds_name, opt_name, OptCls, lr_g, lr_d, {"momentum": momentum}))

METRIC = "SpectralCos"

def tune_by_spectralcos(config, ds_name):
    info = DATASETS[ds_name]
    train_loader, val_loader = build_loader(Path(info["path"]), batch_size=8, device=DEVICE)
    C, T = train_loader.dataset.dataset.x.shape[1:]
    n_cls = num_classes(train_loader.dataset.dataset)

    records = []
    for (d, opt_name, OptCls, lr_g, lr_d, extra_hp) in OPT_GRID:
        if d != ds_name: continue

        G = Generator(C,T,config["dz"],config["de"],n_cls).to(DEVICE)
        D = Discriminator(C,T,config["k"],config["de"],n_cls,
                          use_t=True,use_s=False,use_f=False).to(DEVICE)
        G.apply(init_he); D.apply(init_he)

        optG = OptCls(G.parameters(), lr=lr_g, **extra_hp)
        optD = OptCls(D.parameters(), lr=lr_d, **extra_hp)
        scaler = torch.amp.GradScaler()

        # 5 iteraciones rápidas
        for _ in range(5):
            train_epoch(G,D,train_loader,optG,optD,scaler,
                        device=DEVICE,dz=config["dz"],
                        feedback_ratio=config["FEEDBACK_RATIO"],g_steps=1)

        ex = collect_examples(G, val_loader,
                              dict(device=DEVICE,dz=config["dz"],fs=info["fs"]),
                              seed=0)
        mets = evaluate_examples(ex)
        vals = np.concatenate([mets[c][METRIC] for c in mets])
        records.append((opt_name, lr_g, lr_d, extra_hp, vals.mean(), vals.std()))

    # ordena por mean desc, std asc
    records.sort(key=lambda x: (-x[4], x[5]))

    # imprime encabezado
    hdr = f"{'Opt':<8} {'lrG':<10} {'lrD':<10} {'extra_hp':<20} {'mean_SC':>8} {'std_SC':>8}"
    print(f"\n=== Tuning {ds_name.upper()} ===")
    print(hdr)
    print("-"*len(hdr))
    for opt_name, lr_g, lr_d, extra_hp, m, s in records:
        print(f"{opt_name:<8} {lr_g:<10.6f} {lr_d:<10.6f} {str(extra_hp):<20} {m:8.6f} {s:8.6f}")

if __name__=="__main__":
    cfg = dict(dz=128, de=8, k=16, FEEDBACK_RATIO=0.1)
    for ds in DATASETS:
        tune_by_spectralcos(cfg, ds)
