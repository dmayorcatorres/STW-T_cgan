import matplotlib.pyplot as plt
import numpy as np
import torch
import typing as ty
import matplotlib as mpl
import pywt
import os

# Configuración visual global
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
})

# Parámetros CWT globales
_SCALES = np.arange(1, 64)
_CMAP = "viridis"
_FS = 2048  # Hz

def _cwt_power(signal: np.ndarray) -> np.ndarray:
    """
    Mapa de potencia CWT usando PyWavelets.
    """
    coef, _ = pywt.cwt(signal, _SCALES, 'morl', sampling_period=1/_FS)
    power = np.abs(coef) ** 2
    return power

@torch.no_grad()
def visualize_breakdowns(G, cfg, class_label=0, save_dir=None):
    """
    Genera y guarda visualización de cada rama del generador.

    Args:
        G: Generador.
        cfg: Diccionario de configuración (debe tener device y latent_dim).
        class_label: Clase de ejemplo.
        save_dir: Ruta de carpeta donde guardar las imágenes (opcional).
    """
    z = torch.randn(1, cfg["latent_dim"], device=cfg["device"])
    y = torch.full((1,), class_label, dtype=torch.long, device=cfg["device"])
    fake, (t, s, f, m) = G(z, y, return_br=True)

    for name, tensor in zip(["Generated", "Temporal", "Spatial", "Spectral", "Merged"],
                            [fake[0], t[0], s[0], f[0], m[0]]):
        plot_heat(tensor, title=f"{name} Breakdown")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name}_breakdown.png")
            plt.savefig(path, bbox_inches="tight")
            print(f"[✔] Imagen guardada en {path}")

        plt.close()

        
@torch.no_grad()
def plot_comparison(*, examples: dict[int, dict], channels: ty.Sequence[int],
                    mode="time", save_dir=None, idx_fake=0, idx_reales=None):
    """
    Compara un ejemplo sintético específico contra varios ejemplos reales, por clase.

    Si no se pasan idx_reales, se toma automáticamente el máximo posible.
    Si hay índices fuera de rango, se descartan con advertencia.
    """
    assert mode in ("time", "cwt"), "Modo debe ser 'time' o 'cwt'"

    for cls, data in examples.items():
        real = data["real"]
        fake = data["fake"]
        fs = data["fs"]

        N_real = real.shape[0]
        N_fake = fake.shape[0]

        print(f"[Class {cls}] Real Samples: {N_real} | Generated: {N_fake}")

        if idx_fake >= N_fake:
            print(f"[⚠] idx_fake={idx_fake} fuera de rango, usando idx_fake=0.")
            idx_fake = 0

        if idx_reales is None:
            idx_reales = list(range(min(3, N_real)))
        else:
            idx_reales = [i for i in idx_reales if i < N_real]
            if not idx_reales:
                print(f"[⚠] Ningún índice de idx_reales válido para clase {cls}, se omite.")
                continue

        n_reales = len(idx_reales)
        t_axis = np.arange(real.shape[2]) / fs

        if mode == "time":
            fig, axes = plt.subplots(n_reales + 1, len(channels),
                                     figsize=(len(channels) * 3.6, (n_reales + 1) * 2.2),
                                     sharex=True, sharey='col')
            fig.suptitle(f"Class {cls} · Real Samples vs. Generated (Time)", weight='bold', y=1.02)

            for r_idx, real_idx in enumerate(idx_reales):
                for c_idx, ch in enumerate(channels):
                    ax = axes[r_idx, c_idx]
                    sig = real[real_idx, ch]
                    ax.plot(t_axis, sig, lw=1.0, color="#1f77b4")
                    if r_idx == 0:
                        ax.set_title(f"Channel {ch}")
                    if c_idx == 0:
                        ax.set_ylabel(f"Real {real_idx}")

            for c_idx, ch in enumerate(channels):
                ax = axes[-1, c_idx]
                sig = fake[idx_fake, ch]
                ax.plot(t_axis, sig, lw=1.0, color="#d62728")
                if c_idx == 0:
                    ax.set_ylabel(f"Generated {idx_fake}")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            if save_dir:
                fig.savefig(f"{save_dir}/time_cls{cls}_fake{idx_fake}.png", bbox_inches="tight")
            plt.close()

        elif mode == "cwt":
            all_pw = []
            for real_idx in idx_reales:
                for ch in channels:
                    all_pw.append(_cwt_power(real[real_idx, ch]))
            for ch in channels:
                all_pw.append(_cwt_power(fake[idx_fake, ch]))

            vmax = np.percentile(np.concatenate(all_pw), 99)
            cols = len(channels)
            fig, axes = plt.subplots(n_reales + 1, cols, figsize=(cols * 2, (n_reales + 1) * 2.4))
            fig.suptitle(f"CWT Power · Class {cls}", weight='bold', y=.995)

            for r_idx, real_idx in enumerate(idx_reales):
                for c_idx, ch in enumerate(channels):
                    ax = axes[r_idx, c_idx]
                    pw = _cwt_power(real[real_idx, ch])
                    ax.imshow(pw, cmap=_CMAP, origin='lower', vmax=vmax, aspect='auto')
                    if r_idx == 0:
                        ax.set_title(f"Ch {ch}")
                    if c_idx == 0:
                        ax.set_ylabel(f"Real {real_idx}")

            for c_idx, ch in enumerate(channels):
                ax = axes[-1, c_idx]
                pw = _cwt_power(fake[idx_fake, ch])
                ax.imshow(pw, cmap=_CMAP, origin='lower', vmax=vmax, aspect='auto')
                if c_idx == 0:
                    ax.set_ylabel(f"Generated {idx_fake}")

            plt.tight_layout()
            if save_dir:
                fig.savefig(f"{save_dir}/cwt_cls{cls}_fake{idx_fake}.png", bbox_inches="tight")
            plt.close()



@torch.no_grad()
def plot_heat(t: torch.Tensor, title: str, save_path=None):
    """
    Visualiza un mapa de calor para una señal 2D.
    """
    a = t.cpu().numpy()
    plt.figure(figsize=(6, 2.4))
    plt.imshow(a, origin="lower", aspect="auto", cmap="viridis")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
