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
def plot_comparison(*, examples: dict[int, dict], channels: ty.Sequence[int], mode="time", save_dir=None):
    """
    Visualiza comparaciones real vs. sintético en el dominio temporal o CWT.

    Args:
        examples: Dict de ejemplos (salida de collect_examples).
        channels: Lista de índices de canales a visualizar.
        mode: "time" o "cwt".
        save_dir: Carpeta donde guardar las figuras (None = solo muestra).
    """
    assert mode in ("time", "cwt"), "Modo debe ser 'time' o 'cwt'"

    for cls, data in examples.items():
        real = data["real"]
        fake = data["fake"]
        R, C, T = real.shape
        fs = data["fs"]

        if mode == "time":
            t_axis = np.arange(T) / fs
            fig, axes = plt.subplots(R, len(channels),
                                     figsize=(len(channels) * 3.6, R * 2.2),
                                     sharex=True, sharey='row')
            fig.suptitle(f"Class {cls} · Real vs. Synthetic (Time)", weight='bold', y=1.02)

            for r in range(R):
                for c_idx, ch in enumerate(channels):
                    ax = axes[r, c_idx] if R > 1 else axes[c_idx]
                    real_sig = real[r, ch]
                    fake_sig = fake[r, ch]

                    if c_idx == 0:
                        σ = real_sig.std()
                        ax.set_ylim(real_sig.mean() - 5 * σ, real_sig.mean() + 5 * σ)

                    ax.plot(t_axis, real_sig, lw=1.0, color="#1f77b4", label="Real")
                    ax.plot(t_axis, fake_sig, lw=1.0, ls="--", color="#d62728", label="Fake")

                    if r == 0:
                        ax.set_title(f"Channel {ch}")
                    if c_idx == 0:
                        ax.set_ylabel(f"Ex {r + 1}\nAmplitude")
                    if r == R - 1:
                        ax.set_xlabel("Time [s]")
                    if r == 0 and c_idx == 0:
                        ax.legend(frameon=False, fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            if save_dir:
                fig.savefig(f"{save_dir}/time_cls{cls}.png", bbox_inches="tight")
            plt.show()

        elif mode == "cwt":
            all_pw = []
            for r in range(R):
                for ch in channels:
                    all_pw.append(_cwt_power(real[r, ch]))
                    all_pw.append(_cwt_power(fake[r, ch]))
            vmax = np.percentile(np.concatenate(all_pw), 99)

            cols = len(channels) * 2
            fig, axes = plt.subplots(R, cols, figsize=(cols * 2, R * 2.4))
            fig.suptitle(f"CWT Power · Class {cls}", weight='bold', y=.995)

            for r in range(R):
                for c_idx, ch in enumerate(channels):
                    ax_r = axes[r, 2 * c_idx]
                    ax_f = axes[r, 2 * c_idx + 1]

                    pr = _cwt_power(real[r, ch])
                    pf = _cwt_power(fake[r, ch])
                    im_r = ax_r.imshow(pr, cmap=_CMAP, origin='lower', vmax=vmax, aspect='auto')
                    ax_f.imshow(pf, cmap=_CMAP, origin='lower', vmax=vmax, aspect='auto')

                    if r == 0:
                        ax_r.set_title(f"Ch {ch} – Real")
                        ax_f.set_title(f"Ch {ch} – Fake")

            plt.tight_layout()
            if save_dir:
                fig.savefig(f"{save_dir}/cwt_cls{cls}.png", bbox_inches="tight")
            plt.show()


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
    plt.show()
