"""ecgi.viz — signal visualisation utilities (time-domain and CWT power)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pywt

from ecgi.config import _SCALES, _CMAP


# ── CWT power spectrum helper ──────────────────────────────────────────────────

def _cwt_power(sig: np.ndarray, fs: int) -> np.ndarray:
    coefs, _ = pywt.cwt(sig, _SCALES, "morl", sampling_period=1 / fs)
    return np.abs(coefs) ** 2


# ── 2×1 PCA / t-SNE scatter plot ──────────────────────────────────────────────

def _plot_2x1(Z_pca, Z_tsne, y, dom, title, out_png):
    import matplotlib.cm as cm
    tags = np.unique(y)
    cmap = cm.get_cmap("tab10", len(tags))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)
    for ax, Z, name in ((axes[0], Z_pca, "PCA"), (axes[1], Z_tsne, "t-SNE")):
        for i, t in enumerate(tags):
            m_real = (y == t) & (dom == 0)
            m_fake = (y == t) & (dom == 1)
            ax.scatter(Z[m_real, 0], Z[m_real, 1], s=16, alpha=0.7,
                       label=f"tag {t} · real", color=cmap(i), marker="o", edgecolors="none")
            ax.scatter(Z[m_fake, 0], Z[m_fake, 1], s=18, alpha=0.7,
                       label=f"tag {t} · fake", color=cmap(i), marker="+")
        ax.set_title(f"{title} · {name}")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        handles, labels = ax.get_legend_handles_labels()
        by_tag = {}
        for h, lab in zip(handles, labels):
            t = lab.split("·")[0].strip()
            by_tag.setdefault(t, h)
        ax.legend(by_tag.values(), by_tag.keys(), loc="best", fontsize=9, frameon=True)
        ax.grid(True, linewidth=0.3, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── comparison plot (time or CWT) ─────────────────────────────────────────────

def plot_comparison(
    examples: dict[int, dict],
    *,
    channels: Sequence[int],
    mode: str = "time",
    save_dir: Path | None = None,
    idx_fake: int = 0,
    idx_reales: list[int] | None = None,
    zscore_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    for cls, data in examples.items():
        emg, fake, fs = data["emg"], data["fake"], data["fs"]
        n_real, n_fake = emg.shape[0], fake.shape[0]

        idx_fake   = 0 if idx_fake >= n_fake else idx_fake
        idx_reales = list(range(min(3, n_real))) if idx_reales is None else [
            i for i in idx_reales if i < n_real
        ]

        if not idx_reales:
            continue

        t_axis = np.arange(emg.shape[2]) / fs

        if mode == "time":
            rows, cols = len(idx_reales) + 1, len(channels)
            fig, axes  = plt.subplots(
                rows, cols,
                figsize=(cols * 3.6, rows * 2.2),
                sharex=True, sharey="col",
            )
            fig.suptitle(f"Class {cls} · Real vs. Fake (time)", y=1.02)

            for r, idx_r in enumerate(idx_reales):
                for c, ch in enumerate(channels):
                    ax  = axes[r, c]
                    sig = emg[idx_r, ch]
                    if zscore_fn:
                        sig = zscore_fn(sig)
                    ax.plot(t_axis, sig, lw=1, color="#1f77b4")
                    if r == 0:
                        ax.set_title(f"Ch {ch}")
                    if c == 0:
                        ax.set_ylabel(f"Real {idx_r}")

            for c, ch in enumerate(channels):
                ax  = axes[-1, c]
                sig = fake[idx_fake, ch]
                if zscore_fn:
                    sig = zscore_fn(sig)
                ax.plot(t_axis, sig, lw=1, color="#d62728")
                if c == 0:
                    ax.set_ylabel(f"Fake {idx_fake}")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            if save_dir:
                fig.savefig(
                    save_dir / f"time_cls{cls}_fake{idx_fake}.png",
                    dpi=300, bbox_inches="tight",
                )
            plt.close(fig)

        elif mode == "cwt":
            pw_all = [
                _cwt_power(emg[i, ch], fs)
                for i in idx_reales for ch in channels
            ]
            pw_all += [
                _cwt_power(fake[idx_fake, ch], fs)
                for ch in channels
            ]
            vmax = np.percentile(np.concatenate(pw_all), 99)

            rows, cols = len(idx_reales) + 1, len(channels)
            fig, axes  = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
            fig.suptitle(f"CWT Power · Class {cls}", y=0.995)

            for r, idx_r in enumerate(idx_reales):
                for c, ch in enumerate(channels):
                    ax = axes[r, c]
                    pw = _cwt_power(emg[idx_r, ch], fs)
                    ax.imshow(pw, cmap=_CMAP, origin="lower", vmax=vmax, aspect="auto")
                    if r == 0:
                        ax.set_title(f"Ch {ch}")
                    if c == 0:
                        ax.set_ylabel(f"Real {idx_r}")

            for c, ch in enumerate(channels):
                ax = axes[-1, c]
                pw = _cwt_power(fake[idx_fake, ch], fs)
                ax.imshow(pw, cmap=_CMAP, origin="lower", vmax=vmax, aspect="auto")
                if c == 0:
                    ax.set_ylabel(f"Fake {idx_fake}")

            plt.tight_layout()
            if save_dir:
                fig.savefig(
                    save_dir / f"cwt_cls{cls}_fake{idx_fake}.png",
                    dpi=300, bbox_inches="tight",
                )
            plt.close(fig)
