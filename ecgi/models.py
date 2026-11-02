"""ecgi.models — Generator, Discriminator, RegressionHead, and block registry."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgi.blocks import (
    TransformerDisc, WaveletExtractor,
    DilConvBlock, MultiScaleBlock, SpatialOnlyBlock,
    CWTPool, STFTPool,
)
from ecgi.fusion import FiLMFusion


# ── block registry ─────────────────────────────────────────────────────────────

BLOCK_REGISTRY = {
    "st":         TransformerDisc,
    "w":          WaveletExtractor,
    "dilconv":    DilConvBlock,
    "multiscale": MultiScaleBlock,
    "spatial":    SpatialOnlyBlock,
}


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    1-D generator with learnable per-channel amplitude.

    Parameters
    ----------
    C       : number of output channels
    T       : temporal length (samples)
    dz      : latent vector dimension
    de      : conditional embedding dimension
    n_cls   : number of classes for the embedding
    init_amp: initial amplitude (σ) per channel
    """
    def __init__(
        self,
        C: int,
        T: int,
        dz: int,
        de: int,
        n_cls: int,
        init_amp: float = 3.0,
    ):
        super().__init__()
        self.C, self.T = C, T

        # conditional embedding + FC
        self.label = nn.Embedding(n_cls, de)
        self.fc    = nn.Linear(dz + de, C * T)

        # temporal block (upsample + conv)
        self.temporal = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(C, C, kernel_size=5, padding=2),
            nn.BatchNorm1d(C),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(C, C, kernel_size=5, padding=2),
            nn.BatchNorm1d(C),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # learnable per-channel amplitude — shape (1,C,1) → broadcast over (B,C,T)
        self.amp = nn.Parameter(torch.full((1, C, 1), init_amp))

    def forward(self, z: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        """
        z: (B, dz)   latent vector
        y: (B,)      class labels
        """
        e = self.label(y)                         # (B, de)
        h = torch.cat([z, e], dim=1)              # (B, dz+de)

        x = self.fc(h).view(-1, self.C, self.T)   # (B, C, T)

        x = self.temporal(x)

        x = F.interpolate(x, size=self.T, mode="linear", align_corners=False)

        return torch.tanh(x) * self.amp           # (B, C, T)


# ── Discriminator ──────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(
        self,
        C: int,
        T: int,
        de: int,
        n_cls: int,
        *,
        use_st: bool = True,
        use_w: bool = True,
        block_st: str = "st",
        block_w: str = "w",
        fusion: nn.Module = None,
        tf_mode: str = "cwt",          # "cwt" | "stft" | "none"
        cwt_kwargs: dict = None,
        stft_kwargs: dict = None,
        disc_kwargs: dict = None,
    ):
        super().__init__()
        assert use_st or use_w, "At least one branch must be active"

        self.use_w   = use_w
        self.tf_mode = tf_mode

        # persistent TF pools (instantiated once)
        self.tf_pool = None
        if use_w and tf_mode == "cwt":
            self.tf_pool = CWTPool(C, T, **(cwt_kwargs or {}))
        elif use_w and tf_mode == "stft":
            self.tf_pool = STFTPool(C, T, **(stft_kwargs or {}))

        self.label = nn.Embedding(n_cls, de)

        disc_kwargs = disc_kwargs or {}
        self.branches = nn.ModuleDict()
        k = None

        # ST branch (space-time)
        if use_st:
            if block_st == "st":
                self.branches["st"] = BLOCK_REGISTRY["st"](C, T, **disc_kwargs)
            else:
                self.branches["st"] = BLOCK_REGISTRY[block_st](C, T)
            k = self.branches["st"].k_target

        # W branch (multiresolution / TF / conv banks)
        if use_w:
            if block_w == "w":
                self.branches["w"] = BLOCK_REGISTRY["w"](C, T)
            else:
                self.branches["w"] = BLOCK_REGISTRY[block_w](C, T)
            if k is None:
                k = self.branches["w"].k_target

        self.k = k

        self.fusion = fusion or FiLMFusion(C=C)

        self.head = nn.Sequential(
            nn.Linear(C * self.k + de, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y, debug=False, return_feats: bool = False):
        if debug: print("» Entrada cruda :", x.shape)

        x_tf = None
        if self.use_w and self.tf_pool is not None:
            x_tf = self.tf_pool(x)   # (B,C,T)
            if debug: print("» Entrada TF   :", x_tf.shape, "| modo:", self.tf_mode)

        feats = []
        for name, module in self.branches.items():
            if name == "w" and (x_tf is not None) and isinstance(module, WaveletExtractor):
                inp = x_tf                # CWT/STFT input
            else:
                inp = x                   # dilconv/multiscale or st branch

            z = module(inp)              # (B,C,k_i)
            if z.size(-1) != self.k:
                z = F.interpolate(z, size=self.k, mode="linear", align_corners=False)
            feats.append(z)
            if debug: print(f"✔ Latente {name}:", z.shape)

        merged = self.fusion(feats) if len(feats) > 1 else feats[0]
        if debug: print("★ Fusionado   :", merged.shape)

        flat   = merged.flatten(1)
        e_y    = self.label(y)
        logits = torch.cat([flat, e_y], 1)

        out = self.head(logits)
        if return_feats:
            return out, logits
        return out

    @torch.no_grad()
    def extract_latents(
        self,
        x: torch.Tensor,
        y: torch.LongTensor,
        *,
        detach: bool = True,
    ) -> dict:
        was_training = self.training
        self.eval()

        x_cwt = self._to_cwt(x) if self._use_cwt else None

        feats = {}
        vecs  = {}

        for name, module in self.branches.items():
            inp = x_cwt if name == "w" else x
            z = module(inp)                           # (B,C,k_i)
            if z.size(-1) != self.k:
                z = F.interpolate(z, size=self.k, mode="linear", align_corners=False)
            feats[name] = z
            vecs[f"flat_{name}"] = z.flatten(1)

        if len(feats) == 1:
            fused = next(iter(feats.values()))
        else:
            fused = self.fusion([feats.get("st"), feats.get("w")])  # (B,C,k)

        flat_fused = fused.flatten(1)
        e_y = self.label(y)

        out = {
            "st":          feats.get("st"),
            "w":           feats.get("w"),
            "fused":       fused,
            "flat_st":     vecs.get("flat_st"),
            "flat_w":      vecs.get("flat_w"),
            "flat_fused":  flat_fused,
            "label":       y,
            "emb":         e_y,
        }

        if detach:
            for k, v in list(out.items()):
                if torch.is_tensor(v):
                    out[k] = v.detach().cpu()

        if was_training:
            self.train()
        return out


# ── Regression head ────────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
    """
    Projects the vector [vec(F_fused) || e_y] → BSPM (E_t, T).
    """
    def __init__(self, in_dim: int, E_t: int, T: int):
        super().__init__()
        self.E_t, self.T = E_t, T
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, E_t * T),
        )

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        B     = h_in.size(0)
        y_hat = self.net(h_in)                  # (B, E_t*T)
        return y_hat.view(B, self.E_t, self.T)  # (B, E_t, T)
