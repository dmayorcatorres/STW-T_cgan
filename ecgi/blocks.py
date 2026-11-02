"""ecgi.blocks — all 1-D signal processing building blocks for the discriminator.

Opt-1: TransformerDisc.forward uses @torch.amp.autocast("cuda", enabled=False)
       instead of the deprecated @torch.cuda.amp.autocast(False).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgi.config import n_scales, f_min, f_max, fs as _FS


# ── STFT pooling ───────────────────────────────────────────────────────────────

class STFTPool(nn.Module):
    """
    STFT -> magnitude -> soft compression -> freq-average -> rescale to T -> LayerNorm
    Input : (B,C,T)
    Output: (B,C,T)
    """
    def __init__(
        self,
        C: int,
        T: int,
        fs: int = 2048,
        n_fft: int = 256,
        win_length: int = 256,
        hop_length: int = 64,
        power: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.C, self.T = C, T
        self.n_fft      = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power      = power

        win = torch.hann_window(win_length, periodic=True, device=device or torch.device("cpu"))
        self.register_buffer("window", win, persistent=False)

        self.ln = nn.LayerNorm(C, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,T)
        B, C, T = x.shape
        assert C == self.C, "C mismatch en STFTPool"

        x32 = x.float().reshape(B * C, T)
        X = torch.stft(
            x32,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # (B*C, F, frames)

        mag = X.abs().pow(self.power)            # soft compression
        mag = mag.mean(dim=1)                    # freq-average -> (B*C, frames)
        mag = mag.view(B, C, -1)                 # (B,C,frames)

        mag = F.interpolate(mag, size=T, mode="linear", align_corners=False)

        if self.ln.weight.device != x.device:
            self.ln = self.ln.to(device=x.device, dtype=x.dtype)

        return self.ln(mag.transpose(1, 2)).transpose(1, 2).to(dtype=x.dtype)


# ── depthwise conv helpers ─────────────────────────────────────────────────────

class _DWConvSame(nn.Module):
    """Depthwise Conv1d with 'same' padding, even for even-sized kernels."""
    def __init__(self, C: int, k: int, dilation: int = 1):
        super().__init__()
        self.k = k
        self.d = dilation
        self.conv = nn.Conv1d(C, C, k, padding=0, dilation=dilation, groups=C, bias=False)

    def forward(self, x):
        k_eff = self.d * (self.k - 1) + 1
        left  = (k_eff - 1) // 2
        right = (k_eff - 1) - left
        x = F.pad(x, (left, right))
        return self.conv(x)


class DilConvBlock(nn.Module):
    """
    Bank of dilated convs (k={3,5,7}, d={1,2,4}) -> proj -> downsample -> (B,C,k)
    """
    def __init__(self, C: int, T: int, ks=(3, 5, 7), ds=(1, 2, 4)):
        super().__init__()
        self.bank = nn.ModuleList([_DWConvSame(C, k, d) for k in ks for d in ds])
        self.proj = nn.Conv1d(len(self.bank) * C, C, 1, bias=False)

        self.down = nn.Conv1d(C, C, 8, stride=4, padding=2, groups=C, bias=False)
        nn.init.dirac_(self.down.weight)

        self.post = nn.Sequential(
            nn.GroupNorm(1, C),
            nn.GELU(),
            nn.Conv1d(C, C, 1, bias=False),
        )
        self.k_target = math.ceil(T / 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([b(x) for b in self.bank], dim=1)
        z = self.proj(z)
        z = self.down(z)
        self.k_target = z.size(-1)
        return self.post(z)


class MultiScaleBlock(nn.Module):
    """
    Parallel large-window convs (k={10,20,40}) -> proj -> downsample -> (B,C,k)
    """
    def __init__(self, C: int, T: int, ks=(10, 20, 40)):
        super().__init__()
        self.bank = nn.ModuleList([_DWConvSame(C, k, 1) for k in ks])
        self.proj = nn.Conv1d(len(self.bank) * C, C, 1, bias=False)

        self.down = nn.Conv1d(C, C, 8, stride=4, padding=2, groups=C, bias=False)
        nn.init.dirac_(self.down.weight)

        self.post = nn.Sequential(
            nn.GroupNorm(1, C),
            nn.GELU(),
            nn.Conv1d(C, C, 1, bias=False),
        )
        self.k_target = math.ceil(T / 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([b(x) for b in self.bank], dim=1)
        z = self.proj(z)
        z = self.down(z)
        self.k_target = z.size(-1)
        return self.post(z)


class SpatialOnlyBlock(nn.Module):
    """
    No temporal modelling: summarises per-channel stats and projects to (B,C,k).
    Baseline variant with no temporal branch.
    """
    def __init__(self, C: int, T: int, hidden: int = 512, **_):
        super().__init__()
        self.k_target = math.ceil(T / 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * C, hidden),
            nn.GELU(),
            nn.Linear(hidden, C * self.k_target),
        )
        self.ln = nn.LayerNorm(C, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,T)
        mu = x.mean(dim=-1)
        sd = x.std(dim=-1)
        h  = torch.cat([mu, sd], dim=1)          # (B,2C)
        z  = self.mlp(h).view(x.size(0), x.size(1), self.k_target)  # (B,C,k)
        return self.ln(z.transpose(1, 2)).transpose(1, 2)


# ── transformer discriminator building blocks ──────────────────────────────────

class DWConvNorm(nn.Module):
    """Depth-wise Conv1d → {GroupNorm(1) | LayerNorm} → GELU."""
    def __init__(self, C: int, dilation: int, *, use_ln: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(C, C, 3, padding=dilation,
                              dilation=dilation, groups=C, bias=False)
        self.norm = nn.LayerNorm(C) if use_ln else nn.GroupNorm(1, C)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,C,T)
        x = self.conv(x)
        if isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # LN expects C at -1
        else:
            x = self.norm(x)
        return self.act(x)


class Rotary1D(nn.Module):
    """Rotary positional embedding 1-D (no einops dependency)."""
    def __init__(self, dim: int):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,k,d)
        t     = torch.arange(x.size(1), device=x.device, dtype=x.dtype)
        freqs = torch.einsum("k,d->kd", t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        x_rot = torch.stack((-x[..., 1::2], x[..., ::2]), -1).reshape_as(x)
        return x * emb.cos() + x_rot * emb.sin()


class TransformerDisc(nn.Module):
    """
    Axial 1-D Discriminator
      Input : (B, C, T)
      Output: (B, C, k)  with  k ≈ ceil(T/4)
    """
    def __init__(self, C: int, T: int, *,
                 d_model: int = 128,
                 heads_t: int = 4,
                 heads_c: int = 4,
                 use_rotary: bool = True,
                 use_ln_dw: bool = True):
        super().__init__()
        assert d_model % heads_t == d_model % heads_c == 0

        # 1) Dilated convolutions
        dil = (1, 2, 4, 8, 16, 32)
        self.dilated = nn.ModuleList(
            DWConvNorm(C, d, use_ln=use_ln_dw) for d in dil
        )
        self.proj_dil  = nn.Conv1d(len(dil) * C, C, 1, bias=False)
        self.res_scale = nn.Parameter(torch.zeros(()))

        # 2) Stride-4 downsample
        self.down = nn.Conv1d(C, C, 8, stride=4, padding=2,
                              groups=C, bias=False)
        nn.init.dirac_(self.down.weight)

        # 3) Tokens + LN
        self.proj_in   = nn.Conv1d(C, d_model, 1, bias=False)
        self.ln_tokens = nn.LayerNorm(d_model)

        # 4) Axial transformer
        enc_t = nn.TransformerEncoderLayer(
            d_model, heads_t, d_model * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        enc_c = nn.TransformerEncoderLayer(
            d_model, heads_c, d_model * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.t_enc  = nn.TransformerEncoder(enc_t, 2)
        self.c_enc  = nn.TransformerEncoder(enc_c, 1)
        self.rotary = Rotary1D(d_model) if use_rotary else nn.Identity()

        # 5) Output
        self.proj_out = nn.Conv1d(d_model, C, 1, bias=False)
        self.ln_out   = nn.LayerNorm(C)

        self.k_target = math.ceil(T / 4)

    # Opt-1: use new autocast API (replaces deprecated @torch.cuda.amp.autocast(False))
    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B,C,T)
        # a) dilated convolutions + residual
        x_dil = torch.cat([blk(x) for blk in self.dilated], dim=1)
        x = self.proj_dil(x_dil) + torch.sin(self.res_scale) * x

        # b) stride-4 downsample
        z = self.down(x)                                    # (B,C,T/4)
        self.k_target = z.size(-1)

        # c) project to tokens
        z = self.proj_in(z)                                 # (B,d,k)
        z = self.ln_tokens(z.permute(0, 2, 1))             # (B,k,d)

        # d) axial attention
        z = self.rotary(z)
        z = self.t_enc(z)
        z = self.c_enc(z)

        # e) back to (B,C,k)
        z = self.proj_out(z.permute(0, 2, 1))
        z = self.ln_out(z.transpose(1, 2)).transpose(1, 2)
        return z.to(dtype=x.dtype)


class WaveletExtractor(nn.Module):
    """Reuses TransformerDisc internally (no explicit CWT)."""
    def __init__(self, C, T):
        super().__init__()
        self.core = TransformerDisc(C, T)
        self.k_target = self.core.k_target

    def forward(self, x):
        return self.core(x)


# ── CWT kernels and pooling ────────────────────────────────────────────────────

def _build_cwt_kernels(C: int, T: int, fs: int, f_min: float, f_max: float,
                       n_scales: int, wavelet_bw: float, device, dtype):
    dt    = 1.0 / fs
    t     = torch.linspace(-T // 2 * dt, T // 2 * dt, T, device=device, dtype=dtype)
    freqs = torch.logspace(math.log10(f_min), math.log10(f_max),
                           steps=n_scales, device=device, dtype=dtype)
    kernels = []
    for f in freqs:
        sigma  = wavelet_bw / (2 * math.pi * f)
        gauss  = torch.exp(-t ** 2 / (2 * sigma ** 2))
        kernel = gauss * torch.cos(2 * math.pi * f * t)
        kernels.append(kernel / kernel.norm(p=2))
    return torch.stack(kernels).unsqueeze(1).repeat(C, 1, 1)  # (C*S,1,T)


class CWTPool(nn.Module):
    """CWT + fixed pooling with root compression (|x|^power)."""

    def __init__(self, C: int, T: int, fs: int = 2048,
                 f_min: float = 1, f_max: float = 100,
                 n_scales: int = 64, wavelet_bw: float = 1.5,
                 power: float = 0.5, device: Optional[torch.device] = None):
        super().__init__()
        self.C, self.T, self.power = C, T, power
        kernel = _build_cwt_kernels(C, T, fs, f_min, f_max, n_scales, wavelet_bw,
                                    device or torch.device("cpu"), torch.float32)
        self.register_buffer("kernel", kernel, persistent=False)
        self.ln = nn.LayerNorm(C, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,T)
        B, C, _ = x.shape
        k    = self.kernel.to(dtype=x.dtype, device=x.device)
        spec = F.conv1d(x.float(), k, padding=x.size(-1) // 2, groups=C)  # (B,C*S,T)
        spec = torch.pow(spec.clamp(min=0).abs(), self.power)             # soft compression
        L    = spec.size(-1)
        S    = k.size(0) // C
        spec = spec.view(B, C, S, L).mean(dim=2)  # (B,C,L)
        if self.ln.weight.device != x.device:
            self.ln = self.ln.to(device=x.device, dtype=x.dtype)
        return self.ln(spec.transpose(1, 2)).transpose(1, 2).to(x.dtype)


def tensor_to_cwt(x: torch.Tensor, *, fs: int = 2048, f_min: float = 1,
                  f_max: float = 100, n_scales: int = 64,
                  wavelet_bw: float = 1.5, power: float = 0.5):
    """Functional helper: instantiates CWTPool on-the-fly for quick use."""
    B, C, T = x.shape
    pool = CWTPool(C, T, fs, f_min, f_max, n_scales, wavelet_bw,
                   power, device=x.device).to(x.dtype)
    with torch.no_grad():
        return pool(x)
