"""ecgi.fusion — feature-fusion modules for the discriminator.

No ecgi dependencies (pure nn.Modules).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFusion(nn.Module):
    """
    Simple mean fusion:
      z_st (B, C, k) + z_w (B, C, k) -> (B, C, k)
    with optional residual connection and LayerNorm.
    """
    def __init__(self, C: int, res_w: float = 0.1):
        super().__init__()
        self.C     = C
        self.res_w = res_w
        self.proj  = nn.Conv1d(C, C, kernel_size=1, bias=False)
        self.ln    = nn.LayerNorm(C, elementwise_affine=True)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        z_st, z_w = feats

        pst = self.proj(z_st)  # (B, C, k)
        pw  = self.proj(z_w)   # (B, C, k)

        fused = 0.5 * (pst + pw)
        fused = fused + self.res_w * (z_st + z_w)

        fused = fused.transpose(1, 2)  # -> (B, k, C)
        fused = self.ln(fused)
        return fused.transpose(1, 2)   # -> (B, C, k)


class FiLMFusion(nn.Module):
    """
    FiLM-Fusion:
      z_st (space-time domain)    → generates γ, β
      z_w  (time-frequency domain) → modulated by γ, β
    Then concatenated and projected to C channels.

    Parameters
    ----------
    C        : original channels
    d_hidden : intermediate dimension (embedding)
    reduction: compression ratio for γ, β  (e.g. 4 → C/4)
    res_w    : direct residual weight (0-1)
    """

    def __init__(
        self,
        C: int,
        d_hidden: int = 128,
        reduction: int = 4,
        res_w: float = 0.1,
    ):
        super().__init__()
        self.C     = C
        self.res_w = res_w

        # projection C → d_hidden
        self.st_in = nn.Conv1d(C, d_hidden, 1, bias=False)
        self.w_in  = nn.Conv1d(C, d_hidden, 1, bias=False)

        # FiLM (γ, β) generated from z_st
        filmlat = max(8, d_hidden // reduction)
        self.film = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                      # (B,d,1)
            nn.Conv1d(d_hidden, filmlat, 1), nn.GELU(),   # (B,filmlat,1)
            nn.Conv1d(filmlat, 2 * d_hidden, 1),          # γ‖β  (B,2d,1)
        )

        # mix and output
        self.mix = nn.Conv1d(2 * d_hidden, C, 1, bias=False)
        self.ln  = nn.LayerNorm(C, elementwise_affine=True)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """feats = [z_st, z_w] each (B,C,k)"""
        z_st, z_w = feats
        pst = self.st_in(z_st)           # (B,d,k)
        pw  = self.w_in(z_w)             # (B,d,k)

        film_params = self.film(pst).squeeze(-1)        # (B,2d)
        γ, β = film_params.chunk(2, dim=1)              # (B,d) each
        γ = γ.unsqueeze(-1); β = β.unsqueeze(-1)        # (B,d,1)

        pw_mod = (1 + γ) * pw + β                       # (B,d,k)

        fused_d = torch.cat([pst, pw_mod], dim=1)       # (B,2d,k)
        fused_c = self.mix(fused_d)                     # (B,C,k)

        fused = fused_c + self.res_w * (z_st + z_w)
        return self.ln(fused.transpose(1, 2)).transpose(1, 2)


class CrossAttentionFusion(nn.Module):
    def __init__(self, C: int, d_k: int = 64, heads: int = 4, res_w: float = 0.1):
        super().__init__()
        self.res_w = res_w
        self.heads = heads
        self.d_k   = d_k

        self.to_q = nn.Conv1d(C, heads * d_k, 1, bias=False)
        self.to_k = nn.Conv1d(C, heads * d_k, 1, bias=False)
        self.to_v = nn.Conv1d(C, heads * d_k, 1, bias=False)

        self.out = nn.Conv1d(heads * d_k, C, 1, bias=False)
        self.ln  = nn.LayerNorm(C, elementwise_affine=True)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        z_st, z_w = feats           # each (B, C, k)
        B, C, k = z_st.shape

        Q = self.to_q(z_st)         # (B, h*dk, k)
        K = self.to_k(z_w)
        V = self.to_v(z_w)

        Q = Q.view(B, self.heads, self.d_k, k)
        K = K.view(B, self.heads, self.d_k, k)
        V = V.view(B, self.heads, self.d_k, k)

        scores = torch.einsum("bhdk,bhdk->bhk", Q, K) / (self.d_k ** 0.5)  # (B,h,k)
        attn   = F.softmax(scores, dim=-1).unsqueeze(2)                  # (B,h,1,k)
        out_h  = (attn * V).view(B, -1, k)                               # (B, h*dk, k)

        fused_c = self.out(out_h)                                        # (B,C,k)
        fused   = fused_c + self.res_w * (z_st + z_w)
        fused   = fused.transpose(1, 2)  # (B,k,C)
        fused   = self.ln(fused)
        return fused.transpose(1, 2)     # (B,C,k)


class GatedFusion(nn.Module):
    def __init__(self, C: int, hidden: int = 128):
        super().__init__()
        self.proj_st   = nn.Conv1d(C, hidden, 1, bias=False)
        self.proj_w    = nn.Conv1d(C, hidden, 1, bias=False)
        self.gate_conv = nn.Conv1d(hidden * 2, C, 1)
        self.out_conv  = nn.Conv1d(hidden * 2, C, 1)
        self.ln        = nn.LayerNorm(C, elementwise_affine=True)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        z_st, z_w = feats
        pst = F.relu(self.proj_st(z_st))   # (B, h, k)
        pw  = F.relu(self.proj_w(z_w))     # (B, h, k)
        cat  = torch.cat([pst, pw], dim=1)  # (B, 2h, k)
        gate = torch.sigmoid(self.gate_conv(cat))  # (B, C, k)
        fused = gate * z_st + (1 - gate) * z_w     # adaptive mix
        out   = self.out_conv(torch.cat([pst, pw], dim=1))  # (B,C,k)
        fused = out + fused
        fused = fused.transpose(1, 2)
        fused = self.ln(fused)
        return fused.transpose(1, 2)
