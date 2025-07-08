import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────── Bloques disponibles ───────────────────
from .blocks import TransformerTemporalDisc

BLOCK_REGISTRY = {
    #"basic_temporal":   lambda C, T, k=None: TemporalDisc(C, T),
    #"basic_spatial":    lambda C, T, k:      SpatialDisc(C, T, k),
    #"inception_temporal": lambda C, T, k=None: InceptionTemporalDisc(C, T),
    #"inception_se_temporal": lambda C, T, k=None: InceptionSETemporalDisc(C, T),  # ← nuevo
    "transformer_temporal": TransformerTemporalDisc,      # ← nuevo
    #"resnet_temporal":  lambda C, T, k=None: ResNet1DTemporalDisc(C, T),  # ResNet1DTemporalDisc no está definido aquí
    #"tcn_temporal":     lambda C, T, k=None: TCNTemporalDisc(C, T),  # Temporal Convolutional Network
    #"dilated_temporal": lambda C, T, k=None: DilatedCNNTemporalDisc(C, T),  # ← nuevo
    #"hybrid_temporal":  lambda C, T, k=None: HybridTemporalDisc(C, T),  # ← nuevo
    #"spatial_axial":    lambda C, T, k=None: SpatialAxialDisc(C, T, k),  # ← nuevo
    #"patchtst_temporal": lambda C, T, k=None: PatchTSTTemporalDisc(C, T),  # ← nuevo
}



# ─────────────────────────  Discriminador genérico  ────────────────────────
class Discriminator(nn.Module):
    def __init__(self, C, T, de, n_cls,
                 use_t=True, use_s=False, use_f=False,
                 block_t="tcn_temporal",
                 block_s="basic_spatial"):
        super().__init__()

        self.label = nn.Embedding(n_cls, de)

        # Rama temporal (elige el bloque y toma su k_target)
        if use_t:
            self.t = BLOCK_REGISTRY[block_t](C, T)
            self.k = self.t.k_target
        else:
            self.k = (T - 4) // 4 + 1      # valor por defecto

        # Rama espacial (usa self.k para igualar tamaño)
        if use_s:
            self.s = BLOCK_REGISTRY[block_s](C, T, self.k)

        self.use_t, self.use_s, self.use_f = use_t, use_s, use_f
        self.f = Frequency(C) if use_f else None

        self.head = nn.Sequential(
            nn.Linear(C * self.k + de, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y, debug=False):
        branches = []
        if self.use_t:
            z = self.t(x); branches.append(z)
            if debug: print("✔ Temporal ", z.shape)
        if self.use_s:
            z = self.s(x); branches.append(z)
            if debug: print("✔ Spatial  ", z.shape)
        if self.use_f:
            z = self.f(x); branches.append(z)
            if debug: print("✔ Frequency", z.shape)

        merged = sum(branches) / len(branches)         # (B,C,k)
        flat   = merged.reshape(x.size(0), -1)         # (B,C·k)

        return self.head(torch.cat([flat, self.label(y)], 1))
