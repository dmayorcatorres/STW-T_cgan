# models/discriminator2D.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks2D import SpatioTemporalDisc, Frequency

class Discriminator2D(nn.Module):
    """
    Discriminador 2D que combina una rama espaciotemporal (dos conv2d)
    y opcionalmente una rama frecuencial.
    """
    def __init__(self,
                 C: int,
                 T: int,
                 de: int,
                 n_cls: int,
                 use_st: bool = True,
                 use_f: bool = False):
        super().__init__()
        self.C, self.T = C, T
        self.label = nn.Embedding(n_cls, de)

        self.use_st = use_st
        self.use_f  = use_f
        if use_st:
            # Por defecto usa kernel_size1=(3,11), stride1=(1,2), padding1=(1,5), etc.
            self.st = SpatioTemporalDisc(C, T)
        if use_f:
            self.f  = Frequency(C)

        # La cabeza final espera C*T + de features
        self.head = nn.Sequential(
            nn.Linear(C * T + de, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, C, T)
        y: (B,)
        """
        branches = []
        if self.use_st:
            st = self.st(x)                # → (B, C, T_st_out)
            # Reescalar temporalmente si T_st_out != T
            if st.size(2) != self.T:
                st = F.interpolate(st, size=self.T, mode="linear", align_corners=False)
            branches.append(st)
        if self.use_f:
            f = self.f(x)                  # → (B, C, T)
            branches.append(f)

        # Fusión por media simple
        merged = torch.stack(branches, 0).mean(0)  # → (B, C, T)

        flat = merged.view(x.size(0), -1)          # → (B, C*T)
        cat  = torch.cat([flat, self.label(y)], 1) # → (B, C*T + de)
        return self.head(cat)