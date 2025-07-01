# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import TemporalGen

class Generator(nn.Module):
    def __init__(self, C: int, T: int, dz: int, de: int, n_cls: int):
        """
        C     Número de canales de salida
        T     Longitud temporal de salida
        dz    Dimensión del vector latente
        de    Dimensión del embedding condicional
        n_cls Número de clases para el embedding
        """
        super().__init__()
        self.C, self.T = C, T
        self.label     = nn.Embedding(n_cls, de)
        self.fc        = nn.Linear(dz + de, C * T)
        self.temporal  = TemporalGen(C)
        self.out_act   = nn.Tanh()

    def forward(self, z: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        # z: (B, dz), y: (B,)
        e_l = self.label(y)                         # (B, de)
        h   = torch.cat([z, e_l], dim=1)            # (B, dz+de)
        x   = self.fc(h).view(-1, self.C, self.T)   # (B, C, T)
        x   = self.temporal(x)                      # (B, C, T′)
        x   = F.interpolate(x, size=self.T, mode="linear", align_corners=False)
        return self.out_act(x)                      # (B, C, T)
