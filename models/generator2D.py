# models/generator2D.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks2D import SpatioTemporalGen

# models/generator2D.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks2D import SpatioTemporalGen

class Generator2D(nn.Module):
    """
    Generador condicional 2D:
     - Embedding de etiqueta y FC para semilla 2D (C0×T0)
     - Upsampling 2D espaciotemporal
     - Interpolación lineal a la longitud final T
    """
    def __init__(self,
                 C: int,
                 T: int,
                 dz: int,
                 de: int,
                 n_cls: int,
                 seed_dims: tuple[int,int] = (16, 64)):
        super().__init__()
        self.C, self.T = C, T
        self.seed_C, self.seed_T = seed_dims

        self.label   = nn.Embedding(n_cls, de)
        self.fc      = nn.Linear(dz + de, self.seed_C * self.seed_T)
        self.st_gen  = SpatioTemporalGen(
            seed_dims,
            kernel_size=(11,11),
            stride=(1,2),
            padding=(5,5),
            output_padding=(0,1)
        )
        self.out_act = nn.Tanh()

    def forward(self, z: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        # z: (B, dz), y: (B,)
        e = self.label(y)                         # → (B, de)
        h = torch.cat([z, e], dim=1)              # → (B, dz+de)
        x = self.fc(h).view(-1, self.seed_C, self.seed_T)  # → (B, C0, T0)

        x = self.st_gen(x)                        # → (B, C0, T1)
        x = F.interpolate(
            x.unsqueeze(1), size=(self.C, self.T),
            mode="bilinear", align_corners=False
        ).squeeze(1)                              # → (B, C, T)

        return self.out_act(x)
