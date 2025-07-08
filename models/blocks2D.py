# models/blocks2D.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGen(nn.Module):
    """
    Bloque 2D para upsampling espaciotemporal:
     - Entrada: (B, C0, T0)
     - Se trata como un “mapa” de tamaño C0×T0 con canal único
     - Dos etapas de ConvTranspose2d para duplicar la dimensión temporal
    """
    def __init__(self,
                 seed_dims: tuple[int,int],   # (C0, T0)
                 kernel_size=(11, 11),
                 stride=(1, 2),
                 padding=(5, 5),
                 output_padding=(0, 1)):
        super().__init__()
        self.C0, self.T0 = seed_dims

        self.deconv = nn.Sequential(
            # Stage 1: duplica el eje temporal (stride temporal=2)
            nn.ConvTranspose2d(
                in_channels=1, out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            # Stage 2: mantiene la misma resolución
            nn.ConvTranspose2d(
                in_channels=1, out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                output_padding=(0, 0)
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C0, T0)
        B, C0, T0 = x.shape
        z = x.unsqueeze(1).float()      # → (B,1,C0,T0)
        z = self.deconv(z)              # → (B,1,C0,T1)
        z = z.squeeze(1)                # → (B,C0,T1)
        return z.to(x.dtype)            # conserva dtype


class SpatioTemporalDisc(nn.Module):
    """
    Bloque 2D de discriminador con dos etapas de Conv2d:
     - Entrada:  (B, C, T)
     - Se trata como un “mapa” de tamaño C×T con canal único
     - Dos convoluciones 2D sucesivas (in_channels=1, out_channels=1)
     - Normalización por capa sobre la forma resultante (C, T_out)
    """
    def __init__(self,
                 C: int,
                 T: int,
                 # primera convolución
                 kernel_size1=(3, 11),
                 stride1=(1, 2),
                 padding1=(1, 5),
                 # segunda convolución
                 kernel_size2=(3, 11),
                 stride2=(1, 1),
                 padding2=(1, 5)):
        super().__init__()
        self.C, self.T = C, T

        # Etapa 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size1,
            stride=stride1,
            padding=padding1
        )
        self.bn1   = nn.BatchNorm2d(1)
        self.act1  = nn.LeakyReLU(0.2)

        # Etapa 2
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size2,
            stride=stride2,
            padding=padding2
        )
        self.bn2   = nn.BatchNorm2d(1)
        self.act2  = nn.LeakyReLU(0.2)

        # Calcular T_out tras ambas etapas:
        T1 = (T + 2*padding1[1] - kernel_size1[1]) // stride1[1] + 1
        T2 = (T1 + 2*padding2[1] - kernel_size2[1]) // stride2[1] + 1
        self.T_out = T2

        # LayerNorm sobre (C, T_out)
        self.ln = nn.LayerNorm((C, T2), elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = x.unsqueeze(1).float()     # → (B,1,C,T)
        z = self.act1(self.bn1(self.conv1(z)))  # → (B,1,C,T1)
        z = self.act2(self.bn2(self.conv2(z)))  # → (B,1,C,T2)
        z = z.squeeze(1)               # → (B,C,T2)
        z = self.ln(z)                 # → (B,C,T2)
        return z.to(x.dtype)


class Frequency(nn.Module):
    """
    Bloque frecuencial: FFT → log1p(abs) → iFFT → LayerNorm
      - Entrada: (B, C, T)
      - Salida:  (B, C, T)
    """
    def __init__(self, C: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.log1p(torch.fft.rfft(x.float(), dim=-1).abs())
        spec = torch.fft.irfft(spec, n=x.size(-1))  # (B,C,T)
        return F.layer_norm(spec, spec.shape[1:]).to(x.dtype)
