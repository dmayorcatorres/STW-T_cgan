import torch
import torch.nn as nn

# ╭─────────────────── Model Building Blocks ──────────────╮

class Spatial(nn.Module):
    """
    Bloque espacial que aplica convolución punto a punto (kernel=1) y normalización por capa.
    """
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 1)
        self.ln = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, x):
        z = self.conv(x.float())  # (B, C, T)
        return self.ln(z.transpose(1, 2)).transpose(1, 2).to(x.dtype)


class TemporalGen(nn.Module):
    """
    Rama temporal del Generador: 4 capas de deconvolución para upsampling progresivo.
    """
    def __init__(self, c):
        super().__init__()
        
        self.deconv = nn.Sequential(

            nn.ConvTranspose1d(c, c, 16, stride=2, padding=1),
            nn.BatchNorm1d(c),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(c, c, 32, stride=2, padding=1),
            nn.BatchNorm1d(c),
            nn.LeakyReLU(0.2),
        )

        self.ln = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, x):
        z = self.deconv(x.float())
        z = z.transpose(1, 2)
        z = self.ln(z)
        z = z.transpose(1, 2)
        return z.to(x.dtype)



class TemporalDisc(nn.Module):
    """
    Rama temporal del Discriminador: 4 capas de convolución para extracción jerárquica.
    """
    def __init__(self, c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(c, c, 32, stride=2, padding=1),  # reduce longitud a la mitad
            nn.BatchNorm1d(c),
            nn.LeakyReLU(0.2),

            nn.Conv1d(c, c, 16, stride=2, padding=1),
            nn.BatchNorm1d(c),
            nn.LeakyReLU(0.2),

        )

        self.ln = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, x):
        z = self.conv(x.float())
        z = z.transpose(1, 2)
        z = self.ln(z)
        z = z.transpose(1, 2)
        return z.to(x.dtype)



class Frequency(nn.Module):
    """
    Bloque frecuencial: pasa a dominio de frecuencias con FFT, toma log-magnitud,
    y regresa al dominio temporal para procesarlo con normalización por capa.
    """
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, x):
        spec = torch.log1p(torch.fft.rfft(x.float(), dim=-1).abs())
        spec = torch.fft.irfft(spec, n=x.size(-1))  # Recupera al dominio temporal
        return self.ln(spec.transpose(1, 2)).transpose(1, 2).to(x.dtype)
