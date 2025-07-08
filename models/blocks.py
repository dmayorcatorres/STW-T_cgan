import torch
import torch.nn as nn
import torch.nn.functional as F



class TransformerTemporalDisc(nn.Module):
    def __init__(self, C, T, d_model=128, heads_t=4, heads_c=4):
        super().__init__()
        # ─ downsample temporal ─
        self.down   = nn.Conv1d(C, C, 4, stride=4, groups=C)
        k = (T-4)//4 + 1
        # ─ atención temporal ─
        self.proj_in = nn.Conv1d(C, d_model, 1)
        self.t_enc   = nn.TransformerEncoder(
                           nn.TransformerEncoderLayer(d_model, heads_t,
                                                      d_model*4,
                                                      batch_first=True,
                                                      activation='gelu',
                                                      norm_first=True), 2)

        # ─ atención canal ─
        assert d_model % heads_c == 0
        self.c_enc = nn.TransformerEncoder(
                         nn.TransformerEncoderLayer(d_model, heads_c,
                                                    d_model*4,
                                                    batch_first=True,
                                                    activation='gelu',
                                                    norm_first=True), 1)

        self.proj_out = nn.Conv1d(d_model, C, 1)
        self.ln = nn.LayerNorm(C, elementwise_affine=False)
        self.k_target = k

    def forward(self, x):                     # (B,C,T)
        z = self.down(x.float())              # (B,C,k)
        z = self.proj_in(z)                   # (B,d,k)

        # atención temporal (tokens = k)
        z = self.t_enc(z.permute(0,2,1)).permute(0,2,1)   # (B,d,k)

        # atención canal   (tokens = C)
        z = self.c_enc(z.permute(0,2,1)).permute(0,2,1)   # (B,d,k)

        z = self.proj_out(z)                # (B,C,k)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)


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
