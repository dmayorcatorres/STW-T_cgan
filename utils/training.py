import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import typing as ty
import random

import torch
import torch.nn.functional as F

def spectral_cosine_similarity(fake: torch.Tensor,
                               real: torch.Tensor,
                               eps: float = 1e-8) -> torch.Tensor:
    """
    Calcula la similitud coseno espectral media por batch.
    - fake, real: (B, C, T)
    Devuelve: escalar en [0,1], donde 1 = espectros idénticos.
    """
    # FFT de magnitud
    Ff = torch.fft.rfft(fake, dim=2).abs()    # (B, C, F)
    Fr = torch.fft.rfft(real, dim=2).abs()    # (B, C, F)

    # Aplanar canal+frecuencia
    Ff_flat = Ff.view(Ff.size(0), -1)         # (B, C*F)
    Fr_flat = Fr.view(Fr.size(0), -1)         # (B, C*F)

    # Normalizar
    Ff_norm = Ff_flat / (Ff_flat.norm(dim=1, keepdim=True) + eps)
    Fr_norm = Fr_flat / (Fr_flat.norm(dim=1, keepdim=True) + eps)

    # Cosine similarity por muestra y media batch
    cos = (Ff_norm * Fr_norm).sum(dim=1)       # (B,)
    return cos.mean()                         # escalar

def composite_loss(fake: torch.Tensor,
                   real: torch.Tensor,
                   alpha: float = 0.1) -> torch.Tensor:
    """
    Pérdida compuesta = MAE + alpha * (1 - SpectralCosine).
    alpha regula la importancia de la parte espectral.
    """
    # 1) MAE en el dominio temporal
    loss_time = F.l1_loss(fake, real)

    # 2) Similaridad coseno espectral
    #spec_cos = spectral_cosine_similarity(fake, real)
    #loss_spec = 1.0 - spec_cos

    return loss_time #+ alpha * loss_spec



def init_he(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_epoch(G, D, loader, optG, optD, scaler, device, dz,
                feedback_ratio=0.1, g_steps=1):
    """
    Entrena una epoch del GAN.

    Args:
        G, D: generador y discriminador.
        loader: DataLoader.
        optG, optD: optimizadores.
        scaler: GradScaler.
        device: str o torch.device ('cuda' o 'cpu').
        dz: dimensión del vector latente.
        feedback_ratio: proporción de muestras generadas inyectadas.
        g_steps: número de pasos del generador por batch.
    """
    for xr, y in loader:
        xr, y = xr.to(device), y.to(device)
        B = xr.size(0)

        N_feedback = int(B * feedback_ratio)
        z = torch.randn(B, dz, device=device)
        xf = G(z, y).detach()

        if N_feedback > 0:
            idx = torch.randperm(B)[:N_feedback]
            xr[idx] = xf[idx]

        optD.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda'):
            d_real = D(xr, y)
            d_fake = D(xf, y)
            d_loss = 0.5 * (
                F.mse_loss(d_real, torch.full_like(d_real, 0.9)) +
                F.mse_loss(d_fake, torch.full_like(d_fake, 0.1))
            )
        scaler.scale(d_loss).backward()
        scaler.step(optD)
        scaler.update()

        for _ in range(g_steps):
            z = torch.randn(B, dz, device=device)
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda'):
                gen = G(z, y)
                g_loss = composite_loss(gen, xr,0.5)
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()

    return d_loss.item(), g_loss.item()