import torch
import torch.nn.functional as F

def spectral_cosine_loss(fake: torch.Tensor, real: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 1) FFT y magnitudes
    fake_mag = torch.abs(torch.fft.rfft(fake, dim=-1))
    real_mag = torch.abs(torch.fft.rfft(real, dim=-1))
    # 2) Cosine similarity
    num = (fake_mag * real_mag).sum(dim=-1)
    den = torch.norm(fake_mag, p=2, dim=-1) * torch.norm(real_mag, p=2, dim=-1) + eps
    cos_sim = num / den
    return 1.0 - cos_sim.mean()

def ssim1d_loss(fake: torch.Tensor, real: torch.Tensor,
                window_size: int = 51, sigma: float = 7.0,
                c1: float = 1e-4, c2: float = 9e-4) -> torch.Tensor:
    B, C, T = fake.shape
    # Genera ventana gaussiana
    coords = torch.arange(window_size, device=fake.device) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, window_size)
    pad = window_size // 2

    # Convoluciones para medias y varianzas locales
    def conv(x):
        x = x.view(-1, 1, T)
        return F.conv1d(x, g, padding=pad).view(B, C, T)

    mu_f = conv(fake)
    mu_r = conv(real)
    mu_f2 = mu_f**2
    mu_r2 = mu_r**2
    sigma_f2 = conv(fake*fake) - mu_f2
    sigma_r2 = conv(real*real) - mu_r2
    sigma_fr = conv(fake*real) - mu_f*mu_r

    num = (2*mu_f*mu_r + c1) * (2*sigma_fr + c2)
    den = (mu_f2 + mu_r2 + c1) * (sigma_f2 + sigma_r2 + c2)
    ssim_map = num / (den + 1e-12)
    return 1.0 - ssim_map.mean()

def stft_magnitude_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = None
) -> torch.Tensor:
    """
    Computes MSE between magnitude-STFTs of `fake` and `real`.
    Accepts inputs of shape (B, C, T).
    """
    B, C, T = fake.shape
    hop = hop_length or n_fft // 2

    # Prepare a Hann window on the correct device
    window = torch.hann_window(n_fft, device=fake.device)

    # Flatten batch & channel dims → (B*C, T)
    fake_flat = fake.reshape(-1, T)
    real_flat = real.reshape(-1, T)

    # STFT→ complex spectrograms, then magnitude
    S_fake = torch.stft(
        fake_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True
    ).abs()
    S_real = torch.stft(
        real_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True
    ).abs()

    # MSE over all time–freq bins
    return F.mse_loss(S_fake, S_real)

def composite_loss(fake: torch.Tensor, real: torch.Tensor,
                   alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    """
    L = L_time + alpha·L_spec + beta·L_ssim + gamma·L_stft
    """
    L_spec = spectral_cosine_loss(fake, real)
    #L_ssim = ssim1d_loss(fake, real)
    #L_stft = stft_magnitude_loss(fake, real)
    L_time = F.l1_loss(fake, real)
    return alpha*L_time + beta*L_spec #+ beta*L_ssim + gamma*L_stft
