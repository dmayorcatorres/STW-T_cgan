
#BLOCK_REGISTRY = {
    #"basic_temporal":   lambda C, T, k=None: TemporalDisc(C, T),
    #"basic_spatial":    lambda C, T, k:      SpatialDisc(C, T, k),
    #"inception_temporal": lambda C, T, k=None: InceptionTemporalDisc(C, T),
    #"inception_se_temporal": lambda C, T, k=None: InceptionSETemporalDisc(C, T),  # ← nuevo
    #"transformer_temporal": TransformerTemporalDisc,      # ← nuevo
    #"resnet_temporal":  lambda C, T, k=None: ResNet1DTemporalDisc(C, T),  # ResNet1DTemporalDisc no está definido aquí
    #"tcn_temporal":     lambda C, T, k=None: TCNTemporalDisc(C, T),  # Temporal Convolutional Network
    #"dilated_temporal": lambda C, T, k=None: DilatedCNNTemporalDisc(C, T),  # ← nuevo
    #"hybrid_temporal":  lambda C, T, k=None: HybridTemporalDisc(C, T),  # ← nuevo
    #"spatial_axial":    lambda C, T, k=None: SpatialAxialDisc(C, T, k),  # ← nuevo
    #"patchtst_temporal": lambda C, T, k=None: PatchTSTTemporalDisc(C, T),  # ← nuevo
#}




# ╭────────────────────────  DilatedConv-Temporal Disc  ─────────────────────╮
#  • Pila de convoluciones 1-D con dilataciones 1, 2, 4, 8 …
#  • Dos capas usan stride 2  ⇒  T → ⌊T/2⌋ → ⌊T/4⌋  (misma reducción que otras ramas)
#  • Conexión residual + BatchNorm para estabilidad (estilo TCN)
#  • Salida: (B, C, k_target)  donde  k_target = ⌊(T-1)/4⌋+1
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── PatchTST-Temporal Disc ─────────────────────────╮
#  Adaptación 1-D del modelo PatchTST (Nie & Zhang 2023) para ECG/EEG
#  ────────────────────────────────────────────────────────────────────────────
#  1. Divide cada señal (C,T) en parches solapados de longitud P y stride S.
#  2. Proyecta cada parche a d_model con una conv1d 1×1.
#  3. Añade embedding de posición relativo al patch.
#  4. Pasa por TransformerEncoder L× (MHSA+FFN).
#  5. Proyecta de vuelta a C canales y devuelve (B,C,k_patch).
#  Devuelve misma forma (B,C,k) que las otras ramas → se fusiona sin cambios.
# ╰───────────────────────────────────────────────────────────────────────────╯
import math, torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Conv1d deslizante para extraer parches de longitud P con stride S."""
    def __init__(self, C, P, S, d_model):
        super().__init__()
        self.P, self.S = P, S
        self.proj = nn.Conv1d(C, d_model, kernel_size=P, stride=S)

    def forward(self, x):                                # (B,C,T)
        z = self.proj(x.float())                         # (B,d_model,n_patch)
        z = z.permute(0, 2, 1)                           # (B,n_patch,d_model)
        return z                                         # tokens

class PatchTSTTemporalDisc(nn.Module):
    """
    Entrada : (B,C,T)    donde T ≫ P
    Salida  : (B,C,k)    con k = n_patch = ⌊(T-P)/S⌋+1
    """
    def __init__(self, C:int, T:int,
                 P:int=16, stride:int=8,
                 d_model:int=128, n_heads:int=4,
                 n_layers:int=2, ff_mult:int=4):

        super().__init__()
        self.embed   = PatchEmbedding(C, P, stride, d_model)
        n_patch      = (T - P) // stride + 1
        self.k_target = n_patch

        # learnable positional embedding (n_patch,d_model)
        self.pos     = nn.Parameter(torch.randn(1, n_patch, d_model) / math.sqrt(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*ff_mult,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        self.proj_out = nn.Linear(d_model, C)
        self.ln       = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                    # (B,C,T)
        z = self.embed(x)                    # (B,n_patch,d_model)
        z = z + self.pos                     # pos-embed
        z = self.encoder(z)                  # (B,n_patch,d_model)
        z = self.proj_out(z)                 # (B,n_patch,C)
        z = z.permute(0, 2, 1)               # (B,C,k)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)                 # (B,C,k_target)


# ╭─────────────────── Temporal TCN-Residual Disc  ───────────────────╮
#  • Dos bloques residuales dilatados (à-la WaveNet/TCN).
#  • Cada bloque = depthwise-conv 3×1, BN, ReLU + atajo 1×1.
#  • Cada bloque usa stride 2  →  T → ⌊T/2⌋ → ⌊T/4⌋  ⇒  k_target.
#  • Firma de salida (B, C, k_target) para acoplarse sin cambios.
# ╰────────────────────────────────────────────────────────────────────╯

class _ResDilated(nn.Module):
    """Depthwise-Separable conv 3×1 con residual."""
    def __init__(self, C, dilation=1, stride=1):
        super().__init__()
        pad = dilation                           # mantiene longitud
        self.dw  = nn.Conv1d(C, C, 3, stride=stride,
                             padding=pad, dilation=dilation, groups=C)
        self.pw  = nn.Conv1d(C, C, 1)            # mezcla canales
        self.bn  = nn.BatchNorm1d(C)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Identity() if stride==1 else
                     nn.Conv1d(C, C, 1, stride=stride))

    def forward(self, x):
        out = self.act(self.bn(self.pw(self.dw(x.float()))))
        return out + self.skip(x.float())        # (B,C,L')

class TCNTemporalDisc(nn.Module):
    """
    Extractor temporal tipo TCN con dos bloques residuales.
    Entrada  : (B,C,T)
    Salida   : (B,C,k_target)  con  k_target = ⌊(T-1)/4⌋+1
    """
    def __init__(self, C:int, T:int, dilation=(1,2)):
        super().__init__()
        self.block1 = _ResDilated(C, dilation=dilation[0], stride=2)  # T→T/2
        self.block2 = _ResDilated(C, dilation=dilation[1], stride=2)  # →T/4

        # calcula k_target sin dummy
        self.k_target = (T - 1)//4 + 1

        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                         # (B,C,T)
        z = self.block2(self.block1(x))           # (B,C,k)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)                      # (B,C,k_target)


# ─────────────────────────  Inception-SE Temporal Disc  ─────────────────────

class SE1D(nn.Module):
    """Squeeze-and-Excitation para señales 1-D."""
    def __init__(self, C, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(C, C // r, 1), nn.ReLU(inplace=True),
            nn.Conv1d(C // r, C, 1), nn.Sigmoid()
        )
    def forward(self, x):                  # (B,C,L)
        w = self.fc(x.mean(-1, keepdim=True))  # (B,C,1)
        return x * w                        # recalibra canales


class InceptionSETemporalDisc(nn.Module):
    """
    Inception-1D (k=32/16/8+Pool) + Squeeze-Excite  + stride-2×2 → k = ⌈T/4⌉
    Salida: (B,C,k_target)  compatible con el resto de ramas.
    """
    def __init__(self, C: int, T: int, alpha: float = 0.5):
        super().__init__()
        Cmid          = int(C * alpha)
        self.k_target = (T + 3) // 4                # ceil(T/4)

        self.stage1 = self._make_stage(C, Cmid, stride=2)
        self.stage2 = self._make_stage(C, Cmid, stride=2)
        self.ln     = nn.LayerNorm(C, elementwise_affine=False)

    # ───────── helpers ──────────────────────────────────────────
    @staticmethod
    def _branch(cin, cmid, k, stride):
        pad = (k - stride) // 2
        return nn.Sequential(
            nn.Conv1d(cin, cmid, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(cmid, cmid, k, stride=stride, padding=pad),
            nn.BatchNorm1d(cmid), nn.LeakyReLU(0.2)
        )

    def _make_stage(self, C, Cmid, stride):
        branches = nn.ModuleList([
            self._branch(C, Cmid, 32, stride),
            self._branch(C, Cmid, 16, stride),
            self._branch(C, Cmid,  8, stride),
            nn.Sequential(
                nn.MaxPool1d(3, stride=stride, padding=1),
                nn.Conv1d(C, Cmid, 1), nn.LeakyReLU(0.2)
            )
        ])
        fuse = nn.Conv1d(4*Cmid, C, 1)
        se   = SE1D(C)
        return nn.ModuleList([branches, fuse, se])

    # ───────── etapa con SE ─────────────────────────────────────
    def _run_stage(self, x, stage):
        branches, fuse, se = stage
        outs = [b(x) for b in branches]
        Lmax = max(t.size(-1) for t in outs)           # iguala longitudes
        outs = [F.interpolate(t, size=Lmax, mode='linear', align_corners=False)
                if t.size(-1)!=Lmax else t for t in outs]
        z = torch.cat(outs, dim=1)                     # concat canales
        z = F.leaky_relu(fuse(z), 0.2)                 # 1×1 mezcla
        z = se(z)                                      # Squeeze-Excite
        return z

    # ───────── forward ─────────────────────────────────────────
    def forward(self, x):                              # (B,C,T)
        z = self._run_stage(x.float(), self.stage1)    # (B,C,T/2)
        z = self._run_stage(z,            self.stage2) # (B,C,k)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)                           # (B,C,k_target)


# ╭──────────────────────── ResNet-1D  Depthwise-Separable ───────────────────╮
#   • Cada bloque = DS-Conv(k3) → BN → ReLU  +  atajo 1×1 (residual)
#   • Dos bloques con stride-2 reducen T → ⌊T/4⌋  (igual que las otras ramas)
#   • Salida (B, C, k_target)  para fusionar sin cambios
# ╰───────────────────────────────────────────────────────────────────────────╯
class ResDSBlock(nn.Module):
    """Depthwise-Separable conv 3×1 con residual interno."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.dw  = nn.Conv1d(C, C, 3, stride=stride, padding=1, groups=C)
        self.pw  = nn.Conv1d(C, C, 1)                 # point-wise
        self.bn  = nn.BatchNorm1d(C)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Identity() if stride == 1 else
                     nn.Conv1d(C, C, 1, stride=stride))

    def forward(self, x):
        out = self.act(self.bn(self.pw(self.dw(x))))
        return out + self.skip(x)                     # (B,C,L')  mismo C


class ResNet1DTemporalDisc(nn.Module):
    """
    Temporal extractor basado en ResNet-1D depthwise-separable.
    T → T/2 (bloque-1, stride 2)  →  T/4 (bloque-2, stride 2)
    Salida : (B,C,k_target)  con k_target = ⌊(T-1)/4⌋+1.
    """
    def __init__(self, C: int, T: int):
        super().__init__()
        self.block1 = ResDSBlock(C, stride=2)         # T → T/2
        self.block2 = ResDSBlock(C, stride=2)         # → T/4
        self.k_target = (T - 1) // 4 + 1
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                             # (B,C,T)
        z = self.block2(self.block1(x.float()))       # (B,C,k)
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)
        return z.to(x.dtype)

# ╭──────────────────────── Inception-Temporal Disc ─────────────────────╮
# ───────────────────────────────────────────────────────────────
# InceptionTemporalDisc  •  robusto para T par o impar
# ───────────────────────────────────────────────────────────────
class InceptionTemporalDisc(nn.Module):
    def __init__(self, C: int, T: int, alpha: float = .5):
        super().__init__()
        Cmid          = int(C * alpha)
        self.k_target = (T + 3) // 4          # ceil(T / 4)  → 193 si T=769

        self.ln     = nn.LayerNorm(C, elementwise_affine=False)
        self.stage1 = self._make_stage(C, Cmid, stride=2)   # T  → ⌈T/2⌉
        self.stage2 = self._make_stage(C, Cmid, stride=2)   # ⌈T/2⌉ → ⌈T/4⌉

    # ---------- helpers ----------
    @staticmethod
    def _branch(cin, cmid, k, stride):
        pad = (k - stride) // 2               # garantiza misma L′
        return nn.Sequential(
            nn.Conv1d(cin, cmid, 1), nn.LeakyReLU(.2),
            nn.Conv1d(cmid, cmid, k, stride=stride, padding=pad),
            nn.BatchNorm1d(cmid), nn.LeakyReLU(.2))

    def _make_stage(self, C, Cmid, stride):
        return nn.ModuleList([
            self._branch(C, Cmid, 32, stride),
            self._branch(C, Cmid, 16, stride),
            self._branch(C, Cmid,  8, stride),
            nn.Sequential(
                nn.MaxPool1d(3, stride=stride, padding=1),
                nn.Conv1d(C, Cmid, 1), nn.LeakyReLU(.2)),
            nn.Conv1d(4 * Cmid, C, 1)        # fusión 1×1
        ])

    def _run_stage(self, x, stage):
        b1,b2,b3,b4,mix = stage
        outs = [b1(x), b2(x), b3(x), b4(x)]
        Lmax = max(t.size(-1) for t in outs)                     # intercala ↑
        outs = [F.interpolate(t, size=Lmax, mode='linear', align_corners=False)
                if t.size(-1)!=Lmax else t for t in outs]
        z = torch.cat(outs, dim=1)
        return F.leaky_relu(mix(z), .2)

    def forward(self, x):                    # (B,C,T)
        z = self._run_stage(x.float(), self.stage1)
        z = self._run_stage(z,            self.stage2)          # (B,C,k_tgt)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)
    

# ╭─────────────────────────  Squeeze-Excite para 1-D  ─────────────────────╮
class SE1D(nn.Module):
    """Squeeze-and-Excitation en el eje canal (C).  Peso extra ≈ 2·C/r ≪ C²."""
    def __init__(self, C: int, r: int = 8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)           # (B,C,k) → (B,C,1)
        self.fc = nn.Sequential(
            nn.Conv1d(C, C // r, 1), nn.ReLU(inplace=True),
            nn.Conv1d(C // r, C, 1), nn.Sigmoid()
        )
    def forward(self, x):                                # (B,C,k)
        w = self.fc(self.avgpool(x))                     # (B,C,1)
        return x * w                                     # recalibra canales
# ╰────────────────────────────────────────────────────────────────────────╯


class SpatialDisc(nn.Module):
    """
    Depthwise-Conv sobre electrodos + Squeeze-Excite por canal.
    Salida conserva (B,C,k_target) para fusión.
    """
    def __init__(self, C: int, T: int, k_target: int, r: int = 8):
        super().__init__()
        self.k_target = k_target

        self.conv = nn.Sequential(
            nn.Conv1d(C, C, 32, stride=2, padding=1, groups=C),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 16, stride=2, padding=1, groups=C),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2),
        )
        self.se = SE1D(C, r)                            # < 1 % params extra
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                               # (B,C,T)
        z = self.conv(x.float())                        # (B,C,T/4)
        z = F.interpolate(z, size=self.k_target,
                          mode="linear", align_corners=False)  # (B,C,k)
        z = self.se(z)                                  # pondera canales
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)  # normaliza
        return z.to(x.dtype)                            # (B,C,k)

# ╭────────────────────  Axial-Attention 1-D  (canal)  ───────────────────╮
#   * Aplica self-attention sólo a lo largo del eje canal (C), compartiendo
#     pesos para cada paso temporal t.
#   * Entrada  (B, C, L)   |  Salida (B, C, L)  (misma resolución)
# ╰────────────────────────────────────────────────────────────────────────╯
# ╭────────────────────  Axial Attention (canal)  corregido ──────────────────╮
class AxialAttn1D(nn.Module):
    """
    Self-attention sólo en el eje canal.
      in : (B, C, L)  →  out : (B, C, L)
    C debe ser divisible por n_heads (se comprueba).
    """
    def __init__(self, C: int, n_heads: int = 4):
        super().__init__()
        assert C % n_heads == 0, f"C={C} no divisible por n_heads={n_heads}"
        self.nh, self.dk = n_heads, C // n_heads
        self.qkv  = nn.Conv1d(C, 3*C, 1)
        self.proj = nn.Conv1d(C,   C, 1)
        self.scale = self.dk ** -0.5

    def forward(self, x):                         # (B,C,L)
        B, C, L = x.shape
        q,k,v = self.qkv(x).view(B,3,self.nh,self.dk,L).unbind(1)  # (B,nh,dk,L)

        # ─ permute a (B,L,nh,dk) y aplana B·L ───────────────────
        q = q.permute(0,3,1,2).reshape(B*L, self.nh, self.dk)
        k = k.permute(0,3,1,2).reshape(B*L, self.nh, self.dk)
        v = v.permute(0,3,1,2).reshape(B*L, self.nh, self.dk)

        attn = torch.softmax((q @ k.transpose(-2,-1))*self.scale, dim=-1)
        z    = (attn @ v).reshape(B, L, C).permute(0,2,1)         # (B,C,L)
        return self.proj(z)
# ╰──────────────────────────────────────────────────────────────────────────╯

# ───────────────────────────────────────────────────────────────────────────
#  SpatialAxialDisc  v2
#    • depth-wise downsample
#    • 1×1 bottleneck  (C → C/r → C)   +  Dropout(p)
#    • Axial Attention por canal
#    • LayerNorm
#    • mantiene (B,C,k_target)
# ───────────────────────────────────────────────────────────────────────────
def _pick_heads(C, max_heads=8):
    return max([h for h in range(1, max_heads+1) if C % h == 0])

class SpatialAxialDisc(nn.Module):
    """
    Rama espacial con cuello de botella y Axial-Attention.
    Args
    ----
      C, T          : canales, longitud de entrada
      k_target      : longitud objetivo (igual a la rama temporal)
      r             : factor de reducción del bottleneck (C → C/r)
      p_drop        : prob. de Dropout dentro del bottleneck
      max_heads     : máximo nº de cabezas permitido
    """
    def __init__(self, C: int, T: int, k_target: int,
                 r: int = 16, p_drop: float = 0.5, max_heads: int = 8):
        super().__init__()
        self.k_target = k_target

        # 1) depth-wise stride-2 ×2   (T → T/4)
        self.down = nn.Sequential(
            nn.Conv1d(C, C, 32, stride=2, padding=1, groups=C),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 16, stride=2, padding=1, groups=C),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2)
        )

        # 2) cuello de botella + Dropout
        self.bottle = nn.Sequential(
            nn.Conv1d(C, C // r, 1), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(C // r, C, 1)
        )

        # 3) Axial Attention (canal)  con nº de cabezas adaptativo
        heads = _pick_heads(C, max_heads)
        self.axial = AxialAttn1D(C, n_heads=heads)

        # 4) normalización final
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B,C,T)
        z = self.down(x.float())                            # (B,C,T/4)
        z = F.interpolate(z, size=self.k_target,
                          mode='linear', align_corners=False)  # (B,C,k)
        z = self.bottle(z)                                  # bottleneck + drop
        z = self.axial(z)                                   # atención canal
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)
        return z.to(x.dtype)                                # (B,C,k_target)


class HybridTemporalDisc(nn.Module):
    """
    1) TCN-Residual (dilated depthwise)  ↓4 en el eje temporal
    2) Transformer-1D sobre la secuencia comprimida
    Salida compatible: (B, C, k)  con k = ⌊(T-1)/4⌋+1
    """
    def __init__(self, C, T,
                 d_model=128, n_heads=4, n_layers=2, ff_mult=4):
        super().__init__()

        # ── bloque TCN (local) ───────────────────────────────
        self.tcn = nn.Sequential(
            _DilatedConvUnit(C, dilation=1, stride=2),  # T → T/2
            _DilatedConvUnit(C, dilation=2, stride=1),
            _DilatedConvUnit(C, dilation=4, stride=2),  # → T/4
            _DilatedConvUnit(C, dilation=8, stride=1)
        )
        k = (T - 1)//4 + 1         # longitud tras la TCN

        # ── bloque Transformer (global) ──────────────────────
        self.proj_in  = nn.Conv1d(C, d_model, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*ff_mult,
            batch_first=True, norm_first=True, activation='gelu')
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.proj_out = nn.Conv1d(d_model, C, 1)

        self.k_target = k
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                  # (B,C,T)
        z = self.tcn(x.float())            # (B,C,k)
        z = self.proj_in(z)                # (B,d_model,k)
        z = self.encoder(z.permute(0,2,1)).permute(0,2,1)
        z = self.proj_out(z)               # (B,C,k)
        z = self.ln(z.transpose(1,2)).transpose(1,2)
        return z.to(x.dtype)               # (B,C,k)


class _DilatedConvUnit(nn.Module):
    """Depthwise-dilated conv 3×1 + PW 1×1 + BN + ReLU + skip."""
    def __init__(self, C, dilation=1, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(C, C, 3, stride=stride,
                            padding=dilation, dilation=dilation, groups=C)
        self.pw = nn.Conv1d(C, C, 1)
        self.bn = nn.BatchNorm1d(C)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if stride == 1 else nn.Conv1d(C, C, 1, stride=stride)

    def forward(self, x):
        y = self.act(self.bn(self.pw(self.dw(x.float()))))
        return y + self.skip(x.float())                 # (B,C,L')

class DilatedCNNTemporalDisc(nn.Module):
    """
    Entrada : (B, C, T)
    Salida  : (B, C, k_target)  con  k_target = ⌊(T-1)/4⌋+1
    Bloques: dil=1,stride2 → dil=2 → dil=4,stride2 → dil=8
    """
    def __init__(self, C: int, T: int):
        super().__init__()
        # bloque-1  (dil1, stride2)  ➜ T/2
        # bloque-2  (dil2, stride1)
        # bloque-3  (dil4, stride2)  ➜ T/4
        # bloque-4  (dil8, stride1)
        self.net = nn.Sequential(
            _DilatedConvUnit(C, dilation=1, stride=2),
            _DilatedConvUnit(C, dilation=2, stride=1),
            _DilatedConvUnit(C, dilation=4, stride=2),
            _DilatedConvUnit(C, dilation=8, stride=1)
        )
        self.k_target = (T - 1) // 4 + 1
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                               # (B,C,T)
        z = self.net(x)                                 # (B,C,k)
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)  # normaliza canales
        return z.to(x.dtype)   

class TemporalDisc(nn.Module):
    """Convoluciona a lo largo del eje temporal y reduce T → k."""
    def __init__(self, C: int, T: int):
        super().__init__()
        # k_target = T // 4 con dos stride-2
        k = (T + 2 - 32) // 2 + 1          # primera conv
        k = (k + 2 - 16) // 2 + 1          # segunda conv  → 179 si T=769
        self.k_target = k

        self.conv = nn.Sequential(
            nn.Conv1d(C, C, 32, stride=2, padding=1),  # T→T/2
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 16, stride=2, padding=1),  # →T/4
        )
        self.ln = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x):                   # (B,C,T)
        z = self.conv(x.float())            # (B,C,k)
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)
        return z.to(x.dtype)                # (B,C,k)

