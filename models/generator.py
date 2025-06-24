import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import TemporalGen, Spatial, Frequency  # Si los bloques están separados

class Generator(nn.Module):
    def __init__(self, z_dim, n_cls, C, T, k, emb, use_t=True, use_s=True, use_f=True):
        super().__init__()
        self.C, self.T, self.k = C, T, k
        self.label = nn.Embedding(n_cls, emb)
        self.fc = nn.Linear(z_dim + emb, C * T)

        self.use_t, self.use_s, self.use_f = use_t, use_s, use_f
        if use_t: self.t = TemporalGen(C)
        if use_s: self.s = Spatial(C)
        if use_f: self.f = Frequency(C)

        self.out_act = nn.Tanh()

    def forward(self, z, y, return_br=False):
        h = torch.cat([z, self.label(y)], 1)
        x = self.fc(h).view(-1, self.C, self.T)

        branches = []
        if self.use_t:
            branches.append(self.t(x))
        if self.use_s:
            s = self.s(x).unsqueeze(-1).expand(-1, -1, self.k)
            branches.append(s)
        if self.use_f:
            f = self.f(x).unsqueeze(-1).expand(-1, -1, self.k)
            branches.append(f)

        merged = sum(branches) / len(branches)
        out = self.out_act(F.interpolate(merged, size=self.T, mode="linear"))

        if return_br:
            t = self.t(x) if self.use_t else torch.zeros_like(merged)
            s = self.s(x).unsqueeze(-1).expand(-1, -1, self.k) if self.use_s else torch.zeros_like(merged)
            f = self.f(x).unsqueeze(-1).expand(-1, -1, self.k) if self.use_f else torch.zeros_like(merged)
            return out, (t, s, f, merged)
        else:
            return out
