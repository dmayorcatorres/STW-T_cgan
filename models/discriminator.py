import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import TemporalDisc, Spatial, Frequency

class Discriminator(nn.Module):
    def __init__(self, n_cls, C, T, k, emb, use_t=True, use_s=True, use_f=True):
        super().__init__()
        self.C, self.T, self.k = C, T, k
        self.label = nn.Embedding(n_cls, emb)

        self.use_t, self.use_s, self.use_f = use_t, use_s, use_f
        if use_t: self.t = TemporalDisc(C)
        if use_s: self.s = Spatial(C)
        if use_f: self.f = Frequency(C)

        self.head = nn.Sequential(
            nn.Linear(C * T + emb, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
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
        feat = F.interpolate(merged, size=self.T, mode="linear")
        flat = feat.view(x.size(0), -1)
        return self.head(torch.cat([flat, self.label(y)], 1))
