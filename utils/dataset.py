import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
from pathlib import Path
import numpy as np

SEED = 42  # Semilla fija

class ZScore:
    def __init__(self, μ, σ, eps=1e-8):
        self.μ, self.σ, self.eps = μ, σ, eps

    def __call__(self, x):
        return (x - self.μ) / (self.σ + self.eps)

class ECGDataset(Dataset):
    def __init__(self, path: Path, key_x="emg", key_y="tag", tf=None):
        mat = sio.loadmat(path)
        self.x = torch.tensor(mat[key_x]).float()
        y_raw = torch.tensor(mat[key_y]).long().squeeze()

        unique_labels = torch.unique(y_raw)
        label_map = {int(lbl): idx for idx, lbl in enumerate(unique_labels.tolist())}
        self.y = torch.tensor([label_map[int(lbl)] for lbl in y_raw], dtype=torch.long)

        self.tf = tf

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.tf:
            x = self.tf(x)
        return x, self.y[idx]

def build_loader(path: Path, batch_size: int, device: str, split_ratio: float = 0.7, seed: int = SEED):
    ds_tmp = ECGDataset(path)
    μ, σ = map(float, [ds_tmp.x.mean(), ds_tmp.x.std()])
    dataset = ECGDataset(path, tf=ZScore(μ, σ))

    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=device == "cuda", num_workers=4 if device == "cuda" else 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=device == "cuda", num_workers=4 if device == "cuda" else 0)

    return train_loader, val_loader

def num_classes(dataset: Dataset) -> int:
    return int(dataset.y.max().item() + 1)
