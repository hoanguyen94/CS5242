"""
utils.py — Shared utilities: reproducibility, device selection,
           filesystem helpers, and dataloader construction.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────
# Reproducibility & Device
# ──────────────────────────────────────────────

def set_seed(seed: int = 24) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Filesystem helpers
# ──────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# HF Dataset → PyTorch DataLoader
# ──────────────────────────────────────────────

class HFDatasetWrapper(torch.utils.data.Dataset):
    """Wraps a HF split to return transformed tensors and labels."""

    def __init__(self, ds_split, transform):
        self.ds = ds_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img: Image.Image = ex["image"].convert("RGB")
        x = self.transform(img)
        y = ex["label"]
        return x, y


def make_loaders(ds, train_tf, eval_tf, batch_size: int = 128, num_workers: int = 4):
    train_ds = HFDatasetWrapper(ds["train"], train_tf)
    val_ds = HFDatasetWrapper(ds["validation"], eval_tf)
    test_ds = HFDatasetWrapper(ds["test"], eval_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
