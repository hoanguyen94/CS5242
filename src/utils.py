from pathlib import Path
"""
utils.py — Shared utilities: reproducibility, dataset loading, transforms,
           data exploration, visualisation helpers, and dataloader construction.
"""

import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from datasets import load_dataset


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
# Dataset Utilities
# ──────────────────────────────────────────────

def load_mini_imagenet(subset: Optional[int] = None) -> Dict:
    """
    Loads timm/mini-imagenet splits as Hugging Face datasets.
    Optionally subsets each split to `subset` examples for speed.
    """
    ds = load_dataset("timm/mini-imagenet", download_mode="reuse_cache_if_exists")
    if subset is not None:
        ds = {k: v.select(range(min(subset, len(v)))) for k, v in ds.items()}
    ds["class_names"] = class_names_from_ds(ds["train"])
    return ds


def load_class_mapping(path: Path) -> Dict[str, str]:
    """Loads class ID to name mapping from map_clsloc.txt."""
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def class_names_from_ds(ds_split):
    # Load mapping if available
    map_path = Path(__file__).parent / "map_clsloc.txt"
    if map_path.exists():
        mapping = load_class_mapping(map_path)
        ids = ds_split.features["label"].names
        return [mapping.get(id, id) for id in ids]
    else:
        return ds_split.features["label"].names


def compute_mean_std(ds_split, num_workers: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over a split. Operates over raw images scaled to [0, 1].
    Uses a numerically stable running mean/std (Welford aggregation).
    """
    cnt = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for ex in ds_split:
        img: Image.Image = ex["image"].convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, C
        arr = arr.reshape(-1, 3)
        batch_n = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        delta = batch_mean - mean
        tot_n = cnt + batch_n
        mean = mean + delta * (batch_n / tot_n)
        M2 = M2 + batch_var * batch_n + (delta ** 2) * (cnt * batch_n / tot_n)
        cnt = tot_n

    var = M2 / max(cnt, 1)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def gather_image_meta(ds_split, sample_limit: Optional[int] = None) -> Dict:
    """Collects resolution and format metadata frequency."""
    res_counter = collections.Counter()
    fmt_counter = collections.Counter()
    mode_counter = collections.Counter()
    n = 0
    for ex in ds_split:
        img: Image.Image = ex["image"]
        res_counter[(img.width, img.height)] += 1
        fmt_counter[img.format or "RAW"] += 1
        mode_counter[img.mode] += 1
        n += 1
        if sample_limit and n >= sample_limit:
            break
    return {
        "total": n,
        "resolutions": {f"{w}x{h}": count for (w, h), count in res_counter.items()},
        "formats": dict(fmt_counter),
        "modes": dict(mode_counter),
    }


def show_random_grid(
    ds_split, class_names, per_class=5, classes_to_show=6,
    save_path: Optional[Path] = None
):
    """
    Visual inspection: for a subset of classes, show a grid of random samples.
    """
    idxs_by_class = collections.defaultdict(list)
    for i, ex in enumerate(ds_split):
        idxs_by_class[ex["label"]].append(i)

    chosen_classes = sorted(
        random.sample(list(idxs_by_class.keys()), k=min(classes_to_show, len(idxs_by_class)))
    )
    nrows, ncols = len(chosen_classes), per_class
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))
    if nrows == 1:
        axes = np.array([axes])
    for r, cls in enumerate(chosen_classes):
        cls_name = class_names[cls]
        picks = random.sample(idxs_by_class[cls], k=min(per_class, len(idxs_by_class[cls])))
        for c, idx in enumerate(picks):
            img = ds_split[idx]["image"]
            axes[r, c].imshow(img)
            if c == 0:
                axes[r, c].set_ylabel(cls_name, fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        for c in range(len(picks), ncols):
            axes[r, c].axis("off")
    plt.suptitle("Random Samples by Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visual inspection grid to: {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# Transforms & Visualisation
# ──────────────────────────────────────────────

def make_transforms(
    img_size: int = 224,
    mean=None,
    std=None,
    use_aug: bool = False,
):
    """
    Returns (train_transform, eval_transform, tensor_only) using dataset mean/std.
    Falls back to ImageNet defaults when mean/std are None.
    """
    if mean is None or std is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # For "before vs after" visualisation — no normalization
    tensor_only = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf, tensor_only


def denormalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (t * std) + mean


def visualize_transforms(
    ds_split, train_tf, eval_tf, tensor_only,
    save_dir: Path, n: int = 6, seed: int = 0
):
    """Show original vs eval-transformed and multiple augmented views."""
    random.seed(seed)
    ensure_dir(save_dir)
    idxs = random.sample(range(len(ds_split)), k=min(n, len(ds_split)))
    for k, i in enumerate(idxs):
        img: Image.Image = ds_split[i]["image"].convert("RGB")

        orig = tensor_only(img)
        eval_t = eval_tf(img)
        eval_disp = denormalize(eval_t, mean=eval_tf.transforms[-1].mean, std=eval_tf.transforms[-1].std)

        aug_imgs = [train_tf(img) for _ in range(3)]
        aug_disps = [
            denormalize(t, mean=train_tf.transforms[-1].mean, std=train_tf.transforms[-1].std)
            for t in aug_imgs
        ]

        cols = 2 + len(aug_disps)
        plt.figure(figsize=(3 * cols, 3))
        titles = ["Original", "Eval (resized+center crop)"] + [f"Aug {j+1}" for j in range(len(aug_disps))]
        tensors = [orig, eval_disp] + aug_disps
        for j, t in enumerate(tensors):
            plt.subplot(1, cols, j + 1)
            plt.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
            plt.title(titles[j], fontsize=9)
            plt.axis("off")
        out_path = save_dir / f"transforms_{k:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved transform visualization: {out_path}")


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


def make_loaders(ds, train_tf, eval_tf, batch_size: int = 128, num_workers: int = 2):
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


# ──────────────────────────────────────────────
# Data Exploration
# ──────────────────────────────────────────────

def explore(ds, save_dir: Path) -> None:
    ensure_dir(save_dir)
    train = ds["train"]
    class_names = class_names_from_ds(train)

    # Class distribution
    counts = collections.Counter([ex["label"] for ex in train])
    dist = [counts.get(i, 0) for i in range(len(class_names))]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(class_names)), dist)
    plt.title("Class Distribution (Train)")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution_train.png", dpi=200)
    plt.close()
    print("Saved class distribution plot.")
    print(
        f"Number of classes: {len(class_names)}. Balanced? "
        f"min={min(dist)} max={max(dist)} (ideally equal)"
    )

    # Image metadata
    meta_train = gather_image_meta(train)
    meta_val = gather_image_meta(ds["validation"])
    meta_test = gather_image_meta(ds["test"])
    print("meta_train: ", meta_train)
    print("meta_val: ", meta_val)
    print("meta_test: ", meta_test)

    with open(save_dir / "image_meta.json", "w") as f:
        json.dump({"train": meta_train, "validation": meta_val, "test": meta_test}, f, indent=2)
    print("Saved image metadata (resolutions/formats/modes).")

    # Visual grid
    show_random_grid(train, class_names, per_class=5, classes_to_show=6,
                     save_path=save_dir / "visual_grid.png")

    # Mean/std
    mean, std = compute_mean_std(train)
    with open(save_dir / "train_mean_std.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Computed mean/std: mean={mean}, std={std}. Saved to train_mean_std.json")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)