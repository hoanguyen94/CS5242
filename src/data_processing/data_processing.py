import json
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from datasets import load_dataset, load_from_disk
import os
from ..utils import ensure_dir
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from PIL import Image

def load_mini_imagenet(subset=None, cache_path=None):

    ds = None

    if cache_path:
        try:
            if os.path.isdir(cache_path) and os.listdir(cache_path):
                ds = load_from_disk(cache_path)
        except Exception as e:
            print(f"Failed to load dataset from cache_path={cache_path}: {e}")
            ds = None

    if ds is None:
        ds = load_dataset("timm/mini-imagenet")
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            ds.save_to_disk(cache_path)

    return ds

def class_names_from_ds(ds_split):
    return ds_split.features["label"].names

def gather_image_meta(ds_split, sample_limit: Optional[int] = None) -> Dict[str, int]:
    """
    Collects resolution and format metadata frequency.
    """
    res_counter = Counter()
    fmt_counter = Counter()
    mode_counter = Counter()
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
        "resolutions": {f"{w}x{h}": c for (w, h), c in res_counter.items()},
        "formats": dict(fmt_counter),
        "modes": dict(mode_counter),
    }
    
def show_random_grid(ds_split, class_names, per_class=5, classes_to_show=6, save_path: Optional[Path] = None):
    """
    Visual inspection: for a subset of classes, show a grid of random samples.
    """
    # Build index per class
    idxs_by_class = defaultdict(list)
    for i, ex in enumerate(ds_split):
        idxs_by_class[ex["label"]].append(i)

    chosen_classes = sorted(random.sample(list(idxs_by_class.keys()), k=min(classes_to_show, len(idxs_by_class))))
    nrows = len(chosen_classes)
    ncols = per_class
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.0, nrows*2.0))
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
            axes[r, c].axis('off')
    plt.suptitle("Random Samples by Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visual inspection grid to: {save_path}")
    plt.close()

def compute_mean_std(ds_split, num_workers: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over a split. Operates over raw images scaled to [0,1].
    Uses a numerically stable running mean/std.
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
        # Welford aggregation
        delta = batch_mean - mean
        tot_n = cnt + batch_n
        mean = mean + delta * (batch_n / tot_n)
        M2 = M2 + batch_var * batch_n + (delta**2) * (cnt * batch_n / tot_n)
        cnt = tot_n

    var = M2 / max(cnt, 1)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)

def explore_dataset(ds, save_dir: Path):
    ensure_dir(save_dir)
    train = ds["train"]
    class_names = class_names_from_ds(train)

    # Class distribution
    counts = Counter([ex["label"] for ex in train])
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

    print(f"Number of classes: {len(class_names)}. Balanced? "
          f"min={min(dist)} max={max(dist)} (ideally equal)")

    # Image meta
    meta_train = gather_image_meta(train)
    meta_val = gather_image_meta(ds["validation"])
    meta_test = gather_image_meta(ds["test"])
    with open(save_dir / "image_meta.json", "w") as f:
        json.dump({"train": meta_train, "validation": meta_val, "test": meta_test}, f, indent=2)
    print("Saved image metadata (resolutions/formats/modes).")

    # Visual inspection (random grid)
    show_random_grid(train, class_names, per_class=5, classes_to_show=6, save_path=save_dir / "visual_grid.png")

    # Mean/std for normalization (over train split)
    mean, std = compute_mean_std(train)
    with open(save_dir / "train_mean_std.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Computed mean/std: mean={mean}, std={std}. Saved to train_mean_std.json")
    
    
def make_transforms(img_size=224, mean=None, std=None):
    """
    Returns (train_transform, eval_transform) using dataset mean/std.
    If mean/std are None, fallback to ImageNet defaults.
    """
    if mean is None or std is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # For "before vs after" visualization, also return a "no-op" tensorizer (no normalization)
    tensor_only = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    return train_tf, eval_tf, tensor_only

def denormalize(t: torch.Tensor, mean, std):
    mean = torch.tensor(mean).view(-1,1,1)
    std = torch.tensor(std).view(-1,1,1)
    return (t * std) + mean

def visualize_transforms(ds_split, train_tf, eval_tf, tensor_only, save_dir: Path, n=6, seed=0):
    """
    Show original vs eval-transformed and multiple augmented views from train_tf.
    """
    random.seed(seed)
    ensure_dir(save_dir)
    # Pick n random images
    idxs = random.sample(range(len(ds_split)), k=min(n, len(ds_split)))
    for k, i in enumerate(idxs):
        img: Image.Image = ds_split[i]["image"].convert("RGB")

        # Original
        orig = tensor_only(img)  # no normalization, center crop for consistent view
        # Eval transform (normalized) -> de-normalize for display
        eval_t = eval_tf(img)
        eval_disp = denormalize(eval_t, mean=eval_tf.transforms[-1].mean, std=eval_tf.transforms[-1].std)

        # Train transform — show 3 random augmented variants
        aug_imgs = [train_tf(img) for _ in range(3)]
        aug_disps = [denormalize(t, mean=train_tf.transforms[-1].mean, std=train_tf.transforms[-1].std)
                     for t in aug_imgs]

        # Plot
        cols = 2 + len(aug_disps)
        plt.figure(figsize=(3*cols, 3))
        titles = ["Original", "Eval (resized+center crop)"] + [f"Aug {j+1}" for j in range(len(aug_disps))]
        tensors = [orig, eval_disp] + aug_disps
        for j, t in enumerate(tensors):
            plt.subplot(1, cols, j+1)
            plt.imshow(t.permute(1, 2, 0).clamp(0,1).numpy())
            plt.title(titles[j], fontsize=9)
            plt.axis('off')
        out_path = save_dir / f"transforms_{k:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved transform visualization: {out_path}")