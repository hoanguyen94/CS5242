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


def gather_image_meta(ds_split, sample_limit: Optional[int] = 500) -> Dict:
    """Collects resolution and format metadata from a sample of images."""
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
        "sampled": n,
        "total": len(ds_split),
        "resolutions": {f"{w}x{h}": c for (w, h), c in res_counter.items()},
        "formats": dict(fmt_counter),
        "modes": dict(mode_counter),
    }


def show_random_grid(ds_split, class_names, per_class=5, classes_to_show=6, save_path: Optional[Path] = None):
    """Visual inspection: for a subset of classes, show a grid of random samples."""
    # Use fast column access instead of iterating all rows
    all_labels = ds_split["label"]
    idxs_by_class = defaultdict(list)
    for i, label in enumerate(all_labels):
        idxs_by_class[label].append(i)

    chosen_classes = sorted(random.sample(list(idxs_by_class.keys()), k=min(classes_to_show, len(idxs_by_class))))
    nrows = len(chosen_classes)
    ncols = per_class
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
            axes[r, c].axis('off')
    plt.suptitle("Random Samples by Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visual inspection grid to: {save_path}")
    plt.show()


def compute_mean_std(ds_split, sample_size: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over a random sample of images scaled to [0,1].
    Uses Welford aggregation for numerical stability.
    """
    n_total = len(ds_split)
    indices = random.sample(range(n_total), k=min(sample_size, n_total))

    cnt = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for idx in indices:
        img: Image.Image = ds_split[idx]["image"].convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
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


def explore_dataset(ds, save_dir: Path):
    ensure_dir(save_dir)
    train = ds["train"]
    val = ds["validation"]
    test = ds["test"]
    class_names = class_names_from_ds(train)

    # ── Split size summary ──
    print("=== Split Sizes ===")
    for name, split in [("train", train), ("validation", val), ("test", test)]:
        print(f"  {name:>12s}: {len(split):,} images")

    # ── Class distribution (all splits, single fast column access) ──
    train_labels = train["label"]
    val_labels = val["label"]
    test_labels = test["label"]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    n_classes = len(class_names)
    train_dist = [train_counts.get(i, 0) for i in range(n_classes)]
    val_dist = [val_counts.get(i, 0) for i in range(n_classes)]
    test_dist = [test_counts.get(i, 0) for i in range(n_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, dist, title in zip(axes, [train_dist, val_dist, test_dist],
                                ["Train", "Validation", "Test"]):
        ax.bar(range(n_classes), dist)
        ax.set_title(f"Class Distribution ({title})")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution_all_splits.png", dpi=200)
    plt.show()
    print("Saved class distribution plot (all splits).")

    print(f"Number of classes: {n_classes}")
    print(f"  Train — min={min(train_dist)} max={max(train_dist)}")
    print(f"  Val   — min={min(val_dist)} max={max(val_dist)}")
    print(f"  Test  — min={min(test_dist)} max={max(test_dist)}")

    # ── Image metadata (sampled) ──
    meta_train = gather_image_meta(train, sample_limit=500)
    meta_val = gather_image_meta(val, sample_limit=500)
    meta_test = gather_image_meta(test, sample_limit=500)
    with open(save_dir / "image_meta.json", "w") as f:
        json.dump({"train": meta_train, "validation": meta_val, "test": meta_test}, f, indent=2)
    print("Saved image metadata (sampled 500 per split).")

    # ── Visual inspection (random grid) ──
    show_random_grid(train, class_names, per_class=5, classes_to_show=6,
                     save_path=save_dir / "visual_grid.png")

    # ── Mean/std for normalization (sampled from train) ──
    mean, std = compute_mean_std(train, sample_size=5000)
    with open(save_dir / "train_mean_std.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Computed mean/std (sampled 5000): mean={mean}, std={std}")


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
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf, tensor_only


def denormalize(t: torch.Tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (t * std) + mean


def visualize_transforms(ds_split, train_tf, eval_tf, tensor_only, save_dir: Path, n=6, seed=0):
    """Show original vs eval-transformed and multiple augmented views from train_tf."""
    random.seed(seed)
    ensure_dir(save_dir)
    idxs = random.sample(range(len(ds_split)), k=min(n, len(ds_split)))
    for k, i in enumerate(idxs):
        img: Image.Image = ds_split[i]["image"].convert("RGB")

        orig = tensor_only(img)
        eval_t = eval_tf(img)
        eval_disp = denormalize(eval_t, mean=eval_tf.transforms[-1].mean, std=eval_tf.transforms[-1].std)

        aug_imgs = [train_tf(img) for _ in range(3)]
        aug_disps = [denormalize(t, mean=train_tf.transforms[-1].mean, std=train_tf.transforms[-1].std)
                     for t in aug_imgs]

        cols = 2 + len(aug_disps)
        plt.figure(figsize=(3 * cols, 3))
        titles = ["Original", "Eval (resized+center crop)"] + [f"Aug {j+1}" for j in range(len(aug_disps))]
        tensors = [orig, eval_disp] + aug_disps
        for j, t in enumerate(tensors):
            plt.subplot(1, cols, j + 1)
            plt.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
            plt.title(titles[j], fontsize=9)
            plt.axis('off')
        out_path = save_dir / f"transforms_{k:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.show()
        print(f"Saved transform visualization: {out_path}")
