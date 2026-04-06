"""
main.py — Project Entry-point
==============================
Orchestrates all experiments for CS5242 Mini-ImageNet classification.

Tasks
-----
  explore             — dataset statistics, class distribution, mean/std
  visualize_transforms — visualise augmented training samples
  features_ml         — Approach 1: classical ML on frozen features
  finetune            — Approach 2: pretrained fine-tuning (freeze variants)
  scratch             — Approach 3: train from scratch baseline
  tsne                — t-SNE of ConvNeXt backbone features

Quick start
-----------
  # Run data exploration
  python main.py --task explore --save_dir experiments/run1

  # Approach 1
  python main.py --task features_ml --clf_type logreg --save_dir experiments/run1

  # Approach 2 (pretrained, freeze backbone)
  python main.py --task finetune --freeze_policy backbone --save_dir experiments/run1

  # Approach 3 (from scratch)
  python main.py --task scratch --save_dir experiments/run1

  # t-SNE
  python main.py --task tsne --tsne_split validation --save_dir experiments/run1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import (
    set_seed, get_device, ensure_dir,
    load_mini_imagenet, make_transforms,
    explore, visualize_transforms,
    HFDatasetWrapper,
)
from model import build_backbone, extract_convnext_features
from methods import classical_ml_experiment, train_finetune, train_from_scratch

# ──────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CS5242 Mini-ImageNet experiments")

    # General
    p.add_argument("--task", required=True,
                   choices=["explore", "visualize_transforms",
                             "features_ml", "finetune", "scratch", "tsne"])
    p.add_argument("--save_dir",    default="experiments/temp")
    p.add_argument("--seed",        type=int,   default=24)
    p.add_argument("--use_gpu",     action="store_true")
    p.add_argument("--subset",      type=int,   default=None,
                   help="Limit dataset size per split (for quick testing)")

    # Transforms
    p.add_argument("--img_size",    type=int,   default=32)
    p.add_argument("--use_aug",     action="store_true")

    # Backbone
    p.add_argument("--backbone",    default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1", \
                            "resnet18_scratch", "convnext_tiny_scratch", "ournet"])

    # Training
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="none", choices=["cosine", "step", "none"],
                   help="Learning rate scheduler type.")
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="Number of epochs for linear learning rate warmup.")
    
    # Approach 2: freeze policy
    p.add_argument("--freeze_policy", default="backbone",
                   choices=["backbone", "last_stage", "none"])

    # Approach 1: classifier type
    p.add_argument("--clf_type", default="logreg",
                   choices=["logreg", "linear_svm"])

    # t-SNE
    p.add_argument("--tsne_split",  default="validation",
                   choices=["train", "validation", "test"])
    p.add_argument("--tsne_n",      type=int, default=2000)

    return p.parse_args()

# ──────────────────────────────────────────────
# t-SNE Visualisation
# ──────────────────────────────────────────────

def tsne_visualize(
    ds_split,
    eval_tf,
    device: torch.device,
    backbone: str,
    save_path: Path,
    n_samples: int = 2000,
    seed: int = 42,
) -> None:
    """Extract backbone GAP features and render a 2-D t-SNE plot coloured by class."""
    random.seed(seed)
    idxs = list(range(len(ds_split)))
    if len(idxs) > n_samples:
        idxs = random.sample(idxs, k=n_samples)

    subset = ds_split.select(idxs)
    loader = torch.utils.data.DataLoader(
        HFDatasetWrapper(subset, eval_tf),
        batch_size=256, shuffle=False, num_workers=2,
    )

    model = build_backbone(num_classes=100, pretrained=True, device=device, backbone=backbone)
    feats, labels = extract_convnext_features(model, loader, device)
    print(f"t-SNE on features shape: {feats.shape}")

    tsne = TSNE(
        n_components=2, perplexity=30, learning_rate="auto",
        init="pca", random_state=seed, n_iter=1000, verbose=1,
    )
    Z = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 7))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab20", s=6, alpha=0.8)
    plt.title("t-SNE of ConvNeXt-Tiny Features (validation subset)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")



# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device   = get_device(args.use_gpu)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    print(f"Task: {args.task}  |  device: {device}  |  save_dir: {save_dir}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset: timm/mini-imagenet …")
    ds = load_mini_imagenet(subset=args.subset)
    print({k: len(v) for k, v in ds.items()})

    # ── Load or compute mean/std for normalisation ────────────────────────────
    mean_std_path = save_dir / "train_mean_std.json"
    if mean_std_path.exists():
        with open(mean_std_path) as f:
            ms = json.load(f)
            mean = np.array(ms["mean"], dtype=np.float32)
            std  = np.array(ms["std"],  dtype=np.float32)
        print("Loaded normalisation stats from file.")
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print("Using ImageNet normalisation defaults (run 'explore' to compute dataset stats).")

    train_tf, eval_tf, tensor_only = make_transforms(
        img_size=args.img_size, mean=mean, std=std, use_aug=args.use_aug
    )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    if args.task == "explore":
        explore(ds, save_dir=save_dir)

    elif args.task == "visualize_transforms":
        vis_dir = save_dir / "transform_viz"
        visualize_transforms(ds["train"], train_tf, eval_tf, tensor_only,
                              save_dir=vis_dir, n=8, seed=0)

    elif args.task == "features_ml":
        classical_ml_experiment(
            ds=ds, eval_tf=eval_tf, device=device,
            backbone=args.backbone,
            clf_type=args.clf_type,
            batch_size=max(128, args.batch_size),
            save_dir=save_dir,
        )

    elif args.task == "finetune":
        train_finetune(
            ds=ds, train_tf=train_tf, eval_tf=eval_tf, device=device,
            backbone=args.backbone,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, freeze_policy=args.freeze_policy,
            use_pretrained=True, save_dir=save_dir,
        )

    elif args.task == "scratch":
        train_from_scratch(
            ds=ds, train_tf=train_tf, eval_tf=eval_tf, device=device,
            backbone=args.backbone,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_dir=save_dir,
        )

    elif args.task == "tsne":
        tsne_visualize(
            ds_split=ds[args.tsne_split], eval_tf=eval_tf, device=device,
            backbone=args.backbone,
            save_path=save_dir / f"tsne_{args.tsne_split}.png",
            n_samples=args.tsne_n, seed=args.seed,
        )


if __name__ == "__main__":
    main()
