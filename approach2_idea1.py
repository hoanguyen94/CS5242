"""
approach2_idea1.py — Improved Approach 2: Pretrained Fine-tuning (ResNet18)
===========================================================================
Improvements over the baseline finetune.py:
  1. Richer MLP classifier head        (Linear→BN→ReLU→Dropout→Linear)
  2. Label smoothing                    (CrossEntropyLoss label_smoothing=0.1)
  3. Differential (layer-wise) LRs     (head >> layer4 >> layer3 >> stem)
  4. Cosine LR schedule + linear warmup (OneCycleLR)
  5. Gradual unfreezing                 (epochs split into 3 phases)
  6. Stronger augmentation              (RandAugment + RandomErasing on top of
                                         the existing use_aug transforms)

Usage
-----
  # Basic run (uses existing mean/std file if present)
  python approach2_idea1.py --save_dir experiments/idea1 --use_gpu

  # With subset for quick testing
  python approach2_idea1.py --save_dir experiments/idea1 --use_gpu --subset 2000

  # Full run, more epochs
  python approach2_idea1.py --save_dir experiments/idea1 --use_gpu --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils import (
    ensure_dir, load_mini_imagenet, set_seed, get_device,
    make_loaders, HFDatasetWrapper,
)
from model import build_backbone, count_params, try_flops, evaluate


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

NUM_CLASSES  = 100
BACKBONE     = "resnet18"   # fixed for this experiment
RESNET18_DIM = 512          # output channels before fc


# ──────────────────────────────────────────────
# 1. Richer Classifier Head
# ──────────────────────────────────────────────

def _replace_fc_with_mlp(model: nn.Module, hidden: int = 256, dropout: float = 0.3) -> nn.Module:
    """
    Replace the single nn.Linear fc head with a small BN-MLP:
      fc → Linear(512, hidden) → BN → ReLU → Dropout → Linear(hidden, 100)
    """
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, NUM_CLASSES),
    )
    return model


# ──────────────────────────────────────────────
# 2. Stronger Augmentation
# ──────────────────────────────────────────────

def make_improved_transforms(img_size: int, mean, std):
    """
    Training: RandAugment + RandomErasing on top of crop/flip.
    Eval:     standard resize + center-crop.
    """
    _mean = tuple(mean.tolist() if hasattr(mean, "tolist") else mean)
    _std  = tuple(std.tolist()  if hasattr(std,  "tolist") else std)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2),
                                  ratio=(0.3, 3.3), value="random"),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # slight over-scale then crop
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_tf, eval_tf


# ──────────────────────────────────────────────
# 3. Freeze / Unfreeze Helpers
# ──────────────────────────────────────────────

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_all_except_head(model: nn.Module) -> None:
    """Phase 1: Only the new MLP head trains."""
    _set_grad(model, False)
    _set_grad(model.fc, True)


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    """Phase 2: Head + last residual stage."""
    _set_grad(model, False)
    _set_grad(model.layer4, True)
    _set_grad(model.fc, True)


def unfreeze_all(model: nn.Module) -> None:
    """Phase 3: Full end-to-end fine-tuning."""
    _set_grad(model, True)


# ──────────────────────────────────────────────
# 4. Differential LR Optimizer
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    """
    Assign progressively lower learning rates to earlier layers.
    Stem/BN layers get 10× less than the head; layer3 gets 5× less, etc.
    """
    stem_params   = list(model.conv1.parameters()) + list(model.bn1.parameters())
    layer1_params = list(model.layer1.parameters())
    layer2_params = list(model.layer2.parameters())
    layer3_params = list(model.layer3.parameters())
    layer4_params = list(model.layer4.parameters())
    head_params   = list(model.fc.parameters())

    param_groups = [
        {"params": stem_params,   "lr": base_lr * 0.01},
        {"params": layer1_params, "lr": base_lr * 0.02},
        {"params": layer2_params, "lr": base_lr * 0.05},
        {"params": layer3_params, "lr": base_lr * 0.1},
        {"params": layer4_params, "lr": base_lr * 0.5},
        {"params": head_params,   "lr": base_lr},
    ]
    # Filter out groups with no trainable params
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g["params"])]
    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ──────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────

def train_improved_finetune(
    ds,
    img_size:    int   = 224,
    epochs:      int   = 30,
    batch_size:  int   = 64,
    lr:          float = 3e-4,
    hidden_dim:  int   = 256,
    dropout:     float = 0.3,
    label_smooth: float = 0.1,
    device:      torch.device = torch.device("cpu"),
    save_dir:    Path  = Path("./experiments/idea1"),
    mean=None,
    std=None,
) -> dict:
    """
    Improved Approach 2 training loop.

    Gradual unfreezing schedule (split into thirds):
      Phase 1  epochs 1 .. P1          → head only
      Phase 2  epochs P1+1 .. P2       → head + layer4
      Phase 3  epochs P2+1 .. end      → full model
    """
    ensure_dir(save_dir)

    # ── Default norm stats ──────────────────────────────────────────────────
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Transforms & Loaders ───────────────────────────────────────────────
    train_tf, eval_tf = make_improved_transforms(img_size, mean, std)
    train_loader, val_loader, test_loader = make_loaders(
        ds, train_tf, eval_tf, batch_size=batch_size
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_backbone(BACKBONE, num_classes=NUM_CLASSES,
                           pretrained=True, device=device)
    model = _replace_fc_with_mlp(model, hidden=hidden_dim, dropout=dropout)
    model = model.to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    # 5. Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Gradual unfreeze phases ────────────────────────────────────────────
    phase1_end = max(1, epochs // 3)
    phase2_end = max(phase1_end + 1, (2 * epochs) // 3)
    print(f"Gradual unfreeze: phase1 1–{phase1_end} | "
          f"phase2 {phase1_end+1}–{phase2_end} | "
          f"phase3 {phase2_end+1}–{epochs}")

    # Start with head-only
    freeze_all_except_head(model)
    optimizer = build_optimizer(model, lr)

    # ── LR Scheduler: OneCycleLR (cosine + warmup) ─────────────────────────
    # Covers full training (steps_per_epoch × epochs). We rebuild when we
    # change the optimizer (phase transitions).
    steps_per_epoch = len(train_loader)

    def _make_scheduler(opt, remaining_epochs):
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[g["lr"] for g in opt.param_groups],
            steps_per_epoch=steps_per_epoch,
            epochs=remaining_epochs,
            pct_start=0.1,          # 10% warmup
            anneal_strategy="cos",
            div_factor=10.0,        # start at max_lr / 10
            final_div_factor=1e4,   # end at start_lr / 1e4
        )

    scheduler = _make_scheduler(optimizer, epochs)

    # ── Profiling metadata ─────────────────────────────────────────────────
    tag = f"{BACKBONE}_improved"
    ckpt_path = save_dir / f"{tag}.pt"
    results = {
        "approach":           "finetune_improved",
        "backbone":           BACKBONE,
        "epochs":             epochs,
        "batch_size":         batch_size,
        "base_lr":            lr,
        "label_smoothing":    label_smooth,
        "hidden_dim":         hidden_dim,
        "dropout":            dropout,
        "img_size":           img_size,
        "params_millions":    count_params(model) / 1e6,
        "gflops":             try_flops(model, img_size=img_size, device=device),
        "phase1_end":         phase1_end,
        "phase2_end":         phase2_end,
        "epoch_logs":         [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val = 0.0
    t0_total = time.time()

    for epoch in range(1, epochs + 1):

        # ── Gradual unfreeze transitions ───────────────────────────────────
        if epoch == phase1_end + 1:
            print(f"\n[Epoch {epoch}] → Unfreezing layer4 + head")
            unfreeze_layer4_and_head(model)
            optimizer  = build_optimizer(model, lr)
            remaining  = epochs - epoch + 1
            scheduler  = _make_scheduler(optimizer, remaining)

        elif epoch == phase2_end + 1:
            print(f"\n[Epoch {epoch}] → Full model fine-tuning")
            unfreeze_all(model)
            optimizer  = build_optimizer(model, lr)
            remaining  = epochs - epoch + 1
            scheduler  = _make_scheduler(optimizer, remaining)

        # ── Train one epoch ────────────────────────────────────────────────
        model.train()
        t_epoch = time.time()
        loss_accum, num_seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            # Gradient clipping avoids large spikes when unfreezing new layers
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_accum += loss.item() * y.size(0)
            num_seen   += y.size(0)

        train_loss = loss_accum / max(num_seen, 1)
        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
        log = {
            "epoch":          epoch,
            "phase":          1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3),
            "train_loss":     train_loss,
            "train_acc":      train_acc,
            "val_loss":       val_loss,
            "val_acc":        val_acc,
            "lr_head":        current_lr,
            "epoch_time_sec": epoch_time,
        }
        print(
            f"[{epoch:03d}/{epochs}] phase={log['phase']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.2e}  t={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

    results["total_train_time_sec"] = time.time() - t0_total

    # ── Test evaluation from best checkpoint ──────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["test_loss"] = test_loss
    print(f"\nBest val_acc={best_val:.3f}  →  test_acc={test_acc:.3f}  test_loss={test_loss:.4f}")

    # ── Per-image inference timing ─────────────────────────────────────────
    n_imgs, t_inf = 0, 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            t1 = time.time()
            _ = model(x)
            t_inf += time.time() - t1
            n_imgs += x.size(0)
    results["inference_time_per_image_ms"] = (t_inf / max(n_imgs, 1)) * 1000.0
    results["peak_gpu_mem_mb"] = (
        torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    out_path = save_dir / f"results_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results & checkpoint → {save_dir}")
    return results


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Approach 2 Improved: ResNet18 with all enhancements"
    )
    p.add_argument("--save_dir",      default="experiments/idea1")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--hidden_dim",    type=int,   default=256,
                   help="Hidden size of the MLP head")
    p.add_argument("--dropout",       type=float, default=0.3,
                   help="Dropout rate in MLP head")
    p.add_argument("--label_smooth",  type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=24)
    p.add_argument("--use_gpu",       action="store_true")
    p.add_argument("--subset",        type=int,   default=None,
                   help="Limit examples per split (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_seed(args.seed)
    device   = get_device(args.use_gpu)
    save_dir = Path(args.save_dir)

    print(f"Device: {device}  |  save_dir: {save_dir}")
    print(f"Backbone: {BACKBONE}  |  img_size: {args.img_size}  |  epochs: {args.epochs}")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("Loading dataset: timm/mini-imagenet …")
    ds = load_mini_imagenet(subset=args.subset)
    print({k: len(v) for k, v in ds.items() if k != "class_names"})

    # ── Load or fall back to ImageNet stats ───────────────────────────────
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
        print("Using ImageNet normalisation defaults.")

    train_improved_finetune(
        ds           = ds,
        img_size     = args.img_size,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        hidden_dim   = args.hidden_dim,
        dropout      = args.dropout,
        label_smooth = args.label_smooth,
        device       = device,
        save_dir     = save_dir,
        mean         = mean,
        std          = std,
    )
