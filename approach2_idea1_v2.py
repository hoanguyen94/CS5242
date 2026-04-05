"""
approach2_idea1_v2.py — Improved Approach 2 v2: Anti-Overfitting Edition
=========================================================================
v1 achieved train_acc=0.977 but test_acc=0.878 (~9.9% generalisation gap).
All changes here are aimed at closing that gap.

Changes over v1 (approach2_idea1.py):
  [A] Mixup augmentation (alpha=0.2) in training loop
        → Forces the model to interpolate between examples; the single
          biggest cure for memorisation in fine-tuning.
  [B] Stronger weight decay:  1e-4  →  5e-4
        → More L2 penalty on all 11M parameters during full fine-tuning.
  [C] Higher label smoothing: 0.1   →  0.15
        → Prevents overconfident logits; works synergistically with Mixup.
  [D] Phase 3 scheduler fix: final_div_factor 1e4 → 100
        → Stops the LR collapsing to ~3e-11; keeps gradient signal alive.
  [E] Early stopping (patience=7 on val_acc)
        → Checkpoints the best model and halts training before the network
          memorises phase-3 training examples.
  [F] Higher dropout in the MLP head: 0.3 → 0.5 (configurable)
        → Extra regularisation where it costs nothing in compute.

Usage
-----
  python approach2_idea1_v2.py --save_dir experiments/idea1_v2 --device mps
  python approach2_idea1_v2.py --save_dir experiments/idea1_v2 --device mps --epochs 40
  python approach2_idea1_v2.py --save_dir experiments/idea1_v2 --device cuda
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
    ensure_dir, load_mini_imagenet, set_seed,
    make_loaders, HFDatasetWrapper,
)
from model import build_backbone, count_params, try_flops, evaluate


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

NUM_CLASSES  = 100
BACKBONE     = "resnet18"
RESNET18_DIM = 512


# ──────────────────────────────────────────────
# 1. Richer Classifier Head  (same as v1, dropout raised)
# ──────────────────────────────────────────────

def _replace_fc_with_mlp(model: nn.Module, hidden: int = 256, dropout: float = 0.5) -> nn.Module:
    """
    Replace fc with a BN-MLP:
      Linear(512, hidden) → BN → ReLU → Dropout(p) → Linear(hidden, 100)
    dropout default raised to 0.5 vs 0.3 in v1.
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
# 2. Transforms  (same spatial pipeline as v1)
# ──────────────────────────────────────────────

def make_improved_transforms(img_size: int, mean, std):
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
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_tf, eval_tf


# ──────────────────────────────────────────────
# [A] Mixup helper
# ──────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, device=None):
    """Return mixed inputs and a lambda value.

    Returns (mixed_x, y_a, y_b, lam) for use with mixup_criterion.
    When alpha=0 mixup is effectively disabled (lam=1 always).
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────
# 3. Freeze / Unfreeze Helpers  (identical to v1)
# ──────────────────────────────────────────────

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_all_except_head(model: nn.Module) -> None:
    _set_grad(model, False)
    _set_grad(model.fc, True)


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    _set_grad(model, False)
    _set_grad(model.layer4, True)
    _set_grad(model.fc, True)


def unfreeze_all(model: nn.Module) -> None:
    _set_grad(model, True)


# ──────────────────────────────────────────────
# 4. Differential LR Optimizer  (weight_decay raised: [B])
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, base_lr: float,
                    weight_decay: float = 5e-4) -> torch.optim.Optimizer:
    """
    Differential LRs with stronger weight decay (5e-4, up from 1e-4 in v1).
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
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g["params"])]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────

def train_improved_finetune_v2(
    ds,
    img_size:      int   = 224,
    epochs:        int   = 40,
    batch_size:    int   = 64,
    lr:            float = 3e-4,
    hidden_dim:    int   = 256,
    dropout:       float = 0.5,           # [F] raised from 0.3
    label_smooth:  float = 0.15,          # [C] raised from 0.10
    mixup_alpha:   float = 0.2,           # [A] new
    weight_decay:  float = 5e-4,          # [B] raised from 1e-4
    early_stop_patience: int = 7,         # [E] new
    device:        torch.device = torch.device("cpu"),
    save_dir:      Path  = Path("./experiments/idea1_v2"),
    mean=None,
    std=None,
) -> dict:
    """
    Improved v2 training loop with anti-overfitting measures.

    Gradual unfreezing (same 3-phase schedule as v1):
      Phase 1  epochs 1 .. P1         → head only
      Phase 2  epochs P1+1 .. P2      → head + layer4
      Phase 3  epochs P2+1 .. end     → full model  +  early stopping
    """
    ensure_dir(save_dir)

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

    # ── Loss  ([C] higher label smoothing) ────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Gradual unfreeze phases ────────────────────────────────────────────
    phase1_end = max(1, epochs // 3)
    phase2_end = max(phase1_end + 1, (2 * epochs) // 3)
    print(f"Gradual unfreeze: phase1 1–{phase1_end} | "
          f"phase2 {phase1_end+1}–{phase2_end} | "
          f"phase3 {phase2_end+1}–{epochs}")

    freeze_all_except_head(model)
    optimizer = build_optimizer(model, lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)

    def _make_scheduler(opt, remaining_epochs, is_phase3=False):
        # [D] Phase-3 fix: final_div_factor reduced from 1e4 → 100
        #     so the LR floor stays at ~start_lr/100 not ~3e-11.
        final_div = 100.0 if is_phase3 else 1e4
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[g["lr"] for g in opt.param_groups],
            steps_per_epoch=steps_per_epoch,
            epochs=remaining_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=final_div,
        )

    scheduler = _make_scheduler(optimizer, epochs)

    # ── Metadata ──────────────────────────────────────────────────────────
    tag = f"{BACKBONE}_improved_v2"
    ckpt_path = save_dir / f"{tag}.pt"
    results = {
        "approach":             "finetune_improved_v2",
        "backbone":             BACKBONE,
        "epochs_requested":     epochs,
        "batch_size":           batch_size,
        "base_lr":              lr,
        "label_smoothing":      label_smooth,
        "mixup_alpha":          mixup_alpha,
        "weight_decay":         weight_decay,
        "hidden_dim":           hidden_dim,
        "dropout":              dropout,
        "img_size":             img_size,
        "early_stop_patience":  early_stop_patience,
        "params_millions":      count_params(model) / 1e6,
        "gflops":               try_flops(model, img_size=img_size, device=device),
        "phase1_end":           phase1_end,
        "phase2_end":           phase2_end,
        "epoch_logs":           [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val = 0.0
    no_improve_count = 0   # for early stopping
    t0_total = time.time()

    for epoch in range(1, epochs + 1):

        # ── Phase transitions ──────────────────────────────────────────────
        if epoch == phase1_end + 1:
            print(f"\n[Epoch {epoch}] → Unfreezing layer4 + head")
            unfreeze_layer4_and_head(model)
            optimizer  = build_optimizer(model, lr, weight_decay=weight_decay)
            remaining  = epochs - epoch + 1
            scheduler  = _make_scheduler(optimizer, remaining)

        elif epoch == phase2_end + 1:
            print(f"\n[Epoch {epoch}] → Full model fine-tuning")
            unfreeze_all(model)
            optimizer  = build_optimizer(model, lr, weight_decay=weight_decay)
            remaining  = epochs - epoch + 1
            scheduler  = _make_scheduler(optimizer, remaining, is_phase3=True)

        # ── Train one epoch ────────────────────────────────────────────────
        model.train()
        t_epoch = time.time()
        loss_accum, num_seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # [A] Mixup
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_mix)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_accum += loss.item() * y.size(0)
            num_seen   += y.size(0)

        train_loss = loss_accum / max(num_seen, 1)
        # evaluate without mixup for honest accuracy numbers
        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
        phase = 1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3)
        log = {
            "epoch":          epoch,
            "phase":          phase,
            "train_loss":     train_loss,
            "train_acc":      train_acc,
            "val_loss":       val_loss,
            "val_acc":        val_acc,
            "lr_head":        current_lr,
            "epoch_time_sec": epoch_time,
        }
        print(
            f"[{epoch:03d}/{epochs}] phase={phase}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.2e}  t={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

        # ── Best checkpoint + early stopping ──────────────────────────────
        if val_acc > best_val:
            best_val = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"           ✓ new best val_acc={best_val:.4f} — checkpoint saved")
        else:
            no_improve_count += 1
            print(f"           · no improvement ({no_improve_count}/{early_stop_patience})")

        # [E] Only apply early stopping in Phase 3 (Phase 1/2 transitions
        #     naturally cause a temporary val dip — don't stop there).
        if phase == 3 and no_improve_count >= early_stop_patience:
            print(f"\n[Early stopping] No val_acc improvement for "
                  f"{early_stop_patience} epochs — stopping at epoch {epoch}.")
            results["stopped_early"] = True
            results["early_stop_epoch"] = epoch
            break
    else:
        results["stopped_early"] = False

    results["total_train_time_sec"] = time.time() - t0_total

    # ── Test evaluation from best checkpoint ──────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["test_loss"] = test_loss
    results["best_val_acc"] = best_val
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
        description="Approach 2 v2: Anti-overfitting improvements over idea1"
    )
    p.add_argument("--save_dir",      default="experiments/idea1_v2")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.5,
                   help="Dropout in MLP head (v2 default=0.5, up from 0.3)")
    p.add_argument("--label_smooth",  type=float, default=0.15,
                   help="Label smoothing (v2 default=0.15, up from 0.10)")
    p.add_argument("--mixup_alpha",   type=float, default=0.2,
                   help="Mixup alpha; 0 disables Mixup")
    p.add_argument("--weight_decay",  type=float, default=5e-4,
                   help="AdamW weight decay (v2 default=5e-4, up from 1e-4)")
    p.add_argument("--patience",      type=int,   default=7,
                   help="Early stopping patience (val_acc, Phase 3 only)")
    p.add_argument("--seed",          type=int,   default=24)
    p.add_argument("--device",        default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--subset",        type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_seed(args.seed)
    device   = torch.device(args.device)
    save_dir = Path(args.save_dir)

    print(f"Device: {device}  |  save_dir: {save_dir}")
    print(f"Backbone: {BACKBONE}  |  img_size: {args.img_size}  |  epochs: {args.epochs}")
    print(f"[v2 changes] mixup_alpha={args.mixup_alpha}  weight_decay={args.weight_decay}  "
          f"dropout={args.dropout}  label_smooth={args.label_smooth}  "
          f"early_stop_patience={args.patience}")

    print("Loading dataset: timm/mini-imagenet …")
    ds = load_mini_imagenet(subset=args.subset)
    print({k: len(v) for k, v in ds.items() if k != "class_names"})

    mean_std_path = save_dir / "train_mean_std.json"
    # Fall back to idea1's pre-computed stats if available
    idea1_path = Path("experiments/idea1/train_mean_std.json")
    for candidate in [mean_std_path, idea1_path]:
        if candidate.exists():
            with open(candidate) as f:
                ms = json.load(f)
            mean = np.array(ms["mean"], dtype=np.float32)
            std  = np.array(ms["std"],  dtype=np.float32)
            print(f"Loaded normalisation stats from {candidate}.")
            break
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print("Using ImageNet normalisation defaults.")

    train_improved_finetune_v2(
        ds                   = ds,
        img_size             = args.img_size,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        hidden_dim           = args.hidden_dim,
        dropout              = args.dropout,
        label_smooth         = args.label_smooth,
        mixup_alpha          = args.mixup_alpha,
        weight_decay         = args.weight_decay,
        early_stop_patience  = args.patience,
        device               = device,
        save_dir             = save_dir,
        mean                 = mean,
        std                  = std,
    )
