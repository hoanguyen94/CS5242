"""
approach2_idea1_v3.py — Improved Approach 2 v3: Regularisation "Goldilocks"
========================================================================
Achieving the balance between memorisation (v1) and underfitting (v2).
We freeze backbone BatchNorm stats, use SWA, and dial back data distortion.

Changes over v2:
  [A] Kept Mixup (alpha=0.2) + Label Smoothing (0.1).
  [B] Dialled back RandAugment: magnitude 9 → 5.
  [C] Dialled back Weight Decay: 5e-4 → 1e-4.
  [D] Dialled back Dropout: 0.5 → 0.3.
  [E] Freeze Backbone BN: Added set_bn_eval to keep BatchNorm2d in eval().
  [F] SWA: Added Stochastic Weight Averaging over the final 5 epochs.

Usage
-----
  python approach2_idea1_v3.py --save_dir experiments/idea1_v3 --device mps --epochs 40
"""

import argparse
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel

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
# 1. Richer Classifier Head
# ──────────────────────────────────────────────

def _replace_fc_with_mlp(model: nn.Module, hidden: int = 256, dropout: float = 0.3) -> nn.Module:
    """
    Replace fc with a BN-MLP (dropout default dialled back to 0.3).
    """
    if hasattr(model, "fc"):
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
# 2. Transforms (RandAugment mag=5)
# ──────────────────────────────────────────────

def make_improved_transforms(img_size: int, mean, std, randaug_mag: int = 5):
    _mean = tuple(mean.tolist() if hasattr(mean, "tolist") else mean)
    _std  = tuple(std.tolist()  if hasattr(std,  "tolist") else std)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=randaug_mag),
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
# 3. Freeze / Unfreeze & BN Helpers
# ──────────────────────────────────────────────

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_all_except_head(model: nn.Module) -> None:
    _set_grad(model, False)
    if hasattr(model, "fc"):
        _set_grad(model.fc, True)


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    _set_grad(model, False)
    if hasattr(model, "layer4"):
        _set_grad(model.layer4, True)
    if hasattr(model, "fc"):
        _set_grad(model.fc, True)


def unfreeze_all(model: nn.Module) -> None:
    _set_grad(model, True)


def set_bn_eval(module: nn.Module):
    """
    Forces all BatchNorm2d layers (spatial backbone BN) into eval mode.
    They won't update their running mean/variance on small distorted batches.
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ──────────────────────────────────────────────
# 4. Differential LR Optimizer
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, base_lr: float,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
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

def train_improved_finetune_v3(
    ds,
    img_size:      int   = 224,
    epochs:        int   = 40,
    batch_size:    int   = 64,
    lr:            float = 3e-4,
    hidden_dim:    int   = 256,
    dropout:       float = 0.3,           
    label_smooth:  float = 0.10,          
    mixup_alpha:   float = 0.2,           
    weight_decay:  float = 1e-4,          
    randaug_mag:   int   = 5,             
    early_stop_pat:int   = 7,
    device:        torch.device = torch.device("cpu"),
    save_dir:      Path  = Path("./experiments/idea1_v3"),
    mean=None,
    std=None,
) -> dict:
    ensure_dir(save_dir)

    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Transforms & Loaders ───────────────────────────────────────────────
    train_tf, eval_tf = make_improved_transforms(img_size, mean, std, randaug_mag)
    train_loader, val_loader, test_loader = make_loaders(
        ds, train_tf, eval_tf, batch_size=batch_size
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_backbone(BACKBONE, num_classes=NUM_CLASSES,
                           pretrained=True, device=device)
    model = _replace_fc_with_mlp(model, hidden=hidden_dim, dropout=dropout)
    model = model.to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
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

    # ── SWA Model ─────────────────────────────────────────────────────────
    swa_model = AveragedModel(model)
    swa_start = max(epochs - 5, phase2_end + 1) # last 5 epochs of Phase 3

    # ── Metadata ──────────────────────────────────────────────────────────
    tag = f"{BACKBONE}_improved_v3"
    ckpt_path = save_dir / f"{tag}.pt"
    swa_ckpt_path = save_dir / f"{tag}_swa.pt"
    results = {
        "approach":             "finetune_improved_v3",
        "backbone":             BACKBONE,
        "epochs_requested":     epochs,
        "batch_size":           batch_size,
        "base_lr":              lr,
        "label_smoothing":      label_smooth,
        "mixup_alpha":          mixup_alpha,
        "weight_decay":         weight_decay,
        "randaug_mag":          randaug_mag,
        "hidden_dim":           hidden_dim,
        "dropout":              dropout,
        "img_size":             img_size,
        "early_stop_patience":  early_stop_pat,
        "params_millions":      count_params(model) / 1e6,
        "gflops":               try_flops(model, img_size=img_size, device=device),
        "phase1_end":           phase1_end,
        "phase2_end":           phase2_end,
        "swa_start":            swa_start,
        "epoch_logs":           [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val = 0.0
    no_improve_count = 0
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
        # [E] CRITICAL: Freeze BN running stats so they don't break
        set_bn_eval(model)

        t_epoch = time.time()
        loss_accum, num_seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
        
        # [F] SWA Parameter update
        phase = 1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3)
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
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
            + ("  [SWA]" if epoch >= swa_start else "")
        )
        results["epoch_logs"].append(log)

        # ── Standard model checkpoint ──────────────────────────────────────
        if val_acc > best_val:
            best_val = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"           ✓ new best val_acc={best_val:.4f} — checkpoint saved")
        else:
            no_improve_count += 1

        if phase == 3 and no_improve_count >= early_stop_pat:
            print(f"\n[Early stopping] No improvement for {early_stop_pat} epochs.")
            results["stopped_early"] = True
            results["early_stop_epoch"] = epoch
            break
    else:
        results["stopped_early"] = False

    results["total_train_time_sec"] = time.time() - t0_total

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n--- Final Evaluations ---")
    
    # 1. Base Model Checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["test_loss"] = test_loss
    results["best_val_acc"] = best_val
    print(f"[Base Model] Best val_acc={best_val:.3f}  →  test_acc={test_acc:.3f}")

    # 2. SWA Model Checkpoint (if SWA was active)
    # We save it regardless to avoid errors.
    torch.save(swa_model.state_dict(), swa_ckpt_path)
    swa_model.eval()
    swa_val_acc, swa_val_loss = evaluate(swa_model, val_loader, device)
    swa_test_acc, swa_test_loss = evaluate(swa_model, test_loader, device)
    results["swa_val_acc"]  = swa_val_acc
    results["swa_test_acc"] = swa_test_acc
    print(f"[SWA Model]  val_acc={swa_val_acc:.3f}  →  test_acc={swa_test_acc:.3f}")

    # Inference timing (base model)
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
    print(f"Saved results & checkpoints → {save_dir}")
    return results


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Approach 2 v3: Goldilocks Regularisation + SWA")
    p.add_argument("--save_dir",      default="experiments/idea1_v3")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--label_smooth",  type=float, default=0.10)
    p.add_argument("--mixup_alpha",   type=float, default=0.2)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--randaug_mag",   type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=7)
    p.add_argument("--seed",          type=int,   default=24)
    p.add_argument("--device",        default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--subset",        type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_seed(args.seed)
    device   = torch.device(args.device)
    save_dir = Path(args.save_dir)

    print(f"Device: {device}  |  save_dir: {save_dir}")
    print(f"Backbone: {BACKBONE}  |  img_size: {args.img_size}  |  epochs: {args.epochs}")
    print(f"[v3] randaug={args.randaug_mag}  dropout={args.dropout}  wd={args.weight_decay} "
          f"mixup={args.mixup_alpha} ls={args.label_smooth}")

    print("Loading dataset: timm/mini-imagenet …")
    ds = load_mini_imagenet(subset=args.subset)
    print({k: len(v) for k, v in ds.items() if k != "class_names"})

    mean_std_path = save_dir / "train_mean_std.json"
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

    train_improved_finetune_v3(
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
        randaug_mag          = args.randaug_mag,
        early_stop_pat       = args.patience,
        device               = device,
        save_dir             = save_dir,
        mean                 = mean,
        std                  = std,
    )
