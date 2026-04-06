"""
approach2_idea1_v5.py — Improved Approach 2 v5: ConvNeXt-Tiny Architecture
========================================================================
We migrate to the extremely performant ConvNeXt-Tiny backbone to naturally bypass
the catastrophic forgetting issues of ResNet18.

Changes over v4:
  [A] Changed backbone to `convnext_tiny` (768 dim).
  [B] Reworked custom MLP interceptor to target `model.classifier[-1]`.
  [C] Reworked parameter decay grouping to map to ConvNeXt's `model.features` stages.
  [D] Replaced BatchNorm with LayerNorm in the custom MLP head.
  [E] Completely removed `set_bn_eval` as ConvNeXt exclusively uses LayerNorm.
  
Usage
-----
  python approach2_idea1_v5.py --save_dir experiments/idea1_v5 --device mps
  python3 approach2_idea1_v5.py --save_dir experiments/idea1_v5 --device mps --epochs 40
"""

import argparse
import json
import time
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
# Constants & Helpers
# ──────────────────────────────────────────────

NUM_CLASSES = 100
BACKBONE    = "convnext_tiny"


def get_ema_avg_fn(decay=0.999):
    def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
        return decay * averaged_model_parameter + (1.0 - decay) * model_parameter
    return ema_avg


# ──────────────────────────────────────────────
# 1. Richer Classifier Head (ConvNeXt Version)
# ──────────────────────────────────────────────

def _replace_classifier_with_mlp(model: nn.Module, hidden: int = 256, dropout: float = 0.3) -> nn.Module:
    """
    ConvNeXt stores its head at `model.classifier`:
      Sequential(LayerNorm2d(768), Flatten(1), Linear(768, 1000))
    We replace the final Linear with a deeper MLP using LayerNorm.
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.LayerNorm(hidden),  # Match ConvNeXt design ethos (no BN)
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, NUM_CLASSES)
            )
    return model


# ──────────────────────────────────────────────
# 2. Transforms 
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
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_tf, eval_tf


# ──────────────────────────────────────────────
# 3. Mixup helper
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
# 4. Freeze & Unfreeze Helpers (ConvNeXt Version)
# ──────────────────────────────────────────────

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_all_except_head(model: nn.Module) -> None:
    _set_grad(model, False)
    if hasattr(model, "classifier"):
        _set_grad(model.classifier, True)


def unfreeze_all(model: nn.Module) -> None:
    _set_grad(model, True)


# ──────────────────────────────────────────────
# 5. Layer Decay Optimizer (ConvNeXt Version)
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, base_lr: float, is_phase2: bool = False,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Groups ConvNeXt stages and applies an ultra-protective learning rate
    to the earliest feature extraction blocks.
    """
    head_params = list(model.classifier.parameters())
    
    if not is_phase2:
        param_groups = [{"params": head_params, "lr": base_lr}]
    else:
        # ConvNeXt is organised into model.features[0..7]
        stem_params   = list(model.features[0].parameters())
        stage0_params = list(model.features[1].parameters()) + list(model.features[2].parameters())
        stage1_params = list(model.features[3].parameters()) + list(model.features[4].parameters())
        stage2_params = list(model.features[5].parameters()) + list(model.features[6].parameters())
        stage3_params = list(model.features[7].parameters())
        
        param_groups = [
            {"params": stem_params,   "lr": base_lr * 0.001},
            {"params": stage0_params, "lr": base_lr * 0.001},
            {"params": stage1_params, "lr": base_lr * 0.005},
            {"params": stage2_params, "lr": base_lr * 0.01},
            {"params": stage3_params, "lr": base_lr * 0.05},
            {"params": head_params,   "lr": base_lr},
        ]
        
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g["params"])]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────

def train_improved_finetune_v5(
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
    save_dir:      Path  = Path("./experiments/idea1_v5"),
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
    # We call standard build_backbone (from model.py) first, then our custom interceptor
    model = build_backbone(BACKBONE, num_classes=NUM_CLASSES, pretrained=True, device=device)
    model = _replace_classifier_with_mlp(model, hidden=hidden_dim, dropout=dropout)
    model = model.to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Gradual unfreeze: 2-phases ─────────────────────────────────────────
    phase1_end = max(1, epochs // 3)
    print(f"2-Phase Gradual Unfreeze: phase1 1–{phase1_end} (head) | phase2 {phase1_end+1}–{epochs} (all + EMA)")

    freeze_all_except_head(model)
    optimizer = build_optimizer(model, lr, is_phase2=False, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * phase1_end, eta_min=1e-6
    )

    # ── Continuous EMA Model ─────────────────────────────────────────────
    ema_model = AveragedModel(model, avg_fn=get_ema_avg_fn(0.999))

    # ── Metadata ──────────────────────────────────────────────────────────
    tag = f"{BACKBONE}_improved_v5"
    ckpt_path = save_dir / f"{tag}.pt"
    ema_ckpt_path = save_dir / f"{tag}_ema.pt"
    results = {
        "approach":             "finetune_improved_v5",
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
        "early_stop_patience":  early_stop_pat,
        "phase1_end":           phase1_end,
        "epoch_logs":           [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val_base = 0.0
    best_val_ema = 0.0
    no_improve_count = 0
    t0_total = time.time()

    for epoch in range(1, epochs + 1):
        phase = 1 if epoch <= phase1_end else 2

        # ── Phase 2 Transition ─────────────────────────────────────────────
        if epoch == phase1_end + 1:
            print(f"\n[Epoch {epoch}] → Full model fine-tuning (ultra-low backbone LR)")
            unfreeze_all(model)
            optimizer = build_optimizer(model, lr, is_phase2=True, weight_decay=weight_decay)
            remaining_epochs = epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps_per_epoch * remaining_epochs, eta_min=1e-6
            )

        # ── Train one epoch ────────────────────────────────────────────────
        model.train()
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
            
            # EMA continuous update every batch
            ema_model.update_parameters(model)

            loss_accum += loss.item() * y.size(0)
            num_seen   += y.size(0)

        train_loss = loss_accum / max(num_seen, 1)
        
        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        # Evaluate EMA model 
        ema_val_acc, _ = evaluate(ema_model, val_loader, device)
        
        epoch_time = time.time() - t_epoch
        current_lr = scheduler.get_last_lr()[-1] if hasattr(scheduler, "get_last_lr") else lr
        
        log = {
            "epoch":          epoch,
            "phase":          phase,
            "train_loss":     train_loss,
            "train_acc":      train_acc,
            "val_acc":        val_acc,
            "ema_val_acc":    ema_val_acc,
            "lr_head":        current_lr,
            "epoch_time_sec": epoch_time,
        }
        print(
            f"[{epoch:03d}/{epochs}] phase={phase}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.3f}  [EMA val={ema_val_acc:.3f}]  "
            f"lr={current_lr:.2e}  t={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

        # ── Standard model checkpoint ──────────────────────────────────────
        improved = False
        if val_acc > best_val_base:
            best_val_base = val_acc
            torch.save(model.state_dict(), ckpt_path)
            improved = True
            
        if ema_val_acc > best_val_ema:
            best_val_ema = ema_val_acc
            torch.save(ema_model.state_dict(), ema_ckpt_path)
            improved = True
            
        if improved:
            no_improve_count = 0
            print(f"           ✓ new best checkpoints saved (base: {best_val_base:.4f}, ema: {best_val_ema:.4f})")
        else:
            no_improve_count += 1

        if phase == 2 and no_improve_count >= early_stop_pat:
            print(f"\n[Early stopping] No improvement to base or EMA for {early_stop_pat} epochs.")
            results["stopped_early"] = True
            results["early_stop_epoch"] = epoch
            break
    else:
        results["stopped_early"] = False

    results["total_train_time_sec"] = time.time() - t0_total

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n--- Final Evaluations ---")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["best_val_base"] = best_val_base
    print(f"[Base Model] Best val_acc={best_val_base:.3f}  →  test_acc={test_acc:.3f}")

    ema_model.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
    ema_model.eval()
    ema_test_acc, _ = evaluate(ema_model, test_loader, device)
    results["ema_test_acc"] = ema_test_acc
    results["best_val_ema"] = best_val_ema
    print(f"[EMA Model]  Best val_acc={best_val_ema:.3f}  →  test_acc={ema_test_acc:.3f}")

    out_path = save_dir / f"results_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results & checkpoints → {save_dir}")
    return results


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Approach 2 v5: ConvNeXt-Tiny Architect")
    p.add_argument("--save_dir",      default="experiments/idea1_v5")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--label_smooth",  type=float, default=0.10)
    p.add_argument("--mixup_alpha",   type=float, default=0.2)
    p.add_argument("--weight_decay",  type=float, default=1e-4) # 1e-4 is safe for finetuning
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

    train_improved_finetune_v5(
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
