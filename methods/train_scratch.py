"""
methods/train_scratch.py
========================
Approach 3 — Training from Scratch (Baseline)
----------------------------------------------
Train a backbone (ConvNeXt / ResNet / EfficientNet) from randomly
initialised weights (no pretrained weights) end-to-end on Mini-ImageNet.
This serves as the baseline to measure how much benefit pretrained weights provide.

Internally re-uses `train_finetune` from methods/finetune.py with

Usage (standalone):
    python -m methods.train_scratch --save_dir experiments/scratch
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from utils import (
    ensure_dir, load_mini_imagenet, make_transforms, make_loaders,
    set_seed, get_device,
)
# from model_utils import build_backbone
from model import count_params, try_flops, evaluate, build_backbone


def train_from_scratch(
    ds,
    train_tf,
    eval_tf,
    device,
    backbone: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    lr_scheduler: str = "cosine",
    warmup_epochs: int = 1,
    save_dir: Path = Path("./outputs"),
) -> tuple:
    """
    Train a ConvNeXt-Tiny from random initialisation (no pretrained weights).

    Args:
        ds:          HuggingFace DatasetDict.
        train_tf:    Training transform (may include augmentation).
        eval_tf:     Evaluation transform.
        device:      torch.device.
        backbone:    Name of the backbone model to train.
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        lr:          Learning rate for AdamW.
        lr_scheduler: Type of LR scheduler to use ('cosine', 'step', 'none').
        warmup_epochs: Number of epochs for linear LR warmup.
        save_dir:    Directory for checkpoint and results JSON.

    Returns:
        (results dict, path to best checkpoint)
    """
    ensure_dir(save_dir)
    train_loader, val_loader, test_loader = make_loaders(
        ds, train_tf, eval_tf, batch_size=batch_size
    )

    model = build_backbone(
        backbone=backbone,
        num_classes=100,
        pretrained=False,  # Always False for training from scratch
        device=device,
    )

    params_millions = count_params(model) / 1e6
    print(f"Model: {backbone} | Params: {params_millions:.2f}M")


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    # Create LR scheduler
    scheduler = None
    if lr_scheduler == "cosine":
        # Start scheduler after warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, (epochs - warmup_epochs) // 3), gamma=0.1)

    tag = f"{backbone}_scratch"
    ckpt_path = save_dir / f"{tag}.pt"

    results = {
        "approach":        "scratch",
        "backbone":        backbone,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "lr":              lr,
        "params_millions": params_millions,
        "gflops":          try_flops(model, device=device),
        "epoch_logs":      [],
    }

    best_val = 0.0
    t0_total = time.time()

    for epoch in range(1, epochs + 1):
        # --- LR Warmup ---
        if epoch <= warmup_epochs:
            # Linearly increase learning rate from a small value to the target LR
            lr_scale = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale

        model.train()
        t_epoch = time.time()
        loss_accum, num_seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item() * y.size(0)
            num_seen += y.size(0)

        train_loss = loss_accum / max(num_seen, 1)
        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch

        # --- Step Scheduler ---
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val,
            }
            torch.save(checkpoint, ckpt_path)

        current_lr = optimizer.param_groups[0]['lr']
        log = {
            "epoch":           epoch,
            "train_loss":      train_loss,
            "train_acc":       train_acc,
            "val_loss":        val_loss,
            "val_acc":         val_acc,
            "lr":              current_lr,
            "epoch_time_sec":  epoch_time,
        }
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"lr={current_lr:.6f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"time={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

    results["total_train_time_sec"] = time.time() - t0_total

    # Final test accuracy from best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inference time on validation set
    n_imgs, t_inf = 0, 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device, non_blocking=True)
            t1 = time.time()
            _ = model(x)
            t2 = time.time()
            t_inf += (t2 - t1)
            n_imgs += x.size(0)
    inference_time_ms = (t_inf / max(n_imgs, 1)) * 1000.0
    results["inference_time_per_image_ms"] = inference_time_ms
    print(f"Inference time (validation): {inference_time_ms:.2f} ms/image")


    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["test_loss"] = test_loss


    out_path = save_dir / f"results_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved training results & checkpoint to: {save_dir}")
    return results, ckpt_path


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Approach 3: Train models from scratch")
    p.add_argument("--save_dir",   default="experiments/scratch")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "step", "none"],
                   help="Learning rate scheduler type.")
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="Number of epochs for linear learning rate warmup.")
    p.add_argument("--img_size",   type=int,   default=32)
    p.add_argument("--backbone",   default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1", "resnet18_scratch", "convnext_tiny_scratch"])
    p.add_argument("--use_aug",    action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--use_gpu",    action="store_true")
    p.add_argument("--subset",     type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_seed(args.seed)
    device   = get_device(args.use_gpu)
    save_dir = Path(args.save_dir)

    print("Loading dataset …")
    ds = load_mini_imagenet(subset=args.subset)

    train_tf, eval_tf, _ = make_transforms(img_size=args.img_size, use_aug=args.use_aug)

    train_from_scratch(
        ds=ds,
        train_tf=train_tf,
        eval_tf=eval_tf,
        device=device,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        save_dir=save_dir,
    )
