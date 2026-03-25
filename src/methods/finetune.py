"""
methods/finetune.py
===================
Approach 2 — Pretrained Fine-tuning with Selective Freezing
------------------------------------------------------------
Fine-tune a pretrained backbone (ConvNeXt / ResNet / EfficientNet) on
Mini-ImageNet using one of three freeze policies:
  • backbone   — freeze all feature stages; train classifier head only
  • last_stage — freeze early stages; train the last stage + head
  • none        — fine-tune all parameters end-to-end

Usage (standalone):
    python -m methods.finetune --save_dir experiments/finetune --freeze_policy backbone
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..utils import ensure_dir, make_loaders
from ..model import (
    build_backbone, set_freeze_policy,
    count_params, try_flops, evaluate,
)


def train_finetune(
    ds,
    train_tf,
    eval_tf,
    device: torch.device,
    backbone: str = "convnext_tiny",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    freeze_policy: str = "backbone",
    use_pretrained: bool = True,
    save_dir: Path = Path("./outputs"),
) -> tuple:
    """
    Fine-tune (or train from scratch) a ConvNeXt-Tiny on the provided dataset.

    Args:
        ds:             HuggingFace DatasetDict.
        train_tf:       Training transform (may include augmentation).
        eval_tf:        Evaluation transform.
        device:         torch.device.
        epochs:         Number of training epochs.
        batch_size:     Mini-batch size.
        lr:             Learning rate for AdamW.
        freeze_policy:  'backbone' | 'last_stage' | 'none'.
        use_pretrained: Load ImageNet-pretrained weights when True.
        save_dir:       Directory for checkpoint and results JSON.

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
        pretrained=use_pretrained,
        device=device,
    )
    set_freeze_policy(model, freeze_policy)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    ce = nn.CrossEntropyLoss()

    tag = f"{backbone}_{freeze_policy}_{'pre' if use_pretrained else 'scratch'}"
    ckpt_path = save_dir / f"{tag}.pt"

    results = {
        "approach":        "finetune",
        "freeze_policy":   freeze_policy,
        "use_pretrained":  use_pretrained,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "lr":              lr,
        "params_millions": count_params(model) / 1e6,
        "gflops":          try_flops(model, device=device),
        "epoch_logs":      [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val = 0.0
    t0_total = time.time()

    for epoch in range(1, epochs + 1):
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

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        log = {
            "epoch":           epoch,
            "train_loss":      train_loss,
            "train_acc":       train_acc,
            "val_loss":        val_loss,
            "val_acc":         val_acc,
            "epoch_time_sec":  epoch_time,
        }
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"time={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

    results["total_train_time_sec"] = time.time() - t0_total

    # Per-image inference timing & peak GPU memory
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    n_imgs, t_inf = 0, 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            t1 = time.time()
            _ = model(x)
            t2 = time.time()
            t_inf += (t2 - t1)
            n_imgs += x.size(0)
    results["inference_time_per_image_ms"] = (t_inf / max(n_imgs, 1)) * 1000.0
    results["peak_gpu_mem_mb"] = (
        torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    # Final test accuracy from best checkpoint
    test_acc, test_loss = evaluate(model, test_loader, device)
    results["test_acc"]  = test_acc
    results["test_loss"] = test_loss

    out_path = save_dir / f"results_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved training results & checkpoint to: {save_dir}")
    return results, ckpt_path
