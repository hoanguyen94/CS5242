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
from tqdm.auto import tqdm

from ..utils import ensure_dir, make_loaders
from ..model import (
    build_backbone, set_freeze_policy,
    count_params, try_flops, evaluate,
    print_freeze_summary, extract_features_for_vis,
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
    patience: int = 3,
) -> tuple:
    """
    Fine-tune (or train from scratch) a backbone on the provided dataset.

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
        patience:       Early stopping patience (epochs without val improvement).

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
    print("Model parameters: ", model)
    
    set_freeze_policy(model, freeze_policy)
    print_freeze_summary(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    tag = f"{backbone}_{freeze_policy}_{'pre' if use_pretrained else 'scratch'}"
    ckpt_path = save_dir / f"{tag}.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = count_params(model)

    results = {
        "approach":        "finetune",
        "freeze_policy":   freeze_policy,
        "use_pretrained":  use_pretrained,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "lr":              lr,
        "params_millions": total / 1e6,
        "trainable_params_millions": trainable / 1e6,
        "gflops":          try_flops(model, device=device),
        "epoch_logs":      [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    best_val = 0.0
    epochs_no_improve = 0
    t0_total = time.time()
    representation_snapshots = []

    # Capture initial representations (before training)
    feats, labels = extract_features_for_vis(model, val_loader, device)
    representation_snapshots.append({"epoch": 0, "features": feats, "labels": labels})

    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch = time.time()
        loss_accum, correct, num_seen = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            num_seen += y.size(0)
            pbar.set_postfix(loss=f"{loss_accum/num_seen:.4f}", acc=f"{correct/num_seen:.3f}")

        scheduler.step()

        train_loss = loss_accum / max(num_seen, 1)
        train_acc = correct / max(num_seen, 1)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch

        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        log = {
            "epoch":           epoch,
            "train_loss":      train_loss,
            "train_acc":       train_acc,
            "val_loss":        val_loss,
            "val_acc":         val_acc,
            "lr":              optimizer.param_groups[0]["lr"],
            "epoch_time_sec":  epoch_time,
        }
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={log['lr']:.2e}  time={epoch_time:.1f}s"
        )
        results["epoch_logs"].append(log)

        # Capture representation snapshot at first, middle, and last epochs
        if epoch == 1 or epoch == max(1, epochs // 20) or epoch == epochs or epochs_no_improve >= patience:
            feats, labels = extract_features_for_vis(model, val_loader, device)
            # Avoid duplicate snapshots for the same epoch
            if not representation_snapshots or representation_snapshots[-1]["epoch"] != epoch:
                representation_snapshots.append({"epoch": epoch, "features": feats, "labels": labels})

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    results["total_train_time_sec"] = time.time() - t0_total
    results["epochs_trained"] = epoch

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

    print(f"\n{'='*40}")
    print(f"  Fine-tuning Results ({tag})")
    print(f"{'='*40}")
    print(f"  Best val accuracy  : {best_val:.4f}")
    print(f"  Test accuracy      : {test_acc:.4f}")
    print(f"  Test loss          : {test_loss:.4f}")
    print(f"  Epochs trained     : {epoch}/{epochs}")
    print(f"  Total train time   : {results['total_train_time_sec']:.1f}s")
    print(f"  Params (total)     : {total/1e6:.2f}M")
    print(f"  Params (trainable) : {trainable/1e6:.2f}M")
    print(f"  Inference/image    : {results['inference_time_per_image_ms']:.2f}ms")
    if results["peak_gpu_mem_mb"]:
        print(f"  Peak GPU memory    : {results['peak_gpu_mem_mb']:.0f}MB")
    print(f"{'='*40}")
    print(f"Saved results → {out_path}")
    results["representation_snapshots"] = representation_snapshots
    return results, ckpt_path
