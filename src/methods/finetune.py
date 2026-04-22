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
  • lora       — freeze all weights; inject low-rank adapters into Linear layers

Usage (standalone):
    python -m methods.finetune --save_dir experiments/finetune --freeze_policy backbone
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..utils import ensure_dir, make_loaders
from ..model import (
    build_backbone, set_freeze_policy,
    count_params, try_flops, evaluate,
    print_freeze_summary, extract_features_for_vis,
    measure_pytorch_inference_time_ms,
)


# ──────────────────────────────────────────────
# LoRA (Low-Rank Adaptation)
# ──────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    Output = W_frozen @ x + (B @ A) @ x * (alpha / rank)
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = original.in_features, original.out_features
        dev = original.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=dev) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=dev))

        # Freeze the original weight
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Freeze all params, then inject LoRA adapters into Linear layers in the backbone.

    Returns the number of layers replaced.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and "classifier" not in name and "fc" not in name and "head" not in name:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1

    # Always unfreeze the classifier head
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    print(f"LoRA applied: {replaced} Linear layers adapted (rank={rank}, alpha={alpha})")
    return replaced


# ──────────────────────────────────────────────
# Mixup / CutMix
# ──────────────────────────────────────────────

def mixup(x, y, alpha=0.2):
    """Mixup: blend two random images and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def cutmix(x, y, alpha=1.0):
    """CutMix: paste a random patch from one image onto another."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape

    # Random box
    cut_ratio = np.sqrt(1 - lam)
    rh, rw = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - rh // 2, 0, H)
    y2 = np.clip(cy + rh // 2, 0, H)
    x1 = np.clip(cx - rw // 2, 0, W)
    x2 = np.clip(cx + rw // 2, 0, W)

    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_cut, y, y[idx], lam


def mix_criterion(criterion, logits, y_a, y_b, lam):
    """Compute mixed loss for Mixup/CutMix."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def _transform_img_size(eval_tf, default: int = 224) -> int:
    """Best-effort extraction of image size from a torchvision transform."""
    if hasattr(eval_tf, "transforms"):
        for transform in eval_tf.transforms:
            if hasattr(transform, "size"):
                size = transform.size
                return size if isinstance(size, int) else size[0]
    return default


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
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    mix_mode: str = "none",
    mix_alpha: float = 0.2,
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
    # print("Model parameters: ", model)
    
    if freeze_policy == "lora":
        apply_lora(model, rank=lora_rank, alpha=lora_alpha)
    else:
        set_freeze_policy(model, freeze_policy)
    print_freeze_summary(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    mix_tag = f"_{mix_mode}" if mix_mode != "none" else ""
    tag = f"{backbone}_{freeze_policy}_{'pre' if use_pretrained else 'scratch'}{mix_tag}"
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
        "mix_mode":        mix_mode,
        "mix_alpha":       mix_alpha if mix_mode != "none" else None,
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

            if mix_mode == "mixup":
                x, y_a, y_b, lam = mixup(x, y, alpha=mix_alpha)
                logits = model(x)
                loss = mix_criterion(ce, logits, y_a, y_b, lam)
            elif mix_mode == "cutmix":
                x, y_a, y_b, lam = cutmix(x, y, alpha=mix_alpha)
                logits = model(x)
                loss = mix_criterion(ce, logits, y_a, y_b, lam)
            else:
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
    img_size = _transform_img_size(eval_tf)
    results["inference_time_per_image_ms"] = measure_pytorch_inference_time_ms(
        model=model,
        img_size=img_size,
        device=device,
        n_warmup=10,
        n_runs=100,
    )
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
