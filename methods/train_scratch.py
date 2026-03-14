"""
methods/train_scratch.py
========================
Approach 3 — Training from Scratch (Baseline)
----------------------------------------------
Train a ConvNeXt-Tiny from randomly initialised weights (no pretrained
weights) end-to-end on Mini-ImageNet.  This serves as the baseline to
measure how much benefit pretrained weights provide.

Internally re-uses `train_finetune` from methods/finetune.py with
`use_pretrained=False` and `freeze_policy='none'`.

Usage (standalone):
    python -m methods.train_scratch --save_dir experiments/scratch
"""

import argparse
from pathlib import Path

from utils import (
    ensure_dir, load_mini_imagenet, make_transforms,
    set_seed, get_device,
)
from methods.finetune import train_finetune


def train_from_scratch(
    ds,
    train_tf,
    eval_tf,
    device,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_dir: Path = Path("./outputs"),
) -> tuple:
    """
    Train a ConvNeXt-Tiny from random initialisation (no pretrained weights).

    All feature stages and the classifier head are trained jointly —
    equivalent to `freeze_policy='none'` with `use_pretrained=False`.

    Args:
        ds:          HuggingFace DatasetDict.
        train_tf:    Training transform (may include augmentation).
        eval_tf:     Evaluation transform.
        device:      torch.device.
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        lr:          Learning rate for AdamW.
        save_dir:    Directory for checkpoint and results JSON.

    Returns:
        (results dict, path to best checkpoint)
    """
    return train_finetune(
        ds=ds,
        train_tf=train_tf,
        eval_tf=eval_tf,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_policy="none",
        use_pretrained=False,
        save_dir=save_dir,
    )


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Approach 3: Train ConvNeXt-Tiny from scratch")
    p.add_argument("--save_dir",   default="experiments/scratch")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--use_aug",    action="store_true")
    p.add_argument("--seed",       type=int,   default=24)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=save_dir,
    )
