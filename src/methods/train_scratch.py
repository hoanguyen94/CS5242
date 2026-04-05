"""
methods/train_scratch.py
========================
Approach 3 — Training from Scratch (Baseline)
----------------------------------------------
Train a backbone (ConvNeXt / ResNet / EfficientNet) from randomly
initialised weights (no pretrained weights) end-to-end on Mini-ImageNet.
This serves as the baseline to measure how much benefit pretrained weights provide.

Internally re-uses `train_finetune` from methods/finetune.py with
`use_pretrained=False` and `freeze_policy='none'`.

Usage (standalone):
    python -m methods.train_scratch --save_dir experiments/scratch
"""

from pathlib import Path

from .finetune import train_finetune


def train_from_scratch(
    ds,
    train_tf,
    eval_tf,
    device,
    backbone: str = "convnext_tiny",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_dir: Path = Path("./outputs"),
    patience: int = 7,
    mix_mode: str = "none",
    mix_alpha: float = 0.2,
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
        patience:    Early stopping patience.
        mix_mode:    'none', 'mixup', or 'cutmix'.
        mix_alpha:   Beta distribution parameter for mixing.

    Returns:
        (results dict, path to best checkpoint)
    """
    return train_finetune(
        ds=ds,
        train_tf=train_tf,
        eval_tf=eval_tf,
        device=device,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_policy="none",
        use_pretrained=False,
        save_dir=save_dir,
        patience=patience,
        mix_mode=mix_mode,
        mix_alpha=mix_alpha,
    )
