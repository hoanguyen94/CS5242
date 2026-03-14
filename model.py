"""
model.py — ConvNeXt-Tiny model builder, feature extraction,
           freeze policies, parameter/FLOPs counting, and evaluation.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Optional FLOPs (handled gracefully if not installed)
try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False


# ──────────────────────────────────────────────
# Model Construction
# ──────────────────────────────────────────────

def build_convnext_tiny(
    num_classes: int = 100,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    # Replace classifier head to match target number of classes
    in_features = model.classifier[-1].in_features  # typically 768
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_convnext_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts GAP features from ConvNeXt-Tiny backbone (before classifier).
    Returns features (N, 768) and labels (N,).
    """
    model.eval()
    feats_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feat_map = model.forward_features(x)   # (B, C, H, W)
        feats = feat_map.mean(dim=(2, 3))       # GAP → (B, C)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(y.numpy())
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


# ──────────────────────────────────────────────
# Freeze Policies
# ──────────────────────────────────────────────

def set_freeze_policy(model: nn.Module, policy: str) -> None:
    """
    policy in {'backbone', 'last_stage', 'none'}
      backbone   — freeze all feature stages; train classifier only
      last_stage — freeze early stages; train last stage + classifier
      none       — train all parameters
    """
    # First unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    if policy == "backbone":
        for p in model.features.parameters():
            p.requires_grad = False

    elif policy == "last_stage":
        total_blocks = len(list(model.features.children()))
        cutoff = max(0, (3 * total_blocks) // 4)
        for i, m in enumerate(model.features.children()):
            requires = (i >= cutoff)
            for p in m.parameters():
                p.requires_grad = requires
        # classifier remains trainable

    elif policy == "none":
        pass  # all params already unfrozen above

    else:
        raise ValueError(f"Unknown freeze policy: {policy!r}")


# ──────────────────────────────────────────────
# Model Profiling
# ──────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_flops(
    model: nn.Module,
    img_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    """Returns GFLOPs, or None if thop is not installed."""
    if not THOP_AVAILABLE:
        return None
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model_copy = build_convnext_tiny(num_classes=100, pretrained=False, device=device)
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()
    flops, _ = profile(model_copy, inputs=(dummy,), verbose=False)
    return flops / 1e9


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (top-1 accuracy, average loss)."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1), loss_sum / max(total, 1)
