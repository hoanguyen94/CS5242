"""
model.py — ConvNeXt-Tiny model builder, feature extraction,
           freeze policies, parameter/FLOPs counting, and evaluation.
"""

from pathlib import Path
from typing import Optional, Tuple
import copy
import inspect

import numpy as np
import torch
import torch.nn as nn

import torchvision
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

def _replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classifier head to match `num_classes`."""
    # ConvNeXt style
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model

    # EfficientNet (torchvision) style
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # many EfficientNet variants store classifier as nn.Sequential([Dropout, Linear])
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model

    # ResNet style
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # If model has a head attribute (some models use 'head' or 'heads')
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("Unable to replace classifier head for model type: %s" % type(model))


def build_backbone(
    backbone: str = "convnext_tiny",
    num_classes: int = 100,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Builds a backbone model with the specified number of output classes."""

    if backbone == "resnet18_scratch":
        from methods.model_utils import ResNet, ConvNeXt
        model = ResNet([2, 2, 2, 2], num_classes=num_classes)
        return model.to(device)
    if backbone == "convnext_tiny_scratch":
        model = ConvNeXt(num_classes=num_classes)
        return model.to(device)

    # Use torchvision model builder if available
    if not hasattr(torchvision.models, backbone):
        raise ValueError(f"Backbone '{backbone}' is not supported.")

    model_fn = getattr(torchvision.models, backbone)
    # Some models accept `weights` keyword (newer torchvision), others accept `pretrained`.
    kwargs = {"weights": "DEFAULT"} if "weights" in inspect.signature(model_fn).parameters else {"pretrained": pretrained}
    model = model_fn(**kwargs)
    model = _replace_classifier_head(model, num_classes)
    return model.to(device)


def build_convnext_tiny(
    num_classes: int = 100,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Backward compatible helper for ConvNeXt-Tiny."""
    return build_backbone(
        backbone="convnext_tiny",
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

def _extract_backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return a feature map tensor (B, C, H, W) for a variety of backbones."""
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    if hasattr(model, "features"):
        return model.features(x)

    # torchvision ResNet / ResNeXt style
    if hasattr(model, "fc") and hasattr(model, "avgpool"):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        if hasattr(model, "maxpool"):
            x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)  # (B, C, 1, 1)
        return x

    raise ValueError(
        "Unsupported model type for feature extraction. "
        "Provide a model with `forward_features`, `features`, or ResNet-like API."
    )


@torch.no_grad()
def extract_convnext_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts GAP features from a backbone model.

    Works with ConvNeXt, EfficientNet, ResNet, and other torchvision models.

    Returns:
        features: (N, C)
        labels:   (N,)
    """
    model.eval()
    feats_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feat_map = _extract_backbone_features(model, x)
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
    """Apply a freeze policy to the model's backbone.

    Supported policies:
      * backbone   — freeze all feature stages; train classifier only
      * last_stage — freeze early stages; train last stage + classifier
      * none       — train all parameters

    Works for ConvNeXt, ResNet, EfficientNet variants, and other torchvision backbones.
    """

    # Restore all parameters as trainable
    for p in model.parameters():
        p.requires_grad = True

    if policy == "none":
        return

    def _freeze_module(mod: nn.Module):
        for p in mod.parameters():
            p.requires_grad = False

    def _unfreeze_module(mod: nn.Module):
        for p in mod.parameters():
            p.requires_grad = True

    # Identify feature extractor components
    if hasattr(model, "features"):
        features = model.features
        if policy == "backbone":
            _freeze_module(features)
        elif policy == "last_stage":
            children = list(features.children())
            cutoff = max(0, (3 * len(children)) // 4)
            for i, child in enumerate(children):
                if i < cutoff:
                    _freeze_module(child)
                else:
                    _unfreeze_module(child)
        return

    # ResNet-like models (conv1/bn1 + layer1..layer4)
    if hasattr(model, "layer1") and hasattr(model, "layer4"):
        if policy == "backbone":
            _freeze_module(model.conv1)
            _freeze_module(model.bn1)
            if hasattr(model, "maxpool"):
                _freeze_module(model.maxpool)
            _freeze_module(model.layer1)
            _freeze_module(model.layer2)
            _freeze_module(model.layer3)
            _freeze_module(model.layer4)
        elif policy == "last_stage":
            # Freeze everything except the last residual stage
            _freeze_module(model.conv1)
            _freeze_module(model.bn1)
            if hasattr(model, "maxpool"):
                _freeze_module(model.maxpool)
            _freeze_module(model.layer1)
            _freeze_module(model.layer2)
            _freeze_module(model.layer3)
            _unfreeze_module(model.layer4)
        return

    # Generic fallback: freeze all but the final classifier if present
    if hasattr(model, "classifier"):
        _freeze_module(model)
        # Unfreeze classifier if it exists
        clf = model.classifier
        if isinstance(clf, nn.Module):
            _unfreeze_module(clf)
    elif hasattr(model, "fc"):
        _freeze_module(model)
        _unfreeze_module(model.fc)
    else:
        raise ValueError(f"Unsupported model type for freeze policy: {type(model)}")


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
    model_copy = copy.deepcopy(model).to(device)
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