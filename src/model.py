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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    # ConvNeXt / EfficientNet style (classifier as nn.Sequential)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
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

    # Use torchvision model builder if available
    if not hasattr(torchvision.models, backbone):
        raise ValueError(f"Backbone '{backbone}' is not supported.")

    model_fn = getattr(torchvision.models, backbone)
    sig = inspect.signature(model_fn)

    # Some models accept `weights` keyword (newer torchvision), others accept `pretrained`.
    kwargs = {}

    def _get_weights_enum(name: str):
        # Known weights class naming conventions
        overrides = {
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "convnext_small": "ConvNeXt_Small_Weights",
            "convnext_base": "ConvNeXt_Base_Weights",
            "convnext_large": "ConvNeXt_Large_Weights",
        }
        if name in overrides:
            return overrides[name]
        # Generic conversion: resnet18 -> ResNet18_Weights
        parts = name.split("_")
        camel = "".join(p.capitalize() if p.isalpha() else p.upper() for p in parts)
        return f"{camel}_Weights"

    if "weights" in sig.parameters:
        weights = None
        if pretrained:
            try:
                weights_cls = getattr(torchvision.models, _get_weights_enum(backbone))
                weights = weights_cls.DEFAULT
            except Exception:
                weights = None
        kwargs["weights"] = weights
    elif "pretrained" in sig.parameters:
        kwargs["pretrained"] = pretrained

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


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def print_freeze_summary(model: nn.Module) -> None:
    """Print a per-module summary of frozen vs trainable parameters."""
    print(f"{'Module':<40s} {'Params':>10s} {'Trainable':>10s} {'Status':>10s}")
    print("─" * 74)
    total, total_train = 0, 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += params
        total_train += trainable
        status = "frozen" if trainable == 0 else ("trainable" if trainable == params else "partial")
        print(f"{name:<40s} {params:>10,d} {trainable:>10,d} {status:>10s}")
    print("─" * 74)
    print(f"{'TOTAL':<40s} {total:>10,d} {total_train:>10,d}")
    print(f"Trainable: {total_train/max(total,1)*100:.1f}%\n")


def plot_training_curves(results: dict, save_dir=None) -> None:
    """Plot loss, accuracy, and LR curves from training results."""
    import matplotlib.pyplot as plt

    logs = results["epoch_logs"]
    if not logs:
        print("No epoch logs to plot.")
        return

    epochs = [l["epoch"] for l in logs]
    train_loss = [l["train_loss"] for l in logs]
    val_loss = [l["val_loss"] for l in logs]
    train_acc = [l["train_acc"] for l in logs]
    val_acc = [l["val_acc"] for l in logs]
    has_lr = "lr" in logs[0]

    n_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    # Loss
    axes[0].plot(epochs, train_loss, "o-", label="Train")
    axes[0].plot(epochs, val_loss, "o-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "o-", label="Train")
    axes[1].plot(epochs, val_acc, "o-", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR schedule
    if has_lr:
        lrs = [l["lr"] for l in logs]
        axes[2].plot(epochs, lrs, "o-", color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{results.get('approach', '')} | freeze={results.get('freeze_policy', 'n/a')}")
    plt.tight_layout()
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / "training_curves.png", dpi=200)
    plt.show()


@torch.no_grad()
def extract_features_for_vis(model: nn.Module, loader, device, max_samples: int = 2000):
    """Extract penultimate-layer features and labels for a subset of samples."""
    model.eval()
    feats_list, labels_list = [], []
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feat_map = _extract_backbone_features(model, x)
        feats = feat_map.mean(dim=(2, 3))  # GAP → (B, C)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(y.numpy())
        n += y.size(0)
        if n >= max_samples:
            break
    feats = np.concatenate(feats_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    return feats, labels


def plot_representation_snapshots(snapshots: list, save_dir=None) -> None:
    """
    Plot t-SNE of feature representations at different epochs.

    Args:
        snapshots: list of dicts with keys 'epoch', 'features' (N,C), 'labels' (N,)
        save_dir:  optional directory to save the figure
    """

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, snap in zip(axes, snapshots):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
        proj = tsne.fit_transform(snap["features"])
        scatter = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=snap["labels"], cmap="tab20", s=3, alpha=0.6,
        )
        ax.set_title(f"Epoch {snap['epoch']}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Feature Representations (t-SNE)", fontsize=14)
    plt.tight_layout()
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / "representation_tsne.png", dpi=200)
    plt.show()
