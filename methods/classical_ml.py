"""
methods/classical_ml.py
=======================
Approach 1 — Classical ML on Frozen ConvNeXt-Tiny Features
-----------------------------------------------------------
Extract GAP features from a pretrained (frozen) ConvNeXt-Tiny backbone
and train a classical classifier (Logistic Regression or Linear SVM)
on top of those features.

Usage (standalone):
    python -m methods.classical_ml --save_dir experiments/classical_ml
"""

import argparse
import json
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils import (
    ensure_dir, load_mini_imagenet, make_transforms, make_loaders,
    set_seed, get_device,
)
from model import build_backbone, extract_convnext_features


def classical_ml_experiment(
    ds,
    eval_tf,
    device,
    backbone: str = "convnext_tiny",
    clf_type: str = "logreg",
    batch_size: int = 256,
    save_dir: Path = Path("./outputs"),
) -> dict:
    """
    Extract features from pretrained ConvNeXt-Tiny, then fit a classical
    classifier (Logistic Regression or Linear SVM).

    Args:
        ds:         HuggingFace DatasetDict with 'train'/'validation'/'test' splits.
        eval_tf:    Evaluation transform (no augmentation).
        device:     torch.device to use for feature extraction.
        clf_type:   'logreg' or 'linear_svm'.
        batch_size: Batch size for the feature-extraction DataLoader.
        save_dir:   Directory where the results JSON will be written.

    Returns:
        results dict containing accuracies and timing information.
    """
    ensure_dir(save_dir)

    # Feature extractor — pretrained backbone, classifier head not used
    model = build_backbone(
        backbone=backbone,
        num_classes=100,
        pretrained=True,
        device=device,
    )
    train_loader, val_loader, test_loader = make_loaders(
        ds, eval_tf, eval_tf, batch_size=batch_size
    )

    print("Extracting features (train / val / test) …")
    t0 = time.time()
    X_train, y_train = extract_convnext_features(model, train_loader, device)
    X_val,   y_val   = extract_convnext_features(model, val_loader,   device)
    X_test,  y_test  = extract_convnext_features(model, test_loader,  device)
    feat_time = time.time() - t0
    print(
        f"Feature extraction: {feat_time:.1f}s | "
        f"train={X_train.shape}  val={X_val.shape}  test={X_test.shape}"
    )

    # Build classifier
    if clf_type == "logreg":
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=0)
    elif clf_type == "linear_svm":
        clf = LinearSVC()
    else:
        raise ValueError(f"clf_type must be 'logreg' or 'linear_svm', got {clf_type!r}")

    print(f"Training {clf.__class__.__name__} …")
    t1 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t1

    # Evaluate
    val_acc  = accuracy_score(y_val,  clf.predict(X_val))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    results = {
        "approach":          "classical_ml",
        "classifier":        clf.__class__.__name__,
        "feature_time_sec":  feat_time,
        "train_time_sec":    train_time,
        "val_acc":           float(val_acc),
        "test_acc":          float(test_acc),
        "n_train":           int(len(y_train)),
        "n_val":             int(len(y_val)),
        "n_test":            int(len(y_test)),
    }
    out_path = save_dir / f"classical_ml_{clf_type}_{backbone}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Classical ML results: {results}")
    print(f"Saved results → {out_path}")
    return results


# ──────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Approach 1: Classical ML on frozen features")
    p.add_argument("--save_dir",    default="experiments/classical_ml")
    p.add_argument("--clf_type",    default="logreg", choices=["logreg", "linear_svm"])
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--backbone",    default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1"])
    p.add_argument("--seed",        type=int, default=24)
    p.add_argument("--use_gpu",     action="store_true")
    p.add_argument("--subset",      type=int, default=None,
                   help="Limit dataset size per split (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_seed(args.seed)
    device   = get_device(args.use_gpu)
    save_dir = Path(args.save_dir)

    print("Loading dataset …")
    ds = load_mini_imagenet(subset=args.subset)

    _, eval_tf, _ = make_transforms(img_size=args.img_size)

    classical_ml_experiment(
        ds=ds,
        eval_tf=eval_tf,
        device=device,
        backbone=args.backbone,
        clf_type=args.clf_type,
        batch_size=args.batch_size,
        save_dir=save_dir,
    )
