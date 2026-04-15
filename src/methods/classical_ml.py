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

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, log_loss

from ..utils import ensure_dir, make_loaders
from ..model import build_backbone, extract_convnext_features


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
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=1)
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

    # Compute test loss (cross entropy) — use predict_proba if available
    test_loss = None
    if hasattr(clf, "predict_proba"):
        test_loss = float(log_loss(y_test, clf.predict_proba(X_test)))

    # Count total backbone parameters (millions)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Determine image size from eval transform
    img_size = 224  # default
    if hasattr(eval_tf, 'transforms'):
        for t in eval_tf.transforms:
            if hasattr(t, 'size'):
                sz = t.size
                img_size = sz if isinstance(sz, int) else sz[0]
                break

    # Measure per-image inference time (feature extraction only)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):          # warm-up
            model(dummy)
    n_runs = 100
    t_inf_start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    infer_ms = (time.time() - t_inf_start) / n_runs * 1000

    results = {
        "approach":                     "classical_ml",
        "classifier":                   clf.__class__.__name__,
        "backbone":                     backbone,
        "Image Size":                   img_size,
        "Number of parameters (mil)":   round(n_params, 2),
        "val_acc":                      float(val_acc),
        "Test accuracy (%)":            round(float(test_acc) * 100, 2),
        "Test loss (Cross Entropy)":    test_loss,
        "Training Time (seconds)":      round(feat_time + train_time, 2),
        "Inference Time per Image (ms)": round(infer_ms, 4),
        "n_train":                      int(len(y_train)),
        "n_val":                        int(len(y_val)),
        "n_test":                       int(len(y_test)),
    }
    out_path = save_dir / f"classical_ml_{clf_type}_{backbone}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {out_path}")
    return results
