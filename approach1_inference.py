import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from torchvision import transforms

from src.model import build_backbone, extract_convnext_features
from src.utils import make_loaders
from src.data_processing.data_processing import load_mini_imagenet

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data"
EXPERIMENTS_DIR = Path("experiments/classical_ml")
INFERENCE_DIR   = EXPERIMENTS_DIR / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Load dataset once (shared across all experiments) ─────────────────────────
ds = load_mini_imagenet(cache_path=DATA_PATH)
print(f"Dataset splits: {list(ds.keys())}")
print(f"Test size: {len(ds['test'])}")


def make_eval_transform(img_size: int = 224) -> transforms.Compose:
    """Standard ImageNet-style eval transform."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def run_inference(exp_json_path: Path, ds, device: torch.device) -> dict:
    """
    Load a trained classical-ML experiment and evaluate it on the test split.

    Args:
        exp_json_path: Path to the training-time result JSON.
        ds:            HuggingFace DatasetDict with a 'test' split.
        device:        Torch device for feature extraction.

    Returns:
        Inference result dict saved to experiments/classical_ml/inference/.
    """
    with open(exp_json_path) as f:
        cfg = json.load(f)

    backbone_name = cfg["backbone"]
    img_size      = cfg.get("Image Size", 224)
    clf_path      = Path(cfg["clf_path"])
    backbone_path = Path(cfg["backbone_path"])

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_json_path.stem}")
    print(f"Backbone   : {backbone_name}  |  Image size: {img_size}")
    print(f"Classifier : {cfg['classifier']}")
    print(f"{'='*60}")

    # ── Load backbone ─────────────────────────────────────────────────────────
    model = build_backbone(backbone=backbone_name, num_classes=100, pretrained=False, device=device)
    state = torch.load(backbone_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded backbone   <- {backbone_path}")

    # ── Load classifier ───────────────────────────────────────────────────────
    clf = joblib.load(clf_path)
    print(f"Loaded classifier <- {clf_path}")

    # ── Build test loader ─────────────────────────────────────────────────────
    eval_tf = make_eval_transform(img_size)
    _, _, test_loader = make_loaders(ds, eval_tf, eval_tf, batch_size=256)

    # ── Extract test features ─────────────────────────────────────────────────
    print("Extracting test features ...")
    X_test, y_test = extract_convnext_features(model, test_loader, device)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    y_pred   = clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))

    # ── Test loss (only when predict_proba is available) ──────────────────────
    test_loss = None
    if hasattr(clf, "predict_proba"):
        test_loss = float(log_loss(y_test, clf.predict_proba(X_test)))

    # ── Per-image inference time (backbone forward pass + clf.predict) ────────
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):           # warm-up
            _ = model(dummy)
    n_runs = 100
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            feat = model(dummy)
    infer_ms = (time.time() - t0) / n_runs * 1000

    # ── Count parameters ──────────────────────────────────────────────────────
    n_params = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    result = {
        "approach":                      "classical_ml",
        "classifier":                    cfg["classifier"],
        "backbone":                      backbone_name,
        "Image Size":                    img_size,
        "Number of parameters (mil)":    n_params,
        "Test accuracy (%)":             round(test_acc * 100, 2),
        "Test loss (Cross Entropy)":     test_loss,
        "Inference Time per Image (ms)": round(infer_ms, 4),
        "clf_path":                      str(clf_path),
        "backbone_path":                 str(backbone_path),
    }

    out_path = INFERENCE_DIR / f"{exp_json_path.stem}_inference.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print(json.dumps(result, indent=2))
    return result


# ── Discover all experiment JSON files ────────────────────────────────────────
exp_jsons = sorted(EXPERIMENTS_DIR.glob("classical_ml_*.json"))
print(f"\nFound {len(exp_jsons)} experiment(s):")
for p in exp_jsons:
    print(f"  {p.name}")

# ── Run inference for every discovered experiment ─────────────────────────────
all_results = []
for exp_json in exp_jsons:
    result = run_inference(exp_json, ds, DEVICE)
    all_results.append(result)

print(f"\n{'='*60}")
print(f"Done. {len(all_results)} inference result(s) saved to {INFERENCE_DIR}/")

# ── Comprehensive results summary (matches main.ipynb Cell 41) ────────────────
print("\n\n=== Comprehensive Results Summary ===\n")

if all_results:
    df = pd.DataFrame(all_results)
    df["NetScore"] = 20 * np.log10(
        (df["Test accuracy (%)"] ** 2) /
        ((df["Inference Time per Image (ms)"] ** 0.5) * (df["Number of parameters (mil)"] ** 0.5))
    )

    cols = [
        "approach", "classifier", "backbone", "Image Size",
        "Number of parameters (mil)", "Test accuracy (%)",
        "Test loss (Cross Entropy)", "Inference Time per Image (ms)", "NetScore",
    ]
    df_present = df[[c for c in cols if c in df.columns]]
    df_present = df_present.sort_values(by=["classifier", "backbone"]).reset_index(drop=True)

    df_display = df_present.copy()
    if "Test accuracy (%)" in df_display.columns:
        df_display["Test accuracy (%)"] = df_display["Test accuracy (%)"].round(2).astype(str) + "%"
    if "Test loss (Cross Entropy)" in df_display.columns:
        df_display["Test loss (Cross Entropy)"] = df_display["Test loss (Cross Entropy)"].round(4)
    if "Inference Time per Image (ms)" in df_display.columns:
        df_display["Inference Time per Image (ms)"] = df_display["Inference Time per Image (ms)"].round(2)
    if "NetScore" in df_display.columns:
        df_display["NetScore"] = df_display["NetScore"].round(2)

    print(df_display.to_string(index=False))
else:
    print("No results to display.")