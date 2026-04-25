"""Regenerate every plot used in report/part1/part1_presentation.md.

Run from anywhere: `python report/part1/create_plot.py`.

The goal is slide-readable output: large fonts, generous markers, minimal
clutter — each figure should still be legible when Marp scales it down to
fit a slide column. All inputs are read from the existing artefacts:

    experiments/classical_ml/classical_ml_{svm,logreg}_{backbone}_{res}px.json
    experiments/classical_ml/tsne_cache/tsne_{backbone}_{res}px.npz
    ~/.cache/huggingface/datasets/timm___mini-imagenet  (for visual_grid)

All outputs are written to report/part1/plots/ — no other location is
touched and no other source file is modified.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent          # report/part1
ROOT = HERE.parents[1]                           # repo root
CLASSICAL_DIR = ROOT / "experiments" / "classical_ml"
TSNE_DIR = CLASSICAL_DIR / "tsne_cache"
CLS_MAP_PATH = ROOT / "archive" / "map_clsloc.txt"

OUT_DIR = HERE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = [
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("resnet18", "ResNet-18"),
    ("resnet34", "ResNet-34"),
    ("resnet50", "ResNet-50"),
    ("efficientnet_b0", "EfficientNet-B0"),
]


def _load_result(prefix: str, backbone: str, res: int = 32) -> dict:
    return json.loads((CLASSICAL_DIR / f"classical_ml_{prefix}_{backbone}_{res}px.json").read_text())


# ──────────────────────────────────────────────────────────────────────────
# Slide 3 — class distribution (single-panel grouped bars)
# ──────────────────────────────────────────────────────────────────────────

def make_class_distribution() -> None:
    """Mini-ImageNet is perfectly balanced: 500 / 100 / 50 per class × 100 classes.

    The original 3-panel layout became illegible when the figure was
    shrunk to fit the right column of slide 3. A single panel with three
    grouped bar series conveys the same "flat across every split" message
    at a readable size.
    """
    n_classes = 100
    train = [500] * n_classes
    val = [100] * n_classes
    test = [50] * n_classes

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(n_classes, dtype=float)
    w = 0.28
    ax.bar(x - w, train, w, label=f"Train  (n={max(train)}/class)", color="#1976D2")
    ax.bar(x, val, w, label=f"Val    (n={max(val)}/class)", color="#43A047")
    ax.bar(x + w, test, w, label=f"Test   (n={max(test)}/class)", color="#E53935")
    ax.set_title("Class Distribution — Perfectly Balanced Across All Splits",
                 fontsize=17, fontweight="bold")
    ax.set_xlabel("Class ID", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlim(-1, n_classes)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.95)
    fig.tight_layout()
    out = OUT_DIR / "class_distribution_all_splits.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────
# Slide 3 — visual grid (random samples, requires the HF dataset)
# ──────────────────────────────────────────────────────────────────────────

def make_visual_grid() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skipped visual_grid: `datasets` package not installed.")
        return

    ds = load_dataset("timm/mini-imagenet")
    train = ds["train"]

    label_wnids = train.features["label"].names
    wnid_to_human = {}
    if CLS_MAP_PATH.exists():
        for line in CLS_MAP_PATH.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                wnid_to_human[parts[0]] = parts[2]
    class_names = [wnid_to_human.get(w, w) for w in label_wnids]

    idxs_by_class: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(train["label"]):
        idxs_by_class[lbl].append(i)

    random.seed(0)
    per_class = 5
    classes_to_show = 6
    chosen = sorted(random.sample(list(idxs_by_class), k=classes_to_show))

    fig, axes = plt.subplots(classes_to_show, per_class,
                              figsize=(per_class * 2.0, classes_to_show * 2.0))
    for r, cls in enumerate(chosen):
        picks = random.sample(idxs_by_class[cls], k=per_class)
        for c, idx in enumerate(picks):
            axes[r, c].imshow(train[idx]["image"])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(class_names[cls].replace("_", " "),
                                       fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle("Random Samples by Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "visual_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────
# Slide 5 — efficiency trade-off (accuracy vs params / latency)
# ──────────────────────────────────────────────────────────────────────────

def make_efficiency_tradeoff() -> None:
    rows = []
    for bb, pretty in BACKBONES:
        svm = _load_result("linear_svm", bb, 32)
        lr = _load_result("logreg", bb, 32)
        rows.append({
            "pretty": pretty,
            "params": svm["Number of parameters (mil)"],
            "svm_acc": svm["Test accuracy (%)"],
            "lr_acc": lr["Test accuracy (%)"],
            "svm_ms": svm["Inference Time per Image (ms)"],
            "lr_ms": lr["Inference Time per Image (ms)"],
        })

    styles = {
        "ConvNeXt-Tiny":   ("D", "#2E7D32"),
        "ResNet-18":       ("o", "#1565C0"),
        "ResNet-34":       ("s", "#1976D2"),
        "ResNet-50":       ("^", "#0D47A1"),
        "EfficientNet-B0": ("P", "#EF6C00"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    for ax, xkey, xlabel, title in [
        (axes[0], "params", "Parameters (millions)", "Accuracy vs Model Size"),
        (axes[1], "ms",     "Inference Time (ms/image)", "Accuracy vs Inference Latency"),
    ]:
        for row in rows:
            marker, color = styles[row["pretty"]]
            if xkey == "params":
                x_svm = x_lr = row["params"]
            else:
                x_svm, x_lr = row["svm_ms"], row["lr_ms"]
            ax.scatter(x_svm, row["svm_acc"], marker=marker, s=320, color=color,
                       edgecolor="black", linewidth=1.4, zorder=3)
            ax.scatter(x_lr, row["lr_acc"], marker=marker, s=320, facecolors="none",
                       edgecolor=color, linewidth=2.2, zorder=3)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Test Accuracy (%)", fontsize=15)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=13)
        # headroom so legend in upper region doesn't clip markers
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 1, hi + 2)

    # Legend: one row per backbone (colour+marker), plus two rows for classifier style.
    backbone_handles = [
        Line2D([0], [0], marker=styles[name][0], linestyle="",
               markersize=14, markerfacecolor=styles[name][1],
               markeredgecolor="black", markeredgewidth=1.2, label=name)
        for name, _ in [(p, None) for _, p in BACKBONES]
    ]
    clf_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=14,
               markerfacecolor="gray", markeredgecolor="black",
               markeredgewidth=1.2, label="SVM (filled)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=14,
               markerfacecolor="none", markeredgecolor="gray",
               markeredgewidth=2.0, label="LogReg (hollow)"),
    ]
    fig.legend(handles=backbone_handles + clf_handles,
               loc="lower center", ncol=7, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("A1 — Parameter Efficiency & Latency Trade-off (32×32)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = OUT_DIR / "analysis_efficiency_tradeoff.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────
# Slide 6 — t-SNE (2×2: backbone × resolution)
# ──────────────────────────────────────────────────────────────────────────

def make_tsne() -> None:
    panels = [
        ("convnext_tiny", 32,  "ConvNeXt-Tiny — 32×32"),
        ("resnet18",      32,  "ResNet-18 — 32×32"),
        ("convnext_tiny", 224, "ConvNeXt-Tiny — 224×224"),
        ("resnet18",      224, "ResNet-18 — 224×224"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for (bb, res, title), ax in zip(panels, axes.flat):
        d = np.load(TSNE_DIR / f"tsne_{bb}_{res}px.npz")
        proj, labels = d["proj"], d["labels"]
        ax.scatter(proj[:, 0], proj[:, 1], c=labels, s=9, cmap="tab20",
                   alpha=0.8, linewidth=0)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Frozen Feature Representations (t-SNE) — Test Set",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "downselected_backbones_tsne.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────
# Slide 7 — classifier comparison (SVM vs LogReg grouped bars)
# ──────────────────────────────────────────────────────────────────────────

def make_classifier_accuracy() -> None:
    C_SVM, C_LR = "#EF6C00", "#1E88E5"
    svm = [_load_result("linear_svm", bb, 32)["Test accuracy (%)"] for bb, _ in BACKBONES]
    lr = [_load_result("logreg", bb, 32)["Test accuracy (%)"] for bb, _ in BACKBONES]
    labels = [p for _, p in BACKBONES]
    x = np.arange(len(labels), dtype=float)
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    b_svm = ax.bar(x - w / 2, svm, w, color=C_SVM, edgecolor="black", lw=0.6, alpha=0.9, label="SVM")
    b_lr = ax.bar(x + w / 2, lr, w, color=C_LR, edgecolor="black", lw=0.6, alpha=0.9, label="LogReg")
    for bars, vals in [(b_svm, svm), (b_lr, lr)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=15)
    ax.set_ylim(0, max(max(svm), max(lr)) + 7)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper right", fontsize=14, frameon=True)
    ax.set_title("A1 — Classifier Comparison at 32×32 (Test Accuracy)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "analysis_classifier_accuracy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_class_distribution()
    make_classifier_accuracy()
    make_efficiency_tradeoff()
    make_tsne()
    make_visual_grid()
