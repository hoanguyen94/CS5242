# Summary of Improvements: Classical ML to Approach 2 (v5)

This document outlines the architectural leaps and improvements introduced in the **Idea 1 v5 Pipeline** (`approach2_idea1_v5.py`) compared to the initial flat classical ML baseline (`python3 main.py --task features_ml --clf_type logreg --backbone resnet18 --img_size 224`). 

## Executive Summary

The transition to Approach 2 v5 represents the most performant architecture shift so far. While retaining the powerful regularization (Mixup, EMA, RandAugment) introduced in v4, v5 formally abandons ResNet18 and moves to a native **ConvNeXt-Tiny** topology. This fundamentally resolves spatial-batch-statistics issues and handles feature finetuning more gracefully. The core innovations include:
- **Native ConvNeXt-Tiny Backbone**: Migrates away from ResNet18 into the ultra-modern `convnext_tiny` block architecture (768-dim), inherently minimizing catastrophic forgetting.
- **LayerNorm Centric Modularity**: Exclusively employs robust `LayerNorm` logic everywhere, including in the newly injected MLP classifier head. The previous `set_bn_eval` logic was entirely purged, as ConvNeXt is indifferent to small batch distribution variances.
- **Stage-Mapped Protective Learning Rates**: Finetuning maps directly to ConvNeXt's formal 4-stage hierarchy (`model.features`), protecting deep image extractors with strict `0.001x` scaled-down learning rates.
- **Heavy Regularization & Moving Averages**: Maintains dense augmentation policies alongside continuous Exponential Moving Average (EMA) shadow tracking for superior stability and convergence.

---

## Detailed Improvements

### 1. Shift to High-Performance Backbone (ConvNeXt-Tiny)
- **Classical ML Baseline:** Used a frozen, classical ResNet18.
- **Improved v5:** Incorporates the `convnext_tiny` backbone, extracting representations in a modernized 768-dimension space that naturally responds far better to downstream finetuning without extreme catastrophic forgetting.

### 2. Richer Classifier Head (ConvNeXt Specialization)
- **Classical ML Baseline:** Single linear scikit-learn layer over the GAP vector.
- **Improved v5:** Intercepts ConvNeXt's head (`model.classifier[-1]`) and bolts on a powerful Multi-Layer Perceptron: `Linear(768, 256) -> LayerNorm(256) -> ReLU -> Dropout(0.3) -> Linear(256, 100)`. Note that `LayerNorm` replaces `BatchNorm1d` here to perfectly match ConvNeXt’s internal design ethos.

### 3. Stage-Mapped Protective Learning Rates
- **Classical ML Baseline:** Did not update backbone features; no internal learning rate strategy needed.
- **Improved v5:** Employs a complex optimizer group to map over ConvNeXt's formal stages (`model.features[0..7]`), protecting early convolutional components heavily during full model unfreezing phase:
  - `stem`, `stage0`: `0.001 * base_lr`
  - `stage1`:   `0.005 * base_lr`
  - `stage2`:   `0.010 * base_lr`
  - `stage3`:   `0.050 * base_lr`
  - `head`:     `1.000 * base_lr`

### 4. Eradication of Spatial BatchNorm Issues
- **Classical ML Baseline:** Standard convolutional BatchNorm logic frozen.
- **Improved v5:** Fully abandons spatial BatchNorm layers. The ConvNeXt architecture utilizes LayerNorm blocks which perform equally well regardless of batch size or augmentations. Thus, the hacky `set_bn_eval` trick required in v4 has been completely removed.

### 5. Two-Phase Gradual Unfreezing
- **Classical ML Baseline:** No unfreezing.
- **Improved v5:** Deploys a 2-Phase training loop. 
  - **Phase 1 (~1/3 of epochs):** Trains only the new MLP interceptor.
  - **Phase 2:** Lowers the protective floodgates and unfreezes the entire `convnext_tiny` model.

### 6. Advanced Data Augmentations
- **Classical ML Baseline:** Evaluated entirely on cropped, non-augmented images (`eval_tf`).
- **Improved v5:** Uses a heavily regularized stochastic data pipeline:
  - `RandomResizedCrop` and `RandomHorizontalFlip`
  - `RandAugment(magnitude=5, num_ops=2)`
  - `RandomErasing(p=0.25)`

### 7. Mixup Training & Label Smoothing
- **Improved v5:** Computes continuous **Mixup** (`alpha=0.2`) across image pairs and their target distribution to eliminate class overconfidence. Complimented by `LabelSmoothingCrossEntropyLoss` (smoothing factor `0.10`) instead of absolute hard targets.

### 8. Continuous EMA (Exponential Moving Average) Model
- **Improved v5:** Operates an active shadow parameters AveragedModel (`decay=0.999`) updating continuously across batches. This reliably buffers out mini-batch variance drops and secures the best generalization validation accuracy available. Early stopping formally observes both the standard and EMA paths.

### 9. Robust Optimization (AdamW + Cosine Annealing LR)
- **Improved v5:** Abandons spikes in favor of a reliable `CosineAnnealingLR` drop coupled deeply with the heavy `AdamW` (with `weight_decay=1e-4`) optimizer. Matches well with the complex Mixup and tracking architectures seamlessly.
