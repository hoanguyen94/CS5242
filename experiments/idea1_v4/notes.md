# Summary of Improvements: Classical ML to Approach 2 (v4)

This document summarizes the improvements and architectural changes between the initial classical ML pipeline (`python3 main.py --task features_ml --clf_type logreg --backbone resnet18 --img_size 224`) and the custom end-to-end fine-tuning script (`python approach2_idea1_v4.py --device mps --epochs 40`).

## Executive Summary

The transition from Approach 1 (Classical ML) to Approach 2 (Idea 1 v4 Pipeline) represents a massive shift from simple, static mapping to heavily regularized, robust deep learning finetuning. The core improvements include:
- **Architecture**: Moved from entirely frozen static ImageNet features into a dynamically finetuned 2-Phase training network, while replacing a simple linear layer with a comprehensive Multi-Layer Perceptron (MLP) head with Dropout.
- **Catastrophic Forgetting Mitigations**: Deep layer ImageNet weights are protected via extreme, layer-wise learning rate down-scaling (0.001x of base) and by locking the Spatial BatchNorm components to evaluation mode.
- **Heavy Regularization**: Model overconfidence is aggressively stripped through dense augmentations (RandAugment, RandomErasing), Label Smoothing, and continuous Mixup stochastic regularizations. 
- **Robust Convergence**: A shadow Exponential Moving Average (EMA) network persistently tracks sliding model weights, resisting batch noise and making the final checkpoint notably superior and stable compared to raw early-stopped checkpoints.

---

## Detailed Improvements

### 1. Shift from Static Features to End-to-End Finetuning
- **Classical ML Baseline:** Feature extraction is static. The ResNet18 backbone is entirely frozen, and vectors (Global Average Pooling outputs) are extracted uniformly to train a simple, separate sklearn Logistic Regression model.
- **Improved v4:** Full end-to-end fine-tuning pipeline. Enables the backbone features to gently adapt to the target dataset rather than being entirely rigid. 

### 2. Richer Classifier Head (MLP)
- **Classical ML Baseline:** Single linear layer (equivalent to Logistic Regression).
- **Improved v4:** The `fc` (fully connected) layer of the ResNet18 is replaced with a larger Multi-Layer Perceptron (MLP) head: `Linear(512, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(256, 100)`. Adds non-linearity and regularization to fit the dataset better.

### 3. Two-Phase Gradual Unfreezing
- **Classical ML Baseline:** No unfreezing. 
- **Improved v4:** Implements a 2-phase strategy.
  - **Phase 1 (~1/3 of epochs):** Trains only the new MLP head.
  - **Phase 2 (remaining epochs):** Unfreezes the full model to allow the backbone to adapt.

### 4. Ultra-Protective Layer-wise Learning Rates
- **Improved v4:** Unfreezing the entire backbone naively leads to "catastrophic forgetting" of the pretrained ImageNet weights. The v4 script assigns scaled-down learning rates to deeper layers to protect them:
  - `stem`, `layer1`: `0.001 * base_lr`
  - `layer2`: `0.002 * base_lr`
  - `layer3`: `0.005 * base_lr`
  - `layer4`: `0.010 * base_lr`
  - `head`: `1.0 * base_lr`

### 5. Frozen Spatial BatchNorm
- **Improved v4:** Forces all `BatchNorm2d` layers in the ResNet18 backbone into `eval()` mode (`set_bn_eval`) during training. This ensures they use the robust ImageNet running statistics instead of getting corrupted by small or heavily augmented batches.

### 6. Advanced Data Augmentations
- **Classical ML Baseline:** Used simple `eval_tf` (Resize + CenterCrop) without data augmentation.
- **Improved v4:** Introduces a heavy modern augmentation pipeline during training:
  - `RandomResizedCrop` and `RandomHorizontalFlip`
  - `RandAugment(magnitude=5, num_ops=2)`
  - `RandomErasing(p=0.25)`

### 7. Mixup Training & Label Smoothing
- **Improved v4:** 
  - Applies **Mixup** (`alpha=0.2`) during the forward pass, interpolating both images and their corresponding labels to train a more robust classifier.
  - Uses `LabelSmoothingCrossEntropyLoss` (smoothing factor `0.10`) instead of hard labels to reduce overconfidence.

### 8. Robust Optimization (AdamW + Cosine Annealing)
- **Classical ML Baseline:** Standard optimization through L-BFGS or Liblinear in sklearn.
- **Improved v4:** Uses `AdamW` (with `weight_decay=1e-4`) and strict `CosineAnnealingLR` (dropping previous OneCycleLR spikes) for smoother convergence toward the end of training.

### 9. Continuous EMA (Exponential Moving Average) Model
- **Improved v4:** Tracks a shadow model built from the Exponential Moving Average (`decay=0.999`) of the main model's weights. The EMA model is evaluated alongside the base model and often provides a much more stable and generalized set of weights, preventing performance loss from local batch noise. Early stopping tracks the best Validation Accuracy across both base and EMA checkpoints.
