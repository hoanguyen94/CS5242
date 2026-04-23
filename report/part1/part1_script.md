# Speaker Script — Parts 2 & 3

**Target runtime: 4 minutes.** Covers the Three-Lens overview (Part 2) and Approach 1 — Classical ML (Part 3).

---

## Part 2 — Three Approaches (~30s)

### Slide: The Three-Lens Design

We attack Mini-ImageNet from three angles. Approach 1 — which I'll cover now — freezes pretrained backbones and trains a classical SVM or logistic regression on top. It's cheap and isolates feature quality. Approach 2 fine-tunes those same backbones with selective unfreezing — the proposed improvement. And Approach 3 trains from scratch with no ImageNet prior, so we can measure what pretraining actually buys us. Together, the three let us attribute performance to feature quality, task adaptation, and end-to-end learning separately.

---

## Part 3 — Approach 1: Classical ML (~3 min 30s)

### Slide: Motivation (~15s)

The idea is simple: freeze a pretrained backbone, extract features once, and fit a linear classifier. Advantages — data-efficient, fast, interpretable. Limitations — no task-specific adaptation, a linear ceiling, and sensitivity to resolution mismatch.

---

### Slide: Pipeline (~10s)

Image goes through the frozen backbone, global average pooling gives us a feature vector, and we train either a linear SVM or logistic regression — both sklearn defaults, no hyperparameter sweep.

---

### Slide: Experimental Design (~15s)

Five backbones: ConvNeXt-Tiny, ResNet-18/34/50, and EfficientNet-b0. All crossed with both classifiers at 32×32, plus a 224-pixel run on the two we eventually downselect.

---

### Slide: NetScore Overview (~15s)

Starting with the big picture — ConvNeXt-Tiny wins on raw accuracy at 44%, but ResNet-18 tops NetScore — the efficiency metric from Wong — because it's the smallest and fastest. EfficientNet is worst on both axes.

---

### Slide: Efficiency Comparison Table (~15s)

The detailed breakdown confirms it. ResNet-18 SVM leads NetScore at 45.4 — 11 million parameters, under 3 ms inference. Classifier choice barely moves NetScore since inference is backbone-dominated. ConvNeXt's 28 million parameters penalise its efficiency score despite leading on accuracy.

---

### Slide: Why ConvNeXt Dominates (~15s)

It's the stem. ConvNeXt's 4×4 patchify gets you an 8×8 feature map in one step — enough for the 7×7 depthwise kernels to work with. ResNet's cascading stride-2 conv plus max-pool collapses 32-pixel input too aggressively; deeper layers get spatially degenerate feature maps.

---

### Slide: Why Architecture Matters More Than Depth (~15s)

Within ResNets, depth adds nothing — 18, 34, 50 cluster within 1 pp. ConvNeXt's architecture gap is the big effect. EfficientNet underperforms because it was designed for 224 — at 32 pixels, Squeeze-and-Excitation modules and depthwise kernels get degenerate inputs.

---

### Slide: Generalisation Gap (~10s)

Val–test gap is 0.6–3 pp throughout — linear models on frozen features generalise well. No overfitting concern here.

---

### Slide: Resolution Impact & t-SNE (~15s)

The biggest result: both backbones lose about 50 points going from 224 to 32. ConvNeXt's 10-point lead is preserved across resolutions, so the advantage is architectural, not resolution-dependent. The t-SNE plots confirm it visually — tight clusters at 224, collapse at 32.

---

### Slide: Backbone Downselection (~15s)

We carry forward ConvNeXt-Tiny — highest accuracy — and ResNet-18 — fastest inference — into Approaches 2 and 3. One modern, one classical, side by side.

---

### Slide: Key Takeaways (~20s)

Six things to remember. ConvNeXt-Tiny dominates by 10+ pp, driven by the patchify stem. Architecture matters more than depth — ResNets cluster within 1 pp. Resolution is the biggest factor — 50-point drop. EfficientNet struggles at 32 pixels. Generalisation is solid. And honest caveats: single seed, default hyperparameters, Mini-ImageNet overlaps ImageNet-1K.

---

## Likely Q&A — prep notes

- **"Why is ConvNeXt so much better at 32×32?"** Patchify stem vs cascading downsampling — but we haven't done the ablation.
- **"Why not ViT or Swin?"** Scope — one per family; ConvNeXt represents the modern design.
- **"Is the 50 pp drop universal?"** No — observed on two backbones on Mini-ImageNet; not claimed as a general bound.
- **"ResNet-50 V2 vs V1?"** V2 is ~4 points higher on ImageNet-1K — real confound. Would need to pin all ResNets to V1.
- **"Why no hyperparameter tuning?"** Compute budget constraint — focus was on backbone comparison, not classifier optimisation.
