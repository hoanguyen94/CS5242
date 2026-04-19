---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
    color: #1565C0;
  }
  h2 {
    font-size: 30px;
    color: #1976D2;
  }
  table {
    font-size: 18px;
    margin: 0 auto;
  }
  th {
    background-color: #1976D2;
    color: white;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  blockquote {
    border-left: 4px solid #1976D2;
    background: #E3F2FD;
    padding: 0.5em 1em;
    font-size: 20px;
  }
  .highlight {
    color: #C62828;
    font-weight: bold;
  }
  img {
    display: block;
    margin: 0 auto;
  }
---

# Traditional ML on Mini-ImageNet
## Pretrained Backbones as Frozen Feature Extractors

**CS5242 Project — Section 2**

---

# Agenda

1. **Problem & Motivation**
2. **Approach: Frozen Backbone + Classical Classifier**
3. **Experimental Design**
4. **Results: Backbone & Classifier Comparison (32×32)**
5. **Results: Resolution Impact (224×224 vs 32×32)**
6. **t-SNE Feature Visualisation**
7. **Key Findings & Backbone Downselection**
8. **Conclusion & Next Steps**

---

# Problem & Motivation

**Task:** 100-class image classification on **Mini-ImageNet** (50K train / 10K test)

**Challenge:** Mini-ImageNet is small by deep learning standards — end-to-end training risks overfitting

**Our approach:** Use pretrained ImageNet backbones as **frozen feature extractors**, then train lightweight classical classifiers on the extracted features

<div class="columns">
<div>

### Advantages
- **Data efficient** — only classifier head is learnable
- **Fast** — single forward pass + seconds-to-minutes training
- **Interpretable** — isolates backbone effect

</div>
<div>

### Limitations
- No task-specific feature adaptation
- Resolution sensitivity (pretrained for 224×224)
- Linear classifier ceiling

</div>
</div>

---

# Approach: Frozen Backbone + Classical Classifier

```
Image → [Pretrained Backbone (frozen)] → Global Avg Pool → Feature Vector → [SVM / LogReg] → Class
```

**Pipeline:**
1. Load pretrained ImageNet backbone — freeze all weights
2. Extract feature vectors via forward pass (one-time cost)
3. Train classical classifier on extracted features

**Classifiers:**
- **Linear SVM** — maximises geometric margin (hinge loss + L2)
- **Logistic Regression** — minimises cross-entropy (calibrated probabilities + L2)

---

# Experimental Design

### Backbones (3 architecture families)

| Backbone | Family | Params | Feature Dim | Key Innovation |
|---|---|---|---|---|
| ConvNeXt-Tiny | ConvNeXt (2022) | 27.9M | 768 | 7×7 DW-Conv, LayerNorm, GELU |
| ResNet-18 | ResNet (2015) | 11.2M | 512 | Basic residual blocks |
| ResNet-34 | ResNet | 21.3M | 512 | Deeper basic blocks |
| ResNet-50 | ResNet | 23.7M | 2048 | Bottleneck blocks |
| EfficientNet-b0 | EfficientNet (2019) | 4.1M | 1280 | MBConv + SE + compound scaling |

### Experiment Grid
- **32×32**: All backbones × {SVM, LogReg}
- **224×224**: ConvNeXt-Tiny & ResNet-18 × SVM (downselected)

---

# Results: Test Accuracy at 32×32

| Backbone | SVM | LogReg | Winner |
|---|---|---|---|
| **ConvNeXt-Tiny** | **43.28%** | **44.16%** | LogReg (+0.88pp) |
| ResNet-18 | 32.52% | 32.46% | SVM (+0.06pp) |
| ResNet-34 | 33.08% | 33.16% | LogReg (+0.08pp) |
| ResNet-50 | 27.34% | 33.22% | LogReg (+5.88pp) |
| EfficientNet-b0 | 23.88% | 24.72% | LogReg (+0.84pp) |

> **ConvNeXt-Tiny leads by 10+ pp** across both classifiers.
> LogReg ≥ SVM for all backbones at 32×32 — noisy features favour probabilistic calibration over max-margin.

---

# Why ConvNeXt-Tiny Dominates at 32×32

The key lies in how each architecture's **stem** processes low-resolution input:

<div class="columns">
<div>

### ConvNeXt-Tiny Stem
- **4×4 stride-4 patchify** convolution
- 32×32 input → **8×8 feature map** after stem
- Subsequent **7×7 depthwise kernels** still have meaningful spatial extent to work with
- LayerNorm stabilises activations; GELU preserves gradient flow

</div>
<div>

### ResNet Stem
- **7×7 stride-2** conv + **3×3 stride-2** max-pool
- 32×32 input → **8×8** after stem, then quickly **1×1** through residual stages
- Deeper layers receive **spatially degenerate** feature maps — no local structure to exploit
- BatchNorm statistics are calibrated for 224×224 distributions

</div>
</div>

> ConvNeXt's patchify stem is resolution-adaptive: it reduces spatial dims in one step without the cascading downsampling that collapses small inputs in ResNets.

---

# Why LogReg Outperforms SVM at 32×32

At 32×32, extracted features are **noisy and less linearly separable**. This shifts the advantage from margin-based to probabilistic classifiers:

| Property | SVM (hinge loss) | LogReg (cross-entropy) |
|---|---|---|
| **Objective** | Maximise geometric margin | Minimise calibrated log-likelihood |
| **Noise handling** | Margin amplifies noisy support vectors | Probabilistic weighting downweights ambiguous samples |
| **High-dim behaviour** | Overfits when signal-to-noise is low | L2 + cross-entropy regularises more gracefully |
| **Solver scaling** | QP: scales poorly with dim (2048-d → 3153s) | LBFGS: predictable convergence (2048-d → 680s) |

**ResNet-50 is the extreme case:** 2048-d features with heavy noise at 32×32 → SVM drops to 27.34% while LogReg holds at 33.22% (+5.88 pp). The hinge loss concentrates on a small set of support vectors that are themselves noisy, while cross-entropy loss averages over all samples.

---

# Results: NetScore, Accuracy, Inference & Parameters

![w:1100](analysis_netscore_combined.png)

---

# Results: NetScore & Efficiency (32×32)

$$\text{NetScore} = 20\,\log_{10}\!\left(\frac{A^2}{\sqrt{T}\,\sqrt{P}}\right)$$

| Backbone | Classifier | Test Acc | Infer. Time | Params | Train Time |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | LogReg | **44.16%** | 5.92 ms | 27.9M | **278s** |
| ConvNeXt-Tiny | SVM | 43.28% | 5.96 ms | 27.9M | 1502s |
| ResNet-18 | LogReg | 32.46% | 3.09 ms | 11.2M | 347s |
| ResNet-18 | SVM | 32.52% | **2.89 ms** | **11.2M** | 465s |

> **LogReg trains 1.2–9× faster** than SVM with equal or better accuracy.
> ResNet-18 has the **fastest inference** (≈2.9 ms/image).

---

# Why Architecture Matters More Than Depth

![w:1100](analysis_efficiency_tradeoff.png)

- **ResNets cluster** at 32–33% regardless of depth → resolution-limited, not capacity-limited
- **ConvNeXt-Tiny** leads by 10+ pp — patchify stem + 7×7 DW-Conv preserve spatial structure
- **EfficientNet-b0 collapses** — compound scaling breaks at 7× below design resolution

---

# Why EfficientNet-b0 Fails at 32×32

EfficientNet's **compound scaling** (Tan & Le, 2019) jointly optimises three axes:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

The architecture is **designed for 224×224** ($r = 1.0$). At 32×32 ($r \approx 0.14$):

1. **Squeeze-and-Excitation modules fail** — they globally average-pool feature maps to compute channel attention weights. When pre-pooling maps are 1×1 or 2×2, the "squeeze" output is essentially random noise
2. **MBConv depthwise convolutions underperform** — 3×3 and 5×5 kernels on 1–2 pixel feature maps have no spatial variation to detect
3. **Depth/width are over-scaled for the resolution** — the network has more capacity than the spatial information can support, leading to redundant or contradictory features

> EfficientNet achieves SOTA by balancing all three axes. Collapsing resolution to 14% of design while keeping depth/width fixed **violates the core design principle**.

---

# Resolution Impact: 224×224 → 32×32

![w:750](analysis_accuracy_drop.png)

> Both backbones lose ≈**50 pp** — a fundamental resolution floor for frozen pretrained backbones.
> ConvNeXt-Tiny's 10.6 pp advantage is **preserved** across both resolutions.

---

# Why the ≈50 pp Drop Is Nearly Identical

The uniform ~50 pp drop across both architectures reveals a **shared bottleneck**:

1. **Pretrained filters are calibrated for 224×224** — Gabor-like edge detectors in early layers expect specific spatial frequencies. At 32×32, the Nyquist frequency is 7× lower — fine textures and edges are aliased away before the network even sees them

2. **Receptive field saturation** — at 224×224, a ResNet-18 layer-4 neuron has an effective receptive field of ~100 px (partial image). At 32×32, the same neuron's receptive field covers the **entire image** after just 2 stages, eliminating the hierarchical spatial decomposition that makes CNNs powerful

3. **The gap is preserved (10.6→10.8 pp)** because ConvNeXt's advantage is architectural, not resolution-dependent — its patchify stem, LayerNorm, and wider kernels extract better features at **any** spatial scale

> This suggests ~50 pp is a **hard floor** for frozen ImageNet backbones on 32×32 inputs — task-specific fine-tuning is needed to break through it.

---

# t-SNE Feature Visualisation

![w:680](downselected_backbones_tsne.png)

- **224×224:** ConvNeXt-Tiny → tight clusters (93.88%); ResNet-18 → moderately separated (83.24%)
- **32×32:** ConvNeXt-Tiny retains partial structure; ResNet-18 → near-random scattering

---

# Generalisation Gap

![w:1100](analysis_generalisation_gap.png)

> **Val − Test gap is uniformly small (0.6–3.0 pp)** — linear models on frozen features generalise well.
> Largest gap: ResNet-34 SVM (3.0 pp). Smallest: ResNet-50 LogReg (0.6 pp).

---

# Backbone Downselection for Fine-Tuning

Based on these results, we select **two backbones** for Section 3 (fine-tuning):

<div class="columns">
<div>

### 1. ConvNeXt-Tiny (SOTA)
- **Highest accuracy** at all resolutions
- 93.88% (224×224), 44.16% (32×32)
- Modern architecture innovations
- Best candidate for further improvement

</div>
<div>

### 2. ResNet-18 (Classical Baseline)
- **Fastest inference** (2.9 ms/image)
- Well-established architecture
- Measures how much fine-tuning closes the gap
- Simple & interpretable

</div>
</div>

> This pairing lets us compare fine-tuning strategies (frozen, partial unfreeze, full, LoRA) on a **modern vs classical** architecture.

---

# Downselected Backbones — Comparison

![w:1100](downselected_backbones_comparison.png)

---

# Key Takeaways

1. **ConvNeXt-Tiny is the best backbone** — leads by 10+ pp at any resolution thanks to patchify stem and modern CNN design

2. **Resolution is critical** — 224→32 causes catastrophic ≈50 pp loss for all frozen backbones

3. **LogReg ≥ SVM at 32×32** — noisy features favour probabilistic calibration; LogReg also trains 1.2–9× faster

4. **Architecture > Depth** — ResNet-18/34/50 cluster within 1 pp; design innovations (ConvNeXt) matter far more

5. **EfficientNet-b0 fails at low resolution** — compound scaling breaks down at 7× below design resolution

6. **Strong generalisation** — val-test gap < 3 pp across all configs (linear models + frozen features = implicit regularisation)

---

# Next Steps: Fine-Tuning (Section 3)

With ConvNeXt-Tiny and ResNet-18 as our downselected backbones, we will explore:

- **Frozen backbone** → linear probe (baseline, already done)
- **Partial unfreezing** → unfreeze last N layers
- **Full fine-tuning** → all parameters trainable
- **LoRA** → low-rank adaptation of backbone weights

**Goal:** Determine how much task-specific adaptation can improve over frozen feature extraction, and whether ConvNeXt-Tiny's architectural advantage persists after fine-tuning.

---

<!-- _class: lead -->

# Thank You
## Questions?

**CS5242 Project — Traditional ML Approach**
