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
  .columns-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
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
  .footnote {
    position: absolute;
    bottom: 1.2em;
    left: 2em;
    right: 2em;
    font-size: 14px;
    color: #666;
    border-top: 1px solid #ccc;
    padding-top: 0.3em;
  }
  section.lead h1 {
    font-size: 48px;
    text-align: center;
  }
  section.lead h2 {
    font-size: 32px;
    text-align: center;
    color: #555;
  }
---

<!--
==========================================================================
PRESENTER BREAKDOWN
==========================================================================
Group 23 — David, Hoa, Khoa
Presenter 1 (David, 4 min) — Intro + Part 1 (Problem/Data/Pre-processing)
    + Part 2 (Three Approaches) + Part 3 (Approach 1 — Classical ML)
    6 slides total, each showcasing a single plot at readable size:
      1. Title + Research Question (45s)
      2. Dataset & Pre-processing (60s)
      3. Three-Approach Attribution Design (60s)
      4a. A1 — Architecture beats depth (30s) — efficiency_tradeoff plot
      4b. A1 — Resolution dominates (25s) — t-SNE plot
      4c. A1 — Classifier barely matters + handoff (20s) — NetScore plot
    Each slide is designed to demonstrate understanding of *why* we made
    our choices and *why* results turned out the way they did — not to
    recite numbers. Full speaker script in part1_script.md.
Presenter 2 (Hoa) — Part 4 (Approach 2 — Fine-Tuning)
Presenter 3 (Khoa) — Part 5 (Approach 3 — Scratch) + Part 6 (Analysis) + Part 7 (Conclusions) + References
==========================================================================
-->

<!-- _class: lead -->

# CS5242 Project — Image Classification on Mini-ImageNet

## How much of our accuracy comes from pretraining, fine-tuning, or training from scratch?

**Group 23** — David · Hoa · Khoa

*A three-lens attribution study — 100 classes, small data, low resolution.*

<!--
[Presenter 1 (David) — 0:00–0:45, ~45s]
Frame the project as an ATTRIBUTION question, not an accuracy race — that
matches the professor's philosophy (Project.pdf p.21: "It is about to
understand why it works and why it does not"). Say:

  - "We're Group 23 — David, Hoa, and Khoa."
  - "The task is 100-class classification on Mini-ImageNet, but the
     scientific question isn't 'can we classify it?' — it's *how much of
     our accuracy comes from pretraining, versus task adaptation, versus
     end-to-end learning from scratch*."
  - "We answer that with three approaches — frozen pretrained features,
     fine-tuning, and from scratch — designed so each isolates one
     contribution."
  - "I'll cover the problem setup, our three-approach design, and
     Approach 1. Hoa takes Approach 2, Khoa closes with Approach 3 and
     the cross-approach analysis."

Don't rush — this slide sets the frame for everything else.
-->

---

<!-- _class: "" -->
<style scoped>
section { font-size: 20px; }
h1 { font-size: 30px; }
h3 { font-size: 22px; }
blockquote { font-size: 18px; }
</style>

# Dataset & Pre-processing — Decisions With a Reason

<div class="columns">
<div>

**Mini-ImageNet** [4] — 100 classes ⊂ ImageNet-1K, 50k/10k/5k train/val/test. **Perfectly balanced** (plot →) → no class-weighting confound in later comparisons.

### Decisions we made — and why:

**Two resolutions.** 224×224 matches the **pretraining regime** (upper bound on pretrained feature quality). 32×32 is a deliberate **off-regime stress test** — it forces architectures outside their design window and exposes which ones have the right inductive biases for low-resolution input.

**Our own normalisation stats** (mean ≈ 0.48, 0.45, 0.41 — *not* ImageNet defaults). Resizing to 32 px **shifts** the per-channel pixel distribution; using our train-split stats keeps A3's inputs properly zero-mean.

**Augmentation splits by approach.** A1 caches features once — no train-time aug. A2 & A3 train end-to-end with random crop + horizontal flip.

</div>
<div>

![w:420](../../experiments/eda/class_distribution_all_splits.png)

![w:260](../../experiments/eda/visual_grid.png)

</div>
</div>

<!--
[Presenter 1 (David) — 0:45–1:45, ~60s]
Each pre-processing choice is a chance to demonstrate understanding —
narrate the *reason*, not the value:

  - Balance plot: "Flat across all three splits — later accuracy gaps
    can't be imbalance artefacts."
  - Two resolutions: "224 matches what the backbones were pretrained on
    — it's the ceiling. 32 is *deliberately* off-regime, because that's
    where architectural choices — the stem design, the inductive biases —
    actually matter. At 224 most modern backbones look similar."
  - Own normalisation: "The 32-px resize changes the per-channel
    statistics meaningfully. Using ImageNet's defaults would leave A3's
    inputs off-centre. Trade-off acknowledged: A2 might marginally prefer
    ImageNet defaults because pretrained BatchNorm is calibrated to them
    — but the impact is <1 pp and A3 benefits more."
  - Augmentation: "A1 has no train-time aug because we extract features
    once and cache — any augmentation would force re-extraction. A2 and
    A3 train end-to-end so augmentation is free and useful."

The message: every pre-processing choice was deliberate, tied to the
downstream experiments.
-->

---

# Three-Approach Attribution Design

<div class="columns-3">
<div>

### A1 — Frozen features
**Isolates feature quality** from pretraining alone.
Pretrained backbone (frozen) + linear SVM / LogReg. Cheap, interpretable baseline.

</div>
<div>

### A2 — Fine-tuning
**Measures task-specific adaptation.**
Pretrained init + selective unfreezing.
**Our proposed improvement.**

</div>
<div>

### A3 — Scratch
**Measures in-domain signal alone.**
Random init, full training — no ImageNet prior.

</div>
</div>

> **Why three, not one?** Reporting a single accuracy number conflates feature quality, task adaptation, and end-to-end signal. Running all three lets us **attribute** performance to each source independently — that's the scientific contribution of the design.

> **Why use pretrained models?** As a **well-understood baseline to build on**. A1 uses frozen pretrained backbones — our starting point. A2 fine-tunes the same backbones — our proposed improvement. A3 trains the same architectures from scratch — the first-principles comparison. The pretrained backbone is the common anchor each approach measures against.

<!--
[Presenter 1 (David) — 1:45–2:45, ~60s]
This is the *framing* slide — the professor needs to see why we're
running three approaches rather than one. Land these beats:

  - Each approach answers a different question. A1 = "how good are the
    pretrained features by themselves?" A2 = "how much does adapting
    them help?" A3 = "what can we learn without any ImageNet prior?"
  - The three together form an attribution design: we can separate
    pretraining's contribution from task adaptation's contribution
    from end-to-end learning. A single number can't do that.
  - Pretrained-model justification — this matters for grading (see
    Project.pdf p.23: "use of pre-trained networks should be well
    justified"). Our line: pretrained backbones are a well-understood
    baseline. A1 uses them frozen; A2 fine-tunes the same backbones
    (our proposed improvement); A3 trains the same architectures from
    scratch. The pretrained backbone is the common anchor we build on
    and compare against.

Don't read the columns — point at them and narrate the roles. The
blockquotes are for the audience to read while you talk.
-->

---

<!-- _class: "" -->
<style scoped>
section { font-size: 22px; }
h1 { font-size: 28px; }
</style>

# A1 — Architecture Beats Depth at 32×32

![w:860](../../experiments/classical_ml/analysis_efficiency_tradeoff.png)

- **RN-18 / 34 / 50 within 1 pp** despite 2× the parameters — extra depth gives no headroom at 32×32.
- **ConvNeXt-Tiny wins by 10+ pp** — its **4×4 patchify stem** keeps an 8×8 feature map for the 7×7 depthwise kernels; ResNet's stride-2 conv + max-pool cascade collapses 32-px input. Deeper ResNets inherit the same bad stem.

<!--
[Presenter 1 (David) — 2:45–3:15, ~30s]
Point at the LEFT panel (Accuracy vs Parameters):
  - "ResNet-18, 34, 50 all sit around 33% regardless of parameter count
    — depth doesn't buy us accuracy at 32 pixels."
  - "ConvNeXt-Tiny jumps 10+ pp."
Then the mechanism: "It's the stem. ConvNeXt's 4×4 patchify shrinks
spatial dims in one step and leaves enough structure for the 7×7
depthwise kernels. ResNet's stem squeezes 32-px input down too fast —
and deeper ResNets inherit the same bad stem."
-->

---

<!-- _class: "" -->
<style scoped>
section { font-size: 22px; }
h1 { font-size: 28px; }
</style>

# A1 — Resolution Dominates Any Architecture Gap

<div class="columns">
<div>

![w:480](../../experiments/classical_ml/downselected_backbones_tsne.png)

</div>
<div>

- **Both backbones drop ~50 pp** from 224 → 32 — **larger than any architecture gap.**
- t-SNE shows the mechanism: at **224**, classes form **tight separable clusters**; at **32**, clusters **collapse together**.
- Same backbone, same features — at 32 px there's just no spatial extent left to separate classes.

</div>
</div>

<!--
[Presenter 1 (David) — 3:15–3:40, ~25s]
"Resolution is the biggest single factor — bigger than any architecture
gap. Both backbones lose about 50 points from 224 to 32."
Point at the t-SNE: "At 224, every class forms a tight separable cluster.
At 32, they collapse together. Same backbone, same features — but at
32 pixels there's just no spatial extent left to tell classes apart."
-->

---

<!-- _class: "" -->
<style scoped>
section { font-size: 22px; }
h1 { font-size: 28px; }
</style>

# A1 — Classifier Barely Matters (→ We *Are* Measuring Features)

![w:860](../../experiments/classical_ml/analysis_netscore_combined.png)

- **SVM vs LogReg bars overlap for each backbone** — within 0.5 NetScore.
- Signal is in the **backbone features**, not the classifier — **validating A1's design**, which was built to isolate feature quality.

**Carry forward** to A2 & A3: **ConvNeXt-Tiny** + **ResNet-18**.

<!--
[Presenter 1 (David) — 3:40–3:55, ~15s]
"The linear classifier barely matters. SVM and logistic regression land
within half a NetScore point for every backbone. That's actually a
VALIDATION — we set out to measure feature quality, and swapping the
classifier doesn't change the ranking, so we *are* measuring feature
quality."
"For Approaches 2 and 3 we carry forward two backbones: ConvNeXt-Tiny
and ResNet-18."

[Presenter 1 (David) — 3:55–4:00, ~5s HANDOFF]
"Over to Hoa."
-->

---

<!--
==========================================================================
END OF PRESENTER 1 (4 min) — handoff to HOA (Presenter 2) below
==========================================================================
-->

<!-- _class: lead -->

# Part 4
## Approach 2 — Fine-Tuning

---

<!-- _class: lead -->

# Part 5
## Approach 3 — Training from Scratch

---

<!-- _class: lead -->

# Part 6
## Analysis Across All Approaches

---

# Cross-Approach Analysis

---

<!-- _class: lead -->

# Part 7
## Conclusions

---

# Conclusions

---

<!-- _class: "" -->
<style scoped>
section { font-size: 17px; }
h1 { font-size: 28px; }
p { margin: 0.4em 0; }
</style>

# References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[2] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF CVPR*, 2022, pp. 11976–11986.

[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[4] O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra, "Matching networks for one-shot learning," in *Proc. NeurIPS*, 2016.

[5] W. Luo, Y. Li, R. Urtasun, and R. Zemel, "Understanding the effective receptive field in deep convolutional neural networks," in *Proc. NeurIPS*, 2016.

[6] A. Wong, "NetScore: Towards universal metrics for large-scale performance analysis of deep neural networks," in *Proc. Int. Conf. Image Analysis and Recognition (ICIAR)*, 2019.

[7] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin, "LIBLINEAR: A library for large linear classification," *JMLR*, vol. 9, pp. 1871–1874, 2008.

---

<!-- _class: lead -->

# Thank You
