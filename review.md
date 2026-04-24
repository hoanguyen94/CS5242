Section 1 -- Hoa: Missing description of label distribution in training set and test set

Section 2 -- Khoa:

Section 3 -- David:

**1. Section 3.5 Analysis — LoRA described as a stronger parameter bottleneck than classifier-only.**
The text says *"classifier-only training and **especially** LoRA impose a stronger parameter bottleneck"*. By the notebook's own trainable-parameter counts, LoRA has ~0.61 M trainable params vs. classifier-only's ~0.078 M — LoRA is **~8× less** bottlenecked, not "especially" more so. The structural (low-rank) constraint is not the same as having fewer trainable parameters, and the sentence conflates the two.

**2. Section 3.5 Analysis — ResNet-18 LoRA description contradicts its own trainable-parameter count.**
The analysis states: *"our LoRA implementation only injects adapters into nn.Linear layers while excluding the classifier head. Therefore, ResNet-18 receives little internal representational adaptation from LoRA."*
Torchvision's ResNet-18 has exactly **one** `nn.Linear` layer — `model.fc`, which is the classifier head. The reported trainable params for the ResNet-18 LoRA run (56,200) minus the classifier-only baseline (51,300) gives **4,900 LoRA params**, which equals exactly `(512 + 100) × 8` — a LoRA adapter on that `fc` layer with rank 8. So either (a) the classifier head *is* being adapted (contradicting "excluding the classifier head"), or (b) there should be zero LoRA adapters on ResNet-18. The explanation doesn't match the numbers.

**3. Section 3.1 — ConvNeXt-Tiny classifier description is imprecise.**
*"For convnext_tiny, the classifier is the last Linear layer inside model.classifier, which maps the pooled 768-dimensional representation to the 100 classes."*
Torchvision's `convnext_tiny.classifier` is `Sequential(LayerNorm2d(768), Flatten, Linear(768, C))`. The 78,436 trainable params reported in 3.1 is 76,900 (Linear) + 1,536 (LayerNorm2d) — so when `policy="backbone"` is run, the LayerNorm is also trained, not just "the last Linear layer".

Section 4 -- Hoa: Uncomment your code. 
- I also comment direcly on your code. 

Section 5 -- David:

**4. Accuracy hierarchy claim is only partially true.**
*"full fine-tuning (Approach 2) > frozen features (Approach 1) ≈ lightweight from-scratch models (Approach 3)"*
The table contradicts the "≈": Approach 1 SVM + ConvNeXt-Tiny is **43.28 %**, clearly above Approach 3's best (ConvNeXt-Lite-Inception at **33.20 %**). The rough equivalence holds only for the ResNet-18 SVM row (32.52 %) vs. ConvNeXt-Lite (32.32 %); it is then generalized across the entire approach.

**6. NetScore definition is cited to Wong but uses a modified formula.**
Section 5 describes NetScore as capturing "predictive effectiveness and computational cost", and Section 3.5 cites Wong [8]. Wong's original NetScore is `20·log10(A^α / (P^β · M^γ))` with **M = MACs/FLOPs**. The notebook substitutes **inference time T in ms** for M (`20·log10(A² / (√T · √P))`). That's a legitimate variant, but it isn't flagged as a modification of Wong's metric.
