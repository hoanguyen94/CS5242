# Speaker Script — Presenter 1 (David, Group 23)

**Scope:** Intro + Part 1 (Problem, Data & Pre-processing) + Part 2 (Three Approaches) + Part 3 (Approach 1 — Classical ML). Hands off to **Hoa** (Presenter 2), who takes Approach 2.

**Target runtime: 4 minutes.** 6 slides, each showing a single plot at readable size. The goal is to show we understood *why* we did what we did, not to read numbers off the slide.

**Pacing guide:**

| Slide | Time | Budget |
|---|---|---|
| 1. Title + Research Question | 0:00–0:45 | 45s |
| 2. Dataset & Pre-processing — Decisions With a Reason | 0:45–1:45 | 60s |
| 3. Three-Approach Attribution Design | 1:45–2:45 | 60s |
| 4a. A1 — Architecture beats depth | 2:45–3:15 | 30s |
| 4b. A1 — Resolution dominates | 3:15–3:40 | 25s |
| 4c. A1 — Classifier barely matters + handoff | 3:40–4:00 | 20s |

---

## Slide 1 — Title + Research Question (~45s)

Good [morning / afternoon] — we're Group 23: David, Hoa, and Khoa.

Our task is image classification on Mini-ImageNet — 100 classes.

But the question we really want to answer isn't "can we classify these images?" It's "where does our accuracy come from?" Is it from pretraining? From fine-tuning? Or from training the network from scratch on our data?

To answer that, we run three approaches side by side: frozen pretrained features, fine-tuning, and training from scratch. Each one isolates a different source, so we can tell which part matters.

I'll cover the problem, the data, and Approach 1. Hoa takes Approach 2. Khoa finishes with Approach 3 and the cross-approach analysis.

---

## Slide 2 — Dataset & Pre-processing: Decisions With a Reason (~60s)

Mini-ImageNet has 100 classes. We split it 50,000 / 10,000 / 5,000 for training, validation, and test. The classes are perfectly balanced — you can see that in the plot on the right — so imbalance won't skew our later results.

We made three pre-processing choices, each with a reason.

**First, two resolutions.** 224 by 224 matches what the backbones were pretrained on — that's the best case. 32 by 32 is a deliberate stress test. It forces the models to work outside what they were built for, which is where architecture choices — especially the first layer — actually start to matter.

**Second, we compute our own mean and standard deviation on the training set**, not the ImageNet defaults. Shrinking images to 32 pixels shifts the pixel values a bit, so ImageNet's numbers would be slightly off for us.

**Third, different augmentation per approach.** Approach 1 runs each image through the backbone once and saves the features — so no augmentation during training. Approaches 2 and 3 train end-to-end, with random crops and horizontal flips.

Each choice is tied to a real experiment we run later.

---

## Slide 3 — Three-Approach Attribution Design (~60s)

Three approaches — each answers a different question.

**Approach 1 — frozen features.** A pretrained backbone with the weights frozen, plus a simple linear classifier on top. It tells us how good the pretrained features are on their own. If we swap the classifier and accuracy barely moves, we know we're measuring the backbone, not the classifier.

**Approach 2 — fine-tuning.** We start from the pretrained weights and let some layers update during training. This is our proposed improvement. It tells us how much adapting to Mini-ImageNet adds on top of frozen features.

**Approach 3 — from scratch.** Random starting weights, no ImageNet head start. It tells us how much signal is in Mini-ImageNet alone.

**Why run all three?** A single accuracy number would mix pretraining, adaptation, and in-domain learning into one lump. Running three lets us separate them.

**Why pretrained models in the first place?** As a well-understood baseline to build on. Approach 1 uses pretrained backbones frozen — that's our starting point. Approach 2 fine-tunes the same backbones — that's our proposed improvement. Approach 3 trains the same architectures from scratch — our first-principles comparison. The pretrained backbone is the common anchor that each of our approaches measures against.

---

## Slide 4a — A1: Architecture Beats Depth at 32×32 (~30s)

At 32 pixels, architecture matters much more than depth. Look at ResNet-18, 34, and 50 on the accuracy-versus-parameters plot — they all sit around 33% regardless of whether the model has 11 or 24 million parameters. Adding depth doesn't help. But ConvNeXt-Tiny jumps up by ten-plus points.

The reason is the first layer — the stem. ConvNeXt uses a 4-by-4 patch convolution that shrinks the image in one clean step and leaves enough spatial structure for the later layers to work with. ResNet's stem uses a bigger convolution plus a max-pool, which squeezes a 32-pixel input down too fast. Deeper ResNets inherit the same bad stem, so extra layers can't rescue them.

---

## Slide 4b — A1: Resolution Dominates Any Architecture Gap (~25s)

Resolution is the biggest single factor — bigger than any architecture gap. Both backbones lose about 50 points going from 224 to 32.

The t-SNE plot shows why. At 224, each class forms a tight, separable cluster. At 32, everything collapses together. Same backbone, same features — but at 32 pixels there's just no spatial extent left for the model to tell classes apart.

---

## Slide 4c — A1: Classifier Barely Matters (~15s) + Handoff (~5s)

The linear classifier barely matters. SVM and logistic regression land within half a point on our efficiency score for every backbone — you can see the paired bars sit at almost the same height. That's actually a good sign — it confirms we're measuring feature quality, not the classifier, which validates Approach 1's design.

For Approaches 2 and 3, we carry forward two backbones: **ConvNeXt-Tiny**, the modern one, and **ResNet-18**, the fast classical one.

**Handoff.** Over to Hoa — can fine-tuning close the gap between ConvNeXt and ResNet, and recover some of that 50-point resolution drop?

---

## Likely Q&A — prep notes

- **"Why Mini-ImageNet instead of full ImageNet-1K?"** Two reasons — our compute budget, and the small-data setting (500 images per class) is exactly where the pretraining question is most interesting.

- **"Isn't using pretrained models cheating, since the classes overlap with ImageNet-1K?"** We're using them as a well-understood baseline to build on, not to claim transfer to new data. Approach 2 fine-tunes those same backbones, and Approach 3 trains the same architectures from scratch — so the pretrained model is the common anchor, and our contributions are what we add on top.

- **"Why compute your own mean and standard deviation instead of using ImageNet defaults?"** The 32-pixel resize shifts the pixel distribution a bit. Using our own stats gives cleaner normalisation for Approach 3. Fair point the other way: Approach 2 might slightly prefer ImageNet defaults, because pretrained BatchNorm is calibrated to them — but the effect is under 1 point.

- **"Why is ConvNeXt so much better at 32×32?"** The stem. ConvNeXt's 4-by-4 patch convolution keeps an 8-by-8 feature map, which later layers can work with. ResNet's stem collapses small input too fast. We haven't run a direct ablation, but this explanation fits the data.

- **"Why didn't you include ViT or Swin?"** We picked one representative from each architecture family. ConvNeXt stands in for the modern post-ViT CNN design.

- **"Does the 50-point drop at 32 pixels generalise?"** We've only shown it on two backbones on Mini-ImageNet — so we don't claim it as a general rule. But the explanation (small input collapsing in the stem) would predict something similar for other architectures with the same kind of stem.

- **"ResNet-50 uses V2 weights and the others use V1 — is that a problem?"** Yes, it's a real confound. V2 is about 4 points stronger on ImageNet-1K. To compare ResNet depth cleanly we'd want to pin all of them to V1.

- **"Why not tune hyperparameters?"** Compute budget, and our goal was to compare backbones, not optimise the classifier. Using sklearn defaults keeps the comparison simple and fair.

- **"Why two classifiers?"** To check that what we're measuring is feature quality, not the classifier's inductive bias. SVM and logistic regression land within 1 point for every backbone, which confirms that reading.
