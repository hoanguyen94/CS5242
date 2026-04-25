# CS5242 — Mini-ImageNet Classification

Group 23 project for CS5242 (NUS). We benchmark three families of approaches on
[`timm/mini-imagenet`](https://huggingface.co/datasets/timm/mini-imagenet) (100 classes,
50k / 10k / 5k train/val/test) and analyse the accuracy–compute trade-off across them.

The full write-up lives in [main.ipynb](main.ipynb); this README is a map of the repo.

## Approaches benchmarked

[main.ipynb](main.ipynb) is organised into five sections:

1. **Data exploration & preprocessing** — class balance, channel-wise mean/std
   on a 5k-image sample, transform visualisation at 32×32 and 224×224.
2. **Classical ML on frozen features** — five ImageNet-pretrained backbones
   (ConvNeXt-Tiny, ResNet-18/34/50, EfficientNet-B0) used as frozen feature
   extractors, with Linear SVM and Logistic Regression heads. 32×32 vs 224×224
   resolution sweep on the two downselected backbones.
3. **Transfer learning** — frozen backbone, last-stage fine-tune, full
   fine-tune, and LoRA on ConvNeXt-Tiny / ResNet-18.
4. **Training from scratch** — baseline + early-stopping, scheduler sensitivity,
   and learning-rate regime studies. Includes an Inception-inspired multi-scale
   variant.
5. **Cross-approach discussion** — accuracy, NetScore, and efficiency comparison.

## Repository layout

- [main.ipynb](main.ipynb) — top-level notebook that runs every experiment and
  contains the narrative report.
- [main.py](main.py) — CLI entry point that dispatches the same tasks
  (`explore`, `visualize_transforms`, `features_ml`, `finetune`, `scratch`,
  `tsne`).
- [src/](src/) — implementation package:
  - [src/data_processing/](src/data_processing/) — dataset loading, transforms,
    EDA helpers.
  - [src/methods/](src/methods/) — `classical_ml.py`, `finetune.py`,
    `train_scratch.py`.
  - [src/model.py](src/model.py) — backbone construction, feature extraction,
    plotting.
  - [src/utils.py](src/utils.py) — seeding, device selection, dataloader
    builders.
- [experiments/](experiments/) — per-task outputs (metrics, figures,
  checkpoints) keyed by task name.
- [report/](report/) — written report, slides, and exported PDFs.
- [requirements.txt](requirements.txt) / [environment.yml](environment.yml) —
  dependencies.

## Workflow

Code is **edited locally** (e.g. VSCode), training runs on **Google Colab GPU**,
and stable code is **committed to GitHub**.

To run [main.ipynb](main.ipynb) against a Colab runtime from VSCode, install the
Colab extension and connect to a Colab runtime — see
[the Google blog post](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/)
for setup details. The first cell mounts Drive and `cd`s into the project
directory; subsequent cells import from [src/](src/) directly.

## Running individual tasks via CLI

Each notebook section has a corresponding [main.py](main.py) invocation, e.g.

```bash
# Dataset stats and class distribution
python main.py --task explore --save_dir experiments/eda

# Classical ML: frozen ConvNeXt-Tiny + Logistic Regression at 32×32
python main.py --task features_ml --backbone convnext_tiny \
    --clf_type logreg --img_size 32 --save_dir experiments/classical_ml

# Transfer learning: full fine-tune of ResNet-18
python main.py --task finetune --backbone resnet18 \
    --freeze_policy none --epochs 30 --save_dir experiments/finetune

# Train from scratch
python main.py --task scratch --backbone ournet \
    --epochs 60 --lr_scheduler cosine --save_dir experiments/train_from_scratch
```

Run `python main.py --help` for the full argument list.
