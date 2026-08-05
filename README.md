# Defect Detection (Severstal)

Rare-class steel defect work on the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) dataset.

Same research thread, two evaluation settings:

| Track | Task | Code | What it answers |
|-------|------|------|-----------------|
| **A. Classification** | multi-label CNN | `app/models/` | Do class weights / oversampling / cVAE image synth help rare-class F1? |
| **B. Segmentation** | U-Net + mask-aware gens | `src/rare_defect/` | Does mask-conditioned generation + validation-utility selection improve rare-class Dice / FNR? |

Track A results are already in `results/results.csv`. Track B is the stronger framing for the final write-up (pixel masks, Copy-Paste, cGAN/StyleGAN/Diffusion + sample selection). cVAE stays in Track A only.

## Layout

```
defect-detection/
  app/models/          # Track A: CNN + cVAE classification scripts
  notebooks/           # label / imbalance / VAE split builders
  results/             # Track A metrics table
  src/rare_defect/     # Track B: data, models, training, metrics, report
  scripts/             # Track B entrypoints
  configs/default.yaml
  data/
    severstal/                     # Track A train/val CSVs (tracked)
    splits/                        # Track B frozen Train/Val/Test ids (tracked)
    severstal-steel-defect-detection/   # Kaggle extract (local only)
```

## Data setup

Put the Kaggle extract at `data/severstal-steel-defect-detection/` with:

```
train.csv
train_images/
test_images/          # unused for metrics
sample_submission.csv
```

If you still have it under the course `final_project/` folder, a Windows junction works:

```powershell
cmd /c mklink /J data\severstal-steel-defect-detection ..\final_project\severstal-steel-defect-detection
```

## Install

```bash
cd defect-detection
uv sync
# or: pip install -e .
```

## Track A — classification (existing)

From `app/models/`:

```bash
python train_baseline.py
python train_weighted.py
python train_oversampled.py
python train_cvae.py
python train_vae_augmented.py
python train_vae_oversampled.py

python evaluate_baseline.py
# … matching evaluate_*.py scripts
```

Notebooks under `notebooks/` build the multi-label CSVs and VAE-augmented splits.

## Track B — segmentation + generative selection

```bash
python scripts/smoke_test.py
python scripts/prepare_splits.py   # skip if data/splits/ already present

python scripts/run_experiment.py --arm baseline
python scripts/run_experiment.py --arm weighted
python scripts/run_experiment.py --arm copy_paste

python scripts/train_generator.py --model cgan
python scripts/train_generator.py --model stylegan
python scripts/train_generator.py --model diffusion

python scripts/run_experiment.py --arm cgan_selected
python scripts/run_experiment.py --arm stylegan_selected
python scripts/run_experiment.py --arm diffusion_selected

python scripts/evaluate.py --checkpoint runs/arm_baseline/best.pt --demo-report
```

Primary Track B metrics: mean Dice, Class-2 Dice, FNR, expected cost (`FN ≫ FP`). Synthetics never enter the final test set.

## How the two tracks fit together

1. Track A showed classical rebalancing and naive VAE image synth on a **classification** head — useful baselines and negative/partial results (see `results/results.csv`).
2. Track B moves to **segmentation**, uses **mask-aware** generators, and adds **validation-utility sample selection** so only synthetics that help rare-class Dice on held-out real data are kept.

You can cite Track A as the exploratory phase and Track B as the main experiment in the report.
