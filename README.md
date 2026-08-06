## Unified classifier evaluation

`evaluate_all.py` reruns every completed DefectCNN classifier on the one committed
validation split.  It does not retrain or modify any model.  The seven required
methods are: `baseline`, `weighted`, `oversampled`, `vae_augmented`,
`vae_oversampled`, `gan_oversampled`, and `rl_oversampled`.

### Prerequisite: labelled validation images

The committed `data/severstal/val_split.csv` was built from Kaggle's labelled
`train_images` files.  **Do not use `test_images`:** Kaggle supplies no labels
for them, so precision, recall, PR-AUC, and false-negative cost cannot be
calculated.  Authenticate Kaggle on the machine first (for example with a
Kaggle API token), then download the competition files:

```bash
uv run --with kagglehub python - <<'PY'
import kagglehub
print(kagglehub.competition_download('severstal-steel-defect-detection'))
PY
```

Pass the downloaded directory's `train_images` subdirectory to the command
below.  Alternatively, place or symlink it at `data/severstal/train_images`.

### Exact evaluation command

```bash
uv run python evaluate_all.py \
  --image-dir /absolute/path/from/kaggle/train_images \
  --threshold 0.5 \
  --fn-cost 10 --fp-cost 1 \
  --fn-fp-ratios 1,2,5,10,20 \
  --output-dir results/unified
```

The default threshold-dependent metrics use the same historical rule as the old
scripts: `sigmoid(logit) > 0.5`.  This makes it possible to compare the new
precision, recall and F1 values against `results/results.csv`.  A discrepancy
larger than the configured rounding tolerance makes the command exit non-zero
and is recorded in `results/unified/legacy_comparison.csv`.

### Outputs

- `metrics.csv`: every per-class, macro-average, and total metric row; it
  includes precision, recall, F1, average precision, PR-AUC, FNR, TP, FP, TN,
  FN, and the configured linear cost.
- `predictions.csv.gz`: gzip-compressed, auditable rows per method, image, and
  class with `y_true`, sigmoid `y_score`, `y_pred`, and threshold.  Read it
  directly with `pandas.read_csv("predictions.csv.gz")`, or regenerate it with
  the evaluation command above.
- `legacy_comparison.csv`: the old and rerun threshold-0.5 metrics with deltas.
- `cost_sensitivity.csv` and `figures/cost_sensitivity.png`: explicitly labelled
  FN:FP scenario assumptions.  For each ratio, FP cost is held at `--fp-cost`
  and FN cost is `ratio * FP cost`.
- `figures/model_comparison_macro_metrics.png` and one
  `figures/precision_recall_class_*.png` image per class.
- `run_manifest.json`: command, methods, class ordering, input hashes, and
  generated artifact list for reproducibility.

### Focused tests

```bash
uv run python -m unittest discover -s tests -v
```
