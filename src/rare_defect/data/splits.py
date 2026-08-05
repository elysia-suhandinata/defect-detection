"""Stratified Train/Val/Test from labeled Severstal train images only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _image_labels(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if "EncodedPixels" in df.columns:
        df = df[df["EncodedPixels"].notna() & (df["EncodedPixels"].astype(str).str.len() > 0)]

    all_ids = sorted({p.name for p in (train_csv.parent / "train_images").glob("*.jpg")})
    labels: dict[str, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        labels[str(row["ImageId"])].add(int(row["ClassId"]))

    rows = []
    for image_id in all_ids:
        classes = labels.get(image_id, set())
        rows.append(
            {
                "ImageId": image_id,
                "has_defect": int(len(classes) > 0),
                "has_class_2": int(2 in classes),
                "multi_label": int(len(classes) > 1),
                "classes": sorted(classes),
            }
        )
    return pd.DataFrame(rows)


def _stratum_key(row: pd.Series) -> str:
    return f"d{row.has_defect}_c2{row.has_class_2}_m{row.multi_label}"


def make_splits(
    raw_root: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Freeze a real-only split. Never put synthetics into test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _image_labels(raw_root / "train.csv")
    rng = np.random.default_rng(seed)

    train_ids, val_ids, test_ids = [], [], []
    for _, group in meta.groupby(meta.apply(_stratum_key, axis=1)):
        ids = group["ImageId"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val += n_test
            n_test = 0
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    splits = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
    for name, ids in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    summary = {
        "seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "class2_in_test": int(meta.set_index("ImageId").loc[test_ids, "has_class_2"].sum())
        if test_ids
        else 0,
        "note": "Real images only. Do not add synthetics to test.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_split_ids(splits_dir: Path, split: str) -> list[str]:
    path = Path(splits_dir) / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
