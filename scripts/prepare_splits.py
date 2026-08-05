"""Create stratified Train/Val/Test from labeled Severstal train images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rare_defect.config import load_config
from rare_defect.data import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or (ROOT / cfg["data"]["raw_root"])
    out = args.out or (ROOT / cfg["data"]["splits_dir"])
    print(make_splits(
        data_root,
        out,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["seed"],
    ))


if __name__ == "__main__":
    main()
