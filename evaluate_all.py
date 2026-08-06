"""CLI entry point for unified, reproducible evaluation of all completed classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evaluation.unified import EvaluationConfig, EvaluationInputError, run_evaluation


def parse_ratios(value: str) -> tuple[float, ...]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cost ratios must be comma-separated numbers") from exc
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("cost ratios must contain one or more positive numbers")
    return ratios


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=root / "data/severstal/train_images", help="Directory containing the labelled Kaggle train_images")
    parser.add_argument("--val-csv", type=Path, default=root / "data/severstal/val_split.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "app/models")
    parser.add_argument("--legacy-results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results/unified")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fn-cost", type=float, default=10.0, help="Configured default cost for one false negative")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="Configured default cost for one false positive")
    parser.add_argument("--fn-fp-ratios", type=parse_ratios, default=(1.0, 2.0, 5.0, 10.0, 20.0), help="Scenario FN:FP ratios, e.g. 1,2,5,10,20")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--legacy-tolerance", type=float, default=0.0001)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    config = EvaluationConfig(
        project_root=Path(__file__).resolve().parent,
        image_dir=args.image_dir,
        val_csv=args.val_csv,
        checkpoint_dir=args.checkpoint_dir,
        legacy_results_csv=args.legacy_results,
        output_dir=args.output_dir,
        threshold=args.threshold,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
        fn_fp_ratios=args.fn_fp_ratios,
        batch_size=args.batch_size,
        device=args.device,
        legacy_tolerance=args.legacy_tolerance,
    )
    try:
        manifest = run_evaluation(config)
    except (EvaluationInputError, ValueError) as exc:
        print(f"Evaluation blocked: {exc}", file=sys.stderr)
        return 2
    print("Evaluated methods:", ", ".join(manifest["methods"]))
    print("Output directory:", config.output_dir)
    if not manifest["legacy_matches"]:
        print("Evaluation completed, but legacy results did not reproduce within rounding tolerance. See legacy_comparison.csv.", file=sys.stderr)
        return 3
    print("Legacy precision, recall, and F1 reproduced within rounding tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
