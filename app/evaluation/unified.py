"""End-to-end, auditable evaluation of every completed DefectCNN classifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from app.models.classifier import DefectCNN
from app.models.dataset import SeverstalDataset
from app.evaluation.metrics import calculate_binary_metrics, calculate_cost


CLASS_NAMES = ("class_1", "class_2", "class_3", "class_4")
METHOD_CHECKPOINTS = {
    "baseline": "baseline_cnn.pth",
    "weighted": "weighted_cnn.pth",
    "oversampled": "oversampled_cnn.pth",
    "vae_augmented": "vae_augmented_cnn.pth",
    "vae_oversampled": "vae_oversampled_cnn.pth",
    "gan_oversampled": "gan_oversampled_cnn.pth",
    "rl_oversampled": "rl_oversampled_cnn.pth",
}


@dataclass(frozen=True)
class EvaluationConfig:
    project_root: Path
    image_dir: Path
    val_csv: Path
    checkpoint_dir: Path
    legacy_results_csv: Path
    output_dir: Path
    threshold: float = 0.5
    fn_cost: float = 10.0
    fp_cost: float = 1.0
    fn_fp_ratios: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)
    batch_size: int = 32
    device: str = "auto"
    legacy_tolerance: float = 0.0001


class EvaluationInputError(RuntimeError):
    """Raised before inference when a required input is missing or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_label(path: Path, project_root: Path) -> str:
    """Keep committed manifests reproducible without leaking machine-local paths."""
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return "<external path; supply with --image-dir when rerunning>"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        # This intentionally follows the original evaluation scripts: MPS when
        # available, otherwise CPU.  CUDA is opt-in for transparent comparison.
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, mps, cuda")
    device = torch.device(requested)
    if requested == "mps" and not torch.backends.mps.is_available():
        raise EvaluationInputError("--device mps was requested but MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise EvaluationInputError("--device cuda was requested but CUDA is unavailable")
    return device


def validate_inputs(config: EvaluationConfig) -> pd.DataFrame:
    required_paths = {
        "validation CSV": config.val_csv,
        "image directory": config.image_dir,
        "checkpoint directory": config.checkpoint_dir,
        "legacy results CSV": config.legacy_results_csv,
    }
    missing_paths = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise EvaluationInputError("Missing required inputs:\n- " + "\n- ".join(missing_paths))

    labels = pd.read_csv(config.val_csv)
    required_columns = {"ImageId", *CLASS_NAMES}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        raise EvaluationInputError(f"val_split.csv is missing columns: {sorted(missing_columns)}")
    if labels["ImageId"].duplicated().any():
        raise EvaluationInputError("val_split.csv contains duplicate ImageId values")
    values = labels[list(CLASS_NAMES)].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise EvaluationInputError("validation labels must be binary 0/1 values")

    missing_checkpoints = [
        f"{method}: {config.checkpoint_dir / filename}"
        for method, filename in METHOD_CHECKPOINTS.items()
        if not (config.checkpoint_dir / filename).is_file()
    ]

    missing_images = [image_id for image_id in labels["ImageId"] if not (config.image_dir / image_id).is_file()]
    if missing_images or missing_checkpoints:
        problems = []
        if missing_images:
            preview = ", ".join(missing_images[:10])
            problems.append(
                f"{len(missing_images)} of {len(labels)} validation images are missing from "
                f"{config.image_dir}. Examples: {preview}"
            )
        if missing_checkpoints:
            problems.append("Missing completed classifier checkpoints:\n- " + "\n- ".join(missing_checkpoints))
        raise EvaluationInputError("\n".join(problems))
    return labels


def load_classifier(checkpoint: Path, device: torch.device) -> DefectCNN:
    model = DefectCNN().to(device)
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state_dict = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise EvaluationInputError(f"Incompatible checkpoint {checkpoint}: {exc}") from exc
    model.eval()
    return model


def infer_method(
    method: str,
    checkpoint: Path,
    labels: pd.DataFrame,
    config: EvaluationConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SeverstalDataset(str(config.val_csv), str(config.image_dir))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = load_classifier(checkpoint, device)
    score_batches: list[np.ndarray] = []
    truth_batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_truth in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                raise EvaluationInputError(
                    f"{method} produced shape {tuple(logits.shape)}; expected (batch, {len(CLASS_NAMES)}) raw logits"
                )
            score_batches.append(torch.sigmoid(logits).cpu().numpy())
            truth_batches.append(batch_truth.numpy().astype(np.int8))
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    scores = np.concatenate(score_batches)
    truth = np.concatenate(truth_batches)
    expected_truth = labels[list(CLASS_NAMES)].to_numpy(dtype=np.int8)
    if not np.array_equal(truth, expected_truth):
        raise EvaluationInputError(f"{method} labels changed order during validation DataLoader construction")
    return truth, scores


def make_metric_rows(
    method: str,
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fn_cost: float,
    fp_cost: float,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray | str]]]:
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, np.ndarray | str]] = []
    class_metrics = []
    for index, class_name in enumerate(CLASS_NAMES):
        metrics = calculate_binary_metrics(truth[:, index], scores[:, index], threshold)
        values = metrics.to_dict()
        values.update(
            {
                "method": method,
                "aggregation": "per_class",
                "class": class_name,
                "threshold": threshold,
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "cost": calculate_cost(metrics.fn, metrics.fp, fn_cost, fp_cost),
            }
        )
        rows.append(values)
        class_metrics.append(metrics)
        precision, recall, _ = precision_recall_curve(truth[:, index], scores[:, index])
        curves.append({"method": method, "class": class_name, "precision": precision, "recall": recall})

    metric_keys = ("precision", "recall", "f1", "average_precision", "pr_auc", "false_negative_rate")
    macro = {
        key: float(np.nanmean([getattr(metric, key) for metric in class_metrics])) for key in metric_keys
    }
    macro.update(
        {
            "tp": float(np.mean([metric.tp for metric in class_metrics])),
            "fp": float(np.mean([metric.fp for metric in class_metrics])),
            "tn": float(np.mean([metric.tn for metric in class_metrics])),
            "fn": float(np.mean([metric.fn for metric in class_metrics])),
            "support": float(np.mean([metric.support for metric in class_metrics])),
            "method": method,
            "aggregation": "macro",
            "class": "macro_avg",
            "threshold": threshold,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "cost": float(np.mean([calculate_cost(metric.fn, metric.fp, fn_cost, fp_cost) for metric in class_metrics])),
            "count_aggregation": "mean_per_class",
        }
    )
    total = {
        "tp": int(sum(metric.tp for metric in class_metrics)),
        "fp": int(sum(metric.fp for metric in class_metrics)),
        "tn": int(sum(metric.tn for metric in class_metrics)),
        "fn": int(sum(metric.fn for metric in class_metrics)),
        "support": int(sum(metric.support for metric in class_metrics)),
        "method": method,
        "aggregation": "total",
        "class": "all_classes",
        "threshold": threshold,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "cost": float(sum(calculate_cost(metric.fn, metric.fp, fn_cost, fp_cost) for metric in class_metrics)),
        "count_aggregation": "sum_over_classes",
    }
    rows.extend([macro, total])
    return rows, curves


def make_prediction_rows(
    method: str, labels: pd.DataFrame, truth: np.ndarray, scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    predicted = (scores > threshold).astype(np.int8)
    rows: list[dict[str, Any]] = []
    for sample_index, image_id in enumerate(labels["ImageId"]):
        for class_index, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "method": method,
                    "ImageId": image_id,
                    "class": class_name,
                    "y_true": int(truth[sample_index, class_index]),
                    "y_score": float(scores[sample_index, class_index]),
                    "y_pred": int(predicted[sample_index, class_index]),
                    "threshold": threshold,
                }
            )
    return rows


def compare_to_legacy(metrics: pd.DataFrame, legacy_path: Path, tolerance: float) -> pd.DataFrame:
    legacy = pd.read_csv(legacy_path)
    needed = {"method", "class", "precision", "recall", "f1"}
    if missing := needed.difference(legacy.columns):
        raise EvaluationInputError(f"legacy results file is missing columns: {sorted(missing)}")
    reproduced = metrics.loc[metrics["aggregation"].isin(["per_class", "macro"]), ["method", "class", "precision", "recall", "f1"]]
    merged = legacy.merge(reproduced, on=["method", "class"], how="outer", suffixes=("_legacy", "_reproduced"), indicator=True)
    merged["precision_delta"] = merged["precision_reproduced"] - merged["precision_legacy"]
    merged["recall_delta"] = merged["recall_reproduced"] - merged["recall_legacy"]
    merged["f1_delta"] = merged["f1_reproduced"] - merged["f1_legacy"]
    comparable = merged["_merge"].eq("both")
    rounded_equal = pd.DataFrame(
        {
            metric: merged[f"{metric}_legacy"].round(4).eq(merged[f"{metric}_reproduced"].round(4))
            | (merged[f"{metric}_legacy"] - merged[f"{metric}_reproduced"]).abs().le(tolerance)
            for metric in ("precision", "recall", "f1")
        }
    ).all(axis=1)
    merged["matches_within_rounding_tolerance"] = comparable & rounded_equal
    return merged.sort_values(["method", "class"]).reset_index(drop=True)


def plot_model_comparison(metrics: pd.DataFrame, figure_dir: Path) -> Path:
    macro = metrics.loc[metrics["aggregation"].eq("macro")].set_index("method").loc[list(METHOD_CHECKPOINTS)]
    metric_columns = ("f1", "average_precision", "false_negative_rate")
    labels = ("Macro F1", "Macro average precision", "Macro false-negative rate")
    x = np.arange(len(macro))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, column, label in zip(axes, metric_columns, labels, strict=True):
        axis.bar(x, macro[column], color="#2a6fbb")
        axis.set_title(label)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, macro.index, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    path = figure_dir / "model_comparison_macro_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pr_curves(curves: list[dict[str, np.ndarray | str]], figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for class_name in CLASS_NAMES:
        fig, axis = plt.subplots(figsize=(7, 5))
        for curve in curves:
            if curve["class"] == class_name:
                axis.plot(curve["recall"], curve["precision"], label=str(curve["method"]))
        axis.set(title=f"Precision-recall curve: {class_name}", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncols=2)
        path = figure_dir / f"precision_recall_{class_name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_cost_sensitivity(metrics: pd.DataFrame, ratios: tuple[float, ...], fp_cost: float) -> pd.DataFrame:
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("all FN/FP ratios must be positive")
    totals = metrics.loc[metrics["aggregation"].eq("total"), ["method", "fn", "fp"]]
    rows = []
    for _, total in totals.iterrows():
        for ratio in ratios:
            scenario_fn_cost = ratio * fp_cost
            rows.append(
                {
                    "method": total["method"],
                    "fn_fp_cost_ratio": ratio,
                    "fn_cost": scenario_fn_cost,
                    "fp_cost": fp_cost,
                    "fn": int(total["fn"]),
                    "fp": int(total["fp"]),
                    "cost": calculate_cost(int(total["fn"]), int(total["fp"]), scenario_fn_cost, fp_cost),
                }
            )
    return pd.DataFrame(rows).sort_values(["fn_fp_cost_ratio", "method"]).reset_index(drop=True)


def plot_cost_sensitivity(costs: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_CHECKPOINTS:
        subset = costs.loc[costs["method"].eq(method)]
        axis.plot(subset["fn_fp_cost_ratio"], subset["cost"], marker="o", label=method)
    axis.set(
        title="Cost sensitivity by assumed FN:FP cost ratio",
        xlabel="Assumed FN:FP cost ratio (FP cost held constant)",
        ylabel="Total cost across four classes",
    )
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    path = figure_dir / "cost_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run every required method and write all report-ready artifacts."""
    if not 0 <= config.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    labels = validate_inputs(config)
    device = select_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    curves: list[dict[str, np.ndarray | str]] = []
    for method, filename in METHOD_CHECKPOINTS.items():
        checkpoint = config.checkpoint_dir / filename
        truth, scores = infer_method(method, checkpoint, labels, config, device)
        rows, method_curves = make_metric_rows(method, truth, scores, config.threshold, config.fn_cost, config.fp_cost)
        metric_rows.extend(rows)
        prediction_rows.extend(make_prediction_rows(method, labels, truth, scores, config.threshold))
        curves.extend(method_curves)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics_path = config.output_dir / "metrics.csv"
    # Keep the auditable output in the repository without turning the PR into
    # a 70k-line CSV review.  Pandas can read this directly with
    # ``pd.read_csv('predictions.csv.gz')``.
    predictions_path = config.output_dir / "predictions.csv.gz"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")

    legacy = compare_to_legacy(metrics, config.legacy_results_csv, config.legacy_tolerance)
    legacy_path = config.output_dir / "legacy_comparison.csv"
    legacy.to_csv(legacy_path, index=False)
    costs = make_cost_sensitivity(metrics, config.fn_fp_ratios, config.fp_cost)
    costs_path = config.output_dir / "cost_sensitivity.csv"
    costs.to_csv(costs_path, index=False)
    figure_paths = [plot_model_comparison(metrics, figure_dir), *plot_pr_curves(curves, figure_dir), plot_cost_sensitivity(costs, figure_dir)]

    public_config = {
        key: manifest_path_label(value, config.project_root) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": (
            "uv run python evaluate_all.py "
            "--image-dir '<external train_images directory>' "
            f"--threshold {config.threshold} --fn-cost {config.fn_cost} --fp-cost {config.fp_cost} "
            f"--fn-fp-ratios {','.join(map(str, config.fn_fp_ratios))} "
            f"--output-dir {manifest_path_label(config.output_dir, config.project_root)}"
        ),
        "config": public_config,
        "device_used": str(device),
        "methods": list(METHOD_CHECKPOINTS),
        "class_order": list(CLASS_NAMES),
        "validation_sample_count": len(labels),
        "validation_csv_sha256": sha256_file(config.val_csv),
        "checkpoint_sha256": {method: sha256_file(config.checkpoint_dir / filename) for method, filename in METHOD_CHECKPOINTS.items()},
        "artifacts": [str(path.relative_to(config.output_dir)) for path in [metrics_path, predictions_path, legacy_path, costs_path, *figure_paths]],
        "legacy_matches": bool(legacy["matches_within_rounding_tolerance"].all()),
    }
    manifest_path = config.output_dir / "run_manifest.json"
    manifest["artifacts"].append(str(manifest_path.relative_to(config.output_dir)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
