"""Dice / FNR / cost-model metrics. Always evaluate on real held-out images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegMetrics:
    mean_dice: float
    per_class_dice: list[float]
    rare_dice: float
    fnr: float
    fpr: float
    expected_cost: float


def dice_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    if preds.ndim == 3:
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)
    binary = (preds > threshold).float()
    dims = (0, 2, 3)
    inter = (binary * targets).sum(dim=dims)
    denom = binary.sum(dim=dims) + targets.sum(dim=dims)
    return (2 * inter + eps) / (denom + eps)


def image_level_rates(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float]:
    pred_pos = (preds.reshape(preds.shape[0], -1) > threshold).any(dim=1)
    true_pos = (targets.reshape(targets.shape[0], -1) > 0.5).any(dim=1)
    tp = (pred_pos & true_pos).sum().item()
    fn = (~pred_pos & true_pos).sum().item()
    fp = (pred_pos & ~true_pos).sum().item()
    tn = (~pred_pos & ~true_pos).sum().item()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return float(fnr), float(fpr)


def expected_cost(fnr: float, fpr: float, fn_cost: float, fp_cost: float) -> float:
    return fn_cost * fnr + fp_cost * fpr


def summarize(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rare_class_id: int = 2,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    threshold: float = 0.5,
) -> SegMetrics:
    probs = torch.sigmoid(logits.detach().cpu())
    targets = targets.detach().cpu()
    dices = dice_per_class(probs, targets, threshold=threshold)
    per_class = dices.tolist()
    fnr, fpr = image_level_rates(probs, targets, threshold=threshold)
    rare_idx = rare_class_id - 1
    return SegMetrics(
        mean_dice=float(np.mean(per_class)),
        per_class_dice=per_class,
        rare_dice=float(per_class[rare_idx]),
        fnr=fnr,
        fpr=fpr,
        expected_cost=expected_cost(fnr, fpr, fn_cost, fp_cost),
    )


def summarize_streaming(
    logits_iter,
    targets_iter,
    *,
    num_classes: int,
    rare_class_id: int = 2,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> SegMetrics:
    """Accumulate Dice / FNR without materializing the full val set on CPU."""
    inter = torch.zeros(num_classes)
    pred_sum = torch.zeros(num_classes)
    tgt_sum = torch.zeros(num_classes)
    tp = fn = fp = tn = 0

    for logits, targets in zip(logits_iter, targets_iter):
        probs = torch.sigmoid(logits.detach()).float().cpu()
        targets = targets.detach().float().cpu()
        if probs.ndim == 3:
            probs = probs.unsqueeze(0)
            targets = targets.unsqueeze(0)
        binary = (probs > threshold).float()
        dims = (0, 2, 3)
        inter += (binary * targets).sum(dim=dims)
        pred_sum += binary.sum(dim=dims)
        tgt_sum += targets.sum(dim=dims)

        pred_pos = (binary.reshape(binary.shape[0], -1) > 0).any(dim=1)
        true_pos = (targets.reshape(targets.shape[0], -1) > 0.5).any(dim=1)
        tp += int((pred_pos & true_pos).sum().item())
        fn += int((~pred_pos & true_pos).sum().item())
        fp += int((pred_pos & ~true_pos).sum().item())
        tn += int((~pred_pos & ~true_pos).sum().item())

    dices = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    per_class = dices.tolist()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    rare_idx = rare_class_id - 1
    return SegMetrics(
        mean_dice=float(np.mean(per_class)),
        per_class_dice=per_class,
        rare_dice=float(per_class[rare_idx]),
        fnr=float(fnr),
        fpr=float(fpr),
        expected_cost=expected_cost(float(fnr), float(fpr), fn_cost, fp_cost),
    )
