"""Segmenter train / eval loops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rare_defect.losses import BCEDiceLoss
from rare_defect.metrics import SegMetrics, expected_cost
from rare_defect.models import UNet


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    rare_class_id: int = 2,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> SegMetrics:
    """Stream Dice / FNR over batches — never materialize the full val set."""
    model.eval()
    inter = pred_sum = tgt_sum = None
    tp = fn = fp = tn = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        logits = model(batch["image"].to(device))
        probs = torch.sigmoid(logits).float()
        targets = batch["mask"].to(device=device, dtype=torch.float32)
        binary = (probs > threshold).float()

        if inter is None:
            c = binary.shape[1]
            inter = torch.zeros(c, device=device)
            pred_sum = torch.zeros(c, device=device)
            tgt_sum = torch.zeros(c, device=device)

        dims = (0, 2, 3)
        inter += (binary * targets).sum(dim=dims)
        pred_sum += binary.sum(dim=dims)
        tgt_sum += targets.sum(dim=dims)

        pred_pos = (binary.flatten(1) > 0).any(dim=1)
        true_pos = (targets.flatten(1) > 0.5).any(dim=1)
        tp += int((pred_pos & true_pos).sum().item())
        fn += int((~pred_pos & true_pos).sum().item())
        fp += int((pred_pos & ~true_pos).sum().item())
        tn += int((~pred_pos & ~true_pos).sum().item())

    if inter is None:
        raise RuntimeError("empty eval loader")

    dices = ((2 * inter + eps) / (pred_sum + tgt_sum + eps)).detach().cpu()
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


def fit_segmenter(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    lr,
    weight_decay,
    criterion: BCEDiceLoss,
    device,
    out_dir: Path,
    rare_class_id: int = 2,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_rare = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(
            model, val_loader, device, rare_class_id=rare_class_id, fn_cost=fn_cost, fp_cost=fp_cost
        )
        row = {"epoch": epoch, "train_loss": tr_loss, **metrics.__dict__}
        history.append(row)
        print(
            f"epoch {epoch:03d}  loss={tr_loss:.4f}  "
            f"meanDice={metrics.mean_dice:.4f}  rareDice={metrics.rare_dice:.4f}  "
            f"FNR={metrics.fnr:.3f}  cost={metrics.expected_cost:.3f}"
        )
        if metrics.rare_dice >= best_rare:
            best_rare = metrics.rare_dice
            torch.save(
                {"model": model.state_dict(), "metrics": metrics.__dict__, "epoch": epoch},
                out_dir / "best.pt",
            )

    torch.save({"model": model.state_dict(), "history": history}, out_dir / "last.pt")
    return {"best_rare_dice": best_rare, "history": history}
