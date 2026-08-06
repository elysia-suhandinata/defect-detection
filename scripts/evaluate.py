"""Evaluate a checkpoint on the frozen real test/val split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rare_defect.config import load_config, resolve_device
from rare_defect.data import SeverstalSegDataset
from rare_defect.models import UNet
from rare_defect.report import build_report_from_masks
from rare_defect.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--demo-report", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    ds = SeverstalSegDataset(
        ROOT / cfg["data"]["raw_root"],
        ROOT / cfg["data"]["splits_dir"],
        args.split,
        image_size=(cfg["image_height"], cfg["image_width"]),
    )
    loader = DataLoader(ds, batch_size=cfg["segmenter"]["batch_size"], shuffle=False)

    model = UNet(num_classes=cfg["num_classes"], base=cfg["segmenter"]["base_channels"]).to(device)
    blob = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(blob["model"] if "model" in blob else blob)

    metrics = evaluate(
        model,
        loader,
        device,
        rare_class_id=cfg["data"]["rare_class_id"],
        fn_cost=cfg["cost_model"]["fn_cost"],
        fp_cost=cfg["cost_model"]["fp_cost"],
    )
    print(json.dumps(metrics.__dict__, indent=2))

    if args.demo_report and len(ds) > 0:
        sample = ds[0]
        with torch.no_grad():
            probs = torch.sigmoid(model(sample["image"].unsqueeze(0).to(device))[0])
        areas = {i + 1: float(probs[i].mean()) for i in range(probs.size(0))}
        conf = {i + 1: float((probs[i] > 0.5).float().mean()) for i in range(probs.size(0))}
        print(
            build_report_from_masks(
                sample["image_id"],
                conf,
                areas,
                fn_cost=cfg["cost_model"]["fn_cost"],
                fp_cost=cfg["cost_model"]["fp_cost"],
            )
        )


if __name__ == "__main__":
    main()
