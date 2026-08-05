"""Smoke checks that do not require the full training loop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rare_defect.data.rle import rle_decode, rle_encode
from rare_defect.losses import BCEDiceLoss
from rare_defect.metrics import summarize
from rare_defect.models import MaskDiffusion, PatchGenerator, StyleGANGenerator, UNet
from rare_defect.report import build_report_from_masks


def main() -> None:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    mask[10:20, 100:150] = 1
    assert np.array_equal(mask, rle_decode(rle_encode(mask), (256, 1600)))

    x = torch.rand(2, 3, 256, 1600)
    assert UNet(num_classes=4, base=8)(x).shape == (2, 4, 256, 1600)

    patch = torch.rand(2, 3, 128, 128)
    m = torch.rand(2, 1, 128, 128)
    assert PatchGenerator(latent_dim=16)(torch.randn(2, 16), m).shape == patch.shape

    sg = StyleGANGenerator(latent_dim=16, style_dim=32, base_channels=64)
    assert sg.sample(m[:1], n=2).shape == (2, 3, 128, 128)

    diff = MaskDiffusion(timesteps=4, base_channels=32, time_dim=64)
    pred, noise = diff(patch * 2 - 1, m)
    assert pred.shape == noise.shape
    assert diff.sample(m[:1], n=1).shape == (1, 3, 128, 128)

    metrics = summarize(torch.randn(4, 4, 32, 32), (torch.rand(4, 4, 32, 32) > 0.7).float())
    assert 0.0 <= metrics.mean_dice <= 1.0
    text = build_report_from_masks("demo.jpg", {2: 0.9}, {2: 0.01})
    assert any(k in text for k in ("HOLD", "FLAG", "PASS", "SOFT"))

    loss = BCEDiceLoss()(torch.randn(1, 4, 32, 32), torch.zeros(1, 4, 32, 32))
    print("smoke ok", float(loss))


if __name__ == "__main__":
    main()
