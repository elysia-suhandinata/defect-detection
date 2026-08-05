"""Validation-utility sample selection (Module 9 slim feedback).

Policy = top-k / mix-ratio; reward = val rare Dice. Not PPO on the decoder.
Synthetics never enter the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rare_defect.data import SeverstalSegDataset, load_annotation_map, load_split_ids
from rare_defect.data.rle import rle_decode
from rare_defect.metrics import SegMetrics
from rare_defect.models import UNet


@dataclass
class SyntheticSample:
    image: np.ndarray
    mask: np.ndarray
    score: float = 0.0


class MixedSegDataset(Dataset):
    def __init__(self, real: SeverstalSegDataset, synthetics: list[SyntheticSample]):
        self.real = real
        self.synthetics = synthetics

    def __len__(self) -> int:
        return len(self.real) + len(self.synthetics)

    def __getitem__(self, idx: int) -> dict:
        if idx < len(self.real):
            return self.real[idx]
        s = self.synthetics[idx - len(self.real)]
        return {
            "image": torch.from_numpy(s.image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(s.mask).float(),
            "image_id": f"synth_{idx}",
        }


def _paste_patch_on_clean(clean_img, patch_rgb, patch_mask, class_id, num_classes=4):
    H, W, _ = clean_img.shape
    ph, pw = patch_mask.shape
    ty = np.random.randint(0, max(H - ph, 1))
    tx = np.random.randint(0, max(W - pw, 1))
    out = clean_img.copy()
    binary = patch_mask > 0.5
    region = out[ty : ty + ph, tx : tx + pw]
    region[binary] = patch_rgb[binary]
    out[ty : ty + ph, tx : tx + pw] = region
    masks = np.zeros((num_classes, H, W), dtype=np.float32)
    masks[class_id - 1, ty : ty + ph, tx : tx + pw][binary] = 1.0
    return out, masks


@torch.no_grad()
def generate_candidates(
    generator,
    mask_templates: list[np.ndarray],
    clean_images: list[np.ndarray],
    *,
    class_id: int,
    num_candidates: int,
    device: torch.device,
    model_kind: str = "",
) -> list[SyntheticSample]:
    generator.eval()
    samples: list[SyntheticSample] = []
    for i in range(num_candidates):
        m_np = mask_templates[i % len(mask_templates)]
        clean = clean_images[i % len(clean_images)]
        m = torch.from_numpy(m_np[None, None]).float().to(device)
        patch = np.clip(generator.sample(m, n=1)[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
        img, mask = _paste_patch_on_clean(clean, patch, m_np, class_id)
        samples.append(SyntheticSample(image=img, mask=mask))
    return samples


def score_candidates_with_frozen_segmenter(
    model: UNet,
    candidates: list[SyntheticSample],
    device: torch.device,
    class_id: int,
) -> list[SyntheticSample]:
    model.eval()
    c = class_id - 1
    for s in candidates:
        x = torch.from_numpy(s.image.transpose(2, 0, 1)[None]).float().to(device)
        probs = torch.sigmoid(model(x)[0, c].cpu())
        gt = torch.from_numpy(s.mask[c])
        inter = (probs * gt).sum()
        denom = probs.sum() + gt.sum() + 1e-6
        s.score = float((2 * inter / denom).item())
    candidates.sort(key=lambda s: s.score, reverse=True)
    return candidates


def grid_search_mix_ratio(build_and_eval_fn, ratios: list[float]) -> tuple[float, SegMetrics]:
    best_ratio, best_metrics = 0.0, None
    for r in ratios:
        metrics = build_and_eval_fn(r)
        print(f"mix_ratio={r:.2f}  rareDice={metrics.rare_dice:.4f}  meanDice={metrics.mean_dice:.4f}")
        if best_metrics is None or metrics.rare_dice > best_metrics.rare_dice:
            best_ratio, best_metrics = r, metrics
    assert best_metrics is not None
    return best_ratio, best_metrics


def load_clean_and_mask_bank(
    raw_root: Path,
    splits_dir: Path,
    class_id: int,
    patch_size: int,
    image_size: tuple[int, int],
    max_items: int = 200,
):
    ann = load_annotation_map(raw_root / "train.csv")
    ids = load_split_ids(splits_dir, "train")
    h, w = image_size
    cleans, masks = [], []
    half = patch_size // 2
    for image_id in ids:
        classes = ann.get(image_id, {})
        img = np.array(Image.open(raw_root / "train_images" / image_id).convert("RGB"), dtype=np.float32) / 255.0
        if not classes and len(cleans) < max_items:
            cleans.append(img)
        if class_id in classes and len(masks) < max_items:
            full = rle_decode(classes[class_id], (h, w)).astype(np.float32)
            ys, xs = np.where(full > 0.5)
            if len(ys) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = int(np.clip(cy - half, 0, h - patch_size))
            x0 = int(np.clip(cx - half, 0, w - patch_size))
            masks.append(full[y0 : y0 + patch_size, x0 : x0 + patch_size])
    return cleans, masks
