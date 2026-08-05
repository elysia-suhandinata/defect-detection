"""Severstal segmentation dataset, patch bank, and classical / Copy-Paste aug."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rle import masks_to_multichannel, rle_decode
from .splits import load_split_ids


def load_annotation_map(train_csv: Path) -> dict[str, dict[int, str]]:
    df = pd.read_csv(train_csv)
    mapping: dict[str, dict[int, str]] = defaultdict(dict)
    for _, row in df.iterrows():
        rle = row.get("EncodedPixels")
        if pd.isna(rle) or str(rle).strip() == "":
            continue
        mapping[str(row["ImageId"])][int(row["ClassId"])] = str(rle)
    return dict(mapping)


class SeverstalSegDataset(Dataset):
    def __init__(
        self,
        raw_root: Path | str,
        splits_dir: Path | str,
        split: str,
        image_size: tuple[int, int] = (256, 1600),
        num_classes: int = 4,
        transform=None,
        copy_paste: "CopyPasteAugment | None" = None,
    ):
        self.raw_root = Path(raw_root)
        self.image_dir = self.raw_root / "train_images"
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.copy_paste = copy_paste
        self.ids = load_split_ids(Path(splits_dir), split)
        self.ann = load_annotation_map(self.raw_root / "train.csv")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.ids[idx]
        path = self.image_dir / image_id
        image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        h, w = self.image_size
        if image.shape[0] != h or image.shape[1] != w:
            image = (
                np.array(
                    Image.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )

        masks = masks_to_multichannel(self.ann.get(image_id, {}), (h, w), self.num_classes)

        if self.copy_paste is not None:
            image, masks = self.copy_paste(image, masks)
        if self.transform is not None:
            image, masks = self.transform(image, masks)

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(masks).float(),
            "image_id": image_id,
        }


class PhotometricGeometricAugment:
    def __init__(self, hflip_prob: float = 0.5, brightness: float = 0.15, contrast: float = 0.15):
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, :, ::-1])
        if self.brightness > 0:
            image = np.clip(image + np.random.uniform(-self.brightness, self.brightness), 0, 1)
        if self.contrast > 0:
            mean = image.mean()
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            image = np.clip((image - mean) * factor + mean, 0, 1)
        return image.astype(np.float32), masks.astype(np.float32)


class CopyPasteAugment:
    """Paste real rare-defect patches onto the current sheet (GT masks)."""

    def __init__(
        self,
        donor_bank: list[tuple[np.ndarray, np.ndarray]],
        prob: float = 0.5,
        rare_class_id: int = 2,
    ):
        self.donor_bank = donor_bank
        self.prob = prob
        self.rare_class_id = rare_class_id

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.donor_bank or np.random.rand() > self.prob:
            return image, masks

        src_img, src_mask = self.donor_bank[np.random.randint(len(self.donor_bank))]
        c = self.rare_class_id - 1
        defect = src_mask[c] > 0.5
        if defect.sum() == 0:
            return image, masks

        ys, xs = np.where(defect)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        patch_img = src_img[y0:y1, x0:x1]
        patch_m = src_mask[:, y0:y1, x0:x1]
        ph, pw = patch_img.shape[:2]
        H, W = image.shape[:2]
        if ph >= H or pw >= W:
            return image, masks

        ty = np.random.randint(0, H - ph)
        tx = np.random.randint(0, W - pw)
        binary = patch_m[c] > 0.5
        region = image[ty : ty + ph, tx : tx + pw]
        region[binary] = patch_img[binary]
        image[ty : ty + ph, tx : tx + pw] = region
        for k in range(masks.shape[0]):
            m_region = masks[k, ty : ty + ph, tx : tx + pw]
            m_region[binary] = np.maximum(m_region[binary], patch_m[k][binary])
            masks[k, ty : ty + ph, tx : tx + pw] = m_region
        return image.astype(np.float32), masks.astype(np.float32)


def build_rare_donor_bank(
    raw_root: Path | str,
    splits_dir: Path | str,
    rare_class_id: int = 2,
    image_size: tuple[int, int] = (256, 1600),
    max_donors: int = 256,
) -> list[tuple[np.ndarray, np.ndarray]]:
    raw_root = Path(raw_root)
    ids = load_split_ids(Path(splits_dir), "train")
    ann = load_annotation_map(raw_root / "train.csv")
    h, w = image_size
    bank: list[tuple[np.ndarray, np.ndarray]] = []
    for image_id in ids:
        if rare_class_id not in ann.get(image_id, {}):
            continue
        image = (
            np.array(Image.open(raw_root / "train_images" / image_id).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        masks = masks_to_multichannel(ann[image_id], (h, w))
        bank.append((image, masks))
        if len(bank) >= max_donors:
            break
    return bank


class DefectPatchDataset(Dataset):
    """Cropped defect patches for mask-conditioned generators."""

    def __init__(
        self,
        raw_root: Path | str,
        splits_dir: Path | str,
        class_id: int,
        patch_size: int = 128,
        image_size: tuple[int, int] = (256, 1600),
        split: str = "train",
    ):
        self.patch_size = patch_size
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        raw_root = Path(raw_root)
        ids = load_split_ids(Path(splits_dir), split)
        ann = load_annotation_map(raw_root / "train.csv")
        h, w = image_size
        half = patch_size // 2

        for image_id in ids:
            if class_id not in ann.get(image_id, {}):
                continue
            image = (
                np.array(Image.open(raw_root / "train_images" / image_id).convert("RGB"), dtype=np.float32)
                / 255.0
            )
            full_mask = rle_decode(ann[image_id][class_id], (h, w)).astype(np.float32)
            ys, xs = np.where(full_mask > 0.5)
            if len(ys) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = int(np.clip(cy - half, 0, h - patch_size))
            x0 = int(np.clip(cx - half, 0, w - patch_size))
            img_p = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            m_p = full_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if img_p.shape[0] == patch_size and img_p.shape[1] == patch_size:
                self.samples.append((img_p, m_p))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img, mask = self.samples[idx]
        return {
            "image": torch.from_numpy(img.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask[None]).float(),
        }
