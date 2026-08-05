"""Severstal RLE encode/decode (column-major / Fortran order)."""

from __future__ import annotations

import numpy as np


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE string to binary mask (H, W)."""
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(rle, str) or not rle.strip():
        return mask.reshape(h, w, order="F")

    values = list(map(int, rle.split()))
    starts, lengths = values[0::2], values[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start : start + length] = 1
    return mask.reshape(h, w, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary (H, W) mask to Severstal RLE."""
    pixels = mask.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def masks_to_multichannel(
    class_rles: dict[int, str],
    shape: tuple[int, int],
    num_classes: int = 4,
) -> np.ndarray:
    """Stack ClassId 1..C into (C, H, W) float32 masks."""
    channels = []
    for class_id in range(1, num_classes + 1):
        rle = class_rles.get(class_id, "")
        channels.append(rle_decode(rle, shape).astype(np.float32))
    return np.stack(channels, axis=0)
