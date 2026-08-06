"""Train cgan | stylegan | diffusion on train-only rare defect patches.

cVAE lives in the classification path (`app/models/`) — do not duplicate it here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rare_defect.config import load_config, resolve_device
from rare_defect.data import DefectPatchDataset
from rare_defect.models import (
    MaskDiffusion,
    PatchCritic,
    PatchGenerator,
    StyleGANDiscriminator,
    StyleGANGenerator,
)
from rare_defect.training import train_cgan, train_diffusion, train_stylegan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model", choices=["cgan", "stylegan", "diffusion"], required=True)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    class_id = args.class_id or cfg["data"]["rare_class_id"]
    raw = ROOT / cfg["data"]["raw_root"]
    splits = ROOT / cfg["data"]["splits_dir"]
    epochs = args.epochs or cfg["generator"]["epochs"]

    ds = DefectPatchDataset(
        raw,
        splits,
        class_id=class_id,
        patch_size=cfg["patch_size"],
        image_size=(cfg["image_height"], cfg["image_width"]),
        split="train",
    )
    if len(ds) == 0:
        raise SystemExit("No rare patches found. Run: python scripts/prepare_splits.py")

    loader = DataLoader(
        ds, batch_size=cfg["generator"]["batch_size"], shuffle=True, num_workers=0, drop_last=True
    )
    out = ROOT / cfg["runs_dir"] / f"generator_{args.model}_c{class_id}"
    print(f"model={args.model}  patches={len(ds)}  device={device}  out={out}")

    gcfg = cfg["generator"]
    if args.model == "cgan":
        train_cgan(
            PatchGenerator(latent_dim=gcfg["latent_dim"]),
            PatchCritic(),
            loader,
            epochs=epochs,
            lr=gcfg["lr"],
            device=device,
            out_dir=out,
        )
    elif args.model == "stylegan":
        train_stylegan(
            StyleGANGenerator(
                latent_dim=gcfg["latent_dim"],
                style_dim=gcfg.get("style_dim", 128),
                base_channels=gcfg.get("stylegan_base_channels", 256),
            ),
            StyleGANDiscriminator(),
            loader,
            epochs=epochs,
            lr=gcfg["lr"],
            device=device,
            out_dir=out,
        )
    else:
        train_diffusion(
            MaskDiffusion(
                timesteps=gcfg.get("diffusion_timesteps", 200),
                base_channels=gcfg.get("diffusion_base_channels", 64),
            ),
            loader,
            epochs=epochs,
            lr=gcfg["lr"],
            device=device,
            out_dir=out,
        )


if __name__ == "__main__":
    main()
