"""Run one experimental arm."""

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
from rare_defect.data import (
    CopyPasteAugment,
    PhotometricGeometricAugment,
    SeverstalSegDataset,
    build_rare_donor_bank,
)
from rare_defect.losses import BCEDiceLoss
from rare_defect.models import (
    MaskDiffusion,
    PatchGenerator,
    StyleGANGenerator,
    UNet,
)
from rare_defect.training import (
    MixedSegDataset,
    evaluate,
    fit_segmenter,
    generate_candidates,
    score_candidates_with_frozen_segmenter,
)
from rare_defect.training.selection import load_clean_and_mask_bank

# cVAE is Track A only (`app/models/`); Track B adds mask-aware GAN / diffusion arms.
GEN_ARMS = {
    "cgan_selected": "cgan",
    "stylegan_selected": "stylegan",
    "diffusion_selected": "diffusion",
}
ALL_ARMS = ["baseline", "weighted", "copy_paste", *GEN_ARMS]


def build_loaders(cfg, arm, synthetics=None):
    raw = ROOT / cfg["data"]["raw_root"]
    splits = ROOT / cfg["data"]["splits_dir"]
    size = (cfg["image_height"], cfg["image_width"])
    aug = cfg["augmentation"]
    transform = PhotometricGeometricAugment(
        hflip_prob=aug["hflip_prob"], brightness=aug["brightness"], contrast=aug["contrast"]
    )
    copy_paste = None
    if arm == "copy_paste":
        bank = build_rare_donor_bank(
            raw, splits, rare_class_id=cfg["data"]["rare_class_id"], image_size=size
        )
        copy_paste = CopyPasteAugment(
            bank, prob=aug["copy_paste_prob"], rare_class_id=cfg["data"]["rare_class_id"]
        )

    train_ds = SeverstalSegDataset(
        raw, splits, "train", image_size=size, transform=transform, copy_paste=copy_paste
    )
    if synthetics:
        train_ds = MixedSegDataset(train_ds, synthetics)
    val_ds = SeverstalSegDataset(raw, splits, "val", image_size=size)
    test_ds = SeverstalSegDataset(raw, splits, "test", image_size=size)
    bs, nw = cfg["segmenter"]["batch_size"], cfg["segmenter"]["num_workers"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw),
    )


def load_generator(kind, cfg, class_id, device):
    gcfg = cfg["generator"]
    ckpt_dir = ROOT / cfg["runs_dir"] / f"generator_{kind}_c{class_id}"
    if kind == "cgan":
        gen = PatchGenerator(latent_dim=gcfg["latent_dim"]).to(device)
        gen.load_state_dict(torch.load(ckpt_dir / "cgan.pt", map_location=device)["generator"])
        return gen
    if kind == "stylegan":
        gen = StyleGANGenerator(
            latent_dim=gcfg["latent_dim"],
            style_dim=gcfg.get("style_dim", 128),
            base_channels=gcfg.get("stylegan_base_channels", 256),
        ).to(device)
        gen.load_state_dict(torch.load(ckpt_dir / "stylegan.pt", map_location=device)["generator"])
        return gen
    if kind == "diffusion":
        gen = MaskDiffusion(
            timesteps=gcfg.get("diffusion_timesteps", 200),
            base_channels=gcfg.get("diffusion_base_channels", 64),
        ).to(device)
        gen.load_state_dict(torch.load(ckpt_dir / "diffusion.pt", map_location=device))
        return gen
    raise ValueError(kind)


def maybe_build_synthetics(cfg, arm, device, seed_model):
    if arm not in GEN_ARMS:
        return None
    kind = GEN_ARMS[arm]
    class_id = cfg["data"]["rare_class_id"]
    raw = ROOT / cfg["data"]["raw_root"]
    splits = ROOT / cfg["data"]["splits_dir"]
    cleans, masks = load_clean_and_mask_bank(
        raw,
        splits,
        class_id=class_id,
        patch_size=cfg["patch_size"],
        image_size=(cfg["image_height"], cfg["image_width"]),
    )
    if not cleans or not masks:
        raise SystemExit("Need clean sheets + rare masks from train split.")

    gen = load_generator(kind, cfg, class_id, device)
    candidates = generate_candidates(
        gen,
        masks,
        cleans,
        class_id=class_id,
        num_candidates=cfg["generator"]["num_candidates"],
        device=device,
        model_kind=kind,
    )
    ranked = score_candidates_with_frozen_segmenter(seed_model, candidates, device, class_id)
    selected = ranked[: cfg["generator"]["num_selected"]]
    print(f"[{kind}] selected {len(selected)} / {len(candidates)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    torch.manual_seed(cfg["seed"])

    seed_model = UNet(num_classes=cfg["num_classes"], base=cfg["segmenter"]["base_channels"]).to(device)
    synthetics = None
    if args.arm in GEN_ARMS:
        baseline_ckpt = ROOT / cfg["runs_dir"] / "arm_baseline" / "best.pt"
        if baseline_ckpt.exists():
            seed_model.load_state_dict(torch.load(baseline_ckpt, map_location=device)["model"])
        synthetics = maybe_build_synthetics(cfg, args.arm, device, seed_model)

    train_loader, val_loader, test_loader = build_loaders(cfg, args.arm, synthetics)
    weights = cfg["segmenter"]["class_weights"] if args.arm == "weighted" else None
    criterion = BCEDiceLoss(
        bce_weight=cfg["segmenter"]["bce_weight"],
        dice_weight=cfg["segmenter"]["dice_weight"],
        class_weights=weights,
    )
    model = UNet(num_classes=cfg["num_classes"], base=cfg["segmenter"]["base_channels"]).to(device)
    out = ROOT / cfg["runs_dir"] / f"arm_{args.arm}"
    epochs = args.epochs or cfg["segmenter"]["epochs"]

    result = fit_segmenter(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=cfg["segmenter"]["lr"],
        weight_decay=cfg["segmenter"]["weight_decay"],
        criterion=criterion,
        device=device,
        out_dir=out,
        rare_class_id=cfg["data"]["rare_class_id"],
        fn_cost=cfg["cost_model"]["fn_cost"],
        fp_cost=cfg["cost_model"]["fp_cost"],
    )

    best = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        rare_class_id=cfg["data"]["rare_class_id"],
        fn_cost=cfg["cost_model"]["fn_cost"],
        fp_cost=cfg["cost_model"]["fp_cost"],
    )
    payload = {
        "arm": args.arm,
        "val_best_rare_dice": result["best_rare_dice"],
        "test": test_metrics.__dict__,
    }
    (out / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("TEST", payload)


if __name__ == "__main__":
    main()
