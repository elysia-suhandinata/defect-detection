"""REINFORCE selection loop over generator candidates.

Generators (cGAN / StyleGAN / Diffusion) propose synthetics; this policy learns
which ones to keep. Reward = rare-class Dice on real val after a short UNet
finetune. Not PPO on the generator decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rare_defect.losses import BCEDiceLoss
from rare_defect.models import UNet
from rare_defect.training.segmenter import evaluate, train_one_epoch
from rare_defect.training.selection import MixedSegDataset, SyntheticSample

GENERATOR_KINDS = ("cgan", "stylegan", "diffusion")
FEAT_DIM = 8  # score, mask_area, 3x one-hot, mean_in, std_in, img_mean


class SelectionPolicy(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats)).squeeze(-1)


def _generator_one_hot(kind: str) -> list[float]:
    return [1.0 if kind == k else 0.0 for k in GENERATOR_KINDS]


def candidate_features(
    candidates: list[SyntheticSample],
    class_id: int,
    generator_kind: str,
) -> torch.Tensor:
    """Build per-candidate feature matrix for the selection policy."""
    c = class_id - 1
    rows = []
    for s in candidates:
        kind = s.generator_kind or generator_kind
        one_hot = _generator_one_hot(kind)
        mask = s.mask[c]
        area = float(mask.mean())
        binary = mask > 0.5
        img = s.image
        if binary.any():
            pix = img[binary]
            mean_in = float(pix.mean())
            std_in = float(pix.std())
        else:
            mean_in, std_in = 0.0, 0.0
        rows.append(
            [
                float(s.score),
                area,
                *one_hot,
                mean_in,
                std_in,
                float(img.mean()),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _enforce_keep_budget(
    probs: torch.Tensor,
    actions: torch.Tensor,
    min_keep: int,
    max_keep: int,
) -> torch.Tensor:
    """Adjust sampled Bernoulli keeps into [min_keep, max_keep]."""
    out = actions.clone()
    keep_idx = torch.where(out > 0.5)[0]
    drop_idx = torch.where(out <= 0.5)[0]
    n_keep = int(keep_idx.numel())

    if n_keep > max_keep:
        scores = probs[keep_idx]
        order = torch.argsort(scores, descending=True)
        keep = keep_idx[order[:max_keep]]
        out.zero_()
        out[keep] = 1.0
    elif n_keep < min_keep:
        need = min_keep - n_keep
        if drop_idx.numel() > 0:
            scores = probs[drop_idx]
            order = torch.argsort(scores, descending=True)
            add = drop_idx[order[:need]]
            out[add] = 1.0
    return out


def _bernoulli_log_prob(probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(1e-6, 1 - 1e-6)
    return (actions * torch.log(p) + (1 - actions) * torch.log(1 - p)).sum()


def _short_finetune_reward(
    *,
    seed_state: dict,
    real_train_ds,
    selected: list[SyntheticSample],
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> float:
    """Clone seed UNet, short-train on real subset + selected, return val rare Dice."""
    rl = cfg["rl"]
    scfg = cfg["segmenter"]
    model = UNet(num_classes=cfg["num_classes"], base=scfg["base_channels"]).to(device)
    model.load_state_dict(seed_state)

    n_real = min(int(rl.get("reward_real_samples", 256)), len(real_train_ds))
    idx = torch.randperm(len(real_train_ds))[:n_real].tolist()
    real_subset = Subset(real_train_ds, idx)
    train_ds = MixedSegDataset(real_subset, selected)
    loader = DataLoader(
        train_ds,
        batch_size=scfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    criterion = BCEDiceLoss(
        bce_weight=scfg["bce_weight"],
        dice_weight=scfg["dice_weight"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])
    for _ in range(int(rl["finetune_epochs"])):
        train_one_epoch(model, loader, criterion, opt, device)

    metrics = evaluate(
        model,
        val_loader,
        device,
        rare_class_id=cfg["data"]["rare_class_id"],
        fn_cost=cfg["cost_model"]["fn_cost"],
        fp_cost=cfg["cost_model"]["fp_cost"],
    )
    return float(metrics.rare_dice)


def run_reinforce_selection(
    candidates: list[SyntheticSample],
    *,
    seed_model: UNet,
    real_train_ds,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    generator_kind: str,
    out_dir: Path | None = None,
) -> list[SyntheticSample]:
    """Train a keep/reject policy with REINFORCE; return greedy-selected synthetics."""
    if not candidates:
        return []

    rl = cfg["rl"]
    class_id = cfg["data"]["rare_class_id"]
    min_keep = int(rl["min_keep"])
    max_keep = int(rl["max_keep"])
    min_keep = max(1, min(min_keep, len(candidates)))
    max_keep = max(min_keep, min(max_keep, len(candidates)))

    # Proxy utility scores for features (and as a warm start signal).
    from rare_defect.training.selection import score_candidates_with_frozen_segmenter

    scored = score_candidates_with_frozen_segmenter(
        seed_model, list(candidates), device, class_id
    )
    feats = candidate_features(scored, class_id, generator_kind).to(device)

    policy = SelectionPolicy(feat_dim=feats.shape[1]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(rl["lr"]))
    seed_state = {k: v.detach().cpu().clone() for k, v in seed_model.state_dict().items()}

    baseline = None
    momentum = float(rl["baseline_momentum"])
    history = []

    for ep in range(1, int(rl["episodes"]) + 1):
        policy.train()
        probs = policy(feats)
        actions = torch.bernoulli(probs)
        actions = _enforce_keep_budget(probs.detach(), actions, min_keep, max_keep)

        selected = [scored[i] for i, a in enumerate(actions.tolist()) if a > 0.5]
        reward = _short_finetune_reward(
            seed_state=seed_state,
            real_train_ds=real_train_ds,
            selected=selected,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
        )

        if baseline is None:
            baseline = reward
        advantage = reward - baseline
        baseline = momentum * baseline + (1.0 - momentum) * reward

        log_prob = _bernoulli_log_prob(probs, actions)
        loss = -advantage * log_prob
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {
            "episode": ep,
            "reward": reward,
            "baseline": baseline,
            "advantage": advantage,
            "n_keep": len(selected),
            "loss": float(loss.detach()),
        }
        history.append(row)
        print(
            f"[rl {generator_kind}] ep {ep:03d}  "
            f"reward={reward:.4f}  adv={advantage:+.4f}  keep={len(selected)}"
        )

    policy.eval()
    with torch.no_grad():
        probs = policy(feats)
    order = torch.argsort(probs, descending=True).tolist()
    final = [scored[i] for i in order[:max_keep]]

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": policy.state_dict(),
                "history": history,
                "generator_kind": generator_kind,
                "final_indices": order[:max_keep],
            },
            out_dir / "rl_policy.pt",
        )

    print(f"[rl {generator_kind}] greedy keep {len(final)} / {len(scored)}")
    return final
