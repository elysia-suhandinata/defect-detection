"""Train mask-conditioned patch generators."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rare_defect.models import (
    MaskCVAE,
    MaskDiffusion,
    PatchCritic,
    PatchGenerator,
    StyleGANDiscriminator,
    StyleGANGenerator,
    cvae_loss,
    gradient_penalty,
)


def train_cvae(model: MaskCVAE, loader, *, epochs, lr, device, out_dir, beta=1.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"cvae {epoch}", leave=False):
            x, m = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x, m)
            loss = cvae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"cvae epoch {epoch:03d}  loss={total / max(len(loader.dataset), 1):.4f}")
        torch.save(model.state_dict(), out_dir / "cvae.pt")


def _wgan_loop(generator, critic, loader, *, epochs, lr, device, out_dir, ckpt_name, n_critic=5, gp_weight=10.0, betas=(0.0, 0.9)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator.to(device)
    critic.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    for epoch in range(1, epochs + 1):
        for batch in tqdm(loader, desc=f"{ckpt_name} {epoch}", leave=False):
            real = batch["image"].to(device) * 2 - 1
            mask = batch["mask"].to(device)
            b = real.size(0)
            for _ in range(n_critic):
                z = torch.randn(b, generator.latent_dim, device=device)
                fake = generator(z, mask).detach()
                opt_c.zero_grad(set_to_none=True)
                loss_c = critic(fake, mask).mean() - critic(real, mask).mean()
                loss_c = loss_c + gp_weight * gradient_penalty(critic, real, fake, mask)
                loss_c.backward()
                opt_c.step()
            z = torch.randn(b, generator.latent_dim, device=device)
            opt_g.zero_grad(set_to_none=True)
            loss_g = -critic(generator(z, mask), mask).mean()
            loss_g.backward()
            opt_g.step()
        print(f"{ckpt_name} epoch {epoch:03d}  G={loss_g.item():.4f}  C={loss_c.item():.4f}")
        torch.save({"generator": generator.state_dict(), "critic": critic.state_dict()}, out_dir / f"{ckpt_name}.pt")


def train_cgan(generator: PatchGenerator, critic: PatchCritic, loader, *, epochs, lr, device, out_dir):
    _wgan_loop(generator, critic, loader, epochs=epochs, lr=lr, device=device, out_dir=out_dir, ckpt_name="cgan")


def train_stylegan(generator: StyleGANGenerator, critic: StyleGANDiscriminator, loader, *, epochs, lr, device, out_dir):
    _wgan_loop(
        generator,
        critic,
        loader,
        epochs=epochs,
        lr=lr,
        device=device,
        out_dir=out_dir,
        ckpt_name="stylegan",
        betas=(0.0, 0.99),
    )


def train_diffusion(model: MaskDiffusion, loader, *, epochs, lr, device, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"diffusion {epoch}", leave=False):
            x0 = batch["image"].to(device) * 2 - 1
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            pred, noise = model(x0, mask)
            loss = F.mse_loss(pred, noise)
            loss.backward()
            opt.step()
            total += loss.item() * x0.size(0)
        print(f"diffusion epoch {epoch:03d}  loss={total / max(len(loader.dataset), 1):.4f}")
        torch.save(model.state_dict(), out_dir / "diffusion.pt")
