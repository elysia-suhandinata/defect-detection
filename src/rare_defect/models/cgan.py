"""Mask-conditioned WGAN-GP for defect patches."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, mask_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.mask_enc = nn.Sequential(
            nn.Conv2d(mask_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(latent_dim + 64 * 8 * 8, 256 * 8 * 8)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.mask_enc(mask).flatten(1)
        h = self.fc(torch.cat([z, m], dim=1)).view(-1, 256, 8, 8)
        return self.decode(h)

    def sample(self, mask: torch.Tensor, n: int | None = None) -> torch.Tensor:
        n = n or mask.size(0)
        z = torch.randn(n, self.latent_dim, device=mask.device)
        if mask.size(0) == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        return (self.forward(z, mask) + 1.0) * 0.5


class PatchCritic(nn.Module):
    def __init__(self, mask_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 + mask_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(256 * 8 * 8, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(torch.cat([x, mask], dim=1)).flatten(1))


def gradient_penalty(critic, real, fake, mask) -> torch.Tensor:
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(interp, mask)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0].reshape(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
