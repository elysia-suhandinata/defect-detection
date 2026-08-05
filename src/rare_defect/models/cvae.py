"""Mask-conditioned convolutional VAE for defect patches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, mask_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + mask_channels * 8 * 8, 256 * 8 * 8)
        self.mask_down = nn.Sequential(
            nn.Conv2d(mask_channels, 16, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, mask_channels, 4, 2, 1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.mask_down(mask).flatten(1)
        h = self.fc_decode(torch.cat([z, m], dim=1)).view(-1, 256, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, mask), mu, logvar

    def sample(self, mask: torch.Tensor, n: int | None = None) -> torch.Tensor:
        n = n or mask.size(0)
        z = torch.randn(n, self.latent_dim, device=mask.device)
        if mask.size(0) == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        return self.decode(z, mask)


def cvae_loss(recon, x, mu, logvar, beta: float = 1.0) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld
