"""StyleGAN-lite: mapping z→w + AdaIN synthesis, mask-conditioned patches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim: int = 64, style_dim: int = 128, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, style_dim), nn.LeakyReLU(0.2, inplace=True)]
            in_dim = style_dim
        self.net = nn.Sequential(*layers)
        self.style_dim = style_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(z, dim=1))


class AdaIN(nn.Module):
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class StyleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adain = AdaIN(out_ch, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.act(self.adain(self.conv(x), w))


class StyleGANGenerator(nn.Module):
    """8 → 16 → 32 → 64 → 128 synthesis for patch_size=128."""

    def __init__(
        self,
        latent_dim: int = 64,
        style_dim: int = 128,
        mask_channels: int = 1,
        base_channels: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = MappingNetwork(latent_dim, style_dim)
        self.const = nn.Parameter(torch.randn(1, base_channels, 8, 8))
        self.mask_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(mask_channels, base_channels, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.b1 = StyleBlock(base_channels, base_channels // 2, style_dim)
        self.b2 = StyleBlock(base_channels // 2, base_channels // 4, style_dim)
        self.b3 = StyleBlock(base_channels // 4, base_channels // 8, style_dim)
        self.b4 = StyleBlock(base_channels // 8, base_channels // 16, style_dim)
        self.to_rgb = nn.Sequential(nn.Conv2d(base_channels // 16, 3, 1), nn.Tanh())

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1) + self.mask_proj(mask)
        x = self.b1(x, w)
        x = self.b2(x, w)
        x = self.b3(x, w)
        x = self.b4(x, w)
        return self.to_rgb(x)

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, n: int | None = None) -> torch.Tensor:
        n = n or mask.size(0)
        if mask.size(0) == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        z = torch.randn(n, self.latent_dim, device=mask.device)
        return (self.forward(z, mask) + 1.0) * 0.5


class StyleGANDiscriminator(nn.Module):
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
