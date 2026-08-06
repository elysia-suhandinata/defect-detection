"""Mask-conditioned DDPM for defect patches."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_time_embedding(t, self.dim))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MaskConditionedDenoiser(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256, mask_channels: int = 1):
        super().__init__()
        self.time_emb = TimeMLP(base_channels, time_dim)
        self.in_conv = nn.Conv2d(3 + mask_channels, base_channels, 3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.pool1 = nn.Conv2d(base_channels, base_channels, 4, 2, 1)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.upb3 = ResBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.upb2 = ResBlock(base_channels * 4, base_channels, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1)
        self.upb1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.in_conv(torch.cat([x, mask], dim=1))
        d1 = self.down1(h, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        d3 = self.down3(self.pool2(d2), t_emb)
        m = self.mid2(self.mid1(self.pool3(d3), t_emb), t_emb)
        u3 = self.upb3(torch.cat([self.up3(m), d3], dim=1), t_emb)
        u2 = self.upb2(torch.cat([self.up2(u3), d2], dim=1), t_emb)
        u1 = self.upb1(torch.cat([self.up1(u2), d1], dim=1), t_emb)
        return self.out(u1)


class DiffusionScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        self.device = device
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _extract(self, values, t, shape):
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, xt, t, mask):
        betas_t = self._extract(self.betas, t, xt.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        sqrt_recip = self._extract(self.sqrt_recip_alphas, t, xt.shape)
        eps = model(xt, t, mask)
        mean = sqrt_recip * (xt - betas_t * eps / sqrt_one_minus)
        if (t == 0).all():
            return mean
        var = self._extract(self.posterior_variance, t, xt.shape)
        return mean + torch.sqrt(var) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, mask, n=None):
        model.eval()
        n = n or mask.size(0)
        if mask.size(0) == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        _, _, h, w = mask.shape
        x = torch.randn(n, 3, h, w, device=mask.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=mask.device, dtype=torch.long)
            x = self.p_sample(model, x, t, mask)
        return (x.clamp(-1, 1) + 1) * 0.5


class MaskDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.denoiser = MaskConditionedDenoiser(base_channels=base_channels, time_dim=time_dim)
        self.timesteps = timesteps
        self._scheduler: DiffusionScheduler | None = None

    def scheduler(self, device: torch.device) -> DiffusionScheduler:
        if self._scheduler is None or self._scheduler.device != device:
            self._scheduler = DiffusionScheduler(timesteps=self.timesteps, device=device)
        return self._scheduler

    def forward(self, x0: torch.Tensor, mask: torch.Tensor):
        device = x0.device
        sched = self.scheduler(device)
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, t, noise)
        return self.denoiser(xt, t, mask), noise

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, n: int | None = None) -> torch.Tensor:
        return self.scheduler(mask.device).sample(self.denoiser, mask, n=n)
