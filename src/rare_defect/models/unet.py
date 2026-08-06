"""Compact U-Net for multi-class Severstal segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.down1 = ConvBlock(in_channels, base)
        self.down2 = ConvBlock(base, base * 2)
        self.down3 = ConvBlock(base * 2, base * 4)
        self.down4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))

        u4 = self._match(self.up4(bn), d4)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self._match(self.up3(u4), d3)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self._match(self.up2(u3), d2)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self._match(self.up1(u2), d1)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.head(u1)

    @staticmethod
    def _match(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] == skip.shape[-2:]:
            return up
        _, _, h, w = skip.shape
        return up[:, :, :h, :w]
