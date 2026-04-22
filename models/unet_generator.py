"""
U-Net generator for the Cancelable MinusFace pipeline.

Takes wavelet-encoded face representation (B, 21, 56, 56) and reconstructs
the appearance component. The residue r = x - x' retains identity-discriminative
signal while stripping appearance.

Architecture:
  Encoder: ch → 64 → 128 → 256 → bottleneck 512
  Decoder: 512 → 256 → 128 → 64 → ch
  Skip connections at each encoder level preserve spatial detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double conv block: Conv→BN→ReLU→Conv→BN→ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net for appearance reconstruction from wavelet features.

    The generator is trained to reconstruct x from x (self-supervised),
    so it captures appearance/texture. What it fails to reconstruct is
    the identity residue r = x - x'.
    """

    def __init__(self, ch: int = 21) -> None:
        super().__init__()
        self.e1 = ConvBlock(ch, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bn = ConvBlock(256, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = ConvBlock(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = ConvBlock(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Wavelet features of shape (B, 21, 56, 56).

        Returns:
            Reconstructed features of shape (B, 21, 56, 56).
        """
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bn(self.pool(e3))
        d3 = self.d3(torch.cat([F.interpolate(self.u3(b), e3.shape[2:]), e3], dim=1))
        d2 = self.d2(torch.cat([F.interpolate(self.u2(d3), e2.shape[2:]), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(self.u1(d2), e1.shape[2:]), e1], dim=1))
        return self.out(d1)
