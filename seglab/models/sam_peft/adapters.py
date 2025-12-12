"""Convolutional adapters for SAM high-res features."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class ConvAdapter(nn.Module):
    """Depthwise-separable conv adapter with residual."""

    def __init__(self, channels: int = 256, hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, hidden_dim, 1, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw1(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.pw2(y)
        return x + y

