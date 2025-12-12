"""Topology-aware losses: Dice, clDice, soft skeletonization."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary masks."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable skeletonization."""
    skel = torch.zeros_like(img)
    for _ in range(iters):
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
        img = soft_erode(img)
    return skel


def cldice_loss(pred: torch.Tensor, target: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Centerline Dice loss."""
    skel_pred = soft_skeletonize(pred, iters)
    skel_true = soft_skeletonize(target, iters)

    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_true * pred).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + eps)
    cldice = (2 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1 - cldice.mean()


def boundary_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable boundary Dice using Laplacian edges."""
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred.device, dtype=pred.dtype)
    lap = lap.view(1, 1, 3, 3)
    pred_e = F.conv2d(pred, lap, padding=1).abs()
    tgt_e = F.conv2d(target, lap, padding=1).abs()
    return dice_loss(pred_e, tgt_e, eps=eps)

