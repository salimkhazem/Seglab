"""Standard segmentation metrics and epoch meter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


def compute_segmentation_metrics(
    probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> Dict[str, float]:
    """Compute Dice, IoU, Precision, Recall for a batch."""
    pred = (probs > threshold).long()
    target = target.long()
    if pred.ndim == 4:
        pred = pred[:, 0]
    if target.ndim == 4:
        target = target[:, 0]

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def boundary_f_score(
    pred: np.ndarray, target: np.ndarray, tol: int = 1, eps: float = 1e-6
) -> float:
    """Boundary F-score with pixel tolerance."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_b = pred ^ binary_erosion(pred)
    target_b = target ^ binary_erosion(target)

    pred_dil = binary_dilation(pred_b, iterations=tol)
    target_dil = binary_dilation(target_b, iterations=tol)

    precision = (pred_b & target_dil).sum() / (pred_b.sum() + eps)
    recall = (target_b & pred_dil).sum() / (target_b.sum() + eps)
    return float(2 * precision * recall / (precision + recall + eps))


class SegmentationMeter:
    """Accumulate TP/FP/FN for stable epoch metrics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        pred = (probs > self.threshold).long()
        target = target.long()
        if pred.ndim == 4:
            pred = pred[:, 0]
        if target.ndim == 4:
            target = target[:, 0]

        self.tp += float((pred * target).sum().item())
        self.fp += float((pred * (1 - target)).sum().item())
        self.fn += float(((1 - pred) * target).sum().item())

    def compute(self, eps: float = 1e-6) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        }

    def log(self, pl_module, prefix: str) -> None:
        m = self.compute()
        for k, v in m.items():
            pl_module.log(f"{prefix}/{k}", v, prog_bar=(prefix != "train"))

