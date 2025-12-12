"""Calibration metrics for probabilistic segmentation."""

from __future__ import annotations

from typing import Tuple

import torch


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """Pixel-wise ECE."""
    probs = probs.reshape(-1)
    targets = targets.reshape(-1).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.any():
            acc = targets[mask].mean()
            conf = probs[mask].mean()
            ece += mask.float().mean() * torch.abs(acc - conf)
    return float(ece.item())


def reliability_curve(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-bin confidence and accuracy for plotting."""
    probs = probs.reshape(-1)
    targets = targets.reshape(-1).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    confs = []
    accs = []
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.any():
            confs.append(probs[mask].mean())
            accs.append(targets[mask].mean())
        else:
            confs.append(torch.tensor(0.0, device=probs.device))
            accs.append(torch.tensor(0.0, device=probs.device))
    return torch.stack(confs), torch.stack(accs)

