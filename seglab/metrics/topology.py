"""Topology metrics such as clDice."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from skimage.morphology import skeletonize


def cldice_score(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, eps: float = 1e-6) -> float:
    """Compute clDice score using hard skeletonization."""
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    if isinstance(target, torch.Tensor):
        tgt_np = target.detach().cpu().numpy()
    else:
        tgt_np = target

    pred_np = (pred_np > 0.5).astype(np.uint8)
    tgt_np = (tgt_np > 0.5).astype(np.uint8)

    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
    if tgt_np.ndim == 4:
        tgt_np = tgt_np[:, 0]

    scores = []
    for p, t in zip(pred_np, tgt_np):
        skel_p = skeletonize(p > 0)
        skel_t = skeletonize(t > 0)
        tprec = (skel_p & (t > 0)).sum() / (skel_p.sum() + eps)
        tsens = (skel_t & (p > 0)).sum() / (skel_t.sum() + eps)
        cld = 2 * tprec * tsens / (tprec + tsens + eps)
        scores.append(cld)
    return float(np.mean(scores))

