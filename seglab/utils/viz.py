"""Visualization utilities for figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from seglab.metrics.calibration import reliability_curve


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
IMAGENET_STD = np.array([0.229, 0.224, 0.225])[None, None, :]


def _unnormalize(img: np.ndarray) -> np.ndarray:
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def save_qualitative_grid(
    images: np.ndarray,
    masks: np.ndarray,
    preds: np.ndarray,
    out_path: str | Path,
    max_items: int = 8,
) -> None:
    """Save a qualitative grid (input, GT, pred, error)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_items, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = images[i]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = _unnormalize(img)
        gt = masks[i]
        pr = preds[i]
        err = np.abs(gt - pr)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT")
        axes[i, 2].imshow(pr, cmap="gray")
        axes[i, 2].set_title("Pred")
        axes[i, 3].imshow(err, cmap="magma")
        axes[i, 3].set_title("Error")
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    out_path: str | Path,
) -> None:
    """Plot PR curve for binary segmentation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probs_f = probs.reshape(-1)
    targets_f = targets.reshape(-1).astype(np.uint8)
    precision, recall, _ = precision_recall_curve(targets_f, probs_f)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    out_path: str | Path,
    n_bins: int = 15,
) -> None:
    """Plot calibration reliability diagram."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    probs_t = torch.from_numpy(probs).float()
    targets_t = torch.from_numpy(targets).float()
    confs, accs = reliability_curve(probs_t, targets_t, n_bins=n_bins)
    confs = confs.cpu().numpy()
    accs = accs.cpu().numpy()

    fig = plt.figure()
    plt.plot(confs, accs, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ablation_bars(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    order: Optional[Iterable[str]] = None,
) -> None:
    """Plot ablation bars given a dataframe with columns ['variant', metric]."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if order:
        df = df.set_index("variant").loc[list(order)].reset_index()

    fig = plt.figure(figsize=(8, 4))
    plt.bar(df["variant"], df[metric])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"Ablation: {metric}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

