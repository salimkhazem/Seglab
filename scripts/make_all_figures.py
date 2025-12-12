"""Generate figures for all completed runs.

This script:
  - plots training/validation curves from CSVLogger metrics
  - (optionally) copies per-run test figures already produced during testing

Usage:
  PYTHONNOUSERSITE=1 PYTHONPATH=. .venv/bin/python scripts/make_all_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _find_metrics_csv(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("logs/version_*/metrics.csv"))
    return candidates[-1] if candidates else None


def _series_by_epoch(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns or "epoch" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[["epoch", col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("epoch")[col].last()


def plot_training_curves(metrics_csv: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    train_loss = _series_by_epoch(df, "train/loss")
    val_loss = _series_by_epoch(df, "val/loss")
    train_dice = _series_by_epoch(df, "train/dice")
    val_dice = _series_by_epoch(df, "val/dice")
    train_iou = _series_by_epoch(df, "train/iou")
    val_iou = _series_by_epoch(df, "val/iou")

    lr_cols = [c for c in df.columns if c.startswith("lr-")]
    lr = _series_by_epoch(df, lr_cols[0]) if lr_cols else pd.Series(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    if not train_loss.empty:
        ax.plot(train_loss.index, train_loss.values, label="train")
    if not val_loss.empty:
        ax.plot(val_loss.index, val_loss.values, label="val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    if not train_dice.empty:
        ax.plot(train_dice.index, train_dice.values, label="train")
    if not val_dice.empty:
        ax.plot(val_dice.index, val_dice.values, label="val")
    ax.set_title("Dice")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    if not train_iou.empty:
        ax.plot(train_iou.index, train_iou.values, label="train")
    if not val_iou.empty:
        ax.plot(val_iou.index, val_iou.values, label="val")
    ax.set_title("IoU")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    if not lr.empty:
        ax.plot(lr.index, lr.values, label=lr.name or "lr")
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def iter_run_dirs(results_root: Path) -> Iterable[Path]:
    for cfg_path in results_root.glob("**/config.yaml"):
        run_dir = cfg_path.parent
        if _find_metrics_csv(run_dir) is None:
            continue
        yield run_dir


def run_to_fig_dir(results_root: Path, figures_root: Path, run_dir: Path) -> Optional[Path]:
    try:
        rel = run_dir.relative_to(results_root)
    except Exception:
        return None
    if len(rel.parts) < 5:
        return None
    experiment, dataset, model, seed_part = rel.parts[:4]
    run_id = "-".join(rel.parts[4:])
    return figures_root / experiment / dataset / model / seed_part / run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Results root.")
    parser.add_argument("--figures", default="figures", help="Figures root.")
    args = parser.parse_args()

    results_root = Path(args.results)
    figures_root = Path(args.figures)
    figures_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for run_dir in iter_run_dirs(results_root):
        fig_dir = run_to_fig_dir(results_root, figures_root, run_dir)
        if fig_dir is None:
            continue
        metrics_csv = _find_metrics_csv(run_dir)
        if metrics_csv is None:
            continue
        plot_training_curves(metrics_csv, fig_dir / "curves.png")
        n += 1

    print(f"Generated training curves for {n} runs.")


if __name__ == "__main__":
    main()
