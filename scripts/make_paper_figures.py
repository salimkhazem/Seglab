"""Generate paper-ready figures (high quality, publication style).

Outputs are written under `figures/paper/` by default as both PDF (vector) and
PNG (high-res raster). The script reads completed runs under `results/` and
creates:
  - method schematic (TopoLoRA-SAM overview)
  - benchmark heatmaps (Dice, clDice, BFScore, ECE)
  - parameter-efficiency scatter (performance vs trainable params)
  - qualitative comparison grids (baseline vs ours) for retina + SL-SSDD

Usage:
  PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
    .venv/bin/python scripts/make_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from omegaconf import OmegaConf, DictConfig
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seglab.data import HFRetinaDataset, HFKvasirDataset, SLSSDDDataset, build_transforms


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#7F7F7F",
}


MODEL_LABEL = {
    "unet": "U-Net (R34)",
    "deeplabv3p": "DeepLabV3+ (R50)",
    "segformer": "SegFormer (B0)",
    "mask2former": "Mask2Former (Swin-T)",
    "sam_topolora": "TopoLoRA-SAM (ours)",
}

MODEL_ORDER = ["unet", "deeplabv3p", "segformer", "mask2former", "sam_topolora"]
DATASET_ORDER = ["drive", "stare", "chase", "kvasir", "sl_ssdd"]
RETINA_DATASETS = ["drive", "stare", "chase"]


def set_paper_style() -> None:
    """Configure matplotlib for paper-quality figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed TrueType fonts
            "ps.fonttype": 42,
        }
    )


def save_both(fig: plt.Figure, out_base: Path, dpi: int = 400) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=dpi)


def _human_ds(name: str) -> str:
    return {
        "drive": "DRIVE",
        "stare": "STARE",
        "chase": "CHASE_DB1",
        "kvasir": "Kvasir-SEG",
        "sl_ssdd": "SL-SSDD",
    }.get(name, name)


def _read_json(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text())


def collect_latest_runs(results_root: Path, experiment: str) -> pd.DataFrame:
    """Collect latest run per (dataset,model,seed) under the given experiment."""
    rows: List[Dict[str, object]] = []
    for path in results_root.glob("**/test_metrics.json"):
        try:
            rel = path.relative_to(results_root)
        except Exception:
            continue
        if len(rel.parts) < 6:
            continue
        exp, dataset, model, seed_part = rel.parts[:4]
        if exp != experiment:
            continue
        if not seed_part.startswith("seed"):
            continue
        try:
            seed = int(seed_part.replace("seed", ""))
        except Exception:
            continue
        run_id = "-".join(rel.parts[4:-1])
        metrics = _read_json(path)
        run_dir = path.parent
        row: Dict[str, object] = {
            "experiment": exp,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "run_id": run_id,
            "run_dir": str(run_dir),
        }
        row.update(metrics)
        tp = run_dir / "trainable_params.txt"
        if tp.exists():
            try:
                row["trainable_params"] = int(tp.read_text().strip())
            except Exception:
                pass
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["dataset", "model", "seed", "run_id"])
    df = df.drop_duplicates(subset=["dataset", "model", "seed"], keep="last")
    return df


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float


def summarize_mean_std(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (means, stds) indexed by (dataset,model)."""
    group = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    means = group.xs("mean", axis=1, level=1)
    stds = group.xs("std", axis=1, level=1)
    return means, stds


def _draw_method_schematic(ax: plt.Axes) -> None:
    ax.axis("off")

    def box(xy: Tuple[float, float], wh: Tuple[float, float], text: str, fc: str, ec: str, lw: float = 1.0):
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold")

    def arrow(xy0: Tuple[float, float], xy1: Tuple[float, float], color: str = OKABE_ITO["gray"]):
        ax.add_patch(
            FancyArrowPatch(
                xy0,
                xy1,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.2,
                color=color,
            )
        )

    # Layout in axes coordinates.
    box((0.02, 0.55), (0.14, 0.30), "Input\nimage", fc="#F7F7F7", ec="#B0B0B0")
    arrow((0.16, 0.70), (0.24, 0.70))

    box((0.24, 0.55), (0.28, 0.30), "SAM Image Encoder\n(frozen) + LoRA", fc="#EAF2FF", ec=OKABE_ITO["blue"])
    ax.text(
        0.38,
        0.52,
        r"LoRA: $W \leftarrow W_0 + BA$ (rank-$r$)",
        ha="center",
        va="top",
        fontsize=8,
        color=OKABE_ITO["blue"],
    )

    arrow((0.52, 0.70), (0.60, 0.70))
    box((0.60, 0.55), (0.18, 0.30), "Conv Adapter\n(trainable)", fc="#E8FFF3", ec=OKABE_ITO["bluish_green"])
    arrow((0.78, 0.70), (0.86, 0.70))
    box((0.86, 0.55), (0.12, 0.30), "Mask\nDecoder\n(trainable)", fc="#FFF2E6", ec=OKABE_ITO["orange"])

    # Output arrow & mask.
    arrow((0.92, 0.55), (0.92, 0.30))
    box((0.86, 0.10), (0.12, 0.18), "Mask\n$\\hat{y}$", fc="#F7F7F7", ec="#B0B0B0")

    # Loss block.
    box(
        (0.24, 0.08),
        (0.54, 0.18),
        "Training objective: BCE + Dice + λ·clDice (soft skeleton) + optional boundary loss",
        fc="#F2F2F2",
        ec="#B0B0B0",
        lw=0.9,
    )
    ax.text(
        0.51,
        0.02,
        "Key idea: topology-aware regularization + parameter-efficient tuning (≈5M trainable params)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=OKABE_ITO["black"],
    )


def plot_method_schematic() -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    _draw_method_schematic(ax)
    return fig


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        weight="bold",
    )


def _heatmap(
    data: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
    std: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    ds_labels = [_human_ds(d) for d in data.index.tolist()]
    model_labels = [MODEL_LABEL.get(m, m) for m in data.columns.tolist()]

    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(ds_labels)))
    ax.set_yticklabels(ds_labels)

    # Annotate numbers and highlight best per dataset row.
    for i in range(data.shape[0]):
        row = data.iloc[i].values.astype(float)
        j_best = int(np.nanargmax(row))
        for j in range(data.shape[1]):
            val = float(data.iloc[i, j])
            s = fmt.format(val)
            if std is not None:
                sd = float(std.iloc[i, j])
                if not math.isnan(sd):
                    s = f"{val:.3f}±{sd:.3f}"
            kw = dict(ha="center", va="center", fontsize=8, color="black")
            if j == j_best:
                kw["weight"] = "bold"
            ax.text(j, i, s, **kw)
        ax.add_patch(
            FancyBboxPatch(
                (j_best - 0.48, i - 0.48),
                0.96,
                0.96,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="none",
                edgecolor=OKABE_ITO["vermillion"],
                linewidth=1.6,
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=12)
    fig.tight_layout()
    return fig


def plot_param_tradeoff(df_mean: pd.DataFrame, out_metric: str) -> plt.Figure:
    """Scatter: performance vs trainable params (log-scale)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    # Average metric across retina datasets for each model (paper focus).
    rows = df_mean[df_mean["dataset"].isin(RETINA_DATASETS)].copy()
    agg = rows.groupby("model").agg(
        {
            out_metric: "mean",
            "trainable_params": "mean",
        }
    )
    agg = agg.reindex(MODEL_ORDER)

    for model in agg.index:
        x = float(agg.loc[model, "trainable_params"])
        y = float(agg.loc[model, out_metric])
        color = OKABE_ITO["vermillion"] if model == "sam_topolora" else OKABE_ITO["gray"]
        marker = "o" if model != "sam_topolora" else "D"
        ax.scatter(x, y, s=70, color=color, marker=marker, zorder=3)
        label = MODEL_LABEL.get(model, model)
        ax.text(x * 1.05, y, label, fontsize=8, va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(out_metric.replace("test/", "").upper())
    ax.set_title(f"Retina trade-off: {out_metric.replace('test/', '').upper()} vs trainable params")
    ax.grid(True, which="both", alpha=0.25)
    return fig


def _plot_param_tradeoff_ax(ax: plt.Axes, df_mean: pd.DataFrame, out_metric: str) -> None:
    rows = df_mean[df_mean["dataset"].isin(RETINA_DATASETS)].copy()
    agg = rows.groupby("model").agg({out_metric: "mean", "trainable_params": "mean"}).reindex(MODEL_ORDER)
    for model in agg.index:
        x = float(agg.loc[model, "trainable_params"])
        y = float(agg.loc[model, out_metric])
        color = OKABE_ITO["vermillion"] if model == "sam_topolora" else OKABE_ITO["gray"]
        marker = "o" if model != "sam_topolora" else "D"
        ax.scatter(x, y, s=55, color=color, marker=marker, zorder=3)
        ax.text(x * 1.08, y, MODEL_LABEL.get(model, model), fontsize=7, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable params (log)")
    ax.set_ylabel(out_metric.replace("test/", "").upper())
    ax.set_title("Retina: params vs Dice")
    ax.grid(True, which="both", alpha=0.25)


def _load_run_cfg(run_dir: Path) -> DictConfig:
    return OmegaConf.load(str(run_dir / "config.yaml"))


def _build_test_dataset(cfg: DictConfig):
    tf = build_transforms(
        size=int(cfg.dataset.size),
        train=False,
        sar=bool(cfg.dataset.get("sar", False)),
    )
    ds_type = str(cfg.dataset.type)
    cache_dir = Path(str(cfg.paths.cache_dir))
    if ds_type == "hf_retina":
        name = cfg.dataset.get("hf_name") or cfg.dataset.name
        return HFRetinaDataset(
            name=name,
            split=str(cfg.dataset.test_split),
            transforms=tf,
            cache_dir=cache_dir / "hf",
            image_key=str(cfg.dataset.get("image_key", "image")),
            mask_key=str(cfg.dataset.get("mask_key", "label")),
        )
    if ds_type == "hf_kvasir":
        return HFKvasirDataset(
            split=str(cfg.dataset.test_split),
            transforms=tf,
            cache_dir=cache_dir / "hf",
            hf_name=str(cfg.dataset.get("hf_name") or "Angelou0516/kvasir-seg"),
            image_key=str(cfg.dataset.get("image_key", "image")),
            mask_key=str(cfg.dataset.get("mask_key", "mask")),
        )
    if ds_type == "sl_ssdd":
        def _read_list(p: str) -> List[str]:
            return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

        files = _read_list(str(cfg.dataset.test_list))
        return SLSSDDDataset(root=str(cfg.dataset.root), split_files=files, transforms=tf)
    raise ValueError(f"Unsupported dataset type for figures: {ds_type}")


def _unnormalize_imagenet(img_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    img = img_chw * std + mean
    img = np.clip(img, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def _load_pred_mask(run_dir: Path, index: int) -> np.ndarray:
    p = run_dir / "preds" / f"{index:06d}.png"
    arr = np.array(Image.open(p).convert("L"))
    return (arr > 127).astype(np.uint8)


def _pick_latest_run_dir(results_root: Path, dataset: str, model: str, seed: int) -> Optional[Path]:
    seed_dir = results_root / "benchmark_matrix" / dataset / model / f"seed{seed}"
    if not seed_dir.exists():
        return None
    candidates = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def _pick_best_improvement_index(
    ours_dir: Path,
    base_dir: Path,
    metric: str,
    top_k: int = 200,
) -> int:
    """Pick an index where ours improves most over baseline by a per-image metric."""
    ours_csv = ours_dir / "test_metrics_per_image.csv"
    base_csv = base_dir / "test_metrics_per_image.csv"
    if not ours_csv.exists() or not base_csv.exists():
        return 0
    ours = pd.read_csv(ours_csv)
    base = pd.read_csv(base_csv)
    if "index" not in ours.columns or "index" not in base.columns:
        return 0
    ours["index"] = ours["index"].astype(int)
    base["index"] = base["index"].astype(int)
    if metric not in ours.columns or metric not in base.columns:
        metric = "dice" if "dice" in ours.columns and "dice" in base.columns else metric
    if metric not in ours.columns or metric not in base.columns:
        return int(ours["index"].iloc[0]) if len(ours) else 0

    merged = ours[["index", metric]].merge(base[["index", metric]], on="index", suffixes=("_ours", "_base"))
    merged["delta"] = merged[f"{metric}_ours"] - merged[f"{metric}_base"]
    merged = merged.sort_values("delta", ascending=False)
    merged = merged.head(max(top_k, 1))
    # Prefer non-trivial cases: exclude perfect predictions.
    merged = merged[merged[f"{metric}_ours"] < 0.999]
    if merged.empty:
        merged = ours[["index"]].head(1)
    return int(merged["index"].iloc[0])


def plot_qualitative_comparison(
    results_root: Path,
    dataset: str,
    baseline_model: str,
    ours_model: str,
    seed: int,
) -> plt.Figure:
    ours_dir = _pick_latest_run_dir(results_root, dataset, ours_model, seed)
    base_dir = _pick_latest_run_dir(results_root, dataset, baseline_model, seed)
    if ours_dir is None or base_dir is None:
        raise FileNotFoundError(f"Missing run dirs for {dataset} seed{seed}")

    cfg = _load_run_cfg(ours_dir)
    ds = _build_test_dataset(cfg)

    # Choose an index where topology improves the most (retina) or Dice otherwise.
    metric = "cldice" if dataset in RETINA_DATASETS else "dice"
    idx = _pick_best_improvement_index(ours_dir, base_dir, metric=metric)

    sample = ds[idx]
    img = sample["image"].detach().cpu().numpy()
    gt = sample["mask"].detach().cpu().numpy().astype(np.uint8)
    img_vis = _unnormalize_imagenet(img)
    pr_base = _load_pred_mask(base_dir, idx)
    pr_ours = _load_pred_mask(ours_dir, idx)

    err_base = (pr_base != gt).astype(np.int8)
    err_ours = (pr_ours != gt).astype(np.int8)
    delta_err = err_base - err_ours  # +1 means ours fixed an error

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img_vis)
    axes[0].set_title(f"{_human_ds(dataset)} input")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(pr_base, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"{MODEL_LABEL.get(baseline_model, baseline_model)}")
    axes[3].imshow(pr_ours, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"{MODEL_LABEL.get(ours_model, ours_model)}")

    im = axes[4].imshow(delta_err, cmap="bwr", vmin=-1, vmax=1)
    axes[4].set_title("Δ error\n(+ = ours better)")
    cbar = fig.colorbar(im, ax=axes[4], fraction=0.05, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["baseline better", "same", "ours better"])
    fig.suptitle(
        f"Qualitative comparison on {_human_ds(dataset)} (seed{seed}, index={idx})",
        y=1.05,
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_teaser(
    results_root: Path,
    df_mean: pd.DataFrame,
    seed: int,
    retina_baseline: str,
) -> plt.Figure:
    """A compact, paper-style teaser: method + pareto + qualitative."""
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.7], wspace=0.28)

    # (a) method
    ax0 = fig.add_subplot(gs[0, 0])
    _draw_method_schematic(ax0)
    _panel_label(ax0, "(a)")

    # (b) trade-off
    ax1 = fig.add_subplot(gs[0, 1])
    _plot_param_tradeoff_ax(ax1, df_mean, out_metric="test/dice")
    _panel_label(ax1, "(b)")

    # (c) qualitative (CHASE: baseline vs ours)
    outer = fig.add_subplot(gs[0, 2])
    outer.axis("off")
    _panel_label(outer, "(c)")
    sub = gs[0, 2].subgridspec(1, 4, wspace=0.02)

    ours_dir = _pick_latest_run_dir(results_root, "chase", "sam_topolora", seed)
    base_dir = _pick_latest_run_dir(results_root, "chase", retina_baseline, seed)
    if ours_dir is None or base_dir is None:
        return fig
    cfg = _load_run_cfg(ours_dir)
    ds = _build_test_dataset(cfg)
    idx = _pick_best_improvement_index(ours_dir, base_dir, metric="cldice")
    sample = ds[idx]
    img = _unnormalize_imagenet(sample["image"].detach().cpu().numpy())
    gt = sample["mask"].detach().cpu().numpy().astype(np.uint8)
    pr_base = _load_pred_mask(base_dir, idx)
    pr_ours = _load_pred_mask(ours_dir, idx)
    delta_err = (pr_base != gt).astype(np.int8) - (pr_ours != gt).astype(np.int8)

    titles = ["Input", "GT", "Baseline", "Ours"]
    panels = [img, gt, pr_base, pr_ours]
    for j in range(4):
        ax = fig.add_subplot(sub[0, j])
        ax.axis("off")
        if j == 0:
            ax.imshow(panels[j])
        else:
            ax.imshow(panels[j], cmap="gray", vmin=0, vmax=1)
        ax.set_title(titles[j], fontsize=8)

    # small inset for improvement map
    inset = outer.inset_axes([0.70, 0.02, 0.28, 0.28])
    inset.axis("off")
    inset.imshow(delta_err, cmap="bwr", vmin=-1, vmax=1)
    inset.set_title("Δerr", fontsize=7)

    # Avoid `tight_layout()` here because inset axes can cause layout warnings.
    fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.12, wspace=0.28)
    return fig


def main() -> None:
    set_paper_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Results root directory.")
    parser.add_argument("--experiment", default="benchmark_matrix", help="Experiment name.")
    parser.add_argument("--out", default="figures/paper", help="Output directory for paper figures.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for qualitative comparisons.")
    parser.add_argument("--retina_baseline", default="mask2former", help="Baseline model for retina qualitative.")
    parser.add_argument("--sar_baseline", default="segformer", help="Baseline model for SL-SSDD qualitative.")
    args = parser.parse_args()

    results_root = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_latest_runs(results_root, args.experiment)
    if df.empty:
        raise SystemExit(f"No results found under {results_root} for experiment '{args.experiment}'.")

    metrics = [c for c in df.columns if isinstance(c, str) and c.startswith("test/")]
    if "trainable_params" in df.columns:
        metrics.append("trainable_params")

    # Summary frames for heatmaps.
    df_mean, df_std = summarize_mean_std(df, metrics=metrics)
    df_mean = df_mean.reset_index()
    df_std = df_std.reset_index()

    # Pivot for heatmaps.
    def pivot(metric: str, src: pd.DataFrame) -> pd.DataFrame:
        pv = src.pivot(index="dataset", columns="model", values=metric)
        pv = pv.reindex(DATASET_ORDER)
        pv = pv.reindex(columns=MODEL_ORDER)
        return pv

    # Fig 1: method schematic
    fig1 = plot_method_schematic()
    save_both(fig1, out_dir / "fig1_method")
    plt.close(fig1)

    # Fig 0: teaser (method + pareto + qualitative)
    fig0 = plot_teaser(results_root, df_mean=df_mean, seed=int(args.seed), retina_baseline=str(args.retina_baseline))
    save_both(fig0, out_dir / "fig0_teaser")
    plt.close(fig0)

    # Fig 2: benchmark heatmap (Dice)
    dice_mean = pivot("test/dice", df_mean)
    dice_std = pivot("test/dice", df_std)
    fig2 = _heatmap(
        dice_mean,
        title="Benchmark (mean±std over 3 seeds): Dice",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        fmt="{:.3f}",
        std=dice_std,
    )
    save_both(fig2, out_dir / "fig2_heatmap_dice")
    plt.close(fig2)

    # Fig 3: retina topology heatmap (clDice) if available
    if "test/cldice" in df_mean.columns:
        cld_mean = pivot("test/cldice", df_mean).loc[RETINA_DATASETS]
        cld_std = pivot("test/cldice", df_std).loc[RETINA_DATASETS]
        fig3 = _heatmap(
            cld_mean,
            title="Thin-structure topology (mean±std): clDice (retina only)",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            fmt="{:.3f}",
            std=cld_std,
        )
        save_both(fig3, out_dir / "fig3_heatmap_cldice")
        plt.close(fig3)

    # Fig 4: parameter-efficiency trade-off (retina)
    if "trainable_params" in df_mean.columns:
        fig4 = plot_param_tradeoff(df_mean, out_metric="test/dice")
        save_both(fig4, out_dir / "fig4_tradeoff_retina_dice")
        plt.close(fig4)

    # Fig 5: qualitative comparisons (retina)
    try:
        fig5a = plot_qualitative_comparison(
            results_root, dataset="drive", baseline_model=args.retina_baseline, ours_model="sam_topolora", seed=args.seed
        )
        save_both(fig5a, out_dir / "fig5_qual_drive")
        plt.close(fig5a)
    except Exception as e:
        print(f"[warn] could not generate DRIVE qualitative: {e}")

    try:
        fig5b = plot_qualitative_comparison(
            results_root, dataset="chase", baseline_model=args.retina_baseline, ours_model="sam_topolora", seed=args.seed
        )
        save_both(fig5b, out_dir / "fig5_qual_chase")
        plt.close(fig5b)
    except Exception as e:
        print(f"[warn] could not generate CHASE qualitative: {e}")

    # Fig 6: qualitative comparison (SL-SSDD)
    try:
        fig6 = plot_qualitative_comparison(
            results_root, dataset="sl_ssdd", baseline_model=args.sar_baseline, ours_model="sam_topolora", seed=args.seed
        )
        save_both(fig6, out_dir / "fig6_qual_sl_ssdd")
        plt.close(fig6)
    except Exception as e:
        print(f"[warn] could not generate SL-SSDD qualitative: {e}")

    # Fig 7: calibration (ECE) heatmap if available
    if "test/ece" in df_mean.columns:
        ece_mean = pivot("test/ece", df_mean)
        ece_std = pivot("test/ece", df_std)
        # Lower is better: invert colormap orientation by using reversed cmap.
        vmax = float(np.nanpercentile(ece_mean.values, 95)) if np.isfinite(ece_mean.values).any() else 0.5
        fig7 = _heatmap(
            ece_mean,
            title="Calibration (mean±std): Expected Calibration Error (lower is better)",
            cmap="viridis_r",
            vmin=0.0,
            vmax=max(vmax, 1e-3),
            fmt="{:.3f}",
            std=ece_std,
        )
        save_both(fig7, out_dir / "fig7_heatmap_ece")
        plt.close(fig7)

    print(f"Saved paper figures to {out_dir}")


if __name__ == "__main__":
    main()
