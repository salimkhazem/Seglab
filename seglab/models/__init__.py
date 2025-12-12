"""Model builders and Lightning modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

from seglab.metrics.segmentation import SegmentationMeter, boundary_f_score
from seglab.metrics.topology import cldice_score
from seglab.metrics.calibration import expected_calibration_error
from seglab.models.sam_peft.topo_losses import (
    dice_loss,
    cldice_loss,
    boundary_loss,
)


class LitBinarySeg(pl.LightningModule):
    """Generic Lightning module for binary segmentation."""

    def __init__(self, net: nn.Module, cfg: Any) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))  # type: ignore

        self.train_meter = SegmentationMeter()
        self.val_meter = SegmentationMeter()
        self.test_meter = SegmentationMeter()
        self._test_run_dir: Optional[Path] = None
        self._test_pred_dir: Optional[Path] = None
        self._test_fig_dir: Optional[Path] = None
        self._test_image_counter: int = 0
        self._test_pixel_probs: List[np.ndarray] = []
        self._test_pixel_targets: List[np.ndarray] = []
        self._test_pixel_count: int = 0
        self._test_bf_sum: float = 0.0
        self._test_bf_n: int = 0
        self._test_cldice_sum: float = 0.0
        self._test_cldice_n: int = 0
        self._test_image_metrics: List[Dict[str, float]] = []
        self._best_samples: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._worst_samples: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        return logits

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f = target.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, target_f)
        dice = dice_loss(torch.sigmoid(logits), target_f)

        loss = (
            self.cfg.loss.bce_weight * bce
            + self.cfg.loss.dice_weight * dice
        )

        if getattr(self.cfg.loss, "cldice_weight", 0.0) > 0:
            loss = loss + self.cfg.loss.cldice_weight * cldice_loss(
                torch.sigmoid(logits), target_f
            )
        if getattr(self.cfg.loss, "boundary_weight", 0.0) > 0:
            loss = loss + self.cfg.loss.boundary_weight * boundary_loss(
                torch.sigmoid(logits), target_f
            )
        return loss

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)
        loss = self._compute_loss(logits, y)
        probs = torch.sigmoid(logits)

        meter = getattr(self, f"{stage}_meter")
        meter.update(probs.detach(), y.detach())
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)
        loss = self._compute_loss(logits, y)
        probs = torch.sigmoid(logits)
        self.test_meter.update(probs.detach(), y.detach())
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Save predictions (PNG) + gather samples for PR/calibration + qualitative grids.
        pred_dir = self._test_pred_dir
        max_pixels = int(getattr(getattr(self.cfg, "artifacts", {}), "sample_pixels", 200_000))
        pixels_per_image = int(getattr(getattr(self.cfg, "artifacts", {}), "sample_pixels_per_image", 5_000))
        max_qual = int(getattr(getattr(self.cfg, "artifacts", {}), "max_qual", 8))
        save_preds = bool(getattr(getattr(self.cfg, "artifacts", {}), "save_test_preds", True))

        probs_cpu = probs.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy().astype(np.uint8)
        x_cpu = x.detach().cpu().numpy()
        pred_bin = (probs_cpu > 0.5).astype(np.uint8)

        if save_preds and pred_dir is not None:
            from PIL import Image

            for b in range(pred_bin.shape[0]):
                idx = self._test_image_counter + b
                Image.fromarray(pred_bin[b, 0] * 255).save(
                    pred_dir / f"{idx:06d}.png",
                    format="PNG",
                    compress_level=9,
                    optimize=True,
                )

        # Per-image metrics (for best/worst selection and optional CSV)
        for b in range(pred_bin.shape[0]):
            pr = pred_bin[b, 0]
            gt = y_cpu[b]
            tp = float((pr * gt).sum())
            fp = float((pr * (1 - gt)).sum())
            fn = float(((1 - pr) * gt).sum())
            eps = 1e-6
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            dice = 2 * tp / (2 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)

            self._test_image_metrics.append(
                {
                    "index": float(self._test_image_counter + b),
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                }
            )

            # Boundary F-score (tolerance=1 by default)
            try:
                bf = boundary_f_score(pr, gt, tol=int(getattr(getattr(self.cfg, "artifacts", {}), "bf_tol", 1)))
                self._test_bf_sum += float(bf)
                self._test_bf_n += 1
                self._test_image_metrics[-1]["bfscore"] = float(bf)
            except Exception:
                pass

            # clDice only for retina datasets by default
            if getattr(self.cfg.dataset, "type", "") == "hf_retina":
                try:
                    cld = float(cldice_score(pr[None, None, ...], gt[None, ...]))
                    self._test_cldice_sum += cld
                    self._test_cldice_n += 1
                    self._test_image_metrics[-1]["cldice"] = cld
                except Exception:
                    pass

            # Keep best/worst qualitative samples
            sample = (dice, x_cpu[b], gt, pr)
            self._best_samples.append(sample)
            self._best_samples = sorted(self._best_samples, key=lambda t: t[0], reverse=True)[:max_qual]
            self._worst_samples.append(sample)
            self._worst_samples = sorted(self._worst_samples, key=lambda t: t[0])[:max_qual]

            # Pixel sampling for PR/ECE/reliability
            if self._test_pixel_count < max_pixels:
                flat_p = probs_cpu[b, 0].reshape(-1)
                flat_t = gt.reshape(-1).astype(np.uint8)
                remaining = max_pixels - self._test_pixel_count
                k = int(min(pixels_per_image, remaining, flat_p.size))
                if k > 0:
                    idxs = self._rng.integers(0, flat_p.size, size=k, endpoint=False)
                    self._test_pixel_probs.append(flat_p[idxs])
                    self._test_pixel_targets.append(flat_t[idxs])
                    self._test_pixel_count += k

        self._test_image_counter += int(pred_bin.shape[0])

    def on_train_epoch_end(self) -> None:
        self.train_meter.log(self, prefix="train")
        self.train_meter.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_meter.log(self, prefix="val")
        self.val_meter.reset()

    def on_test_epoch_end(self) -> None:
        self.test_meter.log(self, prefix="test")
        self.test_meter.reset()
        # Aggregate BFScore/clDice means
        if self._test_bf_n > 0:
            self.log("test/bfscore", float(self._test_bf_sum / max(self._test_bf_n, 1)))
        if self._test_cldice_n > 0:
            self.log("test/cldice", float(self._test_cldice_sum / max(self._test_cldice_n, 1)))

        # Calibration + PR curves from sampled pixels
        if self._test_pixel_probs:
            probs_np = np.concatenate(self._test_pixel_probs, axis=0)
            targets_np = np.concatenate(self._test_pixel_targets, axis=0)
            try:
                ece = expected_calibration_error(
                    torch.from_numpy(probs_np), torch.from_numpy(targets_np)
                )
                self.log("test/ece", float(ece))
            except Exception:
                pass

            save_figures = bool(getattr(getattr(self.cfg, "artifacts", {}), "save_figures", True))
            if save_figures and self._test_fig_dir is not None:
                try:
                    from seglab.utils.viz import plot_pr_curve, plot_reliability_diagram, save_qualitative_grid

                    plot_pr_curve(probs_np, targets_np, self._test_fig_dir / "pr_curve.png")
                    plot_reliability_diagram(
                        probs_np, targets_np, self._test_fig_dir / "reliability.png", n_bins=15
                    )

                    if self._best_samples:
                        imgs = np.stack([s[1] for s in self._best_samples], axis=0)
                        gts = np.stack([s[2] for s in self._best_samples], axis=0)
                        prs = np.stack([s[3] for s in self._best_samples], axis=0)
                        save_qualitative_grid(imgs, gts, prs, self._test_fig_dir / "qual_best.png")
                    if self._worst_samples:
                        imgs = np.stack([s[1] for s in self._worst_samples], axis=0)
                        gts = np.stack([s[2] for s in self._worst_samples], axis=0)
                        prs = np.stack([s[3] for s in self._worst_samples], axis=0)
                        save_qualitative_grid(imgs, gts, prs, self._test_fig_dir / "qual_worst.png")
                except Exception as e:
                    self.print(f"[warn] failed to create test figures: {e}")

        # Save per-image test metrics CSV (if possible)
        if self._test_run_dir is not None and self._test_image_metrics:
            try:
                import pandas as pd

                pd.DataFrame(self._test_image_metrics).to_csv(
                    self._test_run_dir / "test_metrics_per_image.csv", index=False
                )
            except Exception:
                pass

        # Reset artifact buffers
        self._test_pixel_probs.clear()
        self._test_pixel_targets.clear()
        self._test_pixel_count = 0
        self._test_bf_sum = 0.0
        self._test_bf_n = 0
        self._test_cldice_sum = 0.0
        self._test_cldice_n = 0
        self._test_image_metrics.clear()
        self._best_samples.clear()
        self._worst_samples.clear()

    def on_test_start(self) -> None:
        run_dir = Path(self.trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._test_run_dir = run_dir

        # Save per-image binary predictions in the run directory.
        pred_dir = run_dir / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        self._test_pred_dir = pred_dir

        # Save figures under global figures/ with the same run id.
        fig_root = Path(getattr(getattr(self.cfg, "paths", {}), "figures_dir", "figures"))
        # Robust run id (supports optional nested tags).
        try:
            seed_root = (
                Path(getattr(getattr(self.cfg, "paths", {}), "results_dir", "results"))
                / str(getattr(self.cfg, "experiment", "default"))
                / str(getattr(self.cfg.dataset, "name", "dataset"))
                / str(getattr(self.cfg.model, "name", "model"))
                / f"seed{int(getattr(self.cfg, 'seed', 0))}"
            )
            run_id = "-".join(run_dir.relative_to(seed_root).parts)
        except Exception:
            run_id = run_dir.name
        fig_dir = (
            fig_root
            / str(getattr(self.cfg, "experiment", "default"))
            / str(getattr(self.cfg.dataset, "name", "dataset"))
            / str(getattr(self.cfg.model, "name", "model"))
            / f"seed{int(getattr(self.cfg, 'seed', 0))}"
            / run_id
        )
        fig_dir.mkdir(parents=True, exist_ok=True)
        self._test_fig_dir = fig_dir

        self._test_image_counter = 0

    def configure_optimizers(self):
        opt_name = self.cfg.optimizer.get("name", "adamw").lower()
        lr = self.cfg.optimizer.get("lr", 1e-4)
        wd = self.cfg.optimizer.get("weight_decay", 1e-4)
        if opt_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sched_cfg = self.cfg.get("scheduler", {})
        if sched_cfg.get("name", "none") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": scheduler}
        return opt


# Import model builders to register them.
from .unet_smp import build_unet  # noqa: E402,F401
from .deeplabv3p_smp import build_deeplabv3p  # noqa: E402,F401
from .segformer_hf import build_segformer  # noqa: E402,F401
from .mask2former_d2 import build_mask2former  # noqa: E402,F401
from .sam_peft.model import build_sam_topolora  # noqa: E402,F401
