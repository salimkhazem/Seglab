"""Training entrypoint.

Usage:
  python -m seglab.train --config configs/base.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader, Subset

from seglab.data import HFRetinaDataset, HFKvasirDataset, SLSSDDDataset, build_transforms
from seglab.data.splits import make_split_indices
from seglab.utils import (
    seed_everything,
    load_config,
    save_config,
    make_run_dir,
    collect_env_info,
    get_model,
)
from seglab.utils.env import load_dotenv
from seglab.utils.io import copy_code_snapshot


def _merge_cfg(base_cfg: DictConfig, dataset_name: str, model_name: str) -> DictConfig:
    cfg = base_cfg.copy()
    ds_cfg_path = Path("configs/datasets") / f"{dataset_name}.yaml"
    if ds_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ds_cfg_path))
    mdl_cfg_path = Path("configs/models") / f"{model_name}.yaml"
    if mdl_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(mdl_cfg_path))
    return cfg


def build_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    cache_dir = Path(cfg.paths.cache_dir)
    ds_type = cfg.dataset.type
    size = cfg.dataset.size

    tf_train = build_transforms(size=size, train=True, sar=cfg.dataset.get("sar", False), aug=cfg.dataset.get("aug"))
    tf_eval = build_transforms(size=size, train=False, sar=cfg.dataset.get("sar", False))

    if ds_type == "hf_retina":
        name = cfg.dataset.get("hf_name") or cfg.dataset.name
        train_full = HFRetinaDataset(
            name=name,
            split=cfg.dataset.train_split,
            transforms=tf_train,
            cache_dir=cache_dir / "hf",
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "label"),
        )
        splits = make_split_indices(
            len(train_full),
            cfg.seed,
            val_ratio=cfg.dataset.val_ratio,
            cache_path=cache_dir / "splits" / f"{cfg.dataset.name}_seed{cfg.seed}.json",
        )
        train_ds = Subset(train_full, splits["train"])
        val_ds = Subset(train_full, splits["val"])
        test_ds = HFRetinaDataset(
            name=name,
            split=cfg.dataset.test_split,
            transforms=tf_eval,
            cache_dir=cache_dir / "hf",
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "label"),
        )
    elif ds_type == "hf_kvasir":
        train_full = HFKvasirDataset(
            split=cfg.dataset.train_split,
            transforms=tf_train,
            cache_dir=cache_dir / "hf",
            hf_name=cfg.dataset.get("hf_name"),
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "mask"),
        )
        splits = make_split_indices(
            len(train_full),
            cfg.seed,
            val_ratio=cfg.dataset.val_ratio,
            cache_path=cache_dir / "splits" / f"{cfg.dataset.name}_seed{cfg.seed}.json",
        )
        train_ds = Subset(train_full, splits["train"])
        val_ds = Subset(train_full, splits["val"])
        test_ds = HFKvasirDataset(
            split=cfg.dataset.test_split,
            transforms=tf_eval,
            cache_dir=cache_dir / "hf",
            hf_name=cfg.dataset.get("hf_name"),
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "mask"),
        )
    elif ds_type == "sl_ssdd":
        root = cfg.dataset.root
        def _read_list(p: str) -> list[str]:
            return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

        train_files = _read_list(cfg.dataset.train_list)
        val_files = _read_list(cfg.dataset.val_list)
        test_files = _read_list(cfg.dataset.test_list)
        train_ds = SLSSDDDataset(root=root, split_files=train_files, transforms=tf_train)
        val_ds = SLSSDDDataset(root=root, split_files=val_files, transforms=tf_eval)
        test_ds = SLSSDDDataset(root=root, split_files=test_files, transforms=tf_eval)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    dl_kwargs = dict(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader, test_loader


def run_experiment(cfg: DictConfig, tag: Optional[str] = None) -> Path:
    load_dotenv()
    seed_everything(int(cfg.seed), deterministic=cfg.trainer.get("deterministic", True))

    run_dir = make_run_dir(
        cfg.paths.results_dir,
        cfg.experiment,
        cfg.dataset.name,
        cfg.model.name,
        int(cfg.seed),
        tag=tag,
    )
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / cfg.experiment / cfg.dataset.name / cfg.model.name / f"seed{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir)
    env_info = collect_env_info()
    (run_dir / "env.json").write_text(json.dumps(env_info, indent=2))
    copy_code_snapshot(run_dir)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model_builder = get_model(cfg.model.name)
    lit_module = model_builder(cfg)

    trainable_params = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    (run_dir / "trainable_params.txt").write_text(str(trainable_params))

    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs")
    loggers = [csv_logger]
    if cfg.logging.get("wandb", False):
        try:
            from pytorch_lightning.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=cfg.logging.project,
                    entity=cfg.logging.get("entity"),
                    tags=list(cfg.logging.get("tags", [])),
                    save_dir=str(run_dir),
                )
            )
        except Exception as e:
            print(f"[warn] wandb logging requested but unavailable: {e}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        filename="{epoch}-{val_dice:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.get("deterministic", True),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.0),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=loggers,
        default_root_dir=str(run_dir),
    )

    trainer.fit(lit_module, train_loader, val_loader)
    best_path = checkpoint_cb.best_model_path
    (run_dir / "best_ckpt.txt").write_text(best_path)
    # Final test on best checkpoint
    test_results = trainer.test(lit_module, dataloaders=test_loader, ckpt_path="best")
    if test_results:
        (run_dir / "test_metrics.json").write_text(json.dumps(test_results[0], indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dataset", default=None, help="Dataset config name (e.g., drive).")
    parser.add_argument("--model", default=None, help="Model config name (e.g., unet).")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    dataset_name = args.dataset or base_cfg.dataset.name
    model_name = args.model or base_cfg.model.name
    cfg = _merge_cfg(base_cfg, dataset_name, model_name)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    run_experiment(cfg)


if __name__ == "__main__":
    main()
