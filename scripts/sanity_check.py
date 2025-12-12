"""Quick sanity checks for dataloaders and models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seglab.train import build_dataloaders, _merge_cfg
from seglab.utils import load_config, get_model


def main() -> None:
    base = load_config("configs/base.yaml")
    cfg = _merge_cfg(base, "drive", "unet")

    try:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
        batch = next(iter(train_loader))
        x = batch["image"]
        print("Loaded batch:", x.shape)
    except Exception as e:
        print("[warn] dataset load failed (download datasets first):", e)
        x = torch.randn(2, 3, cfg.dataset.size, cfg.dataset.size)

    model = get_model(cfg.model.name)(cfg)
    logits = model(x)
    assert logits.shape[-2:] == x.shape[-2:], "Output spatial size mismatch"
    print("Forward pass OK:", logits.shape)


if __name__ == "__main__":
    main()
