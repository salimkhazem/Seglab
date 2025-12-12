"""Utilities to load Segment Anything Model (SAM)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def load_sam(
    sam_type: str = "vit_b",
    checkpoint: str | Path | None = None,
    device: Optional[torch.device] = None,
):
    try:
        from segment_anything import sam_model_registry
    except Exception as e:
        raise ImportError(
            "segment-anything is required for SAM models. Install with "
            "`pip install segment-anything` or see README."
        ) from e

    if checkpoint is None or not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}. "
            "Run scripts/download_sam_weights.py to download weights."
        )
    sam = sam_model_registry[sam_type](checkpoint=str(checkpoint))
    if device is not None:
        sam.to(device)
    return sam

