"""Joint augmentations using Albumentations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    size: int,
    train: bool = True,
    sar: bool = False,
    aug: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    aug = aug or {}
    if train:
        scale = aug.get("scale", (0.8, 1.2))
        try:
            min_scale, max_scale = float(scale[0]), float(scale[1])
        except Exception:
            min_scale, max_scale = 0.8, 1.2

        tfs = [
            A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=1.0),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, fill=0, fill_mask=0, p=1.0),
            A.RandomCrop(height=size, width=size, p=1.0),
            A.HorizontalFlip(p=aug.get("hflip", 0.5)),
            A.VerticalFlip(p=aug.get("vflip", 0.5)),
        ]
        if not sar:
            tfs.append(A.ColorJitter(p=aug.get("color_jitter_p", 0.2)))
        if aug.get("gauss_noise_p", 0.0) > 0:
            tfs.append(A.GaussNoise(p=aug["gauss_noise_p"]))
        tfs += [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        tfs = [
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    return A.Compose(tfs)
