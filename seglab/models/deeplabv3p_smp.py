"""DeepLabV3+ baseline using segmentation_models_pytorch."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp

from seglab.models import LitBinarySeg
from seglab.utils.registry import register_model


@register_model("deeplabv3p")
def build_deeplabv3p(cfg: Any) -> LitBinarySeg:
    net = smp.DeepLabV3Plus(
        encoder_name=cfg.model.encoder,
        encoder_weights="imagenet" if cfg.model.pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return LitBinarySeg(net, cfg)

