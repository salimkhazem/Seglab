"""SegFormer baseline using HuggingFace transformers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from seglab.models import LitBinarySeg
from seglab.utils.registry import register_model


class SegFormerNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except Exception as e:
            raise ImportError(
                "SegFormer requires `transformers` (and a compatible torch/optional CUDA extensions). "
                "If you have issues, try a clean venv without conflicting user-site packages."
            ) from e

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        if not pretrained:
            self.model.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, return_dict=True)
        logits = out.logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


@register_model("segformer")
def build_segformer(cfg: Any) -> LitBinarySeg:
    net = SegFormerNet(cfg.model.backbone, cfg.model.get("pretrained", True))
    return LitBinarySeg(net, cfg)
