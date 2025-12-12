"""Mask2Former baseline (HuggingFace transformers).

Historically, Mask2Former baselines are often trained via detectron2. In this
repo we implement a lightweight binary-segmentation wrapper around
`transformers.Mask2FormerForUniversalSegmentation` so it runs without detectron2.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from seglab.models import LitBinarySeg
from seglab.utils.registry import register_model


class Mask2FormerBinaryNet(nn.Module):
    """Binary semantic segmentation using Mask2Former query masks.

    The underlying Mask2Former model predicts a set of masks (queries) and a
    class distribution per query (including a "no-object" class). We aggregate
    query masks into a single foreground probability via a differentiable union:

      p_fg(pixel) = 1 - Π_q (1 - p_q(class=fg) * sigmoid(mask_q(pixel)))

    This yields a proper probability in [0, 1] and allows training with standard
    BCE/Dice losses used by the rest of the pipeline.
    """

    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import Mask2FormerForUniversalSegmentation
        except Exception as e:
            raise ImportError(
                "Mask2Former requires `transformers` with Mask2Former support. "
                "Install seglab with the default dependencies in `pyproject.toml`."
            ) from e

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            backbone,
            num_labels=1,  # single foreground class (+ no-object)
            ignore_mismatched_sizes=True,
        )
        if not pretrained:
            self.model.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        pixel_mask = torch.ones((b, h, w), device=x.device, dtype=torch.long)
        out = self.model(pixel_values=x, pixel_mask=pixel_mask, return_dict=True)

        class_logits = out.class_queries_logits  # (B, Q, 2) when num_labels=1
        mask_logits = out.masks_queries_logits  # (B, Q, Hm, Wm)

        # Foreground prob per query (exclude the last "no-object" class).
        p_fg_q = torch.softmax(class_logits, dim=-1)[..., 0]  # (B, Q)
        mask_prob = torch.sigmoid(mask_logits)  # (B, Q, Hm, Wm)

        # Differentiable union across queries.
        q_prob = p_fg_q.unsqueeze(-1).unsqueeze(-1) * mask_prob
        eps = 1e-6
        q_prob = q_prob.clamp(0.0, 1.0 - eps)
        log_not = torch.sum(torch.log1p(-q_prob), dim=1)  # (B, Hm, Wm)
        p_fg = 1.0 - torch.exp(log_not)
        p_fg = p_fg.clamp(eps, 1.0 - eps)
        logits = torch.log(p_fg) - torch.log1p(-p_fg)
        logits = logits.unsqueeze(1)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


@register_model("mask2former")
def build_mask2former(cfg: Any) -> LitBinarySeg:
    net = Mask2FormerBinaryNet(cfg.model.backbone, cfg.model.get("pretrained", True))
    return LitBinarySeg(net, cfg)
