"""Lightweight LoRA implementation for SAM image encoder FFNs."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = linear.in_features
        out_f = linear.out_features
        self.lora_A = nn.Parameter(torch.zeros((r, in_f)))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # freeze base weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear(x)
        if self.r > 0:
            x_d = self.dropout(x)
            lora = torch.matmul(x_d, self.lora_A.t())
            lora = torch.matmul(lora, self.lora_B.t())
            res = res + lora * self.scaling
        return res


def inject_lora(
    module: nn.Module,
    target_keywords: Iterable[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """Replace Linear layers whose *full* name contains any keyword with LoRALinear."""

    def _rec(mod: nn.Module, prefix: str = "") -> List[str]:
        rep: List[str] = []
        for name, child in list(mod.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in full_name for k in target_keywords):
                setattr(mod, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                rep.append(full_name)
            else:
                rep += _rec(child, full_name)
        return rep

    return _rec(module, "")
