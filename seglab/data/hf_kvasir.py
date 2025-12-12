"""HuggingFace Kvasir-SEG dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from seglab.utils.registry import register_dataset


HF_KVASIR_NAME = "Angelou0516/kvasir-seg"


@register_dataset("hf_kvasir")
class HFKvasirDataset(Dataset):
    def __init__(
        self,
        split: str,
        transforms: Optional[Callable[[Any, Any], Dict[str, torch.Tensor]]] = None,
        cache_dir: str | Path = "cache/hf",
        hf_name: str = HF_KVASIR_NAME,
        image_key: str = "image",
        mask_key: str = "mask",
    ) -> None:
        self.ds = load_dataset(hf_name, split=split, cache_dir=str(cache_dir))
        self.transforms = transforms
        self.image_key = image_key
        self.mask_key = mask_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.ds[idx]
        img = sample[self.image_key]
        mask = sample[self.mask_key]

        if hasattr(img, "convert"):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = np.array(img)

        if hasattr(mask, "convert"):
            mask_np = np.array(mask.convert("L"))
        else:
            mask_np = np.array(mask)

        mask_np = (mask_np > 0).astype(np.uint8)

        if self.transforms is not None:
            out = self.transforms(image=img_np, mask=mask_np)
            return {"image": out["image"], "mask": out["mask"].long()}

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).long()
        return {"image": img_t, "mask": mask_t}

