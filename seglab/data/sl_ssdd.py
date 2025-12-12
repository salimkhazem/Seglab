"""SL-SSDD SAR sea/land dataset wrapper.

Expected folder structure:
  root/
    images/
      *.png|jpg
    masks/
      *.png (binary)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seglab.utils.registry import register_dataset


@register_dataset("sl_ssdd")
class SLSSDDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_files: Sequence[str],
        transforms: Optional[Callable[[Any, Any], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.images = [self.root / "images" / f for f in split_files]
        self.masks = [self.root / "masks" / f for f in split_files]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        img_np = np.array(img)
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        if self.transforms is not None:
            out = self.transforms(image=img_np, mask=mask_np)
            return {"image": out["image"], "mask": out["mask"].long()}

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).long()
        return {"image": img_t, "mask": mask_t}

