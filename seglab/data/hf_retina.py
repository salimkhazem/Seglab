"""HuggingFace retinal vessel datasets (DRIVE, STARE, CHASE_DB1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Image as HFImage, ClassLabel
from torch.utils.data import Dataset

from seglab.utils.registry import register_dataset


HF_RETINA_MAP = {
    "drive": "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction",
    "stare": "Zomba/STARE-structured-analysis-of-the-retina",
    "chase": "Zomba/CHASE_DB1-retinal-dataset",
}


@register_dataset("hf_retina")
class HFRetinaDataset(Dataset):
    """Generic wrapper for HF retina datasets."""

    def __init__(
        self,
        name: str,
        split: str,
        transforms: Optional[Callable[[Any, Any], Dict[str, torch.Tensor]]] = None,
        cache_dir: str | Path = "cache/hf",
        image_key: str = "image",
        mask_key: str = "label",
    ) -> None:
        if name in HF_RETINA_MAP:
            hf_name = HF_RETINA_MAP[name]
        else:
            hf_name = name
        self.ds = load_dataset(hf_name, split=split, cache_dir=str(cache_dir))
        self.transforms = transforms
        self.image_key = image_key
        self.mask_key = mask_key
        self._paired = True
        self._pairs: List[Tuple[int, int]] = []

        self._init_pairing()

    def _init_pairing(self) -> None:
        """Handle datasets where inputs and labels are stored as separate rows.

        Some Zomba retina datasets store both input images and label masks in the
        same split with a ClassLabel column (names: ["input", "label"]). In that
        case we build (input_idx, label_idx) pairs by matching filenames.
        """
        # If mask_key exists and is an image, treat as already paired per-row.
        if self.mask_key in self.ds.features:
            feat = self.ds.features[self.mask_key]
            if isinstance(feat, HFImage):
                self._paired = True
                return

        # Special case: label column indicates whether image is input or mask.
        if "label" in self.ds.features and isinstance(self.ds.features["label"], ClassLabel):
            try:
                names = list(getattr(self.ds.features["label"], "names", []))
            except Exception:
                names = []
            if names == ["input", "label"]:
                # Use decode=False to access stable paths for pairing.
                ds_paths = self.ds.cast_column(self.image_key, HFImage(decode=False))
                inputs: Dict[str, int] = {}
                labels: Dict[str, int] = {}
                for i in range(len(ds_paths)):
                    ex = ds_paths[i]
                    kind = int(ex["label"])
                    path = ex[self.image_key]["path"]
                    stem = Path(path).stem
                    if kind == 0:
                        inputs[stem] = i
                    else:
                        labels[stem] = i
                common = sorted(set(inputs) & set(labels), key=lambda s: int(s) if s.isdigit() else s)
                if common:
                    self._pairs = [(inputs[k], labels[k]) for k in common]
                    self._paired = False
                    return

                # Fallback: pair by order (many HF scripts store all inputs first, then all labels).
                input_idx = [i for i in range(len(self.ds)) if int(self.ds[i]["label"]) == 0]
                label_idx = [i for i in range(len(self.ds)) if int(self.ds[i]["label"]) == 1]
                if len(input_idx) == len(label_idx) and len(input_idx) > 0:
                    self._pairs = list(zip(input_idx, label_idx))
                    self._paired = False
                    return

        # Fallback: assume paired per-row (image + mask_key present as bitmap-like).
        self._paired = True

    def __len__(self) -> int:
        return len(self._pairs) if not self._paired else len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._paired:
            sample = self.ds[idx]
            img = sample[self.image_key]
            mask = sample[self.mask_key]
        else:
            img_idx, mask_idx = self._pairs[idx]
            img = self.ds[img_idx][self.image_key]
            mask = self.ds[mask_idx][self.image_key]

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

        # fallback to torch tensors
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).long()
        return {"image": img_t, "mask": mask_t}
