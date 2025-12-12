"""Deterministic train/val/test splits with caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def make_split_indices(
    n: int, seed: int, val_ratio: float = 0.2, cache_path: str | Path | None = None
) -> Dict[str, List[int]]:
    """Return indices for train/val given n examples."""
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            train_idx = data.get("train")
            val_idx = data.get("val")
            if isinstance(train_idx, list) and isinstance(val_idx, list):
                all_idx = train_idx + val_idx
                if all_idx and max(all_idx) < n:
                    return {"train": train_idx, "val": val_idx}
            # Cache is incompatible (e.g., dataset length changed); fall through to recompute.

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()
    val_n = int(n * val_ratio)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    splits = {"train": train_idx, "val": val_idx}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(splits))
    return splits
