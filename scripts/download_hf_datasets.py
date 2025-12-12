"""Pre-download HuggingFace datasets to local cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seglab.utils.env import load_dotenv

DATASETS = [
    "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction",
    "Zomba/STARE-structured-analysis-of-the-retina",
    "Zomba/CHASE_DB1-retinal-dataset",
    "Angelou0516/kvasir-seg",
]


def main() -> None:
    # Helps avoid some 429 issues on certain networks.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    load_dotenv()
    for name in DATASETS:
        print(f"Downloading {name} ...")
        _ = load_dataset(name, cache_dir="cache/hf")
    print("Done.")


if __name__ == "__main__":
    main()
