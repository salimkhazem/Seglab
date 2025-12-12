"""Placeholder downloader for SL-SSDD.

The dataset may be gated. Set environment variables:
  SL_SSDD_URL: download URL (zip)
  SL_SSDD_SHA256: optional checksum
"""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sl_ssdd")
    args = parser.parse_args()

    url = os.environ.get("SL_SSDD_URL")
    if not url:
        raise RuntimeError("Set SL_SSDD_URL env var to the dataset zip URL.")
    exp_sha = os.environ.get("SL_SSDD_SHA256")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "sl_ssdd.zip"
    if not zip_path.exists():
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if exp_sha:
        got = sha256(zip_path)
        if got != exp_sha:
            raise RuntimeError(f"Checksum mismatch: expected {exp_sha}, got {got}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")


if __name__ == "__main__":
    main()

