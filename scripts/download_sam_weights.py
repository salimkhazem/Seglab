"""Download SAM checkpoints into checkpoints/."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_SHA256 = {
    # Fill in if you want strict verification.
    "vit_b": None,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="vit_b", choices=SAM_URLS.keys())
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    url = SAM_URLS[args.type]
    out_dir = Path(args.out)
    out_path = out_dir / Path(url).name

    if out_path.exists():
        print(f"{out_path} already exists.")
        return

    print(f"Downloading {args.type} weights...")
    download(url, out_path)
    exp_hash = SAM_SHA256.get(args.type)
    if exp_hash:
        got = sha256(out_path)
        if got != exp_hash:
            print(f"[warn] checksum mismatch: expected {exp_hash}, got {got}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
