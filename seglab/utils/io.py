"""Config and filesystem helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path, overrides: Optional[Sequence[str]] = None) -> DictConfig:
    """Load YAML config and apply key=value overrides."""
    cfg = OmegaConf.load(str(path))
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def save_config(cfg: DictConfig, out_dir: str | Path, filename: str = "config.yaml") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)


def make_run_dir(
    results_root: str | Path,
    experiment: str,
    dataset: str,
    model: str,
    seed: int,
    tag: Optional[str] = None,
) -> Path:
    """Create a unique run directory."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = ts if not tag else f"{ts}-{tag}"
    parts = [experiment, dataset, model, f"seed{seed}", run_name]
    run_dir = Path(results_root).joinpath(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def atomic_write_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def copy_code_snapshot(run_dir: str | Path) -> None:
    """Optionally copy a lightweight code snapshot for reproducibility."""
    run_dir = Path(run_dir)
    snap_dir = run_dir / "code_snapshot"
    if snap_dir.exists():
        return
    snap_dir.mkdir()
    # Copy only python/yaml scripts.
    for rel in ["seglab", "configs", "scripts"]:
        src = Path(rel)
        if src.exists():
            shutil.copytree(src, snap_dir / rel, dirs_exist_ok=True)
