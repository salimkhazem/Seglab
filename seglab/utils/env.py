"""Collect environment and git metadata for reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

def load_dotenv(path: str | Path = ".env", override: bool = False) -> Dict[str, str]:
    """Load key=value pairs from a local .env file into os.environ.

    - Ignores empty lines and comments (# ...)
    - Does not override existing env vars by default
    - Avoids any printing/logging (safe for secrets)
    """
    path = Path(path)
    if not path.exists():
        return {}
    loaded: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _run(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return ""


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    import torch

    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    info["git_commit"] = _run("git rev-parse HEAD")
    info["pip_freeze"] = _run("python -m pip freeze")
    return info
