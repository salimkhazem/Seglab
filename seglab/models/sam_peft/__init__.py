"""SAM parameter-efficient adaptation modules.

Keep this module import-safe (no heavyweight imports at import time) to avoid
circular imports with `seglab.models`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_sam_topolora", "SAMPEFTNet"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module(".model", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
