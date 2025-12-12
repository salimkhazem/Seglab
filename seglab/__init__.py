"""SegLab: reproducible semantic segmentation research codebase."""

from importlib.metadata import version as _version

__all__ = ["__version__"]

try:
    __version__ = _version("seglab")
except Exception:
    __version__ = "0.0.0"

