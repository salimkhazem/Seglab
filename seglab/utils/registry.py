"""Simple registries for datasets and models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")

DATASET_REGISTRY: Dict[str, Type[Any]] = {}
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def _ensure_registered() -> None:
    # Import side-effect modules to populate registries.
    if not DATASET_REGISTRY:
        try:
            import seglab.data  # noqa: F401
        except Exception:
            pass
    if not MODEL_REGISTRY:
        try:
            import seglab.models  # noqa: F401
        except Exception:
            pass


def register_dataset(name: str) -> Callable[[Type[T]], Type[T]]:
    def decorator(cls: Type[T]) -> Type[T]:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def get_dataset(name: str) -> Type[Any]:
    _ensure_registered()
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def get_model(name: str) -> Callable[..., Any]:
    _ensure_registered()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
