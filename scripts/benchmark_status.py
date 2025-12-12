"""Show progress for the benchmark matrix sweep."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def is_done(results_root: Path, experiment: str, dataset: str, model: str, seed: int) -> bool:
    seed_dir = results_root / experiment / dataset / model / f"seed{seed}"
    if not seed_dir.exists():
        return False
    return any(p.exists() for p in seed_dir.glob("*/best_ckpt.txt"))


def is_skipped_combo(dataset: str, model: str) -> bool:
    if dataset == "sl_ssdd":
        # If not prepared, treat as skipped.
        required = [
            Path("data/sl_ssdd/splits/train.txt"),
            Path("data/sl_ssdd/splits/val.txt"),
            Path("data/sl_ssdd/splits/test.txt"),
        ]
        return not all(p.exists() for p in required)
    if model == "mask2former":
        try:
            from transformers import Mask2FormerForUniversalSegmentation  # noqa: F401
        except Exception:
            return True
        return False
    return False


def main() -> None:
    sweep = OmegaConf.load("configs/experiments/benchmark_matrix.yaml")
    experiment = str(sweep.experiment)
    results_root = Path("results")

    total = 0
    done = 0
    skipped = 0
    missing = []

    for dataset in list(sweep.datasets):
        for model in list(sweep.models):
            for seed in list(sweep.seeds):
                total += 1
                if is_done(results_root, experiment, dataset, model, int(seed)):
                    done += 1
                elif is_skipped_combo(dataset, model):
                    skipped += 1
                else:
                    missing.append((dataset, model, int(seed)))

    effective_total = total - skipped
    print(
        f"{done}/{effective_total} completed "
        f"({100.0 * done / max(effective_total, 1):.1f}%), "
        f"{skipped} skipped (optional/missing deps)."
    )
    if missing:
        print("Missing (first 20):")
        for ds, m, s in missing[:20]:
            print(f"  - {ds} {m} seed{s}")


if __name__ == "__main__":
    main()
