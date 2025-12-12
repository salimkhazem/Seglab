"""Run experiment sweeps from YAML."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf, DictConfig

from seglab.train import run_experiment, _merge_cfg
from seglab.eval import evaluate
from seglab.utils import load_config
from seglab.utils.env import load_dotenv


def _has_completed(
    results_root: str | Path, experiment: str, dataset: str, model: str, seed: int, tag: Optional[str]
) -> bool:
    pattern = Path(results_root) / experiment / dataset / model / f"seed{seed}" / "*"
    for p in glob.glob(str(pattern)):
        if (Path(p) / "best_ckpt.txt").exists():
            if tag is None or tag in p:
                return True
    return False


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="Sweep YAML file.")
    args = parser.parse_args()

    sweep_cfg = OmegaConf.load(args.sweep)
    base_cfg = load_config(sweep_cfg.base_config)

    if "runs" in sweep_cfg:
        runs = sweep_cfg.runs
        for run in runs:
            dataset = run.dataset
            model = run.model
            cfg = _merge_cfg(base_cfg, dataset, model)
            cfg.experiment = sweep_cfg.experiment
            if "overrides" in run:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in run.overrides.items()]))
            seeds = run.get("seeds", sweep_cfg.get("seeds", [cfg.seed]))
            for seed in seeds:
                cfg.seed = seed
                tag = run.get("name")
                if _has_completed(cfg.paths.results_dir, cfg.experiment, dataset, model, seed, tag):
                    print(f"[skip] {dataset} {model} seed{seed} ({tag}) already done")
                    continue
                try:
                    run_dir = run_experiment(cfg, tag=tag)
                    if "eval_datasets" in run:
                        best_ckpt = Path((run_dir / "best_ckpt.txt").read_text().strip())
                        for eval_ds in run.eval_datasets:
                            cfg_eval = _merge_cfg(cfg, eval_ds, model)
                            cfg_eval.dataset.name = eval_ds
                            out_dir = run_dir / "cross_eval" / eval_ds
                            evaluate(cfg_eval, best_ckpt, out_dir)
                except (ImportError, NotImplementedError, FileNotFoundError) as e:
                    print(f"[skip] {dataset} {model}: {e}")
                except Exception as e:
                    print(f"[error] {dataset} {model} seed{seed} ({tag}): {type(e).__name__}: {e}")
    else:
        datasets = sweep_cfg.datasets
        models = sweep_cfg.models
        seeds = sweep_cfg.seeds
        for dataset in datasets:
            for model in models:
                for seed in seeds:
                    cfg = _merge_cfg(base_cfg, dataset, model)
                    cfg.experiment = sweep_cfg.experiment
                    cfg.seed = seed
                    if "overrides" in sweep_cfg:
                        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in sweep_cfg.overrides.items()]))
                    if _has_completed(cfg.paths.results_dir, cfg.experiment, dataset, model, seed, None):
                        print(f"[skip] {dataset} {model} seed{seed} already done")
                        continue
                    try:
                        run_experiment(cfg)
                    except (ImportError, NotImplementedError, FileNotFoundError) as e:
                        print(f"[skip] {dataset} {model}: {e}")
                    except Exception as e:
                        print(f"[error] {dataset} {model} seed{seed}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
