"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import torch
from PIL import Image
from tqdm import tqdm

from seglab.metrics.segmentation import compute_segmentation_metrics, boundary_f_score
from seglab.metrics.topology import cldice_score
from seglab.metrics.calibration import expected_calibration_error
from seglab.train import build_dataloaders, _merge_cfg
from seglab.utils import get_model, load_config
from seglab.utils.env import load_dotenv


@torch.no_grad()
def evaluate(cfg: DictConfig, ckpt_path: str | Path, out_dir: str | Path) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_builder = get_model(cfg.model.name)
    lit_module = model_builder(cfg)
    state = torch.load(str(ckpt_path), map_location="cpu")
    lit_module.load_state_dict(state["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit_module.eval().to(device)

    _, _, test_loader = build_dataloaders(cfg)

    all_probs = []
    all_targets = []
    metrics_list = []

    pred_dir = out_dir / "preds"
    pred_dir.mkdir(exist_ok=True)

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        x = batch["image"].to(lit_module.device)
        y = batch["mask"].to(lit_module.device)
        logits = lit_module(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

        m = compute_segmentation_metrics(probs.cpu(), y.cpu())

        # boundary f-score per-image on cpu
        pred_bin = (probs.cpu() > 0.5).numpy().astype(np.uint8)
        tgt_bin = y.cpu().numpy().astype(np.uint8)
        bfs = np.mean([boundary_f_score(p[0], t) for p, t in zip(pred_bin, tgt_bin)])
        m["bfscore"] = float(bfs)

        if cfg.dataset.type == "hf_retina":
            m["cldice"] = cldice_score(pred_bin, tgt_bin)
        metrics_list.append(m)

        # save prediction png
        for b in range(pred_bin.shape[0]):
            Image.fromarray(pred_bin[b, 0] * 255).save(pred_dir / f"{i:05d}_{b}.png")

    probs_cat = torch.cat(all_probs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    ece = expected_calibration_error(probs_cat, targets_cat)

    df = pd.DataFrame(metrics_list)
    summary = df.mean().to_dict()
    summary["ece"] = ece

    df.to_csv(out_dir / "metrics_per_image.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default=None, help="Base config or run config.")
    parser.add_argument("--dataset", default=None, help="Dataset name to evaluate on.")
    parser.add_argument("--model", default=None, help="Model name if config missing.")
    args, overrides = parser.parse_known_args()

    ckpt_path = Path(args.ckpt)

    if args.config:
        base_cfg = load_config(args.config)
        dataset_name = args.dataset or base_cfg.dataset.name
        model_name = args.model or base_cfg.model.name
        cfg = _merge_cfg(base_cfg, dataset_name, model_name)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    else:
        # try to rebuild from checkpoint hyperparameters
        state = torch.load(str(ckpt_path), map_location="cpu")
        cfg_dict = state.get("hyper_parameters", {})
        cfg = OmegaConf.create(cfg_dict)
        if args.dataset:
            cfg.dataset.name = args.dataset
            cfg = _merge_cfg(cfg, args.dataset, cfg.model.name)

    out_dir = ckpt_path.parent.parent / "eval" / cfg.dataset.name
    evaluate(cfg, ckpt_path, out_dir)


if __name__ == "__main__":
    main()
