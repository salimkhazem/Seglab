# SegLab — TopoLoRA‑SAM Research Codebase

This repository implements an end‑to‑end, reproducible benchmark for binary semantic segmentation across thin structures (retinal vessels), natural medical domains (polyps), and noisy SAR sea/land segmentation. The main contribution is **TopoLoRA‑SAM**: topology‑aware, parameter‑efficient adaptation of SAM using LoRA + lightweight conv adapters + clDice regularization.

## Setup

Recommended: create a fresh environment and install with **uv**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt
```

Optional extras:

```bash
uv pip install ".[wandb]"     # W&B logging
uv pip install ".[d2]"       # detectron2 (Mask2Former baseline)
```

## Data

### HuggingFace datasets (retina + Kvasir)

Put your HF token in `.env` (recommended) as `HF_TOKEN=...` or `HUGGINGFACE_HUB_TOKEN=...`, then download once into the HF cache:

```bash
export HF_HUB_DISABLE_XET=1   # helps avoid some 429 issues
PYTHONNOUSERSITE=1 PYTHONPATH=. .venv/bin/python scripts/download_hf_datasets.py
```

Datasets used:
- DRIVE: `Zomba/DRIVE-digital-retinal-images-for-vessel-extraction`
- STARE: `Zomba/STARE-structured-analysis-of-the-retina`
- CHASE_DB1: `Zomba/CHASE_DB1-retinal-dataset`
- Kvasir‑SEG: `Angelou0516/kvasir-seg`

### SL‑SSDD (SAR)

SL‑SSDD may be gated. Provide a URL to a zip release and (optionally) a checksum:

```bash
export SL_SSDD_URL="https://…/sl_ssdd.zip"
export SL_SSDD_SHA256="optional_sha256"
python scripts/download_sl_ssdd.py
```

After extraction, you should have:

```
data/sl_ssdd/
  images/
  masks/
  splits/train.txt
  splits/val.txt
  splits/test.txt
```

## SAM weights

Download SAM ViT‑B weights:

```bash
python scripts/download_sam_weights.py --type vit_b
```

This places `sam_vit_b_01ec64.pth` in `checkpoints/`.

## Quickstart (single experiment)

Train U‑Net on DRIVE:

```bash
PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
  .venv/bin/python -m seglab.train --config configs/base.yaml --dataset drive --model unet seed=0 trainer.max_epochs=10
```

Evaluate the best checkpoint on the test set:

```bash
PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
  .venv/bin/python -m seglab.eval --ckpt checkpoints/default/drive/unet/seed0/*.ckpt --config configs/base.yaml --dataset drive --model unet
```

## Reproduce paper

### Benchmark matrix (5 datasets × 5 models × 3 seeds)

```bash
bash scripts/run_all_benchmarks.sh
```

This will safely skip already‑completed runs.

Monitor progress:

```bash
tail -f benchmark.out
PYTHONNOUSERSITE=1 .venv/bin/python scripts/benchmark_status.py
```

### Ablations and cross‑dataset

```bash
PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
  .venv/bin/python -m seglab.sweep --sweep configs/experiments/ablation_topolora.yaml
PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
  .venv/bin/python -m seglab.sweep --sweep configs/experiments/cross_dataset.yaml
```

### Export tables

```bash
.venv/bin/python scripts/export_tables.py
```

Tables are written to `tables/benchmark_mean_std.csv`, `tables/benchmark_formatted.csv`, and TeX variants.

### Figures

Per‑run qualitative grids, PR curves and calibration diagrams can be generated
from predictions and saved metrics. A minimal helper:

```bash
bash scripts/make_all_figures.sh
```

## Outputs

At runtime, the following folders are created:

```
results/<experiment>/<dataset>/<model>/seed<k>/<timestamp>/
  config.yaml
  env.json
  trainable_params.txt
  best_ckpt.txt
  test_metrics.json
  logs/metrics.csv
checkpoints/<experiment>/<dataset>/<model>/seed<k>/*.ckpt
figures/
tables/
```

## Notes

- Mask2Former runs via HuggingFace `transformers` (no detectron2 required).
- All runs are deterministic given the same seed and config; splits are cached in `cache/splits/`.
