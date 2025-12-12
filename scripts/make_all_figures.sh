#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=.

.venv/bin/python scripts/make_all_figures.py
.venv/bin/python scripts/export_tables.py
