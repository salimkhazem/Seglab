#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=.

run_one () {
  local gpu="$1"
  local seed="$2"
  local cfg="configs/experiments/cross_dataset_seed${seed}.yaml"
  local out="cross_seed${seed}.out"
  local pidf="cross_seed${seed}.pid"
  echo "[start] cross-dataset seed${seed} on GPU ${gpu} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup .venv/bin/python -m seglab.sweep --sweep "${cfg}" > "${out}" 2>&1 &
  echo $! > "${pidf}"
}

run_one 0 0
run_one 1 1
run_one 2 2

echo "PIDs:"
cat cross_seed0.pid cross_seed1.pid cross_seed2.pid
echo "Logs:"
ls -1 cross_seed*.out

