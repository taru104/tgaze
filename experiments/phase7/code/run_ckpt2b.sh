#!/usr/bin/env bash
# Checkpoint 2b: scale-invariant consistency (folds 0,1 x seeds 0,1)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
i=0
for lam in 0.25 0.5 1 2 4; do
  $PY phase7/p7train.py --exp consistency/pilot_scaled_lam$lam --model m8 --lam $lam --cons scaled --folds 0,1 --seeds 0,1 --device cuda:$((i % 2)) > outputs/experiments/phase7/logs/ckpt2b_lam$lam.log 2>&1 &
  i=$((i+1))
done
wait
echo "ckpt2b done"
