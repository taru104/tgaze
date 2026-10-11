#!/usr/bin/env bash
# Checkpoint 2: consistency lambda pilot (folds 0,1 x seeds 0,1); selection on VALIDATION identities only
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
i=0
for lam in 0 0.05 0.1 0.25 0.5 1 2 5; do
  $PY phase7/p7train.py --exp consistency/pilot_lam$lam --model m8 --lam $lam --folds 0,1 --seeds 0,1 --device cuda:$((i % 2)) > outputs/experiments/phase7/logs/ckpt2_lam$lam.log 2>&1 &
  i=$((i+1))
done
for lam in 0.5 2; do
  $PY phase7/p7train.py --exp consistency/pilot_pair_lam$lam --model m8 --lam $lam --cons pairwise --folds 0,1 --seeds 0,1 --device cuda:$((i % 2)) > outputs/experiments/phase7/logs/ckpt2_pair_lam$lam.log 2>&1 &
  i=$((i+1))
done
wait
echo "ckpt2 done"
