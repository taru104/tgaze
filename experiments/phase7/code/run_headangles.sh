#!/usr/bin/env bash
# rich16d head-angle ablation for M8 (Phase 6 pilot follow-up): head_pose 9D (no angles) and Kabsch-Euler angles,
# same identity 5-fold CV x 3 seeds as baseline/m8_p7 (which is the rich16d 12D reference).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
L=outputs/experiments/phase7/logs
for v in none kabsch; do
  for s in 0 1 2; do
    $PY phase7/p7train.py --exp head_angles/m8_$v --model m8 --head-angles $v --folds 0,1,2,3,4 --seeds $s \
        --device cuda:$((s % 2)) > $L/headangles_${v}_s$s.log 2>&1 &
  done
done
wait
echo "[headangles] all done"
