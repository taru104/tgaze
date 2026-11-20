#!/usr/bin/env bash
# Checkpoint 4b: M9 error decomposition with oracle inputs (diagnostic only) + head residual ablation
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
L=outputs/experiments/phase7/logs
C="--model m9 --folds 0,1 --seeds 0"
$PY phase7/p7train.py --exp m9/diag_oracle_head $C --m9-oracle head --device cuda:0 > $L/ckpt4b_a.log 2>&1 &
$PY phase7/p7train.py --exp m9/diag_oracle_origin $C --m9-oracle origin --device cuda:1 > $L/ckpt4b_b.log 2>&1 &
$PY phase7/p7train.py --exp m9/diag_oracle_head_origin $C --m9-oracle head,origin --device cuda:0 > $L/ckpt4b_c.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_vec3_headres $C --m9-head-res 1 --device cuda:1 > $L/ckpt4b_d.log 2>&1 &
wait
echo "ckpt4b done"
