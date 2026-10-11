#!/usr/bin/env bash
# 7G capacity: M8 small / medium / large (+ large without dropout), 5 folds x seed 0
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
L=outputs/experiments/phase7/logs
$PY phase7/p7train.py --exp capacity/m8_medium --model m8 --hidden 128,64,32 --fuse-hidden 128,64,32 --folds 0,1,2,3,4 --seeds 0 --device cuda:0 > $L/cap_medium.log 2>&1 &
$PY phase7/p7train.py --exp capacity/m8_large --model m8 --hidden 256,128,64 --fuse-hidden 256,128,64 --folds 0,1,2,3,4 --seeds 0 --device cuda:1 > $L/cap_large.log 2>&1 &
$PY phase7/p7train.py --exp capacity/m8_large_nodrop --model m8 --hidden 256,128,64 --fuse-hidden 256,128,64 --dropout 0 --wd 0 --patience 80 --folds 0,1,2,3,4 --seeds 0 --device cuda:0 > $L/cap_large_nodrop.log 2>&1 &
$PY phase7/p7train.py --exp capacity/m8_large_reg --model m8 --hidden 256,128,64 --fuse-hidden 256,128,64 --dropout 0.3 --wd 1e-2 --folds 0,1,2,3,4 --seeds 0 --device cuda:1 > $L/cap_large_reg.log 2>&1 &
wait
echo "capacity done"
