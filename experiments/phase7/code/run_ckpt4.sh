#!/usr/bin/env bash
# Checkpoint 4: M9 physical-model pilot (folds 0,1 x seed 0); variant choice on VALIDATION identities
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
L=outputs/experiments/phase7/logs
C="--model m9 --folds 0,1 --seeds 0"
$PY phase7/p7train.py --exp m9/pilot_screen_vec3 $C --device cuda:0 > $L/ckpt4_a.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_angular_vec3 $C --m9-loss angular --device cuda:1 > $L/ckpt4_b.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_localonly_vec3 $C --m9-loss local --device cuda:0 > $L/ckpt4_c.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_yawpitch $C --m9-out yawpitch --device cuda:1 > $L/ckpt4_d.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_vec3_nolocal $C --m9-local-w 0 --device cuda:0 > $L/ckpt4_e.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_vec3_learnroff $C --m9-learn-roff 1 --device cuda:1 > $L/ckpt4_f.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_vec3_h128 $C --hidden 128,64,32 --device cuda:0 > $L/ckpt4_g.log 2>&1 &
$PY phase7/p7train.py --exp m9/pilot_screen_vec3_fixedphys $C --m9-phys-lr 0 --device cuda:1 > $L/ckpt4_h.log 2>&1 &
wait
echo "ckpt4 done"
