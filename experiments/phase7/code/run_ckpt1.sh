#!/usr/bin/env bash
# Checkpoint 1: M8 reproduction. (a) unchanged Phase 5 script, same 5 identity folds, aux 0.3, 1 seed
#                                (b) Phase 7 re-implementation (lambda 0), 5 folds x 3 seeds
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/experiments/phase7/baseline/phase5_script_repro
PY=~/tgaze/.venv/bin/python
for f in 0 1 2 3 4; do
  dev=$([ $((f % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1)
  $PY scripts/train_head_eye_models.py outputs/gaze_dataset_v2 --out $OUT/fold$f --seeds 1 --no-lc \
     --models M8_posehead_eye_fusion_headnorm_aux_r16__screen --aux-weight 0.3 --split $OUT/split_fold$f.json --device $dev > $OUT/fold$f.log 2>&1 &
done
for s in 0 1 2; do
  $PY phase7/p7train.py --exp baseline/m8_p7 --model m8 --folds 0,1,2,3,4 --seeds $s --device cuda:$((s % 2)) > outputs/experiments/phase7/logs/ckpt1_m8_p7_s$s.log 2>&1 &
done
wait
$PY scripts/aggregate_identity_cv.py $OUT
echo "ckpt1 done"
