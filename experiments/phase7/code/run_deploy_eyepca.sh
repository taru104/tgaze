#!/bin/bash
# deploy (Phase 5 final split) EyePCA models for offline re-inference on the Mac, 3 seeds, picked by validation
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
SJ=outputs/experiments/phase7/head_angles/split_phase5_final.json
$PY phase7/p7train.py --exp deploy_p5/m8_eyepca16 --model m8 --eye-pca 16 --split-json $SJ --folds 0 --seeds 0,1,2 --device cuda:1
$PY phase7/p7train.py --exp deploy_p5/m8_eyepca16_iris0.5 --model m8 --eye-pca 16 --iris-aug 0.5 --noise-reps 2 --split-json $SJ --folds 0 --seeds 0,1,2 --device cuda:1
echo "[deploy_eyepca] done"
