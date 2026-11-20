#!/bin/bash
# systematic iris-bias augmentation (sigma 0.5 px per eye, 2 replicas), 5 folds seed 0, then bias sensitivity
set -e
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
$PY phase7/p7train.py --exp iris_aug/m8_eyepca16_iris0.5 --model m8 --eye-pca 16 --iris-aug 0.5 --noise-reps 2 --folds 0,1,2,3,4 --seeds 0 --device cuda:1
$PY phase7/p7train.py --exp iris_aug/m8_iris0.5 --model m8 --iris-aug 0.5 --noise-reps 2 --folds 0,1,2,3,4 --seeds 0 --device cuda:1
$PY phase7/p7iris_bias.py 0,1,2,3,4 400 baseline/m8_p7 diagnostics/m8_eyepca16 iris_aug/m8_iris0.5 iris_aug/m8_eyepca16_iris0.5
