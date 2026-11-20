#!/usr/bin/env bash
# Phase 7E: identity diversity at fixed total sample count. Test = 20 held-out new identities (120-139),
# validation = 140-143 (early stopping). M8 recipe unchanged; 3 seeds (identity subsets fixed, group subsets per seed).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
D=outputs/experiments/phase7/identity_diversity
L=outputs/experiments/phase7/logs
i=0
for sj in $D/split_*.json; do
  k=$(basename $sj .json); k=${k#split_}
  per=$($PY -c "import json;v=json.load(open('$sj'))['per_id_max'];print(v or '')")
  extra=""; [ -n "$per" ] && extra="--per-id-max $per"
  for s in 0 1 2; do
    $PY phase7/p7train.py --exp identity_diversity/$k --model m8 --root outputs/gaze_dataset_p7_ids --split-json $sj $extra --subset-seed $s \
        --folds 0 --seeds $s --device cuda:$((i % 2)) > $L/idexp_${k}_s$s.log 2>&1 &
    i=$((i+1))
    if [ $((i % 8)) -eq 0 ]; then wait; fi
  done
done
for k in T20000_20x1000 T20000_50x400 T20000_100x200; do
  sj=$D/split_$k.json
  per=$($PY -c "import json;v=json.load(open('$sj'))['per_id_max'];print(v or '')")
  extra=""; [ -n "$per" ] && extra="--per-id-max $per"
  $PY phase7/p7train.py --exp identity_diversity/eyepca_$k --model m8 --eye-pca 16 --root outputs/gaze_dataset_p7_ids --split-json $sj $extra \
      --folds 0 --seeds 0 --device cuda:$((i % 2)) > $L/idexp_eyepca_${k}.log 2>&1 &
  i=$((i+1))
done
wait
echo "idexp done"
