#!/usr/bin/env bash
# Phase 7E identity-diversity data (same identity sampler / config as v2, new identity indices).
#   ids  20..69  x 400   train pool A (50 identities)
#   ids  70..119 x 200   train pool B (with A subsampled to 200 -> 100 identities)
#   ids 120..139 x 200   held-out TEST identities (common to every condition)
#   ids 140..143 x 200   VALIDATION identities (early stopping)
# per identity: render (Blender) -> MediaPipe on PNG -> PNG converted to JPEG q92. Resumable (generator is resumable per group).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/gaze_dataset_p7_ids
mkdir -p $OUT/logs
one() {
  k=$1; n=$2
  if [ -f $OUT/identity_$k/landmarks.npz ] && [ -z "$(ls $OUT/identity_$k/images/*.png 2>/dev/null | head -1)" ]; then echo "identity $k cached"; return; fi
  blender --background --python-exit-code 1 --python scripts/generate_gaze_dataset_v2.py -- --identity $k --samples $n --out $OUT > $OUT/logs/identity_$k.log 2>&1 \
    && ~/tgaze/.venv/bin/python phase7/extract_one.py $OUT/identity_$k 2 >> $OUT/logs/identity_$k.log 2>&1 \
    && echo "identity $k ok" || echo "identity $k FAILED"
}
export -f one; export OUT
{ for k in $(seq 120 143); do echo "$k 200"; done; for k in $(seq 20 69); do echo "$k 400"; done; for k in $(seq 70 119); do echo "$k 200"; done; } \
  | xargs -P 6 -L 1 bash -c 'one $0 $1'
echo IDGEN_DONE
