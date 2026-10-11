#!/usr/bin/env bash
# replacements for identities rejected by the generator (62, 81: no valid same_target group): 144 (x400), 145 (x200)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/gaze_dataset_p7_ids
for spec in "144 400" "145 200"; do
  set -- $spec
  ( blender --background --python-exit-code 1 --python scripts/generate_gaze_dataset_v2.py -- --identity $1 --samples $2 --out $OUT > $OUT/logs/identity_$1.log 2>&1 \
    && ~/tgaze/.venv/bin/python phase7/extract_one.py $OUT/identity_$1 4 >> $OUT/logs/identity_$1.log 2>&1 && echo "identity $1 ok" || echo "identity $1 FAILED" ) &
done
wait
echo REPLACE_DONE
