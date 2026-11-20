#!/bin/bash
# Phase 7E all-120-identity models -> Phase 6 kit models_alt/ (secondary analysis only, no retraining).
#   m8_all120          : identity_diversity/Tall_120ids, seed picked by validation cm (ids 140-143) only
#   m8_eyepca16_all120 : identity_diversity/eyepca_Tall_120ids/fold0_seed0 (the only seed)
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
D=outputs/experiments/phase7/identity_diversity
CR=outputs/experiments/phase7/cpu_runtime
SEED=$($PY - <<'EOF'
import json
v = {s: json.load(open(f"outputs/experiments/phase7/identity_diversity/Tall_120ids/fold0_seed{s}/result.json"))["metrics"]["validation"]["all"]["cm_mean"] for s in (0, 1, 2)}
print(min(v, key=v.get))
EOF
)
echo "[kit_all120] Tall_120ids seed by validation: $SEED"
$PY phase7/p7export.py $D/Tall_120ids/fold0_seed$SEED kit_m8_all120
$PY phase7/p7export.py $D/eyepca_Tall_120ids/fold0_seed0 kit_m8_eyepca16_all120
CUDA_VISIBLE_DEVICES= .venv-tools/bin/python phase7/p7onnx_bench.py kit_m8_all120 kit_m8_eyepca16_all120 --tag kit_all120
for k in Tall_120ids eyepca_Tall_120ids; do
  $PY - "$k" <<'EOF'
import json, sys
k = sys.argv[1]
D = "outputs/experiments/phase7/identity_diversity"
seeds = (0, 1, 2) if k == "Tall_120ids" else (0,)
res = {s: json.load(open(f"{D}/{k}/fold0_seed{s}/result.json")) for s in seeds}
val = {s: r["metrics"]["validation"]["all"]["cm_mean"] for s, r in res.items()}
s = min(val, key=val.get)
r = res[s]
sp = r["split"]
name = "kit_m8_all120" if k == "Tall_120ids" else "kit_m8_eyepca16_all120"
extra = {
    "training_data": {
        "datasets": ["outputs/gaze_dataset_v2 (ids 0-19, Phase 5)", "outputs/gaze_dataset_p7_ids (Phase 7E new identities)"],
        "train_ids": sp["train"], "n_train_identities": len(sp["train"]), "n_train_samples": r["n"]["train"],
        "validation_ids": sp["validation"], "n_validation_samples": r["n"]["validation"],
        "test_ids": sp["test"], "n_test_samples": r["n"]["test"],
        "split_json": r["config"]["split_json"], "note": sp.get("note"),
        "warning": "train includes v2 ids 0-19, so Phase 5 test ids [7, 11] are NOT held out for this model"},
    "seed_selection": {
        "selected_seed": s, "criterion": "lowest validation cm_mean (ids 140-143); test and real-device numbers not used",
        "validation_cm_mean_by_seed": {str(a): b for a, b in val.items()},
        **({"note": "only one seed was trained for this condition"} if len(seeds) == 1 else {})},
    "synthetic_test_metrics": {"ids": sp["test"], "cm_mean": r["metrics"]["test"]["all"]["cm_mean"],
                               "cm_p95": r["metrics"]["test"]["all"]["cm_p95"],
                               "pose_invariance_cm": r["metrics"]["test"]["pose_invariance_cm"]["mean"]},
}
json.dump(extra, open(f"outputs/experiments/phase7/cpu_runtime/{name}/kit_extra.json", "w"), indent=1)
print(name, "seed", s, val)
EOF
done
$PY phase7/p7bundle_kit.py kit_m8_all120 models_alt/m8_all120 \
  "Phase 7E all identities: train 120 ids (v2 0-19 + p7_ids 20-119,144,145; 62/81 rejected), 50,000 samples; validation [140-143]; test [120-139]; seed picked by validation cm" \
  $CR/kit_m8_all120/kit_extra.json
$PY phase7/p7bundle_kit.py kit_m8_eyepca16_all120 models_alt/m8_eyepca16_all120 \
  "Phase 7E all identities: train 120 ids (v2 0-19 + p7_ids 20-119,144,145; 62/81 rejected), 50,000 samples; validation [140-143]; test [120-139]; single seed 0" \
  $CR/kit_m8_eyepca16_all120/kit_extra.json
echo "[kit_all120] done"
