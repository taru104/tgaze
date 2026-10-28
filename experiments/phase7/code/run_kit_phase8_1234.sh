#!/bin/bash
# Phase 8 v4 1234-identity models -> Phase 6 kit models_alt/ (secondary analysis only, no retraining).
#   m8_phase8_1234          : phase8/v4_m8_1234ids/fold0_seed0        (landmarks only; the only seed)
#   m8_eyepca16_phase8_1234 : phase8/v4_eyepca16_1234ids/fold0_seed0  (landmarks + eye-patch PCA16; the only seed)
cd "$(dirname "$0")/.."
PY=~/tgaze/.venv/bin/python
export P7_CACHE_DIR=/old-home/rtarutani/data/mpfb2/outputs/cache     # v4_1k base cache lives on the data HDD
E=outputs/experiments/phase8
CR=outputs/experiments/phase7/cpu_runtime
set -e
$PY phase7/p7export.py $E/v4_m8_1234ids/fold0_seed0 kit_m8_phase8_1234
$PY phase7/p7export.py $E/v4_eyepca16_1234ids/fold0_seed0 kit_m8_eyepca16_phase8_1234
CUDA_VISIBLE_DEVICES= .venv-tools/bin/python phase7/p7onnx_bench.py kit_m8_phase8_1234 kit_m8_eyepca16_phase8_1234 --tag kit_phase8_1234
for k in v4_m8_1234ids v4_eyepca16_1234ids; do
  $PY - "$k" <<'EOF'
import collections, json, sys
k = sys.argv[1]
E = "outputs/experiments/phase8"
r = json.load(open(f"{E}/{k}/fold0_seed0/result.json"))
sp = r["split"]
ev = json.load(open(f"{E}/eval_v4.json"))[f"{E}/{k}/fold0_seed0"]
v3 = [i for i in sp["train"] if i < 1000]
v4 = [i for i in sp["train"] if i >= 1000]
name = "kit_m8_phase8_1234" if k == "v4_m8_1234ids" else "kit_m8_eyepca16_phase8_1234"
extra = {
    "training_data": {
        "datasets": ["outputs/gaze_dataset_v3_app (Phase 8 appearance data, 200 samples / identity)",
                     "outputs/gaze_dataset_v4_1k (same generator, 100 samples / identity)"],
        "generator": "scripts/generate_gaze_dataset_v3.py (v2 geometry / sampling + per-identity skin, iris, eyebrows, "
                     "eyelashes, eyelid gains & offset, lighting, camera degradation, hair; no glasses, no clothes)",
        "train_ids": sp["train"], "n_train_identities": len(sp["train"]), "n_train_samples": r["n"]["train"],
        "train_composition": {"v3_app": {"ids": f"{min(v3)}-{max(v3)} minus 271 (generation failed), 435, 436 (0-byte landmarks)",
                                         "n_identities": len(v3), "samples_per_identity": 200},
                              "v4_1k": {"ids": f"{min(v4)}-{max(v4)} (failed ids replaced)", "n_identities": len(v4),
                                        "samples_per_identity": 100}},
        "per_identity_weighting": "none: loss = mean over samples, batches = shuffled whole groups (p7train.py); "
                                  "a v3_app identity (200 samples) weighs 2x a v4_1k identity (100 samples); per_id_max not used",
        "validation_ids": sp["validation"], "n_validation_samples": r["n"]["validation"],
        "test_ids": sp["test"], "n_test_samples": r["n"]["test"],
        "split_json": r["config"]["split_json"], "note": sp.get("note"),
        "skin_pool": "train / validation / test A use the training skins; test B (560-599) uses 4 held-out skins only",
        "warning": "no v2 / Phase 7 ids in train: Phase 5 test ids [7, 11] are unseen, but every identity has a different appearance from v2"},
    "seed_selection": {
        "selected_seed": 0, "criterion": "only seed 0 was trained for this condition (no selection); early stopping on validation ids",
        "validation_cm_mean_by_seed": {"0": r["metrics"]["validation"]["all"]["cm_mean"]}},
    "synthetic_test_metrics": {
        "run_test_ids_520_599": {"cm_mean": r["metrics"]["test"]["all"]["cm_mean"], "cm_p95": r["metrics"]["test"]["all"]["cm_p95"]},
        "by_set_phase8_p8eval": ev,
        "sets": {"A": "v3_app 520-559 (training skins; 559 failed -> 39 ids)", "B": "v3_app 560-599 (held-out skins)",
                 "clean": "gaze_dataset_p7_ids 120-139 (uniform appearance, the Phase 7 test set)"},
        "calib_cm_note": "p7calib protocol: 0/1/5/9 points, 2 'random' samples per point, primary correction (affine ridge lambda 25); "
                         "evaluated on samples not used for calibration, so '0' differs slightly from cm_mean"},
}
json.dump(extra, open(f"outputs/experiments/phase7/cpu_runtime/{name}/kit_extra.json", "w"), indent=1)
print(name, "val", extra["seed_selection"]["validation_cm_mean_by_seed"])
EOF
done
NOTE="Phase 8 v4: train 1234 ids (v3_app 234 ids x200 + v4_1k 1000 ids x100), 146,800 samples; validation [500-518] (519 failed); test [520-599] minus 559; single seed 0"
$PY phase7/p7bundle_kit.py kit_m8_phase8_1234 models_alt/m8_phase8_1234 "$NOTE" $CR/kit_m8_phase8_1234/kit_extra.json
$PY phase7/p7bundle_kit.py kit_m8_eyepca16_phase8_1234 models_alt/m8_eyepca16_phase8_1234 "$NOTE" $CR/kit_m8_eyepca16_phase8_1234/kit_extra.json
echo "[kit_phase8] done"
