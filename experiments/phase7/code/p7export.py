"""Phase 7H step 1 (tgaze venv, torch): dump a trained run to framework-free files for the CPU runtime.

  ~/tgaze/.venv/bin/python phase7/p7export.py RUN_DIR NAME
-> outputs/experiments/phase7/cpu_runtime/NAME/{weights.npz, preprocessing.json, reference_io.npz}
reference_io holds raw MediaPipe landmarks / rich16d / eye patches of 64 test samples and the torch outputs,
so the ONNX graph AND the NumPy feature code are both checked against training-time behaviour.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402
import landmark_features as lf  # noqa: E402

run_dir, name = sys.argv[1], sys.argv[2]
out = os.path.join(D.OUT_ROOT, "cpu_runtime", name)
os.makedirs(out, exist_ok=True)
base = D.load_base()
r = I.Runner(run_dir, base)
sd = {k: v.numpy() for k, v in r.ck["state_dict"].items()}
np.savez(os.path.join(out, "weights.npz"), **sd)
g = base["geometry"]
prep = {"source_run": run_dir, "model": r.kind, "hidden": r.ck["hidden"], "fuse_hidden": r.ck["fuse_hidden"],
        "standardization": {b: [r.stats[b][0].tolist(), r.stats[b][1].tolist()] for b in r.stats},
        "template_rigid": r.template.tolist(), "rigid_idx": lf.RIGID, "eye_idx": lf.EYE,
        "K_measured_raw": g["K_measured_raw"], "K_mirrored": g["K_mirrored"], "image_size": [g["W"], g["H"]],
        "display": {k: g[k] for k in ("P_C", "P_N", "P_R", "P_U")} | {"width_m": g["disp_w"], "height_m": g["disp_h"]},
        "frame": "mirrored image (cv2.flip(frame, 1)); camera frame = OpenCV with x flipped",
        "eye_pca": None, "m9": None, "test_metrics_of_run": r.result["metrics"]["test"]["all"]}
if r.ck.get("pca") is not None:
    prep["eye_pca"] = {"mean": np.asarray(r.ck["pca"]["mean"]).tolist(), "components": np.asarray(r.ck["pca"]["components"]).tolist()}
if r.kind == "m9":
    init = r.ck["init"]
    prep["m9"] = {"out_kind": r.cfg["m9_out"], "a_is_left": init["a_is_left"],
                  "R_off": np.asarray(init["R_off"]).tolist(), "zf_mu": np.asarray(init["zf_mu"]).tolist(), "zf_sd": np.asarray(init["zf_sd"]).tolist()}
json.dump(prep, open(os.path.join(out, "preprocessing.json"), "w"))
# reference: 64 test samples
te = np.load(os.path.join(run_dir, "pred.npz"))["te"][::50][:64]
xs = r.inputs(base["LMS"][te], base["F16"][te], None, base["PATCH"][te])
import torch  # noqa: E402
with torch.no_grad():
    y, _ = r.net(xs)
np.savez(os.path.join(out, "reference_io.npz"), idx=te, sample_id=base["sample_id"][te], lms=base["LMS"][te], f16=base["F16"][te],
         patch=base["PATCH"][te], uv_true=base["UV"][te], y=y.numpy(), **{"in_" + k: v.numpy() for k, v in xs.items()})
print("exported", out)
