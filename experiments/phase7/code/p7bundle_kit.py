"""Bundle a cpu_runtime export (p7export.py + p7onnx_bench.py) into the Phase 6 kit's models_alt/ format
(final_cpu_model.onnx + final_model_preprocessing.json + reference_io.npz), secondary analysis only.
Display / origin / frame keys are copied from an existing kit model (same synthetic geometry).
  python phase7/p7bundle_kit.py CPU_RUNTIME_NAME KIT_DST_DIR "training split note" [EXTRA_JSON]
EXTRA_JSON (optional): keys merged into final_model_preprocessing.json (e.g. training data, seed selection).
"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D
name, dst, note = sys.argv[1], sys.argv[2], sys.argv[3]
extra = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
src = os.path.join(D.OUT_ROOT, "cpu_runtime", name)
kit = os.path.expanduser("~/tgaze/experiments/phase6_kit")
ref = json.load(open(os.path.join(kit, "models_alt", "m8_no_rich16d_angles", "final_model_preprocessing.json")))
p = json.load(open(os.path.join(src, "preprocessing.json")))
out = {k: ref[k] for k in ("branches", "output", "aux_heads_used_at_runtime", "latent", "rigid_idx", "eye_idx", "K_mirrored",
                           "image_size", "display", "origin_mean_cam_m", "uv_scale_cm", "frame") if k in ref}
eyepca = p["eye_pca"] is not None
out.update({"model": "M8 Phase 7 re-implementation" + (" + tgaze eye-patch PCA16 (EyePCA)" if eyepca else " (landmarks only)")
            + ", head_pose angles = rich16d",
            "hidden": p["hidden"], "fuse_hidden": p["fuse_hidden"], "standardization": p["standardization"],
            "template_rigid": p["template_rigid"], "eye_pca": p["eye_pca"],
            "head_angles": "rich16d", "source_run": p["source_run"], "training_split": note,
            "test_metrics_of_run": p["test_metrics_of_run"], "secondary_analysis_only": True,
            "frozen_reference_model_sha256": ref["frozen_reference_model_sha256"]})
if eyepca:
    out["eye_pca_note"] = "eye_norm_r16 input = 168 landmark dims + PCA16 of tgaze appearance.eye_patch (48x32 CLAHE, both eyes, mirrored frame)"
    # appearance OOD reference for reinfer_offline.py: residual fraction of the run's synthetic test patches (as m8_cpu.patch_resid_frac)
    base = D.load_run_base(p["source_run"])
    te = np.load(os.path.join(p["source_run"], "pred.npz"))["te"]
    X = base["PATCH"][te].astype(np.float64) - np.asarray(p["eye_pca"]["mean"])
    comp = np.asarray(p["eye_pca"]["components"])
    R = X - (X @ comp.T) @ comp
    fr = (R ** 2).sum(1) / np.maximum((X ** 2).sum(1), 1e-12)
    ids = json.load(open(os.path.join(p["source_run"], "result.json")))["split"]["test"]
    out["patch_pca_resid_frac_synthetic_test"] = {"ids": ids, "n": int(len(te)),
                                                  **{f"p{q}": float(np.percentile(fr, q)) for q in (5, 50, 95, 99)}}
out.update(extra)
d = os.path.join(kit, dst)
os.makedirs(d, exist_ok=True)
json.dump(out, open(os.path.join(d, "final_model_preprocessing.json"), "w"))
shutil.copy(os.path.join(src, name + ".onnx"), os.path.join(d, "final_cpu_model.onnx"))
z = np.load(os.path.join(src, "reference_io.npz"))
np.savez_compressed(os.path.join(d, "reference_io.npz"), lms=z["lms"], f16=z["f16"], patch=z["patch"].astype(np.float16), y=z["y"])
print("bundled", d)
