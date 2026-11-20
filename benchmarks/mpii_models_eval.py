"""Evaluate the production M8 and every alternative in experiments/phase6_kit/models_alt on MPIIFaceGaze, with the
metric of the gaze-estimation literature (nothing trained; models are only read).

Same geometry chain as benchmarks/mpii_m8_eval.py (imported from there): landmarks (mirrored frame) -> model with
the person's K -> uv on the model's display -> 3D -> ray from the solvePnP eye centre -> the person's screen.
All models are scored on the SAME frames (face found, eye patch present, 1280x720, ray hits the screen for every model).

Metrics per person, then the mean over the 15 persons (the usual leave-one-person-out reporting):
  deg_mean    mean angular error, gaze = (face centre fc -> predicted 3D point) vs (fc -> annotated gt)  <- papers
  deg_median, deg_rmse (REPORT8 / EMC-Gaze), cm_median, cm_mean (on the person's screen)
Conditions: k = 0 (no calibration), and k = 9 / 16 random frames of the person for an affine correction
(exp64 protocol: 5 seeds RandomState(1000*s+7), Ridge(alpha=1e-3) on [pred_norm, 1], evaluated on the other frames).

  python benchmarks/mpii_models_eval.py MPII_ROOT LMS_PATCH_DIR CANONICAL_OBJ OUT_JSON [pXX ...]
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpii_m8_eval as E8  # noqa: E402

KIT = E8.KIT
KS = (0, 9, 16)
FROZEN_SHA = "db19af0b9521dc2e83d66cfbc324acf62905fbfe6ca820ab7ea286a9dac1c25e"


def model_dirs():
    out = ["model"] + sorted(os.path.relpath(d, KIT) for d in glob.glob(os.path.join(KIT, "models_alt", "*"))
                             if os.path.exists(os.path.join(d, "final_cpu_model.onnx")))
    return out


def per_person(pn, yn, gt_xy, fc, gt, scr, R, T):
    wmm, hmm = scr[0], scr[1]

    def score(pred_n, sel):
        pm = pred_n * [wmm, hmm]
        cm = np.linalg.norm(pm - gt_xy[sel], axis=1) / 10.0
        dg = E8.ang(fc[sel], E8.screen_to_cam(pm, R, T), gt[sel])
        return cm, dg

    out = {}
    for k in KS:
        runs = []
        for s in range(5 if k else 1):
            if k:
                cal = np.random.RandomState(1000 * s + 7).choice(len(pn), k, replace=False)
                mask = np.ones(len(pn), bool)
                mask[cal] = False
                pr, sel = E8.adapt(pn[cal], yn[cal], pn[mask]), np.where(mask)[0]
            else:
                pr, sel = pn, np.arange(len(pn))
            cm, dg = score(pr, sel)
            runs.append({"deg_mean": dg.mean(), "deg_median": np.median(dg), "deg_rmse": np.sqrt((dg ** 2).mean()),
                         "cm_median": np.median(cm), "cm_mean": cm.mean()})
        # exp64: medians over seeds for the medians, means for the means / RMSE
        out[f"k{k}"] = {m: float((np.median if "median" in m else np.mean)([r[m] for r in runs])) for m in runs[0]}
    return out


def main():
    mpii, lms_dir, obj_path, out_path = sys.argv[1:5]
    subjects = sys.argv[5:] or [f"p{i:02d}" for i in range(15)]
    sha = hashlib.sha256(open(os.path.join(KIT, "model", "final_cpu_model.onnx"), "rb").read()).hexdigest()
    assert sha == FROZEN_SHA, sha
    model_cm = E8.load_obj(obj_path)
    names = model_dirs()
    preps = {n: json.load(open(os.path.join(KIT, n, "final_model_preprocessing.json"))) for n in names}
    res = {n: {} for n in names}
    counts = {}
    for pid in subjects:
        z = np.load(os.path.join(lms_dir, f"{pid}.npz"))
        K, _, scr, R, T = E8.calib(mpii, pid)
        w, h = 1280, 720
        ok = z["ok"] & (z["img_wh"] == [w, h]).all(1) & np.isfinite(z["eye_patch"]).all(1)
        Eye, _ = E8.eye_centre_pnp(z["lms"], w, h, K, model_cm)
        ok &= np.isfinite(Eye).all(1)
        patch = z["eye_patch"].astype(np.float64)
        hits = {}
        for n in names:
            m = E8.m8_cpu.OfflineM8(n, K, E8.TGAZE, w=w, h=h)
            uv, okm = m.predict_lms(z["lms"], patch if m.eye_pca is not None else None)
            P = E8.Geometry(preps[n]).uv_to_raw_mm(np.nan_to_num(uv))
            xy, t, _ = E8.ray_screen(Eye, P, R, T)
            ok &= okm & np.isfinite(xy).all(1) & (t > 0)
            hits[n] = xy
        idx = np.where(ok)[0]
        counts[pid] = {"n_images": int(len(ok)), "n_common_valid": int(len(idx))}
        gt_xy = (z["gaze_px"] / [scr[2], scr[3]] * [scr[0], scr[1]])[idx]
        yn = gt_xy / [scr[0], scr[1]]
        for n in names:
            res[n][pid] = per_person(hits[n][idx] / [scr[0], scr[1]], yn, gt_xy, z["fc"][idx], z["gt"][idx], scr, R, T)
        print(pid, counts[pid], " ".join(f"{n.split('/')[-1]}={res[n][pid]['k0']['deg_mean']:.2f}/{res[n][pid]['k9']['deg_mean']:.2f}"
                                         for n in names), flush=True)
    summ = {}
    for n in names:
        summ[n] = {}
        for k in KS:
            vals = {m: [res[n][p][f"k{k}"][m] for p in res[n]] for m in res[n][subjects[0]][f"k{k}"]}
            summ[n][f"k{k}"] = {**{f"{m}_mean_over_subjects": float(np.mean(v)) for m, v in vals.items()},
                                "deg_mean_std_over_subjects": float(np.std(vals["deg_mean"])),
                                "cm_median_median_over_subjects_REPORT8_style": float(np.median(vals["cm_median"]))}
        summ[n]["source_run"] = preps[n].get("source_run", "Phase 5 frozen M8")
        summ[n]["training_split"] = preps[n].get("training_split")
    json.dump({"summary": summ, "subjects": res, "counts": counts,
               "settings": {"ks": KS, "seeds": 5, "eye_back_cm": E8.EYE_BACK_CM, "frozen_sha256": FROZEN_SHA}},
              open(out_path, "w"), indent=1)
    for n in names:
        s = summ[n]
        print(f"{n:45s} " + " | ".join(f"k{k}: {s[f'k{k}']['deg_mean_mean_over_subjects']:.2f}° mean, "
                                       f"{s[f'k{k}']['deg_rmse_mean_over_subjects']:.2f}° rmse, {s[f'k{k}']['cm_median_mean_over_subjects']:.2f} cm"
                                       for k in KS))


if __name__ == "__main__":
    main()
