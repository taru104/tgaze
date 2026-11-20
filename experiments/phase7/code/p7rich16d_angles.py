"""rich16d (tgaze 6-point solvePnP) pitch/yaw/roll vs M8's own Kabsch rotation and the Blender ground truth,
on the synthetic training data (Phase 5 gaze_dataset_v2 and Phase 7 gaze_dataset_p7_ids).

All three are expressed with the SAME convention as rich16d: cv2.RQDecomp3x3 of a camera<-head rotation, degrees.
  rich16d : F16[:, 4/5/7] (radians in the feature vector)
  kabsch  : RQDecomp3x3(R_head) with R_head = Kabsch(template -> rigid 40 points), template = the frozen Phase 5 M8 template (same zero as the Mac kit)
  gt      : RQDecomp3x3(H_cam @ R_off^T), R_off = constant rotation fitted so that R_head @ R_off ~ H_cam
            (removes the arbitrary Blender bone-frame offset; leaves the per-sample truth in the Kabsch frame)
  ~/tgaze/.venv/bin/python phase7/p7rich16d_angles.py  -> outputs/experiments/phase7/diagnostics/rich16d_angles.json
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import landmark_features as lf  # noqa: E402

NAMES = ("pitch", "yaw", "roll")


def rq_angles(R):
    """(N,3,3) -> (N,3) degrees, cv2.RQDecomp3x3 + the same pitch folding as tgaze rich16d."""
    a = np.array([cv2.RQDecomp3x3(np.ascontiguousarray(r))[0] for r in R], np.float64)
    p = a[:, 0]
    a[:, 0] = np.where(p > 90, 180 - p, np.where(p < -90, -180 - p, p))
    return a


def fit_offset(Ra, Hb):
    M = np.einsum("nji,njk->ik", Ra, Hb)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    res = y - (a + b * x)
    return {"offset_deg": float(a), "slope": float(b), "corr": float(r), "resid_sd_deg": float(res.std()), "n": int(len(x))}


def dist(v):
    return {"mean": float(v.mean()), "sd": float(v.std()), "p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5)),
            "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}


FROZEN_PREP = os.path.expanduser("~/tgaze/experiments/phase6_kit/model/final_model_preprocessing.json")


def analyse(base, name):
    """Kabsch with the FROZEN M8 template (final_model_preprocessing.json), i.e. the same zero as the Mac kit."""
    g = D.Geometry(base["geometry"])
    NC = lf.to_ncam(base["LMS"], g.K_mir, g.W, g.H)
    tmpl = np.array(json.load(open(FROZEN_PREP))["template_rigid"])
    R = lf.features(NC, tmpl)["R_head"].astype(np.float64)
    H = base["H_cam"].astype(np.float64)
    Roff = fit_offset(R, H)
    A = {"rich16d": np.degrees(base["F16"][:, [4, 5, 7]].astype(np.float64)),
         "kabsch": rq_angles(R), "gt": rq_angles(np.einsum("nij,kj->nik", H, Roff))}
    ident = base["ident"]
    out = {"dataset": name, "n_samples": int(len(R)), "n_identities": int(len(np.unique(ident))), "distributions": {}, "pairs": {}, "per_identity": {},
           "kabsch_vs_rich16d_conditional": {}}
    for k, v in A.items():
        out["distributions"][k] = {n: dist(v[:, i]) for i, n in enumerate(NAMES)}
    for (xk, yk) in (("kabsch", "rich16d"), ("gt", "rich16d"), ("gt", "kabsch")):
        out["pairs"][f"{yk}_on_{xk}"] = {n: ols(A[xk][:, i], A[yk][:, i]) for i, n in enumerate(NAMES)}
        out["pairs"][f"{yk}_minus_{xk}_abs_err_deg"] = {n: float(np.abs(A[yk][:, i] - A[xk][:, i]).mean()) for i, n in enumerate(NAMES)}
    # per identity: rich16d yaw regressed on kabsch yaw (one real subject = one identity)
    for xk in ("kabsch", "gt"):
        rows = []
        for k in np.unique(ident):
            m = ident == k
            rows.append(ols(A[xk][m, 1], A["rich16d"][m, 1]) | {"identity": int(k),
                        "median_rich16d_yaw": float(np.median(A["rich16d"][m, 1])), "median_kabsch_yaw": float(np.median(A["kabsch"][m, 1]))})
        arr = {f: np.array([r[f] for r in rows]) for f in ("offset_deg", "slope", "corr", "median_rich16d_yaw", "median_kabsch_yaw")}
        out["per_identity"][f"yaw_rich16d_on_{xk}"] = {"summary": {f: dist(v) | {"min": float(v.min()), "max": float(v.max())} for f, v in arr.items()},
                                                      "frac_identities_offset_ge_12": float((arr["offset_deg"] >= 12).mean()),
                                                      "frac_identities_slope_ge_2.7": float((arr["slope"] >= 2.7).mean()),
                                                      "identities": rows}
    # where the real pilot falls: kabsch yaw +3..6 deg, rich16d yaw median +14..22 deg
    m = (A["kabsch"][:, 1] >= 3) & (A["kabsch"][:, 1] <= 6)
    ry = A["rich16d"][m, 1]
    out["kabsch_vs_rich16d_conditional"] = {"kabsch_yaw_3_to_6_deg": {"n": int(m.sum()), "rich16d_yaw": dist(ry),
                                            "frac_rich16d_yaw_ge_14": float((ry >= 14).mean()), "frac_rich16d_yaw_ge_22": float((ry >= 22).mean())}}
    return out, A


if __name__ == "__main__":
    res = {}
    b5 = D.load_base()
    res["phase5_v2"], A5 = analyse(b5, "gaze_dataset_v2 (Phase 5 / Phase 7 CV training data)")
    b7 = D.load_base(os.path.join(D.PROJECT, "outputs", "gaze_dataset_p7_ids"))
    res["phase7_ids"], _ = analyse(b7, "gaze_dataset_p7_ids (Phase 7E identity diversity)")
    res["real_pilot_reference"] = {"source": "user's Mac analysis of subject_00 (~1000 frames)", "rich16d_yaw_median_deg": [14, 22],
                                   "kabsch_yaw_median_deg": [3, 6], "rich16d_on_kabsch": {"offset_deg": 12, "slope": 2.7, "corr": 0.78},
                                   "m8_head_pose_standardisation_rich16d_yaw": {"mean_deg": -3.7, "sd_deg": 11.9}}
    out = os.path.join(D.OUT_ROOT, "diagnostics", "rich16d_angles.json")
    D.dump_json(res, out)
    np.savez_compressed(os.path.join(D.OUT_ROOT, "diagnostics", "rich16d_angles_v2.npz"), **A5, ident=b5["ident"])
    for k in ("phase5_v2", "phase7_ids"):
        r = res[k]
        print(k, json.dumps({p: {n: {f: round(v, 3) for f, v in d.items() if f != "n"} for n, d in r["pairs"][p].items()}
                             for p in r["pairs"] if not p.endswith("abs_err_deg")}, ensure_ascii=False))
        print(k, "abs err", {p: r["pairs"][p] for p in r["pairs"] if p.endswith("abs_err_deg")})
        print(k, "dist yaw", {a: {f: round(v, 1) for f, v in r["distributions"][a]["yaw"].items()} for a in r["distributions"]})
        for xk in ("kabsch", "gt"):
            s = r["per_identity"][f"yaw_rich16d_on_{xk}"]
            print(k, "per-id on", xk, {f: (round(v["min"], 2), round(v["p50"], 2), round(v["max"], 2)) for f, v in s["summary"].items()},
                  s["frac_identities_offset_ge_12"], s["frac_identities_slope_ge_2.7"])
        print(k, "cond", r["kabsch_vs_rich16d_conditional"])
