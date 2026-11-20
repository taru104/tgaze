"""Evaluate the frozen production M8 (experiments/phase6_kit/model, not modified, nothing trained) on MPIIFaceGaze.

Chain per image (landmarks from benchmarks/mpii_m8_extract.py, mirrored frame as in the kit):
  1. m8_cpu.OfflineM8("model", K_person, w, h): rich16d + landmark features with THAT person's camera matrix
     (Camera.mat) -> uv on the M8 training display ("this PC's screen": the MacBook Air display defined in the
     model's final_model_preprocessing.json, in the mirrored camera frame).
  2. uv -> 3D point P on that display (same formula as m8_cpu.M8Runtime.angular_error_deg), mirrored -> raw OpenCV
     camera frame (x -> -x), metres -> mm.
  3. Eye centre E: solvePnP of MediaPipe's metric canonical face model (cm) on the mirrored-frame landmark pixels
     with the mirrored K_person (zero distortion, as the kit), x flipped back to the raw frame; cyclopean point of the two eyeball centres = eye-corner midpoints moved
     EYE_BACK_CM into the head.
  4. Ray E -> P intersected with the person's screen plane (monitorPose.mat: screen -> camera, x right / y down, mm)
     -> predicted point on the person's screen (mm), compared with the annotated gaze target.
  Angular error as REPORT8 / exp64: angle between (face centre fc -> predicted 3D point) and (fc -> annotated gt);
  also reported with E as the origin.
Conditions: 0 points (no calibration) and 16-sample affine correction exactly as exp64 (random 16 valid frames of the
person, 5 seeds RandomState(1000*s+7), Ridge(alpha=1e-3) on [pred_norm, 1] in screen-normalised coordinates,
evaluated on the rest; per person: median of the seed medians for cm / deg, mean of the seed RMSEs).

  ~/tgaze/.venv/bin/python benchmarks/mpii_m8_eval.py MPII_ROOT LMS_DIR CANONICAL_OBJ OUT_JSON [pXX ...]
"""
import json
import os
import sys

import cv2
import numpy as np
import scipy.io as sio
from sklearn.linear_model import Ridge

TGAZE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(TGAZE, "experiments", "phase6_kit")
sys.path.insert(0, KIT)
import m8_cpu  # noqa: E402
import landmark_features as lf  # noqa: E402

EYE_BACK_CM = 1.2                 # eyeball centre behind the eye-corner midpoint (assumption)
PNP_IDX = [i for i in lf.RIGID if i < 468] + [33, 133, 263, 362]
SEEDS, K_CAL = 5, 16


def load_obj(path):
    v = [list(map(float, l.split()[1:4])) for l in open(path) if l.startswith("v ")]
    return np.array(v, np.float64)


def calib(mpii, pid):
    c = sio.loadmat(os.path.join(mpii, pid, "Calibration", "Camera.mat"))
    ss = sio.loadmat(os.path.join(mpii, pid, "Calibration", "screenSize.mat"))
    mp_ = sio.loadmat(os.path.join(mpii, pid, "Calibration", "monitorPose.mat"))
    g = lambda k: float(np.ravel(ss[k])[0])
    R = cv2.Rodrigues(np.ravel(mp_["rvects"] if "rvects" in mp_ else mp_["rvecs"]).astype(np.float64))[0]
    T = np.ravel(mp_["tvecs"]).astype(np.float64)
    scr = (g("width_mm"), g("height_mm"), g("width_pixel" if "width_pixel" in ss else "width_px"), g("height_pixel"))
    return np.array(c["cameraMatrix"], np.float64), np.ravel(c["distCoeffs"]), scr, R, T


class Geometry:
    def __init__(self, prep):
        d = prep["display"]
        self.C, self.Rt, self.U = (np.array(d[k]) for k in ("center_cam_m", "right_cam_m", "up_cam_m"))
        self.half = np.array(prep["uv_scale_cm"]) / 100.0          # [0.1453, 0.0945] m
        self.n = np.array(d["normal_cam_m"])

    def uv_to_raw_mm(self, uv):
        """uv (N,2) -> 3D point on the M8 display, raw OpenCV camera frame, mm (m8_cpu.angular_error_deg formula)."""
        p = self.C + uv[:, :1] * self.half[0] * self.Rt + uv[:, 1:2] * self.half[1] * self.U
        return p * np.array([-1.0, 1.0, 1.0]) * 1000.0

    def raw_mm_to_uv(self, E, Q):
        """Inverse, for the round trip: ray E -> Q (raw mm) intersected with the M8 display -> uv."""
        M = np.array([-1.0, 1.0, 1.0])
        e, q = E * M / 1000.0, Q * M / 1000.0                     # to mirrored frame, m
        d = q - e
        t = ((self.C - e) @ self.n) / (d @ self.n)
        p = e + t[:, None] * d - self.C
        return np.stack([p @ self.Rt / self.half[0], p @ self.U / self.half[1]], 1)


def eye_centre_pnp(lms_mir, w, h, K, model_cm):
    """Cyclopean eyeball centre (raw camera frame, mm) and reprojection RMS (px) per frame."""
    E = np.full((len(lms_mir), 3), np.nan)
    rep = np.full(len(lms_mir), np.nan)
    obj = model_cm[PNP_IDX]
    eL = (model_cm[33] + model_cm[133]) / 2 - [0, 0, EYE_BACK_CM]
    eR = (model_cm[263] + model_cm[362]) / 2 - [0, 0, EYE_BACK_CM]
    cyc = (eL + eR) / 2
    # MediaPipe labels the mirrored image as a real face, so the PnP is solved in the mirrored frame with the mirrored K
    # (a consistent pinhole picture of the mirror-world head), then x is flipped back to the raw camera frame.
    Km = lf.mirrored_K(K, w)
    for i, l in enumerate(lms_mir):
        if not np.isfinite(l).all():
            continue
        img = np.stack([l[PNP_IDX, 0] * w, l[PNP_IDX, 1] * h], 1).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(obj, img, Km, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        ok, rv, tv = cv2.solvePnP(obj, img, Km, None, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(obj, rv, tv, Km, None)
        rep[i] = float(np.sqrt(((pr.reshape(-1, 2) - img) ** 2).sum(1).mean()))
        Rm = cv2.Rodrigues(rv)[0]
        E[i] = (Rm @ cyc + tv.ravel()) * 10.0 * np.array([-1.0, 1.0, 1.0])          # mirrored -> raw, cm -> mm
    return E, rep


def ray_screen(E, P, R, T):
    """Intersect rays E -> P (raw camera mm) with the screen plane {R [x y 0] + T}; returns screen xy (mm), t."""
    n = R[:, 2]
    d = P - E
    t = ((T - E) @ n) / (d @ n)
    Q = E + t[:, None] * d
    s = (Q - T) @ R                                                  # R^T (Q - T)
    return s[:, :2], t, Q


def screen_to_cam(xy_mm, R, T):
    return np.hstack([xy_mm, np.zeros((len(xy_mm), 1))]) @ R.T + T


def ang(o, a, b):
    u, v = a - o, b - o
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1)))


def adapt(pc, yc, pt):
    D = np.hstack([pc, np.ones((len(pc), 1))])
    W = Ridge(alpha=1e-3, fit_intercept=False).fit(D, yc).coef_
    return np.hstack([pt, np.ones((len(pt), 1))]) @ W.T


def main():
    mpii, lms_dir, obj_path, out_path = sys.argv[1:5]
    subjects = sys.argv[5:] or [f"p{i:02d}" for i in range(15)]
    model_cm = load_obj(obj_path)
    import hashlib
    sha = hashlib.sha256(open(os.path.join(KIT, "model", "final_cpu_model.onnx"), "rb").read()).hexdigest()
    assert sha == "db19af0b9521dc2e83d66cfbc324acf62905fbfe6ca820ab7ea286a9dac1c25e", sha
    prep = json.load(open(os.path.join(KIT, "model", "final_model_preprocessing.json")))
    geo = Geometry(prep)
    res, checks = {}, {}
    for pid in subjects:
        z = np.load(os.path.join(lms_dir, f"{pid}.npz"))
        K, dist, scr, R, T = calib(mpii, pid)
        wmm, hmm, wpx, hpx = scr
        w, h = 1280, 720                                     # resolution of Camera.mat; other-size images are excluded
        same_size = (z["img_wh"] == [w, h]).all(1)
        m = m8_cpu.OfflineM8("model", K, TGAZE, w=w, h=h)
        uv, ok = m.predict_lms(z["lms"])
        ok &= z["ok"] & same_size
        E, rep = eye_centre_pnp(z["lms"], w, h, K, model_cm)
        P = geo.uv_to_raw_mm(np.nan_to_num(uv))
        # ray origins: "pnp" = the deployable chain (solvePnP eye centre); "fc_oracle" = MPII's annotated face centre
        # (uses the annotation, reference only: how much of the error is the eye-centre estimate)
        origins = {"pnp": E, "fc_oracle": z["fc"].astype(np.float64)}
        hits = {}
        for name, O in origins.items():
            xy_o, t_o, _ = ray_screen(O, P, R, T)
            hits[name] = xy_o
            ok &= np.isfinite(O).all(1) & np.isfinite(xy_o).all(1) & (t_o > 0)
        xy = hits["pnp"]
        gt_xy = z["gaze_px"] / [wpx, hpx] * [wmm, hmm]
        idx = np.where(ok)[0]
        n_all = len(ok)
        fc, gt = z["fc"][idx], z["gt"][idx]
        yn = gt_xy[idx] / [wmm, hmm]

        def metrics(pred_n, sel):
            pm = pred_n * [wmm, hmm]
            cm = np.linalg.norm(pm - gt_xy[idx][sel], axis=1) / 10.0
            Q3 = screen_to_cam(pm, R, T)
            dg = ang(fc[sel], Q3, gt[sel])
            dgE = ang(E[idx][sel], Q3, gt[sel])
            return cm, dg, dgE

        r = {"n_images": int(n_all), "n_valid": int(len(idx)), "n_no_face": int((~z["ok"]).sum()),
             "n_other_resolution": int((~same_size).sum())}
        for name in origins:
            pn = hits[name][idx] / [wmm, hmm]
            cm0, dg0, dgE0 = metrics(pn, np.arange(len(idx)))
            rv = {"k0": {"cm_median": float(np.median(cm0)), "cm_mean": float(cm0.mean()),
                         "deg_median": float(np.median(dg0)), "deg_rmse": float(np.sqrt((dg0 ** 2).mean())),
                         "deg_rmse_eye_origin": float(np.sqrt((dgE0 ** 2).mean()))}}
            cms, cmm, mds, rms, rmsE = [], [], [], [], []
            for s_ in range(SEEDS):
                rs = np.random.RandomState(1000 * s_ + 7)
                cal = rs.choice(len(idx), K_CAL, replace=False)
                mask = np.ones(len(idx), bool)
                mask[cal] = False
                pr = adapt(pn[cal], yn[cal], pn[mask])
                c, d_, dE = metrics(pr, np.where(mask)[0])
                cms.append(np.median(c)); cmm.append(c.mean()); mds.append(np.median(d_))
                rms.append(np.sqrt((d_ ** 2).mean())); rmsE.append(np.sqrt((dE ** 2).mean()))
            rv["k16"] = {"cm_median": float(np.median(cms)), "cm_mean": float(np.mean(cmm)), "deg_median": float(np.median(mds)),
                         "deg_rmse": float(np.mean(rms)), "deg_rmse_eye_origin": float(np.mean(rmsE))}
            rv["corr_x"] = float(np.corrcoef(pn[:, 0], yn[:, 0])[0, 1])
            rv["corr_y"] = float(np.corrcoef(pn[:, 1], yn[:, 1])[0, 1])
            rv["pred_norm_mean"] = pn.mean(0).tolist()
            r[name] = rv
        # diagnostics: eye centre vs annotated face centre, PnP fit
        r["diag"] = {"true_norm_mean": yn.mean(0).tolist(),
                     "pnp_reproj_px_median": float(np.nanmedian(rep[idx])),
                     "E_minus_fc_mm_median": np.median(E[idx] - fc, 0).tolist(),
                     "E_dist_mm_median": float(np.median(np.linalg.norm(E[idx], axis=1))),
                     "fc_dist_mm_median": float(np.median(np.linalg.norm(fc, axis=1))),
                     "m8_uv_mean": np.nanmean(uv[idx], 0).tolist(),
                     "image_wh": [int(w), int(h)], "K": K.tolist(), "dist_coeffs_not_used": dist.tolist(), "screen_mm_px": list(scr)}
        res[pid] = r
        # round trips (first 5 valid frames)
        j = idx[:5]
        gt_xy_rt = gt_xy[j]
        c1 = np.linalg.norm(screen_to_cam(gt_xy_rt, R, T) - z["gt"][j], axis=1)            # annotation px -> 3D vs gt
        uv_gt = geo.raw_mm_to_uv(E[j], z["gt"][j])                                        # gt through this PC's display
        xy_rt, _, _ = ray_screen(E[j], geo.uv_to_raw_mm(uv_gt), R, T)
        c2 = np.linalg.norm(xy_rt - gt_xy_rt, axis=1)
        checks[pid] = {"px_to_3d_vs_annotated_gt_mm": c1.tolist(), "roundtrip_via_m8_display_mm": c2.tolist(),
                       "frames": [{"image": str(z["image"][i]), "gaze_px": z["gaze_px"][i].tolist(),
                                   "gt_cam_mm": z["gt"][i].tolist(), "E_cam_mm": E[i].tolist(), "fc_cam_mm": z["fc"][i].tolist(),
                                   "uv_of_gt_on_m8_display": uv_gt[k].tolist(), "m8_uv": uv[i].tolist(),
                                   "pred_screen_mm_pnp": xy[i].tolist(), "pred_screen_mm_fc_oracle": hits["fc_oracle"][i].tolist(), "gt_screen_mm": gt_xy[i].tolist()} for k, i in enumerate(j[:3])]}
        q = r["pnp"]
        print(pid, r["n_valid"], "/", n_all, "k0 cm", round(q["k0"]["cm_median"], 2), "deg rmse", round(q["k0"]["deg_rmse"], 2),
              "| k16 cm", round(q["k16"]["cm_median"], 2), "deg rmse", round(q["k16"]["deg_rmse"], 2),
              "| corr", round(q["corr_x"], 2), round(q["corr_y"], 2), "| rt", f"{c1.max():.1e} {c2.max():.1e}", flush=True)
    summ = {}
    for name in ("pnp", "fc_oracle"):
        summ[name] = {}
        for k in ("k0", "k16"):
            d_ = {f"{m_}_mean_over_subjects": float(np.mean([res[p][name][k][m_] for p in res])) for m_ in res[subjects[0]][name][k]}
            d_["cm_median_median_over_subjects_REPORT8_style"] = float(np.median([res[p][name][k]["cm_median"] for p in res]))
            d_["deg_median_median_over_subjects_REPORT8_style"] = float(np.median([res[p][name][k]["deg_median"] for p in res]))
            d_["deg_rmse_std_over_subjects"] = float(np.std([res[p][name][k]["deg_rmse"] for p in res]))
            summ[name][k] = d_
    json.dump({"subjects": res, "summary": summ, "checks": checks,
               "settings": {"eye_back_cm": EYE_BACK_CM, "pnp_idx": PNP_IDX, "seeds": SEEDS, "k_cal": K_CAL,
                            "model_sha256_expected": "db19af0b9521dc2e83d66cfbc324acf62905fbfe6ca820ab7ea286a9dac1c25e"}},
              open(out_path, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
