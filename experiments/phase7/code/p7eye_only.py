"""Does the synthetic CV show the real pilot's EYE_DOMINANT pattern (eye-only movements under-estimated, vertical offset)?
Test predictions of the 5-fold CV (seed 0) split by head rotation magnitude (head_mag, deg):
  eye-dominant-like: head_mag < 2;  head-moving: head_mag > 6.
Per group: linear response pred_cm = offset + gain * target_cm per axis, mean error, and the residual slope against the
eye-in-head angles (EYE_YP_L/R mean: yaw, pitch [deg]).    -> diagnostics/eye_only.json"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D
base = D.load_base(); g = D.Geometry(base["geometry"]); uvc = np.asarray(g.uv_cm)
EXPS = sys.argv[1:] or ["baseline/m8_p7", "diagnostics/m8_eyepca16"]
eye = (base["EYE_YP_L"] + base["EYE_YP_R"]) / 2
out = {}
for e in EXPS:
    idx, P = [], []
    for f in range(5):
        z = np.load(os.path.join(D.OUT_ROOT, e, f"fold{f}_seed0", "pred.npz"))
        idx.append(z["te"]); P.append(z["p_te"])
    idx, P = np.concatenate(idx), np.concatenate(P)
    Pc, Tc = P * uvc, base["UV"][idx] * uvc
    R = Pc - Tc
    hm = base["head_mag"][idx]
    res = {}
    for name, m in (("all", np.ones(len(idx), bool)), ("eye_dominant_like head<2", hm < 2), ("head_moving head>6", hm > 6)):
        r = {"n": int(m.sum()), "err_cm": float(np.hypot(*R[m].T).mean())}
        for i, ax in enumerate("xy"):
            gain, off = np.polyfit(Tc[m, i], Pc[m, i], 1)
            sl = np.polyfit(eye[idx][m, i], R[m, i], 1)[0]
            r[ax] = {"gain": float(gain), "offset_cm": float(off), "mean_resid_cm": float(R[m, i].mean()),
                     "resid_per_deg_eye_" + ("yaw" if i == 0 else "pitch"): float(sl)}
        res[name] = r
        print(e, name, r, flush=True)
    out[e] = res
D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "eye_only.json"))
