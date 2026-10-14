"""Sensitivity of M8 to a SYSTEMATIC landmark bias (constant shift of the 10 iris landmarks 468-477, both eyes),
as opposed to i.i.d. jitter: mean prediction shift [cm] per px of shift.  -> diagnostics/iris_bias.json"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D, p7infer as I
base = D.load_base(); g = base["geometry"]; W, H = g["W"], g["H"]
uv_cm = np.asarray(D.Geometry(g).uv_cm)
r = I.Runner(os.path.join(D.OUT_ROOT, "baseline/m8_p7", "fold0_seed0"), base)
te = np.where(np.isin(base["ident"], r.result["split"]["test"]) & base["OK"])[0]
te = np.random.default_rng(3).choice(te, 300, replace=False)
IRIS = np.arange(468, 478)
def run(lms):
    lms = lms.astype(np.float32)
    f16 = I.rich16d_batch(lms, g["K_measured_raw"], W, H)[0]
    return r.predict(lms, f16, patch=base["PATCH"][te])
p0 = run(base["LMS"][te].copy())
out = {}
for name, d in (("iris +0.5px x", (0.5, 0)), ("iris +0.5px y(down)", (0, 0.5)), ("iris +1px y(down)", (0, 1.0))):
    l = base["LMS"][te].copy()
    l[:, IRIS, 0] += d[0] / W; l[:, IRIS, 1] += d[1] / H
    s = (run(l) - p0) * uv_cm
    e0, _ = D.errors(base, te, p0); 
    out[name] = {"mean_shift_cm": s.mean(0).round(3).tolist(), "abs_shift_cm": float(np.hypot(*s.T).mean())}
    print(name, out[name], flush=True)
D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "iris_bias.json"))
