"""Sensitivity of the rich16d-angle M8 (baseline/m8_p7, 5 folds x 3 seeds) to a constant rich16d yaw/pitch offset on
TEST identities, i.e. the real-vs-synthetic offset seen in the Phase 6 pilot (real +12 deg vs synthetic -4 deg per
identity -> about +16 deg). The 9D model (head_angles/m8_none) does not read these inputs, so it is unaffected.
  ~/tgaze/.venv/bin/python phase7/p7rich16d_shift.py -> diagnostics/rich16d_shift.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402

SCALE = np.array([14.53, 9.45])
base = D.load_base()
shifts = [("yaw", 5, d) for d in (-8, 8, 16, 24)] + [("pitch", 4, d) for d in (-8, 8)]
res = {k: [] for k in ["clean"] + [f"{a}{d:+d}" for a, _, d in shifts]}
mv = {f"{a}{d:+d}": [] for a, _, d in shifts}
for rd in sorted(glob.glob(os.path.join(D.OUT_ROOT, "baseline", "m8_p7", "fold*_seed*"))):
    r = I.Runner(rd, base)
    te = np.load(os.path.join(rd, "pred.npz"))["te"]
    lms, f16, uv = base["LMS"][te], base["F16"][te].astype(np.float64), base["UV"][te]
    p0 = r.predict(lms, f16)
    res["clean"].append(float(np.linalg.norm((p0 - uv) * SCALE, axis=1).mean()))
    for a, col, d in shifts:
        f = f16.copy()
        f[:, col] += np.radians(d)
        p = r.predict(lms, f)
        k = f"{a}{d:+d}"
        res[k].append(float(np.linalg.norm((p - uv) * SCALE, axis=1).mean()))
        mv[k].append(((p - p0) * SCALE).mean(0).tolist())
out = {"exp": "baseline/m8_p7 (rich16d angles, 12D)", "runs": len(res["clean"]),
       "test_cm_mean": {k: float(np.mean(v)) for k, v in res.items()},
       "mean_prediction_shift_cm_uv": {k: np.mean(v, 0).tolist() for k, v in mv.items()},
       "note": "u = right on screen, v = up; shift applied to rich16d only (Kabsch inputs unchanged), radians in F16"}
D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "rich16d_shift.json"))
print(json.dumps(out, indent=1))
