"""Sensitivity to a SYSTEMATIC landmark bias (constant shift of a landmark group, both eyes), as opposed to i.i.d. jitter.
Reports the mean prediction shift [cm] and the error change. The eye patch is NOT changed (it is cropped on the eye
corners 33/133 and 362/263, so a MediaPipe iris/eyelid bias leaves it as it is).
  ~/tgaze/.venv/bin/python phase7/p7iris_bias.py [FOLDS] [N_PER_FOLD] [EXP ...]   -> diagnostics/iris_bias.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402

FOLDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
EXPS = sys.argv[3:] or ["baseline/m8_p7", "diagnostics/m8_eyepca16"]
IRIS = np.arange(468, 478)
UPPER_LID = np.array([246, 161, 160, 159, 158, 157, 173, 466, 388, 387, 386, 385, 384, 398])
LOWER_LID = np.array([7, 163, 144, 145, 153, 154, 155, 249, 390, 373, 374, 380, 381, 382])
PERTURB = {"iris +0.5px x": (IRIS, 0.5, 0), "iris +0.5px y(down)": (IRIS, 0, 0.5), "iris +1px y(down)": (IRIS, 0, 1.0),
           "upper lid +0.5px y(down)": (UPPER_LID, 0, 0.5), "both lids +0.5px y(down)": (np.r_[UPPER_LID, LOWER_LID], 0, 0.5)}

base = D.load_base()
g = base["geometry"]
W, H = g["W"], g["H"]
uv_cm = np.asarray(D.Geometry(g).uv_cm)
out = {"folds": FOLDS, "n_per_fold": N, "models": {}}
for e in EXPS:
    acc = {}
    for f in FOLDS:
        r = I.Runner(os.path.join(D.OUT_ROOT, e, f"fold{f}_seed0"), base)
        te = np.where(np.isin(base["ident"], r.result["split"]["test"]) & base["OK"])[0]
        te = np.random.default_rng([3, f]).choice(te, min(N, len(te)), replace=False)

        def run(lms):
            lms = lms.astype(np.float32)
            return r.predict(lms, I.rich16d_batch(lms, g["K_measured_raw"], W, H)[0], patch=base["PATCH"][te])
        p0 = run(base["LMS"][te].copy())
        e0, _ = D.errors(base, te, p0)
        acc.setdefault("clean", {"err_cm": []})["err_cm"].append(float(e0.mean()))
        for name, (idx, dx, dy) in PERTURB.items():
            lms = base["LMS"][te].copy()
            lms[:, idx, 0] += dx / W
            lms[:, idx, 1] += dy / H
            p = run(lms)
            s = (p - p0) * uv_cm
            e1, _ = D.errors(base, te, p)
            d = acc.setdefault(name, {"shift_x_cm": [], "shift_y_cm": [], "abs_shift_cm": [], "err_cm": []})
            d["shift_x_cm"].append(float(s[:, 0].mean()))
            d["shift_y_cm"].append(float(s[:, 1].mean()))
            d["abs_shift_cm"].append(float(np.hypot(*s.T).mean()))
            d["err_cm"].append(float(e1.mean()))
        print(e, f, {k: round(v["err_cm"][-1], 3) for k, v in acc.items()}, flush=True)
    out["models"][e] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "iris_bias.json"))
for e, v in out["models"].items():
    print(e)
    for k, d in v.items():
        print(f"  {k:26s} " + " ".join(f"{m} {x:+.3f}" for m, x in d.items()))
