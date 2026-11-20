"""Evaluate landmark-noise robustness of runs on given folds (seed 0): clean and Gaussian noise on all landmarks.
usage: ~/tgaze/.venv/bin/python phase7/p7noise_eval.py FOLDS EXP [EXP ...]   -> robustness/noise_eval.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D, p7infer as I
folds = [int(x) for x in sys.argv[1].split(",")]
base = D.load_base(); g = base["geometry"]; W, H = g["W"], g["H"]
rng = np.random.default_rng(11)
out = {}
for e in sys.argv[2:]:
    out[e] = {}
    for f in folds:
        r = I.Runner(os.path.join(D.OUT_ROOT, e, f"fold{f}_seed0"), base)
        te = np.where(np.isin(base["ident"], r.result["split"]["test"]) & base["OK"])[0]
        for sig in (0.0, 0.25, 0.5, 1.0):
            rng = np.random.default_rng([11, f, int(sig * 100)])      # same noise for every model
            lms = base["LMS"][te] + (rng.normal(0, sig, size=base["LMS"][te].shape) / np.array([W, H, W]) if sig else 0)
            f16 = base["F16"][te] if sig == 0 else I.rich16d_batch(lms.astype(np.float32), g["K_measured_raw"], W, H)[0]
            p = r.predict(lms.astype(np.float32), f16, patch=base["PATCH"][te])
            cm, _ = D.errors(base, te, p)
            pi = D.pose_invariance(base, te, p)
            out[e].setdefault(str(sig), []).append({"cm": float(cm.mean()), "inv": pi["mean"]})
    print(e, {s: (round(np.mean([v["cm"] for v in vs]), 3), round(np.mean([v["inv"] for v in vs]), 2)) for s, vs in out[e].items()}, flush=True)
p = os.path.join(D.OUT_ROOT, "robustness", "noise_eval.json")
old = json.load(open(p)) if os.path.exists(p) else {}
old.update({f"{k}|folds{sys.argv[1]}": v for k, v in out.items()})
D.dump_json(old, p)
