"""Which i.i.d. landmark noise sigma [px] reproduces the per-frame prediction jitter measured on the real Mac pilot?

For test samples of a fold, R noisy replicas (= R frames of one fixation, as in an 800 ms / 24-frame trial) are fed to
the model; jitter is computed exactly as Phase 6 analyze.jitter (per-axis SD over frames, mm) and the trial
prediction is the median over the replicas (as analyze.trial_pred).
  ~/tgaze/.venv/bin/python phase7/p7jitter_match.py [FOLDS] [N_PER_FOLD]  -> diagnostics/jitter_match.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402

FOLDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
REPS = 24
SIGMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
EXPS = ("baseline/m8_p7", "robustness_aug/m8_noise0.25", "robustness_aug/m8_noise0.5")
REAL = {"source": "Phase 6 pilot subject_00 results/jitter.csv (frozen M8, raw frames, zero-shot), median over trials",
        "NATURAL": {"jitter_x_mm": 2.56, "jitter_y_mm": 4.01, "jitter_radial_mm": 5.97},
        "EYE_DOMINANT": {"jitter_x_mm": 3.76, "jitter_y_mm": 4.85, "jitter_radial_mm": 6.82},
        "HEAD_ALLOWED": {"jitter_x_mm": 3.53, "jitter_y_mm": 5.43, "jitter_radial_mm": 7.30}}

base = D.load_base()
g = base["geometry"]
W, H = g["W"], g["H"]
uv_cm = np.asarray(D.Geometry(g).uv_cm, np.float64)
out = {"real_pilot": REAL, "reps": REPS, "n_per_fold": N, "folds": FOLDS, "models": {}}
for e in EXPS:
    res = {}
    for f in FOLDS:
        r = I.Runner(os.path.join(D.OUT_ROOT, e, f"fold{f}_seed0"), base)
        te = np.where(np.isin(base["ident"], r.result["split"]["test"]) & base["OK"])[0]
        te = np.random.default_rng([5, f]).choice(te, min(N, len(te)), replace=False)
        for sig in SIGMAS:
            rng = np.random.default_rng([7, f, int(sig * 100)])        # same noise for every model
            lms = np.repeat(base["LMS"][te], REPS, 0)
            if sig:
                lms = lms + rng.normal(0, sig, size=lms.shape) / np.array([W, H, W])
            lms = lms.astype(np.float32)
            f16 = np.repeat(base["F16"][te], REPS, 0) if sig == 0 else I.rich16d_batch(lms, g["K_measured_raw"], W, H)[0]
            p = r.predict(lms, f16, patch=np.repeat(base["PATCH"][te], REPS, 0)).reshape(len(te), REPS, 2)
            mm = p * uv_cm * 10
            sd = mm.std(1, ddof=1)
            med = np.median(p, 1)
            cm_trial, _ = D.errors(base, te, med)
            cm_frame = np.hypot(*((p - base["UV"][te][:, None]) * uv_cm).transpose(2, 0, 1)).mean(1)
            d = res.setdefault(str(sig), {"jitter_x_mm": [], "jitter_y_mm": [], "jitter_radial_mm": [], "trial_median_err_cm": [], "frame_err_cm": []})
            d["jitter_x_mm"].append(float(np.median(sd[:, 0])))
            d["jitter_y_mm"].append(float(np.median(sd[:, 1])))
            d["jitter_radial_mm"].append(float(np.median(np.hypot(sd[:, 0], sd[:, 1]))))
            d["trial_median_err_cm"].append(float(cm_trial.mean()))
            d["frame_err_cm"].append(float(cm_frame.mean()))
        print(e, f, {s: {k: round(v[-1], 2) for k, v in d.items()} for s, d in res.items()}, flush=True)
    out["models"][e] = {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in res.items()}
p = os.path.join(D.OUT_ROOT, "diagnostics", "jitter_match.json")
D.dump_json(out, p)
for e, v in out["models"].items():
    print(e)
    for s, d in v.items():
        print(f"  sigma {s:>4}: jitter x {d['jitter_x_mm']:.2f} y {d['jitter_y_mm']:.2f} rad {d['jitter_radial_mm']:.2f} mm | "
              f"trial-median err {d['trial_median_err_cm']:.3f} cm, per-frame err {d['frame_err_cm']:.3f} cm")
