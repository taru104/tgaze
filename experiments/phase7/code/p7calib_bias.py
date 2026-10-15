"""Synthetic personal calibration (p7calib protocol) when every virtual subject carries a SYSTEMATIC landmark bias,
as a stand-in for the real MediaPipe-vs-render difference found in the Phase 6 pilot.

Scenarios (drawn once per test identity, fixed seed; the eye patch is never changed, it is cropped on the eye corners):
  clean         no bias (= p7calib)
  iris          constant iris shift per eye, x and y ~ N(0, 0.5 px)
  iris+lid      iris as above + both eyelids shifted down by kappa * (downward eye pitch / 10 deg) px, kappa ~ U(0, 1)
                (a gaze-dependent bias: the offset changes with where the eyes point, like the pilot's pose-dependent offset)
  ~/tgaze/.venv/bin/python phase7/p7calib_bias.py EXP [EXP ...]   -> diagnostics/calib_bias.json  (fold*_seed0 runs)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7calib as PC  # noqa: E402
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402

IRIS_L, IRIS_R = np.arange(468, 473), np.arange(473, 478)
LIDS = np.array([246, 161, 160, 159, 158, 157, 173, 466, 388, 387, 386, 385, 384, 398,
                 7, 163, 144, 145, 153, 154, 155, 249, 390, 373, 374, 380, 381, 382])
SCEN = ("clean", "iris", "iris+lid")
VARIANTS = ("primary", "secondary_bias")


def perturb(base, te, scen, seed):
    g = base["geometry"]
    W, H = g["W"], g["H"]
    lms = base["LMS"][te].astype(np.float64).copy()
    if scen == "clean":
        return lms.astype(np.float32)
    pitch = ((base["EYE_YP_L"] + base["EYE_YP_R"]) / 2)[te, 1]
    for k in np.unique(base["ident"][te]):
        m = base["ident"][te] == k
        rng = np.random.default_rng([seed, int(k)])
        for sl in (IRIS_L, IRIS_R):
            sh = rng.normal(0, 0.5, 2)
            lms[np.ix_(m, sl, [0])] += sh[0] / W
            lms[np.ix_(m, sl, [1])] += sh[1] / H
        kappa = rng.uniform(0, 1)
        if scen == "iris+lid":
            dy = kappa * np.clip(-pitch[m], 0, None) / 10.0          # px, down
            lms[np.ix_(m, LIDS, [1])] += (dy / H)[:, None, None]
    return lms.astype(np.float32)


def main():
    base = D.load_base()
    g = base["geometry"]
    out = {}
    for exp in sys.argv[1:]:
        out[exp] = {}
        for scen in SCEN:
            recs = []
            for f in range(5):
                rd = os.path.join(D.OUT_ROOT, exp, f"fold{f}_seed0")
                r = I.Runner(rd, base)
                te = np.load(os.path.join(rd, "pred.npz"))["te"]
                lms = perturb(base, te, scen, 77)
                f16 = base["F16"][te] if scen == "clean" else I.rich16d_batch(lms, g["K_measured_raw"], g["W"], g["H"])[0]
                p = r.predict(lms, f16, patch=base["PATCH"][te])
                rng = np.random.default_rng([20261011, 0, f])
                recs += PC.evaluate_run(base, te, p, rng, VARIANTS)
            s = PC.summarize(recs)
            out[exp][scen] = {k: {m: v[m] for m in ("cm_mean", "cm_p95", "pose_inv_cm", "identity_std_cm")} for k, v in s.items()}
            print(exp, scen, {k: round(v["cm_mean"], 3) for k, v in s.items()}, flush=True)
    D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "calib_bias.json"))


if __name__ == "__main__":
    main()
