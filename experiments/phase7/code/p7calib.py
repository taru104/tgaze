"""Phase 7D: synthetic personal calibration on unknown identities (virtual subjects).

  ~/tgaze/.venv/bin/python phase7/p7calib.py baseline/m8_p7 [more experiment dirs ...]

For every run (fold, seed) of an experiment, each TEST identity of that fold is a virtual subject:
  calibration pool = its 'random' singleton samples (single natural-pose samples, like a calibration session);
  for each point of the Phase 6 point sets (real_mac_validation/config.json: 1 = centre, 3 = triangle,
  5 = centre + corners, 9 = 3x3 at +-0.7) we draw 2 samples (repeats_per_point) whose target lies within
  0.15 uv of the point (nearest 2 if fewer); the correction is fitted on those samples' predictions vs their
  true targets with the Phase 6 Correction class (none / bias / affine ridge lambda 25 cm^2 / secondaries).
  evaluation set = all other test samples of the identity minus EVERY group used for calibration at any level
  (same evaluation samples for 0/1/3/5/9 points; no frame and no counterfactual group in both).
Repeated for R random draws per identity. Output: <exp>/calibration.json, and phase7/calibration/summary.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402

sys.path.insert(0, os.path.join(D.PROJECT, "real_mac_validation"))
import common as C6  # noqa: E402

CFG6 = json.load(open(os.path.join(D.PROJECT, "real_mac_validation", "config.json")))
LEVELS = (0, 1, 3, 5, 9)
DRAWS = 10
RADIUS = 0.15


def choose_cal(base, idx_id, rng):
    """Calibration samples for one identity: {n: indices}; indices into the dataset."""
    pool = idx_id[base["gtype"][idx_id] == "random"]
    uv = base["UV"][pool]
    chosen = {}
    used = set()
    for n in LEVELS[1:]:
        pts = np.array(C6.calibration_points(CFG6, n))
        sel = []
        for p in pts:
            dist = np.linalg.norm(uv - p, axis=1)
            cand = [j for j in np.where(dist <= RADIUS)[0] if pool[j] not in used or True]
            if len(cand) >= 2:
                pick = rng.choice(cand, 2, replace=False)
            else:
                pick = np.argsort(dist)[:2]
            sel += [int(pool[j]) for j in pick]
        chosen[n] = np.array(sel)
        used |= set(sel)
    return chosen


def evaluate_run(base, te, p_te, rng, variants=("primary",)):
    """Returns per identity, per level, per variant metrics."""
    pos = {int(i): j for j, i in enumerate(te)}
    out = []
    for k in sorted(set(base["ident"][te])):
        idx_id = te[base["ident"][te] == k]
        for d in range(DRAWS):
            cal = choose_cal(base, idx_id, rng)
            cal_groups = set(base["gid"][np.concatenate(list(cal.values()))])
            ev = idx_id[~np.isin(base["gid"][idx_id], list(cal_groups))]
            p_ev = p_te[[pos[int(i)] for i in ev]]
            rec = {"identity": int(k), "draw": d, "n_eval": int(len(ev))}
            for n in LEVELS:
                for var in variants:
                    spec_all = CFG6["calibration"]["models"][str(n)]
                    if var not in spec_all:
                        continue
                    if n == 0:
                        corr = C6.Correction("none")
                    else:
                        ci = cal[n]
                        spec = spec_all[var]
                        corr = C6.Correction(spec["kind"], spec.get("lambda_cm2", 0.0)).fit(p_te[[pos[int(i)] for i in ci]], base["UV"][ci])
                    pc = corr.apply(p_ev)
                    cm, ang = D.errors(base, ev, pc)
                    pinv = D.pose_invariance(base, ev, pc)
                    rec[f"{n}_{var}"] = {"cm_mean": float(cm.mean()), "cm_median": float(np.median(cm)), "cm_p95": float(np.percentile(cm, 95)),
                                        "cm_p90": float(np.percentile(cm, 90)), "cm_rmse": float(np.sqrt((cm ** 2).mean())),
                                        "deg_mean": float(ang.mean()), "pose_inv_cm": pinv["mean"] if pinv else None}
            out.append(rec)
    return out


def summarize(recs):
    keys = sorted({k for r in recs for k in r if k[0].isdigit()}, key=lambda s: (int(s.split("_")[0]), s))
    summ = {}
    for key in keys:
        vals = [r[key] for r in recs if key in r]
        per_id = {}
        for r in recs:
            if key in r:
                per_id.setdefault(r["identity"], []).append(r[key]["cm_mean"])
        summ[key] = {m: float(np.mean([v[m] for v in vals if v[m] is not None])) for m in vals[0]}
        summ[key]["identity_std_cm"] = float(np.std([np.mean(v) for v in per_id.values()]))
        summ[key]["identities"] = len(per_id)
    return summ


def main():
    base = D.load_base()
    allsum = {}
    variants = ("primary", "secondary_bias_scale", "secondary_bias", "secondary_poly2")
    for exp in sys.argv[1:]:
        edir = os.path.join(D.OUT_ROOT, exp)
        recs = []
        runs = sorted(d for d in os.listdir(edir) if d.startswith("fold") and os.path.exists(os.path.join(edir, d, "pred.npz")))
        for rd in runs:
            z = np.load(os.path.join(edir, rd, "pred.npz"))
            seed = int(rd.split("seed")[1])
            rng = np.random.default_rng([20261011, seed, int(rd[4])])
            for r in evaluate_run(base, z["te"], z["p_te"], rng, variants):
                r["run"] = rd
                recs.append(r)
        s = summarize(recs)
        D.dump_json({"exp": exp, "runs": runs, "draws_per_identity": DRAWS, "radius_uv": RADIUS, "levels": LEVELS,
                     "point_sets": {n: C6.calibration_points(CFG6, n) for n in LEVELS[1:]},
                     "models": CFG6["calibration"]["models"], "summary": s, "records": recs}, os.path.join(edir, "calibration.json"))
        allsum[exp] = s
        print(f"== {exp} ({len(runs)} runs)")
        for key, v in s.items():
            print(f"  {key:28s} mean {v['cm_mean']:.3f}  p95 {v['cm_p95']:.2f}  deg {v['deg_mean']:.2f}  poseInv {v['pose_inv_cm']:.2f}  id-std {v['identity_std_cm']:.2f}")
    prev = os.path.join(D.OUT_ROOT, "calibration", "summary.json")
    old = json.load(open(prev)) if os.path.exists(prev) else {}
    old.update(allsum)
    D.dump_json(old, prev)


if __name__ == "__main__":
    main()
