"""Aggregate Phase 7 runs: identity 5-fold CV x seeds.

  ~/tgaze/.venv/bin/python phase7/p7agg.py EXP [EXP ...]        -> prints a table, writes <exp>/summary.json

CV value of a metric = mean over the 5 folds of the per-fold test value (every identity is a test identity
once; same as Phase 5). Reported as mean +- std over seeds (std over folds is also stored).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402

METRICS = {
    "test_mean": ("test", ["all", "cm_mean"]), "test_median": ("test", ["all", "cm_median"]), "test_rmse": ("test", ["all", "cm_rmse"]),
    "test_p50": ("test", ["all", "cm_p50"]), "test_p90": ("test", ["all", "cm_p90"]), "test_p95": ("test", ["all", "cm_p95"]),
    "test_deg": ("test", ["all", "deg_mean"]), "test_pose_inv": ("test", ["pose_invariance_cm", "mean"]),
    "test_pose_dev": ("test", ["pose_invariance_cm", "mean_dev_from_group_mean_cm"]),
    "train_mean": ("train", ["all", "cm_mean"]), "val_mean": ("validation", ["all", "cm_mean"]),
    "val_pose_inv": ("validation", ["pose_invariance_cm", "mean"]),
}
for b in D.HEAD_BINS:
    METRICS[f"head_{b}"] = ("test", ["by_head_angle", b, "cm_mean"])
for r in D.REGIONS:
    METRICS[f"region_{r}"] = ("test", ["by_screen_region", r, "cm_mean"])
for m in ("EYE_ONLY", "HEAD_FOLLOW", "NECK_HEAD_FOLLOW", "COUNTER_ROTATION"):
    METRICS[f"mode_{m}"] = ("test", ["by_pose_mode", m, "cm_mean"])


def get(res, split, path):
    x = res["metrics"][split]
    for p in path:
        if x is None:
            return None
        x = x.get(p)
    return x


def aggregate(exp, folds=(0, 1, 2, 3, 4)):
    edir = os.path.join(D.OUT_ROOT, exp)
    runs = {}
    for d in sorted(os.listdir(edir)):
        f = os.path.join(edir, d, "result.json")
        if d.startswith("fold") and os.path.exists(f):
            r = json.load(open(f))
            runs[(r["fold"], r["seed"])] = r
    seeds = sorted({s for _, s in runs})
    out = {"exp": exp, "seeds_complete": [], "folds": list(folds), "metrics": {}}
    per_seed = {}
    for s in seeds:
        if not all((f, s) in runs for f in folds):
            continue
        out["seeds_complete"].append(s)
        per_seed[s] = {}
        for name, (split, path) in METRICS.items():
            vals = [get(runs[(f, s)], split, path) for f in folds]
            if any(v is None for v in vals):
                continue
            per_seed[s][name] = (float(np.mean(vals)), float(np.std(vals)), vals)
        per_seed[s]["gap"] = (per_seed[s]["test_mean"][0] - per_seed[s]["train_mean"][0], 0, [])
        per_seed[s]["best_epoch"] = (float(np.mean([runs[(f, s)]["best_epoch"] for f in folds])), 0, [])
        per_seed[s]["train_seconds"] = (float(np.mean([runs[(f, s)]["train_seconds"] for f in folds])), 0, [])
    if not per_seed:
        return out
    names = per_seed[next(iter(per_seed))].keys()
    for name in names:
        v = [per_seed[s][name][0] for s in per_seed if name in per_seed[s]]
        out["metrics"][name] = {"mean": float(np.mean(v)), "std_over_seeds": float(np.std(v)), "per_seed": v,
                                "std_over_folds_seed0": per_seed[min(per_seed)][name][1],
                                "per_fold_seed0": per_seed[min(per_seed)][name][2]}
    any_run = next(iter(runs.values()))
    out["params"] = any_run["params"]
    out["training_device"] = any_run["training_device"]
    out["gpu"] = any_run["gpu"]
    out["config"] = any_run["config"]
    D.dump_json(out, os.path.join(edir, "summary.json"))
    return out


def fmt(m, k):
    if k not in m:
        return "    -     "
    return f"{m[k]['mean']:5.3f}±{m[k]['std_over_seeds']:4.2f}"


def main():
    hdr = f"{'experiment':34s} {'seeds':5s} {'test mean':>11s} {'p95':>11s} {'poseInv':>11s} {'deg':>11s} {'train':>11s} {'val':>11s} {'gap':>11s}"
    print(hdr)
    for exp in sys.argv[1:]:
        o = aggregate(exp)
        m = o["metrics"]
        if not m:
            print(f"{exp:34s} incomplete")
            continue
        print(f"{exp:34s} {len(o['seeds_complete']):5d} {fmt(m, 'test_mean')} {fmt(m, 'test_p95')} {fmt(m, 'test_pose_inv')} {fmt(m, 'test_deg')} "
              f"{fmt(m, 'train_mean')} {fmt(m, 'val_mean')} {fmt(m, 'gap')}")


if __name__ == "__main__":
    main()
