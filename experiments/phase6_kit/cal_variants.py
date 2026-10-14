"""Calibration-method comparison on recorded sessions (secondary / exploratory analysis only; the primary
pre-registered correction of config.json is unchanged and is reported as `primary`).

Inputs are trial medians only (no frames):
  live   : the frozen M8 live predictions in subject_*/metadata.jsonl (m8_raw_pred_uv_median), always included
  reinfer: <root>/reinfer/{cal_trials.csv, trials.csv} written by reinfer_offline.py (every re-inferred model), if present

  python cal_variants.py outputs/real_mac_validation        -> <root>/cal_variants/{summary.md, summary.json}

Variants (fitted on the calibration session, NATURAL, per subject; evaluated on every evaluation trial):
  primary            config.json (bias for 1 point, affine lambda 25 for 3/5/9)
  bias / bias_scale / affine l100 / affine l400 / poly2 (9 points)
  bias x-only        horizontal offset only, vertical left uncorrected
  bias x0.5          the bias shrunk by half
Exploratory (uses evaluation trials as extra calibration, scored on the remaining ones, 300 random draws):
  9pt + k trials of EYE_DOMINANT / HEAD_ALLOWED, k = 1/3/5 (what "calibration points with the head kept still" would buy)
Also: per pose instruction, the linear response pred_cm = offset + gain * target_cm on each axis.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT_DIR)
import common as C  # noqa: E402

CONDS = ("NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED")
COLS = CONDS + ("ALL",)


def load_live(root, cfg):
    use = set(cfg["analysis"]["use_status"])
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "subject_*", "metadata.jsonl"))):
        sid = os.path.basename(os.path.dirname(p))
        M = [json.loads(line) for line in open(p)]
        sel = lambda s: [{"target": m["target"]["uv"], "pred": m["m8_raw_pred_uv_median"], "cond": m["pose_instruction"]}  # noqa: E731
                         for m in M if m["session"] == s and m["status"] in use and m.get("m8_raw_pred_uv_median") is not None]
        out[sid] = {"cal": sel("calibration"), "ev": sel("evaluation")}
    return out


def load_reinfer(root):
    d = os.path.join(root, "reinfer")
    if not os.path.exists(os.path.join(d, "cal_trials.csv")):
        return {}
    models = {}
    for r in csv.DictReader(open(os.path.join(d, "cal_trials.csv"))):
        models.setdefault(r["model"], {}).setdefault(r["subject_id"], {"cal": [], "ev": []})["cal"].append(
            {"target": [float(r["target_u"]), float(r["target_v"])], "pred": [float(r["raw_u"]), float(r["raw_v"])], "cond": "NATURAL"})
    for r in csv.DictReader(open(os.path.join(d, "trials.csv"))):
        models.setdefault(r["model"], {}).setdefault(r["subject_id"], {"cal": [], "ev": []})["ev"].append(
            {"target": [float(r["target_u"]), float(r["target_v"])], "pred": [float(r["raw_u"]), float(r["raw_v"])], "cond": r["pose_instruction"]})
    return models


def arr(S, k):
    return np.array([s[k] for s in S], np.float64).reshape(-1, 2)


def errs(c, S):
    e = C.cm_err(c.apply(arr(S, "pred")), arr(S, "target"))
    cond = np.array([s["cond"] for s in S])
    return {k: (float(e[cond == k].mean()) if (cond == k).any() else np.nan) for k in CONDS} | {"ALL": float(e.mean())}


def variants(cfg, cal):
    pairs = [(tuple(s["target"]), tuple(s["pred"])) for s in cal]
    out = {"0pt none": C.Correction("none")}
    for n in (1, 3, 5, 9):
        T, P = C.select_points(pairs, C.calibration_points(cfg, n))
        if not len(T):
            continue
        out[f"{n}pt primary"] = C.fit_correction(cfg, n, pairs)
        kinds = [("bias", 0)] if n == 1 else [("bias", 0), ("bias_scale", 25), ("affine", 100), ("affine", 400)] + ([("poly2", 25)] if n == 9 else [])
        for kind, lam in kinds:
            out[f"{n}pt {kind}" + (f" l{lam}" if lam else "")] = C.Correction(kind, lam).fit(P, T)
        c = C.Correction("bias").fit(P, T)
        c.b = np.array([c.b[0], 0.0])
        out[f"{n}pt bias x-only"] = c
        c = C.Correction("bias").fit(P, T)
        c.b = 0.5 * c.b
        out[f"{n}pt bias x0.5"] = c
    return out


def mixed(cfg, cal, ev, rng, draws=300):
    T9, P9 = C.select_points([(tuple(s["target"]), tuple(s["pred"])) for s in cal], C.calibration_points(cfg, 9))
    out = {}
    for extra in ("EYE_DOMINANT", "HEAD_ALLOWED"):
        pool = [i for i, s in enumerate(ev) if s["cond"] == extra]
        for k in (1, 3, 5):
            if len(pool) <= k:
                continue
            for kind, lam in (("bias", 0), ("affine", 25)):
                acc = []
                for _ in range(draws):
                    pick = set(rng.choice(pool, k, replace=False).tolist())
                    X = [ev[i] for i in pick]
                    c = C.Correction(kind, lam).fit(np.vstack([P9, arr(X, "pred")]), np.vstack([T9, arr(X, "target")]))
                    acc.append(errs(c, [s for i, s in enumerate(ev) if i not in pick]))
                out[f"9pt + {k} {extra} {kind}" + (f" l{lam}" if lam else "")] = {c: float(np.nanmean([a[c] for a in acc])) for c in COLS}
    return out


def response(S):
    out = {}
    for cond in CONDS:
        X = [s for s in S if s["cond"] == cond]
        if len(X) < 3:
            continue
        T, P = C.uv_to_cm(arr(X, "target")), C.uv_to_cm(arr(X, "pred"))
        out[cond] = {ax: {"gain": float(np.polyfit(T[:, i], P[:, i], 1)[0]), "offset_cm": float(np.polyfit(T[:, i], P[:, i], 1)[1])}
                     for i, ax in enumerate("xy")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = C.load_config()
    sources = {"live (frozen M8)": load_live(a.root, cfg)} | {f"reinfer {m}": v for m, v in load_reinfer(a.root).items()}
    out_dir = a.out or os.path.join(a.root, "cal_variants")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    res, lines = {}, ["# Calibration-method comparison (secondary / exploratory; primary = config.json)", "",
                      "Mean error [cm] over evaluation trials, averaged over subjects. Exploratory rows use evaluation trials as extra "
                      "calibration and score the rest.", ""]
    for src, subj in sources.items():
        per = {}
        for sid, d in subj.items():
            if not d["cal"] or not d["ev"]:
                continue
            per[sid] = {"variants": {k: errs(c, d["ev"]) for k, c in variants(cfg, d["cal"]).items()},
                        "mixed_exploratory": mixed(cfg, d["cal"], d["ev"], rng),
                        "response_eval": response(d["ev"]), "response_cal": response(d["cal"]).get("NATURAL")}
        if not per:
            continue
        mean = lambda key: {k: {c: float(np.nanmean([per[s][key][k][c] for s in per if k in per[s][key]])) for c in COLS}  # noqa: E731
                            for k in dict.fromkeys(k for s in per for k in per[s][key])}
        res[src] = {"subjects": list(per), "variants": mean("variants"), "mixed_exploratory": mean("mixed_exploratory"), "per_subject": per}
        lines += [f"## {src} ({len(per)} subjects)", "", "| variant | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS)]
        for key in ("variants", "mixed_exploratory"):
            for k, v in res[src][key].items():
                lines.append(f"| {k} | " + " | ".join(f"{v[c]:.2f}" for c in COLS) + " |")
        lines += ["", "Linear response pred = offset + gain * target (cm), per subject:", ""]
        for sid in per:
            r = per[sid]["response_eval"] | ({"cal NATURAL": per[sid]["response_cal"]} if per[sid]["response_cal"] else {})
            lines += [f"- {sid}: " + "; ".join(f"{k} x gain {v['x']['gain']:.2f} off {v['x']['offset_cm']:+.2f}, y gain {v['y']['gain']:.2f} "
                                              f"off {v['y']['offset_cm']:+.2f}" for k, v in r.items())]
        lines.append("")
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=C.json_default)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
