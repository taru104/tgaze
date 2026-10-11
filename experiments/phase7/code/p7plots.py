"""Phase 7 plots -> outputs/experiments/phase7/plots/*.png (run after the experiments; missing inputs are skipped)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = D.OUT_ROOT
P = os.path.join(R, "plots")
os.makedirs(P, exist_ok=True)
MODELS = [("B0 mean", "baseline/B0_mean"), ("B1 rich16d Huber", "baseline/B1_rich16d_huber"), ("B2 +PCA16 Huber", "baseline/B2_rich16d_pca16_huber"),
          ("M8", "baseline/m8_p7"), ("M8+Cons", "consistency/m8cons_scaled0.25"), ("M8+Cons(md0.5)", "consistency/m8cons_meandev0.5"),
          ("M9", "m9/m9_cv"), ("M9+Cons", "m9_consistency/m9cons_scaled0.25"), ("M9+Cons(md0.5)", "m9_consistency/m9cons_meandev0.5"),
          ("M8+NoiseAug0.25", "robustness_aug/m8_noise0.25"), ("M8+EyePCA*", "diagnostics/m8_eyepca16"), ("M8+EyePCA+Cons*", "diagnostics/m8_eyepca16_cons_scaled0.25")]


def summ(exp):
    f = os.path.join(R, exp, "summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def model_comparison():
    rows = [(lab, summ(e)) for lab, e in MODELS]
    rows = [(lab, s) for lab, s in rows if s and s["metrics"]]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, key, title in zip(ax, ("test_mean", "test_p95", "test_pose_inv"), ("unknown-identity mean error [cm]", "P95 [cm]", "pose invariance [cm]")):
        v = [s["metrics"][key]["mean"] for _, s in rows]
        e = [s["metrics"][key]["std_over_seeds"] for _, s in rows]
        a.barh([l for l, _ in rows], v, xerr=e, color=["#999"] * 3 + ["tab:blue"] * (len(rows) - 3), capsize=2)
        for i, x in enumerate(v):
            a.text(x, i, f" {x:.2f}", va="center", fontsize=8)
        a.set_title(title)
        a.invert_yaxis()
    fig.suptitle("Phase 7 identity 5-fold CV (mean ± std over seeds; * = uses eye-patch appearance, synthetic appearance caveat)")
    plt.tight_layout()
    plt.savefig(os.path.join(P, "model_comparison.png"), dpi=110)
    plt.close()


def pose_tradeoff():
    pts = []
    for d in sorted(glob.glob(os.path.join(R, "consistency", "pilot_*"))):
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(d, "fold*", "result.json"))]
        if len(rs) < 4:
            continue
        nm = os.path.basename(d).replace("pilot_", "")
        val = np.mean([r["metrics"]["validation"]["all"]["cm_mean"] for r in rs])
        inv = np.mean([r["metrics"]["test"]["pose_invariance_cm"]["mean"] for r in rs])
        te = np.mean([r["metrics"]["test"]["all"]["cm_mean"] for r in rs])
        sp = [r["metrics"]["test"]["pose_invariance_cm"].get("pred_spread_ratio") for r in rs]
        pts.append((nm, inv, te, val, np.mean(sp) if None not in sp else np.nan))
    if not pts:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for nm, inv, te, val, sp in pts:
        col = "tab:red" if nm.startswith("scaled") else ("tab:green" if nm.startswith("pair") else "tab:blue")
        ax[0].scatter(inv, te, color=col)
        ax[0].annotate(nm, (inv, te), fontsize=7)
    xs = sorted((p[1], p[2]) for p in pts)
    front, best = [], 1e9
    for x, y in xs:
        if y < best:
            front.append((x, y))
            best = y
    ax[0].plot(*zip(*front), "k--", lw=1, label="Pareto frontier")
    ax[0].set_xlabel("pose-invariance error [cm] (same target, max pairwise)")
    ax[0].set_ylabel("unknown-identity mean error [cm]")
    ax[0].set_title("consistency lambda pilot (folds 0,1 x 2 seeds)\nblue mean-dev, red scale-invariant, green pairwise")
    ax[0].axvline(2.0, color="gray", ls=":")
    ax[0].legend()
    ax[0].set_ylim(1.5, 7)
    # full-CV models: raw vs after 9-pt calibration (undoes output shrinkage)
    for lab, e in MODELS[3:]:
        c = os.path.join(R, e, "calibration.json")
        s = summ(e)
        if not (s and s["metrics"] and os.path.exists(c)):
            continue
        cs = json.load(open(c))["summary"]
        ax[1].scatter(s["metrics"]["test_pose_inv"]["mean"], s["metrics"]["test_mean"]["mean"], marker="o")
        ax[1].scatter(cs["9_primary"]["pose_inv_cm"], cs["9_primary"]["cm_mean"], marker="x")
        ax[1].annotate(lab, (s["metrics"]["test_pose_inv"]["mean"], s["metrics"]["test_mean"]["mean"]), fontsize=7)
        ax[1].annotate(lab + " (9pt)", (cs["9_primary"]["pose_inv_cm"], cs["9_primary"]["cm_mean"]), fontsize=7, color="gray")
    ax[1].set_xlabel("pose-invariance error [cm]")
    ax[1].set_ylabel("mean error [cm]")
    ax[1].set_title("full CV: zero-shot (o) vs after 9-point calibration (x)")
    ax[1].axvline(2.0, color="gray", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(P, "pose_invariance.png"), dpi=110)
    plt.close()


def calibration_curve():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for lab, e in MODELS[1:]:
        c = os.path.join(R, e, "calibration.json")
        if not os.path.exists(c):
            continue
        s = json.load(open(c))["summary"]
        lv = [0, 1, 3, 5, 9]
        ax[0].plot(lv, [s[f"{n}_primary"]["cm_mean"] for n in lv], marker="o", label=lab)
        ax[1].plot(lv, [s[f"{n}_primary"]["cm_p95"] for n in lv], marker="o", label=lab)
    for a, t in zip(ax, ("mean error [cm]", "P95 (per-identity) [cm]")):
        a.set_xlabel("calibration points (0 / 1 bias / 3, 5, 9 ridge affine)")
        a.set_ylabel(t)
        a.set_xticks([0, 1, 3, 5, 9])
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("synthetic personal calibration on unknown identities (Phase 6 point sets, 2 samples/point, 10 draws)")
    plt.tight_layout()
    plt.savefig(os.path.join(P, "calibration_curve.png"), dpi=110)
    plt.close()


def learning_curve():
    xs, rows = [], []
    for n in (1000, 2500, 5000, 10000, None):
        s = summ(f"learning_curve/m8_n{n}") if n else summ("baseline/m8_p7")
        if not (s and s["metrics"]):
            continue
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(R, f"learning_curve/m8_n{n}" if n else "baseline/m8_p7", "fold*_seed0", "result.json"))]
        xs.append(np.mean([r["n"]["train"] for r in rs]))
        rows.append(s["metrics"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, lab in (("train_mean", "train identities"), ("val_mean", "validation identities"), ("test_mean", "test identities (unknown)")):
        ax.errorbar(xs, [m[k]["mean"] for m in rows], yerr=[m[k]["std_over_seeds"] for m in rows], marker="o", capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("training samples (14 identities, whole groups subsampled)")
    ax.set_ylabel("screen error [cm]")
    ax.set_title("M8 learning curve, identity 5-fold CV")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(P, "learning_curve.png"), dpi=110)
    plt.close()


def train_val_curves():
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    for col, (lab, e) in enumerate([("M8", "baseline/m8_p7"), ("M9", "m9/m9_cv"), ("M8+EyePCA*", "diagnostics/m8_eyepca16")]):
        f = os.path.join(R, e, "fold0_seed0", "curves.json")
        if not os.path.exists(f):
            continue
        h = json.load(open(f))
        ax[0, col].plot(h["train_loss"], label="train")
        ax[0, col].plot(h["val_loss"], label="validation")
        ax[0, col].set_yscale("log")
        ax[0, col].set_title(f"{lab}: loss (fold 0, seed 0)")
        ax[1, col].plot(h["train_cm"], label="train")
        ax[1, col].plot(h["val_cm"], label="validation")
        ax[1, col].set_title(f"{lab}: screen error [cm]")
        for a in ax[:, col]:
            a.legend()
            a.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(P, "train_val_curves.png"), dpi=110)
    plt.close()


def robustness():
    f = os.path.join(R, "robustness", "landmark_tests.json")
    if os.path.exists(f):
        d = json.load(open(f))
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for e, r in d.items():
            ks = r["K_feature_only"]
            fs = [-5, -3, -1, 0, 1, 3, 5]
            ax[0].plot(fs, [ks["nominal" if p == 0 else f"f{'+' if p > 0 else '-'}{abs(p)}%"]["cm_mean"] for p in fs], marker="o", label=e)
            cs = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
            ax[1].plot(cs, [ks["nominal" if p == 0 else f"cx{'+' if p > 0 else '-'}{abs(p)}px"]["cm_mean"] for p in cs], marker="o", label=e + " cx")
            ax[1].plot(cs, [ks["nominal" if p == 0 else f"cy{'+' if p > 0 else '-'}{abs(p)}px"]["cm_mean"] for p in cs], marker="s", ls="--", label=e + " cy")
        ax[0].set_xlabel("fx, fy error used in feature extraction [%]")
        ax[1].set_xlabel("principal point error used in feature extraction [px]")
        for a in ax:
            a.set_ylabel("mean error [cm] (all 20 identities as unknown)")
            a.legend(fontsize=6)
            a.grid(alpha=0.3)
        fig.suptitle("K mismatch in feature extraction only (images unchanged)")
        plt.tight_layout()
        plt.savefig(os.path.join(P, "robustness_K.png"), dpi=110)
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for e, r in d.items():
            ln = r["landmark_noise"]
            sig = [0.1, 0.25, 0.5, 1.0, 2.0]
            for sub, ls in (("all", "-"), ("iris", "--"), ("eyelids", ":"), ("rigid_head_subset", "-.")):
                ax[0].plot(sig, [ln[f"{sub}_{s}px"]["cm_mean"] for s in sig], ls=ls, marker="o", label=f"{e} {sub}")
            sh = [0.5, 1.0, 2.0, 5.0]
            ax[1].plot(sh, [ln[f"face_shift_{s}px"]["cm_mean"] for s in sh], marker="o", label=e)
        ax[0].set_xlabel("Gaussian landmark noise sigma [px, 1920x1080]")
        ax[0].set_xscale("log")
        ax[1].set_xlabel("whole-face rigid shift sigma [px]")
        for a in ax:
            a.set_ylabel("mean error [cm]")
            a.legend(fontsize=5)
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(P, "robustness_landmark_noise.png"), dpi=110)
        plt.close()
    f = os.path.join(R, "robustness", "image_tests.json")
    if os.path.exists(f):
        d = json.load(open(f))
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for e, r in d["models"].items():
            sc = [0, 0.5, 1.0, 1.5]
            ax[0].plot(sc, [r["clean"]["cm_mean"]] + [r[f"distort_{s}x"]["cm_mean"] for s in sc[1:]], marker="o", label=f"{e} distorted")
            ax[0].plot(sc, [r["clean"]["cm_mean"]] + [r[f"distort_{s}x_undistorted"]["cm_mean"] for s in sc[1:]], marker="s", ls="--", label=f"{e} undistorted")
            names = [k for k in r if k.startswith(("Kgeom", "control")) and "|" in k or k.startswith("control")]
            ax[1].barh([n for n in names], [r[n]["shift_vs_clean_cm"] for n in names], alpha=0.5, label=e)
        ax[0].axhline(np.nan)
        ax[0].set_xlabel("lens distortion scale (x measured Mac coefficients)")
        ax[0].set_ylabel("mean error [cm] (fold-0 test identities)")
        ax[0].legend(fontsize=6)
        ax[1].set_xlabel("mean prediction shift vs clean render [cm]")
        ax[1].tick_params(axis="y", labelsize=6)
        ax[1].legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(P, "distortion.png"), dpi=110)
        plt.close()


def identity_curve():
    f = os.path.join(R, "identity_diversity", "summary.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    fig, ax = plt.subplots(figsize=(8, 5))
    for total, rows in d["by_total"].items():
        rows = sorted(rows, key=lambda r: r["n_ids"])
        ax.errorbar([r["n_ids"] for r in rows], [r["test_mean"] for r in rows], yerr=[r["test_std"] for r in rows], marker="o", capsize=3, label=f"total {total} samples")
    ax.set_xlabel("training identities")
    ax.set_ylabel("error on 20 held-out identities [cm]")
    ax.set_xscale("log")
    ax.set_title("identity diversity at fixed sample count (M8)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(P, "identity_curve.png"), dpi=110)
    plt.close()


for fn in (model_comparison, pose_tradeoff, calibration_curve, learning_curve, train_val_curves, robustness, identity_curve):
    try:
        fn()
        print("ok", fn.__name__)
    except Exception as ex:
        print("FAILED", fn.__name__, repr(ex))
