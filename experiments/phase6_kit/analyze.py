"""Phase 6 analysis: frozen M8 on real Mac webcam data, 0/1/3/5/9-point personal calibration, current tgaze baseline.

  python analyze.py outputs/real_mac_validation                  # main subjects (pilot excluded)
  python analyze.py outputs/real_mac_validation --pilot-check subject_00   # pilot QA checklist + calibration-only lambda sweep
  python analyze.py outputs/real_mac_validation --secondary      # + distortion ON/OFF, smoothing, other correction models

Nothing here trains M8. Personal corrections are fit only on the calibration session of the same subject and
evaluated only on that subject's evaluation session (different trials, different frames).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT_DIR)
import common as C  # noqa: E402

CAL_LEVELS = (0, 1, 3, 5, 9)


# =========================================================================== loading
def load_subject(sdir, cfg):
    info = json.load(open(os.path.join(sdir, "subject.json")))
    recs = [json.loads(l) for l in open(os.path.join(sdir, "metadata.jsonl"))]
    status_counts = {}
    final = {}
    for r in recs:
        if r["session"] not in ("calibration", "evaluation"):
            continue
        key = (r["session"], r["slot"]["slot_id"])
        status_counts.setdefault(r["session"], {}).setdefault(r["status"], 0)
        status_counts[r["session"]][r["status"]] += 1
        if r.get("operator_retake"):
            continue
        final[key] = r                                   # last attempt of each planned slot
    trials = []
    for (sess, _), r in sorted(final.items(), key=lambda kv: kv[1]["trial_id"]):
        z = np.load(os.path.join(sdir, r["frames_file"]))
        t = {"rec": r, "session": sess, "status": r["status"], "cond": r["pose_instruction"], "block": r["block"],
             "target": np.array(r["target"]["uv"]), "slot": r["slot"]["slot_id"], "z": z}
        rec = (z["phase"] == "record") & z["ok"]
        t["rec_idx"] = np.where(rec)[0]
        t["all_ok_idx"] = np.where(z["ok"])[0]
        trials.append(t)
    return info, recs, trials, status_counts


def usable(t, cfg):
    return t["status"] in cfg["analysis"]["use_status"] and len(t["rec_idx"]) > 0


def frame_uv(t, key="uv_m8_raw"):
    return t["z"][key][t["rec_idx"]].astype(np.float64)


def trial_pred(uv):
    return np.median(uv, 0)


# =========================================================================== geometry helpers
class Geo:
    def __init__(self, prep):
        d = prep["display"]
        self.c, self.r, self.u = (np.array(d[k]) for k in ("center_cam_m", "right_cam_m", "up_cam_m"))

    def point(self, uv):
        return self.c + uv[0] * C.HALF_CM[0] / 100 * self.r + uv[1] * C.HALF_CM[1] / 100 * self.u

    def ang(self, eye, p_uv, t_uv):
        a, b = self.point(p_uv) - eye, self.point(t_uv) - eye
        return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))


# =========================================================================== baselines (current tgaze)
def tgaze_appearance_fit(cal, points, min_frames=5):
    """Current tgaze default (appearance.AppearancePipeline: 16D + eye-patch PCA16 + Huber), fit on all recording
    frames of the calibration trials at `points` (exactly what tgaze does with its 9-point calibration)."""
    from appearance import AppearancePipeline
    pipe = AppearancePipeline()
    pts = np.array(points)
    for t in cal:
        if np.min(np.abs(pts - t["target"]).max(1)) > 1e-6:
            continue
        for i in t["rec_idx"]:
            p = t["z"]["eye_patch"][i].astype(np.float64)
            if np.isfinite(p).all():
                pipe.collect_point(t["z"]["f16"][i], p, C.uv_to_tgaze(t["target"]))
    if pipe.n_samples < min_frames:
        return None, None
    t0 = time.perf_counter()
    pipe.finalize()
    return pipe, 1e3 * (time.perf_counter() - t0)


def tgaze_appearance_predict(pipe, t):
    out = []
    for i in t["rec_idx"]:
        p = t["z"]["eye_patch"][i].astype(np.float64)
        if np.isfinite(p).all():
            out.append(C.tgaze_to_uv(pipe.predict(t["z"]["f16"][i], p, use_dynamic=False)))
    return np.median(out, 0) if out else None


def rich16d_huber_fit(cal, points):
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    pts = np.array(points)
    X, Y = [], []
    for t in cal:
        if np.min(np.abs(pts - t["target"]).max(1)) > 1e-6:
            continue
        X.append(t["z"]["f16"][t["rec_idx"]])
        Y += [t["target"]] * len(t["rec_idx"])
    X, Y = np.vstack(X).astype(np.float64), np.array(Y)
    sc = StandardScaler().fit(X)
    ms = [HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=3000).fit(sc.transform(X), Y[:, i]) for i in range(2)]
    return lambda F: np.column_stack([m.predict(sc.transform(F)) for m in ms])


# =========================================================================== smoothing (secondary)
def ema(seq, alpha):
    out = np.empty_like(seq)
    out[0] = seq[0]
    for i in range(1, len(seq)):
        out[i] = alpha * seq[i] + (1 - alpha) * out[i - 1]
    return out


def median_filter(seq, k):
    return np.array([np.median(seq[max(0, i - k + 1):i + 1], 0) for i in range(len(seq))])


def jitter(uv):
    mm = C.uv_to_cm(uv) * 10
    if len(mm) < 2:
        return {"jitter_x_mm": np.nan, "jitter_y_mm": np.nan, "jitter_radial_mm": np.nan, "spread_median_radius_mm": np.nan, "spread_p95_radius_mm": np.nan}
    sx, sy = mm.std(0, ddof=1)
    rad = np.linalg.norm(mm - np.median(mm, 0), axis=1)
    return {"jitter_x_mm": float(sx), "jitter_y_mm": float(sy), "jitter_radial_mm": float(np.hypot(sx, sy)),
            "spread_median_radius_mm": float(np.median(rad)), "spread_p95_radius_mm": float(np.percentile(rad, 95))}


# =========================================================================== main analysis
def analyze_subject(sid, sdir, cfg, geo, secondary, runtime=None):
    info, recs, trials, status_counts = load_subject(sdir, cfg)
    cal = [t for t in trials if t["session"] == "calibration" and usable(t, cfg)]
    ev = [t for t in trials if t["session"] == "evaluation" and usable(t, cfg)]
    ev_all = [t for t in trials if t["session"] == "evaluation"]
    cal_pairs = [(tuple(t["target"]), tuple(trial_pred(frame_uv(t)))) for t in cal]
    # head pose reference = this subject's calibration session (natural sitting posture)
    ref_py = np.median(np.vstack([np.degrees(t["z"]["f16"][t["rec_idx"]][:, [4, 5]]) for t in cal]), 0) if cal else np.zeros(2)

    corr, fit_ms = {}, {}
    for n in CAL_LEVELS:
        t0 = time.perf_counter()
        corr[("primary", n)] = C.fit_correction(cfg, n, cal_pairs)
        fit_ms[str(n)] = 1e3 * (time.perf_counter() - t0)
    variants = ["primary"]
    if secondary:
        for n in (3, 5, 9):
            for v in cfg["calibration"]["models"][str(n)]:
                if v != "primary":
                    corr[(v, n)] = C.fit_correction(cfg, n, cal_pairs, v)
                    variants.append(v)
    variants = list(dict.fromkeys(variants))

    base = {}
    pipe9, base_fit_ms = (tgaze_appearance_fit(cal, C.calibration_points(cfg, 9)) if cal else (None, None))
    if pipe9 is not None:
        base["tgaze_16d+patchPCA_huber_9pt"] = pipe9
    if secondary and cal:
        p5, _ = tgaze_appearance_fit(cal, C.calibration_points(cfg, 5))
        if p5 is not None:
            base["tgaze_16d+patchPCA_huber_5pt"] = p5
        base["tgaze_rich16d_huber_9pt"] = rich16d_huber_fit(cal, C.calibration_points(cfg, 9))

    rows = []
    for t in ev:
        uv = frame_uv(t)
        raw = trial_pred(uv)
        eye = np.median(t["z"]["eye_cam_m"][t["rec_idx"]], 0)
        py = np.median(np.degrees(t["z"]["f16"][t["rec_idx"]][:, [4, 5, 7]]), 0)
        dpose = float(np.hypot(*(py[:2] - ref_py)))
        r = {"subject_id": sid, "trial_id": t["rec"]["trial_id"], "slot": t["slot"], "block": t["block"], "pose_instruction": t["cond"],
             "status": t["status"], "target_u": t["target"][0], "target_v": t["target"][1], "region": str(C.screen_region(t["target"])),
             "n_record_frames": len(t["rec_idx"]), "head_pitch_deg": py[0], "head_yaw_deg": py[1], "head_roll_deg": py[2],
             "head_delta_from_cal_deg": dpose, "eye_z_cm": eye[2] * 100,
             "pred_live_u": t["rec"]["m8_raw_pred_uv_median"][0], "pred_live_v": t["rec"]["m8_raw_pred_uv_median"][1]}
        for (v, n), c in corr.items():
            p = c.apply(raw)
            tag = f"C{n}" if v == "primary" else f"C{n}_{v.replace('secondary_', '')}"
            r[f"{tag}_u"], r[f"{tag}_v"] = p
            r[f"{tag}_err_cm"] = float(C.cm_err(p, t["target"]))
            r[f"{tag}_err_deg"] = geo.ang(eye, p, t["target"])
        for name, b in base.items():
            if name.startswith("tgaze_16d+"):
                p = tgaze_appearance_predict(b, t)
            else:
                p = np.median(b(t["z"]["f16"][t["rec_idx"]].astype(np.float64)), 0)
            r[f"{name}_u"], r[f"{name}_v"] = (p if p is not None else (np.nan, np.nan))
            r[f"{name}_err_cm"] = float(C.cm_err(p, t["target"])) if p is not None else np.nan
            r[f"{name}_err_deg"] = geo.ang(eye, p, t["target"]) if p is not None else np.nan
        r.update(jitter(uv))
        if secondary:
            seq = t["z"]["uv_m8_raw"][t["all_ok_idx"]].astype(np.float64)
            is_rec = np.isin(t["all_ok_idx"], t["rec_idx"])
            for name, f in (("ema", lambda s: ema(s, cfg["analysis"]["ema_alpha"])), ("medfilt", lambda s: median_filter(s, cfg["analysis"]["median_window"]))):
                sm = f(seq)[is_rec]
                j = jitter(sm)
                r[f"smooth_{name}_jitter_radial_mm"] = j["jitter_radial_mm"]
                r[f"smooth_{name}_C0_err_cm"] = float(C.cm_err(np.median(sm, 0), t["target"]))
                r[f"smooth_{name}_C9_err_cm"] = float(C.cm_err(corr[("primary", 9)].apply(np.median(sm, 0)), t["target"]))
            if runtime is not None and runtime.get("dist") is not None:
                rt, dist = runtime["rt"], runtime["dist"]
                uvd = []
                for i in t["rec_idx"]:
                    lms_u = __import__("m8_cpu").undistort_lms(t["z"]["lms"][i].astype(np.float64), rt.K_raw, dist, rt.w, rt.h)
                    fe = rt.features_from_lms(lms_u)
                    if fe is not None:
                        uvd.append(rt.infer(fe["inputs"]))
                r["dist_on_C0_err_cm"] = float(C.cm_err(np.median(uvd, 0), t["target"])) if uvd else np.nan
                r["dist_on_raw_u"], r["dist_on_raw_v"] = (np.median(uvd, 0) if uvd else (np.nan, np.nan))
        rows.append(r)

    # distortion-ON needs its own calibration fits (on undistorted calibration predictions)
    if secondary and runtime is not None and runtime.get("dist") is not None and rows:
        rt, dist = runtime["rt"], runtime["dist"]
        cp = []
        for t in cal:
            uvd = []
            for i in t["rec_idx"]:
                fe = rt.features_from_lms(__import__("m8_cpu").undistort_lms(t["z"]["lms"][i].astype(np.float64), rt.K_raw, dist, rt.w, rt.h))
                if fe is not None:
                    uvd.append(rt.infer(fe["inputs"]))
            if uvd:
                cp.append((tuple(t["target"]), tuple(np.median(uvd, 0))))
        for n in (1, 9):
            c = C.fit_correction(cfg, n, cp)
            for r in rows:
                if np.isfinite(r.get("dist_on_raw_u", np.nan)):
                    r[f"dist_on_C{n}_err_cm"] = float(C.cm_err(c.apply(np.array([r["dist_on_raw_u"], r["dist_on_raw_v"]])), np.array([r["target_u"], r["target_v"]])))

    # offline recomputation must equal the live prediction (no hidden state / same pipeline)
    if runtime is not None and ev:
        rt = runtime["rt"]
        diffs = []
        for t in ev[:10]:
            for i in t["rec_idx"][:3]:
                fe = rt.features_from_lms(t["z"]["lms"][i].astype(np.float64))
                diffs.append(np.abs(rt.infer(fe["inputs"]) - t["z"]["uv_m8_raw"][i]).max())
        runtime.setdefault("live_vs_offline_max_abs_uv", {})[sid] = float(np.max(diffs)) if diffs else None

    lat = {}
    for k in ("mediapipe", "features", "onnx", "eye_patch", "total_m8_only", "total"):
        v = np.concatenate([t["z"][f"lat_{k}_ms"][t["rec_idx"]] for t in ev_all if f"lat_{k}_ms" in t["z"]] or [np.array([])])
        v = v[np.isfinite(v)]
        lat[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} if len(v) else None
    perf = [json.loads(l) for l in open(os.path.join(sdir, "perf.jsonl"))] if os.path.exists(os.path.join(sdir, "perf.jsonl")) else []
    perf = [p for p in perf if p.get("session") in ("calibration", "evaluation")]
    benches = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sdir, "cpu_benchmark_*.json")))]
    cam_fps = [t["rec"]["camera_fps"] for t in ev_all if t["rec"].get("camera_fps") == t["rec"].get("camera_fps")]
    eff = []
    for t in ev_all:
        ti = t["z"]["t_rel_s"]
        if len(ti) > 2:
            eff.append((len(ti) - 1) / (ti[-1] - ti[0]))
    on_dev = json.load(open(os.path.join(sdir, "calibration_model.json"))) if os.path.exists(os.path.join(sdir, "calibration_model.json")) else None
    subj = {"subject_id": sid, "role": info.get("role"), "fake_camera": info.get("fake_camera"), "status_counts": status_counts,
            "n_eval_trials_used": len(ev), "n_eval_trials_planned_slots": len({t['slot'] for t in ev_all}), "n_cal_trials_used": len(cal),
            "head_pose_reference_pitch_yaw_deg": ref_py.tolist(),
            "calibration_fit_ms_analysis": fit_ms, "tgaze_9pt_fit_ms_analysis": base_fit_ms,
            "calibration_fit_on_device": on_dev, "corrections": {f"{v}:{n}": c.to_json() for (v, n), c in corr.items()},
            "latency_ms": lat, "camera_fps_median": float(np.median(cam_fps)) if cam_fps else None,
            "processed_fps_in_trials_median": float(np.median(eff)) if eff else None,
            "cpu_percent_one_core_median": float(np.median([p["cpu_percent_one_core"] for p in perf])) if perf else None,
            "rss_mb_max": float(max(p["rss_mb"] for p in perf)) if perf else None, "n_logical_cpu": perf[0]["n_logical_cpu"] if perf else None,
            "benchmarks": benches, "runtime": info.get("runtime"), "camera": info.get("camera"), "camera_matrix": info.get("camera_matrix"),
            "sanity": info.get("sanity"), "geometry_check": info.get("geometry_check"), "frozen_protocol_hash": info.get("frozen_protocol_hash"),
            "model_onnx_sha256": info.get("model_onnx_sha256"), "platform": info.get("platform")}
    return subj, rows


def stats_table(rows, col_err, col_deg=None):
    e = np.array([r[col_err] for r in rows if col_err in r], float)
    d = np.array([r[col_deg] for r in rows if col_deg and col_deg in r], float) if col_deg else None
    return C.error_stats(e, d)


def pose_invariance(rows, levels):
    """Per subject x target: distance between EYE_DOMINANT and HEAD_ALLOWED predictions (same target, different
    strategy), plus NATURAL(first block) vs NATURAL(last block) test-retest noise floor."""
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["subject_id"], r["target_u"], r["target_v"]), []).append(r)
    for (sid, u, v), rs in by.items():
        cond = {}
        for r in rs:
            cond.setdefault(r["pose_instruction"], []).append(r)
        rec = {"subject_id": sid, "target_u": u, "target_v": v}
        for n in levels:
            P = lambda r: np.array([r[f"C{n}_u"], r[f"C{n}_v"]])
            if "EYE_DOMINANT" in cond and "HEAD_ALLOWED" in cond:
                rec[f"C{n}_eye_vs_head_cm"] = float(np.linalg.norm(C.uv_to_cm(P(cond["EYE_DOMINANT"][0])) - C.uv_to_cm(P(cond["HEAD_ALLOWED"][0]))))
            nat = sorted(cond.get("NATURAL", []), key=lambda r: r["block"])
            if len(nat) >= 2:
                rec[f"C{n}_natural_retest_cm"] = float(np.linalg.norm(C.uv_to_cm(P(nat[0])) - C.uv_to_cm(P(nat[-1]))))
            three = [cond[c][0] for c in ("NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED") if c in cond]
            if len(three) == 3:
                pts = [C.uv_to_cm(P(r)) for r in three]
                rec[f"C{n}_max_pair_3cond_cm"] = float(max(np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]))
        if "EYE_DOMINANT" in cond and "HEAD_ALLOWED" in cond:
            e, h = cond["EYE_DOMINANT"][0], cond["HEAD_ALLOWED"][0]
            rec["head_yaw_diff_deg"] = abs(h["head_yaw_deg"] - e["head_yaw_deg"])
            rec["head_pitch_diff_deg"] = abs(h["head_pitch_deg"] - e["head_pitch_deg"])
            rec["head_pose_diff_deg"] = float(np.hypot(rec["head_yaw_diff_deg"], rec["head_pitch_diff_deg"]))
        out.append(rec)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def mean_std(x):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    return [float(x.mean()), float(x.std())] if len(x) else [None, None]


# =========================================================================== plots
def make_plots(pdir, rows, subjects, calcomp, pinv, cfg, base_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if any(s["fake_camera"] for s in subjects):                     # never let a pipeline test look like real data
        _save = plt.savefig

        def savefig(*a, **k):
            plt.gcf().text(0.5, 0.5, "FAKE CAMERA TEST - NOT REAL DATA", ha="center", va="center", fontsize=26, color="red", alpha=0.25, rotation=20)
            return _save(*a, **k)
        plt.savefig = savefig
    os.makedirs(pdir, exist_ok=True)
    ref = cfg["analysis"]["synthetic_reference"]
    try:
        _make_plots(plt, pdir, rows, calcomp, pinv, cfg, base_names, ref)
    finally:
        if any(s["fake_camera"] for s in subjects):
            plt.savefig = _save


def _make_plots(plt, pdir, rows, calcomp, pinv, cfg, base_names, ref):
    sids = sorted({r["subject_id"] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in sids:
        ax.plot(CAL_LEVELS, [np.mean([r[f"C{n}_err_cm"] for r in rows if r["subject_id"] == s]) for n in CAL_LEVELS], "o-", alpha=0.4, lw=1, label=s)
    for key, sty, lab in (("mean_cm", "k-o", "all subjects: mean"), ("median_cm", "k--s", "median"), ("p95_cm", "k:^", "P95")):
        ax.plot(CAL_LEVELS, [calcomp[f"C{n}"]["pooled"][key] for n in CAL_LEVELS], sty, lw=2, label=lab)
    ax.axhline(ref["M8_identity_cv_mean_cm"], color="tab:green", ls="--", label=f"synthetic identity-CV M8 {ref['M8_identity_cv_mean_cm']:.2f} cm")
    for b in base_names:
        if b.endswith("9pt") and b.startswith("tgaze_16d+"):
            ax.plot([9], [np.nanmean([r[f"{b}_err_cm"] for r in rows])], "r*", ms=14, label="current tgaze (16D+patchPCA+Huber), 9-pt")
    ax.set_xticks(CAL_LEVELS), ax.set_xlabel("personal calibration points (M8 frozen)"), ax.set_ylabel("screen error [cm]")
    ax.set_title("real Mac webcam: error vs calibration points"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
    plt.tight_layout(), plt.savefig(os.path.join(pdir, "calibration_curve.png"), dpi=110), plt.close()

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(sids)), 4.5))
    wd = 0.8 / (len(CAL_LEVELS) + 1)
    for j, n in enumerate(CAL_LEVELS):
        ax.bar(np.arange(len(sids)) + j * wd, [np.mean([r[f"C{n}_err_cm"] for r in rows if r["subject_id"] == s]) for s in sids], wd, label=f"M8 + {n}-pt")
    tb = [b for b in base_names if b.startswith("tgaze_16d+") and b.endswith("9pt")]
    if tb:
        ax.bar(np.arange(len(sids)) + len(CAL_LEVELS) * wd, [np.nanmean([r[f"{tb[0]}_err_cm"] for r in rows if r["subject_id"] == s]) for s in sids], wd, color="tab:red", label="tgaze 9-pt")
    ax.set_xticks(np.arange(len(sids)) + 0.4 - wd / 2), ax.set_xticklabels(sids), ax.set_ylabel("mean error [cm]"), ax.legend(fontsize=7)
    ax.set_title("error by subject"), plt.tight_layout(), plt.savefig(os.path.join(pdir, "error_by_subject.png"), dpi=110), plt.close()

    def by(key, cats, fname, title):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        wd = 0.8 / len(CAL_LEVELS)
        for j, n in enumerate(CAL_LEVELS):
            vals = [np.mean([r[f"C{n}_err_cm"] for r in rows if r[key] == c]) if any(r[key] == c for r in rows) else np.nan for c in cats]
            ax.bar(np.arange(len(cats)) + j * wd, vals, wd, label=f"M8 + {n}-pt")
        counts = [sum(r[key] == c for r in rows) for c in cats]
        ax.set_xticks(np.arange(len(cats)) + 0.4 - wd / 2), ax.set_xticklabels([f"{c}\n(n={k})" for c, k in zip(cats, counts)])
        ax.set_ylabel("mean error [cm]"), ax.set_title(title), ax.legend(fontsize=7)
        plt.tight_layout(), plt.savefig(os.path.join(pdir, fname), dpi=110), plt.close()

    by("region", ["center", "middle", "edge"], "error_by_screen_region.png", "error by screen region (5x5 grid rings)")
    b1, b2 = cfg["analysis"]["head_pose_bins_deg"]
    for r in rows:
        d = r["head_delta_from_cal_deg"]
        r["head_pose_bin"] = f"small <{b1:g}" if d < b1 else (f"medium {b1:g}-{b2:g}" if d < b2 else f"large >={b2:g}")
    by("head_pose_bin", [f"small <{b1:g}", f"medium {b1:g}-{b2:g}", f"large >={b2:g}"], "error_by_head_pose.png",
       "error by head pose (|pitch,yaw| from the subject's calibration posture, deg)")
    by("pose_instruction", ["NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED"], "error_by_pose_instruction.png", "error by pose instruction")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    labels, vals, errs = [], [], []
    for n in CAL_LEVELS:
        for k, lab in ((f"C{n}_eye_vs_head_cm", f"EYE vs HEAD\nC{n}"), (f"C{n}_natural_retest_cm", f"NAT retest\nC{n}")):
            v = [p[k] for p in pinv if k in p]
            if v:
                labels.append(lab), vals.append(np.mean(v)), errs.append(np.std(v))
    ax[0].bar(range(len(vals)), vals, yerr=errs, capsize=3, color=["tab:orange" if "EYE" in l else "tab:gray" for l in labels])
    ax[0].axhline(ref["M8_pose_invariance_cm"], color="tab:green", ls="--", label=f"synthetic M8 {ref['M8_pose_invariance_cm']} cm (max over 4 strategies)")
    ax[0].set_xticks(range(len(vals))), ax[0].set_xticklabels(labels, fontsize=7), ax[0].set_ylabel("[cm]"), ax[0].legend(fontsize=7)
    ax[0].set_title("real pose-invariance error (same target, different strategy)")
    hv = [p.get("head_pose_diff_deg") for p in pinv if p.get("head_pose_diff_deg") is not None]
    iv = [p.get("C0_eye_vs_head_cm") for p in pinv if p.get("head_pose_diff_deg") is not None]
    ax[1].scatter(hv, iv, s=12), ax[1].set_xlabel("head pose difference EYE vs HEAD [deg]"), ax[1].set_ylabel("prediction difference, zero-shot [cm]")
    ax[1].set_title("does the prediction move with the head?")
    plt.tight_layout(), plt.savefig(os.path.join(pdir, "pose_invariance.png"), dpi=110), plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = ["synthetic\nidentity-CV"] + [f"real\n{n}-pt" for n in CAL_LEVELS]
    m = [ref["M8_identity_cv_mean_cm"]] + [calcomp[f"C{n}"]["pooled"]["mean_cm"] for n in CAL_LEVELS]
    s = [ref["M8_identity_cv_std_cm"]] + [calcomp[f"C{n}"]["subject_mean_cm_mean_std"][1] or 0 for n in CAL_LEVELS]
    p95 = [ref["M8_identity_cv_p95_cm"]] + [calcomp[f"C{n}"]["pooled"]["p95_cm"] for n in CAL_LEVELS]
    ax.bar(range(len(m)), m, yerr=s, capsize=4, color=["tab:green"] + ["tab:blue"] * len(CAL_LEVELS), label="mean (+- std over subjects / folds)")
    ax.plot(range(len(m)), p95, "k^", label="P95")
    ax.set_xticks(range(len(m))), ax.set_xticklabels(names), ax.set_ylabel("screen error [cm]"), ax.legend()
    ax.set_title("M8: synthetic unseen identities vs real Mac webcam (not identical conditions)")
    plt.tight_layout(), plt.savefig(os.path.join(pdir, "synthetic_vs_real.png"), dpi=110), plt.close()

    n_s = len(sids)
    fig, axs = plt.subplots(1, n_s, figsize=(5 * n_s, 3.8), squeeze=False)
    for a, sid in zip(axs[0], sids):
        rs = [r for r in rows if r["subject_id"] == sid]
        T = np.array([[r["target_u"], r["target_v"]] for r in rs]) * C.HALF_CM
        for n, col in ((0, "tab:orange"), (9, "tab:blue")):
            P = np.array([[r[f"C{n}_u"], r[f"C{n}_v"]] for r in rs]) * C.HALF_CM
            a.scatter(P[:, 0], P[:, 1], s=8, color=col, label=f"{n}-pt")
            for t, p in zip(T, P):
                a.plot([t[0], p[0]], [t[1], p[1]], color=col, lw=0.4, alpha=0.5)
        a.scatter(T[:, 0], T[:, 1], marker="+", color="k", s=40, label="target")
        a.set_xlim(-C.HALF_CM[0], C.HALF_CM[0]), a.set_ylim(-C.HALF_CM[1], C.HALF_CM[1]), a.set_aspect("equal")
        a.set_title(sid), a.legend(fontsize=6)
    plt.tight_layout(), plt.savefig(os.path.join(pdir, "prediction_scatter.png"), dpi=100), plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lab in (("jitter_x_mm", "x"), ("jitter_y_mm", "y"), ("jitter_radial_mm", "radial")):
        ax.hist([r[k] for r in rows if np.isfinite(r[k])], bins=30, alpha=0.5, label=lab)
    ax.set_xlabel("within-trial prediction std [mm] (raw frames, zero-shot, no smoothing)"), ax.legend()
    plt.tight_layout(), plt.savefig(os.path.join(pdir, "jitter.png"), dpi=110), plt.close()


# =========================================================================== pilot check
def pilot_check(root, sid, cfg, rt):
    sdir = os.path.join(root, sid)
    info, recs, trials, status_counts = load_subject(sdir, cfg)
    cal = [t for t in trials if t["session"] == "calibration" and usable(t, cfg)]
    lines = [f"# Pilot check: {sid}", ""]
    chk = lambda ok, text: lines.append(f"- [{'x' if ok else ' '}] {text}")
    cam = info.get("camera") or {}
    chk(cam.get("size") == [1920, 1080], f"webcam resolution {cam.get('size')} (backend {cam.get('backend')}, Center Stage disabled: {cam.get('center_stage_disabled')})")
    ru = info.get("runtime") or {}
    km = info.get("camera_matrix") or {}
    chk(ru.get("K_matches_training") and km.get("host_match"), f"measured K applied (host {km.get('this_machine')} vs json {km.get('host_in_json')}; |K - training K| = {ru.get('K_max_abs_diff_vs_training_px')} px)")
    chk(ru.get("mirrored") is True, f"mirrored=True; {ru.get('rich16d_set_camera_matrix')}")
    chk(ru.get("ort_session_providers") == ["CPUExecutionProvider"] and not ru.get("torch_loaded"),
        f"ONNX Runtime providers {ru.get('ort_session_providers')} (available: {ru.get('ort_available_providers')}); MediaPipe delegate {ru.get('mediapipe_delegate')}; torch loaded: {ru.get('torch_loaded')}")
    geo = info.get("geometry_check") or {}
    chk(geo.get("confirmed"), f"geometry / target coordinate check confirmed with a ruler: {geo.get('confirmed')} layout {geo.get('layout')}")
    s = info.get("sanity") or {}
    chk(s.get("pass"), f"direction sanity check: right-left {s.get('dx_right_minus_left_cm')} cm, up-down {s.get('dy_up_minus_down_cm')} cm")
    uvt = np.array([[0.8, 0.8]])
    chk(abs(C.uv_to_cm(uvt)[0, 0] - 11.624) < 1e-3 and abs(C.uv_to_cm(uvt)[0, 1] - 7.56) < 1e-3, "cm conversion: uv (0.8, 0.8) -> (11.624, 7.560) cm from display centre (290.6 x 189.0 mm)")
    nf = [r["n_frames"] for r in recs if r["session"] in ("calibration", "evaluation")]
    fps = [r["camera_fps"] for r in recs if r.get("camera_fps")]
    chk(bool(nf) and np.median([n["record"] for n in nf]) >= cfg["quality"]["min_record_frames"],
        f"record timing: frames per window median move/settle/record = {[float(np.median([n[p] for n in nf])) for p in ('move', 'settle', 'record')] if nf else None}, camera FPS median {np.median(fps) if fps else None}")
    lat = [r["latency_ms_mean"]["onnx"] for r in recs if r.get("latency_ms_mean", {}).get("onnx")]
    chk(bool(lat), f"M8 ONNX CPU inference ran: mean {np.mean(lat) if lat else None} ms")
    mp = [r["latency_ms_mean"]["mediapipe"] for r in recs if r.get("latency_ms_mean", {}).get("mediapipe")]
    chk(bool(mp), f"MediaPipe ran: mean {np.mean(mp) if mp else None} ms")
    cm = json.load(open(os.path.join(sdir, "calibration_model.json"))) if os.path.exists(os.path.join(sdir, "calibration_model.json")) else None
    chk(cm is not None, f"calibration fitted on device: {({k: round(v['fit_ms'], 3) for k, v in cm['models'].items()} if cm else None)} ms; tgaze 9-pt: {cm.get('tgaze_appearance_9pt') if cm else None}")
    chk(os.path.exists(os.path.join(sdir, "metadata.jsonl")) and any(os.scandir(os.path.join(sdir, "representative_frames"))), "result saving: metadata.jsonl, per-trial npz, representative frames")
    lines += ["", f"status counts: {status_counts}", ""]
    # calibration-only leave-one-point-out lambda sweep (uses calibration trials only, never evaluation targets)
    pairs = [(tuple(t["target"]), tuple(trial_pred(frame_uv(t)))) for t in cal]
    lines += ["## Calibration-only LOO lambda sweep (pilot only; evaluation trials not used)", "",
              "| model | lambda cm^2 | LOO error cm (9 cal points) |", "|---|---|---|"]
    for kind in ("affine", "bias_scale"):
        for lam in (0, 5, 25, 100, 400):
            errs = []
            pts = sorted(set(p for p, _ in pairs))
            for held in pts:
                tr = [(t, p) for t, p in pairs if t != held]
                te = [(t, p) for t, p in pairs if t == held]
                c = C.Correction(kind, lam).fit(np.array([p for _, p in tr]), np.array([t for t, _ in tr]))
                errs += [float(C.cm_err(c.apply(np.array(p)), np.array(t))) for t, p in te]
            lines.append(f"| {kind} | {lam} | {np.mean(errs):.2f} |" if errs else f"| {kind} | {lam} | - |")
    c0 = [float(C.cm_err(np.array(p), np.array(t))) for t, p in pairs]
    lines += ["", f"zero-shot error on the pilot calibration trials: {np.mean(c0):.2f} cm" if c0 else "", "",
              "Change config.json (calibration.models.*.lambda_cm2, timing, quality) only now, before main subjects. "
              "After that the frozen_protocol_hash must stay the same for every main subject."]
    out = os.path.join(root, "results", f"pilot_check_{sid}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# =========================================================================== report
def write_report(path, summary, cfg):
    s = summary
    ref = cfg["analysis"]["synthetic_reference"]
    cc = s["calibration_comparison"]
    L = [f"# Phase 6: frozen M8 on real MacBook Air M4 webcam data", "",
         f"generated {time.strftime('%Y-%m-%d %H:%M')}; subjects: {', '.join(s['subjects'])}" + ("  **(FAKE CAMERA / SYNTHETIC TEST RUN - not real data)**" if s["any_fake_camera"] else ""), "",
         "## Calibration comparison (same evaluation trials for every row)", "",
         "| Calibration | Mean cm | Median cm | RMSE cm | P90 | P95 | Mean deg | subject means (mean +- std) |", "|---|---|---|---|---|---|---|---|"]
    for n in CAL_LEVELS:
        p = cc[f"C{n}"]["pooled"]
        sm = cc[f"C{n}"]["subject_mean_cm_mean_std"]
        L.append(f"| {n}-point | {p['mean_cm']:.2f} | {p['median_cm']:.2f} | {p['rmse_cm']:.2f} | {p['p90_cm']:.2f} | {p['p95_cm']:.2f} | {p.get('mean_deg', float('nan')):.2f} | {sm[0]:.2f} +- {sm[1]:.2f} |")
    for b, v in s["baselines"].items():
        p = v["pooled"]
        L.append(f"| {b} | {p['mean_cm']:.2f} | {p['median_cm']:.2f} | {p['rmse_cm']:.2f} | {p['p90_cm']:.2f} | {p['p95_cm']:.2f} | {p.get('mean_deg', float('nan')):.2f} | {v['subject_mean_cm_mean_std'][0]:.2f} +- {v['subject_mean_cm_mean_std'][1]:.2f} |")
    L += ["", "## Synthetic vs real", "", "```",
          f"Synthetic identity-disjoint M8 = {ref['M8_identity_cv_mean_cm']:.2f} +- {ref['M8_identity_cv_std_cm']:.2f} cm (P95 {ref['M8_identity_cv_p95_cm']:.2f})"]
    for n in CAL_LEVELS:
        sm = cc[f"C{n}"]["subject_mean_cm_mean_std"]
        L.append(f"Real {'zero-shot' if n == 0 else f'+ {n} point'}{' ' * (10 - len('zero-shot' if n == 0 else f'+ {n} point'))} = {cc[f'C{n}']['pooled']['mean_cm']:.2f} cm (subjects {sm[0]:.2f} +- {sm[1]:.2f}, P95 {cc[f'C{n}']['pooled']['p95_cm']:.2f})")
    L += ["```", "", f"real zero-shot - synthetic identity-disjoint = {s['synthetic_to_real_gap_cm']:+.2f} cm. "
          "This difference is NOT a strict domain gap: the synthetic test used rendered faces, random targets over the "
          "whole display, kappa = 0 and a fixed ergonomic geometry, while the real test uses 5 real people, a 5x5 grid, "
          "trial medians over ~0.8 s and real posture. It mixes synthetic-to-real appearance, identity (kappa, eye shape), "
          "geometry and protocol differences.", ""]
    for sec, key in (("Subject", "by_subject"), ("Screen region", "by_region"), ("Head pose (deviation from calibration posture)", "by_head_pose"), ("Pose instruction", "by_pose_instruction")):
        L += [f"## By {sec.lower()}", "", "| " + sec + " | n | " + " | ".join(f"{n}-pt mean cm" for n in CAL_LEVELS) + " | 0-pt P95 | 9-pt P95 |", "|---" * (len(CAL_LEVELS) + 4) + "|"]
        for cat, v in s[key].items():
            L.append(f"| {cat} | {v['C0']['n']} | " + " | ".join(f"{v[f'C{n}']['mean_cm']:.2f}" for n in CAL_LEVELS) + f" | {v['C0']['p95_cm']:.2f} | {v['C9']['p95_cm']:.2f} |")
        L.append("")
    pi = s["pose_invariance"]
    L += ["## Pose invariance (real_pose_invariance_error_cm)", "",
          f"EYE_DOMINANT vs HEAD_ALLOWED, same target: zero-shot {pi['C0_eye_vs_head_cm'][0]:.2f} +- {pi['C0_eye_vs_head_cm'][1]:.2f} cm, "
          f"9-pt {pi['C9_eye_vs_head_cm'][0]:.2f} cm; NATURAL test-retest noise floor {pi['C0_natural_retest_cm'][0]:.2f} cm; "
          f"max pair over 3 strategies {pi['C0_max_pair_3cond_cm'][0]:.2f} cm. Head pose actually differed by {pi['head_pose_diff_deg'][0]:.1f} deg on average. "
          f"Synthetic M8: {ref['M8_pose_invariance_cm']} cm (max over 4 synthetic strategies; not the same definition).", "",
          "## Jitter (raw frames, zero-shot, no smoothing)", "", f"{json.dumps(s['jitter'])}", "",
          "## CPU runtime (no GPU)", "", f"```\n{json.dumps(s['runtime'], indent=1)}\n```", ""]
    if s.get("secondary"):
        L += ["## Secondary experiments (not part of the primary result)", "", f"```\n{json.dumps(s['secondary'], indent=1)}\n```", ""]
    L += ["## Diagnostic indicators", "", f"```\n{json.dumps(s['indicators'], indent=1)}\n```", ""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", default=None, help="comma list (default: all main subjects)")
    ap.add_argument("--include-pilot", action="store_true")
    ap.add_argument("--pilot-check", default=None)
    ap.add_argument("--secondary", action="store_true")
    ap.add_argument("--allow-mixed-protocol", action="store_true")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    tg = C.tgaze_dir(cfg)
    import m8_cpu
    K, kinfo = m8_cpu.load_camera_matrix(cfg, tg, allow_host_mismatch=True)
    rt = m8_cpu.M8Runtime(cfg, tg, K, video_mode="IMAGE")
    runtime = {"rt": rt, "dist": kinfo.get("dist_k1_k2_p1_p2_k3")}
    if a.pilot_check:
        pilot_check(a.root, a.pilot_check, cfg, rt)
        rt.close()
        return
    geo = Geo(rt.prep)
    sids = a.subjects.split(",") if a.subjects else sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.root, "subject_*")))
    subjects, rows = [], []
    for sid in sids:
        info = json.load(open(os.path.join(a.root, sid, "subject.json")))
        if info.get("role") == "pilot" and not a.include_pilot:
            continue
        print(f"[analyze] {sid}")
        sj, rr = analyze_subject(sid, os.path.join(a.root, sid), cfg, geo, a.secondary, runtime)
        subjects.append(sj)
        rows += rr
    hashes = {s["frozen_protocol_hash"] for s in subjects}
    if len(hashes) > 1 and not a.allow_mixed_protocol:
        sys.exit(f"main subjects were recorded with different protocol/config hashes {hashes}; settings must be frozen after the pilot")
    if C.frozen_config_hash(cfg) not in hashes and not a.allow_mixed_protocol:
        sys.exit(f"current config hash {C.frozen_config_hash(cfg)} differs from the recorded {hashes}")
    assert len({s["model_onnx_sha256"] for s in subjects}) == 1

    rdir, pdir = os.path.join(a.root, "results"), os.path.join(a.root, "plots")
    os.makedirs(rdir, exist_ok=True)
    base_names = sorted({k[:-7] for r in rows for k in r if k.endswith("_err_cm") and k.startswith("tgaze_")})
    calcomp = {}
    for n in CAL_LEVELS:
        per = [np.mean([r[f"C{n}_err_cm"] for r in rows if r["subject_id"] == s["subject_id"]]) for s in subjects]
        calcomp[f"C{n}"] = {"pooled": stats_table(rows, f"C{n}_err_cm", f"C{n}_err_deg"), "subject_mean_cm_mean_std": mean_std(per)}
    baselines = {}
    for b in base_names:
        per = [np.nanmean([r[f"{b}_err_cm"] for r in rows if r["subject_id"] == s["subject_id"]]) for s in subjects]
        d = [np.nanmean([r[f"{b}_err_cm"] - r["C9_err_cm"] for r in rows if r["subject_id"] == s["subject_id"]]) for s in subjects]
        baselines[b] = {"pooled": stats_table(rows, f"{b}_err_cm", f"{b}_err_deg"), "subject_mean_cm_mean_std": mean_std(per),
                        "paired_minus_M8_9pt_cm_per_subject": [float(x) for x in d],
                        "subjects_where_M8_9pt_better": int(sum(x > 0 for x in d))}

    def breakdown(key, cats):
        return {c: {f"C{n}": stats_table([r for r in rows if r[key] == c], f"C{n}_err_cm") for n in CAL_LEVELS} for c in cats if any(r[key] == c for r in rows)}

    b1, b2 = cfg["analysis"]["head_pose_bins_deg"]
    for r in rows:
        d = r["head_delta_from_cal_deg"]
        r["head_pose_bin"] = f"small <{b1:g}" if d < b1 else (f"medium {b1:g}-{b2:g}" if d < b2 else f"large >={b2:g}")
    pinv = pose_invariance(rows, CAL_LEVELS)
    pkeys = sorted({k for p in pinv for k in p if k.endswith("_cm") or k.endswith("_deg")})
    jit = {k: mean_std([r[k] for r in rows]) for k in ("jitter_x_mm", "jitter_y_mm", "jitter_radial_mm", "spread_median_radius_mm", "spread_p95_radius_mm")}
    lat_keys = ("mediapipe", "features", "onnx", "eye_patch", "total_m8_only", "total")
    runtime_sum = {
        "latency_ms_mean_over_subjects": {k: mean_std([s["latency_ms"][k]["mean"] if s["latency_ms"].get(k) else None for s in subjects]) for k in lat_keys},
        "latency_ms_p95_over_subjects": {k: mean_std([s["latency_ms"][k]["p95"] if s["latency_ms"].get(k) else None for s in subjects]) for k in lat_keys},
        "camera_fps_median": mean_std([s["camera_fps_median"] for s in subjects]),
        "processed_fps_in_trials": mean_std([s["processed_fps_in_trials_median"] for s in subjects]),
        "cpu_percent_one_core": mean_std([s["cpu_percent_one_core_median"] for s in subjects]),
        "rss_mb_max": mean_std([s["rss_mb_max"] for s in subjects]),
        "benchmarks": [{"subject": s["subject_id"], **{k: b[k] for k in ("camera_fps", "effective_fps", "cpu_percent_of_one_core", "rss_mb", "ort_session_providers", "torch_loaded")},
                        "latency_ms_mean": {k: v["mean"] for k, v in b["latency_ms"].items()}} for s in subjects for b in s["benchmarks"]],
        "ort_session_providers": sorted({str(s["runtime"]["ort_session_providers"]) for s in subjects if s.get("runtime")}),
        "ort_available_providers": sorted({str(s["runtime"]["ort_available_providers"]) for s in subjects if s.get("runtime")}),
        "mediapipe_delegate": sorted({s["runtime"]["mediapipe_delegate"] for s in subjects if s.get("runtime")}),
        "torch_loaded": any(s["runtime"]["torch_loaded"] for s in subjects if s.get("runtime")),
        "calibration_fit_ms_on_device": {n: mean_std([s["calibration_fit_on_device"]["models"][n]["fit_ms"] for s in subjects if s.get("calibration_fit_on_device")]) for n in ("1", "3", "5", "9")},
        "tgaze_9pt_fit_ms_on_device": mean_std([(s.get("calibration_fit_on_device") or {}).get("tgaze_appearance_9pt", {}).get("fit_ms_including_tgaze_LOO") for s in subjects]),
        "live_vs_offline_max_abs_uv": runtime.get("live_vs_offline_max_abs_uv"),
        "platforms": sorted({json.dumps(s["platform"]) for s in subjects}),
    }
    summary = {"subjects": [s["subject_id"] for s in subjects], "any_fake_camera": any(s["fake_camera"] for s in subjects),
               "n_eval_trials": len(rows), "trials_per_subject": {s["subject_id"]: s["n_eval_trials_used"] for s in subjects},
               "status_counts": {s["subject_id"]: s["status_counts"] for s in subjects},
               "calibration_comparison": calcomp, "baselines": baselines,
               "synthetic_to_real_gap_cm": calcomp["C0"]["pooled"]["mean_cm"] - cfg["analysis"]["synthetic_reference"]["M8_identity_cv_mean_cm"],
               "by_subject": breakdown("subject_id", [s["subject_id"] for s in subjects]),
               "by_region": breakdown("region", ["center", "middle", "edge"]),
               "by_head_pose": breakdown("head_pose_bin", [f"small <{b1:g}", f"medium {b1:g}-{b2:g}", f"large >={b2:g}"]),
               "by_pose_instruction": breakdown("pose_instruction", ["NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED"]),
               "pose_invariance": {k: mean_std([p.get(k) for p in pinv]) for k in pkeys}, "jitter": jit, "runtime": runtime_sum,
               "frozen_protocol_hash": sorted(hashes), "model_onnx_sha256": subjects[0]["model_onnx_sha256"],
               "head_pose_by_instruction_deg": {c: {"abs_delta_from_cal": mean_std([r["head_delta_from_cal_deg"] for r in rows if r["pose_instruction"] == c])}
                                                for c in ("NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED")}}
    c0, c1, c9 = (calcomp[f"C{n}"]["pooled"]["mean_cm"] for n in (0, 1, 9))
    summary["indicators"] = {
        "fraction_of_zero_shot_error_removed_by_bias_1pt": 1 - c1 / c0, "fraction_removed_by_affine_9pt": 1 - c9 / c0,
        "residual_9pt_minus_synthetic_cm": c9 - cfg["analysis"]["synthetic_reference"]["M8_identity_cv_mean_cm"],
        "pose_invariance_minus_noise_floor_cm": (summary["pose_invariance"].get("C0_eye_vs_head_cm", [np.nan])[0] or np.nan) - (summary["pose_invariance"].get("C0_natural_retest_cm", [np.nan])[0] or np.nan),
        "mean_bias_cm_per_subject_zero_shot": {s["subject_id"]: (-np.array(s["corrections"]["primary:1"]["b_cm"])).tolist() for s in subjects},
        "affine_gain_9pt_per_subject": {s["subject_id"]: np.diag(np.array(s["corrections"]["primary:9"]["A"])).tolist() for s in subjects}}
    if a.secondary:
        sec = {}
        for k in sorted({k for r in rows for k in r if re.match(r"^C\d+_[a-z]", k) or k.startswith(("smooth_", "dist_on_"))}):
            if k.endswith("_err_cm") or k.endswith("_mm"):
                sec[k] = stats_table(rows, k) if k.endswith("_err_cm") else mean_std([r.get(k) for r in rows])
        sec["reference_C0_err_cm"], sec["reference_C9_err_cm"] = calcomp["C0"]["pooled"], calcomp["C9"]["pooled"]
        sec["reference_raw_jitter_radial_mm"] = jit["jitter_radial_mm"]
        summary["secondary"] = sec

    write_csv(os.path.join(rdir, "trials.csv"), rows)
    write_csv(os.path.join(rdir, "zero_shot.csv"), [{k: r[k] for k in ("subject_id", "trial_id", "block", "pose_instruction", "status", "target_u", "target_v", "region",
                                                                        "C0_u", "C0_v", "C0_err_cm", "C0_err_deg", "head_pitch_deg", "head_yaw_deg", "head_delta_from_cal_deg")} for r in rows])
    write_csv(os.path.join(rdir, "calibration_comparison.csv"), [{"calibration": f"{n}-point", **calcomp[f"C{n}"]["pooled"],
                                                                   "subject_mean_cm": calcomp[f"C{n}"]["subject_mean_cm_mean_std"][0],
                                                                   "subject_std_cm": calcomp[f"C{n}"]["subject_mean_cm_mean_std"][1]} for n in CAL_LEVELS]
              + [{"calibration": b, **v["pooled"], "subject_mean_cm": v["subject_mean_cm_mean_std"][0], "subject_std_cm": v["subject_mean_cm_mean_std"][1]} for b, v in baselines.items()])
    write_csv(os.path.join(rdir, "subject_summary.csv"), [{"subject_id": s["subject_id"], "calibration": f"{n}-point", **summary["by_subject"][s["subject_id"]][f"C{n}"],
                                                            "status_counts": json.dumps(s["status_counts"])} for s in subjects for n in CAL_LEVELS])
    write_csv(os.path.join(rdir, "pose_invariance.csv"), pinv)
    write_csv(os.path.join(rdir, "jitter.csv"), [{k: r[k] for k in ("subject_id", "trial_id", "pose_instruction", "jitter_x_mm", "jitter_y_mm", "jitter_radial_mm",
                                                                     "spread_median_radius_mm", "spread_p95_radius_mm")} for r in rows])
    json.dump(summary, open(os.path.join(rdir, "summary.json"), "w"), indent=1, default=C.json_default)
    json.dump(subjects, open(os.path.join(rdir, "subjects_detail.json"), "w"), indent=1, default=C.json_default)
    make_plots(pdir, rows, subjects, calcomp, pinv, cfg, base_names)
    write_report(os.path.join(rdir, "report.md"), summary, cfg)
    rt.close()
    print(open(os.path.join(rdir, "report.md")).read()[:3000])


if __name__ == "__main__":
    main()
