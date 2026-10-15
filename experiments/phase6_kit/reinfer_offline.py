"""Offline re-inference of recorded Phase 6 sessions with alternative M8 models (secondary analysis only).

Uses only what collect.py saved per trial (lms, R_head, uv_m8_raw) and subject.json (the K used live).
No camera, no MediaPipe, CPU-only ONNX Runtime. The frozen model (model/) is always included as the reference,
and its offline prediction is checked against the live one (uv_m8_raw).

  python reinfer_offline.py outputs/real_mac_validation                       # frozen + every models_alt/* model
  python reinfer_offline.py outputs/real_mac_validation --models models_alt/m8_no_rich16d_angles --subjects subject_00

Writes <root>/reinfer/trials.csv (one row per evaluation trial and model), cal_trials.csv (calibration-trial medians per model,
input of cal_variants.py) and <root>/reinfer/summary.md / summary.json
(mean error per pose instruction x calibration points 0/1/3/5/9, primary correction of config.json, the same
calibration pairs as analyze.py). Nothing under <root>/subject_* is modified.
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
import m8_cpu  # noqa: E402
from analyze import load_subject, trial_pred, usable  # noqa: E402

LEVELS = (0, 1, 3, 5, 9)
CONDS = ("NATURAL", "EYE_DOMINANT", "HEAD_ALLOWED")


def kabsch_deg(R):
    """per-frame Kabsch R_head (N,3,3) -> (N,3) degrees [pitch, yaw, roll], NaN rows kept."""
    out = np.full((len(R), 3), np.nan)
    ok = np.isfinite(R).all(axis=(1, 2))
    if ok.any():
        out[ok] = np.degrees(m8_cpu.kabsch_euler(R[ok]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--models", nargs="*", default=None, help="model dirs relative to the kit (default: models_alt/*)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = C.load_config()
    tgaze = C.tgaze_dir(cfg)
    models = ["model"] + (a.models if a.models is not None else
                          sorted(os.path.relpath(d, KIT_DIR) for d in glob.glob(os.path.join(KIT_DIR, "models_alt", "*"))
                                 if os.path.exists(os.path.join(d, "final_cpu_model.onnx"))))
    out_dir = a.out or os.path.join(a.root, "reinfer")
    os.makedirs(out_dir, exist_ok=True)
    sdirs = sorted(d for d in glob.glob(os.path.join(a.root, "subject_*")) if os.path.isdir(d))
    if a.subjects:
        sdirs = [d for d in sdirs if os.path.basename(d) in a.subjects]
    rows, cal_rows, summary = [], [], {"models": {}, "subjects": [os.path.basename(d) for d in sdirs], "config_frozen_hash": C.frozen_config_hash(cfg)}
    for sdir in sdirs:
        sid = os.path.basename(sdir)
        info, _, trials, _ = load_subject(sdir, cfg)
        K_raw = np.array(info["camera_matrix"]["K"], np.float64)
        cal = [t for t in trials if t["session"] == "calibration" and usable(t, cfg)]
        ev = [t for t in trials if t["session"] == "evaluation" and usable(t, cfg)]
        for mdir in models:
            m = m8_cpu.OfflineM8(mdir, K_raw, tgaze)
            name = os.path.basename(os.path.normpath(mdir))
            pred = {}
            for t in cal + ev:
                z = t["z"]
                uv, ok = m.predict_lms(z["lms"][t["rec_idx"]],
                                       z["eye_patch"][t["rec_idx"]] if m.eye_pca is not None and "eye_patch" in z else None)
                pred[id(t)] = (uv[ok], ok)
            live_diff = max(float(np.nanmax(np.abs(pred[id(t)][0] - t["z"]["uv_m8_raw"][t["rec_idx"]][pred[id(t)][1]])))
                            for t in ev[:10]) if mdir == "model" and ev else None
            cal_pairs = [(tuple(t["target"]), tuple(trial_pred(pred[id(t)][0]))) for t in cal if len(pred[id(t)][0])]
            cal_rows += [{"subject_id": sid, "model": name, "target_u": tu, "target_v": tv, "raw_u": pu, "raw_v": pv}
                         for (tu, tv), (pu, pv) in cal_pairs]
            corr = {n: C.fit_correction(cfg, n, cal_pairs) for n in LEVELS}
            ref = np.nanmedian(np.vstack([kabsch_deg(t["z"]["R_head"][t["rec_idx"]]) for t in cal]), 0) if cal else np.zeros(3)
            for t in ev:
                uv = pred[id(t)][0]
                if not len(uv):
                    continue
                raw = trial_pred(uv)
                kd = np.nanmedian(kabsch_deg(t["z"]["R_head"][t["rec_idx"]]), 0)
                rd = np.nanmedian(np.degrees(t["z"]["f16"][t["rec_idx"]][:, [4, 5, 7]].astype(np.float64)), 0)
                r = {"subject_id": sid, "model": name, "trial_id": t["rec"]["trial_id"], "block": t["block"], "pose_instruction": t["cond"],
                     "target_u": t["target"][0], "target_v": t["target"][1], "raw_u": raw[0], "raw_v": raw[1],
                     "kabsch_pitch_deg": kd[0], "kabsch_yaw_deg": kd[1], "kabsch_roll_deg": kd[2],
                     "kabsch_delta_from_cal_deg": float(np.hypot(*(kd[:2] - ref[:2]))),
                     "rich16d_pitch_deg": rd[0], "rich16d_yaw_deg": rd[1], "rich16d_roll_deg": rd[2]}
                for n, c in corr.items():
                    r[f"C{n}_err_cm"] = float(C.cm_err(c.apply(raw), t["target"]))
                rows.append(r)
            ms = summary["models"].setdefault(name, {"dir": mdir, "onnx_sha256": C.sha256_file(m.onnx_path), "head_angles": m.head_angles,
                                                     "live_vs_offline_max_abs_uv": {}})
            if live_diff is not None:
                ms["live_vs_offline_max_abs_uv"][sid] = live_diff
            print(f"[reinfer] {sid} {name}: {sum(1 for r in rows if r['subject_id'] == sid and r['model'] == name)} trials"
                  + (f", live-vs-offline max |uv| {live_diff:.2e}" if live_diff is not None else ""), flush=True)
    if not rows:
        print("no trials")
        return
    with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "cal_trials.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    lines = ["# Offline re-inference (secondary analysis; frozen M8 = model)", "",
             f"subjects: {', '.join(summary['subjects'])}; mean error [cm] of evaluation trials (median of record frames), "
             "primary correction of config.json fitted on each model's own calibration-session predictions", ""]
    for name in summary["models"]:
        tab = {}
        for cond in CONDS + ("ALL",):
            sel = [r for r in rows if r["model"] == name and (cond == "ALL" or r["pose_instruction"] == cond)]
            tab[cond] = {f"C{n}": float(np.mean([r[f"C{n}_err_cm"] for r in sel])) if sel else None for n in LEVELS} | {"n": len(sel)}
        summary["models"][name]["mean_err_cm"] = tab
        lines += [f"## {name} (head_angles = {summary['models'][name]['head_angles']})", "",
                  "| points | " + " | ".join(CONDS) + " | all |", "|---|" + "---:|" * (len(CONDS) + 1)]
        for n in LEVELS:
            lines.append(f"| {n} | " + " | ".join(f"{tab[c][f'C{n}']:.2f}" if tab[c][f"C{n}"] is not None else "-" for c in CONDS + ("ALL",)) + " |")
        lines.append("")
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=C.json_default)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
