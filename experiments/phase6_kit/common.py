"""Shared pieces of the Phase 6 real-Mac validation kit (collection app + analysis).

Coordinate conventions (identical to Phase 5 / M8):
  display uv  : u, v in [-1, 1] over the ACTIVE display (290.6 x 189.0 mm, notch band included),
                u > 0 = right as the user sees it, v > 0 = up, (0, 0) = display centre.
  display cm  : x_cm = u * 14.53, y_cm = v * 9.45 (centre origin, right / up).
  tgaze norm  : tx = (u + 1) / 2, ty = (1 - v) / 2   (tgaze CALIB_POINTS_9 convention, y down).
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_W_MM, DISPLAY_H_MM = 290.6, 189.0          # measured active display (macOS CGDisplayScreenSize)
HALF_CM = np.array([DISPLAY_W_MM / 20.0, DISPLAY_H_MM / 20.0])
SUBJECT_RE = re.compile(r"^subject_\d{2,3}$")


def load_config(path=None):
    path = path or os.path.join(KIT_DIR, "config.json")
    raw = open(path, "rb").read()
    cfg = json.loads(raw)
    cfg["_path"] = os.path.abspath(path)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def frozen_config_hash(cfg):
    """Hash of the parts that must not change between main subjects (protocol + calibration models)."""
    keys = ("timing", "grids", "calibration", "evaluation", "quality", "analysis")
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tgaze_dir(cfg):
    d = os.path.expanduser(os.environ.get("TGAZE_DIR") or cfg["paths"]["tgaze_dir"])
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def uv_to_cm(uv):
    return np.asarray(uv, np.float64) * HALF_CM


def cm_err(pred_uv, true_uv):
    return np.linalg.norm(uv_to_cm(pred_uv) - uv_to_cm(true_uv), axis=-1)


def uv_to_tgaze(uv):
    uv = np.asarray(uv, np.float64)
    return np.stack([(uv[..., 0] + 1) / 2, (1 - uv[..., 1]) / 2], -1)


def tgaze_to_uv(t):
    t = np.asarray(t, np.float64)
    return np.stack([2 * t[..., 0] - 1, 1 - 2 * t[..., 1]], -1)


def screen_region(uv):
    """Chebyshev ring of the 5x5 evaluation grid: centre (0,0) / middle (|.|<=0.4) / edge (0.8)."""
    r = np.max(np.abs(np.asarray(uv)), axis=-1)
    return np.where(r < 0.2, "center", np.where(r < 0.6, "middle", "edge"))


# --------------------------------------------------------------------------- screen layout
class ScreenLayout:
    """Maps display uv <-> pixels of the full-screen canvas.

    The panel is screen_pt (logical points, whole active display). The full-screen window on a notched
    MacBook starts below the notch band (safe-area top inset), so window_pt = (x0, y0, w, h) in points.
    """

    def __init__(self, screen_pt, window_pt, canvas_px):
        self.sw, self.sh = screen_pt
        self.x0, self.y0, self.ww, self.wh = window_pt
        self.cw, self.ch = canvas_px

    @classmethod
    def detect(cls, canvas_scale=1.0):
        info = {"source": "fallback (MacBook Air 13 M4 default scaling)"}
        sw, sh, top = 1470, 956, 32
        try:
            import AppKit
            scr = AppKit.NSScreen.mainScreen()
            sw, sh = int(scr.frame().size.width), int(scr.frame().size.height)
            top = int(round(scr.safeAreaInsets().top))
            info = {"source": "NSScreen.mainScreen", "backing_scale": float(scr.backingScaleFactor()),
                    "n_screens": len(AppKit.NSScreen.screens())}
        except Exception as e:  # not macOS / no pyobjc
            info["error"] = repr(e)
        lay = cls((sw, sh), (0, top, sw, sh - top), (int(round(sw * canvas_scale)), int(round((sh - top) * canvas_scale))))
        info.update(lay.describe())
        return lay, info

    def describe(self):
        return {"screen_pt": [self.sw, self.sh], "window_pt": [self.x0, self.y0, self.ww, self.wh],
                "canvas_px": [self.cw, self.ch], "display_mm": [DISPLAY_W_MM, DISPLAY_H_MM],
                "mm_per_pt": [DISPLAY_W_MM / self.sw, DISPLAY_H_MM / self.sh]}

    def uv_to_canvas(self, uv):
        u, v = uv
        x_pt = (u + 1) / 2 * self.sw
        y_pt = (1 - v) / 2 * self.sh
        return ((x_pt - self.x0) * self.cw / self.ww, (y_pt - self.y0) * self.ch / self.wh)

    def canvas_to_uv(self, xy):
        x_pt = xy[0] * self.ww / self.cw + self.x0
        y_pt = xy[1] * self.wh / self.ch + self.y0
        return (2 * x_pt / self.sw - 1, 1 - 2 * y_pt / self.sh)

    def px_per_mm(self):
        return self.cw / self.ww * self.sw / DISPLAY_W_MM

    def visible(self, uv, margin_px=0):
        x, y = self.uv_to_canvas(uv)
        return margin_px <= x <= self.cw - margin_px and margin_px <= y <= self.ch - margin_px


# --------------------------------------------------------------------------- calibration models
class Correction:
    """Personal correction in display-cm space: corrected = A p + b (or polynomial). Fit on trial medians."""

    def __init__(self, kind, lam=0.0):
        self.kind, self.lam = kind, float(lam)
        self.A, self.b, self.W = np.eye(2), np.zeros(2), None

    def _poly(self, P):
        x, y = P[:, 0], P[:, 1]
        return np.column_stack([x * x, x * y, y * y]) / 10.0      # cm^2 / 10: comparable scale to cm

    def fit(self, pred_uv, true_uv):
        P, T = uv_to_cm(pred_uv), uv_to_cm(true_uv)
        R = T - P
        if self.kind == "none":
            pass
        elif self.kind == "bias":
            self.b = R.mean(0)
        elif self.kind in ("bias_scale", "affine"):
            pm, rm = P.mean(0), R.mean(0)
            Pc, Rc = P - pm, R - rm
            if self.kind == "bias_scale":
                D = np.diag([(Pc[:, i] @ Rc[:, i]) / (Pc[:, i] @ Pc[:, i] + self.lam) for i in range(2)])
            else:
                D = Rc.T @ Pc @ np.linalg.inv(Pc.T @ Pc + self.lam * np.eye(2))
            self.A = np.eye(2) + D
            self.b = rm - D @ pm
        elif self.kind == "poly2":
            Q = self._poly(P)
            X = np.hstack([P, Q])
            xm, rm = X.mean(0), R.mean(0)
            Xc, Rc = X - xm, R - rm
            pen = np.diag([self.lam, self.lam, 4 * self.lam, 4 * self.lam, 4 * self.lam])
            D = Rc.T @ Xc @ np.linalg.inv(Xc.T @ Xc + pen)
            self.A = np.eye(2) + D[:, :2]
            self.W = D[:, 2:]
            self.b = rm - D @ xm
        else:
            raise ValueError(self.kind)
        return self

    def apply(self, pred_uv):
        P = uv_to_cm(np.atleast_2d(pred_uv))
        out = P @ self.A.T + self.b
        if self.W is not None:
            out = out + self._poly(P) @ self.W.T
        return (out / HALF_CM).reshape(np.shape(pred_uv))

    def to_json(self):
        return {"kind": self.kind, "lambda_cm2": self.lam, "A": self.A.tolist(), "b_cm": self.b.tolist(),
                "W_poly": None if self.W is None else self.W.tolist()}


def calibration_points(cfg, n):
    return [tuple(p) for p in cfg["calibration"]["point_sets"][str(n)]]


def select_points(cal_trials, points, tol=1e-6):
    """cal_trials: list of (target_uv, pred_uv). Keep trials whose target is one of `points`."""
    pts = np.array(points)
    keep = [(t, p) for t, p in cal_trials if np.min(np.abs(pts - np.asarray(t)).max(1)) < tol]
    return np.array([t for t, _ in keep]).reshape(-1, 2), np.array([p for _, p in keep]).reshape(-1, 2)


def fit_correction(cfg, n_points, cal_trials, variant="primary"):
    """Pre-registered correction for an n-point calibration (config.calibration.models)."""
    spec = cfg["calibration"]["models"][str(n_points)][variant]
    if n_points == 0:
        return Correction("none")
    T, P = select_points(cal_trials, calibration_points(cfg, n_points))
    if len(T) == 0:
        raise ValueError(f"no usable calibration trials for the {n_points}-point set")
    return Correction(spec["kind"], spec.get("lambda_cm2", 0.0)).fit(P, T)


# --------------------------------------------------------------------------- metrics
def error_stats(err_cm, ang_deg=None):
    e = np.asarray(err_cm, np.float64)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"n": 0}
    out = {"n": int(len(e)), "mean_cm": float(e.mean()), "median_cm": float(np.median(e)),
           "rmse_cm": float(np.sqrt((e ** 2).mean())), "p50_cm": float(np.percentile(e, 50)),
           "p90_cm": float(np.percentile(e, 90)), "p95_cm": float(np.percentile(e, 95))}
    if ang_deg is not None:
        a = np.asarray(ang_deg, np.float64)
        a = a[np.isfinite(a)]
        out["mean_deg"] = float(a.mean()) if len(a) else None
    return out


# --------------------------------------------------------------------------- trial quality (never uses the target)
def trial_quality(rec_ok, rec_uv, rec_pyr_rad, cfg):
    """rec_ok (n,), rec_uv (n,2), rec_pyr_rad (n,3) over the recording window -> (status, flags dict).

    Uses only face detection, head-pose stability and prediction stability, never the ground-truth target,
    so retakes cannot bias the accuracy."""
    q = cfg["quality"]
    n = len(rec_ok)
    ok = np.asarray(rec_ok, bool)
    flags = {"n_record_frames": int(n), "n_face_ok": int(ok.sum()),
             "face_loss_fraction": float(1 - ok.mean()) if n else 1.0}
    reasons_invalid, reasons_retake = [], []
    if ok.sum() < q["min_record_frames"]:
        reasons_invalid.append("too_few_frames_with_face")
    if flags["face_loss_fraction"] > q["face_loss_invalid_fraction"]:
        reasons_invalid.append("face_detection_failure")
    elif flags["face_loss_fraction"] > q["face_loss_retake_fraction"]:
        reasons_retake.append("landmark_loss")
    if ok.sum() >= 2:
        uv = np.asarray(rec_uv)[ok]
        pyr = np.degrees(np.asarray(rec_pyr_rad)[ok])
        motion = float(np.max(pyr[:, :2].max(0) - pyr[:, :2].min(0)))
        cm = uv_to_cm(uv)
        jump = float(np.max(np.linalg.norm(np.diff(cm, axis=0), axis=1)))
        spread = float(np.median(np.linalg.norm(cm - np.median(cm, 0), axis=1)))
        flags.update({"head_motion_deg": motion, "max_pred_jump_cm": jump, "pred_spread_cm": spread})
        if motion > q["head_motion_invalid_deg"]:
            reasons_invalid.append("large_head_movement")
        elif motion > q["head_motion_retake_deg"]:
            reasons_retake.append("head_movement")
        if jump > q["pred_jump_retake_cm"]:
            reasons_retake.append("extreme_prediction_jump")
        if spread > q["pred_spread_retake_cm"]:
            reasons_retake.append("unstable_prediction")
    status = "INVALID" if reasons_invalid else ("RETAKE_RECOMMENDED" if reasons_retake else "VALID")
    flags["reasons"] = reasons_invalid + reasons_retake
    return status, flags


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
