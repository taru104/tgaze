"""CPU-only M8 runtime for live webcam frames (Phase 6). No torch, no GPU, no CoreML.

  raw BGR frame (1920x1080) -> cv2.flip(frame, 1)                           [same as tgaze main.py]
  -> MediaPipe FaceLandmarker (Delegate.CPU, VIDEO mode like tgaze)         -> 478 landmarks
  -> rich16d (tgaze, set_camera_matrix(K, mirrored=True) like main_camK_mac) + head/eye landmark features
  -> ONNX Runtime, providers=["CPUExecutionProvider"] only                   -> display (u, v)
  The eye patch of tgaze's appearance.py is computed too, so the current tgaze model can be fitted on
  exactly the same frames.

The M8 weights are the frozen Phase 5 file (sha256 recorded in every session); nothing here trains.
"""
import json
import os
import sys
import time

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KIT_DIR, "lib"))
import landmark_features as lf  # noqa: E402

R16_EYE = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
L_IRIS, R_IRIS = list(range(468, 473)), list(range(473, 478))


def kabsch_euler(R):
    """(N,3,3) camera<-head Kabsch rotation -> (N,3) [pitch, yaw, roll] radians, R = Rx(pitch) Ry(yaw) Rz(roll).
    Same sign convention as rich16d (cv2.RQDecomp3x3) for near-frontal heads; within 0.3 deg of it in yaw."""
    R = np.asarray(R, np.float64)
    yaw = np.arcsin(np.clip(R[:, 0, 2], -1, 1))
    pitch = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
    roll = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
    return np.stack([pitch, yaw, roll], 1)


def build_inputs(lms, f16, K_mir, w, h, template, head_angles="rich16d"):
    """lms (N,478,3) mirrored-frame MediaPipe coords, f16 (N,16) rich16d -> (ONNX feature dict, fe).

    head_pose = Kabsch R_head 2 columns (6) + rigid centroid xy (2) + log s (1) + 3 angles:
      rich16d : tgaze 6-point solvePnP pitch/yaw/roll (frozen Phase 5 M8, 12D)
      none    : no angles (9D)
      kabsch  : Euler angles of R_head (12D)"""
    f16 = np.asarray(f16, np.float64)
    fe = lf.features(lf.to_ncam(lms, K_mir, w, h), template, f16[:, [4, 5, 7]])
    head9 = fe["head"][:, :9]
    if head_angles == "rich16d":
        head = np.hstack([head9, fe["head"][:, -3:]])
    elif head_angles == "none":
        head = head9
    elif head_angles == "kabsch":
        head = np.hstack([head9, kabsch_euler(fe["R_head"])])
    else:
        raise ValueError(head_angles)
    feats = {"eye_norm_r16": np.hstack([fe["eye_norm"], f16[:, R16_EYE]]).astype(np.float32), "head_pose": head.astype(np.float32)}
    return feats, fe


def load_camera_matrix(cfg, tgaze, allow_host_mismatch=False):
    """Measured Mac K from tgaze, applied under the same host rule as tgaze main_camK_mac.py."""
    path = os.path.join(tgaze, cfg["paths"]["camera_matrix_json"])
    calib = json.load(open(path, encoding="utf-8"))
    info = {"path": path, "host_in_json": calib.get("host"), "image_size": calib["image_size"],
            "K": calib["K"], "dist_k1_k2_p1_p2_k3": calib.get("dist_k1_k2_p1_p2_k3")}
    try:
        import macos
        info["this_machine"] = macos.machine_id()
    except Exception as e:
        info["this_machine"] = None
        info["machine_id_error"] = repr(e)
    info["host_match"] = info["this_machine"] == info["host_in_json"]
    if not info["host_match"] and cfg["camera"]["require_k_host_match"] and not allow_host_mismatch:
        raise RuntimeError(f"camera_matrix.json was measured on {info['host_in_json']!r}, this is {info['this_machine']!r}. "
                           "tgaze would not apply this K here (main_camK_mac.py rule). Use --allow-k-host-mismatch only for testing.")
    if tuple(calib["image_size"]) != (cfg["camera"]["width"], cfg["camera"]["height"]):
        raise RuntimeError(f"K is for {calib['image_size']}, camera config is {cfg['camera']['width']}x{cfg['camera']['height']}")
    return np.array(calib["K"], np.float64), info


class M8Runtime:
    def __init__(self, cfg, tgaze, K_raw, video_mode=None, dist=None, model_dir=None):
        assert "torch" not in sys.modules, "torch must not be loaded in the CPU runtime"
        import cv2
        import mediapipe as mp
        import onnxruntime as ort
        import rich16d
        from appearance import eye_patch
        self.cv2, self.mp, self.rich16d, self.eye_patch = cv2, mp, rich16d, eye_patch
        mdir = os.path.join(KIT_DIR, model_dir or cfg["paths"]["model_dir"])
        self.prep = json.load(open(os.path.join(mdir, "final_model_preprocessing.json")))
        self.head_angles = self.prep.get("head_angles", "rich16d")      # frozen Phase 5 M8 file has no key -> rich16d
        self.onnx_path = os.path.join(mdir, "final_cpu_model.onnx")
        self.w, self.h = cfg["camera"]["width"], cfg["camera"]["height"]
        assert [self.w, self.h] == self.prep["image_size"], (self.prep["image_size"], self.w, self.h)

        self.K_raw = np.array(K_raw, np.float64)
        self.K_mir = lf.mirrored_K(self.K_raw, self.w)
        k_diff = float(np.abs(self.K_mir - np.array(self.prep["K_mirrored"])).max())
        self.k_matches_training = k_diff < 0.5
        self.k_max_abs_diff_vs_training_px = k_diff
        rich16d.set_camera_matrix(self.K_raw, (self.w, self.h), mirrored=True)   # == main_camK_mac.py

        so = ort.SessionOptions()
        if cfg["camera"]["ort_intra_op_threads"]:
            so.intra_op_num_threads = int(cfg["camera"]["ort_intra_op_threads"])
        self.sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])
        self.ort_providers = self.sess.get_providers()
        self.ort_available = ort.get_available_providers()
        assert self.ort_providers == ["CPUExecutionProvider"], self.ort_providers

        self.template = np.array(self.prep["template_rigid"])
        self.video = (video_mode or cfg["camera"]["mediapipe_running_mode"]).upper() == "VIDEO"
        conf = cfg["camera"]["mediapipe_min_confidence"]
        task = os.path.join(tgaze, cfg["paths"]["face_landmarker_task"])
        V = mp.tasks.vision
        self.lm = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task, delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=V.RunningMode.VIDEO if self.video else V.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=conf, min_face_presence_confidence=conf, min_tracking_confidence=conf))
        self._last_ts = -1
        self.ipd = cfg["analysis"]["assumed_ipd_m"]
        d = self.prep["display"]
        self.disp = {k: np.array(d[k]) for k in ("center_cam_m", "right_cam_m", "up_cam_m")}

    def describe(self):
        return {"onnx_path": self.onnx_path, "ort_session_providers": self.ort_providers,
                "ort_available_providers": self.ort_available, "mediapipe_delegate": "CPU",
                "mediapipe_running_mode": "VIDEO" if self.video else "IMAGE", "mirrored": True,
                "rich16d_set_camera_matrix": "set_camera_matrix(K_raw, (1920,1080), mirrored=True)",
                "K_raw": self.K_raw.tolist(), "K_mirrored": self.K_mir.tolist(),
                "K_matches_training": self.k_matches_training, "K_max_abs_diff_vs_training_px": self.k_max_abs_diff_vs_training_px,
                "torch_loaded": "torch" in sys.modules}

    # ---------------------------------------------------------------- per-frame
    def landmarks(self, mirrored_bgr, ts_ms):
        cv2, mp = self.cv2, self.mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(mirrored_bgr, cv2.COLOR_BGR2RGB)))
        if self.video:
            ts_ms = max(int(ts_ms), self._last_ts + 1)
            self._last_ts = ts_ms
            res = self.lm.detect_for_video(img, ts_ms)
        else:
            res = self.lm.detect(img)
        if not res.face_landmarks or len(res.face_landmarks[0]) < 478:
            return None, None
        return self.rich16d.lms_to_array(res.face_landmarks[0]), res.face_landmarks[0]

    def features_from_lms(self, lms, K_mir=None):
        """lms (478,3) normalized coords in the MIRRORED frame -> ONNX inputs, rich16d, head R."""
        K_mir = self.K_mir if K_mir is None else K_mir
        f16 = self.rich16d.rich_16d_from_lms(lms, self.w, self.h)
        if f16 is None:
            return None
        f16 = np.asarray(f16, np.float64)[None]
        feats, fe = build_inputs(lms[None], f16, K_mir, self.w, self.h, self.template, self.head_angles)
        return {"inputs": {b: feats[b] for b in self.prep["branches"]}, "f16": f16[0].astype(np.float32), "R_head": fe["R_head"][0]}

    def infer(self, inputs):
        y = self.sess.run(["y"], inputs)[0].astype(np.float64)
        assert self.prep["output"] == "screen"
        return y[0, :2]

    def eye_position_cam(self, lms, f16, K_mir=None):
        """Cyclopean eye position (m) in the mirrored camera frame from iris spacing and an assumed IPD."""
        K = self.K_mir if K_mir is None else K_mir
        L = lms[L_IRIS, :2].mean(0) * (self.w, self.h)
        R = lms[R_IRIS, :2].mean(0) * (self.w, self.h)
        ipd_px = float(np.linalg.norm(L - R))
        z = K[0, 0] * self.ipd * max(np.cos(float(f16[5])), 0.5) / max(ipd_px, 1e-6)
        m = (L + R) / 2
        return np.array([(m[0] - K[0, 2]) / K[0, 0] * z, (m[1] - K[1, 2]) / K[1, 1] * z, z])

    def process(self, frame_bgr, ts_ms, want_patch=True):
        """Full live pipeline for one raw camera frame. Returns a dict (ok False if no face)."""
        t0 = time.perf_counter()
        mir = self.cv2.flip(frame_bgr, 1)
        t1 = time.perf_counter()
        lms, mp_lms = self.landmarks(mir, ts_ms)
        t2 = time.perf_counter()
        out = {"ok": False, "lat_ms": {}}
        if lms is not None:
            fe = self.features_from_lms(lms)
            t3 = time.perf_counter()
            if fe is not None:
                uv = self.infer(fe["inputs"])
                t4 = time.perf_counter()
                patch = self.eye_patch(mir, mp_lms) if want_patch else None
                t5 = time.perf_counter()
                out = {"ok": True, "lms": lms.astype(np.float32), "f16": fe["f16"], "R_head": fe["R_head"].astype(np.float32),
                       "uv": uv, "eye_cam_m": self.eye_position_cam(lms, fe["f16"]),
                       "patch": None if patch is None else patch.astype(np.float16),
                       "lat_ms": {"flip": 1e3 * (t1 - t0), "mediapipe": 1e3 * (t2 - t1), "features": 1e3 * (t3 - t2),
                                  "onnx": 1e3 * (t4 - t3), "eye_patch": 1e3 * (t5 - t4), "total": 1e3 * (t5 - t0),
                                  "total_m8_only": 1e3 * (t4 - t0)}}
                return out
        out["lat_ms"] = {"flip": 1e3 * (t1 - t0), "mediapipe": 1e3 * (t2 - t1), "total": 1e3 * (time.perf_counter() - t0)}
        return out

    def angular_error_deg(self, eye_cam, pred_uv, true_uv):
        def pt(uv):
            return self.disp["center_cam_m"] + uv[0] * 0.1453 * self.disp["right_cam_m"] + uv[1] * 0.0945 * self.disp["up_cam_m"]
        a, b = pt(pred_uv) - eye_cam, pt(true_uv) - eye_cam
        return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))

    def close(self):
        self.lm.close()


class OfflineM8:
    """Re-inference from saved landmarks (no camera, no MediaPipe): any model directory with
    final_cpu_model.onnx + final_model_preprocessing.json (frozen M8 or an alternative in models_alt/).

        m = OfflineM8("models_alt/m8_no_rich16d_angles", K_raw, tgaze_dir)   # K_raw: subject.json camera_matrix.K
        uv, ok = m.predict_lms(z["lms"])                                    # z = np.load(trial npz); uv (N,2) display uv
    """

    def __init__(self, model_dir, K_raw, tgaze, w=1920, h=1080, threads=0):
        assert "torch" not in sys.modules, "torch must not be loaded in the CPU runtime"
        if tgaze not in sys.path:
            sys.path.insert(0, tgaze)
        import onnxruntime as ort
        import rich16d
        mdir = model_dir if os.path.isabs(model_dir) else os.path.join(KIT_DIR, model_dir)
        self.prep = json.load(open(os.path.join(mdir, "final_model_preprocessing.json")))
        self.onnx_path = os.path.join(mdir, "final_cpu_model.onnx")
        self.head_angles = self.prep.get("head_angles", "rich16d")
        self.w, self.h = w, h
        self.K_raw = np.array(K_raw, np.float64)
        self.K_mir = lf.mirrored_K(self.K_raw, w)
        self.template = np.array(self.prep["template_rigid"])
        self.rich16d = rich16d
        rich16d.set_camera_matrix(self.K_raw, (w, h), mirrored=True)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])
        assert self.sess.get_providers() == ["CPUExecutionProvider"]

    def predict_lms(self, lms):
        """lms (N,478,3) (frames without a face may be NaN) -> uv (N,2) with NaN where no prediction, ok (N,)."""
        lms = np.asarray(lms, np.float64)
        uv = np.full((len(lms), 2), np.nan)
        ok = np.zeros(len(lms), bool)
        f16 = np.zeros((len(lms), 16))
        for i, l in enumerate(lms):
            if not np.isfinite(l).all():
                continue
            f = self.rich16d.rich_16d_from_lms(l, self.w, self.h)
            if f is not None:
                f16[i], ok[i] = f, True
        if ok.any():
            feats, _ = build_inputs(lms[ok], f16[ok], self.K_mir, self.w, self.h, self.template, self.head_angles)
            uv[ok] = self.sess.run(["y"], {b: feats[b] for b in self.prep["branches"]})[0][:, :2]
        return uv, ok


def undistort_lms(lms, K_raw, dist, w, h):
    """Secondary experiment: undistort landmark pixel positions (mirrored frame) with the measured distortion."""
    import cv2
    p = lms[:, :2].astype(np.float64) * (w, h)
    p[:, 0] = w - p[:, 0]                                     # back to raw (unmirrored) pixels
    u = cv2.undistortPoints(p.reshape(-1, 1, 2), K_raw, np.asarray(dist, np.float64), P=K_raw).reshape(-1, 2)
    u[:, 0] = w - u[:, 0]
    out = lms.astype(np.float64).copy()
    out[:, 0], out[:, 1] = u[:, 0] / w, u[:, 1] / h
    return out
