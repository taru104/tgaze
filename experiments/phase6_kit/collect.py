"""Phase 6 real-Mac data collection app (MacBook Air 13" M4, built-in camera 1920x1080, CPU only).

  python collect.py --subject subject_01            # main subject
  python collect.py --subject subject_00 --pilot    # pilot (settings may still be changed after it)
  python collect.py --next-subject                  # first unused subject_XX

Menu keys:  G geometry check | D direction sanity check | L live cursor | C calibration session
            E evaluation session | P CPU benchmark | Q / Esc quit
During a session: Esc pauses (Space continue, R retake previous trial, B back to menu; progress is saved).

Test without a Mac / webcam (feeds Phase 5 synthetic test-identity images that match the shown target):
  python collect.py --subject subject_90 --pilot --fake-camera ../outputs/gaze_dataset_v2 --fake-identity 7 \
                    --headless --auto G,Y,D,C,E,P,Q --allow-k-host-mismatch --out /tmp/x
"""
import argparse
import datetime as dt
import glob
import json
import os
import platform
import queue
import random
import sys
import threading
import time

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT_DIR)
import common as C  # noqa: E402

ESC, SPACE = 27, 32
PHASES = ("move", "settle", "record")


# =========================================================================== camera
class Camera:
    """Background capture thread: keeps the newest frame + timestamp and measures the camera FPS."""

    def __init__(self, index, width, height):
        import cv2
        self.cv2 = cv2
        self.center_stage_off = None
        if sys.platform == "darwin":
            try:
                import macos
                self.center_stage_off = macos.disable_center_stage()
            except Exception as e:
                print(f"[camera] could not disable Center Stage: {e!r}")
        api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, fr = self.cap.read()
        if not ok or fr.shape[1] != width or fr.shape[0] != height:
            raise RuntimeError(f"requested {width}x{height}, camera gives {None if not ok else fr.shape[1::-1]}")
        self.size = (width, height)
        self._lock = threading.Condition()
        self._frame, self._ts, self._idx = fr, time.monotonic(), 0
        self._stamps = []
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ok, fr = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame, self._ts, self._idx = fr, t, self._idx + 1
                self._stamps.append(t)
                if len(self._stamps) > 300:
                    del self._stamps[:150]
                self._lock.notify_all()

    def latest(self, newer_than=-1, timeout=0.2):
        with self._lock:
            if self._idx <= newer_than:
                self._lock.wait(timeout)
            return self._frame, self._ts, self._idx

    def fps(self, window_s=3.0):
        with self._lock:
            s = [t for t in self._stamps if t > self._stamps[-1] - window_s] if self._stamps else []
        return (len(s) - 1) / (s[-1] - s[0]) if len(s) > 2 else float("nan")

    def describe(self):
        cv2 = self.cv2
        return {"size": list(self.size), "reported_fps_property": float(self.cap.get(cv2.CAP_PROP_FPS)),
                "backend": self.cap.getBackendName(), "center_stage_disabled": self.center_stage_off}

    def close(self):
        self._stop = True
        time.sleep(0.05)
        self.cap.release()


class FakeCamera(Camera):
    """TEST ONLY. Serves Phase 5 synthetic images of one (unseen) identity whose target is nearest to the
    target currently shown, with per-frame pixel noise, at ~30 FPS. Lets the whole app + analysis run
    end to end without a Mac. Never used for reported real-data numbers."""

    MODES = {"NATURAL": None, "EYE_DOMINANT": ("EYE_ONLY",), "HEAD_ALLOWED": ("HEAD_FOLLOW", "NECK_HEAD_FOLLOW")}

    def __init__(self, dataset_root, identity, stim, width=1920, height=1080, fps=30.0, noise=3.0, seed=0):
        import cv2
        self.cv2 = cv2
        d = os.path.join(dataset_root, f"identity_{identity:02d}")
        self.recs = [json.loads(l) for l in open(os.path.join(d, "metadata.jsonl"))]
        self.img_dir = os.path.join(d, "images")
        self.uv = np.array([r["target"]["normalized_uv"] for r in self.recs])
        self.mode = np.array([r["pose_mode"] for r in self.recs])
        self.stim, self.size, self.noise = stim, (width, height), noise
        self.rng = np.random.default_rng(seed)
        self._noise = [self.rng.normal(0, noise, (height, width, 3)).astype(np.int16) for _ in range(6)]
        self._cache = {}
        self.center_stage_off = None
        self._lock = threading.Condition()
        self._frame, self._ts, self._idx, self._stamps, self._stop = self._img(0), time.monotonic(), 0, [], False
        self._key, self._cur = None, 0
        self.period = 1.0 / fps
        threading.Thread(target=self._loop, daemon=True).start()

    def _img(self, i):
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = self.cv2.imread(os.path.join(self.img_dir, self.recs[i]["image"]))
        return self._cache[i]

    def _choose(self, uv, cond):
        allowed = self.MODES.get(cond)
        idx = np.arange(len(self.recs)) if not allowed else np.where(np.isin(self.mode, allowed))[0]
        d = np.linalg.norm(self.uv[idx] - np.asarray(uv), axis=1)
        return int(self.rng.choice(idx[np.argsort(d)[:3]]))

    def _loop(self):
        nxt = time.monotonic()
        while not self._stop:
            key = self.stim.get("key")
            if key != self._key:
                self._key = key
                uv, cond = self.stim.get("uv"), self.stim.get("cond")
                self._cur = self._choose(uv, cond) if uv is not None else int(self.rng.integers(len(self.recs)))
            img = self._img(self._cur)
            fr = np.clip(img.astype(np.int16) + self._noise[int(self.rng.integers(len(self._noise)))], 0, 255).astype(np.uint8)
            nxt += self.period
            time.sleep(max(0.0, nxt - time.monotonic()))
            t = time.monotonic()
            with self._lock:
                self._frame, self._ts, self._idx = fr, t, self._idx + 1
                self._stamps.append(t)
                if len(self._stamps) > 300:
                    del self._stamps[:150]
                self._lock.notify_all()

    def describe(self):
        return {"size": list(self.size), "backend": "FAKE (synthetic Phase 5 images, test only)", "center_stage_disabled": None}

    def close(self):
        self._stop = True


# =========================================================================== display
class UI:
    WIN = "Tgaze Phase 6"

    def __init__(self, layout, headless=False, auto_keys=None):
        import cv2
        self.cv2, self.lay, self.headless = cv2, layout, headless
        self.auto = list(auto_keys or [])
        self.font = None
        try:
            from PIL import ImageFont
            for p in ("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"):
                if os.path.exists(p):
                    self.font_path = p
                    self.font = {s: ImageFont.truetype(p, s) for s in (18, 24, 32, 44)}
                    break
        except Exception:
            self.font = None
        if not headless:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def blank(self, gray=40):
        return np.full((self.lay.ch, self.lay.cw, 3), gray, np.uint8)

    def text(self, img, lines, org=(40, 60), size=24, color=(235, 235, 235), step=None):
        """Draw text lines (Japanese if a CJK font exists, else ASCII-only fallback)."""
        step = step or int(size * 1.5)
        if self.font:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(img[:, :, ::-1])
            d = ImageDraw.Draw(pil)
            for i, l in enumerate(lines):
                d.text((org[0], org[1] + i * step - size), l, font=self.font[min(self.font, key=lambda s: abs(s - size))], fill=color[::-1])
            img[:] = np.asarray(pil)[:, :, ::-1]
        else:
            for i, l in enumerate(lines):
                l = l.encode("ascii", "replace").decode() if any(ord(c) > 127 for c in l) else l
                self.cv2.putText(img, l, (org[0], org[1] + i * step), self.cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 1, self.cv2.LINE_AA)
        return img

    def show(self, img):
        if not self.headless:
            self.cv2.imshow(self.WIN, img)

    def key(self, wait_ms=1, prompt=False):
        """prompt=True: a screen waiting for the operator (scripted keys are consumed only there)."""
        if self.headless:
            time.sleep(wait_ms / 1000.0)
            if prompt:
                return ord(self.auto.pop(0).lower()) if self.auto else ord("q")
            return -1
        k = self.cv2.waitKey(max(1, wait_ms))
        return -1 if k < 0 else k & 0xFF

    def wait_space(self, img):
        """Instruction screen: Space continues, Esc cancels. Headless: continues immediately."""
        self.show(img)
        if self.headless:
            return True
        while True:
            k = self.key(30)
            if k == SPACE:
                return True
            if k == ESC:
                return False

    def close(self):
        if not self.headless:
            self.cv2.destroyAllWindows()


class PerfLog:
    """1 Hz CPU / memory sampling of this process into perf.jsonl."""

    def __init__(self, path):
        import psutil
        self.p = psutil.Process()
        self.p.cpu_percent(None)
        self.fh = open(path, "a")
        self.n_cpu = psutil.cpu_count()
        self._stop = False
        self.context = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            time.sleep(1.0)
            try:
                rec = {"t": time.time(), "cpu_percent_one_core": self.p.cpu_percent(None), "rss_mb": self.p.memory_info().rss / 2 ** 20,
                       "n_threads": self.p.num_threads(), "n_logical_cpu": self.n_cpu, **self.context}
                self.fh.write(json.dumps(rec, default=C.json_default) + "\n")
                self.fh.flush()
            except Exception:
                pass

    def close(self):
        self._stop = True
        self.fh.close()


# =========================================================================== app
class App:
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.tgaze = C.tgaze_dir(cfg)
        import m8_cpu
        self.m8mod = m8_cpu
        K, self.kinfo = m8_cpu.load_camera_matrix(cfg, self.tgaze, a.allow_k_host_mismatch)
        self.rt = m8_cpu.M8Runtime(cfg, self.tgaze, K)
        if not self.rt.k_matches_training:
            raise RuntimeError(f"K differs from the K used for M8 training by {self.rt.k_max_abs_diff_vs_training_px:.2f} px")
        self.layout, self.layout_info = C.ScreenLayout.detect()
        self.ui = UI(self.layout, a.headless, a.auto.split(",") if a.auto else None)
        self.stim = {"key": None, "uv": None, "cond": None}
        if a.fake_camera:
            self.cam = FakeCamera(a.fake_camera, a.fake_identity, self.stim)
        else:
            self.cam = Camera(cfg["camera"]["index"], cfg["camera"]["width"], cfg["camera"]["height"])
        self.sdir = os.path.join(a.out, a.subject)
        for sub in ("calibration", "evaluation", "sanity", "representative_frames", "plans"):
            os.makedirs(os.path.join(self.sdir, sub), exist_ok=True)
        self.meta_path = os.path.join(self.sdir, "metadata.jsonl")
        self.perf = PerfLog(os.path.join(self.sdir, "perf.jsonl"))
        self.subject_num = int(a.subject.split("_")[1])
        self._last_idx = -1
        self._write_subject_json()

    # ------------------------------------------------------------------ bookkeeping
    def _write_subject_json(self):
        p = os.path.join(self.sdir, "subject.json")
        info = json.load(open(p)) if os.path.exists(p) else {"subject_id": self.a.subject, "created": dt.datetime.now().isoformat(timespec="seconds")}
        role = "pilot" if self.a.pilot else "main"
        if info.get("role") and info["role"] != role:
            raise RuntimeError(f"{self.a.subject} was recorded as {info['role']}, not {role}")
        info.update({
            "role": role, "config_path": self.cfg["_path"], "config_sha256": self.cfg["_sha256"],
            "frozen_protocol_hash": C.frozen_config_hash(self.cfg),
            "model_onnx_sha256": C.sha256_file(self.rt.onnx_path), "runtime": self.rt.describe(), "camera_matrix": self.kinfo,
            "camera": self.cam.describe(), "screen_layout": self.layout_info, "fake_camera": bool(self.a.fake_camera),
            "fake_identity": self.a.fake_identity if self.a.fake_camera else None,
            "platform": {"system": platform.system(), "machine": platform.machine(), "mac_ver": platform.mac_ver()[0],
                         "python": sys.version.split()[0], "processor": platform.processor()},
            "sessions": info.get("sessions", []), "sanity": info.get("sanity"), "geometry_check": info.get("geometry_check"),
            "no_personal_names_stored": True})
        try:
            import cv2
            import mediapipe
            import onnxruntime
            info["versions"] = {"opencv": cv2.__version__, "mediapipe": mediapipe.__version__, "onnxruntime": onnxruntime.__version__}
        except Exception:
            pass
        self.info = info
        json.dump(info, open(p, "w"), indent=1, ensure_ascii=False, default=C.json_default)

    def _save_info(self):
        json.dump(self.info, open(os.path.join(self.sdir, "subject.json"), "w"), indent=1, ensure_ascii=False, default=C.json_default)

    def records(self):
        if not os.path.exists(self.meta_path):
            return []
        return [json.loads(l) for l in open(self.meta_path)]

    def event(self, **kw):
        with open(os.path.join(self.sdir, "events.jsonl"), "a") as fh:
            fh.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="milliseconds"), **kw}) + "\n")

    # ------------------------------------------------------------------ drawing
    def draw_target(self, img, uv, radius_mm):
        cv2, T, lay = self.ui.cv2, self.cfg["target"], self.layout
        ppm = lay.px_per_mm()
        x, y = lay.uv_to_canvas(uv)
        c = (int(round(x)), int(round(y)))
        cv2.circle(img, c, int(round(radius_mm * ppm)), tuple(T["color_bgr"]), max(1, int(round(T["ring_width_mm"] * ppm))), cv2.LINE_AA)
        cv2.circle(img, c, max(2, int(round(T["center_dot_radius_mm"] * ppm))), tuple(T["dot_color_bgr"]), -1, cv2.LINE_AA)
        return c

    def progress_bar(self, img, done, total, label):
        h, w = img.shape[:2]
        cv2 = self.ui.cv2
        cv2.rectangle(img, (0, h - 4), (int(w * done / max(total, 1)), h), (90, 90, 90), -1)
        cv2.putText(img, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ one trial
    def run_trial(self, session, slot, target_uv, cond, progress):
        """Show one target, process every new camera frame, store frames + summary. Returns record or 'paused'."""
        cfg, T = self.cfg, self.cfg["timing"]
        t_move, t_settle, t_rec = T["move_ms"] / 1e3, T["settle_ms"] / 1e3, T["record_ms"] / 1e3
        total = t_move + t_settle + t_rec
        trial_id = len(self.records()) + 1
        self.stim.update({"key": f"{session}:{trial_id}", "uv": tuple(target_uv), "cond": cond})
        self.perf.context = {"session": session, "trial_id": trial_id}
        bg = cfg["target"]["background_gray"]
        blank = self.ui.blank(bg)
        self.progress_bar(blank, *progress)
        self.ui.show(blank)
        if self.ui.key(T["inter_trial_blank_ms"]) == ESC:
            return "paused"
        frames, t_on = [], None
        rep_frame, rep_dt = None, None
        mid_rec = t_move + t_settle + t_rec / 2
        disp_px = None
        while True:
            now = time.monotonic()
            el = 0.0 if t_on is None else now - t_on
            if t_on is not None and el >= total:
                break
            img = self.ui.blank(bg)
            r = cfg["target"]["outer_radius_mm"]
            if el < t_move:
                r = cfg["target"]["shrink_from_mm"] + (r - cfg["target"]["shrink_from_mm"]) * (el / t_move)
            disp_px = self.draw_target(img, target_uv, r)
            self.progress_bar(img, *progress)
            self.ui.show(img)
            k = self.ui.key(1)
            if t_on is None:
                t_on = time.monotonic()                 # target on screen (after the first imshow/waitKey)
            if k == ESC:
                self.event(kind="pause", session=session, slot=slot)
                return "paused"
            fr, ts, idx = self.cam.latest(self._last_idx, timeout=0.1)
            if idx <= self._last_idx:
                continue
            self._last_idx = idx
            t_rel = ts - t_on
            if t_rel < 0:
                continue
            phase = "move" if t_rel < t_move else ("settle" if t_rel < t_move + t_settle else ("record" if t_rel < total else None))
            if phase is None:
                continue
            res = self.rt.process(fr, ts * 1000.0, want_patch=cfg["storage"]["save_eye_patch"])
            res.update({"t_rel": t_rel, "phase": phase, "cam_idx": idx})
            frames.append(res)
            if phase == "record" and (rep_dt is None or abs(t_rel - mid_rec) < rep_dt):
                rep_frame, rep_dt = fr, abs(t_rel - mid_rec)
                if cfg["storage"]["save_all_record_frames"]:
                    self._save_jpg(fr, os.path.join(self.sdir, session, "frames", f"trial_{trial_id:04d}", f"{idx:06d}.jpg"))
        return self._finish_trial(session, slot, trial_id, target_uv, cond, frames, disp_px, rep_frame, t_on)

    def _save_jpg(self, fr, path):
        import cv2
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, fr, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["storage"]["jpeg_quality"]])

    def _finish_trial(self, session, slot, trial_id, target_uv, cond, frames, disp_px, rep_frame, t_on):
        keep = frames if self.cfg["timing"]["keep_settle_frames"] else [f for f in frames if f["phase"] == "record"]
        rec_fr = [f for f in frames if f["phase"] == "record"]
        n = len(keep)
        ok = np.array([f["ok"] for f in keep], bool)
        arr = {"t_rel_s": np.array([f["t_rel"] for f in keep]), "phase": np.array([f["phase"] for f in keep]),
               "cam_idx": np.array([f["cam_idx"] for f in keep]), "ok": ok,
               "lms": np.stack([f["lms"] if f["ok"] else np.full((478, 3), np.nan, np.float32) for f in keep]) if n else np.zeros((0, 478, 3), np.float32),
               "f16": np.stack([f["f16"] if f["ok"] else np.full(16, np.nan, np.float32) for f in keep]) if n else np.zeros((0, 16), np.float32),
               "R_head": np.stack([f["R_head"] if f["ok"] else np.full((3, 3), np.nan, np.float32) for f in keep]) if n else np.zeros((0, 3, 3), np.float32),
               "uv_m8_raw": np.stack([f["uv"] if f["ok"] else np.full(2, np.nan) for f in keep]) if n else np.zeros((0, 2)),
               "eye_cam_m": np.stack([f["eye_cam_m"] if f["ok"] else np.full(3, np.nan) for f in keep]) if n else np.zeros((0, 3))}
        for k in ("mediapipe", "features", "onnx", "eye_patch", "total", "total_m8_only"):
            arr[f"lat_{k}_ms"] = np.array([f["lat_ms"].get(k, np.nan) for f in keep])
        if self.cfg["storage"]["save_eye_patch"]:
            arr["eye_patch"] = np.stack([f["patch"] if f["ok"] and f["patch"] is not None else np.full(3072, np.nan, np.float16) for f in keep]) if n else np.zeros((0, 3072), np.float16)
        npz = os.path.join(session, f"trial_{trial_id:04d}.npz")
        np.savez_compressed(os.path.join(self.sdir, npz), **arr)

        r_ok = np.array([f["ok"] for f in rec_fr], bool)
        r_uv = np.array([f["uv"] if f["ok"] else (np.nan, np.nan) for f in rec_fr]).reshape(-1, 2)
        r_pyr = np.array([f["f16"][[4, 5, 7]] if f["ok"] else (np.nan,) * 3 for f in rec_fr]).reshape(-1, 3)
        status, qflags = C.trial_quality(r_ok, r_uv, r_pyr, self.cfg)
        pred = np.nanmedian(r_uv[r_ok], 0).tolist() if r_ok.any() else None
        rep = None
        if rep_frame is not None and self.cfg["storage"]["save_representative_frame"]:
            rep = os.path.join("representative_frames", f"{session}_trial_{trial_id:04d}.jpg")
            self._save_jpg(rep_frame, os.path.join(self.sdir, rep))
        tu = np.asarray(target_uv, float)
        rec = {"timestamp": dt.datetime.now().isoformat(timespec="milliseconds"), "subject_id": self.a.subject,
               "session": session, "session_id": self.session_id, "trial_id": trial_id, "slot": slot,
               "block": slot.get("block"), "pose_instruction": cond,
               "target": {"uv": tu.tolist(), "tgaze_norm": C.uv_to_tgaze(tu).tolist(),
                          "mm_from_display_center": (C.uv_to_cm(tu) * 10).tolist(),
                          "mm_from_top_left": [(tu[0] + 1) / 2 * C.DISPLAY_W_MM, (1 - tu[1]) / 2 * C.DISPLAY_H_MM],
                          "canvas_px": list(disp_px) if disp_px else None},
               "timing_ms": {k: self.cfg["timing"][k] for k in ("move_ms", "settle_ms", "record_ms")},
               "n_frames": {p: int(sum(f["phase"] == p for f in frames)) for p in PHASES},
               "status": status, "quality": qflags,
               "m8_raw_pred_uv_median": pred,
               "head_pose_median_deg": (np.degrees(np.nanmedian(r_pyr[r_ok], 0))).tolist() if r_ok.any() else None,
               "frames_file": npz, "representative_frame": rep,
               "latency_ms_mean": {k: float(np.nanmean(arr[f"lat_{k}_ms"][arr["phase"] == "record"])) if r_ok.any() else None
                                   for k in ("mediapipe", "features", "onnx", "total_m8_only", "total")},
               "camera_fps": self.cam.fps()}
        with open(self.meta_path, "a") as fh:
            fh.write(json.dumps(rec, default=C.json_default) + "\n")
        return rec

    # ------------------------------------------------------------------ sessions
    def make_plan(self, session):
        cfg = self.cfg
        rng = random.Random(f"{self.a.subject}:{session}:{cfg['_sha256']}")
        slots = []
        if session == "calibration":
            pts = cfg["calibration"]["point_sets"]["9"]
            order = [tuple(p) for p in pts for _ in range(cfg["calibration"]["repeats_per_point"])]
            rng.shuffle(order)
            slots = [{"slot_id": f"cal_{i:03d}", "block": 0, "cond": cfg["calibration"]["pose_instruction"], "uv": list(p)} for i, p in enumerate(order)]
        elif session == "sanity":
            st = cfg["grids"]["sanity_targets"]
            order = [k for k in st for _ in range(cfg["grids"]["sanity_repeats"])]
            rng.shuffle(order)
            slots = [{"slot_id": f"san_{i:03d}", "block": 0, "cond": "NATURAL", "uv": st[k], "name": k} for i, k in enumerate(order)]
        else:
            g = cfg["grids"]["evaluation_uv"]
            grid = [(u, v) for v in g[::-1] for u in g]
            blocks = cfg["evaluation"]["pilot_blocks"][0] if self.a.pilot else cfg["evaluation"]["blocks"][(self.subject_num + 1) % 2]
            for b, cond in enumerate(blocks):
                order = grid[:]
                if self.a.pilot and cfg["evaluation"]["pilot_targets_subsample"]:
                    order = rng.sample(order, cfg["evaluation"]["pilot_targets_subsample"])
                rng.shuffle(order)
                slots += [{"slot_id": f"ev_b{b}_{i:03d}", "block": b, "cond": cond, "uv": list(p)} for i, p in enumerate(order)]
        for s in slots:
            vis = self.layout.visible(s["uv"], margin_px=self.layout.px_per_mm() * self.cfg["target"]["outer_radius_mm"])
            s["visible_in_window"] = bool(vis)
        return slots

    def session(self, session):
        path = os.path.join(self.sdir, "plans", f"{session}.json")
        if os.path.exists(path):
            plan = json.load(open(path))
        else:
            plan = {"session": session, "created": dt.datetime.now().isoformat(timespec="seconds"), "slots": self.make_plan(session)}
            json.dump(plan, open(path, "w"), indent=1)
        bad = [s for s in plan["slots"] if not s["visible_in_window"]]
        if bad:
            print(f"[warn] {len(bad)} targets fall outside the window (layout {self.layout.describe()})")
        self.session_id = f"{session}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
        self.info["sessions"].append({"session_id": self.session_id, "session": session, "start": dt.datetime.now().isoformat(timespec="seconds")})
        self._save_info()
        self.event(kind="session_start", session=session, session_id=self.session_id)
        maxr = self.cfg["evaluation"]["max_retakes_per_trial"]

        def todo():
            """Slots still to run: never attempted, retake requested by the operator, or INVALID (auto-retake,
            at most max_retakes_per_trial). Fresh slots of a block come before that block's retakes."""
            recs = [r for r in self.records() if r["session"] == session]
            out = []
            for i, s in enumerate(plan["slots"]):
                att = [r for r in recs if r["slot"]["slot_id"] == s["slot_id"]]
                if not att:
                    out.append((s["block"], 0, i, s))
                elif att[-1].get("operator_retake") and len(att) <= maxr + 2:
                    out.append((s["block"], 1, i, s))
                elif att[-1]["status"] == "INVALID" and self.cfg["evaluation"]["retake_invalid_automatically"] and len(att) <= maxr:
                    out.append((s["block"], 1, i, s))
            return [s for *_, s in sorted(out, key=lambda x: x[:3])]

        cur_block = None
        while True:
            rem = todo()
            if not rem:
                break
            s = rem[0]
            if s["block"] != cur_block or (cur_block is None):
                cur_block = s["block"]
                if not self.instruction_screen(session, s["cond"], s["block"], plan):
                    self.event(kind="session_abort", session=session)
                    return False
            n_done = len(plan["slots"]) - len(rem)
            res = self.run_trial(session, {k: s[k] for k in ("slot_id", "block")}, s["uv"], s["cond"],
                                 (n_done, len(plan["slots"]), f"{session}  block {s['block'] + 1}  {n_done + 1}/{len(plan['slots'])}"))
            if res == "paused":
                act = self.pause_screen()
                if act == "menu":
                    self.event(kind="session_abort", session=session)
                    return False
                if act == "retake":
                    self.mark_operator_retake(session)
                cur_block = None                      # show the block instruction again after a pause
                continue
        self.event(kind="session_complete", session=session, session_id=self.session_id)
        if session == "calibration":
            self.fit_on_device()
        if session == "sanity":
            self.sanity_result()
        return True

    def mark_operator_retake(self, session):
        recs = self.records()
        idx = [i for i, r in enumerate(recs) if r["session"] == session]
        if not idx:
            return
        recs[idx[-1]]["operator_retake"] = True
        with open(self.meta_path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=C.json_default) + "\n")
        self.event(kind="operator_retake", session=session, trial_id=recs[idx[-1]]["trial_id"])

    def instruction_screen(self, session, cond, block, plan):
        img = self.ui.blank(self.cfg["target"]["background_gray"])
        ja, en = self.cfg["instructions"][cond]
        title = {"calibration": "キャリブレーション / Calibration", "evaluation": f"評価 / Evaluation  (block {block + 1})",
                 "sanity": "方向チェック / Direction check"}[session]
        self.ui.text(img, [title, "", ja, en, "", "ターゲットが出たら、中心の赤い点を見続けてください。",
                           "Keep looking at the red dot until the next target appears.", "",
                           "Space: 開始 / start    Esc: 戻る / back"], org=(80, 140), size=32)
        return self.ui.wait_space(img)

    def pause_screen(self):
        img = self.ui.blank(30)
        self.ui.text(img, ["一時停止 / Paused", "", "Space: 続ける / continue", "R: 直前の試行を取り直す / retake previous trial",
                           "B: メニューへ戻る(保存済み) / back to menu (saved)"], org=(80, 160), size=32)
        self.ui.show(img)
        while True:
            k = self.ui.key(30, prompt=True)
            if k in (SPACE, ord(" ")):
                return "continue"
            if k in (ord("r"), ord("R")):
                return "retake"
            if k in (ord("b"), ord("B"), ESC, ord("q")):
                return "menu"

    # ------------------------------------------------------------------ results on device
    def cal_trials(self):
        best = {}
        for r in self.records():
            if r["session"] == "calibration" and r["status"] in self.cfg["analysis"]["use_status"] and r["m8_raw_pred_uv_median"] and not r.get("operator_retake"):
                best[r["slot"]["slot_id"]] = (tuple(r["target"]["uv"]), tuple(r["m8_raw_pred_uv_median"]))
        return list(best.values())

    def fit_on_device(self):
        """Fit the pre-registered corrections on this Mac (CPU) and time them; also time the tgaze 9-point fit."""
        out = {"fitted": dt.datetime.now().isoformat(timespec="seconds"), "models": {}}
        cal = self.cal_trials()
        for n in (1, 3, 5, 9):
            t0 = time.perf_counter()
            corr = C.fit_correction(self.cfg, n, cal)
            out["models"][str(n)] = {**corr.to_json(), "fit_ms": 1e3 * (time.perf_counter() - t0)}
        try:
            from appearance import AppearancePipeline
            pipe, nfr = AppearancePipeline(), 0
            for r in self.records():
                if r["session"] != "calibration" or r["status"] not in self.cfg["analysis"]["use_status"] or r.get("operator_retake"):
                    continue
                z = np.load(os.path.join(self.sdir, r["frames_file"]))
                for i in np.where((z["phase"] == "record") & z["ok"])[0]:
                    if not np.isfinite(z["eye_patch"][i]).all():
                        continue
                    pipe.collect_point(z["f16"][i], z["eye_patch"][i].astype(np.float64), C.uv_to_tgaze(r["target"]["uv"]))
                    nfr += 1
            t0 = time.perf_counter()
            pipe.finalize()
            out["tgaze_appearance_9pt"] = {"fit_ms_including_tgaze_LOO": 1e3 * (time.perf_counter() - t0), "n_frames": nfr}
        except Exception as e:
            out["tgaze_appearance_9pt"] = {"error": repr(e)}
        json.dump(out, open(os.path.join(self.sdir, "calibration_model.json"), "w"), indent=1, default=C.json_default)
        print("[calibration] fit on device:", {k: round(v["fit_ms"], 3) for k, v in out["models"].items()}, out["tgaze_appearance_9pt"])

    def sanity_result(self):
        """Direction check (zero-shot M8): right > left in u, up > down in v."""
        recs = [r for r in self.records() if r["session"] == "sanity" and r["m8_raw_pred_uv_median"]]
        st = self.cfg["grids"]["sanity_targets"]
        med = {}
        for name, uv in st.items():
            p = [r["m8_raw_pred_uv_median"] for r in recs if r["target"]["uv"] == list(map(float, uv))]
            med[name] = np.median(p, 0).tolist() if p else None
        ok = all(med.values())
        res = {"time": dt.datetime.now().isoformat(timespec="seconds"), "median_pred_uv": med}
        if ok:
            res["dx_right_minus_left_cm"] = (med["right"][0] - med["left"][0]) * C.HALF_CM[0]
            res["dy_up_minus_down_cm"] = (med["up"][1] - med["down"][1]) * C.HALF_CM[1]
            res["x_direction_ok"] = res["dx_right_minus_left_cm"] > 2.0
            res["y_direction_ok"] = res["dy_up_minus_down_cm"] > 1.0
            res["pass"] = bool(res["x_direction_ok"] and res["y_direction_ok"])
        else:
            res["pass"] = False
        self.info["sanity"] = res
        self._save_info()
        img = self.ui.blank(30)
        for name, uv in st.items():
            self.draw_target(img, uv, 4.0)
            if med[name]:
                c = tuple(int(round(v)) for v in self.layout.uv_to_canvas(med[name]))
                self.ui.cv2.circle(img, c, 10, (0, 200, 255), -1)
                self.ui.cv2.line(img, tuple(int(round(v)) for v in self.layout.uv_to_canvas(uv)), c, (0, 200, 255), 1)
        verdict = "PASS" if res["pass"] else "FAIL: 左右/上下の向きを確認してください (mirroring?)"
        self.ui.text(img, [f"方向チェック / Direction check: {verdict}",
                           f"right-left = {res.get('dx_right_minus_left_cm', float('nan')):+.1f} cm (must be > +2)   "
                           f"up-down = {res.get('dy_up_minus_down_cm', float('nan')):+.1f} cm (must be > +1)",
                           "白=ターゲット / white = target, 橙=M8予測(補正なし) / orange = M8 zero-shot", "Space: menu"], org=(60, 70), size=24)
        self.ui.wait_space(img)
        print("[sanity]", json.dumps(res, default=C.json_default))

    # ------------------------------------------------------------------ menu screens
    def geometry_check(self):
        cv2, lay = self.ui.cv2, self.layout
        img = self.ui.blank(30)
        ppm = lay.px_per_mm()
        lines = ["画面ジオメトリ確認 / Geometry check (定規で測ってください / measure with a ruler)"]
        for uv in self.cfg["grids"]["geometry_check_marks_uv"]:
            c = tuple(int(round(v)) for v in lay.uv_to_canvas(uv))
            cv2.drawMarker(img, c, (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
        x0, y0 = lay.uv_to_canvas((-0.5, -0.3))
        cv2.line(img, (int(x0), int(y0)), (int(x0 + 100 * ppm), int(y0)), (255, 255, 255), 2)
        cv2.line(img, (int(x0), int(y0)), (int(x0), int(y0 - 50 * ppm)), (255, 255, 255), 2)
        a, b = self.cfg["grids"]["geometry_check_marks_uv"][0], self.cfg["grids"]["geometry_check_marks_uv"][1]
        lines += ["白い横線 = 100 mm、縦線 = 50 mm / white lines: 100 mm and 50 mm",
                  f"左上の十字: 画面の発光部の左端から {(a[0] + 1) / 2 * C.DISPLAY_W_MM:.1f} mm、上端(ノッチ帯を含む)から {(1 - a[1]) / 2 * C.DISPLAY_H_MM:.1f} mm",
                  f"上の2つの十字の間隔 = {(b[0] - a[0]) / 2 * C.DISPLAY_W_MM:.1f} mm",
                  "合っていれば Y、ずれていれば N / Y if correct, N if not"]
        self.ui.text(img, lines, org=(int(lay.cw * 0.12), int(lay.ch * 0.3)), size=22)
        self.ui.show(img)
        while True:
            k = self.ui.key(30, prompt=True)
            if k in (ord("y"), ord("Y"), ord("n"), ord("N")):
                self.info["geometry_check"] = {"confirmed": k in (ord("y"), ord("Y")), "time": dt.datetime.now().isoformat(timespec="seconds"),
                                               "layout": lay.describe()}
                self._save_info()
                return
            if k in (ESC, ord("q")):
                return

    def live_cursor(self, seconds=None):
        """Live zero-shot M8 cursor (no calibration, no smoothing) for an informal direction check."""
        cv2 = self.ui.cv2
        t_end = time.monotonic() + (seconds or (2.0 if self.ui.headless else 1e9))
        self.stim.update({"key": "live", "uv": (0.0, 0.0), "cond": "NATURAL"})
        while time.monotonic() < t_end:
            fr, ts, idx = self.cam.latest(self._last_idx)
            if idx <= self._last_idx:
                continue
            self._last_idx = idx
            res = self.rt.process(fr, ts * 1000.0, want_patch=False)
            img = self.ui.blank(30)
            for uv in self.cfg["grids"]["sanity_targets"].values():
                self.draw_target(img, uv, 3.0)
            if res["ok"]:
                cv2.circle(img, tuple(int(round(v)) for v in self.layout.uv_to_canvas(res["uv"])), 14, (0, 200, 255), 2)
            self.ui.text(img, ["ライブ(補正なし・平滑化なし) / live zero-shot M8, no smoothing.  左右上下を見て、橙の円が同じ向きに動くか確認。 Esc: menu"], org=(40, 40), size=20)
            self.ui.show(img)
            if self.ui.key(1) == ESC:
                break

    def benchmark(self, n=300):
        """Live CPU benchmark: camera FPS, per-stage latency, effective FPS, CPU, memory."""
        import psutil
        p = psutil.Process()
        p.cpu_percent(None)
        cpu0, w0 = p.cpu_times(), time.perf_counter()
        lat, k = [], 0
        self.stim.update({"key": "bench", "uv": (0.0, 0.0), "cond": "NATURAL"})
        while len(lat) < n:
            fr, ts, idx = self.cam.latest(self._last_idx)
            if idx <= self._last_idx:
                continue
            self._last_idx = idx
            r = self.rt.process(fr, ts * 1000.0, want_patch=False)
            if r["ok"]:
                lat.append(r["lat_ms"])
            k += 1
            if k % 15 == 0:
                img = self.ui.blank(30)
                self.ui.text(img, [f"CPU benchmark {len(lat)}/{n}  (画面を見ていてください / keep looking at the screen)"], org=(60, 80), size=28)
                self.ui.show(img)
                if self.ui.key(1) == ESC:
                    break
        wall = time.perf_counter() - w0
        c1 = p.cpu_times()
        st = lambda key: {"mean": float(np.mean([l[key] for l in lat])), "median": float(np.median([l[key] for l in lat])),
                          "p95": float(np.percentile([l[key] for l in lat], 95))}
        out = {"time": dt.datetime.now().isoformat(timespec="seconds"), "frames_processed": k, "frames_with_face": len(lat),
               "camera_fps": self.cam.fps(), "effective_fps": k / wall,
               "latency_ms": {key: st(key) for key in ("mediapipe", "features", "onnx", "total_m8_only", "eye_patch", "total")},
               "cpu_percent_of_one_core": 100 * ((c1.user - cpu0.user) + (c1.system - cpu0.system)) / wall,
               "n_logical_cpu": psutil.cpu_count(), "rss_mb": p.memory_info().rss / 2 ** 20,
               "ort_session_providers": self.rt.ort_providers, "ort_available_providers": self.rt.ort_available,
               "mediapipe_delegate": "CPU", "torch_loaded": "torch" in sys.modules, "fake_camera": bool(self.a.fake_camera)}
        json.dump(out, open(os.path.join(self.sdir, f"cpu_benchmark_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"), "w"), indent=1, default=C.json_default)
        print("[benchmark]", json.dumps({k: out[k] for k in ("camera_fps", "effective_fps", "cpu_percent_of_one_core", "rss_mb")}),
              {k: round(v["mean"], 2) for k, v in out["latency_ms"].items()})
        return out

    def menu(self):
        cv2 = self.ui.cv2
        last = None
        while True:
            fr, ts, idx = self.cam.latest(self._last_idx, timeout=0.05)
            if idx > self._last_idx:
                self._last_idx = idx
                last = self.rt.process(fr, ts * 1000.0, want_patch=False)
                last["frame"] = fr
            img = self.ui.blank(30)
            recs = self.records()
            cnt = lambda s: sum(r["session"] == s and r["status"] != "INVALID" for r in recs)
            san = self.info.get("sanity") or {}
            geo = self.info.get("geometry_check") or {}
            status = [f"{self.a.subject}  ({'PILOT' if self.a.pilot else 'main'})   camera {self.cam.size[0]}x{self.cam.size[1]} @ {self.cam.fps():.1f} fps   "
                      f"K {'measured (host match)' if self.kinfo['host_match'] else 'measured (HOST MISMATCH: test only)'}   mirrored=True   ORT {self.rt.ort_providers}",
                      f"face: {'OK' if last and last['ok'] else 'NOT FOUND'}" + (
                          f"   distance ~{last['eye_cam_m'][2] * 100:.0f} cm   head pitch/yaw {np.degrees(last['f16'][4]):+.0f}/{np.degrees(last['f16'][5]):+.0f} deg   "
                          f"latency {last['lat_ms']['total']:.0f} ms" if last and last["ok"] else ""),
                      "",
                      f"G  画面ジオメトリ確認 / geometry check       [{'OK' if geo.get('confirmed') else ('NG' if geo else '-')}]",
                      f"D  方向チェック / direction sanity check     [{'PASS' if san.get('pass') else ('FAIL' if san else '-')}]",
                      "L  ライブカーソル / live cursor (zero-shot)",
                      f"C  キャリブレーション / calibration session  [{cnt('calibration')} trials]",
                      f"E  評価 / evaluation session                 [{cnt('evaluation')} trials]",
                      "P  CPUベンチマーク / CPU benchmark",
                      "Q / Esc  終了 / quit"]
            if not self.cam.describe().get("center_stage_disabled", True) and not self.a.fake_camera:
                status.append("WARNING: センターフレームを切れていません。コントロールセンター > ビデオエフェクトでオフにしてください。")
            self.ui.text(img, status, org=(60, 60), size=24, step=40)
            if last is not None and "frame" in last:
                pv = cv2.resize(cv2.flip(last["frame"], 1), (480, 270))
                if last["ok"]:
                    for i in list(range(468, 478)) + [33, 133, 263, 362, 1, 152]:
                        cv2.circle(pv, (int(last["lms"][i, 0] * 480), int(last["lms"][i, 1] * 270)), 1, (0, 255, 0), -1)
                h, w = img.shape[:2]
                img[h - 290:h - 20, w - 500:w - 20] = pv
            self.ui.show(img)
            k = self.ui.key(10, prompt=self.ui.headless)
            if k < 0:
                continue
            c = chr(k).upper() if k < 128 else ""
            if k == ESC or c == "Q":
                return
            if c == "G":
                self.geometry_check()
            elif c == "D":
                self.session("sanity")
            elif c == "L":
                self.live_cursor()
            elif c in ("C", "E"):
                if not san.get("pass") and not self.a.force:
                    print("[menu] run the direction check (D) first and make sure it passes (--force to override)")
                    continue
                self.session("calibration" if c == "C" else "evaluation")
            elif c == "P":
                self.benchmark()

    def close(self):
        self.perf.close()
        self.cam.close()
        self.rt.close()
        self.ui.close()


def next_subject(out):
    used = {int(os.path.basename(p).split("_")[1]) for p in glob.glob(os.path.join(out, "subject_*"))}
    i = 1
    while i in used:
        i += 1
    return f"subject_{i:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--next-subject", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="allow C/E without a passed direction check")
    ap.add_argument("--allow-k-host-mismatch", action="store_true", help="TEST ONLY")
    ap.add_argument("--fake-camera", default=None, help="TEST ONLY: Phase 5 dataset root")
    ap.add_argument("--fake-identity", type=int, default=7)
    ap.add_argument("--headless", action="store_true", help="TEST ONLY: no window")
    ap.add_argument("--auto", default=None, help="TEST ONLY: scripted menu keys, e.g. G,Y,D,C,E,P,Q")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    a.out = os.path.abspath(a.out or os.path.join(KIT_DIR, cfg["paths"]["output_dir"]))
    if a.next_subject:
        a.subject = next_subject(a.out)
    if not a.subject or not C.SUBJECT_RE.match(a.subject):
        sys.exit("--subject must be an anonymous id like subject_01 (no names)")
    print(f"[collect] {a.subject} -> {os.path.join(a.out, a.subject)}")
    app = App(a, cfg)
    try:
        app.menu()
    finally:
        app.close()


if __name__ == "__main__":
    main()
