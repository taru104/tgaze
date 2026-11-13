"""Web アプリの M8 系モデル用の推定部分（server.py の App と同じ API）。撮影キット（experiments/phase6_kit）の部品を使う。

  python webapp/server.py --model models_alt/m8_phase8_1234          # 1234人・顔の点のみ
  python webapp/server.py --model frozen                             # 本番（凍結 M8）
  （--kit で撮影キットの場所を指定。既定は ~/tgaze/experiments/phase6_kit）

tgaze.GazeTracker 版との違い:
  - 入力は 1920x1080 のカメラ映像（M8 は実測したカメラ行列と 1920x1080 が前提）。ブラウザは /api/info の capture で撮る
  - キャリブなしでも視線点を返す（M8 は合成データで学習済み）。キャリブすると、本番評価と同じ方式
    （画面 cm での affine、λ=25 cm²）で補正する。1点あたりのフレームの中央値で当てはめる
  - 視線点は「画面全体（ノッチ帯を含む）」に対する [0,1]（tgaze と同じ向き: x 右、y 下）
  - クリック補正（/api/click）はしない
"""
from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np

MIN_POINT_SAMPLES = 5


class OneEuro2D:
    """One Euro filter（Casiez 2012）を cm 空間の x, y にかける（demo_m8.py と同じ設定）。"""

    def __init__(self, min_cutoff=0.8, beta=0.01, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, z, dt):
        z = np.asarray(z, np.float64)
        if self.x is None:
            self.x, self.dx = z, np.zeros(2)
            return z
        self.dx = self.dx + self._a(self.dc, dt) * ((z - self.x) / dt - self.dx)
        self.x = self.x + self._a(self.mc + self.beta * np.abs(self.dx), dt) * (z - self.x)
        return self.x

    def reset(self):
        self.x = self.dx = None


class M8App:
    def __init__(self, model: str, kit: str):
        kit = os.path.abspath(os.path.expanduser(kit))
        sys.path.insert(0, kit)
        import common as C
        self.cfg = C.load_config()
        tg = C.tgaze_dir(self.cfg)       # 撮影キット側の tgaze（実測 K 対応の rich16d）を sys.path の先頭に置いてから読み込む
        import m8_cpu
        self.C, self.m8 = C, m8_cpu
        K, _ = m8_cpu.load_camera_matrix(self.cfg, tg)
        self.rt = m8_cpu.M8Runtime(self.cfg, tg, K)          # 顔の点・目パッチの抽出（と凍結 M8）
        self.alt = None if model in ("frozen", "model") else m8_cpu.OfflineM8(model, K, tg)
        self.needs_patch = self.alt is not None and self.alt.eye_pca is not None
        self.name = "本番 M8（凍結）" if self.alt is None else os.path.basename(os.path.normpath(model))
        self.w, self.h = self.cfg["camera"]["width"], self.cfg["camera"]["height"]
        self.lock = threading.Lock()
        self.filt, self._t_prev, self._ts = OneEuro2D(), None, 0
        self.calib_reset()
        import rich16d
        assert hasattr(rich16d, "set_camera_matrix"), f"古い rich16d を読み込んだ: {rich16d.__file__}"
        self.rich16d_path = rich16d.__file__

    def info(self) -> dict:
        return {"model": self.name, "capture": [self.w, self.h], "zero_shot": True}

    # ------------------------------------------------------------------ 推定
    def _run(self, img):
        """1枚 → (補正前の uv, 顔の情報 dict) か (None, None)。"""
        if img.shape[1] != self.w or img.shape[0] != self.h:
            if abs(img.shape[1] / img.shape[0] - self.w / self.h) > 0.01:
                raise ValueError(f"camera frame {img.shape[1]}x{img.shape[0]} is not {self.w}x{self.h}")
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)   # 同じ縦横比なら拡大して使う
        self._ts += 33                                   # MediaPipe VIDEO モードは時刻が単調増加していればよい
        out = self.rt.process(img, self._ts, want_patch=self.needs_patch)
        if not out["ok"]:
            return None, None
        uv = np.asarray(out["uv"], np.float64)
        if self.alt is not None:
            patch = None if out["patch"] is None else out["patch"][None].astype(np.float64)
            if self.needs_patch and patch is None:
                return None, None
            p, ok = self.alt.predict_lms(out["lms"][None], patch)
            if not ok[0]:
                return None, None
            uv = p[0]
        return uv, {"out": out, "img": img}

    def _checks(self, face):
        if face is None:
            return {"face_full": False, "eyes_open": False, "frontal": False, "bright": False}, None, None
        out, img = face["out"], face["img"]
        lms = out["lms"]                                  # 左右反転した画像での正規化座標
        xs, ys = 1.0 - lms[:, 0], lms[:, 1]               # ブラウザに返すのは反転前の画像の座標
        x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        pitch, yaw, _ = np.degrees(self.m8.kabsch_euler(out["R_head"][None])[0])
        ear = float(np.mean(out["f16"][8:10]))
        roi = img[max(0, int(y0 * self.h)):int(y1 * self.h), max(0, int(x0 * self.w)):int(x1 * self.w)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else roi
        checks = {"face_full": bool(x0 > 0.02 and y0 > 0.02 and x1 < 0.98 and y1 < 0.98),
                  "eyes_open": bool(ear > 0.15),
                  "frontal": bool(abs(yaw) < 20 and abs(pitch) < 20),
                  "bright": bool(gray.size and float(gray.mean()) > 70)}
        return checks, [x0, y0, x1, y1], (float(yaw), float(pitch))

    def frame(self, img: np.ndarray) -> dict:
        with self.lock:
            uv, face = self._run(img)
            checks, bbox, yp = self._checks(face)
            res = {"face": face is not None, "calibrated": self.n_cal > 0, "gaze": None, "checks": checks, "model": self.name}
            if face is None:
                return res
            res["bbox"], res["head_ok"] = bbox, True
            res["yaw_deg"], res["pitch_deg"] = yp
            cal = np.asarray(self.corr.apply(uv), np.float64)
            now = time.monotonic()
            dt = 1 / 30 if self._t_prev is None else max(1e-3, now - self._t_prev)
            self._t_prev = now
            shown = self.filt(self.C.uv_to_cm(cal), dt) / self.C.HALF_CM
            tx, ty = self.C.uv_to_tgaze(shown)
            res["gaze"] = [float(tx), float(ty)]
            return res

    # ------------------------------------------------------------------ キャリブ
    def calib_reset(self) -> dict:
        with self.lock:
            self.corr, self.n_cal, self.samples = self.C.Correction("none"), 0, {}
            self.filt.reset()
        return {"ok": True}

    def calib_sample(self, img: np.ndarray, x: float, y: float) -> dict:
        with self.lock:
            uv, face = self._run(img)
            if uv is not None:
                self.samples.setdefault((x, y), []).append(uv)
            return {"ok": uv is not None, "count": len(self.samples.get((x, y), []))}

    def _fit(self, pts):
        T = np.array([self.C.tgaze_to_uv(np.array(p)) for p, _ in pts])
        P = np.array([np.median(np.array(s), 0) for _, s in pts])
        return self.C.Correction("affine", 25.0).fit(P, T), P, T

    def calib_fit(self) -> dict:
        with self.lock:
            pts = [(p, s) for p, s in self.samples.items() if len(s) >= MIN_POINT_SAMPLES]
            n = sum(len(s) for s in self.samples.values())
            if len(pts) < 3:
                return {"ok": False, "error": f"有効な点が {len(pts)} 個（3個以上必要）", "n_samples": n}
            self.corr, P, T = self._fit(pts)
            self.n_cal = len(pts)
            self.filt.reset()
            # 1点抜きの誤差（画面の幅・高さ = 1 の単位）。uv → [0,1] は x = (u+1)/2, y = (1-v)/2
            d = []
            for i in range(len(pts)):
                c, _, _ = self._fit(pts[:i] + pts[i + 1:])
                e = np.asarray(c.apply(P[i])) - T[i]
                d.append((e[0] / 2, -e[1] / 2))
            d = np.array(d)
            return {"ok": True, "n_samples": n, "n_points": len(self.samples), "valid_points": len(pts),
                    "loo_euc": float(np.mean(np.hypot(d[:, 0], d[:, 1]))),
                    "loo_x": float(np.sqrt(np.mean(d[:, 0] ** 2))), "loo_y": float(np.sqrt(np.mean(d[:, 1] ** 2)))}

    def click(self, x: float, y: float) -> dict:
        return {"ok": True, "note": "M8 版ではクリック補正をしない"}
