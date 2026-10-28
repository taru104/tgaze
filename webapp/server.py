"""Tgaze Web アプリのローカルサーバ（標準ライブラリのみ・追加インストール不要）。

ブラウザがカメラ映像を JPEG で1枚ずつ POST し、サーバが tgaze.GazeTracker で推定して JSON で返す。
映像も結果もこの PC の中だけでやり取りする（外部には送らない）。

    python webapp/server.py            # → http://localhost:8000
    python webapp/server.py --port 8080

API（すべて POST・本文は JPEG）
    /api/frame                 1フレーム推定。顔チェック（カメラ確認ページ用）と視線点を返す
    /api/calib/reset           キャリブレーションをやり直す（本文なし）
    /api/calib/sample?x=&y=    「このフレームで画面の (x, y) を見ていた」（x, y は [0,1]）
    /api/calib/fit             集めたサンプルで学習し、leave-one-point-out 誤差を返す（本文なし）
    /api/click?x=&y=           クリック位置を正解として逐次補正（tracker.record_click）
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tgaze import GazeTracker  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# 顔チェックのしきい値（カメラ確認ページの4項目）
FACE_MARGIN = 0.02        # 顔の外接矩形が画像の端からこれ以上離れていれば「顔全体が映っている」
EAR_OPEN = 0.15           # 目の開き具合（縦/横）がこれ以上なら「目がはっきり見えている」
FRONTAL_DEG = 20.0        # yaw / pitch がこれ以内なら「正面を向いている」
BRIGHT_MIN = 70.0         # 顔領域の平均輝度（0-255）がこれ以上なら「明るい場所」
MIN_POINT_SAMPLES = 5     # 1点あたりこれ以上集まればその点を「有効」とする


class App:
    """1人用の推定状態。MediaPipe の追跡モードは状態を持つので、リクエストは直列に処理する。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.tracker = GazeTracker(appearance=True, video=True)
        self.point_counts: dict[tuple[float, float], int] = {}

    def frame(self, img: np.ndarray) -> dict:
        with self.lock:
            gaze, debug = self.tracker._est.process_frame(img)
            out = {"face": debug is not None, "calibrated": self.tracker.is_calibrated,
                   "gaze": None, "checks": _checks(img, debug)}
            if debug is not None:
                lms = debug["landmarks"]
                xs, ys = [p.x for p in lms], [p.y for p in lms]
                out["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
                out["head_ok"] = bool(debug.get("head_ok", True))
                out["yaw_deg"], out["pitch_deg"] = float(debug["yaw_deg"]), float(debug["pitch_deg"])
            if gaze is not None and self.tracker.is_calibrated:
                out["gaze"] = [float(gaze[0]), float(gaze[1])]
            return out

    def calib_reset(self) -> dict:
        with self.lock:
            self.tracker.reset()
            self.point_counts = {}
        return {"ok": True}

    def calib_sample(self, img: np.ndarray, x: float, y: float) -> dict:
        with self.lock:
            ok = self.tracker.add_calibration_sample(img, (x, y))
            if ok:
                self.point_counts[(x, y)] = self.point_counts.get((x, y), 0) + 1
            return {"ok": ok, "count": self.point_counts.get((x, y), 0)}

    def calib_fit(self) -> dict:
        with self.lock:
            n = sum(self.point_counts.values())
            try:
                self.tracker.fit()
            except ValueError as e:
                return {"ok": False, "error": str(e), "n_samples": n}
            cal = self.tracker._est.calibration
            return {"ok": True, "n_samples": n,
                    "n_points": len(self.point_counts),
                    "valid_points": sum(c >= MIN_POINT_SAMPLES for c in self.point_counts.values()),
                    # 誤差は画面正規化単位（画面幅・高さ = 1.0）。cm への換算はブラウザ側で画面サイズを使って行う
                    "loo_euc": cal.loo_euc, "loo_x": cal.loo_euc_x, "loo_y": cal.loo_euc_y}

    def click(self, x: float, y: float) -> dict:
        with self.lock:
            if self.tracker.is_calibrated:
                self.tracker.record_click((x, y))
        return {"ok": True}


def _checks(img: np.ndarray, debug) -> dict:
    """カメラ確認ページの4項目。顔が取れないときは全部 False。"""
    if debug is None:
        return {"face_full": False, "eyes_open": False, "frontal": False, "bright": False}
    h, w = img.shape[:2]
    lms = debug["landmarks"]
    xs, ys = np.array([p.x for p in lms]), np.array([p.y for p in lms])
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    face = cv2.cvtColor(img[max(0, int(y0 * h)):int(y1 * h), max(0, int(x0 * w)):int(x1 * w)], cv2.COLOR_BGR2GRAY)
    return {"face_full": bool(x0 > FACE_MARGIN and y0 > FACE_MARGIN and x1 < 1 - FACE_MARGIN and y1 < 1 - FACE_MARGIN),
            "eyes_open": bool(debug["ear"] > EAR_OPEN),
            "frontal": bool(abs(debug["yaw_deg"]) < FRONTAL_DEG and abs(debug["pitch_deg"]) < FRONTAL_DEG),
            "bright": bool(face.size and float(face.mean()) > BRIGHT_MIN)}


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, fmt, *args):      # 1フレームごとのアクセスログは出さない
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _image(self):
        n = int(self.headers.get("Content-Length", 0))
        buf = np.frombuffer(self.rfile.read(n), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if n else None

    def do_GET(self):
        path = urlparse(self.path).path
        f = (STATIC / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
        if STATIC not in f.parents or not f.is_file():
            self.send_error(404)
            return
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                         + ("; charset=utf-8" if f.suffix in (".html", ".css", ".js") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: float(v[0]) for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/frame":
                img = self._image()
                return self._json(self.app.frame(img) if img is not None else {"error": "no image"})
            if u.path == "/api/calib/reset":
                return self._json(self.app.calib_reset())
            if u.path == "/api/calib/sample":
                img = self._image()
                return self._json(self.app.calib_sample(img, q["x"], q["y"]) if img is not None else {"ok": False})
            if u.path == "/api/calib/fit":
                return self._json(self.app.calib_fit())
            if u.path == "/api/click":
                return self._json(self.app.click(q["x"], q["y"]))
        except Exception as e:  # 1フレームの失敗でサーバを止めない
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="既定はこのPCからのみ接続可")
    args = ap.parse_args()
    Handler.app = App()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Tgaze web app: http://localhost:{args.port}  (Ctrl+C で終了)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.app.tracker.close()


if __name__ == "__main__":
    main()
