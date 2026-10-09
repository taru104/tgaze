"""GazeTracker — Tgaze の公開ライブラリAPI。

設計方針:
  - 既存の main.py / estimator*.py / calibration.py は**一切変更しない**。ここは薄いラッパのみ。
  - 使う側が知る必要のある概念は「フレームを渡す」「較正する」「[0,1]の視線点を受け取る」の3つだけ。
  - 較正結果は保存/読込できる(2回目以降は較正なしで即使える = 導入摩擦を下げる)。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from estimator_appearance import AppearanceEstimator   # noqa: E402
from estimator import GazeEstimator                    # noqa: E402

# 既定の9点(時計回り・最後が中央)。main.py の calibration.CALIB_POINTS_9 と同じ考え方。
NINE_POINTS: Tuple[Tuple[float, float], ...] = (
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.9, 0.5), (0.9, 0.9), (0.5, 0.9),
    (0.1, 0.9), (0.1, 0.5), (0.5, 0.5),
)


@dataclass(frozen=True)
class GazePoint:
    """1フレームの推定結果。`x`,`y` は画面正規化座標 [0,1]（左上原点）。"""
    x: float
    y: float
    face_detected: bool
    head_pose_ok: bool

    def as_pixels(self, width: int, height: int) -> Tuple[int, int]:
        return int(round(self.x * width)), int(round(self.y * height))

    def __iter__(self):                      # `x, y = tracker.predict(frame)` を許す
        yield self.x
        yield self.y


class GazeTracker:
    """Webカメラ画像から「画面上のどこを見ているか」を推定する。

    Args:
        appearance: True(既定) で 16D幾何 + 目パッチPCA16。実機で暴走(画面外への飛び)が
            約半分になることを実測しているため既定。False で 16D幾何のみ(軽量)。
        video: True(既定) は **連続したwebカメラ映像** を渡す通常の使い方。MediaPipe を
            追跡モードで動かし、時間平滑も掛けるので滑らかで安定する。
            False は **連続していない静止画** を1枚ずつ独立に処理する用途(データセット評価,
            スクリーンショット採点など)。追跡状態と平滑を切り、フレーム順に結果が依存しなく
            なる。動画に False を使うと不必要にブレ、静止画に True を使うと精度が落ちる。

    Note:
        `predict()` は較正前でも例外を投げず、`face_detected=False` などを返す。
        リアルタイムループの中で分岐を書かずに済むようにするため。
    """

    def __init__(self, appearance: bool = True, video: bool = True):
        self._appearance = bool(appearance)
        self._video = bool(video)
        self._est = (AppearanceEstimator(video=self._video) if appearance
                     else GazeEstimator(video=self._video))

    # ── 推定 ────────────────────────────────────────────────
    def predict(self, frame: np.ndarray) -> GazePoint:
        """BGR フレーム(OpenCV の `cap.read()` の戻り値)→ 推定結果。"""
        if not self._video:
            self._est._smoother.reset()      # 各フレームを独立に扱う(静止画バッチ用)
        gaze, debug = self._est.process_frame(frame)
        if gaze is None:
            return GazePoint(float("nan"), float("nan"),
                             face_detected=bool(debug is not None), head_pose_ok=True)
        return GazePoint(float(gaze[0]), float(gaze[1]), True,
                         bool(debug.get("head_ok", True)) if debug else True)

    # ── 較正 ────────────────────────────────────────────────
    def add_calibration_sample(self, frame: np.ndarray, target: Sequence[float],
                               weight: float = 1.0) -> bool:
        """`frame` を撮った瞬間にユーザが `target`([0,1]の画面座標)を見ていた、と教える。

        自前の較正UIを作るときはこれを使う。戻り値 False は「顔/目が取れなかった」。
        """
        self._est.process_frame(frame)
        return bool(self._est.collect_calibration(np.asarray(target, np.float32), weight))

    def fit(self) -> None:
        """集めた較正サンプルから回帰モデルを学習する。サンプルが5点未満なら ValueError。"""
        self._est.finalize_calibration()

    def calibrate(self, points: Iterable[Sequence[float]] = NINE_POINTS,
                  camera: int = 0, samples_per_point: int = 12,
                  settle_frames: int = 12, window: str = "Tgaze calibration") -> float:
        """フルスクリーン較正を実行する(既定9点)。戻り値は leave-one-point-out 誤差(正規化単位)。

        点が出たら**その点を見つめる**だけ。自動で次へ進む。ESC で中断。
        自前UIを持つアプリは代わりに `add_calibration_sample()` + `fit()` を使う。
        """
        import cv2
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ {camera} を開けませんでした")
        cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        sw, sh = _screen_size()
        try:
            for (tx, ty) in points:
                got = 0
                for i in range(settle_frames + samples_per_point * 4):
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    canvas = np.zeros((sh, sw, 3), np.uint8)
                    px, py = int(tx * sw), int(ty * sh)
                    grow = max(6, 26 - got * 2)
                    cv2.circle(canvas, (px, py), grow, (60, 60, 60), 2)
                    cv2.circle(canvas, (px, py), 7, (0, 220, 255), -1)
                    cv2.putText(canvas, "look at the dot  (ESC to cancel)", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
                    cv2.imshow(window, canvas)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt("較正を中断しました")
                    if i >= settle_frames and self.add_calibration_sample(frame, (tx, ty)):
                        got += 1
                        if got >= samples_per_point:
                            break
        finally:
            cap.release()
            cv2.destroyWindow(window)
        self.fit()
        return float(self.calibration_error or float("nan"))

    # ── 状態 ────────────────────────────────────────────────
    @property
    def is_calibrated(self) -> bool:
        return bool(self._est.is_calibrated)

    @property
    def calibration_error(self) -> Optional[float]:
        """較正時の leave-one-point-out 誤差。**画面正規化単位**(画面幅=1.0)。未較正なら None。

        cm が要るなら `calibration_error_cm()` に実画面サイズを渡す。物理サイズを
        ライブラリ側で仮定しないのは、デバイス非依存を保つため。
        """
        return getattr(self._est.calibration, "loo_euc", None)

    def calibration_error_cm(self, screen_width_cm: float,
                             screen_height_cm: float) -> Optional[float]:
        """較正誤差を cm で返す。画面の物理サイズは呼び出し側が渡す(端末ごとに違うため)。"""
        cal = self._est.calibration
        ex, ey = getattr(cal, "loo_euc_x", None), getattr(cal, "loo_euc_y", None)
        if ex is None or ey is None:
            return None
        return float(np.hypot(ex * screen_width_cm, ey * screen_height_cm))

    def record_click(self, target: Sequence[float]) -> None:
        """ユーザが画面の `target` をクリック/タップした = そこを見ていた、として逐次補正する。

        使うほど賢くなる。較正のやり直しなしにドリフトを吸収するための仕組み。
        """
        self._est.record_tap(np.asarray(target, np.float32))

    def reset(self) -> None:
        self._est.reset_calibration()

    # ── 保存 / 読込 ──────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """較正済みモデルを保存する。次回は `load()` するだけで較正不要。"""
        if not self.is_calibrated:
            raise RuntimeError("未較正のモデルは保存できません")
        with open(path, "wb") as f:
            pickle.dump({"appearance": self._appearance,
                         "calibration": self._est.calibration}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if bool(blob.get("appearance")) != self._appearance:
            raise ValueError("保存時と appearance 設定が違います")
        self._est.calibration = blob["calibration"]

    def close(self) -> None:
        self._est.close()

    def __enter__(self) -> "GazeTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _screen_size() -> Tuple[int, int]:
    """画面サイズを API から取得する(物理サイズをユーザに聞かない = デバイス非依存)。"""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        pass
    try:
        import tkinter
        r = tkinter.Tk(); r.withdraw()
        wh = (r.winfo_screenwidth(), r.winfo_screenheight()); r.destroy()
        return wh
    except Exception:
        return 1920, 1080
