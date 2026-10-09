"""
リアルタイム視線推定アプリケーション。
Webカメラから視線推定し、画面上にガゼポイントを表示する。
ｌｌ
FF
    Q/ESC       : 終了
"""

import csv
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from typing import Optional, Tuple

from estimator   import GazeEstimator
from calibration import CALIB_POINTS_9
from raw_landmark_logger import RawLandmarkLogger


# ──── UI 定数 ────────────────────────────────────────────────────────────────

C_BG     = (20,  20,  20 )
C_GAZE   = (0,   180, 255)
C_ACTIVE = (0,   255, 120)
C_DONE   = (80,  80,  80 )
C_TEXT   = (220, 220, 220)
C_WARN   = (60,  60,  200)

CALIB_TOTAL   = 3.0   # 各点の注視時間 (秒)
CALIB_DISCARD = 1.0   # 最初の破棄時間 (秒)
MULTIPOSE_TOTAL = 12.0  # 多姿勢モード: 各点で頭を振りながら収集する時間 (秒)

SCREEN_CM_W = 30.9    # 画面の物理横幅 (cm)
SCREEN_CM_H = 17.4    # 画面の物理縦幅 (cm)


class GazeApp:
    """視線推定アプリ本体。"""

    def __init__(self, cam_id: int = 0, win_w: int = 1280, win_h: int = 720,
                 use_appearance: Optional[bool] = None):
        self.win_w = win_w
        self.win_h = win_h

        # 推定器の選択: 画像クリップ版(16D幾何+目パッチPCA16, 実機で暴走しにくい) か 16D幾何のみか。
        # 既定は config.USE_APPEARANCE。引数 use_appearance で上書き可(main_16d.py 等)。
        if use_appearance is None:
            try:
                from config import USE_APPEARANCE
                use_appearance = USE_APPEARANCE
            except Exception:
                use_appearance = False
        self._use_appearance = bool(use_appearance)
        if self._use_appearance:
            from estimator_appearance import AppearanceEstimator
            self.estimator = AppearanceEstimator()
        else:
            self.estimator = GazeEstimator()

        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {cam_id} を開けません。")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._state         = 'idle'
        self._calib_idx     = 0
        self._calib_t_start = 0.0
        self._multipose     = False
        self._calib_total   = CALIB_TOTAL
        self._debug_mode    = False

        self._fps_t0  = time.time()
        self._fps_cnt = 0
        self._fps     = 0.0

        self._trail: list = []
        self._trail_max   = 60

        self._click_px: Optional[Tuple[int, int]] = None   # 直近クリック画素座標
        self._click_t:  float = 0.0                        # クリック時刻
        # タップ由来の正解座標。そのフレームのログに1回だけ載せて消す
        self._pending_tap: Optional[Tuple[float, float]] = None

        logs_dir   = Path(__file__).parent / 'logs'
        logs_dir.mkdir(exist_ok=True)
        session_id = time.strftime('%Y%m%d_%H%M%S')
        log_path   = logs_dir / f'session_{session_id}.csv'
        self._log_file   = open(log_path, 'w', newline='', encoding='utf-8')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'time_s', 'gaze_x', 'gaze_y', 'raw_x', 'raw_y',
            'pitch_deg', 'yaw_deg', 'ear', 'blink',
            'face_detected', 'calibrated', 'calib_mgae_deg', 'calib_rmse',
            'calib_point_idx', 'calib_target_x', 'calib_target_y',
            'X_feat', 'Y_feat',
            'loo_mgae_deg', 'loo_euc_cm', 'loo_euc_x_cm', 'loo_euc_y_cm',
            'tap_target_x', 'tap_target_y',
        ])
        self._log_path = log_path
        self._t0 = time.time()

        # 生ランドマークロガー(全478点x,y,z)。将来の任意次元特徴を後から再構成できるよう
        # 生データを丸ごと残す。CSV(加工後)と frame_idx で対応づく。
        self._raw_logger = RawLandmarkLogger(logs_dir / f'session_{session_id}_landmarks')
        self._frame_idx  = 0

    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow('Gaze Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaze Estimation', self.win_w, self.win_h)
        cv2.setMouseCallback('Gaze Estimation', self._on_mouse)

        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        gaze: Optional[np.ndarray] = None
        debug = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] フレーム取得失敗")
                continue

            frame = cv2.flip(frame, 1)

            new_gaze, debug = self.estimator.process_frame(frame)
            if new_gaze is not None:
                gaze = new_gaze

            self._update_fps()
            self._write_log(gaze, debug)
            self._log_raw(frame, gaze, debug)
            self._pending_tap = None
            self._frame_idx += 1

            canvas[:] = C_BG

            self._draw_camera_preview(canvas, frame, debug)  # キャリブ点より先に描画

            if self._state == 'calibrating':
                self._draw_calibration(canvas, gaze)
                self._step_calibration(new_gaze)
            else:
                self._draw_running(canvas, gaze, debug)

            self._draw_hud(canvas, debug)

            cv2.imshow('Gaze Estimation', canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self._start_calibration()
            elif key == ord('m'):
                self._start_calibration(multipose=True)
            elif key == ord('r'):
                self.estimator.reset_calibration()
                self._state = 'idle'
                self._trail.clear()
                print("[INFO] キャリブレーションリセット")
            elif key == ord('d'):
                self._debug_mode = not self._debug_mode

        self.estimator.close()
        self.cap.release()
        cv2.destroyAllWindows()
        self._log_file.close()
        self._raw_logger.close()
        print(f"[INFO] Session log saved: {self._log_path}")
        print(f"[INFO] Raw landmarks saved: {self._raw_logger.bin_path} "
              f"({self._raw_logger.n_written} frames)")

    # ──── マウスコールバック ─────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._state == 'calibrating':
            return  # キャリブ中は無視
        if not self.estimator.is_calibrated:
            return  # キャリブ未完了は無視

        gt = np.array([x / self.win_w, y / self.win_h], dtype=np.float32)
        self.estimator.record_tap(gt)
        self._pending_tap = (float(gt[0]), float(gt[1]))
        self._click_px = (x, y)
        self._click_t  = time.time()
        print(f"[INFO] 動的キャリブ記録: click=({gt[0]:.3f}, {gt[1]:.3f})")

    # ──── キャリブレーション制御 ─────────────────────────────────────────────

    def _start_calibration(self, multipose: bool = False):
        self.estimator.reset_calibration()
        self._state         = 'calibrating'
        self._calib_idx     = 0
        self._calib_t_start = time.time()
        self._multipose     = multipose
        self._calib_total   = MULTIPOSE_TOTAL if multipose else CALIB_TOTAL
        self._trail.clear()
        if multipose:
            print(f"[INFO] 多姿勢キャリブ開始: 各点を見たまま頭をゆっくり回してください（各{MULTIPOSE_TOTAL:.0f}秒）")
        else:
            print(f"[INFO] 9点キャリブレーション開始")

    def _step_calibration(self, gaze: Optional[np.ndarray]):
        if self._calib_idx >= len(CALIB_POINTS_9):
            return

        total   = getattr(self, '_calib_total', CALIB_TOTAL)
        elapsed = time.time() - self._calib_t_start

        if elapsed >= CALIB_DISCARD and self.estimator.face_detected:
            target = CALIB_POINTS_9[self._calib_idx]
            # 多姿勢モードは全フレーム平等(頭の各姿勢を等しく学習)。通常は時間で漸増。
            if getattr(self, '_multipose', False):
                weight = 1.0
            else:
                weight = min((elapsed - CALIB_DISCARD) / (CALIB_TOTAL - CALIB_DISCARD), 1.0)
            self.estimator.collect_calibration(target, weight)

        if elapsed >= total:
            self._calib_idx += 1
            self._calib_t_start = time.time()

            if self._calib_idx >= len(CALIB_POINTS_9):
                try:
                    self.estimator.finalize_calibration()
                    self._state = 'running'
                    print(f"[INFO] キャリブレーション完了 "
                          f"({self.estimator.calibration.n_samples} サンプル)")
                except ValueError as e:
                    print(f"[ERROR] {e}")
                    self._state = 'idle'

    # ──── 描画メソッド ───────────────────────────────────────────────────────

    def _calib_pt_px(self, idx: int) -> Tuple[int, int]:
        pt = CALIB_POINTS_9[idx]
        return int(pt[0] * self.win_w), int(pt[1] * self.win_h)

    def _draw_calibration(self, canvas: np.ndarray, gaze: Optional[np.ndarray]):
        w, h = self.win_w, self.win_h
        n_pts = len(CALIB_POINTS_9)

        for i in range(n_pts):
            px, py = self._calib_pt_px(i)
            if i < self._calib_idx:
                cv2.circle(canvas, (px, py), 12, C_DONE, -1)
            elif i == self._calib_idx:
                t = time.time()
                r = int(18 + 6 * np.sin(t * 6.0))
                cv2.circle(canvas, (px, py), r, C_ACTIVE, 2)
                cv2.circle(canvas, (px, py), 6, C_ACTIVE, -1)
            else:
                cv2.circle(canvas, (px, py), 12, (60, 120, 60), 1)

        total = getattr(self, '_calib_total', CALIB_TOTAL)
        if self._calib_idx < n_pts:
            px, py   = self._calib_pt_px(self._calib_idx)
            elapsed  = time.time() - self._calib_t_start
            progress = min(elapsed / total, 1.0)
            bw = 120
            cv2.rectangle(canvas, (px - bw//2, py+28), (px + bw//2, py+40), (40, 40, 40), -1)
            cv2.rectangle(canvas, (px - bw//2, py+28),
                          (px - bw//2 + int(bw * progress), py+40), C_ACTIVE, -1)

        multipose = getattr(self, '_multipose', False)
        if (time.time() - self._calib_t_start) < CALIB_DISCARD:
            status = "Stabilizing..."
        elif multipose:
            status = "Keep looking & SLOWLY rotate your head (up/down/left/right)"
        else:
            status = "Look at the dot"
        mode = "MULTI-POSE " if multipose else ""
        cv2.putText(canvas, f"{mode}CALIBRATION  {self._calib_idx+1}/{n_pts}: {status}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)
        face_ok    = self.estimator.face_detected
        face_color = (0, 220, 80) if face_ok else (60, 60, 200)
        cv2.putText(canvas, "Face: OK" if face_ok else "Face: NOT FOUND",
                    (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    def _draw_running(self, canvas: np.ndarray, gaze: Optional[np.ndarray],
                      debug: Optional[dict]):
        w, h = self.win_w, self.win_h

        if not self.estimator.is_calibrated:
            cv2.putText(canvas, "Press [C] to start 9-point calibration",
                        (w//2 - 260, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WARN, 2)
            return

        # 動的キャリブレーションのクリックフラッシュ（0.6秒）
        if self._click_px is not None:
            age = time.time() - self._click_t
            if age < 0.6:
                alpha = 1.0 - age / 0.6
                r = int(6 + 18 * (1.0 - alpha))
                col = (int(0 * alpha), int(255 * alpha), int(80 * alpha))
                cv2.circle(canvas, self._click_px, r, col, 2)
                cv2.drawMarker(canvas, self._click_px, col,
                               cv2.MARKER_CROSS, 20, 2)
            else:
                self._click_px = None

        if gaze is not None:
            gx = int(gaze[0] * w)
            gy = int(gaze[1] * h)
            self._trail.append((gx, gy))
            if len(self._trail) > self._trail_max:
                self._trail.pop(0)
            for i in range(1, len(self._trail)):
                alpha = i / len(self._trail)
                c = int(alpha * 180)
                cv2.line(canvas, self._trail[i-1], self._trail[i], (c, c//2, 0), 1)
            cv2.circle(canvas, (gx, gy), 24, (40,  80, 140), -1)
            cv2.circle(canvas, (gx, gy), 22, C_GAZE, 2)
            cv2.circle(canvas, (gx, gy),  4, C_GAZE, -1)

    def _draw_camera_preview(self, canvas: np.ndarray, frame: np.ndarray,
                             debug: Optional[dict] = None):
        ph, pw = 160, 213
        small  = cv2.resize(frame, (pw, ph))
        canvas[8:8+ph, 8:8+pw] = small
        cv2.rectangle(canvas, (8, 8), (8+pw, 8+ph), (60, 60, 60), 1)

        # 頭部姿勢の矢印(実際に7Dで使う rich16d の pitch/yaw)。姿勢が正しく取れているか一目で分かる。
        if debug is not None and debug.get('feat7d') is not None:
            import math
            feat = debug['feat7d']
            pitch, yaw = float(feat[4]), float(feat[5])
            cx, cy = 8 + pw // 2, 8 + ph // 2
            L = 55
            dx = int(-math.sin(yaw)  * L)     # 頭を向けた方向に水平に伸びる(符号反転済)
            dy = int(-math.sin(pitch) * L)    # 上を向くと上に伸びる
            cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)
            cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy + dy),
                            (0, 255, 255), 2, tipLength=0.3)
            # 横向きで姿勢が荒れている(|yaw|大)ときは赤で警告色の数値
            col = (0, 200, 255) if abs(math.degrees(yaw)) < 25 else (60, 120, 255)
            cv2.putText(canvas, f"P{math.degrees(pitch):+.0f} Y{math.degrees(yaw):+.0f}deg",
                        (10, 8 + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    def _draw_hud(self, canvas: np.ndarray, debug: Optional[dict]):
        w, h = self.win_w, self.win_h

        if getattr(self, '_use_appearance', False):
            cv2.putText(canvas, "APPEARANCE (image-clip)", (w // 2 - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(canvas, f"FPS: {self._fps:.1f}",
                    (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)

        if self.estimator.is_calibrated:
            cal = self.estimator.calibration
            n   = cal.n_samples
            if cal.loo_euc_x is not None:
                err_x_cm = cal.loo_euc_x * SCREEN_CM_W
                err_y_cm = cal.loo_euc_y * SCREEN_CM_H
                import math
                euc_cm   = math.sqrt(err_x_cm**2 + err_y_cm**2)
                err_str  = (f"  [LOO: {euc_cm:.1f}cm  "
                            f"(X={err_x_cm:.1f} Y={err_y_cm:.1f})  "
                            f"MGAE={cal.loo_mgae:.1f}deg  n={n}]")
            else:
                err_str = f"  [n={n}]"
            cv2.putText(canvas, f"CALIBRATED{err_str}", (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)
        else:
            cv2.putText(canvas, "NOT CALIBRATED", (10, h-52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WARN, 2)
            cv2.putText(canvas, "C = calibrate (face front, hold still)",
                        (10, h-32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1)
            cv2.putText(canvas, "M = multi-pose: look at each dot & SLOWLY rotate head (all angles)",
                        (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # 頭部姿勢ゲート警告
        if debug and not debug.get('head_ok', True):
            dp = debug.get('head_delta_pitch', 0.0)
            dy = debug.get('head_delta_yaw',   0.0)
            cv2.putText(canvas,
                        f"LOW CONFIDENCE  dPitch:{dp:+.1f}deg  dYaw:{dy:+.1f}deg",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1)

        # 瞬き表示
        if debug and debug.get('blink_detected', False):
            cv2.putText(canvas, "BLINK", (w//2 - 40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        if self._debug_mode and debug:
            lines = []
            if 'pitch_deg' in debug:
                lines.append(f"Pitch: {debug['pitch_deg']:+.1f}deg  Yaw: {debug['yaw_deg']:+.1f}deg")
            if 'head_delta_pitch' in debug:
                lines.append(
                    f"Head delta  dPitch:{debug['head_delta_pitch']:+.1f}deg  "
                    f"dYaw:{debug['head_delta_yaw']:+.1f}deg  "
                    f"OK:{debug.get('head_ok', True)}"
                )
            if 'ear' in debug:
                lines.append(f"EAR:{debug['ear']:.3f}  Blink:{debug.get('blink_detected', False)}")
            if 'raw_pred' in debug:
                p = debug['raw_pred']
                lines.append(f"Pred (before smooth): ({p[0]:.3f}, {p[1]:.3f})")

            y0 = h - 45 - len(lines) * 22
            for i, ln in enumerate(lines):
                cv2.putText(canvas, ln, (10, y0 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 160), 1)

    def _write_log(self, gaze: Optional[np.ndarray], debug: Optional[dict]) -> None:
        mgae   = self.estimator.calibration.train_mgae
        rmse   = self.estimator.calibration.train_rmse
        pitch  = debug.get('pitch_deg', '') if debug else ''
        yaw    = debug.get('yaw_deg',   '') if debug else ''
        ear    = debug.get('ear',       '') if debug else ''
        blink  = debug.get('blink_detected', '') if debug else ''
        raw    = debug.get('raw_pred',  None) if debug else None
        x_feat = debug.get('X_feat',   None) if debug else None
        y_feat = debug.get('Y_feat',   None) if debug else None

        # キャリブ中のターゲット座標（正解）
        if self._state == 'calibrating' and self._calib_idx < len(CALIB_POINTS_9):
            cal_idx = self._calib_idx
            cal_pt  = CALIB_POINTS_9[cal_idx]
            cal_tx, cal_ty = f"{cal_pt[0]:.4f}", f"{cal_pt[1]:.4f}"
        else:
            cal_idx, cal_tx, cal_ty = '', '', ''

        # LOO-CV 誤差 (cm換算)
        import math
        cal = self.estimator.calibration
        if cal.loo_euc_x is not None:
            loo_x_cm   = cal.loo_euc_x * SCREEN_CM_W
            loo_y_cm   = cal.loo_euc_y * SCREEN_CM_H
            loo_euc_cm = math.sqrt(loo_x_cm**2 + loo_y_cm**2)
            loo_mgae_s = f"{cal.loo_mgae:.3f}"
            loo_euc_s  = f"{loo_euc_cm:.3f}"
            loo_x_s    = f"{loo_x_cm:.3f}"
            loo_y_s    = f"{loo_y_cm:.3f}"
        else:
            loo_mgae_s = loo_euc_s = loo_x_s = loo_y_s = ''

        self._log_writer.writerow([
            f"{time.time() - self._t0:.3f}",
            f"{gaze[0]:.4f}"  if gaze is not None else '',
            f"{gaze[1]:.4f}"  if gaze is not None else '',
            f"{raw[0]:.4f}"   if raw  is not None else '',
            f"{raw[1]:.4f}"   if raw  is not None else '',
            f"{pitch:.2f}"    if isinstance(pitch, float) else '',
            f"{yaw:.2f}"      if isinstance(yaw,   float) else '',
            f"{ear:.3f}"      if isinstance(ear,   float) else '',
            int(blink)        if isinstance(blink, bool)  else '',
            int(self.estimator.face_detected),
            int(self.estimator.is_calibrated),
            f"{mgae:.3f}"     if mgae is not None else '',
            f"{rmse:.4f}"     if rmse is not None else '',
            cal_idx, cal_tx, cal_ty,
            f"{x_feat:.5f}" if x_feat is not None else '',
            f"{y_feat:.5f}" if y_feat is not None else '',
            loo_mgae_s, loo_euc_s, loo_x_s, loo_y_s,
            f"{self._pending_tap[0]:.4f}" if self._pending_tap else '',
            f"{self._pending_tap[1]:.4f}" if self._pending_tap else '',
        ])

    def _log_raw(self, frame: np.ndarray, gaze: Optional[np.ndarray],
                 debug: Optional[dict]) -> None:
        """全ランドマークを生ログに追記。顔未検出フレームはスキップ(欠番)。"""
        if not debug:
            return
        lms = debug.get('landmarks')
        if lms is None:
            return
        h, w = frame.shape[:2]
        # キャリブ中の正解ターゲット(あれば)を一緒に残す → 将来の再学習で直接使える
        target = None
        if self._state == 'calibrating' and self._calib_idx < len(CALIB_POINTS_9):
            target = CALIB_POINTS_9[self._calib_idx]
        elif self._pending_tap is not None:
            target = self._pending_tap
        self._raw_logger.log(
            frame_idx=self._frame_idx,
            time_s=time.time() - self._t0,
            img_w=w, img_h=h,
            landmarks=lms,
            target=target,
            gaze=gaze,
        )

    def _update_fps(self):
        self._fps_cnt += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._fps     = self._fps_cnt / elapsed
            self._fps_cnt = 0
            self._fps_t0  = time.time()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        app = GazeApp(cam_id=cam_id, win_w=1280, win_h=720)
        app.run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
