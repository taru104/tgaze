"""アプリの特徴・キャリブ モード切り替え。

MODE = '7d'     → 7D + H1Calibration。正面特化(実機1.4cm・使用感◎)。横向きは崩れる。
MODE = '16d'    → 16D + HuberCalibration。横向きが崩れにくい。正面は7Dよりやや落ちる。
MODE = 'hybrid' → 正面(|yaw|<閾値)=7D H1, 横向き(>=閾値)=16D Huber を自動ハード切替。
                  キャリブでは 7D と 16D の両方を並列学習する。正面の使用感 + 横向きの崩れにくさを両取り。

7Dに戻すのも 1行(MODE='7d')。
"""
MODE = '16d'                    # '7d' | '16d' | 'hybrid'  ← 16Dで実機検証中(研究でオフライン全姿勢2.2-2.7cm。要実機確認)
HYBRID_YAW_THRESH_DEG = 10.0    # hybrid: この角度(deg)以上の |yaw| で 16D に切替

# 画像クリップ(目パッチ)版をデフォルトにするか。
# True  = 16D幾何 + 目パッチ(48x32 CLAHE)PCA16。実機で「変なところに飛びにくい」安定性◎(ユーザ選好2026-07-24)。
# False = 現行の16D幾何のみ(点精度LOOは僅かに上だが暴走しやすい)。
# 16Dのみに戻すのも1行(USE_APPEARANCE=False) or `python main_16d.py`。
USE_APPEARANCE = True

# ──── 機種ごとの画面設定 ─────────────────────────────────────────────────────
# HUD の誤差 cm 換算は「実際の窓の大きさ × 画面の cm/px」で行う(窓のままでも全画面でも正しい)。
# SCREEN_* = 画面全体の物理サイズ(cm)と論理解像度(px)、WINDOW_* = 起動時の窓(描画)サイズ。
import platform
import subprocess
import sys


def machine_id() -> str:
    """機種の識別名(実測カメラ行列の保存先・照合に使う)。
    Mac の platform.node() は接続ネットワークで変わるので LocalHostName を使う。"""
    if sys.platform == 'darwin':
        try:
            return subprocess.run(['scutil', '--get', 'LocalHostName'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            pass
    return platform.node()


if sys.platform == 'darwin':
    # MacBook Air 13" (M4) 内蔵画面: CGDisplayScreenSize = 290.6 x 189.0 mm, 論理 1470x956 (パネル 2560x1664 を縮小表示)
    SCREEN_CM_W, SCREEN_CM_H = 29.06, 18.90
    SCREEN_PX_W, SCREEN_PX_H = 1470, 956
    # 全画面で使えるのはノッチ帯(上32pt)を除いた 1470x924。描画をこの縦横比にしないと全画面で右が黒く余る
    try:
        from macos import main_display_fullscreen_size
        _FULL_W, _FULL_H = main_display_fullscreen_size()
    except Exception:
        _FULL_W, _FULL_H = 1470, 924
    WINDOW_W = 1280                                   # F で全画面に切り替え
    WINDOW_H = round(WINDOW_W * _FULL_H / _FULL_W)
else:
    # dynabook P1-M7SD-BW: 1280x720 の窓がほぼ画面全体
    SCREEN_CM_W, SCREEN_CM_H = 30.9, 17.4
    SCREEN_PX_W, SCREEN_PX_H = 1280, 720
    WINDOW_W, WINDOW_H = 1280, 720
