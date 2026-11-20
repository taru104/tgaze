"""macOS (MacBook Air 13" M4) 用の設定と補助処理。元のコードは変えず、Mac 版の起動スクリプトだけが使う。

  main_mac.py / main_camK_mac.py / benchmarks/calibrate_camera_mac.py から import する。
"""
import platform
import subprocess
import sys

# ──── 画面 ────────────────────────────────────────────────────────────────────
# MacBook Air 13" (M4) 内蔵画面: CGDisplayScreenSize = 290.6 x 189.0 mm、論理 1470x956
# (パネル 2560x1664 を縮小表示。窓やマウスの座標はこの論理 px)
SCREEN_CM_W, SCREEN_CM_H = 29.06, 18.90
SCREEN_PX_W, SCREEN_PX_H = 1470, 956
WINDOW_W = 1280          # 起動時の窓幅。高さは全画面域の縦横比から決める(window_size())

# ──── 実測カメラ行列 ─────────────────────────────────────────────────────────
CAMERA_CALIB_DIR = 'results/camera_calib/{w}x{h}_mac'   # リポジトリ直下からの相対


def machine_id() -> str:
    """機種の識別名。Mac の platform.node() は接続ネットワークで変わるので LocalHostName を使う。"""
    try:
        return subprocess.run(['scutil', '--get', 'LocalHostName'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.node()


def disable_center_stage() -> bool:
    """このプロセスのセンターフレーム(Center Stage)をオフにする。切れたら True。

    センターフレームは顔を追ってデジタルにズーム・切り抜きするので、カメラ行列(焦点距離・主点)が
    フレームごとに変わり、実測Kも幾何特徴も狂う。cv2.VideoCapture を開く前に呼ぶ。"""
    if sys.platform != 'darwin':
        return False
    try:
        import AVFoundation as AV
    except ImportError:
        print("[mac] pyobjc-framework-AVFoundation が無いのでセンターフレームを切れない"
              "(コントロールセンター > ビデオエフェクト で手動でオフに)")
        return False
    # 制御モードを App にしないと、アプリからは有効/無効を変えられない(ユーザ操作が優先される)
    AV.AVCaptureDevice.setCenterStageControlMode_(AV.AVCaptureCenterStageControlModeApp)
    AV.AVCaptureDevice.setCenterStageEnabled_(False)
    return not AV.AVCaptureDevice.isCenterStageEnabled()


def fullscreen_size():
    """メイン画面で全画面表示に使える大きさ(pt)。ノッチ付き Mac は上端のノッチ帯(32pt)を除く。"""
    try:
        import AppKit
        scr = AppKit.NSScreen.mainScreen()
        size = scr.frame().size
        return int(size.width), int(size.height - scr.safeAreaInsets().top)
    except Exception:
        return SCREEN_PX_W, SCREEN_PX_H - 32


def window_size():
    """描画(窓)サイズ。全画面域と同じ縦横比にしないと、全画面で右が黒く余る。"""
    fw, fh = fullscreen_size()
    return WINDOW_W, round(WINDOW_W * fh / fw)


def place_window_on_main_display(name: str, win_w: int, top: int = 40) -> None:
    """窓をメイン画面の上部中央に置く。
    OpenCV(Cocoa) の moveWindow は y を「全ディスプレイを合わせた領域の上端」から測るので、
    外部モニターがメイン画面の上にあると窓がそちらへ出たり右にはみ出したりする。その分を補正する。"""
    import cv2
    try:
        import Quartz
        err, ids, n = Quartz.CGGetActiveDisplayList(16, None, None)
        union_top = min(Quartz.CGDisplayBounds(d).origin.y for d in ids[:n])
        main = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        x = int(main.origin.x + max(0, (main.size.width - win_w) // 2))
        y = int(main.origin.y + top - union_top)
    except Exception:
        return
    cv2.moveWindow(name, x, y)
