"""macOS 固有の処理。センターフレーム(Center Stage)を切る / 窓を内蔵(メイン)画面に置く。

センターフレームは顔を追ってデジタルにズーム・切り抜きするので、カメラ行列(焦点距離・主点)が
フレームごとに変わり、実測Kも幾何特徴も狂う。macOS では cv2.VideoCapture を開く前にこれを呼ぶ。
Mac 以外・PyObjC が無い環境では何もしない。
"""
import sys


def disable_center_stage() -> bool:
    """このプロセスのセンターフレームをオフにする。切れたら True。"""
    if sys.platform != 'darwin':
        return False
    try:
        import AVFoundation as AV
    except ImportError:
        print("[camera] pyobjc-framework-AVFoundation が無いのでセンターフレームを切れない"
              "(コントロールセンター > ビデオエフェクト で手動でオフに)")
        return False
    # 制御モードを App にしないと、アプリからは有効/無効を変えられない(ユーザ操作が優先される)
    AV.AVCaptureDevice.setCenterStageControlMode_(AV.AVCaptureCenterStageControlModeApp)
    AV.AVCaptureDevice.setCenterStageEnabled_(False)
    return not AV.AVCaptureDevice.isCenterStageEnabled()


def place_window_on_main_display(name: str, win_w: int, top: int = 40) -> None:
    """窓をメイン画面の上部中央に置く。Mac 以外では何もしない。
    OpenCV(Cocoa) の moveWindow は y を「全ディスプレイを合わせた領域の上端」から測るので、
    外部モニターがメイン画面の上にあると窓がそちらへ出たり右にはみ出したりする。その分を補正する。"""
    if sys.platform != 'darwin':
        return
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


def main_display_fullscreen_size():
    """メイン画面で全画面表示に使える大きさ(pt)。ノッチ付き Mac は上端のノッチ帯を除く。"""
    import AppKit
    scr = AppKit.NSScreen.mainScreen()
    size = scr.frame().size
    return int(size.width), int(size.height - scr.safeAreaInsets().top)
