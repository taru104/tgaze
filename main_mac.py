"""Mac (MacBook Air 13" M4) 版ランチャー。main.py は変えずに、Mac で要る所だけ差し替える。

  python main_mac.py            # 640x480 + f=画像幅（main.py と同じ条件）
  python main_camK_mac.py       # 1920x1080 + 実測K（推奨）

main.py との違い:
  - センターフレームを切ってからカメラを開く(顔追従ズームでカメラ行列が変わるのを防ぐ)
  - 窓を内蔵画面の上部中央に置く(外部モニターが上にあっても右にはみ出さない)
  - 描画を全画面域(ノッチ帯を除く 1470x924)と同じ縦横比にし、F で全画面を切り替える
  - HUD/ログの誤差 cm を「実際の窓の大きさ × 内蔵画面の cm/px」で換算する(窓でも全画面でも正しい)
"""
import sys

import cv2

import macos
import main
from main import GazeApp

WINDOW = 'Gaze Estimation'   # main.GazeApp.run が作る窓の名前


class MacGazeApp(GazeApp):
    def __init__(self, cam_id: int = 0, **kw):
        macos.disable_center_stage()
        win_w, win_h = macos.window_size()
        super().__init__(cam_id=cam_id, win_w=win_w, win_h=win_h, **kw)

    def run(self):
        # run() 内の namedWindow は既存の窓をそのまま使うので、先に作って位置を決めておく
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.win_w, self.win_h)
        macos.place_window_on_main_display(WINDOW, self.win_w)
        # run() のキー処理に F(全画面切替) を足す。このプロセス内でだけ waitKey を包む
        wait_key = cv2.waitKey

        def wait_key_with_fullscreen(delay=0):
            key = wait_key(delay)
            if key & 0xFF == ord('f'):
                full = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
            return key

        cv2.waitKey = wait_key_with_fullscreen
        try:
            super().run()
        finally:
            cv2.waitKey = wait_key

    def _view_cm(self):
        """いまの窓(描画領域)の物理サイズ cm。"""
        try:
            _, _, ww, wh = cv2.getWindowImageRect(WINDOW)
            if ww > 0 and wh > 0:
                return ww * macos.SCREEN_CM_W / macos.SCREEN_PX_W, wh * macos.SCREEN_CM_H / macos.SCREEN_PX_H
        except cv2.error:
            pass
        return macos.SCREEN_CM_W, macos.SCREEN_CM_H

    # main.py は誤差 cm をモジュール定数 SCREEN_CM_W/H で換算するので、使う直前にいまの窓の値にする
    def _draw_hud(self, canvas, debug):
        main.SCREEN_CM_W, main.SCREEN_CM_H = self._view_cm()
        super()._draw_hud(canvas, debug)

    def _write_log(self, gaze, debug):
        main.SCREEN_CM_W, main.SCREEN_CM_H = self._view_cm()
        super()._write_log(gaze, debug)


if __name__ == '__main__':
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        MacGazeApp(cam_id=cam_id).run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
