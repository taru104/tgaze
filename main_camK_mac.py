"""Mac 版: 実測カメラ行列 × 1080p で起動する（main_camK.py の Mac 版。main.py / main_camK.py は非変更）。

  python main_camK_mac.py            # 1920x1080 + 実測K (results/camera_calib/1920x1080_mac/camera_matrix.json)
  python main_camK_mac.py --no-k     # 1920x1080 + 従来の f=画像幅（解像度だけの効果を見る）
  python main_mac.py                 # 640x480 + f=画像幅（基準）

  K は benchmarks/calibrate_camera_mac.py で測る。測った Mac(host一致)でだけ自動使用。
  窓・全画面(F)・センターフレームoff は main_mac.MacGazeApp と同じ。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

import macos
import rich16d
from main_mac import MacGazeApp

ROOT = Path(__file__).parent

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--width', type=int, default=1920)    # 内蔵カメラの最高解像度
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--no-k', action='store_true', help='実測Kを使わない(f=画像幅)')
    ap.add_argument('--k-json', default=None)
    a = ap.parse_args()

    path = Path(a.k_json) if a.k_json else \
        ROOT / macos.CAMERA_CALIB_DIR.format(w=a.width, h=a.height) / 'camera_matrix.json'
    if not a.no_k and not a.k_json and not path.exists():
        print(f"[camK] {path} が無いので実測Kなし(f=画像幅)で起動")
        a.no_k = True
    if not a.no_k and not a.k_json:
        host = json.loads(path.read_text(encoding='utf-8')).get('host')
        if host != macos.machine_id():
            print(f"[camK] {path.name} は別の Mac({host})で測ったKなので使わない(f=画像幅で起動)")
            a.no_k = True
    if not a.no_k:
        calib = json.loads(path.read_text(encoding='utf-8'))
        if tuple(calib['image_size']) != (a.width, a.height):
            sys.exit(f"[ERROR] K は {calib['image_size']} 用。解像度 {a.width}x{a.height} とは合わない")
        # main はフレームを左右反転してから推定するので cx を反転
        rich16d.set_camera_matrix(calib['K'], (a.width, a.height), mirrored=True)
        print(f"[camK] 実測K使用 fx={calib['fx']:.1f} fy={calib['fy']:.1f} ({path})")
    else:
        print("[camK] 実測Kなし (f=画像幅)")

    try:
        app = MacGazeApp(cam_id=a.cam)
        cap = app.cap
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        ok, fr = cap.read()
        if not ok or fr.shape[1] != a.width or fr.shape[0] != a.height:
            got = None if not ok else f"{fr.shape[1]}x{fr.shape[0]}"
            sys.exit(f"[ERROR] 要求 {a.width}x{a.height} に対し実際は {got}")
        print(f"[camK] camera {a.width}x{a.height}")
        app.run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
