"""実測カメラ行列 × 高解像度 で main を起動する比較用ランチャー（main.py は非変更）。

  python main_camK.py            # 1280x720 + 実測K (results/camera_calib/1280x720/camera_matrix.json)
  python main_camK.py --no-k     # 1280x720 + 従来の f=画像幅（解像度だけの効果を見る）
  python main.py                 # 従来 640x480 + f=画像幅（基準）

  同梱Kは測ったPC(host一致)でだけ自動使用。別PC(Mac等)では自動で f=画像幅 になる。

HUD の [LOO] と画面外率を3条件で比べる。生ランドマークは logs/ に残るので、
同じ1280x720セッションで K あり/なしをオフライン再計算して比べることもできる。
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import cv2

import rich16d
from main import GazeApp

ROOT = Path(__file__).parent

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--no-k', action='store_true', help='実測Kを使わない(f=画像幅)')
    ap.add_argument('--k-json', default=None)
    a = ap.parse_args()

    path = Path(a.k_json) if a.k_json else ROOT / 'results' / 'camera_calib' / f'{a.width}x{a.height}' / 'camera_matrix.json'
    if not a.no_k and not a.k_json and not path.exists():
        print(f"[camK] {path} が無いので実測Kなし(f=画像幅)で起動")
        a.no_k = True
    if not a.no_k and not a.k_json:
        # 同梱の 1280x720 K は dynabook 内蔵カメラ用。測ったPC以外では自動で使わない
        host = json.loads(path.read_text(encoding='utf-8')).get('host')
        if host != platform.node():
            print(f"[camK] {path.name} は別PC({host})で測ったKなので使わない(f=画像幅で起動)")
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
        app = GazeApp(cam_id=a.cam, win_w=1280, win_h=720)
        cap = app.cap
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 720pの30fpsはMJPGのみ(YUY2は10fps)
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
