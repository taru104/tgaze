"""Mac 版: Webカメラのカメラ行列を ChArUco ボードで実測する（calibrate_camera.py の Mac 版。元は非変更）。

calibrate_camera.py との違い:
  - 既定が内蔵カメラの最高解像度 1920x1080
  - センターフレームを切ってから撮る(顔追従ズーム中は K が一定にならない)
  - 保存先が results/camera_calib/<WxH>_mac/ (別PCで測った K や撮影画像を上書きしない)
  - host に macos.machine_id() を記録(main_camK_mac.py はこれで照合する)
  - OpenCV 5 の detectBoard は角点を (N,2) で返すので、4.x と同じ (N,1,2) に直す

使い方 (ボード画像は calibrate_camera.py board で作る):
  python benchmarks/calibrate_camera_mac.py capture --square-mm 30.0
  python benchmarks/calibrate_camera_mac.py solve --square-mm 30.0   # 撮った画像から再計算だけ
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import calibrate_camera as base  # noqa: E402
import macos  # noqa: E402

_detect_orig = base.detect


def _detect(detector, gray):
    cc, ci = _detect_orig(detector, gray)
    if cc is not None:
        cc = cc.reshape(-1, 1, 2)   # OpenCV 5 対応(描画関数などは (N,1,2) を要求)
    return cc, ci


base.detect = _detect   # base.solve もこれを使う


def out_dir_for(w, h):
    return os.path.join(base.ROOT, macos.CAMERA_CALIB_DIR.format(w=w, h=h))


def save(res, out_dir):
    res["host"] = macos.machine_id()
    out = os.path.join(out_dir, "camera_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    base.report(res, out)


def cmd_capture(args):
    w, h = args.width, args.height
    out_dir = out_dir_for(w, h)
    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "img_*.png")):
        os.remove(p)
    board = base.make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    macos.disable_center_stage()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("カメラが開けない")
    if frame.shape[1] != w or frame.shape[0] != h:
        raise SystemExit(f"要求 {w}x{h} に対し実際は {frame.shape[1]}x{frame.shape[0]}")
    shots, sigs, last_t = 0, [], 0.0
    # 画面を3x3に分けた位置カバレッジ
    cover = np.zeros((3, 3), int)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci = _detect(detector, gray)
        vis = frame.copy()
        n = 0 if ci is None else len(ci)
        auto = False
        if n >= 8:
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 0))
            pts = cc.reshape(-1, 2)
            c = pts.mean(0) / [w, h]
            span = (pts.max(0) - pts.min(0)) / [w, h]
            sig = np.array([c[0], c[1], span[0], span[1]])
            novel = all(np.linalg.norm(sig - s) > 0.08 for s in sigs)
            auto = novel and n >= 12 and time.time() - last_t > 0.7
        key = cv2.waitKey(1) & 0xFF
        if (auto or key == ord(' ')) and n >= 8:
            cv2.imwrite(os.path.join(out_dir, f"img_{shots:03d}.png"), frame)
            sigs.append(sig); shots += 1; last_t = time.time()
            gx, gy = min(int(c[0] * 3), 2), min(int(c[1] * 3), 2)
            cover[gy, gx] += 1
            vis[:] = 255 - vis  # 撮影フラッシュ
        for gy in range(3):
            for gx in range(3):
                col = (0, 200, 0) if cover[gy, gx] else (0, 0, 255)
                cv2.rectangle(vis, (gx * w // 3, gy * h // 3), ((gx + 1) * w // 3, (gy + 1) * h // 3), col, 1)
        msg = f"shots {shots}  corners {n}  | move/tilt board, red cells = not yet | C=solve Q=quit"
        cv2.putText(vis, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("camera calibration", vis)
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if key == ord('c'):
            break
    cap.release(); cv2.destroyAllWindows()
    save(base.solve(sorted(glob.glob(os.path.join(out_dir, "img_*.png"))), args.square_mm, args.label), out_dir)


def cmd_solve(args):
    out_dir = out_dir_for(args.width, args.height)
    save(base.solve(sorted(glob.glob(os.path.join(out_dir, "img_*.png"))), args.square_mm, args.label), out_dir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "solve"):
        p = sub.add_parser(name)
        p.add_argument("--square-mm", type=float, required=True)
        p.add_argument("--width", type=int, default=1920)
        p.add_argument("--height", type=int, default=1080)
        p.add_argument("--label", default="MacBook Air 13 M4 内蔵カメラ (センターフレームoff)")
        if name == "capture":
            p.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    {"capture": cmd_capture, "solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    main()
