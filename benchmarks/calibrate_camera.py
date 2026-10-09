"""Webカメラのカメラ行列(内部パラメータ)と歪み係数を ChArUco ボードで実測する。

OSはこのカメラの内部パラメータを持っていない(WinRT try_get_camera_intrinsics=None,
汎用UVCドライバ)ので、実測が唯一の確実な手段。

使い方:
  1) ボード画像を作る（A4・100%で印刷。スマホ全画面表示でも可）
       python benchmarks/calibrate_camera.py board
  2) 印刷物のマス1辺を定規で測り(mm)、撮影＋計算
       python benchmarks/calibrate_camera.py capture --square-mm 30.0
     画面: 緑の点=検出コーナー。ボードを色々な位置・距離・傾きで見せると自動で撮る。
     [SPACE]手動撮影 [C]計算して保存 [Q]中止。25枚程度で十分。
  3) 撮った画像から再計算だけしたい時
       python benchmarks/calibrate_camera.py solve --square-mm 30.0 --dir results/camera_calib/640x480

出力: results/camera_calib/<WxH>/camera_matrix.json （K, 歪み, 再投影誤差, 画角）
既定はアプリ(main.py)と同じ cv2.VideoCapture(0) の 640x480。
"""
import argparse
import glob
import json
import os
import platform
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUARES_X, SQUARES_Y = 7, 5          # A4横に30mmマスで収まる
MARKER_RATIO = 0.75
DICT = cv2.aruco.DICT_5X5_100


def make_board(square_mm):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_mm,
                                  square_mm * MARKER_RATIO, d)


def cmd_board(args):
    # A4横 297x210mm @300dpi、マス30mm → 210x150mm の盤面
    dpi = 300
    px_per_mm = dpi / 25.4
    board = make_board(30.0)
    bw, bh = int(SQUARES_X * 30 * px_per_mm), int(SQUARES_Y * 30 * px_per_mm)
    img = board.generateImage((bw, bh), marginSize=0)
    page = np.full((int(210 * px_per_mm), int(297 * px_per_mm)), 255, np.uint8)
    y0, x0 = (page.shape[0] - bh) // 2, (page.shape[1] - bw) // 2
    page[y0:y0 + bh, x0:x0 + bw] = img
    out = os.path.join(ROOT, "results", "camera_calib", "charuco_A4_30mm.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, page)
    # PDFは300dpi指定でA4実寸になる（画像ビューア経由の印刷は拡大縮小・薄くなる事故が起きやすい）
    from PIL import Image
    Image.fromarray(page).convert("1").save(out[:-4] + ".pdf", resolution=float(dpi))
    print(f"saved {out} (+ .pdf)\n  A4横・拡大縮小なし(100%/実際のサイズ)で印刷し、マス1辺を定規で測ってください(設計値30mm)。")


def detect(detector, gray):
    cc, ci, mc, mi = detector.detectBoard(gray)
    return cc, ci


def solve(img_paths, square_mm, label=""):
    board = make_board(square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts, img_pts, used, size = [], [], [], None
    for p in img_paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        cc, ci = detect(detector, gray)
        if ci is None or len(ci) < 8:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        obj_pts.append(op.astype(np.float32)); img_pts.append(ip.astype(np.float32)); used.append(p)
    if len(used) < 8:
        raise SystemExit(f"使える画像が{len(used)}枚しかない(8枚以上必要)")
    w, h = size
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    # 外れ画像を除いて再計算（1回だけ）
    per = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        pr, _ = cv2.projectPoints(op, rv, tv, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, 1)))))
    per = np.array(per)
    keep = per < max(1.0, 3 * np.median(per))
    if keep.sum() >= 8 and keep.sum() < len(per):
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        img_pts = [i for i, k in zip(img_pts, keep) if k]
        used = [u for u, k in zip(used, keep) if k]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    # 主要パラメータの不確かさ（標準偏差）
    _, _, _, _, _, std_int, _, _ = cv2.calibrateCameraExtended(obj_pts, img_pts, size, K.copy(), dist.copy(),
                                                               flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    std = std_int.ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dists = [float(np.linalg.norm(t)) for t in tvecs]
    res = {
        "camera": label or f"{platform.node()} ({platform.platform()})",
        "image_size": [w, h],
        "K": K.tolist(),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "std_fx_fy_cx_cy": std[:4].tolist(),
        "dist_k1_k2_p1_p2_k3": dist.ravel().tolist(),
        "rms_reprojection_px": float(rms),
        "hfov_deg": float(np.degrees(2 * np.arctan(w / (2 * fx)))),
        "vfov_deg": float(np.degrees(2 * np.arctan(h / (2 * fy)))),
        "dfov_deg": float(np.degrees(2 * np.arctan(np.hypot(w / fx, h / fy) / 2))),
        "fx_over_width": fx / w,
        "square_mm": square_mm,
        "n_images": len(used),
        "board_distance_mm_range": [min(dists), max(dists)],
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return res


def cmd_solve(args):
    paths = sorted(glob.glob(os.path.join(args.dir, "img_*.png")))
    res = solve(paths, args.square_mm, args.label)
    out = os.path.join(args.dir, "camera_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    report(res, out)


def report(r, out):
    print(f"\n==== {r['image_size'][0]}x{r['image_size'][1]}  ({r['n_images']}枚) ====")
    print(f"fx={r['fx']:.2f}  fy={r['fy']:.2f}  cx={r['cx']:.2f}  cy={r['cy']:.2f}")
    print(f"  ±std fx,fy,cx,cy = {', '.join(f'{s:.2f}' for s in r['std_fx_fy_cx_cy'])}")
    print(f"dist k1,k2,p1,p2,k3 = {', '.join(f'{v:.4f}' for v in r['dist_k1_k2_p1_p2_k3'])}")
    print(f"再投影誤差 RMS = {r['rms_reprojection_px']:.3f} px  (目安: <0.5 良好)")
    print(f"画角 H={r['hfov_deg']:.1f}°  V={r['vfov_deg']:.1f}°  D={r['dfov_deg']:.1f}°")
    print(f"fx/幅 = {r['fx_over_width']:.3f}  (現コードの仮定 f=画像幅 → 1.000)")
    print(f"ボード距離 {r['board_distance_mm_range'][0]:.0f}〜{r['board_distance_mm_range'][1]:.0f} mm")
    print(f"saved {out}")


def cmd_capture(args):
    w, h = args.width, args.height
    out_dir = os.path.join(ROOT, "results", "camera_calib", f"{w}x{h}")
    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "img_*.png")):
        os.remove(p)
    board = make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("カメラが開けない")
    if frame.shape[1] != w or frame.shape[0] != h:
        raise SystemExit(f"要求 {w}x{h} に対し実際は {frame.shape[1]}x{frame.shape[0]}")
    n_total = SQUARES_X * SQUARES_Y  # 目安
    shots, sigs, last_t = 0, [], 0.0
    # 画面を3x3に分けた位置カバレッジ
    cover = np.zeros((3, 3), int)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci = detect(detector, gray)
        vis = frame.copy()
        n = 0 if ci is None else len(ci)
        auto = False
        if n >= 8:
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 0))
            pts = cc.reshape(-1, 2)
            c = pts.mean(0) / [w, h]
            span = (pts.max(0) - pts.min(0)) / [w, h]
            # 傾き: 左右/上下の見かけ間隔比をざっくり特徴に
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
        # HUD
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
    res = solve(sorted(glob.glob(os.path.join(out_dir, "img_*.png"))), args.square_mm, args.label)
    out = os.path.join(out_dir, "camera_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    report(res, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("board")
    c = sub.add_parser("capture")
    c.add_argument("--square-mm", type=float, required=True)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--cam", type=int, default=0)
    c.add_argument("--label", default="", help="機種名など(JSONに記録)")
    s = sub.add_parser("solve")
    s.add_argument("--square-mm", type=float, required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--label", default="")
    a = ap.parse_args()
    {"board": cmd_board, "capture": cmd_capture, "solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    main()
