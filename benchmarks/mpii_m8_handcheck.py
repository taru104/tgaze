"""Hand check of the geometry used by mpii_m8_eval.py, written independently (only numpy / cv2 / scipy.io, no import
of the evaluation code). For the first 3 images of one subject:
  A. screen pixel -> camera 3D with monitorPose / screenSize, compared with the annotated gt (should be ~0 mm)
  B. a known point in front of the camera (the annotated gt and the face centre fc) -> pixel with K -> mirrored pixel
     -> back-projected with the mirrored K at the known depth -> x flipped back: should return the point (~0 mm)
  C. the annotated gt seen from the eye centre: intersect the ray with the M8 display (the frozen model's display,
     mirrored camera frame) -> uv; then uv -> display 3D -> ray -> the person's screen plane: should return gt (~0 mm)

  python benchmarks/mpii_m8_handcheck.py MPII_ROOT pXX EVAL_JSON
"""
import json
import os
import sys

import cv2
import numpy as np
import scipy.io as sio

mpii, pid, ev = sys.argv[1:4]
cal = os.path.join(mpii, pid, "Calibration")
K = sio.loadmat(os.path.join(cal, "Camera.mat"))["cameraMatrix"].astype(float)
mp_ = sio.loadmat(os.path.join(cal, "monitorPose.mat"))
ss = sio.loadmat(os.path.join(cal, "screenSize.mat"))
Rm = cv2.Rodrigues(np.ravel(mp_["rvects"]).astype(float))[0]
Tm = np.ravel(mp_["tvecs"]).astype(float)
wmm, hmm = float(ss["width_mm"][0, 0]), float(ss["height_mm"][0, 0])
wpx, hpx = float(ss["width_pixel"][0, 0]), float(ss["height_pixel"][0, 0])
prep = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "experiments", "phase6_kit", "model", "final_model_preprocessing.json")))["display"]
C = np.array(prep["center_cam_m"]) * 1000
Rd, Ud, Nd = (np.array(prep[k]) for k in ("right_cam_m", "up_cam_m", "normal_cam_m"))
W2, H2 = 145.3, 94.5                                     # half width / height of the M8 display, mm (uv_scale_cm)
eyes = {f["image"]: f for f in json.load(open(ev))["checks"][pid]["frames"]}
FLIP = np.array([-1.0, 1.0, 1.0])
W = 1280

print(f"{pid}: K fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}; screen {wmm}x{hmm} mm / {wpx:.0f}x{hpx:.0f} px")
print(f"monitorPose R=\n{np.round(Rm, 5)}\nT={np.round(Tm, 3)} mm\n")
lines = open(os.path.join(mpii, pid, f"{pid}.txt")).read().split("\n")[:3]
for ln in lines:
    f = ln.split()
    gx, gy = float(f[1]), float(f[2])
    fc = np.array(f[21:24], float)
    gt = np.array(f[24:27], float)
    # A
    s = np.array([gx / wpx * wmm, gy / hpx * hmm, 0.0])
    gt_from_px = Rm @ s + Tm
    # B: known 3D points -> pixel -> mirrored pixel -> mirrored-K back-projection at the known depth -> unflip
    errB = []
    for X in (gt, fc):
        uvp = K @ X / X[2]
        um = W - uvp[0]                                     # mirrored pixel column
        Km_cx = W - K[0, 2]
        xm = (um - Km_cx) / K[0, 0] * X[2]
        ym = (uvp[1] - K[1, 2]) / K[1, 1] * X[2]
        errB.append(np.linalg.norm(np.array([-xm, ym, X[2]]) - X))
    # C: gt through the M8 display and back
    E = np.array(eyes[f[0]]["E_cam_mm"])
    e_m, g_m = E * FLIP, gt * FLIP                          # mirrored frame
    d = g_m - e_m
    t = ((C - e_m) @ Nd) / (d @ Nd)
    p = e_m + t * d - C
    u, v = p @ Rd / W2, p @ Ud / H2                         # uv of gt on the M8 display
    P = (C + u * W2 * Rd + v * H2 * Ud) * FLIP              # back to raw frame
    n = Rm[:, 2]
    dd = P - E
    tt = ((Tm - E) @ n) / (dd @ n)
    Q = E + tt * dd
    back = Rm.T @ (Q - Tm)
    print(f"{f[0]}: gaze px ({gx:.0f},{gy:.0f}) -> screen ({s[0]:.2f},{s[1]:.2f}) mm")
    print(f"  A  px->3D {np.round(gt_from_px, 4)} vs annotated gt {gt}  |diff| {np.linalg.norm(gt_from_px - gt):.2e} mm")
    print(f"  B  pixel/mirror round trip of gt, fc: {errB[0]:.2e}, {errB[1]:.2e} mm")
    print(f"  C  eye E {np.round(E, 2)} -> gt on the M8 display uv ({u:.4f},{v:.4f}) [eval json {np.round(eyes[f[0]]['uv_of_gt_on_m8_display'], 4)}]"
          f" -> back on the screen ({back[0]:.4f},{back[1]:.4f},{back[2]:.1e}) mm, |diff| {np.linalg.norm(back[:2] - s[:2]):.2e} mm")
    print(f"     M8 prediction uv {np.round(eyes[f[0]]['m8_uv'], 4)} -> screen {np.round(eyes[f[0]]['pred_screen_mm_pnp'], 2)} mm"
          f" (truth {np.round(eyes[f[0]]['gt_screen_mm'], 2)})")
