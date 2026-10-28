"""MPIIFaceGaze -> MediaPipe landmarks in the production M8 kit's input convention (mirrored frame).

Per subject pXX: every annotated image is read, flipped horizontally (cv2.flip(frame, 1), as m8_cpu.M8Runtime),
and run through the same face_landmarker.task (CPU delegate, num_faces 1, confidence 0.5 = kit config.json).
IMAGE mode (MPIIFaceGaze frames are stills, not a continuous video). Nothing is trained.

  ~/tgaze/.venv/bin/python benchmarks/mpii_m8_extract.py MPII_ROOT OUT_DIR [--limit N] [--subjects p00 p01] [--patch]
-> OUT_DIR/pXX.npz: lms (N,478,3) float32 (NaN = no face), ok, lidx, img_wh, gaze_px (N,2), fc (N,3), gt (N,3) [mm]
   --patch adds eye_patch (N,3072) float16: tgaze appearance.eye_patch of the mirrored frame (= M8Runtime.process),
   needed by the EyePCA models in experiments/phase6_kit/models_alt
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

TGAZE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TGAZE)
TASK = os.path.join(TGAZE, "face_landmarker.task")
CONF = 0.5          # experiments/phase6_kit/config.json camera.mediapipe_min_confidence


def landmarker():
    V = mp.tasks.vision
    return V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK, delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=V.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=CONF, min_face_presence_confidence=CONF, min_tracking_confidence=CONF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mpii")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", default=[f"p{i:02d}" for i in range(15)])
    ap.add_argument("--patch", action="store_true")
    a = ap.parse_args()
    if a.patch:
        from appearance import eye_patch
    os.makedirs(a.out, exist_ok=True)
    lm = landmarker()
    t0 = time.time()
    for pid in a.subjects:
        dst = os.path.join(a.out, f"{pid}.npz")
        if os.path.exists(dst) and not a.limit:
            print(f"{pid}: skip (exists)", flush=True)
            continue
        lines = open(os.path.join(a.mpii, pid, f"{pid}.txt")).read().strip().splitlines()
        if a.limit:
            lines = lines[:a.limit]
        n = len(lines)
        lms = np.full((n, 478, 3), np.nan, np.float32)
        wh = np.zeros((n, 2), np.int32)
        gaze, fc, gt = np.zeros((n, 2)), np.zeros((n, 3)), np.zeros((n, 3))
        patch = np.full((n, 3072), np.nan, np.float16) if a.patch else None
        for i, ln in enumerate(lines):
            f = ln.split()
            gaze[i] = float(f[1]), float(f[2])
            fc[i] = [float(v) for v in f[21:24]]
            gt[i] = [float(v) for v in f[24:27]]
            frame = cv2.imread(os.path.join(a.mpii, pid, f[0]))
            if frame is None:
                continue
            wh[i] = frame.shape[1], frame.shape[0]
            mir = cv2.flip(frame, 1)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(mir, cv2.COLOR_BGR2RGB))))
            if res.face_landmarks:
                lms[i] = [(p.x, p.y, p.z) for p in res.face_landmarks[0]]
                if a.patch:
                    pt = eye_patch(mir, res.face_landmarks[0])
                    if pt is not None:
                        patch[i] = pt.astype(np.float16)
            if (i + 1) % 1000 == 0:
                print(f"  {pid}: {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
        ok = np.isfinite(lms).all((1, 2))
        np.savez_compressed(dst, lms=lms, ok=ok, lidx=np.arange(n), img_wh=wh, gaze_px=gaze, fc=fc, gt=gt,
                            image=np.array([ln.split()[0] for ln in lines]), **({"eye_patch": patch} if a.patch else {}))
        print(f"{pid}: {ok.sum()}/{n} faces  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
