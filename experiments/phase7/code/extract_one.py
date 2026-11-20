"""Per-identity MediaPipe extraction (same recipe as scripts/extract_landmarks_v2.py) followed by PNG -> JPEG
(q92) to save disk (Phase 7E identity-diversity data; landmarks are always extracted from the lossless PNG).
usage: ~/tgaze/.venv/bin/python phase7/extract_one.py IDENTITY_DIR [workers]"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import extract_landmarks_v2 as ex

d = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.join(d, "landmarks.npz")
recs = [json.loads(l) for l in open(os.path.join(d, "metadata.jsonl"))]
if not (os.path.exists(out) and len(np.load(out)["sample_id"]) == len(recs)):
    st = json.load(open(os.path.join(d, "scene_static.json")))
    t0 = time.time()
    with Pool(workers, initializer=ex._init, initargs=(st["camera"]["measured_K_px_scaled"], tuple(st["camera"]["resolution"]))) as pool:
        res = pool.map(ex._work, [os.path.join(d, "images", r["image"]) for r in recs], chunksize=8)
    n = len(recs)
    ok = np.array([r is not None for r in res])
    lms = np.zeros((n, 478, 3), np.float32); f16 = np.zeros((n, 16), np.float32); patch = np.zeros((n, 3072), np.float32)
    for i, r in enumerate(res):
        if r is not None:
            lms[i], f16[i], patch[i] = r
    np.savez(out, sample_id=np.array([r["sample_id"] for r in recs]), ok=ok, lms=lms, f16=f16, patch=patch)
    print(f"[extract] {d}: {ok.sum()}/{n} ok in {time.time() - t0:.0f}s", flush=True)
import cv2
for r in recs:
    p = os.path.join(d, "images", r["image"])
    if os.path.exists(p) and p.endswith(".png"):
        cv2.imwrite(p[:-4] + ".jpg", cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 92])
        os.remove(p)
print(f"[jpeg] {d} done", flush=True)
