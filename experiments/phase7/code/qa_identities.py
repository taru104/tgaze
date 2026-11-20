"""Phase 7E identity QA: contact sheet + landmark shape statistics vs the v2 identities (reject |z| > 3.5).
usage: ~/tgaze/.venv/bin/python phase7/qa_identities.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D
import cv2
new = D.load_base(os.path.join(D.PROJECT, "outputs", "gaze_dataset_p7_ids"))
v2 = D.load_base()

def per_id(b):
    out = {}
    W, H = b["geometry"]["W"], b["geometry"]["H"]
    for k in sorted(set(b["ident"])):
        m = (b["ident"] == k) & b["OK"]
        L = b["LMS"][m] * [W, H, W]
        iod = np.linalg.norm(L[:, 33, :2] - L[:, 263, :2], axis=1)            # outer eye corners
        fw = np.linalg.norm(L[:, 234, :2] - L[:, 454, :2], axis=1)            # face width
        fh = np.linalg.norm(L[:, 10, :2] - L[:, 152, :2], axis=1)             # forehead-chin
        ew = np.linalg.norm(L[:, 33, :2] - L[:, 133, :2], axis=1)
        f16 = b["F16"][m]
        out[int(k)] = {"iod_over_face_width": float(np.median(iod / fw)), "face_h_over_w": float(np.median(fh / fw)),
                       "eye_w_over_iod": float(np.median(ew / iod)), "EAR": float(np.median(f16[:, 8:10])),
                       "iris_diam_over_eye_w": float(np.median(f16[:, 12:14])), "eye_depth_m": float(np.median(b["O_c"][m, 2])),
                       "n": int(m.sum())}
    return out
pn, pv = per_id(new), per_id(v2)
keys = [k for k in next(iter(pv.values())) if k != "n"]
ref = {k: (np.mean([v[k] for v in pv.values()]), np.std([v[k] for v in pv.values()])) for k in keys}
flag = {}
for i, v in pn.items():
    z = {k: (v[k] - ref[k][0]) / ref[k][1] for k in keys}
    v["z_vs_v2"] = z
    if max(abs(x) for x in z.values()) > 3.5:
        flag[i] = z
rep = {"v2_reference_mean_std": ref, "new_identities": pn, "flagged_abs_z_gt_3.5": flag,
       "new_vs_v2_spread_ratio": {k: float(np.std([v[k] for v in pn.values()]) / ref[k][1]) for k in keys},
       "generator_rejected": [62, 81], "replacements": {"62": 144, "81": 145}}
os.makedirs(os.path.join(D.OUT_ROOT, "identity_diversity"), exist_ok=True)
D.dump_json(rep, os.path.join(D.OUT_ROOT, "identity_diversity", "qa_identities.json"))
print("flagged:", list(flag), "| spread ratio new/v2:", {k: round(v, 2) for k, v in rep["new_vs_v2_spread_ratio"].items()})
# contact sheet: the sample with the smallest head angle per identity
tiles = []
root = os.path.join(D.PROJECT, "outputs", "gaze_dataset_p7_ids")
for k in sorted(pn):
    m = np.where(new["ident"] == k)[0]
    i = m[np.argmin(new["head_mag"][m] + np.abs(new["UV"][m]).sum(1))]
    p = os.path.join(root, f"identity_{k:02d}", "images", f"{new['sample_id'][i]}.jpg")
    im = cv2.imread(p)
    L = new["LMS"][i] * [im.shape[1], im.shape[0], 1]
    cx, cy = L[:, 0].mean(), L[:, 1].mean()
    t = im[int(cy - 260):int(cy + 260), int(cx - 210):int(cx + 210)]
    t = cv2.resize(t, (168, 208))
    cv2.putText(t, f"{k}" + (" !" if k in flag else ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if k in flag else (0, 0, 0), 1)
    tiles.append(t)
cols = 16
while len(tiles) % cols:
    tiles.append(np.full_like(tiles[0], 128))
sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
cv2.imwrite(os.path.join(D.OUT_ROOT, "identity_diversity", "identities_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("sheet", sheet.shape)
