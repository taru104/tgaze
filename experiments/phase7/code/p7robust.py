"""Phase 7F robustness: camera K, MediaPipe landmark noise, lens distortion, mirroring.

  ~/tgaze/.venv/bin/python phase7/p7robust.py landmark EXP [EXP ...]     # landmark-space tests (all 20 ids)
  ~/tgaze/.venv/bin/python phase7/p7robust.py image EXP [EXP ...]        # image-space tests (fold-0 test ids)

Each identity is predicted by the seed-0 model of the fold in which it is a TEST identity (unknown person).

landmark-space (cheap, all 20,000 samples):
  K_feature_only : images unchanged, the K used for features (ncam + tgaze rich16d) is wrong:
                   fx, fy x (1 +- 1/3/5 %), cx / cy +- 2/5/10/20 px          -> measurement error of K
  noise_*        : Gaussian noise on MediaPipe landmarks, sigma in pixels of the 1920x1080 frame
                   (x, y; z gets the same sigma in its x-like units), applied to: all points, iris only,
                   eyelids / eye contours only, rigid (head-pose) subset only, whole-face rigid shift
image-space (MediaPipe re-run on modified renders, fold-0 test identities, 4,000 images):
  K_true_geometry: the camera really has K' (exact: a pinhole render with K' = affine warp K' K^-1 of the
                   render); features use K' (correctly measured) or the nominal K (mismatch)
  distortion     : measured Mac distortion (tgaze camera_matrix.json) x 0 / 0.5 / 1 / 1.5 applied to the raw
                   render; MediaPipe on the distorted image vs on cv2.undistort()ed image
  mirror         : unmirrored input (flip omitted) -> left/right must reverse (automatic check)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7infer as I  # noqa: E402
import landmark_features as lf  # noqa: E402

TGAZE = os.path.expanduser("~/tgaze")
OUT = os.path.join(D.OUT_ROOT, "robustness")
IRIS = list(range(468, 478))
LIDS = lf.EYE_A + lf.EYE_B


def fold_models(exp, base, seed=0):
    runners, owner = {}, np.full(len(base["UV"]), -1)
    for f in range(5):
        rd = os.path.join(D.OUT_ROOT, exp, f"fold{f}_seed{seed}")
        runners[f] = I.Runner(rd, base)
        owner[np.isin(base["ident"], runners[f].result["split"]["test"])] = f
    return runners, owner


def predict_all(runners, owner, idx, lms, f16, K_mir=None, patch=None):
    out = np.zeros((len(idx), 2))
    for f, r in runners.items():
        m = owner[idx] == f
        if m.any():
            out[m] = r.predict(lms[m], f16[m], K_mir, None if patch is None else patch[m].astype(np.float32))
    return out


def summarize(base, idx, pred, ref=None):
    cm, ang = D.errors(base, idx, pred)
    pi = D.pose_invariance(base, idx, pred)
    o = {"cm_mean": float(cm.mean()), "cm_p95": float(np.percentile(cm, 95)), "deg_mean": float(ang.mean()),
         "pose_inv_cm": pi["mean"] if pi else None}
    if ref is not None:
        g = D.Geometry(base["geometry"])
        sh = (pred - ref) * g.uv_cm
        o["shift_vs_clean_cm"] = float(np.hypot(sh[:, 0], sh[:, 1]).mean())
        o["mean_bias_vs_clean_cm"] = (sh.mean(0)).tolist()
    return o


def k_variants():
    v = [("nominal", {})]
    for p in (1, 3, 5):
        for s in (+1, -1):
            v.append((f"f{'+' if s > 0 else '-'}{p}%", {"f": 1 + s * p / 100}))
    for px in (2, 5, 10, 20):
        for s in (+1, -1):
            v.append((f"cx{'+' if s > 0 else '-'}{px}px", {"cx": s * px}))
            v.append((f"cy{'+' if s > 0 else '-'}{px}px", {"cy": s * px}))
    return v


def perturb_K(K, p):
    K = np.array(K, np.float64).copy()
    K[0, 0] *= p.get("f", 1.0)
    K[1, 1] *= p.get("f", 1.0)
    K[0, 2] += p.get("cx", 0.0)
    K[1, 2] += p.get("cy", 0.0)
    return K


def landmark_tests(exps):
    """Each perturbation is computed once (landmarks + tgaze rich16d) and scored with every model."""
    import torch
    torch.set_num_threads(4)
    base = D.load_base()
    g = base["geometry"]
    W, H = g["W"], g["H"]
    idx = np.where(base["OK"])[0]
    lms0, f160, pch0 = base["LMS"][idx], base["F16"][idx], base["PATCH"][idx]
    models = {e: fold_models(e, base) for e in exps}
    clean = {e: predict_all(*models[e], idx, lms0, f160, patch=pch0) for e in exps}
    res = {e: {"clean": summarize(base, idx, clean[e]), "K_feature_only": {}, "landmark_noise": {}} for e in exps}
    t0 = time.time()
    for name, p in k_variants():
        Kr = perturb_K(g["K_measured_raw"], p)
        f16, _ = I.rich16d_batch(lms0, Kr, W, H)
        for e in exps:
            res[e]["K_feature_only"][name] = summarize(base, idx, predict_all(*models[e], idx, lms0, f16, lf.mirrored_K(Kr, W), pch0), clean[e])
    print(f"[landmark] K done ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(7)
    subsets = {"all": None, "iris": IRIS, "eyelids": LIDS, "rigid_head_subset": lf.RIGID}
    for sname, sub in subsets.items():
        for sig in (0.1, 0.25, 0.5, 1.0, 2.0):
            acc = {e: [] for e in exps}
            for rep in range(2):
                noise = rng.normal(0, sig, size=lms0.shape) / np.array([W, H, W])
                if sub is not None:
                    mask = np.zeros(478, bool)
                    mask[sub] = True
                    noise[:, ~mask] = 0
                lms = (lms0 + noise).astype(np.float32)
                f16, _ = I.rich16d_batch(lms, g["K_measured_raw"], W, H)
                for e in exps:
                    acc[e].append(summarize(base, idx, predict_all(*models[e], idx, lms, f16, patch=pch0), clean[e]))
            for e in exps:
                res[e]["landmark_noise"][f"{sname}_{sig}px"] = {k: float(np.mean([v[k] for v in acc[e]])) for k in ("cm_mean", "cm_p95", "deg_mean", "pose_inv_cm", "shift_vs_clean_cm")}
        print(f"[landmark] noise {sname} done ({time.time() - t0:.0f}s)", flush=True)
    for sig in (0.5, 1.0, 2.0, 5.0):
        sh = rng.normal(0, sig, size=(len(lms0), 1, 2)) / np.array([W, H])
        lms = lms0.copy()
        lms[..., :2] += sh
        f16, _ = I.rich16d_batch(lms, g["K_measured_raw"], W, H)
        for e in exps:
            res[e]["landmark_noise"][f"face_shift_{sig}px"] = summarize(base, idx, predict_all(*models[e], idx, lms, f16, patch=pch0), clean[e])
    print(f"[landmark] all done ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(OUT, "landmark_tests.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    D.dump_json(old, path)


# ------------------------------------------------------------------------------------------ image space
_lm = None


def _init():
    global _lm
    import mediapipe as mp
    _lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(TGAZE, "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1))


def _work(job):
    """job = (path, kind, param) -> lms (478,3) of the MIRRORED processed frame or None."""
    import cv2
    import mediapipe as mp
    path, kind, prm = job
    img = cv2.imread(path)
    if kind == "warpK":                      # camera with K' : x' = A x  (A = K' K^-1, affine)
        img = cv2.warpAffine(img, np.array(prm["A"], np.float64)[:2], (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    elif kind in ("distort", "distort_undistort"):
        mx, my = _distort_maps(tuple(img.shape[:2]), prm["K"], prm["dist"])
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if kind == "distort_undistort":
            img = cv2.undistort(img, np.array(prm["K"]), np.array(prm["dist"]))
    flip = prm.get("flip", True) if isinstance(prm, dict) else True
    if flip:
        img = cv2.flip(img, 1)
    res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks or len(res.face_landmarks[0]) < 478:
        return None
    sys.path.insert(0, TGAZE)
    from appearance import eye_patch
    patch = eye_patch(img, res.face_landmarks[0])
    if patch is None:
        return None
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float32), np.asarray(patch, np.float32)


_maps = {}


def _distort_maps(shape, K, dist):
    """remap maps producing the DISTORTED image from an ideal pinhole render (for each distorted pixel,
    where it comes from in the undistorted image)."""
    import cv2
    key = (shape, tuple(np.ravel(dist)))
    if key not in _maps:
        h, w = shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, np.array(K), np.array(dist), P=np.array(K)).reshape(h, w, 2)
        _maps[key] = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))
    return _maps[key]


def image_tests(exps, workers=10, fold=0, max_per_id=None):
    base = D.load_base()
    g = base["geometry"]
    W, H = g["W"], g["H"]
    K0 = np.array(g["K_measured_raw"])
    Kr = np.array(g["K_render_raw"])                   # the render's true pinhole K
    calib = json.load(open(os.path.join(TGAZE, "results", "camera_calib", "1920x1080_mac", "camera_matrix.json")))
    dist0 = np.array(calib["dist_k1_k2_p1_p2_k3"], np.float64)
    sp = D.cv_split(fold)
    idx = np.where(np.isin(base["ident"], sp["test"]) & base["OK"])[0]
    if max_per_id:
        idx = np.concatenate([idx[base["ident"][idx] == k][:max_per_id] for k in sp["test"]])
    paths = [os.path.join(D.DATA_V2, f"identity_{int(base['ident'][i]):02d}", "images", f"{base['sample_id'][i]}.png") for i in idx]
    conds = [("control_plain_rerun", "plain", {"flip": True}, K0),
             ("control_shift_0.5px", "warpK", {"A": [[1, 0, 0.5], [0, 1, 0.0], [0, 0, 1]]}, K0),
             ("control_shift_1px", "warpK", {"A": [[1, 0, 1.0], [0, 1, 1.0], [0, 0, 1]]}, K0)]
    for name, p in [("f+1%", {"f": 1.01}), ("f-1%", {"f": 0.99}), ("cx+5px", {"cx": 5}), ("f+3%", {"f": 1.03}), ("f-3%", {"f": 0.97}), ("f+5%", {"f": 1.05}), ("f-5%", {"f": 0.95}),
                    ("cx+20px", {"cx": 20}), ("cy+20px", {"cy": 20}), ("cx-10px", {"cx": -10})]:
        Kp = perturb_K(Kr, p)
        conds.append((f"Kgeom_{name}", "warpK", {"A": (Kp @ np.linalg.inv(Kr)).tolist()}, perturb_K(K0, p)))
    for s in (0.5, 1.0, 1.5):
        conds.append((f"distort_{s}x", "distort", {"K": Kr.tolist(), "dist": (dist0 * s).tolist()}, K0))
        conds.append((f"distort_{s}x_undistorted", "distort_undistort", {"K": Kr.tolist(), "dist": (dist0 * s).tolist()}, K0))
    conds.append(("no_mirror_flip", "plain", {"flip": False}, K0))
    cache = os.path.join(OUT, f"image_lms_fold{fold}.npz")
    store = dict(np.load(cache)) if os.path.exists(cache) else {}
    with Pool(workers, initializer=_init) as pool:
        for name, kind, prm, _ in conds:
            if f"lms_{name}" in store:
                continue
            t0 = time.time()
            out = pool.map(_work, [(p, kind, prm) for p in paths], chunksize=16)
            ok = np.array([o is not None for o in out])
            lms = np.stack([o[0] if o is not None else np.zeros((478, 3), np.float32) for o in out])
            patch = np.stack([o[1] if o is not None else np.zeros(3072, np.float32) for o in out]).astype(np.float16)
            store[f"lms_{name}"], store[f"ok_{name}"], store[f"patch_{name}"] = lms, ok, patch
            np.savez(cache, idx=idx, **{k: v for k, v in store.items() if k != "idx"})
            print(f"[image] {name}: {ok.mean() * 100:.1f}% detected ({time.time() - t0:.0f}s)", flush=True)
    res = {"fold": fold, "test_identities": sp["test"], "n_images": len(idx), "dist_measured": dist0.tolist(),
           "K_render_raw": Kr.tolist(), "K_measured_raw": K0.tolist(), "models": {}}
    for exp in exps:
        runners, owner = fold_models(exp, base)
        clean = predict_all(runners, owner, idx, base["LMS"][idx], base["F16"][idx], patch=base["PATCH"][idx])
        r = {"clean": summarize(base, idx, clean)}
        for name, kind, prm, Kp in conds:
            ok = store[f"ok_{name}"]
            lms = store[f"lms_{name}"][ok]
            ii = idx[ok]
            variants = {"K_true": Kp} if name.startswith("Kgeom") else {"": K0}
            if name.startswith("Kgeom"):
                variants["K_nominal"] = K0
            for vname, Kf in variants.items():
                f16, okf = I.rich16d_batch(lms, Kf, W, H)
                pch = store[f"patch_{name}"][ok][okf]
                pred = predict_all(runners, owner, ii[okf], lms[okf], f16[okf], lf.mirrored_K(Kf, W), pch)
                key = name + (f"|{vname}" if vname else "")
                r[key] = summarize(base, ii[okf], pred, clean[ok][okf])
                r[key]["detected_frac"] = float(ok.mean())
            if name == "no_mirror_flip":
                # left/right check: predicted u should anti-correlate with true u when the flip is skipped
                u_true = base["UV"][ii[okf], 0]
                r["mirror_check"] = {"corr_u_flipped_input": float(np.corrcoef(pred[:, 0], u_true)[0, 1]),
                                     "corr_u_normal_input": float(np.corrcoef(clean[ok][okf][:, 0], u_true)[0, 1])}
        res["models"][exp] = r
        print(f"[image] {exp}: done", flush=True)
    path = os.path.join(OUT, "image_tests.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.setdefault("models", {}).update(res["models"])
    old.update({k: v for k, v in res.items() if k != "models"})
    D.dump_json(old, path)


def mirror_unit_test(exps):
    """Automatic left/right test on clean data: for test identities, targets on the right (u > 0.4) must be
    predicted to the right of targets on the left (u < -0.4), for every fold model; same for up/down."""
    base = D.load_base()
    idx = np.where(base["OK"])[0]
    out = {}
    for exp in exps:
        runners, owner = fold_models(exp, base)
        pred = predict_all(runners, owner, idx, base["LMS"][idx], base["F16"][idx], patch=base["PATCH"][idx])
        uv = base["UV"][idx]
        per = {}
        for k in sorted(set(base["ident"][idx])):
            m = base["ident"][idx] == k
            R, L = m & (uv[:, 0] > 0.4), m & (uv[:, 0] < -0.4)
            U, Dn = m & (uv[:, 1] > 0.4), m & (uv[:, 1] < -0.4)
            per[int(k)] = {"right_minus_left_cm": float((pred[R, 0].mean() - pred[L, 0].mean()) * 14.53),
                           "up_minus_down_cm": float((pred[U, 1].mean() - pred[Dn, 1].mean()) * 9.45)}
        passed = all(v["right_minus_left_cm"] > 2 and v["up_minus_down_cm"] > 1 for v in per.values())
        out[exp] = {"pass": passed, "per_identity": per}
        print(f"[mirror] {exp}: {'PASS' if passed else 'FAIL'} min R-L {min(v['right_minus_left_cm'] for v in per.values()):.1f} cm")
    path = os.path.join(OUT, "mirror_unit_test.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    D.dump_json(old, path)


if __name__ == "__main__":
    mode, exps = sys.argv[1], sys.argv[2:]
    {"landmark": landmark_tests, "image": image_tests, "mirror": mirror_unit_test}[mode](exps)
