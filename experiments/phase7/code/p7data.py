"""Phase 7 data layer: labels, per-fold features, evaluation (run with ~/tgaze/.venv/bin/python).

Reads the Phase 5 dataset (outputs/gaze_dataset_v2: metadata.jsonl + landmarks.npz per identity) read-only.
Model inputs are MediaPipe landmarks of the mirrored render only. Blender ground truth is used for labels,
auxiliary labels and evaluation grouping only.

Frames
  cam   : mirrored OpenCV camera frame (x right in the mirrored image, y down, z forward), metres.
          p_cam = MIR @ (R_w2c @ p_world + t_w2c), MIR = diag(-1, 1, 1)   [same as Phase 5]
  head  : Blender 'head' bone frame, mirrored the same way so that it is a proper rotation:
          H_cam = MIR @ R_w2c @ R_bone @ MIR   (det +1), g_local = H_cam^T g_cam
          (Blender eye-in-head convention of the generator: h_world^T @ f, see generate_gaze_dataset_1000.py)
"""
import collections
import glob
import json
import os
import sys

import numpy as np

PHASE7_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(PHASE7_DIR)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import landmark_features as lf  # noqa: E402

OUT_ROOT = os.path.join(PROJECT, "outputs", "experiments", "phase7")
DATA_V2 = os.path.join(PROJECT, "outputs", "gaze_dataset_v2")
CV_SPLITS = os.path.join(PROJECT, "outputs", "experiments", "head_eye_disentanglement", "identity_cv")
MIR = np.diag([-1.0, 1.0, 1.0])
R16_EYE = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
R16_HEAD = [4, 5, 6, 7]
# eye-corner indices (MediaPipe): eye A = 33/133, eye B = 263/362
CORNERS_A, CORNERS_B = (33, 133), (263, 362)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    os.replace(tmp, path)


def load_base(root=DATA_V2, cache=None):
    """All samples of a dataset root -> dict of arrays (labels + raw MediaPipe outputs). Cached as npz."""
    name = os.path.basename(os.path.normpath(root)) + "_base.npz"
    cache = cache or os.path.join(OUT_ROOT, "cache", name)
    # P7_CACHE_DIR (e.g. on the data HDD) holds caches that do not exist yet in the default location
    if not os.path.exists(cache) and os.environ.get("P7_CACHE_DIR"):
        cache = os.path.join(os.environ["P7_CACHE_DIR"], name)
    sig = json.dumps(sorted((os.path.basename(dd), os.path.getsize(os.path.join(dd, "landmarks.npz")))
                            for dd in glob.glob(os.path.join(root, "identity_*")) if os.path.exists(os.path.join(dd, "landmarks.npz"))))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if "signature" not in z.files or str(z["signature"]) == sig:
            d = {k: z[k] for k in z.files if k != "signature"}
            d["geometry"] = json.loads(str(d["geometry"]))
            return d
        print(f"[data] {root}: dataset changed, rebuilding cache", flush=True)
    recs, LMS, F16, PATCH, OK, statics = [], [], [], [], [], {}
    for dd in sorted(glob.glob(os.path.join(root, "identity_*"))):
        if not os.path.exists(os.path.join(dd, "landmarks.npz")):
            continue
        k = int(dd.rsplit("_", 1)[1])
        if len(statics) % 50 == 0:
            print(f"[data] {os.path.basename(os.path.normpath(root))}: loaded {len(statics)} identities", flush=True)
        statics[k] = json.load(open(os.path.join(dd, "scene_static.json")))
        z = np.load(os.path.join(dd, "landmarks.npz"))
        rr = [json.loads(line) for line in open(os.path.join(dd, "metadata.jsonl"))]
        assert [r["sample_id"] for r in rr] == list(z["sample_id"]), dd
        recs += rr
        LMS.append(z["lms"]), F16.append(z["f16"]), PATCH.append(z["patch"]), OK.append(z["ok"])
    N = len(recs)
    ident = np.array([r["identity_index"] for r in recs])
    ids = sorted(statics)
    st0 = statics[ids[0]]
    W, H = st0["camera"]["resolution"]

    def to_cam(k, p):
        e = statics[k]["camera"]["extrinsics_opencv"]
        return MIR @ (np.array(e["R_world_to_cam"]) @ np.asarray(p) + np.array(e["t_world_to_cam"]))

    def dir_to_cam(k, d):
        return MIR @ (np.array(statics[k]["camera"]["extrinsics_opencv"]["R_world_to_cam"]) @ np.asarray(d))

    planes = {}
    for k in ids:
        st = statics[k]
        planes[k] = np.concatenate([to_cam(k, st["display"]["center_world"]), dir_to_cam(k, st["display"]["normal_world"]),
                                    dir_to_cam(k, st["display"]["right_world"]), dir_to_cam(k, st["display"]["up_world"])])
    dev = max(float(np.abs(planes[k] - planes[ids[0]]).max()) for k in ids)
    assert dev < 1e-4, f"display-camera geometry differs between identities ({dev})"
    P = planes[ids[0]]
    Kr = np.array(st0["camera"]["measured_K_px_scaled"], np.float64)
    K_render = np.array(st0["camera"]["intrinsics_K_px"], np.float64)
    geometry = {"P_C": P[0:3].tolist(), "P_N": P[3:6].tolist(), "P_R": P[6:9].tolist(), "P_U": P[9:12].tolist(),
                "disp_w": st0["display"]["width_m"], "disp_h": st0["display"]["height_m"], "W": W, "H": H,
                "K_measured_raw": Kr.tolist(), "K_render_raw": K_render.tolist(),
                "K_mirrored": lf.mirrored_K(Kr, W).tolist(), "identities": ids}

    O_c, O_L, O_R, A_c, G_L, G_R = (np.zeros((N, 3)) for _ in range(6))
    Hc = np.zeros((N, 3, 3))
    pupil_L, pupil_R = np.zeros((N, 3)), np.zeros((N, 3))
    for i, r in enumerate(recs):
        k = r["identity_index"]
        O_c[i] = to_cam(k, r["viewer"]["cyclopean_eye_center_world"])
        O_L[i] = to_cam(k, r["left_eye"]["center_world"])
        O_R[i] = to_cam(k, r["right_eye"]["center_world"])
        pupil_L[i] = to_cam(k, r["left_eye"]["pupil_center_world"])
        pupil_R[i] = to_cam(k, r["right_eye"]["pupil_center_world"])
        A_c[i] = to_cam(k, r["target"]["world_xyz"])
        G_L[i] = dir_to_cam(k, r["left_eye"]["gaze_vector_world"])
        G_R[i] = dir_to_cam(k, r["right_eye"]["gaze_vector_world"])
        Rb = np.array(r["head"]["transform_world"])[:3, :3]
        Rb = Rb / np.linalg.norm(Rb, axis=0, keepdims=True)
        R_w2c = np.array(statics[k]["camera"]["extrinsics_opencv"]["R_world_to_cam"])
        Hc[i] = MIR @ R_w2c @ Rb @ MIR
    G_c = (A_c - O_c) / np.linalg.norm(A_c - O_c, axis=1, keepdims=True)
    d = dict(
        ident=ident, sample_id=np.array([r["sample_id"] for r in recs]),
        LMS=np.concatenate(LMS), F16=np.concatenate(F16), PATCH=np.concatenate(PATCH).astype(np.float32), OK=np.concatenate(OK),
        UV=np.array([r["target"]["normalized_uv"] for r in recs]),
        O_c=O_c, O_L=O_L, O_R=O_R, A_c=A_c, G_c=G_c, G_L=G_L, G_R=G_R, H_cam=Hc, pupil_L=pupil_L, pupil_R=pupil_R,
        HEAD_AUX=np.array([[-r["head"]["world"]["yaw_deg"], r["head"]["world"]["pitch_deg"], -r["head"]["world"]["roll_deg"]] for r in recs]),
        EYE_AUX=np.array([np.mean([r["left_eye"]["eye_in_head_yaw_pitch_deg"], r["right_eye"]["eye_in_head_yaw_pitch_deg"]], 0) * [-1, 1] for r in recs]),
        EYE_YP_L=np.array([r["left_eye"]["eye_in_head_yaw_pitch_deg"] for r in recs]),
        EYE_YP_R=np.array([r["right_eye"]["eye_in_head_yaw_pitch_deg"] for r in recs]),
        mode=np.array([r["pose_mode"] for r in recs]), gtype=np.array([r["group_type"] for r in recs]),
        gid=np.array([r["group_id"] for r in recs]), stratum=np.array([r["target"]["uv_stratum"] for r in recs]),
        head_mag=np.hypot([r["head"]["world"]["yaw_deg"] for r in recs], [r["head"]["world"]["pitch_deg"] - (-4.5) for r in recs]),
        vergence=np.array([r["vergence_deg"] for r in recs]),
        geometry=json.dumps(geometry))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, signature=sig, **d)
    d["geometry"] = geometry
    return d


class Geometry:
    def __init__(self, g):
        self.P_C, self.P_N, self.P_R, self.P_U = (np.array(g[k]) for k in ("P_C", "P_N", "P_R", "P_U"))
        self.W, self.H = g["W"], g["H"]
        self.disp_w, self.disp_h = g["disp_w"], g["disp_h"]
        self.K_mir = np.array(g["K_mirrored"])
        self.uv_cm = np.array([self.disp_w / 2 * 100, self.disp_h / 2 * 100])

    def uv_to_cam(self, uv):
        return self.P_C + np.outer(uv[:, 0] * self.disp_w / 2, self.P_R) + np.outer(uv[:, 1] * self.disp_h / 2, self.P_U)

    def cam_to_uv(self, p):
        d = p - self.P_C
        return np.column_stack([d @ self.P_R / (self.disp_w / 2), d @ self.P_U / (self.disp_h / 2)])

    def intersect(self, o, g):
        t = ((self.P_C - o) @ self.P_N) / (g @ self.P_N)
        return o + g * t[:, None]


def cv_split(fold):
    return json.load(open(os.path.join(CV_SPLITS, f"split_fold{fold}.json")))


def split_indices(base, sp):
    ok = base["OK"]
    ident = base["ident"]
    tr, va, te = (np.where(np.isin(ident, sp[s]) & ok)[0] for s in ("train", "validation", "test"))
    gid = base["gid"]
    assert not (set(gid[tr]) & set(gid[te])) and not (set(gid[tr]) & set(gid[va])) and not (set(gid[va]) & set(gid[te])), "group leakage"
    return tr, va, te


def rot_from_6d(r6):
    """(N,6) first two columns -> (N,3,3) (Gram-Schmidt)."""
    a, b = r6[:, :3], r6[:, 3:]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], -1)


def kabsch_euler(R):
    """(N,3,3) camera<-head Kabsch rotation -> (N,3) [pitch, yaw, roll] radians, R = Rx(pitch) Ry(yaw) Rz(roll).
    NumPy only (runtime safe). Same sign convention as cv2.RQDecomp3x3 / tgaze rich16d for near-frontal heads."""
    yaw = np.arcsin(np.clip(R[:, 0, 2], -1, 1))
    pitch = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
    roll = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
    return np.stack([pitch, yaw, roll], 1)


HEAD_ANGLES = ("rich16d", "none", "kabsch")


def head_pose_input(FE, head_angles="rich16d"):
    """M8 head_pose branch: R_head 2 columns (6) + rigid centroid xy (2) + log s (1) [+ 3 angles].
    rich16d = tgaze solvePnP pitch/yaw/roll (Phase 5 frozen M8, 12D); none = 9D; kabsch = Euler of R_head (12D)."""
    base9 = FE["head"][:, :9]
    if head_angles == "rich16d":
        return np.hstack([base9, FE["head"][:, -3:]]).astype(np.float32)
    if head_angles == "none":
        return base9.astype(np.float32)
    if head_angles == "kabsch":
        return np.hstack([base9, kabsch_euler(FE["R_head"].astype(np.float64))]).astype(np.float32)
    raise ValueError(head_angles)


def fold_features(base, tr, K_mir=None, lms=None, f16=None, template=None, head_angles="rich16d"):
    """Phase 5 features (exact recipe) + M9 geometric inputs. Template from training samples only."""
    g = Geometry(base["geometry"])
    K_mir = g.K_mir if K_mir is None else K_mir
    LMS = base["LMS"] if lms is None else lms
    F16 = base["F16"] if f16 is None else f16
    NC = lf.to_ncam(LMS, K_mir, g.W, g.H)
    if template is None:
        template = lf.build_template(NC[tr[:: max(1, len(tr) // 4000)]])
    FE = lf.features(NC, template, F16[:, [4, 5, 7]])
    X = {"eye_norm_r16": np.hstack([FE["eye_norm"], F16[:, R16_EYE]]).astype(np.float32),
         "head_pose": head_pose_input(FE, head_angles),
         "head": FE["head"], "eye_norm": FE["eye_norm"], "eye_raw": FE["eye_raw"]}
    R = FE["R_head"].astype(np.float64)                       # camera <- template frame (Kabsch)
    rig = NC[:, lf.RIGID]
    c = rig.mean(1)
    s = np.sqrt(((rig - c[:, None]) ** 2).sum(-1).mean(-1))
    eye_mid = np.stack([NC[:, list(CORNERS_A)].mean(1), NC[:, list(CORNERS_B)].mean(1)], 1)   # (N,2,3) ncam
    iris = np.stack([NC[:, 468:473].mean(1), NC[:, 473:478].mean(1)], 1)
    ipd_n = np.linalg.norm(iris[:, 0, :2] - iris[:, 1, :2], axis=1)
    iris_d = np.stack([np.linalg.norm(NC[:, 469, :2] - NC[:, 471, :2], axis=1), np.linalg.norm(NC[:, 474, :2] - NC[:, 476, :2], axis=1)], 1)
    geo = {"R_est": R, "log_s": np.log(s), "centroid": c, "eye_mid": eye_mid, "iris": iris, "log_ipd": np.log(ipd_n),
           "log_iris_d": np.log(iris_d.mean(1))}
    return X, geo, template


# ------------------------------------------------------------------------------------------- evaluation
HEAD_BINS = ("small(<3deg)", "medium(3-7deg)", "large(>7deg)")
REGIONS = ("center", "middle", "edge")


def stats(cm, ang=None):
    if len(cm) == 0:
        return None
    o = {"n": int(len(cm)), "cm_mean": float(cm.mean()), "cm_median": float(np.median(cm)), "cm_rmse": float(np.sqrt((cm ** 2).mean())),
         "cm_p50": float(np.percentile(cm, 50)), "cm_p90": float(np.percentile(cm, 90)), "cm_p95": float(np.percentile(cm, 95))}
    if ang is not None:
        o["deg_mean"] = float(ang.mean())
        o["deg_median"] = float(np.median(ang))
    return o


def errors(base, idx, pred_uv):
    g = Geometry(base["geometry"])
    d = (pred_uv - base["UV"][idx]) * g.uv_cm
    cm = np.hypot(d[:, 0], d[:, 1])
    pg = g.uv_to_cam(pred_uv) - base["O_c"][idx]
    pg /= np.linalg.norm(pg, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((pg * base["G_c"][idx]).sum(1), -1, 1)))
    return cm, ang


def pose_invariance(base, idx, pred_uv):
    """Same-target groups: max pairwise distance of the predictions (cm), averaged over groups (Phase 5 metric)."""
    g = Geometry(base["geometry"])
    P = pred_uv * g.uv_cm
    groups = collections.defaultdict(list)
    for j, i in enumerate(idx):
        if base["gtype"][i] == "same_target":
            groups[base["gid"][i]].append(j)
    inv, mean_dev = [], []
    for js in groups.values():
        if len(js) < 2:
            continue
        x = P[js]
        dd = np.linalg.norm(x[:, None] - x[None], axis=-1)
        inv.append(dd.max())
        mean_dev.append(np.linalg.norm(x - x.mean(0), axis=1).mean())
    if not inv:
        return None
    inv = np.array(inv)
    # output compression: predicted / true spread (std over all evaluated samples, mean of the 2 axes); a model
    # that shrinks toward the centre looks more 'invariant', so the spread-normalised value is reported as well
    spread = float((np.std(P, 0) / np.std(base["UV"][idx] * g.uv_cm, 0)).mean())
    return {"mean": float(inv.mean()), "pred_spread_ratio": spread, "mean_spread_normalised": float(inv.mean() / spread), "median": float(np.median(inv)), "p90": float(np.percentile(inv, 90)), "groups": len(inv),
            "mean_dev_from_group_mean_cm": float(np.mean(mean_dev))}


def evaluate(base, idx, pred_uv, full=True):
    cm, ang = errors(base, idx, pred_uv)
    out = {"all": stats(cm, ang)}
    if not full:
        return out
    hb = np.digitize(base["head_mag"][idx], [3.0, 7.0])
    out["by_pose_mode"] = {m: stats(cm[base["mode"][idx] == m], ang[base["mode"][idx] == m]) for m in sorted(set(base["mode"][idx]))}
    out["by_head_angle"] = {n: stats(cm[hb == b], ang[hb == b]) for b, n in enumerate(HEAD_BINS)}
    out["by_screen_region"] = {n: stats(cm[base["stratum"][idx] == b], ang[base["stratum"][idx] == b]) for b, n in enumerate(REGIONS)}
    out["by_group_type"] = {t: stats(cm[base["gtype"][idx] == t]) for t in sorted(set(base["gtype"][idx]))}
    out["pose_invariance_cm"] = pose_invariance(base, idx, pred_uv)
    return out


def run_roots(run_dir):
    """Dataset roots a p7train.py run was trained on: v2 plus its --root (same order as p7train.main)."""
    root = json.load(open(os.path.join(run_dir, "result.json")))["config"].get("root")
    return [DATA_V2] + ([r if os.path.isabs(r) else os.path.join(PROJECT, r) for r in root.split(",")] if root else [])


def load_run_base(run_dir):
    """The base whose indices the run's pred.npz refers to (v2 alone, or v2 merged with --root datasets as in p7train)."""
    roots = run_roots(run_dir)
    if len(roots) == 1:
        return load_base()
    bases = [load_base(r) for r in roots]
    out = {}
    for k in bases[0]:
        if k == "geometry":
            g = dict(bases[0]["geometry"])
            for b in bases[1:]:
                for key in ("P_C", "P_N", "P_R", "P_U"):
                    assert np.abs(np.array(b["geometry"][key]) - np.array(g[key])).max() < 1e-4, "display geometry differs"
                g["identities"] = sorted(set(g["identities"]) | set(b["geometry"]["identities"]))
            out[k] = g
        else:
            out[k] = np.concatenate([b[k] for b in bases])
    return out


def image_path(roots, ident, sample_id):
    """Rendered image of a sample; datasets after Phase 7E keep JPEG (q92) instead of the lossless PNG."""
    for root in roots:
        for ext in (".png", ".jpg"):
            p = os.path.join(root, f"identity_{ident:02d}", "images", f"{sample_id}{ext}")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"identity {ident} sample {sample_id}")
