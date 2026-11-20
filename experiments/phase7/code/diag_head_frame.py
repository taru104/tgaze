"""Phase 7B diagnostics: Blender head/eye conventions and how well analytic head pose from MediaPipe matches.
Writes outputs/experiments/phase7/diagnostics/head_frame.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D

b = D.load_base()
N = len(b["UV"])
H = b["H_cam"]
out = {}
out["H_det_minmax"] = [float(np.linalg.det(H).min()), float(np.linalg.det(H).max())]
out["H_orth_err"] = float(np.abs(np.einsum("nji,njk->nik", H, H) - np.eye(3)).max())
# eye-in-head from my frame vs metadata yaw/pitch (relative to rest) -> check a constant rest offset only
gl = np.einsum("nji,nj->ni", H, b["G_L"])
yaw = np.degrees(np.arctan2(gl[:, 0], -gl[:, 2])); pit = np.degrees(np.arcsin(np.clip(-gl[:, 1], -1, 1)))
# the metadata yaw sign convention is the person's view; compare correlation and offset
out["eyeL_local_vs_meta"] = {"corr_yaw": float(np.corrcoef(yaw, b["EYE_YP_L"][:, 0])[0, 1]), "corr_pitch": float(np.corrcoef(pit, b["EYE_YP_L"][:, 1])[0, 1]),
    "yaw_diff_std": float(np.std(yaw + b["EYE_YP_L"][:, 0])), "yaw_diff_std_samesign": float(np.std(yaw - b["EYE_YP_L"][:, 0])),
    "pitch_diff_std": float(np.std(pit - b["EYE_YP_L"][:, 1]))}
out["local_gaze_mean"] = gl.mean(0).tolist()
# head forward direction in camera
fwd = H[:, :, 2]
out["H_col_means"] = {"x": H[:, :, 0].mean(0).tolist(), "y": H[:, :, 1].mean(0).tolist(), "z": H[:, :, 2].mean(0).tolist()}
# analytic head pose: Kabsch R_est (fold 0 template) vs H with a constant offset fitted on train
sp = D.cv_split(0); tr, va, te = D.split_indices(b, sp)
X, geo, tmpl = D.fold_features(b, tr)
R = geo["R_est"]
def fit_offset(Ra, Hb, idx):
    M = np.einsum("nji,njk->ik", Ra[idx], Hb[idx])      # sum R^T H
    U, _, Vt = np.linalg.svd(M); d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt
def ang_err(A, B):
    Mx = np.einsum("nji,njk->nik", A, B)
    return np.degrees(np.arccos(np.clip((np.trace(Mx, axis1=1, axis2=2) - 1) / 2, -1, 1)))
Roff = fit_offset(R, H, tr)
e = ang_err(R @ Roff, H)
out["kabsch_vs_gt"] = {"train_mean_deg": float(e[tr].mean()), "test_mean_deg": float(e[te].mean()), "test_p95": float(np.percentile(e[te], 95))}
# per-identity constant offsets (shows identity-specific face-shape bias)
per = {}
for k in sp["test"]:
    m = te[b["ident"][te] == k]
    per[int(k)] = float(ang_err(R[m] @ fit_offset(R, H, m), H[m]).mean())
out["kabsch_vs_gt_per_test_identity_own_offset_deg"] = per
# perspective-corrected: rotate so the rigid centroid ray maps to the optical axis
c = geo["centroid"].copy(); c[:, 2] = 1.0
def ray_rot(c):
    z = c / np.linalg.norm(c, axis=1, keepdims=True)
    a = np.cross(np.array([0, 0, 1.0])[None], z); s = np.linalg.norm(a, axis=1); cth = z[:, 2]
    k = a / np.maximum(s, 1e-12)[:, None]
    Kx = np.zeros((len(z), 3, 3)); Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]; Kx[:, 1, 0] = k[:, 2]; Kx[:, 1, 2] = -k[:, 0]; Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
    return np.eye(3)[None] + s[:, None, None] * Kx + (1 - cth)[:, None, None] * Kx @ Kx
Rr = ray_rot(c)
for name, Ra in (("ray_then_kabsch", Rr @ R),):
    Ro = fit_offset(Ra, H, tr); e2 = ang_err(Ra @ Ro, H)
    out[name + "_vs_gt"] = {"train_mean_deg": float(e2[tr].mean()), "test_mean_deg": float(e2[te].mean()), "test_p95": float(np.percentile(e2[te], 95))}
# yaw/pitch component errors of the Kabsch version
Hf = H[:, :, 2]; Ef = (R @ Roff)[:, :, 2]
out["kabsch_forward_axis_err_deg_test"] = float(np.degrees(np.arccos(np.clip((Hf[te] * Ef[te]).sum(1), -1, 1))).mean())
# correlation of rotation error with eye-in-head gaze (coupling of 'rigid' points with eye motion)
dR = np.einsum("nji,njk->nik", H, R @ Roff)
ry = np.degrees(dR[:, 0, 2]); rp = np.degrees(dR[:, 1, 2])
out["rot_err_vs_eye_gaze_corr_test"] = {"yaw_err~eye_yaw": float(np.corrcoef(ry[te], yaw[te])[0, 1]), "pitch_err~eye_pitch": float(np.corrcoef(rp[te], pit[te])[0, 1]),
    "yaw_err_std_deg": float(ry[te].std()), "pitch_err_std_deg": float(rp[te].std())}
# rich16d solvePnP Euler (tgaze) as rotation
def euler_to_R(p, y, r):
    cx, sx, cy, sy, cz, sz = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    Rx = np.stack([np.stack([np.ones_like(p), 0 * p, 0 * p], -1), np.stack([0 * p, cx, -sx], -1), np.stack([0 * p, sx, cx], -1)], -2)
    Ry = np.stack([np.stack([cy, 0 * p, sy], -1), np.stack([0 * p, np.ones_like(p), 0 * p], -1), np.stack([-sy, 0 * p, cy], -1)], -2)
    Rz = np.stack([np.stack([cz, -sz, 0 * p], -1), np.stack([sz, cz, 0 * p], -1), np.stack([0 * p, 0 * p, np.ones_like(p)], -1)], -2)
    return Rx @ Ry @ Rz
f16 = b["F16"].astype(np.float64)
best = None
for order in ("xyz",):
    Rp = euler_to_R(f16[:, 4], f16[:, 5], f16[:, 7])
    Ro = fit_offset(Rp, H, tr); e3 = ang_err(Rp @ Ro, H)
    out["rich16d_solvepnp_euler_vs_gt"] = {"train_mean_deg": float(e3[tr].mean()), "test_mean_deg": float(e3[te].mean())}
# eye centre from MediaPipe: depth model z = exp(a.[log s, log ipd, log iris] + b) fitted on train
Z = np.column_stack([geo["log_s"], geo["log_ipd"], geo["log_iris_d"], np.ones(N)])
for name, cols in (("s", [0, 3]), ("ipd", [1, 3]), ("iris", [2, 3]), ("all", [0, 1, 2, 3])):
    zt = np.log(b["O_c"][:, 2])
    w, *_ = np.linalg.lstsq(Z[tr][:, cols], zt[tr], rcond=None)
    zp = np.exp(Z[:, cols] @ w)
    out.setdefault("depth_model", {})[name] = {"test_mean_abs_cm": float(np.abs(zp[te] - b["O_c"][te, 2]).mean() * 100), "train_mean_abs_cm": float(np.abs(zp[tr] - b["O_c"][tr, 2]).mean() * 100)}
out["O_c_z_range_m"] = [float(b["O_c"][:, 2].min()), float(b["O_c"][:, 2].max())]
# iris landmark mirror correspondence in the head frame
NCh = None
D.dump_json(out, os.path.join(D.OUT_ROOT, "diagnostics", "head_frame.json"))
print(json.dumps(out, indent=1))
