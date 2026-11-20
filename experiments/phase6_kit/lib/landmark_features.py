"""MediaPipe-landmark features for the Phase 5 head/eye models. NumPy only (CPU runtime safe).

Input: the 478 MediaPipe FaceLandmarker points (normalized x, y in [0,1], z in "x-like" units) of a
MIRRORED frame (tgaze flips the camera frame horizontally before estimation; the synthetic images are
flipped the same way), plus the camera matrix of the mirrored frame (cx -> W - cx).

Coordinate systems
  image     : pixels, origin top-left, of the mirrored frame
  ncam      : normalized camera coords  xn = (px - cx)/fx, yn = (py - cy)/fy, zn = z * W / fx
              (x right, y down, z away from the camera; MediaPipe z is relative depth, same scale as x)
  head      : ncam points minus the rigid-subset centroid, divided by its RMS radius s, rotated by
              R_head^T, where R_head is the Kabsch rotation of the rigid subset onto a mean template
              (template built from training identities only). In the head frame the face is frontal.
Feature groups
  head      : R_head (6D, first two columns), rigid centroid (xn, yn), log(s), rigid subset in the head
              frame (residual shape), rich16d pitch/yaw/roll (solvePnP, tgaze) if supplied
  eye_raw   : eye subset relative to the rigid centroid, / s  (camera orientation, NOT de-rotated)
  eye_norm  : eye subset in the head frame (de-rotated)  -> eye-in-head configuration
  flat      : all 478 ncam points flattened
"""
import numpy as np

# rigid, expression-insensitive points: forehead, nose bridge/tip/base, cheekbones, temples, face sides
RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 234, 454, 93, 323, 127, 356,
         116, 345, 123, 352, 50, 280, 101, 330, 21, 251, 54, 284, 103, 332, 67, 297, 109, 338]
IRIS = list(range(468, 478))
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROWS = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
EYE = IRIS + EYE_A + EYE_B + BROWS


def mirrored_K(K, width):
    K = np.array(K, dtype=np.float64)
    K[0, 2] = width - K[0, 2]
    return K


def to_ncam(lms, K, w, h):
    """lms (N,478,3) normalized MediaPipe -> (N,478,3) normalized camera coordinates."""
    lms = np.asarray(lms, np.float64)
    out = np.empty_like(lms)
    out[..., 0] = (lms[..., 0] * w - K[0, 2]) / K[0, 0]
    out[..., 1] = (lms[..., 1] * h - K[1, 2]) / K[1, 1]
    out[..., 2] = lms[..., 2] * w / K[0, 0]
    return out


def _center_scale(p):
    c = p.mean(axis=-2, keepdims=True)
    q = p - c
    s = np.sqrt((q ** 2).sum(-1).mean(-1, keepdims=True))[..., None]
    return q / s, c[..., 0, :], s[..., 0, 0]


def kabsch(src, dst):
    """Batch rotations R with R @ src_i ~ dst_i (src, dst: (N,P,3), already centered)."""
    H = np.einsum("npi,npj->nij", src, dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    D = np.zeros_like(H)
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 2, 2] = d
    return np.einsum("nji,njk,nlk->nil", Vt, D, U)   # V D U^T


def build_template(ncam, iters=5):
    """Generalized Procrustes mean of the rigid subset (training data only)."""
    q, _, _ = _center_scale(ncam[:, RIGID])
    tmpl = q[0]
    for _ in range(iters):
        R = kabsch(np.broadcast_to(tmpl, q.shape), q)            # R @ tmpl ~ q
        aligned = np.einsum("nji,npj->npi", R, q)                 # R^T q
        tmpl = aligned.mean(0)
        tmpl /= np.sqrt((tmpl ** 2).sum(-1).mean())
    return tmpl


def features(ncam, template, pyr16=None):
    """Return dict of feature groups (float32) for a batch (N,478,3) in ncam."""
    rig, c, s = _center_scale(ncam[:, RIGID])
    R = kabsch(np.broadcast_to(template, rig.shape), rig)          # head rotation (camera <- head)
    rig_head = np.einsum("nji,npj->npi", R, rig)                     # R^T rig
    eye_rel = (ncam[:, EYE] - c[:, None, :]) / s[:, None, None]
    eye_norm = np.einsum("nji,npj->npi", R, eye_rel)
    head = [R[:, :, 0], R[:, :, 1], c[:, :2], np.log(s)[:, None], (rig_head - template).reshape(len(R), -1)]
    if pyr16 is not None:
        head.append(np.asarray(pyr16, np.float64))
    return {
        "head": np.hstack(head).astype(np.float32),
        "eye_raw": eye_rel.reshape(len(R), -1).astype(np.float32),
        "eye_norm": eye_norm.reshape(len(R), -1).astype(np.float32),
        "flat": ncam.reshape(len(R), -1).astype(np.float32),
        "R_head": R.astype(np.float32),
    }
