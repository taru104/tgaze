"""Load a trained Phase 7 run (model.pt) and predict from (possibly perturbed) MediaPipe landmarks.

pred = Runner(run_dir, base).predict(lms, f16, K_mir=None)   -> (N,2) display uv
Features are recomputed exactly as in training (same template, same standardisation).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
import p7train as P  # noqa: E402


class Runner:
    def __init__(self, run_dir, base, device="cpu"):
        import torch
        import torch.nn as nn
        self.torch = torch
        ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        self.ck = ck
        self.base = base
        self.cfg = ck["config"]
        self.kind = ck["model"]
        self.template = np.asarray(ck["template"])
        self.stats = {b: (np.asarray(v[0]), np.asarray(v[1])) for b, v in ck["stats"].items()}
        self.result = json.load(open(os.path.join(run_dir, "result.json")))
        dims = {"head_pose": len(self.stats["head_pose"][0]), "eye_norm_r16": len(self.stats["eye_norm_r16"][0])}
        if self.kind == "m8":
            self.net = P.build_m8(nn, dims, ck["hidden"], ck["fuse_hidden"], 0.0)
        else:
            init = ck["init"]
            self.net = P.build_m9(nn, dims["eye_norm_r16"], ck["hidden"], 0.0, init, self.cfg["m9_out"],
                                  self.cfg["m9_learn_roff"], self.cfg.get("m9_head_res", 0))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.device = device

    def inputs(self, lms, f16, K_mir=None, patch=None):
        X, geo, _ = D.fold_features(self.base, None, K_mir=K_mir, lms=lms, f16=f16, template=self.template,
                                    head_angles=self.cfg.get("head_angles", "rich16d"))
        if self.ck.get("pca") is not None:
            pc = self.ck["pca"]
            X["eye_norm_r16"] = np.hstack([X["eye_norm_r16"], ((patch - pc["mean"]) @ pc["components"].T).astype(np.float32)])
        T = lambda a: self.torch.tensor(np.asarray(a), dtype=self.torch.float32)
        xs = {b: T((X[b] - self.stats[b][0]) / self.stats[b][1]) for b in self.stats}
        if self.kind == "m9":
            xs["R_est"] = T(geo["R_est"])
            xs["zfeat"] = T(np.column_stack([geo["log_s"], geo["log_ipd"], geo["log_iris_d"]]))
            xs["eye_mid"] = T(geo["eye_mid"][:, :, :2])
        return xs

    def predict(self, lms, f16, K_mir=None, patch=None):
        xs = self.inputs(lms, f16, K_mir, patch)
        with self.torch.no_grad():
            y, _ = self.net(xs)
        return y.numpy().astype(np.float64)[:, :2]


def rich16d_batch(lms, K_raw, w=1920, h=1080):
    """tgaze rich16d for many landmark sets with a given (raw) K, mirrored=True (as at runtime)."""
    sys.path.insert(0, os.path.expanduser("~/tgaze"))
    import rich16d
    rich16d.set_camera_matrix(np.asarray(K_raw, np.float64), (w, h), mirrored=True)
    out = np.zeros((len(lms), 16), np.float32)
    ok = np.ones(len(lms), bool)
    for i, l in enumerate(lms):
        f = rich16d.rich_16d_from_lms(np.asarray(l, np.float64), w, h)
        if f is None:
            ok[i] = False
        else:
            out[i] = f
    return out, ok
