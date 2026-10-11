"""Phase 7 trainer: M8 (Phase 5 recipe), M8 + same-target consistency, M9 physical head-eye model.

  ~/tgaze/.venv/bin/python phase7/p7train.py --exp consistency/lam0.25 --model m8 --lam 0.25 --folds 0 --seeds 0

One run = (experiment, fold, seed). Every run writes outputs/experiments/phase7/<exp>/fold<f>_seed<s>/
  result.json   metrics on train / validation / test identities (+ breakdowns), config, timing, device
  curves.json   per-epoch train loss, validation loss, train cm, validation cm, validation pose invariance
  pred.npz      idx, pred_uv for train/validation/test samples (calibration and analysis reuse these)
  model.pt      state dict + everything needed for ONNX export
A run whose result.json exists is skipped (resume). Training device is recorded; inference at runtime is CPU only.

Consistency loss (Phase 7A): for each same-target group in the mini-batch (4 poses, one screen target),
  L_cons = mean_i sqrt(||p_i - mean_group(p)||^2 + eps)   [display cm]
Batches are built from whole groups so every same-target group is complete in its batch.
L_total = L_target (Huber, delta 1 cm, display cm) + lambda * L_cons (+ aux terms as in M8 / M9).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", default="m8", choices=["m8", "m9"])
    ap.add_argument("--lam", type=float, default=0.0, help="consistency weight")
    ap.add_argument("--cons", default="mean_dev", choices=["mean_dev", "pairwise", "scaled"])
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--hidden", default="64,32", help="branch MLP hidden sizes (M8) / eye MLP (M9)")
    ap.add_argument("--fuse-hidden", default="64,32")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-ids", default=None, help="override training identities (comma list)")
    ap.add_argument("--max-train", type=int, default=None, help="subsample training groups to about this many samples")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--per-id-max", type=int, default=None, help="subsample whole groups per training identity to about N samples")
    ap.add_argument("--root", default=None, help="extra dataset root(s) (comma) merged with v2")
    ap.add_argument("--split-json", default=None, help="custom split {train,validation,test}")
    # M9
    ap.add_argument("--m9-out", default="vec3", choices=["vec3", "yawpitch"])
    ap.add_argument("--m9-loss", default="screen", choices=["screen", "angular", "local"])
    ap.add_argument("--m9-local-w", type=float, default=0.3, help="weight of the eye-in-head (Blender head frame) angular aux loss")
    ap.add_argument("--m9-origin-w", type=float, default=1.0)
    ap.add_argument("--m9-phys-lr", type=float, default=1e-4)
    ap.add_argument("--m9-learn-roff", type=int, default=0)
    ap.add_argument("--m9-head-res", type=int, default=0, help="1: small learned head-rotation residual from the eye-frame input (ablation)")
    ap.add_argument("--m9-oracle", default="", help="DIAGNOSTIC ONLY (not deployable): 'head' = Blender head rotation instead of "
                    "the MediaPipe Kabsch rotation, 'origin' = Blender eye centres, 'head,origin' both")
    ap.add_argument("--eye-pca", type=int, default=0, help="diagnostic: append tgaze eye-patch PCA (fit on train) to the eye branch")
    ap.add_argument("--noise-aug", type=float, default=0.0, help="landmark noise augmentation sigma [px] (M8 only)")
    ap.add_argument("--noise-reps", type=int, default=2, help="noisy replicas of the training set")
    ap.add_argument("--head-angles", default="rich16d", choices=["rich16d", "none", "kabsch"],
                    help="last 3 dims of the M8 head_pose input: rich16d solvePnP (Phase 5), none (9D), kabsch Euler")
    ap.add_argument("--save-preds", type=int, default=1)
    ap.add_argument("--tag", default="")
    return ap.parse_args(argv)


# ===================================================================================== data per fold
def prepare(args, fold, base):
    if args.split_json:
        sp = json.load(open(args.split_json))
    else:
        sp = D.cv_split(fold)
    if args.train_ids:
        sp = dict(sp, train=[int(x) for x in args.train_ids.split(",")])
    tr, va, te = D.split_indices(base, sp)
    if args.per_id_max:
        rng = np.random.default_rng([args.subset_seed, args.per_id_max, 77])
        gid, keep = base["gid"], []
        for k in sorted(set(base["ident"][tr])):
            tk = tr[base["ident"][tr] == k]
            cnt = 0
            for gname in rng.permutation(sorted(set(gid[tk]))):
                if cnt >= args.per_id_max:
                    break
                m = tk[gid[tk] == gname]
                keep.append(m)
                cnt += len(m)
        tr = np.sort(np.concatenate(keep))
    if args.max_train and args.max_train < len(tr):
        rng = np.random.default_rng([args.subset_seed, args.max_train, fold])
        gid = base["gid"]
        groups = sorted(set(gid[tr]))
        chosen, cnt = set(), 0
        for g in rng.permutation(groups):
            if cnt >= args.max_train:
                break
            chosen.add(g)
            cnt += int((gid[tr] == g).sum())
        tr = tr[np.isin(gid[tr], list(chosen))]
    X, geo, template = D.fold_features(base, tr, head_angles=args.head_angles)
    if args.eye_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.eye_pca, random_state=0).fit(base["PATCH"][tr])
        X["eye_norm_r16"] = np.hstack([X["eye_norm_r16"], pca.transform(base["PATCH"]).astype(np.float32)])
        geo["pca"] = {"mean": pca.mean_, "components": pca.components_}
    return sp, tr, va, te, X, geo, template


# ===================================================================================== models
def make_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def mlp(nn, i, sizes, p):
    layers = []
    for o in sizes:
        layers += [nn.Linear(i, o), nn.ReLU(), nn.Dropout(p)]
        i = o
    return nn.Sequential(*layers), i


def build_m8(nn, dims, hidden, fuse_hidden, p, latent=16):
    import torch

    class M8(nn.Module):
        """Phase 5 M8: head_pose branch + eye_norm_r16 branch -> fusion MLP -> screen (u, v) [cm-scaled]."""

        def __init__(self):
            super().__init__()
            self.branches = ["head_pose", "eye_norm_r16"]
            self.enc = nn.ModuleDict()
            for b in self.branches:
                body, o = mlp(nn, dims[b], hidden, p)
                self.enc[b] = nn.Sequential(body, nn.Linear(o, latent), nn.ReLU())
            self.fuse, o = mlp(nn, latent * 2, fuse_hidden, p)
            self.out = nn.Linear(o, 2)
            self.aux_head = nn.Linear(latent, 3)
            self.aux_eye = nn.Linear(latent, 2)

        def forward(self, xs):
            lat = {b: self.enc[b](xs[b]) for b in self.branches}
            y = self.out(self.fuse(torch.cat([lat[b] for b in self.branches], 1)))
            return y, {"head": self.aux_head(lat["head_pose"]), "eye": self.aux_eye(lat["eye_norm_r16"])}
    return M8()


def rodrigues(torch, w):
    th = torch.linalg.norm(w) + 1e-12
    k = w / th
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=w.dtype, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def build_m9(nn, dim_eye, hidden, p, init, out_kind, learn_roff, head_res):
    """M9 physical head-eye model.

    eye MLP(eye landmarks in the analytic head frame + rich16d eye features) -> g_local for each eye (unit 3D,
    Blender head-bone frame) ; R_head = R_kabsch @ R_off (analytic, NOT learned except optional constant R_off)
    g_cam_e = R_head @ g_local_e                                  (matrix product, outside the MLP)
    O_e     = z * (x_e, y_e, 1) + R_head @ d_e                    (eye centre: ncam eye-corner midpoint, depth from
              z = exp(w . [log s, log ipd, log iris] + b)          face / iris scale; w, b, d_e = few physical constants)
    P_e     = ray-plane intersection(O_e, g_cam_e, display plane) (geometry)
    screen  = mean(P_L, P_R) -> (u, v)
    """
    import torch

    class M9(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, o = mlp(nn, dim_eye, hidden, p)
            self.out_kind = out_kind
            self.head_res = head_res
            n_out = 6 if out_kind == "vec3" else 4
            self.out = nn.Linear(o, n_out)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.register_buffer("g0", torch.tensor(init["g0"], dtype=torch.float32))          # (2,3) mean local gaze
            self.register_buffer("yp0", torch.tensor(init["yp0"], dtype=torch.float32))        # (2,2)
            self.register_buffer("P_C", torch.tensor(init["P_C"], dtype=torch.float32))
            self.register_buffer("P_N", torch.tensor(init["P_N"], dtype=torch.float32))
            self.register_buffer("P_R", torch.tensor(init["P_R"], dtype=torch.float32))
            self.register_buffer("P_U", torch.tensor(init["P_U"], dtype=torch.float32))
            self.register_buffer("half_wh", torch.tensor(init["half_wh"], dtype=torch.float32))
            self.register_buffer("R_off0", torch.tensor(init["R_off"], dtype=torch.float32))
            self.register_buffer("zf_mu", torch.tensor(init["zf_mu"], dtype=torch.float32))
            self.register_buffer("zf_sd", torch.tensor(init["zf_sd"], dtype=torch.float32))
            self.w_roff = nn.Parameter(torch.zeros(3), requires_grad=bool(learn_roff))
            self.z_w = nn.Parameter(torch.tensor(init["z_w"], dtype=torch.float32))
            self.z_b = nn.Parameter(torch.tensor(init["z_b"], dtype=torch.float32))
            self.d_eye = nn.Parameter(torch.tensor(init["d_eye"], dtype=torch.float32))       # (2,3) head frame, metres
            if head_res:
                self.res = nn.Linear(o, 3)
                nn.init.zeros_(self.res.weight)
                nn.init.zeros_(self.res.bias)

        def local_gaze(self, h):
            y = self.out(h)
            if self.out_kind == "vec3":
                g = y.view(-1, 2, 3) + self.g0[None]
                return g / torch.linalg.norm(g, dim=-1, keepdim=True)
            yp = y.view(-1, 2, 2) + self.yp0[None]
            yaw, pit = yp[..., 0], yp[..., 1]
            return torch.stack([torch.cos(pit) * torch.sin(yaw), torch.sin(pit), torch.cos(pit) * torch.cos(yaw)], -1)

        def forward(self, xs):
            h = self.body(xs["eye_norm_r16"])
            g_loc = self.local_gaze(h)                                               # (N,2,3)
            R_off = rodrigues(torch, self.w_roff) @ self.R_off0 if self.w_roff.requires_grad else self.R_off0
            R = xs["R_est"] @ R_off                                                  # (N,3,3) camera <- head
            if self.head_res:
                R = xs["R_est"] @ (axis_angle_batch(torch, 0.01 * self.res(h)) @ R_off)
            g_cam = torch.einsum("nij,nej->nei", R, g_loc)                           # (N,2,3)
            zf = (xs["zfeat"] - self.zf_mu) / self.zf_sd
            z = torch.exp(zf @ self.z_w + self.z_b)                                  # (N,)
            em = xs["eye_mid"]                                                       # (N,2,2) ncam x, y
            ray = torch.cat([em, torch.ones_like(em[..., :1])], -1)                  # (N,2,3)
            O = z[:, None, None] * ray + torch.einsum("nij,ej->nei", R, self.d_eye)  # (N,2,3)
            if "O_oracle" in xs:
                O = xs["O_oracle"]
            t = ((self.P_C[None, None] - O) @ self.P_N) / (g_cam @ self.P_N)        # (N,2)
            P = O + g_cam * t[..., None]
            Pm = P.mean(1)
            d = Pm - self.P_C
            uv = torch.stack([d @ self.P_R / self.half_wh[0], d @ self.P_U / self.half_wh[1]], 1)
            return uv, {"g_loc": g_loc, "g_cam": g_cam, "O": O, "P": P}
    return M9()


def axis_angle_batch(torch, w):
    th = torch.linalg.norm(w, dim=1, keepdim=True) + 1e-9
    k = w / th
    K = torch.zeros(len(w), 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=w.device)[None] + s * K + (1 - c) * K @ K


def procrustes_offset(Ra, Hb):
    M = np.einsum("nji,njk->ik", Ra, Hb)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def m9_init(base, tr, geo):
    """Constants of the physical model fitted on TRAINING identities only (Blender labels allowed in training)."""
    g = D.Geometry(base["geometry"])
    R_off = procrustes_offset(geo["R_est"][tr], base["H_cam"][tr])
    Rh = geo["R_est"] @ R_off
    # which MediaPipe eye (A = 33/133, B = 263/362) is the Blender left eye?  compare image x of GT eye centres
    em = geo["eye_mid"]
    OLx = base["O_L"][:, 0] / base["O_L"][:, 2]
    a_is_left = np.mean(np.abs(em[tr, 0, 0] - OLx[tr])) < np.mean(np.abs(em[tr, 1, 0] - OLx[tr]))
    O_true = np.stack([base["O_L"], base["O_R"]], 1) if a_is_left else np.stack([base["O_R"], base["O_L"]], 1)
    G_true = np.stack([base["G_L"], base["G_R"]], 1) if a_is_left else np.stack([base["G_R"], base["G_L"]], 1)
    zfeat = np.column_stack([geo["log_s"], geo["log_ipd"], geo["log_iris_d"]])
    mu, sd = zfeat[tr].mean(0), zfeat[tr].std(0) + 1e-9
    Z = np.column_stack([(zfeat - mu) / sd, np.ones(len(zfeat))])
    zc = O_true[:, :, 2].mean(1)
    w, *_ = np.linalg.lstsq(Z[tr], np.log(zc[tr]), rcond=None)
    z = np.exp(Z @ w)
    d_eye = np.zeros((2, 3))
    for e in range(2):
        ray = np.column_stack([em[:, e, :2], np.ones(len(z))])
        resid = O_true[:, e] - z[:, None] * ray
        d_eye[e] = np.einsum("nji,nj->ni", Rh[tr], resid[tr]).mean(0)
    G_loc = np.einsum("nji,nej->nei", base["H_cam"], G_true)                   # Blender head-frame eye-in-head GT
    g0 = G_loc[tr].mean(0)
    g0 /= np.linalg.norm(g0, axis=1, keepdims=True)
    yp0 = np.stack([np.arctan2(g0[:, 0], g0[:, 2]), np.arcsin(g0[:, 1])], 1)
    return {"R_off": R_off, "a_is_left": bool(a_is_left), "zf_mu": mu, "zf_sd": sd, "z_w": w[:3], "z_b": float(w[3]),
            "d_eye": d_eye, "g0": g0, "yp0": yp0, "P_C": g.P_C, "P_N": g.P_N, "P_R": g.P_R, "P_U": g.P_U,
            "half_wh": [g.disp_w / 2, g.disp_h / 2]}, O_true, G_true, G_loc, zfeat


# ===================================================================================== training
def group_batches(gid_tr, batch, rng):
    """Mini-batches of whole groups (index positions into the training array)."""
    order = collections.defaultdict(list)
    for j, gname in enumerate(gid_tr):
        order[gname].append(j)
    groups = list(order.values())
    perm = rng.permutation(len(groups))
    out, cur = [], []
    for k in perm:
        cur += groups[k]
        if len(cur) >= batch:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def run(args, fold, seed, base, prep_cache):
    torch, nn = make_torch()
    rdir = os.path.join(D.OUT_ROOT, args.exp, f"fold{fold}_seed{seed}")
    if os.path.exists(os.path.join(rdir, "result.json")):
        print(f"[skip] {rdir}", flush=True)
        return json.load(open(os.path.join(rdir, "result.json")))
    os.makedirs(rdir, exist_ok=True)
    if fold not in prep_cache:
        prep_cache.clear()
        prep_cache[fold] = prepare(args, fold, base)
    sp, tr, va, te, X, geo, template = prep_cache[fold]
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    g = D.Geometry(base["geometry"])
    hidden = [int(x) for x in args.hidden.split(",") if x]
    fuse_hidden = [int(x) for x in args.fuse_hidden.split(",") if x]
    stats = {b: (X[b][tr].mean(0), X[b][tr].std(0) + 1e-6) for b in ("head_pose", "eye_norm_r16")}
    aux_stats = {"head": (base["HEAD_AUX"][tr].mean(0), base["HEAD_AUX"][tr].std(0) + 1e-6),
                 "eye": (base["EYE_AUX"][tr].mean(0), base["EYE_AUX"][tr].std(0) + 1e-6)}
    init = None
    if args.model == "m9":
        init, O_true, G_true, G_loc, zfeat = m9_init(base, tr, geo)

    def T(a, dtype=torch.float32):
        return torch.tensor(np.asarray(a), dtype=dtype, device=dev)

    def inputs(idx):
        xs = {b: T((X[b][idx] - stats[b][0]) / stats[b][1]) for b in stats}
        if args.model == "m9":
            R_in = geo["R_est"][idx]
            if "head" in args.m9_oracle:          # diagnostic: true head rotation expressed so that R_in @ R_off = H
                R_in = base["H_cam"][idx] @ init["R_off"].T
            xs["R_est"] = T(R_in)
            if "origin" in args.m9_oracle:
                xs["O_oracle"] = T(O_true[idx])
            xs["zfeat"] = T(zfeat[idx])
            xs["eye_mid"] = T(geo["eye_mid"][idx, :, :2])
        return xs

    def targets(idx):
        t = {"uv": T(base["UV"][idx] * g.uv_cm),
             "ha": T((base["HEAD_AUX"][idx] - aux_stats["head"][0]) / aux_stats["head"][1]),
             "ea": T((base["EYE_AUX"][idx] - aux_stats["eye"][0]) / aux_stats["eye"][1])}
        if args.model == "m9":
            t["G"] = T(G_true[idx])
            t["Gl"] = T(G_loc[idx])
            t["O"] = T(O_true[idx])
        return t

    if args.model == "m8":
        net = build_m8(nn, {b: X[b].shape[1] for b in stats}, hidden, fuse_hidden, args.dropout).to(dev)
    else:
        net = build_m9(nn, X["eye_norm_r16"].shape[1], hidden, args.dropout, init, args.m9_out, args.m9_learn_roff, args.m9_head_res).to(dev)
    phys = {"w_roff", "z_w", "z_b", "d_eye"}
    groups = [{"params": [p for n_, p in net.named_parameters() if p.requires_grad and n_ not in phys], "lr": 2e-3, "weight_decay": args.wd}]
    ph = [p for n_, p in net.named_parameters() if p.requires_grad and n_ in phys]
    if ph:   # physical constants: initialised from the training-identity fit, small lr, no weight decay
        groups.append({"params": ph, "lr": args.m9_phys_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    X_tr, X_va = inputs(tr), inputs(va)
    tr_lab = tr
    if args.noise_aug > 0:
        # landmark-noise augmentation: noisy replicas of the training samples (features recomputed exactly as at
        # runtime: noisy landmarks -> tgaze rich16d -> features with the SAME template); labels unchanged
        assert args.model == "m8"
        import p7infer as PI
        rng_n = np.random.default_rng([seed, 99])
        Wd, Hd = base["geometry"]["W"], base["geometry"]["H"]
        for rep in range(args.noise_reps):
            lms_n = (base["LMS"][tr] + rng_n.normal(0, args.noise_aug, size=base["LMS"][tr].shape) / np.array([Wd, Hd, Wd])).astype(np.float32)
            f16_n, _ = PI.rich16d_batch(lms_n, base["geometry"]["K_measured_raw"], Wd, Hd)
            Xn, _, _ = D.fold_features(base, None, lms=lms_n, f16=f16_n, template=template, head_angles=args.head_angles)
            for b in stats:
                X_tr[b] = torch.cat([X_tr[b], T((Xn[b] - stats[b][0]) / stats[b][1])])
            tr_lab = np.concatenate([tr_lab, tr])
    T_tr, T_va = targets(tr_lab), targets(va)
    sc_uv = T(g.uv_cm)
    is_st = base["gtype"][tr_lab] == "same_target"
    gid_tr = base["gid"][tr_lab]

    uv_tr_cm = base["UV"][tr_lab] * g.uv_cm

    def sel_uv_std(j):
        return T(uv_tr_cm[j].std(0))

    def cons_loss(y_cm, j):
        """y_cm (B,2) predictions in display cm for batch positions j (whole groups)."""
        st = torch.as_tensor(is_st[j], device=dev)
        if not bool(st.any()):
            return torch.zeros((), device=dev)
        names = gid_tr[j][is_st[j]]
        _, inv = np.unique(names, return_inverse=True)
        inv_t = torch.as_tensor(inv, device=dev)
        ys = y_cm[st]
        G = int(inv.max()) + 1
        sums = torch.zeros(G, 2, device=dev).index_add_(0, inv_t, ys)
        cnt = torch.zeros(G, device=dev).index_add_(0, inv_t, torch.ones(len(ys), device=dev))
        mean = sums / cnt[:, None]
        if args.cons == "mean_dev":
            return torch.sqrt(((ys - mean[inv_t]) ** 2).sum(1) + 1e-4).mean()
        if args.cons == "scaled":
            # scale-invariant: deviations measured in units of the batch prediction spread (per axis), converted
            # back to cm with the batch TARGET spread -> shrinking all outputs toward the centre does not reduce it
            s_pred = y_cm.std(0) + 1e-3
            s_true = sel_uv_std(j)
            dv = (ys - mean[inv_t]) / s_pred * s_true
            return torch.sqrt((dv ** 2).sum(1) + 1e-4).mean()
        # pairwise: mean over all same-group pairs of the distance
        dd = torch.cdist(ys, ys)
        same = (inv_t[:, None] == inv_t[None]) & ~torch.eye(len(ys), dtype=torch.bool, device=dev)
        return torch.sqrt(dd[same] ** 2 + 1e-4).mean()

    def loss_fn(y, a, Tt, j=None, train=True):
        sel = (lambda t: t) if j is None else (lambda t: t[torch.as_tensor(j, device=dev)])
        y_cm = y[:, :2] * sc_uv if args.model == "m9" else y[:, :2] * sc_uv
        parts = {}
        if args.model == "m8":
            parts["target"] = nn.functional.huber_loss(y_cm, sel(Tt["uv"]), delta=1.0)
            parts["aux"] = args.aux_weight * (nn.functional.mse_loss(a["head"], sel(Tt["ha"])) + nn.functional.mse_loss(a["eye"], sel(Tt["ea"])))
        else:
            if args.m9_loss == "screen":
                parts["target"] = nn.functional.huber_loss(y_cm, sel(Tt["uv"]), delta=1.0)
            elif args.m9_loss == "angular":
                cos = (a["g_cam"] * sel(Tt["G"])).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                parts["target"] = torch.rad2deg(torch.acos(cos)).mean()
            if args.m9_loss == "local" or args.m9_local_w > 0:
                cosl = (a["g_loc"] * sel(Tt["Gl"])).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                w = 1.0 if args.m9_loss == "local" else args.m9_local_w
                parts["local"] = w * torch.rad2deg(torch.acos(cosl)).mean()
            parts["origin"] = args.m9_origin_w * nn.functional.smooth_l1_loss(a["O"] * 100, sel(Tt["O"]) * 100)
        if args.lam > 0 and j is not None and train:
            parts["cons"] = args.lam * cons_loss(y_cm, j)
        return sum(parts.values()), parts

    def predict(xs):
        net.eval()
        with torch.no_grad():
            y, a = net(xs)
        return y.detach().cpu().numpy().astype(np.float64)

    hist = collections.defaultdict(list)
    best = (1e9, None, -1)
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        tot, parts_acc, n = 0.0, collections.Counter(), 0
        for j in group_batches(gid_tr, args.batch, rng):
            jt = torch.as_tensor(j, device=dev)
            y, a = net({k: v[jt] for k, v in X_tr.items()})
            loss, parts = loss_fn(y, a, T_tr, j)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
            n += len(j)
            for k, v in parts.items():
                parts_acc[k] += float(v) * len(j)
        net.eval()
        with torch.no_grad():
            y_va, a_va = net(X_va)
            vloss = float(loss_fn(y_va, a_va, T_va, None, train=False)[0])
            y_tr, _ = net(X_tr)
        puv_tr = y_tr.cpu().numpy().astype(np.float64)[:len(tr)]
        puv_va = y_va.cpu().numpy().astype(np.float64)
        if args.model == "m8":
            puv_tr, puv_va = puv_tr[:, :2], puv_va[:, :2]
        e_tr = D.errors(base, tr, puv_tr)[0].mean()
        e_va = D.errors(base, va, puv_va)[0].mean()
        pinv = D.pose_invariance(base, va, puv_va)
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(vloss)
        hist["train_cm"].append(float(e_tr))
        hist["val_cm"].append(float(e_va))
        hist["val_pose_inv_cm"].append(pinv["mean"] if pinv else None)
        for k, v in parts_acc.items():
            hist["train_" + k].append(v / n)
        sched.step(e_va)
        if e_va < best[0]:
            best = (float(e_va), {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] > args.patience:
            break
    net.load_state_dict(best[1])
    secs = time.time() - t0
    preds = {}
    res = {}
    for s, idx in (("train", tr), ("validation", va), ("test", te)):
        p = predict(inputs(idx))[:, :2]
        preds[s] = p
        res[s] = D.evaluate(base, idx, p)
    gap = res["test"]["all"]["cm_mean"] - res["train"]["all"]["cm_mean"]
    gpu = torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu"
    out = {"exp": args.exp, "fold": fold, "seed": seed, "split": sp, "n": {"train": len(tr), "validation": len(va), "test": len(te)},
           "config": {k: v for k, v in vars(args).items()}, "best_epoch": best[2], "epochs_run": ep + 1, "train_seconds": secs,
           "training_device": str(dev), "gpu": gpu, "params": int(sum(p.numel() for p in net.parameters())),
           "generalization_gap_cm": gap, "metrics": res}
    if init is not None:
        out["m9_init"] = {k: (np.asarray(v).tolist() if not isinstance(v, (bool, float, int)) else v) for k, v in init.items()}
        with torch.no_grad():
            out["m9_learned"] = {"z_w": net.z_w.cpu().numpy().tolist(), "z_b": float(net.z_b), "d_eye": net.d_eye.cpu().numpy().tolist(),
                                 "w_roff_deg": float(np.degrees(np.linalg.norm(net.w_roff.cpu().numpy())))}
    D.dump_json(dict(hist), os.path.join(rdir, "curves.json"))
    if args.save_preds:
        np.savez_compressed(os.path.join(rdir, "pred.npz"), tr=tr, va=va, te=te, p_tr=preds["train"], p_va=preds["validation"], p_te=preds["test"])
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "model": args.model, "hidden": hidden,
                "fuse_hidden": fuse_hidden, "stats": {b: [stats[b][0], stats[b][1]] for b in stats}, "template": template,
                "init": init, "config": vars(args), "pca": geo.get("pca")}, os.path.join(rdir, "model.pt"))
    D.dump_json(out, os.path.join(rdir, "result.json"))
    r = res
    pi = r["test"]["pose_invariance_cm"]["mean"] if r["test"]["pose_invariance_cm"] else float("nan")
    print(f"{args.exp:36s} f{fold} s{seed} train {r['train']['all']['cm_mean']:.3f} | val {r['validation']['all']['cm_mean']:.3f} | "
          f"test {r['test']['all']['cm_mean']:.3f} p95 {r['test']['all']['cm_p95']:.2f} poseInv {pi:.2f} (ep {best[2]}, {secs:.0f}s, {gpu})", flush=True)
    return out


def main(argv=None):
    args = build_args(argv)
    roots = [D.DATA_V2] + ([r for r in args.root.split(",")] if args.root else [])
    if len(roots) == 1:
        base = D.load_base()
    else:
        base = merge_bases([D.load_base(r) for r in roots])
    prep_cache = {}
    for fold in [int(x) for x in args.folds.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            run(args, fold, seed, base, prep_cache)


def merge_bases(bases):
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


if __name__ == "__main__":
    main()
