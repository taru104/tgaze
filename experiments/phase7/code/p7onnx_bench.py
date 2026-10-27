"""Phase 7H step 2 (.venv-tools, NO torch, GPUs hidden): build ONNX from p7export.py files, verify, benchmark on CPU.

  CUDA_VISIBLE_DEVICES= .venv-tools/bin/python phase7/p7onnx_bench.py NAME [NAME ...] [--frames 300] [--threads 0]
  CUDA_VISIBLE_DEVICES= taskset -c 0-3 .venv-tools/bin/python phase7/p7onnx_bench.py NAME --threads 4 --tag 4cores

Graph (all standard ONNX ops, CPUExecutionProvider):
  M8 / M8+Cons : standardise -> 2 branch MLPs -> concat -> fusion MLP -> (u, v) [cm-scaled uv]
  M9 / M9+Cons : standardise -> eye MLP -> 2 x 3D vectors (+ mean local gaze) -> L2 normalise (eye-in-head)
                 R = R_kabsch @ R_off ; g_cam = R g_local ; O = z (x, y, 1) + R d ; z = exp(w.zf + b)
                 ray-plane intersection with the display plane -> mean of both eyes -> (u, v)
Frame pipeline: frame -> cv2.flip -> MediaPipe FaceLandmarker (CPU delegate, IMAGE mode on stills)
  -> tgaze rich16d (measured K, mirrored=True) -> NumPy features (Kabsch head pose etc.) -> ONNX -> (u, v)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

PH7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PH7)
import p7data as D  # noqa: E402  (numpy only)
import landmark_features as lf  # noqa: E402

TGAZE = os.path.expanduser("~/tgaze")
sys.path.insert(0, TGAZE)


def build_onnx(name):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    d = os.path.join(D.OUT_ROOT, "cpu_runtime", name)
    w = dict(np.load(os.path.join(d, "weights.npz")))
    prep = json.load(open(os.path.join(d, "preprocessing.json")))
    nodes, inits, inputs = [], [], []
    cnt = [0]

    def const(arr, nm=None):
        cnt[0] += 1
        nm = nm or f"c{cnt[0]}"
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), nm))
        return nm

    def op(t, ins, **kw):
        cnt[0] += 1
        o = f"t{cnt[0]}"
        nodes.append(helper.make_node(t, ins, [o], **kw))
        return o

    def seq_linears(prefix):
        keys = sorted({k.rsplit(".", 1)[0] for k in w if k.startswith(prefix) and k.endswith(".weight")},
                      key=lambda k: [int(x) for x in k[len(prefix):].split(".") if x.isdigit()])
        return [(w[k + ".weight"], w[k + ".bias"]) for k in keys]

    def mlp(x, layers, relu_last=True):
        for i, (W, b) in enumerate(layers):
            x = op("Add", [op("MatMul", [x, const(W.T)]), const(b)])
            if relu_last or i < len(layers) - 1:
                x = op("Relu", [x])
        return x

    def stdz(b):
        mean, std = (np.array(v, np.float32) for v in prep["standardization"][b])
        inputs.append(helper.make_tensor_value_info(b, TensorProto.FLOAT, ["N", len(mean)]))
        return op("Div", [op("Sub", [b, const(mean)]), const(std)])

    if prep["model"] == "m8":
        lat = []
        for b in ("head_pose", "eye_norm_r16"):
            x = stdz(b)
            x = mlp(x, seq_linears(f"enc.{b}.0.") + seq_linears(f"enc.{b}.1"))
            lat.append(x)
        x = op("Concat", lat, axis=1)
        x = mlp(x, seq_linears("fuse."))
        y = op("Add", [op("MatMul", [x, const(w["out.weight"].T)]), const(w["out.bias"])])
        nodes.append(helper.make_node("Identity", [y], ["y"]))
    else:
        m9 = prep["m9"]
        x = mlp(stdz("eye_norm_r16"), seq_linears("body."))
        o = op("Add", [op("MatMul", [x, const(w["out.weight"].T)]), const(w["out.bias"])])
        assert m9["out_kind"] == "vec3"
        g = op("Add", [op("Reshape", [o, const_i64(inits, numpy_helper, [-1, 2, 3], "shp")]), const(w["g0"][None])])
        nrm = op("ReduceL2", [g, const_i64(inits, numpy_helper, [-1], "ax_last")], keepdims=1)
        gl = op("Div", [g, nrm])                                                   # (N,2,3) eye-in-head
        for nm, shape in (("R_est", ["N", 3, 3]), ("zfeat", ["N", 3]), ("eye_mid", ["N", 2, 2])):
            inputs.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, shape))
        R = op("MatMul", ["R_est", const(w["R_off0"])])                           # (N,3,3)
        gc = op("MatMul", [gl, op("Transpose", [R], perm=[0, 2, 1])])             # g_cam[n,e,:] = R[n] @ gl[n,e]
        zf = op("Div", [op("Sub", ["zfeat", const(w["zf_mu"])]), const(w["zf_sd"])])
        z = op("Exp", [op("Add", [op("MatMul", [zf, const(w["z_w"][:, None])]), const(w["z_b"].reshape(1, 1))])])   # (N,1)
        ones = op("Add", [op("Mul", [op("Slice", ["eye_mid", const_i64(inits, numpy_helper, [0], "s0"), const_i64(inits, numpy_helper, [1], "s1"),
                                                     const_i64(inits, numpy_helper, [2], "s2")]), const(np.zeros(1))]), const(np.ones(1))])
        ray = op("Concat", ["eye_mid", ones], axis=2)                              # (N,2,3)
        dR = op("MatMul", [const(w["d_eye"][None]), op("Transpose", [R], perm=[0, 2, 1])])   # (N,2,3) = R d_e
        O = op("Add", [op("Mul", [ray, op("Unsqueeze", [z, const_i64(inits, numpy_helper, [2], "ax2")])]), dR])
        P_C, P_N = const(w["P_C"]), const(w["P_N"][:, None])
        num = op("MatMul", [op("Sub", [P_C, O]), P_N])                            # (N,2,1)
        den = op("MatMul", [gc, P_N])
        P = op("Add", [O, op("Mul", [gc, op("Div", [num, den])])])
        Pm = op("ReduceMean", [P, const_i64(inits, numpy_helper, [1], "ax1")], keepdims=0)    # (N,3)
        dd = op("Sub", [Pm, P_C])
        RU = np.stack([w["P_R"] / w["half_wh"][0], w["P_U"] / w["half_wh"][1]], 1)  # (3,2)
        nodes.append(helper.make_node("MatMul", [dd, const(RU)], ["y"]))
    graph = helper.make_graph(nodes, f"phase7_{name}", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="phase7", ir_version=9)
    onnx.checker.check_model(model)
    path = os.path.join(d, f"{name}.onnx")
    onnx.save(model, path)
    return path, prep


def const_i64(inits, numpy_helper, v, nm):
    if not any(i.name == nm for i in inits):
        inits.append(numpy_helper.from_array(np.array(v, np.int64), nm))
    return nm


class Pipeline:
    def __init__(self, name, threads=0):
        import onnxruntime as ort
        import mediapipe as mp
        import rich16d
        self.d = os.path.join(D.OUT_ROOT, "cpu_runtime", name)
        self.prep = json.load(open(os.path.join(self.d, "preprocessing.json")))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(self.d, f"{name}.onnx"), so, providers=["CPUExecutionProvider"])
        assert self.sess.get_providers() == ["CPUExecutionProvider"]
        self.rich16d = rich16d
        self.W, self.H = self.prep["image_size"]
        rich16d.set_camera_matrix(np.array(self.prep["K_measured_raw"]), (self.W, self.H), mirrored=True)
        self.K = np.array(self.prep["K_mirrored"])
        self.template = np.array(self.prep["template_rigid"])
        self.stdz = self.prep["standardization"]
        V = mp.tasks.vision
        self.mp = mp
        self.lm = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(TGAZE, "face_landmarker.task"), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=V.RunningMode.IMAGE, num_faces=1))
        if self.prep["eye_pca"]:
            self.pca_mean = np.array(self.prep["eye_pca"]["mean"], np.float32)
            self.pca_comp = np.array(self.prep["eye_pca"]["components"], np.float32)

    def features(self, lms, f16, patch=None):
        NC = lf.to_ncam(lms[None], self.K, self.W, self.H)
        FE = lf.features(NC, self.template, f16[None][:, [4, 5, 7]])
        eye = np.hstack([FE["eye_norm"], f16[None][:, D.R16_EYE]]).astype(np.float32)
        if self.prep["eye_pca"]:
            eye = np.hstack([eye, ((patch[None] - self.pca_mean) @ self.pca_comp.T).astype(np.float32)])
        feeds = {"eye_norm_r16": eye}
        if self.prep["model"] == "m8":
            feeds["head_pose"] = np.hstack([FE["head"][:, :9], FE["head"][:, -3:]]).astype(np.float32)
        else:
            rig = NC[:, lf.RIGID]
            s = np.sqrt(((rig - rig.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
            iris = np.stack([NC[:, 468:473].mean(1), NC[:, 473:478].mean(1)], 1)
            ipd = np.linalg.norm(iris[:, 0, :2] - iris[:, 1, :2], axis=1)
            iris_d = np.stack([np.linalg.norm(NC[:, 469, :2] - NC[:, 471, :2], axis=1), np.linalg.norm(NC[:, 474, :2] - NC[:, 476, :2], axis=1)], 1).mean(1)
            feeds["R_est"] = FE["R_head"].astype(np.float32)
            feeds["zfeat"] = np.column_stack([np.log(s), np.log(ipd), np.log(iris_d)]).astype(np.float32)
            feeds["eye_mid"] = np.stack([NC[:, list(D.CORNERS_A)].mean(1), NC[:, list(D.CORNERS_B)].mean(1)], 1)[:, :, :2].astype(np.float32)
        return feeds

    def frame(self, bgr):
        import cv2
        t0 = time.perf_counter()
        mir = cv2.flip(bgr, 1)
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(mir, cv2.COLOR_BGR2RGB)))
        t1 = time.perf_counter()
        if not res.face_landmarks:
            return None, (t1 - t0, 0, 0)
        lms = self.rich16d.lms_to_array(res.face_landmarks[0])
        f16 = np.asarray(self.rich16d.rich_16d_from_lms(lms, self.W, self.H), np.float64)
        patch = None
        if self.prep["eye_pca"]:
            from appearance import eye_patch
            patch = np.asarray(eye_patch(mir, res.face_landmarks[0]), np.float32)
        feeds = self.features(lms, f16, patch)
        t2 = time.perf_counter()
        y = self.sess.run(["y"], feeds)[0]
        t3 = time.perf_counter()
        return y[0], (t1 - t0, t2 - t1, t3 - t2)       # display (u, v) for both model kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    import cv2
    import onnxruntime as ort
    import psutil
    results = {}
    frames = None
    for name in a.names:
        path, prep = build_onnx(name)
        ref = np.load(os.path.join(D.OUT_ROOT, "cpu_runtime", name, "reference_io.npz"))
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = {}
        for i in sess.get_inputs():                      # reference inputs were saved standardised -> raw
            v = ref["in_" + i.name]
            if i.name in prep["standardization"]:
                v = (v * np.array(prep["standardization"][i.name][1]) + np.array(prep["standardization"][i.name][0])).astype(np.float32)
            feeds[i.name] = v
        y = sess.run(["y"], feeds)[0]
        diff = float(np.abs(y - ref["y"]).max())
        pl = Pipeline(name, a.threads)
        # NumPy feature code == training feature code on the reference landmarks
        # runtime NumPy features (standardised as in the graph) == training-time inputs on the reference landmarks
        fd = 0.0
        for i in range(16):
            ft = pl.features(ref["lms"][i].astype(np.float64), ref["f16"][i].astype(np.float64), ref["patch"][i])
            for k, v in ft.items():
                if k in prep["standardization"]:
                    v = (v - np.array(prep["standardization"][k][0])) / np.array(prep["standardization"][k][1])
                fd = max(fd, float(np.abs(v - ref["in_" + k][i:i + 1]).max()))
        print(f"[{name}] onnx max |diff| vs torch {diff:.2e}; runtime-feature max |diff| vs training inputs {fd:.2e}", flush=True)
        assert diff < 1e-3 and fd < 1e-3
        if frames is None:
            base = D.load_run_base(prep["source_run"])
            roots = D.run_roots(prep["source_run"])
            te = np.load(os.path.join(prep["source_run"], "pred.npz"))["te"]
            sel = np.random.default_rng(0).choice(te, a.frames, replace=False)
            frames = [cv2.imread(D.image_path(roots, int(base["ident"][i]), base["sample_id"][i])) for i in sel]
            uv_true = base["UV"][sel]
        for f in frames[:10]:
            pl.frame(f)
        proc = psutil.Process()
        c0, w0 = proc.cpu_times(), time.perf_counter()
        lat, preds = [], []
        for f in frames:
            t = time.perf_counter()
            uv, parts = pl.frame(f)
            lat.append((time.perf_counter() - t, *parts))
            preds.append(uv)
        wall = time.perf_counter() - w0
        c1 = proc.cpu_times()
        lat = np.array(lat) * 1000
        preds = np.array(preds)
        err = np.hypot((preds[:, 0] - uv_true[:, 0]) * 14.53, (preds[:, 1] - uv_true[:, 1]) * 9.45)
        gpu_apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
        ms = lambda x: {"mean": float(np.mean(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}
        results[name] = {
            "onnx_path": path, "onnx_size_bytes": os.path.getsize(path), "onnx_max_abs_diff_vs_torch": diff, "feature_max_abs_diff": fd,
            "ort_session_providers": sess.get_providers(), "ort_available_providers": ort.get_available_providers(),
            "torch_imported": "torch" in sys.modules, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "this_process_on_gpu": str(proc.pid) in gpu_apps, "cpu_affinity": sorted(proc.cpu_affinity()), "ort_threads": a.threads,
            "latency_ms": {"total": ms(lat[:, 0]), "mediapipe": ms(lat[:, 1]), "features": ms(lat[:, 2]), "onnx": ms(lat[:, 3])},
            "fps_mean": float(1000 / lat[:, 0].mean()), "fps_p95_frame": float(1000 / np.percentile(lat[:, 0], 95)),
            "cpu_percent_of_one_core": float(100 * ((c1.user - c0.user) + (c1.system - c0.system)) / wall),
            "rss_mb": proc.memory_info().rss / 2 ** 20, "frames": len(frames), "pipeline_error_cm_on_these_frames": float(err.mean()),
            "cpu_model": [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]}
        r = results[name]
        print(f"[{name}/{a.tag}] {r['latency_ms']['total']['mean']:.1f} ms -> {r['fps_mean']:.1f} FPS (mediapipe {r['latency_ms']['mediapipe']['mean']:.1f}, "
              f"features {r['latency_ms']['features']['mean']:.2f}, onnx {r['latency_ms']['onnx']['mean']:.3f} ms); CPU {r['cpu_percent_of_one_core']:.0f}%; "
              f"RSS {r['rss_mb']:.0f} MB; size {r['onnx_size_bytes'] / 1024:.1f} KB; torch {r['torch_imported']}; on GPU {r['this_process_on_gpu']}", flush=True)
        pl.lm.close()
        del pl
    p = os.path.join(D.OUT_ROOT, "cpu_runtime", f"benchmark_{a.tag}.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(results)
    D.dump_json(old, p)


if __name__ == "__main__":
    main()
