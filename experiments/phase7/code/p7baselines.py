"""Phase 7 baselines in the Phase 7 run format (deterministic, CPU, sklearn; same recipe as Phase 5):
  B0 mean, B1 rich16d + Huber, B2 rich16d + eye-patch PCA16 + Huber (tgaze default features).
  ~/tgaze/.venv/bin/python phase7/p7baselines.py      -> baseline/B0_mean, baseline/B1_rich16d_huber, baseline/B2_rich16d_pca16_huber
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7data as D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import HuberRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def huber_fit(X, Y):
    sc = StandardScaler().fit(X)
    ms = [HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=3000).fit(sc.transform(X), Y[:, i]) for i in range(2)]
    return lambda Z: np.column_stack([m.predict(sc.transform(Z)) for m in ms])


base = D.load_base()
UV, F16, PATCH = base["UV"], base["F16"], base["PATCH"]
for fold in range(5):
    sp = D.cv_split(fold)
    tr, va, te = D.split_indices(base, sp)
    pca = PCA(n_components=16).fit(PATCH[tr])
    mu = UV[tr].mean(0)
    h1 = huber_fit(F16[tr], UV[tr])
    h2 = huber_fit(np.hstack([F16[tr], pca.transform(PATCH[tr])]), UV[tr])
    models = {"B0_mean": lambda i: np.tile(mu, (len(i), 1)), "B1_rich16d_huber": lambda i: h1(F16[i]),
              "B2_rich16d_pca16_huber": lambda i: h2(np.hstack([F16[i], pca.transform(PATCH[i])]))}
    for name, fn in models.items():
        rdir = os.path.join(D.OUT_ROOT, "baseline", name, f"fold{fold}_seed0")
        os.makedirs(rdir, exist_ok=True)
        t0 = time.time()
        preds = {s: fn(i) for s, i in (("train", tr), ("validation", va), ("test", te))}
        res = {s: D.evaluate(base, i, preds[s]) for s, i in (("train", tr), ("validation", va), ("test", te))}
        D.dump_json({"exp": f"baseline/{name}", "fold": fold, "seed": 0, "split": sp, "params": 0, "best_epoch": 0, "train_seconds": time.time() - t0,
                     "training_device": "cpu", "gpu": "cpu", "config": {"model": name}, "metrics": res,
                     "generalization_gap_cm": res["test"]["all"]["cm_mean"] - res["train"]["all"]["cm_mean"]}, os.path.join(rdir, "result.json"))
        np.savez_compressed(os.path.join(rdir, "pred.npz"), tr=tr, va=va, te=te, p_tr=preds["train"], p_va=preds["validation"], p_te=preds["test"])
        print(fold, name, round(res["test"]["all"]["cm_mean"], 3), flush=True)
