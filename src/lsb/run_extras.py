"""Sensitivity and robustness experiments on a subset of cells.

A. Hyper-parameter sensitivity of the learned metrics: low-rank maps, number of target
   neighbours, and how the continuous outcome is discretised into class labels.
B. Robustness of the main measures to (1) missing practice data and (2) label noise.

Cells: every view at d = 50 (OULAD: its single feature set), n in {100, 500}, 10 repetitions
with the same cohorts as the main grid (same seeds).

    python -I src/lsb/run_extras.py --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def _variants():
    """Learned-metric variants. Each returns the linear map L from (X, y, labels)."""
    from metric_learn import LMNN, MLKR
    from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

    from metrics import _MetricLearn

    def make(name, fn):
        cls = type(name, (_MetricLearn,), {"name": name, "_learn": lambda self, X, y, c: fn(X, y, c)})
        return cls()

    def kk(c, k):
        return int(max(1, min(k, np.bincount(c).min() - 1)))

    def rank(X, r):
        return int(min(r, X.shape[1]))

    return [
        make("lmnn_k1", lambda X, y, c: LMNN(n_neighbors=kk(c, 1), max_iter=200, random_state=0).fit(X, c).components_),
        make("lmnn_k7", lambda X, y, c: LMNN(n_neighbors=kk(c, 7), max_iter=200, random_state=0).fit(X, c).components_),
        make("lmnn_rank10", lambda X, y, c: LMNN(n_neighbors=kk(c, 3), n_components=rank(X, 10), max_iter=200,
                                                 random_state=0).fit(X, c).components_),
        make("nca_rank10", lambda X, y, c: NCA(n_components=rank(X, 10), max_iter=100, random_state=0).fit(X, c).components_),
        make("nca_rank5", lambda X, y, c: NCA(n_components=rank(X, 5), max_iter=100, random_state=0).fit(X, c).components_),
        make("mlkr_rank10", lambda X, y, c: MLKR(n_components=rank(X, 10), max_iter=100, random_state=0).fit(
            X, np.asarray(y, float)).components_),
    ]


def _labels(y, q):
    cuts = np.quantile(y, np.linspace(0, 1, q + 1)[1:-1])
    return np.digitize(y, cuts).astype(int)


def _eval(measure, name, Xtr, ytr, lab_fit, Xte, yte, ref, binary, lab_tr, lab_te, base, extra):
    from protocol import knn_eval

    t0 = time.perf_counter()
    measure.fit(Xtr, ytr, lab_fit)
    fit_s = time.perf_counter() - t0
    s = knn_eval(measure.pairwise(Xte, Xtr), ytr, yte, ref, binary, lab_tr, lab_te)[1]   # k = 10
    return {**base, **extra, "method": name, "fit_s": fit_s, "failed": bool(getattr(measure, "failed_", False)),
            "r2": s["r2"], "nprec": s["nprec"]}


def one(view, n, rep):
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression, RidgeCV

    from metrics import (Cosine, Euclidean, LMNNMeasure, MahalanobisLW, MahalanobisSample, MLKRMeasure,
                         NCAMeasure, PCAProject)
    from protocol import _draw, class_labels, load_view, rep_seed, scores, zscore

    v = load_view(view)
    X, y, part, meta = v["X"], v["y"], v["part"], v["meta"]
    binary = meta["task"] == "warning"
    rng = np.random.default_rng(rep_seed(view, n, rep))        # same cohort as the main grid
    tr = _draw(rng, np.flatnonzero(part == 2), y, n, binary)
    te = np.flatnonzero(part == 1)
    base = dict(view=view, source=meta["source"], task=meta["task"], d=meta["d"], n=n, rep=rep)
    rows = []

    # ---------------- A. sensitivity of the learned metrics (clean data)
    Xtr, Xte = zscore(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    ref = float(ytr.mean())
    lab_tr, cuts = class_labels(ytr, binary)
    lab_te = yte.astype(int) if binary else np.digitize(yte, cuts)
    args = (Xte, yte, ref, binary, lab_tr, lab_te, base)
    for m in [LMNNMeasure(), NCAMeasure(), MLKRMeasure()]:
        rows.append(_eval(m, m.name, Xtr, ytr, lab_tr, *args, dict(part="A", cond="default")))
    for m in _variants():
        rows.append(_eval(m, m.name, Xtr, ytr, lab_tr, *args, dict(part="A", cond="variant")))
    if not binary:
        for q, tag in [(2, "median"), (5, "quintile")]:
            lab_q = _labels(ytr, q)
            if np.bincount(lab_q).min() >= 4:
                for m in [LMNNMeasure(), NCAMeasure()]:
                    rows.append(_eval(m, f"{m.name}_{tag}", Xtr, ytr, lab_q, *args, dict(part="A", cond="labels")))

    # ---------------- C. is retrieval more than matching on overall ability?
    from metrics import LinearMeasure

    class Ability1D(LinearMeasure):          # distance between the learners' mean standardised feature
        name = "ability_1d"

        def fit(self, X, y=None, y_cls=None):
            self.L = np.full((1, X.shape[1]), 1.0 / X.shape[1])
            return self

    class PC1(LinearMeasure):                # distance along the first principal component only
        name = "pc1"

        def fit(self, X, y=None, y_cls=None):
            w, V = np.linalg.eigh(np.atleast_2d(np.cov(X, rowvar=False, bias=True)))
            self.L = V[:, [-1]].T
            return self

    for m in [Euclidean(), Cosine(), PCAProject(0.8), PCAProject(0.5), PC1(), Ability1D()]:
        rows.append(_eval(m, m.name, Xtr, ytr, lab_tr, *args, dict(part="C", cond="clean")))

    # ---------------- B. robustness of the main measures
    prng = np.random.default_rng([rep, n, 7])
    col_mode = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        vals, cnt = np.unique(X[:, j], return_counts=True)
        col_mode[j] = vals[np.argmax(cnt)]             # the "never attempted" value of the column

    def perturb(cond):
        Xa, Xb, ya = X[tr].copy(), X[te].copy(), ytr.copy()
        if cond.startswith("mask"):
            p = int(cond[4:]) / 100.0
            for A in (Xa, Xb):
                hit = (A != col_mode) & (prng.random(A.shape) < p)
                A[hit] = np.broadcast_to(col_mode, A.shape)[hit]
        elif cond == "noise":
            if binary:
                flip = prng.random(len(ya)) < 0.15
                ya = np.where(flip, 1 - ya, ya)
            else:
                ya = np.clip(ya + prng.normal(0, 0.5 * ytr.std(), len(ya)), 0, 1)
        return Xa, Xb, ya

    for cond in ["clean", "mask30", "mask60", "noise"]:
        Xa, Xb, ya = perturb(cond) if cond != "clean" else (X[tr], X[te], ytr)
        if binary and not (3 <= ya.sum() <= len(ya) - 3):
            continue
        Za, Zb = zscore(Xa, Xb)
        refb = float(ya.mean())
        lab_a, cuts_a = class_labels(ya, binary)
        lab_b = yte.astype(int) if binary else np.digitize(yte, cuts_a)
        argsb = (Zb, yte, refb, binary, lab_a, lab_b, base)
        for m in [Euclidean(), Cosine(), MahalanobisSample(), MahalanobisLW(), PCAProject(0.8),
                  NCAMeasure(), MLKRMeasure()]:
            rows.append(_eval(m, m.name, Za, ya, lab_a, *argsb, dict(part="B", cond=cond)))
        if binary:
            p = LogisticRegression(C=0.1, max_iter=1000).fit(Za, ya.astype(int)).predict_proba(Zb)[:, 1]
        else:
            p = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Za, ya).predict(Zb)
        rows.append({**base, "part": "B", "cond": cond, "method": "ridge", "fit_s": 0.0, "failed": False,
                     "r2": scores(yte, np.clip(p, 0, 1), refb, binary)["r2"], "nprec": np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n", nargs="*", type=int, default=[100, 500])
    ap.add_argument("--out", default="results/extras")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(PROC / "views_index.csv")
    idx = idx[~idx["name"].str.contains("randsplit")]
    names = [r["name"] for _, r in idx.iterrows() if r["d"] == 50 or r["task"] == "warning"]
    cells = [(v, n) for v in names for n in a.n
             if int(idx.loc[idx["name"] == v, "n_pool"].iloc[0]) >= 1.5 * n and not (out / f"{v}__n{n}.parquet").exists()]
    print(f"{len(cells)} cells x {a.reps} reps", flush=True)
    t0 = time.time()
    with Parallel(n_jobs=a.jobs, batch_size=1) as par:
        for i, (v, n) in enumerate(cells, 1):
            res = par(delayed(one)(v, n, r) for r in range(a.reps))
            pd.DataFrame([r for rr in res for r in rr]).to_parquet(out / f"{v}__n{n}.parquet")
            print(f"[{i}/{len(cells)}] {v} n={n}  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
