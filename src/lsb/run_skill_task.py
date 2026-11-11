"""Fifth task: SKILL-SPECIFIC future performance.

The four main tasks predict an overall outcome, so a one-dimensional "general ability"
match could be enough. Here the outcome is the learner's future success rate on ONE
knowledge component, for five target KCs per source, so the profile matters by design.

Sources: the four temporal sources with enough practice per KC (ASSISTments 2017, EdNet,
XES3G5M, Slepemapy). Features: the d = 50 history vector of the main views (unchanged).
Target KCs: the five KCs with the most reference-pool learners who attempt them at least
three times in the future segment. Learners without three future attempts on a target KC
are not used for that KC.

    python -I src/lsb/run_skill_task.py --jobs 4
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
KCOUT = PROC / "kc_outcomes"
VIEWS = {"assist17": "build_assist17", "ednet": "build_ednet", "xes3g5m": "build_xes3g5m",
         "slepemapy": "build_slepemapy"}
M_TARGETS, MIN_FUT, D = 5, 3, 50


def build_outcomes():
    """Per-KC future success rates, aligned with the learners of the existing d=50 views."""
    import build_views as bv

    KCOUT.mkdir(parents=True, exist_ok=True)
    todo = [v for v in VIEWS if not (KCOUT / f"{v}.npz").exists()]
    if not todo:
        return
    captured = {}

    def capture(df, source, view, domain, min_total=80, random_split=False):
        if not random_split:
            captured[view] = (df, min_total)
        return []

    bv.temporal_view = capture            # reuse the loaders; nothing is written to the main views
    for v in todo:
        getattr(bv, VIEWS[v])()
        df, min_total = captured[v]
        df = df.sort_values(["user", "order"], kind="mergesort")
        inter = df.drop_duplicates(["user", "iid"])[["user", "iid", "correct"]]
        cnt = inter.groupby("user")["iid"].transform("size").to_numpy()
        pos = inter.groupby("user").cumcount().to_numpy()
        inter = inter.assign(is_hist=pos < np.floor(bv.HIST_FRAC * cnt))[cnt >= min_total]
        fut = df.merge(inter.loc[~inter["is_hist"], ["user", "iid"]], on=["user", "iid"], how="inner")
        z = np.load(PROC / f"{v}__d{D}.npz")
        users, part = z["users"], z["part"]
        g = fut.groupby(["user", "kc"])["correct"].agg(["mean", "size"]).reset_index()
        g = g[(g["size"] >= MIN_FUT) & g["user"].isin(users)]
        ref_users = set(users[part == 0].tolist())
        pop = g[g["user"].isin(ref_users)].groupby("kc")["user"].nunique().sort_values(ascending=False, kind="mergesort")
        kcs = pop.index.to_numpy()[:M_TARGETS]
        Y = np.full((len(users), len(kcs)), np.nan)
        u_idx = pd.Series(np.arange(len(users)), index=users)
        for j, kc in enumerate(kcs):
            s = g[g["kc"] == kc]
            Y[u_idx.reindex(s["user"]).to_numpy().astype(int), j] = s["mean"].to_numpy()
        np.savez_compressed(KCOUT / f"{v}.npz", Y=Y, kcs=kcs, users=users)
        ok = ~np.isnan(Y)
        print(f"  [kc outcomes] {v}: targets={kcs.tolist()} pool with outcome={[(ok[:, j] & (part == 2)).sum() for j in range(len(kcs))]} "
              f"test={[(ok[:, j] & (part == 1)).sum() for j in range(len(kcs))]}", flush=True)
        del captured[v]


def one(view, j, n, rep):
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import RidgeCV

    from metrics import (Cosine, Euclidean, LinearMeasure, MahalanobisLW, MahalanobisSample, MLKRMeasure,
                         NCAMeasure, PCAProject)
    from protocol import class_labels, knn_eval, rep_seed, scores, zscore

    z = np.load(PROC / f"{view}__d{D}.npz")
    X, part = z["X"], z["part"]
    Y = np.load(KCOUT / f"{view}.npz")["Y"][:, j]
    ok = ~np.isnan(Y)
    pool, test = np.flatnonzero((part == 2) & ok), np.flatnonzero((part == 1) & ok)
    rng = np.random.default_rng(rep_seed(f"{view}|kc{j}", n, rep))
    tr = rng.choice(pool, size=n, replace=False)
    Xtr, Xte = zscore(X[tr], X[test])
    ytr, yte = Y[tr], Y[test]
    ref = float(ytr.mean())
    lab_tr, cuts = class_labels(ytr, False)
    lab_te = np.digitize(yte, cuts)
    base = dict(view=view, kc=int(j), d=D, n=n, rep=rep, n_test=len(test))

    class Ability1D(LinearMeasure):
        name = "ability_1d"

        def fit(self, X, y=None, y_cls=None):
            self.L = np.full((1, X.shape[1]), 1.0 / X.shape[1])
            return self

    rows = []
    for m in [Euclidean(), Cosine(), PCAProject(0.8), MahalanobisSample(), MahalanobisLW(), Ability1D(),
              NCAMeasure(), MLKRMeasure()]:
        m.fit(Xtr, ytr, lab_tr)
        s = knn_eval(m.pairwise(Xte, Xtr), ytr, yte, ref, False, lab_tr, lab_te)[1]
        rows.append({**base, "method": m.name, "r2": s["r2"], "nprec": s["nprec"],
                     "failed": bool(getattr(m, "failed_", False))})
    p = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Xtr, ytr).predict(Xte)
    rows.append({**base, "method": "ridge", "r2": scores(yte, np.clip(p, 0, 1), ref, False)["r2"],
                 "nprec": np.nan, "failed": False})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", nargs="*", type=int, default=[100, 500])
    ap.add_argument("--out", default="results/skill_task")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build_outcomes()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for v in VIEWS:
        z = np.load(PROC / f"{v}__d{D}.npz")
        Y = np.load(KCOUT / f"{v}.npz")["Y"]
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[:, j])
            n_pool, n_test = int(((z["part"] == 2) & ok).sum()), int(((z["part"] == 1) & ok).sum())
            for n in a.n:
                if n_pool >= 1.5 * n and n_test >= 100 and not (out / f"{v}__kc{j}__n{n}.parquet").exists():
                    cells.append((v, j, n))
    print(f"{len(cells)} cells x {a.reps} reps", flush=True)
    t0 = time.time()
    with Parallel(n_jobs=a.jobs, batch_size=1) as par:
        for i, (v, j, n) in enumerate(cells, 1):
            res = par(delayed(one)(v, j, n, r) for r in range(a.reps))
            pd.DataFrame([r for rr in res for r in rr]).to_parquet(out / f"{v}__kc{j}__n{n}.parquet")
            print(f"[{i}/{len(cells)}] {v} kc{j} n={n}  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
