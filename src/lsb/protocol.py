"""Evaluation protocol: learner-disjoint sampling, k-NN prediction, scores, diagnostics.

One "cell" is (view, d, n). One repetition draws n training learners from the candidate
pool (without replacement); the test learners are fixed and never used for fitting.
Scaling, covariance estimates and learned metrics are fitted on the training sample only.
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import skew
from sklearn.covariance import OAS, LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score

from metrics import Cosine, Euclidean, LinearMeasure, MahalanobisLW, MahalanobisSample, make_measures

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
KS = (5, 10, 20)
_CACHE: dict[str, dict] = {}


def load_view(name: str) -> dict:
    if name not in _CACHE:
        z = np.load(PROC / f"{name}.npz", allow_pickle=False)
        meta = json.loads((PROC / f"{name}.json").read_text(encoding="utf-8"))
        _CACHE[name] = dict(X=z["X"], y=z["y"], part=z["part"], users=z["users"], meta=meta)
    return _CACHE[name]


def rep_seed(view: str, n: int, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([zlib.crc32(view.encode()), int(n), int(rep)])


Z_CLIP = 5.0
MIN_SUPPORT = 5


def zscore(Xtr: np.ndarray, *others: np.ndarray):
    """Standardise with the TRAINING mean and SD (training-only preprocessing).

    * Support filter: a column is dropped (set to zero everywhere) when fewer than
      MIN_SUPPORT training learners depart from its most frequent value. For a
      knowledge-component column that value is the prior used for "never attempted", so
      the rule removes KCs that fewer than five training learners have practised; for an
      activity-count column it removes activities that fewer than five have used.
    * Clip at +-5 SD to bound the leverage of the remaining low-support columns.
    """
    n, d = Xtr.shape
    support = np.array([n - np.unique(Xtr[:, j], return_counts=True)[1].max() for j in range(d)])
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    drop = (support < MIN_SUPPORT) | (sd < 1e-12)
    sd = np.where(drop, 1.0, sd)
    out = []
    for A in (Xtr,) + others:
        Z = np.clip((A - mu) / sd, -Z_CLIP, Z_CLIP)
        Z[:, drop] = 0.0
        out.append(Z)
    return out


def class_labels(y: np.ndarray, binary: bool):
    """Discrete labels for the supervised metric learners and for neighbour precision.

    Binary outcomes keep their two classes; continuous outcomes are cut at the training
    tertiles (median split if a tertile class would have fewer than four members).
    Returns (labels, cut points)."""
    if binary:
        return y.astype(int), np.array([0.5])
    cuts = np.quantile(y, [1 / 3, 2 / 3])
    lab = np.digitize(y, cuts)
    if len(np.unique(lab)) < 3 or np.bincount(lab, minlength=3).min() < 4:
        cuts = np.array([np.median(y)])
        lab = np.digitize(y, cuts)
    return lab.astype(int), cuts


def scores(y: np.ndarray, p: np.ndarray, ref: float, binary: bool) -> dict:
    """r2 = 1 - MSE(p) / MSE(train-mean predictor).  For a binary outcome and a
    probability p this is the Brier skill score, so every view shares one scale."""
    sse = float(np.mean((y - p) ** 2))
    sse0 = float(np.mean((y - ref) ** 2))
    out = dict(r2=1.0 - sse / max(sse0, 1e-12), mae=float(np.mean(np.abs(y - p))))
    if binary:
        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return out


# ------------------------------------------------------------------------ diagnostics
def signal_alignment(Xtr: np.ndarray, ytr: np.ndarray) -> dict:
    """Where does the predictive signal sit relative to the variance of the features?

    rho_j is the correlation between the outcome and the j-th principal-component score.
    sva = sum_j rho_j^2 lambda_j / (sum_j rho_j^2 * mean lambda)   (signal-variance alignment)
    sva > 1: the outcome is carried by HIGH-variance directions (which whitening shrinks);
    sva < 1: it is carried by LOW-variance directions (which whitening amplifies).
    """
    n = Xtr.shape[0]
    S = np.atleast_2d(np.cov(Xtr, rowvar=False, bias=True))
    lam, V = np.linalg.eigh((S + S.T) / 2.0)
    keep = lam > 1e-10 * max(lam.max(), 1e-12)
    lam, V = lam[keep], V[:, keep]
    sc = (Xtr @ V) / np.sqrt(lam)
    sy = ytr.std()
    if sy < 1e-12 or len(lam) == 0:
        return dict(sva=np.nan, rho2_pc1=np.nan, rho2_sum=np.nan, top_eig_share=np.nan)
    rho2 = (sc.T @ ((ytr - ytr.mean()) / sy) / n) ** 2
    return dict(sva=float((rho2 * lam).sum() / max(rho2.sum() * lam.mean(), 1e-12)),
                rho2_pc1=float(rho2[np.argmax(lam)]), rho2_sum=float(rho2.sum()),
                top_eig_share=float(lam.max() / lam.sum()))


def diagnostics(Xtr: np.ndarray) -> dict:
    """Label-free properties of the (z-scored) training sample."""
    n, d = Xtr.shape
    S = np.atleast_2d(np.cov(Xtr, rowvar=False, bias=True))
    lam = np.clip(np.linalg.eigvalsh((S + S.T) / 2.0), 0.0, None)[::-1]
    tot = max(lam.sum(), 1e-12)
    p = lam[lam > 1e-12] / tot
    erank = float(np.exp(-(p * np.log(p)).sum()))
    pr = float(tot ** 2 / max((lam ** 2).sum(), 1e-12))
    rank = int((lam > 1e-8 * max(lam[0], 1e-12)).sum())
    cond = float(lam[0] / lam[-1]) if lam[-1] > 1e-10 * lam[0] else np.inf
    lw = LedoitWolf().fit(Xtr)
    oas = OAS().fit(Xtr)
    lam_lw = np.linalg.eigvalsh(lw.covariance_)
    off = np.abs(S[np.triu_indices(d, 1)]) if d > 1 else np.array([0.0])
    dist = pdist(Xtr)
    # TwoNN intrinsic dimension (Facco et al., 2017) and distance concentration
    from scipy.spatial.distance import squareform
    D = squareform(dist)
    np.fill_diagonal(D, np.inf)
    two = np.sort(D, axis=1)[:, :2]
    ok = two[:, 0] > 1e-12
    mu = two[ok, 1] / two[ok, 0]
    id_twonn = float(ok.sum() / max(np.log(mu).sum(), 1e-12)) if ok.sum() > 5 else np.nan
    nn = np.argsort(D, axis=1)[:, :10]
    hub = float(skew(np.bincount(nn.ravel(), minlength=n)))
    return dict(n_const=int((Xtr.std(axis=0) < 1e-12).sum()), cov_rank=rank,
                log10_cond=float(np.log10(cond)) if np.isfinite(cond) else np.inf,
                log10_cond_lw=float(np.log10(lam_lw[-1] / max(lam_lw[0], 1e-300))),
                lw_alpha=float(lw.shrinkage_), oas_alpha=float(oas.shrinkage_),
                erank=erank, part_ratio=pr, mean_abs_corr=float(off.mean()),
                dist_cv=float(dist.std() / max(dist.mean(), 1e-12)),
                id_twonn=id_twonn, hub_skew_train=hub, dup_frac=float(1.0 - ok.mean()))


# --------------------------------------------------------------------------- k-NN eval
def knn_eval(D: np.ndarray, ytr: np.ndarray, yte: np.ndarray, ref: float, binary: bool,
             lab_tr: np.ndarray, lab_te: np.ndarray) -> list[dict]:
    order = np.argsort(D, axis=1, kind="stable")[:, :max(KS)]
    n = D.shape[1]
    hub = float(skew(np.bincount(order[:, :10].ravel(), minlength=n)))
    rows = []
    for k in KS:
        idx = order[:, :k]
        pred = ytr[idx].mean(axis=1)
        s = scores(yte, pred, ref, binary)
        s["nprec"] = float((lab_tr[idx] == lab_te[:, None]).mean())
        s["k"] = k
        s["hub_skew"] = hub
        rows.append(s)
    return rows


def loo_r2(Dtt: np.ndarray, ytr: np.ndarray, k: int = 10) -> float:
    """Leave-one-out k-NN skill inside the training sample (used for data-driven selection).

    For label-free measures this is a valid internal estimate. For learned metrics it is a
    resubstitution estimate (the metric has seen the left-out label) and is optimistic."""
    D = Dtt.copy()
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    pred = ytr[idx].mean(axis=1)
    n = len(ytr)
    ref = (ytr.sum() - ytr) / (n - 1)
    return float(1.0 - np.mean((ytr - pred) ** 2) / max(np.mean((ytr - ref) ** 2), 1e-12))


def model_baselines(Xtr, ytr, Xte, binary: bool, seed: int):
    """Non-metric baselines with fixed, disclosed settings."""
    out = {}
    if binary:
        m = LogisticRegressionCV(Cs=8, cv=3, max_iter=1000, random_state=seed).fit(Xtr, ytr.astype(int))
        out["ridge"] = m.predict_proba(Xte)[:, 1]
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(
            Xtr, ytr.astype(int))
        out["rf"] = m.predict_proba(Xte)[:, 1]
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=5,
                                           early_stopping=False, random_state=seed).fit(Xtr, ytr.astype(int))
        out["gbdt"] = m.predict_proba(Xte)[:, 1]
    else:
        m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Xtr, ytr)
        out["ridge"] = m.predict(Xte)
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(Xtr, ytr)
        out["rf"] = m.predict(Xte)
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=5,
                                          early_stopping=False, random_state=seed).fit(Xtr, ytr)
        out["gbdt"] = m.predict(Xte)
    return out


def evaluate_split(Xtr_raw, ytr, Xte_raw, yte, binary: bool, base: dict, rep: int,
                   learned: bool = True, models: bool = True, transductive: bool = True):
    """Fit every measure on the training learners and score it on the test learners."""
    Xtr, Xte = zscore(Xtr_raw, Xte_raw)
    ref = float(ytr.mean())
    lab_tr, cuts = class_labels(ytr, binary)
    lab_te = yte.astype(int) if binary else np.digitize(yte, cuts)
    rows = []

    for m in make_measures(include_learned=learned):
        t0 = time.perf_counter()
        m.fit(Xtr, ytr, lab_tr)
        fit_s = time.perf_counter() - t0
        D = m.pairwise(Xte, Xtr)
        loo = loo_r2(m.pairwise(Xtr, Xtr), ytr)
        for s in knn_eval(D, ytr, yte, ref, binary, lab_tr, lab_te):
            rows.append({**base, "method": m.name, "fit_s": fit_s, "loo_r2": loo,
                         "failed": bool(getattr(m, "failed_", False)), **s})

    # The two measures as used in earlier work: no standardisation of the ability vector.
    for nm, m in [("euclid_raw", Euclidean().fit(Xtr_raw)), ("cosine_raw", Cosine())]:
        D = m.pairwise(Xte_raw, Xtr_raw)
        loo = loo_r2(m.pairwise(Xtr_raw, Xtr_raw), ytr)
        for s in knn_eval(D, ytr, yte, ref, binary, lab_tr, lab_te):
            rows.append({**base, "method": nm, "fit_s": 0.0, "loo_r2": loo, "failed": False, **s})

    if transductive:
        # Protocol contrast: scaling and covariance estimated on train + test features
        # (no labels). This is the common "standardise / whiten the whole table first" shortcut.
        Zall, Ztr, Zte = zscore(np.vstack([Xtr_raw, Xte_raw]), Xtr_raw, Xte_raw)
        for cls, nm in [(MahalanobisLW, "maha_lw_trans"), (MahalanobisSample, "maha_sample_trans")]:
            m = cls().fit(Zall)
            D = m.pairwise(Zte, Ztr)
            for s in knn_eval(D, ytr, yte, ref, binary, lab_tr, lab_te):
                rows.append({**base, "method": nm, "fit_s": 0.0, "loo_r2": np.nan, "failed": False, **s})

    if models:
        t0 = time.perf_counter()
        preds = model_baselines(Xtr, ytr, Xte, binary, seed=rep)
        fit_s = time.perf_counter() - t0
        preds["mean"] = np.full(len(yte), ref)
        for name, p in preds.items():
            # every outcome in the benchmark is a rate or a probability, so model
            # predictions are clipped to [0, 1] (k-NN averages are inside it by construction)
            s = scores(yte, np.clip(p, 0.0, 1.0), ref, binary)
            rows.append({**base, "method": name, "fit_s": fit_s / 3, "loo_r2": np.nan, "failed": False,
                         "k": 0, "nprec": np.nan, "hub_skew": np.nan, **s})

    diag = {**base, **diagnostics(Xtr), **signal_alignment(Xtr, ytr)}
    return rows, diag


def _draw(rng, pool, y, n, binary):
    for _ in range(50):
        tr = rng.choice(pool, size=n, replace=False)
        if not binary or 3 <= y[tr].sum() <= n - 3:
            break
    return tr


def run_rep(view: str, n: int, rep: int, learned: bool = True, models: bool = True,
            transductive: bool = True):
    """One repetition of one cell (training learners drawn from the view's own pool)."""
    v = load_view(view)
    X, y, part, meta = v["X"], v["y"], v["part"], v["meta"]
    binary = meta["task"] == "warning"
    rng = np.random.default_rng(rep_seed(view, n, rep))
    tr = _draw(rng, np.flatnonzero(part == 2), y, n, binary)
    test = np.flatnonzero(part == 1)
    base = dict(view=view, source=meta["source"], task=meta["task"], d=meta["d"], n=n, rep=rep)
    return evaluate_split(X[tr], y[tr], X[test], y[test], binary, base, rep, learned, models, transductive)


def run_lomo_rep(view: str, n: int, rep: int, learned: bool = True, models: bool = True):
    """Leave-one-module-out for OULAD: the test learners are those of `view`; the training
    learners are drawn from the candidate pools of the OTHER modules, after removing every
    learner who also appears in the held-out module (learner- and module-disjoint)."""
    v = load_view(view)
    meta = v["meta"]
    held_ids = set(v["users"].tolist())
    others = sorted(p.stem for p in PROC.glob("oulad_*__d*.npz") if p.stem != view)
    Xs, ys = [], []
    for o in others:
        w = load_view(o)
        keep = (w["part"] == 2) & ~np.isin(w["users"], list(held_ids))
        Xs.append(w["X"][keep]); ys.append(w["y"][keep])
    Xp, yp = np.vstack(Xs), np.concatenate(ys)
    rng = np.random.default_rng(rep_seed(view + "|lomo", n, rep))
    tr = _draw(rng, np.arange(len(yp)), yp, n, True)
    test = np.flatnonzero(v["part"] == 1)
    base = dict(view=view, source=meta["source"], task=meta["task"], d=meta["d"], n=n, rep=rep,
                protocol="leave-one-module-out")
    return evaluate_split(Xp[tr], yp[tr], v["X"][test], v["y"][test], True, base, rep, learned, models,
                          transductive=False)
