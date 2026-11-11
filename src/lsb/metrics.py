"""Similarity measures compared in the benchmark.

Every measure exposes the same two calls:

    m.fit(X, y, y_cls)      X is the z-scored training matrix (n x d); y the continuous
                            (or 0/1) outcome; y_cls integer class labels for the
                            supervised metric learners that need discrete labels.
    m.pairwise(A, B)        distances between the rows of A and the rows of B.

Mahalanobis-type measures (fixed or learned) store a linear map L so that the distance
is the Euclidean distance between A @ L.T and B @ L.T.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS, LedoitWolf

EPS = 1e-10


# --------------------------------------------------------------------------- helpers
def _map_from_psd(M: np.ndarray) -> np.ndarray:
    """Return L with L.T @ L = M for a symmetric positive semi-definite M."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)).T


def _pinv_psd(S: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Moore-Penrose pseudo-inverse of a symmetric PSD matrix (drops null directions)."""
    S = (S + S.T) / 2.0
    w, V = np.linalg.eigh(S)
    keep = w > rtol * max(w.max(), EPS)
    inv = np.zeros_like(w)
    inv[keep] = 1.0 / w[keep]
    return (V * inv) @ V.T


class Measure:
    name = "base"
    family = "fixed"          # fixed | covariance | learned
    supervised = False

    def fit(self, X, y=None, y_cls=None):
        return self

    def pairwise(self, A, B):
        raise NotImplementedError


class LinearMeasure(Measure):
    """Distance = Euclidean distance after the linear map L (Mahalanobis family)."""

    L: np.ndarray | None = None

    def pairwise(self, A, B):
        return cdist(A @ self.L.T, B @ self.L.T, metric="euclidean")

    def matrix(self) -> np.ndarray:
        return self.L.T @ self.L


# --------------------------------------------------------------------- fixed measures
class Euclidean(LinearMeasure):
    name = "euclid"

    def fit(self, X, y=None, y_cls=None):
        self.L = np.eye(X.shape[1])
        return self


class Manhattan(Measure):
    name = "manhattan"

    def pairwise(self, A, B):
        return cdist(A, B, metric="cityblock")


class Cosine(Measure):
    name = "cosine"

    def pairwise(self, A, B):
        An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
        Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), EPS)
        return np.clip(1.0 - An @ Bn.T, 0.0, 2.0)


class Pearson(Measure):
    """1 - Pearson correlation between two learner vectors (row-centred cosine)."""

    name = "pearson"

    def pairwise(self, A, B):
        return Cosine().pairwise(A - A.mean(axis=1, keepdims=True), B - B.mean(axis=1, keepdims=True))


# ------------------------------------------------- covariance-aware (unsupervised)
class MahalanobisSample(LinearMeasure):
    """Mahalanobis with the sample covariance; pseudo-inverse when it is singular."""

    name = "maha_sample"
    family = "covariance"

    def fit(self, X, y=None, y_cls=None):
        S = np.cov(X, rowvar=False, bias=True)
        self.L = _map_from_psd(_pinv_psd(np.atleast_2d(S)))
        return self


class MahalanobisLW(LinearMeasure):
    """Mahalanobis with the Ledoit-Wolf shrinkage covariance (always invertible)."""

    name = "maha_lw"
    family = "covariance"

    def fit(self, X, y=None, y_cls=None):
        est = LedoitWolf(assume_centered=False).fit(X)
        self.shrinkage_ = float(est.shrinkage_)
        self.L = _map_from_psd(np.linalg.inv(est.covariance_))
        return self


class MahalanobisOAS(LinearMeasure):
    """Mahalanobis with the Oracle Approximating Shrinkage covariance."""

    name = "maha_oas"
    family = "covariance"

    def fit(self, X, y=None, y_cls=None):
        est = OAS(assume_centered=False).fit(X)
        self.shrinkage_ = float(est.shrinkage_)
        self.L = _map_from_psd(np.linalg.inv(est.covariance_))
        return self


class MahalanobisAlpha(LinearMeasure):
    """Mahalanobis with a FIXED shrinkage weight: M = ((1 - alpha) S + alpha I)^-1.

    S is the sample covariance of the z-scored features (their correlation matrix).
    alpha = 1 is the Euclidean distance and alpha -> 0 the sample Mahalanobis distance,
    so the family traces the whole path between the two.
    """

    family = "covariance"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.name = f"maha_a{int(round(alpha * 100)):02d}"

    def fit(self, X, y=None, y_cls=None):
        S = np.atleast_2d(np.cov(X, rowvar=False, bias=True))
        self.L = _map_from_psd(np.linalg.inv((1 - self.alpha) * S + self.alpha * np.eye(S.shape[0])))
        return self


class PCAProject(LinearMeasure):
    """Euclidean distance in the span of the leading principal components (no whitening).

    The opposite of whitening: directions of small variance are discarded instead of
    being amplified.
    """

    family = "covariance"

    def __init__(self, var: float = 0.8):
        self.var = var
        self.name = f"pca{int(round(var * 100))}_proj"

    def fit(self, X, y=None, y_cls=None):
        S = np.atleast_2d(np.cov(X, rowvar=False, bias=True))
        w, V = np.linalg.eigh((S + S.T) / 2.0)
        order = np.argsort(w)[::-1]
        w, V = np.clip(w[order], 0.0, None), V[:, order]
        cum = np.cumsum(w) / max(w.sum(), EPS)
        r = max(1, int(np.searchsorted(cum, self.var) + 1))
        self.n_components_ = r
        self.L = V[:, :r].T
        return self


class PCAWhiten(LinearMeasure):
    """Truncated PCA whitening: keep the components explaining `var` of the variance.

    With var = 1 this equals MahalanobisSample; truncation is the classical
    regulariser, so the measure answers the "it is only whitening" objection.
    """

    family = "covariance"

    def __init__(self, var: float = 0.95):
        self.var = var
        self.name = f"pca{int(round(var * 100))}_whiten"

    def fit(self, X, y=None, y_cls=None):
        S = np.atleast_2d(np.cov(X, rowvar=False, bias=True))
        w, V = np.linalg.eigh((S + S.T) / 2.0)
        order = np.argsort(w)[::-1]
        w, V = np.clip(w[order], 0.0, None), V[:, order]
        cum = np.cumsum(w) / max(w.sum(), EPS)
        r = int(np.searchsorted(cum, self.var) + 1)
        r = max(1, min(r, int((w > 1e-8 * max(w[0], EPS)).sum())))
        self.n_components_ = r
        self.L = (V[:, :r] / np.sqrt(w[:r])).T
        return self


# ------------------------------------------------------- learned (supervised) metrics
class _MetricLearn(LinearMeasure):
    """Wrapper around metric-learn / scikit-learn estimators.

    If the learner fails (singular problem, too few points per class, no convergence)
    the measure falls back to the Euclidean distance and records `failed_ = True`,
    so a failure is counted against the method instead of being silently dropped.
    """

    family = "learned"
    supervised = True

    def _learn(self, X, y, y_cls) -> np.ndarray:  # returns the linear map L
        raise NotImplementedError

    def fit(self, X, y=None, y_cls=None):
        self.failed_ = False
        d = X.shape[1]
        nz = np.flatnonzero(X.std(axis=0) > 1e-12)   # columns that vary in the training sample
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Lnz = np.asarray(self._learn(X[:, nz], y, y_cls), dtype=float)
            if not np.all(np.isfinite(Lnz)) or np.allclose(Lnz, 0.0):
                raise FloatingPointError("degenerate transformation")
            L = np.zeros((Lnz.shape[0], d))
            L[:, nz] = Lnz                           # constant training columns get zero weight
            self.L = L
        except Exception as e:  # noqa: BLE001 - any learner failure is a recorded outcome
            self.failed_ = True
            self.error_ = f"{type(e).__name__}: {str(e)[:120]}"
            self.L = np.eye(d)
        return self


class LMNNMeasure(_MetricLearn):
    name = "lmnn"

    def __init__(self, k: int = 3, max_iter: int = 200):
        self.k, self.max_iter = k, max_iter

    def _learn(self, X, y, y_cls):
        from metric_learn import LMNN

        k = int(min(self.k, np.bincount(y_cls).min() - 1))
        if k < 1:
            raise ValueError("a class has fewer than two members")
        m = LMNN(n_neighbors=k, max_iter=self.max_iter, random_state=0).fit(X, y_cls)
        return m.components_


class NCAMeasure(_MetricLearn):
    name = "nca"

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter

    def _learn(self, X, y, y_cls):
        from sklearn.neighbors import NeighborhoodComponentsAnalysis

        m = NeighborhoodComponentsAnalysis(max_iter=self.max_iter, random_state=0).fit(X, y_cls)
        return m.components_


class ITMLMeasure(_MetricLearn):
    """ITML with an identity prior or the Ledoit-Wolf precision as prior."""

    def __init__(self, prior: str = "identity", max_iter: int = 300, n_constraints: int | None = None):
        self.prior, self.max_iter, self.n_constraints = prior, max_iter, n_constraints
        self.name = "itml" if prior == "identity" else "itml_lw"

    def _learn(self, X, y, y_cls):
        from metric_learn import ITML_Supervised

        prior = "identity"
        if self.prior == "lw":
            prior = np.linalg.inv(LedoitWolf().fit(X).covariance_)
        n_cls = len(np.unique(y_cls))
        n_con = self.n_constraints or 20 * n_cls * n_cls  # 20 c^2, as in Davis et al. (2007)
        m = ITML_Supervised(prior=prior, max_iter=self.max_iter, n_constraints=n_con,
                            random_state=0).fit(X, y_cls)
        return m.components_


class SDMLMeasure(_MetricLearn):
    """Sparse Determinant Metric Learning (Qi et al., 2009).

    metric-learn 0.7.0 calls scikit-learn's graphical lasso with an argument removed in
    scikit-learn 1.5, so its SDML cannot run in this environment. This is a direct
    re-implementation of the same estimator:
        M = graphical_lasso( M0^-1 + balance * mean_ij y_ij (x_i - x_j)(x_i - x_j)^T , sparsity )
    with y_ij = +1 for same-class pairs and -1 for different-class pairs, 20 c^2 pairs of
    each kind (the pair loss is averaged per kind, which keeps it on the scale of a
    covariance), and M0 the Ledoit-Wolf precision. Eigenvalues of the input matrix are
    floored at 1e-3 times the largest one so that the solver receives a positive-definite
    matrix.
    """

    name = "sdml"

    def __init__(self, sparsity: float = 0.01, balance: float = 0.5):
        self.sparsity, self.balance = sparsity, balance

    def _learn(self, X, y, y_cls):
        from sklearn.covariance import graphical_lasso

        rng = np.random.default_rng(0)
        n, d = X.shape
        n_cls = len(np.unique(y_cls))
        n_con = 20 * n_cls * n_cls
        a = rng.integers(0, n, size=40 * n_con)
        b = rng.integers(0, n, size=40 * n_con)
        ok = a != b
        a, b = a[ok], b[ok]
        same = y_cls[a] == y_cls[b]
        pos = np.flatnonzero(same)[:n_con]
        neg = np.flatnonzero(~same)[:n_con]
        dp = X[a[pos]] - X[b[pos]]
        dn = X[a[neg]] - X[b[neg]]
        loss = dp.T @ dp / max(len(dp), 1) - dn.T @ dn / max(len(dn), 1)   # pair-averaged loss
        prior_inv = LedoitWolf().fit(X).covariance_          # inverse of the prior metric M0
        emp = prior_inv + self.balance * loss
        emp = (emp + emp.T) / 2.0
        w, V = np.linalg.eigh(emp)
        if w.min() <= 1e-3 * w.max():
            w = np.clip(w, 1e-3 * w.max(), None)
            emp = (V * w) @ V.T
        _, M = graphical_lasso(emp, alpha=self.sparsity, max_iter=100)
        if not np.all(np.isfinite(M)):
            raise FloatingPointError("graphical lasso returned non-finite values")
        return _map_from_psd(M)


class LFDAMeasure(_MetricLearn):
    name = "lfda"

    def _learn(self, X, y, y_cls):
        from metric_learn import LFDA

        k = int(max(1, min(7, np.bincount(y_cls).min() - 1)))
        m = LFDA(k=k).fit(X, y_cls)
        return m.components_


class MLKRMeasure(_MetricLearn):
    """Metric Learning for Kernel Regression: uses the continuous outcome directly."""

    name = "mlkr"

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter

    def _learn(self, X, y, y_cls):
        from metric_learn import MLKR

        m = MLKR(max_iter=self.max_iter, random_state=0).fit(X, np.asarray(y, dtype=float))
        return m.components_


# ------------------------------------------------------------------------- registry
def make_measures(include_learned: bool = True) -> list[Measure]:
    ms: list[Measure] = [
        Euclidean(), Manhattan(), Cosine(), Pearson(),
        MahalanobisSample(), MahalanobisLW(), MahalanobisOAS(), PCAWhiten(0.95), PCAProject(0.8),
        MahalanobisAlpha(0.10), MahalanobisAlpha(0.25), MahalanobisAlpha(0.50),
        MahalanobisAlpha(0.75), MahalanobisAlpha(0.90),
    ]
    if include_learned:
        ms += [LMNNMeasure(), NCAMeasure(), ITMLMeasure("identity"), ITMLMeasure("lw"),
               SDMLMeasure(), LFDAMeasure(), MLKRMeasure()]
    return ms


FAMILY = {
    "euclid": "fixed", "manhattan": "fixed", "cosine": "fixed", "pearson": "fixed",
    "euclid_raw": "fixed", "cosine_raw": "fixed",
    "maha_sample": "covariance", "maha_lw": "covariance", "maha_oas": "covariance",
    "pca95_whiten": "covariance", "pca80_proj": "covariance",
    "maha_a10": "covariance", "maha_a25": "covariance", "maha_a50": "covariance",
    "maha_a75": "covariance", "maha_a90": "covariance",
    "lmnn": "learned", "nca": "learned", "itml": "learned", "itml_lw": "learned",
    "sdml": "learned", "lfda": "learned", "mlkr": "learned",
    "ridge": "model", "rf": "model", "gbdt": "model", "mean": "model",
}

LABEL = {
    "euclid": "Euclidean", "manhattan": "Manhattan", "cosine": "Cosine", "pearson": "Pearson",
    "euclid_raw": "Euclidean (raw features)", "cosine_raw": "Cosine (raw features)",
    "maha_sample": "Mahalanobis (sample Σ)", "maha_lw": "Mahalanobis (Ledoit–Wolf)",
    "maha_oas": "Mahalanobis (OAS)", "pca95_whiten": "PCA-95% whitening",
    "pca80_proj": "PCA-80% projection", "maha_a10": "Mahalanobis α=0.10",
    "maha_a25": "Mahalanobis α=0.25", "maha_a50": "Mahalanobis α=0.50",
    "maha_a75": "Mahalanobis α=0.75", "maha_a90": "Mahalanobis α=0.90",
    "lmnn": "LMNN", "nca": "NCA", "itml": "ITML (identity prior)",
    "itml_lw": "ITML (LW prior)", "sdml": "SDML", "lfda": "LFDA", "mlkr": "MLKR",
    "ridge": "Ridge / logistic", "rf": "Random forest", "gbdt": "Gradient boosting",
    "mean": "Mean predictor",
}
