"""Statistical comparison of several methods over several datasets.

Friedman test with Iman-Davenport correction, Nemenyi critical difference
(Demsar, 2006), Wilcoxon signed-rank with Holm correction, the Bayesian signed-rank
test with a region of practical equivalence (Benavoli et al., 2017), and a
cluster bootstrap for mean differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def average_ranks(scores: pd.DataFrame) -> pd.Series:
    """scores: rows = datasets (blocks), columns = methods, higher is better."""
    return scores.rank(axis=1, ascending=False, method="average").mean(axis=0)


def friedman(scores: pd.DataFrame) -> dict:
    N, K = scores.shape
    R = average_ranks(scores).to_numpy()
    chi2 = 12 * N / (K * (K + 1)) * (np.sum(R ** 2) - K * (K + 1) ** 2 / 4)
    ff = (N - 1) * chi2 / max(N * (K - 1) - chi2, 1e-12)        # Iman-Davenport
    p = float(stats.f.sf(ff, K - 1, (K - 1) * (N - 1)))
    return dict(N=N, K=K, chi2=float(chi2), F=float(ff), p=p)


def nemenyi_cd(K: int, N: int, alpha: float = 0.05) -> float:
    q = stats.studentized_range.ppf(1 - alpha, K, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(K * (K + 1) / (6.0 * N)))


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def bayes_signed_rank(a: np.ndarray, b: np.ndarray, rope: float = 0.01, n_samples: int = 20000,
                      prior: float = 0.5, seed: int = 0) -> dict:
    """Bayesian signed-rank test (Benavoli et al., 2014/2017), as in the `baycomp` package.

    Returns posterior probabilities that a is practically better than b (difference
    above +rope), practically equivalent (within the rope), or worse.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    z = np.concatenate([[0.0], d])                       # prior pseudo-observation at zero
    n = len(z)
    w = rng.dirichlet(np.concatenate([[prior], np.ones(n - 1)]), size=n_samples)
    s = z[:, None] + z[None, :]
    left = (s < -2 * rope).astype(float)
    right = (s > 2 * rope).astype(float)
    mid = 1.0 - left - right
    pl = np.einsum("si,ij,sj->s", w, left, w)
    pm = np.einsum("si,ij,sj->s", w, mid, w)
    pr = np.einsum("si,ij,sj->s", w, right, w)
    win = np.argmax(np.stack([pl, pm, pr], axis=1), axis=1)
    return dict(p_worse=float((win == 0).mean()), p_rope=float((win == 1).mean()),
                p_better=float((win == 2).mean()))


def cluster_bootstrap_diff(d: np.ndarray, clusters: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """Mean of d with a percentile CI that resamples whole clusters (data sources)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    ids = np.unique(clusters)
    groups = [d[clusters == c] for c in ids]
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def cd_diagram(ax, ranks: pd.Series, cd: float, labels: dict | None = None, title: str = ""):
    """Critical-difference diagram (Demsar, 2006): best (lowest) average rank on the left."""
    r = ranks.sort_values()
    names = [labels.get(k, k) if labels else k for k in r.index]
    K = len(r)
    lo, hi = 1, K
    ax.set_xlim(lo - 0.3, hi + 0.3)
    half = (K + 1) // 2
    height = 0.42 * half + 1.3
    ax.set_ylim(height, -0.9)
    ax.hlines(0, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, -0.08, 0.08, color="k", lw=1)
        ax.text(t, -0.2, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(-0.62, lo, lo + cd, color="k", lw=2)
    ax.text(lo + cd / 2, -0.7, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (v, nm) in enumerate(zip(r.to_numpy(), names)):
        left = i < half
        row = i if left else K - 1 - i
        y = 0.55 + 0.42 * row
        x_text = lo - 0.25 if left else hi + 0.25
        ax.plot([v, v], [0, y], color="0.35", lw=0.8)
        ax.plot([v, x_text], [y, y], color="0.35", lw=0.8)
        ax.text(x_text, y, f"{nm} ({v:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    # bars joining groups of methods that are not significantly different
    vals = r.to_numpy()
    bars = []
    for i in range(K):
        j = i
        while j + 1 < K and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in bars):
            bars.append((i, j))
    for b, (i, j) in enumerate(bars):
        yb = 0.14 + 0.09 * (b % 4)
        ax.hlines(yb, vals[i] - 0.04, vals[j] + 0.04, color="k", lw=2.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.axis("off")
