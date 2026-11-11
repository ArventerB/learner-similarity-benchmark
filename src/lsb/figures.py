"""Figures for the manuscript (PNG at 300 dpi and PDF), drawn from results/.

    python -I src/lsb/figures.py

Colour follows the entity: each highlighted measure keeps one hue in every figure;
context series are grey. The eight hues are a validated colour-blind-safe order
(checked with a palette validator: lightness band, chroma floor, adjacent-pair separation
under simulated colour-vision deficiency). Every figure has its numbers in results/tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import (ALPHA_PATH, KNN, LEARNED, MODELS, RAW, RES, TAB, cell_features,  # noqa: E402
                      cell_table, load)
from metrics import FAMILY, LABEL  # noqa: E402
from stats import average_ranks, cd_diagram, cluster_bootstrap_diff, nemenyi_cd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = RES / "figures"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COL = {"euclid": "#2a78d6", "cosine": "#eb6834", "maha_lw": "#1baf7a", "maha_sample": "#eda100",
       "nca": "#e87ba4", "mlkr": "#008300", "lmnn": "#6250d6", "ridge": "#e34948"}
MARK = {"euclid": "o", "cosine": "s", "maha_lw": "^", "maha_sample": "v", "nca": "D", "mlkr": "P",
        "lmnn": "X", "ridge": "*"}
MSIZE = {"ridge": 8.5, "mlkr": 6.5, "lmnn": 6.5}
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal blue ramp for n = 50..500
SHORT = {**LABEL, "maha_sample": "Mahalanobis (sample Σ)", "maha_lw": "Mahalanobis (LW)",
         "ridge": "Ridge / logistic model", "itml": "ITML", "itml_lw": "ITML (LW prior)",
         "euclid_raw": "Euclidean (raw)", "cosine_raw": "Cosine (raw)"}
VIEW_LABEL = {"assist09": "ASSIST09", "assist15": "ASSIST15", "assist17": "ASSIST17", "ednet": "EdNet",
              "slepemapy": "Geography", "xes3g5m": "XES3G5M", "pta_comp_thinking": "PTA comp. thinking",
              "pta_database": "PTA database", "pta_linux": "PTA Linux", "pta_x_c": "PTA → C",
              "pta_x_cpp": "PTA → C++", "pta_x_java": "PTA → Java", "pta_x_python": "PTA → Python",
              "pta_x_ds": "PTA → data struct.", "oulad_BBB": "OULAD BBB", "oulad_CCC": "OULAD CCC",
              "oulad_DDD": "OULAD DDD", "oulad_EEE": "OULAD EEE", "oulad_FFF": "OULAD FFF",
              "oulad_GGG": "OULAD GGG"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 5.5, "legend.frameon": False,
    "legend.fontsize": 8.5, "axes.axisbelow": True, "pdf.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name, flush=True)


def line(ax, x, y, m, **kw):
    ax.plot(x, y, color=COL[m], marker=MARK[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF,
            markeredgewidth=0.8, label=SHORT[m], **kw)


def plain_log_x(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


# --------------------------------------------------------------------------- figure 1
def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, body, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc="#e6f0fd" if accent else SURF, ec=COL["euclid"] if accent else AXIS,
                                    lw=1.3 if accent else 0.9))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=8.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 7.0, body, ha="center", va="top", fontsize=7.4, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                     color=MUTED, shrinkA=0, shrinkB=0))

    top, h1 = 24.5, 16.5
    xs = [0.5, 25.5, 50.5, 75.5]
    w = 21.5
    box(xs[0], top, w, h1, "Interaction logs", "8 public sources\n20 views, 4 tasks")
    box(xs[1], top, w, h1, "Learner partition", "reference pool 20 %\ncandidate pool\nfixed test set")
    box(xs[2], top, w, h1, "Ability vector", "d most practised KCs,\nsmoothed success rate\n(defined on reference)")
    box(xs[3], top, w, h1, "Draw n learners", "n = 50, 100, 200, 500\nfrom the candidate pool,\nrepeated R times", accent=True)
    for i in range(3):
        arrow(xs[i] + w + 0.5, top + 3.0, xs[i + 1] - 0.5, top + 3.0)
    h2 = 16.5
    box(xs[0], 0.5, w, h2, "Fit on training only", "support filter, z-score,\ncovariance estimate,\nlearned metric")
    box(xs[1], 0.5, w, h2, "Retrieve neighbours", "k nearest training\nlearners for every\ntest learner")
    box(xs[2], 0.5, w, h2, "Predict and score", "mean outcome of the\nneighbours; R² and\nneighbour precision")
    box(xs[3], 0.5, w, h2, "Diagnose", "n/d, shrinkage weight,\neffective rank, hubness,\nsignal–variance alignment")
    for i in range(3):
        arrow(xs[i] + w + 0.5, 3.5, xs[i + 1] - 0.5, 3.5)
    # connector from the end of the first row to the start of the second row, routed in the gap
    x_end, y_gap = xs[3] + w / 2, (top + 0.5 + h2) / 2 + 0.1
    ax.plot([x_end, x_end, xs[0] + w / 2], [top - 0.3, y_gap, y_gap], color=MUTED, lw=0.9, solid_capstyle="round")
    arrow(xs[0] + w / 2, y_gap, xs[0] + w / 2, 0.5 + h2 + 0.4)
    save(fig, "fig1_protocol")


# --------------------------------------------------------------------------- figure 2
def fig_learning_curves(C):
    key = ["euclid", "cosine", "maha_lw", "nca", "mlkr", "ridge"]
    views = [v for v in VIEW_LABEL if v in set(C.index.get_level_values("base"))]
    ncol = 5
    nrow = int(np.ceil(len(views) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.2, 2.0 * nrow + 0.5))
    axes = axes.ravel()
    for ax, v in zip(axes, views):
        S = C[C.index.get_level_values("base") == v]
        ds = sorted(S.index.get_level_values("d").unique())
        d = 50 if 50 in ds else ds[0]
        S = S.xs(d, level="d")
        n = S.index.get_level_values("n").to_numpy()
        for m in key:
            line(ax, n, S[m].to_numpy(), m)
        ax.axhline(0, color=AXIS, lw=0.8)
        plain_log_x(ax, [50, 100, 200, 500])
        ax.set_xlim(42, 600)
        ax.set_title(f"{VIEW_LABEL[v]}  (d = {d})", loc="left", fontsize=8.6)
        lo = min(-0.02, np.nanmin(S[key].to_numpy()))
        ax.set_ylim(max(lo, -0.25) - 0.01, np.nanmax(S[key].to_numpy()) * 1.12 + 0.01)
    for ax in axes[len(views):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(views)]):
        if i >= len(views) - ncol:
            ax.set_xlabel("training learners n")
        if i % ncol == 0:
            ax.set_ylabel("R² on test learners")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(key), bbox_to_anchor=(0.5, 1.0), fontsize=8.8,
               columnspacing=1.6, handlelength=2.2)
    fig.tight_layout(h_pad=0.9, w_pad=0.7, rect=(0, 0, 1, 0.965))
    save(fig, "fig2_learning_curves")


# --------------------------------------------------------------------------- figure 3
def fig_ranks(C):
    present = [m for m in KNN if m in C.columns]
    n_lvl = C.index.get_level_values("n")
    ns = sorted(n_lvl.unique())
    R = pd.DataFrame({n: average_ranks(C.loc[n_lvl == n, present].dropna()) for n in ns})
    fig = plt.figure(figsize=(8.6, 10.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.92, 0.92], hspace=0.33)
    ax = fig.add_subplot(gs[0])
    xs = np.arange(len(ns))
    hi = [m for m in ["cosine", "euclid", "maha_lw", "maha_sample", "nca", "mlkr", "lmnn"] if m in present]
    for m in present:
        if m not in hi:
            ax.plot(xs, R.loc[m], color=AXIS, lw=1.1, marker="o", ms=3, zorder=1)
    for m in hi:
        ax.plot(xs, R.loc[m], color=COL[m], marker=MARK[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF,
                markeredgewidth=0.8, zorder=3)
    end = R[ns[-1]].sort_values()                       # direct labels, spread so they never overlap
    ypos = end.to_numpy().copy()
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + 0.66)
    for (m, y0), y in zip(end.items(), ypos):
        ax.plot([xs[-1] + 0.03, xs[-1] + 0.16], [y0, y], color=COL.get(m, AXIS), lw=0.7, clip_on=False)
        ax.text(xs[-1] + 0.18, y, SHORT[m], va="center", fontsize=7.8, color=INK if m in hi else INK2, clip_on=False)
    ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlim(-0.15, len(ns) - 1 + 0.02)
    ax.set_ylim(max(ypos.max(), R.to_numpy().max()) + 0.5, R.to_numpy().min() - 0.5)
    ax.set_ylabel("average rank (1 = best)"); ax.set_xlabel("training learners n")
    ax.set_title("(a) Average rank of the 18 similarity measures across cells, by training size", loc="left")
    ax.grid(axis="x", visible=False)
    for j, n in enumerate([ns[0], ns[-1]]):
        axc = fig.add_subplot(gs[1 + j])
        S = C.loc[n_lvl == n, present].dropna()
        cd = nemenyi_cd(S.shape[1], S.shape[0])
        cd_diagram(axc, average_ranks(S), cd, SHORT,
                   title=f"({'bc'[j]}) Critical-difference diagram, n = {n}  ({S.shape[0]} cells)")
    save(fig, "fig3_ranks_cd")


# --------------------------------------------------------------------------- figure 4
def fig_forest(C):
    """Mean R² difference to the Euclidean distance, by method and training size."""
    present = [m for m in KNN + MODELS if m in C.columns and m != "euclid"]
    n_lvl = C.index.get_level_values("n")
    ns = sorted(n_lvl.unique())
    src = C.index.get_level_values("src").to_numpy()
    rec = {}
    for m in present:
        for n in ns:
            sel = (n_lvl == n) & C[m].notna().to_numpy()
            d = (C.loc[sel, m] - C.loc[sel, "euclid"]).to_numpy()
            rec[(m, n)] = cluster_bootstrap_diff(d, src[sel], n_boot=2000)
    order = sorted(present, key=lambda m: np.mean([rec[(m, n)][0] for n in ns]))
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(order) + 1.3))
    off = np.linspace(-0.27, 0.27, len(ns))
    for i, m in enumerate(order):
        for j, n in enumerate(ns):
            mean, lo, hi = rec[(m, n)]
            ax.plot([lo, hi], [i + off[j]] * 2, color=SEQ[j], lw=1.3, solid_capstyle="round")
            ax.plot(mean, i + off[j], "o", color=SEQ[j], ms=4.6, markeredgecolor=SURF, markeredgewidth=0.6,
                    label=f"n = {n}" if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[m] for m in order])
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(INK)
        if FAMILY[m] == "model":
            t.set_fontstyle("italic")
    ax.set_xlabel("mean R² difference to the Euclidean distance (95 % CI, sources resampled)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="training learners", title_fontsize=8.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    save(fig, "fig4_forest_vs_euclid")
    out = pd.DataFrame([dict(method=LABEL[m], n=n, mean=rec[(m, n)][0], ci_lo=rec[(m, n)][1], ci_hi=rec[(m, n)][2])
                        for m in order for n in ns])
    out.to_csv(TAB / "fig4_data.csv", index=False)


# --------------------------------------------------------------------------- figure 5
def fig_alpha_mechanism(C, F):
    ap = C[[m for m in ALPHA_PATH]].dropna()
    rel = ap.sub(ap["euclid"], axis=0)
    alphas = [ALPHA_PATH[m] for m in rel.columns]
    order = np.argsort(alphas)
    alphas = np.array(alphas)[order]
    rel = rel.iloc[:, order]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.36))
    ax = axes[0]
    for j, (n, g) in enumerate(rel.groupby(level="n")):
        ax.plot(alphas, g.mean().to_numpy(), color=SEQ[j], marker="o", markeredgecolor=SURF, markeredgewidth=0.7,
                label=f"n = {n}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("shrinkage weight α on the identity\n(0 = sample Mahalanobis, 1 = Euclidean)")
    ax.set_ylabel("mean R² difference to Euclidean")
    ax.set_title("(a) Path from Mahalanobis to Euclidean", loc="left")
    ax.legend(loc="upper left", title="training learners", title_fontsize=8.5)
    ax = axes[1]
    sva = F["sva"].reindex(rel.index)
    cuts = np.nanquantile(sva, [1 / 3, 2 / 3])
    grp = np.digitize(sva, cuts)
    names = [f"low (≤ {cuts[0]:.1f})", f"middle", f"high (> {cuts[1]:.1f})"]
    for gi in range(3):
        ax.plot(alphas, rel[grp == gi].mean().to_numpy(), color=[SEQ[0], SEQ[1], SEQ[3]][gi], marker="o",
                markeredgecolor=SURF, markeredgewidth=0.7, label=names[gi])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("shrinkage weight α on the identity")
    ax.set_title("(b) The same path, by alignment tercile", loc="left")
    ax.legend(loc="upper left", title="signal–variance alignment", title_fontsize=8.5)
    ax = axes[2]
    d = (C["maha_lw"] - C["euclid"]).reindex(F.index)
    ok = d.notna() & F["sva"].notna()
    ax.scatter(F.loc[ok, "sva"], d[ok], s=17, color=COL["euclid"], edgecolor=SURF, linewidth=0.5, alpha=0.9)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1, color=AXIS, lw=0.8)
    lo_x, hi_x = F.loc[ok, "sva"].min(), F.loc[ok, "sva"].max()
    plain_log_x(ax, [t for t in [0.5, 1, 2, 5, 10, 20, 50] if lo_x / 1.6 <= t <= hi_x * 1.6])
    rho = sps.spearmanr(F.loc[ok, "sva"], d[ok]).statistic
    ax.text(0.04, 0.05, f"Spearman ρ = {rho:.2f}  ({int(ok.sum())} cells)", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel("signal–variance alignment (log scale)\n> 1: outcome carried by high-variance directions")
    ax.set_ylabel("R²: Mahalanobis (LW) − Euclidean")
    ax.set_title("(c) Whitening effect against alignment, per cell", loc="left")
    save(fig, "fig6_alpha_mechanism")


# --------------------------------------------------------------------------- figure 6
def fig_learned(C):
    n_lvl = C.index.get_level_values("n")
    ns = sorted(n_lvl.unique())
    src = C.index.get_level_values("src").to_numpy()
    Cx = C.copy()
    Cx["best_learned"] = C[[m for m in LEARNED if m in C.columns]].max(axis=1)
    rows = [m for m in LEARNED if m in C.columns] + ["best_learned"]
    names = {**SHORT, "best_learned": "Best learned metric per cell (oracle)"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.5))
    ax = axes[0]
    off = np.linspace(-0.27, 0.27, len(ns))
    for i, m in enumerate(rows):
        for j, n in enumerate(ns):
            sel = (n_lvl == n) & Cx[m].notna().to_numpy()
            mean, lo, hi = cluster_bootstrap_diff((Cx.loc[sel, m] - Cx.loc[sel, "cosine"]).to_numpy(), src[sel], n_boot=2000)
            ax.plot([lo, hi], [i + off[j]] * 2, color=SEQ[j], lw=1.3, solid_capstyle="round")
            ax.plot(mean, i + off[j], "o", color=SEQ[j], ms=4.6, markeredgecolor=SURF, markeredgewidth=0.6,
                    label=f"n = {n}" if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[m] for m in rows])
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean R² difference to cosine (95 % CI, sources resampled)")
    ax.set_title("(a) Learned metrics against the cosine measure", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, title="training learners", title_fontsize=8.5,
              columnspacing=1.2, handletextpad=0.3)
    ax = axes[1]
    nd = (C.index.get_level_values("n") / C.index.get_level_values("d")).to_numpy()
    d = (Cx["best_learned"] - Cx["cosine"]).to_numpy()
    ax.scatter(nd, d, s=17, color=COL["euclid"], edgecolor=SURF, linewidth=0.5, alpha=0.9)
    ax.axhline(0, color=INK2, lw=0.8)
    plain_log_x(ax, [0.5, 1, 2, 5, 10, 25])
    ax.set_xlabel("learners per feature, n / d (log scale)")
    ax.set_ylabel("R²: best learned metric − cosine")
    rho = sps.spearmanr(nd, d, nan_policy="omit").statistic
    ax.text(0.04, 0.05, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_title("(b) Best learned metric against cosine, per cell", loc="left")
    save(fig, "fig5_learned_metrics")


# --------------------------------------------------------------------------- figure 7
def fig_protocol_effects(C, Call):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw=dict(wspace=0.46))
    n_lvl = C.index.get_level_values("n")
    ns = sorted(n_lvl.unique())
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=7.8, columnspacing=1.0, handletextpad=0.4)
    # (a) transductive preprocessing
    ax = axes[0]
    w = 0.36
    for j, (m, lab) in enumerate([("maha_lw", "Mahalanobis (LW)"), ("maha_sample", "Mahalanobis (sample Σ)")]):
        d = (C[f"{m}_trans"] - C[m]).groupby(level="n").mean()
        ax.bar(np.arange(len(ns)) + (j - 0.5) * w, d.to_numpy(), width=w - 0.05, color=COL[m], label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([str(n) for n in ns]); ax.grid(axis="x", visible=False)
    ax.set_xlabel("training learners n"); ax.set_ylabel("R² change when test learners are\nused for scaling and covariance")
    ax.set_title("(a) Transductive preprocessing", loc="left")
    ax.legend(ncol=1, **below)
    # (b) random vs temporal split
    ax = axes[1]
    leak = Call[Call.index.get_level_values("base").str.contains("randsplit")]
    recs = []
    for (view, base, src, task, d, n), r in leak.iterrows():
        try:
            t = Call.xs((view.replace("_randsplit", ""), n), level=("view", "n")).iloc[0]
        except KeyError:
            continue
        recs.append(dict(src=src, temporal=t["ridge"], random=r["ridge"], tc=t["cosine"], rc=r["cosine"]))
    rec = pd.DataFrame(recs).groupby("src").mean()
    for i, (s, r) in enumerate(rec.iterrows()):
        for y, a, b, m in [(i - 0.15, r["temporal"], r["random"], "ridge"), (i + 0.15, r["tc"], r["rc"], "cosine")]:
            ax.plot([a, b], [y, y], color=AXIS, lw=2, zorder=1)
            ax.plot(a, y, marker=MARK[m], color=COL[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF, markeredgewidth=0.6, alpha=0.4)
            ax.plot(b, y, marker=MARK[m], color=COL[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF, markeredgewidth=0.6)
    ax.set_yticks(range(len(rec))); ax.set_yticklabels(rec.index)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_ylim(len(rec) - 0.5, -0.5); ax.grid(axis="y", visible=False)
    ax.plot([], [], marker=MARK["ridge"], color=COL["ridge"], ls="", ms=8.5, label="Ridge model")
    ax.plot([], [], marker=MARK["cosine"], color=COL["cosine"], ls="", ms=5.5, label="Cosine k-NN")
    ax.legend(ncol=2, **below)
    ax.set_xlabel("mean R² (faded: future outcome;\nsolid: randomly drawn outcome)")
    ax.set_title("(b) Random instead of temporal split", loc="left")
    # (c) leave-one-module-out
    ax = axes[2]
    if (TAB / "protocol_lomo.csv").exists():
        lomo = pd.read_csv(TAB / "protocol_lomo.csv")
        for m, nm in [("ridge", "Logistic model"), ("cosine", "Cosine k-NN")]:
            g = lomo[lomo["method"] == LABEL[m]]
            x = range(len(g))
            ax.plot(x, g["auc_within"], color=COL[m], marker=MARK[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF,
                    markeredgewidth=0.7, label=f"{nm}, same module")
            ax.plot(x, g["auc_lomo"], color=COL[m], marker=MARK[m], ms=MSIZE.get(m, 5.5), markeredgecolor=SURF,
                    markeredgewidth=0.7, alpha=0.4, label=f"{nm}, other modules")
        ax.set_xticks(list(x)); ax.set_xticklabels([str(int(n)) for n in g["n"]]); ax.grid(axis="x", visible=False)
        ax.set_xlabel("training learners n"); ax.set_ylabel("AUC on the held-out module")
        ax.legend(ncol=1, **below)
    ax.set_title("(c) OULAD: training on other modules", loc="left")
    save(fig, "fig8_protocol_effects")


# --------------------------------------------------------------------------- figure 8
def fig_selector():
    T = pd.read_csv(TAB / "selector.csv")
    keep = ["depth-2 rule (LOSO)", "depth-1 rule (LOSO)", "best single (LOSO)",
            "internal LOO among label-free measures",
            "internal LOO among all measures (resubstitution for learned)",
            "tuned shrinkage (LOO over the α path)"]
    fixed = T[T["strategy"].str.startswith("always")]
    show = ["always " + LABEL[m] for m in ["cosine", "pca80_proj", "euclid", "nca", "mlkr", "maha_lw", "maha_sample", "cosine_raw"]]
    S = pd.concat([T[T["strategy"].isin(keep)], fixed[fixed["strategy"].isin(show)]]).sort_values("mean_regret")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.08))
    ax = axes[0]
    y = np.arange(len(S))
    is_rule = ~S["strategy"].str.startswith("always")
    ax.barh(y, S["mean_regret"], height=0.62, color=[COL["euclid"] if r else AXIS for r in is_rule])
    for yi, v in zip(y, S["mean_regret"]):
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=7.8, color=INK2)
    lab = [s.replace("internal LOO among all measures (resubstitution for learned)", "LOO pick among all measures*")
           .replace("internal LOO among label-free measures", "LOO pick among label-free measures")
           .replace("tuned shrinkage (LOO over the α path)", "LOO-tuned shrinkage α") for s in S["strategy"]]
    ax.set_yticks(y); ax.set_yticklabels(lab)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, S["mean_regret"].max() * 1.16)
    ax.set_xlabel("mean regret in R² against the best measure of each cell")
    ax.set_title("(a) Selection strategies (blue: data-dependent; grey: one fixed measure)", loc="left")
    ax = axes[1]
    W = pd.read_csv(TAB / "winner_family_by_n.csv").set_index("n")
    fams = [f for f in ["fixed", "covariance", "learned"] if f in W.columns]
    cols = {"fixed": COL["euclid"], "covariance": COL["cosine"], "learned": COL["maha_lw"]}
    names = {"fixed": "fixed measure", "covariance": "covariance-aware", "learned": "learned metric"}
    left = np.zeros(len(W))
    for f in fams:
        ax.barh(range(len(W)), W[f], left=left, height=0.56, color=cols[f], label=names[f], edgecolor=SURF, linewidth=1.5)
        for i, (v, l0) in enumerate(zip(W[f], left)):
            if v >= 0.09:
                ax.text(l0 + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="#ffffff")
        left += W[f].to_numpy()
    ax.set_yticks(range(len(W))); ax.set_yticklabels([f"n = {int(n)}" for n in W.index]); ax.yaxis.tick_right()
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(visible=False)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0 %", "50 %", "100 %"])
    ax.set_xlabel("share of cells in which the best measure\nbelongs to the family")
    ax.set_title("(b) Family of the best measure", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.02), ncol=3, fontsize=8, columnspacing=1.0,
              handlelength=1.2, borderaxespad=0.0)
    ax.set_ylim(len(W) - 0.5, -0.95)
    save(fig, "fig7_selector")


def main():
    rows, diag = load(RAW)
    main_rows, main_diag = rows[~rows["leaky"]], diag[~diag["leaky"]]
    C, Call, F = cell_table(main_rows), cell_table(rows), cell_features(main_diag)
    json.loads((RES / "summary.json").read_text(encoding="utf-8"))   # fail early if the analysis has not run
    fig_protocol()
    fig_learning_curves(C)
    fig_ranks(C)
    fig_forest(C)
    fig_learned(C)
    fig_alpha_mechanism(C, F)
    fig_selector()
    fig_protocol_effects(C, Call)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
