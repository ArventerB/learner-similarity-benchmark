"""Tables, statistical tests, mechanism analysis and the selection rule.

Reads results/raw (main grid) and results/lomo (OULAD leave-one-module-out) and writes
CSV tables to results/tables plus results/summary.json, the single source of every number
quoted in the manuscript.

    python -I src/lsb/analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import FAMILY, LABEL  # noqa: E402
from stats import (average_ranks, bayes_signed_rank, cluster_bootstrap_diff, friedman,  # noqa: E402
                   holm, nemenyi_cd, wilcoxon)

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("LSB_RESULTS", ROOT / "results"))   # override to analyse another run
RAW, LOMO = RES / "raw", RES / "lomo"
TAB, PROC = RES / "tables", ROOT / "data" / "processed"

FIXED = ["euclid", "manhattan", "cosine", "pearson", "euclid_raw", "cosine_raw"]
COV = ["maha_sample", "maha_lw", "maha_oas", "pca95_whiten", "pca80_proj"]
LEARNED = ["lmnn", "nca", "itml", "itml_lw", "sdml", "lfda", "mlkr"]
KNN = FIXED + COV + LEARNED
MODELS = ["ridge", "rf", "gbdt"]
ALPHA_PATH = {"maha_sample": 0.0, "maha_a10": 0.10, "maha_a25": 0.25, "maha_a50": 0.50,
              "maha_a75": 0.75, "maha_a90": 0.90, "euclid": 1.0}
SRC = {"PTADisc cross-course": "PTADisc", "PTADisc single-course": "PTADisc",
       "EdNet-KT1 (20k sample)": "EdNet", "Slepemapy geography": "Slepemapy",
       "ASSISTments 2009": "ASSIST09", "ASSISTments 2015": "ASSIST15",
       "ASSISTments 2017": "ASSIST17", "XES3G5M": "XES3G5M", "OULAD": "OULAD"}
ROPE = 0.01
FEATS = ["log_n_over_d", "sva", "lw_alpha", "erank_ratio", "top_eig_share", "mean_abs_corr",
         "density", "const_share", "hub_skew_train", "id_ratio"]


# ----------------------------------------------------------------------------- loading
def load(folder: Path):
    files = sorted(glob.glob(str(folder / "*.parquet")))
    rows = pd.concat([pd.read_parquet(f) for f in files if ".diag." not in f], ignore_index=True)
    diag = pd.concat([pd.read_parquet(f) for f in files if ".diag." in f], ignore_index=True)
    # Use only the repetitions that every cell has completed, so that a partly finished
    # block of the runner never unbalances the design.
    common = int(rows.groupby(["view", "n"])["rep"].nunique().min())
    rows, diag = rows[rows["rep"] < common].copy(), diag[diag["rep"] < common].copy()
    for t in (rows, diag):
        t["src"] = t["source"].map(SRC)
        t["base"] = t["view"].str.replace(r"__d\d+$", "", regex=True)
        t["leaky"] = t["view"].str.contains("randsplit")
    return rows, diag


def cell_table(rows: pd.DataFrame, k: int = 10, value: str = "r2") -> pd.DataFrame:
    """Mean over repetitions: one row per cell (view, n), one column per method."""
    P = rows[rows["k"].isin([k, 0])]
    return P.groupby(["view", "base", "src", "task", "d", "n", "method"])[value].mean().unstack("method")


def cell_features(diag: pd.DataFrame) -> pd.DataFrame:
    idx = pd.read_csv(PROC / "views_index.csv").set_index("name")
    g = diag.replace([np.inf, -np.inf], np.nan).groupby(["view", "base", "src", "task", "d", "n"])
    F = g[["lw_alpha", "oas_alpha", "erank", "part_ratio", "mean_abs_corr", "dist_cv", "id_twonn",
           "hub_skew_train", "n_const", "cov_rank", "sva", "rho2_pc1", "top_eig_share"]].mean()
    d = F.index.get_level_values("d").to_numpy().astype(float)
    n = F.index.get_level_values("n").to_numpy().astype(float)
    F["n_over_d"] = n / d
    F["log_n_over_d"] = np.log10(n / d)
    F["erank_ratio"] = F["erank"] / d
    F["id_ratio"] = F["id_twonn"] / d
    F["const_share"] = F["n_const"] / d
    F["density"] = idx.reindex(F.index.get_level_values("view"))["feature_density"].to_numpy()
    return F


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Small Markdown table writer (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(df: pd.DataFrame, name: str, floatfmt: str = "{:.3f}", index: bool = False):
    TAB.mkdir(parents=True, exist_ok=True)
    d = df.reset_index() if index else df
    d.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(md(d, floatfmt), encoding="utf-8")


# -------------------------------------------------------------------- selection rules
def fit_tree(R: np.ndarray, X: np.ndarray, feats: list[str], depth: int, min_leaf: int = 8):
    """Greedy regret-minimising tree. R: cells x methods regret; X: cells x features."""
    best_m = int(np.argmin(R.sum(axis=0)))
    node = dict(leaf=True, method=best_m, regret=float(R[:, best_m].sum()), n=len(R))
    if depth == 0 or len(R) < 2 * min_leaf:
        return node
    best = None
    for j in range(X.shape[1]):
        xs = X[:, j]
        for t in np.unique(np.nanquantile(xs, np.linspace(0.1, 0.9, 17))):
            left = xs < t
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            tot = R[left].sum(axis=0).min() + R[~left].sum(axis=0).min()
            if best is None or tot < best[0] - 1e-12:
                best = (tot, j, float(t), left)
    if best is None or best[0] >= node["regret"] - 1e-9:
        return node
    _, j, t, left = best
    return dict(leaf=False, feature=feats[j], j=j, thr=t, n=len(R),
                lo=fit_tree(R[left], X[left], feats, depth - 1, min_leaf),
                hi=fit_tree(R[~left], X[~left], feats, depth - 1, min_leaf))


def predict_tree(node: dict, x: np.ndarray) -> int:
    while not node["leaf"]:
        node = node["lo"] if x[node["j"]] < node["thr"] else node["hi"]
    return node["method"]


def tree_text(node: dict, methods: list[str], indent: str = "") -> str:
    if node["leaf"]:
        return f"{indent}use {LABEL.get(methods[node['method']], methods[node['method']])}  [{node['n']} cells]\n"
    return (f"{indent}if {node['feature']} < {node['thr']:.3g}:\n" + tree_text(node["lo"], methods, indent + "    ")
            + f"{indent}else:\n" + tree_text(node["hi"], methods, indent + "    "))


def selector_analysis(C: pd.DataFrame, F: pd.DataFrame, rows: pd.DataFrame, summary: dict):
    port = [m for m in KNN if m in C.columns]
    M = C[port].dropna()
    F = F.reindex(M.index)
    oracle = M.max(axis=1)
    R = (oracle.to_numpy()[:, None] - M.to_numpy())
    X = F[FEATS].to_numpy()
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    src = M.index.get_level_values("src").to_numpy()
    n_lvl = M.index.get_level_values("n").to_numpy()

    out = {}                                           # strategy -> regret per cell
    for m in port:
        out[f"fixed:{m}"] = R[:, port.index(m)]

    # leave-one-source-out rules
    loso = {"best single (LOSO)": np.zeros(len(M)), "depth-1 rule (LOSO)": np.zeros(len(M)),
            "depth-2 rule (LOSO)": np.zeros(len(M))}
    chosen_single = {}
    for s in np.unique(src):
        tr, te = src != s, src == s
        j = int(np.argmin(R[tr].sum(axis=0)))
        chosen_single[s] = port[j]
        loso["best single (LOSO)"][te] = R[te, j]
        for depth, key in [(1, "depth-1 rule (LOSO)"), (2, "depth-2 rule (LOSO)")]:
            tree = fit_tree(R[tr], X[tr], FEATS, depth)
            pick = np.array([predict_tree(tree, x) for x in X[te]])
            loso[key][te] = R[te, pick][np.arange(te.sum()), np.arange(te.sum())] if False else R[te][np.arange(te.sum()), pick]
    out.update(loso)

    # data-driven selection inside each training sample (leave-one-out k-NN skill)
    P = rows[(rows["k"] == 10) & ~rows["leaky"]]
    sets = {"internal LOO among label-free measures": FIXED + COV,
            "internal LOO among all measures (resubstitution for learned)": port,
            "tuned shrinkage (LOO over the α path)": list(ALPHA_PATH)}
    for name, cand in sets.items():
        sub = P[P["method"].isin(cand)]
        pick = sub.loc[sub.groupby(["view", "n", "rep"])["loo_r2"].idxmax()]
        val = pick.groupby(["view", "n"])["r2"].mean()
        key = list(zip(M.index.get_level_values("view"), M.index.get_level_values("n")))
        out[name] = oracle.to_numpy() - val.reindex(key).to_numpy()
        if name.startswith("tuned"):
            a = pick.assign(alpha=pick["method"].map(ALPHA_PATH)).groupby(["view", "n"])["alpha"].mean()
            F["alpha_loo"] = a.reindex(key).to_numpy()

    tab = []
    for name, r in out.items():
        row = dict(strategy=name, mean_regret=float(np.nanmean(r)), median_regret=float(np.nanmedian(r)),
                   p90_regret=float(np.nanquantile(r, 0.9)), within_0p01=float(np.nanmean(r <= 0.01)))
        for n in sorted(np.unique(n_lvl)):
            row[f"n={n}"] = float(np.nanmean(r[n_lvl == n]))
        tab.append(row)
    T = pd.DataFrame(tab).sort_values("mean_regret").reset_index(drop=True)
    T["strategy"] = T["strategy"].apply(lambda s: "always " + LABEL.get(s[6:], s[6:]) if s.startswith("fixed:") else s)
    save(T, "selector", "{:.4f}")

    # per held-out source
    per = []
    for s in np.unique(src):
        m = src == s
        per.append(dict(source=s, cells=int(m.sum()), best_single_chosen=LABEL[chosen_single[s]],
                        best_single=float(loso["best single (LOSO)"][m].mean()),
                        depth1=float(loso["depth-1 rule (LOSO)"][m].mean()),
                        depth2=float(loso["depth-2 rule (LOSO)"][m].mean()),
                        always_cosine=float(out["fixed:cosine"][m].mean()),
                        always_euclid=float(out["fixed:euclid"][m].mean()),
                        internal_loo=float(np.nanmean(out["internal LOO among label-free measures"][m]))))
    save(pd.DataFrame(per), "selector_by_source", "{:.4f}")

    # the rule fitted on all sources (what a practitioner would take away)
    full1, full2 = fit_tree(R, X, FEATS, 1), fit_tree(R, X, FEATS, 2)
    (TAB / "selector_rule.txt").write_text(
        "Depth-1 rule fitted on all cells\n" + tree_text(full1, port) + "\nDepth-2 rule fitted on all cells\n"
        + tree_text(full2, port), encoding="utf-8")
    best_cell = pd.Series(np.array(port)[np.argmax(M.to_numpy(), axis=1)], index=M.index)
    wins = best_cell.map(FAMILY).groupby(n_lvl).value_counts(normalize=True).unstack(fill_value=0)
    save(wins.rename_axis("n"), "winner_family_by_n", "{:.3f}", index=True)
    bw = best_cell.map(LABEL).value_counts().rename_axis("measure").reset_index(name="cells_won")
    save(bw, "winner_counts", "{:.0f}")
    summary["selector_by_n"] = {r["strategy"]: {int(n): round(float(r[f"n={n}"]), 4) for n in sorted(np.unique(n_lvl))}
                                for _, r in T.iterrows()}
    summary["wins"] = {m: int((best_cell == m).sum()) for m in port}
    summary["win_share"] = {m: round(float((best_cell == m).mean()), 3) for m in port}
    summary["winner_family"] = {int(n): {f: round(float(wins.loc[n, f]), 3) if f in wins.columns else 0.0
                                         for f in ["fixed", "covariance", "learned"]} for n in wins.index}
    lw = {}
    for n in sorted(np.unique(n_lvl)):
        w = best_cell[(n_lvl == n)]
        w = w[w.map(FAMILY) == "learned"].map(LABEL).value_counts()
        lw[int(n)] = ", ".join(f"{k} in {v} cell{'s' if v > 1 else ''}" for k, v in w.items()) or "none"
    summary["learned_winners"] = lw
    summary["loso_best_single"] = chosen_single
    summary["loso_best_single_all_cosine"] = bool(all(v == "cosine" for v in chosen_single.values()))
    summary["selector"] = {r["strategy"]: round(r["mean_regret"], 4) for _, r in T.iterrows()}
    summary["selector_within"] = {r["strategy"]: round(r["within_0p01"], 3) for _, r in T.iterrows()}
    summary["rule_depth1"] = tree_text(full1, port)
    summary["rule_depth2"] = tree_text(full2, port)
    summary["n_cells_selector"] = int(len(M))
    return F, pd.Series(oracle, index=M.index), best_cell


# ------------------------------------------------------------------------------- main
def main():
    import warnings

    warnings.filterwarnings("ignore", category=sps.ConstantInputWarning)
    TAB.mkdir(parents=True, exist_ok=True)
    rows, diag = load(RAW)
    summary: dict = {}
    main_rows, main_diag = rows[~rows["leaky"]], diag[~diag["leaky"]]
    C = cell_table(main_rows)
    F = cell_features(main_diag)
    reps = main_rows.groupby(["view", "n"])["rep"].nunique()
    summary["reps_min"], summary["reps_max"] = int(reps.min()), int(reps.max())
    summary["n_cells"] = int(len(C))
    summary["n_views"] = int(C.index.get_level_values("base").nunique())
    summary["n_sources"] = int(C.index.get_level_values("src").nunique())
    summary["n_fits_knn"] = int((main_rows["k"] == 10).sum())
    n_lvl = C.index.get_level_values("n")
    srcs = C.index.get_level_values("src")
    present = [m for m in KNN if m in C.columns]

    # ---- datasets table
    idx = pd.read_csv(PROC / "views_index.csv")
    idx = idx[~idx["name"].str.contains("randsplit")]
    ds = idx.groupby(["source", "view", "domain", "task"]).agg(
        learners=("n_learners", "first"), pool=("n_pool", "first"), test=("n_test", "first"),
        d=("d", lambda s: "/".join(str(x) for x in sorted(s))),
        density=("feature_density", lambda s: f"{s.min():.2f}" if s.min() == s.max() else f"{s.min():.2f}–{s.max():.2f}"),
        outcome_mean=("y_mean", "first"), outcome_sd=("y_sd", "first")).reset_index()
    ds["task"] = ds["task"].map({"future": "future performance", "transfer": "cross-course transfer",
                                 "unseen": "unseen skills", "warning": "early warning"})
    save(ds.rename(columns={"source": "Source", "view": "View", "domain": "Domain", "task": "Task",
                            "learners": "Learners", "pool": "Pool", "test": "Test", "density": "Density",
                            "outcome_mean": "Outcome mean", "outcome_sd": "Outcome SD"}), "datasets")
    summary["learners_total_unique_views"] = int(ds.groupby("source")["learners"].max().sum())

    # ---- main results by n
    res = []
    for m in present + MODELS:
        row = dict(method=LABEL[m], family=FAMILY[m])
        for n in sorted(n_lvl.unique()):
            row[f"R2 n={n}"] = float(C.loc[n_lvl == n, m].mean())
        row["R2 all"] = float(C[m].mean())
        res.append(row)
    res = pd.DataFrame(res)
    ranks_n = {}
    for n in sorted(n_lvl.unique()):
        S = C.loc[n_lvl == n, present].dropna()
        ranks_n[n] = average_ranks(S)
        res[f"rank n={n}"] = [ranks_n[n].get(m, np.nan) for m in present] + [np.nan] * len(MODELS)
    res["rank all"] = list(average_ranks(C[present].dropna())) + [np.nan] * len(MODELS)
    save(res.rename(columns=lambda c: c.replace("R2 ", "R² ").replace("method", "Measure / model").replace("family", "Family")),
         "main_results")
    summary["mean_r2"] = {m: {int(n): round(float(C.loc[n_lvl == n, m].mean()), 4) for n in sorted(n_lvl.unique())}
                          for m in present + MODELS}
    summary["mean_r2_all"] = {m: round(float(C[m].mean()), 4) for m in present + MODELS}
    summary["avg_rank_all"] = {m: round(float(v), 2) for m, v in average_ranks(C[present].dropna()).items()}
    summary["avg_rank"] = {int(n): {m: round(float(v), 2) for m, v in r.items()} for n, r in ranks_n.items()}

    # ---- Friedman / Nemenyi per n (blocks = cells) and at source level
    fr = []
    for n in sorted(n_lvl.unique()):
        S = C.loc[n_lvl == n, present].dropna()
        f = friedman(S)
        f.update(level=f"cells, n={n}", CD=nemenyi_cd(f["K"], f["N"]))
        fr.append(f)
        S2 = S.groupby(level="src").mean()
        f2 = friedman(S2)
        f2.update(level=f"sources, n={n}", CD=nemenyi_cd(f2["K"], f2["N"]))
        fr.append(f2)
    S = C[present].dropna()
    f = friedman(S); f.update(level="cells, all n", CD=nemenyi_cd(f["K"], f["N"])); fr.append(f)
    fr = pd.DataFrame(fr)[["level", "N", "K", "chi2", "F", "p", "CD"]]
    save(fr, "friedman", "{:.4g}")
    summary["friedman"] = {r["level"]: dict(N=int(r["N"]), K=int(r["K"]), chi2=round(r["chi2"], 1),
                                            p=float(r["p"]), CD=round(r["CD"], 2)) for _, r in fr.iterrows()}

    # ---- pairwise contrasts
    Cx = C.copy()
    Cx["best_learned"] = C[[m for m in LEARNED if m in C.columns]].max(axis=1)
    Cx["best_fixed"] = C[[m for m in FIXED if m in C.columns]].max(axis=1)
    pairs = [("maha_lw", "maha_sample"), ("maha_lw", "euclid"), ("maha_sample", "euclid"),
             ("cosine", "euclid"), ("cosine", "cosine_raw"), ("pca80_proj", "euclid"),
             ("nca", "cosine"), ("mlkr", "cosine"), ("lmnn", "euclid"), ("best_learned", "cosine"),
             ("ridge", "cosine"), ("rf", "cosine"), ("ridge", "best_learned")]
    lab = {**LABEL, "best_learned": "best learned metric (oracle)", "best_fixed": "best fixed measure (oracle)"}
    con = []
    for n in list(sorted(n_lvl.unique())) + ["all"]:
        sel = np.ones(len(Cx), bool) if n == "all" else (n_lvl == n)
        block = []
        for a, b in pairs:
            S = Cx.loc[sel, [a, b]].dropna()
            dlt = (S[a] - S[b]).to_numpy()
            cl = S.index.get_level_values("src").to_numpy()
            mean, lo, hi = cluster_bootstrap_diff(dlt, cl)
            by_src = pd.Series(dlt).groupby(cl).mean()
            bs = bayes_signed_rank(S[a].to_numpy(), S[b].to_numpy(), rope=ROPE)
            block.append(dict(n=n, contrast=f"{lab[a]} − {lab[b]}", a=a, b=b, cells=len(S), mean_diff=mean,
                              ci_lo=lo, ci_hi=hi, wilcoxon_p=wilcoxon(S[a].to_numpy(), S[b].to_numpy()),
                              p_better=bs["p_better"], p_rope=bs["p_rope"], p_worse=bs["p_worse"],
                              sources_pos=f"{int((by_src > 0).sum())}/{len(by_src)}"))
        adj = holm(np.array([r["wilcoxon_p"] for r in block]))
        for r, p in zip(block, adj):
            r["holm_p"] = float(p)
        con += block
    con = pd.DataFrame(con)
    save(con.drop(columns=["a", "b"]), "contrasts", "{:.4g}")
    summary["contrasts"] = {f"{r['a']}-{r['b']}@{r['n']}": dict(
        diff=round(r["mean_diff"], 4), lo=round(r["ci_lo"], 4), hi=round(r["ci_hi"], 4), holm_p=float(r["holm_p"]),
        p_better=round(r["p_better"], 3), p_rope=round(r["p_rope"], 3), p_worse=round(r["p_worse"], 3),
        sources_pos=r["sources_pos"], cells=int(r["cells"])) for _, r in con.iterrows()}

    # ---- per-source summary (independent sources as blocks)
    key = ["euclid", "cosine", "cosine_raw", "maha_sample", "maha_lw", "pca80_proj", "nca", "mlkr", "ridge", "rf"]
    ps = C[key].groupby(level="src").mean()
    ps.insert(0, "cells", C.groupby(level="src").size())
    ps["best k-NN measure"] = C[present].groupby(level="src").mean().idxmax(axis=1).map(LABEL)
    save(ps.rename(columns=LABEL).rename_axis("source"), "by_source", index=True)
    summary["best_knn_by_source"] = ps["best k-NN measure"].to_dict()
    summary["n_sources_cosine_best"] = int((ps["best k-NN measure"] == LABEL["cosine"]).sum())
    summary["by_source_r2"] = {m: {src_: round(float(v), 4) for src_, v in C[m].groupby(level="src").mean().items()}
                               for m in key + ["pearson"]}

    # ---- failures and run time
    P = main_rows[main_rows["k"] == 10]
    ft = P[P["method"].isin(LEARNED)].groupby(["method", "n"]).agg(fail=("failed", "mean"), fit_s=("fit_s", "median"))
    ft = ft.unstack("n")
    ft.columns = [f"{a} n={b}" for a, b in ft.columns]
    ft = ft.rename(index=LABEL).rename_axis("method")
    save(ft, "failures_runtime", "{:.3f}", index=True)
    summary["fail_rate"] = {m: round(float(P[P["method"] == m]["failed"].mean()), 4) for m in LEARNED}
    summary["fail_n50_lfda"] = round(float(P[(P["method"] == "lfda") & (P["n"] == 50)]["failed"].mean()), 4)
    fx = C.loc[srcs == "OULAD", ["euclid", "manhattan", "cosine", "pearson"]].mean()
    summary["oulad_spread"] = round(float(fx.max() - fx.min()), 4)
    big = P[(P["n"] == 500) & (P["d"] == 100)]
    summary["fit_s_n500_d100"] = {m: round(float(big[big["method"] == m]["fit_s"].median()), 2) for m in LEARNED}

    # ---- selector (also returns features with the LOO-chosen alpha)
    F, oracle, best_cell = selector_analysis(C, F, rows, summary)

    # ---- alpha path and mechanism
    ap = C[[m for m in ALPHA_PATH if m in C.columns]].dropna()
    rel = ap.sub(ap["euclid"], axis=0)
    path = rel.groupby(level="n").mean().rename(columns={m: f"α={a:.2f}" for m, a in ALPHA_PATH.items()})
    save(path.rename_axis("n"), "alpha_path_by_n", "{:.4f}", index=True)
    a_star = ap.idxmax(axis=1).map(ALPHA_PATH)
    summary["alpha_star_share_euclid"] = round(float((a_star == 1.0).mean()), 3)
    summary["alpha_star_mean_by_n"] = {int(n): round(float(v), 3) for n, v in a_star.groupby(level="n").mean().items()}
    summary["alpha_star_dist"] = {str(a): round(float(v), 3) for a, v in a_star.value_counts(normalize=True).sort_index().items()}

    D = pd.DataFrame(index=C.index)
    D["d_maha"] = C["maha_lw"] - C["euclid"]
    D["d_shrink"] = C["maha_lw"] - C["maha_sample"]
    D["d_cos"] = C["cosine"] - C["euclid"]
    D["d_learn"] = Cx["best_learned"] - C["cosine"]
    D["d_model"] = C["ridge"] - C["cosine"]
    D["alpha_star"] = a_star.reindex(C.index)
    D = D.join(F)
    D.reset_index().to_csv(TAB / "cell_level.csv", index=False)
    mech = []
    rng = np.random.default_rng(0)
    cl = D.index.get_level_values("src").to_numpy()
    for tgt in ["d_maha", "d_shrink", "d_cos", "d_learn", "d_model", "alpha_star"]:
        for f in FEATS:
            S = D[[tgt, f]].dropna()
            rho = sps.spearmanr(S[tgt], S[f]).statistic
            c = S.index.get_level_values("src").to_numpy()
            ids = np.unique(c)
            boots = []
            for _ in range(1000):
                pick = rng.choice(ids, len(ids))
                T = pd.concat([S[c == p] for p in pick])
                if T[f].nunique() > 2:
                    boots.append(sps.spearmanr(T[tgt], T[f]).statistic)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            mech.append(dict(target=tgt, feature=f, spearman=rho, ci_lo=lo, ci_hi=hi, cells=len(S)))
    mech = pd.DataFrame(mech)
    save(mech, "mechanism_spearman")
    summary["mechanism"] = {f"{r['target']}~{r['feature']}": [round(r["spearman"], 3), round(r["ci_lo"], 3),
                                                             round(r["ci_hi"], 3)] for _, r in mech.iterrows()}
    summary["sva_median"] = round(float(D["sva"].median()), 2)
    summary["sva_share_gt1"] = round(float((D["sva"] > 1).mean()), 3)
    summary["d_maha_neg_share"] = round(float((D["d_maha"] < 0).mean()), 3)
    summary["d_maha_neg_share_sva_gt1"] = round(float((D.loc[D["sva"] > 1, "d_maha"] < 0).mean()), 3)
    by_task = D.groupby(level="task")[["d_maha", "d_cos", "d_learn", "d_model", "sva", "lw_alpha", "density"]].mean()
    save(by_task.rename_axis("task"), "by_task", index=True)
    summary["by_task"] = by_task.round(4).to_dict()

    # ---- protocol effect 1: transductive scaling / covariance
    tr = pd.DataFrame({"maha_lw": C["maha_lw_trans"] - C["maha_lw"],
                       "maha_sample": C["maha_sample_trans"] - C["maha_sample"]})
    flip = ((C["maha_lw_trans"] - C["euclid"]) * (C["maha_lw"] - C["euclid"]) < 0)
    t1 = tr.groupby(level="n").agg(["mean", "min", "max"])
    t1.columns = [f"{a} {b}" for a, b in t1.columns]
    t1["sign flips vs Euclid"] = flip.groupby(level="n").mean()
    save(t1.rename_axis("n"), "protocol_transductive", "{:.4f}", index=True)
    summary["transductive"] = {int(n): [round(float(r["maha_lw mean"]), 4), round(float(r["maha_sample mean"]), 4)]
                               for n, r in t1.iterrows()}
    summary["transductive_abs_mean"] = round(float(tr["maha_lw"].abs().mean()), 4)
    summary["transductive_flip_share"] = round(float(flip.mean()), 3)

    # ---- protocol effect 2: random instead of temporal outcome split
    Call = cell_table(rows)
    leak = Call[Call.index.get_level_values("base").str.contains("randsplit")]
    if len(leak):
        rec = []
        for (view, base, src, task, d, n), r in leak.iterrows():
            tv = view.replace("_randsplit", "")
            try:
                t = Call.xs((tv, n), level=("view", "n")).iloc[0]
            except KeyError:
                continue
            common = [m for m in present + MODELS if not np.isnan(r[m]) and not np.isnan(t[m])]
            rec.append(dict(source=src, d=d, n=n, cosine_temporal=t["cosine"], cosine_random=r["cosine"],
                            ridge_temporal=t["ridge"], ridge_random=r["ridge"],
                            best_knn_temporal=t[present].max(), best_knn_random=r[present].max(),
                            rank_spearman=sps.spearmanr(t[common], r[common]).statistic,
                            same_best=LABEL[t[present].idxmax()] == LABEL[r[present].idxmax()]))
        rec = pd.DataFrame(rec)
        t2 = rec.groupby("source").agg(cells=("n", "size"), cosine_temporal=("cosine_temporal", "mean"),
                                       cosine_random=("cosine_random", "mean"), ridge_temporal=("ridge_temporal", "mean"),
                                       ridge_random=("ridge_random", "mean"), rank_spearman=("rank_spearman", "mean"),
                                       same_best=("same_best", "mean")).reset_index()
        save(t2, "protocol_random_split")
        summary["random_split"] = dict(
            inflation_cosine=round(float((rec["cosine_random"] - rec["cosine_temporal"]).mean()), 4),
            inflation_ridge=round(float((rec["ridge_random"] - rec["ridge_temporal"]).mean()), 4),
            inflation_best_knn=round(float((rec["best_knn_random"] - rec["best_knn_temporal"]).mean()), 4),
            rel_inflation_ridge=round(float((rec["ridge_random"] / rec["ridge_temporal"]).mean() - 1), 3),
            rank_spearman=round(float(rec["rank_spearman"].mean()), 3),
            same_best=round(float(rec["same_best"].mean()), 3), cells=int(len(rec)))

    # ---- protocol effect 3: OULAD leave-one-module-out
    if LOMO.exists() and glob.glob(str(LOMO / "*.parquet")):
        lr, _ = load(LOMO)
        Lr2, Lauc = cell_table(lr), cell_table(lr, value="auc")
        O = C[C.index.get_level_values("src") == "OULAD"]
        Oauc = cell_table(main_rows[main_rows["src"] == "OULAD"], value="auc")
        keym = ["euclid", "cosine", "maha_lw", "pca80_proj", "nca", "mlkr", "ridge", "rf"]
        rec = []
        for n in sorted(Lr2.index.get_level_values("n").unique()):
            for m in keym:
                a = O.xs(n, level="n")[m].droplevel(["base", "src", "task", "d"])
                b = Lr2.xs(n, level="n")[m].droplevel(["base", "src", "task", "d"])
                aa = Oauc.xs(n, level="n")[m].droplevel(["base", "src", "task", "d"])
                bb = Lauc.xs(n, level="n")[m].droplevel(["base", "src", "task", "d"])
                rec.append(dict(n=n, method=LABEL[m], r2_within=a.mean(), r2_lomo=b.reindex(a.index).mean(),
                                auc_within=aa.mean(), auc_lomo=bb.reindex(aa.index).mean()))
        rec = pd.DataFrame(rec)
        rec["auc_gap"] = rec["auc_within"] - rec["auc_lomo"]
        save(rec, "protocol_lomo")
        summary["lomo"] = dict(auc_gap_mean=round(float(rec["auc_gap"].mean()), 4),
                               auc_within_ridge_500=round(float(rec[(rec.n == 500) & (rec.method == LABEL["ridge"])]["auc_within"].iloc[0]), 4),
                               auc_lomo_ridge_500=round(float(rec[(rec.n == 500) & (rec.method == LABEL["ridge"])]["auc_lomo"].iloc[0]), 4),
                               r2_gap_mean=round(float((rec["r2_within"] - rec["r2_lomo"]).mean()), 4),
                               reps=int(lr.groupby(["view", "n"])["rep"].nunique().min()))
        # do the measure rankings survive the shift?
        Lk = Lr2[present].dropna()
        Ok = O[present].reindex(Lk.index).dropna()
        rs = [sps.spearmanr(Ok.loc[i], Lk.loc[i]).statistic for i in Ok.index]
        summary["lomo"]["rank_spearman"] = round(float(np.mean(rs)), 3)

    # ---- sensitivity to k
    ks = {}
    for k in (5, 10, 20):
        Ck = cell_table(main_rows, k=k)[present].dropna()
        ks[k] = average_ranks(Ck)
    kt = pd.DataFrame(ks).rename(index=LABEL)
    kt.columns = [f"avg rank k={k}" for k in kt.columns]
    save(kt.rename_axis("method").sort_values("avg rank k=10"), "k_sensitivity", "{:.2f}", index=True)
    summary["k_spearman"] = {f"10vs{k}": round(float(sps.spearmanr(ks[10], ks[k]).statistic), 3) for k in (5, 20)}

    # ---- neighbour precision and hubness
    npk = cell_table(main_rows, value="nprec")[present]
    hub = cell_table(main_rows, value="hub_skew")[present]
    nh = pd.DataFrame({"neighbour precision@10": npk.mean(), "avg rank (precision)": average_ranks(npk.dropna()),
                       "hubness skew": hub.mean(), "mean R2": C[present].mean()}).rename(index=LABEL)
    save(nh.rename_axis("method").sort_values("avg rank (precision)"), "neighbour_precision", index=True)
    summary["hub"] = {m: round(float(hub[m].mean()), 3) for m in present}
    summary["nprec"] = {m: round(float(npk[m].mean()), 3) for m in present}
    summary["nprec_r2_rank_spearman"] = round(float(sps.spearmanr(average_ranks(npk.dropna()),
                                                                 average_ranks(C[present].dropna())).statistic), 3)
    hub_r = []
    for i in C.index:
        a, b = hub.loc[i], C.loc[i, present]
        if a.notna().all() and b.notna().all():
            hub_r.append(sps.spearmanr(a, b).statistic)
    summary["hub_r2_within_cell_spearman"] = round(float(np.nanmean(hub_r)), 3)

    # ---- sensitivity / robustness runs and the skill-specific task (if present)
    import analysis_extras

    analysis_extras.extras(RES, save, summary)
    analysis_extras.skill_task(RES, save, summary)

    import datetime as _dt

    summary["build_date"] = _dt.date.today().isoformat()
    summary["rope"] = ROPE
    summary["n_measures"] = len(present) + len([m for m in ALPHA_PATH if m not in present])
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["reps_min", "reps_max", "n_cells", "n_views", "n_sources"]}))
    print((TAB / "selector_rule.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
