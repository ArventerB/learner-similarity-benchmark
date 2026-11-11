"""Analysis of the sensitivity / robustness runs and of the skill-specific task.

Called at the end of analysis.py; adds tables and keys to results/summary.json.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from metrics import LABEL
from stats import average_ranks, cluster_bootstrap_diff, wilcoxon

XLAB = {**LABEL, "lmnn_k1": "LMNN, 1 target neighbour", "lmnn_k7": "LMNN, 7 target neighbours",
        "lmnn_rank10": "LMNN, rank 10", "nca_rank10": "NCA, rank 10", "nca_rank5": "NCA, rank 5",
        "mlkr_rank10": "MLKR, rank 10", "lmnn_median": "LMNN, 2 classes (median)",
        "lmnn_quintile": "LMNN, 5 classes (quintiles)", "nca_median": "NCA, 2 classes (median)",
        "nca_quintile": "NCA, 5 classes (quintiles)", "pca50_proj": "PCA-50% projection",
        "pc1": "First principal component only", "ability_1d": "Overall ability (1-D)"}
SRC = {"PTADisc cross-course": "PTADisc", "PTADisc single-course": "PTADisc", "EdNet-KT1 (20k sample)": "EdNet",
       "Slepemapy geography": "Slepemapy", "ASSISTments 2009": "ASSIST09", "ASSISTments 2015": "ASSIST15",
       "ASSISTments 2017": "ASSIST17", "XES3G5M": "XES3G5M", "OULAD": "OULAD"}
VSRC = {"assist17": "ASSIST17", "ednet": "EdNet", "xes3g5m": "XES3G5M", "slepemapy": "Slepemapy"}


def _read(folder: Path):
    fs = sorted(glob.glob(str(folder / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def extras(res: Path, save, summary: dict):
    df = _read(res / "extras")
    if df is None:
        return
    df["src"] = df["source"].map(SRC)
    summary["extras_reps"] = int(df.groupby(["view", "n"])["rep"].nunique().min())
    summary["extras_cells"] = int(df.groupby(["view", "n"]).ngroups)
    cell = df.groupby(["part", "cond", "view", "src", "task", "n", "method"])["r2"].mean().reset_index()

    # ---- A. sensitivity of learned metrics: paired difference to the default configuration and to cosine
    A = cell[cell["part"] == "A"].pivot_table(index=["view", "src", "n"], columns="method", values="r2")
    Cc = cell[cell["part"] == "C"].pivot_table(index=["view", "src", "n"], columns="method", values="r2")
    A = A.join(Cc[["cosine", "euclid"]])
    fam = {"lmnn": ["lmnn", "lmnn_k1", "lmnn_k7", "lmnn_rank10", "lmnn_median", "lmnn_quintile"],
           "nca": ["nca", "nca_rank10", "nca_rank5", "nca_median", "nca_quintile"],
           "mlkr": ["mlkr", "mlkr_rank10"]}
    rec = []
    for base, variants in fam.items():
        for v in variants:
            row = dict(variant=XLAB[v])
            for n in sorted(A.index.get_level_values("n").unique()):
                S = A.xs(n, level="n")[list(dict.fromkeys([v, base, "cosine"]))].dropna()
                c = S.index.get_level_values("src").to_numpy()
                row[f"R2 n={n}"] = float(S[v].mean())
                row[f"vs default n={n}"] = float((S[v] - S[base]).mean()) if v != base else np.nan
                m, lo, hi = cluster_bootstrap_diff((S[v] - S["cosine"]).to_numpy(), c, n_boot=2000)
                row[f"vs cosine n={n}"] = m
                row[f"CI n={n}"] = f"[{lo:+.3f}, {hi:+.3f}]"
                row[f"cells n={n}"] = len(S)
                summary.setdefault("sens", {})[f"{v}@{n}"] = dict(r2=round(row[f"R2 n={n}"], 4), vs_cosine=round(m, 4),
                                                                   lo=round(lo, 4), hi=round(hi, 4), cells=len(S),
                                                                   vs_default=None if v == base else round(row[f"vs default n={n}"], 4))
            rec.append(row)
    save(pd.DataFrame(rec), "sensitivity_learned", "{:+.3f}")
    best_var = A[[v for vs in fam.values() for v in vs]].max(axis=1)
    summary["sens_best_variant_vs_cosine"] = {int(n): round(float((best_var - A["cosine"]).xs(n, level="n").mean()), 4)
                                              for n in sorted(A.index.get_level_values("n").unique())}

    # ---- C. is retrieval more than matching on overall ability?
    order = ["cosine", "pca80_proj", "pca50_proj", "pc1", "ability_1d", "euclid"]
    rows = []
    for m in order:
        row = dict(measure=XLAB[m])
        for n in sorted(Cc.index.get_level_values("n").unique()):
            row[f"R2 n={n}"] = float(Cc.xs(n, level="n")[m].mean())
        rows.append(row)
    save(pd.DataFrame(rows), "ability_1d")
    Ct = cell[cell["part"] == "C"].pivot_table(index=["view", "src", "task", "n"], columns="method", values="r2")
    bt = (Ct["ability_1d"] - Ct["cosine"]).groupby(level="task").mean()
    summary["ability1d_minus_cosine_by_task"] = {k: round(float(v), 4) for k, v in bt.items()}
    d = (Cc["ability_1d"] - Cc["cosine"])
    m, lo, hi = cluster_bootstrap_diff(d.to_numpy(), Cc.index.get_level_values("src").to_numpy(), n_boot=2000)
    summary["ability1d_minus_cosine"] = [round(m, 4), round(lo, 4), round(hi, 4)]
    summary["ability_r2"] = {mm: {int(n): round(float(Cc.xs(n, level="n")[mm].mean()), 4)
                                  for n in sorted(Cc.index.get_level_values("n").unique())} for mm in order}

    # ---- B. robustness
    B = cell[cell["part"] == "B"].pivot_table(index=["view", "src", "n", "method"], columns="cond", values="r2").reset_index()
    conds = [c for c in ["clean", "mask30", "mask60", "noise"] if c in B.columns]
    R = B.groupby("method")[conds].mean()
    for c in conds[1:]:
        R[f"Δ {c}"] = R[c] - R["clean"]
    R = R.rename(index=XLAB).sort_values("clean", ascending=False)
    save(R.rename_axis("method"), "robustness", index=True)
    piv = cell[cell["part"] == "B"].pivot_table(index=["view", "n", "cond"], columns="method", values="r2")
    knn = [m for m in piv.columns if m != "ridge"]
    rk = {c: average_ranks(piv.xs(c, level="cond")[knn].dropna()) for c in conds}
    summary["robust_rank_spearman"] = {c: round(float(sps.spearmanr(rk["clean"], rk[c]).statistic), 3) for c in conds[1:]}
    summary["robust_best"] = {c: XLAB[rk[c].idxmin()] for c in conds}
    summary["robust_mean"] = {m: {c: round(float(R.loc[XLAB[m], c]), 4) for c in conds}
                              for m in ["cosine", "euclid", "maha_lw", "nca", "mlkr", "ridge", "pca80_proj", "maha_sample"]}


def skill_task(res: Path, save, summary: dict):
    df = _read(res / "skill_task")
    if df is None:
        return
    df["src"] = df["view"].map(VSRC)
    cell = df.groupby(["view", "src", "kc", "n", "method"])["r2"].mean().unstack("method")
    summary["skill_cells"] = int(len(cell))
    summary["skill_reps"] = int(df.groupby(["view", "kc", "n"])["rep"].nunique().min())
    summary["skill_targets"] = int(df.groupby(["view", "kc"]).ngroups)
    knn = [m for m in ["cosine", "pca80_proj", "euclid", "ability_1d", "nca", "mlkr", "maha_lw", "maha_sample"] if m in cell.columns]
    rows = []
    n_lvl = cell.index.get_level_values("n")
    for m in knn + ["ridge"]:
        row = dict(method=XLAB[m])
        for n in sorted(n_lvl.unique()):
            row[f"R2 n={n}"] = float(cell.loc[n_lvl == n, m].mean())
        row["avg rank (k-NN)"] = float(average_ranks(cell[knn].dropna())[m]) if m in knn else np.nan
        rows.append(row)
    save(pd.DataFrame(rows).rename(columns=lambda c: c.replace("R2 ", "R² ").replace("method", "Measure / model")), "skill_task")
    summary["skill_r2"] = {m: {int(n): round(float(cell.loc[n_lvl == n, m].mean()), 4) for n in sorted(n_lvl.unique())}
                           for m in knn + ["ridge"]}
    summary["skill_rank"] = {m: round(float(v), 2) for m, v in average_ranks(cell[knn].dropna()).items()}
    con = {}
    for a, b in [("cosine", "euclid"), ("maha_lw", "euclid"), ("ability_1d", "cosine"), ("cosine", "nca"),
                 ("ridge", "cosine"), ("pca80_proj", "cosine")]:
        S = cell[[a, b]].dropna()
        m, lo, hi = cluster_bootstrap_diff((S[a] - S[b]).to_numpy(), S.index.get_level_values("src").to_numpy(), n_boot=2000)
        con[f"{a}-{b}"] = dict(diff=round(m, 4), lo=round(lo, 4), hi=round(hi, 4),
                               wilcoxon_p=float(wilcoxon(S[a].to_numpy(), S[b].to_numpy())),
                               share_pos=round(float(((S[a] - S[b]) > 0).mean()), 3), cells=int(len(S)))
    summary["skill_contrasts"] = con
    summary["skill_by_source"] = {m: {k: round(float(v), 4) for k, v in cell[m].groupby(level="src").mean().items()}
                                  for m in knn + ["ridge"]}
    bs = cell[knn + ["ridge"]].groupby(level="src").mean().rename(columns=XLAB)
    save(bs.rename_axis("source"), "skill_task_by_source", index=True)
