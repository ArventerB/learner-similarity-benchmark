"""Check that every qualitative statement in the manuscript still holds in the data.

Numbers in the manuscript are inserted automatically, but sentences such as "cosine has
the best rank at n <= 200" are written by hand. This script asserts each of them against
results/summary.json and results/tables, so that re-running the experiment with more
repetitions (or other data) flags any sentence that needs rewording.

    python -I src/lsb/check_claims.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("LSB_RESULTS", ROOT / "results"))
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
T = RES / "tables"
LEARNED = ["lmnn", "nca", "itml", "itml_lw", "sdml", "lfda", "mlkr"]
WHITEN = ["maha_sample", "maha_lw", "maha_oas", "pca95_whiten"]
fails: list[str] = []
n_checks = 0


def check(name: str, cond: bool):
    global n_checks
    n_checks += 1
    if not bool(cond):
        fails.append(name)


C = S["contrasts"]
R = S["avg_rank"]
F = S["friedman"]

# ---- 4.1
for n in ("50", "100", "200"):
    check(f"cosine has the best average rank at n={n}", min(R[n], key=R[n].get) == "cosine")
top2 = sorted(R["500"], key=R["500"].get)[:2]
check("cosine and PCA-80% projection are the top two at n=500", set(top2) == {"cosine", "pca80_proj"})
check("their gap at n=500 is below the critical difference",
      abs(R["500"]["cosine"] - R["500"]["pca80_proj"]) < F["cells, n=500"]["CD"])
check("cosine wins the most cells", max(S["wins"], key=S["wins"].get) == "cosine")
check("cosine - Euclidean: interval above zero", C["cosine-euclid@all"]["lo"] > 0)
check("raw cosine is no better than the mean predictor", S["mean_r2_all"]["cosine_raw"] < 0.02)
check("cosine - raw cosine: interval above zero", C["cosine-cosine_raw@all"]["lo"] > 0)
check("OULAD: fixed measures within 0.02", S["oulad_spread"] < 0.02)
for n in ("50", "500"):
    cd = F[f"cells, n={n}"]["CD"]
    check(f"whitening measures rank below cosine by > 2 CD at n={n}",
          all(R[n][m] - R[n]["cosine"] > 2 * cd for m in WHITEN))
    check(f"sample Mahalanobis ranks below Euclidean by > CD at n={n}", R[n]["maha_sample"] - R[n]["euclid"] > cd)
check("Friedman significant at every level", all(v["p"] < 0.001 for v in F.values()))
check("ridge - cosine not significant at n=50", C["ridge-cosine@50"]["holm_p"] > 0.05)
check("ridge - cosine positive at n=100", C["ridge-cosine@100"]["lo"] > 0)
check("ridge - cosine positive at n=500", C["ridge-cosine@500"]["lo"] > 0)
check("ridge - cosine positive overall", C["ridge-cosine@all"]["lo"] > 0)

# ---- 4.2
check("LW improves on sample covariance", C["maha_lw-maha_sample@all"]["lo"] > 0)
check("... by more at n=50 than at n=500", C["maha_lw-maha_sample@50"]["diff"] > C["maha_lw-maha_sample@500"]["diff"])
check("LW Mahalanobis below Euclidean", C["maha_lw-euclid@all"]["hi"] < 0)
check("... in every source", C["maha_lw-euclid@all"]["sources_pos"].startswith("0/"))
check("... and the difference widens with n", C["maha_lw-euclid@500"]["diff"] < C["maha_lw-euclid@50"]["diff"])
check("... P(practically worse) > 0.99", C["maha_lw-euclid@all"]["p_worse"] > 0.99)
for m in ("nca", "mlkr"):
    check(f"{m} below cosine overall", C[f"{m}-cosine@all"]["hi"] < 0)
    check(f"{m} gap narrows with n", abs(C[f"{m}-cosine@500"]["diff"]) < abs(C[f"{m}-cosine@50"]["diff"]))
check("LMNN practically equivalent to Euclidean", C["lmnn-euclid@all"]["p_rope"] > 0.95)
check("SDML is the best learned metric at n=50", min(LEARNED, key=lambda m: R["50"][m]) == "sdml")
check("SDML loses rank with n", R["500"]["sdml"] > R["50"]["sdml"])
check("oracle learned metric below cosine overall", C["best_learned-cosine@all"]["hi"] < 0)
check("oracle interval includes zero at n=500",
      C["best_learned-cosine@500"]["lo"] < 0 < C["best_learned-cosine@500"]["hi"])
if "sens" in S:
    sv = S["sens"]
    check("rank-10 LMNN clearly better than default", all(sv[f"lmnn_rank10@{n}"]["vs_default"] > 0.02 for n in (100, 500)))
    others = [k for k, v in sv.items() if v["vs_default"] is not None and not k.startswith("lmnn_rank10")]
    check("other configuration changes move the result by < 0.03", all(abs(sv[k]["vs_default"]) < 0.03 for k in others))
    check("rank-10 LMNN reaches cosine at n=500 (interval includes zero)",
          sv["lmnn_rank10@500"]["lo"] < 0 < sv["lmnn_rank10@500"]["hi"])
    check("hindsight-best configuration within 0.02 of cosine at n=500",
          abs(S["sens_best_variant_vs_cosine"]["500"]) < 0.02)
    check("hindsight-best configuration not above cosine at n=100", S["sens_best_variant_vs_cosine"]["100"] < 0.005)
check("LFDA fails often at n=50", S["fail_n50_lfda"] > 0.3)
check("ITML fails in 5-30% of fits", 0.05 < S["fail_rate"]["itml"] < 0.3)
check("LMNN, NCA, MLKR never fail", all(S["fail_rate"][m] == 0 for m in ("lmnn", "nca", "mlkr")))
check("SDML fails in < 1%", S["fail_rate"]["sdml"] < 0.01)

# ---- 4.3
check("alignment > 1 in every cell", S["sva_share_gt1"] == 1.0)
ap = pd.read_csv(T / "alpha_path_by_n.csv").drop(columns=["n"])
check("alpha path is monotone at every n", all((ap.diff(axis=1).iloc[:, 1:] > 0).all(axis=1)))
check("Euclidean end is best in > 80% of cells", S["alpha_star_share_euclid"] > 0.8)
M = S["mechanism"]
lab_free = ["log_n_over_d", "lw_alpha", "erank_ratio", "top_eig_share", "mean_abs_corr", "density",
            "const_share", "hub_skew_train", "id_ratio"]
check("whitening loss correlates most strongly with the alignment",
      all(abs(M["d_maha~sva"][0]) > abs(M[f"d_maha~{f}"][0]) for f in lab_free))
check("... and its interval is below zero", M["d_maha~sva"][2] < 0)
check("cosine advantage: density negative", M["d_cos~density"][0] < -0.3)
check("cosine advantage: LW weight positive", M["d_cos~lw_alpha"][0] > 0.3)
check("cosine advantage: first-component share negative", M["d_cos~top_eig_share"][0] < -0.3)
check("cosine is the least hub-prone effective measure",
      all(S["hub"]["cosine"] < S["hub"][m] for m in ["euclid", "manhattan", "pca80_proj", "nca", "mlkr", "sdml", "lmnn", "maha_lw"]))
if "ability_r2" in S:
    a = S["ability_r2"]
    check("1-D ability match level with cosine at n=100", abs(a["ability_1d"]["100"] - a["cosine"]["100"]) < 0.02)
    check("cosine ahead of 1-D ability at n=500", a["cosine"]["500"] - a["ability_1d"]["500"] > 0.015)

# ---- 4.4
sel = S["selector"]
check("always cosine has the lowest regret", min(sel, key=sel.get) in ("always Cosine", "best single (LOSO)"))
check("LOSO best single is cosine everywhere", S["loso_best_single_all_cosine"])
fixed = sorted((v, k) for k, v in sel.items() if k.startswith("always"))
check("next fixed choices are PCA-80% projection and NCA",
      {fixed[1][1], fixed[2][1]} == {"always PCA-80% projection", "always NCA"})
check("rules do not beat always cosine", sel["depth-1 rule (LOSO)"] >= sel["always Cosine"]
      and sel["depth-2 rule (LOSO)"] >= sel["always Cosine"])
sb = S["selector_by_n"]
loo, cos = sb["internal LOO among label-free measures"], sb["always Cosine"]
check("internal LOO worse than cosine at n=50", loo["50"] > cos["50"])
check("internal LOO better than cosine at n=200 and n=500", loo["200"] < cos["200"] and loo["500"] < cos["500"])
check("naive LOO with learned metrics is harmful",
      sel["internal LOO among all measures (resubstitution for learned)"] > 5 * sel["always Cosine"])
check("LOO-tuned shrinkage ~ Euclidean", abs(sel["tuned shrinkage (LOO over the α path)"] - sel["always Euclidean"]) < 0.01)
check("fixed family wins less often at n=500 than at n=50", S["winner_family"]["500"]["fixed"] < S["winner_family"]["50"]["fixed"])

# ---- 4.5
check("transductive change negative at every n", all(v[0] < 0 for v in S["transductive"].values()))
check("transductive sign flips rare", S["transductive_flip_share"] < 0.05)
if "random_split" in S:
    check("random split inflates R2 by > 0.1", S["random_split"]["inflation_cosine"] > 0.1 and S["random_split"]["inflation_ridge"] > 0.1)
    check("random split inflation is about half", 0.4 < S["random_split"]["rel_inflation_ridge"] < 0.65)
    check("random split preserves ranking", S["random_split"]["rank_spearman"] > 0.9)
if "lomo" in S:
    lomo = pd.read_csv(T / "protocol_lomo.csv")
    check("AUC falls when training on other modules", S["lomo"]["auc_gap_mean"] > 0.02)
    check("Brier skill at or below zero for every method under module shift", (lomo["r2_lomo"] < 0.02).all())
    check("measure ranking less stable under module shift", S["lomo"]["rank_spearman"] < 0.8)

# ---- 4.6
k = pd.read_csv(T / "k_sensitivity.csv").set_index("method")
check("cosine first at k = 5, 10, 20", all(k[c].idxmin() == "Cosine" for c in k.columns))
gain = (k["avg rank k=5"] - k["avg rank k=20"]).sort_values(ascending=False)
check("NCA and MLKR gain most from larger k", set(gain.index[:2]) == {"NCA", "MLKR"})
check("k rankings correlate > 0.9", all(v > 0.9 for v in S["k_spearman"].values()))
check("neighbour precision ranking ~ R2 ranking", S["nprec_r2_rank_spearman"] > 0.85)
if "robust_best" in S:
    check("cosine best under every perturbation", all(v == "Cosine" for v in S["robust_best"].values()))
    check("robustness rankings correlate > 0.9", all(v > 0.9 for v in S["robust_rank_spearman"].values()))
if "skill_contrasts" in S:
    sc, sr, sb2 = S["skill_contrasts"], S["skill_r2"], S["skill_by_source"]
    check("skill task: Mahalanobis below Euclidean", sc["maha_lw-euclid"]["hi"] < 0)
    check("skill task: learned metrics below both fixed measures",
          all(sr[m][n] < min(sr["euclid"][n], sr["cosine"][n]) for m in ("nca", "mlkr") for n in sr["nca"]))
    check("skill task: ridge above cosine", sc["ridge-cosine"]["lo"] > 0)
    check("skill task: cosine has the best rank", min(S["skill_rank"], key=S["skill_rank"].get) == "cosine")
    check("skill task: cosine - Euclidean interval includes zero", sc["cosine-euclid"]["lo"] < 0 < sc["cosine-euclid"]["hi"])
    check("skill task: on XES3G5M cosine below Euclidean and PCA projection",
          sb2["cosine"]["XES3G5M"] < sb2["euclid"]["XES3G5M"] and sb2["cosine"]["XES3G5M"] < sb2["pca80_proj"]["XES3G5M"])
    check("skill task: 1-D ability also fails on XES3G5M", sb2["ability_1d"]["XES3G5M"] < sb2["euclid"]["XES3G5M"])

print(f"{n_checks - len(fails)} of {n_checks} statements hold")
if fails:
    print("STATEMENTS THAT NO LONGER HOLD (reword the manuscript):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
