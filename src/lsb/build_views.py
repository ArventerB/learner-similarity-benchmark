"""Turn each public dataset into learner-level "views": a feature matrix X (one row per
learner, d ability features), an outcome y, and a fixed learner partition.

One rule is used for every interaction-log source (see docs/FEATURE_RULE.md):

  * Learners are shuffled once per source (seed 2026) and split into
      reference pool (20 %, at most 3,000)  -> used only to DEFINE the features,
      test set (fixed, at most 1,000)       -> never used for fitting anything,
      candidate pool (the rest)             -> training samples are drawn from it.
  * Features: the d knowledge components (KCs) practised by most reference learners.
    Feature k of learner i is the smoothed success rate
        f_ik = (c_ik + a * p_k) / (n_ik + a),   a = 2,
    where n_ik / c_ik are attempts / (partial) successes of learner i on KC k in the
    feature interactions and p_k is the reference-pool mean success on KC k.
  * Outcome (four task types):
      future    success rate on the learner's last 30 % of interactions
                (features use only the first 70 %)                       [temporal]
      transfer  mean score in a course whose logs are not used for features
      unseen    mean score on KCs disjoint from the feature KCs
      warning   non-completion (fail or withdraw) of a module, predicted from the first
                28 days of activity                                      [OULAD, binary]

Sensitive attributes are never read: OULAD demographic columns (gender, region,
education, IMD band, age band, disability) and the IP-derived columns of the geography
data are not loaded at all.

Usage:  python -I src/lsb/build_views.py [source ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 2026
A_PRIOR = 2.0
D_LIST = (20, 50, 100)
HIST_FRAC = 0.7


# ------------------------------------------------------------------ shared machinery
def partition(users: np.ndarray, seed: int = SEED) -> dict[int, int]:
    """Map learner id -> 0 (reference), 1 (test) or 2 (candidate pool)."""
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(users.tolist())))
    rng.shuffle(u)
    n = len(u)
    n_ref = min(int(round(0.2 * n)), 3000)
    rest = n - n_ref
    n_test = int(min(1000, round(0.4 * rest)))
    part = np.full(n, 2, dtype=np.int8)
    part[:n_ref] = 0
    part[n_ref:n_ref + n_test] = 1
    return dict(zip(u.tolist(), part.tolist()))


def kc_matrix(feat: pd.DataFrame, users: np.ndarray, part: dict[int, int], d_max: int):
    """Smoothed per-KC success rates for the d_max most widely practised KCs.

    feat: columns user, kc, correct (feature interactions only).
    Returns F (len(users) x d_max), N (attempt counts), kc ids ordered by popularity.
    """
    is_ref = feat["user"].map(part).to_numpy() == 0
    ref = feat[is_ref]
    pop = ref.groupby("kc")["user"].nunique().sort_values(ascending=False, kind="mergesort")
    kcs = pop.index.to_numpy()[:d_max]
    p_k = ref.groupby("kc")["correct"].mean().reindex(kcs).to_numpy()
    sub = feat[feat["kc"].isin(kcs)]
    agg = sub.groupby(["user", "kc"], sort=False)["correct"].agg(["sum", "count"]).reset_index()
    u_idx = pd.Series(np.arange(len(users)), index=users)
    k_idx = pd.Series(np.arange(len(kcs)), index=kcs)
    r = u_idx.reindex(agg["user"]).to_numpy()
    c = k_idx.reindex(agg["kc"]).to_numpy()
    ok = ~np.isnan(r)
    r, c = r[ok].astype(int), c[ok].astype(int)
    S = np.zeros((len(users), len(kcs)))
    N = np.zeros((len(users), len(kcs)))
    S[r, c] = agg["sum"].to_numpy()[ok]
    N[r, c] = agg["count"].to_numpy()[ok]
    F = (S + A_PRIOR * p_k[None, :]) / (N + A_PRIOR)
    return F, N, kcs


def save_view(source, view, task, domain, target_desc, users, part, F, N, y, extra=None):
    """Write one file per feature dimension d (prefixes of the popularity-ordered KCs)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pvec = np.array([part[u] for u in users.tolist()], dtype=np.int8)
    rows = []
    d_list = [d for d in D_LIST if d <= F.shape[1]] if F.shape[1] >= D_LIST[0] else [F.shape[1]]
    if task == "warning":
        d_list = [F.shape[1]]
    for d in d_list:
        X = F[:, :d].astype(np.float64)
        dens = float((N[:, :d] > 0).mean()) if N is not None else 1.0
        name = f"{view}__d{d}"
        np.savez_compressed(OUT / f"{name}.npz", X=X, y=np.asarray(y, dtype=np.float64),
                            users=np.asarray(users), part=pvec)
        meta = dict(name=name, source=source, view=view, d=int(d), task=task, domain=domain,
                    target=target_desc, n_learners=int(len(users)),
                    n_ref=int((pvec == 0).sum()), n_test=int((pvec == 1).sum()),
                    n_pool=int((pvec == 2).sum()), feature_density=round(dens, 4),
                    y_mean=round(float(np.mean(y)), 4), y_sd=round(float(np.std(y)), 4))
        if extra:
            meta.update(extra)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        rows.append(meta)
        print(f"  [view] {name:34s} learners={len(users):6d} pool={meta['n_pool']:6d} "
              f"test={meta['n_test']:5d} density={dens:.2f} y={meta['y_mean']:.3f}±{meta['y_sd']:.3f}",
              flush=True)
    return rows


def temporal_view(df, source, view, domain, min_total=80, random_split=False):
    """df: user, kc, correct, order (one row per interaction x KC), `iid` = interaction id."""
    df = df.sort_values(["user", "order"], kind="mergesort")
    inter = df.drop_duplicates(["user", "iid"])[["user", "iid", "correct"]]
    cnt = inter.groupby("user")["iid"].transform("size").to_numpy()
    if random_split:  # leaky variant: outcome interactions are drawn at random, not from the future
        rng = np.random.default_rng(SEED)
        is_hist = rng.random(len(inter)) < HIST_FRAC
    else:
        pos = inter.groupby("user").cumcount().to_numpy()
        is_hist = pos < np.floor(HIST_FRAC * cnt)
    keep = cnt >= min_total
    inter = inter.assign(is_hist=is_hist)[keep]
    tgt = inter[~inter["is_hist"]].groupby("user")["correct"].agg(["mean", "size"])
    users = tgt.index[tgt["size"] >= 15].to_numpy()
    part = partition(users)
    hist_ids = inter.loc[inter["is_hist"], ["user", "iid"]]
    feat = df.merge(hist_ids, on=["user", "iid"], how="inner")
    feat = feat[feat["user"].isin(users)]
    F, N, _ = kc_matrix(feat[["user", "kc", "correct"]], users, part, max(D_LIST))
    y = tgt.loc[users, "mean"].to_numpy()
    desc = ("success rate on a random 30 % of the learner's interactions (leaky variant)"
            if random_split else "success rate on the last 30 % of the learner's interactions")
    return save_view(source, view, "future", domain, desc, users, part, F, N, y,
                     extra=dict(n_interactions=int(len(inter)), n_kc=int(df["kc"].nunique())))


# -------------------------------------------------------------------------- sources
def build_xes3g5m():
    print("== xes3g5m", flush=True)
    base = RAW / "xes3g5m" / "extracted"
    parts = []
    for fn in ["question_level__train_valid_sequences_quelevel.csv", "question_level__test_quelevel.csv"]:
        t = pd.read_csv(base / fn, usecols=["uid", "questions", "concepts", "responses", "timestamps"])
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    rec_u, rec_q, rec_c, rec_r, rec_t = [], [], [], [], []
    for uid, qs, cs, rs, ts in t.itertuples(index=False):
        q = qs.split(","); c = cs.split(","); r = rs.split(","); tm = ts.split(",")
        for qi, ci, ri, ti in zip(q, c, r, tm):
            if ri == "-1" or qi == "-1":
                continue
            rec_u.append(uid); rec_q.append(int(qi)); rec_c.append(ci); rec_r.append(int(ri)); rec_t.append(int(ti))
    df = pd.DataFrame(dict(user=rec_u, q=rec_q, kcs=rec_c, correct=rec_r, order=rec_t))
    df = df.drop_duplicates(["user", "q", "order"])
    df["iid"] = np.arange(len(df))
    df["kc"] = df["kcs"].str.split("_")
    df = df.explode("kc")
    df["kc"] = df["kc"].astype(int)
    rows = temporal_view(df[["user", "kc", "correct", "order", "iid"]], "XES3G5M", "xes3g5m",
                         "primary mathematics")
    rows += temporal_view(df[["user", "kc", "correct", "order", "iid"]], "XES3G5M", "xes3g5m_randsplit",
                          "primary mathematics", random_split=True)
    return rows


def _dkt2(name, source, view, domain):
    df = pd.read_csv(RAW / "dkt2" / "extracted" / f"{name}__preprocessed_df.csv", sep=None, engine="python")
    df = df.rename(columns={"user_id": "user", "skill_id": "kc", "timestamp": "order"})
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(int)
    df["iid"] = np.arange(len(df))
    rows = temporal_view(df[["user", "kc", "correct", "order", "iid"]], source, view, domain)
    rows += temporal_view(df[["user", "kc", "correct", "order", "iid"]], source, view + "_randsplit",
                          domain, random_split=True)
    return rows


def build_assist17():
    print("== assist17", flush=True)
    return _dkt2("assistments17", "ASSISTments 2017", "assist17", "middle-school mathematics")


def build_ednet():
    print("== ednet", flush=True)
    return _dkt2("ednet", "EdNet-KT1 (20k sample)", "ednet", "English (TOEIC) practice")


def build_assist09():
    print("== assist09", flush=True)
    df = pd.read_csv(RAW / "assist09" / "skill_builder_data_corrected_collapsed.csv", encoding="latin1",
                     usecols=["order_id", "user_id", "correct", "skill_id"], low_memory=False)
    df = df.dropna(subset=["skill_id"])
    df = df[df["correct"].isin([0, 1])]
    df = df.rename(columns={"user_id": "user", "order_id": "order"})
    df["iid"] = np.arange(len(df))
    df["kc"] = df["skill_id"].astype(str).str.split("_")
    df = df.explode("kc")
    df["kc"] = pd.factorize(df["kc"])[0]
    return temporal_view(df[["user", "kc", "correct", "order", "iid"]], "ASSISTments 2009", "assist09",
                         "middle-school mathematics", min_total=50)


def build_assist15():
    print("== assist15", flush=True)
    df = pd.read_csv(RAW / "assist15" / "2015_100_skill_builders_main_problems.csv", encoding="latin1")
    df = df[df["correct"].isin([0, 1])]
    df = df.rename(columns={"user_id": "user", "log_id": "order", "sequence_id": "kc"})
    df["correct"] = df["correct"].astype(int)
    df["iid"] = np.arange(len(df))
    return temporal_view(df[["user", "kc", "correct", "order", "iid"]], "ASSISTments 2015", "assist15",
                         "middle-school mathematics", min_total=50)


def build_slepemapy():
    print("== slepemapy", flush=True)
    # only the four columns needed; the IP-derived columns are deliberately not read
    df = pd.read_csv(RAW / "slepemapy" / "extracted" / "answer.csv", sep=";",
                     usecols=["id", "user", "place_asked", "place_answered"])
    df["correct"] = (df["place_asked"] == df["place_answered"]).astype(int)
    df = df.rename(columns={"place_asked": "kc", "id": "order"})
    df["iid"] = df["order"]
    return temporal_view(df[["user", "kc", "correct", "order", "iid"]], "Slepemapy geography",
                         "slepemapy", "geography facts")


def _pta_items(path: Path) -> pd.DataFrame:
    it = pd.read_csv(path)
    it["kc"] = it["knowledge_code"].apply(json.loads)
    return it[["item_id", "kc"]]


def build_ptadisc_cross():
    print("== ptadisc cross-course", flush=True)
    courses = ["c", "cpp", "java", "python", "ds"]
    logs = {}
    for ci, c in enumerate(courses):
        base = RAW / "ptadisc" / "cross_course" / c
        df = pd.concat([pd.read_csv(base / f, dtype={"user_id": np.int32, "item_id": np.int32,
                                                      "score": np.float32})
                        for f in ["train.csv", "valid.csv", "test.csv"]], ignore_index=True)
        it = _pta_items(base / "item.csv").explode("kc")
        it["kc"] = it["kc"].astype(np.int64) + 100000 * (ci + 1)   # KC ids are course-specific
        logs[c] = (df, it)
        print(f"  {c}: {len(df):,} logs, {df['user_id'].nunique():,} learners", flush=True)
    users = np.array(sorted(set.intersection(*[set(l[0]["user_id"].unique().tolist()) for l in logs.values()])))
    part = partition(users)
    rows = []
    for tgt in courses:
        y = logs[tgt][0].groupby("user_id")["score"].mean().reindex(users).to_numpy()
        feats = []
        for c in courses:
            if c == tgt:
                continue
            df, it = logs[c]
            m = df.merge(it, on="item_id", how="inner")[["user_id", "kc", "score"]]
            feats.append(m.rename(columns={"user_id": "user", "score": "correct"}))
        feat = pd.concat(feats, ignore_index=True)
        F, N, _ = kc_matrix(feat, users, part, max(D_LIST))
        rows += save_view("PTADisc cross-course", f"pta_x_{tgt}", "transfer", "university programming",
                          f"mean score in the held-out course ({tgt}); features from the other four courses",
                          users, part, F, N, y, extra=dict(n_interactions=int(len(feat))))
        del feat, feats
    return rows


def build_ptadisc_single():
    print("== ptadisc single-course", flush=True)
    rows = []
    doms = {"comp_thinking": "computational thinking", "database": "database technology",
            "linux": "Linux systems", "probability": "probability and statistics"}
    for c, dom in doms.items():
        base = RAW / "ptadisc" / "single_course" / c
        df = pd.read_csv(base / "dataset.csv")
        it = _pta_items(base / "item.csv")
        rng = np.random.default_rng(SEED)
        all_kc = np.array(sorted({k for ks in it["kc"] for k in ks}))
        is_feat_kc = dict(zip(all_kc.tolist(), (rng.random(len(all_kc)) < 0.7).tolist()))
        it["side"] = it["kc"].apply(lambda ks: "feat" if all(is_feat_kc[k] for k in ks)
                                    else ("tgt" if not any(is_feat_kc[k] for k in ks) else "mixed"))
        m = df.merge(it, on="item_id", how="inner").rename(columns={"user_id": "user", "score": "correct"})
        tgt = m[m["side"] == "tgt"].groupby("user")["correct"].agg(["mean", "size"])
        nf = m[m["side"] == "feat"].groupby("user")["correct"].size()
        users = np.array(sorted(set(tgt.index[tgt["size"] >= 15]) & set(nf.index[nf >= 30])))
        if len(users) < 600:
            print(f"  {c}: only {len(users)} eligible learners -> skipped", flush=True)
            continue
        part = partition(users)
        feat = m[(m["side"] == "feat") & m["user"].isin(users)][["user", "kc", "correct"]].explode("kc")
        feat["kc"] = feat["kc"].astype(int)
        F, N, _ = kc_matrix(feat, users, part, max(D_LIST))
        y = tgt.loc[users, "mean"].to_numpy()
        rows += save_view("PTADisc single-course", f"pta_{c}", "unseen", dom,
                          "mean score on KCs disjoint from the feature KCs", users, part, F, N, y,
                          extra=dict(n_interactions=int(len(m)), n_kc=int(len(all_kc))))
    return rows


OULAD_ACT = ["resource", "oucontent", "url", "homepage", "subpage", "glossary", "forumng",
             "oucollaborate", "dataplus", "quiz", "ouelluminate", "sharedsubpage", "questionnaire",
             "page", "externalquiz", "ouwiki", "dualpane", "repeatactivity", "folder", "htmlactivity"]


def oulad_table() -> pd.DataFrame:
    """One row per (learner, module): early-activity features and the binary outcome."""
    base = RAW / "oulad" / "extracted"
    # demographic columns are deliberately excluded from usecols
    info = pd.read_csv(base / "studentInfo.csv",
                       usecols=["code_module", "code_presentation", "id_student",
                                "num_of_prev_attempts", "studied_credits", "final_result"])
    reg = pd.read_csv(base / "studentRegistration.csv")
    for col in ["date_registration", "date_unregistration"]:   # the UCI copy marks missing as "?"
        reg[col] = pd.to_numeric(reg[col], errors="coerce")
    vle = pd.read_csv(base / "vle.csv", usecols=["id_site", "activity_type"])
    sv = pd.read_csv(base / "studentVle.csv",
                     dtype={"id_student": np.int32, "id_site": np.int32, "date": np.int16, "sum_click": np.int32})
    key = ["code_module", "code_presentation", "id_student"]
    df = info.merge(reg, on=key, how="left")
    # learners still registered after day 28 and registered by day 28
    df = df[(df["date_unregistration"].isna()) | (df["date_unregistration"] > 28)]
    df = df[df["date_registration"].fillna(0) <= 28]
    sv = sv[sv["date"] <= 28].merge(vle, on="id_site", how="left")
    g = sv.groupby(key)
    f = pd.DataFrame({"clicks_total": g["sum_click"].sum(), "active_days": g["date"].nunique(),
                      "n_sites": g["id_site"].nunique()})
    f["clicks_prestart"] = sv[sv["date"] < 0].groupby(key)["sum_click"].sum()
    for w in range(4):
        m = (sv["date"] >= 7 * w) & (sv["date"] < 7 * (w + 1))
        f[f"clicks_week{w + 1}"] = sv[m].groupby(key)["sum_click"].sum()
    act = sv.pivot_table(index=key, columns="activity_type", values="sum_click", aggfunc="sum")
    for a in OULAD_ACT:
        f[f"act_{a}"] = act[a] if a in act.columns else 0.0
    # number of assessments handed in by day 28 (submission behaviour only, no scores)
    ass = pd.read_csv(base / "assessments.csv", usecols=["code_module", "code_presentation", "id_assessment"])
    sa = pd.read_csv(base / "studentAssessment.csv", usecols=["id_assessment", "id_student", "date_submitted"])
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa = sa[sa["date_submitted"] <= 28].merge(ass, on="id_assessment", how="inner")
    f["n_submitted"] = sa.groupby(key)["id_assessment"].nunique()
    df = df.merge(f.reset_index(), on=key, how="left")
    feat_cols = [c for c in f.columns]
    df[feat_cols] = np.log1p(df[feat_cols].fillna(0.0))
    df["reg_lead_days"] = -df["date_registration"].fillna(0)
    df["y"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    df = df.sort_values(key).drop_duplicates(["code_module", "id_student"], keep="first")
    cols = ["reg_lead_days", "num_of_prev_attempts", "studied_credits"] + feat_cols
    return df[["code_module", "id_student", "y"] + cols].reset_index(drop=True), cols


def build_oulad():
    print("== oulad", flush=True)
    df, cols = oulad_table()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "oulad_table.parquet")
    (OUT / "oulad_features.json").write_text(json.dumps(cols), encoding="utf-8")
    rows = []
    for mod, g in df.groupby("code_module"):
        users = g["id_student"].to_numpy()
        if len(users) < 1200:
            print(f"  module {mod}: {len(users)} learners -> skipped (too small)", flush=True)
            continue
        part = partition(users)
        F = g[cols].to_numpy(dtype=float)
        rows += save_view("OULAD", f"oulad_{mod}", "warning", "distance higher education",
                          "non-completion (fail or withdraw), from the first 28 days of activity",
                          users, part, F, None, g["y"].to_numpy(), extra=dict(n_kc=len(cols)))
    return rows


BUILDERS = {"oulad": build_oulad, "assist09": build_assist09, "assist15": build_assist15,
            "assist17": build_assist17, "ednet": build_ednet, "xes3g5m": build_xes3g5m,
            "slepemapy": build_slepemapy, "ptadisc_single": build_ptadisc_single,
            "ptadisc_cross": build_ptadisc_cross}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    wanted = sys.argv[1:] or list(BUILDERS)
    all_rows = []
    for w in wanted:
        all_rows += BUILDERS[w]()
    idx = OUT / "views_index.csv"
    new = pd.DataFrame(all_rows)
    if idx.exists() and len(sys.argv) > 1:
        old = pd.read_csv(idx)
        new = pd.concat([old[~old["name"].isin(new["name"])], new], ignore_index=True)
    new = new[[(OUT / f"{n}.npz").exists() for n in new["name"]]]   # drop entries whose file was removed
    new.sort_values("name").to_csv(idx, index=False)
    print(f"wrote {idx} ({len(new)} view files)")
