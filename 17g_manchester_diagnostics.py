"""
17g_manchester_diagnostics.py
=============================
IMT Attention Bias Paper 2 — Step 17g:
Diagnose the corpus-specific factor that distinguishes Manchester
from English-UK on the exposure-gate effect.

17f ruled out observation window as the sole explanation: UK at the
Manchester-equivalent age range (24-36 mo) still showed β = +0.021 (p < .05),
while Manchester β = -0.007 (n.s.). Three candidate explanations:

  (1) Caregiver-speech profile differences  (MLU, type-token ratio)
  (2) Sampling density per child            (episodes per month observed)
  (3) Cue-inventory composition             (which cues are present and
                                              at what frequency)

This script tests all three in one batch.

────────────────────────────────────────────────────────────────────────────

Section 1 — Descriptives (UK vs Manchester)
  Per-child:
    n_files (sessions)
    n_post_episodes
    obs_window_months
    episodes_per_month   = n_post_episodes / max(obs_window_months, 0.1)
    episodes_per_file
    MLU_child, MLU_caregiver
    types_caregiver_lemmas (lexical diversity ~ vocabulary breadth)
    ttr_caregiver         = types_caregiver_lemmas / total_caregiver_tokens
  Group-level:
    means, sds, Mann-Whitney U test on each metric
  Cue mix:
    Per-cue episode share in each corpus
    Spearman ρ across the union of top-N cues

Section 2 — Density × β analysis
  Within UK + Manchester children with per-child β from 17c:
    WLS  β_i ~ episodes_per_month_z + corpus_dummy + log(n_episodes)_z
  Subgroup meta: UK children stratified by median density;
                  is the high-density UK subgroup more Manchester-like?

Section 3 — Cue-mix matched UK fit
  Weight each UK episode by the Manchester / UK cue-share ratio for its
  cue_subtype. Refit M_new on the reweighted UK data; if β remains
  positive after Manchester-matching the cue distribution, cue mix is
  not the explanation.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17g/)
----------------------
* manchester_vs_uk_descriptives.csv     per-child table for both corpora
* descriptive_group_stats.json          group-level summary + Mann-Whitney
* cue_mix_share.csv                     per-cue share UK vs Manchester
* density_x_beta_results.json
* cue_mix_match_results.json
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / Manchester dx v1 | 2026-06-21
"""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from tqdm import tqdm
    from scipy.stats import mannwhitneyu, spearmanr, norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


UK_CFG = {
    "label":          "English-UK",
    "reuse_csv":      "./output/v16/English-UK_episodes_with_reuse.csv",
    "tagged_csv":     "./output/English-UK_tokens_tagged.csv",
    "joined_csv":     "./output/v11_runA/English-UK_r_plus_joined.csv",
    "json_cache":     "./output/json_cache/English-UK",
    "utterances_csv": "./output/English-UK_utterances.csv",
    "tokens_csv":     "./output/English-UK_tokens.csv",
}

MANC_CFG = {
    "label":          "Manchester",
    "reuse_csv":      "./output/v16/English-Manchester_episodes_with_reuse.csv",
    "tagged_csv":     "./output/English-Manchester_tokens_tagged.csv",
    "joined_csv":     "./output/v11/English-Manchester_r_plus_joined.csv",
    "json_cache":     "./output/json_cache/English-Manchester",
    "utterances_csv": "./output/English-Manchester_utterances.csv",
    "tokens_csv":     "./output/English-Manchester_tokens.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


def build_file_child_map(json_cache_dir: Path) -> Tuple[Dict[str, str], int]:
    if not json_cache_dir.exists():
        return {}, 0
    mapping: Dict[str, str] = {}
    max_len = 0
    for child_dir in json_cache_dir.iterdir():
        if not child_dir.is_dir():
            continue
        for jf in child_dir.rglob("*.json"):
            mapping[jf.stem] = child_dir.name
            if len(jf.stem) > max_len:
                max_len = len(jf.stem)
    return mapping, max_len


def attach_child(df: pd.DataFrame, file_to_child: Dict[str, str], max_len: int) -> pd.DataFrame:
    raw = df["file"].astype(str)
    df = df.copy()
    df["child"] = raw.map(file_to_child)
    miss = df["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        df.loc[miss, "child"] = padded[miss].map(file_to_child)
    return df


def build_child_utt_index(tagged_csv: Path) -> Dict[str, Tuple[np.ndarray, List[set]]]:
    cols = ["file", "speaker_role", "utterance_index", "cue_subtype", "is_cue_token"]
    df = pd.read_csv(tagged_csv, usecols=cols, low_memory=False, dtype={"file": str})
    df = df[df["speaker_role"] == "child"].copy()
    cue_rows = df[df["is_cue_token"].astype(str) == "True"][
        ["file", "utterance_index", "cue_subtype"]
    ].copy()
    cue_rows["cue_subtype"] = cue_rows["cue_subtype"].fillna("").astype(str).str.strip()
    cue_rows = cue_rows[cue_rows["cue_subtype"] != ""]
    cues_per_utt: Dict[Tuple[str, int], set] = {}
    for f, idx, cue in cue_rows.itertuples(index=False, name=None):
        cues_per_utt.setdefault((f, int(idx)), set()).add(cue)
    utts = df[["file", "utterance_index"]].drop_duplicates().copy()
    utts["utterance_index"] = utts["utterance_index"].astype(int)
    file_index: Dict[str, Tuple[np.ndarray, List[set]]] = {}
    for f, g in utts.groupby("file"):
        idxs = np.sort(g["utterance_index"].values.astype(int))
        cue_sets = [cues_per_utt.get((f, int(i)), set()) for i in idxs]
        file_index[str(f)] = (idxs, cue_sets)
    return file_index


def add_prior_local_freq(reuse_df: pd.DataFrame,
                          file_index: Dict[str, Tuple[np.ndarray, List[set]]],
                          prior_window: int) -> pd.DataFrame:
    n = len(reuse_df)
    prior = np.zeros(n, dtype=np.int32)
    files = reuse_df["file"].astype(str).values
    utt_idxs = reuse_df["child_utt_idx"].astype(int).values
    cues = reuse_df["cue_subtype"].astype(str).values
    for i in tqdm(range(n), desc=f"    prior_local_freq W={prior_window}"):
        entry = file_index.get(files[i])
        if entry is None:
            continue
        idxs, cue_sets = entry
        pos = bisect.bisect_left(idxs, utt_idxs[i])
        start = max(0, pos - prior_window)
        window_slice = cue_sets[start:pos]
        c = cues[i]
        prior[i] = sum(1 for s in window_slice if c in s)
    out = reuse_df.copy()
    out["prior_local_freq"] = prior
    return out


def random_effects_meta(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    n = len(betas)
    if n < 2:
        return {"error": "n < 2"}
    w_fe = 1.0 / (ses ** 2)
    beta_fe = float(np.sum(w_fe * betas) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (betas - beta_fe) ** 2))
    df = n - 1
    sum_w = float(np.sum(w_fe))
    sum_w2 = float(np.sum(w_fe ** 2))
    C = sum_w - sum_w2 / sum_w if sum_w > 0 else float("nan")
    tau2 = max(0.0, (Q - df) / C) if C and not math.isnan(C) and C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    w_re = 1.0 / (ses ** 2 + tau2)
    beta_re = float(np.sum(w_re * betas) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z_re = beta_re / se_re if se_re > 0 else float("nan")
    p = float(2 * (1.0 - norm.cdf(abs(z_re)))) if not math.isnan(z_re) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": z_re, "pooled_p_RE": p,
        "pooled_beta_FE": beta_fe,
        "pooled_se_FE": float(np.sqrt(1.0 / sum_w)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Descriptives
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_child_descriptives(cfg: Dict[str, str], window: int,
                                     contingent_only: bool) -> pd.DataFrame:
    print(f"\n  Computing descriptives for {cfg['label']} ...")
    json_cache = Path(cfg["json_cache"])
    file_to_child, max_len = build_file_child_map(json_cache)

    # Utterances → MLU and verbosity
    print(f"    loading utterances ...")
    u = pd.read_csv(cfg["utterances_csv"], usecols=["file", "is_child", "is_caregiver",
                                                       "num_tokens"], low_memory=False,
                     dtype={"file": str})
    u = attach_child(u, file_to_child, max_len).dropna(subset=["child"])
    mlu = u.groupby("child").apply(lambda g: pd.Series({
        "n_files":         g["file"].nunique(),
        "n_utt_child":     int(g["is_child"].sum()),
        "n_utt_caregiver": int(g["is_caregiver"].sum()),
        "mlu_child":       float(g[g["is_child"]]["num_tokens"].mean())
                             if g["is_child"].any() else float("nan"),
        "mlu_caregiver":   float(g[g["is_caregiver"]]["num_tokens"].mean())
                             if g["is_caregiver"].any() else float("nan"),
    })).reset_index()

    # Tokens → caregiver TTR (types / tokens) and lemma vocabulary
    print(f"    loading tokens for caregiver lexical diversity ...")
    t = pd.read_csv(cfg["tokens_csv"], usecols=["file", "speaker_role", "lemma"],
                     low_memory=False, dtype={"file": str})
    t = t[t["speaker_role"] == "caregiver"].copy()
    t = attach_child(t, file_to_child, max_len).dropna(subset=["child"])
    t["lemma"] = t["lemma"].fillna("").astype(str).str.strip()
    t = t[t["lemma"] != ""]
    ttr_rows: List[Dict] = []
    for child_id, g in t.groupby("child"):
        n_tok = int(len(g))
        n_types = int(g["lemma"].nunique())
        ttr_rows.append({
            "child":                       child_id,
            "n_caregiver_tokens":          n_tok,
            "types_caregiver_lemmas":      n_types,
            "ttr_caregiver":               n_types / n_tok if n_tok else float("nan"),
        })
    ttr_df = pd.DataFrame(ttr_rows)

    # Reuse (post-MSR contingent episodes) → density & obs window
    print(f"    loading reuse (post-MSR contingent) ...")
    reuse = pd.read_csv(cfg["reuse_csv"], low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    reuse = attach_child(reuse, file_to_child, max_len).dropna(subset=["child"])
    reuse_post = reuse[reuse["child_age_months"] >= 24].copy()
    dens_rows: List[Dict] = []
    for child_id, g in reuse_post.groupby("child"):
        age_min = float(g["child_age_months"].min())
        age_max = float(g["child_age_months"].max())
        window_mo = max(age_max - age_min, 0.1)
        n_files = int(g["file"].nunique())
        n_eps   = int(len(g))
        dens_rows.append({
            "child":                child_id,
            "n_post_episodes":      n_eps,
            "n_files_post":         n_files,
            "age_min_post":         age_min,
            "age_max_post":         age_max,
            "obs_window_months":    window_mo,
            "episodes_per_month":   n_eps / window_mo,
            "episodes_per_file":    n_eps / max(n_files, 1),
        })
    dens_df = pd.DataFrame(dens_rows)

    merged = mlu.merge(ttr_df, on="child", how="outer").merge(dens_df, on="child", how="outer")
    merged["corpus"] = cfg["label"]
    return merged


def group_stats_compare(uk: pd.DataFrame, manc: pd.DataFrame,
                          metrics: List[str]) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        u_vals = uk[m].dropna().values
        m_vals = manc[m].dropna().values
        if len(u_vals) < 2 or len(m_vals) < 2:
            rows[m] = {"error": "insufficient data"}
            continue
        try:
            U_stat, p_mw = mannwhitneyu(u_vals, m_vals, alternative="two-sided")
        except Exception as exc:
            U_stat, p_mw = float("nan"), float("nan")
        rows[m] = {
            "uk_n":          int(len(u_vals)),
            "uk_mean":       float(np.mean(u_vals)),
            "uk_median":     float(np.median(u_vals)),
            "uk_sd":         float(np.std(u_vals, ddof=1)) if len(u_vals) > 1 else float("nan"),
            "manc_n":        int(len(m_vals)),
            "manc_mean":     float(np.mean(m_vals)),
            "manc_median":   float(np.median(m_vals)),
            "manc_sd":       float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else float("nan"),
            "mw_U":          float(U_stat),
            "mw_p":          float(p_mw),
            "diff_means":    float(np.mean(u_vals) - np.mean(m_vals)),
        }
    return rows


def compute_cue_mix(cfg: Dict[str, str], contingent_only: bool) -> pd.Series:
    reuse = pd.read_csv(cfg["reuse_csv"], usecols=["cue_subtype", "r_plus_label"],
                         low_memory=False)
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"]
    counts = reuse["cue_subtype"].value_counts()
    share = counts / counts.sum()
    return share


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Density × β analysis
# ─────────────────────────────────────────────────────────────────────────────

def density_x_beta(per_child_with_density: pd.DataFrame) -> Dict[str, Any]:
    """
    WLS: β_i ~ episodes_per_month_z + corpus_dummy + log(n_episodes)_z
    Children pool: UK ∪ Manchester (with per-child β from 17c).
    """
    d = per_child_with_density.dropna(subset=[
        "beta_COI_x_cum", "se_COI_x_cum", "episodes_per_month", "n_episodes"
    ]).copy()
    d["log_n_episodes"] = np.log(d["n_episodes"])
    d["episodes_per_month_z"] = z(d["episodes_per_month"])
    d["log_n_episodes_z"]     = z(d["log_n_episodes"])
    d["is_manchester"]        = (d["corpus_label"] == "Manchester").astype(int)
    preds = ["episodes_per_month_z", "log_n_episodes_z", "is_manchester"]
    X = sm.add_constant(d[preds].astype(float))
    y = d["beta_COI_x_cum"].astype(float)
    w = 1.0 / (d["se_COI_x_cum"].astype(float) ** 2)
    wls = sm.WLS(y, X, weights=w).fit()
    params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + preds:
        params[p] = {
            "beta": float(wls.params[p]),
            "se":   float(wls.bse[p]),
            "t":    float(wls.tvalues[p]),
            "p":    float(wls.pvalues[p]),
        }
    out = {"n": int(len(d)), "r2": float(wls.rsquared), "params": params}

    # UK-only WLS (β ~ density + log_n)
    d_uk = d[d["corpus_label"] == "English-UK"].copy()
    if len(d_uk) >= 5:
        X2 = sm.add_constant(d_uk[["episodes_per_month_z", "log_n_episodes_z"]].astype(float))
        y2 = d_uk["beta_COI_x_cum"].astype(float)
        w2 = 1.0 / (d_uk["se_COI_x_cum"].astype(float) ** 2)
        wls2 = sm.WLS(y2, X2, weights=w2).fit()
        out["uk_only"] = {
            "n": int(len(d_uk)), "r2": float(wls2.rsquared),
            "params": {p: {
                "beta": float(wls2.params[p]), "se": float(wls2.bse[p]),
                "p": float(wls2.pvalues[p]),
            } for p in ["const", "episodes_per_month_z", "log_n_episodes_z"]},
        }

    # Stratified meta — UK high-density vs low-density
    if len(d_uk) >= 4:
        med = float(d_uk["episodes_per_month"].median())
        high = d_uk[d_uk["episodes_per_month"] >= med]
        low  = d_uk[d_uk["episodes_per_month"] < med]
        out["uk_density_strata"] = {
            "median_episodes_per_month": med,
            "high_density": {
                "n": int(len(high)),
                "meta": random_effects_meta(
                    high["beta_COI_x_cum"].values.astype(float),
                    high["se_COI_x_cum"].values.astype(float),
                ) if len(high) >= 2 else {"n_studies": int(len(high)), "note": "n<2"},
            },
            "low_density": {
                "n": int(len(low)),
                "meta": random_effects_meta(
                    low["beta_COI_x_cum"].values.astype(float),
                    low["se_COI_x_cum"].values.astype(float),
                ) if len(low) >= 2 else {"n_studies": int(len(low)), "note": "n<2"},
            },
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Cue-mix matched UK fit
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_frame(cfg: Dict[str, str], window: int, prior_window: int,
                        contingent_only: bool) -> pd.DataFrame:
    outcome_col = f"next_{window}_reuse"
    reuse = pd.read_csv(cfg["reuse_csv"], low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    file_index = build_child_utt_index(Path(cfg["tagged_csv"]))
    reuse = add_prior_local_freq(reuse, file_index, prior_window)
    file_to_child, max_len = build_file_child_map(Path(cfg["json_cache"]))
    reuse = attach_child(reuse, file_to_child, max_len).dropna(subset=["child"])
    joined = pd.read_csv(cfg["joined_csv"])
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    return df


def fit_weighted_OLS_cluster(df: pd.DataFrame, outcome_col: str,
                               weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    df = df.copy()
    df["COI_z"]                      = z(df["COI"])
    df["cumulative_cue_attempts_z"]  = z(df["cumulative_cue_attempts"])
    df["prior_local_freq_z"]         = z(df["prior_local_freq"])
    df["log_cue_freq_z"]             = z(df["log_cue_freq"])
    df["COI_x_cumulative"]           = df["COI_z"] * df["cumulative_cue_attempts_z"]
    preds = ["COI_z", "cumulative_cue_attempts_z", "COI_x_cumulative",
             "prior_local_freq_z", "log_cue_freq_z"]
    X = sm.add_constant(df[preds].astype(float), has_constant="add")
    y = df[outcome_col].astype(float)
    if weights is None:
        fit = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["cue_subtype"].astype(str).values},
        )
    else:
        fit = sm.WLS(y, X, weights=weights).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["cue_subtype"].astype(str).values},
        )
    params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + preds:
        params[p] = {
            "beta": float(fit.params[p]),
            "se":   float(fit.bse[p]),
            "t":    float(fit.tvalues[p]),
            "p":    float(fit.pvalues[p]),
        }
    return {
        "n": int(fit.nobs), "n_cues": int(df["cue_subtype"].nunique()),
        "n_children": int(df["child"].nunique()),
        "r2": float(fit.rsquared),
        "params": params,
    }


def cue_mix_matched_fit(uk_post: pd.DataFrame, manc_post: pd.DataFrame,
                          outcome_col: str) -> Dict[str, Any]:
    """Reweight UK episodes so that the cue distribution matches Manchester."""
    uk_share = uk_post["cue_subtype"].value_counts(normalize=True)
    manc_share = manc_post["cue_subtype"].value_counts(normalize=True)
    common = uk_share.index.intersection(manc_share.index)

    ratio = (manc_share[common] / uk_share[common]).rename("weight_ratio")
    weights_per_row = uk_post["cue_subtype"].map(ratio.to_dict()).fillna(0.0)

    uk_in_common = uk_post[uk_post["cue_subtype"].isin(common)].copy()
    w_in_common = weights_per_row[weights_per_row > 0].astype(float).values

    print(f"  cue-mix matching:")
    print(f"    UK cues: {len(uk_share):,}  Manchester cues: {len(manc_share):,}  common: {len(common):,}")
    print(f"    UK rows kept (cue in common): {len(uk_in_common):,}/{len(uk_post):,}")

    result: Dict[str, Any] = {
        "n_cues_uk":        int(len(uk_share)),
        "n_cues_manc":      int(len(manc_share)),
        "n_cues_common":    int(len(common)),
        "n_uk_rows_kept":   int(len(uk_in_common)),
        "weight_stats": {
            "min":    float(w_in_common.min()) if len(w_in_common) else None,
            "median": float(np.median(w_in_common)) if len(w_in_common) else None,
            "max":    float(w_in_common.max()) if len(w_in_common) else None,
        },
    }

    if len(uk_in_common) < 100:
        result["error"] = "fewer than 100 UK rows after common-cue restriction"
        return result

    # Variant 1: weighted by Manchester/UK cue share ratio (continuous)
    print(f"  Variant 1: WLS weighted by Manchester/UK share ratio ...")
    res1 = fit_weighted_OLS_cluster(uk_in_common, outcome_col, weights=w_in_common)
    result["uk_reweighted_match_manchester"] = res1

    # Variant 2: restrict to top-N cues that account for >= 90% of Manchester episodes
    cum = manc_share.sort_values(ascending=False).cumsum()
    top_cues_90 = list(cum[cum <= 0.90].index)
    if len(top_cues_90) < 5:
        top_cues_90 = list(cum.head(20).index)
    print(f"  Variant 2: restrict UK to top-{len(top_cues_90)} Manchester cues "
          f"(cumulative share ≤ 0.90) ...")
    uk_sub = uk_post[uk_post["cue_subtype"].isin(top_cues_90)].copy()
    if len(uk_sub) >= 100:
        res2 = fit_weighted_OLS_cluster(uk_sub, outcome_col, weights=None)
        result["uk_restricted_to_manc_top_cues"] = res2
        result["manc_top_cues_used"] = top_cues_90

    # Reference: UK unweighted full post-MSR
    print(f"  Reference: UK full post-MSR unweighted ...")
    res_ref = fit_weighted_OLS_cluster(uk_post, outcome_col, weights=None)
    result["uk_full_unweighted_reference"] = res_ref

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Manchester vs UK diagnostics for exposure-gate null.")
    parser.add_argument("--per_child_csv",
                         default="./output/v17c/per_child_betas_N5.csv",
                         help="From 17c (per-child β + SE).")
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--include_noncontingent", action="store_true")
    parser.add_argument("--output_dir",  default="./output/v17g")
    args = parser.parse_args()
    contingent_only = not args.include_noncontingent
    outcome_col = f"next_{args.window}_reuse"
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ───── SECTION 1: descriptives ─────
    print("=" * 70)
    print("SECTION 1 — UK vs Manchester descriptives")
    print("=" * 70)
    uk_desc = compute_per_child_descriptives(UK_CFG, args.window, contingent_only)
    manc_desc = compute_per_child_descriptives(MANC_CFG, args.window, contingent_only)
    desc_all = pd.concat([uk_desc, manc_desc], ignore_index=True)
    desc_path = out_dir / "manchester_vs_uk_descriptives.csv"
    desc_all.to_csv(desc_path, index=False)
    print(f"\n  → {desc_path}  ({len(desc_all):,} rows)")

    metrics = ["n_files", "n_utt_caregiver", "mlu_child", "mlu_caregiver",
               "types_caregiver_lemmas", "ttr_caregiver",
               "n_post_episodes", "obs_window_months",
               "episodes_per_month", "episodes_per_file"]
    group_stats = group_stats_compare(uk_desc, manc_desc, metrics)
    print("\n  Group-level comparison (UK vs Manchester, two-sided Mann-Whitney):")
    print(f"  {'metric':<28} {'UK mean':>10} {'Manc mean':>10} {'diff':>9} {'mw_p':>8}")
    for m in metrics:
        r = group_stats.get(m, {})
        if "error" in r:
            print(f"  {m:<28}  {r['error']}")
            continue
        sig = '***' if r['mw_p']<0.001 else ('**' if r['mw_p']<0.01 else ('*' if r['mw_p']<0.05 else ('†' if r['mw_p']<0.10 else '')))
        print(f"  {m:<28} {r['uk_mean']:>10.3f} {r['manc_mean']:>10.3f} {r['diff_means']:>+9.3f} {r['mw_p']:>8.4f} {sig}")

    # Cue mix Spearman
    print("\n  Cue-mix Spearman correlation (UK vs Manchester episode share):")
    uk_share   = compute_cue_mix(UK_CFG,   contingent_only)
    manc_share = compute_cue_mix(MANC_CFG, contingent_only)
    common = uk_share.index.intersection(manc_share.index)
    rho, p_rho = spearmanr(uk_share[common], manc_share[common])
    print(f"    n_common cues = {len(common)}  Spearman ρ = {rho:+.3f}  p = {p_rho:.4f}")
    # Top-25 share table
    top25_uk = uk_share.head(25).index.union(manc_share.head(25).index)
    cue_mix_df = pd.DataFrame({
        "uk_share":   uk_share.reindex(top25_uk).fillna(0.0),
        "manc_share": manc_share.reindex(top25_uk).fillna(0.0),
    }).sort_values("manc_share", ascending=False)
    cue_mix_df.to_csv(out_dir / "cue_mix_share_top.csv")
    print(f"  → {out_dir/'cue_mix_share_top.csv'}")
    # Save full Spearman result
    descrip_summary = {
        "metrics": group_stats,
        "cue_mix_spearman": {"rho": float(rho), "p": float(p_rho), "n_common": int(len(common))},
    }
    with open(out_dir / "descriptive_group_stats.json", "w", encoding="utf-8") as f:
        json.dump(descrip_summary, f, indent=2)
    print(f"  → {out_dir/'descriptive_group_stats.json'}")

    # ───── SECTION 2: density × β ─────
    print("\n" + "=" * 70)
    print("SECTION 2 — Density × β analysis")
    print("=" * 70)
    per_child_csv = Path(args.per_child_csv)
    per_child = pd.read_csv(per_child_csv)
    # Keep only UK + Manchester
    per_child = per_child[per_child["corpus_label"].isin(["English-UK", "Manchester"])].copy()
    # Merge density
    dens_cols = ["child", "n_post_episodes", "obs_window_months",
                  "episodes_per_month", "episodes_per_file", "n_files"]
    dens_long = pd.concat([uk_desc[dens_cols], manc_desc[dens_cols]], ignore_index=True)
    merged = per_child.merge(dens_long, on="child", how="inner")
    print(f"  Merged per-child×density rows: {len(merged)}")
    dens_res = density_x_beta(merged)
    with open(out_dir / "density_x_beta_results.json", "w", encoding="utf-8") as f:
        json.dump(dens_res, f, indent=2, default=str)
    print(f"  → {out_dir/'density_x_beta_results.json'}")

    print("  Pooled WLS β_i ~ episodes_per_month_z + log(n_episodes)_z + is_manchester:")
    for p, v in dens_res.get("params", {}).items():
        sig = '***' if v['p']<0.001 else ('**' if v['p']<0.01 else ('*' if v['p']<0.05 else ('†' if v['p']<0.10 else '')))
        print(f"    {p:<26}: β={v['beta']:+.4f}  SE={v['se']:.4f}  p={v['p']:.4f} {sig}")

    if "uk_only" in dens_res:
        print("  UK-only WLS β_i ~ episodes_per_month_z + log(n_episodes)_z:")
        for p, v in dens_res["uk_only"]["params"].items():
            sig = '***' if v['p']<0.001 else ('**' if v['p']<0.01 else ('*' if v['p']<0.05 else ('†' if v['p']<0.10 else '')))
            print(f"    {p:<26}: β={v['beta']:+.4f}  SE={v['se']:.4f}  p={v['p']:.4f} {sig}")

    if "uk_density_strata" in dens_res:
        s = dens_res["uk_density_strata"]
        print(f"  UK density strata (median = {s['median_episodes_per_month']:.1f} eps/mo):")
        for k in ["high_density", "low_density"]:
            meta = s[k].get("meta", {})
            if "pooled_beta_RE" in meta:
                print(f"    {k:<14} (n={s[k]['n']:>2}): β_pooled={meta['pooled_beta_RE']:+.4f}  p={meta['pooled_p_RE']:.4f}  I²={meta['I2_pct']:.1f}%")

    # ───── SECTION 3: cue-mix matched UK fit ─────
    print("\n" + "=" * 70)
    print("SECTION 3 — Cue-mix matched UK fit")
    print("=" * 70)
    uk = load_episode_frame(UK_CFG, args.window, args.prior_window, contingent_only)
    manc = load_episode_frame(MANC_CFG, args.window, args.prior_window, contingent_only)
    uk_post = uk[uk["child_age_months"] >= 24].copy()
    manc_post = manc[manc["child_age_months"] >= 24].copy()
    cue_match_res = cue_mix_matched_fit(uk_post, manc_post, outcome_col)
    with open(out_dir / "cue_mix_match_results.json", "w", encoding="utf-8") as f:
        json.dump(cue_match_res, f, indent=2, default=str)
    print(f"  → {out_dir/'cue_mix_match_results.json'}")

    for name, key in [
        ("UK full unweighted (reference)",                "uk_full_unweighted_reference"),
        ("UK reweighted to match Manchester cue share",   "uk_reweighted_match_manchester"),
        ("UK restricted to Manchester top cues (≤90%)",   "uk_restricted_to_manc_top_cues"),
    ]:
        blk = cue_match_res.get(key)
        if not blk or "params" not in blk:
            print(f"  {name}: (no result)")
            continue
        cxc = blk["params"]["COI_x_cumulative"]
        sig = '***' if cxc['p']<0.001 else ('**' if cxc['p']<0.01 else ('*' if cxc['p']<0.05 else ('†' if cxc['p']<0.10 else '')))
        print(f"  {name:<48}: n={blk['n']:,}  n_cues={blk['n_cues']}  "
              f"β(COI×cum)={cxc['beta']:+.4f}  SE={cxc['se']:.4f}  p={cxc['p']:.4f} {sig}")

    # ───── SUMMARY.md ─────
    lines: List[str] = []
    lines.append(f"# 17g Manchester diagnostics — outcome window N={args.window}\n")
    lines.append("## Section 1 — Group descriptives (UK vs Manchester, post-MSR contingent)\n")
    lines.append("| Metric | UK mean | Manc mean | Δ | MW p |")
    lines.append("|---|---|---|---|---|")
    for m in metrics:
        r = group_stats.get(m, {})
        if "error" in r: continue
        lines.append(
            f"| {m} | {r['uk_mean']:.3f} | {r['manc_mean']:.3f} | "
            f"{r['diff_means']:+.3f} | {r['mw_p']:.4f} |"
        )
    lines.append(f"\nCue-mix Spearman ρ across {len(common)} common cues = {rho:+.3f} (p = {p_rho:.4f})\n")

    lines.append("\n## Section 2 — Density × β\n")
    lines.append("Pooled WLS (UK + Manchester children) β_i ~ density + log(n) + corpus:\n")
    lines.append("| Predictor | β | SE | p |")
    lines.append("|---|---|---|---|")
    for p, v in dens_res.get("params", {}).items():
        lines.append(f"| {p} | {v['beta']:+.4f} | {v['se']:.4f} | {v['p']:.4f} |")
    if "uk_density_strata" in dens_res:
        s = dens_res["uk_density_strata"]
        lines.append("\nUK density-strata meta:\n")
        lines.append(f"- High-density UK (n={s['high_density']['n']}): "
                      f"β_pooled = {s['high_density']['meta'].get('pooled_beta_RE'):+.4f}, "
                      f"p = {s['high_density']['meta'].get('pooled_p_RE'):.4f}")
        lines.append(f"- Low-density UK (n={s['low_density']['n']}): "
                      f"β_pooled = {s['low_density']['meta'].get('pooled_beta_RE'):+.4f}, "
                      f"p = {s['low_density']['meta'].get('pooled_p_RE'):.4f}")

    lines.append("\n## Section 3 — Cue-mix matched UK fit\n")
    lines.append("| Scenario | n_eps | n_cues | β(COI×cum) | SE | p |")
    lines.append("|---|---|---|---|---|---|")
    for name, key in [
        ("UK full unweighted (reference)",                "uk_full_unweighted_reference"),
        ("UK reweighted to Manchester cue share",         "uk_reweighted_match_manchester"),
        ("UK restricted to Manchester top cues (cum≤.90)","uk_restricted_to_manc_top_cues"),
    ]:
        blk = cue_match_res.get(key)
        if not blk or "params" not in blk: continue
        cxc = blk["params"]["COI_x_cumulative"]
        lines.append(
            f"| {name} | {blk['n']:,} | {blk['n_cues']} | "
            f"{cxc['beta']:+.4f} | {cxc['se']:.4f} | {cxc['p']:.4f} |"
        )

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → {out_dir/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
