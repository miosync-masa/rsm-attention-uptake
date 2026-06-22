"""
17c_child_level_slopes.py
=========================
IMT Attention Bias Paper 2 — Step 17c:
Child-level decomposition of the exposure-gate effect

Builds on 17b_exposure_gate_test.py. The 17b result was mixed:
  Brown β(COI × cumulative) = +0.05*** , English-UK = +0.024**,
  Manchester ≈ 0 (or sig negative in MixedLM).
This script answers two follow-up questions:

  Q1  How big is between-child heterogeneity in the slope, relative to
      the fixed (mean) slope?
        → Random-slope MixedLM in the multi-level model
          (1 + COI_z * cumulative_cue_attempts_z | child)
        → ICC and τ²(β slope) per corpus.

  Q2  Is Manchester's null/negative average pulled by a few children, or
      is the negative direction systemic?
        → Per-child OLS fits of M_new
        → Random-effects meta-analysis across all children
          (DerSimonian-Laird τ², I²)
        → Scatter: child β vs child sample size, colored by corpus.

Supplementary
-------------
* S6 — Child-level moderator regression:
        β_i ~ MLU_child + MLU_caregiver + log(n_episodes_post) + corpus
* S7 — Manchester deep dive: per-child β table sorted by β, plus
        diagnostic descriptives.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17c/)
----------------------
* main_results_N{N}.json
* per_child_betas_N{N}.csv
* child_beta_vs_n_scatter.png
* manchester_deep_dive.csv
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / child-level v1 | 2026-06-21
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
    import statsmodels.formula.api as smf
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":          "Brown",
        "reuse_csv":      "./output/v16/English_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English_tokens_tagged.csv",
        "joined_csv":     "./output/v11_runA/English_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English",
        "utterances_csv": "./output/English_utterances.csv",
    },
    "English-Manchester": {
        "label":          "Manchester",
        "reuse_csv":      "./output/v16/English-Manchester_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English-Manchester_tokens_tagged.csv",
        "joined_csv":     "./output/v11/English-Manchester_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English-Manchester",
        "utterances_csv": "./output/English-Manchester_utterances.csv",
    },
    "English-UK": {
        "label":          "English-UK",
        "reuse_csv":      "./output/v16/English-UK_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English-UK_tokens_tagged.csv",
        "joined_csv":     "./output/v11_runA/English-UK_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English-UK",
        "utterances_csv": "./output/English-UK_utterances.csv",
    },
}

PREDICTORS_MAIN = [
    "COI_z", "cumulative_cue_attempts_z", "COI_x_cumulative",
    "prior_local_freq_z", "log_cue_freq_z",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirror 17b)
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


def build_child_utt_index(tagged_csv: Path) -> Dict[str, Tuple[np.ndarray, List[set]]]:
    print(f"  Building child utt index from {tagged_csv} ...")
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
    print(f"    files indexed: {len(file_index):,}")
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


def add_cumulative_cue_attempts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Multi-level model with random slopes (Q1)
# ─────────────────────────────────────────────────────────────────────────────

def fit_random_slopes(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    MixedLM with random intercept + random slopes of
    (COI_z, cumulative_cue_attempts_z, COI_x_cumulative) by child.

    Falls back to random intercept only if random slopes don't converge.
    """
    fixed_formula = (
        f"{outcome_col} ~ COI_z + cumulative_cue_attempts_z + COI_x_cumulative + "
        "prior_local_freq_z + log_cue_freq_z"
    )
    re_formula = "~ COI_z + cumulative_cue_attempts_z + COI_x_cumulative"

    out: Dict[str, Any] = {"fixed_formula": fixed_formula, "re_formula": re_formula}
    try:
        mlm = smf.mixedlm(
            fixed_formula, data=df, groups=df["child"], re_formula=re_formula,
        ).fit(method="lbfgs", reml=True)
        out["model"] = "random_slopes"
        out["converged"] = bool(getattr(mlm, "converged", True))
    except Exception as exc:
        # Fall back: random intercept only
        try:
            mlm = smf.mixedlm(
                fixed_formula, data=df, groups=df["child"],
            ).fit(method="lbfgs", reml=True)
            out["model"] = f"random_intercept_only (fallback: {type(exc).__name__})"
            out["converged"] = bool(getattr(mlm, "converged", True))
        except Exception as exc2:
            return {"error": f"both random-slopes and random-intercept failed: {exc2}"}

    # Fixed effects
    fixed_names = ["Intercept", "COI_z", "cumulative_cue_attempts_z",
                   "COI_x_cumulative", "prior_local_freq_z", "log_cue_freq_z"]
    out["fixed_effects"] = {
        p: {
            "beta": float(mlm.params[p]),
            "se":   float(mlm.bse[p]),
            "z":    float(mlm.tvalues[p]),
            "p":    float(mlm.pvalues[p]),
        } for p in fixed_names if p in mlm.params
    }

    # Random effect variance components and ICC
    try:
        cov_re = mlm.cov_re
        out["random_effects_cov_matrix"] = cov_re.to_dict()
        # Extract variance for the slopes
        var_intercept = float(cov_re.iloc[0, 0]) if cov_re.shape[0] > 0 else None
        slope_vars: Dict[str, float] = {}
        for name in re_formula.replace("~", "").split("+"):
            name = name.strip()
            if name in cov_re.index:
                slope_vars[name] = float(cov_re.loc[name, name])
        out["random_intercept_variance"] = var_intercept
        out["random_slope_variances"] = slope_vars
        # ICC (intercept only)
        resid_var = float(mlm.scale)
        out["residual_variance"] = resid_var
        if var_intercept is not None and (var_intercept + resid_var) > 0:
            out["icc_intercept"] = float(var_intercept / (var_intercept + resid_var))
    except Exception as exc:
        out["random_effects_error"] = str(exc)

    out["n"] = int(mlm.nobs)
    out["n_children"] = int(df["child"].nunique())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-child OLS fits (Q2)
# ─────────────────────────────────────────────────────────────────────────────

def fit_per_child(df: pd.DataFrame, outcome_col: str,
                   min_episodes: int = 200) -> pd.DataFrame:
    """
    Fit M_new individually for each child with >= min_episodes rows.
    Returns DataFrame with columns:
      child, corpus_label, n_episodes, n_cues,
      beta_COI_x_cumulative, se_COI_x_cumulative, p_COI_x_cumulative
    """
    rows: List[Dict[str, Any]] = []
    for child_id, g in df.groupby("child"):
        if len(g) < min_episodes:
            continue
        # Need variance in COI and cumulative_cue_attempts within this child
        if g["COI"].std() == 0 or g["cumulative_cue_attempts"].std() == 0:
            continue
        gg = g.copy()
        gg["COI_z_local"] = z(gg["COI"])
        gg["cum_z_local"] = z(gg["cumulative_cue_attempts"])
        gg["prior_z_local"] = z(gg["prior_local_freq"])
        gg["logfreq_z_local"] = z(gg["log_cue_freq"])
        gg["COI_x_cum_local"] = gg["COI_z_local"] * gg["cum_z_local"]
        preds = ["COI_z_local", "cum_z_local", "COI_x_cum_local",
                 "prior_z_local", "logfreq_z_local"]
        X = sm.add_constant(gg[preds].astype(float), has_constant="add")
        y = gg[outcome_col].astype(float)
        try:
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
            rows.append({
                "child":            child_id,
                "n_episodes":       int(len(gg)),
                "n_cues":           int(gg["cue_subtype"].nunique()),
                "outcome_mean":     float(gg[outcome_col].mean()),
                "beta_COI_x_cum":   float(fit.params["COI_x_cum_local"]),
                "se_COI_x_cum":     float(fit.bse["COI_x_cum_local"]),
                "p_COI_x_cum":      float(fit.pvalues["COI_x_cum_local"]),
                "beta_COI":         float(fit.params["COI_z_local"]),
                "se_COI":           float(fit.bse["COI_z_local"]),
                "beta_cum":         float(fit.params["cum_z_local"]),
                "se_cum":           float(fit.bse["cum_z_local"]),
                "r2":               float(fit.rsquared),
            })
        except Exception as exc:
            rows.append({
                "child":            child_id,
                "n_episodes":       int(len(gg)),
                "n_cues":           int(gg["cue_subtype"].nunique()),
                "error":            str(exc),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DerSimonian-Laird random-effects meta-analysis
# ─────────────────────────────────────────────────────────────────────────────

def random_effects_meta(betas: List[float], ses: List[float]) -> Dict[str, float]:
    betas = np.array(betas, dtype=float)
    ses   = np.array(ses,   dtype=float)
    n = len(betas)
    if n < 2:
        return {"error": "fewer than 2 effect sizes"}

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
    se_re   = float(np.sqrt(1.0 / np.sum(w_re)))
    z_re    = beta_re / se_re if se_re > 0 else float("nan")
    # two-sided p
    from scipy.stats import norm
    p_re = float(2 * (1.0 - norm.cdf(abs(z_re)))) if not math.isnan(z_re) else float("nan")

    return {
        "n_studies":       int(n),
        "Q":               Q,
        "df":              int(df),
        "tau2":            tau2,
        "I2_pct":          I2,
        "pooled_beta_RE":  beta_re,
        "pooled_se_RE":    se_re,
        "pooled_z_RE":     z_re,
        "pooled_p_RE":     p_re,
        "pooled_beta_FE":  beta_fe,
        "pooled_se_FE":    float(np.sqrt(1.0 / sum_w)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Child-level moderators (S6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_child_moderators(utterances_csv: Path,
                              file_to_child: Dict[str, str]) -> pd.DataFrame:
    """Per-child mean MLU (child + caregiver) and verbosity."""
    use_cols = ["file", "is_child", "is_caregiver", "num_tokens"]
    u = pd.read_csv(utterances_csv, usecols=use_cols, low_memory=False, dtype={"file": str})
    u["child"] = u["file"].astype(str).map(file_to_child)
    miss = u["child"].isna()
    if miss.any():
        # Try zero-padding
        if file_to_child:
            max_len = max(len(k) for k in file_to_child)
            padded = u.loc[miss, "file"].astype(str).str.zfill(max_len)
            u.loc[miss, "child"] = padded.map(file_to_child)
    u = u.dropna(subset=["child"]).copy()
    rows: List[Dict] = []
    for child_id, g in u.groupby("child"):
        gc = g[g["is_child"]]
        ga = g[g["is_caregiver"]]
        rows.append({
            "child":            child_id,
            "n_utt_child":      int(len(gc)),
            "n_utt_caregiver":  int(len(ga)),
            "mlu_child":        float(gc["num_tokens"].mean()) if len(gc) else float("nan"),
            "mlu_caregiver":    float(ga["num_tokens"].mean()) if len(ga) else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_corpus(language: str, reuse_csv: Path, tagged_csv: Path,
                   joined_csv: Path, json_cache: Path,
                   utterances_csv: Path,
                   window: int, prior_window: int,
                   contingent_only: bool,
                   min_episodes_per_child: int,
                   output_dir: Path) -> Dict[str, Any]:
    print(f"\n=== {language} ===")
    outcome_col = f"next_{window}_reuse"

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    print(f"  expanded rows: {len(reuse):,}")
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
        print(f"  contingent rows: {len(reuse):,}")
    if outcome_col not in reuse.columns:
        return {"error": f"missing {outcome_col}"}

    file_index = build_child_utt_index(tagged_csv)
    reuse = add_prior_local_freq(reuse, file_index, prior_window)

    file_to_child, max_len = build_file_child_map(json_cache)
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()

    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    cols = ["cue_subtype", "COI", "log_cue_freq"]
    df = reuse.merge(joined[cols], on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    df = add_cumulative_cue_attempts(df)

    df["age_post"] = (df["child_age_months"] >= 24).astype(int)
    df_post = df[df["age_post"] == 1].copy()
    print(f"  post-MSR rows: {len(df_post):,}    children post: {df_post['child'].nunique()}")

    # Prep z columns (post-MSR-wide standardization)
    df_post["COI_z"]                      = z(df_post["COI"])
    df_post["cumulative_cue_attempts_z"]  = z(df_post["cumulative_cue_attempts"])
    df_post["prior_local_freq_z"]         = z(df_post["prior_local_freq"])
    df_post["log_cue_freq_z"]             = z(df_post["log_cue_freq"])
    df_post["COI_x_cumulative"]           = df_post["COI_z"] * df_post["cumulative_cue_attempts_z"]

    result: Dict[str, Any] = {
        "language":           language,
        "outcome_col":        outcome_col,
        "window":             window,
        "prior_window":       prior_window,
        "n_post":             int(len(df_post)),
        "n_children_post":    int(df_post["child"].nunique()),
        "n_cues_post":        int(df_post["cue_subtype"].nunique()),
    }

    # Q1 — random slopes
    print("  fit random-slopes MixedLM ...")
    result["random_slopes"] = fit_random_slopes(df_post, outcome_col)

    # Q2 — per-child OLS
    print(f"  per-child OLS (min episodes = {min_episodes_per_child}) ...")
    per_child = fit_per_child(df_post, outcome_col, min_episodes=min_episodes_per_child)
    per_child["corpus_label"] = CANONICAL_CORPORA.get(language, {}).get("label", language)
    per_child["language"]     = language
    result["per_child"] = per_child.to_dict(orient="records")

    # S6 — child moderators
    if utterances_csv.exists():
        try:
            mods = compute_child_moderators(utterances_csv, file_to_child)
            if len(mods):
                merged = per_child.merge(mods, on="child", how="left")
                result["per_child_with_moderators"] = merged.to_dict(orient="records")
        except Exception as exc:
            result["moderator_error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-corpus aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_meta_and_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Collect per-child rows
    per_child_records: List[Dict[str, Any]] = []
    for lang, res in all_results.items():
        if lang == "_meta":
            continue
        for row in res.get("per_child", []):
            if "beta_COI_x_cum" not in row:
                continue
            per_child_records.append(row)
    df = pd.DataFrame(per_child_records)
    if df.empty:
        out["error"] = "no per-child fits"
        return out

    out["per_child_table"] = df.to_dict(orient="records")

    # Meta-analysis: pooled across ALL children, and per-corpus
    def _meta(sub: pd.DataFrame) -> Dict[str, Any]:
        sub2 = sub.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"]).copy()
        if len(sub2) < 2:
            return {"error": "n < 2"}
        return random_effects_meta(
            sub2["beta_COI_x_cum"].tolist(),
            sub2["se_COI_x_cum"].tolist(),
        )

    out["meta_pooled_all_children"] = _meta(df)
    out["meta_per_corpus"] = {}
    for corp_label, sub in df.groupby("corpus_label"):
        out["meta_per_corpus"][corp_label] = _meta(sub)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_scatter(per_child_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping scatter plot")
        return
    if per_child_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = {"Brown": "#d62728", "Manchester": "#1f77b4", "English-UK": "#2ca02c"}
    for label, g in per_child_df.groupby("corpus_label"):
        ax.errorbar(
            g["n_episodes"], g["beta_COI_x_cum"],
            yerr=1.96 * g["se_COI_x_cum"], fmt="o", capsize=2,
            ms=6, alpha=0.85, label=label, color=palette.get(label, "gray"),
        )
        for _, row in g.iterrows():
            ax.text(row["n_episodes"] * 1.02, row["beta_COI_x_cum"],
                     str(row["child"])[:8], fontsize=6, color="gray", alpha=0.7)
    ax.axhline(0, color="black", lw=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("n_episodes per child (post-MSR, log scale)")
    ax.set_ylabel("β(COI × cumulative_cue_attempts), per-child OLS")
    ax.set_title("S2 / Q2: per-child slope of COI × cumulative_exposure (post-MSR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Child-level decomposition of exposure-gate effect.")
    p.add_argument("--reuse_csv",   default=None)
    p.add_argument("--tagged_csv",  default=None)
    p.add_argument("--joined_csv",  default=None)
    p.add_argument("--json_cache",  default=None)
    p.add_argument("--utterances_csv", default=None)
    p.add_argument("--language",    default=None)
    p.add_argument("--output_dir",  default="./output/v17c")
    p.add_argument("--window",      type=int, default=5)
    p.add_argument("--prior_window", type=int, default=20)
    p.add_argument("--min_episodes_per_child", type=int, default=200)
    p.add_argument("--include_noncontingent", action="store_true")
    p.add_argument("--batch", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    contingent_only = not args.include_noncontingent

    if args.batch:
        targets = [
            (lang, cfg["reuse_csv"], cfg["tagged_csv"], cfg["joined_csv"],
             cfg["json_cache"], cfg["utterances_csv"])
            for lang, cfg in CANONICAL_CORPORA.items()
        ]
    else:
        if not all([args.language, args.reuse_csv, args.tagged_csv,
                    args.joined_csv, args.json_cache, args.utterances_csv]):
            sys.exit("ERROR: provide all of --language/--reuse_csv/--tagged_csv/"
                     "--joined_csv/--json_cache/--utterances_csv, or use --batch.")
        targets = [(args.language, args.reuse_csv, args.tagged_csv,
                    args.joined_csv, args.json_cache, args.utterances_csv)]

    all_results: Dict[str, Any] = {"_meta": {
        "window": args.window, "prior_window": args.prior_window,
        "contingent_only": contingent_only,
        "min_episodes_per_child": args.min_episodes_per_child,
    }}

    for lang, rc, tc, jc, jcache, ucsv in targets:
        rcp, tcp, jcp, jcache_p, ucsv_p = map(Path, [rc, tc, jc, jcache, ucsv])
        missing = [p for p in [rcp, tcp, jcp, ucsv_p] if not p.exists()]
        if missing:
            print(f"  SKIP {lang}: missing inputs {missing}")
            continue
        res = analyze_corpus(
            lang, rcp, tcp, jcp, jcache_p, ucsv_p,
            window=args.window, prior_window=args.prior_window,
            contingent_only=contingent_only,
            min_episodes_per_child=args.min_episodes_per_child,
            output_dir=out_dir,
        )
        all_results[lang] = res

    # Cross-corpus meta + per-child table
    agg = aggregate_meta_and_summary(all_results)
    all_results["_aggregate"] = agg

    # Persist per-child long table
    per_child_df = pd.DataFrame(agg.get("per_child_table", []))
    per_child_csv = out_dir / f"per_child_betas_N{args.window}.csv"
    per_child_df.to_csv(per_child_csv, index=False)
    print(f"\n  → {per_child_csv}  ({len(per_child_df):,} rows)")

    # Save full JSON
    combined_json = out_dir / f"main_results_N{args.window}.json"
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  → {combined_json}")

    # Scatter
    if not per_child_df.empty:
        scatter_path = out_dir / "child_beta_vs_n_scatter.png"
        make_scatter(per_child_df, scatter_path)

    # Manchester deep dive
    manc_rows = per_child_df[per_child_df["corpus_label"] == "Manchester"].copy()
    if not manc_rows.empty:
        manc_rows = manc_rows.sort_values("beta_COI_x_cum")
        deep_path = out_dir / "manchester_deep_dive.csv"
        manc_rows.to_csv(deep_path, index=False)
        print(f"  → {deep_path}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# 17c child-level slopes — outcome window N={args.window}\n")
    lines.append("## Fixed slope β(COI × cumulative_cue_attempts) per corpus (random-slope MixedLM)\n")
    lines.append("| Corpus | β fixed | SE | p | n | n_children | model | ICC(intercept) | Var(slope_COIxCum) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for lang, res in all_results.items():
        if lang in ("_meta", "_aggregate"):
            continue
        label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
        rs = res.get("random_slopes", {})
        fe = rs.get("fixed_effects", {}).get("COI_x_cumulative", {})
        if not fe:
            lines.append(f"| {label} | (missing) | | | | | | | |")
            continue
        slope_vars = rs.get("random_slope_variances", {})
        var_slope = slope_vars.get("COI_x_cumulative", float("nan"))
        lines.append(
            f"| {label} | {fe.get('beta'):+.4f} | {fe.get('se'):.4f} | {fe.get('p'):.4f} | "
            f"{rs.get('n')} | {rs.get('n_children')} | {rs.get('model')} | "
            f"{rs.get('icc_intercept', float('nan')):.3f} | {var_slope:.4f} |"
        )

    # Meta-analysis section
    lines.append("\n## Random-effects meta-analysis of per-child β(COI × cumulative_cue_attempts)\n")
    lines.append("| Scope | n_studies | pooled β (RE) | SE | p | τ² | I² (%) |")
    lines.append("|---|---|---|---|---|---|---|")
    pooled = agg.get("meta_pooled_all_children", {})
    if "pooled_beta_RE" in pooled:
        lines.append(
            f"| All children | {pooled['n_studies']} | {pooled['pooled_beta_RE']:+.4f} | "
            f"{pooled['pooled_se_RE']:.4f} | {pooled['pooled_p_RE']:.4f} | "
            f"{pooled['tau2']:.4f} | {pooled['I2_pct']:.1f} |"
        )
    for label, m in agg.get("meta_per_corpus", {}).items():
        if "pooled_beta_RE" in m:
            lines.append(
                f"| {label} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                f"{m['tau2']:.4f} | {m['I2_pct']:.1f} |"
            )

    summary_md = out_dir / "SUMMARY.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {summary_md}")

    # Console pass-criterion print
    print("\n========== 17c summary ==========")
    print("\nFixed slope β(COI × cumulative_cue_attempts) — random-slope MixedLM:")
    for lang, res in all_results.items():
        if lang in ("_meta", "_aggregate"):
            continue
        label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
        fe = res.get("random_slopes", {}).get("fixed_effects", {}).get("COI_x_cumulative", {})
        if fe:
            print(f"  {label:<12}: β={fe['beta']:+.4f}  SE={fe['se']:.4f}  p={fe['p']:.4f}")

    print("\nRandom-effects meta-analysis (per-child β):")
    pooled = agg.get("meta_pooled_all_children", {})
    if "pooled_beta_RE" in pooled:
        print(f"  All children (n={pooled['n_studies']}): "
              f"β_pooled = {pooled['pooled_beta_RE']:+.4f}  SE = {pooled['pooled_se_RE']:.4f}  "
              f"p = {pooled['pooled_p_RE']:.4f}  τ² = {pooled['tau2']:.4f}  I² = {pooled['I2_pct']:.1f}%")
    for label, m in agg.get("meta_per_corpus", {}).items():
        if "pooled_beta_RE" in m:
            print(f"  {label:<12} (n={m['n_studies']}): "
                  f"β_pooled = {m['pooled_beta_RE']:+.4f}  SE = {m['pooled_se_RE']:.4f}  "
                  f"p = {m['pooled_p_RE']:.4f}  τ² = {m['tau2']:.4f}  I² = {m['I2_pct']:.1f}%")


if __name__ == "__main__":
    main()
