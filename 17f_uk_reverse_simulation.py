"""
17f_uk_reverse_simulation.py
============================
IMT Attention Bias Paper 2 — Step 17f: UK reverse simulation.

Decisive test of the window-as-gate hypothesis (introduced in 17d, partially
confirmed in 17e):

  PROBLEM
  -------
  17e showed that when we restrict to children with > 12 mo of post-MSR
  observation, β(COI × cumulative_cue_attempts) ≈ +0.036 (pooled), but
  Manchester contributes 0 children at that threshold. A reviewer could
  argue "this is a corpus effect — Brown / English-UK simply have richer
  data, not because the window is the gate."

  SIMULATION
  ----------
  For each English-UK child with a long observation window, we artificially
  restrict their post-MSR episodes to the age band 24 ≤ age < 36 months,
  matching Manchester's observational structure, and refit:

      reuse_nextN ~ COI_z × cumulative_cue_attempts_z
                  + prior_local_freq_z + log_cue_freq_z
      (cluster-robust SE on cue)

  TWO VARIANTS
  ------------
    A. lifetime cumulative — cumulative_cue_attempts is the lifetime
       count (same as 17b/17c), only the episode rows are restricted.
       This is the most direct apples-to-apples comparison with
       Manchester, whose cumulative counter also runs lifetime.
    B. window-local cumulative — recompute cumulative_cue_attempts
       within the 24–36 mo window only (counter resets to 0 at first
       in-window use). This is the most conservative simulation of
       "what if Manchester were the only data we ever saw for these
       children".

  PASS CRITERION
  --------------
    UK β(COI × cumulative) drops from its full-window value (~ +0.024)
    to ≈ 0 (|β| < 0.010 OR p > 0.10) after age-band restriction.
    → corpus is NOT the confound; window IS the gate.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17f/)
----------------------
* uk_reverse_results_N{N}.json
* uk_reverse_summary_N{N}.csv
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / UK reverse v1 | 2026-06-21
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
    from scipy.stats import norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


UK_CFG = {
    "label":        "English-UK",
    "reuse_csv":    "./output/v16/English-UK_episodes_with_reuse.csv",
    "tagged_csv":   "./output/English-UK_tokens_tagged.csv",
    "joined_csv":   "./output/v11_runA/English-UK_r_plus_joined.csv",
    "json_cache":   "./output/json_cache/English-UK",
}

MANC_CFG = {
    "label":        "Manchester",
    "reuse_csv":    "./output/v16/English-Manchester_episodes_with_reuse.csv",
    "tagged_csv":   "./output/English-Manchester_tokens_tagged.csv",
    "joined_csv":   "./output/v11/English-Manchester_r_plus_joined.csv",
    "json_cache":   "./output/json_cache/English-Manchester",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared)
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
# Episode loader (produces per-corpus episode-level frame with child id,
# prior_local_freq, COI, log_cue_freq; cumulative_cue_attempts computed
# either lifetime or window-local depending on variant)
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_frame(cfg: Dict[str, str], window: int, prior_window: int,
                        contingent_only: bool) -> pd.DataFrame:
    outcome_col = f"next_{window}_reuse"
    reuse = pd.read_csv(cfg["reuse_csv"], low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    if outcome_col not in reuse.columns:
        sys.exit(f"  ERROR: missing {outcome_col} in {cfg['reuse_csv']}")

    file_index = build_child_utt_index(Path(cfg["tagged_csv"]))
    reuse = add_prior_local_freq(reuse, file_index, prior_window)

    file_to_child, max_len = build_file_child_map(Path(cfg["json_cache"]))
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()

    joined = pd.read_csv(cfg["joined_csv"])
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    return df


def add_cumulative_lifetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    return df


def add_cumulative_window_local(df: pd.DataFrame) -> pd.DataFrame:
    """Recomputes cumulative within the *current* df only.
    For the UK reverse with df already filtered to 24–36 mo, the counter
    resets to 0 at the first in-window use."""
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

PREDS = ["COI_z", "cumulative_cue_attempts_z", "COI_x_cumulative",
          "prior_local_freq_z", "log_cue_freq_z"]


def add_z_and_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["COI_z"]                      = z(df["COI"])
    df["cumulative_cue_attempts_z"]  = z(df["cumulative_cue_attempts"])
    df["prior_local_freq_z"]         = z(df["prior_local_freq"])
    df["log_cue_freq_z"]             = z(df["log_cue_freq"])
    df["COI_x_cumulative"]           = df["COI_z"] * df["cumulative_cue_attempts_z"]
    return df


def fit_corpus_ols_cluster(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    X = sm.add_constant(df[PREDS].astype(float), has_constant="add")
    y = df[outcome_col].astype(float)
    fit = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["cue_subtype"].astype(str).values},
    )
    params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + PREDS:
        params[p] = {
            "beta": float(fit.params[p]),
            "se":   float(fit.bse[p]),
            "t":    float(fit.tvalues[p]),
            "p":    float(fit.pvalues[p]),
            "ci95_low":  float(fit.conf_int().loc[p, 0]),
            "ci95_high": float(fit.conf_int().loc[p, 1]),
        }
    return {
        "n": int(fit.nobs),
        "n_cues": int(df["cue_subtype"].nunique()),
        "n_children": int(df["child"].nunique()),
        "r2": float(fit.rsquared),
        "params": params,
    }


def fit_per_child(df: pd.DataFrame, outcome_col: str,
                   min_episodes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for child_id, g in df.groupby("child"):
        if len(g) < min_episodes:
            continue
        gg = g.copy()
        if gg["COI"].std() == 0 or gg["cumulative_cue_attempts"].std() == 0:
            continue
        gg["COI_z_local"]       = z(gg["COI"])
        gg["cum_z_local"]       = z(gg["cumulative_cue_attempts"])
        gg["prior_z_local"]     = z(gg["prior_local_freq"])
        gg["logfreq_z_local"]   = z(gg["log_cue_freq"])
        gg["COI_x_cum_local"]   = gg["COI_z_local"] * gg["cum_z_local"]
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
                "age_min":          float(gg["child_age_months"].min()),
                "age_max":          float(gg["child_age_months"].max()),
                "obs_window":       float(gg["child_age_months"].max() - gg["child_age_months"].min()),
                "beta_COI_x_cum":   float(fit.params["COI_x_cum_local"]),
                "se_COI_x_cum":     float(fit.bse["COI_x_cum_local"]),
                "p_COI_x_cum":      float(fit.pvalues["COI_x_cum_local"]),
                "r2":               float(fit.rsquared),
            })
        except Exception as exc:
            rows.append({
                "child":      child_id,
                "n_episodes": int(len(gg)),
                "error":      str(exc),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UK reverse-simulation of window-as-gate.")
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--manchester_age_min", type=float, default=24.0)
    parser.add_argument("--manchester_age_max", type=float, default=36.0)
    parser.add_argument("--min_episodes_per_child", type=int, default=200)
    parser.add_argument("--include_noncontingent", action="store_true")
    parser.add_argument("--output_dir",  default="./output/v17f")
    args = parser.parse_args()
    contingent_only = not args.include_noncontingent
    outcome_col = f"next_{args.window}_reuse"
    age_lo, age_hi = args.manchester_age_min, args.manchester_age_max

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ───── Load UK & Manchester episode frames ─────
    print("\nLoading UK episodes ...")
    uk = load_episode_frame(UK_CFG, args.window, args.prior_window, contingent_only)
    print(f"  UK total contingent post-merge rows: {len(uk):,}")

    print("\nLoading Manchester episodes ...")
    manc = load_episode_frame(MANC_CFG, args.window, args.prior_window, contingent_only)
    print(f"  Manchester total contingent post-merge rows: {len(manc):,}")

    # ───── Compute lifetime cumulative for both ─────
    uk = add_cumulative_lifetime(uk)
    manc = add_cumulative_lifetime(manc)

    # Restrict to age >= 24 (post-MSR) for the full-window UK reference
    uk_post = uk[uk["child_age_months"] >= 24].copy()
    manc_post = manc[manc["child_age_months"] >= 24].copy()

    # ───── UK reference (full post-MSR window) ─────
    print(f"\n=== UK reference (full post-MSR window) ===")
    uk_post_z = add_z_and_interaction(uk_post)
    uk_full_corpus = fit_corpus_ols_cluster(uk_post_z, outcome_col)
    print(f"  n={uk_full_corpus['n']:,}  n_children={uk_full_corpus['n_children']}  "
          f"β(COI×cum)={uk_full_corpus['params']['COI_x_cumulative']['beta']:+.4f}  "
          f"p={uk_full_corpus['params']['COI_x_cumulative']['p']:.4f}")
    uk_full_per_child = fit_per_child(uk_post, outcome_col, args.min_episodes_per_child)

    # ───── Manchester reference (full post-MSR) ─────
    print(f"\n=== Manchester reference (full post-MSR window) ===")
    manc_post_z = add_z_and_interaction(manc_post)
    manc_corpus = fit_corpus_ols_cluster(manc_post_z, outcome_col)
    print(f"  n={manc_corpus['n']:,}  n_children={manc_corpus['n_children']}  "
          f"β(COI×cum)={manc_corpus['params']['COI_x_cumulative']['beta']:+.4f}  "
          f"p={manc_corpus['params']['COI_x_cumulative']['p']:.4f}")
    manc_per_child = fit_per_child(manc_post, outcome_col, args.min_episodes_per_child)

    # ───── UK reverse simulation: restrict to age [24, 36) ─────
    print(f"\n=== UK reverse simulation: restrict to age [{age_lo}, {age_hi}) ===")
    uk_restricted = uk[(uk["child_age_months"] >= age_lo) & (uk["child_age_months"] < age_hi)].copy()
    print(f"  UK rows in {age_lo}-{age_hi}: {len(uk_restricted):,}  "
          f"(children with any row: {uk_restricted['child'].nunique()})")

    # Variant A: keep lifetime cumulative (already computed)
    uk_A_z = add_z_and_interaction(uk_restricted)
    print("  Variant A — lifetime cumulative (filter rows only, keep counts):")
    uk_A_corpus = fit_corpus_ols_cluster(uk_A_z, outcome_col)
    uk_A_per_child = fit_per_child(uk_restricted, outcome_col, args.min_episodes_per_child)
    cxc = uk_A_corpus["params"]["COI_x_cumulative"]
    print(f"    corpus: n={uk_A_corpus['n']:,}  n_children={uk_A_corpus['n_children']}  "
          f"β={cxc['beta']:+.4f}  SE={cxc['se']:.4f}  p={cxc['p']:.4f}")

    # Variant B: window-local cumulative (recompute on restricted slice)
    uk_B_base = uk_restricted.drop(columns=["cumulative_cue_attempts"]).copy()
    uk_B_base = add_cumulative_window_local(uk_B_base)
    uk_B_z = add_z_and_interaction(uk_B_base)
    print("  Variant B — window-local cumulative (counter resets to 0 in [24,36)):")
    uk_B_corpus = fit_corpus_ols_cluster(uk_B_z, outcome_col)
    uk_B_per_child = fit_per_child(uk_B_base, outcome_col, args.min_episodes_per_child)
    cxc = uk_B_corpus["params"]["COI_x_cumulative"]
    print(f"    corpus: n={uk_B_corpus['n']:,}  n_children={uk_B_corpus['n_children']}  "
          f"β={cxc['beta']:+.4f}  SE={cxc['se']:.4f}  p={cxc['p']:.4f}")

    # ───── Meta of per-child β within each scenario ─────
    def _meta(df: pd.DataFrame) -> Dict[str, Any]:
        d = df.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"])
        if len(d) < 2:
            return {"n_studies": int(len(d)),
                     "note": "n<2; reporting first child β if any",
                     **({"beta_single": float(d["beta_COI_x_cum"].iloc[0]),
                          "se_single":   float(d["se_COI_x_cum"].iloc[0])}
                         if len(d) == 1 else {})}
        return random_effects_meta(
            d["beta_COI_x_cum"].values.astype(float),
            d["se_COI_x_cum"].values.astype(float),
        )

    meta_uk_full      = _meta(uk_full_per_child)
    meta_uk_A         = _meta(uk_A_per_child)
    meta_uk_B         = _meta(uk_B_per_child)
    meta_manc         = _meta(manc_per_child)

    print("\n=== Per-child meta (random-effects) ===")
    for name, m in [("UK full post-MSR", meta_uk_full),
                     ("UK 24-36 (A: lifetime cumulative)", meta_uk_A),
                     ("UK 24-36 (B: window-local cumulative)", meta_uk_B),
                     ("Manchester full post-MSR", meta_manc)]:
        if "pooled_beta_RE" in m:
            print(f"  {name:<42}: n={m['n_studies']:>2}  β={m['pooled_beta_RE']:+.4f}  "
                  f"SE={m['pooled_se_RE']:.4f}  p={m['pooled_p_RE']:.4f}  "
                  f"τ²={m['tau2']:.4f}  I²={m['I2_pct']:.1f}%")
        else:
            print(f"  {name:<42}: {m.get('note', m.get('error'))}")

    # ───── Pass/fail verdict ─────
    full_b = uk_full_corpus["params"]["COI_x_cumulative"]
    A_b    = uk_A_corpus["params"]["COI_x_cumulative"]
    B_b    = uk_B_corpus["params"]["COI_x_cumulative"]
    manc_b = manc_corpus["params"]["COI_x_cumulative"]

    def _pass(beta_full, p_full, beta_restricted, p_restricted):
        return (abs(beta_restricted) < 0.010) or (p_restricted > 0.10 and beta_restricted < beta_full * 0.5)

    verdict_A = _pass(full_b["beta"], full_b["p"], A_b["beta"], A_b["p"])
    verdict_B = _pass(full_b["beta"], full_b["p"], B_b["beta"], B_b["p"])
    overall_pass = verdict_A or verdict_B
    print(f"\n========== UK reverse simulation verdict ==========")
    print(f"  Pass criterion: |β_restricted| < 0.010 OR (p > 0.10 AND β shrinks > 50%)")
    print(f"  Variant A: {'PASS' if verdict_A else 'FAIL'}  (β: {full_b['beta']:+.4f} → {A_b['beta']:+.4f})")
    print(f"  Variant B: {'PASS' if verdict_B else 'FAIL'}  (β: {full_b['beta']:+.4f} → {B_b['beta']:+.4f})")
    print(f"  Manchester reference β = {manc_b['beta']:+.4f}  p = {manc_b['p']:.4f}")

    # ───── Persist ─────
    results = {
        "_meta": {
            "window":            args.window,
            "prior_window":      args.prior_window,
            "manchester_age_min": age_lo,
            "manchester_age_max": age_hi,
            "min_episodes_per_child": args.min_episodes_per_child,
            "contingent_only":   contingent_only,
        },
        "uk_full_post_msr":      uk_full_corpus,
        "uk_reverse_A_lifetime": uk_A_corpus,
        "uk_reverse_B_local":    uk_B_corpus,
        "manchester_post_msr":   manc_corpus,
        "meta": {
            "uk_full_post_msr":      meta_uk_full,
            "uk_reverse_A_lifetime": meta_uk_A,
            "uk_reverse_B_local":    meta_uk_B,
            "manchester_post_msr":   meta_manc,
        },
        "verdict": {
            "variant_A_pass": bool(verdict_A),
            "variant_B_pass": bool(verdict_B),
            "overall_pass":   bool(overall_pass),
            "beta_uk_full":   float(full_b["beta"]),
            "p_uk_full":      float(full_b["p"]),
            "beta_uk_A":      float(A_b["beta"]),
            "p_uk_A":         float(A_b["p"]),
            "beta_uk_B":      float(B_b["beta"]),
            "p_uk_B":         float(B_b["p"]),
            "beta_manchester": float(manc_b["beta"]),
            "p_manchester":   float(manc_b["p"]),
        },
    }

    json_path = out_dir / f"uk_reverse_results_N{args.window}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → {json_path}")

    # Persist per-child tables
    per_child_csv = out_dir / f"uk_reverse_per_child_N{args.window}.csv"
    uk_full_per_child["scenario"] = "UK_full_post_MSR"
    uk_A_per_child["scenario"]    = "UK_24-36_lifetime_cum"
    uk_B_per_child["scenario"]    = "UK_24-36_local_cum"
    manc_per_child["scenario"]    = "Manchester_full_post_MSR"
    all_children = pd.concat([uk_full_per_child, uk_A_per_child, uk_B_per_child, manc_per_child],
                              ignore_index=True)
    all_children.to_csv(per_child_csv, index=False)
    print(f"  → {per_child_csv}")

    # Summary CSV (corpus-level β table)
    summary_rows: List[Dict[str, Any]] = []
    for scenario, fit_block in [
        ("UK_full_post_MSR",       uk_full_corpus),
        ("UK_24-36_lifetime_cum",  uk_A_corpus),
        ("UK_24-36_local_cum",     uk_B_corpus),
        ("Manchester_full_post_MSR", manc_corpus),
    ]:
        for pred, vals in fit_block["params"].items():
            summary_rows.append({
                "scenario":  scenario,
                "predictor": pred,
                "beta":      vals["beta"],
                "se":        vals["se"],
                "t":         vals["t"],
                "p":         vals["p"],
                "ci95_low":  vals.get("ci95_low"),
                "ci95_high": vals.get("ci95_high"),
                "n":         fit_block["n"],
                "n_cues":    fit_block["n_cues"],
                "n_children": fit_block["n_children"],
            })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"uk_reverse_summary_N{args.window}.csv", index=False
    )
    print(f"  → {out_dir / f'uk_reverse_summary_N{args.window}.csv'}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# 17f UK reverse simulation — outcome window N={args.window}\n")
    lines.append("## Corpus-level β(COI × cumulative_cue_attempts)\n")
    lines.append("| Scenario | n_children | n_episodes | β | SE | p |")
    lines.append("|---|---|---|---|---|---|")
    for scenario, fit_block in [
        ("UK_full_post_MSR",       uk_full_corpus),
        ("UK_24-36_lifetime_cum (Variant A)", uk_A_corpus),
        ("UK_24-36_local_cum (Variant B)",    uk_B_corpus),
        ("Manchester_full_post_MSR", manc_corpus),
    ]:
        cxc = fit_block["params"]["COI_x_cumulative"]
        lines.append(
            f"| {scenario} | {fit_block['n_children']} | {fit_block['n']:,} | "
            f"{cxc['beta']:+.4f} | {cxc['se']:.4f} | {cxc['p']:.4f} |"
        )
    lines.append("\n## Per-child random-effects meta (β_i pooled)\n")
    lines.append("| Scenario | n | pooled β | SE | p | τ² | I² |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in [
        ("UK_full_post_MSR", meta_uk_full),
        ("UK_24-36_lifetime_cum (Variant A)", meta_uk_A),
        ("UK_24-36_local_cum (Variant B)",    meta_uk_B),
        ("Manchester_full_post_MSR", meta_manc),
    ]:
        if "pooled_beta_RE" in m:
            lines.append(
                f"| {name} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
            )
    lines.append(f"\n## Verdict\n")
    lines.append(
        f"- UK full → UK 24-36 (Variant A, lifetime cumulative): "
        f"β = {full_b['beta']:+.4f} → {A_b['beta']:+.4f}  "
        f"({'PASS' if verdict_A else 'FAIL'})\n"
    )
    lines.append(
        f"- UK full → UK 24-36 (Variant B, window-local cumulative): "
        f"β = {full_b['beta']:+.4f} → {B_b['beta']:+.4f}  "
        f"({'PASS' if verdict_B else 'FAIL'})\n"
    )
    lines.append(f"- Manchester reference β = {manc_b['beta']:+.4f}, p = {manc_b['p']:.4f}\n")
    lines.append(
        f"- **Overall**: {'PASS — window IS the gate; corpus is not the confound.' if overall_pass else 'FAIL — UK retains its effect even at Manchester-equivalent window.'}\n"
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
