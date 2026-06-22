"""
11_rsm_r_plus_join.py
=====================
IMT Attention Bias Paper — Step 11: R+ × Paper 1 Integration (Paper 2 central test)

Joins the Paper 2 R+ output (10_extract_R_plus_v2.py) with Paper 1's
caregiver-side COI (03_compute_attention_index_v3.py) and child uptake
(05_developmental_uptake.py), then tests the Paper 2 central prediction via
hierarchical regression.

────────────────────────────────────────────────────────────────────────────
Inputs (all per cue_subtype):
  {lang}_r_plus_cue_agg.csv      (Paper 2 v2 output)
  {lang}_uptake.csv              (Paper 1 output — provides peak, logFreq, COI)
  {lang}_attention_index.csv     (Paper 1 output — optional, for S_* breakdown)

Hierarchical regression (all variables z-standardized):
  M0:  peak ~ logFreq                                    [exposure baseline]
  M1:  + COI + COI × logFreq                             [Paper 1 replication]
  M2:  + RplusRate + RplusComposite + RplusComposite × COI ★ Paper 2 incremental

Reports (per nested transition):
  - R², adjusted R², ΔR², F-test (compare_f_test)
  - Per-predictor: β (z-scored), SE, t, p, semipartial r
  - Paper 2 central test: ΔR² from M1 → M2 + significance

Output:
  {lang}_r_plus_joined.csv          (merged dataset, n_cues rows)
  {lang}_r_plus_regression.json     (full regression results, all 3 models)

Usage:
  python 11_rsm_r_plus_join.py \\
      --r_plus_csv ./output/v10b/English_r_plus_cue_agg.csv \\
      --uptake_csv ./output/v3/English_uptake.csv \\
      --ai_csv ./output/v3/English_attention_index.csv \\
      --language English \\
      --output_dir ./output/v11/

Author: Torami x Boss | IMT Attention project | Paper 2 / R+ central test | 2026-06-16
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pandas scipy statsmodels")
    sys.exit(1)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchical regression: R+_composite × COI × logFreq → peak production"
    )
    p.add_argument("--r_plus_csv", required=True,
                   help="Paper 2 cue-level R+ aggregation (from 10_extract_R_plus_v2.py)")
    p.add_argument("--uptake_csv", required=True,
                   help="Paper 1 uptake CSV (from 05_developmental_uptake.py)")
    p.add_argument("--ai_csv", default=None,
                   help="(Optional) Paper 1 attention_index CSV for S_* breakdown")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK",
                            "English-NA-Pool", "English-NewmanRatner",
                            "English-BernsteinRatner", "English-Tardif",
                            "English-Higginson", "English-Brent",
                            "English-Rollins", "English-Soderstrom",
                            "English-Manchester", "English-Providence",
                            "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output/v11")
    p.add_argument("--min_attempts", type=int, default=5,
                   help="Filter out cues with fewer than this many child attempts (R+).")
    p.add_argument("--outcome_col", default="peak_rate_per_1k",
                   help="Outcome column in uptake CSV. Default: peak_rate_per_1k.")
    p.add_argument("--composite_col", default="mean_r_plus_composite_contingent",
                   help="(v2.2) R+ composite column to use as primary predictor. "
                        "Options: mean_r_plus_composite, mean_r_plus_composite_contingent, "
                        "mean_r_plus_composite_positive. Default: _contingent.")
    p.add_argument("--rate_col", default="modeling_rate",
                   help="(v2.2) R+ rate column to enter as a secondary predictor. "
                        "Options: r_plus_response_rate, positive_rate, modeling_rate, "
                        "expansion_rate, recast_rate. Default: modeling_rate.")
    p.add_argument("--exclude_rate", action="store_true",
                   help="(v2.2) Exclude RplusRate from M2 model. Useful when R+_rate × "
                        "R+_composite collinearity is high (r > 0.7), to obtain cleaner "
                        "β estimates for R+_composite and Rcomp × COI.")
    p.add_argument("--log_outcome", action="store_true",
                   help="(v2.2) Apply log1p transform to outcome before z-standardizing. "
                        "Useful when outcome is right-skewed (e.g., peak_rate_per_1k).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Loading and merging
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(args: argparse.Namespace) -> pd.DataFrame:
    """Inner-join r_plus_cue_agg + uptake on cue_subtype, optionally with attention_index."""
    print(f"Loading r_plus_cue_agg : {args.r_plus_csv}")
    r_plus = pd.read_csv(args.r_plus_csv)
    print(f"  Rows: {len(r_plus):,}  (cue subtypes)")

    print(f"Loading uptake         : {args.uptake_csv}")
    uptake = pd.read_csv(args.uptake_csv)
    print(f"  Rows: {len(uptake):,}")

    # Merge on cue_subtype
    merged = r_plus.merge(uptake, on="cue_subtype", how="inner", suffixes=("", "_uptake"))
    print(f"\n  Merged (r_plus ⋈ uptake): {len(merged):,} cue subtypes")

    if args.ai_csv:
        print(f"Loading attention_index: {args.ai_csv}")
        ai = pd.read_csv(args.ai_csv)
        # Only carry the breakdown S_* columns, not the main AttentionIndex
        # (we use caregiver_AI from uptake.csv as the canonical COI)
        ai_cols_to_keep = ["cue_subtype"] + [c for c in ai.columns if c.startswith("S_")]
        merged = merged.merge(ai[ai_cols_to_keep], on="cue_subtype", how="left",
                              suffixes=("", "_ai"))
        print(f"  Merged with attention_index: {len(merged):,} cue subtypes "
              f"(+ {len(ai_cols_to_keep) - 1} S_* columns)")

    # Apply minimum-attempts filter
    if args.min_attempts > 0:
        before = len(merged)
        merged = merged[merged["n_child_attempts"] >= args.min_attempts].copy()
        print(f"\n  After min_attempts ≥ {args.min_attempts}: "
              f"{len(merged):,} (dropped {before - len(merged):,})")

    # Sanity: drop rows missing critical fields
    critical = ["caregiver_AI", "log_caregiver_count", args.outcome_col,
                "r_plus_response_rate", "mean_r_plus_composite"]
    missing = merged[critical].isna().any(axis=1)
    if missing.sum() > 0:
        print(f"  Dropping {missing.sum()} rows with missing predictor/outcome values")
        merged = merged[~missing].copy()

    print(f"\n  Final analysis dataset: {len(merged):,} cue subtypes")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Variable preparation
# ─────────────────────────────────────────────────────────────────────────────

def z(s: pd.Series) -> pd.Series:
    """Z-standardize a numeric Series (NaN-safe; constant series → 0s)."""
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def prepare_variables(merged: pd.DataFrame,
                      outcome_col: str,
                      composite_col: str = "mean_r_plus_composite_contingent",
                      rate_col: str = "modeling_rate",
                      log_outcome: bool = False) -> Tuple[pd.DataFrame, str]:
    """Z-standardize predictors and outcome, build interaction terms.

    v2.2: predictor columns for R+ are selectable via CLI to allow comparing
    all-episode mean vs contingent-only vs positive-only composites, and
    among the per-label rate measures (modeling_rate is the recommended default).
    """
    df = merged.copy()

    # Sanity: required columns
    if composite_col not in df.columns:
        raise ValueError(
            f"composite_col '{composite_col}' not in merged dataset. "
            f"Available R+ composite columns: "
            f"{[c for c in df.columns if 'composite' in c]}"
        )
    if rate_col not in df.columns:
        raise ValueError(
            f"rate_col '{rate_col}' not in merged dataset. "
            f"Available rate columns: "
            f"{[c for c in df.columns if c.endswith('_rate')]}"
        )

    # Rename to canonical predictor names
    df["logFreq"]         = df["log_caregiver_count"]
    df["COI"]             = df["caregiver_AI"]
    df["RplusRate"]       = df[rate_col]
    df["RplusComposite"]  = df[composite_col]

    if log_outcome:
        df["outcome"]     = np.log1p(df[outcome_col].clip(lower=0))
    else:
        df["outcome"]     = df[outcome_col]

    # z-standardize
    for col in ["logFreq", "COI", "RplusRate", "RplusComposite", "outcome"]:
        df[f"{col}_z"] = z(df[col])

    # Interaction terms (built on z-scored variables)
    df["COI_x_logFreq"]      = df["COI_z"]      * df["logFreq_z"]
    df["Rcomp_x_COI"]        = df["RplusComposite_z"] * df["COI_z"]
    df["Rcomp_x_logFreq"]    = df["RplusComposite_z"] * df["logFreq_z"]
    df["Rrate_x_Rcomp"]      = df["RplusRate_z"]      * df["RplusComposite_z"]

    return df, "outcome_z"


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit an OLS model with a constant intercept."""
    X = sm.add_constant(df[x_cols].astype(float), has_constant="add")
    y = df[y_col].astype(float)
    return sm.OLS(y, X).fit()


def semipartial_r(model: sm.regression.linear_model.RegressionResultsWrapper,
                  predictor: str, df: pd.DataFrame, y_col: str) -> float:
    """Compute semipartial correlation = sqrt(ΔR² when predictor is dropped)."""
    other_x = [c for c in model.model.exog_names if c not in ("const", predictor)]
    if not other_x:
        return float(model.rsquared)  # only one predictor
    reduced = fit_ols(df, y_col, other_x)
    delta = model.rsquared - reduced.rsquared
    if delta < 0:
        return 0.0
    # Sign follows the predictor's t-stat
    sign = np.sign(model.tvalues[predictor]) if predictor in model.tvalues else 1
    return float(sign * np.sqrt(delta))


def summarize_model(model: sm.regression.linear_model.RegressionResultsWrapper,
                    df: pd.DataFrame, y_col: str,
                    name: str, prev_model: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
                    ) -> Dict[str, Any]:
    """Build a JSON-friendly summary of one regression model."""
    out: Dict[str, Any] = {
        "name": name,
        "n": int(model.nobs),
        "predictors": [c for c in model.model.exog_names if c != "const"],
        "r2": round(float(model.rsquared), 4),
        "r2_adj": round(float(model.rsquared_adj), 4),
        "f_stat_overall": round(float(model.fvalue), 4),
        "f_pvalue_overall": float(model.f_pvalue),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "params": {},
    }
    for pred in out["predictors"]:
        out["params"][pred] = {
            "beta_z":        round(float(model.params[pred]), 4),
            "std_error":     round(float(model.bse[pred]), 4),
            "t":             round(float(model.tvalues[pred]), 4),
            "p":             float(model.pvalues[pred]),
            "ci95_low":      round(float(model.conf_int().loc[pred, 0]), 4),
            "ci95_high":     round(float(model.conf_int().loc[pred, 1]), 4),
            "semipartial_r": round(semipartial_r(model, pred, df, y_col), 4),
        }

    if prev_model is not None:
        delta_r2 = float(model.rsquared - prev_model.rsquared)
        # F-test for nested-model comparison
        try:
            ftest = model.compare_f_test(prev_model)
            out["nested_comparison"] = {
                "delta_r2":    round(delta_r2, 4),
                "f_stat":      round(float(ftest[0]), 4),
                "p":           float(ftest[1]),
                "df_diff":     int(ftest[2]),
            }
        except Exception as e:
            out["nested_comparison"] = {
                "delta_r2":    round(delta_r2, 4),
                "f_stat":      None,
                "p":           None,
                "error":       str(e),
            }

    return out


def run_hierarchical(df: pd.DataFrame, y_col: str,
                     exclude_rate: bool = False) -> Dict[str, Any]:
    """Run M0 → M1 → M2 and collect summaries.

    v2.2: when exclude_rate=True, the M2 model omits RplusRate_z to mitigate
    multicollinearity (relevant when r(RplusRate, RplusComposite) > 0.7).
    """

    print("\n" + "=" * 70)
    print("Hierarchical regression — M0 → M1 → M2"
          + ("  (exclude_rate=True)" if exclude_rate else ""))
    print("=" * 70)

    # M0: exposure-only baseline
    M0_x = ["logFreq_z"]
    m0 = fit_ols(df, y_col, M0_x)
    print(f"\n  M0 (logFreq only)               R² = {m0.rsquared:.4f}")

    # M1: Paper 1 baseline replication
    M1_x = M0_x + ["COI_z", "COI_x_logFreq"]
    m1 = fit_ols(df, y_col, M1_x)
    print(f"  M1 (+ COI + COI×logFreq)        R² = {m1.rsquared:.4f}  "
          f"ΔR² = {m1.rsquared - m0.rsquared:+.4f}")

    # M2: Paper 2 incremental — R+_composite enters
    if exclude_rate:
        M2_x = M1_x + ["RplusComposite_z", "Rcomp_x_COI"]
        m2_label = "M2 (+ R+_composite + Rcomp×COI, rate EXCLUDED)"
    else:
        M2_x = M1_x + ["RplusRate_z", "RplusComposite_z", "Rcomp_x_COI"]
        m2_label = "M2 (+ R+_rate + R+_composite + Rcomp×COI)"
    m2 = fit_ols(df, y_col, M2_x)
    print(f"  {m2_label}")
    print(f"                                    R² = {m2.rsquared:.4f}  "
          f"ΔR² = {m2.rsquared - m1.rsquared:+.4f}  ★ Paper 2 incremental")

    # Optional M3 (exploratory): + Rcomp × logFreq
    M3_x = M2_x + ["Rcomp_x_logFreq"]
    m3 = fit_ols(df, y_col, M3_x)
    print(f"\n  M3 (+ Rcomp × logFreq, exploratory)  R² = {m3.rsquared:.4f}  "
          f"ΔR² = {m3.rsquared - m2.rsquared:+.4f}")

    m2_name = "M2_Paper2_incremental_no_rate" if exclude_rate else "M2_Paper2_incremental"
    return {
        "M0": summarize_model(m0, df, y_col, "M0_logFreq_only"),
        "M1": summarize_model(m1, df, y_col, "M1_Paper1_replication", prev_model=m0),
        "M2": summarize_model(m2, df, y_col, m2_name, prev_model=m1),
        "M3_exploratory": summarize_model(m3, df, y_col, "M3_with_Rcomp_x_logFreq",
                                           prev_model=m2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bivariate descriptives
# ─────────────────────────────────────────────────────────────────────────────

def bivariate_correlations(df: pd.DataFrame, outcome_col: str = "outcome") -> Dict[str, Dict[str, Any]]:
    """Pearson and Spearman correlations of each predictor with outcome."""
    out: Dict[str, Dict[str, Any]] = {}
    for pred in ["logFreq", "COI", "RplusRate", "RplusComposite"]:
        try:
            r_p, p_p = stats.pearsonr(df[pred], df[outcome_col])
            r_s, p_s = stats.spearmanr(df[pred], df[outcome_col])
            out[pred] = {
                "pearson_r":   round(float(r_p), 4),
                "pearson_p":   float(p_p),
                "spearman_r":  round(float(r_s), 4),
                "spearman_p":  float(p_s),
            }
        except Exception as e:
            out[pred] = {"error": str(e)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics (text-based, JSON-friendly)
# ─────────────────────────────────────────────────────────────────────────────

def collinearity_check(df: pd.DataFrame) -> Dict[str, float]:
    """Pearson correlation matrix among the four main predictors."""
    cols = ["logFreq_z", "COI_z", "RplusRate_z", "RplusComposite_z"]
    cmat = df[cols].corr().round(4)
    return {f"{c1}__{c2}": float(cmat.loc[c1, c2])
            for c1 in cols for c2 in cols if c1 < c2}


def descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Mean / median / SD / range for each main predictor and outcome."""
    cols = ["logFreq", "COI", "RplusRate", "RplusComposite", "outcome"]
    desc = df[cols].describe().to_dict()
    return {col: {k: round(float(v), 4) for k, v in vals.items()
                   if isinstance(v, (int, float, np.number))}
            for col, vals in desc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 11_rsm_r_plus_join.py | Paper 2 central test (v2.2) ===")
    print(f"  Language        : {args.language}")
    print(f"  R+ aggregation  : {args.r_plus_csv}")
    print(f"  Uptake          : {args.uptake_csv}")
    print(f"  AttentionIndex  : {args.ai_csv or '(skipped)'}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Min attempts    : {args.min_attempts}")
    print(f"  Outcome column  : {args.outcome_col}"
          f"{'  (log1p transformed)' if args.log_outcome else ''}")
    print(f"  R+ composite    : {args.composite_col}  (v2.2)")
    print(f"  R+ rate         : {args.rate_col}  (v2.2)"
          f"{'  [EXCLUDED from M2]' if args.exclude_rate else ''}\n")

    merged = load_and_merge(args)
    if len(merged) < 10:
        print("\n  ⚠ Fewer than 10 cues remain after merge/filter; "
              "regression will be unreliable.")

    df, y_col = prepare_variables(
        merged, args.outcome_col,
        composite_col=args.composite_col,
        rate_col=args.rate_col,
        log_outcome=args.log_outcome,
    )

    # Save merged dataset
    joined_csv = out_dir / f"{args.language}_r_plus_joined.csv"
    df.to_csv(joined_csv, index=False)
    print(f"\n  Joined dataset → {joined_csv} ({len(df):,} rows)")

    # Descriptive stats
    print("\nDescriptive statistics (raw scale):")
    desc = descriptive_stats(df)
    for col, vals in desc.items():
        print(f"  {col:18s}  mean={vals.get('mean', float('nan')):>+8.3f}  "
              f"sd={vals.get('std', float('nan')):>+8.3f}  "
              f"min={vals.get('min', float('nan')):>+8.3f}  "
              f"max={vals.get('max', float('nan')):>+8.3f}")

    # Bivariate correlations
    print("\nBivariate correlations with outcome (raw scale):")
    bivar = bivariate_correlations(df, outcome_col="outcome")
    for pred, vals in bivar.items():
        if "error" in vals:
            print(f"  {pred:18s}  error: {vals['error']}")
        else:
            sig_p = "***" if vals["pearson_p"] < 0.001 else (
                    "**"  if vals["pearson_p"] < 0.01 else (
                    "*"   if vals["pearson_p"] < 0.05 else "ns"))
            print(f"  {pred:18s}  r = {vals['pearson_r']:+.4f}  ({sig_p})")

    # Collinearity check
    print("\nCollinearity (z-scored predictors, off-diagonal r):")
    collin = collinearity_check(df)
    for pair, r in sorted(collin.items()):
        marker = " ←⚠" if abs(r) > 0.7 else ""
        print(f"  {pair:50s}  r = {r:+.4f}{marker}")

    # Hierarchical regression
    models = run_hierarchical(df, y_col, exclude_rate=args.exclude_rate)

    # Per-predictor printout for M2 (the key model)
    print("\n" + "=" * 70)
    print("M2 predictor table (Paper 2 incremental, z-standardized)")
    print("=" * 70)
    print(f"  {'predictor':22s}  {'β':>8s}  {'SE':>7s}  {'t':>7s}  {'p':>9s}  {'sr':>7s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}")
    for pred, vals in models["M2"]["params"].items():
        p = vals["p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {pred:22s}  {vals['beta_z']:+8.4f}  {vals['std_error']:7.4f}  "
              f"{vals['t']:+7.3f}  {p:9.4g}{sig:>3s}  {vals['semipartial_r']:+7.4f}")

    # Paper 2 central finding summary
    print("\n" + "=" * 70)
    print("Paper 2 central test summary")
    print("=" * 70)
    nc = models["M2"]["nested_comparison"]
    print(f"  M1 → M2 (Paper 2 incremental):")
    print(f"    ΔR²  = {nc['delta_r2']:+.4f}")
    print(f"    F    = {nc['f_stat']}")
    print(f"    p    = {nc['p']:.6g}")
    if "RplusComposite_z" in models["M2"]["params"]:
        rcomp = models["M2"]["params"]["RplusComposite_z"]
        print(f"  R+_composite main effect : β = {rcomp['beta_z']:+.4f},  "
              f"p = {rcomp['p']:.4g},  sr = {rcomp['semipartial_r']:+.4f}")
    if "Rcomp_x_COI" in models["M2"]["params"]:
        rcox = models["M2"]["params"]["Rcomp_x_COI"]
        print(f"  R+_composite × COI       : β = {rcox['beta_z']:+.4f},  "
              f"p = {rcox['p']:.4g},  sr = {rcox['semipartial_r']:+.4f}")

    # Final JSON output
    result = {
        "language": args.language,
        "version": "v1",
        "n_cues_analyzed": int(len(df)),
        "outcome_column": args.outcome_col,
        "settings": {
            "min_attempts":   int(args.min_attempts),
            "composite_col":  args.composite_col,
            "rate_col":       args.rate_col,
            "exclude_rate":   bool(args.exclude_rate),
            "log_outcome":    bool(args.log_outcome),
        },
        "descriptive_statistics": desc,
        "bivariate_correlations": bivar,
        "collinearity_matrix":    collin,
        "models": models,
    }

    summary_json = out_dir / f"{args.language}_r_plus_regression.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n=== Done ===")
    print(f"  Joined dataset : {joined_csv}")
    print(f"  Regression JSON: {summary_json}")


if __name__ == "__main__":
    main()
