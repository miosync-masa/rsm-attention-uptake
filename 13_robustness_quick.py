"""
13_robustness_quick.py
======================
IMT Attention Bias Paper — Step 13 (quick): Robustness Diagnostics for Paper 2

Two compact robustness checks for the Paper 2 primary regression (M2 model):

  (1) Influence diagnostics
        - Cook's distance       (cutoff: 4/n)
        - Leverage (hat values) (cutoff: 2k/n, where k = #predictors)
        - DFFITS                (cutoff: 2 √(k/n))
        Identifies cues that disproportionately drive the β estimate for
        R+_composite × COI. Reports per-cue diagnostics + a flagged list.

  (2) Nonparametric bootstrap CI
        - n_bootstrap resamples (default 5000) of the cue set, with replacement
        - Refit M2 on each resample
        - Report empirical 2.5% / 97.5% percentile CI for Rcomp × COI
        - Compare with the parametric (normal-based) CI from the original fit

Both checks reuse the exact same predictor preparation as 11_rsm_r_plus_join.py
so that results are comparable.

Inputs:
  {language}_r_plus_joined.csv      (from 11_rsm_r_plus_join.py output)

Outputs (per language):
  {language}_influence_diagnostics.csv
  {language}_bootstrap_distribution.csv
  {language}_robustness_quick.json

Usage:
  python 13_robustness_quick.py \\
      --joined_csv ./output/v11_runA/English_r_plus_joined.csv \\
      --language English \\
      --output_dir ./output/v13/ \\
      --n_bootstrap 5000 \\
      --random_seed 42

Author: Torami x Boss | IMT Attention project | Paper 2 / robustness | 2026-06-17
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
    from scipy import stats
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pandas scipy statsmodels tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick robustness diagnostics: Cook's + bootstrap CI for M2."
    )
    p.add_argument("--joined_csv", required=True,
                   help="Output of 11_rsm_r_plus_join.py: {lang}_r_plus_joined.csv")
    p.add_argument("--language", required=True)
    p.add_argument("--output_dir", default="./output/v13")
    p.add_argument("--exclude_rate", action="store_true",
                   help="If set, fit M2 without RplusRate (mirroring 11 --exclude_rate).")
    p.add_argument("--n_bootstrap", type=int, default=5000,
                   help="Number of bootstrap resamples for CI. Default 5000.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--target_effect", default="Rcomp_x_COI",
                   choices=["Rcomp_x_COI", "RplusComposite_z"],
                   help="Which coefficient to bootstrap.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# M2 specification (mirrors 11_rsm_r_plus_join.py)
# ─────────────────────────────────────────────────────────────────────────────

def m2_predictor_list(exclude_rate: bool) -> List[str]:
    """Return the ordered predictor list for M2, consistent with 11."""
    base = ["logFreq_z", "COI_z", "COI_x_logFreq"]
    if exclude_rate:
        return base + ["RplusComposite_z", "Rcomp_x_COI"]
    return base + ["RplusRate_z", "RplusComposite_z", "Rcomp_x_COI"]


def fit_m2(df: pd.DataFrame, exclude_rate: bool) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit the M2 regression on the joined dataset (outcome already z-scored)."""
    X = sm.add_constant(df[m2_predictor_list(exclude_rate)].astype(float),
                        has_constant="add")
    y = df["outcome_z"].astype(float)
    return sm.OLS(y, X).fit()


# ─────────────────────────────────────────────────────────────────────────────
# (1) Influence diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def influence_diagnostics(df: pd.DataFrame, model: sm.regression.linear_model.RegressionResultsWrapper,
                          exclude_rate: bool) -> pd.DataFrame:
    """Compute per-observation Cook's distance, leverage, DFFITS and flag influentials."""
    influence = OLSInfluence(model)
    n = int(model.nobs)
    # k = number of predictors EXCLUDING intercept
    k = len(m2_predictor_list(exclude_rate))

    cooks_d, _ = influence.cooks_distance
    leverage  = influence.hat_matrix_diag
    dffits, _ = influence.dffits

    # Standard cutoffs
    cooks_cutoff   = 4.0 / n
    leverage_cut   = 2.0 * (k + 1) / n   # +1 for intercept
    dffits_cut     = 2.0 * np.sqrt((k + 1) / n)

    out = df[["cue_subtype", "n_child_attempts", "logFreq", "COI",
              "RplusComposite", "outcome"]].copy().reset_index(drop=True)
    out["cooks_distance"]   = cooks_d
    out["leverage"]         = leverage
    out["dffits"]           = dffits
    out["flag_cooks"]       = cooks_d > cooks_cutoff
    out["flag_leverage"]    = leverage > leverage_cut
    out["flag_dffits"]      = np.abs(dffits) > dffits_cut
    out["flag_any"]         = out["flag_cooks"] | out["flag_leverage"] | out["flag_dffits"]
    out["n_flags"]          = (out[["flag_cooks", "flag_leverage", "flag_dffits"]]
                               .sum(axis=1).astype(int))
    out["_cooks_cutoff"]    = cooks_cutoff
    out["_leverage_cutoff"] = leverage_cut
    out["_dffits_cutoff"]   = dffits_cut

    return out.sort_values("cooks_distance", ascending=False).reset_index(drop=True)


def refit_excluding_influentials(df: pd.DataFrame, infl_df: pd.DataFrame,
                                  exclude_rate: bool,
                                  target_effect: str) -> Dict[str, Any]:
    """Refit M2 excluding (a) Cook's-flagged observations, (b) any-flagged observations.
    Reports the new β for the target effect under each exclusion.
    """
    out: Dict[str, Any] = {}

    # Original
    m_orig = fit_m2(df, exclude_rate)
    out["original"] = {
        "n":  int(m_orig.nobs),
        "beta_z": round(float(m_orig.params[target_effect]), 4),
        "se":      round(float(m_orig.bse[target_effect]), 4),
        "p":       float(m_orig.pvalues[target_effect]),
        "ci95_low":  round(float(m_orig.conf_int().loc[target_effect, 0]), 4),
        "ci95_high": round(float(m_orig.conf_int().loc[target_effect, 1]), 4),
    }

    # Drop Cook's flagged
    cooks_flagged_cues = set(infl_df.loc[infl_df["flag_cooks"], "cue_subtype"].tolist())
    df_no_cooks = df[~df["cue_subtype"].isin(cooks_flagged_cues)].copy()
    if len(df_no_cooks) >= 10:
        m_nc = fit_m2(df_no_cooks, exclude_rate)
        out["excluding_cooks_flagged"] = {
            "n":  int(m_nc.nobs),
            "n_excluded": len(cooks_flagged_cues),
            "excluded_cues": sorted(cooks_flagged_cues),
            "beta_z":    round(float(m_nc.params[target_effect]), 4),
            "se":        round(float(m_nc.bse[target_effect]), 4),
            "p":         float(m_nc.pvalues[target_effect]),
            "ci95_low":  round(float(m_nc.conf_int().loc[target_effect, 0]), 4),
            "ci95_high": round(float(m_nc.conf_int().loc[target_effect, 1]), 4),
        }
    else:
        out["excluding_cooks_flagged"] = {"error": "fewer than 10 cues remain"}

    # Drop any-flagged
    any_flagged_cues = set(infl_df.loc[infl_df["flag_any"], "cue_subtype"].tolist())
    df_no_any = df[~df["cue_subtype"].isin(any_flagged_cues)].copy()
    if len(df_no_any) >= 10:
        m_na = fit_m2(df_no_any, exclude_rate)
        out["excluding_any_flagged"] = {
            "n":  int(m_na.nobs),
            "n_excluded": len(any_flagged_cues),
            "excluded_cues": sorted(any_flagged_cues),
            "beta_z":    round(float(m_na.params[target_effect]), 4),
            "se":        round(float(m_na.bse[target_effect]), 4),
            "p":         float(m_na.pvalues[target_effect]),
            "ci95_low":  round(float(m_na.conf_int().loc[target_effect, 0]), 4),
            "ci95_high": round(float(m_na.conf_int().loc[target_effect, 1]), 4),
        }
    else:
        out["excluding_any_flagged"] = {"error": "fewer than 10 cues remain"}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# (2) Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(df: pd.DataFrame, exclude_rate: bool, target_effect: str,
                 n_boot: int, random_seed: int) -> Dict[str, Any]:
    """Nonparametric bootstrap of β for target_effect in M2.

    Resamples cues with replacement, refits M2, collects β. Reports:
      - mean / median / SE (bootstrap SD)
      - percentile CI (2.5% / 97.5%)
      - bias-corrected accelerated (BCa) CI — Efron & Tibshirani (1993)
      - empirical p-value (two-sided): proportion of bootstrap β with sign
        opposite to the observed β.
    """
    rng    = np.random.default_rng(random_seed)
    n      = len(df)
    betas  = np.full(n_boot, np.nan)

    # Original β (for BCa acceleration and p reference)
    m_orig = fit_m2(df, exclude_rate)
    beta_obs = float(m_orig.params[target_effect])

    print(f"  Running {n_boot:,} bootstrap resamples...")
    failed = 0
    for b in tqdm(range(n_boot), desc="  bootstrap"):
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        try:
            m_b = fit_m2(boot_df, exclude_rate)
            betas[b] = float(m_b.params[target_effect])
        except Exception:
            failed += 1
            continue

    valid = betas[~np.isnan(betas)]
    print(f"  Valid resamples: {len(valid):,} / {n_boot:,}  (failed: {failed})")

    # Percentile CI
    ci_lo_pct, ci_hi_pct = np.percentile(valid, [2.5, 97.5])

    # BCa correction (Efron-Tibshirani)
    # z0: bias-correction
    p_below = float(np.mean(valid < beta_obs))
    if p_below in (0.0, 1.0):
        z0 = 0.0  # degenerate; fall back to percentile
    else:
        z0 = float(stats.norm.ppf(p_below))

    # Acceleration via jackknife
    jack_betas = np.full(n, np.nan)
    for i in range(n):
        jack_df = df.drop(df.index[i]).reset_index(drop=True)
        try:
            m_j = fit_m2(jack_df, exclude_rate)
            jack_betas[i] = float(m_j.params[target_effect])
        except Exception:
            continue
    jack_valid = jack_betas[~np.isnan(jack_betas)]
    if len(jack_valid) >= 3:
        jack_mean = np.mean(jack_valid)
        num = np.sum((jack_mean - jack_valid) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_valid) ** 2) ** 1.5)
        a = float(num / den) if den > 0 else 0.0
    else:
        a = 0.0

    # BCa percentiles
    z_lo, z_hi = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
    alpha_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    ci_lo_bca = float(np.percentile(valid, 100 * alpha_lo))
    ci_hi_bca = float(np.percentile(valid, 100 * alpha_hi))

    # Empirical two-sided p: proportion of bootstrap β with sign opposite to observed
    if beta_obs > 0:
        p_emp = 2.0 * float(np.mean(valid <= 0))
    elif beta_obs < 0:
        p_emp = 2.0 * float(np.mean(valid >= 0))
    else:
        p_emp = 1.0
    p_emp = min(1.0, p_emp)

    return {
        "n_bootstrap_requested":  int(n_boot),
        "n_bootstrap_valid":      int(len(valid)),
        "n_bootstrap_failed":     int(failed),
        "observed_beta":          round(beta_obs, 4),
        "bootstrap_mean":         round(float(np.mean(valid)), 4),
        "bootstrap_median":       round(float(np.median(valid)), 4),
        "bootstrap_se":           round(float(np.std(valid, ddof=1)), 4),
        "ci95_percentile":        [round(float(ci_lo_pct), 4),
                                    round(float(ci_hi_pct), 4)],
        "ci95_bca":               [round(ci_lo_bca, 4), round(ci_hi_bca, 4)],
        "z0":                     round(z0, 4),
        "acceleration":           round(a, 4),
        "empirical_p_two_sided":  round(float(p_emp), 4),
        "bootstrap_distribution": [round(float(b), 4) for b in valid[:1000]],  # cap for JSON
    }


# ─────────────────────────────────────────────────────────────────────────────
# (3) VIF (collinearity, bonus check)
# ─────────────────────────────────────────────────────────────────────────────

def vif_table(df: pd.DataFrame, exclude_rate: bool) -> Dict[str, float]:
    """Variance Inflation Factors for all M2 predictors."""
    preds = m2_predictor_list(exclude_rate)
    X = sm.add_constant(df[preds].astype(float), has_constant="add")
    vifs: Dict[str, float] = {}
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        try:
            v = float(variance_inflation_factor(X.values, i))
            vifs[name] = round(v, 3)
        except Exception:
            vifs[name] = float("nan")
    return vifs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 13_robustness_quick.py | Paper 2 robustness diagnostics ===")
    print(f"  Language        : {args.language}")
    print(f"  Joined CSV      : {args.joined_csv}")
    print(f"  Output dir      : {out_dir}")
    print(f"  exclude_rate    : {args.exclude_rate}")
    print(f"  n_bootstrap     : {args.n_bootstrap}")
    print(f"  target_effect   : {args.target_effect}")
    print(f"  random_seed     : {args.random_seed}\n")

    df = pd.read_csv(args.joined_csv)
    print(f"  Joined dataset rows: {len(df)}\n")
    if "outcome_z" not in df.columns:
        raise ValueError("joined CSV must contain 'outcome_z' (z-scored outcome). "
                         "Make sure it came from 11_rsm_r_plus_join.py.")

    # Fit original M2
    m_orig = fit_m2(df, args.exclude_rate)
    target_p_row = {
        "beta_z":    round(float(m_orig.params[args.target_effect]), 4),
        "se":        round(float(m_orig.bse[args.target_effect]), 4),
        "p":         float(m_orig.pvalues[args.target_effect]),
        "ci95_low":  round(float(m_orig.conf_int().loc[args.target_effect, 0]), 4),
        "ci95_high": round(float(m_orig.conf_int().loc[args.target_effect, 1]), 4),
    }
    print(f"Original M2 estimate for {args.target_effect}:")
    print(f"  β = {target_p_row['beta_z']:+.4f}  SE = {target_p_row['se']:.4f}  "
          f"p = {target_p_row['p']:.4g}  "
          f"CI = [{target_p_row['ci95_low']:+.4f}, {target_p_row['ci95_high']:+.4f}]\n")

    # ── (1) Influence diagnostics ──
    print("=" * 70)
    print("(1) Influence diagnostics")
    print("=" * 70)
    infl_df = influence_diagnostics(df, m_orig, args.exclude_rate)

    flagged_any = infl_df[infl_df["flag_any"]]
    print(f"  Total cues             : {len(infl_df)}")
    print(f"  Cook's-flagged         : {int(infl_df['flag_cooks'].sum())}  "
          f"(cutoff = {infl_df['_cooks_cutoff'].iloc[0]:.4f})")
    print(f"  Leverage-flagged       : {int(infl_df['flag_leverage'].sum())}  "
          f"(cutoff = {infl_df['_leverage_cutoff'].iloc[0]:.4f})")
    print(f"  DFFITS-flagged         : {int(infl_df['flag_dffits'].sum())}  "
          f"(cutoff = {infl_df['_dffits_cutoff'].iloc[0]:.4f})")
    print(f"  Any-flagged            : {len(flagged_any)}")

    print(f"\n  Top 10 most influential cues (by Cook's distance):")
    print(f"  {'cue':30s}  {'n_attempts':>10s}  {'cook':>7s}  {'lev':>6s}  "
          f"{'dffits':>7s}  flags")
    for _, r in infl_df.head(10).iterrows():
        flags = []
        if r["flag_cooks"]:    flags.append("C")
        if r["flag_leverage"]: flags.append("L")
        if r["flag_dffits"]:   flags.append("D")
        flag_str = "".join(flags) if flags else "-"
        print(f"  {str(r['cue_subtype'])[:30]:30s}  "
              f"{int(r['n_child_attempts']):>10d}  "
              f"{r['cooks_distance']:7.4f}  "
              f"{r['leverage']:6.4f}  "
              f"{r['dffits']:+7.4f}  {flag_str}")

    # Refit excluding influentials
    print(f"\n  Refit with influentials excluded:")
    refit = refit_excluding_influentials(df, infl_df, args.exclude_rate,
                                          args.target_effect)
    for key, val in refit.items():
        if "error" in val:
            print(f"    {key:30s}: {val['error']}")
        else:
            print(f"    {key:30s}: n = {val['n']:3d}  "
                  f"β = {val['beta_z']:+.4f}  p = {val['p']:.4g}  "
                  f"CI = [{val['ci95_low']:+.4f}, {val['ci95_high']:+.4f}]")

    # Save influence CSV
    infl_csv = out_dir / f"{args.language}_influence_diagnostics.csv"
    infl_df.to_csv(infl_csv, index=False)
    print(f"\n  Influence CSV: {infl_csv}")

    # ── (2) Bootstrap CI ──
    print("\n" + "=" * 70)
    print(f"(2) Bootstrap CI for {args.target_effect}  (n_boot = {args.n_bootstrap})")
    print("=" * 70)
    boot = bootstrap_ci(df, args.exclude_rate, args.target_effect,
                         args.n_bootstrap, args.random_seed)

    print(f"\n  Observed β        : {boot['observed_beta']:+.4f}")
    print(f"  Bootstrap mean    : {boot['bootstrap_mean']:+.4f}")
    print(f"  Bootstrap median  : {boot['bootstrap_median']:+.4f}")
    print(f"  Bootstrap SE      : {boot['bootstrap_se']:.4f}")
    print(f"  Percentile 95% CI : [{boot['ci95_percentile'][0]:+.4f}, "
          f"{boot['ci95_percentile'][1]:+.4f}]")
    print(f"  BCa 95% CI        : [{boot['ci95_bca'][0]:+.4f}, "
          f"{boot['ci95_bca'][1]:+.4f}]")
    print(f"  Empirical p (2-tailed): {boot['empirical_p_two_sided']:.4g}")
    print(f"  z0 (bias)         : {boot['z0']:+.4f}")
    print(f"  acceleration      : {boot['acceleration']:+.4f}")

    # Save bootstrap distribution (first 1000 for JSON portability)
    boot_dist_csv = out_dir / f"{args.language}_bootstrap_distribution.csv"
    pd.DataFrame({"bootstrap_beta": boot["bootstrap_distribution"]}).to_csv(
        boot_dist_csv, index=False
    )

    # ── (3) VIF (bonus) ──
    print("\n" + "=" * 70)
    print("(3) Variance Inflation Factors (VIF) for M2 predictors")
    print("=" * 70)
    vifs = vif_table(df, args.exclude_rate)
    for name, v in vifs.items():
        marker = " ←⚠ HIGH" if v > 10 else (" ← elevated" if v > 5 else "")
        print(f"  {name:20s}: VIF = {v:.3f}{marker}")

    # ── Summary JSON ──
    summary = {
        "language":             args.language,
        "target_effect":        args.target_effect,
        "exclude_rate":         bool(args.exclude_rate),
        "n_cues":               int(len(df)),
        "original_estimate":    target_p_row,
        "influence_summary": {
            "n_total":          len(infl_df),
            "n_cooks_flagged":  int(infl_df["flag_cooks"].sum()),
            "n_leverage_flagged": int(infl_df["flag_leverage"].sum()),
            "n_dffits_flagged": int(infl_df["flag_dffits"].sum()),
            "n_any_flagged":    int(infl_df["flag_any"].sum()),
            "cooks_cutoff":     float(infl_df["_cooks_cutoff"].iloc[0]),
            "leverage_cutoff":  float(infl_df["_leverage_cutoff"].iloc[0]),
            "dffits_cutoff":    float(infl_df["_dffits_cutoff"].iloc[0]),
            "top_5_by_cooks":   infl_df.head(5)[
                ["cue_subtype", "n_child_attempts", "cooks_distance",
                 "leverage", "dffits", "flag_any"]
            ].to_dict(orient="records"),
        },
        "refit_excluding_influentials": refit,
        "bootstrap": {k: v for k, v in boot.items() if k != "bootstrap_distribution"},
        "vif":                  vifs,
    }
    summary_json = out_dir / f"{args.language}_robustness_quick.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Summary JSON: {summary_json}")
    print(f"  Bootstrap dist CSV: {boot_dist_csv}")

    # ── Final verdict line ──
    print("\n" + "=" * 70)
    print("Robustness verdict")
    print("=" * 70)
    orig_sign = np.sign(target_p_row["beta_z"])
    bca_lo, bca_hi = boot["ci95_bca"]
    ci_excludes_zero = (bca_lo > 0 and bca_hi > 0) or (bca_lo < 0 and bca_hi < 0)
    print(f"  Original β sign           : {'+' if orig_sign > 0 else '-'}")
    print(f"  Bootstrap BCa CI excludes 0: {ci_excludes_zero}")
    print(f"  After Cook's exclusion β  : "
          f"{refit['excluding_cooks_flagged'].get('beta_z', 'n/a')}")
    print(f"  Any VIF > 10              : "
          f"{any(v > 10 for v in vifs.values() if not np.isnan(v))}")


if __name__ == "__main__":
    main()
