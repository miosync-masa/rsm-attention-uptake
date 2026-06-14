"""
05_developmental_uptake.py
==========================
IMT Attention Bias Paper — Step 5: Developmental Uptake Analysis

THE CORE HYPOTHESIS TEST.

Steps 02-04 characterized CAREGIVER speech: which cues carry high Attention
Index (AI) in the input. This script asks the question that actually tests the
hypothesis:

    Do children acquire high-AI cues earlier and more strongly than low-AI cues?

It does so by:
  1. Reading the caregiver-side AI values (from 03_v3 output) as the PREDICTOR.
  2. Re-applying the same cue extractor to CHILD utterances.
  3. Binning child cue production by age (months).
  4. Computing, per cue_subtype:
       - first_emergence_month : earliest month the child produces the cue
       - uptake_slope          : rate of increase in production rate over age
       - peak_rate             : max production rate per 1k child utterances
  5. Correlating caregiver AI against these developmental measures.

Output:
  {lang}_uptake.csv          : per-cue developmental measures + caregiver AI
  {lang}_uptake_summary.json : correlations (AI vs emergence, AI vs slope)
  {lang}_uptake_by_age.csv   : long-format age-bin x cue production rates

Usage:
  python 05_developmental_uptake.py \\
      --tagged_csv ./output/v2/English_tokens_tagged.csv \\
      --ai_csv ./output/v3/English_attention_index.csv \\
      --language English \\
      --output_dir ./output/v3/

Note: --tagged_csv must be the 02_v2 output, which already contains cue tags
for ALL speakers (both caregiver and child), plus age_months and speaker_role.

Author: Torami x Boss | IMT Attention project | 2026-06-14
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pandas scipy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Age binning
# ─────────────────────────────────────────────────────────────────────────────

def assign_age_bins(df: pd.DataFrame, bin_width: int = 3) -> pd.DataFrame:
    """
    Add an 'age_bin' column. Bins are [start, start+bin_width) in months.
    Rows without age_months are dropped from developmental analysis.
    """
    work = df[df["age_months"].notna()].copy()
    work["age_months"] = pd.to_numeric(work["age_months"], errors="coerce")
    work = work[work["age_months"].notna()]
    work["age_bin"] = (work["age_months"] // bin_width * bin_width).astype(int)
    return work


# ─────────────────────────────────────────────────────────────────────────────
# Child production rates per cue per age bin
# ─────────────────────────────────────────────────────────────────────────────

def compute_child_production(
    df: pd.DataFrame, bin_width: int = 3, min_utts_per_bin: int = 50
) -> pd.DataFrame:
    """
    For child speech, compute production rate of each cue_subtype per age bin.

    Rate = (cue token count in bin) / (total child utterances in bin) * 1000
         = cues per 1,000 child utterances.

    Normalizing by utterance count (not token count) controls for the fact that
    older children simply talk more. We want RELATIVE prominence of each cue.
    """
    child = df[df["speaker_role"] == "child"].copy()
    if len(child) == 0:
        print("  WARNING: no child speech found.")
        return pd.DataFrame()

    child = assign_age_bins(child, bin_width)
    if len(child) == 0:
        print("  WARNING: no child speech with age_months.")
        return pd.DataFrame()

    # Total child utterances per age bin (denominator).
    # One utterance = one (file, utterance_index) pair.
    utt_counts = (
        child.groupby("age_bin")
        .apply(lambda g: g[["file", "utterance_index"]].drop_duplicates().shape[0])
        .rename("n_child_utterances")
    )

    # Keep only bins with enough utterances for a stable rate
    valid_bins = utt_counts[utt_counts >= min_utts_per_bin].index.tolist()
    if not valid_bins:
        print(f"  WARNING: no age bin reaches min_utts_per_bin={min_utts_per_bin}.")
        # fall back to all bins
        valid_bins = utt_counts.index.tolist()

    child = child[child["age_bin"].isin(valid_bins)]

    # Cue tokens only
    cue_child = child[child["is_cue_token"].astype(bool)].copy()

    rows = []
    for (age_bin, cue_subtype), group in cue_child.groupby(["age_bin", "cue_subtype"]):
        n_cue = len(group)
        n_utt = utt_counts.get(age_bin, np.nan)
        rate = (n_cue / n_utt * 1000) if n_utt and n_utt > 0 else np.nan
        rows.append({
            "age_bin": age_bin,
            "cue_subtype": cue_subtype,
            "cue_type": group["cue_type"].iloc[0],
            "n_cue_tokens": n_cue,
            "n_child_utterances": int(n_utt) if not np.isnan(n_utt) else None,
            "rate_per_1k": round(rate, 3) if not np.isnan(rate) else None,
        })

    long_df = pd.DataFrame(rows)
    return long_df


# ─────────────────────────────────────────────────────────────────────────────
# Per-cue developmental measures
# ─────────────────────────────────────────────────────────────────────────────

def compute_developmental_measures(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the age-bin long format into per-cue developmental measures:
      - first_emergence_month : smallest age_bin where rate_per_1k > 0
      - peak_rate             : max rate_per_1k across bins
      - peak_month            : age_bin of peak
      - uptake_slope          : OLS slope of rate_per_1k ~ age_bin
      - n_bins_present        : number of bins where cue is produced
    """
    if len(long_df) == 0:
        return pd.DataFrame()

    rows = []
    for cue_subtype, group in long_df.groupby("cue_subtype"):
        g = group[group["rate_per_1k"].notna()].sort_values("age_bin")
        produced = g[g["rate_per_1k"] > 0]
        if len(produced) == 0:
            continue

        first_emergence = int(produced["age_bin"].min())
        peak_idx = g["rate_per_1k"].idxmax()
        peak_rate = float(g.loc[peak_idx, "rate_per_1k"])
        peak_month = int(g.loc[peak_idx, "age_bin"])
        n_bins_present = int((g["rate_per_1k"] > 0).sum())

        # Slope via OLS (needs >= 2 points)
        if len(g) >= 2:
            x = g["age_bin"].values.astype(float)
            y = g["rate_per_1k"].values.astype(float)
            slope, intercept, r, p, se = stats.linregress(x, y)
            uptake_slope = float(slope)
            slope_r = float(r)
        else:
            uptake_slope = np.nan
            slope_r = np.nan

        rows.append({
            "cue_subtype": cue_subtype,
            "cue_type": group["cue_type"].iloc[0],
            "first_emergence_month": first_emergence,
            "peak_rate_per_1k": round(peak_rate, 3),
            "peak_month": peak_month,
            "uptake_slope": round(uptake_slope, 4) if not np.isnan(uptake_slope) else None,
            "uptake_slope_r": round(slope_r, 4) if not np.isnan(slope_r) else None,
            "n_bins_present": n_bins_present,
            "total_child_cue_tokens": int(group["n_cue_tokens"].sum()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Merge caregiver AI with child developmental measures + correlations
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_correlate(
    dev_df: pd.DataFrame, ai_df: pd.DataFrame, min_child_tokens: int = 5
) -> tuple:
    """
    Join caregiver AI (predictor) with child developmental measures (outcome),
    then compute correlations that test the hypothesis.

    Returns (merged_df, correlations_dict).
    """
    if len(dev_df) == 0 or len(ai_df) == 0:
        return pd.DataFrame(), {}

    # Caregiver-side predictors
    ai_cols = ["cue_subtype", "cue_type", "count", "AttentionIndex",
               "WeightedAttentionIndex", "Reliability_gra",
               "Reliability_position", "Reliability_form",
               "dominant_gra_relation"]
    ai_cols = [c for c in ai_cols if c in ai_df.columns]
    ai_slim = ai_df[ai_cols].rename(columns={
        "count": "caregiver_count",
        "AttentionIndex": "caregiver_AI",
        "WeightedAttentionIndex": "caregiver_WAI",
        "cue_type": "cue_type_cg",
    })

    merged = dev_df.merge(ai_slim, on="cue_subtype", how="inner")

    # Filter to cues with enough child production for reliable measures
    merged = merged[merged["total_child_cue_tokens"] >= min_child_tokens].copy()

    if len(merged) < 3:
        print(f"  WARNING: only {len(merged)} cues after merge/filter; "
              "correlations unreliable.")

    correlations = {}

    def safe_corr(x, y, name):
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            return None
        xv, yv = x[mask], y[mask]
        if xv.std() == 0 or yv.std() == 0:
            return None
        r_p, p_p = stats.pearsonr(xv, yv)
        r_s, p_s = stats.spearmanr(xv, yv)
        return {
            "n": int(mask.sum()),
            "pearson_r": round(float(r_p), 4),
            "pearson_p": round(float(p_p), 5),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": round(float(p_s), 5),
        }

    # ── THE KEY TESTS ──────────────────────────────────────────────────────
    # H1: higher caregiver AI → earlier child emergence (negative correlation)
    correlations["AI_vs_emergence"] = safe_corr(
        merged["caregiver_AI"], merged["first_emergence_month"],
        "AI vs emergence")
    # H2: higher caregiver AI → steeper uptake slope (positive)
    correlations["AI_vs_uptake_slope"] = safe_corr(
        merged["caregiver_AI"], merged["uptake_slope"],
        "AI vs slope")
    # H3: higher caregiver AI → higher peak production (positive)
    correlations["AI_vs_peak_rate"] = safe_corr(
        merged["caregiver_AI"], merged["peak_rate_per_1k"],
        "AI vs peak")

    # Controls: does raw frequency explain it better than AI?
    if "caregiver_count" in merged.columns:
        merged["log_caregiver_count"] = np.log1p(merged["caregiver_count"])
        correlations["logFreq_vs_emergence"] = safe_corr(
            merged["log_caregiver_count"], merged["first_emergence_month"],
            "logFreq vs emergence")
        correlations["logFreq_vs_peak_rate"] = safe_corr(
            merged["log_caregiver_count"], merged["peak_rate_per_1k"],
            "logFreq vs peak")

    # WAI variants (frequency-weighted AI)
    if "caregiver_WAI" in merged.columns:
        correlations["WAI_vs_emergence"] = safe_corr(
            merged["caregiver_WAI"], merged["first_emergence_month"],
            "WAI vs emergence")
        correlations["WAI_vs_peak_rate"] = safe_corr(
            merged["caregiver_WAI"], merged["peak_rate_per_1k"],
            "WAI vs peak")

    # Reliability variants (does reliable cue get learned faster?)
    for rel in ["Reliability_position", "Reliability_form"]:
        if rel in merged.columns:
            correlations[f"{rel}_vs_emergence"] = safe_corr(
                merged[rel], merged["first_emergence_month"],
                f"{rel} vs emergence")

    return merged, correlations


# ─────────────────────────────────────────────────────────────────────────────
# v2 ADDITION: Multiple regression — disentangling AI, frequency, interaction
# This is the statistical core that tests RSM against the frequency-confound.
# ─────────────────────────────────────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    """Center and scale to unit SD; returns 0 where SD is 0."""
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s - s.mean()
    return (s - s.mean()) / sd


def regression_analysis(merged: pd.DataFrame, outcome: str) -> dict:
    """
    Fit hierarchical regressions to separate the contributions of
    Attention Index, frequency, and their interaction.

    Models (all predictors z-scored; interaction = product of z-scores
    to avoid collinearity):
      M0: outcome ~ logFreq
      M1: outcome ~ logFreq + AI
      M2: outcome ~ logFreq + AI + AI:logFreq

    Reports per-model R2, incremental Delta-R2, and the interaction
    coefficient with its p-value. A significant positive interaction is
    the RSM signature: cues are produced most when BOTH attention-attracting
    AND frequent (= reactively confirmed).

    Also computes semipartial correlations:
      sr(AI | logFreq)   = unique variance AI adds beyond frequency
      sr(logFreq | AI)   = unique variance frequency adds beyond AI
    """
    import statsmodels.api as sm

    work = merged[[outcome, "caregiver_AI", "caregiver_count"]].dropna().copy()
    if len(work) < 8:
        return {"error": f"too few cues for regression (n={len(work)})"}

    work["z_AI"] = _zscore(work["caregiver_AI"])
    work["z_logFreq"] = _zscore(np.log1p(work["caregiver_count"]))
    work["z_inter"] = work["z_AI"] * work["z_logFreq"]
    # v3: standardize the OUTCOME too, so interaction betas are comparable
    # across languages (fully standardized solution).
    work["z_outcome"] = _zscore(work[outcome])
    y = work["z_outcome"].values

    def fit(cols):
        X = sm.add_constant(work[cols], has_constant="add")
        return sm.OLS(y, X).fit()

    m0 = fit(["z_logFreq"])
    m1 = fit(["z_logFreq", "z_AI"])
    m2 = fit(["z_logFreq", "z_AI", "z_inter"])

    # Semipartial (part) correlation:
    # sr(AI|freq)^2 = R2(full) - R2(without AI)
    m_noAI = fit(["z_logFreq"])
    m_noFreq = fit(["z_AI"])
    m_both = m1
    sr_AI_sq = max(0.0, m_both.rsquared - m_noAI.rsquared)
    sr_Freq_sq = max(0.0, m_both.rsquared - m_noFreq.rsquared)

    def coef_info(model, name):
        if name not in model.params.index:
            return None
        return {
            "beta": round(float(model.params[name]), 4),
            "p": round(float(model.pvalues[name]), 5),
            "ci_low": round(float(model.conf_int().loc[name, 0]), 4),
            "ci_high": round(float(model.conf_int().loc[name, 1]), 4),
        }

    return {
        "outcome": outcome,
        "n": int(len(work)),
        "M0_freq_only_R2": round(float(m0.rsquared), 4),
        "M1_freq_AI_R2": round(float(m1.rsquared), 4),
        "M2_full_R2": round(float(m2.rsquared), 4),
        "deltaR2_AI": round(float(m1.rsquared - m0.rsquared), 4),
        "deltaR2_interaction": round(float(m2.rsquared - m1.rsquared), 4),
        "beta_AI": coef_info(m1, "z_AI"),
        "beta_logFreq": coef_info(m1, "z_logFreq"),
        "beta_interaction": coef_info(m2, "z_inter"),
        "semipartial_AI_r": round(float(np.sqrt(sr_AI_sq)), 4),
        "semipartial_logFreq_r": round(float(np.sqrt(sr_Freq_sq)), 4),
        "interaction_sign": ("positive" if m2.params.get("z_inter", 0) > 0
                             else "negative"),
    }


def run_regressions(merged: pd.DataFrame) -> dict:
    """Run regression_analysis for the two key outcomes."""
    results = {}
    # peak production: higher = more uptake (interaction expected POSITIVE)
    if "peak_rate_per_1k" in merged.columns:
        results["peak_rate_per_1k"] = regression_analysis(
            merged, "peak_rate_per_1k")
    # emergence month: lower = earlier (interaction expected NEGATIVE)
    if "first_emergence_month" in merged.columns:
        results["first_emergence_month"] = regression_analysis(
            merged, "first_emergence_month")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Developmental uptake analysis.")
    parser.add_argument("--tagged_csv", required=True,
                        help="02_v2 output (has child+caregiver, age_months, cue tags)")
    parser.add_argument("--ai_csv", required=True,
                        help="03_v3 output (caregiver AI values)")
    parser.add_argument("--language", required=True)
    parser.add_argument("--output_dir", default="./output/v3")
    parser.add_argument("--bin_width", type=int, default=3,
                        help="Age bin width in months (default 3)")
    parser.add_argument("--min_utts_per_bin", type=int, default=50)
    parser.add_argument("--min_child_tokens", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"DEVELOPMENTAL UPTAKE: {args.language}")
    print(f"{'='*64}")

    print(f"Loading tagged: {args.tagged_csv}")
    df = pd.read_csv(args.tagged_csv, low_memory=False)
    print(f"  {len(df):,} tokens.")

    # Quick child-speech sanity check
    n_child = (df["speaker_role"] == "child").sum()
    n_child_aged = df[(df["speaker_role"] == "child")
                      & df["age_months"].notna()].shape[0]
    print(f"  Child tokens: {n_child:,}  (with age_months: {n_child_aged:,})")
    if n_child_aged == 0:
        print("  ERROR: no age-tagged child speech. Cannot compute uptake.")
        sys.exit(1)

    print(f"Loading caregiver AI: {args.ai_csv}")
    ai_df = pd.read_csv(args.ai_csv)
    print(f"  {len(ai_df)} caregiver cue subtypes.")

    # Step 1: child production rates per age bin
    print("\nComputing child production rates by age bin...")
    long_df = compute_child_production(
        df, bin_width=args.bin_width, min_utts_per_bin=args.min_utts_per_bin)
    if len(long_df) == 0:
        print("  ERROR: no child production computed.")
        sys.exit(1)
    long_path = out_dir / f"{args.language}_uptake_by_age.csv"
    long_df.to_csv(long_path, index=False)
    print(f"  Wrote: {long_path}  ({len(long_df)} rows)")

    age_bins_present = sorted(long_df["age_bin"].unique())
    print(f"  Age bins (months): {age_bins_present}")

    # Step 2: per-cue developmental measures
    print("\nComputing per-cue developmental measures...")
    dev_df = compute_developmental_measures(long_df)
    print(f"  {len(dev_df)} cues with child production.")

    # Step 3: merge with caregiver AI + correlate
    print("\nMerging with caregiver AI and computing correlations...")
    merged, correlations = merge_and_correlate(
        dev_df, ai_df, min_child_tokens=args.min_child_tokens)
    merged_path = out_dir / f"{args.language}_uptake.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  Wrote: {merged_path}  ({len(merged)} cues matched)")

    # Step 3b: hierarchical regressions (AI vs frequency vs interaction)
    print("\nRunning hierarchical regressions (RSM interaction test)...")
    regressions = run_regressions(merged)

    # Step 4: summary
    summary = {
        "language": args.language,
        "n_child_tokens": int(n_child),
        "n_child_tokens_aged": int(n_child_aged),
        "age_bins_months": [int(b) for b in age_bins_present],
        "bin_width_months": args.bin_width,
        "n_cues_matched": len(merged),
        "correlations": correlations,
        "regressions": regressions,
    }
    summary_path = out_dir / f"{args.language}_uptake_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Wrote: {summary_path}")

    # Pretty-print the key result
    print(f"\n{'─'*64}")
    print("KEY RESULT — Hypothesis tests:")
    print(f"{'─'*64}")

    def show(name, label, direction):
        c = correlations.get(name)
        if c is None:
            print(f"  {label}: (insufficient data)")
            return
        sig = "***" if c["pearson_p"] < 0.001 else \
              "**" if c["pearson_p"] < 0.01 else \
              "*" if c["pearson_p"] < 0.05 else "ns"
        print(f"  {label}:")
        print(f"      Pearson r={c['pearson_r']:+.3f} (p={c['pearson_p']:.4f}) {sig}"
              f"   Spearman ρ={c['spearman_r']:+.3f}   n={c['n']}")
        print(f"      → expected {direction}")

    show("AI_vs_emergence", "H1: caregiver AI vs child emergence month",
         "NEGATIVE (high AI → earlier)")
    show("AI_vs_uptake_slope", "H2: caregiver AI vs uptake slope",
         "POSITIVE (high AI → faster)")
    show("AI_vs_peak_rate", "H3: caregiver AI vs peak production",
         "POSITIVE (high AI → more)")
    print()
    show("logFreq_vs_emergence", "CONTROL: log-freq vs emergence",
         "compare against H1")
    show("WAI_vs_peak_rate", "WAI vs peak production",
         "POSITIVE")
    print(f"{'─'*64}")

    # ── Regression results — the RSM interaction test ──────────────────────
    print(f"\n{'═'*64}")
    print("REGRESSION — Does AI×Frequency interaction predict uptake? (RSM)")
    print(f"{'═'*64}")
    for outcome, reg in regressions.items():
        if reg is None or "error" in reg:
            print(f"\n  [{outcome}] {reg.get('error') if reg else 'n/a'}")
            continue
        print(f"\n  Outcome: {outcome}  (n={reg['n']})")
        print(f"    Model R²:   freq-only={reg['M0_freq_only_R2']:.3f}"
              f"  → +AI={reg['M1_freq_AI_R2']:.3f}"
              f"  → +interaction={reg['M2_full_R2']:.3f}")
        print(f"    ΔR² from AI:          {reg['deltaR2_AI']:+.3f}")
        print(f"    ΔR² from interaction: {reg['deltaR2_interaction']:+.3f}  "
              "← RSM signature")
        bi = reg["beta_interaction"]
        if bi:
            sig = "***" if bi["p"] < 0.001 else "**" if bi["p"] < 0.01 \
                  else "*" if bi["p"] < 0.05 else "ns"
            print(f"    Interaction β = {bi['beta']:+.3f} "
                  f"(p={bi['p']:.4f}) {sig}  [{reg['interaction_sign']}]")
        print(f"    Semipartial r:  AI|freq={reg['semipartial_AI_r']:.3f}   "
              f"freq|AI={reg['semipartial_logFreq_r']:.3f}")
        ba, bf = reg["beta_AI"], reg["beta_logFreq"]
        if ba and bf:
            print(f"    Standardized β: AI={ba['beta']:+.3f} (p={ba['p']:.3f})   "
                  f"logFreq={bf['beta']:+.3f} (p={bf['p']:.3f})")
    print(f"{'═'*64}")
    print("\nRSM interpretation:")
    print("  A significant POSITIVE interaction on peak production means cues")
    print("  are produced most when BOTH attention-attracting AND frequent —")
    print("  i.e. attention and reactive repetition multiply, as RSM predicts.")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
