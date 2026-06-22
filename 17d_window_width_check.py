"""
17d_window_width_check.py
=========================
IMT Attention Bias Paper 2 — Step 17d:
Window-width sensitivity check for the exposure-gate effect.

Tests whether the per-child slope β_i(COI × cumulative_cue_attempts)
observed in 17c is a function of the child's *observational window*:
the age range during which post-MSR episodes were recorded.

Hypothesis (pass / falsify):
  PASS    β(obs_window_months) > 0, p < 0.05 in the moderator WLS,
          AND pooled β monotonically increases across
          [<6mo, 6–12mo, >12mo] strata.
          → "Manchester's null comes from short observational windows."
  FAIL    β(obs_window_months) ≈ 0 and strata are flat
          → "Manchester is genuinely null; window is not the explanation."

────────────────────────────────────────────────────────────────────────────

Inputs
------
* output/v17c/per_child_betas_N{N}.csv   per-child β + SE
* output/v16/{lang}_episodes_with_reuse.csv  to compute per-child window
* output/json_cache/{lang}/...           for file→child mapping

Outputs (output/v17d/)
----------------------
* per_child_with_window_N{N}.csv      per-child β + window stats
* window_moderator_results_N{N}.json  WLS coefficients & stratified meta
* SUMMARY.md                          one-page table

Author: Torami x Boss | IMT Attention project | Paper 2 / window check v1 | 2026-06-21
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":        "Brown",
        "reuse_csv":    "./output/v16/English_episodes_with_reuse.csv",
        "json_cache":   "./output/json_cache/English",
        "tagged_csv":   "./output/English_tokens_tagged.csv",
        "joined_csv":   "./output/v11_runA/English_r_plus_joined.csv",
    },
    "English-Manchester": {
        "label":        "Manchester",
        "reuse_csv":    "./output/v16/English-Manchester_episodes_with_reuse.csv",
        "json_cache":   "./output/json_cache/English-Manchester",
        "tagged_csv":   "./output/English-Manchester_tokens_tagged.csv",
        "joined_csv":   "./output/v11/English-Manchester_r_plus_joined.csv",
    },
    "English-UK": {
        "label":        "English-UK",
        "reuse_csv":    "./output/v16/English-UK_episodes_with_reuse.csv",
        "json_cache":   "./output/json_cache/English-UK",
        "tagged_csv":   "./output/English-UK_tokens_tagged.csv",
        "joined_csv":   "./output/v11_runA/English-UK_r_plus_joined.csv",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def compute_per_child_window(reuse_csv: Path, json_cache: Path,
                              joined_csv: Path,
                              contingent_only: bool) -> pd.DataFrame:
    """
    Per-child statistics on the post-MSR subset, matching 17c's working set:
      n_post:                episodes (post-MSR contingent ∩ baseline)
      age_min_post, age_max_post, obs_window_months
      cumulative_min, cumulative_max, cumulative_range
    """
    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()

    # Attach child id
    file_to_child, max_len = build_file_child_map(json_cache)
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()

    # Drop rows with no COI/logFreq (mirror 17c filter)
    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    keep = joined[["cue_subtype", "COI", "log_cue_freq"]].copy()
    df = reuse.merge(keep, on="cue_subtype", how="inner")
    df = df.dropna(subset=["COI", "log_cue_freq", "child_age_months"]).copy()

    # Cumulative cue attempts (must match 17c's definition)
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()

    df_post = df[df["child_age_months"] >= 24].copy()
    if df_post.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for child_id, g in df_post.groupby("child"):
        rows.append({
            "child":                child_id,
            "n_post":               int(len(g)),
            "n_cues_post":          int(g["cue_subtype"].nunique()),
            "age_min_post":         float(g["child_age_months"].min()),
            "age_max_post":         float(g["child_age_months"].max()),
            "obs_window_months":    float(g["child_age_months"].max() - g["child_age_months"].min()),
            "cumulative_min":       int(g["cumulative_cue_attempts"].min()),
            "cumulative_max":       int(g["cumulative_cue_attempts"].max()),
            "cumulative_range":     int(g["cumulative_cue_attempts"].max() - g["cumulative_cue_attempts"].min()),
            "cumulative_mean":      float(g["cumulative_cue_attempts"].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DerSimonian-Laird meta-analysis
# ─────────────────────────────────────────────────────────────────────────────

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
    z = beta_re / se_re if se_re > 0 else float("nan")
    p = float(2 * (1.0 - norm.cdf(abs(z)))) if not math.isnan(z) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": z, "pooled_p_RE": p,
        "pooled_beta_FE": beta_fe,
        "pooled_se_FE": float(np.sqrt(1.0 / sum_w)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Window-width sensitivity for exposure-gate β.")
    parser.add_argument("--per_child_csv", default="./output/v17c/per_child_betas_N5.csv")
    parser.add_argument("--output_dir",    default="./output/v17d")
    parser.add_argument("--window",        type=int, default=5)
    parser.add_argument("--include_noncontingent", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-child β table: {args.per_child_csv}")
    per_child = pd.read_csv(args.per_child_csv)
    needed_cols = {"child", "corpus_label", "n_episodes", "beta_COI_x_cum", "se_COI_x_cum"}
    missing = needed_cols - set(per_child.columns)
    if missing:
        sys.exit(f"per_child_csv is missing columns: {missing}")

    # Compute per-child window stats for each corpus and combine
    print("\nComputing per-child window stats per corpus...")
    contingent_only = not args.include_noncontingent
    pieces: List[pd.DataFrame] = []
    for lang, cfg in CANONICAL_CORPORA.items():
        label = cfg["label"]
        reuse_csv = Path(cfg["reuse_csv"])
        json_cache = Path(cfg["json_cache"])
        joined_csv = Path(cfg["joined_csv"])
        if not reuse_csv.exists() or not joined_csv.exists():
            print(f"  SKIP {label}: missing inputs")
            continue
        print(f"  {label}...")
        w = compute_per_child_window(reuse_csv, json_cache, joined_csv,
                                      contingent_only=contingent_only)
        if w.empty:
            print(f"    no post-MSR rows for {label}")
            continue
        w["corpus_label"] = label
        pieces.append(w)
    if not pieces:
        sys.exit("ERROR: no per-child window data collected.")
    window_df = pd.concat(pieces, ignore_index=True)

    merged = per_child.merge(window_df, on=["child", "corpus_label"], how="inner")
    print(f"\nMerged rows: {len(merged):,}  (per_child={len(per_child)},  window_stats={len(window_df)})")

    # Save the per-child table with window stats
    merged_csv = out_dir / f"per_child_with_window_N{args.window}.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"  → {merged_csv}")

    # ─────────────────────────────────────────────────────────────────────
    # MAIN: WLS regression β_i ~ obs_window_months + log(n_episodes) + corpus
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== MAIN: WLS β_i ~ obs_window_months + log(n_episodes) + corpus ===")
    df = merged.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum",
                                  "obs_window_months", "n_episodes"]).copy()
    df["log_n_episodes"] = np.log(df["n_episodes"])
    for c in ["obs_window_months", "log_n_episodes"]:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()

    dummies = pd.get_dummies(df["corpus_label"], prefix="corpus", drop_first=True).astype(float)
    df = pd.concat([df, dummies], axis=1)
    predictors = ["obs_window_months_z", "log_n_episodes_z"] + list(dummies.columns)

    X = sm.add_constant(df[predictors].astype(float))
    y = df["beta_COI_x_cum"].astype(float)
    w = 1.0 / (df["se_COI_x_cum"].astype(float) ** 2)
    wls = sm.WLS(y, X, weights=w).fit()

    wls_params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + predictors:
        if p in wls.params:
            wls_params[p] = {
                "beta": float(wls.params[p]),
                "se":   float(wls.bse[p]),
                "t":    float(wls.tvalues[p]),
                "p":    float(wls.pvalues[p]),
                "ci95_low":  float(wls.conf_int().loc[p, 0]),
                "ci95_high": float(wls.conf_int().loc[p, 1]),
            }
    print(f"  n_children = {len(df)}  R² = {wls.rsquared:.3f}")
    for p, v in wls_params.items():
        sig = '***' if v['p']<0.001 else ('**' if v['p']<0.01 else ('*' if v['p']<0.05 else ('†' if v['p']<0.10 else '')))
        print(f"    {p:<26}: β={v['beta']:+.4f}  SE={v['se']:.4f}  p={v['p']:.4f}  {sig}")

    # Also report the version with unstandardized obs_window (per month)
    X2 = sm.add_constant(df[["obs_window_months", "log_n_episodes"] + list(dummies.columns)].astype(float))
    wls2 = sm.WLS(y, X2, weights=w).fit()
    wls_params_raw: Dict[str, Dict[str, float]] = {}
    for p in ["const", "obs_window_months", "log_n_episodes"] + list(dummies.columns):
        if p in wls2.params:
            wls_params_raw[p] = {
                "beta": float(wls2.params[p]),
                "se":   float(wls2.bse[p]),
                "t":    float(wls2.tvalues[p]),
                "p":    float(wls2.pvalues[p]),
            }

    # ─────────────────────────────────────────────────────────────────────
    # STRATIFIED meta-analysis by obs_window
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STRATIFIED meta-analysis by obs_window ===")
    def _strata(months: float) -> str:
        if months < 6:    return "A_short_<6mo"
        if months < 12:   return "B_medium_6-12mo"
        return "C_long_>12mo"
    merged["window_stratum"] = merged["obs_window_months"].apply(_strata)
    strata_results: Dict[str, Any] = {}
    print(f"  {'Stratum':<22} {'n_children':>10} {'β_pooled':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6} {'corpora':>30}")
    for stratum, g in merged.groupby("window_stratum", sort=True):
        gg = g.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"])
        if len(gg) < 2:
            strata_results[stratum] = {"n_children": len(gg), "error": "n<2"}
            print(f"  {stratum:<22} {len(gg):>10}  (n<2, skipped)")
            continue
        meta = random_effects_meta(
            gg["beta_COI_x_cum"].values.astype(float),
            gg["se_COI_x_cum"].values.astype(float),
        )
        meta["n_children"] = len(gg)
        meta["corpora_breakdown"] = gg["corpus_label"].value_counts().to_dict()
        strata_results[stratum] = meta
        breakdown = "/".join([f"{k}:{v}" for k,v in gg["corpus_label"].value_counts().items()])
        print(f"  {stratum:<22} {len(gg):>10} {meta['pooled_beta_RE']:>+8.4f}  {meta['pooled_se_RE']:.4f}  {meta['pooled_p_RE']:.4f}  {meta['tau2']:.4f}  {meta['I2_pct']:>5.1f}%  {breakdown:>30}")

    # ─────────────────────────────────────────────────────────────────────
    # BONUS: Brown inner contrast
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== Brown inner contrast (Adam / Eve / Sarah) ===")
    brown = merged[merged["corpus_label"] == "Brown"].copy()
    brown = brown.sort_values("obs_window_months")
    if not brown.empty:
        print(f"  {'child':<8} {'age_min':>8} {'age_max':>8} {'window_mo':>10} {'n_eps':>7} {'β':>8} {'SE':>7} {'p':>7}")
        for _, r in brown.iterrows():
            sig = '***' if r['p_COI_x_cum']<0.001 else ('**' if r['p_COI_x_cum']<0.01 else ('*' if r['p_COI_x_cum']<0.05 else ('†' if r['p_COI_x_cum']<0.10 else '')))
            print(f"  {str(r['child']):<8} {r['age_min_post']:>8.1f} {r['age_max_post']:>8.1f} {r['obs_window_months']:>10.1f} {int(r['n_episodes']):>7,} {r['beta_COI_x_cum']:>+7.4f}  {r['se_COI_x_cum']:.4f}  {r['p_COI_x_cum']:.4f} {sig}")
        # Direct correlation between window and β within Brown (just 3 points, descriptive)
        if len(brown) >= 2:
            r_pw = np.corrcoef(brown["obs_window_months"], brown["beta_COI_x_cum"])[0, 1]
            print(f"  → Pearson r(window, β) within Brown n=3: r = {r_pw:+.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # Pass-criterion verdict
    # ─────────────────────────────────────────────────────────────────────
    obs_window_p = wls_params.get("obs_window_months_z", {}).get("p", float("nan"))
    obs_window_b = wls_params.get("obs_window_months_z", {}).get("beta", float("nan"))
    sig_pos_window = (obs_window_b > 0) and (obs_window_p < 0.05)

    # Monotonic check across strata
    ordered = ["A_short_<6mo", "B_medium_6-12mo", "C_long_>12mo"]
    betas_in_order: List[Optional[float]] = []
    for s in ordered:
        m = strata_results.get(s, {})
        b = m.get("pooled_beta_RE")
        betas_in_order.append(b if b is not None else None)
    valid = [b for b in betas_in_order if b is not None]
    is_monotonic = (
        len(valid) >= 2 and all(valid[i] <= valid[i+1] for i in range(len(valid)-1))
    )

    verdict = {
        "obs_window_beta":      obs_window_b,
        "obs_window_p":         obs_window_p,
        "obs_window_sig_pos":   bool(sig_pos_window),
        "strata_betas":         betas_in_order,
        "monotonic_across_strata": bool(is_monotonic),
    }
    if sig_pos_window and is_monotonic:
        verdict["overall"] = "PASS"
    elif (not sig_pos_window) and (len(valid) >= 2 and abs(max(valid) - min(valid)) < 0.01):
        verdict["overall"] = "FAIL — window is not the explanation"
    else:
        verdict["overall"] = "MIXED — see strata"

    # ─────────────────────────────────────────────────────────────────────
    # Persist results
    # ─────────────────────────────────────────────────────────────────────
    results = {
        "_meta": {
            "window":             args.window,
            "contingent_only":    contingent_only,
            "n_children_merged":  int(len(merged)),
        },
        "wls_main_z":        wls_params,
        "wls_main_raw":      wls_params_raw,
        "strata":            strata_results,
        "brown_inner":       brown.to_dict(orient="records"),
        "verdict":           verdict,
    }
    results_json = out_dir / f"window_moderator_results_N{args.window}.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → {results_json}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# SPEC #1d window-width sensitivity — outcome window N={args.window}\n")
    lines.append("## Pass criterion\n")
    lines.append("β(obs_window_months_z) > 0 AND p < 0.05 in WLS, "
                  "AND stratified pooled β monotonically increases with window width.\n")
    lines.append("\n## WLS β_i ~ obs_window_z + log(n_episodes)_z + corpus dummies "
                  "(weights = 1/SE²)\n")
    lines.append("| predictor | β | SE | p |")
    lines.append("|---|---|---|---|")
    for p, v in wls_params.items():
        sig = '***' if v['p']<0.001 else ('**' if v['p']<0.01 else ('*' if v['p']<0.05 else ('†' if v['p']<0.10 else '')))
        lines.append(f"| {p} | {v['beta']:+.4f} {sig} | {v['se']:.4f} | {v['p']:.4f} |")
    lines.append("\n## Stratified random-effects meta-analysis\n")
    lines.append("| Stratum | n_children | β_pooled (RE) | SE | p | τ² | I² | corpora breakdown |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in ordered:
        m = strata_results.get(s, {})
        if "pooled_beta_RE" not in m:
            lines.append(f"| {s} | {m.get('n_children','—')} | — | — | — | — | — | — |")
            continue
        breakdown = ", ".join([f"{k}={v}" for k,v in m.get("corpora_breakdown",{}).items()])
        lines.append(
            f"| {s} | {m['n_children']} | {m['pooled_beta_RE']:+.4f} | "
            f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
            f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% | {breakdown} |"
        )
    lines.append("\n## Brown inner contrast (Adam / Eve / Sarah)\n")
    if not brown.empty:
        lines.append("| child | age_min | age_max | window_mo | n_eps | β | SE | p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in brown.sort_values("obs_window_months").iterrows():
            lines.append(
                f"| {r['child']} | {r['age_min_post']:.1f} | {r['age_max_post']:.1f} | "
                f"{r['obs_window_months']:.1f} | {int(r['n_episodes']):,} | "
                f"{r['beta_COI_x_cum']:+.4f} | {r['se_COI_x_cum']:.4f} | {r['p_COI_x_cum']:.4f} |"
            )
    lines.append(f"\n## Verdict: **{verdict.get('overall','?')}**\n")
    summary_md = out_dir / "SUMMARY.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {summary_md}")

    print(f"\n========== VERDICT: {verdict.get('overall','?')} ==========")


if __name__ == "__main__":
    main()
