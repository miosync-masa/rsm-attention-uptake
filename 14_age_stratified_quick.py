"""
14_age_stratified_quick.py (v2 - uses joined CSV as baseline)
=============================================================
IMT Attention Bias Paper 2 — Step 14: Age-stratified R+ × COI analysis

Tests boss's MSR-precondition hypothesis (§4.6.9 of the discovery memo):

    H1: R+ × COI synergy emerges only in post-MSR child speech samples
        (mirror self-recognition threshold, ~18-24 months).

Architecture (v2):
  - The joined CSV (from 11_rsm_r_plus_join.py) supplies the cue-level
    baseline: logFreq, COI, outcome, n_child_attempts.
  - The episodes CSV (from 10_extract_R_plus_v2.py) supplies episode-level
    R+ measures with child_age_months.
  - For each age subset, R+ measures are RE-AGGREGATED to cue level and
    REPLACE the baseline R+ values. logFreq, COI, outcome remain at the
    full-period aggregate.

This tests the question:
    "Does post-MSR caregiver R+ depth (× COI) predict full-period
     cue stabilization more strongly than pre-MSR R+ depth does?"

Logic of variables held constant vs varied:
  - logFreq    : full-period (corpus-wide cue frequency)
  - COI        : full-period (structural property of cue)
  - outcome    : full-period (peak production rate per 1k)
  - RplusComposite, RplusRate : age-subset specific (recomputed)

Workflow:
  1. Load joined CSV (cue-level baseline)
  2. Load episodes CSV (for age-subset R+ recomputation)
  3. For each age threshold T (e.g., 24, 30 months):
       a. Aggregate R+ from episodes filtered to [, T) and [T, )
       b. Merge baseline (logFreq, COI, outcome) with subset R+
       c. Apply min_attempts filter
       d. Fit M2 regression on each subset
       e. Extract β(Rcomp × COI), SE, p, parametric CI
       f. Bootstrap CI
  4. Output comparison table

Usage:
  python 14_age_stratified_quick.py \\
      --joined_csv ./output/v11_runA/English-UK_r_plus_joined.csv \\
      --episodes_csv ./output/v10b/English-UK_r_plus_episodes.csv \\
      --language English-UK \\
      --output_dir ./output/v14/ \\
      --age_thresholds 24,30 \\
      --min_attempts 5 \\
      --n_bootstrap 2000 \\
      --random_seed 42

Author: Torami x Boss | IMT Attention project | Paper 2 / age-stratified v2 | 2026-06-20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Label normalization (handle both prefixed and non-prefixed)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NORM = {
    'no_contingent_response':  'no_contingent_response',
    '0_no_contingent_response':'no_contingent_response',
    'acknowledgment':          'acknowledgment',
    '1_acknowledgment':        'acknowledgment',
    'repetition':              'repetition',
    '2_repetition':            'repetition',
    'expansion':               'expansion',
    '3_expansion':             'expansion',
    'recast':                  'recast',
    '4_recast':                'recast',
}


def normalize_label(label: Any) -> str:
    if pd.isna(label):
        return 'no_contingent_response'
    return LABEL_NORM.get(str(label), str(label))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Age-stratified R+ × COI regression for MSR-precondition test (v2)."
    )
    p.add_argument("--joined_csv", required=True,
                   help="Joined CSV from 11_rsm_r_plus_join.py (cue-level baseline).")
    p.add_argument("--episodes_csv", required=True,
                   help="Episode-level CSV from 10_extract_R_plus_v2.py.")
    p.add_argument("--language", required=True)
    p.add_argument("--output_dir", default="./output/v14")
    p.add_argument("--age_thresholds", default="24",
                   help="Comma-separated age thresholds in months (e.g., '24,30').")
    p.add_argument("--exclude_rate", action="store_true",
                   help="If set, fit M2 without RplusRate (Spanish-style).")
    p.add_argument("--min_attempts", type=int, default=5,
                   help="Minimum n_child_attempts to include a cue.")
    p.add_argument("--n_bootstrap", type=int, default=2000,
                   help="Number of bootstrap resamples for CI.")
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Age-subset R+ aggregation (from episodes)
# ─────────────────────────────────────────────────────────────────────────────

def get_subset_r_plus(episodes: pd.DataFrame,
                       age_min: Optional[float],
                       age_max: Optional[float]) -> pd.DataFrame:
    """Compute R+ measures per cue from age-filtered episodes.

    Returns DataFrame with columns:
        cue_subtype, subset_n_episodes, subset_n_contingent,
        subset_RplusComposite, subset_RplusRate,
        subset_recast_rate, subset_expansion_rate, subset_mean_age
    """
    df = episodes.copy()
    if age_min is not None:
        df = df[df['child_age_months'] >= age_min]
    if age_max is not None:
        df = df[df['child_age_months'] < age_max]

    if len(df) == 0:
        return pd.DataFrame(columns=['cue_subtype', 'subset_n_episodes',
                                       'subset_n_contingent',
                                       'subset_RplusComposite', 'subset_RplusRate',
                                       'subset_recast_rate', 'subset_expansion_rate',
                                       'subset_mean_age'])

    # Explode pipe-delimited child_cues
    df['cue_list'] = df['child_cues'].fillna('').astype(str).str.split('|')
    df_exp = df.explode('cue_list')
    df_exp['cue_subtype'] = df_exp['cue_list'].str.strip()
    df_exp = df_exp[df_exp['cue_subtype'] != ''].copy()

    if len(df_exp) == 0:
        return pd.DataFrame(columns=['cue_subtype'])

    df_exp['_label'] = df_exp['r_plus_label'].apply(normalize_label)
    df_exp['_is_contingent'] = df_exp['_label'] != 'no_contingent_response'
    df_exp['_is_modeling']   = df_exp['_label'].isin(['expansion', 'recast'])
    df_exp['_is_recast']     = df_exp['_label'] == 'recast'
    df_exp['_is_expansion']  = df_exp['_label'] == 'expansion'

    grouped = df_exp.groupby('cue_subtype')

    rows: List[Dict[str, Any]] = []
    for cue, g in grouped:
        n_total = int(len(g))
        n_contingent = int(g['_is_contingent'].sum())
        n_modeling = int(g['_is_modeling'].sum())
        n_recast = int(g['_is_recast'].sum())
        n_expansion = int(g['_is_expansion'].sum())

        g_cont = g[g['_is_contingent']]
        if len(g_cont) > 0:
            mean_comp_cont = float(g_cont['r_plus_composite'].mean())
        else:
            mean_comp_cont = 0.0

        modeling_rate  = float(n_modeling)  / max(n_contingent, 1)
        recast_rate    = float(n_recast)    / max(n_contingent, 1)
        expansion_rate = float(n_expansion) / max(n_contingent, 1)
        mean_age = float(g['child_age_months'].mean())

        rows.append({
            'cue_subtype':            cue,
            'subset_n_episodes':      n_total,
            'subset_n_contingent':    n_contingent,
            'subset_n_modeling':      n_modeling,
            'subset_RplusComposite':  mean_comp_cont,
            'subset_RplusRate':       modeling_rate,
            'subset_recast_rate':     recast_rate,
            'subset_expansion_rate':  expansion_rate,
            'subset_mean_age':        mean_age,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# M2 fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_m2(df: pd.DataFrame, exclude_rate: bool) -> Dict[str, Any]:
    """Fit M2 on a dataset that has columns:
       logFreq, COI, RplusComposite, RplusRate (optional), outcome.
    Returns dict with Rcomp × COI estimate and Paper 1 baseline reference.
    """
    required = ['logFreq', 'COI', 'RplusComposite', 'outcome']
    if not exclude_rate:
        required.append('RplusRate')

    df_clean = df.dropna(subset=required).copy()
    if len(df_clean) < 10:
        return {'error': f'fewer than 10 cues (n={len(df_clean)})'}

    # z-score
    for col in required:
        m = df_clean[col].mean()
        s = df_clean[col].std()
        df_clean[f'{col}_z'] = (df_clean[col] - m) / s if s > 0 else 0.0

    df_clean['COI_x_logFreq'] = df_clean['COI_z'] * df_clean['logFreq_z']
    df_clean['Rcomp_x_COI']   = df_clean['RplusComposite_z'] * df_clean['COI_z']

    if exclude_rate:
        preds = ['logFreq_z', 'COI_z', 'COI_x_logFreq',
                 'RplusComposite_z', 'Rcomp_x_COI']
    else:
        preds = ['logFreq_z', 'COI_z', 'COI_x_logFreq',
                 'RplusRate_z', 'RplusComposite_z', 'Rcomp_x_COI']

    X = sm.add_constant(df_clean[preds].astype(float), has_constant='add')
    y = df_clean['outcome_z'].astype(float)

    try:
        m = sm.OLS(y, X).fit()
    except Exception as e:
        return {'error': f'OLS failed: {e}'}

    target = 'Rcomp_x_COI'
    return {
        'n_cues':              int(m.nobs),
        'r2_M2':               round(float(m.rsquared), 4),
        'r2_adj_M2':           round(float(m.rsquared_adj), 4),
        'beta_z':              round(float(m.params[target]), 4),
        'se':                  round(float(m.bse[target]), 4),
        't':                   round(float(m.tvalues[target]), 4),
        'p':                   float(m.pvalues[target]),
        'ci95_low':            round(float(m.conf_int().loc[target, 0]), 4),
        'ci95_high':           round(float(m.conf_int().loc[target, 1]), 4),
        'beta_logFreq':        round(float(m.params['logFreq_z']), 4),
        'beta_COI':            round(float(m.params['COI_z']), 4),
        'beta_COI_x_logFreq':  round(float(m.params['COI_x_logFreq']), 4),
        'p_COI_x_logFreq':     float(m.pvalues['COI_x_logFreq']),
        'beta_Rcomp_main':     round(float(m.params['RplusComposite_z']), 4),
        'p_Rcomp_main':        float(m.pvalues['RplusComposite_z']),
    }


def bootstrap_beta(df: pd.DataFrame, exclude_rate: bool,
                    n_boot: int, random_seed: int) -> Dict[str, Any]:
    """Bootstrap CI for β(Rcomp × COI)."""
    n = len(df)
    if n < 10:
        return {'error': f'too few cues for bootstrap (n={n})'}

    rng = np.random.default_rng(random_seed)
    betas: List[float] = []

    for _ in tqdm(range(n_boot), desc='    bootstrap', leave=False):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx].reset_index(drop=True)
        res = fit_m2(sub, exclude_rate)
        if res is not None and 'beta_z' in res:
            betas.append(res['beta_z'])

    if len(betas) < 100:
        return {'error': f'too few valid resamples ({len(betas)})'}

    betas_arr = np.array(betas)
    ci_lo, ci_hi = np.percentile(betas_arr, [2.5, 97.5])

    return {
        'n_bootstrap':  int(len(betas_arr)),
        'mean':         round(float(betas_arr.mean()), 4),
        'median':       round(float(np.median(betas_arr)), 4),
        'se':           round(float(np.std(betas_arr, ddof=1)), 4),
        'ci95_low':     round(float(ci_lo), 4),
        'ci95_high':    round(float(ci_hi), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subset analysis (v2: uses joined baseline)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_subset(label: str, joined_baseline: pd.DataFrame,
                    episodes: pd.DataFrame,
                    age_min: Optional[float], age_max: Optional[float],
                    args: argparse.Namespace) -> Dict[str, Any]:
    """Replace baseline R+ measures with age-subset specific values, then fit M2."""
    print(f"\n  ── {label} ──")
    if age_min is not None:
        print(f"    Age filter: ≥ {age_min} months")
    if age_max is not None:
        print(f"    Age filter: < {age_max} months")

    # Compute subset R+ measures from episodes
    r_plus_subset = get_subset_r_plus(episodes, age_min, age_max)
    n_episodes_in_subset = int(r_plus_subset['subset_n_episodes'].sum()) \
                            if len(r_plus_subset) else 0
    print(f"    Episodes in subset:     {n_episodes_in_subset:,}")
    print(f"    Cue subtypes in subset: {len(r_plus_subset)}")

    if len(r_plus_subset) == 0:
        return {'label': label, 'error': 'no episodes in subset'}

    # Baseline: logFreq, COI, outcome, n_child_attempts (from joined CSV)
    baseline_cols = ['cue_subtype', 'n_child_attempts', 'logFreq', 'COI', 'outcome']
    missing = [c for c in baseline_cols if c not in joined_baseline.columns]
    if missing:
        return {'label': label,
                'error': f"baseline joined CSV missing columns: {missing}"}

    base = joined_baseline[baseline_cols].copy()

    # Merge baseline with subset R+
    subset_joined = base.merge(r_plus_subset, on='cue_subtype', how='inner')

    # Apply min_attempts
    before = len(subset_joined)
    subset_joined = subset_joined[
        subset_joined['n_child_attempts'] >= args.min_attempts
    ].copy()
    print(f"    After min_attempts ≥ {args.min_attempts}: "
          f"{len(subset_joined)} (dropped {before - len(subset_joined)})")

    if len(subset_joined) < 10:
        return {'label': label,
                'n_episodes_in_subset': n_episodes_in_subset,
                'n_cues_after_min': len(subset_joined),
                'error': f'fewer than 10 cues after filtering (n={len(subset_joined)})'}

    # Rename to canonical column names for fit_m2
    subset_joined['RplusComposite'] = subset_joined['subset_RplusComposite']
    subset_joined['RplusRate']      = subset_joined['subset_RplusRate']

    mean_age = float(subset_joined['subset_mean_age'].mean())
    print(f"    Mean child age across cues: {mean_age:.2f} months")

    # Fit M2
    res = fit_m2(subset_joined, args.exclude_rate)
    if 'error' in res:
        print(f"    M2 fit failed: {res['error']}")
        return {'label': label,
                'n_episodes_in_subset': n_episodes_in_subset,
                'n_cues_after_min': int(len(subset_joined)),
                'error': res['error']}

    print(f"\n    M2 results:")
    print(f"      R²(M2)                = {res['r2_M2']:.4f}")
    print(f"      Paper 1 baseline (COI × logFreq):")
    print(f"        β = {res['beta_COI_x_logFreq']:+.4f}  "
          f"p = {res['p_COI_x_logFreq']:.4g}")
    print(f"      Paper 2 central:")
    print(f"        Rcomp main          β = {res['beta_Rcomp_main']:+.4f}  "
          f"p = {res['p_Rcomp_main']:.4g}")
    print(f"        Rcomp × COI         β = {res['beta_z']:+.4f}  "
          f"SE = {res['se']:.4f}  p = {res['p']:.4g}")
    print(f"        Parametric 95% CI   = [{res['ci95_low']:+.4f}, "
          f"{res['ci95_high']:+.4f}]")

    # Bootstrap
    print(f"\n    Bootstrap CI (n={args.n_bootstrap}):")
    boot = bootstrap_beta(subset_joined, args.exclude_rate,
                           args.n_bootstrap, args.random_seed)
    if 'error' in boot:
        print(f"      ERROR: {boot['error']}")
    else:
        print(f"      Bootstrap mean        = {boot['mean']:+.4f}")
        print(f"      Bootstrap median      = {boot['median']:+.4f}")
        print(f"      Bootstrap SE          = {boot['se']:.4f}")
        print(f"      Bootstrap 95% CI      = [{boot['ci95_low']:+.4f}, "
              f"{boot['ci95_high']:+.4f}]")

    return {
        'label':                    label,
        'age_min':                  age_min,
        'age_max':                  age_max,
        'n_episodes_in_subset':     n_episodes_in_subset,
        'mean_age_across_cues':     round(mean_age, 2),
        'n_cues_after_min':         int(len(subset_joined)),
        'M2':                       res,
        'bootstrap':                boot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 14_age_stratified_quick.py (v2) | MSR-precondition test ===")
    print(f"  Language        : {args.language}")
    print(f"  Joined CSV      : {args.joined_csv}")
    print(f"  Episodes CSV    : {args.episodes_csv}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Age thresholds  : {args.age_thresholds}")
    print(f"  exclude_rate    : {args.exclude_rate}")
    print(f"  min_attempts    : {args.min_attempts}")
    print(f"  n_bootstrap     : {args.n_bootstrap}\n")

    # Load joined CSV
    print(f"Loading joined CSV (baseline)...")
    joined_baseline = pd.read_csv(args.joined_csv)
    print(f"  Joined rows: {len(joined_baseline)}")
    print(f"  Has columns: logFreq={('logFreq' in joined_baseline.columns)}, "
          f"COI={('COI' in joined_baseline.columns)}, "
          f"outcome={('outcome' in joined_baseline.columns)}, "
          f"n_child_attempts={('n_child_attempts' in joined_baseline.columns)}")

    # Load episodes
    print(f"\nLoading episodes...")
    use_cols = ['child_age_months', 'child_cues', 'r_plus_label', 'r_plus_composite']
    episodes = pd.read_csv(args.episodes_csv, usecols=use_cols)
    print(f"  Episodes: {len(episodes):,}")
    print(f"  Age range: [{episodes['child_age_months'].min():.2f}, "
          f"{episodes['child_age_months'].max():.2f}] months")

    thresholds = [float(t.strip()) for t in args.age_thresholds.split(',')]

    all_results: Dict[str, Any] = {
        'language':       args.language,
        'n_baseline_cues': int(len(joined_baseline)),
        'n_episodes':     int(len(episodes)),
        'thresholds':     {},
    }

    # Reference: ALL ages (using episodes-derived R+ measures, no age filter)
    print(f"\n{'='*70}")
    print(f"REFERENCE: ALL ages (re-aggregated, no split)")
    print(f"{'='*70}")
    ref = analyze_subset('ALL', joined_baseline, episodes, None, None, args)
    all_results['reference_all_ages'] = ref

    for T in thresholds:
        print(f"\n{'='*70}")
        print(f"AGE THRESHOLD: {T} months")
        print(f"{'='*70}")

        pre = analyze_subset(f'pre-{T}mo',  joined_baseline, episodes,
                              age_min=None, age_max=T, args=args)
        post = analyze_subset(f'post-{T}mo', joined_baseline, episodes,
                              age_min=T, age_max=None, args=args)

        all_results['thresholds'][f'T={T}'] = {
            'threshold_months': T,
            'pre':  pre,
            'post': post,
        }

    # ─── Comparison summary ───
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY — boss's MSR-precondition hypothesis")
    print(f"{'='*70}")
    print(f"  H1: post-MSR β(Rcomp × COI) > pre-MSR β(Rcomp × COI)\n")
    header = (f"  {'Subset':16s} {'n_cues':>7s} {'β(Rcomp×COI)':>14s} "
              f"{'SE':>8s} {'p':>10s}  {'Boot 95% CI':>24s}")
    print(header)
    print(f"  {'-'*16} {'-'*7} {'-'*14} {'-'*8} {'-'*10}  {'-'*24}")

    def fmt_row(label: str, res: Dict[str, Any]) -> str:
        if not res or 'error' in res and 'n_cues_after_min' not in res:
            return f"  {label:16s}   ERROR: {res.get('error', 'n/a')}"
        n_cues = res.get('n_cues_after_min', 0)
        m = res.get('M2', {})
        boot = res.get('bootstrap', {})
        if not m or 'error' in m:
            return f"  {label:16s} {n_cues:>7d}   M2 ERROR: {m.get('error', 'n/a') if m else 'no M2'}"
        beta = m.get('beta_z')
        se = m.get('se')
        p = m.get('p')
        if 'error' in boot or not boot:
            ci_str = "(boot failed)"
        else:
            ci_str = f"[{boot.get('ci95_low', 0):+.3f}, {boot.get('ci95_high', 0):+.3f}]"
        return (f"  {label:16s} {n_cues:>7d} {beta:>+14.4f} {se:>8.4f} "
                f"{p:>10.4g}  {ci_str:>24s}")

    print(fmt_row('ALL', ref))
    for T in thresholds:
        bucket = all_results['thresholds'][f'T={T}']
        print(fmt_row(f"pre-{T}mo",  bucket['pre']))
        print(fmt_row(f"post-{T}mo", bucket['post']))

    # H1 verdict
    print(f"\n  H1 verdict per threshold:")
    for T in thresholds:
        bucket = all_results['thresholds'][f'T={T}']
        pre = bucket.get('pre', {})
        post = bucket.get('post', {})
        pre_m  = pre.get('M2',  {}) if 'M2' in pre  else {}
        post_m = post.get('M2', {}) if 'M2' in post else {}
        pre_b  = pre_m.get('beta_z')
        post_b = post_m.get('beta_z')
        if pre_b is None or post_b is None:
            print(f"    T = {T:.0f} mo: incomplete (pre or post subset M2 missing)")
            continue
        delta = post_b - pre_b
        direction = "✓ post > pre" if delta > 0 else "✗ post ≤ pre"
        print(f"    T = {T:.0f} mo: pre β = {pre_b:+.4f}  "
              f"post β = {post_b:+.4f}  Δ = {delta:+.4f}  {direction}")

    # Save JSON
    out_json = out_dir / f"{args.language}_age_stratified.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results JSON: {out_json}\n")


if __name__ == "__main__":
    main()
