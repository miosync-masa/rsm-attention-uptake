"""
06_meta_analysis.py
===================
IMT Attention Bias Paper — Step 6: Cross-Linguistic Meta-Analysis

Aggregates the per-language regression results from 05 into a single
language-level meta-analysis of the AI x Frequency interaction effect.

THE QUESTION:
    Is the RSM interaction (attention x reactive-repetition) positive
    and consistent ACROSS typologically diverse languages, beyond what
    any single language could establish?

METHOD:
    Each language contributes one effect size: the standardized interaction
    beta for peak production (and, separately, for emergence). We combine
    them with:
      - A simple vote count + sign test (robust, assumption-light).
      - A random-effects meta-analysis (DerSimonian-Laird) on the betas,
        using each language's standard error.
      - Heterogeneity statistics (Q, I^2).

    Random effects (not fixed) because languages are a sample of the world's
    languages and true effects may genuinely vary across them.

INPUT:
    Reads {lang}_uptake_summary.json (produced by 05) for each language.

OUTPUT:
    meta_analysis.csv          : per-language effect sizes
    meta_analysis_summary.json : pooled estimate, CI, heterogeneity, sign test

--------------------------------------------------------------------------
CAVEATS baked in (these double as Limitations-section text). Per Tomoe/Shio-ne:
  (1) These are PRODUCTION data, not comprehension. Children likely COMPREHEND
      high-AI cues even earlier; production lags comprehension. Our uptake
      measures therefore UNDERESTIMATE the timing of acquisition.
  (2) Caregiver AI and child uptake are measured WITHIN THE SAME CORPUS.
      This yields ecological validity but not independent validation. A
      held-out-corpus replication is needed.
  (3) FREQUENCY is a PROXY for reactive repetition (R+), not R+ itself.
      A genuine RSM test requires measuring caregiver responses to child
      output directly (expansion / recast / acknowledgment vs. ignore).
  (4) FIRST EMERGENCE is sensitive to sampling density and recording timing.
      PEAK PRODUCTION and UPTAKE SLOPE are more stable indices and are
      weighted accordingly in interpretation.
  (5) Korean / Mandarin / Russian have small n (cues) and few age bins;
      their interaction estimates are promising but possibly over-estimated.
      The meta-analysis down-weights them via their larger standard errors.
--------------------------------------------------------------------------

Usage:
    python 06_meta_analysis.py --output_dir ./output/v3/ \\
        --languages English Japanese Korean Mandarin Russian Spanish Indonesian

Author: Torami x Boss x Tomoe x Shio-ne | IMT Attention | 2026-06-14
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
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Extract effect sizes from 05 summaries
# ─────────────────────────────────────────────────────────────────────────────

def beta_se_from_ci(beta: float, ci_low: float, ci_high: float) -> float:
    """Recover standard error from a 95% CI: SE = (hi - lo) / (2 * 1.96)."""
    if ci_low is None or ci_high is None:
        return float("nan")
    return (ci_high - ci_low) / (2 * 1.959964)


def load_language_effects(out_dir: Path, languages: list, outcome: str) -> pd.DataFrame:
    """
    Pull the interaction beta, its SE, ΔR2, and n for each language and outcome.
    """
    rows = []
    for lang in languages:
        path = out_dir / f"{lang}_uptake_summary.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        reg = data.get("regressions", {}).get(outcome)
        if not reg or "error" in reg:
            print(f"  WARN: no regression for {lang}/{outcome}")
            continue

        bi = reg.get("beta_interaction")
        if not bi:
            continue

        beta = bi["beta"]
        se = beta_se_from_ci(beta, bi.get("ci_low"), bi.get("ci_high"))

        rows.append({
            "language": lang,
            "n_cues": reg.get("n"),
            "interaction_beta": beta,
            "interaction_se": se,
            "interaction_p": bi.get("p"),
            "deltaR2_interaction": reg.get("deltaR2_interaction"),
            "deltaR2_AI": reg.get("deltaR2_AI"),
            "semipartial_AI": reg.get("semipartial_AI_r"),
            "semipartial_logFreq": reg.get("semipartial_logFreq_r"),
            "M0_freq_only_R2": reg.get("M0_freq_only_R2"),
            "M2_full_R2": reg.get("M2_full_R2"),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sign test (assumption-light robustness check)
# ─────────────────────────────────────────────────────────────────────────────

def sign_test(betas: pd.Series, expected_positive: bool = True) -> dict:
    """
    Binomial sign test: are the interaction betas consistently positive?
    Robust to outliers and effect-size scale; uses only direction.
    """
    valid = betas.dropna()
    n = len(valid)
    if n == 0:
        return {"error": "no valid betas"}
    n_positive = int((valid > 0).sum())
    n_negative = int((valid < 0).sum())
    # Two-sided binomial test against p=0.5
    k = n_positive if expected_positive else n_negative
    p_binom = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    return {
        "n_languages": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "direction_expected": "positive" if expected_positive else "negative",
        "binomial_p_onesided": round(float(p_binom), 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random-effects meta-analysis (DerSimonian-Laird)
# ─────────────────────────────────────────────────────────────────────────────

def random_effects_meta(betas: pd.Series, ses: pd.Series) -> dict:
    """
    DerSimonian-Laird random-effects pooling of effect sizes.

    Returns pooled estimate, its SE and 95% CI, the heterogeneity tau^2,
    Cochran's Q, and I^2.
    """
    mask = betas.notna() & ses.notna() & (ses > 0)
    b = betas[mask].values.astype(float)
    s = ses[mask].values.astype(float)
    k = len(b)
    if k < 2:
        return {"error": f"need >=2 languages with valid SE, got {k}"}

    # Fixed-effect weights
    w = 1.0 / (s ** 2)
    beta_fixed = float(np.sum(w * b) / np.sum(w))

    # Cochran's Q
    Q = float(np.sum(w * (b - beta_fixed) ** 2))
    df = k - 1

    # Between-study variance tau^2 (DL estimator)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects weights
    w_star = 1.0 / (s ** 2 + tau2)
    beta_random = float(np.sum(w_star * b) / np.sum(w_star))
    se_random = float(np.sqrt(1.0 / np.sum(w_star)))
    ci_low = beta_random - 1.959964 * se_random
    ci_high = beta_random + 1.959964 * se_random
    z = beta_random / se_random if se_random > 0 else float("nan")
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else float("nan")

    # I^2
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    Q_p = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else float("nan")

    return {
        "k_languages": k,
        "pooled_beta_random": round(beta_random, 4),
        "pooled_se": round(se_random, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "z": round(z, 4),
        "p_value": round(p, 6),
        "tau2": round(tau2, 4),
        "Q": round(Q, 4),
        "Q_df": df,
        "Q_p": round(Q_p, 5),
        "I2_percent": round(I2, 1),
        "pooled_beta_fixed": round(beta_fixed, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-linguistic meta-analysis.")
    parser.add_argument("--output_dir", default="./output/v3")
    parser.add_argument("--languages", nargs="+", default=[
        "English", "Japanese", "Korean", "Mandarin",
        "Russian", "Spanish", "Indonesian"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()

    print(f"\n{'='*64}")
    print("CROSS-LINGUISTIC META-ANALYSIS — RSM interaction effect")
    print(f"{'='*64}")

    all_summaries = {}

    for outcome in ["peak_rate_per_1k", "first_emergence_month"]:
        print(f"\n{'#'*64}")
        print(f"OUTCOME: {outcome}")
        print(f"{'#'*64}")

        eff = load_language_effects(out_dir, args.languages, outcome)
        if len(eff) == 0:
            print("  No effects loaded.")
            continue

        # Save per-language table
        eff_path = out_dir / f"meta_effects_{outcome}.csv"
        eff.to_csv(eff_path, index=False)
        print(f"\n  Per-language effects → {eff_path}")
        print(eff[["language", "n_cues", "interaction_beta", "interaction_se",
                   "interaction_p", "deltaR2_interaction"]].to_string(index=False))

        # Sign test (peak expects +, emergence interaction expects + per data)
        expected_pos = True
        sign = sign_test(eff["interaction_beta"], expected_positive=expected_pos)
        print(f"\n  Sign test: {sign['n_positive']}/{sign['n_languages']} positive"
              f"  (binomial p={sign['binomial_p_onesided']:.4f})")

        # Random-effects meta
        meta = random_effects_meta(eff["interaction_beta"], eff["interaction_se"])
        if "error" in meta:
            print(f"  Meta: {meta['error']}")
        else:
            print(f"\n  Random-effects pooled interaction β = "
                  f"{meta['pooled_beta_random']:+.3f}  "
                  f"95% CI [{meta['ci_low']:+.3f}, {meta['ci_high']:+.3f}]")
            print(f"    z={meta['z']:.3f}, p={meta['p_value']:.5f}")
            print(f"    Heterogeneity: I²={meta['I2_percent']:.1f}%  "
                  f"Q={meta['Q']:.2f} (df={meta['Q_df']}, p={meta['Q_p']:.4f})  "
                  f"τ²={meta['tau2']:.3f}")

        # Also pool ΔR2(interaction) descriptively
        dr2 = eff["deltaR2_interaction"].dropna()
        if len(dr2):
            print(f"\n  ΔR²(interaction): mean={dr2.mean():.3f}  "
                  f"median={dr2.median():.3f}  "
                  f"range=[{dr2.min():.3f}, {dr2.max():.3f}]")
            dr2_AI = eff["deltaR2_AI"].dropna()
            print(f"  ΔR²(AI alone):    mean={dr2_AI.mean():.3f}  "
                  f"median={dr2_AI.median():.3f}  "
                  f"range=[{dr2_AI.min():.3f}, {dr2_AI.max():.3f}]")
            print(f"  → interaction adds {dr2.mean()/max(dr2_AI.mean(),1e-9):.1f}x "
                  "more variance than AI-as-main-effect")

        all_summaries[outcome] = {
            "per_language": eff.to_dict(orient="records"),
            "sign_test": sign,
            "random_effects_meta": meta,
            "deltaR2_interaction_mean": float(dr2.mean()) if len(dr2) else None,
            "deltaR2_AI_mean": float(eff["deltaR2_AI"].dropna().mean())
                               if eff["deltaR2_AI"].notna().any() else None,
        }

    # Master summary
    caveats = [
        "Production data, not comprehension; uptake timing is an upper bound.",
        "Caregiver AI and child uptake from the same corpus; needs held-out replication.",
        "Frequency proxies reactive repetition (R+); direct response-coding pending.",
        "First-emergence is sampling-sensitive; peak/slope are more stable.",
        "Korean/Mandarin/Russian have small n; down-weighted via SE in meta.",
    ]
    master = {
        "languages": args.languages,
        "outcomes_analyzed": list(all_summaries.keys()),
        "results": all_summaries,
        "caveats": caveats,
    }
    master_path = out_dir / "meta_analysis_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*64}")
    print(f"Wrote master summary → {master_path}")
    print(f"{'='*64}")
    print("\nCaveats (Limitations-section seed):")
    for i, c in enumerate(caveats, 1):
        print(f"  ({i}) {c}")
    print()


if __name__ == "__main__":
    main()
