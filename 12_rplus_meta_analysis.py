"""
12_rplus_meta_analysis.py
=========================
IMT Attention Bias Paper — Step 12: Cross-linguistic Meta-Analysis (Paper 2)

Pools the standardized β estimates of the Paper 2 central interaction
(R+_composite × COI) — and, in parallel, the R+_composite main effect — across
multiple languages using fixed-effects and DerSimonian-Laird (DL) random-effects
models. Reports heterogeneity (Q, I², τ²) and a binomial sign test.

────────────────────────────────────────────────────────────────────────────
Inputs:
  --json_inputs A.json B.json C.json D.json
        Per-language regression result JSONs (from 11_rsm_r_plus_join.py).

  --language_labels English Spanish Japanese Mandarin
        Optional human-readable labels. Defaults to first token of each filename.

Statistical models:
  1. Fixed-effects (inverse-variance):
       w_i      = 1 / SE_i²
       β_FE     = Σ(w_i β_i) / Σ(w_i)
       SE_FE    = √(1 / Σ(w_i))

  2. Random-effects (DerSimonian-Laird):
       Q        = Σ(w_i (β_i − β_FE)²)
       C        = Σ(w_i) − Σ(w_i²) / Σ(w_i)
       τ²       = max(0, (Q − df) / C)
       w_i*     = 1 / (SE_i² + τ²)
       β_RE     = Σ(w_i* β_i) / Σ(w_i*)
       I²       = max(0, 100 (Q − df) / Q)

  3. Sign test (binomial):
       Tests whether the proportion of positive-sign estimates exceeds chance.

  4. Forest plot:
       Per-language β with 95% CI + pooled estimates.

Outputs:
  {out_dir}/rplus_meta_analysis.json    (full results)
  {out_dir}/forest_plot_data.csv        (per-language summary + pooled row)
  {out_dir}/forest_plot.png             (matplotlib forest plot, optional)

Usage:
  mkdir -p output/v12

  python 12_rplus_meta_analysis.py \\
      --json_inputs \\
          ./output/v11_runA/English_r_plus_regression.json \\
          ./output/v11_runA_norate/Spanish_r_plus_regression.json \\
          ./output/v11_runA_t1/Japanese_r_plus_regression.json \\
          ./output/v11_runA_t1/Mandarin_r_plus_regression.json \\
      --language_labels English Spanish Japanese Mandarin \\
      --target_effect Rcomp_x_COI \\
      --model M2 \\
      --output_dir ./output/v12/ \\
      --make_forest_plot

Author: Torami x Boss | IMT Attention project | Paper 2 / cross-linguistic meta | 2026-06-16
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pandas scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-linguistic meta-analysis of R+_composite × COI (Paper 2)"
    )
    p.add_argument("--json_inputs", nargs="+", required=True,
                   help="Per-language regression result JSONs from 11_rsm_r_plus_join.py")
    p.add_argument("--language_labels", nargs="*", default=None,
                   help="(Optional) Custom labels. Defaults to filename stems.")
    p.add_argument("--target_effect", default="Rcomp_x_COI",
                   choices=["Rcomp_x_COI", "RplusComposite_z",
                            "COI_x_logFreq",  # Paper 1 effect, for sanity-check meta
                            "Rcomp_x_logFreq"],
                   help="Which effect to meta-analyze. Default: Rcomp_x_COI (Paper 2 central).")
    p.add_argument("--model", default="M2",
                   choices=["M0", "M1", "M2", "M3_exploratory"],
                   help="Which regression model's effect to use. Default: M2.")
    p.add_argument("--output_dir", default="./output/v12")
    p.add_argument("--make_forest_plot", action="store_true",
                   help="Generate matplotlib forest plot (requires matplotlib).")
    p.add_argument("--also_analyze_main", action="store_true",
                   help="Additionally run a parallel meta-analysis on the R+_composite "
                        "main effect (RplusComposite_z), saved as a second JSON.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-language effect extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_effect(json_path: Path, language_label: str,
                   target_effect: str, model_key: str) -> Dict[str, Any]:
    """Pull β, SE, p, n_cues, sr, ΔR² out of one regression JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if model_key not in d["models"]:
        raise KeyError(
            f"Model '{model_key}' not found in {json_path}. "
            f"Available: {list(d['models'].keys())}"
        )
    model = d["models"][model_key]

    if target_effect not in model["params"]:
        raise KeyError(
            f"Effect '{target_effect}' not found in {model_key} params of {json_path}. "
            f"Available: {list(model['params'].keys())}"
        )
    p_row = model["params"][target_effect]

    return {
        "language":    language_label,
        "json_path":   str(json_path),
        "n_cues":      int(d.get("n_cues_analyzed", 0)),
        "beta":        float(p_row["beta_z"]),
        "se":          float(p_row["std_error"]),
        "t":           float(p_row["t"]),
        "p":           float(p_row["p"]),
        "sr":          float(p_row["semipartial_r"]),
        "ci95_low":    float(p_row["ci95_low"]),
        "ci95_high":   float(p_row["ci95_high"]),
        "delta_r2":    float(model.get("nested_comparison", {}).get("delta_r2", 0.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical methods
# ─────────────────────────────────────────────────────────────────────────────

def fixed_effects_pool(effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inverse-variance fixed-effects meta-analysis."""
    betas = np.array([e["beta"] for e in effects])
    ses   = np.array([e["se"]   for e in effects])

    w = 1.0 / (ses ** 2)
    pooled_beta = float(np.sum(w * betas) / np.sum(w))
    pooled_se   = float(np.sqrt(1.0 / np.sum(w)))
    z           = pooled_beta / pooled_se
    p           = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "model":         "fixed_effects_inverse_variance",
        "k":             len(effects),
        "pooled_beta":   round(pooled_beta, 6),
        "pooled_se":     round(pooled_se, 6),
        "z":             round(z, 4),
        "p":             p,
        "ci95_low":      round(pooled_beta - 1.96 * pooled_se, 6),
        "ci95_high":     round(pooled_beta + 1.96 * pooled_se, 6),
        "weights_normalized": [round(float(wi / np.sum(w)), 4) for wi in w],
    }


def random_effects_pool_DL(effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DerSimonian-Laird random-effects meta-analysis with heterogeneity statistics."""
    betas = np.array([e["beta"] for e in effects])
    ses   = np.array([e["se"]   for e in effects])
    k     = len(effects)

    # Fixed-effects intermediate (needed for Q and τ²)
    fe_w     = 1.0 / (ses ** 2)
    fe_beta  = np.sum(fe_w * betas) / np.sum(fe_w)

    # Q (Cochran heterogeneity statistic)
    Q  = float(np.sum(fe_w * (betas - fe_beta) ** 2))
    df = k - 1

    # C (scale factor) and τ² (DL estimator, clipped at 0)
    C = float(np.sum(fe_w) - (np.sum(fe_w ** 2) / np.sum(fe_w)))
    tau_sq = max(0.0, (Q - df) / C) if C > 0 and df > 0 else 0.0

    # Heterogeneity quantification
    I_sq = max(0.0, 100.0 * (Q - df) / Q) if Q > 0 and df > 0 else 0.0
    H_sq = (Q / df) if df > 0 else 1.0
    Q_p  = float(1.0 - stats.chi2.cdf(Q, df)) if df > 0 else 1.0

    # Random-effects weights
    re_w        = 1.0 / (ses ** 2 + tau_sq)
    pooled_beta = float(np.sum(re_w * betas) / np.sum(re_w))
    pooled_se   = float(np.sqrt(1.0 / np.sum(re_w)))
    z           = pooled_beta / pooled_se
    p           = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "model":         "random_effects_DerSimonian_Laird",
        "k":             k,
        "pooled_beta":   round(pooled_beta, 6),
        "pooled_se":     round(pooled_se, 6),
        "z":             round(z, 4),
        "p":             p,
        "ci95_low":      round(pooled_beta - 1.96 * pooled_se, 6),
        "ci95_high":     round(pooled_beta + 1.96 * pooled_se, 6),
        "tau_squared":   round(tau_sq, 6),
        "tau":           round(np.sqrt(tau_sq), 6),
        "Q":             round(Q, 4),
        "Q_df":          int(df),
        "Q_p":           Q_p,
        "I_squared_pct": round(I_sq, 2),
        "H_squared":     round(H_sq, 4),
        "weights_normalized": [round(float(wi / np.sum(re_w)), 4) for wi in re_w],
    }


def sign_test(effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Binomial sign test on direction of effects (one-tailed for positive)."""
    betas = [e["beta"] for e in effects]
    k     = len(betas)
    n_pos = sum(1 for b in betas if b > 0)
    n_neg = sum(1 for b in betas if b < 0)
    n_zero = sum(1 for b in betas if b == 0)

    # One-tailed: P(X >= n_pos | k trials, p=0.5)
    try:
        p_one = float(stats.binomtest(n_pos, k, p=0.5, alternative="greater").pvalue)
    except AttributeError:
        # Fall back for scipy < 1.7
        p_one = float(stats.binom_test(n_pos, k, p=0.5, alternative="greater"))

    return {
        "n_total":             k,
        "n_positive":          n_pos,
        "n_negative":          n_neg,
        "n_zero":              n_zero,
        "proportion_positive": round(n_pos / k, 4) if k else 0.0,
        "p_one_tail_positive": p_one,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forest plot data + matplotlib forest plot
# ─────────────────────────────────────────────────────────────────────────────

def forest_plot_data(effects: List[Dict[str, Any]],
                     fe: Dict[str, Any],
                     re_pool: Dict[str, Any]) -> pd.DataFrame:
    """Per-language rows + two pooled rows (FE and RE)."""
    rows = []
    for e in effects:
        rows.append({
            "language":   e["language"],
            "n":          e["n_cues"],
            "beta":       e["beta"],
            "se":         e["se"],
            "ci95_low":   e["ci95_low"],
            "ci95_high":  e["ci95_high"],
            "p":          e["p"],
            "row_type":   "study",
        })
    rows.append({
        "language":   "Pooled (FE)",
        "n":          sum(e["n_cues"] for e in effects),
        "beta":       fe["pooled_beta"],
        "se":         fe["pooled_se"],
        "ci95_low":   fe["ci95_low"],
        "ci95_high":  fe["ci95_high"],
        "p":          fe["p"],
        "row_type":   "pooled_fe",
    })
    rows.append({
        "language":   "Pooled (RE/DL)",
        "n":          sum(e["n_cues"] for e in effects),
        "beta":       re_pool["pooled_beta"],
        "se":         re_pool["pooled_se"],
        "ci95_low":   re_pool["ci95_low"],
        "ci95_high":  re_pool["ci95_high"],
        "p":          re_pool["p"],
        "row_type":   "pooled_re",
    })
    return pd.DataFrame(rows)


def make_forest_plot(forest_df: pd.DataFrame, out_path: Path,
                     target_effect: str, model_key: str) -> None:
    """Render a black-and-white forest plot."""
    if not HAS_MATPLOTLIB:
        print("  ⚠ matplotlib not available; skipping forest plot")
        return

    k = len(forest_df)
    fig, ax = plt.subplots(figsize=(9.0, max(3.5, 0.55 * k)))

    y_positions = list(range(k))[::-1]  # top row at top
    for y, (_, r) in zip(y_positions, forest_df.iterrows()):
        is_pooled = str(r["row_type"]).startswith("pooled")
        marker    = "D" if is_pooled else "s"
        color     = "#a40000" if r["row_type"] == "pooled_re" else (
                    "#0040a4" if r["row_type"] == "pooled_fe" else "black")
        msize     = 11 if is_pooled else 8
        lw        = 2.2 if is_pooled else 1.5
        ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=color, lw=lw)
        ax.plot([r["beta"]], [y], marker=marker, color=color,
                markersize=msize, markeredgecolor="black", markeredgewidth=0.6)

    # Reference line at 0
    ax.axvline(0.0, color="gray", linestyle=":", lw=1)

    # Y labels
    ax.set_yticks(y_positions)
    labels = [
        f"{r['language']} (n={int(r['n'])})" if r["row_type"] == "study"
        else f"{r['language']} (Σn={int(r['n'])})"
        for _, r in forest_df.iterrows()
    ]
    ax.set_yticklabels(labels)

    # Beta/CI text on the right
    xlim = ax.get_xlim()
    x_right = xlim[1]
    for y, (_, r) in zip(y_positions, forest_df.iterrows()):
        text = f"β = {r['beta']:+.3f}  [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
        ax.text(x_right + 0.02 * (xlim[1] - xlim[0]), y, text,
                va="center", ha="left", fontsize=9, family="monospace")

    # Title and axis
    effect_label = {
        "Rcomp_x_COI":      "R+_composite × COI",
        "RplusComposite_z": "R+_composite (main)",
        "COI_x_logFreq":    "COI × logFreq (Paper 1)",
        "Rcomp_x_logFreq":  "R+_composite × logFreq",
    }.get(target_effect, target_effect)
    ax.set_xlabel(f"Standardized β  ({effect_label}, from {model_key})", fontsize=11)
    ax.set_title("Cross-linguistic meta-analysis (Paper 2)", fontsize=12,
                 fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Expand right margin so β/CI text doesn't get cut off
    ax.set_xlim(xlim[0], xlim[1] + 0.45 * (xlim[1] - xlim[0]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_meta(effects: List[Dict[str, Any]], target_effect: str, model_key: str,
             out_dir: Path, make_plot: bool, suffix: str = "") -> Dict[str, Any]:
    """Run FE, RE, sign test; save JSON + forest plot. Returns combined result."""

    fe       = fixed_effects_pool(effects)
    re_pool  = random_effects_pool_DL(effects)
    signs    = sign_test(effects)

    # ── Console report ──
    print(f"\n{'=' * 70}")
    print(f"Per-language effects ({target_effect}, from {model_key})")
    print(f"{'=' * 70}")
    print(f"  {'language':18s}  {'n':>4s}  {'β':>8s}  {'SE':>6s}  "
          f"{'95% CI':>22s}  {'p':>9s}")
    for e in effects:
        sig = "***" if e["p"] < 0.001 else ("**" if e["p"] < 0.01 else (
                "*" if e["p"] < 0.05 else (
                "·" if e["p"] < 0.10 else "")))
        ci = f"[{e['ci95_low']:+.3f}, {e['ci95_high']:+.3f}]"
        print(f"  {e['language']:18s}  {e['n_cues']:>4d}  {e['beta']:+8.4f}  "
              f"{e['se']:6.4f}  {ci:>22s}  {e['p']:9.4g}{sig:>3s}")

    print(f"\n{'=' * 70}")
    print(f"Fixed-effects meta-analysis (inverse-variance)")
    print(f"{'=' * 70}")
    print(f"  k            = {fe['k']}")
    print(f"  Pooled β     = {fe['pooled_beta']:+.4f}")
    print(f"  SE           = {fe['pooled_se']:.4f}")
    print(f"  95% CI       = [{fe['ci95_low']:+.4f}, {fe['ci95_high']:+.4f}]")
    print(f"  z            = {fe['z']:+.4f}")
    print(f"  p            = {fe['p']:.4g}")
    print(f"  weights (norm): {[round(w, 3) for w in fe['weights_normalized']]}")

    print(f"\n{'=' * 70}")
    print(f"Random-effects meta-analysis (DerSimonian-Laird)")
    print(f"{'=' * 70}")
    print(f"  Pooled β     = {re_pool['pooled_beta']:+.4f}")
    print(f"  SE           = {re_pool['pooled_se']:.4f}")
    print(f"  95% CI       = [{re_pool['ci95_low']:+.4f}, {re_pool['ci95_high']:+.4f}]")
    print(f"  z            = {re_pool['z']:+.4f}")
    print(f"  p            = {re_pool['p']:.4g}")
    print(f"  τ²           = {re_pool['tau_squared']:.4f}   (τ = {re_pool['tau']:.4f})")
    print(f"  Q            = {re_pool['Q']:.4f}  "
          f"(df={re_pool['Q_df']}, p={re_pool['Q_p']:.4g})")
    print(f"  I²           = {re_pool['I_squared_pct']:.1f}%   "
          f"H² = {re_pool['H_squared']:.3f}")

    print(f"\n{'=' * 70}")
    print(f"Sign test (binomial, one-tailed for positive)")
    print(f"{'=' * 70}")
    print(f"  Positive : {signs['n_positive']}/{signs['n_total']}  "
          f"(proportion = {signs['proportion_positive']:.2f})")
    print(f"  Negative : {signs['n_negative']}/{signs['n_total']}")
    if signs["n_zero"]:
        print(f"  Zero     : {signs['n_zero']}/{signs['n_total']}")
    print(f"  p (one-tail positive): {signs['p_one_tail_positive']:.4g}")

    # Save forest plot data
    forest_df = forest_plot_data(effects, fe, re_pool)
    forest_csv = out_dir / f"forest_plot_data{suffix}.csv"
    forest_df.to_csv(forest_csv, index=False)
    print(f"\n  Forest plot data: {forest_csv}")

    # Optional forest plot
    plot_path = None
    if make_plot:
        plot_path = out_dir / f"forest_plot{suffix}.png"
        make_forest_plot(forest_df, plot_path, target_effect, model_key)
        if plot_path.exists():
            print(f"  Forest plot:      {plot_path}")

    result = {
        "target_effect": target_effect,
        "model_key":     model_key,
        "languages":     [e["language"] for e in effects],
        "per_language":  effects,
        "fixed_effects": fe,
        "random_effects": re_pool,
        "sign_test":     signs,
        "forest_plot_data": forest_df.to_dict(orient="records"),
        "forest_plot_path": str(plot_path) if plot_path else None,
    }
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build language labels
    if args.language_labels:
        labels = args.language_labels
        if len(labels) != len(args.json_inputs):
            raise ValueError(
                f"language_labels count ({len(labels)}) "
                f"does not match json_inputs count ({len(args.json_inputs)})"
            )
    else:
        labels = [Path(p).stem.split("_")[0] for p in args.json_inputs]

    print(f"\n=== 12_rplus_meta_analysis.py | Paper 2 cross-linguistic pooling ===")
    print(f"  Languages       : {labels}")
    print(f"  Target effect   : {args.target_effect}")
    print(f"  Model           : {args.model}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Forest plot     : {'on' if args.make_forest_plot else 'off'}")
    if not HAS_MATPLOTLIB and args.make_forest_plot:
        print(f"  (matplotlib not found — forest plot will be skipped)")
    print()

    # Extract per-language effects for the primary target
    primary_effects = []
    for path, label in zip(args.json_inputs, labels):
        e = extract_effect(Path(path), label, args.target_effect, args.model)
        primary_effects.append(e)

    # Run meta-analysis on primary target
    primary_result = run_meta(
        primary_effects, args.target_effect, args.model, out_dir,
        make_plot=args.make_forest_plot, suffix=""
    )

    out_json = out_dir / "rplus_meta_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(primary_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full results JSON: {out_json}")

    # Optional: parallel meta on main effect
    if args.also_analyze_main and args.target_effect != "RplusComposite_z":
        print("\n" + "#" * 72)
        print("# ALSO running parallel meta-analysis on R+_composite MAIN effect")
        print("#" * 72)

        main_effects = []
        for path, label in zip(args.json_inputs, labels):
            try:
                me = extract_effect(Path(path), label, "RplusComposite_z", args.model)
                main_effects.append(me)
            except KeyError as e:
                print(f"  ⚠ Skipping {label}: {e}")

        if len(main_effects) >= 2:
            main_result = run_meta(
                main_effects, "RplusComposite_z", args.model, out_dir,
                make_plot=args.make_forest_plot, suffix="_main"
            )
            out_json_main = out_dir / "rplus_meta_analysis_main.json"
            with open(out_json_main, "w", encoding="utf-8") as f:
                json.dump(main_result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  Main-effect meta JSON: {out_json_main}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
