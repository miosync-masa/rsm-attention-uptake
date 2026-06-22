"""
15_within_between_decomposition.py
==================================
IMT Attention Bias Paper 2 — Step 15: Mundlak-style within/between decomposition of R+

Decomposes caregiver R+ into:
  R+_between(c)   = mean r_plus_composite over contingent episodes for cue c
                  → "this cue tends to elicit deep responses from caregivers"
                    (compensatory / cue-level structural bias)
  R+_within(c, e) = r_plus_composite[e] - R+_between(c)
                  → "this specific exchange was deeper than the cue's baseline"
                    (functional / momentary scaffolding)

Two regression specifications:

  Spec A (cue-level OLS, all cues with n_contingent ≥ min_attempts):
    outcome_z ~ logFreq_z + COI_z + COI_z×logFreq_z
              + R+_between_z + R+_within_proxy_z
              + COI_z × R+_between_z
              + COI_z × R+_within_proxy_z

    where R+_within_proxy(c) = mean(|R+_within(c, e)|) over contingent episodes for c
    (cue-level proxy for the magnitude of within-cue scaffolding variability).

  Spec B (episode-level OLS with cluster-robust SE on cue_subtype):
    outcome_z[e] ~ logFreq_z(c) + COI_z(c) + COI_z×logFreq_z
                 + R+_between_z(c) + R+_within_z(c, e)
                 + COI_z(c) × R+_between_z(c)
                 + COI_z(c) × R+_within_z(c, e)
    cov_type = 'cluster', groups = cue_subtype

    Same β as a naive OLS, but SE accounts for arbitrary within-cue
    error correlation. This is the correct inferential setup for the
    Mundlak between term whose predictor varies only at cue level.

    Caveat — outcome[e] is broadcast from cue level (peak_rate_per_1k),
    so it does not vary within cue. The COI × R+_within slope cannot be
    identified by this outcome. To test R+_within properly, use
    Spec C with an episode-varying outcome (see 16_episode_outcome_uptake.py).

Each corpus is analyzed at three age cuts:
  * all ages (full episode set)
  * pre-MSR  (< 24 mo)
  * post-MSR (≥ 24 mo)

Per the boss's hypothesis (Paper 2 central novel finding):
  * COI × R+_between : pre-MSR NEGATIVE  (compensatory caregiver bias),
                       post-MSR weakens
  * COI × R+_within  : pre-MSR ≈ 0,
                       post-MSR POSITIVE (functional Layer-2 scaffolding)

Usage:
  python 15_within_between_decomposition.py \\
      --episodes_csv ./output/v10b/English_r_plus_episodes.csv \\
      --joined_csv   ./output/v11_runA/English_r_plus_joined.csv \\
      --language     English \\
      --output_dir   ./output/v15

  # Multi-corpus convenience: pass --batch to run Brown/Manchester/English-UK
  # using the canonical paths.

Author: Torami x Boss | IMT Attention project | Paper 2 / Mundlak v1 | 2026-06-21
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
    import statsmodels.formula.api as smf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pandas statsmodels")
    sys.exit(1)

warnings.filterwarnings("ignore", category=RuntimeWarning)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":        "Brown",
        "episodes_csv": "./output/v10b/English_r_plus_episodes.csv",
        "joined_csv":   "./output/v11_runA/English_r_plus_joined.csv",
    },
    "English-Manchester": {
        "label":        "Manchester",
        "episodes_csv": "./output/v10b/English-Manchester_r_plus_episodes.csv",
        "joined_csv":   "./output/v11/English-Manchester_r_plus_joined.csv",
    },
    "English-UK": {
        "label":        "English-UK",
        "episodes_csv": "./output/v10b/English-UK_r_plus_episodes.csv",
        "joined_csv":   "./output/v11_runA/English-UK_r_plus_joined.csv",
    },
}

AGE_SUBSETS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("all",     None, None),
    ("pre-24",  None, 24.0),
    ("post-24", 24.0, None),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


def filter_contingent(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["r_plus_label"] != "no_contingent_response"].copy()


def apply_age(df: pd.DataFrame, age_min: Optional[float], age_max: Optional[float]) -> pd.DataFrame:
    out = df
    if age_min is not None:
        out = out[out["child_age_months"] >= age_min]
    if age_max is not None:
        out = out[out["child_age_months"] < age_max]
    return out


def explode_cues(df: pd.DataFrame) -> pd.DataFrame:
    """Explode child_cues pipe list to one row per (episode, cue)."""
    df = df.copy()
    df["cue_list"] = df["child_cues"].fillna("").astype(str).str.split("|")
    df = df.explode("cue_list")
    df["cue_subtype"] = df["cue_list"].str.strip()
    df = df[df["cue_subtype"] != ""]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Within/between decomposition
# ─────────────────────────────────────────────────────────────────────────────

def decompose(eps_contingent_exploded: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R+_between(c) and R+_within(c, e), and a cue-level |within| proxy.

    Returns episode-level DataFrame with columns:
      cue_subtype, r_plus_composite, R_between, R_within
    """
    cue_means = (
        eps_contingent_exploded.groupby("cue_subtype")["r_plus_composite"]
        .mean()
        .rename("R_between")
    )
    out = eps_contingent_exploded.merge(
        cue_means, left_on="cue_subtype", right_index=True, how="left"
    )
    out["R_within"] = out["r_plus_composite"] - out["R_between"]
    return out


def aggregate_cue_level(decomposed: pd.DataFrame, min_contingent: int = 5) -> pd.DataFrame:
    """
    Cue-level aggregation:
      n_contingent_subset
      R_between       = cue mean of r_plus_composite (same as decompose)
      R_within_proxy  = mean(|R_within|) within cue
      R_within_sd     = sd(r_plus_composite) within cue (informational)
    """
    agg = decomposed.groupby("cue_subtype").agg(
        n_contingent_subset=("r_plus_composite", "size"),
        R_between=("R_between", "first"),
        R_within_proxy=("R_within", lambda s: s.abs().mean()),
        R_within_sd=("r_plus_composite", "std"),
    ).reset_index()
    agg = agg[agg["n_contingent_subset"] >= min_contingent].copy()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Spec A — cue-level OLS
# ─────────────────────────────────────────────────────────────────────────────

def fit_spec_a(merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Inputs: merged cue-level frame with columns
      cue_subtype, logFreq, COI, outcome, R_between, R_within_proxy
    Returns dict with model results.
    """
    required = ["logFreq", "COI", "outcome", "R_between", "R_within_proxy"]
    df = merged.dropna(subset=required).copy()
    if len(df) < 10:
        return {"error": f"fewer than 10 cues (n={len(df)})"}

    for col in required:
        df[f"{col}_z"] = z(df[col])
    df["COI_x_logFreq"]    = df["COI_z"] * df["logFreq_z"]
    df["COI_x_R_between"]  = df["COI_z"] * df["R_between_z"]
    df["COI_x_R_within"]   = df["COI_z"] * df["R_within_proxy_z"]

    preds = [
        "logFreq_z", "COI_z", "COI_x_logFreq",
        "R_between_z", "R_within_proxy_z",
        "COI_x_R_between", "COI_x_R_within",
    ]
    X = sm.add_constant(df[preds].astype(float), has_constant="add")
    y = df["outcome_z"].astype(float)
    fit = sm.OLS(y, X).fit()

    params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + preds:
        params[p] = {
            "beta": float(fit.params[p]),
            "se":   float(fit.bse[p]),
            "t":    float(fit.tvalues[p]),
            "p":    float(fit.pvalues[p]),
            "ci95_low":  float(fit.conf_int().loc[p, 0]),
            "ci95_high": float(fit.conf_int().loc[p, 1]),
        }

    return {
        "n_cues":  int(fit.nobs),
        "r2":      float(fit.rsquared),
        "r2_adj":  float(fit.rsquared_adj),
        "params":  params,
        "predictor_correlations": (
            df[["R_between_z", "R_within_proxy_z", "COI_z", "logFreq_z"]]
            .corr().round(3).to_dict()
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spec B — episode-level multilevel
# ─────────────────────────────────────────────────────────────────────────────

def fit_spec_b(decomposed: pd.DataFrame, cue_baseline: pd.DataFrame,
               min_contingent: int = 5) -> Dict[str, Any]:
    """
    Episode-level OLS with cluster-robust SE on cue_subtype.

    Coefficients are identical to a naive OLS; only SE/CI/p adjust for
    arbitrary within-cluster error correlation. This is the correct
    inferential setup when the between-term predictor varies only at
    cue level while the regression sample is at episode level.

    decomposed:    one row per (episode, cue) — contingent only — with
                   columns cue_subtype, r_plus_composite, R_between, R_within
    cue_baseline:  cue-level frame with logFreq, COI, outcome
    """
    counts = decomposed.groupby("cue_subtype").size()
    eligible = counts[counts >= min_contingent].index
    df = decomposed[decomposed["cue_subtype"].isin(eligible)].copy()
    if df.empty:
        return {"error": "no eligible episodes"}

    base = cue_baseline[["cue_subtype", "logFreq", "COI", "outcome"]].copy()
    df = df.merge(base, on="cue_subtype", how="inner")
    df = df.dropna(subset=["logFreq", "COI", "outcome", "R_between", "R_within"]).copy()
    if len(df) < 50:
        return {"error": f"fewer than 50 contingent episodes after merge (n={len(df)})"}

    for col in ["logFreq", "COI", "outcome", "R_between", "R_within"]:
        df[f"{col}_z"] = z(df[col])
    df["COI_x_logFreq"]   = df["COI_z"] * df["logFreq_z"]
    df["COI_x_R_between"] = df["COI_z"] * df["R_between_z"]
    df["COI_x_R_within"]  = df["COI_z"] * df["R_within_z"]

    preds = [
        "logFreq_z", "COI_z", "COI_x_logFreq",
        "R_between_z", "R_within_z",
        "COI_x_R_between", "COI_x_R_within",
    ]
    X = sm.add_constant(df[preds].astype(float), has_constant="add")
    y = df["outcome_z"].astype(float)

    try:
        fit_naive   = sm.OLS(y, X).fit()
        fit_cluster = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["cue_subtype"].astype(str).values},
        )
    except Exception as exc:
        return {"error": f"OLS failed: {exc}"}

    params: Dict[str, Dict[str, float]] = {}
    for p in ["const"] + preds:
        params[p] = {
            "beta":         float(fit_cluster.params[p]),
            "se_cluster":   float(fit_cluster.bse[p]),
            "se_naive":     float(fit_naive.bse[p]),
            "t_cluster":    float(fit_cluster.tvalues[p]),
            "p_cluster":    float(fit_cluster.pvalues[p]),
            "p_naive":      float(fit_naive.pvalues[p]),
            "ci95_low_cluster":  float(fit_cluster.conf_int().loc[p, 0]),
            "ci95_high_cluster": float(fit_cluster.conf_int().loc[p, 1]),
        }

    return {
        "n_episodes":      int(fit_cluster.nobs),
        "n_cues_in_model": int(df["cue_subtype"].nunique()),
        "r2":              float(fit_cluster.rsquared),
        "r2_adj":          float(fit_cluster.rsquared_adj),
        "se_method":       "cluster-robust on cue_subtype",
        "params":          params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus driver
# ─────────────────────────────────────────────────────────────────────────────

def analyze_corpus(language: str, episodes_csv: Path, joined_csv: Path,
                   min_contingent: int) -> Dict[str, Any]:
    print(f"\n=== {language} ===")
    print(f"  episodes_csv : {episodes_csv}")
    print(f"  joined_csv   : {joined_csv}")

    use_cols = ["file", "child_age_months", "child_cues", "r_plus_label", "r_plus_composite"]
    eps = pd.read_csv(episodes_csv, usecols=use_cols, low_memory=False)
    print(f"  episodes total          : {len(eps):,}")
    cont = filter_contingent(eps)
    print(f"  contingent episodes     : {len(cont):,}  ({100*len(cont)/max(len(eps),1):.2f}%)")
    if len(cont) == 0:
        return {"language": language, "error": "no contingent episodes"}

    joined = pd.read_csv(joined_csv)
    baseline = joined[["cue_subtype", "n_child_attempts", "logFreq", "COI", "outcome"]].copy()

    out: Dict[str, Any] = {"language": language, "subsets": {}}
    for name, lo, hi in AGE_SUBSETS:
        sub_eps = apply_age(cont, lo, hi)
        exploded = explode_cues(sub_eps)
        if exploded.empty:
            out["subsets"][name] = {"error": "no episodes after filters", "n_episodes": 0}
            continue

        decomposed = decompose(exploded)
        cue_agg    = aggregate_cue_level(decomposed, min_contingent=min_contingent)
        merged     = baseline.merge(cue_agg, on="cue_subtype", how="inner")
        print(
            f"  [{name:<7}] n_eps_in_subset={len(sub_eps):>7,}  "
            f"n_cues_with_n≥{min_contingent}={len(cue_agg):>3}  "
            f"n_cues_for_regression={len(merged):>3}"
        )

        spec_a = fit_spec_a(merged)
        spec_b = fit_spec_b(decomposed, baseline, min_contingent=min_contingent)
        out["subsets"][name] = {
            "n_episodes_in_subset":   int(len(sub_eps)),
            "n_cues_after_filter":    int(len(cue_agg)),
            "n_cues_in_regression":   int(len(merged)),
            "spec_a_cue_level":       spec_a,
            "spec_b_episode_level":   spec_b,
        }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def flatten_summary(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lang, res in all_results.items():
        label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
        for subset_name, sub in res.get("subsets", {}).items():
            base_row = {
                "corpus":               label,
                "language":             lang,
                "age_subset":           subset_name,
                "n_episodes_in_subset": sub.get("n_episodes_in_subset"),
                "n_cues_in_regression": sub.get("n_cues_in_regression"),
            }
            spec_a = sub.get("spec_a_cue_level", {})
            spec_b = sub.get("spec_b_episode_level", {})
            for spec_name, spec in [("A_cue", spec_a), ("B_episode", spec_b)]:
                if "params" not in spec:
                    continue
                for predictor, vals in spec["params"].items():
                    row = dict(base_row)
                    if spec_name == "B_episode":
                        row.update({
                            "spec":      spec_name,
                            "predictor": predictor,
                            "beta":      vals.get("beta"),
                            "se":        vals.get("se_cluster"),
                            "stat":      vals.get("t_cluster"),
                            "p":         vals.get("p_cluster"),
                            "p_naive":   vals.get("p_naive"),
                            "se_naive":  vals.get("se_naive"),
                        })
                    else:
                        row.update({
                            "spec":      spec_name,
                            "predictor": predictor,
                            "beta":      vals.get("beta"),
                            "se":        vals.get("se"),
                            "stat":      vals.get("t", vals.get("z")),
                            "p":         vals.get("p"),
                        })
                    rows.append(row)
    return pd.DataFrame(rows)


def make_forest_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping forest plot")
        return

    targets = ["COI_x_R_between", "COI_x_R_within"]
    spec = "A_cue"
    subset_order = ["pre-24", "post-24", "all"]
    plot_df = summary_df[
        (summary_df["spec"] == spec) & (summary_df["predictor"].isin(targets))
    ].copy()
    if plot_df.empty:
        print("  No Spec A rows for forest plot; skipping")
        return

    corpora = sorted(plot_df["corpus"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(11, max(3, 0.7 * len(corpora) * len(subset_order))),
                              sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        rows = []
        for corp in corpora:
            for sub in subset_order:
                r = plot_df[
                    (plot_df["corpus"] == corp)
                    & (plot_df["age_subset"] == sub)
                    & (plot_df["predictor"] == target)
                ]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({
                    "label": f"{corp} | {sub}",
                    "beta":  r["beta"],
                    "se":    r["se"],
                    "p":     r["p"],
                })
        ys = list(range(len(rows)))
        betas = [r["beta"] for r in rows]
        ses   = [r["se"]   for r in rows]
        labels = [r["label"] for r in rows]
        ax.errorbar(betas, ys, xerr=[1.96*s for s in ses], fmt="o", capsize=3, color="black")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"β({target})", fontsize=10)
        ax.set_xlabel("β (z-standardized)", fontsize=9)
        for y, r in zip(ys, rows):
            ax.text(r["beta"], y + 0.15, f"p={r['p']:.3f}", fontsize=7, color="dimgray")

    fig.suptitle("Mundlak-style decomposition (Spec A, cue-level)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Within/between R+ decomposition (Mundlak-style).")
    p.add_argument("--episodes_csv", default=None,
                   help="Path to R+ episodes CSV (single-corpus mode).")
    p.add_argument("--joined_csv", default=None,
                   help="Path to joined CSV (provides logFreq/COI/outcome).")
    p.add_argument("--language", default=None,
                   help="Language label (single-corpus mode).")
    p.add_argument("--output_dir", default="./output/v15")
    p.add_argument("--min_contingent", type=int, default=5,
                   help="Minimum contingent episodes per cue for inclusion.")
    p.add_argument("--batch", action="store_true",
                   help="Run all canonical corpora (Brown/Manchester/English-UK).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        targets = [
            (lang, Path(cfg["episodes_csv"]), Path(cfg["joined_csv"]))
            for lang, cfg in CANONICAL_CORPORA.items()
        ]
    else:
        if not (args.language and args.episodes_csv and args.joined_csv):
            sys.exit("ERROR: provide --language, --episodes_csv, --joined_csv (or --batch).")
        targets = [(args.language, Path(args.episodes_csv), Path(args.joined_csv))]

    all_results: Dict[str, Dict[str, Any]] = {}
    for lang, ep_path, jn_path in targets:
        if not ep_path.exists():
            print(f"  SKIP {lang}: episodes CSV missing ({ep_path})")
            continue
        if not jn_path.exists():
            print(f"  SKIP {lang}: joined CSV missing ({jn_path})")
            continue
        res = analyze_corpus(lang, ep_path, jn_path, args.min_contingent)
        all_results[lang] = res
        out_json = out_dir / f"{lang}_within_between_decomposition.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  → {out_json}")

    if all_results:
        summary = flatten_summary(all_results)
        summary_csv = out_dir / "within_between_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"\n  → {summary_csv}  ({len(summary):,} rows)")
        forest_png = out_dir / "within_between_forest_plot.png"
        make_forest_plot(summary, forest_png)


if __name__ == "__main__":
    main()
