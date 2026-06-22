"""
17_within_effect_test.py
========================
IMT Attention Bias Paper 2 — Step 17: Spec C (within-effect test)

Episode-level multilevel logistic-style regression of next-N-reuse
on the Mundlak-decomposed R+ terms, with the critical 3-way interaction
  age_post × COI × R+_within
which tests the boss's "functional Layer-2 scaffolding" prediction.

────────────────────────────────────────────────────────────────────────────

Inputs
------
* output/v16/{lang}_episodes_with_reuse.csv (from 16_episode_outcome_uptake.py)
* output/v11*/{lang}_r_plus_joined.csv (provides per-cue COI, logFreq)
* output/json_cache/{corpus_dir}/{child}/*.json — used to derive a file→child
  mapping (required for the (1 | child) random intercept).

Model (Spec C)
--------------
Restrict to CONTINGENT episodes (r_plus_label != no_contingent_response),
so the R+_within term is the episode's deviation from the cue's mean
R+_composite over contingent episodes.

  next_N_reuse[e] ~
      logFreq_z(c)
    + COI_z(c)
    + COI_z × logFreq_z
    + R+_between_z(c) + R+_within_z(c, e)
    + COI_z × R+_between_z + COI_z × R+_within_z
    + age_post
    + age_post × COI_z
    + age_post × R+_within_z
    + age_post × COI_z × R+_within_z      ★ 3-way: functional Layer-2 test
    + (1 | cue_subtype) + (1 | child)

We fit:
  (1) statsmodels MixedLM (linear probability model with random
      intercept by cue; child intercepts pooled into the same random
      effect specification by using a nested groups variable
      cue_subtype:child for a single grouping factor since
      statsmodels MixedLM does not natively support crossed REs).
  (2) Linear OLS with cluster-robust SE on cue_subtype as a sanity
      check on inference.

Hypothesis
----------
  β(age_post × COI × R+_within) > 0 at sig.
  Interpreted as: post-MSR, momentary high-R+ exchanges on high-COI
  cues drive cue reuse — i.e., functional within-episode scaffolding
  emerges as a Layer-2 mechanism after MSR threshold.

Usage
-----
  python 17_within_effect_test.py --batch --output_dir ./output/v17 --window 5

Author: Torami x Boss | IMT Attention project | Paper 2 / Spec C v1 | 2026-06-21
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":         "Brown",
        "reuse_csv":     "./output/v16/English_episodes_with_reuse.csv",
        "joined_csv":    "./output/v11_runA/English_r_plus_joined.csv",
        "json_cache":    "./output/json_cache/English",
    },
    "English-Manchester": {
        "label":         "Manchester",
        "reuse_csv":     "./output/v16/English-Manchester_episodes_with_reuse.csv",
        "joined_csv":    "./output/v11/English-Manchester_r_plus_joined.csv",
        "json_cache":    "./output/json_cache/English-Manchester",
    },
    "English-UK": {
        "label":         "English-UK",
        "reuse_csv":     "./output/v16/English-UK_episodes_with_reuse.csv",
        "joined_csv":    "./output/v11_runA/English-UK_r_plus_joined.csv",
        "json_cache":    "./output/json_cache/English-UK",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


def build_file_child_map(json_cache_dir: Path) -> Dict[str, str]:
    """file_stem -> child (top-level subdir name under json_cache_dir)."""
    if not json_cache_dir.exists():
        return {}
    mapping: Dict[str, str] = {}
    for child_dir in json_cache_dir.iterdir():
        if not child_dir.is_dir():
            continue
        for jf in child_dir.rglob("*.json"):
            mapping[jf.stem] = child_dir.name
    return mapping


def add_mundlak_columns(df_contingent: pd.DataFrame) -> pd.DataFrame:
    """Compute R+_between(c) and R+_within(c, e) on contingent rows."""
    df = df_contingent.copy()
    cue_means = df.groupby("cue_subtype")["r_plus_composite"].transform("mean")
    df["R_between"] = cue_means
    df["R_within"]  = df["r_plus_composite"] - cue_means
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

PREDICTORS = [
    "logFreq_z", "COI_z", "COI_x_logFreq",
    "R_between_z", "R_within_z",
    "COI_x_R_between", "COI_x_R_within",
    "age_post",
    "age_x_COI", "age_x_R_within",
    "age_x_COI_x_R_within",
]


def fit_specC(df: pd.DataFrame, outcome_col: str, use_child_re: bool) -> Dict[str, Any]:
    """
    Fit Spec C on contingent (episode, cue) rows.

    Two estimators reported:
      (a) MixedLM linear probability model, random intercept by cue
          (and optionally by child via nested grouping).
      (b) OLS with cluster-robust SE on cue.
    """
    df = df.dropna(subset=[outcome_col, "logFreq_z", "COI_z",
                            "R_between_z", "R_within_z", "age_post"]).copy()
    if len(df) < 100:
        return {"error": f"fewer than 100 rows (n={len(df)})"}

    df["COI_x_logFreq"]      = df["COI_z"] * df["logFreq_z"]
    df["COI_x_R_between"]    = df["COI_z"] * df["R_between_z"]
    df["COI_x_R_within"]     = df["COI_z"] * df["R_within_z"]
    df["age_x_COI"]          = df["age_post"] * df["COI_z"]
    df["age_x_R_within"]     = df["age_post"] * df["R_within_z"]
    df["age_x_COI_x_R_within"] = df["age_post"] * df["COI_z"] * df["R_within_z"]

    # (a) MixedLM
    formula = "outcome ~ " + " + ".join(PREDICTORS)
    df = df.rename(columns={outcome_col: "outcome"})
    mlm_result: Dict[str, Any] = {}
    try:
        if use_child_re and "child" in df.columns and df["child"].notna().any():
            # Nested grouping: use child as random group, with vc_formula for cue
            mlm = smf.mixedlm(
                formula, data=df, groups=df["child"],
                vc_formula={"cue_subtype": "0 + C(cue_subtype)"},
            ).fit(method="lbfgs", reml=True)
        else:
            mlm = smf.mixedlm(formula, data=df, groups=df["cue_subtype"]).fit(method="lbfgs")
        for p in ["Intercept"] + PREDICTORS:
            if p in mlm.params:
                mlm_result[p] = {
                    "beta": float(mlm.params[p]),
                    "se":   float(mlm.bse[p]),
                    "z":    float(mlm.tvalues[p]),
                    "p":    float(mlm.pvalues[p]),
                }
        mlm_meta = {
            "n":               int(mlm.nobs),
            "converged":       bool(getattr(mlm, "converged", True)),
            "groups_variable": "child" if use_child_re else "cue_subtype",
        }
    except Exception as exc:
        mlm_result = {}
        mlm_meta = {"error": f"MixedLM failed: {exc}"}

    # (b) OLS with cluster-robust SE on cue
    X = sm.add_constant(df[PREDICTORS].astype(float), has_constant="add")
    y = df["outcome"].astype(float)
    try:
        ols_naive   = sm.OLS(y, X).fit()
        ols_cluster = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["cue_subtype"].astype(str).values},
        )
    except Exception as exc:
        return {"error": f"OLS failed: {exc}"}

    ols_result: Dict[str, Any] = {}
    for p in ["const"] + PREDICTORS:
        ols_result[p] = {
            "beta":         float(ols_cluster.params[p]),
            "se_cluster":   float(ols_cluster.bse[p]),
            "se_naive":     float(ols_naive.bse[p]),
            "p_cluster":    float(ols_cluster.pvalues[p]),
            "p_naive":      float(ols_naive.pvalues[p]),
            "ci95_low_cluster":  float(ols_cluster.conf_int().loc[p, 0]),
            "ci95_high_cluster": float(ols_cluster.conf_int().loc[p, 1]),
        }

    return {
        "n_total":             int(ols_cluster.nobs),
        "n_cues":              int(df["cue_subtype"].nunique()),
        "n_children":          int(df["child"].nunique()) if "child" in df.columns else None,
        "outcome":             outcome_col,
        "mlm":                 {"meta": mlm_meta, "params": mlm_result},
        "ols_cluster_robust":  {"params": ols_result,
                                  "r2": float(ols_cluster.rsquared)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus driver
# ─────────────────────────────────────────────────────────────────────────────

def analyze_corpus(language: str, reuse_csv: Path, joined_csv: Path,
                   json_cache: Path, window: int) -> Dict[str, Any]:
    print(f"\n=== {language} ===")
    print(f"  reuse_csv  : {reuse_csv}")
    print(f"  joined_csv : {joined_csv}")
    print(f"  json_cache : {json_cache}")
    print(f"  window     : {window}")

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    print(f"  expanded rows: {len(reuse):,}")
    contingent = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    print(f"  contingent rows: {len(contingent):,}")
    if len(contingent) < 100:
        return {"language": language, "error": "insufficient contingent rows"}

    joined = pd.read_csv(joined_csv)
    base = joined[["cue_subtype", "logFreq", "COI"]].copy()
    df = contingent.merge(base, on="cue_subtype", how="inner")
    df = df.dropna(subset=["logFreq", "COI", "r_plus_composite", "child_age_months"])

    # Mundlak decomposition on contingent rows
    df = add_mundlak_columns(df)

    # Child mapping — try string and zero-padded variants because pandas may
    # have coerced numeric filenames to int and stripped leading zeros.
    file_to_child = build_file_child_map(json_cache)
    print(f"  file→child mapping entries: {len(file_to_child):,}")
    if file_to_child:
        sample_keys = list(file_to_child.keys())
        max_key_len = max(len(k) for k in sample_keys)
    else:
        max_key_len = 0
    raw = df["file"].astype(str)
    df["child"] = raw.map(file_to_child)
    miss = df["child"].isna()
    if miss.any() and max_key_len > 0:
        padded = raw.str.zfill(max_key_len)
        df.loc[miss, "child"] = padded[miss].map(file_to_child)
    n_mapped = df["child"].notna().sum()
    print(f"  rows mapped to a child: {n_mapped:,} ({100*n_mapped/len(df):.1f}%)")

    # Age binary
    df["age_post"] = (df["child_age_months"] >= 24).astype(int)

    # z-score across the working frame
    for col in ["logFreq", "COI", "R_between", "R_within"]:
        df[f"{col}_z"] = z(df[col])

    outcome_col = f"next_{window}_reuse"
    if outcome_col not in df.columns:
        return {"language": language, "error": f"missing outcome column {outcome_col}"}

    use_child_re = df["child"].notna().mean() > 0.95
    res = fit_specC(df, outcome_col, use_child_re=use_child_re)
    return {"language": language, "window": window, "result": res}


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV (flat β table)
# ─────────────────────────────────────────────────────────────────────────────

def flatten_summary(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict] = []
    for lang, top in all_results.items():
        label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
        res = top.get("result", {})
        if "ols_cluster_robust" not in res:
            continue
        for predictor, vals in res["ols_cluster_robust"]["params"].items():
            rows.append({
                "corpus":     label,
                "language":   lang,
                "window":     top.get("window"),
                "spec":       "C_OLS_cluster",
                "predictor":  predictor,
                "beta":       vals.get("beta"),
                "se":         vals.get("se_cluster"),
                "p":          vals.get("p_cluster"),
                "p_naive":    vals.get("p_naive"),
                "ci95_low":   vals.get("ci95_low_cluster"),
                "ci95_high":  vals.get("ci95_high_cluster"),
            })
        mlm = res.get("mlm", {}).get("params", {})
        for predictor, vals in mlm.items():
            rows.append({
                "corpus":     label,
                "language":   lang,
                "window":     top.get("window"),
                "spec":       "C_MixedLM",
                "predictor":  predictor,
                "beta":       vals.get("beta"),
                "se":         vals.get("se"),
                "p":          vals.get("p"),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spec C: 3-way interaction test for functional R+ within effect.")
    p.add_argument("--reuse_csv",  default=None)
    p.add_argument("--joined_csv", default=None)
    p.add_argument("--json_cache", default=None)
    p.add_argument("--language",   default=None)
    p.add_argument("--window",     type=int, default=5)
    p.add_argument("--output_dir", default="./output/v17")
    p.add_argument("--batch", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        targets = [
            (lang, Path(cfg["reuse_csv"]), Path(cfg["joined_csv"]), Path(cfg["json_cache"]))
            for lang, cfg in CANONICAL_CORPORA.items()
        ]
    else:
        if not (args.language and args.reuse_csv and args.joined_csv and args.json_cache):
            sys.exit("ERROR: provide --language, --reuse_csv, --joined_csv, --json_cache (or --batch).")
        targets = [(args.language, Path(args.reuse_csv), Path(args.joined_csv), Path(args.json_cache))]

    all_results: Dict[str, Dict[str, Any]] = {}
    for lang, reuse_path, joined_path, jcache in targets:
        if not reuse_path.exists():
            print(f"  SKIP {lang}: reuse CSV missing ({reuse_path})")
            continue
        if not joined_path.exists():
            print(f"  SKIP {lang}: joined CSV missing ({joined_path})")
            continue
        res = analyze_corpus(lang, reuse_path, joined_path, jcache, args.window)
        all_results[lang] = res
        out_json = out_dir / f"{lang}_within_effect_test_N{args.window}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  → {out_json}")

    if all_results:
        summary = flatten_summary(all_results)
        summary_csv = out_dir / f"within_effect_summary_N{args.window}.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"\n  → {summary_csv}  ({len(summary):,} rows)")


if __name__ == "__main__":
    main()
