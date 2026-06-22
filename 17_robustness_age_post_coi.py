"""
17_robustness_age_post_coi.py
=============================
IMT Attention Bias Paper 2 — Step 17: Robustness battery for age_post × COI

Pre-registered robustness suite for the Paper 2 "Layer-2 candidate"
finding from 17_within_effect_test.py: a 2-way age_post × COI interaction
that consistently emerged across Brown/Manchester/English-UK.

Specs:
  BASELINE   reuse_nextN ~ age_post * COI  (+ random effects)
  SPEC #1    + log_cue_freq + prior_local_freq
             + age_post:log_cue_freq + COI:log_cue_freq
             (frequency confound — CRITICAL pass gate)
  SPEC #2    MixedLM with random intercept by child + cluster-SE by cue
  SPEC #3    sensitivity across N ∈ {3, 5, 10}
  SPEC #4    continuous-age / spline
  SPEC #5    formulaic exclusion

This file currently implements BASELINE + SPEC #1 + SPEC #2 (per-corpus and
pooled) so the central pass gate (SPEC #1) can be evaluated before any
further work. Remaining specs (#3-#5) live in this same script behind the
`--specs` flag and will be added in subsequent revisions.

────────────────────────────────────────────────────────────────────────────

Outputs
-------
* output/v17/results_robustness.json
    {corpus: {spec: {predictor: {beta, se, p, ...}}}}
* output/v17/results_robustness_summary.csv (long table, one row per
    (corpus, spec, model, predictor))

Pass criterion for SPEC #1:
    age_post × COI β > 0.020 and p < 0.01 in ≥ 2/3 corpora.

────────────────────────────────────────────────────────────────────────────

Usage
-----
  python 17_robustness_age_post_coi.py --batch --window 5 --prior_window 20

Author: Torami x Boss | IMT Attention project | Paper 2 / robustness v1 | 2026-06-21
"""

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":          "Brown",
        "reuse_csv":      "./output/v16/English_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English_tokens_tagged.csv",
        "joined_csv":     "./output/v11_runA/English_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English",
    },
    "English-Manchester": {
        "label":          "Manchester",
        "reuse_csv":      "./output/v16/English-Manchester_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English-Manchester_tokens_tagged.csv",
        "joined_csv":     "./output/v11/English-Manchester_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English-Manchester",
    },
    "English-UK": {
        "label":          "English-UK",
        "reuse_csv":      "./output/v16/English-UK_episodes_with_reuse.csv",
        "tagged_csv":     "./output/English-UK_tokens_tagged.csv",
        "joined_csv":     "./output/v11_runA/English-UK_r_plus_joined.csv",
        "json_cache":     "./output/json_cache/English-UK",
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


def build_child_utt_index(tagged_csv: Path) -> Dict[str, Tuple[np.ndarray, List[set]]]:
    """Same shape as 16_episode_outcome_uptake.build_child_utt_index."""
    print(f"  Building child utt index from {tagged_csv} ...")
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
    print(f"    files indexed: {len(file_index):,}")
    return file_index


def add_prior_local_freq(reuse_df: pd.DataFrame,
                          file_index: Dict[str, Tuple[np.ndarray, List[set]]],
                          prior_window: int) -> pd.DataFrame:
    """
    For each (episode, cue) row, count occurrences of the same cue in the
    `prior_window` child utterances immediately preceding the episode's
    child_utt_idx, within the same file. Cue-free utterances still count
    toward the window (so the window is "previous K child turns" not
    "previous K cue-bearing turns").
    """
    n = len(reuse_df)
    prior = np.zeros(n, dtype=np.int32)
    observed = np.zeros(n, dtype=np.int32)
    truncated = np.zeros(n, dtype=np.int8)

    files = reuse_df["file"].astype(str).values
    utt_idxs = reuse_df["child_utt_idx"].astype(int).values
    cues = reuse_df["cue_subtype"].astype(str).values

    for i in tqdm(range(n), desc=f"    prior_local_freq (W={prior_window})"):
        entry = file_index.get(files[i])
        if entry is None:
            truncated[i] = 1
            continue
        idxs, cue_sets = entry
        pos = bisect.bisect_left(idxs, utt_idxs[i])
        start = max(0, pos - prior_window)
        window_slice = cue_sets[start:pos]
        observed[i] = len(window_slice)
        if observed[i] < prior_window:
            truncated[i] = 1
        c = cues[i]
        prior[i] = sum(1 for s in window_slice if c in s)

    out = reuse_df.copy()
    out["prior_local_freq"] = prior
    out["prior_window_observed"] = observed
    out["prior_window_truncated"] = truncated
    return out


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


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

def _params_dict(fit, predictors: List[str], use_z_param: bool) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    ci = fit.conf_int()
    stat = fit.tvalues
    for p in predictors:
        if p not in fit.params:
            continue
        out[p] = {
            "beta":      float(fit.params[p]),
            "se":        float(fit.bse[p]),
            "stat":      float(stat[p]),
            "p":         float(fit.pvalues[p]),
            "ci95_low":  float(ci.loc[p, 0]),
            "ci95_high": float(ci.loc[p, 1]),
        }
    return out


def fit_ols_cluster(df: pd.DataFrame, predictors: List[str], outcome_col: str,
                     cluster_col: str) -> Dict[str, Any]:
    X = sm.add_constant(df[predictors].astype(float), has_constant="add")
    y = df[outcome_col].astype(float)
    fit = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col].astype(str).values},
    )
    return {
        "n": int(fit.nobs),
        "r2": float(fit.rsquared),
        "params": _params_dict(fit, ["const"] + predictors, use_z_param=False),
    }


def fit_mixedlm(df: pd.DataFrame, predictors: List[str], outcome_col: str,
                 group_col: str) -> Dict[str, Any]:
    formula = f"{outcome_col} ~ " + " + ".join(predictors)
    try:
        fit = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(method="lbfgs")
    except Exception as exc:
        return {"error": f"MixedLM failed: {exc}"}
    fixed = ["Intercept"] + predictors
    return {
        "n": int(fit.nobs),
        "group_var": group_col,
        "random_intercept_sd": (
            float(np.sqrt(float(fit.cov_re.iloc[0, 0])))
            if fit.cov_re.size > 0 else float("nan")
        ),
        "params": {
            p: {
                "beta": float(fit.params[p]),
                "se":   float(fit.bse[p]),
                "stat": float(fit.tvalues[p]),
                "p":    float(fit.pvalues[p]),
            } for p in fixed if p in fit.params
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus driver — BASELINE + SPEC #1 + SPEC #2
# ─────────────────────────────────────────────────────────────────────────────

def analyze_corpus(language: str, reuse_csv: Path, tagged_csv: Path,
                   joined_csv: Path, json_cache: Path,
                   window: int, prior_window: int,
                   contingent_only: bool) -> Dict[str, Any]:
    print(f"\n=== {language} ===")
    print(f"  reuse_csv     : {reuse_csv}")
    print(f"  tagged_csv    : {tagged_csv}")
    print(f"  joined_csv    : {joined_csv}")
    print(f"  json_cache    : {json_cache}")
    print(f"  outcome window: {window}   prior window: {prior_window}")
    print(f"  contingent_only: {contingent_only}")

    outcome_col = f"next_{window}_reuse"

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    print(f"  expanded rows: {len(reuse):,}")
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
        print(f"  contingent rows: {len(reuse):,}")
    if outcome_col not in reuse.columns:
        return {"error": f"missing {outcome_col}"}

    file_index = build_child_utt_index(tagged_csv)
    reuse = add_prior_local_freq(reuse, file_index, prior_window)

    joined = pd.read_csv(joined_csv)
    cols = ["cue_subtype"]
    if "COI" in joined.columns:
        cols.append("COI")
    if "log_caregiver_count" in joined.columns:
        cols.append("log_caregiver_count")
    elif "logFreq" in joined.columns:
        cols.append("logFreq")
    base = joined[cols].copy()
    base = base.rename(columns={"log_caregiver_count": "log_cue_freq",
                                  "logFreq": "log_cue_freq"})

    df = reuse.merge(base, on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    print(f"  after merge & dropna: {len(df):,}")

    # Add child id (best-effort)
    file_to_child, max_len = build_file_child_map(json_cache)
    raw = df["file"].astype(str)
    df["child"] = raw.map(file_to_child)
    miss = df["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        df.loc[miss, "child"] = padded[miss].map(file_to_child)
    n_child_mapped = df["child"].notna().sum()
    print(f"  rows mapped to a child: {n_child_mapped:,} ({100*n_child_mapped/len(df):.1f}%)")

    df["age_post"] = (df["child_age_months"] >= 24).astype(int)

    # z-score continuous predictors GLOBALLY (within this corpus)
    df["COI_z"]              = z(df["COI"])
    df["log_cue_freq_z"]     = z(df["log_cue_freq"])
    df["prior_local_freq_z"] = z(df["prior_local_freq"])

    # Build interaction columns
    df["age_x_COI"]                  = df["age_post"] * df["COI_z"]
    df["age_x_log_cue_freq"]         = df["age_post"] * df["log_cue_freq_z"]
    df["COI_x_log_cue_freq"]         = df["COI_z"]    * df["log_cue_freq_z"]

    base_predictors  = ["age_post", "COI_z", "age_x_COI"]
    spec1_predictors = base_predictors + [
        "log_cue_freq_z", "prior_local_freq_z",
        "age_x_log_cue_freq", "COI_x_log_cue_freq",
    ]

    results: Dict[str, Any] = {
        "n_rows":         int(len(df)),
        "n_cues":         int(df["cue_subtype"].nunique()),
        "n_files":        int(df["file"].nunique()),
        "n_children":     int(df["child"].nunique()) if df["child"].notna().any() else None,
        "outcome_col":    outcome_col,
        "outcome_mean":   float(df[outcome_col].mean()),
        "prior_window":   prior_window,
        "contingent_only": contingent_only,
    }

    # BASELINE — OLS cluster
    print("  fit BASELINE (OLS + cluster-robust SE on cue) ...")
    results["BASELINE_OLS_cluster"] = fit_ols_cluster(
        df, base_predictors, outcome_col, cluster_col="cue_subtype",
    )

    # BASELINE — MixedLM (random by cue) and by child
    print("  fit BASELINE (MixedLM, random by cue) ...")
    results["BASELINE_MixedLM_cue"] = fit_mixedlm(
        df, base_predictors, outcome_col, group_col="cue_subtype",
    )
    if df["child"].notna().any():
        print("  fit BASELINE (MixedLM, random by child) ...")
        df_c = df.dropna(subset=["child"]).copy()
        results["BASELINE_MixedLM_child"] = fit_mixedlm(
            df_c, base_predictors, outcome_col, group_col="child",
        )

    # SPEC #1 — OLS cluster
    print("  fit SPEC1 (OLS + cluster-robust SE on cue) ...")
    results["SPEC1_OLS_cluster"] = fit_ols_cluster(
        df, spec1_predictors, outcome_col, cluster_col="cue_subtype",
    )

    # SPEC #1 — MixedLM
    print("  fit SPEC1 (MixedLM, random by cue) ...")
    results["SPEC1_MixedLM_cue"] = fit_mixedlm(
        df, spec1_predictors, outcome_col, group_col="cue_subtype",
    )
    if df["child"].notna().any():
        print("  fit SPEC1 (MixedLM, random by child) ...")
        df_c = df.dropna(subset=["child"]).copy()
        results["SPEC1_MixedLM_child"] = fit_mixedlm(
            df_c, spec1_predictors, outcome_col, group_col="child",
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pooled analysis (SPEC #2 — pooled across corpora)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pooled(per_corpus_dfs: Dict[str, pd.DataFrame], outcome_col: str,
                    contingent_only: bool) -> Dict[str, Any]:
    if not per_corpus_dfs:
        return {"error": "no per-corpus frames"}
    frames = []
    for lang, df in per_corpus_dfs.items():
        d = df.copy()
        d["corpus"] = lang
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Re-z within pooled frame
    df["COI_z"]              = z(df["COI"])
    df["log_cue_freq_z"]     = z(df["log_cue_freq"])
    df["prior_local_freq_z"] = z(df["prior_local_freq"])
    df["age_x_COI"]                  = df["age_post"] * df["COI_z"]
    df["age_x_log_cue_freq"]         = df["age_post"] * df["log_cue_freq_z"]
    df["COI_x_log_cue_freq"]         = df["COI_z"]    * df["log_cue_freq_z"]

    # Corpus dummies
    corpus_dummies = pd.get_dummies(df["corpus"], prefix="corpus", drop_first=True).astype(float)
    df = pd.concat([df, corpus_dummies], axis=1)
    corp_cols = list(corpus_dummies.columns)

    base_predictors  = ["age_post", "COI_z", "age_x_COI"] + corp_cols
    spec1_predictors = base_predictors + [
        "log_cue_freq_z", "prior_local_freq_z",
        "age_x_log_cue_freq", "COI_x_log_cue_freq",
    ]

    out: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_corpora": int(df["corpus"].nunique()),
        "outcome_col": outcome_col,
        "contingent_only": contingent_only,
    }
    out["BASELINE_OLS_cluster"] = fit_ols_cluster(
        df, base_predictors, outcome_col, cluster_col="cue_subtype",
    )
    out["SPEC1_OLS_cluster"]    = fit_ols_cluster(
        df, spec1_predictors, outcome_col, cluster_col="cue_subtype",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Flatten summary
# ─────────────────────────────────────────────────────────────────────────────

def flatten(results: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict] = []
    for corpus_lang, by_spec in results.items():
        if corpus_lang == "_meta":
            continue
        label = CANONICAL_CORPORA.get(corpus_lang, {}).get("label", corpus_lang)
        for spec_name, spec_res in by_spec.items():
            if not isinstance(spec_res, dict) or "params" not in spec_res:
                continue
            params = spec_res["params"]
            for predictor, vals in params.items():
                rows.append({
                    "corpus":    label,
                    "language":  corpus_lang,
                    "spec":      spec_name,
                    "predictor": predictor,
                    "beta":      vals.get("beta"),
                    "se":        vals.get("se"),
                    "stat":      vals.get("stat"),
                    "p":         vals.get("p"),
                    "ci95_low":  vals.get("ci95_low"),
                    "ci95_high": vals.get("ci95_high"),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness battery for age_post × COI.")
    p.add_argument("--reuse_csv",   default=None)
    p.add_argument("--tagged_csv",  default=None)
    p.add_argument("--joined_csv",  default=None)
    p.add_argument("--json_cache",  default=None)
    p.add_argument("--language",    default=None)
    p.add_argument("--output_dir",  default="./output/v17")
    p.add_argument("--window",      type=int, default=5,
                   help="Outcome window N for next_N_reuse.")
    p.add_argument("--prior_window", type=int, default=20,
                   help="Prior-window for prior_local_freq (child utterances).")
    p.add_argument("--include_noncontingent", action="store_true",
                   help="By default we restrict to contingent rows. Set this to include all rows.")
    p.add_argument("--batch", action="store_true",
                   help="Run all canonical corpora (Brown/Manchester/English-UK).")
    p.add_argument("--pooled", action="store_true",
                   help="Also fit pooled (across-corpora) model when --batch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        targets = [
            (lang, cfg["reuse_csv"], cfg["tagged_csv"], cfg["joined_csv"], cfg["json_cache"])
            for lang, cfg in CANONICAL_CORPORA.items()
        ]
    else:
        if not (args.language and args.reuse_csv and args.tagged_csv
                and args.joined_csv and args.json_cache):
            sys.exit("ERROR: provide all of --language/--reuse_csv/--tagged_csv/"
                     "--joined_csv/--json_cache, or use --batch.")
        targets = [(args.language, args.reuse_csv, args.tagged_csv, args.joined_csv, args.json_cache)]

    contingent_only = not args.include_noncontingent

    all_results: Dict[str, Any] = {"_meta": {
        "window": args.window,
        "prior_window": args.prior_window,
        "contingent_only": contingent_only,
    }}
    per_corpus_frames: Dict[str, pd.DataFrame] = {}

    for lang, reuse_csv, tagged_csv, joined_csv, json_cache in targets:
        rc, tc, jc, jcache = map(Path, [reuse_csv, tagged_csv, joined_csv, json_cache])
        missing = [p for p in [rc, tc, jc] if not p.exists()]
        if missing:
            print(f"  SKIP {lang}: missing inputs {missing}")
            continue
        res = analyze_corpus(lang, rc, tc, jc, jcache,
                              window=args.window, prior_window=args.prior_window,
                              contingent_only=contingent_only)
        all_results[lang] = res

        # Save per-corpus
        suffix = f"N{args.window}_PW{args.prior_window}"
        out_json = out_dir / f"{lang}_robustness_{suffix}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  → {out_json}")

    # SUMMARY CSV
    summary = flatten(all_results)
    summary_csv = out_dir / f"results_robustness_summary_N{args.window}_PW{args.prior_window}.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n  → {summary_csv}  ({len(summary):,} rows)")

    out_json = out_dir / f"results_robustness_N{args.window}_PW{args.prior_window}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → {out_json}")

    # Pass-criterion report for SPEC #1
    target_pred = "age_x_COI"
    print("\n========== SPEC #1 pass-criterion report ==========")
    print(f"  Pass = β({target_pred}) > 0.020 AND p < 0.01")
    for spec_key in ["SPEC1_OLS_cluster", "SPEC1_MixedLM_cue", "SPEC1_MixedLM_child"]:
        print(f"\n  -- {spec_key} --")
        passes = 0; total = 0
        for lang, res in all_results.items():
            if lang == "_meta":
                continue
            spec = res.get(spec_key, {})
            params = spec.get("params", {})
            row = params.get(target_pred)
            if row is None:
                print(f"    {CANONICAL_CORPORA.get(lang,{}).get('label',lang):<12}: (missing)")
                continue
            beta, p = row["beta"], row["p"]
            sig = beta > 0.020 and p < 0.01
            passes += int(sig)
            total += 1
            mark = "✓ PASS" if sig else "✗"
            print(f"    {CANONICAL_CORPORA.get(lang,{}).get('label',lang):<12}: β={beta:+.4f}  SE={row['se']:.4f}  p={p:.4f}  {mark}")
        if total > 0:
            verdict = "PASS (≥2/3)" if passes >= 2 else "FAIL (<2/3)"
            print(f"    → {passes}/{total}  {verdict}")


if __name__ == "__main__":
    main()
