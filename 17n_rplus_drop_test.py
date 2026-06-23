"""
17n_rplus_drop_test.py
======================
IMT Attention Bias Paper 2 — Step 17n: R+ drop sensitivity (independence check).

Tests whether β(COI × cumulative_cue_attempts) — the main exposure-gate
slope reported in 17b/17c — is genuinely independent of R+ scaffolding,
or whether it is absorbing a tier-1 R+ × COI synergy.

The reviewer attack we are pre-empting:
  "Maybe the COI × cumulative_attempts effect is just R+ × COI under
   a different name."

Two per-child OLS models, both fit on contingent post-MSR episodes with
cluster-robust SE on cue_subtype:

  MODEL A — champion (R+ present):
    next_N_reuse ~ COI_z + cumulative_z + COI_x_cumulative
                 + prior_local_freq_z + log_cue_freq_z
                 + r_plus_composite_z        (episode-level within R+)
                 + r_plus_between_z          (cue-level mean contingent R+)
                 + r_plus_composite_x_COI
                 + r_plus_between_x_COI

  MODEL B — R+ dropped (the current 17b/17c specification):
    next_N_reuse ~ COI_z + cumulative_z + COI_x_cumulative
                 + prior_local_freq_z + log_cue_freq_z

The exposure-gate term  COI_x_cumulative  is the comparison object.

Acceptance criteria (consult SPEC #17n in conversation):
  ✓ Direction of β(COI_x_cumulative) agrees A and B (both >0)
  ✓ |Δβ / β| < 0.30  (effect is robust to R+ inclusion)
  ✓ Sig p < 0.05 for the FULL and DROP-MANC pooled meta in both models

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17n/)
----------------------
* per_child_betas_Rplus_kept_N{5,10}.csv      Model A per-child β
* per_child_betas_Rplus_dropped_N{5,10}.csv   Model B per-child β
* meta_Rplus_dropped.json                       per-window FULL + DROP-MANC
                                                meta for both models
* comparison_table_FullModel_vs_RplusDropped.csv  long-form table
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / R+ drop v1 | 2026-06-22
"""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from tqdm import tqdm
    from scipy.stats import norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":        "Brown",
        "reuse_csv":    "./output/v16/English_episodes_with_reuse.csv",
        "tagged_csv":   "./output/English_tokens_tagged.csv",
        "joined_csv":   "./output/v11_runA/English_r_plus_joined.csv",
        "json_cache":   "./output/json_cache/English",
    },
    "English-Manchester": {
        "label":        "Manchester",
        "reuse_csv":    "./output/v16/English-Manchester_episodes_with_reuse.csv",
        "tagged_csv":   "./output/English-Manchester_tokens_tagged.csv",
        "joined_csv":   "./output/v11/English-Manchester_r_plus_joined.csv",
        "json_cache":   "./output/json_cache/English-Manchester",
    },
    "English-UK": {
        "label":        "English-UK",
        "reuse_csv":    "./output/v16/English-UK_episodes_with_reuse.csv",
        "tagged_csv":   "./output/English-UK_tokens_tagged.csv",
        "joined_csv":   "./output/v11_runA/English-UK_r_plus_joined.csv",
        "json_cache":   "./output/json_cache/English-UK",
    },
    "English-NA-Pool": {
        "label":        "NA-Pool",
        "reuse_csv":    "./output/v16/English-NA-Pool_episodes_with_reuse.csv",
        "tagged_csv":   "./output/English-NA-Pool_tokens_tagged.csv",
        "joined_csv":   "./output/v11/English-NA-Pool_r_plus_joined.csv",
        "json_cache":   "./output/json_cache/English-NA-Pool",
        "child_filter": ["April"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with 17b/17c)
# ─────────────────────────────────────────────────────────────────────────────

def z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


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


def build_child_utt_index(tagged_csv: Path) -> Dict[str, Tuple[np.ndarray, List[set]]]:
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
    return file_index


def add_prior_local_freq(reuse_df: pd.DataFrame,
                          file_index: Dict[str, Tuple[np.ndarray, List[set]]],
                          prior_window: int) -> pd.DataFrame:
    n = len(reuse_df)
    prior = np.zeros(n, dtype=np.int32)
    files = reuse_df["file"].astype(str).values
    utt_idxs = reuse_df["child_utt_idx"].astype(int).values
    cues = reuse_df["cue_subtype"].astype(str).values
    for i in tqdm(range(n), desc=f"    prior_local_freq W={prior_window}"):
        entry = file_index.get(files[i])
        if entry is None:
            continue
        idxs, cue_sets = entry
        pos = bisect.bisect_left(idxs, utt_idxs[i])
        start = max(0, pos - prior_window)
        window_slice = cue_sets[start:pos]
        c = cues[i]
        prior[i] = sum(1 for s in window_slice if c in s)
    out = reuse_df.copy()
    out["prior_local_freq"] = prior
    return out


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
    zv = beta_re / se_re if se_re > 0 else float("nan")
    p = float(2 * (1.0 - norm.cdf(abs(zv)))) if not math.isnan(zv) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": zv, "pooled_p_RE": p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-child fit with R+ kept / dropped
# ─────────────────────────────────────────────────────────────────────────────

PREDS_BASE = [
    "COI_z_local", "cum_z_local", "COI_x_cum_local",
    "prior_z_local", "logfreq_z_local",
]
PREDS_WITH_RPLUS = PREDS_BASE + [
    "rplus_episode_z_local", "rplus_between_z_local",
    "rplus_ep_x_COI_local", "rplus_bet_x_COI_local",
]


def fit_per_child_model(df: pd.DataFrame, outcome_col: str, model: str,
                          min_episodes: int) -> pd.DataFrame:
    """Fit Model A (model='with_rplus') or Model B (model='base') per child."""
    rows: List[Dict[str, Any]] = []
    preds = PREDS_WITH_RPLUS if model == "with_rplus" else PREDS_BASE
    for child_id, g in df.groupby("child"):
        if len(g) < min_episodes:
            continue
        if g["COI"].std() == 0 or g["cumulative_cue_attempts"].std() == 0:
            continue
        gg = g.copy()
        gg["COI_z_local"]       = z(gg["COI"])
        gg["cum_z_local"]       = z(gg["cumulative_cue_attempts"])
        gg["prior_z_local"]     = z(gg["prior_local_freq"])
        gg["logfreq_z_local"]   = z(gg["log_cue_freq"])
        gg["COI_x_cum_local"]   = gg["COI_z_local"] * gg["cum_z_local"]
        if model == "with_rplus":
            # Build R+_between (cue-level mean across this child's data)
            cue_mean_r = gg.groupby("cue_subtype")["r_plus_composite"].transform("mean")
            gg["rplus_between_local"] = cue_mean_r
            gg["rplus_episode_z_local"] = z(gg["r_plus_composite"])
            gg["rplus_between_z_local"] = z(cue_mean_r)
            gg["rplus_ep_x_COI_local"]  = gg["rplus_episode_z_local"] * gg["COI_z_local"]
            gg["rplus_bet_x_COI_local"] = gg["rplus_between_z_local"] * gg["COI_z_local"]
        X = sm.add_constant(gg[preds].astype(float), has_constant="add")
        y = gg[outcome_col].astype(float)
        try:
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
            row = {
                "child":            child_id,
                "model":            model,
                "n_episodes":       int(len(gg)),
                "n_cues":           int(gg["cue_subtype"].nunique()),
                "outcome_mean":     float(gg[outcome_col].mean()),
                "beta_COI_x_cum":   float(fit.params["COI_x_cum_local"]),
                "se_COI_x_cum":     float(fit.bse["COI_x_cum_local"]),
                "p_COI_x_cum":      float(fit.pvalues["COI_x_cum_local"]),
                "r2":               float(fit.rsquared),
            }
            if model == "with_rplus":
                for p in ["rplus_episode_z_local", "rplus_between_z_local",
                           "rplus_ep_x_COI_local", "rplus_bet_x_COI_local"]:
                    row[f"beta_{p}"] = float(fit.params[p])
                    row[f"p_{p}"]    = float(fit.pvalues[p])
            rows.append(row)
        except Exception as exc:
            rows.append({"child": child_id, "model": model, "error": str(exc)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus data assembly (mirrors 17c's analyze_corpus first half)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_corpus(language: str, cfg: Dict[str, str], window: int,
                     prior_window: int) -> Optional[pd.DataFrame]:
    outcome_col = f"next_{window}_reuse"
    reuse_csv = Path(cfg["reuse_csv"])
    tagged_csv = Path(cfg["tagged_csv"])
    joined_csv = Path(cfg["joined_csv"])
    json_cache = Path(cfg["json_cache"])
    missing = [p for p in [reuse_csv, tagged_csv, joined_csv] if not p.exists()]
    if missing:
        print(f"  SKIP {language}: missing {missing}")
        return None

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    if outcome_col not in reuse.columns:
        print(f"  SKIP {language}: missing {outcome_col} in reuse CSV "
              f"(re-run 16 with --windows including {window})")
        return None

    file_index = build_child_utt_index(tagged_csv)
    reuse = add_prior_local_freq(reuse, file_index, prior_window)
    file_to_child, max_len = build_file_child_map(json_cache)
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()

    child_filter = cfg.get("child_filter")
    if child_filter:
        reuse = reuse[reuse["child"].isin(child_filter)].copy()
        print(f"  Restricting {language} to children {child_filter}: {len(reuse):,} rows")

    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq",
                            "child_age_months", "r_plus_composite"]).copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    df_post = df[df["child_age_months"] >= 24].copy()
    return df_post


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="R+ drop sensitivity for COI × cumulative_attempts.")
    parser.add_argument("--windows", default="5,10")
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--min_episodes_per_child", type=int, default=200)
    parser.add_argument("--output_dir", default="./output/v17n")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(x.strip()) for x in args.windows.split(",")]

    # Sub-corpus labels copied from 17h for downstream meta filters
    subcorpus_labels = pd.read_csv("./output/v17h/uk_subcorpora_per_child_N5.csv")
    subcorpus_labels = subcorpus_labels[
        ~subcorpus_labels["subcorpus"].astype(str).str.contains("also in UK pool", na=False)
    ][["child", "subcorpus"]].drop_duplicates(subset="child")

    def _norm_sub(s: str) -> str:
        if pd.isna(s): return "?"
        s = str(s)
        if "long_longitudinal" in s: return "UK_long_long"
        if "short_observation" in s: return "UK_short_obs"
        if "NA_other" in s or "April" in s: return "NA_other"
        return s

    all_rows: List[pd.DataFrame] = []
    for window in windows:
        print(f"\n=== Window N = {window} ===")
        for language, cfg in CANONICAL_CORPORA.items():
            print(f"\n  --- {language} ---")
            df_post = assemble_corpus(language, cfg, window, args.prior_window)
            if df_post is None or len(df_post) == 0:
                continue
            for model in ["with_rplus", "base"]:
                per_child = fit_per_child_model(df_post, f"next_{window}_reuse",
                                                  model=model,
                                                  min_episodes=args.min_episodes_per_child)
                per_child["language"] = language
                per_child["window"]    = window
                all_rows.append(per_child)
                print(f"    {model:<12}: {len(per_child)} children fitted")

    long_df = pd.concat(all_rows, ignore_index=True, sort=False)
    long_df = long_df.merge(subcorpus_labels, on="child", how="left")
    long_df["subcorpus_norm"] = long_df["subcorpus"].apply(_norm_sub)
    # NA / April rows still won't have subcorpus from the 17h table; tag manually
    long_df.loc[long_df["language"] == "English-NA-Pool", "subcorpus_norm"] = "NA_other"
    # Brown / Manchester from 17h table fallback
    long_df.loc[long_df["subcorpus_norm"].isna() & (long_df["language"] == "English"), "subcorpus_norm"] = "Brown"
    long_df.loc[long_df["subcorpus_norm"].isna() & (long_df["language"] == "English-Manchester"), "subcorpus_norm"] = "Manchester"

    # Save per-model / per-window per-child tables
    for window in windows:
        for model in ["with_rplus", "base"]:
            tag = "Rplus_kept" if model == "with_rplus" else "Rplus_dropped"
            sub = long_df[(long_df["window"] == window) & (long_df["model"] == model)].copy()
            sub.to_csv(out_dir / f"per_child_betas_{tag}_N{window}.csv", index=False)

    # Drop the duplicates that come from 17h's English-UK pool sharing
    # Manchester children. We collapse via:
    #   if (child appears both in language='English-Manchester' AND 'English-UK'),
    #   keep the Manchester row only.
    has_both = long_df[long_df["language"] == "English-Manchester"]["child"].unique()
    long_df = long_df[~((long_df["language"] == "English-UK") & long_df["child"].isin(has_both))].copy()

    # Meta-analysis per (model, window) for FULL and DROP-MANCHESTER
    meta_results: Dict[str, Any] = {}
    print(f"\n=== Random-effects meta per (model, scope, window) ===")
    print(f"  {'model':<14} {'window':>6} {'scope':<10} {'n':>3} {'β':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6}")
    print("  " + "-" * 80)
    for model in ["with_rplus", "base"]:
        for window in windows:
            sub = long_df[(long_df["model"] == model) & (long_df["window"] == window)].copy()
            sub = sub.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"])
            full = random_effects_meta(sub["beta_COI_x_cum"].values.astype(float),
                                         sub["se_COI_x_cum"].values.astype(float))
            drop = sub[sub["subcorpus_norm"] != "Manchester"].copy()
            drop_meta = random_effects_meta(drop["beta_COI_x_cum"].values.astype(float),
                                              drop["se_COI_x_cum"].values.astype(float))
            meta_results.setdefault(model, {}).setdefault(str(window), {})
            meta_results[model][str(window)] = {
                "FULL":      full,
                "DROP_MANC": drop_meta,
                "n_full":    int(len(sub)),
                "n_drop":    int(len(drop)),
            }
            for label, m in [("FULL", full), ("DROP-MANC", drop_meta)]:
                if "pooled_beta_RE" not in m:
                    continue
                sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
                print(f"  {model:<14} N={window:>4} {label:<10} {m['n_studies']:>3} {m['pooled_beta_RE']:>+8.4f} "
                      f"{m['pooled_se_RE']:.4f} {m['pooled_p_RE']:.4f}{sig:<3} {m['tau2']:.4f} {m['I2_pct']:>5.1f}%")

    # Save meta JSON
    with open(out_dir / "meta_Rplus_dropped.json", "w", encoding="utf-8") as f:
        json.dump(meta_results, f, indent=2, default=str)
    print(f"\n  → {out_dir / 'meta_Rplus_dropped.json'}")

    # Comparison table
    comp_rows: List[Dict[str, Any]] = []
    print(f"\n=== Model A (R+ kept) vs Model B (R+ dropped) ===")
    print(f"  {'Scope':<10} {'N':>3} {'β_A':>9} {'β_B':>9} {'Δβ':>8} {'%change':>10} {'p_A':>8} {'p_B':>8}")
    print("  " + "-" * 75)
    for window in windows:
        for scope, key in [("FULL", "FULL"), ("DROP-MANC", "DROP_MANC")]:
            a = meta_results["with_rplus"][str(window)][key]
            b = meta_results["base"][str(window)][key]
            if "pooled_beta_RE" not in a or "pooled_beta_RE" not in b:
                continue
            delta = b["pooled_beta_RE"] - a["pooled_beta_RE"]
            pct = (delta / a["pooled_beta_RE"]) * 100 if a["pooled_beta_RE"] != 0 else float("nan")
            comp_rows.append({
                "scope":       scope, "window": window,
                "beta_A":      a["pooled_beta_RE"], "se_A": a["pooled_se_RE"], "p_A": a["pooled_p_RE"], "I2_A": a["I2_pct"],
                "beta_B":      b["pooled_beta_RE"], "se_B": b["pooled_se_RE"], "p_B": b["pooled_p_RE"], "I2_B": b["I2_pct"],
                "delta_beta":  delta, "pct_change":  pct,
            })
            print(f"  {scope:<10} {window:>3} {a['pooled_beta_RE']:>+8.4f} {b['pooled_beta_RE']:>+8.4f} "
                  f"{delta:>+8.4f} {pct:>+9.1f}% {a['pooled_p_RE']:>8.4f} {b['pooled_p_RE']:>8.4f}")
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "comparison_table_FullModel_vs_RplusDropped.csv", index=False)
    print(f"\n  → {out_dir / 'comparison_table_FullModel_vs_RplusDropped.csv'}")

    # Acceptance criteria
    direction_ok = all((r["beta_A"] > 0 and r["beta_B"] > 0) for r in comp_rows)
    magnitude_ok = all(abs(r["pct_change"] / 100.0) < 0.30 for r in comp_rows
                        if not math.isnan(r["pct_change"]))
    sig_ok = all((r["p_A"] < 0.05 and r["p_B"] < 0.05) for r in comp_rows)

    # SUMMARY.md
    lines: List[str] = []
    lines.append("# 17n R+ drop sensitivity (Model A vs Model B)\n")
    lines.append("## Comparison table (β = β(COI × cumulative_cue_attempts))\n")
    lines.append("| Scope | N | β_A (R+ kept) | SE_A | p_A | β_B (R+ dropped) | SE_B | p_B | Δβ | % change |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in comp_rows:
        sig_a = "***" if r['p_A']<0.001 else ("**" if r['p_A']<0.01 else ("*" if r['p_A']<0.05 else ("†" if r['p_A']<0.10 else "")))
        sig_b = "***" if r['p_B']<0.001 else ("**" if r['p_B']<0.01 else ("*" if r['p_B']<0.05 else ("†" if r['p_B']<0.10 else "")))
        lines.append(
            f"| {r['scope']} | N={r['window']} | {r['beta_A']:+.4f}{sig_a} | "
            f"{r['se_A']:.4f} | {r['p_A']:.4f} | "
            f"{r['beta_B']:+.4f}{sig_b} | {r['se_B']:.4f} | {r['p_B']:.4f} | "
            f"{r['delta_beta']:+.4f} | {r['pct_change']:+.1f}% |"
        )

    lines.append("\n## Acceptance criteria\n")
    lines.append(f"- ✓ Direction agrees (both >0) across all cells: **{direction_ok}**")
    lines.append(f"- ✓ |Δβ / β_A| < 0.30 across all cells: **{magnitude_ok}**")
    lines.append(f"- ✓ Sig p<0.05 across all cells (both models): **{sig_ok}**")

    if direction_ok and magnitude_ok and sig_ok:
        verdict = "PASS — Exposure-gate β is independent of R+ scaffolding (Tier-1 absorption ruled out)."
    elif direction_ok and sig_ok:
        verdict = "PARTIAL — direction and sig hold but magnitude shift > 30% in at least one cell. Refine interpretation."
    else:
        verdict = "FAIL — exposure-gate β is not independent of R+. Theoretical interpretation needs revision."
    lines.append(f"\n**Verdict**: {verdict}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → {out_dir / 'SUMMARY.md'}")
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
