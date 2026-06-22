"""
17o_window_null_mechanism.py
============================
IMT Attention Bias Paper 2 — Step 17o: Window-null mechanism diagnostic.

#17m showed pooled β(COI × cumulative) is sig at N=5 and N=10 but null
at the boundary windows N=3 and N=20. A reviewer can read this as
window cherry-picking: "the effect only works at the windows you chose."

#17o pre-empts the attack by demonstrating that the boundary nulls are
mechanically expected:

  (1) N=3 floor   — short outcome window has low information per
                    episode; the binary "any reuse" outcome has
                    insufficient variance and high truncation rate
                    (file-end terminations).
  (2) N=20 ceiling — long outcome window saturates the binary reuse
                    rate, compressing outcome variance and mechanically
                    attenuating linear-probability-model (LPM) slopes.

Five diagnostic legs:

  A. Logit GLM at all six windows.
     If LPM N=20 null is a ceiling artifact, logit-link β
     should recover sig (logit is robust to base-rate compression).

  B. Continuous fraction outcome (n_reuses_in_window / N) at all six
     windows. Re-fit per-child OLS. Continuous outcome lets the model
     express variance that the binary "any reuse" floors at 1.

  C. Variance-compression panel.
     Var(binary_outcome) and Var(fraction_outcome) per window, per
     sub-corpus. Expected: monotonic decline of binary Var as N
     grows (ceiling); fraction-outcome Var stays elevated.

  D. Truncation rate panel.
     Per window, share of episodes where fewer than N child utterances
     remained in the file. Expected: monotonic decline as N shortens.
     N=3 truncation ≤ a few %; N=20 truncation may explain marginal
     base-rate.

  E. Pseudo-windows N=7 and N=15.
     Adds two gap windows to the {3,5,10,20} grid so the pooled-β
     trajectory has six points. A smooth monotonic-then-ceiling curve
     refutes "the effect only works at specific windows."

────────────────────────────────────────────────────────────────────────────

Inputs
------
* output/v16/{lang}_episodes_with_reuse.csv (with windows 3,5,7,10,15,20)
* output/{lang}_tokens_tagged.csv          (to compute n_reuses_in_window)
* output/v11*/{lang}_r_plus_joined.csv     (COI, log_cue_freq)
* output/json_cache/{lang}/...             (child mapping)

Outputs (output/v17o/)
----------------------
* per_child_window_outcomes_long.csv       per (child, window, mode) row
* meta_six_windows.json                    LPM / Logit / Continuous meta
* variance_compression_table.csv           Var(outcome) per (window × subcorpus)
* truncation_rate_table.csv                truncation % per (window × subcorpus)
* trajectory_three_modes.png               pooled β vs window for 3 modes
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / window mechanism v1 | 2026-06-22
"""

import argparse
import bisect
import json
import math
import sys
import warnings
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

warnings.filterwarnings("ignore", category=RuntimeWarning)


CANONICAL_CORPORA: Dict[str, Dict[str, Any]] = {
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

MANCHESTER_CHILDREN = {
    "Anne", "Aran", "Becky", "Carl", "Dominic", "Gail",
    "Joel", "John", "Liz", "Ruth", "Warren",
}
UK_LONG_LONG_CHILDREN = {"Thomas", "Fraser", "Helen", "Eleanor", "Nicole"}


def assign_subcorpus(child: str, language: str) -> str:
    if language == "English":
        return "Brown"
    if language == "English-NA-Pool":
        return "NA_other"
    if child in MANCHESTER_CHILDREN:
        return "Manchester"
    if child in UK_LONG_LONG_CHILDREN:
        return "UK_long_long"
    if language == "English-UK":
        return "UK_short_obs"
    return "?"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
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


def add_next_N_counts(reuse_df: pd.DataFrame,
                       file_index: Dict[str, Tuple[np.ndarray, List[set]]],
                       windows: List[int]) -> pd.DataFrame:
    """For each (episode, cue), count the number of next-N child
    utterances that contain the target cue (continuous outcome basis)."""
    max_w = max(windows)
    n = len(reuse_df)
    counts_per_w: Dict[int, np.ndarray] = {w: np.zeros(n, dtype=np.int32) for w in windows}
    observed_per_w: Dict[int, np.ndarray] = {w: np.zeros(n, dtype=np.int32) for w in windows}

    files = reuse_df["file"].astype(str).values
    utt_idxs = reuse_df["child_utt_idx"].astype(int).values
    cues = reuse_df["cue_subtype"].astype(str).values
    for i in tqdm(range(n), desc=f"    next-N counts (max={max_w})"):
        entry = file_index.get(files[i])
        if entry is None:
            continue
        idxs, cue_sets = entry
        pos = bisect.bisect_right(idxs, utt_idxs[i])
        if pos >= len(idxs):
            continue
        end_max = min(pos + max_w, len(idxs))
        window_slice = cue_sets[pos:end_max]
        c = cues[i]
        for w in windows:
            obs = min(w, len(window_slice))
            observed_per_w[w][i] = obs
            counts_per_w[w][i] = sum(1 for s in window_slice[:obs] if c in s)
    out = reuse_df.copy()
    for w in windows:
        out[f"next_{w}_count"]    = counts_per_w[w]
        out[f"next_{w}_observed"] = observed_per_w[w]
        out[f"next_{w}_fraction"] = np.where(
            observed_per_w[w] > 0, counts_per_w[w] / observed_per_w[w], 0.0
        )
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
# Per-child OLS / Logit fit
# ─────────────────────────────────────────────────────────────────────────────

PREDS_BASE = ["COI_z_local", "cum_z_local", "COI_x_cum_local",
              "prior_z_local", "logfreq_z_local"]


def fit_one_child(gg: pd.DataFrame, outcome_col: str,
                    mode: str) -> Optional[Dict[str, Any]]:
    """mode ∈ {'LPM', 'Logit', 'Continuous'}."""
    if gg["COI"].std() == 0 or gg["cumulative_cue_attempts"].std() == 0:
        return None
    gg = gg.copy()
    gg["COI_z_local"]       = z(gg["COI"])
    gg["cum_z_local"]       = z(gg["cumulative_cue_attempts"])
    gg["prior_z_local"]     = z(gg["prior_local_freq"])
    gg["logfreq_z_local"]   = z(gg["log_cue_freq"])
    gg["COI_x_cum_local"]   = gg["COI_z_local"] * gg["cum_z_local"]
    X = sm.add_constant(gg[PREDS_BASE].astype(float), has_constant="add")
    y = gg[outcome_col].astype(float)
    try:
        if mode == "LPM":
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
        elif mode == "Logit":
            # Need binary outcome — skip if there's no variation
            if y.std() == 0 or set(np.unique(y.values)) - {0.0, 1.0}:
                return None
            fit = sm.Logit(y, X).fit(disp=False, maxiter=100)
        elif mode == "Continuous":
            # Continuous outcome ∈ [0, 1] — OLS with cluster SE
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
        else:
            return None
    except Exception:
        return None
    if "COI_x_cum_local" not in fit.params:
        return None
    return {
        "mode":            mode,
        "n_episodes":      int(len(gg)),
        "n_cues":          int(gg["cue_subtype"].nunique()),
        "outcome_mean":    float(gg[outcome_col].mean()),
        "outcome_var":     float(gg[outcome_col].var()),
        "beta":            float(fit.params["COI_x_cum_local"]),
        "se":              float(fit.bse["COI_x_cum_local"]),
        "p":               float(fit.pvalues["COI_x_cum_local"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus assembly + per-(child,window,mode) fits
# ─────────────────────────────────────────────────────────────────────────────

def process_corpus(language: str, cfg: Dict[str, Any],
                    windows: List[int], prior_window: int,
                    min_episodes: int) -> List[Dict[str, Any]]:
    print(f"\n  --- {language} ---")
    reuse_csv = Path(cfg["reuse_csv"])
    tagged_csv = Path(cfg["tagged_csv"])
    joined_csv = Path(cfg["joined_csv"])
    json_cache = Path(cfg["json_cache"])
    if not all(p.exists() for p in [reuse_csv, tagged_csv, joined_csv]):
        print(f"    SKIP: missing inputs")
        return []

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    file_index = build_child_utt_index(tagged_csv)
    reuse = add_prior_local_freq(reuse, file_index, prior_window)
    # Compute next-N counts for fraction outcome
    reuse = add_next_N_counts(reuse, file_index, windows)

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

    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    needed_outcomes = []
    for w in windows:
        needed_outcomes += [f"next_{w}_reuse", f"next_{w}_count",
                             f"next_{w}_observed", f"next_{w}_fraction"]
    df = df.dropna(subset=["COI", "log_cue_freq", "child_age_months"] + [c for c in needed_outcomes if c in df.columns]).copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    df_post = df[df["child_age_months"] >= 24].copy()
    print(f"    post-MSR rows: {len(df_post):,}    children: {df_post['child'].nunique()}")

    rows: List[Dict[str, Any]] = []
    for child_id, gg in df_post.groupby("child"):
        if len(gg) < min_episodes:
            continue
        subcorpus = assign_subcorpus(child_id, language)
        for w in windows:
            binary_col = f"next_{w}_reuse"
            fraction_col = f"next_{w}_fraction"
            observed_col = f"next_{w}_observed"
            truncation = float((gg[observed_col] < w).mean()) if observed_col in gg.columns else float("nan")
            base_row = {
                "language":     language,
                "child":        child_id,
                "subcorpus":    subcorpus,
                "window":       w,
                "n_episodes":   int(len(gg)),
                "truncation":   truncation,
            }
            for mode, col in [("LPM", binary_col),
                              ("Logit", binary_col),
                              ("Continuous", fraction_col)]:
                if col not in gg.columns:
                    continue
                res = fit_one_child(gg, col, mode)
                if res is None:
                    continue
                row = dict(base_row)
                row.update(res)
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Window-null mechanism diagnostic.")
    parser.add_argument("--windows", default="3,5,7,10,15,20")
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--min_episodes_per_child", type=int, default=200)
    parser.add_argument("--output_dir", default="./output/v17o")
    args = parser.parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",")]
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"17o — windows {windows}; prior window {args.prior_window}")
    print("=" * 70)
    all_rows: List[Dict[str, Any]] = []
    for language, cfg in CANONICAL_CORPORA.items():
        rows = process_corpus(language, cfg, windows,
                                args.prior_window, args.min_episodes_per_child)
        all_rows.extend(rows)

    long_df = pd.DataFrame(all_rows)
    # Deduplicate Manchester children appearing in English-UK pool
    has_manc = set(long_df[long_df["language"] == "English-Manchester"]["child"].unique())
    long_df = long_df[~((long_df["language"] == "English-UK") & long_df["child"].isin(has_manc))].copy()
    long_csv = out_dir / "per_child_window_outcomes_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"\n  → {long_csv}  ({len(long_df):,} rows)")

    # ───── Meta per (window, mode, FULL/DROP-MANC) ─────
    meta: Dict[str, Dict[str, Any]] = {}
    print("\n=== Random-effects meta per (window, mode, scope) ===")
    print(f"  {'W':>3} {'mode':<11} {'scope':<10} {'n':>3} {'β':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6}")
    print("  " + "-" * 75)
    for w in windows:
        meta[str(w)] = {}
        for mode in ["LPM", "Logit", "Continuous"]:
            sub = long_df[(long_df["window"] == w) & (long_df["mode"] == mode)].copy()
            sub = sub.dropna(subset=["beta", "se"])
            # FULL
            full = random_effects_meta(sub["beta"].values.astype(float),
                                         sub["se"].values.astype(float))
            # DROP-MANC
            drop = sub[sub["subcorpus"] != "Manchester"]
            drop_meta = random_effects_meta(drop["beta"].values.astype(float),
                                              drop["se"].values.astype(float))
            meta[str(w)][mode] = {"FULL": full, "DROP_MANC": drop_meta,
                                    "n_full": int(len(sub)), "n_drop": int(len(drop))}
            for label, m in [("FULL", full), ("DROP-MANC", drop_meta)]:
                if "pooled_beta_RE" not in m:
                    continue
                sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
                print(f"  {w:>3} {mode:<11} {label:<10} {m['n_studies']:>3} {m['pooled_beta_RE']:>+8.4f} "
                      f"{m['pooled_se_RE']:.4f} {m['pooled_p_RE']:.4f}{sig:<3} {m['tau2']:.4f} {m['I2_pct']:>5.1f}%")

    with open(out_dir / "meta_six_windows.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  → {out_dir / 'meta_six_windows.json'}")

    # ───── Variance compression table ─────
    print("\n=== Variance compression (binary vs continuous outcome) ===")
    var_rows: List[Dict[str, Any]] = []
    for w in windows:
        bin_sub  = long_df[(long_df["window"] == w) & (long_df["mode"] == "LPM")]
        cont_sub = long_df[(long_df["window"] == w) & (long_df["mode"] == "Continuous")]
        for sc in sorted(long_df["subcorpus"].unique()):
            bv = bin_sub[bin_sub["subcorpus"] == sc]["outcome_var"].mean() if len(bin_sub) else float("nan")
            cv = cont_sub[cont_sub["subcorpus"] == sc]["outcome_var"].mean() if len(cont_sub) else float("nan")
            bm = bin_sub[bin_sub["subcorpus"] == sc]["outcome_mean"].mean() if len(bin_sub) else float("nan")
            cm = cont_sub[cont_sub["subcorpus"] == sc]["outcome_mean"].mean() if len(cont_sub) else float("nan")
            var_rows.append({"window": w, "subcorpus": sc,
                              "binary_mean": bm, "binary_var": bv,
                              "continuous_mean": cm, "continuous_var": cv})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(out_dir / "variance_compression_table.csv", index=False)
    pivot = var_df.pivot(index="window", columns="subcorpus", values="binary_var")
    print("\n  Binary outcome Var per (window × subcorpus):")
    print(pivot.round(4).to_string())
    pivot_c = var_df.pivot(index="window", columns="subcorpus", values="continuous_var")
    print("\n  Continuous outcome Var per (window × subcorpus):")
    print(pivot_c.round(4).to_string())
    print(f"\n  → {out_dir / 'variance_compression_table.csv'}")

    # ───── Truncation rate table ─────
    trunc_rows: List[Dict[str, Any]] = []
    for w in windows:
        sub = long_df[(long_df["window"] == w) & (long_df["mode"] == "LPM")]
        for sc in sorted(sub["subcorpus"].unique()):
            mean_trunc = sub[sub["subcorpus"] == sc]["truncation"].mean()
            trunc_rows.append({"window": w, "subcorpus": sc,
                                "truncation_rate": float(mean_trunc)})
    trunc_df = pd.DataFrame(trunc_rows)
    trunc_df.to_csv(out_dir / "truncation_rate_table.csv", index=False)
    pivot_t = trunc_df.pivot(index="window", columns="subcorpus", values="truncation_rate")
    print("\n  Truncation rate per (window × subcorpus):")
    print(pivot_t.round(4).to_string())
    print(f"\n  → {out_dir / 'truncation_rate_table.csv'}")

    # ───── Trajectory plot (3 modes overlaid) ─────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"LPM": "#1f77b4", "Logit": "#d62728", "Continuous": "#2ca02c"}
        for mode in ["LPM", "Logit", "Continuous"]:
            xs, bs, los, his = [], [], [], []
            for w in windows:
                m = meta[str(w)][mode]["FULL"]
                if "pooled_beta_RE" not in m:
                    continue
                xs.append(w)
                bs.append(m["pooled_beta_RE"])
                los.append(m["pooled_beta_RE"] - 1.96 * m["pooled_se_RE"])
                his.append(m["pooled_beta_RE"] + 1.96 * m["pooled_se_RE"])
            xs, bs, los, his = map(np.array, [xs, bs, los, his])
            ax.fill_between(xs, los, his, color=colors[mode], alpha=0.12)
            ax.plot(xs, bs, "o-", color=colors[mode], lw=2, label=f"{mode} (FULL n=32)",
                     markersize=7)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xticks(windows)
        ax.set_xlabel("Outcome window N", fontsize=11)
        ax.set_ylabel("Pooled β(COI × cumulative_cue_attempts) with 95% CI", fontsize=11)
        ax.set_title("17o — Pooled β trajectory across 3 outcome modes\n"
                      "(LPM = linear probability; Logit = logit GLM; Continuous = fraction outcome)",
                      fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        traj_png = out_dir / "trajectory_three_modes.png"
        plt.savefig(traj_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → {traj_png}")
    except Exception as exc:
        print(f"  matplotlib failed: {exc}")

    # ───── SUMMARY.md ─────
    lines: List[str] = []
    lines.append(f"# 17o Window-null mechanism diagnostic — windows {windows}\n")
    lines.append("## Hypothesis under test\n")
    lines.append("- N=3 null is a *short-window low-info floor* (high truncation + low outcome variance)\n")
    lines.append("- N=20 null is an *LPM ceiling artifact* (binary outcome saturates; logit and continuous outcomes recover sig)\n")
    lines.append("\n## Pooled meta per window per mode\n")
    lines.append("| W | Mode | Scope | n | β | SE | p | τ² | I² |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for w in windows:
        for mode in ["LPM", "Logit", "Continuous"]:
            for label, key in [("FULL", "FULL"), ("DROP-MANC", "DROP_MANC")]:
                m = meta[str(w)][mode][key]
                if "pooled_beta_RE" not in m:
                    continue
                sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
                lines.append(
                    f"| {w} | {mode} | {label} | {m['n_studies']} | "
                    f"{m['pooled_beta_RE']:+.4f}{sig} | {m['pooled_se_RE']:.4f} | "
                    f"{m['pooled_p_RE']:.4f} | {m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
                )
    lines.append("\n## Variance compression (binary outcome Var per window × subcorpus)\n")
    bin_pivot = var_df.pivot(index="window", columns="subcorpus", values="binary_var")
    lines.append("| window | " + " | ".join(bin_pivot.columns) + " |")
    lines.append("|---" * (len(bin_pivot.columns) + 1) + "|")
    for w in bin_pivot.index:
        vals = [f"{bin_pivot.loc[w, c]:.4f}" if pd.notna(bin_pivot.loc[w, c]) else "—" for c in bin_pivot.columns]
        lines.append(f"| N={w} | " + " | ".join(vals) + " |")
    lines.append("\n## Variance compression (continuous outcome Var per window × subcorpus)\n")
    cont_pivot = var_df.pivot(index="window", columns="subcorpus", values="continuous_var")
    lines.append("| window | " + " | ".join(cont_pivot.columns) + " |")
    lines.append("|---" * (len(cont_pivot.columns) + 1) + "|")
    for w in cont_pivot.index:
        vals = [f"{cont_pivot.loc[w, c]:.4f}" if pd.notna(cont_pivot.loc[w, c]) else "—" for c in cont_pivot.columns]
        lines.append(f"| N={w} | " + " | ".join(vals) + " |")
    lines.append("\n## Truncation rate per window × subcorpus\n")
    lines.append("| window | " + " | ".join(pivot_t.columns) + " |")
    lines.append("|---" * (len(pivot_t.columns) + 1) + "|")
    for w in pivot_t.index:
        vals = [f"{pivot_t.loc[w, c]*100:.1f}%" if pd.notna(pivot_t.loc[w, c]) else "—" for c in pivot_t.columns]
        lines.append(f"| N={w} | " + " | ".join(vals) + " |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
