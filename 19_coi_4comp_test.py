"""
19_coi_4comp_test.py
====================
IMT Attention Bias Paper 2 — Step 19: 4-component COI sensitivity test.

The Paper 2 exposure-gate analyses (17b / 17c / 17m / 17n / 17o) use the
Paper 1 5-component COI (caregiver_AI = mean of S_acoustic, S_positional,
S_frequency_normalized, S_repetition, S_perceptual; equal weights 0.2 each).

Per docs/attention_index_formalization_v1.md §3.3, S_frequency_normalized
is a sigmoid of z-scored log-frequency by construction, so the COI is
mechanically correlated with log_cue_freq (r ≈ 0.49 – 0.91 across languages,
r ≈ 0.74 pooled within-language z-scored). The per-child regression
already includes log_cue_freq_z as a separate control, so β(COI × cumulative)
is interpretable as the partial effect after the linear frequency
component is removed. The reviewer-shield this script provides is the
sensitivity test:

  "Does β(COI × cumulative) change materially when COI itself is rebuilt
   without S_frequency_normalized (4-component, equal weights 0.25 each)?"

If β is essentially unchanged, the exposure-gate effect is **not** driven
by the S_frequency_normalized contribution to COI. If β drops or flips
sign, S_frequency_normalized is doing the work.

────────────────────────────────────────────────────────────────────────────

Procedure
---------

  For each corpus (Brown / Manchester / English-UK / NA-Pool-April):

    1.  Load the joined CSV; replace COI with COI_4comp computed from
        (S_acoustic + S_positional + S_repetition + S_perceptual) / 4.
        (S_frequency_normalized is dropped; the remaining four components
         are also equally weighted, so the comparison is purely the
         identity of the frequency component.)

    2.  Run the same per-child OLS as 17c:

          next_N_reuse ~ COI_4comp_z + cumulative_z + COI_4comp_x_cumulative
                       + prior_local_freq_z + log_cue_freq_z

        with cluster-robust SE on cue_subtype. Outcome window N ∈ {5, 10}.

    3.  Dedup Manchester (same rule as 17h / 17m).

    4.  Append the April (NA other longitudinal) per-child fit.

  Then compare to the 5-component baseline read from
  output/v17c/per_child_betas_N{5,10}.csv.

  Random-effects meta-analysis (FULL n=32 and DROP-MANC n=21) at N=5 and
  N=10, both COI variants, plus Egger small-study tests.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v19/)
---------------------
* per_child_betas_4comp_N{5,10}.csv
* comparison_5comp_vs_4comp.csv      side-by-side β/SE/p table
* meta_4comp.json                    meta + Egger per window per scope
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / 4-comp COI v1 | 2026-06-23
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
UK_LONG_LONG = {"Thomas", "Fraser", "Helen", "Eleanor", "Nicole"}

FOUR_COMP_COLS = ["S_acoustic", "S_positional", "S_repetition", "S_perceptual"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (kept verbatim with 17b/17c for direct comparability)
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


def egger_test(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    if len(betas) < 3:
        return {"error": "n < 3"}
    snd  = betas / ses
    prec = 1.0 / ses
    X = sm.add_constant(prec)
    fit = sm.OLS(snd, X).fit()
    return {
        "intercept":   float(fit.params[0]),
        "intercept_se":float(fit.bse[0]),
        "intercept_p": float(fit.pvalues[0]),
        "n":           int(fit.nobs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-child fit with COI_4comp
# ─────────────────────────────────────────────────────────────────────────────

PREDS_BASE = ["COI4_z_local", "cum_z_local", "COI4_x_cum_local",
              "prior_z_local", "logfreq_z_local"]


def fit_per_child(df: pd.DataFrame, outcome_col: str,
                   min_episodes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for child_id, g in df.groupby("child"):
        if len(g) < min_episodes:
            continue
        if g["COI_4comp"].std() == 0 or g["cumulative_cue_attempts"].std() == 0:
            continue
        gg = g.copy()
        gg["COI4_z_local"]      = z(gg["COI_4comp"])
        gg["cum_z_local"]       = z(gg["cumulative_cue_attempts"])
        gg["prior_z_local"]     = z(gg["prior_local_freq"])
        gg["logfreq_z_local"]   = z(gg["log_cue_freq"])
        gg["COI4_x_cum_local"]  = gg["COI4_z_local"] * gg["cum_z_local"]
        X = sm.add_constant(gg[PREDS_BASE].astype(float), has_constant="add")
        y = gg[outcome_col].astype(float)
        try:
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
        except Exception as exc:
            rows.append({"child": child_id, "error": str(exc)})
            continue
        rows.append({
            "child":           child_id,
            "n_episodes":      int(len(gg)),
            "n_cues":          int(gg["cue_subtype"].nunique()),
            "outcome_mean":    float(gg[outcome_col].mean()),
            "beta_COI4_x_cum": float(fit.params["COI4_x_cum_local"]),
            "se_COI4_x_cum":   float(fit.bse["COI4_x_cum_local"]),
            "p_COI4_x_cum":    float(fit.pvalues["COI4_x_cum_local"]),
            "r2":              float(fit.rsquared),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus driver
# ─────────────────────────────────────────────────────────────────────────────

def process_corpus(language: str, cfg: Dict[str, Any],
                    windows: List[int], prior_window: int,
                    min_episodes: int) -> List[pd.DataFrame]:
    print(f"\n  --- {language} ---")
    reuse_csv = Path(cfg["reuse_csv"])
    tagged_csv = Path(cfg["tagged_csv"])
    joined_csv = Path(cfg["joined_csv"])
    json_cache = Path(cfg["json_cache"])
    if not all(p.exists() for p in [reuse_csv, tagged_csv, joined_csv]):
        print("    SKIP: missing inputs")
        return []

    # Joined CSV exposes the 4 component salience columns already (carried
    # through 11_rsm_r_plus_join.py from the 03 attention_index CSV).
    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    missing_comps = [c for c in FOUR_COMP_COLS if c not in joined.columns]
    if missing_comps:
        print(f"    SKIP {language}: joined CSV missing {missing_comps}")
        return []
    joined["COI_4comp"] = joined[FOUR_COMP_COLS].mean(axis=1)

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
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

    df = reuse.merge(joined[["cue_subtype", "COI_4comp", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    needed = []
    for w in windows:
        col = f"next_{w}_reuse"
        if col in df.columns:
            needed.append(col)
    df = df.dropna(subset=["COI_4comp", "log_cue_freq", "child_age_months"] + needed).copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    df_post = df[df["child_age_months"] >= 24].copy()
    print(f"    post-MSR rows: {len(df_post):,}    children: {df_post['child'].nunique()}")

    out_per_w: List[pd.DataFrame] = []
    for w in windows:
        outcome_col = f"next_{w}_reuse"
        if outcome_col not in df_post.columns:
            continue
        per_child = fit_per_child(df_post, outcome_col, min_episodes)
        per_child["language"] = language
        per_child["window"]   = w
        out_per_w.append(per_child)
    return out_per_w


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="4-component COI sensitivity test.")
    parser.add_argument("--windows", default="5,10")
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--min_episodes_per_child", type=int, default=200)
    parser.add_argument("--output_dir", default="./output/v19")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(x.strip()) for x in args.windows.split(",")]

    print("=" * 70)
    print(f"19 — windows {windows}; prior window {args.prior_window}")
    print("    COI_4comp = mean(S_acoustic, S_positional, S_repetition, S_perceptual)")
    print("    (drops S_frequency_normalized from the 5-component baseline)")
    print("=" * 70)

    all_rows: List[pd.DataFrame] = []
    for language, cfg in CANONICAL_CORPORA.items():
        rows = process_corpus(language, cfg, windows,
                                args.prior_window, args.min_episodes_per_child)
        all_rows.extend(rows)
    long_df = pd.concat(all_rows, ignore_index=True, sort=False)

    # Sub-corpus assignment + Manchester dedup
    def assign_sub(row):
        lang = row["language"]; ch = row["child"]
        if lang == "English":           return "Brown"
        if lang == "English-NA-Pool":   return "NA_other"
        if ch in MANCHESTER_CHILDREN:   return "Manchester"
        if ch in UK_LONG_LONG:          return "UK_long_long"
        if lang == "English-UK":        return "UK_short_obs"
        return "?"

    long_df["subcorpus"] = long_df.apply(assign_sub, axis=1)
    # Drop "English-UK" duplicates of Manchester children
    has_manc = set(long_df[long_df["language"] == "English-Manchester"]["child"].unique())
    long_df = long_df[~((long_df["language"] == "English-UK") & long_df["child"].isin(has_manc))].copy()
    long_df = long_df[long_df["subcorpus"] != "?"].copy()

    # Save per-window 4-comp tables
    for w in windows:
        sub = long_df[long_df["window"] == w].copy()
        sub.to_csv(out_dir / f"per_child_betas_4comp_N{w}.csv", index=False)

    # ───── Meta (FULL + DROP-MANC) at each window ─────
    print("\n=== 4-comp meta vs 5-comp baseline ===")
    print(f"  {'W':>3} {'COI':<7} {'scope':<10} {'n':>3} {'β':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6} {'Egger int':>10} {'p':>6}")
    print("  " + "-" * 90)
    comparisons: List[Dict[str, Any]] = []
    meta_results: Dict[str, Any] = {}
    for w in windows:
        # 4-comp
        sub = long_df[long_df["window"] == w].dropna(subset=["beta_COI4_x_cum", "se_COI4_x_cum"]).copy()
        full4 = random_effects_meta(sub["beta_COI4_x_cum"].values.astype(float),
                                      sub["se_COI4_x_cum"].values.astype(float))
        full4_e = egger_test(sub["beta_COI4_x_cum"].values.astype(float),
                              sub["se_COI4_x_cum"].values.astype(float))
        drop = sub[sub["subcorpus"] != "Manchester"]
        drop4 = random_effects_meta(drop["beta_COI4_x_cum"].values.astype(float),
                                      drop["se_COI4_x_cum"].values.astype(float))
        drop4_e = egger_test(drop["beta_COI4_x_cum"].values.astype(float),
                              drop["se_COI4_x_cum"].values.astype(float))

        # 5-comp baseline from 17c
        bl_path = Path(f"./output/v17c/per_child_betas_N{w}.csv")
        bl = pd.read_csv(bl_path)
        # Dedup Manchester in baseline (same logic as 17h/17m)
        is_manc = bl["child"].isin(MANCHESTER_CHILDREN)
        bl = bl[(~is_manc) | (is_manc & (bl["corpus_label"] == "Manchester"))].copy()

        # Append April from 17j for baseline
        try:
            april5 = pd.read_csv("./output/v17j/na_pool_per_child_betas_loose.csv")
        except FileNotFoundError:
            april5 = None
        if april5 is not None and w == 5:
            april_row = {
                "child":           "April",
                "beta_COI_x_cum":  float(april5["beta_COI_x_cum"].iloc[0]),
                "se_COI_x_cum":    float(april5["se_COI_x_cum"].iloc[0]),
                "corpus_label":    "NA-Pool-April",
            }
            bl = pd.concat([bl, pd.DataFrame([april_row])], ignore_index=True, sort=False)
        elif w == 10:
            # April N=10 from 17m combined
            m_comb = pd.read_csv("./output/v17m/per_child_betas_combined_4windows.csv")
            ap10 = m_comb[(m_comb["child"] == "April") & (m_comb["window"] == 10)]
            if not ap10.empty:
                bl = pd.concat([bl, pd.DataFrame([{
                    "child":          "April",
                    "beta_COI_x_cum": float(ap10["beta"].iloc[0]),
                    "se_COI_x_cum":   float(ap10["se"].iloc[0]),
                    "corpus_label":   "NA-Pool-April",
                }])], ignore_index=True, sort=False)

        def assign_sub_bl(row):
            ch = row["child"]; cl = str(row.get("corpus_label", ""))
            if cl == "Brown":      return "Brown"
            if cl in ("Manchester",): return "Manchester"
            if ch in MANCHESTER_CHILDREN: return "Manchester"
            if cl.startswith("NA-Pool"):    return "NA_other"
            if ch in UK_LONG_LONG: return "UK_long_long"
            if cl == "English-UK": return "UK_short_obs"
            return "?"
        bl["subcorpus"] = bl.apply(assign_sub_bl, axis=1)
        bl = bl[bl["subcorpus"] != "?"].dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"]).copy()
        full5 = random_effects_meta(bl["beta_COI_x_cum"].values.astype(float),
                                      bl["se_COI_x_cum"].values.astype(float))
        full5_e = egger_test(bl["beta_COI_x_cum"].values.astype(float),
                              bl["se_COI_x_cum"].values.astype(float))
        bl_drop = bl[bl["subcorpus"] != "Manchester"]
        drop5 = random_effects_meta(bl_drop["beta_COI_x_cum"].values.astype(float),
                                      bl_drop["se_COI_x_cum"].values.astype(float))
        drop5_e = egger_test(bl_drop["beta_COI_x_cum"].values.astype(float),
                              bl_drop["se_COI_x_cum"].values.astype(float))

        for label, m5, m4, e5, e4 in [
            ("FULL",      full5, full4, full5_e, full4_e),
            ("DROP-MANC", drop5, drop4, drop5_e, drop4_e),
        ]:
            for tag, m, e in [("5-comp", m5, e5), ("4-comp", m4, e4)]:
                sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
                print(f"  {w:>3} {tag:<7} {label:<10} {m['n_studies']:>3} {m['pooled_beta_RE']:>+8.4f} "
                      f"{m['pooled_se_RE']:.4f} {m['pooled_p_RE']:.4f}{sig:<3} {m['tau2']:.4f} {m['I2_pct']:>5.1f}%  "
                      f"{e.get('intercept', float('nan')):>+9.3f} {e.get('intercept_p', float('nan')):>6.3f}")
            delta = m4["pooled_beta_RE"] - m5["pooled_beta_RE"]
            pct = (delta / m5["pooled_beta_RE"]) * 100 if m5["pooled_beta_RE"] != 0 else float("nan")
            comparisons.append({
                "window": w, "scope": label,
                "beta_5comp": m5["pooled_beta_RE"], "se_5comp": m5["pooled_se_RE"], "p_5comp": m5["pooled_p_RE"],
                "I2_5comp":   m5["I2_pct"],
                "beta_4comp": m4["pooled_beta_RE"], "se_4comp": m4["pooled_se_RE"], "p_4comp": m4["pooled_p_RE"],
                "I2_4comp":   m4["I2_pct"],
                "delta_beta": delta, "pct_change": pct,
                "egger_intercept_5comp_p": e5.get("intercept_p", float("nan")),
                "egger_intercept_4comp_p": e4.get("intercept_p", float("nan")),
            })

        meta_results[str(w)] = {
            "FULL_5comp":      full5, "FULL_4comp":      full4,
            "DROP_5comp":      drop5, "DROP_4comp":      drop4,
            "Egger_FULL_5":    full5_e, "Egger_FULL_4":  full4_e,
            "Egger_DROP_5":    drop5_e, "Egger_DROP_4":  drop4_e,
        }

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_dir / "comparison_5comp_vs_4comp.csv", index=False)
    print(f"\n  → {out_dir / 'comparison_5comp_vs_4comp.csv'}")
    with open(out_dir / "meta_4comp.json", "w", encoding="utf-8") as f:
        json.dump(meta_results, f, indent=2, default=str)
    print(f"  → {out_dir / 'meta_4comp.json'}")

    # SUMMARY.md
    direction_ok = all((r["beta_5comp"] > 0 and r["beta_4comp"] > 0) for r in comparisons)
    magnitude_ok = all(abs(r["pct_change"] / 100.0) < 0.30 for r in comparisons
                        if not math.isnan(r["pct_change"]))
    sig_ok = all((r["p_5comp"] < 0.05 and r["p_4comp"] < 0.05) for r in comparisons)
    lines: List[str] = []
    lines.append(f"# 19 — 4-component COI sensitivity test\n")
    lines.append("## β(COI × cumulative) comparison: 5-comp vs 4-comp\n")
    lines.append("| Window | Scope | β_5comp | SE_5 | p_5 | β_4comp | SE_4 | p_4 | Δβ | % change | Egger p (5/4) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in comparisons:
        sig5 = "***" if r['p_5comp']<0.001 else ("**" if r['p_5comp']<0.01 else ("*" if r['p_5comp']<0.05 else ("†" if r['p_5comp']<0.10 else "")))
        sig4 = "***" if r['p_4comp']<0.001 else ("**" if r['p_4comp']<0.01 else ("*" if r['p_4comp']<0.05 else ("†" if r['p_4comp']<0.10 else "")))
        lines.append(
            f"| N={r['window']} | {r['scope']} | {r['beta_5comp']:+.4f}{sig5} | "
            f"{r['se_5comp']:.4f} | {r['p_5comp']:.4f} | "
            f"{r['beta_4comp']:+.4f}{sig4} | {r['se_4comp']:.4f} | {r['p_4comp']:.4f} | "
            f"{r['delta_beta']:+.4f} | {r['pct_change']:+.1f}% | "
            f"{r['egger_intercept_5comp_p']:.3f} / {r['egger_intercept_4comp_p']:.3f} |"
        )
    lines.append("\n## Acceptance criteria\n")
    lines.append(f"- ✓ Direction agrees (both β > 0) all cells: **{direction_ok}**")
    lines.append(f"- ✓ |Δβ / β_5comp| < 0.30 all cells: **{magnitude_ok}**")
    lines.append(f"- ✓ Sig p < 0.05 all cells (both COI variants): **{sig_ok}**")
    if direction_ok and magnitude_ok and sig_ok:
        verdict = "PASS — exposure-gate β is independent of S_frequency_normalized. Frequency is not driving the effect."
    elif direction_ok and sig_ok:
        verdict = "PARTIAL — direction and sig hold but magnitude shift > 30% somewhere; clarify in SI."
    else:
        verdict = "FAIL — exposure-gate β depends materially on S_frequency_normalized inclusion."
    lines.append(f"\n**Verdict:** {verdict}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir / 'SUMMARY.md'}\n")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
