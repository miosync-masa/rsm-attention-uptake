"""
17e_adequate_window_subset.py
=============================
IMT Attention Bias Paper 2 — Step 17e:
Re-run SPEC #1b / #1c restricted to children with adequate observation window.

Following 17d's finding that obs_window_months is a positive moderator of
β_i(COI × cumulative_cue_attempts), this script tests:

  "If we restrict the analysis to children whose post-MSR observation
   window is at least T months, does the exposure-gate effect become
   robust across corpora?"

For three thresholds T ∈ {12, 9, 6} mo, we:
  (1) Identify the eligible children (post-MSR obs window ≥ T).
  (2) Refit the corpus-level model
        reuse_nextN ~ COI_z × cumulative_cue_attempts_z
                    + prior_local_freq_z + log_cue_freq_z
      with cluster-robust SE on cue, on each corpus's restricted episode set.
  (3) Per-child random-effects meta-analysis on β_i (from 17c's per-child
      OLS fits), restricted to eligible children.
  (4) Per-corpus availability table.

Caveat: Manchester children all have post-MSR observation < 12 mo. At
T = 12 mo Manchester contributes 0 children; at T = 9 mo only a few; at
T = 6 mo most are included. The script reports availability honestly.

────────────────────────────────────────────────────────────────────────────

Inputs
------
* output/v17d/per_child_with_window_N{N}.csv   per-child β + obs_window
* output/v16/{lang}_episodes_with_reuse.csv
* output/v11*/{lang}_r_plus_joined.csv
* output/json_cache/{lang}/...

Outputs (output/v17e/)
----------------------
* adequate_window_results_N{N}.json
* adequate_window_summary_N{N}.csv
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / adequate window v1 | 2026-06-21
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
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with 17b/17c/17d)
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
    z_re = beta_re / se_re if se_re > 0 else float("nan")
    p = float(2 * (1.0 - norm.cdf(abs(z_re)))) if not math.isnan(z_re) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": z_re, "pooled_p_RE": p,
        "pooled_beta_FE": beta_fe,
        "pooled_se_FE": float(np.sqrt(1.0 / sum_w)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus-level refit on a child whitelist
# ─────────────────────────────────────────────────────────────────────────────

def refit_corpus_subset(language: str, cfg: Dict[str, str],
                          eligible_children: List[str],
                          window: int, prior_window: int,
                          contingent_only: bool) -> Dict[str, Any]:
    """Replicates 17b's M_new on the post-MSR subset restricted to
    `eligible_children` only."""
    outcome_col = f"next_{window}_reuse"

    reuse_csv  = Path(cfg["reuse_csv"])
    tagged_csv = Path(cfg["tagged_csv"])
    joined_csv = Path(cfg["joined_csv"])
    json_cache = Path(cfg["json_cache"])

    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    if outcome_col not in reuse.columns:
        return {"error": f"missing {outcome_col}"}

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

    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()

    # Cumulative cue attempts (lifetime, computed before restricting to subset)
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()

    # Post-MSR + eligible children
    df_post = df[df["child_age_months"] >= 24].copy()
    df_sub = df_post[df_post["child"].isin(eligible_children)].copy()
    if len(df_sub) < 100:
        return {
            "language": language,
            "n_eligible_children": len(eligible_children),
            "n_post_in_subset": int(len(df_sub)),
            "error": "n_post_in_subset < 100",
        }

    df_sub["COI_z"]                      = z(df_sub["COI"])
    df_sub["cumulative_cue_attempts_z"]  = z(df_sub["cumulative_cue_attempts"])
    df_sub["prior_local_freq_z"]         = z(df_sub["prior_local_freq"])
    df_sub["log_cue_freq_z"]             = z(df_sub["log_cue_freq"])
    df_sub["COI_x_cumulative"]           = df_sub["COI_z"] * df_sub["cumulative_cue_attempts_z"]

    preds = ["COI_z", "cumulative_cue_attempts_z", "COI_x_cumulative",
             "prior_local_freq_z", "log_cue_freq_z"]
    X = sm.add_constant(df_sub[preds].astype(float), has_constant="add")
    y = df_sub[outcome_col].astype(float)
    try:
        fit = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_sub["cue_subtype"].astype(str).values},
        )
    except Exception as exc:
        return {"language": language, "error": f"OLS failed: {exc}"}

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
        "language":              language,
        "n_eligible_children":   len(eligible_children),
        "n_post_in_subset":      int(len(df_sub)),
        "n_cues":                int(df_sub["cue_subtype"].nunique()),
        "r2":                    float(fit.rsquared),
        "params":                params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Refit on adequate-window subset.")
    parser.add_argument("--per_child_with_window_csv",
                         default="./output/v17d/per_child_with_window_N5.csv")
    parser.add_argument("--output_dir",  default="./output/v17e")
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--thresholds",  default="12,9,6",
                         help="Comma-separated obs_window thresholds (months).")
    parser.add_argument("--include_noncontingent", action="store_true")
    args = parser.parse_args()
    contingent_only = not args.include_noncontingent

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-child window table: {args.per_child_with_window_csv}")
    per_child = pd.read_csv(args.per_child_with_window_csv)
    needed = {"child", "corpus_label", "obs_window_months",
              "beta_COI_x_cum", "se_COI_x_cum"}
    missing = needed - set(per_child.columns)
    if missing:
        sys.exit(f"  ERROR: missing columns in per_child CSV: {missing}")

    label_to_lang = {v["label"]: lang for lang, v in CANONICAL_CORPORA.items()}

    all_results: Dict[str, Any] = {"_meta": {
        "window": args.window, "prior_window": args.prior_window,
        "thresholds": thresholds, "contingent_only": contingent_only,
    }, "thresholds": {}}

    for T in thresholds:
        print(f"\n========== Threshold T = {T:.0f} months ==========")
        elig = per_child[per_child["obs_window_months"] >= T].copy()
        avail = elig["corpus_label"].value_counts().to_dict()
        print(f"  Eligible children: total = {len(elig)}  | per corpus = {avail}")

        # SPEC #1c-style per-child meta on the subset
        meta_overall = random_effects_meta(
            elig["beta_COI_x_cum"].astype(float).values,
            elig["se_COI_x_cum"].astype(float).values,
        ) if len(elig) >= 2 else {"error": "n < 2"}

        meta_by_corpus: Dict[str, Any] = {}
        for label, g in elig.groupby("corpus_label"):
            meta_by_corpus[label] = (
                random_effects_meta(
                    g["beta_COI_x_cum"].astype(float).values,
                    g["se_COI_x_cum"].astype(float).values,
                ) if len(g) >= 2 else {
                    "n_studies": int(len(g)),
                    "note": "n<2; reporting single-study β if available",
                    **(
                        {"beta_single": float(g["beta_COI_x_cum"].iloc[0]),
                         "se_single":   float(g["se_COI_x_cum"].iloc[0])}
                        if len(g) == 1 else {}
                    ),
                }
            )

        # SPEC #1b-style corpus-level refit on the subset (per corpus)
        corpus_refits: Dict[str, Any] = {}
        for label, lang in label_to_lang.items():
            cfg = CANONICAL_CORPORA[lang]
            eligible_kids = elig[elig["corpus_label"] == label]["child"].astype(str).tolist()
            if len(eligible_kids) == 0:
                corpus_refits[label] = {"n_eligible_children": 0, "note": "no eligible children"}
                continue
            print(f"\n  Refitting {label} (T={T:.0f}) on {len(eligible_kids)} eligible children ...")
            res = refit_corpus_subset(
                lang, cfg, eligible_kids,
                window=args.window, prior_window=args.prior_window,
                contingent_only=contingent_only,
            )
            corpus_refits[label] = res

        # Print summary for this threshold
        print(f"\n  -- SPEC #1b-style corpus refit, β(COI × cumulative) --")
        for label in ["Brown", "Manchester", "English-UK"]:
            r = corpus_refits.get(label, {})
            if "params" not in r:
                print(f"    {label:<12}: (no refit — {r.get('error', r.get('note', 'n/a'))})")
                continue
            cxc = r["params"]["COI_x_cumulative"]
            sig = '***' if cxc['p']<0.001 else ('**' if cxc['p']<0.01 else ('*' if cxc['p']<0.05 else ('†' if cxc['p']<0.10 else '')))
            print(
                f"    {label:<12}: n_children={r['n_eligible_children']:>2}  n_post={r['n_post_in_subset']:>7,}  "
                f"β={cxc['beta']:+.4f}  SE={cxc['se']:.4f}  p={cxc['p']:.4f} {sig}"
            )

        print(f"\n  -- SPEC #1c-style per-child meta (subset) --")
        if "pooled_beta_RE" in meta_overall:
            print(
                f"    All children (n={meta_overall['n_studies']}): "
                f"β_pooled={meta_overall['pooled_beta_RE']:+.4f}  "
                f"SE={meta_overall['pooled_se_RE']:.4f}  p={meta_overall['pooled_p_RE']:.4f}  "
                f"τ²={meta_overall['tau2']:.4f}  I²={meta_overall['I2_pct']:.1f}%"
            )
        else:
            print(f"    All children: {meta_overall.get('error','?')}")
        for label in ["Brown", "Manchester", "English-UK"]:
            m = meta_by_corpus.get(label, {})
            if "pooled_beta_RE" in m:
                print(
                    f"    {label:<12} (n={m['n_studies']}): "
                    f"β_pooled={m['pooled_beta_RE']:+.4f}  SE={m['pooled_se_RE']:.4f}  "
                    f"p={m['pooled_p_RE']:.4f}  τ²={m['tau2']:.4f}  I²={m['I2_pct']:.1f}%"
                )
            else:
                print(f"    {label:<12}: {m.get('note', m.get('error','—'))}")

        all_results["thresholds"][str(int(T))] = {
            "T_months":          T,
            "n_eligible_total":  int(len(elig)),
            "availability":      avail,
            "meta_overall":      meta_overall,
            "meta_by_corpus":    meta_by_corpus,
            "corpus_refits":     corpus_refits,
        }

    # Persist
    json_path = out_dir / f"adequate_window_results_N{args.window}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {json_path}")

    # Build summary CSV (long form: one row per threshold × corpus × predictor)
    rows: List[Dict[str, Any]] = []
    for T_key, block in all_results["thresholds"].items():
        for label, refit in block["corpus_refits"].items():
            if "params" not in refit:
                continue
            for pred, vals in refit["params"].items():
                rows.append({
                    "threshold_months":     block["T_months"],
                    "corpus":               label,
                    "n_children":           refit["n_eligible_children"],
                    "n_post":               refit["n_post_in_subset"],
                    "n_cues":               refit["n_cues"],
                    "spec":                 "SPEC1b_corpus_refit",
                    "predictor":            pred,
                    "beta":                 vals["beta"],
                    "se":                   vals["se"],
                    "t":                    vals["t"],
                    "p":                    vals["p"],
                    "ci95_low":             vals.get("ci95_low"),
                    "ci95_high":            vals.get("ci95_high"),
                })
        # Per-corpus meta rows
        for label, m in block["meta_by_corpus"].items():
            if "pooled_beta_RE" in m:
                rows.append({
                    "threshold_months":     block["T_months"],
                    "corpus":               label,
                    "n_children":           m["n_studies"],
                    "spec":                 "SPEC1c_meta_per_corpus",
                    "predictor":            "pooled_COI_x_cumulative",
                    "beta":                 m["pooled_beta_RE"],
                    "se":                   m["pooled_se_RE"],
                    "p":                    m["pooled_p_RE"],
                    "tau2":                 m["tau2"],
                    "I2_pct":               m["I2_pct"],
                })
        # Overall meta
        m = block["meta_overall"]
        if "pooled_beta_RE" in m:
            rows.append({
                "threshold_months":         block["T_months"],
                "corpus":                   "ALL",
                "n_children":               m["n_studies"],
                "spec":                     "SPEC1c_meta_overall",
                "predictor":                "pooled_COI_x_cumulative",
                "beta":                     m["pooled_beta_RE"],
                "se":                       m["pooled_se_RE"],
                "p":                        m["pooled_p_RE"],
                "tau2":                     m["tau2"],
                "I2_pct":                   m["I2_pct"],
            })
    summary = pd.DataFrame(rows)
    csv_path = out_dir / f"adequate_window_summary_N{args.window}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  → {csv_path}  ({len(summary):,} rows)")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# SPEC #1e adequate-window subset — outcome window N={args.window}\n")
    for T_key, block in all_results["thresholds"].items():
        T = block["T_months"]
        lines.append(f"\n## T = {T:.0f} months   (n_eligible total = {block['n_eligible_total']})\n")
        lines.append("### Availability per corpus\n")
        avail = block["availability"]
        lines.append("| Corpus | Eligible children |")
        lines.append("|---|---|")
        for label in ["Brown", "Manchester", "English-UK"]:
            lines.append(f"| {label} | {avail.get(label, 0)} |")

        lines.append("\n### SPEC #1b corpus refit β(COI × cumulative)\n")
        lines.append("| Corpus | n_children | n_post | β | SE | p | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for label in ["Brown", "Manchester", "English-UK"]:
            refit = block["corpus_refits"].get(label, {})
            if "params" not in refit:
                lines.append(f"| {label} | {refit.get('n_eligible_children', 0)} | — | — | — | — | n/a |")
                continue
            cxc = refit["params"]["COI_x_cumulative"]
            ok = (cxc["beta"] > 0) and (cxc["p"] < 0.01)
            verdict = "✓ PASS" if ok else "✗"
            lines.append(
                f"| {label} | {refit['n_eligible_children']} | {refit['n_post_in_subset']:,} | "
                f"{cxc['beta']:+.4f} | {cxc['se']:.4f} | {cxc['p']:.4f} | {verdict} |"
            )

        lines.append("\n### SPEC #1c per-child meta (β_i pooled)\n")
        lines.append("| Scope | n | pooled β (RE) | SE | p | τ² | I² |")
        lines.append("|---|---|---|---|---|---|---|")
        m = block["meta_overall"]
        if "pooled_beta_RE" in m:
            lines.append(
                f"| All eligible | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
            )
        for label in ["Brown", "Manchester", "English-UK"]:
            m = block["meta_by_corpus"].get(label, {})
            if "pooled_beta_RE" in m:
                lines.append(
                    f"| {label} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                    f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                    f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
                )
            else:
                lines.append(f"| {label} | {m.get('n_studies', 0)} | n<2 or unavailable | | | | |")

    summary_md = out_dir / "SUMMARY.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {summary_md}")


if __name__ == "__main__":
    main()
