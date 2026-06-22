"""
17j_na_pool_per_child.py
========================
IMT Attention Bias Paper 2 — Step 17j: NA-Pool per-child Brown extension.

Brown (Adam, Eve, Sarah) is the only NA longitudinal corpus we have
per-child β estimates for. The English-NA-Pool was processed only as a
single corpus pool. This script splits the pool into individual children
(based on the json_cache subdirectory structure) and refits the
exposure-gate model per child on the NA-Pool data, then merges these
extra "Brown-extension" children into the 17h meta-analysis.

Eligible NA-Pool children (sufficient .cha file counts):
  Rollins (190), NewmanRatner07-24 (54-124 each), Tardif (25),
  Higginson family children (single-file each), single-name children
  (Alice, Amelia, ..., May) with ~14 files each.

This script restricts to NA-Pool children with ≥ 15 .cha files (excludes
single-file Higginson-style data) AND post-MSR contingent episodes ≥ 200.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17j/)
----------------------
* na_pool_per_child_betas.csv
* na_pool_extended_meta.json
* combined_31_plus_na_table.csv          combined with 17h's 31-child table
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / NA Brown extension v1 | 2026-06-21
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


NA_POOL_CFG = {
    "label":          "English-NA-Pool",
    "reuse_csv":      "./output/v16/English-NA-Pool_episodes_with_reuse.csv",
    "tagged_csv":     "./output/English-NA-Pool_tokens_tagged.csv",
    "joined_csv":     "./output/v11/English-NA-Pool_r_plus_joined.csv",
    "json_cache":     "./output/json_cache/English-NA-Pool",
}


def z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


def build_file_child_map(json_cache_dir: Path) -> Tuple[Dict[str, str], int, Dict[str, int]]:
    """Returns (file_stem -> child, max_stem_len, child -> n_files)."""
    mapping: Dict[str, str] = {}
    n_files: Dict[str, int] = {}
    max_len = 0
    if not json_cache_dir.exists():
        return mapping, 0, n_files
    for child_dir in json_cache_dir.iterdir():
        if not child_dir.is_dir():
            continue
        files = list(child_dir.rglob("*.json"))
        n_files[child_dir.name] = len(files)
        for jf in files:
            mapping[jf.stem] = child_dir.name
            if len(jf.stem) > max_len:
                max_len = len(jf.stem)
    return mapping, max_len, n_files


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
        "pooled_beta_FE": beta_fe,
        "pooled_se_FE": float(np.sqrt(1.0 / sum_w)),
    }


def fit_per_child(df: pd.DataFrame, outcome_col: str,
                   min_episodes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
        preds = ["COI_z_local", "cum_z_local", "COI_x_cum_local",
                 "prior_z_local", "logfreq_z_local"]
        X = sm.add_constant(gg[preds].astype(float), has_constant="add")
        y = gg[outcome_col].astype(float)
        try:
            fit = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
            )
            rows.append({
                "child":            child_id,
                "n_episodes":       int(len(gg)),
                "n_cues":           int(gg["cue_subtype"].nunique()),
                "age_min":          float(gg["child_age_months"].min()),
                "age_max":          float(gg["child_age_months"].max()),
                "obs_window":       float(gg["child_age_months"].max() - gg["child_age_months"].min()),
                "outcome_mean":     float(gg[outcome_col].mean()),
                "beta_COI_x_cum":   float(fit.params["COI_x_cum_local"]),
                "se_COI_x_cum":     float(fit.bse["COI_x_cum_local"]),
                "p_COI_x_cum":      float(fit.pvalues["COI_x_cum_local"]),
                "r2":               float(fit.rsquared),
            })
        except Exception as exc:
            rows.append({"child": child_id, "n_episodes": int(len(gg)), "error": str(exc)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="NA-Pool per-child Brown extension.")
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--prior_window", type=int, default=20)
    parser.add_argument("--min_files_per_child",   type=int, default=15)
    parser.add_argument("--min_episodes_per_child", type=int, default=200)
    parser.add_argument("--exclude_brown", action="store_true", default=True,
                         help="Brown (Adam/Eve/Sarah) already covered in 17c.")
    parser.add_argument("--exclude_providence", action="store_true", default=True,
                         help="Providence has near-zero caregiver speech.")
    parser.add_argument("--include_noncontingent", action="store_true")
    parser.add_argument("--combined_17h_csv",
                         default="./output/v17h/uk_subcorpora_per_child_N5.csv")
    parser.add_argument("--output_dir", default="./output/v17j")
    args = parser.parse_args()
    contingent_only = not args.include_noncontingent
    outcome_col = f"next_{args.window}_reuse"

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading NA-Pool data ===")
    reuse_csv = Path(NA_POOL_CFG["reuse_csv"])
    if not reuse_csv.exists():
        sys.exit(f"  ERROR: missing {reuse_csv}")
    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    if contingent_only:
        reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    if outcome_col not in reuse.columns:
        sys.exit(f"  ERROR: missing {outcome_col} in reuse CSV")
    print(f"  contingent rows: {len(reuse):,}")

    file_index = build_child_utt_index(Path(NA_POOL_CFG["tagged_csv"]))
    reuse = add_prior_local_freq(reuse, file_index, args.prior_window)

    file_to_child, max_len, n_files_per_child = build_file_child_map(Path(NA_POOL_CFG["json_cache"]))
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()

    # Restrict to eligible children
    eligible_children = {
        c for c, n in n_files_per_child.items() if n >= args.min_files_per_child
    }
    if args.exclude_brown:
        eligible_children -= {"Adam", "Eve", "Sarah"}
    if args.exclude_providence:
        eligible_children.discard("Providence")
    print(f"  NA-Pool children passing min_files ≥ {args.min_files_per_child}: "
          f"{len(eligible_children)} → {sorted(eligible_children)}")

    reuse = reuse[reuse["child"].isin(eligible_children)].copy()
    print(f"  reuse rows after child filter: {len(reuse):,}")

    # Merge COI / log_cue_freq from joined CSV
    joined_csv = Path(NA_POOL_CFG["joined_csv"])
    if not joined_csv.exists():
        sys.exit(f"  ERROR: missing {joined_csv}")
    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]],
                       on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    print(f"  after merge & dropna: {len(df):,}")

    # Cumulative cue attempts (lifetime per child × cue)
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()

    df_post = df[df["child_age_months"] >= 24].copy()
    print(f"  post-MSR rows: {len(df_post):,}    "
          f"children with any post-MSR rows: {df_post['child'].nunique()}")

    # Per-child fit
    print(f"\n=== Per-child OLS (min_episodes = {args.min_episodes_per_child}) ===")
    per_child = fit_per_child(df_post, outcome_col, args.min_episodes_per_child)
    per_child["n_files_in_corpus_dir"] = per_child["child"].map(n_files_per_child).fillna(0).astype(int)
    per_child["corpus_label"]          = "NA-Pool-other"
    per_child["language"]              = "English-NA-Pool"

    # Classify by file count (mirror 17h's threshold scheme)
    def _classify(n):
        return "NA_long_longitudinal" if n >= 100 else (
            "NA_medium" if n >= 30 else "NA_short"
        )
    per_child["subcorpus"] = per_child["n_files_in_corpus_dir"].apply(_classify)
    per_child.to_csv(out_dir / "na_pool_per_child_betas.csv", index=False)
    print(f"  → {out_dir / 'na_pool_per_child_betas.csv'}  ({len(per_child)} rows)")

    if len(per_child) == 0:
        sys.exit("  No eligible NA-Pool children — stopping.")

    print(f"\n  Per-child results sorted by n_files:")
    show = per_child.sort_values("n_files_in_corpus_dir", ascending=False)
    print(f"  {'child':<18} {'sub':<22} {'files':>6} {'n_eps':>6} {'window':>7} {'β':>8} {'SE':>7} {'p':>7}")
    for _, r in show.iterrows():
        sig = '***' if r['p_COI_x_cum']<0.001 else ('**' if r['p_COI_x_cum']<0.01 else ('*' if r['p_COI_x_cum']<0.05 else ('†' if r['p_COI_x_cum']<0.10 else '')))
        print(f"  {str(r['child']):<18} {r['subcorpus']:<22} {int(r['n_files_in_corpus_dir']):>6} {int(r['n_episodes']):>6,} {r['obs_window']:>7.1f} {r['beta_COI_x_cum']:>+7.4f}  {r['se_COI_x_cum']:.4f}  {r['p_COI_x_cum']:.4f} {sig}")

    # Meta within NA-Pool extension
    print(f"\n=== NA-Pool extension meta ===")
    by_sub: Dict[str, Dict[str, Any]] = {}
    for sub, g in per_child.groupby("subcorpus"):
        meta = (
            random_effects_meta(
                g["beta_COI_x_cum"].astype(float).values,
                g["se_COI_x_cum"].astype(float).values,
            ) if len(g) >= 2 else {
                "n_studies": int(len(g)),
                "note": "n<2",
                **({"beta_single": float(g["beta_COI_x_cum"].iloc[0]),
                    "se_single":   float(g["se_COI_x_cum"].iloc[0])}
                    if len(g) == 1 else {}),
            }
        )
        by_sub[sub] = meta
        if "pooled_beta_RE" in meta:
            print(f"  {sub:<22} (n={meta['n_studies']}): "
                  f"β_pooled = {meta['pooled_beta_RE']:+.4f}  "
                  f"SE = {meta['pooled_se_RE']:.4f}  p = {meta['pooled_p_RE']:.4f}  "
                  f"τ² = {meta['tau2']:.4f}  I² = {meta['I2_pct']:.1f}%")
        else:
            print(f"  {sub:<22} (n={meta.get('n_studies',0)}): {meta.get('note','—')}")

    # Combined with 17h's 31-child table
    print(f"\n=== Combined with 17h 31-child table ===")
    if Path(args.combined_17h_csv).exists():
        prev = pd.read_csv(args.combined_17h_csv)
        cols = ["child", "n_episodes", "obs_window", "beta_COI_x_cum",
                "se_COI_x_cum", "p_COI_x_cum", "subcorpus", "corpus_label"]
        prev_sub = prev.rename(columns={"obs_window_months": "obs_window"})
        prev_sub = prev_sub[[c for c in cols if c in prev_sub.columns]].copy()
        new_sub = per_child[[c for c in cols if c in per_child.columns]].copy()
        combined = pd.concat([prev_sub, new_sub], ignore_index=True)
        combined.to_csv(out_dir / "combined_17h_plus_na_table.csv", index=False)
        print(f"  → {out_dir / 'combined_17h_plus_na_table.csv'}  ({len(combined)} rows)")

        # Final meta — pool of all groups
        print(f"\n  Final per-subcorpus meta (Manchester deduplicated; NA additions included):")
        # Drop "(also in UK pool)" Manchester duplicates if any
        combined_dedup = combined[~combined["subcorpus"].astype(str).str.contains(
            "also in UK pool", na=False)].copy()
        for sub, g in combined_dedup.groupby("subcorpus"):
            g = g.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"])
            if len(g) < 2:
                if len(g) == 1:
                    print(f"    {sub:<40} n=1  β={g['beta_COI_x_cum'].iloc[0]:+.4f}  SE={g['se_COI_x_cum'].iloc[0]:.4f}")
                continue
            meta = random_effects_meta(
                g["beta_COI_x_cum"].astype(float).values,
                g["se_COI_x_cum"].astype(float).values,
            )
            print(f"    {sub:<40} n={meta['n_studies']:>3}  "
                  f"β={meta['pooled_beta_RE']:+.4f}  "
                  f"SE={meta['pooled_se_RE']:.4f}  p={meta['pooled_p_RE']:.4f}  "
                  f"τ²={meta['tau2']:.4f}  I²={meta['I2_pct']:.1f}%")

        # Grand-pooled across all children
        all_meta = random_effects_meta(
            combined_dedup["beta_COI_x_cum"].astype(float).values,
            combined_dedup["se_COI_x_cum"].astype(float).values,
        )
        print(f"\n  Pooled across all children (n={all_meta['n_studies']}): "
              f"β = {all_meta['pooled_beta_RE']:+.4f}, "
              f"SE = {all_meta['pooled_se_RE']:.4f}, "
              f"p = {all_meta['pooled_p_RE']:.4f}, "
              f"τ² = {all_meta['tau2']:.4f}, I² = {all_meta['I2_pct']:.1f}%")
    else:
        combined = None
        print(f"  ({args.combined_17h_csv} not found — skipping combined meta)")
        all_meta = None

    # Persist
    summary = {
        "_meta": {
            "window":        args.window,
            "prior_window":  args.prior_window,
            "min_files_per_child":    args.min_files_per_child,
            "min_episodes_per_child": args.min_episodes_per_child,
        },
        "na_pool_per_child":    per_child.to_dict(orient="records"),
        "meta_by_subcorpus_na": by_sub,
    }
    if all_meta is not None:
        summary["grand_pooled_meta"] = all_meta
    with open(out_dir / "na_pool_extended_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → {out_dir / 'na_pool_extended_meta.json'}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# 17j NA-Pool Brown extension — outcome window N={args.window}\n")
    lines.append("## Per-child β (sorted by n_files)\n")
    lines.append("| Child | sub | files | n_eps | window (mo) | β | SE | p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in per_child.sort_values("n_files_in_corpus_dir", ascending=False).iterrows():
        lines.append(
            f"| {r['child']} | {r['subcorpus']} | {int(r['n_files_in_corpus_dir'])} | "
            f"{int(r['n_episodes']):,} | {r['obs_window']:.1f} | "
            f"{r['beta_COI_x_cum']:+.4f} | {r['se_COI_x_cum']:.4f} | "
            f"{r['p_COI_x_cum']:.4f} |"
        )
    lines.append("\n## NA-Pool extension meta by sub-group\n")
    lines.append("| Sub-group | n_children | β pooled | SE | p | τ² | I² |")
    lines.append("|---|---|---|---|---|---|---|")
    for sub, m in by_sub.items():
        if "pooled_beta_RE" in m:
            lines.append(
                f"| {sub} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
            )
        else:
            lines.append(f"| {sub} | {m.get('n_studies',0)} | {m.get('note','—')} |  |  |  |  |")
    if all_meta is not None:
        lines.append(f"\n## Grand-pooled meta (17h 31 children + NA-Pool additions)\n")
        lines.append(f"- n = {all_meta['n_studies']}")
        lines.append(f"- β_pooled = {all_meta['pooled_beta_RE']:+.4f}")
        lines.append(f"- SE = {all_meta['pooled_se_RE']:.4f}")
        lines.append(f"- p = {all_meta['pooled_p_RE']:.4f}")
        lines.append(f"- τ² = {all_meta['tau2']:.4f}")
        lines.append(f"- I² = {all_meta['I2_pct']:.1f}%")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
