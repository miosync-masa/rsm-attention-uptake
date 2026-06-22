"""
17h_uk_subcorpora_meta.py
=========================
IMT Attention Bias Paper 2 — Step 17h: UK sub-corpora replication

17g ruled out window, MLU, density, and cue-mix as sole explanations for
the Manchester null. A structural property remained: the UK "English-UK"
pool used in 17b-17g actually mixes:

  - Manchester children                (~30-35 .cha files / child)
  - Long-longitudinal single-child:
      Thomas      (379 .cha files)
      Fraser      (217)
      Helen       (184)
      Eleanor     (181)
      Nicole      (34)
  - Short-observation multi-child       (~10 .cha files / child)
      Abigail, Frances, Geoffrey, Jack, Jason, Jonathan,
      Laura, Neville, Penny, Samantha, Sean, Stella

This script splits the 17 UK-non-Manchester children into these two
sub-groups (long-longitudinal vs short-observation) using the per-child
n_files (sessions) attribute, then re-runs the random-effects meta-analysis
and checks whether the Manchester null replicates in the short-observation
sub-corpus.

────────────────────────────────────────────────────────────────────────────

Sub-groups
----------
  L  UK_long_longitudinal :  n_files ≥ 30   (Thomas/Fraser/Helen/Eleanor/Nicole)
  S  UK_short_observation :  n_files <  30  (Wells-Bristol-like multi-child)
  M  Manchester           :  the 11 Manchester children (medium n_files)
  B  Brown                :  Adam/Eve/Sarah

If the exposure-gate effect is driven by per-child observation density:
  L should resemble Brown (β > 0, sig)
  S should resemble Manchester (β ≈ 0)
  → Manchester's null becomes part of a continuous "density" pattern.

If the L sub-group ALSO shows the effect and S sub-group also shows it,
Manchester is genuinely a corpus-specific outlier.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17h/)
----------------------
* uk_subcorpora_meta_N{N}.json
* uk_subcorpora_per_child_N{N}.csv
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / UK subcorpora v1 | 2026-06-21
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


CHILDES_UK_DIR = Path("~/childes_data/English-UK").expanduser()

MANCHESTER_CHILDREN = {
    "Anne", "Aran", "Becky", "Carl", "Dominic", "Gail",
    "Joel", "John", "Liz", "Ruth", "Warren",
}


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
    z = beta_re / se_re if se_re > 0 else float("nan")
    p = float(2 * (1.0 - norm.cdf(abs(z)))) if not math.isnan(z) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": z, "pooled_p_RE": p,
        "pooled_beta_FE": beta_fe,
        "pooled_se_FE": float(np.sqrt(1.0 / sum_w)),
    }


def count_n_files(child: str) -> int:
    p = CHILDES_UK_DIR / child
    if not p.exists():
        return 0
    return len(list(p.rglob("*.cha")))


def main() -> None:
    parser = argparse.ArgumentParser(description="UK sub-corpora meta-analysis.")
    parser.add_argument("--per_child_csv",
                         default="./output/v17c/per_child_betas_N5.csv")
    parser.add_argument("--per_child_with_window_csv",
                         default="./output/v17d/per_child_with_window_N5.csv")
    parser.add_argument("--output_dir", default="./output/v17h")
    parser.add_argument("--window",     type=int, default=5)
    parser.add_argument("--long_threshold", type=int, default=30,
                         help="n_files ≥ threshold → long-longitudinal sub-corpus")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-child β table: {args.per_child_csv}")
    per_child = pd.read_csv(args.per_child_csv)
    # Drop error rows
    per_child = per_child.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"]).copy()

    # Add window stats if available
    try:
        win = pd.read_csv(args.per_child_with_window_csv)
        per_child = per_child.merge(
            win[["child", "corpus_label", "obs_window_months",
                  "age_min_post", "age_max_post"]],
            on=["child", "corpus_label"], how="left",
        )
    except FileNotFoundError:
        print(f"  ({args.per_child_with_window_csv} not found — proceeding without window stats)")

    # Annotate UK pool children with sub-corpus
    def _subcorpus(row):
        if row["corpus_label"] == "Brown":
            return "Brown"
        if row["corpus_label"] == "Manchester":
            return "Manchester"
        # corpus_label == "English-UK" — distinguish Manchester vs UK-other
        if row["child"] in MANCHESTER_CHILDREN:
            # This is a Manchester child counted under English-UK pool
            return "Manchester (also in UK pool)"
        n_files = count_n_files(row["child"])
        if n_files >= args.long_threshold:
            return f"UK_long_longitudinal (n_files≥{args.long_threshold})"
        return f"UK_short_observation (n_files<{args.long_threshold})"

    per_child["subcorpus"] = per_child.apply(_subcorpus, axis=1)
    per_child["n_files_in_corpus_dir"] = per_child["child"].apply(count_n_files)

    print("\nPer-child sub-corpus assignment:")
    for sc, g in per_child.groupby("subcorpus"):
        print(f"  {sc:<46}: n = {len(g)}")
        for _, r in g.sort_values("n_files_in_corpus_dir", ascending=False).iterrows():
            sig = '***' if r['p_COI_x_cum']<0.001 else ('**' if r['p_COI_x_cum']<0.01 else ('*' if r['p_COI_x_cum']<0.05 else ''))
            window_str = f"window={r.get('obs_window_months', float('nan')):.1f}mo" if pd.notna(r.get('obs_window_months', np.nan)) else "window=N/A"
            print(f"    {str(r['child']):<12} files={int(r['n_files_in_corpus_dir']):>4}  "
                  f"n_eps={int(r['n_episodes']):>6,}  {window_str:<16}  "
                  f"β={r['beta_COI_x_cum']:+.4f}  SE={r['se_COI_x_cum']:.4f}  p={r['p_COI_x_cum']:.3f}{sig}")

    # ───── Meta-analysis per sub-corpus ─────
    print("\n========== Random-effects meta-analysis per sub-corpus ==========")
    print(f"  {'Sub-corpus':<46} {'n':>3} {'β_pooled':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6}")
    print('  ' + '-' * 90)
    meta_by_subcorpus: Dict[str, Dict[str, Any]] = {}
    for sc, g in per_child.groupby("subcorpus"):
        meta = random_effects_meta(
            g["beta_COI_x_cum"].astype(float).values,
            g["se_COI_x_cum"].astype(float).values,
        ) if len(g) >= 2 else {
            "n_studies": int(len(g)),
            "note": "n<2",
            **({"beta_single": float(g["beta_COI_x_cum"].iloc[0]),
                "se_single":   float(g["se_COI_x_cum"].iloc[0])}
                if len(g) == 1 else {}),
        }
        meta_by_subcorpus[sc] = meta
        if "pooled_beta_RE" in meta:
            print(f"  {sc:<46} {meta['n_studies']:>3} {meta['pooled_beta_RE']:>+8.4f} "
                  f"{meta['pooled_se_RE']:.4f} {meta['pooled_p_RE']:.4f} "
                  f"{meta['tau2']:.4f} {meta['I2_pct']:>5.1f}%")
        else:
            note = meta.get("note", "—")
            print(f"  {sc:<46} {meta.get('n_studies',0):>3} ({note})")

    # ───── Specific contrast: UK_long_longitudinal vs Manchester vs UK_short_observation ─────
    print("\n========== Key contrast: 4-way meta ==========")
    # Deduplicate Manchester children (they appear in both 'Manchester' and
    # 'Manchester (also in UK pool)' subcorpora). Use the 'Manchester'-labeled
    # rows as primary; drop duplicates.
    dedup = per_child[per_child["subcorpus"] != "Manchester (also in UK pool)"].copy()
    # Use shorter names for the final 4-way table
    def _short(sc: str) -> str:
        if sc.startswith("UK_long"):  return "UK_long_long"
        if sc.startswith("UK_short"): return "UK_short_obs"
        return sc
    dedup["group"] = dedup["subcorpus"].apply(_short)

    print(f"  After Manchester dedup: total unique children = {len(dedup)}")
    print(f"  Distribution by group: {dedup['group'].value_counts().to_dict()}")

    print(f"\n  {'Group':<22} {'n':>3} {'β_pooled':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6}")
    print('  ' + '-' * 66)
    meta_4way: Dict[str, Dict[str, Any]] = {}
    for grp, g in dedup.groupby("group"):
        meta = random_effects_meta(
            g["beta_COI_x_cum"].astype(float).values,
            g["se_COI_x_cum"].astype(float).values,
        ) if len(g) >= 2 else {
            "n_studies": int(len(g)),
            "note": "n<2",
            **({"beta_single": float(g["beta_COI_x_cum"].iloc[0]),
                "se_single":   float(g["se_COI_x_cum"].iloc[0])}
                if len(g) == 1 else {}),
        }
        meta_4way[grp] = meta
        if "pooled_beta_RE" in meta:
            print(f"  {grp:<22} {meta['n_studies']:>3} {meta['pooled_beta_RE']:>+8.4f} "
                  f"{meta['pooled_se_RE']:.4f} {meta['pooled_p_RE']:.4f} "
                  f"{meta['tau2']:.4f} {meta['I2_pct']:>5.1f}%")
        else:
            note = meta.get("note", "—")
            print(f"  {grp:<22} {meta.get('n_studies',0):>3} ({note})")

    # Combined all (no Manchester dedup): for reference
    all_meta = random_effects_meta(
        dedup["beta_COI_x_cum"].astype(float).values,
        dedup["se_COI_x_cum"].astype(float).values,
    )
    print(f"\n  Pooled across all 4 groups (n={all_meta['n_studies']}): "
          f"β = {all_meta['pooled_beta_RE']:+.4f}, "
          f"p = {all_meta['pooled_p_RE']:.4f}, "
          f"τ² = {all_meta['tau2']:.4f}, "
          f"I² = {all_meta['I2_pct']:.1f}%")

    # ───── Save outputs ─────
    out = {
        "_meta": {
            "window": args.window,
            "long_threshold_n_files": args.long_threshold,
        },
        "per_child_table": dedup.to_dict(orient="records"),
        "meta_by_subcorpus": meta_by_subcorpus,
        "meta_4way": meta_4way,
        "meta_all_4_groups": all_meta,
    }
    with open(out_dir / f"uk_subcorpora_meta_N{args.window}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  → {out_dir / f'uk_subcorpora_meta_N{args.window}.json'}")

    dedup.to_csv(out_dir / f"uk_subcorpora_per_child_N{args.window}.csv", index=False)
    print(f"  → {out_dir / f'uk_subcorpora_per_child_N{args.window}.csv'}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# 17h UK sub-corpora replication — outcome window N={args.window}\n")
    lines.append("## Per-child sub-corpus assignment\n")
    lines.append("| Sub-corpus | n_children | mean β | β range |")
    lines.append("|---|---|---|---|")
    for sc, g in per_child.groupby("subcorpus"):
        if len(g) > 0:
            lines.append(f"| {sc} | {len(g)} | {g['beta_COI_x_cum'].mean():+.4f} | "
                          f"{g['beta_COI_x_cum'].min():+.4f} to {g['beta_COI_x_cum'].max():+.4f} |")

    lines.append("\n## 4-way random-effects meta (Manchester deduplicated)\n")
    lines.append("| Group | n_children | pooled β | SE | p | τ² | I² |")
    lines.append("|---|---|---|---|---|---|---|")
    for grp in ["Brown", "Manchester", "UK_long_long", "UK_short_obs"]:
        m = meta_4way.get(grp)
        if m is None:
            continue
        if "pooled_beta_RE" in m:
            lines.append(
                f"| {grp} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | "
                f"{m['tau2']:.4f} | {m['I2_pct']:.1f}% |"
            )
        else:
            lines.append(f"| {grp} | {m.get('n_studies',0)} | {m.get('note','—')} |  |  |  |  |")

    # Verdict
    lines.append("\n## Verdict\n")
    long_pass = (
        "UK_long_long" in meta_4way and
        meta_4way["UK_long_long"].get("pooled_beta_RE", 0) > 0 and
        meta_4way["UK_long_long"].get("pooled_p_RE", 1) < 0.05
    )
    short_null = (
        "UK_short_obs" in meta_4way and
        abs(meta_4way["UK_short_obs"].get("pooled_beta_RE", 0)) < 0.015
    )
    if long_pass and short_null:
        lines.append("**PASS** — UK_long_longitudinal replicates the Brown-style exposure-gate effect, "
                      "while UK_short_observation replicates the Manchester-style null. "
                      "The Manchester null is part of a continuous observation-density pattern.")
    elif long_pass and not short_null:
        lines.append("UK_long_longitudinal replicates the effect, but UK_short_observation does not "
                      "fully replicate the Manchester null. Manchester remains a partial outlier.")
    else:
        lines.append("**FAIL or AMBIGUOUS** — the long/short stratification does not cleanly account "
                      "for the Manchester null.")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
