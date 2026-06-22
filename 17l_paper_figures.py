"""
17l_paper_figures.py
====================
IMT Attention Bias Paper 2 — Step 17l: Paper-ready figures and final meta.

Builds the publication-quality figures that consolidate the 17b-17k
findings into a single visual stack.

Sections:
  A. Drop-Manchester sensitivity meta  (n = 20 children) at N=5 and N=10,
     including Egger small-study test.
  B. Forest plot — 2-panel (N=5 and N=10), sub-corpus colored, sorted
     within sub-corpus by β. Zero line + grand-pooled summary diamond.
  C. Funnel plot — β vs SE, sub-corpus colored, Egger regression line
     overlaid. p-value reported in the title.
  D. Theakston signature radar — 5 standardised axes
     (1) stimulus_standardization  = 1 - (unique_activities / total_files)
     (2) structured_play_pct        = % files with "Structured Play" or
                                       "playing with toys" @Situation
     (3) pct_INV                    = investigator-speech share
     (4) mot_to_chi_ratio           = mean MOT / mean CHI speech share
     (5) density_utt_per_min        = mean utterances per minute
     each axis 0–1 after min-max across the 4 sub-corpora.

Inputs
------
* output/v17c/per_child_betas_N5.csv
* output/v17c/per_child_betas_N10.csv
* output/v17d/per_child_with_window_N5.csv       (for obs_window_months)
* output/v17h/uk_subcorpora_per_child_N5.csv     (for subcorpus labels)
* output/v17h_N10/uk_subcorpora_per_child_N10.csv
* output/v17j/na_pool_per_child_betas_loose.csv  (April)
* output/v17k/protocol_features_per_file.csv     (for radar axes)

Outputs (output/v17l/)
----------------------
* forest_plot_N5_vs_N10.png
* funnel_plot_N5.png
* funnel_plot_N10.png
* theakston_radar.png
* drop_manchester_meta.json
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / figures v1 | 2026-06-21
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


# Sub-corpus palette (consistent across plots)
PALETTE = {
    "Brown":         "#d62728",   # red
    "Manchester":    "#1f77b4",   # blue
    "UK_long_long":  "#2ca02c",   # green
    "UK_short_obs":  "#ff7f0e",   # orange
    "NA_other":      "#9467bd",   # purple (April)
}

# Sub-corpus order for plotting
SUBCORPUS_ORDER = ["Brown", "Manchester", "UK_long_long", "UK_short_obs", "NA_other"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def egger_test(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    """
    Standard Egger regression for small-study/publication bias:
      (beta / SE)  =  intercept  +  slope * (1 / SE)
    A sig intercept indicates funnel asymmetry.
    """
    if len(betas) < 3:
        return {"error": "n < 3"}
    snd  = betas / ses
    prec = 1.0 / ses
    X = sm.add_constant(prec)
    fit = sm.OLS(snd, X).fit()
    return {
        "intercept":         float(fit.params[0]),
        "intercept_se":      float(fit.bse[0]),
        "intercept_t":       float(fit.tvalues[0]),
        "intercept_p":       float(fit.pvalues[0]),
        "slope":             float(fit.params[1]),
        "slope_se":          float(fit.bse[1]),
        "slope_p":           float(fit.pvalues[1]),
        "n":                 int(fit.nobs),
    }


def _norm_subcorpus(s: str) -> str:
    if pd.isna(s):
        return "?"
    s = str(s)
    if "long_longitudinal" in s or s == "UK_long_long": return "UK_long_long"
    if "short_observation" in s or s == "UK_short_obs": return "UK_short_obs"
    if "NA_other" in s or "April" in s:                  return "NA_other"
    if s == "Manchester (also in UK pool)":             return "Manchester"
    return s


def load_per_child_table(window: int) -> pd.DataFrame:
    """Build the merged per-child table for a given outcome window N."""
    uk_path = Path(f"./output/v17h{'_N10' if window == 10 else ''}/"
                    f"uk_subcorpora_per_child_N{window}.csv")
    if not uk_path.exists():
        sys.exit(f"  ERROR: missing {uk_path}")
    uk = pd.read_csv(uk_path)
    # Drop "(also in UK pool)" duplicates (10 Manchester children)
    uk = uk[~uk["subcorpus"].astype(str).str.contains("also in UK pool", na=False)].copy()
    uk["subcorpus_norm"] = uk["subcorpus"].apply(_norm_subcorpus)

    # Add April (only N=5 is computed; reuse same β for N=10 if missing)
    april_path = Path("./output/v17j/na_pool_per_child_betas_loose.csv")
    if april_path.exists():
        april = pd.read_csv(april_path)
        april["subcorpus_norm"] = "NA_other"
        april["subcorpus"] = "NA_other_longitudinal (April)"
        april["corpus_label"] = "NA-Pool-April"
        april = april.rename(columns={"obs_window": "obs_window_months"})
        keep = ["child", "n_episodes", "obs_window_months",
                "beta_COI_x_cum", "se_COI_x_cum", "p_COI_x_cum",
                "subcorpus_norm", "subcorpus", "corpus_label"]
        april = april[[c for c in keep if c in april.columns]]
        uk = pd.concat([uk[[c for c in keep if c in uk.columns]], april], ignore_index=True)

    # Add window info (from 17d's N=5 table; obs_window is a child property, N-invariant)
    w_path = Path("./output/v17d/per_child_with_window_N5.csv")
    if w_path.exists() and "obs_window_months" not in uk.columns:
        win = pd.read_csv(w_path)[["child", "obs_window_months"]]
        uk = uk.merge(win, on="child", how="left")
    elif w_path.exists() and uk["obs_window_months"].isna().any():
        win = pd.read_csv(w_path)[["child", "obs_window_months"]]
        uk = uk.merge(win, on="child", how="left", suffixes=("", "_d"))
        uk["obs_window_months"] = uk["obs_window_months"].fillna(uk.get("obs_window_months_d"))
    return uk


# ─────────────────────────────────────────────────────────────────────────────
# Section A — drop-Manchester meta + Egger
# ─────────────────────────────────────────────────────────────────────────────

def drop_manchester_meta(df: pd.DataFrame) -> Dict[str, Any]:
    full = df.dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"]).copy()
    full_meta = random_effects_meta(
        full["beta_COI_x_cum"].values.astype(float),
        full["se_COI_x_cum"].values.astype(float),
    )
    full_egger = egger_test(
        full["beta_COI_x_cum"].values.astype(float),
        full["se_COI_x_cum"].values.astype(float),
    )
    drop = full[full["subcorpus_norm"] != "Manchester"].copy()
    drop_meta = random_effects_meta(
        drop["beta_COI_x_cum"].values.astype(float),
        drop["se_COI_x_cum"].values.astype(float),
    )
    drop_egger = egger_test(
        drop["beta_COI_x_cum"].values.astype(float),
        drop["se_COI_x_cum"].values.astype(float),
    )
    return {
        "full_n":          int(len(full)),
        "full_meta":       full_meta,
        "full_egger":      full_egger,
        "drop_manc_n":     int(len(drop)),
        "drop_manc_meta":  drop_meta,
        "drop_manc_egger": drop_egger,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section B — Forest plot (2-panel)
# ─────────────────────────────────────────────────────────────────────────────

def forest_plot(df_n5: pd.DataFrame, df_n10: pd.DataFrame, output_path: Path,
                  meta_n5: Dict[str, Any], meta_n10: Dict[str, Any]) -> None:

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        # Sort within sub-corpus by β descending; sub-corpus order from SUBCORPUS_ORDER
        df = df.copy()
        df["sub_rank"] = df["subcorpus_norm"].map({s: i for i, s in enumerate(SUBCORPUS_ORDER)})
        df = df.sort_values(["sub_rank", "beta_COI_x_cum"], ascending=[True, False]).reset_index(drop=True)
        df["y"] = np.arange(len(df))
        return df

    a5  = _prep(df_n5)
    a10 = _prep(df_n10)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(7, 0.32 * len(a5))),
        sharey=True, gridspec_kw={"wspace": 0.08},
    )

    for ax, df_panel, meta, n_lab in [
        (axes[0], a5,  meta_n5["full_meta"],  "N = 5"),
        (axes[1], a10, meta_n10["full_meta"], "N = 10"),
    ]:
        for _, r in df_panel.iterrows():
            color = PALETTE.get(r["subcorpus_norm"], "gray")
            ax.errorbar(
                r["beta_COI_x_cum"], r["y"],
                xerr=1.96 * r["se_COI_x_cum"],
                fmt="o", color=color, ecolor=color,
                ms=5, capsize=2, alpha=0.85,
            )
            ax.text(-0.30, r["y"], str(r["child"])[:14], fontsize=7, va="center",
                     transform=ax.get_yaxis_transform())
        ax.axvline(0, color="black", lw=0.6, ls="--")
        # Grand pooled diamond
        if "pooled_beta_RE" in meta:
            mb = meta["pooled_beta_RE"]
            ms = meta["pooled_se_RE"]
            ax.scatter([mb], [len(df_panel) + 0.5], marker="D", s=100, color="black", zorder=5)
            ax.errorbar(
                [mb], [len(df_panel) + 0.5],
                xerr=1.96 * ms, fmt="none", color="black",
                ecolor="black", capsize=4, lw=2,
            )
            ax.text(
                mb, len(df_panel) + 1.4,
                f"pooled β = {mb:+.4f}\np = {meta['pooled_p_RE']:.4f}, I²={meta['I2_pct']:.0f}%",
                ha="center", fontsize=9,
            )
        ax.set_xlim(-0.20, 0.20)
        ax.set_ylim(-1, len(df_panel) + 2.5)
        ax.set_xlabel(f"β(COI × cumulative_cue_attempts)  [outcome window {n_lab}]", fontsize=10)
        ax.set_title(n_lab, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
        ax.set_yticks([])

    # Legend (shared)
    handles = [Patch(facecolor=PALETTE[k], label=k) for k in SUBCORPUS_ORDER
                if k in df_n5["subcorpus_norm"].unique() or k in df_n10["subcorpus_norm"].unique()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
                bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Forest plot — per-child β(COI × cumulative_cue_attempts)",
                  fontsize=12, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section C — Funnel plot
# ─────────────────────────────────────────────────────────────────────────────

def funnel_plot(df: pd.DataFrame, output_path: Path, label: str,
                  meta: Dict[str, Any], egger: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for sub in SUBCORPUS_ORDER:
        g = df[df["subcorpus_norm"] == sub]
        if g.empty:
            continue
        ax.scatter(
            g["beta_COI_x_cum"], g["se_COI_x_cum"],
            color=PALETTE.get(sub, "gray"), label=sub, alpha=0.85,
            edgecolors="black", linewidths=0.4, s=55,
        )

    # Symmetric funnel around pooled β
    if "pooled_beta_RE" in meta:
        mb = meta["pooled_beta_RE"]
        max_se = df["se_COI_x_cum"].max() * 1.05
        se_range = np.linspace(0, max_se, 100)
        ax.plot(mb - 1.96 * se_range, se_range, color="gray", lw=0.8, ls="--")
        ax.plot(mb + 1.96 * se_range, se_range, color="gray", lw=0.8, ls="--")
        ax.axvline(mb, color="black", lw=0.7, ls="-", label=f"pooled β = {mb:+.4f}")

    ax.invert_yaxis()
    ax.set_xlabel(f"β(COI × cumulative_cue_attempts)  [{label}]", fontsize=10)
    ax.set_ylabel("SE", fontsize=10)
    title = f"Funnel plot ({label})  |  Egger intercept = {egger.get('intercept', float('nan')):+.4f}  "
    title += f"(SE {egger.get('intercept_se', float('nan')):.4f}, p = {egger.get('intercept_p', float('nan')):.4f})"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section D — Theakston radar
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_TOKENS = {"structured play", "sructured play", "playing with toys"}


def compute_radar_axes() -> pd.DataFrame:
    path = Path("./output/v17k/protocol_features_per_file.csv")
    if not path.exists():
        sys.exit(f"  ERROR: missing {path}")
    df = pd.read_csv(path)
    out_rows: List[Dict[str, Any]] = []
    for sub, g in df.groupby("subcorpus"):
        n_files = len(g)
        unique_acts = g["Activities"].fillna("").astype(str).str.strip().replace("", np.nan).dropna().nunique()
        sit_lower = g["Situation"].fillna("").astype(str).str.lower().str.strip()
        structured_pct = float(sit_lower.isin(STRUCTURED_TOKENS).mean())
        pct_inv = float(g["pct_INV"].mean())
        # MOT / CHI ratio
        mot = g["pct_MOT"].mean()
        chi = g["pct_CHI"].mean()
        mot_chi_ratio = float(mot / chi) if chi > 0 else float("nan")
        # density utt/min
        dur = g["duration_minutes"].dropna()
        density = float((g["n_utt"] / g["duration_minutes"]).replace([np.inf, -np.inf], np.nan).dropna().mean()) \
                  if (g["duration_minutes"].dropna() > 0).any() else float("nan")
        # stimulus standardization: 1 - (unique_activities / n_files) where activity field non-empty
        n_with_act = (g["Activities"].fillna("").astype(str).str.strip() != "").sum()
        std = 1.0 - (unique_acts / n_with_act) if n_with_act > 0 else float("nan")
        out_rows.append({
            "subcorpus":              sub,
            "n_files":                n_files,
            "stimulus_standardization": std,
            "structured_play_pct":    structured_pct,
            "pct_INV":                pct_inv,
            "mot_to_chi_ratio":       mot_chi_ratio,
            "density_utt_per_min":    density,
        })
    return pd.DataFrame(out_rows)


def radar_plot(df: pd.DataFrame, output_path: Path) -> None:
    axes_metrics = [
        "stimulus_standardization", "structured_play_pct",
        "pct_INV", "mot_to_chi_ratio", "density_utt_per_min",
    ]
    nice = {
        "stimulus_standardization": "Stimulus\nstandardisation",
        "structured_play_pct":      "Structured-play\n% of sessions",
        "pct_INV":                  "Investigator\nspeech share",
        "mot_to_chi_ratio":         "MOT / CHI\nspeech ratio",
        "density_utt_per_min":      "Utterance density\n(utt/min)",
    }
    # Min-max normalize each axis across sub-corpora
    norm: Dict[str, np.ndarray] = {}
    for m in axes_metrics:
        col = df[m].astype(float).copy()
        # Some metrics may have NaN — replace with column min for normalization
        col = col.fillna(col.min())
        lo, hi = col.min(), col.max()
        norm[m] = (col - lo) / (hi - lo) if hi > lo else col * 0.0

    angles = np.linspace(0, 2 * np.pi, len(axes_metrics), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    for i, sub in enumerate(df["subcorpus"].tolist()):
        vals = [norm[m].iloc[i] for m in axes_metrics] + [norm[axes_metrics[0]].iloc[i]]
        color = PALETTE.get(sub, "gray")
        ax.plot(angles_closed, vals, color=color, lw=2.0, label=sub)
        ax.fill(angles_closed, vals, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles), [nice[m] for m in axes_metrics], fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.grid(alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    ax.set_title("Theakston-Manchester protocol signature\n"
                  "(min-max normalised within 4 sub-corpora)",
                  fontsize=11, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready figures and final meta.")
    parser.add_argument("--output_dir", default="./output/v17l")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-child tables ...")
    df_n5  = load_per_child_table(5)
    df_n10 = load_per_child_table(10)
    print(f"  N=5  table: {len(df_n5)} children")
    print(f"  N=10 table: {len(df_n10)} children")

    # ───── A) Drop-Manchester meta + Egger ─────
    print("\n=== Section A: drop-Manchester sensitivity ===")
    meta_n5  = drop_manchester_meta(df_n5)
    meta_n10 = drop_manchester_meta(df_n10)
    for N, blk in [(5, meta_n5), (10, meta_n10)]:
        full_m, full_e = blk["full_meta"], blk["full_egger"]
        drop_m, drop_e = blk["drop_manc_meta"], blk["drop_manc_egger"]
        print(f"\n  --- N = {N} ---")
        print(f"  FULL (n={blk['full_n']}):       "
              f"β = {full_m['pooled_beta_RE']:+.4f}  "
              f"SE = {full_m['pooled_se_RE']:.4f}  p = {full_m['pooled_p_RE']:.4f}  "
              f"τ² = {full_m['tau2']:.4f}  I² = {full_m['I2_pct']:.1f}%")
        if "intercept" in full_e:
            print(f"   Egger intercept = {full_e['intercept']:+.4f}  (SE {full_e['intercept_se']:.4f}, p = {full_e['intercept_p']:.4f})")
        print(f"  DROP-MANC (n={blk['drop_manc_n']}): "
              f"β = {drop_m['pooled_beta_RE']:+.4f}  "
              f"SE = {drop_m['pooled_se_RE']:.4f}  p = {drop_m['pooled_p_RE']:.4f}  "
              f"τ² = {drop_m['tau2']:.4f}  I² = {drop_m['I2_pct']:.1f}%")
        if "intercept" in drop_e:
            print(f"   Egger intercept = {drop_e['intercept']:+.4f}  (SE {drop_e['intercept_se']:.4f}, p = {drop_e['intercept_p']:.4f})")

    drop_path = out_dir / "drop_manchester_meta.json"
    with open(drop_path, "w", encoding="utf-8") as f:
        json.dump({"N5": meta_n5, "N10": meta_n10}, f, indent=2, default=str)
    print(f"\n  → {drop_path}")

    # ───── B) Forest plot ─────
    print("\n=== Section B: forest plot ===")
    forest_plot(df_n5, df_n10, out_dir / "forest_plot_N5_vs_N10.png", meta_n5, meta_n10)

    # ───── C) Funnel plots ─────
    print("\n=== Section C: funnel plots ===")
    funnel_plot(df_n5, out_dir / "funnel_plot_N5.png",  label="N = 5",
                  meta=meta_n5["full_meta"],  egger=meta_n5["full_egger"])
    funnel_plot(df_n10, out_dir / "funnel_plot_N10.png", label="N = 10",
                  meta=meta_n10["full_meta"], egger=meta_n10["full_egger"])

    # ───── D) Theakston radar ─────
    print("\n=== Section D: Theakston radar ===")
    radar_df = compute_radar_axes()
    print("  Radar axis values (raw):")
    print(radar_df.to_string(index=False))
    radar_plot(radar_df, out_dir / "theakston_radar.png")
    radar_df.to_csv(out_dir / "theakston_radar_data.csv", index=False)

    # ───── SUMMARY.md ─────
    lines: List[str] = []
    lines.append("# 17l Paper-ready figures + drop-Manchester sensitivity\n")
    lines.append("\n## Drop-Manchester meta + Egger\n")
    lines.append("| Window | Scope | n | β | SE | p | τ² | I² | Egger intercept | Egger p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for N, blk in [(5, meta_n5), (10, meta_n10)]:
        for scope, mkey, ekey, nkey in [
            ("FULL", "full_meta", "full_egger", "full_n"),
            ("DROP-MANC", "drop_manc_meta", "drop_manc_egger", "drop_manc_n"),
        ]:
            m = blk[mkey]; e = blk[ekey]
            lines.append(
                f"| N={N} | {scope} | {blk[nkey]} | "
                f"{m.get('pooled_beta_RE', float('nan')):+.4f} | "
                f"{m.get('pooled_se_RE', float('nan')):.4f} | "
                f"{m.get('pooled_p_RE', float('nan')):.4f} | "
                f"{m.get('tau2', float('nan')):.4f} | "
                f"{m.get('I2_pct', float('nan')):.1f}% | "
                f"{e.get('intercept', float('nan')):+.4f} | "
                f"{e.get('intercept_p', float('nan')):.4f} |"
            )
    lines.append("\n## Theakston radar — raw axis values (post min-max normalised in figure)\n")
    lines.append("| Sub-corpus | n_files | stimulus_standardisation | structured_play_pct | pct_INV | MOT/CHI ratio | density_utt_per_min |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in radar_df.iterrows():
        lines.append(
            f"| {r['subcorpus']} | {int(r['n_files'])} | "
            f"{r['stimulus_standardization']:.3f} | "
            f"{r['structured_play_pct']:.3f} | "
            f"{r['pct_INV']:.4f} | "
            f"{r['mot_to_chi_ratio']:.3f} | "
            f"{r['density_utt_per_min']:.2f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
