"""
04_visualize_results_v3.py
==========================
IMT Attention Bias Paper — Step 4 (v3): Publication Figures

Rebuilt for the 8-sample, v3 pipeline (non-circular Reliability; English-UK
dialect pair added; meta-analysis outputs from 06).

v3 changes vs. the original 04:
  - 8 samples incl. English-UK (was 7).
  - Reads v3 column schema:
        AttentionIndex, WeightedAttentionIndex,
        Reliability_gra, Reliability_position, Reliability_form,
        count, cue_subtype, cue_type, dominant_gra_relation
    (the old S_frequency_normalized / S_perceptual columns are gone).
  - NEW main-result figures driven by the meta-analysis:
        Fig1  Forest plot of per-sample interaction β + pooled diamond
        Fig2  Two-phase dissociation (peak vs first-emergence)
        Fig3  Variance decomposition (ΔR² interaction vs AI-alone)
    These are the figures that actually carry the paper's claim.
  - Retained, schema-fixed descriptive figures:
        Fig4  Cue position vs AI (edge-salience), one panel per sample
        Fig5  AI distribution across samples

Inputs (all in --output_dir, i.e. output/v3/):
    {sample}_attention_index.csv         (from 03)
    {sample}_uptake_summary.json         (from 05)
    meta_effects_peak_rate_per_1k.csv    (from 06)
    meta_effects_first_emergence_month.csv
    meta_analysis_summary.json

Usage:
    python 04_visualize_results_v3.py --output_dir ./output/v3/

Author: Torami x Boss | IMT Attention project | 2026-06-14
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Polygon
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy pandas matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 8 samples in display order. English-UK sits next to English (US) so the
# dialect pair reads as a replication at a glance.
SAMPLES = ["English", "English-UK", "Spanish", "Japanese", "Korean",
           "Mandarin", "Russian", "Indonesian"]

# Pretty labels for axes/legends
SAMPLE_LABEL = {
    "English":    "English (US)",
    "English-UK": "English (UK)",
    "Spanish":    "Spanish",
    "Japanese":   "Japanese",
    "Korean":     "Korean",
    "Mandarin":   "Mandarin",
    "Russian":    "Russian",
    "Indonesian": "Indonesian",
}

# Typology-aware palette
SAMPLE_COLORS = {
    "English":    "#1f77b4",   # IE Germanic (US)
    "English-UK": "#4a98d6",   # IE Germanic (UK) — lighter, same hue family
    "Spanish":    "#17becf",   # IE Romance
    "Russian":    "#9467bd",   # IE Slavic
    "Japanese":   "#d62728",   # Japonic
    "Korean":     "#ff7f0e",   # Koreanic
    "Mandarin":   "#2ca02c",   # Sino-Tibetan
    "Indonesian": "#bcbd22",   # Austronesian
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_meta_effects(out_dir: Path, outcome: str) -> Optional[pd.DataFrame]:
    """Load per-sample interaction effects produced by 06_meta_analysis.py."""
    path = out_dir / f"meta_effects_{outcome}.csv"
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    df = pd.read_csv(path)
    # Order rows by our display order where possible
    df["__order"] = df["language"].map({s: i for i, s in enumerate(SAMPLES)})
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return df


def load_meta_summary(out_dir: Path) -> Optional[dict]:
    path = out_dir / "meta_analysis_summary.json"
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_all_attention(out_dir: Path) -> pd.DataFrame:
    frames = []
    for s in SAMPLES:
        csv_path = out_dir / f"{s}_attention_index.csv"
        if not csv_path.exists():
            print(f"  WARN: missing {csv_path}, skipping {s}")
            continue
        df = pd.read_csv(csv_path)
        df["sample"] = s
        frames.append(df)
        print(f"  Loaded {SAMPLE_LABEL[s]}: {len(df)} cue subtypes")
    if not frames:
        print("ERROR: no attention_index CSVs found.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def load_uptake_summaries(out_dir: Path) -> Dict[str, dict]:
    """Load each {sample}_uptake_summary.json (for ΔR² decomposition figure)."""
    out = {}
    for s in SAMPLES:
        path = out_dir / f"{s}_uptake_summary.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path) as f:
            out[s] = json.load(f)
    return out


def _pooled_from_summary(meta: dict, outcome: str) -> Optional[dict]:
    """Extract pooled stats robustly from the meta summary JSON.

    06 writes a nested dict keyed by outcome. We defend against minor key
    naming differences so the figure code does not break if 06 evolves.
    """
    if meta is None:
        return None
    block = None
    if outcome in meta:
        block = meta[outcome]
    else:
        # search one level down
        for v in meta.values():
            if isinstance(v, dict) and v.get("outcome") == outcome:
                block = v
                break
    if block is None:
        return None

    def g(*names, default=None):
        for n in names:
            if n in block:
                return block[n]
        return default

    return {
        "beta": g("pooled_beta", "pooled_interaction_beta", "beta"),
        "ci_low": g("ci_low", "ci_lower", "pooled_ci_low"),
        "ci_high": g("ci_high", "ci_upper", "pooled_ci_high"),
        "z": g("z", "z_value"),
        "p": g("p", "p_value"),
        "I2": g("I2", "i_squared", "I_squared"),
        "Q": g("Q", "q_stat"),
        "Q_p": g("Q_p", "q_p"),
        "sign_pos": g("sign_test_positive", "sign_positive"),
        "sign_n": g("sign_test_n", "n_samples"),
        "sign_p": g("sign_test_p", "binomial_p"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Forest plot (THE main result)
# ─────────────────────────────────────────────────────────────────────────────

def fig_forest(meta_df: pd.DataFrame, pooled: Optional[dict],
               out_path: Path, outcome_label: str = "peak production"):
    """Classic meta-analysis forest plot: per-sample β ± 95% CI + pooled diamond."""
    df = meta_df.copy()
    beta = df["interaction_beta"].values
    se = df["interaction_se"].values
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    n_cues = df["n_cues"].values
    labels = [SAMPLE_LABEL.get(l, l) for l in df["language"]]

    # weight ∝ inverse variance → marker size
    w = 1.0 / (se ** 2)
    w = w / w.max()
    sizes = 60 + w * 240

    n = len(df)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * n + 2.2))

    y = np.arange(n)[::-1]   # top sample at top

    for i in range(n):
        lang = df["language"].iloc[i]
        color = SAMPLE_COLORS.get(lang, "#444444")
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=color, lw=2, alpha=0.8,
                solid_capstyle="round")
        ax.scatter(beta[i], y[i], s=sizes[i], color=color, edgecolor="black",
                   linewidth=0.7, zorder=3)
        # annotate β and n on the right margin
        ax.text(1.02, y[i], f"{beta[i]:+.2f}  [{ci_lo[i]:+.2f}, {ci_hi[i]:+.2f}]   n={int(n_cues[i])}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)

    # Pooled diamond
    if pooled and pooled.get("beta") is not None:
        pb = pooled["beta"]
        plo = pooled.get("ci_low", pb)
        phi = pooled.get("ci_high", pb)
        ydia = -1.2
        diamond = Polygon(
            [(plo, ydia), (pb, ydia + 0.35), (phi, ydia), (pb, ydia - 0.35)],
            closed=True, facecolor="#222222", edgecolor="black", zorder=4
        )
        ax.add_patch(diamond)
        ztxt = f"z={pooled['z']:.1f}" if pooled.get("z") is not None else ""
        i2txt = f"I²={pooled['I2']:.1f}%" if pooled.get("I2") is not None else ""
        sgn = ""
        if pooled.get("sign_pos") is not None and pooled.get("sign_n") is not None:
            sgn = f"sign {int(pooled['sign_pos'])}/{int(pooled['sign_n'])}"
        meta_line = "  ".join(t for t in [ztxt, i2txt, sgn] if t)
        ax.text(1.02, ydia, f"Pooled {pb:+.2f}  [{plo:+.2f}, {phi:+.2f}]",
                transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, fontweight="bold")
        if meta_line:
            ax.text(1.02, ydia - 0.9, meta_line,
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                    color="#333333")

    ax.axvline(0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-2.2, n - 0.3)
    ax.set_xlabel(f"AI × frequency interaction β  (z-standardized) on {outcome_label}")
    ax.set_title(f"Cross-linguistic forest plot — interaction effect on {outcome_label}",
                 fontsize=12)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Two-phase dissociation (peak vs first-emergence)
# ─────────────────────────────────────────────────────────────────────────────

def fig_two_phase(peak_df: pd.DataFrame, emrg_df: pd.DataFrame,
                  pooled_peak: Optional[dict], pooled_emrg: Optional[dict],
                  out_path: Path):
    """Paired dot plot: each sample's interaction β for peak vs emergence,
    with pooled estimates highlighted. The core RSM signature in one figure."""
    # align on language
    merged = peak_df[["language", "interaction_beta", "interaction_se"]].merge(
        emrg_df[["language", "interaction_beta", "interaction_se"]],
        on="language", suffixes=("_peak", "_emrg")
    )
    merged["__order"] = merged["language"].map({s: i for i, s in enumerate(SAMPLES)})
    merged = merged.sort_values("__order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x_peak, x_emrg = 0.0, 1.0

    # De-collide the left-side (peak) labels: sort by peak β and spread label
    # y-positions with a minimum gap so dense clusters stay readable.
    order = merged.sort_values("interaction_beta_peak", ascending=False).reset_index(drop=True)
    ymax = order["interaction_beta_peak"].max()
    ymin = order["interaction_beta_peak"].min()
    span = max(ymax - ymin, 1e-6)
    min_gap = span * 0.085          # minimum vertical gap between stacked labels
    label_y = {}
    prev = None
    for _, row in order.iterrows():       # top → down
        target = row["interaction_beta_peak"]
        if prev is not None and (prev - target) < min_gap:
            target = prev - min_gap        # push down to keep spacing
        label_y[row["language"]] = target
        prev = target

    for _, row in merged.iterrows():
        lang = row["language"]
        color = SAMPLE_COLORS.get(lang, "#444444")
        ax.plot([x_emrg, x_peak],
                [row["interaction_beta_emrg"], row["interaction_beta_peak"]],
                color=color, alpha=0.5, lw=1.3, zorder=1)
        ax.scatter([x_emrg], [row["interaction_beta_emrg"]], s=70, color=color,
                   edgecolor="black", linewidth=0.5, zorder=2)
        ax.scatter([x_peak], [row["interaction_beta_peak"]], s=70, color=color,
                   edgecolor="black", linewidth=0.5, zorder=2)
        # leader line from dot to (possibly shifted) label, then the label
        ly = label_y[lang]
        ax.plot([x_peak - 0.035, x_peak - 0.012], [ly, row["interaction_beta_peak"]],
                color=color, lw=0.6, alpha=0.6, zorder=1)
        ax.text(x_peak - 0.045, ly,
                SAMPLE_LABEL.get(lang, lang), ha="right", va="center", fontsize=7.5)

    # pooled markers
    if pooled_emrg and pooled_emrg.get("beta") is not None:
        ax.scatter([x_emrg], [pooled_emrg["beta"]], s=260, marker="D",
                   color="#222222", edgecolor="white", linewidth=1.2, zorder=5)
    if pooled_peak and pooled_peak.get("beta") is not None:
        ax.scatter([x_peak], [pooled_peak["beta"]], s=260, marker="D",
                   color="#222222", edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(x_peak + 0.06, pooled_peak["beta"],
                f"pooled {pooled_peak['beta']:+.2f}", va="center",
                fontsize=8.5, fontweight="bold")
    if pooled_emrg and pooled_emrg.get("beta") is not None:
        ax.text(x_emrg - 0.06, pooled_emrg["beta"],
                f"pooled {pooled_emrg['beta']:+.2f}", va="center", ha="right",
                fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.set_xticks([x_emrg, x_peak])
    ax.set_xticklabels(["First emergence\n(initial registration)",
                        "Peak production\n(entrenchment)"])
    ax.set_xlim(-0.5, 1.45)
    ax.set_ylabel("AI × frequency interaction β (z-standardized)")
    ax.set_title("Two-phase dissociation: emergence vs entrenchment\n"
                 "(heterogeneous, ~null on emergence → homogeneous, strong on peak)",
                 fontsize=11.5)

    # annotate the contrast with I² if available
    notes = []
    if pooled_emrg and pooled_emrg.get("I2") is not None:
        notes.append(f"emergence I²={pooled_emrg['I2']:.0f}% (heterogeneous)")
    if pooled_peak and pooled_peak.get("I2") is not None:
        notes.append(f"peak I²={pooled_peak['I2']:.0f}% (homogeneous)")
    if notes:
        ax.text(0.5, -0.16, "   |   ".join(notes), transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Variance decomposition (ΔR² interaction vs AI-alone)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_delta_r2(uptake: dict) -> Optional[Dict[str, float]]:
    """Pull ΔR²(AI alone) and ΔR²(interaction) for peak from an uptake summary.

    Defends against schema drift in 05's JSON output.
    """
    block = None
    for key in ("peak_rate_per_1k", "peak", "peak_production"):
        if key in uptake:
            block = uptake[key]
            break
    if block is None:
        # maybe summary is flat
        block = uptake
    da = block.get("deltaR2_AI", block.get("dR2_AI", block.get("delta_r2_ai")))
    di = block.get("deltaR2_interaction",
                   block.get("dR2_interaction", block.get("delta_r2_interaction")))
    if da is None or di is None:
        return None
    return {"ai": float(da), "interaction": float(di)}


def fig_variance_decomp(uptake_summaries: Dict[str, dict],
                        meta_peak: Optional[dict],
                        out_path: Path):
    """Grouped bars: per-sample ΔR² from AI-alone vs from the interaction term."""
    rows = []
    for s in SAMPLES:
        if s not in uptake_summaries:
            continue
        d = _extract_delta_r2(uptake_summaries[s])
        if d is None:
            continue
        rows.append({"sample": s, "ai": d["ai"], "interaction": d["interaction"]})

    if not rows:
        print("  WARN: no ΔR² data found in uptake summaries; skipping Fig3.")
        return

    df = pd.DataFrame(rows)
    n = len(df)
    x = np.arange(n)
    bw = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - bw / 2, df["ai"], bw, label="ΔR²  (AI as main effect)",
           color="#cccccc", edgecolor="black", linewidth=0.5)
    bars_int = ax.bar(x + bw / 2, df["interaction"], bw,
                      label="ΔR²  (AI × frequency interaction)",
                      color=[SAMPLE_COLORS[s] for s in df["sample"]],
                      edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SAMPLE_LABEL[s] for s in df["sample"]], rotation=20, ha="right")
    ax.set_ylabel("ΔR² on peak production")
    ax.set_title("Variance carried by the interaction vs. attention alone")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # ratio annotation from the meta summary if available
    mean_ai = df["ai"].mean()
    mean_int = df["interaction"].mean()
    if mean_ai > 0:
        ratio = mean_int / mean_ai
        ax.text(0.02, 0.95,
                f"mean ΔR²(interaction) = {mean_int:.3f}\n"
                f"mean ΔR²(AI alone)   = {mean_ai:.3f}\n"
                f"→ interaction adds {ratio:.1f}× more variance",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#888888"))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Cue position vs AI (edge salience), one panel per sample
# ─────────────────────────────────────────────────────────────────────────────

def fig_position_vs_ai(df: pd.DataFrame, out_path: Path):
    samples_present = [s for s in SAMPLES if s in df["sample"].values]
    n = len(samples_present)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.1 * n_rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    # position column: prefer pos_mean; fall back gracefully
    pos_col = "pos_mean" if "pos_mean" in df.columns else (
        "Reliability_position" if "Reliability_position" in df.columns else None)

    for i, s in enumerate(samples_present):
        ax = axes[i]
        sub = df[df["sample"] == s]
        xvals = sub[pos_col] if pos_col else np.full(len(sub), np.nan)
        sizes = np.clip(np.log1p(sub["count"]) * 8, 8, 120)
        ax.scatter(xvals, sub["AttentionIndex"], s=sizes,
                   color=SAMPLE_COLORS[s], alpha=0.65,
                   edgecolor="black", linewidth=0.4)
        # label top-4 cues by AI
        for _, row in sub.nlargest(4, "AttentionIndex").iterrows():
            ax.annotate(str(row["cue_subtype"])[:12],
                        (row[pos_col] if pos_col else 0.5, row["AttentionIndex"]),
                        fontsize=6, alpha=0.85,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(SAMPLE_LABEL[s], fontsize=10.5, color=SAMPLE_COLORS[s])
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.text(0.5, 0.00, "Mean position in utterance (0 = initial, 1 = final)",
             ha="center", fontsize=11)
    fig.text(0.005, 0.5, "Attention Index", va="center", rotation="vertical",
             fontsize=11)
    fig.suptitle("Edge salience: cue position vs Attention Index\n"
                 "(point size ∝ log frequency)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — AI distribution across samples (violin + jitter)
# ─────────────────────────────────────────────────────────────────────────────

def fig_ai_distribution(df: pd.DataFrame, out_path: Path):
    samples_present = [s for s in SAMPLES if s in df["sample"].values]

    weighted_data = []
    for s in samples_present:
        sub = df[df["sample"] == s]
        vals = []
        for _, row in sub.iterrows():
            wt = max(1, int(np.log1p(row["count"])))
            vals.extend([row["AttentionIndex"]] * wt)
        weighted_data.append(vals)

    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(weighted_data, showmeans=True, showmedians=False,
                          widths=0.75)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(SAMPLE_COLORS[samples_present[i]])
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    for key in ("cmeans", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1)

    for i, s in enumerate(samples_present):
        sub = df[df["sample"] == s]
        x = np.full(len(sub), i + 1) + np.random.uniform(-0.1, 0.1, len(sub))
        sizes = np.clip(np.log1p(sub["count"]) * 6, 5, 80)
        ax.scatter(x, sub["AttentionIndex"], s=sizes,
                   color=SAMPLE_COLORS[s], alpha=0.5,
                   edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(1, len(samples_present) + 1))
    ax.set_xticklabels([SAMPLE_LABEL[s] for s in samples_present], rotation=20, ha="right")
    ax.set_ylabel("Attention Index (AI)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Attention Index distribution across samples\n"
                 "(violin = log-count-weighted; points = cue subtypes, size ∝ log freq)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate v3 publication figures (8 samples + meta results)."
    )
    parser.add_argument("--output_dir", default="./output/v3/",
                        help="Directory holding v3 CSVs and meta outputs.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading meta-analysis outputs ===")
    peak_df = load_meta_effects(out_dir, "peak_rate_per_1k")
    emrg_df = load_meta_effects(out_dir, "first_emergence_month")
    meta = load_meta_summary(out_dir)
    pooled_peak = _pooled_from_summary(meta, "peak_rate_per_1k")
    pooled_emrg = _pooled_from_summary(meta, "first_emergence_month")

    print("\n=== Loading attention-index CSVs ===")
    ai_df = load_all_attention(out_dir)

    print("\n=== Loading uptake summaries (for ΔR²) ===")
    uptake_summaries = load_uptake_summaries(out_dir)

    print("\n=== Generating MAIN-RESULT figures ===")
    if peak_df is not None:
        fig_forest(peak_df, pooled_peak, fig_dir / "Fig1_forest_peak.png",
                   outcome_label="peak production")
    if peak_df is not None and emrg_df is not None:
        fig_two_phase(peak_df, emrg_df, pooled_peak, pooled_emrg,
                      fig_dir / "Fig2_two_phase_dissociation.png")
    if uptake_summaries:
        fig_variance_decomp(uptake_summaries, pooled_peak,
                            fig_dir / "Fig3_variance_decomposition.png")
    # Bonus: emergence forest, useful for the supplement
    if emrg_df is not None:
        fig_forest(emrg_df, pooled_emrg, fig_dir / "FigS_forest_emergence.png",
                   outcome_label="first emergence")

    print("\n=== Generating DESCRIPTIVE figures ===")
    fig_position_vs_ai(ai_df, fig_dir / "Fig4_position_x_AI.png")
    fig_ai_distribution(ai_df, fig_dir / "Fig5_AI_distribution.png")

    print("\n=== Done ===")
    print(f"All figures in: {fig_dir}")
    print("Main results:  Fig1 (forest), Fig2 (two-phase), Fig3 (ΔR²)")
    print("Descriptive:   Fig4 (edge salience), Fig5 (AI distribution)")
    print("Supplement:    FigS (emergence forest)")


if __name__ == "__main__":
    main()
