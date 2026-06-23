"""
17m_window_range_sensitivity.py
===============================
IMT Attention Bias Paper 2 — Step 17m: Reuse-window-range sensitivity.

Extends the 17l sensitivity story from {N=5, N=10} to the full
{N=3, N=5, N=10, N=20} grid. For each outcome window N we report:

  * Per-child β(COI × cumulative_cue_attempts) at N
  * Random-effects meta — FULL n=32  AND  DROP-MANCHESTER n=21
  * Egger small-study / publication-bias intercept
  * 4-panel forest plot
  * Pooled β trajectory (β vs N for both scopes)

────────────────────────────────────────────────────────────────────────────

Inputs
------
* output/v17c/per_child_betas_N{3,5,10,20}.csv   (produced via 17c batch)
* output/v17j/na_pool_per_child_betas_loose.csv  (April; per-window
                                                    recomputed inline)
* output/v17h/uk_subcorpora_per_child_N5.csv     (subcorpus labels)

Outputs (output/v17m/)
----------------------
* per_child_betas_combined_4windows.csv    long-form table all 4 windows
* four_window_meta.json                    per-window FULL + DROP-MANC meta
* egger_4windows.json                      per-window Egger results
* forest_4panel.png                        N=3 | N=5 | N=10 | N=20
* pooled_beta_trajectory.png               β vs N, Full + Drop-Manc
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / window range v1 | 2026-06-22
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


PALETTE = {
    "Brown":         "#d62728",
    "Manchester":    "#1f77b4",
    "UK_long_long":  "#2ca02c",
    "UK_short_obs":  "#ff7f0e",
    "NA_other":      "#9467bd",
}

SUBCORPUS_ORDER = ["Brown", "Manchester", "UK_long_long", "UK_short_obs", "NA_other"]


def _norm_subcorpus(s: str) -> str:
    if pd.isna(s):
        return "?"
    s = str(s)
    if "long_longitudinal" in s or s == "UK_long_long": return "UK_long_long"
    if "short_observation" in s or s == "UK_short_obs": return "UK_short_obs"
    if "NA_other" in s or "April" in s:                  return "NA_other"
    if s == "Manchester (also in UK pool)":              return "Manchester"
    return s


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
    # 95% prediction interval for a new study (Higgins/Thompson):
    pi_se = float(np.sqrt(tau2 + se_re ** 2))
    pi_lo = beta_re - 1.96 * pi_se if not math.isnan(pi_se) else float("nan")
    pi_hi = beta_re + 1.96 * pi_se if not math.isnan(pi_se) else float("nan")
    return {
        "n_studies": int(n),
        "Q": Q, "df": int(df), "tau2": tau2, "I2_pct": I2,
        "pooled_beta_RE": beta_re, "pooled_se_RE": se_re,
        "pooled_z_RE": z, "pooled_p_RE": p,
        "pred_interval_low": pi_lo, "pred_interval_high": pi_hi,
    }


def egger_test(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    if len(betas) < 3:
        return {"error": "n < 3"}
    snd  = betas / ses
    prec = 1.0 / ses
    X = sm.add_constant(prec)
    fit = sm.OLS(snd, X).fit()
    return {
        "intercept":         float(fit.params[0]),
        "intercept_se":      float(fit.bse[0]),
        "intercept_p":       float(fit.pvalues[0]),
        "slope":             float(fit.params[1]),
        "slope_p":           float(fit.pvalues[1]),
        "n":                 int(fit.nobs),
    }


MANCHESTER_CHILDREN = {
    "Anne", "Aran", "Becky", "Carl", "Dominic", "Gail",
    "Joel", "John", "Liz", "Ruth", "Warren",
}

UK_LONG_LONG_CHILDREN = {"Thomas", "Fraser", "Helen", "Eleanor", "Nicole"}


def load_per_child_table(window: int) -> pd.DataFrame:
    """
    Combine per-child β estimates for a given outcome window, drawing from
    output/v17c/per_child_betas_N{N}.csv (Brown / Manchester / English-UK).

    Deduplication rule (10 Manchester children appear twice in 17c output
    because the English-UK json_cache shares its child sub-dirs with the
    Manchester corpus):
      - Manchester children → keep the row where corpus_label == "Manchester"
      - Brown children       → keep the row where corpus_label == "Brown"
      - All other UK children → keep the row where corpus_label == "English-UK"
    """
    n_path = Path(f"./output/v17c/per_child_betas_N{window}.csv")
    if not n_path.exists():
        sys.exit(f"  ERROR: missing {n_path} (run 17c at --window {window})")
    df = pd.read_csv(n_path)
    df["child"] = df["child"].astype(str)
    df["corpus_label"] = df["corpus_label"].astype(str)

    # Deduplicate: for Manchester children keep only corpus_label == Manchester
    is_manc_child = df["child"].isin(MANCHESTER_CHILDREN)
    keep_mask = (
        (~is_manc_child) |
        (is_manc_child & (df["corpus_label"] == "Manchester"))
    )
    df = df[keep_mask].copy()

    # Assign sub-corpus
    def _sc(row) -> str:
        c = row["child"]
        cl = row["corpus_label"]
        if cl == "Brown":
            return "Brown"
        if cl == "Manchester" or c in MANCHESTER_CHILDREN:
            return "Manchester"
        if c in UK_LONG_LONG_CHILDREN:
            return "UK_long_long"
        if cl == "English-UK":
            return "UK_short_obs"
        return "?"

    df["subcorpus_norm"] = df.apply(_sc, axis=1)
    df = df[df["subcorpus_norm"] != "?"].copy()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# April (NA single child) per-window re-fit using the reuse CSV directly
# ─────────────────────────────────────────────────────────────────────────────

import bisect

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return s - s.mean()
    return (s - s.mean()) / sd


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
                          prior_window: int = 20) -> pd.DataFrame:
    n = len(reuse_df)
    prior = np.zeros(n, dtype=np.int32)
    files = reuse_df["file"].astype(str).values
    utt_idxs = reuse_df["child_utt_idx"].astype(int).values
    cues = reuse_df["cue_subtype"].astype(str).values
    for i in range(n):
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


def refit_april_at_window(window: int) -> Optional[pd.DataFrame]:
    """Re-fit April's per-child OLS using next_{window}_reuse from the
    NA-Pool reuse CSV."""
    reuse_csv = Path("./output/v16/English-NA-Pool_episodes_with_reuse.csv")
    tagged_csv = Path("./output/English-NA-Pool_tokens_tagged.csv")
    joined_csv = Path("./output/v11/English-NA-Pool_r_plus_joined.csv")
    json_cache = Path("./output/json_cache/English-NA-Pool")
    if not all(p.exists() for p in [reuse_csv, tagged_csv, joined_csv, json_cache]):
        return None
    outcome_col = f"next_{window}_reuse"
    reuse = pd.read_csv(reuse_csv, low_memory=False, dtype={"file": str})
    if outcome_col not in reuse.columns:
        return None
    reuse = reuse[reuse["r_plus_label"] != "no_contingent_response"].copy()
    # Build file → child map
    file_to_child: Dict[str, str] = {}
    for child_dir in json_cache.iterdir():
        if child_dir.is_dir():
            for jf in child_dir.rglob("*.json"):
                file_to_child[jf.stem] = child_dir.name
    reuse["child"] = reuse["file"].astype(str).map(file_to_child)
    reuse = reuse.dropna(subset=["child"]).copy()
    # April only
    reuse = reuse[reuse["child"] == "April"].copy()
    if len(reuse) < 100:
        return None
    fi = build_child_utt_index(tagged_csv)
    reuse = add_prior_local_freq(reuse, fi, prior_window=20)
    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    df = reuse.merge(joined[["cue_subtype", "COI", "log_cue_freq"]], on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    df = df.sort_values(["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    df_post = df[df["child_age_months"] >= 24].copy()
    if len(df_post) < 100:
        return None
    df_post["COI_z_local"]     = _z(df_post["COI"])
    df_post["cum_z_local"]     = _z(df_post["cumulative_cue_attempts"])
    df_post["prior_z_local"]   = _z(df_post["prior_local_freq"])
    df_post["logfreq_z_local"] = _z(df_post["log_cue_freq"])
    df_post["COI_x_cum_local"] = df_post["COI_z_local"] * df_post["cum_z_local"]
    preds = ["COI_z_local", "cum_z_local", "COI_x_cum_local", "prior_z_local", "logfreq_z_local"]
    X = sm.add_constant(df_post[preds].astype(float), has_constant="add")
    y = df_post[outcome_col].astype(float)
    try:
        fit = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_post["cue_subtype"].astype(str).values},
        )
    except Exception:
        return None
    return pd.DataFrame([{
        "child":           "April",
        "n_episodes":      int(len(df_post)),
        "n_cues":          int(df_post["cue_subtype"].nunique()),
        "beta_COI_x_cum":  float(fit.params["COI_x_cum_local"]),
        "se_COI_x_cum":    float(fit.bse["COI_x_cum_local"]),
        "p_COI_x_cum":     float(fit.pvalues["COI_x_cum_local"]),
        "obs_window":      float(df_post["child_age_months"].max() - df_post["child_age_months"].min()),
        "corpus_label":    "NA-Pool-April",
        "subcorpus_norm":  "NA_other",
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse window-range sensitivity (N=3,5,10,20).")
    parser.add_argument("--output_dir", default="./output/v17m")
    parser.add_argument("--windows", default="3,5,10,20")
    args = parser.parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",")]

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-child β at each window, append April
    per_window: Dict[int, pd.DataFrame] = {}
    for N in windows:
        print(f"\n=== Loading window N = {N} ===")
        df = load_per_child_table(N)
        # Append April
        april = refit_april_at_window(N)
        if april is not None and not april.empty:
            df = pd.concat([df, april], ignore_index=True, sort=False)
            print(f"  Appended April: β={april['beta_COI_x_cum'].iloc[0]:+.4f}  SE={april['se_COI_x_cum'].iloc[0]:.4f}")
        print(f"  n_children at N={N}: {len(df)}")
        per_window[N] = df

    # Combined long-form CSV
    combined_rows: List[Dict[str, Any]] = []
    for N, df in per_window.items():
        for _, r in df.iterrows():
            combined_rows.append({
                "window":          N,
                "child":           r["child"],
                "subcorpus_norm":  r.get("subcorpus_norm", "?"),
                "n_episodes":      int(r.get("n_episodes", 0)),
                "obs_window":      float(r.get("obs_window", float("nan"))),
                "beta":            float(r["beta_COI_x_cum"]),
                "se":              float(r["se_COI_x_cum"]),
                "p":               float(r["p_COI_x_cum"]),
            })
    combined_df = pd.DataFrame(combined_rows)
    combined_csv = out_dir / "per_child_betas_combined_4windows.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n  → {combined_csv}  ({len(combined_df):,} rows)")

    # Per-window meta + Egger
    meta_results: Dict[str, Dict[str, Any]] = {}
    egger_results: Dict[str, Dict[str, Any]] = {}
    print("\n=== Per-window meta + Egger (FULL n=32, DROP-MANC n=21) ===")
    print(f"  {'N':>2} {'scope':<14} {'n':>3} {'β':>9} {'SE':>7} {'p':>7} {'τ²':>7} {'I²':>6} {'95% PI':>20} | Egger int {'p':>6}")
    print("  " + "-" * 110)
    for N in windows:
        df = per_window[N].dropna(subset=["beta_COI_x_cum", "se_COI_x_cum"])
        full = random_effects_meta(df["beta_COI_x_cum"].values.astype(float),
                                     df["se_COI_x_cum"].values.astype(float))
        full_egger = egger_test(df["beta_COI_x_cum"].values.astype(float),
                                  df["se_COI_x_cum"].values.astype(float))
        drop = df[df["subcorpus_norm"] != "Manchester"].copy()
        drop_meta = random_effects_meta(drop["beta_COI_x_cum"].values.astype(float),
                                          drop["se_COI_x_cum"].values.astype(float))
        drop_egger = egger_test(drop["beta_COI_x_cum"].values.astype(float),
                                  drop["se_COI_x_cum"].values.astype(float))
        meta_results[str(N)]  = {"FULL": full, "DROP_MANC": drop_meta}
        egger_results[str(N)] = {"FULL": full_egger, "DROP_MANC": drop_egger}
        for label, m, e in [("FULL", full, full_egger), ("DROP-MANC", drop_meta, drop_egger)]:
            sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
            pi = f"[{m.get('pred_interval_low', float('nan')):+.3f}, {m.get('pred_interval_high', float('nan')):+.3f}]"
            ei = e.get('intercept', float('nan'))
            ep = e.get('intercept_p', float('nan'))
            print(f"  {N:>2} {label:<14} {m['n_studies']:>3} {m['pooled_beta_RE']:>+8.4f} {m['pooled_se_RE']:.4f} "
                  f"{m['pooled_p_RE']:.4f}{sig:<3} {m['tau2']:.4f} {m['I2_pct']:>5.1f}% {pi:>20} | {ei:>+7.3f} {ep:>6.3f}")

    with open(out_dir / "four_window_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_results, f, indent=2, default=str)
    with open(out_dir / "egger_4windows.json", "w", encoding="utf-8") as f:
        json.dump(egger_results, f, indent=2, default=str)
    print(f"\n  → {out_dir / 'four_window_meta.json'}")
    print(f"  → {out_dir / 'egger_4windows.json'}")

    # ───── 4-panel forest plot ─────
    print("\n=== Building 4-panel forest plot ===")
    fig, axes = plt.subplots(1, 4, figsize=(18, max(8, 0.32 * len(per_window[windows[0]]))),
                              sharey=True, gridspec_kw={"wspace": 0.06})
    for ax_idx, N in enumerate(windows):
        ax = axes[ax_idx]
        df_panel = per_window[N].copy()
        # Sort within sub-corpus by β descending
        df_panel["sub_rank"] = df_panel["subcorpus_norm"].map({s: i for i, s in enumerate(SUBCORPUS_ORDER)})
        df_panel = df_panel.sort_values(["sub_rank", "beta_COI_x_cum"], ascending=[True, False]).reset_index(drop=True)
        df_panel["y"] = np.arange(len(df_panel))
        for _, r in df_panel.iterrows():
            color = PALETTE.get(r["subcorpus_norm"], "gray")
            ax.errorbar(r["beta_COI_x_cum"], r["y"], xerr=1.96 * r["se_COI_x_cum"],
                         fmt="o", color=color, ecolor=color, ms=4, capsize=2, alpha=0.85)
            if ax_idx == 0:
                ax.text(-0.35, r["y"], str(r["child"])[:14], fontsize=6, va="center",
                         transform=ax.get_yaxis_transform())
        ax.axvline(0, color="black", lw=0.6, ls="--")
        # Pool diamond at bottom
        full = meta_results[str(N)]["FULL"]
        if "pooled_beta_RE" in full:
            mb, ms = full["pooled_beta_RE"], full["pooled_se_RE"]
            ax.scatter([mb], [len(df_panel) + 0.5], marker="D", s=80, color="black", zorder=5)
            ax.errorbar([mb], [len(df_panel) + 0.5], xerr=1.96 * ms, fmt="none",
                         color="black", ecolor="black", capsize=4, lw=2)
            ax.text(mb, len(df_panel) + 1.4,
                     f"β = {mb:+.4f}\np = {full['pooled_p_RE']:.4f}, I²={full['I2_pct']:.0f}%",
                     ha="center", fontsize=8)
        ax.set_xlim(-0.20, 0.20)
        ax.set_ylim(-1, len(df_panel) + 2.5)
        ax.set_xlabel(f"β  (next-{N})", fontsize=10)
        ax.set_title(f"N = {N}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
        ax.set_yticks([])
    handles = [Patch(facecolor=PALETTE[k], label=k) for k in SUBCORPUS_ORDER
                if any(k in per_window[N]["subcorpus_norm"].unique() for N in windows)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
                bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("4-window forest plot — β(COI × cumulative_cue_attempts)",
                  fontsize=12, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    forest_png = out_dir / "forest_4panel.png"
    plt.savefig(forest_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {forest_png}")

    # ───── Pooled β trajectory ─────
    print("\n=== Building pooled β trajectory ===")
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, key, color in [("Full (n=32)", "FULL", "black"),
                                ("Drop-Manchester (n=21)", "DROP_MANC", "#d62728")]:
        xs = []
        bs, lows, highs = [], [], []
        for N in windows:
            m = meta_results[str(N)][key]
            if "pooled_beta_RE" not in m:
                continue
            xs.append(N)
            bs.append(m["pooled_beta_RE"])
            lows.append(m["pooled_beta_RE"] - 1.96 * m["pooled_se_RE"])
            highs.append(m["pooled_beta_RE"] + 1.96 * m["pooled_se_RE"])
        xs = np.array(xs); bs = np.array(bs); lows = np.array(lows); highs = np.array(highs)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.15)
        ax.plot(xs, bs, "o-", color=color, lw=2, label=label, markersize=8)
        for x, b in zip(xs, bs):
            ax.annotate(f"{b:+.3f}", (x, b), textcoords="offset points",
                         xytext=(8, 5), fontsize=8, color=color)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks(windows)
    ax.set_xticklabels([f"N = {n}" for n in windows])
    ax.set_xlabel("Outcome window (next-N child utterances)", fontsize=11)
    ax.set_ylabel("Pooled β(COI × cumulative_cue_attempts) with 95% CI", fontsize=11)
    ax.set_title("Pooled β trajectory across reuse-window range",
                  fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    trajectory_png = out_dir / "pooled_beta_trajectory.png"
    plt.savefig(trajectory_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {trajectory_png}")

    # ───── SUMMARY.md ─────
    lines: List[str] = []
    lines.append("# 17m Reuse-window-range sensitivity (N = 3, 5, 10, 20)\n")
    lines.append("## Pooled meta + Egger per window\n")
    lines.append("| N | Scope | n | β (RE) | SE | p | τ² | I² | 95% PI | Egger int | Egger p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for N in windows:
        for label, mkey, ekey in [("FULL", "FULL", "FULL"), ("DROP-MANC", "DROP_MANC", "DROP_MANC")]:
            m = meta_results[str(N)][mkey]; e = egger_results[str(N)][ekey]
            if "pooled_beta_RE" not in m:
                continue
            sig = "***" if m['pooled_p_RE']<0.001 else ("**" if m['pooled_p_RE']<0.01 else ("*" if m['pooled_p_RE']<0.05 else ("†" if m['pooled_p_RE']<0.10 else "")))
            lines.append(
                f"| N={N} | {label} | {m['n_studies']} | {m['pooled_beta_RE']:+.4f}{sig} | "
                f"{m['pooled_se_RE']:.4f} | {m['pooled_p_RE']:.4f} | {m['tau2']:.4f} | "
                f"{m['I2_pct']:.1f}% | "
                f"[{m['pred_interval_low']:+.3f}, {m['pred_interval_high']:+.3f}] | "
                f"{e.get('intercept', float('nan')):+.3f} | {e.get('intercept_p', float('nan')):.3f} |"
            )
    lines.append("\n## Acceptance criteria\n")
    # Check criteria
    all_full_pos = all(meta_results[str(N)]["FULL"].get("pooled_beta_RE", -1) > 0 for N in windows)
    all_drop_sig = all(meta_results[str(N)]["DROP_MANC"].get("pooled_p_RE", 1) < 0.05 for N in windows)
    egger_clean = all(egger_results[str(N)]["FULL"].get("intercept_p", 0) > 0.10
                       and egger_results[str(N)]["DROP_MANC"].get("intercept_p", 0) > 0.10
                       for N in windows)
    lines.append(f"- ✓ FULL β positive at all 4 windows: **{all_full_pos}**")
    lines.append(f"- ✓ DROP-MANC β sig (p<.05) at all 4 windows: **{all_drop_sig}**")
    lines.append(f"- ✓ Egger p > 0.10 at all 8 cells: **{egger_clean}**")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
