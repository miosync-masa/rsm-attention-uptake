"""
17b_exposure_gate_test.py
=========================
IMT Attention Bias Paper 2 — Step 17b: Exposure-based gate test

Replaces the broken `age_post × COI` interaction (SPEC #1 of
17_robustness_age_post_coi.py) with the boss-proposed exposure-gate
formulation:

  Layer 0 saturation proxy = cumulative_cue_attempts(child, cue, t)
    "number of times THIS child has used THIS cue up to (but not
    including) the current episode"

  Main model (per corpus, run on POST-MSR subset only):
    reuse_nextN[e]  ~  COI_z(c) × cumulative_cue_attempts_z(child, c, t)
                     + prior_local_freq_z[e]      (short autoregression)
                     + log_cue_freq_z(c)          (corpus-level frequency)
                     + (1 | child)                (random intercept)
    Cluster-robust SE on cue_subtype.

  Pass criterion:
    β(COI × cumulative_cue_attempts) > 0  AND  p < 0.01  in 3/3 corpora.

  Interpretation (if pass):
    Grammar gate is exposure accumulation, not age. Paper 1's
    COI × frequency interaction is time-scale-invariant: once a child has
    enough cumulative experience with a cue, attention amplifies reuse.

────────────────────────────────────────────────────────────────────────────

Supplementary tests in the same script
--------------------------------------
S1  Same model on PRE-MSR subset. Expected: interaction null or weak.
S2  Pearson/Spearman correlation between cumulative_cue_attempts and
    child_age_months (within post-MSR), plus variance-inflation check.
S3  Time-window sensitivity (N ∈ {3, 5, 10}).
S4  Continuous age within post-MSR replacing the post-MSR split:
    reuse_nextN ~ COI_z × cumulative_cue_attempts_z
                + ns(age_months)
                + prior_local_freq_z + log_cue_freq_z
    (S4 is approximated by linear age; full spline left to follow-up.)

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17b/)
----------------------
* {lang}_exposure_gate_N{N}.json    per-corpus result (M_new + supplementary)
* main_results_N{N}.json            combined dict
* cumulative_vs_age_scatter.png     visualization for S2 (post-MSR only)
* exposure_gate_summary_N{N}.csv    long-format β table
* SUMMARY.md                        one-page pass-criterion report

Author: Torami x Boss | IMT Attention project | Paper 2 / exposure gate v1 | 2026-06-21
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
# Shared helpers (mirrors of 17_robustness)
# ─────────────────────────────────────────────────────────────────────────────

def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
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


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative cue attempts per (child, cue)
# ─────────────────────────────────────────────────────────────────────────────

def add_cumulative_cue_attempts(df_with_child: pd.DataFrame) -> pd.DataFrame:
    """
    For each (child, cue_subtype), count prior occurrences of that cue in
    child speech, in chronological order. The current row is NOT counted
    in its own cumulative value (cumcount gives 0 for the first instance).

    Sort key: child_age_months, then file, then child_utt_idx
        — child_age_months is the natural developmental clock
        — within an episode session (same age, same file) we break ties
          on child_utt_idx
    """
    df = df_with_child.copy()
    df = df.sort_values(
        ["child", "cue_subtype", "child_age_months", "file", "child_utt_idx"]
    ).reset_index(drop=True)
    df["cumulative_cue_attempts"] = df.groupby(["child", "cue_subtype"]).cumcount()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

def _params(fit, names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    ci = fit.conf_int()
    stat = fit.tvalues
    for p in names:
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
        "cluster_col": cluster_col,
        "params": _params(fit, ["const"] + predictors),
    }


def fit_mixedlm_child(df: pd.DataFrame, predictors: List[str],
                       outcome_col: str) -> Dict[str, Any]:
    if "child" not in df.columns or df["child"].isna().all():
        return {"error": "no child column"}
    df = df.dropna(subset=["child"]).copy()
    formula = f"{outcome_col} ~ " + " + ".join(predictors)
    try:
        fit = smf.mixedlm(formula, data=df, groups=df["child"]).fit(method="lbfgs")
    except Exception as exc:
        return {"error": f"MixedLM failed: {exc}"}
    fixed = ["Intercept"] + predictors
    return {
        "n": int(fit.nobs),
        "n_children": int(df["child"].nunique()),
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
# Per-corpus pipeline
# ─────────────────────────────────────────────────────────────────────────────

PREDICTORS_MAIN = [
    "COI_z", "cumulative_cue_attempts_z", "COI_x_cumulative",
    "prior_local_freq_z", "log_cue_freq_z",
]


def analyze_corpus(language: str, reuse_csv: Path, tagged_csv: Path,
                   joined_csv: Path, json_cache: Path,
                   window: int, prior_window: int,
                   contingent_only: bool,
                   output_dir: Path) -> Dict[str, Any]:
    print(f"\n=== {language} ===")
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

    # Attach child id from json_cache mapping
    file_to_child, max_len = build_file_child_map(json_cache)
    raw = reuse["file"].astype(str)
    reuse["child"] = raw.map(file_to_child)
    miss = reuse["child"].isna()
    if miss.any() and max_len > 0:
        padded = raw.str.zfill(max_len)
        reuse.loc[miss, "child"] = padded[miss].map(file_to_child)
    n_mapped = reuse["child"].notna().sum()
    print(f"  rows mapped to a child: {n_mapped:,} ({100*n_mapped/len(reuse):.1f}%)")
    if reuse["child"].isna().any():
        reuse = reuse.dropna(subset=["child"]).copy()

    # Merge baseline COI / log_cue_freq from joined CSV
    joined = pd.read_csv(joined_csv)
    if "log_caregiver_count" in joined.columns:
        joined = joined.rename(columns={"log_caregiver_count": "log_cue_freq"})
    elif "logFreq" in joined.columns:
        joined = joined.rename(columns={"logFreq": "log_cue_freq"})
    cols = ["cue_subtype", "COI", "log_cue_freq"]
    if not all(c in joined.columns for c in cols):
        return {"error": f"joined CSV missing required columns ({cols})"}
    df = reuse.merge(joined[cols], on="cue_subtype", how="inner")
    df = df.dropna(subset=[outcome_col, "COI", "log_cue_freq", "child_age_months"]).copy()
    print(f"  after merge & dropna: {len(df):,}")

    # Cumulative cue attempts (computed BEFORE subsetting so the count
    # reflects lifetime exposure to that point, including pre-MSR uses)
    df = add_cumulative_cue_attempts(df)

    # Splits
    df["age_post"] = (df["child_age_months"] >= 24).astype(int)
    df_post = df[df["age_post"] == 1].copy()
    df_pre  = df[df["age_post"] == 0].copy()
    print(f"  post-MSR rows: {len(df_post):,}    pre-MSR rows: {len(df_pre):,}")

    # Standardize predictors WITHIN each subset (so β interpretation is
    # "per-SD within this developmental band")
    def _prep(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["COI_z"]                       = z(d["COI"])
        d["cumulative_cue_attempts_z"]   = z(d["cumulative_cue_attempts"])
        d["prior_local_freq_z"]          = z(d["prior_local_freq"])
        d["log_cue_freq_z"]              = z(d["log_cue_freq"])
        d["COI_x_cumulative"]            = d["COI_z"] * d["cumulative_cue_attempts_z"]
        return d

    df_post = _prep(df_post)
    df_pre  = _prep(df_pre)

    res: Dict[str, Any] = {
        "language":             language,
        "outcome_col":          outcome_col,
        "window":               window,
        "prior_window":         prior_window,
        "contingent_only":      contingent_only,
        "n_post_msr":           int(len(df_post)),
        "n_pre_msr":            int(len(df_pre)),
        "n_cues_post":          int(df_post["cue_subtype"].nunique()),
        "n_cues_pre":           int(df_pre["cue_subtype"].nunique()),
        "n_children_post":      int(df_post["child"].nunique()),
        "n_children_pre":       int(df_pre["child"].nunique()),
        "outcome_mean_post":    float(df_post[outcome_col].mean()) if len(df_post) else None,
        "outcome_mean_pre":     float(df_pre[outcome_col].mean()) if len(df_pre) else None,
    }

    if len(df_post) < 100:
        return {**res, "error": "post-MSR rows < 100"}

    # M_new — post-MSR
    print("  fit M_new (post-MSR, OLS + cluster-robust on cue) ...")
    res["M_new_post_OLS_cluster_cue"] = fit_ols_cluster(
        df_post, PREDICTORS_MAIN, outcome_col, cluster_col="cue_subtype",
    )
    print("  fit M_new (post-MSR, OLS + cluster-robust on child) ...")
    res["M_new_post_OLS_cluster_child"] = fit_ols_cluster(
        df_post, PREDICTORS_MAIN, outcome_col, cluster_col="child",
    )
    print("  fit M_new (post-MSR, MixedLM with random by child) ...")
    res["M_new_post_MixedLM_child"] = fit_mixedlm_child(
        df_post, PREDICTORS_MAIN, outcome_col,
    )

    # S1 — pre-MSR same model (if enough rows)
    if len(df_pre) >= 100:
        print("  fit M_new (pre-MSR, OLS + cluster-robust on cue) ...")
        res["S1_pre_OLS_cluster_cue"] = fit_ols_cluster(
            df_pre, PREDICTORS_MAIN, outcome_col, cluster_col="cue_subtype",
        )
        res["S1_pre_MixedLM_child"] = fit_mixedlm_child(
            df_pre, PREDICTORS_MAIN, outcome_col,
        )

    # S2 — collinearity of cumulative_cue_attempts and age (post-MSR)
    if len(df_post) >= 100:
        x = df_post["cumulative_cue_attempts"].astype(float)
        y_age = df_post["child_age_months"].astype(float)
        r_p = float(np.corrcoef(x, y_age)[0, 1])
        rank_p = float(pd.Series(x).rank().corr(pd.Series(y_age).rank()))
        res["S2_cumulative_vs_age_post"] = {
            "pearson_r":  r_p,
            "spearman_r": rank_p,
            "n": int(len(df_post)),
        }

    # S4 — continuous age within post-MSR
    if len(df_post) >= 100:
        df_post_age = df_post.copy()
        df_post_age["age_z"] = z(df_post_age["child_age_months"])
        preds_age = PREDICTORS_MAIN + ["age_z"]
        print("  fit S4 (post-MSR + linear age, OLS + cluster on cue) ...")
        res["S4_post_OLS_cluster_cue"] = fit_ols_cluster(
            df_post_age, preds_age, outcome_col, cluster_col="cue_subtype",
        )

    # Persist long-form CSV slice for plotting (S2 scatter)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{language}_post_msr_cumulative_age.csv"
    df_post[["child", "cue_subtype", "child_age_months", "cumulative_cue_attempts"]]\
        .to_csv(plot_path, index=False)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_for_summary(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict] = []
    for lang, res in all_results.items():
        if lang == "_meta":
            continue
        label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
        for spec_name, spec in res.items():
            if not isinstance(spec, dict) or "params" not in spec:
                continue
            for predictor, vals in spec["params"].items():
                rows.append({
                    "corpus":     label,
                    "language":   lang,
                    "spec":       spec_name,
                    "predictor":  predictor,
                    "beta":       vals.get("beta"),
                    "se":         vals.get("se"),
                    "stat":       vals.get("stat"),
                    "p":          vals.get("p"),
                    "ci95_low":   vals.get("ci95_low"),
                    "ci95_high":  vals.get("ci95_high"),
                })
    return pd.DataFrame(rows)


def make_scatter(all_results: Dict[str, Dict[str, Any]], output_dir: Path,
                  window: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping scatter plot")
        return

    panels = []
    for lang in all_results:
        if lang == "_meta":
            continue
        csv = output_dir / f"{lang}_post_msr_cumulative_age.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if len(df) == 0:
            continue
        panels.append((CANONICAL_CORPORA.get(lang, {}).get("label", lang), df))

    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, df) in zip(axes, panels):
        sample = df.sample(min(20000, len(df)), random_state=42)
        ax.scatter(sample["child_age_months"], sample["cumulative_cue_attempts"],
                    s=2, alpha=0.15, color="steelblue")
        r = np.corrcoef(df["child_age_months"], df["cumulative_cue_attempts"])[0, 1]
        ax.set_title(f"{label} (post-MSR)\nPearson r = {r:+.3f}", fontsize=10)
        ax.set_xlabel("child_age_months", fontsize=9)
        ax.set_ylabel("cumulative_cue_attempts", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"S2: cumulative_cue_attempts vs age (post-MSR)  | outcome window N={window}",
                  fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = output_dir / "cumulative_vs_age_scatter.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  → {out_png}")


def write_summary_md(all_results: Dict[str, Dict[str, Any]], output_dir: Path,
                      window: int) -> None:
    target = "COI_x_cumulative"
    lines: List[str] = []
    lines.append(f"# SPEC #1b exposure-gate test — outcome window N={window}\n")
    lines.append("## Pass criterion\n")
    lines.append("β(COI × cumulative_cue_attempts) > 0 AND p < 0.01 in 3/3 corpora.\n")

    for spec_block, title in [
        ("M_new_post_OLS_cluster_cue",   "M_new (post-MSR, OLS + cluster on cue)"),
        ("M_new_post_OLS_cluster_child", "M_new (post-MSR, OLS + cluster on child)"),
        ("M_new_post_MixedLM_child",     "M_new (post-MSR, MixedLM random by child)"),
        ("S1_pre_OLS_cluster_cue",       "S1 (pre-MSR, OLS + cluster on cue) — expected null"),
        ("S4_post_OLS_cluster_cue",      "S4 (post-MSR + linear age, OLS + cluster on cue)"),
    ]:
        lines.append(f"\n## {title}\n")
        lines.append(f"| Corpus | β | SE | p | verdict |")
        lines.append(f"|---|---|---|---|---|")
        passes, total = 0, 0
        for lang in all_results:
            if lang == "_meta":
                continue
            r = all_results[lang].get(spec_block, {}).get("params", {}).get(target)
            label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
            if r is None:
                lines.append(f"| {label} | — | — | — | n/a |")
                continue
            ok = (r["beta"] > 0) and (r["p"] < 0.01)
            verdict = "✓ PASS" if ok else "✗"
            passes += int(ok); total += 1
            lines.append(
                f"| {label} | {r['beta']:+.4f} | {r['se']:.4f} | {r['p']:.4f} | {verdict} |"
            )
        if total > 0 and spec_block.startswith("M_new"):
            lines.append(f"\n→ **{passes}/{total} corpora pass**")

    md_path = output_dir / "SUMMARY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPEC #1b: exposure-gate test.")
    p.add_argument("--reuse_csv",   default=None)
    p.add_argument("--tagged_csv",  default=None)
    p.add_argument("--joined_csv",  default=None)
    p.add_argument("--json_cache",  default=None)
    p.add_argument("--language",    default=None)
    p.add_argument("--output_dir",  default="./output/v17b")
    p.add_argument("--window",      type=int, default=5)
    p.add_argument("--prior_window", type=int, default=20)
    p.add_argument("--include_noncontingent", action="store_true")
    p.add_argument("--batch", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    contingent_only = not args.include_noncontingent

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
        targets = [(args.language, args.reuse_csv, args.tagged_csv,
                    args.joined_csv, args.json_cache)]

    all_results: Dict[str, Any] = {"_meta": {
        "window": args.window, "prior_window": args.prior_window,
        "contingent_only": contingent_only,
    }}
    for lang, rc, tc, jc, jcache in targets:
        rcp, tcp, jcp, jcp_cache = map(Path, [rc, tc, jc, jcache])
        missing = [p for p in [rcp, tcp, jcp] if not p.exists()]
        if missing:
            print(f"  SKIP {lang}: missing inputs {missing}")
            continue
        res = analyze_corpus(lang, rcp, tcp, jcp, jcp_cache,
                              window=args.window,
                              prior_window=args.prior_window,
                              contingent_only=contingent_only,
                              output_dir=out_dir)
        all_results[lang] = res
        per_corpus_json = out_dir / f"{lang}_exposure_gate_N{args.window}.json"
        with open(per_corpus_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  → {per_corpus_json}")

    # Combined outputs
    combined_json = out_dir / f"main_results_N{args.window}.json"
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  → {combined_json}")

    summary = flatten_for_summary(all_results)
    summary_csv = out_dir / f"exposure_gate_summary_N{args.window}.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"  → {summary_csv}  ({len(summary):,} rows)")

    make_scatter(all_results, out_dir, window=args.window)
    write_summary_md(all_results, out_dir, window=args.window)

    # Pass-criterion print to stdout
    target_pred = "COI_x_cumulative"
    print("\n========== SPEC #1b pass-criterion report ==========")
    print(f"  Pass = β({target_pred}) > 0 AND p < 0.01\n")
    for spec_key in ["M_new_post_OLS_cluster_cue",
                     "M_new_post_OLS_cluster_child",
                     "M_new_post_MixedLM_child"]:
        print(f"  -- {spec_key} --")
        passes, total = 0, 0
        for lang in all_results:
            if lang == "_meta":
                continue
            label = CANONICAL_CORPORA.get(lang, {}).get("label", lang)
            r = all_results[lang].get(spec_key, {}).get("params", {}).get(target_pred)
            if r is None:
                print(f"    {label:<12}: (missing)")
                continue
            ok = (r["beta"] > 0) and (r["p"] < 0.01)
            mark = "✓ PASS" if ok else "✗"
            passes += int(ok); total += 1
            print(f"    {label:<12}: β={r['beta']:+.4f}  SE={r['se']:.4f}  p={r['p']:.4f}  {mark}")
        if total > 0:
            verdict = "PASS (3/3)" if passes >= 3 else f"FAIL ({passes}/{total})"
            print(f"    → {passes}/{total}  {verdict}\n")


if __name__ == "__main__":
    main()
