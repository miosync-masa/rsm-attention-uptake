"""
03_compute_attention_index.py
=============================
IMT Attention Bias Paper — Step 3: 5-Dimensional Salience & Attention Index

Reads the cue-tagged tokens CSV from step 02 and computes, per cue_subtype:

  S_acoustic     — utterance-edge proxy (lengthening effect when no audio)
  S_positional   — position-distribution edge bias
  S_frequency    — log-frequency z-score (per language)
  S_repetition   — type/host diversity (lemma host set)
  S_perceptual   — orthographic distinctiveness from nearest contrast

Then:
  AI(c, L) = mean of the 5 normalized salience scores per cue
  Reliability(c) = P(most-likely function | cue surface form)

Aggregates results into a per-cue table for downstream statistical modeling
(script 06).

Usage:
    python 03_compute_attention_index.py \\
        --tagged_csv ./output/English_tokens_tagged.csv \\
        --language English \\
        --output_dir ./output/

Author: Torami x Boss | IMT Attention project | 2026-06-13
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy pandas tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Attention Index per cue.")
    p.add_argument("--tagged_csv", required=True,
                   help="Tagged tokens CSV from 02_extract_cues.py")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK", "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--caregiver_only", action="store_true", default=True,
                   help="Compute AI from caregiver speech only (CDS).")
    p.add_argument("--min_cue_count", type=int, default=5,
                   help="Minimum occurrences for a cue_subtype to be analyzed.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Salience dimension computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_acoustic_proxy(group: pd.DataFrame) -> float:
    """
    S_acoustic proxy from text data:
        utterance-final position correlates with lengthening (phrase-final
        lengthening is a robust prosodic universal). We use the proportion of
        cue occurrences that fall in utterance-final position as a proxy.

    Returns a value in [0, 1].
    """
    if len(group) == 0:
        return 0.0
    return float(group["is_utterance_final"].mean())


def compute_positional_salience(group: pd.DataFrame, n_bins: int = 10) -> float:
    """
    S_positional: edge-vs-uniform bias.
    A cue with all occurrences at position 0..0.1 or 0.9..1.0 scores high.
    A uniformly-distributed cue scores near 0.

    Returns max(P(first decile), P(last decile)) - uniform_baseline.
    """
    if len(group) == 0:
        return 0.0
    pos = group["position_in_utterance"].astype(float)
    first_decile = float((pos <= 0.1).mean())
    last_decile = float((pos >= 0.9).mean())
    uniform_baseline = 1.0 / n_bins  # = 0.1
    return max(first_decile, last_decile) - uniform_baseline


def compute_frequency_zscore(
    counts: Dict[str, int]
) -> Dict[str, float]:
    """
    S_frequency: log-frequency z-score within language.
    Returns dict: cue_subtype -> z-scored log-frequency.
    """
    if not counts:
        return {}
    log_counts = {k: np.log(v) for k, v in counts.items() if v > 0}
    if not log_counts:
        return {}
    values = np.array(list(log_counts.values()))
    mean = values.mean()
    std = values.std() if values.std() > 0 else 1.0
    return {k: float((v - mean) / std) for k, v in log_counts.items()}


def compute_repetition_salience(group: pd.DataFrame) -> float:
    """
    S_repetition: how many distinct host lemmas does the cue attach to?
    Normalized via log scaling.

    A cue attached to ~1 lemma -> very low score (lexicalized).
    A cue attached to many lemmas -> high score (productive).
    """
    if len(group) == 0:
        return 0.0
    # For affixes/particles: 'lemma' is the cue itself; we want host word diversity.
    # Approximate host via raw_text (which may include hosts for clitics).
    hosts = group["raw_text"].fillna("").astype(str).str.lower()
    n_hosts = hosts.nunique()
    # log scaling, capped at 100 hosts -> 1.0
    return float(min(np.log1p(n_hosts) / np.log(101), 1.0))


def compute_perceptual_salience(
    cue_form: str, all_cue_forms: List[str]
) -> float:
    """
    S_perceptual: orthographic distinctiveness.
    Defined as 1 - (min Levenshtein distance to other cue forms) / max_len.

    A cue that is very similar to other cues (e.g., 'は' vs 'わ' in Japanese
    written hiragana) scores low.
    """
    if not cue_form or len(all_cue_forms) <= 1:
        return 0.5  # neutral if no comparison possible
    others = [f for f in all_cue_forms if f and f != cue_form]
    if not others:
        return 1.0
    min_dist = min(_levenshtein(cue_form, o) for o in others)
    max_len = max(len(cue_form), max(len(o) for o in others), 1)
    return float(min_dist / max_len)


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # deletion
                curr_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost, # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Reliability computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reliability(group: pd.DataFrame) -> float:
    """
    Reliability_gra = how concentrated this cue is in one gra_relation.

    ⚠️ CIRCULARITY WARNING:
    For cues whose cue_subtype was DEFINED from gra_relation (e.g.,
    subject_position from gra=NSUBJ), this metric reflects tag-definition
    consistency rather than genuine predictive power. For such cues,
    Reliability_gra ≈ 1.0 is tautological.

    Use Reliability_position and Reliability_form (below) for non-circular
    role prediction. They use ONLY position bin or surface form as predictors,
    never gra_relation.
    """
    if len(group) == 0:
        return 0.0
    if "gra_relation" not in group.columns:
        return float("nan")
    gra = group["gra_relation"].dropna()
    gra = gra[gra.astype(str).str.len() > 0]
    if len(gra) == 0:
        return float("nan")
    rel_counts = gra.value_counts()
    return float(rel_counts.iloc[0] / rel_counts.sum())


def compute_function_entropy(group: pd.DataFrame) -> float:
    """
    Conditional entropy H(gra_relation | cue) normalized by log(|relations|).
    Subject to the same circularity caveat as Reliability_gra.
    """
    if len(group) == 0:
        return float("nan")
    if "gra_relation" not in group.columns:
        return float("nan")
    gra = group["gra_relation"].dropna()
    gra = gra[gra.astype(str).str.len() > 0]
    if len(gra) <= 1:
        return 0.0
    counts = gra.value_counts()
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -float((p * np.log(p)).sum())
    h_max = np.log(len(counts))
    return float(h / h_max) if h_max > 0 else 0.0


def compute_position_reliability_raw(group: pd.DataFrame) -> float:
    """
    Position consistency: how tight is the cue's position distribution?
    Low std → reliable positional cue. Returns 1 - normalized_std in [0, 1].
    """
    if len(group) == 0:
        return float("nan")
    pos = group["position_in_utterance"].dropna()
    if len(pos) <= 1:
        return 1.0
    std = float(pos.std())
    return float(max(0.0, 1.0 - std / 0.289))


def _bin_position(p: float, n_bins: int = 5) -> int:
    """Discretize position in [0,1] into bins for cross-tabulation."""
    if pd.isna(p):
        return -1
    return min(int(p * n_bins), n_bins - 1)


def compute_reliability_position(
    cue_group: pd.DataFrame, global_pos_role_table: pd.DataFrame
) -> float:
    """
    NON-CIRCULAR ROLE PREDICTION via position bin only.

    Given the position bin distribution of THIS cue, predict the most likely
    gra_relation using the GLOBAL P(role | position_bin) lookup built from
    the entire caregiver corpus (across all cue types).

    Returns: max accuracy achievable by a position-only classifier on this cue.

    For 'subject_position' defined from gra=NSUBJ, this asks:
    "If we know ONLY position bin, how often do we recover NSUBJ?"
    If position alone gives NSUBJ 95% of the time, that's genuine
    predictive validity. If it gives NSUBJ only 40%, the gra-based
    Reliability_gra=1.0 was circular.
    """
    if len(cue_group) == 0 or global_pos_role_table.empty:
        return float("nan")

    # Most common gra_relation for this cue
    gra_vals = cue_group["gra_relation"].dropna()
    gra_vals = gra_vals[gra_vals.astype(str).str.len() > 0]
    if len(gra_vals) == 0:
        return float("nan")
    target_role = gra_vals.value_counts().index[0]

    # Position bin distribution of this cue
    pos_bins = cue_group["position_in_utterance"].dropna().apply(_bin_position)
    pos_bins = pos_bins[pos_bins >= 0]
    if len(pos_bins) == 0:
        return float("nan")

    # For each token in this cue, use the global P(role | bin) table to
    # predict role from position alone. Score = fraction where prediction = target.
    correct = 0
    total = 0
    for b in pos_bins:
        if b in global_pos_role_table.index:
            predicted_role = global_pos_role_table.loc[b].idxmax()
            if predicted_role == target_role:
                correct += 1
            total += 1
    return correct / total if total > 0 else float("nan")


def compute_reliability_form(
    cue_group: pd.DataFrame, global_form_role_table: pd.DataFrame
) -> float:
    """
    NON-CIRCULAR ROLE PREDICTION via surface form only.

    Same as compute_reliability_position but using cleaned_text instead of bin.
    For morphological cues (e.g., Japanese particles), this should be very high
    (the form 'が' robustly predicts NSUBJ). For positional cues like
    'subject_position', this is near-trivial since the form varies wildly.
    """
    if len(cue_group) == 0 or global_form_role_table.empty:
        return float("nan")

    gra_vals = cue_group["gra_relation"].dropna()
    gra_vals = gra_vals[gra_vals.astype(str).str.len() > 0]
    if len(gra_vals) == 0:
        return float("nan")
    target_role = gra_vals.value_counts().index[0]

    forms = cue_group["cleaned_text"].dropna().astype(str)
    forms = forms[forms.str.len() > 0]
    if len(forms) == 0:
        return float("nan")

    correct = 0
    total = 0
    for f in forms:
        if f in global_form_role_table.index:
            predicted_role = global_form_role_table.loc[f].idxmax()
            if predicted_role == target_role:
                correct += 1
            total += 1
    return correct / total if total > 0 else float("nan")


def build_global_position_role_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build P(gra_relation | position_bin) from ALL caregiver tokens
    (not filtered to cue tokens). This is the non-circular predictor.
    """
    work = df[["position_in_utterance", "gra_relation"]].dropna().copy()
    work = work[work["gra_relation"].astype(str).str.len() > 0]
    if len(work) == 0:
        return pd.DataFrame()
    work["bin"] = work["position_in_utterance"].apply(_bin_position)
    work = work[work["bin"] >= 0]
    # Crosstab: rows = bin, cols = gra_relation, values = count
    table = pd.crosstab(work["bin"], work["gra_relation"])
    return table


def build_global_form_role_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build P(gra_relation | cleaned_text) from ALL caregiver tokens.
    Only forms with count >= 3 are retained to avoid noise.
    """
    work = df[["cleaned_text", "gra_relation"]].dropna().copy()
    work = work[work["gra_relation"].astype(str).str.len() > 0]
    work = work[work["cleaned_text"].astype(str).str.len() > 0]
    if len(work) == 0:
        return pd.DataFrame()
    table = pd.crosstab(work["cleaned_text"], work["gra_relation"])
    # Filter to forms with at least 3 occurrences total
    row_sums = table.sum(axis=1)
    table = table[row_sums >= 3]
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_indices(
    df: pd.DataFrame, min_count: int
) -> pd.DataFrame:
    """
    For each cue_subtype, compute the 5 salience dimensions + reliability + AI.
    v3: adds non-circular reliability metrics (position-only, form-only).
    """
    # Filter to cue tokens only
    cue_df = df[df["is_cue_token"].astype(bool)].copy()
    if len(cue_df) == 0:
        return pd.DataFrame()

    # ─── v3: Build global non-circular predictor tables from ALL tokens ───
    # These tables are NOT restricted to cue tokens; they reflect the corpus.
    print("  Building global position-role and form-role tables (v3)...")
    global_pos_role = build_global_position_role_table(df)
    global_form_role = build_global_form_role_table(df)
    print(f"    Position-role: {global_pos_role.shape}")
    print(f"    Form-role:     {global_form_role.shape}")

    # Pre-compute language-level frequency z-scores
    cue_counts = cue_df["cue_subtype"].value_counts().to_dict()
    freq_zscores = compute_frequency_zscore(cue_counts)

    # Get all cue surface forms for perceptual comparison
    all_forms = (
        cue_df.groupby("cue_subtype")["cleaned_text"]
        .agg(lambda x: x.value_counts().index[0] if len(x) else "")
        .to_dict()
    )
    all_form_strings = [v for v in all_forms.values() if v]

    rows = []
    for cue_subtype, group in tqdm(
        cue_df.groupby("cue_subtype"), desc="Computing AI"
    ):
        count = len(group)
        if count < min_count:
            continue

        cue_form = all_forms.get(cue_subtype, "")
        cue_type = group["cue_type"].iloc[0] if len(group) else ""

        # Five dimensions
        s_acoustic = compute_acoustic_proxy(group)
        s_positional = compute_positional_salience(group)
        s_frequency_z = freq_zscores.get(cue_subtype, 0.0)
        s_repetition = compute_repetition_salience(group)
        s_perceptual = compute_perceptual_salience(cue_form, all_form_strings)

        # Normalize frequency z-score to [0, 1] via sigmoid for AI combination
        s_frequency_n = 1.0 / (1.0 + np.exp(-s_frequency_z))

        # Combine: equal weights (per attention_index_formalization_v1 §6)
        ai = (s_acoustic + s_positional + s_frequency_n
              + s_repetition + s_perceptual) / 5.0

        # ─── v2: Weighted AI suppresses rare-cue overweight (Tomoe指摘 #2) ───
        # WAI = AI * log1p(count) / log1p(max_count)  — normalized to [0,1]
        max_count = max(cue_counts.values()) if cue_counts else 1
        weight_factor = np.log1p(count) / np.log1p(max_count) if max_count > 1 else 1.0
        wai = ai * weight_factor

        # ─── v2: Reliability now measured via gra_relation diversity ────────
        # ⚠️ For cues defined from gra_relation, this is tautological — see warning.
        reliability_gra = compute_reliability(group)
        entropy = compute_function_entropy(group)
        pos_consistency = compute_position_reliability_raw(group)

        # ─── v3: Non-circular reliability metrics (Boss's critical fix) ─────
        reliability_position = compute_reliability_position(group, global_pos_role)
        reliability_form = compute_reliability_form(group, global_form_role)

        # Detect circularity: True if this cue_subtype was defined via gra
        # (heuristic: name ends in '_position' / matches a gra-defined family)
        gra_defined_cues = {
            "subject_position", "object_position",
            "indirect_object", "oblique_argument",
        }
        is_gra_defined = cue_subtype in gra_defined_cues

        # Composite: average of position-pred and form-pred (when both valid)
        non_circular_vals = [
            v for v in [reliability_position, reliability_form]
            if v is not None and not np.isnan(v)
        ]
        composite_reliability = (
            float(np.mean(non_circular_vals)) if non_circular_vals else float("nan")
        )

        # Dominant gra_relation for interpretability
        gra_vals = group["gra_relation"].dropna() if "gra_relation" in group.columns else pd.Series(dtype=str)
        gra_vals = gra_vals[gra_vals.astype(str).str.len() > 0]
        dominant_gra = gra_vals.value_counts().index[0] if len(gra_vals) else ""
        n_distinct_gra = int(gra_vals.nunique())

        # Position stats
        pos_mean = float(group["position_in_utterance"].mean())
        pos_std = float(group["position_in_utterance"].std())

        rows.append({
            "cue_subtype": cue_subtype,
            "cue_type": cue_type,
            "cue_form": cue_form,
            "count": count,
            "S_acoustic": round(s_acoustic, 4),
            "S_positional": round(s_positional, 4),
            "S_frequency_zscore": round(s_frequency_z, 4),
            "S_frequency_normalized": round(s_frequency_n, 4),
            "S_repetition": round(s_repetition, 4),
            "S_perceptual": round(s_perceptual, 4),
            "AttentionIndex": round(ai, 4),
            "WeightedAttentionIndex": round(wai, 4),
            "Reliability_gra": round(reliability_gra, 4) if not np.isnan(reliability_gra) else None,
            "Reliability_gra_is_circular": is_gra_defined,
            "Reliability_position": round(reliability_position, 4) if not np.isnan(reliability_position) else None,
            "Reliability_form": round(reliability_form, 4) if not np.isnan(reliability_form) else None,
            "Reliability_composite_noncircular": round(composite_reliability, 4) if not np.isnan(composite_reliability) else None,
            "FunctionEntropy_gra": round(entropy, 4) if not np.isnan(entropy) else None,
            "PositionConsistency": round(pos_consistency, 4) if not np.isnan(pos_consistency) else None,
            "dominant_gra_relation": dominant_gra,
            "n_distinct_gra_relations": n_distinct_gra,
            "pos_mean": round(pos_mean, 4),
            "pos_std": round(pos_std, 4),
            "n_distinct_functions": int(group["cue_function"].nunique()),
            "most_common_function": (
                group["cue_function"].value_counts().index[0]
                if len(group) and group["cue_function"].notna().any() else ""
            ),
        })

    out_df = pd.DataFrame(rows).sort_values(
        "WeightedAttentionIndex", ascending=False
    ).reset_index(drop=True)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# v2: cue_type aggregation (Tomoe指摘 #6)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_cue_type(ai_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up cue_subtypes into cue_type categories for a coarser, more
    comparable cross-linguistic profile.
    """
    if len(ai_df) == 0:
        return pd.DataFrame()

    rows = []
    for cue_type, group in ai_df.groupby("cue_type"):
        total_count = int(group["count"].sum())
        # Count-weighted means
        weights = group["count"]
        rows.append({
            "cue_type": cue_type,
            "n_subtypes": len(group),
            "total_count": total_count,
            "AI_weighted_mean": float(np.average(group["AttentionIndex"], weights=weights)),
            "WAI_weighted_mean": float(np.average(group["WeightedAttentionIndex"], weights=weights)),
            "AI_unweighted_mean": float(group["AttentionIndex"].mean()),
            "S_acoustic_w": float(np.average(group["S_acoustic"], weights=weights)),
            "S_positional_w": float(np.average(group["S_positional"], weights=weights)),
            "S_repetition_w": float(np.average(group["S_repetition"], weights=weights)),
            "pos_mean_w": float(np.average(group["pos_mean"], weights=weights)),
        })

    return pd.DataFrame(rows).sort_values("total_count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.tagged_csv}")
    df = pd.read_csv(args.tagged_csv, low_memory=False)
    print(f"  {len(df):,} tokens loaded.")

    if args.caregiver_only:
        df = df[df["speaker_role"] == "caregiver"].copy()
        print(f"  Filtered to caregiver speech: {len(df):,} tokens.")

    print(f"\nComputing 5-dimensional salience for {args.language}...")
    ai_df = compute_attention_indices(df, args.min_cue_count)

    if len(ai_df) == 0:
        print("WARNING: no cues survived the min_count filter.")
        return

    out_csv = out_dir / f"{args.language}_attention_index.csv"
    ai_df.to_csv(out_csv, index=False)
    print(f"\n  Wrote: {out_csv}  ({len(ai_df)} cue subtypes)")

    # v2: cue_type aggregation
    type_df = aggregate_by_cue_type(ai_df)
    type_csv = out_dir / f"{args.language}_cue_type_profile.csv"
    type_df.to_csv(type_csv, index=False)
    print(f"  Wrote: {type_csv}  ({len(type_df)} cue types)")

    # Summary with three top-rankings (巴指摘 #2: AI / count / WAI separated)
    # v3: Reliability_position / Reliability_form (non-circular) added
    summary = {
        "language": args.language,
        "n_cue_subtypes": len(ai_df),
        "n_total_cue_tokens": int(ai_df["count"].sum()),
        "AI_mean": float(ai_df["AttentionIndex"].mean()),
        "AI_std": float(ai_df["AttentionIndex"].std()),
        "WAI_mean": float(ai_df["WeightedAttentionIndex"].mean()),
        "WAI_std": float(ai_df["WeightedAttentionIndex"].std()),
        "Reliability_gra_mean": (
            float(ai_df["Reliability_gra"].mean())
            if "Reliability_gra" in ai_df.columns
               and ai_df["Reliability_gra"].notna().any()
            else None
        ),
        "Reliability_position_mean": (
            float(ai_df["Reliability_position"].mean())
            if "Reliability_position" in ai_df.columns
               and ai_df["Reliability_position"].notna().any()
            else None
        ),
        "Reliability_form_mean": (
            float(ai_df["Reliability_form"].mean())
            if "Reliability_form" in ai_df.columns
               and ai_df["Reliability_form"].notna().any()
            else None
        ),
        "Reliability_noncircular_mean": (
            float(ai_df["Reliability_composite_noncircular"].mean())
            if "Reliability_composite_noncircular" in ai_df.columns
               and ai_df["Reliability_composite_noncircular"].notna().any()
            else None
        ),
        "n_circular_cues": int(ai_df["Reliability_gra_is_circular"].sum()),
        "top10_by_WAI": ai_df.nlargest(10, "WeightedAttentionIndex")[
            ["cue_subtype", "cue_type", "count",
             "AttentionIndex", "WeightedAttentionIndex",
             "Reliability_gra", "Reliability_gra_is_circular",
             "Reliability_position", "Reliability_form",
             "dominant_gra_relation"]
        ].to_dict(orient="records"),
        "top10_by_AI": ai_df.nlargest(10, "AttentionIndex")[
            ["cue_subtype", "cue_type", "count",
             "AttentionIndex", "WeightedAttentionIndex",
             "Reliability_gra", "Reliability_gra_is_circular",
             "Reliability_position", "Reliability_form",
             "dominant_gra_relation"]
        ].to_dict(orient="records"),
        "top10_by_count": ai_df.nlargest(10, "count")[
            ["cue_subtype", "cue_type", "count",
             "AttentionIndex", "WeightedAttentionIndex",
             "Reliability_gra", "Reliability_gra_is_circular",
             "Reliability_position", "Reliability_form",
             "dominant_gra_relation"]
        ].to_dict(orient="records"),
        "cue_type_profile_top10": type_df.head(10).to_dict(orient="records"),
    }
    summary_json = out_dir / f"{args.language}_AI_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Wrote: {summary_json}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
