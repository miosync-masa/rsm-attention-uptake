"""
16_episode_outcome_uptake.py
============================
IMT Attention Bias Paper 2 — Step 16: episode-level uptake outcome

For each (episode, target_cue) pair, compute whether the child reuses the
SAME cue within the next N child utterances in the same transcript file.

This produces an episode-varying outcome required for Spec C (Mundlak
within-effect test) in 17_within_effect_test.py.

────────────────────────────────────────────────────────────────────────────

Definitions
-----------
* "Episode" = a child utterance with one or more cue tokens (as built by
  10_extract_R_plus_v2.py). The episode row carries pipe-delimited
  `child_cues`.
* "Target cue" = each individual cue in `child_cues` (after explosion).
* "Next N child utterances" = the next N child utterances in the same
  file that have utterance_index strictly greater than the current
  episode's child_utt_idx, ordered by utterance_index. We use the
  tokens_tagged.csv to enumerate child utterances and their cue sets,
  so cue-free child utterances still count toward the window.
* `next_N_reuse` = 1 if target_cue appears in any of those N child
  utterances' cue sets, else 0. If fewer than N subsequent child
  utterances exist in the file, the available ones are used and the
  flag `truncated_window` is set.

Output
------
Per corpus:
  output/v16/{lang}_episodes_with_reuse.csv
    columns:
      file, child_utt_idx, child_age_months,
      cue_subtype, r_plus_label, r_plus_composite,
      next_5_reuse, next_5_observed,
      next_10_reuse, next_10_observed,
      truncated_window_5, truncated_window_10

Also a summary JSON with reuse base-rates.

Usage
-----
  python 16_episode_outcome_uptake.py \\
      --episodes_csv  ./output/v10b/English_r_plus_episodes.csv \\
      --tagged_csv    ./output/English_tokens_tagged.csv \\
      --language      English \\
      --output_dir    ./output/v16 \\
      --windows       5,10

  python 16_episode_outcome_uptake.py --batch --output_dir ./output/v16

Author: Torami x Boss | IMT Attention project | Paper 2 / episode outcome v1 | 2026-06-21
"""

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install pandas numpy tqdm")
    sys.exit(1)


CANONICAL_CORPORA: Dict[str, Dict[str, str]] = {
    "English": {
        "label":          "Brown",
        "episodes_csv":   "./output/v10b/English_r_plus_episodes.csv",
        "tagged_csv":     "./output/English_tokens_tagged.csv",
    },
    "English-Manchester": {
        "label":          "Manchester",
        "episodes_csv":   "./output/v10b/English-Manchester_r_plus_episodes.csv",
        "tagged_csv":     "./output/English-Manchester_tokens_tagged.csv",
    },
    "English-UK": {
        "label":          "English-UK",
        "episodes_csv":   "./output/v10b/English-UK_r_plus_episodes.csv",
        "tagged_csv":     "./output/English-UK_tokens_tagged.csv",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Build per-file child utterance index
# ─────────────────────────────────────────────────────────────────────────────

def build_child_utt_index(tagged_csv: Path) -> Dict[str, Tuple[np.ndarray, List[set]]]:
    """
    Returns dict: file_stem -> (sorted_utt_idx_array, list_of_cue_sets).
    list_of_cue_sets[i] is the set of cue_subtypes used by the child
    in the utterance at sorted_utt_idx_array[i].
    Utterances without any cue still appear with an empty set, so the
    next-N window counts them as cue-free turns.
    """
    print(f"  Loading tagged tokens: {tagged_csv}")
    cols = ["file", "speaker_role", "utterance_index", "cue_subtype", "is_cue_token"]
    df = pd.read_csv(tagged_csv, usecols=cols, low_memory=False)
    print(f"    rows: {len(df):,}")

    df = df[df["speaker_role"] == "child"].copy()
    print(f"    child rows: {len(df):,}")

    # Build (file, utt_idx) -> set of cues
    cue_rows = df[df["is_cue_token"].astype(str) == "True"][
        ["file", "utterance_index", "cue_subtype"]
    ].copy()
    cue_rows["cue_subtype"] = cue_rows["cue_subtype"].fillna("").astype(str).str.strip()
    cue_rows = cue_rows[cue_rows["cue_subtype"] != ""]

    cues_per_utt: Dict[Tuple[str, int], set] = {}
    for f, idx, cue in cue_rows.itertuples(index=False, name=None):
        cues_per_utt.setdefault((f, int(idx)), set()).add(cue)

    # All child utterance indices per file
    utts = df[["file", "utterance_index"]].drop_duplicates().copy()
    utts["utterance_index"] = utts["utterance_index"].astype(int)

    file_index: Dict[str, Tuple[np.ndarray, List[set]]] = {}
    for f, g in utts.groupby("file"):
        idxs = np.sort(g["utterance_index"].values.astype(int))
        cue_sets = [cues_per_utt.get((f, int(i)), set()) for i in idxs]
        file_index[str(f)] = (idxs, cue_sets)

    print(f"    files indexed: {len(file_index):,}")
    return file_index


# ─────────────────────────────────────────────────────────────────────────────
# Compute next-N reuse for each (episode, cue)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reuse_rows(
    episodes: pd.DataFrame,
    file_index: Dict[str, Tuple[np.ndarray, List[set]]],
    windows: List[int],
) -> pd.DataFrame:
    """
    Iterate over episodes, explode child_cues, look ahead in the
    per-file child-utterance sequence for each window size, and record
    whether the target cue appears in the next N child utterances.
    """
    max_window = max(windows)
    out_rows: List[Dict] = []

    for row in tqdm(episodes.itertuples(index=False), total=len(episodes),
                    desc="Computing next-N reuse"):
        f = str(row.file)
        utt_idx = int(row.child_utt_idx)
        cues_str = "" if pd.isna(row.child_cues) else str(row.child_cues)
        target_cues = [c.strip() for c in cues_str.split("|") if c.strip()]
        if not target_cues:
            continue

        entry = file_index.get(f)
        if entry is None:
            continue
        idxs, cue_sets = entry

        # Position of the first child utterance with utterance_index > utt_idx
        pos = bisect.bisect_right(idxs, utt_idx)
        if pos >= len(idxs):
            # No further child utterances in this file — record truncated entries
            for c in target_cues:
                out = {
                    "file":               f,
                    "child_utt_idx":      utt_idx,
                    "child_age_months":   row.child_age_months,
                    "cue_subtype":        c,
                    "r_plus_label":       row.r_plus_label,
                    "r_plus_composite":   row.r_plus_composite,
                }
                for w in windows:
                    out[f"next_{w}_reuse"]      = 0
                    out[f"next_{w}_observed"]   = 0
                    out[f"truncated_window_{w}"] = 1
                out_rows.append(out)
            continue

        # Pre-compute window cue sets up to max_window
        end_max = min(pos + max_window, len(idxs))
        window_cue_sets = cue_sets[pos:end_max]

        for c in target_cues:
            out = {
                "file":               f,
                "child_utt_idx":      utt_idx,
                "child_age_months":   row.child_age_months,
                "cue_subtype":        c,
                "r_plus_label":       row.r_plus_label,
                "r_plus_composite":   row.r_plus_composite,
            }
            for w in windows:
                obs = min(w, len(window_cue_sets))
                window_slice = window_cue_sets[:obs]
                reused = any(c in s for s in window_slice)
                out[f"next_{w}_reuse"]      = int(reused)
                out[f"next_{w}_observed"]   = obs
                out[f"truncated_window_{w}"] = int(obs < w)
            out_rows.append(out)

    return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus driver
# ─────────────────────────────────────────────────────────────────────────────

def process_corpus(language: str, episodes_csv: Path, tagged_csv: Path,
                   windows: List[int], output_dir: Path) -> Dict:
    print(f"\n=== {language} ===")
    file_index = build_child_utt_index(tagged_csv)

    print(f"  Loading episodes: {episodes_csv}")
    eps_cols = ["file", "child_utt_idx", "child_age_months",
                "child_cues", "r_plus_label", "r_plus_composite"]
    eps = pd.read_csv(episodes_csv, usecols=eps_cols, low_memory=False)
    print(f"    episodes: {len(eps):,}")

    expanded = compute_reuse_rows(eps, file_index, windows)
    print(f"  Expanded rows (episode × cue): {len(expanded):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{language}_episodes_with_reuse.csv"
    expanded.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")

    summary: Dict[str, object] = {
        "language":              language,
        "n_expanded_rows":       int(len(expanded)),
        "n_unique_cues":         int(expanded["cue_subtype"].nunique()) if len(expanded) else 0,
        "windows":               windows,
    }
    for w in windows:
        col = f"next_{w}_reuse"
        trunc = f"truncated_window_{w}"
        observed = expanded[expanded[trunc] == 0]
        summary[f"reuse_rate_next_{w}_full"]      = float(observed[col].mean()) if len(observed) else None
        summary[f"reuse_rate_next_{w}_all"]       = float(expanded[col].mean()) if len(expanded) else None
        summary[f"n_truncated_window_{w}"]        = int(expanded[trunc].sum())

    contingent = expanded[expanded["r_plus_label"] != "no_contingent_response"].copy()
    summary["n_contingent_rows"] = int(len(contingent))
    for w in windows:
        col = f"next_{w}_reuse"
        summary[f"contingent_reuse_rate_next_{w}"] = (
            float(contingent[col].mean()) if len(contingent) else None
        )

    out_json = output_dir / f"{language}_reuse_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out_json}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute next-N-utterance cue reuse outcome.")
    p.add_argument("--episodes_csv", default=None)
    p.add_argument("--tagged_csv",   default=None)
    p.add_argument("--language",     default=None)
    p.add_argument("--output_dir",   default="./output/v16")
    p.add_argument("--windows",      default="5,10",
                   help="Comma-separated list of N values for next-N-reuse.")
    p.add_argument("--batch", action="store_true",
                   help="Run all canonical corpora (Brown/Manchester/English-UK).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    output_dir = Path(args.output_dir).expanduser()

    if args.batch:
        targets = [
            (lang, Path(cfg["episodes_csv"]), Path(cfg["tagged_csv"]))
            for lang, cfg in CANONICAL_CORPORA.items()
        ]
    else:
        if not (args.language and args.episodes_csv and args.tagged_csv):
            sys.exit("ERROR: provide --language, --episodes_csv, --tagged_csv (or --batch).")
        targets = [(args.language, Path(args.episodes_csv), Path(args.tagged_csv))]

    all_summaries: List[Dict] = []
    for lang, ep_path, tg_path in targets:
        if not ep_path.exists():
            print(f"  SKIP {lang}: episodes CSV missing ({ep_path})")
            continue
        if not tg_path.exists():
            print(f"  SKIP {lang}: tagged CSV missing ({tg_path})")
            continue
        s = process_corpus(lang, ep_path, tg_path, windows, output_dir)
        all_summaries.append(s)

    if all_summaries:
        index_json = output_dir / "reuse_index.json"
        with open(index_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n  → {index_json}")


if __name__ == "__main__":
    main()
