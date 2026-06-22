"""
17i_corpus_protocol_diff.py
===========================
IMT Attention Bias Paper 2 — Step 17i: Corpus-protocol qualitative diff.

After 17g/17h ruled out window, MLU, density, and cue mix as explanations
for Manchester's exposure-gate null, the remaining suspect is the
recording / transcription / coding protocol itself. This script extracts
quantitative features from a sample of CHA files (via their cached JSON
output from 01_load_corpus_json.py) and compares Manchester to its UK
neighbors (Wells-style and Thomas-style) and to Brown.

The metrics are coarse but corpus-distinctive:

  Per JSON file:
    * n_utterances, n_child_utts, n_caregiver_utts
    * caregiver_to_child_ratio
    * % utterances with MOR tier present
    * % utterances with GRA tier present
    * % utterances with at least one non-Mor/Gra dependent tier
      ("rich annotation" — %com, %sit, %act, %pho, etc.)
    * utterance terminator distribution (period / question / exclamation / ...)
    * mean and median num_tokens (utterance length)
    * fraction of zero-token (= event-only) utterances
    * % of word tokens whose raw_text contains a CHAT control character
      (0, &, +, [, /, <, >, x, ...) — proxy for transcription complexity
    * header field types present (begin, age, sex, group, ses, role, ...)

The script then aggregates within each sub-corpus (Manchester /
UK_short_obs / UK_long_long_Thomas / UK_long_long_Wells-Bristol /
Brown / NA-Pool-other longitudinals if available) and reports group-level
means / sds plus a Mann-Whitney comparison for each metric.

────────────────────────────────────────────────────────────────────────────

Output (output/v17i/)
---------------------
* protocol_diff_per_file.csv     per-file feature table
* protocol_diff_per_corpus.csv   per-corpus aggregate
* protocol_diff_pairwise.json    Mann-Whitney pairwise tests for the
                                  key contrasts (Manchester vs each other)
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / protocol diff v1 | 2026-06-21
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


# Children whose JSON-cache lives under each canonical pool, mapped to a
# sub-corpus label. This drives both the file iteration and the labeling.
SUBCORPUS_MAP: Dict[str, List[Tuple[str, str]]] = {
    # (pool, json_cache_dir)
    "Brown": [
        ("English", "Adam"), ("English", "Eve"), ("English", "Sarah"),
    ],
    "Manchester": [
        ("English-Manchester", "Anne"), ("English-Manchester", "Aran"),
        ("English-Manchester", "Becky"), ("English-Manchester", "Carl"),
        ("English-Manchester", "Dominic"), ("English-Manchester", "Gail"),
        ("English-Manchester", "Joel"),  ("English-Manchester", "John"),
        ("English-Manchester", "Liz"),   ("English-Manchester", "Ruth"),
        ("English-Manchester", "Warren"),
    ],
    "UK_long_long": [
        ("English-UK", "Thomas"),   ("English-UK", "Fraser"),
        ("English-UK", "Helen"),    ("English-UK", "Eleanor"),
        ("English-UK", "Nicole"),
    ],
    "UK_short_obs": [
        ("English-UK", "Abigail"), ("English-UK", "Frances"),
        ("English-UK", "Geoffrey"),("English-UK", "Jack"),
        ("English-UK", "Jason"),   ("English-UK", "Jonathan"),
        ("English-UK", "Laura"),   ("English-UK", "Neville"),
        ("English-UK", "Penny"),   ("English-UK", "Samantha"),
        ("English-UK", "Sean"),    ("English-UK", "Stella"),
    ],
}

JSON_CACHE_ROOT = Path("./output/json_cache")

# Regex used to flag CHAT control characters inside raw token text.
CHAT_CONTROL = re.compile(r"[0&+\[\]/<>()\\xX:]")


# ─────────────────────────────────────────────────────────────────────────────
# Per-file feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    lines = data.get("lines", [])

    header_types: List[str] = []
    n_utt = 0
    n_child = 0
    n_caregiver = 0
    n_other = 0
    n_with_mor = 0
    n_with_gra = 0
    n_with_rich_dep = 0
    n_zero_tokens = 0
    terminator_counts: Dict[str, int] = {}
    token_lens: List[int] = []
    n_total_word_tokens = 0
    n_control_word_tokens = 0
    speaker_role_counter: Dict[str, int] = {}

    # First pass — header types
    for line in lines:
        if line.get("line_type") == "header":
            h = line.get("header", {})
            if h.get("type"):
                header_types.append(h["type"])

    # Build speaker_code → role map from "id" headers
    speaker_role: Dict[str, str] = {}
    for line in lines:
        if line.get("line_type") != "header":
            continue
        h = line.get("header", {})
        if h.get("type") == "id":
            code = h.get("speaker")
            role = (h.get("role") or "").lower()
            if code and role:
                speaker_role[code] = role

    CHILD_ROLES = {"target_child", "child"}
    CAREGIVER_ROLES = {
        "mother", "father", "parent", "grandmother", "grandfather",
        "grandparent", "aunt", "uncle", "sister", "brother", "sibling",
        "adult", "female_adult", "male_adult", "caretaker", "babysitter",
    }

    for line in lines:
        if line.get("line_type") != "utterance":
            continue
        n_utt += 1
        main = line.get("main", {})
        speaker = main.get("speaker", "")
        role = speaker_role.get(speaker, "").lower()
        if role in CHILD_ROLES:
            n_child += 1
        elif role in CAREGIVER_ROLES:
            n_caregiver += 1
        else:
            n_other += 1
        speaker_role_counter[role or "?"] = speaker_role_counter.get(role or "?", 0) + 1

        content = main.get("content", {})
        words = [w for w in content.get("content", []) if w.get("type") == "word"]
        if not words:
            n_zero_tokens += 1
        token_lens.append(len(words))
        for w in words:
            n_total_word_tokens += 1
            raw = str(w.get("raw_text", ""))
            if CHAT_CONTROL.search(raw):
                n_control_word_tokens += 1

        term = content.get("terminator", {})
        tterm = term.get("type", "") if isinstance(term, dict) else ""
        terminator_counts[tterm] = terminator_counts.get(tterm, 0) + 1

        # Dependent tier presence
        dep = line.get("dependent_tiers") or []
        types = {t.get("type") for t in dep}
        if "Mor" in types:
            n_with_mor += 1
        if "Gra" in types:
            n_with_gra += 1
        if any(t not in {"Mor", "Gra"} for t in types if t):
            n_with_rich_dep += 1

    if n_utt == 0:
        return None
    return {
        "file":             json_path.stem,
        "n_utt":            n_utt,
        "n_child":          n_child,
        "n_caregiver":      n_caregiver,
        "n_other":          n_other,
        "caregiver_to_child_ratio": (n_caregiver / n_child) if n_child else float("nan"),
        "pct_mor":          n_with_mor / n_utt,
        "pct_gra":          n_with_gra / n_utt,
        "pct_rich_dep":     n_with_rich_dep / n_utt,
        "pct_zero_tokens":  n_zero_tokens / n_utt,
        "mean_tokens":      float(np.mean(token_lens)) if token_lens else float("nan"),
        "median_tokens":    float(np.median(token_lens)) if token_lens else float("nan"),
        "n_total_word_tokens":    int(n_total_word_tokens),
        "pct_control_word_tokens": (
            n_control_word_tokens / n_total_word_tokens if n_total_word_tokens else float("nan")
        ),
        "pct_period":       terminator_counts.get("period", 0) / n_utt,
        "pct_question":     terminator_counts.get("question", 0) / n_utt,
        "pct_exclamation":  terminator_counts.get("exclamation", 0) / n_utt,
        "pct_other_term":   sum(v for k, v in terminator_counts.items()
                                 if k not in {"period", "question", "exclamation"}) / n_utt,
        "header_types":     "|".join(sorted(set(header_types))),
        "n_distinct_header_types": len(set(header_types)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-protocol qualitative diff.")
    parser.add_argument("--output_dir", default="./output/v17i")
    parser.add_argument("--sample_per_child", type=int, default=10,
                         help="Cap per child to bound runtime; 0 = no cap.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for subcorpus, entries in SUBCORPUS_MAP.items():
        for pool, child in entries:
            cache_dir = JSON_CACHE_ROOT / pool / child
            if not cache_dir.exists():
                print(f"  SKIP {pool}/{child} — not found")
                continue
            files = sorted(cache_dir.rglob("*.json"))
            if args.sample_per_child > 0:
                files = files[: args.sample_per_child]
            print(f"  {subcorpus:<14} | {pool}/{child:<10} ({len(files)} files)")
            for jf in files:
                feats = extract_features(jf)
                if feats is None:
                    continue
                feats["subcorpus"] = subcorpus
                feats["pool"]      = pool
                feats["child"]     = child
                rows.append(feats)

    if not rows:
        sys.exit("ERROR: no features extracted.")
    per_file_df = pd.DataFrame(rows)
    per_file_csv = out_dir / "protocol_diff_per_file.csv"
    per_file_df.to_csv(per_file_csv, index=False)
    print(f"\n  → {per_file_csv}  ({len(per_file_df):,} rows)")

    # Per-corpus aggregate
    metrics = [
        "n_utt", "n_child", "n_caregiver", "caregiver_to_child_ratio",
        "pct_mor", "pct_gra", "pct_rich_dep",
        "pct_zero_tokens", "mean_tokens",
        "pct_control_word_tokens",
        "pct_period", "pct_question", "pct_exclamation", "pct_other_term",
        "n_distinct_header_types",
    ]
    agg = per_file_df.groupby("subcorpus")[metrics].agg(["mean", "median", "std"])
    agg_csv = out_dir / "protocol_diff_per_corpus.csv"
    agg.to_csv(agg_csv)
    print(f"  → {agg_csv}")

    # Pairwise Manchester vs others, per metric
    contrasts = ["Brown", "UK_long_long", "UK_short_obs"]
    pairwise: Dict[str, Dict[str, Dict[str, float]]] = {}
    print(f"\n  Mann-Whitney pairwise: Manchester vs others")
    print(f"  {'metric':<32} {'Manc median':>12} {'vs Brown':>10} {'vs UK_LL':>10} {'vs UK_SO':>10}")
    print("  " + "-" * 80)
    manc = per_file_df[per_file_df["subcorpus"] == "Manchester"]
    for m in metrics:
        manc_vals = manc[m].dropna().values
        manc_med = float(np.median(manc_vals)) if len(manc_vals) else float("nan")
        per_metric: Dict[str, Dict[str, float]] = {"manchester_median": manc_med}
        col_str = f"  {m:<32} {manc_med:>12.3f}"
        for other in contrasts:
            oth = per_file_df[per_file_df["subcorpus"] == other]
            o_vals = oth[m].dropna().values
            if len(manc_vals) < 5 or len(o_vals) < 5:
                col_str += f" {'n/a':>10}"
                per_metric[other] = {"p": float("nan"), "other_median": float(np.median(o_vals)) if len(o_vals) else float("nan")}
                continue
            try:
                U, p = mannwhitneyu(manc_vals, o_vals, alternative="two-sided")
            except Exception:
                U, p = float("nan"), float("nan")
            per_metric[other] = {
                "U":             float(U),
                "p":             float(p),
                "other_median":  float(np.median(o_vals)),
            }
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('†' if p<0.10 else '')))
            col_str += f"  p={p:.3f}{sig:<3}"
        pairwise[m] = per_metric
        print(col_str)

    out_json = out_dir / "protocol_diff_pairwise.json"
    out_json.write_text(json.dumps(pairwise, indent=2), encoding="utf-8")
    print(f"\n  → {out_json}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append("# 17i corpus-protocol diff (Manchester vs others)\n")
    lines.append(f"\nSample: up to {args.sample_per_child} files per child "
                  f"(or all if smaller).\n")
    lines.append("\n## Per-corpus medians\n")
    lines.append("| Metric | Brown | Manchester | UK_long_long | UK_short_obs |")
    lines.append("|---|---|---|---|---|")
    med = per_file_df.groupby("subcorpus")[metrics].median()
    for m in metrics:
        row = "| " + m + " |"
        for c in ["Brown", "Manchester", "UK_long_long", "UK_short_obs"]:
            v = med.loc[c, m] if c in med.index else float("nan")
            row += f" {v:.3f} |"
        lines.append(row)
    lines.append("\n## Manchester vs others (Mann-Whitney p)\n")
    lines.append("| Metric | vs Brown | vs UK_long_long | vs UK_short_obs |")
    lines.append("|---|---|---|---|")
    for m in metrics:
        row = f"| {m} |"
        for c in ["Brown", "UK_long_long", "UK_short_obs"]:
            p = pairwise.get(m, {}).get(c, {}).get("p")
            if p is None or (isinstance(p, float) and np.isnan(p)):
                row += " n/a |"
            else:
                sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('†' if p<0.10 else '')))
                row += f" {p:.4f}{sig} |"
        lines.append(row)
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
