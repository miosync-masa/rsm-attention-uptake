"""
17k_manchester_qualitative.py
=============================
IMT Attention Bias Paper 2 — Step 17k:
Quantitative extraction of qualitative features from raw .cha files.

Goal: identify the Theakston-lab Manchester corpus signature that escaped
the 17g protocol diff (which only used the JSON-cache). Raw .cha files
expose meta-data that the JSON loader does not retain — most importantly:

  @Situation               (Free Play, Mealtime, ...)
  @Activities              (the specific game / activity description)
  @Time Duration           (recording length)
  @Location                (home, lab, ...)
  @Types                   (e.g., "long, toyplay, TD")
  participant roles        (presence of Investigator vs naturalistic only)
  unintelligible markers   (xxx, yyy, www) — proxy for transcription
                            policy on unclear speech
  event annotations        (&-prefixed events, [+ I], [+ N], time codes)
  speech-act / phonological dependent tiers (%spa, %xpho, etc.)

For each sample of .cha files (per child, capped), we tally these
features and aggregate per sub-corpus (Brown / Manchester /
UK_long_long / UK_short_obs). Per-corpus median + Mann-Whitney
contrasts vs Manchester are reported.

────────────────────────────────────────────────────────────────────────────

Outputs (output/v17k/)
----------------------
* protocol_features_per_file.csv     one row per .cha file with extracted features
* activities_top_per_corpus.csv      top distinct @Activities values per corpus
* protocol_features_per_corpus.csv   per-corpus aggregate
* SUMMARY.md

Author: Torami x Boss | IMT Attention project | Paper 2 / Manchester qual v1 | 2026-06-21
"""

import argparse
import json
import re
import sys
from collections import Counter
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


CHILDES_DATA_ROOT = Path("~/childes_data").expanduser()

# Sub-corpus → list of (corpus_dir_relative_to_childes_data, child_subdir)
SUBCORPUS_MAP: Dict[str, List[Tuple[str, str]]] = {
    "Brown": [
        ("English", "Adam"), ("English", "Eve"), ("English", "Sarah"),
    ],
    "Manchester": [
        ("English-UK", c) for c in
        ["Anne", "Aran", "Becky", "Carl", "Dominic", "Gail",
         "Joel", "John", "Liz", "Ruth", "Warren"]
    ],
    "UK_long_long": [
        ("English-UK", c) for c in
        ["Thomas", "Fraser", "Helen", "Eleanor", "Nicole"]
    ],
    "UK_short_obs": [
        ("English-UK", c) for c in
        ["Abigail", "Frances", "Geoffrey", "Jack", "Jason", "Jonathan",
         "Laura", "Neville", "Penny", "Samantha", "Sean", "Stella"]
    ],
}


HEADER_RE          = re.compile(r"^@([A-Za-z][A-Za-z _]*):\s*(.*)$")
PARTICIPANTS_RE    = re.compile(r"^@Participants:\s*(.*)$")
UTTERANCE_RE       = re.compile(r"^\*([A-Z]{3,4}):")
DEP_TIER_RE        = re.compile(r"^%([a-z]+):")

# Inline content markers
XXX_RE   = re.compile(r"\bxxx\b")
YYY_RE   = re.compile(r"\byyy\b")
WWW_RE   = re.compile(r"\bwww\b")
AMP_RE   = re.compile(r"&[A-Za-z=~]")  # event/sound markers like &laughs &=cough &~no
TIMECODE_RE = re.compile(r"\b\d{3,7}_\d{3,7}\b")  # e.g., 18595_20396
PLUS_INFO_RE = re.compile(r"\[\+\s*[A-Z]\]")       # [+ I], [+ N] info-status codes


# ─────────────────────────────────────────────────────────────────────────────
# Parse a single .cha file
# ─────────────────────────────────────────────────────────────────────────────

def parse_cha(cha_path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = cha_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = text.splitlines()

    # Logical lines: CHA continuation: a line beginning with whitespace
    # continues the previous logical line. Build a logical-line list.
    logical: List[str] = []
    for line in lines:
        if line.startswith((" ", "\t")) and logical:
            logical[-1] += " " + line.strip()
        else:
            logical.append(line)

    headers: Dict[str, str] = {}
    n_utt = 0
    speakers: Counter = Counter()
    dep_tiers: Counter = Counter()
    n_xxx = 0
    n_yyy = 0
    n_www = 0
    n_amp = 0
    n_timecode = 0
    n_plus_info = 0
    participants_raw = ""

    last_speaker = None
    for line in logical:
        if line.startswith("@"):
            m = HEADER_RE.match(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                # Last-wins for repeated keys; @Comment etc. omitted from per-key
                if key not in {"Comment"}:
                    headers[key] = val
                if key == "Participants":
                    participants_raw = val
            continue
        if line.startswith("*"):
            m = UTTERANCE_RE.match(line)
            if m:
                n_utt += 1
                code = m.group(1)
                speakers[code] += 1
                last_speaker = code
                # Inline markers in the utterance body
                body = line.split(":", 1)[1] if ":" in line else ""
                if XXX_RE.search(body):
                    n_xxx += 1
                if YYY_RE.search(body):
                    n_yyy += 1
                if WWW_RE.search(body):
                    n_www += 1
                if AMP_RE.search(body):
                    n_amp += 1
                if TIMECODE_RE.search(body):
                    n_timecode += 1
                if PLUS_INFO_RE.search(body):
                    n_plus_info += 1
            continue
        if line.startswith("%"):
            m = DEP_TIER_RE.match(line)
            if m:
                dep_tiers[m.group(1)] += 1

    # Participants info: code -> role
    # Format: "CHI Target_Child, MOT Mother, INV Investigator"
    participants_codes: List[str] = []
    has_investigator = False
    roles: List[str] = []
    n_caregiver_codes = 0
    if participants_raw:
        for chunk in participants_raw.split(","):
            chunk = chunk.strip()
            parts = chunk.split(maxsplit=1)
            if not parts:
                continue
            code = parts[0]
            role = parts[1].lower() if len(parts) > 1 else ""
            participants_codes.append(code)
            roles.append(role)
            if "investigator" in role or "researcher" in role or "observer" in role:
                has_investigator = True
            if any(r in role for r in [
                "mother", "father", "parent", "grandmother", "grandfather",
                "aunt", "uncle", "sister", "brother", "sibling", "caretaker",
                "adult", "female_adult", "male_adult",
            ]):
                n_caregiver_codes += 1

    # Parse @Time Duration like "14:45-15:15"
    duration_min: Optional[float] = None
    td = headers.get("Time Duration") or headers.get("Time_Duration")
    if td and "-" in td:
        try:
            a, b = td.split("-", 1)
            def _to_min(s: str) -> float:
                h, m = s.strip().split(":")
                return int(h) * 60 + int(m)
            duration_min = float(_to_min(b) - _to_min(a))
        except Exception:
            duration_min = None

    return {
        "file":             cha_path.stem,
        "corpus_dir":       cha_path.parents[1].name,
        "child":            cha_path.parent.name,
        "n_utt":            n_utt,
        "n_speakers":       int(len(speakers)),
        "n_participants_declared": int(len(participants_codes)),
        "has_investigator": int(has_investigator),
        "n_caregiver_codes": int(n_caregiver_codes),
        "duration_minutes": duration_min,
        "Situation":        headers.get("Situation", ""),
        "Activities":       headers.get("Activities", ""),
        "Location":         headers.get("Location", ""),
        "Types":            headers.get("Types", ""),
        "Transcriber":      headers.get("Transcriber", ""),
        # Per-utterance rates
        "pct_xxx":          n_xxx / n_utt if n_utt else 0.0,
        "pct_yyy":          n_yyy / n_utt if n_utt else 0.0,
        "pct_www":          n_www / n_utt if n_utt else 0.0,
        "pct_amp_event":    n_amp / n_utt if n_utt else 0.0,
        "pct_timecode":     n_timecode / n_utt if n_utt else 0.0,
        "pct_plus_info":    n_plus_info / n_utt if n_utt else 0.0,
        # Speaker shares
        "pct_CHI":          speakers.get("CHI", 0) / n_utt if n_utt else 0.0,
        "pct_MOT":          speakers.get("MOT", 0) / n_utt if n_utt else 0.0,
        "pct_FAT":          speakers.get("FAT", 0) / n_utt if n_utt else 0.0,
        "pct_INV":          speakers.get("INV", 0) / n_utt if n_utt else 0.0,
        # Dep tier presence (per-utt rate)
        "pct_mor":          dep_tiers.get("mor", 0) / n_utt if n_utt else 0.0,
        "pct_gra":          dep_tiers.get("gra", 0) / n_utt if n_utt else 0.0,
        "pct_com":          dep_tiers.get("com", 0) / n_utt if n_utt else 0.0,
        "pct_sit":          dep_tiers.get("sit", 0) / n_utt if n_utt else 0.0,
        "pct_act":          dep_tiers.get("act", 0) / n_utt if n_utt else 0.0,
        "pct_exp":          dep_tiers.get("exp", 0) / n_utt if n_utt else 0.0,
        "pct_spa":          dep_tiers.get("spa", 0) / n_utt if n_utt else 0.0,
        "pct_xpho":         dep_tiers.get("xpho", 0) / n_utt if n_utt else 0.0,
        "pct_err":          dep_tiers.get("err", 0) / n_utt if n_utt else 0.0,
        "pct_add":          dep_tiers.get("add", 0) / n_utt if n_utt else 0.0,
        "n_distinct_dep_tiers": int(len(dep_tiers)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + comparisons
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_METRICS = [
    "n_utt", "n_speakers", "n_participants_declared",
    "has_investigator", "n_caregiver_codes", "duration_minutes",
    "pct_xxx", "pct_yyy", "pct_www", "pct_amp_event",
    "pct_timecode", "pct_plus_info",
    "pct_CHI", "pct_MOT", "pct_FAT", "pct_INV",
    "pct_mor", "pct_gra",
    "pct_com", "pct_sit", "pct_act", "pct_exp",
    "pct_spa", "pct_xpho", "pct_err", "pct_add",
    "n_distinct_dep_tiers",
]


def collect_files(subcorpus: str, entries: List[Tuple[str, str]],
                   sample_per_child: int) -> List[Path]:
    files: List[Path] = []
    for corpus_dir, child in entries:
        d = CHILDES_DATA_ROOT / corpus_dir / child
        if not d.exists():
            continue
        c = sorted(d.rglob("*.cha"))
        if sample_per_child > 0:
            c = c[: sample_per_child]
        files.extend(c)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Manchester qualitative diff from raw .cha files.")
    parser.add_argument("--sample_per_child", type=int, default=10,
                         help="Per-child cap to bound runtime; 0 = no cap.")
    parser.add_argument("--output_dir", default="./output/v17k")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for subcorpus, entries in SUBCORPUS_MAP.items():
        files = collect_files(subcorpus, entries, args.sample_per_child)
        print(f"  {subcorpus:<14}: {len(files)} files from {len(entries)} children")
        for f in tqdm(files, desc=f"  {subcorpus}"):
            r = parse_cha(f)
            if r is None:
                continue
            r["subcorpus"] = subcorpus
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("ERROR: no .cha files parsed.")
    per_file_csv = out_dir / "protocol_features_per_file.csv"
    df.to_csv(per_file_csv, index=False)
    print(f"\n  → {per_file_csv}  ({len(df):,} rows)")

    # Per-corpus aggregate (median, mean, sd)
    agg = df.groupby("subcorpus")[NUMERIC_METRICS].agg(["median", "mean", "std"])
    agg_csv = out_dir / "protocol_features_per_corpus.csv"
    agg.to_csv(agg_csv)
    print(f"  → {agg_csv}")

    # Top distinct @Activities per corpus
    print("\n  Top 8 distinct @Activities values per corpus:")
    activities_rows: List[Dict[str, Any]] = []
    for sc, g in df.groupby("subcorpus"):
        counts = g["Activities"].fillna("").astype(str).str.strip().value_counts()
        counts = counts[counts.index != ""]
        print(f"\n    [{sc}]  unique = {len(counts)}, top:")
        for activity, cnt in counts.head(8).items():
            print(f"      {cnt:>4}  {activity[:80]}")
            activities_rows.append({"subcorpus": sc, "activity": activity, "count": int(cnt)})
        # Save full list
        if len(counts) > 8:
            print(f"      ... and {len(counts) - 8} more")
    activities_df = pd.DataFrame(activities_rows)
    activities_df.to_csv(out_dir / "activities_top_per_corpus.csv", index=False)
    print(f"\n  → {out_dir / 'activities_top_per_corpus.csv'}")

    # Top distinct @Situation per corpus
    print("\n  Top 5 @Situation per corpus:")
    for sc, g in df.groupby("subcorpus"):
        counts = g["Situation"].fillna("").astype(str).str.strip().value_counts()
        counts = counts[counts.index != ""]
        top = counts.head(5)
        print(f"    [{sc}]  unique = {len(counts)}, top: {top.to_dict()}")

    # Manchester vs others: Mann-Whitney on key metrics
    print("\n  Mann-Whitney pairwise (Manchester vs others):")
    contrasts = ["Brown", "UK_long_long", "UK_short_obs"]
    pairwise: Dict[str, Dict[str, Dict[str, float]]] = {}
    print(f"  {'metric':<28} {'Manc med':>10} {'vs Brown':>14} {'vs UK_LL':>14} {'vs UK_SO':>14}")
    print('  ' + '-' * 80)
    manc = df[df["subcorpus"] == "Manchester"]
    for m in NUMERIC_METRICS:
        manc_vals = manc[m].dropna().values
        manc_med = float(np.median(manc_vals)) if len(manc_vals) else float("nan")
        per_metric: Dict[str, Dict[str, float]] = {"manchester_median": manc_med}
        line = f"  {m:<28} {manc_med:>10.3f}"
        for other in contrasts:
            oth = df[df["subcorpus"] == other][m].dropna().values
            if len(manc_vals) < 5 or len(oth) < 5:
                line += f" {'n/a':>14}"
                per_metric[other] = {"p": float("nan")}
                continue
            try:
                U, p = mannwhitneyu(manc_vals, oth, alternative="two-sided")
            except Exception:
                U, p = float("nan"), float("nan")
            per_metric[other] = {"U": float(U), "p": float(p),
                                   "other_median": float(np.median(oth))}
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('†' if p<0.10 else '')))
            line += f"  p={p:.3f}{sig:<3}({np.median(oth):>+5.2f})"
        pairwise[m] = per_metric
        print(line)

    json_path = out_dir / "protocol_features_pairwise.json"
    json_path.write_text(json.dumps(pairwise, indent=2), encoding="utf-8")
    print(f"\n  → {json_path}")

    # SUMMARY.md
    lines: List[str] = []
    lines.append(f"# 17k Manchester qualitative deep dive — sample {args.sample_per_child} files/child\n")
    lines.append("## Per-corpus medians (raw .cha features)\n")
    lines.append("| Metric | Brown | Manchester | UK_long_long | UK_short_obs |")
    lines.append("|---|---|---|---|---|")
    med = df.groupby("subcorpus")[NUMERIC_METRICS].median()
    for m in NUMERIC_METRICS:
        row = "| " + m + " |"
        for c in ["Brown", "Manchester", "UK_long_long", "UK_short_obs"]:
            v = med.loc[c, m] if c in med.index else float("nan")
            row += f" {v:.3f} |"
        lines.append(row)
    lines.append("\n## Mann-Whitney (Manchester vs others)\n")
    lines.append("| Metric | vs Brown | vs UK_long_long | vs UK_short_obs |")
    lines.append("|---|---|---|---|")
    for m in NUMERIC_METRICS:
        row = f"| {m} |"
        for c in ["Brown", "UK_long_long", "UK_short_obs"]:
            p = pairwise.get(m, {}).get(c, {}).get("p")
            if p is None or (isinstance(p, float) and np.isnan(p)):
                row += " n/a |"
            else:
                sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('†' if p<0.10 else '')))
                row += f" {p:.4f}{sig} |"
        lines.append(row)
    lines.append("\n## Top @Activities per corpus\n")
    for sc, g in df.groupby("subcorpus"):
        counts = g["Activities"].fillna("").astype(str).str.strip().value_counts()
        counts = counts[counts.index != ""]
        lines.append(f"\n### {sc}  (unique values = {len(counts)})\n")
        for activity, cnt in counts.head(8).items():
            lines.append(f"- {cnt}: {activity}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
