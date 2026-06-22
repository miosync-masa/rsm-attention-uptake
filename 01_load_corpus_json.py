"""
01_load_corpus_json.py  (v2 - schema-confirmed)
================================================
IMT Attention Bias Paper — Step 1: Corpus Loader (Chatter JSON edition)

Schema confirmed against Brown/Adam/020304.cha on 2026-06-13.

JSON structure (verified):
    {
      "lines": [
        {"line_type": "header", "header": {...}},
        {
          "line_type": "utterance",
          "main": {
            "speaker": "CHI",
            "content": {
              "content": [
                {"type": "word", "raw_text": "play", "cleaned_text": "play", ...},
                ...
              ],
              "terminator": {"type": "period" | "question" | "exclamation"}
            }
          },
          "dependent_tiers": [
            {"type": "Mor", "data": {"items": [{"main": {"pos": "verb", "lemma": "play", "features": [...]}}]}},
            {"type": "Gra", "data": {"relations": [{"index": 1, "head": 0, "relation": "ROOT"}, ...]}}
          ],
          "utterance_language": {"code": "eng"}
        }
      ]
    }

Header types observed: utf8, pid, begin, languages, participants, id.
The 'id' header carries per-speaker metadata (age, sex, group, ses, role).

Usage:
    python 01_load_corpus_json.py --corpus_path ~/childes_data/Brown \\
                                   --language English \\
                                   --output_dir ./output/ \\
                                   --convert

Author: Torami x Boss | IMT Attention project | 2026-06-13
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Role-based speaker classification (uses 'role' field, not speaker code)
# ─────────────────────────────────────────────────────────────────────────────

CHILD_ROLES = {"Target_Child", "Child"}
CAREGIVER_ROLES = {
    "Mother", "Father", "Parent",
    "Grandmother", "Grandfather", "Grandparent",
    "Aunt", "Uncle", "Sister", "Brother", "Sibling",
    "Adult", "Female_Adult", "Male_Adult",
    "Caretaker", "Babysitter",
}
INVESTIGATOR_ROLES = {"Investigator", "Researcher", "Observer"}


def classify_speaker(role: Optional[str]) -> str:
    if not role:
        return "other"
    if role in CHILD_ROLES:
        return "child"
    if role in CAREGIVER_ROLES:
        return "caregiver"
    if role in INVESTIGATOR_ROLES:
        return "investigator"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load CHILDES via Chatter JSON")
    p.add_argument("--corpus_path", default=None,
                   help="Directory of .cha files (with --convert).")
    p.add_argument("--json_path", default=None,
                   help="Directory of pre-converted JSON files.")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK",
                            "English-NA-Pool", "English-NewmanRatner",
                            "English-BernsteinRatner", "English-Tardif",
                            "English-Higginson", "English-Brent",
                            "English-Rollins", "English-Soderstrom",
                            "English-Manchester", "English-Providence",
                            "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--convert", action="store_true",
                   help="Run chatter to-json first.")
    p.add_argument("--chatter_bin", default="chatter")
    p.add_argument("--min_age_months", type=int, default=12)
    p.add_argument("--max_age_months", type=int, default=72)
    p.add_argument("--include_investigators", action="store_true",
                   help="Include investigator utterances (default: exclude).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Age parsing — CHAT format "Y;MM.DD" → total months
# ─────────────────────────────────────────────────────────────────────────────

def parse_chat_age(age_str: Optional[str]) -> Optional[float]:
    """E.g. '2;03.04' → 2*12 + 3 + 4/30.44 ≈ 27.13"""
    if not age_str or not isinstance(age_str, str):
        return None
    try:
        if ";" in age_str:
            y_part, rest = age_str.split(";", 1)
            years = int(y_part) if y_part.strip() else 0
        else:
            years, rest = 0, age_str
        if "." in rest:
            m_part, d_part = rest.split(".", 1)
            months = int(m_part) if m_part.strip() else 0
            days = int(d_part) if d_part.strip() else 0
        else:
            months = int(rest) if rest.strip() else 0
            days = 0
        return years * 12 + months + days / 30.44
    except (ValueError, IndexError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Header parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_headers(lines: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Walk header lines, build {speaker_code: {role, age_months, sex, ...}}"""
    speakers: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        if line.get("line_type") != "header":
            continue
        hdr = line.get("header", {})
        htype = hdr.get("type")

        if htype == "participants":
            for entry in hdr.get("entries", []):
                code = entry.get("speaker_code")
                if code:
                    speakers.setdefault(code, {})
                    speakers[code]["role"] = entry.get("role")
                    speakers[code]["name"] = entry.get("name")

        elif htype == "id":
            code = hdr.get("speaker")
            if code:
                speakers.setdefault(code, {})
                if hdr.get("role"):
                    speakers[code]["role"] = hdr.get("role")
                speakers[code]["sex"] = hdr.get("sex")
                speakers[code]["group"] = hdr.get("group")
                speakers[code]["ses"] = hdr.get("ses")
                speakers[code]["corpus"] = hdr.get("corpus")
                speakers[code]["language"] = hdr.get("language")
                age_raw = hdr.get("age")
                if age_raw:
                    speakers[code]["age_months"] = parse_chat_age(age_raw)
                    speakers[code]["age_raw"] = age_raw

    return speakers


# ─────────────────────────────────────────────────────────────────────────────
# Utterance parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_word_tokens(utt: Dict) -> List[Dict[str, Any]]:
    content = utt.get("main", {}).get("content", {}).get("content", [])
    return [
        {"raw_text": item.get("raw_text", ""), "cleaned_text": item.get("cleaned_text", "")}
        for item in content if item.get("type") == "word"
    ]


def extract_mor_items(utt: Dict) -> List[Dict[str, Any]]:
    for tier in utt.get("dependent_tiers", []):
        if tier.get("type") == "Mor":
            return tier.get("data", {}).get("items", [])
    return []


def extract_gra_relations(utt: Dict) -> List[Dict[str, Any]]:
    for tier in utt.get("dependent_tiers", []):
        if tier.get("type") == "Gra":
            return tier.get("data", {}).get("relations", [])
    return []


def get_terminator(utt: Dict) -> str:
    term = utt.get("main", {}).get("content", {}).get("terminator", {})
    return term.get("type", "") if isinstance(term, dict) else ""


def parse_utterance(
    utt: Dict, utt_idx: int, file_stem: str,
    speakers: Dict[str, Dict[str, Any]],
) -> Tuple[Dict, List[Dict]]:
    speaker_code = utt.get("main", {}).get("speaker", "")
    speaker_meta = speakers.get(speaker_code, {})
    speaker_role = classify_speaker(speaker_meta.get("role"))
    age_months = speaker_meta.get("age_months")

    words = extract_word_tokens(utt)
    mor_items = extract_mor_items(utt)
    gra_relations = extract_gra_relations(utt)
    terminator = get_terminator(utt)

    num_tokens = len(words)
    utterance_text = " ".join(w["cleaned_text"] for w in words if w["cleaned_text"])
    utt_lang = utt.get("utterance_language", {}).get("code", "")

    utt_row = {
        "file": file_stem,
        "speaker": speaker_code,
        "speaker_role": speaker_role,
        "role_raw": speaker_meta.get("role", ""),
        "age_months": age_months,
        "age_raw": speaker_meta.get("age_raw", ""),
        "utterance_index": utt_idx,
        "utterance_text": utterance_text,
        "num_tokens": num_tokens,
        "terminator": terminator,
        "utterance_language": utt_lang,
        "is_child": speaker_role == "child",
        "is_caregiver": speaker_role == "caregiver",
        "is_investigator": speaker_role == "investigator",
        "has_mor": len(mor_items) > 0,
        "has_gra": len(gra_relations) > 0,
    }

    # Gra lookup: word position (1-indexed) -> relation
    gra_by_index: Dict[int, Dict] = {
        rel.get("index", 0): rel for rel in gra_relations
    }

    token_rows = []
    for tok_idx, word in enumerate(words):
        mor_item = mor_items[tok_idx] if tok_idx < len(mor_items) else None
        mor_main = mor_item.get("main", {}) if mor_item else {}
        pos = mor_main.get("pos", "")
        lemma = mor_main.get("lemma", "")
        features = mor_main.get("features", [])

        # Gra: 1-indexed
        gra_entry = gra_by_index.get(tok_idx + 1, {})

        token_rows.append({
            "file": file_stem,
            "speaker": speaker_code,
            "speaker_role": speaker_role,
            "age_months": age_months,
            "utterance_index": utt_idx,
            "token_index": tok_idx,
            "raw_text": word["raw_text"],
            "cleaned_text": word["cleaned_text"],
            "pos": pos,
            "lemma": lemma,
            "features": "|".join(features) if features else "",
            "gra_relation": gra_entry.get("relation", ""),
            "gra_head": gra_entry.get("head", -1),
            "position_in_utterance": (
                tok_idx / max(num_tokens - 1, 1) if num_tokens > 1 else 0.5
            ),
            "is_utterance_initial": tok_idx == 0,
            "is_utterance_final": tok_idx == num_tokens - 1,
            "utterance_terminator": terminator,
        })

    return utt_row, token_rows


# ─────────────────────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────────────────────

def process_json_file(json_path: Path) -> Tuple[List[Dict], List[Dict]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  ERROR: bad JSON in {json_path.name}: {e}")
        return [], []

    lines = data.get("lines", [])
    speakers = parse_headers(lines)

    utt_rows: List[Dict] = []
    tok_rows: List[Dict] = []
    utt_idx = 0
    for line in lines:
        if line.get("line_type") != "utterance":
            continue
        u_row, t_rows = parse_utterance(line, utt_idx, json_path.stem, speakers)
        utt_rows.append(u_row)
        tok_rows.extend(t_rows)
        utt_idx += 1

    return utt_rows, tok_rows


# ─────────────────────────────────────────────────────────────────────────────
# Chatter conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_json(corpus_path: Path, output_dir: Path, chatter_bin: str) -> Path:
    json_root = output_dir / "json_cache" / corpus_path.name
    json_root.mkdir(parents=True, exist_ok=True)

    cha_files = sorted(corpus_path.rglob("*.cha"))
    print(f"Found {len(cha_files)} .cha files; converting via chatter...")

    converted = skipped = failed = 0
    for cha_file in tqdm(cha_files, desc="Converting"):
        rel = cha_file.relative_to(corpus_path)
        json_file = json_root / rel.with_suffix(".json")
        json_file.parent.mkdir(parents=True, exist_ok=True)

        if json_file.exists() and json_file.stat().st_size > 0:
            skipped += 1
            continue

        try:
            result = subprocess.run(
                [chatter_bin, "to-json", str(cha_file)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                failed += 1
                continue
            json_file.write_text(result.stdout, encoding="utf-8")
            converted += 1
        except (subprocess.TimeoutExpired, OSError):
            failed += 1
            continue

    print(f"  Converted: {converted} | Skipped: {skipped} | Failed: {failed}")
    print(f"  JSON output: {json_root}")
    return json_root


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.convert:
        if not args.corpus_path:
            print("ERROR: --convert requires --corpus_path"); sys.exit(2)
        json_root = convert_to_json(
            Path(args.corpus_path).expanduser(), out_dir, args.chatter_bin
        )
    elif args.json_path:
        json_root = Path(args.json_path).expanduser()
    else:
        print("ERROR: provide --corpus_path with --convert, or --json_path"); sys.exit(2)

    json_files = sorted(json_root.rglob("*.json"))
    print(f"\nProcessing {len(json_files)} JSON files...\n")

    all_utts: List[Dict] = []
    all_tokens: List[Dict] = []
    parse_errors = 0

    for jf in tqdm(json_files, desc="Parsing"):
        try:
            u_rows, t_rows = process_json_file(jf)
        except Exception as e:
            print(f"  ERROR in {jf.name}: {type(e).__name__}: {e}")
            parse_errors += 1
            continue

        def in_age(row):
            a = row["age_months"]
            return a is None or args.min_age_months <= a <= args.max_age_months

        u_rows = [u for u in u_rows if in_age(u)]
        t_rows = [t for t in t_rows if in_age(t)]

        if not args.include_investigators:
            u_rows = [u for u in u_rows if not u["is_investigator"]]
            t_rows = [t for t in t_rows if t["speaker_role"] != "investigator"]

        all_utts.extend(u_rows)
        all_tokens.extend(t_rows)

    utt_df = pd.DataFrame(all_utts)
    tok_df = pd.DataFrame(all_tokens)

    utt_csv = out_dir / f"{args.language}_utterances.csv"
    tok_csv = out_dir / f"{args.language}_tokens.csv"
    summary_json = out_dir / f"{args.language}_summary.json"

    utt_df.to_csv(utt_csv, index=False)
    tok_df.to_csv(tok_csv, index=False)

    # Summary
    summary = {
        "language": args.language,
        "n_json_files": len(json_files),
        "n_parse_errors": parse_errors,
        "n_utterances_total": len(utt_df),
        "n_utterances_child": int(utt_df["is_child"].sum()) if len(utt_df) else 0,
        "n_utterances_caregiver": int(utt_df["is_caregiver"].sum()) if len(utt_df) else 0,
        "n_tokens_total": len(tok_df),
        "n_utterances_with_mor": int(utt_df["has_mor"].sum()) if len(utt_df) else 0,
        "n_utterances_with_gra": int(utt_df["has_gra"].sum()) if len(utt_df) else 0,
    }

    if len(utt_df) and utt_df["age_months"].notna().any():
        summary["child_age_range_months"] = [
            float(utt_df["age_months"].min()),
            float(utt_df["age_months"].max()),
        ]
    else:
        summary["child_age_range_months"] = [None, None]

    if len(utt_df):
        cu = utt_df[utt_df["is_child"]]
        ca = utt_df[utt_df["is_caregiver"]]
        summary["mlu_child_mean"] = float(cu["num_tokens"].mean()) if len(cu) else None
        summary["mlu_caregiver_mean"] = float(ca["num_tokens"].mean()) if len(ca) else None

    if len(tok_df):
        summary["vocab_size_child"] = int(
            tok_df[tok_df["speaker_role"] == "child"]["lemma"].nunique()
        )
        summary["vocab_size_caregiver"] = int(
            tok_df[tok_df["speaker_role"] == "caregiver"]["lemma"].nunique()
        )
        ca_tok = tok_df[tok_df["speaker_role"] == "caregiver"]
        if len(ca_tok):
            summary["top10_pos_caregiver"] = (
                ca_tok["pos"].value_counts().head(10).to_dict()
            )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Utterances : {utt_csv}  ({len(utt_df):,} rows)")
    print(f"  Tokens     : {tok_csv}  ({len(tok_df):,} rows)")
    print(f"  Summary    : {summary_json}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
