"""
10_extract_R_plus_v2.py
=======================
IMT Attention Bias Paper — Step 10 (v2): Response-Contingent Input Extraction (Paper 2)

For each child utterance containing a grammatical cue attempt, identifies the
immediately following caregiver turn (response-contingency window) and codes it
along two parallel scales:

  R+_depth (ordinal 0-4) — for clinical interpretation
      0 = no contingent linguistic response / topic shift
      1 = acknowledgment
      2 = repetition / confirmation
      3 = expansion
      4 = recast / targeted reformulation

  R+_composite (continuous, range [-1.0, +1.0]) — Paper 2 PRIMARY predictor
      = 0.25 * lexical_overlap_containment
      + 0.25 * target_cue_modeled
      + 0.25 * morphosyntactic_change_flag
      + 0.25 * expansion_depth_normalized
      - 1.00 * repair_flag

────────────────────────────────────────────────────────────────────────────
CHANGES FROM v1 (data-driven calibration on Brown English pilot, n=24,757)
────────────────────────────────────────────────────────────────────────────

Pilot diagnostic surfaced three issues, each addressed below:

  v2-Fix A: 'what' false-positive repair (2,844 episodes, 11.5% of contingent)
    The v1 lexicon match treated 'what is that' / 'what's that' as repair, because
    the lexicon contained 'what' and matching was substring-based after punctuation
    strip. These are wh-question expansions, not repair requests.

    v2 separates `has_repair_only`: a caregiver utterance is treated as repair ONLY
    when it is short (≤ args.repair_max_tokens, default 3) AND contains a
    repair-lexicon item. 'what is that' (3 tokens, but with content beyond 'what')
    no longer flags as repair. The lexicon itself is unchanged.

  v2-Fix B: morph_change threshold too low
    v1 flagged morphosyntactic change at >=1 added feature, causing 72.7% of
    contingent episodes to be 'morph_change=True'. Pilot percentiles showed
    p75=5 features added; p50=2.

    v2 makes the threshold a CLI parameter (--morph_change_threshold, default=5)
    so that recast = "substantial reformulation" (≥5 added morph features) and
    expansion = "moderate reformulation" (≥1 but <5).

  v2-Fix C: Step 7 (expansion) overly restrictive
    v1 required NOT morphosyntactic_change for expansion, which combined with
    Fix B's old threshold left only 111 expansion episodes (0.4% of contingent).

    v2 drops that condition. Step 8 (recast) now claims episodes with
    morph_added >= morph_change_threshold; Step 7 (expansion) takes the rest of
    the contingent-and-cue-modeled episodes. Step ordering preserves recast > exp.

────────────────────────────────────────────────────────────────────────────

Operational decisions (locked in RSM_R_plus_paper2_seed_v1.md):
  - Contingency window = immediately following caregiver turn only
  - Lexical overlap   = child-to-caregiver containment (NOT Jaccard)
  - Content units     = language-specific (POS-filtered lemmas)
  - Cue inventory     = reuses Paper 1 / 02_extract_cues_v2.py output
  - Repair            = NOT counted as positive R+; tracked via separate flag
  - Composite weights = equal (0.25 each) — falsifiability constraint

Usage:
  python 10_extract_R_plus_v2.py \\
      --tagged_csv ./output/v2/English_tokens_tagged.csv \\
      --utterances_csv ./output/English_utterances.csv \\
      --lexicon_json ./trans_dict_v2.json \\
      --language English \\
      --output_dir ./output/v10b/ \\
      --morph_change_threshold 5

Author: Torami x Boss | IMT Attention project | Paper 2 / R+ direct-coding | 2026-06-16
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas numpy tqdm")
    sys.exit(1)

tqdm.pandas()


# ─────────────────────────────────────────────────────────────────────────────
# Language-specific content unit definitions
# ─────────────────────────────────────────────────────────────────────────────

CONTENT_POS: Dict[str, Set[str]] = {
    "English":    {"noun", "verb", "adj", "adv", "n", "v", "part"},
    "English-UK": {"noun", "verb", "adj", "adv", "n", "v", "part"},
    "Japanese":   {"noun", "verb", "adj", "n", "v", "adv"},
    "Mandarin":   {"noun", "verb", "adj", "n", "v", "adv"},
    "Spanish":    {"noun", "verb", "adj", "adv", "n", "v"},
    "Korean":     {"noun", "verb", "adj", "n", "v"},
    "Russian":    {"noun", "verb", "adj", "adv", "n", "v"},
    "Indonesian": {"noun", "verb", "adj", "n", "v"},
}

EXCLUDED_POS: Dict[str, Set[str]] = {
    "English":    {"det", "art", "prep", "conj", "aux", "co", "inf", "pro"},
    "English-UK": {"det", "art", "prep", "conj", "aux", "co", "inf", "pro"},
    "Japanese":   {"ptl", "particle", "case", "cop", "aux", "fil"},
    "Mandarin":   {"part", "particle", "asp", "cl", "classifier", "punct"},
    "Spanish":    {"det", "art", "prep", "conj", "aux", "clitic", "pro"},
    "Korean":     {"ptl", "particle", "case", "cop"},
    "Russian":    {"prep", "conj", "part"},
    "Indonesian": {"det", "prep", "conj"},
}

CJK_LANGUAGES: Set[str] = {"Japanese", "Mandarin", "Korean"}


def _base_language(language: str) -> str:
    """Strip dialect/corpus suffix: 'English-NA-Pool' -> 'English'."""
    return language.replace("_", "-").split("-")[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Label table (R+ depth ordinal scale, 0-4)
# ─────────────────────────────────────────────────────────────────────────────

R_PLUS_LABELS: Dict[int, str] = {
    0: "no_contingent_response",
    1: "acknowledgment",
    2: "repetition",
    3: "expansion",
    4: "recast",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract response-contingent caregiver input (R+) from CHILDES tokens."
    )
    p.add_argument("--tagged_csv", required=True,
                   help="Input cue-tagged tokens CSV from 02_extract_cues_v2.py")
    p.add_argument("--utterances_csv", required=True,
                   help="Input utterances CSV from 01_load_corpus_json.py")
    p.add_argument("--lexicon_json", required=True,
                   help="trans_dict_v2.json containing acknowledgment / repair lexicons")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK",
                            "English-NA-Pool", "English-NewmanRatner",
                            "English-BernsteinRatner", "English-Tardif",
                            "English-Higginson", "English-Brent",
                            "English-Rollins", "English-Soderstrom",
                            "English-Manchester", "English-Providence",
                            "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output/v10b")

    # v2: new / changed CLI args
    p.add_argument("--morph_change_threshold", type=int, default=5,
                   help="(v2) Min morph features added to count as morphosyntactic change. "
                        "Default=5, set from p75 of Brown English pilot.")
    p.add_argument("--repair_max_tokens", type=int, default=2,
                   help="(v2) Max caregiver utterance length (tokens) for "
                        "SINGLE-WORD repair markers. Multi-word repair phrases "
                        "are not subject to this gate. Default=2.")

    # carried over from v1
    p.add_argument("--ack_added_token_limit", type=int, default=3,
                   help="Max added tokens for an utterance to count as bare acknowledgment.")
    p.add_argument("--rep_overlap_threshold", type=float, default=0.8,
                   help="Min lexical overlap (containment) for repetition/confirmation label.")
    p.add_argument("--expansion_depth_norm", type=int, default=10,
                   help="Token-diff used to normalize expansion_depth into [0,1].")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — text normalization
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_text(text: str) -> str:
    """Lowercase + strip ASCII punctuation; preserve CJK characters and word spacing."""
    text = safe_str(text).lower().strip()
    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_tokens(text: str, language: str) -> int:
    """Approximate token count: whitespace-split for Latin, char-count for CJK."""
    text_norm = normalize_text(text)
    if not text_norm:
        return 0
    if language in CJK_LANGUAGES:
        # CJK: count non-whitespace chars (rough proxy, but consistent)
        return len([c for c in text_norm if not c.isspace()])
    return len(text_norm.split())


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon loading and matching
# ─────────────────────────────────────────────────────────────────────────────

def load_lexicon(path: Path, language: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        full = json.load(f)
    key = language.lower()
    if key not in full:
        base_key = _base_language(language).lower()
        if base_key in full:
            key = base_key
        else:
            raise ValueError(
                f"No lexicon entry for language '{language}' "
                f"(available: {[k for k in full if not k.startswith('_')]})"
            )
    return {
        "acknowledgment": full[key].get("acknowledgment", []),
        "repair":         full[key].get("repair", []),
    }


def has_acknowledgment(text: str, items: List[str], language: str) -> bool:
    """v2: kept same logic as v1 (lexicon substring/word-boundary match)."""
    if not text or not items:
        return False
    text_norm = normalize_text(text)
    if language in CJK_LANGUAGES:
        for item in items:
            item_norm = normalize_text(item)
            if item_norm and item_norm in text_norm:
                return True
        return False
    else:
        for item in items:
            item_norm = normalize_text(item)
            if not item_norm:
                continue
            if " " in item_norm:
                if f" {item_norm} " in f" {text_norm} ":
                    return True
            else:
                pattern = r"\b" + re.escape(item_norm) + r"\b"
                if re.search(pattern, text_norm):
                    return True
        return False


def has_repair_only(text: str, items: List[str], language: str,
                    max_tokens: int) -> bool:
    """v2-Fix A (v2.1 patch): a caregiver utterance is treated as repair ONLY
    when one of the following holds:

      (i) a MULTI-WORD repair phrase appears as substring (e.g., 'what did
          you say', 'say it again'). These are unambiguous clarification
          requests regardless of utterance length.

     (ii) Latin scripts only: a SINGLE-WORD repair marker (e.g., 'what',
          'huh', 'pardon') appears AND the caregiver utterance is short
          (≤ max_tokens). This gate distinguishes 'what?' from 'what is
          that'.

    (iii) CJK scripts: the caregiver utterance is approximately equal to a
          repair item (allowing up to 2 extra chars for trailing particles
          such as Japanese 「？」「の」「？？」). This handles compound
          words like 'もう一回' which are single lexical units but multiple
          chars.

    Rationale (data-driven): v1 mis-flagged 2,844 episodes (11.5% of Brown
    English contingent pairs) such as 'what is that' as repair, simply
    because they contained the word 'what'. The v2.1 logic eliminates this
    false-positive class while preserving genuine multi-word and CJK
    repair phrases.
    """
    if not text or not items:
        return False

    text_norm = normalize_text(text)
    if not text_norm:
        return False

    # Split lexicon items into multi-word (contains space) vs single
    multi_items: List[str] = []
    single_items: List[str] = []
    for item in items:
        item_norm = normalize_text(item)
        if not item_norm:
            continue
        if " " in item_norm:
            multi_items.append(item_norm)
        else:
            single_items.append(item_norm)

    # (i) Multi-word phrases — substring match, no length gate
    for item_norm in multi_items:
        if language in CJK_LANGUAGES:
            if item_norm in text_norm:
                return True
        else:
            if f" {item_norm} " in f" {text_norm} " or text_norm == item_norm:
                return True

    if language in CJK_LANGUAGES:
        # (iii) CJK: approximate exact-match (allowing trailing particles up to 2 chars)
        text_compact = text_norm.replace(" ", "")
        for item_norm in single_items:
            item_compact = item_norm.replace(" ", "")
            if not item_compact:
                continue
            if text_compact == item_compact:
                return True
            # Allow item to be a prefix with up to 2 trailing extra chars
            # (e.g., 「もう一回」 → 「もう一回？」, 「なに」 → 「なに？」)
            if (text_compact.startswith(item_compact)
                    and len(text_compact) - len(item_compact) <= 2):
                return True
        return False
    else:
        # (ii) Latin: single-word marker requires short utterance
        n_tokens = count_tokens(text, language)
        if n_tokens == 0 or n_tokens > max_tokens:
            return False

        for item_norm in single_items:
            pattern = r"\b" + re.escape(item_norm) + r"\b"
            if re.search(pattern, text_norm):
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance content extraction (from tagged token rows)
# ─────────────────────────────────────────────────────────────────────────────

def _is_content_pos(pos_value: Any, language: str) -> bool:
    pos_norm = safe_str(pos_value).lower().strip()
    if not pos_norm:
        return False
    base = _base_language(language)
    excluded = EXCLUDED_POS.get(language) or EXCLUDED_POS.get(base, set())
    if pos_norm in excluded:
        return False
    for ex in excluded:
        if pos_norm.startswith(ex + ":"):
            return False
    content = CONTENT_POS.get(language) or CONTENT_POS.get(base, set())
    for cp in content:
        if pos_norm == cp or pos_norm.startswith(cp + ":"):
            return True
    return False


def get_content_lemmas(utt_tokens: pd.DataFrame, language: str) -> Set[str]:
    if utt_tokens is None or len(utt_tokens) == 0:
        return set()
    mask = utt_tokens["pos"].fillna("").map(lambda p: _is_content_pos(p, language))
    lemmas = (
        utt_tokens.loc[mask, "lemma"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return {l for l in lemmas if l}


def get_utt_cues(utt_tokens: pd.DataFrame) -> Set[str]:
    if utt_tokens is None or len(utt_tokens) == 0:
        return set()
    if "is_cue_token" not in utt_tokens.columns:
        return set()
    cue_rows = utt_tokens[utt_tokens["is_cue_token"].fillna(False).astype(bool)]
    if len(cue_rows) == 0:
        return set()
    if "cue_subtype" in cue_rows.columns:
        cues = cue_rows["cue_subtype"].fillna("").astype(str).str.strip()
    else:
        cues = cue_rows["cue_type"].fillna("").astype(str).str.strip()
    return {c for c in cues if c}


def get_utt_features_union(utt_tokens: pd.DataFrame) -> Set[str]:
    if utt_tokens is None or len(utt_tokens) == 0:
        return set()
    feats: Set[str] = set()
    for f_raw in utt_tokens["features"].fillna(""):
        f_str = safe_str(f_raw)
        if not f_str:
            continue
        feats.update(p.strip() for p in f_str.split("|") if p.strip())
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level R+ feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_r_plus_features(
    child_text: str,
    caregiver_text: str,
    child_content: Set[str],
    caregiver_content: Set[str],
    child_cues: Set[str],
    caregiver_cues: Set[str],
    child_features: Set[str],
    caregiver_features: Set[str],
    child_num_tokens: int,
    caregiver_num_tokens: int,
    lexicon: Dict[str, List[str]],
    language: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compute the full feature vector for a child-attempt × caregiver-response pair."""

    # Lexical overlap — child-to-caregiver containment
    if len(child_content) == 0:
        lexical_overlap = 0.0
    else:
        lexical_overlap = len(child_content & caregiver_content) / len(child_content)

    # Cue features
    target_cue_repeated = bool(caregiver_cues & child_cues)
    has_new_cue = bool(caregiver_cues - child_cues)

    # v2: keep target_cue_added gate (lexical_overlap > 0 OR ack marker present)
    ack_marker = has_acknowledgment(caregiver_text, lexicon["acknowledgment"], language)
    target_cue_added = (
        has_new_cue
        and (lexical_overlap > 0.0 or ack_marker)
    )
    target_cue_modeled = target_cue_repeated or target_cue_added

    # v2-Fix B: morph_change uses CLI threshold instead of hardcoded >=1
    morph_added = caregiver_features - child_features
    morph_added_count = len(morph_added)
    morphosyntactic_change = morph_added_count >= args.morph_change_threshold

    # v2-Fix A: repair detection now requires short utterance
    repair_flag = has_repair_only(
        caregiver_text, lexicon["repair"], language, args.repair_max_tokens
    )

    # MLU delta + expansion depth
    mlu_delta = caregiver_num_tokens - child_num_tokens
    added_token_count = max(0, mlu_delta)
    expansion_depth_normalized = min(
        1.0, added_token_count / max(1, args.expansion_depth_norm)
    )

    return {
        "lexical_overlap_containment":  round(lexical_overlap, 4),
        "target_cue_repeated":          bool(target_cue_repeated),
        "target_cue_added":             bool(target_cue_added),
        "target_cue_modeled":           bool(target_cue_modeled),
        "morphosyntactic_change":       bool(morphosyntactic_change),
        "morph_features_added_count":   int(morph_added_count),
        "acknowledgment_marker":        bool(ack_marker),
        "repair_flag":                  bool(repair_flag),
        "mlu_delta":                    int(mlu_delta),
        "added_token_count":            int(added_token_count),
        "expansion_depth":              int(added_token_count),
        "expansion_depth_normalized":   round(expansion_depth_normalized, 4),
        "caregiver_num_tokens":         int(caregiver_num_tokens),
        "child_num_tokens":             int(child_num_tokens),
    }


# ─────────────────────────────────────────────────────────────────────────────
# R+ classification — depth (ordinal) and composite (continuous)
# ─────────────────────────────────────────────────────────────────────────────

def assign_r_plus_depth(
    features: Dict[str, Any],
    ack_added_token_limit: int,
    rep_overlap_threshold: float,
) -> int:
    """v2 classifier — Step 1-8.

    Key change in v2:
      - Step 7 (expansion) no longer requires NOT morphosyntactic_change.
      - Step 8 (recast) is now defined by morphosyntactic_change=True
        (which itself uses the v2 threshold >=5).
      - Recast (8) is still checked BEFORE expansion (7) so that a high
        morph-change response is correctly labeled recast.
    """
    # Step 1 handled by caller (only contingent pairs reach here)

    # Step 2: repair → not positive R+
    if features["repair_flag"]:
        return 0

    # Step 3: bare acknowledgment (low added material, no cue addition / morph change)
    if (features["acknowledgment_marker"]
            and features["added_token_count"] <= ack_added_token_limit
            and not features["target_cue_added"]
            and not features["morphosyntactic_change"]):
        return 1

    overlap = features["lexical_overlap_containment"]

    # Step 6: repetition / confirmation
    if (overlap >= rep_overlap_threshold
            and features["added_token_count"] <= ack_added_token_limit
            and features["target_cue_repeated"]
            and not features["target_cue_added"]
            and not features["morphosyntactic_change"]):
        return 2

    # Step 8 (checked before 7): recast = overlap + cue_modeled + morph change ≥ threshold
    if (overlap > 0.0
            and features["target_cue_modeled"]
            and features["morphosyntactic_change"]):
        return 4

    # Step 7: expansion = overlap + cue_modeled + added tokens
    #   (v2: morph_change condition removed — Step 8 already caught those)
    if (overlap > 0.0
            and features["added_token_count"] > 0
            and features["target_cue_modeled"]):
        return 3

    return 0


def compute_r_plus_composite(features: Dict[str, Any]) -> float:
    """Paper 2 PRIMARY predictor. Unchanged from v1."""
    positive = (
        0.25 * features["lexical_overlap_containment"]
        + 0.25 * float(features["target_cue_modeled"])
        + 0.25 * float(features["morphosyntactic_change"])
        + 0.25 * features["expansion_depth_normalized"]
    )
    negative = 1.0 * float(features["repair_flag"])
    composite = positive - negative
    return round(max(-1.0, min(1.0, composite)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main episode extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_episodes(
    tagged_df: pd.DataFrame,
    utt_df: pd.DataFrame,
    lexicon: Dict[str, List[str]],
    language: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    print("  Pre-indexing tokens by (file, utterance_index)...")
    tagged_df = tagged_df.copy()
    tagged_df["file"] = tagged_df["file"].astype(str)
    tagged_df["utterance_index"] = tagged_df["utterance_index"].astype(int)
    tok_grouped = dict(list(tagged_df.groupby(["file", "utterance_index"])))

    utt_df = utt_df.copy()
    utt_df["file"] = utt_df["file"].astype(str)
    utt_df["utterance_index"] = utt_df["utterance_index"].astype(int)
    utt_df = utt_df.sort_values(["file", "utterance_index"]).reset_index(drop=True)

    episodes: List[Dict[str, Any]] = []
    n_skipped_no_cue = 0
    n_skipped_no_next = 0
    n_non_contingent = 0

    for file_id, file_utts in tqdm(
        utt_df.groupby("file", sort=False), desc="Files"
    ):
        file_utts = file_utts.sort_values("utterance_index").reset_index(drop=True)
        utt_lookup = dict(zip(file_utts["utterance_index"], file_utts.index))

        for _, child_row in file_utts.iterrows():
            if not bool(child_row.get("is_child", False)):
                continue

            child_idx = int(child_row["utterance_index"])
            child_tokens = tok_grouped.get((file_id, child_idx))
            if child_tokens is None or len(child_tokens) == 0:
                continue

            child_cues = get_utt_cues(child_tokens)
            if len(child_cues) == 0:
                n_skipped_no_cue += 1
                continue

            next_idx = child_idx + 1
            if next_idx not in utt_lookup:
                n_skipped_no_next += 1
                continue
            next_row = file_utts.iloc[utt_lookup[next_idx]]
            is_caregiver_next = bool(next_row.get("is_caregiver", False))

            if not is_caregiver_next:
                n_non_contingent += 1
                episode = _build_episode_record(
                    file_id=file_id,
                    language=language,
                    child_row=child_row,
                    child_tokens=child_tokens,
                    child_cues=child_cues,
                    caregiver_row=None,
                    caregiver_tokens=None,
                    contingency_window=False,
                    lexicon=lexicon,
                    args=args,
                )
                episodes.append(episode)
                continue

            caregiver_tokens = tok_grouped.get((file_id, next_idx))
            episode = _build_episode_record(
                file_id=file_id,
                language=language,
                child_row=child_row,
                child_tokens=child_tokens,
                child_cues=child_cues,
                caregiver_row=next_row,
                caregiver_tokens=(caregiver_tokens
                                  if caregiver_tokens is not None
                                  else pd.DataFrame()),
                contingency_window=True,
                lexicon=lexicon,
                args=args,
            )
            episodes.append(episode)

    print(f"  Episodes built              : {len(episodes):,}")
    print(f"  Skipped (no cue in child)   : {n_skipped_no_cue:,}")
    print(f"  Skipped (no next utterance) : {n_skipped_no_next:,}")
    print(f"  Recorded non-contingent     : {n_non_contingent:,}")

    return pd.DataFrame(episodes)


def _build_episode_record(
    file_id: str,
    language: str,
    child_row: pd.Series,
    child_tokens: pd.DataFrame,
    child_cues: Set[str],
    caregiver_row: Optional[pd.Series],
    caregiver_tokens: Optional[pd.DataFrame],
    contingency_window: bool,
    lexicon: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    child_content = get_content_lemmas(child_tokens, language)
    child_features = get_utt_features_union(child_tokens)
    child_num_tokens = int(child_row.get("num_tokens", 0))
    child_text = safe_str(child_row.get("utterance_text", ""))
    child_age = child_row.get("age_months", None)

    if contingency_window and caregiver_row is not None and caregiver_tokens is not None:
        caregiver_content = get_content_lemmas(caregiver_tokens, language)
        caregiver_cues = get_utt_cues(caregiver_tokens)
        caregiver_features = get_utt_features_union(caregiver_tokens)
        caregiver_num_tokens = int(caregiver_row.get("num_tokens", 0))
        caregiver_text = safe_str(caregiver_row.get("utterance_text", ""))
        caregiver_utt_idx = int(caregiver_row["utterance_index"])

        features = compute_r_plus_features(
            child_text=child_text,
            caregiver_text=caregiver_text,
            child_content=child_content,
            caregiver_content=caregiver_content,
            child_cues=child_cues,
            caregiver_cues=caregiver_cues,
            child_features=child_features,
            caregiver_features=caregiver_features,
            child_num_tokens=child_num_tokens,
            caregiver_num_tokens=caregiver_num_tokens,
            lexicon=lexicon,
            language=language,
            args=args,
        )

        r_plus_depth = assign_r_plus_depth(
            features,
            ack_added_token_limit=args.ack_added_token_limit,
            rep_overlap_threshold=args.rep_overlap_threshold,
        )
        r_plus_composite = compute_r_plus_composite(features)
    else:
        caregiver_text = ""
        caregiver_utt_idx = -1
        caregiver_cues = set()
        features = {
            "lexical_overlap_containment":  0.0,
            "target_cue_repeated":          False,
            "target_cue_added":             False,
            "target_cue_modeled":           False,
            "morphosyntactic_change":       False,
            "morph_features_added_count":   0,
            "acknowledgment_marker":        False,
            "repair_flag":                  False,
            "mlu_delta":                    0,
            "added_token_count":            0,
            "expansion_depth":              0,
            "expansion_depth_normalized":   0.0,
            "caregiver_num_tokens":         0,
            "child_num_tokens":             child_num_tokens,
        }
        r_plus_depth = 0
        r_plus_composite = 0.0

    return {
        "file":                   file_id,
        "language":               language,
        "child_utt_idx":          int(child_row["utterance_index"]),
        "child_age_months":       child_age,
        "child_text":             child_text,
        "child_num_tokens":       child_num_tokens,
        "child_content_lemmas":   "|".join(sorted(child_content)),
        "child_cues":             "|".join(sorted(child_cues)),
        "caregiver_utt_idx":      caregiver_utt_idx,
        "caregiver_text":         caregiver_text,
        "caregiver_cues":         "|".join(sorted(caregiver_cues)),
        "contingency_window":     contingency_window,
        **features,
        "r_plus_depth":           int(r_plus_depth),
        "r_plus_label":           R_PLUS_LABELS[r_plus_depth],
        "r_plus_composite":       float(r_plus_composite),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cue-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_cue(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode-level R+ features per cue subtype.

    v2.2 (data-driven, post-Brown-pilot):
      Adds three families of measures to disentangle 'caregiver presence' from
      'caregiver response quality':

      (1) Contingent-only means:
          mean_r_plus_{depth,composite}_contingent
            — averaged ONLY over episodes where a caregiver utterance followed
              the child attempt. Strips out the dilution from non-contingent
              episodes (label 0 / next-turn-not-caregiver).

      (2) Positive-only means:
          mean_r_plus_{depth,composite}_positive
            — averaged ONLY over episodes where r_plus_depth > 0.
              The cleanest 'pure R+ quality given a positive R+ event' signal.

      (3) Per-label rates conditional on caregiver response:
          positive_rate, modeling_rate, expansion_rate, recast_rate,
          ack_rate, repetition_rate
            — proportion of caregiver responses falling into each R+ category.
              `modeling_rate` (expansion + recast) is hypothesized to be the
              most predictive (combines the two depth-rich categories).

    Rationale: Brown English pilot (Step E) showed that
    `mean_r_plus_composite` (all-episode mean) had bivariate r ≈ 0 with peak
    production, because non-contingent episodes (≈55% of the total) flatten the
    cue-level signal. Contingent-only and positive-only means recover the
    cue-level variance that actual R+ exposure creates.
    """
    if len(episodes_df) == 0:
        return pd.DataFrame()

    df = episodes_df.copy()
    df["cue_list"] = df["child_cues"].fillna("").str.split("|")
    df = df.explode("cue_list").reset_index(drop=True)
    df = df[df["cue_list"].astype(str).str.len() > 0].copy()
    df = df.rename(columns={"cue_list": "cue_subtype"})

    grouped = df.groupby(["language", "cue_subtype"], as_index=False)

    # ── (Base) all-episode aggregations ── kept for backward compatibility
    agg = grouped.agg(
        n_child_attempts=("child_utt_idx", "count"),
        n_caregiver_responses=("contingency_window", "sum"),
        mean_r_plus_depth=("r_plus_depth", "mean"),
        mean_r_plus_composite=("r_plus_composite", "mean"),
        median_r_plus_composite=("r_plus_composite", "median"),
        prop_target_cue_repeated=("target_cue_repeated", "mean"),
        prop_target_cue_added=("target_cue_added", "mean"),
        prop_target_cue_modeled=("target_cue_modeled", "mean"),
        prop_morphosyntactic_change=("morphosyntactic_change", "mean"),
        prop_acknowledgment_marker=("acknowledgment_marker", "mean"),
        prop_repair_flag=("repair_flag", "mean"),
        mean_lexical_overlap=("lexical_overlap_containment", "mean"),
        mean_expansion_depth=("expansion_depth", "mean"),
        mean_mlu_delta=("mlu_delta", "mean"),
    )

    # ── (v2.2 NEW) Contingent-only aggregations ──
    cont_only = df[df["contingency_window"]]
    if len(cont_only) > 0:
        cont_agg = (
            cont_only.groupby(["language", "cue_subtype"], as_index=False)
            .agg(
                n_contingent=("child_utt_idx", "count"),
                mean_r_plus_depth_contingent=("r_plus_depth", "mean"),
                mean_r_plus_composite_contingent=("r_plus_composite", "mean"),
                median_r_plus_composite_contingent=("r_plus_composite", "median"),
                mean_lexical_overlap_contingent=("lexical_overlap_containment", "mean"),
            )
        )
        agg = agg.merge(cont_agg, on=["language", "cue_subtype"], how="left")
    else:
        for col in ["n_contingent", "mean_r_plus_depth_contingent",
                    "mean_r_plus_composite_contingent",
                    "median_r_plus_composite_contingent",
                    "mean_lexical_overlap_contingent"]:
            agg[col] = 0.0

    # ── (v2.2 NEW) Positive-R+-only aggregations ──
    pos_only = df[df["r_plus_depth"] > 0]
    if len(pos_only) > 0:
        pos_agg = (
            pos_only.groupby(["language", "cue_subtype"], as_index=False)
            .agg(
                n_positive_rplus=("child_utt_idx", "count"),
                mean_r_plus_depth_positive=("r_plus_depth", "mean"),
                mean_r_plus_composite_positive=("r_plus_composite", "mean"),
            )
        )
        agg = agg.merge(pos_agg, on=["language", "cue_subtype"], how="left")
    else:
        for col in ["n_positive_rplus", "mean_r_plus_depth_positive",
                    "mean_r_plus_composite_positive"]:
            agg[col] = 0.0

    # NaN handling for cues with no contingent / no positive episodes
    fill_zero_cols = [
        "n_contingent", "n_positive_rplus",
        "mean_r_plus_depth_contingent", "mean_r_plus_composite_contingent",
        "median_r_plus_composite_contingent", "mean_lexical_overlap_contingent",
        "mean_r_plus_depth_positive", "mean_r_plus_composite_positive",
    ]
    for col in fill_zero_cols:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)

    # ── Per-label counts ──
    for depth, label in R_PLUS_LABELS.items():
        col = f"n_label_{depth}_{label}"
        counts = (
            df.assign(_flag=(df["r_plus_depth"] == depth).astype(int))
              .groupby(["language", "cue_subtype"], as_index=False)
              .agg(_count=("_flag", "sum"))
              .rename(columns={"_count": col})
        )
        agg = agg.merge(counts, on=["language", "cue_subtype"], how="left")

    # ── Overall response rate ──
    agg["r_plus_response_rate"] = (
        agg["n_caregiver_responses"] / agg["n_child_attempts"].replace(0, np.nan)
    ).fillna(0.0)

    # ── (v2.2 NEW) Per-label rates CONDITIONAL on a caregiver response ──
    # These are the cleanest "what kind of R+ did the caregiver typically give"
    # measures. `modeling_rate` is the hypothesized strongest predictor.
    n_resp = agg["n_caregiver_responses"].replace(0, np.nan)
    agg["ack_rate"]        = (agg["n_label_1_acknowledgment"] / n_resp).fillna(0.0)
    agg["repetition_rate"] = (agg["n_label_2_repetition"]     / n_resp).fillna(0.0)
    agg["expansion_rate"]  = (agg["n_label_3_expansion"]      / n_resp).fillna(0.0)
    agg["recast_rate"]     = (agg["n_label_4_recast"]         / n_resp).fillna(0.0)
    agg["modeling_rate"]   = ((agg["n_label_3_expansion"]
                               + agg["n_label_4_recast"]) / n_resp).fillna(0.0)
    agg["positive_rate"]   = ((agg["n_label_1_acknowledgment"]
                               + agg["n_label_2_repetition"]
                               + agg["n_label_3_expansion"]
                               + agg["n_label_4_recast"]) / n_resp).fillna(0.0)

    return agg.sort_values("n_child_attempts", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarize(episodes_df: pd.DataFrame, language: str,
              args: argparse.Namespace) -> Dict[str, Any]:
    if len(episodes_df) == 0:
        return {"language": language, "n_episodes": 0}

    cont_df = episodes_df[episodes_df["contingency_window"]]

    depth_counts = (
        episodes_df["r_plus_depth"].value_counts().sort_index().to_dict()
    )
    depth_counts_labeled = {
        f"{int(d)}_{R_PLUS_LABELS[int(d)]}": int(c) for d, c in depth_counts.items()
    }

    summary: Dict[str, Any] = {
        "language": language,
        "version": "v2",
        "v2_settings": {
            "morph_change_threshold": int(args.morph_change_threshold),
            "repair_max_tokens":      int(args.repair_max_tokens),
            "ack_added_token_limit":  int(args.ack_added_token_limit),
            "rep_overlap_threshold":  float(args.rep_overlap_threshold),
            "expansion_depth_norm":   int(args.expansion_depth_norm),
        },
        "n_episodes_total":           int(len(episodes_df)),
        "n_episodes_contingent":      int(len(cont_df)),
        "contingency_rate":           round(
            len(cont_df) / max(1, len(episodes_df)), 4
        ),
        "n_unique_files":             int(episodes_df["file"].nunique()),
        "r_plus_depth_distribution":  depth_counts_labeled,
        "mean_r_plus_composite":      round(
            float(episodes_df["r_plus_composite"].mean()), 4
        ),
        "median_r_plus_composite":    round(
            float(episodes_df["r_plus_composite"].median()), 4
        ),
        "mean_lexical_overlap_contingent": round(
            float(cont_df["lexical_overlap_containment"].mean())
            if len(cont_df) else 0.0, 4
        ),
        "prop_repair":                round(
            float(cont_df["repair_flag"].mean()) if len(cont_df) else 0.0, 4
        ),
        "prop_morphosyntactic_change": round(
            float(cont_df["morphosyntactic_change"].mean()) if len(cont_df) else 0.0, 4
        ),
        "prop_target_cue_modeled":    round(
            float(cont_df["target_cue_modeled"].mean()) if len(cont_df) else 0.0, 4
        ),
        "n_unique_cues":              int(
            episodes_df["child_cues"].fillna("").str.split("|").explode()
            .pipe(lambda s: s[s.astype(str).str.len() > 0]).nunique()
        ),
    }

    if episodes_df["child_age_months"].notna().any():
        summary["child_age_range_months"] = [
            float(episodes_df["child_age_months"].min()),
            float(episodes_df["child_age_months"].max()),
        ]
    else:
        summary["child_age_range_months"] = [None, None]

    monotone_check = (
        episodes_df.groupby("r_plus_depth")["r_plus_composite"]
        .mean().to_dict()
    )
    summary["composite_mean_by_depth"] = {
        f"{int(d)}_{R_PLUS_LABELS[int(d)]}": round(float(v), 4)
        for d, v in monotone_check.items()
    }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    tagged_csv = Path(args.tagged_csv).expanduser()
    utterances_csv = Path(args.utterances_csv).expanduser()
    lexicon_json = Path(args.lexicon_json).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 10_extract_R_plus_v2.py | Paper 2 / R+ direct-coding (v2) ===")
    print(f"  Language                  : {args.language}")
    print(f"  Tagged CSV                : {tagged_csv}")
    print(f"  Utterances                : {utterances_csv}")
    print(f"  Lexicon JSON              : {lexicon_json}")
    print(f"  Output dir                : {out_dir}")
    print(f"  morph_change_threshold    : {args.morph_change_threshold}  (v2-Fix B)")
    print(f"  repair_max_tokens         : {args.repair_max_tokens}  (v2-Fix A)")
    print(f"  ack_added_token_limit     : {args.ack_added_token_limit}")
    print(f"  rep_overlap_threshold     : {args.rep_overlap_threshold}\n")

    print("Loading lexicon...")
    lexicon = load_lexicon(lexicon_json, args.language)
    print(f"  Acknowledgment items: {len(lexicon['acknowledgment'])}")
    print(f"  Repair items        : {len(lexicon['repair'])}\n")

    print("Loading tagged tokens...")
    tagged_df = pd.read_csv(tagged_csv, low_memory=False)
    print(f"  Rows: {len(tagged_df):,}")

    print("Loading utterances...")
    utt_df = pd.read_csv(utterances_csv, low_memory=False)
    print(f"  Rows: {len(utt_df):,}\n")

    print("Extracting R+ episodes...")
    episodes_df = extract_episodes(tagged_df, utt_df, lexicon, args.language, args)
    print(f"  Total episodes: {len(episodes_df):,}\n")

    print("Aggregating by cue subtype...")
    cue_agg_df = aggregate_by_cue(episodes_df)
    print(f"  Cue subtypes: {len(cue_agg_df)}\n")

    print("Summarizing...")
    summary = summarize(episodes_df, args.language, args)

    episodes_csv = out_dir / f"{args.language}_r_plus_episodes.csv"
    cue_agg_csv  = out_dir / f"{args.language}_r_plus_cue_agg.csv"
    summary_json = out_dir / f"{args.language}_r_plus_summary.json"

    episodes_df.to_csv(episodes_csv, index=False)
    cue_agg_df.to_csv(cue_agg_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"  Episodes : {episodes_csv}  ({len(episodes_df):,} rows)")
    print(f"  Cue agg  : {cue_agg_csv}   ({len(cue_agg_df):,} rows)")
    print(f"  Summary  : {summary_json}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
