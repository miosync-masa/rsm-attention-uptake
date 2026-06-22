"""
10_extract_R_plus.py
====================
IMT Attention Bias Paper — Step 10: Response-Contingent Input Extraction (Paper 2)

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
      - 1.00 * repair_flag                              (negative R+, separated)

Reads:
  {language}_tokens_tagged.csv      (output of 02_extract_cues_v2.py)
  {language}_utterances.csv          (output of 01_load_corpus_json.py)
  trans_dict.json                    (acknowledgment / repair lexicons)

Writes:
  {language}_r_plus_episodes.csv     (per child-attempt × caregiver-response pair)
  {language}_r_plus_cue_agg.csv      (cue-subtype level aggregation, for join with COI)
  {language}_r_plus_summary.json     (distribution statistics, monotonicity check)

Operational decisions (locked in RSM_R_plus_paper2_seed_v1.md):
  - Contingency window = immediately following caregiver turn only (delayed responses excluded)
  - Lexical overlap   = child-to-caregiver containment (NOT Jaccard)
  - Content units     = language-specific (POS-filtered lemmas)
  - Cue inventory     = reuses Paper 1 / 02_extract_cues_v2.py output (column: is_cue_token)
  - Repair            = NOT counted as positive R+; tracked via separate repair_flag
  - Composite weights = equal (0.25 each) — falsifiability constraint, same as Paper 1 COI

Usage:
  python 10_extract_R_plus.py \\
      --tagged_csv ./output/v2/English_tokens_tagged.csv \\
      --utterances_csv ./output/English_utterances.csv \\
      --lexicon_json ./trans_dict.json \\
      --language English \\
      --output_dir ./output/v10/

Author: Torami x Boss | IMT Attention project | Paper 2 / R+ direct-coding | 2026-06-16
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
#
# Content units = lemmas/morphemes used for child-to-caregiver lexical overlap.
# These follow CHILDES MOR conventions. POS tags vary slightly across corpora;
# the prefix-match logic (`pos == cp or pos.startswith(cp + ":")`) handles
# subtype tags like "n:prop", "v:cop", "v:aux", etc.
#
# Verify against actual data on first run by inspecting the per-language POS
# distribution (see docs/cue_candidates_v1.md for cue-bearing POS).
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

# Explicitly excluded POS — function words, particles, copulas, articles.
# Even if a content lemma is attached, these POS tags rule the token out.
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

# Languages requiring character-based (not word-boundary) lexicon matching.
CJK_LANGUAGES: Set[str] = {"Japanese", "Mandarin", "Korean"}


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
                   help="trans_dict.json containing acknowledgment / repair lexicons")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK", "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output/v10")
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
    """Coerce any value (NaN, None, float, etc.) to a clean string."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon loading and matching
# ─────────────────────────────────────────────────────────────────────────────

def load_lexicon(path: Path, language: str) -> Dict[str, List[str]]:
    """Load acknowledgment / repair lexicons for the target language."""
    with open(path, "r", encoding="utf-8") as f:
        full = json.load(f)
    key = language.lower()
    if key not in full:
        # Fallback for English variants
        if key in ("english-uk", "english_uk") and "english" in full:
            key = "english"
        else:
            raise ValueError(
                f"No lexicon entry for language '{language}' "
                f"(available: {[k for k in full if not k.startswith('_')]})"
            )
    return {
        "acknowledgment": full[key].get("acknowledgment", []),
        "repair":         full[key].get("repair", []),
    }


def has_lexicon_match(text: str, items: List[str], language: str) -> bool:
    """Check whether any lexicon item appears in the utterance text."""
    if not text or not items:
        return False
    text_norm = normalize_text(text)
    if language in CJK_LANGUAGES:
        # Character-based substring matching (no word boundaries in CJK)
        for item in items:
            item_norm = normalize_text(item)
            if item_norm and item_norm in text_norm:
                return True
        return False
    else:
        # Word-boundary matching for Latin scripts
        for item in items:
            item_norm = normalize_text(item)
            if not item_norm:
                continue
            if " " in item_norm:
                # Multi-word: match as bounded substring
                if f" {item_norm} " in f" {text_norm} ":
                    return True
            else:
                pattern = r"\b" + re.escape(item_norm) + r"\b"
                if re.search(pattern, text_norm):
                    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance content extraction (from tagged token rows)
# ─────────────────────────────────────────────────────────────────────────────

def _is_content_pos(pos_value: Any, language: str) -> bool:
    """Return True iff POS is in the language's content set and not excluded."""
    pos_norm = safe_str(pos_value).lower().strip()
    if not pos_norm:
        return False
    excluded = EXCLUDED_POS.get(language, set())
    if pos_norm in excluded:
        return False
    # Also exclude if matches an excluded prefix (e.g., "aux:cop")
    for ex in excluded:
        if pos_norm.startswith(ex + ":"):
            return False
    content = CONTENT_POS.get(language, set())
    for cp in content:
        if pos_norm == cp or pos_norm.startswith(cp + ":"):
            return True
    return False


def get_content_lemmas(utt_tokens: pd.DataFrame, language: str) -> Set[str]:
    """Return the set of content-word/morpheme lemmas in an utterance."""
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
    """Return the set of cue subtypes present in an utterance (from is_cue_token)."""
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
    """Return the union of all morphological features across tokens in an utterance.

    Used as a coarse proxy for morphosyntactic structure; the *difference*
    between caregiver and child feature sets indicates whether the caregiver
    supplied additional morphological information (a recast signature).
    """
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
    expansion_depth_norm: int,
) -> Dict[str, Any]:
    """Compute the full feature vector for a child-attempt × caregiver-response pair.

    Returns the inputs that feed both `assign_r_plus_depth` (ordinal scale)
    and `compute_r_plus_composite` (continuous primary predictor).
    """

    # Lexical overlap — child-to-caregiver containment (巴-designed).
    # Containment, not Jaccard: expansions add tokens, which would unfairly
    # depress Jaccard. Containment captures "how much of the child's content
    # the caregiver preserved", which is the construct of interest.
    if len(child_content) == 0:
        lexical_overlap = 0.0
    else:
        lexical_overlap = len(child_content & caregiver_content) / len(child_content)

    # Cue-level features (reuses Paper 1 cue inventory).
    target_cue_repeated = bool(caregiver_cues & child_cues)
    has_new_cue = bool(caregiver_cues - child_cues)

    # target_cue_added requires evidence of contingency — otherwise a caregiver
    # whose next utterance happens to contain any cue would spuriously count.
    target_cue_added = (
        has_new_cue
        and (lexical_overlap > 0.0
             or has_lexicon_match(caregiver_text, lexicon["acknowledgment"], language))
    )
    target_cue_modeled = target_cue_repeated or target_cue_added

    # Morphosyntactic change: features present in caregiver but not in child.
    # Conservative threshold: ≥1 added feature flags the change.
    morph_added = caregiver_features - child_features
    morphosyntactic_change = len(morph_added) >= 1

    # Acknowledgment / repair markers.
    ack_marker = has_lexicon_match(caregiver_text, lexicon["acknowledgment"], language)
    repair_flag = has_lexicon_match(caregiver_text, lexicon["repair"], language)

    # MLU delta + expansion depth.
    mlu_delta = caregiver_num_tokens - child_num_tokens
    added_token_count = max(0, mlu_delta)
    expansion_depth_normalized = min(
        1.0, added_token_count / max(1, expansion_depth_norm)
    )

    return {
        "lexical_overlap_containment":  round(lexical_overlap, 4),
        "target_cue_repeated":          bool(target_cue_repeated),
        "target_cue_added":             bool(target_cue_added),
        "target_cue_modeled":           bool(target_cue_modeled),
        "morphosyntactic_change":       bool(morphosyntactic_change),
        "morph_features_added_count":   int(len(morph_added)),
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
    """巴-designed Step 1-8 classifier.

    Order matters: repair is checked before any positive label; recast is
    checked before expansion so that a morphosyntactic-change response is
    not mislabeled as a mere expansion.
    """
    # Step 1: contingency window check is performed by the caller
    # (only contingent pairs reach this function).

    # Step 2: repair → not positive R+
    if features["repair_flag"]:
        return 0

    # Step 3: bare acknowledgment (low added material, no cue/morph addition)
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

    # Step 8 (checked before 7): recast = overlap + cue_modeled + morphosyntactic_change
    if (overlap > 0.0
            and features["target_cue_modeled"]
            and features["morphosyntactic_change"]):
        return 4

    # Step 7: expansion = overlap + longer response + cue_modeled, w/o morph change
    if (overlap > 0.0
            and features["added_token_count"] > 0
            and features["target_cue_modeled"]):
        return 3

    # Else: no contingent linguistic model
    return 0


def compute_r_plus_composite(features: Dict[str, Any]) -> float:
    """Paper 2 PRIMARY predictor.

    R+_composite =
       0.25 * lexical_overlap_containment
     + 0.25 * target_cue_modeled
     + 0.25 * morphosyntactic_change_flag
     + 0.25 * expansion_depth_normalized
     - 1.00 * repair_flag

    Equal weighting is a falsifiability constraint (same convention as the
    Paper 1 COI). The result is clipped to [-1.0, +1.0].
    """
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
    """Walk each file, find child utterances containing cue attempts, locate the
    immediately following caregiver utterance (if any), and compute R+ features.
    """
    # Pre-index tagged tokens by (file, utterance_index) for fast lookup
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
                # Paper 2 scope: only utterances containing cue attempts
                n_skipped_no_cue += 1
                continue

            # Find immediately following utterance in the same file
            next_idx = child_idx + 1
            if next_idx not in utt_lookup:
                n_skipped_no_next += 1
                continue
            next_row = file_utts.iloc[utt_lookup[next_idx]]
            is_caregiver_next = bool(next_row.get("is_caregiver", False))

            if not is_caregiver_next:
                # Next turn is not caregiver → non-contingent (label 0)
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
    """Construct a single episode record (per child-attempt × caregiver-response)."""

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
            expansion_depth_norm=args.expansion_depth_norm,
        )

        r_plus_depth = assign_r_plus_depth(
            features,
            ack_added_token_limit=args.ack_added_token_limit,
            rep_overlap_threshold=args.rep_overlap_threshold,
        )
        r_plus_composite = compute_r_plus_composite(features)
    else:
        # No contingent response — fill features as zeros for consistency
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

    Each child attempt may contain multiple cues; we explode by cue so that
    each cue receives credit for responses to attempts containing it. This
    output joins with Paper 1 attention_index / uptake CSVs in step 11.
    """
    if len(episodes_df) == 0:
        return pd.DataFrame()

    df = episodes_df.copy()
    df["cue_list"] = df["child_cues"].fillna("").str.split("|")
    df = df.explode("cue_list").reset_index(drop=True)
    df = df[df["cue_list"].astype(str).str.len() > 0].copy()
    df = df.rename(columns={"cue_list": "cue_subtype"})

    grouped = df.groupby(["language", "cue_subtype"], as_index=False)

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

    # Per-label counts (n at each ordinal depth)
    for depth, label in R_PLUS_LABELS.items():
        col = f"n_label_{depth}_{label}"
        counts = (
            df.assign(_flag=(df["r_plus_depth"] == depth).astype(int))
              .groupby(["language", "cue_subtype"], as_index=False)
              .agg(_count=("_flag", "sum"))
              .rename(columns={"_count": col})
        )
        agg = agg.merge(counts, on=["language", "cue_subtype"], how="left")

    # Overall response rate per cue
    agg["r_plus_response_rate"] = (
        agg["n_caregiver_responses"] / agg["n_child_attempts"].replace(0, np.nan)
    ).fillna(0.0)

    return agg.sort_values("n_child_attempts", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarize(episodes_df: pd.DataFrame, language: str) -> Dict[str, Any]:
    """Compute distribution stats for diagnostic / sanity-check reporting."""
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

    # Monotonicity sanity check — composite mean should rise with depth
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

    print(f"\n=== 10_extract_R_plus.py | Paper 2 / R+ direct-coding ===")
    print(f"  Language     : {args.language}")
    print(f"  Tagged CSV   : {tagged_csv}")
    print(f"  Utterances   : {utterances_csv}")
    print(f"  Lexicon JSON : {lexicon_json}")
    print(f"  Output dir   : {out_dir}\n")

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
    summary = summarize(episodes_df, args.language)

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
