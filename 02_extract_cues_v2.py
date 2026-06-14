"""
02_extract_cues.py
==================
IMT Attention Bias Paper — Step 2: Cross-Linguistic Cue Extraction

Reads the standardized tokens CSV from step 01 and tags each token with:
    - cue_type        : the grammatical cue category (e.g., "case_nom", "word_order", "tone")
    - cue_function    : the grammatical role/meaning carried (e.g., "agent", "patient")
    - cue_subtype     : finer-grained label (e.g., "ga" for Japanese nominative ga)
    - is_cue_token    : boolean, whether this token is a cue bearer

Language-specific extractors are kept in separate functions for clarity and
auditability. Each extractor follows the cue taxonomy documented in
docs/cue_candidates_v1.md.

Usage:
    python 02_extract_cues.py --tokens_csv ./output/English_tokens.csv \\
                               --language English \\
                               --output_dir ./output/

    python 02_extract_cues.py --tokens_csv ./output/Japanese_tokens.csv \\
                               --language Japanese \\
                               --output_dir ./output/

Outputs:
    {language}_tokens_tagged.csv  — original tokens + cue annotations

Author: Torami x Boss | IMT Attention project | 2026-06-13
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-linguistic cue extraction.")
    p.add_argument("--tokens_csv", required=True,
                   help="Input tokens CSV from 01_load_corpus_json.py")
    p.add_argument("--language", required=True,
                   choices=["English", "English-UK", "Japanese", "Korean", "Mandarin",
                            "Russian", "Spanish", "Indonesian"])
    p.add_argument("--output_dir", default="./output")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Feature parsing helper
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(value) -> str:
    """Coerce any value (NaN, None, float, etc.) to a clean string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_features(features_str) -> set:
    """'Plur|Acc' -> {'Plur', 'Acc'}; empty/NaN -> set()."""
    s = safe_str(features_str)
    if not s:
        return set()
    return set(s.split("|"))


def feat_has(features_str, *target_features) -> bool:
    """True if any of target_features is in the features string."""
    f = parse_features(features_str)
    return any(t in f for t in target_features)


# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH cue extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_english_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    """Return (cue_type, cue_function, cue_subtype, is_cue)."""
    pos = safe_str(row.get("pos")).lower()
    lemma = safe_str(row.get("lemma")).lower()
    cleaned = safe_str(row.get("cleaned_text")).lower()
    features = safe_str(row.get("features"))
    gra = safe_str(row.get("gra_relation")).upper()

    # E1: SVO word order — captured via gra_relation
    if gra in ("SUBJ", "NSUBJ", "CSUBJ"):
        return ("word_order", "agent", "subject_position", True)
    if gra in ("OBJ", "DOBJ", "POBJ"):
        return ("word_order", "patient", "object_position", True)
    if gra in ("IOBJ",):
        return ("word_order", "recipient", "indirect_object", True)

    # E2: subject-verb agreement (3sg -s)
    if pos == "verb" and feat_has(features, "3S", "3s", "Pres3S"):
        return ("agreement", "subject_3sg", "verb_3s", True)

    # E3: past tense
    if pos == "verb" and feat_has(features, "Past", "PAST"):
        return ("tense", "past", "verb_past", True)

    # E4: auxiliary
    if pos == "aux":
        return ("aux", "tense_aspect_modal", "aux_" + lemma, True)

    # E5: determiners
    if pos == "det":
        return ("determiner", "np_boundary", "det_" + lemma, True)

    # E6: prepositions
    if pos in ("adp", "prep"):
        return ("preposition", "role_or_location", "prep_" + lemma, True)

    # E7: pronouns (with case info if available)
    if pos == "pron":
        if feat_has(features, "Nom"):
            return ("pronoun", "agent", "pron_nom_" + lemma, True)
        if feat_has(features, "Acc"):
            return ("pronoun", "patient", "pron_acc_" + lemma, True)
        return ("pronoun", "reference", "pron_" + lemma, True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# JAPANESE cue extractor
# ─────────────────────────────────────────────────────────────────────────────

# Japanese particles (CHAT Miyata morphology uses pos='part' for these)
JP_PARTICLE_FUNCTION = {
    "が": ("case_nom", "agent", "ga"),
    "を": ("case_acc", "patient", "o"),
    "に": ("case_dat", "recipient_or_goal", "ni"),
    "は": ("topic", "topic", "wa"),
    "で": ("case_loc_inst", "location_or_instrument", "de"),
    "と": ("case_com", "comitative", "to"),
    "の": ("case_gen", "genitive_or_nominalizer", "no"),
    "から": ("case_abl", "source", "kara"),
    "まで": ("case_term", "terminus", "made"),
    "へ": ("case_dir", "direction", "e"),
    "より": ("case_compar", "comparative", "yori"),
    "も": ("focus", "additive", "mo"),
    "か": ("question", "interrogative", "ka"),
    "ね": ("sfp", "confirmation", "ne"),
    "よ": ("sfp", "assertion", "yo"),
    "な": ("sfp", "prohibition_or_emphasis", "na"),
}


def extract_japanese_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    pos = safe_str(row.get("pos")).lower()
    cleaned = safe_str(row.get("cleaned_text"))
    lemma = safe_str(row.get("lemma"))
    gra = safe_str(row.get("gra_relation")).upper()
    is_final = bool(row.get("is_utterance_final", False))

    # SPECIAL: の の多義性 — gra_relationで分割（巴の指摘）
    if cleaned == "の" or lemma == "の":
        if gra in ("CASE", "NMOD"):
            return ("case_gen", "genitive", "particle_no_gen", True)
        elif gra == "MARK":
            if is_final:
                return ("nominalizer", "explanatory_final", "particle_no_final", True)
            return ("nominalizer", "nominalizer", "particle_no_nom", True)
        elif gra == "ACL":
            return ("nominalizer", "relative_clause", "particle_no_rel", True)
        else:
            return ("particle_no_other", "ambiguous", "particle_no_other", True)

    # J1-J5: case particles (pos='part' or 'adp')
    if pos in ("part", "adp"):
        # Try matching by cleaned_text or lemma
        for key, (ctype, cfunc, csub) in JP_PARTICLE_FUNCTION.items():
            if cleaned == key or lemma == key:
                return (ctype, cfunc, "particle_" + csub, True)
        # Discourse markers (って, etc.) — split from generic
        if cleaned in ("って", "なあ", "もん", "わ", "さ", "し", "っけ", "ょ"):
            return ("discourse_marker", "discourse", "dm_" + cleaned, True)
        # Long vowel marker
        if cleaned == "ー":
            return ("prosodic_elongation", "prosody", "elong_long", True)
        # Generic particle if not in our table
        return ("particle_other", "structural", "part_" + cleaned, True)

    # J6: verb-final morphology (utterance-final aux/verb carries tense/aspect)
    if is_final and pos in ("aux", "verb"):
        # Specific Japanese verb endings
        if any(end in cleaned for end in ["た", "だ"]):
            return ("verb_final", "past_or_copula", "vfin_past", True)
        if any(end in cleaned for end in ["る", "う", "く", "す", "つ", "ぬ", "ふ", "む", "ゆ"]):
            return ("verb_final", "nonpast", "vfin_nonpast", True)
        if "ます" in cleaned or "です" in cleaned:
            return ("verb_final", "polite", "vfin_polite", True)
        if "て" in cleaned:
            return ("verb_final", "te_form", "vfin_te", True)
        return ("verb_final", "other", "vfin_other", True)

    # J7: potential / passive (often marked in features)
    features = safe_str(row.get("features"))
    if pos == "verb" and feat_has(features, "Pot", "Pass", "POT", "PASS"):
        return ("voice", "potential_or_passive", "v_pot_pass", True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# KOREAN cue extractor
# ─────────────────────────────────────────────────────────────────────────────

KR_PARTICLE_FUNCTION = {
    "이": ("case_nom", "agent", "i"),
    "가": ("case_nom", "agent", "ga"),
    "을": ("case_acc", "patient", "eul"),
    "를": ("case_acc", "patient", "reul"),
    "에": ("case_loc", "location", "e"),
    "에서": ("case_loc", "location_source", "eseo"),
    "에게": ("case_dat", "recipient", "ege"),
    "한테": ("case_dat", "recipient_plain", "hante"),
    "께": ("case_dat", "recipient_honorific", "kke"),
    "와": ("case_com", "comitative", "wa"),
    "과": ("case_com", "comitative", "gwa"),
    "의": ("case_gen", "genitive", "ui"),
    "은": ("topic", "topic", "eun"),
    "는": ("topic", "topic", "neun"),
    "도": ("focus", "additive", "do"),
}


def extract_korean_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    """
    Korean Ko corpus has agglutinated tokens (e.g., '이름이' = name+NOM)
    rather than separate particles. We extract case particles via SUFFIX matching
    on cleaned_text or lemma, prioritized over POS-based detection.

    Diagnostic confirmed: 이/가 appear only 167+160 as standalone tokens,
    but should appear thousands of times as suffixes within nouns.
    """
    pos = safe_str(row.get("pos")).lower()
    cleaned = safe_str(row.get("cleaned_text"))
    lemma = safe_str(row.get("lemma"))
    features = safe_str(row.get("features"))
    gra = safe_str(row.get("gra_relation")).upper()
    is_final = bool(row.get("is_utterance_final", False))

    # Korean Ko corpus token pattern: many tokens are noun+particle compounds
    # Strategy: detect particle suffix on tokens that are content words.
    # We check both standalone particles AND suffixes on cm/noun/pron tokens.

    # Suffix priority order (longer first to avoid prefix conflicts)
    SUFFIX_PARTICLES = [
        # (suffix, cue_type, cue_function, cue_subtype)
        ("에서", "case_loc", "location_source", "particle_eseo"),
        ("에게", "case_dat", "recipient", "particle_ege"),
        ("한테", "case_dat", "recipient_plain", "particle_hante"),
        ("부터", "case_abl", "source", "particle_buteo"),
        ("까지", "case_term", "terminus", "particle_kkaji"),
        ("으로", "case_inst", "instrument_or_direction", "particle_euro"),
        ("으", "case_inst", "instrument_or_direction", "particle_euro_short"),
        ("를", "case_acc", "patient", "particle_reul"),
        ("을", "case_acc", "patient", "particle_eul"),
        ("이", "case_nom", "agent", "particle_i"),
        ("가", "case_nom", "agent", "particle_ga"),
        ("는", "topic", "topic", "particle_neun"),
        ("은", "topic", "topic", "particle_eun"),
        ("에", "case_loc", "location", "particle_e"),
        ("도", "focus", "additive", "particle_do"),
        ("의", "case_gen", "genitive", "particle_ui"),
        ("와", "case_com", "comitative", "particle_wa"),
        ("과", "case_com", "comitative", "particle_gwa"),
    ]

    # Step 1: standalone particles (when tokenization separated them)
    if pos in ("adp", "part"):
        for suf, ct, cf, cs in SUFFIX_PARTICLES:
            if cleaned == suf or lemma == suf:
                return (ct, cf, cs + "_standalone", True)

    # Step 2: SUFFIX matching on noun/pron/cm tokens (the key fix)
    # Many cm-tagged tokens are noun+particle compounds.
    if pos in ("noun", "pron", "propn", "cm") and len(cleaned) >= 2:
        for suf, ct, cf, cs in SUFFIX_PARTICLES:
            if cleaned.endswith(suf) and cleaned != suf:
                # Avoid false positives: ensure the stem is non-trivial
                stem = cleaned[:-len(suf)]
                if len(stem) >= 1:
                    return (ct, cf, cs + "_suffix", True)

    # Step 3: verb-final speech-level endings (utterance-final verbs/aux)
    if is_final and pos in ("verb", "aux", "sconj"):
        if cleaned.endswith("거예요") or cleaned.endswith("예요") or cleaned.endswith("이에요"):
            return ("verb_final", "polite_decl", "vfin_yeyo", True)
        if cleaned.endswith("니다") or cleaned.endswith("ㅂ니다"):
            return ("verb_final", "formal", "vfin_sumnida", True)
        if cleaned.endswith("요"):
            return ("verb_final", "polite", "vfin_yo", True)
        if cleaned.endswith("다"):
            return ("verb_final", "plain_decl", "vfin_da", True)
        if cleaned.endswith("어") or cleaned.endswith("아"):
            return ("verb_final", "casual", "vfin_eoa", True)
        if cleaned.endswith("지"):
            return ("verb_final", "confirmation", "vfin_ji", True)
        if cleaned.endswith("네"):
            return ("verb_final", "exclamation", "vfin_ne", True)
        return ("verb_final", "other", "vfin_other", True)

    # Step 4: connective endings (non-final verbs)
    if pos in ("verb", "sconj") and not is_final:
        if cleaned.endswith("고"):
            return ("connective", "conjunctive", "conn_go", True)
        if cleaned.endswith("서"):
            return ("connective", "causal_sequential", "conn_seo", True)
        if cleaned.endswith("면"):
            return ("connective", "conditional", "conn_myeon", True)
        if cleaned.endswith("니까") or cleaned.endswith("으니까"):
            return ("connective", "causal", "conn_nikka", True)

    # Step 5: honorific
    if "Hon" in features or "Si" in features:
        return ("honorific", "subject_honorific", "v_hon", True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# MANDARIN cue extractor
# ─────────────────────────────────────────────────────────────────────────────

ZH_CONSTRUCTION_MARKERS = {
    "把": ("construction_ba", "object_disposal", "ba"),
    "被": ("construction_bei", "passive", "bei"),
    "了": ("aspect_le", "perfective_or_change", "le"),
    "着": ("aspect_zhe", "progressive", "zhe"),
    "过": ("aspect_guo", "experiential", "guo"),
    "的": ("particle_de", "modifier_link", "de_mod"),
    "得": ("particle_de", "verb_complement", "de_comp"),
    "地": ("particle_de", "adverbializer", "de_adv"),
    "给": ("preposition_gei", "recipient", "gei"),
}


def extract_mandarin_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    pos = safe_str(row.get("pos")).lower()
    cleaned = safe_str(row.get("cleaned_text"))
    lemma = safe_str(row.get("lemma"))
    gra = safe_str(row.get("gra_relation")).upper()

    # M2-M5, M7-M8: construction/aspect/particle markers
    for key, (ctype, cfunc, csub) in ZH_CONSTRUCTION_MARKERS.items():
        if cleaned == key or lemma == key:
            return (ctype, cfunc, csub, True)

    # M1: SVO word order via gra
    if gra in ("SUBJ", "NSUBJ"):
        return ("word_order", "agent", "subject_position", True)
    if gra in ("OBJ", "DOBJ"):
        return ("word_order", "patient", "object_position", True)

    # M6: tone — encoded in lemma as digit suffix (e.g., 'ma1')
    # We tag whether the token has a tone marker observable
    if lemma and any(d in lemma for d in "1234"):
        # tone is intrinsic to every Mandarin word but we don't tag every token
        # as cue. Instead, this is captured as a feature elsewhere.
        pass

    # M7: classifiers (POS varies; sometimes 'noun', sometimes 'classifier')
    if pos in ("clf", "classifier"):
        return ("classifier", "np_classifier", "clf_" + cleaned, True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# RUSSIAN cue extractor
# ─────────────────────────────────────────────────────────────────────────────

RU_CASE_FEATURES = {
    "Nom": ("case_nom", "agent"),
    "Acc": ("case_acc", "patient"),
    "Dat": ("case_dat", "recipient"),
    "Gen": ("case_gen", "possession_or_partitive"),
    "Ins": ("case_ins", "instrument"),
    "Loc": ("case_loc", "location"),
    "Prep": ("case_loc", "location"),
}


def extract_russian_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    pos = safe_str(row.get("pos")).lower()
    features = safe_str(row.get("features"))
    lemma = safe_str(row.get("lemma"))

    # R1-R6: case on nouns/pronouns/adjectives
    if pos in ("noun", "pron", "adj", "propn"):
        for case_feat, (ctype, cfunc) in RU_CASE_FEATURES.items():
            if feat_has(features, case_feat):
                return (ctype, cfunc, f"case_{case_feat.lower()}", True)

    # R7: gender agreement
    if pos in ("adj", "verb") and feat_has(features, "Masc", "Fem", "Neut"):
        feats = parse_features(features)
        gender = next((g for g in ("Masc", "Fem", "Neut") if g in feats), "")
        if gender:
            return ("agreement_gender", "concord", f"gender_{gender.lower()}", True)

    # R8: verb aspect
    if pos == "verb" and feat_has(features, "Perf", "Imp", "Imperf"):
        feats = parse_features(features)
        aspect = "perf" if "Perf" in feats else "imperf"
        return ("aspect", "event_structure", f"aspect_{aspect}", True)

    # Prepositions
    if pos in ("adp", "prep"):
        return ("preposition", "spatial_role", "prep_" + lemma, True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# SPANISH cue extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_spanish_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    pos = safe_str(row.get("pos")).lower()
    features = safe_str(row.get("features"))
    lemma = safe_str(row.get("lemma")).lower()
    cleaned = safe_str(row.get("cleaned_text")).lower()
    gra = safe_str(row.get("gra_relation")).upper()

    # SVO and pro-drop language
    if gra in ("SUBJ", "NSUBJ"):
        return ("word_order", "agent", "subject_position", True)
    if gra in ("OBJ", "DOBJ"):
        return ("word_order", "patient", "object_position", True)

    # Verb morphology (rich)
    if pos == "verb":
        if feat_has(features, "3S", "3s"):
            return ("agreement", "subject_3sg", "verb_3s", True)
        if feat_has(features, "1S", "1s"):
            return ("agreement", "subject_1sg", "verb_1s", True)
        if feat_has(features, "Pret", "PRET"):
            return ("tense", "preterite", "verb_pret", True)
        if feat_has(features, "Impf", "IMPF"):
            return ("tense", "imperfect", "verb_impf", True)
        if feat_has(features, "Subj", "SUBJ"):
            return ("mood", "subjunctive", "verb_subj", True)

    # Determiners (gender-marked: el/la/los/las/un/una)
    if pos == "det":
        gender_feat = ""
        if feat_has(features, "Masc"): gender_feat = "m"
        elif feat_has(features, "Fem"): gender_feat = "f"
        return ("determiner", "np_boundary", f"det_{lemma}_{gender_feat}", True)

    # Prepositions
    if pos in ("adp", "prep"):
        return ("preposition", "role_or_location", "prep_" + lemma, True)

    # Pronouns (Spanish has rich clitic system)
    if pos == "pron":
        if lemma in ("me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las"):
            return ("pronoun_clitic", "argument_marking", "clitic_" + lemma, True)
        return ("pronoun", "reference", "pron_" + lemma, True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# INDONESIAN cue extractor
# ─────────────────────────────────────────────────────────────────────────────

# Indonesian uses verbal affixes for voice/argument structure.
# Each prefix maps to (cue_type, cue_function, min_stem_len). The min_stem_len
# guards against false positives where the prefix is actually part of a
# monomorphemic root (e.g. "pesawat" = airplane, NOT pe-+sawat).
INDO_AFFIX_PATTERNS = {
    # prefix: (cue_type, cue_function)
    "me": ("voice_active", "agent_focus"),
    "di": ("voice_passive", "patient_focus"),
    "ber": ("voice_intrans", "stative_or_reciprocal"),
    "ter": ("voice_resultative", "accidental_or_capable"),
    "pe": ("nominalizer", "agent_nominal"),
    "ke-an": ("nominalizer", "abstract_noun"),
}

# Minimum length of the residual stem after stripping the prefix. Indonesian
# roots are overwhelmingly disyllabic (>= 3-4 chars). A residue shorter than
# this almost always means the "prefix" is really part of the root.
INDO_MIN_STEM_LEN = 3

# Common monomorphemic words that begin with prefix-like strings but are NOT
# affixed. This is a small targeted blocklist for the worst offenders; the
# stem-length and lemma checks catch the long tail.
INDO_AFFIX_FALSE_FRIENDS = {
    # pe-
    "pesawat", "perut", "pena", "peta", "pegang", "penuh", "pendek",
    "perlu", "percaya", "pergi", "perang", "pesan", "petang",
    # me-
    "meja", "mereka", "merah", "memang", "menang",  # menang=win is me+? ambiguous; treat as root
    "mewah", "melayu",
    # di- (also a preposition "at"! handled separately)
    "dia", "diam", "dingin", "dinding",
    # ber-
    "berat", "beras", "berani", "bersih",  # some are truly ber-, but high FP risk as roots
    # ter-
    "teri", "terus", "ternak",
}


def _indo_is_real_affix(cleaned: str, lemma: str, prefix: str) -> bool:
    """
    Decide whether `cleaned` genuinely carries `prefix` as a productive affix,
    rather than the prefix being part of a monomorphemic root.

    Strategy (in priority order):
      1. Blocklist: known false friends are never treated as affixed.
      2. Lemma check: if a lemma is available and the prefix-stripped form
         matches (or closely relates to) the lemma, it's a real affix.
         This is the strongest signal because MOR lemmas encode morphology.
      3. Stem-length heuristic: the residue after stripping must be a
         plausible root (>= INDO_MIN_STEM_LEN chars).
    """
    if cleaned in INDO_AFFIX_FALSE_FRIENDS:
        return False

    stem = cleaned[len(prefix):]

    # Residue too short to be a real Indonesian root → not an affix.
    if len(stem) < INDO_MIN_STEM_LEN:
        return False

    # Strongest signal: the morphological lemma.
    # If MOR gives a lemma, a real affix means lemma == stem (or lemma is the
    # root and differs from the surface form by exactly this prefix).
    if lemma:
        # lemma already equals the surface → no affixation happened
        if lemma == cleaned:
            return False
        # lemma equals the stripped stem → genuine affix
        if lemma == stem:
            return True
        # lemma is the bare root and the surface added the prefix:
        # surface == prefix + lemma  (e.g. lemma="makan", cleaned="memakan")
        if cleaned == prefix + lemma:
            return True
        # lemma present but inconsistent with a clean strip → be conservative
        # (only accept if stem is reasonably long, suggesting real morphology)
        return len(stem) >= 4

    # No lemma available: fall back to stem-length only (already passed >=3).
    # Require >=4 for the highly ambiguous "pe"/"di" to cut false positives.
    if prefix in ("pe", "di"):
        return len(stem) >= 4
    return True


def extract_indonesian_cues(row: pd.Series) -> Tuple[str, str, str, bool]:
    pos = safe_str(row.get("pos")).lower()
    cleaned = safe_str(row.get("cleaned_text")).lower()
    lemma = safe_str(row.get("lemma")).lower()
    gra = safe_str(row.get("gra_relation")).upper()

    # Verbal voice prefixes. Now guarded by _indo_is_real_affix to avoid
    # treating monomorphemic roots (pesawat, meja, ...) as affixed forms.
    if pos == "verb" and cleaned != lemma:
        # longest prefixes first so "ber"/"ter" win over any 2-char overlap
        for prefix in sorted(INDO_AFFIX_PATTERNS, key=len, reverse=True):
            ctype, cfunc = INDO_AFFIX_PATTERNS[prefix]
            if "-" in prefix:
                continue  # circumfix ke-an handled elsewhere; skip here
            if cleaned.startswith(prefix) and not lemma.startswith(prefix):
                if _indo_is_real_affix(cleaned, lemma, prefix):
                    return (ctype, cfunc, f"affix_{prefix}", True)

    # yang (relativizer / cleft marker)
    if cleaned == "yang" or lemma == "yang":
        return ("relativizer", "clause_link", "yang", True)

    # Word order
    if gra in ("SUBJ", "NSUBJ"):
        return ("word_order", "agent", "subject_position", True)
    if gra in ("OBJ", "DOBJ"):
        return ("word_order", "patient", "object_position", True)

    # Adpositions (di=at, ke=to, dari=from)
    if pos in ("adp", "prep"):
        return ("preposition", "spatial_role", "prep_" + lemma, True)

    return ("", "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTOR_MAP = {
    "English": extract_english_cues,
    "Japanese": extract_japanese_cues,
    "Korean": extract_korean_cues,
    "Mandarin": extract_mandarin_cues,
    "Russian": extract_russian_cues,
    "Spanish": extract_spanish_cues,
    "Indonesian": extract_indonesian_cues,
}


def resolve_extractor(language: str):
    """
    Resolve the cue extractor for a language label, tolerating dialect
    variants and casing. E.g. "English-UK", "English_UK", "english-uk" all
    map to the English extractor. The base language is taken as the substring
    before the first hyphen/underscore.
    """
    if language in EXTRACTOR_MAP:
        return EXTRACTOR_MAP[language]
    # Strip dialect suffix: "English-UK" -> "English", "Mandarin_TW" -> "Mandarin"
    base = language.replace("_", "-").split("-")[0].strip()
    # Case-insensitive match against known keys
    for key, fn in EXTRACTOR_MAP.items():
        if key.lower() == base.lower():
            return fn
    raise KeyError(
        f"No extractor for language '{language}'. Known: "
        f"{list(EXTRACTOR_MAP)}. Dialect variants like 'English-UK' are "
        f"mapped by the part before '-'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_csv = Path(args.tokens_csv).expanduser()
    print(f"Loading: {tokens_csv}")
    df = pd.read_csv(tokens_csv, low_memory=False)
    print(f"  {len(df):,} tokens loaded.")

    extractor = resolve_extractor(args.language)
    print(f"Tagging cues for {args.language}...")

    # Apply extractor row-wise
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tagging"):
        results.append(extractor(row))

    cue_df = pd.DataFrame(
        results, columns=["cue_type", "cue_function", "cue_subtype", "is_cue_token"]
    )

    df = pd.concat([df.reset_index(drop=True), cue_df], axis=1)

    # Stats
    n_cues = int(df["is_cue_token"].sum())
    n_total = len(df)
    print(f"\n  Tagged cues: {n_cues:,} / {n_total:,}  ({100*n_cues/n_total:.1f}%)")

    # Distribution of cue types (caregiver speech only)
    care = df[df["speaker_role"] == "caregiver"]
    cue_care = care[care["is_cue_token"]]
    if len(cue_care):
        cue_type_dist = cue_care["cue_type"].value_counts().head(15).to_dict()
        cue_subtype_dist = cue_care["cue_subtype"].value_counts().head(15).to_dict()
    else:
        cue_type_dist = {}
        cue_subtype_dist = {}

    out_csv = out_dir / f"{args.language}_tokens_tagged.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Wrote: {out_csv}")

    summary_json = out_dir / f"{args.language}_cue_summary.json"
    summary = {
        "language": args.language,
        "n_tokens_total": n_total,
        "n_cue_tokens": n_cues,
        "cue_token_rate": round(n_cues / n_total, 4) if n_total else 0,
        "top15_cue_types_caregiver": cue_type_dist,
        "top15_cue_subtypes_caregiver": cue_subtype_dist,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Wrote: {summary_json}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
