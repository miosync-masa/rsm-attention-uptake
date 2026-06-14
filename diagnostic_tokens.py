"""
diagnostic_tokens.py
====================
IMT Attention Bias Paper — Diagnostic Tool

Inspects tokens CSV for a given language to understand:
  - Available columns
  - POS distribution
  - What's hiding in 'cm', 'part', 'adp', 'aux', 'sconj' POS categories
  - Specific surface forms (e.g., は/が/を for Japanese, 가/이 for Korean)
  - Where grammatical info actually lives (token? lemma? mor? gra?)

Use this BEFORE re-running 02_extract_cues.py so the pattern table can be
adjusted to match the actual data.

Usage:
    python diagnostic_tokens.py --tokens_csv ./output/Korean_tokens.csv \\
                                 --language Korean

Author: Torami x Boss x Tomoe x Shio-ne | IMT Attention | 2026-06-13
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Install: pip install pandas")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-language probe patterns: surface forms we expect to find
# ─────────────────────────────────────────────────────────────────────────────

PROBES = {
    "Japanese": {
        "case_nom_ga": ["が", "ga"],
        "case_acc_wo": ["を", "wo", "o"],
        "case_dat_ni": ["に", "ni"],
        "topic_wa":    ["は", "wa"],
        "case_loc_de": ["で", "de"],
        "gen_no":      ["の", "no"],
        "sfp_ne":      ["ね", "ne"],
        "sfp_yo":      ["よ", "yo"],
        "question_ka": ["か", "ka"],
        "quote_tte":   ["って", "tte"],
    },
    "Korean": {
        "case_nom_iga":   ["이", "가", "i", "ga"],
        "case_acc_eulreul": ["을", "를", "ul", "lul", "eul", "reul"],
        "case_loc_e":     ["에", "ey", "e"],
        "case_loc_eseo":  ["에서", "eyse", "eseo"],
        "case_dat_ege":   ["에게", "eykey", "ege", "한테", "hante"],
        "topic_eun":      ["은", "는", "un", "nun", "eun", "neun"],
        "honorific_si":   ["si", "hon"],
        "vfin_da":        ["다"],
        "vfin_yo":        ["요"],
    },
    "Mandarin": {
        "ba":    ["把"],
        "bei":   ["被"],
        "le":    ["了"],
        "de_mod":["的"],
        "gei":   ["给"],
        "zhe":   ["着"],
        "guo":   ["过"],
    },
    "English": {
        "the": ["the"], "a": ["a"], "is": ["is"], "do": ["do"], "did": ["did"],
        "ed_suffix": ["ed", "past"], "s_suffix": ["s", "3s", "pres3s"],
    },
    "Russian": {
        "nom_case":  ["nom", "именит"],
        "acc_case":  ["acc", "винит"],
        "dat_case":  ["dat", "датель"],
        "gen_case":  ["gen", "родит"],
        "ins_case":  ["ins", "творит"],
        "loc_case":  ["loc", "prep"],
        "perf":      ["perf"],
        "imperf":    ["imp"],
    },
    "Spanish": {
        "el": ["el"], "la": ["la"], "los": ["los"], "las": ["las"],
        "se_clitic": ["se"], "me_clitic": ["me"], "te_clitic": ["te"],
        "preterite": ["pret", "3s"],
    },
    "Indonesian": {
        "yang":   ["yang"],
        "di_pre": ["di"],
        "ke_pre": ["ke"],
        "ber_pre":["ber"],
        "me_pre": ["me"],
        "ter_pre":["ter"],
    },
}


# Suspect POS tags to investigate per language
SUSPECT_POS = {
    "Japanese":   ["part", "adp", "aux", "sconj", "cm", "intj"],
    "Korean":     ["cm", "sconj", "part", "adp", "aux", "intj", "unk"],
    "Mandarin":   ["part", "adp", "aux", "sconj", "cm", ""],
    "English":    ["part", "adp", "aux", "sconj", "cm", "det"],
    "Russian":    ["adp", "part", "cm", "sconj"],
    "Spanish":    ["det", "adp", "pron", "part"],
    "Indonesian": ["cm", "part", "x", ""],
}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic functions
# ─────────────────────────────────────────────────────────────────────────────

def report_columns(df: pd.DataFrame):
    print("\n=== Available columns ===")
    for c in df.columns:
        non_null = df[c].notna().sum()
        sample = df[c].dropna().head(3).tolist()
        print(f"  {c:<25} non-null={non_null:>7,}  sample={sample}")


def report_pos_distribution(df: pd.DataFrame, top_n: int = 25):
    cg = df[df.get("speaker_role", pd.Series("")) == "caregiver"]
    if len(cg) == 0:
        cg = df
        print("(No caregiver filter applied — using all rows)")

    print(f"\n=== POS distribution (caregiver speech, top {top_n}) ===")
    pos_col = "pos" if "pos" in df.columns else "pos_normalized"
    counts = cg[pos_col].value_counts().head(top_n)
    for pos, n in counts.items():
        print(f"  {pos:<15} {n:>8,}")


def inspect_suspect_pos(df: pd.DataFrame, suspect_list, n_samples: int = 25):
    """Show what's actually inside each suspect POS category."""
    cg = df[df.get("speaker_role", pd.Series("")) == "caregiver"]
    if len(cg) == 0:
        cg = df

    pos_col = "pos" if "pos" in df.columns else "pos_normalized"

    inspect_cols = [c for c in [
        "cleaned_text", "raw_text", "lemma", "features", "gra_relation",
        "is_utterance_final"
    ] if c in df.columns]

    for p in suspect_list:
        sub = cg[cg[pos_col] == p]
        print(f"\n--- POS = '{p}'  (N={len(sub):,}) ---")
        if len(sub) == 0:
            continue

        # Top cleaned_text values
        if "cleaned_text" in df.columns:
            top_forms = sub["cleaned_text"].value_counts().head(15)
            print(f"  Top surface forms:")
            for form, n in top_forms.items():
                print(f"    {form!r:<20} {n:>6,}")

        # Random sample of rows
        sample_n = min(n_samples, len(sub))
        sample = sub.sample(sample_n, random_state=42) if len(sub) > sample_n else sub
        print(f"\n  Random sample ({sample_n} rows):")
        print(sample[inspect_cols].to_string(index=False, max_colwidth=30))


def probe_surface_forms(df: pd.DataFrame, probes: dict):
    """For each probe pattern, count how many tokens contain it."""
    cg = df[df.get("speaker_role", pd.Series("")) == "caregiver"]
    if len(cg) == 0:
        cg = df

    search_cols = [c for c in [
        "cleaned_text", "raw_text", "lemma", "features"
    ] if c in df.columns]

    print(f"\n=== Surface form probes (searching: {search_cols}) ===")
    for probe_name, patterns in probes.items():
        print(f"\n  {probe_name}  (patterns: {patterns})")
        for pat in patterns:
            # Match exact cleaned_text
            exact_hits = (
                (cg["cleaned_text"].astype(str) == pat).sum()
                if "cleaned_text" in df.columns else 0
            )
            # Match exact lemma
            lemma_hits = (
                (cg["lemma"].astype(str) == pat).sum()
                if "lemma" in df.columns else 0
            )
            # Substring in features
            feat_hits = (
                cg["features"].astype(str).str.contains(
                    pat, case=False, regex=False, na=False
                ).sum()
                if "features" in df.columns else 0
            )
            print(f"    {pat!r:<15}  exact_cleaned={exact_hits:>6,}  "
                  f"exact_lemma={lemma_hits:>6,}  in_features={feat_hits:>6,}")


def find_grammatical_role_in_gra(df: pd.DataFrame, n_samples: int = 30):
    """Show what gra_relation values exist and their distribution."""
    cg = df[df.get("speaker_role", pd.Series("")) == "caregiver"]
    if len(cg) == 0:
        cg = df

    if "gra_relation" not in df.columns:
        print("\n=== gra_relation column not found ===")
        return

    print(f"\n=== gra_relation distribution (caregiver) ===")
    counts = cg["gra_relation"].value_counts().head(30)
    for rel, n in counts.items():
        print(f"  {rel:<25} {n:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose tokens CSV.")
    parser.add_argument("--tokens_csv", required=True)
    parser.add_argument("--language", required=True,
                        choices=list(PROBES.keys()))
    parser.add_argument("--n_samples", type=int, default=25)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {args.language}")
    print(f"File: {args.tokens_csv}")
    print(f"{'='*70}")

    df = pd.read_csv(args.tokens_csv, low_memory=False)
    print(f"\nTotal rows: {len(df):,}")

    report_columns(df)
    report_pos_distribution(df)
    find_grammatical_role_in_gra(df)
    inspect_suspect_pos(df, SUSPECT_POS[args.language], args.n_samples)
    probe_surface_forms(df, PROBES[args.language])

    print(f"\n{'='*70}")
    print("Done. Copy/paste this output back to Torami for pattern refinement.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
