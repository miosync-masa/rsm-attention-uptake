# Attention as Pre-Grammatical Selection Pressure

**A cross-linguistic test of the Reactive Schematization Model (RSM)**

This repository contains the analysis pipeline, theoretical documents, and
aggregated results for the paper:

> *Attention as Pre-Grammatical Selection Pressure: A Cross-Linguistic Test of
> the Reactive Schematization Model*
> Masamichi Iizumi & Torami (Miosync, Inc.)
> ORCID [0009-0007-0755-403X](https://orcid.org/0009-0007-0755-403X)

The study quantifies caregiver **attention bias** across eight CHILDES samples
(seven languages plus an American/British English dialect pair) and tests
whether the *attention × frequency* interaction predicts children's grammatical
uptake. The central empirical finding is a **two-phase dissociation**:
initial cue *emergence* is frequency-driven and cross-linguistically
heterogeneous, while *entrenchment* to peak production is driven by the
attention × frequency interaction and is cross-linguistically homogeneous.

---

## Headline result

| Outcome | Pooled interaction beta | z | p | I-squared | Sign test |
|---|---:|---:|---:|---:|---:|
| **Peak production** (entrenchment) | **+0.45** [+0.38, +0.53] | 12.1 | <.00001 | 23.6% (homogeneous) | **8/8** (p=.004) |
| First emergence (registration) | +0.07 [-0.08, +0.22] | 0.96 | .34 (ns) | 70.1% (heterogeneous) | 5/8 (ns) |

The interaction adds **~13.7x more variance** to peak production than attention
as a main effect (mean delta-R-squared 0.179 vs 0.013). See
`docs/supplemental_material_v1.md` for the full statistics.

---

## Repository layout

```
.
├── 01_load_corpus_json.py            # CHILDES CHA -> JSON -> standardized token CSV
├── 02_extract_cues_v2.py             # per-language grammatical cue tagging
├── 03_compute_attention_index_v3.py  # 5-dim Attention Index + non-circular Reliability
├── 05_developmental_uptake.py        # child uptake measures + hierarchical regressions
├── 06_meta_analysis.py               # random-effects meta-analysis across samples
├── 04_visualize_results_v3.py        # publication figures (forest, two-phase, dR2)
│
├── docs/                             # theory + write-up
│   ├── reactive_schematization_model_v1.md   # RSM theoretical framework
│   ├── attention_index_formalization_v1.md   # AI mathematical definition
│   ├── cue_candidates_v1.md                  # per-language cue inventory
│   ├── discussion_tourist_phenomenon_v1.md   # Discussion extension
│   └── supplemental_material_v1.md           # full results tables (8 samples)
│
├── results/                         # AGGREGATED outputs only (safe to share)
│   ├── {sample}_attention_index.csv
│   ├── {sample}_uptake.csv
│   ├── {sample}_uptake_by_age.csv
│   ├── {sample}_uptake_summary.json
│   ├── meta_effects_peak_rate_per_1k.csv
│   ├── meta_effects_first_emergence_month.csv
│   ├── meta_analysis_summary.json
│   └── figures/                     # generated PNGs
│
├── README.md
├── LICENSE                          # MIT (code + aggregated results)
└── .gitignore
```

> **Note on data.** This repository does **not** redistribute any CHILDES /
> TalkBank source material. Only **aggregated, cue-level statistics** (from
> which no original utterance can be reconstructed) are included under
> `results/`. To reproduce from raw data you must obtain the corpora yourself
> (see below) — they are governed by the TalkBank Ground Rules.

---

## Samples

| Sample | CHILDES corpora | Morphological type |
|---|---|---|
| English (US) | Brown | analytic / word-order |
| English (UK) | Manchester, MPI-EVA-Manchester, Thomas, Wells, Forrester, ... | analytic / word-order (dialect replication) |
| Japanese | Miyata | agglutinating (SOV, particles) |
| Korean | Ko, Ryu, Jiwon | agglutinating (SOV, particles) |
| Mandarin | Erbaugh, TCCM, Tong, Zhou3 | isolating |
| Russian | RusLan, Tanja | fusional (case-rich) |
| Spanish | Ornat, Vila, Aguirre, Linaza, Marrero, Remedi, ... | fusional (Romance) |
| Indonesian | Jakarta | analytic with voice affixation |

---

## Reproducing from raw corpora

### 1. Environment

```bash
python -m venv imt_env
source imt_env/bin/activate          # Windows: imt_env\Scripts\activate
pip install pandas numpy scipy statsmodels matplotlib tqdm
```

Tested on Python 3.12. The `02`-`06` scripts require `scipy` and `statsmodels`.

### 2. Obtain the corpora (you, not this repo)

1. Register at <https://talkbank.org/> and accept the
   [Ground Rules](https://talkbank.org/share/rules.html).
2. Install the Chatter CLI (used for CHA->JSON conversion):
   <https://talkbank.org/software/chatter.html>
3. Download the corpora listed above from the CHILDES browser and arrange them
   so each language has its own folder; sub-corpora may be nested freely
   (the loader recurses):

```
~/childes_data/
├── English/        ├── Korean/       ├── Russian/
├── English-UK/     ├── Mandarin/     ├── Spanish/
├── Japanese/       └── Indonesian/
```

### 3. Run the pipeline

```bash
for lang in English English-UK Japanese Korean Mandarin Russian Spanish Indonesian; do
    python 01_load_corpus_json.py --corpus_path ~/childes_data/${lang}/ --language ${lang} --output_dir ./output/ --convert
    python 02_extract_cues_v2.py        --tokens_csv ./output/${lang}_tokens.csv               --language ${lang} --output_dir ./output/v2/
    python 03_compute_attention_index_v3.py --tagged_csv ./output/v2/${lang}_tokens_tagged.csv --language ${lang} --output_dir ./output/v3/
    python 05_developmental_uptake.py   --tagged_csv ./output/v2/${lang}_tokens_tagged.csv \
                                        --ai_csv ./output/v3/${lang}_attention_index.csv       --language ${lang} --output_dir ./output/v3/
done

python 06_meta_analysis.py --output_dir ./output/v3/ \
    --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian

python 04_visualize_results_v3.py --output_dir ./output/v3/
```

The aggregated CSV/JSON written to `output/v3/` corresponds to what is committed
under `results/`. (Intermediate `output/*_tokens.csv`, `output/v2/*_tagged.csv`,
and `output/json_cache/` are TalkBank-derived and are **git-ignored**.)

---

## The Attention Index (AI)

For each grammatical cue *c* in language *L*, the AI is the unweighted mean of
five salience dimensions, with **cross-linguistically fixed weights** (the
constraint that makes the model falsifiable):

```
AI(c, L) = 0.2*S_acoustic + 0.2*S_positional + 0.2*S_frequency
         + 0.2*S_repetition + 0.2*S_perceptual
```

`Reliability` (how informative a cue is, once attended) is a **separate**
companion variable, reported in three non-circular forms (`gra`, `position`,
`form`). Full definitions: `docs/attention_index_formalization_v1.md`.

---

## The Reactive Schematization Model (RSM)

The empirical pipeline tests a theoretical proposal that the infant is not
primarily *acquiring grammar* but *seeking communicative success*, with grammar
emerging as the sedimented residue of response-confirmed signaling patterns:

```
I -> A -> O -> R -> B -> schema -> next-generation O
```

(Input -> Attention -> Output -> Response -> Book -> schema.) The two-phase
dissociation reported here is the model's predicted signature: frequency drives
what enters the **Book**; the attention x frequency product drives what
**entrenches** into a schema. See `docs/reactive_schematization_model_v1.md`.

---

## Citation

```bibtex
@unpublished{iizumi_attention_2026,
  author = {Iizumi, Masamichi and Torami},
  title  = {Attention as Pre-Grammatical Selection Pressure:
            A Cross-Linguistic Test of the Reactive Schematization Model},
  year   = {2026},
  note   = {Manuscript. ORCID 0009-0007-0755-403X}
}
```

If you use the corpora themselves, please also cite CHILDES
(MacWhinney, 2000) and the individual corpus contributors.

---

## License

Code and aggregated results in this repository are released under the
**MIT License** (see `LICENSE`). This license covers **only** the contents of
this repository; it does **not** extend to the CHILDES/TalkBank source corpora,
which remain governed by the [TalkBank Ground Rules](https://talkbank.org/share/rules.html).
