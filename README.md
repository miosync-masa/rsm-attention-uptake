# Responsive Input and Productive Stabilization

**Cross-linguistic evidence for the Reactive Schematization Model (RSM)**

This repository contains the analysis pipeline, theoretical documents, and
aggregated results for the manuscript:

> *Responsive Input and Productive Stabilization: Cross-Linguistic Evidence for
> the Reactive Schematization Model*
> Author(s) omitted for anonymous peer review.

The study quantifies caregiver **cue orienting potential** across eight CHILDES
samples (seven languages plus an American/British English dialect pair) and
tests whether the *cue-orienting × frequency* interaction predicts children's
grammatical uptake. The central empirical finding is a **two-phase
dissociation**: initial cue *emergence* is frequency-driven and
cross-linguistically heterogeneous, while *stabilization* to peak production is
driven by the cue-orienting × frequency interaction and is cross-linguistically
homogeneous.

---

## Headline result

| Outcome | Pooled interaction beta | z | p | I-squared | Sign test |
|---|---:|---:|---:|---:|---:|
| **Peak production** (stabilization) | **+0.45** [+0.38, +0.53] | 12.1 | <.00001 | 23.6% (homogeneous) | **8/8** (p=.004) |
| First emergence (registration) | +0.07 [-0.08, +0.22] | 0.96 | .34 (ns) | 70.1% (heterogeneous) | 5/8 (ns) |

The interaction adds **~13.7x more variance** to peak production than cue
orienting as a main effect (mean delta-R-squared 0.179 vs 0.013). See
`docs/supplemental_material.md` for the full statistics.

---

## Repository layout

```
.
├── 01_load_corpus_json.py            # CHILDES CHA -> JSON -> standardized token CSV
├── 02_extract_cues_v2.py             # per-language grammatical cue tagging
├── 03_compute_attention_index_v3.py  # 5-dim Cue Orienting Index + non-circular Reliability
├── 05_developmental_uptake.py        # child uptake measures + hierarchical regressions
├── 06_meta_analysis.py               # random-effects meta-analysis across samples
├── 04_visualize_results_v3.py        # publication figures (forest, two-phase, dR2)
├── 07_temporal_structure.py          # caregiver cue temporal structure (span/dispersion/
│                                     #   persistence/burstiness), frequency-orthogonalized
├── 08_temporal_meta.py               # meta-analysis of temporal-structure x frequency
├── 09_temporal_diagnostics.py        # separability + recording-density confound diagnostics
│
├── docs/                             # theory + write-up
│   ├── reactive_schematization_model.md      # RSM theoretical framework
│   ├── attention_index_formalization.md      # COI mathematical definition
│   ├── cue_candidates.md                     # per-language cue inventory
│   ├── discussion_tourist_phenomenon.md      # Discussion extension
│   └── supplemental_material.md              # full results tables (8 samples)
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

> **Naming note.** Several script and output filenames retain the string
> `attention_index` for internal consistency with earlier development versions.
> The construct these files compute is the **Cue Orienting Index (COI)** as
> defined in the manuscript and in `docs/attention_index_formalization.md`;
> the two names refer to the same quantity.

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
python -m venv rsm_env
source rsm_env/bin/activate          # Windows: rsm_env\Scripts\activate
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
                                        --coi_csv ./output/v3/${lang}_attention_index.csv      --language ${lang} --output_dir ./output/v3/
done

python 06_meta_analysis.py --output_dir ./output/v3/ \
    --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian

python 04_visualize_results_v3.py --output_dir ./output/v3/
```

### 4. (Optional) Temporal-structure separability probe

This auxiliary analysis (SI §S6) tests whether caregiver-cue temporal
distribution can be measured independently of token frequency. It recovers a
developmental age axis for caregiver utterances by file-level inheritance (CHILDES
marks age only on the target child), then computes frequency-orthogonalized
span/dispersion/persistence/burstiness and checks them against recording-density
confounds.

```bash
for lang in English English-UK Japanese Korean Mandarin Russian Spanish Indonesian; do
    python 07_temporal_structure.py --tagged_csv ./output/v2/${lang}_tokens_tagged.csv \
                                    --uptake_csv ./results/${lang}_uptake.csv \
                                    --language ${lang} --output_dir ./output/v3/ --bin_width 3
done

python 08_temporal_meta.py        --output_dir ./output/v3/ \
    --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian

python 09_temporal_diagnostics.py --output_dir ./output/v3/ --results_dir ./results/ \
    --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian
```

The probe's conclusion is reported in SI §S6: temporal structure is *separable*
from frequency at the level of measurement, but its *independent* effect on
stabilization is confounded with recording density and is left to future
time-resolved corpora.

The aggregated CSV/JSON written to `output/v3/` corresponds to what is committed
under `results/`. (Intermediate `output/*_tokens.csv`, `output/v2/*_tagged.csv`,
and `output/json_cache/` are TalkBank-derived and are **git-ignored**.)

---

## Paper 2 (exposure-gate extension)

> **Branch.** All Paper 2 material — additional scripts, intermediate evidence
> files, and the meta-analytic stack — lives on the
> [`paper2-exposure-gate`](https://github.com/miosync-masa/rsm-attention-uptake/tree/paper2-exposure-gate)
> branch. The `main` branch contains only the Paper 1 (RSM) pipeline.

### Hypothesis

Paper 1 establishes that **COI × frequency** predicts cue stabilization. Paper 2
asks a developmental follow-up: does the **same** interaction become readable on
*online* cue reuse once the child has accumulated enough exposure to the cue?
The exposure-gate prediction is:

> Within the post-MSR developmental window (child age ≥ 24 months), the
> probability that a child reuses a cue *c* in the next *N* utterances is
> predicted by **COI(c) × cumulative_cue_attempts(child, c, t)**, controlling
> for short-range autoregression (prior 20-utterance local frequency) and
> corpus-level cue frequency.

In words: post-MSR, attention amplifies reuse for cues the child has practiced
more — a time-scale-invariant extension of Paper 1's COI × frequency finding.

### Headline meta-analytic result

Random-effects meta-analysis across **32 children from 5 sub-corpora**
(Brown, Manchester, UK long-longitudinal {Thomas / Fraser / Helen / Eleanor /
Nicole}, UK short-observation {Wells-Bristol-style}, NA other longitudinal
{April}), per-child OLS with cluster-robust SE on cue_subtype:

| Outcome window | n | Pooled β(COI × cumulative_cue_attempts) | SE | p | I² |
|---|---:|---:|---:|---:|---:|
| **next 5 utt reuse** | 32 | **+0.024** | 0.008 | **0.002** | 63.4% |
| **next 10 utt reuse** | 32 | **+0.033** | 0.009 | **<0.001** | 36.1% |
| next 5 utt, drop Manchester | 21 | **+0.036** | 0.011 | **<0.001** | 70.2% |
| next 10 utt, drop Manchester | 21 | **+0.045** | 0.012 | **<0.001** | 46.6% |

* 6 / 32 children individually significant (positive); **0 / 32 significant
  negative**; 21 / 32 in the positive direction.
* Egger small-study tests p > 0.35 across all four cells — no funnel asymmetry.
* Heterogeneity (I²) is concentrated in the **Manchester** sub-corpus
  (n = 11, β = +0.001 at N=5; β = +0.013 at N=10; I² = 0% within Manchester).
* Window-range sensitivity (SPEC #17m / #17o, N ∈ {3, 5, 7, 10, 15, 20}):
  smooth monotonic-then-attenuation; 4 / 6 windows reach LPM significance,
  boundary nulls (N=3 floor; N=20 ceiling) recovered by Logit and continuous
  fraction-outcome modes (SPEC #17o).
* R+ independence (SPEC #17n): adding 4 R+ terms (episode-level and cue-level
  R+_composite plus their × COI interactions) to the champion model shifts
  β(COI × cumulative) by ≤ 8.1 %; all four cells stay p < 0.001.
* Anti-circularity (SPEC #19): rebuilding COI from four non-frequency
  components (S_acoustic + S_positional + S_repetition + S_perceptual; equal
  weights 0.25) leaves β within ≤ 17.7 % at every cell and reduces
  heterogeneity (I² N=10 FULL 36.1 % → 19.4 %), confirming the effect is
  not driven by S_frequency_normalized.

### Manchester null is methodological

A four-section diagnostic stack (17g protocol features; 17h sub-corpus split;
17i JSON-cache feature diff; 17k raw `.cha` qualitative features) traces the
Manchester null to a single design choice in the Theakston-lab Manchester
corpus:

* **88% of sessions** declare `@Situation` as *"playing with toys"* or
  *"Structured Play"*.
* **All 11 Manchester children share one stimulus set** — *"Duplo Zoo + basket
  of food + doll"* — with only 16 minor activity-string variants across
  374 thirty-minute sessions.
* **Investigator-mediated interaction** (`pct_INV` = 3.6%) is unique to
  Manchester among the four sub-corpora.
* Mother / child speech-share ratio is **1.55** vs Brown's 0.70 and
  UK_short_obs's 0.73 (mother verbally dominates the structured play).
* Utterance density is **44 utt / min** vs Brown's 23 utt / min.

The standardized stimulus + investigator-mediated 30-min protocol compresses
cue-context variability, creating a ceiling on within-cue cumulative attention
that the Mundlak decomposition cannot disentangle from autoregressive
vocabulary patterns. Manchester's null is therefore a **methodological
consequence** of its corpus design, not a counter-example to the exposure-gate
prediction. The effect generalises across all corpora collected under
naturalistic conditions.

### Pipeline

The Paper 2 scripts share their inputs with Paper 1 (the tokens, tagged
tokens, attention index, and uptake CSVs from `01-05`). New stages add R+
(response-contingent caregiver input) extraction, Mundlak-style
between/within decomposition, episode-level next-N-reuse outcome, and the
exposure-gate per-child meta:

| Stage | Script | Role |
|---|---|---|
| 10  | `10_extract_R_plus_v2.py` | Episode-level R+ extraction (acknowledgment / repetition / expansion / recast) |
| 11  | `11_rsm_r_plus_join.py` | Join R+ × Paper 1 COI + uptake; M0 / M1 / M2 hierarchical regression |
| 12  | `12_rplus_meta_analysis.py` | Cross-language meta of the M2 R+_composite × COI interaction |
| 13  | `13_robustness_quick.py` | M2 robustness (Cook's d, leverage, bootstrap CI) |
| 14  | `14_age_stratified_quick.py` | Pre / post-MSR (24 mo) age split for M2 |
| 15  | `15_within_between_decomposition.py` | Mundlak split: R+_between(c) vs R+_within(c, e) |
| 16  | `16_episode_outcome_uptake.py` | Per-episode outcome: **next-N child utterance cue reuse** |
| 17  | `17_robustness_age_post_coi.py` | Frequency-confound gate for **age_post × COI** (SPEC #1) |
| 17  | `17_within_effect_test.py` | 3-way age × COI × R+_within multilevel logit-LPM (Spec C) |
| 17b | `17b_exposure_gate_test.py` | **Main exposure-gate model**: COI × cumulative_cue_attempts |
| 17c | `17c_child_level_slopes.py` | Per-child OLS + random-slope MixedLM + DerSimonian-Laird meta |
| 17d | `17d_window_width_check.py` | Observation-window moderator of per-child β |
| 17e | `17e_adequate_window_subset.py` | β stratified by min window threshold (6 / 9 / 12 mo) |
| 17f | `17f_uk_reverse_simulation.py` | Restrict UK children to Manchester age band (24-36 mo) and refit |
| 17g | `17g_manchester_diagnostics.py` | UK vs Manchester MLU / density / cue-mix triple check |
| 17h | `17h_uk_subcorpora_meta.py` | Split UK pool into long- vs short-observation sub-corpora |
| 17i | `17i_corpus_protocol_diff.py` | Protocol features extracted from the JSON cache |
| 17j | `17j_na_pool_per_child.py` | NA-Pool Brown extension (April; NewmanRatner shown to be cross-sectional) |
| 17k | `17k_manchester_qualitative.py` | Raw-`.cha` protocol features (Theakston signature extraction) |
| 17l | `17l_paper_figures.py` | Forest + funnel + Theakston radar; drop-Manchester sensitivity meta |
| 17m | `17m_window_range_sensitivity.py` | 4-window grid (N=3,5,10,20) + trajectory + Egger per window |
| 17n | `17n_rplus_drop_test.py` | Model A (R+ kept, 9-pred champion) vs Model B (R+ dropped) per-child comparison |
| 17o | `17o_window_null_mechanism.py` | 6-window × 3-mode (LPM / Logit / Continuous fraction) + variance / truncation diagnostics |
| 19 | `19_coi_4comp_test.py` | 4-component COI sensitivity (drop S_frequency_normalized); SI S1 anti-circularity |

### Reproducing the exposure-gate stack

Assuming Paper 1 outputs already exist under `output/v3/` (for `caregiver_AI`
and `log_cue_freq`) and the appropriate tokens / utterances / tagged CSVs are
present:

```bash
# For each sample, build the R+ episode set and the per-cue regression input.
for lang in English English-Manchester English-UK; do
    python 10_extract_R_plus_v2.py \
        --tagged_csv ./output/${lang}_tokens_tagged.csv \
        --utterances_csv ./output/${lang}_utterances.csv \
        --lexicon_json ./trans_dict_v2.json \
        --language ${lang} --output_dir ./output/v10b/

    python 11_rsm_r_plus_join.py \
        --r_plus_csv  ./output/v10b/${lang}_r_plus_cue_agg.csv \
        --uptake_csv  ./output/v3/${lang}_uptake.csv \
        --ai_csv      ./output/${lang}_attention_index.csv \
        --language ${lang} --output_dir ./output/v11/

    # Episode-level outcome (next-N child utterance cue reuse)
    python 16_episode_outcome_uptake.py \
        --episodes_csv ./output/v10b/${lang}_r_plus_episodes.csv \
        --tagged_csv   ./output/${lang}_tokens_tagged.csv \
        --language ${lang} --output_dir ./output/v16/ --windows 5,10
done

# Per-child exposure-gate β with cluster-robust SE on cue
python 17b_exposure_gate_test.py --batch --window 5 --output_dir ./output/v17b/
python 17c_child_level_slopes.py --batch --window 5 --output_dir ./output/v17c/
python 17c_child_level_slopes.py --batch --window 10 --output_dir ./output/v17c/

# Diagnostics and stratification
python 17d_window_width_check.py --output_dir ./output/v17d/
python 17e_adequate_window_subset.py --output_dir ./output/v17e/ --thresholds 12,9,6
python 17f_uk_reverse_simulation.py --output_dir ./output/v17f/
python 17g_manchester_diagnostics.py --output_dir ./output/v17g/
python 17h_uk_subcorpora_meta.py --output_dir ./output/v17h/  --window 5
python 17h_uk_subcorpora_meta.py --output_dir ./output/v17h_N10/ --window 10 \
    --per_child_csv ./output/v17c/per_child_betas_N10.csv

# Qualitative protocol features (JSON cache + raw .cha)
python 17i_corpus_protocol_diff.py --output_dir ./output/v17i/
python 17k_manchester_qualitative.py --output_dir ./output/v17k/

# Sensitivity stack (window range, R+ independence, window-null mechanism,
# anti-circularity)
python 17m_window_range_sensitivity.py --output_dir ./output/v17m/ --windows 3,5,10,20
python 17n_rplus_drop_test.py          --output_dir ./output/v17n/ --windows 5,10
python 17o_window_null_mechanism.py    --output_dir ./output/v17o/ \
    --windows 3,5,7,10,15,20
python 19_coi_4comp_test.py            --output_dir ./output/v19/  --windows 5,10

# Final paper-ready figures + drop-Manchester sensitivity meta
python 17l_paper_figures.py --output_dir ./output/v17l/
```

> **Note on April (NA sole-child) per-window β.** `17l_paper_figures.py`
> calls a window-aware refit (`_refit_april_at_window`) ported from 17m so
> April's per-window β is computed against the requested outcome window N
> rather than reusing a cached N=5 value. This was fixed in SPEC #21 — the
> pre-fix repository, the SPEC #21 commit message, and `output/v17l/SUMMARY.md`
> all carry the correction.

### Committed Paper 2 evidence (small aggregates only)

Only paper-ready aggregates are tracked under git; the underlying
TalkBank-derived intermediates remain ignored (see `.gitignore`).

```
output/
├── v15/                                  # Mundlak decomposition (Spec A + Spec B')
├── v17/                                  # SPEC #1 + Spec C results
├── v17b/   exposure_gate_summary_N5.csv  # Main exposure-gate per-corpus β
├── v17c/   per_child_betas_N{5,10}.csv   # Per-child OLS β + SE
├── v17d/   window_moderator_results.json # Window-width moderator (WLS)
├── v17e/   adequate_window_summary.csv   # β by window-threshold strata
├── v17f/   uk_reverse_results.json       # UK age-band restriction
├── v17g/   manchester_vs_uk_descriptives.csv
├── v17h/   uk_subcorpora_meta_N5.json    # 4-way sub-corpus meta
├── v17h_N10/                             # N=10 sensitivity rerun of 17h
├── v17i/   protocol_diff_per_corpus.csv  # JSON-cache protocol features
├── v17j/   final_32_child_meta.json      # NA + April extension
├── v17k/   protocol_features_per_corpus.csv  # Raw-.cha protocol features
├── v17l/   forest_plot_N5_vs_N10.png     # Paper-ready figures
│        funnel_plot_N{5,10}.png
│        theakston_radar.png
│        drop_manchester_meta.json
├── v17m/   four_window_meta.json         # Window-range grid (N=3,5,10,20)
│        forest_4panel.png
│        pooled_beta_trajectory.png
├── v17n/   comparison_table_FullModel_vs_RplusDropped.csv  # R+ independence
│        per_child_betas_Rplus_{kept,dropped}_N{5,10}.csv
├── v17o/   meta_six_windows.json         # LPM/Logit/Continuous × 6 windows
│        variance_compression_table.csv
│        truncation_rate_table.csv
│        trajectory_three_modes.png
├── v18/    18a_coi_spec.json             # SI fill — COI weights + r/VIF + anti-circ
│        18a_coi_anticircularity.json
│        18b_per_child_full.csv           # SI Table S3.2 (32 rows)
│        18c_corpus_descriptives.csv      # SI Table S2.2 (5 rows)
│        18d_boundary_exact.csv           # SI S5.1 / S5.2 boundary cells
│        18e_rplus_spec.md                # SI S9 R+ predictor / estimator
│        18f_environment.md               # SI S10 versions + script map
└── v19/    comparison_5comp_vs_4comp.csv # SI S1.2 frequency-drop sensitivity
         meta_4comp.json
         per_child_betas_4comp_N{5,10}.csv
```

Every `output/v17*/SUMMARY.md` contains a stand-alone, copy-paste-ready
summary of that stage's tables and verdict.

### Paper 2 publication-ready figures

Mirroring the Paper 1 convention (`results/figures/Fig1_forest_peak.png` etc.),
the Paper 2 figures are also published under `results/figures/` with a
`Paper2_` prefix so reviewers who only browse the curated `results/` tree see
them:

| File | Source script / output | Used in manuscript as |
|---|---|---|
| `results/figures/Paper2_Fig1_forest_N5_N10.png`         | `17l_paper_figures.py` → `output/v17l/forest_plot_N5_vs_N10.png`       | **Figure 1** — per-child forest, N=5 and N=10 panels with corrected diamond |
| `results/figures/Paper2_Fig2a_funnel_N5.png`            | `17l_paper_figures.py` → `output/v17l/funnel_plot_N5.png`             | **Figure 2a** — funnel + Egger, outcome window N=5 |
| `results/figures/Paper2_Fig2b_funnel_N10.png`           | `17l_paper_figures.py` → `output/v17l/funnel_plot_N10.png`            | **Figure 2b** — funnel + Egger, outcome window N=10 |
| `results/figures/Paper2_Fig3_theakston_radar.png`       | `17l_paper_figures.py` → `output/v17l/theakston_radar.png`            | **Figure 3** — 5-axis Theakston-Manchester protocol signature |
| `results/figures/Paper2_Fig4_window_grid_4panel.png`    | `17m_window_range_sensitivity.py` → `output/v17m/forest_4panel.png`   | **Figure 4** — 4-window forest (N = 3, 5, 10, 20) |
| `results/figures/Paper2_Fig5_pooled_beta_trajectory.png`| `17m_window_range_sensitivity.py` → `output/v17m/pooled_beta_trajectory.png` | **Figure 5** — pooled β trajectory FULL vs DROP-MANC across N |
| `results/figures/Paper2_Fig6_three_outcome_modes.png`   | `17o_window_null_mechanism.py` → `output/v17o/trajectory_three_modes.png`    | **Figure 6** — LPM / Logit / Continuous β trajectory across six windows |

The `results/` copies are kept byte-identical to the upstream `output/v17*/`
PNGs at commit time; when a script regenerates an upstream PNG, the
corresponding `results/figures/Paper2_*.png` should be refreshed in the same
commit to keep the two trees in sync.

### Companion lexicon

`trans_dict_v2.json` contains the per-language acknowledgment / repair token
lexicons consumed by `10_extract_R_plus_v2.py` (English, Japanese, Korean,
Mandarin, Russian, Spanish, Indonesian; English entries cover both Brown and
UK dialects).

---

## The Cue Orienting Index (COI)

For each grammatical cue *c* in language *L*, the COI is the unweighted mean of
five salience dimensions, with **cross-linguistically fixed weights** (the
constraint that makes the model falsifiable):

```
COI(c, L) = 0.2*S_acoustic + 0.2*S_positional + 0.2*S_frequency
          + 0.2*S_repetition + 0.2*S_perceptual
```

COI estimates the *orienting potential* of a cue from caregiver corpus
statistics; it is a property of the input signal, not a direct measure of child
attentional behavior. `Reliability` (how informative a cue is, once attended) is
a **separate** companion variable, reported in three non-circular forms (`gra`,
`position`, `form`). Full definitions: `docs/attention_index_formalization.md`.

---

## The Reactive Schematization Model (RSM)

The empirical pipeline tests a theoretical proposal that the learner is not
primarily *acquiring grammar* but *seeking communicative success*, with grammar
emerging as the sedimented residue of response-confirmed signaling patterns:

```
I -> A -> O -> R -> B -> schema -> next-generation O
```

(Input -> Attention -> Output -> Response -> store -> schema.) The two-phase
dissociation reported here is the model's predicted signature: frequency drives
what enters the store; the cue-orienting × frequency product drives what
**stabilizes** into a schema. See `docs/reactive_schematization_model.md`.

---

## Citation

> Citation details omitted for anonymous peer review. A full reference,
> including authorship and DOI, will be restored on acceptance.

If you use the corpora themselves, please also cite CHILDES
(MacWhinney, 2000) and the individual corpus contributors.

---

## License

Code and aggregated results in this repository are released under the
**MIT License** (see `LICENSE`). This license covers **only** the contents of
this repository; it does **not** extend to the CHILDES/TalkBank source corpora,
which remain governed by the [TalkBank Ground Rules](https://talkbank.org/share/rules.html).
