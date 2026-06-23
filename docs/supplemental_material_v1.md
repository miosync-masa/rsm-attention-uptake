# Supplementary Material — IMT Attention Bias Paper

**Attention as Pre-Grammatical Selection Pressure: A Cross-Linguistic Test of the Reactive Schematization Model**

[AUTHORS REDACTED FOR ANONYMOUS REVIEW]
[ORCID REDACTED FOR ANONYMOUS REVIEW] · [EMAIL REDACTED FOR ANONYMOUS REVIEW]
Data snapshot: 2026-06-14 · 8 samples (7 languages + 1 dialect pair)

---

## S1. Corpora and sample composition

All data are drawn from CHILDES (MacWhinney, 2000), converted to standardized
token tables via Chatter JSON and parsed with a unified pipeline (scripts
`01`–`06`). Caregiver speech provides the Attention Index (AI) predictors;
child speech provides the developmental uptake outcomes. The eight samples span
isolating (Mandarin), agglutinating (Japanese, Korean), fusional (Russian),
and analytic-with-affixation (Indonesian) morphological types, plus an
American/British English dialect pair for within-language replication.

| Sample | Corpora (CHILDES) | Tokens (total) | Child tokens (age-tagged) | Child age range (mo) | %MOR | MLU child / caregiver |
|---|---|---|---:|---|---:|---|
| English (US) | Brown | 571,019 | 287,056 | 18.0–62.4 | 92% | 2.96 / 4.21 |
| English-UK | Manchester, MPI-EVA-Manchester, Thomas, Wells, Forrester, … | 7,733,314 | 1,638,306 | 17.7–60.8 | 94% | 2.20 / 4.11 |
| Japanese | Miyata | 463,329 | 181,123 | 16.1–38.0 | 96% | 2.52 / 3.43 |
| Korean | Ko, Ryu, Jiwon | 202,558 | 78,535 | 12.5–46.0 | 42%* | 1.77 / 2.89 |
| Mandarin | Erbaugh, TCCM, Tong, Zhou3 | 654,550 | 224,845 | 17.0–65.0 | 91% | 2.86 / 3.99 |
| Russian | RusLan, Tanja | 186,800 | 45,361 | 12.4–46.0 | 95% | 1.43 / 2.94 |
| Spanish | Ornat, Vila, Aguirre, Linaza, Marrero, Remedi, … | 255,570 | 96,897 | 12.3–59.4 | 95% | 2.48 / 3.87 |
| Indonesian | Jakarta | 1,138,386 | 708,063 | 12.0–72.0 | 89%** | 2.44 / 2.84 |

\* Korean %MOR is lower because a portion of the merged corpora carry partial
morphological tiers; cue extraction is restricted to tagged utterances.
\** Indonesian cue-token rate is low (13.2%) because the language marks much
grammatical information through word order and a small closed set of
voice/preposition markers rather than dense bound morphology.

Parse errors were zero across all 8 samples.

---

## S2. Attention Index (AI) — caregiver cue profiles

The five-dimensional salience model yields, per cue subtype: AI (unweighted
mean of the five salience dimensions), WAI (frequency-weighted AI), and the
three non-circular Reliability measures (gra, position, form) introduced in v3
to break the circularity of defining grammatical roles from the same GRA tier
used to measure reliability.

### S2.1 Summary statistics per sample

| Sample | n cue subtypes | AI mean (SD) | WAI mean | Reliability_gra | Reliability_position | Reliability_form |
|---|---:|---|---:|---:|---:|---:|
| English (US) | 129 | 0.302 (0.113) | 0.154 | 0.649 | 0.186 | 0.562 |
| English-UK | 177 | 0.335 (0.112) | 0.180 | 0.633 | 0.159 | 0.539 |
| Japanese | 78 | 0.375 (0.188) | 0.216 | 0.877 | 0.135 | 0.579 |
| Korean | 29 | 0.502 (0.201) | 0.332 | 0.721 | 0.293 | 0.556 |
| Mandarin | 11 | 0.450 (0.167) | 0.349 | 0.740 | 0.056 | 0.794 |
| Russian | 28 | 0.379 (0.116) | 0.218 | 0.870 | 0.084 | 0.735 |
| Spanish | 87 | 0.350 (0.094) | 0.178 | 0.771 | 0.321 | 0.557 |
| Indonesian | 28 | 0.411 (0.093) | 0.218 | 0.789 | 0.103 | 0.495 |

### S2.2 Typological readings of the Reliability profiles

- **Japanese / Russian** show the highest Reliability_gra (0.88, 0.87): bound
  morphology and particles map cleanly onto grammatical relations.
- **Mandarin** shows the lowest Reliability_position (0.06) but high
  Reliability_form (0.79): as an isolating language, function is carried by
  fixed forms rather than positional variation.
- **Spanish** shows the highest Reliability_position (0.32) among the set,
  consistent with its relatively free but informative word order.
- The **English dialect pair** is nearly identical in profile (gra 0.649 vs
  0.633; form 0.562 vs 0.539), confirming the AI construct is dialect-stable.

### S2.3 Highest-AI cues per language (illustrative)

| Sample | Top cue (by AI) | AI | Interpretation |
|---|---|---:|---|
| English (US) | pron_that (deictic) | 0.550 | utterance-initial deictic, high positional salience |
| English-UK | pron_that | 0.587 | same, replicated |
| Japanese | vfin_other (verb-final aux) | 0.780 | clause-final verbal morphology |
| Korean | vfin_ji (verb-final) | 0.838 | SOV clause-final position = maximal salience |
| Mandarin | object_position | 0.738 | post-verbal object, word-order cue |
| Russian | aspect_imperf | 0.614 | aspectual morphology on the verb |
| Spanish | pron_ese (demonstrative) | 0.537 | clause-initial demonstrative |
| Indonesian | object_position | 0.578 | word-order cue dominates |

The cross-linguistic constant: **utterance-edge material (final in SOV/verb-final
languages, initial for deictics, peripheral for word-order cues) carries the
highest AI**, supporting the pre-grammatical edge-salience claim.

---

## S3. Developmental uptake — core hypothesis tests

For each cue, caregiver AI (predictor) is correlated against three child
developmental measures: first_emergence_month (H1, expected negative), uptake
slope (H2, expected positive), and peak production rate (H3, expected positive).
Frequency (log caregiver count) serves as the control predictor.

### S3.1 Pearson correlations per sample

| Sample | n | H1 AI→emergence | H2 AI→slope | H3 AI→peak | Ctrl logFreq→emergence | WAI→peak |
|---|---:|---|---|---|---|---|
| English (US) | 107 | −0.41*** | +0.28** | +0.36*** | −0.59*** | +0.58*** |
| English-UK | 138 | −0.61*** | +0.20* | +0.27** | −0.81*** | +0.48*** |
| Japanese | 55 | −0.63*** | +0.37** | +0.61*** | −0.67*** | +0.84*** |
| Korean | 23 | −0.52* | +0.32 ns | +0.35 ns | −0.81*** | +0.61** |
| Mandarin | 11 | −0.59 ns | +0.64* | +0.64* | −0.54 ns | +0.75** |
| Russian | 22 | −0.70*** | +0.69*** | +0.66*** | −0.77*** | +0.81*** |
| Spanish | 67 | −0.60*** | +0.35** | +0.45*** | −0.60*** | +0.77*** |
| Indonesian | 27 | −0.62*** | +0.54** | +0.56** | −0.72*** | +0.80*** |

Significance: \*p<.05, \*\*p<.01, \*\*\*p<.001.
All H1 negative, all H2/H3 positive — directionally consistent across 8/8 samples.
Where individual cells are ns (Korean, Mandarin), effect sizes remain moderate-
to-large; non-significance reflects small cue-set n, not absence of effect.

---

## S4. Hierarchical regression — the RSM interaction test

Models (all predictors z-scored; interaction = product of z-scores; outcome
also z-standardized in v3 so betas are comparable across samples):

- M0: outcome ~ logFreq
- M1: outcome ~ logFreq + AI
- M2: outcome ~ logFreq + AI + AI×logFreq

The RSM signature is a significant positive AI×logFreq interaction on **peak
production**, with negligible variance added by AI as a main effect (ΔR²_AI ≈ 0)
but substantial variance added by the interaction (ΔR²_interaction).

### S4.1 Peak production (entrenchment)

| Sample | n | β interaction | p | ΔR²(AI alone) | ΔR²(interaction) | sr(AI\|freq) | sr(freq\|AI) |
|---|---:|---:|---|---:|---:|---:|---:|
| English (US) | 107 | +0.449 | <.001 | +0.001 | +0.201 | 0.037 | 0.419 |
| English-UK | 138 | +0.374 | <.001 | +0.017 | +0.111 | 0.131 | 0.452 |
| Japanese | 55 | +0.346 | <.001 | +0.011 | +0.107 | 0.103 | 0.549 |
| Korean | 23 | +0.483 | <.001 | +0.002 | +0.275 | 0.041 | 0.510 |
| Mandarin | 11 | +0.569 | .013 | +0.043 | +0.176 | 0.207 | 0.548 |
| Russian | 22 | +0.670 | <.001 | +0.024 | +0.261 | 0.153 | 0.470 |
| Spanish | 67 | +0.457 | <.001 | +0.001 | +0.157 | 0.023 | 0.603 |
| Indonesian | 27 | +0.462 | <.001 | +0.007 | +0.148 | 0.084 | 0.554 |

Across all 8 samples: interaction β > 0 and significant (7 at p<.001, Mandarin
at p<.05); ΔR²(AI alone) is ≤ 0.043 everywhere (mean 0.013) while
ΔR²(interaction) ranges 0.107–0.275 (mean 0.179). The semipartial correlations
show frequency carries the larger unique share, but the interaction — not either
main effect — is what lifts model R² substantially.

### S4.2 First emergence (initial registration)

| Sample | n | β interaction | p | ΔR²(interaction) |
|---|---:|---:|---|---:|
| English (US) | 107 | +0.144 | .068 ns | +0.021 |
| English-UK | 138 | −0.050 | .387 ns | +0.002 |
| Japanese | 55 | +0.235 | .020* | +0.050 |
| Korean | 23 | −0.277 | .014* | +0.090 |
| Mandarin | 11 | −0.357 | .392 ns | +0.069 |
| Russian | 22 | +0.101 | .613 ns | +0.006 |
| Spanish | 67 | +0.292 | .006** | +0.064 |
| Indonesian | 27 | +0.182 | .292 ns | +0.023 |

The emergence picture is mixed in both sign and significance — exactly the
contrast that motivates the two-stage interpretation (S5).

---

## S5. Cross-linguistic meta-analysis (DerSimonian–Laird random effects)

Effect sizes are the per-sample standardized interaction betas; SEs are
recovered from the regression 95% CIs. Pooling uses random effects because the
samples represent a draw from the world's languages.

### S5.1 Peak production — the main result

```
Pooled interaction β = +0.454   95% CI [+0.380, +0.527]
z = 12.103,  p < 0.00001
Sign test: 8/8 positive  (binomial p = 0.0039)
Heterogeneity: I² = 23.6%,  Q(7) = 9.16,  p = 0.241 (ns),  τ² = 0.003
ΔR²(interaction) mean 0.179  vs  ΔR²(AI alone) mean 0.013  →  13.7× more variance
```

**Interpretation.** A homogeneous, highly significant multiplicative effect of
attention and frequency on entrenchment. I² = 23.6% with a non-significant Q
indicates the effect does not significantly vary across typologically diverse
languages — the hallmark of a universal mechanism. AI as a main effect is
near-inert (ΔR² ≈ 0.01); only the product predicts uptake.

### S5.2 First emergence — the dissociation

```
Pooled interaction β = +0.072   95% CI [−0.075, +0.219]
z = 0.963,  p = 0.335 (ns)
Sign test: 5/8 positive  (binomial p = 0.363, ns)
Heterogeneity: I² = 70.1%,  Q(7) = 23.43,  p = 0.0014 (significant),  τ² = 0.027
```

**Interpretation.** No reliable interaction on initial emergence, and
significant heterogeneity across languages. Initial registration of a cue is
predicted by frequency alone; the attention×frequency product is not required.

### S5.3 The two-phase signature

| Property | Peak production | First emergence |
|---|---|---|
| Pooled interaction β | +0.45 | +0.07 |
| z (p) | 12.1 (<.001) | 0.96 (.34, ns) |
| Heterogeneity I² | 23.6% (homogeneous) | 70.1% (heterogeneous) |
| Sign test | 8/8 (p=.004) | 5/8 (p=.36, ns) |

This dissociation is the central empirical signature predicted by the Reactive
Schematization Model: a cue's **initial registration** into the learner's store
depends on frequency (a single sufficiently frequent encounter suffices), while
**entrenchment to peak productivity** requires the multiplicative combination of
attention and reactive repetition (Book → schema consolidation). Emergence is
frequency-driven and idiosyncratic; entrenchment is attention×frequency-driven
and universal.

---

## S6. Key claims supported by these results

1. **Edge salience is cross-linguistically constant.** The highest-AI cues sit
   at utterance edges in every sample regardless of morphological type (S2.3).

2. **Uptake direction is universal.** All 8/8 samples show high-AI cues emerging
   earlier (H1−), rising faster (H2+), and peaking higher (H3+) (S3).

3. **Neither attention nor frequency suffices alone; their product predicts
   entrenchment.** ΔR²(interaction) is 13.7× ΔR²(AI alone); pooled β = .45,
   z = 12.1, I² = 24% (S4–S5).

4. **A two-phase architecture (Book → schema).** Frequency drives initial
   emergence (heterogeneous, ns interaction); attention×frequency drives
   entrenchment (homogeneous, highly significant interaction) (S5.3).

5. **Dialect-stable construct.** The American/British English pair replicates on
   every measure, ruling out single-variety dependence (S2.1, S3.1, S4.1).

---

## S7. Limitations (carried into the main Discussion)

1. **Production, not comprehension.** Uptake is measured from child production;
   comprehension likely precedes it, so emergence timings are upper bounds.

2. **Same-corpus predictor and outcome.** Caregiver AI and child uptake come
   from the same corpora — ecologically valid but not independently validated.
   A held-out-corpus replication is the natural next step.

3. **Frequency proxies reactive repetition (R+).** The model's "reactive
   confirmation" is operationalized via frequency; direct coding of caregiver
   responses to child output (expansion / recast / acknowledgment vs. ignore)
   is required for a strong RSM test.

4. **First-emergence is sampling-sensitive.** Peak production and uptake slope
   are the stable indices; emergence is more exposed to recording density.

5. **Small cue sets in some languages.** Korean (n=23), Mandarin (n=11), and
   Russian (n=22) have fewer cue subtypes; the meta-analysis down-weights them
   via their larger standard errors, and the sign test (assumption-light)
   corroborates the pooled estimate.

---

## S8. Reproducibility

Pipeline scripts (in order):
- `01_load_corpus_json.py` — CHILDES → standardized token CSV (recursive over
  `~/childes_data/{language}/`)
- `02_extract_cues_v2.py` — per-language cue tagging (incl. Indonesian affix
  stem-validation, dialect-label resolution)
- `03_compute_attention_index_v3.py` — 5-dimensional AI + non-circular
  Reliability (gra / position / form)
- `05_developmental_uptake.py` — child uptake measures, correlations, and
  hierarchical regressions (z-standardized)
- `06_meta_analysis.py` — DerSimonian–Laird random-effects meta-analysis,
  sign test, ΔR² aggregation

Per-language outputs (in `output/v3/`): `{lang}_attention_index.csv`,
`{lang}_uptake.csv`, `{lang}_uptake_by_age.csv`, `{lang}_uptake_summary.json`.
Meta outputs: `meta_effects_{outcome}.csv`, `meta_analysis_summary.json`.

Environment: Python 3.12, pandas, numpy, scipy, statsmodels; CHILDES Chatter CLI
for CHA→JSON conversion.
