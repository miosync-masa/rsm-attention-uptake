# Cross-Linguistic Cue Candidate List (v1)

**IMT Attention Bias Paper — Supplementary Document**
*Date: 2026-06-13 | Maintainer: Torami × Boss*

---

## Purpose

This document specifies, for each of the five target languages, the **grammatical cues**
that learners must attend to in order to recover argument structure
(who-did-what-to-whom). For each cue, we list:

- **Locus**: where the cue appears in the utterance
- **Function**: what grammatical information it carries
- **Detection method**: how to extract it from CHILDES CHAT/XML files
- **Expected salience profile**: which of the 5 dimensions (acoustic / positional /
  frequency / repetition / perceptual) we predict it scores high on
- **Documented acquisition trajectory**: existing findings on when children acquire it

This list defines the **operational scope** of the Attention Index computation.

---

## 1. English (Eng-NA, Brown corpus)

English is a **word-order-dominant, low-morphology** language. Argument roles are
primarily signaled by **position** relative to the verb.

| # | Cue | Locus | Function | Detection | Expected salience |
|---|-----|-------|----------|-----------|-------------------|
| E1 | SVO word order | Linear position | Agent / Patient role | Token order around main verb | High positional, low acoustic |
| E2 | Subject-verb agreement (-s) | Verb morphology | 3rd-singular present | Suffix on verb token | Low acoustic (unstressed), low frequency |
| E3 | Past tense (-ed / irregular) | Verb morphology | Temporal anchoring | Suffix or stem alternation | Medium frequency, high error site |
| E4 | Auxiliary placement (inversion) | Sentence position | Question / declarative | Position of *is/do/can* | High positional |
| E5 | Determiners (the/a) | Pre-nominal slot | Definiteness, NP boundary | Token before noun | Very high frequency, low semantic load |
| E6 | Prepositions (to, on, in, with) | Pre-NP | Spatial / role assignment | Closed-class token before NP | High frequency, high reliability |
| E7 | Pronouns (he/him/she/her) | Argument slot | Case + reference | Closed-class lexical set | High frequency, dual case marking |

**Key overgeneralization sites** (for Prediction 3):
- *goed, foots, mouses, sheeps* → past tense / plural overregularization
- Pronoun case errors: *me did it / her went home*

---

## 2. Japanese (Japanese, Miyata corpus)

Japanese is a **particle-dominant, SOV-default, drop-rich** language. Argument roles
are signaled by **post-positional case particles** that attach to NPs.

| # | Cue | Locus | Function | Detection | Expected salience |
|---|-----|-------|----------|-----------|-------------------|
| J1 | が (ga) | Post-NP | Nominative / subject | Particle token after noun | High positional (post-NP), medium frequency |
| J2 | を (o/wo) | Post-NP | Accusative / object | Particle token after noun | High reliability, medium frequency |
| J3 | に (ni) | Post-NP | Dative / goal / location | Particle token after noun | High reliability, polysemous |
| J4 | は (wa) | Post-NP | Topic (not case) | Particle token after noun | High frequency, contrastive with が |
| J5 | で (de) | Post-NP | Instrument / location | Particle token after noun | Medium frequency |
| J6 | Verb-final morphology (-ta / -te / -ru / -masu) | Utterance-final | Tense / aspect / register | Suffix on utterance-final verb | **Very high positional (utterance-end)**, high acoustic |
| J7 | Potential -reru / -rareru | Verb morphology | Ability / possibility | Suffix on verb | Overgeneralization site (ら抜き言葉) |
| J8 | SOV vs OSV order (scrambling) | Linear position | Information structure | Token order | Low (particles do the work) |

**Key overgeneralization sites**:
- 食べれる instead of 食べられる (ら抜き)
- が/は confusion (topic vs nominative)
- に/で confusion (location-goal)

---

## 3. Korean (Korean, e.g., Ryu or Jiwon corpus)

Korean is structurally similar to Japanese (SOV, particle-marking) but adds **rich
verbal morphology including honorifics and connective endings**.

| # | Cue | Locus | Function | Detection | Expected salience |
|---|-----|-------|----------|-----------|-------------------|
| K1 | 이/가 (i/ga) | Post-NP | Nominative | Particle after noun | High positional |
| K2 | 을/를 (eul/reul) | Post-NP | Accusative | Particle after noun | High reliability |
| K3 | 에/에서 (e/eseo) | Post-NP | Location / source | Particle after noun | Medium frequency |
| K4 | 께/한테 (kke/hante) | Post-NP | Dative (honorific / plain) | Particle after noun | Social register marker |
| K5 | Verb-final endings (-da/-yo/-sumnida) | Utterance-final | Speech level / register | Suffix on utterance-final verb | Very high positional |
| K6 | Connective endings (-go/-seo/-myeon) | Verb-medial | Clause linking | Suffix on non-final verb | High frequency |
| K7 | Honorific -si- | Verb-internal | Subject honorification | Infix | Low salience but high reliability for social marking |
| K8 | Subject drop rate | Argument omission | Discourse-dependence | Count of null arguments | Structural cue |

**Key overgeneralization sites**:
- Particle 이/가 vs 을/를 swapping in early acquisition
- Honorific overuse / underuse

---

## 4. Mandarin Chinese (EastAsian/Chinese/Mandarin, Tong corpus)

Mandarin is a **tonal, isolating, word-order-and-construction-dominant** language. It
has minimal morphology but rich **construction markers** (ba, bei, le, de).

| # | Cue | Locus | Function | Detection | Expected salience |
|---|-----|-------|----------|-----------|-------------------|
| M1 | SVO baseline word order | Linear position | Default argument structure | Token order | High positional |
| M2 | 把 (bǎ) construction | Pre-object marker | Disposal / object fronting | Token *ba* before object NP | Construction-level cue |
| M3 | 被 (bèi) construction | Pre-agent marker | Passive | Token *bei* | Low frequency, high reliability |
| M4 | 了 (le) | Post-verbal or sentence-final | Perfective / change of state | Position-dependent function | **Positional ambiguity** = high learning challenge |
| M5 | 的 (de) | Modifier-head linker | Possession / relativization | Token between modifier and head | High frequency |
| M6 | Tone (1-4 + neutral) | Syllable-internal | Lexical identity | F0 contour per syllable | **Very high acoustic salience**, mandatory |
| M7 | Classifiers (个/只/张/etc.) | Pre-noun after numeral | NP structure | Token between Num and N | High frequency, semantic class marker |
| M8 | Aspect markers (着/过) | Post-verbal | Aspect (progressive / experiential) | Suffix on verb | Medium frequency |

**Key overgeneralization sites**:
- Tone errors (especially T2/T3 contrast)
- 了 (le) placement errors
- Classifier overgeneralization (个 used for everything)

---

## 5. Russian (Slavic/Russian, Protassova corpus)

Russian is a **case-rich, free-word-order, gender-marked** Indo-European language.
Argument roles are encoded primarily by **nominal case inflection**.

| # | Cue | Locus | Function | Detection | Expected salience |
|---|-----|-------|----------|-----------|-------------------|
| R1 | Nominative case | Noun ending | Subject | Suffix on noun | High frequency (citation form) |
| R2 | Accusative case | Noun ending | Direct object | Suffix on noun (often animacy-dependent) | High reliability |
| R3 | Dative case | Noun ending | Recipient / experiencer | Suffix on noun | Medium frequency |
| R4 | Genitive case | Noun ending | Possession / negation / partitive | Suffix on noun | High frequency, polyfunctional |
| R5 | Instrumental case | Noun ending | Instrument / means / predicate | Suffix on noun | Medium frequency |
| R6 | Prepositional case | Noun ending | Location (with preposition) | Suffix + preposition | Always co-occurs with preposition |
| R7 | Gender agreement (M/F/N) | Adjective / verb suffix | NP head gender | Suffix matching head | Cross-word cue |
| R8 | Verb aspect (perf / imperf) | Lexical pairs / prefixes | Event structure | Verb stem identity | **Hardest cue cross-linguistically** |
| R9 | Free word order | Linear position | Information structure (not roles) | Token order | Low reliability for roles |

**Key overgeneralization sites**:
- Case ending overregularization (e.g., -ы/-и plural endings)
- Aspectual pair confusion
- Animacy-dependent accusative errors

---

## Cross-Linguistic Cue Mapping for Argument Identification

This is the **theoretically central comparison**: how does each language signal
"who did what to whom"?

| Function | English | Japanese | Korean | Mandarin | Russian |
|----------|---------|----------|--------|----------|---------|
| Agent identification | Pre-verbal position (E1) | が (J1) | 이/가 (K1) | Pre-verbal position (M1) | Nominative case (R1) |
| Patient identification | Post-verbal position (E1) | を (J2) | 을/를 (K2) | Post-verbal or 把-marked (M1, M2) | Accusative case (R2) |
| Recipient identification | "to" + NP (E6) | に (J3) | 께/한테 (K4) | 给 + NP | Dative case (R3) |
| Event type | Verb morphology (E3) | Verb-final morph (J6) | Verb-final morph (K5) | Aspect markers (M8) | Verb aspect (R8) |

**Critical observation**: the **locus** of grammatical information differs systematically:

- **English / Mandarin**: linear position (pre/post-verbal)
- **Japanese / Korean**: post-nominal particles + utterance-final morphology
- **Russian**: noun-internal case inflection

This locus difference is what the Attention Index must capture.

---

## Notes for Implementation

1. **Tokenization layer**: CHAT format uses `%mor` tier for morphological tagging. For
   particles (Japanese/Korean) and case endings (Russian), `%mor` tier provides
   gold-standard segmentation. For Mandarin tones, lexical entries in `%mor` tag
   tone numbers (e.g., `ma1` vs `ma3`).

2. **Cue ambiguity**: Some cues are polyfunctional (Japanese に, Russian Genitive,
   Mandarin 了). These should be counted **per function**, not per surface form,
   to compute reliability properly.

3. **Null arguments**: Japanese / Korean / Mandarin allow extensive argument drop.
   Drop rates themselves are a cue (signaling that other cues must do more work).

4. **Reference corpus per language** (initial, can expand):
   - English: Brown (Adam / Eve / Sarah)
   - Japanese: Miyata (Aki / Ryo / Tai)
   - Korean: Ryu (one child longitudinal)
   - Mandarin: Tong (longitudinal)
   - Russian: Protassova (longitudinal Russian-Finnish bilingual; consider also
     Tanja monolingual corpus if available)

---

*End of v1. Revisions will be tracked in this file.*
