# Attention Index: Operational Definition

**IMT Attention Bias Paper — Mathematical Formalization v1**
*Date: 2026-06-13 | Maintainer: Torami × Boss*

---

## 1. Conceptual Definition

The **Attention Index** AI(c, L, t) is a scalar quantity that estimates the
*expected attentional weight* a learner of language L at developmental
time t allocates to grammatical cue c.

It is a **latent variable**, inferred (not directly measured) from converging
observables in caregiver input, child output, and computational ablation.

---

## 2. Operational Decomposition

We decompose AI into five orthogonal-ish dimensions, each operationally defined.
The decomposition follows Torami's salience taxonomy proposed during the
2026-06-13 working session.

For cue c in language L:

```
AI(c, L) = w_a · S_acoustic(c, L)
        + w_p · S_positional(c, L)
        + w_f · S_frequency(c, L)
        + w_r · S_repetition(c, L)
        + w_d · S_perceptual(c, L)
```

with cross-linguistically fixed weights:
- w_a + w_p + w_f + w_r + w_d = 1
- w_i ≥ 0 for all i

The constraint of cross-linguistic weight fixity is **critical for falsifiability**:
if the weights are allowed to vary per language, the model becomes unfalsifiable.

We propose, as initial defaults, w_a = w_p = w_f = w_r = w_d = 0.2 (uniform prior),
to be sensitivity-analyzed in §6.

---

## 3. The Five Dimensions

### 3.1 Acoustic Salience S_acoustic(c, L)

Defined as the average phonetic prominence of cue tokens in caregiver speech.
We use **z-score normalization within speaker × utterance** to control for
individual differences in speaking style.

For cue token x of type c in utterance u:

```
acoustic_prominence(x) = α · z(F0_range(x))
                      + β · z(intensity(x))
                      + γ · z(duration(x))
```

with α = β = γ = 1/3 (uniform prior, to be tuned).

Aggregate to cue level:

```
S_acoustic(c, L) = mean over all tokens x of type c in CDS corpus L of
                   acoustic_prominence(x)
```

**Data source**: PhonBank phonological annotations when available; otherwise
estimated via Praat/Parselmouth on aligned audio. For text-only corpora,
S_acoustic falls back to **utterance-position priors** (utterance-final cues
get a constant acoustic bonus, justified by lengthening effects).

### 3.2 Positional Salience S_positional(c, L)

Defined as how predictable the cue's position is within the utterance.
**Predictable position = high salience**, because the learner's attention can
anticipate where to look.

We use **utterance-edge bias**: positions near utterance edges (first or last
token) are most salient.

For cue c with position distribution P(pos | c) over normalized positions
[0, 1] in utterance:

```
S_positional(c, L) = max(P(pos=0..0.1 | c), P(pos=0.9..1.0 | c))
                    - uniform_baseline
```

where uniform_baseline = 0.1 (the value for a uniformly distributed cue).

**Interpretation**: a cue that appears in the first 10% or last 10% of
utterances gets a high score; a cue uniformly distributed gets ~0.

### 3.3 Frequency Salience S_frequency(c, L)

Defined as **log-token frequency** of the cue in caregiver speech, normalized
per language to enable cross-linguistic comparison.

```
raw_freq(c, L) = count(c) / total_tokens(L_CDS)
S_frequency(c, L) = log(raw_freq(c, L)) - mean_log_freq(L)
                  ──────────────────────────────────────
                  std_log_freq(L)
```

Z-score per language ensures we measure **relative prominence**, not absolute.

**Note**: extremely frequent cues (e.g., English "the") may have *reduced*
attentional weight by habituation. This is a known issue. We model habituation
in §5 as a Book-side correction.

### 3.4 Repetition Salience S_repetition(c, L)

Defined as the type-token-ratio inverse: cues that appear with **many different
host words** are more cue-like (less lexicalized), thus more salient *as cues*.

For cue c attached to host word set H_c in CDS:

```
S_repetition(c, L) = log(|H_c|) - log_min_H
                    ─────────────────────
                    log_max_H - log_min_H
```

normalized to [0, 1] within the language.

**Rationale**: English "-ed" attached to many verbs → cue-like.
"-ed" only attached to "walked" → memorized, not cue-like.

### 3.5 Perceptual Salience S_perceptual(c, L)

Defined as the **acoustic distinctiveness** of the cue from its neighbors
in phonological space. Cues that form *minimal contrasts* are easier to detect.

```
S_perceptual(c, L) = min over closest_contrast c' of
                     phonetic_distance(c, c') / max_distance(L)
```

For text corpora without phonetic annotation, we approximate using
**orthographic edit distance** to nearest contrastive form, normalized by
typical word length in the language.

**Example**: Japanese が/を are phonetically distant (g-vowel vs vowel-only) →
high S_perceptual. が/は phonetically distinct (g vs h) → high S_perceptual.
English "-s" vs nothing (zero morpheme) → low S_perceptual.

---

## 4. Cue Reliability (Companion Variable, Not Part of AI)

The Attention Index measures **what attracts attention**.
Cue **reliability** measures **how informative** the cue is, once attended.

These are **separate variables** in our framework, and their interaction is
the empirical heart of the paper.

For cue c predicting grammatical function f:

```
Reliability(c, f) = P(f | c) = count(c ∧ f) / count(c)
```

For multi-function cues (e.g., Japanese に for dative, location, goal):

```
Reliability(c) = max over f of Reliability(c, f)
                  // most-predictable function
```

Or alternatively, the **entropy-based** definition:

```
Reliability(c) = 1 - H(F | c) / log(|F|)
                  // 1 minus normalized conditional entropy
```

We will report both in the paper.

---

## 5. Book Maturity (Developmental Modulator)

The Attention Index alone does not predict acquisition order: a high-AI cue
in a low-Book learner may not yet be usable. We model this with a Book
maturity index.

For a child at age t, Book maturity is operationalized as:

```
B(t) = w_voc · z(vocabulary_size(t))
     + w_mlu · z(MLU(t))
     + w_ttr · z(type_token_ratio(t))
```

with w_voc = 0.5, w_mlu = 0.3, w_ttr = 0.2 (working defaults, sensitivity-tested).

**External reference**: Wordbank CDI percentiles provide language-normed
vocabulary size estimates. MLU follows Brown (1973).

---

## 6. The Acquisition Prediction Equation

The central prediction of the model:

```
Productivity(c, t) = σ( a₀ + a₁ · AI(c, L) + a₂ · Reliability(c)
                       + a₃ · B(t) + a₄ · AI · B(t)
                       + ε )
```

where σ is the logistic function and Productivity is the proportion of
opportunities in which the child correctly uses cue c at time t.

**Key parameter**: a₄ (interaction term).

Our model predicts a₄ > 0: high-AI cues become productive only after Book
matures. **This is the testable signature of content-function separation.**

If a₄ ≈ 0, the model reduces to pure statistical learning.
If a₁ ≈ 0, the model reduces to pure UG-driven (cue-blind) acquisition.
**Only a₁ > 0 ∧ a₄ > 0 supports our hypothesis.**

---

## 7. Overgeneralization Prediction (Hypothesis 3)

For a cue c, overgeneralization rate is predicted to peak when:

```
peak_t = argmax_t  AI(c, L) · (1 - Coverage(c, B(t)))
```

where Coverage(c, B(t)) is the proportion of exceptions to cue c that the child
has stored in Book.

**Operationally**, Coverage is approximated by:

```
Coverage(c, B(t)) ≈ |child_lexicon(t) ∩ exceptions(c)|
                    ──────────────────────────────────
                    |exceptions(c)|
```

For English past tense: exceptions(c) = irregular verbs.
For Japanese potential: exceptions(c) = group-1 verbs that take -reru.

**Prediction**: the U-shape (correct → over → correct) emerges naturally:
- Phase 1 (low coverage, low AI use): individual memorization, correct
- Phase 2 (low coverage, high AI use): productive cue, overgeneralization
- Phase 3 (high coverage, high AI use): productive with exception lookup, correct

---

## 8. Falsifiability Conditions

The framework is **falsified** if any of the following holds across languages:

| Condition | Falsifier outcome |
|-----------|-------------------|
| AI(c, L) does not predict cue acquisition order within L | a₁ ≤ 0 in Eq. §6 |
| Same AI value across languages → different acquisition timing | Cross-linguistic invariance fails |
| Overgeneralization peaks independent of AI × Book interaction | §7 prediction fails |
| Some language shows acquisition perfectly predicted by raw frequency alone | content-function separation unnecessary |
| Attention bias requires language-specific weights w_i | universal Attention bias claim fails |

The third condition is the **hardest**: if frequency alone explains acquisition
in any single language, our model is over-engineered for that language.

---

## 9. Computational Ablation as Independent Estimator

The convergent validation comes from a separate, computational estimate of
"which cues matter most" — derived without observing any child.

Train a semantic-role-labeling model on CDS for language L:
- Input: tokenized utterance with all cue features
- Output: argument role labels (Agent, Patient, Goal, etc.)

For each cue c, compute:

```
Ablation_Importance(c, L) = Accuracy(full model) - Accuracy(model without cue c)
```

**Convergence prediction**: Ablation_Importance(c, L) ≈ AI(c, L) × Reliability(c)
modulo scaling.

If the two estimates correlate strongly (r > 0.7 across cues, within language),
this is strong evidence that **AI captures what is computationally informative**,
not just what is intuitively salient.

---

## 10. Summary Equation Stack

```
AI(c, L)            = Σ wᵢ · Sᵢ(c, L)              // §2
Reliability(c)      = max_f P(f | c)                  // §4
B(t)                = Σ wⱼ · z(metricⱼ(t))           // §5
Productivity(c, t)  = σ(a₀ + a₁·AI + a₂·R + a₃·B + a₄·AI·B + ε)  // §6
Overgen_peak(c)     = argmax_t AI(c,L)·(1−Cov(c,B(t)))  // §7
Ablation_Imp(c, L)  ≈ AI(c, L) · Reliability(c)       // §9
```

---

## Open Questions for Boss

1. **Should w_i be free parameters or fixed?**
   Fixed = strong falsifiable claim. Free = better fit but weaker theory.
   Torami's recommendation: **fixed at 0.2 each in main analysis, free as
   sensitivity check**.

2. **Should we treat Reliability as part of AI or separate?**
   Current draft: separate. Rationale: reliability is content-side (Book uses it),
   not function-side (Attention uses it).

3. **What's the "minimum reportable effect"?**
   Suggest: a₁ standardized coefficient > 0.2 with 95% CI not crossing zero,
   and r > 0.5 between AI and Ablation_Importance.

---

*End of v1.*
