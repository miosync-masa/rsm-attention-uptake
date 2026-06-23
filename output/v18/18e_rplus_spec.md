# 18e — R+ specification (SI S9)

**Source code:** `10_extract_R_plus_v2.py` (extraction), `17n_rplus_drop_test.py` (champion model).

---

## 1. r_plus_composite definition

**Formula** (`10_extract_R_plus_v2.py` lines 548–558):

```
positive(e) = 0.25 * lexical_overlap_containment(e)
            + 0.25 * target_cue_modeled(e)
            + 0.25 * morphosyntactic_change(e)
            + 0.25 * expansion_depth_normalized(e)

negative(e) = 1.0 * repair_flag(e)

r_plus_composite(e) = clip( positive(e) − negative(e),  −1.0,  +1.0 )
```

**Component definitions** (per episode `e`):

| Term | Definition |
|---|---|
| `lexical_overlap_containment` | (lemmas in caregiver utt ∩ lemmas in child utt) / (lemmas in child utt). In [0, 1]. |
| `target_cue_modeled` | Binary {0,1}. 1 ⇔ the child's target cue surface form appears in the caregiver's response. |
| `morphosyntactic_change` | Binary {0,1}. 1 ⇔ the caregiver added ≥ `--morph_change_threshold` (default 5) morph features absent from the child utt (recast-style reformulation). |
| `expansion_depth_normalized` | min(1, added_token_count / `--expansion_depth_norm`). Default norm=10. In [0, 1]. |
| `repair_flag` | Binary {0,1}. 1 ⇔ the caregiver response is **short** (≤ `--repair_max_tokens`, default 3 tokens) **and** contains a repair-lexicon item (e.g., "what?", "hmm?"). Excludes wh-question expansions per v2-Fix A. |

**Ordinal R+ depth** (computed alongside; used in §15 / §17 R+ depth tables, not in #17n):

| Depth | Label | Definition |
|---|---|---|
| 0 | no_contingent_response | Caregiver utt is not present, is a repair, or is a topic shift. |
| 1 | acknowledgment | Caregiver utt is short (≤3 added tokens) and matches acknowledgment lexicon. |
| 2 | repetition | Caregiver utt has lexical_overlap_containment ≥ 0.8. |
| 3 | expansion | Lexical overlap > 0, added_token_count > 0, target_cue_modeled = 1, and morph_added < morph_change_threshold. |
| 4 | recast | All of expansion's conditions plus morph_added ≥ morph_change_threshold (default 5). |

## 2. Contingency window

The caregiver utterance is treated as a contingent response **iff** it is the
**immediately next utterance** in the transcript with `is_caregiver = True`
(speaker code MOT, FAT, etc.). Specifically (lines 608–613 of `10_extract_R_plus_v2.py`):

```
next_idx = child_idx + 1
if utt_lookup[next_idx] is missing                 → not contingent (episode dropped at extraction)
elif next utterance speaker is not caregiver       → not contingent (r_plus_label = "no_contingent_response")
else                                               → contingent (r_plus_label ∈ {ack, rep, exp, recast})
```

**No real-time gap measure** is used (the CHA / Chatter JSON inputs do not
expose audio onset times). Contingency is purely turn-adjacency.

## 3. Mundlak within / between — IMPORTANT CLARIFICATION

**SPEC #18e item 2 asked:** "rplus_within / rplus_between — confirm these are
the Mundlak within-child deviation and child-level mean of r_plus_composite."

**What `17n_rplus_drop_test.py` actually fits is NOT child-level Mundlak.**
The decomposition is performed at the **cue level within each child**:

```python
# inside fit_per_child_model, model='with_rplus'
cue_mean_r = gg.groupby("cue_subtype")["r_plus_composite"].transform("mean")
gg["rplus_between_local"]   = cue_mean_r                        # cue-level mean (per child)
gg["rplus_episode_z_local"] = z(gg["r_plus_composite"])         # raw episode-level
gg["rplus_between_z_local"] = z(cue_mean_r)
```

So:

| Term in the regression | Mathematical content |
|---|---|
| `rplus_between_z` (a.k.a. R+_between) | z-score of the **cue-level mean** of r_plus_composite, taken across this child's contingent episodes for that cue. Constant within (child, cue). |
| `rplus_episode_z` (a.k.a. R+_within in S9 prose) | z-score of the **raw** r_plus_composite at the episode level. Varies within (child, cue) by episode-to-episode deviation. |

This is a **Mundlak-equivalent specification** because including both the raw
value and the cue-level mean is mathematically equivalent to including the
within-cue deviation (raw − cue mean) and the cue mean — the β coefficients
have different scales but the model fit and the predicted values are
identical. **The decomposition is at the cue level, not the child level**;
the SI prose should clarify this if it currently calls the within term a
"within-child" deviation.

`15_within_between_decomposition.py` (Spec A in that file) implements the
**explicit** Mundlak form (`R_within = r_plus_composite − R_between`) and
gives identical β(COI × cumulative) estimates to #17n's specification once
predictors are properly re-mapped. The two forms are interchangeable.

## 4. Per-child estimator and standard errors

**Estimator:** OLS linear-probability model (`statsmodels.OLS.fit()`).
Outcomes (`next_5_reuse`, `next_10_reuse`) are binary; β is interpreted as a
change in reuse probability per SD of the predictor. **Not a logistic /
logit GLM** at the per-child stage (see #17o §A for the Logit comparison;
LPM is the headline model).

**Standard errors:** Cluster-robust on `cue_subtype` within each child:

```python
fit = sm.OLS(y, X).fit(
    cov_type="cluster",
    cov_kwds={"groups": gg["cue_subtype"].astype(str).values},
)
```

The clustering unit is the cue subtype: episodes sharing the same target cue
are allowed arbitrary within-cluster residual correlation, which is the
right structure when the cue-level Mundlak term (R+_between) varies only at
the cue level.

The 32 per-child β / SE values are then combined by the **DerSimonian–Laird
random-effects meta-analysis** (FULL and DROP-MANC), producing the pooled β
reported in #17n SUMMARY.md and replicated in S9 Tables 3a / 3b.

## 5. Model A / Model B formulas (verification)

**S9 prose currently states:** Model A has 8 z-standardized predictors,
Model B has 5.

**What was actually run** (PREDS_BASE = 5, champion = base + 4 R+ terms = 9):

### Model A — champion (R+ kept), 9 predictors + intercept

```
next_N_reuse  ~  α
              + β_COI               · COI_z
              + β_cum                · cumulative_z
              + β_COIxcum            · (COI_z × cumulative_z)
              + β_prior              · prior_local_freq_z
              + β_logF               · log_cue_freq_z
              + β_R+_episode         · r_plus_composite_z          # episode-level
              + β_R+_between         · cue_mean_r_plus_z           # cue-level mean
              + β_R+_episode_x_COI   · (r_plus_composite_z × COI_z)
              + β_R+_between_x_COI   · (cue_mean_r_plus_z × COI_z)
              + ε    [cluster-robust SE on cue_subtype]
```

### Model B — base (R+ dropped), 5 predictors + intercept

```
next_N_reuse  ~  α
              + β_COI               · COI_z
              + β_cum                · cumulative_z
              + β_COIxcum            · (COI_z × cumulative_z)
              + β_prior              · prior_local_freq_z
              + β_logF               · log_cue_freq_z
              + ε    [cluster-robust SE on cue_subtype]
```

### ⚠️ Flag for Torami

The SI's "8 predictors in Model A" count likely treats R+ as a single main
term and R+ × COI as a single interaction (i.e., 5 base + 2 R+ = 7 + intercept
= 8). The code instead distinguishes **between** (cue-level mean) and
**within** (episode-level) R+, giving **4 R+ terms** (R+_episode,
R+_between, both × COI) and **9 predictors + intercept**.

The Δβ ≤ 8.1% / all-p < 0.001 verdict reported in #17n holds for the
champion specification (9 predictors). If the SI prose intends the simpler
2-R+-term model (single R+ main + single R+ × COI), it must either:
(a) re-state Model A as the 9-predictor champion (recommended; aligns with
    code and the published evidence), or
(b) trigger a re-run of `17n_rplus_drop_test.py` with a simplified
    R+ specification.

All predictors are z-standardized **within each child's data** before fit
(see `z()` helper used in `fit_per_child_model`). This is intentional so
that per-child β have the same "per-SD-of-predictor" interpretation across
children with different absolute scales.
