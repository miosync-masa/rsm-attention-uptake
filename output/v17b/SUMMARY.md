# SPEC #1b exposure-gate test — outcome window N=5

## Pass criterion

β(COI × cumulative_cue_attempts) > 0 AND p < 0.01 in 3/3 corpora.


## M_new (post-MSR, OLS + cluster on cue)

| Corpus | β | SE | p | verdict |
|---|---|---|---|---|
| Brown | +0.0516 | 0.0130 | 0.0001 | ✓ PASS |
| Manchester | -0.0065 | 0.0136 | 0.6326 | ✗ |
| English-UK | +0.0243 | 0.0096 | 0.0113 | ✗ |

→ **1/3 corpora pass**

## M_new (post-MSR, OLS + cluster on child)

| Corpus | β | SE | p | verdict |
|---|---|---|---|---|
| Brown | +0.0516 | 0.0095 | 0.0000 | ✓ PASS |
| Manchester | -0.0065 | 0.0051 | 0.2038 | ✗ |
| English-UK | +0.0243 | 0.0048 | 0.0000 | ✓ PASS |

→ **2/3 corpora pass**

## M_new (post-MSR, MixedLM random by child)

| Corpus | β | SE | p | verdict |
|---|---|---|---|---|
| Brown | +0.0433 | 0.0096 | 0.0000 | ✓ PASS |
| Manchester | -0.0186 | 0.0038 | 0.0000 | ✗ |
| English-UK | +0.0133 | 0.0018 | 0.0000 | ✓ PASS |

→ **2/3 corpora pass**

## S1 (pre-MSR, OLS + cluster on cue) — expected null

| Corpus | β | SE | p | verdict |
|---|---|---|---|---|
| Brown | -0.0614 | 0.0424 | 0.1476 | ✗ |
| Manchester | -0.0246 | 0.0093 | 0.0080 | ✗ |
| English-UK | -0.0168 | 0.0106 | 0.1144 | ✗ |

## S4 (post-MSR + linear age, OLS + cluster on cue)

| Corpus | β | SE | p | verdict |
|---|---|---|---|---|
| Brown | +0.0303 | 0.0143 | 0.0340 | ✗ |
| Manchester | -0.0015 | 0.0132 | 0.9124 | ✗ |
| English-UK | +0.0281 | 0.0096 | 0.0036 | ✓ PASS |