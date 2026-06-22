# 17g Manchester diagnostics — outcome window N=5

## Section 1 — Group descriptives (UK vs Manchester, post-MSR contingent)

| Metric | UK mean | Manc mean | Δ | MW p |
|---|---|---|---|---|
| n_files | 20.417 | 22.636 | -2.220 | 0.0001 |
| n_utt_caregiver | 27125.417 | 29239.727 | -2114.311 | 0.0001 |
| mlu_child | 2.186 | 2.166 | +0.021 | 0.9148 |
| mlu_caregiver | 3.525 | 3.799 | -0.274 | 0.1218 |
| types_caregiver_lemmas | 1523.521 | 2567.364 | -1043.843 | 0.0004 |
| ttr_caregiver | 0.161 | 0.025 | +0.136 | 0.0002 |
| n_post_episodes | 7418.561 | 7054.727 | +363.834 | 0.0004 |
| obs_window_months | 10.665 | 9.960 | +0.705 | 0.3560 |
| episodes_per_month | 849.620 | 709.760 | +139.860 | 0.0059 |
| episodes_per_file | 262.754 | 332.556 | -69.802 | 0.0730 |

Cue-mix Spearman ρ across 137 common cues = +0.975 (p = 0.0000)


## Section 2 — Density × β

Pooled WLS (UK + Manchester children) β_i ~ density + log(n) + corpus:

| Predictor | β | SE | p |
|---|---|---|---|
| const | +0.0272 | 0.0060 | 0.0000 |
| episodes_per_month_z | +0.0049 | 0.0057 | 0.3920 |
| log_n_episodes_z | -0.0147 | 0.0049 | 0.0041 |
| is_manchester | -0.0166 | 0.0094 | 0.0841 |

UK density-strata meta:

- High-density UK (n=19): β_pooled = +0.0142, p = 0.0360
- Low-density UK (n=18): β_pooled = +0.0342, p = 0.0187

## Section 3 — Cue-mix matched UK fit

| Scenario | n_eps | n_cues | β(COI×cum) | SE | p |
|---|---|---|---|---|---|
| UK full unweighted (reference) | 303,980 | 137 | +0.0243 | 0.0096 | 0.0113 |
| UK reweighted to Manchester cue share | 303,555 | 111 | +0.0258 | 0.0082 | 0.0018 |
| UK restricted to Manchester top cues (cum≤.90) | 264,765 | 28 | +0.0202 | 0.0083 | 0.0150 |