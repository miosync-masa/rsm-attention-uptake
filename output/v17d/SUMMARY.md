# SPEC #1d window-width sensitivity — outcome window N=5

## Pass criterion

β(obs_window_months_z) > 0 AND p < 0.05 in WLS, AND stratified pooled β monotonically increases with window width.


## WLS β_i ~ obs_window_z + log(n_episodes)_z + corpus dummies (weights = 1/SE²)

| predictor | β | SE | p |
|---|---|---|---|
| const | +0.0178  | 0.0188 | 0.3510 |
| obs_window_months_z | +0.0156 * | 0.0065 | 0.0215 |
| log_n_episodes_z | -0.0172 ** | 0.0055 | 0.0034 |
| corpus_English-UK | +0.0144  | 0.0197 | 0.4684 |
| corpus_Manchester | +0.0012  | 0.0237 | 0.9600 |

## Stratified random-effects meta-analysis

| Stratum | n_children | β_pooled (RE) | SE | p | τ² | I² | corpora breakdown |
|---|---|---|---|---|---|---|---|
| A_short_<6mo | 6 | +0.0222 | 0.0223 | 0.3192 | 0.0017 | 58.3% | English-UK=5, Brown=1 |
| B_medium_6-12mo | 20 | +0.0103 | 0.0089 | 0.2455 | 0.0006 | 44.6% | Manchester=11, English-UK=9 |
| C_long_>12mo | 15 | +0.0359 | 0.0114 | 0.0017 | 0.0011 | 68.3% | English-UK=13, Brown=2 |

## Brown inner contrast (Adam / Eve / Sarah)

| child | age_min | age_max | window_mo | n_eps | β | SE | p |
|---|---|---|---|---|---|---|---|
| Eve | 24.0 | 27.0 | 3.0 | 1,570 | -0.0524 | 0.0326 | 0.1074 |
| Sarah | 27.2 | 61.0 | 33.8 | 6,682 | +0.0570 | 0.0176 | 0.0012 |
| Adam | 27.1 | 62.4 | 35.3 | 6,707 | +0.0400 | 0.0172 | 0.0200 |

## Verdict: **MIXED — see strata**
