# 17i corpus-protocol diff (Manchester vs others)


Sample: up to 10 files per child (or all if smaller).


## Per-corpus medians

| Metric | Brown | Manchester | UK_long_long | UK_short_obs |
|---|---|---|---|---|
| n_utt | 1135.000 | 1558.500 | 1569.500 | 296.000 |
| n_child | 541.500 | 585.000 | 585.500 | 139.000 |
| n_caregiver | 387.500 | 916.000 | 861.000 | 134.000 |
| caregiver_to_child_ratio | 0.821 | 1.610 | 1.367 | 1.004 |
| pct_mor | 0.984 | 0.970 | 0.929 | 0.860 |
| pct_gra | 0.984 | 0.970 | 0.929 | 0.860 |
| pct_rich_dep | 0.190 | 0.029 | 0.015 | 0.175 |
| pct_zero_tokens | 0.017 | 0.016 | 0.068 | 0.045 |
| mean_tokens | 2.870 | 2.853 | 2.873 | 2.781 |
| pct_control_word_tokens | 0.039 | 0.030 | 0.036 | 0.074 |
| pct_period | 0.719 | 0.758 | 0.735 | 0.834 |
| pct_question | 0.264 | 0.229 | 0.241 | 0.152 |
| pct_exclamation | 0.003 | 0.000 | 0.001 | 0.000 |
| pct_other_term | 0.010 | 0.008 | 0.014 | 0.010 |
| n_distinct_header_types | 12.000 | 14.000 | 14.000 | 14.000 |

## Manchester vs others (Mann-Whitney p)

| Metric | vs Brown | vs UK_long_long | vs UK_short_obs |
|---|---|---|---|
| n_utt | 0.0000*** | 0.8309 | 0.0000*** |
| n_child | 0.8669 | 0.8338 | 0.0000*** |
| n_caregiver | 0.0000*** | 0.0407* | 0.0000*** |
| caregiver_to_child_ratio | 0.0000*** | 0.0130* | 0.0000*** |
| pct_mor | 0.1776 | 0.0000*** | 0.0000*** |
| pct_gra | 0.1776 | 0.0000*** | 0.0000*** |
| pct_rich_dep | 0.0000*** | 0.0000*** | 0.0000*** |
| pct_zero_tokens | 0.7413 | 0.0000*** | 0.0000*** |
| mean_tokens | 0.1825 | 0.2865 | 0.0213* |
| pct_control_word_tokens | 0.0000*** | 0.0268* | 0.0000*** |
| pct_period | 0.0041** | 0.0186* | 0.0000*** |
| pct_question | 0.0349* | 0.2799 | 0.0000*** |
| pct_exclamation | 0.0000*** | 0.0002*** | 0.0002*** |
| pct_other_term | 0.1514 | 0.0008*** | 0.2358 |
| n_distinct_header_types | 0.0000*** | 0.7334 | 0.0041** |