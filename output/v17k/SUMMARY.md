# 17k Manchester qualitative deep dive — sample 10 files/child

## Per-corpus medians (raw .cha features)

| Metric | Brown | Manchester | UK_long_long | UK_short_obs |
|---|---|---|---|---|
| n_utt | 1135.000 | 1558.500 | 1569.500 | 296.000 |
| n_speakers | 5.000 | 3.000 | 3.000 | 5.000 |
| n_participants_declared | 5.000 | 3.000 | 3.000 | 5.000 |
| has_investigator | 1.000 | 1.000 | 0.000 | 0.000 |
| n_caregiver_codes | 1.000 | 1.000 | 1.000 | 3.000 |
| duration_minutes | 60.000 | 30.000 | 30.000 | nan |
| pct_xxx | 0.051 | 0.038 | 0.059 | 0.152 |
| pct_yyy | 0.001 | 0.000 | 0.000 | 0.000 |
| pct_www | 0.000 | 0.005 | 0.001 | 0.000 |
| pct_amp_event | 0.006 | 0.012 | 0.019 | 0.021 |
| pct_timecode | 0.000 | 0.000 | 0.980 | 0.000 |
| pct_plus_info | 0.000 | 0.000 | 0.014 | 0.000 |
| pct_CHI | 0.505 | 0.372 | 0.415 | 0.455 |
| pct_MOT | 0.367 | 0.590 | 0.536 | 0.325 |
| pct_FAT | 0.000 | 0.000 | 0.000 | 0.023 |
| pct_INV | 0.000 | 0.026 | 0.000 | 0.000 |
| pct_mor | 0.984 | 0.970 | 0.926 | 0.860 |
| pct_gra | 0.984 | 0.970 | 0.926 | 0.860 |
| pct_com | 0.024 | 0.008 | 0.006 | 0.094 |
| pct_sit | 0.003 | 0.000 | 0.000 | 0.000 |
| pct_act | 0.055 | 0.000 | 0.000 | 0.000 |
| pct_exp | 0.008 | 0.001 | 0.000 | 0.000 |
| pct_spa | 0.000 | 0.000 | 0.000 | 0.000 |
| pct_xpho | 0.000 | 0.000 | 0.000 | 0.000 |
| pct_err | 0.002 | 0.005 | 0.000 | 0.000 |
| pct_add | 0.004 | 0.005 | 0.000 | 0.003 |
| n_distinct_dep_tiers | 11.000 | 6.000 | 3.000 | 5.000 |

## Mann-Whitney (Manchester vs others)

| Metric | vs Brown | vs UK_long_long | vs UK_short_obs |
|---|---|---|---|
| n_utt | 0.0000*** | 0.8195 | 0.0000*** |
| n_speakers | 0.0000*** | 0.0000*** | 0.0000*** |
| n_participants_declared | 0.0000*** | 0.0000*** | 0.0000*** |
| has_investigator | 0.6199 | 0.0000*** | 0.0000*** |
| n_caregiver_codes | 0.0000*** | 0.3635 | 0.0000*** |
| duration_minutes | 0.0000*** | 0.7624 | n/a |
| pct_xxx | 0.0135* | 0.0045** | 0.0000*** |
| pct_yyy | 0.0000*** | 0.0553† | 0.0000*** |
| pct_www | 0.0000*** | 0.0010*** | 0.0000*** |
| pct_amp_event | 0.0000*** | 0.0010*** | 0.0000*** |
| pct_timecode | 1.0000 | 0.0000*** | 1.0000 |
| pct_plus_info | 0.4661 | 0.0000*** | 1.0000 |
| pct_CHI | 0.0000*** | 0.0595† | 0.0000*** |
| pct_MOT | 0.0000*** | 0.0000*** | 0.0000*** |
| pct_FAT | 0.0000*** | 0.2489 | 0.0000*** |
| pct_INV | 0.0000*** | 0.0000*** | 0.0000*** |
| pct_mor | 0.1776 | 0.0000*** | 0.0000*** |
| pct_gra | 0.1776 | 0.0000*** | 0.0000*** |
| pct_com | 0.0001*** | 0.2826 | 0.0000*** |
| pct_sit | 0.0000*** | 0.0183* | 0.0098** |
| pct_act | 0.0000*** | 0.0800† | 0.0000*** |
| pct_exp | 0.0000*** | 0.0000*** | 0.0000*** |
| pct_spa | 0.0000*** | 1.0000 | 0.1620 |
| pct_xpho | 0.0000*** | 1.0000 | 0.0142* |
| pct_err | 0.0102* | 0.0000*** | 0.0000*** |
| pct_add | 0.1013 | 0.0000*** | 0.0021** |
| n_distinct_dep_tiers | 0.0000*** | 0.0000*** | 0.0000*** |

## Top @Activities per corpus


### Brown  (unique values = 0)


### Manchester  (unique values = 16)

- 4: Duplo Zoo , basket of food and doll
- 2: Duplo zoo , basket of food , doll , car with panda , and hoops .
- 2: Duplo zoo , basket of food , panda and car , hoops , doll
- 2: Duplo zoo , basket of food , doll , panda and car , hoops
- 2: basket of food , doll and Duplo zoo
- 1: Duplo Zoo , doll and basket of food
- 1: doll , Duplo zoo and basket of food
- 1: putting a train+set together

### UK_long_long  (unique values = 3)

- 1: playing with toy kitchen
- 1: Duplo , doll , car and panda , basket of food , hoops , and a helicopter from the wug toys
- 1: Duplo zoo , basket of food , doll , panda and car , hoops

### UK_short_obs  (unique values = 62)

- 9: Free play alone
- 8: Watching TV
- 7: Free play with another child
- 6: Free play
- 6: Free play with adult participation
- 5: Talking
- 5: Other nonplay
- 4: Talk as the main activity