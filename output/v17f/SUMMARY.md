# 17f UK reverse simulation — outcome window N=5

## Corpus-level β(COI × cumulative_cue_attempts)

| Scenario | n_children | n_episodes | β | SE | p |
|---|---|---|---|---|---|
| UK_full_post_MSR | 41 | 303,980 | +0.0243 | 0.0096 | 0.0113 |
| UK_24-36_lifetime_cum (Variant A) | 28 | 243,120 | +0.0211 | 0.0100 | 0.0356 |
| UK_24-36_local_cum (Variant B) | 28 | 243,120 | +0.0211 | 0.0099 | 0.0340 |
| Manchester_full_post_MSR | 11 | 77,533 | -0.0065 | 0.0136 | 0.6326 |

## Per-child random-effects meta (β_i pooled)

| Scenario | n | pooled β | SE | p | τ² | I² |
|---|---|---|---|---|---|---|
| UK_full_post_MSR | 27 | +0.0308 | 0.0093 | 0.0010 | 0.0013 | 63.2% |
| UK_24-36_lifetime_cum (Variant A) | 25 | +0.0230 | 0.0089 | 0.0094 | 0.0009 | 54.0% |
| UK_24-36_local_cum (Variant B) | 25 | +0.0255 | 0.0089 | 0.0042 | 0.0009 | 54.0% |
| Manchester_full_post_MSR | 11 | +0.0014 | 0.0081 | 0.8594 | 0.0000 | 0.0% |

## Verdict

- UK full → UK 24-36 (Variant A, lifetime cumulative): β = +0.0243 → +0.0211  (FAIL)

- UK full → UK 24-36 (Variant B, window-local cumulative): β = +0.0243 → +0.0211  (FAIL)

- Manchester reference β = -0.0065, p = 0.6326

- **Overall**: FAIL — UK retains its effect even at Manchester-equivalent window.
