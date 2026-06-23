# 17n R+ drop sensitivity (Model A vs Model B)

## Comparison table (β = β(COI × cumulative_cue_attempts))

| Scope | N | β_A (R+ kept) | SE_A | p_A | β_B (R+ dropped) | SE_B | p_B | Δβ | % change |
|---|---|---|---|---|---|---|---|---|---|
| FULL | N=5 | +0.0238** | 0.0077 | 0.0020 | +0.0241** | 0.0079 | 0.0022 | +0.0003 | +1.3% |
| DROP-MANC | N=5 | +0.0359*** | 0.0105 | 0.0007 | +0.0363*** | 0.0105 | 0.0006 | +0.0004 | +1.1% |
| FULL | N=10 | +0.0339*** | 0.0095 | 0.0004 | +0.0325*** | 0.0087 | 0.0002 | -0.0013 | -4.0% |
| DROP-MANC | N=10 | +0.0485*** | 0.0132 | 0.0002 | +0.0446*** | 0.0116 | 0.0001 | -0.0039 | -8.1% |

## Acceptance criteria

- ✓ Direction agrees (both >0) across all cells: **True**
- ✓ |Δβ / β_A| < 0.30 across all cells: **True**
- ✓ Sig p<0.05 across all cells (both models): **True**

**Verdict**: PASS — Exposure-gate β is independent of R+ scaffolding (Tier-1 absorption ruled out).