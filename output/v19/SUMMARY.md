# 19 — 4-component COI sensitivity test

## β(COI × cumulative) comparison: 5-comp vs 4-comp

| Window | Scope | β_5comp | SE_5 | p_5 | β_4comp | SE_4 | p_4 | Δβ | % change | Egger p (5/4) |
|---|---|---|---|---|---|---|---|---|---|---|
| N=5 | FULL | +0.0241** | 0.0079 | 0.0022 | +0.0220** | 0.0067 | 0.0010 | -0.0021 | -8.7% | 0.587 / 0.572 |
| N=5 | DROP-MANC | +0.0363*** | 0.0105 | 0.0006 | +0.0311*** | 0.0091 | 0.0006 | -0.0052 | -14.2% | 0.353 / 0.336 |
| N=10 | FULL | +0.0325*** | 0.0087 | 0.0002 | +0.0383*** | 0.0066 | 0.0000 | +0.0057 | +17.7% | 0.850 / 0.851 |
| N=10 | DROP-MANC | +0.0446*** | 0.0116 | 0.0001 | +0.0441*** | 0.0092 | 0.0000 | -0.0005 | -1.1% | 0.380 / 0.346 |

## Acceptance criteria

- ✓ Direction agrees (both β > 0) all cells: **True**
- ✓ |Δβ / β_5comp| < 0.30 all cells: **True**
- ✓ Sig p < 0.05 all cells (both COI variants): **True**

**Verdict:** PASS — exposure-gate β is independent of S_frequency_normalized. Frequency is not driving the effect.