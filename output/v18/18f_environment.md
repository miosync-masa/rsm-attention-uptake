# 18f — Software environment + script→output map (SI S10)

## Table S10.1 — Software versions used to produce the Paper 2 evidence stack

| Package | Version | Role in the pipeline |
|---|---|---|
| **Python**       | **3.12.4** | Runtime |
| pandas       | 3.0.3   | All tabular processing |
| numpy        | 2.4.6   | Vector / matrix ops |
| scipy        | 1.17.1  | Mann-Whitney U, Spearman ρ, `norm.cdf` for meta p-values |
| **statsmodels**  | **0.14.6** | OLS, WLS, Logit, MixedLM, cluster-robust SE (`cov_type="cluster"`) |
| matplotlib   | 3.11.0  | All figures |
| tqdm         | 4.68.2  | Progress bars |

### Meta-analysis implementation

The DerSimonian–Laird random-effects meta-analysis is **in-house**, defined as
`random_effects_meta(betas, ses)` in each of
`17c_child_level_slopes.py` (§ Q2),
`17h_uk_subcorpora_meta.py`,
`17l_paper_figures.py`,
`17m_window_range_sensitivity.py`,
`17o_window_null_mechanism.py`, and `18f` analyses.
The implementation is reproduced verbatim across these files (identical
code, no functional drift). It computes:

- weights wᵢ = 1 / SEᵢ²
- fixed-effect β = Σwᵢβᵢ / Σwᵢ
- Q = Σwᵢ(βᵢ − βₚ)², df = n − 1
- τ² = max(0, (Q − df) / C) where C = Σwᵢ − Σwᵢ² / Σwᵢ
- I² = max(0, (Q − df) / Q) · 100
- random-effects weights wᵢ* = 1 / (SEᵢ² + τ²)
- pooled β_RE = Σwᵢ*βᵢ / Σwᵢ*, SE_RE = √(1 / Σwᵢ*)
- z = β_RE / SE_RE, p = 2 · (1 − Φ(|z|)) via `scipy.stats.norm.cdf`

No `metafor`, `meta`, `PythonMeta` or other published package is used.

### Egger small-study / publication-bias test

The Egger regression in `17l_paper_figures.py` (function `egger_test`) and
`17m_window_range_sensitivity.py` is **in-house**:

- regress (β/SE) on (1/SE) by OLS via `statsmodels.OLS`
- intercept (and its p-value) is the small-study-effect test statistic.

### Logit GLM (#17o leg A)

`17o_window_null_mechanism.py` uses `statsmodels.Logit.fit(disp=False, maxiter=100)`
per child. Children with constant outcome (Y is all 0 or all 1) are skipped.
This is the only per-child estimator that is not OLS / LPM.

---

## Table S10.2 — Script → output map

Mapping from each analysis script to the SI table / main figure it generates.
All scripts live in the repository root; their committed outputs live under
`output/v17*/` (per-step) and `output/v18/` (this SI fill).

| Script | Stage | Primary outputs | SI section / Figure |
|---|---|---|---|
| `01_load_corpus_json.py` | Loader — CHA → JSON → tokens.csv | `output/{lang}_{tokens,utterances,summary}.{csv,json}` | (Paper 1 SI; pre-Paper-2) |
| `02_extract_cues_v2.py` | Per-language cue tagging | `output/{lang}_tokens_tagged.csv` | (Paper 1) |
| `03_compute_attention_index_v3.py` | 5-component COI / attention_index | `output/{lang}_attention_index.csv` | **S1** (COI weights), and per-cue COI used throughout Paper 2 |
| `05_developmental_uptake.py` | Uptake measures (Paper 1 outcome) | `output/v3/{lang}_uptake.csv` | (Paper 1) |
| `10_extract_R_plus_v2.py` | Episode-level R+ extraction | `output/v10b/{lang}_r_plus_{episodes,cue_agg,summary}.{csv,json}` | **S9** (R+ definition) |
| `11_rsm_r_plus_join.py` | M0 / M1 / M2 hierarchical regression | `output/v11*/{lang}_r_plus_{joined,regression}.{csv,json}` | Paper 1 ↔ Paper 2 bridge |
| `15_within_between_decomposition.py` | Mundlak within / between (Spec A + B') | `output/v15/{lang}_within_between_decomposition.json` | (referenced in S9) |
| `16_episode_outcome_uptake.py` | next-N child cue reuse outcome | `output/v16/{lang}_episodes_with_reuse.csv` | Source for **S3 reuse base rates**, drives #17c / 17m / 17n / 17o |
| `17b_exposure_gate_test.py` | Main exposure-gate model (COI × cumulative) | `output/v17b/` | **Table 1**, S6 |
| `17c_child_level_slopes.py` | Per-child OLS + meta + random slopes | `output/v17c/per_child_betas_N{5,10,3,20}.csv` | **Figure 1** (forest plot), **S3 Table S3.2** |
| `17d_window_width_check.py` | obs_window as moderator (WLS) | `output/v17d/per_child_with_window_N5.csv` | S4, S5 |
| `17e_adequate_window_subset.py` | β stratified by window threshold | `output/v17e/adequate_window_*.csv` | S5 |
| `17f_uk_reverse_simulation.py` | UK restricted to Manchester 24-36mo | `output/v17f/uk_reverse_*.csv` | S5 (window-as-gate falsification) |
| `17g_manchester_diagnostics.py` | UK vs Manchester MLU / density / cue-mix | `output/v17g/manchester_vs_uk_descriptives.csv` | **S8**, **Figure 4** (radar) |
| `17h_uk_subcorpora_meta.py` | 4-way sub-corpus meta | `output/v17h*/uk_subcorpora_*.{csv,json}` | **S6 Table S6** (4-way meta) |
| `17i_corpus_protocol_diff.py` | JSON-cache protocol features | `output/v17i/protocol_diff_per_corpus.csv` | S8 |
| `17j_na_pool_per_child.py` | NA-Pool Brown extension (April) | `output/v17j/na_pool_per_child_betas_loose.csv` | S3 (n=32 row for April) |
| `17k_manchester_qualitative.py` | Raw `.cha` protocol features (Theakston signature) | `output/v17k/protocol_features_per_*.csv` | **S8.2 + Figure 4** (Theakston radar inputs) |
| `17l_paper_figures.py` | Forest + funnel + radar + drop-Manc + Egger | `output/v17l/{forest_plot,funnel_plot,theakston_radar}.png` | **Figures 1–4**, **Table 2**, **S7 Egger** |
| `17m_window_range_sensitivity.py` | 4-window grid (N=3,5,10,20) + trajectory | `output/v17m/four_window_meta.json` | **S4 Table S4** (4-window meta) |
| `17n_rplus_drop_test.py` | Model A (R+ champion) vs Model B | `output/v17n/comparison_table_*.csv` | **S9 Table 3a / 3b** |
| `17o_window_null_mechanism.py` | LPM/Logit/Continuous × 6 windows + variance + truncation | `output/v17o/{meta_six_windows.json,variance_compression_table.csv,truncation_rate_table.csv,trajectory_three_modes.png}` | **S4 Table S4 (extended)** + **S5 Tables S5.1, S5.2** |

### Repository / branch

All Paper 2 evidence is committed on branch
**`paper2-exposure-gate`** of [REPO URL REDACTED FOR ANONYMOUS REVIEW].
The `main` branch contains only the Paper 1 (RSM) pipeline.

This SI fill (#18a–#18f) is produced under `output/v18/` and committed to
the same branch.
