# PROJECT_FILE_INDEX.md

> GKX (2020) three-characteristic public-data replication — project file map and development history  
> Updated: 2026-11-19  
> Repository: [Shaneycui/gkx-2020-asset-pricing-replication](https://github.com/Shaneycui/gkx-2020-asset-pricing-replication)

## How to read this index / 阅读说明

This is a **research and learning history**, not an assertion that every named file is currently committed to GitHub. The phase order follows the research workflow documented in chat and `RESEARCH_PROGRESS.md`; the order *within a phase* is approximate unless a dated run or commit exists.

**Evidence / status codes**

- **G — GitHub verified:** exact script has been fetched from the repository's `main` branch.
- **R — Recorded:** file or work was explicitly mentioned in past chat, PyCharm screenshots, or project progress notes; source content/current repository presence has **not** been independently checked here.
- **V — Run verified:** a result, log, or validation was supplied in the conversation, but this does not imply the script is currently uploaded.
- **D — Draft / pending:** provided as a new draft or planned analysis; no finished end-to-end result confirmed.
- **? — Tentative name:** name/use appeared in prior conversational inventory, but existence and precise behavior need verification against the local folder. Do **not** cite these as verified project artifacts.

**Important:** `data/` is Git-ignored. GitHub code/docs do **not** back up the processed data or the long-running model outputs. File paths below are relative to the project's local root, except where indicated.

## One-page workflow / 项目主线

```text
data/datashare.csv (4,117,300 original stock-month rows)
    |
    v
preprocess_features_stream.py
    -> data/gkx_features_3char_full.csv
    |
    v
impute_features.py + validate_imputed_features.py
    -> data/gkx_features_3char_imputed.csv
    |
    +--- validate_mom1m_returns.py (independent 7-stock exploratory checks)
    |
    v
build_gkx_full_panel.py
    -> data/gkx_panel_mom1m_public.csv (4,078,842 eligible rows)
    |
    +---------------------------+
    |                           |
    v                           v
train_elastic_net_recursive.py   train_random_forest_recursive.py
    |                           |
    v                           v
EN yearly + predictions         RF yearly + validation + predictions
    |                           |
    +-------------+-------------+
                  v
          compare_models.py
                  |
                  v
     test_model_significance.py  [planned/awaiting run]
                  
Separate solver-comparison branch:
train_elastic_net_apg_recursive.py [draft; check loss scaling and convergence]
```

The main target is a **return proxy reconstructed from the following calendar month's raw `mom1m`**, *not* the small Yahoo-ticker exploratory panel and *not* original CRSP excess returns.

## Phase 1 — Initial data exploration & Python practice (approximately Nov 9–11)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `inspect_data.py` | Inspect raw Xiu feature columns, identifiers, dates, counts. | `datashare.csv` → console diagnostics | R, ? precise name |
| `inspect_returns.py` | Inspect the public independent monthly-return CSV, equity subset, coverage. | `monthly_returns_by_ticker.csv` → diagnostics | R |
| `validate_returns.py` | Basic returns sanity checks; precise implementation requires source inspection. | monthly returns → checks | R, ? |
| `match_aapl.py` | Practice PERMNO filtering, calendar-month alignment, one-to-one merges for AAPL. | raw features + AAPL returns → paired observations | R |
| `inspect_mom1m_outlier.py` | Check outlying raw `mom1m` and monthly 1%/99% winsorization behavior. | `datashare.csv` → outlier diagnostics | R/V |
| `practice_groupby.py` | Learn monthly `groupby`, `rank`, `transform`, rank-to-[-1,1] normalization. | small practice data → console output | R |
| `practice_month_buffer.py` | Prototype preserving an incomplete final month across CSV chunks. | chunked sample → processed whole months | R/V |
| `practice_chunk_boundary.py` | Investigate CSV chunk/month boundary conditions. | practice inputs → diagnostics | R, source not inspected |
| `practice_incremental_write.py` | Practice append/incremental CSV writing. | practice inputs → example CSV | R, source not inspected |

These scripts explain the *learning path*. They are not the validated final data pipeline.

## Phase 2 — Three-characteristic processing (approximately Nov 11–15)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `preprocess_features.py` | Earlier prototype of feature processing. | raw characteristics → experimental output | R, ? precise role |
| `preprocess_features_stream.py` | Monthly cross-sectional rank of `mom1m`, `bm`, `mvel1`, transformed as `2*rank/(n+1)-1`; handles months across CSV chunks. | `data/datashare.csv` → `data/gkx_features_3char_full.csv` | R/V — 4,117,300 rows |
| `impute_features.py` | Fill missing scaled features with within-month medians; for months with all BM missing, project-specific zero fallback. | `data/gkx_features_3char_full.csv` → `data/gkx_features_3char_imputed.csv` | R/V |
| `validate_imputed_features.py` | Reopen final file; check counts, missing values, duplicate keys, and preservation of originally observed feature values. | imputed + original processed CSVs → validation logs | R/V — checks passed |

**Verified observations from the research record:** source and full features each have 4,117,300 rows; 780 months; imputed output has zero remaining NA values in the three selected features. The BM zero fallback applies to 72,077 observations in 66 months (1957-01 to 1962-06) and is **not** a claimed GKX-mandated convention.

## Phase 3 — Initial ticker mapping & returns experiments (approximately Nov 11–15)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `prepare_aapl_label.py` | Construct the AAPL monthly feature/independent-return alignment prototype. | Xiu AAPL features + Yahoo return data → paired prototype | R/V |
| `prepare_msft_label.py` | Repeat preliminary return alignment for MSFT. | Xiu MSFT features + public returns → prototype | R |
| `inspect_permno.py` | Examine identifier behavior; confirm exact code/name locally. | raw feature identifiers → diagnostics | ?, unverified |
| `prepare_ticker_mapping.py` | Record candidate ticker/PERMNO links. | external candidate list → exploratory mapping | R |
| `check_ticker_mapping.py` | Review candidate mapping validity; **not** a complete historical crosswalk. | mapping candidates → diagnostics | R, source not inspected |
| `build_stock_panel.py` | Construct small exploratory multi-ticker panel from provisional mappings. **Not** the main full-market panel. | 3-feature data + Yahoo returns + candidate mappings → exploratory CSV | R, G source originally documented |
| `validate_mom1m_returns.py` | Compare next-calendar-month raw `mom1m` against separately acquired returns for seven candidate ticker/PERMNO pairs. | raw `mom1m` + independent return observations → alignment statistics | R/V |

**Exploratory validation, not certification:** 3,335 consecutive paired observations showed correlation ~0.99581 and MAE ~0.00066 under the next-calendar-month alignment. Historical ticker/PERMNO mapping intervals remain unverified, and raw `mom1m` may contain upstream winsorization.

## Phase 4 — Final simplified public-data panel (approximately Nov 15–16)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `build_gkx_full_panel.py` | Preserve/audit feature-to-return matches and select stock-months with an available next-calendar-month return proxy; **no ticker mapping required** for the main panel. | imputed features + raw `mom1m` → `data/gkx_panel_mom1m_public.csv` and `data/gkx_panel_mom1m_audit_by_year.csv` | R/V |
| `split_gkx_panel.py` | Verify chronological Training/Validation/Test partition boundaries and counts. | eligible panel → split diagnostics | R/V |

Eligible panel: **4,078,842** rows, **32,642** distinct PERMNOs, six columns: `permno`, `DATE`, `mom1m_scaled`, `bm_scaled`, `mvel1_scaled`, `return_proxy`. About **38,458** feature records lack the next-calendar-month return proxy and are excluded after audit, not imputed.

Initial fixed split checked: Training 1957–1986, Validation 1987–1999, Testing 2000–2021. This fixed split was a *learning/pilot configuration*, not the eventual GKX-style yearly recursive scheme.

## Phase 5 — Elastic Net training (approximately Nov 16–17)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `train_elastic_net.py` | Initial fixed-split benchmark, learn validation-MSE selection and zero-benchmark OOS R². | eligible panel → prototype predictions/metrics | R |
| [`train_elastic_net_recursive.py`](train_elastic_net_recursive.py) | Full 1987–2016 yearly recursive EN; expanding training, 12-year rolling validation, 10 candidate `alpha` values, yearly refit, predictions and metrics. | `data/gkx_panel_mom1m_public.csv` → `data/elastic_net_recursive_yearly.csv`, `data/elastic_net_recursive_predictions.csv` | **G/V** |

**Exact saved experiment:** `sklearn.linear_model.ElasticNet` (coordinate-descent solver), `l1_ratio=0.5`, `np.logspace(-4,-1,10)`; **2,489,185** predictions and pooled OOS R² **+0.24952075885404934%** against zero predicted returns. The original completed yearly CSV did **not** capture coefficients; coefficient-saving lines were subsequently added to the script, and only take effect on a future run.

## Phase 6 — Random Forest training (approximately Nov 16–17)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| [`train_random_forest_pilot.py`](train_random_forest_pilot.py) | Quick 1987 pilot with 100,000 sampled training rows / 50,000 sampled validation rows; depth candidates [2,4], 50 trees. | eligible panel → `data/rf_pilot_*.csv` | **G/V** |
| [`train_random_forest_full_1987.py`](train_random_forest_full_1987.py) | 1987 full-history computation/selection test; depth 1–6, 50 trees. | eligible panel → `data/rf_full_1987_*.csv` | **G/V** |
| [`train_random_forest_recursive.py`](train_random_forest_recursive.py) | Full 1987–2016 recursive RF, saved annually with separate validation/prediction files and resumability. | eligible panel → `data/rf_recursive_50trees_leaf100/rf_recursive_yearly.csv`, `rf_validation_YYYY.csv`, `rf_predictions_YYYY.csv` | **G/V** |

**Exact recursive configuration:** 50 trees, candidate depths [1,2,3,4,5,6], `max_features=3`, `min_samples_leaf=100`, `random_state=42`. Pooled OOS R² **+0.3859393441365433%** from **2,489,185** forecasts. GKX's 300-tree specification was **not** reproduced. Selected depth 6 each year from 2000 through 2011 is the upper search boundary, not a proven unconstrained optimum.

## Phase 7 — Forecast comparison & inferential statistics (approximately Nov 17–19)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `compare_models.py` | Outer one-to-one join on (`permno`, `DATE`); audit keys and actual proxy values; compute annual and pooled EN/RF MSE, R², and 360 monthly mean loss differences. | EN prediction CSV + 30 RF prediction CSVs → `data/model_comparison/comparison_overall.csv`, `comparison_yearly.csv`, `comparison_monthly_losses.csv` | **V** (user ran; GitHub presence not independently verified) |
| `test_model_significance.py` | Test zero mean of month-aggregated squared-error differential with Newey–West/Bartlett HAC SE; pilot lag settings 0, 6, 12; save confidence intervals/p-values. | `comparison_monthly_losses.csv` and `comparison_overall.csv` → `comparison_hac_results.csv` | **D** (script provided; completed run not confirmed) |

**Comparison audit already reported:** 2,489,185 matched records, EN-only 0, RF-only 0, maximum realized target difference 0. Overall RF–EN improvement **+0.1364186 R² percentage points** (rounded). This is a **descriptive** advantage until robust inference is performed.

## Phase 8 — Optimization-method fidelity / APG branch (Nov 19)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `train_elastic_net_apg_recursive.py` | Experimental APG implementation reflecting the optimization algorithm described in GKX Internet Appendix B.1; intended 1987 single-year validation then possible recursion. | eligible panel → separate APG output directory (intended) | **D** (provided as attachment; no confirmed full empirical run or GitHub upload) |

APG requires explicit documentation of `lambda` grid, squared-loss normalization, step size, stopping tolerance, iteration limit and intercept convention. The published APG algorithm does **not** by itself verify that all our numerical choices match the original authors' empirical settings. Preserve the completed scikit-learn results as an independent benchmark.

## Suggested study order / 以后重学的顺序

1. **Data:** `preprocess_features_stream.py` → `impute_features.py` → `validate_imputed_features.py`.
2. **Target:** `validate_mom1m_returns.py` → `build_gkx_full_panel.py` → `split_gkx_panel.py`.
3. **Elastic Net:** `train_elastic_net.py` → `train_elastic_net_recursive.py` (line-by-line; training / validation / refit / testing / regularization).
4. **RF:** `train_random_forest_pilot.py` → `train_random_forest_full_1987.py` → `train_random_forest_recursive.py`.
5. **Evaluation:** `compare_models.py` → `test_model_significance.py`.
6. **Solver methods:** `train_elastic_net_apg_recursive.py`, compare APG with coordinate descent using an identical objective before claiming numerical replication.

## Main local outputs to preserve separately

| Local file/folder | Content |
|---|---|
| `data/datashare.csv` | Raw downloaded characteristics |
| `data/gkx_features_3char_full.csv` | Ranked / scaled features before final imputation |
| `data/gkx_features_3char_imputed.csv` | Complete three-feature panel |
| `data/gkx_panel_mom1m_public.csv` | Eligible features + reconstructed return proxy |
| `data/gkx_panel_mom1m_audit_by_year.csv` | Panel matching / exclusion audit |
| `data/elastic_net_recursive_yearly.csv` | Original completed EN yearly metrics |
| `data/elastic_net_recursive_predictions.csv` | Original completed EN stock-month forecasts |
| `data/rf_recursive_50trees_leaf100/` | Original completed RF yearly validation, metrics, and forecast files |
| `data/model_comparison/` | Paired accuracy comparison and eventually HAC test outputs |

## Research restrictions and next steps

1. **High priority:** Independent audit of `return_proxy` timing / mechanical correlations with `mom1m_scaled`, information availability, winsorization, and missing/delisting selection. Current work is **not** a faithful return-target reproduction of the original CRSP study.
2. **Pending:** Run and interpret month-level HAC / modified DM comparison, then check sensitivity to reasonable lag choices.
3. **Pending:** Review yearly RF MDI importance (not causal attribution); original completed EN coefficient file is unavailable without re-running or a separate recovery method.
4. **Pending:** Validate APG against exactly the same convex objective and data, retaining the prior solver for cross-checking; do not overwrite earlier results.
5. **Documentation:** Report source versions/hashes, implementation-specific choices, and limits of a three-characteristic project.

## Related documentation

- [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md) — chronological research findings and experiment logs.
- [DATA_SOURCES.md](DATA_SOURCES.md) — source provenance, preprocessing assumptions and data caveats.
- Original paper: Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- This index should be revised whenever a new script is added or an unverified historical filename is confirmed/corrected.
