# PROJECT_FILE_INDEX.md

> GKX (2020) three-characteristic public-data replication — project file map and development history  
> Updated: 2026-11-21  
> Repository: [Shaneycui/gkx-2020-asset-pricing-replication](https://github.com/Shaneycui/gkx-2020-asset-pricing-replication)

## How to read this index / 阅读说明

This is a **research and learning history**, not an assertion that every named file is currently committed to GitHub. The phase order follows the research workflow documented in chat and `RESEARCH_PROGRESS.md`; the order *within a phase* is approximate unless a dated run or commit exists.

**Evidence / status codes**

- **G — GitHub verified:** exact file is present on the repository's `main` branch (checked against the current GitHub tree).
- **R — Recorded:** file or work was explicitly mentioned in past chat, PyCharm screenshots, or project progress notes; source content/current repository presence has **not** been independently checked here.
- **V — Run verified:** a result, log, or validation was supplied in the conversation, but this does not imply the script is currently uploaded.
- **D — Draft / pending:** provided as a new draft or planned analysis; no finished end-to-end result confirmed.


**Important:** `data/` is Git-ignored. GitHub code/docs do **not** back up the processed data or the long-running model outputs. File paths below are relative to the project's local root, except where indicated.

## Current status (21 November 2026) / 当前状态

**Completed local research:** three-feature preprocessing, audited proxy panel, 1987–2016 expanding-window CD Elastic Net, APG Elastic Net, 50-tree Random Forest, matched forecasts, APG-vs-CD numerical consistency, APG-vs-RF modified-DM-style inference, and annual RF MDI analysis. **A locally completed experiment is not the same as a GitHub-committed script.**

**GitHub now contains:** original feature preprocessing, final proxy-panel constructor, CD and APG Elastic Net trainers, RF trainer, APG/CD solver comparison, APG/RF significance scripts, RF MDI analysis, and the research/reporting documents. The GitHub tree was checked before this update.

**Still local-only / not verified on `main`:** a few auxiliary validation tools and historical CD/RF comparison scripts. The main workflow scripts are now present, but a clean-clone end-to-end reproduction has **not** been independently executed. The `data/` directory is ignored; no large raw or forecast CSVs are stored in GitHub.

### Current experiment in one diagram

```text
data/datashare.csv  (4,117,300 rows; NOT committed)
    |
    v
preprocess_features_stream.py             [G/V]
    -> data/gkx_features_3char_full.csv
    |
    v
impute_features.py                        [G/V]
    -> data/gkx_features_3char_imputed.csv
    |
    v
build_gkx_full_panel.py                   [G/V]
    -> data/gkx_panel_mom1m_public.csv (4,078,842 eligible rows)
    |
    +-------------------------+---------------------------+
    |                         |                           |
    v                         v                           v
train_elastic_net_       train_elastic_net_        train_random_forest_
recursive.py [G/V]      apg_recursive.py [G/V]     recursive.py [G/V]
    |                         |                           |
    v                         v                           v
CD predictions           APG predictions              RF predictions
    |                         |                           |
    +--- compare_elastic_     +--- compare_apg_rf_ --------+
         net_solvers.py [G/V]       significance.py [G/V]     |
                                     |                    |
                                     v                    v
                          test_modified_dm_       analyze_rf_feature_
                          apg_rf.py [G/V]           importance.py [G/V]
                                     |                    |
                                     v                    v
                           Modified DM HAC        Annual MDI and plots
```

**Main results:** 2,489,185 matched OOS stock-month predictions (1987–2016). Pooled OOS R²: **CD +0.249520759%**, **APG +0.249536393%**, **RF +0.385939344%**. Primary APG-minus-RF monthly-loss test: **z = +2.0860, two-sided p = 0.036975**, Newey–West HAC lag 12, *unadjusted*. Mean RF MDI: **mom1m 67.44%, mvel1 22.53%, bm 10.03%**.

**Main limitation:** the outcome is the following-calendar-month raw `mom1m` **return proxy**, not GKX's CRSP excess-return target; the three predictors and RF hyperparameter grid are simplified.

## Phase 1 — Initial data exploration & Python practice (approximately Nov 9–11)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `inspect_data.py` | Inspect raw Xiu feature columns, identifiers, dates, counts. | `datashare.csv` → console diagnostics | R |
| `inspect_returns.py` | Inspect the public independent monthly-return CSV, equity subset, coverage. | `monthly_returns_by_ticker.csv` → diagnostics | R |
| `validate_returns.py` | Basic returns sanity checks; precise implementation requires source inspection. | monthly returns → checks | R |
| `match_aapl.py` | Practice PERMNO filtering, calendar-month alignment, one-to-one merges for AAPL. | raw features + AAPL returns → paired observations | R |
| `inspect_mom1m_outlier.py` | Check outlying raw `mom1m` and monthly 1%/99% winsorization behavior. | `datashare.csv` → outlier diagnostics | R/V |


These scripts explain the *learning path*. They are not the validated final data pipeline.

## Phase 2 — Three-characteristic processing (approximately Nov 11–15)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `preprocess_features.py` | Earlier prototype of feature processing. | raw characteristics → experimental output | R, ? precise role |
| [`preprocess_features_stream.py`](preprocess_features_stream.py) | Cross-sectional monthly ranks, chunk buffering, incremental CSV writing, duplicate/count audits. | `data/datashare.csv` → `data/gkx_features_3char_full.csv` | **G/V** — 4,117,300 rows |
| [`impute_features.py`](impute_features.py) | Within-month median imputation, all-missing-month zero fallback; verifies existing nonmissing data unchanged. | `data/gkx_features_3char_full.csv` → `data/gkx_features_3char_imputed.csv` | **G/V** |
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
| [`build_gkx_full_panel.py`](build_gkx_full_panel.py) | Preserve/audit feature-to-return matches and select stock-months with an available next-calendar-month return proxy; **no ticker mapping required** for the main panel. | imputed features + raw `mom1m` → `data/gkx_panel_mom1m_public.csv` and `data/gkx_panel_mom1m_audit_by_year.csv` | **G/V** |
| `split_gkx_panel.py` | Verify chronological Training/Validation/Test partition boundaries and counts. | eligible panel → split diagnostics | R/V |

Eligible panel: **4,078,842** rows, **32,642** distinct PERMNOs, six columns: `permno`, `DATE`, `mom1m_scaled`, `bm_scaled`, `mvel1_scaled`, `return_proxy`. About **38,458** feature records lack the next-calendar-month return proxy and are excluded after audit, not imputed.

Initial fixed split checked: Training 1957–1986, Validation 1987–1999, Testing 2000–2021. This fixed split was a *learning/pilot configuration*, not the eventual GKX-style yearly recursive scheme.

## Phase 5 — Elastic Net training (approximately Nov 16–17)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| `train_elastic_net.py` | Initial fixed-split benchmark, learn validation-MSE selection and zero-benchmark OOS R². | eligible panel → prototype predictions/metrics | R |
| [`train_elastic_net_recursive.py`](train_elastic_net_recursive.py) | Full 1987–2016 yearly recursive EN; expanding training, 12-year rolling validation, 10 candidate `alpha` values, yearly refit, predictions and metrics. | `data/gkx_panel_mom1m_public.csv` → `data/elastic_net_recursive_yearly.csv`, `data/elastic_net_recursive_predictions.csv` | **G/V** |

**Exact saved experiment:** `sklearn.linear_model.ElasticNet` (coordinate-descent solver), `l1_ratio=0.5`, `np.logspace(-4,-1,10)`; **2,489,185** predictions and pooled OOS R² **+0.24952075885404934%** against zero predicted returns. The historical completed CD yearly file was reported without saved coefficient columns at the time of the original solver comparison; the **current GitHub training source** has coefficient-saving logic. Whether a particular local CSV already contains these columns must be verified from that file. No paired coefficient/objective-value audit has been completed.

## Phase 6 — Random Forest training (approximately Nov 16–17)

| Python file | Purpose / 作用 | Main input → output | Status |
|---|---|---|---|
| [`train_random_forest_pilot.py`](train_random_forest_pilot.py) | Quick 1987 pilot with 100,000 sampled training rows / 50,000 sampled validation rows; depth candidates [2,4], 50 trees. | eligible panel → `data/rf_pilot_*.csv` | **G/V** |
| [`train_random_forest_full_1987.py`](train_random_forest_full_1987.py) | 1987 full-history computation/selection test; depth 1–6, 50 trees. | eligible panel → `data/rf_full_1987_*.csv` | **G/V** |
| [`train_random_forest_recursive.py`](train_random_forest_recursive.py) | Full 1987–2016 recursive RF, saved annually with separate validation/prediction files and resumability. | eligible panel → `data/rf_recursive_50trees_leaf100/rf_recursive_yearly.csv`, `rf_validation_YYYY.csv`, `rf_predictions_YYYY.csv` | **G/V** |

**Exact recursive configuration:** 50 trees, candidate depths [1,2,3,4,5,6], `max_features=3`, `min_samples_leaf=100`, `random_state=42`. Pooled OOS R² **+0.3859393441365433%** from **2,489,185** forecasts. GKX's 300-tree specification was **not** reproduced. Selected depth 6 each year from 2000 through 2011 is the upper search boundary, not a proven unconstrained optimum.

## Phase 7 — Forecast comparison & inference (Nov 17–20)

| File | Role / outputs | Status |
|---|---|---|
| `compare_models.py` | Audits matched CD/RF stock-month keys and labels; saves overall, annual and monthly comparison to `data/model_comparison/`. | **V** — local run, not on `main` |
| `test_model_significance.py` | An earlier significance-script filename; see next row for the **confirmed completed** implementation. | **R** — several historical drafts; do not assume every version produced final output |
| `test_model_significance-v2.py` | Completed Newey–West/Bartlett HAC inference for **CD versus RF**, 0/6/12 monthly lags. | **V** — output recorded, not on `main` |
| [`compare_apg_rf_significance.py`](compare_apg_rf_significance.py) | Loads all 30 APG and RF annual forecasts; validates `permno`/`DATE` and labels; saves pooled, annual and monthly losses in `data/apg_rf_significance/`. | **G/V** — source committed; completed local run |
| [`test_modified_dm_apg_rf.py`](test_modified_dm_apg_rf.py) | Completed GKX-style modified DM on 360 monthly cross-sectional MSFE differences, Bartlett/Newey–West 0/6/12 lags. | **V** — completed local run, not on `main` |

**Main APG-versus-RF test (1987–2016):** 360 consecutive months; 2,489,185 matched forecasts; 195 RF-winning months, 165 APG-winning months. Mean equal-month `MSFE_APG - MSFE_RF` = **+0.0000282605** (stock-month-weighted counterpart: **+0.0000324277**).

| HAC lags | DM z | Two-sided p | Result |
|---:|---:|---:|---|
| 0 | +3.0592 | 0.002219 | RF lower mean loss |
| 6 | +2.3348 | 0.019553 | RF lower mean loss |
| **12 (primary)** | **+2.0860** | **0.036975** | RF lower mean loss at **unadjusted 5%** |

95% CI, primary 12-lag monthly mean difference: **[+0.0000017076, +0.0000548133]**. This is **not** the original paper's entire multiple-model/Bonferroni-testing procedure. In particular, 2.49 million stock-months must not be presented as 2.49 million independent time-series observations.

**CD-versus-RF robustness:** 12-lag **z = +2.0864, p = 0.036938**, versus APG **z = +2.0860, p = 0.036975**. Do not mislabel the CD p-value as APG.

## Phase 8 — APG optimization and solver fidelity (Nov 19–20)

| File | Role / outputs | Status |
|---|---|---|
| [`train_elastic_net_apg_recursive.py`](train_elastic_net_apg_recursive.py) | Completed all 30 APG annual train/validation/refit/prediction runs, separate `data/elastic_net_apg_recursive/` outputs. | **G/V** — source committed; completed local run |
| [`compare_elastic_net_solvers.py`](compare_elastic_net_solvers.py) | One-to-one compare of all APG/CD forecasts and selected numerical penalties, save `data/elastic_net_solver_comparison/`. | **V** — completed local run, **not committed** |

APG settings: mixing `rho = 0.5`, **10 log-spaced** penalty choices 1e-4 through 1e-1, `max_iter=20000`, `tol=1e-10`, appendix-inspired accelerated proximal updates and intercept handling. The **grid and numerical stopping controls are project choices**, not exact parameter-for-parameter replication claims.

APG pooled OOS R²: **+0.249536392766%**; CD: **+0.249520758854%**. The models chose the same numerical penalty in **30/30** test years. Mean absolute APG/CD prediction difference: **1.26318949431e-6**; maximum **9.12524972048e-5**. All 30 APG annual results passed the recorded residual check. These establish close forecast agreement, **not** a full paired-coefficient or exact-objective proof.

## Phase 9 — RF feature importance (Nov 20–21)

| File | Role / outputs | Status |
|---|---|---|
| [`analyze_rf_feature_importance.py`](analyze_rf_feature_importance.py) | Reads previously saved RF annual `mdi_mom1m`, `mdi_bm`, `mdi_mvel1`; writes `rf_mdi_summary.csv`, `rf_mdi_yearly.csv`, `rf_mdi_mean.png`, `rf_mdi_by_year.png`. No refitting. | **G/V** — source committed; successful local run |
| `data/rf_recursive_50trees_leaf100/rf_recursive_yearly.csv` | Original RF annual validation/depth/performance/MDI metadata. | **V**, local data only |
| `data/rf_feature_importance_analysis/` | Result CSVs and figures. | **V**, local data only |

Average annual MDI: **mom1m 0.674356 (67.44%; #1 in 30/30 years)**, **mvel1 0.225308 (22.53%)**, **bm 0.100336 (10.03%)**. In 1987, 1988, 1989, 2014, and 2015 `mom1m` supplied all recorded MDI; selected depths were respectively **1, 2, 2, 1, 1**. In 1995 the selected depth was 3, with approximately **66.21% mom1m / 21.37% mvel1 / 12.42% bm**. 2000–2011 selected depth 6 (the **upper bound** of the grid).

MDI is **training-time normalized split-impurity importance**, not the percentage of future returns explained, OOS ablation importance, or causal evidence about macroeconomic states. The RF was not retrained for this summary.

## Phase 10 — Research report and GitHub publishing (Nov 20–21)

| GitHub file | Purpose | Status |
|---|---|---|
| [`README.md`](README.md) | Concise project homepage and final results. | **G** |
| [`REPLICATION_WALKTHROUGH.md`](REPLICATION_WALKTHROUGH.md) | Modeled on the explain–code–result style. | **G** |
| [`REPLICATION_REPORT.md`](REPLICATION_REPORT.md) | Detailed final three-characteristic research report: APG/CD/RF, OOS R², modified DM, MDI, caveats. | **G** |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment setup, output conventions, code availability audit and checkpoints. | **G** |
| [`requirements.txt`](requirements.txt) | Dependency compatibility ranges, **not** a fixed original environment lock. | **G** |
| [`preprocess_features_stream.py`](preprocess_features_stream.py) and [`impute_features.py`](impute_features.py) | User-supplied actual locally used sources uploaded on Nov 21, replacing tutorial-only pseudocode references. | **G/V** |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) / [`RESEARCH_PROGRESS.md`](RESEARCH_PROGRESS.md) | Earlier provenance / historical experiment log; **some older status headings are superseded by the current report and this index**. | **G** |

**Reproducibility status:** the six additional original local scripts were committed on 21 November 2026: the main proxy-panel constructor, APG trainer, solver comparison, APG/RF forecast comparison, modified-DM test, and RF MDI analysis. Auxiliary validation scripts and historical CD/RF comparison are still absent. Main workflow code is present, but the full pipeline still needs a clean-environment rerun with documented data hashes and package versions.

## Suggested reading and study order / 以后重学的顺序

1. **Overview:** [README](README.md) → [walkthrough](REPLICATION_WALKTHROUGH.md) → [full report](REPLICATION_REPORT.md).
2. **Preprocessing:** [`preprocess_features_stream.py`](preprocess_features_stream.py) → [`impute_features.py`](impute_features.py) → local `validate_imputed_features.py`.
3. **Return proxy and sample splits:** local `validate_mom1m_returns.py` → `build_gkx_full_panel.py` → `split_gkx_panel.py`.
4. **Models:** [`train_elastic_net_recursive.py`](train_elastic_net_recursive.py), local `train_elastic_net_apg_recursive.py`, then [`train_random_forest_recursive.py`](train_random_forest_recursive.py).
5. **Model comparisons:** local `compare_elastic_net_solvers.py`, `compare_models.py`, `compare_apg_rf_significance.py`, `test_modified_dm_apg_rf.py`.
6. **Interpretation:** local `analyze_rf_feature_importance.py`; read RF MDI and selected depth alongside the prediction metrics.

Historical practice files in Phases 1–4 remain listed for their learning value; they are not substitutes for the full main-panel experiment.

## Main local outputs to preserve separately

| Local file/folder | Content |
|---|---|
| `data/datashare.csv` | Source Xiu characteristics (not committed) |
| `data/gkx_features_3char_full.csv` | Three-feature ranked panel |
| `data/gkx_features_3char_imputed.csv` | Imputed feature panel |
| `data/gkx_panel_mom1m_public.csv` | Eligible feature+return-proxy panel |
| `data/gkx_panel_mom1m_audit_by_year.csv` | Panel inclusion/exclusion audit |
| `data/elastic_net_recursive_yearly.csv` / `data/elastic_net_recursive_predictions.csv` | CD annual metrics and forecasts |
| `data/elastic_net_apg_recursive/` | APG annual tuning, predictions and convergence outputs |
| `data/elastic_net_solver_comparison/` | APG/CD matched forecast consistency |
| `data/rf_recursive_50trees_leaf100/` | RF yearly validation, depth, MDI and prediction files |
| `data/model_comparison/` | CD/RF comparison and HAC outputs |
| `data/apg_rf_significance/` | APG/RF matched, annual and monthly losses |
| `data/apg_rf_significance/modified_dm/` | Formal monthly modified-DM report and result CSV |
| `data/rf_feature_importance_analysis/` | RF annual MDI summary, CSVs and charts |

**Do not commit large, licensed or uncontrolled raw data.** All outputs in `data/` are ignored by `.gitignore`. Back up numerical results outside GitHub and record the original source hash/versions before a reproducibility claim.

### Research restrictions and remaining tasks

**Completed:** prediction experiments, matched CD/APG/RF comparison, 0/6/12-lag modified DM, descriptive RF MDI, and readable research documentation.

**Remaining:** upload remaining auxiliary validation scripts, perform a clean-clone rerun, independently audit `mom1m` timing and CRSP equivalence, compare APG and CD objectives/coefficients if needed, preserve environment versions and data hashes, and consider better-target/higher-dimensional follow-up studies. The original paper's macro interactions and portfolio economic evaluation are outside this three-feature run.

## Related documentation

- [REPLICATION_WALKTHROUGH.md](REPLICATION_WALKTHROUGH.md) — readable code-interleaved tutorial.
- [REPLICATION_REPORT.md](REPLICATION_REPORT.md) — main results and inference.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — exact repository limitations and run prerequisites.
- [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md) — chronological research findings and experiment logs; early notes are historical.
- [DATA_SOURCES.md](DATA_SOURCES.md) — source provenance, preprocessing assumptions and data caveats.
- Original paper: Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- This index should be revised whenever a new script is added or an unverified historical filename is confirmed/corrected.
