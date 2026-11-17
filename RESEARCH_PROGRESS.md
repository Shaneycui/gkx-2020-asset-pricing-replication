# Research Progress — Simplified GKX (2020) Replication

Last updated: 2026-11-17

**Research scope:** Reproduce selected empirical machine-learning asset-pricing methodology of Gu, Kelly, Xiu (2020) using 3 characteristics; this is not an exact replication of the paper's full dataset or all published results.

## Confirmed data inputs

Refer to [DATA_SOURCES.md](DATA_SOURCES.md) for provenance and unresolved caveats.

- Stock characteristics: Dacheng Xiu's `datashare.zip` (`datashare.csv`).
- Monthly returns: HerdingAI/stock-price-returns (`monthly_returns_by_ticker.csv`), originally Yahoo Finance adjusted monthly stock prices.
- Do not confuse characteristic inputs with the separate returns target, or simulation code with a full empirical reproduction.

## Completed and locally validated

1. **Three-characteristic processing prototype:** `mom1m`, `bm`, `mvel1` month-specific cross-sectional ranks mapped to [-1,1].
2. **Full feature processing:** `preprocess_features_stream.py` output `data/gkx_features_3char_full.csv`: 4,117,300 rows, 780 months (1957-01 to 2021-12); output row/month counts matched input; no duplicate (DATE, permno) keys in output. Some raw/processed characteristic values remain missing.
3. **Return file inspection:** entire source 118,024 rows / 336 tickers; filter `Equity (single stock)` yields 54,592 rows and 128 tickers. **This is not the full CRSP universe.**
4. **AAPL prototype:** `prepare_aapl_label.py` converted feature and returns dates to calendar-month Periods, sorted the monthly return series, computed `future_return = monthly_return.shift(-1)`, and merged one-to-one on month; 443 observations, 0 nulls across 3 scaled predictors and `future_return`. Earlier feature history 492 observations (1981-01 to 2021-12); returns history 498 observations (1985-02 to 2026-07).
5. **AAPL preliminary return check:** 498 observations, no null return, mean 0.023786, std 0.119880, minimum -0.577436 (2000-09), maximum 0.453783. Extreme observations are not presumed erroneous solely due to magnitude.
6. **Source traceability:** `DATA_SOURCES.md` records Xiu's official data URL and HerdingAI's monthly-return computation (`auto_adjust=True`, `pct_change()`).

## Remaining / not yet verified

- Historical, date-specific ticker ↔ PERMNO mapping for stocks beyond AAPL; no mass merging based on unverified mappings.
- Separate independent checks of actual returns against adjusted price history for sampled stock-months.
- Three-feature imputation completed and locally verified; **historical full-month BM zero fallback is a documented project-specific rule**, not a paper-mandated default. Revisit if using pre-1962-07 samples.
- Point-in-time predictor/target availability and lookahead-prevention audit.
- Train/validation/test chronological partitions, out-of-sample baselines and metrics.
- Elastic Net, Random Forest, robustness tests, documented reproducibility report.
- CRSP-quality delisting returns / survival and selection biases are unresolved with current free Yahoo dataset.

## Immediate next action

Inspect whether reliable publicly accessible identifier mappings (historical ticker → PERMNO with validity dates) can be established. If not, preserve current AAPL study as a prototype while selecting a defensible limited-universe approach. Do not fabricate mappings.

## Status reporting

Previous conversational percentage estimates (e.g., ~55%) were informal, not a measured completion rate. Prioritize confirmed deliverables and pending blockers over a fabricated precise overall percentage.

7. **Full market feature imputation, independently reloaded:** `impute_features.py` writes `data/gkx_features_3char_imputed.csv`. Initial missing counts: `mom1m_scaled` 31,738, `bm_scaled` 1,074,711, `mvel1_scaled` 3,070. Monthly cross-sectional median filling leaves only 72,077 `bm_scaled` nulls, all in 66 wholly missing-BM months (1957-01 through 1962-06); a documented **project-specific fallback** fills these with zero. Original non-null values unchanged for all 3 features (`True`). Saved/reloaded file checked by `validate_imputed_features.py`: 4,117,300 rows, 0 remaining missing across the 3 columns, 0 duplicate (`DATE`, `permno`) keys; `ALL VALIDATIONS PASSED`, process exit 0. These checks establish integrity, not original dataset's point-in-time correctness.
8. **MSFT provisional label alignment:** candidate PERMNO 10107; 429 feature months (1986-04 through 2021-12); 484 Yahoo return months (1986-04 through 2026-07); prior merged data report 414 completely observed three-feature/target rows before imputation, with `mom1m_scaled` 1 missing, `bm_scaled` 15 missing, `mvel1_scaled` and `future_return` 0 missing. Historical mapping remains *candidate*, not verified.

## 2026-11-15: Batch PERMNO candidate evidence and exploratory panel code

- Added `research/permno_candidates.csv`: 7 candidate pairs drawn from the public Financial Risk Forecasting course handout (AAPL 14593, MSFT 10107, JPM 47896, C 70519, XOM 11850, MCD 43449, GE 12060). URL: https://www.financialriskforecasting.com/seminars/1.pdf . Handout labels these as the **latest** tickers; it does **not** supply complete CRSP historical ticker validity intervals. All entries remain **candidate / unverified**, not production-training mappings.
- Added `build_stock_panel.py`: optional exploratory multi-stock merge of existing Xiu 3-characteristic imputed feature data with HerdingAI single-stock returns, using the candidate mappings. Uses calendar-month return labels with a consecutive-month guard, audits duplicate keys, and restricts feature months to 2000-01 through 2021-11 as a provisional research window. Writes `data/gkx_panel_exploratory_candidates_2000_2021.csv` with `mapping_status=candidate_UNVERIFIED`. **No output row counts validated yet**; the researcher must run the script locally. This is not an authorized validated training sample.
- Current blocking requirement: verify PERMNO ↔ ticker historical match intervals and identity transitions, especially JPM/C/GE merger and reorganization histories, before any main experiment. Do not confuse evidence of a cross-sectional CRSP match at one date with proof of continuous identity.

## Mandatory panel construction audit (user requirement — 2026-11-15)

**DO NOT skip this audit when moving from exploratory candidate data to the official modeling panel.**

1. Start from the stock-feature panel as the LEFT table. Join the monthly-return labels using a **LEFT MERGE**, not an inner merge, retaining unmatched feature-month observations for audit.
2. Preserve the original stock-feature row count and identifiers through the audit; distinguish rows lacking any ticker/return match from rows whose following calendar-month return is missing or invalid. Do not silently discard either category.
3. Report matched and unmatched counts and rates overall, **by ticker/PERMNO and by month (or year)**; inspect systematic gaps and selection/survivorship risk.
4. Verify candidate ticker↔PERMNO identity and date validity before promoting observations to the official panel. Candidate evidence alone is insufficient.
5. **Only after recording the audit**, explicitly select eligible observations with valid next-calendar-month targets; never impute unknown target returns with zero. Keep an auditable exclusion summary.
6. Train and evaluate Elastic Net, Random Forest, and baseline on the **same eligible evaluation observations**.

The current `build_stock_panel.py` uses an **inner merge for the exploratory-only panel**; it does NOT satisfy this mandatory final-data audit. Change/extend it before using the output for the main study.


---

## 2026-11-17 — Confirmed milestone: both recursive models completed

**Status note:** This is the current verified experiment status and supersedes older "remaining work" and "immediate next action" sections above, which describe the earlier Yahoo/ticker-mapping exploration. The actual full-universe modeling target now uses the public **next-calendar-month mom1m return proxy**, *not* the limited Yahoo ticker panel. Earlier experiments remain useful as an independent cross-check and are not part of the main 4M-row modeling panel.

### Modeling dataset and audit

- Original Xiu characteristics source: `data/datashare.csv`, **4,117,300** stock-month records, 1957-01–2021-12.
- Cross-sectional feature ranks normalized to [-1, 1] for `mom1m`, `bm`, and `mvel1`; month-median imputation, with a documented project-specific BM zero fallback for entirely missing months.
- Full eligible modeling panel: `data/gkx_panel_mom1m_public.csv`, **4,078,842** stock-month records, 32,642 PERMNO identifiers, 1957–2021. Columns: `permno`, `DATE`, `mom1m_scaled`, `bm_scaled`, `mvel1_scaled`, `return_proxy`.
- Return proxy: for a given stock-month, a shifted next-calendar-month raw `mom1m` value. Independent Yahoo checks for seven *candidate* ticker/PERMNO pairs showed correlation about **0.99581**, MAE about **0.00066** across 3,335 paired consecutive-month observations. These candidate mappings are not certified historical CRSP crosswalks.
- **Research limitations:** This proxy is not the original CRSP excess-return target, may reflect upstream winsorization and missing/delisting exclusions, and does not remove potential survivorship, time-availability or mechanical-alignment concerns. Independent point-in-time/no-leakage audit remains outstanding. The original GKX full set of characteristics and macro interactions are not reproduced.

### Evaluation design

- Annual out-of-sample test years: **1987–2016** (30 years).
- Expanding training history starting in 1957, rolling preceding 12-year validation window, yearly validation-based hyperparameter selection and re-fitting on all pre-test-year observations.
- Both models produced **2,489,185** out-of-sample stock-month forecasts on the same eligible test period (prediction-key equality remains to be independently confirmed in comparison script).
- OOS R-squared benchmark is **zero predicted return**, calculated as `1 - sum((actual - prediction)^2) / sum(actual^2)`.

### Elastic Net — completed

- Implementation in local `train_elastic_net_recursive.py`; **not yet uploaded to this GitHub repository** as of this update.
- `l1_ratio=0.5`; ten log-spaced `alpha` candidates over [1e-4, 1e-1] (the ten-point grid is project-specific).
- Overall 1987–2016 OOS R²: **+0.24952075885404934%**.
- Local outputs: `data/elastic_net_recursive_yearly.csv` and `data/elastic_net_recursive_predictions.csv`.
- Annual intercept and three feature coefficients were added to the *local source code for future runs*, but the completed 30-year historical output **does not yet contain those coefficients**. Do not claim they were retrospectively saved.

### Random Forest — completed

- Local scripts: `train_random_forest_pilot.py`, `train_random_forest_full_1987.py`, and `train_random_forest_recursive.py`; **not yet uploaded to this GitHub repository** as of this update.
- Recursive configuration: 50 trees per forest, max depth candidates 1–6, `max_features=3`, `min_samples_leaf=100`, `random_state=42`, `n_jobs=2`. Full historical training/validation observations (no subsampling in the recursive run).
- Overall 1987–2016 OOS R²: **+0.3859393441365433%**.
- 1987 OOS R² about **-0.7993%**; 1988 about **+1.2729%**; best depths 1 and 2 respectively. The whole 1987–1988 pilot recursive window combined R² about **+0.0320%**.
- Selected best depth was 6 in every year **2000–2011**, but 6 is also the largest candidate value; avoid claiming this is the unconstrained optimal depth.
- Local outputs under `data/rf_recursive_50trees_leaf100/`: `rf_recursive_yearly.csv`, per-year `rf_validation_YYYY.csv`, per-year `rf_predictions_YYYY.csv`. Saved metrics include annual validation/test MSE, OOS R², depth, hyperparameters, MDI feature importance and runtime.
- 2004 wall-clock log reads **29,544.6 seconds**, unusually high relative to surrounding years; cause unknown, do not presume market-year computational difficulty.

### Preliminary model comparison (descriptive only)

| Model | 1987–2016 overall OOS R² | Test predictions |
|---|---:|---:|
| Elastic Net | +0.2495% | 2,489,185 |
| Random Forest | +0.3859% | 2,489,185 |

- Random Forest's overall R² exceeds Elastic Net's by approximately **0.1364 percentage points**, under current distinct model configurations.
- **No paired, dependence-robust significance test completed**; observed advantage is descriptive rather than a proven statistically significant model improvement.
- Not a full replication of GKX Table A.5: RF uses 50 (not 300) trees and project-specific `min_samples_leaf=100`.

### Next tasks (in priority order)

1. Upload **exact current local Python scripts** through the researcher's own Git tooling (or attach the files here for GitHub upload); this connector cannot read the Windows PyCharm directory.
2. Develop `compare_models.py` to join predictions one-to-one on (`permno`, `DATE`), check that realized proxy labels agree, compare OOS R² and MSE on a common sample, and report annual performance.
3. Compare forecast losses with appropriate time/dependence-aware inference (for example, monthly aggregated losses and HAC inference); don't treat all stock-months as independent observations.
4. Audit target construction, point-in-time information, proxy/winsorization and missing/delisting-selection concerns; document precise limits of replicability.
5. Generate plots, feature-importance interpretation, reproducibility report and final research write-up.

### Storage and reproducibility

- Large `data/` files, model prediction CSVs, and `.venv/` should remain outside Git. **GitHub holds code and text research records, not the locally computed outputs.** Back up the result CSV files separately because the RF recursive run is computationally expensive.
