# Research Progress — Simplified GKX (2020) Replication

Last updated: 2026-11-15

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
