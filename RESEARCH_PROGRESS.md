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
- Appropriate treatment of missing characteristics; do not claim all rows are complete.
- Point-in-time predictor/target availability and lookahead-prevention audit.
- Train/validation/test chronological partitions, out-of-sample baselines and metrics.
- Elastic Net, Random Forest, robustness tests, documented reproducibility report.
- CRSP-quality delisting returns / survival and selection biases are unresolved with current free Yahoo dataset.

## Immediate next action

Inspect whether reliable publicly accessible identifier mappings (historical ticker → PERMNO with validity dates) can be established. If not, preserve current AAPL study as a prototype while selecting a defensible limited-universe approach. Do not fabricate mappings.

## Status reporting

Previous conversational percentage estimates (e.g., ~55%) were informal, not a measured completion rate. Prioritize confirmed deliverables and pending blockers over a fabricated precise overall percentage.
