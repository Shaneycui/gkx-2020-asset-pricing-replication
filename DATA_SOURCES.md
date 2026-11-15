# Data Sources and Provenance

Last updated: 2026-11-15

This document distinguishes **actual file provenance**, **methodological references**, **observed local data properties**, and **unverified assumptions**. It does not claim a full reproduction of Gu, Kelly, and Xiu (2020).

## 1. Stock characteristics (X)

- **Actual download source** (confirmed by the researcher): Dacheng Xiu's University of Chicago Booth website, `https://dachxiu.chicagobooth.edu/download/datashare.zip`.
- Downloaded archive contains `datashare.csv`; local input: `data/datashare.csv`.
- Official directory listing: `https://dachxiu.chicagobooth.edu/download/`. The listing contains `datashare.zip` and `datashare_OLD.zip`. **The listing's file timestamp is not proof of dataset version or underlying observation vintage.**
- **Local observations reported during preprocessing**: 4,117,300 records, 780 distinct months, raw date coverage 1957-01-31 through 2021-12-31. These are local pipeline observations; a reproducible raw-file hash, download date, and archive version have not yet been recorded.
- Identifiers: `permno`, `DATE` (YYYYMMDD). Selected characteristics: `mom1m`, `bm`, `mvel1`.
- These are characteristic observations **not** an independently obtained CRSP monthly individual-stock returns panel.
- Broader historical financial data used to construct characteristics may involve CRSP, Compustat, and other sources, but individual field lineage and coverage require source documentation; they have **not** all been independently verified here.

### Feature preprocessing used in this project

- Implementation: `preprocess_features_stream.py`; resulting file: `data/gkx_features_3char_full.csv`.
- For each calendar month and each characteristic, rank **cross-sectionally** among non-missing stocks using average ranks for ties; map to `[-1, 1]` using `2 * rank / (n + 1) - 1`, where `n` is the non-missing stock count for that characteristic and month.
- The intermediate ranked file preserves missing characteristics as NaN. Separate script `impute_features.py` performs **monthly cross-sectional median imputation** on scaled feature columns and writes `data/gkx_features_3char_imputed.csv`, without overwriting the ranked intermediate file. Any residual NaNs when an entire month has no valid characteristic (BM only: 66 months 1957-01 to 1962-06, 72,077 stock-month rows) receive a project-specific **zero fallback**, which must not be characterized as a rule explicitly specified by GKX.
- Previously checked: input/output row counts agree at 4,117,300; 780 months; duplicate (`DATE`, `permno`) count 0; saved monthly counts reconciled. These tests do not imply every feature value has been independently verified.

## 2. Monthly stock returns (y)

- **Actual download source** (confirmed by the researcher): HerdingAI / stock-price-returns, `https://github.com/HerdingAI/stock-price-returns`.
- Local input: `data/monthly_returns_by_ticker.csv`.
- Generating script: `https://github.com/HerdingAI/stock-price-returns/blob/main/pull_returns.py`.
- Upstream data source: Yahoo Finance via `yfinance`. The script requests `interval="1mo"`, `period="max"`, `auto_adjust=True`, `actions=False`, and reads the returned `Close` column. With automatic adjustment enabled, this is intended to be split/dividend-adjusted close for stocks, not raw unadjusted close.
- Monthly returns are calculated in the upstream script with `prices.pct_change()`, i.e. `P_adj(t) / P_adj(t-1) - 1`.
- Note that `actions=False` suppresses separate corporate-action columns, **not** the adjustments requested by `auto_adjust=True`.
- Upstream documentation: `https://github.com/HerdingAI/stock-price-returns/blob/main/docs/nuances_and_caveats.md`. Yahoo historical adjustments, data quality, missing delisted issues, and asset-specific exceptions need consideration.
- **Local observations**: 118,024 rows across 336 tickers in the entire return file; `asset_class == "Equity (single stock)"` filters to 54,592 rows across 128 distinct tickers. Other asset classes (ETFs, indexes, funds, yields) must not be mixed with single-stock target returns.
- Coverage and selection are not equivalent to a full CRSP equity universe. Ticker reuse/renames, incomplete delisting returns and survivor/selection bias remain limitations.
- Do **not** describe the Yahoo file as official GKX/CRSP returns.

## 3. GKX paper, code, and related references

- Paper: Gu, Shihao; Kelly, Bryan T.; Xiu, Dacheng (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies, `https://doi.org/10.1093/rfs/hhaa009`.
- Xiu-linked resources supplied by researcher:
  - `https://github.com/xiubooth/ML_Codes`: simulation/Monte Carlo code; **not** treated as a full empirical model/data pipeline.
  - Supplemental material: `https://dachxiu.chicagobooth.edu/download/ML_supp.pdf`.
  - Separate third-party characteristic-construction reference: `https://feng-cityuhk.github.io/EquityCharacteristics/`. It is **not** official GKX empirical-replication code and indicates WRDS subscriptions for its underlying data workflow.
- Exact characteristic definitions, raw preprocessing/winsorization status and the paper's precise missing-value imputation conventions must be cross-checked against primary documentation; do not assert them solely from summaries.

## 4. Verified AAPL prototype and labeling

- Preliminary mapping used: AAPL ticker ↔ PERMNO 14593; historical identifier continuity still needs confirmation before scaling to all stocks.
- AAPL feature rows: 492 (1981-01 through 2021-12).
- AAPL available Yahoo monthly return rows: 498 (1985-02 through 2026-07).
- Pipeline sorts returns by month, defines `future_return = monthly_return.shift(-1)`, then joins features with this target on calendar month, `validate="one_to_one"`.
- Result: 443 aligned feature/target observations, zero missing in `mom1m_scaled`, `bm_scaled`, `mvel1_scaled`, `future_return`.
- AAPL raw monthly return summary (498): mean 0.023786, standard deviation 0.119880, min -0.577436 (2000-09), max 0.453783. These are descriptive sanity checks, not complete independent historical price verification.
- Label alignment and price-adjustment conventions are separate checks. The present prototype does not establish valid data for every stock.

## 5. Outstanding work / research restrictions

1. Establish reliable **date-sensitive ticker ↔ PERMNO** mapping; do not guess identifiers.
2. Independently test stock-level label alignment and coverage when expanding beyond AAPL.
3. Establish an explicit missing-predictor policy, point-in-time availability rules, and time-based train/validation/test split.
4. Address survivorship/selection bias, Yahoo data limitations and absence of complete CRSP delisting-return coverage in reporting.
5. Evaluate Elastic Net and Random Forest with the same sample and appropriate out-of-sample baselines.
6. Record download date, version information where available, raw data file hashes, and project code commit hashes for reproducibility.

**Scope statement:** This project is a simplified replication of the GKX empirical machine-learning *method*, with 3 selected characteristics and third-party Yahoo-based returns. It is not the original article's full empirical dataset or results replication.

### Locally verified imputation diagnostics (2026-11-15)

- Input file: `data/gkx_features_3char_full.csv`; output file: `data/gkx_features_3char_imputed.csv`.
- Missing before: mom1m 31,738; bm 1,074,711; mvel1 3,070. Following per-`DATE` median imputation: 0; 72,077; 0 respectively. The residual 72,077 BM records correspond *exactly* to 66 entire months from 1957-01 to 1962-06 with no nonmissing BM observations; zero fallback applied as explicitly documented project-specific choice.
- Source values originally non-null were checked unchanged for all three columns. Reopened output: 4,117,300 rows; 0 missing across the 3 scaled characteristics; 0 duplicate (`DATE`, `permno`) keys.
- These diagnostics do not validate the underlying historical data's release timing or the Yahoo-to-PERMNO mapping. Avoid calling this an exact recreation of the paper's complete preprocessing.
