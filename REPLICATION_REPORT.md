# Replication Report: Empirical Asset Pricing via Machine Learning

**A three-characteristic implementation based on public data**  
**Project:** `Shaneycui/gkx-2020-asset-pricing-replication`  
**Report status:** Completed three-characteristic methodological replication; full GKX replication not claimed  
**Version date:** 20 November 2026

---

## Abstract

This project implements a simplified empirical asset-pricing prediction exercise inspired by Gu, Kelly, and Xiu (2020, hereafter GKX), *Empirical Asset Pricing via Machine Learning*. The purpose is to build a transparent, reproducible workflow for monthly stock-level return prediction, compare a regularized linear model (Elastic Net) with a nonlinear ensemble model (Random Forest), and develop the research and programming skills required for more extensive machine-learning asset-pricing work.

The implementation uses three cross-sectionally ranked firm characteristics — `mom1m`, `bm`, and `mvel1` — from the public characteristic file hosted by Dacheng Xiu. The resulting panel contains 4,078,842 eligible stock-month observations and 32,642 PERMNO identifiers. Because a matching CRSP excess-return panel was not available in the current project environment, the dependent variable is a **proxy** constructed from the following calendar month's raw `mom1m` value. This proxy is not equivalent to the CRSP total or excess return used in GKX.

Using an expanding training sample, a rolling 12-year validation window, and annual model refitting, the 1987–2016 experiments produce 2,489,185 matched out-of-sample forecasts per model. Pooled zero-benchmark OOS R² is **0.249536%** for Elastic Net estimated with accelerated proximal gradient (APG), **0.249521%** for a coordinate-descent (CD) robustness benchmark, and **0.385939%** for Random Forest (RF). In a prespecified two-sided comparison of equal-weight monthly squared forecast losses, a GKX-style modified Diebold–Mariano (DM) statistic with 12-lag Newey–West HAC standard errors is **2.0860 (p = 0.036975)**, favoring RF at the unadjusted 5% level. Training-time RF mean decrease in impurity (MDI) assigns average importance of **67.44%** to `mom1m`, **22.53%** to `mvel1`, and **10.03%** to `bm` across 30 annual models.

The results should therefore be interpreted as a **three-characteristic methodological replication and pipeline prototype**, not as a reproduction of GKX's original data, full predictor set, optimization procedures, or headline empirical findings.

## 1. Research objective and reference design

### 1.1 Original study

GKX investigate whether machine-learning methods can improve the prediction of individual stock returns using a large set of firm characteristics and macroeconomic state variables. Their empirical design compares several methods under a common out-of-sample forecasting framework and studies both predictive accuracy and the economic value of the resulting forecasts.

The reference implementation for this project is Stefan Voigt's 2024 Tidy Finance article, [“Replicating Gu, Kelly & Xiu (2020)”](https://blog.tidy-finance.org/posts/gu-kelly-xiu-replication/). The article is useful because it translates key parts of the paper and its online appendix into a practical data and model-fitting workflow. It is an independent implementation, however, and does not remove the need to distinguish the original paper's specifications from the blog author's implementation choices.

### 1.2 Scope of this project

The current project focuses on a deliberately restricted first experiment:

- Three stock characteristics: `mom1m`, `bm`, and `mvel1`.
- Two prediction methods: Elastic Net (APG primary; CD as numerical robustness benchmark) and Random Forest.
- GKX-style modified DM inference on monthly paired prediction losses.
- Descriptive annual RF MDI importance and selected-depth diagnostics.
- Annual recursive/expanding estimation from 1987 through 2016.
- A common stock-month test sample for both models.
- Forecast accuracy relative to a zero-return benchmark.

The current experiment does **not** include the full 94 firm characteristics, 74 industry indicators, eight macroeconomic predictors, or their interaction terms described in the original study and the Tidy Finance implementation. It also does not yet construct decile portfolios or evaluate long-short portfolio returns.

## 2. Data

### 2.1 Firm characteristics

The characteristic data were downloaded from Dacheng Xiu's public data page:

- Source archive: [datashare.zip](https://dachxiu.chicagobooth.edu/download/datashare.zip)
- Local raw file: `data/datashare.csv`
- Original observations reported by the local preprocessing audit: **4,117,300 stock-month rows**
- Date coverage: January 1957 to December 2021
- Identifiers: `permno` and `DATE`

The current model uses the following columns:

| Variable | Interpretation in this project | Role |
|---|---|---|
| `mom1m` | One-month momentum characteristic as supplied in the characteristic file | Predictor |
| `bm` | Book-to-market characteristic | Predictor |
| `mvel1` | Market-equity / size-related characteristic as supplied in the file | Predictor |
| `permno` | Stock identifier | Panel key |
| `DATE` | Characteristic observation date | Time key |

The project uses only these three characteristics. It does not claim that they represent the complete GKX predictor set.

### 2.2 Cross-sectional ranking and scaling

Following the broad preprocessing principle described by GKX and the Tidy Finance article, each characteristic is ranked cross-sectionally within each month and mapped to a bounded interval.

The project implementation uses average ranks for ties and the transformation:

[
x^{scaled}_{i,t} = 2\frac{rank_{i,t}}{n_t+1}-1,
]

where (n_t) is the number of non-missing observations for that characteristic in month (t).

The resulting predictors are:

- `mom1m_scaled`
- `bm_scaled`
- `mvel1_scaled`

**Implementation difference to record:** the Tidy Finance article's displayed R function maps the minimum and maximum non-missing ranks exactly to -1 and +1 using a min/max-rank transformation. Our project uses (2rank/(n+1)-1). Both produce bounded rank-based transformations, but they are not numerically identical, particularly at the endpoints. The difference should be resolved or explicitly retained as a documented implementation choice before claiming preprocessing equivalence.

### 2.3 Missing predictor values

The project applies monthly cross-sectional median imputation to the three scaled characteristics. If an entire month has no observed value for a characteristic, the median is undefined. For the 66 months in which all `bm` observations are missing (January 1957–June 1962), the remaining BM values are filled with zero.

This is consistent with the practical two-step interpretation discussed in the Tidy Finance article: use the monthly cross-sectional median where it exists, then use zero for residual all-missing cases. The article notes that the paper's written description and a coauthor's Q&A statement differ on missing-value treatment; it interprets zero as a fallback when a monthly median cannot be calculated.

Project validation recorded:

- 4,117,300 rows retained.
- 780 distinct months.
- No duplicate (`DATE`, `permno`) keys.
- No remaining missing values in the three model features after imputation.
- Originally observed non-missing values were preserved.

The zero fallback is documented as a project preprocessing choice for all-missing months, not as proof that every detail of the original GKX preprocessing has been reproduced.

### 2.4 Dependent variable: next-month `mom1m` proxy

The original GKX study uses monthly CRSP individual-stock returns (and excess returns in the prediction specification). The Tidy Finance implementation merges the characteristic data with a prepared CRSP monthly file and uses the return corresponding to the characteristic date, relying on the data provider's note that the characteristic file's timing has already been adjusted.

Our current full-universe panel does not use that CRSP return series. Instead, it constructs:

[
y_{i,t}^{proxy} = mom1m_{i,t+1},
]

where (t+1) must be the **next consecutive calendar month** for the same PERMNO. The next-row value is not accepted if the next observation is not from the following calendar month.

This construction was selected after an independent alignment experiment. Across seven candidate ticker/PERMNO pairs, the next-row alignment matched separate Yahoo-based monthly return observations much more closely than same-month or previous-row alignment (3,335 consecutive paired observations; correlation approximately 0.99581 and MAE approximately 0.00066). This is useful evidence for the timing relationship in the tested sample, but it does not establish that `mom1m` is an unmodified CRSP return or that it is equivalent to a CRSP total/excess-return series.

Important limitations of this proxy:

1. The raw `mom1m` field may be winsorized or otherwise transformed upstream.
2. It is not established as CRSP excess return.
3. It may not include complete delisting-return treatment.
4. Its point-in-time construction and information timing require further independent audit.
5. The cross-stock validation used candidate identifier mappings, not a certified historical ticker-to-PERMNO crosswalk.

The report therefore consistently calls this variable the **next-month `mom1m` return proxy**, not “CRSP return” or “excess return.”

### 2.5 Final modeling panel and audit

The panel builder retained the feature panel as the audit base, checked the availability of a consecutive following-month target, and excluded observations without a usable target only after recording the counts.

| Audit item | Count |
|---|---:|
| Original stock-month feature rows | 4,117,300 |
| Rows with consecutive following month / usable proxy | 4,078,842 |
| Rows without usable proxy | 38,458 |
| Eligible distinct PERMNO identifiers | 32,642 |

The eligible panel is saved locally as:

`data/gkx_panel_mom1m_public.csv`

The panel contains:

`permno`, `DATE`, `mom1m_scaled`, `bm_scaled`, `mvel1_scaled`, `return_proxy`.

The annual audit is saved as `data/gkx_panel_mom1m_audit_by_year.csv`. The audit confirms key matching and target availability under the project's construction rules; it does not certify the proxy as an original CRSP return.

## 3. Sample splitting and hyperparameter tuning

### 3.1 Annual expanding-window design

The project follows the core temporal structure described in the Tidy Finance article and GKX: the first out-of-sample year is 1987; the initial training period is 1957–1974; the validation period is 1975–1986; and the first test year is 1987.

For each test year (T):

- Training: January 1957 through December (T-13).
- Validation: the 12 calendar years from (T-12) through (T-1).
- Test: the calendar year (T).

The model is refitted once per year, not once per month. Each year, the training sample expands and the fixed-length 12-year validation window rolls forward by one year.

| Test year | Training period | Validation period | Test period |
|---|---|---|---|
| 1987 | 1957–1974 | 1975–1986 | 1987 |
| 1988 | 1957–1975 | 1976–1987 | 1988 |
| 1989 | 1957–1976 | 1977–1988 | 1989 |
| … | Expands by one year | Rolls forward by one year | One year |
| 2016 | 1957–2003 | 2004–2015 | 2016 |

No ordinary randomly shuffled cross-validation is used. The temporal ordering is retained to avoid using future observations to select models for earlier test periods. This follows the reason explicitly given in GKX and highlighted in the Tidy Finance article.

### 3.2 Validation and final refit

For each test year, candidate hyperparameter settings are fitted using the training data only. Their prediction errors are evaluated on the validation window. The candidate with the lowest validation MSE is selected.

The final model is then refitted using all observations available before the test year — training plus validation — with the selected hyperparameters held fixed. It predicts the current test year, which is not used in tuning or fitting.

This separation is essential:

- Training estimates candidate models.
- Validation chooses hyperparameters.
- Training plus validation estimates the final pre-test model.
- Test observations evaluate out-of-sample performance only.

The Tidy Finance article describes the same selection-then-refit logic.

## 4. Model specifications

### 4.1 Elastic Net: APG primary, coordinate descent robustness

Both solvers optimize Elastic Net with an intercept and three ranked characteristics. The common objective in this project is

\[
\min_{b_0,\beta}\;\frac{1}{2N}\|y-b_0-X\beta\|_2^2
+\lambda(1-\rho)\|\beta\|_1+\frac{\lambda\rho}{2}\|\beta\|_2^2,
\qquad \rho=0.5.
\]

The ten penalty candidates are `np.logspace(-4, -1, 10)`; this finite grid is a **project choice**. Each annual test selects the penalty on the prior 12-year validation sample and refits on all observations strictly prior to the test year.

| Setting | APG primary implementation | CD robustness benchmark |
|---|---|---|
| Mixing parameter | `rho=0.5` | `l1_ratio=0.5` |
| Penalty candidates | Ten log-spaced values, 1e-4 to 1e-1 | Same numerical candidates |
| Intercept | Centered/analytically recovered; unpenalized | `fit_intercept=True` |
| Solver | Accelerated proximal gradient | scikit-learn coordinate descent |
| Stopping parameters | `tol=1e-10`, `max_iter=20000` | `tol=1e-4`, `max_iter=5000` |
| Out-of-sample years | 1987–2016 | 1987–2016 |

The APG implementation uses the three-feature Gram matrix to compute prox-gradient updates efficiently. Its optimization and penalty grid follow the broad online-appendix framework, but details such as step-size selection, finite grid and convergence criterion remain **implementation choices**, not an assertion of exact identity with every GKX algorithmic setting.

#### 4.1.1 Solver consistency: APG versus CD

A key-aligned comparison matched **2,489,185** forecasts, with zero unmatched stock-months and zero realized-label mismatches. Both solvers selected the same *numerical* penalty in **30/30** test years. All 30 APG annual outputs passed the *recorded* fixed-point residual check.

| Comparison item | Result |
|---|---:|
| CD pooled OOS R² | +0.249520758854% |
| APG pooled OOS R² | +0.249536392766% |
| APG minus CD OOS R² | +0.000015634 percentage points |
| Mean absolute paired forecast difference | 1.26318949431e-6 |
| RMSE of paired forecast differences | 8.79242829215e-6 |
| Maximum absolute paired forecast difference | 9.12524972048e-5 |
| Same selected numerical penalty | 30/30 years |
| APG recorded residual criterion passed | 30/30 years |

This is strong **numerical prediction agreement**, not proof of identical optimizer iterates or exact objective solutions. The CD training script saves annual intercepts and coefficients in `data/elastic_net_recursive_yearly.csv`; the existing solver-comparison run did **not** complete a direct paired coefficient/objective-value comparison. Its historical message that CD coefficients were unavailable should not be interpreted as meaning the underlying CD yearly CSV lacks those columns. Different solver tolerances remain a plausible source of small discrepancies.

### 4.2 Random Forest

The completed recursive Random Forest uses scikit-learn's `RandomForestRegressor`.

| Setting | Project value |
|---|---|
| Number of trees | 50 |
| Candidate maximum depths | 1, 2, 3, 4, 5, 6 |
| `max_features` | 3 |
| `min_samples_leaf` | 100 |
| Random seed | 42 |
| Parallel jobs | 2 |
| Selection metric | Validation MSE |

For every test year, six candidate maximum depths are evaluated on the same training and validation periods. The depth with the lowest validation MSE is selected, and the final forest is refitted on all pre-test observations.

**Important difference from the paper:** the Tidy Finance article reports that GKX's Appendix Table A.5 specifies 300 trees and tunes the number of features considered at each split over values such as 3, 5, 10, 20, 30, and 50, alongside tree depth. The article's own R implementation uses 300 trees and approximates depth tuning through `min_n`, because its `ranger` engine does not expose maximum depth in the same way.

Our completed experiment instead uses 50 trees, fixes `max_features=3`, and fixes `min_samples_leaf=100`. This was a computationally manageable initial configuration, not a faithful reproduction of the full GKX Random Forest hyperparameter search. Consequently, the current RF result cannot be used to claim that the original paper's RF specification has been replicated.

## 5. Evaluation methodology

### 5.1 Mean squared prediction error

For realized proxy return (y_{i,t}) and forecast (hat y_{i,t}), test MSE is:

[
MSE = \frac{1}{N}\sum_{i,t}(y_{i,t}-\hat y_{i,t})^2.
]

Lower MSE indicates smaller average squared forecast errors on the evaluation sample.

### 5.2 Out-of-sample R-squared relative to zero

The project reports OOS R-squared against a zero-return forecast:

[
R^2_{OOS}
=1-\frac{\sum_{i,t}(y_{i,t}-\hat y_{i,t})^2}
{\sum_{i,t}y_{i,t}^{2}}.
]

The denominator is the squared-error loss of predicting zero for every observation. Thus:

- Positive OOS R²: the model improves on the zero-return benchmark.
- Zero OOS R²: the model matches the benchmark.
- Negative OOS R²: the model performs worse than the benchmark.

This is not the conventional in-sample regression R² based on deviations from the sample mean.

### 5.3 Common-sample comparison

The comparison script matched Elastic Net and Random Forest forecasts one-to-one by (`permno`, `DATE`). The reported audit found:

- Elastic Net forecasts: 2,489,185.
- Random Forest forecasts: 2,489,185.
- Matched observations: 2,489,185.
- Elastic-Net-only observations: 0.
- Random-Forest-only observations: 0.
- Maximum difference between the matched realized target values: 0.

This supports a like-for-like descriptive comparison on the same stock-month test observations.

## 6. Empirical results

### 6.1 Overall performance, 1987–2016

| Model | Stock-month forecasts | Pooled OOS R² |
|---|---:|---:|
| **Elastic Net (APG; primary)** | 2,489,185 | **+0.249536393%** |
| Elastic Net (CD; robustness) | 2,489,185 | +0.249520759% |
| **Random Forest (50 trees)** | 2,489,185 | **+0.385939344%** |
| RF minus APG | Matched sample | **+0.136402951 percentage points** |
| RF minus CD | Matched sample | +0.136418585 percentage points |

The pooled R² comparison is descriptive and **stock-month weighted** through the pooled sum of squared errors. The inferential test below uses a different estimand: the **equally weighted time-series mean of monthly cross-sectional loss differences**. These should not be conflated.

### 6.2 Annual performance

The annual OOS R² values below are from the completed original CD-versus-RF recursive runs (CD entries generally round closely to APG but are **not** re-labelled APG). Values are percentages; APG annual details are stored separately in the APG output directory.

| Year | Elastic Net | Random Forest |
|---:|---:|---:|
| 1987 | -0.7709 | -0.7993 |
| 1988 | +1.2052 | +1.2729 |
| 1989 | +0.0693 | +0.3598 |
| 1990 | -2.1592 | -1.9811 |
| 1991 | +1.4916 | +1.7452 |
| 1992 | +0.9723 | +1.4715 |
| 1993 | +1.3180 | +1.6980 |
| 1994 | -1.3931 | -1.1921 |
| 1995 | +1.6001 | +1.8791 |
| 1996 | +0.7143 | +0.9006 |
| 1997 | +0.9498 | +1.2411 |
| 1998 | -0.3860 | -0.2833 |
| 1999 | +0.7560 | +1.0575 |
| 2000 | -0.7338 | -0.5386 |
| 2001 | +0.6901 | +1.2097 |
| 2002 | -0.7155 | -1.0868 |
| 2003 | +2.9028 | +2.8877 |
| 2004 | +1.6457 | +1.4086 |
| 2005 | -0.2435 | -0.6465 |
| 2006 | +1.6151 | +1.2823 |
| 2007 | -2.1481 | -2.7102 |
| 2008 | -3.2159 | -3.0833 |
| 2009 | +1.7396 | +2.0037 |
| 2010 | +1.7769 | +1.9640 |
| 2011 | -1.8754 | -2.4200 |
| 2012 | +1.2517 | +1.1648 |
| 2013 | +2.9914 | +3.0413 |
| 2014 | -0.5842 | -0.9500 |
| 2015 | -1.6612 | -1.6612 |
| 2016 | +0.8800 | +0.7998 |

The annual results show meaningful variation across years. Both models have negative OOS R² in several test years, including 2007–2008. A positive pooled OOS R² therefore does not mean the models outperform the zero benchmark in every year.

### 6.3 Interpretation of predictive performance

1. **Positive but small pooled gains:** all three models modestly beat the zero-return benchmark in pooled OOS R².
2. **RF advantage is not universal:** there are annual negative model R² values and months when APG has lower mean squared loss.
3. **Statistical evidence is conditional:** section 7 reports a 5%-level unadjusted result for the equal-month loss contrast, not a guarantee of superior portfolio performance, full-sample GKX replication, or robustness to all inference choices.
4. **Reduced specification:** three characteristics and a return proxy omit the original empirical study's broader predictors, adjusted excess-return target and economic-value analysis.

### 6.4 Random Forest variable importance (MDI)

The RF annual output `data/rf_recursive_50trees_leaf100/rf_recursive_yearly.csv` records normalized training-time mean-decrease-in-impurity (MDI) contributions for all three features in the final refitted forest. The analysis aggregates annual importance using **equal weight per model year** (not weight by number of stocks).

| Feature | Equal-year mean MDI | Annual minimum | Annual maximum | Years ranked #1 |
|---|---:|---:|---:|---:|
| `mom1m` | **0.674356 (67.44%)** | 0.5188 | 1.0000 | **30/30** |
| `mvel1` | 0.225308 (22.53%) | 0.0000 | 0.3630 | 0/30 |
| `bm` | 0.100336 (10.03%) | 0.0000 | 0.1428 | 0/30 |

In **1987, 1988, 1989, 2014 and 2015**, `mom1m` accounts for 100% of recorded MDI. Selected maximum depths in these years are **1, 2, 2, 1, 1**, respectively. In 1995, with depth 3, the approximate MDI allocation is **66.21%** `mom1m`, **21.37%** `mvel1`, and **12.42%** `bm`. The depth-selection pattern is descriptive: the six candidate depths are chosen by historical validation MSE, with 2000–2011 selecting depth 6.

**Interpretation boundary:** MDI is a *within-model, training-time* split-impurity metric. It is **not** the share of realized return variance explained, independent OOS feature value, a causal factor effect, or proof of macroeconomic regime dependence. Correlated predictors, shallow trees, a restricted three-feature choice set and changing expanding estimation windows influence these importances. The dominance of short-term reversal is qualitatively compatible with GKX's broader attention to price-trend signals but cannot reproduce their full-universe feature ranking. OOS permutation or ablation importance would require access to fitted forests or additional refitting; it has not been conducted.

Artifacts: `data/rf_feature_importance_analysis/rf_mdi_summary.csv`, `rf_mdi_yearly.csv`, `rf_mdi_mean.png`, and `rf_mdi_by_year.png`.

## 7. Statistical comparison and inference: GKX-style modified DM

### 7.1 Cross-sectionally aggregated loss differential

The completed principal test compares **Elastic Net (APG)** and **Random Forest** on the same 2,489,185 stock-month observations. Rather than treating individual stocks in the same month as independent statistical units, it first averages squared forecast errors cross-sectionally:

\[
MSFE_{m,t}=\frac{1}{N_t}\sum_{i=1}^{N_t}(y_{i,t}-\hat y_{i,t}^{(m)})^2,
\qquad
d_t=MSFE_{APG,t}-MSFE_{RF,t}.
\]

A positive `d_t` favors RF. The null and two-sided alternative are `H0: E[d_t] = 0` and `H1: E[d_t] != 0`. The modified DM-style statistic is

\[
DM_{NW}=\frac{\bar d}{SE_{NW}(\bar d)}.
\]

The Newey–West long-run variance estimator uses Bartlett weights and the centered monthly series. Two-sided p-values use the large-sample standard-normal approximation. **Twelve lags were chosen as the primary project specification**; zero and six lags are sensitivity checks. These choices should not be presented as verified identical to GKX's exact bandwidth/finite-sample correction. This is a **GKX-style cross-sectionally aggregated modified DM**, not an exact claim to reproduce the paper's entire multiple-model testing procedure.

### 7.2 Main APG versus RF inference

The 360 consecutive months from 1987-01 through 2016-12 contain 2,489,185 paired predictions. RF has a smaller cross-sectional monthly MSFE in **195** months; APG is better in **165** months; none are tied. This count is descriptive, not the test statistic.

- Equal-month mean `APG MSFE - RF MSFE`: **+0.0000282605**.
- Stock-month-weighted counterpart: **+0.0000324277**.

| Newey–West lags | Modified DM z | HAC standard error | Two-sided p-value | 95% confidence interval for mean monthly loss difference |
|---:|---:|---:|---:|---|
| 0 | +3.0592 | 0.0000092379 | 0.002219 | [+0.0000101541, +0.0000463668] |
| 6 | +2.3348 | 0.0000121039 | 0.019553 | [+0.0000045368, +0.0000519841] |
| **12 (primary)** | **+2.0860** | **0.0000135474** | **0.036975** | **[+0.0000017076, +0.0000548133]** |

The principal 12-lag comparison rejects equal average monthly forecast losses at the **unadjusted 5% level** in favor of RF; it does not reject at 1%. The conclusion assumes an adequate HAC large-sample approximation and does not constitute a multi-model familywise-error-adjusted claim.

### 7.3 Alternative Elastic Net solver: CD versus RF

Re-running the same loss-difference inference with the coordinate-descent Elastic Net benchmark gives:

| Metric (12 lags) | APG vs RF (primary) | CD vs RF (robustness) |
|---|---:|---:|
| Mean monthly EN-minus-RF loss | +0.0000282605 | +0.0000282650 |
| DM z | +2.0860 | +2.0864 |
| Two-sided p-value | **0.036975** | **0.036938** |
| Unadjusted 5%-level conclusion | RF lower average loss | RF lower average loss |

This solver swap leaves the observed significance conclusion unchanged. It does **not** address the separate return-proxy, predictor-selection, tuned-hyperparameter or macroeconomic-regime limitations.

Files: `data/apg_rf_significance/modified_dm/apg_rf_modified_dm.csv`, `apg_rf_modified_dm_report.md`, and `data/model_comparison/comparison_hac_results.csv`.

## 8. Reproducibility and implementation inventory

Scripts used in the three-characteristic workflow (some reside **only on the local Windows project**, not necessarily in the GitHub default branch):

| Script | Role / completion |
|---|---|
| `preprocess_features_stream.py` | Monthly cross-sectional ranking/scaling |
| `impute_features.py` | Monthly median imputation with documented all-missing fallback |
| `validate_imputed_features.py` | Feature-panel integrity checks |
| `validate_mom1m_returns.py` | Independent seven-pair Yahoo return proxy alignment check |
| `build_gkx_full_panel.py` | Full-universe eligible stock-month panel with proxy target |
| `audit_gkx_features_labels.py` | Raw feature coverage and three-offset mechanical target diagnostics |
| `split_gkx_panel.py` | Temporal partition audit |
| `train_elastic_net_recursive.py` | 30-year CD benchmark; annual coefficients and forecasts |
| `train_elastic_net_apg_recursive.py` | **Completed** 30-year APG; annual tuning, convergence and forecasts |
| `compare_elastic_net_solvers.py` | **Completed** 2,489,185-key APG-versus-CD forecast and penalty comparison |
| `train_random_forest_recursive.py` | **Completed** 30-year RF; predictions, depth and yearly MDI |
| `compare_models.py` | Initial CD-vs-RF matched-sample and monthly loss comparison |
| `test_model_significance-v2.py` | Completed CD-vs-RF 0/6/12-lag HAC inference |
| `compare_apg_rf_significance.py` | Completed APG-vs-RF matched forecasts, monthly loss series |
| `test_modified_dm_apg_rf.py` | **Completed** GKX-style modified DM with 0/6/12-lag HAC and Markdown results |
| `analyze_rf_feature_importance.py` | **Completed** 30-model RF annual MDI summary and plots |

Principal experiment artifacts (local, not assumed available in GitHub):

- `data/gkx_panel_mom1m_public.csv`: 4,078,842 eligible rows and three scaled features.
- `data/elastic_net_recursive_yearly.csv` and `data/elastic_net_recursive_predictions.csv`: CD.
- `data/elastic_net_apg_recursive/`: APG yearly summary and per-year validation/predictions.
- `data/elastic_net_solver_comparison/`: solver agreement metrics.
- `data/rf_recursive_50trees_leaf100/`: RF yearly summary and per-year predictions.
- `data/model_comparison/`: CD vs RF, including HAC sensitivity.
- `data/apg_rf_significance/`: APG vs RF comparison and `modified_dm/` outputs.
- `data/rf_feature_importance_analysis/`: annual MDI summary/plots.
- `data/feature_label_audit/`: source-characteristic completeness and mechanical alignment checks.

**Re-running:** preserve input files and locally generated scripts, use one environment and consistent package versions, and run data construction → model fitting → model comparisons → modified DM → MDI. The comparison, DM and MDI scripts do not refit the models. Record hashes for `datashare.csv` and the eligible panel, environment versions, seeds and the exact repository commit before an external reproducibility claim. Some script/output files have not yet been committed or backed up remotely.

## 9. Limitations and threats to interpretation

The following limitations are material and should remain visible in any presentation of the results.

### 9.1 Proxy target rather than CRSP returns

The largest limitation is the dependent variable. The next-month `mom1m` proxy is not the CRSP excess-return series used by GKX. It may contain upstream winsorization and does not establish full delisting-return coverage. Any conclusion about stock-return predictability must therefore be qualified.

### 9.2 Reduced predictor set

Only three characteristics are included. The original design uses a much broader firm-characteristic set, industry indicators, macro predictors, and interactions. The project cannot be expected to reproduce the original model ranking or predictive performance with this restricted feature set.

### 9.3 Preprocessing equivalence

The rank transformation formula differs slightly from the displayed formula in the Tidy Finance article. The median-imputation and all-missing fallback are documented, but the full original preprocessing pipeline has not been reproduced.

### 9.4 Hyperparameter and solver fidelity

The completed APG follows the appendix-inspired update structure but uses project-specific penalty candidates, step size and stopping rules. A separate CD implementation provides numerical cross-checks, not proof of exact coefficient equality. The Random Forest has **50 trees**, `max_features=3` (all available features considered for splitting), and `min_samples_leaf=100`, rather than the reported 300-tree and broader `max_features` search of the original study.

### 9.5 Statistical uncertainty

The **primary 12-lag modified DM p-value of 0.036975** is an unadjusted, two-sided approximate p-value for an equal-month average loss differential, not a test of the pooled OOS R² difference itself. It is near the 5% threshold and depends on lag choice, time-series assumptions and the restricted model specification. There is no comprehensive multiple-testing adjustment or independent block-bootstrap/subperiod robustness analysis.

### 9.6 Economic value not yet tested

The current report evaluates prediction errors only. It does not yet reproduce the paper's monthly decile sorts, value-weighted or equal-weighted portfolios, long-short strategy, transaction-cost analysis, or investment performance evaluation.

## 10. Remaining work (not required to document the completed three-feature exercise)

The main forecast, solver comparison, modified DM, and descriptive MDI analyses are now **completed**. Remaining issues for stronger scientific equivalence or further research are:

1. **Return-label and point-in-time audit:** establish source `mom1m` construction, winsorization, delisting handling, stock ID correspondence, predictor release lags and CRSP-compatible *excess-return* targets. The prior seven-stock Yahoo alignment (correlation ~0.99581; MAE ~0.00066) is supportive, not certification of all stock-months.
2. **Direct optimization audit:** pair saved CD and APG coefficients/intercepts, compute objective values, and examine stopping tolerances, including the final APG fixed-point residual criterion.
3. **Preprocessing sensitivity:** evaluate the impact of the project's bounded rank formula versus the reference alternative, especially ties and missing values.
4. **Inference sensitivity:** consider block bootstrap, subperiod DM tests, and prespecified multiple-testing adjustments if the comparison family is expanded.
5. **OOS importance and economics:** if scientifically valuable, refit or preserve fitted forests for out-of-sample permutation/ablation importance; separately study portfolio sorts, turnover and transaction costs.
6. **Longer-term expansion, not part of the present report:** richer company characteristics, macro predictors and interactions; revisit RF 300-tree and random-subset hyperparameter settings once input dimensionality justifies it.
7. **Reproducibility packaging:** commit locally verified scripts, document environment versions, archive hashes and accessible result summaries without uploading licensed or overly large underlying data without appropriate rights.

## 11. Conclusion

This project completes an end-to-end **three-characteristic methodological replication** inspired by GKX (2020): reproducible preprocessing, a 4,078,842-row eligible stock-month panel, 30 annual expanding-window predictions for two modeling methods, a 30-year APG versus CD numerical check, an APG-versus-RF GKX-style modified DM test, and descriptive RF variable-importance analysis.

On 2,489,185 matched OOS stock-month forecasts from 1987–2016, RF's pooled OOS R² is **+0.385939%**, versus **+0.249536%** for APG Elastic Net. The primary 12-month Newey–West modified-DM-style test detects lower equal-month average squared forecast loss for RF (**z = 2.0860, two-sided p = 0.036975**), at the unadjusted 5% significance level. The conclusion is essentially unchanged under the CD substitution. Among the *three available predictors*, `mom1m` dominates training-time RF MDI (mean **67.44%**, top in **30/30** years).

These are meaningful completed **methodological and computational results**, but are conditional on the `mom1m` return proxy, only three characteristics, approximate original hyperparameters, and limited economic/robustness analysis. **This document does not claim to reproduce GKX's full predictor universe, original CRSP excess-return outcomes or headline empirical performance.**

---

## References

1. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009
2. Voigt, S. (2024, June 17). *Replicating Gu, Kelly & Xiu (2020)*. Tidy Finance. https://blog.tidy-finance.org/posts/gu-kelly-xiu-replication/
3. Xiu, D. (public data). *Stock Characteristics Data*. https://dachxiu.chicagobooth.edu/download/datashare.zip
4. GKX Online Appendix / supplemental material. https://dachxiu.chicagobooth.edu/download/ML_supp.pdf
