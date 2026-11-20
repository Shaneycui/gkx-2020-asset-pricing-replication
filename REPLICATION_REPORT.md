# Replication Report: Empirical Asset Pricing via Machine Learning

**A three-characteristic implementation based on public data**  
**Project:** `Shaneycui/gkx-2020-asset-pricing-replication`  
**Report status:** Preliminary computational report — not a full replication  
**Version date:** 20 November 2026

---

## Abstract

This project implements a simplified empirical asset-pricing prediction exercise inspired by Gu, Kelly, and Xiu (2020, hereafter GKX), *Empirical Asset Pricing via Machine Learning*. The purpose is to build a transparent, reproducible workflow for monthly stock-level return prediction, compare a regularized linear model (Elastic Net) with a nonlinear ensemble model (Random Forest), and develop the research and programming skills required for more extensive machine-learning asset-pricing work.

The implementation uses three cross-sectionally ranked firm characteristics — `mom1m`, `bm`, and `mvel1` — from the public characteristic file hosted by Dacheng Xiu. The resulting panel contains 4,078,842 eligible stock-month observations and 32,642 PERMNO identifiers. Because a matching CRSP excess-return panel was not available in the current project environment, the dependent variable is a **proxy** constructed from the following calendar month's raw `mom1m` value. This proxy is not equivalent to the CRSP total or excess return used in GKX.

Using an expanding training sample, a rolling 12-year validation window, and annual model refitting, the completed 1987–2016 experiments produce 2,489,185 out-of-sample forecasts for each model. The pooled out-of-sample R-squared relative to a zero-return forecast is 0.2495% for Elastic Net and 0.3859% for Random Forest. The Random Forest's descriptive advantage is 0.1364 percentage points. No dependence-robust significance conclusion has yet been established.

The results should therefore be interpreted as a **three-characteristic methodological replication and pipeline prototype**, not as a reproduction of GKX's original data, full predictor set, optimization procedures, or headline empirical findings.

## 1. Research objective and reference design

### 1.1 Original study

GKX investigate whether machine-learning methods can improve the prediction of individual stock returns using a large set of firm characteristics and macroeconomic state variables. Their empirical design compares several methods under a common out-of-sample forecasting framework and studies both predictive accuracy and the economic value of the resulting forecasts.

The reference implementation for this project is Stefan Voigt's 2024 Tidy Finance article, [“Replicating Gu, Kelly & Xiu (2020)”](https://blog.tidy-finance.org/posts/gu-kelly-xiu-replication/). The article is useful because it translates key parts of the paper and its online appendix into a practical data and model-fitting workflow. It is an independent implementation, however, and does not remove the need to distinguish the original paper's specifications from the blog author's implementation choices.

### 1.2 Scope of this project

The current project focuses on a deliberately restricted first experiment:

- Three stock characteristics: `mom1m`, `bm`, and `mvel1`.
- Two prediction methods: Elastic Net and Random Forest.
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

### 4.1 Elastic Net

The completed recursive Elastic Net experiment uses scikit-learn's `ElasticNet` estimator.

| Setting | Project value |
|---|---|
| Mixture parameter (`l1_ratio`) | 0.5 |
| Candidate penalty values | 10 log-spaced values from (10^{-4}) to (10^{-1}) |
| Selection metric | Validation MSE |
| Intercept | Included |
| Maximum iterations | 5,000 |
| Solver | scikit-learn coordinate descent |

The Tidy Finance article fixes the Elastic Net mixture at 0.5 and illustrates a two-point penalty grid containing (10^{-4}) and (10^{-1}). Our ten-point log-spaced grid is a project-specific refinement; it is not the exact two-point grid shown in that article.

A further methodological distinction has been identified: GKX's online appendix describes an accelerated proximal-gradient (APG) optimization algorithm. The completed historical Elastic Net results were generated with scikit-learn's coordinate-descent solver, not APG. A separate APG implementation has been drafted for solver-fidelity testing, but it has not yet been confirmed as a completed 30-year run. The coordinate-descent results must not be relabelled as APG results.

### 4.1.1 Optimization Solver Robustness: APG versus Coordinate Descent

To assess numerical robustness, we compared two implementations of Elastic Net: coordinate descent using scikit-learn and an independently implemented accelerated proximal gradient (APG) algorithm following the update structure described in Gu, Kelly, and Xiu (2020).

Both implementations were evaluated over the 1987–2016 out-of-sample period, yielding 2,489,185 perfectly matched stock-month predictions. The realized return-proxy labels were identical across the two datasets, and both implementations selected the same numerical regularization parameter in all 30 test years.

The pooled out-of-sample R² was 0.249521% for coordinate descent and 0.249536% for APG, a difference of approximately 0.000016 percentage points. The mean absolute difference between individual predictions was 1.263 × 10⁻⁶, while the maximum absolute difference was 9.125 × 10⁻⁵.

All 30 APG annual models passed the recorded proximal fixed-point residual check. These findings indicate strong numerical agreement between the two implementations under the current experimental configuration.

Nevertheless, the comparison does not establish exact equality of the optimization objectives or their solutions. Historical coordinate-descent coefficient estimates were not available for direct comparison, and the solvers use different convergence criteria. The results are therefore interpreted as evidence of numerical robustness rather than proof of exact computational equivalence.


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

| Model | Forecast observations | Pooled OOS R² |
|---|---:|---:|
| Elastic Net | 2,489,185 | +0.249521% |
| Random Forest | 2,489,185 | +0.385939% |
| RF minus Elastic Net | Same sample | +0.136419 percentage points |

Under the current proxy target and model configurations, Random Forest has a higher pooled OOS R² than Elastic Net by approximately 0.1364 percentage points.

This difference is descriptive. It is not yet evidence of statistically significant superiority. The models also use different regularization/tree settings and have not both been brought into exact alignment with the original paper's optimizer and hyperparameter search.

### 6.2 Annual performance

The annual OOS R² values below were reported by the completed recursive runs. Values are percentages.

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

### 6.3 Preliminary interpretation

Three restrained observations can be made:

1. **Predictive gains are modest.** The pooled OOS R² values are positive but small relative to the zero-return benchmark.
2. **The RF advantage is not uniform.** Random Forest performs better in many years but worse in others; the annual table does not show universal dominance.
3. **The current feature set may be too narrow to reproduce the full GKX result.** Three characteristics omit much of the cross-sectional and macroeconomic information used in the original paper. The present results cannot isolate whether differences arise from the reduced feature set, target proxy, tuning grid, optimizer, RF configuration, or other data-processing differences.

## 7. Statistical comparison and inference

The comparison workflow also aggregates monthly squared forecast-loss differences. This is preferable to treating millions of stock-month observations as independent because observations within the same month and across adjacent months may be dependent.

A Newey–West heteroskedasticity-and-autocorrelation-consistent (HAC) standard error procedure has been proposed for inference on the mean monthly loss differential. This is a project-level inferential extension; it should not be described as a procedure already completed or as an exact requirement of the Tidy Finance blog's code.

**Status:** the paired descriptive comparison is complete, but the HAC significance test has not yet been confirmed as successfully run and reviewed. Until that is done, no statistical-significance claim should be made about the RF–Elastic Net difference.

## 8. Reproducibility and implementation inventory

The principal scripts and their roles are:

| Script | Role |
|---|---|
| `preprocess_features_stream.py` | Cross-sectional monthly ranking and scaling of the three characteristics |
| `impute_features.py` | Monthly median imputation and documented all-missing fallback |
| `validate_imputed_features.py` | Validate the saved feature panel |
| `validate_mom1m_returns.py` | Independently test candidate timing alignment against external returns |
| `build_gkx_full_panel.py` | Build the eligible full-universe proxy-return panel and audit exclusions |
| `split_gkx_panel.py` | Check time-based sample partitions |
| `train_elastic_net_recursive.py` | Completed 1987–2016 Elastic Net recursive experiment using coordinate descent |
| `train_random_forest_recursive.py` | Completed 1987–2016 Random Forest recursive experiment |
| `compare_models.py` | One-to-one paired prediction comparison and annual/monthly metrics |
| `test_model_significance.py` | Planned HAC inference on monthly loss differences |
| `train_elastic_net_apg_recursive.py` | Draft APG solver branch; not yet a confirmed completed full-period run |

Large source data, processed panels, prediction CSVs, and fitted outputs are stored locally under `data/` and are not backed up by the GitHub code repository. Reproducibility requires preserving those files separately and recording raw-file hashes, software versions, and commit identifiers.

## 9. Limitations and threats to interpretation

The following limitations are material and should remain visible in any presentation of the results.

### 9.1 Proxy target rather than CRSP returns

The largest limitation is the dependent variable. The next-month `mom1m` proxy is not the CRSP excess-return series used by GKX. It may contain upstream winsorization and does not establish full delisting-return coverage. Any conclusion about stock-return predictability must therefore be qualified.

### 9.2 Reduced predictor set

Only three characteristics are included. The original design uses a much broader firm-characteristic set, industry indicators, macro predictors, and interactions. The project cannot be expected to reproduce the original model ranking or predictive performance with this restricted feature set.

### 9.3 Preprocessing equivalence

The rank transformation formula differs slightly from the displayed formula in the Tidy Finance article. The median-imputation and all-missing fallback are documented, but the full original preprocessing pipeline has not been reproduced.

### 9.4 Hyperparameter and solver differences

The Elastic Net uses a ten-value penalty grid and coordinate descent in the completed run; the APG version remains a separate validation task. The Random Forest uses 50 trees, fixed `max_features=3`, and `min_samples_leaf=100`, rather than the original 300-tree setup and broader tuning described in the reference material.

### 9.5 Statistical uncertainty

The RF's higher pooled OOS R² is a descriptive result. Monthly dependence-aware inference is still pending. No claim of statistically significant superiority is warranted.

### 9.6 Economic value not yet tested

The current report evaluates prediction errors only. It does not yet reproduce the paper's monthly decile sorts, value-weighted or equal-weighted portfolios, long-short strategy, transaction-cost analysis, or investment performance evaluation.

## 10. Next steps

The next steps are ordered to address the most consequential threats first.

1. **Audit the proxy target.** Establish the precise timing, construction, winsorization, missingness, and relationship between raw `mom1m` and realized returns. Where possible, obtain an independently sourced CRSP-compatible return panel.
2. **Reconcile the rank transform.** Decide whether to adopt the Tidy Finance min/max-rank transformation or retain the current (2rank/(n+1)-1) formula, and quantify the effect.
3. **Validate APG.** Compare APG and coordinate descent on the exact same Elastic Net objective, including intercept handling, loss normalization, penalty scaling, convergence tolerance, and candidate `lambda` grid. Do not overwrite the completed coordinate-descent benchmark.
4. **Improve RF fidelity.** Increase tree count and implement a defensible tuning grid consistent with the original appendix, subject to computational feasibility.
5. **Complete paired inference.** Run and review HAC inference on monthly loss differences, with lag-sensitivity checks.
6. **Expand predictors.** Add further firm characteristics and, in a later stage, macroeconomic predictors and interactions.
7. **Evaluate economic value.** Construct monthly forecast-sorted portfolios and evaluate realized returns, risk, turnover, and transaction costs.
8. **Improve reproducibility.** Archive input hashes, package versions, model configurations, validation results, predictions, and exact code commits.

## 11. Conclusion

The project has completed a substantial end-to-end computational exercise: a public-data feature pipeline, a large eligible stock-month panel, annual recursive Elastic Net and Random Forest forecasts for 1987–2016, and a paired comparison on 2.49 million common out-of-sample observations.

Within this simplified design, Random Forest achieves a pooled OOS R² of 0.3859%, compared with 0.2495% for Elastic Net, both measured against a zero-return forecast. This is an encouraging engineering and research milestone, but it is not yet evidence that the original GKX conclusions have been reproduced. The proxy target, reduced predictor set, implementation differences, and pending statistical inference materially constrain interpretation.

The most defensible description at this stage is: **a transparent, three-characteristic methodological replication prototype inspired by GKX (2020), with completed recursive model comparisons and clearly documented deviations from the original empirical design.**

---

## References

1. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009
2. Voigt, S. (2024, June 17). *Replicating Gu, Kelly & Xiu (2020)*. Tidy Finance. https://blog.tidy-finance.org/posts/gu-kelly-xiu-replication/
3. Xiu, D. (public data). *Stock Characteristics Data*. https://dachxiu.chicagobooth.edu/download/datashare.zip
4. GKX Online Appendix / supplemental material. https://dachxiu.chicagobooth.edu/download/ML_supp.pdf
