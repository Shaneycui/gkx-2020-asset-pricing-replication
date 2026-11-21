# A three-feature Python replication of GKX (2020)

I wanted to understand how the machine-learning asset-pricing workflow in Gu, Kelly, and Xiu (2020) works in practice. I started with only three firm characteristics and two models: Elastic Net and Random Forest.

This walkthrough follows the readable **explanation → short code → result** approach of [Tidy Finance](https://blog.tidy-finance.org/posts/gu-kelly-xiu-replication/#sample-splitting-and-tuning). The examples below describe my Python implementation, not a copy of the website's R code. See the [full report](REPLICATION_REPORT.md) for all settings.

## 1. Three characteristics

The starting data is Dacheng Xiu's public `datashare.csv`: **4,117,300 stock-month observations** from 1957–2021. I use:

| Variable | Meaning |
|---|---|
| `mom1m` | One-month return / short-term reversal |
| `bm` | Book-to-market |
| `mvel1` | Size |

To put the variables on a comparable scale, I rank each characteristic across stocks in the same month:

```python
# Short illustration of the project's rank scaling
rank = df.groupby("month")["mom1m"].rank(method="average")
n = df.groupby("month")["mom1m"].transform("count")
df["mom1m_scaled"] = 2 * rank / (n + 1) - 1
```

The complete local script buffers months across CSV chunks. Missing values are filled with the monthly median; early months with no `bm` data use a zero fallback.

**Lesson learned:** rows split between CSV chunks must still be ranked together. And a fully missing month cannot have a median.

## 2. A future-return *proxy*, not CRSP returns

The paper predicts subsequent stock excess returns. I did not have its full matching CRSP outcome data, so I used the following month's raw `mom1m` as a *proxy* for each stock.

```python
# Simplified: the real panel builder also verifies consecutive months
df = df.sort_values(["permno", "DATE"])
df["return_proxy"] = df.groupby("permno")["mom1m"].shift(-1)
```

Only targets from the **immediately following calendar month** are kept, leaving **4,078,842** usable records. I checked this alignment against external Yahoo-derived monthly returns for seven candidate stock mappings (3,335 observations; correlation ~0.99581; MAE ~0.00066).

**What I am still checking:** Is Xiu's raw `mom1m` a modified historical monthly return? Was the value used as `return_proxy[t]` actually observable *after* the predictors at time `t`? The independent return comparison is encouraging, but matching a shifted column to itself is only a mechanical consistency check. I cannot yet claim that the proxy is identical to CRSP total or excess returns, or that its information timing is fully verified. Historical ticker/PERMNO identity and delisting treatment also matter.

## 3. Training without looking forward

For the first test year, I trained on **1957–1974**, validated on **1975–1986** and predicted **1987**. Each year moves the window forward. This is taken from the committed [RF script](train_random_forest_recursive.py):

```python
val_start_year = test_year - validation_years
train_mask = years < val_start_year
val_mask = (years >= val_start_year) & (years < test_year)
test_mask = years == test_year
```

I chose hyperparameters using validation MSE and refitted on all observations before each test year. No random train/test shuffle was used.

## 4. Elastic Net: APG and coordinate descent

The first Elastic Net version uses scikit-learn's coordinate descent:

```python
model = ElasticNet(
    alpha=float(alpha), l1_ratio=0.5,
    fit_intercept=True, max_iter=5000, tol=1e-4
)
```

I then implemented accelerated proximal gradient (APG), following the optimization idea in GKX's appendix. Here is the central operation (a **short excerpt**, not the complete solver):

```python
z = theta - gamma * (gram @ theta - cross)
threshold = lam * (1 - rho) * gamma
beta = np.sign(z) * np.maximum(np.abs(z) - threshold, 0) / (
    1 + lam * rho * gamma
)
```

I kept `rho=0.5`, but chose a ten-point penalty grid, `max_iter=20000` and `tol=1e-10` for APG. Those are **my tuning and numerical choices**, not parameters copied unchanged from GKX. The two solvers selected the same penalty in all **30** years. Across **2,489,185** matched predictions, their mean absolute difference was **1.26 × 10⁻⁶**. That is reassuring numerical agreement, though the solvers have different stopping rules.

## 5. Random Forest

I used 50 trees and selected the maximum tree depth by validation MSE:

```python
for depth in [1, 2, 3, 4, 5, 6]:
    model = RandomForestRegressor(
        n_estimators=50, max_depth=depth,
        max_features=3, min_samples_leaf=100,
        random_state=42, n_jobs=2
    )
    model.fit(X_train, y_train)
    val_mse = mean_squared_error(y_val, model.predict(X_val))
```

[Complete RF training script](train_random_forest_recursive.py).

The original study's RF configuration differs notably: its reference appendix describes **300 trees** and a much broader predictor/split search. My 50-tree run fixes `min_samples_leaf=100` and tests depths 1–6. These are **simplified project parameters**, not the original paper's complete RF search. With three predictors and `max_features=3`, all three variables are eligible for each split.

## 6. Out-of-sample results

The models predict **1987–2016**, with **2,489,185** common stock-month observations. I measured prediction error against a zero-return forecast:

| Model | Pooled OOS R² |
|---|---:|
| Elastic Net (APG) | **+0.249536%** |
| Elastic Net (coordinate descent) | +0.249521% |
| Random Forest | **+0.385939%** |

RF does better on this measure, but an R² difference alone does not tell us whether that improvement is statistically reliable.

## 7. Modified Diebold–Mariano comparison

I first calculated squared forecast errors for each stock and averaged them **within each month**. This gives 360 monthly differences:

```python
# After matching the two models on the same stock/months
monthly["loss_difference"] = monthly["apg_mse"] - monthly["rf_mse"]
```

A positive difference favors RF. A Newey–West HAC standard error accounts for autocorrelation in this monthly series.

| HAC lags | DM z | Two-sided p |
|---|---:|---:|
| 0 | +3.0592 | 0.002219 |
| 6 | +2.3348 | 0.019553 |
| **12** | **+2.0860** | **0.036975** |

RF has lower monthly mean squared loss in **195** of 360 months. At 12 lags, the difference is significant at the **unadjusted 5% level**, under the chosen large-sample test assumptions. The code uses cross-sectional monthly loss averages and a normal-approximation HAC test; this is *GKX-style* modified DM, not a claim to reproduce all of GKX's finite-sample or **multiple-comparison/Bonferroni** procedures.

## 8. Which features mattered?

I used the Random Forest's training-time mean decrease in impurity (MDI) scores and averaged them over the 30 yearly models:

| Feature | Mean MDI |
|---|---:|
| `mom1m` | **67.44%** |
| `mvel1` | 22.53% |
| `bm` | 10.03% |

`mom1m` ranks first in every year. Five annual models assign it all recorded MDI importance; their selected trees are shallow.

**Lesson learned:** MDI measures relative splits in a fitted forest. It does **not** mean `mom1m` explains 67% of stock-return variation or prove macroeconomic state dependence.

## 9. Notes I would give another student

The main lessons from my replication notes are easy to miss if one only reads the final R² numbers:

- **Features versus target:** `mom1m`, `bm`, and `mvel1` are useful starting characteristics, but three inputs leave out much of the economic information in GKX. I have **not included interactions between macroeconomic predictors and stock characteristics**.
- **Model settings matter:** the sample split follows the core annual GKX design, while the small feature set, APG stopping rule and several RF hyperparameters reflect my own choices. A method inspired by a paper is not automatically a parameter-for-parameter reproduction.
- **Significance has a scope:** the comparison tests monthly *squared forecast errors*, not the profitability of an investment strategy; the primary HAC specification uses 12 lags and reports an unadjusted p-value.
- **Be precise about `mom1m`:** in this project it is a short-term reversal/one-month return characteristic. It should not be casually described as a standard 12-month momentum signal. I still need stronger independent verification of the next-month return proxy.

The remaining candidate variables recorded in my notes are `mom12m` (medium-term momentum), `beta` (market exposure), `idiovol` (idiosyncratic volatility), `turn` (turnover), and `agr` (asset growth). These are **possible next experiments**, not inputs used for the reported results.

## 10. What this project establishes

I completed a three-feature forecasting workflow: preparing a large panel, using a time-respecting train/validation/test split, fitting Elastic Net and RF, checking solver consistency, testing monthly loss differences, and interpreting RF splits.

The results are useful within that simplified design. For a closer empirical GKX replication, the main improvements would be a verified excess-return target, a broader predictor set and more faithful RF settings. I have kept these limitations explicit rather than treating the current model as a full reproduction.

The original [research progress log](RESEARCH_PROGRESS.md) preserves earlier experiments, open questions and debugging decisions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the scripts and data still needed for an independent rerun.
