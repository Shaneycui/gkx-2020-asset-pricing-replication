"""GKX-style modified Diebold-Mariano (DM) test: Elastic Net APG vs RF.

Uses EXISTING comparison outputs from compare_apg_rf_significance.py:
  data/apg_rf_significance/apg_rf_monthly_losses.csv
  data/apg_rf_significance/apg_rf_overall.csv

No model fitting and no changes to original predictions or previous results.

GKX modification: calculate each model's cross-sectional mean squared forecast
error (MSFE) in each month, THEN apply a time-series mean-difference test to
these 360 monthly observations. Newey-West Bartlett HAC accounts for serial
correlation in the monthly loss differential; no independence is assumed
between stocks within a month.

This is NOT a Harvey-Leybourne-Newbold finite-sample correction, and the choice
of 12 HAC lags (plus 0 and 6 sensitivity checks) is project-specific, NOT
claimed to replicate the original paper's bandwidth choice.
"""
from pathlib import Path
from math import erfc, sqrt

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "apg_rf_significance"
MONTHLY_PATH = INPUT_DIR / "apg_rf_monthly_losses.csv"
OVERALL_PATH = INPUT_DIR / "apg_rf_overall.csv"
OUTPUT_DIR = INPUT_DIR / "modified_dm"
PRIMARY_LAG = 12
HAC_LAGS = (0, 6, PRIMARY_LAG)
START_YEAR = 1987
END_YEAR = 2016


def parse_months(column):
    """Allow YYYYMM and YYYY-MM input; return YYYYMM integers."""
    s = column.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    raw = s.str.replace("-", "", regex=False)
    if not raw.str.fullmatch(r"\d{6}").all():
        raise ValueError("Month must be YYYYMM or YYYY-MM, e.g., 198701")
    result = raw.astype(int)
    if not result.mod(100).between(1, 12).all():
        raise ValueError("Invalid month value found")
    return result


def modified_dm_test(d, lag):
    """Cross-sectionally aggregated DM statistic using Bartlett Newey-West HAC.

    d[t] = monthly mean APG squared loss - monthly mean RF squared loss.
    Positive d favors RF. H0: E[d[t]] == 0. Two-sided asymptotic normal test.
    """
    x = np.asarray(d, dtype=np.float64)
    T = x.size
    if T < 3 or not np.isfinite(x).all() or not 0 <= lag < T:
        raise ValueError("Invalid input series or HAC lag")

    mu = float(np.mean(x))
    v = x - mu
    # gamma(k) = (1/T) sum_{t=k+1}^T (d_t - mu)(d_{t-k} - mu)
    lrv = float(np.dot(v, v) / T)
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(v[k:], v[:-k]) / T)
        lrv += 2 * (1 - k / (lag + 1)) * gamma_k

    if not np.isfinite(lrv) or lrv <= 0:
        raise ValueError(f"Nonpositive/invalid HAC long-run variance, lag={lag}")
    se = sqrt(lrv / T)
    dm_z = mu / se
    return {
        "comparison": "ElasticNet_APG_vs_RandomForest",
        "test": "GKX_style_modified_Diebold_Mariano_cross_sectional_MSFE_HAC",
        "n_months": T,
        "nw_lags": lag,
        "primary": lag == PRIMARY_LAG,
        "mean_apg_minus_rf_msfe": mu,
        "nw_long_run_variance": lrv,
        "nw_se_of_mean": se,
        "dm_z": dm_z,
        "p_value_two_sided": erfc(abs(dm_z) / sqrt(2)),
        "ci95_lower": mu - 1.96 * se,
        "ci95_upper": mu + 1.96 * se,
        "reject_h0_at_5pct": erfc(abs(dm_z) / sqrt(2)) < 0.05,
    }


def main():
    if not MONTHLY_PATH.is_file() or not OVERALL_PATH.is_file():
        raise FileNotFoundError(
            "Run compare_apg_rf_significance.py first. Missing:\n"
            f"  {MONTHLY_PATH}\n  {OVERALL_PATH}"
        )
    monthly = pd.read_csv(MONTHLY_PATH)
    overall = pd.read_csv(OVERALL_PATH)
    req = {"month", "n_obs", "apg_mse", "rf_mse", "apg_minus_rf_mse"}
    if missing := req - set(monthly):
        raise ValueError(f"Missing monthly columns: {sorted(missing)}")
    if len(overall) != 1 or "n_obs" not in overall:
        raise ValueError("Overall comparison must contain exactly one row and n_obs")

    monthly["month"] = parse_months(monthly["month"])
    monthly = monthly.sort_values("month").reset_index(drop=True)
    expected = [year * 100 + m for year in range(START_YEAR, END_YEAR + 1)
                for m in range(1, 13)]
    if monthly["month"].tolist() != expected:
        raise ValueError("Expected every month, exactly once, from 198701 to 201612")
    numeric = ["n_obs", "apg_mse", "rf_mse", "apg_minus_rf_mse"]
    if not np.isfinite(monthly[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Monthly columns contain NaN or infinite values")
    if (monthly["n_obs"] <= 0).any() or not np.allclose(
        monthly["n_obs"], np.round(monthly["n_obs"])
    ):
        raise ValueError("Monthly observation counts must be positive integers")
    if (monthly[["apg_mse", "rf_mse"]] < 0).any().any():
        raise ValueError("Negative mean squared forecast error")

    loss = monthly["apg_minus_rf_mse"].to_numpy(dtype=float)
    if not np.allclose(loss, monthly["apg_mse"] - monthly["rf_mse"],
                       atol=1e-12, rtol=1e-9):
        raise ValueError("Monthly loss differential is not APG MSFE minus RF MSFE")
    weights = monthly["n_obs"].to_numpy(dtype=np.int64)
    n_obs = int(weights.sum())
    if n_obs != int(overall.loc[0, "n_obs"]):
        raise ValueError("Sum of monthly stock observations does not match pooled total")
    weighted_loss = float(np.average(loss, weights=weights))
    if "stock_month_weighted_loss_diff" in overall:
        reported = float(overall.loc[0, "stock_month_weighted_loss_diff"])
        if not np.isclose(weighted_loss, reported, atol=1e-12, rtol=1e-9):
            raise ValueError("Stock-month weighted loss difference does not match pooled results")

    results = pd.DataFrame([modified_dm_test(loss, lag) for lag in HAC_LAGS])
    primary = results.loc[results["nw_lags"] == PRIMARY_LAG].iloc[0]
    rf_win = int((loss > 0).sum())
    apg_win = int((loss < 0).sum())
    tie = int((loss == 0).sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "apg_rf_modified_dm.csv"
    md_path = OUTPUT_DIR / "apg_rf_modified_dm_report.md"
    results.to_csv(csv_path, index=False)

    conclusion = ("reject" if primary["reject_h0_at_5pct"] else "do not reject")
    direction = ("favors RF" if primary["mean_apg_minus_rf_msfe"] > 0
                 else "favors APG" if primary["mean_apg_minus_rf_msfe"] < 0
                 else "is tied")
    markdown = f"""# GKX-style Modified Diebold–Mariano Test: APG vs RF

## Design

- Sample: {START_YEAR}–{END_YEAR}; {len(loss)} consecutive months; {n_obs:,} matched stock-month forecasts.
- Models: three-characteristic Elastic Net (APG) and Random Forest (50 trees).
- For each month, compute cross-sectional average squared forecast errors (MSFE).
- Define `d_t = MSFE_APG,t − MSFE_RF,t`; positive values favor RF.
- Null: `E[d_t] = 0`; alternative: `E[d_t] != 0` (two-sided).
- Test: `DM = mean(d_t) / NW_SE(mean(d_t))`, Bartlett/Newey–West HAC.
- Primary bandwidth: {PRIMARY_LAG} monthly lags; sensitivity: 0 and 6 lags.
- Asymptotic normal p-values, 95% confidence intervals; no multiple-comparison correction for this single prespecified pair.

## Result

| NW lag | Monthly mean loss diff | HAC SE | DM z | Two-sided p | 95% CI |
|---:|---:|---:|---:|---:|---|
"""
    for x in results.itertuples():
        markdown += (f"| {x.nw_lags} | {x.mean_apg_minus_rf_msfe:+.10f} | "
                     f"{x.nw_se_of_mean:.10f} | {x.dm_z:+.4f} | "
                     f"{x.p_value_two_sided:.6f} | "
                     f"[{x.ci95_lower:+.10f}, {x.ci95_upper:+.10f}] |\n")
    markdown += f"""
The prespecified {PRIMARY_LAG}-lag test yields DM z = {primary['dm_z']:+.4f}
(two-sided p = {primary['p_value_two_sided']:.6f}); we **{conclusion}**
equal mean monthly prediction loss at the 5% level. The estimated difference
{direction}. RF has lower monthly MSFE in {rf_win} months, APG in {apg_win}
months, with {tie} ties. The **equal-month-weighted** mean loss difference is
{loss.mean():+.10f}; the **stock-month-weighted** loss difference is
{weighted_loss:+.10f}. The latter is a descriptive pooled quantity, *not* the
object tested by the monthly DM statistic.

## Interpretation and limitations

This is the **cross-sectionally aggregated** modification described by
Gu, Kelly, and Xiu (2020): stocks within a month are not treated as
independent observations for inference. We choose our own HAC bandwidths;
this is not a claim of exact replication of GKX's bandwidth or full testing
family. The normal approximation relies on sufficiently weak serial
dependence in the aggregated monthly loss differential; 360 months does not
guarantee validity. The current three-characteristic design uses a
next-month `mom1m` **return proxy**, not original CRSP excess returns.
The forest has 50 trees and a constrained tuning grid, not GKX's full RF
configuration. Differences in pooled OOS R² alone are not this statistical
test. The result is **not** evidence of a trading strategy's net profitability.

Reference: Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine
Learning*, *Review of Financial Studies*, section comparing forecast methods.
"""
    md_path.write_text(markdown, encoding="utf-8")

    print("=== GKX-STYLE MODIFIED DIEBOLD-MARIANO: ELASTIC NET APG vs RF ===")
    print(f"360 monthly observations verified: {len(loss)} months; {n_obs:,} stock-month forecasts")
    print(f"Monthly RF wins: {rf_win} | APG wins: {apg_win} | ties: {tie}")
    print(f"Mean monthly APG-RF MSFE: {loss.mean():+.10f}")
    print(f"Stock-month-weighted APG-RF MSFE: {weighted_loss:+.10f}")
    print("H0: E[MSFE_APG,t - MSFE_RF,t] = 0; two-sided normal p-values")
    for x in results.itertuples(index=False):
        tag = " [PRIMARY]" if x.primary else ""
        print(f"Lag {x.nw_lags:2d}{tag} | DM z {x.dm_z:+.4f} | HAC SE {x.nw_se_of_mean:.10f} "
              f"| p={x.p_value_two_sided:.6f} | "
              f"95% CI [{x.ci95_lower:+.10f}, {x.ci95_upper:+.10f}]")
    print(f"Primary conclusion: {conclusion} H0 at 5%; effect {direction}.")
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("Original predictions and comparisons unchanged.")


if __name__ == "__main__":
    main()