"""Compare saved 1987-2016 APG Elastic Net and RF forecasts, then run HAC tests.

No fitting; never modifies source predictions or prior CD-vs-RF comparisons.
Positive APG-minus-RF monthly squared loss favors Random Forest.
The primary inference is for the EQUAL-WEIGHT MEAN across months (not the
stock-month-weighted pooled OOS R-squared difference).
"""
from pathlib import Path
from math import erfc, sqrt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
APG_DIR = DATA / "elastic_net_apg_recursive"
RF_DIR = DATA / "rf_recursive_50trees_leaf100"
OUT_DIR = DATA / "apg_rf_significance"  # Separate from existing CD/RF files
FIRST_YEAR, LAST_YEAR = 1987, 2016
LAGS = (0, 6, 12)  # 12 months is the primary prespecified choice
KEYS = ["permno", "DATE"]
TARGET = "return_proxy"
APG_PRED = "elastic_net_apg_prediction"
RF_PRED = "rf_prediction"


def hac_mean_test(values, lag):
    """Bartlett Newey-West HAC variance of the time-series sample mean."""
    d = np.asarray(values, dtype=np.float64)
    n = len(d)
    if n < 2 or not np.isfinite(d).all() or not 0 <= lag < n:
        raise ValueError("Invalid monthly loss series or HAC lag")
    centered = d - d.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run_variance += 2.0 * (1.0 - k / (lag + 1)) * gamma
    if long_run_variance <= 0 or not np.isfinite(long_run_variance):
        raise ValueError(f"Invalid HAC variance for lag={lag}: {long_run_variance}")
    se = sqrt(long_run_variance / n)
    mean = float(d.mean())
    z = mean / se
    return {
        "months": n,
        "hac_lags": lag,
        "mean_apg_minus_rf_mse": mean,
        "hac_se": se,
        "z_statistic": z,
        "p_value_two_sided": erfc(abs(z) / sqrt(2.0)),
        "ci95_lower": mean - 1.96 * se,
        "ci95_upper": mean + 1.96 * se,
    }


def read_predictions(path, prediction_column):
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved predictions: {path}")
    data = pd.read_csv(path, usecols=KEYS + [TARGET, prediction_column])
    if data.empty or data[KEYS + [TARGET, prediction_column]].isna().any().any():
        raise ValueError(f"Empty/null values in {path}")
    if data.duplicated(KEYS).any():
        raise ValueError(f"Duplicate PERMNO/DATE keys in {path}")
    values = data[[TARGET, prediction_column]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite target or predictions in {path}")
    return data


def main():
    monthly_rows = []
    yearly_rows = []
    total_apg_sse = total_rf_sse = total_zero_sse = 0.0
    total_apg_rows = total_rf_rows = total_matched = 0

    print("=== APG ELASTIC NET vs RANDOM FOREST: 1987-2016 ===", flush=True)
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        apg_path = APG_DIR / f"elastic_net_apg_predictions_{year}.csv"
        rf_path = RF_DIR / f"rf_predictions_{year}.csv"
        a = read_predictions(apg_path, APG_PRED)
        r = read_predictions(rf_path, RF_PRED)
        if not (a["DATE"] // 10000 == year).all():
            raise ValueError(f"APG predictions from other years in {apg_path}")
        if not (r["DATE"] // 10000 == year).all():
            raise ValueError(f"RF predictions from other years in {rf_path}")
        total_apg_rows += len(a)
        total_rf_rows += len(r)
        paired = a.merge(r, on=KEYS, how="outer", validate="one_to_one",
                         indicator=True, suffixes=("_apg", "_rf"))
        if not paired["_merge"].eq("both").all():
            counts = paired["_merge"].value_counts().to_dict()
            raise ValueError(f"Year {year}: differing stock/month keys: {counts}")
        if len(paired) != len(a) or len(paired) != len(r):
            raise ValueError(f"Year {year}: mismatched prediction counts")
        y = paired[f"{TARGET}_apg"].to_numpy(dtype=np.float64)
        y_rf = paired[f"{TARGET}_rf"].to_numpy(dtype=np.float64)
        max_diff = float(np.max(np.abs(y - y_rf)))
        if max_diff > 1e-12:
            raise ValueError(f"Year {year}: target labels differ (max {max_diff})")
        apg_loss = (y - paired[APG_PRED].to_numpy(dtype=np.float64)) ** 2
        rf_loss = (y - paired[RF_PRED].to_numpy(dtype=np.float64)) ** 2
        month = paired["DATE"].astype(np.int64) // 100
        # Keep month-level means separate from pooled stock-month averages.
        tmp = pd.DataFrame({"month": month, "apg_loss": apg_loss, "rf_loss": rf_loss})
        m = tmp.groupby("month", sort=True).agg(
            n_obs=("apg_loss", "size"),
            apg_mse=("apg_loss", "mean"),
            rf_mse=("rf_loss", "mean"),
        ).reset_index()
        m["apg_minus_rf_mse"] = m["apg_mse"] - m["rf_mse"]
        monthly_rows.append(m)
        n = len(y)
        a_sse, r_sse, z_sse = float(apg_loss.sum()), float(rf_loss.sum()), float((y * y).sum())
        if z_sse <= 0:
            raise ValueError(f"Year {year}: zero baseline SSE")
        yearly_rows.append({"test_year": year, "n_obs": n,
                            "apg_oos_r2_pct": (1-a_sse/z_sse)*100,
                            "rf_oos_r2_pct": (1-r_sse/z_sse)*100,
                            "rf_minus_apg_r2_pp": (a_sse-r_sse)/z_sse*100,
                            "apg_mse": a_sse/n, "rf_mse": r_sse/n})
        total_apg_sse += a_sse
        total_rf_sse += r_sse
        total_zero_sse += z_sse
        total_matched += n
        print(f"{year} | matched {n:,} | APG R2 {(1-a_sse/z_sse)*100:+.5f}% "
              f"| RF R2 {(1-r_sse/z_sse)*100:+.5f}%", flush=True)

    monthly = pd.concat(monthly_rows, ignore_index=True).sort_values("month").reset_index(drop=True)
    expected = [year * 100 + month for year in range(FIRST_YEAR, LAST_YEAR + 1)
                for month in range(1, 13)]
    if monthly["month"].tolist() != expected:
        raise ValueError("Expected exactly 360 consecutive months from 198701 to 201612")
    if int(monthly["n_obs"].sum()) != total_matched:
        raise ValueError("Monthly counts do not match yearly counts")
    pooled_loss_difference = (total_apg_sse-total_rf_sse)/total_matched
    weighted_monthly = float(np.average(monthly["apg_minus_rf_mse"], weights=monthly["n_obs"]))
    if not np.isclose(pooled_loss_difference, weighted_monthly, rtol=1e-8, atol=1e-12):
        raise ValueError("Monthly weighted loss disagrees with pooled squared errors")
    if total_apg_rows != total_rf_rows or total_apg_rows != total_matched:
        raise ValueError("Not all predictions matched")

    apg_r2 = (1-total_apg_sse/total_zero_sse)*100
    rf_r2 = (1-total_rf_sse/total_zero_sse)*100
    loss = monthly["apg_minus_rf_mse"].to_numpy(dtype=np.float64)
    hac = pd.DataFrame([hac_mean_test(loss, lag) for lag in LAGS])
    overall = pd.DataFrame([{
        "n_obs": total_matched,
        "months": len(monthly),
        "apg_oos_r2_pct": apg_r2,
        "rf_oos_r2_pct": rf_r2,
        "rf_minus_apg_r2_pp": rf_r2-apg_r2,
        "equal_weight_monthly_loss_diff": float(loss.mean()),
        "stock_month_weighted_loss_diff": weighted_monthly,
        "rf_better_months": int((loss > 0).sum()),
        "apg_better_months": int((loss < 0).sum()),
        "tied_months": int((loss == 0).sum()),
    }])

    # Protect previous CD vs RF comparisons by using a dedicated folder.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_DIR / "apg_rf_monthly_losses.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT_DIR / "apg_rf_yearly.csv", index=False)
    overall.to_csv(OUT_DIR / "apg_rf_overall.csv", index=False)
    hac.to_csv(OUT_DIR / "apg_rf_hac_results.csv", index=False)

    print("\n=== POOLED PERFORMANCE ===")
    print(f"Matched observations: {total_matched:,} | Months: {len(monthly)}")
    print(f"APG OOS R2: {apg_r2:+.9f}%")
    print(f"RF  OOS R2: {rf_r2:+.9f}%")
    print(f"RF - APG R2: {rf_r2-apg_r2:+.9f} percentage points")
    print(f"RF better months: {(loss>0).sum()} | APG better months: {(loss<0).sum()}")
    print(f"Equal-weight monthly mean (APG MSE - RF MSE): {loss.mean():+.10f}")
    print(f"Stock-month-weighted mean loss: {weighted_monthly:+.10f}")
    print("\n=== NEWEY-WEST HAC: EQUAL-WEIGHT MONTHLY LOSS ===")
    print("H0: mean monthly (APG MSE - RF MSE) = 0; positive favors RF")
    print("Primary: 12 lags; sensitivity: 0 and 6; two-sided normal approximation")
    for row in hac.itertuples(index=False):
        print(f"Lag {row.hac_lags:2d} | mean {row.mean_apg_minus_rf_mse:+.10f} "
              f"| SE {row.hac_se:.10f} | z {row.z_statistic:+.4f} "
              f"| p(two-sided) {row.p_value_two_sided:.6f} "
              f"| 95% CI [{row.ci95_lower:+.10f}, {row.ci95_upper:+.10f}]")
    print("\nSaved results in:", OUT_DIR)
    print("Original APG, RF and CD comparison files left unchanged.")


if __name__ == "__main__":
    main()