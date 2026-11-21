"""Compare saved Elastic Net coordinate-descent and APG forecasts (1987–2016).

Read-only inputs: no training; never overwrites original model outputs.
Run from the GKX project root. Requires pandas and numpy.

This is an empirical prediction-consistency audit, NOT proof that the solvers
used exactly the same objective, attained an identical optimum, or that the
return proxy is equivalent to the original GKX target.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CD_PRED_PATH = DATA_DIR / "elastic_net_recursive_predictions.csv"
CD_YEARLY_PATH = DATA_DIR / "elastic_net_recursive_yearly.csv"
APG_DIR = DATA_DIR / "elastic_net_apg_recursive"
APG_YEARLY_PATH = APG_DIR / "elastic_net_apg_yearly.csv"
OUT_DIR = DATA_DIR / "elastic_net_solver_comparison"
FIRST_YEAR, LAST_YEAR = 1987, 2016
KEYS = ["permno", "DATE"]
TARGET = "return_proxy"
CD_PRED = "elastic_net_prediction"
APG_PRED = "elastic_net_apg_prediction"
# Important: a user may use a different candidate grid; do not silently
# treat 'best_alpha' and 'best_lambda' as equivalent without warning.
TARGET_ATOL = 1e-12
LAMBDA_ATOL = 1e-12
APG_TOL = 1e-10  # current APG run configuration; check config.json if changed


def require_columns(frame, columns, path):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def read_prediction(path, pred_col):
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = KEYS + [TARGET, pred_col]
    frame = pd.read_csv(path, usecols=columns)
    if frame[columns].isna().any().any():
        raise ValueError(f"Missing values in {path}")
    if not np.isfinite(frame[[TARGET, pred_col]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite target/prediction in {path}")
    if frame.duplicated(KEYS).any():
        examples = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head()
        raise ValueError(f"Duplicate stock-month keys in {path}:\n{examples}")
    return frame


def read_yearly(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, ["test_year"], path)
    if frame["test_year"].isna().any() or frame["test_year"].duplicated().any():
        raise ValueError(f"Invalid or duplicated test_year in {path}")
    frame["test_year"] = frame["test_year"].astype(int)
    return frame


def calc_metrics(actual, cd, apg):
    actual = np.asarray(actual, dtype=np.float64)
    cd = np.asarray(cd, dtype=np.float64)
    apg = np.asarray(apg, dtype=np.float64)
    zero_sse = float(np.sum(actual * actual))
    if zero_sse <= 0:
        raise ValueError("Zero baseline SSE is zero; OOS R² is undefined")
    cd_sse = float(np.sum((actual - cd) ** 2))
    apg_sse = float(np.sum((actual - apg) ** 2))
    d = apg - cd
    return dict(
        n_obs=len(actual),
        cd_mse=cd_sse / len(actual),
        apg_mse=apg_sse / len(actual),
        cd_oos_r2=1 - cd_sse / zero_sse,
        apg_oos_r2=1 - apg_sse / zero_sse,
        apg_minus_cd_r2_pp=100 * (cd_sse - apg_sse) / zero_sse,
        max_abs_prediction_diff=float(np.max(np.abs(d))),
        mean_abs_prediction_diff=float(np.mean(np.abs(d))),
        rmse_prediction_diff=float(np.sqrt(np.mean(d * d))),
        mean_signed_prediction_diff=float(np.mean(d)),
        zero_sse=zero_sse,
        cd_sse=cd_sse,
        apg_sse=apg_sse,
    )


def main():
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    # Load all expected files before writing any output.
    required = [CD_PRED_PATH, CD_YEARLY_PATH, APG_YEARLY_PATH] + [
        APG_DIR / f"elastic_net_apg_predictions_{y}.csv" for y in years
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    cd_yearly = read_yearly(CD_YEARLY_PATH)
    apg_yearly = read_yearly(APG_YEARLY_PATH)
    cd_yearly = cd_yearly.set_index("test_year")
    apg_yearly = apg_yearly.set_index("test_year")
    for name, frame in [("CD", cd_yearly), ("APG", apg_yearly)]:
        missing_years = sorted(set(years) - set(frame.index))
        if missing_years:
            raise ValueError(f"{name} summary missing years: {missing_years}")

    cd = read_prediction(CD_PRED_PATH, CD_PRED)
    cd["test_year"] = cd["DATE"] // 10000
    cd = cd.loc[cd["test_year"].between(FIRST_YEAR, LAST_YEAR)].copy()
    apg = pd.concat(
        [read_prediction(APG_DIR / f"elastic_net_apg_predictions_{y}.csv", APG_PRED)
         for y in years], ignore_index=True
    )
    apg["test_year"] = apg["DATE"] // 10000
    if len(apg) == 0 or not apg["test_year"].between(FIRST_YEAR, LAST_YEAR).all():
        raise ValueError("APG prediction files contain dates outside 1987–2016")
    if apg.duplicated(KEYS).any():
        raise ValueError("Duplicate stock-month keys ACROSS APG yearly files")

    # Outer merge explicitly identifies missing predictions on either side.
    merged = cd.merge(apg, on=KEYS, how="outer", validate="one_to_one",
                      suffixes=("_cd", "_apg"), indicator=True)
    unmatched = merged["_merge"].value_counts()
    cd_only = int(unmatched.get("left_only", 0))
    apg_only = int(unmatched.get("right_only", 0))
    print(f"CD rows: {len(cd):,} | APG rows: {len(apg):,}")
    print(f"Matched: {int(unmatched.get('both', 0)):,} | CD-only: {cd_only:,} | APG-only: {apg_only:,}")
    if cd_only or apg_only:
        raise ValueError("Different stock-month samples. Fix before comparing accuracy.")

    target_diff = np.abs(
        merged[f"{TARGET}_cd"].to_numpy(dtype=float)
        - merged[f"{TARGET}_apg"].to_numpy(dtype=float)
    )
    max_target_diff = float(np.max(target_diff))
    print(f"Maximum realized-target mismatch: {max_target_diff:.3e}")
    if not np.isfinite(target_diff).all() or max_target_diff > TARGET_ATOL:
        raise ValueError("Different realized proxy targets. Fix before comparing accuracy.")

    # After auditing alignment, retain one target field and ensure year matches.
    merged["test_year"] = (merged["DATE"] // 10000).astype(int)
    actual = merged[f"{TARGET}_cd"].to_numpy(dtype=float)
    cd_pred = merged[CD_PRED].to_numpy(dtype=float)
    apg_pred = merged[APG_PRED].to_numpy(dtype=float)
    overall = calc_metrics(actual, cd_pred, apg_pred)
    overall["first_test_year"] = FIRST_YEAR
    overall["last_test_year"] = LAST_YEAR
    overall["matched_rows"] = len(merged)
    overall["cd_only_rows"] = cd_only
    overall["apg_only_rows"] = apg_only
    overall["max_abs_actual_diff"] = max_target_diff

    annual = []
    for year in years:
        section = merged.loc[merged["test_year"] == year]
        if section.empty:
            raise ValueError(f"No test forecasts for {year}")
        row = {"test_year": year}
        row.update(calc_metrics(
            section[f"{TARGET}_cd"].to_numpy(),
            section[CD_PRED].to_numpy(),
            section[APG_PRED].to_numpy(),
        ))
        a, b = cd_yearly.loc[year], apg_yearly.loc[year]
        row["cd_alpha"] = float(a["best_alpha"]) if "best_alpha" in cd_yearly.columns else np.nan
        row["apg_lambda"] = float(b["best_lambda"]) if "best_lambda" in apg_yearly.columns else np.nan
        row["same_penalty_numeric_value"] = (
            bool(np.isclose(row["cd_alpha"], row["apg_lambda"], rtol=0, atol=LAMBDA_ATOL))
            if np.isfinite(row["cd_alpha"]) and np.isfinite(row["apg_lambda"]) else np.nan
        )
        # APG coefficients, intercept and convergence are documented; the
        # historical CD yearly output may not contain coefficient columns.
        for col in ["intercept", "coef_mom1m_scaled", "coef_bm_scaled",
                    "coef_mvel1_scaled", "iterations", "prox_residual"]:
            row[f"apg_{col}"] = b[col] if col in apg_yearly.columns else np.nan
        row["apg_meets_residual_threshold"] = (
            bool(float(b["prox_residual"]) <= APG_TOL * (1 + max(
                abs(float(b.get("coef_mom1m_scaled", np.nan))),
                abs(float(b.get("coef_bm_scaled", np.nan))),
                abs(float(b.get("coef_mvel1_scaled", np.nan))),
            ))) if all(k in apg_yearly.columns for k in
                       ["prox_residual", "coef_mom1m_scaled", "coef_bm_scaled", "coef_mvel1_scaled"])
            else np.nan
        )
        # The *completed* historical CD summary often lacks these columns.
        cd_coeff_map = {"intercept": "intercept", "coef_mom1m": "coef_mom1m_scaled",
                        "coef_bm": "coef_bm_scaled", "coef_mvel1": "coef_mvel1_scaled"}
        for cd_col, apg_col in cd_coeff_map.items():
            row[f"cd_{cd_col}"] = a[cd_col] if cd_col in cd_yearly.columns else np.nan
            row[f"abs_coef_diff_{cd_col}"] = (
                abs(float(a[cd_col]) - float(b[apg_col]))
                if cd_col in cd_yearly.columns and apg_col in apg_yearly.columns
                and pd.notna(a[cd_col]) and pd.notna(b[apg_col]) else np.nan
            )
        annual.append(row)

    annual_df = pd.DataFrame(annual)
    overall_df = pd.DataFrame([overall])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annual_file = OUT_DIR / "solver_comparison_yearly.csv"
    overall_file = OUT_DIR / "solver_comparison_overall.csv"
    annual_df.to_csv(annual_file, index=False)
    overall_df.to_csv(overall_file, index=False)

    print("\n=== OVERALL SOLVER COMPARISON ===")
    print(f"CD  OOS R²: {100*overall['cd_oos_r2']:+.9f}%")
    print(f"APG OOS R²: {100*overall['apg_oos_r2']:+.9f}%")
    print(f"APG - CD:   {overall['apg_minus_cd_r2_pp']:+.9f} percentage points")
    print(f"Max absolute prediction difference:  {overall['max_abs_prediction_diff']:.12g}")
    print(f"Mean absolute prediction difference: {overall['mean_abs_prediction_diff']:.12g}")
    print(f"RMSE of prediction differences:     {overall['rmse_prediction_diff']:.12g}")
    n_same = annual_df["same_penalty_numeric_value"].eq(True).sum()
    print(f"Years selecting same numeric penalty: {n_same}/{len(annual_df)}")
    n_res = annual_df["apg_meets_residual_threshold"].eq(True).sum()
    print(f"APG years passing recorded residual check: {n_res}/{len(annual_df)}")
    if annual_df[["cd_intercept", "cd_coef_mom1m", "cd_coef_bm", "cd_coef_mvel1"]].isna().all().all():
        print("Historical CD coefficients unavailable: cannot compare coefficients without saved CD refit outputs.")
    print("NOTE: Matching predictions do not by themselves prove identical objectives or optimums.")
    print("Annual output:", annual_file)
    print("Overall output:", overall_file)


if __name__ == "__main__":
    main()