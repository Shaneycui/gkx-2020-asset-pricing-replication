
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# === 1. Configuration ===

data_dir = Path(__file__).parent / "data"
panel_path = data_dir / "gkx_panel_mom1m_public.csv"

output_dir = data_dir / "rf_recursive_50trees_leaf100"
output_dir.mkdir(parents=True, exist_ok=True)

feature_columns = [
    "mom1m_scaled",
    "bm_scaled",
    "mvel1_scaled"
]

first_test_year = 1987

# First test two years. Change to 2016 after checking.
last_test_year = 2016

validation_years = 12
depths = [1, 2, 3, 4, 5, 6]

n_trees = 50
min_leaf = 100
max_features = 3
random_seed = 42
n_jobs = 2

summary_path = output_dir / "rf_recursive_yearly.csv"

# === 2. Load data once ===

panel = pd.read_csv(
    panel_path,
    usecols=["permno", "DATE", "return_proxy"] + feature_columns
)

panel["year"] = panel["DATE"] // 10000

assert not panel.duplicated(["permno", "DATE"]).any()
assert panel[feature_columns + ["return_proxy"]].notna().all().all()

X = panel[feature_columns].to_numpy()
y = panel["return_proxy"].to_numpy()
years = panel["year"].to_numpy()

assert years.min() == 1957
assert years.max() >= last_test_year

print("=== RECURSIVE RANDOM FOREST ===")
print("Panel rows:", len(panel))
print("Test years:", first_test_year, "-", last_test_year)

# === 3. Load previously completed years ===

if summary_path.exists():
    previous_results = pd.read_csv(summary_path)

    completed = {
        int(row["test_year"]): row.to_dict()
        for _, row in previous_results.iterrows()
    }
else:
    completed = {}

# === 4. Annual recursive training ===

for test_year in range(first_test_year, last_test_year + 1):

    prediction_path = (
        output_dir / f"rf_predictions_{test_year}.csv"
    )

    validation_path = (
        output_dir / f"rf_validation_{test_year}.csv"
    )

    if (
        test_year in completed
        and prediction_path.exists()
        and validation_path.exists()
    ):
        print(f"Year {test_year} already completed. Skipping.")
        continue

    start_time = time.perf_counter()

    val_start_year = test_year - validation_years

    train_mask = years < val_start_year

    val_mask = (
        (years >= val_start_year)
        & (years < test_year)
    )

    test_mask = years == test_year

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # === 5. Hyperparameter selection ===

    validation_results = []

    best_depth = None
    best_val_mse = float("inf")

    for depth in depths:

        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=depth,
            max_features=max_features,
            min_samples_leaf=min_leaf,
            random_state=random_seed,
            n_jobs=n_jobs
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        val_mse = mean_squared_error(y_val, val_pred)

        validation_results.append({
            "test_year": test_year,
            "depth": depth,
            "validation_mse": val_mse,
            "n_estimators": n_trees,
            "max_features": max_features,
            "min_samples_leaf": min_leaf,
            "random_state": random_seed
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_depth = depth

    # === 6. Final refit ===

    # All history before the test year
    final_train_mask = years < test_year

    X_final = X[final_train_mask]
    y_final = y[final_train_mask]

    final_model = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=best_depth,
        max_features=max_features,
        min_samples_leaf=min_leaf,
        random_state=random_seed,
        n_jobs=n_jobs
    )

    final_model.fit(X_final, y_final)

    # === 7. Test prediction ===

    test_pred = final_model.predict(X_test)

    model_mse = mean_squared_error(y_test, test_pred)
    zero_mse = np.mean(y_test ** 2)

    oos_r2 = 1 - model_mse / zero_mse

    # === 8. Feature importance ===

    importance = final_model.feature_importances_

    elapsed = time.perf_counter() - start_time

    # === 9. Save each year's results ===

    validation_df = pd.DataFrame(validation_results)

    validation_df.to_csv(
        validation_path,
        index=False
    )

    predictions = panel.loc[
        test_mask,
        ["permno", "DATE", "return_proxy"]
    ].copy()

    predictions["rf_prediction"] = test_pred

    predictions.to_csv(
        prediction_path,
        index=False
    )

    yearly_result = {
        "test_year": test_year,
        "train_start_year": 1957,
        "train_end_year": val_start_year - 1,
        "validation_start_year": val_start_year,
        "validation_end_year": test_year - 1,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "final_train_rows": len(X_final),
        "test_rows": len(X_test),
        "best_depth": best_depth,
        "validation_mse": best_val_mse,
        "n_estimators": n_trees,
        "max_features": max_features,
        "min_samples_leaf": min_leaf,
        "random_state": random_seed,
        "test_mse": model_mse,
        "zero_mse": zero_mse,
        "oos_r2": oos_r2,
        "mdi_mom1m": float(importance[0]),
        "mdi_bm": float(importance[1]),
        "mdi_mvel1": float(importance[2]),
        "elapsed_seconds": elapsed
    }

    completed[test_year] = yearly_result

    summary_df = pd.DataFrame(
        [completed[year] for year in sorted(completed)]
    )

    temp_path = summary_path.with_suffix(".tmp")
    summary_df.to_csv(temp_path, index=False)
    temp_path.replace(summary_path)

    print(
        f"Year {test_year} | "
        f"Depth {best_depth} | "
        f"Val MSE {best_val_mse:.8f} | "
        f"OOS R2 {oos_r2 * 100:.4f}% | "
        f"Seconds {elapsed:.1f}",
        flush=True
    )

# === 10. Aggregate all completed test years ===

summary_df = pd.read_csv(summary_path)

eligible_years = set(
    range(first_test_year, last_test_year + 1)
)

finished_years = set(
    summary_df["test_year"].astype(int)
)

if not eligible_years.issubset(finished_years):
    print("Some test years are not yet complete.")
else:
    all_actual = []
    all_predicted = []

    for year in sorted(eligible_years):
        df = pd.read_csv(
            output_dir / f"rf_predictions_{year}.csv"
        )

        all_actual.append(
            df["return_proxy"].to_numpy()
        )

        all_predicted.append(
            df["rf_prediction"].to_numpy()
        )

    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_predicted)

    overall_r2 = 1 - (
        np.sum((actual - predicted) ** 2)
        / np.sum(actual ** 2)
    )

    print("\n=== FINISHED ===")
    print("Years:", first_test_year, "-", last_test_year)
    print("Predictions:", len(actual))
    print("Overall OOS R2 (%):", overall_r2 * 100)
    print("Summary:", summary_path)
    print("Yearly prediction files:", output_dir)