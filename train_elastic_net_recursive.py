
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# === 1. Configuration ===

data_dir = Path(__file__).parent / "data"
panel_path = data_dir / "gkx_panel_mom1m_public.csv"

feature_columns = [
    "mom1m_scaled",
    "bm_scaled",
    "mvel1_scaled"
]

# GKX Appendix A.5: rho = 0.5, lambda in [1e-4, 1e-1]
# Ten log-spaced points are OUR implementation choice.
alphas = np.logspace(-4, -1, 10)
l1_ratio = 0.5

first_test_year = 1987
last_test_year = 2016
validation_years = 12

# === 2. Load data once ===

panel = pd.read_csv(
    panel_path,
    usecols=["permno", "DATE", "return_proxy"] + feature_columns
)

panel["year"] = panel["DATE"] // 10000
panel = panel.sort_values("DATE").reset_index(drop=True)

assert not panel.duplicated(["permno", "DATE"]).any()
assert panel[feature_columns + ["return_proxy"]].notna().all().all()

X = panel[feature_columns].to_numpy()
y = panel["return_proxy"].to_numpy()
years = panel["year"].to_numpy()

assert years.min() == 1957
assert years.max() >= last_test_year

results = []
first_prediction = True

summary_path = data_dir / "elastic_net_recursive_yearly.csv"
prediction_path = data_dir / "elastic_net_recursive_predictions.csv"

print("=== RECURSIVE ELASTIC NET ===")

# === 3. Refit once for each test year ===

for test_year in range(first_test_year, last_test_year + 1):

    val_start_year = test_year - validation_years

    # Locate year boundaries in the sorted arrays
    val_start = np.searchsorted(
        years, val_start_year, side="left"
    )
    test_start = np.searchsorted(
        years, test_year, side="left"
    )
    test_end = np.searchsorted(
        years, test_year + 1, side="left"
    )

    # Recursive training window: 1957 onward
    X_train = X[:val_start]
    y_train = y[:val_start]

    # Rolling validation window: exactly 12 years
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]

    # Test: current year only
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # === 4. Select alpha using validation ===

    best_alpha = None
    best_val_mse = float("inf")

    for alpha in alphas:

        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=5000,
            tol=1e-4
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        val_mse = mean_squared_error(y_val, val_pred)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = float(alpha)

    # === 5. Refit with all pre-test history ===

    final_model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=5000,
        tol=1e-4
    )

    final_model.fit(
        X[:test_start],
        y[:test_start]
    )

    # === 6. Predict the test year ===

    test_pred = final_model.predict(X_test)

    model_mse = mean_squared_error(y_test, test_pred)
    zero_mse = np.mean(y_test ** 2)
    oos_r2 = 1 - model_mse / zero_mse

    results.append({
        "test_year": test_year,
        "train_end_year": val_start_year - 1,
        "validation_start_year": val_start_year,
        "validation_end_year": test_year - 1,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "best_alpha": best_alpha,
        "l1_ratio": l1_ratio,
        "intercept": float(final_model.intercept_),
        "coef_mom1m": float(final_model.coef_[0]),
        "coef_bm": float(final_model.coef_[1]),
        "coef_mvel1": float(final_model.coef_[2]),
        "validation_mse": best_val_mse,
        "test_mse": model_mse,
        "zero_mse": zero_mse,
        "oos_r2": oos_r2
    })

    # Save yearly predictions with stock/month identifiers
    preds = panel.iloc[test_start:test_end][
        ["permno", "DATE", "return_proxy"]
    ].copy()

    preds["elastic_net_prediction"] = test_pred

    preds.to_csv(
        prediction_path,
        mode="w" if first_prediction else "a",
        header=first_prediction,
        index=False
    )

    first_prediction = False

    # Save yearly progress
    pd.DataFrame(results).to_csv(
        summary_path,
        index=False
    )

    print(
        f"Year {test_year} | "
        f"Alpha {best_alpha:.6f} | "
        f"Val MSE {best_val_mse:.8f} | "
        f"OOS R2 {oos_r2 * 100:.4f}%"
    )

# === 7. Overall OOS R2 ===

predictions = pd.read_csv(prediction_path)

actual = predictions["return_proxy"].to_numpy()
predicted = predictions["elastic_net_prediction"].to_numpy()

overall_r2 = 1 - (
    np.sum((actual - predicted) ** 2)
    / np.sum(actual ** 2)
)

print("\n=== FINISHED ===")
print("Test years:", first_test_year, "-", last_test_year)
print("Predictions:", len(predictions))
print("Overall OOS R2 (%):", overall_r2 * 100)
print("Yearly results saved:", summary_path)
print("Predictions saved:", prediction_path)