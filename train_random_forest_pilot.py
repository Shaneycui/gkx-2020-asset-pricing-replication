
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# === 1. Configuration ===

data_dir = Path(__file__).parent / "data"
panel_path = data_dir / "gkx_panel_mom1m_public.csv"

features = [
    "mom1m_scaled",
    "bm_scaled",
    "mvel1_scaled"
]

test_year = 1987
depths = [2, 4]

# Pilot settings only
n_trees = 50
min_leaf = 100
random_seed = 42

# === 2. Load historical data ===

chunks = []

for chunk in pd.read_csv(
    panel_path,
    usecols=["permno", "DATE", "return_proxy"] + features,
    chunksize=200_000
):
    chunk = chunk[
        chunk["DATE"] < 19880101
    ]

    if len(chunk) > 0:
        chunks.append(chunk)

panel = pd.concat(chunks, ignore_index=True)

panel["year"] = panel["DATE"] // 10000

# === 3. Time-based split ===

train = panel[panel["year"] <= 1974].copy()
val = panel[panel["year"].between(1975, 1986)].copy()
test = panel[panel["year"] == 1987].copy()

assert len(train) > 0
assert len(val) > 0
assert len(test) > 0

assert train["DATE"].max() < val["DATE"].min()
assert val["DATE"].max() < test["DATE"].min()

assert not panel.duplicated(["permno", "DATE"]).any()
assert panel[features + ["return_proxy"]].notna().all().all()

# Sample historical observations to reduce computation
train_sample = train.sample(
    n=min(100_000, len(train)),
    random_state=random_seed
)

val_sample = val.sample(
    n=min(50_000, len(val)),
    random_state=random_seed
)

print("=== SAMPLE SIZES ===")
print("Train:", len(train_sample))
print("Validation:", len(val_sample))
print("Test:", len(test))

# === 4. Tune tree depth ===

results = []

for depth in depths:

    model = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=depth,
        max_features=3,
        min_samples_leaf=min_leaf,
        random_state=random_seed,
        n_jobs=2
    )

    model.fit(
        train_sample[features],
        train_sample["return_proxy"]
    )

    val_pred = model.predict(val_sample[features])

    val_mse = mean_squared_error(
        val_sample["return_proxy"],
        val_pred
    )

    results.append({
        "depth": depth,
        "validation_mse": val_mse
    })

    print(
        f"Depth={depth} | "
        f"Validation MSE={val_mse:.8f}"
    )

# === 5. Select best depth ===

results_df = pd.DataFrame(results)

best_depth = int(
    results_df.loc[
        results_df["validation_mse"].idxmin(),
        "depth"
    ]
)

print("\nBest depth:", best_depth)

# === 6. Refit and test ===

train_val = pd.concat(
    [train_sample, val_sample],
    ignore_index=True
)

final_model = RandomForestRegressor(
    n_estimators=n_trees,
    max_depth=best_depth,
    max_features=3,
    min_samples_leaf=min_leaf,
    random_state=random_seed,
    n_jobs=2
)

final_model.fit(
    train_val[features],
    train_val["return_proxy"]
)

test_pred = final_model.predict(test[features])

model_mse = mean_squared_error(
    test["return_proxy"],
    test_pred
)

zero_mse = np.mean(
    test["return_proxy"].to_numpy() ** 2
)

oos_r2 = 1 - model_mse / zero_mse

print("\n=== TEST RESULTS ===")
print("Test year:", test_year)
print("Test MSE:", model_mse)
print("Zero baseline MSE:", zero_mse)
print("OOS R2 (%):", oos_r2 * 100)

# === 7. Feature importance ===

print("\n=== FEATURE IMPORTANCE ===")

for name, importance in zip(
    features,
    final_model.feature_importances_
):
    print(name, ":", importance)


# === 8. Save complete pilot results ===

# 8.1 Save all validation candidates
results_df["n_trees"] = n_trees
results_df["max_features"] = 3
results_df["min_samples_leaf"] = min_leaf
results_df["random_seed"] = random_seed

results_df.to_csv(
    data_dir / "rf_pilot_validation.csv",
    index=False
)

# 8.2 Save final model configuration and evaluation
final_summary = pd.DataFrame([{
    "test_year": test_year,
    "train_start_year": 1957,
    "train_end_year": 1974,
    "validation_start_year": 1975,
    "validation_end_year": 1986,
    "train_sample_rows": len(train_sample),
    "validation_sample_rows": len(val_sample),
    "final_train_rows": len(train_val),
    "test_rows": len(test),
    "best_depth": best_depth,
    "n_estimators": n_trees,
    "max_features": 3,
    "min_samples_leaf": min_leaf,
    "random_state": random_seed,
    "test_mse": model_mse,
    "zero_mse": zero_mse,
    "oos_r2": oos_r2
}])

final_summary.to_csv(
    data_dir / "rf_pilot_summary.csv",
    index=False
)

# 8.3 Save feature importance
importance_df = pd.DataFrame({
    "test_year": test_year,
    "feature": features,
    "importance_mdi": final_model.feature_importances_
})

importance_df.to_csv(
    data_dir / "rf_pilot_feature_importance.csv",
    index=False
)

# 8.4 Save stock-month predictions
predictions = test[
    ["permno", "DATE", "return_proxy"]
].copy()

predictions["rf_prediction"] = test_pred

predictions.to_csv(
    data_dir / "rf_pilot_1987_predictions.csv",
    index=False
)

print("\n=== PILOT FINISHED ===")

print("Best depth:", best_depth)
print("OOS R2 (%):", oos_r2 * 100)

print("\nSaved:")
print("rf_pilot_validation.csv")
print("rf_pilot_summary.csv")
print("rf_pilot_feature_importance.csv")
print("rf_pilot_1987_predictions.csv")
