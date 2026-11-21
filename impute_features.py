
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "data"

input_path = data_dir / "gkx_features_3char_full.csv"
output_path = data_dir / "gkx_features_3char_imputed.csv"

feature_columns = [
    "mom1m_scaled",
    "bm_scaled",
    "mvel1_scaled"
]

print("=== LOADING FEATURES ===")

df = pd.read_csv(input_path)

print("Total rows:", len(df))

print("\n=== MISSING VALUES BEFORE ===")
print(df[feature_columns].isna().sum())

# Save original feature values before imputation
original_values = df[feature_columns].copy()

# Calculate the median separately for each DATE
monthly_medians = df.groupby("DATE")[
    feature_columns
].transform("median")

print("\n=== MONTHS WITHOUT ANY BM VALUES ===")

bm_counts = df.groupby("DATE")["bm_scaled"].count()

empty_bm_months = bm_counts[bm_counts == 0]

print("Number of months:", len(empty_bm_months))

print(empty_bm_months.to_string())

print(
    "Rows in these months:",
    df["DATE"].isin(empty_bm_months.index).sum()
)

# Fill missing values with the corresponding
# month's cross-sectional median
df[feature_columns] = df[feature_columns].fillna(
    monthly_medians
)

print("\n=== MISSING VALUES AFTER MEDIAN ===")
print(df[feature_columns].isna().sum())

# Fallback only for months where an entire
# characteristic has no valid observations
remaining_missing = df[feature_columns].isna().sum().sum()

print("\nRemaining missing:", remaining_missing)

if remaining_missing > 0:
    print("Applying documented zero fallback")
    df[feature_columns] = df[feature_columns].fillna(0)

assert df[feature_columns].notna().all().all()

print("\n=== ORIGINAL VALUES VALIDATION ===")

for col in feature_columns:

    # Find rows where the original feature was not NaN
    valid_mask = original_values[col].notna()

    # Compare original valid values with values after imputation
    unchanged = df.loc[valid_mask, col].equals(
        original_values.loc[valid_mask, col]
    )

    print(col, "Original values unchanged:", unchanged)

    assert unchanged, f"{col}: original values changed!"

print("\n=== SAVING ===")

df.to_csv(output_path, index=False)

print("Saved rows:", len(df))
print("Saved to:", output_path)

print("\n=== MSFT AFTER IMPUTATION ===")

msft = df[df["permno"] == 10107]

print(
    msft[
        ["DATE"] + feature_columns
    ].head(5).to_string(index=False)
)