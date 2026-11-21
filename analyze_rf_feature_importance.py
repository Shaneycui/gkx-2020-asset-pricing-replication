"""Three-characteristic GKX-style RF: descriptive MDI importance audit.

Reads previously saved annual Random Forest summaries. Does not retrain models.
MDI is in-sample reduction in squared-error impurity, not a test-set R2
contribution or evidence for macroeconomic state dependence.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "rf_recursive_50trees_leaf100" / "rf_recursive_yearly.csv"
OUTPUT_DIR = ROOT / "data" / "rf_feature_importance_analysis"
FEATURE_COLS = {"mom1m": "mdi_mom1m", "bm": "mdi_bm", "mvel1": "mdi_mvel1"}
YEARS = list(range(1987, 2017))


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Cannot find RF yearly file: {INPUT_PATH}")
    raw = pd.read_csv(INPUT_PATH)
    needed = ["test_year", "test_rows", *FEATURE_COLS.values()]
    missing = sorted(set(needed).difference(raw.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if raw["test_year"].isna().any() or raw["test_year"].duplicated().any():
        raise ValueError("Missing or duplicate test years; inspect source file")
    df = raw.sort_values("test_year").reset_index(drop=True)
    if df["test_year"].tolist() != YEARS:
        raise ValueError(f"Expected exactly years 1987..2016; got {df['test_year'].tolist()}")
    if df[needed].isna().any().any():
        raise ValueError("Missing values in yearly importance fields")
    for col in FEATURE_COLS.values():
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite MDI values in {col}")
        if not df[col].between(-1e-9, 1 + 1e-9).all():
            raise ValueError(f"MDI outside [0, 1]: {col}")
    sums = df[list(FEATURE_COLS.values())].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Annual MDI scores should sum to 1; inspect source data")
    if (df["test_rows"] <= 0).any():
        raise ValueError("Missing or nonpositive test_rows")

    # Equal weight per test-year: this is the main descriptive summary.
    # Test-row-weighted average is a sensitivity summary of yearly MDI,
    # NOT a pooled forest importance and NOT an OOS performance measure.
    imp = df[list(FEATURE_COLS.values())].to_numpy(dtype=float)
    ranked = np.argsort(-imp, axis=1, kind="stable")
    n_top = np.bincount(ranked[:, 0], minlength=len(FEATURE_COLS))
    labels = list(FEATURE_COLS.keys())
    weights = df["test_rows"].to_numpy(dtype=float)
    summary = pd.DataFrame({
        "feature": labels,
        "mean_mdi_equal_year_weight": imp.mean(axis=0),
        "median_mdi": np.median(imp, axis=0),
        "std_mdi_across_years": imp.std(axis=0, ddof=1),
        "min_mdi": imp.min(axis=0),
        "max_mdi": imp.max(axis=0),
        "years_ranked_first": n_top,
        "test_row_weighted_yearly_mdi": np.average(imp, axis=0, weights=weights),
    }).sort_values("mean_mdi_equal_year_weight", ascending=False).reset_index(drop=True)

    detail = df[["test_year", "test_rows", *FEATURE_COLS.values()]].copy()
    detail = detail.rename(columns={v: k for k, v in FEATURE_COLS.items()})
    detail["top_feature"] = [labels[i] for i in ranked[:, 0]]
    detail["mdi_sum"] = sums
    for label in labels:
        detail[f"rank_{label}"] = [int(np.flatnonzero(ranked[j] == labels.index(label))[0]) + 1 for j in range(len(df))]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "rf_mdi_summary.csv"
    years_path = OUTPUT_DIR / "rf_mdi_yearly.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(years_path, index=False)

    plt.figure(figsize=(8, 4.7))
    plt.bar(summary["feature"], 100 * summary["mean_mdi_equal_year_weight"])
    plt.ylabel("Mean MDI across years (%)")
    plt.title("Random Forest feature importance, 1987–2016 (equal year weights)")
    plt.ylim(0, max(100 * summary["mean_mdi_equal_year_weight"]) * 1.2)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "rf_mdi_mean.png"
    plt.savefig(bar_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for label in labels:
        plt.plot(detail["test_year"], detail[label], marker=".", label=label)
    plt.xlabel("Test year (model refit with expanding history)")
    plt.ylabel("MDI importance (within-year shares)")
    plt.title("Random Forest MDI importance by annual refit")
    plt.xticks(list(range(1987, 2017, 3)))
    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.tight_layout()
    line_path = OUTPUT_DIR / "rf_mdi_by_year.png"
    plt.savefig(line_path, dpi=180)
    plt.close()

    print("=== THREE-FEATURE RANDOM FOREST: MDI AUDIT ===")
    print("Source:", INPUT_PATH)
    print("Years:", int(df.test_year.min()), "-", int(df.test_year.max()))
    print("Annual MDI sums approximately 1: yes")
    print("\nEqual-year-weight importance and annual top rankings:")
    for row in summary.itertuples(index=False):
        print(f"  {row.feature:6} | mean MDI {row.mean_mdi_equal_year_weight:.6f} "
              f"({100*row.mean_mdi_equal_year_weight:.2f}%) | first in "
              f"{row.years_ranked_first}/30 years | min {row.min_mdi:.4f} "
              f"| max {row.max_mdi:.4f}")
    print("\nGenerated:")
    for p in (summary_path, years_path, bar_path, line_path):
        print(" ", p)
    print("\nInterpretation: MDI is normalized within each trained RF, not a fraction of")
    print("stock-return variance explained or an identified macro-regime effect.")
    print("No models retrained. Original yearly RF CSV unchanged.")


if __name__ == "__main__":
    main()