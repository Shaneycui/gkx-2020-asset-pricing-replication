"""
Build an exploratory multi-stock GKX panel using sourced PERMNO *candidates*.

IMPORTANT: This is a RESEARCH DIAGNOSTIC, NOT a verified training dataset.
No candidate has been verified across all historical months. Do not use for
publication-quality asset-pricing conclusions.

Source:
  Features: Dacheng Xiu official datashare.zip, preprocessed/imputed locally.
  Returns: HerdingAI stock-price-returns, Yahoo adjusted close pct_change.
  Candidate PERMNOs: research/permno_candidates.csv.

Run from the repo root: python build_stock_panel.py

Requires only pandas. Does not modify source data.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CANDIDATES = ROOT / "research" / "permno_candidates.csv"
FEATURES = DATA / "gkx_features_3char_imputed.csv"
RETURNS = DATA / "monthly_returns_by_ticker.csv"
OUTPUT = DATA / "gkx_panel_exploratory_candidates_2000_2021.csv"
FEATURE_COLS = ["mom1m_scaled", "bm_scaled", "mvel1_scaled"]

# A cautious *research-window choice*, not historical PERMNO verification.
FIRST_MONTH = pd.Period("2000-01", freq="M")
LAST_MONTH = pd.Period("2021-11", freq="M")
# LAST_MONTH is the last feature month whose t+1 return is no later than 2021-12.

def main():
    candidates = pd.read_csv(CANDIDATES, dtype={"ticker": "string", "permno": "Int64"})
    assert candidates["ticker"].is_unique, "Duplicate tickers in candidate list"
    assert candidates["permno"].is_unique, "Duplicate PERMNOs in candidate list"
    assert candidates["permno"].notna().all()
    assert (candidates["status"] == "candidate").all()
    assert candidates[["valid_from", "valid_to"]].isna().all().all(), (
        "Date-qualified mappings must be handled by a separately verified pipeline"
    )

    stock_returns = pd.read_csv(
        RETURNS,
        usecols=["date", "ticker", "asset_class", "monthly_return"],
    )
    stock_returns = stock_returns[
        stock_returns["asset_class"].eq("Equity (single stock)") &
        stock_returns["ticker"].isin(candidates["ticker"])
    ].copy()
    stock_returns["month"] = pd.to_datetime(
        stock_returns["date"], errors="raise"
    ).dt.to_period("M")
    assert not stock_returns.duplicated(["ticker", "month"]).any(), (
        "Duplicate ticker-month returns"
    )

    stock_returns.sort_values(["ticker", "month"], inplace=True)
    # At predictor month t, target is the return in the *following calendar month*.
    stock_returns["following_month"] = (
        stock_returns.groupby("ticker")["month"].shift(-1)
    )
    stock_returns["future_return"] = (
        stock_returns.groupby("ticker")["monthly_return"].shift(-1)
    )
    has_next_calendar_month = (
        stock_returns["following_month"] == (stock_returns["month"] + 1)
    )
    stock_returns.loc[~has_next_calendar_month, "future_return"] = float("nan")
    label_table = stock_returns[
        ["ticker", "month", "future_return"]
    ].copy()

    ids = set(candidates["permno"].astype(int))
    chunks = []
    for chunk in pd.read_csv(
        FEATURES,
        usecols=["permno", "DATE"] + FEATURE_COLS,
        chunksize=200_000,
    ):
        part = chunk[chunk["permno"].isin(ids)].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError("No candidate PERMNOs found in the feature file")

    features = pd.concat(chunks, ignore_index=True)
    features["month"] = pd.to_datetime(
        features["DATE"].astype(str), format="%Y%m%d"
    ).dt.to_period("M")
    assert not features.duplicated(["permno", "month"]).any(), (
        "Duplicate PERMNO-month feature observations"
    )

    # Here ticker comes ONLY from candidate evidence. This is an unverified join.
    features = features.merge(
        candidates[["permno", "ticker"]],
        on="permno", how="inner", validate="many_to_one"
    )
    panel = features.merge(
        label_table, on=["ticker", "month"],
        how="inner", validate="one_to_one"
    )
    panel = panel[
        panel["month"].between(FIRST_MONTH, LAST_MONTH)
    ].copy()
    panel = panel.dropna(subset=FEATURE_COLS + ["future_return"])
    panel["mapping_status"] = "candidate_UNVERIFIED"
    panel["data_scope"] = "exploratory_only"
    panel.sort_values(["month", "ticker"], inplace=True)
    assert not panel.duplicated(["permno", "month"]).any()
    assert not panel.duplicated(["ticker", "month"]).any()
    assert panel[FEATURE_COLS + ["future_return"]].notna().all().all()
    if panel.empty:
        raise ValueError("No joined observations in the exploratory window")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT, index=False)
    summary = panel.groupby("ticker").agg(
        permno=("permno", "first"),
        observations=("month", "size"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    )
    print("=== EXPLORATORY / UNVERIFIED MULTI-STOCK PANEL ===")
    print("Sourced mapping candidates:", len(candidates))
    print("Candidate tickers with matched labels:", panel["ticker"].nunique())
    print("Stock-month observations:", len(panel))
    print("Month range:", panel["month"].min(), "to", panel["month"].max())
    print("\nPer-ticker coverage:")
    print(summary.to_string())
    print("\nSaved:", OUTPUT)
    print("WARNING: ALL PERMNO mappings are CANDIDATES. DO NOT USE AS VERIFIED TRAINING DATA.")

if __name__ == "__main__":
    main()
