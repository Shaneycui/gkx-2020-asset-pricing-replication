"""Build a public-data GKX stock-month panel from Xiu's raw mom1m.

IMPORTANT: The source `mom1m` may be winsorized and is not CRSP total return.
The Xiu data readme says characteristics at DATE=t align with CRSP RET at t.
Independent validation showed raw mom1m at DATE=t+1 approximates RET at t.
Therefore label(t) = raw_mom1m(t+1), only for consecutive months.

Requires, under data/:
    datashare.csv
    gkx_features_3char_imputed.csv

Does not change either input. Outputs an audit and panel CSV under data/.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "datashare.csv"
FEATURES = DATA / "gkx_features_3char_imputed.csv"
PANEL_OUT = DATA / "gkx_panel_mom1m_public.csv"
AUDIT_YEAR_OUT = DATA / "gkx_panel_mom1m_audit_by_year.csv"

KEY = ["permno", "DATE"]
X = ["mom1m_scaled", "bm_scaled", "mvel1_scaled"]

for path in (RAW, FEATURES):
    if not path.is_file():
        raise FileNotFoundError(f"Input file missing: {path}")

# 1. Reconstruct a raw (unscaled) return proxy, by PERMNO.
raw = pd.read_csv(RAW, usecols=KEY + ["mom1m"])
assert raw[KEY].notna().all().all(), "Missing raw stock/date keys"
assert not raw.duplicated(KEY).any(), "Duplicate raw stock-month keys"
raw = raw.sort_values(KEY).reset_index(drop=True)
raw["month"] = pd.to_datetime(
    raw["DATE"].astype(str), format="%Y%m%d", errors="raise"
).dt.to_period("M")
by_stock = raw.groupby("permno", sort=False)
raw["following_month"] = by_stock["month"].shift(-1)
raw["return_proxy"] = by_stock["mom1m"].shift(-1)
raw["consecutive"] = raw["following_month"].eq(raw["month"] + 1)
raw.loc[~raw["consecutive"], "return_proxy"] = float("nan")

# Keep auditing fields from original raw data before any sample removal.
labels = raw[KEY + ["return_proxy", "consecutive"]].copy()
print("RAW STOCK-MONTH ROWS:", len(raw))
print("Rows with a consecutive following month:", int(raw["consecutive"].sum()))
print("Rows with non-null reconstructed return:", int(raw["return_proxy"].notna().sum()))
del raw

# 2. Read the already-ranked and imputed features; preserve all feature rows.
features = pd.read_csv(FEATURES, usecols=KEY + X)
assert features[KEY].notna().all().all(), "Missing feature stock/date keys"
assert not features.duplicated(KEY).any(), "Duplicate feature stock-month keys"
assert features[X].notna().all().all(), "Unexpected missing processed predictors"

panel = features.merge(
    labels, on=KEY, how="left", validate="one_to_one", indicator=True
)
assert len(panel) == len(features), "Left merge changed feature row count"
print("\n=== MATCH AUDIT ===")
print("Processed feature rows:", len(features))
print(panel["_merge"].value_counts().to_string())
print("Rows with no raw key match:", int(panel["_merge"].eq("left_only").sum()))
print("Rows without a usable return proxy:", int(panel["return_proxy"].isna().sum()))

# 3. Audit *before* exclusions, including all years.
panel["year"] = panel["DATE"] // 10000
audit = panel.groupby("year", sort=True).agg(
    feature_rows=("permno", "size"),
    key_matched=("_merge", lambda x: int((x == "both").sum())),
    usable_labels=("return_proxy", "count"),
    distinct_stocks=("permno", "nunique"),
)
audit["missing_labels"] = audit["feature_rows"] - audit["usable_labels"]
audit["usable_pct"] = (100 * audit["usable_labels"] / audit["feature_rows"]).round(2)
audit.to_csv(AUDIT_YEAR_OUT)
print("\n=== YEARLY LABEL AUDIT (last 12 years) ===")
print(audit.tail(12).to_string())

# 4. Final dataset contains only eligible observations, NOT imputed returns.
eligible = panel[panel["return_proxy"].notna()].copy()
eligible = eligible.drop(columns=["_merge", "consecutive", "year"])
assert eligible[X + ["return_proxy"]].notna().all().all()
assert not eligible.duplicated(KEY).any()
eligible = eligible.sort_values(KEY).reset_index(drop=True)
eligible.to_csv(PANEL_OUT, index=False)

print("\n=== FINISHED ===")
print("Eligible stock-month rows:", len(eligible))
print("Distinct PERMNO:", eligible["permno"].nunique())
print("Date range:", int(eligible["DATE"].min()), "to", int(eligible["DATE"].max()))
print("Panel saved:", PANEL_OUT)
print("Yearly audit saved:", AUDIT_YEAR_OUT)
print("WARNING: return_proxy comes from raw mom1m and is potentially winsorized.")
print("WARNING: not equivalent to CRSP total/excess returns or delisting-adjusted returns.")