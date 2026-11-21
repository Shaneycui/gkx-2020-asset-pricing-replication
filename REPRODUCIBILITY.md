# Reproducing this project

The experiments were run on Windows with a Python virtual environment and PyCharm. **The main construction, APG, CD and RF experiment scripts are now in GitHub.** The large data and prediction outputs remain local, and a few auxiliary validation/CD-comparison scripts are still absent. A fully automated, independently tested clean-clone rerun has not yet been performed.

## Environment and inputs

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip freeze > local_environment_freeze.txt
Get-FileHash .\data\datashare.csv -Algorithm SHA256
```

The checked-in dependency ranges do **not** record the exact original package versions. Save the interpreter version, `pip freeze`, source download date, file hashes, and code commit for a closer reproduction.

Download original characteristics from the source documented in [DATA_SOURCES.md](DATA_SOURCES.md). Do not commit large raw panels, private or licensed financial data, or result files unnecessarily. The `data/` folder is ignored in Git.

## Local data layout

```text
data/
  datashare.csv
  gkx_features_3char_full.csv
  gkx_features_3char_imputed.csv
  gkx_panel_mom1m_public.csv
  elastic_net_recursive_yearly.csv
  elastic_net_recursive_predictions.csv
  elastic_net_apg_recursive/             # annual APG forecasts
  rf_recursive_50trees_leaf100/         # annual RF forecasts and MDI
  model_comparison/                     # CD vs RF
  elastic_net_solver_comparison/        # APG vs CD
  apg_rf_significance/modified_dm/      # APG vs RF significance
  rf_feature_importance_analysis/       # RF MDI summaries and charts
```

## GitHub code-availability audit (21 November 2026)

Verified on `main`:

| Stage | Committed scripts |
|---|---|
| Feature preprocessing | [`preprocess_features_stream.py`](preprocess_features_stream.py), [`impute_features.py`](impute_features.py) |
| Full-universe proxy panel | [`build_gkx_full_panel.py`](build_gkx_full_panel.py) |
| Elastic Net | [`train_elastic_net_recursive.py`](train_elastic_net_recursive.py) (CD), [`train_elastic_net_apg_recursive.py`](train_elastic_net_apg_recursive.py) (APG), [`compare_elastic_net_solvers.py`](compare_elastic_net_solvers.py) |
| Random Forest | [`train_random_forest_recursive.py`](train_random_forest_recursive.py), plus the original pilot scripts |
| Primary model comparison | [`compare_apg_rf_significance.py`](compare_apg_rf_significance.py), [`test_modified_dm_apg_rf.py`](test_modified_dm_apg_rf.py) |
| Feature importance | [`analyze_rf_feature_importance.py`](analyze_rf_feature_importance.py) |

The committed `build_stock_panel.py` builds a **different exploratory candidate-ticker panel**, not the main 4.08-million-row panel.

**Still not verified on `main`:** `validate_imputed_features.py`, `validate_mom1m_returns.py`, `split_gkx_panel.py`, `audit_gkx_features_labels.py`, `compare_models.py`, and `test_model_significance-v2.py` (historical CD-vs-RF comparison). The APG source references `APG_NOTES.md`, which is also not currently committed. The **primary** APG-versus-RF comparison can be run using the two committed comparison scripts once yearly predictions exist.

### Commands for the committed primary workflow

Run from the repository root, after downloading `data/datashare.csv`:

```powershell
.\.venv\Scripts\python.exe preprocess_features_stream.py
.\.venv\Scripts\python.exe impute_features.py
.\.venv\Scripts\python.exe build_gkx_full_panel.py
.\.venv\Scripts\python.exe train_elastic_net_recursive.py
.\.venv\Scripts\python.exe train_elastic_net_apg_recursive.py
.\.venv\Scripts\python.exe train_random_forest_recursive.py
.\.venv\Scripts\python.exe compare_elastic_net_solvers.py
.\.venv\Scripts\python.exe compare_apg_rf_significance.py
.\.venv\Scripts\python.exe test_modified_dm_apg_rf.py
.\.venv\Scripts\python.exe analyze_rf_feature_importance.py
```

**Important:** `preprocess_features_stream.py` intentionally refuses to overwrite its final or incomplete CSV. Remove or archive the prior outputs only when deliberately rebuilding. Other scripts may overwrite derived outputs; archive any experiment outputs before rerunning. Training RF on millions of rows can be computationally expensive. These instructions have been checked against the uploaded file paths but **have not been exercised as an end-to-end rerun on a clean clone**.

The execution order is **rank features → impute → construct panel → train CD/APG/RF → compare saved forecasts → modified DM → RF MDI**. See [the walkthrough](REPLICATION_WALKTHROUGH.md) for explanations.

**Caution:** `train_elastic_net_recursive.py` rewrites its original CD yearly/prediction CSVs. Back up local outputs before rerunning models. Summary and comparison scripts use saved forecasts and do not retrain.

## Expected checkpoints from completed local runs

| Check | Recorded result |
|---|---:|
| Source rows | 4,117,300 |
| Eligible modeling rows | 4,078,842 |
| Matched OOS forecasts per model | 2,489,185 |
| APG pooled OOS R² | +0.249536% |
| CD pooled OOS R² | +0.249521% |
| RF pooled OOS R² | +0.385939% |
| APG/RF modified DM, HAC(12) | z = +2.0860, p = 0.036975 |
| RF mean MDI: mom1m / mvel1 / bm | 0.674356 / 0.225308 / 0.100336 |

These are useful audit targets, not guarantees of bitwise reproducibility on a different machine.

## What a reader should know

The full sample uses next-consecutive-month raw `mom1m` as an outcome **proxy**, not actual CRSP excess returns. The seven-stock Yahoo check supports its alignment within that exploratory sample but does not certify all historical returns or delisting adjustments. Monthly median imputation includes a documented zero fallback for 66 entirely missing-`bm` months. RF uses 50 trees and three eligible split features, rather than the original paper's larger setup. The reported HAC p-value is unadjusted and tests the mean *monthly* loss differential; MDI is a training-time relative measure, not a causal economic explanation.
