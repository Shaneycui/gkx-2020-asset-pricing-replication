# Reproducing this project

The experiments were run on Windows with a Python virtual environment and PyCharm. **The GitHub repository is not yet sufficient for a fresh end-to-end rerun:** some working scripts and all large model data/output files still live in the local project.

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

## GitHub code-availability audit (20 November 2026)

Verified on `main`: `train_elastic_net_recursive.py`, `train_random_forest_recursive.py`, `train_random_forest_full_1987.py`, `train_random_forest_pilot.py`. The committed `build_stock_panel.py` is an *exploratory candidate-ticker panel*, not the final 4.08m-row panel constructor.

The following have been run or referenced **locally** but still need to be checked in from the actual project (avoid substituting guessed or rewritten source code):

| Script | Function |
|---|---|
| `preprocess_features_stream.py` | Monthly ranks and scaling |
| `impute_features.py`, `validate_imputed_features.py` | Monthly imputation and checks |
| `build_gkx_full_panel.py` | Full eligible panel with return proxy |
| `validate_mom1m_returns.py` | Independent Yahoo-based check |
| `train_elastic_net_apg_recursive.py` | APG 30-year experiment |
| `compare_elastic_net_solvers.py` | APG vs CD predictions |
| `compare_models.py`, `test_model_significance-v2.py` | CD vs RF comparison |
| `compare_apg_rf_significance.py`, `test_modified_dm_apg_rf.py` | Main APG vs RF inference |
| `analyze_rf_feature_importance.py` | RF MDI summaries and charts |
| `audit_gkx_features_labels.py` | Feature/target diagnostic |

Once those sources are available, the execution order is: **rank features → impute → build eligible panel → train Elastic Net and RF → compare predictions → modified DM test → MDI analysis**. See [the walkthrough](REPLICATION_WALKTHROUGH.md) for method explanations.

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
