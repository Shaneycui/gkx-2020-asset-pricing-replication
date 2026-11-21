# GKX (2020): Three-Feature Replication in Python

I built a small replication of Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, to understand their prediction workflow. Instead of using the full predictor set, I focus on three stock characteristics: `mom1m` (short-term reversal), `bm` (book-to-market), and `mvel1` (size).

**Start here:** [Read the step-by-step walkthrough with Python examples](REPLICATION_WALKTHROUGH.md).

## Results, 1987–2016

| Model | Pooled OOS R² |
|---|---:|
| Elastic Net, APG | **+0.249536%** |
| Elastic Net, coordinate descent | +0.249521% |
| Random Forest, 50 trees | **+0.385939%** |

Both methods were compared on **2,489,185 matched stock-month forecasts**. A cross-sectionally aggregated modified-DM-style test gives **z = 2.0860, two-sided p = 0.036975** (Newey–West HAC with 12 lags). The result favors RF at the *unadjusted* 5% level.

**Scope:** The target is a next-month `mom1m` **return proxy**, not the original paper's CRSP excess return; RF tuning is simplified. This is a **three-characteristic methodological replication**, not a replication of GKX's complete empirical results.

## Documentation

- [Code-interleaved walkthrough](REPLICATION_WALKTHROUGH.md) — methods, short code excerpts, results, and decisions.
- [Full academic report](REPLICATION_REPORT.md) — detailed methodology and statistical analysis.
- [Reproducibility guide](REPRODUCIBILITY.md) — environment, inputs, file availability, and checks.
- [Research log](RESEARCH_PROGRESS.md) and [data provenance](DATA_SOURCES.md) — problems we encountered and how we handled them.

For Windows/PyCharm:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `data/` directory is Git-ignored. Several scripts used locally have not been committed yet; consult the reproducibility checklist before attempting an end-to-end rerun.

**Citation:** Gu, S., Kelly, B., and Xiu, D. (2020), *Review of Financial Studies* 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009
