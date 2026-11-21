"""GKX-inspired Elastic Net, using Appendix B.1 Algorithm 1 (APG).

This is a three-characteristic public-data approximation, NOT an exact GKX
replication. See APG_NOTES.md for objective-scale assumptions and deviations.

The original coordinate-descent outputs are left UNTOUCHED. Start with
LAST_TEST_YEAR=1987; after checking, change it to 2016. Completed annual
results are skipped on subsequent runs with the same experiment configuration.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
PANEL_PATH = DATA_DIR / "gkx_panel_mom1m_public.csv"
OUTPUT_DIR = DATA_DIR / "elastic_net_apg_recursive"

FEATURES = ["mom1m_scaled", "bm_scaled", "mvel1_scaled"]
FIRST_TEST_YEAR = 1987
LAST_TEST_YEAR = 2016  # Once validated, change only this to 2016.
VALIDATION_YEARS = 12
LAMBDAS = np.logspace(-4, -1, 10)  # Project-chosen 10-point grid
RHO_GKX = 0.5  # GKX: L1 weight = 1-rho; L2 weight = rho
MAX_ITER = 20000
TOL = 1e-10  # Project-specific prox fixed-point convergence tolerance


def fit_elastic_net_apg(X, y, lam, rho=RHO_GKX, tol=TOL, max_iter=MAX_ITER):
    """Fit by the exact APG *update structure* in GKX Appendix B.1 Algorithm 1.

    Objective (explicit project loss normalization):
        1/(2N) * ||y - intercept - X@beta||^2
        + lambda*(1-rho)*||beta||_1 + (lambda*rho/2)*||beta||_2^2.

    Intercept is NOT penalized. We analytically profile it out by centering X,y.
    Gradient and prox are computed from a tiny p x p Gram matrix; no N-sized
    predictions are constructed inside the APG iteration.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be (N, p), y must be (N,)")
    if len(y) == 0 or not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Empty or nonfinite input")
    if not (lam > 0 and 0 <= rho <= 1):
        raise ValueError("Require lambda > 0, 0 <= rho <= 1")

    mean_x = X.mean(axis=0)
    mean_y = y.mean()
    Xc = X - mean_x
    yc = y - mean_y
    n, p = X.shape
    gram = (Xc.T @ Xc) / n
    cross = (Xc.T @ yc) / n
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    # For lambda > 0 and rho >= 0, a null Gram can be handled directly.
    if lipschitz <= 0:
        beta = np.zeros(p)
        return {"coef": beta, "intercept": float(mean_y), "iterations": 0,
                "converged": True, "prox_residual": 0.0}
    gamma = 1.0 / lipschitz

    def prox_gradient(theta):
        # Smooth loss gradient: G@theta - Xc.T@yc/N
        z = theta - gamma * (gram @ theta - cross)
        # GKX elastic-net prox: soft_threshold(z, lambda*(1-rho)*gamma)
        #                         / (1 + lambda*rho*gamma)
        threshold = lam * (1.0 - rho) * gamma
        return (np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
                / (1.0 + lam * rho * gamma))

    theta = np.zeros(p, dtype=np.float64)
    converged = False
    residual = float("inf")
    for m in range(max_iter):
        # GKX Algorithm 1 uses momentum m/(m+3) on the prox step.
        prox_theta = prox_gradient(theta)
        residual = float(np.max(np.abs(prox_theta - theta)))
        if residual <= tol * (1.0 + float(np.max(np.abs(theta)))):
            theta = prox_theta
            converged = True
            iterations = m + 1
            break
        theta_new = prox_theta + (m / (m + 3.0)) * (prox_theta - theta)
        theta = theta_new
    else:
        iterations = max_iter

    # Use the proximal fixed point (rather than extrapolated state) for the
    # reported coefficients. Recheck first-order fixed-point residual.
    beta = prox_gradient(theta)
    final_residual = float(np.max(np.abs(prox_gradient(beta) - beta)))
    if final_residual <= tol * (1.0 + float(np.max(np.abs(beta)))):
        converged = True
    intercept = float(mean_y - mean_x @ beta)
    return {"coef": beta, "intercept": intercept, "iterations": iterations,
            "converged": bool(converged), "prox_residual": final_residual}


def predict(model, X):
    return np.asarray(X) @ model["coef"] + model["intercept"]


def save_csv_atomic(df, path):
    temp = path.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "optimizer": "GKX Appendix B.1 APG Algorithm 1 (momentum m/(m+3))",
        "objective": "0.5*mean(squared_error) + lambda*(1-rho)*L1 + 0.5*lambda*rho*L2_squared",
        "intercept": "unpenalized, analytically profiled by centering",
        "features": FEATURES,
        "lambda_candidates": [float(x) for x in LAMBDAS],
        "rho_gkx": RHO_GKX,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "train_start_year": 1957,
        "first_test_year": FIRST_TEST_YEAR,
        "validation_years": VALIDATION_YEARS,
        # LAST_TEST_YEAR deliberately omitted to support continuation.
        "target": "return_proxy",
        "dataset": "gkx_panel_mom1m_public.csv",
    }
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Experiment configuration changed. Use a NEW output directory.")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    panel = pd.read_csv(PANEL_PATH, usecols=["permno", "DATE", "return_proxy"] + FEATURES)
    panel["year"] = panel["DATE"] // 10000
    panel = panel.sort_values(["DATE", "permno"]).reset_index(drop=True)
    assert not panel.duplicated(["permno", "DATE"]).any()
    assert panel[FEATURES + ["return_proxy"]].notna().all().all()
    X = panel[FEATURES].to_numpy(dtype=np.float64)
    y = panel["return_proxy"].to_numpy(dtype=np.float64)
    years = panel["year"].to_numpy()
    assert years.min() == 1957 and years.max() >= LAST_TEST_YEAR
    yearly_path = OUTPUT_DIR / "elastic_net_apg_yearly.csv"
    if yearly_path.exists():
        completed = {int(r["test_year"]): r.to_dict()
                     for _, r in pd.read_csv(yearly_path).iterrows()}
    else:
        completed = {}

    print("=== RECURSIVE ELASTIC NET: GKX-STYLE APG ===", flush=True)
    print("Rows:", len(panel), "Test years:", FIRST_TEST_YEAR, "-", LAST_TEST_YEAR, flush=True)
    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        prediction_path = OUTPUT_DIR / f"elastic_net_apg_predictions_{year}.csv"
        validation_path = OUTPUT_DIR / f"elastic_net_apg_validation_{year}.csv"
        if year in completed and prediction_path.exists() and validation_path.exists():
            print(f"{year}: already completed, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        val_start = np.searchsorted(years, year - VALIDATION_YEARS)
        test_start = np.searchsorted(years, year)
        test_end = np.searchsorted(years, year + 1)
        X_train, y_train = X[:val_start], y[:val_start]
        X_val, y_val = X[val_start:test_start], y[val_start:test_start]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        assert len(X_train) and len(X_val) and len(X_test)

        candidates = []
        for lam in LAMBDAS:
            model = fit_elastic_net_apg(X_train, y_train, float(lam))
            if not model["converged"]:
                raise RuntimeError(f"No APG convergence: year={year}, lambda={lam}, {model}")
            val_mse = float(np.mean((y_val - predict(model, X_val)) ** 2))
            candidates.append({"test_year": year, "lambda_gkx": float(lam),
                               "validation_mse": val_mse, "iterations": model["iterations"],
                               "prox_residual": model["prox_residual"]})
        val_df = pd.DataFrame(candidates)
        selected = val_df.loc[val_df["validation_mse"].idxmin()]
        lam_best = float(selected["lambda_gkx"])

        final = fit_elastic_net_apg(X[:test_start], y[:test_start], lam_best)
        if not final["converged"]:
            raise RuntimeError(f"Final APG did not converge for year {year}")
        test_pred = predict(final, X_test)
        test_mse = float(np.mean((y_test - test_pred) ** 2))
        zero_mse = float(np.mean(y_test ** 2))
        oos_r2 = 1.0 - test_mse / zero_mse
        elapsed = time.perf_counter() - t0

        # Individual-year files support resuming without overwriting the original experiment.
        out = panel.iloc[test_start:test_end][["permno", "DATE", "return_proxy"]].copy()
        out["elastic_net_apg_prediction"] = test_pred
        save_csv_atomic(val_df, validation_path)
        save_csv_atomic(out, prediction_path)
        completed[year] = {
            "test_year": year,
            "train_end_year": year - VALIDATION_YEARS - 1,
            "validation_start_year": year - VALIDATION_YEARS,
            "validation_end_year": year - 1,
            "train_rows": len(X_train), "validation_rows": len(X_val),
            "final_train_rows": test_start, "test_rows": len(y_test),
            "best_lambda": lam_best, "rho_gkx": RHO_GKX,
            "intercept": final["intercept"],
            **{f"coef_{feature}": float(coef)
               for feature, coef in zip(FEATURES, final["coef"])},
            "validation_mse": float(selected["validation_mse"]),
            "test_mse": test_mse, "zero_mse": zero_mse, "oos_r2": oos_r2,
            "iterations": final["iterations"],
            "prox_residual": final["prox_residual"],
            "elapsed_seconds": elapsed,
        }
        save_csv_atomic(pd.DataFrame([completed[k] for k in sorted(completed)]), yearly_path)
        print(f"{year} | lambda={lam_best:.6g} | R2={100*oos_r2:+.6f}% "
              f"| APG iters={final['iterations']} | secs={elapsed:.1f}", flush=True)

    # Aggregate the exact combined squared loss; do NOT average yearly R^2.
    total_squared_error = 0.0
    total_squared_actual = 0.0
    total_predictions = 0
    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        p = OUTPUT_DIR / f"elastic_net_apg_predictions_{year}.csv"
        if not p.exists():
            raise RuntimeError(f"Missing {p}")
        df = pd.read_csv(p)
        actual = df["return_proxy"].to_numpy()
        predicted = df["elastic_net_apg_prediction"].to_numpy()
        total_squared_error += float(np.sum((actual - predicted) ** 2))
        total_squared_actual += float(np.sum(actual ** 2))
        total_predictions += len(df)
    print("=== FINISHED ===")
    print("Predictions:", total_predictions)
    print("Overall OOS R2 (%):", 100 * (1 - total_squared_error / total_squared_actual))
    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()