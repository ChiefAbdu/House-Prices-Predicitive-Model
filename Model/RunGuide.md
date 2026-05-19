# ML CEP — Phase 2 (Model Implementation & Evaluation)

**Students:** Abdullah Nadeem (2023-EE-126), Shah Eman (2023-EE-178)

This project trains three models on the California housing regression task:

| Role | Model | Implementation |
|------|--------|----------------|
| Baseline | Linear regression | From scratch (`cep_phase2/linear_regression.py`) |
| Main | Random forest | From scratch (`cep_phase2/decision_tree.py`, `cep_phase2/random_forest.py`) |
| Flexible | Gradient boosting | `scikit-learn` `GradientBoostingRegressor` |

---

## Environment (tested)

| Item | Version / detail |
|------|-------------------|
| **OS** | Windows 11 Pro (build `22631`) |
| **Python** | `3.13.6` (`py -3 --version`) |
| **NumPy** | `2.4.4` |
| **scikit-learn** | `1.8.0` |
| **matplotlib** | `>=3.7` (install via `requirements.txt`) |

Other Python versions (e.g. 3.10–3.12) should work if dependencies install cleanly. The `requirements.txt` file lists **minimum** versions; the table above records what this repo was **verified against**.

---

## Setup

From the project root folder (`ML CEP`):

```powershell
py -3 -m pip install -r requirements.txt
```

---

## Run Phase 2

**Option A — from project root (recommended):**

```powershell
py -3 run_phase2.py
```

**Option B — module form:**

```powershell
py -3 -m cep_phase2.run_phase2
```

### Optional CLI flags

| Flag | Meaning |
|------|---------|
| `--n-splits N` | Number of KFold splits (default `5`). |
| `--seed S` | Random seed for CV, splits, and ensemble RNGs (default `42`). |
| `--skip-holdout` | K-fold CV only (no 60/20/20 train/validation/test block). |
| `--no-plots` | Do not save matplotlib PNG figures. |
| `--plots-dir DIR` | Where to write figures (default `phase2_figures`). |

Example:

```powershell
py -3 run_phase2.py --n-splits 5 --seed 42
```

---

## Output

The script prints:

1. **Dataset and split confirmation** (sample count, \(K\)-fold settings, holdout sizes when enabled).
2. Per-model **training/evaluation** progress lines (`[1/3]` ... `[3/3]`).
3. **K-fold CV** metrics on validation folds: RMSE, MAE, R² — mean +/- std, fold variance, per-fold values.
4. **In-sample train metrics per CV fold** (same fit as validation) plus mean **val - train** RMSE gap for overfitting discussion.
5. A **comparison table** (ASCII) of CV validation vs training RMSE.
6. Unless `--skip-holdout`: **holdout** train / validation / test metrics (one fit per model) and a second comparison table.
7. Paths to saved figures.
**Figures (default on):** PNGs under `--plots-dir` (default `phase2_figures/`):

- `cv_metrics_bars.png` — CV mean RMSE, MAE, and R^2 with error bars (fold std).
- `cv_rmse_per_fold.png` — validation RMSE for each fold and model (same splits).
- `cv_train_vs_val_rmse.png` — mean CV train RMSE vs mean CV validation RMSE by model.
- `holdout_pred_vs_actual.png` — predicted vs actual on the holdout **test** set (three panels; skipped if `--skip-holdout` or `--no-plots`).
- `holdout_residuals.png` — residual vs predicted on holdout test (three panels).



### Common issues

- **`ModuleNotFoundError: matplotlib`** — Run `py -3 -m pip install -r requirements.txt`, or use **`--no-plots`** so matplotlib is not imported.
- **PDF build fails on missing figures** — Run `py -3 run_phase2.py` first so `phase2_figures/` exists.

---

## Attribution in code

Each module, class, function, and the root launcher includes inline comments of the form:

`# Responsible: <Name> | Registration No.: <roll no.>` (e.g. 2023-EE-126, 2023-EE-178)

---

## Repository layout

```
ML CEP/
  README.md
  requirements.txt
  run_phase2.py
  cep_phase2/
    __init__.py
    linear_regression.py     # baseline (scratch)
    decision_tree.py          # tree for main model (scratch)
    random_forest.py          # main model (scratch)
    evaluation.py             # CV + holdout + reporting helpers
    plots.py                  # matplotlib figures for the report
    run_phase2.py             # CLI entry / experiment orchestration
```
