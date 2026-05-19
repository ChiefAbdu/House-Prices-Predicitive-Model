"""
Phase 2 entry point: California Housing Prices (regression).

Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
Responsible: Shah Eman      | Registration No.: 2023-EE-178
(Script wiring / orchestration — both members; replace blanks with official registration numbers.)

Baseline: LinearRegressionScratch (OLS + pinv; no sklearn estimator)
Main:      RandomForestRegressorScratch (no sklearn ensemble)
Flexible:  sklearn GradientBoostingRegressor (allowed comparison model)

Evaluation: K-fold CV (mean +/- std and fold variance for RMSE/MAE/R^2) plus an optional
60/20/20 train/validation/test holdout using the same preprocessing rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from cep_phase2.evaluation import (
    CvTrainValSummary,
    HoldoutResult,
    cross_validate_train_val,
    evaluate_holdout_train_val_test_once,
    format_cv_comparison_table,
    format_cv_train_val,
    format_holdout,
    format_holdout_comparison_table,
    split_train_val_test_holdout,
)
from cep_phase2.linear_regression import LinearRegressionScratch
from cep_phase2.random_forest import RandomForestRegressorScratch


def load_xy() -> tuple[np.ndarray, np.ndarray]:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    data = fetch_california_housing(as_frame=False)
    x = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.float64).ravel()
    return x, y


def hypothesis_guidance(
    summaries: list[CvTrainValSummary], holdouts: list[HoldoutResult] | None = None
) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """Template language tying metrics to Phase 1; edit for your final report."""
    by_rmse = sorted(summaries, key=lambda s: s.val.rmse_mean)
    best_cv = by_rmse[0].name
    lines = [
        "--- Hypothesis check (draft for your write-up) ---",
        "",
        "Phase 1 predicted: a strong linear trend (especially median income) should make a linear ",
        "baseline reasonable but incomplete; tree ensembles should better match piecewise coastal/inland ",
        "structure and nonlinearities; very flexible models may chase noise given the capped target ",
        "and outlying occupancy/room counts.",
        "",
        f"K-fold CV - lowest mean RMSE (validation folds): {best_cv}.",
    ]
    if holdouts:
        by_te = sorted(holdouts, key=lambda h: h.test.rmse)
        best_te = by_te[0].name
        lines.append(f"Holdout test set - lowest RMSE: {best_te}.")
    else:
        lines.append("Holdout test set - not run (use default mode without --skip-holdout).")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- If the scratch random forest clearly beats scratch OLS (lower RMSE / higher R^2 with similar ",
            "  fold spread), that supports needing nonlinear partitions/interactions beyond a single global plane.",
            "- If gradient boosting edges out the forest, that is consistent with sequential bias correction ",
            "  helping; if it wins on RMSE but with worse MAE or shuffles rank across folds, discuss stability.",
            "- If the forest and boosting are effectively tied, the 'more flexible GBM' hypothesis is only weakly ",
            "  supported versus a simpler bagged-tree main model.",
            "- Compare train vs validation RMSE: much lower train error suggests overfitting to fold noise.",
            "- Tie results to known artifacts: the $500k cap compresses high-end signal; outliers can dominate ",
            "  MAE in some folds or inflate variance.",
        ]
    )
    return "\n".join(lines)


def _print_methodology(n_splits: int, seed: int) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    print("Methodology (Phase 2 requirements)")
    print("----------------------------------")
    print(f"- Task: regression on fetch_california_housing (sklearn), predicting median house value.")
    print(f"- Baseline & main models: implemented in-repo (numpy only for learners); flexible model: sklearn GBM.")
    print(
        f"- Fair comparison: identical {n_splits}-fold shuffled KFold splits; "
        "each fold fits StandardScaler on train only, then transforms val rows."
    )
    print(
        "- Holdout (60/20/20 train/val/test): same scaler rule; models never fit on val/test rows; "
        f"test metrics use models trained only on the train split (seed={seed})."
    )
    print("- Metrics: RMSE, MAE, R^2 (sklearn.metrics). For CV, report mean +/- std across folds and variance.")
    print(
        "- Figures: unless --no-plots, saves PNGs under --plots-dir (CV bar charts, per-fold RMSE, "
        "train-vs-val RMSE, holdout predicted-vs-actual, holdout residuals)."
    )
    print()


def _print_dataset_and_splits(
    x: np.ndarray, y: np.ndarray, n_splits: int, rng_seed: int, skip_holdout: bool
) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    n, p = x.shape
    print("Dataset and split summary")
    print("-------------------------")
    print(f"Samples (n): {n}, features: {p}, target: median_house_value (California housing, sklearn).")
    print(f"Cross-validation: {n_splits}-fold, shuffle=True, random_state={rng_seed} (same folds for all models).")
    if not skip_holdout:
        _x_tr, x_va, _x_te, y_tr, y_va, y_te = split_train_val_test_holdout(
            x, y, random_state=rng_seed
        )
        print(
            f"Holdout sizes (60/20/20): training={len(y_tr)}, validation={len(y_va)}, test={len(y_te)} "
            f"(random_state={rng_seed}; matches sklearn train_test_split sizes for this n)."
        )
    print()


def main() -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    parser = argparse.ArgumentParser(description="Phase 2: train baseline, main, and flexible models.")
    parser.add_argument("--n-splits", type=int, default=5, help="KFold splits for cross-validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (CV, splits, ensemble RNGs).")
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Only run K-fold CV (assignment requires CV or train/val/test; default runs both).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving matplotlib PNG figures.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="phase2_figures",
        help="Directory for saved figures (created if missing).",
    )
    args = parser.parse_args()

    x_raw, y = load_xy()
    rng_seed = int(args.seed)
    n_splits = int(args.n_splits)

    def linear_tv(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_va_s = scaler.transform(x_va)
        m = LinearRegressionScratch(fit_intercept=True).fit(x_tr_s, y_tr)
        return m.predict(x_va_s), m.predict(x_tr_s)

    def rf_tv(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_va_s = scaler.transform(x_va)
        rf = RandomForestRegressorScratch(
            n_estimators=45,
            max_depth=12,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features=None,
            max_samples=10000,
            random_state=rng_seed,
        )
        rf.fit(x_tr_s, y_tr)
        return rf.predict(x_va_s), rf.predict(x_tr_s)

    def gbrt_tv(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_va_s = scaler.transform(x_va)
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=rng_seed,
        )
        gb.fit(x_tr_s, y_tr)
        return gb.predict(x_va_s), gb.predict(x_tr_s)

    def linear_once(
        x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        m = LinearRegressionScratch(fit_intercept=True).fit(x_tr_s, y_tr)
        return m.predict(scaler.transform(x_va)), m.predict(scaler.transform(x_te)), m.predict(x_tr_s)

    def rf_once(
        x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        rf = RandomForestRegressorScratch(
            n_estimators=45,
            max_depth=12,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features=None,
            max_samples=10000,
            random_state=rng_seed,
        )
        rf.fit(x_tr_s, y_tr)
        return rf.predict(scaler.transform(x_va)), rf.predict(scaler.transform(x_te)), rf.predict(x_tr_s)

    def gbrt_once(
        x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=rng_seed,
        )
        gb.fit(x_tr_s, y_tr)
        return gb.predict(scaler.transform(x_va)), gb.predict(scaler.transform(x_te)), gb.predict(x_tr_s)

    _print_methodology(n_splits=n_splits, seed=rng_seed)
    _print_dataset_and_splits(x_raw, y, n_splits, rng_seed, args.skip_holdout)

    print("Model training and evaluation (each model: separate estimator; same CV folds)")
    print("---------------------------------------------------------------------")

    specs: list[tuple[str, object, object]] = [
        ("Baseline: Linear Regression (scratch OLS)", linear_tv, linear_once),
        ("Main: Random Forest (scratch)", rf_tv, rf_once),
        ("Flexible: Gradient Boosting (sklearn)", gbrt_tv, gbrt_once),
    ]

    summaries_tv: list[CvTrainValSummary] = []
    for i, (name, tv, _once) in enumerate(specs, start=1):
        print(f"[{i}/3] Training + evaluating: {name} ...")
        summaries_tv.append(
            cross_validate_train_val(name, tv, x_raw, y, n_splits=n_splits, random_state=rng_seed)
        )
        print(f"      Done ({name}).")
    print()

    print("K-fold cross-validation (validation + training-fold diagnostics)")
    print("=================================================================")
    for s in summaries_tv:
        print(format_cv_train_val(s))
        print()

    print("Comparison table (CV): validation vs in-sample training RMSE")
    print("==============================================================")
    print(format_cv_comparison_table(summaries_tv))
    print()

    if not args.no_plots:
        from cep_phase2 import plots  # optional dependency: matplotlib

        out_dir = Path(args.plots_dir)
        val_only = [s.val for s in summaries_tv]
        plots.save_cv_summary_bar_charts(val_only, out_dir / "cv_metrics_bars.png")
        plots.save_cv_rmse_per_fold(val_only, out_dir / "cv_rmse_per_fold.png")
        plots.save_cv_val_vs_train_rmse(summaries_tv, out_dir / "cv_train_vs_val_rmse.png")
        print(f"Saved CV figures to: {out_dir.resolve()}")
        print("  - cv_metrics_bars.png")
        print("  - cv_rmse_per_fold.png")
        print("  - cv_train_vs_val_rmse.png")
        print()

    holdouts: list[HoldoutResult] = []
    if not args.skip_holdout:
        print("Train/validation/test holdout (60/20/20); one fit per model")
        print("============================================================")
        for name, _tv, once in specs:
            holdouts.append(
                evaluate_holdout_train_val_test_once(
                    name, once, x_raw, y, random_state=rng_seed
                )
            )
        for h in holdouts:
            print(format_holdout(h))
            print()
        print("Holdout comparison (RMSE)")
        print("------------------------")
        print(format_holdout_comparison_table(holdouts))
        print()

        if not args.no_plots:
            from cep_phase2 import plots

            x_tr, x_va, x_te, y_tr, _y_va, y_te = split_train_val_test_holdout(
                x_raw, y, random_state=rng_seed
            )
            pred_pairs = [
                ("Linear (OLS)", linear_once(x_tr, y_tr, x_va, x_te)[1]),
                ("RF (scratch)", rf_once(x_tr, y_tr, x_va, x_te)[1]),
                ("GBM (sklearn)", gbrt_once(x_tr, y_tr, x_va, x_te)[1]),
            ]
            out_dir = Path(args.plots_dir)
            plots.save_holdout_pred_vs_actual(y_te, pred_pairs, out_dir / "holdout_pred_vs_actual.png")
            plots.save_holdout_residuals(y_te, pred_pairs, out_dir / "holdout_residuals.png")
            print(f"Saved holdout figures: pred vs actual, residuals -> {out_dir.resolve()}")
            print()

    print(hypothesis_guidance(summaries_tv, holdouts if holdouts else None))


if __name__ == "__main__":
    main()
