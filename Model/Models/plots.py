# Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
# Responsible: Shah Eman      | Registration No.: 2023-EE-178
# (Phase 2 report figures.)

"""Save matplotlib figures for model comparison (non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cep_phase2.evaluation import CvSummary, CvTrainValSummary


def model_short_label(full_name: str) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    if "Linear" in full_name:
        return "Linear (OLS)"
    if "Random Forest" in full_name:
        return "RF (scratch)"
    if "Gradient Boosting" in full_name:
        return "GBM (sklearn)"
    return full_name[:24]


def save_cv_summary_bar_charts(summaries: list[CvSummary], path: Path) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """Grouped bar-style subplot: CV mean RMSE, MAE, R^2 with error bars (fold std)."""
    labels = [model_short_label(s.name) for s in summaries]
    x = np.arange(len(labels))
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("K-fold CV: mean metric +/- std across folds", fontsize=12)

    rmses = [s.rmse_mean for s in summaries]
    rmse_e = [s.rmse_std for s in summaries]
    maes = [s.mae_mean for s in summaries]
    mae_e = [s.mae_std for s in summaries]
    r2s = [s.r2_mean for s in summaries]
    r2_e = [s.r2_std for s in summaries]

    axes[0].bar(x, rmses, yerr=rmse_e, capsize=4, color=["#4e79a7", "#f28e2b", "#59a14f"], ecolor="#333333")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_title("RMSE")

    axes[1].bar(x, maes, yerr=mae_e, capsize=4, color=["#4e79a7", "#f28e2b", "#59a14f"], ecolor="#333333")
    axes[1].set_ylabel("MAE (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("MAE")

    axes[2].bar(x, r2s, yerr=r2_e, capsize=4, color=["#4e79a7", "#f28e2b", "#59a14f"], ecolor="#333333")
    axes[2].set_ylabel("R^2 (higher is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_title("R^2")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cv_rmse_per_fold(summaries: list[CvSummary], path: Path) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """Line plot: RMSE in each fold for every model."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    folds = range(1, len(summaries[0].fold_details) + 1)
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    for i, s in enumerate(summaries):
        ys = [f.rmse for f in s.fold_details]
        ax.plot(folds, ys, marker="o", label=model_short_label(s.name), color=colors[i % len(colors)])
    ax.set_xticks(list(folds))
    ax.set_xlabel("Fold (validation)")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-fold validation RMSE (same KFold splits for all models)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_holdout_pred_vs_actual(
    y_test: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """One row of scatter plots: predicted vs actual on the holdout test set."""
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, constrained_layout=True)
    lo = float(min(y_test.min(), min(p.min() for _, p in predictions)))
    hi = float(max(y_test.max(), max(p.max() for _, p in predictions)))

    for ax, (title, y_hat) in zip(axes[0], predictions):
        y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
        ax.scatter(y_test, y_hat, alpha=0.35, s=12, color="#4e79a7")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
        ax.set_xlabel("Actual median value")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Holdout test: predicted vs actual", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cv_val_vs_train_rmse(summaries: list[CvTrainValSummary], path: Path) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """Bar chart comparing mean CV validation RMSE vs mean in-sample train RMSE per model."""
    labels = [model_short_label(s.name) for s in summaries]
    val_r = [s.val.rmse_mean for s in summaries]
    tr_r = [s.train_rmse_mean for s in summaries]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(x - w / 2, val_r, w, label="Validation RMSE (mean over folds)", color="#4e79a7")
    ax.bar(x + w / 2, tr_r, w, label="Train RMSE (in-sample, mean over folds)", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Overfitting view: train vs validation RMSE (K-fold)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_holdout_residuals(
    y_test: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """Residuals (actual - predicted) vs predicted on holdout test."""
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), squeeze=False, constrained_layout=True)
    for ax, (title, y_hat) in zip(axes[0], predictions):
        y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
        resid = y_test - y_hat
        ax.scatter(y_hat, resid, alpha=0.35, s=12, color="#59a14f")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - pred)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Holdout test: residual plots", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
