# Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
# Responsible: Shah Eman      | Registration No.: 2023-EE-178
# (Shared evaluation utilities for Phase 2.)

"""Cross-validation and holdout evaluation (metrics via sklearn.metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


@dataclass
class FoldMetrics:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178

    rmse: float
    mae: float
    r2: float


@dataclass
class HoldoutResult:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178

    name: str
    train: FoldMetrics
    val: FoldMetrics
    test: FoldMetrics


@dataclass
class CvSummary:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178

    name: str
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    fold_details: list[FoldMetrics]


@dataclass
class CvTrainValSummary:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178

    """Validation-fold metrics (generalization) plus metrics on the training portion of each fold (in-sample)."""

    name: str
    val: CvSummary
    train_fold_details: list[FoldMetrics]
    train_rmse_mean: float
    train_rmse_std: float
    train_mae_mean: float
    train_mae_std: float
    train_r2_mean: float
    train_r2_std: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return FoldMetrics(
        rmse=_rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def cross_validate_regressor(
    name: str,
    build_fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> CvSummary:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """build_fit_predict(X_train, y_train, X_val) -> y_val_pred"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[FoldMetrics] = []
    for train_idx, val_idx in kf.split(x):
        x_tr, x_va = x[train_idx], x[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        y_hat = build_fit_predict(x_tr, y_tr, x_va)
        folds.append(evaluate_fold(y_va, y_hat))

    def stats(vals: Sequence[float]) -> tuple[float, float]:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        # Responsible: Shah Eman      | Registration No.: 2023-EE-178
        a = np.asarray(vals, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a, ddof=0))

    rm_m, rm_s = stats([f.rmse for f in folds])
    ma_m, ma_s = stats([f.mae for f in folds])
    r2_m, r2_s = stats([f.r2 for f in folds])
    return CvSummary(name, rm_m, rm_s, ma_m, ma_s, r2_m, r2_s, folds)


def cross_validate_train_val(
    name: str,
    build_fit_predict_train_val: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> CvTrainValSummary:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """
    build_fit_predict_train_val(X_train, y_train, X_val) -> (y_val_pred, y_train_pred)
    One fit per fold; train predictions are in-sample on the fold's training indices.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_val: list[FoldMetrics] = []
    fold_train: list[FoldMetrics] = []
    for train_idx, val_idx in kf.split(x):
        x_tr, x_va = x[train_idx], x[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        y_hat_va, y_hat_tr = build_fit_predict_train_val(x_tr, y_tr, x_va)
        fold_val.append(evaluate_fold(y_va, y_hat_va))
        fold_train.append(evaluate_fold(y_tr, y_hat_tr))

    def stats2(vals: Sequence[float]) -> tuple[float, float]:
        a = np.asarray(vals, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a, ddof=0))

    rm_m, rm_s = stats2([f.rmse for f in fold_val])
    ma_m, ma_s = stats2([f.mae for f in fold_val])
    r2_m, r2_s = stats2([f.r2 for f in fold_val])
    val_sum = CvSummary(name, rm_m, rm_s, ma_m, ma_s, r2_m, r2_s, fold_val)

    tr_rm_m, tr_rm_s = stats2([f.rmse for f in fold_train])
    tr_ma_m, tr_ma_s = stats2([f.mae for f in fold_train])
    tr_r2_m, tr_r2_s = stats2([f.r2 for f in fold_train])
    return CvTrainValSummary(
        name=name,
        val=val_sum,
        train_fold_details=fold_train,
        train_rmse_mean=tr_rm_m,
        train_rmse_std=tr_rm_s,
        train_mae_mean=tr_ma_m,
        train_mae_std=tr_ma_s,
        train_r2_mean=tr_r2_m,
        train_r2_std=tr_r2_s,
    )


def split_train_val_test_holdout(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size_of_train: float = 0.25,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """
    Same 60/20/20 style split as evaluate_holdout_train_val_test_once.

    Returns (x_tr, x_va, x_te, y_tr, y_va, y_te).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    x_tr_full, x_te, y_tr_full, y_te = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr_full, y_tr_full, test_size=val_size_of_train, random_state=random_state
    )
    return x_tr, x_va, x_te, y_tr, y_va, y_te


def evaluate_holdout_train_val_test_once(
    name: str,
    fit_once: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size_of_train: float = 0.25,
    random_state: int | None = 42,
) -> HoldoutResult:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    """
    One fit per model on the training split, then predictions on val, test, and train.
    fit_once(X_train, y_train, X_val, X_test) -> (pred_val, pred_test, pred_train)
    """
    x_tr, x_va, x_te, y_tr, y_va, y_te = split_train_val_test_holdout(
        x, y, test_size=test_size, val_size_of_train=val_size_of_train, random_state=random_state
    )
    y_hat_va, y_hat_te, y_hat_tr = fit_once(x_tr, y_tr, x_va, x_te)
    return HoldoutResult(
        name=name,
        train=evaluate_fold(y_tr, y_hat_tr),
        val=evaluate_fold(y_va, y_hat_va),
        test=evaluate_fold(y_te, y_hat_te),
    )


def format_cv_train_val(s: CvTrainValSummary) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    val_lines = format_summary(s.val).split("\n")[1:]
    lines = [
        f"=== {s.name} ===",
        "--- Validation fold (out-of-sample within CV) ---",
        *val_lines,
        "",
        "--- Training portion of each fold (in-sample; overfitting diagnostic) ---",
        f"RMSE (train): {s.train_rmse_mean:.4f} +/- {s.train_rmse_std:.4f}",
        f"MAE  (train): {s.train_mae_mean:.4f} +/- {s.train_mae_std:.4f}",
        f"R^2  (train): {s.train_r2_mean:.4f} +/- {s.train_r2_std:.4f}",
        "Per-fold train RMSE: " + ", ".join(f"{f.rmse:.4f}" for f in s.train_fold_details),
        "",
        f"Mean (val RMSE - train RMSE): {s.val.rmse_mean - s.train_rmse_mean:+.4f}",
        "  (Usually positive: validation harder than in-sample training predictions.)",
    ]
    return "\n".join(lines)


def format_summary(s: CvSummary) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    lines = [
        f"=== {s.name} ===",
        f"RMSE: {s.rmse_mean:.4f} +/- {s.rmse_std:.4f}  (fold variance: {s.rmse_std ** 2:.6f})",
        f"MAE:  {s.mae_mean:.4f} +/- {s.mae_std:.4f}  (fold variance: {s.mae_std ** 2:.6f})",
        f"R^2:  {s.r2_mean:.4f} +/- {s.r2_std:.4f}  (fold variance: {s.r2_std ** 2:.6f})",
        "Per-fold RMSE: " + ", ".join(f"{f.rmse:.4f}" for f in s.fold_details),
        "Per-fold MAE:  " + ", ".join(f"{f.mae:.4f}" for f in s.fold_details),
        "Per-fold R^2:  " + ", ".join(f"{f.r2:.4f}" for f in s.fold_details),
    ]
    return "\n".join(lines)


def format_holdout(h: HoldoutResult) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    return "\n".join(
        [
            f"=== {h.name} (holdout) ===",
            f"Train      - RMSE: {h.train.rmse:.4f}, MAE: {h.train.mae:.4f}, R^2: {h.train.r2:.4f}  (single fit; in-sample)",
            f"Validation - RMSE: {h.val.rmse:.4f}, MAE: {h.val.mae:.4f}, R^2: {h.val.r2:.4f}",
            f"Test       - RMSE: {h.test.rmse:.4f}, MAE: {h.test.mae:.4f}, R^2: {h.test.r2:.4f}",
            f"Gap (val RMSE - train RMSE): {h.val.rmse - h.train.rmse:+.4f}",
        ]
    )


def format_cv_comparison_table(summaries: Sequence[CvTrainValSummary]) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    header = f"{'Model':<38} {'CV val RMSE':>14} {'+/-':>8} {'CV train RMSE':>16} {'val-train':>10}"
    sep = "-" * len(header)
    rows = [header, sep]
    for s in summaries:
        v = s.val
        short = v.name.replace("Baseline: ", "").replace("Main: ", "").replace("Flexible: ", "")[:36]
        gap = v.rmse_mean - s.train_rmse_mean
        rows.append(
            f"{short:<38} {v.rmse_mean:14.4f} {v.rmse_std:8.4f} {s.train_rmse_mean:16.4f} {gap:+8.4f}"
        )
    return "\n".join(rows)


def format_holdout_comparison_table(holdouts: Sequence[HoldoutResult]) -> str:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
    # Responsible: Shah Eman      | Registration No.: 2023-EE-178
    header = f"{'Model':<38} {'train RMSE':>12} {'val RMSE':>12} {'test RMSE':>12} {'val-train':>12}"
    sep = "-" * len(header)
    rows = [header, sep]
    for h in holdouts:
        short = h.name.replace("Baseline: ", "").replace("Main: ", "").replace("Flexible: ", "")[:36]
        g = h.val.rmse - h.train.rmse
        rows.append(
            f"{short:<38} {h.train.rmse:12.4f} {h.val.rmse:12.4f} {h.test.rmse:12.4f} {g:+12.4f}"
        )
    return "\n".join(rows)
