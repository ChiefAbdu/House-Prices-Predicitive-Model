# Responsible: Shah Eman | Registration No.: 2023-EE-178
# (Baseline model — Phase 1.)

"""Ordinary least squares linear regression (no scikit-learn estimator)."""

from __future__ import annotations

import numpy as np


class LinearRegressionScratch:
    # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126

    """Multivariate OLS with intercept via pseudoinverse (stable with collinearity)."""

    def __init__(self, fit_intercept: bool = True) -> None:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegressionScratch:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.fit_intercept:
            x_design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
        else:
            x_design = x
        w = np.linalg.pinv(x_design) @ y
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
        if self.coef_ is None:
            raise RuntimeError("Call fit before predict.")
        x = np.asarray(x, dtype=np.float64)
        return x @ self.coef_ + float(self.intercept_)
