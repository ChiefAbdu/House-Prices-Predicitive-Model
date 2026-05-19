# Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
# (Phase 1 main model — random forest from scratch.)

"""Random forest regressor from scratch (bootstrap + bagged trees)."""

from __future__ import annotations

import numpy as np

from cep_phase2.decision_tree import DecisionTreeRegressorScratch


class RandomForestRegressorScratch:
    # Responsible: Shah Eman | Registration No.: 2023-EE-178

    def __init__(
        self,
        n_estimators: int = 75,
        max_depth: int | None = 14,
        min_samples_leaf: int = 8,
        min_samples_split: int = 16,
        max_features: int | None = None,
        max_samples: int | None = 10000,
        random_state: int | None = None,
    ) -> None:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressorScratch] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> RandomForestRegressorScratch:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = x.shape[0]
        p = x.shape[1]
        rng = np.random.default_rng(self.random_state)
        m_try = self.max_features
        if m_try is None:
            m_try = max(1, int(np.sqrt(p)))
        boot_size = n if self.max_samples is None else int(min(self.max_samples, n))

        self.estimators_ = []
        for t in range(self.n_estimators):
            idx = rng.integers(0, n, size=boot_size)
            xt = x[idx]
            yt = y[idx]
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=m_try,
                random_state=None if self.random_state is None else int(self.random_state) + t * 9973,
            )
            tree.fit(xt, yt)
            self.estimators_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        if not self.estimators_:
            raise RuntimeError("Call fit before predict.")
        x = np.asarray(x, dtype=np.float64)
        preds = np.stack([e.predict(x) for e in self.estimators_], axis=0)
        return np.mean(preds, axis=0)
