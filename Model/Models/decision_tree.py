# Responsible: Abdullah Nadeem | Registration No.: 2023-EE-126
# (Regression tree building block for the Phase 1 main model — random forest.)

"""Regression decision tree (building block for random forest from scratch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class _TreeNode:
    # Responsible: Shah Eman | Registration No.: 2023-EE-178
    is_leaf: bool
    value: float | None = None
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


def _sse(sum_y: float, sum_y2: float, n: int) -> float:
    # Responsible: Shah Eman | Registration No.: 2023-EE-178
    if n <= 0:
        return 0.0
    return float(sum_y2 - (sum_y * sum_y) / n)


class DecisionTreeRegressorScratch:
    # Responsible: Shah Eman | Registration No.: 2023-EE-178

    """Axis-aligned CART-style regression tree using greedy MSE reduction."""

    def __init__(
        self,
        max_depth: int | None = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 10,
        max_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None
        self._rng: np.random.Generator | None = None
        self.n_features_in_: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> DecisionTreeRegressorScratch:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = x.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        if self.max_features is None:
            feat_count = self.n_features_in_
        else:
            feat_count = int(min(self.max_features, self.n_features_in_))
        self.root_ = self._build(
            x,
            y,
            depth=0,
            feat_pool_size=feat_count,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        if self.root_ is None:
            raise RuntimeError("Call fit before predict.")
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        idx = np.arange(n, dtype=np.int64)
        self._predict_bulk(self.root_, x, idx, out)
        return out

    def _predict_bulk(self, node: _TreeNode, x: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        """Assign predictions for rows `idx` using subtree rooted at `node`."""
        if idx.size == 0:
            return
        if node.is_leaf:
            assert node.value is not None
            out[idx] = node.value
            return
        assert node.feature is not None and node.threshold is not None
        col = x[idx, node.feature]
        left_mask = col <= node.threshold
        idx_l = idx[left_mask]
        idx_r = idx[~left_mask]
        self._predict_bulk(node.left, x, idx_l, out)  # type: ignore[arg-type]
        self._predict_bulk(node.right, x, idx_r, out)  # type: ignore[arg-type]

    def _build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: int,
        feat_pool_size: int,
    ) -> _TreeNode:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        n = x.shape[0]
        mean = float(np.mean(y)) if n > 0 else 0.0
        if n == 0:
            return _TreeNode(is_leaf=True, value=0.0)

        if self.max_depth is not None and depth >= self.max_depth:
            return _TreeNode(is_leaf=True, value=mean)
        if n < self.min_samples_split:
            return _TreeNode(is_leaf=True, value=mean)

        best_feat, best_thr, best_cost = self._best_split(x, y, feat_pool_size)
        if best_feat is None:
            return _TreeNode(is_leaf=True, value=mean)

        left_mask = x[:, best_feat] <= best_thr
        right_mask = ~left_mask
        if int(np.sum(left_mask)) < self.min_samples_leaf or int(np.sum(right_mask)) < self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=mean)

        left = self._build(x[left_mask], y[left_mask], depth + 1, feat_pool_size)
        right = self._build(x[right_mask], y[right_mask], depth + 1, feat_pool_size)
        return _TreeNode(
            is_leaf=False,
            feature=best_feat,
            threshold=float(best_thr),
            left=left,
            right=right,
        )

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feat_pool_size: int,
    ) -> tuple[int | None, float | None, float]:
        # Responsible: Shah Eman | Registration No.: 2023-EE-178
        n, p = x.shape
        assert self._rng is not None
        all_idx = np.arange(p)
        if feat_pool_size >= p:
            cand_feats: List[int] = list(range(p))
        else:
            cand_feats = list(self._rng.choice(all_idx, size=feat_pool_size, replace=False))

        sum_y_all = float(np.sum(y))
        sum_y2_all = float(np.sum(y * y))
        parent_sse = _sse(sum_y_all, sum_y2_all, n)

        best_cost = parent_sse
        best_feat: int | None = None
        best_thr: float | None = None

        for j in cand_feats:
            col = x[:, j]
            order = np.argsort(col, kind="mergesort")
            x_sorted = col[order]
            y_sorted = y[order]

            y_prefix = np.cumsum(y_sorted)
            y2_prefix = np.cumsum(y_sorted * y_sorted)

            for k in range(self.min_samples_leaf - 1, n - self.min_samples_leaf):
                if x_sorted[k] == x_sorted[k + 1]:
                    continue
                n_l = k + 1
                n_r = n - n_l
                sum_y_l = float(y_prefix[k])
                sum_y2_l = float(y2_prefix[k])
                sum_y_r = sum_y_all - sum_y_l
                sum_y2_r = sum_y2_all - sum_y2_l
                cost = _sse(sum_y_l, sum_y2_l, n_l) + _sse(sum_y_r, sum_y2_r, n_r)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    thr = 0.5 * (float(x_sorted[k]) + float(x_sorted[k + 1]))
                    best_feat = j
                    best_thr = thr

        if best_feat is None:
            return None, None, parent_sse
        return best_feat, best_thr, best_cost
