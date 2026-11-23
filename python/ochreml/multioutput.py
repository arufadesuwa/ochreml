"""
OchreML Multi-Output Meta-Estimators (＾▽＾)
Implements MultiOutputRegressor fitting one regressor per target (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from .pipeline import _BaseMetaEstimator, _check_is_fitted, clone_estimator


class MultiOutputRegressor(_BaseMetaEstimator):
    """
    Multi-target regression meta-estimator (o´∀｀o).

    Fits one cloned regressor per target column `y[:, j]`.

    Parameters
    ----------
    estimator : object
        A regressor object implementing `fit` and `predict`.
    n_jobs : int or None, default=None
        Kept for Scikit-Learn API compatibility.

    Attributes
    ----------
    estimators_ : list of estimators
        Estimators used for predictions, one per output target.
    n_outputs_ : int
        Number of output targets.
    n_features_in_ : int
        Number of features seen during `fit`.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        n_jobs: Optional[int] = None,
    ):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "MultiOutputRegressor":
        """Fit the model to data, one regressor per target column (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 2:
            raise ValueError(
                f"y must have 2 dimensions for MultiOutputRegressor, got {y_arr.ndim}D ( >_< )"
            )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples ( >_< )")

        self.n_features_in_ = int(X_arr.shape[1])
        self.n_outputs_ = int(y_arr.shape[1])
        self.estimators_: List[Any] = []

        for j in range(self.n_outputs_):
            est = clone_estimator(self.estimator)
            if sample_weight is not None:
                est.fit(X_arr, y_arr[:, j], sample_weight=sample_weight)
            else:
                est.fit(X_arr, y_arr[:, j])
            self.estimators_.append(est)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict multi-output targets using all fitted estimators (＾▽＾)."""
        _check_is_fitted(self, ["estimators_", "n_outputs_"])
        X_arr = np.asarray(X, dtype=np.float64)
        preds = [np.asarray(est.predict(X_arr), dtype=np.float64).ravel() for est in self.estimators_]
        return np.column_stack(preds)

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """Return the uniform-average R^2 score across all target outputs (o´∀｀o)."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()[:, None]
            y_mean = np.sum(sw * y_arr, axis=0) / np.sum(sw)
            u = np.sum(sw * (y_arr - preds) ** 2, axis=0)
            v = np.sum(sw * (y_arr - y_mean) ** 2, axis=0)
        else:
            y_mean = np.mean(y_arr, axis=0)
            u = np.sum((y_arr - preds) ** 2, axis=0)
            v = np.sum((y_arr - y_mean) ** 2, axis=0)
        r2 = np.where(v <= 1e-12, np.where(u <= 1e-12, 1.0, 0.0), 1.0 - u / v)
        return float(np.mean(r2))

    def __repr__(self) -> str:
        return f"MultiOutputRegressor(estimator={self.estimator!r}) (o´∀｀o)"


__all__ = ["MultiOutputRegressor"]
