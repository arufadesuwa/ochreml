"""
OchreML Multiclass Meta-Estimators (＾▽＾)
Implements OneVsRestClassifier with full OchreML and Scikit-Learn estimator support (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from .pipeline import _BaseMetaEstimator, _check_is_fitted, clone_estimator


def _sigmoid(x: Any) -> Any:
    x_arr = np.asarray(x, dtype=np.float64)
    clipped = np.clip(x_arr, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class OneVsRestClassifier(_BaseMetaEstimator):
    """
    One-vs-the-rest (OvR) multiclass strategy (o´∀｀o).

    Fits one binary classifier per class against all other classes.

    Parameters
    ----------
    estimator : object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.
    n_jobs : int or None, default=None
        Kept for Scikit-Learn API compatibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    estimators_ : list of estimators
        Estimators used for predictions.
    classes_ : numpy.ndarray
        Array of unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during `fit`.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> "OneVsRestClassifier":
        """Fit underlying binary estimators (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got {X_arr.ndim}D ( >_< )")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples ( >_< )")

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = int(X_arr.shape[1])
        self.estimators_: List[Any] = []

        if self.n_classes_ == 2:
            est = clone_estimator(self.estimator)
            y_bin = (y_arr == self.classes_[1]).astype(np.float64)
            est.fit(X_arr, y_bin)
            self.estimators_.append(est)
        else:
            for cls in self.classes_:
                est = clone_estimator(self.estimator)
                y_bin = (y_arr == cls).astype(np.float64)
                est.fit(X_arr, y_bin)
                self.estimators_.append(est)

        return self

    def _binary_scores(self, est: Any, X_arr: np.ndarray) -> np.ndarray:
        if hasattr(est, "decision_function"):
            df = np.asarray(est.decision_function(X_arr), dtype=np.float64)
            if df.ndim == 2:
                return df[:, -1]
            return df.ravel()
        if hasattr(est, "predict_proba"):
            proba = np.asarray(est.predict_proba(X_arr), dtype=np.float64)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        raise AttributeError(
            f"Base estimator {type(est).__name__} must implement decision_function or predict_proba ( >_< )"
        )

    def _binary_positive_proba(self, est: Any, X_arr: np.ndarray) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            proba = np.asarray(est.predict_proba(X_arr), dtype=np.float64)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        if hasattr(est, "decision_function"):
            df = np.asarray(est.decision_function(X_arr), dtype=np.float64)
            if df.ndim == 2:
                df = df[:, -1]
            return _sigmoid(df.ravel())
        raise AttributeError(
            f"Base estimator {type(est).__name__} must implement predict_proba or decision_function ( >_< )"
        )

    def decision_function(self, X: Any) -> np.ndarray:
        """Decision function for the OneVsRestClassifier (＾▽＾)."""
        _check_is_fitted(self, ["estimators_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        if self.n_classes_ == 2:
            return self._binary_scores(self.estimators_[0], X_arr)
        cols = [self._binary_scores(est, X_arr) for est in self.estimators_]
        return np.column_stack(cols)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Probability estimates for all classes (*≧ω≦*)."""
        _check_is_fitted(self, ["estimators_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        if self.n_classes_ == 2:
            p1 = self._binary_positive_proba(self.estimators_[0], X_arr)
            p1 = np.clip(p1, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        probs = np.column_stack(
            [self._binary_positive_proba(est, X_arr) for est in self.estimators_]
        )
        row_sums = probs.sum(axis=1, keepdims=True)
        safe_probs = np.where(
            row_sums > 0.0,
            probs / np.where(row_sums > 0.0, row_sums, 1.0),
            1.0 / max(1, self.n_classes_),
        )
        return safe_probs

    def predict(self, X: Any) -> np.ndarray:
        """Predict multi-class targets using underlying estimators (o´∀｀o)."""
        _check_is_fitted(self, ["estimators_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        if self.n_classes_ == 1:
            return np.full(X_arr.shape[0], self.classes_[0])
        if self.n_classes_ == 2:
            if hasattr(self.estimators_[0], "decision_function"):
                scores = self._binary_scores(self.estimators_[0], X_arr)
                return self.classes_[(scores > 0.0).astype(int)]
            proba = self.predict_proba(X_arr)
            return self.classes_[np.argmax(proba, axis=1)]

        if hasattr(self.estimators_[0], "decision_function"):
            scores = self.decision_function(X_arr)
            return self.classes_[np.argmax(scores, axis=1)]
        proba = self.predict_proba(X_arr)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> float:
        """Return the mean accuracy on the given test data and labels (＾▽＾)."""
        preds = self.predict(X)
        y_arr = np.asarray(y)
        matches = (preds == y_arr).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            return float(np.sum(sw * matches) / np.sum(sw))
        return float(np.mean(matches))

    def __repr__(self) -> str:
        return f"OneVsRestClassifier(estimator={self.estimator!r}) (＾▽＾)"


__all__ = ["OneVsRestClassifier"]
