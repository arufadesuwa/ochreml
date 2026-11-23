"""
OchreML Meta-Ensemble Estimators (＾▽＾)
Implements BaggingClassifier, BaggingRegressor, VotingClassifier,
and StackingClassifier (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._meta_selection import _resolve_cv_splits
from .pipeline import (
    _BaseMetaEstimator,
    _check_is_fitted,
    _get_estimator_params,
    _set_estimator_params,
    clone_estimator,
)


def _resolve_rng(random_state: Optional[Union[int, np.random.RandomState]]) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def _resolve_subset_size(spec: Union[int, float], total: int, name: str) -> int:
    if isinstance(spec, (int, np.integer)):
        val = int(spec)
        if val < 1 or val > total:
            raise ValueError(f"{name}={val} must be in [1, {total}] ( >_< )")
        return val
    if isinstance(spec, float):
        if not (0.0 < spec <= 1.0):
            raise ValueError(f"{name}={spec} must be in (0.0, 1.0] ( >_< )")
        return max(1, int(round(spec * total)))
    raise TypeError(f"{name} must be int or float, got {type(spec)} ( >_< )")


class _BaseBagging(_BaseMetaEstimator):
    """Shared bootstrap aggregating logic for BaggingClassifier & BaggingRegressor."""

    def __init__(
        self,
        estimator: Optional[Any] = None,
        n_estimators: int = 10,
        *,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _default_estimator(self) -> Any:
        raise NotImplementedError

    def _fit_bagging(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got {X_arr.ndim}D ( >_< )")
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = int(n_features)

        n_sub_samples = _resolve_subset_size(self.max_samples, n_samples, "max_samples")
        n_sub_features = _resolve_subset_size(self.max_features, n_features, "max_features")

        base = self.estimator if self.estimator is not None else self._default_estimator()
        rng = _resolve_rng(self.random_state)

        self.estimators_: List[Any] = []
        self.estimators_features_: List[np.ndarray] = []

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(
                n_samples, size=n_sub_samples, replace=bool(self.bootstrap)
            )
            if n_sub_features == n_features and not self.bootstrap_features:
                feat_idx = np.arange(n_features, dtype=int)
            else:
                feat_idx = rng.choice(
                    n_features, size=n_sub_features, replace=bool(self.bootstrap_features)
                )

            est = clone_estimator(base)
            X_sub = X_arr[sample_idx][:, feat_idx]
            y_sub = y_arr[sample_idx]
            if sample_weight is not None:
                sw_sub = np.asarray(sample_weight, dtype=np.float64).ravel()[sample_idx]
                est.fit(X_sub, y_sub, sample_weight=sw_sub)
            else:
                est.fit(X_sub, y_sub)

            self.estimators_.append(est)
            self.estimators_features_.append(feat_idx)

        return X_arr, y_arr


class BaggingClassifier(_BaseBagging):
    """
    A Bagging classifier (o´∀｀o).

    Fits base classifiers each on random subsets of the original dataset and
    aggregates their individual predictions by averaging probabilities.

    Parameters
    ----------
    estimator : object or None, default=None
        Base estimator to fit on random subsets (defaults to DecisionTreeClassifier).
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number (or fraction) of samples to draw from X to train each base estimator.
    max_features : int or float, default=1.0
        The number (or fraction) of features to draw from X to train each base estimator.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement.
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
    random_state : int or None, default=None
        Controls the random resampling.
    """

    def _default_estimator(self) -> Any:
        from . import DecisionTreeClassifier

        return DecisionTreeClassifier()

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "BaggingClassifier":
        """Build a Bagging ensemble of classifiers from the training set (*≧ω≦*)."""
        _, y_arr = self._fit_bagging(X, y, sample_weight=sample_weight)
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for X (＾▽＾)."""
        _check_is_fitted(self, ["estimators_", "estimators_features_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        proba_sum = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            X_sub = X_arr[:, feat_idx]
            if hasattr(est, "predict_proba"):
                p = np.asarray(est.predict_proba(X_sub), dtype=np.float64)
                est_classes = getattr(est, "classes_", self.classes_)
                if len(est_classes) == self.n_classes_ and np.array_equal(est_classes, self.classes_):
                    proba_sum += p
                else:
                    for col_j, c in enumerate(est_classes):
                        if c in cls_to_idx:
                            proba_sum[:, cls_to_idx[c]] += p[:, col_j]
            else:
                preds = np.asarray(est.predict(X_sub))
                for row_i, pred in enumerate(preds):
                    if pred in cls_to_idx:
                        proba_sum[row_i, cls_to_idx[pred]] += 1.0

        return proba_sum / len(self.estimators_)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class for X (*≧ω≦*)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> float:
        """Return the mean accuracy on the given test data and labels (o´∀｀o)."""
        preds = self.predict(X)
        matches = (preds == np.asarray(y)).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            return float(np.sum(sw * matches) / np.sum(sw))
        return float(np.mean(matches))

    def __repr__(self) -> str:
        return (
            f"BaggingClassifier(estimator={self.estimator!r}, "
            f"n_estimators={self.n_estimators}) (＾▽＾)"
        )


class BaggingRegressor(_BaseBagging):
    """
    A Bagging regressor (＾▽＾).

    Fits base regressors each on random subsets of the original dataset and
    aggregates their individual predictions by averaging.

    Parameters
    ----------
    estimator : object or None, default=None
        Base estimator to fit on random subsets (defaults to DecisionTreeRegressor).
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number (or fraction) of samples to draw from X to train each base estimator.
    max_features : int or float, default=1.0
        The number (or fraction) of features to draw from X to train each base estimator.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement.
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
    random_state : int or None, default=None
        Controls the random resampling.
    """

    def _default_estimator(self) -> Any:
        from . import DecisionTreeRegressor

        return DecisionTreeRegressor()

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "BaggingRegressor":
        """Build a Bagging ensemble of regressors from the training set (*≧ω≦*)."""
        self._fit_bagging(X, np.asarray(y, dtype=np.float64), sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression target for X (＾▽＾)."""
        _check_is_fitted(self, ["estimators_", "estimators_features_"])
        X_arr = np.asarray(X, dtype=np.float64)
        preds = [
            np.asarray(est.predict(X_arr[:, feat_idx]), dtype=np.float64).ravel()
            for est, feat_idx in zip(self.estimators_, self.estimators_features_)
        ]
        return np.mean(np.column_stack(preds), axis=1)

    def score(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> float:
        """Return the coefficient of determination R^2 of the prediction (o´∀｀o)."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            y_mean = np.sum(sw * y_arr) / np.sum(sw)
            u = np.sum(sw * (y_arr - preds) ** 2)
            v = np.sum(sw * (y_arr - y_mean) ** 2)
        else:
            y_mean = np.mean(y_arr)
            u = np.sum((y_arr - preds) ** 2)
            v = np.sum((y_arr - y_mean) ** 2)
        return 1.0 if v <= 1e-12 and u <= 1e-12 else (
            0.0 if v <= 1e-12 else float(1.0 - u / v)
        )

    def __repr__(self) -> str:
        return (
            f"BaggingRegressor(estimator={self.estimator!r}, "
            f"n_estimators={self.n_estimators}) (o´∀｀o)"
        )


class _NamedEstimatorsMixin(_BaseMetaEstimator):
    """Shared get_params/set_params routing for named estimator lists."""

    estimators: List[Tuple[str, Any]]

    def _get_named_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, est in self.estimators:
            out[name] = est
            if deep and est is not None and est != "drop":
                for sub_k, sub_v in _get_estimator_params(est, deep=True).items():
                    out[f"{name}__{sub_k}"] = sub_v
        return out

    def _set_named_params(self, **params: Any) -> Any:
        est_map = {name: idx for idx, (name, _) in enumerate(self.estimators)}
        nested: Dict[str, Dict[str, Any]] = {}
        remaining: Dict[str, Any] = {}
        for key, val in params.items():
            if "__" in key:
                head, tail = key.split("__", 1)
                if head in est_map:
                    nested.setdefault(head, {})[tail] = val
                else:
                    remaining[key] = val
            elif key in est_map:
                idx = est_map[key]
                self.estimators[idx] = (key, val)
            else:
                remaining[key] = val

        for head, sub_params in nested.items():
            idx = est_map[head]
            est = self.estimators[idx][1]
            if est is None or est == "drop":
                raise ValueError(f"Cannot set parameters on '{est}' estimator '{head}' ( >_< )")
            _set_estimator_params(est, **sub_params)

        if remaining:
            super().set_params(**remaining)
        return self


class VotingClassifier(_NamedEstimatorsMixin):
    """
    Soft Voting / Majority Rule classifier for unfitted estimators (＾▽＾).

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the `fit` method on the `VotingClassifier` will fit clones
        of those original estimators.
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the sums of
        the predicted probabilities.
    weights : array-like of shape (n_classifiers,) or None, default=None
        Sequence of weights to weight the occurrences of predicted class labels
        or class probabilities.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, Any]],
        *,
        voting: str = "hard",
        weights: Optional[Sequence[float]] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
        verbose: bool = False,
    ):
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {voting!r} ( >_< )")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = super().get_params(deep=deep)
        if deep:
            out.update(self._get_named_params(deep=True))
        return out

    def set_params(self, **params: Any) -> "VotingClassifier":
        return self._set_named_params(**params)

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "VotingClassifier":
        """Fit the estimators (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = int(X_arr.shape[1])
        self.estimators_: List[Any] = []
        self.named_estimators_: Dict[str, Any] = {}

        for name, est in self.estimators:
            if est is None or est == "drop":
                self.named_estimators_[name] = est
                continue
            cloned = clone_estimator(est)
            if sample_weight is not None:
                cloned.fit(X_arr, y_arr, sample_weight=sample_weight)
            else:
                cloned.fit(X_arr, y_arr)
            self.estimators_.append(cloned)
            self.named_estimators_[name] = cloned

        if not self.estimators_:
            raise ValueError("VotingClassifier requires at least one non-dropped estimator ( >_< )")
        return self

    def _active_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.ones(len(self.estimators_), dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64).ravel()
        if len(w) != len(self.estimators_):
            raise ValueError(
                f"Number of weights ({len(w)}) != number of active estimators ({len(self.estimators_)}) ( >_< )"
            )
        return w

    def predict_proba(self, X: Any) -> np.ndarray:
        """Compute probabilities of possible outcomes for samples in X (＾▽＾)."""
        if self.voting != "soft":
            raise AttributeError(
                "predict_proba is not available when voting='hard' (´-ω-`)"
            )
        _check_is_fitted(self, ["estimators_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        weights = self._active_weights()
        avg = np.zeros((X_arr.shape[0], len(self.classes_)), dtype=np.float64)
        for est, w in zip(self.estimators_, weights):
            avg += w * np.asarray(est.predict_proba(X_arr), dtype=np.float64)
        return avg / np.sum(weights)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for X (*≧ω≦*)."""
        _check_is_fitted(self, ["estimators_", "classes_"])
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        X_arr = np.asarray(X, dtype=np.float64)
        weights = self._active_weights()
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((X_arr.shape[0], len(self.classes_)), dtype=np.float64)
        for est, w in zip(self.estimators_, weights):
            preds = np.asarray(est.predict(X_arr))
            for row_i, pred in enumerate(preds):
                if pred in cls_to_idx:
                    votes[row_i, cls_to_idx[pred]] += w
        return self.classes_[np.argmax(votes, axis=1)]

    def score(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> float:
        """Return the mean accuracy on the given test data and labels (o´∀｀o)."""
        preds = self.predict(X)
        matches = (preds == np.asarray(y)).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            return float(np.sum(sw * matches) / np.sum(sw))
        return float(np.mean(matches))

    def __repr__(self) -> str:
        return (
            f"VotingClassifier(estimators={self.estimators!r}, "
            f"voting={self.voting!r}) (＾▽＾)"
        )


def _stack_predict_block(
    est: Any, X: np.ndarray, stack_method: str, n_classes: int
) -> np.ndarray:
    method = stack_method
    if method == "auto":
        if hasattr(est, "predict_proba"):
            method = "predict_proba"
        elif hasattr(est, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"

    if method == "predict_proba":
        proba = np.asarray(est.predict_proba(X), dtype=np.float64)
        if n_classes == 2 and proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1:2]
        return proba if proba.ndim == 2 else proba.reshape(-1, 1)

    if method == "decision_function":
        df = np.asarray(est.decision_function(X), dtype=np.float64)
        return df.reshape(-1, 1) if df.ndim == 1 else df

    preds = np.asarray(est.predict(X), dtype=np.float64)
    return preds.reshape(-1, 1) if preds.ndim == 1 else preds


class StackingClassifier(_NamedEstimatorsMixin):
    """
    Stack of estimators with a final classifier (o´∀｀o).

    Stacked generalization consists in stacking the output of individual
    estimators and using a final classifier to compute the final prediction.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Base estimators which will be stacked together.
    final_estimator : estimator or None, default=None
        A classifier which will be used to combine the base estimators
        (defaults to OchreML LogisticRegression).
    cv : int, cross-validation generator, or iterable, default=5
        Determines the cross-validation splitting strategy used for
        out-of-fold predictions.
    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, default='auto'
        Methods called for each base estimator.
    passthrough : bool, default=False
        When True, original features X are concatenated with base estimator
        predictions as input to `final_estimator`.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, Any]],
        final_estimator: Optional[Any] = None,
        *,
        cv: Optional[Any] = 5,
        stack_method: str = "auto",
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
        verbose: int = 0,
    ):
        self.estimators = list(estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = super().get_params(deep=deep)
        if deep:
            out.update(self._get_named_params(deep=True))
        return out

    def set_params(self, **params: Any) -> "StackingClassifier":
        return self._set_named_params(**params)

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "StackingClassifier":
        """Fit the base estimators and final meta-estimator (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.n_features_in_ = int(X_arr.shape[1])

        self.estimators_ = []
        self.named_estimators_ = {}
        active_specs: List[Tuple[str, Any]] = []

        for name, est in self.estimators:
            if est is None or est == "drop":
                self.named_estimators_[name] = est
                continue
            active_specs.append((name, est))
            full_est = clone_estimator(est)
            if sample_weight is not None:
                full_est.fit(X_arr, y_arr, sample_weight=sample_weight)
            else:
                full_est.fit(X_arr, y_arr)
            self.estimators_.append(full_est)
            self.named_estimators_[name] = full_est

        if not active_specs:
            raise ValueError("StackingClassifier requires at least one non-dropped estimator ( >_< )")

        splits = _resolve_cv_splits(self.cv, X_arr, y_arr, estimator=active_specs[0][1])
        oof_blocks: List[np.ndarray] = []

        for _, base_est in active_specs:
            oof_arr: Optional[np.ndarray] = None
            for train_idx, test_idx in splits:
                fold_est = clone_estimator(base_est)
                fold_est.fit(X_arr[train_idx], y_arr[train_idx])
                block = _stack_predict_block(
                    fold_est, X_arr[test_idx], self.stack_method, n_classes
                )
                if oof_arr is None:
                    oof_arr = np.zeros((X_arr.shape[0], block.shape[1]), dtype=np.float64)
                oof_arr[test_idx] = block
            if oof_arr is not None:
                oof_blocks.append(oof_arr)

        meta_X = np.hstack(oof_blocks)
        if self.passthrough:
            meta_X = np.hstack([meta_X, X_arr])

        if self.final_estimator is None:
            from . import LogisticRegression

            final_est = LogisticRegression()
        else:
            final_est = clone_estimator(self.final_estimator)

        if sample_weight is not None:
            final_est.fit(meta_X, y_arr, sample_weight=sample_weight)
        else:
            final_est.fit(meta_X, y_arr)
        self.final_estimator_ = final_est
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return class labels or probabilities for X for each estimator (＾▽＾)."""
        _check_is_fitted(self, ["estimators_", "final_estimator_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)
        blocks = [
            _stack_predict_block(est, X_arr, self.stack_method, n_classes)
            for est in self.estimators_
        ]
        meta_X = np.hstack(blocks)
        if self.passthrough:
            meta_X = np.hstack([meta_X, X_arr])
        return meta_X

    def predict(self, X: Any) -> np.ndarray:
        """Predict target for X (*≧ω≦*)."""
        return np.asarray(self.final_estimator_.predict(self.transform(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for X (＾▽＾)."""
        return np.asarray(
            self.final_estimator_.predict_proba(self.transform(X)), dtype=np.float64
        )

    def decision_function(self, X: Any) -> np.ndarray:
        """Predict decision function for X."""
        return np.asarray(
            self.final_estimator_.decision_function(self.transform(X)), dtype=np.float64
        )

    def score(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> float:
        """Return the mean accuracy on the given test data and labels (o´∀｀o)."""
        preds = self.predict(X)
        matches = (preds == np.asarray(y)).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            return float(np.sum(sw * matches) / np.sum(sw))
        return float(np.mean(matches))

    def __repr__(self) -> str:
        return (
            f"StackingClassifier(estimators={self.estimators!r}, "
            f"final_estimator={self.final_estimator!r}) (o´∀｀o)"
        )


__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "VotingClassifier",
    "StackingClassifier",
]
