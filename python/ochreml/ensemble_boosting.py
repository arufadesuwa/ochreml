"""
OchreML Boosting Ensembles Module (＾▽＾)
Provides Scikit-Learn-compatible GradientBoostingClassifier, GradientBoostingRegressor,
AdaBoostClassifier, and AdaBoostRegressor (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    AdaBoostClassifier as _RustAdaBoostClassifier,
    AdaBoostRegressor as _RustAdaBoostRegressor,
    GradientBoostingClassifier as _RustGradientBoostingClassifier,
    GradientBoostingRegressor as _RustGradientBoostingRegressor,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_sample_weight, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_2d(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if not is_buf:
        X_data = np.asarray(X_data, dtype=np.float64)
    return X_data


def _to_f64_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if not is_buf:
        y_data = np.asarray(y_data, dtype=np.float64)
    return y_data


def _to_opt_sample_weight(sample_weight: Optional[Any]) -> Optional[Any]:
    is_buf, sw_data = process_sample_weight(sample_weight)
    if sw_data is None:
        return None
    if not is_buf:
        sw_data = np.asarray(sw_data, dtype=np.float64)
    return sw_data


class _BaseBoosting:
    """Shared prediction, persistence, and device management mixin for Boosting estimators."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def predict(self, X: Any) -> Any:
        """
        Predict target values for samples in X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted values or class labels.
        """
        X_data = _to_f64_2d(X)
        raw = self._model.predict_buffer(X_data)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Any) -> float:
        """
        Return the evaluation score on the given test data and targets (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets or labels.

        Returns
        -------
        score : float
            Accuracy (classifiers) or R^2 score (regressors).
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        return float(self._model.score_buffer(X_data, y_data))

    def save(self, path: str) -> None:
        """Save fitted boosting model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load boosting model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> Any:
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    @property
    def feature_importances_(self) -> Optional[Any]:
        imp = self._model.feature_importances_
        return np.asarray(imp, dtype=np.float64) if imp is not None else None

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class _BaseBoostingClassifier(_BaseBoosting):
    """Shared classifier methods for GradientBoostingClassifier and AdaBoostClassifier."""

    def predict_proba(self, X: Any) -> Any:
        """
        Predict class probabilities for samples in X (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        p : numpy.ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def decision_function(self, X: Any) -> Any:
        """
        Compute the decision function of X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        score : numpy.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.decision_function_buffer(X_data)
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return np.asarray(c, dtype=np.float64) if c is not None else None


class GradientBoostingRegressor(_BaseBoosting):
    """
    Gradient Boosting for regression (＾▽＾).

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'huber'}, default='squared_error'
        Loss function to be optimized.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    n_estimators : int, default=100
        The number of boosting stages to perform.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base learners.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    alpha : float, default=0.9
        The alpha-quantile of the huber loss function.
    random_state : int or None, default=None
        Controls the random seed given to each Tree estimator at each boosting iteration.
    device : str or None, default=None
        Execution device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustGradientBoostingRegressor

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        alpha: float = 0.9,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        resolved_dev = _resolve_device(device)
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.alpha = alpha
        self.random_state = random_state
        self.device = resolved_dev
        self._model = _RustGradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            alpha=alpha,
            random_state=random_state,
            device=resolved_dev,
        )

    @classmethod
    def _from_rust(cls, rust_obj: Any, device: str) -> "GradientBoostingRegressor":
        obj = cls.__new__(cls)
        obj._model = rust_obj
        obj.loss = rust_obj.loss
        obj.learning_rate = rust_obj.learning_rate
        obj.n_estimators = rust_obj.n_estimators
        obj.subsample = rust_obj.subsample
        obj.min_samples_split = rust_obj.min_samples_split
        obj.min_samples_leaf = rust_obj.min_samples_leaf
        obj.max_depth = rust_obj.max_depth
        obj.alpha = rust_obj.alpha
        obj.random_state = rust_obj.random_state
        obj.device = rust_obj.device_ or device
        return obj

    def __setstate__(self, state: bytes) -> None:
        rust_obj = _RustGradientBoostingRegressor.from_bytes(state, None)
        self._model = rust_obj
        self.loss = rust_obj.loss
        self.learning_rate = rust_obj.learning_rate
        self.n_estimators = rust_obj.n_estimators
        self.subsample = rust_obj.subsample
        self.min_samples_split = rust_obj.min_samples_split
        self.min_samples_leaf = rust_obj.min_samples_leaf
        self.max_depth = rust_obj.max_depth
        self.alpha = rust_obj.alpha
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or "auto"

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "GradientBoostingRegressor":
        """
        Fit the gradient boosting model (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.

        Returns
        -------
        self : GradientBoostingRegressor
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def staged_predict(self, X: Any) -> Iterator[Any]:
        """
        Predict regression target at each stage for X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Yields
        ------
        y : numpy.ndarray of shape (n_samples,)
            The predicted value of the input samples at each stage.
        """
        X_data = _to_f64_2d(X)
        n_stages, n_samples, raw = self._model.staged_predict_buffer(X_data)
        mat = bytes_to_f64_2d(n_stages, n_samples, raw)
        for m in range(n_stages):
            yield mat[m].copy()


class GradientBoostingClassifier(_BaseBoostingClassifier):
    """
    Gradient Boosting for classification (＾▽＾).

    Parameters
    ----------
    loss : {'log_loss'}, default='log_loss'
        The loss function to be optimized.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    n_estimators : int, default=100
        The number of boosting stages to perform.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base learners.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    random_state : int or None, default=None
        Controls the random seed given to each Tree estimator at each boosting iteration.
    device : str or None, default=None
        Execution device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustGradientBoostingClassifier

    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        resolved_dev = _resolve_device(device)
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.device = resolved_dev
        self._model = _RustGradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
            device=resolved_dev,
        )

    @classmethod
    def _from_rust(cls, rust_obj: Any, device: str) -> "GradientBoostingClassifier":
        obj = cls.__new__(cls)
        obj._model = rust_obj
        obj.loss = rust_obj.loss
        obj.learning_rate = rust_obj.learning_rate
        obj.n_estimators = rust_obj.n_estimators
        obj.subsample = rust_obj.subsample
        obj.min_samples_split = rust_obj.min_samples_split
        obj.min_samples_leaf = rust_obj.min_samples_leaf
        obj.max_depth = rust_obj.max_depth
        obj.random_state = rust_obj.random_state
        obj.device = rust_obj.device_ or device
        return obj

    def __setstate__(self, state: bytes) -> None:
        rust_obj = _RustGradientBoostingClassifier.from_bytes(state, None)
        self._model = rust_obj
        self.loss = rust_obj.loss
        self.learning_rate = rust_obj.learning_rate
        self.n_estimators = rust_obj.n_estimators
        self.subsample = rust_obj.subsample
        self.min_samples_split = rust_obj.min_samples_split
        self.min_samples_leaf = rust_obj.min_samples_leaf
        self.max_depth = rust_obj.max_depth
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or "auto"

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "GradientBoostingClassifier":
        """
        Fit the gradient boosting classifier (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.

        Returns
        -------
        self : GradientBoostingClassifier
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self


class AdaBoostClassifier(_BaseBoostingClassifier):
    """
    An AdaBoost classifier using SAMME weighted decision trees (＾▽＾).

    Parameters
    ----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    max_depth : int, default=1
        Maximum depth of the decision tree base estimators.
    random_state : int or None, default=None
        Controls the random seed given at each boosting iteration.
    device : str or None, default=None
        Execution device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustAdaBoostClassifier

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        resolved_dev = _resolve_device(device)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.device = resolved_dev
        self._model = _RustAdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            device=resolved_dev,
        )

    @classmethod
    def _from_rust(cls, rust_obj: Any, device: str) -> "AdaBoostClassifier":
        obj = cls.__new__(cls)
        obj._model = rust_obj
        obj.n_estimators = rust_obj.n_estimators
        obj.learning_rate = rust_obj.learning_rate
        obj.max_depth = rust_obj.max_depth
        obj.random_state = rust_obj.random_state
        obj.device = rust_obj.device_ or device
        return obj

    def __setstate__(self, state: bytes) -> None:
        rust_obj = _RustAdaBoostClassifier.from_bytes(state, None)
        self._model = rust_obj
        self.n_estimators = rust_obj.n_estimators
        self.learning_rate = rust_obj.learning_rate
        self.max_depth = rust_obj.max_depth
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or "auto"

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "AdaBoostClassifier":
        """
        Build a boosted classifier from the training set (X, y) (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.

        Returns
        -------
        self : AdaBoostClassifier
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    @property
    def estimator_weights_(self) -> Any:
        return np.asarray(self._model.estimator_weights_, dtype=np.float64)

    @property
    def estimator_errors_(self) -> Any:
        return np.asarray(self._model.estimator_errors_, dtype=np.float64)


class AdaBoostRegressor(_BaseBoosting):
    """
    An AdaBoost.R2 regressor with weighted median prediction (＾▽＾).

    Parameters
    ----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration.
    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each boosting iteration.
    max_depth : int, default=3
        Maximum depth of the decision tree base estimators.
    random_state : int or None, default=None
        Controls the random seed given at each boosting iteration.
    device : str or None, default=None
        Execution device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustAdaBoostRegressor

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        max_depth: int = 3,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        resolved_dev = _resolve_device(device)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.random_state = random_state
        self.device = resolved_dev
        self._model = _RustAdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            max_depth=max_depth,
            random_state=random_state,
            device=resolved_dev,
        )

    @classmethod
    def _from_rust(cls, rust_obj: Any, device: str) -> "AdaBoostRegressor":
        obj = cls.__new__(cls)
        obj._model = rust_obj
        obj.n_estimators = rust_obj.n_estimators
        obj.learning_rate = rust_obj.learning_rate
        obj.loss = rust_obj.loss
        obj.max_depth = rust_obj.max_depth
        obj.random_state = rust_obj.random_state
        obj.device = rust_obj.device_ or device
        return obj

    def __setstate__(self, state: bytes) -> None:
        rust_obj = _RustAdaBoostRegressor.from_bytes(state, None)
        self._model = rust_obj
        self.n_estimators = rust_obj.n_estimators
        self.learning_rate = rust_obj.learning_rate
        self.loss = rust_obj.loss
        self.max_depth = rust_obj.max_depth
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or "auto"

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "AdaBoostRegressor":
        """
        Build a boosted regressor from the training set (X, y) (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.

        Returns
        -------
        self : AdaBoostRegressor
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    @property
    def estimator_weights_(self) -> Any:
        return np.asarray(self._model.estimator_weights_, dtype=np.float64)

    @property
    def estimator_errors_(self) -> Any:
        return np.asarray(self._model.estimator_errors_, dtype=np.float64)


__all__ = [
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
]
