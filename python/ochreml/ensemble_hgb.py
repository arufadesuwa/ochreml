"""
OchreML Histogram-based Gradient Boosting Submodule (*≧ω≦*)
Provides Scikit-Learn-compatible HistGradientBoostingClassifier and
HistGradientBoostingRegressor backed by Rust N6 GPU/CPU histogram primitives.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    HistGradientBoostingClassifier as _RustHistGradientBoostingClassifier,
    HistGradientBoostingRegressor as _RustHistGradientBoostingRegressor,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_sample_weight, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _prep_x(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    return np.asarray(X_data, dtype=np.float64)


def _prep_y(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if is_buf:
        return y_data
    return np.asarray(y_data, dtype=np.float64)


def _prep_sw(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if is_buf:
        return sw_data
    return np.asarray(sw_data, dtype=np.float64)


class _BaseHgbMixin:
    """Shared persistence, device management, and buffer helpers for HGB models."""

    _model: Any
    device: str

    def _sync_params_from_rust(self) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save the fitted model to a `.dit` file (＾▽＾).

        Parameters
        ----------
        path : str
            Destination file path ending with `.dit`.
        """
        self._model.save(validate_dit_path(path))

    @classmethod
    def _from_rust(cls, rust_model: Any, device: Optional[str] = None) -> Any:
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or _resolve_device(device)
        obj._sync_params_from_rust()
        return obj

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """
        Load a fitted model from a `.dit` file (o´∀｀o).

        Parameters
        ----------
        path : str
            Source file path ending with `.dit`.
        device : str or None, default=None
            Target device (`'auto'`, `'cpu'`, `'cuda:0'`, `'wgpu'`).

        Returns
        -------
        model : HistGradientBoostingClassifier or HistGradientBoostingRegressor
            Loaded estimator instance.
        """
        target_dev = _resolve_device(device)
        rust_cls = (
            _RustHistGradientBoostingClassifier
            if cls is HistGradientBoostingClassifier
            else _RustHistGradientBoostingRegressor
        )
        rust_model = rust_cls.load(validate_dit_path(path), target_dev)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = target_dev
        obj._sync_params_from_rust()
        return obj

    def to(self, device: str) -> Any:
        """
        Move the estimator to another execution device (*≧ω≦*).

        Parameters
        ----------
        device : str
            Target device string.

        Returns
        -------
        self : object
            Estimator configured for the requested device.
        """
        target_dev = _resolve_device(device)
        self._model.set_device(target_dev)
        self.device = target_dev
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {
            "state": self._model.to_bytes(),
            "device": self.device,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        rust_cls = (
            _RustHistGradientBoostingClassifier
            if type(self) is HistGradientBoostingClassifier
            else _RustHistGradientBoostingRegressor
        )
        self.device = state.get("device", "auto")
        self._model = rust_cls.from_bytes(state["state"], self.device)
        self._sync_params_from_rust()

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class HistGradientBoostingRegressor(_BaseHgbMixin):
    """
    Histogram-based Gradient Boosting Regression Tree (＾▽＾).

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function to use in the boosting process.
    learning_rate : float, default=0.1
        The learning rate, also known as shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process (number of trees).
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree.
    max_depth : int or None, default=None
        The maximum depth of each tree.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    l2_regularization : float, default=0.0
        The L2 regularization parameter used in leaf value and split gain computation.
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values (`2..=255`).
    random_state : int or None, default=None
        Random seed for reproducibility.
    device : str or None, default=None
        Execution device (`'auto'`, `'cpu'`, `'cuda:0'`, `'wgpu'`).
    """

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustHistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.loss = self._model.loss
        self.learning_rate = self._model.learning_rate
        self.max_iter = self._model.max_iter
        self.max_leaf_nodes = self._model.max_leaf_nodes
        self.max_depth = self._model.max_depth
        self.min_samples_leaf = self._model.min_samples_leaf
        self.l2_regularization = self._model.l2_regularization
        self.max_bins = self._model.max_bins
        self.random_state = self._model.random_state

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "HistGradientBoostingRegressor":
        """
        Fit the histogram gradient boosting regressor (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : HistGradientBoostingRegressor
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X), _prep_y(y), _prep_sw(sample_weight))
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict target values for X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R^2 of ``self.predict(X)`` w.r.t. ``y``.
        """
        return float(self._model.score_buffer(_prep_x(X), _prep_y(y)))

    def __repr__(self) -> str:
        return (
            f"HistGradientBoostingRegressor(loss='{self.loss}', "
            f"learning_rate={self.learning_rate}, max_iter={self.max_iter}, "
            f"max_leaf_nodes={self.max_leaf_nodes}, device='{self.device}')"
        )


class HistGradientBoostingClassifier(_BaseHgbMixin):
    """
    Histogram-based Gradient Boosting Classification Tree (＾▽＾).

    Parameters
    ----------
    learning_rate : float, default=0.1
        The learning rate, also known as shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree.
    max_depth : int or None, default=None
        The maximum depth of each tree.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    l2_regularization : float, default=0.0
        The L2 regularization parameter used in leaf value and split gain computation.
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values (`2..=255`).
    random_state : int or None, default=None
        Random seed for reproducibility.
    device : str or None, default=None
        Execution device (`'auto'`, `'cpu'`, `'cuda:0'`, `'wgpu'`).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustHistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.learning_rate = self._model.learning_rate
        self.max_iter = self._model.max_iter
        self.max_leaf_nodes = self._model.max_leaf_nodes
        self.max_depth = self._model.max_depth
        self.min_samples_leaf = self._model.min_samples_leaf
        self.l2_regularization = self._model.l2_regularization
        self.max_bins = self._model.max_bins
        self.random_state = self._model.random_state

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "HistGradientBoostingClassifier":
        """
        Fit the histogram gradient boosting classifier (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : HistGradientBoostingClassifier
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X), _prep_y(y), _prep_sw(sample_weight))
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict class labels for X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return np.frombuffer(raw, dtype=np.float64).copy()

    def predict_proba(self, X: Any) -> Any:
        """
        Predict class probabilities for X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities of the input samples.
        """
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def decision_function(self, X: Any) -> Any:
        """
        Compute the decision function of X (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Raw decision scores before sigmoid/softmax transformation.
        """
        rows, cols, raw = self._model.decision_function_buffer(_prep_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. ``y``.
        """
        return float(self._model.score_buffer(_prep_x(X), _prep_y(y)))

    @property
    def classes_(self) -> Any:
        return np.asarray(self._model.classes_, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"HistGradientBoostingClassifier(learning_rate={self.learning_rate}, "
            f"max_iter={self.max_iter}, max_leaf_nodes={self.max_leaf_nodes}, "
            f"device='{self.device}')"
        )
