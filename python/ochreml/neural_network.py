"""
OchreML Multi-Layer Perceptron Submodule (*≧ω≦*)
Provides Scikit-Learn-compatible MLPClassifier and MLPRegressor backed by
Rust hardware-accelerated matrix and elementwise primitives.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    MLPClassifier as _RustMLPClassifier,
    MLPRegressor as _RustMLPRegressor,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _normalize_hidden_layers(
    hidden_layer_sizes: Union[int, Sequence[int]],
) -> list[int]:
    if isinstance(hidden_layer_sizes, int):
        return [int(hidden_layer_sizes)]
    return [int(x) for x in hidden_layer_sizes]


def _normalize_batch_size(batch_size: Union[str, int, None]) -> Optional[int]:
    if batch_size is None or batch_size == "auto":
        return None
    if isinstance(batch_size, (int, np.integer if np is not None else int)):
        return int(batch_size)
    raise ValueError(
        f"batch_size must be 'auto' or a positive integer, got {batch_size!r} ( >_< )"
    )


def _prep_x(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    return np.asarray(X_data, dtype=np.float64)


def _prep_y_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if is_buf:
        return y_data
    return np.asarray(y_data, dtype=np.float64)


def _prep_y_reg(y: Any) -> tuple[Any, bool]:
    if y is None:
        raise ValueError("Target y cannot be None ( >_< )")
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.float64), True
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float64), False
    raise ValueError(
        f"Target array y must be 1D or 2D, got {arr.ndim}D ( >_< )"
    )


class _BaseMlpMixin:
    """Shared persistence, device management, and attribute helpers for MLP models."""

    _model: Any
    device: str

    def _sync_params_from_rust(self) -> None:
        self.hidden_layer_sizes = tuple(self._model.hidden_layer_sizes)
        self.activation = self._model.activation
        self.solver = self._model.solver
        self.alpha = self._model.alpha
        rs_bs = self._model.batch_size
        self.batch_size = "auto" if rs_bs is None else rs_bs
        self.learning_rate = self._model.learning_rate
        self.learning_rate_init = self._model.learning_rate_init
        self.power_t = self._model.power_t
        self.max_iter = self._model.max_iter
        self.shuffle = self._model.shuffle
        self.random_state = self._model.random_state
        self.tol = self._model.tol
        self.momentum = self._model.momentum
        self.nesterovs_momentum = self._model.nesterovs_momentum
        self.early_stopping = self._model.early_stopping
        self.validation_fraction = self._model.validation_fraction
        self.beta_1 = self._model.beta_1
        self.beta_2 = self._model.beta_2
        self.epsilon = self._model.epsilon
        self.n_iter_no_change = self._model.n_iter_no_change

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
        model : MLPClassifier or MLPRegressor
            Loaded estimator instance.
        """
        target_dev = _resolve_device(device)
        rust_cls = (
            _RustMLPClassifier
            if cls is MLPClassifier
            else _RustMLPRegressor
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
            _RustMLPClassifier
            if type(self) is MLPClassifier
            else _RustMLPRegressor
        )
        self.device = state.get("device", "auto")
        self._model = rust_cls.from_bytes(state["state"], self.device)
        self._sync_params_from_rust()

    @property
    def coefs_(self) -> Optional[list[Any]]:
        raw = self._model.coefs_
        if raw is None:
            return None
        return [bytes_to_f64_2d(r, c, buf) for (r, c, buf) in raw]

    @property
    def intercepts_(self) -> Optional[list[Any]]:
        raw = self._model.intercepts_
        if raw is None:
            return None
        return [np.asarray(b, dtype=np.float64) for b in raw]

    @property
    def loss_(self) -> Optional[float]:
        return self._model.loss_

    @property
    def loss_curve_(self) -> list[float]:
        return list(self._model.loss_curve_)

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter_

    @property
    def n_layers_(self) -> Optional[int]:
        return self._model.n_layers_

    @property
    def n_outputs_(self) -> Optional[int]:
        return self._model.n_outputs_

    @property
    def out_activation_(self) -> Optional[str]:
        return self._model.out_activation_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class MLPClassifier(_BaseMlpMixin):
    """
    Multi-layer Perceptron classifier (＾▽＾).

    Optimizes log-loss using Adam, SGD with momentum, or L-BFGS, accelerated
    by Rust E1/E2/N2 backend primitives.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int or int, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layers.
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.
    alpha : float, default=0.0001
        Strength of the L2 regularization term.
    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates (used when `solver='sgd'`).
    learning_rate_init : float, default=0.001
        The initial learning rate used.
    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
    max_iter : int, default=200
        Maximum number of iterations.
    shuffle : bool, default=True
        Whether to shuffle samples in each iteration.
    random_state : int or None, default=None
        Determines random number generation for weights and shuffling.
    tol : float, default=1e-4
        Tolerance for the optimization.
    momentum : float, default=0.9
        Momentum for gradient descent update.
    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum.
    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set.
    beta_1 : float, default=0.9
        Exponential decay rate for first moment estimates in Adam.
    beta_2 : float, default=0.999
        Exponential decay rate for second moment estimates in Adam.
    epsilon : float, default=1e-8
        Value for numerical stability in Adam.
    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement.
    device : str or None, default=None
        Execution device (`'auto'`, `'cpu'`, `'cuda:0'`, `'wgpu'`).
    """

    def __init__(
        self,
        hidden_layer_sizes: Union[int, Sequence[int]] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[str, int] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        device: Optional[str] = None,
    ) -> None:
        self.device = _resolve_device(device)
        hls = _normalize_hidden_layers(hidden_layer_sizes)
        bs = _normalize_batch_size(batch_size)
        self._model = _RustMLPClassifier(
            hidden_layer_sizes=hls,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=bs,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            device=self.device,
        )
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Any) -> "MLPClassifier":
        """Fit the model to data matrix X and target(s) y (*≧ω≦*)."""
        self._model.fit_buffer(_prep_x(X), _prep_y_1d(y))
        return self

    def partial_fit(
        self, X: Any, y: Any, classes: Optional[Any] = None
    ) -> "MLPClassifier":
        """Update the model with a single iteration over the given data."""
        cls_list = None if classes is None else [float(c) for c in classes]
        self._model.partial_fit_buffer(_prep_x(X), _prep_y_1d(y), classes=cls_list)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the multi-layer perceptron classifier."""
        raw = self._model.predict_buffer(_prep_x(X))
        return np.frombuffer(raw, dtype=np.float64).copy()

    def predict_proba(self, X: Any) -> Any:
        """Probability estimates."""
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def predict_log_proba(self, X: Any) -> Any:
        """Return the logarithm of probability estimates."""
        rows, cols, raw = self._model.predict_log_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, y: Any) -> float:
        """Return the mean accuracy on the given test data and labels."""
        return float(self._model.score_buffer(_prep_x(X), _prep_y_1d(y)))

    @property
    def classes_(self) -> Optional[Any]:
        cls = self._model.classes_
        if cls is None:
            return None
        return np.asarray(cls, dtype=np.float64)


class MLPRegressor(_BaseMlpMixin):
    """
    Multi-layer Perceptron regressor (o´∀｀o).

    Optimizes squared error using Adam, SGD with momentum, or L-BFGS,
    accelerated by Rust E1/E2/N2 backend primitives.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int or int, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layers.
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.
    alpha : float, default=0.0001
        Strength of the L2 regularization term.
    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates (used when `solver='sgd'`).
    learning_rate_init : float, default=0.001
        The initial learning rate used.
    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
    max_iter : int, default=200
        Maximum number of iterations.
    shuffle : bool, default=True
        Whether to shuffle samples in each iteration.
    random_state : int or None, default=None
        Determines random number generation for weights and shuffling.
    tol : float, default=1e-4
        Tolerance for the optimization.
    momentum : float, default=0.9
        Momentum for gradient descent update.
    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum.
    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set.
    beta_1 : float, default=0.9
        Exponential decay rate for first moment estimates in Adam.
    beta_2 : float, default=0.999
        Exponential decay rate for second moment estimates in Adam.
    epsilon : float, default=1e-8
        Value for numerical stability in Adam.
    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement.
    device : str or None, default=None
        Execution device (`'auto'`, `'cpu'`, `'cuda:0'`, `'wgpu'`).
    """

    def __init__(
        self,
        hidden_layer_sizes: Union[int, Sequence[int]] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[str, int] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        device: Optional[str] = None,
    ) -> None:
        self.device = _resolve_device(device)
        hls = _normalize_hidden_layers(hidden_layer_sizes)
        bs = _normalize_batch_size(batch_size)
        self._model = _RustMLPRegressor(
            hidden_layer_sizes=hls,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=bs,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            device=self.device,
        )
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Any) -> "MLPRegressor":
        """Fit the model to data matrix X and target(s) y (*≧ω≦*)."""
        y_arr, _ = _prep_y_reg(y)
        self._model.fit_buffer(_prep_x(X), y_arr)
        return self

    def partial_fit(self, X: Any, y: Any) -> "MLPRegressor":
        """Update the model with a single iteration over the given data."""
        y_arr, _ = _prep_y_reg(y)
        self._model.partial_fit_buffer(_prep_x(X), y_arr)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the multi-layer perceptron model."""
        rows, cols, raw = self._model.predict_buffer(_prep_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        y_arr, _ = _prep_y_reg(y)
        return float(self._model.score_buffer(_prep_x(X), y_arr))
