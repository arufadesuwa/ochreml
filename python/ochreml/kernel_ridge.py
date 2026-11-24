"""
OchreML Kernel Ridge Regression Module (＾▽＾)
Provides Scikit-Learn-compatible KernelRidge accelerated by Rust hardware backends (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import KernelRidge as _RustKernelRidge
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
    return np.ascontiguousarray(X_data, dtype=np.float64)


def _to_sample_weight_1d(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if not is_buf and sw_data is not None:
        sw_data = np.asarray(sw_data, dtype=np.float64)
    return np.ascontiguousarray(sw_data, dtype=np.float64) if sw_data is not None else None


class KernelRidge:
    """
    Kernel ridge regression (＾▽＾).

    Combines ridge regression (linear least squares with l2-norm regularization)
    with the kernel trick. It learns a linear function in the space induced by
    the respective kernel and the data (*≧ω≦*).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a non-negative float.
    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default='linear'
        Kernel mapping used internally.
    gamma : float or None, default=None
        Gamma parameter for the RBF, poly, and sigmoid kernels.
        If None, defaults to ``1.0 / n_features``.
    degree : int, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1.0
        Zero coefficient for polynomial and sigmoid kernels.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Representation of weight vector(s) in kernel space.
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training data, which is also required for prediction.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustKernelRidge

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.device = _resolve_device(device)
        self._y_1d = True
        self._model = _RustKernelRidge(
            alpha=float(alpha),
            kernel=str(kernel),
            gamma=float(gamma) if gamma is not None else None,
            degree=int(degree),
            coef0=float(coef0),
            device=self.device,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "KernelRidge":
        """Fit Kernel Ridge regression model (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            self._y_1d = True
            y_2d = np.ascontiguousarray(y_arr.reshape(-1, 1), dtype=np.float64)
        elif y_arr.ndim == 2:
            self._y_1d = False
            y_2d = np.ascontiguousarray(y_arr, dtype=np.float64)
        else:
            raise ValueError(f"Expected 1D or 2D target y, got {y_arr.ndim}D ( >_< )")

        sw_arr = _to_sample_weight_1d(sample_weight)
        self._model.fit(X_arr, y_2d, sample_weight=sw_arr)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the kernel ridge model (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.predict(X_arr)
        out = bytes_to_f64_2d(rows, cols, raw)
        if self._y_1d and cols == 1:
            return out.ravel()
        return out

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """Return the coefficient of determination R^2 of the prediction (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_2d = np.ascontiguousarray(y_arr.reshape(-1, 1), dtype=np.float64)
        elif y_arr.ndim == 2:
            y_2d = np.ascontiguousarray(y_arr, dtype=np.float64)
        else:
            raise ValueError(f"Expected 1D or 2D target y, got {y_arr.ndim}D ( >_< )")
        sw_arr = _to_sample_weight_1d(sample_weight)
        return float(self._model.score(X_arr, y_2d, sample_weight=sw_arr))

    @property
    def dual_coef_(self) -> Optional[Any]:
        res = self._model.dual_coef_
        if res is None:
            return None
        rows, cols, raw = res
        arr = bytes_to_f64_2d(rows, cols, raw)
        if self._y_1d and cols == 1:
            return arr.ravel()
        return arr

    @property
    def X_fit_(self) -> Optional[Any]:
        res = self._model.x_fit_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def to(self, device: Optional[str]) -> "KernelRidge":
        """Move model compute target to another device (・`ω´・)."""
        self._model.set_device(device)
        self.device = device if device is not None else "auto"
        return self

    def save(self, path: str) -> None:
        """Save the fitted model to a '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def _from_rust(cls, rust_model: Any, device: Optional[str] = None) -> "KernelRidge":
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or (device or "auto")
        dc = rust_model.dual_coef_
        obj._y_1d = dc is not None and dc[1] == 1
        obj.alpha = 1.0
        obj.kernel = "linear"
        obj.gamma = None
        obj.degree = 3
        obj.coef0 = 1.0
        return obj

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KernelRidge":
        """Load a model from a '.dit' file (*≧ω≦*)."""
        rust_model = _RustKernelRidge.load(validate_dit_path(path), device=device)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or "auto"
        dc = rust_model.dual_coef_
        obj._y_1d = dc is not None and dc[1] == 1
        obj.alpha = 1.0
        obj.kernel = "linear"
        obj.gamma = None
        obj.degree = 3
        obj.coef0 = 1.0
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "alpha": self.alpha,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "device": self.device,
            "_y_1d": self._y_1d,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.alpha = state["alpha"]
        self.kernel = state["kernel"]
        self.gamma = state["gamma"]
        self.degree = state["degree"]
        self.coef0 = state["coef0"]
        self.device = state["device"]
        self._y_1d = state.get("_y_1d", True)
        self._model = _RustKernelRidge.from_bytes(state["bytes"], device=self.device)
