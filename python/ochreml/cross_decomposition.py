"""
OchreML Matrix & Cross-Decomposition Module (＾▽＾)
Provides Scikit-Learn-compatible NMF, PLSRegression, and CCA accelerated by Rust hardware backends (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    CCA as _RustCCA,
    NMF as _RustNMF,
    PLSRegression as _RustPLSRegression,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, to_feature_matrix


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_2d_x(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    if np is not None:
        return np.ascontiguousarray(X_data, dtype=np.float64)
    return to_feature_matrix(X)


def _to_f64_2d_y(Y: Any) -> Tuple[Any, int]:
    if Y is None:
        raise ValueError("Target array Y cannot be None ( >_< )")
    if np is not None:
        if hasattr(Y, "to_numpy"):
            arr = Y.to_numpy()
        else:
            arr = np.asarray(Y, dtype=np.float64)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr.reshape(-1, 1), dtype=np.float64), 1
        if arr.ndim == 2:
            return np.ascontiguousarray(arr, dtype=np.float64), 2
        raise ValueError(f"Expected 1D or 2D target array Y, got {arr.ndim}D ( >_< )")
    raise TypeError("NumPy is required for 2D target conversion ( >_< )")


class NMF:
    """
    Non-Negative Matrix Factorization (NMF) (＾▽＾)

    Find two non-negative matrices (W, H) whose product approximates the
    non-negative matrix X using Multiplicative Updates (*≧ω≦*).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components, if n_components is not set all features are kept.
    init : {'nndsvd', 'nndsvda', 'random'} or None, default=None
        Method used to initialize the procedure.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    l1_ratio : float, default=0.0
        The regularization mixing parameter, with ``0 <= l1_ratio <= 1``.
    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of ``W``.
    alpha_H : float or None, default=None
        Constant that multiplies the regularization terms of ``H``.
        If None, defaults to ``alpha_W``.
    random_state : int or None, default=None
        Used for randomization when ``init='random'``.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    components_ : ndarray of shape (n_components_, n_features_in_)
        Factorization matrix, sometimes called 'dictionary'.
    n_components_ : int
        The number of components.
    reconstruction_err_ : float
        Frobenius norm of the matrix difference between the training data X
        and the reconstructed data WH.
    n_iter_ : int
        Actual number of iterations.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustNMF

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Optional[str] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        l1_ratio: float = 0.0,
        alpha_W: float = 0.0,
        alpha_H: Optional[float] = None,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustNMF(
            n_components=n_components,
            init=init,
            max_iter=max_iter,
            tol=tol,
            l1_ratio=l1_ratio,
            alpha_w=alpha_W,
            alpha_h=alpha_H,
            random_state=random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustNMF, device: str) -> "NMF":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.init = rust_obj.init
        inst.max_iter = rust_obj.max_iter
        inst.tol = rust_obj.tol
        inst.l1_ratio = rust_obj.l1_ratio
        inst.alpha_W = rust_obj.alpha_w
        inst.alpha_H = rust_obj.alpha_h
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Optional[Any] = None) -> "NMF":
        """Learn a NMF model for the data X (o´∀｀o)."""
        X_data = _to_f64_2d_x(X)
        self._model.fit_buffer(X_data)
        return self

    def transform(self, X: Any) -> Any:
        """Transform the data X according to the fitted NMF model (*≧ω≦*)."""
        X_data = _to_f64_2d_x(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Learn a NMF model for the data X and return the transformed data (＾▽＾)."""
        X_data = _to_f64_2d_x(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, W: Any) -> Any:
        """Transform data back to its original space (o´∀｀o)."""
        W_data = _to_f64_2d_x(W)
        rows, cols, raw = self._model.inverse_transform_buffer(W_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def save(self, path: str) -> None:
        """Save fitted NMF model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "NMF":
        """Load NMF model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "NMF":
        """Move NMF execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustNMF.from_bytes(state, target_dev)
        self._model = rust_obj
        self.n_components = rust_obj.n_components
        self.init = rust_obj.init
        self.max_iter = rust_obj.max_iter
        self.tol = rust_obj.tol
        self.l1_ratio = rust_obj.l1_ratio
        self.alpha_W = rust_obj.alpha_w
        self.alpha_H = rust_obj.alpha_h
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or target_dev

    @property
    def components_(self) -> Optional[Any]:
        res = self._model.components_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def n_components_(self) -> Optional[int]:
        return self._model.n_components_

    @property
    def reconstruction_err_(self) -> Optional[float]:
        return self._model.reconstruction_err_

    @property
    def n_iter_(self) -> Optional[int]:
        return self._model.n_iter_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class PLSRegression:
    """
    Partial Least Squares (PLS) Regression (＾▽＾)

    Implements the NIPALS algorithm for multi-target and single-target
    partial least squares regression (*≧ω≦*).

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    scale : bool, default=True
        Whether to scale X and Y using sample standard deviation (ddof=1).
    max_iter : int, default=500
        Maximum number of iterations of the NIPALS inner loop.
    tol : float, default=1e-6
        Tolerance used as iterative stopping condition.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustPLSRegression

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.device = _resolve_device(device)
        self._y_ndim = 2
        self._model = _RustPLSRegression(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustPLSRegression, device: str) -> "PLSRegression":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.scale = rust_obj.scale
        inst.max_iter = rust_obj.max_iter
        inst.tol = rust_obj.tol
        inst.device = rust_obj.device_ or device
        inst._y_ndim = 2
        return inst

    def fit(self, X: Any, Y: Any) -> "PLSRegression":
        """Fit model to data (X, Y) (o´∀｀o)."""
        X_data = _to_f64_2d_x(X)
        Y_data, y_ndim = _to_f64_2d_y(Y)
        self._y_ndim = y_ndim
        self._model.fit_buffer(X_data, Y_data)
        return self

    def transform(self, X: Any, Y: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """Apply the dimension reduction learned on the train data (*≧ω≦*)."""
        X_data = _to_f64_2d_x(X)
        if Y is None:
            (rows, cols, raw), _ = self._model.transform_buffer(X_data, None)
            return bytes_to_f64_2d(rows, cols, raw)
        Y_data, _ = _to_f64_2d_y(Y)
        (xr, xc, xraw), y_res = self._model.transform_buffer(X_data, Y_data)
        x_scores = bytes_to_f64_2d(xr, xc, xraw)
        if y_res is None:
            return x_scores
        yr, yc, yraw = y_res
        return x_scores, bytes_to_f64_2d(yr, yc, yraw)

    def fit_transform(self, X: Any, Y: Any) -> Tuple[Any, Any]:
        """Learn and apply the dimension reduction on the train data (＾▽＾)."""
        X_data = _to_f64_2d_x(X)
        Y_data, y_ndim = _to_f64_2d_y(Y)
        self._y_ndim = y_ndim
        (xr, xc, xraw), (yr, yc, yraw) = self._model.fit_transform_buffer(X_data, Y_data)
        return bytes_to_f64_2d(xr, xc, xraw), bytes_to_f64_2d(yr, yc, yraw)

    def predict(self, X: Any) -> Any:
        """Predict targets of given samples (o´∀｀o)."""
        X_data = _to_f64_2d_x(X)
        rows, cols, raw = self._model.predict_buffer(X_data)
        out = bytes_to_f64_2d(rows, cols, raw)
        if self._y_ndim == 1 and cols == 1:
            return out.ravel()
        return out

    def inverse_transform(self, X: Any) -> Any:
        """Transform data back to its original space (*≧ω≦*)."""
        X_data = _to_f64_2d_x(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, Y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (＾▽＾)."""
        X_data = _to_f64_2d_x(X)
        Y_data, _ = _to_f64_2d_y(Y)
        return float(self._model.score_buffer(X_data, Y_data))

    def save(self, path: str) -> None:
        """Save fitted PLSRegression model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PLSRegression":
        """Load PLSRegression model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "PLSRegression":
        """Move PLSRegression execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> Tuple[bytes, int]:
        return (self._model.to_bytes(), self._y_ndim)

    def __setstate__(self, state: Union[bytes, Tuple[bytes, int]]) -> None:
        target_dev = _resolve_device(None)
        if isinstance(state, tuple):
            raw_bytes, y_ndim = state
        else:
            raw_bytes, y_ndim = state, 2
        rust_obj = _RustPLSRegression.from_bytes(raw_bytes, target_dev)
        self._model = rust_obj
        self.n_components = rust_obj.n_components
        self.scale = rust_obj.scale
        self.max_iter = rust_obj.max_iter
        self.tol = rust_obj.tol
        self.device = rust_obj.device_ or target_dev
        self._y_ndim = y_ndim

    @property
    def x_weights_(self) -> Optional[Any]:
        res = self._model.x_weights_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_weights_(self) -> Optional[Any]:
        res = self._model.y_weights_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def x_loadings_(self) -> Optional[Any]:
        res = self._model.x_loadings_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_loadings_(self) -> Optional[Any]:
        res = self._model.y_loadings_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def x_rotations_(self) -> Optional[Any]:
        res = self._model.x_rotations_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_rotations_(self) -> Optional[Any]:
        res = self._model.y_rotations_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def coef_(self) -> Optional[Any]:
        res = self._model.coef_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def intercept_(self) -> Optional[Any]:
        raw = self._model.intercept_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy() if np is not None else list(raw)

    @property
    def n_iter_(self) -> Optional[Any]:
        return self._model.n_iter_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class CCA:
    """
    Canonical Correlation Analysis (CCA) (＾▽＾)

    Implements NIPALS Mode B canonical correlation analysis to find linear
    projections maximizing correlation between X and Y (*≧ω≦*).

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    scale : bool, default=True
        Whether to scale X and Y using sample standard deviation (ddof=1).
    max_iter : int, default=500
        Maximum number of iterations of the NIPALS inner loop.
    tol : float, default=1e-6
        Tolerance used as iterative stopping condition.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustCCA

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.device = _resolve_device(device)
        self._model = _RustCCA(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustCCA, device: str) -> "CCA":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.scale = rust_obj.scale
        inst.max_iter = rust_obj.max_iter
        inst.tol = rust_obj.tol
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, Y: Any) -> "CCA":
        """Fit model to data (X, Y) (o´∀｀o)."""
        X_data = _to_f64_2d_x(X)
        Y_data, _ = _to_f64_2d_y(Y)
        self._model.fit_buffer(X_data, Y_data)
        return self

    def transform(self, X: Any, Y: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """Apply the dimension reduction learned on the train data (*≧ω≦*)."""
        X_data = _to_f64_2d_x(X)
        if Y is None:
            (rows, cols, raw), _ = self._model.transform_buffer(X_data, None)
            return bytes_to_f64_2d(rows, cols, raw)
        Y_data, _ = _to_f64_2d_y(Y)
        (xr, xc, xraw), y_res = self._model.transform_buffer(X_data, Y_data)
        x_scores = bytes_to_f64_2d(xr, xc, xraw)
        if y_res is None:
            return x_scores
        yr, yc, yraw = y_res
        return x_scores, bytes_to_f64_2d(yr, yc, yraw)

    def fit_transform(self, X: Any, Y: Any) -> Tuple[Any, Any]:
        """Learn and apply the dimension reduction on the train data (＾▽＾)."""
        X_data = _to_f64_2d_x(X)
        Y_data, _ = _to_f64_2d_y(Y)
        (xr, xc, xraw), (yr, yc, yraw) = self._model.fit_transform_buffer(X_data, Y_data)
        return bytes_to_f64_2d(xr, xc, xraw), bytes_to_f64_2d(yr, yc, yraw)

    def predict(self, X: Any) -> Any:
        """Predict targets of given samples (o´∀｀o)."""
        X_data = _to_f64_2d_x(X)
        rows, cols, raw = self._model.predict_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, Y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (＾▽＾)."""
        X_data = _to_f64_2d_x(X)
        Y_data, _ = _to_f64_2d_y(Y)
        return float(self._model.score_buffer(X_data, Y_data))

    def save(self, path: str) -> None:
        """Save fitted CCA model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CCA":
        """Load CCA model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "CCA":
        """Move CCA execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustCCA.from_bytes(state, target_dev)
        self._model = rust_obj
        self.n_components = rust_obj.n_components
        self.scale = rust_obj.scale
        self.max_iter = rust_obj.max_iter
        self.tol = rust_obj.tol
        self.device = rust_obj.device_ or target_dev

    @property
    def x_weights_(self) -> Optional[Any]:
        res = self._model.x_weights_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_weights_(self) -> Optional[Any]:
        res = self._model.y_weights_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def x_loadings_(self) -> Optional[Any]:
        res = self._model.x_loadings_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_loadings_(self) -> Optional[Any]:
        res = self._model.y_loadings_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def x_rotations_(self) -> Optional[Any]:
        res = self._model.x_rotations_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def y_rotations_(self) -> Optional[Any]:
        res = self._model.y_rotations_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def coef_(self) -> Optional[Any]:
        res = self._model.coef_
        return bytes_to_f64_2d(*res) if res is not None else None

    @property
    def intercept_(self) -> Optional[Any]:
        raw = self._model.intercept_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy() if np is not None else list(raw)

    @property
    def n_iter_(self) -> Optional[Any]:
        return self._model.n_iter_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


__all__ = [
    "NMF",
    "PLSRegression",
    "CCA",
]
