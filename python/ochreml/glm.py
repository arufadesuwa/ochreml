"""
OchreML Robust & Generalized Linear Models Submodule (＾▽＾)
Provides Scikit-Learn-compatible HuberRegressor, QuantileRegressor, PoissonRegressor,
GammaRegressor, and TweedieRegressor accelerated by Rust hardware backends (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    GammaRegressor as _RustGammaRegressor,
    HuberRegressor as _RustHuberRegressor,
    PoissonRegressor as _RustPoissonRegressor,
    QuantileRegressor as _RustQuantileRegressor,
    TweedieRegressor as _RustTweedieRegressor,
)
from ._preproc_utils import validate_dit_path
from ._utils import process_features, process_sample_weight, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_2d(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    if np is not None:
        return np.ascontiguousarray(X_data, dtype=np.float64)
    raise RuntimeError("NumPy is required for buffer conversion ( >_< )")


def _to_f64_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if is_buf:
        return y_data
    if np is not None:
        return np.ascontiguousarray(y_data, dtype=np.float64).ravel()
    raise RuntimeError("NumPy is required for buffer conversion ( >_< )")


def _to_f64_sw(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if is_buf:
        return sw_data
    if np is not None:
        return np.ascontiguousarray(sw_data, dtype=np.float64).ravel()
    raise RuntimeError("NumPy is required for buffer conversion ( >_< )")


class _BaseGLM:
    """Shared wrapper mechanics for Robust & Generalized Linear Models (*≧ω≦*)."""

    _rust_cls: Any = None
    _param_names: tuple[str, ...] = ()

    def _sync_params_from_rust(self) -> None:
        for name in self._param_names:
            setattr(self, name, getattr(self._model, name))
        self.device = self._model.device_

    @classmethod
    def _from_rust(cls, rust_obj: Any, device: str) -> Any:
        inst = cls.__new__(cls)
        inst._model = rust_obj
        for name in cls._param_names:
            setattr(inst, name, getattr(rust_obj, name))
        inst.device = rust_obj.device_ or device
        return inst

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """Fit the model according to the given training data (o´∀｀o)."""
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_f64_sw(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the linear model (*≧ω≦*)."""
        X_data = _to_f64_2d(X)
        raw = self._model.predict_buffer(X_data)
        if np is not None:
            return np.frombuffer(raw, dtype=np.float64).copy()
        import array

        return list(array.array("d", raw))

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """Return the goodness-of-fit score on the given test data and labels (＾▽＾)."""
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_f64_sw(sample_weight)
        return float(self._model.score_buffer(X_data, y_data, sw_data))

    def save(self, path: str) -> None:
        """Save model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> Any:
        """Move model execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = self._rust_cls.from_bytes(state, target_dev)
        self._sync_params_from_rust()
        if not self.device:
            self.device = target_dev

    @property
    def coef_(self) -> Optional[Any]:
        c = self._model.coef_
        if c is None:
            return None
        return np.asarray(c, dtype=np.float64) if np is not None else c

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class HuberRegressor(_BaseGLM):
    """
    L2-regularized linear regression model that is robust to outliers (＾▽＾)

    Optimizes Owen's Huber loss jointly over weights, intercept, and scale
    using hardware-accelerated L-BFGS (*≧ω≦*).

    Parameters
    ----------
    epsilon : float, default=1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. Must be >= 1.0.
    max_iter : int, default=100
        Maximum number of iterations that ``L-BFGS`` should run for.
    alpha : float, default=0.0001
        Strength of the squared L2 regularization. Must be >= 0.0.
    fit_intercept : bool, default=True
        Whether or not to fit the intercept.
    tol : float, default=1e-5
        Tolerance for stopping criterion.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Features got by optimizing the L2-regularized Huber loss.
    intercept_ : float
        Bias term.
    scale_ : float
        The value by which ``|y - X @ coef_ - intercept_|`` is scaled.
    outliers_ : ndarray of shape (n_samples,) of bool
        Boolean mask indicating whether each training sample is an outlier.
    n_iter_ : int
        Number of iterations that L-BFGS ran for.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustHuberRegressor
    _param_names = ("epsilon", "max_iter", "alpha", "fit_intercept", "tol")

    def __init__(
        self,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        tol: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.tol = float(tol)
        self.device = _resolve_device(device)
        self._model = _RustHuberRegressor(
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            device=self.device,
        )

    @property
    def scale_(self) -> float:
        return float(self._model.scale_)

    @property
    def outliers_(self) -> Optional[Any]:
        o = self._model.outliers_
        if o is None:
            return None
        return np.asarray(o, dtype=bool) if np is not None else o


class QuantileRegressor(_BaseGLM):
    """
    Linear regression model that predicts conditional quantiles (*≧ω≦*)

    Minimizes the sample-weighted pinball loss plus an L1 penalty
    ``alpha * ||coef_||_1`` via IRLS and warm-started Coordinate Descent (o´∀｀o).

    Parameters
    ----------
    quantile : float, default=0.5
        The quantile that the model tries to predict. Must be strictly in (0, 1).
    alpha : float, default=1.0
        Regularization constant that multiplies the L1 penalty term.
    fit_intercept : bool, default=True
        Whether or not to fit the intercept.
    max_iter : int, default=1000
        Maximum number of IRLS iterations.
    tol : float, default=1e-6
        Tolerance for stopping criterion.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the features.
    intercept_ : float
        The intercept of the model, aka bias term.
    n_iter_ : int
        Number of IRLS iterations performed.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustQuantileRegressor
    _param_names = ("quantile", "alpha", "fit_intercept", "max_iter", "tol")

    def __init__(
        self,
        quantile: float = 0.5,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.quantile = float(quantile)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = _resolve_device(device)
        self._model = _RustQuantileRegressor(
            quantile=self.quantile,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )


class TweedieRegressor(_BaseGLM):
    """
    Generalized Linear Model with a Tweedie distribution (＾▽＾)

    Optimizes L2-penalized Tweedie deviance with identity or log link
    via hardware-accelerated L-BFGS (*≧ω≦*).

    Parameters
    ----------
    power : float, default=0.0
        The power determines the underlying target distribution.
        0 = Normal, 1 = Poisson, (1, 2) = Compound Poisson-Gamma, 2 = Gamma.
        Must not be in (0, 1).
    alpha : float, default=1.0
        Constant that multiplies the L2 penalty term. Must be >= 0.0.
    fit_intercept : bool, default=True
        Specifies if a constant (bias or intercept) should be added.
    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, mapping from linear predictor to mean.
    max_iter : int, default=100
        The maximal number of iterations for the L-BFGS solver.
    tol : float, default=1e-4
        Stopping criterion for the L-BFGS solver.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear predictor.
    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.
    n_iter_ : int
        Actual number of iterations used in the solver.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustTweedieRegressor
    _param_names = ("power", "alpha", "fit_intercept", "link", "max_iter", "tol")

    def __init__(
        self,
        power: float = 0.0,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        link: str = "auto",
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.power = float(power)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.link = str(link)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = _resolve_device(device)
        self._model = _RustTweedieRegressor(
            power=self.power,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            link=self.link,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )


class PoissonRegressor(_BaseGLM):
    """
    Generalized Linear Model with a Poisson distribution and log link (o´∀｀o)

    Equivalent to ``TweedieRegressor(power=1.0, link='log')`` (*≧ω≦*).

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L2 penalty term. Must be >= 0.0.
    fit_intercept : bool, default=True
        Specifies if a constant (bias or intercept) should be added.
    max_iter : int, default=100
        The maximal number of iterations for the L-BFGS solver.
    tol : float, default=1e-4
        Stopping criterion for the L-BFGS solver.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear predictor.
    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.
    n_iter_ : int
        Actual number of iterations used in the solver.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustPoissonRegressor
    _param_names = ("alpha", "fit_intercept", "max_iter", "tol")

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = _resolve_device(device)
        self._model = _RustPoissonRegressor(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )


class GammaRegressor(_BaseGLM):
    """
    Generalized Linear Model with a Gamma distribution and log link (＾▽＾)

    Equivalent to ``TweedieRegressor(power=2.0, link='log')`` (*≧ω≦*).

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L2 penalty term. Must be >= 0.0.
    fit_intercept : bool, default=True
        Specifies if a constant (bias or intercept) should be added.
    max_iter : int, default=100
        The maximal number of iterations for the L-BFGS solver.
    tol : float, default=1e-4
        Stopping criterion for the L-BFGS solver.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear predictor.
    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.
    n_iter_ : int
        Actual number of iterations used in the solver.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustGammaRegressor
    _param_names = ("alpha", "fit_intercept", "max_iter", "tol")

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = _resolve_device(device)
        self._model = _RustGammaRegressor(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )


__all__ = [
    "HuberRegressor",
    "QuantileRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
