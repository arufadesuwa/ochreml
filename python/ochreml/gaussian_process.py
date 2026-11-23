"""
OchreML Gaussian Processes Submodule (＾▽＾)
Provides Scikit-Learn-compatible GaussianProcessRegressor and GaussianProcessClassifier
accelerated by Rust hardware backends (N3 pairwise_l2_sq, N2 elementwise_op, E1 matvec/matmul)
with Cholesky factorization, Laplace approximation, and Nelder-Mead hyperparameter optimization (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    GaussianProcessClassifier as _RustGaussianProcessClassifier,
    GaussianProcessRegressor as _RustGaussianProcessRegressor,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_target


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


def _parse_kernel_descriptor(
    kernel: Optional[Union[str, dict, Any]],
    length_scale: float,
    constant_value: float,
    nu: float,
    noise_level: float,
    rq_alpha: float,
) -> Tuple[str, float, float, float, float, float]:
    """Normalize string or simple kernel descriptors into Rust constructor parameters (*≧ω≦*)."""
    if kernel is None:
        return "rbf", float(length_scale), float(constant_value), float(nu), float(noise_level), float(rq_alpha)
    if isinstance(kernel, str):
        return kernel, float(length_scale), float(constant_value), float(nu), float(noise_level), float(rq_alpha)
    if isinstance(kernel, dict):
        k_name = str(kernel.get("kernel", kernel.get("name", "rbf")))
        return (
            k_name,
            float(kernel.get("length_scale", length_scale)),
            float(kernel.get("constant_value", constant_value)),
            float(kernel.get("nu", nu)),
            float(kernel.get("noise_level", noise_level)),
            float(kernel.get("rq_alpha", kernel.get("alpha", rq_alpha))),
        )
    cls_name = type(kernel).__name__.lower()
    if "rbf" in cls_name:
        return (
            "rbf",
            float(getattr(kernel, "length_scale", length_scale)),
            float(getattr(kernel, "constant_value", constant_value)),
            float(nu),
            float(noise_level),
            float(rq_alpha),
        )
    if "matern" in cls_name:
        return (
            "matern",
            float(getattr(kernel, "length_scale", length_scale)),
            float(getattr(kernel, "constant_value", constant_value)),
            float(getattr(kernel, "nu", nu)),
            float(noise_level),
            float(rq_alpha),
        )
    if "white" in cls_name:
        return (
            "white",
            float(length_scale),
            float(constant_value),
            float(nu),
            float(getattr(kernel, "noise_level", noise_level)),
            float(rq_alpha),
        )
    if "rationalquadratic" in cls_name:
        return (
            "rational_quadratic",
            float(getattr(kernel, "length_scale", length_scale)),
            float(getattr(kernel, "constant_value", constant_value)),
            float(nu),
            float(noise_level),
            float(getattr(kernel, "alpha", rq_alpha)),
        )
    raise ValueError(f"Unsupported kernel descriptor: {kernel!r} ( >_< )")


class GaussianProcessRegressor:
    """
    Gaussian Process Regressor (＾▽＾)
    Scikit-Learn-compatible Gaussian Process regression backed by Rust Cholesky
    factorization, Backend N3/N2/E1 primitives, and Nelder-Mead LML optimization (*≧ω≦*).

    Parameters
    ----------
    kernel : str, dict, or None, default='rbf'
        Kernel specification ('rbf', 'matern', 'white', 'rational_quadratic', 'rbf_white').
    length_scale : float, default=1.0
        Length scale of the kernel (> 0).
    constant_value : float, default=1.0
        Constant variance scale of the kernel (> 0).
    nu : float, default=1.5
        Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5).
    alpha : float, default=1e-10
        Value added to the diagonal of the kernel matrix during fitting (>= 0).
    noise_level : float, default=1.0
        Noise variance for 'white' and 'rbf_white' kernels (> 0).
    rq_alpha : float, default=1.0
        Scale mixture parameter for 'rational_quadratic' kernel (> 0).
    optimize_hyperparams : bool, default=False
        Whether to optimize kernel hyperparameters by maximizing log marginal likelihood.
    n_restarts_optimizer : int, default=0
        Number of restarts of the optimizer for finding kernel parameters maximizing LML.
    normalize_y : bool, default=False
        Whether to normalize the target values y by removing mean and scaling to unit variance.
    random_state : int or None, default=None
        Random seed used for optimizer restarts.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        kernel: Optional[Union[str, dict, Any]] = "rbf",
        length_scale: float = 1.0,
        constant_value: float = 1.0,
        nu: float = 1.5,
        alpha: float = 1e-10,
        noise_level: float = 1.0,
        rq_alpha: float = 1.0,
        optimize_hyperparams: bool = False,
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        k_str, ls, cv, nu_v, nl, rqa = _parse_kernel_descriptor(
            kernel, length_scale, constant_value, nu, noise_level, rq_alpha
        )
        self.kernel = k_str
        self.length_scale = ls
        self.constant_value = cv
        self.nu = nu_v
        self.alpha = float(alpha)
        self.noise_level = nl
        self.rq_alpha = rqa
        self.optimize_hyperparams = bool(optimize_hyperparams)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.normalize_y = bool(normalize_y)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustGaussianProcessRegressor(
            kernel=self.kernel,
            length_scale=self.length_scale,
            constant_value=self.constant_value,
            nu=self.nu,
            alpha=self.alpha,
            noise_level=self.noise_level,
            rq_alpha=self.rq_alpha,
            optimize_hyperparams=self.optimize_hyperparams,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.kernel = str(self._model.kernel)
        self.length_scale = float(self._model.length_scale)
        self.constant_value = float(self._model.constant_value)
        self.nu = float(self._model.nu)
        self.alpha = float(self._model.alpha)
        self.noise_level = float(self._model.noise_level)
        self.rq_alpha = float(self._model.rq_alpha)
        self.optimize_hyperparams = bool(self._model.optimize_hyperparams)
        self.n_restarts_optimizer = int(self._model.n_restarts_optimizer)
        self.normalize_y = bool(self._model.normalize_y)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustGaussianProcessRegressor, device: str
    ) -> "GaussianProcessRegressor":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "GaussianProcessRegressor":
        """
        Fit Gaussian process regression model (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GaussianProcessRegressor
            Fitted estimator.
        """
        self._model.fit_buffer(_to_f64_2d(X), _to_f64_1d(y))
        return self

    def predict(
        self,
        X: Any,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> Any:
        """
        Predict using the Gaussian process regression model (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution is returned.
        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution is returned.

        Returns
        -------
        y_mean : numpy.ndarray of shape (n_samples,)
            Mean of predictive distribution at query points.
        y_std : numpy.ndarray of shape (n_samples,), optional
            Standard deviation of predictive distribution at query points.
        y_cov : numpy.ndarray of shape (n_samples, n_samples), optional
            Covariance of joint predictive distribution at query points.
        """
        mean_bytes, std_bytes, cov_tuple = self._model.predict_buffer(
            _to_f64_2d(X), return_std=bool(return_std), return_cov=bool(return_cov)
        )
        y_mean = np.frombuffer(mean_bytes, dtype=np.float64).copy()
        if return_std and std_bytes is not None:
            y_std = np.frombuffer(std_bytes, dtype=np.float64).copy()
            return y_mean, y_std
        if return_cov and cov_tuple is not None:
            rows, cols, cov_bytes = cov_tuple
            y_cov = bytes_to_f64_2d(rows, cols, cov_bytes)
            return y_mean, y_cov
        return y_mean

    def log_marginal_likelihood(self) -> float:
        """Return log-marginal likelihood of the fitted hyperparameters (＾▽＾)."""
        return float(self._model.log_marginal_likelihood())

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (o´∀｀o)."""
        return float(self._model.score_buffer(_to_f64_2d(X), _to_f64_1d(y)))

    def save(self, path: str) -> None:
        """Save the fitted GaussianProcessRegressor to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "GaussianProcessRegressor":
        """Load GaussianProcessRegressor from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustGaussianProcessRegressor.load(validate_dit_path(path), target_dev),
            target_dev,
        )

    def to(self, device: str) -> "GaussianProcessRegressor":
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustGaussianProcessRegressor.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    @property
    def length_scale_(self) -> float:
        return float(self._model.length_scale_)

    @property
    def constant_value_(self) -> float:
        return float(self._model.constant_value_)

    @property
    def noise_level_(self) -> float:
        return float(self._model.noise_level_)

    @property
    def rq_alpha_(self) -> float:
        return float(self._model.rq_alpha_)

    @property
    def y_train_mean_(self) -> float:
        return float(self._model.y_train_mean_)

    @property
    def y_train_std_(self) -> float:
        return float(self._model.y_train_std_)

    @property
    def alpha_(self) -> Optional[Any]:
        a = self._model.alpha_
        return None if a is None else np.asarray(a, dtype=np.float64)

    @property
    def X_train_(self) -> Optional[Any]:
        res = self._model.x_train_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def L_(self) -> Optional[Any]:
        res = self._model.l_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def log_marginal_likelihood_value_(self) -> Optional[float]:
        return self._model.log_marginal_likelihood_value_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class GaussianProcessClassifier:
    """
    Gaussian Process Classifier (＾▽＾)
    Scikit-Learn-compatible Gaussian Process classification via Laplace approximation
    (binary) and One-vs-Rest (multiclass) backed by Rust hardware backends (*≧ω≦*).

    Parameters
    ----------
    kernel : str, dict, or None, default='rbf'
        Kernel specification ('rbf', 'matern', 'white', 'rational_quadratic', 'rbf_white').
    length_scale : float, default=1.0
        Length scale of the kernel (> 0).
    constant_value : float, default=1.0
        Constant variance scale of the kernel (> 0).
    nu : float, default=1.5
        Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5).
    noise_level : float, default=1.0
        Noise variance for 'white' and 'rbf_white' kernels (> 0).
    rq_alpha : float, default=1.0
        Scale mixture parameter for 'rational_quadratic' kernel (> 0).
    max_iter_predict : int, default=100
        Maximum number of Newton iterations during Laplace approximation.
    optimize_hyperparams : bool, default=False
        Whether to optimize kernel hyperparameters by maximizing log marginal likelihood.
    n_restarts_optimizer : int, default=0
        Number of restarts of the optimizer for finding kernel parameters maximizing LML.
    random_state : int or None, default=None
        Random seed used for optimizer restarts.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        kernel: Optional[Union[str, dict, Any]] = "rbf",
        length_scale: float = 1.0,
        constant_value: float = 1.0,
        nu: float = 1.5,
        noise_level: float = 1.0,
        rq_alpha: float = 1.0,
        max_iter_predict: int = 100,
        optimize_hyperparams: bool = False,
        n_restarts_optimizer: int = 0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        k_str, ls, cv, nu_v, nl, rqa = _parse_kernel_descriptor(
            kernel, length_scale, constant_value, nu, noise_level, rq_alpha
        )
        self.kernel = k_str
        self.length_scale = ls
        self.constant_value = cv
        self.nu = nu_v
        self.noise_level = nl
        self.rq_alpha = rqa
        self.max_iter_predict = int(max_iter_predict)
        self.optimize_hyperparams = bool(optimize_hyperparams)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustGaussianProcessClassifier(
            kernel=self.kernel,
            length_scale=self.length_scale,
            constant_value=self.constant_value,
            nu=self.nu,
            noise_level=self.noise_level,
            rq_alpha=self.rq_alpha,
            max_iter_predict=self.max_iter_predict,
            optimize_hyperparams=self.optimize_hyperparams,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.kernel = str(self._model.kernel)
        self.length_scale = float(self._model.length_scale)
        self.constant_value = float(self._model.constant_value)
        self.nu = float(self._model.nu)
        self.noise_level = float(self._model.noise_level)
        self.rq_alpha = float(self._model.rq_alpha)
        self.max_iter_predict = int(self._model.max_iter_predict)
        self.optimize_hyperparams = bool(self._model.optimize_hyperparams)
        self.n_restarts_optimizer = int(self._model.n_restarts_optimizer)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustGaussianProcessClassifier, device: str
    ) -> "GaussianProcessClassifier":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "GaussianProcessClassifier":
        """
        Fit Gaussian process classification model (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : GaussianProcessClassifier
            Fitted estimator.
        """
        self._model.fit_buffer(_to_f64_2d(X), _to_f64_1d(y))
        return self

    def predict(self, X: Any) -> Any:
        """
        Perform classification on an array of test vectors X (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples,)
            Predicted target class labels for X.
        """
        raw = self._model.predict_buffer(_to_f64_2d(X))
        return np.frombuffer(raw, dtype=np.float64).copy()

    def predict_proba(self, X: Any) -> Any:
        """
        Return probability estimates for the test vector X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Probability of the samples for each class in the model.
        """
        rows, cols, raw = self._model.predict_proba_buffer(_to_f64_2d(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def log_marginal_likelihood(self) -> float:
        """Return log-marginal likelihood of the fitted hyperparameters (＾▽＾)."""
        return float(self._model.log_marginal_likelihood())

    def score(self, X: Any, y: Any) -> float:
        """Return the mean accuracy on the given test data and labels (o´∀｀o)."""
        return float(self._model.score_buffer(_to_f64_2d(X), _to_f64_1d(y)))

    def save(self, path: str) -> None:
        """Save the fitted GaussianProcessClassifier to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(
        cls, path: str, device: Optional[str] = None
    ) -> "GaussianProcessClassifier":
        """Load GaussianProcessClassifier from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustGaussianProcessClassifier.load(validate_dit_path(path), target_dev),
            target_dev,
        )

    def to(self, device: str) -> "GaussianProcessClassifier":
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustGaussianProcessClassifier.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return None if c is None else np.asarray(c, dtype=np.float64)

    @property
    def n_classes_(self) -> Optional[int]:
        return self._model.n_classes_

    @property
    def log_marginal_likelihood_value_(self) -> Optional[float]:
        return self._model.log_marginal_likelihood_value_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_
