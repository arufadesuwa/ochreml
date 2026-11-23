"""
OchreML Kernel Support Vector Machines & Kernel PCA Module (＾▽＾)
Provides Scikit-Learn-compatible SVC, SVR, OneClassSVM, and KernelPCA (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    KernelPCA as _RustKernelPCA,
    OneClassSVM as _RustOneClassSVM,
    SVC as _RustSVC,
    SVR as _RustSVR,
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
    if not is_buf:
        X_data = np.asarray(X_data, dtype=np.float64)
    return np.ascontiguousarray(X_data, dtype=np.float64)


def _to_f64_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if not is_buf:
        y_data = np.asarray(y_data, dtype=np.float64)
    return np.ascontiguousarray(y_data, dtype=np.float64)


def _parse_gamma(gamma: Union[str, float]) -> Tuple[str, Optional[float]]:
    if isinstance(gamma, str):
        return gamma, None
    if isinstance(gamma, (int, float)) and not isinstance(gamma, bool):
        return "scale", float(gamma)
    raise ValueError(f"gamma must be 'scale', 'auto', or a positive float, got {gamma!r} (・`ω´・)")


class SVC:
    """
    C-Support Vector Classification (＾▽＾).

    Implemented with LIBSVM-style WSS3 Sequential Minimal Optimization (SMO),
    One-vs-One multiclass decomposition, and optional Platt scaling (*≧ω≦*).

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Must be strictly positive.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float, default=0.0
        Independent term in kernel function ('poly' and 'sigmoid').
    probability : bool, default=False
        Whether to enable probability estimates via Platt scaling.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    random_state : int or None, default=None
        Controls pseudo random number generation for probability calibration.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustSVC

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        probability: bool = False,
        tol: float = 1e-3,
        max_iter: int = -1,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = _resolve_device(device)

        gamma_mode, gamma_val = _parse_gamma(gamma)
        max_it_opt: Optional[int]
        if max_iter == -1:
            max_it_opt = None
        elif max_iter >= 0:
            max_it_opt = int(max_iter)
        else:
            raise ValueError(f"max_iter must be -1 or >= 1, got {max_iter} (・`ω´・)")

        self._model = _RustSVC(
            c=float(C),
            kernel=str(kernel),
            degree=int(degree),
            gamma_mode=gamma_mode,
            gamma=gamma_val,
            coef0=float(coef0),
            probability=bool(probability),
            tol=float(tol),
            max_iter=max_it_opt,
            random_state=int(random_state) if random_state is not None else None,
            device=self.device,
        )

    def fit(self, X: Any, y: Any) -> "SVC":
        """Fit the SVM model according to the given training data (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = _to_f64_1d(y)
        self._model.fit(X_arr, y_arr)
        return self

    def decision_function(self, X: Any) -> Any:
        """Evaluate the decision function for the samples in X (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.decision_function(X_arr)
        out = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return out.ravel()
        return out

    def predict(self, X: Any) -> Any:
        """Perform classification on samples in X (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        raw = self._model.predict(X_arr)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def predict_proba(self, X: Any) -> Any:
        """Compute probabilities of possible outcomes for samples in X (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_proba(X_arr)
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, y: Any) -> float:
        """Return the mean accuracy on the given test data and labels (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = _to_f64_1d(y)
        return float(self._model.score(X_arr, y_arr))

    @property
    def support_(self) -> Optional[Any]:
        s = self._model.support_
        return np.asarray(s, dtype=np.int64) if s is not None else None

    @property
    def support_vectors_(self) -> Optional[Any]:
        res = self._model.support_vectors_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def n_support_(self) -> Optional[Any]:
        ns = self._model.n_support_
        return np.asarray(ns, dtype=np.int64) if ns is not None else None

    @property
    def dual_coef_(self) -> Optional[Any]:
        res = self._model.dual_coef_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def intercept_(self) -> Optional[Any]:
        raw = self._model.intercept_
        return np.frombuffer(raw, dtype=np.float64).copy() if raw is not None else None

    @property
    def classes_(self) -> Optional[Any]:
        raw = self._model.classes_
        return np.frombuffer(raw, dtype=np.float64).copy() if raw is not None else None

    @property
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def to(self, device: Optional[str]) -> "SVC":
        """Move model compute target to another device (・`ω´・)."""
        self._model.set_device(device)
        self.device = device if device is not None else "auto"
        return self

    def save(self, path: str) -> None:
        """Save the fitted model to a '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SVC":
        """Load a model from a '.dit' file (*≧ω≦*)."""
        rust_model = _RustSVC.load(validate_dit_path(path), device=device)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or "auto"
        obj.C = 1.0
        obj.kernel = "rbf"
        obj.degree = 3
        obj.gamma = "scale"
        obj.coef0 = 0.0
        obj.probability = False
        obj.tol = 1e-3
        obj.max_iter = -1
        obj.random_state = None
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "probability": self.probability,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "device": self.device,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.C = state["C"]
        self.kernel = state["kernel"]
        self.degree = state["degree"]
        self.gamma = state["gamma"]
        self.coef0 = state["coef0"]
        self.probability = state["probability"]
        self.tol = state["tol"]
        self.max_iter = state["max_iter"]
        self.random_state = state["random_state"]
        self.device = state["device"]
        self._model = _RustSVC.from_bytes(state["bytes"], device=self.device)


class SVR:
    """
    Epsilon-Support Vector Regression (＾▽＾).

    Implemented with 2N LIBSVM-style dual Sequential Minimal Optimization (SMO)
    and hardware-accelerated kernel Gram matrices (*≧ω≦*).

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float, default=0.0
        Independent term in kernel function ('poly' and 'sigmoid').
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    C : float, default=1.0
        Regularization parameter. Must be strictly positive.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustSVR

    def __init__(
        self,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        max_iter: int = -1,
        device: Optional[str] = None,
    ) -> None:
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.device = _resolve_device(device)

        gamma_mode, gamma_val = _parse_gamma(gamma)
        max_it_opt: Optional[int]
        if max_iter == -1:
            max_it_opt = None
        elif max_iter >= 0:
            max_it_opt = int(max_iter)
        else:
            raise ValueError(f"max_iter must be -1 or >= 1, got {max_iter} (・`ω´・)")

        self._model = _RustSVR(
            kernel=str(kernel),
            degree=int(degree),
            gamma_mode=gamma_mode,
            gamma=gamma_val,
            coef0=float(coef0),
            tol=float(tol),
            c=float(C),
            epsilon=float(epsilon),
            max_iter=max_it_opt,
            device=self.device,
        )

    def fit(self, X: Any, y: Any) -> "SVR":
        """Fit the SVR model according to the given training data (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = _to_f64_1d(y)
        self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X: Any) -> Any:
        """Perform regression on samples in X (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        raw = self._model.predict(X_arr)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        y_arr = _to_f64_1d(y)
        return float(self._model.score(X_arr, y_arr))

    @property
    def support_(self) -> Optional[Any]:
        s = self._model.support_
        return np.asarray(s, dtype=np.int64) if s is not None else None

    @property
    def support_vectors_(self) -> Optional[Any]:
        res = self._model.support_vectors_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def dual_coef_(self) -> Optional[Any]:
        res = self._model.dual_coef_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def intercept_(self) -> Optional[Any]:
        raw = self._model.intercept_
        return np.frombuffer(raw, dtype=np.float64).copy() if raw is not None else None

    @property
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def to(self, device: Optional[str]) -> "SVR":
        """Move model compute target to another device (・`ω´・)."""
        self._model.set_device(device)
        self.device = device if device is not None else "auto"
        return self

    def save(self, path: str) -> None:
        """Save the fitted model to a '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SVR":
        """Load a model from a '.dit' file (*≧ω≦*)."""
        rust_model = _RustSVR.load(validate_dit_path(path), device=device)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or "auto"
        obj.kernel = "rbf"
        obj.degree = 3
        obj.gamma = "scale"
        obj.coef0 = 0.0
        obj.tol = 1e-3
        obj.C = 1.0
        obj.epsilon = 0.1
        obj.max_iter = -1
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "C": self.C,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "device": self.device,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.kernel = state["kernel"]
        self.degree = state["degree"]
        self.gamma = state["gamma"]
        self.coef0 = state["coef0"]
        self.tol = state["tol"]
        self.C = state["C"]
        self.epsilon = state["epsilon"]
        self.max_iter = state["max_iter"]
        self.device = state["device"]
        self._model = _RustSVR.from_bytes(state["bytes"], device=self.device)


class OneClassSVM:
    """
    Unsupervised Outlier Detection using Scholkopf One-Class SVM (＾▽＾).

    Estimates the support of a high-dimensional distribution using dual SMO
    and hardware-accelerated kernel Gram matrices (*≧ω≦*).

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float, default=0.0
        Independent term in kernel function ('poly' and 'sigmoid').
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    nu : float, default=0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Must be in (0, 1].
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustOneClassSVM

    def __init__(
        self,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        nu: float = 0.5,
        max_iter: int = -1,
        device: Optional[str] = None,
    ) -> None:
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.max_iter = max_iter
        self.device = _resolve_device(device)

        gamma_mode, gamma_val = _parse_gamma(gamma)
        max_it_opt: Optional[int]
        if max_iter == -1:
            max_it_opt = None
        elif max_iter >= 0:
            max_it_opt = int(max_iter)
        else:
            raise ValueError(f"max_iter must be -1 or >= 1, got {max_iter} (・`ω´・)")

        self._model = _RustOneClassSVM(
            kernel=str(kernel),
            degree=int(degree),
            gamma_mode=gamma_mode,
            gamma=gamma_val,
            coef0=float(coef0),
            tol=float(tol),
            nu=float(nu),
            max_iter=max_it_opt,
            device=self.device,
        )

    def fit(self, X: Any) -> "OneClassSVM":
        """Detect the soft boundary of the set of samples X (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        self._model.fit(X_arr)
        return self

    def score_samples(self, X: Any) -> Any:
        """Raw scoring function of the samples (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        raw = self._model.score_samples(X_arr)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def decision_function(self, X: Any) -> Any:
        """Signed distance to the separating hyperplane (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        raw = self._model.decision_function(X_arr)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def predict(self, X: Any) -> Any:
        """Perform classification on samples in X (+1 for inliers, -1 for outliers) (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        raw = self._model.predict(X_arr)
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def support_(self) -> Optional[Any]:
        s = self._model.support_
        return np.asarray(s, dtype=np.int64) if s is not None else None

    @property
    def support_vectors_(self) -> Optional[Any]:
        res = self._model.support_vectors_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def dual_coef_(self) -> Optional[Any]:
        res = self._model.dual_coef_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def intercept_(self) -> Optional[Any]:
        raw = self._model.intercept_
        return np.frombuffer(raw, dtype=np.float64).copy() if raw is not None else None

    @property
    def offset_(self) -> float:
        return float(self._model.offset_)

    @property
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def to(self, device: Optional[str]) -> "OneClassSVM":
        """Move model compute target to another device (・`ω´・)."""
        self._model.set_device(device)
        self.device = device if device is not None else "auto"
        return self

    def save(self, path: str) -> None:
        """Save the fitted model to a '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "OneClassSVM":
        """Load a model from a '.dit' file (*≧ω≦*)."""
        rust_model = _RustOneClassSVM.load(validate_dit_path(path), device=device)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or "auto"
        obj.kernel = "rbf"
        obj.degree = 3
        obj.gamma = "scale"
        obj.coef0 = 0.0
        obj.tol = 1e-3
        obj.nu = 0.5
        obj.max_iter = -1
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "nu": self.nu,
            "max_iter": self.max_iter,
            "device": self.device,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.kernel = state["kernel"]
        self.degree = state["degree"]
        self.gamma = state["gamma"]
        self.coef0 = state["coef0"]
        self.tol = state["tol"]
        self.nu = state["nu"]
        self.max_iter = state["max_iter"]
        self.device = state["device"]
        self._model = _RustOneClassSVM.from_bytes(state["bytes"], device=self.device)


class KernelPCA:
    """
    Kernel Principal Component Analysis (KPCA) (＾▽＾).

    Non-linear dimensionality reduction through the use of kernels and
    double-centered Gram matrix eigendecomposition (*≧ω≦*).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components. If None, all non-zero components are kept.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='linear'
        Kernel used for PCA.
    gamma : float or None, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels.
        If None, defaults to ``1.0 / n_features``.
    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, default=1.0
        Independent term in poly and sigmoid kernels.
    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels via KernelRidge.
    alpha : float, default=1.0
        Hyperparameter of the ridge regression that learns the inverse transform.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    eigenvalues_ : ndarray of shape (n_components,)
        Eigenvalues of the centered kernel matrix in decreasing order.
    eigenvectors_ : ndarray of shape (n_samples, n_components)
        Eigenvectors of the centered kernel matrix.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustKernelPCA

    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: str = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        fit_inverse_transform: bool = False,
        alpha: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.fit_inverse_transform = fit_inverse_transform
        self.alpha = alpha
        self.device = _resolve_device(device)
        self._model = _RustKernelPCA(
            n_components=int(n_components) if n_components is not None else None,
            kernel=str(kernel),
            gamma=float(gamma) if gamma is not None else None,
            degree=int(degree),
            coef0=float(coef0),
            fit_inverse_transform=bool(fit_inverse_transform),
            alpha=float(alpha),
            device=self.device,
        )

    def fit(self, X: Any) -> "KernelPCA":
        """Fit the model from data in X (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        self._model.fit(X_arr)
        return self

    def transform(self, X: Any) -> Any:
        """Transform X into the kernel principal component space (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.transform(X_arr)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any) -> Any:
        """Fit the model from data in X and transform X (＾▽＾)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.fit_transform(X_arr)
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> Any:
        """Transform data back to original space (requires fit_inverse_transform=True) (*≧ω≦*)."""
        X_arr = _to_f64_2d(X)
        rows, cols, raw = self._model.inverse_transform(X_arr)
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def eigenvalues_(self) -> Optional[Any]:
        raw = self._model.eigenvalues_
        return np.frombuffer(raw, dtype=np.float64).copy() if raw is not None else None

    @property
    def eigenvectors_(self) -> Optional[Any]:
        res = self._model.eigenvectors_
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

    def to(self, device: Optional[str]) -> "KernelPCA":
        """Move model compute target to another device (・`ω´・)."""
        self._model.set_device(device)
        self.device = device if device is not None else "auto"
        return self

    def save(self, path: str) -> None:
        """Save the fitted model to a '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KernelPCA":
        """Load a model from a '.dit' file (*≧ω≦*)."""
        rust_model = _RustKernelPCA.load(validate_dit_path(path), device=device)
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj.device = rust_model.device_ or "auto"
        obj.n_components = None
        obj.kernel = "linear"
        obj.gamma = None
        obj.degree = 3
        obj.coef0 = 1.0
        obj.fit_inverse_transform = False
        obj.alpha = 1.0
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "n_components": self.n_components,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "fit_inverse_transform": self.fit_inverse_transform,
            "alpha": self.alpha,
            "device": self.device,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.n_components = state["n_components"]
        self.kernel = state["kernel"]
        self.gamma = state["gamma"]
        self.degree = state["degree"]
        self.coef0 = state["coef0"]
        self.fit_inverse_transform = state["fit_inverse_transform"]
        self.alpha = state["alpha"]
        self.device = state["device"]
        self._model = _RustKernelPCA.from_bytes(state["bytes"], device=self.device)
