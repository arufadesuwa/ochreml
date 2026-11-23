"""
OchreML Matrix Decomposition & Dimensionality Reduction Submodule (＾▽＾)
Provides Scikit-Learn-compatible PCA and TruncatedSVD accelerated by Rust hardware backends (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    PCA as _RustPCA,
    TruncatedSVD as _RustTruncatedSVD,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, to_feature_matrix


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_array(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    if np is not None:
        return np.ascontiguousarray(X_data, dtype=np.float64)
    return to_feature_matrix(X)


class PCA:
    """
    Principal Component Analysis (PCA) (＾▽＾)

    Linear dimensionality reduction using Singular Value Decomposition of the
    centered data to project it to a lower dimensional space (*≧ω≦*).

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of components to keep.
        If None, all ``min(n_samples, n_features)`` components are kept.
        If ``0 < n_components < 1``, select the number of components such that
        the amount of variance that needs to be explained is greater than the
        percentage specified by ``n_components``.
    whiten : bool, default=False
        When True, the ``components_`` vectors are multiplied by the square root
        of ``n_samples`` and divided by the singular values to ensure uncorrelated
        outputs with unit component-wise variances.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    components_ : ndarray of shape (n_components_, n_features_in_)
        Principal axes in feature space, representing the directions of maximum
        variance in the data.
    explained_variance_ : ndarray of shape (n_components_,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Percentage of variance explained by each of the selected components.
    singular_values_ : ndarray of shape (n_components_,)
        The singular values corresponding to each of the selected components.
    mean_ : ndarray of shape (n_features_in_,)
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The estimated number of components.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_ : int
        Number of samples in the training data.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model.
    """

    _rust_cls = _RustPCA

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.device = _resolve_device(device)

        n_comp_int: Optional[int] = None
        var_target: Optional[float] = None
        if n_components is not None:
            if isinstance(n_components, bool):
                raise ValueError("n_components cannot be a boolean ( >_< )")
            if isinstance(n_components, float):
                if not (0.0 < n_components < 1.0):
                    raise ValueError(
                        f"n_components as float ({n_components}) must be in (0.0, 1.0) ( >_< )"
                    )
                var_target = float(n_components)
            elif isinstance(n_components, int):
                if n_components < 1:
                    raise ValueError(
                        f"n_components ({n_components}) must be >= 1 ( >_< )"
                    )
                n_comp_int = int(n_components)
            else:
                raise TypeError(
                    f"n_components must be int, float, or None, got {type(n_components)} (・`ω´・)"
                )

        self._model = _RustPCA(
            n_components=n_comp_int,
            explained_variance_ratio_target=var_target,
            whiten=whiten,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustPCA, device: str) -> "PCA":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        if rust_obj.explained_variance_ratio_target is not None:
            inst.n_components = rust_obj.explained_variance_ratio_target
        else:
            inst.n_components = rust_obj.n_components
        inst.whiten = rust_obj.whiten
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Optional[Any] = None) -> "PCA":
        """Fit the model with X (o´∀｀o)."""
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def transform(self, X: Any) -> Any:
        """Apply dimensionality reduction to X (*≧ω≦*)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit the model with X and apply the dimensionality reduction on X (＾▽＾)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> Any:
        """Transform data back to its original space (o´∀｀o)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def get_covariance(self) -> Any:
        """Compute data covariance with the generative model (*≧ω≦*)."""
        rows, cols, raw = self._model.get_covariance_buffer()
        return bytes_to_f64_2d(rows, cols, raw)

    def score_samples(self, X: Any) -> Any:
        """Return the log-likelihood of each sample (＾▽＾)."""
        X_data = _to_f64_array(X)
        raw = self._model.score_samples_buffer(X_data)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Optional[Any] = None) -> float:
        """Return the average log-likelihood of all samples (*≧ω≦*)."""
        X_data = _to_f64_array(X)
        return float(self._model.score_buffer(X_data))

    def save(self, path: str) -> None:
        """Save fitted PCA model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PCA":
        """Load PCA model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "PCA":
        """Move PCA execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustPCA.from_bytes(state, target_dev)
        if self._model.explained_variance_ratio_target is not None:
            self.n_components = self._model.explained_variance_ratio_target
        else:
            self.n_components = self._model.n_components
        self.whiten = self._model.whiten
        self.device = self._model.device_ or target_dev

    @property
    def components_(self) -> Optional[Any]:
        res = self._model.components_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def explained_variance_(self) -> Optional[Any]:
        v = self._model.explained_variance_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def explained_variance_ratio_(self) -> Optional[Any]:
        v = self._model.explained_variance_ratio_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def singular_values_(self) -> Optional[Any]:
        v = self._model.singular_values_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def mean_(self) -> Optional[Any]:
        v = self._model.mean_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def n_components_(self) -> Optional[int]:
        return self._model.n_components_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_samples_(self) -> Optional[int]:
        return self._model.n_samples_

    @property
    def noise_variance_(self) -> Optional[float]:
        return self._model.noise_variance_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class TruncatedSVD:
    """
    Dimensionality reduction using truncated SVD (aka LSA) (*≧ω≦*)

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition (＾▽＾).

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
    algorithm : {'randomized', 'arpack', 'exact'}, default='randomized'
        SVD solver to use.
    n_iter : int, default=5
        Number of iterations for randomized SVD solver.
    random_state : int or None, default=None
        Used during randomized SVD. Pass an int for reproducible results.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features_in_)
        The right singular vectors of the input data.
    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustTruncatedSVD

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustTruncatedSVD(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
            algorithm=algorithm,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustTruncatedSVD, device: str) -> "TruncatedSVD":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.algorithm = rust_obj.algorithm
        inst.n_iter = rust_obj.n_iter
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Optional[Any] = None) -> "TruncatedSVD":
        """Fit model on training data X (o´∀｀o)."""
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def transform(self, X: Any) -> Any:
        """Perform dimensionality reduction on X (*≧ω≦*)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit model to X and perform dimensionality reduction on X (＾▽＾)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> Any:
        """Transform X back to its original space (o´∀｀o)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def save(self, path: str) -> None:
        """Save fitted TruncatedSVD model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TruncatedSVD":
        """Load TruncatedSVD model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "TruncatedSVD":
        """Move TruncatedSVD execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustTruncatedSVD.from_bytes(state, target_dev)
        self.n_components = self._model.n_components
        self.algorithm = self._model.algorithm
        self.n_iter = self._model.n_iter
        self.random_state = self._model.random_state
        self.device = self._model.device_ or target_dev

    @property
    def components_(self) -> Optional[Any]:
        res = self._model.components_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def explained_variance_(self) -> Optional[Any]:
        v = self._model.explained_variance_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def explained_variance_ratio_(self) -> Optional[Any]:
        v = self._model.explained_variance_ratio_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def singular_values_(self) -> Optional[Any]:
        v = self._model.singular_values_
        return np.asarray(v, dtype=np.float64) if (v is not None and np is not None) else v

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


from .cross_decomposition import NMF
from .kernel_svm import KernelPCA

__all__ = [
    "PCA",
    "TruncatedSVD",
    "KernelPCA",
    "NMF",
]
