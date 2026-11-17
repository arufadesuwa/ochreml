"""
OchreML Feature Engineering Estimators (*≧ω≦*)
Implements PolynomialFeatures, SplineTransformer, KernelCenterer, and add_dummy_feature.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from .._ochreml import (
    KernelCenterer as _RustKernelCenterer,
    PolynomialFeatures as _RustPolynomialFeatures,
    SplineTransformer as _RustSplineTransformer,
)
from .._preproc_utils import bytes_to_f64_2d, validate_dit_path
from .._utils import process_features


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


class _BaseFeatureEng:
    """Shared persistence & buffer mixin for feature engineering transformers (＾▽＾)."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def _fit_x(self, X: Any) -> np.ndarray:
        is_buf, X_data = process_features(X)
        return X_data if is_buf else np.asarray(X_data, dtype=np.float64)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it (*≧ω≦*)."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save transformer to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load transformer from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> Any:
        """Set execution device target (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class PolynomialFeatures(_BaseFeatureEng):
    """
    Generate polynomial and interaction features (＾▽＾)

    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        The maximal degree (or (min, max) degree range) of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=True
        If True, include a bias column (power zero).
    order : {'C', 'F'}, default='C'
        Order of output array in the dense case.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustPolynomialFeatures

    def __init__(
        self,
        degree: Union[int, Tuple[int, int]] = 2,
        *,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: str = "C",
        device: Optional[str] = None,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.device = _resolve_device(device)

        if isinstance(degree, tuple):
            min_d, max_d = int(degree[0]), int(degree[1])
        else:
            min_d, max_d = 0, int(degree)

        self._model = _RustPolynomialFeatures(
            min_degree=min_d,
            max_degree=max_d,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustPolynomialFeatures, device: str) -> "PolynomialFeatures":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.degree = (
            rust_obj.max_degree
            if rust_obj.min_degree == 0
            else (rust_obj.min_degree, rust_obj.max_degree)
        )
        inst.interaction_only = rust_obj.interaction_only
        inst.include_bias = rust_obj.include_bias
        inst.order = rust_obj.order
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustPolynomialFeatures.from_bytes(state, target_dev)
        self.degree = (
            self._model.max_degree
            if self._model.min_degree == 0
            else (self._model.min_degree, self._model.max_degree)
        )
        self.interaction_only = self._model.interaction_only
        self.include_bias = self._model.include_bias
        self.order = self._model.order
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "PolynomialFeatures":
        """Compute number of output features (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform data to polynomial features (o´∀｀o)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw, order=self.order)

    @property
    def n_output_features_(self) -> Optional[int]:
        return self._model.n_output_features_

    @property
    def powers_(self) -> Optional[np.ndarray]:
        p = self._model.powers_
        n_out = self._model.n_output_features_
        n_in = self._model.n_features_in_
        if p is None or n_out is None or n_in is None:
            return None
        return np.asarray(p, dtype=np.int64).reshape((n_out, n_in))

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        powers = self.powers_
        if powers is None or self.n_features_in_ is None:
            raise ValueError("PolynomialFeatures is not fitted yet (´-ω-`)")
        in_names = (
            [f"x{i}" for i in range(self.n_features_in_)]
            if input_features is None
            else list(input_features)
        )
        names: List[str] = []
        for row in powers:
            if not np.any(row):
                names.append("1")
                continue
            parts = []
            for name, exp in zip(in_names, row):
                if exp == 1:
                    parts.append(str(name))
                elif exp > 1:
                    parts.append(f"{name}^{exp}")
            names.append(" ".join(parts))
        return np.asarray(names, dtype=object)

    def __repr__(self) -> str:
        return (
            f"PolynomialFeatures(degree={self.degree}, interaction_only={self.interaction_only}, "
            f"include_bias={self.include_bias}) (＾▽＾)"
        )


class SplineTransformer(_BaseFeatureEng):
    """
    Generate univariate B-spline bases for features (o´∀｀o)

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots of the splines.
    degree : int, default=3
        The polynomial degree of the spline basis.
    knots : {'uniform', 'quantile'} or array-like, default='uniform'
        Set knot positions.
    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'}, default='constant'
        Extrapolation strategy beyond min and max knots.
    include_bias : bool, default=True
        If True, include the first spline basis column.
    order : {'C', 'F'}, default='C'
        Order of output array.
    sparse_output : bool, default=False
        Whether to return a scipy.sparse.csr_matrix.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustSplineTransformer

    def __init__(
        self,
        n_knots: int = 5,
        degree: int = 3,
        *,
        knots: Union[str, Any] = "uniform",
        extrapolation: str = "constant",
        include_bias: bool = True,
        order: str = "C",
        sparse_output: bool = False,
        device: Optional[str] = None,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.sparse_output = sparse_output
        self.device = _resolve_device(device)

        if isinstance(knots, str):
            knots_mode = knots
            self._custom_knots = None
            n_knots_eff = n_knots
        else:
            ck = np.ascontiguousarray(knots, dtype=np.float64)
            if ck.ndim != 2:
                raise ValueError("Custom knots array must be 2D (n_knots, n_features) ( >_< )")
            knots_mode = "custom"
            self._custom_knots = ck
            n_knots_eff = ck.shape[0]

        self._model = _RustSplineTransformer(
            n_knots=n_knots_eff,
            degree=degree,
            knots_mode=knots_mode,
            extrapolation=extrapolation,
            include_bias=include_bias,
            order=order,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustSplineTransformer, device: str) -> "SplineTransformer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_knots = rust_obj.n_knots
        inst.degree = rust_obj.degree
        inst.knots = rust_obj.knots_mode
        inst.extrapolation = rust_obj.extrapolation
        inst.include_bias = rust_obj.include_bias
        inst.order = rust_obj.order
        inst.sparse_output = False
        inst._custom_knots = None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustSplineTransformer.from_bytes(state, target_dev)
        self.n_knots = self._model.n_knots
        self.degree = self._model.degree
        self.knots = self._model.knots_mode
        self.extrapolation = self._model.extrapolation
        self.include_bias = self._model.include_bias
        self.order = self._model.order
        self.sparse_output = False
        self._custom_knots = None
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "SplineTransformer":
        """Compute knot positions of B-splines (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X), self._custom_knots)
        return self

    def transform(self, X: Any) -> Any:
        """Transform each feature data to B-splines (＾▽＾)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw, order=self.order)
        if self.sparse_output:
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr

    @property
    def n_features_out_(self) -> Optional[int]:
        return self._model.n_features_out_

    @property
    def bsplines_(self) -> Optional[List[np.ndarray]]:
        ks = self._model.knots_
        return None if ks is None else [np.asarray(k, dtype=np.float64) for k in ks]

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise ValueError("SplineTransformer is not fitted yet (´-ω-`)")
        in_names = (
            [f"x{i}" for i in range(self.n_features_in_)]
            if input_features is None
            else list(input_features)
        )
        per_feat = self.n_features_out_ // self.n_features_in_
        return np.asarray(
            [f"{name}_sp_{k}" for name in in_names for k in range(per_feat)], dtype=object
        )

    def __repr__(self) -> str:
        return (
            f"SplineTransformer(n_knots={self.n_knots}, degree={self.degree}, "
            f"extrapolation='{self.extrapolation}') (o´∀｀o)"
        )


class KernelCenterer(_BaseFeatureEng):
    """
    Center an arbitrary kernel matrix (*≧ω≦*)
    Uses Rust Backend::col_reduce_stats (N1).
    """

    _rust_cls = _RustKernelCenterer

    def __init__(self, device: Optional[str] = None):
        self.device = _resolve_device(device)
        self._model = _RustKernelCenterer(device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustKernelCenterer, device: str) -> "KernelCenterer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustKernelCenterer.from_bytes(state, target_dev)
        self.device = self._model.device_ or target_dev

    def fit(self, K: Any, y: Optional[Any] = None) -> "KernelCenterer":
        """Fit KernelCenterer (＾▽＾)."""
        self._model.fit_buffer(self._fit_x(K))
        return self

    def transform(self, K: Any, copy: bool = True) -> np.ndarray:
        """Center kernel matrix K (*≧ω≦*)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(K))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def K_fit_rows_(self) -> Optional[np.ndarray]:
        v = self._model.k_fit_rows_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def K_fit_all_(self) -> Optional[float]:
        return self._model.k_fit_all_

    def __repr__(self) -> str:
        return f"KernelCenterer(device='{self.device}') (*≧ω≦*)"


def add_dummy_feature(X: Any, value: float = 1.0) -> Any:
    """Augment dataset with an additional dummy feature column (＾▽＾)."""
    if hasattr(X, "tocsc") or hasattr(X, "tocsr"):
        from scipy import sparse

        dummy = sparse.csr_matrix(np.full((X.shape[0], 1), value, dtype=np.float64))
        return sparse.hstack([dummy, X], format="csr")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D ( >_< )")
    dummy = np.full((arr.shape[0], 1), value, dtype=arr.dtype)
    return np.hstack([dummy, arr])

