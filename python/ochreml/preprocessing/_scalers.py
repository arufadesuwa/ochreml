"""
OchreML Scaling & Normalization Estimators & Functions (*≧ω≦*)
Accelerated by Rust Backend N1 (col_reduce_stats, col_scale_shift) and N2 (elementwise_op).
"""

from typing import Any, Optional, Tuple, Union
import numpy as np

from .._ochreml import (
    MaxAbsScaler as _RustMaxAbsScaler,
    MinMaxScaler as _RustMinMaxScaler,
    Normalizer as _RustNormalizer,
    RobustScaler as _RustRobustScaler,
    StandardScaler as _RustStandardScaler,
    _rust_normalize_into,
)
from .._preproc_utils import bytes_to_f64_2d, validate_dit_path
from .._utils import process_features, process_sample_weight


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


class _BaseScaler:
    """Shared mixin for OchreML hardware-accelerated scalers (＾▽＾)."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def _fit_x(self, X: Any) -> Any:
        is_buf, X_data = process_features(X)
        if not is_buf:
            X_data = np.asarray(X_data, dtype=np.float64)
        return X_data

    def _transform_into(self, X: Any) -> np.ndarray:
        X_data = self._fit_x(X)
        out = np.empty(X_data.shape, dtype=np.float64)
        self._model.transform_into(X_data, out)
        return out

    def fit_transform(self, X: Any, y: Optional[Any] = None, **fit_params: Any) -> Any:
        """Fit to data, then transform it (*≧ω≦*)."""
        X_data = self._fit_x(X)
        out = np.empty(X_data.shape, dtype=np.float64)
        self._model.fit_transform_into(X_data, out)
        return out

    def save(self, path: str) -> None:
        """Save fitted scaler to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load scaler from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> Any:
        """Move scaler execution target to a new hardware device (o´∀｀o)."""
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

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> Any:
        """Get output feature names for transformation (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("Scaler is not fitted yet. Call fit() first (´-ω-`)")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        names = np.asarray(input_features, dtype=object)
        if len(names) != self.n_features_in_:
            raise ValueError(
                f"input_features length ({len(names)}) != n_features_in_ ({self.n_features_in_}) ( >_< )"
            )
        return names


class StandardScaler(_BaseScaler):
    """
    Standardize features by removing the mean and scaling to unit variance (*≧ω≦*)

    Parameters
    ----------
    copy : bool, default=True
        Whether to avoid inplace modifications (OchreML returns a new array).
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustStandardScaler

    def __init__(
        self,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        device: Optional[str] = None,
    ):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.device = _resolve_device(device)
        self._model = _RustStandardScaler(
            with_mean=with_mean, with_std=with_std, device=self.device
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustStandardScaler, device: str) -> "StandardScaler":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.copy = True
        inst.with_mean = rust_obj.with_mean
        inst.with_std = rust_obj.with_std
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustStandardScaler.from_bytes(state, target_dev)
        self.copy = True
        self.with_mean = self._model.with_mean
        self.with_std = self._model.with_std
        self.device = self._model.device_ or target_dev

    def fit(
        self, X: Any, y: Optional[Any] = None, sample_weight: Optional[Any] = None
    ) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling (＾▽＾)."""
        X_data = self._fit_x(X)
        _, sw_data = process_sample_weight(sample_weight)
        if sw_data is not None and not isinstance(sw_data, np.ndarray):
            sw_data = np.asarray(sw_data, dtype=np.float64)
        self._model.fit_buffer(X_data, sw_data)
        return self

    def partial_fit(
        self, X: Any, y: Optional[Any] = None, sample_weight: Optional[Any] = None
    ) -> "StandardScaler":
        """Online computation of mean and std on X for later scaling (o´∀｀o)."""
        X_data = self._fit_x(X)
        _, sw_data = process_sample_weight(sample_weight)
        if sw_data is not None and not isinstance(sw_data, np.ndarray):
            sw_data = np.asarray(sw_data, dtype=np.float64)
        self._model.partial_fit_buffer(X_data, sw_data)
        return self

    def fit_transform(
        self, X: Any, y: Optional[Any] = None, sample_weight: Optional[Any] = None
    ) -> np.ndarray:
        """Fit to data, then standardize it in one fused pass (*≧ω≦*)."""
        X_data = self._fit_x(X)
        _, sw_data = process_sample_weight(sample_weight)
        if sw_data is not None and not isinstance(sw_data, np.ndarray):
            sw_data = np.asarray(sw_data, dtype=np.float64)
        out = np.empty(X_data.shape, dtype=np.float64)
        self._model.fit_transform_into(X_data, out, sw_data)
        return out

    def transform(self, X: Any) -> np.ndarray:
        """Perform standardization by centering and scaling (*≧ω≦*)."""
        return self._transform_into(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Scale back the data to the original representation (＾▽＾)."""
        X_data = self._fit_x(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def mean_(self) -> Optional[np.ndarray]:
        m = self._model.mean_
        return None if m is None else np.asarray(m, dtype=np.float64)

    @property
    def var_(self) -> Optional[np.ndarray]:
        v = self._model.var_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def scale_(self) -> Optional[np.ndarray]:
        s = self._model.scale_
        return None if s is None else np.asarray(s, dtype=np.float64)

    @property
    def n_samples_seen_(self) -> int:
        return self._model.n_samples_seen_

    def __repr__(self) -> str:
        return (
            f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std}, "
            f"device='{self.device}') (＾▽＾)"
        )


class MinMaxScaler(_BaseScaler):
    """
    Transform features by scaling each feature to a given range (o´∀｀o)

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Whether to return a copy.
    clip : bool, default=False
        Set to True to clip transformed values of held-out data to provided feature range.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustMinMaxScaler

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        copy: bool = True,
        clip: bool = False,
        device: Optional[str] = None,
    ):
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self.copy = copy
        self.clip = clip
        self.device = _resolve_device(device)
        self._model = _RustMinMaxScaler(
            feature_range=self.feature_range, clip=clip, device=self.device
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustMinMaxScaler, device: str) -> "MinMaxScaler":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.feature_range = rust_obj.feature_range
        inst.copy = True
        inst.clip = rust_obj.clip
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustMinMaxScaler.from_bytes(state, target_dev)
        self.feature_range = self._model.feature_range
        self.copy = True
        self.clip = self._model.clip
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def partial_fit(self, X: Any, y: Optional[Any] = None) -> "MinMaxScaler":
        """Online computation of min and max on X for later scaling (＾▽＾)."""
        self._model.partial_fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Scale features of X according to feature_range (o´∀｀o)."""
        return self._transform_into(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Undo the scaling of X according to feature_range (＾▽＾)."""
        rows, cols, raw = self._model.inverse_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def min_(self) -> Optional[np.ndarray]:
        v = self._model.min_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def scale_(self) -> Optional[np.ndarray]:
        v = self._model.scale_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def data_min_(self) -> Optional[np.ndarray]:
        v = self._model.data_min_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def data_max_(self) -> Optional[np.ndarray]:
        v = self._model.data_max_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def data_range_(self) -> Optional[np.ndarray]:
        v = self._model.data_range_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def n_samples_seen_(self) -> int:
        return self._model.n_samples_seen_

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range}, clip={self.clip}, device='{self.device}') (o´∀｀o)"


class MaxAbsScaler(_BaseScaler):
    """
    Scale each feature by its maximum absolute value (*≧ω≦*)

    Parameters
    ----------
    copy : bool, default=True
        Whether to return a copy.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustMaxAbsScaler

    def __init__(self, copy: bool = True, device: Optional[str] = None):
        self.copy = copy
        self.device = _resolve_device(device)
        self._model = _RustMaxAbsScaler(device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustMaxAbsScaler, device: str) -> "MaxAbsScaler":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.copy = True
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustMaxAbsScaler.from_bytes(state, target_dev)
        self.copy = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "MaxAbsScaler":
        """Compute the maximum absolute value to be used for later scaling (＾▽＾)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def partial_fit(self, X: Any, y: Optional[Any] = None) -> "MaxAbsScaler":
        """Online computation of max abs value on X for later scaling (o´∀｀o)."""
        self._model.partial_fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Scale the data (*≧ω≦*)."""
        return self._transform_into(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Scale back the data to the original representation (＾▽＾)."""
        rows, cols, raw = self._model.inverse_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def max_abs_(self) -> Optional[np.ndarray]:
        v = self._model.max_abs_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def scale_(self) -> Optional[np.ndarray]:
        v = self._model.scale_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def n_samples_seen_(self) -> int:
        return self._model.n_samples_seen_

    def __repr__(self) -> str:
        return f"MaxAbsScaler(device='{self.device}') (*≧ω≦*)"


class RobustScaler(_BaseScaler):
    """
    Scale features using statistics that are robust to outliers (＾▽＾)

    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to interquartile range.
    quantile_range : tuple (q_min, q_max), default=(25.0, 75.0)
        Quantile range used to calculate scale_.
    copy : bool, default=True
        Whether to return a copy.
    unit_variance : bool, default=False
        If True, scale data so normally distributed features have variance of 1.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustRobustScaler

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
        device: Optional[str] = None,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = (float(quantile_range[0]), float(quantile_range[1]))
        self.copy = copy
        self.unit_variance = unit_variance
        self.device = _resolve_device(device)
        self._model = _RustRobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=self.quantile_range,
            unit_variance=unit_variance,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustRobustScaler, device: str) -> "RobustScaler":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.with_centering = rust_obj.with_centering
        inst.with_scaling = rust_obj.with_scaling
        inst.quantile_range = rust_obj.quantile_range
        inst.copy = True
        inst.unit_variance = rust_obj.unit_variance
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustRobustScaler.from_bytes(state, target_dev)
        self.with_centering = self._model.with_centering
        self.with_scaling = self._model.with_scaling
        self.quantile_range = self._model.quantile_range
        self.copy = True
        self.unit_variance = self._model.unit_variance
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "RobustScaler":
        """Compute the median and quantiles to be used for scaling (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Center and scale the data (＾▽＾)."""
        return self._transform_into(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Scale back the data to the original representation (o´∀｀o)."""
        rows, cols, raw = self._model.inverse_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def center_(self) -> Optional[np.ndarray]:
        v = self._model.center_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def scale_(self) -> Optional[np.ndarray]:
        v = self._model.scale_
        return None if v is None else np.asarray(v, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"RobustScaler(quantile_range={self.quantile_range}, "
            f"unit_variance={self.unit_variance}, device='{self.device}') (＾▽＾)"
        )


class Normalizer(_BaseScaler):
    """
    Normalize samples individually to unit norm ('l1', 'l2', or 'max') (o´∀｀o)

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non-zero sample.
    copy : bool, default=True
        Whether to return a copy.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustNormalizer

    def __init__(self, norm: str = "l2", *, copy: bool = True, device: Optional[str] = None):
        self.norm = norm
        self.copy = copy
        self.device = _resolve_device(device)
        self._model = _RustNormalizer(norm=norm, device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustNormalizer, device: str) -> "Normalizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.norm = rust_obj.norm
        inst.copy = True
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustNormalizer.from_bytes(state, target_dev)
        self.norm = self._model.norm
        self.copy = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "Normalizer":
        """Only validates estimator's parameters and n_features_in_ (＾▽＾)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any, copy: Optional[bool] = None) -> np.ndarray:
        """Scale each non zero row of X to unit norm (*≧ω≦*)."""
        return self._transform_into(X)

    def __repr__(self) -> str:
        return f"Normalizer(norm='{self.norm}', device='{self.device}') (o´∀｀o)"


def _apply_axis_stateless(X: Any, axis: int, fn: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    was_1d = arr.ndim == 1
    if was_1d:
        arr2d = arr.reshape(-1, 1) if axis == 0 else arr.reshape(1, -1)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D ( >_< )")

    if axis == 1:
        out = fn(np.ascontiguousarray(arr2d.T)).T
    elif axis == 0:
        out = fn(np.ascontiguousarray(arr2d))
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis} ( >_< )")

    return out.ravel() if was_1d else out


def scale(
    X: Any,
    *,
    axis: int = 0,
    with_mean: bool = True,
    with_std: bool = True,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Standardize a dataset along any axis (*≧ω≦*)."""
    return _apply_axis_stateless(
        X,
        axis,
        lambda mat: StandardScaler(
            copy=copy, with_mean=with_mean, with_std=with_std, device=device
        ).fit_transform(mat),
    )


def minmax_scale(
    X: Any,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    *,
    axis: int = 0,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Transform features by scaling each feature to a given range (o´∀｀o)."""
    return _apply_axis_stateless(
        X,
        axis,
        lambda mat: MinMaxScaler(
            feature_range=feature_range, copy=copy, device=device
        ).fit_transform(mat),
    )


def maxabs_scale(
    X: Any,
    *,
    axis: int = 0,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Scale each feature by its maximum absolute value (＾▽＾)."""
    return _apply_axis_stateless(
        X,
        axis,
        lambda mat: MaxAbsScaler(copy=copy, device=device).fit_transform(mat),
    )


def robust_scale(
    X: Any,
    *,
    axis: int = 0,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: Tuple[float, float] = (25.0, 75.0),
    copy: bool = True,
    unit_variance: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """Standardize a dataset along any axis using median and IQR (*≧ω≦*)."""
    return _apply_axis_stateless(
        X,
        axis,
        lambda mat: RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
            device=device,
        ).fit_transform(mat),
    )


def normalize(
    X: Any,
    norm: str = "l2",
    *,
    axis: int = 1,
    copy: bool = True,
    return_norm: bool = False,
    device: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Scale input vectors individually to unit norm (vector length) (＾▽＾)."""
    arr = np.asarray(X, dtype=np.float64)
    was_1d = arr.ndim == 1
    if was_1d:
        arr2d = np.ascontiguousarray(arr.reshape(1, -1))
    elif arr.ndim == 2:
        arr2d = np.ascontiguousarray(arr)
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D ( >_< )")

    out = np.empty(arr2d.shape, dtype=np.float64)
    norms = _rust_normalize_into(arr2d, out, norm=norm, axis=axis)
    if was_1d:
        out = out.ravel()
    if return_norm:
        return out, np.asarray(norms, dtype=np.float64)
    return out

