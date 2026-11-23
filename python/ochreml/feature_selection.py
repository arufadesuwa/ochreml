"""
OchreML Feature Selection Module
Provides Scikit-Learn-compatible statistical tests (f_regression, f_classif, chi2)
and feature selectors (VarianceThreshold, SelectKBest, SelectPercentile).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    SelectKBest as _RustSelectKBest,
    SelectPercentile as _RustSelectPercentile,
    VarianceThreshold as _RustVarianceThreshold,
    chi2_buffer as _rust_chi2_buffer,
    f_classif_buffer as _rust_f_classif_buffer,
    f_regression_buffer as _rust_f_regression_buffer,
)
from ._preproc_utils import bytes_to_f64_2d, bytes_to_i64_1d, validate_dit_path
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


def _to_1d_f64(y: Any) -> Any:
    if y is None:
        raise ValueError("Target y cannot be None")
    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    else:
        arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_1d_i64_list(y: Any) -> list[int]:
    if y is None:
        raise ValueError("Target y cannot be None")
    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    else:
        arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        return [int(v) for v in arr.tolist()]
    _, inv = np.unique(arr, return_inverse=True)
    return [int(v) for v in inv.tolist()]


def f_regression(
    X: Any,
    y: Any,
    *,
    center: bool = True,
    force_finite: bool = True,
) -> Tuple[Any, Any]:
    """Univariate linear regression tests returning F-statistics and p-values."""
    X_data = _to_f64_array(X)
    y_data = _to_1d_f64(y)
    f_raw, p_raw = _rust_f_regression_buffer(X_data, y_data, center, force_finite)
    return (
        np.frombuffer(f_raw, dtype=np.float64).copy(),
        np.frombuffer(p_raw, dtype=np.float64).copy(),
    )


def f_classif(X: Any, y: Any) -> Tuple[Any, Any]:
    """Compute the ANOVA F-value for the provided sample."""
    X_data = _to_f64_array(X)
    y_list = _to_1d_i64_list(y)
    f_raw, p_raw = _rust_f_classif_buffer(X_data, y_list)
    return (
        np.frombuffer(f_raw, dtype=np.float64).copy(),
        np.frombuffer(p_raw, dtype=np.float64).copy(),
    )


def chi2(X: Any, y: Any) -> Tuple[Any, Any]:
    """Compute chi-squared stats between each non-negative feature and class."""
    X_data = _to_f64_array(X)
    y_list = _to_1d_i64_list(y)
    c_raw, p_raw = _rust_chi2_buffer(X_data, y_list)
    return (
        np.frombuffer(c_raw, dtype=np.float64).copy(),
        np.frombuffer(p_raw, dtype=np.float64).copy(),
    )


_BUILTIN_SCORE_FUNCS: Dict[Any, str] = {
    f_classif: "f_classif",
    f_regression: "f_regression",
    chi2: "chi2",
}

_NAME_TO_SCORE_FUNC: Dict[str, Callable[..., Any]] = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "chi2": chi2,
}


def _resolve_score_func_name(score_func: Any) -> str:
    if score_func in _BUILTIN_SCORE_FUNCS:
        return _BUILTIN_SCORE_FUNCS[score_func]
    if isinstance(score_func, str) and score_func in _NAME_TO_SCORE_FUNC:
        return score_func
    return "custom"


class VarianceThreshold:
    """Feature selector that removes all low-variance features."""

    _rust_cls = _RustVarianceThreshold

    def __init__(
        self,
        threshold: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.device = _resolve_device(device)
        self._model = _RustVarianceThreshold(threshold=float(threshold), device=self.device)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustVarianceThreshold, device: str
    ) -> "VarianceThreshold":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.threshold = rust_obj.threshold
        inst.device = rust_obj.device_ or device
        return inst

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"threshold": self.threshold, "device": self.device}

    def set_params(self, **params: Any) -> "VarianceThreshold":
        for key, val in params.items():
            if key not in {"threshold", "device"}:
                raise ValueError(f"Invalid parameter '{key}' for VarianceThreshold")
            if key == "device":
                self.device = _resolve_device(val)
            else:
                setattr(self, key, val)
        self._model = _RustVarianceThreshold(
            threshold=float(self.threshold), device=self.device
        )
        return self

    def fit(self, X: Any, y: Optional[Any] = None) -> "VarianceThreshold":
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def get_support(self, indices: bool = False) -> Any:
        if indices:
            return bytes_to_i64_1d(self._model.get_support_indices())
        return np.asarray(self._model.get_support_mask(), dtype=bool)

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "VarianceThreshold":
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "VarianceThreshold":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustVarianceThreshold.from_bytes(state, target_dev)
        self._model = rust_obj
        self.threshold = rust_obj.threshold
        self.device = rust_obj.device_ or target_dev

    @property
    def variances_(self) -> Optional[Any]:
        raw = self._model.variances_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class SelectKBest:
    """Select features according to the k highest scores."""

    _rust_cls = _RustSelectKBest

    def __init__(
        self,
        score_func: Callable[..., Any] = f_classif,
        k: Union[int, str] = 10,
        device: Optional[str] = None,
    ) -> None:
        self.score_func = score_func
        self.k = k
        self.device = _resolve_device(device)
        self._init_rust_model()

    def _init_rust_model(self) -> None:
        sf_name = _resolve_score_func_name(self.score_func)
        if self.k == "all":
            k_opt: Optional[int] = None
        elif isinstance(self.k, int) and not isinstance(self.k, bool):
            if self.k < 0:
                raise ValueError(f"k must be >= 0 or 'all', got {self.k}")
            k_opt = int(self.k)
        else:
            raise ValueError(f"k must be a non-negative int or 'all', got {self.k}")

        self._model = _RustSelectKBest(
            score_func_name=sf_name,
            k=k_opt,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustSelectKBest, device: str) -> "SelectKBest":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.score_func = _NAME_TO_SCORE_FUNC.get(rust_obj.score_func_name, f_classif)
        inst.k = "all" if rust_obj.k is None else rust_obj.k
        inst.device = rust_obj.device_ or device
        return inst

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "score_func": self.score_func,
            "k": self.k,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "SelectKBest":
        for key, val in params.items():
            if key not in {"score_func", "k", "device"}:
                raise ValueError(f"Invalid parameter '{key}' for SelectKBest")
            if key == "device":
                self.device = _resolve_device(val)
            else:
                setattr(self, key, val)
        self._init_rust_model()
        return self

    def fit(self, X: Any, y: Any) -> "SelectKBest":
        X_data = _to_f64_array(X)
        sf_name = _resolve_score_func_name(self.score_func)
        if sf_name != "custom":
            if sf_name in {"f_classif", "chi2"}:
                y_list = _to_1d_i64_list(y)
                y_arr = np.asarray(y_list, dtype=np.float64)
            else:
                y_arr = _to_1d_f64(y)
            self._model.fit_buffer(X_data, y_arr)
        else:
            res = self.score_func(X, y)
            if isinstance(res, tuple) and len(res) == 2:
                scores, pvalues = res
                p_list = (
                    [float(v) for v in np.asarray(pvalues, dtype=np.float64).ravel().tolist()]
                    if pvalues is not None
                    else None
                )
            else:
                scores = res
                p_list = None
            s_list = [float(v) for v in np.asarray(scores, dtype=np.float64).ravel().tolist()]
            self._model.fit_with_scores(X_data.shape[1], s_list, p_list)
        return self

    def transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Any) -> Any:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def get_support(self, indices: bool = False) -> Any:
        if indices:
            return bytes_to_i64_1d(self._model.get_support_indices())
        return np.asarray(self._model.get_support_mask(), dtype=bool)

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SelectKBest":
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "SelectKBest":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "score_func": self.score_func if _resolve_score_func_name(self.score_func) == "custom" else None,
        }

    def __setstate__(self, state: Union[bytes, Dict[str, Any]]) -> None:
        target_dev = _resolve_device(None)
        if isinstance(state, dict):
            raw_bytes = state["bytes"]
            custom_sf = state.get("score_func")
        else:
            raw_bytes = state
            custom_sf = None
        rust_obj = _RustSelectKBest.from_bytes(raw_bytes, target_dev)
        self._model = rust_obj
        self.score_func = (
            custom_sf
            if custom_sf is not None
            else _NAME_TO_SCORE_FUNC.get(rust_obj.score_func_name, f_classif)
        )
        self.k = "all" if rust_obj.k is None else rust_obj.k
        self.device = rust_obj.device_ or target_dev

    @property
    def scores_(self) -> Optional[Any]:
        raw = self._model.scores_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def pvalues_(self) -> Optional[Any]:
        raw = self._model.pvalues_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class SelectPercentile:
    """Select features according to a percentile of the highest scores."""

    _rust_cls = _RustSelectPercentile

    def __init__(
        self,
        score_func: Callable[..., Any] = f_classif,
        percentile: Union[int, float] = 10,
        device: Optional[str] = None,
    ) -> None:
        self.score_func = score_func
        self.percentile = percentile
        self.device = _resolve_device(device)
        self._init_rust_model()

    def _init_rust_model(self) -> None:
        sf_name = _resolve_score_func_name(self.score_func)
        self._model = _RustSelectPercentile(
            score_func_name=sf_name,
            percentile=float(self.percentile),
            device=self.device,
        )

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustSelectPercentile, device: str
    ) -> "SelectPercentile":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.score_func = _NAME_TO_SCORE_FUNC.get(rust_obj.score_func_name, f_classif)
        inst.percentile = rust_obj.percentile
        inst.device = rust_obj.device_ or device
        return inst

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "score_func": self.score_func,
            "percentile": self.percentile,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "SelectPercentile":
        for key, val in params.items():
            if key not in {"score_func", "percentile", "device"}:
                raise ValueError(f"Invalid parameter '{key}' for SelectPercentile")
            if key == "device":
                self.device = _resolve_device(val)
            else:
                setattr(self, key, val)
        self._init_rust_model()
        return self

    def fit(self, X: Any, y: Any) -> "SelectPercentile":
        X_data = _to_f64_array(X)
        sf_name = _resolve_score_func_name(self.score_func)
        if sf_name != "custom":
            if sf_name in {"f_classif", "chi2"}:
                y_list = _to_1d_i64_list(y)
                y_arr = np.asarray(y_list, dtype=np.float64)
            else:
                y_arr = _to_1d_f64(y)
            self._model.fit_buffer(X_data, y_arr)
        else:
            res = self.score_func(X, y)
            if isinstance(res, tuple) and len(res) == 2:
                scores, pvalues = res
                p_list = (
                    [float(v) for v in np.asarray(pvalues, dtype=np.float64).ravel().tolist()]
                    if pvalues is not None
                    else None
                )
            else:
                scores = res
                p_list = None
            s_list = [float(v) for v in np.asarray(scores, dtype=np.float64).ravel().tolist()]
            self._model.fit_with_scores(X_data.shape[1], s_list, p_list)
        return self

    def transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Any) -> Any:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.inverse_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def get_support(self, indices: bool = False) -> Any:
        if indices:
            return bytes_to_i64_1d(self._model.get_support_indices())
        return np.asarray(self._model.get_support_mask(), dtype=bool)

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SelectPercentile":
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "SelectPercentile":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "bytes": self._model.to_bytes(),
            "score_func": self.score_func if _resolve_score_func_name(self.score_func) == "custom" else None,
        }

    def __setstate__(self, state: Union[bytes, Dict[str, Any]]) -> None:
        target_dev = _resolve_device(None)
        if isinstance(state, dict):
            raw_bytes = state["bytes"]
            custom_sf = state.get("score_func")
        else:
            raw_bytes = state
            custom_sf = None
        rust_obj = _RustSelectPercentile.from_bytes(raw_bytes, target_dev)
        self._model = rust_obj
        self.score_func = (
            custom_sf
            if custom_sf is not None
            else _NAME_TO_SCORE_FUNC.get(rust_obj.score_func_name, f_classif)
        )
        self.percentile = rust_obj.percentile
        self.device = rust_obj.device_ or target_dev

    @property
    def scores_(self) -> Optional[Any]:
        raw = self._model.scores_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def pvalues_(self) -> Optional[Any]:
        raw = self._model.pvalues_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


from ._meta_selection import RFE, SelectFromModel

__all__ = [
    "VarianceThreshold",
    "SelectKBest",
    "SelectPercentile",
    "f_regression",
    "f_classif",
    "chi2",
    "RFE",
    "SelectFromModel",
]
