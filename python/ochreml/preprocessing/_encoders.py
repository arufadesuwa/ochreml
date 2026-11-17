"""
OchreML Categorical & Target Encoding Estimators (*≧ω≦*)
Powered by Rust Rayon hash/sort kernels with zero-copy buffers and .dit persistence.
"""

from typing import Any, List, Optional, Sequence, Union
import numpy as np

from .._ochreml import (
    LabelBinarizer as _RustLabelBinarizer,
    LabelEncoder as _RustLabelEncoder,
    MultiLabelBinarizer as _RustMultiLabelBinarizer,
    OneHotEncoder as _RustOneHotEncoder,
    OrdinalEncoder as _RustOrdinalEncoder,
    TargetEncoder as _RustTargetEncoder,
)
from .._preproc_utils import (
    as_1d_list,
    as_2d_array_or_rows,
    as_object_col_lists,
    as_utf32_1d_view,
    as_utf32_2d_view,
    bytes_to_f64_2d,
    bytes_to_i64_1d,
    bytes_to_i64_2d,
    csr_bytes_to_scipy,
    validate_dit_path,
)


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


def _cats_to_numpy_list(cats: Optional[List[List[Any]]]) -> Optional[List[np.ndarray]]:
    if cats is None:
        return None
    out = []
    for col in cats:
        arr = np.asarray(col)
        out.append(arr)
    return out


def _csr_parts_to_dense(parts: Any, dtype: Any) -> np.ndarray:
    _, idx_b, ptr_b, nr, nc = parts
    out = np.zeros((nr, nc), dtype=dtype)
    if nr == 0 or nc == 0:
        return out
    indices = np.frombuffer(idx_b, dtype=np.int64)
    if indices.size == 0:
        return out
    indptr = np.frombuffer(ptr_b, dtype=np.int64)
    counts = np.diff(indptr)
    if counts.min() == counts.max():
        k = int(counts[0])
        rows = np.arange(nr, dtype=np.int64)[:, None]
        out.ravel()[(rows * nc + indices.reshape(nr, k)).ravel()] = 1.0
    else:
        rows = np.repeat(np.arange(nr, dtype=np.int64), counts)
        out.ravel()[rows * nc + indices] = 1.0
    return out


class _BaseEncoder:
    """Shared persistence & device mixin for OchreML encoders (＾▽＾)."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit encoder to X, then transform X (*≧ω≦*)."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save encoder to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load encoder from a binary '.dit' file (*≧ω≦*)."""
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
    def device_(self) -> Optional[str]:
        return self._model.device_


class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array (*≧ω≦*)

    Parameters
    ----------
    categories : 'auto' or list of array-like, default='auto'
        Categories (unique values) per feature.
    drop : {'first', 'if_binary'} or array-like, default=None
        Methodology to drop one of the categories per feature.
    sparse_output : bool, default=True
        Will return sparse matrix if set True else will return an array.
    dtype : number type, default=np.float64
        Desired dtype of output.
    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'}, default='error'
        Specifies how unknown categories are handled during transform.
    min_frequency : int or float, default=None
        Minimum frequency below which a category is considered infrequent.
    max_categories : int, default=None
        Upper limit on the number of output features for each input feature.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustOneHotEncoder

    def __init__(
        self,
        *,
        categories: Union[str, Sequence[Sequence[Any]]] = "auto",
        drop: Optional[Union[str, Sequence[Any]]] = None,
        sparse_output: bool = True,
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.device = _resolve_device(device)

        drop_mode = drop if isinstance(drop, str) else None
        mf = None if min_frequency is None else float(min_frequency)
        self._model = _RustOneHotEncoder(
            handle_unknown=handle_unknown,
            drop_mode=drop_mode,
            min_frequency=mf,
            max_categories=max_categories,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustOneHotEncoder, device: str) -> "OneHotEncoder":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.categories = "auto"
        inst.drop = None
        inst.sparse_output = True
        inst.dtype = np.float64
        inst.handle_unknown = "error"
        inst.min_frequency = None
        inst.max_categories = None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustOneHotEncoder.from_bytes(state, target_dev)
        self.categories = "auto"
        self.drop = None
        self.sparse_output = True
        self.dtype = np.float64
        self.handle_unknown = "error"
        self.min_frequency = None
        self.max_categories = None
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "OneHotEncoder":
        """Fit OneHotEncoder to X (＾▽＾)."""
        explicit_cats = (
            None
            if isinstance(self.categories, str) and self.categories == "auto"
            else [list(c) for c in self.categories]
        )
        explicit_drop = (
            None
            if (self.drop is None or isinstance(self.drop, str))
            else [d for d in self.drop]
        )
        if explicit_cats is None:
            u32_view = as_utf32_2d_view(X)
            if u32_view is not None:
                self._model.fit_utf32(*u32_view, explicit_drop)
                return self
            col_lists = as_object_col_lists(X)
            if col_lists is not None:
                self._model.fit_col_lists(col_lists, explicit_drop)
                return self
        is_num, payload = as_2d_array_or_rows(X)
        if is_num:
            self._model.fit_buffer(payload, explicit_cats, explicit_drop)
        else:
            self._model.fit_objects(payload, explicit_cats, explicit_drop)
        return self

    def transform(self, X: Any) -> Any:
        """Transform X using one-hot encoding (*≧ω≦*)."""
        u32_view = as_utf32_2d_view(X)
        if u32_view is not None:
            parts = self._model.transform_utf32_csr(*u32_view)
            return csr_bytes_to_scipy(parts, dtype=self.dtype) if self.sparse_output else _csr_parts_to_dense(parts, self.dtype)

        col_lists = as_object_col_lists(X)
        if col_lists is not None:
            parts = self._model.transform_col_lists_csr(col_lists)
            return csr_bytes_to_scipy(parts, dtype=self.dtype) if self.sparse_output else _csr_parts_to_dense(parts, self.dtype)

        is_num, payload = as_2d_array_or_rows(X)
        parts = (
            self._model.transform_buffer_csr(payload)
            if is_num
            else self._model.transform_objects_csr(payload)
        )
        return csr_bytes_to_scipy(parts, dtype=self.dtype) if self.sparse_output else _csr_parts_to_dense(parts, self.dtype)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit OneHotEncoder and transform X in one pass (*≧ω≦*)."""
        if isinstance(self.categories, str) and self.categories == "auto":
            explicit_drop = (
                None
                if (self.drop is None or isinstance(self.drop, str))
                else [d for d in self.drop]
            )
            u32_view = as_utf32_2d_view(X)
            if u32_view is not None:
                parts = self._model.fit_transform_utf32_csr(*u32_view, explicit_drop)
                return csr_bytes_to_scipy(parts, dtype=self.dtype) if self.sparse_output else _csr_parts_to_dense(parts, self.dtype)
            col_lists = as_object_col_lists(X)
            if col_lists is not None:
                parts = self._model.fit_transform_col_lists_csr(col_lists, explicit_drop)
                return csr_bytes_to_scipy(parts, dtype=self.dtype) if self.sparse_output else _csr_parts_to_dense(parts, self.dtype)
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Convert the data back to the original representation (o´∀｀o)."""
        if hasattr(X, "toarray"):
            arr = np.ascontiguousarray(X.toarray(), dtype=np.float64)
        else:
            arr = np.ascontiguousarray(X, dtype=np.float64)
        rows = self._model.inverse_transform_buffer(arr)
        return np.asarray(rows, dtype=object)

    @property
    def categories_(self) -> Optional[List[np.ndarray]]:
        return _cats_to_numpy_list(self._model.categories_)

    @property
    def drop_idx_(self) -> Optional[np.ndarray]:
        d = self._model.drop_idx_
        return None if d is None else np.asarray(d, dtype=object)

    @property
    def infrequent_categories_(self) -> Optional[List[Optional[np.ndarray]]]:
        cats = self._model.categories_
        inf = self._model.infrequent_indices_
        if cats is None or inf is None:
            return None
        return [
            np.asarray([cats[j][i] for i in inf[j]]) if inf[j] else None
            for j in range(len(cats))
        ]

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        cats = self.categories_
        if cats is None or self.n_features_in_ is None:
            raise ValueError("OneHotEncoder is not fitted yet (´-ω-`)")
        if input_features is None:
            in_names = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            in_names = list(input_features)
        drops = self._model.drop_idx_
        inf = self._model.infrequent_indices_ or [[] for _ in cats]
        names: List[str] = []
        for j, feat_name in enumerate(in_names):
            drop_j = drops[j] if drops is not None else None
            inf_set = set(inf[j])
            for idx, c in enumerate(cats[j]):
                if idx in inf_set or idx == drop_j:
                    continue
                names.append(f"{feat_name}_{c}")
            if inf_set:
                names.append(f"{feat_name}_infrequent_sklearn")
        return np.asarray(names, dtype=object)

    def __repr__(self) -> str:
        return f"OneHotEncoder(handle_unknown='{self.handle_unknown}', sparse_output={self.sparse_output}) (*≧ω≦*)"


class OrdinalEncoder(_BaseEncoder):
    """
    Encode categorical features as an integer array (＾▽＾)

    Parameters
    ----------
    categories : 'auto' or list of array-like, default='auto'
        Categories (unique values) per feature.
    dtype : number type, default=np.float64
        Desired dtype of output.
    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        Whether to raise or use unknown_value on unseen categories.
    unknown_value : int or np.nan, default=None
        Value used when handle_unknown='use_encoded_value'.
    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories.
    min_frequency : int or float, default=None
        Minimum frequency for non-infrequent categories.
    max_categories : int, default=None
        Maximum number of categories per feature.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustOrdinalEncoder

    def __init__(
        self,
        *,
        categories: Union[str, Sequence[Sequence[Any]]] = "auto",
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        unknown_value: Optional[float] = None,
        encoded_missing_value: float = np.nan,
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.device = _resolve_device(device)

        uv = None if unknown_value is None else float(unknown_value)
        mf = None if min_frequency is None else float(min_frequency)
        self._model = _RustOrdinalEncoder(
            handle_unknown=handle_unknown,
            unknown_value=uv,
            encoded_missing_value=float(encoded_missing_value),
            min_frequency=mf,
            max_categories=max_categories,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustOrdinalEncoder, device: str) -> "OrdinalEncoder":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.categories = "auto"
        inst.dtype = np.float64
        inst.handle_unknown = "error"
        inst.unknown_value = None
        inst.encoded_missing_value = np.nan
        inst.min_frequency = None
        inst.max_categories = None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustOrdinalEncoder.from_bytes(state, target_dev)
        self.categories = "auto"
        self.dtype = np.float64
        self.handle_unknown = "error"
        self.unknown_value = None
        self.encoded_missing_value = np.nan
        self.min_frequency = None
        self.max_categories = None
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "OrdinalEncoder":
        """Fit the OrdinalEncoder to X (o´∀｀o)."""
        explicit_cats = (
            None
            if isinstance(self.categories, str) and self.categories == "auto"
            else [list(c) for c in self.categories]
        )
        if explicit_cats is None:
            u32_view = as_utf32_2d_view(X)
            if u32_view is not None:
                self._model.fit_utf32(*u32_view)
                return self
            col_lists = as_object_col_lists(X)
            if col_lists is not None:
                self._model.fit_col_lists(col_lists)
                return self
        is_num, payload = as_2d_array_or_rows(X)
        if is_num:
            self._model.fit_buffer(payload, explicit_cats)
        else:
            self._model.fit_objects(payload, explicit_cats)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform X to ordinal codes (*≧ω≦*)."""
        u32_view = as_utf32_2d_view(X)
        if u32_view is not None:
            out = np.empty((u32_view[1], u32_view[2]), dtype=np.float64)
            self._model.transform_utf32_into(*u32_view, out)
            return out.astype(self.dtype, copy=False) if np.dtype(self.dtype) != np.float64 else out
        col_lists = as_object_col_lists(X)
        if col_lists is not None:
            out = np.empty((len(col_lists[0]), len(col_lists)), dtype=np.float64)
            self._model.transform_col_lists_into(col_lists, out)
            return out.astype(self.dtype, copy=False) if np.dtype(self.dtype) != np.float64 else out
        is_num, payload = as_2d_array_or_rows(X)
        rows, cols, raw = (
            self._model.transform_buffer(payload)
            if is_num
            else self._model.transform_objects(payload)
        )
        arr = bytes_to_f64_2d(rows, cols, raw)
        return arr.astype(self.dtype, copy=False) if np.dtype(self.dtype) != np.float64 else arr

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit OrdinalEncoder and transform X in one pass (*≧ω≦*)."""
        if isinstance(self.categories, str) and self.categories == "auto":
            u32_view = as_utf32_2d_view(X)
            if u32_view is not None:
                out = np.empty((u32_view[1], u32_view[2]), dtype=np.float64)
                self._model.fit_transform_utf32_into(*u32_view, out)
                return out.astype(self.dtype, copy=False) if np.dtype(self.dtype) != np.float64 else out
            col_lists = as_object_col_lists(X)
            if col_lists is not None:
                out = np.empty((len(col_lists[0]), len(col_lists)), dtype=np.float64)
                self._model.fit_transform_col_lists_into(col_lists, out)
                return out.astype(self.dtype, copy=False) if np.dtype(self.dtype) != np.float64 else out
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Convert the data back to the original representation (＾▽＾)."""
        arr = np.ascontiguousarray(X, dtype=np.float64)
        return np.asarray(self._model.inverse_transform_buffer(arr), dtype=object)

    @property
    def categories_(self) -> Optional[List[np.ndarray]]:
        return _cats_to_numpy_list(self._model.categories_)

    @property
    def infrequent_categories_(self) -> Optional[List[Optional[np.ndarray]]]:
        cats = self._model.categories_
        inf = self._model.infrequent_indices_
        if cats is None or inf is None:
            return None
        return [
            np.asarray([cats[j][i] for i in inf[j]]) if inf[j] else None
            for j in range(len(cats))
        ]

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("OrdinalEncoder is not fitted yet (´-ω-`)")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)

    def __repr__(self) -> str:
        return f"OrdinalEncoder(handle_unknown='{self.handle_unknown}') (＾▽＾)"


class LabelEncoder(_BaseEncoder):
    """
    Encode target labels with value between 0 and n_classes - 1 (o´∀｀o)
    """

    _rust_cls = _RustLabelEncoder

    def __init__(self, device: Optional[str] = None):
        self.device = _resolve_device(device)
        self._model = _RustLabelEncoder(device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustLabelEncoder, device: str) -> "LabelEncoder":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLabelEncoder.from_bytes(state, target_dev)
        self.device = self._model.device_ or target_dev

    def fit(self, y: Any) -> "LabelEncoder":
        """Fit label encoder (＾▽＾)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            self._model.fit_utf32(*u32_view)
            return self
        self._model.fit_objects(as_1d_list(y))
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Transform labels to normalized encoding (*≧ω≦*)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            out = np.empty(u32_view[1], dtype=np.int64)
            self._model.transform_utf32_into(*u32_view, out)
            return out
        raw = self._model.transform_objects(as_1d_list(y))
        return bytes_to_i64_1d(raw)

    def fit_transform(self, y: Any) -> np.ndarray:
        """Fit label encoder and return encoded labels (o´∀｀o)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            out = np.empty(u32_view[1], dtype=np.int64)
            self._model.fit_transform_utf32_into(*u32_view, out)
            return out
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Any) -> np.ndarray:
        """Transform labels back to original encoding (＾▽＾)."""
        idx = [int(v) for v in as_1d_list(y)]
        return np.asarray(self._model.inverse_transform_indices(idx))

    @property
    def classes_(self) -> Optional[np.ndarray]:
        c = self._model.classes_
        return None if c is None else np.asarray(c)

    def __repr__(self) -> str:
        return "LabelEncoder() (o´∀｀o)"


class LabelBinarizer(_BaseEncoder):
    """
    Binarize labels in a one-vs-all fashion (*≧ω≦*)
    """

    _rust_cls = _RustLabelBinarizer

    def __init__(
        self,
        *,
        neg_label: int = 0,
        pos_label: int = 1,
        sparse_output: bool = False,
        device: Optional[str] = None,
    ):
        self.neg_label = int(neg_label)
        self.pos_label = int(pos_label)
        self.sparse_output = sparse_output
        self.device = _resolve_device(device)
        self._model = _RustLabelBinarizer(
            neg_label=self.neg_label, pos_label=self.pos_label, device=self.device
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustLabelBinarizer, device: str) -> "LabelBinarizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.neg_label = rust_obj.neg_label
        inst.pos_label = rust_obj.pos_label
        inst.sparse_output = False
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLabelBinarizer.from_bytes(state, target_dev)
        self.neg_label = self._model.neg_label
        self.pos_label = self._model.pos_label
        self.sparse_output = False
        self.device = self._model.device_ or target_dev

    def fit(self, y: Any) -> "LabelBinarizer":
        """Fit label binarizer (＾▽＾)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            self._model.fit_utf32(*u32_view)
            return self
        self._model.fit_objects(as_1d_list(y))
        return self

    def _finish_binarized(self, rows: int, cols: int, raw: bytes) -> Any:
        arr = bytes_to_i64_2d(rows, cols, raw)
        if self.sparse_output:
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr

    def transform(self, y: Any) -> Any:
        """Transform multi-class labels to binary labels (*≧ω≦*)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            rows, cols, raw = self._model.transform_utf32(*u32_view)
            return self._finish_binarized(rows, cols, raw)
        rows, cols, raw = self._model.transform_objects(as_1d_list(y))
        return self._finish_binarized(rows, cols, raw)

    def fit_transform(self, y: Any) -> Any:
        """Fit label binarizer and transform multi-class labels to binary labels (o´∀｀o)."""
        u32_view = as_utf32_1d_view(y)
        if u32_view is not None:
            rows, cols, raw = self._model.fit_transform_utf32(*u32_view)
            return self._finish_binarized(rows, cols, raw)
        return self.fit(y).transform(y)

    def inverse_transform(self, Y: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Transform binary labels back to multi-class labels (＾▽＾)."""
        if hasattr(Y, "toarray"):
            arr = np.ascontiguousarray(Y.toarray(), dtype=np.float64)
        else:
            arr = np.ascontiguousarray(Y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.asarray(self._model.inverse_transform_buffer(arr, threshold))

    @property
    def classes_(self) -> Optional[np.ndarray]:
        c = self._model.classes_
        return None if c is None else np.asarray(c)

    @property
    def y_type_(self) -> str:
        c = self.classes_
        if c is None:
            return "unknown"
        return "binary" if len(c) <= 2 else "multiclass"

    def __repr__(self) -> str:
        return f"LabelBinarizer(neg_label={self.neg_label}, pos_label={self.pos_label}) (*≧ω≦*)"


def label_binarize(
    y: Any,
    *,
    classes: Sequence[Any],
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> Any:
    """Binarize labels in a one-vs-all fashion given explicit classes (＾▽＾)."""
    lb = LabelBinarizer(neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output)
    lb._model.fit_classes(list(classes))
    return lb.transform(y)


class MultiLabelBinarizer(_BaseEncoder):
    """
    Transform between iterable of iterables and a multilabel format (o´∀｀o)
    """

    _rust_cls = _RustMultiLabelBinarizer

    def __init__(
        self,
        *,
        classes: Optional[Sequence[Any]] = None,
        sparse_output: bool = False,
        device: Optional[str] = None,
    ):
        self.classes = classes
        self.sparse_output = sparse_output
        self.device = _resolve_device(device)
        self._model = _RustMultiLabelBinarizer(device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustMultiLabelBinarizer, device: str) -> "MultiLabelBinarizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.classes = None
        inst.sparse_output = False
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustMultiLabelBinarizer.from_bytes(state, target_dev)
        self.classes = None
        self.sparse_output = False
        self.device = self._model.device_ or target_dev

    def fit(self, y: Sequence[Sequence[Any]]) -> "MultiLabelBinarizer":
        """Fit the label sets binarizer, storing classes_ (＾▽＾)."""
        rows = [list(row) for row in y]
        ec = None if self.classes is None else list(self.classes)
        self._model.fit_objects(rows, ec)
        return self

    def transform(self, y: Sequence[Sequence[Any]]) -> Any:
        """Transform the given label sets (*≧ω≦*)."""
        n_rows, n_cols, raw = self._model.transform_py_rows(y)
        arr = bytes_to_i64_2d(n_rows, n_cols, raw)
        if self.sparse_output:
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr

    def fit_transform(self, y: Sequence[Sequence[Any]]) -> Any:
        """Fit and transform label sets (o´∀｀o)."""
        if self.classes is None:
            n_rows, n_cols, raw = self._model.fit_transform_py_rows(y)
            arr = bytes_to_i64_2d(n_rows, n_cols, raw)
            if self.sparse_output:
                from scipy import sparse

                return sparse.csr_matrix(arr)
            return arr
        return self.fit(y).transform(y)

    def inverse_transform(self, yt: Any) -> List[tuple]:
        """Transform the given indicator matrix into label sets (＾▽＾)."""
        if hasattr(yt, "toarray"):
            arr = np.ascontiguousarray(yt.toarray(), dtype=np.float64)
        else:
            arr = np.ascontiguousarray(yt, dtype=np.float64)
        return [tuple(row) for row in self._model.inverse_transform_buffer(arr)]

    @property
    def classes_(self) -> Optional[np.ndarray]:
        c = self._model.classes_
        return None if c is None else np.asarray(c)

    def __repr__(self) -> str:
        return f"MultiLabelBinarizer(sparse_output={self.sparse_output}) (o´∀｀o)"


def _infer_target_and_prepare(
    y: Any, target_type: str
) -> tuple[List[List[float]], Optional[List[Any]], str, np.ndarray]:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.ravel()

    if target_type == "auto":
        if np.issubdtype(y_arr.dtype, np.floating) and not np.all(np.isfinite(y_arr) & (y_arr == np.floor(y_arr))):
            inferred = "continuous"
        else:
            uniq = np.unique(y_arr)
            if len(uniq) == 2:
                inferred = "binary"
            elif np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype.kind in ("U", "S", "O", "b"):
                inferred = "multiclass" if len(uniq) > 2 else "binary"
            else:
                inferred = "continuous"
    else:
        inferred = target_type

    if inferred == "binary":
        le = LabelEncoder().fit(y_arr)
        y_enc = le.transform(y_arr).astype(np.float64)
        return [y_enc.tolist()], le.classes_.tolist(), "binary", y_enc
    if inferred == "multiclass":
        lb = LabelBinarizer().fit(y_arr)
        y_bin = lb.transform(y_arr).astype(np.float64)
        y_cols = [y_bin[:, t].tolist() for t in range(y_bin.shape[1])]
        y_int = LabelEncoder().fit_transform(y_arr)
        return y_cols, lb.classes_.tolist(), "multiclass", y_int
    y_f = np.asarray(y_arr, dtype=np.float64)
    return [y_f.tolist()], None, "continuous", y_f


def _build_cv_fold_ids(
    y_strat: np.ndarray,
    inferred_type: str,
    cv: int,
    shuffle: bool,
    random_state: Optional[int],
) -> List[int]:
    n = len(y_strat)
    fold_ids = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(random_state) if shuffle else None

    if inferred_type in ("binary", "multiclass"):
        _, inverse = np.unique(y_strat, return_inverse=True)
        fold_cursor = 0
        for cls_idx in range(int(inverse.max()) + 1):
            idx = np.where(inverse == cls_idx)[0]
            if rng is not None:
                rng.shuffle(idx)
            for pos, sample_i in enumerate(idx):
                fold_ids[sample_i] = (fold_cursor + pos) % cv
            fold_cursor = (fold_cursor + len(idx)) % cv
    else:
        idx = np.arange(n)
        if rng is not None:
            rng.shuffle(idx)
        for pos, sample_i in enumerate(idx):
            fold_ids[sample_i] = pos % cv

    return fold_ids.tolist()


class TargetEncoder(_BaseEncoder):
    """
    Target encoding for categorical features using empirical Bayes shrinkage (＾▽＾)

    Parameters
    ----------
    categories : 'auto' or list of array-like, default='auto'
        Categories per feature.
    target_type : {'auto', 'continuous', 'binary', 'multiclass'}, default='auto'
        Type of target.
    smooth : 'auto' or float, default='auto'
        Amount of mixing of the target mean conditioned on the value of the category
        with the global target mean.
    cv : int, default=5
        Number of folds for cross-fitting in fit_transform.
    shuffle : bool, default=True
        Whether to shuffle data before splitting into CV folds.
    random_state : int or None, default=None
        Seed used when shuffle=True.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustTargetEncoder

    def __init__(
        self,
        *,
        categories: Union[str, Sequence[Sequence[Any]]] = "auto",
        target_type: str = "auto",
        smooth: Union[str, float] = "auto",
        cv: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = _resolve_device(device)

        smooth_val = None if smooth == "auto" else float(smooth)
        self._model = _RustTargetEncoder(
            smooth=smooth_val,
            cv=cv,
            shuffle=shuffle,
            random_state=random_state,
            target_type=target_type,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustTargetEncoder, device: str) -> "TargetEncoder":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.categories = "auto"
        inst.target_type = rust_obj.target_type_ or "auto"
        inst.smooth = "auto"
        inst.cv = 5
        inst.shuffle = True
        inst.random_state = None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustTargetEncoder.from_bytes(state, target_dev)
        self.categories = "auto"
        self.target_type = self._model.target_type_ or "auto"
        self.smooth = "auto"
        self.cv = 5
        self.shuffle = True
        self.random_state = None
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Any) -> "TargetEncoder":
        """Fit target encoder on all (X, y) data (*≧ω≦*)."""
        y_cols, classes, inferred, _ = _infer_target_and_prepare(y, self.target_type)
        explicit_cats = (
            None
            if isinstance(self.categories, str) and self.categories == "auto"
            else [list(c) for c in self.categories]
        )
        u32_view = as_utf32_2d_view(X)
        if u32_view is not None:
            self._model.fit_utf32(*u32_view, y_cols, classes, inferred, explicit_cats)
            return self
        col_lists = as_object_col_lists(X)
        if col_lists is not None:
            self._model.fit_col_lists(col_lists, y_cols, classes, inferred, explicit_cats)
            return self
        _, payload = as_2d_array_or_rows(X)
        rows = payload.tolist() if isinstance(payload, np.ndarray) else payload
        self._model.fit_objects(rows, y_cols, classes, inferred, explicit_cats)
        return self

    def fit_transform(self, X: Any, y: Any) -> np.ndarray:
        """Fit on full data and return cross-fitted encodings to avoid leakage (＾▽＾)."""
        y_cols, classes, inferred, y_strat = _infer_target_and_prepare(y, self.target_type)
        fold_ids = _build_cv_fold_ids(
            y_strat, inferred, self.cv, self.shuffle, self.random_state
        )
        explicit_cats = (
            None
            if isinstance(self.categories, str) and self.categories == "auto"
            else [list(c) for c in self.categories]
        )
        u32_view = as_utf32_2d_view(X)
        if u32_view is not None:
            n_rows, n_cols, raw = self._model.fit_transform_utf32(
                *u32_view, y_cols, classes, inferred, fold_ids, explicit_cats
            )
            return bytes_to_f64_2d(n_rows, n_cols, raw)
        col_lists = as_object_col_lists(X)
        if col_lists is not None:
            n_rows, n_cols, raw = self._model.fit_transform_col_lists(
                col_lists, y_cols, classes, inferred, fold_ids, explicit_cats
            )
            return bytes_to_f64_2d(n_rows, n_cols, raw)
        _, payload = as_2d_array_or_rows(X)
        rows = payload.tolist() if isinstance(payload, np.ndarray) else payload
        n_rows, n_cols, raw = self._model.fit_transform_objects(
            rows, y_cols, classes, inferred, fold_ids, explicit_cats
        )
        return bytes_to_f64_2d(n_rows, n_cols, raw)

    def transform(self, X: Any) -> np.ndarray:
        """Transform X using learned target encodings (o´∀｀o)."""
        u32_view = as_utf32_2d_view(X)
        if u32_view is not None:
            n_rows, n_cols, raw = self._model.transform_utf32(*u32_view)
            return bytes_to_f64_2d(n_rows, n_cols, raw)
        col_lists = as_object_col_lists(X)
        if col_lists is not None:
            n_rows, n_cols, raw = self._model.transform_col_lists(col_lists)
            return bytes_to_f64_2d(n_rows, n_cols, raw)
        _, payload = as_2d_array_or_rows(X)
        rows = payload.tolist() if isinstance(payload, np.ndarray) else payload
        n_rows, n_cols, raw = self._model.transform_objects(rows)
        return bytes_to_f64_2d(n_rows, n_cols, raw)

    @property
    def categories_(self) -> Optional[List[np.ndarray]]:
        return _cats_to_numpy_list(self._model.categories_)

    @property
    def encodings_(self) -> Optional[List[np.ndarray]]:
        e = self._model.encodings_
        return None if e is None else [np.asarray(v, dtype=np.float64) for v in e]

    @property
    def target_mean_(self) -> Optional[Union[float, np.ndarray]]:
        tm = self._model.target_mean_
        if tm is None:
            return None
        return float(tm[0]) if len(tm) == 1 else np.asarray(tm, dtype=np.float64)

    @property
    def target_type_(self) -> Optional[str]:
        return self._model.target_type_

    @property
    def classes_(self) -> Optional[np.ndarray]:
        c = self._model.classes_
        return None if c is None else np.asarray(c)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("TargetEncoder is not fitted yet (´-ω-`)")
        in_names = (
            [f"x{i}" for i in range(self.n_features_in_)]
            if input_features is None
            else list(input_features)
        )
        if self.target_type_ == "multiclass" and self.classes_ is not None:
            return np.asarray(
                [f"{fn}_{cls}" for fn in in_names for cls in self.classes_], dtype=object
            )
        return np.asarray(in_names, dtype=object)

    def __repr__(self) -> str:
        return f"TargetEncoder(smooth={self.smooth!r}, cv={self.cv}) (＾▽＾)"

