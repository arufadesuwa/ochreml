"""
OchreML Feature Extraction Submodule (*≧ω≦*)
Provides DictVectorizer, FeatureHasher (MurmurHash3), and the text subpackage.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from .._ochreml import (
    DictVectorizer as _RustDictVectorizer,
    FeatureHasher as _RustFeatureHasher,
)
from .._preproc_utils import csr_bytes_to_scipy, validate_dit_path
from . import text


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


def _dict_to_pairs(d: Dict[str, Any], separator: str) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for k, v in d.items():
        if isinstance(v, str):
            pairs.append((f"{k}{separator}{v}", 1.0))
        elif isinstance(v, bool):
            pairs.append((str(k), 1.0 if v else 0.0))
        else:
            val = float(v)
            if val != 0.0:
                pairs.append((str(k), val))
    return pairs


class DictVectorizer:
    """
    Transforms lists of feature-value mappings to vectors (＾▽＾)

    Parameters
    ----------
    dtype : dtype, default=np.float64
        The type of feature values.
    separator : str, default='='
        Separator string used when constructing new features for one-hot string values.
    sparse : bool, default=True
        Whether transform should produce scipy.sparse matrices.
    sort : bool, default=True
        Whether feature_names_ and vocabulary_ should be sorted when fitting.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustDictVectorizer

    def __init__(
        self,
        *,
        dtype: Any = np.float64,
        separator: str = "=",
        sparse: bool = True,
        sort: bool = True,
        device: Optional[str] = None,
    ):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.device = _resolve_device(device)
        self._model = _RustDictVectorizer(
            separator=separator, sort=sort, device=self.device
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustDictVectorizer, device: str) -> "DictVectorizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.dtype = np.float64
        inst.separator = "="
        inst.sparse = True
        inst.sort = True
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustDictVectorizer.from_bytes(state, target_dev)
        self.dtype = np.float64
        self.separator = "="
        self.sparse = True
        self.sort = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Iterable[Dict[str, Any]], y: Optional[Any] = None) -> "DictVectorizer":
        """Learn a list of feature name -> indices mappings (*≧ω≦*)."""
        self._model.fit_dicts(X)
        return self

    def transform(self, X: Iterable[Dict[str, Any]]) -> Any:
        """Transform feature->value dicts to array or sparse matrix (o´∀｀o)."""
        parts = self._model.transform_dicts_csr(X)
        mat = csr_bytes_to_scipy(parts, dtype=self.dtype)
        return mat if self.sparse else mat.toarray()

    def fit_transform(self, X: Iterable[Dict[str, Any]], y: Optional[Any] = None) -> Any:
        """Learn a list of feature name -> indices mappings and transform X (＾▽＾)."""
        parts = self._model.fit_transform_dicts_csr(X)
        mat = csr_bytes_to_scipy(parts, dtype=self.dtype)
        return mat if self.sparse else mat.toarray()

    def inverse_transform(self, X: Any, dict_type: Any = dict) -> List[Dict[str, Any]]:
        """Transform array or sparse matrix X back to feature mappings (*≧ω≦*)."""
        names = self.feature_names_
        if names is None:
            raise ValueError("DictVectorizer is not fitted yet (´-ω-`)")
        out: List[Dict[str, Any]] = []
        if hasattr(X, "tocsr"):
            csr = X.tocsr()
            for r in range(csr.shape[0]):
                d = dict_type()
                for idx in range(csr.indptr[r], csr.indptr[r + 1]):
                    d[names[csr.indices[idx]]] = csr.data[idx]
                out.append(d)
        else:
            arr = np.asarray(X)
            for row in arr:
                d = dict_type()
                for j, val in enumerate(row):
                    if val != 0:
                        d[names[j]] = val
                out.append(d)
        return out

    def restrict(self, support: Sequence[bool], indices: bool = False) -> "DictVectorizer":
        """Restrict the features to those in support using feature selection (＾▽＾)."""
        names = self.feature_names_
        if names is None:
            raise ValueError("DictVectorizer is not fitted yet (´-ω-`)")
        if indices:
            mask = [False] * len(names)
            for i in support:
                mask[int(i)] = True
        else:
            mask = [bool(s) for s in support]
        self._model.restrict(mask)
        return self

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        names = self.feature_names_
        if names is None:
            raise ValueError("DictVectorizer is not fitted yet (´-ω-`)")
        return np.asarray(names, dtype=object)

    @property
    def feature_names_(self) -> Optional[List[str]]:
        return self._model.feature_names_

    @property
    def vocabulary_(self) -> Optional[Dict[str, int]]:
        return self._model.vocabulary_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DictVectorizer":
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustDictVectorizer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "DictVectorizer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"DictVectorizer(separator='{self.separator}', sparse={self.sparse}) (＾▽＾)"


class FeatureHasher:
    """
    Implements feature hashing, aka the hashing trick, via Rust MurmurHash3 (*≧ω≦*)

    Parameters
    ----------
    n_features : int, default=2**20
        The number of features (columns) in the output matrices.
    input_type : {'dict', 'pair', 'string'}, default='dict'
        Choose a string from {'dict', 'pair', 'string'}.
    dtype : numpy dtype, default=np.float64
        The type of feature values.
    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustFeatureHasher

    def __init__(
        self,
        n_features: int = 2**20,
        *,
        input_type: str = "dict",
        dtype: Any = np.float64,
        alternate_sign: bool = True,
        device: Optional[str] = None,
    ):
        self.n_features = n_features
        self.input_type = input_type
        self.dtype = dtype
        self.alternate_sign = alternate_sign
        self.device = _resolve_device(device)
        self._model = _RustFeatureHasher(
            n_features=n_features,
            input_type=input_type,
            alternate_sign=alternate_sign,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustFeatureHasher, device: str) -> "FeatureHasher":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_features = rust_obj.n_features
        inst.input_type = rust_obj.input_type
        inst.dtype = np.float64
        inst.alternate_sign = rust_obj.alternate_sign
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustFeatureHasher.from_bytes(state, target_dev)
        self.n_features = self._model.n_features
        self.input_type = self._model.input_type
        self.dtype = np.float64
        self.alternate_sign = self._model.alternate_sign
        self.device = self._model.device_ or target_dev

    def fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "FeatureHasher":
        """Only validates estimator's parameters (stateless) (＾▽＾)."""
        return self

    def transform(self, raw_X: Iterable[Any]) -> Any:
        """Transform a sequence of instances to a scipy.sparse matrix (*≧ω≦*)."""
        if self.input_type == "dict":
            parts = self._model.transform_dicts_csr(raw_X)
            return csr_bytes_to_scipy(parts, dtype=self.dtype)
        rows: List[List[Tuple[str, float]]] = []
        if self.input_type == "pair":
            for r in raw_X:
                rows.append([(str(k), float(v)) for k, v in r])
        else:
            for r in raw_X:
                rows.append([(str(s), 1.0) for s in r])

        parts = self._model.transform_pairs_csr(rows)
        return csr_bytes_to_scipy(parts, dtype=self.dtype)

    def fit_transform(self, raw_X: Iterable[Any], y: Optional[Any] = None) -> Any:
        """Fit and transform raw_X (o´∀｀o)."""
        return self.fit(raw_X, y).transform(raw_X)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "FeatureHasher":
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustFeatureHasher.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "FeatureHasher":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"FeatureHasher(n_features={self.n_features}, input_type='{self.input_type}') (*≧ω≦*)"


__all__ = [
    "DictVectorizer",
    "FeatureHasher",
    "text",
]

