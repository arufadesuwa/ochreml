"""
OchreML Non-Linear Distributions & Discretization Estimators (*≧ω≦*)
Implements PowerTransformer, QuantileTransformer, FunctionTransformer,
KBinsDiscretizer, Binarizer, and their stateless counterparts.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Union
import numpy as np

from .._ochreml import (
    Binarizer as _RustBinarizer,
    KBinsDiscretizer as _RustKBinsDiscretizer,
    PowerTransformer as _RustPowerTransformer,
    QuantileTransformer as _RustQuantileTransformer,
    _rust_binarize_into,
)
from .._preproc_utils import (
    bytes_to_f64_2d,
    load_dit_envelope,
    save_dit_envelope,
    validate_dit_path,
)
from .._utils import process_features

_TAG_FUNCTION_TRANSFORMER = 35


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


class _BaseTransform:
    """Shared persistence & buffer helpers for Rust-backed transforms (＾▽＾)."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def _fit_x(self, X: Any) -> np.ndarray:
        is_buf, X_data = process_features(X)
        return X_data if is_buf else np.asarray(X_data, dtype=np.float64)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
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

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("Transformer is not fitted yet (´-ω-`)")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class PowerTransformer(_BaseTransform):
    """
    Apply a power transform featurewise to make data more Gaussian-like (*≧ω≦*)

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method.
    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the transformed output.
    copy : bool, default=True
        Whether to return a copy.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustPowerTransformer

    def __init__(
        self,
        method: str = "yeo-johnson",
        *,
        standardize: bool = True,
        copy: bool = True,
        device: Optional[str] = None,
    ):
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.device = _resolve_device(device)
        self._model = _RustPowerTransformer(
            method=method, standardize=standardize, device=self.device
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustPowerTransformer, device: str) -> "PowerTransformer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.method = rust_obj.method
        inst.standardize = rust_obj.standardize
        inst.copy = True
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustPowerTransformer.from_bytes(state, target_dev)
        self.method = self._model.method
        self.standardize = self._model.standardize
        self.copy = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "PowerTransformer":
        """Estimate the optimal parameter lambda for each feature (＾▽＾)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit and transform X in a single fused Rust pass (*≧ω≦*)."""
        rows, cols, raw = self._model.fit_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def transform(self, X: Any) -> np.ndarray:
        """Apply the power transform to each feature using fitted lambdas (o´∀｀o)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Apply the inverse power transformation using fitted lambdas (＾▽＾)."""
        rows, cols, raw = self._model.inverse_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def lambdas_(self) -> Optional[np.ndarray]:
        lam = self._model.lambdas_
        return None if lam is None else np.asarray(lam, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"PowerTransformer(method='{self.method}', standardize={self.standardize}) (*≧ω≦*)"
        )


def power_transform(
    X: Any,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Parametric, monotonic transformation to make data more Gaussian-like (＾▽＾)."""
    return PowerTransformer(
        method=method, standardize=standardize, copy=copy, device=device
    ).fit_transform(X)


class QuantileTransformer(_BaseTransform):
    """
    Transform features using quantiles information (＾▽＾)

    Parameters
    ----------
    n_quantiles : int, default=1000
        Number of quantiles to be computed.
    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data.
    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices (kept for API parity).
    subsample : int, default=10000
        Maximum number of samples used to estimate the quantiles.
    random_state : int or None, default=None
        Seed for subsampling.
    copy : bool, default=True
        Whether to return a copy.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustQuantileTransformer

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = 10_000,
        random_state: Optional[int] = None,
        copy: bool = True,
        device: Optional[str] = None,
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.device = _resolve_device(device)
        self._model = _RustQuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustQuantileTransformer, device: str) -> "QuantileTransformer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_quantiles = rust_obj.n_quantiles
        inst.output_distribution = rust_obj.output_distribution
        inst.ignore_implicit_zeros = False
        inst.subsample = 10_000
        inst.random_state = None
        inst.copy = True
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustQuantileTransformer.from_bytes(state, target_dev)
        self.n_quantiles = self._model.n_quantiles
        self.output_distribution = self._model.output_distribution
        self.ignore_implicit_zeros = False
        self.subsample = 10_000
        self.random_state = None
        self.copy = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "QuantileTransformer":
        """Compute the quantiles used for transforming (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Feature-wise transformation of the data (＾▽＾)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Back-projection to the original space (o´∀｀o)."""
        rows, cols, raw = self._model.inverse_transform_buffer(self._fit_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def n_quantiles_(self) -> Optional[int]:
        return self._model.n_quantiles_

    @property
    def references_(self) -> Optional[np.ndarray]:
        r = self._model.references_
        return None if r is None else np.asarray(r, dtype=np.float64)

    @property
    def quantiles_(self) -> Optional[np.ndarray]:
        qb = self._model.quantiles_bytes
        if qb is None:
            return None
        rows, cols, raw = qb
        return bytes_to_f64_2d(rows, cols, raw)

    def __repr__(self) -> str:
        return (
            f"QuantileTransformer(n_quantiles={self.n_quantiles}, "
            f"output_distribution='{self.output_distribution}') (＾▽＾)"
        )


def quantile_transform(
    X: Any,
    *,
    axis: int = 0,
    n_quantiles: int = 1000,
    output_distribution: str = "uniform",
    ignore_implicit_zeros: bool = False,
    subsample: int = 10_000,
    random_state: Optional[int] = None,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Transform features using quantiles information (o´∀｀o)."""
    arr = np.asarray(X, dtype=np.float64)
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=random_state,
        copy=copy,
        device=device,
    )
    if axis == 1:
        return qt.fit_transform(np.ascontiguousarray(arr.T)).T
    if axis == 0:
        return qt.fit_transform(arr)
    raise ValueError(f"axis must be 0 or 1, got {axis} ( >_< )")


class FunctionTransformer:
    """
    Constructs a transformer from an arbitrary callable (o´∀｀o)

    Parameters
    ----------
    func : callable or None, default=None
        The callable to use for the transformation (identity if None).
    inverse_func : callable or None, default=None
        The callable to use for the inverse transformation.
    validate : bool, default=False
        Indicate that input X should be validated as 2D array before calling func.
    accept_sparse : bool, default=False
        Whether sparse matrices are allowed when validate=True.
    check_inverse : bool, default=True
        Whether to check that func and inverse_func are inverses of each other.
    feature_names_out : callable, 'one-to-one' or None, default=None
        Determines the list of feature names returned by get_feature_names_out.
    kw_args : dict or None, default=None
        Keyword arguments passed to func.
    inv_kw_args : dict or None, default=None
        Keyword arguments passed to inverse_func.
    """

    def __init__(
        self,
        func: Optional[Callable[..., Any]] = None,
        inverse_func: Optional[Callable[..., Any]] = None,
        *,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = True,
        feature_names_out: Optional[Union[str, Callable[..., Any]]] = None,
        kw_args: Optional[Dict[str, Any]] = None,
        inv_kw_args: Optional[Dict[str, Any]] = None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self.n_features_in_: Optional[int] = None

    def _validate_x(self, X: Any) -> Any:
        if not self.validate:
            return X
        arr = X if (self.accept_sparse and hasattr(X, "tocsr")) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D ( >_< )")
        return arr

    def fit(self, X: Any, y: Optional[Any] = None) -> "FunctionTransformer":
        """Fit transformer by checking X and recording n_features_in_ (＾▽＾)."""
        Xv = self._validate_x(X)
        if hasattr(Xv, "shape") and len(Xv.shape) == 2:
            self.n_features_in_ = int(Xv.shape[1])
        elif isinstance(Xv, list) and Xv and isinstance(Xv[0], (list, tuple)):
            self.n_features_in_ = len(Xv[0])
        return self

    def transform(self, X: Any) -> Any:
        """Transform X using the forward function (*≧ω≦*)."""
        Xv = self._validate_x(X)
        if self.func is None:
            return Xv
        return self.func(Xv, **(self.kw_args or {}))

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit and transform X (o´∀｀o)."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> Any:
        """Transform X using the inverse function (＾▽＾)."""
        Xv = self._validate_x(X)
        if self.inverse_func is None:
            return Xv
        return self.inverse_func(Xv, **(self.inv_kw_args or {}))

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("FunctionTransformer is not fitted yet (´-ω-`)")
        in_names = (
            np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        if self.feature_names_out == "one-to-one":
            return in_names
        if callable(self.feature_names_out):
            return np.asarray(self.feature_names_out(self, in_names), dtype=object)
        raise ValueError("feature_names_out is not configured ( >_< )")

    def save(self, path: str) -> None:
        """Save FunctionTransformer to a '.dit' file (*≧ω≦*)."""
        save_dit_envelope(_TAG_FUNCTION_TRANSFORMER, self.__dict__, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "FunctionTransformer":
        """Load FunctionTransformer from a '.dit' file (＾▽＾)."""
        with open(validate_dit_path(path), "rb") as f:
            state = load_dit_envelope(_TAG_FUNCTION_TRANSFORMER, f.read())
        inst = cls.__new__(cls)
        inst.__dict__.update(state)
        return inst

    def __repr__(self) -> str:
        return f"FunctionTransformer(func={self.func!r}) (o´∀｀o)"


class KBinsDiscretizer(_BaseTransform):
    """
    Bin continuous data into intervals (＾▽＾)

    Parameters
    ----------
    n_bins : int or array-like of int, default=5
        Number of bins to produce.
    encode : {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.
    quantile_method : str, default='averaged_inverted_cdf'
        Quantile method (averaged_inverted_cdf matches sklearn).
    dtype : number type or None, default=None
        Desired output dtype.
    subsample : int or None, default=200000
        Maximum number of samples used to fit bin edges.
    random_state : int or None, default=None
        Seed for subsampling.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustKBinsDiscretizer

    def __init__(
        self,
        n_bins: Union[int, Sequence[int]] = 5,
        *,
        encode: str = "onehot",
        strategy: str = "quantile",
        quantile_method: str = "averaged_inverted_cdf",
        dtype: Optional[Any] = None,
        subsample: Optional[int] = 200_000,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
        self.device = _resolve_device(device)

        n_bins_param = [int(n_bins)] if isinstance(n_bins, (int, np.integer)) else [int(b) for b in n_bins]
        self._model = _RustKBinsDiscretizer(
            n_bins_param=n_bins_param,
            encode=encode,
            strategy=strategy,
            subsample=subsample,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustKBinsDiscretizer, device: str) -> "KBinsDiscretizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_bins = 5
        inst.encode = rust_obj.encode
        inst.strategy = rust_obj.strategy
        inst.quantile_method = "averaged_inverted_cdf"
        inst.dtype = None
        inst.subsample = 200_000
        inst.random_state = None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustKBinsDiscretizer.from_bytes(state, target_dev)
        self.n_bins = 5
        self.encode = self._model.encode
        self.strategy = self._model.strategy
        self.quantile_method = "averaged_inverted_cdf"
        self.dtype = None
        self.subsample = 200_000
        self.random_state = None
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "KBinsDiscretizer":
        """Fit the estimator (*≧ω≦*)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> Any:
        """Discretize the data (＾▽＾)."""
        rows, cols, raw = self._model.transform_buffer(self._fit_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw)
        if self.dtype is not None:
            arr = arr.astype(self.dtype, copy=False)
        if self.encode == "onehot":
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr

    def inverse_transform(self, Xt: Any) -> np.ndarray:
        """Transform discretized data back to original feature space (o´∀｀o)."""
        arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt, dtype=np.float64)
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        rows, cols, raw = self._model.inverse_transform_buffer(arr)
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def bin_edges_(self) -> Optional[np.ndarray]:
        edges = self._model.bin_edges_
        if edges is None:
            return None
        out = np.empty(len(edges), dtype=object)
        for i, e in enumerate(edges):
            out[i] = np.asarray(e, dtype=np.float64)
        return out

    @property
    def n_bins_(self) -> Optional[np.ndarray]:
        nb = self._model.n_bins_
        return None if nb is None else np.asarray(nb, dtype=np.int64)

    def __repr__(self) -> str:
        return f"KBinsDiscretizer(encode='{self.encode}', strategy='{self.strategy}') (＾▽＾)"


class Binarizer(_BaseTransform):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold (*≧ω≦*)
    """

    _rust_cls = _RustBinarizer

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        copy: bool = True,
        device: Optional[str] = None,
    ):
        self.threshold = float(threshold)
        self.copy = copy
        self.device = _resolve_device(device)
        self._model = _RustBinarizer(threshold=self.threshold, device=self.device)

    @classmethod
    def _from_rust(cls, rust_obj: _RustBinarizer, device: str) -> "Binarizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.threshold = rust_obj.threshold
        inst.copy = True
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustBinarizer.from_bytes(state, target_dev)
        self.threshold = self._model.threshold
        self.copy = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "Binarizer":
        """Only validates estimator's parameters (＾▽＾)."""
        self._model.fit_buffer(self._fit_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Binarize each element of X (*≧ω≦*)."""
        X_data = self._fit_x(X)
        out = np.empty(X_data.shape, dtype=np.float64)
        self._model.transform_into(X_data, out)
        return out

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit and binarize each element of X in one pass (*≧ω≦*)."""
        X_data = self._fit_x(X)
        out = np.empty(X_data.shape, dtype=np.float64)
        self._model.fit_transform_into(X_data, out)
        return out

    def __repr__(self) -> str:
        return f"Binarizer(threshold={self.threshold}) (*≧ω≦*)"


def binarize(
    X: Any,
    *,
    threshold: float = 0.0,
    copy: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Boolean thresholding of array-like or scipy.sparse matrices (＾▽＾)."""
    arr = np.ascontiguousarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D ( >_< )")
    out = np.empty(arr.shape, dtype=np.float64)
    _rust_binarize_into(arr, out, threshold=float(threshold))
    return out

