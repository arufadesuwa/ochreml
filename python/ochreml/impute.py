"""
OchreML Missing Value Imputation Submodule (*≧ω≦*)
Implements SimpleImputer (N1), MissingIndicator, KNNImputer (N3+N4),
and IterativeImputer powered by OchreML Ridge / LinearRegression.
"""

import copy
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from ._ochreml import (
    KNNImputer as _RustKNNImputer,
    MissingIndicator as _RustMissingIndicator,
    SimpleImputer as _RustSimpleImputer,
)
from ._preproc_utils import (
    as_2d_array_or_rows,
    bytes_to_f64_2d,
    load_dit_envelope,
    save_dit_envelope,
    validate_dit_path,
)
from ._utils import process_features

_TAG_ITERATIVE_IMPUTER = 34


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


class MissingIndicator:
    """
    Binary indicators for missing values (＾▽＾)

    Parameters
    ----------
    missing_values : int, float, or np.nan, default=np.nan
        The placeholder for missing values.
    features : {'missing-only', 'all'}, default='missing-only'
        Whether to represent all features or only those containing missing values at fit time.
    sparse : {'auto', True, False}, default='auto'
        Whether to return sparse matrix if True (auto returns dense ndarray).
    error_on_new : bool, default=True
        Raise ValueError if features='missing-only' and new missing values appear at transform.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustMissingIndicator

    def __init__(
        self,
        *,
        missing_values: float = np.nan,
        features: str = "missing-only",
        sparse: Union[str, bool] = "auto",
        error_on_new: bool = True,
        device: Optional[str] = None,
    ):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new
        self.device = _resolve_device(device)
        self._model = _RustMissingIndicator(
            missing_values=float(missing_values),
            features_mode=features,
            error_on_new=error_on_new,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustMissingIndicator, device: str) -> "MissingIndicator":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.missing_values = np.nan
        inst.features = "missing-only"
        inst.sparse = "auto"
        inst.error_on_new = True
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustMissingIndicator.from_bytes(state, target_dev)
        self.missing_values = np.nan
        self.features = "missing-only"
        self.sparse = "auto"
        self.error_on_new = True
        self.device = self._model.device_ or target_dev

    def _prep_x(self, X: Any) -> np.ndarray:
        is_buf, X_data = process_features(X)
        return X_data if is_buf else np.asarray(X_data, dtype=np.float64)

    def fit(self, X: Any, y: Optional[Any] = None) -> "MissingIndicator":
        """Fit the transformer on X (*≧ω≦*)."""
        self._model.fit_buffer(self._prep_x(X))
        return self

    def _u8_to_output(self, rows: int, cols: int, raw: bytes) -> Any:
        arr = np.frombuffer(raw, dtype=bool).reshape(rows, cols).copy()
        if self.sparse is True:
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr

    def transform(self, X: Any) -> Any:
        """Generate missing values indicator for X (＾▽＾)."""
        rows, cols, raw = self._model.transform_u8_buffer(self._prep_x(X))
        return self._u8_to_output(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Generate missing values indicator for X (o´∀｀o)."""
        rows, cols, raw = self._model.fit_transform_u8_buffer(self._prep_x(X))
        return self._u8_to_output(rows, cols, raw)

    @property
    def features_(self) -> Optional[np.ndarray]:
        f = self._model.features_
        return None if f is None else np.asarray(f, dtype=np.int64)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save MissingIndicator to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MissingIndicator":
        """Load MissingIndicator from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustMissingIndicator.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "MissingIndicator":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        if self.features_ is None or self.n_features_in_ is None:
            raise ValueError("MissingIndicator is not fitted yet (´-ω-`)")
        in_names = (
            [f"x{i}" for i in range(self.n_features_in_)]
            if input_features is None
            else list(input_features)
        )
        return np.asarray([f"missingindicator_{in_names[i]}" for i in self.features_], dtype=object)


class SimpleImputer:
    """
    Univariate imputer for completing missing values with simple strategies (o´∀｀o)
    Uses Rust Backend::col_reduce_stats (N1) for mean and Rayon kernels for median/most_frequent.

    Parameters
    ----------
    missing_values : int, float, str, None or np.nan, default=np.nan
        The placeholder for missing values.
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        The imputation strategy.
    fill_value : str or numerical value, default=None
        When strategy == 'constant', fill_value is used to replace missing_values.
    copy : bool, default=True
        Whether to return a copy.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will stack onto output.
    keep_empty_features : bool, default=False
        If True, features consisting exclusively of missing values are retained.
    device : str or None, default=None
        Target compute device.
    """

    _rust_cls = _RustSimpleImputer

    def __init__(
        self,
        *,
        missing_values: Any = np.nan,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Optional[str] = None,
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = _resolve_device(device)
        self._is_numeric = True
        self._model = _RustSimpleImputer(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustSimpleImputer, device: str) -> "SimpleImputer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.missing_values = np.nan
        inst.strategy = "mean"
        inst.fill_value = None
        inst.copy = True
        inst.add_indicator = False
        inst.keep_empty_features = False
        inst._is_numeric = True
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustSimpleImputer.from_bytes(state, target_dev)
        self.missing_values = np.nan
        self.strategy = "mean"
        self.fill_value = None
        self.copy = True
        self.add_indicator = False
        self.keep_empty_features = False
        self._is_numeric = True
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "SimpleImputer":
        """Fit the imputer on X (*≧ω≦*)."""
        is_num, payload = as_2d_array_or_rows(X)
        self._is_numeric = is_num
        if is_num:
            self._model.fit_buffer(payload)
        else:
            self._model.fit_objects(payload)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Impute all missing values in X (＾▽＾)."""
        is_num, payload = as_2d_array_or_rows(X)
        if is_num and self._is_numeric:
            if not self.add_indicator and len(self._model.keep_cols_) == payload.shape[1]:
                out = np.empty(payload.shape, dtype=np.float64)
                if self._model.try_transform_into(payload, out):
                    return out
            rows, cols, raw = self._model.transform_buffer(payload)
            return bytes_to_f64_2d(rows, cols, raw)
        rows_in = payload.tolist() if is_num else payload
        return np.asarray(self._model.transform_objects(rows_in), dtype=object)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it (o´∀｀o)."""
        is_num, payload = as_2d_array_or_rows(X)
        self._is_numeric = is_num
        if is_num:
            if not self.add_indicator:
                out = np.empty(payload.shape, dtype=np.float64)
                if self._model.try_fit_transform_into(payload, out):
                    return out
            self._model.fit_buffer(payload)
            rows, cols, raw = self._model.transform_buffer(payload)
            return bytes_to_f64_2d(rows, cols, raw)
        self._model.fit_objects(payload)
        return np.asarray(self._model.transform_objects(payload), dtype=object)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Convert the data back to the original representation when add_indicator=True (＾▽＾)."""
        arr = np.asarray(X, dtype=np.float64)
        if not self.add_indicator:
            raise ValueError("inverse_transform is only supported when add_indicator=True ( >_< )")
        ind_feats = self._model.indicator_features_ or []
        n_keep = len(self._model.keep_cols_)
        data = arr[:, :n_keep].copy()
        ind = arr[:, n_keep:].astype(bool)
        for k, feat_idx in enumerate(ind_feats):
            if feat_idx < n_keep:
                data[ind[:, k], feat_idx] = self.missing_values
        return data

    @property
    def statistics_(self) -> Optional[np.ndarray]:
        st = self._model.statistics_
        return None if st is None else np.asarray(st)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save SimpleImputer to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SimpleImputer":
        """Load SimpleImputer from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustSimpleImputer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "SimpleImputer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"SimpleImputer(strategy='{self.strategy}', add_indicator={self.add_indicator}) (o´∀｀o)"


class KNNImputer:
    """
    Imputation for completing missing values using k-Nearest Neighbors (*≧ω≦*)
    Accelerated by Rust Backend N3 (pairwise_l2_sq / matmul) and N4 (topk_argmin).

    Parameters
    ----------
    missing_values : int, float, or np.nan, default=np.nan
        The placeholder for missing values.
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'nan_euclidean'}, default='nan_euclidean'
        Distance metric for searching neighbors.
    copy : bool, default=True
        Whether to return a copy.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will stack onto output.
    keep_empty_features : bool, default=False
        If True, features consisting exclusively of missing values are retained.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustKNNImputer

    def __init__(
        self,
        *,
        missing_values: float = np.nan,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Optional[str] = None,
    ):
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = _resolve_device(device)
        self._model = _RustKNNImputer(
            missing_values=float(missing_values),
            n_neighbors=n_neighbors,
            weights=weights,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustKNNImputer, device: str) -> "KNNImputer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.missing_values = np.nan
        inst.n_neighbors = 5
        inst.weights = "uniform"
        inst.metric = "nan_euclidean"
        inst.copy = True
        inst.add_indicator = False
        inst.keep_empty_features = False
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustKNNImputer.from_bytes(state, target_dev)
        self.missing_values = np.nan
        self.n_neighbors = 5
        self.weights = "uniform"
        self.metric = "nan_euclidean"
        self.copy = True
        self.add_indicator = False
        self.keep_empty_features = False
        self.device = self._model.device_ or target_dev

    def _prep_x(self, X: Any) -> np.ndarray:
        is_buf, X_data = process_features(X)
        return X_data if is_buf else np.asarray(X_data, dtype=np.float64)

    def fit(self, X: Any, y: Optional[Any] = None) -> "KNNImputer":
        """Fit the imputer on X (＾▽＾)."""
        self._model.fit_buffer(self._prep_x(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Impute all missing values in X (*≧ω≦*)."""
        rows, cols, raw = self._model.transform_buffer(self._prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it (o´∀｀o)."""
        return self.fit(X, y).transform(X)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save KNNImputer to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KNNImputer":
        """Load KNNImputer from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustKNNImputer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "KNNImputer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"KNNImputer(n_neighbors={self.n_neighbors}, weights='{self.weights}') (*≧ω≦*)"


class IterativeImputer:
    """
    Multivariate imputer that estimates each feature from all the others (＾▽＾)
    Defaults to OchreML Ridge regression with hardware acceleration.

    Parameters
    ----------
    estimator : estimator object or None, default=None
        The estimator to use at each step of the round-robin imputation (defaults to OchreML Ridge).
    missing_values : float or np.nan, default=np.nan
        The placeholder for missing values.
    sample_posterior : bool, default=False
        Kept for API compatibility.
    max_iter : int, default=10
        Maximum number of imputation rounds.
    tol : float, default=1e-3
        Tolerance of the stopping condition.
    n_nearest_features : int or None, default=None
        Number of other features to use to estimate the missing values of each feature column.
    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        Which strategy to use to initialize the missing values.
    fill_value : float or None, default=None
        Fill value when initial_strategy='constant'.
    imputation_order : {'ascending', 'descending', 'roman', 'arabic', 'random'}, default='ascending'
        The order in which the features will be imputed.
    skip_complete : bool, default=False
        If True, features with no missing values during fit are skipped.
    min_value : float or array-like, default=-np.inf
        Minimum possible imputed value.
    max_value : float or array-like, default=np.inf
        Maximum possible imputed value.
    verbose : int, default=0
        Verbosity flag.
    random_state : int or None, default=None
        Seed for random imputation order / n_nearest_features.
    add_indicator : bool, default=False
        If True, append MissingIndicator columns.
    keep_empty_features : bool, default=False
        If True, retain all-missing columns.
    device : str or None, default=None
        Target compute device for default Ridge estimator.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        *,
        missing_values: float = np.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = None,
        initial_strategy: str = "mean",
        fill_value: Optional[float] = None,
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        min_value: Union[float, np.ndarray] = -np.inf,
        max_value: Union[float, np.ndarray] = np.inf,
        verbose: int = 0,
        random_state: Optional[int] = None,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Optional[str] = None,
    ):
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = _resolve_device(device)

        self.initial_imputer_: Optional[SimpleImputer] = None
        self.imputation_sequence_: List[Tuple[int, np.ndarray, Any]] = []
        self.indicator_: Optional[MissingIndicator] = None
        self.n_iter_: int = 0
        self.n_features_in_: Optional[int] = None
        self.n_features_with_missing_: int = 0

    def _get_base_estimator(self) -> Any:
        if self.estimator is not None:
            return self.estimator
        from . import Ridge

        return Ridge(alpha=1.0, device=self.device)

    def _missing_mask(self, X: np.ndarray) -> np.ndarray:
        if np.isnan(self.missing_values):
            return np.isnan(X)
        return X == self.missing_values

    def _ordered_features(self, missing_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_feats = missing_mask.shape[1]
        n_miss = np.sum(missing_mask, axis=0)
        if self.skip_complete:
            candidates = np.where(n_miss > 0)[0]
        else:
            candidates = np.arange(n_feats)
        if len(candidates) == 0:
            return candidates
        if self.imputation_order == "ascending":
            return candidates[np.argsort(n_miss[candidates], kind="mergesort")]
        if self.imputation_order == "descending":
            return candidates[np.argsort(-n_miss[candidates], kind="mergesort")]
        if self.imputation_order == "roman":
            return candidates
        if self.imputation_order == "arabic":
            return candidates[::-1]
        if self.imputation_order == "random":
            order = candidates.copy()
            rng.shuffle(order)
            return order
        raise ValueError(f"Invalid imputation_order '{self.imputation_order}' ( >_< )")

    def _neighbor_indices(
        self, feat_idx: int, X_filled: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_feats = X_filled.shape[1]
        others = np.array([j for j in range(n_feats) if j != feat_idx], dtype=np.int64)
        if self.n_nearest_features is None or self.n_nearest_features >= len(others):
            return others
        # Correlation-based sampling matching sklearn
        col_f = X_filled[:, feat_idx]
        corrs = np.zeros(len(others), dtype=np.float64)
        for idx, j in enumerate(others):
            col_j = X_filled[:, j]
            c = np.corrcoef(col_f, col_j)[0, 1]
            corrs[idx] = abs(c) if np.isfinite(c) else 0.0
        top_idx = np.argsort(-corrs)[: self.n_nearest_features]
        return np.sort(others[top_idx])

    def _clip_col(self, vals: np.ndarray, feat_idx: int) -> np.ndarray:
        lo = self.min_value[feat_idx] if isinstance(self.min_value, (list, tuple, np.ndarray)) else self.min_value
        hi = self.max_value[feat_idx] if isinstance(self.max_value, (list, tuple, np.ndarray)) else self.max_value
        return np.clip(vals, lo, hi)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit the imputer on X and return the transformed X (*≧ω≦*)."""
        is_buf, X_data = process_features(X)
        X_arr = X_data if is_buf else np.asarray(X_data, dtype=np.float64)
        self.n_features_in_ = X_arr.shape[1]
        full_mask = self._missing_mask(X_arr)

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values,
                features="missing-only",
                error_on_new=False,
                device=self.device,
            ).fit(X_arr)
        else:
            self.indicator_ = None

        self.initial_imputer_ = SimpleImputer(
            missing_values=self.missing_values,
            strategy=self.initial_strategy,
            fill_value=self.fill_value,
            keep_empty_features=self.keep_empty_features,
            device=self.device,
        )
        X_filled = self.initial_imputer_.fit_transform(X_arr).copy()
        keep_cols = self.initial_imputer_._model.keep_cols_
        mask = full_mask[:, keep_cols]
        self.n_features_with_missing_ = X_filled.shape[1]

        self.imputation_sequence_ = []
        self.n_iter_ = 0
        if X_filled.shape[1] <= 1 or not np.any(mask):
            if self.indicator_ is not None:
                return np.hstack([X_filled, self.indicator_.transform(X_arr).astype(np.float64)])
            return X_filled

        obs_vals = X_filled[~mask]
        scale_ref = float(np.max(np.abs(obs_vals))) if obs_vals.size > 0 else 1.0
        if scale_ref == 0.0:
            scale_ref = 1.0

        rng = np.random.default_rng(self.random_state)
        base_est = self._get_base_estimator()
        X_prev = X_filled.copy()

        for it in range(self.max_iter):
            order = self._ordered_features(mask, rng)
            for feat_idx in order:
                neighbors = self._neighbor_indices(int(feat_idx), X_filled, rng)
                if len(neighbors) == 0:
                    continue
                miss_row = mask[:, feat_idx]
                obs_row = ~miss_row
                if not np.any(obs_row):
                    continue
                est = copy.deepcopy(base_est)
                est.fit(X_filled[np.ix_(obs_row, neighbors)], X_filled[obs_row, feat_idx])
                if np.any(miss_row):
                    preds = est.predict(X_filled[np.ix_(miss_row, neighbors)])
                    X_filled[miss_row, feat_idx] = self._clip_col(np.asarray(preds, dtype=np.float64), int(feat_idx))
                self.imputation_sequence_.append((int(feat_idx), neighbors, est))

            self.n_iter_ = it + 1
            if np.any(mask):
                diff = float(np.max(np.abs(X_filled - X_prev)))
                if diff / scale_ref < self.tol:
                    break
            X_prev = X_filled.copy()

        if self.indicator_ is not None:
            return np.hstack([X_filled, self.indicator_.transform(X_arr).astype(np.float64)])
        return X_filled

    def fit(self, X: Any, y: Optional[Any] = None) -> "IterativeImputer":
        """Fit the imputer on X (＾▽＾)."""
        self.fit_transform(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Impute all missing values in X using fitted round-robin estimators (o´∀｀o)."""
        if self.initial_imputer_ is None or self.n_features_in_ is None:
            raise ValueError("IterativeImputer is not fitted yet (´-ω-`)")
        is_buf, X_data = process_features(X)
        X_arr = X_data if is_buf else np.asarray(X_data, dtype=np.float64)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: got {X_arr.shape[1]}, expected {self.n_features_in_} ( >_< )"
            )
        full_mask = self._missing_mask(X_arr)
        X_filled = self.initial_imputer_.transform(X_arr).copy()
        keep_cols = self.initial_imputer_._model.keep_cols_
        mask = full_mask[:, keep_cols]

        for feat_idx, neighbors, est in self.imputation_sequence_:
            miss_row = mask[:, feat_idx]
            if np.any(miss_row):
                preds = est.predict(X_filled[np.ix_(miss_row, neighbors)])
                X_filled[miss_row, feat_idx] = self._clip_col(
                    np.asarray(preds, dtype=np.float64), feat_idx
                )

        if self.indicator_ is not None:
            return np.hstack([X_filled, self.indicator_.transform(X_arr).astype(np.float64)])
        return X_filled

    def save(self, path: str) -> None:
        """Save IterativeImputer to a '.dit' file (*≧ω≦*)."""
        save_dit_envelope(_TAG_ITERATIVE_IMPUTER, self.__dict__, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "IterativeImputer":
        """Load IterativeImputer from a '.dit' file (＾▽＾)."""
        with open(validate_dit_path(path), "rb") as f:
            state = load_dit_envelope(_TAG_ITERATIVE_IMPUTER, f.read())
        inst = cls.__new__(cls)
        inst.__dict__.update(state)
        if device is not None:
            inst.device = device
        return inst

    def __repr__(self) -> str:
        return f"IterativeImputer(max_iter={self.max_iter}, imputation_order='{self.imputation_order}') (＾▽＾)"


__all__ = [
    "SimpleImputer",
    "MissingIndicator",
    "KNNImputer",
    "IterativeImputer",
]

