"""
OchreML Composite Transformers & Target Regressors (*≧ω≦*)
Implements ColumnTransformer, TransformedTargetRegressor, make_column_transformer,
and make_column_selector with .dit persistence and full Scikit-Learn compatibility.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._preproc_utils import load_dit_envelope, save_dit_envelope, validate_dit_path
from .preprocessing._transforms import FunctionTransformer

_TAG_COLUMN_TRANSFORMER = 36
_TAG_TRANSFORMED_TARGET_REGRESSOR = 37


def _is_sparse(x: Any) -> bool:
    return hasattr(x, "tocsr") and hasattr(x, "nnz")


def _extract_column_names_and_dtypes(X: Any) -> Tuple[Optional[List[str]], Optional[List[Any]]]:
    module = type(X).__module__
    if "pandas" in module and hasattr(X, "columns"):
        return [str(c) for c in X.columns], list(X.dtypes)
    if "polars" in module and hasattr(X, "columns"):
        return [str(c) for c in X.columns], list(X.dtypes)
    return None, None


def _select_columns(X: Any, cols: Any) -> Any:
    if callable(cols):
        cols = cols(X)

    col_names, _ = _extract_column_names_and_dtypes(X)
    is_df = col_names is not None

    if isinstance(cols, (str, int, np.integer)):
        if is_df:
            if isinstance(cols, (int, np.integer)):
                name = col_names[int(cols)]
                return X[[name]] if hasattr(X, "select") else X.iloc[:, [int(cols)]]
            return X[[cols]] if hasattr(X, "select") else X[[cols]]
        arr = np.asarray(X)
        return arr[:, [int(cols)]]

    if isinstance(cols, slice):
        if is_df:
            idx = list(range(len(col_names)))[cols]
            names = [col_names[i] for i in idx]
            return X.select(names) if hasattr(X, "select") else X[names]
        return np.asarray(X)[:, cols]

    cols_list = list(cols)
    if len(cols_list) == 0:
        n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.empty((n_rows, 0), dtype=np.float64)

    if all(isinstance(c, (bool, np.bool_)) for c in cols_list):
        if is_df:
            names = [name for name, keep in zip(col_names, cols_list) if keep]
            return X.select(names) if hasattr(X, "select") else X[names]
        return np.asarray(X)[:, np.asarray(cols_list, dtype=bool)]

    if all(isinstance(c, str) for c in cols_list):
        if not is_df:
            raise ValueError("String column names require a DataFrame input ( >_< )")
        return X.select(cols_list) if hasattr(X, "select") else X[cols_list]

    idx_list = [int(c) for c in cols_list]
    if is_df:
        names = [col_names[i] for i in idx_list]
        return X.select(names) if hasattr(X, "select") else X[names]
    return np.asarray(X)[:, idx_list]


def _resolve_col_indices(X: Any, cols: Any) -> List[int]:
    if callable(cols):
        cols = cols(X)
    col_names, _ = _extract_column_names_and_dtypes(X)
    n_cols = len(col_names) if col_names is not None else np.asarray(X).shape[1]

    if isinstance(cols, str):
        if col_names is None:
            raise ValueError("String column names require a DataFrame input ( >_< )")
        return [col_names.index(cols)]
    if isinstance(cols, (int, np.integer)):
        idx = int(cols)
        return [idx if idx >= 0 else n_cols + idx]
    if isinstance(cols, slice):
        return list(range(n_cols))[cols]

    cols_list = list(cols)
    if len(cols_list) == 0:
        return []
    if all(isinstance(c, (bool, np.bool_)) for c in cols_list):
        return [i for i, keep in enumerate(cols_list) if keep]
    if all(isinstance(c, str) for c in cols_list):
        if col_names is None:
            raise ValueError("String column names require a DataFrame input ( >_< )")
        return [col_names.index(c) for c in cols_list]
    return [int(c) if int(c) >= 0 else n_cols + int(c) for c in cols_list]


def _ensure_2d(out: Any) -> Any:
    if _is_sparse(out):
        return out
    arr = np.asarray(out)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


class make_column_selector:
    """
    Create a callable to select columns to be used with ColumnTransformer (＾▽＾)

    Parameters
    ----------
    pattern : str or None, default=None
        Name of columns containing this regex pattern will be included.
    dtype_include : column dtype or list of dtypes, default=None
        A selection of dtypes to include.
    dtype_exclude : column dtype or list of dtypes, default=None
        A selection of dtypes to exclude.
    """

    def __init__(
        self,
        pattern: Optional[str] = None,
        *,
        dtype_include: Optional[Any] = None,
        dtype_exclude: Optional[Any] = None,
    ):
        self.pattern = pattern
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df: Any) -> List[str]:
        if hasattr(df, "select_dtypes"):
            # Pandas DataFrame
            sub = df
            if self.dtype_include is not None or self.dtype_exclude is not None:
                sub = sub.select_dtypes(
                    include=self.dtype_include, exclude=self.dtype_exclude
                )
            cols = [str(c) for c in sub.columns]
            if self.pattern is not None:
                pat = re.compile(self.pattern)
                cols = [c for c in cols if pat.search(c)]
            return cols

        col_names, dtypes = _extract_column_names_and_dtypes(df)
        if col_names is None:
            raise ValueError("make_column_selector requires a DataFrame ( >_< )")
        cols = col_names
        if self.pattern is not None:
            pat = re.compile(self.pattern)
            cols = [c for c in cols if pat.search(c)]
        return cols


class ColumnTransformer:
    """
    Applies transformers to columns of an array or pandas/polars DataFrame (*≧ω≦*)

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the transformer
        objects to be applied to subsets of the data.
    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        What to do with columns not explicitly selected.
    sparse_threshold : float, default=0.3
        Threshold below which output is returned as sparse CSR matrix if any block is sparse.
    n_jobs : int or None, default=None
        Kept for API compatibility.
    transformer_weights : dict or None, default=None
        Multiplicative weights for features per transformer.
    verbose : bool, default=False
        Kept for API compatibility.
    verbose_feature_names_out : bool, default=True
        Prefix feature names with transformer name in get_feature_names_out.
    force_int_remainder_cols : bool, default=True
        Kept for API compatibility.
    """

    def __init__(
        self,
        transformers: Sequence[Tuple[str, Any, Any]],
        *,
        remainder: Union[str, Any] = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
        force_int_remainder_cols: bool = True,
    ):
        self.transformers = list(transformers)
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.force_int_remainder_cols = force_int_remainder_cols

        self.transformers_: List[Tuple[str, Any, Any]] = []
        self.output_indices_: Dict[str, slice] = {}
        self.sparse_output_: bool = False
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None

    @property
    def named_transformers_(self) -> Dict[str, Any]:
        return {name: trans for name, trans, _ in self.transformers_}

    def _fit_transform_internal(self, X: Any, y: Optional[Any], fit: bool) -> Any:
        col_names, _ = _extract_column_names_and_dtypes(X)
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        n_cols = len(col_names) if col_names is not None else np.asarray(X).shape[1]

        if fit:
            self.n_features_in_ = n_cols
            self.feature_names_in_ = (
                np.asarray(col_names, dtype=object) if col_names is not None else None
            )
            self.transformers_ = []
            used_indices = set()
            for name, trans, cols in self.transformers:
                resolved_cols = cols(X) if callable(cols) else cols
                idx_list = _resolve_col_indices(X, resolved_cols)
                used_indices.update(idx_list)
                if trans == "drop" or len(idx_list) == 0:
                    self.transformers_.append((name, "drop", resolved_cols))
                elif trans == "passthrough":
                    self.transformers_.append((name, "passthrough", resolved_cols))
                else:
                    self.transformers_.append((name, copy.deepcopy(trans), resolved_cols))

            rem_idx = [i for i in range(n_cols) if i not in used_indices]
            if rem_idx:
                if self.remainder == "drop":
                    self.transformers_.append(("remainder", "drop", rem_idx))
                elif self.remainder == "passthrough":
                    self.transformers_.append(("remainder", "passthrough", rem_idx))
                else:
                    self.transformers_.append(("remainder", copy.deepcopy(self.remainder), rem_idx))

        Xs: List[Any] = []
        self.output_indices_ = {}
        col_cursor = 0
        for name, trans, cols in self.transformers_:
            if trans == "drop":
                self.output_indices_[name] = slice(0, 0)
                continue
            X_sub = _select_columns(X, cols)
            if trans == "passthrough":
                out = _ensure_2d(X_sub.to_numpy() if hasattr(X_sub, "to_numpy") else X_sub)
            elif fit:
                if hasattr(trans, "fit_transform"):
                    out = trans.fit_transform(X_sub, y) if y is not None else trans.fit_transform(X_sub)
                else:
                    trans.fit(X_sub, y) if y is not None else trans.fit(X_sub)
                    out = trans.transform(X_sub)
                out = _ensure_2d(out)
            else:
                out = _ensure_2d(trans.transform(X_sub))

            if self.transformer_weights and name in self.transformer_weights:
                out = out * self.transformer_weights[name]

            width = out.shape[1]
            self.output_indices_[name] = slice(col_cursor, col_cursor + width)
            col_cursor += width
            Xs.append(out)

        if not Xs:
            self.sparse_output_ = False
            return np.empty((n_samples, 0), dtype=np.float64)

        has_sparse = any(_is_sparse(x) for x in Xs)
        if has_sparse and self.sparse_threshold > 0:
            total_elems = n_samples * col_cursor
            nnz_total = sum(
                x.nnz if _is_sparse(x) else x.shape[0] * x.shape[1] for x in Xs
            )
            density = nnz_total / total_elems if total_elems > 0 else 1.0
            self.sparse_output_ = density <= self.sparse_threshold
        else:
            self.sparse_output_ = False

        if self.sparse_output_:
            from scipy import sparse

            blocks = [x if _is_sparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float64)) for x in Xs]
            return sparse.hstack(blocks).tocsr()

        dense_blocks = [
            x.toarray() if _is_sparse(x) else np.asarray(x) for x in Xs
        ]
        return np.hstack(dense_blocks)

    def fit(self, X: Any, y: Optional[Any] = None) -> "ColumnTransformer":
        """Fit all transformers using X (＾▽＾)."""
        self._fit_transform_internal(X, y, fit=True)
        return self

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit all transformers, transform the data and concatenate results (*≧ω≦*)."""
        return self._fit_transform_internal(X, y, fit=True)

    def transform(self, X: Any) -> Any:
        """Transform X separately by each transformer, concatenate results (o´∀｀o)."""
        if not self.transformers_:
            raise ValueError("ColumnTransformer is not fitted yet (´-ω-`)")
        return self._fit_transform_internal(X, None, fit=False)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        if self.n_features_in_ is None:
            raise ValueError("ColumnTransformer is not fitted yet (´-ω-`)")
        in_names = (
            list(input_features)
            if input_features is not None
            else (
                self.feature_names_in_.tolist()
                if self.feature_names_in_ is not None
                else [f"x{i}" for i in range(self.n_features_in_)]
            )
        )
        out_names: List[str] = []
        for name, trans, cols in self.transformers_:
            if trans == "drop":
                continue
            if isinstance(cols, slice):
                sub_names = in_names[cols]
            elif isinstance(cols, (str, int)):
                sub_names = [in_names[cols] if isinstance(cols, int) else cols]
            else:
                sub_names = [
                    in_names[c] if isinstance(c, (int, np.integer)) else str(c)
                    for c in cols
                ]

            if trans == "passthrough":
                feat_names = sub_names
            elif hasattr(trans, "get_feature_names_out"):
                feat_names = list(trans.get_feature_names_out(sub_names))
            else:
                width = self.output_indices_[name].stop - self.output_indices_[name].start
                feat_names = sub_names if len(sub_names) == width else [f"x{i}" for i in range(width)]

            if self.verbose_feature_names_out:
                out_names.extend(f"{name}__{fn}" for fn in feat_names)
            else:
                out_names.extend(str(fn) for fn in feat_names)
        return np.asarray(out_names, dtype=object)

    def save(self, path: str) -> None:
        """Save ColumnTransformer to a '.dit' file (*≧ω≦*)."""
        save_dit_envelope(_TAG_COLUMN_TRANSFORMER, self.__dict__, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ColumnTransformer":
        """Load ColumnTransformer from a '.dit' file (＾▽＾)."""
        with open(validate_dit_path(path), "rb") as f:
            state = load_dit_envelope(_TAG_COLUMN_TRANSFORMER, f.read())
        inst = cls.__new__(cls)
        inst.__dict__.update(state)
        return inst

    def __repr__(self) -> str:
        return f"ColumnTransformer(transformers={len(self.transformers)}, remainder={self.remainder!r}) (*≧ω≦*)"


def make_column_transformer(
    *transformers: Tuple[Any, Any],
    remainder: Union[str, Any] = "drop",
    sparse_threshold: float = 0.3,
    n_jobs: Optional[int] = None,
    verbose_feature_names_out: bool = True,
    verbose: bool = False,
    force_int_remainder_cols: bool = True,
) -> ColumnTransformer:
    """Construct a ColumnTransformer from the given transformers (＾▽＾)."""
    named: List[Tuple[str, Any, Any]] = []
    seen_names: Dict[str, int] = {}
    for trans, cols in transformers:
        if isinstance(trans, str):
            base_name = trans
        else:
            base_name = type(trans).__name__.lower()
        cnt = seen_names.get(base_name, 0) + 1
        seen_names[base_name] = cnt
        name = f"{base_name}-{cnt}" if cnt > 1 else base_name
        named.append((name, trans, cols))
    return ColumnTransformer(
        named,
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        n_jobs=n_jobs,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
        force_int_remainder_cols=force_int_remainder_cols,
    )


class TransformedTargetRegressor:
    """
    Meta-estimator to regress on a transformed target y (o´∀｀o)

    Parameters
    ----------
    regressor : object or None, default=None
        Regressor object (defaults to OchreML LinearRegression).
    transformer : object or None, default=None
        Estimator object to transform the target y.
    func : callable or None, default=None
        Function to apply to y before passing to fit (ignored if transformer is set).
    inverse_func : callable or None, default=None
        Inverse function to apply to predictions.
    check_inverse : bool, default=True
        Whether to check that transform followed by inverse_transform is identity.
    """

    def __init__(
        self,
        regressor: Optional[Any] = None,
        *,
        transformer: Optional[Any] = None,
        func: Optional[Callable[..., Any]] = None,
        inverse_func: Optional[Callable[..., Any]] = None,
        check_inverse: bool = True,
    ):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse

        self.regressor_: Any = None
        self.transformer_: Any = None
        self.n_features_in_: Optional[int] = None
        self._training_dim: int = 1

    def _fit_transformer(self, y: np.ndarray) -> None:
        if self.transformer is not None:
            self.transformer_ = copy.deepcopy(self.transformer)
        elif self.func is not None or self.inverse_func is not None:
            self.transformer_ = FunctionTransformer(
                func=self.func,
                inverse_func=self.inverse_func,
                validate=True,
                check_inverse=self.check_inverse,
            )
        else:
            self.transformer_ = FunctionTransformer(func=None, inverse_func=None, validate=False)
        self.transformer_.fit(y)

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "TransformedTargetRegressor":
        """Fit the model according to the given training data (*≧ω≦*)."""
        y_arr = np.asarray(y, dtype=np.float64)
        self._training_dim = y_arr.ndim
        y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

        self._fit_transformer(y_2d)
        y_trans = self.transformer_.transform(y_2d)
        if self._training_dim == 1 and y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.ravel()

        if self.regressor is None:
            from . import LinearRegression

            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = copy.deepcopy(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)
        self.n_features_in_ = getattr(self.regressor_, "n_features_in_", None)
        return self

    def predict(self, X: Any, **predict_params: Any) -> np.ndarray:
        """Predict using the base regressor, applying inverse target transform (＾▽＾)."""
        if self.regressor_ is None or self.transformer_ is None:
            raise ValueError("TransformedTargetRegressor is not fitted yet (´-ω-`)")
        pred = np.asarray(self.regressor_.predict(X, **predict_params), dtype=np.float64)
        pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred
        inv = np.asarray(self.transformer_.inverse_transform(pred_2d), dtype=np.float64)
        if self._training_dim == 1 and inv.ndim == 2 and inv.shape[1] == 1:
            return inv.ravel()
        return inv

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """Return R^2 score of the prediction (o´∀｀o)."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()[:, None]
            y_mean = np.sum(sw * y_arr, axis=0) / np.sum(sw)
            u = np.sum(sw * (y_arr - preds) ** 2, axis=0)
            v = np.sum(sw * (y_arr - y_mean) ** 2, axis=0)
        else:
            y_mean = np.mean(y_arr, axis=0)
            u = np.sum((y_arr - preds) ** 2, axis=0)
            v = np.sum((y_arr - y_mean) ** 2, axis=0)
        r2 = np.where(v <= 1e-12, np.where(u <= 1e-12, 1.0, 0.0), 1.0 - u / v)
        return float(np.mean(r2))

    def save(self, path: str) -> None:
        """Save TransformedTargetRegressor to a '.dit' file (*≧ω≦*)."""
        save_dit_envelope(_TAG_TRANSFORMED_TARGET_REGRESSOR, self.__dict__, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TransformedTargetRegressor":
        """Load TransformedTargetRegressor from a '.dit' file (＾▽＾)."""
        with open(validate_dit_path(path), "rb") as f:
            state = load_dit_envelope(_TAG_TRANSFORMED_TARGET_REGRESSOR, f.read())
        inst = cls.__new__(cls)
        inst.__dict__.update(state)
        return inst

    def __repr__(self) -> str:
        return (
            f"TransformedTargetRegressor(regressor={self.regressor!r}, "
            f"transformer={self.transformer!r}) (o´∀｀o)"
        )


__all__ = [
    "ColumnTransformer",
    "TransformedTargetRegressor",
    "make_column_transformer",
    "make_column_selector",
]

