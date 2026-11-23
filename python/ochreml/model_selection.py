"""
OchreML Model Selection Splitters Module
Provides Scikit-Learn-compatible train_test_split, KFold, StratifiedKFold,
GroupKFold, and TimeSeriesSplit backed by Rust index generators.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    group_kfold_split as _rust_group_kfold_split,
    kfold_split as _rust_kfold_split,
    stratified_kfold_split as _rust_stratified_kfold_split,
    time_series_split as _rust_time_series_split,
    train_test_split_indices as _rust_train_test_split_indices,
)
from ._preproc_utils import bytes_to_i64_1d


def _num_samples(x: Any) -> int:
    if hasattr(x, "shape"):
        return int(x.shape[0])
    return len(x)


def _to_i64_labels(arr: Any) -> List[int]:
    if hasattr(arr, "to_numpy"):
        np_arr = arr.to_numpy()
    else:
        np_arr = np.asarray(arr)
    if np_arr.ndim != 1:
        np_arr = np_arr.ravel()
    if np.issubdtype(np_arr.dtype, np.integer) or np_arr.dtype == np.bool_:
        return [int(v) for v in np_arr.tolist()]
    _, inv = np.unique(np_arr, return_inverse=True)
    return [int(v) for v in inv.tolist()]


def _slice_array(arr: Any, indices: Any) -> Any:
    module = type(arr).__module__
    # Pandas DataFrame / Series
    if "pandas" in module and hasattr(arr, "iloc"):
        return arr.iloc[indices]
    # Polars DataFrame / Series
    if "polars" in module:
        idx_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        if hasattr(arr, "take"):
            return arr.take(idx_list)
        return arr[idx_list]
    # NumPy ndarray or array-like with integer array indexing
    if isinstance(arr, np.ndarray):
        return arr[indices]
    if isinstance(arr, list):
        return [arr[int(i)] for i in indices]
    if isinstance(arr, tuple):
        return tuple(arr[int(i)] for i in indices)
    return np.asarray(arr)[indices]


class KFold:
    """K-Fold cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if not shuffle and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is False."
            )
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        n_samples = _num_samples(X)
        folds = _rust_kfold_split(
            n_samples,
            self.n_splits,
            self.shuffle,
            self.random_state,
        )
        for tr_b, te_b in folds:
            yield bytes_to_i64_1d(tr_b), bytes_to_i64_1d(te_b)


class StratifiedKFold:
    """Stratified K-Fold cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if not shuffle and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is False."
            )
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        if y is None:
            raise ValueError("The 'y' parameter should not be None for StratifiedKFold.")
        n_samples = _num_samples(X)
        y_labels = _to_i64_labels(y)
        if len(y_labels) != n_samples:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {n_samples} != {len(y_labels)}"
            )
        folds = _rust_stratified_kfold_split(
            y_labels,
            self.n_splits,
            self.shuffle,
            self.random_state,
        )
        for tr_b, te_b in folds:
            yield bytes_to_i64_1d(tr_b), bytes_to_i64_1d(te_b)


class GroupKFold:
    """K-fold iterator variant with non-overlapping groups."""

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = int(n_splits)

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Any = None,
    ) -> Iterator[Tuple[Any, Any]]:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        n_samples = _num_samples(X)
        group_labels = _to_i64_labels(groups)
        if len(group_labels) != n_samples:
            raise ValueError(
                f"X and groups have inconsistent numbers of samples: {n_samples} != {len(group_labels)}"
            )
        folds = _rust_group_kfold_split(group_labels, self.n_splits)
        for tr_b, te_b in folds:
            yield bytes_to_i64_1d(tr_b), bytes_to_i64_1d(te_b)


class TimeSeriesSplit:
    """Time Series cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ) -> None:
        self.n_splits = int(n_splits)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = int(gap)

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        n_samples = _num_samples(X)
        folds = _rust_time_series_split(
            n_samples,
            self.n_splits,
            self.max_train_size,
            self.test_size,
            self.gap,
        )
        for tr_b, te_b in folds:
            yield bytes_to_i64_1d(tr_b), bytes_to_i64_1d(te_b)


def _resolve_train_test_counts(
    n_samples: int,
    test_size: Optional[Union[float, int]],
    train_size: Optional[Union[float, int]],
) -> Tuple[int, int]:
    if test_size is None and train_size is None:
        test_size = 0.25

    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size={test_size} as float must be in (0.0, 1.0)")
        n_test = math.ceil(test_size * n_samples)
    elif isinstance(test_size, int) and not isinstance(test_size, bool):
        if test_size < 0 or test_size >= n_samples:
            raise ValueError(f"test_size={test_size} must be in [0, {n_samples})")
        n_test = int(test_size)
    elif test_size is None:
        n_test = -1
    else:
        raise ValueError(f"Invalid test_size: {test_size}")

    if isinstance(train_size, float):
        if not (0.0 < train_size < 1.0):
            raise ValueError(f"train_size={train_size} as float must be in (0.0, 1.0)")
        n_train = math.floor(train_size * n_samples)
    elif isinstance(train_size, int) and not isinstance(train_size, bool):
        if train_size < 0 or train_size >= n_samples:
            raise ValueError(f"train_size={train_size} must be in [0, {n_samples})")
        n_train = int(train_size)
    elif train_size is None:
        n_train = -1
    else:
        raise ValueError(f"Invalid train_size: {train_size}")

    if n_test == -1:
        n_test = n_samples - n_train
    if n_train == -1:
        n_train = n_samples - n_test

    if n_test + n_train > n_samples:
        raise ValueError(
            f"The sum of test_size ({n_test}) and train_size ({n_train}) exceeds n_samples ({n_samples})"
        )
    return n_train, n_test


def train_test_split(
    *arrays: Any,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[Any] = None,
) -> List[Any]:
    """Split arrays or matrices into random train and test subsets."""
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    n_samples = _num_samples(arrays[0])
    for arr in arrays[1:]:
        if _num_samples(arr) != n_samples:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: {[ _num_samples(a) for a in arrays ]}"
            )

    n_train, n_test = _resolve_train_test_counts(n_samples, test_size, train_size)
    stratify_labels = _to_i64_labels(stratify) if stratify is not None else None

    tr_b, te_b = _rust_train_test_split_indices(
        n_samples,
        n_test,
        n_train,
        bool(shuffle),
        random_state,
        stratify_labels,
    )
    train_idx = bytes_to_i64_1d(tr_b)
    test_idx = bytes_to_i64_1d(te_b)

    result: List[Any] = []
    for arr in arrays:
        result.append(_slice_array(arr, train_idx))
        result.append(_slice_array(arr, test_idx))
    return result


from ._meta_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

__all__ = [
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "TimeSeriesSplit",
    "GridSearchCV",
    "RandomizedSearchCV",
    "cross_val_score",
]
