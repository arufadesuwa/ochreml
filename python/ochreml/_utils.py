# OchreML utility module (＾▽＾)
# Data format converters & Zero-Copy Buffer Protocol processors (*≧ω≦*)
# Supports: NumPy ndarray, Polars DataFrame/Series, Pandas DataFrame/Series, and Python Lists.

from typing import Any, List, Tuple


def process_features(X: Any) -> Tuple[bool, Any]:
    """
    Process feature matrix X for Zero-Copy Buffer Protocol or Python fallback (*≧ω≦*)
    Returns (is_buffer, processed_data).
    """
    if X is None:
        raise ValueError("Feature matrix X cannot be None (´；ω；`)")

    type_name = type(X).__name__
    module_name = type(X).__module__

    # 1. NumPy ndarray, Polars DataFrame, or Pandas DataFrame
    if (
        hasattr(X, "__array__")
        or type_name == "ndarray"
        or "pandas" in module_name
        or "polars" in module_name
        or hasattr(X, "to_numpy")
    ):
        try:
            import numpy as np

            if hasattr(X, "to_numpy"):
                arr = X.to_numpy()
            else:
                arr = np.asarray(X)

            arr = np.ascontiguousarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                raise ValueError(
                    "Feature matrix X must be a 2D array (n_samples, n_features) ( >_< )\n"
                    "Tip: Use X.reshape(-1, 1) if data has only 1 feature."
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"Feature matrix dimension ({arr.ndim}D) is not supported. Must be 2D matrix (・`ω´・)"
                )
            return True, arr
        except ValueError as e:
            if "Feature matrix" in str(e):
                raise
        except Exception:
            pass

    # 2. Python nested lists / tuples
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            raise ValueError("Feature matrix X cannot be empty (´-ω-`)")
        first = X[0]
        if not isinstance(first, (list, tuple)):
            raise ValueError(
                "Feature matrix X must be a nested list [[x1, x2], ...] instead of a 1D list ( >_< )"
            )
        try:
            import numpy as np

            arr = np.ascontiguousarray(X, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError("Feature matrix X must be a 2D array ( >_< )")
            return True, arr
        except Exception:
            matrix = []
            expected_cols = len(first)
            for idx, row in enumerate(X):
                if not isinstance(row, (list, tuple)):
                    raise ValueError(
                        f"Element at row {idx} must be a list or tuple (・`ω´・)"
                    )
                if len(row) != expected_cols:
                    raise ValueError(
                        f"Row {idx} column count ({len(row)}) differs from first row ({expected_cols}) (´；д；`)"
                    )
                matrix.append([float(val) for val in row])
            return False, matrix

    raise TypeError(
        f"Data format {type(X)} is not supported. Please use List, NumPy, Pandas, or Polars (T_T)"
    )


def process_target(y: Any) -> Tuple[bool, Any]:
    """
    Process target vector y for Zero-Copy Buffer Protocol or Python fallback (o´∀｀o)
    Returns (is_buffer, processed_data).
    """
    if y is None:
        raise ValueError("Target vector y cannot be None (´；ω；`)")

    type_name = type(y).__name__
    module_name = type(y).__module__

    if (
        hasattr(y, "__array__")
        or type_name == "ndarray"
        or "pandas" in module_name
        or "polars" in module_name
        or hasattr(y, "to_numpy")
    ):
        try:
            import numpy as np

            if hasattr(y, "to_numpy"):
                arr = y.to_numpy()
            else:
                arr = np.asarray(y)
            arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
            return True, arr
        except Exception:
            pass

    if isinstance(y, (list, tuple)):
        try:
            import numpy as np

            arr = np.ascontiguousarray(y, dtype=np.float64).ravel()
            return True, arr
        except Exception:
            return False, [float(v) for v in y]

    raise TypeError(
        f"Target format ({type(y)}) is not supported. Please use List, NumPy, Pandas, or Polars (・`ω´・)"
    )


def to_feature_matrix(X: Any) -> List[List[float]]:
    """
    Convert feature input X to a 2D matrix list[list[float]] (*≧ω≦*)
    Supports: Python nested lists, NumPy ndarrays, Pandas DataFrames, and Polars DataFrames.
    """
    is_buf, data = process_features(X)
    if is_buf:
        return data.tolist()
    return data


def to_target_vector(y: Any) -> List[float]:
    """
    Convert target y to a 1D vector list[float] (o´∀｀o)
    Supports: Python lists, NumPy 1D ndarrays, Pandas Series, and Polars Series.
    """
    is_buf, data = process_target(y)
    if is_buf:
        return data.tolist()
    return data
