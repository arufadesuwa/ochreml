# OchreML utility module (＾▽＾)
# Data format converters: List, NumPy ndarray, Pandas DataFrame/Series, and Polars DataFrame/Series.

from typing import Any, List


def to_feature_matrix(X: Any) -> List[List[float]]:
    """
    Convert feature input X to a 2D matrix list[list[float]] (*≧ω≦*)
    Supports: Python nested lists, NumPy ndarrays, Pandas DataFrames, and Polars DataFrames.
    """
    if X is None:
        raise ValueError("Feature matrix X cannot be None (´；ω；`)")

    type_name = type(X).__name__
    module_name = type(X).__module__

    # 1. Convert from Polars DataFrame
    if "polars" in module_name or (type_name == "DataFrame" and hasattr(X, "iter_rows")):
        try:
            if hasattr(X, "to_numpy"):
                import numpy as np
                return np.asarray(X.to_numpy(), dtype=float).tolist()
            else:
                return [[float(val) for val in row] for row in X.iter_rows()]
        except Exception:
            return [[float(val) for val in row] for row in X.iter_rows()]

    # 2. Convert from Pandas DataFrame
    if "pandas" in module_name or (hasattr(X, "to_numpy") and hasattr(X, "columns")):
        import numpy as np
        arr = X.to_numpy()
        return np.asarray(arr, dtype=float).tolist()

    # 3. Convert from NumPy ndarray
    if hasattr(X, "__array__") or type_name == "ndarray":
        import numpy as np
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            raise ValueError(
                "Feature matrix X must be a 2D array (n_samples, n_features) ( >_< )\n"
                "Tip: Use X.reshape(-1, 1) if data has only 1 feature."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"Feature matrix dimension ({arr.ndim}D) is not supported. Must be 2D matrix (・`ω´・)"
            )
        return arr.tolist()

    # 4. Convert from standard Python nested lists
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            raise ValueError("Feature matrix X cannot be empty (´-ω-`)")
        first = X[0]
        if not isinstance(first, (list, tuple)):
            raise ValueError(
                "Feature matrix X must be a nested list [[x1, x2], ...] instead of a 1D list ( >_< )"
            )
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
        return matrix

    raise TypeError(
        f"Data format {type(X)} is not supported. Please use List, NumPy, Pandas, or Polars (T_T)"
    )


def to_target_vector(y: Any) -> List[float]:
    """
    Convert target y to a 1D vector list[float] (o´∀｀o)
    Supports: Python lists, NumPy 1D ndarrays, Pandas Series, and Polars Series.
    """
    if y is None:
        raise ValueError("Target vector y cannot be None (´；ω；`)")

    type_name = type(y).__name__
    module_name = type(y).__module__

    # 1. Convert from Polars Series / 1-column DataFrame
    if "polars" in module_name:
        try:
            if hasattr(y, "to_list"):
                return [float(v) for v in y.to_list()]
            if hasattr(y, "to_numpy"):
                import numpy as np
                return np.asarray(y.to_numpy(), dtype=float).ravel().tolist()
        except Exception:
            pass

    # 2. Convert from Pandas Series / DataFrame
    if "pandas" in module_name or hasattr(y, "to_numpy"):
        import numpy as np
        arr = np.asarray(y.to_numpy(), dtype=float).ravel()
        return arr.tolist()

    # 3. Convert from NumPy ndarray
    if hasattr(y, "__array__") or type_name == "ndarray":
        import numpy as np
        arr = np.asarray(y, dtype=float).ravel()
        return arr.tolist()

    # 4. Convert from standard Python list
    if isinstance(y, (list, tuple)):
        return [float(v) for v in y]

    raise TypeError(
        f"Target format ({type(y)}) is not supported. Please use List, NumPy, Pandas, or Polars (・`ω´・)"
    )
