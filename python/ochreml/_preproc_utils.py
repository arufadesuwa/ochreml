# OchreML Preprocessing Python Helpers (*≧ω≦*)
# Zero-copy PyBytes -> NumPy / SciPy CSR reconstruction and .dit envelope helpers.

from typing import Any, Optional, Tuple
import pickle

try:
    import numpy as _np
except ImportError:
    _np = None

_DIT_MAGIC = b"DIT\x01"


def validate_dit_path(path: str) -> str:
    path_str = str(path)
    if not path_str.strip().lower().endswith(".dit"):
        raise ValueError(
            f"Invalid file extension for '{path_str}'. Model files must use the '.dit' extension ( >_< )"
        )
    return path_str


def bytes_to_f64_2d(rows: int, cols: int, raw: bytes, order: str = "C") -> Any:
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.float64).reshape((rows, cols)).copy()
    if order == "F":
        return np.asfortranarray(arr)
    return arr


def bytes_to_i64_1d(raw: bytes) -> Any:
    import numpy as np

    return np.frombuffer(raw, dtype=np.int64).copy()


def bytes_to_i64_2d(rows: int, cols: int, raw: bytes) -> Any:
    import numpy as np

    return np.frombuffer(raw, dtype=np.int64).reshape((rows, cols)).copy()


def csr_bytes_to_scipy(
    parts: Tuple[bytes, bytes, bytes, int, int],
    dtype: Optional[Any] = None,
) -> Any:
    import numpy as np
    from scipy import sparse

    data_b, idx_b, ptr_b, n_rows, n_cols = parts
    data = np.frombuffer(data_b, dtype=np.float64).copy()
    if dtype is not None and np.dtype(dtype) != np.float64:
        data = data.astype(dtype)
    indices = np.frombuffer(idx_b, dtype=np.int64).copy()
    indptr = np.frombuffer(ptr_b, dtype=np.int64).copy()
    return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def as_2d_array_or_rows(X: Any) -> Tuple[bool, Any]:
    """
    Inspect X and return `(is_numeric_f64_buf, payload)`:
    - `(True, contiguous_f64_ndarray)` when X is numeric 2D (fast PyBuffer<f64> path)
    - `(False, list_of_lists)` when X contains strings/objects/mixed types
    """
    import numpy as np

    if hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        arr = np.asarray(X)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {arr.ndim}D array instead ( >_< )"
        )

    if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
        return True, np.ascontiguousarray(arr, dtype=np.float64)

    return False, arr.tolist()


def as_1d_list(y: Any) -> Any:
    import numpy as np

    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    elif isinstance(y, np.ndarray):
        arr = y
    else:
        arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr.tolist()


def as_utf32_2d_view(X: Any) -> Optional[Tuple[Any, int, int, int]]:
    """Return `(flat_u32_buf, n_rows, n_cols, width)` for 2D Unicode NumPy arrays, else None."""
    import numpy as np

    if hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        arr = np.asarray(X)

    if arr.ndim == 2 and arr.dtype.kind == "U" and arr.dtype.itemsize >= 4:
        width = arr.dtype.itemsize // 4
        if arr.dtype.byteorder == ">":
            arr = arr.astype(f"<U{width}")
        u32 = np.ascontiguousarray(arr).view(np.uint32).reshape(-1)
        return u32, int(arr.shape[0]), int(arr.shape[1]), int(width)
    return None


def as_utf32_1d_view(y: Any) -> Optional[Tuple[Any, int, int]]:
    """Return `(flat_u32_buf, n_rows, width)` for 1D Unicode NumPy arrays, else None."""
    import numpy as np

    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    elif isinstance(y, np.ndarray):
        arr = y
    else:
        arr = np.asarray(y)

    if arr.ndim == 1 and arr.dtype.kind == "U" and arr.dtype.itemsize >= 4:
        width = arr.dtype.itemsize // 4
        if arr.dtype.byteorder == ">":
            arr = arr.astype(f"<U{width}")
        u32 = np.ascontiguousarray(arr).view(np.uint32).reshape(-1)
        return u32, int(arr.shape[0]), int(width)
    return None


def as_object_col_lists(X: Any) -> Optional[Any]:
    """Return `[col_list, ...]` for non-numeric 2D object/DataFrame inputs to avoid row-wise tolist."""
    import numpy as np

    module = type(X).__module__
    if "pandas" in module and hasattr(X, "columns"):
        if all(not (np.issubdtype(dt, np.number) or dt == np.bool_) for dt in X.dtypes if isinstance(dt, np.dtype)):
            return [X[c].tolist() for c in X.columns]
        arr = X.to_numpy()
    elif hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        arr = np.asarray(X)

    if arr.ndim == 2 and not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
        return [arr[:, j].tolist() for j in range(arr.shape[1])]
    return None


def save_dit_envelope(tag: int, state: Any, path: str) -> None:
    path_str = validate_dit_path(path)
    payload = _DIT_MAGIC + bytes([tag]) + pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_str, "wb") as f:
        f.write(payload)


def load_dit_envelope(expected_tag: int, data: bytes) -> Any:
    if len(data) < 5 or data[:4] != _DIT_MAGIC:
        raise ValueError("Invalid .dit file format (missing DIT header) (´-ω-`)")
    if data[4] != expected_tag:
        raise ValueError(
            f"Model type mismatch in .dit file (expected tag {expected_tag}, got {data[4]}) (・`ω´・)"
        )
    return pickle.loads(data[5:])

