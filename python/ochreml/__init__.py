"""
OchreML: Classical Machine Learning library in Rust with PyO3 bindings (＾▽＾)
Hardware accelerated for CPU, GPU, Multi-GPU, and TPU (*≧ω≦*)
Equipped with Zero-Copy Python Buffer Protocol and SIMD / Rayon solvers (o´∀｀o)
"""

from typing import Any, Dict, List, Optional
import os as _os

# Auto-discover Python-packaged libtpu.so (e.g., Google Colab / Kaggle TPU v6e-1) before loading _ochreml
if "TPU_LIBRARY_PATH" not in _os.environ and "PJRT_PLUGIN_LIBRARY_PATH" not in _os.environ:
    try:
        import libtpu as _libtpu  # type: ignore

        _tpu_lib = _libtpu.get_library_path()
        if _tpu_lib and _os.path.exists(_tpu_lib):
            _os.environ["TPU_LIBRARY_PATH"] = _tpu_lib
    except Exception:
        pass

from ._utils import (
    process_features,
    process_sample_weight,
    process_target,
    process_target_multi,
    to_feature_matrix,
    to_target_vector,
)
from ._ochreml import (
    LinearRegression as _RustLinearRegression,
    Ridge as _RustRidge,
    RidgeClassifier as _RustRidgeClassifier,
    RidgeCV as _RustRidgeCV,
    DecisionTreeClassifier as _RustDecisionTreeClassifier,
    DecisionTreeRegressor as _RustDecisionTreeRegressor,
    get_available_devices as _get_available_devices,
    get_device_info as _get_device_info,
)

__version__ = "0.3.2b3"
__all__ = [
    "LinearRegression",
    "Ridge",
    "RidgeClassifier",
    "RidgeCV",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "save",
    "load",
    "get_available_devices",
    "get_device_info",
    "get_default_device",
    "set_default_device",
    "preprocessing",
    "impute",
    "feature_extraction",
    "compose",
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    "PowerTransformer",
    "QuantileTransformer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "Binarizer",
    "PolynomialFeatures",
    "SplineTransformer",
    "KernelCenterer",
    "SimpleImputer",
    "MissingIndicator",
    "KNNImputer",
    "IterativeImputer",
    "DictVectorizer",
    "FeatureHasher",
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
    "HashingVectorizer",
    "ColumnTransformer",
    "TransformedTargetRegressor",
    "make_column_transformer",
    "make_column_selector",
]

_DEFAULT_DEVICE: str = "auto"
_DIT_MAGIC = b"DIT\x01"


def _validate_dit_path(path: str) -> str:
    path_str = str(path)
    if not path_str.strip().lower().endswith(".dit"):
        raise ValueError(
            f"Invalid file extension for '{path_str}'. Model files must use the '.dit' extension ( >_< )"
        )
    return path_str


def get_default_device() -> str:
    """Return current global default hardware device (defaults to 'auto') (*^▽^*)."""
    return _DEFAULT_DEVICE


def set_default_device(device: str) -> None:
    """
    Set the global default hardware device for all OchreML estimators (*≧ω≦*).

    Parameters
    ----------
    device : str
        Target device identifier: 'auto', 'cpu', 'cuda:0', 'cuda:all', or 'tpu'.
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def get_available_devices() -> List[str]:
    """
    Return list of discovered hardware accelerators on host system (o´∀｀o).

    Returns
    -------
    devices : list of str
        List containing detected devices, e.g. ['cpu', 'cuda:0', 'tpu:0'].
    """
    return _get_available_devices()


def get_device_info() -> Dict[str, str]:
    """
    Return diagnostic metadata for host CPU, GPU, and TPU hardware (＾▽＾).

    Returns
    -------
    info : dict of str to str
        Hardware capabilities dictionary.
    """
    return _get_device_info()


def save(model: Any, path: str) -> None:
    """
    Save any OchreML model into a compact binary '.dit' file (*≧ω≦*).
    """
    path_str = _validate_dit_path(path)
    if hasattr(model, "save"):
        model.save(path_str)
    else:
        raise TypeError(f"Object of type {type(model).__name__} is not a valid OchreML model ( >_< )")


def load(path: str, device: Optional[str] = None) -> Any:
    """
    Load an OchreML model from a '.dit' file and automatically configure its target device (＾▽＾).
    Supports both native '.dit' binary files and joblib/pickle '.dit' archives.
    """
    path_str = _validate_dit_path(path)
    with open(path_str, "rb") as f:
        data = f.read()

    if data.startswith(_DIT_MAGIC) and len(data) >= 5:
        tag = data[4]
        target_dev = device or get_default_device()
        if tag == 1:
            return LinearRegression._from_rust(_RustLinearRegression.from_bytes(data, target_dev), target_dev)
        if tag == 2:
            return DecisionTreeClassifier._from_rust(_RustDecisionTreeClassifier.from_bytes(data, target_dev), target_dev)
        if tag == 3:
            return DecisionTreeRegressor._from_rust(_RustDecisionTreeRegressor.from_bytes(data, target_dev), target_dev)
        if tag == 4:
            return Ridge._from_rust(_RustRidge.from_bytes(data, target_dev), target_dev)
        if tag == 5:
            return RidgeClassifier._from_rust(_RustRidgeClassifier.from_bytes(data, target_dev), target_dev)
        if tag == 6:
            return RidgeCV._from_rust(_RustRidgeCV.from_bytes(data, target_dev), target_dev)
        if 7 <= tag <= 37:
            return _load_preprocessing_dit_tag(tag, data, path_str, target_dev)
        raise ValueError(f"Unknown model tag ({tag}) in .dit file (・`ω´・)")

    # Fallback if the .dit file was written via joblib.dump or pickle.dump
    try:
        import joblib

        obj = joblib.load(path_str)
    except Exception:
        import pickle

        obj = pickle.loads(data)

    if device is not None and hasattr(obj, "to"):
        obj.to(device)
    return obj


def _fast_linear_predict_1d(X_data: Any, w: Any, b: float) -> Any:
    """L1/L2-tiled GEMV + in-place bias addition without OpenBLAS/Rayon thread-pool collision (＾▽＾)."""
    import numpy as np

    n = X_data.shape[0]
    step = 4096
    if n <= step:
        out = np.dot(X_data, w)
        if b != 0.0:
            out += b
        return out
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, step):
        sl = out[i : i + step]
        np.dot(X_data[i : i + step], w, out=sl)
        if b != 0.0:
            sl += b
    return out


def _fast_linear_predict_2d(X_data: Any, w_t: Any, b: Any) -> Any:
    """L1/L2-tiled GEMM + in-place bias addition without OpenBLAS/Rayon thread-pool collision (＾▽＾)."""
    import numpy as np

    n = X_data.shape[0]
    step = 2048
    if n <= step:
        out = np.dot(X_data, w_t)
        out += b
        return out
    out = np.empty((n, w_t.shape[1]), dtype=np.float64)
    for i in range(0, n, step):
        sl = out[i : i + step]
        np.dot(X_data[i : i + step], w_t, out=sl)
        sl += b
    return out


class _BaseLinearEstimator:
    """Shared Python wrapper infrastructure for linear regression estimators (＾▽＾)."""

    _rust_cls: Any = None

    def _sync_cache(self) -> None:
        try:
            import numpy as np

            if self._model.is_multi_target_:
                if self._model.coef_multi_ is not None:
                    self._coef_arr = np.asarray(self._model.coef_multi_, dtype=np.float64)
                    self._coef_arr_t = np.ascontiguousarray(self._coef_arr.T)
                else:
                    self._coef_arr = None
                    self._coef_arr_t = None
                if self._model.intercept_multi_ is not None:
                    self._intercept_val = np.asarray(self._model.intercept_multi_, dtype=np.float64)
                else:
                    self._intercept_val = np.zeros(self._model.n_targets_, dtype=np.float64)
            else:
                if self._model.coef_ is not None:
                    self._coef_arr = np.asarray(self._model.coef_, dtype=np.float64)
                else:
                    self._coef_arr = None
                self._coef_arr_t = None
                self._intercept_val = float(self._model.intercept_ or 0.0)
        except Exception:
            self._coef_arr = None
            self._coef_arr_t = None
            if self._model.is_multi_target_:
                self._intercept_val = self._model.intercept_multi_
            else:
                self._intercept_val = float(self._model.intercept_ or 0.0)

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> Any:
        """
        Fit linear model with optional sample_weight and multi-output target support (*≧ω≦*)

        Parameters
        ----------
        X : array-like, nested list, pandas.DataFrame, polars.DataFrame
            Training feature matrix of shape (n_samples, n_features).
        y : array-like, list, pandas.Series, polars.Series, or 2D matrix
            Target values of shape (n_samples,) or (n_samples, n_targets).
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted estimator instance.
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data, is_2d = process_target_multi(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            sw_list = sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            if is_2d:
                y_mat = y_data.tolist() if is_buf_y else y_data
                self._model.fit_multi(X_mat, y_mat, sw_list)
            else:
                y_vec = y_data.tolist() if is_buf_y else y_data
                self._model.fit(X_mat, y_vec, sw_list)

        self._sync_cache()
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict using the linear model (＾▽＾)

        Parameters
        ----------
        X : array-like, nested list, pandas.DataFrame, polars.DataFrame
            Samples to predict.

        Returns
        -------
        predictions : numpy.ndarray or list of float
            Returns predicted values of shape (n_samples,) or (n_samples, n_targets).
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            # Sub-millisecond direct BLAS Level 2/3 GEMV/GEMM inference (*≧ω≦*)
            try:
                import numpy as np

                if self._coef_arr is not None:
                    if self._model.is_multi_target_:
                        if X_data.shape[1] != self._coef_arr.shape[1]:
                            raise ValueError(
                                f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[1]}) (・`ω´・)"
                            )
                        w_t = getattr(self, "_coef_arr_t", None)
                        if w_t is None:
                            w_t = np.ascontiguousarray(self._coef_arr.T)
                        return _fast_linear_predict_2d(X_data, w_t, self._intercept_val)
                    else:
                        if X_data.shape[1] != self._coef_arr.shape[0]:
                            raise ValueError(
                                f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[0]}) (・`ω´・)"
                            )
                        return _fast_linear_predict_1d(X_data, self._coef_arr, self._intercept_val)
            except ValueError:
                raise
            except Exception:
                pass
            if self._model.is_multi_target_:
                X_mat = X_data.tolist()
                preds = self._model.predict_multi(X_mat)
            else:
                preds = self._model.predict_buffer(X_data)
        else:
            if self._model.is_multi_target_:
                preds = self._model.predict_multi(X_data)
            else:
                preds = self._model.predict(X_data)
        try:
            import numpy as np

            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (*^▽^*)
        Supports single-output, multi-output (uniform_average), and sample_weight.
        """
        try:
            import numpy as np

            preds = self.predict(X)
            _, y_data, _ = process_target_multi(y)
            y_arr = np.asarray(y_data, dtype=np.float64)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if y_arr.shape != preds.shape:
                raise ValueError(
                    f"Target shape {y_arr.shape} does not match prediction shape {preds.shape} ( >_< )"
                )

            if sample_weight is not None:
                _, sw_data = process_sample_weight(sample_weight)
                sw = np.asarray(sw_data, dtype=np.float64).ravel()
                if sw.shape[0] != y_arr.shape[0]:
                    raise ValueError(
                        f"Sample weight length ({sw.shape[0]}) must match test sample count ({y_arr.shape[0]}) ( >_< )"
                    )
                if np.any(sw < 0.0) or np.any(np.isnan(sw)):
                    raise ValueError("Sample weights must be non-negative valid numbers (・`ω´・)")
                sw_sum = np.sum(sw)
                if sw_sum <= 0.0:
                    raise ValueError("Sum of sample weights must be strictly positive (´；ω；`)")
                sw_col = sw[:, None]
                y_mean = np.sum(sw_col * y_arr, axis=0) / sw_sum
                u = np.sum(sw_col * ((y_arr - preds) ** 2), axis=0)
                v = np.sum(sw_col * ((y_arr - y_mean) ** 2), axis=0)
            else:
                y_mean = np.mean(y_arr, axis=0)
                u = np.sum((y_arr - preds) ** 2, axis=0)
                v = np.sum((y_arr - y_mean) ** 2, axis=0)

            r2_cols = np.where(v <= 1e-12, np.where(u <= 1e-12, 1.0, 0.0), 1.0 - u / v)
            return float(np.mean(r2_cols))
        except ValueError:
            raise
        except Exception:
            is_buf_x, X_data = process_features(X)
            is_buf_y, y_data, is_2d = process_target_multi(y)
            is_buf_sw, sw_data = process_sample_weight(sample_weight)
            if is_buf_x and is_buf_y and is_buf_sw:
                return self._model.score_buffer(X_data, y_data, sw_data)
            else:
                X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
                sw_list = sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
                if is_2d:
                    y_mat = y_data.tolist() if is_buf_y else y_data
                    return self._model.score_multi(X_mat, y_mat, sw_list)
                else:
                    y_vec = y_data.tolist() if is_buf_y else y_data
                    return self._model.score(X_mat, y_vec, sw_list)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        target_dev = device or get_default_device()
        rust_model = cls._rust_cls.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> Any:
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    @property
    def coef_(self) -> Optional[Any]:
        if self._coef_arr is not None:
            return self._coef_arr
        if self._model.is_multi_target_:
            return self._model.coef_multi_
        return self._model.coef_

    @property
    def intercept_(self) -> Optional[Any]:
        if self._model.is_multi_target_:
            if self._coef_arr is not None:
                return self._intercept_val
            return self._model.intercept_multi_
        return self._model.intercept_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class LinearRegression(_BaseLinearEstimator):
    """
    Ordinary Least Squares (OLS) Linear Regression model (＾▽＾)

    Solves linear regression using the Normal Equation: (X^T * X)^(-1) * X^T * y
    equipped with automated Cholesky decomposition, Tikhonov regularization,
    and Zero-Copy Buffer Protocol acceleration (*≧ω≦*).

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'tpu').
        If None, the global default device (set via set_default_device, default 'auto') is used.

    Attributes
    ----------
    coef_ : numpy.ndarray or list of float or None
        Estimated coefficients for the linear regression problem.
    intercept_ : float, numpy.ndarray, or None
        Independent term in the linear model.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training (e.g. 'cpu', 'cuda:0', 'tpu:0').
    """

    _rust_cls = _RustLinearRegression

    def __init__(self, fit_intercept: bool = True, device: Optional[str] = None):
        self.fit_intercept = fit_intercept
        self.device = device or get_default_device()
        self._model = _RustLinearRegression(
            fit_intercept=fit_intercept,
            device=self.device,
        )
        self._coef_arr: Optional[Any] = None
        self._intercept_val: Any = 0.0

    @classmethod
    def _from_rust(cls, rust_model: _RustLinearRegression, device: str) -> "LinearRegression":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.fit_intercept = rust_model.fit_intercept
        inst.device = rust_model.device_ or device
        inst._coef_arr = None
        inst._intercept_val = 0.0
        inst._sync_cache()
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustLinearRegression.from_bytes(state, target_dev)
        self.fit_intercept = self._model.fit_intercept
        self.device = self._model.device_ or target_dev
        self._coef_arr = None
        self._intercept_val = 0.0
        self._classes_arr = None
        self._sync_cache()

    def __repr__(self) -> str:
        return f"LinearRegression(fit_intercept={self.fit_intercept}, device='{self.device}') (＾▽＾)"


class Ridge(_BaseLinearEstimator):
    """
    Linear least squares with L2 (Tikhonov) regularization (*≧ω≦*)

    Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
    using normal equations computed on CPU, CUDA, wgpu, or TPU backends without
    penalizing the intercept term.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L2 term, controlling regularization strength.
        Must be a non-negative finite float.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'wgpu', 'tpu').

    Attributes
    ----------
    coef_ : numpy.ndarray or list of float or None
        Weight vector(s) of shape (n_features,) or (n_targets, n_features).
    intercept_ : float, numpy.ndarray, or None
        Independent term in decision function. Set to 0.0 if fit_intercept=False.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training.
    """

    _rust_cls = _RustRidge

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.device = device or get_default_device()
        self._model = _RustRidge(
            alpha=self.alpha,
            fit_intercept=fit_intercept,
            device=self.device,
        )
        self._coef_arr: Optional[Any] = None
        self._intercept_val: Any = 0.0

    @classmethod
    def _from_rust(cls, rust_model: _RustRidge, device: str) -> "Ridge":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.alpha = rust_model.alpha
        inst.fit_intercept = rust_model.fit_intercept
        inst.device = rust_model.device_ or device
        inst._coef_arr = None
        inst._intercept_val = 0.0
        inst._sync_cache()
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustRidge.from_bytes(state, target_dev)
        self.alpha = self._model.alpha
        self.fit_intercept = self._model.fit_intercept
        self.device = self._model.device_ or target_dev
        self._coef_arr = None
        self._intercept_val = 0.0
        self._classes_arr = None
        self._sync_cache()

    def __repr__(self) -> str:
        return (
            f"Ridge(alpha={self.alpha}, fit_intercept={self.fit_intercept}, "
            f"device='{self.device}') (*≧ω≦*)"
        )


class RidgeClassifier:
    """
    Classifier using Ridge regression (o´∀｀o)

    Converts binary targets into {-1.0, +1.0} or multiclass targets into a
    one-vs-rest {-1.0, +1.0} matrix and solves L2-regularized normal equations
    across CPU, CUDA, wgpu, and TPU backends.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a non-negative finite float.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'wgpu', 'tpu').

    Attributes
    ----------
    coef_ : numpy.ndarray or list of float or None
        Coefficient of the features in the decision function.
    intercept_ : float, numpy.ndarray, or None
        Independent term in decision function.
    classes_ : numpy.ndarray or list of float
        Unique class labels discovered during fitting.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.device = device or get_default_device()
        self._model = _RustRidgeClassifier(
            alpha=self.alpha,
            fit_intercept=fit_intercept,
            device=self.device,
        )
        self._coef_arr: Optional[Any] = None
        self._coef_arr_t: Optional[Any] = None
        self._intercept_val: Any = 0.0
        self._classes_arr: Optional[Any] = None

    def _sync_cache(self) -> None:
        try:
            import numpy as np

            self._classes_arr = (
                np.asarray(self._model.classes_, dtype=np.float64)
                if self._model.classes_
                else None
            )
            if self._model.is_multiclass_:
                if self._model.coef_multi_ is not None:
                    self._coef_arr = np.asarray(self._model.coef_multi_, dtype=np.float64)
                    self._coef_arr_t = np.ascontiguousarray(self._coef_arr.T)
                else:
                    self._coef_arr = None
                    self._coef_arr_t = None
                if self._model.intercept_multi_ is not None:
                    self._intercept_val = np.asarray(self._model.intercept_multi_, dtype=np.float64)
                else:
                    self._intercept_val = np.zeros(len(self._model.classes_), dtype=np.float64)
            else:
                if self._model.coef_ is not None:
                    self._coef_arr = np.asarray(self._model.coef_, dtype=np.float64)
                else:
                    self._coef_arr = None
                self._coef_arr_t = None
                self._intercept_val = float(self._model.intercept_ or 0.0)
        except Exception:
            self._coef_arr = None
            self._coef_arr_t = None
            self._classes_arr = None
            if self._model.is_multiclass_:
                self._intercept_val = self._model.intercept_multi_
            else:
                self._intercept_val = float(self._model.intercept_ or 0.0)

    @classmethod
    def _from_rust(cls, rust_model: _RustRidgeClassifier, device: str) -> "RidgeClassifier":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.alpha = rust_model.alpha
        inst.fit_intercept = rust_model.fit_intercept
        inst.device = rust_model.device_ or device
        inst._coef_arr = None
        inst._coef_arr_t = None
        inst._intercept_val = 0.0
        inst._classes_arr = None
        inst._sync_cache()
        return inst

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "RidgeClassifier":
        """
        Fit Ridge classifier model (＾▽＾)

        Parameters
        ----------
        X : array-like, nested list, pandas.DataFrame, polars.DataFrame
            Training data of shape (n_samples, n_features).
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : RidgeClassifier
            Fitted estimator instance.
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            sw_list = sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            self._model.fit(X_mat, y_vec, sw_list)

        self._sync_cache()
        return self

    def decision_function(self, X: Any) -> Any:
        """
        Predict confidence scores for samples (o´∀｀o)

        Parameters
        ----------
        X : array-like, nested list, pandas.DataFrame, polars.DataFrame
            The data matrix for which we want to compute confidence scores.

        Returns
        -------
        scores : numpy.ndarray or list
            Confidence scores of shape (n_samples,) for binary classification or
            (n_samples, n_classes) for multiclass classification.
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            try:
                import numpy as np

                if self._coef_arr is not None:
                    if self._model.is_multiclass_:
                        if X_data.shape[1] != self._coef_arr.shape[1]:
                            raise ValueError(
                                f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[1]}) (・`ω´・)"
                            )
                        w_t = getattr(self, "_coef_arr_t", None)
                        if w_t is None:
                            w_t = np.ascontiguousarray(self._coef_arr.T)
                        return _fast_linear_predict_2d(X_data, w_t, self._intercept_val)
                    else:
                        if X_data.shape[1] != self._coef_arr.shape[0]:
                            raise ValueError(
                                f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[0]}) (・`ω´・)"
                            )
                        return _fast_linear_predict_1d(X_data, self._coef_arr, self._intercept_val)
            except ValueError:
                raise
            except Exception:
                pass
            if self._model.is_multiclass_:
                scores = self._model.decision_function_multi_buffer(X_data)
            else:
                scores = self._model.decision_function_buffer(X_data)
        else:
            if self._model.is_multiclass_:
                scores = self._model.decision_function_multi(X_data)
            else:
                scores = self._model.decision_function(X_data)
        try:
            import numpy as np

            return np.asarray(scores)
        except ImportError:
            return scores

    def predict(self, X: Any) -> Any:
        """
        Predict class labels for samples in X (＾▽＾)
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            try:
                import numpy as np

                if self._classes_arr is not None:
                    if len(self._classes_arr) == 1:
                        if X_data.shape[1] != self._model.n_features_in_:
                            raise ValueError(
                                f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._model.n_features_in_}) (・`ω´・)"
                            )
                        return np.full(X_data.shape[0], self._classes_arr[0], dtype=np.float64)
                    if self._coef_arr is not None:
                        if self._model.is_multiclass_:
                            if X_data.shape[1] != self._coef_arr.shape[1]:
                                raise ValueError(
                                    f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[1]}) (・`ω´・)"
                                )
                            w_t = getattr(self, "_coef_arr_t", None)
                            if w_t is None:
                                w_t = np.ascontiguousarray(self._coef_arr.T)
                            n = X_data.shape[0]
                            step = 2048
                            if n <= step:
                                scores = np.dot(X_data, w_t)
                                scores += self._intercept_val
                                return self._classes_arr[np.argmax(scores, axis=1)]
                            out = np.empty(n, dtype=self._classes_arr.dtype)
                            buf = np.empty((step, w_t.shape[1]), dtype=np.float64)
                            for i in range(0, n, step):
                                end = min(i + step, n)
                                sl = buf[: end - i]
                                np.dot(X_data[i:end], w_t, out=sl)
                                sl += self._intercept_val
                                out[i:end] = self._classes_arr[np.argmax(sl, axis=1)]
                            return out
                        else:
                            if X_data.shape[1] != self._coef_arr.shape[0]:
                                raise ValueError(
                                    f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[0]}) (・`ω´・)"
                                )
                            scores = _fast_linear_predict_1d(
                                X_data, self._coef_arr, self._intercept_val
                            )
                            return np.where(scores > 0.0, self._classes_arr[1], self._classes_arr[0])
            except ValueError:
                raise
            except Exception:
                pass
            preds = self._model.predict_buffer(X_data)
        else:
            preds = self._model.predict(X_data)
        try:
            import numpy as np

            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """
        Return the mean accuracy on the given test data and labels (*^▽^*)
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)
        if is_buf_x and is_buf_y and is_buf_sw:
            return self._model.score_buffer(X_data, y_data, sw_data)
        X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
        y_vec = to_target_vector(y_data) if is_buf_y else y_data
        sw_list = sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
        return self._model.score(X_mat, y_vec, sw_list)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RidgeClassifier":
        target_dev = device or get_default_device()
        rust_model = _RustRidgeClassifier.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> "RidgeClassifier":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustRidgeClassifier.from_bytes(state, target_dev)
        self.alpha = self._model.alpha
        self.fit_intercept = self._model.fit_intercept
        self.device = self._model.device_ or target_dev
        self._coef_arr = None
        self._intercept_val = 0.0
        self._classes_arr = None
        self._sync_cache()

    @property
    def classes_(self) -> Any:
        try:
            import numpy as np

            return np.asarray(self._model.classes_, dtype=np.float64)
        except ImportError:
            return self._model.classes_

    @property
    def coef_(self) -> Optional[Any]:
        if self._coef_arr is not None:
            return self._coef_arr
        if self._model.is_multiclass_:
            return self._model.coef_multi_
        return self._model.coef_

    @property
    def intercept_(self) -> Optional[Any]:
        if self._model.is_multiclass_:
            if self._coef_arr is not None:
                return self._intercept_val
            return self._model.intercept_multi_
        return self._model.intercept_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"RidgeClassifier(alpha={self.alpha}, fit_intercept={self.fit_intercept}, "
            f"device='{self.device}') (o´∀｀o)"
        )


class RidgeCV(_BaseLinearEstimator):
    """
    Ridge regression with built-in cross-validation (*^▽^*)

    Computes (X^T W X, X^T W Y) once on the configured hardware backend (CPU,
    CUDA, wgpu, or TPU), then evaluates each candidate alpha via closed-form
    Leave-One-Out CV (when cv=None) or K-Fold CV (when cv is an integer >= 2).

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try. Must be strictly positive when cv=None,
        or non-negative when cv is an integer.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    cv : int or None, default=None
        Cross-validation splitting strategy. None performs efficient closed-form
        Leave-One-Out CV; an integer k >= 2 performs k-fold CV.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'wgpu', 'tpu').

    Attributes
    ----------
    alpha_ : float or None
        Estimated regularization parameter maximizing cross-validation score.
    best_score_ : float or None
        Highest cross-validation score achieved (negative LOO MSE when cv=None,
        mean R^2 when cv is an integer).
    coef_ : numpy.ndarray or list of float or None
        Weight vector(s) for the selected alpha.
    intercept_ : float, numpy.ndarray, or None
        Independent term in decision function.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training.
    """

    _rust_cls = _RustRidgeCV

    def __init__(
        self,
        alphas: Any = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        cv: Optional[int] = None,
        device: Optional[str] = None,
    ):
        try:
            alpha_list = [float(a) for a in alphas]
        except TypeError:
            alpha_list = [float(alphas)]
        self.alphas = tuple(alpha_list)
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.device = device or get_default_device()
        self._model = _RustRidgeCV(
            alphas=alpha_list,
            fit_intercept=fit_intercept,
            cv=cv,
            device=self.device,
        )
        self._coef_arr: Optional[Any] = None
        self._intercept_val: Any = 0.0

    @classmethod
    def _from_rust(cls, rust_model: _RustRidgeCV, device: str) -> "RidgeCV":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.alphas = tuple(rust_model.alphas)
        inst.fit_intercept = rust_model.fit_intercept
        inst.cv = rust_model.cv
        inst.device = rust_model.device_ or device
        inst._coef_arr = None
        inst._intercept_val = 0.0
        inst._sync_cache()
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustRidgeCV.from_bytes(state, target_dev)
        self.alphas = tuple(self._model.alphas)
        self.fit_intercept = self._model.fit_intercept
        self.cv = self._model.cv
        self.device = self._model.device_ or target_dev
        self._coef_arr = None
        self._intercept_val = 0.0
        self._classes_arr = None
        self._sync_cache()

    @property
    def alpha_(self) -> Optional[float]:
        return self._model.alpha_

    @property
    def best_score_(self) -> Optional[float]:
        return self._model.best_score_

    def __repr__(self) -> str:
        return (
            f"RidgeCV(alphas={self.alphas}, fit_intercept={self.fit_intercept}, "
            f"cv={self.cv}, device='{self.device}') (*^▽^*)"
        )


class DecisionTreeClassifier:
    """
    Decision Tree Classifier model (*≧ω≦*)

    Builds a classification tree using Gini Impurity or Entropy criteria,
    equipped with Zero-Copy Buffer Protocol and Rayon parallelization.

    Parameters
    ----------
    criterion : str, default="gini"
        The function to measure the quality of a split ("gini" or "entropy").
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'tpu').

    Attributes
    ----------
    classes_ : list of float
        The unique classes labels discovered during fitting.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training (e.g. 'cpu', 'cuda:0', 'tpu:0').
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        device: Optional[str] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.device = device or get_default_device()
        self._model = _RustDecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustDecisionTreeClassifier, device: str) -> "DecisionTreeClassifier":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.criterion = rust_model.criterion
        inst.max_depth = rust_model.max_depth
        inst.min_samples_split = rust_model.min_samples_split
        inst.min_samples_leaf = rust_model.min_samples_leaf
        inst.device = rust_model.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        """
        Build a decision tree classifier from the training set (X, y) (o´∀｀o)
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            self._model.fit_buffer(X_data, y_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            self._model.fit(X_mat, y_vec)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict class value for X (＾▽＾)
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            preds = self._model.predict_buffer(X_data)
        else:
            preds = self._model.predict(X_data)
        try:
            import numpy as np
            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (*^▽^*)
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            return self._model.score_buffer(X_data, y_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            return self._model.score(X_mat, y_vec)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DecisionTreeClassifier":
        target_dev = device or get_default_device()
        rust_model = _RustDecisionTreeClassifier.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> "DecisionTreeClassifier":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustDecisionTreeClassifier.from_bytes(state, target_dev)
        self.criterion = self._model.criterion
        self.max_depth = self._model.max_depth
        self.min_samples_split = self._model.min_samples_split
        self.min_samples_leaf = self._model.min_samples_leaf
        self.device = self._model.device_ or target_dev

    @property
    def classes_(self) -> List[float]:
        return self._model.classes_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"DecisionTreeClassifier(criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, device='{self.device}') (*≧ω≦*)"
        )


class DecisionTreeRegressor:
    """
    Decision Tree Regressor model (o´∀｀o)

    Builds a non-linear regression tree by minimizing Mean Squared Error (MSE),
    equipped with Zero-Copy Buffer Protocol and Rayon parallelization.

    Parameters
    ----------
    criterion : str, default="squared_error"
        The function to measure split quality ("squared_error" or "mse").
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'cuda:all', 'tpu').

    Attributes
    ----------
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training (e.g. 'cpu', 'cuda:0', 'tpu:0').
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        device: Optional[str] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.device = device or get_default_device()
        self._model = _RustDecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustDecisionTreeRegressor, device: str) -> "DecisionTreeRegressor":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.criterion = rust_model.criterion
        inst.max_depth = rust_model.max_depth
        inst.min_samples_split = rust_model.min_samples_split
        inst.min_samples_leaf = rust_model.min_samples_leaf
        inst.device = rust_model.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "DecisionTreeRegressor":
        """
        Build a decision tree regressor from the training set (X, y) (*≧ω≦*)
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            self._model.fit_buffer(X_data, y_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            self._model.fit(X_mat, y_vec)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict continuous target values for X (＾▽＾)
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            preds = self._model.predict_buffer(X_data)
        else:
            preds = self._model.predict(X_data)
        try:
            import numpy as np
            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (*^▽^*)
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            return self._model.score_buffer(X_data, y_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            return self._model.score(X_mat, y_vec)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DecisionTreeRegressor":
        target_dev = device or get_default_device()
        rust_model = _RustDecisionTreeRegressor.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> "DecisionTreeRegressor":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustDecisionTreeRegressor.from_bytes(state, target_dev)
        self.criterion = self._model.criterion
        self.max_depth = self._model.max_depth
        self.min_samples_split = self._model.min_samples_split
        self.min_samples_leaf = self._model.min_samples_leaf
        self.device = self._model.device_ or target_dev

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"DecisionTreeRegressor(criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, device='{self.device}') (o´∀｀o)"
        )


def _load_preprocessing_dit_tag(tag: int, data: bytes, path_str: str, target_dev: str) -> Any:
    """Dispatch additive DIT\\x01 tags 7..=37 for preprocessing, impute, feature_extraction, and compose (＾▽＾)."""
    from ._ochreml import (
        Binarizer as _RustBinarizer,
        CountVectorizer as _RustCountVectorizer,
        DictVectorizer as _RustDictVectorizer,
        FeatureHasher as _RustFeatureHasher,
        HashingVectorizer as _RustHashingVectorizer,
        KBinsDiscretizer as _RustKBinsDiscretizer,
        KNNImputer as _RustKNNImputer,
        KernelCenterer as _RustKernelCenterer,
        LabelBinarizer as _RustLabelBinarizer,
        LabelEncoder as _RustLabelEncoder,
        MaxAbsScaler as _RustMaxAbsScaler,
        MinMaxScaler as _RustMinMaxScaler,
        MissingIndicator as _RustMissingIndicator,
        MultiLabelBinarizer as _RustMultiLabelBinarizer,
        Normalizer as _RustNormalizer,
        OneHotEncoder as _RustOneHotEncoder,
        OrdinalEncoder as _RustOrdinalEncoder,
        PolynomialFeatures as _RustPolynomialFeatures,
        PowerTransformer as _RustPowerTransformer,
        QuantileTransformer as _RustQuantileTransformer,
        RobustScaler as _RustRobustScaler,
        SimpleImputer as _RustSimpleImputer,
        SplineTransformer as _RustSplineTransformer,
        StandardScaler as _RustStandardScaler,
        TargetEncoder as _RustTargetEncoder,
        TfidfTransformer as _RustTfidfTransformer,
    )

    rust_dispatch = {
        7: (StandardScaler, _RustStandardScaler),
        8: (MinMaxScaler, _RustMinMaxScaler),
        9: (MaxAbsScaler, _RustMaxAbsScaler),
        10: (RobustScaler, _RustRobustScaler),
        11: (Normalizer, _RustNormalizer),
        12: (Binarizer, _RustBinarizer),
        13: (KernelCenterer, _RustKernelCenterer),
        14: (PolynomialFeatures, _RustPolynomialFeatures),
        15: (KBinsDiscretizer, _RustKBinsDiscretizer),
        16: (PowerTransformer, _RustPowerTransformer),
        17: (QuantileTransformer, _RustQuantileTransformer),
        18: (SplineTransformer, _RustSplineTransformer),
        19: (OneHotEncoder, _RustOneHotEncoder),
        20: (OrdinalEncoder, _RustOrdinalEncoder),
        21: (LabelEncoder, _RustLabelEncoder),
        22: (LabelBinarizer, _RustLabelBinarizer),
        23: (MultiLabelBinarizer, _RustMultiLabelBinarizer),
        24: (TargetEncoder, _RustTargetEncoder),
        25: (SimpleImputer, _RustSimpleImputer),
        26: (MissingIndicator, _RustMissingIndicator),
        27: (KNNImputer, _RustKNNImputer),
        28: (DictVectorizer, _RustDictVectorizer),
        29: (FeatureHasher, _RustFeatureHasher),
        30: (CountVectorizer, _RustCountVectorizer),
        31: (TfidfTransformer, _RustTfidfTransformer),
        33: (HashingVectorizer, _RustHashingVectorizer),
    }
    if tag in rust_dispatch:
        py_cls, rust_cls = rust_dispatch[tag]
        return py_cls._from_rust(rust_cls.from_bytes(data, target_dev), target_dev)
    if tag == 32:
        return TfidfVectorizer.load(path_str, target_dev)
    if tag == 34:
        return IterativeImputer.load(path_str, target_dev)
    if tag == 35:
        return FunctionTransformer.load(path_str, target_dev)
    if tag == 36:
        return ColumnTransformer.load(path_str, target_dev)
    if tag == 37:
        return TransformedTargetRegressor.load(path_str, target_dev)
    raise ValueError(f"Unknown model tag ({tag}) in .dit file (・`ω´・)")


from . import compose, feature_extraction, impute, preprocessing
from .compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
    make_column_transformer,
)
from .feature_extraction import DictVectorizer, FeatureHasher
from .feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from .impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from .preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)
