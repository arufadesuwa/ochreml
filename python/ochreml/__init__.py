"""
OchreML: Classical Machine Learning library in Rust with PyO3 bindings (＾▽＾)
Hardware accelerated for CPU, GPU, Multi-GPU, and TPU (*≧ω≦*)
Equipped with Zero-Copy Python Buffer Protocol and SIMD / Rayon solvers (o´∀｀o)
"""

from typing import Any, Dict, List, Optional
from ._utils import (
    process_features,
    process_target,
    to_feature_matrix,
    to_target_vector,
)
from ._ochreml import (
    LinearRegression as _RustLinearRegression,
    DecisionTreeClassifier as _RustDecisionTreeClassifier,
    DecisionTreeRegressor as _RustDecisionTreeRegressor,
    get_available_devices as _get_available_devices,
    get_device_info as _get_device_info,
)

__version__ = "0.2.3"
__all__ = [
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "save",
    "load",
    "get_available_devices",
    "get_device_info",
    "get_default_device",
    "set_default_device",
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


class LinearRegression:
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
    coef_ : list of float or None
        Estimated coefficients for the linear regression problem.
    intercept_ : float or None
        Independent term in the linear model.
    n_features_in_ : int or None
        Number of features seen during fitting.
    device_ : str or None
        Active device utilized during training (e.g. 'cpu', 'cuda:0', 'tpu:0').
    """

    def __init__(self, fit_intercept: bool = True, device: Optional[str] = None):
        self.fit_intercept = fit_intercept
        self.device = device or get_default_device()
        self._model = _RustLinearRegression(
            fit_intercept=fit_intercept,
            device=self.device,
        )
        self._coef_arr: Optional[Any] = None
        self._intercept_val: float = 0.0

    def _sync_cache(self) -> None:
        try:
            import numpy as np

            if self._model.coef_ is not None:
                self._coef_arr = np.asarray(self._model.coef_, dtype=np.float64)
            else:
                self._coef_arr = None
            self._intercept_val = float(self._model.intercept_ or 0.0)
        except Exception:
            self._coef_arr = None
            self._intercept_val = float(self._model.intercept_ or 0.0)

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

    def fit(self, X: Any, y: Any) -> "LinearRegression":
        """
        Fit linear model (*≧ω≦*)

        Parameters
        ----------
        X : array-like, nested list, pandas.DataFrame, polars.DataFrame
            Training feature matrix of shape (n_samples, n_features).
        y : array-like, list, pandas.Series, polars.Series
            Target values of shape (n_samples,).

        Returns
        -------
        self : LinearRegression
            Fitted estimator instance.
        """
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            self._model.fit_buffer(X_data, y_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            self._model.fit(X_mat, y_vec)

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
            Returns predicted values.
        """
        is_buf_x, X_data = process_features(X)
        if is_buf_x:
            # Sub-millisecond direct BLAS Level 2 GEMV inference (*≧ω≦*)
            try:
                import numpy as np

                if self._coef_arr is not None:
                    if X_data.shape[1] != self._coef_arr.shape[0]:
                        raise ValueError(
                            f"Number of features in test data ({X_data.shape[1]}) does not match model ({self._coef_arr.shape[0]}) (・`ω´・)"
                        )
                    return np.dot(X_data, self._coef_arr) + self._intercept_val
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

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (*^▽^*)
        """
        try:
            import numpy as np

            preds = self.predict(X)
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            u = np.sum((y_arr - preds) ** 2)
            v = np.sum((y_arr - np.mean(y_arr)) ** 2)
            if v <= 1e-15:
                return 0.0
            return float(1.0 - u / v)
        except Exception:
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
    def load(cls, path: str, device: Optional[str] = None) -> "LinearRegression":
        target_dev = device or get_default_device()
        rust_model = _RustLinearRegression.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> "LinearRegression":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = get_default_device()
        self._model = _RustLinearRegression.from_bytes(state, target_dev)
        self.fit_intercept = self._model.fit_intercept
        self.device = self._model.device_ or target_dev
        self._coef_arr = None
        self._intercept_val = 0.0
        self._sync_cache()

    @property
    def coef_(self) -> Optional[List[float]]:
        return self._model.coef_

    @property
    def intercept_(self) -> Optional[float]:
        return self._model.intercept_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return f"LinearRegression(fit_intercept={self.fit_intercept}, device='{self.device}') (＾▽＾)"


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
