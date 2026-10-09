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

__version__ = "0.2.1"
__all__ = [
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "get_available_devices",
    "get_device_info",
    "get_default_device",
    "set_default_device",
]

_DEFAULT_DEVICE: str = "auto"


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
