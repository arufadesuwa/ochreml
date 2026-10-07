"""
OchreML: Classical Machine Learning library in Rust with PyO3 bindings (＾▽＾)
"""

from typing import Any, List, Optional
from ._utils import to_feature_matrix, to_target_vector
from ._ochreml import (
    LinearRegression as _RustLinearRegression,
    DecisionTreeClassifier as _RustDecisionTreeClassifier,
    DecisionTreeRegressor as _RustDecisionTreeRegressor,
)

__version__ = "0.1.0"
__all__ = [
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]


class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression model (＾▽＾)

    Solves linear regression using the Normal Equation: (X^T * X)^(-1) * X^T * y
    equipped with automated Tikhonov (Ridge) regularization when matrices are ill-conditioned.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    Attributes
    ----------
    coef_ : list of float or None
        Estimated coefficients for the linear regression problem.
    intercept_ : float or None
        Independent term in the linear model.
    n_features_in_ : int or None
        Number of features seen during fitting.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self._model = _RustLinearRegression(fit_intercept=fit_intercept)

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
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
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
        X_mat = to_feature_matrix(X)
        preds = self._model.predict(X_mat)
        try:
            import numpy as np
            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (*^▽^*)
        """
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
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

    def __repr__(self) -> str:
        return f"LinearRegression(fit_intercept={self.fit_intercept}) (＾▽＾)"


class DecisionTreeClassifier:
    """
    Decision Tree Classifier model (*≧ω≦*)

    Builds a classification tree using Gini Impurity or Entropy criteria.

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

    Attributes
    ----------
    classes_ : list of float
        The unique classes labels discovered during fitting.
    n_features_in_ : int or None
        Number of features seen during fitting.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._model = _RustDecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        """
        Build a decision tree classifier from the training set (X, y) (o´∀｀o)
        """
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
        self._model.fit(X_mat, y_vec)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict class value for X (＾▽＾)
        """
        X_mat = to_feature_matrix(X)
        preds = self._model.predict(X_mat)
        try:
            import numpy as np
            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (*^▽^*)
        """
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
        return self._model.score(X_mat, y_vec)

    @property
    def classes_(self) -> List[float]:
        return self._model.classes_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    def __repr__(self) -> str:
        return (
            f"DecisionTreeClassifier(criterion='{self.criterion}', "
            f"max_depth={self.max_depth}) (*≧ω≦*)"
        )


class DecisionTreeRegressor:
    """
    Decision Tree Regressor model (o´∀｀o)

    Builds a non-linear regression tree by minimizing Mean Squared Error (MSE).

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

    Attributes
    ----------
    n_features_in_ : int or None
        Number of features seen during fitting.
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._model = _RustDecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    def fit(self, X: Any, y: Any) -> "DecisionTreeRegressor":
        """
        Build a decision tree regressor from the training set (X, y) (*≧ω≦*)
        """
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
        self._model.fit(X_mat, y_vec)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict continuous target values for X (＾▽＾)
        """
        X_mat = to_feature_matrix(X)
        preds = self._model.predict(X_mat)
        try:
            import numpy as np
            return np.asarray(preds)
        except ImportError:
            return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (*^▽^*)
        """
        X_mat = to_feature_matrix(X)
        y_vec = to_target_vector(y)
        return self._model.score(X_mat, y_vec)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    def __repr__(self) -> str:
        return (
            f"DecisionTreeRegressor(criterion='{self.criterion}', "
            f"max_depth={self.max_depth}) (o´∀｀o)"
        )
