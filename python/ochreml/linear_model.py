"""
OchreML Linear & Generalized Linear Models Submodule (＾▽＾)
Provides Scikit-Learn-compatible LogisticRegression, Lasso, ElasticNet,
SGDClassifier, SGDRegressor, and re-exports LinearRegression, Ridge, RidgeClassifier, RidgeCV (*≧ω≦*).
"""

from typing import Any, Dict, List, Optional, Tuple

from ._ochreml import (
    ElasticNet as _RustElasticNet,
    Lasso as _RustLasso,
    LogisticRegression as _RustLogisticRegression,
    SGDClassifier as _RustSGDClassifier,
    SGDRegressor as _RustSGDRegressor,
)
from ._utils import (
    process_features,
    process_sample_weight,
    process_target,
    process_target_multi,
    to_feature_matrix,
    to_target_vector,
)


def _get_default_device() -> str:
    from . import get_default_device

    return get_default_device()


def _validate_dit_path(path: str) -> str:
    path_str = str(path)
    if not path_str.strip().lower().endswith(".dit"):
        raise ValueError(
            f"Invalid file extension for '{path_str}'. Model files must use the '.dit' extension ( >_< )"
        )
    return path_str


def _parse_class_weight(
    class_weight: Optional[Any],
) -> Tuple[Optional[str], Optional[List[Tuple[float, float]]]]:
    if class_weight is None:
        return None, None
    if isinstance(class_weight, str):
        return class_weight, None
    if isinstance(class_weight, dict):
        return None, [(float(k), float(v)) for k, v in class_weight.items()]
    raise TypeError(
        f"class_weight must be None, 'balanced', or a dict, got {type(class_weight)} (・`ω´・)"
    )


def _to_ndarray_or_raw(val: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(val, dtype=np.float64)
    except ImportError:
        return val


class LogisticRegression:
    """
    L2-regularized Logistic Regression (binary & multinomial) using L-BFGS (＾▽＾)

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a strictly positive float.
    fit_intercept : bool, default=True
        Specifies if a constant (bias/intercept) should be added to the decision function.
    max_iter : int, default=100
        Maximum number of L-BFGS iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        class_weight: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.device = device or _get_default_device()
        cw_mode, cw_map = _parse_class_weight(class_weight)
        self._model = _RustLogisticRegression(
            c=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight_mode=cw_mode,
            class_weight_map=cw_map,
            device=self.device,
        )

    @classmethod
    def _from_rust(
        cls, rust_model: _RustLogisticRegression, device: str
    ) -> "LogisticRegression":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.C = rust_model.c
        inst.fit_intercept = rust_model.fit_intercept
        inst.max_iter = rust_model.max_iter
        inst.tol = rust_model.tol
        inst.class_weight = None
        inst.device = rust_model.device_ or device
        return inst

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "LogisticRegression":
        """Fit the logistic regression model according to the given training data (＾▽＾)."""
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            sw_list = (
                sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            )
            self._model.fit(X_mat, y_vec, sw_list)
        return self

    def decision_function(self, X: Any) -> Any:
        """Predict confidence scores for samples (*≧ω≦*)."""
        is_buf, X_data = process_features(X)
        if self._model.is_multiclass_:
            res = (
                self._model.decision_function_multi_buffer(X_data)
                if is_buf
                else self._model.decision_function_multi(X_data)
            )
        else:
            res = (
                self._model.decision_function_buffer(X_data)
                if is_buf
                else self._model.decision_function(X_data)
            )
        return _to_ndarray_or_raw(res)

    def predict_proba(self, X: Any) -> Any:
        """Probability estimates for samples (o´∀｀o)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_proba_buffer(X_data)
            if is_buf
            else self._model.predict_proba(X_data)
        )
        return _to_ndarray_or_raw(res)

    def predict_log_proba(self, X: Any) -> Any:
        """Predict logarithm of probability estimates (o´∀｀o)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_log_proba_buffer(X_data)
            if is_buf
            else self._model.predict_log_proba(X_data)
        )
        return _to_ndarray_or_raw(res)

    def predict(self, X: Any) -> Any:
        """Predict class labels for samples in X (＾▽＾)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_buffer(X_data)
            if is_buf
            else self._model.predict(X_data)
        )
        return _to_ndarray_or_raw(res)

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """Return the mean accuracy on the given test data and labels (*^▽^*)."""
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
    def load(cls, path: str, device: Optional[str] = None) -> "LogisticRegression":
        target_dev = device or _get_default_device()
        rust_model = _RustLogisticRegression.load(_validate_dit_path(path), target_dev)
        return cls._from_rust(rust_model, target_dev)

    def to(self, device: str) -> "LogisticRegression":
        self._model.set_device(device)
        self.device = device
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "state": self._model.to_bytes(),
            "device": self.device,
            "class_weight": self.class_weight,
        }

    def __setstate__(self, data: Dict[str, Any]) -> None:
        dev = data.get("device", _get_default_device())
        rust_model = _RustLogisticRegression.from_bytes(data["state"], dev)
        self._model = rust_model
        self.C = rust_model.c
        self.fit_intercept = rust_model.fit_intercept
        self.max_iter = rust_model.max_iter
        self.tol = rust_model.tol
        self.class_weight = data.get("class_weight", None)
        self.device = rust_model.device_ or dev

    @property
    def coef_(self) -> Any:
        if self._model.coef_multi_ is None:
            return None
        return _to_ndarray_or_raw(self._model.coef_multi_)

    @property
    def intercept_(self) -> Any:
        if self._model.intercept_multi_ is None:
            return None
        return _to_ndarray_or_raw(self._model.intercept_multi_)

    @property
    def classes_(self) -> Any:
        return _to_ndarray_or_raw(self._model.classes_)

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"LogisticRegression(C={self.C}, fit_intercept={self.fit_intercept}, "
            f"max_iter={self.max_iter}, tol={self.tol}, device={self.device!r})"
        )


class _BaseSparseLinearEstimator:
    """Shared Python wrapper for Lasso and ElasticNet (*≧ω≦*)."""

    _rust_cls: Any = None

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "_BaseSparseLinearEstimator":
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data, is_2d = process_target_multi(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            if is_2d:
                self._model.fit_multi_buffer(X_data, y_data, sw_data)
            else:
                self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            sw_list = (
                sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            )
            if is_2d:
                y_mat = y_data.tolist() if is_buf_y else y_data
                self._model.fit_multi(X_mat, y_mat, sw_list)
            else:
                y_vec = to_target_vector(y_data) if is_buf_y else y_data
                self._model.fit(X_mat, y_vec, sw_list)
        return self

    def predict(self, X: Any) -> Any:
        is_buf, X_data = process_features(X)
        if self._model.is_multi_target_:
            res = (
                self._model.predict_multi_buffer(X_data)
                if is_buf
                else self._model.predict_multi(X_data)
            )
        else:
            res = (
                self._model.predict_buffer(X_data)
                if is_buf
                else self._model.predict(X_data)
            )
        return _to_ndarray_or_raw(res)

    def score(self, X: Any, y: Any) -> float:
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data, is_2d = process_target_multi(y)
        if is_buf_x and is_buf_y:
            if self._model.is_multi_target_ or is_2d:
                if not is_2d:
                    import numpy as np

                    y_data = np.asarray(y_data, dtype=np.float64).reshape(-1, 1)
                return self._model.score_multi_buffer(X_data, y_data)
            return self._model.score_buffer(X_data, y_data)
        X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
        if self._model.is_multi_target_ or is_2d:
            y_mat = y_data.tolist() if is_buf_y else y_data
            if not is_2d:
                y_mat = [[v] for v in y_mat]
            return self._model.score_multi(X_mat, y_mat)
        y_vec = to_target_vector(y_data) if is_buf_y else y_data
        return self._model.score(X_mat, y_vec)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    def to(self, device: str) -> "_BaseSparseLinearEstimator":
        self._model.set_device(device)
        self.device = device
        return self

    @property
    def coef_(self) -> Any:
        if self._model.is_multi_target_:
            if self._model.coef_multi_ is None:
                return None
            return _to_ndarray_or_raw(self._model.coef_multi_)
        if self._model.coef_ is None:
            return None
        return _to_ndarray_or_raw(self._model.coef_)

    @property
    def intercept_(self) -> Any:
        if self._model.is_multi_target_:
            if self._model.intercept_multi_ is None:
                return None
            return _to_ndarray_or_raw(self._model.intercept_multi_)
        return self._model.intercept_

    @property
    def n_iter_(self) -> Any:
        iters = self._model.n_iter_
        if not iters:
            return 0
        if self._model.is_multi_target_:
            return iters
        return iters[0]

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class Lasso(_BaseSparseLinearEstimator):
    """
    Linear Model trained with L1 prior as regularizer (aka the Lasso) (＾▽＾)

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    max_iter : int, default=1000
        The maximum number of iterations.
    tol : float, default=1e-4
        The tolerance for the optimization.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustLasso

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
        device: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.positive = bool(positive)
        self.device = device or _get_default_device()
        self._model = _RustLasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustLasso, device: str) -> "Lasso":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.alpha = rust_model.alpha
        inst.fit_intercept = rust_model.fit_intercept
        inst.max_iter = rust_model.max_iter
        inst.tol = rust_model.tol
        inst.positive = rust_model.positive
        inst.device = rust_model.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "Lasso":
        target_dev = device or _get_default_device()
        return cls._from_rust(
            _RustLasso.load(_validate_dit_path(path), target_dev), target_dev
        )

    def __getstate__(self) -> Dict[str, Any]:
        return {"state": self._model.to_bytes(), "device": self.device}

    def __setstate__(self, data: Dict[str, Any]) -> None:
        dev = data.get("device", _get_default_device())
        rust_model = _RustLasso.from_bytes(data["state"], dev)
        self._model = rust_model
        self.alpha = rust_model.alpha
        self.fit_intercept = rust_model.fit_intercept
        self.max_iter = rust_model.max_iter
        self.tol = rust_model.tol
        self.positive = rust_model.positive
        self.device = rust_model.device_ or dev

    def __repr__(self) -> str:
        return (
            f"Lasso(alpha={self.alpha}, fit_intercept={self.fit_intercept}, "
            f"max_iter={self.max_iter}, tol={self.tol}, positive={self.positive}, "
            f"device={self.device!r})"
        )


class ElasticNet(_BaseSparseLinearEstimator):
    """
    Linear regression with combined L1 and L2 priors as regularizer (＾▽＾)

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not.
    max_iter : int, default=1000
        The maximum number of iterations.
    tol : float, default=1e-4
        The tolerance for the optimization.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    _rust_cls = _RustElasticNet

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
        device: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.positive = bool(positive)
        self.device = device or _get_default_device()
        self._model = _RustElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustElasticNet, device: str) -> "ElasticNet":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.alpha = rust_model.alpha
        inst.l1_ratio = rust_model.l1_ratio
        inst.fit_intercept = rust_model.fit_intercept
        inst.max_iter = rust_model.max_iter
        inst.tol = rust_model.tol
        inst.positive = rust_model.positive
        inst.device = rust_model.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ElasticNet":
        target_dev = device or _get_default_device()
        return cls._from_rust(
            _RustElasticNet.load(_validate_dit_path(path), target_dev), target_dev
        )

    def __getstate__(self) -> Dict[str, Any]:
        return {"state": self._model.to_bytes(), "device": self.device}

    def __setstate__(self, data: Dict[str, Any]) -> None:
        dev = data.get("device", _get_default_device())
        rust_model = _RustElasticNet.from_bytes(data["state"], dev)
        self._model = rust_model
        self.alpha = rust_model.alpha
        self.l1_ratio = rust_model.l1_ratio
        self.fit_intercept = rust_model.fit_intercept
        self.max_iter = rust_model.max_iter
        self.tol = rust_model.tol
        self.positive = rust_model.positive
        self.device = rust_model.device_ or dev

    def __repr__(self) -> str:
        return (
            f"ElasticNet(alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
            f"fit_intercept={self.fit_intercept}, max_iter={self.max_iter}, "
            f"tol={self.tol}, positive={self.positive}, device={self.device!r})"
        )


class SGDClassifier:
    """
    Linear classifiers (SVM, logistic regression, etc.) with SGD training (＾▽＾)

    Parameters
    ----------
    loss : {'hinge', 'log_loss', 'modified_huber'}, default='hinge'
        The loss function to be used.
    penalty : {'l2', 'l1', 'elasticnet', 'none'}, default='l2'
        The penalty (regularization term) to be used.
    alpha : float, default=0.0001
        Constant that multiplies the regularization term.
    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter.
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not.
    max_iter : int, default=1000
        The maximum number of passes over the training data (epochs).
    tol : float or None, default=1e-3
        The stopping criterion.
    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.
    learning_rate : {'optimal', 'constant', 'invscaling'}, default='optimal'
        The learning rate schedule.
    eta0 : float, default=0.0
        The initial learning rate for 'constant' or 'invscaling'.
    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
    random_state : int or None, default=None
        Used for shuffling the data, when shuffle is set to True.
    class_weight : dict, 'balanced' or None, default=None
        Preset for the class_weight fit parameter.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    def __init__(
        self,
        loss: str = "hinge",
        penalty: Optional[str] = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        shuffle: bool = True,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        random_state: Optional[int] = None,
        class_weight: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.loss = str(loss)
        self.penalty = "none" if penalty is None else str(penalty)
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = None if tol is None else float(tol)
        self.shuffle = bool(shuffle)
        self.learning_rate = str(learning_rate)
        self.eta0 = float(eta0)
        self.power_t = float(power_t)
        self.random_state = random_state
        self.class_weight = class_weight
        self.device = device or _get_default_device()
        cw_mode, cw_map = _parse_class_weight(class_weight)
        self._model = _RustSGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            random_state=self.random_state,
            class_weight_mode=cw_mode,
            class_weight_map=cw_map,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustSGDClassifier, device: str) -> "SGDClassifier":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.loss = rust_model.loss
        inst.penalty = rust_model.penalty
        inst.alpha = rust_model.alpha
        inst.l1_ratio = rust_model.l1_ratio
        inst.fit_intercept = rust_model.fit_intercept
        inst.max_iter = rust_model.max_iter
        inst.tol = rust_model.tol
        inst.shuffle = rust_model.shuffle
        inst.learning_rate = rust_model.learning_rate
        inst.eta0 = rust_model.eta0
        inst.power_t = rust_model.power_t
        inst.random_state = rust_model.random_state
        inst.class_weight = None
        inst.device = rust_model.device_ or device
        return inst

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "SGDClassifier":
        """Fit linear model with Stochastic Gradient Descent (＾▽＾)."""
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            sw_list = (
                sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            )
            self._model.fit(X_mat, y_vec, sw_list)
        return self

    def decision_function(self, X: Any) -> Any:
        """Predict confidence scores for samples (*≧ω≦*)."""
        is_buf, X_data = process_features(X)
        if self._model.is_multiclass_:
            res = (
                self._model.decision_function_multi_buffer(X_data)
                if is_buf
                else self._model.decision_function_multi(X_data)
            )
        else:
            res = (
                self._model.decision_function_buffer(X_data)
                if is_buf
                else self._model.decision_function(X_data)
            )
        return _to_ndarray_or_raw(res)

    def predict_proba(self, X: Any) -> Any:
        """Probability estimates (available for loss='log_loss' or 'modified_huber') (o´∀｀o)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_proba_buffer(X_data)
            if is_buf
            else self._model.predict_proba(X_data)
        )
        return _to_ndarray_or_raw(res)

    def predict_log_proba(self, X: Any) -> Any:
        """Log of probability estimates (o´∀｀o)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_log_proba_buffer(X_data)
            if is_buf
            else self._model.predict_log_proba(X_data)
        )
        return _to_ndarray_or_raw(res)

    def predict(self, X: Any) -> Any:
        """Predict class labels for samples in X (＾▽＾)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_buffer(X_data)
            if is_buf
            else self._model.predict(X_data)
        )
        return _to_ndarray_or_raw(res)

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """Return the mean accuracy on the given test data and labels (*^▽^*)."""
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
    def load(cls, path: str, device: Optional[str] = None) -> "SGDClassifier":
        target_dev = device or _get_default_device()
        return cls._from_rust(
            _RustSGDClassifier.load(_validate_dit_path(path), target_dev), target_dev
        )

    def to(self, device: str) -> "SGDClassifier":
        self._model.set_device(device)
        self.device = device
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "state": self._model.to_bytes(),
            "device": self.device,
            "class_weight": self.class_weight,
        }

    def __setstate__(self, data: Dict[str, Any]) -> None:
        dev = data.get("device", _get_default_device())
        rust_model = _RustSGDClassifier.from_bytes(data["state"], dev)
        self._model = rust_model
        self.loss = rust_model.loss
        self.penalty = rust_model.penalty
        self.alpha = rust_model.alpha
        self.l1_ratio = rust_model.l1_ratio
        self.fit_intercept = rust_model.fit_intercept
        self.max_iter = rust_model.max_iter
        self.tol = rust_model.tol
        self.shuffle = rust_model.shuffle
        self.learning_rate = rust_model.learning_rate
        self.eta0 = rust_model.eta0
        self.power_t = rust_model.power_t
        self.random_state = rust_model.random_state
        self.class_weight = data.get("class_weight", None)
        self.device = rust_model.device_ or dev

    @property
    def coef_(self) -> Any:
        if self._model.coef_multi_ is None:
            return None
        return _to_ndarray_or_raw(self._model.coef_multi_)

    @property
    def intercept_(self) -> Any:
        if self._model.intercept_multi_ is None:
            return None
        return _to_ndarray_or_raw(self._model.intercept_multi_)

    @property
    def classes_(self) -> Any:
        return _to_ndarray_or_raw(self._model.classes_)

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter_

    @property
    def t_(self) -> float:
        return self._model.t_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"SGDClassifier(loss={self.loss!r}, penalty={self.penalty!r}, "
            f"alpha={self.alpha}, max_iter={self.max_iter}, device={self.device!r})"
        )


class SGDRegressor:
    """
    Linear model fitted by minimizing a regularized empirical loss with SGD (＾▽＾)

    Parameters
    ----------
    loss : {'squared_error', 'huber', 'epsilon_insensitive'}, default='squared_error'
        The loss function to be used.
    penalty : {'l2', 'l1', 'elasticnet', 'none'}, default='l2'
        The penalty (regularization term) to be used.
    alpha : float, default=0.0001
        Constant that multiplies the regularization term.
    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter.
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not.
    max_iter : int, default=1000
        The maximum number of passes over the training data (epochs).
    tol : float or None, default=1e-3
        The stopping criterion.
    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.
    learning_rate : {'invscaling', 'constant', 'optimal'}, default='invscaling'
        The learning rate schedule.
    eta0 : float, default=0.01
        The initial learning rate.
    power_t : float, default=0.25
        The exponent for inverse scaling learning rate.
    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive / huber loss functions.
    random_state : int or None, default=None
        Used for shuffling the data, when shuffle is set to True.
    device : str or None, default=None
        Target computation device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu').
    """

    def __init__(
        self,
        loss: str = "squared_error",
        penalty: Optional[str] = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        shuffle: bool = True,
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        power_t: float = 0.25,
        epsilon: float = 0.1,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.loss = str(loss)
        self.penalty = "none" if penalty is None else str(penalty)
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = None if tol is None else float(tol)
        self.shuffle = bool(shuffle)
        self.learning_rate = str(learning_rate)
        self.eta0 = float(eta0)
        self.power_t = float(power_t)
        self.epsilon = float(epsilon)
        self.random_state = random_state
        self.device = device or _get_default_device()
        self._model = _RustSGDRegressor(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            epsilon=self.epsilon,
            random_state=self.random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_model: _RustSGDRegressor, device: str) -> "SGDRegressor":
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.loss = rust_model.loss
        inst.penalty = rust_model.penalty
        inst.alpha = rust_model.alpha
        inst.l1_ratio = rust_model.l1_ratio
        inst.fit_intercept = rust_model.fit_intercept
        inst.max_iter = rust_model.max_iter
        inst.tol = rust_model.tol
        inst.shuffle = rust_model.shuffle
        inst.learning_rate = rust_model.learning_rate
        inst.eta0 = rust_model.eta0
        inst.power_t = rust_model.power_t
        inst.epsilon = rust_model.epsilon
        inst.random_state = rust_model.random_state
        inst.device = rust_model.device_ or device
        return inst

    def fit(
        self, X: Any, y: Any, sample_weight: Optional[Any] = None
    ) -> "SGDRegressor":
        """Fit linear model with Stochastic Gradient Descent (＾▽＾)."""
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        is_buf_sw, sw_data = process_sample_weight(sample_weight)

        if is_buf_x and is_buf_y and is_buf_sw:
            self._model.fit_buffer(X_data, y_data, sw_data)
        else:
            X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
            y_vec = to_target_vector(y_data) if is_buf_y else y_data
            sw_list = (
                sw_data.tolist() if (is_buf_sw and sw_data is not None) else sw_data
            )
            self._model.fit(X_mat, y_vec, sw_list)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the linear model (＾▽＾)."""
        is_buf, X_data = process_features(X)
        res = (
            self._model.predict_buffer(X_data)
            if is_buf
            else self._model.predict(X_data)
        )
        return _to_ndarray_or_raw(res)

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (*^▽^*)."""
        is_buf_x, X_data = process_features(X)
        is_buf_y, y_data = process_target(y)
        if is_buf_x and is_buf_y:
            return self._model.score_buffer(X_data, y_data)
        X_mat = to_feature_matrix(X_data) if is_buf_x else X_data
        y_vec = to_target_vector(y_data) if is_buf_y else y_data
        return self._model.score(X_mat, y_vec)

    def save(self, path: str) -> None:
        self._model.save(_validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SGDRegressor":
        target_dev = device or _get_default_device()
        return cls._from_rust(
            _RustSGDRegressor.load(_validate_dit_path(path), target_dev), target_dev
        )

    def to(self, device: str) -> "SGDRegressor":
        self._model.set_device(device)
        self.device = device
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {"state": self._model.to_bytes(), "device": self.device}

    def __setstate__(self, data: Dict[str, Any]) -> None:
        dev = data.get("device", _get_default_device())
        rust_model = _RustSGDRegressor.from_bytes(data["state"], dev)
        self._model = rust_model
        self.loss = rust_model.loss
        self.penalty = rust_model.penalty
        self.alpha = rust_model.alpha
        self.l1_ratio = rust_model.l1_ratio
        self.fit_intercept = rust_model.fit_intercept
        self.max_iter = rust_model.max_iter
        self.tol = rust_model.tol
        self.shuffle = rust_model.shuffle
        self.learning_rate = rust_model.learning_rate
        self.eta0 = rust_model.eta0
        self.power_t = rust_model.power_t
        self.epsilon = rust_model.epsilon
        self.random_state = rust_model.random_state
        self.device = rust_model.device_ or dev

    @property
    def coef_(self) -> Any:
        if self._model.coef_ is None:
            return None
        return _to_ndarray_or_raw(self._model.coef_)

    @property
    def intercept_(self) -> Any:
        if self._model.intercept_ is None:
            return None
        return _to_ndarray_or_raw([self._model.intercept_])

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter_

    @property
    def t_(self) -> float:
        return self._model.t_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def __repr__(self) -> str:
        return (
            f"SGDRegressor(loss={self.loss!r}, penalty={self.penalty!r}, "
            f"alpha={self.alpha}, max_iter={self.max_iter}, device={self.device!r})"
        )


def __getattr__(name: str) -> Any:
    if name in ("LinearRegression", "Ridge", "RidgeClassifier", "RidgeCV"):
        import ochreml

        return getattr(ochreml, name)
    raise AttributeError(f"module 'ochreml.linear_model' has no attribute {name!r}")


__all__ = [
    "LinearRegression",
    "Ridge",
    "RidgeClassifier",
    "RidgeCV",
    "LogisticRegression",
    "Lasso",
    "ElasticNet",
    "SGDClassifier",
    "SGDRegressor",
]

