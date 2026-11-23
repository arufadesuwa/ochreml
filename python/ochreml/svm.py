"""
OchreML Support Vector Machines Module (＾▽＾)
Provides Scikit-Learn-compatible LinearSVC and LinearSVR (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    LinearSVC as _RustLinearSVC,
    LinearSVR as _RustLinearSVR,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_sample_weight, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_2d(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if not is_buf:
        X_data = np.asarray(X_data, dtype=np.float64)
    return X_data


def _to_f64_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if not is_buf:
        y_data = np.asarray(y_data, dtype=np.float64)
    return y_data


def _to_sample_weight_1d(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if not is_buf and sw_data is not None:
        sw_data = np.asarray(sw_data, dtype=np.float64)
    return sw_data


def _parse_class_weight(
    class_weight: Optional[Any],
) -> Tuple[Optional[str], Optional[List[Tuple[float, float]]]]:
    if class_weight is None:
        return None, None
    if isinstance(class_weight, str):
        return class_weight, None
    if isinstance(class_weight, dict):
        return None, [(float(k), float(v)) for k, v in class_weight.items()]
    raise ValueError(
        f"class_weight must be None, 'balanced', or a dict, got {type(class_weight)} (・`ω´・)"
    )


class LinearSVC:
    """
    Linear Support Vector Classification (＾▽＾).

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear-style dual coordinate descent and primal L-BFGS / coordinate
    descent for better scalability on large datasets (*≧ω≦*).

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default='l2'
        Specifies the norm used in the penalization.
    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
        Specifies the loss function.
    dual : 'auto' or bool, default='auto'
        Select the algorithm to either solve the dual or primal optimization
        problem. When 'auto', selects `n_samples < n_features` for L2
        squared_hinge, `True` for hinge loss, and `False` for L1 penalty.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    fit_intercept : bool, default=True
        Whether or not to fit an intercept.
    intercept_scaling : float, default=1.0
        Synthetic feature value appended when `fit_intercept=True`.
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes.
    random_state : int or None, default=None
        Controls the pseudo random number generation for shuffling the data.
    max_iter : int, default=1000
        The maximum number of iterations to be run.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustLinearSVC

    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        *,
        dual: Union[str, bool] = "auto",
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Optional[Any] = None,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        device: Optional[str] = None,
    ) -> None:
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.device = _resolve_device(device)

        if dual != "auto" and not isinstance(dual, bool):
            raise ValueError(
                f"dual must be 'auto' or a bool, got {dual!r} (・`ω´・)"
            )

        init_dual = self._resolve_dual_init()
        cw_mode, cw_map = _parse_class_weight(class_weight)
        self._model = _RustLinearSVC(
            c=float(C),
            loss=loss,
            penalty=penalty,
            dual=init_dual,
            tol=float(tol),
            fit_intercept=bool(fit_intercept),
            intercept_scaling=float(intercept_scaling),
            class_weight_mode=cw_mode,
            class_weight_map=cw_map,
            max_iter=int(max_iter),
            random_state=int(random_state) if random_state is not None else None,
            device=self.device,
        )

    def _resolve_dual_init(self) -> bool:
        if self.dual == "auto":
            if self.loss == "hinge":
                return True
            if self.penalty == "l1":
                return False
            return True
        return bool(self.dual)

    def _resolve_dual_for_shape(self, n_samples: int, n_features: int) -> bool:
        if self.dual == "auto":
            if self.loss == "hinge":
                return True
            if self.penalty == "l1":
                return False
            return n_samples < n_features
        return bool(self.dual)

    @classmethod
    def _from_rust(cls, rust_obj: _RustLinearSVC, device: str) -> "LinearSVC":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.C = rust_obj.c
        inst.loss = rust_obj.loss
        inst.penalty = rust_obj.penalty
        inst.dual = rust_obj.dual
        inst.tol = rust_obj.tol
        inst.fit_intercept = rust_obj.fit_intercept
        inst.intercept_scaling = rust_obj.intercept_scaling
        inst.class_weight = None
        inst.max_iter = rust_obj.max_iter
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "LinearSVC":
        """
        Fit the model according to the given training data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Array of weights that are assigned to individual samples.

        Returns
        -------
        self : LinearSVC
            An instance of the estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_sample_weight_1d(sample_weight)

        if self.dual == "auto":
            resolved_dual = self._resolve_dual_for_shape(
                int(X_data.shape[0]), int(X_data.shape[1])
            )
            if resolved_dual != self._model.dual:
                cw_mode, cw_map = _parse_class_weight(self.class_weight)
                self._model = _RustLinearSVC(
                    c=float(self.C),
                    loss=self.loss,
                    penalty=self.penalty,
                    dual=resolved_dual,
                    tol=float(self.tol),
                    fit_intercept=bool(self.fit_intercept),
                    intercept_scaling=float(self.intercept_scaling),
                    class_weight_mode=cw_mode,
                    class_weight_map=cw_map,
                    max_iter=int(self.max_iter),
                    random_state=(
                        int(self.random_state) if self.random_state is not None else None
                    ),
                    device=self.device,
                )

        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def decision_function(self, X: Any) -> Any:
        """
        Predict confidence scores for samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.

        Returns
        -------
        scores : numpy.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per (n_samples, n_classes) combination.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.decision_function_buffer(X_data)
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    def predict(self, X: Any) -> Any:
        """
        Predict class labels for samples in X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        X_data = _to_f64_2d(X)
        raw = self._model.predict_buffer(X_data)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. ``y``.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        return float(self._model.score_buffer(X_data, y_data))

    def save(self, path: str) -> None:
        """Save fitted LinearSVC model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LinearSVC":
        """Load LinearSVC model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "LinearSVC":
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model": self._model.to_bytes(),
            "dual": self.dual,
            "class_weight": self.class_weight,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLinearSVC.from_bytes(state["model"], target_dev)
        self.C = self._model.c
        self.loss = self._model.loss
        self.penalty = self._model.penalty
        self.dual = state.get("dual", self._model.dual)
        self.tol = self._model.tol
        self.fit_intercept = self._model.fit_intercept
        self.intercept_scaling = self._model.intercept_scaling
        self.class_weight = state.get("class_weight", None)
        self.max_iter = self._model.max_iter
        self.random_state = self._model.random_state
        self.device = self._model.device_ or target_dev

    @property
    def coef_(self) -> Optional[Any]:
        res = self._model.coef_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def intercept_(self) -> Optional[Any]:
        b = self._model.intercept_
        return np.asarray(b, dtype=np.float64) if b is not None else None

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return np.asarray(c, dtype=np.float64) if c is not None else None

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)


class LinearSVR:
    """
    Linear Support Vector Regression (＾▽＾).

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear-style dual coordinate descent and primal L-BFGS for better
    scalability on large datasets (*≧ω≦*).

    Parameters
    ----------
    epsilon : float, default=0.0
        Epsilon parameter in the epsilon-insensitive loss function.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Regularization parameter. Must be strictly positive.
    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, default='epsilon_insensitive'
        Specifies the loss function.
    fit_intercept : bool, default=True
        Whether or not to fit an intercept.
    intercept_scaling : float, default=1.0
        Synthetic feature value appended when `fit_intercept=True`.
    dual : 'auto' or bool, default='auto'
        Select the algorithm to either solve the dual or primal optimization
        problem. When 'auto', selects `True` for `epsilon_insensitive` and
        `n_samples < n_features` for `squared_epsilon_insensitive`.
    random_state : int or None, default=None
        Controls the pseudo random number generation for shuffling the data.
    max_iter : int, default=1000
        The maximum number of iterations to be run.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustLinearSVR

    def __init__(
        self,
        *,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        C: float = 1.0,
        loss: str = "epsilon_insensitive",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        dual: Union[str, bool] = "auto",
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        device: Optional[str] = None,
    ) -> None:
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.random_state = random_state
        self.max_iter = max_iter
        self.device = _resolve_device(device)

        if dual != "auto" and not isinstance(dual, bool):
            raise ValueError(
                f"dual must be 'auto' or a bool, got {dual!r} (・`ω´・)"
            )

        init_dual = True if dual == "auto" else bool(dual)
        self._model = _RustLinearSVR(
            epsilon=float(epsilon),
            tol=float(tol),
            c=float(C),
            loss=loss,
            fit_intercept=bool(fit_intercept),
            intercept_scaling=float(intercept_scaling),
            dual=init_dual,
            max_iter=int(max_iter),
            random_state=int(random_state) if random_state is not None else None,
            device=self.device,
        )

    def _resolve_dual_for_shape(self, n_samples: int, n_features: int) -> bool:
        if self.dual == "auto":
            if self.loss == "epsilon_insensitive":
                return True
            return n_samples < n_features
        return bool(self.dual)

    @classmethod
    def _from_rust(cls, rust_obj: _RustLinearSVR, device: str) -> "LinearSVR":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.epsilon = rust_obj.epsilon
        inst.tol = rust_obj.tol
        inst.C = rust_obj.c
        inst.loss = rust_obj.loss
        inst.fit_intercept = rust_obj.fit_intercept
        inst.intercept_scaling = rust_obj.intercept_scaling
        inst.dual = rust_obj.dual
        inst.max_iter = rust_obj.max_iter
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "LinearSVR":
        """
        Fit the model according to the given training data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Array of weights that are assigned to individual samples.

        Returns
        -------
        self : LinearSVR
            An instance of the estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_sample_weight_1d(sample_weight)

        if self.dual == "auto":
            resolved_dual = self._resolve_dual_for_shape(
                int(X_data.shape[0]), int(X_data.shape[1])
            )
            if resolved_dual != self._model.dual:
                self._model = _RustLinearSVR(
                    epsilon=float(self.epsilon),
                    tol=float(self.tol),
                    c=float(self.C),
                    loss=self.loss,
                    fit_intercept=bool(self.fit_intercept),
                    intercept_scaling=float(self.intercept_scaling),
                    dual=resolved_dual,
                    max_iter=int(self.max_iter),
                    random_state=(
                        int(self.random_state) if self.random_state is not None else None
                    ),
                    device=self.device,
                )

        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict using the linear model (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples,)
            Returns predicted values.
        """
        X_data = _to_f64_2d(X)
        raw = self._model.predict_buffer(X_data)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of ``self.predict(X)`` w.r.t. ``y``.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        return float(self._model.score_buffer(X_data, y_data))

    def save(self, path: str) -> None:
        """Save fitted LinearSVR model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LinearSVR":
        """Load LinearSVR model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "LinearSVR":
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model": self._model.to_bytes(),
            "dual": self.dual,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLinearSVR.from_bytes(state["model"], target_dev)
        self.epsilon = self._model.epsilon
        self.tol = self._model.tol
        self.C = self._model.c
        self.loss = self._model.loss
        self.fit_intercept = self._model.fit_intercept
        self.intercept_scaling = self._model.intercept_scaling
        self.dual = state.get("dual", self._model.dual)
        self.max_iter = self._model.max_iter
        self.random_state = self._model.random_state
        self.device = self._model.device_ or target_dev

    @property
    def coef_(self) -> Optional[Any]:
        c = self._model.coef_
        return np.asarray(c, dtype=np.float64) if c is not None else None

    @property
    def intercept_(self) -> Optional[Any]:
        b = self._model.intercept_
        return np.asarray([b], dtype=np.float64) if b is not None else None

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)
