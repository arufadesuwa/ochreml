"""
OchreML Forest & Isolation Ensembles Submodule (＾▽＾)
Provides Scikit-Learn-compatible RandomForestClassifier, RandomForestRegressor,
ExtraTreesClassifier, ExtraTreesRegressor, and IsolationForest backed by Rust
N6 histogram acceleration on CUDA/wgpu and Rayon parallelism on CPU/TPU (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    ExtraTreesClassifier as _RustExtraTreesClassifier,
    ExtraTreesRegressor as _RustExtraTreesRegressor,
    IsolationForest as _RustIsolationForest,
    RandomForestClassifier as _RustRandomForestClassifier,
    RandomForestRegressor as _RustRandomForestRegressor,
)
from ._preproc_utils import bytes_to_f64_2d, bytes_to_i64_1d, validate_dit_path
from ._utils import process_features, process_sample_weight, process_target


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _prep_x(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    return np.asarray(X_data, dtype=np.float64)


def _prep_y(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if is_buf:
        return y_data
    return np.asarray(y_data, dtype=np.float64)


def _prep_sw(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if is_buf:
        return sw_data
    return np.asarray(sw_data, dtype=np.float64)


def _bytes_to_f64_1d(raw: bytes) -> Any:
    return np.frombuffer(raw, dtype=np.float64).copy()


def _normalize_max_features(max_features: Union[str, int, float, None]) -> Optional[str]:
    if max_features is None:
        return None
    if isinstance(max_features, bool):
        raise ValueError("max_features cannot be a bool ( >_< )")
    if isinstance(max_features, int):
        if max_features < 1:
            raise ValueError(f"Integer max_features={max_features} must be >= 1 ( >_< )")
        return str(max_features)
    if isinstance(max_features, float):
        if not np.isfinite(max_features) or max_features <= 0.0 or max_features > 1.0:
            raise ValueError(
                f"Float max_features={max_features} must be in (0.0, 1.0] ( >_< )"
            )
        return repr(float(max_features))
    if isinstance(max_features, str):
        return max_features
    raise TypeError(f"Unsupported max_features type: {type(max_features)} ( >_< )")


class _BaseForestMixin:
    """Shared persistence, device management, and attribute helpers for forest estimators."""

    _model: Any
    device: str

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def feature_importances_(self) -> Optional[Any]:
        raw = self._model.feature_importances_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save the fitted ensemble model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    def to(self, device: str) -> Any:
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def set_device(self, device: Optional[str] = None) -> Any:
        """Set compute device for the estimator (*≧ω≦*)."""
        resolved = _resolve_device(device)
        self._model.set_device(resolved)
        self.device = self._model.device_ or resolved
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()


class _BaseForestClassifier(_BaseForestMixin):
    """Shared fit/predict/predict_proba/score implementation for forest classifiers."""

    _rust_cls: Any

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.device = _resolve_device(device)

        mf_str = _normalize_max_features(max_features)
        self._model = self._rust_cls(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=mf_str,
            bootstrap=bootstrap,
            random_state=random_state,
            device=self.device,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """Fit the forest classifier on training data (＾▽＾)."""
        self._model.fit_buffer(_prep_x(X), _prep_y(y), _prep_sw(sample_weight))
        return self

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities for X (o´∀｀o)."""
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def predict(self, X: Any) -> Any:
        """Predict class labels for X (＾▽＾)."""
        raw = self._model.predict_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on the given test data and labels (*≧ω≦*)."""
        return float(self._model.score_buffer(_prep_x(X), _prep_y(y)))

    @property
    def classes_(self) -> Optional[Any]:
        raw = self._model.classes_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load a fitted classifier from a '.dit' file (*≧ω≦*)."""
        inst = cls.__new__(cls)
        inst._model = cls._rust_cls.load(validate_dit_path(path), device)
        inst.device = inst._model.device_ or _resolve_device(device)
        return inst

    def __setstate__(self, state: bytes) -> None:
        self._model = self._rust_cls.from_bytes(state, None)
        self.device = self._model.device_ or "auto"


class _BaseForestRegressor(_BaseForestMixin):
    """Shared fit/predict/score implementation for forest regressors."""

    _rust_cls: Any

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.device = _resolve_device(device)

        mf_str = _normalize_max_features(max_features)
        self._model = self._rust_cls(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=mf_str,
            bootstrap=bootstrap,
            random_state=random_state,
            device=self.device,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """Fit the forest regressor on training data (＾▽＾)."""
        self._model.fit_buffer(_prep_x(X), _prep_y(y), _prep_sw(sample_weight))
        return self

    def predict(self, X: Any) -> Any:
        """Predict regression targets for X (o´∀｀o)."""
        raw = self._model.predict_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R^2 of the prediction (*≧ω≦*)."""
        return float(self._model.score_buffer(_prep_x(X), _prep_y(y)))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load a fitted regressor from a '.dit' file (*≧ω≦*)."""
        inst = cls.__new__(cls)
        inst._model = cls._rust_cls.load(validate_dit_path(path), device)
        inst.device = inst._model.device_ or _resolve_device(device)
        return inst

    def __setstate__(self, state: bytes) -> None:
        self._model = self._rust_cls.from_bytes(state, None)
        self.device = self._model.device_ or "auto"


class RandomForestClassifier(_BaseForestClassifier):
    """
    Random Forest Classifier (＾▽＾)
    An ensemble of randomized decision trees trained with bootstrap sampling and
    feature subsampling in parallel across CPU/TPU cores (*≧ω≦*).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'}, int, float, or None, default='sqrt'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    random_state : int or None, default=None
        Controls both the randomness of the bootstrapping and feature sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu', etc.).
    """

    _rust_cls = _RustRandomForestClassifier


class ExtraTreesClassifier(_BaseForestClassifier):
    """
    Extra-Trees Classifier (o´∀｀o)
    An ensemble of extremely randomized trees that draw random split thresholds
    for candidate features (*≧ω≦*).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'}, int, float, or None, default='sqrt'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees.
    random_state : int or None, default=None
        Controls the randomness of threshold selection and feature sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu', etc.).
    """

    _rust_cls = _RustExtraTreesClassifier

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = "sqrt",
        bootstrap: bool = False,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            device=device,
        )


class RandomForestRegressor(_BaseForestRegressor):
    """
    Random Forest Regressor (＾▽＾)
    An ensemble of regression trees using N6 histogram acceleration on CUDA/wgpu
    and Rayon parallelism on CPU/TPU (*≧ω≦*).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'squared_error'}, default='squared_error'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'}, int, float, or None, default=1.0
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    random_state : int or None, default=None
        Controls both the randomness of the bootstrapping and feature sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu', etc.).
    """

    _rust_cls = _RustRandomForestRegressor


class ExtraTreesRegressor(_BaseForestRegressor):
    """
    Extra-Trees Regressor (o´∀｀o)
    An ensemble of extremely randomized regression trees that draw random split
    thresholds for candidate features (*≧ω≦*).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'squared_error'}, default='squared_error'
        The function to measure the quality of a split.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'}, int, float, or None, default=1.0
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees.
    random_state : int or None, default=None
        Controls the randomness of threshold selection and feature sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu', etc.).
    """

    _rust_cls = _RustExtraTreesRegressor

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            device=device,
        )


class IsolationForest(_BaseForestMixin):
    """
    Isolation Forest (＾▽＾)
    Unsupervised anomaly detection algorithm (Liu et al., 2008) that isolates
    observations by randomly selecting features and split thresholds (*≧ω≦*).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base isolation trees in the ensemble.
    max_samples : 'auto', int, or float, default='auto'
        The number of samples to draw from X to train each base estimator.
    contamination : 'auto' or float, default='auto'
        The proportion of outliers in the data set, in (0, 0.5].
    max_features : int or float, default=1.0
        The number (or fraction) of features to draw from X to train each tree.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets sampled with replacement.
    random_state : int or None, default=None
        Controls pseudo-randomness of feature and split selection.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu', etc.).
    """

    _rust_cls = _RustIsolationForest

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.device = _resolve_device(device)

        ms_rust = self._validate_max_samples(max_samples)
        cont_rust = self._validate_contamination(contamination)
        mf_rust = self._validate_max_features_init(max_features)

        self._model = _RustIsolationForest(
            n_estimators=n_estimators,
            max_samples=ms_rust,
            contamination=cont_rust,
            max_features=mf_rust,
            bootstrap=bootstrap,
            random_state=random_state,
            device=self.device,
        )

    @staticmethod
    def _validate_max_samples(max_samples: Union[str, int, float]) -> Optional[int]:
        if isinstance(max_samples, str):
            if max_samples == "auto":
                return None
            raise ValueError(f"Invalid max_samples='{max_samples}' ( >_< )")
        if isinstance(max_samples, bool):
            raise ValueError("max_samples cannot be a bool ( >_< )")
        if isinstance(max_samples, int):
            if max_samples < 1:
                raise ValueError(f"max_samples={max_samples} must be >= 1 ( >_< )")
            return max_samples
        if isinstance(max_samples, float):
            if not np.isfinite(max_samples) or max_samples <= 0.0 or max_samples > 1.0:
                raise ValueError(
                    f"Float max_samples={max_samples} must be in (0.0, 1.0] ( >_< )"
                )
            return None
        raise TypeError(f"Unsupported max_samples type: {type(max_samples)} ( >_< )")

    @staticmethod
    def _validate_contamination(contamination: Union[str, float]) -> Optional[float]:
        if isinstance(contamination, str):
            if contamination == "auto":
                return None
            raise ValueError(f"Invalid contamination='{contamination}' ( >_< )")
        if isinstance(contamination, bool):
            raise ValueError("contamination cannot be a bool ( >_< )")
        if isinstance(contamination, (int, float)):
            c = float(contamination)
            if not np.isfinite(c) or c <= 0.0 or c > 0.5:
                raise ValueError(
                    f"contamination={c} must be in (0.0, 0.5] or 'auto' ( >_< )"
                )
            return c
        raise TypeError(f"Unsupported contamination type: {type(contamination)} ( >_< )")

    @staticmethod
    def _validate_max_features_init(max_features: Union[int, float]) -> float:
        if isinstance(max_features, bool):
            raise ValueError("max_features cannot be a bool ( >_< )")
        if isinstance(max_features, int):
            if max_features < 1:
                raise ValueError(f"max_features={max_features} must be >= 1 ( >_< )")
            return 1.0
        if isinstance(max_features, float):
            if not np.isfinite(max_features) or max_features <= 0.0 or max_features > 1.0:
                raise ValueError(
                    f"Float max_features={max_features} must be in (0.0, 1.0] ( >_< )"
                )
            return float(max_features)
        raise TypeError(f"Unsupported max_features type: {type(max_features)} ( >_< )")

    def fit(self, X: Any, y: Optional[Any] = None) -> IsolationForest:
        """Fit the IsolationForest estimator on X (＾▽＾)."""
        X_arr = _prep_x(X)
        n_samples, n_features = X_arr.shape

        ms_rust = self._validate_max_samples(self.max_samples)
        if isinstance(self.max_samples, float):
            ms_rust = max(1, int(self.max_samples * n_samples))

        if isinstance(self.max_features, int) and not isinstance(self.max_features, bool):
            if self.max_features > n_features:
                raise ValueError(
                    f"max_features={self.max_features} exceeds n_features={n_features} ( >_< )"
                )
            mf_rust = float(self.max_features) / float(n_features)
        else:
            mf_rust = float(self.max_features)

        if isinstance(self.max_samples, float) or (
            isinstance(self.max_features, int) and not isinstance(self.max_features, bool)
        ):
            self._model = _RustIsolationForest(
                n_estimators=self.n_estimators,
                max_samples=ms_rust,
                contamination=self._validate_contamination(self.contamination),
                max_features=mf_rust,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                device=self.device,
            )

        self._model.fit_buffer(X_arr)
        return self

    def score_samples(self, X: Any) -> Any:
        """Opposite of the anomaly score defined in Liu et al. (2008) (o´∀｀o)."""
        raw = self._model.score_samples_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def decision_function(self, X: Any) -> Any:
        """Average anomaly score of X of the base classifiers (*≧ω≦*)."""
        raw = self._model.decision_function_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def predict(self, X: Any) -> Any:
        """Predict if a particular sample is an outlier (-1) or inlier (1) (＾▽＾)."""
        raw = self._model.predict_buffer(_prep_x(X))
        return bytes_to_i64_1d(raw)

    @property
    def max_samples_(self) -> Optional[int]:
        return self._model.max_samples_

    @property
    def offset_(self) -> Optional[float]:
        return self._model.offset_

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> IsolationForest:
        """Load a fitted IsolationForest from a '.dit' file (*≧ω≦*)."""
        inst = cls.__new__(cls)
        inst._model = _RustIsolationForest.load(validate_dit_path(path), device)
        inst.device = inst._model.device_ or _resolve_device(device)
        return inst

    def __setstate__(self, state: bytes) -> None:
        self._model = _RustIsolationForest.from_bytes(state, None)
        self.device = self._model.device_ or "auto"
