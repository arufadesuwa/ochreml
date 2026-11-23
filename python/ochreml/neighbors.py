"""
OchreML Nearest Neighbors Submodule (＾▽＾)
Provides Scikit-Learn-compatible KNeighborsClassifier, KNeighborsRegressor,
RadiusNeighborsClassifier, and RadiusNeighborsRegressor backed by Rust
hardware primitives N3 (pairwise_l2_sq) and N4 (topk_argmin) (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    KNeighborsClassifier as _RustKNeighborsClassifier,
    KNeighborsRegressor as _RustKNeighborsRegressor,
    RadiusNeighborsClassifier as _RustRadiusNeighborsClassifier,
    RadiusNeighborsRegressor as _RustRadiusNeighborsRegressor,
)
from ._preproc_utils import (
    bytes_to_f64_2d,
    bytes_to_i64_1d,
    bytes_to_i64_2d,
    validate_dit_path,
)
from ._utils import process_features, process_target, process_target_multi


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


def _prep_y_1d(y: Any) -> Any:
    is_buf, y_data = process_target(y)
    if is_buf:
        return y_data
    return np.asarray(y_data, dtype=np.float64)


def _prep_y_multi(y: Any) -> Tuple[Any, bool]:
    is_buf, y_data, is_2d = process_target_multi(y)
    if not is_buf:
        y_data = np.asarray(y_data, dtype=np.float64)
    return y_data, is_2d


def _bytes_to_f64_1d(raw: bytes) -> Any:
    return np.frombuffer(raw, dtype=np.float64).copy()


def _unpack_radius_neighbors(
    off_bytes: bytes,
    idx_bytes: bytes,
    dist_bytes: bytes,
    return_distance: bool,
) -> Any:
    offsets = bytes_to_i64_1d(off_bytes)
    indices_flat = bytes_to_i64_1d(idx_bytes)
    n_queries = len(offsets) - 1
    idx_arr = np.empty(n_queries, dtype=object)
    for i in range(n_queries):
        idx_arr[i] = indices_flat[offsets[i] : offsets[i + 1]]
    if not return_distance:
        return idx_arr

    dists_flat = _bytes_to_f64_1d(dist_bytes)
    dist_arr = np.empty(n_queries, dtype=object)
    for i in range(n_queries):
        dist_arr[i] = dists_flat[offsets[i] : offsets[i + 1]]
    return dist_arr, idx_arr


class _BaseNeighborsMixin:
    """Shared persistence, device management, and property helpers for neighbors estimators."""

    _model: Any
    device: str

    def _sync_params_from_rust(self) -> None:
        raise NotImplementedError

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_samples_fit_(self) -> Optional[int]:
        return self._model.n_samples_fit_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save the fitted neighbors estimator to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    def to(self, device: str) -> Any:
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        self._model = self._rust_cls.from_bytes(state, None)
        self._sync_params_from_rust()


class _KNeighborsMixin(_BaseNeighborsMixin):
    """Shared kneighbors query helper for KNeighborsClassifier and KNeighborsRegressor."""

    def kneighbors(
        self,
        X: Optional[Any] = None,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Any:
        """
        Find the K-neighbors of a point (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            The query point or points. If None, neighbors of each indexed
            point are returned (excluding the point itself).
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            `return_distance=True`.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        x_buf = _prep_x(X) if X is not None else None
        k_arg = int(n_neighbors) if n_neighbors is not None else None
        rows, cols, d_bytes, i_bytes = self._model.kneighbors_buffer(x_buf, k_arg)
        indices = bytes_to_i64_2d(rows, cols, i_bytes)
        if not return_distance:
            return indices
        distances = bytes_to_f64_2d(rows, cols, d_bytes)
        return distances, indices


class _RadiusNeighborsMixin(_BaseNeighborsMixin):
    """Shared radius_neighbors query helper for RadiusNeighborsClassifier and RadiusNeighborsRegressor."""

    def radius_neighbors(
        self,
        X: Optional[Any] = None,
        radius: Optional[float] = None,
        return_distance: bool = True,
        sort_results: bool = False,
    ) -> Any:
        """
        Find the neighbors within a given radius of a point or points (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            The query point or points. If None, neighbors of each indexed
            point are returned (excluding the point itself).
        radius : float, default=None
            Limiting distance of neighbors to return. The default is the value
            passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.
        sort_results : bool, default=False
            If True, the distances and indices will be sorted by increasing
            distance before being returned.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries,) of arrays
            Array representing the distances to each point, only present if
            `return_distance=True`.
        neigh_ind : ndarray of shape (n_queries,) of arrays
            An array of arrays of indices of the approximate nearest points.
        """
        x_buf = _prep_x(X) if X is not None else None
        r_arg = float(radius) if radius is not None else None
        off_bytes, idx_bytes, dist_bytes = self._model.radius_neighbors_buffer(
            x_buf, r_arg, bool(sort_results)
        )
        return _unpack_radius_neighbors(
            off_bytes, idx_bytes, dist_bytes, bool(return_distance)
        )


class KNeighborsClassifier(_KNeighborsMixin):
    """
    Classifier implementing the k-nearest neighbors vote (＾▽＾).

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'minkowski', 'euclidean', 'l2', 'manhattan', 'l1', 'chebyshev', 'cosine'}, default='minkowski'
        Metric to use for distance computation.
    p : float, default=2.0
        Power parameter for the Minkowski metric.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustKNeighborsClassifier

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.p = float(p)
        self.device = _resolve_device(device)
        self._model = _RustKNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.device = self._model.device_ or "cpu"

    def fit(self, X: Any, y: Any) -> "KNeighborsClassifier":
        """
        Fit the k-nearest neighbors classifier from the training dataset (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        self._model.fit_buffer(_prep_x(X), _prep_y_1d(y))
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict the class labels for the provided data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def predict_proba(self, X: Any) -> Any:
        """
        Return probability estimates for the test data X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes)
            The class probabilities of the input samples.
        """
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return float(self._model.score_buffer(_prep_x(X), _prep_y_1d(y)))

    @property
    def classes_(self) -> Optional[Any]:
        raw = self._model.classes_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KNeighborsClassifier":
        """Load a fitted KNeighborsClassifier from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_model = _RustKNeighborsClassifier.load(validate_dit_path(path), target_dev)
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.n_neighbors = 5
        inst.weights = "uniform"
        inst.metric = "minkowski"
        inst.p = 2.0
        inst.device = rust_model.device_ or target_dev
        return inst


class KNeighborsRegressor(_KNeighborsMixin):
    """
    Regression based on k-nearest neighbors (＾▽＾).

    Supports both single-output 1D targets and multi-output 2D targets (*≧ω≦*).

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'minkowski', 'euclidean', 'l2', 'manhattan', 'l1', 'chebyshev', 'cosine'}, default='minkowski'
        Metric to use for distance computation.
    p : float, default=2.0
        Power parameter for the Minkowski metric.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustKNeighborsRegressor

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.p = float(p)
        self.device = _resolve_device(device)
        self._model = _RustKNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.device = self._model.device_ or "cpu"

    def fit(self, X: Any, y: Any) -> "KNeighborsRegressor":
        """
        Fit the k-nearest neighbors regressor from the training dataset (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsRegressor
            The fitted k-nearest neighbors regressor.
        """
        x_buf = _prep_x(X)
        y_buf, is_2d = _prep_y_multi(y)
        if is_2d:
            self._model.fit_multi_buffer(x_buf, y_buf)
        else:
            self._model.fit_buffer(x_buf, y_buf)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict the target for the provided data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Target values.
        """
        rows, cols, raw = self._model.predict_buffer(_prep_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw)
        if not self._model.is_multi_output_:
            return arr.ravel()
        return arr

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        score : float
            R^2 of `self.predict(X)` w.r.t. `y`.
        """
        x_buf = _prep_x(X)
        y_buf, is_2d = _prep_y_multi(y)
        if is_2d:
            return float(self._model.score_multi_buffer(x_buf, y_buf))
        return float(self._model.score_buffer(x_buf, y_buf))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KNeighborsRegressor":
        """Load a fitted KNeighborsRegressor from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_model = _RustKNeighborsRegressor.load(validate_dit_path(path), target_dev)
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.n_neighbors = 5
        inst.weights = "uniform"
        inst.metric = "minkowski"
        inst.p = 2.0
        inst.device = rust_model.device_ or target_dev
        return inst


class RadiusNeighborsClassifier(_RadiusNeighborsMixin):
    """
    Classifier implementing a vote among neighbors within a given radius (＾▽＾).

    Parameters
    ----------
    radius : float, default=1.0
        Range of parameter space to use by default for `radius_neighbors` queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'minkowski', 'euclidean', 'l2', 'manhattan', 'l1', 'chebyshev', 'cosine'}, default='minkowski'
        Metric to use for distance computation.
    p : float, default=2.0
        Power parameter for the Minkowski metric.
    outlier_label : float, int, 'most_frequent', or None, default=None
        Label for outlier samples (samples with no neighbors in given radius).
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustRadiusNeighborsClassifier

    def __init__(
        self,
        radius: float = 1.0,
        *,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: float = 2.0,
        outlier_label: Optional[Union[float, int, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.radius = float(radius)
        self.weights = weights
        self.metric = metric
        self.p = float(p)
        self.outlier_label = outlier_label
        self.device = _resolve_device(device)

        if outlier_label is None:
            ol_val: Optional[float] = None
            ol_most_freq = False
        elif outlier_label == "most_frequent":
            ol_val = None
            ol_most_freq = True
        elif isinstance(outlier_label, (int, float)) and not isinstance(outlier_label, bool):
            ol_val = float(outlier_label)
            ol_most_freq = False
        else:
            raise ValueError(
                f"outlier_label must be None, 'most_frequent', or a number, got {outlier_label!r} ( >_< )"
            )

        self._model = _RustRadiusNeighborsClassifier(
            radius=self.radius,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            outlier_label=ol_val,
            outlier_most_frequent=ol_most_freq,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.device = self._model.device_ or "cpu"

    def fit(self, X: Any, y: Any) -> "RadiusNeighborsClassifier":
        """
        Fit the radius neighbors classifier from the training dataset (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RadiusNeighborsClassifier
            The fitted radius neighbors classifier.
        """
        self._model.fit_buffer(_prep_x(X), _prep_y_1d(y))
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict the class labels for the provided data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return _bytes_to_f64_1d(raw)

    def predict_proba(self, X: Any) -> Any:
        """
        Return probability estimates for the test data X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes)
            The class probabilities of the input samples.
        """
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return float(self._model.score_buffer(_prep_x(X), _prep_y_1d(y)))

    @property
    def classes_(self) -> Optional[Any]:
        raw = self._model.classes_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @property
    def outlier_label_(self) -> Optional[float]:
        return self._model.outlier_label_

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RadiusNeighborsClassifier":
        """Load a fitted RadiusNeighborsClassifier from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_model = _RustRadiusNeighborsClassifier.load(validate_dit_path(path), target_dev)
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.radius = 1.0
        inst.weights = "uniform"
        inst.metric = "minkowski"
        inst.p = 2.0
        inst.outlier_label = rust_model.outlier_label_
        inst.device = rust_model.device_ or target_dev
        return inst


class RadiusNeighborsRegressor(_RadiusNeighborsMixin):
    """
    Regression based on neighbors within a fixed radius (＾▽＾).

    Supports both single-output 1D targets and multi-output 2D targets (*≧ω≦*).

    Parameters
    ----------
    radius : float, default=1.0
        Range of parameter space to use by default for `radius_neighbors` queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'minkowski', 'euclidean', 'l2', 'manhattan', 'l1', 'chebyshev', 'cosine'}, default='minkowski'
        Metric to use for distance computation.
    p : float, default=2.0
        Power parameter for the Minkowski metric.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustRadiusNeighborsRegressor

    def __init__(
        self,
        radius: float = 1.0,
        *,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        self.radius = float(radius)
        self.weights = weights
        self.metric = metric
        self.p = float(p)
        self.device = _resolve_device(device)
        self._model = _RustRadiusNeighborsRegressor(
            radius=self.radius,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.device = self._model.device_ or "cpu"

    def fit(self, X: Any, y: Any) -> "RadiusNeighborsRegressor":
        """
        Fit the radius neighbors regressor from the training dataset (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : RadiusNeighborsRegressor
            The fitted radius neighbors regressor.
        """
        x_buf = _prep_x(X)
        y_buf, is_2d = _prep_y_multi(y)
        if is_2d:
            self._model.fit_multi_buffer(x_buf, y_buf)
        else:
            self._model.fit_buffer(x_buf, y_buf)
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict the target for the provided data (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Target values. Samples with no neighbors within `radius` receive `NaN`.
        """
        rows, cols, raw = self._model.predict_buffer(_prep_x(X))
        arr = bytes_to_f64_2d(rows, cols, raw)
        if not self._model.is_multi_output_:
            return arr.ravel()
        return arr

    def score(self, X: Any, y: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        score : float
            R^2 of `self.predict(X)` w.r.t. `y`.
        """
        x_buf = _prep_x(X)
        y_buf, is_2d = _prep_y_multi(y)
        if is_2d:
            return float(self._model.score_multi_buffer(x_buf, y_buf))
        return float(self._model.score_buffer(x_buf, y_buf))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RadiusNeighborsRegressor":
        """Load a fitted RadiusNeighborsRegressor from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_model = _RustRadiusNeighborsRegressor.load(validate_dit_path(path), target_dev)
        inst = cls.__new__(cls)
        inst._model = rust_model
        inst.radius = 1.0
        inst.weights = "uniform"
        inst.metric = "minkowski"
        inst.p = 2.0
        inst.device = rust_model.device_ or target_dev
        return inst
