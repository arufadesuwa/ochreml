"""
OchreML Clustering Submodule (＾▽＾)
Provides Scikit-Learn-compatible KMeans and MiniBatchKMeans backed by Rust
hardware primitives N3 (pairwise_l2_sq) and N4 (row_argmin) (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    KMeans as _RustKMeans,
    MiniBatchKMeans as _RustMiniBatchKMeans,
)
from ._preproc_utils import bytes_to_f64_2d, bytes_to_i64_1d, validate_dit_path
from ._utils import process_features, process_sample_weight


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


def _prep_sw(sample_weight: Optional[Any]) -> Optional[Any]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if is_buf:
        return sw_data
    return np.asarray(sw_data, dtype=np.float64)


class _BaseClusterMixin:
    """Shared buffer conversion, persistence, and device helpers for clustering estimators."""

    _model: Any
    device: str

    def _sync_params_from_rust(self) -> None:
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        """
        Predict the closest cluster each sample in X belongs to (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return bytes_to_i64_1d(raw)

    def transform(self, X: Any) -> Any:
        """
        Transform X to a cluster-distance space (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Euclidean distance from each sample to each cluster center.
        """
        rows, cols, raw = self._model.transform_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def score(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """
        Opposite of the value of X on the K-means objective (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        score : float
            Negative inertia (sum of squared distances to closest cluster center).
        """
        return float(self._model.score_buffer(_prep_x(X), _prep_sw(sample_weight)))

    @property
    def cluster_centers_(self) -> Optional[Any]:
        res = self._model.cluster_centers_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def labels_(self) -> Optional[Any]:
        raw = self._model.labels_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def inertia_(self) -> Optional[float]:
        return self._model.inertia_

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save the fitted clustering model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    def to(self, device: str) -> Any:
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()


class KMeans(_BaseClusterMixin):
    """
    K-Means clustering (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`) and N4 (`row_argmin`) with k-means++ initialization.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.
    n_init : int or 'auto', default=10
        Number of time the k-means algorithm is run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the
        cluster centers of two consecutive iterations to declare convergence.
    random_state : int or None, default=None
        Determines random number generation for centroid initialization.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    _rust_cls = _RustKMeans

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: Union[int, str] = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        resolved_n_init = 10 if n_init == "auto" else int(n_init)
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = resolved_n_init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_clusters = int(self._model.n_clusters)
        self.init = str(self._model.init)
        self.n_init = int(self._model.n_init)
        self.max_iter = int(self._model.max_iter)
        self.tol = float(self._model.tol)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(cls, rust_obj: _RustKMeans, device: str) -> "KMeans":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KMeans":
        """Load KMeans from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustKMeans.load(validate_dit_path(path), target_dev), target_dev
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustKMeans.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "KMeans":
        """
        Compute k-means clustering (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X), _prep_sw(sample_weight))
        return self

    def fit_predict(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """
        Compute cluster centers and predict cluster index for each sample (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        raw = self._model.fit_predict_buffer(_prep_x(X), _prep_sw(sample_weight))
        return bytes_to_i64_1d(raw)

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """
        Compute clustering and transform X to cluster-distance space (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        rows, cols, raw = self._model.fit_transform_buffer(
            _prep_x(X), _prep_sw(sample_weight)
        )
        return bytes_to_f64_2d(rows, cols, raw)


class MiniBatchKMeans(_BaseClusterMixin):
    """
    Mini-Batch K-Means clustering (*≧ω≦*)
    Uses Rust Backend N3 (`pairwise_l2_sq`) and N4 (`row_argmin`) with online SGD center updates.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.
    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before stopping.
    batch_size : int, default=1024
        Size of the mini batches.
    tol : float, default=0.0
        Control early stopping based on the relative center changes / EWMA inertia.
    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches that
        do not yield an improvement on the smoothed inertia.
    n_init : int or 'auto', default=3
        Number of random initializations that are tried.
    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to be reassigned.
    random_state : int or None, default=None
        Determines random number generation for centroid initialization and mini-batch sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    _rust_cls = _RustMiniBatchKMeans

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        max_iter: int = 100,
        batch_size: int = 1024,
        tol: float = 0.0,
        max_no_improvement: int = 10,
        n_init: Union[int, str] = 3,
        reassignment_ratio: float = 0.01,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        resolved_n_init = 3 if n_init == "auto" else int(n_init)
        self.n_clusters = int(n_clusters)
        self.init = init
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.tol = float(tol)
        self.max_no_improvement = int(max_no_improvement)
        self.n_init = resolved_n_init
        self.reassignment_ratio = float(reassignment_ratio)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustMiniBatchKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            tol=self.tol,
            max_no_improvement=self.max_no_improvement,
            n_init=self.n_init,
            reassignment_ratio=self.reassignment_ratio,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_clusters = int(self._model.n_clusters)
        self.init = str(self._model.init)
        self.max_iter = int(self._model.max_iter)
        self.batch_size = int(self._model.batch_size)
        self.tol = float(self._model.tol)
        self.max_no_improvement = int(self._model.max_no_improvement)
        self.n_init = int(self._model.n_init)
        self.reassignment_ratio = float(self._model.reassignment_ratio)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustMiniBatchKMeans, device: str
    ) -> "MiniBatchKMeans":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MiniBatchKMeans":
        """Load MiniBatchKMeans from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustMiniBatchKMeans.load(validate_dit_path(path), target_dev), target_dev
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustMiniBatchKMeans.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    @property
    def n_steps_(self) -> int:
        return int(self._model.n_steps_)

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "MiniBatchKMeans":
        """
        Compute the centroids on X by chunking it into mini-batches (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        self : MiniBatchKMeans
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X), _prep_sw(sample_weight))
        return self

    def partial_fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "MiniBatchKMeans":
        """
        Update k means estimate on a single mini-batch X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Coordinates of the data points to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X.

        Returns
        -------
        self : MiniBatchKMeans
            Updated estimator.
        """
        self._model.partial_fit_buffer(_prep_x(X), _prep_sw(sample_weight))
        return self

    def fit_predict(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """
        Compute cluster centers and predict cluster index for each sample (＾▽＾).
        """
        raw = self._model.fit_predict_buffer(_prep_x(X), _prep_sw(sample_weight))
        return bytes_to_i64_1d(raw)

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """
        Compute clustering and transform X to cluster-distance space (*≧ω≦*).
        """
        rows, cols, raw = self._model.fit_transform_buffer(
            _prep_x(X), _prep_sw(sample_weight)
        )
        return bytes_to_f64_2d(rows, cols, raw)


__all__ = ["KMeans", "MiniBatchKMeans"]

