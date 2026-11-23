"""
OchreML Advanced Clustering & Outlier Detection Submodule (＾▽＾)
Provides Scikit-Learn-compatible DBSCAN, OPTICS, SpectralClustering,
AgglomerativeClustering, and LocalOutlierFactor backed by Rust hardware
primitives N3 (pairwise_l2_sq), N4 (topk_argmin), N5 (small_sym_eig_svd),
and KMeans (*≧ω≦*).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    AgglomerativeClustering as _RustAgglomerativeClustering,
    DBSCAN as _RustDBSCAN,
    LocalOutlierFactor as _RustLocalOutlierFactor,
    OPTICS as _RustOPTICS,
    SpectralClustering as _RustSpectralClustering,
)
from ._preproc_utils import (
    bytes_to_f64_2d,
    bytes_to_i64_1d,
    bytes_to_i64_2d,
    validate_dit_path,
)
from ._utils import process_features


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


def _bytes_to_f64_1d(raw: bytes) -> Any:
    return np.frombuffer(raw, dtype=np.float64).copy()


class _BaseAdvClusterMixin:
    """Shared persistence, device, and pickle helpers for advanced clustering estimators."""

    _model: Any
    device: str

    def _sync_params_from_rust(self) -> None:
        raise NotImplementedError

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        """Save the estimator to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    def to(self, device: str) -> Any:
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()


class DBSCAN(_BaseAdvClusterMixin):
    """
    Density-Based Spatial Clustering of Applications with Noise (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`) with chunked distance evaluation.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        The metric to use when calculating distance between instances.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        device: Optional[str] = None,
    ):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = str(metric)
        self.device = _resolve_device(device)
        self._model = _RustDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.eps = float(self._model.eps)
        self.min_samples = int(self._model.min_samples)
        self.metric = str(self._model.metric)
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(cls, rust_obj: _RustDBSCAN, device: str) -> "DBSCAN":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DBSCAN":
        """Load DBSCAN from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustDBSCAN.load(validate_dit_path(path), target_dev), target_dev
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustDBSCAN.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "DBSCAN":
        """
        Perform DBSCAN clustering from features (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : DBSCAN
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Compute clusters from a data matrix and predict labels (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        return bytes_to_i64_1d(self._model.fit_predict_buffer(_prep_x(X)))

    @property
    def labels_(self) -> Optional[Any]:
        raw = self._model.labels_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def core_sample_indices_(self) -> Optional[Any]:
        raw = self._model.core_sample_indices_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def components_(self) -> Optional[Any]:
        res = self._model.components_bytes
        if res is None:
            n_feat = self.n_features_in_
            if n_feat is not None:
                return np.empty((0, n_feat), dtype=np.float64)
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)


class OPTICS(_BaseAdvClusterMixin):
    """
    Ordering Points To Identify the Clustering Structure (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`) and N4 (`topk_argmin`) with xi extraction.

    Parameters
    ----------
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    max_eps : float, default=inf
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        Metric to use for distance computation.
    xi : float, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary.
    min_cluster_size : int or None, default=None
        Minimum number of samples in an OPTICS cluster.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        min_samples: int = 5,
        max_eps: float = math.inf,
        metric: str = "euclidean",
        xi: float = 0.05,
        min_cluster_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.min_samples = int(min_samples)
        self.max_eps = float(max_eps)
        self.metric = str(metric)
        self.xi = float(xi)
        self.min_cluster_size = (
            int(min_cluster_size) if min_cluster_size is not None else None
        )
        self.device = _resolve_device(device)
        self._model = _RustOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric=self.metric,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.min_samples = int(self._model.min_samples)
        self.max_eps = float(self._model.max_eps)
        self.metric = str(self._model.metric)
        self.xi = float(self._model.xi)
        self.min_cluster_size = self._model.min_cluster_size
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(cls, rust_obj: _RustOPTICS, device: str) -> "OPTICS":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "OPTICS":
        """Load OPTICS from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustOPTICS.load(validate_dit_path(path), target_dev), target_dev
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustOPTICS.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "OPTICS":
        """
        Perform OPTICS clustering (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : OPTICS
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Perform OPTICS clustering and return cluster labels (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        return bytes_to_i64_1d(self._model.fit_predict_buffer(_prep_x(X)))

    @property
    def labels_(self) -> Optional[Any]:
        raw = self._model.labels_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def reachability_(self) -> Optional[Any]:
        raw = self._model.reachability_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @property
    def core_distances_(self) -> Optional[Any]:
        raw = self._model.core_distances_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @property
    def ordering_(self) -> Optional[Any]:
        raw = self._model.ordering_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)


class SpectralClustering(_BaseAdvClusterMixin):
    """
    Spectral Clustering (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`), N4 (`topk_argmin`), N5 (`small_sym_eig_svd`),
    and KMeans on normalized Laplacian eigenvectors.

    Parameters
    ----------
    n_clusters : int, default=8
        The dimension of the projection subspace and number of clusters.
    affinity : {'rbf', 'nearest_neighbors'}, default='rbf'
        How to construct the affinity matrix.
    gamma : float, default=1.0
        Kernel coefficient for rbf kernel.
    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    random_state : int or None, default=None
        Determines random number generation for k-means centroid initialization.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        n_init: int = 10,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.n_clusters = int(n_clusters)
        self.affinity = str(affinity)
        self.gamma = float(gamma)
        self.n_neighbors = int(n_neighbors)
        self.n_init = int(n_init)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustSpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            n_init=self.n_init,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_clusters = int(self._model.n_clusters)
        self.affinity = str(self._model.affinity)
        self.gamma = float(self._model.gamma)
        self.n_neighbors = int(self._model.n_neighbors)
        self.n_init = int(self._model.n_init)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustSpectralClustering, device: str
    ) -> "SpectralClustering":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SpectralClustering":
        """Load SpectralClustering from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustSpectralClustering.load(validate_dit_path(path), target_dev),
            target_dev,
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustSpectralClustering.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "SpectralClustering":
        """
        Perform spectral clustering from features (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : SpectralClustering
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Compute spectral clustering and return cluster labels (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return bytes_to_i64_1d(self._model.fit_predict_buffer(_prep_x(X)))

    @property
    def labels_(self) -> Optional[Any]:
        raw = self._model.labels_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def affinity_matrix_(self) -> Optional[Any]:
        res = self._model.affinity_matrix_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)


class AgglomerativeClustering(_BaseAdvClusterMixin):
    """
    Agglomerative Hierarchical Clustering (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`) for initial distances with Lance-Williams linkage.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        Metric used to compute the linkage. If linkage is 'ward', only 'euclidean' is accepted.
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: str = "euclidean",
        linkage: str = "ward",
        device: Optional[str] = None,
    ):
        self.n_clusters = int(n_clusters)
        self.metric = str(metric)
        self.linkage = str(linkage)
        self.device = _resolve_device(device)
        self._model = _RustAgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_clusters = int(self._model.n_clusters)
        self.metric = str(self._model.metric)
        self.linkage = str(self._model.linkage)
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustAgglomerativeClustering, device: str
    ) -> "AgglomerativeClustering":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(
        cls, path: str, device: Optional[str] = None
    ) -> "AgglomerativeClustering":
        """Load AgglomerativeClustering from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustAgglomerativeClustering.load(validate_dit_path(path), target_dev),
            target_dev,
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustAgglomerativeClustering.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "AgglomerativeClustering":
        """
        Fit the hierarchical clustering from features (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : AgglomerativeClustering
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Fit and return the result of each sample's clustering assignment (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return bytes_to_i64_1d(self._model.fit_predict_buffer(_prep_x(X)))

    @property
    def labels_(self) -> Optional[Any]:
        raw = self._model.labels_bytes
        if raw is None:
            return None
        return bytes_to_i64_1d(raw)

    @property
    def n_leaves_(self) -> int:
        return int(self._model.n_leaves_)

    @property
    def n_connected_components_(self) -> int:
        return int(self._model.n_connected_components_)

    @property
    def children_(self) -> Optional[Any]:
        res = self._model.children_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_i64_2d(rows, cols, raw)


class LocalOutlierFactor(_BaseAdvClusterMixin):
    """
    Unsupervised Outlier Detection using the Local Outlier Factor (LOF) (＾▽＾)
    Uses Rust Backend N3 (`pairwise_l2_sq`) and N4 (`topk_argmin`).

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for kneighbors queries.
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. When 'auto', the threshold is set to -1.5.
    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty=True to use predict,
        decision_function, and score_samples on new unseen data.
    metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        Metric to use for distance computation.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: Union[str, float, None] = "auto",
        novelty: bool = False,
        metric: str = "euclidean",
        device: Optional[str] = None,
    ):
        if contamination is None or contamination == "auto":
            cont_val: Optional[float] = None
            self.contamination: Union[str, float] = "auto"
        elif isinstance(contamination, str):
            raise ValueError(
                f"contamination must be 'auto' or a float in (0.0, 0.5], got '{contamination}' ( >_< )"
            )
        else:
            cont_val = float(contamination)
            self.contamination = cont_val

        self.n_neighbors = int(n_neighbors)
        self.novelty = bool(novelty)
        self.metric = str(metric)
        self.device = _resolve_device(device)
        self._model = _RustLocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=cont_val,
            novelty=self.novelty,
            metric=self.metric,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_neighbors = int(self._model.n_neighbors)
        c = self._model.contamination
        self.contamination = "auto" if c is None else float(c)
        self.novelty = bool(self._model.novelty)
        self.metric = str(self._model.metric)
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(
        cls, rust_obj: _RustLocalOutlierFactor, device: str
    ) -> "LocalOutlierFactor":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LocalOutlierFactor":
        """Load LocalOutlierFactor from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustLocalOutlierFactor.load(validate_dit_path(path), target_dev),
            target_dev,
        )

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLocalOutlierFactor.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "LocalOutlierFactor":
        """
        Fit the local outlier factor detector from the training dataset (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : LocalOutlierFactor
            Fitted estimator.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Fit the model to the training set X and return the labels (o´∀｀o).
        Only available when novelty=False.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        return bytes_to_i64_1d(self._model.fit_predict_buffer(_prep_x(X)))

    def predict(self, X: Any) -> Any:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to LOF (＾▽＾).
        Only available when novelty=True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor wrt training samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        return bytes_to_i64_1d(self._model.predict_buffer(_prep_x(X)))

    def decision_function(self, X: Any) -> Any:
        """
        Shifted opposite of the Local Outlier Factor of X (*≧ω≦*).
        Only available when novelty=True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        shifted_opposite_lof_scores : ndarray of shape (n_samples,)
            Negative values are outliers, non-negative values are inliers.
        """
        return _bytes_to_f64_1d(self._model.decision_function_buffer(_prep_x(X)))

    def score_samples(self, X: Any) -> Any:
        """
        Opposite of the Local Outlier Factor of X (o´∀｀o).
        Only available when novelty=True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        opposite_lof_scores : ndarray of shape (n_samples,)
            Opposite of the LOF of each input sample (lower means more abnormal).
        """
        return _bytes_to_f64_1d(self._model.score_samples_buffer(_prep_x(X)))

    @property
    def negative_outlier_factor_(self) -> Optional[Any]:
        raw = self._model.negative_outlier_factor_bytes
        if raw is None:
            return None
        return _bytes_to_f64_1d(raw)

    @property
    def offset_(self) -> float:
        return float(self._model.offset_)

    @property
    def n_neighbors_(self) -> int:
        return int(self._model.n_neighbors_)


__all__ = [
    "DBSCAN",
    "OPTICS",
    "SpectralClustering",
    "AgglomerativeClustering",
    "LocalOutlierFactor",
]
