"""
OchreML Manifold Learning Submodule (＾▽＾)
Provides Scikit-Learn-compatible TSNE and MDS accelerated by Rust hardware backends (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    MDS as _RustMDS,
    TSNE as _RustTSNE,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, to_feature_matrix


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_array(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    if np is not None:
        return np.ascontiguousarray(X_data, dtype=np.float64)
    return to_feature_matrix(X)


class TSNE:
    """
    t-distributed Stochastic Neighbor Embedding (TSNE) (＾▽＾)

    Nonlinear dimensionality reduction that converts pairwise Euclidean distances
    into conditional probabilities and minimizes Kullback-Leibler divergence
    using exact gradient descent with momentum and adaptive gains (*≧ω≦*).

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Related to the number of nearest neighbors used in other manifold
        learning algorithms. Must be strictly less than ``n_samples``.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in the
        embedded space during the first 250 iterations.
    learning_rate : float or {"auto"}, default="auto"
        Learning rate for t-SNE. When ``"auto"``, it is set to
        ``max(n_samples / early_exaggeration / 4.0, 50.0)``.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    init : {"pca", "random"}, default="pca"
        Initialization of embedding.
    random_state : int or None, default=None
        Determines the random number generator for reproducible initialization.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    learning_rate_ : float
        Effective learning rate used during optimization.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustTSNE

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[str, float] = "auto",
        max_iter: int = 1000,
        init: str = "pca",
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components ({n_components}) must be an integer >= 1 ( >_< )")

        lr_opt: Optional[float]
        if isinstance(learning_rate, str):
            if learning_rate.lower() != "auto":
                raise ValueError(
                    f"learning_rate '{learning_rate}' is invalid; expected 'auto' or positive float ( >_< )"
                )
            lr_opt = None
        elif isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate must be 'auto' or float, got {type(learning_rate)} (・`ω´・)"
            )
        else:
            lr_opt = float(learning_rate)

        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.early_exaggeration = float(early_exaggeration)
        self.learning_rate = learning_rate
        self.max_iter = int(max_iter)
        self.init = str(init)
        self.random_state = random_state
        self.device = _resolve_device(device)

        self._model = _RustTSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            learning_rate=lr_opt,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustTSNE, device: str) -> "TSNE":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.perplexity = rust_obj.perplexity
        inst.early_exaggeration = rust_obj.early_exaggeration
        inst.learning_rate = "auto" if rust_obj.learning_rate is None else rust_obj.learning_rate
        inst.max_iter = rust_obj.max_iter
        inst.init = rust_obj.init
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Optional[Any] = None) -> "TSNE":
        """Fit X into an embedded space (o´∀｀o)."""
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit X into an embedded space and return that transformed output (＾▽＾)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def save(self, path: str) -> None:
        """Save fitted TSNE model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TSNE":
        """Load TSNE model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "TSNE":
        """Move TSNE execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustTSNE.from_bytes(state, target_dev)
        self._model = rust_obj
        self.n_components = rust_obj.n_components
        self.perplexity = rust_obj.perplexity
        self.early_exaggeration = rust_obj.early_exaggeration
        self.learning_rate = "auto" if rust_obj.learning_rate is None else rust_obj.learning_rate
        self.max_iter = rust_obj.max_iter
        self.init = rust_obj.init
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or target_dev

    @property
    def embedding_(self) -> Any:
        res = self._model.embedding_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def kl_divergence_(self) -> Optional[float]:
        return self._model.kl_divergence_

    @property
    def n_iter_(self) -> Optional[int]:
        return self._model.n_iter_

    @property
    def learning_rate_(self) -> Optional[float]:
        return self._model.learning_rate_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class MDS:
    """
    Multidimensional Scaling (MDS) (＾▽＾)

    Finds a low-dimensional embedding that preserves pairwise dissimilarities
    using SMACOF majorization, supporting both metric and non-metric
    (Kruskal isotonic regression) scaling (*≧ω≦*).

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    metric : bool, default=True
        If True, perform metric MDS; otherwise, perform non-metric MDS using
        Kruskal isotonic regression.
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare convergence.
    dissimilarity : {"euclidean", "precomputed"}, default="euclidean"
        Dissimilarity measure to use.
    random_state : int or None, default=None
        Determines the random number generator used to initialize the centers.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.
    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points.
    n_iter_ : int
        The number of iterations corresponding to the best stress.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _rust_cls = _RustMDS

    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        eps: float = 1e-3,
        dissimilarity: str = "euclidean",
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components ({n_components}) must be an integer >= 1 ( >_< )")

        self.n_components = int(n_components)
        self.metric = bool(metric)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.dissimilarity = str(dissimilarity)
        self.random_state = random_state
        self.device = _resolve_device(device)

        self._model = _RustMDS(
            n_components=self.n_components,
            metric=self.metric,
            n_init=self.n_init,
            max_iter=self.max_iter,
            eps=self.eps,
            dissimilarity=self.dissimilarity,
            random_state=self.random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustMDS, device: str) -> "MDS":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.n_components = rust_obj.n_components
        inst.metric = rust_obj.metric
        inst.n_init = rust_obj.n_init
        inst.max_iter = rust_obj.max_iter
        inst.eps = rust_obj.eps
        inst.dissimilarity = rust_obj.dissimilarity
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Optional[Any] = None) -> "MDS":
        """Compute the position of the points in the embedding space (o´∀｀o)."""
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit the data from X, and returns the embedded coordinates (＾▽＾)."""
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def save(self, path: str) -> None:
        """Save fitted MDS model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MDS":
        """Load MDS model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "MDS":
        """Move MDS execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustMDS.from_bytes(state, target_dev)
        self._model = rust_obj
        self.n_components = rust_obj.n_components
        self.metric = rust_obj.metric
        self.n_init = rust_obj.n_init
        self.max_iter = rust_obj.max_iter
        self.eps = rust_obj.eps
        self.dissimilarity = rust_obj.dissimilarity
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or target_dev

    @property
    def embedding_(self) -> Any:
        res = self._model.embedding_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def stress_(self) -> Optional[float]:
        return self._model.stress_

    @property
    def dissimilarity_matrix_(self) -> Any:
        res = self._model.dissimilarity_matrix_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def n_iter_(self) -> Optional[int]:
        return self._model.n_iter_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


__all__ = ["TSNE", "MDS"]
