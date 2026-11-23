"""
OchreML Gaussian Mixture Model Submodule (＾▽＾)
Provides Scikit-Learn-compatible GaussianMixture backed by Rust EM solvers,
Cholesky log-likelihoods, and Backend N3/N4 k-means++ initialization (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import GaussianMixture as _RustGaussianMixture
from ._preproc_utils import bytes_to_f64_2d, bytes_to_i64_1d, validate_dit_path
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


class GaussianMixture:
    """
    Gaussian Mixture Model (＾▽＾)
    Representation of a Gaussian mixture model probability distribution, estimated
    using the Expectation-Maximization (EM) algorithm with Cholesky precision factors (*≧ω≦*).

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the lower bound
        average gain is below this threshold.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
    max_iter : int, default=100
        The number of EM iterations to perform.
    n_init : int, default=1
        The number of initializations to perform. The best results are kept.
    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the precisions.
    random_state : int or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters and to generate samples.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', etc.).
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.init_params = init_params
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._model = _RustGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            device=self.device,
        )

    def _sync_params_from_rust(self) -> None:
        self.n_components = int(self._model.n_components)
        self.covariance_type = str(self._model.covariance_type)
        self.tol = float(self._model.tol)
        self.reg_covar = float(self._model.reg_covar)
        self.max_iter = int(self._model.max_iter)
        self.n_init = int(self._model.n_init)
        self.init_params = str(self._model.init_params)
        self.random_state = self._model.random_state
        self.device = self._model.device_ or _resolve_device(None)

    @classmethod
    def _from_rust(cls, rust_obj: _RustGaussianMixture, device: str) -> "GaussianMixture":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst._sync_params_from_rust()
        inst.device = rust_obj.device_ or device
        return inst

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "GaussianMixture":
        """Load GaussianMixture from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        return cls._from_rust(
            _RustGaussianMixture.load(validate_dit_path(path), target_dev), target_dev
        )

    def save(self, path: str) -> None:
        """Save the fitted GaussianMixture model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    def to(self, device: str) -> "GaussianMixture":
        """Move estimator compute target to the specified device (*≧ω≦*)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustGaussianMixture.from_bytes(state, target_dev)
        self._sync_params_from_rust()

    def fit(self, X: Any, y: Optional[Any] = None) -> "GaussianMixture":
        """
        Estimate model parameters with the EM algorithm (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : GaussianMixture
            The fitted mixture.
        """
        self._model.fit_buffer(_prep_x(X))
        return self

    def fit_predict(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Estimate model parameters using X and predict the labels for X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
        raw = self._model.fit_predict_buffer(_prep_x(X))
        return bytes_to_i64_1d(raw)

    def predict(self, X: Any) -> Any:
        """
        Predict the labels for the data samples in X using trained model (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
        raw = self._model.predict_buffer(_prep_x(X))
        return bytes_to_i64_1d(raw)

    def predict_proba(self, X: Any) -> Any:
        """
        Evaluate the components' density for each sample (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each observation.
        """
        rows, cols, raw = self._model.predict_proba_buffer(_prep_x(X))
        return bytes_to_f64_2d(rows, cols, raw)

    def score_samples(self, X: Any) -> Any:
        """
        Compute the log-likelihood of each sample (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.

        Returns
        -------
        log_prob : ndarray of shape (n_samples,)
            Log-likelihood of each sample in X under the current model.
        """
        raw = self._model.score_samples_buffer(_prep_x(X))
        return np.frombuffer(raw, dtype=np.float64).copy()

    def score(self, X: Any, y: Optional[Any] = None) -> float:
        """
        Compute the per-sample average log-likelihood of the given data X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of X under the Gaussian mixture model.
        """
        return float(self._model.score_buffer(_prep_x(X)))

    def aic(self, X: Any) -> float:
        """
        Akaike information criterion for the current model on the input X (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        return float(self._model.aic_buffer(_prep_x(X)))

    def bic(self, X: Any) -> float:
        """
        Bayesian information criterion for the current model on the input X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return float(self._model.bic_buffer(_prep_x(X)))

    def sample(self, n_samples: int = 1) -> Tuple[Any, Any]:
        """
        Generate random samples from the fitted Gaussian distribution (＾▽＾).

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Randomly generated sample.
        y : ndarray of shape (n_samples,)
            Component labels.
        """
        rows, cols, x_raw, y_raw = self._model.sample_buffer(int(n_samples))
        return bytes_to_f64_2d(rows, cols, x_raw), bytes_to_i64_1d(y_raw)

    def _reshape_cov_like(self, raw: Optional[bytes]) -> Optional[Any]:
        if raw is None:
            return None
        p = self.n_features_in_
        if p is None:
            return None
        k = self.n_components
        arr = np.frombuffer(raw, dtype=np.float64).copy()
        if self.covariance_type == "full":
            return arr.reshape((k, p, p))
        if self.covariance_type == "tied":
            return arr.reshape((p, p))
        if self.covariance_type == "diag":
            return arr.reshape((k, p))
        return arr.reshape((k,))

    @property
    def weights_(self) -> Optional[Any]:
        raw = self._model.weights_bytes
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def means_(self) -> Optional[Any]:
        res = self._model.means_bytes
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def covariances_(self) -> Optional[Any]:
        return self._reshape_cov_like(self._model.covariances_bytes)

    @property
    def precisions_cholesky_(self) -> Optional[Any]:
        return self._reshape_cov_like(self._model.precisions_cholesky_bytes)

    @property
    def converged_(self) -> bool:
        return bool(self._model.converged_)

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)

    @property
    def lower_bound_(self) -> Optional[float]:
        return self._model.lower_bound_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


__all__ = ["GaussianMixture"]
