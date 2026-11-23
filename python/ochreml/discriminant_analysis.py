"""
OchreML Discriminant Analysis Module (＾▽＾)
Provides Scikit-Learn-compatible LinearDiscriminantAnalysis and QuadraticDiscriminantAnalysis (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    LinearDiscriminantAnalysis as _RustLDA,
    QuadraticDiscriminantAnalysis as _RustQDA,
)
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, process_target


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


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis (LDA) classifier and dimensionality reducer (＾▽＾).

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use:
        - 'svd': Singular value decomposition (recommended for large feature counts).
        - 'lsqr': Least squares solution, can be combined with shrinkage.
        - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.
    shrinkage : 'auto', float, or None, default=None
        Shrinkage parameter, possible values:
        - None: no shrinkage.
        - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
        - float between 0 and 1: fixed shrinkage parameter.
    priors : array-like of shape (n_classes,) or None, default=None
        The class prior probabilities. By default, the class proportions are inferred.
    n_components : int or None, default=None
        Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction.
    tol : float, default=1e-4
        Absolute threshold for a singular value to be considered significant.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustLDA

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: Optional[Union[str, float]] = None,
        priors: Optional[Any] = None,
        n_components: Optional[int] = None,
        tol: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.tol = tol
        self.device = _resolve_device(device)

        shrinkage_val: Optional[float] = None
        shrinkage_auto = False
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                shrinkage_auto = True
            else:
                raise ValueError(
                    f"Invalid shrinkage string '{shrinkage}'; expected 'auto', float, or None (・`ω´・)"
                )
        elif shrinkage is not None:
            shrinkage_val = float(shrinkage)

        priors_list = [float(p) for p in priors] if priors is not None else None
        self._model = _RustLDA(
            solver=solver,
            shrinkage=shrinkage_val,
            shrinkage_auto=shrinkage_auto,
            priors=priors_list,
            n_components=n_components,
            tol=float(tol),
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustLDA, device: str) -> "LinearDiscriminantAnalysis":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.solver = rust_obj.solver
        if rust_obj.shrinkage_auto:
            inst.shrinkage = "auto"
        else:
            inst.shrinkage = rust_obj.shrinkage
        inst.priors = list(rust_obj.priors_) if rust_obj.priors_ is not None else None
        inst.n_components = rust_obj.n_components
        inst.tol = rust_obj.tol
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "LinearDiscriminantAnalysis":
        """
        Fit the Linear Discriminant Analysis model (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : LinearDiscriminantAnalysis
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        self._model.fit_buffer(X_data, y_data)
        return self

    def decision_function(self, X: Any) -> Any:
        """
        Predict confidence scores for samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : numpy.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per (n_samples, n_classes) combination (1D for binary).
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.decision_function_buffer(X_data)
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    def predict_proba(self, X: Any) -> Any:
        """
        Estimate probability (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict_log_proba(self, X: Any) -> Any:
        """
        Estimate log probability (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_log_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict(self, X: Any) -> Any:
        """
        Predict class labels for samples in X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_data = _to_f64_2d(X)
        raw = self._model.predict_buffer(X_data)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def transform(self, X: Any) -> Any:
        """
        Project data to maximize class separation (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : numpy.ndarray of shape (n_samples, n_components_)
            Transformed data.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Any) -> Any:
        """
        Fit to data, then transform it (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        X_new : numpy.ndarray of shape (n_samples, n_components_)
            Transformed data.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        rows, cols, raw = self._model.fit_transform_buffer(X_data, y_data)
        return bytes_to_f64_2d(rows, cols, raw)

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
        """Save fitted LDA model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LinearDiscriminantAnalysis":
        """Load LDA model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "LinearDiscriminantAnalysis":
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustLDA.from_bytes(state, target_dev)
        self.solver = self._model.solver
        self.shrinkage = "auto" if self._model.shrinkage_auto else self._model.shrinkage
        self.priors = list(self._model.priors_) if self._model.priors_ is not None else None
        self.n_components = self._model.n_components
        self.tol = self._model.tol
        self.device = self._model.device_ or target_dev

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return np.asarray(c, dtype=np.float64) if c is not None else None

    @property
    def priors_(self) -> Optional[Any]:
        p = self._model.priors_
        return np.asarray(p, dtype=np.float64) if p is not None else None

    @property
    def means_(self) -> Optional[Any]:
        res = self._model.means_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def xbar_(self) -> Optional[Any]:
        xb = self._model.xbar_
        return np.asarray(xb, dtype=np.float64) if xb is not None else None

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
    def scalings_(self) -> Optional[Any]:
        res = self._model.scalings_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def explained_variance_ratio_(self) -> Optional[Any]:
        evr = self._model.explained_variance_ratio_
        return np.asarray(evr, dtype=np.float64) if evr is not None else None

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def n_components_(self) -> Optional[int]:
        return self._model.n_components_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class QuadraticDiscriminantAnalysis:
    """
    Quadratic Discriminant Analysis (QDA) classifier (*≧ω≦*).

    A classifier with a quadratic decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    Parameters
    ----------
    priors : array-like of shape (n_classes,) or None, default=None
        Class priors. By default, the class proportions are inferred from the training data.
    reg_param : float, default=0.0
        Regularizes the per-class covariance estimates by transforming
        ``S2 = (1 - reg_param) * Sigma + reg_param * I``.
    tol : float, default=1e-4
        Absolute threshold for the eigenvalues of the covariance matrices.
    store_covariance : bool, default=False
        If True, the class covariance matrices are explicitly computed and stored
        in the ``covariance_`` attribute.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustQDA

    def __init__(
        self,
        priors: Optional[Any] = None,
        reg_param: float = 0.0,
        tol: float = 1e-4,
        store_covariance: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.priors = priors
        self.reg_param = float(reg_param)
        self.tol = float(tol)
        self.store_covariance = bool(store_covariance)
        self.device = _resolve_device(device)

        priors_list = [float(p) for p in priors] if priors is not None else None
        self._model = _RustQDA(
            priors=priors_list,
            reg_param=self.reg_param,
            tol=self.tol,
            store_covariance=self.store_covariance,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustQDA, device: str) -> "QuadraticDiscriminantAnalysis":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.priors = list(rust_obj.priors_) if rust_obj.priors_ is not None else None
        inst.reg_param = rust_obj.reg_param
        inst.tol = rust_obj.tol
        inst.store_covariance = rust_obj.store_covariance
        inst.device = rust_obj.device_ or device
        return inst

    def fit(self, X: Any, y: Any) -> "QuadraticDiscriminantAnalysis":
        """
        Fit the Quadratic Discriminant Analysis model (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : QuadraticDiscriminantAnalysis
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        self._model.fit_buffer(X_data, y_data)
        return self

    def decision_function(self, X: Any) -> Any:
        """
        Apply decision function to an array of samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values (1D for binary classification).
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.decision_function_buffer(X_data)
        arr = bytes_to_f64_2d(rows, cols, raw)
        if cols == 1:
            return arr.ravel()
        return arr

    def predict_proba(self, X: Any) -> Any:
        """
        Return posterior probabilities of classification (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Posterior probabilities of classification per class.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict_log_proba(self, X: Any) -> Any:
        """
        Return log of posterior probabilities of classification (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Posterior log-probabilities of classification per class.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_log_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict(self, X: Any) -> Any:
        """
        Perform classification on an array of test vectors X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples,)
            Predicted class labels.
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
        """Save fitted QDA model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "QuadraticDiscriminantAnalysis":
        """Load QDA model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "QuadraticDiscriminantAnalysis":
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustQDA.from_bytes(state, target_dev)
        self.priors = list(self._model.priors_) if self._model.priors_ is not None else None
        self.reg_param = self._model.reg_param
        self.tol = self._model.tol
        self.store_covariance = self._model.store_covariance
        self.device = self._model.device_ or target_dev

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return np.asarray(c, dtype=np.float64) if c is not None else None

    @property
    def priors_(self) -> Optional[Any]:
        p = self._model.priors_
        return np.asarray(p, dtype=np.float64) if p is not None else None

    @property
    def means_(self) -> Optional[Any]:
        res = self._model.means_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def scalings_(self) -> Optional[Any]:
        s = self._model.scalings_
        if s is None:
            return None
        return [np.asarray(v, dtype=np.float64) for v in s]

    @property
    def rotations_(self) -> Optional[Any]:
        res = self._model.rotations_bytes()
        if res is None:
            return None
        return [bytes_to_f64_2d(r, c, raw) for r, c, raw in res]

    @property
    def covariance_(self) -> Optional[Any]:
        res = self._model.covariance_bytes()
        if res is None:
            return None
        return [bytes_to_f64_2d(r, c, raw) for r, c, raw in res]

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_
