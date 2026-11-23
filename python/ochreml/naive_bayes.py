"""
OchreML Naive Bayes Module (＾▽＾)
Provides Scikit-Learn-compatible GaussianNB, MultinomialNB, BernoulliNB, and ComplementNB (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import (
    BernoulliNB as _RustBernoulliNB,
    ComplementNB as _RustComplementNB,
    GaussianNB as _RustGaussianNB,
    MultinomialNB as _RustMultinomialNB,
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


def _to_opt_f64_1d(v: Optional[Any]) -> Optional[Any]:
    if v is None:
        return None
    return _to_f64_1d(v)


def _to_opt_sample_weight(sample_weight: Optional[Any]) -> Optional[Any]:
    is_buf, sw_data = process_sample_weight(sample_weight)
    if sw_data is None:
        return None
    if not is_buf:
        sw_data = np.asarray(sw_data, dtype=np.float64)
    return sw_data


class _BaseNaiveBayes:
    """Shared prediction, persistence, and device management mixin for Naive Bayes estimators."""

    _rust_cls: Any = None
    _model: Any
    device: str

    def predict_joint_log_likelihood(self, X: Any) -> Any:
        """
        Compute the unnormalized posterior log probability of X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        jll : numpy.ndarray of shape (n_samples, n_classes)
            Joint log-likelihood values for each sample and class.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_joint_log_likelihood_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict_log_proba(self, X: Any) -> Any:
        """
        Return log-probability estimates for the test vector X (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Log-probability of the samples for each class in the model.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_log_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict_proba(self, X: Any) -> Any:
        """
        Return probability estimates for the test vector X (＾▽＾).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples, n_classes)
            Probability of the samples for each class in the model.
        """
        X_data = _to_f64_2d(X)
        rows, cols, raw = self._model.predict_proba_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def predict(self, X: Any) -> Any:
        """
        Perform classification on an array of test vectors X (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        C : numpy.ndarray of shape (n_samples,)
            Predicted target values for X.
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
        """Save fitted Naive Bayes model to a binary '.dit' file (＾▽＾)."""
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load Naive Bayes model from a binary '.dit' file (*≧ω≦*)."""
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> Any:
        """Move estimator execution target to a new hardware device (o´∀｀o)."""
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    @property
    def classes_(self) -> Optional[Any]:
        c = self._model.classes_
        return np.asarray(c, dtype=np.float64) if c is not None else None

    @property
    def class_count_(self) -> Optional[Any]:
        cc = self._model.class_count_
        return np.asarray(cc, dtype=np.float64) if cc is not None else None

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


class GaussianNB(_BaseNaiveBayes):
    """
    Gaussian Naive Bayes (GaussianNB) classifier (＾▽＾).

    Can perform online updates to model parameters via ``partial_fit``.

    Parameters
    ----------
    priors : array-like of shape (n_classes,) or None, default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustGaussianNB

    def __init__(
        self,
        priors: Optional[Any] = None,
        var_smoothing: float = 1e-9,
        device: Optional[str] = None,
    ) -> None:
        self.priors = priors
        self.var_smoothing = float(var_smoothing)
        self.device = _resolve_device(device)
        priors_list = [float(p) for p in priors] if priors is not None else None
        self._model = _RustGaussianNB(
            priors=priors_list,
            var_smoothing=self.var_smoothing,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustGaussianNB, device: str) -> "GaussianNB":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.priors = list(rust_obj.priors) if rust_obj.priors is not None else None
        inst.var_smoothing = rust_obj.var_smoothing
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustGaussianNB.from_bytes(state, target_dev)
        self._model = rust_obj
        self.priors = list(rust_obj.priors) if rust_obj.priors is not None else None
        self.var_smoothing = rust_obj.var_smoothing
        self.device = rust_obj.device_ or target_dev

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "GaussianNB":
        """
        Fit Gaussian Naive Bayes according to X, y (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : GaussianNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        classes: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "GaussianNB":
        """
        Incremental fit on a batch of samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like of shape (n_classes,) or None, default=None
            List of all the classes that can possibly appear in the y vector.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : GaussianNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        cls_data = _to_opt_f64_1d(classes)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.partial_fit_buffer(X_data, y_data, cls_data, sw_data)
        return self

    @property
    def class_prior_(self) -> Optional[Any]:
        cp = self._model.class_prior_
        return np.asarray(cp, dtype=np.float64) if cp is not None else None

    @property
    def theta_(self) -> Optional[Any]:
        res = self._model.theta_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def var_(self) -> Optional[Any]:
        res = self._model.var_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def epsilon_(self) -> float:
        return float(self._model.epsilon_)


class MultinomialNB(_BaseNaiveBayes):
    """
    Naive Bayes classifier for multinomial models (＾▽＾).

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
    class_prior : array-like of shape (n_classes,) or None, default=None
        Prior probabilities of the classes.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustMultinomialNB

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.class_prior = class_prior
        self.device = _resolve_device(device)
        cp_list = [float(p) for p in class_prior] if class_prior is not None else None
        self._model = _RustMultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=cp_list,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustMultinomialNB, device: str) -> "MultinomialNB":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.alpha = rust_obj.alpha
        inst.fit_prior = rust_obj.fit_prior
        inst.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustMultinomialNB.from_bytes(state, target_dev)
        self._model = rust_obj
        self.alpha = rust_obj.alpha
        self.fit_prior = rust_obj.fit_prior
        self.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        self.device = rust_obj.device_ or target_dev

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "MultinomialNB":
        """
        Fit Multinomial Naive Bayes classifier according to X, y (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors (non-negative).
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : MultinomialNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        classes: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "MultinomialNB":
        """
        Incremental fit on a batch of samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors (non-negative).
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like of shape (n_classes,) or None, default=None
            List of all the classes that can possibly appear in the y vector.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : MultinomialNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        cls_data = _to_opt_f64_1d(classes)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.partial_fit_buffer(X_data, y_data, cls_data, sw_data)
        return self

    @property
    def class_log_prior_(self) -> Optional[Any]:
        clp = self._model.class_log_prior_
        return np.asarray(clp, dtype=np.float64) if clp is not None else None

    @property
    def feature_count_(self) -> Optional[Any]:
        res = self._model.feature_count_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def feature_log_prob_(self) -> Optional[Any]:
        res = self._model.feature_log_prob_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)


class BernoulliNB(_BaseNaiveBayes):
    """
    Naive Bayes classifier for multivariate Bernoulli models (＾▽＾).

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
    class_prior : array-like of shape (n_classes,) or None, default=None
        Prior probabilities of the classes.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustBernoulliNB

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.binarize = float(binarize) if binarize is not None else None
        self.fit_prior = bool(fit_prior)
        self.class_prior = class_prior
        self.device = _resolve_device(device)
        cp_list = [float(p) for p in class_prior] if class_prior is not None else None
        self._model = _RustBernoulliNB(
            alpha=self.alpha,
            binarize=self.binarize,
            fit_prior=self.fit_prior,
            class_prior=cp_list,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustBernoulliNB, device: str) -> "BernoulliNB":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.alpha = rust_obj.alpha
        inst.binarize = rust_obj.binarize
        inst.fit_prior = rust_obj.fit_prior
        inst.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustBernoulliNB.from_bytes(state, target_dev)
        self._model = rust_obj
        self.alpha = rust_obj.alpha
        self.binarize = rust_obj.binarize
        self.fit_prior = rust_obj.fit_prior
        self.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        self.device = rust_obj.device_ or target_dev

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "BernoulliNB":
        """
        Fit Bernoulli Naive Bayes classifier according to X, y (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : BernoulliNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        classes: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "BernoulliNB":
        """
        Incremental fit on a batch of samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like of shape (n_classes,) or None, default=None
            List of all the classes that can possibly appear in the y vector.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : BernoulliNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        cls_data = _to_opt_f64_1d(classes)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.partial_fit_buffer(X_data, y_data, cls_data, sw_data)
        return self

    @property
    def class_log_prior_(self) -> Optional[Any]:
        clp = self._model.class_log_prior_
        return np.asarray(clp, dtype=np.float64) if clp is not None else None

    @property
    def feature_count_(self) -> Optional[Any]:
        res = self._model.feature_count_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def feature_log_prob_(self) -> Optional[Any]:
        res = self._model.feature_log_prob_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)


class ComplementNB(_BaseNaiveBayes):
    """
    Complement Naive Bayes (CNB) classifier (＾▽＾).

    Particularly suited for imbalanced data sets.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
    class_prior : array-like of shape (n_classes,) or None, default=None
        Prior probabilities of the classes.
    norm : bool, default=False
        Whether or not a second normalization of the weights is performed.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustComplementNB

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[Any] = None,
        norm: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.class_prior = class_prior
        self.norm = bool(norm)
        self.device = _resolve_device(device)
        cp_list = [float(p) for p in class_prior] if class_prior is not None else None
        self._model = _RustComplementNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=cp_list,
            norm=self.norm,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustComplementNB, device: str) -> "ComplementNB":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.alpha = rust_obj.alpha
        inst.fit_prior = rust_obj.fit_prior
        inst.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        inst.norm = rust_obj.norm
        inst.device = rust_obj.device_ or device
        return inst

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustComplementNB.from_bytes(state, target_dev)
        self._model = rust_obj
        self.alpha = rust_obj.alpha
        self.fit_prior = rust_obj.fit_prior
        self.class_prior = list(rust_obj.class_prior) if rust_obj.class_prior is not None else None
        self.norm = rust_obj.norm
        self.device = rust_obj.device_ or target_dev

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "ComplementNB":
        """
        Fit Complement Naive Bayes classifier according to X, y (*≧ω≦*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors (non-negative).
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : ComplementNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.fit_buffer(X_data, y_data, sw_data)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        classes: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "ComplementNB":
        """
        Incremental fit on a batch of samples (o´∀｀o).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors (non-negative).
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like of shape (n_classes,) or None, default=None
            List of all the classes that can possibly appear in the y vector.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights applied to individual samples.

        Returns
        -------
        self : ComplementNB
            Fitted estimator.
        """
        X_data = _to_f64_2d(X)
        y_data = _to_f64_1d(y)
        cls_data = _to_opt_f64_1d(classes)
        sw_data = _to_opt_sample_weight(sample_weight)
        self._model.partial_fit_buffer(X_data, y_data, cls_data, sw_data)
        return self

    @property
    def class_log_prior_(self) -> Optional[Any]:
        clp = self._model.class_log_prior_
        return np.asarray(clp, dtype=np.float64) if clp is not None else None

    @property
    def feature_count_(self) -> Optional[Any]:
        res = self._model.feature_count_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def feature_all_(self) -> Optional[Any]:
        fa = self._model.feature_all_
        return np.asarray(fa, dtype=np.float64) if fa is not None else None

    @property
    def feature_log_prob_(self) -> Optional[Any]:
        res = self._model.feature_log_prob_bytes()
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)
