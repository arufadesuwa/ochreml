# OchreML Wave 3 Naive Bayes Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, partial_fit, sample_weight, .dit/pickle round-trips, and edge cases.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import (
    BernoulliNB as SkBernoulliNB,
    ComplementNB as SkComplementNB,
    GaussianNB as SkGaussianNB,
    MultinomialNB as SkMultinomialNB,
)

from ochreml.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)


def _make_count_data(seed=42, n_samples=160, n_features=12, n_classes=3):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 8, size=(n_samples, n_features)).astype(np.float64)
    y = rng.randint(0, n_classes, size=n_samples).astype(np.float64)
    w = rng.uniform(0.5, 2.0, size=n_samples)
    return X, y, w


def test_gaussian_nb_parity_and_partial_fit():
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    rng = np.random.RandomState(7)
    w = rng.uniform(0.5, 2.5, size=len(y))

    for priors in (None, [0.2, 0.5, 0.3]):
        sk = SkGaussianNB(priors=priors, var_smoothing=1e-9).fit(X, y, sample_weight=w)
        oc = GaussianNB(priors=priors, var_smoothing=1e-9, device="cpu").fit(
            X, y, sample_weight=w
        )

        np.testing.assert_allclose(oc.theta_, sk.theta_, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(oc.var_, sk.var_, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(oc.class_prior_, sk.class_prior_, atol=1e-12)
        np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
        np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-10)
        np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-10)
        assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)

    # Incremental partial_fit parity
    sk_pf = SkGaussianNB(var_smoothing=1e-9)
    oc_pf = GaussianNB(var_smoothing=1e-9, device="cpu")
    classes = np.array([0.0, 1.0, 2.0])
    sk_pf.partial_fit(X[:60], y[:60], classes=classes, sample_weight=w[:60])
    oc_pf.partial_fit(X[:60], y[:60], classes=classes, sample_weight=w[:60])
    sk_pf.partial_fit(X[60:], y[60:], sample_weight=w[60:])
    oc_pf.partial_fit(X[60:], y[60:], sample_weight=w[60:])

    np.testing.assert_allclose(oc_pf.theta_, sk_pf.theta_, atol=1e-12)
    np.testing.assert_allclose(oc_pf.var_, sk_pf.var_, atol=1e-12)
    np.testing.assert_allclose(oc_pf.predict_proba(X), sk_pf.predict_proba(X), atol=1e-10)


@pytest.mark.parametrize("fit_prior", [True, False])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_multinomial_nb_parity_and_partial_fit(fit_prior, alpha):
    X, y, w = _make_count_data(seed=11)

    sk = SkMultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(X, y, sample_weight=w)
    oc = MultinomialNB(alpha=alpha, fit_prior=fit_prior, device="cpu").fit(
        X, y, sample_weight=w
    )

    np.testing.assert_allclose(oc.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(oc.class_log_prior_, sk.class_log_prior_, atol=1e-12)
    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-10)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-10)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)

    # partial_fit
    oc_pf = MultinomialNB(alpha=alpha, fit_prior=fit_prior, device="cpu")
    classes = np.array([0.0, 1.0, 2.0])
    oc_pf.partial_fit(X[:80], y[:80], classes=classes, sample_weight=w[:80])
    oc_pf.partial_fit(X[80:], y[80:], sample_weight=w[80:])
    np.testing.assert_allclose(oc_pf.feature_log_prob_, oc.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(oc_pf.predict_proba(X), oc.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize("binarize", [0.0, 2.5, None])
def test_bernoulli_nb_parity_and_partial_fit(binarize):
    rng = np.random.RandomState(23)
    if binarize is None:
        X = rng.randint(0, 2, size=(140, 10)).astype(np.float64)
    else:
        X = rng.randn(140, 10) + 1.0
    y = rng.randint(0, 3, size=140).astype(np.float64)
    w = rng.uniform(0.5, 2.0, size=140)

    sk = SkBernoulliNB(alpha=1.0, binarize=binarize).fit(X, y, sample_weight=w)
    oc = BernoulliNB(alpha=1.0, binarize=binarize, device="cpu").fit(X, y, sample_weight=w)

    np.testing.assert_allclose(oc.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(oc.class_log_prior_, sk.class_log_prior_, atol=1e-12)
    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-10)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-10)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)

    # partial_fit
    oc_pf = BernoulliNB(alpha=1.0, binarize=binarize, device="cpu")
    oc_pf.partial_fit(X[:70], y[:70], classes=[0.0, 1.0, 2.0], sample_weight=w[:70])
    oc_pf.partial_fit(X[70:], y[70:], sample_weight=w[70:])
    np.testing.assert_allclose(oc_pf.feature_log_prob_, oc.feature_log_prob_, atol=1e-12)


@pytest.mark.parametrize("norm", [False, True])
def test_complement_nb_parity_and_partial_fit(norm):
    X, y, w = _make_count_data(seed=37)

    sk = SkComplementNB(alpha=1.0, norm=norm).fit(X, y, sample_weight=w)
    oc = ComplementNB(alpha=1.0, norm=norm, device="cpu").fit(X, y, sample_weight=w)

    np.testing.assert_allclose(oc.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-10)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-10)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)

    # partial_fit
    oc_pf = ComplementNB(alpha=1.0, norm=norm, device="cpu")
    oc_pf.partial_fit(X[:80], y[:80], classes=[0.0, 1.0, 2.0], sample_weight=w[:80])
    oc_pf.partial_fit(X[80:], y[80:], sample_weight=w[80:])
    np.testing.assert_allclose(oc_pf.feature_log_prob_, oc.feature_log_prob_, atol=1e-12)


def test_naive_bayes_save_load_pickle_and_devices(tmp_path):
    X, y, _ = _make_count_data(seed=55, n_samples=100, n_features=6, n_classes=2)

    for cls, kwargs, fname in [
        (GaussianNB, {"var_smoothing": 1e-8}, "gnb.dit"),
        (MultinomialNB, {"alpha": 1.0}, "mnb.dit"),
        (BernoulliNB, {"alpha": 1.0, "binarize": 1.5}, "bnb.dit"),
        (ComplementNB, {"alpha": 1.0, "norm": True}, "cnb.dit"),
    ]:
        for dev in ("cpu", "auto"):
            est = cls(**kwargs, device=dev).fit(X, y)
            assert est.device_ is not None

        path = tmp_path / fname
        est.save(str(path))
        loaded = cls.load(str(path), device="cpu")
        np.testing.assert_array_equal(loaded.predict(X), est.predict(X))
        np.testing.assert_allclose(loaded.predict_proba(X), est.predict_proba(X), atol=1e-12)

        loaded.to("auto")
        assert loaded.device_ is not None

        restored = pickle.loads(pickle.dumps(est))
        np.testing.assert_array_equal(restored.predict(X), est.predict(X))
        np.testing.assert_allclose(restored.predict_proba(X), est.predict_proba(X), atol=1e-12)


def test_naive_bayes_error_handling():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 3.0], [3.0, 0.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    # Invalid parameters
    with pytest.raises(ValueError):
        GaussianNB(var_smoothing=-1e-9)
    with pytest.raises(ValueError):
        MultinomialNB(alpha=-0.5)
    with pytest.raises(ValueError):
        BernoulliNB(alpha=-1.0)
    with pytest.raises(ValueError):
        ComplementNB(alpha=-0.1)

    # Negative X for MultinomialNB and ComplementNB
    X_neg = X.copy()
    X_neg[0, 0] = -1.0
    with pytest.raises(ValueError):
        MultinomialNB().fit(X_neg, y)
    with pytest.raises(ValueError):
        ComplementNB().fit(X_neg, y)

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    X_inf = X.copy()
    X_inf[1, 1] = np.inf
    for cls in (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB):
        with pytest.raises(ValueError):
            cls().fit(X_nan, y)
        with pytest.raises(ValueError):
            cls().fit(X_inf, y)

        # Empty data
        with pytest.raises(ValueError):
            cls().fit(np.empty((0, 2)), np.empty((0,)))

        # Shape mismatch
        fitted = cls().fit(X, y)
        with pytest.raises(ValueError):
            fitted.predict(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            cls().fit(X, y[:3])
