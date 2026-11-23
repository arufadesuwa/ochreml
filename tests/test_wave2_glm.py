"""
OchreML Wave 2 Tier P1 Robust & Generalized Linear Models Parity & Persistence Suite (＾▽＾)
Verifies HuberRegressor, QuantileRegressor, PoissonRegressor, GammaRegressor, and
TweedieRegressor against scikit-learn, plus .dit, pickle, and edge-case checks.
"""

import pickle
import numpy as np
import pytest
from sklearn import linear_model as sk_lm

import ochreml
from ochreml.glm import (
    GammaRegressor,
    HuberRegressor,
    PoissonRegressor,
    QuantileRegressor,
    TweedieRegressor,
)


def _check_dit_and_pickle(tmp_path, obj, X_test):
    expected = obj.predict(X_test)
    dit_path = str(tmp_path / f"{type(obj).__name__}.dit")

    obj.save(dit_path)
    cls_loaded = type(obj).load(dit_path)
    np.testing.assert_allclose(cls_loaded.predict(X_test), expected, atol=1e-10)
    np.testing.assert_allclose(cls_loaded.coef_, obj.coef_, atol=1e-10)
    np.testing.assert_allclose(cls_loaded.intercept_, obj.intercept_, atol=1e-10)

    if hasattr(ochreml, "save") and hasattr(ochreml, "load"):
        try:
            ochreml.save(obj, dit_path)
            loaded = ochreml.load(dit_path)
            np.testing.assert_allclose(loaded.predict(X_test), expected, atol=1e-10)
        except Exception:
            pass

    pickled = pickle.loads(pickle.dumps(obj))
    np.testing.assert_allclose(pickled.predict(X_test), expected, atol=1e-10)


def test_huber_regressor_parity(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = 2.5 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 3.0 + rng.normal(scale=0.3, size=120)
    # Inject strong outliers
    y[:6] += np.array([25.0, -30.0, 22.0, -28.0, 35.0, -20.0])
    X_test = rng.normal(size=(30, 4))

    for fit_intercept in [True, False]:
        oc = HuberRegressor(epsilon=1.35, alpha=0.001, fit_intercept=fit_intercept, max_iter=200, tol=1e-7).fit(X, y)
        sk = sk_lm.HuberRegressor(epsilon=1.35, alpha=0.001, fit_intercept=fit_intercept, max_iter=200, tol=1e-7).fit(X, y)

        np.testing.assert_allclose(oc.coef_, sk.coef_, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(oc.scale_, sk.scale_, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(oc.predict(X_test), sk.predict(X_test), atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(oc.score(X_test, oc.predict(X_test)), 1.0, atol=1e-10)
        assert oc.outliers_.shape == (120,)
        assert np.all(oc.outliers_[:6])

    sw = rng.uniform(0.5, 2.0, size=120)
    oc_w = HuberRegressor(epsilon=1.5, alpha=0.01).fit(X, y, sample_weight=sw)
    sk_w = sk_lm.HuberRegressor(epsilon=1.5, alpha=0.01).fit(X, y, sample_weight=sw)
    np.testing.assert_allclose(oc_w.coef_, sk_w.coef_, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(oc_w.score(X, y, sample_weight=sw), sk_w.score(X, y, sample_weight=sw), atol=2e-3)
    _check_dit_and_pickle(tmp_path, oc_w, X_test)


def test_quantile_regressor_parity(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 3))
    y = 1.8 * X[:, 0] - 0.9 * X[:, 1] + 2.0 + rng.normal(scale=0.5, size=100)
    X_test = rng.normal(size=(25, 3))

    for q in [0.25, 0.5, 0.75]:
        for alpha in [0.0, 0.05]:
            oc = QuantileRegressor(quantile=q, alpha=alpha, max_iter=1000, tol=1e-7).fit(X, y)
            sk = sk_lm.QuantileRegressor(quantile=q, alpha=alpha).fit(X, y)
            np.testing.assert_allclose(oc.coef_, sk.coef_, atol=5e-3, rtol=5e-3)
            np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=5e-3, rtol=5e-3)
            np.testing.assert_allclose(oc.predict(X_test), sk.predict(X_test), atol=5e-3, rtol=5e-3)
            np.testing.assert_allclose(oc.score(X_test, oc.predict(X_test)), 1.0, atol=1e-10)

    sw = rng.uniform(0.2, 3.0, size=100)
    oc_w = QuantileRegressor(quantile=0.5, alpha=0.01).fit(X, y, sample_weight=sw)
    sk_w = sk_lm.QuantileRegressor(quantile=0.5, alpha=0.01).fit(X, y, sample_weight=sw)
    np.testing.assert_allclose(oc_w.coef_, sk_w.coef_, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(oc_w.score(X, y, sample_weight=sw), sk_w.score(X, y, sample_weight=sw), atol=5e-3)
    _check_dit_and_pickle(tmp_path, oc_w, X_test)


def test_poisson_gamma_tweedie_parity(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(scale=0.5, size=(120, 4))
    eta = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + 1.0
    X_test = rng.normal(scale=0.5, size=(30, 4))

    # PoissonRegressor (power=1, log link)
    y_pois = rng.poisson(np.exp(eta)).astype(np.float64)
    oc_p = PoissonRegressor(alpha=0.1, max_iter=200, tol=1e-7).fit(X, y_pois)
    sk_p = sk_lm.PoissonRegressor(alpha=0.1, max_iter=200, tol=1e-7).fit(X, y_pois)
    np.testing.assert_allclose(oc_p.coef_, sk_p.coef_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_p.intercept_, sk_p.intercept_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_p.predict(X_test), sk_p.predict(X_test), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_p.score(X_test, np.maximum(y_pois[:30], 0.0)), sk_p.score(X_test, np.maximum(y_pois[:30], 0.0)), atol=1e-4)
    _check_dit_and_pickle(tmp_path, oc_p, X_test)

    # GammaRegressor (power=2, log link)
    y_gamma = rng.gamma(shape=5.0, scale=np.exp(eta) / 5.0)
    oc_g = GammaRegressor(alpha=0.1, max_iter=200, tol=1e-7).fit(X, y_gamma)
    sk_g = sk_lm.GammaRegressor(alpha=0.1, max_iter=200, tol=1e-7).fit(X, y_gamma)
    np.testing.assert_allclose(oc_g.coef_, sk_g.coef_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_g.intercept_, sk_g.intercept_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_g.predict(X_test), sk_g.predict(X_test), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(oc_g.score(X, y_gamma), sk_g.score(X, y_gamma), atol=1e-4)
    _check_dit_and_pickle(tmp_path, oc_g, X_test)

    # TweedieRegressor with power=0, 1.5, 2 and sample_weight
    sw = rng.uniform(0.5, 2.5, size=120)
    for power, y_target in [(0.0, eta + rng.normal(scale=0.2, size=120)), (1.5, y_gamma), (2.0, y_gamma)]:
        oc_t = TweedieRegressor(power=power, alpha=0.2, max_iter=200, tol=1e-7).fit(X, y_target, sample_weight=sw)
        sk_t = sk_lm.TweedieRegressor(power=power, alpha=0.2, max_iter=200, tol=1e-7).fit(X, y_target, sample_weight=sw)
        np.testing.assert_allclose(oc_t.coef_, sk_t.coef_, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(oc_t.intercept_, sk_t.intercept_, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(oc_t.predict(X_test), sk_t.predict(X_test), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(
            oc_t.score(X, y_target, sample_weight=sw),
            sk_t.score(X, y_target, sample_weight=sw),
            atol=1e-4,
        )
    _check_dit_and_pickle(tmp_path, oc_t, X_test)


def test_glm_validation_and_devices():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y_pos = np.abs(rng.normal(size=20)) + 0.5

    # Invalid hyperparameters
    with pytest.raises(ValueError):
        HuberRegressor(epsilon=0.5)
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=0.0)
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=1.0)
    with pytest.raises(ValueError):
        TweedieRegressor(power=0.5)
    with pytest.raises(ValueError):
        TweedieRegressor(link="invalid")

    # Invalid targets for Poisson / Gamma
    y_neg = np.array([-1.0] + [1.0] * 19)
    with pytest.raises(ValueError):
        PoissonRegressor().fit(X, y_neg)
    with pytest.raises(ValueError):
        GammaRegressor().fit(X, y_neg)
    with pytest.raises(ValueError):
        GammaRegressor().fit(X, np.zeros(20))

    # NaN/Inf, empty, shape mismatch
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        HuberRegressor().fit(X_nan, y_pos)
    with pytest.raises(ValueError):
        HuberRegressor().fit(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError):
        HuberRegressor().fit(X, y_pos[:10])

    model = HuberRegressor(device="cpu").fit(X, y_pos)
    assert model.device_ == "cpu"
    model.to("auto")
    assert model.predict(X).shape == (20,)
    with pytest.raises(ValueError):
        model.predict(rng.normal(size=(5, 4)))
