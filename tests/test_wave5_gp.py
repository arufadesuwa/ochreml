# OchreML Wave 5 Tier P3 Gaussian Process Tests (＾▽＾)
# Verifies GaussianProcessRegressor and GaussianProcessClassifier parity with scikit-learn,
# kernel variants, return_std/return_cov, normalize_y, optimizer restarts, .dit/pickle persistence,
# device switching, and validation/error handling.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.gaussian_process import (
    GaussianProcessClassifier as SkGaussianProcessClassifier,
    GaussianProcessRegressor as SkGaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, WhiteKernel

from ochreml.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor


def test_gpr_rbf_parity_std_cov():
    rng = np.random.RandomState(42)
    X_train = np.linspace(0.0, 5.0, 20).reshape(-1, 1)
    y_train = np.sin(X_train[:, 0]) + 0.05 * rng.randn(20)
    X_test = np.linspace(-0.5, 5.5, 15).reshape(-1, 1)

    sk_kernel = C(1.5, "fixed") * RBF(1.2, "fixed")
    sk = SkGaussianProcessRegressor(
        kernel=sk_kernel, alpha=1e-3, optimizer=None, normalize_y=False
    ).fit(X_train, y_train)
    sk_mean, sk_std = sk.predict(X_test, return_std=True)
    _, sk_cov = sk.predict(X_test, return_cov=True)

    oc = GaussianProcessRegressor(
        kernel="rbf",
        length_scale=1.2,
        constant_value=1.5,
        alpha=1e-3,
        optimize_hyperparams=False,
        normalize_y=False,
        device="cpu",
    ).fit(X_train, y_train)

    oc_mean, oc_std = oc.predict(X_test, return_std=True)
    oc_mean2, oc_cov = oc.predict(X_test, return_cov=True)

    np.testing.assert_allclose(oc_mean, sk_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(oc_mean2, sk_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(oc_std, sk_std, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(oc_cov, sk_cov, rtol=1e-6, atol=1e-6)
    assert oc.log_marginal_likelihood() == pytest.approx(
        sk.log_marginal_likelihood_value_, rel=1e-6
    )
    assert oc.score(X_train, y_train) == pytest.approx(sk.score(X_train, y_train), rel=1e-6)


def test_gpr_normalize_y_and_kernels_parity():
    rng = np.random.RandomState(7)
    X_train = rng.uniform(-2.0, 2.0, size=(25, 2))
    y_train = 100.0 + np.cos(X_train[:, 0]) * np.sin(X_train[:, 1])
    X_test = rng.uniform(-2.0, 2.0, size=(10, 2))

    for nu in (0.5, 1.5, 2.5):
        sk = SkGaussianProcessRegressor(
            kernel=C(2.0, "fixed") * Matern(length_scale=1.3, nu=nu, length_scale_bounds="fixed"),
            alpha=1e-4,
            optimizer=None,
            normalize_y=True,
        ).fit(X_train, y_train)
        oc = GaussianProcessRegressor(
            kernel="matern",
            length_scale=1.3,
            constant_value=2.0,
            nu=nu,
            alpha=1e-4,
            normalize_y=True,
            device="auto",
        ).fit(X_train, y_train)

        sk_mean, sk_std = sk.predict(X_test, return_std=True)
        oc_mean, oc_std = oc.predict(X_test, return_std=True)
        np.testing.assert_allclose(oc_mean, sk_mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(oc_std, sk_std, rtol=1e-5, atol=1e-5)

    # RationalQuadratic
    sk_rq = SkGaussianProcessRegressor(
        kernel=C(1.2, "fixed") * RationalQuadratic(length_scale=0.8, alpha=2.0, length_scale_bounds="fixed", alpha_bounds="fixed"),
        alpha=1e-4,
        optimizer=None,
    ).fit(X_train, y_train)
    oc_rq = GaussianProcessRegressor(
        kernel="rational_quadratic",
        length_scale=0.8,
        constant_value=1.2,
        rq_alpha=2.0,
        alpha=1e-4,
    ).fit(X_train, y_train)
    np.testing.assert_allclose(oc_rq.predict(X_test), sk_rq.predict(X_test), rtol=1e-5, atol=1e-5)

    # RBF + WhiteKernel
    sk_rw = SkGaussianProcessRegressor(
        kernel=C(1.5, "fixed") * RBF(1.1, "fixed") + WhiteKernel(0.05, "fixed"),
        alpha=1e-6,
        optimizer=None,
    ).fit(X_train, y_train)
    oc_rw = GaussianProcessRegressor(
        kernel="rbf_white",
        length_scale=1.1,
        constant_value=1.5,
        noise_level=0.05,
        alpha=1e-6,
    ).fit(X_train, y_train)
    np.testing.assert_allclose(oc_rw.predict(X_test), sk_rw.predict(X_test), rtol=1e-5, atol=1e-5)


def test_gpr_optimizer_restarts():
    rng = np.random.RandomState(12)
    X_train = np.linspace(0.0, 6.0, 24).reshape(-1, 1)
    y_train = 2.0 * np.sin(1.5 * X_train[:, 0]) + 0.1 * rng.randn(24)

    gp_fixed = GaussianProcessRegressor(
        kernel="rbf",
        length_scale=5.0,
        constant_value=0.1,
        alpha=0.05,
        optimize_hyperparams=False,
    ).fit(X_train, y_train)

    gp_opt = GaussianProcessRegressor(
        kernel="rbf",
        length_scale=5.0,
        constant_value=0.1,
        alpha=0.05,
        optimize_hyperparams=True,
        n_restarts_optimizer=3,
        random_state=12,
    ).fit(X_train, y_train)

    assert gp_opt.log_marginal_likelihood() >= gp_fixed.log_marginal_likelihood()
    assert gp_opt.score(X_train, y_train) > 0.9


def test_gpc_binary_and_multiclass_parity():
    # Binary classification
    X_bin, y_bin = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    sk_bin = SkGaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), optimizer=None, random_state=42
    ).fit(X_bin, y_bin)
    oc_bin = GaussianProcessClassifier(
        kernel="rbf", length_scale=1.0, constant_value=1.0, random_state=42, device="cpu"
    ).fit(X_bin, y_bin)

    np.testing.assert_array_equal(oc_bin.classes_, sk_bin.classes_)
    np.testing.assert_allclose(oc_bin.predict_proba(X_bin), sk_bin.predict_proba(X_bin), atol=5e-3)
    np.testing.assert_array_equal(oc_bin.predict(X_bin), sk_bin.predict(X_bin))
    assert oc_bin.score(X_bin, y_bin) == pytest.approx(sk_bin.score(X_bin, y_bin), abs=1e-6)
    assert oc_bin.log_marginal_likelihood() == pytest.approx(
        sk_bin.log_marginal_likelihood_value_, rel=1e-3
    )

    # Multiclass classification (One-vs-Rest)
    X_mc, y_mc = make_classification(
        n_samples=75,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=99,
    )
    sk_mc = SkGaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), optimizer=None, multi_class="one_vs_rest", random_state=99
    ).fit(X_mc, y_mc)
    oc_mc = GaussianProcessClassifier(
        kernel="rbf", length_scale=1.0, constant_value=1.0, random_state=99, device="auto"
    ).fit(X_mc, y_mc)

    assert oc_mc.n_classes_ == 3
    probs_mc = oc_mc.predict_proba(X_mc)
    assert probs_mc.shape == (75, 3)
    np.testing.assert_allclose(probs_mc.sum(axis=1), np.ones(75), atol=1e-9)
    np.testing.assert_allclose(probs_mc, sk_mc.predict_proba(X_mc), atol=5e-3)
    assert oc_mc.score(X_mc, y_mc) > 0.85


def test_gp_persistence_dit_pickle_and_device(tmp_path):
    X = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 1.9], [3.0, 3.1]])
    y_reg = np.array([0.2, 1.1, 2.0, 3.2])
    y_cls = np.array([0, 0, 1, 1])

    gpr = GaussianProcessRegressor(
        kernel="matern", nu=1.5, alpha=1e-4, normalize_y=True, device="cpu"
    ).fit(X, y_reg)
    gpr_path = tmp_path / "gpr.dit"
    gpr.save(str(gpr_path))
    gpr_loaded = GaussianProcessRegressor.load(str(gpr_path), device="auto")
    np.testing.assert_allclose(gpr_loaded.predict(X), gpr.predict(X), atol=1e-12)
    gpr_unpickled = pickle.loads(pickle.dumps(gpr))
    np.testing.assert_allclose(gpr_unpickled.predict(X), gpr.predict(X), atol=1e-12)
    gpr.to("cpu")
    assert gpr.device_ == "cpu"

    gpc = GaussianProcessClassifier(kernel="rbf", device="cpu").fit(X, y_cls)
    gpc_path = tmp_path / "gpc.dit"
    gpc.save(str(gpc_path))
    gpc_loaded = GaussianProcessClassifier.load(str(gpc_path), device="auto")
    np.testing.assert_allclose(gpc_loaded.predict_proba(X), gpc.predict_proba(X), atol=1e-12)
    gpc_unpickled = pickle.loads(pickle.dumps(gpc))
    np.testing.assert_array_equal(gpc_unpickled.predict(X), gpc.predict(X))
    gpc.to("cpu")
    assert gpc.device_ == "cpu"


def test_gp_validation_and_error_handling():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y_reg = np.array([0.0, 1.0, 2.0])
    y_cls = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        GaussianProcessRegressor(alpha=-1e-3)
    with pytest.raises(ValueError):
        GaussianProcessRegressor(length_scale=0.0)
    with pytest.raises(ValueError):
        GaussianProcessRegressor(nu=2.0)
    with pytest.raises(ValueError):
        GaussianProcessRegressor(kernel="unknown_kernel")
    with pytest.raises(ValueError):
        GaussianProcessClassifier(length_scale=-1.0)
    with pytest.raises(ValueError):
        GaussianProcessClassifier(max_iter_predict=0)

    gpr = GaussianProcessRegressor().fit(X, y_reg)
    with pytest.raises(ValueError):
        gpr.predict(X, return_std=True, return_cov=True)
    with pytest.raises(ValueError):
        gpr.predict(np.array([[0.0, 1.0, 2.0]]))
    with pytest.raises(ValueError):
        GaussianProcessRegressor().fit(np.empty((0, 2)), np.empty((0,)))
    with pytest.raises(ValueError):
        GaussianProcessRegressor().fit(np.array([[0.0, np.nan], [1.0, 1.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        GaussianProcessRegressor().fit(X, np.array([0.0, 1.0]))

    gpc = GaussianProcessClassifier().fit(X, y_cls)
    with pytest.raises(ValueError):
        gpc.predict(np.array([[0.0, np.inf]]))
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(X, np.array([1.0, 1.0, 1.0]))
