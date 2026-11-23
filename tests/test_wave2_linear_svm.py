# OchreML Wave 2 Linear SVM Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, and edge-case validation.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.svm import LinearSVC as SkLinearSVC, LinearSVR as SkLinearSVR

from ochreml.svm import LinearSVC, LinearSVR


@pytest.mark.parametrize("loss,penalty,dual", [
    ("squared_hinge", "l2", True),
    ("hinge", "l2", True),
    ("squared_hinge", "l2", False),
    ("squared_hinge", "l1", False),
    ("squared_hinge", "l2", "auto"),
])
def test_linear_svc_binary_parity(loss, penalty, dual):
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    sk = SkLinearSVC(
        C=1.0, loss=loss, penalty=penalty, dual=dual, tol=1e-5, max_iter=5000, random_state=42
    )
    sk.fit(X, y)

    oc = LinearSVC(
        C=1.0,
        loss=loss,
        penalty=penalty,
        dual=dual,
        tol=1e-5,
        max_iter=5000,
        random_state=42,
        device="cpu",
    )
    oc.fit(X, y)

    assert oc.coef_.shape == (1, 6)
    assert oc.intercept_.shape == (1,)
    assert oc.score(X, y) >= sk.score(X, y) - 0.02
    np.testing.assert_allclose(oc.decision_function(X), sk.decision_function(X), atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=5e-2, rtol=5e-2)
    assert np.mean(oc.predict(X) == sk.predict(X)) >= 0.98


def test_linear_svc_multiclass_parity():
    X, y = make_classification(
        n_samples=210,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )
    sk = SkLinearSVC(C=1.0, tol=1e-5, max_iter=5000, random_state=7)
    sk.fit(X, y)

    oc = LinearSVC(C=1.0, tol=1e-5, max_iter=5000, random_state=7, device="cpu")
    oc.fit(X, y)

    assert oc.coef_.shape == (3, 6)
    assert oc.intercept_.shape == (3,)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=0.02)
    np.testing.assert_allclose(oc.decision_function(X), sk.decision_function(X), atol=5e-2, rtol=5e-2)
    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))


def test_linear_svc_class_weight_and_sample_weight():
    X, y = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.85, 0.15],
        random_state=11,
    )
    sk = SkLinearSVC(class_weight="balanced", tol=1e-5, max_iter=5000, random_state=11)
    sk.fit(X, y)
    oc = LinearSVC(class_weight="balanced", tol=1e-5, max_iter=5000, random_state=11, device="cpu")
    oc.fit(X, y)
    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=5e-2, rtol=5e-2)

    rng = np.random.default_rng(123)
    sw = rng.uniform(0.2, 2.0, size=len(y))
    sk_w = SkLinearSVC(tol=1e-5, max_iter=5000, random_state=11).fit(X, y, sample_weight=sw)
    oc_w = LinearSVC(tol=1e-5, max_iter=5000, random_state=11, device="cpu").fit(
        X, y, sample_weight=sw
    )
    np.testing.assert_allclose(oc_w.coef_, sk_w.coef_, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("loss,dual", [
    ("epsilon_insensitive", True),
    ("squared_epsilon_insensitive", True),
    ("squared_epsilon_insensitive", False),
    ("epsilon_insensitive", "auto"),
])
def test_linear_svr_parity(loss, dual):
    X, y = make_regression(
        n_samples=150, n_features=5, n_informative=4, noise=0.5, random_state=21
    )
    sk = SkLinearSVR(
        epsilon=0.1, C=1.0, loss=loss, dual=dual, tol=1e-5, max_iter=5000, random_state=21
    )
    sk.fit(X, y)

    oc = LinearSVR(
        epsilon=0.1,
        C=1.0,
        loss=loss,
        dual=dual,
        tol=1e-5,
        max_iter=5000,
        random_state=21,
        device="cpu",
    )
    oc.fit(X, y)

    assert oc.coef_.shape == (5,)
    assert oc.intercept_.shape == (1,)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-2)
    np.testing.assert_allclose(oc.predict(X), sk.predict(X), atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=5e-2, rtol=5e-2)


def test_linear_svr_sample_weight():
    X, y = make_regression(n_samples=120, n_features=4, noise=0.3, random_state=5)
    rng = np.random.default_rng(5)
    sw = rng.uniform(0.5, 3.0, size=len(y))
    sk = SkLinearSVR(epsilon=0.05, C=2.0, tol=1e-5, max_iter=5000, random_state=5).fit(
        X, y, sample_weight=sw
    )
    oc = LinearSVR(
        epsilon=0.05, C=2.0, tol=1e-5, max_iter=5000, random_state=5, device="cpu"
    ).fit(X, y, sample_weight=sw)
    np.testing.assert_allclose(oc.predict(X), sk.predict(X), atol=5e-2, rtol=5e-2)


def test_dit_save_load_pickle_and_devices(tmp_path):
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    svc = LinearSVC(device="auto", random_state=0).fit(X, y)
    svc_path = str(tmp_path / "svc.dit")
    svc.save(svc_path)
    svc_loaded = LinearSVC.load(svc_path, device="cpu")
    np.testing.assert_array_equal(svc.predict(X), svc_loaded.predict(X))
    np.testing.assert_allclose(svc.decision_function(X), svc_loaded.decision_function(X))

    svc_pickled = pickle.loads(pickle.dumps(svc))
    np.testing.assert_array_equal(svc.predict(X), svc_pickled.predict(X))
    svc_pickled.to("cpu")
    assert svc_pickled.device == "cpu"

    Xr, yr = make_regression(n_samples=80, n_features=4, random_state=0)
    svr = LinearSVR(device="auto", random_state=0).fit(Xr, yr)
    svr_path = str(tmp_path / "svr.dit")
    svr.save(svr_path)
    svr_loaded = LinearSVR.load(svr_path, device="cpu")
    np.testing.assert_allclose(svr.predict(Xr), svr_loaded.predict(Xr))

    svr_pickled = pickle.loads(pickle.dumps(svr))
    np.testing.assert_allclose(svr.predict(Xr), svr_pickled.predict(Xr))


def test_validation_and_edge_cases():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        LinearSVC(C=0.0)
    with pytest.raises(ValueError):
        LinearSVC(C=-1.0)
    with pytest.raises(ValueError):
        LinearSVR(C=-0.5)
    with pytest.raises(ValueError):
        LinearSVR(epsilon=-0.1)

    svc = LinearSVC().fit(X, y)
    with pytest.raises(ValueError):
        svc.predict(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        svc.fit(np.array([[np.nan, 1.0], [2.0, 3.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        svc.fit(np.empty((0, 2)), np.empty((0,)))

    svr = LinearSVR().fit(X, y)
    with pytest.raises(ValueError):
        svr.predict(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        svr.fit(np.array([[np.inf, 1.0], [2.0, 3.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        svr.fit(np.empty((0, 2)), np.empty((0,)))
