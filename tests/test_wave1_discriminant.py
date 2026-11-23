# OchreML Wave 1 Discriminant Analysis Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, and edge-case validation.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SkLDA,
    QuadraticDiscriminantAnalysis as SkQDA,
)

from ochreml.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_lda_binary_parity(solver):
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    sk = SkLDA(solver=solver)
    sk.fit(X, y)

    oc = LinearDiscriminantAnalysis(solver=solver, device="cpu")
    oc.fit(X, y)

    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
    np.testing.assert_allclose(oc.decision_function(X), sk.decision_function(X), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-4, rtol=1e-4)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)
    assert oc.coef_.shape == (1, 6)
    assert oc.intercept_.shape == (1,)


@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_lda_multiclass_parity_and_transform(solver):
    X, y = make_classification(
        n_samples=210,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )
    sk = SkLDA(solver=solver, n_components=2)
    sk.fit(X, y)

    oc = LinearDiscriminantAnalysis(solver=solver, n_components=2, device="cpu")
    oc.fit(X, y)

    np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-4, rtol=1e-4)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)
    assert oc.coef_.shape == (3, 6)
    assert oc.intercept_.shape == (3,)

    if solver in ("svd", "eigen"):
        X_oc = oc.transform(X)
        X_sk = sk.transform(X)
        assert X_oc.shape == (210, 2)
        # Eigenvector sign may flip; compare column-wise absolute projections
        for c in range(2):
            dot = np.dot(X_oc[:, c], X_sk[:, c])
            sign = 1.0 if dot >= 0 else -1.0
            np.testing.assert_allclose(sign * X_oc[:, c], X_sk[:, c], atol=1e-4, rtol=1e-4)

        X_ft = LinearDiscriminantAnalysis(solver=solver, n_components=2).fit_transform(X, y)
        assert X_ft.shape == (210, 2)
    else:
        with pytest.raises(ValueError):
            oc.transform(X)


@pytest.mark.parametrize("solver", ["lsqr", "eigen"])
@pytest.mark.parametrize("shrinkage", [0.25, "auto"])
def test_lda_shrinkage_parity(solver, shrinkage):
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=99,
    )
    sk = SkLDA(solver=solver, shrinkage=shrinkage)
    sk.fit(X, y)

    oc = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, device="cpu")
    oc.fit(X, y)

    tol = 0.06 if shrinkage == "auto" else 2e-2
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=tol, rtol=tol)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=0.02)


def test_qda_binary_and_multiclass_parity():
    for n_classes in (2, 3):
        X, y = make_classification(
            n_samples=240,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=123 + n_classes,
        )
        for reg_param in (0.0, 0.2):
            sk = SkQDA(reg_param=reg_param, store_covariance=True)
            sk.fit(X, y)

            oc = QuadraticDiscriminantAnalysis(
                reg_param=reg_param, store_covariance=True, device="cpu"
            )
            oc.fit(X, y)

            np.testing.assert_array_equal(oc.predict(X), sk.predict(X))
            np.testing.assert_allclose(
                oc.decision_function(X), sk.decision_function(X), atol=1e-5, rtol=1e-4
            )
            np.testing.assert_allclose(
                oc.predict_proba(X), sk.predict_proba(X), atol=1e-5, rtol=1e-4
            )
            np.testing.assert_allclose(
                oc.predict_log_proba(X), sk.predict_log_proba(X), atol=1e-4, rtol=1e-4
            )
            assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-12)
            assert oc.covariance_ is not None
            assert len(oc.covariance_) == n_classes
            for k in range(n_classes):
                np.testing.assert_allclose(oc.covariance_[k], sk.covariance_[k], atol=1e-6)


def test_discriminant_save_load_pickle_and_devices(tmp_path):
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=11
    )

    for cls, kwargs, fname in [
        (LinearDiscriminantAnalysis, {"solver": "svd"}, "lda.dit"),
        (QuadraticDiscriminantAnalysis, {"reg_param": 0.1, "store_covariance": True}, "qda.dit"),
    ]:
        for dev in ("cpu", "auto"):
            est = cls(**kwargs, device=dev)
            est.fit(X, y)
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


def test_discriminant_error_handling():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.5, 4.5]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    # Invalid hyperparameters
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis(solver="invalid")
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis(solver="svd", shrinkage=0.5)
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage=-0.1)
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis(tol=-1e-4)
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis(reg_param=1.5)
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis(tol=-0.1)

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(X_nan, y)
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis().fit(X_nan, y)

    X_inf = X.copy()
    X_inf[1, 1] = np.inf
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(X_inf, y)

    # Empty data
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(np.empty((0, 2)), np.empty((0,)))
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis().fit(np.empty((0, 2)), np.empty((0,)))

    # Single class
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(X, np.zeros(4))
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis().fit(X, np.zeros(4))

    # QDA class with < 2 samples
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis().fit(X[:3], np.array([0.0, 0.0, 1.0]))

    # Length / shape mismatch
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(X, y[:3])
    lda = LinearDiscriminantAnalysis().fit(X, y)
    with pytest.raises(ValueError):
        lda.predict(np.zeros((2, 3)))
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    with pytest.raises(ValueError):
        qda.predict(np.zeros((2, 3)))

    # Invalid priors
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis(priors=[0.3, 0.3]).fit(X, y)
    with pytest.raises(ValueError):
        QuadraticDiscriminantAnalysis(priors=[-0.2, 1.2]).fit(X, y)
