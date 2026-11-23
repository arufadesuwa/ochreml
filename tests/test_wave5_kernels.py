# OchreML Wave 5 Tier P3 Kernel Methods Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, and edge-case validation.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression, make_circles
from sklearn.decomposition import KernelPCA as SkKernelPCA
from sklearn.kernel_ridge import KernelRidge as SkKernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import OneClassSVM as SkOneClassSVM, SVC as SkSVC, SVR as SkSVR

from ochreml.kernel_ridge import KernelRidge
from ochreml.kernel_svm import KernelPCA, OneClassSVM, SVC, SVR


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
def test_svc_binary_parity(kernel):
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    sk = SkSVC(C=1.0, kernel=kernel, degree=3, gamma="scale", coef0=0.5, tol=1e-4, random_state=42)
    sk.fit(X, y)

    oc = SVC(
        C=1.0,
        kernel=kernel,
        degree=3,
        gamma="scale",
        coef0=0.5,
        tol=1e-4,
        random_state=42,
        device="cpu",
    )
    oc.fit(X, y)

    assert oc.n_features_in_ == 6
    assert oc.classes_.shape == (2,)
    assert oc.support_vectors_ is not None
    assert oc.n_support_.sum() == len(oc.support_)
    assert oc.score(X, y) >= sk.score(X, y) - 0.02
    np.testing.assert_allclose(oc.decision_function(X), sk.decision_function(X), atol=5e-2, rtol=5e-2)
    assert np.mean(oc.predict(X) == sk.predict(X)) >= 0.98


def test_svc_multiclass_and_probability():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )
    sk = SkSVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=7)
    sk.fit(X, y)

    oc = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=7, device="cpu")
    oc.fit(X, y)

    assert oc.classes_.shape == (3,)
    assert oc.n_support_.shape == (3,)
    assert oc.dual_coef_.shape[0] == 2
    assert oc.intercept_.shape == (3,)
    assert oc.score(X, y) >= sk.score(X, y) - 0.03
    assert np.mean(oc.predict(X) == sk.predict(X)) >= 0.95

    probs = oc.predict_proba(X)
    assert probs.shape == (150, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(150), atol=1e-6)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.mean(np.argmax(probs, axis=1) == y) >= 0.85


def test_svc_precomputed():
    X, y = make_classification(n_samples=80, n_features=4, random_state=13)
    K_train = rbf_kernel(X[:60], X[:60], gamma=0.5)
    K_test = rbf_kernel(X[60:], X[:60], gamma=0.5)

    sk = SkSVC(C=1.0, kernel="precomputed").fit(K_train, y[:60])
    oc = SVC(C=1.0, kernel="precomputed", device="cpu").fit(K_train, y[:60])

    assert oc.support_vectors_ is None
    np.testing.assert_array_equal(oc.predict(K_test), sk.predict(K_test))


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly"])
def test_svr_parity(kernel):
    X, y = make_regression(n_samples=100, n_features=5, n_informative=4, noise=0.5, random_state=21)
    sk = SkSVR(kernel=kernel, C=2.0, epsilon=0.1, gamma="scale", tol=1e-4)
    sk.fit(X, y)

    oc = SVR(kernel=kernel, C=2.0, epsilon=0.1, gamma="scale", tol=1e-4, device="cpu")
    oc.fit(X, y)

    assert oc.n_features_in_ == 5
    assert oc.intercept_.shape == (1,)
    assert oc.score(X, y) >= sk.score(X, y) - 0.02
    np.testing.assert_allclose(oc.predict(X), sk.predict(X), atol=5e-2, rtol=5e-2)


def test_one_class_svm_parity():
    rng = np.random.default_rng(42)
    X_in = 0.3 * rng.standard_normal((80, 2))
    X_out = rng.uniform(low=-4, high=4, size=(20, 2))

    sk = SkOneClassSVM(kernel="rbf", gamma=0.5, nu=0.1, tol=1e-4).fit(X_in)
    oc = OneClassSVM(kernel="rbf", gamma=0.5, nu=0.1, tol=1e-4, device="cpu").fit(X_in)

    assert oc.n_features_in_ == 2
    assert oc.intercept_.shape == (1,)
    np.testing.assert_allclose(oc.decision_function(X_in), sk.decision_function(X_in), atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(oc.score_samples(X_in), sk.score_samples(X_in), atol=5e-2, rtol=5e-2)
    assert np.mean(oc.predict(X_in) == 1.0) >= 0.80
    assert np.mean(oc.predict(X_out) == -1.0) >= 0.70


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_kernel_ridge_1d_and_multioutput(kernel):
    X, y = make_regression(n_samples=80, n_features=5, n_informative=4, noise=0.5, random_state=33)
    sk = SkKernelRidge(alpha=0.5, kernel=kernel, gamma=0.2, degree=2, coef0=1.0).fit(X, y)
    oc = KernelRidge(alpha=0.5, kernel=kernel, gamma=0.2, degree=2, coef0=1.0, device="cpu").fit(X, y)

    np.testing.assert_allclose(oc.predict(X), sk.predict(X), atol=1e-6, rtol=1e-6)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-6)

    # Multi-output + sample_weight
    y_multi = np.column_stack([y, 2.0 * y + 1.0])
    rng = np.random.default_rng(9)
    sw = rng.uniform(0.5, 2.0, size=len(y))
    sk_m = SkKernelRidge(alpha=0.5, kernel=kernel, gamma=0.2, degree=2, coef0=1.0).fit(
        X, y_multi, sample_weight=sw
    )
    oc_m = KernelRidge(alpha=0.5, kernel=kernel, gamma=0.2, degree=2, coef0=1.0, device="cpu").fit(
        X, y_multi, sample_weight=sw
    )
    np.testing.assert_allclose(oc_m.predict(X), sk_m.predict(X), atol=1e-5, rtol=1e-5)
    assert oc_m.score(X, y_multi, sample_weight=sw) == pytest.approx(
        sk_m.score(X, y_multi, sample_weight=sw), abs=1e-5
    )


def test_kernel_pca_and_inverse_transform():
    X, _ = make_circles(n_samples=60, factor=0.3, noise=0.05, random_state=42)
    sk = SkKernelPCA(
        n_components=3, kernel="rbf", gamma=10.0, fit_inverse_transform=True, alpha=0.1
    )
    Z_sk = sk.fit_transform(X)

    oc = KernelPCA(
        n_components=3,
        kernel="rbf",
        gamma=10.0,
        fit_inverse_transform=True,
        alpha=0.1,
        device="cpu",
    )
    Z_oc = oc.fit_transform(X)

    assert Z_oc.shape == (60, 3)
    assert oc.eigenvalues_.shape == (3,)
    assert oc.eigenvectors_.shape == (60, 3)
    assert oc.n_features_in_ == 2
    np.testing.assert_allclose(oc.eigenvalues_, sk.eigenvalues_, atol=1e-5, rtol=1e-5)

    # Component-wise projections match up to sign
    for j in range(3):
        corr = np.abs(np.corrcoef(Z_oc[:, j], Z_sk[:, j])[0, 1])
        assert corr > 0.999

    X_rec_oc = oc.inverse_transform(Z_oc)
    X_rec_sk = sk.inverse_transform(Z_oc)
    assert X_rec_oc.shape == X.shape
    np.testing.assert_allclose(X_rec_oc, X_rec_sk, atol=1e-4, rtol=1e-4)


def test_dit_and_pickle_persistence_and_devices(tmp_path):
    X, y = make_classification(n_samples=60, n_features=4, random_state=5)
    y_reg = X[:, 0] * 2.0 + 1.0

    models = [
        (SVC(C=1.0, probability=True, random_state=5, device="auto").fit(X, y), X, "svc.dit"),
        (SVR(C=1.0, device="auto").fit(X, y_reg), X, "svr.dit"),
        (OneClassSVM(nu=0.2, device="auto").fit(X), X, "ocsvm.dit"),
        (KernelRidge(alpha=0.5, kernel="rbf", device="auto").fit(X, y_reg), X, "kr.dit"),
        (KernelPCA(n_components=2, kernel="rbf", device="auto").fit(X), X, "kpca.dit"),
    ]

    for model, X_in, fname in models:
        path = tmp_path / fname
        model.save(str(path))
        loaded = model.__class__.load(str(path), device="cpu")
        assert loaded.device_ == "cpu"
        loaded.to("auto")
        assert loaded.device_ == "auto"

        pickled = pickle.loads(pickle.dumps(model))
        if hasattr(model, "predict"):
            np.testing.assert_allclose(model.predict(X_in), loaded.predict(X_in), atol=1e-12)
            np.testing.assert_allclose(model.predict(X_in), pickled.predict(X_in), atol=1e-12)
        else:
            np.testing.assert_allclose(model.transform(X_in), loaded.transform(X_in), atol=1e-12)
            np.testing.assert_allclose(model.transform(X_in), pickled.transform(X_in), atol=1e-12)


def test_validation_and_error_handling():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        SVC(C=0.0)
    with pytest.raises(ValueError):
        SVC(C=-1.0)
    with pytest.raises(ValueError):
        OneClassSVM(nu=0.0)
    with pytest.raises(ValueError):
        OneClassSVM(nu=1.5)
    with pytest.raises(ValueError):
        KernelRidge(alpha=-0.1)
    with pytest.raises(ValueError):
        SVC(kernel="unknown_kernel")

    # NaN/Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        SVC().fit(X_nan, y)
    with pytest.raises(ValueError):
        KernelRidge().fit(X_nan, y)
    with pytest.raises(ValueError):
        KernelPCA().fit(X_nan)

    # Empty data
    with pytest.raises(ValueError):
        SVC().fit(np.empty((0, 2)), np.empty((0,)))

    # Shape mismatch
    svc = SVC().fit(X, y)
    with pytest.raises(ValueError):
        svc.predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        KernelPCA(fit_inverse_transform=False).fit(X).inverse_transform(np.ones((4, 2)))
