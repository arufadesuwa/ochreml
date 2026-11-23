# OchreML Wave 3 Tier P1 Gaussian Mixture Model Tests (＾▽＾)
# Verifies GaussianMixture numerical parity with scikit-learn across all 4 covariance types,
# init_params, n_init, score/aic/bic/sample, .dit/pickle persistence, and validation.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from ochreml.mixture import GaussianMixture


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_covariance_types_parity(covariance_type):
    X, y_true = make_blobs(
        n_samples=300, n_features=4, centers=3, cluster_std=0.8, random_state=42
    )

    sk = SkGaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=200,
        n_init=3,
        init_params="kmeans",
        random_state=42,
    ).fit(X)

    oc = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=200,
        n_init=3,
        init_params="kmeans",
        random_state=42,
        device="cpu",
    )
    labels = oc.fit_predict(X)

    assert oc.converged_ is True
    assert oc.n_iter_ >= 1
    assert oc.n_features_in_ == 4
    assert oc.weights_.shape == (3,)
    assert np.isclose(np.sum(oc.weights_), 1.0, atol=1e-9)
    assert oc.means_.shape == (3, 4)

    expected_cov_shapes = {
        "full": (3, 4, 4),
        "tied": (4, 4),
        "diag": (3, 4),
        "spherical": (3,),
    }
    assert oc.covariances_.shape == expected_cov_shapes[covariance_type]
    assert oc.precisions_cholesky_.shape == expected_cov_shapes[covariance_type]

    assert adjusted_rand_score(y_true, labels) > 0.95
    assert adjusted_rand_score(sk.predict(X), labels) > 0.95

    probs = oc.predict_proba(X)
    assert probs.shape == (300, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(300), atol=1e-9)
    np.testing.assert_array_equal(oc.predict(X), labels)

    ll_samples = oc.score_samples(X)
    assert ll_samples.shape == (300,)
    assert oc.score(X) == pytest.approx(float(np.mean(ll_samples)), rel=1e-12)
    assert oc.score(X) == pytest.approx(sk.score(X), rel=1e-2)
    assert oc.aic(X) == pytest.approx(sk.aic(X), rel=1e-2)
    assert oc.bic(X) == pytest.approx(sk.bic(X), rel=1e-2)


def test_gmm_random_init_and_n_init():
    X, y_true = make_blobs(
        n_samples=240, n_features=3, centers=3, cluster_std=0.7, random_state=7
    )
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=5,
        init_params="random",
        max_iter=200,
        random_state=7,
        device="auto",
    ).fit(X)

    assert gmm.converged_ is True
    assert gmm.lower_bound_ is not None
    assert adjusted_rand_score(y_true, gmm.predict(X)) > 0.90


def test_gmm_sample():
    X, _ = make_blobs(
        n_samples=200, n_features=3, centers=3, cluster_std=0.6, random_state=11
    )
    for cov_type in ("full", "tied", "diag", "spherical"):
        gmm = GaussianMixture(
            n_components=3, covariance_type=cov_type, random_state=11
        ).fit(X)
        X_syn, y_syn = gmm.sample(50)
        assert X_syn.shape == (50, 3)
        assert y_syn.shape == (50,)
        assert np.all(np.isfinite(X_syn))
        assert np.all((y_syn >= 0) & (y_syn < 3))


def test_gmm_persistence_dit_and_pickle(tmp_path):
    X, _ = make_blobs(
        n_samples=150, n_features=3, centers=3, cluster_std=0.7, random_state=99
    )
    model = GaussianMixture(
        n_components=3, covariance_type="full", random_state=99, device="cpu"
    ).fit(X)

    dit_path = tmp_path / "gmm.dit"
    model.save(str(dit_path))

    loaded = GaussianMixture.load(str(dit_path), device="auto")
    np.testing.assert_allclose(loaded.weights_, model.weights_, atol=1e-12)
    np.testing.assert_allclose(loaded.means_, model.means_, atol=1e-12)
    np.testing.assert_allclose(loaded.covariances_, model.covariances_, atol=1e-12)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    assert loaded.score(X) == pytest.approx(model.score(X), rel=1e-12)

    unpickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(unpickled.predict(X), model.predict(X))
    assert unpickled.score(X) == pytest.approx(model.score(X), rel=1e-12)

    model.to("cpu")
    assert model.device_ == "cpu"


def test_gmm_validation_and_error_handling():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    with pytest.raises(ValueError):
        GaussianMixture(n_components=0)
    with pytest.raises(ValueError):
        GaussianMixture(reg_covar=-1e-6)
    with pytest.raises(ValueError):
        GaussianMixture(covariance_type="invalid")
    with pytest.raises(ValueError):
        GaussianMixture(init_params="invalid")
    with pytest.raises(ValueError):
        GaussianMixture(n_components=5).fit(X)
    with pytest.raises(ValueError):
        GaussianMixture(n_components=2).fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        GaussianMixture(n_components=2).fit(np.array([[0.0, np.nan], [1.0, 1.0]]))

    fitted = GaussianMixture(n_components=2, random_state=0).fit(X)
    with pytest.raises(ValueError):
        fitted.predict(np.array([[0.0, 1.0, 2.0]]))
