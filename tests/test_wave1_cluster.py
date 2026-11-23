# OchreML Wave 1 Tier P0 Clustering Tests (＾▽＾)
# Verifies KMeans and MiniBatchKMeans numerical parity with scikit-learn,
# .dit/pickle round-trips, edge cases, and device fallback.

import pickle
import numpy as np
import pytest
from sklearn.cluster import (
    KMeans as SkKMeans,
    MiniBatchKMeans as SkMiniBatchKMeans,
)
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ochreml.cluster import KMeans, MiniBatchKMeans


def test_kmeans_parity_k_means_pp_and_random():
    X, y_true = make_blobs(
        n_samples=300, n_features=5, centers=4, cluster_std=0.8, random_state=42
    )

    for init_method in ("k-means++", "random"):
        sk = SkKMeans(
            n_clusters=4, init=init_method, n_init=10, max_iter=300, tol=1e-4, random_state=42
        )
        sk.fit(X)

        oc = KMeans(
            n_clusters=4,
            init=init_method,
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=42,
            device="cpu",
        )
        labels = oc.fit_predict(X)

        assert labels.shape == (300,)
        assert oc.cluster_centers_.shape == (4, 5)
        assert oc.n_features_in_ == 5
        assert oc.n_iter_ >= 1
        assert adjusted_rand_score(y_true, labels) > 0.95
        assert adjusted_rand_score(sk.labels_, labels) > 0.95
        assert oc.inertia_ == pytest.approx(sk.inertia_, rel=1e-3)
        assert oc.score(X) == pytest.approx(-oc.inertia_, rel=1e-12)
        np.testing.assert_array_equal(oc.predict(X), labels)


def test_kmeans_sample_weight_and_transform():
    X, _ = make_blobs(
        n_samples=150, n_features=3, centers=3, cluster_std=0.9, random_state=7
    )
    sw = np.linspace(0.5, 2.5, len(X))

    sk = SkKMeans(n_clusters=3, n_init=10, random_state=7).fit(X, sample_weight=sw)
    oc = KMeans(n_clusters=3, n_init=10, random_state=7, device="auto").fit(
        X, sample_weight=sw
    )

    assert oc.inertia_ == pytest.approx(sk.inertia_, rel=1e-3)
    assert oc.score(X, sample_weight=sw) == pytest.approx(
        sk.score(X, sample_weight=sw), rel=1e-3
    )

    dist = oc.transform(X)
    assert dist.shape == (150, 3)
    assert np.all(dist >= 0.0)
    np.testing.assert_array_equal(np.argmin(dist, axis=1), oc.predict(X))

    dist2 = KMeans(n_clusters=3, n_init=10, random_state=7).fit_transform(
        X, sample_weight=sw
    )
    np.testing.assert_allclose(dist2, dist, atol=1e-12)


def test_minibatch_kmeans_fit_and_partial_fit():
    X, y_true = make_blobs(
        n_samples=400, n_features=4, centers=4, cluster_std=0.8, random_state=13
    )

    sk = SkMiniBatchKMeans(
        n_clusters=4, batch_size=64, max_iter=100, n_init=3, random_state=13
    )
    sk.fit(X)

    oc = MiniBatchKMeans(
        n_clusters=4,
        batch_size=64,
        max_iter=100,
        n_init=3,
        random_state=13,
        device="cpu",
    )
    labels = oc.fit_predict(X)

    assert oc.cluster_centers_.shape == (4, 4)
    assert adjusted_rand_score(y_true, labels) > 0.90
    assert oc.inertia_ <= sk.inertia_ * 1.25
    assert oc.score(X) == pytest.approx(-oc.inertia_, rel=1e-12)

    dist = oc.transform(X)
    assert dist.shape == (400, 4)
    np.testing.assert_array_equal(np.argmin(dist, axis=1), oc.predict(X))

    # Incremental partial_fit
    mbk_inc = MiniBatchKMeans(
        n_clusters=4, batch_size=50, random_state=13, device="auto"
    )
    for start in range(0, len(X), 50):
        mbk_inc.partial_fit(X[start : start + 50])
    assert mbk_inc.n_steps_ == 8
    assert adjusted_rand_score(y_true, mbk_inc.predict(X)) > 0.85


def test_cluster_persistence_dit_and_pickle(tmp_path):
    X, _ = make_blobs(
        n_samples=120, n_features=3, centers=3, cluster_std=0.7, random_state=99
    )

    for cls, fname in [(KMeans, "km.dit"), (MiniBatchKMeans, "mbk.dit")]:
        model = cls(n_clusters=3, random_state=99, device="cpu").fit(X)
        dit_path = tmp_path / fname
        model.save(str(dit_path))

        loaded = cls.load(str(dit_path), device="auto")
        np.testing.assert_allclose(loaded.cluster_centers_, model.cluster_centers_, atol=1e-12)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        np.testing.assert_allclose(loaded.transform(X), model.transform(X), atol=1e-12)
        assert loaded.score(X) == pytest.approx(model.score(X), rel=1e-12)

        unpickled = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(unpickled.predict(X), model.predict(X))

        model.to("cpu")
        assert model.device_ == "cpu"


def test_cluster_validation_and_error_handling():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    for cls in (KMeans, MiniBatchKMeans):
        with pytest.raises(ValueError):
            cls(n_clusters=0)

        with pytest.raises(ValueError):
            cls(n_clusters=5).fit(X)

        with pytest.raises(ValueError):
            cls(n_clusters=2).fit(np.empty((0, 2)))

        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError):
            cls(n_clusters=2).fit(X_nan)

        X_inf = X.copy()
        X_inf[1, 1] = np.inf
        with pytest.raises(ValueError):
            cls(n_clusters=2).fit(X_inf)

        fitted = cls(n_clusters=2, random_state=0).fit(X)
        with pytest.raises(ValueError):
            fitted.predict(np.zeros((2, 4)))
        with pytest.raises(ValueError):
            fitted.transform(np.zeros((2, 4)))

