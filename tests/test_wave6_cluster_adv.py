# OchreML Wave 6 Advanced Clustering & Outlier Detection Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, device handling, and edge-case validation.

import pickle
import numpy as np
import pytest
from sklearn.cluster import (
    DBSCAN as SkDBSCAN,
    OPTICS as SkOPTICS,
    SpectralClustering as SkSpectralClustering,
    AgglomerativeClustering as SkAgglomerativeClustering,
)
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor as SkLocalOutlierFactor

from ochreml.cluster_advanced import (
    DBSCAN,
    OPTICS,
    SpectralClustering,
    AgglomerativeClustering,
    LocalOutlierFactor,
)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_dbscan_parity(metric, device):
    X, _ = make_blobs(n_samples=90, centers=3, n_features=4, cluster_std=0.4, random_state=42)
    X_with_noise = np.vstack([X, np.array([[25.0, 25.0, 25.0, 25.0]])])
    eps = 0.2 if metric == "cosine" else 2.5

    sk = SkDBSCAN(eps=eps, min_samples=4, metric=metric)
    sk_labels = sk.fit_predict(X_with_noise)

    oc = DBSCAN(eps=eps, min_samples=4, metric=metric, device=device)
    oc_labels = oc.fit_predict(X_with_noise)

    assert adjusted_rand_score(sk_labels, oc_labels) == pytest.approx(1.0, abs=1e-6)
    assert oc_labels[-1] == -1
    np.testing.assert_array_equal(oc.core_sample_indices_, sk.core_sample_indices_)
    np.testing.assert_allclose(oc.components_, sk.components_, atol=1e-12)
    assert oc.n_features_in_ == 4


def test_dbscan_zero_core_samples():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    oc = DBSCAN(eps=0.5, min_samples=3, device="cpu").fit(X)
    np.testing.assert_array_equal(oc.labels_, np.array([-1, -1, -1]))
    assert len(oc.core_sample_indices_) == 0
    assert oc.components_.shape == (0, 2)


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_optics_parity(device):
    X, y = make_blobs(n_samples=90, centers=3, n_features=3, cluster_std=0.25, random_state=7)
    sk = SkOPTICS(min_samples=5, xi=0.05, min_cluster_size=15)
    sk_labels = sk.fit_predict(X)

    oc = OPTICS(min_samples=5, xi=0.05, min_cluster_size=15, device=device)
    oc_labels = oc.fit_predict(X)

    assert oc.ordering_.shape == (90,)
    assert oc.reachability_.shape == (90,)
    assert oc.core_distances_.shape == (90,)
    assert np.isinf(oc.reachability_[oc.ordering_[0]])
    np.testing.assert_allclose(oc.core_distances_, sk.core_distances_, atol=1e-10)
    assert adjusted_rand_score(y, oc_labels) >= 0.85
    assert adjusted_rand_score(sk_labels, oc_labels) >= 0.85


@pytest.mark.parametrize("affinity", ["rbf", "nearest_neighbors"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_spectral_clustering_parity(affinity, device):
    X, y = make_moons(n_samples=80, noise=0.05, random_state=42)
    gamma = 15.0 if affinity == "rbf" else 1.0

    sk = SkSpectralClustering(
        n_clusters=2, affinity=affinity, gamma=gamma, n_neighbors=5, random_state=42
    )
    sk_labels = sk.fit_predict(X)

    oc = SpectralClustering(
        n_clusters=2,
        affinity=affinity,
        gamma=gamma,
        n_neighbors=5,
        n_init=10,
        random_state=42,
        device=device,
    )
    oc_labels = oc.fit_predict(X)

    assert oc.affinity_matrix_.shape == (80, 80)
    assert oc.n_features_in_ == 2
    assert adjusted_rand_score(sk_labels, oc_labels) >= 0.85
    assert adjusted_rand_score(y, oc_labels) >= 0.85


@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_agglomerative_clustering_parity(linkage, device):
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, cluster_std=0.5, random_state=12)

    sk = SkAgglomerativeClustering(n_clusters=3, metric="euclidean", linkage=linkage)
    sk_labels = sk.fit_predict(X)

    oc = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage=linkage, device=device)
    oc_labels = oc.fit_predict(X)

    assert adjusted_rand_score(sk_labels, oc_labels) == pytest.approx(1.0, abs=1e-6)
    assert oc.n_leaves_ == 60
    assert oc.n_connected_components_ == 1
    assert oc.children_.shape == (59, 2)
    assert oc.n_features_in_ == 4


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_local_outlier_factor_outlier_and_novelty(device):
    rng = np.random.RandomState(42)
    X_inliers = 0.3 * rng.randn(60, 2)
    X_outliers = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]])
    X = np.vstack([X_inliers, X_outliers])

    # novelty=False
    sk = SkLocalOutlierFactor(n_neighbors=10, contamination="auto", novelty=False)
    sk_pred = sk.fit_predict(X)

    oc = LocalOutlierFactor(n_neighbors=10, contamination="auto", novelty=False, device=device)
    oc_pred = oc.fit_predict(X)

    np.testing.assert_array_equal(oc_pred, sk_pred)
    np.testing.assert_allclose(oc.negative_outlier_factor_, sk.negative_outlier_factor_, atol=1e-9)
    assert oc.offset_ == pytest.approx(-1.5)
    assert oc.n_neighbors_ == 10

    # novelty=True
    sk_nov = SkLocalOutlierFactor(n_neighbors=10, contamination=0.1, novelty=True).fit(X_inliers)
    oc_nov = LocalOutlierFactor(n_neighbors=10, contamination=0.1, novelty=True, device=device).fit(
        X_inliers
    )

    X_test = np.vstack([0.2 * rng.randn(10, 2), X_outliers])
    np.testing.assert_allclose(oc_nov.score_samples(X_test), sk_nov.score_samples(X_test), atol=1e-9)
    np.testing.assert_allclose(
        oc_nov.decision_function(X_test), sk_nov.decision_function(X_test), atol=1e-9
    )
    np.testing.assert_array_equal(oc_nov.predict(X_test), sk_nov.predict(X_test))


def test_dit_and_pickle_roundtrip(tmp_path):
    X, _ = make_blobs(n_samples=40, centers=2, n_features=3, cluster_std=0.4, random_state=5)
    X_nov = np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]])

    models = [
        (DBSCAN(eps=1.5, min_samples=3, device="cpu").fit(X), DBSCAN),
        (OPTICS(min_samples=3, device="cpu").fit(X), OPTICS),
        (SpectralClustering(n_clusters=2, random_state=42, device="cpu").fit(X), SpectralClustering),
        (AgglomerativeClustering(n_clusters=2, linkage="ward", device="cpu").fit(X), AgglomerativeClustering),
        (LocalOutlierFactor(n_neighbors=5, novelty=True, device="cpu").fit(X), LocalOutlierFactor),
    ]

    for idx, (model, cls) in enumerate(models):
        path = str(tmp_path / f"model_{idx}.dit")
        model.save(path)
        loaded = cls.load(path, device="cpu")
        unpickled = pickle.loads(pickle.dumps(model))

        if isinstance(model, LocalOutlierFactor):
            np.testing.assert_allclose(
                loaded.negative_outlier_factor_, model.negative_outlier_factor_, atol=1e-12
            )
            np.testing.assert_array_equal(loaded.predict(X_nov), model.predict(X_nov))
            np.testing.assert_array_equal(unpickled.predict(X_nov), model.predict(X_nov))
        else:
            np.testing.assert_array_equal(loaded.labels_, model.labels_)
            np.testing.assert_array_equal(unpickled.labels_, model.labels_)


def test_validation_and_edge_cases():
    X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    X_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
    X_empty = np.empty((0, 2), dtype=np.float64)

    for cls in [DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering, LocalOutlierFactor]:
        est = cls()
        with pytest.raises(ValueError):
            est.fit(X_nan)
        with pytest.raises(ValueError):
            est.fit(X_inf)
        with pytest.raises(ValueError):
            est.fit(X_empty)

    with pytest.raises(ValueError):
        DBSCAN(eps=-0.1)
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)
    with pytest.raises(ValueError):
        OPTICS(xi=1.5)
    with pytest.raises(ValueError):
        SpectralClustering(affinity="invalid")
    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="ward", metric="manhattan")
    with pytest.raises(ValueError):
        LocalOutlierFactor(contamination=0.8)

    lof_no_nov = LocalOutlierFactor(n_neighbors=3, novelty=False).fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    )
    with pytest.raises(ValueError):
        lof_no_nov.predict(np.array([[0.5, 0.5]]))

    lof_nov = LocalOutlierFactor(n_neighbors=3, novelty=True)
    with pytest.raises(ValueError):
        lof_nov.fit_predict(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
