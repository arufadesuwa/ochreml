"""
OchreML Wave 6 Tier P3 Manifold Learning Parity & Persistence Suite (＾▽＾)
Verifies TSNE and MDS against scikit-learn, plus .dit, pickle, and edge-case checks.
"""

import pickle
import numpy as np
import pytest
from sklearn import manifold as sk_manifold
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, silhouette_score

import ochreml
from ochreml.manifold import MDS, TSNE


def _check_dit_and_pickle(tmp_path, obj):
    expected = obj.embedding_
    dit_path = str(tmp_path / f"{type(obj).__name__}.dit")

    obj.save(dit_path)
    cls_loaded = type(obj).load(dit_path)
    np.testing.assert_allclose(cls_loaded.embedding_, expected, atol=1e-12)

    if hasattr(ochreml, "save") and hasattr(ochreml, "load"):
        try:
            ochreml.save(obj, dit_path)
            loaded = ochreml.load(dit_path)
            np.testing.assert_allclose(loaded.embedding_, expected, atol=1e-12)
        except Exception:
            pass

    pickled = pickle.loads(pickle.dumps(obj))
    np.testing.assert_allclose(pickled.embedding_, expected, atol=1e-12)


@pytest.mark.parametrize("init_mode", ["pca", "random"])
def test_tsne_cluster_preservation_and_parity(tmp_path, init_mode):
    X, y = make_blobs(n_samples=60, n_features=6, centers=3, cluster_std=0.6, random_state=42)

    oc = TSNE(
        n_components=2,
        perplexity=12.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=500,
        init=init_mode,
        random_state=42,
    )
    Z_oc = oc.fit_transform(X)

    sk = sk_manifold.TSNE(
        n_components=2,
        perplexity=12.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=500,
        init=init_mode,
        method="exact",
        random_state=42,
    )
    Z_sk = sk.fit_transform(X)

    assert Z_oc.shape == (60, 2)
    assert oc.n_features_in_ == 6
    assert oc.n_iter_ is not None and oc.n_iter_ >= 1
    assert oc.learning_rate_ == pytest.approx(max(60 / 12.0 / 4.0, 50.0))
    assert np.isfinite(oc.kl_divergence_)

    # Trustworthiness & silhouette cluster preservation
    trust_oc = sk_manifold.trustworthiness(X, Z_oc, n_neighbors=8)
    trust_sk = sk_manifold.trustworthiness(X, Z_sk, n_neighbors=8)
    assert trust_oc > 0.90
    assert abs(trust_oc - trust_sk) < 0.08

    sil_oc = silhouette_score(Z_oc, y)
    assert sil_oc > 0.65

    _check_dit_and_pickle(tmp_path, oc)


def test_mds_metric_and_nonmetric_parity(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(loc=0.0, scale=1.5, size=(30, 5))
    D = pairwise_distances(X, metric="euclidean")

    # 1. Metric MDS (euclidean)
    oc_m = MDS(n_components=2, metric=True, n_init=4, max_iter=300, eps=1e-4, random_state=42)
    Z_oc = oc_m.fit_transform(X)
    sk_m = sk_manifold.MDS(
        n_components=2,
        metric=True,
        n_init=4,
        max_iter=300,
        eps=1e-4,
        dissimilarity="euclidean",
        random_state=42,
    )
    _ = sk_m.fit_transform(X)

    assert Z_oc.shape == (30, 2)
    assert oc_m.dissimilarity_matrix_.shape == (30, 30)
    np.testing.assert_allclose(oc_m.dissimilarity_matrix_, D, atol=1e-10)
    assert oc_m.n_features_in_ == 5
    assert oc_m.stress_ == pytest.approx(sk_m.stress_, rel=0.15)
    _check_dit_and_pickle(tmp_path, oc_m)

    # 2. Metric MDS (precomputed)
    oc_pre = MDS(
        n_components=2,
        metric=True,
        n_init=4,
        max_iter=300,
        eps=1e-4,
        dissimilarity="precomputed",
        random_state=42,
    )
    Z_pre = oc_pre.fit_transform(D)
    assert Z_pre.shape == (30, 2)
    assert oc_pre.stress_ == pytest.approx(oc_m.stress_, rel=1e-5)

    # 3. Non-metric MDS (Kruskal isotonic regression)
    oc_nm = MDS(n_components=2, metric=False, n_init=4, max_iter=300, eps=1e-4, random_state=42)
    Z_nm = oc_nm.fit_transform(X)
    assert Z_nm.shape == (30, 2)
    assert np.isfinite(oc_nm.stress_) and oc_nm.stress_ >= 0.0
    trust_nm = sk_manifold.trustworthiness(X, Z_nm, n_neighbors=6)
    assert trust_nm > 0.85
    _check_dit_and_pickle(tmp_path, oc_nm)


def test_manifold_edge_cases_and_devices():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))

    for dev in ["cpu", "auto"]:
        tsne = TSNE(n_components=2, perplexity=5.0, max_iter=50, device=dev).fit(X)
        assert tsne.device_ is not None
        tsne.to("cpu")
        assert tsne.device_ == "cpu"

        mds = MDS(n_components=2, n_init=1, max_iter=50, device=dev).fit(X)
        assert mds.device_ is not None
        mds.to("cpu")
        assert mds.device_ == "cpu"

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        TSNE(perplexity=5.0).fit(X_nan)
    with pytest.raises(ValueError):
        MDS().fit(X_nan)

    X_inf = X.copy()
    X_inf[1, 2] = np.inf
    with pytest.raises(ValueError):
        TSNE(perplexity=5.0).fit(X_inf)
    with pytest.raises(ValueError):
        MDS().fit(X_inf)

    # Empty data
    with pytest.raises(ValueError):
        TSNE(perplexity=5.0).fit(np.empty((0, 4)))
    with pytest.raises(ValueError):
        MDS().fit(np.empty((0, 4)))

    # Invalid perplexity
    with pytest.raises(ValueError):
        TSNE(perplexity=0.0)
    with pytest.raises(ValueError):
        TSNE(perplexity=-5.0)
    with pytest.raises(ValueError):
        TSNE(perplexity=20.0).fit(X)  # perplexity >= n_samples

    # Invalid n_components == 0
    with pytest.raises(ValueError):
        TSNE(n_components=0)
    with pytest.raises(ValueError):
        MDS(n_components=0)

    # Invalid precomputed dissimilarity
    with pytest.raises(ValueError):
        MDS(dissimilarity="precomputed").fit(np.ones((5, 4)))  # non-square
    bad_sym = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        MDS(dissimilarity="precomputed").fit(bad_sym)  # asymmetric
    bad_neg = np.array([[0.0, -1.0], [-1.0, 0.0]])
    with pytest.raises(ValueError):
        MDS(dissimilarity="precomputed").fit(bad_neg)  # negative
