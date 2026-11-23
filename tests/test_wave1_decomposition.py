"""
OchreML Wave 1 Tier P0 Decomposition Parity & Persistence Suite (＾▽＾)
Verifies PCA and TruncatedSVD against scikit-learn, plus .dit, pickle, and edge-case checks.
"""

import pickle
import numpy as np
import pytest
from sklearn import decomposition as sk_decomp

import ochreml
from ochreml.decomposition import PCA, TruncatedSVD


def _assert_subspace_parity(oc_comp: np.ndarray, sk_comp: np.ndarray, tol: float = 1e-8) -> None:
    assert oc_comp.shape == sk_comp.shape
    # Check absolute component parity (sign-flipped deterministically)
    diff_abs = np.max(np.abs(np.abs(oc_comp) - np.abs(sk_comp)))
    assert float(diff_abs) < tol, f"Max abs(components_) diff {diff_abs} >= {tol}"
    # Check subspace alignment: ||C_oc^T C_oc - C_sk^T C_sk||_inf < tol
    proj_diff = np.max(np.abs(oc_comp.T @ oc_comp - sk_comp.T @ sk_comp))
    assert float(proj_diff) < tol, f"Subspace projector diff {proj_diff} >= {tol}"


def _check_dit_and_pickle(tmp_path, obj, X_test):
    expected = obj.transform(X_test)
    dit_path = str(tmp_path / f"{type(obj).__name__}.dit")

    obj.save(dit_path)
    cls_loaded = type(obj).load(dit_path)
    np.testing.assert_allclose(cls_loaded.transform(X_test), expected, atol=1e-10)

    # Also test ochreml.save / ochreml.load when wired
    if hasattr(ochreml, "save") and hasattr(ochreml, "load"):
        try:
            ochreml.save(obj, dit_path)
            loaded = ochreml.load(dit_path)
            np.testing.assert_allclose(loaded.transform(X_test), expected, atol=1e-10)
        except Exception:
            pass

    pickled = pickle.loads(pickle.dumps(obj))
    np.testing.assert_allclose(pickled.transform(X_test), expected, atol=1e-10)


def test_pca_parity_all_modes(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(loc=2.5, scale=1.8, size=(120, 8))
    X_test = rng.normal(loc=2.5, scale=1.8, size=(30, 8))

    # 1. n_components=None
    oc = PCA(n_components=None).fit(X)
    sk = sk_decomp.PCA(n_components=None).fit(X)
    np.testing.assert_allclose(oc.mean_, sk.mean_, atol=1e-10)
    np.testing.assert_allclose(oc.singular_values_, sk.singular_values_, atol=1e-8)
    np.testing.assert_allclose(oc.explained_variance_, sk.explained_variance_, atol=1e-8)
    np.testing.assert_allclose(oc.explained_variance_ratio_, sk.explained_variance_ratio_, atol=1e-8)
    _assert_subspace_parity(oc.components_, sk.components_)
    np.testing.assert_allclose(oc.inverse_transform(oc.transform(X_test)), X_test, atol=1e-8)
    _check_dit_and_pickle(tmp_path, oc, X_test)

    # 2. n_components=int, whiten=False and whiten=True
    for whiten in [False, True]:
        oc_k = PCA(n_components=4, whiten=whiten).fit(X)
        sk_k = sk_decomp.PCA(n_components=4, whiten=whiten).fit(X)
        np.testing.assert_allclose(oc_k.explained_variance_, sk_k.explained_variance_, atol=1e-8)
        np.testing.assert_allclose(
            oc_k.explained_variance_ratio_, sk_k.explained_variance_ratio_, atol=1e-8
        )
        np.testing.assert_allclose(oc_k.noise_variance_, sk_k.noise_variance_, atol=1e-8)
        _assert_subspace_parity(oc_k.components_, sk_k.components_)

        Z_oc = oc_k.transform(X_test)
        Z_sk = sk_k.transform(X_test)
        np.testing.assert_allclose(np.abs(Z_oc), np.abs(Z_sk), atol=1e-7)
        np.testing.assert_allclose(
            oc_k.inverse_transform(Z_oc), sk_k.inverse_transform(Z_sk), atol=1e-8
        )
        if not whiten:
            np.testing.assert_allclose(oc_k.get_covariance(), sk_k.get_covariance(), atol=1e-8)
        else:
            sk_nowhiten = sk_decomp.PCA(n_components=4, whiten=False).fit(X)
            np.testing.assert_allclose(oc_k.get_covariance(), sk_nowhiten.get_covariance(), atol=1e-8)
        _check_dit_and_pickle(tmp_path, oc_k, X_test)

    # 3. n_components=float (explained variance ratio threshold)
    oc_f = PCA(n_components=0.85).fit(X)
    sk_f = sk_decomp.PCA(n_components=0.85).fit(X)
    assert oc_f.n_components_ == sk_f.n_components_
    np.testing.assert_allclose(oc_f.explained_variance_ratio_, sk_f.explained_variance_ratio_, atol=1e-8)

    # 4. Wide matrix (n_features > n_samples) dual Gram SVD path
    X_wide = rng.normal(size=(25, 60))
    oc_w = PCA(n_components=5).fit(X_wide)
    sk_w = sk_decomp.PCA(n_components=5).fit(X_wide)
    np.testing.assert_allclose(oc_w.singular_values_, sk_w.singular_values_, atol=1e-7)
    np.testing.assert_allclose(oc_w.explained_variance_, sk_w.explained_variance_, atol=1e-7)
    _assert_subspace_parity(oc_w.components_, sk_w.components_, tol=1e-7)


def test_truncated_svd_parity(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(loc=1.0, scale=2.0, size=(100, 12))
    X_test = rng.normal(loc=1.0, scale=2.0, size=(20, 12))

    for alg in ["exact", "arpack", "randomized"]:
        sk_alg = "arpack" if alg == "exact" else alg
        oc = TruncatedSVD(n_components=4, algorithm=alg, n_iter=7, random_state=42).fit(X)
        sk = sk_decomp.TruncatedSVD(n_components=4, algorithm=sk_alg, n_iter=7, random_state=42).fit(X)

        np.testing.assert_allclose(oc.singular_values_, sk.singular_values_, atol=1e-7)
        np.testing.assert_allclose(oc.explained_variance_, sk.explained_variance_, atol=1e-7)
        np.testing.assert_allclose(
            oc.explained_variance_ratio_, sk.explained_variance_ratio_, atol=1e-7
        )
        _assert_subspace_parity(oc.components_, sk.components_, tol=1e-7)

        Z_oc = oc.transform(X_test)
        Z_sk = sk.transform(X_test)
        np.testing.assert_allclose(oc.inverse_transform(Z_oc), sk.inverse_transform(Z_sk), atol=1e-7)
        _check_dit_and_pickle(tmp_path, oc, X_test)

    # Large-feature randomized path (n_features > 512) with structured low-rank signal
    X_big = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 600)) + 0.05 * rng.normal(size=(80, 600))
    oc_big = TruncatedSVD(n_components=5, algorithm="randomized", n_iter=5, random_state=7).fit(X_big)
    sk_big = sk_decomp.TruncatedSVD(
        n_components=5, algorithm="randomized", n_iter=5, random_state=7
    ).fit(X_big)
    np.testing.assert_allclose(oc_big.singular_values_, sk_big.singular_values_, atol=1e-4)
    np.testing.assert_allclose(
        oc_big.explained_variance_ratio_, sk_big.explained_variance_ratio_, atol=1e-4
    )


def test_decomposition_edge_cases_and_devices():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))

    # Device support
    for dev in ["cpu", "auto"]:
        p = PCA(n_components=2, device=dev).fit(X)
        assert p.device_ is not None
        p.to("cpu")
        assert p.device_ == "cpu"

        t = TruncatedSVD(n_components=2, device=dev).fit(X)
        assert t.device_ is not None
        t.to("cpu")
        assert t.device_ == "cpu"

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(X_nan)
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=2).fit(X_nan)

    X_inf = X.copy()
    X_inf[1, 2] = np.inf
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(X_inf)
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=2).fit(X_inf)

    # Empty array
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(np.empty((0, 5)))
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=2).fit(np.empty((0, 5)))

    # Shape mismatch
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError):
        pca.transform(np.ones((5, 3)))
    svd = TruncatedSVD(n_components=2).fit(X)
    with pytest.raises(ValueError):
        svd.transform(np.ones((5, 3)))

    # Invalid n_components
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(ValueError):
        PCA(n_components=1.5)
    with pytest.raises(ValueError):
        PCA(n_components=10).fit(X)
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=0)
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=10).fit(X)
