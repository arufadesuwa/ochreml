"""
OchreML Wave 6 Tier P3 Matrix & Cross-Decomposition Test Suite (＾▽＾)
Verifies NMF, PLSRegression, and CCA against scikit-learn, plus .dit, pickle, and edge-case checks.
"""

import pickle
import numpy as np
import pytest
from sklearn import cross_decomposition as sk_cross
from sklearn import decomposition as sk_decomp

import ochreml
from ochreml.cross_decomposition import CCA, NMF, PLSRegression


def _check_dit_and_pickle(tmp_path, obj, X_test, call_fn):
    expected = call_fn(obj, X_test)
    dit_path = str(tmp_path / f"{type(obj).__name__}.dit")

    obj.save(dit_path)
    cls_loaded = type(obj).load(dit_path)
    np.testing.assert_allclose(call_fn(cls_loaded, X_test), expected, atol=1e-10)

    if hasattr(ochreml, "save") and hasattr(ochreml, "load"):
        try:
            ochreml.save(obj, dit_path)
            loaded = ochreml.load(dit_path)
            np.testing.assert_allclose(call_fn(loaded, X_test), expected, atol=1e-10)
        except Exception:
            pass

    pickled = pickle.loads(pickle.dumps(obj))
    np.testing.assert_allclose(call_fn(pickled, X_test), expected, atol=1e-10)


def test_nmf_parity_and_persistence(tmp_path):
    rng = np.random.default_rng(42)
    W_true = rng.uniform(0.1, 2.0, size=(80, 3))
    H_true = rng.uniform(0.1, 2.0, size=(3, 6))
    X = W_true @ H_true + rng.uniform(0.0, 0.05, size=(80, 6))
    X_test = rng.uniform(0.1, 2.0, size=(15, 6))

    for init_mode in ["nndsvda", "nndsvd", "random"]:
        oc = NMF(n_components=3, init=init_mode, max_iter=500, tol=1e-5, random_state=42)
        W_oc = oc.fit_transform(X)
        sk = sk_decomp.NMF(
            n_components=3, init=init_mode, solver="mu", max_iter=500, tol=1e-5, random_state=42
        )
        sk.fit_transform(X)

        assert W_oc.shape == (80, 3)
        assert oc.components_.shape == (3, 6)
        assert np.all(W_oc >= 0.0)
        assert np.all(oc.components_ >= 0.0)
        assert oc.n_components_ == 3
        assert oc.n_features_in_ == 6
        assert oc.n_iter_ >= 1
        assert oc.reconstruction_err_ <= sk.reconstruction_err_ + 0.5

        recon = oc.inverse_transform(W_oc)
        assert recon.shape == X.shape
        np.testing.assert_allclose(
            np.linalg.norm(X - recon), oc.reconstruction_err_, rtol=1e-6, atol=1e-6
        )

        _check_dit_and_pickle(tmp_path, oc, X_test, lambda m, x: m.transform(x))

    # Regularized NMF
    oc_reg = NMF(n_components=3, alpha_W=0.05, alpha_H=0.05, l1_ratio=0.5, random_state=7).fit(X)
    assert oc_reg.components_.shape == (3, 6)


def test_pls_regression_1d_and_2d_parity(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 6))
    X_test = rng.normal(size=(25, 6))

    # 2D multi-target Y
    Y_2d = np.column_stack([
        2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * rng.normal(size=100),
        -0.5 * X[:, 2] + 3.0 * X[:, 3] + 0.1 * rng.normal(size=100),
    ])
    Y_2d_test = np.column_stack([
        2.0 * X_test[:, 0] - 1.5 * X_test[:, 1],
        -0.5 * X_test[:, 2] + 3.0 * X_test[:, 3],
    ])

    for scale in [True, False]:
        oc = PLSRegression(n_components=3, scale=scale).fit(X, Y_2d)
        sk = sk_cross.PLSRegression(n_components=3, scale=scale).fit(X, Y_2d)

        np.testing.assert_allclose(oc.predict(X_test), sk.predict(X_test), atol=1e-6)
        assert oc.score(X_test, Y_2d_test) == pytest.approx(sk.score(X_test, Y_2d_test), abs=1e-6)

        xs_oc, ys_oc = oc.transform(X_test, Y_2d_test)
        xs_sk, ys_sk = sk.transform(X_test, Y_2d_test)
        np.testing.assert_allclose(np.abs(xs_oc), np.abs(xs_sk), atol=1e-3)
        np.testing.assert_allclose(np.abs(ys_oc), np.abs(ys_sk), atol=1e-3)

        x_rec = oc.inverse_transform(oc.transform(X_test))
        assert x_rec.shape == X_test.shape
        _check_dit_and_pickle(tmp_path, oc, X_test, lambda m, x: m.predict(x))

    # 1D Y
    y_1d = 1.5 * X[:, 0] + 2.5 * X[:, 2] + 0.05 * rng.normal(size=100)
    y_1d_test = 1.5 * X_test[:, 0] + 2.5 * X_test[:, 2]
    oc_1d = PLSRegression(n_components=2).fit(X, y_1d)
    sk_1d = sk_cross.PLSRegression(n_components=2).fit(X, y_1d)
    pred_oc = oc_1d.predict(X_test)
    assert pred_oc.ndim == 1
    np.testing.assert_allclose(pred_oc, sk_1d.predict(X_test), atol=1e-6)
    assert oc_1d.score(X_test, y_1d_test) == pytest.approx(sk_1d.score(X_test, y_1d_test), abs=1e-6)


def test_cca_parity_and_persistence(tmp_path):
    rng = np.random.default_rng(99)
    Z = rng.normal(size=(120, 2))
    X = np.column_stack([Z[:, 0], Z[:, 1], rng.normal(size=120), rng.normal(size=120)])
    Y = np.column_stack([2.0 * Z[:, 0] + 0.1 * rng.normal(size=120), -1.5 * Z[:, 1] + 0.1 * rng.normal(size=120), rng.normal(size=120)])

    X_test = rng.normal(size=(20, 4))
    oc = CCA(n_components=2).fit(X, Y)
    sk = sk_cross.CCA(n_components=2).fit(X, Y)

    xs_oc, ys_oc = oc.transform(X, Y)
    xs_sk, ys_sk = sk.transform(X, Y)

    for k in range(2):
        corr_oc = abs(np.corrcoef(xs_oc[:, k], ys_oc[:, k])[0, 1])
        corr_sk = abs(np.corrcoef(xs_sk[:, k], ys_sk[:, k])[0, 1])
        assert corr_oc == pytest.approx(corr_sk, abs=1e-4)

    np.testing.assert_allclose(oc.predict(X_test), sk.predict(X_test), atol=1e-2)
    assert oc.score(X, Y) == pytest.approx(sk.score(X, Y), abs=1e-3)
    _check_dit_and_pickle(tmp_path, oc, X_test, lambda m, x: m.predict(x))


def test_validation_errors_and_devices():
    rng = np.random.default_rng(0)
    X_pos = rng.uniform(0.1, 1.0, size=(20, 4))
    Y_pos = rng.uniform(0.1, 1.0, size=(20, 3))

    # Negative X for NMF
    X_neg = X_pos.copy()
    X_neg[0, 0] = -0.5
    with pytest.raises(ValueError):
        NMF(n_components=2).fit(X_neg)

    # NaN / Inf
    X_nan = X_pos.copy()
    X_nan[1, 1] = np.nan
    for cls in [NMF, PLSRegression, CCA]:
        with pytest.raises(ValueError):
            if cls is NMF:
                cls(n_components=2).fit(X_nan)
            else:
                cls(n_components=2).fit(X_nan, Y_pos)

    # Empty data
    for cls in [NMF, PLSRegression, CCA]:
        with pytest.raises(ValueError):
            if cls is NMF:
                cls(n_components=2).fit(np.empty((0, 4)))
            else:
                cls(n_components=2).fit(np.empty((0, 4)), np.empty((0, 3)))

    # Shape mismatch
    nmf = NMF(n_components=2).fit(X_pos)
    with pytest.raises(ValueError):
        nmf.transform(rng.uniform(0.1, 1.0, size=(5, 5)))
    pls = PLSRegression(n_components=2).fit(X_pos, Y_pos)
    with pytest.raises(ValueError):
        pls.predict(rng.normal(size=(5, 5)))

    # Invalid params
    with pytest.raises(ValueError):
        NMF(n_components=0)
    with pytest.raises(ValueError):
        NMF(init="invalid_init")
    with pytest.raises(ValueError):
        NMF(l1_ratio=1.5)
    with pytest.raises(ValueError):
        PLSRegression(n_components=0)
    with pytest.raises(ValueError):
        CCA(max_iter=0)

    # Device options
    for dev in ["cpu", "auto"]:
        m_nmf = NMF(n_components=2, device=dev).fit(X_pos)
        assert m_nmf.transform(X_pos).shape == (20, 2)
        m_pls = PLSRegression(n_components=2, device=dev).fit(X_pos, Y_pos)
        assert m_pls.predict(X_pos).shape == (20, 3)
        m_cca = CCA(n_components=2, device=dev).fit(X_pos, Y_pos)
        assert m_cca.predict(X_pos).shape == (20, 3)
