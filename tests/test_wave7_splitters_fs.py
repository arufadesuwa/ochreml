"""
OchreML Wave 7 Test Suite: RBFSampler, Feature Selection & Model Selection Splitters.
Verifies parity against scikit-learn, .dit persistence, pickle, get_params/set_params, and edge cases.
"""

import pickle
import numpy as np
import pytest
from sklearn import feature_selection as sk_fs
from sklearn import kernel_approximation as sk_ka
from sklearn import model_selection as sk_ms

import ochreml
from ochreml.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
)
from ochreml.kernel_approximation import RBFSampler
from ochreml.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)


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


def test_rbf_sampler_parity_and_persistence(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 6))
    X_test = rng.normal(size=(30, 6))

    for gamma in [0.5, "scale"]:
        oc = RBFSampler(gamma=gamma, n_components=400, random_state=42)
        Z_oc = oc.fit_transform(X)
        assert Z_oc.shape == (120, 400)
        assert oc.random_weights_.shape == (6, 400)
        assert oc.random_offset_.shape == (400,)
        assert oc.n_features_in_ == 6

        # Kernel approximation check against exact RBF kernel
        gamma_val = oc.gamma_
        diff = X_test[:10, None, :] - X_test[None, :10, :]
        K_exact = np.exp(-gamma_val * np.sum(diff * diff, axis=-1))
        Z_test = oc.transform(X_test[:10])
        K_approx = Z_test @ Z_test.T
        np.testing.assert_allclose(K_approx, K_exact, atol=0.12)

        _check_dit_and_pickle(tmp_path, oc, X_test, lambda m, x: m.transform(x))

    # get_params / set_params
    params = oc.get_params()
    assert params["n_components"] == 400
    oc.set_params(n_components=50, gamma=0.25)
    assert oc.n_components == 50
    assert oc.gamma == 0.25
    assert oc.fit_transform(X).shape == (120, 50)


def test_statistical_tests_parity():
    rng = np.random.default_rng(7)
    X = rng.uniform(0.5, 5.0, size=(80, 6))
    y_reg = 3.0 * X[:, 0] - 2.0 * X[:, 2] + 0.1 * rng.normal(size=80)
    y_cls = (X[:, 0] + X[:, 1] > 5.5).astype(int)

    # f_regression
    f_oc, p_oc = f_regression(X, y_reg)
    f_sk, p_sk = sk_fs.f_regression(X, y_reg)
    np.testing.assert_allclose(f_oc, f_sk, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p_oc, p_sk, rtol=1e-6, atol=1e-8)

    f_oc_nc, p_oc_nc = f_regression(X, y_reg, center=False)
    f_sk_nc, p_sk_nc = sk_fs.f_regression(X, y_reg, center=False)
    np.testing.assert_allclose(f_oc_nc, f_sk_nc, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p_oc_nc, p_sk_nc, rtol=1e-6, atol=1e-8)

    # f_classif
    fc_oc, pc_oc = f_classif(X, y_cls)
    fc_sk, pc_sk = sk_fs.f_classif(X, y_cls)
    np.testing.assert_allclose(fc_oc, fc_sk, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pc_oc, pc_sk, rtol=1e-6, atol=1e-8)

    # chi2
    c2_oc, pc2_oc = chi2(X, y_cls)
    c2_sk, pc2_sk = sk_fs.chi2(X, y_cls)
    np.testing.assert_allclose(c2_oc, c2_sk, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pc2_oc, pc2_sk, rtol=1e-6, atol=1e-8)

    # chi2 negative input check
    X_neg = X.copy()
    X_neg[0, 0] = -1.0
    with pytest.raises(ValueError):
        chi2(X_neg, y_cls)


def _custom_score(X_in, _y_in):
    scores = np.arange(X_in.shape[1], dtype=np.float64)
    pvals = np.ones(X_in.shape[1], dtype=np.float64) * 0.05
    return scores, pvals


def test_feature_selectors_parity_and_persistence(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 4.0, size=(100, 6))
    X[:, 2] = 1.5  # zero variance column
    X_test = rng.uniform(0.0, 4.0, size=(20, 6))
    X_test[:, 2] = 1.5
    y_cls = (X[:, 0] * 2.0 + X[:, 4] > 5.0).astype(int)

    # VarianceThreshold
    vt_oc = VarianceThreshold(threshold=0.0)
    vt_sk = sk_fs.VarianceThreshold(threshold=0.0)
    X_vt_oc = vt_oc.fit_transform(X)
    X_vt_sk = vt_sk.fit_transform(X)
    np.testing.assert_allclose(X_vt_oc, X_vt_sk, atol=1e-10)
    np.testing.assert_allclose(vt_oc.variances_, vt_sk.variances_, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(vt_oc.get_support(), vt_sk.get_support())
    np.testing.assert_array_equal(vt_oc.get_support(indices=True), vt_sk.get_support(indices=True))
    inv_vt = vt_oc.inverse_transform(X_vt_oc)
    assert inv_vt.shape == X.shape
    _check_dit_and_pickle(tmp_path, vt_oc, X_test, lambda m, x: m.transform(x))

    # SelectKBest
    skb_oc = SelectKBest(score_func=f_classif, k=3)
    skb_sk = sk_fs.SelectKBest(score_func=sk_fs.f_classif, k=3)
    X_skb_oc = skb_oc.fit_transform(X[:, [0, 1, 3, 4, 5]], y_cls)
    X_skb_sk = skb_sk.fit_transform(X[:, [0, 1, 3, 4, 5]], y_cls)
    np.testing.assert_allclose(X_skb_oc, X_skb_sk, atol=1e-10)
    np.testing.assert_allclose(skb_oc.scores_, skb_sk.scores_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(skb_oc.pvalues_, skb_sk.pvalues_, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(skb_oc.get_support(), skb_sk.get_support())
    assert skb_oc.inverse_transform(X_skb_oc).shape == (100, 5)
    _check_dit_and_pickle(tmp_path, skb_oc, X_test[:, [0, 1, 3, 4, 5]], lambda m, x: m.transform(x))

    # Custom Python score_func
    skb_custom = SelectKBest(score_func=_custom_score, k=2).fit(X, y_cls)
    np.testing.assert_array_equal(skb_custom.get_support(indices=True), np.array([4, 5]))
    pickled_custom = pickle.loads(pickle.dumps(skb_custom))
    np.testing.assert_allclose(pickled_custom.transform(X_test), skb_custom.transform(X_test))

    # SelectPercentile
    sp_oc = SelectPercentile(score_func=f_classif, percentile=40)
    sp_sk = sk_fs.SelectPercentile(score_func=sk_fs.f_classif, percentile=40)
    X_sp_oc = sp_oc.fit_transform(X[:, [0, 1, 3, 4, 5]], y_cls)
    X_sp_sk = sp_sk.fit_transform(X[:, [0, 1, 3, 4, 5]], y_cls)
    np.testing.assert_allclose(X_sp_oc, X_sp_sk, atol=1e-10)
    np.testing.assert_array_equal(sp_oc.get_support(), sp_sk.get_support())
    _check_dit_and_pickle(tmp_path, sp_oc, X_test[:, [0, 1, 3, 4, 5]], lambda m, x: m.transform(x))


def test_model_selection_splitters():
    X = np.arange(40).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])

    # KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    assert kf.get_n_splits() == 5
    folds = list(kf.split(X))
    assert len(folds) == 5
    all_test = np.concatenate([te for _, te in folds])
    assert set(all_test.tolist()) == set(range(20))

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert skf.get_n_splits() == 5
    sfolds = list(skf.split(X, y))
    assert len(sfolds) == 5
    for tr, te in sfolds:
        assert len(te) == 4
        assert np.sum(y[te] == 0) == 2
        assert np.sum(y[te] == 1) == 2
        assert len(set(tr.tolist()) & set(te.tolist())) == 0

    # GroupKFold
    gkf = GroupKFold(n_splits=5)
    assert gkf.get_n_splits() == 5
    gfolds = list(gkf.split(X, y, groups))
    assert len(gfolds) == 5
    for tr, te in gfolds:
        assert set(groups[tr].tolist()).isdisjoint(set(groups[te].tolist()))

    # TimeSeriesSplit
    tss_oc = TimeSeriesSplit(n_splits=4, max_train_size=6, test_size=3, gap=1)
    tss_sk = sk_ms.TimeSeriesSplit(n_splits=4, max_train_size=6, test_size=3, gap=1)
    for (tr_oc, te_oc), (tr_sk, te_sk) in zip(tss_oc.split(X), tss_sk.split(X)):
        np.testing.assert_array_equal(tr_oc, tr_sk)
        np.testing.assert_array_equal(te_oc, te_sk)

    # train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    assert X_tr.shape == (14, 2)
    assert X_te.shape == (6, 2)
    assert np.sum(y_te == 0) == 3
    assert np.sum(y_te == 1) == 3

    # List slicing support in train_test_split
    items = [f"item_{i}" for i in range(20)]
    items_tr, items_te = train_test_split(items, test_size=5, shuffle=False)
    assert items_tr == items[:15]
    assert items_te == items[15:]
