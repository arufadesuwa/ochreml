# OchreML Wave 4 Tier P2 Forest & Isolation Ensembles Tests (＾▽＾)
# Verifies RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
# ExtraTreesRegressor, and IsolationForest against scikit-learn, .dit/pickle
# persistence, edge cases, and device fallback.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier as SkExtraTreesClassifier,
    ExtraTreesRegressor as SkExtraTreesRegressor,
    IsolationForest as SkIsolationForest,
    RandomForestClassifier as SkRandomForestClassifier,
    RandomForestRegressor as SkRandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score

from ochreml.ensemble_forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_forest_classifiers_parity():
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test = X[:220], X[220:]
    y_train, y_test = y[:220], y[220:]

    for oc_cls, sk_cls in [
        (RandomForestClassifier, SkRandomForestClassifier),
        (ExtraTreesClassifier, SkExtraTreesClassifier),
    ]:
        for criterion in ("gini", "entropy"):
            sk = sk_cls(
                n_estimators=80, criterion=criterion, max_depth=8, random_state=42
            ).fit(X_train, y_train)
            oc = oc_cls(
                n_estimators=80,
                criterion=criterion,
                max_depth=8,
                random_state=42,
                device="cpu",
            ).fit(X_train, y_train)

            sk_acc = accuracy_score(y_test, sk.predict(X_test))
            oc_acc = oc.score(X_test, y_test)
            assert oc_acc >= sk_acc - 0.08
            assert oc_acc > 0.62

            proba = oc.predict_proba(X_test)
            sk_proba = sk.predict_proba(X_test)
            assert proba.shape == (len(X_test), 3)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
            corr = np.corrcoef(proba.ravel(), sk_proba.ravel())[0, 1]
            assert corr > 0.85

            assert oc.n_features_in_ == 8
            np.testing.assert_allclose(oc.classes_, [0.0, 1.0, 2.0])
            imp = oc.feature_importances_
            assert imp.shape == (8,)
            assert imp.sum() == pytest.approx(1.0, abs=1e-9)


def test_forest_classifiers_sample_weight_and_max_features():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=7
    )
    sw = np.linspace(0.5, 2.0, len(X))

    for mf in ("sqrt", "log2", 3, 0.5, None):
        oc = RandomForestClassifier(
            n_estimators=30, max_features=mf, random_state=7, device="auto"
        ).fit(X, y, sample_weight=sw)
        assert oc.score(X, y) > 0.85


def test_forest_regressors_parity():
    X, y = make_regression(
        n_samples=260, n_features=6, n_informative=5, noise=5.0, random_state=11
    )
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    sw = np.linspace(0.8, 1.5, len(X_train))

    for oc_cls, sk_cls in [
        (RandomForestRegressor, SkRandomForestRegressor),
        (ExtraTreesRegressor, SkExtraTreesRegressor),
    ]:
        sk = sk_cls(n_estimators=80, max_depth=10, random_state=11).fit(
            X_train, y_train, sample_weight=sw
        )
        oc = oc_cls(
            n_estimators=80, max_depth=10, random_state=11, device="cpu"
        ).fit(X_train, y_train, sample_weight=sw)

        sk_r2 = r2_score(y_test, sk.predict(X_test))
        oc_r2 = oc.score(X_test, y_test)
        assert oc_r2 >= sk_r2 - 0.08
        assert oc_r2 > 0.70

        preds = oc.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert oc.n_features_in_ == 6
        imp = oc.feature_importances_
        assert imp.shape == (6,)
        assert imp.sum() == pytest.approx(1.0, abs=1e-9)


def test_isolation_forest_outlier_separation():
    rng = np.random.default_rng(42)
    X_inliers, _ = make_blobs(
        n_samples=200, n_features=4, centers=[[0.0, 0.0, 0.0, 0.0]], cluster_std=0.5, random_state=42
    )
    X_outliers = rng.uniform(low=-6.0, high=6.0, size=(20, 4))
    X_outliers = np.where(np.abs(X_outliers) < 3.0, 4.5 * np.sign(X_outliers + 1e-3), X_outliers)
    X_all = np.vstack([X_inliers, X_outliers])

    sk = SkIsolationForest(
        n_estimators=100, contamination=0.1, random_state=42
    ).fit(X_all)
    oc = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42, device="cpu"
    ).fit(X_all)

    inlier_scores = oc.score_samples(X_inliers)
    outlier_scores = oc.score_samples(X_outliers)
    assert inlier_scores.mean() > outlier_scores.mean() + 0.08

    preds = oc.predict(X_all)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert (preds[:200] == 1).mean() > 0.90
    assert (preds[200:] == -1).mean() > 0.75

    dec = oc.decision_function(X_all)
    np.testing.assert_allclose(dec, oc.score_samples(X_all) - oc.offset_, atol=1e-12)
    assert np.corrcoef(oc.score_samples(X_all), sk.score_samples(X_all))[0, 1] > 0.90

    # Auto contamination and float/int max_samples
    oc_auto = IsolationForest(
        n_estimators=50, max_samples=0.5, contamination="auto", max_features=3, random_state=42
    ).fit(X_inliers)
    assert oc_auto.offset_ == pytest.approx(-0.5, abs=1e-12)
    assert oc_auto.max_samples_ == 100


def test_forest_persistence_dit_and_pickle(tmp_path):
    X, y_clf = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=9
    )
    y_reg = X[:, 0] * 2.0 - X[:, 1] + 0.5

    for cls, y_data, fname in [
        (RandomForestClassifier, y_clf, "rfc.dit"),
        (ExtraTreesClassifier, y_clf, "etc.dit"),
        (RandomForestRegressor, y_reg, "rfr.dit"),
        (ExtraTreesRegressor, y_reg, "etr.dit"),
    ]:
        model = cls(n_estimators=20, max_depth=5, random_state=9, device="cpu").fit(
            X, y_data
        )
        dit_path = tmp_path / fname
        model.save(str(dit_path))

        loaded = cls.load(str(dit_path), device="auto")
        np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-12)
        assert loaded.score(X, y_data) == pytest.approx(model.score(X, y_data), rel=1e-12)

        unpickled = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(unpickled.predict(X), model.predict(X), atol=1e-12)

        model.to("cpu")
        assert model.device_ == "cpu"

    iso = IsolationForest(n_estimators=25, random_state=9, device="cpu").fit(X)
    iso_path = tmp_path / "iso.dit"
    iso.save(str(iso_path))
    iso_loaded = IsolationForest.load(str(iso_path), device="auto")
    np.testing.assert_array_equal(iso_loaded.predict(X), iso.predict(X))
    np.testing.assert_allclose(
        iso_loaded.score_samples(X), iso.score_samples(X), atol=1e-12
    )
    iso_unpickled = pickle.loads(pickle.dumps(iso))
    np.testing.assert_array_equal(iso_unpickled.predict(X), iso.predict(X))


def test_forest_validation_and_error_handling():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])

    for cls in (
        RandomForestClassifier,
        ExtraTreesClassifier,
        RandomForestRegressor,
        ExtraTreesRegressor,
        IsolationForest,
    ):
        with pytest.raises(ValueError):
            cls(n_estimators=0)

        m = cls(n_estimators=5, random_state=1)
        with pytest.raises(ValueError):
            m.predict(X)

        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError):
            if cls is IsolationForest:
                m.fit(X_nan)
            else:
                m.fit(X_nan, y)

        X_inf = X.copy()
        X_inf[1, 1] = np.inf
        with pytest.raises(ValueError):
            if cls is IsolationForest:
                m.fit(X_inf)
            else:
                m.fit(X_inf, y)

        with pytest.raises(ValueError):
            if cls is IsolationForest:
                m.fit(np.empty((0, 2)))
            else:
                m.fit(np.empty((0, 2)), np.empty(0))

        # Shape mismatch on predict
        if cls is IsolationForest:
            m.fit(X)
        else:
            m.fit(X, y)
        with pytest.raises(ValueError):
            m.predict(np.ones((3, 5)))

    with pytest.raises(ValueError):
        RandomForestClassifier(criterion="invalid")
    with pytest.raises(ValueError):
        RandomForestRegressor(criterion="invalid")
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.8)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=0)
