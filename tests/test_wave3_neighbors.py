# OchreML Wave 3 Neighbors Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, and edge-case validation.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import (
    KNeighborsClassifier as SkKNeighborsClassifier,
    KNeighborsRegressor as SkKNeighborsRegressor,
    RadiusNeighborsClassifier as SkRadiusNeighborsClassifier,
    RadiusNeighborsRegressor as SkRadiusNeighborsRegressor,
)

from ochreml.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_kneighbors_classifier_parity(weights, metric):
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    sk = SkKNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric)
    sk.fit(X_train, y_train)

    oc = KNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric, device="cpu")
    oc.fit(X_train, y_train)

    np.testing.assert_array_equal(oc.classes_, sk.classes_)
    np.testing.assert_allclose(oc.predict_proba(X_test), sk.predict_proba(X_test), atol=1e-12)
    np.testing.assert_array_equal(oc.predict(X_test), sk.predict(X_test))
    assert oc.score(X_test, y_test) == pytest.approx(sk.score(X_test, y_test), abs=1e-12)

    sk_d, sk_i = sk.kneighbors(X_test, n_neighbors=3, return_distance=True)
    oc_d, oc_i = oc.kneighbors(X_test, n_neighbors=3, return_distance=True)
    np.testing.assert_allclose(oc_d, sk_d, atol=1e-10)
    np.testing.assert_array_equal(oc_i, sk_i)

    # Self-query (X=None)
    sk_sd, sk_si = sk.kneighbors(None, n_neighbors=3, return_distance=True)
    oc_sd, oc_si = oc.kneighbors(None, n_neighbors=3, return_distance=True)
    np.testing.assert_allclose(oc_sd, sk_sd, atol=1e-10)
    np.testing.assert_array_equal(oc_si, sk_si)


def test_kneighbors_exact_zero_distance_weight():
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    y_train = np.array([1.0, 0.0, 0.0, 0.0])
    X_test = np.array([[0.0, 0.0]])

    oc = KNeighborsClassifier(n_neighbors=3, weights="distance", device="cpu").fit(X_train, y_train)
    np.testing.assert_allclose(oc.predict_proba(X_test), np.array([[0.0, 1.0]]), atol=1e-12)
    np.testing.assert_array_equal(oc.predict(X_test), np.array([1.0]))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_kneighbors_regressor_parity_1d_and_multioutput(weights, metric):
    X, y = make_regression(
        n_samples=140, n_features=5, n_informative=4, n_targets=2, noise=0.2, random_state=7
    )
    X_train, X_test = X[:110], X[110:]
    y1_train, y1_test = y[:110, 0], y[110:, 0]
    y2_train, y2_test = y[:110], y[110:]

    sk1 = SkKNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric).fit(X_train, y1_train)
    oc1 = KNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric, device="cpu").fit(
        X_train, y1_train
    )
    np.testing.assert_allclose(oc1.predict(X_test), sk1.predict(X_test), atol=1e-10)
    assert oc1.score(X_test, y1_test) == pytest.approx(sk1.score(X_test, y1_test), abs=1e-10)

    sk2 = SkKNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric).fit(X_train, y2_train)
    oc2 = KNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric, device="cpu").fit(
        X_train, y2_train
    )
    np.testing.assert_allclose(oc2.predict(X_test), sk2.predict(X_test), atol=1e-10)
    assert oc2.score(X_test, y2_test) == pytest.approx(sk2.score(X_test, y2_test), abs=1e-10)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_radius_neighbors_classifier_parity_and_outlier_label(weights):
    X_train = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.2],
        [5.0, 5.0],
        [5.1, 5.1],
        [5.2, 4.9],
        [5.0, 5.2],
    ])
    y_train = np.array([0, 0, 1, 1, 1, 1, 0])
    X_test = np.array([[0.05, 0.05], [5.05, 5.05], [20.0, 20.0]])

    # Without outlier_label: raises on point with 0 neighbors
    oc_err = RadiusNeighborsClassifier(radius=1.0, weights=weights, device="cpu").fit(X_train, y_train)
    with pytest.raises(ValueError):
        oc_err.predict(X_test)

    # With numeric outlier_label
    sk_num = SkRadiusNeighborsClassifier(radius=1.0, weights=weights, outlier_label=0).fit(
        X_train, y_train
    )
    oc_num = RadiusNeighborsClassifier(
        radius=1.0, weights=weights, outlier_label=0, device="cpu"
    ).fit(X_train, y_train)
    np.testing.assert_array_equal(oc_num.predict(X_test), sk_num.predict(X_test))
    np.testing.assert_allclose(oc_num.predict_proba(X_test), sk_num.predict_proba(X_test), atol=1e-12)

    # With outlier_label="most_frequent"
    sk_mf = SkRadiusNeighborsClassifier(
        radius=1.0, weights=weights, outlier_label="most_frequent"
    ).fit(X_train, y_train)
    oc_mf = RadiusNeighborsClassifier(
        radius=1.0, weights=weights, outlier_label="most_frequent", device="cpu"
    ).fit(X_train, y_train)
    np.testing.assert_array_equal(oc_mf.predict(X_test), sk_mf.predict(X_test))
    np.testing.assert_allclose(oc_mf.predict_proba(X_test), sk_mf.predict_proba(X_test), atol=1e-12)

    # radius_neighbors query
    oc_d, oc_i = oc_mf.radius_neighbors(X_test, radius=1.0, return_distance=True, sort_results=True)
    sk_d, sk_i = sk_mf.radius_neighbors(X_test, radius=1.0, return_distance=True, sort_results=True)
    for o_d, s_d, o_ind, s_ind in zip(oc_d, sk_d, oc_i, sk_i):
        np.testing.assert_allclose(o_d, s_d, atol=1e-12)
        np.testing.assert_array_equal(o_ind, s_ind)


def test_radius_neighbors_regressor_parity_and_empty_neighborhood():
    X_train = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 3.0], [3.2, 3.1]])
    y_train_1d = np.array([1.0, 3.0, 10.0, 14.0])
    y_train_2d = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [14.0, 24.0]])
    X_test = np.array([[0.25, 0.0], [3.1, 3.05]])

    sk1 = SkRadiusNeighborsRegressor(radius=1.0, weights="distance").fit(X_train, y_train_1d)
    oc1 = RadiusNeighborsRegressor(radius=1.0, weights="distance", device="cpu").fit(
        X_train, y_train_1d
    )
    np.testing.assert_allclose(oc1.predict(X_test), sk1.predict(X_test), atol=1e-10)
    assert oc1.score(X_test, np.array([2.0, 12.0])) == pytest.approx(
        sk1.score(X_test, np.array([2.0, 12.0])), abs=1e-10
    )

    sk2 = SkRadiusNeighborsRegressor(radius=1.0, weights="uniform").fit(X_train, y_train_2d)
    oc2 = RadiusNeighborsRegressor(radius=1.0, weights="uniform", device="cpu").fit(
        X_train, y_train_2d
    )
    np.testing.assert_allclose(oc2.predict(X_test), sk2.predict(X_test), atol=1e-10)

    # Outlier point -> NaN prediction
    pred_outlier = oc1.predict(np.array([[100.0, 100.0]]))
    assert np.isnan(pred_outlier[0])


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (KNeighborsClassifier, {"n_neighbors": 3}),
        (KNeighborsRegressor, {"n_neighbors": 3}),
        (RadiusNeighborsClassifier, {"radius": 2.0, "outlier_label": "most_frequent"}),
        (RadiusNeighborsRegressor, {"radius": 2.0}),
    ],
)
def test_neighbors_dit_and_pickle_roundtrip_and_devices(tmp_path, cls, kwargs):
    X, y = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=9
    )
    if "Regressor" in cls.__name__:
        y = y.astype(np.float64) * 2.5

    for dev in ("cpu", "auto"):
        est = cls(**kwargs, device=dev).fit(X, y)
        preds = est.predict(X[:10])

        dit_path = tmp_path / f"{cls.__name__}_{dev}.dit"
        est.save(str(dit_path))
        loaded = cls.load(str(dit_path), device=dev)
        np.testing.assert_allclose(loaded.predict(X[:10]), preds, atol=1e-12)

        unpickled = pickle.loads(pickle.dumps(est))
        np.testing.assert_allclose(unpickled.predict(X[:10]), preds, atol=1e-12)


def test_neighbors_validation_errors():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError):
        KNeighborsClassifier(n_neighbors=0)
    with pytest.raises(ValueError):
        KNeighborsRegressor(n_neighbors=0)
    with pytest.raises(ValueError):
        RadiusNeighborsClassifier(radius=0.0)
    with pytest.raises(ValueError):
        RadiusNeighborsRegressor(radius=-1.0)

    # n_neighbors > n_samples
    with pytest.raises(ValueError):
        KNeighborsClassifier(n_neighbors=5).fit(X, y)

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        KNeighborsClassifier(n_neighbors=2).fit(X_nan, y)

    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError):
        KNeighborsRegressor(n_neighbors=2).fit(X_inf, y)

    # Empty array
    with pytest.raises(ValueError):
        KNeighborsClassifier(n_neighbors=2).fit(np.empty((0, 2)), np.empty((0,)))

    # Shape mismatch
    clf = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(np.array([[1.0, 2.0, 3.0]]))
