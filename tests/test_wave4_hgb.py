import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    HistGradientBoostingClassifier as SkHGBClassifier,
    HistGradientBoostingRegressor as SkHGBRegressor,
)
from sklearn.model_selection import train_test_split

import ochreml as oml
from ochreml import DecisionTreeClassifier, DecisionTreeRegressor
from ochreml.ensemble_hgb import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_hgb_classifier_binary_parity(device):
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    sk_clf = SkHGBClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    ).fit(X_train, y_train)
    oml_clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
        device=device,
    ).fit(X_train, y_train)

    sk_acc = sk_clf.score(X_test, y_test)
    oml_acc = oml_clf.score(X_test, y_test)
    assert abs(oml_acc - sk_acc) < 0.08
    assert oml_acc > 0.80

    proba = oml_clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    df = oml_clf.decision_function(X_test)
    assert df.shape == (len(X_test),)
    assert oml_clf.n_features_in_ == 10
    assert oml_clf.n_iter_ == 50
    assert oml_clf.device_ is not None
    assert np.array_equal(oml_clf.classes_, np.array([0.0, 1.0]))


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_hgb_classifier_multiclass_parity(device):
    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=7,
        n_classes=3,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    sk_clf = SkHGBClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        random_state=7,
    ).fit(X_train, y_train)
    oml_clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        random_state=7,
        device=device,
    ).fit(X_train, y_train)

    sk_acc = sk_clf.score(X_test, y_test)
    oml_acc = oml_clf.score(X_test, y_test)
    assert abs(oml_acc - sk_acc) < 0.10
    assert oml_acc > 0.65

    proba = oml_clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    df = oml_clf.decision_function(X_test)
    assert df.shape == (len(X_test), 3)


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_hgb_regressor_parity_and_sample_weight(loss, device):
    X, y = make_regression(
        n_samples=300,
        n_features=8,
        n_informative=6,
        noise=2.0,
        random_state=13,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=13
    )
    rng = np.random.default_rng(13)
    sw = rng.uniform(0.5, 2.0, size=len(X_train))

    sk_reg = SkHGBRegressor(
        loss=loss,
        learning_rate=0.1,
        max_iter=60,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        random_state=13,
    ).fit(X_train, y_train, sample_weight=sw)
    oml_reg = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.1,
        max_iter=60,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        random_state=13,
        device=device,
    ).fit(X_train, y_train, sample_weight=sw)

    sk_r2 = sk_reg.score(X_test, y_test)
    oml_r2 = oml_reg.score(X_test, y_test)
    assert abs(oml_r2 - sk_r2) < 0.12
    assert oml_r2 > 0.75
    assert oml_reg.n_features_in_ == 8
    assert oml_reg.n_iter_ == 60


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_decision_tree_cpu_and_auto(device):
    X_cls, y_cls = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        random_state=21,
    )
    clf = DecisionTreeClassifier(max_depth=5, device=device).fit(X_cls, y_cls)
    assert clf.score(X_cls, y_cls) > 0.85

    X_reg, y_reg = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=1.0,
        random_state=21,
    )
    reg = DecisionTreeRegressor(max_depth=6, device=device).fit(X_reg, y_reg)
    assert reg.score(X_reg, y_reg) > 0.80


def test_hgb_dit_save_load_and_pickle(tmp_path):
    X, y = make_classification(n_samples=120, n_features=5, random_state=5)
    clf = HistGradientBoostingClassifier(max_iter=20, min_samples_leaf=10).fit(X, y)
    clf_path = tmp_path / "hgb_clf.dit"
    clf.save(str(clf_path))
    loaded_clf = HistGradientBoostingClassifier.load(str(clf_path), device="cpu")
    assert np.allclose(clf.predict_proba(X), loaded_clf.predict_proba(X))
    assert np.array_equal(clf.predict(X), loaded_clf.predict(X))

    unpickled_clf = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(clf.predict(X), unpickled_clf.predict(X))

    Xr, yr = make_regression(n_samples=120, n_features=5, random_state=5)
    reg = HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=10).fit(Xr, yr)
    reg_path = tmp_path / "hgb_reg.dit"
    reg.save(str(reg_path))
    loaded_reg = HistGradientBoostingRegressor.load(str(reg_path), device="cpu")
    assert np.allclose(reg.predict(Xr), loaded_reg.predict(Xr))

    unpickled_reg = pickle.loads(pickle.dumps(reg))
    assert np.allclose(reg.predict(Xr), unpickled_reg.predict(Xr))

    loaded_reg.to("auto")
    assert loaded_reg.device == "auto"


def test_hgb_validation_and_edge_cases():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_cls = np.array([0.0, 0.0, 1.0, 1.0])
    y_reg = np.array([1.0, 2.0, 3.0, 4.0])

    # Invalid hyperparameters
    with pytest.raises(ValueError):
        HistGradientBoostingClassifier(max_bins=256)
    with pytest.raises(ValueError):
        HistGradientBoostingClassifier(max_bins=1)
    with pytest.raises(ValueError):
        HistGradientBoostingClassifier(learning_rate=0.0)
    with pytest.raises(ValueError):
        HistGradientBoostingClassifier(learning_rate=-0.1)
    with pytest.raises(ValueError):
        HistGradientBoostingRegressor(min_samples_leaf=0)
    with pytest.raises(ValueError):
        HistGradientBoostingRegressor(loss="invalid_loss")

    # NaN / Inf
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        HistGradientBoostingClassifier(min_samples_leaf=1).fit(X_nan, y_cls)

    X_inf = X.copy()
    X_inf[1, 1] = np.inf
    with pytest.raises(ValueError):
        HistGradientBoostingRegressor(min_samples_leaf=1).fit(X_inf, y_reg)

    # Empty data
    with pytest.raises(ValueError):
        HistGradientBoostingRegressor().fit(np.empty((0, 2)), np.empty((0,)))

    # Shape mismatch
    clf = HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=1).fit(X, y_cls)
    with pytest.raises(ValueError):
        clf.predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        clf.fit(X, np.array([0.0, 1.0]))
