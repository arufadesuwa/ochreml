# OchreML Wave 4 Boosting Ensembles Tests (＾▽＾)
# Verifies GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier,
# and AdaBoostRegressor against scikit-learn, plus .dit, pickle, device, and edge cases.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostClassifier as SkAdaBoostClassifier,
    AdaBoostRegressor as SkAdaBoostRegressor,
    GradientBoostingClassifier as SkGradientBoostingClassifier,
    GradientBoostingRegressor as SkGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SkDecisionTreeRegressor

from ochreml.ensemble_boosting import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "huber"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_gradient_boosting_regressor_losses_and_staged(loss, device):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=2.0,
        random_state=42,
    )
    rng = np.random.RandomState(7)
    w = rng.uniform(0.5, 2.0, size=len(y))

    sk = SkGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.1,
        n_estimators=50,
        max_depth=3,
        random_state=42,
    ).fit(X, y, sample_weight=w)
    oc = GradientBoostingRegressor(
        loss=loss,
        learning_rate=0.1,
        n_estimators=50,
        max_depth=3,
        random_state=42,
        device=device,
    ).fit(X, y, sample_weight=w)

    oc_r2 = oc.score(X, y)
    sk_r2 = sk.score(X, y)
    assert oc_r2 > 0.85
    assert abs(oc_r2 - sk_r2) < 0.05

    staged = list(oc.staged_predict(X))
    assert len(staged) == 50
    np.testing.assert_allclose(staged[-1], oc.predict(X), atol=1e-12)
    assert oc.feature_importances_ is not None
    assert oc.feature_importances_.shape == (6,)
    assert oc.feature_importances_.sum() == pytest.approx(1.0, abs=1e-10)


def test_gradient_boosting_regressor_subsample():
    X, y = make_regression(n_samples=120, n_features=5, noise=1.0, random_state=13)
    oc = GradientBoostingRegressor(
        n_estimators=40,
        subsample=0.7,
        learning_rate=0.1,
        random_state=13,
        device="cpu",
    ).fit(X, y)
    assert oc.score(X, y) > 0.85


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_gradient_boosting_classifier_parity(n_classes, device):
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    rng = np.random.RandomState(11)
    w = rng.uniform(0.5, 2.0, size=len(y))

    sk = SkGradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ).fit(X, y, sample_weight=w)
    oc = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        device=device,
    ).fit(X, y, sample_weight=w)

    assert oc.score(X, y) >= 0.90
    assert abs(oc.score(X, y) - sk.score(X, y)) < 0.08
    probs = oc.predict_proba(X)
    assert probs.shape == (len(y), n_classes)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    df = oc.decision_function(X)
    if n_classes == 2:
        assert df.shape == (len(y),)
    else:
        assert df.shape == (len(y), n_classes)

    np.testing.assert_array_equal(oc.classes_, np.unique(y).astype(np.float64))
    assert oc.feature_importances_.sum() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_adaboost_classifier_parity(n_classes, device):
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=21,
    )
    rng = np.random.RandomState(5)
    w = rng.uniform(0.5, 2.0, size=len(y))

    sk = SkAdaBoostClassifier(
        estimator=SkDecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=21,
    ).fit(X, y, sample_weight=w)
    oc = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        max_depth=1,
        random_state=21,
        device=device,
    ).fit(X, y, sample_weight=w)

    assert oc.score(X, y) >= 0.78
    assert abs(oc.score(X, y) - sk.score(X, y)) < 0.12
    probs = oc.predict_proba(X)
    assert probs.shape == (len(y), n_classes)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    df = oc.decision_function(X)
    if n_classes == 2:
        assert df.shape == (len(y),)
    else:
        assert df.shape == (len(y), n_classes)

    assert len(oc.estimator_weights_) > 0
    assert len(oc.estimator_errors_) == len(oc.estimator_weights_)
    assert oc.feature_importances_.sum() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("loss", ["linear", "square", "exponential"])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_adaboost_regressor_losses(loss, device):
    X, y = make_regression(
        n_samples=140,
        n_features=5,
        n_informative=4,
        noise=1.5,
        random_state=33,
    )
    rng = np.random.RandomState(9)
    w = rng.uniform(0.5, 2.0, size=len(y))

    sk = SkAdaBoostRegressor(
        estimator=SkDecisionTreeRegressor(max_depth=3),
        n_estimators=40,
        learning_rate=1.0,
        loss=loss,
        random_state=33,
    ).fit(X, y, sample_weight=w)
    oc = AdaBoostRegressor(
        n_estimators=40,
        learning_rate=1.0,
        loss=loss,
        max_depth=3,
        random_state=33,
        device=device,
    ).fit(X, y, sample_weight=w)

    assert oc.score(X, y) > 0.80
    assert abs(oc.score(X, y) - sk.score(X, y)) < 0.10
    assert len(oc.estimator_weights_) > 0
    assert len(oc.estimator_errors_) == len(oc.estimator_weights_)
    assert oc.feature_importances_.sum() == pytest.approx(1.0, abs=1e-10)


def test_boosting_dit_and_pickle_roundtrip(tmp_path):
    X_clf, y_clf = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_reg, y_reg = make_regression(n_samples=90, n_features=5, random_state=42)

    models = [
        (GradientBoostingClassifier(n_estimators=15, random_state=42, device="cpu").fit(X_clf, y_clf), X_clf, "gbc"),
        (GradientBoostingRegressor(n_estimators=15, random_state=42, device="cpu").fit(X_reg, y_reg), X_reg, "gbr"),
        (AdaBoostClassifier(n_estimators=15, random_state=42, device="cpu").fit(X_clf, y_clf), X_clf, "abc"),
        (AdaBoostRegressor(n_estimators=15, random_state=42, device="cpu").fit(X_reg, y_reg), X_reg, "abr"),
    ]

    for model, X_test, name in models:
        dit_path = tmp_path / f"{name}.dit"
        model.save(str(dit_path))
        loaded = type(model).load(str(dit_path), device="cpu")
        np.testing.assert_allclose(loaded.predict(X_test), model.predict(X_test), atol=1e-12)

        unpickled = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(unpickled.predict(X_test), model.predict(X_test), atol=1e-12)

        model.to("auto")
        assert model.device_ is not None


def test_boosting_edge_cases_and_validation():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_clf = np.array([0.0, 0.0, 1.0, 1.0])
    y_reg = np.array([1.0, 2.0, 3.0, 4.0])

    # Invalid parameters
    with pytest.raises(ValueError):
        GradientBoostingRegressor(loss="invalid_loss")
    with pytest.raises(ValueError):
        GradientBoostingRegressor(learning_rate=0.0)
    with pytest.raises(ValueError):
        GradientBoostingRegressor(subsample=1.5)
    with pytest.raises(ValueError):
        GradientBoostingClassifier(loss="squared_error")
    with pytest.raises(ValueError):
        AdaBoostClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        AdaBoostRegressor(loss="cubic")

    # NaN / Inf / Empty / Shape mismatch
    for cls, y in [
        (GradientBoostingClassifier, y_clf),
        (GradientBoostingRegressor, y_reg),
        (AdaBoostClassifier, y_clf),
        (AdaBoostRegressor, y_reg),
    ]:
        m = cls(n_estimators=5, device="cpu")
        with pytest.raises(ValueError):
            m.fit(np.empty((0, 2)), np.empty(0))

        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError):
            m.fit(X_nan, y)

        X_inf = X.copy()
        X_inf[0, 1] = np.inf
        with pytest.raises(ValueError):
            m.fit(X_inf, y)

        m.fit(X, y)
        with pytest.raises(ValueError):
            m.predict(np.ones((2, 3)))
