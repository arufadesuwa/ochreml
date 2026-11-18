# OchreML Batch 1 Linear & GLM Tests (＾▽＾)
# Verifies numerical parity with scikit-learn, .dit/pickle round-trips, and device fallback.

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    ElasticNet as SkElasticNet,
    Lasso as SkLasso,
    LogisticRegression as SkLogisticRegression,
    SGDClassifier as SkSGDClassifier,
    SGDRegressor as SkSGDRegressor,
)
from sklearn.preprocessing import StandardScaler

import ochreml
from ochreml import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from ochreml import linear_model as ochre_lm


def test_version_and_linear_model_exports():
    assert ochreml.__version__ == "0.3.3b1"
    assert ochre_lm.LogisticRegression is LogisticRegression
    assert ochre_lm.Lasso is Lasso
    assert ochre_lm.ElasticNet is ElasticNet
    assert ochre_lm.SGDClassifier is SGDClassifier
    assert ochre_lm.SGDRegressor is SGDRegressor


def test_logistic_regression_binary_parity_and_serialization(tmp_path):
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=42
    )
    X = StandardScaler().fit_transform(X)

    sk = SkLogisticRegression(C=1.5, max_iter=300, tol=1e-6)
    sk.fit(X, y)

    oc = LogisticRegression(C=1.5, max_iter=300, tol=1e-6, device="cpu")
    oc.fit(X, y)

    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=1e-2)
    np.testing.assert_allclose(oc.predict_log_proba(X), sk.predict_log_proba(X), atol=2e-2)
    np.testing.assert_allclose(oc.decision_function(X), sk.decision_function(X), atol=2e-2)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=0.02)

    dit_path = tmp_path / "logreg_bin.dit"
    oc.save(str(dit_path))
    loaded = ochreml.load(str(dit_path), device="auto")
    np.testing.assert_array_equal(loaded.predict(X), oc.predict(X))
    np.testing.assert_allclose(loaded.predict_proba(X), oc.predict_proba(X), atol=1e-12)

    unpickled = pickle.loads(pickle.dumps(oc))
    np.testing.assert_array_equal(unpickled.predict(X), oc.predict(X))


def test_logistic_regression_multiclass_weights_and_class_weight():
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        random_state=7,
    )
    X = StandardScaler().fit_transform(X)
    sw = np.linspace(0.5, 2.0, len(y))

    sk = SkLogisticRegression(
        C=2.0, max_iter=300, tol=1e-6, class_weight="balanced"
    )
    sk.fit(X, y, sample_weight=sw)

    oc = LogisticRegression(
        C=2.0, max_iter=300, tol=1e-6, class_weight="balanced", device="auto"
    )
    oc.fit(X, y, sample_weight=sw)

    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(oc.predict_proba(X), sk.predict_proba(X), atol=2e-2)
    assert oc.score(X, y, sample_weight=sw) == pytest.approx(
        sk.score(X, y, sample_weight=sw), abs=0.02
    )

    sk_dict = SkLogisticRegression(C=1.0, class_weight={0: 1.0, 1: 2.0, 2: 1.5})
    sk_dict.fit(X, y)
    oc_dict = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 2.0, 2: 1.5})
    oc_dict.fit(X, y)
    assert oc_dict.score(X, y) == pytest.approx(sk_dict.score(X, y), abs=0.02)


def test_lasso_1d_multi_output_positive_and_dit(tmp_path):
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=5, noise=0.5, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    sw = np.linspace(0.5, 1.5, len(y))

    sk = SkLasso(alpha=0.2, max_iter=2000, tol=1e-6)
    sk.fit(X, y, sample_weight=sw)

    oc = Lasso(alpha=0.2, max_iter=2000, tol=1e-6, device="cpu")
    oc.fit(X, y, sample_weight=sw)

    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=1e-3, rtol=1e-3)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-4)

    # Positive constraint
    oc_pos = Lasso(alpha=0.1, positive=True, max_iter=2000, tol=1e-6)
    oc_pos.fit(X, y)
    assert np.all(oc_pos.coef_ >= -1e-12)

    # Multi-output 2D target
    Y2 = np.column_stack([y, 0.5 * y + 3.0])
    sk_m = SkLasso(alpha=0.2, max_iter=2000, tol=1e-6)
    sk_m.fit(X, Y2)
    oc_m = Lasso(alpha=0.2, max_iter=2000, tol=1e-6, device="auto")
    oc_m.fit(X, Y2)
    np.testing.assert_allclose(oc_m.coef_, sk_m.coef_, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(oc_m.intercept_, sk_m.intercept_, atol=1e-3, rtol=1e-3)

    dit_path = tmp_path / "lasso.dit"
    oc_m.save(str(dit_path))
    loaded = ochreml.load(str(dit_path))
    np.testing.assert_allclose(loaded.predict(X), oc_m.predict(X), atol=1e-12)
    unpickled = pickle.loads(pickle.dumps(oc))
    np.testing.assert_allclose(unpickled.predict(X), oc.predict(X), atol=1e-12)


def test_elastic_net_1d_multi_output_and_dit(tmp_path):
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=5, noise=0.5, random_state=99
    )
    X = StandardScaler().fit_transform(X)

    sk = SkElasticNet(alpha=0.2, l1_ratio=0.6, max_iter=2000, tol=1e-6)
    sk.fit(X, y)

    oc = ElasticNet(alpha=0.2, l1_ratio=0.6, max_iter=2000, tol=1e-6, device="cpu")
    oc.fit(X, y)

    np.testing.assert_allclose(oc.coef_, sk.coef_, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(oc.intercept_, sk.intercept_, atol=1e-3, rtol=1e-3)
    assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=1e-4)

    Y2 = np.column_stack([y, -y + 2.0])
    oc_m = ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=2000, tol=1e-6, device="auto")
    oc_m.fit(X, Y2)
    assert oc_m.score(X, Y2) > 0.95

    dit_path = tmp_path / "elasticnet.dit"
    oc.save(str(dit_path))
    loaded = ElasticNet.load(str(dit_path))
    np.testing.assert_allclose(loaded.predict(X), oc.predict(X), atol=1e-12)
    unpickled = pickle.loads(pickle.dumps(oc_m))
    np.testing.assert_allclose(unpickled.predict(X), oc_m.predict(X), atol=1e-12)


def test_sgd_classifier_losses_penalties_and_serialization(tmp_path):
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=12
    )
    X = StandardScaler().fit_transform(X)

    for loss in ("hinge", "log_loss", "modified_huber"):
        sk = SkSGDClassifier(
            loss=loss, penalty="elasticnet", alpha=0.001, max_iter=1000, random_state=42
        )
        sk.fit(X, y)

        oc = SGDClassifier(
            loss=loss,
            penalty="elasticnet",
            alpha=0.001,
            max_iter=1000,
            random_state=42,
            device="cpu",
        )
        oc.fit(X, y)
        assert oc.score(X, y) >= 0.80
        assert abs(oc.score(X, y) - sk.score(X, y)) < 0.08

        if loss == "hinge":
            with pytest.raises(ValueError):
                oc.predict_proba(X)
        else:
            probs = oc.predict_proba(X)
            assert probs.shape == (200, 2)
            np.testing.assert_allclose(probs.sum(axis=1), np.ones(200), atol=1e-9)

    # Multiclass + constant learning_rate + .dit / pickle
    X_m, y_m = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        random_state=3,
    )
    X_m = StandardScaler().fit_transform(X_m)
    sk_m = SkSGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=0.001,
        learning_rate="constant",
        eta0=0.02,
        max_iter=500,
        random_state=7,
    )
    sk_m.fit(X_m, y_m)
    oc_m = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=0.001,
        learning_rate="constant",
        eta0=0.02,
        max_iter=500,
        random_state=7,
    )
    oc_m.fit(X_m, y_m)
    assert abs(oc_m.score(X_m, y_m) - sk_m.score(X_m, y_m)) < 0.08
    assert oc_m.predict_proba(X_m).shape == (180, 3)

    dit_path = tmp_path / "sgd_clf.dit"
    oc_m.save(str(dit_path))
    loaded = ochreml.load(str(dit_path))
    np.testing.assert_array_equal(loaded.predict(X_m), oc_m.predict(X_m))
    unpickled = pickle.loads(pickle.dumps(oc_m))
    np.testing.assert_array_equal(unpickled.predict(X_m), oc_m.predict(X_m))


def test_sgd_regressor_losses_and_serialization(tmp_path):
    X, y = make_regression(
        n_samples=200, n_features=5, n_informative=4, noise=0.2, random_state=21
    )
    X = StandardScaler().fit_transform(X)
    y = (y - y.mean()) / y.std()

    for loss in ("squared_error", "huber", "epsilon_insensitive"):
        sk = SkSGDRegressor(
            loss=loss, penalty="l2", alpha=0.0001, max_iter=1000, random_state=42
        )
        sk.fit(X, y)

        oc = SGDRegressor(
            loss=loss,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42,
            device="cpu",
        )
        oc.fit(X, y)
        assert oc.score(X, y) > 0.95
        assert oc.score(X, y) == pytest.approx(sk.score(X, y), abs=0.02)

    dit_path = tmp_path / "sgd_reg.dit"
    oc.save(str(dit_path))
    loaded = ochreml.load(str(dit_path))
    np.testing.assert_allclose(loaded.predict(X), oc.predict(X), atol=1e-12)
    unpickled = pickle.loads(pickle.dumps(oc))
    np.testing.assert_allclose(unpickled.predict(X), oc.predict(X), atol=1e-12)

