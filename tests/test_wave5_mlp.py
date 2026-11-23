import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import (
    MLPClassifier as SkMLPClassifier,
    MLPRegressor as SkMLPRegressor,
)
from sklearn.preprocessing import StandardScaler

from ochreml.neural_network import MLPClassifier, MLPRegressor


@pytest.mark.parametrize("device", ["cpu", "auto"])
@pytest.mark.parametrize("solver", ["adam", "sgd", "lbfgs"])
def test_mlp_classifier_binary_parity(device, solver):
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sk_clf = SkMLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver=solver,
        learning_rate_init=0.01,
        max_iter=300,
        random_state=42,
    ).fit(X_train, y_train)
    oml_clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver=solver,
        learning_rate_init=0.01,
        max_iter=300,
        random_state=42,
        device=device,
    ).fit(X_train, y_train)

    sk_acc = sk_clf.score(X_test, y_test)
    oml_acc = oml_clf.score(X_test, y_test)
    assert abs(oml_acc - sk_acc) < 0.12
    assert oml_acc > 0.80

    proba = oml_clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    log_proba = oml_clf.predict_log_proba(X_test)
    assert np.allclose(np.exp(log_proba), proba, atol=1e-5)

    assert oml_clf.n_layers_ == 4
    assert oml_clf.n_outputs_ == 1
    assert oml_clf.out_activation_ == "logistic"
    assert oml_clf.n_features_in_ == 8
    assert len(oml_clf.coefs_) == 3
    assert len(oml_clf.intercepts_) == 3
    assert oml_clf.loss_ is not None
    assert len(oml_clf.loss_curve_) > 0
    assert oml_clf.n_iter_ > 0
    assert np.array_equal(oml_clf.classes_, np.array([0.0, 1.0]))


@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic"])
def test_mlp_classifier_multiclass_activations(activation):
    X, y = make_classification(
        n_samples=270,
        n_features=8,
        n_informative=6,
        n_classes=3,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sk_clf = SkMLPClassifier(
        hidden_layer_sizes=(32,),
        activation=activation,
        solver="adam",
        learning_rate_init=0.01,
        max_iter=300,
        random_state=7,
    ).fit(X_train, y_train)
    oml_clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation=activation,
        solver="adam",
        learning_rate_init=0.01,
        max_iter=300,
        random_state=7,
        device="cpu",
    ).fit(X_train, y_train)

    assert oml_clf.out_activation_ == "softmax"
    assert oml_clf.n_outputs_ == 3
    assert abs(oml_clf.score(X_test, y_test) - sk_clf.score(X_test, y_test)) < 0.15
    assert oml_clf.score(X_test, y_test) > 0.60


def test_mlp_classifier_partial_fit():
    X, y = make_classification(n_samples=150, n_features=6, random_state=11)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        solver="adam",
        learning_rate_init=0.01,
        random_state=11,
        device="cpu",
    )
    for _ in range(30):
        clf.partial_fit(X, y, classes=[0, 1])

    assert clf.n_iter_ == 30
    assert clf.score(X, y) > 0.80


@pytest.mark.parametrize("solver", ["adam", "lbfgs"])
def test_mlp_regressor_1d_and_multioutput(solver):
    X, y = make_regression(
        n_samples=200,
        n_features=6,
        n_informative=5,
        noise=0.5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sk_reg = SkMLPRegressor(
        hidden_layer_sizes=(32, 16),
        solver=solver,
        learning_rate_init=0.05,
        max_iter=400,
        random_state=42,
    ).fit(X_train, y_train)
    oml_reg = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        solver=solver,
        learning_rate_init=0.05,
        max_iter=400,
        random_state=42,
        device="cpu",
    ).fit(X_train, y_train)

    preds = oml_reg.predict(X_test)
    assert preds.ndim == 1
    assert oml_reg.out_activation_ == "identity"
    assert abs(oml_reg.score(X_test, y_test) - sk_reg.score(X_test, y_test)) < 0.10
    assert oml_reg.score(X_test, y_test) > 0.85

    # Multi-output regression
    y_multi = np.column_stack([y, -0.5 * y + 2.0])
    y_tr_m, y_te_m = y_multi[:150], y_multi[150:]
    X_tr_m = scaler.fit_transform(X[:150])
    X_te_m = scaler.transform(X[150:])
    oml_multi = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        solver=solver,
        learning_rate_init=0.05,
        max_iter=400,
        random_state=42,
        device="auto",
    ).fit(X_tr_m, y_tr_m)
    preds_m = oml_multi.predict(X_te_m)
    assert preds_m.shape == (50, 2)
    assert oml_multi.n_outputs_ == 2
    assert oml_multi.score(X_te_m, y_te_m) > 0.85


def test_mlp_dit_save_load_and_pickle(tmp_path):
    X, y = make_classification(n_samples=120, n_features=5, random_state=3)
    clf = MLPClassifier(hidden_layer_sizes=16, max_iter=50, random_state=3, device="cpu")
    clf.fit(X, y)

    dit_path = tmp_path / "mlp_clf.dit"
    clf.save(str(dit_path))
    loaded = MLPClassifier.load(str(dit_path), device="cpu")
    assert np.allclose(loaded.predict_proba(X), clf.predict_proba(X))
    assert np.array_equal(loaded.predict(X), clf.predict(X))

    pickled = pickle.loads(pickle.dumps(clf))
    assert np.allclose(pickled.predict_proba(X), clf.predict_proba(X))


def test_mlp_validation_and_edge_cases():
    with pytest.raises(ValueError):
        MLPClassifier(alpha=-0.1)
    with pytest.raises(ValueError):
        MLPClassifier(hidden_layer_sizes=())
    with pytest.raises(ValueError):
        MLPClassifier(hidden_layer_sizes=(10, 0))
    with pytest.raises(ValueError):
        MLPRegressor(activation="invalid_act")

    clf = MLPClassifier(max_iter=10, random_state=0)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((5, 3)))

    X_bad = np.array([[1.0, np.nan], [2.0, 3.0]])
    y_ok = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        clf.fit(X_bad, y_ok)

    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.empty((0,)))

    clf.fit(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError):
        clf.predict(np.zeros((2, 5)))
