# OchreML Unit Tests (＾▽＾)
# Verifies compatibility across Python Lists, NumPy ndarrays, Pandas, and Polars.

import pytest
import numpy as np
import pandas as pd
import polars as pl

from ochreml import (
    LinearRegression,
    Ridge,
    RidgeClassifier,
    RidgeCV,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


# =========================================================================
# Linear Regression Tests
# =========================================================================

def test_linear_regression_with_python_lists():
    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [5.0, 7.0, 9.0, 11.0, 13.0]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    assert model.coef_ is not None
    assert pytest.approx(model.coef_[0], rel=1e-3) == 2.0
    assert pytest.approx(model.intercept_, rel=1e-3) == 3.0
    assert pytest.approx(model.score(X, y), rel=1e-3) == 1.0

    preds = model.predict([[6.0], [7.0]])
    assert pytest.approx(preds[0], rel=1e-3) == 15.0
    assert pytest.approx(preds[1], rel=1e-3) == 17.0


def test_linear_regression_without_intercept():
    X = [[1.0], [2.0], [3.0]]
    y = [4.0, 8.0, 12.0]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    assert pytest.approx(model.coef_[0], rel=1e-3) == 4.0
    assert pytest.approx(model.intercept_, abs=1e-5) == 0.0


def test_linear_regression_with_numpy():
    np.random.seed(42)
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = 3.0 * X[:, 0] - 1.0 * X[:, 1] + 4.0

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    assert pytest.approx(model.coef_[0], rel=1e-3) == 3.0
    assert pytest.approx(model.coef_[1], rel=1e-3) == -1.0
    assert pytest.approx(model.intercept_, rel=1e-3) == 4.0
    assert pytest.approx(model.score(X, y), rel=1e-3) == 1.0


def test_linear_regression_with_pandas():
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "feat2": [0.5, 1.0, 1.5, 2.0],
    })
    s_y = pd.Series([3.5, 6.0, 8.5, 11.0])

    model = LinearRegression(fit_intercept=True)
    model.fit(df, s_y)

    assert pytest.approx(model.score(df, s_y), rel=1e-3) == 1.0
    preds = model.predict(df)
    assert len(preds) == 4


def test_linear_regression_with_polars():
    pl_df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
    })
    pl_y = pl.Series("target", [12.0, 24.0, 36.0, 48.0])

    model = LinearRegression(fit_intercept=True)
    model.fit(pl_df, pl_y)

    assert pytest.approx(model.score(pl_df, pl_y), rel=1e-3) == 1.0


# =========================================================================
# Decision Tree Classifier Tests
# =========================================================================

def test_decision_tree_classifier_with_lists():
    X = [[0.5], [1.0], [1.5], [3.5], [4.0], [4.5]]
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
    clf.fit(X, y)

    assert clf.score(X, y) == 1.0
    preds = clf.predict([[0.8], [4.2]])
    assert preds[0] == 0.0
    assert preds[1] == 1.0


def test_decision_tree_classifier_entropy_numpy():
    X = np.array([
        [1.0, 10.0],
        [1.0, 20.0],
        [5.0, 10.0],
        [5.0, 20.0],
    ])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    clf.fit(X, y)

    assert clf.score(X, y) == 1.0
    test_X = np.array([[1.0, 15.0], [5.0, 15.0]])
    preds = clf.predict(test_X)
    assert preds[0] == 0.0
    assert preds[1] == 1.0


def test_decision_tree_classifier_polars_and_pandas():
    p_df = pd.DataFrame({"x": [10.0, 15.0, 30.0, 35.0]})
    p_y = pd.Series([1.0, 1.0, 2.0, 2.0])

    clf = DecisionTreeClassifier(criterion="gini", max_depth=2)
    clf.fit(p_df, p_y)
    assert clf.score(p_df, p_y) == 1.0

    pl_test = pl.DataFrame({"x": [12.0, 32.0]})
    preds = clf.predict(pl_test)
    assert preds[0] == 1.0
    assert preds[1] == 2.0


# =========================================================================
# Decision Tree Regressor Tests
# =========================================================================

def test_decision_tree_regressor_with_numpy():
    X = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
    y = np.array([10.0, 10.0, 10.0, 50.0, 50.0, 50.0])

    reg = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
    reg.fit(X, y)

    assert pytest.approx(reg.score(X, y), rel=1e-3) == 1.0

    preds = reg.predict(np.array([[2.5], [8.5]]))
    assert pytest.approx(preds[0], rel=1e-3) == 10.0
    assert pytest.approx(preds[1], rel=1e-3) == 50.0


def test_decision_tree_regressor_with_polars():
    pl_df = pl.DataFrame({
        "feature": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    })
    pl_y = pl.Series("target", [5.0, 5.0, 5.0, 25.0, 25.0, 25.0])

    reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(pl_df, pl_y)

    assert pytest.approx(reg.score(pl_df, pl_y), rel=1e-3) == 1.0


# =========================================================================
# Input Validation and Error Handling Tests
# =========================================================================

def test_unfitted_model_errors():
    model = LinearRegression()
    with pytest.raises(Exception):
        model.predict([[1.0, 2.0]])

    clf = DecisionTreeClassifier()
    with pytest.raises(Exception):
        clf.predict([[1.0]])

    reg = DecisionTreeRegressor()
    with pytest.raises(Exception):
        reg.predict([[1.0]])


def test_dimension_mismatch_error():
    model = LinearRegression()
    with pytest.raises(Exception):
        model.fit([[1.0], [2.0]], [1.0])


# =========================================================================
# Hardware Acceleration & Device Tests (＾▽＾)
# =========================================================================

def test_device_discovery():
    from ochreml import (
        get_available_devices,
        get_device_info,
        get_default_device,
        set_default_device,
    )

    devices = get_available_devices()
    assert isinstance(devices, list)
    assert "cpu" in devices

    info = get_device_info()
    assert isinstance(info, dict)
    assert "cpu_cores" in info
    assert "default_auto_device" in info

    assert get_default_device() == "auto"
    set_default_device("cpu")
    assert get_default_device() == "cpu"
    set_default_device("auto")
    assert get_default_device() == "auto"


def test_device_parameter_on_models():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]

    # Linear Regression with auto & explicit cpu
    lr_auto = LinearRegression(device="auto")
    lr_auto.fit(X, y)
    assert lr_auto.device_ is not None

    lr_cpu = LinearRegression(device="cpu")
    lr_cpu.fit(X, y)
    assert lr_cpu.device_ == "cpu"

    # Decision Tree Classifier with auto
    y_cls = [0.0, 0.0, 1.0, 1.0]
    clf = DecisionTreeClassifier(device="auto")
    clf.fit(X, y_cls)
    assert clf.device_ is not None

    # Decision Tree Regressor with auto
    reg = DecisionTreeRegressor(device="auto")
    reg.fit(X, y)
    assert reg.device_ is not None


def test_invalid_device_raises_informative_error():
    with pytest.raises(ValueError) as excinfo:
        LinearRegression(device="non_existent_accelerator_999")
    assert "Unrecognized device string" in str(excinfo.value)


def test_dit_save_load_and_pickle(tmp_path):
    import pickle
    import ochreml

    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y_reg = np.array([5.0, 4.0, 11.0, 10.0])
    y_cls = np.array([0.0, 0.0, 1.0, 1.0])

    # 1. Unified ochreml.save / ochreml.load (.dit binary format)
    lr = LinearRegression(device="cpu").fit(X, y_reg)
    lr_path = str(tmp_path / "lr_model.dit")
    ochreml.save(lr, lr_path)

    loaded_lr = ochreml.load(lr_path)
    assert isinstance(loaded_lr, LinearRegression)
    assert loaded_lr.device_ is not None
    assert np.allclose(loaded_lr.predict(X), lr.predict(X))

    # 2. DecisionTreeClassifier & DecisionTreeRegressor via ochreml.save / ochreml.load
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, device="cpu").fit(X, y_cls)
    clf_path = str(tmp_path / "clf_model.dit")
    ochreml.save(clf, clf_path)
    loaded_clf = ochreml.load(clf_path)
    assert isinstance(loaded_clf, DecisionTreeClassifier)
    assert np.array_equal(loaded_clf.predict(X), clf.predict(X))

    reg = DecisionTreeRegressor(max_depth=3, device="cpu").fit(X, y_reg)
    reg_path = str(tmp_path / "reg_model.dit")
    reg.save(reg_path)
    loaded_reg = ochreml.load(reg_path, device="cpu")
    assert isinstance(loaded_reg, DecisionTreeRegressor)
    assert np.allclose(loaded_reg.predict(X), reg.predict(X))

    # 3. Pickle / Joblib style compatibility
    pickled_lr = pickle.loads(pickle.dumps(lr))
    assert np.allclose(pickled_lr.predict(X), lr.predict(X))
    assert pickled_lr.device_ is not None

    # 4. Enforce .dit extension requirement
    with pytest.raises(ValueError) as excinfo:
        ochreml.save(lr, str(tmp_path / "bad_ext.pkl"))
    assert ".dit" in str(excinfo.value)


def test_linear_regression_sample_weight_parity():
    from sklearn.linear_model import LinearRegression as SklearnLR

    rng = np.random.RandomState(42)
    X = rng.randn(500, 8)
    true_coef = rng.randn(8)
    y = X @ true_coef + 2.5 + rng.randn(500) * 0.1
    weights = rng.uniform(0.1, 10.0, size=500)

    for fit_intercept in [True, False]:
        ochre_lr = LinearRegression(fit_intercept=fit_intercept).fit(X, y, sample_weight=weights)
        sk_lr = SklearnLR(fit_intercept=fit_intercept).fit(X, y, sample_weight=weights)

        assert np.allclose(ochre_lr.coef_, sk_lr.coef_, atol=1e-10)
        assert pytest.approx(float(ochre_lr.intercept_), abs=1e-10) == float(sk_lr.intercept_)
        assert np.allclose(ochre_lr.predict(X), sk_lr.predict(X), atol=1e-10)
        assert pytest.approx(ochre_lr.score(X, y, sample_weight=weights), abs=1e-10) == sk_lr.score(
            X, y, sample_weight=weights
        )


def test_linear_regression_multi_output_parity():
    from sklearn.linear_model import LinearRegression as SklearnLR

    rng = np.random.RandomState(123)
    X = rng.randn(400, 6)
    true_W = rng.randn(6, 3)
    true_b = np.array([1.5, -2.0, 0.75])
    Y = X @ true_W + true_b + rng.randn(400, 3) * 0.05

    for fit_intercept in [True, False]:
        ochre_lr = LinearRegression(fit_intercept=fit_intercept).fit(X, Y)
        sk_lr = SklearnLR(fit_intercept=fit_intercept).fit(X, Y)

        assert ochre_lr.coef_.shape == sk_lr.coef_.shape == (3, 6)
        assert np.allclose(ochre_lr.coef_, sk_lr.coef_, atol=1e-10)
        if fit_intercept:
            assert ochre_lr.intercept_.shape == sk_lr.intercept_.shape == (3,)
            assert np.allclose(ochre_lr.intercept_, sk_lr.intercept_, atol=1e-10)

        preds = ochre_lr.predict(X)
        sk_preds = sk_lr.predict(X)
        assert preds.shape == sk_preds.shape == (400, 3)
        assert np.allclose(preds, sk_preds, atol=1e-10)
        assert pytest.approx(ochre_lr.score(X, Y), abs=1e-10) == sk_lr.score(X, Y)


def test_linear_regression_multi_output_with_weights_parity():
    from sklearn.linear_model import LinearRegression as SklearnLR

    rng = np.random.RandomState(777)
    X = rng.randn(1200, 10)
    Y = X @ rng.randn(10, 4) + np.array([3.0, -1.0, 0.5, 2.0]) + rng.randn(1200, 4) * 0.2
    weights = rng.uniform(0.5, 5.0, size=1200)

    ochre_lr = LinearRegression(fit_intercept=True).fit(X, Y, sample_weight=weights)
    sk_lr = SklearnLR(fit_intercept=True).fit(X, Y, sample_weight=weights)

    assert np.allclose(ochre_lr.coef_, sk_lr.coef_, atol=1e-10)
    assert np.allclose(ochre_lr.intercept_, sk_lr.intercept_, atol=1e-10)
    assert np.allclose(ochre_lr.predict(X), sk_lr.predict(X), atol=1e-10)
    assert pytest.approx(ochre_lr.score(X, Y, sample_weight=weights), abs=1e-10) == sk_lr.score(
        X, Y, sample_weight=weights
    )


def test_sample_weight_validation():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError) as excinfo:
        LinearRegression().fit(X, y, sample_weight=np.array([1.0, -1.0, 2.0]))
    assert "non-negative" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        LinearRegression().fit(X, y, sample_weight=np.array([1.0, 2.0]))
    assert "Sample weight length" in str(excinfo.value)


def test_dit_save_load_multi_output(tmp_path):
    import ochreml

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]])
    Y = np.array([[3.0, 9.0], [4.0, 4.0], [6.0, 8.0], [14.0, 10.0]])
    w = np.array([1.0, 2.0, 1.5, 3.0])

    model = LinearRegression().fit(X, Y, sample_weight=w)
    dit_path = str(tmp_path / "multi_lr.dit")
    ochreml.save(model, dit_path)

    loaded = ochreml.load(dit_path)
    assert np.allclose(loaded.coef_, model.coef_)
    assert np.allclose(loaded.intercept_, model.intercept_)
    assert np.allclose(loaded.predict(X), model.predict(X))


# =========================================================================
# Ridge, RidgeClassifier, and RidgeCV Tests (＾▽＾)
# =========================================================================

def test_ridge_unpenalized_intercept():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = 3.0 * X[:, 0] + 42.0

    model = Ridge(alpha=1e4, fit_intercept=True, device="cpu").fit(X, y)
    assert pytest.approx(float(model.intercept_), abs=1e-10) == 42.0
    assert abs(float(model.coef_[0])) < 0.01


def test_ridge_parity_with_sklearn():
    from sklearn.linear_model import Ridge as SklearnRidge

    rng = np.random.RandomState(2026)
    X = rng.randn(300, 6)
    y_1d = X @ rng.randn(6) + 2.5 + rng.randn(300) * 0.2
    Y_2d = X @ rng.randn(6, 3) + np.array([1.0, -2.0, 0.5]) + rng.randn(300, 3) * 0.2
    weights = rng.uniform(0.2, 5.0, size=300)

    for alpha in [0.01, 1.0, 100.0]:
        for fit_intercept in [True, False]:
            for sw in [None, weights]:
                ochre_1d = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu").fit(
                    X, y_1d, sample_weight=sw
                )
                sk_1d = SklearnRidge(alpha=alpha, fit_intercept=fit_intercept).fit(
                    X, y_1d, sample_weight=sw
                )
                assert np.allclose(ochre_1d.coef_, sk_1d.coef_, atol=1e-8)
                assert pytest.approx(float(ochre_1d.intercept_), abs=1e-8) == float(sk_1d.intercept_)
                assert np.allclose(ochre_1d.predict(X), sk_1d.predict(X), atol=1e-8)
                assert pytest.approx(ochre_1d.score(X, y_1d, sample_weight=sw), abs=1e-8) == sk_1d.score(
                    X, y_1d, sample_weight=sw
                )

                ochre_2d = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu").fit(
                    X, Y_2d, sample_weight=sw
                )
                sk_2d = SklearnRidge(alpha=alpha, fit_intercept=fit_intercept).fit(
                    X, Y_2d, sample_weight=sw
                )
                assert ochre_2d.coef_.shape == sk_2d.coef_.shape == (3, 6)
                assert np.allclose(ochre_2d.coef_, sk_2d.coef_, atol=1e-8)
                if fit_intercept:
                    assert np.allclose(ochre_2d.intercept_, sk_2d.intercept_, atol=1e-8)
                else:
                    assert np.allclose(ochre_2d.intercept_, np.zeros(3), atol=1e-10)
                assert np.allclose(ochre_2d.predict(X), sk_2d.predict(X), atol=1e-8)
                assert pytest.approx(ochre_2d.score(X, Y_2d, sample_weight=sw), abs=1e-8) == sk_2d.score(
                    X, Y_2d, sample_weight=sw
                )


def test_ridge_input_formats():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]})
    s_y = pd.Series([5.0, 5.0, 11.0, 11.0])

    r1 = Ridge(alpha=0.5).fit(df, s_y)
    assert r1.score(df, s_y) > 0.98

    pl_df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]})
    pl_y = pl.Series("y", [5.0, 5.0, 11.0, 11.0])
    r2 = Ridge(alpha=0.5).fit(pl_df, pl_y)
    assert np.allclose(r1.coef_, r2.coef_)

    r3 = Ridge(alpha=0.5).fit([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]], [5.0, 5.0, 11.0, 11.0])
    assert np.allclose(r1.coef_, r3.coef_)


def test_ridge_classifier_parity_with_sklearn():
    from sklearn.linear_model import RidgeClassifier as SklearnRC

    rng = np.random.RandomState(42)
    X = rng.randn(240, 5)
    weights = rng.uniform(0.3, 4.0, size=240)

    # Binary classification with custom numeric labels (-1.0, 2.0)
    logits = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3
    y_bin = np.where(logits > 0.0, 2.0, -1.0)

    # Multiclass classification (3 classes)
    scores_mc = np.column_stack([
        X[:, 0] + 0.5 * X[:, 1],
        -X[:, 0] + X[:, 2],
        -X[:, 1] - X[:, 2] + 0.4,
    ])
    y_mc = np.argmax(scores_mc, axis=1).astype(np.float64)

    for y_cls in [y_bin, y_mc]:
        for fit_intercept in [True, False]:
            for sw in [None, weights]:
                ochre_rc = RidgeClassifier(alpha=1.0, fit_intercept=fit_intercept, device="cpu").fit(
                    X, y_cls, sample_weight=sw
                )
                sk_rc = SklearnRC(alpha=1.0, fit_intercept=fit_intercept).fit(
                    X, y_cls, sample_weight=sw
                )

                assert np.array_equal(ochre_rc.classes_, sk_rc.classes_)
                assert np.allclose(ochre_rc.coef_, sk_rc.coef_, atol=1e-8)
                assert np.allclose(ochre_rc.intercept_, sk_rc.intercept_, atol=1e-8)
                assert np.allclose(ochre_rc.decision_function(X), sk_rc.decision_function(X), atol=1e-8)
                assert np.array_equal(ochre_rc.predict(X), sk_rc.predict(X))
                assert pytest.approx(ochre_rc.score(X, y_cls, sample_weight=sw), abs=1e-10) == sk_rc.score(
                    X, y_cls, sample_weight=sw
                )


def test_ridge_cv_parity_with_sklearn():
    from sklearn.linear_model import RidgeCV as SklearnRidgeCV

    rng = np.random.RandomState(99)
    X = rng.randn(150, 8)
    y_1d = X @ np.array([2.0, -1.5, 0.5, 0.0, 1.0, -0.5, 0.2, 0.0]) + 3.0 + rng.randn(150) * 0.8
    Y_2d = X @ rng.randn(8, 3) + np.array([1.0, -1.0, 2.0]) + rng.randn(150, 3) * 0.8
    weights = rng.uniform(0.5, 3.0, size=150)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)

    for fit_intercept in [True, False]:
        for sw in [None, weights]:
            ochre_1d = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, device="cpu").fit(
                X, y_1d, sample_weight=sw
            )
            sk_1d = SklearnRidgeCV(alphas=alphas, fit_intercept=fit_intercept).fit(
                X, y_1d, sample_weight=sw
            )
            assert ochre_1d.alpha_ == pytest.approx(float(sk_1d.alpha_))
            assert ochre_1d.best_score_ == pytest.approx(float(sk_1d.best_score_), abs=1e-7)
            assert np.allclose(ochre_1d.coef_, sk_1d.coef_, atol=1e-7)
            assert pytest.approx(float(ochre_1d.intercept_), abs=1e-7) == float(sk_1d.intercept_)
            assert np.allclose(ochre_1d.predict(X), sk_1d.predict(X), atol=1e-7)

            ochre_2d = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, device="cpu").fit(
                X, Y_2d, sample_weight=sw
            )
            sk_2d = SklearnRidgeCV(alphas=alphas, fit_intercept=fit_intercept).fit(
                X, Y_2d, sample_weight=sw
            )
            assert ochre_2d.alpha_ == pytest.approx(float(sk_2d.alpha_))
            assert ochre_2d.best_score_ == pytest.approx(float(sk_2d.best_score_), abs=1e-7)
            assert np.allclose(ochre_2d.coef_, sk_2d.coef_, atol=1e-7)
            if fit_intercept:
                assert np.allclose(ochre_2d.intercept_, sk_2d.intercept_, atol=1e-7)
            assert np.allclose(ochre_2d.predict(X), sk_2d.predict(X), atol=1e-7)

    # K-Fold CV parity check
    ochre_kf = RidgeCV(alphas=alphas, fit_intercept=True, cv=5, device="cpu").fit(X, y_1d)
    sk_kf = SklearnRidgeCV(alphas=alphas, fit_intercept=True, cv=5).fit(X, y_1d)
    assert ochre_kf.alpha_ == pytest.approx(float(sk_kf.alpha_))
    assert ochre_kf.best_score_ == pytest.approx(float(sk_kf.best_score_), abs=1e-7)
    assert np.allclose(ochre_kf.coef_, sk_kf.coef_, atol=1e-7)
    assert pytest.approx(float(ochre_kf.intercept_), abs=1e-7) == float(sk_kf.intercept_)


def test_ridge_family_dit_save_load_and_pickle(tmp_path):
    import pickle
    import ochreml

    rng = np.random.RandomState(7)
    X = rng.randn(40, 4)
    y_1d = X @ np.array([1.5, -2.0, 0.5, 1.0]) + 3.0
    Y_2d = X @ rng.randn(4, 2) + np.array([1.0, -1.0])
    y_bin = (y_1d > 3.0).astype(np.float64)
    y_mc = np.array([0.0, 1.0, 2.0] * 13 + [0.0])

    models = [
        ("ridge_1d.dit", Ridge(alpha=2.5, device="cpu").fit(X, y_1d)),
        ("ridge_2d.dit", Ridge(alpha=0.5, device="cpu").fit(X, Y_2d)),
        ("rc_bin.dit", RidgeClassifier(alpha=1.0, device="cpu").fit(X, y_bin)),
        ("rc_mc.dit", RidgeClassifier(alpha=1.0, device="cpu").fit(X, y_mc)),
        ("ridge_cv.dit", RidgeCV(alphas=(0.1, 1.0, 10.0), device="cpu").fit(X, y_1d)),
    ]

    for fname, model in models:
        dit_path = str(tmp_path / fname)
        ochreml.save(model, dit_path)
        loaded = ochreml.load(dit_path)
        assert type(loaded) is type(model)
        assert np.allclose(loaded.coef_, model.coef_)
        assert np.allclose(loaded.intercept_, model.intercept_)
        assert np.allclose(loaded.predict(X), model.predict(X))
        if isinstance(model, RidgeClassifier):
            assert np.array_equal(loaded.classes_, model.classes_)
            assert np.allclose(loaded.decision_function(X), model.decision_function(X))
        if isinstance(model, RidgeCV):
            assert loaded.alpha_ == model.alpha_
            assert loaded.best_score_ == pytest.approx(model.best_score_)

        unpickled = pickle.loads(pickle.dumps(model))
        assert np.allclose(unpickled.predict(X), model.predict(X))


def test_ridge_invalid_params_and_unfitted():
    with pytest.raises(ValueError):
        Ridge(alpha=-1.0)
    with pytest.raises(ValueError):
        Ridge(alpha=float("nan"))
    with pytest.raises(ValueError):
        RidgeClassifier(alpha=-0.5)
    with pytest.raises(ValueError):
        RidgeCV(alphas=())
    with pytest.raises(ValueError):
        RidgeCV(alphas=(-1.0, 1.0))
    with pytest.raises(ValueError):
        RidgeCV(alphas=(0.0, 1.0), cv=None)
    with pytest.raises(ValueError):
        RidgeCV(alphas=(1.0,), cv=1)

    with pytest.raises(ValueError):
        Ridge().predict([[1.0, 2.0]])
    with pytest.raises(ValueError):
        RidgeClassifier().decision_function([[1.0, 2.0]])
    with pytest.raises(ValueError):
        RidgeCV().predict([[1.0, 2.0]])



