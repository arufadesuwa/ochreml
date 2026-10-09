# OchreML Unit Tests (＾▽＾)
# Verifies compatibility across Python Lists, NumPy ndarrays, Pandas, and Polars.

import pytest
import numpy as np
import pandas as pd
import polars as pl

from ochreml import LinearRegression, DecisionTreeClassifier, DecisionTreeRegressor


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
