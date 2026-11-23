"""
OchreML Wave 7 Python Pipelines & Meta-Estimators Test Suite (＾▽＾)
Verifies Pipeline, OneVsRestClassifier, MultiOutputRegressor, GridSearchCV,
RandomizedSearchCV, cross_val_score, RFE, SelectFromModel, BaggingClassifier,
BaggingRegressor, VotingClassifier, and StackingClassifier (*≧ω≦*).
"""

import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import ochreml
from ochreml import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
    Ridge,
    StandardScaler,
)
from ochreml.pipeline import Pipeline, clone_estimator, make_pipeline
from ochreml.multiclass import OneVsRestClassifier
from ochreml.multioutput import MultiOutputRegressor
from ochreml._meta_selection import (
    RFE,
    GridSearchCV,
    RandomizedSearchCV,
    SelectFromModel,
    cross_val_score,
)
from ochreml._meta_ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    VotingClassifier,
)


def _check_dit_and_pickle(tmp_path, model, X_test, check_proba=False):
    preds = model.predict(X_test)
    dit_path = str(tmp_path / f"{type(model).__name__}.dit")

    model.save(dit_path)
    loaded_cls = type(model).load(dit_path)
    np.testing.assert_allclose(loaded_cls.predict(X_test), preds, atol=1e-12)

    ochreml.save(model, dit_path)
    loaded_top = ochreml.load(dit_path)
    np.testing.assert_allclose(loaded_top.predict(X_test), preds, atol=1e-12)

    unpickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(unpickled.predict(X_test), preds, atol=1e-12)

    if check_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        np.testing.assert_allclose(loaded_top.predict_proba(X_test), proba, atol=1e-12)
        np.testing.assert_allclose(unpickled.predict_proba(X_test), proba, atol=1e-12)


def test_clone_and_nested_params():
    lr = LogisticRegression(C=0.5, max_iter=50)
    cloned = clone_estimator(lr)
    assert cloned is not lr
    assert cloned.C == 0.5
    assert cloned.max_iter == 50

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0))])
    params = pipe.get_params(deep=True)
    assert "clf__C" in params
    assert params["clf__C"] == 1.0

    pipe.set_params(clf__C=0.25)
    assert pipe["clf"].C == 0.25


def test_pipeline_and_make_pipeline(tmp_path):
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pass", "passthrough"),
            ("clf", LogisticRegression(C=1.0, max_iter=100)),
        ]
    )
    pipe.fit(X, y)
    assert len(pipe) == 3
    assert pipe[0] is pipe.named_steps["scaler"]
    assert pipe["pass"] == "passthrough"
    assert len(pipe[:2]) == 2
    assert pipe.score(X, y) > 0.75
    assert pipe.predict_proba(X).shape == (120, 2)
    assert pipe.decision_function(X).shape[0] == 120
    np.testing.assert_array_equal(pipe.classes_, np.array([0, 1]))

    _check_dit_and_pickle(tmp_path, pipe, X[:10], check_proba=True)

    mp = make_pipeline(StandardScaler(), LogisticRegression())
    preds = mp.fit_predict(X, y)
    assert preds.shape == (120,)

    trans_pipe = make_pipeline(StandardScaler(), "passthrough")
    Xt = trans_pipe.fit_transform(X)
    assert Xt.shape == X.shape
    np.testing.assert_allclose(trans_pipe.transform(X), Xt, atol=1e-12)


def test_one_vs_rest_classifier(tmp_path):
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    ovr_lr = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=100))
    ovr_lr.fit(X, y)
    assert len(ovr_lr.estimators_) == 3
    np.testing.assert_array_equal(ovr_lr.classes_, np.array([0, 1, 2]))
    proba = ovr_lr.predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(150), atol=1e-6)
    assert ovr_lr.score(X, y) > 0.70
    _check_dit_and_pickle(tmp_path, ovr_lr, X[:10], check_proba=True)

    # Classifier with decision_function only
    ovr_svc = OneVsRestClassifier(LinearSVC(C=1.0, max_iter=200))
    ovr_svc.fit(X, y)
    assert ovr_svc.decision_function(X).shape == (150, 3)
    assert ovr_svc.predict_proba(X).shape == (150, 3)
    assert ovr_svc.score(X, y) > 0.65


def test_multioutput_regressor(tmp_path):
    X, y = make_regression(
        n_samples=100, n_features=5, n_targets=3, noise=0.1, random_state=42
    )
    mor = MultiOutputRegressor(Ridge(alpha=1.0))
    mor.fit(X, y)
    assert mor.n_outputs_ == 3
    assert len(mor.estimators_) == 3
    preds = mor.predict(X)
    assert preds.shape == (100, 3)
    assert mor.score(X, y) > 0.95
    _check_dit_and_pickle(tmp_path, mor, X[:10])


def test_cross_val_score_and_search_cv(tmp_path):
    X_clf, y_clf = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    scores_acc = cross_val_score(
        LogisticRegression(), X_clf, y_clf, scoring="accuracy", cv=3
    )
    assert scores_acc.shape == (3,)
    assert np.all(scores_acc > 0.70)

    scores_auc = cross_val_score(
        LogisticRegression(), X_clf, y_clf, scoring="roc_auc", cv=3
    )
    assert np.all(scores_auc > 0.70)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    gs = GridSearchCV(
        pipe,
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
        scoring="accuracy",
        cv=3,
    )
    gs.fit(X_clf, y_clf)
    assert "mean_test_score" in gs.cv_results_
    assert gs.best_params_["clf__C"] in [0.1, 1.0, 10.0]
    assert gs.best_score_ > 0.75
    assert gs.predict(X_clf).shape == (120,)
    assert gs.predict_proba(X_clf).shape == (120, 2)
    _check_dit_and_pickle(tmp_path, gs, X_clf[:10], check_proba=True)

    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.5, random_state=42
    )
    rs = RandomizedSearchCV(
        Ridge(),
        param_distributions={"alpha": [0.01, 0.1, 1.0, 10.0]},
        n_iter=3,
        scoring="r2",
        cv=3,
        random_state=42,
    )
    rs.fit(X_reg, y_reg)
    assert len(rs.cv_results_["params"]) == 3
    assert rs.score(X_reg, y_reg) > 0.90


def test_rfe_and_select_from_model():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=3,
        n_redundant=2,
        random_state=42,
    )
    rfe = RFE(LogisticRegression(), n_features_to_select=3, step=1)
    Xt = rfe.fit_transform(X, y)
    assert Xt.shape == (120, 3)
    assert int(np.sum(rfe.support_)) == 3
    assert rfe.ranking_.shape == (8,)
    assert np.sum(rfe.ranking_ == 1) == 3
    assert rfe.get_support(indices=True).shape == (3,)
    assert rfe.predict(X).shape == (120,)

    sfm = SelectFromModel(
        RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42),
        max_features=4,
    )
    Xt_sfm = sfm.fit_transform(X, y)
    assert Xt_sfm.shape[0] == 120
    assert 1 <= Xt_sfm.shape[1] <= 4

    # Prefit SelectFromModel
    lr_fit = LogisticRegression().fit(X, y)
    sfm_pre = SelectFromModel(lr_fit, prefit=True, threshold="mean")
    assert sfm_pre.transform(X).shape[0] == 120


def test_bagging_classifier_and_regressor(tmp_path):
    X_clf, y_clf = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=42
    )
    bag_c = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=7,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
    )
    bag_c.fit(X_clf, y_clf)
    assert len(bag_c.estimators_) == 7
    assert len(bag_c.estimators_features_) == 7
    assert bag_c.predict_proba(X_clf).shape == (120, 2)
    assert bag_c.score(X_clf, y_clf) > 0.80
    _check_dit_and_pickle(tmp_path, bag_c, X_clf[:10], check_proba=True)

    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.5, random_state=42
    )
    bag_r = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=6),
        n_estimators=7,
        random_state=42,
    )
    bag_r.fit(X_reg, y_reg)
    assert len(bag_r.estimators_) == 7
    assert bag_r.predict(X_reg).shape == (100,)
    assert bag_r.score(X_reg, y_reg) > 0.70
    _check_dit_and_pickle(tmp_path, bag_r, X_reg[:10])


def test_voting_and_stacking_classifier(tmp_path):
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    voting_hard = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression()),
            ("dt", DecisionTreeClassifier(max_depth=4)),
        ],
        voting="hard",
    )
    voting_hard.fit(X, y)
    assert voting_hard.score(X, y) > 0.75
    assert "lr" in voting_hard.named_estimators_

    voting_soft = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression()),
            ("rf", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)),
        ],
        voting="soft",
        weights=[2.0, 1.0],
    )
    voting_soft.fit(X, y)
    assert voting_soft.predict_proba(X).shape == (150, 2)
    assert voting_soft.score(X, y) > 0.75
    _check_dit_and_pickle(tmp_path, voting_soft, X[:10], check_proba=True)

    stack = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression()),
            ("rf", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)),
        ],
        cv=3,
        passthrough=True,
    )
    stack.fit(X, y)
    assert len(stack.estimators_) == 2
    assert stack.final_estimator_ is not None
    assert stack.predict(X).shape == (150,)
    assert stack.predict_proba(X).shape == (150, 2)
    assert stack.score(X, y) > 0.75
    _check_dit_and_pickle(tmp_path, stack, X[:10], check_proba=True)
