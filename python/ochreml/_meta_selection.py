"""
OchreML Model Selection, Cross-Validation & Feature Selection Meta-Estimators (＾▽＾)
Implements cross_val_score, GridSearchCV, RandomizedSearchCV, RFE, and SelectFromModel (*≧ω≦*).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from .pipeline import (
    _BaseMetaEstimator,
    _check_is_fitted,
    _set_estimator_params,
    clone_estimator,
)


def _is_classifier_estimator(estimator: Any, y: Optional[Any] = None) -> bool:
    """Heuristic to determine if an estimator is a classifier."""
    if estimator is None:
        return False
    if getattr(estimator, "_estimator_type", None) == "classifier":
        return True
    cls_name = type(estimator).__name__
    if cls_name.endswith("Classifier") or cls_name in (
        "LogisticRegression",
        "SVC",
        "LinearSVC",
        "SGDClassifier",
        "GaussianNB",
        "MultinomialNB",
        "BernoulliNB",
        "ComplementNB",
        "LinearDiscriminantAnalysis",
        "QuadraticDiscriminantAnalysis",
    ):
        return True
    if hasattr(estimator, "predict_proba") and not cls_name.endswith("Regressor"):
        return True
    return False


def _fallback_kfold_splits(
    n_samples: int, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    folds = np.array_split(indices, n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, test_idx))
    return splits


def _fallback_stratified_splits(
    y_arr: np.ndarray, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    _, y_inv = np.unique(y_arr, return_inverse=True)
    n_classes = int(y_inv.max()) + 1 if len(y_inv) > 0 else 0
    per_class_folds: List[List[np.ndarray]] = [[] for _ in range(n_splits)]
    for c in range(n_classes):
        cls_idx = np.where(y_inv == c)[0]
        parts = np.array_split(cls_idx, n_splits)
        for i in range(n_splits):
            per_class_folds[i].append(parts[i])
    folds = [
        np.concatenate(per_class_folds[i]) if per_class_folds[i] else np.empty(0, dtype=int)
        for i in range(n_splits)
    ]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_idx = np.sort(folds[i])
        train_idx = np.sort(
            np.concatenate([folds[j] for j in range(n_splits) if j != i])
        )
        splits.append((train_idx, test_idx))
    return splits


def _resolve_cv_splits(
    cv: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    groups: Optional[Any] = None,
    estimator: Optional[Any] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Resolve cv parameter into a list of (train_idx, test_idx) tuples (＾▽＾)."""
    if cv is None:
        cv = 5

    if isinstance(cv, (int, np.integer)):
        n_splits = int(cv)
        if n_splits < 2:
            raise ValueError(f"cv must be >= 2, got {n_splits} ( >_< )")
        use_stratified = y is not None and _is_classifier_estimator(estimator, y)
        if use_stratified:
            try:
                from .model_selection import StratifiedKFold

                splitter = StratifiedKFold(n_splits=n_splits)
                return [
                    (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
                    for tr, te in splitter.split(X, y)
                ]
            except ImportError:
                return _fallback_stratified_splits(np.asarray(y), n_splits)
            except Exception:
                return _fallback_stratified_splits(np.asarray(y), n_splits)
        try:
            from .model_selection import KFold

            splitter = KFold(n_splits=n_splits)
            return [
                (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
                for tr, te in splitter.split(X, y)
            ]
        except ImportError:
            return _fallback_kfold_splits(X.shape[0], n_splits)

    if hasattr(cv, "split"):
        try:
            raw_splits = cv.split(X, y, groups)
        except TypeError:
            raw_splits = cv.split(X, y)
        return [
            (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for tr, te in raw_splits
        ]

    return [
        (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
        for tr, te in cv
    ]


def _score_estimator(
    estimator: Any,
    X_val: np.ndarray,
    y_val: Optional[np.ndarray],
    scoring: Optional[Union[str, Callable[..., float]]] = None,
) -> float:
    """Evaluate a fitted estimator on validation data (*≧ω≦*)."""
    if scoring is None:
        if y_val is None:
            return float(estimator.score(X_val))
        return float(estimator.score(X_val, y_val))

    if callable(scoring):
        return float(scoring(estimator, X_val, y_val))

    if isinstance(scoring, str):
        key = scoring.lower()
        if key == "accuracy":
            y_pred = np.asarray(estimator.predict(X_val))
            try:
                from .metrics import accuracy_score

                return float(accuracy_score(y_val, y_pred))
            except ImportError:
                return float(np.mean(np.asarray(y_val) == y_pred))

        if key == "r2":
            y_pred = np.asarray(estimator.predict(X_val), dtype=np.float64)
            try:
                from .metrics import r2_score

                return float(r2_score(y_val, y_pred))
            except ImportError:
                y_true = np.asarray(y_val, dtype=np.float64)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return 1.0 if ss_tot <= 1e-12 and ss_res <= 1e-12 else (
                    0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
                )

        if key in ("neg_mean_squared_error", "neg_mse"):
            y_pred = np.asarray(estimator.predict(X_val), dtype=np.float64)
            y_true = np.asarray(y_val, dtype=np.float64)
            return -float(np.mean((y_true - y_pred) ** 2))

        if key in ("f1", "f1_macro", "f1_weighted"):
            y_pred = np.asarray(estimator.predict(X_val))
            try:
                from .metrics import f1_score

                avg = "binary" if key == "f1" else key.split("_", 1)[1]
                return float(f1_score(y_val, y_pred, average=avg))
            except ImportError:
                y_true = np.asarray(y_val)
                tp = float(np.sum((y_true == 1) & (y_pred == 1)))
                fp = float(np.sum((y_true != 1) & (y_pred == 1)))
                fn = float(np.sum((y_true == 1) & (y_pred != 1)))
                denom = 2.0 * tp + fp + fn
                return (2.0 * tp / denom) if denom > 0.0 else 0.0

        if key == "roc_auc":
            if hasattr(estimator, "predict_proba"):
                proba = np.asarray(estimator.predict_proba(X_val), dtype=np.float64)
                y_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
            elif hasattr(estimator, "decision_function"):
                y_score = np.asarray(estimator.decision_function(X_val), dtype=np.float64).ravel()
            else:
                y_score = np.asarray(estimator.predict(X_val), dtype=np.float64).ravel()
            try:
                from .metrics import roc_auc_score

                return float(roc_auc_score(y_val, y_score))
            except ImportError:
                y_true = np.asarray(y_val, dtype=np.float64).ravel()
                order = np.argsort(y_score)
                y_sorted = y_true[order]
                n_pos = float(np.sum(y_sorted == 1))
                n_neg = float(np.sum(y_sorted == 0))
                if n_pos == 0 or n_neg == 0:
                    return 0.5
                ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
                auc = (np.sum(ranks[y_sorted == 1]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
                return float(auc)

        raise ValueError(f"Unsupported scoring metric '{scoring}' ( >_< )")

    raise TypeError(f"scoring must be None, str, or callable, got {type(scoring)} ( >_< )")


def cross_val_score(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    groups: Optional[Any] = None,
    scoring: Optional[Union[str, Callable[..., float]]] = None,
    cv: Optional[Any] = None,
) -> np.ndarray:
    """
    Evaluate a score by cross-validation (＾▽＾).

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or None, default=None
        The target variable to try to predict.
    groups : array-like or None, default=None
        Group labels for the samples used while splitting the dataset.
    scoring : str, callable, or None, default=None
        Scoring metric ('accuracy', 'r2', 'neg_mean_squared_error', 'f1', 'roc_auc',
        callable, or None for estimator.score).
    cv : int, cross-validation generator, or iterable, default=None
        Determines the cross-validation splitting strategy (defaults to 5-fold).

    Returns
    -------
    scores : numpy.ndarray of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y) if y is not None else None
    splits = _resolve_cv_splits(cv, X_arr, y_arr, groups=groups, estimator=estimator)

    scores: List[float] = []
    for train_idx, test_idx in splits:
        est = clone_estimator(estimator)
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        if y_arr is not None:
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            est.fit(X_train, y_train)
            scores.append(_score_estimator(est, X_test, y_test, scoring=scoring))
        else:
            est.fit(X_train)
            scores.append(_score_estimator(est, X_test, None, scoring=scoring))

    return np.asarray(scores, dtype=np.float64)


def _rank_scores(mean_scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-mean_scores, kind="mergesort")
    ranks = np.empty(len(mean_scores), dtype=int)
    ranks[order] = np.arange(1, len(mean_scores) + 1)
    return ranks


class _BaseSearchCV(_BaseMetaEstimator):
    """Shared cross-validation search logic for GridSearchCV & RandomizedSearchCV (*≧ω≦*)."""

    def _run_search(
        self,
        candidates: List[Dict[str, Any]],
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Any:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y) if y is not None else None
        splits = _resolve_cv_splits(
            self.cv, X_arr, y_arr, groups=groups, estimator=self.estimator
        )
        n_splits = len(splits)
        self.n_splits_ = n_splits
        self.n_features_in_ = int(X_arr.shape[1])

        n_cands = len(candidates)
        if n_cands == 0:
            raise ValueError("Parameter grid/distributions produced 0 candidates ( >_< )")

        split_scores = np.empty((n_splits, n_cands), dtype=np.float64)
        for c_idx, cand_params in enumerate(candidates):
            for s_idx, (train_idx, test_idx) in enumerate(splits):
                est = clone_estimator(self.estimator)
                _set_estimator_params(est, **cand_params)
                try:
                    if y_arr is not None:
                        est.fit(X_arr[train_idx], y_arr[train_idx])
                        sc = _score_estimator(
                            est, X_arr[test_idx], y_arr[test_idx], scoring=self.scoring
                        )
                    else:
                        est.fit(X_arr[train_idx])
                        sc = _score_estimator(
                            est, X_arr[test_idx], None, scoring=self.scoring
                        )
                except Exception:
                    if np.isnan(self.error_score):
                        sc = float("nan")
                    else:
                        raise
                split_scores[s_idx, c_idx] = sc

        mean_scores = np.nanmean(split_scores, axis=0)
        std_scores = np.nanstd(split_scores, axis=0)
        ranks = _rank_scores(np.nan_to_num(mean_scores, nan=-np.inf))

        cv_results: Dict[str, Any] = {
            "params": candidates,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            "rank_test_score": ranks,
        }
        for s_idx in range(n_splits):
            cv_results[f"split{s_idx}_test_score"] = split_scores[s_idx]
        self.cv_results_ = cv_results

        best_idx = int(np.nanargmax(mean_scores))
        self.best_index_ = best_idx
        self.best_score_ = float(mean_scores[best_idx])
        self.best_params_ = dict(candidates[best_idx])

        if self.refit:
            best_est = clone_estimator(self.estimator)
            _set_estimator_params(best_est, **self.best_params_)
            if y_arr is not None:
                best_est.fit(X_arr, y_arr)
            else:
                best_est.fit(X_arr)
            self.best_estimator_ = best_est
        else:
            self.best_estimator_ = None

        return self

    def _check_best_estimator(self) -> Any:
        _check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_

    @property
    def classes_(self) -> Any:
        est = self._check_best_estimator()
        return est.classes_

    def predict(self, X: Any) -> Any:
        """Call predict on the estimator with the best found parameters (＾▽＾)."""
        return self._check_best_estimator().predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Call predict_proba on the estimator with the best found parameters."""
        return self._check_best_estimator().predict_proba(X)

    def decision_function(self, X: Any) -> Any:
        """Call decision_function on the estimator with the best found parameters."""
        return self._check_best_estimator().decision_function(X)

    def transform(self, X: Any) -> Any:
        """Call transform on the estimator with the best found parameters."""
        return self._check_best_estimator().transform(X)

    def score(self, X: Any, y: Optional[Any] = None) -> float:
        """Return the score on the given data using best_estimator_ (o´∀｀o)."""
        est = self._check_best_estimator()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y) if y is not None else None
        return _score_estimator(est, X_arr, y_arr, scoring=self.scoring)


class GridSearchCV(_BaseSearchCV):
    """
    Exhaustive search over specified parameter values for an estimator (＾▽＾).

    Parameters
    ----------
    estimator : estimator object
        Base estimator to optimize.
    param_grid : dict or list of dicts
        Dictionary with parameter names (`str`) as keys and lists of parameter
        settings to try as values.
    scoring : str, callable, or None, default=None
        Strategy to evaluate the performance of the cross-validated model.
    cv : int, cross-validation generator, or iterable, default=5
        Cross-validation splitting strategy.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    verbose : int, default=0
        Controls the verbosity.
    error_score : float, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Union[Dict[str, Sequence[Any]], Sequence[Dict[str, Sequence[Any]]]],
        *,
        scoring: Optional[Union[str, Callable[..., float]]] = None,
        cv: Optional[Any] = 5,
        refit: bool = True,
        verbose: int = 0,
        error_score: float = float("nan"),
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score

    def fit(
        self, X: Any, y: Optional[Any] = None, groups: Optional[Any] = None
    ) -> "GridSearchCV":
        """Run fit with all sets of parameters (*≧ω≦*)."""
        grids = [self.param_grid] if isinstance(self.param_grid, dict) else list(self.param_grid)
        candidates: List[Dict[str, Any]] = []
        for grid in grids:
            if not grid:
                candidates.append({})
                continue
            keys = sorted(grid.keys())
            values = [list(grid[k]) for k in keys]
            for combo in itertools.product(*values):
                candidates.append(dict(zip(keys, combo)))
        return self._run_search(candidates, X, y, groups=groups)

    def __repr__(self) -> str:
        return f"GridSearchCV(estimator={self.estimator!r}, cv={self.cv!r}) (＾▽＾)"


class RandomizedSearchCV(_BaseSearchCV):
    """
    Randomized search on hyperparameters (o´∀｀o).

    Parameters
    ----------
    estimator : estimator object
        Base estimator to optimize.
    param_distributions : dict or list of dicts
        Dictionary with parameter names (`str`) as keys and distributions
        or lists of parameters to try.
    n_iter : int, default=10
        Number of parameter settings that are sampled.
    scoring : str, callable, or None, default=None
        Strategy to evaluate the performance of the cross-validated model.
    cv : int, cross-validation generator, or iterable, default=5
        Cross-validation splitting strategy.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    random_state : int or None, default=None
        Pseudo-random number generator state used for sampling.
    verbose : int, default=0
        Controls the verbosity.
    error_score : float, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        *,
        n_iter: int = 10,
        scoring: Optional[Union[str, Callable[..., float]]] = None,
        cv: Optional[Any] = 5,
        refit: bool = True,
        random_state: Optional[int] = None,
        verbose: int = 0,
        error_score: float = float("nan"),
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score

    def fit(
        self, X: Any, y: Optional[Any] = None, groups: Optional[Any] = None
    ) -> "RandomizedSearchCV":
        """Run fit with randomly sampled sets of parameters (*≧ω≦*)."""
        rng = np.random.RandomState(self.random_state)
        dists = (
            [self.param_distributions]
            if isinstance(self.param_distributions, dict)
            else list(self.param_distributions)
        )
        candidates: List[Dict[str, Any]] = []
        for _ in range(self.n_iter):
            dist = dists[int(rng.randint(0, len(dists)))]
            sample: Dict[str, Any] = {}
            for k, v in dist.items():
                if hasattr(v, "rvs"):
                    sample[k] = v.rvs(random_state=rng)
                else:
                    items = list(v)
                    sample[k] = items[int(rng.randint(0, len(items)))]
            candidates.append(sample)
        return self._run_search(candidates, X, y, groups=groups)

    def __repr__(self) -> str:
        return (
            f"RandomizedSearchCV(estimator={self.estimator!r}, n_iter={self.n_iter}) (o´∀｀o)"
        )


def _extract_feature_importances(estimator: Any, norm_order: int = 1) -> np.ndarray:
    """Extract 1D non-negative feature importances from coef_ or feature_importances_."""
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        arr = np.asarray(coef, dtype=np.float64)
        if arr.ndim == 1:
            return np.abs(arr)
        if arr.ndim == 2:
            if norm_order == 1:
                return np.sum(np.abs(arr), axis=0)
            return np.linalg.norm(arr, ord=norm_order, axis=0)
        return np.abs(arr.ravel())

    fi = getattr(estimator, "feature_importances_", None)
    if fi is not None:
        return np.asarray(fi, dtype=np.float64).ravel()

    raise ValueError(
        f"Estimator {type(estimator).__name__} does not expose 'coef_' or 'feature_importances_' ( >_< )"
    )


class RFE(_BaseMetaEstimator):
    """
    Feature ranking with recursive feature elimination (＾▽＾).

    Parameters
    ----------
    estimator : object
        Supervised learning estimator with `fit` and `coef_` or `feature_importances_`.
    n_features_to_select : int, float, or None, default=None
        Number of features to select. If None, half of the features are selected.
    step : int or float, default=1
        Number (or fraction if in (0, 1)) of features to remove at each iteration.
    verbose : int, default=0
        Controls verbosity.

    Attributes
    ----------
    support_ : numpy.ndarray of shape (n_features,)
        Mask of selected features.
    ranking_ : numpy.ndarray of shape (n_features,)
        Feature ranking, such that `ranking_[i]` is the ranking of feature `i` (1 = selected).
    estimator_ : object
        Estimator fitted on the selected set of features.
    """

    def __init__(
        self,
        estimator: Any,
        n_features_to_select: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    def fit(self, X: Any, y: Optional[Any] = None) -> "RFE":
        """Fit the RFE model and automatically tune the number of selected features (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y) if y is not None else None
        n_features = X_arr.shape[1]
        self.n_features_in_ = int(n_features)

        if self.n_features_to_select is None:
            n_target = max(1, n_features // 2)
        elif isinstance(self.n_features_to_select, float) and 0.0 < self.n_features_to_select <= 1.0:
            n_target = max(1, int(n_features * self.n_features_to_select))
        else:
            n_target = max(1, min(int(self.n_features_to_select), n_features))

        if isinstance(self.step, float) and 0.0 < self.step < 1.0:
            step_size = max(1, int(self.step * n_features))
        else:
            step_size = max(1, int(self.step))

        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)

        while int(np.sum(support)) > n_target:
            features = np.where(support)[0]
            est = clone_estimator(self.estimator)
            if y_arr is not None:
                est.fit(X_arr[:, features], y_arr)
            else:
                est.fit(X_arr[:, features])

            importances = _extract_feature_importances(est)
            ranks = np.argsort(importances, kind="mergesort")
            threshold = min(step_size, int(np.sum(support)) - n_target)

            eliminated = features[ranks[:threshold]]
            support[eliminated] = False
            ranking[np.logical_not(support)] += 1

        # Fit final estimator on selected features
        selected = np.where(support)[0]
        final_est = clone_estimator(self.estimator)
        if y_arr is not None:
            final_est.fit(X_arr[:, selected], y_arr)
        else:
            final_est.fit(X_arr[:, selected])

        self.support_ = support
        self.ranking_ = ranking
        self.n_features_ = int(np.sum(support))
        self.estimator_ = final_est
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask, or integer index, of the features selected (＾▽＾)."""
        _check_is_fitted(self, ["support_"])
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()

    def transform(self, X: Any) -> np.ndarray:
        """Reduce X to the selected features (*≧ω≦*)."""
        _check_is_fitted(self, ["support_"])
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr[:, self.support_]

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it (o´∀｀o)."""
        return self.fit(X, y).transform(X)

    def predict(self, X: Any) -> Any:
        """Reduce X to selected features and predict using fitted estimator."""
        _check_is_fitted(self, ["estimator_", "support_"])
        return self.estimator_.predict(self.transform(X))

    def predict_proba(self, X: Any) -> Any:
        """Reduce X to selected features and predict probabilities."""
        _check_is_fitted(self, ["estimator_", "support_"])
        return self.estimator_.predict_proba(self.transform(X))

    def decision_function(self, X: Any) -> Any:
        """Reduce X to selected features and compute decision function."""
        _check_is_fitted(self, ["estimator_", "support_"])
        return self.estimator_.decision_function(self.transform(X))

    def score(self, X: Any, y: Any) -> float:
        """Reduce X to selected features and return score of fitted estimator."""
        _check_is_fitted(self, ["estimator_", "support_"])
        return float(self.estimator_.score(self.transform(X), y))

    def __repr__(self) -> str:
        return (
            f"RFE(estimator={self.estimator!r}, "
            f"n_features_to_select={self.n_features_to_select!r}) (＾▽＾)"
        )


def _calculate_threshold(
    importances: np.ndarray, threshold: Optional[Union[str, float, int]]
) -> float:
    if threshold is None:
        return float(np.mean(importances))
    if isinstance(threshold, (int, float, np.number)):
        return float(threshold)
    if isinstance(threshold, str):
        scale = 1.0
        expr = threshold.strip().lower()
        if "*" in expr:
            scale_str, expr = expr.split("*", 1)
            scale = float(scale_str.strip())
            expr = expr.strip()
        if expr == "mean":
            return scale * float(np.mean(importances))
        if expr == "median":
            return scale * float(np.median(importances))
        return float(threshold)
    raise TypeError(f"Unsupported threshold type: {type(threshold)} ( >_< )")


class SelectFromModel(_BaseMetaEstimator):
    """
    Meta-transformer for selecting features based on importance weights (o´∀｀o).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    threshold : str, float, or None, default=None
        Threshold value to use for feature selection ('mean', 'median', float, or None).
    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor directly.
    norm_order : int, default=1
        Order of the norm used to filter the vectors of coefficients below `threshold`.
    max_features : int or None, default=None
        The maximum number of features to select.

    Attributes
    ----------
    estimator_ : object
        The fitted base estimator.
    threshold_ : float
        The threshold value used for feature selection.
    support_ : numpy.ndarray
        Boolean mask of selected features.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        threshold: Optional[Union[str, float, int]] = None,
        prefit: bool = False,
        norm_order: int = 1,
        max_features: Optional[int] = None,
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features

    def fit(self, X: Any, y: Optional[Any] = None) -> "SelectFromModel":
        """Fit the SelectFromModel meta-transformer (*≧ω≦*)."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = int(X_arr.shape[1])

        if self.prefit:
            self.estimator_ = self.estimator
        else:
            est = clone_estimator(self.estimator)
            if y is not None:
                est.fit(X_arr, np.asarray(y))
            else:
                est.fit(X_arr)
            self.estimator_ = est

        importances = _extract_feature_importances(
            self.estimator_, norm_order=self.norm_order
        )
        self.threshold_ = _calculate_threshold(importances, self.threshold)
        support = importances >= self.threshold_

        if self.max_features is not None and int(np.sum(support)) > self.max_features:
            kept_idx = np.where(support)[0]
            top_order = np.argsort(-importances[kept_idx], kind="mergesort")[
                : self.max_features
            ]
            new_support = np.zeros_like(support, dtype=bool)
            new_support[kept_idx[top_order]] = True
            support = new_support

        self.support_ = support
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask, or integer index, of the features selected (＾▽＾)."""
        if self.prefit and not hasattr(self, "support_"):
            importances = _extract_feature_importances(
                self.estimator, norm_order=self.norm_order
            )
            thresh = _calculate_threshold(importances, self.threshold)
            support = importances >= thresh
            if self.max_features is not None and int(np.sum(support)) > self.max_features:
                kept_idx = np.where(support)[0]
                top_order = np.argsort(-importances[kept_idx], kind="mergesort")[
                    : self.max_features
                ]
                new_support = np.zeros_like(support, dtype=bool)
                new_support[kept_idx[top_order]] = True
                support = new_support
            self.estimator_ = self.estimator
            self.threshold_ = thresh
            self.support_ = support

        _check_is_fitted(self, ["support_"])
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()

    def transform(self, X: Any) -> np.ndarray:
        """Reduce X to the selected features (*≧ω≦*)."""
        mask = self.get_support(indices=False)
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr[:, mask]

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it (o´∀｀o)."""
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return (
            f"SelectFromModel(estimator={self.estimator!r}, "
            f"threshold={self.threshold!r}) (o´∀｀o)"
        )


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
    "cross_val_score",
    "RFE",
    "SelectFromModel",
]
