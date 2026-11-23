"""
OchreML Ensemble Methods Submodule (*≧ω≦*)
Unified Scikit-Learn-compatible exports for Histogram Gradient Boosting,
Random Forests, Extra Trees, Gradient Boosting, AdaBoost, and IsolationForest (＾▽＾).
"""

from .ensemble_boosting import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .ensemble_forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .ensemble_hgb import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from ._meta_ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    VotingClassifier,
)

__all__ = [
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "IsolationForest",
    "BaggingClassifier",
    "BaggingRegressor",
    "VotingClassifier",
    "StackingClassifier",
]

