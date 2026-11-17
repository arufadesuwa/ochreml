"""
OchreML Preprocessing & Feature Engineering Submodule (＾▽＾)
Hardware-accelerated scaling, encoding, distribution transforms, discretization,
and polynomial/spline/kernel feature engineering (*≧ω≦*).
"""

from ._encoders import (
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    label_binarize,
)
from ._features import (
    KernelCenterer,
    PolynomialFeatures,
    SplineTransformer,
    add_dummy_feature,
)
from ._scalers import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
    maxabs_scale,
    minmax_scale,
    normalize,
    robust_scale,
    scale,
)
from ._transforms import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    PowerTransformer,
    QuantileTransformer,
    binarize,
    power_transform,
    quantile_transform,
)

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "scale",
    "minmax_scale",
    "maxabs_scale",
    "robust_scale",
    "normalize",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    "label_binarize",
    "PowerTransformer",
    "QuantileTransformer",
    "FunctionTransformer",
    "power_transform",
    "quantile_transform",
    "KBinsDiscretizer",
    "Binarizer",
    "binarize",
    "PolynomialFeatures",
    "SplineTransformer",
    "KernelCenterer",
    "add_dummy_feature",
]

