"""
OchreML Preprocessing, Imputation, Feature Extraction & Compose Parity Suite (＾▽＾)
Verifies numerical parity (max |diff| < 1e-10) against scikit-learn, plus .dit and pickle round-trips.
"""

import pickle
import numpy as np
import pandas as pd
import pytest

from sklearn import compose as sk_compose
from sklearn import feature_extraction as sk_fe
from sklearn import impute as sk_impute
from sklearn import preprocessing as sk_pre
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SkIterativeImputer
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge

import ochreml
from ochreml import compose as oc_compose
from ochreml import feature_extraction as oc_fe
from ochreml import impute as oc_impute
from ochreml import preprocessing as oc_pre


def assert_parity(a, b, tol=1e-10):
    a_arr = a.toarray() if hasattr(a, "toarray") else np.asarray(a, dtype=np.float64)
    b_arr = b.toarray() if hasattr(b, "toarray") else np.asarray(b, dtype=np.float64)
    assert a_arr.shape == b_arr.shape
    mask = np.isnan(a_arr) & np.isnan(b_arr)
    diff = np.abs(np.where(mask, 0.0, a_arr) - np.where(mask, 0.0, b_arr))
    assert float(np.max(diff)) < tol, f"Max diff {np.max(diff)} >= {tol}"


def check_dit_and_pickle(tmp_path, obj, transform_fn):
    expected = transform_fn(obj)

    dit_path = str(tmp_path / f"{type(obj).__name__}.dit")
    ochreml.save(obj, dit_path)
    loaded = ochreml.load(dit_path)
    assert_parity(transform_fn(loaded), expected)

    cls_loaded = type(obj).load(dit_path)
    assert_parity(transform_fn(cls_loaded), expected)

    pickled = pickle.loads(pickle.dumps(obj))
    assert_parity(transform_fn(pickled), expected)


# =========================================================================
# 1. Scaling & Normalization
# =========================================================================

def test_scalers_and_stateless_parity(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(loc=5.0, scale=3.0, size=(80, 5))
    X_test = rng.normal(loc=4.0, scale=2.0, size=(20, 5))
    sw = rng.uniform(0.5, 3.0, size=80)

    # StandardScaler (with sample_weight, partial_fit, inverse_transform)
    for with_mean, with_std in [(True, True), (True, False), (False, True)]:
        oc = oc_pre.StandardScaler(with_mean=with_mean, with_std=with_std).fit(X, sample_weight=sw)
        sk = sk_pre.StandardScaler(with_mean=with_mean, with_std=with_std).fit(X, sample_weight=sw)
        assert_parity(oc.transform(X_test), sk.transform(X_test))
        assert_parity(oc.inverse_transform(oc.transform(X_test)), X_test)
        if with_std:
            assert_parity(oc.var_, sk.var_)
            assert_parity(oc.scale_, sk.scale_)
        assert_parity(oc.mean_, sk.mean_)
        check_dit_and_pickle(tmp_path, oc, lambda m: m.transform(X_test))

    oc_pf = oc_pre.StandardScaler().partial_fit(X[:40]).partial_fit(X[40:])
    sk_pf = sk_pre.StandardScaler().partial_fit(X[:40]).partial_fit(X[40:])
    assert_parity(oc_pf.transform(X_test), sk_pf.transform(X_test))

    # MinMaxScaler
    for clip in [False, True]:
        oc_mm = oc_pre.MinMaxScaler(feature_range=(-2.0, 3.0), clip=clip).fit(X)
        sk_mm = sk_pre.MinMaxScaler(feature_range=(-2.0, 3.0), clip=clip).fit(X)
        assert_parity(oc_mm.transform(X_test), sk_mm.transform(X_test))
        if not clip:
            assert_parity(oc_mm.inverse_transform(oc_mm.transform(X_test)), X_test)
        check_dit_and_pickle(tmp_path, oc_mm, lambda m: m.transform(X_test))

    # MaxAbsScaler
    oc_ma = oc_pre.MaxAbsScaler().fit(X)
    sk_ma = sk_pre.MaxAbsScaler().fit(X)
    assert_parity(oc_ma.transform(X_test), sk_ma.transform(X_test))
    assert_parity(oc_ma.inverse_transform(oc_ma.transform(X_test)), X_test)
    check_dit_and_pickle(tmp_path, oc_ma, lambda m: m.transform(X_test))

    # RobustScaler
    for uv in [False, True]:
        oc_rs = oc_pre.RobustScaler(quantile_range=(15.0, 85.0), unit_variance=uv).fit(X)
        sk_rs = sk_pre.RobustScaler(quantile_range=(15.0, 85.0), unit_variance=uv).fit(X)
        assert_parity(oc_rs.transform(X_test), sk_rs.transform(X_test))
        assert_parity(oc_rs.inverse_transform(oc_rs.transform(X_test)), X_test)
        check_dit_and_pickle(tmp_path, oc_rs, lambda m: m.transform(X_test))

    # Normalizer & normalize
    for norm in ["l1", "l2", "max"]:
        oc_n = oc_pre.Normalizer(norm=norm).fit(X)
        sk_n = sk_pre.Normalizer(norm=norm).fit(X)
        assert_parity(oc_n.transform(X_test), sk_n.transform(X_test))
        check_dit_and_pickle(tmp_path, oc_n, lambda m: m.transform(X_test))

        o_res, o_norms = oc_pre.normalize(X_test, norm=norm, axis=1, return_norm=True)
        s_res, s_norms = sk_pre.normalize(X_test, norm=norm, axis=1, return_norm=True)
        assert_parity(o_res, s_res)
        assert_parity(o_norms, s_norms)

    # Stateless scale functions (axis=0 and axis=1)
    for ax in [0, 1]:
        assert_parity(oc_pre.scale(X, axis=ax), sk_pre.scale(X, axis=ax))
        assert_parity(oc_pre.minmax_scale(X, axis=ax), sk_pre.minmax_scale(X, axis=ax))
        assert_parity(oc_pre.maxabs_scale(X, axis=ax), sk_pre.maxabs_scale(X, axis=ax))
        assert_parity(oc_pre.robust_scale(X, axis=ax), sk_pre.robust_scale(X, axis=ax))


# =========================================================================
# 2. Categorical & Target Encoding
# =========================================================================

def test_categorical_and_target_encoders(tmp_path):
    X_cat = np.array([
        ["cat", "red", 1],
        ["dog", "blue", 2],
        ["cat", "green", 1],
        ["bird", "red", 3],
        ["dog", "red", 2],
        ["cat", "blue", 1],
    ], dtype=object)
    X_test = np.array([
        ["dog", "green", 3],
        ["unknown", "red", 1],
    ], dtype=object)

    # OneHotEncoder (drop='first', sparse/dense, handle_unknown='ignore', min_frequency)
    oc_ohe = oc_pre.OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_cat)
    sk_ohe = sk_pre.OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_cat)
    assert_parity(oc_ohe.transform(X_test), sk_ohe.transform(X_test))
    assert np.array_equal(oc_ohe.inverse_transform(oc_ohe.transform(X_cat)), X_cat)
    check_dit_and_pickle(tmp_path, oc_ohe, lambda m: m.transform(X_test))

    oc_ohe_sp = oc_pre.OneHotEncoder(drop="first", sparse_output=True).fit(X_cat)
    sk_ohe_sp = sk_pre.OneHotEncoder(drop="first", sparse_output=True).fit(X_cat)
    assert_parity(oc_ohe_sp.transform(X_cat), sk_ohe_sp.transform(X_cat))
    assert np.array_equal(oc_ohe_sp.inverse_transform(oc_ohe_sp.transform(X_cat)), X_cat)

    # OrdinalEncoder
    oc_ord = oc_pre.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_cat)
    sk_ord = sk_pre.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_cat)
    assert_parity(oc_ord.transform(X_test), sk_ord.transform(X_test))
    assert np.array_equal(oc_ord.inverse_transform(oc_ord.transform(X_cat)), X_cat)
    check_dit_and_pickle(tmp_path, oc_ord, lambda m: m.transform(X_test))

    # LabelEncoder
    y_lbl = ["apple", "banana", "apple", "cherry"]
    oc_le = oc_pre.LabelEncoder().fit(y_lbl)
    sk_le = sk_pre.LabelEncoder().fit(y_lbl)
    assert_parity(oc_le.transform(y_lbl), sk_le.transform(y_lbl))
    assert list(oc_le.inverse_transform(oc_le.transform(y_lbl))) == y_lbl
    check_dit_and_pickle(tmp_path, oc_le, lambda m: m.transform(y_lbl))

    # LabelBinarizer & label_binarize
    oc_lb = oc_pre.LabelBinarizer(neg_label=-1, pos_label=2).fit(y_lbl)
    sk_lb = sk_pre.LabelBinarizer(neg_label=-1, pos_label=2).fit(y_lbl)
    assert_parity(oc_lb.transform(y_lbl), sk_lb.transform(y_lbl))
    assert list(oc_lb.inverse_transform(oc_lb.transform(y_lbl))) == y_lbl
    check_dit_and_pickle(tmp_path, oc_lb, lambda m: m.transform(y_lbl))

    assert_parity(
        oc_pre.label_binarize(["a", "b", "c"], classes=["a", "b", "c"]),
        sk_pre.label_binarize(["a", "b", "c"], classes=["a", "b", "c"]),
    )

    # MultiLabelBinarizer
    y_ml = [("sci-fi", "thriller"), ("comedy",), ("sci-fi", "comedy")]
    oc_mlb = oc_pre.MultiLabelBinarizer().fit(y_ml)
    sk_mlb = sk_pre.MultiLabelBinarizer().fit(y_ml)
    assert_parity(oc_mlb.transform(y_ml), sk_mlb.transform(y_ml))
    assert oc_mlb.inverse_transform(oc_mlb.transform(y_ml)) == sk_mlb.inverse_transform(sk_mlb.transform(y_ml))
    check_dit_and_pickle(tmp_path, oc_mlb, lambda m: m.transform(y_ml))

    # TargetEncoder (continuous & binary & multiclass)
    y_cont = np.array([1.5, 2.5, 1.8, 10.1, 3.2, 2.1])
    oc_te = oc_pre.TargetEncoder(target_type="continuous", smooth="auto", cv=3, random_state=0).fit(X_cat[:, :2], y_cont)
    sk_te = sk_pre.TargetEncoder(target_type="continuous", smooth="auto", cv=3, random_state=0).fit(X_cat[:, :2], y_cont)
    assert_parity(oc_te.transform(X_test[:, :2]), sk_te.transform(X_test[:, :2]))
    check_dit_and_pickle(tmp_path, oc_te, lambda m: m.transform(X_test[:, :2]))
    cv_out = oc_pre.TargetEncoder(target_type="continuous", cv=3, random_state=42).fit_transform(X_cat[:, :2], y_cont)
    assert cv_out.shape == (6, 2)


# =========================================================================
# 3. Non-Linear Distributions, Discretization & Feature Engineering
# =========================================================================

def test_nonlinear_discretization_and_feature_engineering(tmp_path):
    rng = np.random.default_rng(12)
    X = np.exp(rng.normal(size=(60, 3))) + 1.0
    X_test = np.exp(rng.normal(size=(15, 3))) + 1.0

    # PowerTransformer (yeo-johnson & box-cox) & power_transform
    for method in ["yeo-johnson", "box-cox"]:
        oc_pt = oc_pre.PowerTransformer(method=method).fit(X)
        sk_pt = sk_pre.PowerTransformer(method=method).fit(X)
        assert_parity(oc_pt.lambdas_, sk_pt.lambdas_, tol=1e-7)
        assert_parity(oc_pt.transform(X_test), sk_pt.transform(X_test), tol=1e-7)
        assert_parity(oc_pt.inverse_transform(oc_pt.transform(X_test)), X_test, tol=1e-7)
        check_dit_and_pickle(tmp_path, oc_pt, lambda m: m.transform(X_test))

    assert_parity(
        oc_pre.power_transform(X, method="yeo-johnson"),
        sk_pre.power_transform(X, method="yeo-johnson"),
        tol=1e-7,
    )

    # QuantileTransformer & quantile_transform
    for dist in ["uniform", "normal"]:
        oc_qt = oc_pre.QuantileTransformer(n_quantiles=30, output_distribution=dist, random_state=0).fit(X)
        sk_qt = sk_pre.QuantileTransformer(n_quantiles=30, output_distribution=dist, random_state=0).fit(X)
        assert_parity(oc_qt.transform(X_test), sk_qt.transform(X_test))
        assert_parity(
            oc_qt.inverse_transform(oc_qt.transform(X_test)),
            sk_qt.inverse_transform(sk_qt.transform(X_test)),
        )
        check_dit_and_pickle(tmp_path, oc_qt, lambda m: m.transform(X_test))

    assert_parity(
        oc_pre.quantile_transform(X, n_quantiles=30, output_distribution="uniform"),
        sk_pre.quantile_transform(X, n_quantiles=30, output_distribution="uniform"),
    )

    # FunctionTransformer
    oc_ft = oc_pre.FunctionTransformer(np.log1p, np.expm1, validate=True).fit(X)
    sk_ft = sk_pre.FunctionTransformer(np.log1p, np.expm1, validate=True).fit(X)
    assert_parity(oc_ft.transform(X_test), sk_ft.transform(X_test))
    assert_parity(oc_ft.inverse_transform(oc_ft.transform(X_test)), X_test)
    check_dit_and_pickle(tmp_path, oc_ft, lambda m: m.transform(X_test))

    # KBinsDiscretizer
    for strat in ["uniform", "quantile", "kmeans"]:
        for enc in ["ordinal", "onehot-dense", "onehot"]:
            oc_kb = oc_pre.KBinsDiscretizer(n_bins=4, encode=enc, strategy=strat).fit(X)
            sk_kb = sk_pre.KBinsDiscretizer(
                n_bins=4, encode=enc, strategy=strat, quantile_method="averaged_inverted_cdf"
            ).fit(X)
            if strat != "kmeans":
                assert_parity(oc_kb.transform(X_test), sk_kb.transform(X_test))
            else:
                assert oc_kb.transform(X_test).shape == sk_kb.transform(X_test).shape
            check_dit_and_pickle(tmp_path, oc_kb, lambda m: m.transform(X_test))

    # Binarizer & binarize
    oc_bin = oc_pre.Binarizer(threshold=2.5).fit(X)
    sk_bin = sk_pre.Binarizer(threshold=2.5).fit(X)
    assert_parity(oc_bin.transform(X_test), sk_bin.transform(X_test))
    assert_parity(oc_pre.binarize(X_test, threshold=2.5), sk_pre.binarize(X_test, threshold=2.5))
    check_dit_and_pickle(tmp_path, oc_bin, lambda m: m.transform(X_test))

    # PolynomialFeatures
    oc_pf = oc_pre.PolynomialFeatures(degree=3, include_bias=True).fit(X)
    sk_pf = sk_pre.PolynomialFeatures(degree=3, include_bias=True).fit(X)
    assert_parity(oc_pf.transform(X_test), sk_pf.transform(X_test))
    assert oc_pf.get_feature_names_out().tolist() == sk_pf.get_feature_names_out().tolist()
    check_dit_and_pickle(tmp_path, oc_pf, lambda m: m.transform(X_test))

    # SplineTransformer
    for ext in ["constant", "linear", "continue", "periodic"]:
        oc_sp = oc_pre.SplineTransformer(n_knots=5, degree=3, extrapolation=ext).fit(X)
        sk_sp = sk_pre.SplineTransformer(n_knots=5, degree=3, extrapolation=ext).fit(X)
        assert_parity(oc_sp.transform(X_test), sk_sp.transform(X_test))
        check_dit_and_pickle(tmp_path, oc_sp, lambda m: m.transform(X_test))

    # KernelCenterer & add_dummy_feature
    K_train = X[:20] @ X[:20].T
    K_test = X_test @ X[:20].T
    oc_kc = oc_pre.KernelCenterer().fit(K_train)
    sk_kc = sk_pre.KernelCenterer().fit(K_train)
    assert_parity(oc_kc.transform(K_test), sk_kc.transform(K_test))
    check_dit_and_pickle(tmp_path, oc_kc, lambda m: m.transform(K_test))

    assert_parity(oc_pre.add_dummy_feature(X_test, value=2.5), sk_pre.add_dummy_feature(X_test, value=2.5))


# =========================================================================
# 4. Imputation
# =========================================================================

def test_imputers_parity(tmp_path):
    X_nan = np.array([
        [1.0, 2.0, np.nan, 4.0],
        [3.0, np.nan, 3.0, 2.0],
        [np.nan, 6.0, 5.0, 4.0],
        [8.0, 8.0, 7.0, np.nan],
        [2.0, 2.0, 5.0, 4.0],
    ])
    X_test = np.array([[np.nan, 4.0, np.nan, 3.0], [1.0, 2.0, 3.0, 4.0]])

    for strat in ["mean", "median", "most_frequent", "constant"]:
        fv = -99.0 if strat == "constant" else None
        oc_si = oc_impute.SimpleImputer(strategy=strat, fill_value=fv, add_indicator=True).fit(X_nan)
        sk_si = sk_impute.SimpleImputer(strategy=strat, fill_value=fv, add_indicator=True).fit(X_nan)
        assert_parity(oc_si.transform(X_test), sk_si.transform(X_test))
        check_dit_and_pickle(tmp_path, oc_si, lambda m: m.transform(X_test))

    oc_mi = oc_impute.MissingIndicator().fit(X_nan)
    sk_mi = sk_impute.MissingIndicator().fit(X_nan)
    assert_parity(oc_mi.transform(X_test), sk_mi.transform(X_test))
    check_dit_and_pickle(tmp_path, oc_mi, lambda m: m.transform(X_test))

    for w in ["uniform", "distance"]:
        oc_knn = oc_impute.KNNImputer(n_neighbors=2, weights=w, add_indicator=True).fit(X_nan)
        sk_knn = sk_impute.KNNImputer(n_neighbors=2, weights=w, add_indicator=True).fit(X_nan)
        assert_parity(oc_knn.transform(X_test), sk_knn.transform(X_test))
        check_dit_and_pickle(tmp_path, oc_knn, lambda m: m.transform(X_test))

    # IterativeImputer using OchreML Ridge vs sklearn Ridge
    oc_iter = oc_impute.IterativeImputer(estimator=ochreml.Ridge(alpha=1.0), max_iter=5, tol=1e-6)
    sk_iter = SkIterativeImputer(estimator=SkRidge(alpha=1.0), max_iter=5, tol=1e-6)
    assert_parity(oc_iter.fit_transform(X_nan), sk_iter.fit_transform(X_nan), tol=1e-8)
    assert_parity(oc_iter.transform(X_test), sk_iter.transform(X_test), tol=1e-8)
    check_dit_and_pickle(tmp_path, oc_iter, lambda m: m.transform(X_test))


# =========================================================================
# 5. Feature Extraction & Text Vectorization
# =========================================================================

def test_feature_extraction_and_text_parity(tmp_path):
    dicts = [
        {"city": "Jakarta", "temp": 31.0, "humid": 80},
        {"city": "Bandung", "temp": 22.0},
        {"city": "Jakarta", "humid": 75.0},
    ]
    oc_dv = oc_fe.DictVectorizer().fit(dicts)
    sk_dv = sk_fe.DictVectorizer().fit(dicts)
    assert_parity(oc_dv.transform(dicts), sk_dv.transform(dicts))
    assert oc_dv.inverse_transform(oc_dv.transform(dicts)) == sk_dv.inverse_transform(sk_dv.transform(dicts))
    check_dit_and_pickle(tmp_path, oc_dv, lambda m: m.transform(dicts))

    oc_fh = oc_fe.FeatureHasher(n_features=16)
    sk_fh = sk_fe.FeatureHasher(n_features=16)
    assert_parity(oc_fh.transform(dicts), sk_fh.transform(dicts))
    check_dit_and_pickle(tmp_path, oc_fh, lambda m: m.transform(dicts))

    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Rust and PyO3 make machine learning blazing fast",
        "OchreML supports quick brown machine learning pipelines",
    ]
    oc_cv = oc_fe.text.CountVectorizer(ngram_range=(1, 2), stop_words="english").fit(docs)
    sk_cv = sk_fe.text.CountVectorizer(ngram_range=(1, 2), stop_words="english").fit(docs)
    assert oc_cv.vocabulary_ == sk_cv.vocabulary_
    assert_parity(oc_cv.transform(docs), sk_cv.transform(docs))
    check_dit_and_pickle(tmp_path, oc_cv, lambda m: m.transform(docs))

    counts = oc_cv.transform(docs)
    oc_tt = oc_fe.text.TfidfTransformer(sublinear_tf=True).fit(counts)
    sk_tt = sk_fe.text.TfidfTransformer(sublinear_tf=True).fit(counts)
    assert_parity(oc_tt.transform(counts), sk_tt.transform(counts))
    check_dit_and_pickle(tmp_path, oc_tt, lambda m: m.transform(counts))

    oc_tv = oc_fe.text.TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(docs)
    sk_tv = sk_fe.text.TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(docs)
    assert_parity(oc_tv.transform(docs), sk_tv.transform(docs))
    check_dit_and_pickle(tmp_path, oc_tv, lambda m: m.transform(docs))

    oc_hv = oc_fe.text.HashingVectorizer(n_features=32, ngram_range=(1, 2))
    sk_hv = sk_fe.text.HashingVectorizer(n_features=32, ngram_range=(1, 2))
    assert_parity(oc_hv.transform(docs), sk_hv.transform(docs))
    check_dit_and_pickle(tmp_path, oc_hv, lambda m: m.transform(docs))


# =========================================================================
# 6. Compose (ColumnTransformer & TransformedTargetRegressor)
# =========================================================================

def test_compose_parity(tmp_path):
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "a", "c", "b"],
    })

    oc_ct = oc_compose.ColumnTransformer([
        ("num", oc_pre.StandardScaler(), oc_compose.make_column_selector(dtype_include=np.number)),
        ("cat", oc_pre.OneHotEncoder(sparse_output=False), ["cat"]),
    ])
    sk_ct = sk_compose.ColumnTransformer([
        ("num", sk_pre.StandardScaler(), sk_compose.make_column_selector(dtype_include=np.number)),
        ("cat", sk_pre.OneHotEncoder(sparse_output=False), ["cat"]),
    ])
    assert_parity(oc_ct.fit_transform(df), sk_ct.fit_transform(df))
    check_dit_and_pickle(tmp_path, oc_ct, lambda m: m.transform(df))

    oc_mct = oc_compose.make_column_transformer(
        (oc_pre.MinMaxScaler(), ["num1", "num2"]),
        remainder="drop",
    )
    sk_mct = sk_compose.make_column_transformer(
        (sk_pre.MinMaxScaler(), ["num1", "num2"]),
        remainder="drop",
    )
    assert_parity(oc_mct.fit_transform(df), sk_mct.fit_transform(df))

    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.exp(2.0 * X.ravel() + 1.0)
    oc_ttr = oc_compose.TransformedTargetRegressor(
        regressor=ochreml.LinearRegression(),
        func=np.log,
        inverse_func=np.exp,
    ).fit(X, y)
    sk_ttr = sk_compose.TransformedTargetRegressor(
        regressor=SkLinearRegression(),
        func=np.log,
        inverse_func=np.exp,
    ).fit(X, y)
    assert_parity(oc_ttr.predict(X), sk_ttr.predict(X))
    check_dit_and_pickle(tmp_path, oc_ttr, lambda m: m.predict(X))


def test_top_level_exports_and_edge_cases():
    # Top-level ochreml exports
    for name in [
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer",
        "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder",
        "LabelBinarizer", "MultiLabelBinarizer", "PowerTransformer", "QuantileTransformer",
        "FunctionTransformer", "KBinsDiscretizer", "Binarizer", "PolynomialFeatures",
        "SplineTransformer", "KernelCenterer", "SimpleImputer", "MissingIndicator",
        "KNNImputer", "IterativeImputer", "DictVectorizer", "FeatureHasher",
        "CountVectorizer", "TfidfTransformer", "TfidfVectorizer", "HashingVectorizer",
        "ColumnTransformer", "TransformedTargetRegressor",
        "make_column_transformer", "make_column_selector",
    ]:
        assert hasattr(ochreml, name), f"Missing top-level export: {name}"

    # Categorical SimpleImputer (most_frequent & constant)
    X_str = np.array([["a", None], ["a", "x"], ["b", "x"], [None, "y"]], dtype=object)
    oc_si = oc_impute.SimpleImputer(missing_values=None, strategy="most_frequent").fit(X_str)
    sk_si = sk_impute.SimpleImputer(missing_values=None, strategy="most_frequent").fit(X_str)
    assert np.array_equal(oc_si.transform(X_str), sk_si.transform(X_str))

    # TargetEncoder binary & multiclass parity
    X_cat = np.array([["a"], ["a"], ["b"], ["b"], ["c"], ["c"]], dtype=object)
    y_bin = np.array([0, 1, 0, 0, 1, 1])
    oc_te_b = oc_pre.TargetEncoder(target_type="binary", smooth=1.0).fit(X_cat, y_bin)
    sk_te_b = sk_pre.TargetEncoder(target_type="binary", smooth=1.0).fit(X_cat, y_bin)
    assert_parity(oc_te_b.transform(X_cat), sk_te_b.transform(X_cat))

    y_mc = np.array(["x", "y", "z", "x", "y", "z"])
    oc_te_m = oc_pre.TargetEncoder(target_type="multiclass", smooth=1.0).fit(X_cat, y_mc)
    sk_te_m = sk_pre.TargetEncoder(target_type="multiclass", smooth=1.0).fit(X_cat, y_mc)
    assert_parity(oc_te_m.transform(X_cat), sk_te_m.transform(X_cat))

    # Device dispatch & PolynomialFeatures order='F'
    X_num = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ss_cpu = ochreml.StandardScaler(device="cpu").fit_transform(X_num)
    ss_auto = ochreml.StandardScaler(device="auto").fit_transform(X_num)
    assert_parity(ss_cpu, ss_auto)
    pf_f = ochreml.PolynomialFeatures(degree=2, order="F").fit_transform(X_num)
    assert pf_f.flags.f_contiguous


