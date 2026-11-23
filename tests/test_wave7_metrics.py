"""
OchreML Wave 7 Metrics Parity Suite (＾▽＾)
Verifies Regression, Classification, and Clustering metrics against scikit-learn.
"""

import numpy as np
import pytest
from sklearn import metrics as sk_metrics
from sklearn.datasets import make_blobs

from ochreml import metrics as oc_metrics


# ---------------------------------------------------------------------------
# Regression Metrics Parity
# ---------------------------------------------------------------------------


def test_regression_metrics_1d_and_weighted():
    rng = np.random.default_rng(42)
    y_true = rng.normal(10.0, 3.0, size=60)
    y_pred = y_true + rng.normal(0.0, 0.8, size=60)
    sw = rng.uniform(0.5, 2.5, size=60)

    for weights in [None, sw]:
        assert oc_metrics.mean_squared_error(y_true, y_pred, sample_weight=weights) == pytest.approx(
            sk_metrics.mean_squared_error(y_true, y_pred, sample_weight=weights), rel=1e-12
        )
        assert oc_metrics.root_mean_squared_error(
            y_true, y_pred, sample_weight=weights
        ) == pytest.approx(
            sk_metrics.root_mean_squared_error(y_true, y_pred, sample_weight=weights), rel=1e-12
        )
        assert oc_metrics.mean_absolute_error(y_true, y_pred, sample_weight=weights) == pytest.approx(
            sk_metrics.mean_absolute_error(y_true, y_pred, sample_weight=weights), rel=1e-12
        )
        assert oc_metrics.r2_score(y_true, y_pred, sample_weight=weights) == pytest.approx(
            sk_metrics.r2_score(y_true, y_pred, sample_weight=weights), rel=1e-12
        )
        assert oc_metrics.mean_absolute_percentage_error(
            y_true, y_pred, sample_weight=weights
        ) == pytest.approx(
            sk_metrics.mean_absolute_percentage_error(y_true, y_pred, sample_weight=weights),
            rel=1e-12,
        )
        assert oc_metrics.explained_variance_score(
            y_true, y_pred, sample_weight=weights
        ) == pytest.approx(
            sk_metrics.explained_variance_score(y_true, y_pred, sample_weight=weights), rel=1e-12
        )


@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
def test_regression_metrics_multioutput(multioutput):
    rng = np.random.default_rng(7)
    y_true = rng.uniform(1.0, 10.0, size=(40, 3))
    y_pred = y_true + rng.normal(0.0, 0.5, size=(40, 3))

    for fn_name in [
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_absolute_error",
        "r2_score",
        "mean_absolute_percentage_error",
        "explained_variance_score",
    ]:
        oc_val = getattr(oc_metrics, fn_name)(y_true, y_pred, multioutput=multioutput)
        sk_val = getattr(sk_metrics, fn_name)(y_true, y_pred, multioutput=multioutput)
        np.testing.assert_allclose(oc_val, sk_val, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Classification Metrics Parity
# ---------------------------------------------------------------------------


def test_accuracy_and_confusion_matrix_numeric_and_string():
    yt = ["cat", "dog", "bird", "cat", "dog", "bird", "cat"]
    yp = ["cat", "cat", "bird", "cat", "dog", "dog", "bird"]
    sw = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 1.0])

    assert oc_metrics.accuracy_score(yt, yp) == pytest.approx(sk_metrics.accuracy_score(yt, yp))
    assert oc_metrics.accuracy_score(yt, yp, normalize=False) == sk_metrics.accuracy_score(
        yt, yp, normalize=False
    )
    assert oc_metrics.accuracy_score(yt, yp, sample_weight=sw) == pytest.approx(
        sk_metrics.accuracy_score(yt, yp, sample_weight=sw)
    )

    for norm in [None, "true", "pred", "all"]:
        cm_oc = oc_metrics.confusion_matrix(yt, yp, sample_weight=sw, normalize=norm)
        cm_sk = sk_metrics.confusion_matrix(yt, yp, sample_weight=sw, normalize=norm)
        np.testing.assert_allclose(cm_oc, cm_sk, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
def test_prf_scores_multiclass(average):
    yt = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0, 2])
    yp = np.array([0, 2, 2, 0, 1, 0, 2, 1, 1, 2])
    sw = np.linspace(0.5, 2.0, len(yt))

    p_oc, r_oc, f_oc, s_oc = oc_metrics.precision_recall_fscore_support(
        yt, yp, beta=1.5, average=average, sample_weight=sw, zero_division=0.0
    )
    p_sk, r_sk, f_sk, s_sk = sk_metrics.precision_recall_fscore_support(
        yt, yp, beta=1.5, average=average, sample_weight=sw, zero_division=0.0
    )
    np.testing.assert_allclose(p_oc, p_sk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r_oc, r_sk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f_oc, f_sk, rtol=1e-12, atol=1e-12)
    if average is None:
        np.testing.assert_allclose(s_oc, s_sk, rtol=1e-12, atol=1e-12)
    else:
        assert s_oc is None


def test_binary_precision_recall_f1_and_report():
    yt = [0, 1, 1, 0, 1, 0, 1, 1]
    yp = [0, 1, 0, 0, 1, 1, 1, 1]

    assert oc_metrics.precision_score(yt, yp) == pytest.approx(sk_metrics.precision_score(yt, yp))
    assert oc_metrics.recall_score(yt, yp) == pytest.approx(sk_metrics.recall_score(yt, yp))
    assert oc_metrics.f1_score(yt, yp) == pytest.approx(sk_metrics.f1_score(yt, yp))

    rep_oc = oc_metrics.classification_report(yt, yp, output_dict=True)
    rep_sk = sk_metrics.classification_report(yt, yp, output_dict=True)
    assert rep_oc["accuracy"] == pytest.approx(rep_sk["accuracy"])
    assert rep_oc["macro avg"]["f1-score"] == pytest.approx(rep_sk["macro avg"]["f1-score"])
    assert rep_oc["weighted avg"]["f1-score"] == pytest.approx(rep_sk["weighted avg"]["f1-score"])

    rep_str = oc_metrics.classification_report(yt, yp, output_dict=False)
    assert "macro avg" in rep_str and "weighted avg" in rep_str


def test_roc_auc_binary_and_multiclass():
    rng = np.random.default_rng(123)
    yt_bin = rng.integers(0, 2, size=50)
    ys_bin = rng.uniform(0, 1, size=50)
    # Inject ties
    ys_bin[:5] = 0.5
    sw = rng.uniform(0.5, 2.0, size=50)

    assert oc_metrics.roc_auc_score(yt_bin, ys_bin, sample_weight=sw) == pytest.approx(
        sk_metrics.roc_auc_score(yt_bin, ys_bin, sample_weight=sw), rel=1e-12
    )

    yt_mc = rng.integers(0, 3, size=60)
    logits = rng.normal(size=(60, 3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    for mc in ["ovr", "ovo"]:
        for avg in ["macro", "weighted"]:
            assert oc_metrics.roc_auc_score(
                yt_mc, probs, average=avg, multi_class=mc
            ) == pytest.approx(
                sk_metrics.roc_auc_score(yt_mc, probs, average=avg, multi_class=mc),
                rel=1e-12,
            )


def test_log_loss_binary_and_multiclass():
    yt_bin = np.array([0, 1, 1, 0, 1])
    yp_bin = np.array([0.1, 0.9, 0.7, 0.2, 0.8])
    assert oc_metrics.log_loss(yt_bin, yp_bin) == pytest.approx(
        sk_metrics.log_loss(yt_bin, yp_bin), rel=1e-12
    )

    yt_mc = np.array([0, 1, 2, 1])
    yp_mc = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
        ]
    )
    assert oc_metrics.log_loss(yt_mc, yp_mc) == pytest.approx(
        sk_metrics.log_loss(yt_mc, yp_mc), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Clustering Metrics Parity
# ---------------------------------------------------------------------------


def test_clustering_metrics_parity():
    X, labels = make_blobs(n_samples=90, n_features=5, centers=3, cluster_std=0.8, random_state=42)

    for metric in ["euclidean", "manhattan", "cosine"]:
        assert oc_metrics.silhouette_score(X, labels, metric=metric, device="cpu") == pytest.approx(
            sk_metrics.silhouette_score(X, labels, metric=metric), rel=1e-10
        )

    assert oc_metrics.davies_bouldin_score(X, labels) == pytest.approx(
        sk_metrics.davies_bouldin_score(X, labels), rel=1e-12
    )
    assert oc_metrics.calinski_harabasz_score(X, labels) == pytest.approx(
        sk_metrics.calinski_harabasz_score(X, labels), rel=1e-12
    )

    rng = np.random.default_rng(9)
    labels_pred = rng.integers(0, 3, size=90)
    assert oc_metrics.adjusted_rand_score(labels, labels_pred) == pytest.approx(
        sk_metrics.adjusted_rand_score(labels, labels_pred), rel=1e-12
    )
    for avg in ["min", "geometric", "arithmetic", "max"]:
        assert oc_metrics.normalized_mutual_info_score(
            labels, labels_pred, average_method=avg
        ) == pytest.approx(
            sk_metrics.normalized_mutual_info_score(labels, labels_pred, average_method=avg),
            rel=1e-12,
        )
