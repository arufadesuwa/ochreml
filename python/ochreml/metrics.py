"""
OchreML Metrics Module (＾▽＾)
Scikit-Learn-compatible Regression, Classification, and Clustering metrics
backed by Rust and Hardware Backend N3 (*≧ω≦*).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ._ochreml import (
    _metrics_accuracy_score,
    _metrics_adjusted_rand_score,
    _metrics_binary_roc_auc,
    _metrics_calinski_harabasz_score,
    _metrics_confusion_matrix,
    _metrics_davies_bouldin_score,
    _metrics_explained_variance_score,
    _metrics_log_loss,
    _metrics_mean_absolute_error,
    _metrics_mean_absolute_percentage_error,
    _metrics_mean_squared_error,
    _metrics_multiclass_roc_auc,
    _metrics_normalized_mutual_info_score,
    _metrics_prf_support,
    _metrics_r2_score,
    _metrics_root_mean_squared_error,
    _metrics_silhouette_score,
)
from ._utils import process_features, process_sample_weight


# ---------------------------------------------------------------------------
# Internal Array / Label Helpers
# ---------------------------------------------------------------------------


def _prep_x(X: Any) -> np.ndarray:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    return np.ascontiguousarray(X_data, dtype=np.float64)


def _prep_reg_targets(y: Any) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr.reshape(-1, 1), dtype=np.float64), True
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float64), False
    raise ValueError(f"Expected 1D or 2D array for regression targets, got {arr.ndim}D ( >_< )")


def _prep_sw(sample_weight: Optional[Any]) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    is_buf, sw_data = process_sample_weight(sample_weight)
    if is_buf:
        return np.ascontiguousarray(sw_data, dtype=np.float64)
    return np.ascontiguousarray(sw_data, dtype=np.float64)


def _encode_pair_labels(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    yt_arr = np.asarray(y_true)
    yp_arr = np.asarray(y_pred)
    if yt_arr.ndim != 1 or yp_arr.ndim != 1:
        raise ValueError("Classification labels must be 1D arrays ( >_< )")
    if yt_arr.shape[0] != yp_arr.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: [{yt_arr.shape[0]}, {yp_arr.shape[0]}] ( >_< )"
        )

    if labels is None:
        unique_vals = sorted(set(yt_arr.tolist()) | set(yp_arr.tolist()))
    else:
        unique_vals = list(labels)

    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    yt_enc = np.array([mapping.get(v, -1) for v in yt_arr.tolist()], dtype=np.int64)
    yp_enc = np.array([mapping.get(v, -1) for v in yp_arr.tolist()], dtype=np.int64)
    return yt_enc, yp_enc, unique_vals


def _encode_single_labels(y: Any) -> Tuple[np.ndarray, list]:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("Labels must be a 1D array ( >_< )")
    if np.issubdtype(arr.dtype, np.integer) and arr.flags.c_contiguous and arr.dtype == np.int64:
        return arr, sorted(set(arr.tolist()))
    vals = arr.tolist()
    unique_vals = sorted(set(vals))
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    enc = np.array([mapping[v] for v in vals], dtype=np.int64)
    return enc, unique_vals


def _resolve_device(device: Optional[str]) -> str:
    if device is not None and device != "auto":
        return device
    if device is None:
        from . import get_default_device

        return get_default_device()
    return "auto"


def _format_reg_output(res: Sequence[float], multioutput: str) -> Union[float, np.ndarray]:
    if multioutput == "raw_values":
        return np.asarray(res, dtype=np.float64)
    return float(res[0])


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------


def mean_squared_error(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Mean squared error regression loss (＾▽＾)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_mean_squared_error(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


def root_mean_squared_error(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Root mean squared error regression loss (＾▽＾)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_root_mean_squared_error(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


def mean_absolute_error(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Mean absolute error regression loss (＾▽＾)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_mean_absolute_error(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


def mean_absolute_percentage_error(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Mean absolute percentage error (MAPE) regression loss (＾▽＾)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_mean_absolute_percentage_error(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


def r2_score(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Coefficient of determination (R^2) regression score (*≧ω≦*)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_r2_score(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


def explained_variance_score(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """Explained variance regression score function (*≧ω≦*)."""
    yt, _ = _prep_reg_targets(y_true)
    yp, _ = _prep_reg_targets(y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_explained_variance_score(yt, yp, sw, multioutput)
    return _format_reg_output(res, multioutput)


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------


def accuracy_score(
    y_true: Any,
    y_pred: Any,
    normalize: bool = True,
    sample_weight: Optional[Any] = None,
) -> Union[float, int]:
    """Accuracy classification score (＾▽＾)."""
    yt_enc, yp_enc, _ = _encode_pair_labels(y_true, y_pred)
    sw = _prep_sw(sample_weight)
    res = _metrics_accuracy_score(yt_enc, yp_enc, normalize, sw)
    if not normalize and sw is None:
        return int(round(res))
    return float(res)


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    sample_weight: Optional[Any] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification (＾▽＾)."""
    yt_enc, yp_enc, unique_vals = _encode_pair_labels(y_true, y_pred, labels=labels)
    sw = _prep_sw(sample_weight)
    n_labels = len(unique_vals)
    flat = _metrics_confusion_matrix(yt_enc, yp_enc, n_labels, sw, normalize)
    arr = np.asarray(flat, dtype=np.float64).reshape(n_labels, n_labels)
    if normalize is None and sw is None:
        return arr.astype(np.int64)
    return arr


def _resolve_zero_division(zero_division: Union[int, float, str]) -> float:
    if isinstance(zero_division, str):
        if zero_division == "warn":
            return 0.0
        raise ValueError(f"Invalid zero_division='{zero_division}' ( >_< )")
    return float(zero_division)


def precision_recall_fscore_support(
    y_true: Any,
    y_pred: Any,
    beta: float = 1.0,
    labels: Optional[Sequence[Any]] = None,
    pos_label: Any = 1,
    average: Optional[str] = None,
    sample_weight: Optional[Any] = None,
    zero_division: Union[int, float, str] = 0.0,
) -> Tuple[Any, Any, Any, Any]:
    """Compute precision, recall, F-measure and support for each class (*≧ω≦*)."""
    yt_enc, yp_enc, unique_vals = _encode_pair_labels(y_true, y_pred, labels=labels)
    sw = _prep_sw(sample_weight)
    zd = _resolve_zero_division(zero_division)
    n_labels = len(unique_vals)

    if average == "binary":
        if labels is None and n_labels > 2:
            raise ValueError(
                "Target is multiclass but average='binary'. Please choose another average setting, "
                "one of [None, 'micro', 'macro', 'weighted'] ( >_< )"
            )
        if pos_label not in unique_vals:
            raise ValueError(
                f"pos_label={pos_label!r} is not a valid label. Valid labels: {unique_vals!r} ( >_< )"
            )
        pos_idx = unique_vals.index(pos_label)
    else:
        pos_idx = 0

    p, r, f, s = _metrics_prf_support(
        yt_enc, yp_enc, n_labels, float(beta), pos_idx, average, sw, zd
    )
    if average is None:
        sup_dtype = np.int64 if sw is None else np.float64
        return (
            np.asarray(p, dtype=np.float64),
            np.asarray(r, dtype=np.float64),
            np.asarray(f, dtype=np.float64),
            np.asarray(s, dtype=sup_dtype),
        )
    return float(p[0]), float(r[0]), float(f[0]), None


def precision_score(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    pos_label: Any = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[Any] = None,
    zero_division: Union[int, float, str] = 0.0,
) -> Union[float, np.ndarray]:
    """Compute the precision classification score (＾▽＾)."""
    p, _, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return p


def recall_score(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    pos_label: Any = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[Any] = None,
    zero_division: Union[int, float, str] = 0.0,
) -> Union[float, np.ndarray]:
    """Compute the recall classification score (＾▽＾)."""
    _, r, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return r


def f1_score(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    pos_label: Any = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[Any] = None,
    zero_division: Union[int, float, str] = 0.0,
) -> Union[float, np.ndarray]:
    """Compute the F1 score, also known as balanced F-score (＾▽＾)."""
    _, _, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return f


def roc_auc_score(
    y_true: Any,
    y_score: Any,
    average: Optional[str] = "macro",
    sample_weight: Optional[Any] = None,
    multi_class: str = "raise",
    labels: Optional[Sequence[Any]] = None,
) -> Union[float, np.ndarray]:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) (*≧ω≦*)."""
    yt_arr = np.asarray(y_true)
    ys_arr = np.asarray(y_score, dtype=np.float64)
    sw = _prep_sw(sample_weight)

    if yt_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array for roc_auc_score ( >_< )")

    if labels is None:
        unique_vals = sorted(set(yt_arr.tolist()))
    else:
        unique_vals = list(labels)

    if len(unique_vals) < 2:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score is not defined in that case (´；ω；`)"
        )

    if ys_arr.ndim == 1 or (ys_arr.ndim == 2 and ys_arr.shape[1] == 1):
        if len(unique_vals) > 2:
            raise ValueError(
                "y_score must be 2D with shape (n_samples, n_classes) for multiclass ROC AUC ( >_< )"
            )
        pos_val = unique_vals[1]
        yt_bin = np.array([1 if v == pos_val else 0 for v in yt_arr.tolist()], dtype=np.int64)
        ys_1d = np.ascontiguousarray(ys_arr.reshape(-1), dtype=np.float64)
        return float(_metrics_binary_roc_auc(yt_bin, ys_1d, sw))

    if ys_arr.ndim == 2:
        if len(unique_vals) == 2 and multi_class == "raise":
            # Binary case with 2-column probability matrix: use positive class column
            pos_val = unique_vals[1]
            yt_bin = np.array([1 if v == pos_val else 0 for v in yt_arr.tolist()], dtype=np.int64)
            ys_1d = np.ascontiguousarray(ys_arr[:, 1], dtype=np.float64)
            return float(_metrics_binary_roc_auc(yt_bin, ys_1d, sw))
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr') for multiclass ROC AUC ( >_< )")
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        yt_enc = np.array([mapping.get(v, -1) for v in yt_arr.tolist()], dtype=np.int64)
        ys_2d = np.ascontiguousarray(ys_arr, dtype=np.float64)
        res = _metrics_multiclass_roc_auc(yt_enc, ys_2d, average, multi_class, sw)
        if average is None:
            return np.asarray(res, dtype=np.float64)
        return float(res[0])

    raise ValueError(f"Expected 1D or 2D y_score, got {ys_arr.ndim}D ( >_< )")


def log_loss(
    y_true: Any,
    y_pred: Any,
    normalize: bool = True,
    sample_weight: Optional[Any] = None,
    labels: Optional[Sequence[Any]] = None,
) -> float:
    """Log loss, aka logistic loss or cross-entropy loss (＾▽＾)."""
    yt_arr = np.asarray(y_true)
    yp_arr = np.asarray(y_pred, dtype=np.float64)
    sw = _prep_sw(sample_weight)

    if labels is None:
        unique_vals = sorted(set(yt_arr.tolist()))
    else:
        unique_vals = list(labels)

    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    yt_enc = np.array([mapping.get(v, -1) for v in yt_arr.tolist()], dtype=np.int64)

    if yp_arr.ndim == 1:
        p1 = np.ascontiguousarray(yp_arr, dtype=np.float64)
        yp_2d = np.ascontiguousarray(np.column_stack([1.0 - p1, p1]), dtype=np.float64)
    elif yp_arr.ndim == 2 and yp_arr.shape[1] == 1:
        p1 = np.ascontiguousarray(yp_arr[:, 0], dtype=np.float64)
        yp_2d = np.ascontiguousarray(np.column_stack([1.0 - p1, p1]), dtype=np.float64)
    elif yp_arr.ndim == 2:
        yp_2d = np.ascontiguousarray(yp_arr, dtype=np.float64)
    else:
        raise ValueError(f"Expected 1D or 2D y_pred, got {yp_arr.ndim}D ( >_< )")

    return float(_metrics_log_loss(yt_enc, yp_2d, normalize, sw))


def classification_report(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    target_names: Optional[Sequence[str]] = None,
    sample_weight: Optional[Any] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[int, float, str] = 0.0,
) -> Union[str, Dict[str, Any]]:
    """Build a text report or dict showing the main classification metrics (o´∀｀o)."""
    if labels is None:
        yt_arr = np.asarray(y_true)
        yp_arr = np.asarray(y_pred)
        resolved_labels = sorted(set(yt_arr.tolist()) | set(yp_arr.tolist()))
    else:
        resolved_labels = list(labels)

    if target_names is not None:
        if len(target_names) != len(resolved_labels):
            raise ValueError(
                f"Number of classes ({len(resolved_labels)}) does not match target_names ({len(target_names)}) ( >_< )"
            )
        names = [str(n) for n in target_names]
    else:
        names = [str(lbl) for lbl in resolved_labels]

    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=resolved_labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    acc = accuracy_score(y_true, y_pred, normalize=True, sample_weight=sample_weight)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=resolved_labels,
        average="macro",
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=1.0,
        labels=resolved_labels,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    total_support = float(np.sum(s)) if sample_weight is not None else int(np.sum(s))

    if output_dict:
        report: Dict[str, Any] = {}
        for idx, name in enumerate(names):
            report[name] = {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1-score": float(f1[idx]),
                "support": float(s[idx]) if sample_weight is not None else int(s[idx]),
            }
        report["accuracy"] = float(acc)
        report["macro avg"] = {
            "precision": float(p_mac),
            "recall": float(r_mac),
            "f1-score": float(f_mac),
            "support": total_support,
        }
        report["weighted avg"] = {
            "precision": float(p_wt),
            "recall": float(r_wt),
            "f1-score": float(f_wt),
            "support": total_support,
        }
        return report

    headers = ["precision", "recall", "f1-score", "support"]
    longest_last_line_heading = "weighted avg"
    name_width = max([len(n) for n in names] + [len(longest_last_line_heading)])
    width = max(name_width, digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    lines = [head_fmt.format("", *headers, width=width), ""]

    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for idx, name in enumerate(names):
        sup_val = f"{s[idx]:.2f}" if sample_weight is not None else str(int(s[idx]))
        lines.append(
            row_fmt.format(
                name, p[idx], r[idx], f1[idx], sup_val, width=width, digits=digits
            ).rstrip("\n")
        )
    lines.append("")

    sup_str = f"{total_support:.2f}" if sample_weight is not None else str(total_support)
    acc_fmt = "{:>{width}s} " + " {:>9}" * 2 + " {:>9.{digits}f} {:>9}"
    lines.append(
        acc_fmt.format("accuracy", "", "", acc, sup_str, width=width, digits=digits)
    )
    lines.append(
        row_fmt.format(
            "macro avg", p_mac, r_mac, f_mac, sup_str, width=width, digits=digits
        ).rstrip("\n")
    )
    lines.append(
        row_fmt.format(
            "weighted avg", p_wt, r_wt, f_wt, sup_str, width=width, digits=digits
        ).rstrip("\n")
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Clustering Metrics
# ---------------------------------------------------------------------------


def silhouette_score(
    X: Any,
    labels: Any,
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: Optional[Any] = None,
    device: str = "auto",
) -> float:
    """
    Compute the mean Silhouette Coefficient of all samples (＾▽＾).
    Uses Hardware Backend N3 (`pairwise_l2_sq`) for Euclidean distances.
    """
    X_arr = _prep_x(X)
    lbl_enc, _ = _encode_single_labels(labels)
    if X_arr.shape[0] != lbl_enc.shape[0]:
        raise ValueError(
            f"X ({X_arr.shape[0]}) and labels ({lbl_enc.shape[0]}) must have the same length ( >_< )"
        )

    if sample_size is not None and sample_size < X_arr.shape[0]:
        if isinstance(random_state, np.random.RandomState):
            rng = random_state
        elif isinstance(random_state, np.random.Generator):
            idx = random_state.choice(X_arr.shape[0], size=sample_size, replace=False)
            X_arr = np.ascontiguousarray(X_arr[idx], dtype=np.float64)
            lbl_enc = np.ascontiguousarray(lbl_enc[idx], dtype=np.int64)
            dev = _resolve_device(device)
            return float(_metrics_silhouette_score(X_arr, lbl_enc, metric, dev))
        else:
            rng = np.random.RandomState(random_state)
        idx = rng.choice(X_arr.shape[0], size=sample_size, replace=False)
        X_arr = np.ascontiguousarray(X_arr[idx], dtype=np.float64)
        lbl_enc = np.ascontiguousarray(lbl_enc[idx], dtype=np.int64)

    dev = _resolve_device(device)
    return float(_metrics_silhouette_score(X_arr, lbl_enc, metric, dev))


def davies_bouldin_score(X: Any, labels: Any) -> float:
    """Compute the Davies-Bouldin score (lower is better) (＾▽＾)."""
    X_arr = _prep_x(X)
    lbl_enc, _ = _encode_single_labels(labels)
    return float(_metrics_davies_bouldin_score(X_arr, lbl_enc))


def calinski_harabasz_score(X: Any, labels: Any) -> float:
    """Compute the Calinski and Harabasz score (Variance Ratio Criterion) (*≧ω≦*)."""
    X_arr = _prep_x(X)
    lbl_enc, _ = _encode_single_labels(labels)
    return float(_metrics_calinski_harabasz_score(X_arr, lbl_enc))


def adjusted_rand_score(labels_true: Any, labels_pred: Any) -> float:
    """Rand index adjusted for chance (＾▽＾)."""
    lt_enc, _ = _encode_single_labels(labels_true)
    lp_enc, _ = _encode_single_labels(labels_pred)
    return float(_metrics_adjusted_rand_score(lt_enc, lp_enc))


def normalized_mutual_info_score(
    labels_true: Any,
    labels_pred: Any,
    average_method: str = "arithmetic",
) -> float:
    """Normalized Mutual Information between two clusterings (*≧ω≦*)."""
    lt_enc, _ = _encode_single_labels(labels_true)
    lp_enc, _ = _encode_single_labels(labels_pred)
    return float(_metrics_normalized_mutual_info_score(lt_enc, lp_enc, average_method))


__all__ = [
    "accuracy_score",
    "adjusted_rand_score",
    "calinski_harabasz_score",
    "classification_report",
    "confusion_matrix",
    "davies_bouldin_score",
    "explained_variance_score",
    "f1_score",
    "log_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "normalized_mutual_info_score",
    "precision_recall_fscore_support",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "root_mean_squared_error",
    "silhouette_score",
]
