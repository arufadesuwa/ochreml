// OchreML Metrics Module (＾▽＾)
// Regression, Classification, and Clustering metrics with Hardware Backend N3 support (*≧ω≦*).

use crate::backend::get_backend;
use crate::device::parse_device;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, buffer_to_typed_1d,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Validation & Shared Helpers
// ---------------------------------------------------------------------------

fn validate_weights(n_samples: usize, sample_weight: Option<&[f64]>) -> OchreResult<f64> {
    match sample_weight {
        None => Ok(n_samples as f64),
        Some(w) => {
            if w.len() != n_samples {
                return Err(OchreError::WeightLengthMismatch {
                    weights: w.len(),
                    samples: n_samples,
                });
            }
            let mut sum = 0.0;
            for &val in w {
                if !val.is_finite() || val < 0.0 {
                    return Err(OchreError::InvalidWeights);
                }
                sum += val;
            }
            if sum <= 0.0 {
                return Err(OchreError::NonPositiveWeightSum);
            }
            Ok(sum)
        }
    }
}

fn validate_regression_inputs(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
) -> OchreResult<f64> {
    if y_true.rows == 0 || y_true.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if y_true.rows != y_pred.rows || y_true.cols != y_pred.cols {
        return Err(OchreError::DimensionMismatch {
            expected_rows: y_true.rows,
            expected_cols: y_true.cols,
            actual_rows: y_pred.rows,
            actual_cols: y_pred.cols,
        });
    }
    validate_weights(y_true.rows, sample_weight)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MultioutputMode {
    UniformAverage,
    RawValues,
    VarianceWeighted,
}

fn parse_multioutput(
    multioutput: &str,
    allow_variance_weighted: bool,
) -> OchreResult<MultioutputMode> {
    match multioutput {
        "uniform_average" => Ok(MultioutputMode::UniformAverage),
        "raw_values" => Ok(MultioutputMode::RawValues),
        "variance_weighted" if allow_variance_weighted => Ok(MultioutputMode::VarianceWeighted),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported multioutput='{other}' ( >_< )"
        ))),
    }
}

fn aggregate_multioutput(
    values: &[f64],
    mode: MultioutputMode,
    col_weights: Option<&[f64]>,
) -> Vec<f64> {
    match mode {
        MultioutputMode::RawValues => values.to_vec(),
        MultioutputMode::UniformAverage => {
            let mean = values.iter().sum::<f64>() / (values.len() as f64);
            vec![mean]
        }
        MultioutputMode::VarianceWeighted => {
            if let Some(cw) = col_weights {
                let total: f64 = cw.iter().sum();
                if total > 0.0 {
                    let weighted: f64 = values.iter().zip(cw.iter()).map(|(v, w)| v * w).sum();
                    return vec![weighted / total];
                }
            }
            let mean = values.iter().sum::<f64>() / (values.len() as f64);
            vec![mean]
        }
    }
}

#[inline]
fn weight_at(sample_weight: Option<&[f64]>, i: usize) -> f64 {
    sample_weight.map_or(1.0, |w| w[i])
}

// ---------------------------------------------------------------------------
// Regression Metrics
// ---------------------------------------------------------------------------

pub fn mean_squared_error_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let w_sum = validate_regression_inputs(y_true, y_pred, sample_weight)?;
    let mode = parse_multioutput(multioutput, false)?;
    let n = y_true.rows;
    let k = y_true.cols;
    let mut out = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            let d = row_t[j] - row_p[j];
            out[j] += w * d * d;
        }
    }
    for v in &mut out {
        *v /= w_sum;
    }
    Ok(aggregate_multioutput(&out, mode, None))
}

pub fn root_mean_squared_error_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let mode = parse_multioutput(multioutput, false)?;
    let mut mse = mean_squared_error_core(y_true, y_pred, sample_weight, "raw_values")?;
    for v in &mut mse {
        *v = v.sqrt();
    }
    Ok(aggregate_multioutput(&mse, mode, None))
}

pub fn mean_absolute_error_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let w_sum = validate_regression_inputs(y_true, y_pred, sample_weight)?;
    let mode = parse_multioutput(multioutput, false)?;
    let n = y_true.rows;
    let k = y_true.cols;
    let mut out = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            out[j] += w * (row_t[j] - row_p[j]).abs();
        }
    }
    for v in &mut out {
        *v /= w_sum;
    }
    Ok(aggregate_multioutput(&out, mode, None))
}

pub fn mean_absolute_percentage_error_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let w_sum = validate_regression_inputs(y_true, y_pred, sample_weight)?;
    let mode = parse_multioutput(multioutput, false)?;
    let n = y_true.rows;
    let k = y_true.cols;
    let eps = f64::EPSILON;
    let mut out = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            let denom = row_t[j].abs().max(eps);
            out[j] += w * ((row_t[j] - row_p[j]).abs() / denom);
        }
    }
    for v in &mut out {
        *v /= w_sum;
    }
    Ok(aggregate_multioutput(&out, mode, None))
}

pub fn r2_score_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let w_sum = validate_regression_inputs(y_true, y_pred, sample_weight)?;
    let mode = parse_multioutput(multioutput, true)?;
    let n = y_true.rows;
    let k = y_true.cols;

    if n < 2 {
        return Ok(aggregate_multioutput(&vec![f64::NAN; k], mode, None));
    }

    let mut y_mean = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        for j in 0..k {
            y_mean[j] += w * row_t[j];
        }
    }
    for m in &mut y_mean {
        *m /= w_sum;
    }

    let mut ss_res = vec![0.0; k];
    let mut ss_tot = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            let res = row_t[j] - row_p[j];
            let tot = row_t[j] - y_mean[j];
            ss_res[j] += w * res * res;
            ss_tot[j] += w * tot * tot;
        }
    }

    let mut scores = vec![0.0; k];
    for j in 0..k {
        if ss_tot[j] == 0.0 {
            scores[j] = if ss_res[j] == 0.0 { 1.0 } else { 0.0 };
        } else {
            scores[j] = 1.0 - (ss_res[j] / ss_tot[j]);
        }
    }
    Ok(aggregate_multioutput(&scores, mode, Some(&ss_tot)))
}

pub fn explained_variance_score_core(
    y_true: &Matrix,
    y_pred: &Matrix,
    sample_weight: Option<&[f64]>,
    multioutput: &str,
) -> OchreResult<Vec<f64>> {
    let w_sum = validate_regression_inputs(y_true, y_pred, sample_weight)?;
    let mode = parse_multioutput(multioutput, true)?;
    let n = y_true.rows;
    let k = y_true.cols;

    let mut y_mean = vec![0.0; k];
    let mut e_mean = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            y_mean[j] += w * row_t[j];
            e_mean[j] += w * (row_t[j] - row_p[j]);
        }
    }
    for j in 0..k {
        y_mean[j] /= w_sum;
        e_mean[j] /= w_sum;
    }

    let mut var_y = vec![0.0; k];
    let mut var_e = vec![0.0; k];
    for i in 0..n {
        let w = weight_at(sample_weight, i);
        let row_t = &y_true.data[i * k..(i + 1) * k];
        let row_p = &y_pred.data[i * k..(i + 1) * k];
        for j in 0..k {
            let dy = row_t[j] - y_mean[j];
            let de = (row_t[j] - row_p[j]) - e_mean[j];
            var_y[j] += w * dy * dy;
            var_e[j] += w * de * de;
        }
    }

    let mut scores = vec![0.0; k];
    for j in 0..k {
        if var_y[j] == 0.0 {
            scores[j] = if var_e[j] == 0.0 { 1.0 } else { 0.0 };
        } else {
            scores[j] = 1.0 - (var_e[j] / var_y[j]);
        }
    }
    Ok(aggregate_multioutput(&scores, mode, Some(&var_y)))
}

// ---------------------------------------------------------------------------
// Classification Metrics
// ---------------------------------------------------------------------------

pub fn accuracy_score_core(
    y_true: &[i64],
    y_pred: &[i64],
    normalize: bool,
    sample_weight: Option<&[f64]>,
) -> OchreResult<f64> {
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_pred.len(),
        });
    }
    let w_sum = validate_weights(y_true.len(), sample_weight)?;
    let mut correct = 0.0;
    for i in 0..y_true.len() {
        if y_true[i] == y_pred[i] {
            correct += weight_at(sample_weight, i);
        }
    }
    if normalize {
        Ok(correct / w_sum)
    } else {
        Ok(correct)
    }
}

pub fn confusion_matrix_core(
    y_true: &[i64],
    y_pred: &[i64],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
    normalize: Option<&str>,
) -> OchreResult<Vec<f64>> {
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_pred.len(),
        });
    }
    validate_weights(y_true.len(), sample_weight)?;
    if n_labels == 0 {
        return Err(OchreError::InvalidParameter(
            "n_labels must be at least 1 ( >_< )".to_string(),
        ));
    }

    let mut cm = vec![0.0; n_labels * n_labels];
    for i in 0..y_true.len() {
        let t = y_true[i];
        let p = y_pred[i];
        if t >= 0 && p >= 0 && (t as usize) < n_labels && (p as usize) < n_labels {
            cm[(t as usize) * n_labels + (p as usize)] += weight_at(sample_weight, i);
        }
    }

    match normalize {
        None => {}
        Some("true") => {
            for r in 0..n_labels {
                let row_sum: f64 = cm[r * n_labels..(r + 1) * n_labels].iter().sum();
                if row_sum > 0.0 {
                    for c in 0..n_labels {
                        cm[r * n_labels + c] /= row_sum;
                    }
                }
            }
        }
        Some("pred") => {
            for c in 0..n_labels {
                let mut col_sum = 0.0;
                for r in 0..n_labels {
                    col_sum += cm[r * n_labels + c];
                }
                if col_sum > 0.0 {
                    for r in 0..n_labels {
                        cm[r * n_labels + c] /= col_sum;
                    }
                }
            }
        }
        Some("all") => {
            let total: f64 = cm.iter().sum();
            if total > 0.0 {
                for v in &mut cm {
                    *v /= total;
                }
            }
        }
        Some(other) => {
            return Err(OchreError::InvalidParameter(format!(
                "normalize must be one of ('true', 'pred', 'all') or None, got '{other}' ( >_< )"
            )));
        }
    }
    Ok(cm)
}

#[inline]
fn prf_divide(num: f64, den: f64, zero_division: f64) -> f64 {
    if den > 0.0 { num / den } else { zero_division }
}

#[inline]
fn compute_f_beta(tp: f64, fp: f64, fn_: f64, beta: f64, zero_division: f64) -> f64 {
    if beta == 0.0 {
        return prf_divide(tp, tp + fp, zero_division);
    }
    if beta.is_infinite() {
        return prf_divide(tp, tp + fn_, zero_division);
    }
    let beta2 = beta * beta;
    let num = (1.0 + beta2) * tp;
    let den = (1.0 + beta2) * tp + beta2 * fn_ + fp;
    prf_divide(num, den, zero_division)
}

pub struct PrfResult {
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub fscore: Vec<f64>,
    pub support: Vec<f64>,
}

/// Core precision/recall/fscore/support on 0..n_labels encoded y_true/y_pred (out-of-range < 0 ignored).
#[allow(clippy::too_many_arguments)]
pub fn prf_core(
    y_true: &[i64],
    y_pred: &[i64],
    n_labels: usize,
    beta: f64,
    pos_label_idx: usize,
    average: Option<&str>,
    sample_weight: Option<&[f64]>,
    zero_division: f64,
) -> OchreResult<PrfResult> {
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_pred.len(),
        });
    }
    if beta < 0.0 || beta.is_nan() {
        return Err(OchreError::InvalidParameter(format!(
            "beta must be >= 0, got {beta} ( >_< )"
        )));
    }
    validate_weights(y_true.len(), sample_weight)?;
    if n_labels == 0 {
        return Err(OchreError::InvalidParameter(
            "labels cannot be empty ( >_< )".to_string(),
        ));
    }

    let mut tp = vec![0.0; n_labels];
    let mut fp = vec![0.0; n_labels];
    let mut fn_ = vec![0.0; n_labels];
    let mut support = vec![0.0; n_labels];

    for i in 0..y_true.len() {
        let w = weight_at(sample_weight, i);
        let t = y_true[i];
        let p = y_pred[i];
        let t_valid = t >= 0 && (t as usize) < n_labels;
        let p_valid = p >= 0 && (p as usize) < n_labels;

        if t_valid {
            support[t as usize] += w;
        }
        if t_valid && p_valid {
            let tu = t as usize;
            let pu = p as usize;
            if tu == pu {
                tp[tu] += w;
            } else {
                fn_[tu] += w;
                fp[pu] += w;
            }
        } else if t_valid {
            fn_[t as usize] += w;
        } else if p_valid {
            fp[p as usize] += w;
        }
    }

    match average {
        None => {
            let mut precision = Vec::with_capacity(n_labels);
            let mut recall = Vec::with_capacity(n_labels);
            let mut fscore = Vec::with_capacity(n_labels);
            for c in 0..n_labels {
                precision.push(prf_divide(tp[c], tp[c] + fp[c], zero_division));
                recall.push(prf_divide(tp[c], tp[c] + fn_[c], zero_division));
                fscore.push(compute_f_beta(tp[c], fp[c], fn_[c], beta, zero_division));
            }
            Ok(PrfResult {
                precision,
                recall,
                fscore,
                support,
            })
        }
        Some("binary") => {
            if pos_label_idx >= n_labels {
                return Err(OchreError::InvalidParameter(
                    "pos_label is not present in labels ( >_< )".to_string(),
                ));
            }
            let c = pos_label_idx;
            let p = prf_divide(tp[c], tp[c] + fp[c], zero_division);
            let r = prf_divide(tp[c], tp[c] + fn_[c], zero_division);
            let f = compute_f_beta(tp[c], fp[c], fn_[c], beta, zero_division);
            Ok(PrfResult {
                precision: vec![p],
                recall: vec![r],
                fscore: vec![f],
                support: vec![support[c]],
            })
        }
        Some("micro") => {
            let tp_sum: f64 = tp.iter().sum();
            let fp_sum: f64 = fp.iter().sum();
            let fn_sum: f64 = fn_.iter().sum();
            let sup_sum: f64 = support.iter().sum();
            let p = prf_divide(tp_sum, tp_sum + fp_sum, zero_division);
            let r = prf_divide(tp_sum, tp_sum + fn_sum, zero_division);
            let f = compute_f_beta(tp_sum, fp_sum, fn_sum, beta, zero_division);
            Ok(PrfResult {
                precision: vec![p],
                recall: vec![r],
                fscore: vec![f],
                support: vec![sup_sum],
            })
        }
        Some("macro") => {
            let k = n_labels as f64;
            let mut p_sum = 0.0;
            let mut r_sum = 0.0;
            let mut f_sum = 0.0;
            for c in 0..n_labels {
                p_sum += prf_divide(tp[c], tp[c] + fp[c], zero_division);
                r_sum += prf_divide(tp[c], tp[c] + fn_[c], zero_division);
                f_sum += compute_f_beta(tp[c], fp[c], fn_[c], beta, zero_division);
            }
            Ok(PrfResult {
                precision: vec![p_sum / k],
                recall: vec![r_sum / k],
                fscore: vec![f_sum / k],
                support: vec![support.iter().sum()],
            })
        }
        Some("weighted") => {
            let sup_sum: f64 = support.iter().sum();
            if sup_sum <= 0.0 {
                return Ok(PrfResult {
                    precision: vec![0.0],
                    recall: vec![0.0],
                    fscore: vec![0.0],
                    support: vec![0.0],
                });
            }
            let mut p_sum = 0.0;
            let mut r_sum = 0.0;
            let mut f_sum = 0.0;
            for c in 0..n_labels {
                let w = support[c];
                if w > 0.0 {
                    p_sum += w * prf_divide(tp[c], tp[c] + fp[c], zero_division);
                    r_sum += w * prf_divide(tp[c], tp[c] + fn_[c], zero_division);
                    f_sum += w * compute_f_beta(tp[c], fp[c], fn_[c], beta, zero_division);
                }
            }
            Ok(PrfResult {
                precision: vec![p_sum / sup_sum],
                recall: vec![r_sum / sup_sum],
                fscore: vec![f_sum / sup_sum],
                support: vec![sup_sum],
            })
        }
        Some(other) => Err(OchreError::InvalidParameter(format!(
            "average must be None, 'binary', 'micro', 'macro', or 'weighted', got '{other}' ( >_< )"
        ))),
    }
}

/// Exact tie-aware trapezoidal binary ROC AUC for binary labels (0/1).
pub fn binary_roc_auc_core(
    y_true: &[i64],
    y_score: &[f64],
    sample_weight: Option<&[f64]>,
) -> OchreResult<f64> {
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if y_true.len() != y_score.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_score.len(),
        });
    }
    validate_weights(y_true.len(), sample_weight)?;

    let mut order: Vec<usize> = (0..y_true.len()).collect();
    for &idx in &order {
        if !y_score[idx].is_finite() {
            return Err(OchreError::InvalidParameter(
                "y_score contains NaN or infinite values ( >_< )".to_string(),
            ));
        }
    }
    order.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut tp_total = 0.0;
    let mut fp_total = 0.0;
    for (i, &t) in y_true.iter().enumerate() {
        let w = weight_at(sample_weight, i);
        if t == 1 {
            tp_total += w;
        } else {
            fp_total += w;
        }
    }
    if tp_total <= 0.0 || fp_total <= 0.0 {
        return Err(OchreError::InvalidParameter(
            "Only one class present in y_true. ROC AUC score is not defined in that case (´；ω；`)"
                .to_string(),
        ));
    }

    let mut auc = 0.0;
    let mut tp_prev = 0.0;
    let mut i = 0;
    while i < order.len() {
        let score = y_score[order[i]];
        let mut tp_group = 0.0;
        let mut fp_group = 0.0;
        while i < order.len() && y_score[order[i]] == score {
            let idx = order[i];
            let w = weight_at(sample_weight, idx);
            if y_true[idx] == 1 {
                tp_group += w;
            } else {
                fp_group += w;
            }
            i += 1;
        }
        let tp_new = tp_prev + tp_group;
        auc += 0.5 * (tp_prev + tp_new) * fp_group;
        tp_prev = tp_new;
    }

    Ok(auc / (tp_total * fp_total))
}

/// Multiclass ROC AUC ("ovr" or "ovo") for 0..n_classes encoded y_true and (n_samples, n_classes) y_score.
pub fn multiclass_roc_auc_core(
    y_true: &[i64],
    y_score: &Matrix,
    average: Option<&str>,
    multi_class: &str,
    sample_weight: Option<&[f64]>,
) -> OchreResult<Vec<f64>> {
    let n = y_true.len();
    if n == 0 {
        return Err(OchreError::EmptyData);
    }
    if y_score.rows != n {
        return Err(OchreError::LengthMismatch {
            features: n,
            targets: y_score.rows,
        });
    }
    let k = y_score.cols;
    if k < 2 {
        return Err(OchreError::InsufficientClasses(k));
    }
    validate_weights(n, sample_weight)?;

    match multi_class {
        "ovr" => {
            if average == Some("micro") {
                let mut flat_y = Vec::with_capacity(n * k);
                let mut flat_s = Vec::with_capacity(n * k);
                let mut flat_w = sample_weight.map(|_| Vec::with_capacity(n * k));
                for (i, &t) in y_true.iter().enumerate() {
                    let w = weight_at(sample_weight, i);
                    for c in 0..k {
                        flat_y.push(if t == c as i64 { 1 } else { 0 });
                        flat_s.push(y_score.data[i * k + c]);
                        if let Some(ref mut fw) = flat_w {
                            fw.push(w);
                        }
                    }
                }
                let score = binary_roc_auc_core(&flat_y, &flat_s, flat_w.as_deref())?;
                return Ok(vec![score]);
            }

            let mut per_class = Vec::with_capacity(k);
            let mut class_weights = vec![0.0; k];
            for (i, &t) in y_true.iter().enumerate() {
                if t >= 0 && (t as usize) < k {
                    class_weights[t as usize] += weight_at(sample_weight, i);
                }
            }
            for c in 0..k {
                let bin_y: Vec<i64> = y_true
                    .iter()
                    .map(|&t| if t == c as i64 { 1 } else { 0 })
                    .collect();
                let col_s: Vec<f64> = (0..n).map(|i| y_score.data[i * k + c]).collect();
                per_class.push(binary_roc_auc_core(&bin_y, &col_s, sample_weight)?);
            }

            match average {
                None => Ok(per_class),
                Some("macro") => Ok(vec![per_class.iter().sum::<f64>() / (k as f64)]),
                Some("weighted") => {
                    let tot: f64 = class_weights.iter().sum();
                    if tot <= 0.0 {
                        return Err(OchreError::NonPositiveWeightSum);
                    }
                    let wsum: f64 = per_class
                        .iter()
                        .zip(class_weights.iter())
                        .map(|(a, w)| a * w)
                        .sum();
                    Ok(vec![wsum / tot])
                }
                Some(other) => Err(OchreError::InvalidParameter(format!(
                    "average='{other}' is not supported for multi_class='ovr' ( >_< )"
                ))),
            }
        }
        "ovo" => {
            if average.is_none() || average == Some("micro") {
                return Err(OchreError::InvalidParameter(
                    "average must be 'macro' or 'weighted' for multi_class='ovo' ( >_< )"
                        .to_string(),
                ));
            }
            let mut class_weights = vec![0.0; k];
            for (i, &t) in y_true.iter().enumerate() {
                if t >= 0 && (t as usize) < k {
                    class_weights[t as usize] += weight_at(sample_weight, i);
                }
            }

            let mut pair_aucs = Vec::new();
            let mut pair_weights = Vec::new();
            for a in 0..k {
                for b in (a + 1)..k {
                    let mut sub_y_a = Vec::new();
                    let mut sub_s_a = Vec::new();
                    let mut sub_s_b = Vec::new();
                    let mut sub_w = sample_weight.map(|_| Vec::new());

                    for (i, &t) in y_true.iter().enumerate() {
                        if t == a as i64 || t == b as i64 {
                            sub_y_a.push(if t == a as i64 { 1 } else { 0 });
                            sub_s_a.push(y_score.data[i * k + a]);
                            sub_s_b.push(y_score.data[i * k + b]);
                            if let Some(ref mut sw) = sub_w {
                                sw.push(weight_at(sample_weight, i));
                            }
                        }
                    }
                    let auc_ab = binary_roc_auc_core(&sub_y_a, &sub_s_a, sub_w.as_deref())?;
                    let sub_y_b: Vec<i64> = sub_y_a.iter().map(|&v| 1 - v).collect();
                    let auc_ba = binary_roc_auc_core(&sub_y_b, &sub_s_b, sub_w.as_deref())?;
                    pair_aucs.push(0.5 * (auc_ab + auc_ba));
                    pair_weights.push(class_weights[a] + class_weights[b]);
                }
            }

            match average {
                Some("macro") => Ok(vec![
                    pair_aucs.iter().sum::<f64>() / (pair_aucs.len() as f64),
                ]),
                Some("weighted") => {
                    let tot: f64 = pair_weights.iter().sum();
                    if tot <= 0.0 {
                        return Err(OchreError::NonPositiveWeightSum);
                    }
                    let wsum: f64 = pair_aucs
                        .iter()
                        .zip(pair_weights.iter())
                        .map(|(a, w)| a * w)
                        .sum();
                    Ok(vec![wsum / tot])
                }
                None => Err(OchreError::InvalidParameter(
                    "average=None is not supported for multi_class='ovo' ( >_< )".to_string(),
                )),
                Some(other) => Err(OchreError::InvalidParameter(format!(
                    "average='{other}' is not supported for multi_class='ovo' ( >_< )"
                ))),
            }
        }
        other => Err(OchreError::InvalidParameter(format!(
            "multi_class must be 'raise', 'ovr', or 'ovo', got '{other}' ( >_< )"
        ))),
    }
}

pub fn log_loss_core(
    y_true: &[i64],
    y_pred: &Matrix,
    normalize: bool,
    sample_weight: Option<&[f64]>,
) -> OchreResult<f64> {
    let n = y_true.len();
    if n == 0 {
        return Err(OchreError::EmptyData);
    }
    if y_pred.rows != n {
        return Err(OchreError::LengthMismatch {
            features: n,
            targets: y_pred.rows,
        });
    }
    let k = y_pred.cols;
    if k < 2 {
        return Err(OchreError::InvalidParameter(
            "y_pred must have at least 2 probability columns for log_loss ( >_< )".to_string(),
        ));
    }
    let w_sum = validate_weights(n, sample_weight)?;
    let eps = 1e-15_f64;

    let mut total_loss = 0.0;
    for (i, &t) in y_true.iter().enumerate() {
        if t < 0 || (t as usize) >= k {
            return Err(OchreError::InvalidParameter(format!(
                "y_true[{i}]={t} is out of bounds for {k} classes ( >_< )"
            )));
        }
        let row = &y_pred.data[i * k..(i + 1) * k];
        let mut row_sum = 0.0;
        for &p in row {
            if !p.is_finite() || p < 0.0 {
                return Err(OchreError::InvalidParameter(
                    "y_pred probabilities must be finite and non-negative ( >_< )".to_string(),
                ));
            }
            row_sum += p.clamp(eps, 1.0 - eps);
        }
        if row_sum <= 0.0 {
            return Err(OchreError::InvalidParameter(
                "y_pred row sum must be positive ( >_< )".to_string(),
            ));
        }
        let p_t = row[t as usize].clamp(eps, 1.0 - eps) / row_sum;
        total_loss -= weight_at(sample_weight, i) * p_t.ln();
    }

    if normalize {
        Ok(total_loss / w_sum)
    } else {
        Ok(total_loss)
    }
}

// ---------------------------------------------------------------------------
// Clustering Metrics
// ---------------------------------------------------------------------------

fn encode_cluster_labels(labels: &[i64]) -> OchreResult<(Vec<usize>, Vec<usize>)> {
    if labels.is_empty() {
        return Err(OchreError::EmptyData);
    }
    let mut map: HashMap<i64, usize> = HashMap::new();
    let mut encoded = Vec::with_capacity(labels.len());
    let mut counts = Vec::new();
    for &lbl in labels {
        let next_id = counts.len();
        let id = *map.entry(lbl).or_insert_with(|| {
            counts.push(0usize);
            next_id
        });
        counts[id] += 1;
        encoded.push(id);
    }
    let k = counts.len();
    if k < 2 || k >= labels.len() {
        return Err(OchreError::InvalidParameter(format!(
            "Number of labels is {k}. Valid values are 2 to n_samples - 1 (inclusive) ( >_< )"
        )));
    }
    Ok((encoded, counts))
}

fn cosine_distance_row(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        if na == 0.0 && nb == 0.0 { 0.0 } else { 1.0 }
    } else {
        (1.0 - dot / denom).clamp(0.0, 2.0)
    }
}

fn manhattan_distance_row(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// Silhouette score using Hardware Backend N3 (`pairwise_l2_sq`) for Euclidean metric
/// (chunked to bound memory) and CPU Rayon for Manhattan/Cosine (*≧ω≦*).
pub fn silhouette_score_core(
    x: &Matrix,
    labels: &[i64],
    metric: &str,
    device_str: Option<&str>,
) -> OchreResult<f64> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != labels.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: labels.len(),
        });
    }
    let (enc, counts) = encode_cluster_labels(labels)?;
    let n = x.rows;
    let d = x.cols;
    let k = counts.len();

    let chunk_size = 512usize;
    let mut total_s = 0.0;

    match metric {
        "euclidean" => {
            let dev = parse_device(device_str)?;
            let backend = get_backend(&dev);
            let mut start = 0;
            while start < n {
                let end = (start + chunk_size).min(n);
                let c_rows = end - start;
                let chunk = Matrix::from_vec(c_rows, d, x.data[start * d..end * d].to_vec())?;
                let dist_sq = backend.pairwise_l2_sq(&chunk, x)?;
                let chunk_s: f64 = (0..c_rows)
                    .into_par_iter()
                    .map(|local_i| {
                        let global_i = start + local_i;
                        let ci = enc[global_i];
                        if counts[ci] <= 1 {
                            return 0.0;
                        }
                        let row = &dist_sq.data[local_i * n..(local_i + 1) * n];
                        let mut cluster_sums = vec![0.0; k];
                        for j in 0..n {
                            let dist = row[j].max(0.0).sqrt();
                            cluster_sums[enc[j]] += dist;
                        }
                        let a_i = cluster_sums[ci] / ((counts[ci] - 1) as f64);
                        let mut b_i = f64::INFINITY;
                        for c in 0..k {
                            if c != ci {
                                let mean_c = cluster_sums[c] / (counts[c] as f64);
                                if mean_c < b_i {
                                    b_i = mean_c;
                                }
                            }
                        }
                        let denom = a_i.max(b_i);
                        if denom == 0.0 {
                            0.0
                        } else {
                            (b_i - a_i) / denom
                        }
                    })
                    .sum();
                total_s += chunk_s;
                start = end;
            }
        }
        "manhattan" | "cosine" => {
            let is_cosine = metric == "cosine";
            let sum_s: f64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let ci = enc[i];
                    if counts[ci] <= 1 {
                        return 0.0;
                    }
                    let xi = &x.data[i * d..(i + 1) * d];
                    let mut cluster_sums = vec![0.0; k];
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let xj = &x.data[j * d..(j + 1) * d];
                        let dist = if is_cosine {
                            cosine_distance_row(xi, xj)
                        } else {
                            manhattan_distance_row(xi, xj)
                        };
                        cluster_sums[enc[j]] += dist;
                    }
                    let a_i = cluster_sums[ci] / ((counts[ci] - 1) as f64);
                    let mut b_i = f64::INFINITY;
                    for c in 0..k {
                        if c != ci {
                            let mean_c = cluster_sums[c] / (counts[c] as f64);
                            if mean_c < b_i {
                                b_i = mean_c;
                            }
                        }
                    }
                    let denom = a_i.max(b_i);
                    if denom == 0.0 {
                        0.0
                    } else {
                        (b_i - a_i) / denom
                    }
                })
                .sum();
            total_s = sum_s;
        }
        other => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported metric '{other}'. Use 'euclidean', 'manhattan', or 'cosine' ( >_< )"
            )));
        }
    }

    Ok(total_s / (n as f64))
}

pub fn davies_bouldin_score_core(x: &Matrix, labels: &[i64]) -> OchreResult<f64> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != labels.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: labels.len(),
        });
    }
    let (enc, counts) = encode_cluster_labels(labels)?;
    let d = x.cols;
    let k = counts.len();

    let mut centroids = vec![0.0; k * d];
    for (i, &c) in enc.iter().enumerate() {
        let row = &x.data[i * d..(i + 1) * d];
        let cent = &mut centroids[c * d..(c + 1) * d];
        for (slot, &rv) in cent.iter_mut().zip(row.iter()) {
            *slot += rv;
        }
    }
    for c in 0..k {
        let inv = 1.0 / (counts[c] as f64);
        for slot in &mut centroids[c * d..(c + 1) * d] {
            *slot *= inv;
        }
    }

    let mut intra_disp = vec![0.0; k];
    for (i, &c) in enc.iter().enumerate() {
        let row = &x.data[i * d..(i + 1) * d];
        let cent = &centroids[c * d..(c + 1) * d];
        let mut sq = 0.0;
        for (&rv, &cv) in row.iter().zip(cent.iter()) {
            let diff = rv - cv;
            sq += diff * diff;
        }
        intra_disp[c] += sq.sqrt();
    }
    for c in 0..k {
        intra_disp[c] /= counts[c] as f64;
    }

    let mut db_sum = 0.0;
    for i in 0..k {
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let mut m_ij_sq = 0.0;
            for f in 0..d {
                let diff = centroids[i * d + f] - centroids[j * d + f];
                m_ij_sq += diff * diff;
            }
            let m_ij = m_ij_sq.sqrt();
            let r_ij = if m_ij == 0.0 {
                0.0
            } else {
                (intra_disp[i] + intra_disp[j]) / m_ij
            };
            if r_ij > max_r {
                max_r = r_ij;
            }
        }
        db_sum += max_r;
    }

    Ok(db_sum / (k as f64))
}

pub fn calinski_harabasz_score_core(x: &Matrix, labels: &[i64]) -> OchreResult<f64> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != labels.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: labels.len(),
        });
    }
    let (enc, counts) = encode_cluster_labels(labels)?;
    let n = x.rows;
    let d = x.cols;
    let k = counts.len();

    let mut mean = vec![0.0; d];
    let mut centroids = vec![0.0; k * d];
    for (i, &c) in enc.iter().enumerate() {
        let row = &x.data[i * d..(i + 1) * d];
        let cent = &mut centroids[c * d..(c + 1) * d];
        for ((m, ct), &rv) in mean.iter_mut().zip(cent.iter_mut()).zip(row.iter()) {
            *m += rv;
            *ct += rv;
        }
    }
    let inv_n = 1.0 / (n as f64);
    for m in &mut mean {
        *m *= inv_n;
    }
    for c in 0..k {
        let inv_c = 1.0 / (counts[c] as f64);
        for slot in &mut centroids[c * d..(c + 1) * d] {
            *slot *= inv_c;
        }
    }

    let mut extra_disp = 0.0;
    for c in 0..k {
        let mut sq = 0.0;
        for (&cv, &mv) in centroids[c * d..(c + 1) * d].iter().zip(mean.iter()) {
            let diff = cv - mv;
            sq += diff * diff;
        }
        extra_disp += (counts[c] as f64) * sq;
    }

    let mut intra_disp = 0.0;
    for (i, &c) in enc.iter().enumerate() {
        let row = &x.data[i * d..(i + 1) * d];
        let cent = &centroids[c * d..(c + 1) * d];
        for (&rv, &cv) in row.iter().zip(cent.iter()) {
            let diff = rv - cv;
            intra_disp += diff * diff;
        }
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

struct Contingency {
    n: usize,
    row_sums: Vec<usize>,
    col_sums: Vec<usize>,
    pairs: Vec<(usize, usize, usize)>,
}

fn build_contingency(labels_true: &[i64], labels_pred: &[i64]) -> OchreResult<Contingency> {
    if labels_true.len() != labels_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: labels_true.len(),
            targets: labels_pred.len(),
        });
    }
    let n = labels_true.len();
    let mut map_t: HashMap<i64, usize> = HashMap::new();
    let mut map_p: HashMap<i64, usize> = HashMap::new();
    let mut row_sums = Vec::new();
    let mut col_sums = Vec::new();
    let mut cell_counts: HashMap<(usize, usize), usize> = HashMap::new();

    for i in 0..n {
        let rt = *map_t.entry(labels_true[i]).or_insert_with(|| {
            row_sums.push(0);
            row_sums.len() - 1
        });
        let cp = *map_p.entry(labels_pred[i]).or_insert_with(|| {
            col_sums.push(0);
            col_sums.len() - 1
        });
        row_sums[rt] += 1;
        col_sums[cp] += 1;
        *cell_counts.entry((rt, cp)).or_insert(0) += 1;
    }

    let pairs = cell_counts
        .into_iter()
        .map(|((r, c), v)| (r, c, v))
        .collect();
    Ok(Contingency {
        n,
        row_sums,
        col_sums,
        pairs,
    })
}

#[inline]
fn comb2(x: usize) -> f64 {
    let xf = x as f64;
    0.5 * xf * (xf - 1.0)
}

pub fn adjusted_rand_score_core(labels_true: &[i64], labels_pred: &[i64]) -> OchreResult<f64> {
    let cont = build_contingency(labels_true, labels_pred)?;
    if cont.n <= 1 || (cont.row_sums.len() == 1 && cont.col_sums.len() == 1) {
        return Ok(1.0);
    }
    let sum_comb: f64 = cont.pairs.iter().map(|&(_, _, v)| comb2(v)).sum();
    let sum_a: f64 = cont.row_sums.iter().map(|&a| comb2(a)).sum();
    let sum_b: f64 = cont.col_sums.iter().map(|&b| comb2(b)).sum();
    let total_comb = comb2(cont.n);
    if total_comb == 0.0 {
        return Ok(1.0);
    }
    let expected = (sum_a * sum_b) / total_comb;
    let max_idx = 0.5 * (sum_a + sum_b);
    let denom = max_idx - expected;
    if denom == 0.0 {
        return Ok(if sum_comb == expected { 1.0 } else { 0.0 });
    }
    Ok((sum_comb - expected) / denom)
}

#[inline]
fn entropy_from_counts(counts: &[usize], n: f64) -> f64 {
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = (c as f64) / n;
            h -= p * p.ln();
        }
    }
    h
}

pub fn normalized_mutual_info_score_core(
    labels_true: &[i64],
    labels_pred: &[i64],
    average_method: &str,
) -> OchreResult<f64> {
    let cont = build_contingency(labels_true, labels_pred)?;
    if cont.n == 0 {
        return Ok(1.0);
    }
    let nf = cont.n as f64;
    let h_u = entropy_from_counts(&cont.row_sums, nf);
    let h_v = entropy_from_counts(&cont.col_sums, nf);

    if h_u == 0.0 && h_v == 0.0 {
        return Ok(1.0);
    }

    let mut mi = 0.0;
    for &(r, c, v) in &cont.pairs {
        if v > 0 {
            let vf = v as f64;
            let a = cont.row_sums[r] as f64;
            let b = cont.col_sums[c] as f64;
            mi += (vf / nf) * ((nf * vf) / (a * b)).ln();
        }
    }
    mi = mi.max(0.0);

    let normalizer = match average_method {
        "min" => h_u.min(h_v),
        "geometric" => (h_u * h_v).sqrt(),
        "arithmetic" => 0.5 * (h_u + h_v),
        "max" => h_u.max(h_v),
        other => {
            return Err(OchreError::InvalidParameter(format!(
                "average_method must be 'min', 'geometric', 'arithmetic', or 'max', got '{other}' ( >_< )"
            )));
        }
    };

    if normalizer == 0.0 {
        return Ok(0.0);
    }
    Ok((mi / normalizer).clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// PyO3 Thin Bindings
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_mean_squared_error(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(mean_squared_error_core(
        &yt,
        &yp,
        sw.as_deref(),
        multioutput,
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_root_mean_squared_error(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(root_mean_squared_error_core(
        &yt,
        &yp,
        sw.as_deref(),
        multioutput,
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_mean_absolute_error(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(mean_absolute_error_core(
        &yt,
        &yp,
        sw.as_deref(),
        multioutput,
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_mean_absolute_percentage_error(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(mean_absolute_percentage_error_core(
        &yt,
        &yp,
        sw.as_deref(),
        multioutput,
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_r2_score(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(r2_score_core(&yt, &yp, sw.as_deref(), multioutput)?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
fn _metrics_explained_variance_score(
    py: Python<'_>,
    y_true: PyBuffer<f64>,
    y_pred: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
    multioutput: &str,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_pooled_matrix(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(explained_variance_score_core(
        &yt,
        &yp,
        sw.as_deref(),
        multioutput,
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, normalize=true, sample_weight=None))]
fn _metrics_accuracy_score(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_pred: PyBuffer<i64>,
    normalize: bool,
    sample_weight: Option<PyBuffer<f64>>,
) -> PyResult<f64> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let yp = buffer_to_typed_1d(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(accuracy_score_core(&yt, &yp, normalize, sw.as_deref())?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels, sample_weight=None, normalize=None))]
fn _metrics_confusion_matrix(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_pred: PyBuffer<i64>,
    n_labels: usize,
    sample_weight: Option<PyBuffer<f64>>,
    normalize: Option<&str>,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let yp = buffer_to_typed_1d(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(confusion_matrix_core(
        &yt,
        &yp,
        n_labels,
        sw.as_deref(),
        normalize,
    )?)
}

type PrfTuple = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y_true, y_pred, n_labels, beta=1.0, pos_label_idx=0, average=None, sample_weight=None, zero_division=0.0))]
fn _metrics_prf_support(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_pred: PyBuffer<i64>,
    n_labels: usize,
    beta: f64,
    pos_label_idx: usize,
    average: Option<&str>,
    sample_weight: Option<PyBuffer<f64>>,
    zero_division: f64,
) -> PyResult<PrfTuple> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let yp = buffer_to_typed_1d(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    let res = prf_core(
        &yt,
        &yp,
        n_labels,
        beta,
        pos_label_idx,
        average,
        sw.as_deref(),
        zero_division,
    )?;
    Ok((res.precision, res.recall, res.fscore, res.support))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_score, sample_weight=None))]
fn _metrics_binary_roc_auc(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_score: PyBuffer<f64>,
    sample_weight: Option<PyBuffer<f64>>,
) -> PyResult<f64> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let ys = buffer_to_f64_1d(py, &y_score)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(binary_roc_auc_core(&yt, &ys, sw.as_deref())?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_score, average=Some("macro"), multi_class="ovr", sample_weight=None))]
fn _metrics_multiclass_roc_auc(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_score: PyBuffer<f64>,
    average: Option<&str>,
    multi_class: &str,
    sample_weight: Option<PyBuffer<f64>>,
) -> PyResult<Vec<f64>> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let ys = buffer_to_pooled_matrix(py, &y_score)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(multiclass_roc_auc_core(
        &yt,
        &ys,
        average,
        multi_class,
        sw.as_deref(),
    )?)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, normalize=true, sample_weight=None))]
fn _metrics_log_loss(
    py: Python<'_>,
    y_true: PyBuffer<i64>,
    y_pred: PyBuffer<f64>,
    normalize: bool,
    sample_weight: Option<PyBuffer<f64>>,
) -> PyResult<f64> {
    let yt = buffer_to_typed_1d(py, &y_true)?;
    let yp = buffer_to_pooled_matrix(py, &y_pred)?;
    let sw = match sample_weight {
        Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
        None => None,
    };
    Ok(log_loss_core(&yt, &yp, normalize, sw.as_deref())?)
}

#[pyfunction]
#[pyo3(signature = (x, labels, metric="euclidean", device="auto"))]
fn _metrics_silhouette_score(
    py: Python<'_>,
    x: PyBuffer<f64>,
    labels: PyBuffer<i64>,
    metric: &str,
    device: &str,
) -> PyResult<f64> {
    let xm = buffer_to_pooled_matrix(py, &x)?;
    let lbls = buffer_to_typed_1d(py, &labels)?;
    Ok(silhouette_score_core(&xm, &lbls, metric, Some(device))?)
}

#[pyfunction]
fn _metrics_davies_bouldin_score(
    py: Python<'_>,
    x: PyBuffer<f64>,
    labels: PyBuffer<i64>,
) -> PyResult<f64> {
    let xm = buffer_to_pooled_matrix(py, &x)?;
    let lbls = buffer_to_typed_1d(py, &labels)?;
    Ok(davies_bouldin_score_core(&xm, &lbls)?)
}

#[pyfunction]
fn _metrics_calinski_harabasz_score(
    py: Python<'_>,
    x: PyBuffer<f64>,
    labels: PyBuffer<i64>,
) -> PyResult<f64> {
    let xm = buffer_to_pooled_matrix(py, &x)?;
    let lbls = buffer_to_typed_1d(py, &labels)?;
    Ok(calinski_harabasz_score_core(&xm, &lbls)?)
}

#[pyfunction]
fn _metrics_adjusted_rand_score(
    py: Python<'_>,
    labels_true: PyBuffer<i64>,
    labels_pred: PyBuffer<i64>,
) -> PyResult<f64> {
    let lt = buffer_to_typed_1d(py, &labels_true)?;
    let lp = buffer_to_typed_1d(py, &labels_pred)?;
    Ok(adjusted_rand_score_core(&lt, &lp)?)
}

#[pyfunction]
#[pyo3(signature = (labels_true, labels_pred, average_method="arithmetic"))]
fn _metrics_normalized_mutual_info_score(
    py: Python<'_>,
    labels_true: PyBuffer<i64>,
    labels_pred: PyBuffer<i64>,
    average_method: &str,
) -> PyResult<f64> {
    let lt = buffer_to_typed_1d(py, &labels_true)?;
    let lp = buffer_to_typed_1d(py, &labels_pred)?;
    Ok(normalized_mutual_info_score_core(&lt, &lp, average_method)?)
}

/// Register all Wave 7 Metrics functions onto the `_ochreml` Python module (＾▽＾).
pub fn register_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_metrics_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_root_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_mean_absolute_error, m)?)?;
    m.add_function(wrap_pyfunction!(
        _metrics_mean_absolute_percentage_error,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(_metrics_r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_explained_variance_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_prf_support, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_binary_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_multiclass_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_log_loss, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_davies_bouldin_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_adjusted_rand_score, m)?)?;
    m.add_function(wrap_pyfunction!(_metrics_normalized_mutual_info_score, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        Matrix::from_vec(rows, cols, data).expect("valid matrix")
    }

    #[test]
    fn test_regression_metrics_1d_and_weighted() {
        let yt = mat(4, 1, vec![3.0, -0.5, 2.0, 7.0]);
        let yp = mat(4, 1, vec![2.5, 0.0, 2.0, 8.0]);

        let mse = mean_squared_error_core(&yt, &yp, None, "uniform_average").unwrap();
        assert!((mse[0] - 0.375).abs() < 1e-12);

        let rmse = root_mean_squared_error_core(&yt, &yp, None, "uniform_average").unwrap();
        assert!((rmse[0] - 0.375_f64.sqrt()).abs() < 1e-12);

        let mae = mean_absolute_error_core(&yt, &yp, None, "uniform_average").unwrap();
        assert!((mae[0] - 0.5).abs() < 1e-12);

        let r2 = r2_score_core(&yt, &yp, None, "uniform_average").unwrap();
        assert!((r2[0] - 0.9486081370449679).abs() < 1e-12);

        let ev = explained_variance_score_core(&yt, &yp, None, "uniform_average").unwrap();
        assert!((ev[0] - 0.9571734475374732).abs() < 1e-12);

        let w = vec![1.0, 2.0, 1.0, 2.0];
        let mse_w = mean_squared_error_core(&yt, &yp, Some(&w), "uniform_average").unwrap();
        // (0.25 + 2*0.25 + 0 + 2*1.0) / 6 = 2.75 / 6
        assert!((mse_w[0] - (2.75 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_regression_multioutput_and_mape() {
        let yt = mat(3, 2, vec![1.0, 10.0, 2.0, 20.0, 4.0, 40.0]);
        let yp = mat(3, 2, vec![1.0, 12.0, 3.0, 20.0, 4.0, 36.0]);

        let raw = mean_squared_error_core(&yt, &yp, None, "raw_values").unwrap();
        assert_eq!(raw.len(), 2);
        assert!((raw[0] - (1.0 / 3.0)).abs() < 1e-12);
        assert!((raw[1] - (20.0 / 3.0)).abs() < 1e-12);

        let mape = mean_absolute_percentage_error_core(&yt, &yp, None, "raw_values").unwrap();
        assert!((mape[0] - (0.5 / 3.0)).abs() < 1e-12);
        assert!((mape[1] - (0.3 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_classification_metrics() {
        let yt = vec![0, 1, 2, 2, 1, 0];
        let yp = vec![0, 2, 2, 2, 1, 1];

        let acc = accuracy_score_core(&yt, &yp, true, None).unwrap();
        assert!((acc - (4.0 / 6.0)).abs() < 1e-12);

        let cm = confusion_matrix_core(&yt, &yp, 3, None, None).unwrap();
        assert_eq!(cm, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0]);

        let prf_macro = prf_core(&yt, &yp, 3, 1.0, 0, Some("macro"), None, 0.0).unwrap();
        // c=0: tp=1,fp=0,fn=1 -> P=1, R=0.5, F=2/3
        // c=1: tp=1,fp=1,fn=1 -> P=0.5, R=0.5, F=0.5
        // c=2: tp=2,fp=1,fn=0 -> P=2/3, R=1.0, F=0.8
        let exp_p = (1.0 + 0.5 + 2.0 / 3.0) / 3.0;
        let exp_r = (0.5 + 0.5 + 1.0) / 3.0;
        let exp_f = (2.0 / 3.0 + 0.5 + 0.8) / 3.0;
        assert!((prf_macro.precision[0] - exp_p).abs() < 1e-12);
        assert!((prf_macro.recall[0] - exp_r).abs() < 1e-12);
        assert!((prf_macro.fscore[0] - exp_f).abs() < 1e-12);
    }

    #[test]
    fn test_roc_auc_ties_and_log_loss() {
        let yt = vec![0, 0, 1, 1];
        let ys = vec![0.1, 0.4, 0.35, 0.8];
        let auc = binary_roc_auc_core(&yt, &ys, None).unwrap();
        assert!((auc - 0.75).abs() < 1e-12);

        // Ties
        let ys_tie = vec![0.5, 0.5, 0.5, 0.5];
        let auc_tie = binary_roc_auc_core(&yt, &ys_tie, None).unwrap();
        assert!((auc_tie - 0.5).abs() < 1e-12);

        // Log loss
        let yt_ll = vec![0, 1];
        let yp_ll = mat(2, 2, vec![0.9, 0.1, 0.2, 0.8]);
        let ll = log_loss_core(&yt_ll, &yp_ll, true, None).unwrap();
        let exp = -0.5 * (0.9_f64.ln() + 0.8_f64.ln());
        assert!((ll - exp).abs() < 1e-12);
    }

    #[test]
    fn test_clustering_metrics() {
        let x = mat(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        );
        let labels = vec![0, 0, 0, 1, 1, 1];

        let sil = silhouette_score_core(&x, &labels, "euclidean", Some("cpu")).unwrap();
        assert!(sil > 0.98);

        let db = davies_bouldin_score_core(&x, &labels).unwrap();
        assert!(db < 0.02);

        let ch = calinski_harabasz_score_core(&x, &labels).unwrap();
        assert!(ch > 10000.0);

        let ari = adjusted_rand_score_core(&labels, &[1, 1, 1, 0, 0, 0]).unwrap();
        assert!((ari - 1.0).abs() < 1e-12);

        let nmi =
            normalized_mutual_info_score_core(&labels, &[1, 1, 1, 0, 0, 0], "arithmetic").unwrap();
        assert!((nmi - 1.0).abs() < 1e-12);
    }
}
