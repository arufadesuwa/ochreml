// OchreML Naive Bayes Module (*≧ω≦*)
// Implements GaussianNB, MultinomialNB, BernoulliNB, and ComplementNB
// accelerated by Backend Primitives (N1 col_reduce_stats, E1 matmul) and Rayon.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64,
    write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::logistic_regression::{extract_sorted_classes, map_labels_to_indices};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::f64::consts::PI;

pub const TAG_GAUSSIAN_NB: u8 = 60;
pub const TAG_MULTINOMIAL_NB: u8 = 61;
pub const TAG_BERNOULLI_NB: u8 = 62;
pub const TAG_COMPLEMENT_NB: u8 = 63;

const MATMUL_THRESHOLD: usize = 8192;
const STATS_PRIMITIVE_THRESHOLD: usize = 1024;

// ---------------------------------------------------------------------------
// Shared validation & numerical helpers
// ---------------------------------------------------------------------------

fn validate_matrix_finite(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_non_negative(x: &Matrix) -> OchreResult<()> {
    if x.data.iter().any(|&v| v < 0.0) {
        return Err(OchreError::InvalidParameter(
            "Negative values in data passed to Naive Bayes (must be non-negative) ( >_< )"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_y_finite(y: &[f64], expected_rows: usize) -> OchreResult<()> {
    if y.len() != expected_rows {
        return Err(OchreError::LengthMismatch {
            features: expected_rows,
            targets: y.len(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Target y contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_sample_weight(sw: Option<&[f64]>, expected_rows: usize) -> OchreResult<()> {
    if let Some(w) = sw {
        if w.len() != expected_rows {
            return Err(OchreError::WeightLengthMismatch {
                weights: w.len(),
                samples: expected_rows,
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
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> OchreResult<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(OchreError::InvalidAlpha(alpha));
    }
    Ok(())
}

fn validate_var_smoothing(vs: f64) -> OchreResult<()> {
    if !vs.is_finite() || vs < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "var_smoothing ({vs}) must be a non-negative finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_priors(priors: &[f64], n_classes: usize) -> OchreResult<Vec<f64>> {
    if priors.len() != n_classes {
        return Err(OchreError::InvalidParameter(format!(
            "Number of priors ({}) must match number of classes ({n_classes}) ( >_< )",
            priors.len()
        )));
    }
    let mut sum = 0.0;
    for &p in priors {
        if !p.is_finite() || p < 0.0 {
            return Err(OchreError::InvalidParameter(
                "priors must be non-negative finite numbers (・`ω´・)".to_string(),
            ));
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-6 {
        return Err(OchreError::InvalidParameter(format!(
            "priors must sum to 1.0, got {sum} ( >_< )"
        )));
    }
    Ok(priors.iter().map(|&p| p / sum).collect())
}

fn validate_and_sort_classes(classes: &[f64]) -> OchreResult<Vec<f64>> {
    if classes.is_empty() {
        return Err(OchreError::InvalidParameter(
            "classes cannot be empty ( >_< )".to_string(),
        ));
    }
    if classes.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "classes contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(extract_sorted_classes(classes))
}

fn compute_class_log_prior(
    class_count: &[f64],
    fit_prior: bool,
    class_prior: Option<&[f64]>,
) -> OchreResult<Vec<f64>> {
    let n_classes = class_count.len();
    if let Some(priors) = class_prior {
        let norm = validate_priors(priors, n_classes)?;
        return Ok(norm
            .into_iter()
            .map(|p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .collect());
    }
    if fit_prior {
        let total: f64 = class_count.iter().sum();
        if total <= 0.0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
        Ok(class_count
            .iter()
            .map(|&c| {
                if c > 0.0 {
                    (c / total).ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect())
    } else {
        let val = -(n_classes as f64).ln();
        Ok(vec![val; n_classes])
    }
}

fn log_proba_from_jll(jll: &Matrix) -> OchreResult<Matrix> {
    let n_samples = jll.rows;
    let n_classes = jll.cols;
    let mut out = vec![0.0; n_samples * n_classes];

    out.par_chunks_mut(n_classes)
        .enumerate()
        .for_each(|(i, dst)| {
            let row = &jll.data[(i * n_classes)..((i + 1) * n_classes)];
            let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !max_val.is_finite() {
                let uniform = -(n_classes as f64).ln();
                dst.fill(uniform);
                return;
            }
            let sum_exp: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
            let log_norm = max_val + sum_exp.ln();
            for (k, &v) in row.iter().enumerate() {
                dst[k] = v - log_norm;
            }
        });

    Matrix::from_vec(n_samples, n_classes, out)
}

fn proba_from_jll(jll: &Matrix) -> OchreResult<Matrix> {
    let mut log_p = log_proba_from_jll(jll)?;
    log_p.data.par_iter_mut().for_each(|v| {
        *v = v.exp();
    });
    Ok(log_p)
}

fn predict_from_jll(jll: &Matrix, classes: &[f64]) -> Vec<f64> {
    let n_samples = jll.rows;
    let n_classes = jll.cols;
    let mut preds = vec![0.0; n_samples];
    preds.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let row = &jll.data[(i * n_classes)..((i + 1) * n_classes)];
        let mut best_k = 0;
        let mut best_v = row[0];
        for (k, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best_k = k;
            }
        }
        *dst = classes[best_k];
    });
    preds
}

fn accuracy_score(preds: &[f64], y: &[f64]) -> OchreResult<f64> {
    validate_y_finite(y, preds.len())?;
    let correct = preds
        .iter()
        .zip(y.iter())
        .filter(|&(p, t)| (p - t).abs() <= 1e-12)
        .count();
    Ok(correct as f64 / y.len() as f64)
}

fn transpose_matrix(mat: &Matrix) -> OchreResult<Matrix> {
    let mut data = vec![0.0; mat.rows * mat.cols];
    for r in 0..mat.rows {
        for c in 0..mat.cols {
            data[c * mat.rows + r] = mat.data[r * mat.cols + c];
        }
    }
    Matrix::from_vec(mat.cols, mat.rows, data)
}

/// Compute `X @ W^T + bias` using `backend.matmul` (E1) on large inputs and Rayon on small ones.
fn linear_jll(
    backend: &Backend<'_>,
    x: &Matrix,
    weights: &Matrix,
    bias: Option<&[f64]>,
) -> OchreResult<Matrix> {
    let n_samples = x.rows;
    let n_features = x.cols;
    let n_classes = weights.rows;

    let mut out = if n_samples * n_features * n_classes >= MATMUL_THRESHOLD {
        let wt = transpose_matrix(weights)?;
        backend.matmul(x, &wt)?
    } else {
        let mut data = vec![0.0; n_samples * n_classes];
        data.par_chunks_mut(n_classes)
            .enumerate()
            .for_each(|(i, dst)| {
                let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
                for (k, slot) in dst.iter_mut().enumerate() {
                    let w_row = &weights.data[(k * n_features)..((k + 1) * n_features)];
                    let mut acc = 0.0;
                    for j in 0..n_features {
                        acc += x_row[j] * w_row[j];
                    }
                    *slot = acc;
                }
            });
        Matrix::from_vec(n_samples, n_classes, data)?
    };

    if let Some(b) = bias {
        out.data.par_chunks_mut(n_classes).for_each(|row| {
            for (dst, &bk) in row.iter_mut().zip(b.iter()) {
                *dst += bk;
            }
        });
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// 1. GaussianNB
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GaussianNB {
    pub priors: Option<Vec<f64>>,
    pub var_smoothing: f64,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    pub class_count_: Option<Vec<f64>>,
    pub class_prior_: Option<Vec<f64>>,
    pub theta_: Option<Matrix>,
    pub var_: Option<Matrix>,
    pub epsilon_: f64,
    pub n_features_in_: Option<usize>,
}

impl GaussianNB {
    pub fn new(
        priors: Option<Vec<f64>>,
        var_smoothing: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_var_smoothing(var_smoothing)?;
        let resolved_dev_str = device.clone().or_else(|| Some("auto".to_string()));
        let parsed_dev = parse_device(resolved_dev_str.as_deref())?;
        Ok(Self {
            priors,
            var_smoothing,
            device_: resolved_dev_str,
            device: parsed_dev,
            classes_: None,
            class_count_: None,
            class_prior_: None,
            theta_: None,
            var_: None,
            epsilon_: 0.0,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.unwrap_or_else(|| "auto".to_string());
        self.device = parse_device(Some(&resolved))?;
        self.device_ = Some(resolved);
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.classes_ = None;
        self.class_count_ = None;
        self.class_prior_ = None;
        self.theta_ = None;
        self.var_ = None;
        self.epsilon_ = 0.0;
        self.n_features_in_ = None;
        self.partial_fit(x, y, None, sample_weight)
    }

    pub fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &[f64],
        classes: Option<&[f64]>,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        validate_sample_weight(sample_weight, x.rows)?;

        let n_samples = x.rows;
        let n_features = x.cols;
        let backend = get_backend(&self.device);

        let is_first_call = self.classes_.is_none();
        if is_first_call {
            let sorted_classes = match classes {
                Some(c) => validate_and_sort_classes(c)?,
                None => extract_sorted_classes(y),
            };
            let n_classes = sorted_classes.len();
            if let Some(ref p) = self.priors {
                validate_priors(p, n_classes)?;
            }

            let global_stats = backend.col_reduce_stats(x, None)?;
            let max_var = global_stats.var.iter().copied().fold(0.0_f64, f64::max);
            self.epsilon_ = self.var_smoothing * max_var;

            self.classes_ = Some(sorted_classes);
            self.class_count_ = Some(vec![0.0; n_classes]);
            self.theta_ = Some(Matrix::zeros(n_classes, n_features));
            self.var_ = Some(Matrix::zeros(n_classes, n_features));
            self.n_features_in_ = Some(n_features);
        } else if let Some(expected_p) = self.n_features_in_
            && n_features != expected_p
        {
            return Err(OchreError::FeatureCountMismatch {
                features: n_features,
                expected: expected_p,
            });
        }

        let classes_ref = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes_ref.len();
        let y_idx = map_labels_to_indices(y, classes_ref)?;

        let class_count = self.class_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let theta = self.theta_.as_mut().ok_or(OchreError::NotFitted)?;
        let var = self.var_.as_mut().ok_or(OchreError::NotFitted)?;

        if !is_first_call {
            for v in var.data.iter_mut() {
                *v = (*v - self.epsilon_).max(0.0);
            }
        }

        // Group row indices per class
        let mut class_rows: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &cls) in y_idx.iter().enumerate() {
            class_rows[cls].push(i);
        }

        for (c, rows_c) in class_rows.iter().enumerate() {
            if rows_c.is_empty() {
                continue;
            }

            let (n_new, mu_new, var_new) = if rows_c.len() >= STATS_PRIMITIVE_THRESHOLD {
                let mut sub_data = Vec::with_capacity(rows_c.len() * n_features);
                let mut sub_w = sample_weight.map(|_| Vec::with_capacity(rows_c.len()));
                for &r in rows_c {
                    sub_data.extend_from_slice(&x.data[(r * n_features)..((r + 1) * n_features)]);
                    if let (Some(sw_all), Some(sw_sub)) = (sample_weight, sub_w.as_mut()) {
                        sw_sub.push(sw_all[r]);
                    }
                }
                let sub_mat = Matrix::from_vec(rows_c.len(), n_features, sub_data)?;
                let stats = backend.col_reduce_stats(&sub_mat, sub_w.as_deref())?;
                (stats.total_weight, stats.mean, stats.var)
            } else {
                let mut n_new = 0.0;
                let mut mu_new = vec![0.0; n_features];
                for &r in rows_c {
                    let w = sample_weight.map_or(1.0, |sw| sw[r]);
                    n_new += w;
                    let row = &x.data[(r * n_features)..((r + 1) * n_features)];
                    for j in 0..n_features {
                        mu_new[j] += w * row[j];
                    }
                }
                if n_new <= 0.0 {
                    continue;
                }
                for m in mu_new.iter_mut() {
                    *m /= n_new;
                }
                let mut var_new = vec![0.0; n_features];
                for &r in rows_c {
                    let w = sample_weight.map_or(1.0, |sw| sw[r]);
                    let row = &x.data[(r * n_features)..((r + 1) * n_features)];
                    for j in 0..n_features {
                        let d = row[j] - mu_new[j];
                        var_new[j] += w * d * d;
                    }
                }
                for v in var_new.iter_mut() {
                    *v /= n_new;
                }
                (n_new, mu_new, var_new)
            };

            if n_new <= 0.0 {
                continue;
            }

            let n_old = class_count[c];
            let theta_row = &mut theta.data[(c * n_features)..((c + 1) * n_features)];
            let var_row = &mut var.data[(c * n_features)..((c + 1) * n_features)];

            if n_old == 0.0 {
                theta_row.copy_from_slice(&mu_new);
                var_row.copy_from_slice(&var_new);
                class_count[c] = n_new;
            } else {
                let n_total = n_old + n_new;
                for j in 0..n_features {
                    let mu_old = theta_row[j];
                    let v_old = var_row[j];
                    let mu_comb = (n_old * mu_old + n_new * mu_new[j]) / n_total;
                    let delta = mu_old - mu_new[j];
                    let ssd = n_old * v_old
                        + n_new * var_new[j]
                        + (n_old * n_new / n_total) * delta * delta;
                    theta_row[j] = mu_comb;
                    var_row[j] = ssd / n_total;
                }
                class_count[c] = n_total;
            }
        }

        for v in var.data.iter_mut() {
            *v += self.epsilon_;
        }

        let total_count: f64 = class_count.iter().sum();
        if total_count <= 0.0 && n_samples > 0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
        let priors_vec = match self.priors {
            Some(ref p) => validate_priors(p, n_classes)?,
            None => class_count.iter().map(|&c| c / total_count).collect(),
        };
        self.class_prior_ = Some(priors_vec);

        Ok(())
    }

    pub fn predict_joint_log_likelihood(&self, x: &Matrix) -> OchreResult<Matrix> {
        let theta = self.theta_.as_ref().ok_or(OchreError::NotFitted)?;
        let var = self.var_.as_ref().ok_or(OchreError::NotFitted)?;
        let priors = self.class_prior_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let n_samples = x.rows;
        let n_features = x.cols;
        let n_classes = priors.len();

        let mut class_const = vec![0.0; n_classes];
        for c in 0..n_classes {
            let log_prior = if priors[c] > 0.0 {
                priors[c].ln()
            } else {
                f64::NEG_INFINITY
            };
            let var_row = &var.data[(c * n_features)..((c + 1) * n_features)];
            let log_det_sum: f64 = var_row.iter().map(|&v| (2.0 * PI * v).ln()).sum();
            class_const[c] = log_prior - 0.5 * log_det_sum;
        }

        let mut jll = vec![0.0; n_samples * n_classes];
        jll.par_chunks_mut(n_classes)
            .enumerate()
            .for_each(|(i, dst)| {
                let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
                for c in 0..n_classes {
                    if !class_const[c].is_finite() {
                        dst[c] = f64::NEG_INFINITY;
                        continue;
                    }
                    let theta_row = &theta.data[(c * n_features)..((c + 1) * n_features)];
                    let var_row = &var.data[(c * n_features)..((c + 1) * n_features)];
                    let mut quad = 0.0;
                    for j in 0..n_features {
                        let d = x_row[j] - theta_row[j];
                        quad += (d * d) / var_row[j];
                    }
                    dst[c] = class_const[c] - 0.5 * quad;
                }
            });

        Matrix::from_vec(n_samples, n_classes, jll)
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        log_proba_from_jll(&jll)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        proba_from_jll(&jll)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let jll = self.predict_joint_log_likelihood(x)?;
        Ok(predict_from_jll(&jll, classes))
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        accuracy_score(&preds, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GAUSSIAN_NB);
        write_opt_f64_vec(&mut buf, &self.priors);
        write_f64(&mut buf, self.var_smoothing);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.class_count_);
        write_opt_f64_vec(&mut buf, &self.class_prior_);
        write_opt_matrix(&mut buf, &self.theta_);
        write_opt_matrix(&mut buf, &self.var_);
        write_f64(&mut buf, self.epsilon_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_GAUSSIAN_NB)?;
        let priors = reader.read_opt_f64_vec()?;
        let var_smoothing = reader.read_f64()?;
        let stored_dev = reader.read_opt_str()?;
        let classes_ = reader.read_opt_f64_vec()?;
        let class_count_ = reader.read_opt_f64_vec()?;
        let class_prior_ = reader.read_opt_f64_vec()?;
        let theta_ = reader.read_opt_matrix()?;
        let var_ = reader.read_opt_matrix()?;
        let epsilon_ = reader.read_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (resolved_dev, dev_str) =
            resolve_loaded_device(stored_dev, classes_.is_some(), device)?;
        Ok(Self {
            priors,
            var_smoothing,
            device_: dev_str,
            device: resolved_dev,
            classes_,
            class_count_,
            class_prior_,
            theta_,
            var_,
            epsilon_,
            n_features_in_,
        })
    }
}

#[pyclass(name = "GaussianNB", module = "ochreml._ochreml")]
pub struct PyGaussianNB {
    pub inner: GaussianNB,
}

crate::pymethods_with_dit!(PyGaussianNB, GaussianNB, {
    #[new]
    #[pyo3(signature = (priors = None, var_smoothing = 1e-9, device = None))]
    pub fn new(
        priors: Option<Vec<f64>>,
        var_smoothing: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianNB::new(priors, var_smoothing, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, classes = None, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        classes: Option<PyBuffer<f64>>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let cls_vec = classes
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner
            .partial_fit(&x_mat, &y_vec, cls_vec.as_deref(), sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_joint_log_likelihood_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_joint_log_likelihood(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn priors(&self) -> Option<Vec<f64>> {
        self.inner.priors.clone()
    }

    #[getter]
    pub fn var_smoothing(&self) -> f64 {
        self.inner.var_smoothing
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn class_count_(&self) -> Option<Vec<f64>> {
        self.inner.class_count_.clone()
    }

    #[getter]
    pub fn class_prior_(&self) -> Option<Vec<f64>> {
        self.inner.class_prior_.clone()
    }

    #[getter]
    pub fn epsilon_(&self) -> f64 {
        self.inner.epsilon_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    pub fn theta_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .theta_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }

    pub fn var_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .var_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
});

// ---------------------------------------------------------------------------
// 2. MultinomialNB
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MultinomialNB {
    pub alpha: f64,
    pub fit_prior: bool,
    pub class_prior: Option<Vec<f64>>,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    pub class_count_: Option<Vec<f64>>,
    pub class_log_prior_: Option<Vec<f64>>,
    pub feature_count_: Option<Matrix>,
    pub feature_log_prob_: Option<Matrix>,
    pub n_features_in_: Option<usize>,
}

impl MultinomialNB {
    pub fn new(
        alpha: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        let resolved_dev_str = device.clone().or_else(|| Some("auto".to_string()));
        let parsed_dev = parse_device(resolved_dev_str.as_deref())?;
        Ok(Self {
            alpha,
            fit_prior,
            class_prior,
            device_: resolved_dev_str,
            device: parsed_dev,
            classes_: None,
            class_count_: None,
            class_log_prior_: None,
            feature_count_: None,
            feature_log_prob_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.unwrap_or_else(|| "auto".to_string());
        self.device = parse_device(Some(&resolved))?;
        self.device_ = Some(resolved);
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.classes_ = None;
        self.class_count_ = None;
        self.class_log_prior_ = None;
        self.feature_count_ = None;
        self.feature_log_prob_ = None;
        self.n_features_in_ = None;
        self.partial_fit(x, y, None, sample_weight)
    }

    pub fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &[f64],
        classes: Option<&[f64]>,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_non_negative(x)?;
        validate_y_finite(y, x.rows)?;
        validate_sample_weight(sample_weight, x.rows)?;

        let n_features = x.cols;
        if self.classes_.is_none() {
            let sorted_classes = match classes {
                Some(c) => validate_and_sort_classes(c)?,
                None => extract_sorted_classes(y),
            };
            let n_classes = sorted_classes.len();
            if let Some(ref p) = self.class_prior {
                validate_priors(p, n_classes)?;
            }
            self.classes_ = Some(sorted_classes);
            self.class_count_ = Some(vec![0.0; n_classes]);
            self.feature_count_ = Some(Matrix::zeros(n_classes, n_features));
            self.feature_log_prob_ = Some(Matrix::zeros(n_classes, n_features));
            self.n_features_in_ = Some(n_features);
        } else if let Some(expected_p) = self.n_features_in_
            && n_features != expected_p
        {
            return Err(OchreError::FeatureCountMismatch {
                features: n_features,
                expected: expected_p,
            });
        }

        let classes_ref = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes_ref.len();
        let y_idx = map_labels_to_indices(y, classes_ref)?;

        let class_count = self.class_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_count = self.feature_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_log_prob = self
            .feature_log_prob_
            .as_mut()
            .ok_or(OchreError::NotFitted)?;

        for (i, &cls) in y_idx.iter().enumerate() {
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            class_count[cls] += w;
            let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
            let fc_row = &mut feature_count.data[(cls * n_features)..((cls + 1) * n_features)];
            for j in 0..n_features {
                fc_row[j] += w * x_row[j];
            }
        }

        for c in 0..n_classes {
            let fc_row = &feature_count.data[(c * n_features)..((c + 1) * n_features)];
            let flp_row = &mut feature_log_prob.data[(c * n_features)..((c + 1) * n_features)];
            let smoothed_cc: f64 = fc_row.iter().map(|&v| v + self.alpha).sum();
            for j in 0..n_features {
                let num = fc_row[j] + self.alpha;
                flp_row[j] = if num > 0.0 && smoothed_cc > 0.0 {
                    num.ln() - smoothed_cc.ln()
                } else {
                    f64::NEG_INFINITY
                };
            }
        }

        self.class_log_prior_ = Some(compute_class_log_prior(
            class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?);

        Ok(())
    }

    pub fn predict_joint_log_likelihood(&self, x: &Matrix) -> OchreResult<Matrix> {
        let flp = self
            .feature_log_prob_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let clp = self
            .class_log_prior_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let backend = get_backend(&self.device);
        linear_jll(&backend, x, flp, Some(clp))
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        log_proba_from_jll(&jll)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        proba_from_jll(&jll)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let jll = self.predict_joint_log_likelihood(x)?;
        Ok(predict_from_jll(&jll, classes))
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        accuracy_score(&preds, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MULTINOMIAL_NB);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_prior);
        write_opt_f64_vec(&mut buf, &self.class_prior);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.class_count_);
        write_opt_f64_vec(&mut buf, &self.class_log_prior_);
        write_opt_matrix(&mut buf, &self.feature_count_);
        write_opt_matrix(&mut buf, &self.feature_log_prob_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_MULTINOMIAL_NB)?;
        let alpha = reader.read_f64()?;
        let fit_prior = reader.read_bool()?;
        let class_prior = reader.read_opt_f64_vec()?;
        let stored_dev = reader.read_opt_str()?;
        let classes_ = reader.read_opt_f64_vec()?;
        let class_count_ = reader.read_opt_f64_vec()?;
        let class_log_prior_ = reader.read_opt_f64_vec()?;
        let feature_count_ = reader.read_opt_matrix()?;
        let feature_log_prob_ = reader.read_opt_matrix()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (resolved_dev, dev_str) =
            resolve_loaded_device(stored_dev, classes_.is_some(), device)?;
        Ok(Self {
            alpha,
            fit_prior,
            class_prior,
            device_: dev_str,
            device: resolved_dev,
            classes_,
            class_count_,
            class_log_prior_,
            feature_count_,
            feature_log_prob_,
            n_features_in_,
        })
    }
}

#[pyclass(name = "MultinomialNB", module = "ochreml._ochreml")]
pub struct PyMultinomialNB {
    pub inner: MultinomialNB,
}

crate::pymethods_with_dit!(PyMultinomialNB, MultinomialNB, {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_prior = true, class_prior = None, device = None))]
    pub fn new(
        alpha: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultinomialNB::new(alpha, fit_prior, class_prior, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, classes = None, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        classes: Option<PyBuffer<f64>>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let cls_vec = classes
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner
            .partial_fit(&x_mat, &y_vec, cls_vec.as_deref(), sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_joint_log_likelihood_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_joint_log_likelihood(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn fit_prior(&self) -> bool {
        self.inner.fit_prior
    }

    #[getter]
    pub fn class_prior(&self) -> Option<Vec<f64>> {
        self.inner.class_prior.clone()
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn class_count_(&self) -> Option<Vec<f64>> {
        self.inner.class_count_.clone()
    }

    #[getter]
    pub fn class_log_prior_(&self) -> Option<Vec<f64>> {
        self.inner.class_log_prior_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    pub fn feature_count_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_count_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }

    pub fn feature_log_prob_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_log_prob_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
});

// ---------------------------------------------------------------------------
// 3. BernoulliNB
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BernoulliNB {
    pub alpha: f64,
    pub binarize: Option<f64>,
    pub fit_prior: bool,
    pub class_prior: Option<Vec<f64>>,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    pub class_count_: Option<Vec<f64>>,
    pub class_log_prior_: Option<Vec<f64>>,
    pub feature_count_: Option<Matrix>,
    pub feature_log_prob_: Option<Matrix>,
    pub n_features_in_: Option<usize>,
}

fn binarize_matrix(x: &Matrix, threshold: Option<f64>) -> OchreResult<Matrix> {
    match threshold {
        Some(thresh) => {
            let mut data = vec![0.0; x.data.len()];
            data.par_iter_mut()
                .zip(x.data.par_iter())
                .for_each(|(dst, &src)| {
                    *dst = if src > thresh { 1.0 } else { 0.0 };
                });
            Matrix::from_vec(x.rows, x.cols, data)
        }
        None => Ok(x.clone()),
    }
}

impl BernoulliNB {
    pub fn new(
        alpha: f64,
        binarize: Option<f64>,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        if let Some(b) = binarize
            && !b.is_finite()
        {
            return Err(OchreError::InvalidParameter(
                "binarize threshold must be finite (・`ω´・)".to_string(),
            ));
        }
        let resolved_dev_str = device.clone().or_else(|| Some("auto".to_string()));
        let parsed_dev = parse_device(resolved_dev_str.as_deref())?;
        Ok(Self {
            alpha,
            binarize,
            fit_prior,
            class_prior,
            device_: resolved_dev_str,
            device: parsed_dev,
            classes_: None,
            class_count_: None,
            class_log_prior_: None,
            feature_count_: None,
            feature_log_prob_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.unwrap_or_else(|| "auto".to_string());
        self.device = parse_device(Some(&resolved))?;
        self.device_ = Some(resolved);
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.classes_ = None;
        self.class_count_ = None;
        self.class_log_prior_ = None;
        self.feature_count_ = None;
        self.feature_log_prob_ = None;
        self.n_features_in_ = None;
        self.partial_fit(x, y, None, sample_weight)
    }

    pub fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &[f64],
        classes: Option<&[f64]>,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        validate_sample_weight(sample_weight, x.rows)?;

        let n_features = x.cols;
        if self.classes_.is_none() {
            let sorted_classes = match classes {
                Some(c) => validate_and_sort_classes(c)?,
                None => extract_sorted_classes(y),
            };
            let n_classes = sorted_classes.len();
            if let Some(ref p) = self.class_prior {
                validate_priors(p, n_classes)?;
            }
            self.classes_ = Some(sorted_classes);
            self.class_count_ = Some(vec![0.0; n_classes]);
            self.feature_count_ = Some(Matrix::zeros(n_classes, n_features));
            self.feature_log_prob_ = Some(Matrix::zeros(n_classes, n_features));
            self.n_features_in_ = Some(n_features);
        } else if let Some(expected_p) = self.n_features_in_
            && n_features != expected_p
        {
            return Err(OchreError::FeatureCountMismatch {
                features: n_features,
                expected: expected_p,
            });
        }

        let x_bin = binarize_matrix(x, self.binarize)?;
        let classes_ref = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes_ref.len();
        let y_idx = map_labels_to_indices(y, classes_ref)?;

        let class_count = self.class_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_count = self.feature_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_log_prob = self
            .feature_log_prob_
            .as_mut()
            .ok_or(OchreError::NotFitted)?;

        for (i, &cls) in y_idx.iter().enumerate() {
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            class_count[cls] += w;
            let x_row = &x_bin.data[(i * n_features)..((i + 1) * n_features)];
            let fc_row = &mut feature_count.data[(cls * n_features)..((cls + 1) * n_features)];
            for j in 0..n_features {
                fc_row[j] += w * x_row[j];
            }
        }

        for (c, &cc) in class_count.iter().enumerate().take(n_classes) {
            let fc_row = &feature_count.data[(c * n_features)..((c + 1) * n_features)];
            let flp_row = &mut feature_log_prob.data[(c * n_features)..((c + 1) * n_features)];
            let denom = cc + 2.0 * self.alpha;
            for j in 0..n_features {
                let num = fc_row[j] + self.alpha;
                flp_row[j] = if num > 0.0 && denom > 0.0 {
                    num.ln() - denom.ln()
                } else {
                    f64::NEG_INFINITY
                };
            }
        }

        self.class_log_prior_ = Some(compute_class_log_prior(
            class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?);

        Ok(())
    }

    pub fn predict_joint_log_likelihood(&self, x: &Matrix) -> OchreResult<Matrix> {
        let flp = self
            .feature_log_prob_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let clp = self
            .class_log_prior_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let x_bin = binarize_matrix(x, self.binarize)?;
        let n_classes = flp.rows;
        let n_features = flp.cols;

        // neg_prob = log(1 - exp(feature_log_prob_))
        let mut neg_prob = vec![0.0; n_classes * n_features];
        let mut all_finite = true;
        for (dst, &lp) in neg_prob.iter_mut().zip(flp.data.iter()) {
            let q = (1.0 - lp.exp()).max(0.0);
            let lq = if q > 0.0 { q.ln() } else { f64::NEG_INFINITY };
            if !lq.is_finite() || !lp.is_finite() {
                all_finite = false;
            }
            *dst = lq;
        }

        if all_finite {
            let mut delta_data = vec![0.0; n_classes * n_features];
            for idx in 0..delta_data.len() {
                delta_data[idx] = flp.data[idx] - neg_prob[idx];
            }
            let delta = Matrix::from_vec(n_classes, n_features, delta_data)?;
            let mut bias = vec![0.0; n_classes];
            for c in 0..n_classes {
                let sum_neg: f64 = neg_prob[(c * n_features)..((c + 1) * n_features)]
                    .iter()
                    .sum();
                bias[c] = clp[c] + sum_neg;
            }
            let backend = get_backend(&self.device);
            return linear_jll(&backend, &x_bin, &delta, Some(&bias));
        }

        // Exact fallback when alpha == 0 produces log(0) = -inf, avoiding 0 * inf = NaN
        let n_samples = x_bin.rows;
        let mut jll = vec![0.0; n_samples * n_classes];
        jll.par_chunks_mut(n_classes)
            .enumerate()
            .for_each(|(i, dst)| {
                let x_row = &x_bin.data[(i * n_features)..((i + 1) * n_features)];
                for c in 0..n_classes {
                    let lp_row = &flp.data[(c * n_features)..((c + 1) * n_features)];
                    let lq_row = &neg_prob[(c * n_features)..((c + 1) * n_features)];
                    let mut sum = clp[c];
                    for j in 0..n_features {
                        let xij = x_row[j];
                        if xij != 0.0 {
                            sum += xij * lp_row[j];
                        }
                        let one_minus_x = 1.0 - xij;
                        if one_minus_x != 0.0 {
                            sum += one_minus_x * lq_row[j];
                        }
                    }
                    dst[c] = sum;
                }
            });
        Matrix::from_vec(n_samples, n_classes, jll)
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        log_proba_from_jll(&jll)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        proba_from_jll(&jll)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let jll = self.predict_joint_log_likelihood(x)?;
        Ok(predict_from_jll(&jll, classes))
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        accuracy_score(&preds, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_BERNOULLI_NB);
        write_f64(&mut buf, self.alpha);
        write_opt_f64(&mut buf, self.binarize);
        write_bool(&mut buf, self.fit_prior);
        write_opt_f64_vec(&mut buf, &self.class_prior);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.class_count_);
        write_opt_f64_vec(&mut buf, &self.class_log_prior_);
        write_opt_matrix(&mut buf, &self.feature_count_);
        write_opt_matrix(&mut buf, &self.feature_log_prob_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_BERNOULLI_NB)?;
        let alpha = reader.read_f64()?;
        let binarize = reader.read_opt_f64()?;
        let fit_prior = reader.read_bool()?;
        let class_prior = reader.read_opt_f64_vec()?;
        let stored_dev = reader.read_opt_str()?;
        let classes_ = reader.read_opt_f64_vec()?;
        let class_count_ = reader.read_opt_f64_vec()?;
        let class_log_prior_ = reader.read_opt_f64_vec()?;
        let feature_count_ = reader.read_opt_matrix()?;
        let feature_log_prob_ = reader.read_opt_matrix()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (resolved_dev, dev_str) =
            resolve_loaded_device(stored_dev, classes_.is_some(), device)?;
        Ok(Self {
            alpha,
            binarize,
            fit_prior,
            class_prior,
            device_: dev_str,
            device: resolved_dev,
            classes_,
            class_count_,
            class_log_prior_,
            feature_count_,
            feature_log_prob_,
            n_features_in_,
        })
    }
}

#[pyclass(name = "BernoulliNB", module = "ochreml._ochreml")]
pub struct PyBernoulliNB {
    pub inner: BernoulliNB,
}

crate::pymethods_with_dit!(PyBernoulliNB, BernoulliNB, {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        binarize = Some(0.0),
        fit_prior = true,
        class_prior = None,
        device = None
    ))]
    pub fn new(
        alpha: f64,
        binarize: Option<f64>,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: BernoulliNB::new(alpha, binarize, fit_prior, class_prior, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, classes = None, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        classes: Option<PyBuffer<f64>>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let cls_vec = classes
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner
            .partial_fit(&x_mat, &y_vec, cls_vec.as_deref(), sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_joint_log_likelihood_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_joint_log_likelihood(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn binarize(&self) -> Option<f64> {
        self.inner.binarize
    }

    #[getter]
    pub fn fit_prior(&self) -> bool {
        self.inner.fit_prior
    }

    #[getter]
    pub fn class_prior(&self) -> Option<Vec<f64>> {
        self.inner.class_prior.clone()
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn class_count_(&self) -> Option<Vec<f64>> {
        self.inner.class_count_.clone()
    }

    #[getter]
    pub fn class_log_prior_(&self) -> Option<Vec<f64>> {
        self.inner.class_log_prior_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    pub fn feature_count_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_count_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }

    pub fn feature_log_prob_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_log_prob_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
});

// ---------------------------------------------------------------------------
// 4. ComplementNB
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ComplementNB {
    pub alpha: f64,
    pub fit_prior: bool,
    pub class_prior: Option<Vec<f64>>,
    pub norm: bool,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    pub class_count_: Option<Vec<f64>>,
    pub class_log_prior_: Option<Vec<f64>>,
    pub feature_count_: Option<Matrix>,
    pub feature_all_: Option<Vec<f64>>,
    pub feature_log_prob_: Option<Matrix>,
    pub n_features_in_: Option<usize>,
}

impl ComplementNB {
    pub fn new(
        alpha: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        norm: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        let resolved_dev_str = device.clone().or_else(|| Some("auto".to_string()));
        let parsed_dev = parse_device(resolved_dev_str.as_deref())?;
        Ok(Self {
            alpha,
            fit_prior,
            class_prior,
            norm,
            device_: resolved_dev_str,
            device: parsed_dev,
            classes_: None,
            class_count_: None,
            class_log_prior_: None,
            feature_count_: None,
            feature_all_: None,
            feature_log_prob_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.unwrap_or_else(|| "auto".to_string());
        self.device = parse_device(Some(&resolved))?;
        self.device_ = Some(resolved);
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.classes_ = None;
        self.class_count_ = None;
        self.class_log_prior_ = None;
        self.feature_count_ = None;
        self.feature_all_ = None;
        self.feature_log_prob_ = None;
        self.n_features_in_ = None;
        self.partial_fit(x, y, None, sample_weight)
    }

    pub fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &[f64],
        classes: Option<&[f64]>,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_non_negative(x)?;
        validate_y_finite(y, x.rows)?;
        validate_sample_weight(sample_weight, x.rows)?;

        let n_features = x.cols;
        if self.classes_.is_none() {
            let sorted_classes = match classes {
                Some(c) => validate_and_sort_classes(c)?,
                None => extract_sorted_classes(y),
            };
            let n_classes = sorted_classes.len();
            if let Some(ref p) = self.class_prior {
                validate_priors(p, n_classes)?;
            }
            self.classes_ = Some(sorted_classes);
            self.class_count_ = Some(vec![0.0; n_classes]);
            self.feature_count_ = Some(Matrix::zeros(n_classes, n_features));
            self.feature_all_ = Some(vec![0.0; n_features]);
            self.feature_log_prob_ = Some(Matrix::zeros(n_classes, n_features));
            self.n_features_in_ = Some(n_features);
        } else if let Some(expected_p) = self.n_features_in_
            && n_features != expected_p
        {
            return Err(OchreError::FeatureCountMismatch {
                features: n_features,
                expected: expected_p,
            });
        }

        let classes_ref = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes_ref.len();
        let y_idx = map_labels_to_indices(y, classes_ref)?;

        let class_count = self.class_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_count = self.feature_count_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_all = self.feature_all_.as_mut().ok_or(OchreError::NotFitted)?;
        let feature_log_prob = self
            .feature_log_prob_
            .as_mut()
            .ok_or(OchreError::NotFitted)?;

        for (i, &cls) in y_idx.iter().enumerate() {
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            class_count[cls] += w;
            let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
            let fc_row = &mut feature_count.data[(cls * n_features)..((cls + 1) * n_features)];
            for j in 0..n_features {
                let inc = w * x_row[j];
                fc_row[j] += inc;
                feature_all[j] += inc;
            }
        }

        for c in 0..n_classes {
            let fc_row = &feature_count.data[(c * n_features)..((c + 1) * n_features)];
            let flp_row = &mut feature_log_prob.data[(c * n_features)..((c + 1) * n_features)];

            let mut comp_total = 0.0;
            for j in 0..n_features {
                let comp = (feature_all[j] + self.alpha - fc_row[j]).max(0.0);
                flp_row[j] = comp;
                comp_total += comp;
            }

            let log_total = if comp_total > 0.0 {
                comp_total.ln()
            } else {
                0.0
            };
            for val in flp_row.iter_mut() {
                let comp = *val;
                let logged = if comp > 0.0 && comp_total > 0.0 {
                    comp.ln() - log_total
                } else {
                    f64::NEG_INFINITY
                };
                *val = -logged;
            }

            if self.norm {
                let l1: f64 = flp_row.iter().map(|v| v.abs()).sum();
                if l1 > 0.0 && l1.is_finite() {
                    for val in flp_row.iter_mut() {
                        *val /= l1;
                    }
                }
            }
        }

        self.class_log_prior_ = Some(compute_class_log_prior(
            class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?);

        Ok(())
    }

    pub fn predict_joint_log_likelihood(&self, x: &Matrix) -> OchreResult<Matrix> {
        let flp = self
            .feature_log_prob_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let clp = self
            .class_log_prior_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let backend = get_backend(&self.device);
        let bias = if clp.len() == 1 {
            Some(clp.as_slice())
        } else {
            None
        };
        linear_jll(&backend, x, flp, bias)
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        log_proba_from_jll(&jll)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let jll = self.predict_joint_log_likelihood(x)?;
        proba_from_jll(&jll)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let jll = self.predict_joint_log_likelihood(x)?;
        Ok(predict_from_jll(&jll, classes))
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        accuracy_score(&preds, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_COMPLEMENT_NB);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_prior);
        write_opt_f64_vec(&mut buf, &self.class_prior);
        write_bool(&mut buf, self.norm);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.class_count_);
        write_opt_f64_vec(&mut buf, &self.class_log_prior_);
        write_opt_matrix(&mut buf, &self.feature_count_);
        write_opt_f64_vec(&mut buf, &self.feature_all_);
        write_opt_matrix(&mut buf, &self.feature_log_prob_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_COMPLEMENT_NB)?;
        let alpha = reader.read_f64()?;
        let fit_prior = reader.read_bool()?;
        let class_prior = reader.read_opt_f64_vec()?;
        let norm = reader.read_bool()?;
        let stored_dev = reader.read_opt_str()?;
        let classes_ = reader.read_opt_f64_vec()?;
        let class_count_ = reader.read_opt_f64_vec()?;
        let class_log_prior_ = reader.read_opt_f64_vec()?;
        let feature_count_ = reader.read_opt_matrix()?;
        let feature_all_ = reader.read_opt_f64_vec()?;
        let feature_log_prob_ = reader.read_opt_matrix()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (resolved_dev, dev_str) =
            resolve_loaded_device(stored_dev, classes_.is_some(), device)?;
        Ok(Self {
            alpha,
            fit_prior,
            class_prior,
            norm,
            device_: dev_str,
            device: resolved_dev,
            classes_,
            class_count_,
            class_log_prior_,
            feature_count_,
            feature_all_,
            feature_log_prob_,
            n_features_in_,
        })
    }
}

#[pyclass(name = "ComplementNB", module = "ochreml._ochreml")]
pub struct PyComplementNB {
    pub inner: ComplementNB,
}

crate::pymethods_with_dit!(PyComplementNB, ComplementNB, {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        fit_prior = true,
        class_prior = None,
        norm = false,
        device = None
    ))]
    pub fn new(
        alpha: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        norm: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ComplementNB::new(alpha, fit_prior, class_prior, norm, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, classes = None, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        classes: Option<PyBuffer<f64>>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let cls_vec = classes
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|b| buffer_to_f64_1d(py, b))
            .transpose()?;
        self.inner
            .partial_fit(&x_mat, &y_vec, cls_vec.as_deref(), sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_joint_log_likelihood_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_joint_log_likelihood(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn fit_prior(&self) -> bool {
        self.inner.fit_prior
    }

    #[getter]
    pub fn class_prior(&self) -> Option<Vec<f64>> {
        self.inner.class_prior.clone()
    }

    #[getter]
    pub fn norm(&self) -> bool {
        self.inner.norm
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn class_count_(&self) -> Option<Vec<f64>> {
        self.inner.class_count_.clone()
    }

    #[getter]
    pub fn class_log_prior_(&self) -> Option<Vec<f64>> {
        self.inner.class_log_prior_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    pub fn feature_count_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_count_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }

    pub fn feature_all_(&self) -> Option<Vec<f64>> {
        self.inner.feature_all_.clone()
    }

    pub fn feature_log_prob_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        self.inner
            .feature_log_prob_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
});

// ---------------------------------------------------------------------------
// PyO3 Module Registration
// ---------------------------------------------------------------------------

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianNB>()?;
    m.add_class::<PyMultinomialNB>()?;
    m.add_class::<PyBernoulliNB>()?;
    m.add_class::<PyComplementNB>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Rust Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb_fit_and_partial_fit() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 3.0, 0.0]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut gnb = GaussianNB::new(None, 1e-9, Some("cpu".to_string())).unwrap();
        gnb.fit(&x, &y, None).unwrap();
        assert_eq!(gnb.score(&x, &y).unwrap(), 1.0);

        let bytes = gnb.to_bytes();
        let restored = GaussianNB::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(restored.predict(&x).unwrap(), gnb.predict(&x).unwrap());
    }

    #[test]
    fn test_multinomial_bernoulli_complement_nb() {
        let x = Matrix::from_vec(4, 2, vec![3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 2.0]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut mnb = MultinomialNB::new(1.0, true, None, Some("cpu".to_string())).unwrap();
        mnb.fit(&x, &y, None).unwrap();
        assert_eq!(mnb.score(&x, &y).unwrap(), 1.0);

        let mut bnb =
            BernoulliNB::new(1.0, Some(0.0), true, None, Some("cpu".to_string())).unwrap();
        bnb.fit(&x, &y, None).unwrap();
        assert_eq!(bnb.score(&x, &y).unwrap(), 1.0);

        let mut cnb = ComplementNB::new(1.0, true, None, true, Some("cpu".to_string())).unwrap();
        cnb.fit(&x, &y, None).unwrap();
        assert_eq!(cnb.score(&x, &y).unwrap(), 1.0);
    }
}
