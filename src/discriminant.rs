// OchreML Discriminant Analysis Module (*≧ω≦*)
// Implements LinearDiscriminantAnalysis (LDA) and QuadraticDiscriminantAnalysis (QDA)
// accelerated by Backend Primitives (N1, N5, E1, E2, E4).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize, write_str,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::logistic_regression::{extract_sorted_classes, map_labels_to_indices};
use crate::matrix::{Matrix, solve_cholesky_multi};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_LDA: u8 = 45;
pub const TAG_QDA: u8 = 46;

const PARALLEL_ROW_THRESHOLD: usize = 4096;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < 131_072 {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn dot4(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

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

fn validate_tol(tol: f64) -> OchreResult<()> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be non-negative and finite (・`ω´・)"
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

type ClassStatsResult = (Vec<usize>, Vec<f64>, Matrix, Vec<f64>);

fn compute_class_stats(
    x: &Matrix,
    y_idx: &[usize],
    n_classes: usize,
    priors_param: Option<&[f64]>,
) -> OchreResult<ClassStatsResult> {
    let n_samples = x.rows;
    let n_features = x.cols;
    let mut counts = vec![0usize; n_classes];
    let mut means_data = vec![0.0; n_classes * n_features];

    for (i, &cls) in y_idx.iter().enumerate() {
        counts[cls] += 1;
        let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
        let m_row = &mut means_data[(cls * n_features)..((cls + 1) * n_features)];
        for j in 0..n_features {
            m_row[j] += x_row[j];
        }
    }

    for k in 0..n_classes {
        if counts[k] == 0 {
            return Err(OchreError::InvalidParameter(format!(
                "Class index {k} has zero samples ( >_< )"
            )));
        }
        let inv = 1.0 / (counts[k] as f64);
        let m_row = &mut means_data[(k * n_features)..((k + 1) * n_features)];
        for val in m_row.iter_mut() {
            *val *= inv;
        }
    }

    let priors = match priors_param {
        Some(p) => validate_priors(p, n_classes)?,
        None => counts
            .iter()
            .map(|&c| (c as f64) / (n_samples as f64))
            .collect(),
    };

    let mut xbar = vec![0.0; n_features];
    for k in 0..n_classes {
        let pk = priors[k];
        let m_row = &means_data[(k * n_features)..((k + 1) * n_features)];
        for j in 0..n_features {
            xbar[j] += pk * m_row[j];
        }
    }

    let means = Matrix::from_vec(n_classes, n_features, means_data)?;
    Ok((counts, priors, means, xbar))
}

fn ledoit_wolf_shrinkage(xc: &Matrix, s_emp: &Matrix) -> f64 {
    let n = xc.rows as f64;
    let p = xc.cols;
    if p == 0 || xc.rows == 0 {
        return 0.0;
    }
    let p_f = p as f64;
    let mut trace = 0.0;
    for i in 0..p {
        trace += s_emp.data[i * p + i];
    }
    let mu = trace / p_f;

    let mut delta = 0.0;
    let mut frob_sq_s = 0.0;
    for r in 0..p {
        for c in 0..p {
            let v = s_emp.data[r * p + c];
            frob_sq_s += v * v;
            let diff = if r == c { v - mu } else { v };
            delta += diff * diff;
        }
    }
    delta /= p_f;
    if delta <= 1e-30 {
        return 0.0;
    }

    let sum_norm4: f64 = if xc.rows >= PARALLEL_ROW_THRESHOLD {
        xc.data
            .par_chunks_exact(p)
            .map(|row| {
                let sq: f64 = row.iter().map(|&v| v * v).sum();
                sq * sq
            })
            .sum()
    } else {
        xc.data
            .chunks_exact(p)
            .map(|row| {
                let sq: f64 = row.iter().map(|&v| v * v).sum();
                sq * sq
            })
            .sum()
    };

    let beta_bar = (sum_norm4 / (n * n) - frob_sq_s / n) / p_f;
    let beta = beta_bar.max(0.0).min(delta);
    (beta / delta).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn compute_within_class_cov(
    backend: &Backend<'_>,
    x: &Matrix,
    y_idx: &[usize],
    means: &Matrix,
    priors: &[f64],
    counts: &[usize],
    shrinkage: Option<f64>,
    shrinkage_auto: bool,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let n_classes = means.rows;

    let mut xc_data = vec![0.0; n * p];
    for (i, &cls) in y_idx.iter().enumerate() {
        let x_row = &x.data[(i * p)..((i + 1) * p)];
        let m_row = &means.data[(cls * p)..((cls + 1) * p)];
        let out_row = &mut xc_data[(i * p)..((i + 1) * p)];
        for j in 0..p {
            out_row[j] = x_row[j] - m_row[j];
        }
    }
    let xc = Matrix::from_vec(n, p, xc_data)?;

    // Compute S_w = sum_k priors[k] * (1 / n_k) * X_k_c^T X_k_c
    let mut s_w = Matrix::zeros(p, p);
    for k in 0..n_classes {
        let nk = counts[k];
        if nk == 0 {
            continue;
        }
        let mut xk_data = Vec::with_capacity(nk * p);
        for (i, &cls) in y_idx.iter().enumerate() {
            if cls == k {
                xk_data.extend_from_slice(&xc.data[(i * p)..((i + 1) * p)]);
            }
        }
        let xk = Matrix::from_vec(nk, p, xk_data)?;
        let dummy_y = vec![0.0; nk];
        let (gram_k, _) = backend.compute_normal_equation(&xk, &dummy_y, false)?;
        let scale = priors[k] / (nk as f64);
        for idx in 0..(p * p) {
            s_w.data[idx] += scale * gram_k.data[idx];
        }
    }

    let shrink = if shrinkage_auto {
        let dummy_y = vec![0.0; n];
        let (gram_all, _) = backend.compute_normal_equation(&xc, &dummy_y, false)?;
        let mut s_emp = gram_all;
        let inv_n = 1.0 / (n as f64);
        for v in s_emp.data.iter_mut() {
            *v *= inv_n;
        }
        Some(ledoit_wolf_shrinkage(&xc, &s_emp))
    } else {
        shrinkage
    };

    if let Some(alpha) = shrink {
        let mut trace = 0.0;
        for i in 0..p {
            trace += s_w.data[i * p + i];
        }
        let mu = trace / (p as f64);
        let one_minus = 1.0 - alpha;
        for r in 0..p {
            for c in 0..p {
                let idx = r * p + c;
                s_w.data[idx] *= one_minus;
                if r == c {
                    s_w.data[idx] += alpha * mu;
                }
            }
        }
    }

    Ok(s_w)
}

fn log_softmax_matrix(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let k = scores.cols;
    let mut out = vec![0.0; n * k];
    let compute_row = |(dst, src): (&mut [f64], &[f64])| {
        let max_v = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = src.iter().map(|&v| (v - max_v).exp()).sum();
        let log_norm = max_v + sum_exp.ln();
        for j in 0..k {
            dst[j] = src[j] - log_norm;
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(k)
            .zip(scores.data.par_chunks_exact(k))
            .for_each(compute_row);
    } else {
        out.chunks_exact_mut(k)
            .zip(scores.data.chunks_exact(k))
            .for_each(compute_row);
    }
    Matrix::from_vec(n, k, out)
}

fn softmax_matrix(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let k = scores.cols;
    let mut out = vec![0.0; n * k];
    let compute_row = |(dst, src): (&mut [f64], &[f64])| {
        let max_v = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0;
        for j in 0..k {
            let e = (src[j] - max_v).exp();
            dst[j] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for v in dst.iter_mut() {
            *v *= inv_sum;
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(k)
            .zip(scores.data.par_chunks_exact(k))
            .for_each(compute_row);
    } else {
        out.chunks_exact_mut(k)
            .zip(scores.data.chunks_exact(k))
            .for_each(compute_row);
    }
    Matrix::from_vec(n, k, out)
}

fn binary_log_proba_from_decision(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let mut out = vec![0.0; n * 2];
    let compute_row = |(dst, &z): (&mut [f64], &f64)| {
        // log P(y=1) = -softplus(-z), log P(y=0) = -softplus(z)
        let (lp0, lp1) = if z >= 0.0 {
            let e = (-z).exp();
            let sp = e.ln_1p();
            (-z - sp, -sp)
        } else {
            let e = z.exp();
            let sp = e.ln_1p();
            (-sp, z - sp)
        };
        dst[0] = lp0;
        dst[1] = lp1;
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(2)
            .zip(scores.data.par_iter())
            .for_each(compute_row);
    } else {
        for (dst, &z) in out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(scores.data.iter())
        {
            compute_row((dst, &z));
        }
    }
    Matrix::from_vec(n, 2, out)
}

fn binary_proba_from_decision(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let mut out = vec![0.0; n * 2];
    let compute_row = |(dst, &z): (&mut [f64], &f64)| {
        let p1 = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        };
        dst[0] = 1.0 - p1;
        dst[1] = p1;
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(2)
            .zip(scores.data.par_iter())
            .for_each(compute_row);
    } else {
        for (dst, &z) in out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(scores.data.iter())
        {
            compute_row((dst, &z));
        }
    }
    Matrix::from_vec(n, 2, out)
}

fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> OchreResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_pred.len(),
        });
    }
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(a, b)| (a - b).abs() < 1e-12)
        .count();
    Ok((correct as f64) / (y_true.len() as f64))
}

/// Linear Discriminant Analysis (LDA) classifier & dimensionality reducer (＾▽＾)
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    pub solver: String,
    pub shrinkage: Option<f64>,
    pub shrinkage_auto: bool,
    pub priors: Option<Vec<f64>>,
    pub n_components: Option<usize>,
    pub tol: f64,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<f64>>,
    pub priors_: Option<Vec<f64>>,
    pub means_: Option<Matrix>,
    pub xbar_: Option<Vec<f64>>,
    pub coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub scalings_: Option<Matrix>,
    pub explained_variance_ratio_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub n_components_: Option<usize>,
}

impl LinearDiscriminantAnalysis {
    pub fn new(
        solver: String,
        shrinkage: Option<f64>,
        shrinkage_auto: bool,
        priors: Option<Vec<f64>>,
        n_components: Option<usize>,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if solver != "svd" && solver != "lsqr" && solver != "eigen" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid solver '{solver}'. Supported solvers are 'svd', 'lsqr', and 'eigen' (・`ω´・)"
            )));
        }
        if solver == "svd" && (shrinkage.is_some() || shrinkage_auto) {
            return Err(OchreError::InvalidParameter(
                "shrinkage is not supported with solver='svd'; use 'lsqr' or 'eigen' (・`ω´・)"
                    .to_string(),
            ));
        }
        if let Some(s) = shrinkage
            && (!s.is_finite() || !(0.0..=1.0).contains(&s))
        {
            return Err(OchreError::InvalidParameter(format!(
                "shrinkage ({s}) must be a finite float in [0.0, 1.0] (・`ω´・)"
            )));
        }
        if let Some(nc) = n_components
            && nc == 0
        {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        validate_tol(tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            solver,
            shrinkage,
            shrinkage_auto,
            priors,
            n_components,
            tol,
            device: dev,
            device_: None,
            classes_: None,
            priors_: None,
            means_: None,
            xbar_: None,
            coef_: None,
            intercept_: None,
            scalings_: None,
            explained_variance_ratio_: None,
            n_features_in_: None,
            n_components_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.n_features_in_.is_some() || self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    fn reset_fitted(&mut self) {
        self.classes_ = None;
        self.priors_ = None;
        self.means_ = None;
        self.xbar_ = None;
        self.coef_ = None;
        self.intercept_ = None;
        self.scalings_ = None;
        self.explained_variance_ratio_ = None;
        self.n_features_in_ = None;
        self.n_components_ = None;
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.reset_fitted();
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;

        let n_samples = x.rows;
        let n_features = x.cols;
        let classes = extract_sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(OchreError::InsufficientClasses(n_classes));
        }
        if n_samples <= n_classes {
            return Err(OchreError::InvalidParameter(
                "The number of samples must be more than the number of classes ( >_< )".to_string(),
            ));
        }

        let max_components = (n_classes - 1).min(n_features);
        let resolved_n_comp = match self.n_components {
            Some(nc) => {
                if nc > max_components {
                    return Err(OchreError::InvalidParameter(format!(
                        "n_components ({nc}) cannot be larger than min(n_features, n_classes - 1) = {max_components} ( >_< )"
                    )));
                }
                nc
            }
            None => max_components,
        };

        let y_idx = map_labels_to_indices(y, &classes)?;
        let (counts, priors, means, xbar) =
            compute_class_stats(x, &y_idx, n_classes, self.priors.as_deref())?;

        let backend = effective_backend(&self.device, n_samples * n_features);

        // Centered class means: M_c = means - xbar (n_classes x n_features)
        let mut means_centered_data = vec![0.0; n_classes * n_features];
        for k in 0..n_classes {
            for j in 0..n_features {
                means_centered_data[k * n_features + j] = means.data[k * n_features + j] - xbar[j];
            }
        }
        let means_centered = Matrix::from_vec(n_classes, n_features, means_centered_data)?;

        let (coef_full, scalings_opt, evr_opt) = match self.solver.as_str() {
            "svd" => self.solve_svd(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                n_samples,
                n_features,
                n_classes,
            )?,
            "eigen" => self.solve_eigen(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                &counts,
                n_features,
                n_classes,
            )?,
            "lsqr" => self.solve_lsqr(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                &counts,
                n_features,
                n_classes,
            )?,
            _ => unreachable!(),
        };

        // Compute intercepts: b_k = -0.5 * (mu_k + xbar)^T w_k + ln(priors[k])
        let mut intercept_full = vec![0.0; n_classes];
        for (k, b_out) in intercept_full.iter_mut().enumerate() {
            let mut dot = 0.0;
            for (j, &xb) in xbar.iter().enumerate() {
                let mu_plus_xbar = means.data[k * n_features + j] + xb;
                dot += mu_plus_xbar * coef_full.data[k * n_features + j];
            }
            let log_p = if priors[k] > 0.0 {
                priors[k].ln()
            } else {
                f64::NEG_INFINITY
            };
            *b_out = -0.5 * dot + log_p;
        }

        // Binary sklearn convention: coef_ is (1, n_features) = w_1 - w_0, intercept_ is (1,) = b_1 - b_0
        let (coef_stored, intercept_stored) = if n_classes == 2 {
            let mut w_bin = vec![0.0; n_features];
            for (j, dst) in w_bin.iter_mut().enumerate() {
                *dst = coef_full.data[n_features + j] - coef_full.data[j];
            }
            let b_bin = vec![intercept_full[1] - intercept_full[0]];
            (Matrix::from_vec(1, n_features, w_bin)?, b_bin)
        } else {
            (coef_full, intercept_full)
        };

        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
        self.means_ = Some(means);
        self.xbar_ = Some(xbar);
        self.coef_ = Some(coef_stored);
        self.intercept_ = Some(intercept_stored);
        self.scalings_ = scalings_opt;
        self.explained_variance_ratio_ = evr_opt;
        self.n_features_in_ = Some(n_features);
        self.n_components_ = Some(resolved_n_comp);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_svd(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let mut xc_data = vec![0.0; n_samples * n_features];
        let mut col_sq = vec![0.0; n_features];
        for (i, &cls) in y_idx.iter().enumerate() {
            let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
            let m_row = &means.data[(cls * n_features)..((cls + 1) * n_features)];
            let out_row = &mut xc_data[(i * n_features)..((i + 1) * n_features)];
            for j in 0..n_features {
                let d = x_row[j] - m_row[j];
                out_row[j] = d;
                col_sq[j] += d * d;
            }
        }

        let inv_n = 1.0 / (n_samples as f64);
        let stds: Vec<f64> = col_sq
            .iter()
            .map(|&s| {
                let v = (s * inv_n).sqrt();
                if v == 0.0 { 1.0 } else { v }
            })
            .collect();

        let xc = Matrix::from_vec(n_samples, n_features, xc_data)?;
        let dummy_y = vec![0.0; n_samples];
        let (mut gram, _) = backend.compute_normal_equation(&xc, &dummy_y, false)?;

        let fac = 1.0 / (n_samples as f64);
        for r in 0..n_features {
            for c in 0..n_features {
                gram.data[r * n_features + c] *= fac / (stds[r] * stds[c]);
            }
        }

        let eig_w = backend.small_sym_eig_svd(&gram)?;
        let s_vals: Vec<f64> = eig_w
            .eigenvalues
            .iter()
            .map(|&lam| lam.max(0.0).sqrt())
            .collect();

        let rank = s_vals.iter().filter(|&&s| s > self.tol).count();
        if rank == 0 {
            return Err(OchreError::SingularMatrix(
                "All within-class singular values are below tol".to_string(),
            ));
        }

        // scalings_w has shape (n_features, rank)
        let mut scalings_w_data = vec![0.0; n_features * rank];
        for r in 0..n_features {
            for k in 0..rank {
                scalings_w_data[r * rank + k] =
                    eig_w.eigenvectors.data[r * n_features + k] / (stds[r] * s_vals[k]);
            }
        }
        let scalings_w = Matrix::from_vec(n_features, rank, scalings_w_data)?;

        // Project weighted centered class means onto whitened space:
        // X_b = diag(sqrt(n_samples * priors)) * means_centered * scalings_w  (n_classes x rank)
        let mut xb_raw = backend.matmul(means_centered, &scalings_w)?;
        for (k, &pk) in priors.iter().enumerate() {
            let w = (n_samples as f64 * pk).sqrt();
            for j in 0..rank {
                xb_raw.data[k * rank + j] *= w;
            }
        }
        let dummy_yb = vec![0.0; n_classes];
        let (gram_b, _) = backend.compute_normal_equation(&xb_raw, &dummy_yb, false)?;
        let eig_b = backend.small_sym_eig_svd(&gram_b)?;

        let n_comp_avail = rank.min(n_classes - 1);
        let scalings_full = backend.matmul(&scalings_w, &eig_b.eigenvectors)?;

        let total_ev: f64 = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| v.max(0.0))
            .sum();
        let evr: Vec<f64> = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| {
                if total_ev > 0.0 {
                    v.max(0.0) / total_ev
                } else {
                    0.0
                }
            })
            .collect();

        // coef_full = means_centered * scalings_w * scalings_w^T (n_classes x n_features)
        let proj = backend.matmul(means_centered, &scalings_w)?;
        let scalings_w_t = scalings_w.transpose();
        let coef_full = backend.matmul(&proj, &scalings_w_t)?;

        Ok((coef_full, Some(scalings_full), Some(evr)))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_eigen(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        counts: &[usize],
        n_features: usize,
        n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let s_w = compute_within_class_cov(
            backend,
            x,
            y_idx,
            means,
            priors,
            counts,
            self.shrinkage,
            self.shrinkage_auto,
        )?;

        let eig_w = backend.small_sym_eig_svd(&s_w)?;
        let max_lam = eig_w.eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
        let thresh = self.tol * max_lam.max(1.0);
        let rank = eig_w
            .eigenvalues
            .iter()
            .filter(|&&lam| lam > thresh && lam > 0.0)
            .count();
        if rank == 0 {
            return Err(OchreError::SingularMatrix(
                "Within-class covariance matrix is zero or singular".to_string(),
            ));
        }

        // S_w^{-1/2} = V_r * diag(1 / sqrt(lam_r))  (n_features x rank)
        let mut whiten_data = vec![0.0; n_features * rank];
        for r in 0..n_features {
            for k in 0..rank {
                whiten_data[r * rank + k] =
                    eig_w.eigenvectors.data[r * n_features + k] / eig_w.eigenvalues[k].sqrt();
            }
        }
        let whiten = Matrix::from_vec(n_features, rank, whiten_data)?;

        // S_b = sum_k priors[k] * (mu_k - xbar)(mu_k - xbar)^T
        let mut sb_data = vec![0.0; n_features * n_features];
        for (k, &pk) in priors.iter().enumerate() {
            let row = &means_centered.data[(k * n_features)..((k + 1) * n_features)];
            for r in 0..n_features {
                for c in 0..n_features {
                    sb_data[r * n_features + c] += pk * row[r] * row[c];
                }
            }
        }
        let s_b = Matrix::from_vec(n_features, n_features, sb_data)?;

        // M = whiten^T * S_b * whiten (rank x rank)
        let sb_whiten = backend.matmul(&s_b, &whiten)?;
        let m_mat = backend.matmul_at_b(&whiten, &sb_whiten)?;
        let eig_b = backend.small_sym_eig_svd(&m_mat)?;

        let n_comp_avail = rank.min(n_classes - 1);
        let scalings_full = backend.matmul(&whiten, &eig_b.eigenvectors)?;

        let total_ev: f64 = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| v.max(0.0))
            .sum();
        let evr: Vec<f64> = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| {
                if total_ev > 0.0 {
                    v.max(0.0) / total_ev
                } else {
                    0.0
                }
            })
            .collect();

        // coef_full = means_centered * whiten * whiten^T
        let proj = backend.matmul(means_centered, &whiten)?;
        let whiten_t = whiten.transpose();
        let coef_full = backend.matmul(&proj, &whiten_t)?;

        Ok((coef_full, Some(scalings_full), Some(evr)))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_lsqr(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        counts: &[usize],
        n_features: usize,
        _n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let s_w = compute_within_class_cov(
            backend,
            x,
            y_idx,
            means,
            priors,
            counts,
            self.shrinkage,
            self.shrinkage_auto,
        )?;

        let rhs = means_centered.transpose();
        let coef_full = match solve_cholesky_multi(&s_w, &rhs) {
            Ok(sol) => sol.transpose(),
            Err(_) => {
                // Fallback to eigendecomposition pseudo-inverse for rank-deficient S_w
                let eig = backend.small_sym_eig_svd(&s_w)?;
                let max_lam = eig.eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
                let thresh = self.tol * max_lam.max(1.0);
                let rank = eig
                    .eigenvalues
                    .iter()
                    .filter(|&&lam| lam > thresh && lam > 0.0)
                    .count();
                if rank == 0 {
                    return Err(OchreError::SingularMatrix(
                        "Within-class covariance matrix is zero or singular".to_string(),
                    ));
                }
                let mut whiten_data = vec![0.0; n_features * rank];
                for r in 0..n_features {
                    for k in 0..rank {
                        whiten_data[r * rank + k] =
                            eig.eigenvectors.data[r * n_features + k] / eig.eigenvalues[k].sqrt();
                    }
                }
                let whiten = Matrix::from_vec(n_features, rank, whiten_data)?;
                let proj = backend.matmul(means_centered, &whiten)?;
                let whiten_t = whiten.transpose();
                backend.matmul(&proj, &whiten_t)?
            }
        };

        Ok((coef_full, None, None))
    }

    fn validate_predict_input(&self, x: &Matrix) -> OchreResult<usize> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        Ok(expected)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let p = self.validate_predict_input(x)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = x.rows;
        let k = coef.rows;
        let backend = effective_backend(&self.device, n * p);

        if matches!(backend, Backend::Cpu) {
            let mut out = vec![0.0; n * k];
            let x_data = &x.data;
            let c_data = &coef.data;
            if k == 1 {
                let w0 = &c_data[..p];
                let b0 = intercept[0];
                let compute_row = |(r, dst): (usize, &mut f64)| {
                    let x_row = &x_data[r * p..(r + 1) * p];
                    *dst = dot4(x_row, w0) + b0;
                };
                if n >= PARALLEL_ROW_THRESHOLD {
                    out.par_iter_mut().enumerate().for_each(compute_row);
                } else {
                    out.iter_mut().enumerate().for_each(compute_row);
                }
            } else {
                let compute_row = |(r, dst): (usize, &mut [f64])| {
                    let x_row = &x_data[r * p..(r + 1) * p];
                    for c in 0..k {
                        let c_row = &c_data[c * p..(c + 1) * p];
                        dst[c] = dot4(x_row, c_row) + intercept[c];
                    }
                };
                if n >= PARALLEL_ROW_THRESHOLD {
                    out.par_chunks_exact_mut(k)
                        .enumerate()
                        .for_each(compute_row);
                } else {
                    out.chunks_exact_mut(k).enumerate().for_each(compute_row);
                }
            }
            return Matrix::from_vec(n, k, out);
        }

        let coef_t = coef.transpose();
        let mut scores = backend.matmul(x, &coef_t)?;
        for r in 0..scores.rows {
            for (c, &b) in intercept.iter().enumerate().take(k) {
                scores.data[r * k + c] += b;
            }
        }
        Ok(scores)
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let scores = self.decision_function(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            binary_log_proba_from_decision(&scores)
        } else {
            log_softmax_matrix(&scores)
        }
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let scores = self.decision_function(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            binary_proba_from_decision(&scores)
        } else {
            softmax_matrix(&scores)
        }
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let p = self.validate_predict_input(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = x.rows;
        let backend = effective_backend(&self.device, n * p);

        if matches!(backend, Backend::Cpu) {
            let x_data = &x.data;
            let c_data = &coef.data;
            let mut preds = vec![0.0; n];
            if classes.len() == 2 {
                let w0 = &c_data[..p];
                let b0 = intercept[0];
                let c0 = classes[0];
                let c1 = classes[1];
                let compute_row = |(r, dst): (usize, &mut f64)| {
                    let x_row = &x_data[r * p..(r + 1) * p];
                    *dst = if dot4(x_row, w0) + b0 > 0.0 { c1 } else { c0 };
                };
                if n >= PARALLEL_ROW_THRESHOLD {
                    preds.par_iter_mut().enumerate().for_each(compute_row);
                } else {
                    preds.iter_mut().enumerate().for_each(compute_row);
                }
            } else {
                let k = classes.len();
                let compute_row = |(r, dst): (usize, &mut f64)| {
                    let x_row = &x_data[r * p..(r + 1) * p];
                    let mut best_c = 0;
                    let mut best_v = dot4(x_row, &c_data[..p]) + intercept[0];
                    for c in 1..k {
                        let v = dot4(x_row, &c_data[c * p..(c + 1) * p]) + intercept[c];
                        if v > best_v {
                            best_v = v;
                            best_c = c;
                        }
                    }
                    *dst = classes[best_c];
                };
                if n >= PARALLEL_ROW_THRESHOLD {
                    preds.par_iter_mut().enumerate().for_each(compute_row);
                } else {
                    preds.iter_mut().enumerate().for_each(compute_row);
                }
            }
            return Ok(preds);
        }

        let scores = self.decision_function(x)?;
        if classes.len() == 2 {
            Ok((0..n)
                .map(|i| {
                    if scores.data[i] > 0.0 {
                        classes[1]
                    } else {
                        classes[0]
                    }
                })
                .collect())
        } else {
            let k = classes.len();
            let mut preds = Vec::with_capacity(n);
            for i in 0..n {
                let row = &scores.data[(i * k)..((i + 1) * k)];
                let mut best_c = 0;
                let mut best_v = row[0];
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best_v {
                        best_v = v;
                        best_c = c;
                    }
                }
                preds.push(classes[best_c]);
            }
            Ok(preds)
        }
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.validate_predict_input(x)?;
        if self.solver == "lsqr" {
            return Err(OchreError::InvalidParameter(
                "transform is not implemented for solver='lsqr'; use 'svd' or 'eigen' (・`ω´・)"
                    .to_string(),
            ));
        }
        let scalings = self.scalings_.as_ref().ok_or(OchreError::NotFitted)?;
        let xbar = self.xbar_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_comp = self.n_components_.ok_or(OchreError::NotFitted)?;
        let p = x.cols;
        let n = x.rows;
        let cols_to_take = n_comp.min(scalings.cols);
        let backend = effective_backend(&self.device, n * p);

        if matches!(backend, Backend::Cpu) {
            // Transpose first cols_to_take columns of scalings into contiguous rows (cols_to_take x p)
            let mut s_t = vec![0.0; cols_to_take * p];
            for c in 0..cols_to_take {
                for r in 0..p {
                    s_t[c * p + r] = scalings.data[r * scalings.cols + c];
                }
            }
            let offset: Vec<f64> = if self.solver == "eigen" {
                vec![0.0; cols_to_take]
            } else {
                (0..cols_to_take)
                    .map(|c| dot4(xbar, &s_t[c * p..(c + 1) * p]))
                    .collect()
            };
            let mut out = vec![0.0; n * cols_to_take];
            let x_data = &x.data;
            let compute_row = |(r, dst): (usize, &mut [f64])| {
                let x_row = &x_data[r * p..(r + 1) * p];
                for c in 0..cols_to_take {
                    dst[c] = dot4(x_row, &s_t[c * p..(c + 1) * p]) - offset[c];
                }
            };
            if n >= PARALLEL_ROW_THRESHOLD {
                out.par_chunks_exact_mut(cols_to_take)
                    .enumerate()
                    .for_each(compute_row);
            } else {
                out.chunks_exact_mut(cols_to_take)
                    .enumerate()
                    .for_each(compute_row);
            }
            return Matrix::from_vec(n, cols_to_take, out);
        }

        // Slice first n_comp columns of scalings (p x n_comp)
        let mut s_sub_data = Vec::with_capacity(p * cols_to_take);
        for r in 0..p {
            for c in 0..cols_to_take {
                s_sub_data.push(scalings.data[r * scalings.cols + c]);
            }
        }
        let s_sub = Matrix::from_vec(p, cols_to_take, s_sub_data)?;

        let x_in = if self.solver == "eigen" {
            x.clone()
        } else {
            let mut x_centered_data = vec![0.0; n * p];
            for r in 0..n {
                for c in 0..p {
                    x_centered_data[r * p + c] = x.data[r * p + c] - xbar[c];
                }
            }
            Matrix::from_vec(n, p, x_centered_data)?
        };
        backend.matmul(&x_in, &s_sub)
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LDA);
        write_str(&mut buf, &self.solver);
        write_opt_f64(&mut buf, self.shrinkage);
        write_bool(&mut buf, self.shrinkage_auto);
        write_opt_f64_vec(&mut buf, &self.priors);
        write_opt_usize(&mut buf, self.n_components);
        write_f64(&mut buf, self.tol);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.priors_);
        write_opt_matrix(&mut buf, &self.means_);
        write_opt_f64_vec(&mut buf, &self.xbar_);
        write_opt_matrix(&mut buf, &self.coef_);
        write_opt_f64_vec(&mut buf, &self.intercept_);
        write_opt_matrix(&mut buf, &self.scalings_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_ratio_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_components_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LDA)?;
        let solver = r.read_str()?;
        let shrinkage = r.read_opt_f64()?;
        let shrinkage_auto = r.read_bool()?;
        let priors = r.read_opt_f64_vec()?;
        let n_components = r.read_opt_usize()?;
        let tol = r.read_f64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let priors_ = r.read_opt_f64_vec()?;
        let means_ = r.read_opt_matrix()?;
        let xbar_ = r.read_opt_f64_vec()?;
        let coef_ = r.read_opt_matrix()?;
        let intercept_ = r.read_opt_f64_vec()?;
        let scalings_ = r.read_opt_matrix()?;
        let explained_variance_ratio_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_components_ = r.read_opt_usize()?;

        let is_fitted = n_features_in_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            solver,
            shrinkage,
            shrinkage_auto,
            priors,
            n_components,
            tol,
            device: dev,
            device_,
            classes_,
            priors_,
            means_,
            xbar_,
            coef_,
            intercept_,
            scalings_,
            explained_variance_ratio_,
            n_features_in_,
            n_components_,
        })
    }
}

/// Quadratic Discriminant Analysis (QDA) classifier (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    pub priors: Option<Vec<f64>>,
    pub reg_param: f64,
    pub tol: f64,
    pub store_covariance: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<f64>>,
    pub priors_: Option<Vec<f64>>,
    pub means_: Option<Matrix>,
    pub scalings_: Option<Vec<Vec<f64>>>,
    pub rotations_: Option<Vec<Matrix>>,
    pub covariance_: Option<Vec<Matrix>>,
    pub n_features_in_: Option<usize>,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new(
        priors: Option<Vec<f64>>,
        reg_param: f64,
        tol: f64,
        store_covariance: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !reg_param.is_finite() || !(0.0..=1.0).contains(&reg_param) {
            return Err(OchreError::InvalidParameter(format!(
                "reg_param ({reg_param}) must be a finite float in [0.0, 1.0] (・`ω´・)"
            )));
        }
        validate_tol(tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            priors,
            reg_param,
            tol,
            store_covariance,
            device: dev,
            device_: None,
            classes_: None,
            priors_: None,
            means_: None,
            scalings_: None,
            rotations_: None,
            covariance_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.n_features_in_.is_some() || self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    fn reset_fitted(&mut self) {
        self.classes_ = None;
        self.priors_ = None;
        self.means_ = None;
        self.scalings_ = None;
        self.rotations_ = None;
        self.covariance_ = None;
        self.n_features_in_ = None;
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.reset_fitted();
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;

        let n_features = x.cols;
        let classes = extract_sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(OchreError::InsufficientClasses(n_classes));
        }

        let y_idx = map_labels_to_indices(y, &classes)?;
        let (counts, priors, means, _xbar) =
            compute_class_stats(x, &y_idx, n_classes, self.priors.as_deref())?;

        for (k, &cnt) in counts.iter().enumerate() {
            if cnt < 2 {
                return Err(OchreError::InvalidParameter(format!(
                    "y has a class ({}) with fewer than 2 samples (got {cnt}) ( >_< )",
                    classes[k]
                )));
            }
        }

        let backend = effective_backend(&self.device, x.rows * n_features);
        let mut scalings_list = Vec::with_capacity(n_classes);
        let mut rotations_list = Vec::with_capacity(n_classes);
        let mut cov_list = if self.store_covariance {
            Some(Vec::with_capacity(n_classes))
        } else {
            None
        };

        for (k, &nk) in counts.iter().enumerate() {
            let mut xk_data = Vec::with_capacity(nk * n_features);
            let m_row = &means.data[(k * n_features)..((k + 1) * n_features)];
            for (i, &cls) in y_idx.iter().enumerate() {
                if cls == k {
                    let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
                    for j in 0..n_features {
                        xk_data.push(x_row[j] - m_row[j]);
                    }
                }
            }
            let xk = Matrix::from_vec(nk, n_features, xk_data)?;
            let dummy_y = vec![0.0; nk];
            let (mut cov_k, _) = backend.compute_normal_equation(&xk, &dummy_y, false)?;

            let inv_n = 1.0 / (nk as f64);
            let one_minus = 1.0 - self.reg_param;
            for r in 0..n_features {
                for c in 0..n_features {
                    let idx = r * n_features + c;
                    cov_k.data[idx] = one_minus * (cov_k.data[idx] * inv_n);
                    if r == c {
                        cov_k.data[idx] += self.reg_param;
                    }
                }
            }

            if let Some(ref mut covs) = cov_list {
                covs.push(cov_k.clone());
            }

            let eig = backend.small_sym_eig_svd(&cov_k)?;
            let floor = self.tol.max(1e-12);
            let s2: Vec<f64> = eig.eigenvalues.iter().map(|&v| v.max(floor)).collect();
            scalings_list.push(s2);
            rotations_list.push(eig.eigenvectors);
        }

        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
        self.means_ = Some(means);
        self.scalings_ = Some(scalings_list);
        self.rotations_ = Some(rotations_list);
        self.covariance_ = cov_list;
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn validate_predict_input(&self, x: &Matrix) -> OchreResult<usize> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        Ok(expected)
    }

    fn decision_scores_all(&self, x: &Matrix) -> OchreResult<Matrix> {
        let p = self.validate_predict_input(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let priors = self.priors_.as_ref().ok_or(OchreError::NotFitted)?;
        let means = self.means_.as_ref().ok_or(OchreError::NotFitted)?;
        let scalings = self.scalings_.as_ref().ok_or(OchreError::NotFitted)?;
        let rotations = self.rotations_.as_ref().ok_or(OchreError::NotFitted)?;

        let n_classes = classes.len();
        let n_samples = x.rows;
        let backend = effective_backend(&self.device, n_samples * p);

        if matches!(backend, Backend::Cpu) {
            // Precompute per-class transposed rotations, projected means, inv_s2, and constants
            let mut rot_ts = Vec::with_capacity(n_classes);
            let mut proj_means = Vec::with_capacity(n_classes);
            let mut inv_s2s = Vec::with_capacity(n_classes);
            let mut class_consts = Vec::with_capacity(n_classes);

            for k in 0..n_classes {
                let m_row = &means.data[(k * p)..((k + 1) * p)];
                let rot = &rotations[k];
                let mut rot_t = vec![0.0; p * p];
                for c in 0..p {
                    for r in 0..p {
                        rot_t[c * p + r] = rot.data[r * p + c];
                    }
                }
                let mut pm = Vec::with_capacity(p);
                for c in 0..p {
                    pm.push(dot4(m_row, &rot_t[c * p..(c + 1) * p]));
                }
                let s2 = &scalings[k];
                let inv_s2: Vec<f64> = s2.iter().map(|&v| 1.0 / v).collect();
                let logdet: f64 = s2.iter().map(|&v| v.ln()).sum();
                let log_prior = if priors[k] > 0.0 {
                    priors[k].ln()
                } else {
                    f64::NEG_INFINITY
                };
                rot_ts.push(rot_t);
                proj_means.push(pm);
                inv_s2s.push(inv_s2);
                class_consts.push(-0.5 * logdet + log_prior);
            }

            let mut scores = vec![0.0; n_samples * n_classes];
            let x_data = &x.data;
            let compute_row = |(r, dst): (usize, &mut [f64])| {
                let x_row = &x_data[(r * p)..((r + 1) * p)];
                for k in 0..n_classes {
                    let rot_t = &rot_ts[k];
                    let pm = &proj_means[k];
                    let inv_s2 = &inv_s2s[k];
                    let mut norm2 = 0.0;
                    for c in 0..p {
                        let proj_c = dot4(x_row, &rot_t[c * p..(c + 1) * p]) - pm[c];
                        norm2 += proj_c * proj_c * inv_s2[c];
                    }
                    dst[k] = class_consts[k] - 0.5 * norm2;
                }
            };
            if n_samples >= PARALLEL_ROW_THRESHOLD {
                scores
                    .par_chunks_exact_mut(n_classes)
                    .enumerate()
                    .for_each(compute_row);
            } else {
                scores
                    .chunks_exact_mut(n_classes)
                    .enumerate()
                    .for_each(compute_row);
            }
            return Matrix::from_vec(n_samples, n_classes, scores);
        }

        let mut scores = vec![0.0; n_samples * n_classes];

        for k in 0..n_classes {
            let m_row = &means.data[(k * p)..((k + 1) * p)];
            let mut xc_data = vec![0.0; n_samples * p];
            for r in 0..n_samples {
                let x_row = &x.data[(r * p)..((r + 1) * p)];
                let out_row = &mut xc_data[(r * p)..((r + 1) * p)];
                for j in 0..p {
                    out_row[j] = x_row[j] - m_row[j];
                }
            }
            let xc = Matrix::from_vec(n_samples, p, xc_data)?;
            let proj = backend.matmul(&xc, &rotations[k])?;
            let s2 = &scalings[k];
            let logdet: f64 = s2.iter().map(|&v| v.ln()).sum();
            let log_prior = if priors[k] > 0.0 {
                priors[k].ln()
            } else {
                f64::NEG_INFINITY
            };

            for r in 0..n_samples {
                let p_row = &proj.data[(r * p)..((r + 1) * p)];
                let mut norm2 = 0.0;
                for j in 0..p {
                    norm2 += (p_row[j] * p_row[j]) / s2[j];
                }
                scores[r * n_classes + k] = -0.5 * (norm2 + logdet) + log_prior;
            }
        }

        Matrix::from_vec(n_samples, n_classes, scores)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            let n = all_scores.rows;
            let mut diff = Vec::with_capacity(n);
            for i in 0..n {
                diff.push(all_scores.data[2 * i + 1] - all_scores.data[2 * i]);
            }
            Matrix::from_vec(n, 1, diff)
        } else {
            Ok(all_scores)
        }
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        log_softmax_matrix(&all_scores)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        softmax_matrix(&all_scores)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let all_scores = self.decision_scores_all(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = all_scores.rows;
        let k = classes.len();
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let row = &all_scores.data[(i * k)..((i + 1) * k)];
            let mut best_c = 0;
            let mut best_v = row[0];
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            preds.push(classes[best_c]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_QDA);
        write_opt_f64_vec(&mut buf, &self.priors);
        write_f64(&mut buf, self.reg_param);
        write_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.store_covariance);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.priors_);
        write_opt_matrix(&mut buf, &self.means_);
        match &self.scalings_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for v in list {
                    write_f64_vec(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.rotations_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for m in list {
                    crate::dit_io::write_matrix(&mut buf, m);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.covariance_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for m in list {
                    crate::dit_io::write_matrix(&mut buf, m);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_QDA)?;
        let priors = r.read_opt_f64_vec()?;
        let reg_param = r.read_f64()?;
        let tol = r.read_f64()?;
        let store_covariance = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let priors_ = r.read_opt_f64_vec()?;
        let means_ = r.read_opt_matrix()?;

        let scalings_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_f64_vec()?);
            }
            Some(list)
        } else {
            None
        };

        let rotations_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_matrix()?);
            }
            Some(list)
        } else {
            None
        };

        let covariance_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_matrix()?);
            }
            Some(list)
        } else {
            None
        };

        let n_features_in_ = r.read_opt_usize()?;
        let is_fitted = n_features_in_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            priors,
            reg_param,
            tol,
            store_covariance,
            device: dev,
            device_,
            classes_,
            priors_,
            means_,
            scalings_,
            rotations_,
            covariance_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "LinearDiscriminantAnalysis")]
#[derive(Clone)]
pub struct PyLinearDiscriminantAnalysis {
    inner: LinearDiscriminantAnalysis,
}

crate::pymethods_with_dit!(PyLinearDiscriminantAnalysis, LinearDiscriminantAnalysis, {
    #[new]
    #[pyo3(signature = (
        solver = "svd".to_string(),
        shrinkage = None,
        shrinkage_auto = false,
        priors = None,
        n_components = None,
        tol = 1e-4,
        device = None
    ))]
    pub fn new(
        solver: String,
        shrinkage: Option<f64>,
        shrinkage_auto: bool,
        priors: Option<Vec<f64>>,
        n_components: Option<usize>,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LinearDiscriminantAnalysis::new(
                solver,
                shrinkage,
                shrinkage_auto,
                priors,
                n_components,
                tol,
                device,
            )?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
        Ok(())
    }

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.decision_function(&x_mat)?;
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

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
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

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let out = self.inner.fit_transform(&x_mat, &y_vec)?;
        matrix_to_pybytes(py, &out)
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
    pub fn solver(&self) -> String {
        self.inner.solver.clone()
    }

    #[getter]
    pub fn shrinkage(&self) -> Option<f64> {
        self.inner.shrinkage
    }

    #[getter]
    pub fn shrinkage_auto(&self) -> bool {
        self.inner.shrinkage_auto
    }

    #[getter]
    pub fn n_components(&self) -> Option<usize> {
        self.inner.n_components
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn priors_(&self) -> Option<Vec<f64>> {
        self.inner.priors_.clone()
    }

    #[getter]
    pub fn xbar_(&self) -> Option<Vec<f64>> {
        self.inner.xbar_.clone()
    }

    #[getter]
    pub fn intercept_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_.clone()
    }

    #[getter]
    pub fn explained_variance_ratio_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_ratio_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_components_(&self) -> Option<usize> {
        self.inner.n_components_
    }

    pub fn means_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.means_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    pub fn coef_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    pub fn scalings_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.scalings_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }
});

#[pyclass(from_py_object, name = "QuadraticDiscriminantAnalysis")]
#[derive(Clone)]
pub struct PyQuadraticDiscriminantAnalysis {
    inner: QuadraticDiscriminantAnalysis,
}

crate::pymethods_with_dit!(
    PyQuadraticDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    {
        #[new]
        #[pyo3(signature = (
            priors = None,
            reg_param = 0.0,
            tol = 1e-4,
            store_covariance = false,
            device = None
        ))]
        pub fn new(
            priors: Option<Vec<f64>>,
            reg_param: f64,
            tol: f64,
            store_covariance: bool,
            device: Option<String>,
        ) -> PyResult<Self> {
            Ok(Self {
                inner: QuadraticDiscriminantAnalysis::new(
                    priors,
                    reg_param,
                    tol,
                    store_covariance,
                    device,
                )?,
            })
        }

        pub fn fit_buffer(
            &mut self,
            py: Python<'_>,
            x: PyBuffer<f64>,
            y: PyBuffer<f64>,
        ) -> PyResult<()> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let y_vec = buffer_to_f64_1d(py, &y)?;
            self.inner.fit(&x_mat, &y_vec)?;
            Ok(())
        }

        pub fn decision_function_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let out = self.inner.decision_function(&x_mat)?;
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

        pub fn predict_log_proba_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let out = self.inner.predict_log_proba(&x_mat)?;
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
        pub fn reg_param(&self) -> f64 {
            self.inner.reg_param
        }

        #[getter]
        pub fn tol(&self) -> f64 {
            self.inner.tol
        }

        #[getter]
        pub fn store_covariance(&self) -> bool {
            self.inner.store_covariance
        }

        #[getter]
        pub fn classes_(&self) -> Option<Vec<f64>> {
            self.inner.classes_.clone()
        }

        #[getter]
        pub fn priors_(&self) -> Option<Vec<f64>> {
            self.inner.priors_.clone()
        }

        #[getter]
        pub fn scalings_(&self) -> Option<Vec<Vec<f64>>> {
            self.inner.scalings_.clone()
        }

        #[getter]
        pub fn n_features_in_(&self) -> Option<usize> {
            self.inner.n_features_in_
        }

        pub fn means_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
            match &self.inner.means_ {
                Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
                None => Ok(None),
            }
        }

        #[allow(clippy::type_complexity)]
        pub fn rotations_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
            match &self.inner.rotations_ {
                Some(list) => {
                    let mut out = Vec::with_capacity(list.len());
                    for m in list {
                        out.push(matrix_to_pybytes(py, m)?);
                    }
                    Ok(Some(out))
                }
                None => Ok(None),
            }
        }

        #[allow(clippy::type_complexity)]
        pub fn covariance_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
            match &self.inner.covariance_ {
                Some(list) => {
                    let mut out = Vec::with_capacity(list.len());
                    for m in list {
                        out.push(matrix_to_pybytes(py, m)?);
                    }
                    Ok(Some(out))
                }
                None => Ok(None),
            }
        }
    }
);

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearDiscriminantAnalysis>()?;
    m.add_class::<PyQuadraticDiscriminantAnalysis>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda_binary_and_multiclass_solvers() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.2],
            vec![0.2, 0.0],
            vec![-0.1, -0.1],
            vec![0.1, 0.1],
            vec![5.0, 5.2],
            vec![5.2, 5.0],
            vec![4.9, 4.9],
            vec![5.1, 5.1],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        for solver in ["svd", "lsqr", "eigen"] {
            let mut lda = LinearDiscriminantAnalysis::new(
                solver.to_string(),
                None,
                false,
                None,
                None,
                1e-4,
                Some("cpu".to_string()),
            )
            .unwrap();
            lda.fit(&x, &y).unwrap();
            assert_eq!(lda.score(&x, &y).unwrap(), 1.0);
            let probs = lda.predict_proba(&x).unwrap();
            assert_eq!(probs.rows, 8);
            assert_eq!(probs.cols, 2);
            for r in 0..8 {
                let sum = probs.data[2 * r] + probs.data[2 * r + 1];
                assert!((sum - 1.0).abs() < 1e-12);
            }
            let bytes = lda.to_bytes();
            let loaded = LinearDiscriminantAnalysis::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), y);
        }
    }

    #[test]
    fn test_qda_fit_predict_and_dit() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![-0.2, 0.1],
            vec![0.2, -0.1],
            vec![4.0, 4.0],
            vec![4.2, 3.8],
            vec![3.9, 4.1],
            vec![4.1, 4.2],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut qda =
            QuadraticDiscriminantAnalysis::new(None, 0.1, 1e-4, true, Some("cpu".to_string()))
                .unwrap();
        qda.fit(&x, &y).unwrap();
        assert_eq!(qda.score(&x, &y).unwrap(), 1.0);
        assert!(qda.covariance_.is_some());
        let bytes = qda.to_bytes();
        let loaded = QuadraticDiscriminantAnalysis::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), y);
    }
}
