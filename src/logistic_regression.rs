use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{dot_unrolled, predict_linear_1d, predict_linear_multi};
use crate::matrix::Matrix;
use crate::primitives::ElementwiseOp;
use rayon::prelude::*;

pub const TAG_LOGISTIC_REGRESSION: u8 = 38;
const LBFGS_HISTORY: usize = 10;
const SMALL_MATRIX_GPU_THRESHOLD: usize = 262_144;

fn validate_c(c: f64) -> OchreResult<()> {
    if c <= 0.0 || !c.is_finite() {
        return Err(OchreError::InvalidC(c));
    }
    Ok(())
}

fn validate_max_iter(max_iter: usize) -> OchreResult<()> {
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_tol(tol: f64) -> OchreResult<()> {
    if tol < 0.0 || !tol.is_finite() {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be non-negative and finite"
        )));
    }
    Ok(())
}

pub(crate) fn validate_sample_weights(
    sample_weight: Option<&[f64]>,
    n_samples: usize,
) -> OchreResult<()> {
    if let Some(sw) = sample_weight {
        if sw.len() != n_samples {
            return Err(OchreError::WeightLengthMismatch {
                weights: sw.len(),
                samples: n_samples,
            });
        }
        let mut sum = 0.0;
        for &w in sw {
            if w < 0.0 || !w.is_finite() {
                return Err(OchreError::InvalidWeights);
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
    }
    Ok(())
}

pub(crate) fn extract_sorted_classes(y: &[f64]) -> Vec<f64> {
    let mut classes = y.to_vec();
    classes.sort_by(|a, b| a.total_cmp(b));
    classes.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    classes
}

pub(crate) fn map_labels_to_indices(y: &[f64], classes: &[f64]) -> OchreResult<Vec<usize>> {
    let mut indices = Vec::with_capacity(y.len());
    for &val in y {
        let pos = classes
            .iter()
            .position(|&c| (c - val).abs() < 1e-12)
            .ok_or_else(|| {
                OchreError::InvalidParameter(format!("Unknown class label {val} encountered"))
            })?;
        indices.push(pos);
    }
    Ok(indices)
}

pub(crate) fn build_effective_weights(
    y_idx: &[usize],
    n_classes: usize,
    classes: &[f64],
    sample_weight: Option<&[f64]>,
    class_weight_mode: Option<&str>,
    class_weight_map: Option<&[(f64, f64)]>,
) -> OchreResult<Vec<f64>> {
    let n = y_idx.len();
    let mut weights = match sample_weight {
        Some(sw) => sw.to_vec(),
        None => vec![1.0; n],
    };

    if let Some(mode) = class_weight_mode {
        if mode != "balanced" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid class_weight '{mode}'. Expected None, 'balanced', or a dict"
            )));
        }
        let mut class_counts = vec![0.0; n_classes];
        let mut total = 0.0;
        for (&idx, &w) in y_idx.iter().zip(weights.iter()) {
            class_counts[idx] += w;
            total += w;
        }
        for i in 0..n {
            let cnt = class_counts[y_idx[i]];
            if cnt > 0.0 {
                weights[i] *= total / (n_classes as f64 * cnt);
            }
        }
    } else if let Some(cw_map) = class_weight_map {
        for i in 0..n {
            let cls_val = classes[y_idx[i]];
            if let Some((_, cw)) = cw_map.iter().find(|&&(k, _)| (k - cls_val).abs() < 1e-12) {
                if *cw < 0.0 || !cw.is_finite() {
                    return Err(OchreError::InvalidWeights);
                }
                weights[i] *= *cw;
            }
        }
    }

    Ok(weights)
}

#[inline]
fn stable_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[inline(always)]
fn sigmoid_and_softplus(z: f64) -> (f64, f64) {
    if z >= 0.0 {
        let e = (-z).exp();
        (1.0 / (1.0 + e), z + e.ln_1p())
    } else {
        let e = z.exp();
        (e / (1.0 + e), e.ln_1p())
    }
}

#[inline]
fn softplus(z: f64) -> f64 {
    if z >= 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_binary_loss_grad(
    backend: &Backend<'_>,
    x: &Matrix,
    y01: &[f64],
    weights: &[f64],
    theta: &[f64],
    fit_intercept: bool,
    inv_c: f64,
    grad_out: &mut [f64],
) -> OchreResult<f64> {
    let n = x.rows;
    let p = x.cols;
    let w = &theta[..p];
    let b = if fit_intercept { theta[p] } else { 0.0 };

    let mut data_loss = 0.0;
    let mut grad_b = 0.0;

    if matches!(backend, Backend::Cpu) {
        if n >= 2048 {
            const CHUNK_ROWS: usize = 32_768;
            let (par_loss, par_grad_w, par_grad_b) = x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .zip(y01.par_chunks(CHUNK_ROWS))
                .zip(weights.par_chunks(CHUNK_ROWS))
                .fold(
                    || (0.0, vec![0.0; p], 0.0),
                    |(mut loc_loss, mut loc_grad_w, mut loc_grad_b),
                     ((x_chunk, y_chunk), w_chunk)| {
                        let rows = y_chunk.len();
                        for i in 0..rows {
                            let row = &x_chunk[i * p..(i + 1) * p];
                            let z = dot_unrolled(row, w) + b;
                            let (prob, sp) = sigmoid_and_softplus(z);
                            let wi = w_chunk[i];
                            let yi = y_chunk[i];
                            loc_loss += wi * (sp - yi * z);
                            let r = wi * (prob - yi);
                            loc_grad_b += r;
                            for j in 0..p {
                                loc_grad_w[j] += r * row[j];
                            }
                        }
                        (loc_loss, loc_grad_w, loc_grad_b)
                    },
                )
                .reduce(
                    || (0.0, vec![0.0; p], 0.0),
                    |(l1, mut gw1, gb1), (l2, gw2, gb2)| {
                        for j in 0..p {
                            gw1[j] += gw2[j];
                        }
                        (l1 + l2, gw1, gb1 + gb2)
                    },
                );

            data_loss = par_loss;
            grad_b = par_grad_b;
            grad_out[..p].copy_from_slice(&par_grad_w);
        } else {
            grad_out[..p].fill(0.0);
            for i in 0..n {
                let row = &x.data[i * p..(i + 1) * p];
                let z = dot_unrolled(row, w) + b;
                let (prob, sp) = sigmoid_and_softplus(z);
                let wi = weights[i];
                let yi = y01[i];
                data_loss += wi * (sp - yi * z);
                let r = wi * (prob - yi);
                grad_b += r;
                for j in 0..p {
                    grad_out[j] += r * row[j];
                }
            }
        }
        let mut reg_loss = 0.0;
        for j in 0..p {
            let wj = w[j];
            reg_loss += wj * wj;
            grad_out[j] += inv_c * wj;
        }
        if fit_intercept {
            grad_out[p] = grad_b;
        }
        return Ok(data_loss + 0.5 * inv_c * reg_loss);
    }

    let mut logits = backend.matvec(x, w)?;
    if b != 0.0 {
        for z in &mut logits {
            *z += b;
        }
    }

    let probs = if n >= 65_536 {
        let z_mat = Matrix::from_vec(n, 1, logits.clone())?;
        backend.elementwise_op(&z_mat, ElementwiseOp::Sigmoid)?.data
    } else {
        logits.iter().map(|&z| stable_sigmoid(z)).collect()
    };

    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        let wi = weights[i];
        let z = logits[i];
        let yi = y01[i];
        data_loss += wi * (softplus(z) - yi * z);
        let r = wi * (probs[i] - yi);
        residuals.push(r);
        grad_b += r;
    }

    let r_mat = Matrix::from_vec(n, 1, residuals)?;
    let grad_w_mat = backend.matmul_at_b(x, &r_mat)?;

    let mut reg_loss = 0.0;
    for j in 0..p {
        let wj = w[j];
        reg_loss += wj * wj;
        grad_out[j] = grad_w_mat.data[j] + inv_c * wj;
    }
    if fit_intercept {
        grad_out[p] = grad_b;
    }

    Ok(data_loss + 0.5 * inv_c * reg_loss)
}

#[allow(clippy::too_many_arguments)]
fn eval_multinomial_loss_grad(
    backend: &Backend<'_>,
    x: &Matrix,
    y_idx: &[usize],
    weights: &[f64],
    n_classes: usize,
    theta: &[f64],
    fit_intercept: bool,
    inv_c: f64,
    grad_out: &mut [f64],
) -> OchreResult<f64> {
    let n = x.rows;
    let p = x.cols;
    let w_len = n_classes * p;

    if matches!(backend, Backend::Cpu) && n >= 2048 {
        const CHUNK_ROWS: usize = 1024;
        let b_slice = if fit_intercept {
            &theta[w_len..w_len + n_classes]
        } else {
            &[]
        };
        let (par_loss, par_gw, par_gb) = x
            .data
            .par_chunks(CHUNK_ROWS * p)
            .zip(y_idx.par_chunks(CHUNK_ROWS))
            .zip(weights.par_chunks(CHUNK_ROWS))
            .map(|((x_chunk, y_chunk), w_chunk)| {
                let mut loc_loss = 0.0;
                let mut loc_gw = vec![0.0; w_len];
                let mut loc_gb = vec![0.0; n_classes];
                let mut logits = vec![0.0; n_classes];
                for i in 0..y_chunk.len() {
                    let row = &x_chunk[i * p..(i + 1) * p];
                    let mut max_z = f64::NEG_INFINITY;
                    for c in 0..n_classes {
                        let w_row = &theta[c * p..(c + 1) * p];
                        let mut z = dot_unrolled(row, w_row);
                        if fit_intercept {
                            z += b_slice[c];
                        }
                        logits[c] = z;
                        if z > max_z {
                            max_z = z;
                        }
                    }
                    let mut sum_exp = 0.0;
                    for c in 0..n_classes {
                        let e = (logits[c] - max_z).exp();
                        logits[c] = e;
                        sum_exp += e;
                    }
                    let wi = w_chunk[i];
                    let yi = y_chunk[i];
                    let inv_sum = 1.0 / sum_exp;
                    let p_true = (logits[yi] * inv_sum).max(1e-15);
                    loc_loss -= wi * p_true.ln();

                    for c in 0..n_classes {
                        let pc = logits[c] * inv_sum;
                        let target = if c == yi { 1.0 } else { 0.0 };
                        let r = wi * (pc - target);
                        loc_gb[c] += r;
                        let gw_row = &mut loc_gw[c * p..(c + 1) * p];
                        for j in 0..p {
                            gw_row[j] += r * row[j];
                        }
                    }
                }
                (loc_loss, loc_gw, loc_gb)
            })
            .reduce(
                || (0.0, vec![0.0; w_len], vec![0.0; n_classes]),
                |(l1, mut gw1, mut gb1), (l2, gw2, gb2)| {
                    for idx in 0..w_len {
                        gw1[idx] += gw2[idx];
                    }
                    for c in 0..n_classes {
                        gb1[c] += gb2[c];
                    }
                    (l1 + l2, gw1, gb1)
                },
            );

        let mut reg_loss = 0.0;
        for idx in 0..w_len {
            let wv = theta[idx];
            reg_loss += wv * wv;
            grad_out[idx] = par_gw[idx] + inv_c * wv;
        }
        if fit_intercept {
            grad_out[w_len..w_len + n_classes].copy_from_slice(&par_gb);
        }
        return Ok(par_loss + 0.5 * inv_c * reg_loss);
    }

    // Build W^T of shape (p, n_classes) for X * W^T
    let mut wt_data = vec![0.0; p * n_classes];
    for c in 0..n_classes {
        let w_row = &theta[c * p..(c + 1) * p];
        for j in 0..p {
            wt_data[j * n_classes + c] = w_row[j];
        }
    }
    let wt_mat = Matrix::from_vec(p, n_classes, wt_data)?;
    let mut logits = backend.matmul(x, &wt_mat)?;

    if fit_intercept {
        let b = &theta[w_len..w_len + n_classes];
        for r in 0..n {
            let row = &mut logits.data[r * n_classes..(r + 1) * n_classes];
            for c in 0..n_classes {
                row[c] += b[c];
            }
        }
    }

    let probs = backend.row_softmax(&logits)?;
    let mut data_loss = 0.0;
    let mut residuals = vec![0.0; n * n_classes];
    let mut grad_b = vec![0.0; n_classes];

    for i in 0..n {
        let wi = weights[i];
        let yi = y_idx[i];
        let p_row = &probs.data[i * n_classes..(i + 1) * n_classes];
        let r_row = &mut residuals[i * n_classes..(i + 1) * n_classes];

        let p_true = p_row[yi].max(1e-15);
        data_loss -= wi * p_true.ln();

        for c in 0..n_classes {
            let target = if c == yi { 1.0 } else { 0.0 };
            let r = wi * (p_row[c] - target);
            r_row[c] = r;
            grad_b[c] += r;
        }
    }

    // grad_W = R^T * X -> shape (n_classes, p)
    let r_mat = Matrix::from_vec(n, n_classes, residuals)?;
    let grad_w_mat = backend.matmul_at_b(&r_mat, x)?;

    let mut reg_loss = 0.0;
    for idx in 0..w_len {
        let wv = theta[idx];
        reg_loss += wv * wv;
        grad_out[idx] = grad_w_mat.data[idx] + inv_c * wv;
    }
    if fit_intercept {
        grad_out[w_len..w_len + n_classes].copy_from_slice(&grad_b);
    }

    Ok(data_loss + 0.5 * inv_c * reg_loss)
}

fn lbfgs_two_loop(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let dim = grad.len();
    let mut q = grad.to_vec();
    let k = s_hist.len();
    let mut alpha = vec![0.0; k];

    for i in (0..k).rev() {
        let a = rho_hist[i] * dot_unrolled(&s_hist[i], &q);
        alpha[i] = a;
        for d in 0..dim {
            q[d] -= a * y_hist[i][d];
        }
    }

    let gamma = if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
        let yy = dot_unrolled(y_last, y_last);
        if yy > 1e-20 {
            dot_unrolled(s_last, y_last) / yy
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut r: Vec<f64> = q.into_iter().map(|v| v * gamma).collect();
    for i in 0..k {
        let beta = rho_hist[i] * dot_unrolled(&y_hist[i], &r);
        let diff = alpha[i] - beta;
        for d in 0..dim {
            r[d] += diff * s_hist[i][d];
        }
    }

    for v in &mut r {
        *v = -*v;
    }
    r
}

fn run_lbfgs<F>(
    dim: usize,
    n_samples: usize,
    max_iter: usize,
    tol: f64,
    mut eval_fn: F,
) -> OchreResult<(Vec<f64>, usize)>
where
    F: FnMut(&[f64], &mut [f64]) -> OchreResult<f64>,
{
    let mut theta = vec![0.0; dim];
    let mut grad = vec![0.0; dim];
    let mut loss = eval_fn(&theta, &mut grad)?;

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_HISTORY);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_HISTORY);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(LBFGS_HISTORY);

    let mut next_theta = vec![0.0; dim];
    let mut next_grad = vec![0.0; dim];
    let mut n_iter = 0usize;
    let grad_tol = tol * (n_samples as f64).max(1.0);

    for iter in 0..max_iter {
        let g_inf = grad.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if g_inf <= grad_tol {
            break;
        }
        n_iter = iter + 1;

        let mut dir = lbfgs_two_loop(&grad, &s_hist, &y_hist, &rho_hist);
        let mut dg = dot_unrolled(&dir, &grad);
        if dg >= 0.0 {
            // Reset direction to steepest descent if curvature became non-descent
            for (d, &g) in dir.iter_mut().zip(grad.iter()) {
                *d = -g;
            }
            dg = -dot_unrolled(&grad, &grad);
        }

        let mut step = if s_hist.is_empty() {
            let g_norm = dot_unrolled(&grad, &grad).sqrt();
            1.0 / g_norm.max(1.0)
        } else {
            1.0
        };
        let c1 = 1e-4;
        let mut accepted = false;
        let mut next_loss = loss;

        for _ in 0..30 {
            for d in 0..dim {
                next_theta[d] = theta[d] + step * dir[d];
            }
            next_loss = eval_fn(&next_theta, &mut next_grad)?;
            if next_loss.is_finite() && next_loss <= loss + c1 * step * dg {
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if !accepted {
            break;
        }

        let mut s_vec = vec![0.0; dim];
        let mut y_vec = vec![0.0; dim];
        for d in 0..dim {
            s_vec[d] = next_theta[d] - theta[d];
            y_vec[d] = next_grad[d] - grad[d];
        }
        let sy = dot_unrolled(&s_vec, &y_vec);

        let rel_change = (loss - next_loss).abs() / loss.abs().max(1.0);
        theta.copy_from_slice(&next_theta);
        grad.copy_from_slice(&next_grad);
        loss = next_loss;

        if sy > 1e-12 {
            if s_hist.len() == LBFGS_HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s_vec);
            y_hist.push(y_vec);
        }

        if rel_change < 1e-12 {
            break;
        }
    }

    Ok((theta, n_iter))
}

/// L2-regularized Logistic Regression classifier (binary & multinomial) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub c: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub class_weight_mode: Option<String>,
    pub class_weight_map: Option<Vec<(f64, f64)>>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub classes_: Vec<f64>,
    pub is_multiclass_: bool,
    pub n_iter_: usize,
    pub n_features_in_: Option<usize>,
}

impl LogisticRegression {
    /// Create a new LogisticRegression estimator.
    pub fn new(
        c: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        class_weight_mode: Option<String>,
        class_weight_map: Option<Vec<(f64, f64)>>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_c(c)?;
        validate_max_iter(max_iter)?;
        validate_tol(tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            c,
            fit_intercept,
            max_iter,
            tol,
            class_weight_mode,
            class_weight_map,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            classes_: Vec::new(),
            is_multiclass_: false,
            n_iter_: 0,
            n_features_in_: None,
        })
    }

    /// Fit logistic regression model on X and y with optional sample_weight.
    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.data.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
            return Err(OchreError::InvalidParameter(
                "Input X or y contains NaN or Inf values ( >_< )".to_string(),
            ));
        }
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weights(sample_weight, x.rows)?;

        let classes = extract_sorted_classes(y);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }

        let y_idx = map_labels_to_indices(y, &classes)?;
        let n_classes = classes.len();
        let weights = build_effective_weights(
            &y_idx,
            n_classes,
            &classes,
            sample_weight,
            self.class_weight_mode.as_deref(),
            self.class_weight_map.as_deref(),
        )?;

        let p = x.cols;
        let inv_c = 1.0 / self.c;
        // Avoid repeated PCIe/kernel-launch overhead per L-BFGS step on small or narrow matrices
        let backend = if x.rows.saturating_mul(p) < SMALL_MATRIX_GPU_THRESHOLD || p <= 64 {
            Backend::Cpu
        } else {
            get_backend(&self.device)
        };

        if n_classes == 2 {
            let y01: Vec<f64> = y_idx.iter().map(|&idx| idx as f64).collect();
            let dim = p + usize::from(self.fit_intercept);
            let fit_intercept = self.fit_intercept;
            let (theta, n_iter) = run_lbfgs(dim, x.rows, self.max_iter, self.tol, |th, gr| {
                eval_binary_loss_grad(&backend, x, &y01, &weights, th, fit_intercept, inv_c, gr)
            })?;

            let coef = theta[..p].to_vec();
            let intercept = if self.fit_intercept { theta[p] } else { 0.0 };
            self.coef_ = Some(coef.clone());
            self.intercept_ = Some(intercept);
            self.coef_multi_ = Some(Matrix::from_vec(1, p, coef)?);
            self.intercept_multi_ = Some(vec![intercept]);
            self.is_multiclass_ = false;
            self.n_iter_ = n_iter;
        } else {
            let w_len = n_classes * p;
            let dim = w_len + if self.fit_intercept { n_classes } else { 0 };
            let fit_intercept = self.fit_intercept;
            let (theta, n_iter) = run_lbfgs(dim, x.rows, self.max_iter, self.tol, |th, gr| {
                eval_multinomial_loss_grad(
                    &backend,
                    x,
                    &y_idx,
                    &weights,
                    n_classes,
                    th,
                    fit_intercept,
                    inv_c,
                    gr,
                )
            })?;

            let coef_mat = Matrix::from_vec(n_classes, p, theta[..w_len].to_vec())?;
            let intercepts = if self.fit_intercept {
                theta[w_len..w_len + n_classes].to_vec()
            } else {
                vec![0.0; n_classes]
            };
            self.coef_ = None;
            self.intercept_ = None;
            self.coef_multi_ = Some(coef_mat);
            self.intercept_multi_ = Some(intercepts);
            self.is_multiclass_ = true;
            self.n_iter_ = n_iter;
        }

        self.classes_ = classes;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Compute decision function for binary classification (shape: n_samples).
    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if self.is_multiclass_ {
            return Err(OchreError::InvalidParameter(
                "Use decision_function_multi for multiclass LogisticRegression".to_string(),
            ));
        }
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Compute decision function for multiclass classification (shape: n_samples x n_classes).
    pub fn decision_function_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept_vec = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercept_vec)
    }

    /// Predict class probabilities of shape (n_samples, n_classes).
    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        if self.classes_.is_empty() {
            return Err(OchreError::NotFitted);
        }
        if !self.is_multiclass_ {
            let scores = self.decision_function(x)?;
            let mut data = Vec::with_capacity(x.rows * 2);
            for z in scores {
                let p1 = stable_sigmoid(z);
                data.push(1.0 - p1);
                data.push(p1);
            }
            return Matrix::from_vec(x.rows, 2, data);
        }

        let logits = self.decision_function_multi(x)?;
        get_backend(&self.device).row_softmax(&logits)
    }

    /// Predict log-probabilities of shape (n_samples, n_classes).
    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let probs = self.predict_proba(x)?;
        let data = probs.data.into_iter().map(|p| p.max(1e-15).ln()).collect();
        Matrix::from_vec(probs.rows, probs.cols, data)
    }

    /// Predict class labels for samples in X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if self.classes_.is_empty() {
            return Err(OchreError::NotFitted);
        }
        if !self.is_multiclass_ {
            let scores = self.decision_function(x)?;
            let c0 = self.classes_[0];
            let c1 = self.classes_[1];
            return Ok(scores
                .into_iter()
                .map(|s| if s > 0.0 { c1 } else { c0 })
                .collect());
        }

        let scores = self.decision_function_multi(x)?;
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut best_idx = 0;
            let mut best_score = scores[(r, 0)];
            for c in 1..scores.cols {
                let s = scores[(r, c)];
                if s > best_score {
                    best_score = s;
                    best_idx = c;
                }
            }
            preds.push(self.classes_[best_idx]);
        }
        Ok(preds)
    }

    /// Compute classification accuracy with optional sample_weight.
    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weights(sample_weight, x.rows)?;
        let preds = self.predict(x)?;
        if preds.is_empty() {
            return Ok(0.0);
        }
        match sample_weight {
            Some(sw) => {
                let mut weighted_correct = 0.0;
                let mut total_weight = 0.0;
                for ((p, actual), &w) in preds.iter().zip(y.iter()).zip(sw.iter()) {
                    total_weight += w;
                    if (*p - *actual).abs() < 1e-9 {
                        weighted_correct += w;
                    }
                }
                Ok(weighted_correct / total_weight)
            }
            None => {
                let correct = preds
                    .iter()
                    .zip(y.iter())
                    .filter(|(p, actual)| (**p - **actual).abs() < 1e-9)
                    .count();
                Ok(correct as f64 / y.len() as f64)
            }
        }
    }

    /// Update target computation device.
    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    /// Serialize model to `.dit` binary bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LOGISTIC_REGRESSION);
        write_f64(&mut buf, self.c);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_opt_str(&mut buf, &self.class_weight_mode);
        match &self.class_weight_map {
            Some(pairs) => {
                write_u8_bool(&mut buf, true);
                write_usize(&mut buf, pairs.len());
                for &(k, v) in pairs {
                    write_f64(&mut buf, k);
                    write_f64(&mut buf, v);
                }
            }
            None => write_u8_bool(&mut buf, false),
        }
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        write_f64_vec(&mut buf, &self.classes_);
        write_bool(&mut buf, self.is_multiclass_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    /// Deserialize model from `.dit` binary bytes.
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LOGISTIC_REGRESSION)?;
        let c = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let class_weight_mode = r.read_opt_str()?;
        let has_map = r.read_bool()?;
        let class_weight_map = if has_map {
            let len = r.read_usize()?;
            let mut pairs = Vec::with_capacity(len);
            for _ in 0..len {
                pairs.push((r.read_f64()?, r.read_f64()?));
            }
            Some(pairs)
        } else {
            None
        };
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let classes_ = r.read_f64_vec()?;
        let is_multiclass_ = r.read_bool()?;
        let n_iter_ = r.read_usize()?;
        let n_features_in_ = r.read_opt_usize()?;

        let is_fitted = !classes_.is_empty();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            c,
            fit_intercept,
            max_iter,
            tol,
            class_weight_mode,
            class_weight_map,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            classes_,
            is_multiclass_,
            n_iter_,
            n_features_in_,
        })
    }
}

#[inline]
fn write_u8_bool(buf: &mut Vec<u8>, v: bool) {
    write_bool(buf, v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression_binary_and_dit_roundtrip() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                -2.0, -1.0, -1.5, -2.0, -1.0, -1.0, 1.0, 1.5, 2.0, 1.0, 2.5, 2.0,
            ],
        )
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut clf =
            LogisticRegression::new(1.0, true, 200, 1e-6, None, None, Some("cpu".to_string()))
                .unwrap();
        clf.fit_weighted(&x, &y, None).unwrap();

        assert!(!clf.is_multiclass_);
        assert_eq!(clf.classes_, vec![0.0, 1.0]);
        assert!(clf.score(&x, &y, None).unwrap() > 0.99);

        let probs = clf.predict_proba(&x).unwrap();
        assert_eq!(probs.rows, 6);
        assert_eq!(probs.cols, 2);
        for r in 0..6 {
            assert!((probs[(r, 0)] + probs[(r, 1)] - 1.0).abs() < 1e-9);
        }

        let bytes = clf.to_bytes();
        let loaded = LogisticRegression::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), clf.predict(&x).unwrap());
    }

    #[test]
    fn test_logistic_regression_multiclass_and_balanced_weights() {
        let x = Matrix::from_vec(
            9,
            2,
            vec![
                -3.0, 0.0, -2.5, 0.5, -2.8, -0.4, 0.0, 3.0, 0.3, 2.7, -0.2, 3.1, 3.0, -1.0, 2.8,
                -1.2, 3.2, -0.8,
            ],
        )
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let mut clf = LogisticRegression::new(
            10.0,
            true,
            300,
            1e-6,
            Some("balanced".to_string()),
            None,
            Some("cpu".to_string()),
        )
        .unwrap();
        clf.fit_weighted(&x, &y, None).unwrap();

        assert!(clf.is_multiclass_);
        assert_eq!(clf.classes_, vec![0.0, 1.0, 2.0]);
        assert!((clf.score(&x, &y, None).unwrap() - 1.0).abs() < 1e-9);

        let probs = clf.predict_proba(&x).unwrap();
        for r in 0..9 {
            let row_sum: f64 = (0..3).map(|c| probs[(r, c)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-9);
        }
    }
}
