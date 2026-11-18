// OchreML Sparse Linear Models: Lasso & ElasticNet (＾▽＾)
// Hybrid hardware-accelerated Coordinate Descent (via Backend normal equations) &
// FISTA (via Backend matmul/matvec + N2 SoftThreshold), supporting 1D, multi-output,
// positive constraints, and sample_weight.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64,
    write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize, write_usize,
    write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{dot_unrolled, predict_linear_1d, predict_linear_multi};
use crate::logistic_regression::validate_sample_weights;
use crate::matrix::{Matrix, r2_score, r2_score_multi};
use crate::primitives::ElementwiseOp;
use crate::primitives::stats_ops::apply_scalar_op;

pub const TAG_LASSO: u8 = 39;
pub const TAG_ELASTIC_NET: u8 = 40;
const GRAM_CD_THRESHOLD: usize = 2048;

fn validate_sparse_params(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64) -> OchreResult<()> {
    if alpha < 0.0 || !alpha.is_finite() {
        return Err(OchreError::InvalidAlpha(alpha));
    }
    if !(0.0..=1.0).contains(&l1_ratio) || !l1_ratio.is_finite() {
        return Err(OchreError::InvalidL1Ratio(l1_ratio));
    }
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1".to_string(),
        ));
    }
    if tol < 0.0 || !tol.is_finite() {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be non-negative and finite"
        )));
    }
    Ok(())
}

struct CenteredData {
    x_c: Matrix,
    y_c: Matrix,
    x_mean: Vec<f64>,
    y_mean: Vec<f64>,
    total_w: f64,
}

fn prepare_centered_data(
    backend: &Backend<'_>,
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> OchreResult<CenteredData> {
    let n = x.rows;
    let p = x.cols;
    let k = y.cols;

    let total_weight = match sample_weight {
        Some(sw) => sw.iter().sum(),
        None => n as f64,
    };

    if !fit_intercept {
        return Ok(CenteredData {
            x_c: x.clone(),
            y_c: y.clone(),
            x_mean: vec![0.0; p],
            y_mean: vec![0.0; k],
            total_w: total_weight,
        });
    }

    let x_stats = backend.col_reduce_stats(x, sample_weight)?;
    let y_stats = backend.col_reduce_stats(y, sample_weight)?;
    let ones_x = vec![1.0; p];
    let ones_y = vec![1.0; k];
    let x_centered = backend.col_scale_shift(x, &ones_x, &x_stats.mean)?;
    let y_centered = backend.col_scale_shift(y, &ones_y, &y_stats.mean)?;

    Ok(CenteredData {
        x_c: x_centered,
        y_c: y_centered,
        x_mean: x_stats.mean,
        y_mean: y_stats.mean,
        total_w: total_weight,
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_gram_cd_single(
    gram: &Matrix,
    xty: &[f64],
    total_weight: f64,
    l1_reg: f64,
    l2_reg: f64,
    max_iter: usize,
    tol: f64,
    positive: bool,
) -> (Vec<f64>, usize) {
    let p = gram.cols;
    let inv_w = 1.0 / total_weight;
    let g_norm: Vec<f64> = gram.data.iter().map(|&v| v * inv_w).collect();
    let c_norm: Vec<f64> = xty.iter().map(|&v| v * inv_w).collect();

    let mut w = vec![0.0; p];
    let mut gw = vec![0.0; p];
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut max_delta = 0.0_f64;

        for j in 0..p {
            let w_old = w[j];
            let g_jj = g_norm[j * p + j];
            let denom = g_jj + l2_reg;
            if denom <= 0.0 {
                continue;
            }

            let z = c_norm[j] - gw[j] + g_jj * w_old;
            let mut w_new =
                apply_scalar_op(z, ElementwiseOp::SoftThreshold { lambda: l1_reg }) / denom;
            if positive && w_new < 0.0 {
                w_new = 0.0;
            }

            let delta = w_new - w_old;
            if delta.abs() > 0.0 {
                w[j] = w_new;
                let col_offset = j * p;
                for m in 0..p {
                    gw[m] += delta * g_norm[col_offset + m];
                }
                if delta.abs() > max_delta {
                    max_delta = delta.abs();
                }
            }
        }

        if max_delta <= tol {
            break;
        }
    }

    (w, n_iter)
}

fn estimate_lipschitz_fista(
    backend: &Backend<'_>,
    x: &Matrix,
    sample_weight: Option<&[f64]>,
    total_weight: f64,
    l2_reg: f64,
) -> OchreResult<f64> {
    let p = x.cols;
    let inv_w = 1.0 / total_weight;
    let mut v: Vec<f64> = (0..p).map(|j| 1.0 / ((j + 1) as f64)).collect();
    let norm0 = dot_unrolled(&v, &v).sqrt().max(1e-12);
    for val in &mut v {
        *val /= norm0;
    }

    let mut eig_est = 1.0 + l2_reg;
    for _ in 0..15 {
        let mut xv = backend.matvec(x, &v)?;
        if let Some(sw) = sample_weight {
            for (r, &w) in xv.iter_mut().zip(sw.iter()) {
                *r *= w * inv_w;
            }
        } else {
            for r in &mut xv {
                *r *= inv_w;
            }
        }
        let r_mat = Matrix::from_vec(x.rows, 1, xv)?;
        let mut xtxv = backend.matmul_at_b(x, &r_mat)?.data;
        for j in 0..p {
            xtxv[j] += l2_reg * v[j];
        }
        let nrm = dot_unrolled(&xtxv, &xtxv).sqrt();
        if nrm <= 1e-15 {
            return Ok(1.0 + l2_reg);
        }
        eig_est = nrm;
        for j in 0..p {
            v[j] = xtxv[j] / nrm;
        }
    }

    Ok((eig_est * 1.05).max(1e-8))
}

#[allow(clippy::too_many_arguments)]
fn solve_fista_single(
    backend: &Backend<'_>,
    x: &Matrix,
    y: &[f64],
    sample_weight: Option<&[f64]>,
    total_weight: f64,
    l1_reg: f64,
    l2_reg: f64,
    max_iter: usize,
    tol: f64,
    positive: bool,
) -> OchreResult<(Vec<f64>, usize)> {
    let n = x.rows;
    let p = x.cols;
    let inv_w = 1.0 / total_weight;
    let lipschitz = estimate_lipschitz_fista(backend, x, sample_weight, total_weight, l2_reg)?;
    let step = 1.0 / lipschitz;

    let mut w = vec![0.0; p];
    let mut u = vec![0.0; p];
    let mut t = 1.0_f64;
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let xu = backend.matvec(x, &u)?;
        let mut res = vec![0.0; n];
        match sample_weight {
            Some(sw) => {
                for i in 0..n {
                    res[i] = (xu[i] - y[i]) * sw[i] * inv_w;
                }
            }
            None => {
                for i in 0..n {
                    res[i] = (xu[i] - y[i]) * inv_w;
                }
            }
        }
        let res_mat = Matrix::from_vec(n, 1, res)?;
        let grad_data = backend.matmul_at_b(x, &res_mat)?.data;

        let mut v_vec = Vec::with_capacity(p);
        for j in 0..p {
            let g = grad_data[j] + l2_reg * u[j];
            v_vec.push(u[j] - step * g);
        }

        let v_mat = Matrix::from_vec(1, p, v_vec)?;
        let mut w_next = backend
            .elementwise_op(
                &v_mat,
                ElementwiseOp::SoftThreshold {
                    lambda: step * l1_reg,
                },
            )?
            .data;
        if positive {
            for val in &mut w_next {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }

        let mut max_diff = 0.0_f64;
        let mut restart_dot = 0.0_f64;
        for j in 0..p {
            let diff = w_next[j] - w[j];
            max_diff = max_diff.max(diff.abs());
            restart_dot += (u[j] - w_next[j]) * diff;
        }

        if max_diff <= tol {
            w = w_next;
            break;
        }

        if restart_dot > 0.0 {
            t = 1.0;
            u.copy_from_slice(&w_next);
        } else {
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
            let beta = (t - 1.0) / t_next;
            for j in 0..p {
                u[j] = w_next[j] + beta * (w_next[j] - w[j]);
            }
            t = t_next;
        }
        w = w_next;
    }

    Ok((w, n_iter))
}

#[allow(clippy::too_many_arguments)]
fn fit_sparse_linear_core(
    device: &DeviceType,
    x: &Matrix,
    y_mat: &Matrix,
    sample_weight: Option<&[f64]>,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    positive: bool,
) -> OchreResult<(Matrix, Vec<f64>, Vec<usize>)> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != y_mat.rows {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y_mat.rows,
        });
    }
    validate_sample_weights(sample_weight, x.rows)?;

    let backend = get_backend(device);
    let CenteredData {
        x_c,
        y_c,
        x_mean,
        y_mean,
        total_w,
    } = prepare_centered_data(&backend, x, y_mat, sample_weight, fit_intercept)?;

    let p = x.cols;
    let k = y_mat.cols;
    let l1_reg = alpha * l1_ratio;
    let l2_reg = alpha * (1.0 - l1_ratio);

    let mut coef_data = vec![0.0; k * p];
    let mut intercepts = vec![0.0; k];
    let mut n_iters = Vec::with_capacity(k);

    if p <= GRAM_CD_THRESHOLD {
        let (gram, xty_mat) =
            backend.compute_normal_equation_multi(&x_c, &y_c, sample_weight, false)?;
        for t_idx in 0..k {
            let mut xty_col = Vec::with_capacity(p);
            for j in 0..p {
                xty_col.push(xty_mat[(j, t_idx)]);
            }
            let (w, n_it) = solve_gram_cd_single(
                &gram, &xty_col, total_w, l1_reg, l2_reg, max_iter, tol, positive,
            );
            let intercept = if fit_intercept {
                y_mean[t_idx] - dot_unrolled(&x_mean, &w)
            } else {
                0.0
            };
            coef_data[t_idx * p..(t_idx + 1) * p].copy_from_slice(&w);
            intercepts[t_idx] = intercept;
            n_iters.push(n_it);
        }
    } else {
        for t_idx in 0..k {
            let mut y_col = Vec::with_capacity(x.rows);
            for r in 0..x.rows {
                y_col.push(y_c[(r, t_idx)]);
            }
            let (w, n_it) = solve_fista_single(
                &backend,
                &x_c,
                &y_col,
                sample_weight,
                total_w,
                l1_reg,
                l2_reg,
                max_iter,
                tol,
                positive,
            )?;
            let intercept = if fit_intercept {
                y_mean[t_idx] - dot_unrolled(&x_mean, &w)
            } else {
                0.0
            };
            coef_data[t_idx * p..(t_idx + 1) * p].copy_from_slice(&w);
            intercepts[t_idx] = intercept;
            n_iters.push(n_it);
        }
    }

    let coef_mat = Matrix::from_vec(k, p, coef_data)?;
    Ok((coef_mat, intercepts, n_iters))
}

/// L1-regularized Linear Regression (Lasso) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct Lasso {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub positive: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub is_multi_target_: bool,
    pub n_targets_: usize,
    pub n_iter_: Vec<usize>,
    pub n_features_in_: Option<usize>,
}

impl Lasso {
    /// Create a new Lasso estimator.
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        positive: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_sparse_params(alpha, 1.0, max_iter, tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            alpha,
            fit_intercept,
            max_iter,
            tol,
            positive,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            is_multi_target_: false,
            n_targets_: 1,
            n_iter_: Vec::new(),
            n_features_in_: None,
        })
    }

    /// Fit 1D Lasso model with optional sample_weight.
    pub fn fit_weighted_1d(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y.to_vec())?;
        let (coef_mat, intercepts, n_iters) = fit_sparse_linear_core(
            &self.device,
            x,
            &y_mat,
            sample_weight,
            self.alpha,
            1.0,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.positive,
        )?;
        self.coef_ = Some(coef_mat.data.clone());
        self.intercept_ = Some(intercepts[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercepts);
        self.is_multi_target_ = false;
        self.n_targets_ = 1;
        self.n_iter_ = n_iters;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Fit multi-output Lasso model with optional sample_weight.
    pub fn fit_weighted_multi(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        let (coef_mat, intercepts, n_iters) = fit_sparse_linear_core(
            &self.device,
            x,
            y,
            sample_weight,
            self.alpha,
            1.0,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.positive,
        )?;
        self.coef_ = if y.cols == 1 {
            Some(coef_mat.data.clone())
        } else {
            None
        };
        self.intercept_ = if y.cols == 1 {
            Some(intercepts[0])
        } else {
            None
        };
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercepts);
        self.is_multi_target_ = true;
        self.n_targets_ = y.cols;
        self.n_iter_ = n_iters;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Predict 1D target values.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Predict 2D multi-output target values.
    pub fn predict_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercepts = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercepts)
    }

    /// Compute R^2 score for 1D targets.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    /// Compute R^2 score for 2D multi-output targets.
    pub fn score_multi(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        let preds = self.predict_multi(x)?;
        Ok(r2_score_multi(y, &preds, None))
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
        let mut buf = write_header(TAG_LASSO);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.positive);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        write_usize_vec(&mut buf, &self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    /// Deserialize model from `.dit` binary bytes.
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LASSO)?;
        let alpha = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let positive = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let is_multi_target_ = r.read_bool()?;
        let n_targets_ = r.read_usize()?;
        let n_iter_ = r.read_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let is_fitted = coef_multi_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            alpha,
            fit_intercept,
            max_iter,
            tol,
            positive,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_iter_,
            n_features_in_,
        })
    }
}

/// ElasticNet (combined L1 and L2 regularized) Linear Regression (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct ElasticNet {
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub positive: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub is_multi_target_: bool,
    pub n_targets_: usize,
    pub n_iter_: Vec<usize>,
    pub n_features_in_: Option<usize>,
}

impl ElasticNet {
    /// Create a new ElasticNet estimator.
    pub fn new(
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        positive: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_sparse_params(alpha, l1_ratio, max_iter, tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            positive,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            is_multi_target_: false,
            n_targets_: 1,
            n_iter_: Vec::new(),
            n_features_in_: None,
        })
    }

    /// Fit 1D ElasticNet model with optional sample_weight.
    pub fn fit_weighted_1d(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y.to_vec())?;
        let (coef_mat, intercepts, n_iters) = fit_sparse_linear_core(
            &self.device,
            x,
            &y_mat,
            sample_weight,
            self.alpha,
            self.l1_ratio,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.positive,
        )?;
        self.coef_ = Some(coef_mat.data.clone());
        self.intercept_ = Some(intercepts[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercepts);
        self.is_multi_target_ = false;
        self.n_targets_ = 1;
        self.n_iter_ = n_iters;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Fit multi-output ElasticNet model with optional sample_weight.
    pub fn fit_weighted_multi(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        let (coef_mat, intercepts, n_iters) = fit_sparse_linear_core(
            &self.device,
            x,
            y,
            sample_weight,
            self.alpha,
            self.l1_ratio,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.positive,
        )?;
        self.coef_ = if y.cols == 1 {
            Some(coef_mat.data.clone())
        } else {
            None
        };
        self.intercept_ = if y.cols == 1 {
            Some(intercepts[0])
        } else {
            None
        };
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercepts);
        self.is_multi_target_ = true;
        self.n_targets_ = y.cols;
        self.n_iter_ = n_iters;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Predict 1D target values.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Predict 2D multi-output target values.
    pub fn predict_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercepts = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercepts)
    }

    /// Compute R^2 score for 1D targets.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    /// Compute R^2 score for 2D multi-output targets.
    pub fn score_multi(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        let preds = self.predict_multi(x)?;
        Ok(r2_score_multi(y, &preds, None))
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
        let mut buf = write_header(TAG_ELASTIC_NET);
        write_f64(&mut buf, self.alpha);
        write_f64(&mut buf, self.l1_ratio);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.positive);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        write_usize_vec(&mut buf, &self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    /// Deserialize model from `.dit` binary bytes.
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ELASTIC_NET)?;
        let alpha = r.read_f64()?;
        let l1_ratio = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let positive = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let is_multi_target_ = r.read_bool()?;
        let n_targets_ = r.read_usize()?;
        let n_iter_ = r.read_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let is_fitted = coef_multi_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            positive,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_iter_,
            n_features_in_,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_1d_and_positive_and_dit() {
        let x = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 3.0, 0.5, 1.0, 4.0, 2.0, 0.5, 5.0, 1.5, 1.5, 6.0,
                2.5, 2.0,
            ],
        )
        .unwrap();
        let y = vec![2.5, 4.5, 6.5, 8.5, 10.5, 12.5];

        let mut model = Lasso::new(0.05, true, 2000, 1e-7, true, Some("cpu".to_string())).unwrap();
        model.fit_weighted_1d(&x, &y, None).unwrap();

        assert!(model.score(&x, &y).unwrap() > 0.99);
        for &c in model.coef_.as_ref().unwrap() {
            assert!(c >= 0.0);
        }

        let bytes = model.to_bytes();
        let loaded = Lasso::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), model.predict(&x).unwrap());
    }

    #[test]
    fn test_elastic_net_multi_output_and_fista_fallback() {
        let x =
            Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0]).unwrap();
        let y = Matrix::from_vec(
            5,
            2,
            vec![5.0, 3.0, 4.0, 7.0, 11.0, 7.0, 10.0, 11.0, 15.0, 12.0],
        )
        .unwrap();

        let mut en =
            ElasticNet::new(0.01, 0.5, true, 2000, 1e-7, false, Some("cpu".to_string())).unwrap();
        en.fit_weighted_multi(&x, &y, None).unwrap();
        assert!(en.is_multi_target_);
        assert_eq!(en.n_targets_, 2);
        assert!(en.score_multi(&x, &y).unwrap() > 0.98);

        let bytes = en.to_bytes();
        let loaded = ElasticNet::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(
            loaded.predict_multi(&x).unwrap().data,
            en.predict_multi(&x).unwrap().data
        );
    }
}
