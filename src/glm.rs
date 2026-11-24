// OchreML Robust & Generalized Linear Models (*≧ω≦*)
// Implements HuberRegressor, QuantileRegressor, PoissonRegressor, GammaRegressor, and TweedieRegressor
// with hardware-accelerated L-BFGS / IRLS+CD solvers and .dit persistence.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64_vec,
    write_opt_str, write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{dot_unrolled, predict_linear_1d};
use crate::logistic_regression::validate_sample_weights;
use crate::matrix::{Matrix, r2_score, solve_cholesky};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes,
};
use crate::primitives::ElementwiseOp;
use crate::primitives::stats_ops::apply_scalar_op;
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use rayon::prelude::*;

pub const TAG_HUBER_REGRESSOR: u8 = 49;
pub const TAG_QUANTILE_REGRESSOR: u8 = 50;
pub const TAG_POISSON_REGRESSOR: u8 = 51;
pub const TAG_GAMMA_REGRESSOR: u8 = 52;
pub const TAG_TWEEDIE_REGRESSOR: u8 = 53;

const LBFGS_HISTORY: usize = 10;
const SMALL_MATRIX_GPU_THRESHOLD: usize = 262_144;
const SMALL_VEC_GPU_THRESHOLD: usize = 16_384;
const FUSED_PAR_THRESHOLD: usize = 4096;
const FUSED_CHUNK_ROWS: usize = 1024;

#[inline(always)]
fn axpy_add_unrolled(dst: &mut [f64], src: &[f64], scale: f64) {
    let (d_chunks, d_rem) = dst.as_chunks_mut::<4>();
    let (s_chunks, s_rem) = src.as_chunks::<4>();
    for (dc, sc) in d_chunks.iter_mut().zip(s_chunks.iter()) {
        dc[0] += scale * sc[0];
        dc[1] += scale * sc[1];
        dc[2] += scale * sc[2];
        dc[3] += scale * sc[3];
    }
    for (dj, &sj) in d_rem.iter_mut().zip(s_rem.iter()) {
        *dj += scale * sj;
    }
}

fn median_in_place(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (_, med_hi, left) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        *med_hi
    } else {
        let med_lo = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (med_lo + *med_hi)
    }
}

#[inline]
fn effective_backend<'a>(device: &'a DeviceType, rows: usize, cols: usize) -> Backend<'a> {
    if rows.saturating_mul(cols) < SMALL_MATRIX_GPU_THRESHOLD {
        Backend::Cpu
    } else {
        get_backend(device)
    }
}

fn validate_fit_inputs(x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != y.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y.len(),
        });
    }
    for &v in &x.data {
        if !v.is_finite() {
            return Err(OchreError::InvalidParameter(
                "Input feature matrix X contains NaN or Inf".to_string(),
            ));
        }
    }
    for &v in y {
        if !v.is_finite() {
            return Err(OchreError::InvalidParameter(
                "Target vector y contains NaN or Inf".to_string(),
            ));
        }
    }
    validate_sample_weights(sample_weight, x.rows)?;
    Ok(())
}

fn validate_predict_inputs(x: &Matrix, expected_features: Option<usize>) -> OchreResult<()> {
    let expected = expected_features.ok_or(OchreError::NotFitted)?;
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.cols != expected {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected,
        });
    }
    for &v in &x.data {
        if !v.is_finite() {
            return Err(OchreError::InvalidParameter(
                "Input feature matrix X contains NaN or Inf".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> OchreResult<()> {
    if alpha < 0.0 || !alpha.is_finite() {
        return Err(OchreError::InvalidAlpha(alpha));
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

fn weighted_r2_score(
    y_true: &[f64],
    y_pred: &[f64],
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
    validate_sample_weights(sample_weight, y_true.len())?;

    let Some(sw) = sample_weight else {
        return Ok(r2_score(y_true, y_pred));
    };

    let total_w: f64 = sw.iter().sum();
    let y_mean = y_true
        .iter()
        .zip(sw.iter())
        .map(|(&y, &w)| y * w)
        .sum::<f64>()
        / total_w;

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for ((&yt, &yp), &w) in y_true.iter().zip(y_pred.iter()).zip(sw.iter()) {
        ss_tot += w * (yt - y_mean).powi(2);
        ss_res += w * (yt - yp).powi(2);
    }
    if ss_tot.abs() < 1e-12 {
        return Ok(if ss_res.abs() < 1e-12 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - (ss_res / ss_tot))
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

fn run_lbfgs<F, C>(
    init_theta: Vec<f64>,
    max_iter: usize,
    tol: f64,
    clamp_fn: C,
    eval_fn: F,
) -> OchreResult<(Vec<f64>, usize)>
where
    F: FnMut(&[f64], &mut [f64]) -> OchreResult<f64>,
    C: FnMut(&mut [f64]),
{
    run_lbfgs_scaled(init_theta, max_iter, tol, 1.0, clamp_fn, eval_fn)
}

fn run_lbfgs_scaled<F, C>(
    init_theta: Vec<f64>,
    max_iter: usize,
    tol: f64,
    grad_scale: f64,
    mut clamp_fn: C,
    mut eval_fn: F,
) -> OchreResult<(Vec<f64>, usize)>
where
    F: FnMut(&[f64], &mut [f64]) -> OchreResult<f64>,
    C: FnMut(&mut [f64]),
{
    let dim = init_theta.len();
    let scale_denom = grad_scale.max(1.0);
    let mut theta = init_theta;
    clamp_fn(&mut theta);

    let mut grad = vec![0.0; dim];
    let mut loss = eval_fn(&theta, &mut grad)?;

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_HISTORY);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_HISTORY);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(LBFGS_HISTORY);

    let mut next_theta = vec![0.0; dim];
    let mut next_grad = vec![0.0; dim];
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        let g_inf = grad.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let scaled_g_inf = g_inf / scale_denom;
        if scaled_g_inf <= tol {
            break;
        }
        n_iter = iter + 1;

        let mut dir = lbfgs_two_loop(&grad, &s_hist, &y_hist, &rho_hist);
        let mut dg = dot_unrolled(&dir, &grad);
        if dg >= 0.0 || !dg.is_finite() {
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

        for _ in 0..15 {
            for d in 0..dim {
                next_theta[d] = theta[d] + step * dir[d];
            }
            clamp_fn(&mut next_theta);
            next_loss = eval_fn(&next_theta, &mut next_grad)?;
            if next_loss.is_finite()
                && next_grad.iter().all(|g| g.is_finite())
                && next_loss <= loss + c1 * step * dg
            {
                accepted = true;
                break;
            }
            if next_loss.is_finite() {
                let denom = 2.0 * (next_loss - loss - step * dg);
                let quad_step = if denom > 1e-20 {
                    -(dg * step * step) / denom
                } else {
                    0.5 * step
                };
                step = quad_step.clamp(0.1 * step, 0.5 * step);
            } else {
                step *= 0.25;
            }
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

        let rel_change = (loss - next_loss).abs() / loss.abs().max(scale_denom);
        theta.copy_from_slice(&next_theta);
        grad.copy_from_slice(&next_grad);
        loss = next_loss;

        if sy > 1e-16 {
            if s_hist.len() == LBFGS_HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s_vec);
            y_hist.push(y_vec);
        }

        let next_g_inf = grad.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let next_scaled_g_inf = next_g_inf / scale_denom;
        if next_scaled_g_inf <= tol || (rel_change <= 1e-9 && next_scaled_g_inf <= tol.max(1e-3)) {
            break;
        }
    }

    Ok((theta, n_iter))
}

fn compute_linear_eta(
    backend: &Backend<'_>,
    x: &Matrix,
    w: &[f64],
    b: f64,
) -> OchreResult<Vec<f64>> {
    if matches!(backend, Backend::Cpu) {
        if x.rows < 8192 {
            if x.cols != w.len() {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: w.len(),
                });
            }
            let d = x.cols;
            if d == 0 {
                return Ok(vec![b; x.rows]);
            }
            let mut predictions = Vec::with_capacity(x.rows);
            for row_slice in x.data.chunks_exact(d) {
                predictions.push(b + dot_unrolled(row_slice, w));
            }
            return Ok(predictions);
        }
        predict_linear_1d(x, w, b)
    } else {
        let mut eta = backend.matvec(x, w)?;
        if b != 0.0 {
            for v in &mut eta {
                *v += b;
            }
        }
        Ok(eta)
    }
}

fn compute_xt_r(backend: &Backend<'_>, x: &Matrix, r: &[f64]) -> OchreResult<Vec<f64>> {
    let n = x.rows;
    let p = x.cols;
    if matches!(backend, Backend::Cpu) {
        if n >= 2048 {
            const CHUNK_ROWS: usize = 1024;
            let grad_w = x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .zip(r.par_chunks(CHUNK_ROWS))
                .map(|(x_chunk, r_chunk)| {
                    let mut loc = vec![0.0; p];
                    for i in 0..r_chunk.len() {
                        let ri = r_chunk[i];
                        let row = &x_chunk[i * p..(i + 1) * p];
                        for j in 0..p {
                            loc[j] += ri * row[j];
                        }
                    }
                    loc
                })
                .reduce(
                    || vec![0.0; p],
                    |mut a, b| {
                        for j in 0..p {
                            a[j] += b[j];
                        }
                        a
                    },
                );
            Ok(grad_w)
        } else {
            let mut grad_w = vec![0.0; p];
            for (i, &ri) in r.iter().enumerate() {
                let row = &x.data[i * p..(i + 1) * p];
                for j in 0..p {
                    grad_w[j] += ri * row[j];
                }
            }
            Ok(grad_w)
        }
    } else {
        let r_mat = Matrix::from_vec(n, 1, r.to_vec())?;
        let gw_mat = backend.matmul_at_b(x, &r_mat)?;
        Ok(gw_mat.data)
    }
}

fn write_bool_vec(buf: &mut Vec<u8>, v: &[bool]) {
    write_usize(buf, v.len());
    for &b in v {
        write_bool(buf, b);
    }
}

fn write_opt_bool_vec(buf: &mut Vec<u8>, v: &Option<Vec<bool>>) {
    match v {
        Some(vec) => {
            write_bool(buf, true);
            write_bool_vec(buf, vec);
        }
        None => write_bool(buf, false),
    }
}

fn read_opt_bool_vec(reader: &mut ByteReader<'_>) -> OchreResult<Option<Vec<bool>>> {
    if reader.read_bool()? {
        let len = reader.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(reader.read_bool()?);
        }
        Ok(Some(vec))
    } else {
        Ok(None)
    }
}

// ============================================================================
// 1. HuberRegressor (TAG = 49)
// ============================================================================

#[derive(Debug, Clone)]
pub struct HuberRegressor {
    pub epsilon: f64,
    pub max_iter: usize,
    pub alpha: f64,
    pub fit_intercept: bool,
    pub tol: f64,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: f64,
    pub scale_: f64,
    pub n_iter_: usize,
    pub outliers_: Option<Vec<bool>>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    parsed_device: DeviceType,
}

impl HuberRegressor {
    pub fn new(
        epsilon: f64,
        max_iter: usize,
        alpha: f64,
        fit_intercept: bool,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if epsilon < 1.0 || !epsilon.is_finite() {
            return Err(OchreError::InvalidParameter(format!(
                "epsilon ({epsilon}) must be >= 1.0 and finite"
            )));
        }
        validate_max_iter(max_iter)?;
        validate_alpha(alpha)?;
        validate_tol(tol)?;

        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed_device = parse_device(Some(dev_str))?;
        let resolved_str = parsed_device.to_string();

        Ok(Self {
            epsilon,
            max_iter,
            alpha,
            fit_intercept,
            tol,
            coef_: None,
            intercept_: 0.0,
            scale_: 1.0,
            n_iter_: 0,
            outliers_: None,
            n_features_in_: None,
            device_: Some(resolved_str),
            parsed_device,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed = parse_device(Some(dev_str))?;
        self.parsed_device = parsed.clone();
        self.device_ = Some(parsed.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_fit_inputs(x, y, sample_weight)?;
        let n = x.rows;
        let p = x.cols;

        let backend = effective_backend(&self.parsed_device, n, p);
        let dim = p + usize::from(self.fit_intercept) + 1;
        let mut init_theta = vec![0.0; dim];

        // Warm-start scale from 1.4826 * MAD(y) (fallback std(y), clamped >= 1e-3)
        // and w, b from fast Ridge/normal equation (p <= 256) or median intercept.
        let mut y_buf = y.to_vec();
        let med_y = median_in_place(&mut y_buf);
        for v in &mut y_buf {
            *v = (*v - med_y).abs();
        }
        let mad = median_in_place(&mut y_buf);
        let mad_scale = 1.4826 * mad;
        let init_scale = if mad_scale.is_finite() && mad_scale > 1e-6 {
            mad_scale.max(1e-3)
        } else {
            let mean_y = y.iter().sum::<f64>() / (n as f64);
            let var_y = y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / (n as f64);
            var_y.sqrt().max(1e-3)
        };

        let mut w_init = vec![0.0; p];
        let mut b_init = if self.fit_intercept { med_y } else { 0.0 };
        if p <= 256 && matches!(backend, Backend::Cpu) {
            let (x_bar, y_bar) = if self.fit_intercept {
                let (mut xb, mut yb, mut sw_sum) = (vec![0.0; p], 0.0, 0.0);
                if let Some(sw) = sample_weight {
                    for i in 0..n {
                        let wi = sw[i];
                        sw_sum += wi;
                        yb += wi * y[i];
                        axpy_add_unrolled(&mut xb, &x.data[i * p..(i + 1) * p], wi);
                    }
                } else {
                    sw_sum = n as f64;
                    for (i, &yi) in y.iter().enumerate() {
                        yb += yi;
                        axpy_add_unrolled(&mut xb, &x.data[i * p..(i + 1) * p], 1.0);
                    }
                }
                if sw_sum > 0.0 {
                    let inv = 1.0 / sw_sum;
                    for v in &mut xb {
                        *v *= inv;
                    }
                    (xb, yb * inv)
                } else {
                    (vec![0.0; p], 0.0)
                }
            } else {
                (vec![0.0; p], 0.0)
            };

            let (mut gram_data, rhs) = if n >= FUSED_PAR_THRESHOLD {
                let (g, r) = x
                    .data
                    .par_chunks(FUSED_CHUNK_ROWS * p)
                    .zip(y.par_chunks(FUSED_CHUNK_ROWS))
                    .enumerate()
                    .fold(
                        || (vec![0.0; p * p], vec![0.0; p], vec![0.0; p]),
                        |(mut lg, mut lr, mut centered), (c_idx, (xc, yc))| {
                            let base = c_idx * FUSED_CHUNK_ROWS;
                            for i in 0..yc.len() {
                                let wi = sample_weight.map_or(1.0, |s| s[base + i]);
                                let row = &xc[i * p..(i + 1) * p];
                                for j in 0..p {
                                    centered[j] = row[j] - x_bar[j];
                                }
                                let yci = wi * (yc[i] - y_bar);
                                for j in 0..p {
                                    let w_xj = wi * centered[j];
                                    lr[j] += yci * centered[j];
                                    let g_row = &mut lg[j * p..(j + 1) * p];
                                    for k in j..p {
                                        g_row[k] += w_xj * centered[k];
                                    }
                                }
                            }
                            (lg, lr, centered)
                        },
                    )
                    .map(|(g, r, _)| (g, r))
                    .reduce(
                        || (vec![0.0; p * p], vec![0.0; p]),
                        |(mut g1, mut r1), (g2, r2)| {
                            for idx in 0..p * p {
                                g1[idx] += g2[idx];
                            }
                            for idx in 0..p {
                                r1[idx] += r2[idx];
                            }
                            (g1, r1)
                        },
                    );
                (g, r)
            } else {
                let mut lg = vec![0.0; p * p];
                let mut lr = vec![0.0; p];
                let mut centered = vec![0.0; p];
                for i in 0..n {
                    let wi = sample_weight.map_or(1.0, |s| s[i]);
                    let row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        centered[j] = row[j] - x_bar[j];
                    }
                    let yci = wi * (y[i] - y_bar);
                    for j in 0..p {
                        let w_xj = wi * centered[j];
                        lr[j] += yci * centered[j];
                        let g_row = &mut lg[j * p..(j + 1) * p];
                        for k in j..p {
                            g_row[k] += w_xj * centered[k];
                        }
                    }
                }
                (lg, lr)
            };

            for j in 0..p {
                for k in (j + 1)..p {
                    gram_data[k * p + j] = gram_data[j * p + k];
                }
            }
            if let Ok(mut gram) = Matrix::from_vec(p, p, gram_data) {
                gram.add_ridge_diagonal(self.alpha.max(1e-4));
                if let Ok(sol) = solve_cholesky(&gram, &rhs)
                    && sol.iter().all(|v| v.is_finite())
                {
                    w_init = sol;
                    if self.fit_intercept {
                        b_init = y_bar - dot_unrolled(&x_bar, &w_init);
                    }
                }
            }
        }

        init_theta[..p].copy_from_slice(&w_init);
        if self.fit_intercept {
            init_theta[p] = b_init;
        }
        init_theta[dim - 1] = init_scale.ln().clamp(-27.6, 50.0);

        let eps = self.epsilon;
        let eps_sq = eps * eps;
        let alpha = self.alpha;
        let fit_intercept = self.fit_intercept;
        let is_cpu = matches!(backend, Backend::Cpu);
        let weight_sum: f64 = sample_weight.map_or(n as f64, |sw| sw.iter().sum());

        let (theta, n_iter) = run_lbfgs_scaled(
            init_theta,
            self.max_iter,
            self.tol,
            weight_sum.max(1.0),
            |t| {
                t[dim - 1] = t[dim - 1].clamp(-27.6, 50.0);
            },
            |t, grad_out| {
                let w = &t[..p];
                let b = if fit_intercept { t[p] } else { 0.0 };
                let scale = t[dim - 1].exp();
                let inv_scale = 1.0 / scale;
                let threshold = eps * scale;

                let (data_loss, grad_b, grad_scale) = if is_cpu {
                    if n >= FUSED_PAR_THRESHOLD {
                        let (dl, gb, gs, gw) = x
                            .data
                            .par_chunks(FUSED_CHUNK_ROWS * p)
                            .zip(y.par_chunks(FUSED_CHUNK_ROWS))
                            .enumerate()
                            .fold(
                                || (0.0_f64, 0.0_f64, 0.0_f64, vec![0.0_f64; p]),
                                |(mut dl, mut gb, mut gs, mut gw), (c_idx, (xc, yc))| {
                                    let base = c_idx * FUSED_CHUNK_ROWS;
                                    for i in 0..yc.len() {
                                        let row = &xc[i * p..(i + 1) * p];
                                        let r = dot_unrolled(row, w) + b - yc[i];
                                        let abs_r = r.abs();
                                        let wi = sample_weight.map_or(1.0, |s| s[base + i]);
                                        let gr = if abs_r <= threshold {
                                            let z = r * inv_scale;
                                            dl += wi * (scale + r * z);
                                            gs += wi * (1.0 - z * z);
                                            wi * 2.0 * z
                                        } else {
                                            dl += wi * (scale + 2.0 * eps * abs_r - eps_sq * scale);
                                            gs += wi * (1.0 - eps_sq);
                                            wi * 2.0 * eps * r.signum()
                                        };
                                        gb += gr;
                                        axpy_add_unrolled(&mut gw, row, gr);
                                    }
                                    (dl, gb, gs, gw)
                                },
                            )
                            .reduce(
                                || (0.0_f64, 0.0_f64, 0.0_f64, vec![0.0_f64; p]),
                                |(dl1, gb1, gs1, mut gw1), (dl2, gb2, gs2, gw2)| {
                                    for j in 0..p {
                                        gw1[j] += gw2[j];
                                    }
                                    (dl1 + dl2, gb1 + gb2, gs1 + gs2, gw1)
                                },
                            );
                        grad_out[..p].copy_from_slice(&gw);
                        (dl, gb, gs)
                    } else {
                        grad_out[..p].fill(0.0);
                        let mut dl = 0.0;
                        let mut gb = 0.0;
                        let mut gs = 0.0;
                        for i in 0..n {
                            let row = &x.data[i * p..(i + 1) * p];
                            let r = dot_unrolled(row, w) + b - y[i];
                            let abs_r = r.abs();
                            let wi = sample_weight.map_or(1.0, |s| s[i]);
                            let gr = if abs_r <= threshold {
                                let z = r * inv_scale;
                                dl += wi * (scale + r * z);
                                gs += wi * (1.0 - z * z);
                                wi * 2.0 * z
                            } else {
                                dl += wi * (scale + 2.0 * eps * abs_r - eps_sq * scale);
                                gs += wi * (1.0 - eps_sq);
                                wi * 2.0 * eps * r.signum()
                            };
                            gb += gr;
                            axpy_add_unrolled(&mut grad_out[..p], row, gr);
                        }
                        (dl, gb, gs)
                    }
                } else {
                    let preds = compute_linear_eta(&backend, x, w, b)?;
                    let mut grad_r = vec![0.0; n];
                    let mut dl = 0.0;
                    let mut gb = 0.0;
                    let mut gs = 0.0;

                    for i in 0..n {
                        let r = preds[i] - y[i];
                        let abs_r = r.abs();
                        let wi = sample_weight.map_or(1.0, |s| s[i]);
                        if abs_r <= threshold {
                            let z = r * inv_scale;
                            dl += wi * (scale + r * z);
                            let gr = wi * 2.0 * z;
                            grad_r[i] = gr;
                            gb += gr;
                            gs += wi * (1.0 - z * z);
                        } else {
                            dl += wi * (scale + 2.0 * eps * abs_r - eps_sq * scale);
                            let gr = wi * 2.0 * eps * r.signum();
                            grad_r[i] = gr;
                            gb += gr;
                            gs += wi * (1.0 - eps_sq);
                        }
                    }

                    let grad_w = compute_xt_r(&backend, x, &grad_r)?;
                    grad_out[..p].copy_from_slice(&grad_w);
                    (dl, gb, gs)
                };

                let mut reg_loss = 0.0;
                for j in 0..p {
                    let wj = w[j];
                    reg_loss += wj * wj;
                    grad_out[j] += 2.0 * alpha * wj;
                }
                if fit_intercept {
                    grad_out[p] = grad_b;
                }
                grad_out[dim - 1] = scale * grad_scale;

                Ok(data_loss + alpha * reg_loss)
            },
        )?;

        let coef = theta[..p].to_vec();
        let intercept = if self.fit_intercept { theta[p] } else { 0.0 };
        let scale = theta[dim - 1].clamp(-27.6, 50.0).exp();
        let preds = compute_linear_eta(&backend, x, &coef, intercept)?;
        let threshold = self.epsilon * scale;
        let outliers: Vec<bool> = preds
            .iter()
            .zip(y.iter())
            .map(|(&yp, &yt)| (yt - yp).abs() > threshold)
            .collect();

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
        self.scale_ = scale;
        self.n_iter_ = n_iter;
        self.outliers_ = Some(outliers);
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        validate_predict_inputs(x, self.n_features_in_)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let backend = effective_backend(&self.parsed_device, x.rows, x.cols);
        compute_linear_eta(&backend, x, coef, self.intercept_)
    }

    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        weighted_r2_score(y, &preds, sample_weight)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_HUBER_REGRESSOR);
        write_f64(&mut buf, self.epsilon);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_f64(&mut buf, self.tol);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_f64(&mut buf, self.intercept_);
        write_f64(&mut buf, self.scale_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_bool_vec(&mut buf, &self.outliers_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_HUBER_REGRESSOR)?;
        let epsilon = reader.read_f64()?;
        let max_iter = reader.read_usize()?;
        let alpha = reader.read_f64()?;
        let fit_intercept = reader.read_bool()?;
        let tol = reader.read_f64()?;
        let coef_ = reader.read_opt_f64_vec()?;
        let intercept_ = reader.read_f64()?;
        let scale_ = reader.read_f64()?;
        let n_iter_ = reader.read_usize()?;
        let outliers_ = read_opt_bool_vec(&mut reader)?;
        let n_features_in_ = reader.read_opt_usize()?;
        let saved_device = reader.read_opt_str()?;
        let (parsed_device, device_) =
            resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            epsilon,
            max_iter,
            alpha,
            fit_intercept,
            tol,
            coef_,
            intercept_,
            scale_,
            n_iter_,
            outliers_,
            n_features_in_,
            device_,
            parsed_device,
        })
    }
}

#[pyclass(from_py_object, name = "HuberRegressor")]
#[derive(Clone)]
pub struct PyHuberRegressor {
    inner: HuberRegressor,
}

pymethods_with_dit!(PyHuberRegressor, HuberRegressor, {
    #[new]
    #[pyo3(signature = (epsilon = 1.35, max_iter = 100, alpha = 0.0001, fit_intercept = true, tol = 1e-5, device = None))]
    pub fn new(
        epsilon: f64,
        max_iter: usize,
        alpha: f64,
        fit_intercept: bool,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: HuberRegressor::new(epsilon, max_iter, alpha, fit_intercept, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &y_vec, sw_vec.as_deref())?)
    }

    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> f64 {
        self.inner.intercept_
    }
    #[getter]
    pub fn scale_(&self) -> f64 {
        self.inner.scale_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn outliers_(&self) -> Option<Vec<bool>> {
        self.inner.outliers_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

// ============================================================================
// 2. QuantileRegressor (TAG = 50)
// ============================================================================

fn weighted_quantile(values: &[f64], weights_norm: &[f64], q: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights_norm.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cum = 0.0;
    for &(val, w) in &pairs {
        cum += w;
        if cum >= q {
            return val;
        }
    }
    pairs.last().map_or(0.0, |p| p.0)
}

fn weighted_std(values: &[f64], weights_norm: &[f64]) -> f64 {
    let mean: f64 = values
        .iter()
        .zip(weights_norm.iter())
        .map(|(&v, &w)| v * w)
        .sum();
    let var: f64 = values
        .iter()
        .zip(weights_norm.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum();
    var.sqrt()
}

fn compute_weighted_gram_and_rhs(
    backend: &Backend<'_>,
    x: &Matrix,
    y_tilde: &[f64],
    wt: &[f64],
    fit_intercept: bool,
) -> OchreResult<(Matrix, Vec<f64>, Vec<f64>, f64)> {
    let n = x.rows;
    let p = x.cols;
    let sum_wt: f64 = wt.iter().sum();

    let (x_bar, y_bar) = if fit_intercept && sum_wt > 0.0 {
        let mut xb = vec![0.0; p];
        let mut yb = 0.0;
        for i in 0..n {
            let wi = wt[i];
            yb += wi * y_tilde[i];
            let row = &x.data[i * p..(i + 1) * p];
            for j in 0..p {
                xb[j] += wi * row[j];
            }
        }
        let inv = 1.0 / sum_wt;
        for v in &mut xb {
            *v *= inv;
        }
        (xb, yb * inv)
    } else {
        (vec![0.0; p], 0.0)
    };

    if !matches!(backend, Backend::Cpu) {
        let mut wx_data = vec![0.0; n * p];
        let mut x_c_data = vec![0.0; n * p];
        let mut wy_c = vec![0.0; n];
        for i in 0..n {
            let wi = wt[i];
            wy_c[i] = wi * (y_tilde[i] - y_bar);
            let row = &x.data[i * p..(i + 1) * p];
            let xc_row = &mut x_c_data[i * p..(i + 1) * p];
            let wx_row = &mut wx_data[i * p..(i + 1) * p];
            for j in 0..p {
                let xc = row[j] - x_bar[j];
                xc_row[j] = xc;
                wx_row[j] = wi * xc;
            }
        }
        let x_c = Matrix::from_vec(n, p, x_c_data)?;
        let wx = Matrix::from_vec(n, p, wx_data)?;
        let wy_mat = Matrix::from_vec(n, 1, wy_c)?;
        let gram = backend.matmul_at_b(&wx, &x_c)?;
        let rhs_mat = backend.matmul_at_b(&x_c, &wy_mat)?;
        return Ok((gram, rhs_mat.data, x_bar, y_bar));
    }

    let mut gram_data = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];

    if n >= 1024 && p <= 256 {
        const CHUNK: usize = 512;
        let (par_g, par_r) = x
            .data
            .par_chunks(CHUNK * p)
            .zip(y_tilde.par_chunks(CHUNK))
            .zip(wt.par_chunks(CHUNK))
            .map(|((xc, yc), wc)| {
                let mut lg = vec![0.0; p * p];
                let mut lr = vec![0.0; p];
                let mut centered = vec![0.0; p];
                for i in 0..yc.len() {
                    let wi = wc[i];
                    let row = &xc[i * p..(i + 1) * p];
                    for j in 0..p {
                        centered[j] = row[j] - x_bar[j];
                    }
                    let yci = wi * (yc[i] - y_bar);
                    for j in 0..p {
                        let w_xj = wi * centered[j];
                        lr[j] += yci * centered[j];
                        let g_row = &mut lg[j * p..(j + 1) * p];
                        for k in j..p {
                            g_row[k] += w_xj * centered[k];
                        }
                    }
                }
                (lg, lr)
            })
            .reduce(
                || (vec![0.0; p * p], vec![0.0; p]),
                |(mut g1, mut r1), (g2, r2)| {
                    for idx in 0..p * p {
                        g1[idx] += g2[idx];
                    }
                    for idx in 0..p {
                        r1[idx] += r2[idx];
                    }
                    (g1, r1)
                },
            );
        gram_data = par_g;
        rhs = par_r;
    } else {
        let mut centered = vec![0.0; p];
        for i in 0..n {
            let wi = wt[i];
            let row = &x.data[i * p..(i + 1) * p];
            for j in 0..p {
                centered[j] = row[j] - x_bar[j];
            }
            let yci = wi * (y_tilde[i] - y_bar);
            for j in 0..p {
                let w_xj = wi * centered[j];
                rhs[j] += yci * centered[j];
                let g_row = &mut gram_data[j * p..(j + 1) * p];
                for k in j..p {
                    g_row[k] += w_xj * centered[k];
                }
            }
        }
    }

    // Symmetrize upper -> lower
    for j in 0..p {
        for k in (j + 1)..p {
            gram_data[k * p + j] = gram_data[j * p + k];
        }
    }

    Ok((Matrix::from_vec(p, p, gram_data)?, rhs, x_bar, y_bar))
}

#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    pub quantile: f64,
    pub alpha: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: f64,
    pub n_iter_: usize,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    parsed_device: DeviceType,
}

impl QuantileRegressor {
    pub fn new(
        quantile: f64,
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !(0.0 < quantile && quantile < 1.0) || !quantile.is_finite() {
            return Err(OchreError::InvalidParameter(format!(
                "quantile ({quantile}) must be strictly between 0.0 and 1.0"
            )));
        }
        validate_alpha(alpha)?;
        validate_max_iter(max_iter)?;
        validate_tol(tol)?;

        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed_device = parse_device(Some(dev_str))?;
        let resolved_str = parsed_device.to_string();

        Ok(Self {
            quantile,
            alpha,
            fit_intercept,
            max_iter,
            tol,
            coef_: None,
            intercept_: 0.0,
            n_iter_: 0,
            n_features_in_: None,
            device_: Some(resolved_str),
            parsed_device,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed = parse_device(Some(dev_str))?;
        self.parsed_device = parsed.clone();
        self.device_ = Some(parsed.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_fit_inputs(x, y, sample_weight)?;
        let n = x.rows;
        let p = x.cols;
        let sw_norm: Vec<f64> = match sample_weight {
            Some(sw) => {
                let s: f64 = sw.iter().sum();
                sw.iter().map(|&v| v / s).collect()
            }
            None => vec![1.0 / (n as f64); n],
        };

        let backend = effective_backend(&self.parsed_device, n, p);
        let mut w = vec![0.0; p];
        let mut b = if self.fit_intercept {
            weighted_quantile(y, &sw_norm, self.quantile)
        } else {
            0.0
        };

        let y_scale = weighted_std(y, &sw_norm).max(1.0);
        let mut eps = (1e-3 * y_scale).max(1e-6);
        let min_eps = 1e-8;
        let q_shift = 2.0 * self.quantile - 1.0;

        let mut wt = vec![0.0; n];
        let mut y_tilde = vec![0.0; n];
        let mut n_iter = 0usize;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            if matches!(backend, Backend::Cpu) {
                if n >= FUSED_PAR_THRESHOLD {
                    x.data
                        .par_chunks(FUSED_CHUNK_ROWS * p)
                        .zip(y.par_chunks(FUSED_CHUNK_ROWS))
                        .zip(sw_norm.par_chunks(FUSED_CHUNK_ROWS))
                        .zip(wt.par_chunks_mut(FUSED_CHUNK_ROWS))
                        .zip(y_tilde.par_chunks_mut(FUSED_CHUNK_ROWS))
                        .for_each(|((((xc, yc), swc), wtc), ytc)| {
                            for i in 0..yc.len() {
                                let row = &xc[i * p..(i + 1) * p];
                                let r = yc[i] - (dot_unrolled(row, &w) + b);
                                let d = r.abs().max(eps);
                                wtc[i] = swc[i] / (2.0 * d);
                                ytc[i] = yc[i] + d * q_shift;
                            }
                        });
                } else {
                    for i in 0..n {
                        let row = &x.data[i * p..(i + 1) * p];
                        let r = y[i] - (dot_unrolled(row, &w) + b);
                        let d = r.abs().max(eps);
                        wt[i] = sw_norm[i] / (2.0 * d);
                        y_tilde[i] = y[i] + d * q_shift;
                    }
                }
            } else {
                let preds = compute_linear_eta(&backend, x, &w, b)?;
                for i in 0..n {
                    let r = y[i] - preds[i];
                    let d = r.abs().max(eps);
                    wt[i] = sw_norm[i] / (2.0 * d);
                    y_tilde[i] = y[i] + d * q_shift;
                }
            }

            let (mut gram, rhs, x_bar, y_bar) =
                compute_weighted_gram_and_rhs(&backend, x, &y_tilde, &wt, self.fit_intercept)?;

            let mut w_new = w.clone();
            if self.alpha == 0.0 && p <= 256 {
                gram.add_ridge_diagonal(1e-12);
                if let Ok(sol) = solve_cholesky(&gram, &rhs) {
                    w_new = sol;
                }
            } else {
                // Warm-started Coordinate Descent for L1-penalized weighted least squares
                let mut gw = vec![0.0; p];
                for (j, gw_j) in gw.iter_mut().enumerate() {
                    *gw_j = dot_unrolled(&gram.data[j * p..(j + 1) * p], &w_new);
                }
                for _cd in 0..250 {
                    let mut cd_delta = 0.0_f64;
                    for j in 0..p {
                        let g_jj = gram.data[j * p + j];
                        if g_jj <= 1e-15 {
                            if w_new[j] != 0.0 {
                                let diff = -w_new[j];
                                w_new[j] = 0.0;
                                for (k, gw_k) in gw.iter_mut().enumerate() {
                                    *gw_k += gram.data[j * p + k] * diff;
                                }
                            }
                            continue;
                        }
                        let w_old = w_new[j];
                        let z = rhs[j] - gw[j] + g_jj * w_old;
                        let new_val =
                            apply_scalar_op(z, ElementwiseOp::SoftThreshold { lambda: self.alpha })
                                / g_jj;
                        let diff = new_val - w_old;
                        if diff != 0.0 {
                            w_new[j] = new_val;
                            for (k, gw_k) in gw.iter_mut().enumerate() {
                                *gw_k += gram.data[j * p + k] * diff;
                            }
                            cd_delta = cd_delta.max(diff.abs());
                        }
                    }
                    if cd_delta < 1e-9 {
                        break;
                    }
                }
            }

            let b_new = if self.fit_intercept {
                y_bar - dot_unrolled(&x_bar, &w_new)
            } else {
                0.0
            };

            let mut max_change = (b_new - b).abs();
            for j in 0..p {
                max_change = max_change.max((w_new[j] - w[j]).abs());
            }

            w = w_new;
            b = b_new;

            if eps > min_eps {
                if max_change < (eps * 0.25).max(self.tol) {
                    eps = (eps * 0.25).max(min_eps);
                }
            } else if max_change <= self.tol {
                break;
            }
        }

        self.coef_ = Some(w);
        self.intercept_ = b;
        self.n_iter_ = n_iter;
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        validate_predict_inputs(x, self.n_features_in_)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let backend = effective_backend(&self.parsed_device, x.rows, x.cols);
        compute_linear_eta(&backend, x, coef, self.intercept_)
    }

    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        weighted_r2_score(y, &preds, sample_weight)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_QUANTILE_REGRESSOR);
        write_f64(&mut buf, self.quantile);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_f64(&mut buf, self.intercept_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_QUANTILE_REGRESSOR)?;
        let quantile = reader.read_f64()?;
        let alpha = reader.read_f64()?;
        let fit_intercept = reader.read_bool()?;
        let max_iter = reader.read_usize()?;
        let tol = reader.read_f64()?;
        let coef_ = reader.read_opt_f64_vec()?;
        let intercept_ = reader.read_f64()?;
        let n_iter_ = reader.read_usize()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let saved_device = reader.read_opt_str()?;
        let (parsed_device, device_) =
            resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            quantile,
            alpha,
            fit_intercept,
            max_iter,
            tol,
            coef_,
            intercept_,
            n_iter_,
            n_features_in_,
            device_,
            parsed_device,
        })
    }
}

#[pyclass(from_py_object, name = "QuantileRegressor")]
#[derive(Clone)]
pub struct PyQuantileRegressor {
    inner: QuantileRegressor,
}

pymethods_with_dit!(PyQuantileRegressor, QuantileRegressor, {
    #[new]
    #[pyo3(signature = (quantile = 0.5, alpha = 1.0, fit_intercept = true, max_iter = 1000, tol = 1e-6, device = None))]
    pub fn new(
        quantile: f64,
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: QuantileRegressor::new(quantile, alpha, fit_intercept, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &y_vec, sw_vec.as_deref())?)
    }

    #[getter]
    pub fn quantile(&self) -> f64 {
        self.inner.quantile
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> f64 {
        self.inner.intercept_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

// ============================================================================
// 3. TweedieRegressor, PoissonRegressor, GammaRegressor (TAGS = 53, 51, 52)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmLink {
    Identity,
    Log,
}

fn resolve_glm_link(link: &str, power: f64) -> OchreResult<GlmLink> {
    match link {
        "auto" => {
            if power <= 0.0 {
                Ok(GlmLink::Identity)
            } else {
                Ok(GlmLink::Log)
            }
        }
        "identity" => Ok(GlmLink::Identity),
        "log" => Ok(GlmLink::Log),
        other => Err(OchreError::InvalidParameter(format!(
            "Invalid link '{other}'. Expected 'auto', 'identity', or 'log'"
        ))),
    }
}

fn validate_tweedie_power(power: f64) -> OchreResult<()> {
    if !power.is_finite() || (0.0 < power && power < 1.0) {
        return Err(OchreError::InvalidParameter(format!(
            "Tweedie power ({power}) must be finite and not in (0, 1)"
        )));
    }
    Ok(())
}

fn validate_tweedie_targets(y: &[f64], power: f64) -> OchreResult<()> {
    if (power - 1.0).abs() < 1e-12 || (1.0 < power && power < 2.0) {
        for &val in y {
            if val < 0.0 {
                return Err(OchreError::InvalidParameter(format!(
                    "Target values y must be non-negative (>= 0) for power={power}, got {val}"
                )));
            }
        }
    } else if power >= 2.0 {
        for &val in y {
            if val <= 0.0 {
                return Err(OchreError::InvalidParameter(format!(
                    "Target values y must be strictly positive (> 0) for power={power}, got {val}"
                )));
            }
        }
    }
    Ok(())
}

#[inline]
fn unit_tweedie_deviance(y: f64, mu: f64, power: f64) -> f64 {
    if power == 0.0 {
        let d = y - mu;
        d * d
    } else if (power - 1.0).abs() < 1e-12 {
        let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
        2.0 * (term - y + mu)
    } else if (power - 2.0).abs() < 1e-12 {
        2.0 * ((mu / y).ln() + y / mu - 1.0)
    } else {
        let one_m_p = 1.0 - power;
        let two_m_p = 2.0 - power;
        let y_term = if y > 0.0 {
            y.powf(two_m_p) / (one_m_p * two_m_p)
        } else {
            0.0
        };
        2.0 * (y_term - y * mu.powf(one_m_p) / one_m_p + mu.powf(two_m_p) / two_m_p)
    }
}

fn tweedie_d2_score(
    y_true: &[f64],
    y_pred: &[f64],
    power: f64,
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
    validate_sample_weights(sample_weight, y_true.len())?;

    if power == 0.0 {
        return weighted_r2_score(y_true, y_pred, sample_weight);
    }

    let n = y_true.len();
    let total_w: f64 = match sample_weight {
        Some(sw) => sw.iter().sum(),
        None => n as f64,
    };
    let y_mean: f64 = match sample_weight {
        Some(sw) => {
            y_true
                .iter()
                .zip(sw.iter())
                .map(|(&y, &w)| y * w)
                .sum::<f64>()
                / total_w
        }
        None => y_true.iter().sum::<f64>() / total_w,
    };

    let mut dev_model = 0.0;
    let mut dev_null = 0.0;
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let yi = y_true[i];
        let mui = y_pred[i];
        dev_model += w * unit_tweedie_deviance(yi, mui, power);
        dev_null += w * unit_tweedie_deviance(yi, y_mean, power);
    }

    if dev_null.abs() < 1e-12 {
        return Ok(if dev_model.abs() < 1e-12 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - dev_model / dev_null)
}

#[derive(Debug, Clone, Copy)]
enum TweediePowerKind {
    Normal,
    Poisson,
    Gamma,
    General {
        power: f64,
        one_m_p: f64,
        two_m_p: f64,
    },
}

impl TweediePowerKind {
    fn from_power(power: f64) -> Self {
        if power == 0.0 {
            Self::Normal
        } else if (power - 1.0).abs() < 1e-12 {
            Self::Poisson
        } else if (power - 2.0).abs() < 1e-12 {
            Self::Gamma
        } else {
            Self::General {
                power,
                one_m_p: 1.0 - power,
                two_m_p: 2.0 - power,
            }
        }
    }

    #[inline(always)]
    fn loss_and_grad(self, eff_link: GlmLink, yi: f64, mui: f64, wi: f64) -> (f64, f64) {
        match self {
            Self::Normal => {
                let d = mui - yi;
                let loss = 0.5 * wi * d * d;
                let g = match eff_link {
                    GlmLink::Identity => wi * d,
                    GlmLink::Log => wi * d * mui,
                };
                (loss, g)
            }
            Self::Poisson => {
                let term = if yi > 0.0 { yi * (yi / mui).ln() } else { 0.0 };
                let loss = wi * (term - yi + mui);
                let g = match eff_link {
                    GlmLink::Identity => wi * (mui - yi) / mui,
                    GlmLink::Log => wi * (mui - yi),
                };
                (loss, g)
            }
            Self::Gamma => {
                let loss = wi * ((mui / yi).ln() + yi / mui - 1.0);
                let g = match eff_link {
                    GlmLink::Identity => wi * (mui - yi) / (mui * mui),
                    GlmLink::Log => wi * (mui - yi) / mui,
                };
                (loss, g)
            }
            Self::General {
                power,
                one_m_p,
                two_m_p,
            } => {
                let y_term = if yi > 0.0 {
                    yi.powf(two_m_p) / (one_m_p * two_m_p)
                } else {
                    0.0
                };
                let mu_1mp = mui.powf(one_m_p);
                let mu_2mp = mu_1mp * mui;
                let loss = wi * (y_term - yi * mu_1mp / one_m_p + mu_2mp / two_m_p);
                let g = match eff_link {
                    GlmLink::Identity => wi * (mui - yi) * mui.powf(-power),
                    GlmLink::Log => wi * (mui - yi) * mu_1mp,
                };
                (loss, g)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TweedieRegressor {
    pub power: f64,
    pub alpha: f64,
    pub fit_intercept: bool,
    pub link: String,
    pub max_iter: usize,
    pub tol: f64,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: f64,
    pub n_iter_: usize,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    parsed_device: DeviceType,
}

impl TweedieRegressor {
    pub fn new(
        power: f64,
        alpha: f64,
        fit_intercept: bool,
        link: String,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_tweedie_power(power)?;
        validate_alpha(alpha)?;
        let _ = resolve_glm_link(&link, power)?;
        validate_max_iter(max_iter)?;
        validate_tol(tol)?;

        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed_device = parse_device(Some(dev_str))?;
        let resolved_str = parsed_device.to_string();

        Ok(Self {
            power,
            alpha,
            fit_intercept,
            link,
            max_iter,
            tol,
            coef_: None,
            intercept_: 0.0,
            n_iter_: 0,
            n_features_in_: None,
            device_: Some(resolved_str),
            parsed_device,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev_str = device.as_deref().unwrap_or("auto");
        let parsed = parse_device(Some(dev_str))?;
        self.parsed_device = parsed.clone();
        self.device_ = Some(parsed.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_fit_inputs(x, y, sample_weight)?;
        validate_tweedie_targets(y, self.power)?;
        let eff_link = resolve_glm_link(&self.link, self.power)?;

        let n = x.rows;
        let p = x.cols;
        let inv_n = 1.0 / (n as f64);
        let sw_norm_opt: Option<Vec<f64>> = sample_weight.map(|sw| {
            let s: f64 = sw.iter().sum();
            sw.iter().map(|&v| v / s).collect()
        });
        let sw_slice = sw_norm_opt.as_deref();

        let y_mean: f64 = match sw_slice {
            Some(sw) => y.iter().zip(sw.iter()).map(|(&yi, &wi)| yi * wi).sum(),
            None => y.iter().sum::<f64>() * inv_n,
        };
        let backend = effective_backend(&self.parsed_device, n, p);

        let dim = p + usize::from(self.fit_intercept);
        let mut init_theta = vec![0.0; dim];
        if self.fit_intercept {
            init_theta[p] = match eff_link {
                GlmLink::Identity => y_mean,
                GlmLink::Log => y_mean.max(1e-12).ln(),
            };
        }

        let power = self.power;
        let power_kind = TweediePowerKind::from_power(power);
        let alpha = self.alpha;
        let fit_intercept = self.fit_intercept;
        let is_cpu = matches!(backend, Backend::Cpu);
        let check_pos_mu = eff_link == GlmLink::Identity && power > 0.0;

        let (theta, n_iter) = run_lbfgs(
            init_theta,
            self.max_iter,
            self.tol,
            |_| {},
            |t, grad_out| {
                let w = &t[..p];
                let b = if fit_intercept { t[p] } else { 0.0 };

                let (data_loss, grad_b) = if is_cpu {
                    if n >= FUSED_PAR_THRESHOLD {
                        let (dl, gb, gw, invalid) = x
                            .data
                            .par_chunks(FUSED_CHUNK_ROWS * p)
                            .zip(y.par_chunks(FUSED_CHUNK_ROWS))
                            .enumerate()
                            .fold(
                                || (0.0_f64, 0.0_f64, vec![0.0_f64; p], false),
                                |(mut dl, mut gb, mut gw, mut inv), (c_idx, (xc, yc))| {
                                    if inv {
                                        return (dl, gb, gw, true);
                                    }
                                    let base = c_idx * FUSED_CHUNK_ROWS;
                                    for i in 0..yc.len() {
                                        let row = &xc[i * p..(i + 1) * p];
                                        let eta = dot_unrolled(row, w) + b;
                                        let mui = match eff_link {
                                            GlmLink::Identity => eta,
                                            GlmLink::Log => eta.clamp(-50.0, 50.0).exp(),
                                        };
                                        if (check_pos_mu && mui <= 1e-15) || !mui.is_finite() {
                                            inv = true;
                                            break;
                                        }
                                        let wi = sw_slice.map_or(inv_n, |s| s[base + i]);
                                        let (l_i, g) =
                                            power_kind.loss_and_grad(eff_link, yc[i], mui, wi);
                                        dl += l_i;
                                        gb += g;
                                        axpy_add_unrolled(&mut gw, row, g);
                                    }
                                    (dl, gb, gw, inv)
                                },
                            )
                            .reduce(
                                || (0.0_f64, 0.0_f64, vec![0.0_f64; p], false),
                                |(dl1, gb1, mut gw1, inv1), (dl2, gb2, gw2, inv2)| {
                                    if inv1 || inv2 {
                                        return (0.0, 0.0, gw1, true);
                                    }
                                    for j in 0..p {
                                        gw1[j] += gw2[j];
                                    }
                                    (dl1 + dl2, gb1 + gb2, gw1, false)
                                },
                            );
                        if invalid {
                            return Ok(f64::INFINITY);
                        }
                        grad_out[..p].copy_from_slice(&gw);
                        (dl, gb)
                    } else {
                        grad_out[..p].fill(0.0);
                        let mut dl = 0.0;
                        let mut gb = 0.0;
                        for i in 0..n {
                            let row = &x.data[i * p..(i + 1) * p];
                            let eta = dot_unrolled(row, w) + b;
                            let mui = match eff_link {
                                GlmLink::Identity => eta,
                                GlmLink::Log => eta.clamp(-50.0, 50.0).exp(),
                            };
                            if (check_pos_mu && mui <= 1e-15) || !mui.is_finite() {
                                return Ok(f64::INFINITY);
                            }
                            let wi = sw_slice.map_or(inv_n, |s| s[i]);
                            let (l_i, g) = power_kind.loss_and_grad(eff_link, y[i], mui, wi);
                            dl += l_i;
                            gb += g;
                            axpy_add_unrolled(&mut grad_out[..p], row, g);
                        }
                        (dl, gb)
                    }
                } else {
                    let eta = compute_linear_eta(&backend, x, w, b)?;
                    let mu: Vec<f64> = match eff_link {
                        GlmLink::Identity => eta,
                        GlmLink::Log => {
                            if n >= SMALL_VEC_GPU_THRESHOLD {
                                let clamped: Vec<f64> =
                                    eta.iter().map(|&v| v.clamp(-50.0, 50.0)).collect();
                                let eta_mat = Matrix::from_vec(n, 1, clamped)?;
                                backend.elementwise_op(&eta_mat, ElementwiseOp::Exp)?.data
                            } else {
                                eta.iter().map(|&v| v.clamp(-50.0, 50.0).exp()).collect()
                            }
                        }
                    };

                    let mut dl = 0.0;
                    let mut grad_eta = vec![0.0; n];
                    let mut gb = 0.0;
                    for i in 0..n {
                        let mui = mu[i];
                        if (check_pos_mu && mui <= 1e-15) || !mui.is_finite() {
                            return Ok(f64::INFINITY);
                        }
                        let wi = sw_slice.map_or(inv_n, |s| s[i]);
                        let (l_i, g) = power_kind.loss_and_grad(eff_link, y[i], mui, wi);
                        dl += l_i;
                        grad_eta[i] = g;
                        gb += g;
                    }
                    let grad_w = compute_xt_r(&backend, x, &grad_eta)?;
                    grad_out[..p].copy_from_slice(&grad_w);
                    (dl, gb)
                };

                let mut reg_loss = 0.0;
                for j in 0..p {
                    let wj = w[j];
                    reg_loss += wj * wj;
                    grad_out[j] += alpha * wj;
                }
                if fit_intercept {
                    grad_out[p] = grad_b;
                }

                Ok(data_loss + 0.5 * alpha * reg_loss)
            },
        )?;

        self.coef_ = Some(theta[..p].to_vec());
        self.intercept_ = if self.fit_intercept { theta[p] } else { 0.0 };
        self.n_iter_ = n_iter;
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        validate_predict_inputs(x, self.n_features_in_)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let eff_link = resolve_glm_link(&self.link, self.power)?;
        let backend = effective_backend(&self.parsed_device, x.rows, x.cols);
        let eta = compute_linear_eta(&backend, x, coef, self.intercept_)?;

        match eff_link {
            GlmLink::Identity => Ok(eta),
            GlmLink::Log => {
                if x.rows >= SMALL_VEC_GPU_THRESHOLD && !matches!(backend, Backend::Cpu) {
                    let eta_mat = Matrix::from_vec(x.rows, 1, eta)?;
                    Ok(backend.elementwise_op(&eta_mat, ElementwiseOp::Exp)?.data)
                } else {
                    let mut out = eta;
                    for v in &mut out {
                        *v = v.exp();
                    }
                    Ok(out)
                }
            }
        }
    }

    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        tweedie_d2_score(y, &preds, self.power, sample_weight)
    }

    fn to_bytes_with_tag(&self, tag: u8) -> Vec<u8> {
        let mut buf = write_header(tag);
        write_f64(&mut buf, self.power);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_str(&mut buf, &self.link);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_f64(&mut buf, self.intercept_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    fn from_bytes_with_tag(
        bytes: &[u8],
        expected_tag: u8,
        device: Option<&str>,
    ) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, expected_tag)?;
        let power = reader.read_f64()?;
        let alpha = reader.read_f64()?;
        let fit_intercept = reader.read_bool()?;
        let link = reader.read_str()?;
        let max_iter = reader.read_usize()?;
        let tol = reader.read_f64()?;
        let coef_ = reader.read_opt_f64_vec()?;
        let intercept_ = reader.read_f64()?;
        let n_iter_ = reader.read_usize()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let saved_device = reader.read_opt_str()?;
        let (parsed_device, device_) =
            resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            power,
            alpha,
            fit_intercept,
            link,
            max_iter,
            tol,
            coef_,
            intercept_,
            n_iter_,
            n_features_in_,
            device_,
            parsed_device,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_with_tag(TAG_TWEEDIE_REGRESSOR)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        Self::from_bytes_with_tag(bytes, TAG_TWEEDIE_REGRESSOR, device)
    }
}

#[pyclass(from_py_object, name = "TweedieRegressor")]
#[derive(Clone)]
pub struct PyTweedieRegressor {
    inner: TweedieRegressor,
}

pymethods_with_dit!(PyTweedieRegressor, TweedieRegressor, {
    #[new]
    #[pyo3(signature = (power = 0.0, alpha = 1.0, fit_intercept = true, link = "auto".to_string(), max_iter = 100, tol = 1e-4, device = None))]
    pub fn new(
        power: f64,
        alpha: f64,
        fit_intercept: bool,
        link: String,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TweedieRegressor::new(power, alpha, fit_intercept, link, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &y_vec, sw_vec.as_deref())?)
    }

    #[getter]
    pub fn power(&self) -> f64 {
        self.inner.power
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn link(&self) -> String {
        self.inner.link.clone()
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> f64 {
        self.inner.intercept_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

// ============================================================================
// 4. PoissonRegressor (TAG = 51) & 5. GammaRegressor (TAG = 52)
// ============================================================================

#[derive(Debug, Clone)]
pub struct PoissonRegressor {
    inner: TweedieRegressor,
    pub device_: Option<String>,
}

impl PoissonRegressor {
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let inner = TweedieRegressor::new(
            1.0,
            alpha,
            fit_intercept,
            "log".to_string(),
            max_iter,
            tol,
            device,
        )?;
        let device_ = inner.device_.clone();
        Ok(Self { inner, device_ })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.inner.set_device(device)?;
        self.device_ = self.inner.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.inner.fit(x, y, sample_weight)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.inner.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        self.inner.score(x, y, sample_weight)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes_with_tag(TAG_POISSON_REGRESSOR)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let inner = TweedieRegressor::from_bytes_with_tag(bytes, TAG_POISSON_REGRESSOR, device)?;
        let device_ = inner.device_.clone();
        Ok(Self { inner, device_ })
    }
}

#[pyclass(from_py_object, name = "PoissonRegressor")]
#[derive(Clone)]
pub struct PyPoissonRegressor {
    inner: PoissonRegressor,
}

pymethods_with_dit!(PyPoissonRegressor, PoissonRegressor, {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, max_iter = 100, tol = 1e-4, device = None))]
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PoissonRegressor::new(alpha, fit_intercept, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &y_vec, sw_vec.as_deref())?)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.inner.tol
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> f64 {
        self.inner.inner.intercept_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.inner.n_features_in_
    }
});

#[derive(Debug, Clone)]
pub struct GammaRegressor {
    inner: TweedieRegressor,
    pub device_: Option<String>,
}

impl GammaRegressor {
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let inner = TweedieRegressor::new(
            2.0,
            alpha,
            fit_intercept,
            "log".to_string(),
            max_iter,
            tol,
            device,
        )?;
        let device_ = inner.device_.clone();
        Ok(Self { inner, device_ })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.inner.set_device(device)?;
        self.device_ = self.inner.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.inner.fit(x, y, sample_weight)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.inner.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        self.inner.score(x, y, sample_weight)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes_with_tag(TAG_GAMMA_REGRESSOR)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let inner = TweedieRegressor::from_bytes_with_tag(bytes, TAG_GAMMA_REGRESSOR, device)?;
        let device_ = inner.device_.clone();
        Ok(Self { inner, device_ })
    }
}

#[pyclass(from_py_object, name = "GammaRegressor")]
#[derive(Clone)]
pub struct PyGammaRegressor {
    inner: GammaRegressor,
}

pymethods_with_dit!(PyGammaRegressor, GammaRegressor, {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, max_iter = 100, tol = 1e-4, device = None))]
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GammaRegressor::new(alpha, fit_intercept, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = match sample_weight {
            Some(sw) => Some(buffer_to_f64_1d(py, &sw)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &y_vec, sw_vec.as_deref())?)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.inner.tol
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> f64 {
        self.inner.inner.intercept_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHuberRegressor>()?;
    m.add_class::<PyQuantileRegressor>()?;
    m.add_class::<PyPoissonRegressor>()?;
    m.add_class::<PyGammaRegressor>()?;
    m.add_class::<PyTweedieRegressor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(n: usize, p: usize) -> (Matrix, Vec<f64>) {
        let mut x_data = Vec::with_capacity(n * p);
        let mut y = Vec::with_capacity(n);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_f64 = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            u * 2.0 - 1.0
        };
        for i in 0..n {
            let mut lin = 1.5;
            for j in 0..p {
                let v = next_f64();
                x_data.push(v);
                lin += ((j + 1) as f64 * 0.5) * v;
            }
            let noise = 0.05 * next_f64();
            let outlier = if i % 50 == 0 { 25.0 } else { 0.0 };
            y.push(lin + noise + outlier);
        }
        (Matrix::from_vec(n, p, x_data).unwrap(), y)
    }

    #[test]
    fn test_huber_small_and_large_fused() {
        for n in [200, 5000] {
            let (x, y) = synthetic_data(n, 6);
            let mut huber =
                HuberRegressor::new(1.35, 100, 1e-4, true, 1e-5, Some("cpu".to_string())).unwrap();
            huber.fit(&x, &y, None).unwrap();
            let coef = huber.coef_.as_ref().unwrap();
            assert!((coef[0] - 0.5).abs() < 0.1, "coef[0]={}", coef[0]);
            assert!((coef[1] - 1.0).abs() < 0.1, "coef[1]={}", coef[1]);
            assert!(
                (huber.intercept_ - 1.5).abs() < 0.15,
                "intercept={}",
                huber.intercept_
            );
            assert!(
                huber.n_iter_ <= 40,
                "Expected warm-started Huber to converge quickly, took {} iters",
                huber.n_iter_
            );
            let bytes = huber.to_bytes();
            let loaded = HuberRegressor::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), huber.predict(&x).unwrap());
        }

        // Large-N (20k) convergence check: normalized L-BFGS stopping should converge well under max_iter
        let (x_big, y_big) = synthetic_data(20_000, 8);
        let mut huber_big =
            HuberRegressor::new(1.35, 100, 1e-4, true, 1e-5, Some("cpu".to_string())).unwrap();
        huber_big.fit(&x_big, &y_big, None).unwrap();
        let coef_big = huber_big.coef_.as_ref().unwrap();
        assert!((coef_big[0] - 0.5).abs() < 0.02, "coef[0]={}", coef_big[0]);
        assert!((coef_big[1] - 1.0).abs() < 0.02, "coef[1]={}", coef_big[1]);
        assert!(
            huber_big.n_iter_ <= 25,
            "Expected normalized Huber L-BFGS on 20k rows to converge in <=25 iters, took {}",
            huber_big.n_iter_
        );
    }

    #[test]
    fn test_glms_and_quantile_fused() {
        let n = 4500;
        let p = 4;
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_pos = Vec::with_capacity(n);
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next_f64 = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        for _ in 0..n {
            let mut eta = 0.5;
            for j in 0..p {
                let v = next_f64();
                x_data.push(v);
                eta += 0.3 * ((j + 1) as f64) * v;
            }
            y_pos.push(eta.exp());
        }
        let x = Matrix::from_vec(n, p, x_data).unwrap();

        let mut pois =
            PoissonRegressor::new(1e-4, true, 100, 1e-4, Some("cpu".to_string())).unwrap();
        pois.fit(&x, &y_pos, None).unwrap();
        assert!(pois.score(&x, &y_pos, None).unwrap() > 0.99);

        let mut gamma =
            GammaRegressor::new(1e-4, true, 100, 1e-4, Some("cpu".to_string())).unwrap();
        gamma.fit(&x, &y_pos, None).unwrap();
        assert!(gamma.score(&x, &y_pos, None).unwrap() > 0.99);

        let mut tweedie = TweedieRegressor::new(
            1.5,
            1e-4,
            true,
            "log".to_string(),
            100,
            1e-4,
            Some("cpu".to_string()),
        )
        .unwrap();
        tweedie.fit(&x, &y_pos, None).unwrap();
        assert!(tweedie.score(&x, &y_pos, None).unwrap() > 0.99);

        let mut quant =
            QuantileRegressor::new(0.5, 0.0, true, 100, 1e-5, Some("cpu".to_string())).unwrap();
        quant.fit(&x, &y_pos, None).unwrap();
        assert!(quant.score(&x, &y_pos, None).unwrap() > 0.90);
    }
}
