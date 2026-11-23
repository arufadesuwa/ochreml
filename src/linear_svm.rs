// OchreML Linear Support Vector Machines Module (＾▽＾)
// Implements LinearSVC and LinearSVR with LIBLINEAR-style Dual Coordinate Descent,
// Primal L-BFGS / Coordinate Descent, Rayon OvR multiclass, and Backend primitives.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_u64,
    write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{dot_unrolled, predict_linear_1d, predict_linear_multi};
use crate::logistic_regression::{
    build_effective_weights, extract_sorted_classes, map_labels_to_indices, validate_sample_weights,
};
use crate::matrix::{Matrix, r2_score};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::primitives::ElementwiseOp;
use crate::primitives::stats_ops::apply_scalar_op;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_LINEAR_SVC: u8 = 54;
pub const TAG_LINEAR_SVR: u8 = 55;

const LBFGS_HISTORY: usize = 10;
const LARGE_MATRIX_THRESHOLD: usize = 262_144;

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle(&mut self, slice: &mut [usize]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

fn validate_finite_matrix(x: &Matrix) -> OchreResult<()> {
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

fn validate_finite_y(y: &[f64], expected_rows: usize) -> OchreResult<()> {
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

fn validate_c(c: f64) -> OchreResult<()> {
    if !c.is_finite() || c <= 0.0 {
        return Err(OchreError::InvalidC(c));
    }
    Ok(())
}

fn validate_epsilon(epsilon: f64) -> OchreResult<()> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "epsilon ({epsilon}) must be a non-negative finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_tol(tol: f64) -> OchreResult<()> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be a non-negative finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_max_iter(max_iter: usize) -> OchreResult<()> {
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    Ok(())
}

fn validate_intercept_scaling(intercept_scaling: f64) -> OchreResult<()> {
    if !intercept_scaling.is_finite() || intercept_scaling <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "intercept_scaling ({intercept_scaling}) must be a strictly positive finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn lbfgs_two_loop(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let m = s_hist.len();
    let dim = grad.len();
    let mut q = grad.to_vec();
    if m == 0 {
        for v in &mut q {
            *v = -*v;
        }
        return q;
    }

    let mut alpha = vec![0.0; m];
    for i in (0..m).rev() {
        let a = rho_hist[i] * dot_unrolled(&s_hist[i], &q);
        alpha[i] = a;
        for d in 0..dim {
            q[d] -= a * y_hist[i][d];
        }
    }

    let yy = dot_unrolled(&y_hist[m - 1], &y_hist[m - 1]);
    let sy = dot_unrolled(&s_hist[m - 1], &y_hist[m - 1]);
    let gamma = if yy > 1e-20 && sy > 0.0 { sy / yy } else { 1.0 };

    let mut r = vec![0.0; dim];
    for d in 0..dim {
        r[d] = gamma * q[d];
    }

    for i in 0..m {
        let beta = rho_hist[i] * dot_unrolled(&y_hist[i], &r);
        let coeff = alpha[i] - beta;
        for d in 0..dim {
            r[d] += coeff * s_hist[i][d];
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

#[derive(Clone, Copy)]
struct SvmSolveConfig {
    c: f64,
    tol: f64,
    max_iter: usize,
    fit_intercept: bool,
    intercept_scaling: f64,
    random_state: Option<u64>,
}

/// LIBLINEAR Dual Coordinate Descent for L2-regularized L1-loss / L2-loss SVC.
fn solve_svc_dual_l2(
    x: &Matrix,
    y_pm1: &[f64],
    sample_weights: &[f64],
    loss: &str,
    cfg: SvmSolveConfig,
) -> (Vec<f64>, f64, usize) {
    let n = x.rows;
    let p = x.cols;
    let b_scale = if cfg.fit_intercept {
        cfg.intercept_scaling
    } else {
        0.0
    };
    let b_sq = b_scale * b_scale;
    let is_l2_loss = loss == "squared_hinge";

    let mut w = vec![0.0; p];
    let mut w_bias = 0.0_f64;
    let mut alpha = vec![0.0; n];
    let mut qd = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut diag = vec![0.0; n];

    for i in 0..n {
        let row = &x.data[i * p..(i + 1) * p];
        let x_sq = dot_unrolled(row, row) + b_sq;
        let ci = cfg.c * sample_weights[i];
        if ci <= 0.0 {
            diag[i] = 0.0;
            upper[i] = 0.0;
            qd[i] = x_sq.max(1e-12);
        } else if is_l2_loss {
            let d = 0.5 / ci;
            diag[i] = d;
            upper[i] = f64::INFINITY;
            qd[i] = (x_sq + d).max(1e-12);
        } else {
            diag[i] = 0.0;
            upper[i] = ci;
            qd[i] = x_sq.max(1e-12);
        }
    }

    let mut index: Vec<usize> = (0..n).filter(|&i| sample_weights[i] > 0.0).collect();
    let mut active_size = index.len();
    let mut rng = SplitMix64::new(cfg.random_state);

    let mut pg_max_old = f64::INFINITY;
    let mut pg_min_old = f64::NEG_INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;
        let mut pg_max_new = f64::NEG_INFINITY;
        let mut pg_min_new = f64::INFINITY;

        rng.shuffle(&mut index[..active_size]);

        let mut s = 0usize;
        while s < active_size {
            let i = index[s];
            let yi = y_pm1[i];
            let row = &x.data[i * p..(i + 1) * p];
            let wx = dot_unrolled(&w, row) + w_bias * b_scale;
            let g = yi * wx - 1.0 + diag[i] * alpha[i];

            let mut pg = 0.0;
            let u = upper[i];
            let a_i = alpha[i];

            if a_i == 0.0 {
                if g > pg_max_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g < 0.0 {
                    pg = g;
                }
            } else if a_i == u {
                if g < pg_min_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g > 0.0 {
                    pg = g;
                }
            } else {
                pg = g;
            }

            pg_max_new = pg_max_new.max(pg);
            pg_min_new = pg_min_new.min(pg);

            if pg.abs() > 1e-12 {
                let alpha_old = a_i;
                let alpha_new = (a_i - g / qd[i]).clamp(0.0, u);
                alpha[i] = alpha_new;
                let d = (alpha_new - alpha_old) * yi;
                if d != 0.0 {
                    for (wj, &xij) in w.iter_mut().zip(row.iter()) {
                        *wj += d * xij;
                    }
                    w_bias += d * b_scale;
                }
            }

            s += 1;
        }

        if pg_max_new - pg_min_new <= cfg.tol {
            if active_size == index.len() {
                break;
            }
            active_size = index.len();
            pg_max_old = f64::INFINITY;
            pg_min_old = f64::NEG_INFINITY;
        } else {
            pg_max_old = if pg_max_new <= 0.0 {
                f64::INFINITY
            } else {
                pg_max_new
            };
            pg_min_old = if pg_min_new >= 0.0 {
                f64::NEG_INFINITY
            } else {
                pg_min_new
            };
        }
    }

    (w, w_bias * b_scale, n_iter)
}

/// Primal L-BFGS for L2-regularized squared-hinge SVC.
fn solve_svc_primal_l2(
    backend: &Backend<'_>,
    x: &Matrix,
    y_pm1: &[f64],
    sample_weights: &[f64],
    cfg: SvmSolveConfig,
) -> OchreResult<(Vec<f64>, f64, usize)> {
    let n = x.rows;
    let p = x.cols;
    let b_scale = if cfg.fit_intercept {
        cfg.intercept_scaling
    } else {
        0.0
    };
    let dim = p + usize::from(cfg.fit_intercept);
    let use_backend = n.saturating_mul(p) >= LARGE_MATRIX_THRESHOLD && p > 64;

    let (theta, n_iter) = run_lbfgs(dim, n, cfg.max_iter, cfg.tol, |th, grad_out| {
        let w = &th[..p];
        let w_bias = if cfg.fit_intercept { th[p] } else { 0.0 };

        let mut data_loss = 0.0;
        if use_backend {
            let mut scores = backend.matvec(x, w)?;
            if cfg.fit_intercept {
                let shift = w_bias * b_scale;
                let s_mat = Matrix::from_vec(n, 1, scores)?;
                let shifted =
                    backend.elementwise_op(&s_mat, ElementwiseOp::Affine { scale: 1.0, shift })?;
                scores = shifted.data;
            }
            let mut coeff = vec![0.0; n];
            let mut grad_b = 0.0;
            for i in 0..n {
                let yi = y_pm1[i];
                let m = 1.0 - yi * scores[i];
                if m > 0.0 {
                    let ci = cfg.c * sample_weights[i];
                    data_loss += ci * m * m;
                    let g_i = -2.0 * ci * yi * m;
                    coeff[i] = g_i;
                    grad_b += g_i * b_scale;
                }
            }
            let c_mat = Matrix::from_vec(n, 1, coeff)?;
            let gw_mat = backend.matmul_at_b(x, &c_mat)?;
            for j in 0..p {
                grad_out[j] = w[j] + gw_mat.data[j];
            }
            if cfg.fit_intercept {
                grad_out[p] = w_bias + grad_b;
            }
        } else {
            grad_out.fill(0.0);
            for i in 0..n {
                let row = &x.data[i * p..(i + 1) * p];
                let score = dot_unrolled(w, row) + w_bias * b_scale;
                let yi = y_pm1[i];
                let m = 1.0 - yi * score;
                if m > 0.0 {
                    let ci = cfg.c * sample_weights[i];
                    data_loss += ci * m * m;
                    let g_i = -2.0 * ci * yi * m;
                    for (gj, &xij) in grad_out[..p].iter_mut().zip(row.iter()) {
                        *gj += g_i * xij;
                    }
                    if cfg.fit_intercept {
                        grad_out[p] += g_i * b_scale;
                    }
                }
            }
            for j in 0..p {
                grad_out[j] += w[j];
            }
            if cfg.fit_intercept {
                grad_out[p] += w_bias;
            }
        }

        let reg_loss = 0.5 * (dot_unrolled(w, w) + w_bias * w_bias);
        Ok(reg_loss + data_loss)
    })?;

    let w = theta[..p].to_vec();
    let intercept = if cfg.fit_intercept {
        theta[p] * b_scale
    } else {
        0.0
    };
    Ok((w, intercept, n_iter))
}

/// Primal Coordinate Descent for L1-regularized squared-hinge SVC.
fn solve_svc_primal_l1(
    x: &Matrix,
    y_pm1: &[f64],
    sample_weights: &[f64],
    cfg: SvmSolveConfig,
) -> (Vec<f64>, f64, usize) {
    let n = x.rows;
    let p = x.cols;
    let b_scale = if cfg.fit_intercept {
        cfg.intercept_scaling
    } else {
        0.0
    };
    let dim = p + usize::from(cfg.fit_intercept);

    // Transpose X into column-major for fast coordinate updates
    let mut cols_data = vec![0.0; dim * n];
    for i in 0..n {
        let row = &x.data[i * p..(i + 1) * p];
        for (j, &val) in row.iter().enumerate() {
            cols_data[j * n + i] = val;
        }
        if cfg.fit_intercept {
            cols_data[p * n + i] = b_scale;
        }
    }

    let mut w = vec![0.0; dim];
    // b_margin[i] = 1 - y_i * w^T x_i (initially 1.0 since w = 0)
    let mut b_margin = vec![1.0; n];
    let mut index: Vec<usize> = (0..dim).collect();
    let mut rng = SplitMix64::new(cfg.random_state);
    let mut n_iter = 0usize;

    let compute_loss = |w_vec: &[f64], margins: &[f64]| -> f64 {
        let mut l1 = 0.0;
        for &wj in w_vec {
            l1 += wj.abs();
        }
        let mut hinge_sq = 0.0;
        for i in 0..n {
            if margins[i] > 0.0 {
                hinge_sq += cfg.c * sample_weights[i] * margins[i] * margins[i];
            }
        }
        l1 + hinge_sq
    };

    let mut current_loss = compute_loss(&w, &b_margin);

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;
        rng.shuffle(&mut index);
        let mut max_delta = 0.0_f64;

        for &j in &index {
            let col = &cols_data[j * n..(j + 1) * n];
            let mut g = 0.0;
            let mut h = 0.0;
            for i in 0..n {
                if b_margin[i] > 0.0 {
                    let ci = cfg.c * sample_weights[i];
                    let yx = y_pm1[i] * col[i];
                    g -= 2.0 * ci * b_margin[i] * yx;
                    h += 2.0 * ci * yx * yx;
                }
            }
            if h <= 1e-12 {
                continue;
            }
            let z = w[j] - g / h;
            let w_target = apply_scalar_op(z, ElementwiseOp::SoftThreshold { lambda: 1.0 / h });
            let d = w_target - w[j];
            if d.abs() < 1e-14 {
                continue;
            }

            // Backtracking line search along coordinate j
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..10 {
                let cand_d = step * d;
                let w_cand = w[j] + cand_d;
                let mut l1_delta = w_cand.abs() - w[j].abs();
                let mut hinge_new = 0.0;
                let mut hinge_old = 0.0;
                for i in 0..n {
                    let ci = cfg.c * sample_weights[i];
                    if b_margin[i] > 0.0 {
                        hinge_old += ci * b_margin[i] * b_margin[i];
                    }
                    let m_new = b_margin[i] - cand_d * y_pm1[i] * col[i];
                    if m_new > 0.0 {
                        hinge_new += ci * m_new * m_new;
                    }
                }
                l1_delta += hinge_new - hinge_old;
                if l1_delta <= 0.0 {
                    w[j] = w_cand;
                    for i in 0..n {
                        b_margin[i] -= cand_d * y_pm1[i] * col[i];
                    }
                    current_loss += l1_delta;
                    max_delta = max_delta.max(cand_d.abs());
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            let _ = accepted;
        }

        if max_delta <= cfg.tol * 0.1 || current_loss < 1e-15 {
            break;
        }
    }

    let coef = w[..p].to_vec();
    let intercept = if cfg.fit_intercept {
        w[p] * b_scale
    } else {
        0.0
    };
    (coef, intercept, n_iter)
}

/// LIBLINEAR Dual Coordinate Descent for L1/L2-loss epsilon-insensitive SVR.
fn solve_svr_dual(
    x: &Matrix,
    y: &[f64],
    sample_weights: &[f64],
    epsilon: f64,
    loss: &str,
    cfg: SvmSolveConfig,
) -> (Vec<f64>, f64, usize) {
    let n = x.rows;
    let p = x.cols;
    let b_scale = if cfg.fit_intercept {
        cfg.intercept_scaling
    } else {
        0.0
    };
    let b_sq = b_scale * b_scale;
    let is_l2_loss = loss == "squared_epsilon_insensitive";

    let mut w = vec![0.0; p];
    let mut w_bias = 0.0_f64;
    let mut beta = vec![0.0; n];
    let mut qd = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lambda = vec![0.0; n];

    for i in 0..n {
        let row = &x.data[i * p..(i + 1) * p];
        let x_sq = dot_unrolled(row, row) + b_sq;
        let ci = cfg.c * sample_weights[i];
        if ci <= 0.0 {
            lambda[i] = 0.0;
            upper[i] = 0.0;
            qd[i] = x_sq.max(1e-12);
        } else if is_l2_loss {
            let lam = 0.5 / ci;
            lambda[i] = lam;
            upper[i] = f64::INFINITY;
            qd[i] = (x_sq + lam).max(1e-12);
        } else {
            lambda[i] = 0.0;
            upper[i] = ci;
            qd[i] = x_sq.max(1e-12);
        }
    }

    let mut index: Vec<usize> = (0..n).filter(|&i| sample_weights[i] > 0.0).collect();
    let mut rng = SplitMix64::new(cfg.random_state);
    let mut n_iter = 0usize;

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;
        rng.shuffle(&mut index);
        let mut max_vio = 0.0_f64;

        for &i in &index {
            let row = &x.data[i * p..(i + 1) * p];
            let wx = dot_unrolled(&w, row) + w_bias * b_scale;
            let g = wx - y[i] + lambda[i] * beta[i];
            let gp = g + epsilon;
            let gm = g - epsilon;

            let b_old = beta[i];
            let u = upper[i];
            let vio = if b_old == 0.0 {
                (-gp).max(gm).max(0.0)
            } else if b_old > 0.0 {
                if b_old >= u { gp.max(0.0) } else { gp.abs() }
            } else if b_old <= -u {
                (-gm).max(0.0)
            } else {
                gm.abs()
            };

            max_vio = max_vio.max(vio);
            if vio <= 1e-12 {
                continue;
            }

            let z = b_old - g / qd[i];
            let thresh = epsilon / qd[i];
            let b_new =
                apply_scalar_op(z, ElementwiseOp::SoftThreshold { lambda: thresh }).clamp(-u, u);
            let d = b_new - b_old;
            if d != 0.0 {
                beta[i] = b_new;
                for (wj, &xij) in w.iter_mut().zip(row.iter()) {
                    *wj += d * xij;
                }
                w_bias += d * b_scale;
            }
        }

        if max_vio <= cfg.tol {
            break;
        }
    }

    (w, w_bias * b_scale, n_iter)
}

/// Primal L-BFGS for squared epsilon-insensitive SVR.
fn solve_svr_primal_l2(
    backend: &Backend<'_>,
    x: &Matrix,
    y: &[f64],
    sample_weights: &[f64],
    epsilon: f64,
    cfg: SvmSolveConfig,
) -> OchreResult<(Vec<f64>, f64, usize)> {
    let n = x.rows;
    let p = x.cols;
    let b_scale = if cfg.fit_intercept {
        cfg.intercept_scaling
    } else {
        0.0
    };
    let dim = p + usize::from(cfg.fit_intercept);
    let use_backend = n.saturating_mul(p) >= LARGE_MATRIX_THRESHOLD && p > 64;

    let (theta, n_iter) = run_lbfgs(dim, n, cfg.max_iter, cfg.tol, |th, grad_out| {
        let w = &th[..p];
        let w_bias = if cfg.fit_intercept { th[p] } else { 0.0 };
        let mut data_loss = 0.0;

        if use_backend {
            let mut preds = backend.matvec(x, w)?;
            if cfg.fit_intercept {
                let shift = w_bias * b_scale;
                let p_mat = Matrix::from_vec(n, 1, preds)?;
                let shifted =
                    backend.elementwise_op(&p_mat, ElementwiseOp::Affine { scale: 1.0, shift })?;
                preds = shifted.data;
            }
            let mut coeff = vec![0.0; n];
            let mut grad_b = 0.0;
            for i in 0..n {
                let r = preds[i] - y[i];
                let abs_r = r.abs();
                if abs_r > epsilon {
                    let e = abs_r - epsilon;
                    let ci = cfg.c * sample_weights[i];
                    data_loss += ci * e * e;
                    let g_i = 2.0 * ci * r.signum() * e;
                    coeff[i] = g_i;
                    grad_b += g_i * b_scale;
                }
            }
            let c_mat = Matrix::from_vec(n, 1, coeff)?;
            let gw_mat = backend.matmul_at_b(x, &c_mat)?;
            for j in 0..p {
                grad_out[j] = w[j] + gw_mat.data[j];
            }
            if cfg.fit_intercept {
                grad_out[p] = w_bias + grad_b;
            }
        } else {
            grad_out.fill(0.0);
            for i in 0..n {
                let row = &x.data[i * p..(i + 1) * p];
                let pred = dot_unrolled(w, row) + w_bias * b_scale;
                let r = pred - y[i];
                let abs_r = r.abs();
                if abs_r > epsilon {
                    let e = abs_r - epsilon;
                    let ci = cfg.c * sample_weights[i];
                    data_loss += ci * e * e;
                    let g_i = 2.0 * ci * r.signum() * e;
                    for (gj, &xij) in grad_out[..p].iter_mut().zip(row.iter()) {
                        *gj += g_i * xij;
                    }
                    if cfg.fit_intercept {
                        grad_out[p] += g_i * b_scale;
                    }
                }
            }
            for j in 0..p {
                grad_out[j] += w[j];
            }
            if cfg.fit_intercept {
                grad_out[p] += w_bias;
            }
        }

        let reg_loss = 0.5 * (dot_unrolled(w, w) + w_bias * w_bias);
        Ok(reg_loss + data_loss)
    })?;

    let w = theta[..p].to_vec();
    let intercept = if cfg.fit_intercept {
        theta[p] * b_scale
    } else {
        0.0
    };
    Ok((w, intercept, n_iter))
}

/// Linear Support Vector Classification (LinearSVC) (*≧ω≦*).
#[derive(Debug, Clone)]
pub struct LinearSVC {
    pub c: f64,
    pub loss: String,
    pub penalty: String,
    pub dual: bool,
    pub tol: f64,
    pub fit_intercept: bool,
    pub intercept_scaling: f64,
    pub class_weight_mode: Option<String>,
    pub class_weight_map: Option<Vec<(f64, f64)>>,
    pub max_iter: usize,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub classes_: Vec<f64>,
    pub n_features_in_: Option<usize>,
    pub n_iter_: usize,
}

impl LinearSVC {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: f64,
        loss: String,
        penalty: String,
        dual: bool,
        tol: f64,
        fit_intercept: bool,
        intercept_scaling: f64,
        class_weight_mode: Option<String>,
        class_weight_map: Option<Vec<(f64, f64)>>,
        max_iter: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_c(c)?;
        validate_tol(tol)?;
        validate_max_iter(max_iter)?;
        validate_intercept_scaling(intercept_scaling)?;
        if loss != "hinge" && loss != "squared_hinge" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid loss '{loss}'. Expected 'hinge' or 'squared_hinge' (・`ω´・)"
            )));
        }
        if penalty != "l1" && penalty != "l2" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid penalty '{penalty}'. Expected 'l1' or 'l2' (・`ω´・)"
            )));
        }
        if penalty == "l1" && loss == "hinge" {
            return Err(OchreError::InvalidParameter(
                "Unsupported set of arguments: penalty='l1' and loss='hinge' are incompatible (・`ω´・)"
                    .to_string(),
            ));
        }
        if penalty == "l1" && dual {
            return Err(OchreError::InvalidParameter(
                "Unsupported set of arguments: penalty='l1' requires dual=False (・`ω´・)"
                    .to_string(),
            ));
        }
        if loss == "hinge" && !dual {
            return Err(OchreError::InvalidParameter(
                "Unsupported set of arguments: loss='hinge' requires dual=True (・`ω´・)"
                    .to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            c,
            loss,
            penalty,
            dual,
            tol,
            fit_intercept,
            intercept_scaling,
            class_weight_mode,
            class_weight_map,
            max_iter,
            random_state,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            classes_: Vec::new(),
            n_features_in_: None,
            n_iter_: 0,
        })
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted(x, y, None)
    }

    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        validate_finite_y(y, x.rows)?;
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

        let cfg = SvmSolveConfig {
            c: self.c,
            tol: self.tol,
            max_iter: self.max_iter,
            fit_intercept: self.fit_intercept,
            intercept_scaling: self.intercept_scaling,
            random_state: self.random_state,
        };
        let backend = get_backend(&self.device);
        let p = x.cols;

        if n_classes == 2 {
            let y_pm1: Vec<f64> = y_idx
                .iter()
                .map(|&idx| if idx == 1 { 1.0 } else { -1.0 })
                .collect();
            let (w, b, n_iter) = self.solve_binary(&backend, x, &y_pm1, &weights, cfg)?;
            self.coef_ = Some(Matrix::from_vec(1, p, w)?);
            self.intercept_ = Some(vec![b]);
            self.n_iter_ = n_iter;
        } else {
            let results: Vec<OchreResult<(Vec<f64>, f64, usize)>> = (0..n_classes)
                .into_par_iter()
                .map(|cls_idx| {
                    let y_pm1: Vec<f64> = y_idx
                        .iter()
                        .map(|&idx| if idx == cls_idx { 1.0 } else { -1.0 })
                        .collect();
                    self.solve_binary(&backend, x, &y_pm1, &weights, cfg)
                })
                .collect();

            let mut coef_data = Vec::with_capacity(n_classes * p);
            let mut intercepts = Vec::with_capacity(n_classes);
            let mut max_n_iter = 0usize;
            for res in results {
                let (w, b, ni) = res?;
                coef_data.extend_from_slice(&w);
                intercepts.push(b);
                max_n_iter = max_n_iter.max(ni);
            }
            self.coef_ = Some(Matrix::from_vec(n_classes, p, coef_data)?);
            self.intercept_ = Some(intercepts);
            self.n_iter_ = max_n_iter;
        }

        self.classes_ = classes;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn solve_binary(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_pm1: &[f64],
        weights: &[f64],
        cfg: SvmSolveConfig,
    ) -> OchreResult<(Vec<f64>, f64, usize)> {
        if self.penalty == "l1" {
            Ok(solve_svc_primal_l1(x, y_pm1, weights, cfg))
        } else if self.dual {
            Ok(solve_svc_dual_l2(x, y_pm1, weights, &self.loss, cfg))
        } else {
            solve_svc_primal_l2(backend, x, y_pm1, weights, cfg)
        }
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x)?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        if coef.rows == 1 {
            let scores = predict_linear_1d(x, &coef.data, intercept[0])?;
            Matrix::from_vec(x.rows, 1, scores)
        } else {
            predict_linear_multi(x, coef, intercept)
        }
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let scores = self.decision_function(x)?;
        let n = scores.rows;
        let k = scores.cols;
        let mut preds = Vec::with_capacity(n);
        if k == 1 {
            let neg = self.classes_[0];
            let pos = self.classes_[1];
            for &s in &scores.data {
                preds.push(if s > 0.0 { pos } else { neg });
            }
        } else {
            for r in 0..n {
                let row = &scores.data[r * k..(r + 1) * k];
                let mut best_idx = 0usize;
                let mut best_val = row[0];
                for (c, &val) in row.iter().enumerate().skip(1) {
                    if val > best_val {
                        best_val = val;
                        best_idx = c;
                    }
                }
                preds.push(self.classes_[best_idx]);
            }
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_finite_y(y, x.rows)?;
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|&(&p, &t)| (p - t).abs() < 1e-12)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LINEAR_SVC);
        write_f64(&mut buf, self.c);
        write_str(&mut buf, &self.loss);
        write_str(&mut buf, &self.penalty);
        write_bool(&mut buf, self.dual);
        write_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.fit_intercept);
        write_f64(&mut buf, self.intercept_scaling);
        write_opt_str(&mut buf, &self.class_weight_mode);
        match &self.class_weight_map {
            Some(map) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, map.len());
                for &(k, v) in map {
                    write_f64(&mut buf, k);
                    write_f64(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_usize(&mut buf, self.max_iter);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.coef_);
        write_opt_f64_vec(&mut buf, &self.intercept_);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_iter_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LINEAR_SVC)?;
        let c = r.read_f64()?;
        let loss = r.read_str()?;
        let penalty = r.read_str()?;
        let dual = r.read_bool()?;
        let tol = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let intercept_scaling = r.read_f64()?;
        let class_weight_mode = r.read_opt_str()?;
        let class_weight_map = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut m = Vec::with_capacity(len);
            for _ in 0..len {
                m.push((r.read_f64()?, r.read_f64()?));
            }
            Some(m)
        } else {
            None
        };
        let max_iter = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_matrix()?;
        let intercept_ = r.read_opt_f64_vec()?;
        let classes_ = r.read_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_iter_ = r.read_usize()?;

        let (dev, dev_str) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;
        Ok(Self {
            c,
            loss,
            penalty,
            dual,
            tol,
            fit_intercept,
            intercept_scaling,
            class_weight_mode,
            class_weight_map,
            max_iter,
            random_state,
            device: dev,
            device_: dev_str,
            coef_,
            intercept_,
            classes_,
            n_features_in_,
            n_iter_,
        })
    }
}

/// Linear Support Vector Regression (LinearSVR) (*≧ω≦*).
#[derive(Debug, Clone)]
pub struct LinearSVR {
    pub epsilon: f64,
    pub tol: f64,
    pub c: f64,
    pub loss: String,
    pub fit_intercept: bool,
    pub intercept_scaling: f64,
    pub dual: bool,
    pub max_iter: usize,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub n_features_in_: Option<usize>,
    pub n_iter_: usize,
}

impl LinearSVR {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epsilon: f64,
        tol: f64,
        c: f64,
        loss: String,
        fit_intercept: bool,
        intercept_scaling: f64,
        dual: bool,
        max_iter: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_epsilon(epsilon)?;
        validate_tol(tol)?;
        validate_c(c)?;
        validate_max_iter(max_iter)?;
        validate_intercept_scaling(intercept_scaling)?;
        if loss != "epsilon_insensitive" && loss != "squared_epsilon_insensitive" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid loss '{loss}'. Expected 'epsilon_insensitive' or 'squared_epsilon_insensitive' (・`ω´・)"
            )));
        }
        if loss == "epsilon_insensitive" && !dual {
            return Err(OchreError::InvalidParameter(
                "Unsupported set of arguments: loss='epsilon_insensitive' requires dual=True (・`ω´・)"
                    .to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            epsilon,
            tol,
            c,
            loss,
            fit_intercept,
            intercept_scaling,
            dual,
            max_iter,
            random_state,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
            n_iter_: 0,
        })
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted(x, y, None)
    }

    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        validate_finite_y(y, x.rows)?;
        validate_sample_weights(sample_weight, x.rows)?;

        let weights = match sample_weight {
            Some(sw) => sw.to_vec(),
            None => vec![1.0; x.rows],
        };

        let cfg = SvmSolveConfig {
            c: self.c,
            tol: self.tol,
            max_iter: self.max_iter,
            fit_intercept: self.fit_intercept,
            intercept_scaling: self.intercept_scaling,
            random_state: self.random_state,
        };
        let backend = get_backend(&self.device);

        let (w, b, n_iter) = if self.dual {
            solve_svr_dual(x, y, &weights, self.epsilon, &self.loss, cfg)
        } else {
            solve_svr_primal_l2(&backend, x, y, &weights, self.epsilon, cfg)?
        };

        self.coef_ = Some(w);
        self.intercept_ = Some(b);
        self.n_features_in_ = Some(x.cols);
        self.n_iter_ = n_iter;
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x)?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        predict_linear_1d(x, coef, intercept)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_finite_y(y, x.rows)?;
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LINEAR_SVR);
        write_f64(&mut buf, self.epsilon);
        write_f64(&mut buf, self.tol);
        write_f64(&mut buf, self.c);
        write_str(&mut buf, &self.loss);
        write_bool(&mut buf, self.fit_intercept);
        write_f64(&mut buf, self.intercept_scaling);
        write_bool(&mut buf, self.dual);
        write_usize(&mut buf, self.max_iter);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_iter_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LINEAR_SVR)?;
        let epsilon = r.read_f64()?;
        let tol = r.read_f64()?;
        let c = r.read_f64()?;
        let loss = r.read_str()?;
        let fit_intercept = r.read_bool()?;
        let intercept_scaling = r.read_f64()?;
        let dual = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_iter_ = r.read_usize()?;

        let (dev, dev_str) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;
        Ok(Self {
            epsilon,
            tol,
            c,
            loss,
            fit_intercept,
            intercept_scaling,
            dual,
            max_iter,
            random_state,
            device: dev,
            device_: dev_str,
            coef_,
            intercept_,
            n_features_in_,
            n_iter_,
        })
    }
}

#[pyclass(from_py_object, name = "LinearSVC")]
#[derive(Clone)]
pub struct PyLinearSVC {
    inner: LinearSVC,
}

crate::pymethods_with_dit!(PyLinearSVC, LinearSVC, {
    #[new]
    #[pyo3(signature = (
        c = 1.0,
        loss = "squared_hinge".to_string(),
        penalty = "l2".to_string(),
        dual = true,
        tol = 1e-4,
        fit_intercept = true,
        intercept_scaling = 1.0,
        class_weight_mode = None,
        class_weight_map = None,
        max_iter = 1000,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: f64,
        loss: String,
        penalty: String,
        dual: bool,
        tol: f64,
        fit_intercept: bool,
        intercept_scaling: f64,
        class_weight_mode: Option<String>,
        class_weight_map: Option<Vec<(f64, f64)>>,
        max_iter: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LinearSVC::new(
                c,
                loss,
                penalty,
                dual,
                tol,
                fit_intercept,
                intercept_scaling,
                class_weight_mode,
                class_weight_map,
                max_iter,
                random_state,
                device,
            )?,
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
        let sw_vec = match sample_weight {
            Some(buf) => Some(buffer_to_f64_1d(py, &buf)?),
            None => None,
        };
        self.inner.fit_weighted(&x_mat, &y_vec, sw_vec.as_deref())?;
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
    pub fn c(&self) -> f64 {
        self.inner.c
    }

    #[getter]
    pub fn loss(&self) -> String {
        self.inner.loss.clone()
    }

    #[getter]
    pub fn penalty(&self) -> String {
        self.inner.penalty.clone()
    }

    #[getter]
    pub fn dual(&self) -> bool {
        self.inner.dual
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn intercept_scaling(&self) -> f64 {
        self.inner.intercept_scaling
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        if self.inner.classes_.is_empty() {
            None
        } else {
            Some(self.inner.classes_.clone())
        }
    }

    #[getter]
    pub fn intercept_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
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
});

#[pyclass(from_py_object, name = "LinearSVR")]
#[derive(Clone)]
pub struct PyLinearSVR {
    inner: LinearSVR,
}

crate::pymethods_with_dit!(PyLinearSVR, LinearSVR, {
    #[new]
    #[pyo3(signature = (
        epsilon = 0.0,
        tol = 1e-4,
        c = 1.0,
        loss = "epsilon_insensitive".to_string(),
        fit_intercept = true,
        intercept_scaling = 1.0,
        dual = true,
        max_iter = 1000,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epsilon: f64,
        tol: f64,
        c: f64,
        loss: String,
        fit_intercept: bool,
        intercept_scaling: f64,
        dual: bool,
        max_iter: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LinearSVR::new(
                epsilon,
                tol,
                c,
                loss,
                fit_intercept,
                intercept_scaling,
                dual,
                max_iter,
                random_state,
                device,
            )?,
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
        let sw_vec = match sample_weight {
            Some(buf) => Some(buffer_to_f64_1d(py, &buf)?),
            None => None,
        };
        self.inner.fit_weighted(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
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
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    pub fn c(&self) -> f64 {
        self.inner.c
    }

    #[getter]
    pub fn loss(&self) -> String {
        self.inner.loss.clone()
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn intercept_scaling(&self) -> f64 {
        self.inner.intercept_scaling
    }

    #[getter]
    pub fn dual(&self) -> bool {
        self.inner.dual
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearSVC>()?;
    m.add_class::<PyLinearSVR>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_svc_binary_and_multiclass() {
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

        for (loss, penalty, dual) in [
            ("squared_hinge", "l2", true),
            ("hinge", "l2", true),
            ("squared_hinge", "l2", false),
            ("squared_hinge", "l1", false),
        ] {
            let mut svc = LinearSVC::new(
                1.0,
                loss.to_string(),
                penalty.to_string(),
                dual,
                1e-4,
                true,
                1.0,
                None,
                None,
                1000,
                Some(42),
                Some("cpu".to_string()),
            )
            .unwrap();
            svc.fit(&x, &y).unwrap();
            assert_eq!(svc.score(&x, &y).unwrap(), 1.0);
            let bytes = svc.to_bytes();
            let loaded = LinearSVC::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), y);
        }

        // Multiclass OvR
        let xm = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![5.0, 0.0],
            vec![5.1, 0.2],
            vec![0.0, 5.0],
            vec![0.2, 5.1],
        ])
        .unwrap();
        let ym = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let mut svc_m = LinearSVC::new(
            1.0,
            "squared_hinge".to_string(),
            "l2".to_string(),
            true,
            1e-4,
            true,
            1.0,
            None,
            None,
            1000,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        svc_m.fit(&xm, &ym).unwrap();
        assert_eq!(svc_m.score(&xm, &ym).unwrap(), 1.0);
    }

    #[test]
    fn test_linear_svr_dual_and_primal() {
        let x = Matrix::from_2d(&[vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        for (loss, dual) in [
            ("epsilon_insensitive", true),
            ("squared_epsilon_insensitive", true),
            ("squared_epsilon_insensitive", false),
        ] {
            let mut svr = LinearSVR::new(
                0.01,
                1e-4,
                100.0,
                loss.to_string(),
                true,
                1.0,
                dual,
                2000,
                Some(42),
                Some("cpu".to_string()),
            )
            .unwrap();
            svr.fit(&x, &y).unwrap();
            let r2 = svr.score(&x, &y).unwrap();
            assert!(
                r2 > 0.99,
                "Expected R2 > 0.99 for {loss}/dual={dual}, got {r2}"
            );
            let bytes = svr.to_bytes();
            let loaded = LinearSVR::from_bytes(&bytes, Some("cpu")).unwrap();
            assert!((loaded.score(&x, &y).unwrap() - r2).abs() < 1e-12);
        }
    }
}
