// OchreML Gaussian Processes Module (＾▽＾)
// Implements GaussianProcessRegressor and GaussianProcessClassifier (Laplace + OvR)
// accelerated by Backend Primitives N3 (pairwise_l2_sq), N2 (elementwise_op),
// and E1 (matvec/matmul) with Cholesky factorization and Nelder-Mead hyperparameter optimization (*≧ω≦*).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_matrix, write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_u64,
    write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, r2_score};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes,
};
use crate::primitives::ElementwiseOp;
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::f64::consts::PI;

pub const TAG_GP_REGRESSOR: u8 = 83;
pub const TAG_GP_CLASSIFIER: u8 = 84;

const LOG_BOUND_MIN: f64 = -11.512925464970229; // ln(1e-5)
const LOG_BOUND_MAX: f64 = 11.512925464970229; // ln(1e5)
const GPU_MIN_ELEMS: usize = 131_072;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < GPU_MIN_ELEMS {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelKind {
    Rbf,
    Matern,
    White,
    RationalQuadratic,
    RbfWhite,
}

impl KernelKind {
    fn from_str(s: &str) -> OchreResult<Self> {
        match s.trim().to_lowercase().as_str() {
            "rbf" => Ok(KernelKind::Rbf),
            "matern" => Ok(KernelKind::Matern),
            "white" => Ok(KernelKind::White),
            "rational_quadratic" => Ok(KernelKind::RationalQuadratic),
            "rbf_white" => Ok(KernelKind::RbfWhite),
            _ => Err(OchreError::InvalidParameter(format!(
                "Unknown kernel '{s}'. Supported: 'rbf', 'matern', 'white', 'rational_quadratic', 'rbf_white'"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelParams {
    kind: KernelKind,
    length_scale: f64,
    constant_value: f64,
    nu: f64,
    noise_level: f64,
    rq_alpha: f64,
}

fn validate_positive_finite(name: &str, v: f64) -> OchreResult<()> {
    if !v.is_finite() || v <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "{name} ({v}) must be a strictly positive finite number"
        )));
    }
    Ok(())
}

fn validate_non_negative_finite(name: &str, v: f64) -> OchreResult<()> {
    if !v.is_finite() || v < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "{name} ({v}) must be a non-negative finite number"
        )));
    }
    Ok(())
}

fn validate_nu(nu: f64) -> OchreResult<()> {
    if !nu.is_finite()
        || ((nu - 0.5).abs() > 1e-12 && (nu - 1.5).abs() > 1e-12 && (nu - 2.5).abs() > 1e-12)
    {
        return Err(OchreError::InvalidParameter(format!(
            "nu ({nu}) for Matérn kernel must be 0.5, 1.5, or 2.5"
        )));
    }
    Ok(())
}

fn validate_matrix_finite(x: &Matrix, context: &str) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{context} contains NaN or Inf values"
        )));
    }
    Ok(())
}

fn validate_vec_finite(v: &[f64], context: &str) -> OchreResult<()> {
    if v.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{context} contains NaN or Inf values"
        )));
    }
    Ok(())
}

/// Deterministic SplitMix64 PRNG for optimizer restarts.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64_uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        low + u * (high - low)
    }
}

/// Compute kernel matrix K(a, b) using Backend N3 (pairwise_l2_sq) and N2 (elementwise_op).
/// When `is_self` is true (a == b, square symmetric), white-noise diagonal is added for
/// White and RbfWhite kernels.
fn compute_kernel_matrix(
    backend: &Backend<'_>,
    a: &Matrix,
    b: &Matrix,
    params: &KernelParams,
    is_self: bool,
) -> OchreResult<Matrix> {
    let rows = a.rows;
    let cols = b.rows;

    if params.kind == KernelKind::White {
        let mut k = Matrix::zeros(rows, cols);
        if is_self {
            for i in 0..rows.min(cols) {
                k.set(i, i, params.noise_level);
            }
        }
        return Ok(k);
    }

    // N3: Squared Euclidean distance matrix D[i, j] = ||a_i - b_j||^2
    let mut k = backend.pairwise_l2_sq(a, b)?;
    let ls = params.length_scale;
    let ls_sq = ls * ls;
    let c = params.constant_value;

    match params.kind {
        KernelKind::Rbf | KernelKind::RbfWhite => {
            let inv_two_ls_sq = -0.5 / ls_sq;
            if k.data.len() >= 16_384 {
                k.data.par_iter_mut().for_each(|d| {
                    *d = d.max(0.0) * inv_two_ls_sq;
                });
                backend.elementwise_op_inplace(&mut k, ElementwiseOp::Exp)?;
                if (c - 1.0).abs() > 0.0 {
                    backend.elementwise_op_inplace(
                        &mut k,
                        ElementwiseOp::Affine {
                            scale: c,
                            shift: 0.0,
                        },
                    )?;
                }
            } else {
                for d in &mut k.data {
                    *d = c * (d.max(0.0) * inv_two_ls_sq).exp();
                }
            }
            if is_self && params.kind == KernelKind::RbfWhite {
                for i in 0..rows.min(cols) {
                    let cur = k.get(i, i);
                    k.set(i, i, cur + params.noise_level);
                }
            }
        }
        KernelKind::Matern => {
            let nu = params.nu;
            if (nu - 0.5).abs() <= 1e-12 {
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() / ls;
                    *d = c * (-r).exp();
                });
            } else if (nu - 1.5).abs() <= 1e-12 {
                let s3 = 3.0_f64.sqrt() / ls;
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() * s3;
                    *d = c * (1.0 + r) * (-r).exp();
                });
            } else {
                let s5 = 5.0_f64.sqrt() / ls;
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() * s5;
                    *d = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                });
            }
        }
        KernelKind::RationalQuadratic => {
            let alpha = params.rq_alpha;
            let denom = 2.0 * alpha * ls_sq;
            k.data.par_iter_mut().for_each(|d| {
                let base = 1.0 + d.max(0.0) / denom;
                *d = c * base.powf(-alpha);
            });
        }
        KernelKind::White => unreachable!(),
    }

    if is_self {
        for i in 0..rows.min(cols) {
            let diag = match params.kind {
                KernelKind::White => params.noise_level,
                KernelKind::RbfWhite => params.constant_value + params.noise_level,
                _ => params.constant_value,
            };
            k.set(i, i, diag);
        }
    }

    Ok(k)
}

/// Build kernel matrix K from precomputed squared Euclidean distance matrix D2.
fn compute_kernel_from_d2(
    backend: &Backend<'_>,
    d2: &Matrix,
    params: &KernelParams,
    is_self: bool,
) -> OchreResult<Matrix> {
    let rows = d2.rows;
    let cols = d2.cols;
    if params.kind == KernelKind::White {
        let mut k = Matrix::zeros(rows, cols);
        if is_self {
            for i in 0..rows.min(cols) {
                k.set(i, i, params.noise_level);
            }
        }
        return Ok(k);
    }

    let mut k = d2.clone();
    let ls = params.length_scale;
    let ls_sq = ls * ls;
    let c = params.constant_value;

    match params.kind {
        KernelKind::Rbf | KernelKind::RbfWhite => {
            let inv_two_ls_sq = -0.5 / ls_sq;
            if k.data.len() >= 16_384 {
                k.data.par_iter_mut().for_each(|d| {
                    *d = d.max(0.0) * inv_two_ls_sq;
                });
                backend.elementwise_op_inplace(&mut k, ElementwiseOp::Exp)?;
                if (c - 1.0).abs() > 0.0 {
                    backend.elementwise_op_inplace(
                        &mut k,
                        ElementwiseOp::Affine {
                            scale: c,
                            shift: 0.0,
                        },
                    )?;
                }
            } else {
                for d in &mut k.data {
                    *d = c * (d.max(0.0) * inv_two_ls_sq).exp();
                }
            }
            if is_self && params.kind == KernelKind::RbfWhite {
                for i in 0..rows.min(cols) {
                    let cur = k.get(i, i);
                    k.set(i, i, cur + params.noise_level);
                }
            }
        }
        KernelKind::Matern => {
            let nu = params.nu;
            if (nu - 0.5).abs() <= 1e-12 {
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() / ls;
                    *d = c * (-r).exp();
                });
            } else if (nu - 1.5).abs() <= 1e-12 {
                let s3 = 3.0_f64.sqrt() / ls;
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() * s3;
                    *d = c * (1.0 + r) * (-r).exp();
                });
            } else {
                let s5 = 5.0_f64.sqrt() / ls;
                k.data.par_iter_mut().for_each(|d| {
                    let r = d.max(0.0).sqrt() * s5;
                    *d = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                });
            }
        }
        KernelKind::RationalQuadratic => {
            let alpha = params.rq_alpha;
            let denom = 2.0 * alpha * ls_sq;
            k.data.par_iter_mut().for_each(|d| {
                let base = 1.0 + d.max(0.0) / denom;
                *d = c * base.powf(-alpha);
            });
        }
        KernelKind::White => unreachable!(),
    }

    if is_self {
        for i in 0..rows.min(cols) {
            let diag = match params.kind {
                KernelKind::White => params.noise_level,
                KernelKind::RbfWhite => params.constant_value + params.noise_level,
                _ => params.constant_value,
            };
            k.set(i, i, diag);
        }
    }

    Ok(k)
}

#[inline]
fn kernel_diag_value(params: &KernelParams) -> f64 {
    match params.kind {
        KernelKind::White => params.noise_level,
        KernelKind::RbfWhite => params.constant_value + params.noise_level,
        _ => params.constant_value,
    }
}

#[inline]
fn dot_unrolled(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut idx = 0;
    for _ in 0..chunks {
        s0 += a[idx] * b[idx];
        s1 += a[idx + 1] * b[idx + 1];
        s2 += a[idx + 2] * b[idx + 2];
        s3 += a[idx + 3] * b[idx + 3];
        idx += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while idx < n {
        sum += a[idx] * b[idx];
        idx += 1;
    }
    sum
}

/// Lower-triangular Cholesky factorization: A = L * L^T.
fn cholesky_lower(a: &Matrix) -> OchreResult<Matrix> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        let row_i_off = i * n;
        for j in 0..=i {
            let sum = dot_unrolled(&l.data[row_i_off..row_i_off + j], &l.data[j * n..j * n + j]);
            if i == j {
                let diag = a.data[row_i_off + i] - sum;
                if !diag.is_finite() || diag <= 1e-15 {
                    return Err(OchreError::SingularMatrix(
                        "Kernel matrix is not positive definite; consider increasing alpha or noise_level".to_string(),
                    ));
                }
                l.data[row_i_off + i] = diag.sqrt();
            } else {
                let ljj = l.data[j * n + j];
                if ljj.abs() <= 1e-15 {
                    return Err(OchreError::SingularMatrix(
                        "Zero diagonal encountered during Cholesky factorization".to_string(),
                    ));
                }
                l.data[row_i_off + j] = (a.data[row_i_off + j] - sum) / ljj;
            }
        }
    }
    Ok(l)
}

/// Forward substitution into preallocated buffer: solve L * z = b where L is lower triangular (n x n).
#[inline]
fn solve_lower_triangular_into(l: &Matrix, b: &[f64], z: &mut [f64]) {
    let n = l.rows;
    for i in 0..n {
        let row_off = i * n;
        let sum = dot_unrolled(&l.data[row_off..row_off + i], &z[..i]);
        z[i] = (b[i] - sum) / l.data[row_off + i];
    }
}

/// Forward substitution: solve L * z = b where L is lower triangular (n x n).
fn solve_lower_triangular(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; l.rows];
    solve_lower_triangular_into(l, b, &mut z);
    z
}

/// Backward substitution: solve L^T * x = z where L is lower triangular (n x n).
/// Uses contiguous row-major scatter-subtract for cache locality and 4x unrolling.
fn solve_upper_triangular_t(l: &Matrix, z: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = z[..n].to_vec();
    for i in (0..n).rev() {
        let row_off = i * n;
        let xi = x[i] / l.data[row_off + i];
        x[i] = xi;
        let row_i = &l.data[row_off..row_off + i];
        let mut k = 0;
        let chunks = i / 4;
        for _ in 0..chunks {
            x[k] -= row_i[k] * xi;
            x[k + 1] -= row_i[k + 1] * xi;
            x[k + 2] -= row_i[k + 2] * xi;
            x[k + 3] -= row_i[k + 3] * xi;
            k += 4;
        }
        while k < i {
            x[k] -= row_i[k] * xi;
            k += 1;
        }
    }
    x
}

/// Solve (L * L^T) * x = b given lower-triangular Cholesky factor L.
fn solve_cholesky_from_l(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let z = solve_lower_triangular(l, b);
    solve_upper_triangular_t(l, &z)
}

/// Compute (L^{-1})^T where L is lower triangular (n x n).
/// Entry (i, k) stores (L^{-1})_{k, i} for k >= i, so columns of L^{-1} are stored contiguously as rows.
fn invert_lower_triangular_t(l: &Matrix) -> Vec<f64> {
    let n = l.rows;
    let mut m_t = vec![0.0; n * n];
    for i in 0..n {
        let mi_off = i * n;
        m_t[mi_off + i] = 1.0 / l.data[i * n + i];
        for k in (i + 1)..n {
            let lk_off = k * n;
            let sum = dot_unrolled(
                &l.data[lk_off + i..lk_off + k],
                &m_t[mi_off + i..mi_off + k],
            );
            m_t[mi_off + k] = -sum / l.data[lk_off + k];
        }
    }
    m_t
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    dot_unrolled(a, b)
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Nelder-Mead minimizer in `dim` dimensions with box clamping `[LOG_BOUND_MIN, LOG_BOUND_MAX]`.
fn nelder_mead_minimize<F>(init: &[f64], max_iter: usize, tol: f64, mut f: F) -> (Vec<f64>, f64)
where
    F: FnMut(&[f64]) -> f64,
{
    let n = init.len();
    if n == 0 {
        let val = f(init);
        return (init.to_vec(), val);
    }

    let clamp_pt = |pt: &mut [f64]| {
        for v in pt.iter_mut() {
            *v = v.clamp(LOG_BOUND_MIN, LOG_BOUND_MAX);
        }
    };

    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    let mut x0 = init.to_vec();
    clamp_pt(&mut x0);
    let f0 = f(&x0);
    simplex.push((x0.clone(), f0));

    for i in 0..n {
        let mut xi = x0.clone();
        let step = if xi[i].abs() > 1e-3 {
            0.25 * xi[i].abs()
        } else {
            0.25
        };
        xi[i] += step;
        clamp_pt(&mut xi);
        let fi = f(&xi);
        simplex.push((xi, fi));
    }

    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let f_best = simplex[0].1;
        let f_worst = simplex[n].1;
        if (f_worst - f_best).abs() < tol {
            let mut max_span = 0.0_f64;
            for i in 1..=n {
                for j in 0..n {
                    max_span = max_span.max((simplex[i].0[j] - simplex[0].0[j]).abs());
                }
            }
            if max_span < tol {
                break;
            }
        }

        let mut centroid = vec![0.0; n];
        for item in simplex.iter().take(n) {
            for (j, cj) in centroid.iter_mut().enumerate().take(n) {
                *cj += item.0[j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let mut xr = vec![0.0; n];
        for j in 0..n {
            xr[j] = centroid[j] + 1.0 * (centroid[j] - simplex[n].0[j]);
        }
        clamp_pt(&mut xr);
        let fr = f(&xr);

        if fr < simplex[n - 1].1 && fr >= simplex[0].1 {
            simplex[n] = (xr, fr);
            continue;
        }

        // Expansion
        if fr < simplex[0].1 {
            let mut xe = vec![0.0; n];
            for j in 0..n {
                xe[j] = centroid[j] + 2.0 * (xr[j] - centroid[j]);
            }
            clamp_pt(&mut xe);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = (xe, fe);
            } else {
                simplex[n] = (xr, fr);
            }
            continue;
        }

        // Contraction
        let mut xc = vec![0.0; n];
        if fr < simplex[n].1 {
            for j in 0..n {
                xc[j] = centroid[j] + 0.5 * (xr[j] - centroid[j]);
            }
        } else {
            for j in 0..n {
                xc[j] = centroid[j] - 0.5 * (centroid[j] - simplex[n].0[j]);
            }
        }
        clamp_pt(&mut xc);
        let fc = f(&xc);
        if fc < simplex[n].1.min(fr) {
            simplex[n] = (xc, fc);
            continue;
        }

        // Shrink
        let best_pt = simplex[0].0.clone();
        for item in simplex.iter_mut().skip(1) {
            for (j, &bp) in best_pt.iter().enumerate().take(n) {
                item.0[j] = bp + 0.5 * (item.0[j] - bp);
            }
            clamp_pt(&mut item.0);
            item.1 = f(&item.0);
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex
        .into_iter()
        .next()
        .unwrap_or_else(|| (init.to_vec(), f0))
}

fn params_to_log_vec(p: &KernelParams) -> Vec<f64> {
    match p.kind {
        KernelKind::White => vec![p.noise_level.ln()],
        KernelKind::Rbf | KernelKind::Matern => vec![p.constant_value.ln(), p.length_scale.ln()],
        KernelKind::RationalQuadratic => {
            vec![p.constant_value.ln(), p.length_scale.ln(), p.rq_alpha.ln()]
        }
        KernelKind::RbfWhite => {
            vec![
                p.constant_value.ln(),
                p.length_scale.ln(),
                p.noise_level.ln(),
            ]
        }
    }
}

fn apply_log_vec(base: &KernelParams, theta: &[f64]) -> KernelParams {
    let mut out = *base;
    match base.kind {
        KernelKind::White => {
            if !theta.is_empty() {
                out.noise_level = theta[0].exp();
            }
        }
        KernelKind::Rbf | KernelKind::Matern => {
            if theta.len() >= 2 {
                out.constant_value = theta[0].exp();
                out.length_scale = theta[1].exp();
            }
        }
        KernelKind::RationalQuadratic => {
            if theta.len() >= 3 {
                out.constant_value = theta[0].exp();
                out.length_scale = theta[1].exp();
                out.rq_alpha = theta[2].exp();
            }
        }
        KernelKind::RbfWhite => {
            if theta.len() >= 3 {
                out.constant_value = theta[0].exp();
                out.length_scale = theta[1].exp();
                out.noise_level = theta[2].exp();
            }
        }
    }
    out
}

fn optimize_lml<F>(
    base: KernelParams,
    n_restarts: usize,
    random_state: Option<u64>,
    mut eval_lml: F,
) -> KernelParams
where
    F: FnMut(&KernelParams) -> Option<f64>,
{
    let init = params_to_log_vec(&base);
    let dim = init.len();
    let mut best_theta = init.clone();
    let mut best_lml = eval_lml(&base).unwrap_or(f64::NEG_INFINITY);

    let (opt_theta, opt_neg_lml) = nelder_mead_minimize(&init, 150, 1e-6, |pt| {
        let cand = apply_log_vec(&base, pt);
        match eval_lml(&cand) {
            Some(lml) if lml.is_finite() => -lml,
            _ => 1e30,
        }
    });
    if -opt_neg_lml > best_lml {
        best_lml = -opt_neg_lml;
        best_theta = opt_theta;
    }

    if n_restarts > 0 {
        let mut rng = SplitMix64::new(random_state.unwrap_or(42));
        for _ in 0..n_restarts {
            let mut start = Vec::with_capacity(dim);
            for _ in 0..dim {
                start.push(rng.next_f64_uniform(-3.0, 3.0));
            }
            let (cand_theta, cand_neg_lml) = nelder_mead_minimize(&start, 150, 1e-6, |pt| {
                let cand = apply_log_vec(&base, pt);
                match eval_lml(&cand) {
                    Some(lml) if lml.is_finite() => -lml,
                    _ => 1e30,
                }
            });
            if -cand_neg_lml > best_lml {
                best_lml = -cand_neg_lml;
                best_theta = cand_theta;
            }
        }
    }

    apply_log_vec(&base, &best_theta)
}

/// Compact box-constrained L-BFGS minimizer on `[LOG_BOUND_MIN, LOG_BOUND_MAX]^dim`.
/// `f_and_grad` returns `Some((neg_lml, neg_grad))` or `None` if evaluation fails (e.g. non-PD).
fn lbfgs_minimize<F>(init: &[f64], max_iter: usize, gtol: f64, mut f_and_grad: F) -> (Vec<f64>, f64)
where
    F: FnMut(&[f64]) -> Option<(f64, Vec<f64>)>,
{
    let dim = init.len();
    if dim == 0 {
        return (Vec::new(), 0.0);
    }

    let clamp_vec = |v: &mut [f64]| {
        for x in v.iter_mut() {
            *x = x.clamp(LOG_BOUND_MIN, LOG_BOUND_MAX);
        }
    };

    let mut x = init.to_vec();
    clamp_vec(&mut x);
    let (mut f_val, mut grad) = match f_and_grad(&x) {
        Some((fv, g)) if fv.is_finite() && g.iter().all(|v| v.is_finite()) => (fv, g),
        _ => return (x, 1e30),
    };

    let mut best_x = x.clone();
    let mut best_f = f_val;

    const M: usize = 8;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(M);

    let mut q = vec![0.0; dim];
    let mut dir = vec![0.0; dim];
    let mut x_cand = vec![0.0; dim];
    let mut alpha_buf = [0.0_f64; M];

    for _iter in 0..max_iter {
        // Projected gradient infinity norm check
        let mut pg_inf = 0.0_f64;
        for j in 0..dim {
            let gj = if (x[j] <= LOG_BOUND_MIN && grad[j] > 0.0)
                || (x[j] >= LOG_BOUND_MAX && grad[j] < 0.0)
            {
                0.0
            } else {
                grad[j]
            };
            pg_inf = pg_inf.max(gj.abs());
        }
        if pg_inf < gtol {
            break;
        }

        // L-BFGS two-loop recursion
        q.copy_from_slice(&grad);
        let h_len = s_hist.len();
        for idx in (0..h_len).rev() {
            let a = rho_hist[idx] * dot_unrolled(&s_hist[idx], &q);
            alpha_buf[idx] = a;
            let yi = &y_hist[idx];
            for j in 0..dim {
                q[j] -= a * yi[j];
            }
        }
        let gamma = if h_len > 0 {
            let s_last = &s_hist[h_len - 1];
            let y_last = &y_hist[h_len - 1];
            let yy = dot_unrolled(y_last, y_last);
            if yy > 1e-24 {
                (dot_unrolled(s_last, y_last) / yy).clamp(1e-6, 1e6)
            } else {
                1.0
            }
        } else {
            1.0
        };
        for j in 0..dim {
            dir[j] = gamma * q[j];
        }
        for idx in 0..h_len {
            let beta = rho_hist[idx] * dot_unrolled(&y_hist[idx], &dir);
            let diff = alpha_buf[idx] - beta;
            let si = &s_hist[idx];
            for j in 0..dim {
                dir[j] += si[j] * diff;
            }
        }
        for j in 0..dim {
            dir[j] = -dir[j];
            if (x[j] <= LOG_BOUND_MIN && dir[j] < 0.0) || (x[j] >= LOG_BOUND_MAX && dir[j] > 0.0) {
                dir[j] = 0.0;
            }
        }

        // Fallback to projected steepest descent if direction is not a descent direction
        let mut dg = dot_unrolled(&dir, &grad);
        if !dg.is_finite() || dg >= 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            for j in 0..dim {
                let gj = if (x[j] <= LOG_BOUND_MIN && grad[j] > 0.0)
                    || (x[j] >= LOG_BOUND_MAX && grad[j] < 0.0)
                {
                    0.0
                } else {
                    grad[j]
                };
                dir[j] = -gj;
            }
            dg = dot_unrolled(&dir, &grad);
            if !dg.is_finite() || dg >= 0.0 {
                break;
            }
        }

        // Cap initial step when history is empty so log-space parameters don't explode
        let max_d = dir.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let mut step = if s_hist.is_empty() && max_d > 1.0 {
            (1.0 / max_d).min(1.0)
        } else if max_d > 4.0 {
            4.0 / max_d
        } else {
            1.0
        };

        let mut accepted = false;
        let mut next_f = f_val;
        let mut next_grad = Vec::new();
        for _ls in 0..12 {
            let mut actual_dg = 0.0;
            for j in 0..dim {
                let xj_new = (x[j] + step * dir[j]).clamp(LOG_BOUND_MIN, LOG_BOUND_MAX);
                x_cand[j] = xj_new;
                actual_dg += grad[j] * (xj_new - x[j]);
            }
            if let Some((fc, gc)) = f_and_grad(&x_cand)
                && fc.is_finite()
                && gc.iter().all(|v| v.is_finite())
            {
                if fc < best_f {
                    best_f = fc;
                    best_x.copy_from_slice(&x_cand);
                }
                if fc <= f_val + 1e-4 * actual_dg {
                    next_f = fc;
                    next_grad = gc;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }

        if !accepted {
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        }

        let mut s_vec = vec![0.0; dim];
        let mut y_vec = vec![0.0; dim];
        let mut step_norm_inf = 0.0_f64;
        for j in 0..dim {
            s_vec[j] = x_cand[j] - x[j];
            y_vec[j] = next_grad[j] - grad[j];
            step_norm_inf = step_norm_inf.max(s_vec[j].abs());
        }
        let sy = dot_unrolled(&s_vec, &y_vec);
        let ss = dot_unrolled(&s_vec, &s_vec);
        let yy = dot_unrolled(&y_vec, &y_vec);
        if sy > 1e-12 * (ss * yy).sqrt() {
            if s_hist.len() == M {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        let f_diff = (f_val - next_f).abs();
        x.copy_from_slice(&x_cand);
        f_val = next_f;
        grad = next_grad;

        if step_norm_inf < 1e-7 && f_diff < 1e-9 * (1.0 + f_val.abs()) {
            break;
        }
    }

    (best_x, best_f)
}

fn optimize_lml_lbfgs<F>(
    base: KernelParams,
    n_restarts: usize,
    random_state: Option<u64>,
    mut eval_lml_grad: F,
) -> KernelParams
where
    F: FnMut(&KernelParams) -> Option<(f64, Vec<f64>)>,
{
    let init = params_to_log_vec(&base);
    let dim = init.len();
    let mut best_theta = init.clone();
    let mut best_lml = f64::NEG_INFINITY;

    let (opt_theta, opt_neg_lml) = lbfgs_minimize(&init, 30, 1e-5, |pt| {
        let cand = apply_log_vec(&base, pt);
        eval_lml_grad(&cand).map(|(lml, g)| (-lml, g.into_iter().map(|v| -v).collect()))
    });
    if -opt_neg_lml > best_lml {
        best_lml = -opt_neg_lml;
        best_theta = opt_theta;
    }

    if n_restarts > 0 {
        let mut rng = SplitMix64::new(random_state.unwrap_or(42));
        for _ in 0..n_restarts {
            let mut start = Vec::with_capacity(dim);
            for _ in 0..dim {
                start.push(rng.next_f64_uniform(-3.0, 3.0));
            }
            let (cand_theta, cand_neg_lml) = lbfgs_minimize(&start, 30, 1e-5, |pt| {
                let cand = apply_log_vec(&base, pt);
                eval_lml_grad(&cand).map(|(lml, g)| (-lml, g.into_iter().map(|v| -v).collect()))
            });
            if -cand_neg_lml > best_lml {
                best_lml = -cand_neg_lml;
                best_theta = cand_theta;
            }
        }
    }

    apply_log_vec(&base, &best_theta)
}

// ============================================================================
// GaussianProcessRegressor
// ============================================================================

#[derive(Clone)]
pub struct GaussianProcessRegressor {
    pub kernel: String,
    pub length_scale: f64,
    pub constant_value: f64,
    pub nu: f64,
    pub alpha: f64,
    pub noise_level: f64,
    pub rq_alpha: f64,
    pub optimize_hyperparams: bool,
    pub n_restarts_optimizer: usize,
    pub normalize_y: bool,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    parsed_device: DeviceType,

    // Fitted attributes
    pub x_train_: Option<Matrix>,
    pub y_train_mean_: f64,
    pub y_train_std_: f64,
    pub l_: Option<Matrix>,
    pub alpha_: Option<Vec<f64>>,
    pub log_marginal_likelihood_value_: Option<f64>,
    pub length_scale_: f64,
    pub constant_value_: f64,
    pub noise_level_: f64,
    pub rq_alpha_: f64,
    pub n_features_in_: Option<usize>,
}

impl GaussianProcessRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        length_scale: f64,
        constant_value: f64,
        nu: f64,
        alpha: f64,
        noise_level: f64,
        rq_alpha: f64,
        optimize_hyperparams: bool,
        n_restarts_optimizer: usize,
        normalize_y: bool,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let _ = KernelKind::from_str(&kernel)?;
        validate_positive_finite("length_scale", length_scale)?;
        validate_positive_finite("constant_value", constant_value)?;
        validate_nu(nu)?;
        validate_non_negative_finite("alpha", alpha)?;
        validate_positive_finite("noise_level", noise_level)?;
        validate_positive_finite("rq_alpha", rq_alpha)?;

        let parsed_device = parse_device(device.as_deref())?;
        Ok(Self {
            kernel,
            length_scale,
            constant_value,
            nu,
            alpha,
            noise_level,
            rq_alpha,
            optimize_hyperparams,
            n_restarts_optimizer,
            normalize_y,
            random_state,
            device_: device,
            parsed_device,
            x_train_: None,
            y_train_mean_: 0.0,
            y_train_std_: 1.0,
            l_: None,
            alpha_: None,
            log_marginal_likelihood_value_: None,
            length_scale_: length_scale,
            constant_value_: constant_value,
            noise_level_: noise_level,
            rq_alpha_: rq_alpha,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let parsed = parse_device(device.as_deref())?;
        self.parsed_device = parsed;
        self.device_ = device;
        Ok(())
    }

    fn base_kernel_params(&self) -> OchreResult<KernelParams> {
        Ok(KernelParams {
            kind: KernelKind::from_str(&self.kernel)?,
            length_scale: self.length_scale,
            constant_value: self.constant_value,
            nu: self.nu,
            noise_level: self.noise_level,
            rq_alpha: self.rq_alpha,
        })
    }

    fn fitted_kernel_params(&self) -> OchreResult<KernelParams> {
        Ok(KernelParams {
            kind: KernelKind::from_str(&self.kernel)?,
            length_scale: self.length_scale_,
            constant_value: self.constant_value_,
            nu: self.nu,
            noise_level: self.noise_level_,
            rq_alpha: self.rq_alpha_,
        })
    }

    fn eval_gpr_fit_from_d2(
        backend: &Backend<'_>,
        d2: Option<&Matrix>,
        n: usize,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
    ) -> OchreResult<(Matrix, Vec<f64>, f64)> {
        let mut k = match d2 {
            Some(d) => compute_kernel_from_d2(backend, d, params, true)?,
            None => {
                let mut m = Matrix::zeros(n, n);
                for i in 0..n {
                    m.set(i, i, params.noise_level);
                }
                m
            }
        };
        for i in 0..n {
            let cur = k.get(i, i);
            k.set(i, i, cur + alpha_reg);
        }
        let l = cholesky_lower(&k)?;
        let alpha_vec = solve_cholesky_from_l(&l, y_norm);

        let mut log_det_half = 0.0;
        for i in 0..n {
            log_det_half += l.data[i * n + i].ln();
        }
        let lml =
            -0.5 * dot(y_norm, &alpha_vec) - log_det_half - 0.5 * (n as f64) * (2.0 * PI).ln();

        Ok((l, alpha_vec, lml))
    }

    /// Evaluate GPR LML and analytical gradient wrt log-hyperparameters for `Rbf` and `RbfWhite`.
    /// Gradient formula: d(LML)/d(theta_m) = 0.5 * tr((alpha * alpha^T - K^{-1}) * dK/d(theta_m)).
    fn eval_gpr_lml_and_grad(
        d2: &Matrix,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let n = d2.rows;
        let ls = params.length_scale;
        let ls_sq = ls * ls;
        let inv_ls_sq = 1.0 / ls_sq;
        let inv_two_ls_sq = -0.5 * inv_ls_sq;
        let c = params.constant_value;
        let is_rbf_white = params.kind == KernelKind::RbfWhite;
        let extra_diag = if is_rbf_white {
            params.noise_level + alpha_reg
        } else {
            alpha_reg
        };

        // Build K0 (RBF part) and K = K0 + extra_diag * I
        let mut k0 = Matrix::zeros(n, n);
        let mut k = Matrix::zeros(n, n);
        for i in 0..n {
            let row = i * n;
            k0.data[row + i] = c;
            k.data[row + i] = c + extra_diag;
            for j in 0..i {
                let d_ij = d2.data[row + j].max(0.0);
                let val = c * (d_ij * inv_two_ls_sq).exp();
                k0.data[row + j] = val;
                k0.data[j * n + i] = val;
                k.data[row + j] = val;
                k.data[j * n + i] = val;
            }
        }

        let l = cholesky_lower(&k)?;
        let alpha_vec = solve_cholesky_from_l(&l, y_norm);

        let mut log_det_half = 0.0;
        for i in 0..n {
            log_det_half += l.data[i * n + i].ln();
        }
        let lml =
            -0.5 * dot(y_norm, &alpha_vec) - log_det_half - 0.5 * (n as f64) * (2.0 * PI).ln();

        // Compute M^T = (L^{-1})^T so K^{-1}_{ij} = dot(m_t[i, i..n], m_t[j, i..n]) for i >= j
        let m_t = invert_lower_triangular_t(&l);

        let mut g_log_c = 0.0;
        let mut g_log_ls = 0.0;
        let mut g_log_noise = 0.0;

        for i in 0..n {
            let row = i * n;
            let ai = alpha_vec[i];
            let mi = &m_t[row + i..row + n];
            let k_inv_ii = dot_unrolled(mi, mi);
            let w_ii = ai * ai - k_inv_ii;
            g_log_c += 0.5 * w_ii * c;
            if is_rbf_white {
                g_log_noise += 0.5 * w_ii * params.noise_level;
            }

            for j in 0..i {
                let mj = &m_t[j * n + i..j * n + n];
                let k_inv_ij = dot_unrolled(mi, mj);
                let w_ij = ai * alpha_vec[j] - k_inv_ij;
                let k0_ij = k0.data[row + j];
                // Factor 2 for symmetric off-diagonal * 0.5 in LML grad = 1.0
                let wk = w_ij * k0_ij;
                g_log_c += wk;
                g_log_ls += wk * (d2.data[row + j].max(0.0) * inv_ls_sq);
            }
        }

        let grad = if is_rbf_white {
            vec![g_log_c, g_log_ls, g_log_noise]
        } else {
            vec![g_log_c, g_log_ls]
        };
        Ok((lml, grad))
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_matrix_finite(x, "Feature matrix X")?;
        validate_vec_finite(y, "Target vector y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }

        let n = x.rows;
        let (y_mean, y_std, y_norm) = if self.normalize_y {
            let mean = y.iter().sum::<f64>() / (n as f64);
            let var = y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64);
            let std = var.sqrt();
            let safe_std = if std == 0.0 { 1.0 } else { std };
            let normed: Vec<f64> = y.iter().map(|&v| (v - mean) / safe_std).collect();
            (mean, safe_std, normed)
        } else {
            (0.0, 1.0, y.to_vec())
        };

        let backend = effective_backend(&self.parsed_device, n.saturating_mul(n));
        let base_params = self.base_kernel_params()?;
        let d2 = if base_params.kind == KernelKind::White {
            None
        } else {
            Some(backend.pairwise_l2_sq(x, x)?)
        };

        let best_params = if self.optimize_hyperparams || self.n_restarts_optimizer > 0 {
            match (base_params.kind, d2.as_ref()) {
                (KernelKind::Rbf | KernelKind::RbfWhite, Some(d2_mat)) => optimize_lml_lbfgs(
                    base_params,
                    self.n_restarts_optimizer,
                    self.random_state,
                    |cand| Self::eval_gpr_lml_and_grad(d2_mat, &y_norm, self.alpha, cand).ok(),
                ),
                _ => optimize_lml(
                    base_params,
                    self.n_restarts_optimizer,
                    self.random_state,
                    |cand| {
                        Self::eval_gpr_fit_from_d2(
                            &backend,
                            d2.as_ref(),
                            n,
                            &y_norm,
                            self.alpha,
                            cand,
                        )
                        .ok()
                        .map(|r| r.2)
                    },
                ),
            }
        } else {
            base_params
        };

        let (l, alpha_vec, lml) = Self::eval_gpr_fit_from_d2(
            &backend,
            d2.as_ref(),
            n,
            &y_norm,
            self.alpha,
            &best_params,
        )?;

        self.x_train_ = Some(x.clone());
        self.y_train_mean_ = y_mean;
        self.y_train_std_ = y_std;
        self.l_ = Some(l);
        self.alpha_ = Some(alpha_vec);
        self.log_marginal_likelihood_value_ = Some(lml);
        self.length_scale_ = best_params.length_scale;
        self.constant_value_ = best_params.constant_value;
        self.noise_level_ = best_params.noise_level;
        self.rq_alpha_ = best_params.rq_alpha;
        self.n_features_in_ = Some(x.cols);
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn predict_with_uncertainty(
        &self,
        x: &Matrix,
        return_std: bool,
        return_cov: bool,
    ) -> OchreResult<(Vec<f64>, Option<Vec<f64>>, Option<Matrix>)> {
        if return_std && return_cov {
            return Err(OchreError::InvalidParameter(
                "Not returning std and cov at the same time".to_string(),
            ));
        }
        let x_train = self.x_train_.as_ref().ok_or(OchreError::NotFitted)?;
        let l = self.l_.as_ref().ok_or(OchreError::NotFitted)?;
        let alpha_vec = self.alpha_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_cols = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x, "Feature matrix X")?;
        if x.cols != expected_cols {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }

        let n_test = x.rows;
        let n_train = x_train.rows;
        let elems = if return_cov {
            n_test.saturating_mul(n_train.max(n_test))
        } else {
            n_test.saturating_mul(n_train)
        };
        let backend = effective_backend(&self.parsed_device, elems);
        let params = self.fitted_kernel_params()?;
        let k_trans = compute_kernel_matrix(&backend, x, x_train, &params, false)?;

        // y_mean = y_std * (K_trans @ alpha_) + y_mean
        let raw_mean = backend.matvec(&k_trans, alpha_vec)?;
        let y_mean: Vec<f64> = raw_mean
            .into_iter()
            .map(|v| v * self.y_train_std_ + self.y_train_mean_)
            .collect();

        let var_scale = self.y_train_std_ * self.y_train_std_;

        if !return_std && !return_cov {
            return Ok((y_mean, None, None));
        }

        if return_std {
            let k_diag = kernel_diag_value(&params);
            let y_std: Vec<f64> = if n_test <= 256 {
                let mut v = vec![0.0; n_train];
                let mut out = Vec::with_capacity(n_test);
                for i in 0..n_test {
                    let row = &k_trans.data[i * n_train..(i + 1) * n_train];
                    solve_lower_triangular_into(l, row, &mut v);
                    let v_sq = dot_unrolled(&v, &v);
                    out.push(((k_diag - v_sq).max(0.0) * var_scale).sqrt());
                }
                out
            } else {
                (0..n_test)
                    .into_par_iter()
                    .map_init(
                        || vec![0.0; n_train],
                        |v, i| {
                            let row = &k_trans.data[i * n_train..(i + 1) * n_train];
                            solve_lower_triangular_into(l, row, v);
                            let v_sq = dot_unrolled(v, v);
                            ((k_diag - v_sq).max(0.0) * var_scale).sqrt()
                        },
                    )
                    .collect()
            };
            return Ok((y_mean, Some(y_std), None));
        }

        // Solve L * V = K_trans^T => V has shape (n_train, n_test), stored column-by-column
        let v_cols: Vec<Vec<f64>> = (0..n_test)
            .into_par_iter()
            .map(|i| {
                let row = &k_trans.data[i * n_train..(i + 1) * n_train];
                solve_lower_triangular(l, row)
            })
            .collect();

        // return_cov == true
        let mut y_cov = compute_kernel_matrix(&backend, x, x, &params, true)?;
        let mut v_mat = Matrix::zeros(n_train, n_test);
        for (j, col) in v_cols.iter().enumerate() {
            for (i, &val) in col.iter().enumerate().take(n_train) {
                v_mat.set(i, j, val);
            }
        }
        let vt_v = backend.matmul_at_b(&v_mat, &v_mat)?;
        for idx in 0..y_cov.data.len() {
            y_cov.data[idx] = (y_cov.data[idx] - vt_v.data[idx]) * var_scale;
        }
        // Symmetrize & clamp tiny negative diagonal
        for i in 0..n_test {
            let diag = y_cov.get(i, i).max(0.0);
            y_cov.set(i, i, diag);
            for j in (i + 1)..n_test {
                let avg = 0.5 * (y_cov.get(i, j) + y_cov.get(j, i));
                y_cov.set(i, j, avg);
                y_cov.set(j, i, avg);
            }
        }

        Ok((y_mean, None, Some(y_cov)))
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let (mean, _, _) = self.predict_with_uncertainty(x, false, false)?;
        Ok(mean)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_vec_finite(y, "Target vector y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn log_marginal_likelihood(&self) -> OchreResult<f64> {
        self.log_marginal_likelihood_value_
            .ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GP_REGRESSOR);
        write_str(&mut buf, &self.kernel);
        write_f64(&mut buf, self.length_scale);
        write_f64(&mut buf, self.constant_value);
        write_f64(&mut buf, self.nu);
        write_f64(&mut buf, self.alpha);
        write_f64(&mut buf, self.noise_level);
        write_f64(&mut buf, self.rq_alpha);
        write_bool(&mut buf, self.optimize_hyperparams);
        write_usize(&mut buf, self.n_restarts_optimizer);
        write_bool(&mut buf, self.normalize_y);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);

        write_opt_matrix(&mut buf, &self.x_train_);
        write_f64(&mut buf, self.y_train_mean_);
        write_f64(&mut buf, self.y_train_std_);
        write_opt_matrix(&mut buf, &self.l_);
        write_opt_f64_vec(&mut buf, &self.alpha_);
        write_opt_f64(&mut buf, self.log_marginal_likelihood_value_);
        write_f64(&mut buf, self.length_scale_);
        write_f64(&mut buf, self.constant_value_);
        write_f64(&mut buf, self.noise_level_);
        write_f64(&mut buf, self.rq_alpha_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_GP_REGRESSOR)?;

        let kernel = reader.read_str()?;
        let length_scale = reader.read_f64()?;
        let constant_value = reader.read_f64()?;
        let nu = reader.read_f64()?;
        let alpha = reader.read_f64()?;
        let noise_level = reader.read_f64()?;
        let rq_alpha = reader.read_f64()?;
        let optimize_hyperparams = reader.read_bool()?;
        let n_restarts_optimizer = reader.read_usize()?;
        let normalize_y = reader.read_bool()?;
        let random_state = reader.read_opt_u64()?;
        let saved_device = reader.read_opt_str()?;

        let x_train_ = reader.read_opt_matrix()?;
        let y_train_mean_ = reader.read_f64()?;
        let y_train_std_ = reader.read_f64()?;
        let l_ = reader.read_opt_matrix()?;
        let alpha_ = reader.read_opt_f64_vec()?;
        let log_marginal_likelihood_value_ = reader.read_opt_f64()?;
        let length_scale_ = reader.read_f64()?;
        let constant_value_ = reader.read_f64()?;
        let noise_level_ = reader.read_f64()?;
        let rq_alpha_ = reader.read_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (parsed_device, device_) =
            resolve_loaded_device(saved_device, x_train_.is_some(), device)?;

        Ok(Self {
            kernel,
            length_scale,
            constant_value,
            nu,
            alpha,
            noise_level,
            rq_alpha,
            optimize_hyperparams,
            n_restarts_optimizer,
            normalize_y,
            random_state,
            device_,
            parsed_device,
            x_train_,
            y_train_mean_,
            y_train_std_,
            l_,
            alpha_,
            log_marginal_likelihood_value_,
            length_scale_,
            constant_value_,
            noise_level_,
            rq_alpha_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "GaussianProcessRegressor")]
#[derive(Clone)]
pub struct PyGaussianProcessRegressor {
    inner: GaussianProcessRegressor,
}

pymethods_with_dit!(PyGaussianProcessRegressor, GaussianProcessRegressor, {
    #[new]
    #[pyo3(signature = (
        kernel = "rbf".to_string(),
        length_scale = 1.0,
        constant_value = 1.0,
        nu = 1.5,
        alpha = 1e-10,
        noise_level = 1.0,
        rq_alpha = 1.0,
        optimize_hyperparams = false,
        n_restarts_optimizer = 0,
        normalize_y = false,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        length_scale: f64,
        constant_value: f64,
        nu: f64,
        alpha: f64,
        noise_level: f64,
        rq_alpha: f64,
        optimize_hyperparams: bool,
        n_restarts_optimizer: usize,
        normalize_y: bool,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianProcessRegressor::new(
                kernel,
                length_scale,
                constant_value,
                nu,
                alpha,
                noise_level,
                rq_alpha,
                optimize_hyperparams,
                n_restarts_optimizer,
                normalize_y,
                random_state,
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
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&mat, &y_vec)?;
        Ok(())
    }

    #[pyo3(signature = (x, return_std = false, return_cov = false))]
    #[allow(clippy::type_complexity)]
    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        return_std: bool,
        return_cov: bool,
    ) -> PyResult<(
        Bound<'py, PyBytes>,
        Option<Bound<'py, PyBytes>>,
        Option<(usize, usize, Bound<'py, PyBytes>)>,
    )> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let (mean, std_opt, cov_opt) = self
            .inner
            .predict_with_uncertainty(&mat, return_std, return_cov)?;
        let mean_bytes = f64_slice_to_pybytes(py, &mean)?;
        let std_bytes = match std_opt {
            Some(s) => Some(f64_slice_to_pybytes(py, &s)?),
            None => None,
        };
        let cov_bytes = match cov_opt {
            Some(c) => Some((c.rows, c.cols, f64_slice_to_pybytes(py, &c.data)?)),
            None => None,
        };
        Ok((mean_bytes, std_bytes, cov_bytes))
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&mat, &y_vec)?)
    }

    pub fn log_marginal_likelihood(&self) -> PyResult<f64> {
        Ok(self.inner.log_marginal_likelihood()?)
    }

    #[getter]
    pub fn kernel(&self) -> String {
        self.inner.kernel.clone()
    }
    #[getter]
    pub fn length_scale(&self) -> f64 {
        self.inner.length_scale
    }
    #[getter]
    pub fn constant_value(&self) -> f64 {
        self.inner.constant_value
    }
    #[getter]
    pub fn nu(&self) -> f64 {
        self.inner.nu
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn noise_level(&self) -> f64 {
        self.inner.noise_level
    }
    #[getter]
    pub fn rq_alpha(&self) -> f64 {
        self.inner.rq_alpha
    }
    #[getter]
    pub fn optimize_hyperparams(&self) -> bool {
        self.inner.optimize_hyperparams
    }
    #[getter]
    pub fn n_restarts_optimizer(&self) -> usize {
        self.inner.n_restarts_optimizer
    }
    #[getter]
    pub fn normalize_y(&self) -> bool {
        self.inner.normalize_y
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn length_scale_(&self) -> f64 {
        self.inner.length_scale_
    }
    #[getter]
    pub fn constant_value_(&self) -> f64 {
        self.inner.constant_value_
    }
    #[getter]
    pub fn noise_level_(&self) -> f64 {
        self.inner.noise_level_
    }
    #[getter]
    pub fn rq_alpha_(&self) -> f64 {
        self.inner.rq_alpha_
    }
    #[getter]
    pub fn y_train_mean_(&self) -> f64 {
        self.inner.y_train_mean_
    }
    #[getter]
    pub fn y_train_std_(&self) -> f64 {
        self.inner.y_train_std_
    }
    #[getter]
    pub fn alpha_(&self) -> Option<Vec<f64>> {
        self.inner.alpha_.clone()
    }
    #[getter]
    pub fn log_marginal_likelihood_value_(&self) -> Option<f64> {
        self.inner.log_marginal_likelihood_value_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn x_train_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.x_train_ {
            Some(m) => Ok(Some((m.rows, m.cols, f64_slice_to_pybytes(py, &m.data)?))),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn l_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.l_ {
            Some(m) => Ok(Some((m.rows, m.cols, f64_slice_to_pybytes(py, &m.data)?))),
            None => Ok(None),
        }
    }
});

// ============================================================================
// GaussianProcessClassifier (Binary Laplace + One-vs-Rest Multiclass)
// ============================================================================

#[derive(Clone)]
struct BinaryLaplaceGP {
    y_train_minus_pi: Vec<f64>,
    w_sqrt: Vec<f64>,
    l: Matrix,
    lml: f64,
    params: KernelParams,
}

impl BinaryLaplaceGP {
    fn fit_single_from_d2(
        backend: &Backend<'_>,
        d2: Option<&Matrix>,
        n: usize,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
    ) -> OchreResult<Self> {
        let k = match d2 {
            Some(d) => compute_kernel_from_d2(backend, d, params, true)?,
            None => {
                let mut m = Matrix::zeros(n, n);
                for i in 0..n {
                    m.set(i, i, params.noise_level);
                }
                m
            }
        };
        let mut f = vec![0.0; n];

        let mut pi = vec![0.5; n];
        let mut w = vec![0.25; n];
        let mut w_sqrt = vec![0.5; n];
        let mut a = vec![0.0; n];
        let mut b_mat = Matrix::zeros(n, n);
        let mut b_vec = vec![0.0; n];
        let mut rhs = vec![0.0; n];

        for _iter in 0..max_iter_predict {
            for i in 0..n {
                let p = sigmoid(f[i]);
                pi[i] = p;
                let wi = (p * (1.0 - p)).max(1e-12);
                w[i] = wi;
                w_sqrt[i] = wi.sqrt();
            }

            // B = I + W^{1/2} K W^{1/2}
            for i in 0..n {
                let wsi = w_sqrt[i];
                let row_off = i * n;
                for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                    let val = wsi * k.data[row_off + j] * wsj;
                    b_mat.data[row_off + j] = if i == j { 1.0 + val } else { val };
                }
            }
            let l_iter = cholesky_lower(&b_mat)?;

            // b_vec = W * f + (y_bin - pi)
            for i in 0..n {
                b_vec[i] = w[i] * f[i] + (y_bin[i] - pi[i]);
            }

            // kb = K * b_vec
            let kb = backend.matvec(&k, &b_vec)?;
            for i in 0..n {
                rhs[i] = w_sqrt[i] * kb[i];
            }
            let solved = solve_cholesky_from_l(&l_iter, &rhs);
            for i in 0..n {
                a[i] = b_vec[i] - w_sqrt[i] * solved[i];
            }

            let f_new = backend.matvec(&k, &a)?;
            let max_diff = f
                .iter()
                .zip(f_new.iter())
                .map(|(&old, &new)| (old - new).abs())
                .fold(0.0_f64, f64::max);
            f = f_new;
            if max_diff < 1e-8 {
                break;
            }
        }

        // Recompute pi, w_sqrt, L at converged f
        for i in 0..n {
            let p = sigmoid(f[i]);
            pi[i] = p;
            w_sqrt[i] = (p * (1.0 - p)).max(1e-12).sqrt();
        }
        for i in 0..n {
            let wsi = w_sqrt[i];
            let row_off = i * n;
            for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                let val = wsi * k.data[row_off + j] * wsj;
                b_mat.data[row_off + j] = if i == j { 1.0 + val } else { val };
            }
        }
        let l = cholesky_lower(&b_mat)?;

        let mut log_lik = 0.0;
        for i in 0..n {
            let fi = f[i];
            let log_1_plus_exp = fi.max(0.0) + (-fi.abs()).exp().ln_1p();
            log_lik += y_bin[i] * fi - log_1_plus_exp;
        }
        let mut log_det_b_half = 0.0;
        for i in 0..n {
            log_det_b_half += l.data[i * n + i].ln();
        }
        let lml = -0.5 * dot(&a, &f) + log_lik - log_det_b_half;
        let y_train_minus_pi: Vec<f64> = y_bin
            .iter()
            .zip(pi.iter())
            .map(|(&yi, &pii)| yi - pii)
            .collect();

        Ok(Self {
            y_train_minus_pi,
            w_sqrt,
            l,
            lml,
            params: *params,
        })
    }

    /// Evaluate Laplace LML and analytical gradient wrt log-hyperparameters (Rasmussen & Williams Alg 5.1)
    /// for `Rbf` and `RbfWhite` kernels.
    fn fit_single_lml_and_grad(
        backend: &Backend<'_>,
        d2: &Matrix,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let n = d2.rows;
        let ls = params.length_scale;
        let ls_sq = ls * ls;
        let inv_ls_sq = 1.0 / ls_sq;
        let inv_two_ls_sq = -0.5 * inv_ls_sq;
        let c = params.constant_value;
        let is_rbf_white = params.kind == KernelKind::RbfWhite;

        let mut k0 = Matrix::zeros(n, n);
        let mut k = Matrix::zeros(n, n);
        let extra_diag = if is_rbf_white {
            params.noise_level
        } else {
            0.0
        };
        for i in 0..n {
            let row = i * n;
            k0.data[row + i] = c;
            k.data[row + i] = c + extra_diag;
            for j in 0..i {
                let val = c * (d2.data[row + j].max(0.0) * inv_two_ls_sq).exp();
                k0.data[row + j] = val;
                k0.data[j * n + i] = val;
                k.data[row + j] = val;
                k.data[j * n + i] = val;
            }
        }

        let mut f = vec![0.0; n];
        let mut pi = vec![0.5; n];
        let mut w = vec![0.25; n];
        let mut w_sqrt = vec![0.5; n];
        let mut a = vec![0.0; n];
        let mut b_mat = Matrix::zeros(n, n);
        let mut b_vec = vec![0.0; n];
        let mut rhs = vec![0.0; n];

        for _iter in 0..max_iter_predict {
            for i in 0..n {
                let p = sigmoid(f[i]);
                pi[i] = p;
                let wi = (p * (1.0 - p)).max(1e-12);
                w[i] = wi;
                w_sqrt[i] = wi.sqrt();
            }
            for i in 0..n {
                let wsi = w_sqrt[i];
                let row_off = i * n;
                for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                    let val = wsi * k.data[row_off + j] * wsj;
                    b_mat.data[row_off + j] = if i == j { 1.0 + val } else { val };
                }
            }
            let l_iter = cholesky_lower(&b_mat)?;
            for i in 0..n {
                b_vec[i] = w[i] * f[i] + (y_bin[i] - pi[i]);
            }
            let kb = backend.matvec(&k, &b_vec)?;
            for i in 0..n {
                rhs[i] = w_sqrt[i] * kb[i];
            }
            let solved = solve_cholesky_from_l(&l_iter, &rhs);
            for i in 0..n {
                a[i] = b_vec[i] - w_sqrt[i] * solved[i];
            }
            let f_new = backend.matvec(&k, &a)?;
            let max_diff = f
                .iter()
                .zip(f_new.iter())
                .map(|(&old, &new)| (old - new).abs())
                .fold(0.0_f64, f64::max);
            f = f_new;
            if max_diff < 1e-8 {
                break;
            }
        }

        for i in 0..n {
            let p = sigmoid(f[i]);
            pi[i] = p;
            let wi = (p * (1.0 - p)).max(1e-12);
            w[i] = wi;
            w_sqrt[i] = wi.sqrt();
        }
        for i in 0..n {
            let wsi = w_sqrt[i];
            let row_off = i * n;
            for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                let val = wsi * k.data[row_off + j] * wsj;
                b_mat.data[row_off + j] = if i == j { 1.0 + val } else { val };
            }
        }
        let l = cholesky_lower(&b_mat)?;

        let mut log_lik = 0.0;
        for i in 0..n {
            let fi = f[i];
            let log_1_plus_exp = fi.max(0.0) + (-fi.abs()).exp().ln_1p();
            log_lik += y_bin[i] * fi - log_1_plus_exp;
        }
        let mut log_det_b_half = 0.0;
        for i in 0..n {
            log_det_b_half += l.data[i * n + i].ln();
        }
        let lml = -0.5 * dot(&a, &f) + log_lik - log_det_b_half;

        // R = W^{1/2} * B^{-1} * W^{1/2} via M^T = (L^{-1})^T
        let m_t = invert_lower_triangular_t(&l);
        let mut r_mat = Matrix::zeros(n, n);
        for i in 0..n {
            let row = i * n;
            let mi = &m_t[row + i..row + n];
            let wsi = w_sqrt[i];
            r_mat.data[row + i] = wsi * wsi * dot_unrolled(mi, mi);
            for j in 0..i {
                let mj = &m_t[j * n + i..j * n + n];
                let val = wsi * w_sqrt[j] * dot_unrolled(mi, mj);
                r_mat.data[row + j] = val;
                r_mat.data[j * n + i] = val;
            }
        }

        // s2[i] = -0.5 * (K_{ii} - ||L^{-1} W^{1/2} K_{:,i}||^2) * d^3 log p(y_i | f_i) / df_i^3
        // where d^3 log p / df_i^3 = -pi_i * (1 - pi_i) * (1 - 2 * pi_i)
        let mut s2 = vec![0.0; n];
        let mut wk_col = vec![0.0; n];
        let mut c_col = vec![0.0; n];
        for i in 0..n {
            let k_row = &k.data[i * n..(i + 1) * n];
            for j in 0..n {
                wk_col[j] = w_sqrt[j] * k_row[j];
            }
            solve_lower_triangular_into(&l, &wk_col, &mut c_col);
            let post_var_i = k.data[i * n + i] - dot_unrolled(&c_col, &c_col);
            let p = pi[i];
            let dw_df = p * (1.0 - p) * (1.0 - 2.0 * p);
            s2[i] = -0.5 * post_var_i * dw_df;
        }

        // Precompute u = (I - R K) s2 so s2^T (I - K R) b_m = u^T b_m, where b_m = dK_m * a
        // Since R and K are symmetric: (s2^T (I - K R))^T = s2 - R (K s2)
        let k_s2 = backend.matvec(&k, &s2)?;
        let r_k_s2 = backend.matvec(&r_mat, &k_s2)?;
        let mut u = vec![0.0; n];
        for i in 0..n {
            u[i] = s2[i] - r_k_s2[i];
        }

        // Accumulate analytical gradients for log(c), log(ls), and (if RbfWhite) log(noise_level)
        // d(LML)/d(theta_m) = 0.5 * sum_{i,j} (a_i a_j - R_{ij}) * dK_{ij} + sum_{i,j} u_i * dK_{ij} * a_j
        // By symmetry of dK, off-diagonal (i > j) contributes:
        //   (a_i a_j - R_{ij} + u_i a_j + u_j a_i) * dK_{ij}
        // Diagonal (i == j) contributes:
        //   (0.5 * (a_i^2 - R_{ii}) + u_i a_i) * dK_{ii}
        let mut g_log_c = 0.0;
        let mut g_log_ls = 0.0;
        let mut g_log_noise = 0.0;

        for i in 0..n {
            let row = i * n;
            let ai = a[i];
            let ui = u[i];
            let diag_coeff = 0.5 * (ai * ai - r_mat.data[row + i]) + ui * ai;
            g_log_c += diag_coeff * c;
            if is_rbf_white {
                g_log_noise += diag_coeff * params.noise_level;
            }
            for j in 0..i {
                let aj = a[j];
                let coeff = (ai * aj - r_mat.data[row + j]) + ui * aj + u[j] * ai;
                let k0_ij = k0.data[row + j];
                let ck = coeff * k0_ij;
                g_log_c += ck;
                g_log_ls += ck * (d2.data[row + j].max(0.0) * inv_ls_sq);
            }
        }

        let grad = if is_rbf_white {
            vec![g_log_c, g_log_ls, g_log_noise]
        } else {
            vec![g_log_c, g_log_ls]
        };
        Ok((lml, grad))
    }

    fn predict_pos_proba(
        &self,
        backend: &Backend<'_>,
        x_train: &Matrix,
        x_test: &Matrix,
    ) -> OchreResult<Vec<f64>> {
        let k_star = compute_kernel_matrix(backend, x_test, x_train, &self.params, false)?;
        let f_star = backend.matvec(&k_star, &self.y_train_minus_pi)?;
        let n_test = x_test.rows;
        let n_train = x_train.rows;
        let k_diag = kernel_diag_value(&self.params);

        let probs: Vec<f64> = if n_test <= 256 {
            let mut wk = vec![0.0; n_train];
            let mut v = vec![0.0; n_train];
            let mut out = Vec::with_capacity(n_test);
            for (i, &fs) in f_star.iter().enumerate().take(n_test) {
                let k_row = &k_star.data[i * n_train..(i + 1) * n_train];
                for j in 0..n_train {
                    wk[j] = self.w_sqrt[j] * k_row[j];
                }
                solve_lower_triangular_into(&self.l, &wk, &mut v);
                let var_f = (k_diag - dot_unrolled(&v, &v)).max(0.0);
                // Probit approximation matching sklearn GaussianProcessClassifier
                let kappa = 1.0 / (1.0 + (PI / 8.0) * var_f).sqrt();
                out.push(sigmoid(kappa * fs));
            }
            out
        } else {
            (0..n_test)
                .into_par_iter()
                .map_init(
                    || (vec![0.0; n_train], vec![0.0; n_train]),
                    |(wk, v), i| {
                        let k_row = &k_star.data[i * n_train..(i + 1) * n_train];
                        for j in 0..n_train {
                            wk[j] = self.w_sqrt[j] * k_row[j];
                        }
                        solve_lower_triangular_into(&self.l, wk, v);
                        let var_f = (k_diag - dot_unrolled(v, v)).max(0.0);
                        // Probit approximation matching sklearn GaussianProcessClassifier
                        let kappa = 1.0 / (1.0 + (PI / 8.0) * var_f).sqrt();
                        sigmoid(kappa * f_star[i])
                    },
                )
                .collect()
        };

        Ok(probs)
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        write_f64_vec(buf, &self.y_train_minus_pi);
        write_f64_vec(buf, &self.w_sqrt);
        write_matrix(buf, &self.l);
        write_f64(buf, self.lml);
        write_f64(buf, self.params.length_scale);
        write_f64(buf, self.params.constant_value);
        write_f64(buf, self.params.nu);
        write_f64(buf, self.params.noise_level);
        write_f64(buf, self.params.rq_alpha);
    }

    fn read_from(reader: &mut ByteReader<'_>, kind: KernelKind) -> OchreResult<Self> {
        let y_train_minus_pi = reader.read_f64_vec()?;
        let w_sqrt = reader.read_f64_vec()?;
        let l = reader.read_matrix()?;
        let lml = reader.read_f64()?;
        let length_scale = reader.read_f64()?;
        let constant_value = reader.read_f64()?;
        let nu = reader.read_f64()?;
        let noise_level = reader.read_f64()?;
        let rq_alpha = reader.read_f64()?;
        Ok(Self {
            y_train_minus_pi,
            w_sqrt,
            l,
            lml,
            params: KernelParams {
                kind,
                length_scale,
                constant_value,
                nu,
                noise_level,
                rq_alpha,
            },
        })
    }
}

#[derive(Clone)]
pub struct GaussianProcessClassifier {
    pub kernel: String,
    pub length_scale: f64,
    pub constant_value: f64,
    pub nu: f64,
    pub noise_level: f64,
    pub rq_alpha: f64,
    pub max_iter_predict: usize,
    pub optimize_hyperparams: bool,
    pub n_restarts_optimizer: usize,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    parsed_device: DeviceType,

    // Fitted attributes
    pub x_train_: Option<Matrix>,
    pub classes_: Option<Vec<f64>>,
    estimators_: Vec<BinaryLaplaceGP>,
    pub log_marginal_likelihood_value_: Option<f64>,
    pub n_features_in_: Option<usize>,
}

impl GaussianProcessClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        length_scale: f64,
        constant_value: f64,
        nu: f64,
        noise_level: f64,
        rq_alpha: f64,
        max_iter_predict: usize,
        optimize_hyperparams: bool,
        n_restarts_optimizer: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let _ = KernelKind::from_str(&kernel)?;
        validate_positive_finite("length_scale", length_scale)?;
        validate_positive_finite("constant_value", constant_value)?;
        validate_nu(nu)?;
        validate_positive_finite("noise_level", noise_level)?;
        validate_positive_finite("rq_alpha", rq_alpha)?;
        if max_iter_predict == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter_predict must be at least 1".to_string(),
            ));
        }

        let parsed_device = parse_device(device.as_deref())?;
        Ok(Self {
            kernel,
            length_scale,
            constant_value,
            nu,
            noise_level,
            rq_alpha,
            max_iter_predict,
            optimize_hyperparams,
            n_restarts_optimizer,
            random_state,
            device_: device,
            parsed_device,
            x_train_: None,
            classes_: None,
            estimators_: Vec::new(),
            log_marginal_likelihood_value_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let parsed = parse_device(device.as_deref())?;
        self.parsed_device = parsed;
        self.device_ = device;
        Ok(())
    }

    fn base_kernel_params(&self) -> OchreResult<KernelParams> {
        Ok(KernelParams {
            kind: KernelKind::from_str(&self.kernel)?,
            length_scale: self.length_scale,
            constant_value: self.constant_value,
            nu: self.nu,
            noise_level: self.noise_level,
            rq_alpha: self.rq_alpha,
        })
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_matrix_finite(x, "Feature matrix X")?;
        validate_vec_finite(y, "Target vector y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }

        let n = x.rows;
        let backend = effective_backend(&self.parsed_device, n.saturating_mul(n));
        let base_params = self.base_kernel_params()?;
        let do_opt = self.optimize_hyperparams || self.n_restarts_optimizer > 0;
        let d2 = if base_params.kind == KernelKind::White {
            None
        } else {
            Some(backend.pairwise_l2_sq(x, x)?)
        };

        let mut estimators = Vec::new();
        let mut total_lml = 0.0;

        let fit_binary = |y_bin: &[f64]| -> OchreResult<BinaryLaplaceGP> {
            let best_params = if do_opt {
                match (base_params.kind, d2.as_ref()) {
                    (KernelKind::Rbf | KernelKind::RbfWhite, Some(d2_mat)) => optimize_lml_lbfgs(
                        base_params,
                        self.n_restarts_optimizer,
                        self.random_state,
                        |cand| {
                            BinaryLaplaceGP::fit_single_lml_and_grad(
                                &backend,
                                d2_mat,
                                y_bin,
                                cand,
                                self.max_iter_predict,
                            )
                            .ok()
                        },
                    ),
                    _ => optimize_lml(
                        base_params,
                        self.n_restarts_optimizer,
                        self.random_state,
                        |cand| {
                            BinaryLaplaceGP::fit_single_from_d2(
                                &backend,
                                d2.as_ref(),
                                n,
                                y_bin,
                                cand,
                                self.max_iter_predict,
                            )
                            .ok()
                            .map(|e| e.lml)
                        },
                    ),
                }
            } else {
                base_params
            };
            BinaryLaplaceGP::fit_single_from_d2(
                &backend,
                d2.as_ref(),
                n,
                y_bin,
                &best_params,
                self.max_iter_predict,
            )
        };

        if classes.len() == 2 {
            let pos_cls = classes[1];
            let y_bin: Vec<f64> = y
                .iter()
                .map(|&v| {
                    if (v - pos_cls).abs() <= 1e-12 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();

            let est = fit_binary(&y_bin)?;
            total_lml = est.lml;
            estimators.push(est);
        } else {
            for &cls in &classes {
                let y_bin: Vec<f64> = y
                    .iter()
                    .map(|&v| if (v - cls).abs() <= 1e-12 { 1.0 } else { 0.0 })
                    .collect();

                let est = fit_binary(&y_bin)?;
                total_lml += est.lml;
                estimators.push(est);
            }
        }

        self.x_train_ = Some(x.clone());
        self.classes_ = Some(classes);
        self.estimators_ = estimators;
        self.log_marginal_likelihood_value_ = Some(total_lml);
        self.n_features_in_ = Some(x.cols);
        Ok(())
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let x_train = self.x_train_.as_ref().ok_or(OchreError::NotFitted)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_cols = self.n_features_in_.ok_or(OchreError::NotFitted)?;

        validate_matrix_finite(x, "Feature matrix X")?;
        if x.cols != expected_cols {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }

        let n_test = x.rows;
        let backend = effective_backend(&self.parsed_device, n_test.saturating_mul(x_train.rows));
        let n_classes = classes.len();
        let mut proba = Matrix::zeros(n_test, n_classes);

        if n_classes == 2 {
            let p1 = self.estimators_[0].predict_pos_proba(&backend, x_train, x)?;
            for (i, &p1i) in p1.iter().enumerate().take(n_test) {
                let prob1 = p1i.clamp(0.0, 1.0);
                proba.set(i, 0, 1.0 - prob1);
                proba.set(i, 1, prob1);
            }
        } else {
            let mut class_probs = Vec::with_capacity(n_classes);
            for est in &self.estimators_ {
                class_probs.push(est.predict_pos_proba(&backend, x_train, x)?);
            }
            for i in 0..n_test {
                let mut row_sum = 0.0;
                for cp in class_probs.iter().take(n_classes) {
                    row_sum += cp[i];
                }
                if row_sum > 0.0 && row_sum.is_finite() {
                    for (c, cp) in class_probs.iter().enumerate().take(n_classes) {
                        proba.set(i, c, cp[i] / row_sum);
                    }
                } else {
                    let uniform = 1.0 / (n_classes as f64);
                    for c in 0..n_classes {
                        proba.set(i, c, uniform);
                    }
                }
            }
        }

        Ok(proba)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let proba = self.predict_proba(x)?;
        let n_test = proba.rows;
        let n_classes = proba.cols;
        let mut preds = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let row = &proba.data[i * n_classes..(i + 1) * n_classes];
            let mut best_c = 0;
            let mut best_p = row[0];
            for (c, &p) in row.iter().enumerate().skip(1) {
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            preds.push(classes[best_c]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_vec_finite(y, "Target vector y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() <= 1e-12)
            .count();
        Ok((correct as f64) / (y.len() as f64))
    }

    pub fn log_marginal_likelihood(&self) -> OchreResult<f64> {
        self.log_marginal_likelihood_value_
            .ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GP_CLASSIFIER);
        write_str(&mut buf, &self.kernel);
        write_f64(&mut buf, self.length_scale);
        write_f64(&mut buf, self.constant_value);
        write_f64(&mut buf, self.nu);
        write_f64(&mut buf, self.noise_level);
        write_f64(&mut buf, self.rq_alpha);
        write_usize(&mut buf, self.max_iter_predict);
        write_bool(&mut buf, self.optimize_hyperparams);
        write_usize(&mut buf, self.n_restarts_optimizer);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);

        write_opt_matrix(&mut buf, &self.x_train_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_usize(&mut buf, self.estimators_.len());
        for est in &self.estimators_ {
            est.write_to(&mut buf);
        }
        write_opt_f64(&mut buf, self.log_marginal_likelihood_value_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_GP_CLASSIFIER)?;

        let kernel = reader.read_str()?;
        let kind = KernelKind::from_str(&kernel)?;
        let length_scale = reader.read_f64()?;
        let constant_value = reader.read_f64()?;
        let nu = reader.read_f64()?;
        let noise_level = reader.read_f64()?;
        let rq_alpha = reader.read_f64()?;
        let max_iter_predict = reader.read_usize()?;
        let optimize_hyperparams = reader.read_bool()?;
        let n_restarts_optimizer = reader.read_usize()?;
        let random_state = reader.read_opt_u64()?;
        let saved_device = reader.read_opt_str()?;

        let x_train_ = reader.read_opt_matrix()?;
        let classes_ = reader.read_opt_f64_vec()?;
        let n_est = reader.read_usize()?;
        let mut estimators_ = Vec::with_capacity(n_est);
        for _ in 0..n_est {
            estimators_.push(BinaryLaplaceGP::read_from(&mut reader, kind)?);
        }
        let log_marginal_likelihood_value_ = reader.read_opt_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (parsed_device, device_) =
            resolve_loaded_device(saved_device, x_train_.is_some(), device)?;

        Ok(Self {
            kernel,
            length_scale,
            constant_value,
            nu,
            noise_level,
            rq_alpha,
            max_iter_predict,
            optimize_hyperparams,
            n_restarts_optimizer,
            random_state,
            device_,
            parsed_device,
            x_train_,
            classes_,
            estimators_,
            log_marginal_likelihood_value_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "GaussianProcessClassifier")]
#[derive(Clone)]
pub struct PyGaussianProcessClassifier {
    inner: GaussianProcessClassifier,
}

pymethods_with_dit!(PyGaussianProcessClassifier, GaussianProcessClassifier, {
    #[new]
    #[pyo3(signature = (
        kernel = "rbf".to_string(),
        length_scale = 1.0,
        constant_value = 1.0,
        nu = 1.5,
        noise_level = 1.0,
        rq_alpha = 1.0,
        max_iter_predict = 100,
        optimize_hyperparams = false,
        n_restarts_optimizer = 0,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        length_scale: f64,
        constant_value: f64,
        nu: f64,
        noise_level: f64,
        rq_alpha: f64,
        max_iter_predict: usize,
        optimize_hyperparams: bool,
        n_restarts_optimizer: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianProcessClassifier::new(
                kernel,
                length_scale,
                constant_value,
                nu,
                noise_level,
                rq_alpha,
                max_iter_predict,
                optimize_hyperparams,
                n_restarts_optimizer,
                random_state,
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
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&mat, &y_vec)?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let proba = self.inner.predict_proba(&mat)?;
        Ok((
            proba.rows,
            proba.cols,
            f64_slice_to_pybytes(py, &proba.data)?,
        ))
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&mat, &y_vec)?)
    }

    pub fn log_marginal_likelihood(&self) -> PyResult<f64> {
        Ok(self.inner.log_marginal_likelihood()?)
    }

    #[getter]
    pub fn kernel(&self) -> String {
        self.inner.kernel.clone()
    }
    #[getter]
    pub fn length_scale(&self) -> f64 {
        self.inner.length_scale
    }
    #[getter]
    pub fn constant_value(&self) -> f64 {
        self.inner.constant_value
    }
    #[getter]
    pub fn nu(&self) -> f64 {
        self.inner.nu
    }
    #[getter]
    pub fn noise_level(&self) -> f64 {
        self.inner.noise_level
    }
    #[getter]
    pub fn rq_alpha(&self) -> f64 {
        self.inner.rq_alpha
    }
    #[getter]
    pub fn max_iter_predict(&self) -> usize {
        self.inner.max_iter_predict
    }
    #[getter]
    pub fn optimize_hyperparams(&self) -> bool {
        self.inner.optimize_hyperparams
    }
    #[getter]
    pub fn n_restarts_optimizer(&self) -> usize {
        self.inner.n_restarts_optimizer
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }
    #[getter]
    pub fn n_classes_(&self) -> Option<usize> {
        self.inner.classes_.as_ref().map(|c| c.len())
    }
    #[getter]
    pub fn log_marginal_likelihood_value_(&self) -> Option<f64> {
        self.inner.log_marginal_likelihood_value_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianProcessRegressor>()?;
    m.add_class::<PyGaussianProcessClassifier>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_regression_data() -> (Matrix, Vec<f64>) {
        let mut x_data = Vec::new();
        let mut y = Vec::new();
        for i in 0..24 {
            let t = -3.0 + 6.0 * (i as f64) / 23.0;
            x_data.push(t);
            x_data.push(0.5 * t * t);
            y.push(t.sin() + 0.1 * t);
        }
        let x = Matrix::from_vec(24, 2, x_data).expect("valid matrix");
        (x, y)
    }

    fn sample_classification_data() -> (Matrix, Vec<f64>) {
        let mut x_data = Vec::new();
        let mut y = Vec::new();
        for i in 0..24 {
            let t = -2.5 + 5.0 * (i as f64) / 23.0;
            x_data.push(t);
            x_data.push((t * 1.3).cos());
            y.push(if t > 0.0 { 1.0 } else { 0.0 });
        }
        let x = Matrix::from_vec(24, 2, x_data).expect("valid matrix");
        (x, y)
    }

    #[test]
    fn test_gpr_analytical_grad_matches_finite_diff() {
        let (x, y) = sample_regression_data();
        let d2 = Backend::Cpu.pairwise_l2_sq(&x, &x).expect("d2");
        let eps = 1e-6;

        for kind in [KernelKind::Rbf, KernelKind::RbfWhite] {
            let base = KernelParams {
                kind,
                length_scale: 1.3,
                constant_value: 2.1,
                nu: 1.5,
                noise_level: 0.25,
                rq_alpha: 1.0,
            };
            let theta = params_to_log_vec(&base);
            let (_lml, grad) =
                GaussianProcessRegressor::eval_gpr_lml_and_grad(&d2, &y, 1e-4, &base)
                    .expect("eval_gpr_lml_and_grad");
            assert_eq!(grad.len(), theta.len());

            for m in 0..theta.len() {
                let mut t_plus = theta.clone();
                let mut t_minus = theta.clone();
                t_plus[m] += eps;
                t_minus[m] -= eps;
                let p_plus = apply_log_vec(&base, &t_plus);
                let p_minus = apply_log_vec(&base, &t_minus);
                let lml_plus = GaussianProcessRegressor::eval_gpr_fit_from_d2(
                    &Backend::Cpu,
                    Some(&d2),
                    x.rows,
                    &y,
                    1e-4,
                    &p_plus,
                )
                .expect("plus")
                .2;
                let lml_minus = GaussianProcessRegressor::eval_gpr_fit_from_d2(
                    &Backend::Cpu,
                    Some(&d2),
                    x.rows,
                    &y,
                    1e-4,
                    &p_minus,
                )
                .expect("minus")
                .2;
                let fd = (lml_plus - lml_minus) / (2.0 * eps);
                let err = (grad[m] - fd).abs();
                let scale = 1.0 + fd.abs();
                assert!(
                    err / scale < 1e-5,
                    "GPR grad mismatch for {kind:?} dim {m}: analytical={} fd={}",
                    grad[m],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_gpc_analytical_grad_matches_finite_diff() {
        let (x, y) = sample_classification_data();
        let d2 = Backend::Cpu.pairwise_l2_sq(&x, &x).expect("d2");
        let eps = 1e-5;

        for kind in [KernelKind::Rbf, KernelKind::RbfWhite] {
            let base = KernelParams {
                kind,
                length_scale: 1.2,
                constant_value: 1.7,
                nu: 1.5,
                noise_level: 0.15,
                rq_alpha: 1.0,
            };
            let theta = params_to_log_vec(&base);
            let (_lml, grad) =
                BinaryLaplaceGP::fit_single_lml_and_grad(&Backend::Cpu, &d2, &y, &base, 100)
                    .expect("fit_single_lml_and_grad");
            assert_eq!(grad.len(), theta.len());

            for m in 0..theta.len() {
                let mut t_plus = theta.clone();
                let mut t_minus = theta.clone();
                t_plus[m] += eps;
                t_minus[m] -= eps;
                let p_plus = apply_log_vec(&base, &t_plus);
                let p_minus = apply_log_vec(&base, &t_minus);
                let lml_plus = BinaryLaplaceGP::fit_single_from_d2(
                    &Backend::Cpu,
                    Some(&d2),
                    x.rows,
                    &y,
                    &p_plus,
                    100,
                )
                .expect("plus")
                .lml;
                let lml_minus = BinaryLaplaceGP::fit_single_from_d2(
                    &Backend::Cpu,
                    Some(&d2),
                    x.rows,
                    &y,
                    &p_minus,
                    100,
                )
                .expect("minus")
                .lml;
                let fd = (lml_plus - lml_minus) / (2.0 * eps);
                let err = (grad[m] - fd).abs();
                let scale = 1.0 + fd.abs();
                assert!(
                    err / scale < 1e-4,
                    "GPC grad mismatch for {kind:?} dim {m}: analytical={} fd={}",
                    grad[m],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_gpr_and_gpc_lbfgs_optimization_and_predict_paths() {
        let (x_reg, y_reg) = sample_regression_data();
        let mut gpr = GaussianProcessRegressor::new(
            "rbf".to_string(),
            0.5,
            1.0,
            1.5,
            1e-4,
            1.0,
            1.0,
            true,
            0,
            false,
            Some(42),
            None,
        )
        .expect("new gpr");
        gpr.fit(&x_reg, &y_reg).expect("fit gpr");
        let (mean, std_opt, _) = gpr
            .predict_with_uncertainty(&x_reg, true, false)
            .expect("predict gpr");
        assert_eq!(mean.len(), x_reg.rows);
        assert!(
            std_opt
                .expect("std")
                .iter()
                .all(|&s| s.is_finite() && s >= 0.0)
        );

        let bytes = gpr.to_bytes();
        let loaded = GaussianProcessRegressor::from_bytes(&bytes, None).expect("from_bytes");
        let mean2 = loaded.predict(&x_reg).expect("predict loaded");
        for (a, b) in mean.iter().zip(mean2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        let (x_cls, y_cls) = sample_classification_data();
        let mut gpc = GaussianProcessClassifier::new(
            "rbf".to_string(),
            0.5,
            1.0,
            1.5,
            1.0,
            1.0,
            100,
            true,
            0,
            Some(42),
            None,
        )
        .expect("new gpc");
        gpc.fit(&x_cls, &y_cls).expect("fit gpc");
        let acc = gpc.score(&x_cls, &y_cls).expect("score gpc");
        assert!(acc >= 0.85, "expected high training accuracy, got {acc}");

        // Exercise both small (<= 256) and large (> 256) predict_pos_proba paths
        let mut x_big = Vec::with_capacity(300 * 2);
        for i in 0..300 {
            let t = -2.5 + 5.0 * (i as f64) / 299.0;
            x_big.push(t);
            x_big.push((t * 1.3).cos());
        }
        let x_big_mat = Matrix::from_vec(300, 2, x_big).expect("big mat");
        let proba = gpc.predict_proba(&x_big_mat).expect("predict_proba big");
        assert_eq!(proba.rows, 300);
        assert_eq!(proba.cols, 2);
        for r in 0..300 {
            let sum = proba.get(r, 0) + proba.get(r, 1);
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }
}
