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

#[inline]
fn kernel_diag_value(params: &KernelParams) -> f64 {
    match params.kind {
        KernelKind::White => params.noise_level,
        KernelKind::RbfWhite => params.constant_value + params.noise_level,
        _ => params.constant_value,
    }
}

/// Lower-triangular Cholesky factorization: A = L * L^T.
fn cholesky_lower(a: &Matrix) -> OchreResult<Matrix> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            let row_i = &l.data[i * n..i * n + j];
            let row_j = &l.data[j * n..j * n + j];
            for k in 0..j {
                sum += row_i[k] * row_j[k];
            }
            if i == j {
                let diag = a.data[i * n + i] - sum;
                if !diag.is_finite() || diag <= 1e-15 {
                    return Err(OchreError::SingularMatrix(
                        "Kernel matrix is not positive definite; consider increasing alpha or noise_level".to_string(),
                    ));
                }
                l.data[i * n + i] = diag.sqrt();
            } else {
                let ljj = l.data[j * n + j];
                if ljj.abs() <= 1e-15 {
                    return Err(OchreError::SingularMatrix(
                        "Zero diagonal encountered during Cholesky factorization".to_string(),
                    ));
                }
                l.data[i * n + j] = (a.data[i * n + j] - sum) / ljj;
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve L * z = b where L is lower triangular (n x n).
fn solve_lower_triangular(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        let row_i = &l.data[i * n..i * n + i];
        for k in 0..i {
            sum += row_i[k] * z[k];
        }
        z[i] = (b[i] - sum) / l.data[i * n + i];
    }
    z
}

/// Backward substitution: solve L^T * x = z where L is lower triangular (n x n).
fn solve_upper_triangular_t(l: &Matrix, z: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for (k, &xk) in x.iter().enumerate().take(n).skip(i + 1) {
            sum += l.data[k * n + i] * xk;
        }
        x[i] = (z[i] - sum) / l.data[i * n + i];
    }
    x
}

/// Solve (L * L^T) * x = b given lower-triangular Cholesky factor L.
fn solve_cholesky_from_l(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let z = solve_lower_triangular(l, b);
    solve_upper_triangular_t(l, &z)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
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

    fn eval_gpr_fit(
        backend: &Backend<'_>,
        x: &Matrix,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
    ) -> OchreResult<(Matrix, Vec<f64>, f64)> {
        let mut k = compute_kernel_matrix(backend, x, x, params, true)?;
        let n = x.rows;
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

        let backend = get_backend(&self.parsed_device);
        let base_params = self.base_kernel_params()?;

        let best_params = if self.optimize_hyperparams || self.n_restarts_optimizer > 0 {
            optimize_lml(
                base_params,
                self.n_restarts_optimizer,
                self.random_state,
                |cand| {
                    Self::eval_gpr_fit(&backend, x, &y_norm, self.alpha, cand)
                        .ok()
                        .map(|r| r.2)
                },
            )
        } else {
            base_params
        };

        let (l, alpha_vec, lml) =
            Self::eval_gpr_fit(&backend, x, &y_norm, self.alpha, &best_params)?;

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

        let backend = get_backend(&self.parsed_device);
        let params = self.fitted_kernel_params()?;
        let k_trans = compute_kernel_matrix(&backend, x, x_train, &params, false)?;

        // y_mean = y_std * (K_trans @ alpha_) + y_mean
        let raw_mean = backend.matvec(&k_trans, alpha_vec)?;
        let y_mean: Vec<f64> = raw_mean
            .into_iter()
            .map(|v| v * self.y_train_std_ + self.y_train_mean_)
            .collect();

        let n_test = x.rows;
        let n_train = x_train.rows;
        let var_scale = self.y_train_std_ * self.y_train_std_;

        if !return_std && !return_cov {
            return Ok((y_mean, None, None));
        }

        // Solve L * V = K_trans^T => V has shape (n_train, n_test), stored column-by-column
        let v_cols: Vec<Vec<f64>> = (0..n_test)
            .into_par_iter()
            .map(|i| {
                let row = &k_trans.data[i * n_train..(i + 1) * n_train];
                solve_lower_triangular(l, row)
            })
            .collect();

        if return_std {
            let k_diag = kernel_diag_value(&params);
            let y_std: Vec<f64> = v_cols
                .into_par_iter()
                .map(|v| {
                    let v_sq = dot(&v, &v);
                    ((k_diag - v_sq).max(0.0) * var_scale).sqrt()
                })
                .collect();
            return Ok((y_mean, Some(y_std), None));
        }

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
    fn fit_single(
        backend: &Backend<'_>,
        x: &Matrix,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
    ) -> OchreResult<Self> {
        let n = x.rows;
        let k = compute_kernel_matrix(backend, x, x, params, true)?;
        let mut f = vec![0.0; n];

        let mut pi = vec![0.5; n];
        let mut w_sqrt = vec![0.5; n];
        let mut a = vec![0.0; n];

        for _iter in 0..max_iter_predict {
            let mut w = vec![0.0; n];
            for i in 0..n {
                let p = sigmoid(f[i]);
                pi[i] = p;
                let wi = (p * (1.0 - p)).max(1e-12);
                w[i] = wi;
                w_sqrt[i] = wi.sqrt();
            }

            // B = I + W^{1/2} K W^{1/2}
            let mut b_mat = Matrix::zeros(n, n);
            for i in 0..n {
                let wsi = w_sqrt[i];
                for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                    let val = wsi * k.data[i * n + j] * wsj;
                    b_mat.data[i * n + j] = if i == j { 1.0 + val } else { val };
                }
            }
            let l_iter = cholesky_lower(&b_mat)?;

            // b_vec = W * f + (y_bin - pi)
            let mut b_vec = vec![0.0; n];
            for i in 0..n {
                b_vec[i] = w[i] * f[i] + (y_bin[i] - pi[i]);
            }

            // kb = K * b_vec
            let kb = backend.matvec(&k, &b_vec)?;
            let mut rhs = vec![0.0; n];
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
        let mut b_mat = Matrix::zeros(n, n);
        for i in 0..n {
            let wsi = w_sqrt[i];
            for (j, &wsj) in w_sqrt.iter().enumerate().take(n) {
                let val = wsi * k.data[i * n + j] * wsj;
                b_mat.data[i * n + j] = if i == j { 1.0 + val } else { val };
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

        let probs: Vec<f64> = (0..n_test)
            .into_par_iter()
            .map(|i| {
                let k_row = &k_star.data[i * n_train..(i + 1) * n_train];
                let mut wk = vec![0.0; n_train];
                for j in 0..n_train {
                    wk[j] = self.w_sqrt[j] * k_row[j];
                }
                let v = solve_lower_triangular(&self.l, &wk);
                let var_f = (k_diag - dot(&v, &v)).max(0.0);
                // Probit approximation matching sklearn GaussianProcessClassifier
                let kappa = 1.0 / (1.0 + (PI / 8.0) * var_f).sqrt();
                sigmoid(kappa * f_star[i])
            })
            .collect();

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

        let backend = get_backend(&self.parsed_device);
        let base_params = self.base_kernel_params()?;
        let do_opt = self.optimize_hyperparams || self.n_restarts_optimizer > 0;

        let mut estimators = Vec::new();
        let mut total_lml = 0.0;

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

            let best_params = if do_opt {
                optimize_lml(
                    base_params,
                    self.n_restarts_optimizer,
                    self.random_state,
                    |cand| {
                        BinaryLaplaceGP::fit_single(
                            &backend,
                            x,
                            &y_bin,
                            cand,
                            self.max_iter_predict,
                        )
                        .ok()
                        .map(|e| e.lml)
                    },
                )
            } else {
                base_params
            };

            let est = BinaryLaplaceGP::fit_single(
                &backend,
                x,
                &y_bin,
                &best_params,
                self.max_iter_predict,
            )?;
            total_lml = est.lml;
            estimators.push(est);
        } else {
            for &cls in &classes {
                let y_bin: Vec<f64> = y
                    .iter()
                    .map(|&v| if (v - cls).abs() <= 1e-12 { 1.0 } else { 0.0 })
                    .collect();

                let best_params = if do_opt {
                    optimize_lml(
                        base_params,
                        self.n_restarts_optimizer,
                        self.random_state,
                        |cand| {
                            BinaryLaplaceGP::fit_single(
                                &backend,
                                x,
                                &y_bin,
                                cand,
                                self.max_iter_predict,
                            )
                            .ok()
                            .map(|e| e.lml)
                        },
                    )
                } else {
                    base_params
                };

                let est = BinaryLaplaceGP::fit_single(
                    &backend,
                    x,
                    &y_bin,
                    &best_params,
                    self.max_iter_predict,
                )?;
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

        let backend = get_backend(&self.parsed_device);
        let n_test = x.rows;
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
