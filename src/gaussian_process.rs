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
use crate::matrix::{
    Matrix, cholesky_lower_with_diag_shift_into, dot_slice_unrolled, r2_score,
    solve_lower_triangular_4col_4row_into, solve_lower_triangular_4row_into,
    solve_upper_triangular_t_4row_into,
};
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
const GPU_MIN_ELEMS: usize = 524_288;
const CPU_PAR_THRESHOLD: usize = 512;
const CPU_PAR_ELEMS: usize = 262_144;

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

#[inline(always)]
fn row_sq_norm_unrolled(row: &[f64]) -> f64 {
    let cols = row.len();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut c = 0;
    while c + 4 <= cols {
        let v0 = row[c];
        let v1 = row[c + 1];
        let v2 = row[c + 2];
        let v3 = row[c + 3];
        s0 += v0 * v0;
        s1 += v1 * v1;
        s2 += v2 * v2;
        s3 += v3 * v3;
        c += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while c < cols {
        sum += row[c] * row[c];
        c += 1;
    }
    sum
}

#[inline(always)]
fn direct_l2_sq_1x1(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for c in 0..a.len() {
        let d = a[c] - b[c];
        sum += d * d;
    }
    sum
}

#[inline(always)]
fn direct_l2_sq_4x4(a: [&[f64]; 4], b: [&[f64]; 4]) -> [[f64; 4]; 4] {
    let mut d = [[0.0f64; 4]; 4];
    for c in 0..a[0].len() {
        let av0 = a[0][c];
        let av1 = a[1][c];
        let av2 = a[2][c];
        let av3 = a[3][c];
        let bv0 = b[0][c];
        let bv1 = b[1][c];
        let bv2 = b[2][c];
        let bv3 = b[3][c];

        let t00 = av0 - bv0;
        let t01 = av0 - bv1;
        let t02 = av0 - bv2;
        let t03 = av0 - bv3;
        d[0][0] += t00 * t00;
        d[0][1] += t01 * t01;
        d[0][2] += t02 * t02;
        d[0][3] += t03 * t03;

        let t10 = av1 - bv0;
        let t11 = av1 - bv1;
        let t12 = av1 - bv2;
        let t13 = av1 - bv3;
        d[1][0] += t10 * t10;
        d[1][1] += t11 * t11;
        d[1][2] += t12 * t12;
        d[1][3] += t13 * t13;

        let t20 = av2 - bv0;
        let t21 = av2 - bv1;
        let t22 = av2 - bv2;
        let t23 = av2 - bv3;
        d[2][0] += t20 * t20;
        d[2][1] += t21 * t21;
        d[2][2] += t22 * t22;
        d[2][3] += t23 * t23;

        let t30 = av3 - bv0;
        let t31 = av3 - bv1;
        let t32 = av3 - bv2;
        let t33 = av3 - bv3;
        d[3][0] += t30 * t30;
        d[3][1] += t31 * t31;
        d[3][2] += t32 * t32;
        d[3][3] += t33 * t33;
    }
    d
}

#[inline(always)]
fn dot_4x4(a: [&[f64]; 4], b: [&[f64]; 4]) -> [[f64; 4]; 4] {
    let mut s = [[0.0f64; 4]; 4];
    for c in 0..a[0].len() {
        let av0 = a[0][c];
        let av1 = a[1][c];
        let av2 = a[2][c];
        let av3 = a[3][c];
        let bv0 = b[0][c];
        let bv1 = b[1][c];
        let bv2 = b[2][c];
        let bv3 = b[3][c];

        s[0][0] += av0 * bv0;
        s[0][1] += av0 * bv1;
        s[0][2] += av0 * bv2;
        s[0][3] += av0 * bv3;

        s[1][0] += av1 * bv0;
        s[1][1] += av1 * bv1;
        s[1][2] += av1 * bv2;
        s[1][3] += av1 * bv3;

        s[2][0] += av2 * bv0;
        s[2][1] += av2 * bv1;
        s[2][2] += av2 * bv2;
        s[2][3] += av2 * bv3;

        s[3][0] += av3 * bv0;
        s[3][1] += av3 * bv1;
        s[3][2] += av3 * bv2;
        s[3][3] += av3 * bv3;
    }
    s
}

/// Sequential 4x4 register-blocked pairwise squared L2 distance matrix on CPU
/// for medium/small matrices (`n_a * n_b < CPU_PAR_ELEMS`), avoiding Rayon wakeups.
fn compute_pairwise_l2_sq_seq(a: &Matrix, b: &Matrix, is_self: bool) -> Matrix {
    let na = a.rows;
    let nb = b.rows;
    let cols = a.cols;
    let mut d2 = Matrix::zeros(na, nb);
    let use_direct = cols <= 16;

    if is_self && na == nb {
        let norms = if use_direct {
            Vec::new()
        } else {
            a.data
                .chunks_exact(cols)
                .map(row_sq_norm_unrolled)
                .collect()
        };
        let mut i = 0;
        while i + 4 <= na {
            let x0 = &a.data[i * cols..(i + 1) * cols];
            let x1 = &a.data[(i + 1) * cols..(i + 2) * cols];
            let x2 = &a.data[(i + 2) * cols..(i + 3) * cols];
            let x3 = &a.data[(i + 3) * cols..(i + 4) * cols];

            let mut j = 0;
            if use_direct {
                while j + 4 <= i {
                    let y0 = &a.data[j * cols..(j + 1) * cols];
                    let y1 = &a.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &a.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &a.data[(j + 3) * cols..(j + 4) * cols];
                    let d = direct_l2_sq_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    for c in 0..4 {
                        d2.data[i * nb + j + c] = d[0][c];
                        d2.data[(i + 1) * nb + j + c] = d[1][c];
                        d2.data[(i + 2) * nb + j + c] = d[2][c];
                        d2.data[(i + 3) * nb + j + c] = d[3][c];
                    }
                    j += 4;
                }
                d2.data[(i + 1) * nb + i] = direct_l2_sq_1x1(x1, x0);
                d2.data[(i + 2) * nb + i] = direct_l2_sq_1x1(x2, x0);
                d2.data[(i + 2) * nb + i + 1] = direct_l2_sq_1x1(x2, x1);
                d2.data[(i + 3) * nb + i] = direct_l2_sq_1x1(x3, x0);
                d2.data[(i + 3) * nb + i + 1] = direct_l2_sq_1x1(x3, x1);
                d2.data[(i + 3) * nb + i + 2] = direct_l2_sq_1x1(x3, x2);
            } else {
                let ni0 = norms[i];
                let ni1 = norms[i + 1];
                let ni2 = norms[i + 2];
                let ni3 = norms[i + 3];
                while j + 4 <= i {
                    let y0 = &a.data[j * cols..(j + 1) * cols];
                    let y1 = &a.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &a.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &a.data[(j + 3) * cols..(j + 4) * cols];
                    let dots = dot_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    for c in 0..4 {
                        let nj = norms[j + c];
                        d2.data[i * nb + j + c] = (ni0 + nj - 2.0 * dots[0][c]).max(0.0);
                        d2.data[(i + 1) * nb + j + c] = (ni1 + nj - 2.0 * dots[1][c]).max(0.0);
                        d2.data[(i + 2) * nb + j + c] = (ni2 + nj - 2.0 * dots[2][c]).max(0.0);
                        d2.data[(i + 3) * nb + j + c] = (ni3 + nj - 2.0 * dots[3][c]).max(0.0);
                    }
                    j += 4;
                }
                d2.data[(i + 1) * nb + i] = (ni1 + ni0 - 2.0 * dot_slice_unrolled(x1, x0)).max(0.0);
                d2.data[(i + 2) * nb + i] = (ni2 + ni0 - 2.0 * dot_slice_unrolled(x2, x0)).max(0.0);
                d2.data[(i + 2) * nb + i + 1] =
                    (ni2 + ni1 - 2.0 * dot_slice_unrolled(x2, x1)).max(0.0);
                d2.data[(i + 3) * nb + i] = (ni3 + ni0 - 2.0 * dot_slice_unrolled(x3, x0)).max(0.0);
                d2.data[(i + 3) * nb + i + 1] =
                    (ni3 + ni1 - 2.0 * dot_slice_unrolled(x3, x1)).max(0.0);
                d2.data[(i + 3) * nb + i + 2] =
                    (ni3 + ni2 - 2.0 * dot_slice_unrolled(x3, x2)).max(0.0);
            }
            i += 4;
        }
        while i < na {
            let xi = &a.data[i * cols..(i + 1) * cols];
            for j in 0..i {
                let xj = &a.data[j * cols..(j + 1) * cols];
                d2.data[i * nb + j] = if use_direct {
                    direct_l2_sq_1x1(xi, xj)
                } else {
                    (norms[i] + norms[j] - 2.0 * dot_slice_unrolled(xi, xj)).max(0.0)
                };
            }
            i += 1;
        }
        mirror_lower_to_upper(&mut d2);
        return d2;
    }

    let (a_norms, b_norms) = if use_direct {
        (Vec::new(), Vec::new())
    } else {
        (
            a.data
                .chunks_exact(cols)
                .map(row_sq_norm_unrolled)
                .collect(),
            b.data
                .chunks_exact(cols)
                .map(row_sq_norm_unrolled)
                .collect(),
        )
    };

    let mut i = 0;
    while i + 4 <= na {
        let x0 = &a.data[i * cols..(i + 1) * cols];
        let x1 = &a.data[(i + 1) * cols..(i + 2) * cols];
        let x2 = &a.data[(i + 2) * cols..(i + 3) * cols];
        let x3 = &a.data[(i + 3) * cols..(i + 4) * cols];
        let mut j = 0;
        if use_direct {
            while j + 4 <= nb {
                let y0 = &b.data[j * cols..(j + 1) * cols];
                let y1 = &b.data[(j + 1) * cols..(j + 2) * cols];
                let y2 = &b.data[(j + 2) * cols..(j + 3) * cols];
                let y3 = &b.data[(j + 3) * cols..(j + 4) * cols];
                let d = direct_l2_sq_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                for c in 0..4 {
                    d2.data[i * nb + j + c] = d[0][c];
                    d2.data[(i + 1) * nb + j + c] = d[1][c];
                    d2.data[(i + 2) * nb + j + c] = d[2][c];
                    d2.data[(i + 3) * nb + j + c] = d[3][c];
                }
                j += 4;
            }
            while j < nb {
                let yj = &b.data[j * cols..(j + 1) * cols];
                d2.data[i * nb + j] = direct_l2_sq_1x1(x0, yj);
                d2.data[(i + 1) * nb + j] = direct_l2_sq_1x1(x1, yj);
                d2.data[(i + 2) * nb + j] = direct_l2_sq_1x1(x2, yj);
                d2.data[(i + 3) * nb + j] = direct_l2_sq_1x1(x3, yj);
                j += 1;
            }
        } else {
            let ni0 = a_norms[i];
            let ni1 = a_norms[i + 1];
            let ni2 = a_norms[i + 2];
            let ni3 = a_norms[i + 3];
            while j + 4 <= nb {
                let y0 = &b.data[j * cols..(j + 1) * cols];
                let y1 = &b.data[(j + 1) * cols..(j + 2) * cols];
                let y2 = &b.data[(j + 2) * cols..(j + 3) * cols];
                let y3 = &b.data[(j + 3) * cols..(j + 4) * cols];
                let dots = dot_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                for c in 0..4 {
                    let nj = b_norms[j + c];
                    d2.data[i * nb + j + c] = (ni0 + nj - 2.0 * dots[0][c]).max(0.0);
                    d2.data[(i + 1) * nb + j + c] = (ni1 + nj - 2.0 * dots[1][c]).max(0.0);
                    d2.data[(i + 2) * nb + j + c] = (ni2 + nj - 2.0 * dots[2][c]).max(0.0);
                    d2.data[(i + 3) * nb + j + c] = (ni3 + nj - 2.0 * dots[3][c]).max(0.0);
                }
                j += 4;
            }
            while j < nb {
                let yj = &b.data[j * cols..(j + 1) * cols];
                let nj = b_norms[j];
                d2.data[i * nb + j] = (ni0 + nj - 2.0 * dot_slice_unrolled(x0, yj)).max(0.0);
                d2.data[(i + 1) * nb + j] = (ni1 + nj - 2.0 * dot_slice_unrolled(x1, yj)).max(0.0);
                d2.data[(i + 2) * nb + j] = (ni2 + nj - 2.0 * dot_slice_unrolled(x2, yj)).max(0.0);
                d2.data[(i + 3) * nb + j] = (ni3 + nj - 2.0 * dot_slice_unrolled(x3, yj)).max(0.0);
                j += 1;
            }
        }
        i += 4;
    }
    while i < na {
        let xi = &a.data[i * cols..(i + 1) * cols];
        for j in 0..nb {
            let yj = &b.data[j * cols..(j + 1) * cols];
            d2.data[i * nb + j] = if use_direct {
                direct_l2_sq_1x1(xi, yj)
            } else {
                (a_norms[i] + b_norms[j] - 2.0 * dot_slice_unrolled(xi, yj)).max(0.0)
            };
        }
        i += 1;
    }
    d2
}

fn compute_pairwise_l2_sq_fast(
    backend: &Backend<'_>,
    a: &Matrix,
    b: &Matrix,
    is_self: bool,
) -> OchreResult<Matrix> {
    if a.cols != b.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: b.cols,
            expected: a.cols,
        });
    }
    if a.rows == 0 || b.rows == 0 {
        return Err(OchreError::EmptyData);
    }
    if a.rows.saturating_mul(b.rows) < CPU_PAR_ELEMS {
        return Ok(compute_pairwise_l2_sq_seq(a, b, is_self));
    }
    backend.pairwise_l2_sq(a, b)
}

/// Fill full kernel matrix `k_out` (`rows x cols`) from precomputed `d2` (and `d_sqrt` for Matern).
fn fill_kernel_matrix_from_d2_into(
    d2: Option<&Matrix>,
    d_sqrt: Option<&Matrix>,
    rows: usize,
    cols: usize,
    params: &KernelParams,
    is_self: bool,
    k_out: &mut Matrix,
) {
    if params.kind == KernelKind::White {
        k_out.data.fill(0.0);
        if is_self {
            for i in 0..rows.min(cols) {
                k_out.set(i, i, params.noise_level);
            }
        }
        return;
    }

    let c = params.constant_value;
    let ls = params.length_scale;
    let ls_sq = ls * ls;
    let use_par = rows.saturating_mul(cols) >= CPU_PAR_ELEMS;

    match params.kind {
        KernelKind::Rbf | KernelKind::RbfWhite => {
            if let Some(d2_mat) = d2 {
                let inv_two_ls_sq = -0.5 / ls_sq;
                if use_par {
                    k_out
                        .data
                        .par_iter_mut()
                        .zip(d2_mat.data.par_iter())
                        .for_each(|(dst, &d)| {
                            *dst = c * (d.max(0.0) * inv_two_ls_sq).exp();
                        });
                } else {
                    for (dst, &d) in k_out.data.iter_mut().zip(d2_mat.data.iter()) {
                        *dst = c * (d.max(0.0) * inv_two_ls_sq).exp();
                    }
                }
            }
        }
        KernelKind::Matern => {
            let nu = params.nu;
            if let Some(ds_mat) = d_sqrt {
                if (nu - 0.5).abs() <= 1e-12 {
                    let inv_ls = 1.0 / ls;
                    if use_par {
                        k_out
                            .data
                            .par_iter_mut()
                            .zip(ds_mat.data.par_iter())
                            .for_each(|(dst, &r)| *dst = c * (-(r * inv_ls)).exp());
                    } else {
                        for (dst, &r) in k_out.data.iter_mut().zip(ds_mat.data.iter()) {
                            *dst = c * (-(r * inv_ls)).exp();
                        }
                    }
                } else if (nu - 1.5).abs() <= 1e-12 {
                    let s3 = 3.0_f64.sqrt() / ls;
                    if use_par {
                        k_out
                            .data
                            .par_iter_mut()
                            .zip(ds_mat.data.par_iter())
                            .for_each(|(dst, &d)| {
                                let r = d * s3;
                                *dst = c * (1.0 + r) * (-r).exp();
                            });
                    } else {
                        for (dst, &d) in k_out.data.iter_mut().zip(ds_mat.data.iter()) {
                            let r = d * s3;
                            *dst = c * (1.0 + r) * (-r).exp();
                        }
                    }
                } else {
                    let s5 = 5.0_f64.sqrt() / ls;
                    if use_par {
                        k_out
                            .data
                            .par_iter_mut()
                            .zip(ds_mat.data.par_iter())
                            .for_each(|(dst, &d)| {
                                let r = d * s5;
                                *dst = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                            });
                    } else {
                        for (dst, &d) in k_out.data.iter_mut().zip(ds_mat.data.iter()) {
                            let r = d * s5;
                            *dst = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                        }
                    }
                }
            } else if let Some(d2_mat) = d2 {
                if (nu - 0.5).abs() <= 1e-12 {
                    let inv_ls = 1.0 / ls;
                    for (dst, &d) in k_out.data.iter_mut().zip(d2_mat.data.iter()) {
                        let r = d.max(0.0).sqrt() * inv_ls;
                        *dst = c * (-r).exp();
                    }
                } else if (nu - 1.5).abs() <= 1e-12 {
                    let s3 = 3.0_f64.sqrt() / ls;
                    for (dst, &d) in k_out.data.iter_mut().zip(d2_mat.data.iter()) {
                        let r = d.max(0.0).sqrt() * s3;
                        *dst = c * (1.0 + r) * (-r).exp();
                    }
                } else {
                    let s5 = 5.0_f64.sqrt() / ls;
                    for (dst, &d) in k_out.data.iter_mut().zip(d2_mat.data.iter()) {
                        let r = d.max(0.0).sqrt() * s5;
                        *dst = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                    }
                }
            }
        }
        KernelKind::RationalQuadratic => {
            if let Some(d2_mat) = d2 {
                let alpha = params.rq_alpha;
                let denom = 2.0 * alpha * ls_sq;
                if use_par {
                    k_out
                        .data
                        .par_iter_mut()
                        .zip(d2_mat.data.par_iter())
                        .for_each(|(dst, &d)| {
                            let base = 1.0 + d.max(0.0) / denom;
                            *dst = c * base.powf(-alpha);
                        });
                } else {
                    for (dst, &d) in k_out.data.iter_mut().zip(d2_mat.data.iter()) {
                        let base = 1.0 + d.max(0.0) / denom;
                        *dst = c * base.powf(-alpha);
                    }
                }
            }
        }
        KernelKind::White => {}
    }

    if is_self {
        let diag = kernel_diag_value(params);
        for i in 0..rows.min(cols) {
            k_out.set(i, i, diag);
        }
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
    let mut k = compute_pairwise_l2_sq_fast(backend, a, b, is_self)?;
    let ls = params.length_scale;
    let ls_sq = ls * ls;
    let c = params.constant_value;
    let use_par = k.data.len() >= CPU_PAR_ELEMS;

    match params.kind {
        KernelKind::Rbf | KernelKind::RbfWhite => {
            let inv_two_ls_sq = -0.5 / ls_sq;
            if use_par {
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
                if use_par {
                    k.data.par_iter_mut().for_each(|d| {
                        let r = d.max(0.0).sqrt() / ls;
                        *d = c * (-r).exp();
                    });
                } else {
                    for d in &mut k.data {
                        let r = d.max(0.0).sqrt() / ls;
                        *d = c * (-r).exp();
                    }
                }
            } else if (nu - 1.5).abs() <= 1e-12 {
                let s3 = 3.0_f64.sqrt() / ls;
                if use_par {
                    k.data.par_iter_mut().for_each(|d| {
                        let r = d.max(0.0).sqrt() * s3;
                        *d = c * (1.0 + r) * (-r).exp();
                    });
                } else {
                    for d in &mut k.data {
                        let r = d.max(0.0).sqrt() * s3;
                        *d = c * (1.0 + r) * (-r).exp();
                    }
                }
            } else {
                let s5 = 5.0_f64.sqrt() / ls;
                if use_par {
                    k.data.par_iter_mut().for_each(|d| {
                        let r = d.max(0.0).sqrt() * s5;
                        *d = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                    });
                } else {
                    for d in &mut k.data {
                        let r = d.max(0.0).sqrt() * s5;
                        *d = c * (1.0 + r + (r * r) / 3.0) * (-r).exp();
                    }
                }
            }
        }
        KernelKind::RationalQuadratic => {
            let alpha = params.rq_alpha;
            let denom = 2.0 * alpha * ls_sq;
            if use_par {
                k.data.par_iter_mut().for_each(|d| {
                    let base = 1.0 + d.max(0.0) / denom;
                    *d = c * base.powf(-alpha);
                });
            } else {
                for d in &mut k.data {
                    let base = 1.0 + d.max(0.0) / denom;
                    *d = c * base.powf(-alpha);
                }
            }
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

#[inline(always)]
fn dot_unrolled(a: &[f64], b: &[f64]) -> f64 {
    dot_slice_unrolled(a, b)
}

/// Sequential 4-row lower-triangular Cholesky factorization into preallocated `l` (`n x n`).
/// Unrolls 4 rows at a time sharing `L[j, 0..j]` loads in L1 cache and avoids Rayon spawns
/// for `n < CPU_PAR_THRESHOLD` (512).
fn cholesky_in_place_lower_4row(
    a: &Matrix,
    diag_tol: f64,
    l: &mut Matrix,
    inv_diag_buf: &mut Vec<f64>,
) -> bool {
    let n = a.rows;
    if n == 0 {
        return true;
    }
    if inv_diag_buf.len() < n {
        inv_diag_buf.resize(n, 0.0);
    }
    let inv_diag = &mut inv_diag_buf[..n];

    for i in 0..n {
        let row_off = i * n;
        let src_row = &a.data[row_off..row_off + n];
        let dst_row = &mut l.data[row_off..row_off + n];
        dst_row[..=i].copy_from_slice(&src_row[..=i]);
        dst_row[i + 1..].fill(0.0);
    }

    let mut i = 0;
    while i + 4 <= n {
        let (top, bot) = l.data.split_at_mut(i * n);
        let (r0, rest) = bot.split_at_mut(n);
        let (r1, rest) = rest.split_at_mut(n);
        let (r2, rest) = rest.split_at_mut(n);
        let r3 = &mut rest[..n];

        for j in 0..i {
            let lj = &top[j * n..j * n + j];
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut s3 = 0.0;
            for m in 0..j {
                let lm = lj[m];
                s0 += r0[m] * lm;
                s1 += r1[m] * lm;
                s2 += r2[m] * lm;
                s3 += r3[m] * lm;
            }
            let inv_j = inv_diag[j];
            r0[j] = (r0[j] - s0) * inv_j;
            r1[j] = (r1[j] - s1) * inv_j;
            r2[j] = (r2[j] - s2) * inv_j;
            r3[j] = (r3[j] - s3) * inv_j;
        }

        let d0 = r0[i] - dot_slice_unrolled(&r0[..i], &r0[..i]);
        if !d0.is_finite() || d0 <= diag_tol {
            return false;
        }
        let l00 = d0.sqrt();
        let inv0 = 1.0 / l00;
        r0[i] = l00;
        inv_diag[i] = inv0;

        let l10 = (r1[i] - dot_slice_unrolled(&r1[..i], &r0[..i])) * inv0;
        let l20 = (r2[i] - dot_slice_unrolled(&r2[..i], &r0[..i])) * inv0;
        let l30 = (r3[i] - dot_slice_unrolled(&r3[..i], &r0[..i])) * inv0;
        r1[i] = l10;
        r2[i] = l20;
        r3[i] = l30;

        let d1 = r1[i + 1] - dot_slice_unrolled(&r1[..i + 1], &r1[..i + 1]);
        if !d1.is_finite() || d1 <= diag_tol {
            return false;
        }
        let l11 = d1.sqrt();
        let inv1 = 1.0 / l11;
        r1[i + 1] = l11;
        inv_diag[i + 1] = inv1;

        let l21 = (r2[i + 1] - dot_slice_unrolled(&r2[..i + 1], &r1[..i + 1])) * inv1;
        let l31 = (r3[i + 1] - dot_slice_unrolled(&r3[..i + 1], &r1[..i + 1])) * inv1;
        r2[i + 1] = l21;
        r3[i + 1] = l31;

        let d2 = r2[i + 2] - dot_slice_unrolled(&r2[..i + 2], &r2[..i + 2]);
        if !d2.is_finite() || d2 <= diag_tol {
            return false;
        }
        let l22 = d2.sqrt();
        let inv2 = 1.0 / l22;
        r2[i + 2] = l22;
        inv_diag[i + 2] = inv2;

        let l32 = (r3[i + 2] - dot_slice_unrolled(&r3[..i + 2], &r2[..i + 2])) * inv2;
        r3[i + 2] = l32;

        let d3 = r3[i + 3] - dot_slice_unrolled(&r3[..i + 3], &r3[..i + 3]);
        if !d3.is_finite() || d3 <= diag_tol {
            return false;
        }
        let l33 = d3.sqrt();
        r3[i + 3] = l33;
        inv_diag[i + 3] = 1.0 / l33;

        i += 4;
    }

    while i < n {
        let (top, bot) = l.data.split_at_mut(i * n);
        let ri = &mut bot[..n];
        for j in 0..i {
            let lj = &top[j * n..j * n + j];
            ri[j] = (ri[j] - dot_slice_unrolled(&ri[..j], lj)) * inv_diag[j];
        }
        let diag = ri[i] - dot_slice_unrolled(&ri[..i], &ri[..i]);
        if !diag.is_finite() || diag <= diag_tol {
            return false;
        }
        let lii = diag.sqrt();
        ri[i] = lii;
        inv_diag[i] = 1.0 / lii;
        i += 1;
    }

    true
}

/// Lower-triangular Cholesky factorization into preallocated `l` and `panel_buf`.
#[inline]
fn cholesky_lower_into(a: &Matrix, l: &mut Matrix, panel_buf: &mut Vec<f64>) -> OchreResult<()> {
    let ok = if a.rows < CPU_PAR_THRESHOLD {
        cholesky_in_place_lower_4row(a, 1e-15, l, panel_buf)
    } else {
        cholesky_lower_with_diag_shift_into(a, 0.0, 1e-15, l, panel_buf)
    };
    if !ok {
        return Err(OchreError::SingularMatrix(
            "Kernel matrix is not positive definite; consider increasing alpha or noise_level"
                .to_string(),
        ));
    }
    Ok(())
}

/// Forward substitution into preallocated buffer: solve L * z = b where L is lower triangular (n x n).
#[inline]
fn solve_lower_triangular_into(l: &Matrix, b: &[f64], z: &mut [f64]) {
    solve_lower_triangular_4row_into(l, b, z);
}

/// Solve (L * L^T) * x = b using preallocated buffers `z` and `x`.
#[inline]
fn solve_cholesky_from_l_into(l: &Matrix, b: &[f64], z: &mut [f64], x: &mut [f64]) {
    solve_lower_triangular_4row_into(l, b, z);
    solve_upper_triangular_t_4row_into(l, z, x);
}

/// Compute `(L^{-1})^T` into preallocated `m_t` (`n * n`), zeroing `m_t[i, 0..i]`.
/// Each row `i` of `M^T` depends only on `L` and itself, so rows run in parallel when `n >= CPU_PAR_THRESHOLD`.
fn invert_lower_triangular_t_into(l: &Matrix, m_t: &mut [f64]) {
    let n = l.rows;
    let fill_row = |(i, mi_row): (usize, &mut [f64])| {
        mi_row[..i].fill(0.0);
        mi_row[i] = 1.0 / l.data[i * n + i];
        for k in (i + 1)..n {
            let lk_off = k * n;
            let sum = dot_slice_unrolled(&l.data[lk_off + i..lk_off + k], &mi_row[i..k]);
            mi_row[k] = -sum / l.data[lk_off + k];
        }
    };
    if n >= CPU_PAR_THRESHOLD {
        m_t.par_chunks_mut(n).enumerate().for_each(fill_row);
    } else {
        m_t.chunks_exact_mut(n).enumerate().for_each(fill_row);
    }
}

/// Zero-allocation CPU matrix-vector multiply `out = mat * v`.
#[inline]
fn matvec_into(mat: &Matrix, v: &[f64], out: &mut [f64]) {
    let cols = mat.cols;
    if mat.rows >= CPU_PAR_THRESHOLD && mat.rows.saturating_mul(cols) >= CPU_PAR_ELEMS {
        out.par_iter_mut()
            .zip(mat.data.par_chunks(cols))
            .for_each(|(dst, row)| {
                *dst = dot_slice_unrolled(row, v);
            });
    } else {
        for (dst, row) in out.iter_mut().zip(mat.data.chunks_exact(cols)) {
            *dst = dot_slice_unrolled(row, v);
        }
    }
}

/// Precompute Euclidean distance matrix `D = sqrt(max(D2, 0))` once for Matern optimization.
fn compute_d_sqrt(d2: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(d2.rows, d2.cols);
    if d2.data.len() >= CPU_PAR_ELEMS {
        out.data
            .par_iter_mut()
            .zip(d2.data.par_iter())
            .for_each(|(dst, &d)| {
                *dst = d.max(0.0).sqrt();
            });
    } else {
        for (dst, &d) in out.data.iter_mut().zip(d2.data.iter()) {
            *dst = d.max(0.0).sqrt();
        }
    }
    out
}

/// Fill ONLY the lower triangle (`j <= i`) of `k_out` from `d2` (or `d_sqrt` for Matern)
/// with `diag = kernel_diag_value(params) + alpha_reg`.
fn fill_kernel_lower_into(
    d2: Option<&Matrix>,
    d_sqrt: Option<&Matrix>,
    n: usize,
    params: &KernelParams,
    alpha_reg: f64,
    k_out: &mut Matrix,
) {
    let diag_val = kernel_diag_value(params) + alpha_reg;
    if params.kind == KernelKind::White {
        for i in 0..n {
            let row = &mut k_out.data[i * n..i * n + n];
            row[..i].fill(0.0);
            row[i] = diag_val;
        }
        return;
    }

    let c = params.constant_value;
    let ls = params.length_scale;
    let ls_sq = ls * ls;

    match params.kind {
        KernelKind::Rbf | KernelKind::RbfWhite => {
            if let Some(d2_mat) = d2 {
                let inv_two_ls_sq = -0.5 / ls_sq;
                let fill_row = |(i, k_row): (usize, &mut [f64])| {
                    let d_row = &d2_mat.data[i * n..i * n + i];
                    for j in 0..i {
                        k_row[j] = c * (d_row[j].max(0.0) * inv_two_ls_sq).exp();
                    }
                    k_row[i] = diag_val;
                };
                if n >= CPU_PAR_THRESHOLD {
                    k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                } else {
                    k_out
                        .data
                        .chunks_exact_mut(n)
                        .enumerate()
                        .for_each(fill_row);
                }
            }
        }
        KernelKind::Matern => {
            let nu = params.nu;
            if let Some(ds_mat) = d_sqrt {
                if (nu - 0.5).abs() <= 1e-12 {
                    let inv_ls = 1.0 / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &ds_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            k_row[j] = c * (-(d_row[j] * inv_ls)).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                } else if (nu - 1.5).abs() <= 1e-12 {
                    let s3 = 3.0_f64.sqrt() / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &ds_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            let r = d_row[j] * s3;
                            k_row[j] = c * (1.0 + r) * (-r).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                } else {
                    let s5 = 5.0_f64.sqrt() / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &ds_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            let r = d_row[j] * s5;
                            k_row[j] = c * (1.0 + r + (r * r) * (1.0 / 3.0)) * (-r).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                }
            } else if let Some(d2_mat) = d2 {
                if (nu - 0.5).abs() <= 1e-12 {
                    let inv_ls = 1.0 / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &d2_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            k_row[j] = c * (-(d_row[j].max(0.0).sqrt() * inv_ls)).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                } else if (nu - 1.5).abs() <= 1e-12 {
                    let s3 = 3.0_f64.sqrt() / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &d2_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            let r = d_row[j].max(0.0).sqrt() * s3;
                            k_row[j] = c * (1.0 + r) * (-r).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                } else {
                    let s5 = 5.0_f64.sqrt() / ls;
                    let fill_row = |(i, k_row): (usize, &mut [f64])| {
                        let d_row = &d2_mat.data[i * n..i * n + i];
                        for j in 0..i {
                            let r = d_row[j].max(0.0).sqrt() * s5;
                            k_row[j] = c * (1.0 + r + (r * r) * (1.0 / 3.0)) * (-r).exp();
                        }
                        k_row[i] = diag_val;
                    };
                    if n >= CPU_PAR_THRESHOLD {
                        k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                    } else {
                        k_out
                            .data
                            .chunks_exact_mut(n)
                            .enumerate()
                            .for_each(fill_row);
                    }
                }
            }
        }
        KernelKind::RationalQuadratic => {
            if let Some(d2_mat) = d2 {
                let alpha = params.rq_alpha;
                let denom = 2.0 * alpha * ls_sq;
                let fill_row = |(i, k_row): (usize, &mut [f64])| {
                    let d_row = &d2_mat.data[i * n..i * n + i];
                    for j in 0..i {
                        k_row[j] = c * (1.0 + d_row[j].max(0.0) / denom).powf(-alpha);
                    }
                    k_row[i] = diag_val;
                };
                if n >= CPU_PAR_THRESHOLD {
                    k_out.data.par_chunks_mut(n).enumerate().for_each(fill_row);
                } else {
                    k_out
                        .data
                        .chunks_exact_mut(n)
                        .enumerate()
                        .for_each(fill_row);
                }
            }
        }
        KernelKind::White => unreachable!(),
    }
}

#[inline]
fn mirror_lower_to_upper(mat: &mut Matrix) {
    let n = mat.rows;
    for i in 0..n {
        let row_off = i * n;
        for j in 0..i {
            mat.data[j * n + i] = mat.data[row_off + j];
        }
    }
}

/// Compute `4 x 4` dot products `(M^T * M)[i_base..i_base+4, j_base..j_base+4]` where `j_base <= i_base`
/// and `m_t[r, k] == 0` for `k < r`, starting summation at `k = i_base`.
#[inline(always)]
fn dot_4x4_m_t(m_t: &[f64], n: usize, i_base: usize, j_base: usize) -> [[f64; 4]; 4] {
    let m0 = &m_t[i_base * n..(i_base + 1) * n];
    let m1 = &m_t[(i_base + 1) * n..(i_base + 2) * n];
    let m2 = &m_t[(i_base + 2) * n..(i_base + 3) * n];
    let m3 = &m_t[(i_base + 3) * n..(i_base + 4) * n];
    let n0 = &m_t[j_base * n..(j_base + 1) * n];
    let n1 = &m_t[(j_base + 1) * n..(j_base + 2) * n];
    let n2 = &m_t[(j_base + 2) * n..(j_base + 3) * n];
    let n3 = &m_t[(j_base + 3) * n..(j_base + 4) * n];

    let mut acc = [[0.0f64; 4]; 4];
    for k in i_base..n {
        let a0 = m0[k];
        let a1 = m1[k];
        let a2 = m2[k];
        let a3 = m3[k];
        let b0 = n0[k];
        let b1 = n1[k];
        let b2 = n2[k];
        let b3 = n3[k];

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
    }
    acc
}

/// Compute `dK_{ij} / d(log length_scale)` multiplier `ratio` such that `dK_{ij}/d(log ls) = K_{ij} * ratio`.
#[inline(always)]
fn dlog_ls_factor(
    kind: KernelKind,
    nu: f64,
    ls: f64,
    inv_ls_sq: f64,
    d2_val: f64,
    d_sqrt_val: f64,
) -> f64 {
    match kind {
        KernelKind::Rbf | KernelKind::RbfWhite => d2_val * inv_ls_sq,
        KernelKind::Matern => {
            if (nu - 0.5).abs() <= 1e-12 {
                d_sqrt_val / ls
            } else if (nu - 1.5).abs() <= 1e-12 {
                let r = 3.0_f64.sqrt() * d_sqrt_val / ls;
                (r * r) / (1.0 + r)
            } else {
                let r = 5.0_f64.sqrt() * d_sqrt_val / ls;
                let r2 = r * r;
                (r2 * (1.0 + r)) / (3.0 + 3.0 * r + r2)
            }
        }
        _ => 0.0,
    }
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
) -> Option<KernelParams>
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
    if opt_neg_lml < 1e29 && -opt_neg_lml > best_lml {
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
            if cand_neg_lml < 1e29 && -cand_neg_lml > best_lml {
                best_lml = -cand_neg_lml;
                best_theta = cand_theta;
            }
        }
    }

    if best_lml.is_finite() {
        Some(apply_log_vec(&base, &best_theta))
    } else {
        None
    }
}

struct GprScratch {
    k: Matrix,
    l: Matrix,
    m_t: Vec<f64>,
    panel_buf: Vec<f64>,
    z: Vec<f64>,
    alpha: Vec<f64>,
}

impl GprScratch {
    fn new(n: usize) -> Self {
        Self {
            k: Matrix::zeros(n, n),
            l: Matrix::zeros(n, n),
            m_t: vec![0.0; n * n],
            panel_buf: vec![0.0; 4 * n],
            z: vec![0.0; n],
            alpha: vec![0.0; n],
        }
    }
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

    fn eval_gpr_fit_from_d2_into(
        d2: Option<&Matrix>,
        d_sqrt: Option<&Matrix>,
        n: usize,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
        scratch: &mut GprScratch,
    ) -> OchreResult<f64> {
        fill_kernel_lower_into(d2, d_sqrt, n, params, alpha_reg, &mut scratch.k);
        cholesky_lower_into(&scratch.k, &mut scratch.l, &mut scratch.panel_buf)?;
        solve_cholesky_from_l_into(&scratch.l, y_norm, &mut scratch.z, &mut scratch.alpha);

        let mut log_det_half = 0.0;
        for i in 0..n {
            log_det_half += scratch.l.data[i * n + i].ln();
        }
        let lml = -0.5 * dot_slice_unrolled(y_norm, &scratch.alpha)
            - log_det_half
            - 0.5 * (n as f64) * (2.0 * PI).ln();

        Ok(lml)
    }

    #[allow(dead_code)]
    fn eval_gpr_fit_from_d2(
        _backend: &Backend<'_>,
        d2: Option<&Matrix>,
        n: usize,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
    ) -> OchreResult<(Matrix, Vec<f64>, f64)> {
        let mut scratch = GprScratch::new(n);
        let lml =
            Self::eval_gpr_fit_from_d2_into(d2, None, n, y_norm, alpha_reg, params, &mut scratch)?;
        Ok((scratch.l, scratch.alpha, lml))
    }

    /// Evaluate GPR LML and analytical gradient wrt log-hyperparameters for `Rbf`, `RbfWhite`, and `Matern`.
    /// Gradient formula: d(LML)/d(theta_m) = 0.5 * tr((alpha * alpha^T - K^{-1}) * dK/d(theta_m)).
    /// Reuses preallocated `GprScratch` buffers (zero `n x n` allocations).
    fn eval_gpr_lml_and_grad_into(
        d2: &Matrix,
        d_sqrt: Option<&Matrix>,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
        scratch: &mut GprScratch,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let n = d2.rows;
        let ls = params.length_scale;
        let inv_ls_sq = 1.0 / (ls * ls);
        let c = params.constant_value;
        let is_rbf_white = params.kind == KernelKind::RbfWhite;
        let kind = params.kind;
        let nu = params.nu;
        let noise_level = params.noise_level;

        fill_kernel_lower_into(Some(d2), d_sqrt, n, params, alpha_reg, &mut scratch.k);
        cholesky_lower_into(&scratch.k, &mut scratch.l, &mut scratch.panel_buf)?;
        solve_cholesky_from_l_into(&scratch.l, y_norm, &mut scratch.z, &mut scratch.alpha);

        let mut log_det_half = 0.0;
        for i in 0..n {
            log_det_half += scratch.l.data[i * n + i].ln();
        }
        let lml = -0.5 * dot_slice_unrolled(y_norm, &scratch.alpha)
            - log_det_half
            - 0.5 * (n as f64) * (2.0 * PI).ln();

        // Compute M^T = (L^{-1})^T into scratch.m_t so K^{-1}_{ij} = dot(m_t[i, i..n], m_t[j, i..n]) for i >= j
        invert_lower_triangular_t_into(&scratch.l, &mut scratch.m_t);

        let k_ref = &scratch.k.data;
        let m_t_ref = &scratch.m_t;
        let alpha_ref = &scratch.alpha;
        let d2_ref = &d2.data;
        let ds_ref = d_sqrt.map(|m| m.data.as_slice());

        let accum_block_row = |i_block: usize| -> (f64, f64, f64) {
            let i_base = i_block * 4;
            let mut gc = 0.0;
            let mut gl = 0.0;
            let mut gn = 0.0;

            if i_base + 4 <= n {
                // Strictly below diagonal 4x4 blocks: j_base < i_base
                let mut j_base = 0;
                while j_base < i_base {
                    let k_inv_blk = dot_4x4_m_t(m_t_ref, n, i_base, j_base);
                    for (dr, k_inv_row) in k_inv_blk.iter().enumerate() {
                        let i = i_base + dr;
                        let row = i * n;
                        let ai = alpha_ref[i];
                        for (dc, &k_inv_ij) in k_inv_row.iter().enumerate() {
                            let j = j_base + dc;
                            let w_ij = ai * alpha_ref[j] - k_inv_ij;
                            let wk = w_ij * k_ref[row + j];
                            gc += wk;
                            let d2_val = d2_ref[row + j].max(0.0);
                            let ds_val = match ds_ref {
                                Some(ds) => ds[row + j],
                                None => d2_val.sqrt(),
                            };
                            gl += wk * dlog_ls_factor(kind, nu, ls, inv_ls_sq, d2_val, ds_val);
                        }
                    }
                    j_base += 4;
                }

                // Diagonal 4x4 block: j_base == i_base
                let k_inv_diag = dot_4x4_m_t(m_t_ref, n, i_base, i_base);
                for (dr, k_inv_row) in k_inv_diag.iter().enumerate() {
                    let i = i_base + dr;
                    let row = i * n;
                    let ai = alpha_ref[i];
                    let w_ii = ai * ai - k_inv_row[dr];
                    gc += 0.5 * w_ii * c;
                    if is_rbf_white {
                        gn += 0.5 * w_ii * noise_level;
                    }
                    for (dc, &k_inv_ij) in k_inv_row.iter().enumerate().take(dr) {
                        let j = i_base + dc;
                        let w_ij = ai * alpha_ref[j] - k_inv_ij;
                        let wk = w_ij * k_ref[row + j];
                        gc += wk;
                        let d2_val = d2_ref[row + j].max(0.0);
                        let ds_val = match ds_ref {
                            Some(ds) => ds[row + j],
                            None => d2_val.sqrt(),
                        };
                        gl += wk * dlog_ls_factor(kind, nu, ls, inv_ls_sq, d2_val, ds_val);
                    }
                }
            } else {
                // Remainder rows at bottom (0..=3 rows)
                for i in i_base..n {
                    let row = i * n;
                    let ai = alpha_ref[i];
                    let mi = &m_t_ref[row + i..row + n];
                    let k_inv_ii = dot_slice_unrolled(mi, mi);
                    let w_ii = ai * ai - k_inv_ii;
                    gc += 0.5 * w_ii * c;
                    if is_rbf_white {
                        gn += 0.5 * w_ii * noise_level;
                    }
                    for j in 0..i {
                        let mj = &m_t_ref[j * n + i..j * n + n];
                        let k_inv_ij = dot_slice_unrolled(mi, mj);
                        let w_ij = ai * alpha_ref[j] - k_inv_ij;
                        let wk = w_ij * k_ref[row + j];
                        gc += wk;
                        let d2_val = d2_ref[row + j].max(0.0);
                        let ds_val = match ds_ref {
                            Some(ds) => ds[row + j],
                            None => d2_val.sqrt(),
                        };
                        gl += wk * dlog_ls_factor(kind, nu, ls, inv_ls_sq, d2_val, ds_val);
                    }
                }
            }
            (gc, gl, gn)
        };

        let n_blocks = n.div_ceil(4);
        let (g_log_c, g_log_ls, g_log_noise) = if n >= CPU_PAR_THRESHOLD {
            (0..n_blocks)
                .into_par_iter()
                .map(accum_block_row)
                .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
        } else {
            (0..n_blocks)
                .map(accum_block_row)
                .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
        };

        let grad = if is_rbf_white {
            vec![g_log_c, g_log_ls, g_log_noise]
        } else {
            vec![g_log_c, g_log_ls]
        };
        Ok((lml, grad))
    }

    #[allow(dead_code)]
    fn eval_gpr_lml_and_grad(
        d2: &Matrix,
        y_norm: &[f64],
        alpha_reg: f64,
        params: &KernelParams,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let mut scratch = GprScratch::new(d2.rows);
        Self::eval_gpr_lml_and_grad_into(d2, None, y_norm, alpha_reg, params, &mut scratch)
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
            Some(compute_pairwise_l2_sq_fast(&backend, x, x, true)?)
        };
        let d_sqrt = if base_params.kind == KernelKind::Matern {
            d2.as_ref().map(compute_d_sqrt)
        } else {
            None
        };

        let mut scratch = GprScratch::new(n);

        let best_params = if self.optimize_hyperparams || self.n_restarts_optimizer > 0 {
            let lbfgs_opt = match (base_params.kind, d2.as_ref()) {
                (KernelKind::Rbf | KernelKind::RbfWhite | KernelKind::Matern, Some(d2_mat)) => {
                    optimize_lml_lbfgs(
                        base_params,
                        self.n_restarts_optimizer,
                        self.random_state,
                        |cand| {
                            Self::eval_gpr_lml_and_grad_into(
                                d2_mat,
                                d_sqrt.as_ref(),
                                &y_norm,
                                self.alpha,
                                cand,
                                &mut scratch,
                            )
                            .ok()
                        },
                    )
                }
                _ => None,
            };
            match lbfgs_opt {
                Some(p) => p,
                None => optimize_lml(
                    base_params,
                    self.n_restarts_optimizer,
                    self.random_state,
                    |cand| {
                        Self::eval_gpr_fit_from_d2_into(
                            d2.as_ref(),
                            d_sqrt.as_ref(),
                            n,
                            &y_norm,
                            self.alpha,
                            cand,
                            &mut scratch,
                        )
                        .ok()
                    },
                ),
            }
        } else {
            base_params
        };

        let lml = Self::eval_gpr_fit_from_d2_into(
            d2.as_ref(),
            d_sqrt.as_ref(),
            n,
            &y_norm,
            self.alpha,
            &best_params,
            &mut scratch,
        )?;

        self.x_train_ = Some(x.clone());
        self.y_train_mean_ = y_mean;
        self.y_train_std_ = y_std;
        self.l_ = Some(scratch.l);
        self.alpha_ = Some(scratch.alpha);
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
        let raw_mean = if matches!(backend, Backend::Cpu) {
            let mut out = vec![0.0; n_test];
            matvec_into(&k_trans, alpha_vec, &mut out);
            out
        } else {
            backend.matvec(&k_trans, alpha_vec)?
        };
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
            let y_std: Vec<f64> = if n_test < CPU_PAR_THRESHOLD && elems < CPU_PAR_ELEMS {
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

        // Solve L * V = K_trans^T => V has shape (n_train, n_test) using 4-column blocked forward solve
        let mut y_cov = compute_kernel_matrix(&backend, x, x, &params, true)?;
        let mut v_mat = Matrix::zeros(n_train, n_test);
        let n_blocks = n_test.div_ceil(4);
        let solve_4col = |blk: usize, v_block: &mut [f64]| {
            let j_start = blk * 4;
            let bs = (n_test - j_start).min(4);
            for r in 0..n_train {
                let dst = &mut v_block[r * 4..r * 4 + 4];
                for (c, d) in dst.iter_mut().enumerate().take(bs) {
                    *d = k_trans.data[(j_start + c) * n_train + r];
                }
                dst[bs..].fill(0.0);
            }
            solve_lower_triangular_4col_4row_into(l, v_block, n_train);
        };
        if n_test >= 256 && n_test.saturating_mul(n_train) >= 65_536 {
            let solved_blocks: Vec<(usize, Vec<f64>)> = (0..n_blocks)
                .into_par_iter()
                .map(|blk| {
                    let mut v_block = vec![0.0; n_train * 4];
                    solve_4col(blk, &mut v_block);
                    (blk, v_block)
                })
                .collect();
            for (blk, v_block) in solved_blocks {
                let j_start = blk * 4;
                let bs = (n_test - j_start).min(4);
                for r in 0..n_train {
                    let dst = &mut v_mat.data[r * n_test + j_start..r * n_test + j_start + bs];
                    dst.copy_from_slice(&v_block[r * 4..r * 4 + bs]);
                }
            }
        } else {
            let mut v_block = vec![0.0; n_train * 4];
            for blk in 0..n_blocks {
                solve_4col(blk, &mut v_block);
                let j_start = blk * 4;
                let bs = (n_test - j_start).min(4);
                for r in 0..n_train {
                    let dst = &mut v_mat.data[r * n_test + j_start..r * n_test + j_start + bs];
                    dst.copy_from_slice(&v_block[r * 4..r * 4 + bs]);
                }
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

struct GpcScratch {
    k: Matrix,
    b_mat: Matrix,
    l: Matrix,
    m_t: Vec<f64>,
    r_mat: Matrix,
    panel_buf: Vec<f64>,
    f: Vec<f64>,
    pi: Vec<f64>,
    w: Vec<f64>,
    w_sqrt: Vec<f64>,
    a: Vec<f64>,
    b_vec: Vec<f64>,
    kb: Vec<f64>,
    rhs: Vec<f64>,
    z: Vec<f64>,
    solved: Vec<f64>,
    f_new: Vec<f64>,
    s2: Vec<f64>,
    k_s2: Vec<f64>,
    r_k_s2: Vec<f64>,
    u: Vec<f64>,
}

impl GpcScratch {
    fn new(n: usize) -> Self {
        Self {
            k: Matrix::zeros(n, n),
            b_mat: Matrix::zeros(n, n),
            l: Matrix::zeros(n, n),
            m_t: vec![0.0; n * n],
            r_mat: Matrix::zeros(n, n),
            panel_buf: vec![0.0; 4 * n],
            f: vec![0.0; n],
            pi: vec![0.5; n],
            w: vec![0.25; n],
            w_sqrt: vec![0.5; n],
            a: vec![0.0; n],
            b_vec: vec![0.0; n],
            kb: vec![0.0; n],
            rhs: vec![0.0; n],
            z: vec![0.0; n],
            solved: vec![0.0; n],
            f_new: vec![0.0; n],
            s2: vec![0.0; n],
            k_s2: vec![0.0; n],
            r_k_s2: vec![0.0; n],
            u: vec![0.0; n],
        }
    }

    #[inline]
    fn fill_b_mat_lower(&mut self, n: usize) {
        let k_data = &self.k.data;
        let w_sqrt = &self.w_sqrt;
        let fill_row = |(i, b_row): (usize, &mut [f64])| {
            let wsi = w_sqrt[i];
            let k_row = &k_data[i * n..i * n + i + 1];
            for j in 0..i {
                b_row[j] = wsi * k_row[j] * w_sqrt[j];
            }
            b_row[i] = 1.0 + wsi * k_row[i] * wsi;
        };
        if n >= CPU_PAR_THRESHOLD {
            self.b_mat
                .data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(fill_row);
        } else {
            self.b_mat
                .data
                .chunks_exact_mut(n)
                .enumerate()
                .for_each(fill_row);
        }
    }
}

#[derive(Clone)]
struct BinaryLaplaceGP {
    y_train_minus_pi: Vec<f64>,
    w_sqrt: Vec<f64>,
    l: Matrix,
    lml: f64,
    params: KernelParams,
}

impl BinaryLaplaceGP {
    fn fit_single_lml_from_d2_into(
        d2: Option<&Matrix>,
        d_sqrt: Option<&Matrix>,
        n: usize,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
        scratch: &mut GpcScratch,
    ) -> OchreResult<f64> {
        fill_kernel_lower_into(d2, d_sqrt, n, params, 0.0, &mut scratch.k);
        mirror_lower_to_upper(&mut scratch.k);

        scratch.f.fill(0.0);
        scratch.a.fill(0.0);

        for _iter in 0..max_iter_predict {
            for i in 0..n {
                let p = sigmoid(scratch.f[i]);
                scratch.pi[i] = p;
                let wi = (p * (1.0 - p)).max(1e-12);
                scratch.w[i] = wi;
                scratch.w_sqrt[i] = wi.sqrt();
            }

            scratch.fill_b_mat_lower(n);
            cholesky_lower_into(&scratch.b_mat, &mut scratch.l, &mut scratch.panel_buf)?;

            for (i, ((&yi, &pi), (&wi, &fi))) in y_bin
                .iter()
                .zip(scratch.pi.iter())
                .zip(scratch.w.iter().zip(scratch.f.iter()))
                .enumerate()
            {
                scratch.b_vec[i] = wi * fi + (yi - pi);
            }

            matvec_into(&scratch.k, &scratch.b_vec, &mut scratch.kb);
            for i in 0..n {
                scratch.rhs[i] = scratch.w_sqrt[i] * scratch.kb[i];
            }
            solve_cholesky_from_l_into(
                &scratch.l,
                &scratch.rhs,
                &mut scratch.z,
                &mut scratch.solved,
            );
            for i in 0..n {
                scratch.a[i] = scratch.b_vec[i] - scratch.w_sqrt[i] * scratch.solved[i];
            }

            matvec_into(&scratch.k, &scratch.a, &mut scratch.f_new);
            let mut max_diff = 0.0_f64;
            for i in 0..n {
                let d = (scratch.f[i] - scratch.f_new[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            std::mem::swap(&mut scratch.f, &mut scratch.f_new);
            if max_diff < 1e-8 {
                break;
            }
        }

        // Recompute pi, w_sqrt, L at converged f
        for i in 0..n {
            let p = sigmoid(scratch.f[i]);
            scratch.pi[i] = p;
            let wi = (p * (1.0 - p)).max(1e-12);
            scratch.w[i] = wi;
            scratch.w_sqrt[i] = wi.sqrt();
        }
        scratch.fill_b_mat_lower(n);
        cholesky_lower_into(&scratch.b_mat, &mut scratch.l, &mut scratch.panel_buf)?;

        let mut log_lik = 0.0;
        for (&yi, &fi) in y_bin.iter().zip(scratch.f.iter()) {
            let log_1_plus_exp = fi.max(0.0) + (-fi.abs()).exp().ln_1p();
            log_lik += yi * fi - log_1_plus_exp;
        }
        let mut log_det_b_half = 0.0;
        for i in 0..n {
            log_det_b_half += scratch.l.data[i * n + i].ln();
        }
        let lml = -0.5 * dot_slice_unrolled(&scratch.a, &scratch.f) + log_lik - log_det_b_half;
        Ok(lml)
    }

    fn fit_single_from_d2_into(
        d2: Option<&Matrix>,
        d_sqrt: Option<&Matrix>,
        n: usize,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
        scratch: &mut GpcScratch,
    ) -> OchreResult<Self> {
        let lml = Self::fit_single_lml_from_d2_into(
            d2,
            d_sqrt,
            n,
            y_bin,
            params,
            max_iter_predict,
            scratch,
        )?;
        let y_train_minus_pi: Vec<f64> = y_bin
            .iter()
            .zip(scratch.pi.iter())
            .map(|(&yi, &pii)| yi - pii)
            .collect();

        Ok(Self {
            y_train_minus_pi,
            w_sqrt: scratch.w_sqrt.clone(),
            l: scratch.l.clone(),
            lml,
            params: *params,
        })
    }

    #[allow(dead_code)]
    fn fit_single_from_d2(
        _backend: &Backend<'_>,
        d2: Option<&Matrix>,
        n: usize,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
    ) -> OchreResult<Self> {
        let mut scratch = GpcScratch::new(n);
        Self::fit_single_from_d2_into(d2, None, n, y_bin, params, max_iter_predict, &mut scratch)
    }

    /// Evaluate Laplace LML and analytical gradient wrt log-hyperparameters (Rasmussen & Williams Alg 5.1)
    /// for `Rbf`, `RbfWhite`, and `Matern` kernels, reusing preallocated `GpcScratch` buffers.
    fn fit_single_lml_and_grad_into(
        d2: &Matrix,
        d_sqrt: Option<&Matrix>,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
        scratch: &mut GpcScratch,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let n = d2.rows;
        let ls = params.length_scale;
        let inv_ls_sq = 1.0 / (ls * ls);
        let c = params.constant_value;
        let is_rbf_white = params.kind == KernelKind::RbfWhite;
        let kind = params.kind;
        let nu = params.nu;
        let noise_level = params.noise_level;

        let lml = Self::fit_single_lml_from_d2_into(
            Some(d2),
            d_sqrt,
            n,
            y_bin,
            params,
            max_iter_predict,
            scratch,
        )?;

        // R = W^{1/2} * B^{-1} * W^{1/2} via M^T = (L^{-1})^T using 4x4 register blocking
        invert_lower_triangular_t_into(&scratch.l, &mut scratch.m_t);
        {
            let m_t_ref = &scratch.m_t;
            let w_sqrt_ref = &scratch.w_sqrt;
            let fill_r_chunk = |(chunk_idx, rows_slice): (usize, &mut [f64])| {
                let i_base = chunk_idx * 4;
                let n_rows_chunk = rows_slice.len() / n;
                if n_rows_chunk == 4 {
                    let mut j_base = 0;
                    while j_base < i_base {
                        let block = dot_4x4_m_t(m_t_ref, n, i_base, j_base);
                        for dr in 0..4 {
                            let wsi = w_sqrt_ref[i_base + dr];
                            let dst = &mut rows_slice[dr * n + j_base..dr * n + j_base + 4];
                            for dc in 0..4 {
                                dst[dc] = wsi * w_sqrt_ref[j_base + dc] * block[dr][dc];
                            }
                        }
                        j_base += 4;
                    }
                    let diag_blk = dot_4x4_m_t(m_t_ref, n, i_base, i_base);
                    for dr in 0..4 {
                        let wsi = w_sqrt_ref[i_base + dr];
                        let dst = &mut rows_slice[dr * n + i_base..dr * n + i_base + dr + 1];
                        for dc in 0..=dr {
                            dst[dc] = wsi * w_sqrt_ref[i_base + dc] * diag_blk[dr][dc];
                        }
                    }
                } else {
                    for r_local in 0..n_rows_chunk {
                        let i = i_base + r_local;
                        let wsi = w_sqrt_ref[i];
                        let mi = &m_t_ref[i * n + i..i * n + n];
                        let dst = &mut rows_slice[r_local * n..r_local * n + i + 1];
                        for j in 0..=i {
                            let mj = &m_t_ref[j * n + i..j * n + n];
                            dst[j] = wsi * w_sqrt_ref[j] * dot_slice_unrolled(mi, mj);
                        }
                    }
                }
            };
            if n >= CPU_PAR_THRESHOLD {
                scratch
                    .r_mat
                    .data
                    .par_chunks_mut(4 * n)
                    .enumerate()
                    .for_each(fill_r_chunk);
            } else {
                scratch
                    .r_mat
                    .data
                    .chunks_mut(4 * n)
                    .enumerate()
                    .for_each(fill_r_chunk);
            }
            mirror_lower_to_upper(&mut scratch.r_mat);
        }

        // s2[i] = -0.5 * (K_{ii} - ||L^{-1} W^{1/2} K_{:,i}||^2) * d^3 log p(y_i | f_i) / df_i^3
        // Solve L * C = W^{1/2} * K in 4-column blocked passes.
        {
            let l_ref = &scratch.l;
            let k_ref = &scratch.k.data;
            let w_sqrt_ref = &scratch.w_sqrt;
            let pi_ref = &scratch.pi;
            let compute_s2_chunk = |v_block: &mut [f64], (blk, s2_chunk): (usize, &mut [f64])| {
                let i_start = blk * 4;
                let bs = s2_chunk.len();
                if bs == 4 {
                    let k0 = &k_ref[i_start * n..(i_start + 1) * n];
                    let k1 = &k_ref[(i_start + 1) * n..(i_start + 2) * n];
                    let k2 = &k_ref[(i_start + 2) * n..(i_start + 3) * n];
                    let k3 = &k_ref[(i_start + 3) * n..(i_start + 4) * n];
                    for j in 0..n {
                        let ws = w_sqrt_ref[j];
                        let dst = &mut v_block[j * 4..j * 4 + 4];
                        dst[0] = ws * k0[j];
                        dst[1] = ws * k1[j];
                        dst[2] = ws * k2[j];
                        dst[3] = ws * k3[j];
                    }
                } else {
                    for j in 0..n {
                        let ws = w_sqrt_ref[j];
                        let dst = &mut v_block[j * 4..j * 4 + 4];
                        for (c, d) in dst.iter_mut().enumerate().take(bs) {
                            *d = ws * k_ref[(i_start + c) * n + j];
                        }
                        dst[bs..].fill(0.0);
                    }
                }
                let norm_sq = solve_lower_triangular_4col_4row_into(l_ref, v_block, n);
                for (c, s2_out) in s2_chunk.iter_mut().enumerate() {
                    let i = i_start + c;
                    let post_var_i = k_ref[i * n + i] - norm_sq[c];
                    let p = pi_ref[i];
                    let dw_df = p * (1.0 - p) * (1.0 - 2.0 * p);
                    *s2_out = -0.5 * post_var_i * dw_df;
                }
            };
            if n >= CPU_PAR_THRESHOLD {
                scratch.s2.par_chunks_mut(4).enumerate().for_each_init(
                    || vec![0.0; n * 4],
                    |v_block, (blk, chunk)| compute_s2_chunk(v_block, (blk, chunk)),
                );
            } else {
                let mut v_block = vec![0.0; n * 4];
                for (blk, chunk) in scratch.s2.chunks_mut(4).enumerate() {
                    compute_s2_chunk(&mut v_block, (blk, chunk));
                }
            }
        }

        // u = (I - R K) s2 = s2 - R (K s2)
        matvec_into(&scratch.k, &scratch.s2, &mut scratch.k_s2);
        matvec_into(&scratch.r_mat, &scratch.k_s2, &mut scratch.r_k_s2);
        for i in 0..n {
            scratch.u[i] = scratch.s2[i] - scratch.r_k_s2[i];
        }

        let a_ref = &scratch.a;
        let u_ref = &scratch.u;
        let r_ref = &scratch.r_mat.data;
        let k_ref = &scratch.k.data;
        let d2_ref = &d2.data;
        let ds_ref = d_sqrt.map(|m| m.data.as_slice());

        let accum_row = |i: usize| -> (f64, f64, f64) {
            let row = i * n;
            let ai = a_ref[i];
            let ui = u_ref[i];
            let diag_coeff = 0.5 * (ai * ai - r_ref[row + i]) + ui * ai;
            let mut gc = diag_coeff * c;
            let mut gl = 0.0;
            let gn = if is_rbf_white {
                diag_coeff * noise_level
            } else {
                0.0
            };
            for j in 0..i {
                let aj = a_ref[j];
                let coeff = (ai * aj - r_ref[row + j]) + ui * aj + u_ref[j] * ai;
                let ck = coeff * k_ref[row + j];
                gc += ck;
                let d2_val = d2_ref[row + j].max(0.0);
                let ds_val = match ds_ref {
                    Some(ds) => ds[row + j],
                    None => d2_val.sqrt(),
                };
                gl += ck * dlog_ls_factor(kind, nu, ls, inv_ls_sq, d2_val, ds_val);
            }
            (gc, gl, gn)
        };

        let (g_log_c, g_log_ls, g_log_noise) = if n >= CPU_PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(accum_row)
                .reduce(|| (0.0, 0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2))
        } else {
            (0..n)
                .map(accum_row)
                .fold((0.0, 0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2))
        };

        let grad = if is_rbf_white {
            vec![g_log_c, g_log_ls, g_log_noise]
        } else {
            vec![g_log_c, g_log_ls]
        };
        Ok((lml, grad))
    }

    #[allow(dead_code)]
    fn fit_single_lml_and_grad(
        _backend: &Backend<'_>,
        d2: &Matrix,
        y_bin: &[f64],
        params: &KernelParams,
        max_iter_predict: usize,
    ) -> OchreResult<(f64, Vec<f64>)> {
        let mut scratch = GpcScratch::new(d2.rows);
        Self::fit_single_lml_and_grad_into(d2, None, y_bin, params, max_iter_predict, &mut scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_pos_proba_range(
        &self,
        k_star: &Matrix,
        f_star: &[f64],
        n_train: usize,
        k_diag: f64,
        start: usize,
        v_block: &mut [f64],
        out: &mut [f64],
    ) {
        let count = out.len();
        let mut offset = 0usize;
        while offset < count {
            let bs = (count - offset).min(4);
            let global_start = start + offset;
            if bs == 4 {
                let k0 = &k_star.data[global_start * n_train..(global_start + 1) * n_train];
                let k1 = &k_star.data[(global_start + 1) * n_train..(global_start + 2) * n_train];
                let k2 = &k_star.data[(global_start + 2) * n_train..(global_start + 3) * n_train];
                let k3 = &k_star.data[(global_start + 3) * n_train..(global_start + 4) * n_train];
                for j in 0..n_train {
                    let ws = self.w_sqrt[j];
                    let dst = &mut v_block[j * 4..(j + 1) * 4];
                    dst[0] = ws * k0[j];
                    dst[1] = ws * k1[j];
                    dst[2] = ws * k2[j];
                    dst[3] = ws * k3[j];
                }
            } else {
                for j in 0..n_train {
                    let ws = self.w_sqrt[j];
                    let dst = &mut v_block[j * 4..(j + 1) * 4];
                    for (c, d) in dst.iter_mut().enumerate().take(bs) {
                        *d = ws * k_star.data[(global_start + c) * n_train + j];
                    }
                    for d in dst.iter_mut().skip(bs) {
                        *d = 0.0;
                    }
                }
            }

            let norm_sq = solve_lower_triangular_4col_4row_into(&self.l, v_block, n_train);
            for c in 0..bs {
                let var_f = (k_diag - norm_sq[c]).max(0.0);
                // Probit approximation matching sklearn GaussianProcessClassifier
                let kappa = 1.0 / (1.0 + (PI / 8.0) * var_f).sqrt();
                out[offset + c] = sigmoid(kappa * f_star[global_start + c]);
            }
            offset += bs;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_pos_proba_from_d2_into(
        &self,
        backend: &Backend<'_>,
        d2: Option<&Matrix>,
        d_sqrt: Option<&Matrix>,
        n_test: usize,
        n_train: usize,
        k_star: &mut Matrix,
        f_star: &mut [f64],
        v_block: &mut [f64],
    ) -> OchreResult<Vec<f64>> {
        fill_kernel_matrix_from_d2_into(d2, d_sqrt, n_test, n_train, &self.params, false, k_star);
        if matches!(backend, Backend::Cpu) || n_test.saturating_mul(n_train) < CPU_PAR_ELEMS {
            matvec_into(k_star, &self.y_train_minus_pi, f_star);
        } else {
            let f_vec = backend.matvec(k_star, &self.y_train_minus_pi)?;
            f_star.copy_from_slice(&f_vec);
        }
        let k_diag = kernel_diag_value(&self.params);

        let mut probs = vec![0.0; n_test];
        if n_test <= CPU_PAR_THRESHOLD || n_test.saturating_mul(n_train) < CPU_PAR_ELEMS {
            self.predict_pos_proba_range(k_star, f_star, n_train, k_diag, 0, v_block, &mut probs);
        } else {
            const PAR_CHUNK_TEST: usize = 64;
            probs
                .par_chunks_mut(PAR_CHUNK_TEST)
                .enumerate()
                .for_each_init(
                    || vec![0.0; n_train * 4],
                    |par_v_block, (chunk_idx, out_chunk)| {
                        self.predict_pos_proba_range(
                            k_star,
                            f_star,
                            n_train,
                            k_diag,
                            chunk_idx * PAR_CHUNK_TEST,
                            par_v_block,
                            out_chunk,
                        );
                    },
                );
        }

        Ok(probs)
    }

    #[allow(dead_code)]
    fn predict_pos_proba(
        &self,
        backend: &Backend<'_>,
        x_train: &Matrix,
        x_test: &Matrix,
    ) -> OchreResult<Vec<f64>> {
        let n_test = x_test.rows;
        let n_train = x_train.rows;
        let d2 = if self.params.kind == KernelKind::White {
            None
        } else {
            Some(compute_pairwise_l2_sq_fast(
                backend, x_test, x_train, false,
            )?)
        };
        let d_sqrt = if self.params.kind == KernelKind::Matern {
            d2.as_ref().map(compute_d_sqrt)
        } else {
            None
        };
        let mut k_star = Matrix::zeros(n_test, n_train);
        let mut f_star = vec![0.0; n_test];
        let mut v_block = vec![0.0; n_train * 4];
        self.predict_pos_proba_from_d2_into(
            backend,
            d2.as_ref(),
            d_sqrt.as_ref(),
            n_test,
            n_train,
            &mut k_star,
            &mut f_star,
            &mut v_block,
        )
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
            Some(compute_pairwise_l2_sq_fast(&backend, x, x, true)?)
        };

        let d_sqrt = if base_params.kind == KernelKind::Matern {
            d2.as_ref().map(compute_d_sqrt)
        } else {
            None
        };
        let mut scratch = GpcScratch::new(n);

        let mut estimators = Vec::new();
        let mut total_lml = 0.0;

        let mut fit_binary = |y_bin: &[f64]| -> OchreResult<BinaryLaplaceGP> {
            let best_params = if do_opt {
                let lbfgs_opt = match (base_params.kind, d2.as_ref()) {
                    (KernelKind::Rbf | KernelKind::RbfWhite | KernelKind::Matern, Some(d2_mat)) => {
                        optimize_lml_lbfgs(
                            base_params,
                            self.n_restarts_optimizer,
                            self.random_state,
                            |cand| {
                                BinaryLaplaceGP::fit_single_lml_and_grad_into(
                                    d2_mat,
                                    d_sqrt.as_ref(),
                                    y_bin,
                                    cand,
                                    self.max_iter_predict,
                                    &mut scratch,
                                )
                                .ok()
                            },
                        )
                    }
                    _ => None,
                };
                match lbfgs_opt {
                    Some(p) => p,
                    None => optimize_lml(
                        base_params,
                        self.n_restarts_optimizer,
                        self.random_state,
                        |cand| {
                            BinaryLaplaceGP::fit_single_lml_from_d2_into(
                                d2.as_ref(),
                                d_sqrt.as_ref(),
                                n,
                                y_bin,
                                cand,
                                self.max_iter_predict,
                                &mut scratch,
                            )
                            .ok()
                        },
                    ),
                }
            } else {
                base_params
            };
            BinaryLaplaceGP::fit_single_from_d2_into(
                d2.as_ref(),
                d_sqrt.as_ref(),
                n,
                y_bin,
                &best_params,
                self.max_iter_predict,
                &mut scratch,
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
        let n_train = x_train.rows;
        let elems = n_test.saturating_mul(n_train);
        // Keep n_test * n_train < 262_144 on CPU so small/medium batches never hit GPU or Rayon overhead.
        let backend = if elems < CPU_PAR_ELEMS {
            Backend::Cpu
        } else {
            effective_backend(&self.parsed_device, elems)
        };
        let n_classes = classes.len();
        let mut proba = Matrix::zeros(n_test, n_classes);

        // Precompute d2(X_test, X_train) ONCE and share it across all OvR estimators.
        let kind = KernelKind::from_str(&self.kernel)?;
        let d2 = if kind == KernelKind::White {
            None
        } else {
            Some(compute_pairwise_l2_sq_fast(&backend, x, x_train, false)?)
        };
        let d_sqrt = if kind == KernelKind::Matern {
            d2.as_ref().map(compute_d_sqrt)
        } else {
            None
        };

        let mut k_star = Matrix::zeros(n_test, n_train);
        let mut f_star = vec![0.0; n_test];
        let mut v_block = vec![0.0; n_train * 4];

        if n_classes == 2 {
            let p1 = self.estimators_[0].predict_pos_proba_from_d2_into(
                &backend,
                d2.as_ref(),
                d_sqrt.as_ref(),
                n_test,
                n_train,
                &mut k_star,
                &mut f_star,
                &mut v_block,
            )?;
            for (i, &p1i) in p1.iter().enumerate().take(n_test) {
                let prob1 = p1i.clamp(0.0, 1.0);
                proba.set(i, 0, 1.0 - prob1);
                proba.set(i, 1, prob1);
            }
        } else {
            let mut class_probs = Vec::with_capacity(n_classes);
            for est in &self.estimators_ {
                class_probs.push(est.predict_pos_proba_from_d2_into(
                    &backend,
                    d2.as_ref(),
                    d_sqrt.as_ref(),
                    n_test,
                    n_train,
                    &mut k_star,
                    &mut f_star,
                    &mut v_block,
                )?);
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

        // Exercise both sequential (<= 512) and parallel (> 512) blocked predict_pos_proba paths
        let mut x_big = Vec::with_capacity(600 * 2);
        for i in 0..600 {
            let t = -2.5 + 5.0 * (i as f64) / 599.0;
            x_big.push(t);
            x_big.push((t * 1.3).cos());
        }
        let x_big_mat = Matrix::from_vec(600, 2, x_big).expect("big mat");
        let proba = gpc.predict_proba(&x_big_mat).expect("predict_proba big");
        assert_eq!(proba.rows, 600);
        assert_eq!(proba.cols, 2);
        for r in 0..600 {
            let sum = proba.get(r, 0) + proba.get(r, 1);
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gpr_and_gpc_matern_analytical_grad_matches_finite_diff() {
        let (x_reg, y_reg) = sample_regression_data();
        let (x_cls, y_cls) = sample_classification_data();
        let d2_reg = Backend::Cpu.pairwise_l2_sq(&x_reg, &x_reg).expect("d2_reg");
        let ds_reg = compute_d_sqrt(&d2_reg);
        let d2_cls = Backend::Cpu.pairwise_l2_sq(&x_cls, &x_cls).expect("d2_cls");
        let ds_cls = compute_d_sqrt(&d2_cls);
        let eps = 1e-5;

        for nu in [0.5, 1.5, 2.5] {
            let base = KernelParams {
                kind: KernelKind::Matern,
                length_scale: 1.4,
                constant_value: 1.9,
                nu,
                noise_level: 0.1,
                rq_alpha: 1.0,
            };
            let theta = params_to_log_vec(&base);

            // GPR Matern analytical grad check
            let mut gpr_scratch = GprScratch::new(x_reg.rows);
            let (_lml, gpr_grad) = GaussianProcessRegressor::eval_gpr_lml_and_grad_into(
                &d2_reg,
                Some(&ds_reg),
                &y_reg,
                1e-4,
                &base,
                &mut gpr_scratch,
            )
            .expect("gpr matern grad");
            for m in 0..theta.len() {
                let mut tp = theta.clone();
                let mut tm = theta.clone();
                tp[m] += eps;
                tm[m] -= eps;
                let lp = GaussianProcessRegressor::eval_gpr_fit_from_d2_into(
                    Some(&d2_reg),
                    Some(&ds_reg),
                    x_reg.rows,
                    &y_reg,
                    1e-4,
                    &apply_log_vec(&base, &tp),
                    &mut gpr_scratch,
                )
                .expect("plus");
                let lm = GaussianProcessRegressor::eval_gpr_fit_from_d2_into(
                    Some(&d2_reg),
                    Some(&ds_reg),
                    x_reg.rows,
                    &y_reg,
                    1e-4,
                    &apply_log_vec(&base, &tm),
                    &mut gpr_scratch,
                )
                .expect("minus");
                let fd = (lp - lm) / (2.0 * eps);
                let err = (gpr_grad[m] - fd).abs();
                assert!(
                    err / (1.0 + fd.abs()) < 1e-4,
                    "GPR Matern(nu={nu}) grad mismatch dim {m}: analytical={} fd={fd}",
                    gpr_grad[m]
                );
            }

            // GPC Matern analytical grad check
            let mut gpc_scratch = GpcScratch::new(x_cls.rows);
            let (_lml, gpc_grad) = BinaryLaplaceGP::fit_single_lml_and_grad_into(
                &d2_cls,
                Some(&ds_cls),
                &y_cls,
                &base,
                100,
                &mut gpc_scratch,
            )
            .expect("gpc matern grad");
            for m in 0..theta.len() {
                let mut tp = theta.clone();
                let mut tm = theta.clone();
                tp[m] += eps;
                tm[m] -= eps;
                let lp = BinaryLaplaceGP::fit_single_lml_from_d2_into(
                    Some(&d2_cls),
                    Some(&ds_cls),
                    x_cls.rows,
                    &y_cls,
                    &apply_log_vec(&base, &tp),
                    100,
                    &mut gpc_scratch,
                )
                .expect("plus");
                let lm = BinaryLaplaceGP::fit_single_lml_from_d2_into(
                    Some(&d2_cls),
                    Some(&ds_cls),
                    x_cls.rows,
                    &y_cls,
                    &apply_log_vec(&base, &tm),
                    100,
                    &mut gpc_scratch,
                )
                .expect("minus");
                let fd = (lp - lm) / (2.0 * eps);
                let err = (gpc_grad[m] - fd).abs();
                assert!(
                    err / (1.0 + fd.abs()) < 1e-4,
                    "GPC Matern(nu={nu}) grad mismatch dim {m}: analytical={} fd={fd}",
                    gpc_grad[m]
                );
            }
        }
    }

    #[test]
    fn test_gpr_and_gpc_280_lbfgs_matern_and_rbf_parity() {
        let n = 280usize;
        let p = 4usize;
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_reg = Vec::with_capacity(n);
        let mut y_cls = Vec::with_capacity(n);
        for i in 0..n {
            let t = -2.5 + 5.0 * (i as f64) / ((n - 1) as f64);
            let x0 = t;
            let x1 = (t * 1.4).sin();
            let x2 = (t * 0.7).cos();
            let x3 = 0.2 * t * t - 0.5;
            x_data.extend_from_slice(&[x0, x1, x2, x3]);
            y_reg.push(x0.sin() + 0.5 * x1 - 0.3 * x2 + 0.05 * x3);
            y_cls.push(if x0 + 0.6 * x1 > 0.0 { 1.0 } else { 0.0 });
        }
        let x = Matrix::from_vec(n, p, x_data).expect("x");

        for kernel in ["rbf", "matern"] {
            let mut gpr = GaussianProcessRegressor::new(
                kernel.to_string(),
                0.5,
                1.0,
                1.5,
                1e-3,
                1.0,
                1.0,
                true,
                0,
                true,
                Some(42),
                None,
            )
            .expect("new gpr");
            gpr.fit(&x, &y_reg).expect("fit gpr");
            let r2 = gpr.score(&x, &y_reg).expect("score gpr");
            assert!(r2 > 0.99, "GPR 280 {kernel} R2={r2}");
            let (_m, _s, cov) = gpr
                .predict_with_uncertainty(&x, false, true)
                .expect("predict cov");
            assert_eq!(cov.expect("cov").rows, n);

            let mut gpc = GaussianProcessClassifier::new(
                kernel.to_string(),
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
            gpc.fit(&x, &y_cls).expect("fit gpc");
            let acc = gpc.score(&x, &y_cls).expect("score gpc");
            assert!(acc >= 0.95, "GPC 280 {kernel} acc={acc}");
        }
    }

    #[test]
    fn test_gpc_ovr_70x280_shared_d2_and_cholesky_4row_parity() {
        let n_train = 280usize;
        let n_test = 70usize;
        let p = 6usize;
        let mut x_tr = Vec::with_capacity(n_train * p);
        let mut y_tr = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let cls = (i % 3) as f64;
            let angle = 2.0 * PI * cls / 3.0 + ((i / 3) as f64) * 0.01;
            for c in 0..p {
                let v = if c == 0 {
                    2.0 * angle.cos()
                } else if c == 1 {
                    2.0 * angle.sin()
                } else {
                    (((i * 7 + c * 13) % 19) as f64 - 9.0) * 0.05
                };
                x_tr.push(v);
            }
            y_tr.push(cls);
        }
        let x_train = Matrix::from_vec(n_train, p, x_tr).expect("x_train");
        let x_test =
            Matrix::from_vec(n_test, p, x_train.data[..n_test * p].to_vec()).expect("x_test");

        let mut gpc = GaussianProcessClassifier::new(
            "rbf".to_string(),
            1.0,
            1.0,
            1.5,
            1.0,
            1.0,
            50,
            true,
            0,
            Some(42),
            None,
        )
        .expect("new gpc");
        gpc.fit(&x_train, &y_tr).expect("fit ovr gpc");
        let proba = gpc.predict_proba(&x_test).expect("predict_proba");
        assert_eq!(proba.rows, n_test);
        assert_eq!(proba.cols, 3);

        // Compare shared-d2 predict_proba against per-estimator explicit predict_pos_proba
        let mut est_probs = Vec::new();
        for est in &gpc.estimators_ {
            est_probs.push(
                est.predict_pos_proba(&Backend::Cpu, &x_train, &x_test)
                    .expect("per-estimator"),
            );
        }
        for i in 0..n_test {
            let sum = est_probs[0][i] + est_probs[1][i] + est_probs[2][i];
            for c in 0..3 {
                let expected = est_probs[c][i] / sum;
                assert!(
                    (proba.get(i, c) - expected).abs() < 1e-12,
                    "OvR mismatch at ({i}, {c}): {} vs {expected}",
                    proba.get(i, c)
                );
            }
        }
        let acc = gpc.score(&x_train, &y_tr).expect("score");
        assert!(acc >= 0.95, "OvR 280x6 acc={acc}");

        // Verify 4-row sequential Cholesky matches matrix::cholesky_lower_with_diag_shift_into
        let d2 = compute_pairwise_l2_sq_seq(&x_train, &x_train, true);
        let mut k = Matrix::zeros(n_train, n_train);
        let params = KernelParams {
            kind: KernelKind::Rbf,
            length_scale: 1.2,
            constant_value: 1.5,
            nu: 1.5,
            noise_level: 1e-3,
            rq_alpha: 1.0,
        };
        fill_kernel_lower_into(Some(&d2), None, n_train, &params, 1e-3, &mut k);
        let mut l_fast = Matrix::zeros(n_train, n_train);
        let mut l_ref = Matrix::zeros(n_train, n_train);
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        assert!(cholesky_in_place_lower_4row(
            &k,
            1e-15,
            &mut l_fast,
            &mut buf1
        ));
        assert!(cholesky_lower_with_diag_shift_into(
            &k, 0.0, 1e-15, &mut l_ref, &mut buf2
        ));
        for idx in 0..l_fast.data.len() {
            assert!((l_fast.data[idx] - l_ref.data[idx]).abs() < 1e-11);
        }
    }
}
