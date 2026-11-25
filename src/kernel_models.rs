// OchreML Wave 5 Tier P3 Kernel Methods Engine (＾▽＾)
// Implements KernelRidge, KernelPCA, SVC, SVR, and OneClassSVM accelerated by
// Hardware Backend Primitives (E1 matmul, N1 col_reduce_stats, N2 elementwise_op,
// N3 pairwise_l2_sq, N5 small_sym_eig_svd) and LIBSVM-style WSS3 SMO (*≧ω≦*).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_str,
    write_usize, write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::logistic_regression::{extract_sorted_classes, validate_sample_weights};
use crate::matrix::{Matrix, r2_score, r2_score_multi, solve_cholesky_multi};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::primitives::ElementwiseOp;
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_SVC: u8 = 76;
pub const TAG_SVR: u8 = 77;
pub const TAG_ONE_CLASS_SVM: u8 = 78;
pub const TAG_KERNEL_RIDGE: u8 = 79;
pub const TAG_KERNEL_PCA: u8 = 80;

const LARGE_ELEMENTWISE_THRESHOLD: usize = 131_072;
const LARGE_MATMUL_THRESHOLD: usize = 4_096;
const GPU_MIN_ELEMS: usize = 131_072;
const KRR_GPU_MIN_ELEMS: usize = 524_288;
const FUSED_PREDICT_MAX_ELEMS: usize = 65_536;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < GPU_MIN_ELEMS {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn effective_backend_krr<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < KRR_GPU_MIN_ELEMS {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelKind {
    Rbf,
    Linear,
    Poly,
    Sigmoid,
    Precomputed,
}

impl KernelKind {
    fn parse(s: &str) -> OchreResult<Self> {
        match s.trim().to_lowercase().as_str() {
            "rbf" => Ok(Self::Rbf),
            "linear" => Ok(Self::Linear),
            "poly" => Ok(Self::Poly),
            "sigmoid" => Ok(Self::Sigmoid),
            "precomputed" => Ok(Self::Precomputed),
            _ => Err(OchreError::InvalidParameter(format!(
                "Invalid kernel '{s}'. Supported kernels are 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed' (・`ω´・)"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        Self {
            state: seed.unwrap_or(0x9E37_79B9_7F4A_7C15),
        }
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

fn validate_alpha(alpha: f64) -> OchreResult<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(OchreError::InvalidAlpha(alpha));
    }
    Ok(())
}

fn validate_c(c: f64) -> OchreResult<()> {
    if !c.is_finite() || c <= 0.0 {
        return Err(OchreError::InvalidC(c));
    }
    Ok(())
}

fn validate_nu(nu: f64) -> OchreResult<()> {
    if !nu.is_finite() || nu <= 0.0 || nu > 1.0 {
        return Err(OchreError::InvalidParameter(format!(
            "nu ({nu}) must be a finite number in (0.0, 1.0] (・`ω´・)"
        )));
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
    if !tol.is_finite() || tol <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be a strictly positive finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_coef0(coef0: f64) -> OchreResult<()> {
    if !coef0.is_finite() {
        return Err(OchreError::InvalidParameter(format!(
            "coef0 ({coef0}) must be finite (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_max_iter(max_iter: Option<usize>) -> OchreResult<usize> {
    match max_iter {
        Some(0) => Err(OchreError::InvalidParameter(
            "max_iter must be >= 1 (・`ω´・)".to_string(),
        )),
        Some(m) => Ok(m),
        None => Ok(200_000),
    }
}

fn resolve_gamma(
    backend: &Backend<'_>,
    x: &Matrix,
    kernel: KernelKind,
    gamma_mode: &str,
    gamma: Option<f64>,
) -> OchreResult<f64> {
    if kernel == KernelKind::Precomputed {
        return Ok(1.0);
    }
    if let Some(g) = gamma {
        if !g.is_finite() || g <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        return Ok(g);
    }
    let n_features = x.cols.max(1) as f64;
    match gamma_mode.trim().to_lowercase().as_str() {
        "auto" => Ok(1.0 / n_features),
        "scale" => {
            // Compute flattened X.var() using N1 col_reduce_stats
            let stats = backend.col_reduce_stats(x, None)?;
            let p = stats.mean.len();
            if p == 0 {
                return Ok(1.0);
            }
            let global_mean = stats.mean.iter().sum::<f64>() / (p as f64);
            let total_var = stats
                .var
                .iter()
                .zip(stats.mean.iter())
                .map(|(&v, &m)| {
                    let diff = m - global_mean;
                    v + diff * diff
                })
                .sum::<f64>()
                / (p as f64);
            if total_var <= 1e-18 || !total_var.is_finite() {
                Ok(1.0)
            } else {
                Ok(1.0 / (n_features * total_var))
            }
        }
        _ => Err(OchreError::InvalidParameter(format!(
            "Invalid gamma mode '{gamma_mode}'. Supported modes are 'scale' and 'auto' (・`ω´・)"
        ))),
    }
}

fn transpose_matrix(m: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(m.cols, m.rows);
    for r in 0..m.rows {
        let row_offset = r * m.cols;
        for c in 0..m.cols {
            out.data[c * m.rows + r] = m.data[row_offset + c];
        }
    }
    out
}

fn compute_dot_gram(backend: &Backend<'_>, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    if a.cols != b.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: a.cols,
            expected: b.cols,
        });
    }
    let b_t = transpose_matrix(b);
    if a.rows * b.rows >= LARGE_MATMUL_THRESHOLD {
        backend.matmul(a, &b_t)
    } else {
        a.matmul(&b_t)
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
fn dot_1x1(a: &[f64], b: &[f64]) -> f64 {
    let cols = a.len();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut c = 0;
    while c + 4 <= cols {
        s0 += a[c] * b[c];
        s1 += a[c + 1] * b[c + 1];
        s2 += a[c + 2] * b[c + 2];
        s3 += a[c + 3] * b[c + 3];
        c += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while c < cols {
        sum += a[c] * b[c];
        c += 1;
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

/// Fused symmetric RBF kernel on CPU (`n <= 2048`): computes only the lower triangle (`j <= i`)
/// in 4x4 register blocks, fuses `exp(-gamma * d2)` in registers, and mirrors to the upper triangle.
fn fused_rbf_symmetric_cpu(x: &Matrix, gamma: f64) -> Matrix {
    let n = x.rows;
    let cols = x.cols;
    let mut k = Matrix::zeros(n, n);
    let neg_gamma = -gamma;
    let use_direct = cols <= 16;
    let norms = if use_direct {
        Vec::new()
    } else if n >= 384 {
        x.data.par_chunks(cols).map(row_sq_norm_unrolled).collect()
    } else {
        x.data
            .chunks_exact(cols)
            .map(row_sq_norm_unrolled)
            .collect()
    };

    let fill_chunk = |(chunk_idx, rows_slice): (usize, &mut [f64])| {
        let i_base = chunk_idx * 4;
        let n_rows = rows_slice.len() / n;
        if n_rows == 4 {
            let (row0, rest) = rows_slice.split_at_mut(n);
            let (row1, rest) = rest.split_at_mut(n);
            let (row2, row3) = rest.split_at_mut(n);

            let x0 = &x.data[i_base * cols..(i_base + 1) * cols];
            let x1 = &x.data[(i_base + 1) * cols..(i_base + 2) * cols];
            let x2 = &x.data[(i_base + 2) * cols..(i_base + 3) * cols];
            let x3 = &x.data[(i_base + 3) * cols..(i_base + 4) * cols];

            let mut j = 0;
            if use_direct {
                while j + 4 <= i_base {
                    let y0 = &x.data[j * cols..(j + 1) * cols];
                    let y1 = &x.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &x.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &x.data[(j + 3) * cols..(j + 4) * cols];
                    let d = direct_l2_sq_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    for c in 0..4 {
                        row0[j + c] = (neg_gamma * d[0][c]).exp();
                        row1[j + c] = (neg_gamma * d[1][c]).exp();
                        row2[j + c] = (neg_gamma * d[2][c]).exp();
                        row3[j + c] = (neg_gamma * d[3][c]).exp();
                    }
                    j += 4;
                }
            } else {
                let ni0 = norms[i_base];
                let ni1 = norms[i_base + 1];
                let ni2 = norms[i_base + 2];
                let ni3 = norms[i_base + 3];
                while j + 4 <= i_base {
                    let y0 = &x.data[j * cols..(j + 1) * cols];
                    let y1 = &x.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &x.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &x.data[(j + 3) * cols..(j + 4) * cols];
                    let dots = dot_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    let nj0 = norms[j];
                    let nj1 = norms[j + 1];
                    let nj2 = norms[j + 2];
                    let nj3 = norms[j + 3];
                    let nj = [nj0, nj1, nj2, nj3];
                    for c in 0..4 {
                        let n_j = nj[c];
                        row0[j + c] = (neg_gamma * (ni0 + n_j - 2.0 * dots[0][c]).max(0.0)).exp();
                        row1[j + c] = (neg_gamma * (ni1 + n_j - 2.0 * dots[1][c]).max(0.0)).exp();
                        row2[j + c] = (neg_gamma * (ni2 + n_j - 2.0 * dots[2][c]).max(0.0)).exp();
                        row3[j + c] = (neg_gamma * (ni3 + n_j - 2.0 * dots[3][c]).max(0.0)).exp();
                    }
                    j += 4;
                }
            }

            // Diagonal 4x4 block (lower triangle)
            let pair_rbf = |r_a: &[f64], r_b: &[f64], na: f64, nb: f64| -> f64 {
                let d2 = if use_direct {
                    direct_l2_sq_1x1(r_a, r_b)
                } else {
                    (na + nb - 2.0 * dot_1x1(r_a, r_b)).max(0.0)
                };
                (neg_gamma * d2).exp()
            };
            let na0 = if use_direct { 0.0 } else { norms[i_base] };
            let na1 = if use_direct { 0.0 } else { norms[i_base + 1] };
            let na2 = if use_direct { 0.0 } else { norms[i_base + 2] };
            let na3 = if use_direct { 0.0 } else { norms[i_base + 3] };

            row0[i_base] = 1.0;
            row1[i_base] = pair_rbf(x1, x0, na1, na0);
            row1[i_base + 1] = 1.0;
            row2[i_base] = pair_rbf(x2, x0, na2, na0);
            row2[i_base + 1] = pair_rbf(x2, x1, na2, na1);
            row2[i_base + 2] = 1.0;
            row3[i_base] = pair_rbf(x3, x0, na3, na0);
            row3[i_base + 1] = pair_rbf(x3, x1, na3, na1);
            row3[i_base + 2] = pair_rbf(x3, x2, na3, na2);
            row3[i_base + 3] = 1.0;
        } else {
            for r_local in 0..n_rows {
                let i = i_base + r_local;
                let xi = &x.data[i * cols..(i + 1) * cols];
                let dst = &mut rows_slice[r_local * n..(r_local + 1) * n];
                if use_direct {
                    for (j, dst_j) in dst.iter_mut().enumerate().take(i) {
                        let xj = &x.data[j * cols..(j + 1) * cols];
                        *dst_j = (neg_gamma * direct_l2_sq_1x1(xi, xj)).exp();
                    }
                } else {
                    let ni = norms[i];
                    for (j, dst_j) in dst.iter_mut().enumerate().take(i) {
                        let xj = &x.data[j * cols..(j + 1) * cols];
                        let d2 = (ni + norms[j] - 2.0 * dot_1x1(xi, xj)).max(0.0);
                        *dst_j = (neg_gamma * d2).exp();
                    }
                }
                dst[i] = 1.0;
            }
        }
    };

    if n >= 384 {
        k.data
            .par_chunks_mut(4 * n)
            .enumerate()
            .for_each(fill_chunk);
    } else {
        k.data.chunks_mut(4 * n).enumerate().for_each(fill_chunk);
    }

    for i in 0..n {
        let row_i_off = i * n;
        for j in 0..i {
            k.data[j * n + i] = k.data[row_i_off + j];
        }
    }
    k
}

/// Fused non-symmetric RBF kernel on CPU: computes `exp(-gamma * ||a_i - b_j||^2)` in 4x4 register blocks
/// without allocating intermediate `d2` or `b_t` matrices.
fn fused_rbf_cross_cpu(a: &Matrix, b: &Matrix, gamma: f64) -> Matrix {
    let na = a.rows;
    let nb = b.rows;
    let cols = a.cols;
    let mut k = Matrix::zeros(na, nb);
    let neg_gamma = -gamma;
    let use_direct = cols <= 16;
    let (a_norms, b_norms) = if use_direct {
        (Vec::new(), Vec::new())
    } else {
        let an = if na >= 384 {
            a.data.par_chunks(cols).map(row_sq_norm_unrolled).collect()
        } else {
            a.data
                .chunks_exact(cols)
                .map(row_sq_norm_unrolled)
                .collect()
        };
        let bn = if nb >= 384 {
            b.data.par_chunks(cols).map(row_sq_norm_unrolled).collect()
        } else {
            b.data
                .chunks_exact(cols)
                .map(row_sq_norm_unrolled)
                .collect()
        };
        (an, bn)
    };

    let fill_chunk = |(chunk_idx, rows_slice): (usize, &mut [f64])| {
        let i_base = chunk_idx * 4;
        let n_rows = rows_slice.len() / nb;
        if n_rows == 4 {
            let (row0, rest) = rows_slice.split_at_mut(nb);
            let (row1, rest) = rest.split_at_mut(nb);
            let (row2, row3) = rest.split_at_mut(nb);

            let x0 = &a.data[i_base * cols..(i_base + 1) * cols];
            let x1 = &a.data[(i_base + 1) * cols..(i_base + 2) * cols];
            let x2 = &a.data[(i_base + 2) * cols..(i_base + 3) * cols];
            let x3 = &a.data[(i_base + 3) * cols..(i_base + 4) * cols];

            let mut j = 0;
            if use_direct {
                while j + 4 <= nb {
                    let y0 = &b.data[j * cols..(j + 1) * cols];
                    let y1 = &b.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &b.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &b.data[(j + 3) * cols..(j + 4) * cols];
                    let d = direct_l2_sq_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    for c in 0..4 {
                        row0[j + c] = (neg_gamma * d[0][c]).exp();
                        row1[j + c] = (neg_gamma * d[1][c]).exp();
                        row2[j + c] = (neg_gamma * d[2][c]).exp();
                        row3[j + c] = (neg_gamma * d[3][c]).exp();
                    }
                    j += 4;
                }
                while j < nb {
                    let yj = &b.data[j * cols..(j + 1) * cols];
                    row0[j] = (neg_gamma * direct_l2_sq_1x1(x0, yj)).exp();
                    row1[j] = (neg_gamma * direct_l2_sq_1x1(x1, yj)).exp();
                    row2[j] = (neg_gamma * direct_l2_sq_1x1(x2, yj)).exp();
                    row3[j] = (neg_gamma * direct_l2_sq_1x1(x3, yj)).exp();
                    j += 1;
                }
            } else {
                let ni0 = a_norms[i_base];
                let ni1 = a_norms[i_base + 1];
                let ni2 = a_norms[i_base + 2];
                let ni3 = a_norms[i_base + 3];
                while j + 4 <= nb {
                    let y0 = &b.data[j * cols..(j + 1) * cols];
                    let y1 = &b.data[(j + 1) * cols..(j + 2) * cols];
                    let y2 = &b.data[(j + 2) * cols..(j + 3) * cols];
                    let y3 = &b.data[(j + 3) * cols..(j + 4) * cols];
                    let dots = dot_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                    let nj = [b_norms[j], b_norms[j + 1], b_norms[j + 2], b_norms[j + 3]];
                    for c in 0..4 {
                        let n_j = nj[c];
                        row0[j + c] = (neg_gamma * (ni0 + n_j - 2.0 * dots[0][c]).max(0.0)).exp();
                        row1[j + c] = (neg_gamma * (ni1 + n_j - 2.0 * dots[1][c]).max(0.0)).exp();
                        row2[j + c] = (neg_gamma * (ni2 + n_j - 2.0 * dots[2][c]).max(0.0)).exp();
                        row3[j + c] = (neg_gamma * (ni3 + n_j - 2.0 * dots[3][c]).max(0.0)).exp();
                    }
                    j += 4;
                }
                while j < nb {
                    let yj = &b.data[j * cols..(j + 1) * cols];
                    let n_j = b_norms[j];
                    row0[j] = (neg_gamma * (ni0 + n_j - 2.0 * dot_1x1(x0, yj)).max(0.0)).exp();
                    row1[j] = (neg_gamma * (ni1 + n_j - 2.0 * dot_1x1(x1, yj)).max(0.0)).exp();
                    row2[j] = (neg_gamma * (ni2 + n_j - 2.0 * dot_1x1(x2, yj)).max(0.0)).exp();
                    row3[j] = (neg_gamma * (ni3 + n_j - 2.0 * dot_1x1(x3, yj)).max(0.0)).exp();
                    j += 1;
                }
            }
        } else {
            for r_local in 0..n_rows {
                let i = i_base + r_local;
                let xi = &a.data[i * cols..(i + 1) * cols];
                let dst = &mut rows_slice[r_local * nb..(r_local + 1) * nb];
                if use_direct {
                    for (j, dst_j) in dst.iter_mut().enumerate().take(nb) {
                        let yj = &b.data[j * cols..(j + 1) * cols];
                        *dst_j = (neg_gamma * direct_l2_sq_1x1(xi, yj)).exp();
                    }
                } else {
                    let ni = a_norms[i];
                    for (j, dst_j) in dst.iter_mut().enumerate().take(nb) {
                        let yj = &b.data[j * cols..(j + 1) * cols];
                        let d2 = (ni + b_norms[j] - 2.0 * dot_1x1(xi, yj)).max(0.0);
                        *dst_j = (neg_gamma * d2).exp();
                    }
                }
            }
        }
    };

    if na * nb >= 131_072 && na >= 32 {
        k.data
            .par_chunks_mut(4 * nb)
            .enumerate()
            .for_each(fill_chunk);
    } else {
        k.data.chunks_mut(4 * nb).enumerate().for_each(fill_chunk);
    }
    k
}

#[inline(always)]
fn validate_row_finite(row: &[f64]) -> OchreResult<()> {
    let mut ok = true;
    for &v in row {
        ok &= v.is_finite();
    }
    if !ok {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

#[inline(always)]
fn validate_and_sq_norm(row: &[f64]) -> OchreResult<f64> {
    let mut ok = true;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut c = 0;
    while c + 4 <= row.len() {
        let v0 = row[c];
        let v1 = row[c + 1];
        let v2 = row[c + 2];
        let v3 = row[c + 3];
        ok &= v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite();
        s0 += v0 * v0;
        s1 += v1 * v1;
        s2 += v2 * v2;
        s3 += v3 * v3;
        c += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while c < row.len() {
        let v = row[c];
        ok &= v.is_finite();
        sum += v * v;
        c += 1;
    }
    if !ok {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(sum)
}

/// Fused zero-allocation CPU RBF prediction for 1D dual coefficients (`offset + sum_j K(x_i, sv_j) * coef_j`).
/// Fuses finite validation into the test-row loops and avoids allocating an `(n_test x n_sv)` matrix.
fn fused_rbf_1d_cpu(
    x: &Matrix,
    sv: &Matrix,
    coef: &[f64],
    gamma: f64,
    offset: f64,
) -> OchreResult<Vec<f64>> {
    let n_test = x.rows;
    let n_sv = sv.rows;
    let cols = x.cols;
    let neg_gamma = -gamma;
    let use_direct = cols <= 16;
    let sv_norms = if use_direct {
        Vec::new()
    } else {
        sv.data
            .chunks_exact(cols)
            .map(row_sq_norm_unrolled)
            .collect()
    };

    let mut out = vec![offset; n_test];
    let mut i = 0;
    while i + 4 <= n_test {
        let x0 = &x.data[i * cols..(i + 1) * cols];
        let x1 = &x.data[(i + 1) * cols..(i + 2) * cols];
        let x2 = &x.data[(i + 2) * cols..(i + 3) * cols];
        let x3 = &x.data[(i + 3) * cols..(i + 4) * cols];

        let mut acc0 = offset;
        let mut acc1 = offset;
        let mut acc2 = offset;
        let mut acc3 = offset;

        if use_direct {
            validate_row_finite(x0)?;
            validate_row_finite(x1)?;
            validate_row_finite(x2)?;
            validate_row_finite(x3)?;

            let mut j = 0;
            while j + 4 <= n_sv {
                let y0 = &sv.data[j * cols..(j + 1) * cols];
                let y1 = &sv.data[(j + 1) * cols..(j + 2) * cols];
                let y2 = &sv.data[(j + 2) * cols..(j + 3) * cols];
                let y3 = &sv.data[(j + 3) * cols..(j + 4) * cols];
                let d = direct_l2_sq_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                let c0 = coef[j];
                let c1 = coef[j + 1];
                let c2 = coef[j + 2];
                let c3 = coef[j + 3];
                acc0 += (neg_gamma * d[0][0]).exp() * c0
                    + (neg_gamma * d[0][1]).exp() * c1
                    + (neg_gamma * d[0][2]).exp() * c2
                    + (neg_gamma * d[0][3]).exp() * c3;
                acc1 += (neg_gamma * d[1][0]).exp() * c0
                    + (neg_gamma * d[1][1]).exp() * c1
                    + (neg_gamma * d[1][2]).exp() * c2
                    + (neg_gamma * d[1][3]).exp() * c3;
                acc2 += (neg_gamma * d[2][0]).exp() * c0
                    + (neg_gamma * d[2][1]).exp() * c1
                    + (neg_gamma * d[2][2]).exp() * c2
                    + (neg_gamma * d[2][3]).exp() * c3;
                acc3 += (neg_gamma * d[3][0]).exp() * c0
                    + (neg_gamma * d[3][1]).exp() * c1
                    + (neg_gamma * d[3][2]).exp() * c2
                    + (neg_gamma * d[3][3]).exp() * c3;
                j += 4;
            }
            while j < n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                let cj = coef[j];
                acc0 += (neg_gamma * direct_l2_sq_1x1(x0, yj)).exp() * cj;
                acc1 += (neg_gamma * direct_l2_sq_1x1(x1, yj)).exp() * cj;
                acc2 += (neg_gamma * direct_l2_sq_1x1(x2, yj)).exp() * cj;
                acc3 += (neg_gamma * direct_l2_sq_1x1(x3, yj)).exp() * cj;
                j += 1;
            }
        } else {
            let ni0 = validate_and_sq_norm(x0)?;
            let ni1 = validate_and_sq_norm(x1)?;
            let ni2 = validate_and_sq_norm(x2)?;
            let ni3 = validate_and_sq_norm(x3)?;

            let mut j = 0;
            while j + 4 <= n_sv {
                let y0 = &sv.data[j * cols..(j + 1) * cols];
                let y1 = &sv.data[(j + 1) * cols..(j + 2) * cols];
                let y2 = &sv.data[(j + 2) * cols..(j + 3) * cols];
                let y3 = &sv.data[(j + 3) * cols..(j + 4) * cols];
                let dots = dot_4x4([x0, x1, x2, x3], [y0, y1, y2, y3]);
                let nj0 = sv_norms[j];
                let nj1 = sv_norms[j + 1];
                let nj2 = sv_norms[j + 2];
                let nj3 = sv_norms[j + 3];
                let c0 = coef[j];
                let c1 = coef[j + 1];
                let c2 = coef[j + 2];
                let c3 = coef[j + 3];
                acc0 += (neg_gamma * (ni0 + nj0 - 2.0 * dots[0][0]).max(0.0)).exp() * c0
                    + (neg_gamma * (ni0 + nj1 - 2.0 * dots[0][1]).max(0.0)).exp() * c1
                    + (neg_gamma * (ni0 + nj2 - 2.0 * dots[0][2]).max(0.0)).exp() * c2
                    + (neg_gamma * (ni0 + nj3 - 2.0 * dots[0][3]).max(0.0)).exp() * c3;
                acc1 += (neg_gamma * (ni1 + nj0 - 2.0 * dots[1][0]).max(0.0)).exp() * c0
                    + (neg_gamma * (ni1 + nj1 - 2.0 * dots[1][1]).max(0.0)).exp() * c1
                    + (neg_gamma * (ni1 + nj2 - 2.0 * dots[1][2]).max(0.0)).exp() * c2
                    + (neg_gamma * (ni1 + nj3 - 2.0 * dots[1][3]).max(0.0)).exp() * c3;
                acc2 += (neg_gamma * (ni2 + nj0 - 2.0 * dots[2][0]).max(0.0)).exp() * c0
                    + (neg_gamma * (ni2 + nj1 - 2.0 * dots[2][1]).max(0.0)).exp() * c1
                    + (neg_gamma * (ni2 + nj2 - 2.0 * dots[2][2]).max(0.0)).exp() * c2
                    + (neg_gamma * (ni2 + nj3 - 2.0 * dots[2][3]).max(0.0)).exp() * c3;
                acc3 += (neg_gamma * (ni3 + nj0 - 2.0 * dots[3][0]).max(0.0)).exp() * c0
                    + (neg_gamma * (ni3 + nj1 - 2.0 * dots[3][1]).max(0.0)).exp() * c1
                    + (neg_gamma * (ni3 + nj2 - 2.0 * dots[3][2]).max(0.0)).exp() * c2
                    + (neg_gamma * (ni3 + nj3 - 2.0 * dots[3][3]).max(0.0)).exp() * c3;
                j += 4;
            }
            while j < n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                let nj = sv_norms[j];
                let cj = coef[j];
                acc0 += (neg_gamma * (ni0 + nj - 2.0 * dot_1x1(x0, yj)).max(0.0)).exp() * cj;
                acc1 += (neg_gamma * (ni1 + nj - 2.0 * dot_1x1(x1, yj)).max(0.0)).exp() * cj;
                acc2 += (neg_gamma * (ni2 + nj - 2.0 * dot_1x1(x2, yj)).max(0.0)).exp() * cj;
                acc3 += (neg_gamma * (ni3 + nj - 2.0 * dot_1x1(x3, yj)).max(0.0)).exp() * cj;
                j += 1;
            }
        }

        out[i] = acc0;
        out[i + 1] = acc1;
        out[i + 2] = acc2;
        out[i + 3] = acc3;
        i += 4;
    }

    while i < n_test {
        let xi = &x.data[i * cols..(i + 1) * cols];
        let mut acc = offset;
        if use_direct {
            validate_row_finite(xi)?;
            for j in 0..n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                acc += (neg_gamma * direct_l2_sq_1x1(xi, yj)).exp() * coef[j];
            }
        } else {
            let ni = validate_and_sq_norm(xi)?;
            for j in 0..n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                let d2 = (ni + sv_norms[j] - 2.0 * dot_1x1(xi, yj)).max(0.0);
                acc += (neg_gamma * d2).exp() * coef[j];
            }
        }
        out[i] = acc;
        i += 1;
    }

    Ok(out)
}

/// Fused zero-allocation CPU RBF prediction for multi-output `dual_coef` (`n_sv x n_out`).
fn fused_rbf_multi_cpu(
    x: &Matrix,
    sv: &Matrix,
    dual_coef: &Matrix,
    gamma: f64,
) -> OchreResult<Matrix> {
    let n_test = x.rows;
    let n_sv = sv.rows;
    let cols = x.cols;
    let n_out = dual_coef.cols;
    let neg_gamma = -gamma;
    let use_direct = cols <= 16;
    let sv_norms = if use_direct {
        Vec::new()
    } else {
        sv.data
            .chunks_exact(cols)
            .map(row_sq_norm_unrolled)
            .collect()
    };

    let mut out = Matrix::zeros(n_test, n_out);
    for i in 0..n_test {
        let xi = &x.data[i * cols..(i + 1) * cols];
        let out_row = &mut out.data[i * n_out..(i + 1) * n_out];
        if use_direct {
            validate_row_finite(xi)?;
            for j in 0..n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                let kv = (neg_gamma * direct_l2_sq_1x1(xi, yj)).exp();
                let dc_row = &dual_coef.data[j * n_out..(j + 1) * n_out];
                for o in 0..n_out {
                    out_row[o] += kv * dc_row[o];
                }
            }
        } else {
            let ni = validate_and_sq_norm(xi)?;
            for j in 0..n_sv {
                let yj = &sv.data[j * cols..(j + 1) * cols];
                let d2 = (ni + sv_norms[j] - 2.0 * dot_1x1(xi, yj)).max(0.0);
                let kv = (neg_gamma * d2).exp();
                let dc_row = &dual_coef.data[j * n_out..(j + 1) * n_out];
                for o in 0..n_out {
                    out_row[o] += kv * dc_row[o];
                }
            }
        }
    }
    Ok(out)
}

/// Sequential 4-row lower-triangular Cholesky + solve for `n < 512`, avoiding Rayon overhead
/// from `matrix::cholesky_lower_with_diag_shift_into` on medium matrices (e.g., `280 x 280`).
fn try_cholesky_lower_seq(a: &Matrix, lambda: f64, l: &mut Matrix, inv_diag: &mut [f64]) -> bool {
    let n = a.rows;
    let diag_tol = 1e-14;
    for i in 0..n {
        let row_off = i * n;
        let src_row = &a.data[row_off..row_off + n];
        let dst_row = &mut l.data[row_off..row_off + n];
        dst_row[..=i].copy_from_slice(&src_row[..=i]);
        dst_row[i] += lambda;
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

        let d0 = r0[i] - dot_1x1(&r0[..i], &r0[..i]);
        if !d0.is_finite() || d0 <= diag_tol {
            return false;
        }
        let l00 = d0.sqrt();
        let inv0 = 1.0 / l00;
        r0[i] = l00;
        inv_diag[i] = inv0;

        let l10 = (r1[i] - dot_1x1(&r1[..i], &r0[..i])) * inv0;
        let l20 = (r2[i] - dot_1x1(&r2[..i], &r0[..i])) * inv0;
        let l30 = (r3[i] - dot_1x1(&r3[..i], &r0[..i])) * inv0;
        r1[i] = l10;
        r2[i] = l20;
        r3[i] = l30;

        let d1 = r1[i + 1] - dot_1x1(&r1[..i + 1], &r1[..i + 1]);
        if !d1.is_finite() || d1 <= diag_tol {
            return false;
        }
        let l11 = d1.sqrt();
        let inv1 = 1.0 / l11;
        r1[i + 1] = l11;
        inv_diag[i + 1] = inv1;

        let l21 = (r2[i + 1] - dot_1x1(&r2[..i + 1], &r1[..i + 1])) * inv1;
        let l31 = (r3[i + 1] - dot_1x1(&r3[..i + 1], &r1[..i + 1])) * inv1;
        r2[i + 1] = l21;
        r3[i + 1] = l31;

        let d2 = r2[i + 2] - dot_1x1(&r2[..i + 2], &r2[..i + 2]);
        if !d2.is_finite() || d2 <= diag_tol {
            return false;
        }
        let l22 = d2.sqrt();
        let inv2 = 1.0 / l22;
        r2[i + 2] = l22;
        inv_diag[i + 2] = inv2;

        let l32 = (r3[i + 2] - dot_1x1(&r3[..i + 2], &r2[..i + 2])) * inv2;
        r3[i + 2] = l32;

        let d3 = r3[i + 3] - dot_1x1(&r3[..i + 3], &r3[..i + 3]);
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
            ri[j] = (ri[j] - dot_1x1(&ri[..j], lj)) * inv_diag[j];
        }
        let diag = ri[i] - dot_1x1(&ri[..i], &ri[..i]);
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

fn solve_cholesky_multi_fast(a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    let n = a.rows;
    if n >= 512 || n == 0 || b.cols == 0 {
        return solve_cholesky_multi(a, b);
    }
    let mut l = Matrix::zeros(n, n);
    let mut inv_diag = vec![0.0; n];
    for &lambda in &[0.0, 1e-12, 1e-9, 1e-6, 1e-3] {
        if try_cholesky_lower_seq(a, lambda, &mut l, &mut inv_diag) {
            let k_targets = b.cols;
            let mut theta = Matrix::zeros(n, k_targets);
            let mut z = vec![0.0; n];
            let mut x = vec![0.0; n];
            for col in 0..k_targets {
                for r in 0..n {
                    let row = &l.data[r * n..r * n + r];
                    z[r] = (b.data[r * k_targets + col] - dot_1x1(row, &z[..r])) * inv_diag[r];
                }
                x.copy_from_slice(&z);
                for r in (0..n).rev() {
                    let xr = x[r] * inv_diag[r];
                    x[r] = xr;
                    let row = &l.data[r * n..r * n + r];
                    for m in 0..r {
                        x[m] -= row[m] * xr;
                    }
                }
                for r in 0..n {
                    theta.data[r * k_targets + col] = x[r];
                }
            }
            return Ok(theta);
        }
    }
    solve_cholesky_multi(a, b)
}

fn compute_kernel_matrix(
    backend: &Backend<'_>,
    a: &Matrix,
    b: &Matrix,
    kernel: KernelKind,
    gamma: f64,
    degree: usize,
    coef0: f64,
) -> OchreResult<Matrix> {
    match kernel {
        KernelKind::Precomputed => Ok(a.clone()),
        KernelKind::Linear => compute_dot_gram(backend, a, b),
        KernelKind::Rbf => {
            if a.cols != b.cols {
                return Err(OchreError::FeatureCountMismatch {
                    features: a.cols,
                    expected: b.cols,
                });
            }
            if matches!(backend, Backend::Cpu) {
                if std::ptr::eq(a, b) && a.rows <= 2048 {
                    return Ok(fused_rbf_symmetric_cpu(a, gamma));
                }
                return Ok(fused_rbf_cross_cpu(a, b, gamma));
            }
            let mut d2 = backend.pairwise_l2_sq(a, b)?;
            let len = d2.data.len();
            if len >= LARGE_ELEMENTWISE_THRESHOLD {
                for v in &mut d2.data {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
                let scaled = backend.elementwise_op(
                    &d2,
                    ElementwiseOp::Affine {
                        scale: -gamma,
                        shift: 0.0,
                    },
                )?;
                backend.elementwise_op(&scaled, ElementwiseOp::Exp)
            } else {
                for v in &mut d2.data {
                    let dist = (*v).max(0.0);
                    *v = (-gamma * dist).exp();
                }
                Ok(d2)
            }
        }
        KernelKind::Poly => {
            let mut k = compute_dot_gram(backend, a, b)?;
            let deg = degree as i32;
            if k.data.len() >= LARGE_ELEMENTWISE_THRESHOLD {
                k.data.par_iter_mut().for_each(|v| {
                    *v = (gamma * *v + coef0).powi(deg);
                });
            } else {
                for v in &mut k.data {
                    *v = (gamma * *v + coef0).powi(deg);
                }
            }
            Ok(k)
        }
        KernelKind::Sigmoid => {
            let mut k = compute_dot_gram(backend, a, b)?;
            if k.data.len() >= LARGE_ELEMENTWISE_THRESHOLD {
                let affine = backend.elementwise_op(
                    &k,
                    ElementwiseOp::Affine {
                        scale: gamma,
                        shift: coef0,
                    },
                )?;
                backend.elementwise_op(&affine, ElementwiseOp::Tanh)
            } else {
                for v in &mut k.data {
                    *v = (gamma * *v + coef0).tanh();
                }
                Ok(k)
            }
        }
    }
}

fn gather_rows(x: &Matrix, indices: &[usize]) -> Matrix {
    let cols = x.cols;
    let mut out = Matrix::zeros(indices.len(), cols);
    for (dst_r, &src_r) in indices.iter().enumerate() {
        let src_start = src_r * cols;
        let dst_start = dst_r * cols;
        out.data[dst_start..dst_start + cols].copy_from_slice(&x.data[src_start..src_start + cols]);
    }
    out
}

fn slice_precomputed_cols(k_test: &Matrix, support_indices: &[usize]) -> Matrix {
    let rows = k_test.rows;
    let sv_len = support_indices.len();
    let mut out = Matrix::zeros(rows, sv_len);
    for r in 0..rows {
        let src_row = &k_test.data[r * k_test.cols..(r + 1) * k_test.cols];
        let dst_row = &mut out.data[r * sv_len..(r + 1) * sv_len];
        for (j, &idx) in support_indices.iter().enumerate() {
            dst_row[j] = src_row[idx];
        }
    }
    out
}

fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| (*a - *b).abs() <= 1e-12)
        .count();
    (correct as f64) / (y_true.len() as f64)
}

// ============================================================================
// 1. KernelRidge
// ============================================================================

#[derive(Clone, Debug)]
pub struct KernelRidge {
    pub alpha: f64,
    pub kernel: String,
    pub gamma: Option<f64>,
    pub degree: usize,
    pub coef0: f64,
    pub device_: Option<String>,
    pub dual_coef_: Option<Matrix>,
    pub x_fit_: Option<Matrix>,
    pub gamma_: Option<f64>,
    pub n_features_in_: usize,
    pub n_samples_fit_: usize,
    pub n_outputs_: usize,
}

impl KernelRidge {
    pub fn new(
        alpha: f64,
        kernel: String,
        gamma: Option<f64>,
        degree: usize,
        coef0: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        let _ = KernelKind::parse(&kernel)?;
        if let Some(g) = gamma
            && (!g.is_finite() || g <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        validate_coef0(coef0)?;
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        Ok(Self {
            alpha,
            kernel,
            gamma,
            degree,
            coef0,
            device_: device,
            dual_coef_: None,
            x_fit_: None,
            gamma_: None,
            n_features_in_: 0,
            n_samples_fit_: 0,
            n_outputs_: 0,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        self.device_ = device;
        Ok(())
    }

    fn device_type(&self) -> OchreResult<DeviceType> {
        parse_device(self.device_.as_deref())
    }

    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        if y.rows != x.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        if y.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if y.data.iter().any(|v| !v.is_finite()) {
            return Err(OchreError::InvalidParameter(
                "Target y contains NaN or Inf values ( >_< )".to_string(),
            ));
        }

        let k_kind = KernelKind::parse(&self.kernel)?;
        if k_kind == KernelKind::Precomputed && x.rows != x.cols {
            return Err(OchreError::NonSquareMatrix);
        }

        let n = x.rows;
        let sw = if let Some(w) = sample_weight {
            validate_sample_weights(Some(w), n)?;
            Some(w)
        } else {
            None
        };

        let dev = self.device_type()?;
        let backend = effective_backend_krr(&dev, n.saturating_mul(n).max(x.data.len()));
        let gamma_val = resolve_gamma(&backend, x, k_kind, "auto", self.gamma)?;
        let mut k_mat =
            compute_kernel_matrix(&backend, x, x, k_kind, gamma_val, self.degree, self.coef0)?;

        let dual_coef = if let Some(w) = sw {
            let sqrt_w: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();
            for i in 0..n {
                let swi = sqrt_w[i];
                for (j, &swj) in sqrt_w.iter().enumerate().take(n) {
                    k_mat.data[i * n + j] *= swi * swj;
                }
                k_mat.data[i * n + i] += self.alpha;
            }
            let mut y_scaled = y.clone();
            for (i, &swi) in sqrt_w.iter().enumerate().take(n) {
                for c in 0..y_scaled.cols {
                    y_scaled.data[i * y_scaled.cols + c] *= swi;
                }
            }
            let mut c_tilde = solve_cholesky_multi_fast(&k_mat, &y_scaled)?;
            for (i, &swi) in sqrt_w.iter().enumerate().take(n) {
                for c in 0..c_tilde.cols {
                    c_tilde.data[i * c_tilde.cols + c] *= swi;
                }
            }
            c_tilde
        } else {
            for i in 0..n {
                k_mat.data[i * n + i] += self.alpha;
            }
            solve_cholesky_multi_fast(&k_mat, y)?
        };

        self.dual_coef_ = Some(dual_coef);
        self.x_fit_ = if k_kind == KernelKind::Precomputed {
            None
        } else {
            Some(x.clone())
        };
        self.gamma_ = Some(gamma_val);
        self.n_features_in_ = x.cols;
        self.n_samples_fit_ = n;
        self.n_outputs_ = y.cols;
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Matrix> {
        let dual_coef = self.dual_coef_.as_ref().ok_or(OchreError::NotFitted)?;
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let k_kind = KernelKind::parse(&self.kernel)?;
        let gamma_val = self.gamma_.unwrap_or(1.0);

        if k_kind == KernelKind::Rbf
            && x.rows.saturating_mul(self.n_samples_fit_) < FUSED_PREDICT_MAX_ELEMS
        {
            let x_fit = self.x_fit_.as_ref().ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            if dual_coef.cols == 1 {
                let preds = fused_rbf_1d_cpu(x, x_fit, &dual_coef.data, gamma_val, 0.0)?;
                return Matrix::from_vec(x.rows, 1, preds);
            }
            return fused_rbf_multi_cpu(x, x_fit, dual_coef, gamma_val);
        }

        validate_finite_matrix(x)?;
        let dev = self.device_type()?;
        let backend = effective_backend_krr(
            &dev,
            x.rows
                .saturating_mul(self.n_samples_fit_.max(dual_coef.cols)),
        );

        let k_test = if k_kind == KernelKind::Precomputed {
            if x.cols != self.n_samples_fit_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_samples_fit_,
                });
            }
            x.clone()
        } else {
            let x_fit = self.x_fit_.as_ref().ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            compute_kernel_matrix(
                &backend,
                x,
                x_fit,
                k_kind,
                gamma_val,
                self.degree,
                self.coef0,
            )?
        };

        if k_test.rows * dual_coef.cols >= LARGE_MATMUL_THRESHOLD {
            backend.matmul(&k_test, dual_coef)
        } else {
            k_test.matmul(dual_coef)
        }
    }

    pub fn score(&self, x: &Matrix, y: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        if y.rows != x.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let preds = self.predict(x)?;
        if y.cols == 1 && sample_weight.is_none() {
            Ok(r2_score(&y.data, &preds.data))
        } else {
            Ok(r2_score_multi(y, &preds, sample_weight))
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KERNEL_RIDGE);
        write_f64(&mut buf, self.alpha);
        write_str(&mut buf, &self.kernel);
        write_opt_f64(&mut buf, self.gamma);
        write_usize(&mut buf, self.degree);
        write_f64(&mut buf, self.coef0);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.dual_coef_);
        write_opt_matrix(&mut buf, &self.x_fit_);
        write_opt_f64(&mut buf, self.gamma_);
        write_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_fit_);
        write_usize(&mut buf, self.n_outputs_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_KERNEL_RIDGE)?;
        let alpha = r.read_f64()?;
        let kernel = r.read_str()?;
        let gamma = r.read_opt_f64()?;
        let degree = r.read_usize()?;
        let coef0 = r.read_f64()?;
        let stored_dev = r.read_opt_str()?;
        let dual_coef_ = r.read_opt_matrix()?;
        let x_fit_ = r.read_opt_matrix()?;
        let gamma_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_usize()?;
        let n_samples_fit_ = r.read_usize()?;
        let n_outputs_ = r.read_usize()?;
        let (_, resolved_dev) = resolve_loaded_device(stored_dev, true, device)?;
        Ok(Self {
            alpha,
            kernel,
            gamma,
            degree,
            coef0,
            device_: resolved_dev,
            dual_coef_,
            x_fit_,
            gamma_,
            n_features_in_,
            n_samples_fit_,
            n_outputs_,
        })
    }
}

// ============================================================================
// 2. KernelPCA
// ============================================================================

#[derive(Clone, Debug)]
pub struct KernelPCA {
    pub n_components: Option<usize>,
    pub kernel: String,
    pub gamma: Option<f64>,
    pub degree: usize,
    pub coef0: f64,
    pub fit_inverse_transform: bool,
    pub alpha: f64,
    pub device_: Option<String>,
    pub eigenvalues_: Option<Vec<f64>>,
    pub eigenvectors_: Option<Matrix>,
    pub alphas_: Option<Matrix>,
    pub x_fit_: Option<Matrix>,
    pub k_fit_rows_: Option<Vec<f64>>,
    pub k_fit_all_: f64,
    pub gamma_: Option<f64>,
    pub dual_coef_: Option<Matrix>,
    pub x_transformed_fit_: Option<Matrix>,
    pub n_features_in_: usize,
    pub n_samples_fit_: usize,
}

impl KernelPCA {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: Option<usize>,
        kernel: String,
        gamma: Option<f64>,
        degree: usize,
        coef0: f64,
        fit_inverse_transform: bool,
        alpha: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if let Some(0) = n_components {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 (・`ω´・)".to_string(),
            ));
        }
        let _ = KernelKind::parse(&kernel)?;
        if let Some(g) = gamma
            && (!g.is_finite() || g <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        validate_coef0(coef0)?;
        validate_alpha(alpha)?;
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        Ok(Self {
            n_components,
            kernel,
            gamma,
            degree,
            coef0,
            fit_inverse_transform,
            alpha,
            device_: device,
            eigenvalues_: None,
            eigenvectors_: None,
            alphas_: None,
            x_fit_: None,
            k_fit_rows_: None,
            k_fit_all_: 0.0,
            gamma_: None,
            dual_coef_: None,
            x_transformed_fit_: None,
            n_features_in_: 0,
            n_samples_fit_: 0,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        self.device_ = device;
        Ok(())
    }

    fn device_type(&self) -> OchreResult<DeviceType> {
        parse_device(self.device_.as_deref())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        let k_kind = KernelKind::parse(&self.kernel)?;
        if k_kind == KernelKind::Precomputed && x.rows != x.cols {
            return Err(OchreError::NonSquareMatrix);
        }
        if self.fit_inverse_transform && k_kind == KernelKind::Precomputed {
            return Err(OchreError::InvalidParameter(
                "fit_inverse_transform is not supported with kernel='precomputed' (・`ω´・)"
                    .to_string(),
            ));
        }

        let n = x.rows;
        let dev = self.device_type()?;
        let backend = effective_backend_krr(&dev, n.saturating_mul(n).max(x.data.len()));
        let gamma_val = resolve_gamma(&backend, x, k_kind, "auto", self.gamma)?;
        let k_mat =
            compute_kernel_matrix(&backend, x, x, k_kind, gamma_val, self.degree, self.coef0)?;

        // Double-center Gram matrix: K_c = K - row_mean - col_mean + total_mean
        let mut row_means = vec![0.0; n];
        for (i, rm) in row_means.iter_mut().enumerate().take(n) {
            let row = &k_mat.data[i * n..(i + 1) * n];
            *rm = row.iter().sum::<f64>() / (n as f64);
        }
        let total_mean = row_means.iter().sum::<f64>() / (n as f64);

        let mut k_centered = Matrix::zeros(n, n);
        for i in 0..n {
            let ri = row_means[i];
            for (j, &rj) in row_means.iter().enumerate().take(n) {
                k_centered.data[i * n + j] = k_mat.data[i * n + j] - ri - rj + total_mean;
            }
        }

        let max_comp = self.n_components.unwrap_or(n).min(n);
        let eig = if n >= 64 && max_comp >= 1 && max_comp <= n / 4 {
            crate::primitives::eig_svd::cpu_top_k_sym_eig(&k_centered, max_comp)?
        } else {
            backend.small_sym_eig_svd(&k_centered)?
        };

        let mut pos_indices = Vec::with_capacity(max_comp);
        for (idx, &lam) in eig.eigenvalues.iter().enumerate() {
            if pos_indices.len() >= max_comp {
                break;
            }
            if lam > 1e-12 {
                pos_indices.push(idx);
            }
        }

        let k_kept = pos_indices.len();
        let mut kept_eigs = Vec::with_capacity(k_kept);
        let mut kept_vecs = Matrix::zeros(n, k_kept);
        let mut kept_alphas = Matrix::zeros(n, k_kept);
        let eig_cols = eig.eigenvectors.cols;

        for (out_col, &eig_col) in pos_indices.iter().enumerate() {
            let lam = eig.eigenvalues[eig_col];
            kept_eigs.push(lam);
            let inv_sqrt = 1.0 / lam.sqrt();
            // Deterministic sign convention: largest abs element in column is positive
            let mut max_abs = -1.0f64;
            let mut sign = 1.0f64;
            for r in 0..n {
                let v = eig.eigenvectors.data[r * eig_cols + eig_col];
                if v.abs() > max_abs {
                    max_abs = v.abs();
                    sign = if v < 0.0 { -1.0 } else { 1.0 };
                }
            }
            for r in 0..n {
                let v = eig.eigenvectors.data[r * eig_cols + eig_col] * sign;
                kept_vecs.data[r * k_kept + out_col] = v;
                kept_alphas.data[r * k_kept + out_col] = v * inv_sqrt;
            }
        }

        // Fit inverse transform via KernelRidge(Z -> X) if requested
        let (dual_coef_inv, x_trans_fit) = if self.fit_inverse_transform && k_kept > 0 {
            let z = if n * k_kept >= LARGE_MATMUL_THRESHOLD {
                backend.matmul(&k_centered, &kept_alphas)?
            } else {
                k_centered.matmul(&kept_alphas)?
            };
            let mut k_z = compute_kernel_matrix(
                &backend,
                &z,
                &z,
                k_kind,
                gamma_val,
                self.degree,
                self.coef0,
            )?;
            for i in 0..n {
                k_z.data[i * n + i] += self.alpha;
            }
            let dc = solve_cholesky_multi(&k_z, x)?;
            (Some(dc), Some(z))
        } else {
            (None, None)
        };

        self.eigenvalues_ = Some(kept_eigs);
        self.eigenvectors_ = Some(kept_vecs);
        self.alphas_ = Some(kept_alphas);
        self.x_fit_ = if k_kind == KernelKind::Precomputed {
            None
        } else {
            Some(x.clone())
        };
        self.k_fit_rows_ = Some(row_means);
        self.k_fit_all_ = total_mean;
        self.gamma_ = Some(gamma_val);
        self.dual_coef_ = dual_coef_inv;
        self.x_transformed_fit_ = x_trans_fit;
        self.n_features_in_ = x.cols;
        self.n_samples_fit_ = n;
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let alphas = self.alphas_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_fit_rows = self.k_fit_rows_.as_ref().ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x)?;
        let k_kind = KernelKind::parse(&self.kernel)?;
        let dev = self.device_type()?;
        let backend = effective_backend(
            &dev,
            x.rows.saturating_mul(self.n_samples_fit_.max(alphas.cols)),
        );
        let gamma_val = self.gamma_.unwrap_or(1.0);

        let mut k_test = if k_kind == KernelKind::Precomputed {
            if x.cols != self.n_samples_fit_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_samples_fit_,
                });
            }
            x.clone()
        } else {
            let x_fit = self.x_fit_.as_ref().ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            compute_kernel_matrix(
                &backend,
                x,
                x_fit,
                k_kind,
                gamma_val,
                self.degree,
                self.coef0,
            )?
        };

        let m = k_test.rows;
        let n = k_test.cols;
        for i in 0..m {
            let row_slice = &mut k_test.data[i * n..(i + 1) * n];
            let test_row_mean = row_slice.iter().sum::<f64>() / (n as f64);
            for j in 0..n {
                row_slice[j] = row_slice[j] - test_row_mean - k_fit_rows[j] + self.k_fit_all_;
            }
        }

        if alphas.cols == 0 {
            return Ok(Matrix::zeros(m, 0));
        }
        if m * alphas.cols >= LARGE_MATMUL_THRESHOLD {
            backend.matmul(&k_test, alphas)
        } else {
            k_test.matmul(alphas)
        }
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, z: &Matrix) -> OchreResult<Matrix> {
        let dual_coef = self.dual_coef_.as_ref().ok_or_else(|| {
            OchreError::InvalidParameter(
                "KernelPCA was not fitted with fit_inverse_transform=True (・`ω´・)".to_string(),
            )
        })?;
        let x_trans_fit = self
            .x_transformed_fit_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(z)?;
        if z.cols != x_trans_fit.cols {
            return Err(OchreError::FeatureCountMismatch {
                features: z.cols,
                expected: x_trans_fit.cols,
            });
        }
        let k_kind = KernelKind::parse(&self.kernel)?;
        let dev = self.device_type()?;
        let backend = effective_backend(
            &dev,
            z.rows.saturating_mul(x_trans_fit.rows.max(dual_coef.cols)),
        );
        let gamma_val = self.gamma_.unwrap_or(1.0);
        let k_z = compute_kernel_matrix(
            &backend,
            z,
            x_trans_fit,
            k_kind,
            gamma_val,
            self.degree,
            self.coef0,
        )?;
        if k_z.rows * dual_coef.cols >= LARGE_MATMUL_THRESHOLD {
            backend.matmul(&k_z, dual_coef)
        } else {
            k_z.matmul(dual_coef)
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KERNEL_PCA);
        write_opt_usize(&mut buf, self.n_components);
        write_str(&mut buf, &self.kernel);
        write_opt_f64(&mut buf, self.gamma);
        write_usize(&mut buf, self.degree);
        write_f64(&mut buf, self.coef0);
        write_bool(&mut buf, self.fit_inverse_transform);
        write_f64(&mut buf, self.alpha);
        write_opt_str(&mut buf, &self.device_);
        match &self.eigenvalues_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_matrix(&mut buf, &self.eigenvectors_);
        write_opt_matrix(&mut buf, &self.alphas_);
        write_opt_matrix(&mut buf, &self.x_fit_);
        match &self.k_fit_rows_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_f64(&mut buf, self.k_fit_all_);
        write_opt_f64(&mut buf, self.gamma_);
        write_opt_matrix(&mut buf, &self.dual_coef_);
        write_opt_matrix(&mut buf, &self.x_transformed_fit_);
        write_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_KERNEL_PCA)?;
        let n_components = r.read_opt_usize()?;
        let kernel = r.read_str()?;
        let gamma = r.read_opt_f64()?;
        let degree = r.read_usize()?;
        let coef0 = r.read_f64()?;
        let fit_inverse_transform = r.read_bool()?;
        let alpha = r.read_f64()?;
        let stored_dev = r.read_opt_str()?;
        let eigenvalues_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let eigenvectors_ = r.read_opt_matrix()?;
        let alphas_ = r.read_opt_matrix()?;
        let x_fit_ = r.read_opt_matrix()?;
        let k_fit_rows_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let k_fit_all_ = r.read_f64()?;
        let gamma_ = r.read_opt_f64()?;
        let dual_coef_ = r.read_opt_matrix()?;
        let x_transformed_fit_ = r.read_opt_matrix()?;
        let n_features_in_ = r.read_usize()?;
        let n_samples_fit_ = r.read_usize()?;
        let (_, resolved_dev) = resolve_loaded_device(stored_dev, true, device)?;
        Ok(Self {
            n_components,
            kernel,
            gamma,
            degree,
            coef0,
            fit_inverse_transform,
            alpha,
            device_: resolved_dev,
            eigenvalues_,
            eigenvectors_,
            alphas_,
            x_fit_,
            k_fit_rows_,
            k_fit_all_,
            gamma_,
            dual_coef_,
            x_transformed_fit_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// Shared LIBSVM WSS3 SMO Solvers
// ============================================================================

/// Binary SVC SMO solver on precomputed Gram matrix `k` (indices `subset` into `k`, or full `k`).
/// `y_sign` has values `+1.0` or `-1.0`, length `m = subset.len()`.
fn solve_svc_binary_smo(
    k_full: &Matrix,
    subset: &[usize],
    y_sign: &[f64],
    c_pos: f64,
    c_neg: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    let m = subset.len();
    let n_full = k_full.cols;
    let mut alpha = vec![0.0; m];
    let mut grad = vec![-1.0; m];

    let upper_bound = |idx: usize| -> f64 { if y_sign[idx] > 0.0 { c_pos } else { c_neg } };

    for _iter in 0..max_iter {
        // 1. Select i maximizing -y_t * grad[t] in I_up
        let mut g_max = f64::NEG_INFINITY;
        let mut i_opt = usize::MAX;
        for t in 0..m {
            let ub = upper_bound(t);
            let in_up = (y_sign[t] > 0.0 && alpha[t] < ub) || (y_sign[t] < 0.0 && alpha[t] > 0.0);
            if in_up {
                let val = -y_sign[t] * grad[t];
                if val >= g_max {
                    g_max = val;
                    i_opt = t;
                }
            }
        }
        if i_opt == usize::MAX {
            break;
        }

        // 2. Select j in I_low maximizing b^2 / a
        let mut g_min = f64::INFINITY;
        let mut obj_diff_min = f64::INFINITY;
        let mut j_opt = usize::MAX;

        let si = subset[i_opt];
        let k_ii = k_full.data[si * n_full + si];

        for t in 0..m {
            let ub = upper_bound(t);
            let in_low = (y_sign[t] > 0.0 && alpha[t] > 0.0) || (y_sign[t] < 0.0 && alpha[t] < ub);
            if in_low {
                let neg_yg = -y_sign[t] * grad[t];
                if neg_yg <= g_min {
                    g_min = neg_yg;
                }
                let b = g_max - neg_yg;
                if b > 0.0 {
                    let st = subset[t];
                    let k_tt = k_full.data[st * n_full + st];
                    let k_it = k_full.data[si * n_full + st];
                    let mut a = k_ii + k_tt - 2.0 * k_it;
                    if a <= 0.0 {
                        a = 1e-12;
                    }
                    let obj_diff = -(b * b) / a;
                    if obj_diff <= obj_diff_min {
                        obj_diff_min = obj_diff;
                        j_opt = t;
                    }
                }
            }
        }

        if g_max - g_min < tol || j_opt == usize::MAX {
            break;
        }

        // 3. Update alpha[i_opt] and alpha[j_opt]
        let sj = subset[j_opt];
        let k_jj = k_full.data[sj * n_full + sj];
        let k_ij = k_full.data[si * n_full + sj];
        let mut a = k_ii + k_jj - 2.0 * k_ij;
        if a <= 0.0 {
            a = 1e-12;
        }
        let b = (-y_sign[i_opt] * grad[i_opt]) - (-y_sign[j_opt] * grad[j_opt]);

        let old_ai = alpha[i_opt];
        let old_aj = alpha[j_opt];
        let ub_i = upper_bound(i_opt);
        let ub_j = upper_bound(j_opt);

        alpha[i_opt] += y_sign[i_opt] * b / a;
        alpha[j_opt] -= y_sign[j_opt] * b / a;

        let sum = y_sign[i_opt] * old_ai + y_sign[j_opt] * old_aj;
        if alpha[i_opt] > ub_i {
            alpha[i_opt] = ub_i;
        } else if alpha[i_opt] < 0.0 {
            alpha[i_opt] = 0.0;
        }
        alpha[j_opt] = y_sign[j_opt] * (sum - y_sign[i_opt] * alpha[i_opt]);

        if alpha[j_opt] > ub_j {
            alpha[j_opt] = ub_j;
        } else if alpha[j_opt] < 0.0 {
            alpha[j_opt] = 0.0;
        }
        alpha[i_opt] = y_sign[i_opt] * (sum - y_sign[j_opt] * alpha[j_opt]);

        let delta_i = alpha[i_opt] - old_ai;
        let delta_j = alpha[j_opt] - old_aj;
        let y_di = y_sign[i_opt] * delta_i;
        let y_dj = y_sign[j_opt] * delta_j;

        // 4. Update gradients
        let row_i = &k_full.data[si * n_full..(si + 1) * n_full];
        let row_j = &k_full.data[sj * n_full..(sj + 1) * n_full];
        for t in 0..m {
            let st = subset[t];
            grad[t] += y_sign[t] * (row_i[st] * y_di + row_j[st] * y_dj);
        }
    }

    // Compute intercept b (decision = sum_i y_i alpha_i K(x_i, x) + intercept)
    let mut sum_free = 0.0;
    let mut nr_free = 0usize;
    let mut ub_val = f64::INFINITY;
    let mut lb_val = f64::NEG_INFINITY;
    for t in 0..m {
        let ub = upper_bound(t);
        let yg = y_sign[t] * grad[t];
        if alpha[t] >= ub {
            if y_sign[t] < 0.0 {
                ub_val = ub_val.min(yg);
            } else {
                lb_val = lb_val.max(yg);
            }
        } else if alpha[t] <= 0.0 {
            if y_sign[t] > 0.0 {
                ub_val = ub_val.min(yg);
            } else {
                lb_val = lb_val.max(yg);
            }
        } else {
            nr_free += 1;
            sum_free += yg;
        }
    }
    let intercept = if nr_free > 0 {
        -sum_free / (nr_free as f64)
    } else {
        -0.5 * (ub_val + lb_val)
    };

    (alpha, intercept)
}

/// Platt's sigmoid calibration (Lin, Lin, Weng 2007) fitting `P(y=1|f) = 1 / (1 + exp(A*f + B))`.
fn fit_platt_scaling(dec_values: &[f64], labels_bool: &[bool]) -> (f64, f64) {
    let prior1 = labels_bool.iter().filter(|&&b| b).count() as f64;
    let prior0 = (labels_bool.len() as f64) - prior1;
    let hi_target = (prior1 + 1.0) / (prior1 + 2.0);
    let lo_target = 1.0 / (prior0 + 2.0);

    let mut a = 0.0f64;
    let mut b = ((prior0 + 1.0) / (prior1 + 1.0)).ln();
    let max_iter = 100;
    let min_step = 1e-10;
    let sigma = 1e-12;
    let eps = 1e-5;

    let mut fval = 0.0;
    for (&f, &lbl) in dec_values.iter().zip(labels_bool.iter()) {
        let t = if lbl { hi_target } else { lo_target };
        let f_ap_b = f * a + b;
        if f_ap_b >= 0.0 {
            fval += t * f_ap_b + (1.0 + (-f_ap_b).exp()).ln();
        } else {
            fval += (t - 1.0) * f_ap_b + (1.0 + f_ap_b.exp()).ln();
        }
    }

    for _ in 0..max_iter {
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = 0.0;
        let mut g1 = 0.0;
        let mut g2 = 0.0;

        for (&f, &lbl) in dec_values.iter().zip(labels_bool.iter()) {
            let t = if lbl { hi_target } else { lo_target };
            let f_ap_b = f * a + b;
            let (p, q) = if f_ap_b >= 0.0 {
                let e = (-f_ap_b).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = f_ap_b.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d2 = p * q;
            h11 += f * f * d2;
            h22 += d2;
            h21 += f * d2;
            let d1 = t - p;
            g1 += f * d1;
            g2 += d1;
        }

        if g1.abs() < eps && g2.abs() < eps {
            break;
        }

        let det = h11 * h22 - h21 * h21;
        let d_a = -(h22 * g1 - h21 * g2) / det;
        let d_b = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * d_a + g2 * d_b;

        let mut step = 1.0;
        while step >= min_step {
            let new_a = a + step * d_a;
            let new_b = b + step * d_b;
            let mut new_f = 0.0;
            for (&f, &lbl) in dec_values.iter().zip(labels_bool.iter()) {
                let t = if lbl { hi_target } else { lo_target };
                let f_ap_b = f * new_a + new_b;
                if f_ap_b >= 0.0 {
                    new_f += t * f_ap_b + (1.0 + (-f_ap_b).exp()).ln();
                } else {
                    new_f += (t - 1.0) * f_ap_b + (1.0 + f_ap_b.exp()).ln();
                }
            }
            if new_f < fval + 0.0001 * step * gd {
                a = new_a;
                b = new_b;
                fval = new_f;
                break;
            }
            step *= 0.5;
        }
        if step < min_step {
            break;
        }
    }

    (a, b)
}

#[inline]
fn sigmoid_predict(dec: f64, a: f64, b: f64) -> f64 {
    let f_ap_b = dec * a + b;
    if f_ap_b >= 0.0 {
        (-f_ap_b).exp() / (1.0 + (-f_ap_b).exp())
    } else {
        1.0 / (1.0 + f_ap_b.exp())
    }
}

/// Pairwise coupling (Wu, Lin, Weng 2004) for multiclass probability estimation.
fn multiclass_probability(k: usize, r: &[f64]) -> Vec<f64> {
    let mut p = vec![1.0 / (k as f64); k];
    let mut q = vec![0.0; k * k];
    let eps = 0.005 / (k as f64);

    for t in 0..k {
        q[t * k + t] = 0.0;
        for j in 0..t {
            let r_jt = r[j * k + t];
            q[t * k + t] += r_jt * r_jt;
            q[t * k + j] = q[j * k + t];
        }
        for j in (t + 1)..k {
            let r_jt = r[j * k + t];
            q[t * k + t] += r_jt * r_jt;
            q[t * k + j] = -r[t * k + j] * r_jt;
        }
    }

    let mut qp = vec![0.0; k];
    for _ in 0..(100 * k) {
        let mut p_qp = 0.0;
        for t in 0..k {
            let mut s = 0.0;
            for j in 0..k {
                s += q[t * k + j] * p[j];
            }
            qp[t] = s;
            p_qp += p[t] * s;
        }

        let mut max_error = 0.0f64;
        for &qp_t in qp.iter().take(k) {
            max_error = max_error.max((qp_t - p_qp).abs());
        }
        if max_error < eps {
            break;
        }

        for t in 0..k {
            let diff = (-qp[t] + p_qp) / q[t * k + t].max(1e-12);
            p[t] += diff;
            p_qp =
                (p_qp + diff * (diff * q[t * k + t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
            for j in 0..k {
                qp[j] = (qp[j] + diff * q[t * k + j]) / (1.0 + diff);
                p[j] /= 1.0 + diff;
            }
        }
    }

    let sum: f64 = p.iter().sum();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    }
    p
}

// ============================================================================
// 3. SVC
// ============================================================================

#[derive(Clone, Debug)]
pub struct SVC {
    pub c: f64,
    pub kernel: String,
    pub degree: usize,
    pub gamma_mode: String,
    pub gamma: Option<f64>,
    pub coef0: f64,
    pub probability: bool,
    pub tol: f64,
    pub max_iter: Option<usize>,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    pub classes_: Option<Vec<f64>>,
    pub support_: Option<Vec<usize>>,
    pub support_vectors_: Option<Matrix>,
    pub n_support_: Option<Vec<usize>>,
    pub dual_coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub prob_a_: Option<Vec<f64>>,
    pub prob_b_: Option<Vec<f64>>,
    pub gamma_: Option<f64>,
    pub n_features_in_: usize,
    pub n_samples_fit_: usize,
}

impl SVC {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: f64,
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        probability: bool,
        tol: f64,
        max_iter: Option<usize>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_c(c)?;
        let _ = KernelKind::parse(&kernel)?;
        if let Some(g) = gamma
            && (!g.is_finite() || g <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        validate_coef0(coef0)?;
        validate_tol(tol)?;
        let _ = validate_max_iter(max_iter)?;
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        Ok(Self {
            c,
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            probability,
            tol,
            max_iter,
            random_state,
            device_: device,
            classes_: None,
            support_: None,
            support_vectors_: None,
            n_support_: None,
            dual_coef_: None,
            intercept_: None,
            prob_a_: None,
            prob_b_: None,
            gamma_: None,
            n_features_in_: 0,
            n_samples_fit_: 0,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        self.device_ = device;
        Ok(())
    }

    fn device_type(&self) -> OchreResult<DeviceType> {
        parse_device(self.device_.as_deref())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        validate_finite_y(y, x.rows)?;
        let k_kind = KernelKind::parse(&self.kernel)?;
        if k_kind == KernelKind::Precomputed && x.rows != x.cols {
            return Err(OchreError::NonSquareMatrix);
        }

        let classes = extract_sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(OchreError::InsufficientClasses(n_classes));
        }

        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(x.rows).max(x.data.len()));
        let gamma_val = resolve_gamma(&backend, x, k_kind, &self.gamma_mode, self.gamma)?;
        let k_mat =
            compute_kernel_matrix(&backend, x, x, k_kind, gamma_val, self.degree, self.coef0)?;
        let max_it = validate_max_iter(self.max_iter)?;

        // Group sample indices by class
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (idx, &yv) in y.iter().enumerate() {
            if let Some(c_idx) = classes.iter().position(|&c| (c - yv).abs() <= 1e-12) {
                class_indices[c_idx].push(idx);
            }
        }

        let n_pairs = n_classes * (n_classes - 1) / 2;
        let mut pairs = Vec::with_capacity(n_pairs);
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                pairs.push((i, j));
            }
        }

        // Solve OvO binary SMO problems
        let pair_results: Vec<(Vec<usize>, Vec<f64>, f64)> = pairs
            .par_iter()
            .map(|&(ci, cj)| {
                let mut subset =
                    Vec::with_capacity(class_indices[ci].len() + class_indices[cj].len());
                let mut y_sign = Vec::with_capacity(subset.capacity());
                // Binary case (n_classes == 2): class 1 is +1, class 0 is -1 (matching sklearn)
                // Multiclass OvO (n_classes > 2): class ci is +1, class cj is -1
                if n_classes == 2 {
                    for &idx in &class_indices[1] {
                        subset.push(idx);
                        y_sign.push(1.0);
                    }
                    for &idx in &class_indices[0] {
                        subset.push(idx);
                        y_sign.push(-1.0);
                    }
                } else {
                    for &idx in &class_indices[ci] {
                        subset.push(idx);
                        y_sign.push(1.0);
                    }
                    for &idx in &class_indices[cj] {
                        subset.push(idx);
                        y_sign.push(-1.0);
                    }
                }
                let (alpha, intercept) = solve_svc_binary_smo(
                    &k_mat, &subset, &y_sign, self.c, self.c, self.tol, max_it,
                );
                let signed_alpha: Vec<f64> = alpha
                    .iter()
                    .zip(y_sign.iter())
                    .map(|(&a, &ys)| a * ys)
                    .collect();
                (subset, signed_alpha, intercept)
            })
            .collect();

        // Determine support vectors per class (sorted by class order, matching LIBSVM)
        let n = x.rows;
        let mut is_sv = vec![false; n];
        for (subset, signed_alpha, _) in &pair_results {
            for (&orig_idx, &sa) in subset.iter().zip(signed_alpha.iter()) {
                if sa.abs() > 1e-12 {
                    is_sv[orig_idx] = true;
                }
            }
        }

        let mut support_indices = Vec::new();
        let mut n_support = vec![0usize; n_classes];
        let mut orig_to_sv = vec![usize::MAX; n];
        for (c_idx, idxs) in class_indices.iter().enumerate() {
            for &orig_idx in idxs {
                if is_sv[orig_idx] {
                    orig_to_sv[orig_idx] = support_indices.len();
                    support_indices.push(orig_idx);
                    n_support[c_idx] += 1;
                }
            }
        }

        let total_sv = support_indices.len();
        let mut dual_coef = Matrix::zeros(n_classes - 1, total_sv);
        let mut intercepts = Vec::with_capacity(n_pairs);

        if n_classes == 2 {
            let (subset, signed_alpha, intercept) = &pair_results[0];
            intercepts.push(*intercept);
            for (&orig_idx, &sa) in subset.iter().zip(signed_alpha.iter()) {
                if sa.abs() > 1e-12 {
                    let sv_col = orig_to_sv[orig_idx];
                    dual_coef.data[sv_col] = sa;
                }
            }
        } else {
            for (p_idx, &(ci, cj)) in pairs.iter().enumerate() {
                let (subset, signed_alpha, intercept) = &pair_results[p_idx];
                intercepts.push(*intercept);
                for (&orig_idx, &sa) in subset.iter().zip(signed_alpha.iter()) {
                    if sa.abs() > 1e-12 {
                        let sv_col = orig_to_sv[orig_idx];
                        if class_indices[ci].contains(&orig_idx) {
                            dual_coef.data[(cj - 1) * total_sv + sv_col] = sa;
                        } else {
                            dual_coef.data[ci * total_sv + sv_col] = sa;
                        }
                    }
                }
            }
        }

        // Optional Platt scaling via 3-fold CV on each pair
        let (prob_a, prob_b) = if self.probability {
            let mut rng = SplitMix64::new(self.random_state);
            let mut pa_vec = Vec::with_capacity(n_pairs);
            let mut pb_vec = Vec::with_capacity(n_pairs);
            for &(ci, cj) in &pairs {
                let mut subset =
                    Vec::with_capacity(class_indices[ci].len() + class_indices[cj].len());
                let mut y_bool = Vec::with_capacity(subset.capacity());
                let (pos_c, neg_c) = if n_classes == 2 { (1, 0) } else { (ci, cj) };
                for &idx in &class_indices[pos_c] {
                    subset.push(idx);
                    y_bool.push(true);
                }
                for &idx in &class_indices[neg_c] {
                    subset.push(idx);
                    y_bool.push(false);
                }
                let m = subset.len();
                let mut perm: Vec<usize> = (0..m).collect();
                rng.shuffle(&mut perm);
                let n_folds = 3.min(m).max(2);
                let mut dec_vals = vec![0.0; m];
                for fold in 0..n_folds {
                    let mut train_sub = Vec::new();
                    let mut train_y = Vec::new();
                    let mut val_pos = Vec::new();
                    for (pos, &orig_p) in perm.iter().enumerate() {
                        if pos % n_folds == fold {
                            val_pos.push(orig_p);
                        } else {
                            train_sub.push(subset[orig_p]);
                            train_y.push(if y_bool[orig_p] { 1.0 } else { -1.0 });
                        }
                    }
                    if train_sub.is_empty() || val_pos.is_empty() {
                        continue;
                    }
                    let (a_fold, b_fold) = solve_svc_binary_smo(
                        &k_mat, &train_sub, &train_y, self.c, self.c, self.tol, max_it,
                    );
                    for &vp in &val_pos {
                        let sv_idx = subset[vp];
                        let mut s = b_fold;
                        for (t, &tr_idx) in train_sub.iter().enumerate() {
                            s += a_fold[t] * train_y[t] * k_mat.data[tr_idx * n + sv_idx];
                        }
                        dec_vals[vp] = s;
                    }
                }
                let (pa, pb) = fit_platt_scaling(&dec_vals, &y_bool);
                pa_vec.push(pa);
                pb_vec.push(pb);
            }
            (Some(pa_vec), Some(pb_vec))
        } else {
            (None, None)
        };

        self.classes_ = Some(classes);
        self.support_ = Some(support_indices.clone());
        self.support_vectors_ = if k_kind == KernelKind::Precomputed {
            None
        } else {
            Some(gather_rows(x, &support_indices))
        };
        self.n_support_ = Some(n_support);
        self.dual_coef_ = Some(dual_coef);
        self.intercept_ = Some(intercepts);
        self.prob_a_ = prob_a;
        self.prob_b_ = prob_b;
        self.gamma_ = Some(gamma_val);
        self.n_features_in_ = x.cols;
        self.n_samples_fit_ = n;
        Ok(())
    }

    fn compute_pairwise_decisions(&self, x: &Matrix) -> OchreResult<Matrix> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let support = self.support_.as_ref().ok_or(OchreError::NotFitted)?;
        let dual_coef = self.dual_coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercepts = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_support = self.n_support_.as_ref().ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x)?;

        let k_kind = KernelKind::parse(&self.kernel)?;
        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(support.len()));
        let gamma_val = self.gamma_.unwrap_or(1.0);

        let k_sv = if k_kind == KernelKind::Precomputed {
            if x.cols != self.n_samples_fit_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_samples_fit_,
                });
            }
            slice_precomputed_cols(x, support)
        } else {
            let sv = self
                .support_vectors_
                .as_ref()
                .ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            compute_kernel_matrix(&backend, x, sv, k_kind, gamma_val, self.degree, self.coef0)?
        };

        let n_classes = classes.len();
        let n_pairs = intercepts.len();
        let m = x.rows;
        let total_sv = support.len();
        let mut dec = Matrix::zeros(m, n_pairs);

        if n_classes == 2 {
            for r in 0..m {
                let k_row = &k_sv.data[r * total_sv..(r + 1) * total_sv];
                let mut sum = intercepts[0];
                for (c, &kv) in k_row.iter().enumerate().take(total_sv) {
                    sum += dual_coef.data[c] * kv;
                }
                dec.data[r] = sum;
            }
            return Ok(dec);
        }

        let mut sv_start = vec![0usize; n_classes];
        for c in 1..n_classes {
            sv_start[c] = sv_start[c - 1] + n_support[c - 1];
        }

        for r in 0..m {
            let k_row = &k_sv.data[r * total_sv..(r + 1) * total_sv];
            let mut p_idx = 0;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let mut sum = intercepts[p_idx];
                    let si = sv_start[i];
                    let ci = n_support[i];
                    for k in 0..ci {
                        sum += dual_coef.data[(j - 1) * total_sv + si + k] * k_row[si + k];
                    }
                    let sj = sv_start[j];
                    let cj = n_support[j];
                    for k in 0..cj {
                        sum += dual_coef.data[i * total_sv + sj + k] * k_row[sj + k];
                    }
                    dec.data[r * n_pairs + p_idx] = sum;
                    p_idx += 1;
                }
            }
        }
        Ok(dec)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let pair_dec = self.compute_pairwise_decisions(x)?;
        let n_classes = classes.len();
        if n_classes == 2 {
            return Ok(pair_dec);
        }

        let m = pair_dec.rows;
        let n_pairs = pair_dec.cols;
        let mut out = Matrix::zeros(m, n_classes);
        for r in 0..m {
            let mut votes = vec![0.0; n_classes];
            let mut conf = vec![0.0; n_classes];
            let mut p_idx = 0;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let d = pair_dec.data[r * n_pairs + p_idx];
                    conf[i] += d;
                    conf[j] -= d;
                    if d > 0.0 {
                        votes[i] += 1.0;
                    } else {
                        votes[j] += 1.0;
                    }
                    p_idx += 1;
                }
            }
            for c in 0..n_classes {
                out.data[r * n_classes + c] = votes[c] + conf[c] / (3.0 * (conf[c].abs() + 1.0));
            }
        }
        Ok(out)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let dec = self.decision_function(x)?;
        let n_classes = classes.len();
        let mut preds = Vec::with_capacity(dec.rows);
        if n_classes == 2 {
            for &d in &dec.data {
                preds.push(if d > 0.0 { classes[1] } else { classes[0] });
            }
        } else {
            for r in 0..dec.rows {
                let row = &dec.data[r * n_classes..(r + 1) * n_classes];
                let mut best_c = 0;
                let mut best_v = f64::NEG_INFINITY;
                for (c, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best_c = c;
                    }
                }
                preds.push(classes[best_c]);
            }
        }
        Ok(preds)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let pa = self.prob_a_.as_ref().ok_or_else(|| {
            OchreError::InvalidParameter(
                "predict_proba is not available when probability=False (・`ω´・)".to_string(),
            )
        })?;
        let pb = self.prob_b_.as_ref().ok_or(OchreError::NotFitted)?;
        let pair_dec = self.compute_pairwise_decisions(x)?;
        let n_classes = classes.len();
        let m = pair_dec.rows;
        let mut probs = Matrix::zeros(m, n_classes);

        if n_classes == 2 {
            for r in 0..m {
                let p1 = sigmoid_predict(pair_dec.data[r], pa[0], pb[0]).clamp(1e-7, 1.0 - 1e-7);
                probs.data[r * 2] = 1.0 - p1;
                probs.data[r * 2 + 1] = p1;
            }
            return Ok(probs);
        }

        let n_pairs = pair_dec.cols;
        let mut pair_mat = vec![0.0; n_classes * n_classes];
        for r in 0..m {
            let mut p_idx = 0;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let d = pair_dec.data[r * n_pairs + p_idx];
                    let p_ij = sigmoid_predict(d, pa[p_idx], pb[p_idx]).clamp(1e-7, 1.0 - 1e-7);
                    pair_mat[i * n_classes + j] = p_ij;
                    pair_mat[j * n_classes + i] = 1.0 - p_ij;
                    p_idx += 1;
                }
            }
            let p_vec = multiclass_probability(n_classes, &pair_mat);
            probs.data[r * n_classes..(r + 1) * n_classes].copy_from_slice(&p_vec);
        }
        Ok(probs)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        if y.len() != x.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        Ok(accuracy_score(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SVC);
        write_f64(&mut buf, self.c);
        write_str(&mut buf, &self.kernel);
        write_usize(&mut buf, self.degree);
        write_str(&mut buf, &self.gamma_mode);
        write_opt_f64(&mut buf, self.gamma);
        write_f64(&mut buf, self.coef0);
        write_bool(&mut buf, self.probability);
        write_f64(&mut buf, self.tol);
        write_opt_usize(&mut buf, self.max_iter);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        match &self.classes_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        match &self.support_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_matrix(&mut buf, &self.support_vectors_);
        match &self.n_support_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_matrix(&mut buf, &self.dual_coef_);
        match &self.intercept_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        match &self.prob_a_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        match &self.prob_b_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_f64(&mut buf, self.gamma_);
        write_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SVC)?;
        let c = r.read_f64()?;
        let kernel = r.read_str()?;
        let degree = r.read_usize()?;
        let gamma_mode = r.read_str()?;
        let gamma = r.read_opt_f64()?;
        let coef0 = r.read_f64()?;
        let probability = r.read_bool()?;
        let tol = r.read_f64()?;
        let max_iter = r.read_opt_usize()?;
        let random_state = r.read_opt_u64()?;
        let stored_dev = r.read_opt_str()?;
        let classes_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let support_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let support_vectors_ = r.read_opt_matrix()?;
        let n_support_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let dual_coef_ = r.read_opt_matrix()?;
        let intercept_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let prob_a_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let prob_b_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let gamma_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_usize()?;
        let n_samples_fit_ = r.read_usize()?;
        let (_, resolved_dev) = resolve_loaded_device(stored_dev, true, device)?;
        Ok(Self {
            c,
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            probability,
            tol,
            max_iter,
            random_state,
            device_: resolved_dev,
            classes_,
            support_,
            support_vectors_,
            n_support_,
            dual_coef_,
            intercept_,
            prob_a_,
            prob_b_,
            gamma_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// 4. SVR
// ============================================================================

#[derive(Clone, Debug)]
pub struct SVR {
    pub kernel: String,
    pub degree: usize,
    pub gamma_mode: String,
    pub gamma: Option<f64>,
    pub coef0: f64,
    pub tol: f64,
    pub c: f64,
    pub epsilon: f64,
    pub max_iter: Option<usize>,
    pub device_: Option<String>,
    pub support_: Option<Vec<usize>>,
    pub support_vectors_: Option<Matrix>,
    pub dual_coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub gamma_: Option<f64>,
    pub n_features_in_: usize,
    pub n_samples_fit_: usize,
}

impl SVR {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        tol: f64,
        c: f64,
        epsilon: f64,
        max_iter: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let _ = KernelKind::parse(&kernel)?;
        if let Some(g) = gamma
            && (!g.is_finite() || g <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        validate_coef0(coef0)?;
        validate_tol(tol)?;
        validate_c(c)?;
        validate_epsilon(epsilon)?;
        let _ = validate_max_iter(max_iter)?;
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        Ok(Self {
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            tol,
            c,
            epsilon,
            max_iter,
            device_: device,
            support_: None,
            support_vectors_: None,
            dual_coef_: None,
            intercept_: None,
            gamma_: None,
            n_features_in_: 0,
            n_samples_fit_: 0,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        self.device_ = device;
        Ok(())
    }

    fn device_type(&self) -> OchreResult<DeviceType> {
        parse_device(self.device_.as_deref())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        validate_finite_y(y, x.rows)?;
        let k_kind = KernelKind::parse(&self.kernel)?;
        if k_kind == KernelKind::Precomputed && x.rows != x.cols {
            return Err(OchreError::NonSquareMatrix);
        }

        let n = x.rows;
        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(x.rows).max(x.data.len()));
        let gamma_val = resolve_gamma(&backend, x, k_kind, &self.gamma_mode, self.gamma)?;
        let k_mat =
            compute_kernel_matrix(&backend, x, x, k_kind, gamma_val, self.degree, self.coef0)?;
        let max_it = validate_max_iter(self.max_iter)?;

        // 2N dual SMO for epsilon-SVR
        let m = 2 * n;
        let mut alpha = vec![0.0; m];
        let mut grad = Vec::with_capacity(m);
        for &yi in y {
            grad.push(self.epsilon - yi);
        }
        for &yi in y {
            grad.push(self.epsilon + yi);
        }
        let sign = |u: usize| -> f64 { if u < n { 1.0 } else { -1.0 } };

        for _iter in 0..max_it {
            let mut g_max = f64::NEG_INFINITY;
            let mut i_opt = usize::MAX;
            for t in 0..m {
                let s = sign(t);
                let in_up = (s > 0.0 && alpha[t] < self.c) || (s < 0.0 && alpha[t] > 0.0);
                if in_up {
                    let val = -s * grad[t];
                    if val >= g_max {
                        g_max = val;
                        i_opt = t;
                    }
                }
            }
            if i_opt == usize::MAX {
                break;
            }

            let mut g_min = f64::INFINITY;
            let mut obj_diff_min = f64::INFINITY;
            let mut j_opt = usize::MAX;
            let si = i_opt % n;
            let k_ii = k_mat.data[si * n + si];

            for t in 0..m {
                let s = sign(t);
                let in_low = (s > 0.0 && alpha[t] > 0.0) || (s < 0.0 && alpha[t] < self.c);
                if in_low {
                    let neg_sg = -s * grad[t];
                    if neg_sg <= g_min {
                        g_min = neg_sg;
                    }
                    let b = g_max - neg_sg;
                    if b > 0.0 {
                        let st = t % n;
                        let k_tt = k_mat.data[st * n + st];
                        let k_it = k_mat.data[si * n + st];
                        let mut a = k_ii + k_tt - 2.0 * k_it;
                        if a <= 0.0 {
                            a = 1e-12;
                        }
                        let obj_diff = -(b * b) / a;
                        if obj_diff <= obj_diff_min {
                            obj_diff_min = obj_diff;
                            j_opt = t;
                        }
                    }
                }
            }

            if g_max - g_min < self.tol || j_opt == usize::MAX {
                break;
            }

            let sj = j_opt % n;
            let k_jj = k_mat.data[sj * n + sj];
            let k_ij = k_mat.data[si * n + sj];
            let mut a = k_ii + k_jj - 2.0 * k_ij;
            if a <= 0.0 {
                a = 1e-12;
            }
            let s_i = sign(i_opt);
            let s_j = sign(j_opt);
            let b = (-s_i * grad[i_opt]) - (-s_j * grad[j_opt]);

            let old_ai = alpha[i_opt];
            let old_aj = alpha[j_opt];
            alpha[i_opt] += s_i * b / a;
            alpha[j_opt] -= s_j * b / a;

            let sum = s_i * old_ai + s_j * old_aj;
            alpha[i_opt] = alpha[i_opt].clamp(0.0, self.c);
            alpha[j_opt] = s_j * (sum - s_i * alpha[i_opt]);
            alpha[j_opt] = alpha[j_opt].clamp(0.0, self.c);
            alpha[i_opt] = s_i * (sum - s_j * alpha[j_opt]);

            let s_di = s_i * (alpha[i_opt] - old_ai);
            let s_dj = s_j * (alpha[j_opt] - old_aj);
            let row_i = &k_mat.data[si * n..(si + 1) * n];
            let row_j = &k_mat.data[sj * n..(sj + 1) * n];
            for t in 0..n {
                let upd = row_i[t] * s_di + row_j[t] * s_dj;
                grad[t] += upd;
                grad[t + n] -= upd;
            }
        }

        // Compute intercept
        let mut sum_free = 0.0;
        let mut nr_free = 0usize;
        let mut ub_val = f64::INFINITY;
        let mut lb_val = f64::NEG_INFINITY;
        for t in 0..m {
            let s = sign(t);
            let sg = s * grad[t];
            if alpha[t] >= self.c {
                if s < 0.0 {
                    ub_val = ub_val.min(sg);
                } else {
                    lb_val = lb_val.max(sg);
                }
            } else if alpha[t] <= 0.0 {
                if s > 0.0 {
                    ub_val = ub_val.min(sg);
                } else {
                    lb_val = lb_val.max(sg);
                }
            } else {
                nr_free += 1;
                sum_free += sg;
            }
        }
        let intercept = if nr_free > 0 {
            -sum_free / (nr_free as f64)
        } else {
            -0.5 * (ub_val + lb_val)
        };

        let mut support_indices = Vec::new();
        let mut sv_coefs = Vec::new();
        for i in 0..n {
            let beta = alpha[i] - alpha[i + n];
            if beta.abs() > 1e-12 {
                support_indices.push(i);
                sv_coefs.push(beta);
            }
        }

        let total_sv = support_indices.len();
        self.support_ = Some(support_indices.clone());
        self.support_vectors_ = if k_kind == KernelKind::Precomputed {
            None
        } else {
            Some(gather_rows(x, &support_indices))
        };
        self.dual_coef_ = Some(Matrix::from_vec(1, total_sv, sv_coefs)?);
        self.intercept_ = Some(vec![intercept]);
        self.gamma_ = Some(gamma_val);
        self.n_features_in_ = x.cols;
        self.n_samples_fit_ = n;
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let support = self.support_.as_ref().ok_or(OchreError::NotFitted)?;
        let dual_coef = self.dual_coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?[0];
        validate_finite_matrix(x)?;

        let k_kind = KernelKind::parse(&self.kernel)?;
        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(support.len()));
        let gamma_val = self.gamma_.unwrap_or(1.0);

        let k_sv = if k_kind == KernelKind::Precomputed {
            if x.cols != self.n_samples_fit_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_samples_fit_,
                });
            }
            slice_precomputed_cols(x, support)
        } else {
            let sv = self
                .support_vectors_
                .as_ref()
                .ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            compute_kernel_matrix(&backend, x, sv, k_kind, gamma_val, self.degree, self.coef0)?
        };

        let total_sv = support.len();
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let k_row = &k_sv.data[r * total_sv..(r + 1) * total_sv];
            let mut sum = intercept;
            for (c, &kv) in k_row.iter().enumerate().take(total_sv) {
                sum += dual_coef.data[c] * kv;
            }
            preds.push(sum);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        if y.len() != x.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SVR);
        write_str(&mut buf, &self.kernel);
        write_usize(&mut buf, self.degree);
        write_str(&mut buf, &self.gamma_mode);
        write_opt_f64(&mut buf, self.gamma);
        write_f64(&mut buf, self.coef0);
        write_f64(&mut buf, self.tol);
        write_f64(&mut buf, self.c);
        write_f64(&mut buf, self.epsilon);
        write_opt_usize(&mut buf, self.max_iter);
        write_opt_str(&mut buf, &self.device_);
        match &self.support_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_matrix(&mut buf, &self.support_vectors_);
        write_opt_matrix(&mut buf, &self.dual_coef_);
        match &self.intercept_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_f64(&mut buf, self.gamma_);
        write_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SVR)?;
        let kernel = r.read_str()?;
        let degree = r.read_usize()?;
        let gamma_mode = r.read_str()?;
        let gamma = r.read_opt_f64()?;
        let coef0 = r.read_f64()?;
        let tol = r.read_f64()?;
        let c = r.read_f64()?;
        let epsilon = r.read_f64()?;
        let max_iter = r.read_opt_usize()?;
        let stored_dev = r.read_opt_str()?;
        let support_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let support_vectors_ = r.read_opt_matrix()?;
        let dual_coef_ = r.read_opt_matrix()?;
        let intercept_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let gamma_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_usize()?;
        let n_samples_fit_ = r.read_usize()?;
        let (_, resolved_dev) = resolve_loaded_device(stored_dev, true, device)?;
        Ok(Self {
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            tol,
            c,
            epsilon,
            max_iter,
            device_: resolved_dev,
            support_,
            support_vectors_,
            dual_coef_,
            intercept_,
            gamma_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// 5. OneClassSVM
// ============================================================================

#[derive(Clone, Debug)]
pub struct OneClassSVM {
    pub kernel: String,
    pub degree: usize,
    pub gamma_mode: String,
    pub gamma: Option<f64>,
    pub coef0: f64,
    pub tol: f64,
    pub nu: f64,
    pub max_iter: Option<usize>,
    pub device_: Option<String>,
    pub support_: Option<Vec<usize>>,
    pub support_vectors_: Option<Matrix>,
    pub dual_coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub offset_: f64,
    pub gamma_: Option<f64>,
    pub n_features_in_: usize,
    pub n_samples_fit_: usize,
}

impl OneClassSVM {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        tol: f64,
        nu: f64,
        max_iter: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let _ = KernelKind::parse(&kernel)?;
        if let Some(g) = gamma
            && (!g.is_finite() || g <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "gamma ({g}) must be a strictly positive finite number (・`ω´・)"
            )));
        }
        validate_coef0(coef0)?;
        validate_tol(tol)?;
        validate_nu(nu)?;
        let _ = validate_max_iter(max_iter)?;
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        Ok(Self {
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            tol,
            nu,
            max_iter,
            device_: device,
            support_: None,
            support_vectors_: None,
            dual_coef_: None,
            intercept_: None,
            offset_: 0.0,
            gamma_: None,
            n_features_in_: 0,
            n_samples_fit_: 0,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        if let Some(ref d) = device {
            let _ = parse_device(Some(d.as_str()))?;
        }
        self.device_ = device;
        Ok(())
    }

    fn device_type(&self) -> OchreResult<DeviceType> {
        parse_device(self.device_.as_deref())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        let k_kind = KernelKind::parse(&self.kernel)?;
        if k_kind == KernelKind::Precomputed && x.rows != x.cols {
            return Err(OchreError::NonSquareMatrix);
        }

        let n = x.rows;
        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(x.rows).max(x.data.len()));
        let gamma_val = resolve_gamma(&backend, x, k_kind, &self.gamma_mode, self.gamma)?;
        let k_mat =
            compute_kernel_matrix(&backend, x, x, k_kind, gamma_val, self.degree, self.coef0)?;
        let max_it = validate_max_iter(self.max_iter)?;

        // Schölkopf One-Class SVM: 0 <= alpha_i <= 1/(nu*n), sum(alpha) = 1
        let upper = 1.0 / (self.nu * (n as f64));
        let mut alpha = vec![0.0; n];
        let mut rem: f64 = 1.0;
        for a in &mut alpha {
            if rem <= 0.0 {
                break;
            }
            let take = rem.min(upper);
            *a = take;
            rem -= take;
        }

        let mut grad = vec![0.0; n];
        for (i, g) in grad.iter_mut().enumerate().take(n) {
            let mut s = 0.0;
            for (j, &aj) in alpha.iter().enumerate().take(n) {
                if aj > 0.0 {
                    s += k_mat.data[i * n + j] * aj;
                }
            }
            *g = s;
        }

        for _iter in 0..max_it {
            let mut g_max = f64::NEG_INFINITY;
            let mut i_opt = usize::MAX;
            for t in 0..n {
                if alpha[t] < upper {
                    let val = -grad[t];
                    if val >= g_max {
                        g_max = val;
                        i_opt = t;
                    }
                }
            }
            if i_opt == usize::MAX {
                break;
            }

            let mut g_min = f64::INFINITY;
            let mut obj_diff_min = f64::INFINITY;
            let mut j_opt = usize::MAX;
            let k_ii = k_mat.data[i_opt * n + i_opt];

            for t in 0..n {
                if alpha[t] > 0.0 {
                    let neg_g = -grad[t];
                    if neg_g <= g_min {
                        g_min = neg_g;
                    }
                    let b = g_max - neg_g;
                    if b > 0.0 {
                        let k_tt = k_mat.data[t * n + t];
                        let k_it = k_mat.data[i_opt * n + t];
                        let mut a = k_ii + k_tt - 2.0 * k_it;
                        if a <= 0.0 {
                            a = 1e-12;
                        }
                        let obj_diff = -(b * b) / a;
                        if obj_diff <= obj_diff_min {
                            obj_diff_min = obj_diff;
                            j_opt = t;
                        }
                    }
                }
            }

            if g_max - g_min < self.tol || j_opt == usize::MAX {
                break;
            }

            let k_jj = k_mat.data[j_opt * n + j_opt];
            let k_ij = k_mat.data[i_opt * n + j_opt];
            let mut a = k_ii + k_jj - 2.0 * k_ij;
            if a <= 0.0 {
                a = 1e-12;
            }
            let b = grad[j_opt] - grad[i_opt];

            let old_ai = alpha[i_opt];
            let old_aj = alpha[j_opt];
            alpha[i_opt] += b / a;
            alpha[j_opt] -= b / a;

            let sum = old_ai + old_aj;
            alpha[i_opt] = alpha[i_opt].clamp(0.0, upper);
            alpha[j_opt] = (sum - alpha[i_opt]).clamp(0.0, upper);
            alpha[i_opt] = sum - alpha[j_opt];

            let di = alpha[i_opt] - old_ai;
            let dj = alpha[j_opt] - old_aj;
            let row_i = &k_mat.data[i_opt * n..(i_opt + 1) * n];
            let row_j = &k_mat.data[j_opt * n..(j_opt + 1) * n];
            for t in 0..n {
                grad[t] += row_i[t] * di + row_j[t] * dj;
            }
        }

        let mut sum_free = 0.0;
        let mut nr_free = 0usize;
        let mut ub_val = f64::INFINITY;
        let mut lb_val = f64::NEG_INFINITY;
        for t in 0..n {
            if alpha[t] >= upper {
                lb_val = lb_val.max(grad[t]);
            } else if alpha[t] <= 0.0 {
                ub_val = ub_val.min(grad[t]);
            } else {
                nr_free += 1;
                sum_free += grad[t];
            }
        }
        let rho = if nr_free > 0 {
            sum_free / (nr_free as f64)
        } else {
            0.5 * (ub_val + lb_val)
        };

        let nu_n = self.nu * (n as f64);
        let rho_scaled = rho * nu_n;

        let mut support_indices = Vec::new();
        let mut sv_coefs = Vec::new();
        for (i, &a) in alpha.iter().enumerate() {
            if a > 1e-12 {
                support_indices.push(i);
                sv_coefs.push(a * nu_n);
            }
        }

        let total_sv = support_indices.len();
        self.support_ = Some(support_indices.clone());
        self.support_vectors_ = if k_kind == KernelKind::Precomputed {
            None
        } else {
            Some(gather_rows(x, &support_indices))
        };
        self.dual_coef_ = Some(Matrix::from_vec(1, total_sv, sv_coefs)?);
        self.intercept_ = Some(vec![-rho_scaled]);
        self.offset_ = -rho_scaled;
        self.gamma_ = Some(gamma_val);
        self.n_features_in_ = x.cols;
        self.n_samples_fit_ = n;
        Ok(())
    }

    fn eval_scores_with_offset(&self, x: &Matrix, offset: f64) -> OchreResult<Vec<f64>> {
        let support = self.support_.as_ref().ok_or(OchreError::NotFitted)?;
        let dual_coef = self.dual_coef_.as_ref().ok_or(OchreError::NotFitted)?;
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }

        let k_kind = KernelKind::parse(&self.kernel)?;
        let gamma_val = self.gamma_.unwrap_or(1.0);

        if k_kind == KernelKind::Rbf
            && x.rows.saturating_mul(support.len()) < FUSED_PREDICT_MAX_ELEMS
        {
            let sv = self
                .support_vectors_
                .as_ref()
                .ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            return fused_rbf_1d_cpu(x, sv, &dual_coef.data, gamma_val, offset);
        }

        validate_finite_matrix(x)?;
        let dev = self.device_type()?;
        let backend = effective_backend(&dev, x.rows.saturating_mul(support.len()));

        let k_sv = if k_kind == KernelKind::Precomputed {
            if x.cols != self.n_samples_fit_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_samples_fit_,
                });
            }
            slice_precomputed_cols(x, support)
        } else {
            let sv = self
                .support_vectors_
                .as_ref()
                .ok_or(OchreError::NotFitted)?;
            if x.cols != self.n_features_in_ {
                return Err(OchreError::FeatureCountMismatch {
                    features: x.cols,
                    expected: self.n_features_in_,
                });
            }
            compute_kernel_matrix(&backend, x, sv, k_kind, gamma_val, self.degree, self.coef0)?
        };

        let total_sv = support.len();
        let mut scores = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let k_row = &k_sv.data[r * total_sv..(r + 1) * total_sv];
            let mut sum = offset;
            for (c, &kv) in k_row.iter().enumerate().take(total_sv) {
                sum += dual_coef.data[c] * kv;
            }
            scores.push(sum);
        }
        Ok(scores)
    }

    pub fn score_samples(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.eval_scores_with_offset(x, 0.0)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.eval_scores_with_offset(x, self.offset_)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let mut dec = self.decision_function(x)?;
        for d in &mut dec {
            *d = if *d >= 0.0 { 1.0 } else { -1.0 };
        }
        Ok(dec)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ONE_CLASS_SVM);
        write_str(&mut buf, &self.kernel);
        write_usize(&mut buf, self.degree);
        write_str(&mut buf, &self.gamma_mode);
        write_opt_f64(&mut buf, self.gamma);
        write_f64(&mut buf, self.coef0);
        write_f64(&mut buf, self.tol);
        write_f64(&mut buf, self.nu);
        write_opt_usize(&mut buf, self.max_iter);
        write_opt_str(&mut buf, &self.device_);
        match &self.support_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_matrix(&mut buf, &self.support_vectors_);
        write_opt_matrix(&mut buf, &self.dual_coef_);
        match &self.intercept_ {
            Some(v) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, v);
            }
            None => write_bool(&mut buf, false),
        }
        write_f64(&mut buf, self.offset_);
        write_opt_f64(&mut buf, self.gamma_);
        write_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ONE_CLASS_SVM)?;
        let kernel = r.read_str()?;
        let degree = r.read_usize()?;
        let gamma_mode = r.read_str()?;
        let gamma = r.read_opt_f64()?;
        let coef0 = r.read_f64()?;
        let tol = r.read_f64()?;
        let nu = r.read_f64()?;
        let max_iter = r.read_opt_usize()?;
        let stored_dev = r.read_opt_str()?;
        let support_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let support_vectors_ = r.read_opt_matrix()?;
        let dual_coef_ = r.read_opt_matrix()?;
        let intercept_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let offset_ = r.read_f64()?;
        let gamma_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_usize()?;
        let n_samples_fit_ = r.read_usize()?;
        let (_, resolved_dev) = resolve_loaded_device(stored_dev, true, device)?;
        Ok(Self {
            kernel,
            degree,
            gamma_mode,
            gamma,
            coef0,
            tol,
            nu,
            max_iter,
            device_: resolved_dev,
            support_,
            support_vectors_,
            dual_coef_,
            intercept_,
            offset_,
            gamma_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// PyO3 Bindings
// ============================================================================

#[pyclass(name = "KernelRidge")]
pub struct PyKernelRidge {
    pub inner: KernelRidge,
}

pymethods_with_dit!(PyKernelRidge, KernelRidge, {
    #[new]
    #[pyo3(signature = (alpha=1.0, kernel="linear".to_string(), gamma=None, degree=3, coef0=1.0, device=None))]
    pub fn new(
        alpha: f64,
        kernel: String,
        gamma: Option<f64>,
        degree: usize,
        coef0: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KernelRidge::new(alpha, kernel, gamma, degree, coef0, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    pub fn fit(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        let sw = match sample_weight {
            Some(ref buf) => Some(buffer_to_f64_1d(py, buf)?),
            None => None,
        };
        self.inner.fit(&x_mat, &y_mat, sw.as_deref())?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        matrix_to_pybytes(py, &preds)
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    pub fn score(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        let sw = match sample_weight {
            Some(ref buf) => Some(buffer_to_f64_1d(py, buf)?),
            None => None,
        };
        Ok(self.inner.score(&x_mat, &y_mat, sw.as_deref())?)
    }

    #[getter]
    pub fn dual_coef_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.dual_coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn x_fit_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.x_fit_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> usize {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "KernelPCA")]
pub struct PyKernelPCA {
    pub inner: KernelPCA,
}

pymethods_with_dit!(PyKernelPCA, KernelPCA, {
    #[new]
    #[pyo3(signature = (n_components=None, kernel="linear".to_string(), gamma=None, degree=3, coef0=1.0, fit_inverse_transform=false, alpha=1.0, device=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: Option<usize>,
        kernel: String,
        gamma: Option<f64>,
        degree: usize,
        coef0: f64,
        fit_inverse_transform: bool,
        alpha: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KernelPCA::new(
                n_components,
                kernel,
                gamma,
                degree,
                coef0,
                fit_inverse_transform,
                alpha,
                device,
            )?,
        })
    }

    pub fn fit(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&x_mat)?;
        Ok(())
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let z_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&z_mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn eigenvalues_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.eigenvalues_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn eigenvectors_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.eigenvectors_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> usize {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "SVC")]
pub struct PySVC {
    pub inner: SVC,
}

pymethods_with_dit!(PySVC, SVC, {
    #[new]
    #[pyo3(signature = (c=1.0, kernel="rbf".to_string(), degree=3, gamma_mode="scale".to_string(), gamma=None, coef0=0.0, probability=false, tol=1e-3, max_iter=None, random_state=None, device=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: f64,
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        probability: bool,
        tol: f64,
        max_iter: Option<usize>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SVC::new(
                c,
                kernel,
                degree,
                gamma_mode,
                gamma,
                coef0,
                probability,
                tol,
                max_iter,
                random_state,
                device,
            )?,
        })
    }

    pub fn fit(&mut self, py: Python<'_>, x: PyBuffer<f64>, y: PyBuffer<f64>) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
        Ok(())
    }

    pub fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let dec = self.inner.decision_function(&x_mat)?;
        matrix_to_pybytes(py, &dec)
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyBuffer<f64>) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &probs)
    }

    pub fn score(&self, py: Python<'_>, x: PyBuffer<f64>, y: PyBuffer<f64>) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn support_(&self) -> Option<Vec<usize>> {
        self.inner.support_.clone()
    }

    #[getter]
    pub fn support_vectors_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.support_vectors_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_support_(&self) -> Option<Vec<usize>> {
        self.inner.n_support_.clone()
    }

    #[getter]
    pub fn dual_coef_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.dual_coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.intercept_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.classes_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> usize {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "SVR")]
pub struct PySVR {
    pub inner: SVR,
}

pymethods_with_dit!(PySVR, SVR, {
    #[new]
    #[pyo3(signature = (kernel="rbf".to_string(), degree=3, gamma_mode="scale".to_string(), gamma=None, coef0=0.0, tol=1e-3, c=1.0, epsilon=0.1, max_iter=None, device=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        tol: f64,
        c: f64,
        epsilon: f64,
        max_iter: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SVR::new(
                kernel, degree, gamma_mode, gamma, coef0, tol, c, epsilon, max_iter, device,
            )?,
        })
    }

    pub fn fit(&mut self, py: Python<'_>, x: PyBuffer<f64>, y: PyBuffer<f64>) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
        Ok(())
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyBuffer<f64>) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score(&self, py: Python<'_>, x: PyBuffer<f64>, y: PyBuffer<f64>) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn support_(&self) -> Option<Vec<usize>> {
        self.inner.support_.clone()
    }

    #[getter]
    pub fn support_vectors_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.support_vectors_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn dual_coef_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.dual_coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.intercept_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> usize {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "OneClassSVM")]
pub struct PyOneClassSVM {
    pub inner: OneClassSVM,
}

pymethods_with_dit!(PyOneClassSVM, OneClassSVM, {
    #[new]
    #[pyo3(signature = (kernel="rbf".to_string(), degree=3, gamma_mode="scale".to_string(), gamma=None, coef0=0.0, tol=1e-3, nu=0.5, max_iter=None, device=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: String,
        degree: usize,
        gamma_mode: String,
        gamma: Option<f64>,
        coef0: f64,
        tol: f64,
        nu: f64,
        max_iter: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OneClassSVM::new(
                kernel, degree, gamma_mode, gamma, coef0, tol, nu, max_iter, device,
            )?,
        })
    }

    pub fn fit(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&x_mat)?;
        Ok(())
    }

    pub fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let scores = self.inner.score_samples(&x_mat)?;
        f64_slice_to_pybytes(py, &scores)
    }

    pub fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let dec = self.inner.decision_function(&x_mat)?;
        f64_slice_to_pybytes(py, &dec)
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyBuffer<f64>) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    #[getter]
    pub fn support_(&self) -> Option<Vec<usize>> {
        self.inner.support_.clone()
    }

    #[getter]
    pub fn support_vectors_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.support_vectors_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn dual_coef_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.dual_coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.intercept_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn offset_(&self) -> f64 {
        self.inner.offset_
    }

    #[getter]
    pub fn n_features_in_(&self) -> usize {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySVC>()?;
    m.add_class::<PySVR>()?;
    m.add_class::<PyOneClassSVM>()?;
    m.add_class::<PyKernelRidge>()?;
    m.add_class::<PyKernelPCA>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_ridge_linear_and_rbf() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        let y = Matrix::from_2d(&[vec![0.0], vec![1.0], vec![1.0], vec![2.0]]).unwrap();
        let mut kr = KernelRidge::new(1e-3, "linear".to_string(), None, 3, 1.0, None).unwrap();
        kr.fit(&x, &y, None).unwrap();
        let r2 = kr.score(&x, &y, None).unwrap();
        assert!(r2 > 0.99);

        let bytes = kr.to_bytes();
        let kr_loaded = KernelRidge::from_bytes(&bytes, Some("cpu")).unwrap();
        let r2_loaded = kr_loaded.score(&x, &y, None).unwrap();
        assert!((r2 - r2_loaded).abs() < 1e-12);
    }

    #[test]
    fn test_kernel_pca_and_inverse() {
        let x = Matrix::from_2d(&[
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![4.0, 8.0, 12.0],
        ])
        .unwrap();
        let mut kpca = KernelPCA::new(
            Some(1),
            "linear".to_string(),
            None,
            3,
            1.0,
            true,
            1e-3,
            None,
        )
        .unwrap();
        let z = kpca.fit_transform(&x).unwrap();
        assert_eq!(z.rows, 4);
        assert_eq!(z.cols, 1);
        let x_rec = kpca.inverse_transform(&z).unwrap();
        assert_eq!(x_rec.rows, 4);
        assert_eq!(x_rec.cols, 3);
    }

    #[test]
    fn test_kernel_pca_top_k_fast_path() {
        let n = 80usize;
        let mut x_data = vec![0.0; n * 2];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let r = if i % 2 == 0 { 1.0 } else { 0.3 };
            x_data[i * 2] = r * theta.cos();
            x_data[i * 2 + 1] = r * theta.sin();
        }
        let x = Matrix::from_vec(n, 2, x_data).unwrap();
        let mut kpca = KernelPCA::new(
            Some(4),
            "rbf".to_string(),
            Some(5.0),
            3,
            1.0,
            true,
            0.1,
            Some("cpu".to_string()),
        )
        .unwrap();
        let z = kpca.fit_transform(&x).unwrap();
        assert_eq!(z.rows, n);
        assert_eq!(z.cols, 4);
        let eigs = kpca.eigenvalues_.as_ref().unwrap();
        assert_eq!(eigs.len(), 4);
        assert!(eigs[0] >= eigs[1] && eigs[1] >= eigs[2] && eigs[2] >= eigs[3]);
        let x_rec = kpca.inverse_transform(&z).unwrap();
        assert_eq!(x_rec.rows, n);
        assert_eq!(x_rec.cols, 2);
    }

    #[test]
    fn test_svc_binary_and_multiclass() {
        let x = Matrix::from_2d(&[
            vec![-2.0, -2.0],
            vec![-1.5, -1.0],
            vec![1.5, 1.0],
            vec![2.0, 2.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut svc = SVC::new(
            1.0,
            "rbf".to_string(),
            3,
            "scale".to_string(),
            None,
            0.0,
            true,
            1e-3,
            None,
            Some(42),
            None,
        )
        .unwrap();
        svc.fit(&x, &y).unwrap();
        assert_eq!(svc.predict(&x).unwrap(), y);
        let probs = svc.predict_proba(&x).unwrap();
        assert_eq!(probs.rows, 4);
        assert_eq!(probs.cols, 2);
    }

    #[test]
    fn test_svr_and_one_class_svm() {
        let x = Matrix::from_2d(&[vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 2.0, 4.0, 6.0];
        let mut svr = SVR::new(
            "linear".to_string(),
            3,
            "scale".to_string(),
            None,
            0.0,
            1e-3,
            10.0,
            0.1,
            None,
            None,
        )
        .unwrap();
        svr.fit(&x, &y).unwrap();
        assert!(svr.score(&x, &y).unwrap() > 0.95);

        let mut ocsvm = OneClassSVM::new(
            "rbf".to_string(),
            3,
            "scale".to_string(),
            None,
            0.0,
            1e-3,
            0.5,
            None,
            None,
        )
        .unwrap();
        ocsvm.fit(&x).unwrap();
        let preds = ocsvm.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_fused_rbf_kernel_and_krr_280_parity() {
        for cols in [8usize, 24usize] {
            let n = 280usize;
            let mut x_data = vec![0.0; n * cols];
            let mut y_data = vec![0.0; n];
            for i in 0..n {
                let mut target = 0.0;
                for c in 0..cols {
                    let v = (((i * 11 + c * 5) % 43) as f64) * 0.08 - 1.5;
                    x_data[i * cols + c] = v;
                    target += (v * 0.7).sin();
                }
                y_data[i] = target;
            }
            let x = Matrix::from_vec(n, cols, x_data).unwrap();
            let y = Matrix::from_vec(n, 1, y_data).unwrap();

            // Verify fused symmetric RBF matches pairwise_l2_sq + exp(-gamma * d2)
            let gamma = 0.15;
            let k_fused = fused_rbf_symmetric_cpu(&x, gamma);
            let d2_ref = Backend::Cpu.pairwise_l2_sq(&x, &x).unwrap();
            for idx in 0..k_fused.data.len() {
                let expected = (-gamma * d2_ref.data[idx].max(0.0)).exp();
                assert!((k_fused.data[idx] - expected).abs() < 1e-12);
            }

            // Verify KernelRidge fit & predict with auto device (stays on CPU < 524_288)
            let mut krr =
                KernelRidge::new(0.1, "rbf".to_string(), Some(gamma), 3, 1.0, None).unwrap();
            krr.fit(&x, &y, None).unwrap();
            let r2 = krr.score(&x, &y, None).unwrap();
            assert!(r2 > 0.95, "KRR 280x{cols} R2 = {r2}");

            // Verify fused predict parity against explicit kernel matrix * dual_coef
            let x_test = Matrix::from_vec(70, cols, x.data[..70 * cols].to_vec()).unwrap();
            let pred_fused = krr.predict(&x_test).unwrap();
            let k_ref = fused_rbf_cross_cpu(&x_test, krr.x_fit_.as_ref().unwrap(), gamma);
            let pred_ref = k_ref.matmul(krr.dual_coef_.as_ref().unwrap()).unwrap();
            for idx in 0..pred_fused.data.len() {
                assert!(
                    (pred_fused.data[idx] - pred_ref.data[idx]).abs() < 1e-11,
                    "KRR fused predict mismatch at {idx}: {} vs {}",
                    pred_fused.data[idx],
                    pred_ref.data[idx]
                );
            }

            // Verify OneClassSVM fused score_samples / decision_function / predict parity
            let mut ocsvm = OneClassSVM::new(
                "rbf".to_string(),
                3,
                "scale".to_string(),
                Some(gamma),
                0.0,
                1e-3,
                0.2,
                None,
                None,
            )
            .unwrap();
            ocsvm.fit(&x).unwrap();
            let scores = ocsvm.score_samples(&x_test).unwrap();
            let dec = ocsvm.decision_function(&x_test).unwrap();
            let preds = ocsvm.predict(&x_test).unwrap();
            let sv = ocsvm.support_vectors_.as_ref().unwrap();
            let dc = ocsvm.dual_coef_.as_ref().unwrap();
            let k_oc = fused_rbf_cross_cpu(&x_test, sv, gamma);
            for r in 0..x_test.rows {
                let mut ref_s = 0.0;
                for c in 0..sv.rows {
                    ref_s += k_oc[(r, c)] * dc[(0, c)];
                }
                assert!((scores[r] - ref_s).abs() < 1e-11);
                assert!((dec[r] - (ref_s + ocsvm.offset_)).abs() < 1e-11);
                assert_eq!(preds[r], if dec[r] >= 0.0 { 1.0 } else { -1.0 });
            }

            // Verify NaN rejection in fused predict paths
            let mut bad_x = x_test.clone();
            bad_x.data[cols + 1] = f64::NAN;
            assert!(krr.predict(&bad_x).is_err());
            assert!(ocsvm.decision_function(&bad_x).is_err());
        }
    }
}
