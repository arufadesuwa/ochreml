// OchreML Dimensionality Reduction & Matrix Decomposition Engine (*≧ω≦*)
// Implements PCA and TruncatedSVD accelerated by Backend Primitives
// N1 (col_reduce_stats, col_scale_shift), N5 (gram_svd, small_sym_eig_svd), and E1/E2 (matmul, matmul_at_b).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_opt_f64, write_opt_f64_vec,
    write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::dot_unrolled;
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_PCA: u8 = 43;
pub const TAG_TRUNCATED_SVD: u8 = 44;

const PARALLEL_ROW_THRESHOLD: usize = 512;
const TRANSFORM_GPU_THRESHOLD: usize = 2_000_000;
const PROJECT_CHUNK_ROWS: usize = 64;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < 131_072 {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn effective_transform_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < TRANSFORM_GPU_THRESHOLD {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline(always)]
fn dot4_4rows(a0: &[f64], a1: &[f64], a2: &[f64], a3: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let mut s0 = [0.0f64; 4];
    let mut s1 = [0.0f64; 4];
    let mut s2 = [0.0f64; 4];
    let mut s3 = [0.0f64; 4];
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    let (a0_chunks, a0_rem) = a0.as_chunks::<4>();
    let (a1_chunks, a1_rem) = a1.as_chunks::<4>();
    let (a2_chunks, a2_rem) = a2.as_chunks::<4>();
    let (a3_chunks, a3_rem) = a3.as_chunks::<4>();
    for ((((&bc, &c0), &c1), &c2), &c3) in b_chunks
        .iter()
        .zip(a0_chunks.iter())
        .zip(a1_chunks.iter())
        .zip(a2_chunks.iter())
        .zip(a3_chunks.iter())
    {
        for l in 0..4 {
            let bv = bc[l];
            s0[l] += c0[l] * bv;
            s1[l] += c1[l] * bv;
            s2[l] += c2[l] * bv;
            s3[l] += c3[l] * bv;
        }
    }
    let mut r0 = (s0[0] + s0[1]) + (s0[2] + s0[3]);
    let mut r1 = (s1[0] + s1[1]) + (s1[2] + s1[3]);
    let mut r2 = (s2[0] + s2[1]) + (s2[2] + s2[3]);
    let mut r3 = (s3[0] + s3[1]) + (s3[2] + s3[3]);
    for j in 0..b_rem.len() {
        let bv = b_rem[j];
        r0 += a0_rem[j] * bv;
        r1 += a1_rem[j] * bv;
        r2 += a2_rem[j] * bv;
        r3 += a3_rem[j] * bv;
    }
    (r0, r1, r2, r3)
}

/// Pack groups of 4 component rows `(k x p)` into contiguous `(p x 4)` strips for SIMD FMA.
fn pack_components_4(comp_data: &[f64], k: usize, p: usize) -> Vec<f64> {
    let n_blocks = k / 4;
    let mut packed = vec![0.0; n_blocks * p * 4];
    for blk in 0..n_blocks {
        let c_base = blk * 4;
        let c0 = &comp_data[c_base * p..(c_base + 1) * p];
        let c1 = &comp_data[(c_base + 1) * p..(c_base + 2) * p];
        let c2 = &comp_data[(c_base + 2) * p..(c_base + 3) * p];
        let c3 = &comp_data[(c_base + 3) * p..(c_base + 4) * p];
        let dst = &mut packed[blk * p * 4..(blk + 1) * p * 4];
        for f in 0..p {
            let off = f * 4;
            dst[off] = c0[f];
            dst[off + 1] = c1[f];
            dst[off + 2] = c2[f];
            dst[off + 3] = c3[f];
        }
    }
    packed
}

#[allow(clippy::too_many_arguments)]
fn project_4x4_chunk(
    x_chunk: &[f64],
    n_rows: usize,
    p: usize,
    comp_packed: &[f64],
    comp_data: &[f64],
    k: usize,
    offset: Option<&[f64]>,
    scales: Option<&[f64]>,
    out_chunk: &mut [f64],
) {
    let n_blocks = k / 4;
    let k_rem_start = n_blocks * 4;
    let mut r = 0usize;

    while r + 4 <= n_rows {
        let x0 = &x_chunk[r * p..(r + 1) * p];
        let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
        let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
        let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
        let o0_off = r * k;
        let o1_off = (r + 1) * k;
        let o2_off = (r + 2) * k;
        let o3_off = (r + 3) * k;

        for blk in 0..n_blocks {
            let c = blk * 4;
            let c_blk = &comp_packed[blk * p * 4..(blk + 1) * p * 4];
            let mut acc0 = [0.0f64; 4];
            let mut acc1 = [0.0f64; 4];
            let mut acc2 = [0.0f64; 4];
            let mut acc3 = [0.0f64; 4];
            for (f, cv) in c_blk.as_chunks::<4>().0.iter().enumerate() {
                let xv0 = x0[f];
                let xv1 = x1[f];
                let xv2 = x2[f];
                let xv3 = x3[f];
                for lane in 0..4 {
                    let cval = cv[lane];
                    acc0[lane] += xv0 * cval;
                    acc1[lane] += xv1 * cval;
                    acc2[lane] += xv2 * cval;
                    acc3[lane] += xv3 * cval;
                }
            }
            if let Some(off) = offset {
                let o = &off[c..c + 4];
                for lane in 0..4 {
                    acc0[lane] -= o[lane];
                    acc1[lane] -= o[lane];
                    acc2[lane] -= o[lane];
                    acc3[lane] -= o[lane];
                }
            }
            if let Some(sc) = scales {
                let s = &sc[c..c + 4];
                for lane in 0..4 {
                    acc0[lane] *= s[lane];
                    acc1[lane] *= s[lane];
                    acc2[lane] *= s[lane];
                    acc3[lane] *= s[lane];
                }
            }
            out_chunk[o0_off + c..o0_off + c + 4].copy_from_slice(&acc0);
            out_chunk[o1_off + c..o1_off + c + 4].copy_from_slice(&acc1);
            out_chunk[o2_off + c..o2_off + c + 4].copy_from_slice(&acc2);
            out_chunk[o3_off + c..o3_off + c + 4].copy_from_slice(&acc3);
        }

        for j in k_rem_start..k {
            let comp_row = &comp_data[j * p..(j + 1) * p];
            let (mut d0, mut d1, mut d2, mut d3) = dot4_4rows(x0, x1, x2, x3, comp_row);
            if let Some(off) = offset {
                let o = off[j];
                d0 -= o;
                d1 -= o;
                d2 -= o;
                d3 -= o;
            }
            if let Some(sc) = scales {
                let s = sc[j];
                d0 *= s;
                d1 *= s;
                d2 *= s;
                d3 *= s;
            }
            out_chunk[o0_off + j] = d0;
            out_chunk[o1_off + j] = d1;
            out_chunk[o2_off + j] = d2;
            out_chunk[o3_off + j] = d3;
        }
        r += 4;
    }

    while r < n_rows {
        let x_row = &x_chunk[r * p..(r + 1) * p];
        let dst = &mut out_chunk[r * k..(r + 1) * k];
        for j in 0..k {
            let mut val = dot_unrolled(x_row, &comp_data[j * p..(j + 1) * p]);
            if let Some(off) = offset {
                val -= off[j];
            }
            if let Some(sc) = scales {
                val *= sc[j];
            }
            dst[j] = val;
        }
        r += 1;
    }
}

fn project_cpu_matrix(
    x: &Matrix,
    components: &Matrix,
    offset: Option<&[f64]>,
    scales: Option<&[f64]>,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let k = components.rows;
    let comp_data = &components.data;
    let comp_packed = pack_components_4(comp_data, k, p);
    let mut out_data = vec![0.0; n * k];

    if n >= PARALLEL_ROW_THRESHOLD {
        out_data
            .par_chunks_mut(PROJECT_CHUNK_ROWS * k)
            .zip(x.data.par_chunks(PROJECT_CHUNK_ROWS * p))
            .for_each(|(out_chunk, x_chunk)| {
                let n_rows = out_chunk.len() / k;
                project_4x4_chunk(
                    x_chunk,
                    n_rows,
                    p,
                    &comp_packed,
                    comp_data,
                    k,
                    offset,
                    scales,
                    out_chunk,
                );
            });
    } else {
        project_4x4_chunk(
            &x.data,
            n,
            p,
            &comp_packed,
            comp_data,
            k,
            offset,
            scales,
            &mut out_data,
        );
    }
    Matrix::from_vec(n, k, out_data)
}

// Deterministic SplitMix64 PRNG with Box-Muller Gaussian sampling for randomized SVD
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

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        // Map top 53 bits into (0, 1)
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01().max(1e-15);
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn validate_finite_matrix(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(())
}

/// Apply deterministic SVD sign convention: largest magnitude element in each component row is positive.
fn svd_flip_rows(components: &mut Matrix) {
    let cols = components.cols;
    for r in 0..components.rows {
        let row = &mut components.data[r * cols..(r + 1) * cols];
        let mut max_idx = 0usize;
        let mut max_abs = -1.0f64;
        for (c, &val) in row.iter().enumerate() {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
                max_idx = c;
            }
        }
        if row[max_idx] < 0.0 {
            for val in row.iter_mut() {
                *val = -*val;
            }
        }
    }
}

/// Modified Gram-Schmidt column orthonormalization for `(rows x cols)` matrix.
fn mgs_orthonormalize(mat: &mut Matrix) {
    let rows = mat.rows;
    let cols = mat.cols;
    for j in 0..cols {
        for prev in 0..j {
            let mut dot = 0.0;
            for r in 0..rows {
                dot += mat.data[r * cols + prev] * mat.data[r * cols + j];
            }
            for r in 0..rows {
                mat.data[r * cols + j] -= dot * mat.data[r * cols + prev];
            }
        }
        let mut norm_sq = 0.0;
        for r in 0..rows {
            let v = mat.data[r * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for r in 0..rows {
                mat.data[r * cols + j] *= inv;
            }
        } else {
            for r in 0..rows {
                mat.data[r * cols + j] = 0.0;
            }
        }
    }
}

/// Extract first `k` columns of `v_right` (`p x m`, column-major eigenvectors) as a `(k x p)` row-major Matrix,
/// then apply deterministic SVD sign flip.
fn extract_components_from_v_cols(v_right: &Matrix, k: usize) -> OchreResult<Matrix> {
    let p = v_right.rows;
    let mut data = Vec::with_capacity(k * p);
    for comp_idx in 0..k {
        for feat_idx in 0..p {
            data.push(v_right.data[feat_idx * v_right.cols + comp_idx]);
        }
    }
    let mut components = Matrix::from_vec(k, p, data)?;
    svd_flip_rows(&mut components);
    Ok(components)
}

/// Compute SVD when `p > n` and `n <= 4096` via `n x n` Gram matrix `X_work * X_work^T`.
/// Returns `(singular_values (len n), v_right_cols (p x n))`.
fn dual_gram_svd(
    backend: &Backend<'_>,
    x_work: &Matrix,
    k_needed: usize,
) -> OchreResult<(Vec<f64>, Matrix)> {
    let n = x_work.rows;
    let p = x_work.cols;
    let x_t = x_work.transpose();
    let gram_nn = backend.matmul(x_work, &x_t)?;
    let eig = backend.small_sym_eig_svd(&gram_nn)?;
    let singular_values: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|&lam| lam.max(0.0).sqrt())
        .collect();

    // V_raw = X_work^T * U_n -> shape (p x n)
    let mut v_raw = backend.matmul_at_b(x_work, &eig.eigenvectors)?;
    let s_max = singular_values.first().copied().unwrap_or(0.0);
    let tol = 1e-12 * s_max.max(1e-30);
    let cols_to_norm = k_needed.min(n);
    for (col, &s) in singular_values.iter().enumerate().take(cols_to_norm) {
        if s > tol {
            let inv_s = 1.0 / s;
            for r in 0..p {
                v_raw.data[r * n + col] *= inv_s;
            }
        } else {
            for r in 0..p {
                v_raw.data[r * n + col] = 0.0;
            }
        }
    }
    Ok((singular_values, v_raw))
}

/// Principal Component Analysis (PCA) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct PCA {
    pub n_components: Option<usize>,
    pub explained_variance_ratio_target: Option<f64>,
    pub whiten: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub components_: Option<Matrix>,
    pub explained_variance_: Option<Vec<f64>>,
    pub explained_variance_ratio_: Option<Vec<f64>>,
    pub singular_values_: Option<Vec<f64>>,
    pub mean_: Option<Vec<f64>>,
    pub n_components_: Option<usize>,
    pub n_features_in_: Option<usize>,
    pub n_samples_: Option<usize>,
    pub noise_variance_: Option<f64>,
}

impl PCA {
    pub fn new(
        n_components: Option<usize>,
        explained_variance_ratio_target: Option<f64>,
        whiten: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if let Some(k) = n_components
            && k == 0
        {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if let Some(t) = explained_variance_ratio_target {
            if !t.is_finite() || t <= 0.0 || t >= 1.0 {
                return Err(OchreError::InvalidParameter(format!(
                    "explained_variance_ratio_target ({}) must be in (0.0, 1.0) ( >_< )",
                    t
                )));
            }
            if n_components.is_some() {
                return Err(OchreError::InvalidParameter(
                    "Cannot specify both n_components and explained_variance_ratio_target ( >_< )"
                        .to_string(),
                ));
            }
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            explained_variance_ratio_target,
            whiten,
            device: dev,
            device_: None,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            mean_: None,
            n_components_: None,
            n_features_in_: None,
            n_samples_: None,
            noise_variance_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        if self.n_features_in_.is_some() {
            self.device_ = Some(self.device.to_string());
        }
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        let n = x.rows;
        let p = x.cols;
        let max_components = n.min(p);

        if let Some(k) = self.n_components
            && k > max_components
        {
            return Err(OchreError::InvalidParameter(format!(
                "n_components={} must be between 1 and min(n_samples, n_features)={} ( >_< )",
                k, max_components
            )));
        }

        let backend = effective_backend(&self.device, n * p);
        let (mut singular_values_all, v_right, mean) = if n >= p || n > 4096 {
            backend.gram_svd(x, true)?
        } else {
            let stats = backend.col_reduce_stats(x, None)?;
            let ones = vec![1.0; p];
            let x_c = backend.col_scale_shift(x, &ones, &stats.mean)?;
            let k_needed = self.n_components.unwrap_or(max_components);
            let (s_vals, v_cols) = dual_gram_svd(&backend, &x_c, k_needed)?;
            (s_vals, v_cols, stats.mean)
        };

        let dof = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let mut exp_var_all: Vec<f64> = singular_values_all
            .iter()
            .take(max_components)
            .map(|&s| (s * s) / dof)
            .collect();
        let total_var: f64 = exp_var_all.iter().sum();
        let mut exp_var_ratio_all: Vec<f64> = exp_var_all
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect();

        let k = if let Some(k_fixed) = self.n_components {
            k_fixed
        } else if let Some(target) = self.explained_variance_ratio_target {
            let mut cumsum = 0.0;
            let mut chosen = max_components;
            for (idx, &r) in exp_var_ratio_all.iter().enumerate() {
                cumsum += r;
                if cumsum >= target {
                    chosen = idx + 1;
                    break;
                }
            }
            chosen.max(1)
        } else {
            max_components
        };

        let noise_variance = if k < max_components {
            let tail_sum: f64 = exp_var_all[k..].iter().sum();
            tail_sum / ((max_components - k) as f64)
        } else {
            0.0
        };

        let components = extract_components_from_v_cols(&v_right, k)?;

        exp_var_all.truncate(k);
        exp_var_ratio_all.truncate(k);
        singular_values_all.truncate(k);
        self.components_ = Some(components);
        self.explained_variance_ = Some(exp_var_all);
        self.explained_variance_ratio_ = Some(exp_var_ratio_all);
        self.singular_values_ = Some(singular_values_all);
        self.mean_ = Some(mean);
        self.n_components_ = Some(k);
        self.n_features_in_ = Some(p);
        self.n_samples_ = Some(n);
        self.noise_variance_ = Some(noise_variance);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let components = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let mean = self.mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let exp_var = self
            .explained_variance_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;

        let n = x.rows;
        let p = expected;
        let k = components.rows;

        // Precompute proj_mean = mean @ components^T (length k) to avoid allocating full X_c
        let mut proj_mean = Vec::with_capacity(k);
        for j in 0..k {
            let comp_row = &components.data[j * p..(j + 1) * p];
            proj_mean.push(dot_unrolled(mean, comp_row));
        }

        let inv_stds: Option<Vec<f64>> = if self.whiten {
            Some(
                exp_var
                    .iter()
                    .take(k)
                    .map(|&v| if v > 1e-12 { 1.0 / v.sqrt() } else { 0.0 })
                    .collect(),
            )
        } else {
            None
        };

        let backend = effective_transform_backend(&self.device, n * p);
        if matches!(backend, Backend::Cpu) {
            return project_cpu_matrix(x, components, Some(&proj_mean), inv_stds.as_deref());
        }

        // Large GPU path: compute X @ components^T directly and subtract proj_mean in-place
        let comp_t = components.transpose();
        let mut out = backend.matmul(x, &comp_t)?;
        if let Some(ref scales) = inv_stds {
            for r in 0..n {
                let row = &mut out.data[r * k..(r + 1) * k];
                for j in 0..k {
                    row[j] = (row[j] - proj_mean[j]) * scales[j];
                }
            }
        } else {
            for r in 0..n {
                let row = &mut out.data[r * k..(r + 1) * k];
                for j in 0..k {
                    row[j] -= proj_mean[j];
                }
            }
        }
        Ok(out)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let k = self.n_components_.ok_or(OchreError::NotFitted)?;
        if x.cols != k {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: k,
            });
        }
        let components = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let mean = self.mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let exp_var = self
            .explained_variance_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;

        let backend = effective_backend(&self.device, x.rows * components.cols);
        let x_rec = if self.whiten {
            let stds: Vec<f64> = exp_var.iter().map(|&v| v.max(0.0).sqrt()).collect();
            let mut scaled = x.clone();
            for r in 0..scaled.rows {
                let row = &mut scaled.data[r * k..(r + 1) * k];
                for j in 0..k {
                    row[j] *= stds[j];
                }
            }
            backend.matmul(&scaled, components)?
        } else {
            backend.matmul(x, components)?
        };

        let p = components.cols;
        let ones = vec![1.0; p];
        let neg_mean: Vec<f64> = mean.iter().map(|&m| -m).collect();
        backend.col_scale_shift(&x_rec, &ones, &neg_mean)
    }

    pub fn get_covariance(&self) -> OchreResult<Matrix> {
        let components = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let exp_var = self
            .explained_variance_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let noise_var = self.noise_variance_.ok_or(OchreError::NotFitted)?;
        let k = components.rows;
        let p = components.cols;

        // C = V * diag(exp_var - noise_var) * V^T + noise_var * I
        let mut cov_data = vec![0.0; p * p];
        for r in 0..p {
            cov_data[r * p + r] = noise_var;
            for c in r..p {
                let mut acc = 0.0;
                for (comp, &ev) in exp_var.iter().enumerate().take(k) {
                    let w = ev - noise_var;
                    acc += w * components.data[comp * p + r] * components.data[comp * p + c];
                }
                cov_data[r * p + c] += acc;
                if r != c {
                    cov_data[c * p + r] += acc;
                }
            }
        }
        Matrix::from_vec(p, p, cov_data)
    }

    pub fn score_samples(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        validate_finite_matrix(x)?;
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let cov = self.get_covariance()?;
        let mean = self.mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let p = expected;
        let eig = effective_backend(&self.device, p * p).small_sym_eig_svd(&cov)?;

        let mut log_det = 0.0;
        let mut inv_eigs = vec![0.0; p];
        for (i, dst) in inv_eigs.iter_mut().enumerate().take(p) {
            let lam = eig.eigenvalues[i].max(1e-12);
            log_det += lam.ln();
            *dst = 1.0 / lam;
        }

        let backend = effective_backend(&self.device, x.rows * p);
        let ones = vec![1.0; p];
        let x_c = backend.col_scale_shift(x, &ones, mean)?;
        let proj = backend.matmul(&x_c, &eig.eigenvectors)?;
        let const_term = -0.5 * ((p as f64) * (2.0 * std::f64::consts::PI).ln() + log_det);

        let mut scores = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut quad = 0.0;
            let row = &proj.data[r * p..(r + 1) * p];
            for j in 0..p {
                quad += row[j] * row[j] * inv_eigs[j];
            }
            scores.push(const_term - 0.5 * quad);
        }
        Ok(scores)
    }

    pub fn score(&self, x: &Matrix) -> OchreResult<f64> {
        let scores = self.score_samples(x)?;
        Ok(scores.iter().sum::<f64>() / (scores.len() as f64))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_PCA);
        write_opt_usize(&mut buf, self.n_components);
        write_opt_f64(&mut buf, self.explained_variance_ratio_target);
        write_bool(&mut buf, self.whiten);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.components_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_ratio_);
        write_opt_f64_vec(&mut buf, &self.singular_values_);
        write_opt_f64_vec(&mut buf, &self.mean_);
        write_opt_usize(&mut buf, self.n_components_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_samples_);
        write_opt_f64(&mut buf, self.noise_variance_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_PCA)?;
        let n_components = r.read_opt_usize()?;
        let explained_variance_ratio_target = r.read_opt_f64()?;
        let whiten = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let components_ = r.read_opt_matrix()?;
        let explained_variance_ = r.read_opt_f64_vec()?;
        let explained_variance_ratio_ = r.read_opt_f64_vec()?;
        let singular_values_ = r.read_opt_f64_vec()?;
        let mean_ = r.read_opt_f64_vec()?;
        let n_components_ = r.read_opt_usize()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_samples_ = r.read_opt_usize()?;
        let noise_variance_ = r.read_opt_f64()?;

        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;

        Ok(Self {
            n_components,
            explained_variance_ratio_target,
            whiten,
            device: dev,
            device_,
            components_,
            explained_variance_,
            explained_variance_ratio_,
            singular_values_,
            mean_,
            n_components_,
            n_features_in_,
            n_samples_,
            noise_variance_,
        })
    }
}

#[pyclass(from_py_object, name = "PCA")]
#[derive(Clone)]
pub struct PyPCA {
    inner: PCA,
}

pymethods_with_dit!(PyPCA, PCA, {
    #[new]
    #[pyo3(signature = (n_components = None, explained_variance_ratio_target = None, whiten = false, device = None))]
    pub fn new(
        n_components: Option<usize>,
        explained_variance_ratio_target: Option<f64>,
        whiten: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PCA::new(
                n_components,
                explained_variance_ratio_target,
                whiten,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn get_covariance_buffer<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cov = self.inner.get_covariance()?;
        matrix_to_pybytes(py, &cov)
    }

    pub fn score_samples_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let scores = self.inner.score_samples(&mat)?;
        f64_slice_to_pybytes(py, &scores)
    }

    pub fn score_buffer(&self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        Ok(self.inner.score(&mat)?)
    }

    #[getter]
    pub fn n_components(&self) -> Option<usize> {
        self.inner.n_components
    }
    #[getter]
    pub fn explained_variance_ratio_target(&self) -> Option<f64> {
        self.inner.explained_variance_ratio_target
    }
    #[getter]
    pub fn whiten(&self) -> bool {
        self.inner.whiten
    }
    #[getter]
    pub fn components_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.components_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn explained_variance_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_.clone()
    }
    #[getter]
    pub fn explained_variance_ratio_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_ratio_.clone()
    }
    #[getter]
    pub fn singular_values_(&self) -> Option<Vec<f64>> {
        self.inner.singular_values_.clone()
    }
    #[getter]
    pub fn mean_(&self) -> Option<Vec<f64>> {
        self.inner.mean_.clone()
    }
    #[getter]
    pub fn n_components_(&self) -> Option<usize> {
        self.inner.n_components_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_samples_(&self) -> Option<usize> {
        self.inner.n_samples_
    }
    #[getter]
    pub fn noise_variance_(&self) -> Option<f64> {
        self.inner.noise_variance_
    }
});

/// Truncated Singular Value Decomposition (LSA / linear dimensionality reduction without centering) (＾▽＾)
#[derive(Debug, Clone)]
pub struct TruncatedSVD {
    pub n_components: usize,
    pub n_iter: usize,
    pub random_state: Option<u64>,
    pub algorithm: String,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub components_: Option<Matrix>,
    pub explained_variance_: Option<Vec<f64>>,
    pub explained_variance_ratio_: Option<Vec<f64>>,
    pub singular_values_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl TruncatedSVD {
    pub fn new(
        n_components: usize,
        n_iter: usize,
        random_state: Option<u64>,
        algorithm: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !matches!(algorithm.as_str(), "randomized" | "arpack" | "exact") {
            return Err(OchreError::InvalidParameter(format!(
                "algorithm must be 'randomized', 'arpack', or 'exact', got '{}' ( >_< )",
                algorithm
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            n_iter,
            random_state,
            algorithm,
            device: dev,
            device_: None,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        if self.n_features_in_.is_some() {
            self.device_ = Some(self.device.to_string());
        }
        Ok(())
    }

    fn randomized_svd(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        k: usize,
    ) -> OchreResult<(Vec<f64>, Matrix)> {
        let n = x.rows;
        let p = x.cols;
        let oversampling = 10usize;
        let l = (k + oversampling).min(p).min(n);

        let mut rng = SplitMix64::new(self.random_state);
        let mut omega_data = Vec::with_capacity(p * l);
        for _ in 0..(p * l) {
            omega_data.push(rng.next_standard_normal());
        }
        let omega = Matrix::from_vec(p, l, omega_data)?;

        let mut q = backend.matmul(x, &omega)?;
        mgs_orthonormalize(&mut q);

        for _ in 0..self.n_iter {
            let mut z = backend.matmul_at_b(x, &q)?;
            mgs_orthonormalize(&mut z);
            q = backend.matmul(x, &z)?;
            mgs_orthonormalize(&mut q);
        }

        // B = Q^T * X -> (l x p)
        let b = backend.matmul_at_b(&q, x)?;
        let b_t = b.transpose();
        let gram_ll = backend.matmul(&b, &b_t)?;
        let eig = backend.small_sym_eig_svd(&gram_ll)?;

        let singular_values: Vec<f64> = eig
            .eigenvalues
            .iter()
            .take(k)
            .map(|&lam| lam.max(0.0).sqrt())
            .collect();

        // V_cols = B^T * U_b -> (p x l)
        let mut v_cols = backend.matmul_at_b(&b, &eig.eigenvectors)?;
        let s_max = singular_values.first().copied().unwrap_or(0.0);
        let tol = 1e-12 * s_max.max(1e-30);
        for (col, &s) in singular_values.iter().enumerate().take(k) {
            if s > tol {
                let inv_s = 1.0 / s;
                for r in 0..p {
                    v_cols.data[r * l + col] *= inv_s;
                }
            } else {
                for r in 0..p {
                    v_cols.data[r * l + col] = 0.0;
                }
            }
        }
        let components = extract_components_from_v_cols(&v_cols, k)?;
        Ok((singular_values, components))
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        let n = x.rows;
        let p = x.cols;
        if self.n_components > p {
            return Err(OchreError::InvalidParameter(format!(
                "n_components ({}) must be <= n_features ({}) ( >_< )",
                self.n_components, p
            )));
        }
        let k = self.n_components.min(n).min(p);
        let backend = effective_backend(&self.device, n * p);

        let (singular_values, components) =
            if self.algorithm == "arpack" || self.algorithm == "exact" || p <= 512 {
                if n >= p || p <= 512 {
                    let (mut s_vals, v_cols, _) = backend.gram_svd(x, false)?;
                    let comps = extract_components_from_v_cols(&v_cols, k)?;
                    s_vals.truncate(k);
                    (s_vals, comps)
                } else if n <= 4096 {
                    let (mut s_vals, v_cols) = dual_gram_svd(&backend, x, k)?;
                    let comps = extract_components_from_v_cols(&v_cols, k)?;
                    s_vals.truncate(k);
                    (s_vals, comps)
                } else {
                    self.randomized_svd(&backend, x, k)?
                }
            } else {
                self.randomized_svd(&backend, x, k)?
            };

        // Explained variance (ddof=0) matching sklearn.decomposition.TruncatedSVD
        let full_stats = backend.col_reduce_stats(x, None)?;
        let total_var: f64 = full_stats.var.iter().sum();
        let exp_var = if matches!(
            effective_transform_backend(&self.device, n * p),
            Backend::Cpu
        ) {
            let comp_data = &components.data;
            let comp_packed = pack_components_4(comp_data, k, p);
            let mut anchor_t = vec![0.0; k];
            let first_row = &x.data[..p];
            for j in 0..k {
                anchor_t[j] = dot_unrolled(first_row, &comp_data[j * p..(j + 1) * p]);
            }
            let accumulate_chunk =
                |x_chunk: &[f64], buf: &mut Vec<f64>, sum_t: &mut [f64], sum_sq_t: &mut [f64]| {
                    let m = x_chunk.len() / p;
                    buf.resize(m * k, 0.0);
                    project_4x4_chunk(
                        x_chunk,
                        m,
                        p,
                        &comp_packed,
                        comp_data,
                        k,
                        Some(&anchor_t),
                        None,
                        buf,
                    );
                    for row in buf.chunks_exact(k) {
                        for j in 0..k {
                            let d = row[j];
                            sum_t[j] += d;
                            sum_sq_t[j] += d * d;
                        }
                    }
                };
            let (sum_t, sum_sq_t) = if n >= PARALLEL_ROW_THRESHOLD {
                x.data
                    .par_chunks(PROJECT_CHUNK_ROWS * p)
                    .fold(
                        || {
                            (
                                Vec::with_capacity(PROJECT_CHUNK_ROWS * k),
                                vec![0.0; k],
                                vec![0.0; k],
                            )
                        },
                        |(mut buf, mut st, mut ssq), x_chunk| {
                            accumulate_chunk(x_chunk, &mut buf, &mut st, &mut ssq);
                            (buf, st, ssq)
                        },
                    )
                    .map(|(_, st, ssq)| (st, ssq))
                    .reduce(
                        || (vec![0.0; k], vec![0.0; k]),
                        |(mut a_s, mut a_sq), (b_s, b_sq)| {
                            for j in 0..k {
                                a_s[j] += b_s[j];
                                a_sq[j] += b_sq[j];
                            }
                            (a_s, a_sq)
                        },
                    )
            } else {
                let mut buf = Vec::with_capacity(n * k);
                let mut st = vec![0.0; k];
                let mut ssq = vec![0.0; k];
                accumulate_chunk(&x.data, &mut buf, &mut st, &mut ssq);
                (st, ssq)
            };
            let inv_n = 1.0 / (n as f64);
            (0..k)
                .map(|j| {
                    let mean_t = sum_t[j] * inv_n;
                    (sum_sq_t[j] * inv_n - mean_t * mean_t).max(0.0)
                })
                .collect()
        } else {
            let comp_t = components.transpose();
            let x_transformed = backend.matmul(x, &comp_t)?;
            backend.col_reduce_stats(&x_transformed, None)?.var
        };
        let exp_var_ratio: Vec<f64> = exp_var
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect();

        self.components_ = Some(components);
        self.explained_variance_ = Some(exp_var);
        self.explained_variance_ratio_ = Some(exp_var_ratio);
        self.singular_values_ = Some(singular_values);
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let components = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let backend = effective_transform_backend(&self.device, x.rows * expected);
        if matches!(backend, Backend::Cpu) {
            return project_cpu_matrix(x, components, None, None);
        }
        let comp_t = components.transpose();
        backend.matmul(x, &comp_t)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let components = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let k = components.rows;
        if x.cols != k {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: k,
            });
        }
        effective_backend(&self.device, x.rows * components.cols).matmul(x, components)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_TRUNCATED_SVD);
        write_usize(&mut buf, self.n_components);
        write_usize(&mut buf, self.n_iter);
        write_opt_u64(&mut buf, self.random_state);
        write_str(&mut buf, &self.algorithm);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.components_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_ratio_);
        write_opt_f64_vec(&mut buf, &self.singular_values_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_TRUNCATED_SVD)?;
        let n_components = r.read_usize()?;
        let n_iter = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let algorithm = r.read_str()?;
        let saved_device = r.read_opt_str()?;
        let components_ = r.read_opt_matrix()?;
        let explained_variance_ = r.read_opt_f64_vec()?;
        let explained_variance_ratio_ = r.read_opt_f64_vec()?;
        let singular_values_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;

        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;

        Ok(Self {
            n_components,
            n_iter,
            random_state,
            algorithm,
            device: dev,
            device_,
            components_,
            explained_variance_,
            explained_variance_ratio_,
            singular_values_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "TruncatedSVD")]
#[derive(Clone)]
pub struct PyTruncatedSVD {
    inner: TruncatedSVD,
}

pymethods_with_dit!(PyTruncatedSVD, TruncatedSVD, {
    #[new]
    #[pyo3(signature = (n_components = 2, n_iter = 5, random_state = None, algorithm = "randomized".to_string(), device = None))]
    pub fn new(
        n_components: usize,
        n_iter: usize,
        random_state: Option<u64>,
        algorithm: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TruncatedSVD::new(n_components, n_iter, random_state, algorithm, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }
    #[getter]
    pub fn n_iter(&self) -> usize {
        self.inner.n_iter
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn algorithm(&self) -> String {
        self.inner.algorithm.clone()
    }
    #[getter]
    pub fn components_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.components_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn explained_variance_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_.clone()
    }
    #[getter]
    pub fn explained_variance_ratio_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_ratio_.clone()
    }
    #[getter]
    pub fn singular_values_(&self) -> Option<Vec<f64>> {
        self.inner.singular_values_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPCA>()?;
    m.add_class::<PyTruncatedSVD>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_and_truncated_svd_parallel_4x4_parity() {
        let n = 651;
        let p = 13;
        let k = 6; // tests 1 full 4-component block + 2 remainder components, and 4-row blocks + 3 tail rows
        let mut data = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                data[i * p + j] = ((i * 11 + j * 17) % 53) as f64 * 0.1 + (j as f64) * 0.5;
            }
        }
        let x = Matrix::from_vec(n, p, data).unwrap();

        for whiten in [false, true] {
            let mut pca = PCA::new(Some(k), None, whiten, Some("cpu".to_string())).unwrap();
            let z = pca.fit_transform(&x).unwrap();
            assert_eq!(z.rows, n);
            assert_eq!(z.cols, k);
            let comps = pca.components_.as_ref().unwrap();
            let mean = pca.mean_.as_ref().unwrap();
            let exp_var = pca.explained_variance_.as_ref().unwrap();
            for r in [0, 3, 4, 511, 512, 650] {
                for c in 0..k {
                    let mut ref_val = 0.0;
                    for j in 0..p {
                        ref_val += (x[(r, j)] - mean[j]) * comps[(c, j)];
                    }
                    if whiten {
                        ref_val /= exp_var[c].sqrt();
                    }
                    assert!(
                        (z[(r, c)] - ref_val).abs() < 1e-10,
                        "PCA mismatch at ({r}, {c}): {} vs {ref_val}",
                        z[(r, c)]
                    );
                }
            }
        }

        let mut svd = TruncatedSVD::new(
            k,
            5,
            Some(42),
            "randomized".to_string(),
            Some("cpu".to_string()),
        )
        .unwrap();
        let z_svd = svd.fit_transform(&x).unwrap();
        assert_eq!(z_svd.rows, n);
        assert_eq!(z_svd.cols, k);
        let comps = svd.components_.as_ref().unwrap();
        for r in [0, 3, 4, 511, 512, 650] {
            for c in 0..k {
                let mut ref_val = 0.0;
                for j in 0..p {
                    ref_val += x[(r, j)] * comps[(c, j)];
                }
                assert!(
                    (z_svd[(r, c)] - ref_val).abs() < 1e-10,
                    "SVD mismatch at ({r}, {c}): {} vs {ref_val}",
                    z_svd[(r, c)]
                );
            }
        }
    }
}
