// OchreML Gaussian Mixture Model Module (＾▽＾)
// Implements GaussianMixture with EM (full, tied, diag, spherical covariances),
// k-means++ initialization via Backend N3 (pairwise_l2_sq) & N4 (row_argmin),
// Cholesky log-likelihoods, .dit persistence, and PyO3 zero-copy buffer bindings.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64,
    write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_str,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes, usize_slice_to_i64_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

pub const TAG_GAUSSIAN_MIXTURE: u8 = 64;

const MAX_DIST_ELEMENTS: usize = 4_000_000;
const KMEANS_INIT_MAX_ITER: usize = 10;
const GMM_GPU_THRESHOLD: usize = 2_000_000;
const PARALLEL_ROW_THRESHOLD: usize = 512;
const GMM_CHUNK_ROWS: usize = 64;
const GMM_PREDICT_CHUNK_ROWS: usize = 256;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < GMM_GPU_THRESHOLD {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn slice_all_finite(data: &[f64]) -> bool {
    let mut acc = [0.0f64; 4];
    let (chunks, rem) = data.as_chunks::<4>();
    for &c in chunks {
        acc[0] += c[0] * 0.0;
        acc[1] += c[1] * 0.0;
        acc[2] += c[2] * 0.0;
        acc[3] += c[3] * 0.0;
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for &v in rem {
        s += v * 0.0;
    }
    s == 0.0
}

#[inline(always)]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0f64; 4];
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (&xa, &xb) in ac.iter().zip(bc.iter()) {
        for l in 0..4 {
            s[l] += xa[l] * xb[l];
        }
    }
    let mut r = (s[0] + s[1]) + (s[2] + s[3]);
    for (&xa, &xb) in ar.iter().zip(br.iter()) {
        r += xa * xb;
    }
    r
}

/// Deterministic SplitMix64 + Xorshift64 PRNG (zero external crate dependencies).
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or(0x9e37_79b9_7f4a_7c15);
        let mut z = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        Self {
            state: if z == 0 { 0x9e37_79b9_7f4a_7c15 } else { z },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    /// Generate a pair of independent N(0, 1) variates via Box-Muller.
    #[inline]
    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

fn validate_x(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let has_non_finite = if x.data.len() >= 65_536 {
        x.data
            .par_chunks(16_384)
            .any(|chunk| !slice_all_finite(chunk))
    } else {
        !slice_all_finite(&x.data)
    };
    if has_non_finite {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or infinite values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_fitted_cols(n_features_in: Option<usize>, cols: usize) -> OchreResult<usize> {
    let expected = n_features_in.ok_or(OchreError::NotFitted)?;
    if cols != expected {
        return Err(OchreError::FeatureCountMismatch {
            features: cols,
            expected,
        });
    }
    Ok(expected)
}

#[allow(clippy::too_many_arguments)]
fn validate_params(
    n_components: usize,
    covariance_type: &str,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: &str,
) -> OchreResult<()> {
    if n_components == 0 {
        return Err(OchreError::InvalidParameter(
            "n_components must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if !matches!(covariance_type, "full" | "tied" | "diag" | "spherical") {
        return Err(OchreError::InvalidParameter(format!(
            "covariance_type must be 'full', 'tied', 'diag', or 'spherical', got '{}' (・`ω´・)",
            covariance_type
        )));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(OchreError::InvalidParameter(
            "tol must be a non-negative finite number (・`ω´・)".to_string(),
        ));
    }
    if !reg_covar.is_finite() || reg_covar < 0.0 {
        return Err(OchreError::InvalidParameter(
            "reg_covar must be a non-negative finite number (・`ω´・)".to_string(),
        ));
    }
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if n_init == 0 {
        return Err(OchreError::InvalidParameter(
            "n_init must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if init_params != "kmeans" && init_params != "random" {
        return Err(OchreError::InvalidParameter(format!(
            "init_params must be 'kmeans' or 'random', got '{}' (・`ω´・)",
            init_params
        )));
    }
    Ok(())
}

/// Compute lower Cholesky factor `L` such that `L * L^T = A` for symmetric positive-definite `A` (p x p).
fn cholesky_lower(a: &[f64], p: usize) -> OchreResult<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * p + k] * l[j * p + k];
            }
            if i == j {
                let diag = a[i * p + i] - sum;
                if !diag.is_finite() || diag <= 0.0 {
                    return Err(OchreError::SingularMatrix(
                        "Covariance matrix is not positive definite; try increasing reg_covar"
                            .to_string(),
                    ));
                }
                l[i * p + i] = diag.sqrt();
            } else {
                let l_jj = l[j * p + j];
                if l_jj <= 0.0 || !l_jj.is_finite() {
                    return Err(OchreError::SingularMatrix(
                        "Covariance matrix is not positive definite; try increasing reg_covar"
                            .to_string(),
                    ));
                }
                l[i * p + j] = (a[i * p + j] - sum) / l_jj;
            }
        }
    }
    Ok(l)
}

/// Compute upper triangular `prec_chol = (L^{-1})^T` from covariance matrix `cov` (p x p),
/// matching `sklearn.mixture.GaussianMixture.precisions_cholesky_`.
fn compute_prec_chol_from_cov(cov: &[f64], p: usize) -> OchreResult<Vec<f64>> {
    let l = cholesky_lower(cov, p)?;
    // Solve L * Y = I_p (lower triangular forward substitution) and store U = Y^T.
    let mut u = vec![0.0; p * p];
    for col in 0..p {
        for row in col..p {
            let mut sum = if row == col { 1.0 } else { 0.0 };
            for k in col..row {
                sum -= l[row * p + k] * u[col * p + k];
            }
            u[col * p + row] = sum / l[row * p + row];
        }
    }
    Ok(u)
}

fn assign_labels_n3_n4(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<Vec<usize>> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if n.saturating_mul(k) <= MAX_DIST_ELEMENTS {
        let dist = backend.pairwise_l2_sq(x, centers)?;
        let (labels, _) = backend.row_argmin(&dist)?;
        return Ok(labels);
    }

    let chunk_rows = (MAX_DIST_ELEMENTS / k.max(1)).max(1).min(n);
    let mut all_labels = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_rows).min(n);
        let chunk = Matrix::from_slice(end - start, p, &x.data[start * p..end * p])?;
        let dist = backend.pairwise_l2_sq(&chunk, centers)?;
        let (labels, _) = backend.row_argmin(&dist)?;
        all_labels.extend(labels);
        start = end;
    }
    Ok(all_labels)
}

fn init_centers_kmeans_pp(
    backend: &Backend<'_>,
    x: &Matrix,
    n_clusters: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let mut centers_data = vec![0.0; n_clusters * p];

    let first_idx = rng.gen_range(n);
    centers_data[0..p].copy_from_slice(&x.data[first_idx * p..(first_idx + 1) * p]);

    if n_clusters == 1 {
        return Matrix::from_vec(n_clusters, p, centers_data);
    }

    let first_mat = Matrix::from_slice(1, p, &centers_data[0..p])?;
    let mut min_sq_dist = backend.pairwise_l2_sq(x, &first_mat)?.data;

    for c in 1..n_clusters {
        let mut total = 0.0;
        for &d in &min_sq_dist {
            total += d.max(0.0);
        }

        let next_idx = if total <= 0.0 || !total.is_finite() {
            rng.gen_range(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cum = 0.0;
            let mut chosen = n - 1;
            for (i, &d) in min_sq_dist.iter().enumerate() {
                cum += d.max(0.0);
                if cum >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        centers_data[c * p..(c + 1) * p].copy_from_slice(&x.data[next_idx * p..(next_idx + 1) * p]);

        if c + 1 < n_clusters {
            let new_c = Matrix::from_slice(1, p, &centers_data[c * p..(c + 1) * p])?;
            let d_new = backend.pairwise_l2_sq(x, &new_c)?.data;
            for i in 0..n {
                if d_new[i] < min_sq_dist[i] {
                    min_sq_dist[i] = d_new[i];
                }
            }
        }
    }

    Matrix::from_vec(n_clusters, p, centers_data)
}

fn init_kmeans_labels(
    backend: &Backend<'_>,
    x: &Matrix,
    n_components: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Vec<usize>> {
    let n = x.rows;
    let p = x.cols;
    let mut centers = init_centers_kmeans_pp(backend, x, n_components, rng)?;
    let mut labels = assign_labels_n3_n4(backend, x, &centers)?;

    for _ in 0..KMEANS_INIT_MAX_ITER {
        let mut counts = vec![0usize; n_components];
        let mut new_centers = vec![0.0; n_components * p];
        for (i, &c) in labels.iter().enumerate() {
            counts[c] += 1;
            let row = &x.data[i * p..(i + 1) * p];
            let c_row = &mut new_centers[c * p..(c + 1) * p];
            for j in 0..p {
                c_row[j] += row[j];
            }
        }
        for c in 0..n_components {
            if counts[c] > 0 {
                let inv = 1.0 / (counts[c] as f64);
                for j in 0..p {
                    new_centers[c * p + j] *= inv;
                }
            } else {
                let ridx = rng.gen_range(n);
                new_centers[c * p..(c + 1) * p].copy_from_slice(&x.data[ridx * p..(ridx + 1) * p]);
            }
        }
        centers = Matrix::from_vec(n_components, p, new_centers)?;
        let new_labels = assign_labels_n3_n4(backend, x, &centers)?;
        if new_labels == labels {
            break;
        }
        labels = new_labels;
    }

    Ok(labels)
}

fn estimate_gaussian_parameters_from_labels(
    x: &Matrix,
    labels: &[usize],
    k: usize,
    covariance_type: &str,
    reg_covar: f64,
    precomputed_xtx: Option<&[f64]>,
) -> OchreResult<GaussianParamsEstimate> {
    let n = x.rows;
    let p = x.cols;
    let mut nk = vec![0.0; k];
    let mut xk = vec![0.0; k * p];
    for (i, &c) in labels.iter().enumerate() {
        nk[c] += 1.0;
        let x_row = &x.data[i * p..(i + 1) * p];
        let xk_c = &mut xk[c * p..(c + 1) * p];
        for j in 0..p {
            xk_c[j] += x_row[j];
        }
    }

    let eps_pad = 10.0 * f64::EPSILON;
    let r_sum = nk.clone();
    let mut nk_sum = 0.0;
    for val in &mut nk {
        *val += eps_pad;
        nk_sum += *val;
    }
    let weights: Vec<f64> = nk.iter().map(|&v| v / nk_sum).collect();
    let mut means_data = vec![0.0; k * p];
    for c in 0..k {
        let inv = 1.0 / nk[c];
        for j in 0..p {
            means_data[c * p + j] = xk[c * p + j] * inv;
        }
    }
    let means = Matrix::from_vec(k, p, means_data)?;

    let (covariances, precisions_cholesky) = match covariance_type {
        "full" => {
            let mut covs = vec![0.0; k * p * p];
            let mut prec_chols = vec![0.0; k * p * p];
            for (i, &c) in labels.iter().enumerate() {
                let x_row = &x.data[i * p..(i + 1) * p];
                let m_c = &means.data[c * p..(c + 1) * p];
                let cov_c = &mut covs[c * p * p..(c + 1) * p * p];
                for a in 0..p {
                    let da = x_row[a] - m_c[a];
                    for b in a..p {
                        cov_c[a * p + b] += da * (x_row[b] - m_c[b]);
                    }
                }
            }
            for c in 0..k {
                let cov_c = &mut covs[c * p * p..(c + 1) * p * p];
                let inv_nk = 1.0 / nk[c];
                for a in 0..p {
                    for b in a..p {
                        let val = cov_c[a * p + b] * inv_nk + if a == b { reg_covar } else { 0.0 };
                        cov_c[a * p + b] = val;
                        cov_c[b * p + a] = val;
                    }
                }
                let u = compute_prec_chol_from_cov(cov_c, p)?;
                prec_chols[c * p * p..(c + 1) * p * p].copy_from_slice(&u);
            }
            (covs, prec_chols)
        }
        "tied" => {
            let mut cov = if let Some(xtx) = precomputed_xtx {
                let mut c_mat = xtx.to_vec();
                for c in 0..k {
                    let rc = r_sum[c];
                    let m_c = &means.data[c * p..(c + 1) * p];
                    let xk_c = &xk[c * p..(c + 1) * p];
                    for a in 0..p {
                        for b in a..p {
                            c_mat[a * p + b] +=
                                -xk_c[a] * m_c[b] - xk_c[b] * m_c[a] + rc * m_c[a] * m_c[b];
                        }
                    }
                }
                c_mat
            } else {
                let mut c_mat = vec![0.0; p * p];
                for (i, &c) in labels.iter().enumerate() {
                    let x_row = &x.data[i * p..(i + 1) * p];
                    let m_c = &means.data[c * p..(c + 1) * p];
                    for a in 0..p {
                        let da = x_row[a] - m_c[a];
                        for b in a..p {
                            c_mat[a * p + b] += da * (x_row[b] - m_c[b]);
                        }
                    }
                }
                c_mat
            };
            let inv_nk_sum = 1.0 / nk_sum;
            for a in 0..p {
                for b in a..p {
                    let mut val = cov[a * p + b] * inv_nk_sum;
                    if a == b {
                        val = val.max(0.0) + reg_covar;
                    }
                    cov[a * p + b] = val;
                    cov[b * p + a] = val;
                }
            }
            let prec_chol = compute_prec_chol_from_cov(&cov, p)?;
            (cov, prec_chol)
        }
        "diag" => {
            let mut covs = vec![0.0; k * p];
            let mut prec_chols = vec![0.0; k * p];
            for (i, &c) in labels.iter().enumerate() {
                let x_row = &x.data[i * p..(i + 1) * p];
                let m_c = &means.data[c * p..(c + 1) * p];
                let cov_c = &mut covs[c * p..(c + 1) * p];
                for j in 0..p {
                    let d = x_row[j] - m_c[j];
                    cov_c[j] += d * d;
                }
            }
            for c in 0..k {
                let cov_c = &mut covs[c * p..(c + 1) * p];
                let inv_nk = 1.0 / nk[c];
                for j in 0..p {
                    let v = cov_c[j] * inv_nk + reg_covar;
                    cov_c[j] = v;
                    if !v.is_finite() || v <= 0.0 {
                        return Err(OchreError::SingularMatrix(
                            "Diagonal covariance is non-positive; try increasing reg_covar"
                                .to_string(),
                        ));
                    }
                    prec_chols[c * p + j] = 1.0 / v.sqrt();
                }
            }
            (covs, prec_chols)
        }
        "spherical" => {
            let mut covs = vec![0.0; k];
            let mut prec_chols = vec![0.0; k];
            let inv_p = 1.0 / (p as f64);
            for (i, &c) in labels.iter().enumerate() {
                let x_row = &x.data[i * p..(i + 1) * p];
                let m_c = &means.data[c * p..(c + 1) * p];
                let mut sq = 0.0;
                for j in 0..p {
                    let d = x_row[j] - m_c[j];
                    sq += d * d;
                }
                covs[c] += sq;
            }
            for c in 0..k {
                let v = covs[c] * (1.0 / nk[c]) * inv_p + reg_covar;
                if !v.is_finite() || v <= 0.0 {
                    return Err(OchreError::SingularMatrix(
                        "Spherical covariance is non-positive; try increasing reg_covar"
                            .to_string(),
                    ));
                }
                covs[c] = v;
                prec_chols[c] = 1.0 / v.sqrt();
            }
            (covs, prec_chols)
        }
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported covariance_type '{}'",
                covariance_type
            )));
        }
    };

    let _ = n;
    Ok((weights, means, covariances, precisions_cholesky))
}

enum GmmEvalData {
    Full {
        u_packed_all: Vec<[f64; 4]>,
        u_comp_stride: usize,
        mu_packed_all: Vec<[f64; 4]>,
        nb: usize,
        offsets: Vec<f64>,
    },
    Tied {
        u_packed: Vec<[f64; 4]>,
        mu_packed_all: Vec<[f64; 4]>,
        nb: usize,
        offsets: Vec<f64>,
    },
    Diag {
        m_scaled: Vec<f64>,
        offsets: Vec<f64>,
    },
    Spherical {
        prec_sq: Vec<f64>,
        offsets: Vec<f64>,
    },
}

/// Pack upper-triangular `prec_chol` (`p x p`) into zero-padded 4-column blocks `[f64; 4]`.
/// Only rows `0..min((b+1)*4, p)` are stored per block `b`, keeping unused entries strictly `0.0`.
fn pack_upper_chol_blocks(u_c: &[f64], p: usize, nb: usize, u_dst: &mut [[f64; 4]]) {
    let mut ptr = 0usize;
    for b in 0..nb {
        let col_start = b * 4;
        let r_end = (col_start + 4).min(p);
        for r_idx in 0..r_end {
            let mut blk = [0.0f64; 4];
            for (l, slot) in blk.iter_mut().enumerate() {
                let j = col_start + l;
                if j < p && r_idx <= j {
                    *slot = u_c[r_idx * p + j];
                }
            }
            u_dst[ptr + r_idx] = blk;
        }
        ptr += r_end;
    }
}

/// Project mean vector `mu_c` through packed upper-triangular blocks (`mu_c * U`).
fn project_mean_packed(
    mu_c: &[f64],
    u_packed: &[[f64; 4]],
    p: usize,
    nb: usize,
    mu_dst: &mut [[f64; 4]],
) {
    let mut ptr = 0usize;
    for (b, mu_slot) in mu_dst.iter_mut().enumerate().take(nb) {
        let r_end = ((b + 1) * 4).min(p);
        let mut acc = [0.0f64; 4];
        for r_idx in 0..r_end {
            let mv = mu_c[r_idx];
            let u4 = u_packed[ptr + r_idx];
            for l in 0..4 {
                acc[l] += mv * u4[l];
            }
        }
        *mu_slot = acc;
        ptr += r_end;
    }
}

fn prepare_gmm_eval_data(
    weights: Option<&[f64]>,
    means: &Matrix,
    precisions_cholesky: &[f64],
    covariance_type: &str,
) -> OchreResult<GmmEvalData> {
    let k = means.rows;
    let p = means.cols;
    let const_term = (p as f64) * (2.0 * PI).ln();
    let log_w: Vec<f64> = match weights {
        Some(ws) => ws
            .iter()
            .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
            .collect(),
        None => vec![0.0; k],
    };

    match covariance_type {
        "full" => {
            let nb = p.div_ceil(4);
            let u_comp_stride: usize = (0..nb).map(|b| ((b + 1) * 4).min(p)).sum();
            let mut u_packed_all = vec![[0.0f64; 4]; k * u_comp_stride];
            let mut mu_packed_all = vec![[0.0f64; 4]; k * nb];
            let mut offsets = vec![0.0; k];
            for c in 0..k {
                let u_c = &precisions_cholesky[c * p * p..(c + 1) * p * p];
                let u_dst = &mut u_packed_all[c * u_comp_stride..(c + 1) * u_comp_stride];
                let mut log_det = 0.0;
                for j in 0..p {
                    log_det += u_c[j * p + j].ln();
                }
                pack_upper_chol_blocks(u_c, p, nb, u_dst);
                let mu_c = &means.data[c * p..(c + 1) * p];
                let mu_dst = &mut mu_packed_all[c * nb..(c + 1) * nb];
                project_mean_packed(mu_c, u_dst, p, nb, mu_dst);
                offsets[c] = -0.5 * const_term + log_det + log_w[c];
            }
            Ok(GmmEvalData::Full {
                u_packed_all,
                u_comp_stride,
                mu_packed_all,
                nb,
                offsets,
            })
        }
        "tied" => {
            let nb = p.div_ceil(4);
            let u_stride: usize = (0..nb).map(|b| ((b + 1) * 4).min(p)).sum();
            let mut u_packed = vec![[0.0f64; 4]; u_stride];
            let mut log_det = 0.0;
            for j in 0..p {
                log_det += precisions_cholesky[j * p + j].ln();
            }
            pack_upper_chol_blocks(precisions_cholesky, p, nb, &mut u_packed);
            let mut mu_packed_all = vec![[0.0f64; 4]; k * nb];
            for c in 0..k {
                let mu_c = &means.data[c * p..(c + 1) * p];
                let mu_dst = &mut mu_packed_all[c * nb..(c + 1) * nb];
                project_mean_packed(mu_c, &u_packed, p, nb, mu_dst);
            }
            let base = -0.5 * const_term + log_det;
            let offsets: Vec<f64> = log_w.iter().map(|&lw| base + lw).collect();
            Ok(GmmEvalData::Tied {
                u_packed,
                mu_packed_all,
                nb,
                offsets,
            })
        }
        "diag" => {
            let mut m_scaled = vec![0.0; k * p];
            let mut offsets = vec![0.0; k];
            for c in 0..k {
                let pc = &precisions_cholesky[c * p..(c + 1) * p];
                let mu_c = &means.data[c * p..(c + 1) * p];
                let ms_c = &mut m_scaled[c * p..(c + 1) * p];
                let mut log_det = 0.0;
                for j in 0..p {
                    log_det += pc[j].ln();
                    ms_c[j] = mu_c[j] * pc[j];
                }
                offsets[c] = -0.5 * const_term + log_det + log_w[c];
            }
            Ok(GmmEvalData::Diag { m_scaled, offsets })
        }
        "spherical" => {
            let mut prec_sq = Vec::with_capacity(k);
            let mut offsets = Vec::with_capacity(k);
            for (c, &pc) in precisions_cholesky.iter().enumerate().take(k) {
                prec_sq.push(pc * pc);
                let log_det = (p as f64) * pc.ln();
                offsets.push(-0.5 * const_term + log_det + log_w[c]);
            }
            Ok(GmmEvalData::Spherical { prec_sq, offsets })
        }
        _ => Err(OchreError::InvalidParameter(format!(
            "Unsupported covariance_type '{}'",
            covariance_type
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_log_prob_chunk(
    eval: &GmmEvalData,
    x_chunk: &[f64],
    m: usize,
    p: usize,
    k: usize,
    means_data: &[f64],
    prec_chols: &[f64],
    out_lp: &mut [f64],
    xu_scratch: &mut Vec<f64>,
) {
    match eval {
        GmmEvalData::Full {
            u_packed_all,
            u_comp_stride,
            mu_packed_all,
            nb,
            offsets,
        } => {
            let nb = *nb;
            let u_comp_stride = *u_comp_stride;
            let mut r = 0usize;
            while r + 4 <= m {
                let x0 = &x_chunk[r * p..(r + 1) * p];
                let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
                let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
                let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
                for c in 0..k {
                    let u_packed = &u_packed_all[c * u_comp_stride..(c + 1) * u_comp_stride];
                    let mu_packed = &mu_packed_all[c * nb..(c + 1) * nb];
                    let mut m0 = [0.0f64; 4];
                    let mut m1 = [0.0f64; 4];
                    let mut m2 = [0.0f64; 4];
                    let mut m3 = [0.0f64; 4];
                    let mut ptr = 0usize;
                    for (b, &mu4) in mu_packed.iter().enumerate().take(nb) {
                        let r_end = ((b + 1) * 4).min(p);
                        let u_blk = &u_packed[ptr..ptr + r_end];
                        let mut y0 = [0.0f64; 4];
                        let mut y1 = [0.0f64; 4];
                        let mut y2 = [0.0f64; 4];
                        let mut y3 = [0.0f64; 4];
                        for r_idx in 0..r_end {
                            let u4 = u_blk[r_idx];
                            let v0 = x0[r_idx];
                            let v1 = x1[r_idx];
                            let v2 = x2[r_idx];
                            let v3 = x3[r_idx];
                            for l in 0..4 {
                                y0[l] += v0 * u4[l];
                                y1[l] += v1 * u4[l];
                                y2[l] += v2 * u4[l];
                                y3[l] += v3 * u4[l];
                            }
                        }
                        ptr += r_end;
                        for l in 0..4 {
                            let e0 = y0[l] - mu4[l];
                            let e1 = y1[l] - mu4[l];
                            let e2 = y2[l] - mu4[l];
                            let e3 = y3[l] - mu4[l];
                            m0[l] += e0 * e0;
                            m1[l] += e1 * e1;
                            m2[l] += e2 * e2;
                            m3[l] += e3 * e3;
                        }
                    }
                    let off = offsets[c];
                    let ms0 = (m0[0] + m0[1]) + (m0[2] + m0[3]);
                    let ms1 = (m1[0] + m1[1]) + (m1[2] + m1[3]);
                    let ms2 = (m2[0] + m2[1]) + (m2[2] + m2[3]);
                    let ms3 = (m3[0] + m3[1]) + (m3[2] + m3[3]);
                    out_lp[r * k + c] = off - 0.5 * ms0;
                    out_lp[(r + 1) * k + c] = off - 0.5 * ms1;
                    out_lp[(r + 2) * k + c] = off - 0.5 * ms2;
                    out_lp[(r + 3) * k + c] = off - 0.5 * ms3;
                }
                r += 4;
            }
            while r < m {
                let x_row = &x_chunk[r * p..(r + 1) * p];
                for c in 0..k {
                    let u_packed = &u_packed_all[c * u_comp_stride..(c + 1) * u_comp_stride];
                    let mu_packed = &mu_packed_all[c * nb..(c + 1) * nb];
                    let mut m0 = [0.0f64; 4];
                    let mut ptr = 0usize;
                    for (b, &mu4) in mu_packed.iter().enumerate().take(nb) {
                        let r_end = ((b + 1) * 4).min(p);
                        let u_blk = &u_packed[ptr..ptr + r_end];
                        let mut y0 = [0.0f64; 4];
                        for r_idx in 0..r_end {
                            let u4 = u_blk[r_idx];
                            let v0 = x_row[r_idx];
                            for l in 0..4 {
                                y0[l] += v0 * u4[l];
                            }
                        }
                        ptr += r_end;
                        for l in 0..4 {
                            let e0 = y0[l] - mu4[l];
                            m0[l] += e0 * e0;
                        }
                    }
                    let maha_sq = (m0[0] + m0[1]) + (m0[2] + m0[3]);
                    out_lp[r * k + c] = offsets[c] - 0.5 * maha_sq;
                }
                r += 1;
            }
        }
        GmmEvalData::Tied {
            u_packed,
            mu_packed_all,
            nb,
            offsets,
        } => {
            let nb = *nb;
            let p_pad = nb * 4;
            xu_scratch.resize(4 * p_pad, 0.0);
            let mut r = 0usize;
            while r + 4 <= m {
                let x0 = &x_chunk[r * p..(r + 1) * p];
                let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
                let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
                let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
                let (xu01, xu23) = xu_scratch.split_at_mut(2 * p_pad);
                let (xu0, xu1) = xu01.split_at_mut(p_pad);
                let (xu2, xu3) = xu23.split_at_mut(p_pad);
                let mut ptr = 0usize;
                for b in 0..nb {
                    let r_end = ((b + 1) * 4).min(p);
                    let u_blk = &u_packed[ptr..ptr + r_end];
                    let mut y0 = [0.0f64; 4];
                    let mut y1 = [0.0f64; 4];
                    let mut y2 = [0.0f64; 4];
                    let mut y3 = [0.0f64; 4];
                    for r_idx in 0..r_end {
                        let u4 = u_blk[r_idx];
                        let v0 = x0[r_idx];
                        let v1 = x1[r_idx];
                        let v2 = x2[r_idx];
                        let v3 = x3[r_idx];
                        for l in 0..4 {
                            y0[l] += v0 * u4[l];
                            y1[l] += v1 * u4[l];
                            y2[l] += v2 * u4[l];
                            y3[l] += v3 * u4[l];
                        }
                    }
                    ptr += r_end;
                    let off_b = b * 4;
                    xu0[off_b..off_b + 4].copy_from_slice(&y0);
                    xu1[off_b..off_b + 4].copy_from_slice(&y1);
                    xu2[off_b..off_b + 4].copy_from_slice(&y2);
                    xu3[off_b..off_b + 4].copy_from_slice(&y3);
                }
                for c in 0..k {
                    let mu_packed = &mu_packed_all[c * nb..(c + 1) * nb];
                    let mut m0 = [0.0f64; 4];
                    let mut m1 = [0.0f64; 4];
                    let mut m2 = [0.0f64; 4];
                    let mut m3 = [0.0f64; 4];
                    for (b, &mu4) in mu_packed.iter().enumerate().take(nb) {
                        let off_b = b * 4;
                        for l in 0..4 {
                            let e0 = xu0[off_b + l] - mu4[l];
                            let e1 = xu1[off_b + l] - mu4[l];
                            let e2 = xu2[off_b + l] - mu4[l];
                            let e3 = xu3[off_b + l] - mu4[l];
                            m0[l] += e0 * e0;
                            m1[l] += e1 * e1;
                            m2[l] += e2 * e2;
                            m3[l] += e3 * e3;
                        }
                    }
                    let off = offsets[c];
                    out_lp[r * k + c] = off - 0.5 * ((m0[0] + m0[1]) + (m0[2] + m0[3]));
                    out_lp[(r + 1) * k + c] = off - 0.5 * ((m1[0] + m1[1]) + (m1[2] + m1[3]));
                    out_lp[(r + 2) * k + c] = off - 0.5 * ((m2[0] + m2[1]) + (m2[2] + m2[3]));
                    out_lp[(r + 3) * k + c] = off - 0.5 * ((m3[0] + m3[1]) + (m3[2] + m3[3]));
                }
                r += 4;
            }
            while r < m {
                let x_row = &x_chunk[r * p..(r + 1) * p];
                let xu = &mut xu_scratch[..p_pad];
                let mut ptr = 0usize;
                for b in 0..nb {
                    let r_end = ((b + 1) * 4).min(p);
                    let u_blk = &u_packed[ptr..ptr + r_end];
                    let mut y0 = [0.0f64; 4];
                    for r_idx in 0..r_end {
                        let u4 = u_blk[r_idx];
                        let v0 = x_row[r_idx];
                        for l in 0..4 {
                            y0[l] += v0 * u4[l];
                        }
                    }
                    ptr += r_end;
                    xu[b * 4..b * 4 + 4].copy_from_slice(&y0);
                }
                for c in 0..k {
                    let mu_packed = &mu_packed_all[c * nb..(c + 1) * nb];
                    let mut m0 = [0.0f64; 4];
                    for (b, &mu4) in mu_packed.iter().enumerate().take(nb) {
                        let off_b = b * 4;
                        for l in 0..4 {
                            let e = xu[off_b + l] - mu4[l];
                            m0[l] += e * e;
                        }
                    }
                    let maha_sq = (m0[0] + m0[1]) + (m0[2] + m0[3]);
                    out_lp[r * k + c] = offsets[c] - 0.5 * maha_sq;
                }
                r += 1;
            }
        }
        GmmEvalData::Diag { m_scaled, offsets } => {
            let mut r = 0usize;
            while r + 4 <= m {
                let x0 = &x_chunk[r * p..(r + 1) * p];
                let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
                let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
                let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
                for c in 0..k {
                    let pc = &prec_chols[c * p..(c + 1) * p];
                    let ms = &m_scaled[c * p..(c + 1) * p];
                    let mut s0 = [0.0f64; 4];
                    let mut s1 = [0.0f64; 4];
                    let mut s2 = [0.0f64; 4];
                    let mut s3 = [0.0f64; 4];
                    let (pc_c, pc_r) = pc.as_chunks::<4>();
                    let (ms_c, ms_r) = ms.as_chunks::<4>();
                    let (x0_c, x0_r) = x0.as_chunks::<4>();
                    let (x1_c, x1_r) = x1.as_chunks::<4>();
                    let (x2_c, x2_r) = x2.as_chunks::<4>();
                    let (x3_c, x3_r) = x3.as_chunks::<4>();
                    for (((((&p_v, &m_v), &c0), &c1), &c2), &c3) in pc_c
                        .iter()
                        .zip(ms_c.iter())
                        .zip(x0_c.iter())
                        .zip(x1_c.iter())
                        .zip(x2_c.iter())
                        .zip(x3_c.iter())
                    {
                        for l in 0..4 {
                            let d0 = c0[l] * p_v[l] - m_v[l];
                            let d1 = c1[l] * p_v[l] - m_v[l];
                            let d2 = c2[l] * p_v[l] - m_v[l];
                            let d3 = c3[l] * p_v[l] - m_v[l];
                            s0[l] += d0 * d0;
                            s1[l] += d1 * d1;
                            s2[l] += d2 * d2;
                            s3[l] += d3 * d3;
                        }
                    }
                    let mut m0 = (s0[0] + s0[1]) + (s0[2] + s0[3]);
                    let mut m1 = (s1[0] + s1[1]) + (s1[2] + s1[3]);
                    let mut m2 = (s2[0] + s2[1]) + (s2[2] + s2[3]);
                    let mut m3 = (s3[0] + s3[1]) + (s3[2] + s3[3]);
                    for j in 0..pc_r.len() {
                        let d0 = x0_r[j] * pc_r[j] - ms_r[j];
                        let d1 = x1_r[j] * pc_r[j] - ms_r[j];
                        let d2 = x2_r[j] * pc_r[j] - ms_r[j];
                        let d3 = x3_r[j] * pc_r[j] - ms_r[j];
                        m0 += d0 * d0;
                        m1 += d1 * d1;
                        m2 += d2 * d2;
                        m3 += d3 * d3;
                    }
                    let off = offsets[c];
                    out_lp[r * k + c] = off - 0.5 * m0;
                    out_lp[(r + 1) * k + c] = off - 0.5 * m1;
                    out_lp[(r + 2) * k + c] = off - 0.5 * m2;
                    out_lp[(r + 3) * k + c] = off - 0.5 * m3;
                }
                r += 4;
            }
            while r < m {
                let x_row = &x_chunk[r * p..(r + 1) * p];
                for c in 0..k {
                    let pc = &prec_chols[c * p..(c + 1) * p];
                    let ms = &m_scaled[c * p..(c + 1) * p];
                    let mut maha_sq = 0.0;
                    for j in 0..p {
                        let d = x_row[j] * pc[j] - ms[j];
                        maha_sq += d * d;
                    }
                    out_lp[r * k + c] = offsets[c] - 0.5 * maha_sq;
                }
                r += 1;
            }
        }
        GmmEvalData::Spherical { prec_sq, offsets } => {
            let mut r = 0usize;
            while r + 4 <= m {
                let x0 = &x_chunk[r * p..(r + 1) * p];
                let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
                let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
                let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
                for c in 0..k {
                    let mu = &means_data[c * p..(c + 1) * p];
                    let mut s0 = [0.0f64; 4];
                    let mut s1 = [0.0f64; 4];
                    let mut s2 = [0.0f64; 4];
                    let mut s3 = [0.0f64; 4];
                    let (mu_c, mu_r) = mu.as_chunks::<4>();
                    let (x0_c, x0_r) = x0.as_chunks::<4>();
                    let (x1_c, x1_r) = x1.as_chunks::<4>();
                    let (x2_c, x2_r) = x2.as_chunks::<4>();
                    let (x3_c, x3_r) = x3.as_chunks::<4>();
                    for ((((&m_v, &c0), &c1), &c2), &c3) in mu_c
                        .iter()
                        .zip(x0_c.iter())
                        .zip(x1_c.iter())
                        .zip(x2_c.iter())
                        .zip(x3_c.iter())
                    {
                        for l in 0..4 {
                            let d0 = c0[l] - m_v[l];
                            let d1 = c1[l] - m_v[l];
                            let d2 = c2[l] - m_v[l];
                            let d3 = c3[l] - m_v[l];
                            s0[l] += d0 * d0;
                            s1[l] += d1 * d1;
                            s2[l] += d2 * d2;
                            s3[l] += d3 * d3;
                        }
                    }
                    let mut m0 = (s0[0] + s0[1]) + (s0[2] + s0[3]);
                    let mut m1 = (s1[0] + s1[1]) + (s1[2] + s1[3]);
                    let mut m2 = (s2[0] + s2[1]) + (s2[2] + s2[3]);
                    let mut m3 = (s3[0] + s3[1]) + (s3[2] + s3[3]);
                    for j in 0..mu_r.len() {
                        let d0 = x0_r[j] - mu_r[j];
                        let d1 = x1_r[j] - mu_r[j];
                        let d2 = x2_r[j] - mu_r[j];
                        let d3 = x3_r[j] - mu_r[j];
                        m0 += d0 * d0;
                        m1 += d1 * d1;
                        m2 += d2 * d2;
                        m3 += d3 * d3;
                    }
                    let half_psq = 0.5 * prec_sq[c];
                    let off = offsets[c];
                    out_lp[r * k + c] = off - half_psq * m0;
                    out_lp[(r + 1) * k + c] = off - half_psq * m1;
                    out_lp[(r + 2) * k + c] = off - half_psq * m2;
                    out_lp[(r + 3) * k + c] = off - half_psq * m3;
                }
                r += 4;
            }
            while r < m {
                let x_row = &x_chunk[r * p..(r + 1) * p];
                for c in 0..k {
                    let mu = &means_data[c * p..(c + 1) * p];
                    let mut d2 = 0.0;
                    for j in 0..p {
                        let d = x_row[j] - mu[j];
                        d2 += d * d;
                    }
                    out_lp[r * k + c] = offsets[c] - 0.5 * prec_sq[c] * d2;
                }
                r += 1;
            }
        }
    }
}

fn compute_xtx_upper_cpu(x: &Matrix) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;
    let accum_chunk = |x_chunk: &[f64], acc: &mut [f64]| {
        let m = x_chunk.len() / p;
        for i in 0..m {
            let x_row = &x_chunk[i * p..(i + 1) * p];
            for a in 0..p {
                let xa = x_row[a];
                let dst = &mut acc[a * p..a * p + p];
                for b in a..p {
                    dst[b] += xa * x_row[b];
                }
            }
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        x.data
            .par_chunks(GMM_CHUNK_ROWS * p)
            .fold(
                || vec![0.0; p * p],
                |mut acc, x_chunk| {
                    accum_chunk(x_chunk, &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0.0; p * p],
                |mut a, b| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        let mut acc = vec![0.0; p * p];
        accum_chunk(&x.data, &mut acc);
        acc
    }
}

struct EmThreadAcc {
    resp_buf: Vec<f64>,
    xu_scratch: Vec<f64>,
    log_prob_sum: f64,
    nk: Vec<f64>,
    xk: Vec<f64>,
    sk: Vec<f64>,
}

impl EmThreadAcc {
    fn new(k: usize, p: usize, sk_len: usize) -> Self {
        Self {
            resp_buf: vec![0.0; GMM_CHUNK_ROWS * k],
            xu_scratch: Vec::with_capacity(4 * p),
            log_prob_sum: 0.0,
            nk: vec![0.0; k],
            xk: vec![0.0; k * p],
            sk: vec![0.0; sk_len],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_em_step_cpu(
    x: &Matrix,
    weights: &[f64],
    means: &Matrix,
    prec_chols: &[f64],
    covariance_type: &str,
    reg_covar: f64,
    precomputed_xtx: Option<&[f64]>,
) -> OchreResult<(f64, GaussianParamsEstimate)> {
    let n = x.rows;
    let p = x.cols;
    let k = weights.len();
    let eval = prepare_gmm_eval_data(Some(weights), means, prec_chols, covariance_type)?;
    let sk_len = match covariance_type {
        "full" => k * p * p,
        "tied" => 0,
        "diag" => k * p,
        "spherical" => k,
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported covariance_type '{}'",
                covariance_type
            )));
        }
    };
    let means_data = &means.data;

    let process_chunk = |acc: &mut EmThreadAcc, x_chunk: &[f64]| {
        let m = x_chunk.len() / p;
        acc.resp_buf.resize(m * k, 0.0);
        eval_log_prob_chunk(
            &eval,
            x_chunk,
            m,
            p,
            k,
            means_data,
            prec_chols,
            &mut acc.resp_buf,
            &mut acc.xu_scratch,
        );

        // In-place softmax + log_prob_sum so resp stays in L1/L2 cache
        for i in 0..m {
            let row = &mut acc.resp_buf[i * k..(i + 1) * k];
            let mut max_v = f64::NEG_INFINITY;
            for &v in row.iter() {
                if v > max_v {
                    max_v = v;
                }
            }
            if max_v == f64::NEG_INFINITY {
                acc.log_prob_sum += f64::NEG_INFINITY;
                row.fill(0.0);
            } else {
                let mut sum_exp = 0.0;
                for v in row.iter_mut() {
                    let e = (*v - max_v).exp();
                    *v = e;
                    sum_exp += e;
                }
                acc.log_prob_sum += max_v + sum_exp.ln();
                let inv_sum = 1.0 / sum_exp;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            }
        }

        // Accumulate nk and xk from cache-hot resp_buf
        for i in 0..m {
            let r_row = &acc.resp_buf[i * k..(i + 1) * k];
            let x_row = &x_chunk[i * p..(i + 1) * p];
            for (c, &r) in r_row.iter().enumerate().take(k) {
                acc.nk[c] += r;
                if r > 0.0 {
                    let xk_c = &mut acc.xk[c * p..(c + 1) * p];
                    for j in 0..p {
                        xk_c[j] += r * x_row[j];
                    }
                }
            }
        }

        // Accumulate sk from cache-hot resp_buf
        match covariance_type {
            "full" => {
                for c in 0..k {
                    let sk_c = &mut acc.sk[c * p * p..(c + 1) * p * p];
                    for i in 0..m {
                        let r = acc.resp_buf[i * k + c];
                        if r <= 0.0 {
                            continue;
                        }
                        let x_row = &x_chunk[i * p..(i + 1) * p];
                        for a in 0..p {
                            let da = r * x_row[a];
                            let dst = &mut sk_c[a * p..(a + 1) * p];
                            for b in a..p {
                                dst[b] += da * x_row[b];
                            }
                        }
                    }
                }
            }
            "diag" => {
                for i in 0..m {
                    let r_row = &acc.resp_buf[i * k..(i + 1) * k];
                    let x_row = &x_chunk[i * p..(i + 1) * p];
                    for (c, &r) in r_row.iter().enumerate().take(k) {
                        if r <= 0.0 {
                            continue;
                        }
                        let sk_c = &mut acc.sk[c * p..(c + 1) * p];
                        for j in 0..p {
                            let xj = x_row[j];
                            sk_c[j] += r * xj * xj;
                        }
                    }
                }
            }
            "spherical" => {
                for i in 0..m {
                    let x_row = &x_chunk[i * p..(i + 1) * p];
                    let norm2 = dot_slice(x_row, x_row);
                    let r_row = &acc.resp_buf[i * k..(i + 1) * k];
                    for (c, &r) in r_row.iter().enumerate().take(k) {
                        acc.sk[c] += r * norm2;
                    }
                }
            }
            _ => {}
        }
    };

    let merged = if n >= PARALLEL_ROW_THRESHOLD || n * k * p >= 4096 {
        x.data
            .par_chunks(GMM_CHUNK_ROWS * p)
            .fold(
                || EmThreadAcc::new(k, p, sk_len),
                |mut acc, x_chunk| {
                    process_chunk(&mut acc, x_chunk);
                    acc
                },
            )
            .reduce(
                || EmThreadAcc::new(k, p, sk_len),
                |mut a, b| {
                    a.log_prob_sum += b.log_prob_sum;
                    for c in 0..k {
                        a.nk[c] += b.nk[c];
                    }
                    for i in 0..a.xk.len() {
                        a.xk[i] += b.xk[i];
                    }
                    for i in 0..a.sk.len() {
                        a.sk[i] += b.sk[i];
                    }
                    a
                },
            )
    } else {
        let mut acc = EmThreadAcc::new(k, p, sk_len);
        for x_chunk in x.data.chunks(GMM_CHUNK_ROWS * p) {
            process_chunk(&mut acc, x_chunk);
        }
        acc
    };

    let lower_bound = merged.log_prob_sum / (n as f64);
    let r_sum = merged.nk;
    let eps_pad = 10.0 * f64::EPSILON;
    let mut nk = Vec::with_capacity(k);
    let mut nk_sum = 0.0;
    for &rs in &r_sum {
        let v = rs + eps_pad;
        nk.push(v);
        nk_sum += v;
    }
    let new_weights: Vec<f64> = nk.iter().map(|&v| v / nk_sum).collect();

    let mut means_data_new = vec![0.0; k * p];
    for c in 0..k {
        let inv = 1.0 / nk[c];
        for j in 0..p {
            means_data_new[c * p + j] = merged.xk[c * p + j] * inv;
        }
    }
    let new_means = Matrix::from_vec(k, p, means_data_new)?;

    let (covariances, precisions_cholesky) = match covariance_type {
        "full" => {
            let mut covs = merged.sk;
            let mut prec_chols = vec![0.0; k * p * p];
            for c in 0..k {
                let inv_nk = 1.0 / nk[c];
                let rc = r_sum[c];
                let m_c = &new_means.data[c * p..(c + 1) * p];
                let xk_c = &merged.xk[c * p..(c + 1) * p];
                let cov_c = &mut covs[c * p * p..(c + 1) * p * p];
                for a in 0..p {
                    for b in a..p {
                        let raw = cov_c[a * p + b] - xk_c[a] * m_c[b] - xk_c[b] * m_c[a]
                            + rc * m_c[a] * m_c[b];
                        let mut val = raw * inv_nk;
                        if a == b {
                            val = val.max(0.0) + reg_covar;
                        }
                        cov_c[a * p + b] = val;
                        cov_c[b * p + a] = val;
                    }
                }
                let u = compute_prec_chol_from_cov(cov_c, p)?;
                prec_chols[c * p * p..(c + 1) * p * p].copy_from_slice(&u);
            }
            (covs, prec_chols)
        }
        "tied" => {
            let mut cov = match precomputed_xtx {
                Some(xtx) => xtx.to_vec(),
                None => compute_xtx_upper_cpu(x),
            };
            for (c, &rc) in r_sum.iter().enumerate().take(k) {
                let m_c = &new_means.data[c * p..(c + 1) * p];
                let xk_c = &merged.xk[c * p..(c + 1) * p];
                for a in 0..p {
                    for b in a..p {
                        cov[a * p + b] +=
                            -xk_c[a] * m_c[b] - xk_c[b] * m_c[a] + rc * m_c[a] * m_c[b];
                    }
                }
            }
            let inv_nk_sum = 1.0 / nk_sum;
            for a in 0..p {
                for b in a..p {
                    let mut val = cov[a * p + b] * inv_nk_sum;
                    if a == b {
                        val = val.max(0.0) + reg_covar;
                    }
                    cov[a * p + b] = val;
                    cov[b * p + a] = val;
                }
            }
            let prec_chol = compute_prec_chol_from_cov(&cov, p)?;
            (cov, prec_chol)
        }
        "diag" => {
            let mut covs = merged.sk;
            let mut prec_chols = vec![0.0; k * p];
            for c in 0..k {
                let inv_nk = 1.0 / nk[c];
                let rc = r_sum[c];
                let m_c = &new_means.data[c * p..(c + 1) * p];
                let xk_c = &merged.xk[c * p..(c + 1) * p];
                let cov_c = &mut covs[c * p..(c + 1) * p];
                for j in 0..p {
                    let raw = cov_c[j] - 2.0 * xk_c[j] * m_c[j] + rc * m_c[j] * m_c[j];
                    let v = (raw * inv_nk).max(0.0) + reg_covar;
                    cov_c[j] = v;
                    if !v.is_finite() || v <= 0.0 {
                        return Err(OchreError::SingularMatrix(
                            "Diagonal covariance is non-positive; try increasing reg_covar"
                                .to_string(),
                        ));
                    }
                    prec_chols[c * p + j] = 1.0 / v.sqrt();
                }
            }
            (covs, prec_chols)
        }
        "spherical" => {
            let mut covs = vec![0.0; k];
            let mut prec_chols = vec![0.0; k];
            let inv_p = 1.0 / (p as f64);
            for c in 0..k {
                let inv_nk = 1.0 / nk[c];
                let rc = r_sum[c];
                let m_c = &new_means.data[c * p..(c + 1) * p];
                let xk_c = &merged.xk[c * p..(c + 1) * p];
                let dot_xk_m = dot_slice(xk_c, m_c);
                let dot_m_m = dot_slice(m_c, m_c);
                let sum_sq = (merged.sk[c] - 2.0 * dot_xk_m + rc * dot_m_m).max(0.0);
                let v = sum_sq * inv_nk * inv_p + reg_covar;
                if !v.is_finite() || v <= 0.0 {
                    return Err(OchreError::SingularMatrix(
                        "Spherical covariance is non-positive; try increasing reg_covar"
                            .to_string(),
                    ));
                }
                covs[c] = v;
                prec_chols[c] = 1.0 / v.sqrt();
            }
            (covs, prec_chols)
        }
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported covariance_type '{}'",
                covariance_type
            )));
        }
    };

    Ok((
        lower_bound,
        (new_weights, new_means, covariances, precisions_cholesky),
    ))
}

fn init_responsibilities_random(
    n_samples: usize,
    n_components: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
    let mut resp = vec![0.0; n_samples * n_components];
    for i in 0..n_samples {
        let row = &mut resp[i * n_components..(i + 1) * n_components];
        let mut sum = 0.0;
        for val in row.iter_mut() {
            let v = rng.next_f64().max(1e-12);
            *val = v;
            sum += v;
        }
        let inv = 1.0 / sum;
        for val in row.iter_mut() {
            *val *= inv;
        }
    }
    Matrix::from_vec(n_samples, n_components, resp)
}

type GaussianParamsEstimate = (Vec<f64>, Matrix, Vec<f64>, Vec<f64>);

fn estimate_gaussian_parameters(
    backend: &Backend<'_>,
    x: &Matrix,
    resp: &Matrix,
    covariance_type: &str,
    reg_covar: f64,
) -> OchreResult<GaussianParamsEstimate> {
    let n = x.rows;
    let p = x.cols;
    let k = resp.cols;

    let mut nk = vec![0.0; k];
    for i in 0..n {
        let r_row = &resp.data[i * k..(i + 1) * k];
        for c in 0..k {
            nk[c] += r_row[c];
        }
    }
    let eps_pad = 10.0 * f64::EPSILON;
    let mut nk_sum = 0.0;
    for val in &mut nk {
        *val += eps_pad;
        nk_sum += *val;
    }

    let weights: Vec<f64> = nk.iter().map(|&v| v / nk_sum).collect();

    // means = (resp^T * X) / nk[:, None]
    let mut means = backend.matmul_at_b(resp, x)?;
    for (c, &nk_c) in nk.iter().enumerate().take(k) {
        let inv = 1.0 / nk_c;
        for j in 0..p {
            means.data[c * p + j] *= inv;
        }
    }

    let (covariances, precisions_cholesky) = match covariance_type {
        "full" => {
            let mut covs = vec![0.0; k * p * p];
            let mut prec_chols = vec![0.0; k * p * p];
            for c in 0..k {
                let cov_c = &mut covs[c * p * p..(c + 1) * p * p];
                let mean_c = &means.data[c * p..(c + 1) * p];
                let inv_nk = 1.0 / nk[c];

                for i in 0..n {
                    let r = resp.data[i * k + c];
                    if r <= 0.0 {
                        continue;
                    }
                    let x_row = &x.data[i * p..(i + 1) * p];
                    for a in 0..p {
                        let da = r * (x_row[a] - mean_c[a]);
                        for b in a..p {
                            cov_c[a * p + b] += da * (x_row[b] - mean_c[b]);
                        }
                    }
                }
                for a in 0..p {
                    for b in a..p {
                        let val = cov_c[a * p + b] * inv_nk + if a == b { reg_covar } else { 0.0 };
                        cov_c[a * p + b] = val;
                        cov_c[b * p + a] = val;
                    }
                }
                let u = compute_prec_chol_from_cov(cov_c, p)?;
                prec_chols[c * p * p..(c + 1) * p * p].copy_from_slice(&u);
            }
            (covs, prec_chols)
        }
        "tied" => {
            let mut cov = vec![0.0; p * p];
            let inv_nk_sum = 1.0 / nk_sum;
            for c in 0..k {
                let mean_c = &means.data[c * p..(c + 1) * p];
                for i in 0..n {
                    let r = resp.data[i * k + c];
                    if r <= 0.0 {
                        continue;
                    }
                    let x_row = &x.data[i * p..(i + 1) * p];
                    for a in 0..p {
                        let da = r * (x_row[a] - mean_c[a]);
                        for b in a..p {
                            cov[a * p + b] += da * (x_row[b] - mean_c[b]);
                        }
                    }
                }
            }
            for a in 0..p {
                for b in a..p {
                    let val = cov[a * p + b] * inv_nk_sum + if a == b { reg_covar } else { 0.0 };
                    cov[a * p + b] = val;
                    cov[b * p + a] = val;
                }
            }
            let prec_chol = compute_prec_chol_from_cov(&cov, p)?;
            (cov, prec_chol)
        }
        "diag" => {
            let mut covs = vec![0.0; k * p];
            let mut prec_chols = vec![0.0; k * p];
            for c in 0..k {
                let cov_c = &mut covs[c * p..(c + 1) * p];
                let mean_c = &means.data[c * p..(c + 1) * p];
                let inv_nk = 1.0 / nk[c];
                for i in 0..n {
                    let r = resp.data[i * k + c];
                    if r <= 0.0 {
                        continue;
                    }
                    let x_row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        let d = x_row[j] - mean_c[j];
                        cov_c[j] += r * d * d;
                    }
                }
                for j in 0..p {
                    let v = cov_c[j] * inv_nk + reg_covar;
                    cov_c[j] = v;
                    if !v.is_finite() || v <= 0.0 {
                        return Err(OchreError::SingularMatrix(
                            "Diagonal covariance is non-positive; try increasing reg_covar"
                                .to_string(),
                        ));
                    }
                    prec_chols[c * p + j] = 1.0 / v.sqrt();
                }
            }
            (covs, prec_chols)
        }
        "spherical" => {
            let mut covs = vec![0.0; k];
            let mut prec_chols = vec![0.0; k];
            let inv_p = 1.0 / (p as f64);
            for c in 0..k {
                let mean_c = &means.data[c * p..(c + 1) * p];
                let inv_nk = 1.0 / nk[c];
                let mut sum_sq = 0.0;
                for i in 0..n {
                    let r = resp.data[i * k + c];
                    if r <= 0.0 {
                        continue;
                    }
                    let x_row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        let d = x_row[j] - mean_c[j];
                        sum_sq += r * d * d;
                    }
                }
                let v = sum_sq * inv_nk * inv_p + reg_covar;
                if !v.is_finite() || v <= 0.0 {
                    return Err(OchreError::SingularMatrix(
                        "Spherical covariance is non-positive; try increasing reg_covar"
                            .to_string(),
                    ));
                }
                covs[c] = v;
                prec_chols[c] = 1.0 / v.sqrt();
            }
            (covs, prec_chols)
        }
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported covariance_type '{}'",
                covariance_type
            )));
        }
    };

    Ok((weights, means, covariances, precisions_cholesky))
}

fn estimate_log_gaussian_prob(
    backend: &Backend<'_>,
    x: &Matrix,
    means: &Matrix,
    precisions_cholesky: &[f64],
    covariance_type: &str,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let k = means.rows;
    let const_term = (p as f64) * (2.0 * PI).ln();
    let mut log_prob = vec![0.0; n * k];

    match covariance_type {
        "full" => {
            for c in 0..k {
                let u_c =
                    Matrix::from_slice(p, p, &precisions_cholesky[c * p * p..(c + 1) * p * p])?;
                let mut log_det = 0.0;
                for j in 0..p {
                    log_det += u_c.data[j * p + j].ln();
                }

                // y = X * U_c - mu_c * U_c
                let x_u = backend.matmul(x, &u_c)?;
                let mu_mat = Matrix::from_slice(1, p, &means.data[c * p..(c + 1) * p])?;
                let mu_u = backend.matmul(&mu_mat, &u_c)?;

                for i in 0..n {
                    let row = &x_u.data[i * p..(i + 1) * p];
                    let mut maha_sq = 0.0;
                    for (j, &rj) in row.iter().enumerate().take(p) {
                        let d = rj - mu_u.data[j];
                        maha_sq += d * d;
                    }
                    log_prob[i * k + c] = -0.5 * (const_term + maha_sq) + log_det;
                }
            }
        }
        "tied" => {
            let u = Matrix::from_slice(p, p, precisions_cholesky)?;
            let mut log_det = 0.0;
            for j in 0..p {
                log_det += u.data[j * p + j].ln();
            }
            let x_u = backend.matmul(x, &u)?;
            let mu_u = backend.matmul(means, &u)?;

            log_prob
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(i, lp_row)| {
                    let x_row = &x_u.data[i * p..(i + 1) * p];
                    for (c, lp_val) in lp_row.iter_mut().enumerate() {
                        let m_row = &mu_u.data[c * p..(c + 1) * p];
                        let mut maha_sq = 0.0;
                        for j in 0..p {
                            let d = x_row[j] - m_row[j];
                            maha_sq += d * d;
                        }
                        *lp_val = -0.5 * (const_term + maha_sq) + log_det;
                    }
                });
        }
        "diag" => {
            let mut log_dets = vec![0.0; k];
            for c in 0..k {
                let pc = &precisions_cholesky[c * p..(c + 1) * p];
                log_dets[c] = pc.iter().map(|v| v.ln()).sum();
            }
            log_prob
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(i, lp_row)| {
                    let x_row = &x.data[i * p..(i + 1) * p];
                    for c in 0..k {
                        let m_row = &means.data[c * p..(c + 1) * p];
                        let pc = &precisions_cholesky[c * p..(c + 1) * p];
                        let mut maha_sq = 0.0;
                        for j in 0..p {
                            let d = (x_row[j] - m_row[j]) * pc[j];
                            maha_sq += d * d;
                        }
                        lp_row[c] = -0.5 * (const_term + maha_sq) + log_dets[c];
                    }
                });
        }
        "spherical" => {
            let sq_dists = if n.saturating_mul(k) <= MAX_DIST_ELEMENTS {
                backend.pairwise_l2_sq(x, means)?.data
            } else {
                let mut all = Vec::with_capacity(n * k);
                let chunk_rows = (MAX_DIST_ELEMENTS / k.max(1)).max(1).min(n);
                let mut start = 0;
                while start < n {
                    let end = (start + chunk_rows).min(n);
                    let chunk = Matrix::from_slice(end - start, p, &x.data[start * p..end * p])?;
                    all.extend(backend.pairwise_l2_sq(&chunk, means)?.data);
                    start = end;
                }
                all
            };

            let log_dets: Vec<f64> = precisions_cholesky
                .iter()
                .map(|&pc| (p as f64) * pc.ln())
                .collect();
            let prec_sq: Vec<f64> = precisions_cholesky.iter().map(|&pc| pc * pc).collect();

            log_prob
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(i, lp_row)| {
                    let d_row = &sq_dists[i * k..(i + 1) * k];
                    for c in 0..k {
                        let maha_sq = d_row[c].max(0.0) * prec_sq[c];
                        lp_row[c] = -0.5 * (const_term + maha_sq) + log_dets[c];
                    }
                });
        }
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Unsupported covariance_type '{}'",
                covariance_type
            )));
        }
    }

    Matrix::from_vec(n, k, log_prob)
}

fn estimate_weighted_log_prob(
    backend: &Backend<'_>,
    x: &Matrix,
    weights: &[f64],
    means: &Matrix,
    precisions_cholesky: &[f64],
    covariance_type: &str,
    check_finite: bool,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let k = weights.len();
    if matches!(backend, Backend::Cpu) {
        let eval =
            prepare_gmm_eval_data(Some(weights), means, precisions_cholesky, covariance_type)?;
        let mut wlp = vec![0.0; n * k];
        let means_data = &means.data;
        let has_non_finite = AtomicBool::new(false);
        let nb = p.div_ceil(4);
        let scratch_cap = 4 * nb * 4;
        if n >= PARALLEL_ROW_THRESHOLD {
            wlp.par_chunks_mut(GMM_PREDICT_CHUNK_ROWS * k)
                .zip(x.data.par_chunks(GMM_PREDICT_CHUNK_ROWS * p))
                .for_each_init(
                    || Vec::with_capacity(scratch_cap),
                    |xu_scratch, (out_chunk, x_chunk)| {
                        if check_finite {
                            if has_non_finite.load(Ordering::Relaxed) {
                                return;
                            }
                            if !slice_all_finite(x_chunk) {
                                has_non_finite.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                        let m = out_chunk.len() / k;
                        eval_log_prob_chunk(
                            &eval,
                            x_chunk,
                            m,
                            p,
                            k,
                            means_data,
                            precisions_cholesky,
                            out_chunk,
                            xu_scratch,
                        );
                    },
                );
            if check_finite && has_non_finite.load(Ordering::Relaxed) {
                return Err(OchreError::InvalidParameter(
                    "Input X contains NaN or infinite values ( >_< )".to_string(),
                ));
            }
        } else {
            let mut xu_scratch = Vec::with_capacity(scratch_cap);
            for (out_chunk, x_chunk) in wlp
                .chunks_mut(GMM_PREDICT_CHUNK_ROWS * k)
                .zip(x.data.chunks(GMM_PREDICT_CHUNK_ROWS * p))
            {
                if check_finite && !slice_all_finite(x_chunk) {
                    return Err(OchreError::InvalidParameter(
                        "Input X contains NaN or infinite values ( >_< )".to_string(),
                    ));
                }
                let m = out_chunk.len() / k;
                eval_log_prob_chunk(
                    &eval,
                    x_chunk,
                    m,
                    p,
                    k,
                    means_data,
                    precisions_cholesky,
                    out_chunk,
                    &mut xu_scratch,
                );
            }
        }
        return Matrix::from_vec(n, k, wlp);
    }

    if check_finite {
        validate_x(x)?;
    }
    let mut lp =
        estimate_log_gaussian_prob(backend, x, means, precisions_cholesky, covariance_type)?;
    let log_w: Vec<f64> = weights
        .iter()
        .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
        .collect();
    for row in lp.data.chunks_mut(k) {
        for c in 0..k {
            row[c] += log_w[c];
        }
    }
    Ok(lp)
}

#[allow(clippy::too_many_arguments)]
fn e_step(
    backend: &Backend<'_>,
    x: &Matrix,
    weights: &[f64],
    means: &Matrix,
    precisions_cholesky: &[f64],
    covariance_type: &str,
    check_finite: bool,
    normalize_to_probs: bool,
) -> OchreResult<(Vec<f64>, Matrix)> {
    let n = x.rows;
    let k = weights.len();
    let mut wlp = estimate_weighted_log_prob(
        backend,
        x,
        weights,
        means,
        precisions_cholesky,
        covariance_type,
        check_finite,
    )?;

    if normalize_to_probs {
        let normalize_row = |row: &mut [f64]| {
            let mut max_v = f64::NEG_INFINITY;
            for &v in row.iter() {
                if v > max_v {
                    max_v = v;
                }
            }
            if max_v == f64::NEG_INFINITY {
                row.fill(0.0);
            } else {
                let mut sum_exp = 0.0;
                for v in row.iter_mut() {
                    let e = (*v - max_v).exp();
                    *v = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            }
        };
        if n >= PARALLEL_ROW_THRESHOLD {
            wlp.data.par_chunks_exact_mut(k).for_each(normalize_row);
        } else {
            wlp.data.chunks_exact_mut(k).for_each(normalize_row);
        }
        return Ok((Vec::new(), wlp));
    }

    let mut log_prob_norm = vec![0.0; n];
    let compute_lse = |(lpn, row): (&mut f64, &mut [f64])| {
        let mut max_v = f64::NEG_INFINITY;
        for &v in row.iter() {
            if v > max_v {
                max_v = v;
            }
        }
        let lse = if max_v == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            let mut sum_exp = 0.0;
            for &v in row.iter() {
                sum_exp += (v - max_v).exp();
            }
            max_v + sum_exp.ln()
        };
        *lpn = lse;
        for v in row.iter_mut() {
            *v -= lse;
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        log_prob_norm
            .par_iter_mut()
            .zip(wlp.data.par_chunks_exact_mut(k))
            .for_each(compute_lse);
    } else {
        log_prob_norm
            .iter_mut()
            .zip(wlp.data.chunks_exact_mut(k))
            .for_each(compute_lse);
    }

    Ok((log_prob_norm, wlp))
}

#[derive(Debug, Clone)]
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: String,
    pub tol: f64,
    pub reg_covar: f64,
    pub max_iter: usize,
    pub n_init: usize,
    pub init_params: String,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    pub device_type: DeviceType,

    pub weights_: Option<Vec<f64>>,
    pub means_: Option<Matrix>,
    pub covariances_: Option<Vec<f64>>,
    pub precisions_cholesky_: Option<Vec<f64>>,
    pub converged_: bool,
    pub n_iter_: usize,
    pub lower_bound_: Option<f64>,
    pub n_features_in_: Option<usize>,
}

impl GaussianMixture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: usize,
        covariance_type: String,
        tol: f64,
        reg_covar: f64,
        max_iter: usize,
        n_init: usize,
        init_params: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_params(
            n_components,
            &covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            &init_params,
        )?;
        let device_type = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
            random_state,
            device_: Some(device_type.to_string()),
            device_type,
            weights_: None,
            means_: None,
            covariances_: None,
            precisions_cholesky_: None,
            converged_: false,
            n_iter_: 0,
            lower_bound_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dt = parse_device(device.as_deref())?;
        self.device_type = dt;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        if x.rows < self.n_components {
            return Err(OchreError::InvalidParameter(format!(
                "Expected n_samples >= n_components, got n_samples={} < n_components={} (・`ω´・)",
                x.rows, self.n_components
            )));
        }

        let n = x.rows;
        let p = x.cols;
        let k = self.n_components;
        let backend = effective_backend(&self.device_type, n * p);

        // Center X once when reasonably sized so sufficient-statistics M-step never loses precision on large offsets.
        let (x_shift, x_centered_opt) = if n * p <= 10_000_000 {
            let mut col_sum = vec![0.0; p];
            for row in x.data.chunks_exact(p) {
                for j in 0..p {
                    col_sum[j] += row[j];
                }
            }
            let inv_n = 1.0 / (n as f64);
            let shift: Vec<f64> = col_sum.iter().map(|&s| s * inv_n).collect();
            let mut shifted = vec![0.0; n * p];
            for (dst, src) in shifted.chunks_exact_mut(p).zip(x.data.chunks_exact(p)) {
                for j in 0..p {
                    dst[j] = src[j] - shift[j];
                }
            }
            (Some(shift), Some(Matrix::from_vec(n, p, shifted)?))
        } else {
            (None, None)
        };
        let x_work = x_centered_opt.as_ref().unwrap_or(x);

        let precomputed_xtx = if matches!(backend, Backend::Cpu) && self.covariance_type == "tied" {
            Some(compute_xtx_upper_cpu(x_work))
        } else {
            None
        };

        let mut best_lower_bound = f64::NEG_INFINITY;
        type BestGmmParams = (Vec<f64>, Matrix, Vec<f64>, Vec<f64>, bool, usize);
        let mut best_params: Option<BestGmmParams> = None;

        for init_idx in 0..self.n_init {
            let seed = self
                .random_state
                .map(|s| s.wrapping_add((init_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
            let mut rng = DeterministicRng::new(seed);

            let (mut weights, mut means, mut covariances, mut prec_chols) =
                if self.init_params == "kmeans" {
                    let labels = init_kmeans_labels(&backend, x_work, k, &mut rng)?;
                    estimate_gaussian_parameters_from_labels(
                        x_work,
                        &labels,
                        k,
                        &self.covariance_type,
                        self.reg_covar,
                        precomputed_xtx.as_deref(),
                    )?
                } else {
                    let init_resp = init_responsibilities_random(n, k, &mut rng)?;
                    estimate_gaussian_parameters(
                        &backend,
                        x_work,
                        &init_resp,
                        &self.covariance_type,
                        self.reg_covar,
                    )?
                };

            let mut prev_lower_bound = f64::NEG_INFINITY;
            let mut converged = false;
            let mut n_iter = 0;

            for iter in 1..=self.max_iter {
                let (lower_bound, (w_new, m_new, c_new, pc_new)) =
                    if matches!(backend, Backend::Cpu) {
                        fused_em_step_cpu(
                            x_work,
                            &weights,
                            &means,
                            &prec_chols,
                            &self.covariance_type,
                            self.reg_covar,
                            precomputed_xtx.as_deref(),
                        )?
                    } else {
                        let (log_prob_norm, log_resp) = e_step(
                            &backend,
                            x_work,
                            &weights,
                            &means,
                            &prec_chols,
                            &self.covariance_type,
                            false,
                            false,
                        )?;
                        let lb = log_prob_norm.iter().sum::<f64>() / (n as f64);
                        let resp_data: Vec<f64> = log_resp.data.iter().map(|&v| v.exp()).collect();
                        let resp = Matrix::from_vec(n, k, resp_data)?;
                        let params = estimate_gaussian_parameters(
                            &backend,
                            x_work,
                            &resp,
                            &self.covariance_type,
                            self.reg_covar,
                        )?;
                        (lb, params)
                    };
                weights = w_new;
                means = m_new;
                covariances = c_new;
                prec_chols = pc_new;
                n_iter = iter;

                let change = (lower_bound - prev_lower_bound).abs();
                if change < self.tol {
                    converged = true;
                    prev_lower_bound = lower_bound;
                    break;
                }
                prev_lower_bound = lower_bound;
            }

            if best_params.is_none() || prev_lower_bound > best_lower_bound {
                best_lower_bound = prev_lower_bound;
                best_params = Some((weights, means, covariances, prec_chols, converged, n_iter));
            }
        }

        if let Some((weights, mut means, covariances, prec_chols, converged, n_iter)) = best_params
        {
            if let Some(ref shift) = x_shift {
                for row in means.data.chunks_exact_mut(p) {
                    for j in 0..p {
                        row[j] += shift[j];
                    }
                }
            }
            self.weights_ = Some(weights);
            self.means_ = Some(means);
            self.covariances_ = Some(covariances);
            self.precisions_cholesky_ = Some(prec_chols);
            self.converged_ = converged;
            self.n_iter_ = n_iter;
            self.lower_bound_ = Some(best_lower_bound);
            self.n_features_in_ = Some(p);
        }

        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<usize>> {
        self.fit(x)?;
        self.predict(x)
    }

    fn get_fitted(&self) -> OchreResult<(&[f64], &Matrix, &[f64])> {
        let weights = self.weights_.as_deref().ok_or(OchreError::NotFitted)?;
        let means = self.means_.as_ref().ok_or(OchreError::NotFitted)?;
        let prec_chols = self
            .precisions_cholesky_
            .as_deref()
            .ok_or(OchreError::NotFitted)?;
        Ok((weights, means, prec_chols))
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = effective_backend(&self.device_type, x.rows * x.cols);
        let (_, probs) = e_step(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
            true,
            true,
        )?;
        Ok(probs)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<usize>> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = effective_backend(&self.device_type, x.rows * x.cols);
        let k = self.n_components;
        let p = x.cols;
        let n = x.rows;
        if matches!(backend, Backend::Cpu) {
            let eval =
                prepare_gmm_eval_data(Some(weights), means, prec_chols, &self.covariance_type)?;
            let means_data = &means.data;
            let mut labels = vec![0usize; n];
            let has_non_finite = AtomicBool::new(false);
            let nb = p.div_ceil(4);
            let scratch_cap = 4 * nb * 4;
            let predict_chunk = |(wlp_buf, xu_scratch): &mut (Vec<f64>, Vec<f64>),
                                 lbl_chunk: &mut [usize],
                                 x_chunk: &[f64]| {
                if has_non_finite.load(Ordering::Relaxed) {
                    return;
                }
                if !slice_all_finite(x_chunk) {
                    has_non_finite.store(true, Ordering::Relaxed);
                    return;
                }
                let m = lbl_chunk.len();
                wlp_buf.resize(m * k, 0.0);
                eval_log_prob_chunk(
                    &eval, x_chunk, m, p, k, means_data, prec_chols, wlp_buf, xu_scratch,
                );
                for (i, dst) in lbl_chunk.iter_mut().enumerate() {
                    let row = &wlp_buf[i * k..(i + 1) * k];
                    let mut best_c = 0;
                    let mut best_v = f64::NEG_INFINITY;
                    for (c, &v) in row.iter().enumerate() {
                        if v > best_v {
                            best_v = v;
                            best_c = c;
                        }
                    }
                    *dst = best_c;
                }
            };
            if n >= PARALLEL_ROW_THRESHOLD {
                labels
                    .par_chunks_mut(GMM_PREDICT_CHUNK_ROWS)
                    .zip(x.data.par_chunks(GMM_PREDICT_CHUNK_ROWS * p))
                    .for_each_init(
                        || {
                            (
                                Vec::with_capacity(GMM_PREDICT_CHUNK_ROWS * k),
                                Vec::with_capacity(scratch_cap),
                            )
                        },
                        |scratch, (lbl_chunk, x_chunk)| {
                            predict_chunk(scratch, lbl_chunk, x_chunk);
                        },
                    );
            } else {
                let mut scratch = (
                    Vec::with_capacity(GMM_PREDICT_CHUNK_ROWS * k),
                    Vec::with_capacity(scratch_cap),
                );
                for (lbl_chunk, x_chunk) in labels
                    .chunks_mut(GMM_PREDICT_CHUNK_ROWS)
                    .zip(x.data.chunks(GMM_PREDICT_CHUNK_ROWS * p))
                {
                    predict_chunk(&mut scratch, lbl_chunk, x_chunk);
                    if has_non_finite.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
            if has_non_finite.load(Ordering::Relaxed) {
                return Err(OchreError::InvalidParameter(
                    "Input X contains NaN or infinite values ( >_< )".to_string(),
                ));
            }
            return Ok(labels);
        }

        let wlp = estimate_weighted_log_prob(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
            true,
        )?;
        let mut labels = Vec::with_capacity(x.rows);
        for row in wlp.data.chunks(k) {
            let mut best_c = 0;
            let mut best_v = f64::NEG_INFINITY;
            for (c, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            labels.push(best_c);
        }
        Ok(labels)
    }

    pub fn score_samples(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = effective_backend(&self.device_type, x.rows * x.cols);
        let (log_prob_norm, _) = e_step(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
            true,
            false,
        )?;
        Ok(log_prob_norm)
    }

    pub fn score(&self, x: &Matrix) -> OchreResult<f64> {
        let scores = self.score_samples(x)?;
        Ok(scores.iter().sum::<f64>() / (scores.len() as f64))
    }

    fn n_parameters(&self, n_features: usize) -> usize {
        let k = self.n_components;
        let cov_params = match self.covariance_type.as_str() {
            "full" => k * n_features * (n_features + 1) / 2,
            "tied" => n_features * (n_features + 1) / 2,
            "diag" => k * n_features,
            "spherical" => k,
            _ => 0,
        };
        let mean_params = n_features * k;
        cov_params + mean_params + (k - 1)
    }

    pub fn aic(&self, x: &Matrix) -> OchreResult<f64> {
        let mean_ll = self.score(x)?;
        let p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let n_params = self.n_parameters(p) as f64;
        Ok(-2.0 * mean_ll * (x.rows as f64) + 2.0 * n_params)
    }

    pub fn bic(&self, x: &Matrix) -> OchreResult<f64> {
        let mean_ll = self.score(x)?;
        let p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let n_params = self.n_parameters(p) as f64;
        let n = x.rows as f64;
        Ok(-2.0 * mean_ll * n + n_params * n.ln())
    }

    pub fn sample(&self, n_samples: usize) -> OchreResult<(Matrix, Vec<usize>)> {
        if n_samples == 0 {
            return Err(OchreError::InvalidParameter(
                "n_samples must be >= 1 (・`ω´・)".to_string(),
            ));
        }
        let weights = self.weights_.as_deref().ok_or(OchreError::NotFitted)?;
        let means = self.means_.as_ref().ok_or(OchreError::NotFitted)?;
        let covs = self.covariances_.as_deref().ok_or(OchreError::NotFitted)?;
        let p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let k = self.n_components;

        let mut rng = DeterministicRng::new(self.random_state);

        // Build cumulative weights for categorical sampling
        let mut cdf = Vec::with_capacity(k);
        let mut acc = 0.0;
        for &w in weights {
            acc += w;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let mut counts = vec![0usize; k];
        for _ in 0..n_samples {
            let u = rng.next_f64();
            let mut chosen = k - 1;
            for (c, &thresh) in cdf.iter().enumerate() {
                if u <= thresh {
                    chosen = c;
                    break;
                }
            }
            counts[chosen] += 1;
        }

        // Precompute Cholesky factors for full / tied
        let chol_factors: Vec<Vec<f64>> = match self.covariance_type.as_str() {
            "full" => (0..k)
                .map(|c| cholesky_lower(&covs[c * p * p..(c + 1) * p * p], p))
                .collect::<OchreResult<Vec<_>>>()?,
            "tied" => vec![cholesky_lower(covs, p)?],
            _ => Vec::new(),
        };

        let mut x_out = vec![0.0; n_samples * p];
        let mut y_out = Vec::with_capacity(n_samples);
        let mut z_buf = vec![0.0; p];
        let mut row_idx = 0;

        for (c, &count) in counts.iter().enumerate() {
            let mean_c = &means.data[c * p..(c + 1) * p];
            for _ in 0..count {
                let mut j = 0;
                while j < p {
                    let (z0, z1) = rng.next_normal_pair();
                    z_buf[j] = z0;
                    if j + 1 < p {
                        z_buf[j + 1] = z1;
                    }
                    j += 2;
                }

                let dst = &mut x_out[row_idx * p..(row_idx + 1) * p];
                match self.covariance_type.as_str() {
                    "full" => {
                        let l = &chol_factors[c];
                        for a in 0..p {
                            let mut val = mean_c[a];
                            for b in 0..=a {
                                val += l[a * p + b] * z_buf[b];
                            }
                            dst[a] = val;
                        }
                    }
                    "tied" => {
                        let l = &chol_factors[0];
                        for a in 0..p {
                            let mut val = mean_c[a];
                            for b in 0..=a {
                                val += l[a * p + b] * z_buf[b];
                            }
                            dst[a] = val;
                        }
                    }
                    "diag" => {
                        let cov_c = &covs[c * p..(c + 1) * p];
                        for a in 0..p {
                            dst[a] = mean_c[a] + cov_c[a].sqrt() * z_buf[a];
                        }
                    }
                    "spherical" => {
                        let std_c = covs[c].sqrt();
                        for a in 0..p {
                            dst[a] = mean_c[a] + std_c * z_buf[a];
                        }
                    }
                    _ => {}
                }
                y_out.push(c);
                row_idx += 1;
            }
        }

        Ok((Matrix::from_vec(n_samples, p, x_out)?, y_out))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GAUSSIAN_MIXTURE);
        write_usize(&mut buf, self.n_components);
        write_str(&mut buf, &self.covariance_type);
        write_f64(&mut buf, self.tol);
        write_f64(&mut buf, self.reg_covar);
        write_usize(&mut buf, self.max_iter);
        write_usize(&mut buf, self.n_init);
        write_str(&mut buf, &self.init_params);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);

        write_opt_f64_vec(&mut buf, &self.weights_);
        write_opt_matrix(&mut buf, &self.means_);
        write_opt_f64_vec(&mut buf, &self.covariances_);
        write_opt_f64_vec(&mut buf, &self.precisions_cholesky_);
        write_bool(&mut buf, self.converged_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_f64(&mut buf, self.lower_bound_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_GAUSSIAN_MIXTURE)?;
        let n_components = reader.read_usize()?;
        let covariance_type = reader.read_str()?;
        let tol = reader.read_f64()?;
        let reg_covar = reader.read_f64()?;
        let max_iter = reader.read_usize()?;
        let n_init = reader.read_usize()?;
        let init_params = reader.read_str()?;
        let random_state = reader.read_opt_u64()?;
        let loaded_device = reader.read_opt_str()?;

        validate_params(
            n_components,
            &covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            &init_params,
        )?;

        let weights_ = reader.read_opt_f64_vec()?;
        let means_ = reader.read_opt_matrix()?;
        let covariances_ = reader.read_opt_f64_vec()?;
        let precisions_cholesky_ = reader.read_opt_f64_vec()?;
        let converged_ = reader.read_bool()?;
        let n_iter_ = reader.read_usize()?;
        let lower_bound_ = reader.read_opt_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;

        let (dt, dev_str) = resolve_loaded_device(loaded_device, weights_.is_some(), device)?;
        Ok(Self {
            n_components,
            covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
            random_state,
            device_: dev_str,
            device_type: dt,
            weights_,
            means_,
            covariances_,
            precisions_cholesky_,
            converged_,
            n_iter_,
            lower_bound_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "GaussianMixture")]
#[derive(Clone)]
pub struct PyGaussianMixture {
    inner: GaussianMixture,
}

crate::pymethods_with_dit!(PyGaussianMixture, GaussianMixture, {
    #[new]
    #[pyo3(signature = (
        n_components = 1,
        covariance_type = "full".to_string(),
        tol = 1e-3,
        reg_covar = 1e-6,
        max_iter = 100,
        n_init = 1,
        init_params = "kmeans".to_string(),
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: usize,
        covariance_type: String,
        tol: f64,
        reg_covar: f64,
        max_iter: usize,
        n_init: usize,
        init_params: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianMixture::new(
                n_components,
                covariance_type,
                tol,
                reg_covar,
                max_iter,
                n_init,
                init_params,
                random_state,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn fit_predict_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let labels = self.inner.fit_predict(&mat)?;
        usize_slice_to_i64_pybytes(py, &labels)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let labels = self.inner.predict(&mat)?;
        usize_slice_to_i64_pybytes(py, &labels)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&mat)?;
        matrix_to_pybytes(py, &probs)
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

    pub fn aic_buffer(&self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        Ok(self.inner.aic(&mat)?)
    }

    pub fn bic_buffer(&self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        Ok(self.inner.bic(&mat)?)
    }

    #[pyo3(signature = (n_samples = 1))]
    pub fn sample_buffer<'py>(
        &self,
        py: Python<'py>,
        n_samples: usize,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let (x_mat, y_labels) = self.inner.sample(n_samples)?;
        let (rows, cols, x_bytes) = matrix_to_pybytes(py, &x_mat)?;
        let y_bytes = usize_slice_to_i64_pybytes(py, &y_labels)?;
        Ok((rows, cols, x_bytes, y_bytes))
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }

    #[getter]
    pub fn covariance_type(&self) -> String {
        self.inner.covariance_type.clone()
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    pub fn reg_covar(&self) -> f64 {
        self.inner.reg_covar
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn n_init(&self) -> usize {
        self.inner.n_init
    }

    #[getter]
    pub fn init_params(&self) -> String {
        self.inner.init_params.clone()
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn weights_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match self.inner.weights_ {
            Some(ref w) => Ok(Some(f64_slice_to_pybytes(py, w)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn means_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match self.inner.means_ {
            Some(ref m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn covariances_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match self.inner.covariances_ {
            Some(ref c) => Ok(Some(f64_slice_to_pybytes(py, c)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn precisions_cholesky_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match self.inner.precisions_cholesky_ {
            Some(ref pc) => Ok(Some(f64_slice_to_pybytes(py, pc)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn converged_(&self) -> bool {
        self.inner.converged_
    }

    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }

    #[getter]
    pub fn lower_bound_(&self) -> Option<f64> {
        self.inner.lower_bound_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianMixture>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_clusters() -> Matrix {
        let data = vec![
            0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 0.2, 0.0, 0.0, 0.2, -0.2, -0.1, 10.0, 10.0, 10.1, 9.9,
            9.9, 10.1, 10.2, 10.0, 10.0, 10.2, 9.8, 9.9,
        ];
        Matrix::from_vec(12, 2, data).unwrap()
    }

    #[test]
    fn test_gmm_all_covariance_types_and_dit() {
        let x = make_two_clusters();
        for cov_type in ["full", "tied", "diag", "spherical"] {
            let mut gmm = GaussianMixture::new(
                2,
                cov_type.to_string(),
                1e-4,
                1e-6,
                100,
                2,
                "kmeans".to_string(),
                Some(42),
                Some("cpu".to_string()),
            )
            .unwrap();
            let labels = gmm.fit_predict(&x).unwrap();
            assert_eq!(labels.len(), 12);
            assert!(gmm.converged_);
            // First 6 and last 6 should be in separate clusters
            assert!(labels[0..6].iter().all(|&l| l == labels[0]));
            assert!(labels[6..12].iter().all(|&l| l == labels[6]));
            assert_ne!(labels[0], labels[6]);

            let probs = gmm.predict_proba(&x).unwrap();
            assert_eq!(probs.rows, 12);
            assert_eq!(probs.cols, 2);
            for r in 0..12 {
                let sum = probs[(r, 0)] + probs[(r, 1)];
                assert!((sum - 1.0).abs() < 1e-9);
            }

            let aic = gmm.aic(&x).unwrap();
            let bic = gmm.bic(&x).unwrap();
            assert!(aic.is_finite() && bic.is_finite());

            let (sx, sy) = gmm.sample(20).unwrap();
            assert_eq!(sx.rows, 20);
            assert_eq!(sx.cols, 2);
            assert_eq!(sy.len(), 20);

            let bytes = gmm.to_bytes();
            let loaded = GaussianMixture::from_bytes(&bytes, Some("auto")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), labels);
            assert!((loaded.score(&x).unwrap() - gmm.score(&x).unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gmm_random_init_and_validation() {
        let x = make_two_clusters();
        let mut gmm = GaussianMixture::new(
            2,
            "full".to_string(),
            1e-3,
            1e-6,
            100,
            1,
            "random".to_string(),
            Some(7),
            None,
        )
        .unwrap();
        gmm.fit(&x).unwrap();
        assert!(gmm.score(&x).unwrap().is_finite());

        assert!(
            GaussianMixture::new(
                0,
                "full".to_string(),
                1e-3,
                1e-6,
                100,
                1,
                "kmeans".to_string(),
                None,
                None,
            )
            .is_err()
        );

        assert!(
            GaussianMixture::new(
                2,
                "invalid".to_string(),
                1e-3,
                1e-6,
                100,
                1,
                "kmeans".to_string(),
                None,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn test_gmm_large_parallel_fused_em_all_cov_types() {
        let n = 651;
        let p = 7;
        let k = 3;
        let mut data = vec![0.0; n * p];
        for i in 0..n {
            let cls = i % k;
            let base = (cls as f64) * 8.0 + 1000.0;
            for j in 0..p {
                data[i * p + j] = base + (((i * 13 + j * 5) % 17) as f64) * 0.1;
            }
        }
        let x = Matrix::from_vec(n, p, data).unwrap();
        for cov_type in ["full", "tied", "diag", "spherical"] {
            let mut gmm = GaussianMixture::new(
                k,
                cov_type.to_string(),
                1e-4,
                1e-5,
                50,
                1,
                "kmeans".to_string(),
                Some(42),
                Some("cpu".to_string()),
            )
            .unwrap();
            let labels = gmm.fit_predict(&x).unwrap();
            assert_eq!(labels.len(), n);
            let probs = gmm.predict_proba(&x).unwrap();
            assert_eq!(probs.rows, n);
            assert_eq!(probs.cols, k);
            for r in 0..n {
                let mut sum = 0.0;
                for c in 0..k {
                    sum += probs[(r, c)];
                }
                assert!((sum - 1.0).abs() < 1e-9);
            }
            assert!(gmm.score(&x).unwrap().is_finite());
        }
    }
}
