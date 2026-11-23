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

pub const TAG_GAUSSIAN_MIXTURE: u8 = 64;

const MAX_DIST_ELEMENTS: usize = 4_000_000;
const KMEANS_INIT_MAX_ITER: usize = 10;

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
    if x.data.iter().any(|v| !v.is_finite()) {
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

fn init_responsibilities_kmeans(
    backend: &Backend<'_>,
    x: &Matrix,
    n_components: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
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

    let mut resp = vec![0.0; n * n_components];
    for (i, &c) in labels.iter().enumerate() {
        resp[i * n_components + c] = 1.0;
    }
    Matrix::from_vec(n, n_components, resp)
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
) -> OchreResult<Matrix> {
    let mut lp =
        estimate_log_gaussian_prob(backend, x, means, precisions_cholesky, covariance_type)?;
    let log_w: Vec<f64> = weights
        .iter()
        .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
        .collect();
    let k = weights.len();
    for row in lp.data.chunks_mut(k) {
        for c in 0..k {
            row[c] += log_w[c];
        }
    }
    Ok(lp)
}

fn e_step(
    backend: &Backend<'_>,
    x: &Matrix,
    weights: &[f64],
    means: &Matrix,
    precisions_cholesky: &[f64],
    covariance_type: &str,
) -> OchreResult<(Vec<f64>, Matrix)> {
    let n = x.rows;
    let k = weights.len();
    let wlp = estimate_weighted_log_prob(
        backend,
        x,
        weights,
        means,
        precisions_cholesky,
        covariance_type,
    )?;

    let mut log_prob_norm = vec![0.0; n];
    let mut log_resp = vec![0.0; n * k];

    for i in 0..n {
        let row = &wlp.data[i * k..(i + 1) * k];
        let mut max_v = f64::NEG_INFINITY;
        for &v in row {
            if v > max_v {
                max_v = v;
            }
        }
        let lse = if max_v == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            let mut sum_exp = 0.0;
            for &v in row {
                sum_exp += (v - max_v).exp();
            }
            max_v + sum_exp.ln()
        };
        log_prob_norm[i] = lse;
        let out_row = &mut log_resp[i * k..(i + 1) * k];
        for c in 0..k {
            out_row[c] = row[c] - lse;
        }
    }

    Ok((log_prob_norm, Matrix::from_vec(n, k, log_resp)?))
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

        let backend = get_backend(&self.device_type);
        let n = x.rows;
        let p = x.cols;
        let k = self.n_components;

        let mut best_lower_bound = f64::NEG_INFINITY;
        type BestGmmParams = (Vec<f64>, Matrix, Vec<f64>, Vec<f64>, bool, usize);
        let mut best_params: Option<BestGmmParams> = None;

        for init_idx in 0..self.n_init {
            let seed = self
                .random_state
                .map(|s| s.wrapping_add((init_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
            let mut rng = DeterministicRng::new(seed);

            let init_resp = if self.init_params == "kmeans" {
                init_responsibilities_kmeans(&backend, x, k, &mut rng)?
            } else {
                init_responsibilities_random(n, k, &mut rng)?
            };

            let (mut weights, mut means, mut covariances, mut prec_chols) =
                estimate_gaussian_parameters(
                    &backend,
                    x,
                    &init_resp,
                    &self.covariance_type,
                    self.reg_covar,
                )?;

            let mut prev_lower_bound = f64::NEG_INFINITY;
            let mut converged = false;
            let mut n_iter = 0;

            for iter in 1..=self.max_iter {
                let (log_prob_norm, log_resp) = e_step(
                    &backend,
                    x,
                    &weights,
                    &means,
                    &prec_chols,
                    &self.covariance_type,
                )?;
                let lower_bound = log_prob_norm.iter().sum::<f64>() / (n as f64);

                let resp_data: Vec<f64> = log_resp.data.iter().map(|&v| v.exp()).collect();
                let resp = Matrix::from_vec(n, k, resp_data)?;

                let (w_new, m_new, c_new, pc_new) = estimate_gaussian_parameters(
                    &backend,
                    x,
                    &resp,
                    &self.covariance_type,
                    self.reg_covar,
                )?;
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

        if let Some((weights, means, covariances, prec_chols, converged, n_iter)) = best_params {
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
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = get_backend(&self.device_type);
        let (_, log_resp) = e_step(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
        )?;
        let data: Vec<f64> = log_resp.data.into_iter().map(|v| v.exp()).collect();
        Matrix::from_vec(x.rows, self.n_components, data)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<usize>> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = get_backend(&self.device_type);
        let wlp = estimate_weighted_log_prob(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
        )?;
        let k = self.n_components;
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
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let (weights, means, prec_chols) = self.get_fitted()?;
        let backend = get_backend(&self.device_type);
        let (log_prob_norm, _) = e_step(
            &backend,
            x,
            weights,
            means,
            prec_chols,
            &self.covariance_type,
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
}
