// OchreML Clustering Module (*≧ω≦*)
// Implements KMeans and MiniBatchKMeans accelerated by Backend Primitives N3 (pairwise_l2_sq)
// and N4 (row_argmin), with .dit persistence and PyO3 zero-copy buffer bindings.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_f64, write_header, write_opt_f64, write_opt_f64_vec,
    write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_opt_usize_vec,
    write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
    usize_slice_to_i64_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub const TAG_KMEANS: u8 = 47;
pub const TAG_MINI_BATCH_KMEANS: u8 = 48;

const MAX_DIST_ELEMENTS: usize = 4_000_000;

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

fn validate_sample_weight(sw: Option<&[f64]>, n_samples: usize) -> OchreResult<Option<&[f64]>> {
    match sw {
        None => Ok(None),
        Some(w) => {
            if w.len() != n_samples {
                return Err(OchreError::WeightLengthMismatch {
                    weights: w.len(),
                    samples: n_samples,
                });
            }
            let mut sum = 0.0;
            for &val in w {
                if !val.is_finite() || val < 0.0 {
                    return Err(OchreError::InvalidWeights);
                }
                sum += val;
            }
            if sum <= 0.0 || !sum.is_finite() {
                return Err(OchreError::NonPositiveWeightSum);
            }
            Ok(Some(w))
        }
    }
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

fn validate_common_params(
    n_clusters: usize,
    init: &str,
    n_init: usize,
    max_iter: usize,
    tol: f64,
) -> OchreResult<()> {
    if n_clusters == 0 {
        return Err(OchreError::InvalidParameter(
            "n_clusters must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if init != "k-means++" && init != "random" {
        return Err(OchreError::InvalidParameter(format!(
            "init must be 'k-means++' or 'random', got '{}' (・`ω´・)",
            init
        )));
    }
    if n_init == 0 {
        return Err(OchreError::InvalidParameter(
            "n_init must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1 (・`ω´・)".to_string(),
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(OchreError::InvalidParameter(
            "tol must be a non-negative finite number (・`ω´・)".to_string(),
        ));
    }
    Ok(())
}

fn compute_mean_variance(x: &Matrix, sample_weight: Option<&[f64]>) -> f64 {
    let n = x.rows;
    let p = x.cols;
    let mut means = vec![0.0; p];
    let mut w_sum = 0.0;

    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        w_sum += w;
        let row = &x.data[i * p..(i + 1) * p];
        for j in 0..p {
            means[j] += w * row[j];
        }
    }
    if w_sum <= 0.0 {
        return 0.0;
    }
    for m in &mut means {
        *m /= w_sum;
    }

    let mut var_sum = 0.0;
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let row = &x.data[i * p..(i + 1) * p];
        for j in 0..p {
            let d = row[j] - means[j];
            var_sum += w * d * d;
        }
    }
    var_sum / (w_sum * p as f64)
}

/// Compute nearest center labels and squared distances using N3 (`pairwise_l2_sq`) + N4 (`row_argmin`),
/// chunking rows when `n_samples * n_clusters` exceeds `MAX_DIST_ELEMENTS`.
fn assign_labels_and_sq_dists(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<(Vec<usize>, Vec<f64>)> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if n.saturating_mul(k) <= MAX_DIST_ELEMENTS {
        let dist = backend.pairwise_l2_sq(x, centers)?;
        return backend.row_argmin(&dist);
    }

    let chunk_rows = (MAX_DIST_ELEMENTS / k.max(1)).max(1).min(n);
    let mut all_labels = Vec::with_capacity(n);
    let mut all_dists = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_rows).min(n);
        let rows = end - start;
        let slice = &x.data[start * p..end * p];
        let chunk = Matrix::from_slice(rows, p, slice)?;
        let dist = backend.pairwise_l2_sq(&chunk, centers)?;
        let (labels, dists) = backend.row_argmin(&dist)?;
        all_labels.extend(labels);
        all_dists.extend(dists);
        start = end;
    }
    Ok((all_labels, all_dists))
}

/// Compute Euclidean distance matrix `sqrt(pairwise_l2_sq(X, centers))`, chunked for large inputs.
fn compute_euclidean_transform(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if n.saturating_mul(k) <= MAX_DIST_ELEMENTS {
        let mut dist = backend.pairwise_l2_sq(x, centers)?;
        for v in &mut dist.data {
            *v = v.max(0.0).sqrt();
        }
        return Ok(dist);
    }

    let chunk_rows = (MAX_DIST_ELEMENTS / k.max(1)).max(1).min(n);
    let mut out = Vec::with_capacity(n * k);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_rows).min(n);
        let rows = end - start;
        let chunk = Matrix::from_slice(rows, p, &x.data[start * p..end * p])?;
        let dist = backend.pairwise_l2_sq(&chunk, centers)?;
        for v in dist.data {
            out.push(v.max(0.0).sqrt());
        }
        start = end;
    }
    Matrix::from_vec(n, k, out)
}

fn sample_weighted_index(weights: &[f64], rng: &mut DeterministicRng) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return rng.gen_range(weights.len());
    }
    let target = rng.next_f64() * total;
    let mut cum = 0.0;
    for (idx, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= target {
            return idx;
        }
    }
    weights.len() - 1
}

fn init_centers_kmeans_pp(
    backend: &Backend<'_>,
    x: &Matrix,
    sample_weight: Option<&[f64]>,
    n_clusters: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let mut centers_data = vec![0.0; n_clusters * p];

    let first_idx = match sample_weight {
        Some(sw) => sample_weighted_index(sw, rng),
        None => rng.gen_range(n),
    };
    centers_data[0..p].copy_from_slice(&x.data[first_idx * p..(first_idx + 1) * p]);

    if n_clusters == 1 {
        return Matrix::from_vec(n_clusters, p, centers_data);
    }

    let first_mat = Matrix::from_slice(1, p, &centers_data[0..p])?;
    let mut min_sq_dist = backend.pairwise_l2_sq(x, &first_mat)?.data;
    let mut potentials = vec![0.0; n];

    for c in 1..n_clusters {
        let mut total = 0.0;
        for i in 0..n {
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            let pot = min_sq_dist[i].max(0.0) * w;
            potentials[i] = pot;
            total += pot;
        }

        let next_idx = if total <= 0.0 || !total.is_finite() {
            rng.gen_range(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cum = 0.0;
            let mut chosen = n - 1;
            for (i, &pot) in potentials.iter().enumerate() {
                cum += pot;
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

fn init_centers_random(
    x: &Matrix,
    n_clusters: usize,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut centers_data = vec![0.0; n_clusters * p];

    for c in 0..n_clusters {
        let swap_idx = c + rng.gen_range(n - c);
        indices.swap(c, swap_idx);
        let idx = indices[c];
        centers_data[c * p..(c + 1) * p].copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
    }

    Matrix::from_vec(n_clusters, p, centers_data)
}

fn init_centers(
    backend: &Backend<'_>,
    x: &Matrix,
    sample_weight: Option<&[f64]>,
    n_clusters: usize,
    init: &str,
    rng: &mut DeterministicRng,
) -> OchreResult<Matrix> {
    match init {
        "k-means++" => init_centers_kmeans_pp(backend, x, sample_weight, n_clusters, rng),
        "random" => init_centers_random(x, n_clusters, rng),
        _ => Err(OchreError::InvalidParameter(format!(
            "init must be 'k-means++' or 'random', got '{}' (・`ω´・)",
            init
        ))),
    }
}

fn compute_weighted_inertia(sq_dists: &[f64], sample_weight: Option<&[f64]>) -> f64 {
    match sample_weight {
        Some(sw) => sq_dists
            .iter()
            .zip(sw.iter())
            .map(|(&d, &w)| d.max(0.0) * w)
            .sum(),
        None => sq_dists.iter().map(|&d| d.max(0.0)).sum(),
    }
}

/// Core K-Means Clustering Engine (＾▽＾)
#[derive(Debug, Clone)]
pub struct KMeans {
    pub n_clusters: usize,
    pub init: String,
    pub n_init: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
    pub cluster_centers_: Option<Matrix>,
    pub labels_: Option<Vec<usize>>,
    pub inertia_: Option<f64>,
    pub n_iter_: usize,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl KMeans {
    pub fn new(
        n_clusters: usize,
        init: String,
        n_init: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_common_params(n_clusters, &init, n_init, max_iter, tol)?;
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            n_clusters,
            init,
            n_init,
            max_iter,
            tol,
            random_state,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: 0,
            n_features_in_: None,
            device_: Some(dev.to_string()),
            device_type: dev,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_x(x)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        if self.n_clusters > x.rows {
            return Err(OchreError::InvalidParameter(format!(
                "n_clusters={} should be <= n_samples={} ( >_< )",
                self.n_clusters, x.rows
            )));
        }

        let backend = get_backend(&self.device_type);
        let mean_var = compute_mean_variance(x, sw);
        let tol_scaled = self.tol * mean_var;
        let mut rng = DeterministicRng::new(self.random_state);

        let n = x.rows;
        let p = x.cols;
        let k = self.n_clusters;

        let mut best_centers: Option<Matrix> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0usize;

        for _run in 0..self.n_init {
            let mut centers = init_centers(&backend, x, sw, k, &self.init, &mut rng)?;
            let mut run_iters = 0usize;

            for iter in 0..self.max_iter {
                let (mut labels, mut sq_dists) = assign_labels_and_sq_dists(&backend, x, &centers)?;

                let mut new_centers_data = vec![0.0; k * p];
                let mut cluster_weights = vec![0.0; k];

                for i in 0..n {
                    let c = labels[i];
                    let w = sw.map_or(1.0, |weights| weights[i]);
                    cluster_weights[c] += w;
                    let row = &x.data[i * p..(i + 1) * p];
                    let dst = &mut new_centers_data[c * p..(c + 1) * p];
                    for j in 0..p {
                        dst[j] += w * row[j];
                    }
                }

                // Reassign empty clusters using the farthest samples
                for c in 0..k {
                    if cluster_weights[c] <= 0.0 {
                        let mut far_idx = 0usize;
                        let mut far_dist = -1.0f64;
                        for (i, &d) in sq_dists.iter().enumerate() {
                            if d > far_dist {
                                far_dist = d;
                                far_idx = i;
                            }
                        }
                        let old_c = labels[far_idx];
                        let w_far = sw.map_or(1.0, |weights| weights[far_idx]);
                        if cluster_weights[old_c] > w_far {
                            cluster_weights[old_c] -= w_far;
                            let row = &x.data[far_idx * p..(far_idx + 1) * p];
                            let old_dst = &mut new_centers_data[old_c * p..(old_c + 1) * p];
                            for j in 0..p {
                                old_dst[j] -= w_far * row[j];
                            }
                        }
                        sq_dists[far_idx] = 0.0;
                        labels[far_idx] = c;
                        cluster_weights[c] = w_far.max(1.0);
                        let row = &x.data[far_idx * p..(far_idx + 1) * p];
                        let dst = &mut new_centers_data[c * p..(c + 1) * p];
                        for j in 0..p {
                            dst[j] = cluster_weights[c] * row[j];
                        }
                    }
                }

                for c in 0..k {
                    let cw = cluster_weights[c];
                    if cw > 0.0 {
                        let inv = 1.0 / cw;
                        let dst = &mut new_centers_data[c * p..(c + 1) * p];
                        for v in dst.iter_mut() {
                            *v *= inv;
                        }
                    }
                }

                let mut shift_sq = 0.0;
                for (a, b) in new_centers_data.iter().zip(centers.data.iter()) {
                    let d = a - b;
                    shift_sq += d * d;
                }

                centers = Matrix::from_vec(k, p, new_centers_data)?;
                run_iters = iter + 1;

                if shift_sq <= tol_scaled {
                    break;
                }
            }

            let (final_labels, final_sq_dists) = assign_labels_and_sq_dists(&backend, x, &centers)?;
            let inertia = compute_weighted_inertia(&final_sq_dists, sw);

            if inertia < best_inertia || best_centers.is_none() {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(final_labels);
                best_n_iter = run_iters;
            }
        }

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = best_n_iter;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<usize>> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        let (labels, _) = assign_labels_and_sq_dists(&backend, x, centers)?;
        Ok(labels)
    }

    pub fn fit_predict(
        &mut self,
        x: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<Vec<usize>> {
        self.fit(x, sample_weight)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        compute_euclidean_transform(&backend, x, centers)
    }

    pub fn fit_transform(
        &mut self,
        x: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<Matrix> {
        self.fit(x, sample_weight)?;
        self.transform(x)
    }

    pub fn score(&self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        let (_, sq_dists) = assign_labels_and_sq_dists(&backend, x, centers)?;
        Ok(-compute_weighted_inertia(&sq_dists, sw))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KMEANS);
        write_usize(&mut buf, self.n_clusters);
        write_str(&mut buf, &self.init);
        write_usize(&mut buf, self.n_init);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.tol);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_matrix(&mut buf, &self.cluster_centers_);
        write_opt_usize_vec(&mut buf, &self.labels_);
        write_opt_f64(&mut buf, self.inertia_);
        write_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_KMEANS)?;
        let n_clusters = reader.read_usize()?;
        let init = reader.read_str()?;
        let n_init = reader.read_usize()?;
        let max_iter = reader.read_usize()?;
        let tol = reader.read_f64()?;
        let random_state = reader.read_opt_u64()?;
        let cluster_centers_ = reader.read_opt_matrix()?;
        let labels_ = reader.read_opt_usize_vec()?;
        let inertia_ = reader.read_opt_f64()?;
        let n_iter_ = reader.read_usize()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let saved_device = reader.read_opt_str()?;

        let is_fitted = cluster_centers_.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            n_clusters,
            init,
            n_init,
            max_iter,
            tol,
            random_state,
            cluster_centers_,
            labels_,
            inertia_,
            n_iter_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

/// Mini-Batch K-Means Clustering Engine (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    pub init: String,
    pub max_iter: usize,
    pub batch_size: usize,
    pub tol: f64,
    pub max_no_improvement: usize,
    pub n_init: usize,
    pub reassignment_ratio: f64,
    pub random_state: Option<u64>,
    pub cluster_centers_: Option<Matrix>,
    pub labels_: Option<Vec<usize>>,
    pub inertia_: Option<f64>,
    pub n_iter_: usize,
    pub n_steps_: usize,
    pub n_features_in_: Option<usize>,
    pub counts_: Option<Vec<f64>>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl MiniBatchKMeans {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_clusters: usize,
        init: String,
        max_iter: usize,
        batch_size: usize,
        tol: f64,
        max_no_improvement: usize,
        n_init: usize,
        reassignment_ratio: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_common_params(n_clusters, &init, n_init, max_iter, tol)?;
        if batch_size == 0 {
            return Err(OchreError::InvalidParameter(
                "batch_size must be >= 1 (・`ω´・)".to_string(),
            ));
        }
        if !reassignment_ratio.is_finite() || reassignment_ratio < 0.0 || reassignment_ratio > 1.0 {
            return Err(OchreError::InvalidParameter(
                "reassignment_ratio must be in [0.0, 1.0] (・`ω´・)".to_string(),
            ));
        }
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            n_clusters,
            init,
            max_iter,
            batch_size,
            tol,
            max_no_improvement,
            n_init,
            reassignment_ratio,
            random_state,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: 0,
            n_steps_: 0,
            n_features_in_: None,
            counts_: None,
            device_: Some(dev.to_string()),
            device_type: dev,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    fn minibatch_step(
        backend: &Backend<'_>,
        x_batch: &Matrix,
        w_batch: Option<&[f64]>,
        centers: &mut Matrix,
        counts: &mut [f64],
        reassignment_ratio: f64,
        rng: &mut DeterministicRng,
    ) -> OchreResult<f64> {
        let (labels, mut sq_dists) = assign_labels_and_sq_dists(backend, x_batch, centers)?;
        let n = x_batch.rows;
        let p = x_batch.cols;
        let k = centers.rows;

        let batch_inertia = compute_weighted_inertia(&sq_dists, w_batch);

        for i in 0..n {
            let w = w_batch.map_or(1.0, |sw| sw[i]);
            if w <= 0.0 {
                continue;
            }
            let c = labels[i];
            counts[c] += w;
            let eta = w / counts[c];
            let row = &x_batch.data[i * p..(i + 1) * p];
            let center = &mut centers.data[c * p..(c + 1) * p];
            for j in 0..p {
                center[j] += eta * (row[j] - center[j]);
            }
        }

        if reassignment_ratio > 0.0 {
            let max_count = counts.iter().cloned().fold(0.0f64, f64::max);
            let threshold = reassignment_ratio * max_count;
            if max_count > 0.0 {
                for (c, cnt) in counts.iter_mut().enumerate().take(k) {
                    if *cnt < threshold {
                        // Pick farthest sample in batch (or random if all zero)
                        let mut far_idx = rng.gen_range(n);
                        let mut far_dist = -1.0f64;
                        for (i, &d) in sq_dists.iter().enumerate() {
                            if d > far_dist {
                                far_dist = d;
                                far_idx = i;
                            }
                        }
                        sq_dists[far_idx] = 0.0;
                        centers.data[c * p..(c + 1) * p]
                            .copy_from_slice(&x_batch.data[far_idx * p..(far_idx + 1) * p]);
                        *cnt = 1.0;
                    }
                }
            }
        }

        Ok(batch_inertia)
    }

    pub fn fit(&mut self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_x(x)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        if self.n_clusters > x.rows {
            return Err(OchreError::InvalidParameter(format!(
                "n_clusters={} should be <= n_samples={} ( >_< )",
                self.n_clusters, x.rows
            )));
        }

        let backend = get_backend(&self.device_type);
        let mean_var = compute_mean_variance(x, sw);
        let tol_scaled = self.tol * mean_var;
        let mut rng = DeterministicRng::new(self.random_state);

        let n = x.rows;
        let p = x.cols;
        let k = self.n_clusters;
        let eff_batch = self.batch_size.min(n).max(1);
        let ewma_alpha = ((2.0 * eff_batch as f64) / (n as f64 + 1.0)).clamp(0.01, 1.0);

        let mut best_centers: Option<Matrix> = None;
        let mut best_counts: Option<Vec<f64>> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0usize;
        let mut best_n_steps = 0usize;

        for _run in 0..self.n_init {
            let mut centers = init_centers(&backend, x, sw, k, &self.init, &mut rng)?;
            let mut counts = vec![0.0; k];
            let mut ewma_inertia: Option<f64> = None;
            let mut min_ewma = f64::INFINITY;
            let mut no_improvement = 0usize;
            let mut steps = 0usize;

            let mut batch_data = vec![0.0; eff_batch * p];
            let mut batch_weights = vec![0.0; eff_batch];

            for _step in 0..self.max_iter {
                let mut w_sum = 0.0;
                for b in 0..eff_batch {
                    let idx = rng.gen_range(n);
                    batch_data[b * p..(b + 1) * p].copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
                    let w = sw.map_or(1.0, |weights| weights[idx]);
                    batch_weights[b] = w;
                    w_sum += w;
                }

                let x_batch = Matrix::from_slice(eff_batch, p, &batch_data)?;
                let w_opt = if sw.is_some() {
                    Some(batch_weights.as_slice())
                } else {
                    None
                };

                let batch_inertia = Self::minibatch_step(
                    &backend,
                    &x_batch,
                    w_opt,
                    &mut centers,
                    &mut counts,
                    self.reassignment_ratio,
                    &mut rng,
                )?;
                steps += 1;

                let mean_batch_inertia = if w_sum > 0.0 {
                    batch_inertia / w_sum
                } else {
                    0.0
                };
                let ewma = match ewma_inertia {
                    None => mean_batch_inertia,
                    Some(prev) => prev * (1.0 - ewma_alpha) + mean_batch_inertia * ewma_alpha,
                };
                ewma_inertia = Some(ewma);

                if ewma < min_ewma - tol_scaled {
                    min_ewma = ewma;
                    no_improvement = 0;
                } else {
                    no_improvement += 1;
                    if self.max_no_improvement > 0 && no_improvement >= self.max_no_improvement {
                        break;
                    }
                }
            }

            let (final_labels, final_sq_dists) = assign_labels_and_sq_dists(&backend, x, &centers)?;
            let inertia = compute_weighted_inertia(&final_sq_dists, sw);

            if inertia < best_inertia || best_centers.is_none() {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_counts = Some(counts);
                best_labels = Some(final_labels);
                best_n_iter = steps.div_ceil((n.div_ceil(eff_batch)).max(1));
                best_n_steps = steps;
            }
        }

        self.cluster_centers_ = best_centers;
        self.counts_ = best_counts;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = best_n_iter;
        self.n_steps_ = best_n_steps;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn partial_fit(&mut self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_x(x)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let backend = get_backend(&self.device_type);
        let mut rng = DeterministicRng::new(
            self.random_state
                .map(|s| s.wrapping_add(self.n_steps_ as u64)),
        );

        if self.cluster_centers_.is_none() {
            if self.n_clusters > x.rows {
                return Err(OchreError::InvalidParameter(format!(
                    "n_clusters={} should be <= n_samples={} for initial partial_fit ( >_< )",
                    self.n_clusters, x.rows
                )));
            }
            let centers = init_centers(&backend, x, sw, self.n_clusters, &self.init, &mut rng)?;
            self.cluster_centers_ = Some(centers);
            self.counts_ = Some(vec![0.0; self.n_clusters]);
            self.n_features_in_ = Some(x.cols);
        } else {
            validate_fitted_cols(self.n_features_in_, x.cols)?;
        }

        let centers = self
            .cluster_centers_
            .as_mut()
            .ok_or(OchreError::NotFitted)?;
        let counts = self.counts_.as_mut().ok_or(OchreError::NotFitted)?;

        let _ = Self::minibatch_step(
            &backend,
            x,
            sw,
            centers,
            counts,
            self.reassignment_ratio,
            &mut rng,
        )?;

        let (labels, sq_dists) = assign_labels_and_sq_dists(&backend, x, centers)?;
        self.labels_ = Some(labels);
        self.inertia_ = Some(compute_weighted_inertia(&sq_dists, sw));
        self.n_steps_ += 1;
        self.n_iter_ += 1;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<usize>> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        let (labels, _) = assign_labels_and_sq_dists(&backend, x, centers)?;
        Ok(labels)
    }

    pub fn fit_predict(
        &mut self,
        x: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<Vec<usize>> {
        self.fit(x, sample_weight)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        compute_euclidean_transform(&backend, x, centers)
    }

    pub fn fit_transform(
        &mut self,
        x: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<Matrix> {
        self.fit(x, sample_weight)?;
        self.transform(x)
    }

    pub fn score(&self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        let (_, sq_dists) = assign_labels_and_sq_dists(&backend, x, centers)?;
        Ok(-compute_weighted_inertia(&sq_dists, sw))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MINI_BATCH_KMEANS);
        write_usize(&mut buf, self.n_clusters);
        write_str(&mut buf, &self.init);
        write_usize(&mut buf, self.max_iter);
        write_usize(&mut buf, self.batch_size);
        write_f64(&mut buf, self.tol);
        write_usize(&mut buf, self.max_no_improvement);
        write_usize(&mut buf, self.n_init);
        write_f64(&mut buf, self.reassignment_ratio);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_matrix(&mut buf, &self.cluster_centers_);
        write_opt_usize_vec(&mut buf, &self.labels_);
        write_opt_f64(&mut buf, self.inertia_);
        write_usize(&mut buf, self.n_iter_);
        write_usize(&mut buf, self.n_steps_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_f64_vec(&mut buf, &self.counts_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_MINI_BATCH_KMEANS)?;
        let n_clusters = reader.read_usize()?;
        let init = reader.read_str()?;
        let max_iter = reader.read_usize()?;
        let batch_size = reader.read_usize()?;
        let tol = reader.read_f64()?;
        let max_no_improvement = reader.read_usize()?;
        let n_init = reader.read_usize()?;
        let reassignment_ratio = reader.read_f64()?;
        let random_state = reader.read_opt_u64()?;
        let cluster_centers_ = reader.read_opt_matrix()?;
        let labels_ = reader.read_opt_usize_vec()?;
        let inertia_ = reader.read_opt_f64()?;
        let n_iter_ = reader.read_usize()?;
        let n_steps_ = reader.read_usize()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let counts_ = reader.read_opt_f64_vec()?;
        let saved_device = reader.read_opt_str()?;

        let is_fitted = cluster_centers_.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            n_clusters,
            init,
            max_iter,
            batch_size,
            tol,
            max_no_improvement,
            n_init,
            reassignment_ratio,
            random_state,
            cluster_centers_,
            labels_,
            inertia_,
            n_iter_,
            n_steps_,
            n_features_in_,
            counts_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// PyO3 Bindings (*≧ω≦*)
// ============================================================================

#[pyclass(from_py_object, name = "KMeans")]
#[derive(Clone)]
pub struct PyKMeans {
    inner: KMeans,
}

crate::pymethods_with_dit!(PyKMeans, KMeans, {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8,
        init = "k-means++".to_string(),
        n_init = 10,
        max_iter = 300,
        tol = 1e-4,
        random_state = None,
        device = None
    ))]
    pub fn new(
        n_clusters: usize,
        init: String,
        n_init: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KMeans::new(
                n_clusters,
                init,
                n_init,
                max_iter,
                tol,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.fit(&mat, sw.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_predict_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        let labels = self.inner.fit_predict(&mat, sw.as_deref())?;
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

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        let out = self.inner.fit_transform(&mat, sw.as_deref())?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        Ok(self.inner.score(&mat, sw.as_deref())?)
    }

    #[getter]
    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }
    #[getter]
    pub fn init(&self) -> String {
        self.inner.init.clone()
    }
    #[getter]
    pub fn n_init(&self) -> usize {
        self.inner.n_init
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
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn cluster_centers_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.cluster_centers_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(lbl) => Ok(Some(usize_slice_to_i64_pybytes(py, lbl)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn inertia_(&self) -> Option<f64> {
        self.inner.inertia_
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

#[pyclass(from_py_object, name = "MiniBatchKMeans")]
#[derive(Clone)]
pub struct PyMiniBatchKMeans {
    inner: MiniBatchKMeans,
}

crate::pymethods_with_dit!(PyMiniBatchKMeans, MiniBatchKMeans, {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8,
        init = "k-means++".to_string(),
        max_iter = 100,
        batch_size = 1024,
        tol = 0.0,
        max_no_improvement = 10,
        n_init = 3,
        reassignment_ratio = 0.01,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_clusters: usize,
        init: String,
        max_iter: usize,
        batch_size: usize,
        tol: f64,
        max_no_improvement: usize,
        n_init: usize,
        reassignment_ratio: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MiniBatchKMeans::new(
                n_clusters,
                init,
                max_iter,
                batch_size,
                tol,
                max_no_improvement,
                n_init,
                reassignment_ratio,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.fit(&mat, sw.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.partial_fit(&mat, sw.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_predict_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        let labels = self.inner.fit_predict(&mat, sw.as_deref())?;
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

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        let out = self.inner.fit_transform(&mat, sw.as_deref())?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        Ok(self.inner.score(&mat, sw.as_deref())?)
    }

    #[getter]
    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }
    #[getter]
    pub fn init(&self) -> String {
        self.inner.init.clone()
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn max_no_improvement(&self) -> usize {
        self.inner.max_no_improvement
    }
    #[getter]
    pub fn n_init(&self) -> usize {
        self.inner.n_init
    }
    #[getter]
    pub fn reassignment_ratio(&self) -> f64 {
        self.inner.reassignment_ratio
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn cluster_centers_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.cluster_centers_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(lbl) => Ok(Some(usize_slice_to_i64_pybytes(py, lbl)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn inertia_(&self) -> Option<f64> {
        self.inner.inertia_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn n_steps_(&self) -> usize {
        self.inner.n_steps_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn counts_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.counts_ {
            Some(c) => Ok(Some(f64_slice_to_pybytes(py, c)?)),
            None => Ok(None),
        }
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKMeans>()?;
    m.add_class::<PyMiniBatchKMeans>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_blobs() -> Matrix {
        Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, -0.1],
            vec![-0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
            vec![9.9, 10.1],
        ])
        .unwrap()
    }

    #[test]
    fn test_kmeans_fit_predict_transform_dit() {
        let x = sample_blobs();
        let mut km = KMeans::new(
            2,
            "k-means++".to_string(),
            5,
            100,
            1e-4,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        let labels = km.fit_predict(&x, None).unwrap();
        assert_eq!(labels.len(), 6);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
        assert!(km.inertia_.unwrap() < 0.2);

        let dists = km.transform(&x).unwrap();
        assert_eq!(dists.rows, 6);
        assert_eq!(dists.cols, 2);

        let bytes = km.to_bytes();
        let loaded = KMeans::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), labels);
        assert!((loaded.score(&x, None).unwrap() + km.inertia_.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_minibatch_kmeans_fit_and_partial_fit() {
        let x = sample_blobs();
        let mut mbk = MiniBatchKMeans::new(
            2,
            "k-means++".to_string(),
            50,
            4,
            0.0,
            10,
            3,
            0.01,
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        mbk.fit(&x, None).unwrap();
        let labels = mbk.predict(&x).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);

        let mut mbk2 = MiniBatchKMeans::new(
            2,
            "random".to_string(),
            50,
            4,
            0.0,
            10,
            1,
            0.01,
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        mbk2.partial_fit(&x, None).unwrap();
        mbk2.partial_fit(&x, None).unwrap();
        assert_eq!(mbk2.n_steps_, 2);
        let bytes = mbk2.to_bytes();
        let loaded = MiniBatchKMeans::from_bytes(&bytes, Some("auto")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), mbk2.predict(&x).unwrap());
    }
}
