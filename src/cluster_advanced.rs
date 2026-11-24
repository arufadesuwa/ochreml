// OchreML Advanced Clustering & Outlier Detection Module (*≧ω≦*)
// Implements DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering, and LocalOutlierFactor
// accelerated by Backend Primitives N3 (pairwise_l2_sq), N4 (topk_argmin), N5 (small_sym_eig_svd),
// and crate::cluster::KMeans, with .dit persistence and PyO3 zero-copy buffer bindings.

use crate::backend::{Backend, get_backend};
use crate::cluster::KMeans;
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_i64, write_opt_f64,
    write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize,
    write_opt_usize_vec, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_pooled_matrix, f64_slice_to_pybytes, i64_slice_to_pybytes, matrix_to_pybytes,
    usize_slice_to_i64_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};

pub const TAG_DBSCAN: u8 = 85;
pub const TAG_OPTICS: u8 = 86;
pub const TAG_SPECTRAL_CLUSTERING: u8 = 87;
pub const TAG_AGGLOMERATIVE_CLUSTERING: u8 = 88;
pub const TAG_LOCAL_OUTLIER_FACTOR: u8 = 89;

const MAX_DIST_ELEMENTS: usize = 4_000_000;
const QUERY_CHUNK_ROWS: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DistMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

impl DistMetric {
    fn parse(metric: &str) -> OchreResult<Self> {
        match metric {
            "euclidean" | "l2" => Ok(Self::Euclidean),
            "manhattan" | "l1" | "cityblock" => Ok(Self::Manhattan),
            "cosine" => Ok(Self::Cosine),
            _ => Err(OchreError::InvalidParameter(format!(
                "metric must be 'euclidean', 'manhattan', or 'cosine', got '{}' (・`ω´・)",
                metric
            ))),
        }
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

fn slice_rows_matrix(x: &Matrix, start_row: usize, end_row: usize) -> OchreResult<Matrix> {
    let cols = x.cols;
    let rows = end_row.saturating_sub(start_row);
    let data = x.data[start_row * cols..end_row * cols].to_vec();
    Matrix::from_vec(rows, cols, data)
}

#[inline]
fn row_distance(a: &[f64], b: &[f64], metric: DistMetric) -> f64 {
    match metric {
        DistMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            .sqrt(),
        DistMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum::<f64>(),
        DistMetric::Cosine => {
            let mut dot = 0.0;
            let mut na = 0.0;
            let mut nb = 0.0;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 && nb == 0.0 {
                0.0
            } else if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                let sim = dot / (na.sqrt() * nb.sqrt());
                (1.0 - sim).clamp(0.0, 2.0)
            }
        }
    }
}

/// Compute chunk distance matrix `(q_chunk.rows x x_ref.rows)`.
/// Returns `(matrix, is_squared_l2)`.
fn compute_chunk_distances(
    backend: &Backend<'_>,
    q_chunk: &Matrix,
    x_ref: &Matrix,
    metric: DistMetric,
) -> OchreResult<(Matrix, bool)> {
    if metric == DistMetric::Euclidean {
        let dist_sq = backend.pairwise_l2_sq(q_chunk, x_ref)?;
        return Ok((dist_sq, true));
    }
    let n_q = q_chunk.rows;
    let n_r = x_ref.rows;
    let cols = q_chunk.cols;
    let mut out = vec![0.0; n_q * n_r];

    let fill_row = |(qi, out_row): (usize, &mut [f64])| {
        let q_row = &q_chunk.data[qi * cols..(qi + 1) * cols];
        for (ri, slot) in out_row.iter_mut().enumerate() {
            let r_row = &x_ref.data[ri * cols..(ri + 1) * cols];
            *slot = row_distance(q_row, r_row, metric);
        }
    };

    if n_q * n_r < 16_384 {
        out.chunks_exact_mut(n_r).enumerate().for_each(fill_row);
    } else {
        out.par_chunks_mut(n_r).enumerate().for_each(fill_row);
    }

    Ok((Matrix::from_vec(n_q, n_r, out)?, false))
}

/// Compute full pairwise Euclidean distance matrix (or non-Euclidean) using N3 in chunks.
fn compute_full_pairwise_distances(
    backend: &Backend<'_>,
    x: &Matrix,
    metric: DistMetric,
) -> OchreResult<Matrix> {
    let n = x.rows;
    if n * n <= MAX_DIST_ELEMENTS {
        let (mut mat, is_sq) = compute_chunk_distances(backend, x, x, metric)?;
        if is_sq {
            for v in mat.data.iter_mut() {
                *v = v.max(0.0).sqrt();
            }
        }
        return Ok(mat);
    }

    let chunk_rows = (MAX_DIST_ELEMENTS / n.max(1)).clamp(1, QUERY_CHUNK_ROWS);
    let mut full = vec![0.0; n * n];
    let mut start = 0;
    while start < n {
        let end = (start + chunk_rows).min(n);
        let q_chunk = slice_rows_matrix(x, start, end)?;
        let (mat, is_sq) = compute_chunk_distances(backend, &q_chunk, x, metric)?;
        let dst = &mut full[start * n..end * n];
        if is_sq {
            for (d, &s) in dst.iter_mut().zip(mat.data.iter()) {
                *d = s.max(0.0).sqrt();
            }
        } else {
            dst.copy_from_slice(&mat.data);
        }
        start = end;
    }
    Matrix::from_vec(n, n, full)
}

/// Compute `k` nearest neighbors of `x_query` in `x_ref` using N3 + N4 in chunks.
fn chunked_kneighbors(
    backend: &Backend<'_>,
    x_query: &Matrix,
    x_ref: &Matrix,
    k: usize,
    metric: DistMetric,
    exclude_self: bool,
) -> OchreResult<(Vec<usize>, Matrix)> {
    let n_q = x_query.rows;
    let n_r = x_ref.rows;
    let max_k = if exclude_self {
        n_r.saturating_sub(1)
    } else {
        n_r
    };
    if k == 0 || k > max_k {
        return Err(OchreError::InvalidParameter(format!(
            "n_neighbors={} must be within 1..={} ( >_< )",
            k, max_k
        )));
    }

    let k_fetch = if exclude_self { k + 1 } else { k };
    let chunk_rows = (MAX_DIST_ELEMENTS / n_r.max(1)).clamp(1, QUERY_CHUNK_ROWS);

    let mut all_idx = Vec::with_capacity(n_q * k);
    let mut all_dists = Vec::with_capacity(n_q * k);

    let mut start = 0;
    while start < n_q {
        let end = (start + chunk_rows).min(n_q);
        let q_chunk = slice_rows_matrix(x_query, start, end)?;
        let (dist_mat, is_sq) = compute_chunk_distances(backend, &q_chunk, x_ref, metric)?;
        let (topk_idx, topk_vals) = backend.topk_argmin(&dist_mat, k_fetch)?;

        let q_len = end - start;
        for local_r in 0..q_len {
            let global_r = start + local_r;
            let row_off = local_r * k_fetch;
            let mut taken = 0usize;
            for j in 0..k_fetch {
                let idx = topk_idx[row_off + j];
                if exclude_self && idx == global_r {
                    continue;
                }
                let raw_d = topk_vals.data[row_off + j];
                let d = if is_sq { raw_d.max(0.0).sqrt() } else { raw_d };
                all_idx.push(idx);
                all_dists.push(d);
                taken += 1;
                if taken == k {
                    break;
                }
            }
            if taken < k {
                // Fallback if duplicate indices occurred (should not happen with topk_argmin)
                while taken < k {
                    all_idx.push(0);
                    all_dists.push(0.0);
                    taken += 1;
                }
            }
        }
        start = end;
    }

    Ok((all_idx, Matrix::from_vec(n_q, k, all_dists)?))
}

fn write_opt_i64_vec(buf: &mut Vec<u8>, v: &Option<Vec<i64>>) {
    match v {
        Some(vec) => {
            write_bool(buf, true);
            write_usize(buf, vec.len());
            for &x in vec {
                write_i64(buf, x);
            }
        }
        None => write_bool(buf, false),
    }
}

fn read_opt_i64_vec(r: &mut ByteReader<'_>) -> OchreResult<Option<Vec<i64>>> {
    if !r.read_bool()? {
        return Ok(None);
    }
    let len = r.read_usize()?;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(r.read_i64()?);
    }
    Ok(Some(vec))
}

// ============================================================================
// 1. DBSCAN (TAG_DBSCAN = 85) (*≧ω≦*)
// ============================================================================

#[derive(Debug, Clone)]
pub struct DBSCAN {
    pub eps: f64,
    pub min_samples: usize,
    pub metric: String,
    pub labels_: Option<Vec<i64>>,
    pub core_sample_indices_: Option<Vec<usize>>,
    pub components_: Option<Matrix>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl DBSCAN {
    pub fn new(
        eps: f64,
        min_samples: usize,
        metric: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "eps must be a positive finite number, got {} ( >_< )",
                eps
            )));
        }
        if min_samples == 0 {
            return Err(OchreError::InvalidParameter(
                "min_samples must be >= 1 ( >_< )".to_string(),
            ));
        }
        DistMetric::parse(&metric)?;
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            eps,
            min_samples,
            metric,
            labels_: None,
            core_sample_indices_: None,
            components_: None,
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

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        let dist_metric = DistMetric::parse(&self.metric)?;
        let backend = get_backend(&self.device_type);
        let n = x.rows;
        let p = x.cols;

        let chunk_rows = (MAX_DIST_ELEMENTS / n.max(1)).clamp(1, QUERY_CHUNK_ROWS);
        let eps_sq = self.eps * self.eps;
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

        let mut start = 0;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let q_chunk = slice_rows_matrix(x, start, end)?;
            let (dist_mat, is_sq) = compute_chunk_distances(&backend, &q_chunk, x, dist_metric)?;
            let thresh = if is_sq { eps_sq } else { self.eps };

            for local_r in 0..(end - start) {
                let global_r = start + local_r;
                let row_slice = &dist_mat.data[local_r * n..(local_r + 1) * n];
                let mut adj = Vec::new();
                for (j, &d) in row_slice.iter().enumerate() {
                    if d.max(0.0) <= thresh {
                        adj.push(j);
                    }
                }
                neighbors[global_r] = adj;
            }
            start = end;
        }

        let is_core: Vec<bool> = neighbors
            .iter()
            .map(|adj| adj.len() >= self.min_samples)
            .collect();

        let mut core_indices = Vec::new();
        for (i, &core) in is_core.iter().enumerate() {
            if core {
                core_indices.push(i);
            }
        }

        let mut labels = vec![-1i64; n];
        let mut cluster_id = 0i64;
        let mut queue = VecDeque::new();

        for i in 0..n {
            if labels[i] != -1 || !is_core[i] {
                continue;
            }
            labels[i] = cluster_id;
            queue.push_back(i);
            while let Some(curr) = queue.pop_front() {
                if is_core[curr] {
                    for &nb in &neighbors[curr] {
                        if labels[nb] == -1 {
                            labels[nb] = cluster_id;
                            queue.push_back(nb);
                        }
                    }
                }
            }
            cluster_id += 1;
        }

        let components = if core_indices.is_empty() {
            None
        } else {
            let mut comp_data = Vec::with_capacity(core_indices.len() * p);
            for &idx in &core_indices {
                comp_data.extend_from_slice(&x.data[idx * p..(idx + 1) * p]);
            }
            Some(Matrix::from_vec(core_indices.len(), p, comp_data)?)
        };

        self.labels_ = Some(labels);
        self.core_sample_indices_ = Some(core_indices);
        self.components_ = components;
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<i64>> {
        self.fit(x)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_DBSCAN);
        write_f64(&mut buf, self.eps);
        write_usize(&mut buf, self.min_samples);
        write_str(&mut buf, &self.metric);
        write_opt_i64_vec(&mut buf, &self.labels_);
        write_opt_usize_vec(&mut buf, &self.core_sample_indices_);
        write_opt_matrix(&mut buf, &self.components_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DBSCAN)?;
        let eps = r.read_f64()?;
        let min_samples = r.read_usize()?;
        let metric = r.read_str()?;
        let labels_ = read_opt_i64_vec(&mut r)?;
        let core_sample_indices_ = r.read_opt_usize_vec()?;
        let components_ = r.read_opt_matrix()?;
        let n_features_in_ = r.read_opt_usize()?;
        let saved_dev = r.read_opt_str()?;
        let (device_type, device_) =
            resolve_loaded_device(saved_dev, n_features_in_.is_some(), device)?;
        Ok(Self {
            eps,
            min_samples,
            metric,
            labels_,
            core_sample_indices_,
            components_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// 2. OPTICS (TAG_OPTICS = 86) (*≧ω≦*)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct ReachEntry {
    reach: f64,
    idx: usize,
}

impl PartialEq for ReachEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.reach.to_bits() == other.reach.to_bits()
    }
}

impl Eq for ReachEntry {}

impl PartialOrd for ReachEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReachEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.reach
            .total_cmp(&other.reach)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

#[derive(Clone, Debug)]
struct SteepDownArea {
    start: usize,
    end: usize,
    mib: f64,
}

fn extend_xi_region(
    steep_mask: &[bool],
    opposite_mask: &[bool],
    start: usize,
    min_samples: usize,
) -> usize {
    let n = steep_mask.len();
    let mut end = start;
    let mut non_steep_count = 0usize;
    let mut idx = start + 1;
    while idx < n {
        if opposite_mask[idx] {
            break;
        }
        if steep_mask[idx] {
            non_steep_count = 0;
            end = idx;
        } else {
            non_steep_count += 1;
            if non_steep_count > min_samples {
                break;
            }
        }
        idx += 1;
    }
    end
}

fn extract_xi_clusters(
    ordering: &[usize],
    reachability: &[f64],
    xi: f64,
    min_samples: usize,
    min_cluster_size: usize,
) -> Vec<i64> {
    let n = ordering.len();
    let mut labels = vec![-1i64; n];
    if n < 2 || min_cluster_size > n {
        return labels;
    }

    let mut r = Vec::with_capacity(n + 1);
    for &idx in ordering {
        r.push(reachability[idx]);
    }
    r.push(f64::INFINITY);

    let one_minus_xi = 1.0 - xi;
    let mut steep_up = vec![false; n];
    let mut steep_down = vec![false; n];
    let mut upward = vec![false; n];
    let mut downward = vec![false; n];
    for i in 0..n {
        let a = r[i];
        let b = r[i + 1];
        if a.is_infinite() {
            if b.is_finite() {
                steep_down[i] = true;
                downward[i] = true;
            }
        } else if b.is_infinite() {
            if a.is_finite() {
                steep_up[i] = true;
                upward[i] = true;
            }
        } else {
            steep_up[i] = a <= b * one_minus_xi;
            steep_down[i] = a * one_minus_xi >= b;
            upward[i] = a < b;
            downward[i] = a > b;
        }
    }

    let mut sdas: Vec<SteepDownArea> = Vec::new();
    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut mib = 0.0f64;
    let mut index = 0usize;

    for steep_idx in 0..n {
        if !(steep_up[steep_idx] || steep_down[steep_idx]) || steep_idx < index {
            continue;
        }
        for &val in &r[index..=steep_idx] {
            if val > mib {
                mib = val;
            }
        }

        if steep_down[steep_idx] {
            sdas.retain(|sda| r[sda.start] * one_minus_xi >= mib);
            for sda in sdas.iter_mut() {
                sda.mib = sda.mib.max(mib);
            }
            let d_end = extend_xi_region(&steep_down, &upward, steep_idx, min_samples);
            sdas.push(SteepDownArea {
                start: steep_idx,
                end: d_end,
                mib: 0.0,
            });
            index = d_end + 1;
            mib = r[index];
        } else {
            sdas.retain(|sda| r[sda.start] * one_minus_xi >= mib);
            for sda in sdas.iter_mut() {
                sda.mib = sda.mib.max(mib);
            }
            let u_start = steep_idx;
            let u_end = extend_xi_region(&steep_up, &downward, u_start, min_samples);
            index = u_end + 1;
            mib = r[index];

            let mut u_clusters = Vec::new();
            let up_max = r[u_end + 1];
            for sda in &sdas {
                if up_max * one_minus_xi < sda.mib {
                    continue;
                }
                let r_start = r[sda.start];
                let mut c_start = sda.start;
                let mut c_end = u_end;

                if r_start.is_infinite() {
                    if up_max.is_finite() {
                        while c_start < sda.end && r[c_start + 1] > up_max {
                            c_start += 1;
                        }
                    }
                } else if up_max.is_infinite() {
                    while c_end > u_start && r[c_end - 1] > r_start {
                        c_end -= 1;
                    }
                } else if r_start * one_minus_xi >= up_max {
                    while c_start < sda.end && r[c_start + 1] > up_max {
                        c_start += 1;
                    }
                } else if up_max * one_minus_xi >= r_start {
                    while c_end > u_start && r[c_end - 1] > r_start {
                        c_end -= 1;
                    }
                }

                if c_start <= sda.end
                    && c_end >= u_start
                    && c_end >= c_start
                    && (c_end - c_start + 1) >= min_cluster_size
                {
                    u_clusters.push((c_start, c_end));
                }
            }
            u_clusters.reverse();
            clusters.extend(u_clusters);
        }
    }

    clusters.sort_by_key(|&(s, e)| (e - s, s));
    let mut assigned = vec![false; n];
    let mut next_label = 0i64;
    for &(s, e) in &clusters {
        if (s..=e).any(|pos| assigned[pos]) {
            continue;
        }
        for pos in s..=e {
            assigned[pos] = true;
            labels[ordering[pos]] = next_label;
        }
        next_label += 1;
    }

    labels
}

#[derive(Debug, Clone)]
pub struct OPTICS {
    pub min_samples: usize,
    pub max_eps: f64,
    pub metric: String,
    pub xi: f64,
    pub min_cluster_size: Option<usize>,
    pub labels_: Option<Vec<i64>>,
    pub reachability_: Option<Vec<f64>>,
    pub core_distances_: Option<Vec<f64>>,
    pub ordering_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl OPTICS {
    pub fn new(
        min_samples: usize,
        max_eps: f64,
        metric: String,
        xi: f64,
        min_cluster_size: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if min_samples == 0 {
            return Err(OchreError::InvalidParameter(
                "min_samples must be >= 1 ( >_< )".to_string(),
            ));
        }
        if max_eps.is_nan() || max_eps <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "max_eps must be > 0 or infinity, got {} ( >_< )",
                max_eps
            )));
        }
        if !xi.is_finite() || !(0.0..=1.0).contains(&xi) {
            return Err(OchreError::InvalidParameter(format!(
                "xi must be in [0.0, 1.0], got {} ( >_< )",
                xi
            )));
        }
        if let Some(mcs) = min_cluster_size
            && mcs == 0
        {
            return Err(OchreError::InvalidParameter(
                "min_cluster_size must be >= 1 when specified ( >_< )".to_string(),
            ));
        }
        DistMetric::parse(&metric)?;
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            min_samples,
            max_eps,
            metric,
            xi,
            min_cluster_size,
            labels_: None,
            reachability_: None,
            core_distances_: None,
            ordering_: None,
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

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        let dist_metric = DistMetric::parse(&self.metric)?;
        let backend = get_backend(&self.device_type);
        let n = x.rows;
        let p = x.cols;

        // Compute core distances via N3 + N4
        let mut core_distances = vec![f64::INFINITY; n];
        if self.min_samples <= n {
            let (_idx, dists) =
                chunked_kneighbors(&backend, x, x, self.min_samples, dist_metric, false)?;
            for (i, cd) in core_distances.iter_mut().enumerate() {
                let d = dists.data[i * self.min_samples + (self.min_samples - 1)];
                if d <= self.max_eps {
                    *cd = d;
                }
            }
        }

        // Compute pairwise distances for neighbor expansion
        let dist_mat = compute_full_pairwise_distances(&backend, x, dist_metric)?;

        let mut processed = vec![false; n];
        let mut reachability = vec![f64::INFINITY; n];
        let mut ordering = Vec::with_capacity(n);
        let mut heap: BinaryHeap<Reverse<ReachEntry>> = BinaryHeap::new();

        for start in 0..n {
            if processed[start] {
                continue;
            }
            processed[start] = true;
            ordering.push(start);

            let cd_start = core_distances[start];
            if cd_start.is_finite() {
                let row = &dist_mat.data[start * n..(start + 1) * n];
                for (j, &d) in row.iter().enumerate() {
                    if !processed[j] && d <= self.max_eps {
                        let new_reach = cd_start.max(d);
                        if new_reach < reachability[j] {
                            reachability[j] = new_reach;
                            heap.push(Reverse(ReachEntry {
                                reach: new_reach,
                                idx: j,
                            }));
                        }
                    }
                }

                while let Some(Reverse(entry)) = heap.pop() {
                    let q = entry.idx;
                    if processed[q] {
                        continue;
                    }
                    processed[q] = true;
                    ordering.push(q);

                    let cd_q = core_distances[q];
                    if cd_q.is_finite() {
                        let q_row = &dist_mat.data[q * n..(q + 1) * n];
                        for (j, &d) in q_row.iter().enumerate() {
                            if !processed[j] && d <= self.max_eps {
                                let new_reach = cd_q.max(d);
                                if new_reach < reachability[j] {
                                    reachability[j] = new_reach;
                                    heap.push(Reverse(ReachEntry {
                                        reach: new_reach,
                                        idx: j,
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mcs = self
            .min_cluster_size
            .unwrap_or(self.min_samples.max(2))
            .max(2);
        let labels = extract_xi_clusters(&ordering, &reachability, self.xi, self.min_samples, mcs);

        self.labels_ = Some(labels);
        self.reachability_ = Some(reachability);
        self.core_distances_ = Some(core_distances);
        self.ordering_ = Some(ordering);
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<i64>> {
        self.fit(x)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_OPTICS);
        write_usize(&mut buf, self.min_samples);
        write_f64(&mut buf, self.max_eps);
        write_str(&mut buf, &self.metric);
        write_f64(&mut buf, self.xi);
        write_opt_usize(&mut buf, self.min_cluster_size);
        write_opt_i64_vec(&mut buf, &self.labels_);
        write_opt_f64_vec(&mut buf, &self.reachability_);
        write_opt_f64_vec(&mut buf, &self.core_distances_);
        write_opt_usize_vec(&mut buf, &self.ordering_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_OPTICS)?;
        let min_samples = r.read_usize()?;
        let max_eps = r.read_f64()?;
        let metric = r.read_str()?;
        let xi = r.read_f64()?;
        let min_cluster_size = r.read_opt_usize()?;
        let labels_ = read_opt_i64_vec(&mut r)?;
        let reachability_ = r.read_opt_f64_vec()?;
        let core_distances_ = r.read_opt_f64_vec()?;
        let ordering_ = r.read_opt_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let saved_dev = r.read_opt_str()?;
        let (device_type, device_) =
            resolve_loaded_device(saved_dev, n_features_in_.is_some(), device)?;
        Ok(Self {
            min_samples,
            max_eps,
            metric,
            xi,
            min_cluster_size,
            labels_,
            reachability_,
            core_distances_,
            ordering_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// 3. SpectralClustering (TAG_SPECTRAL_CLUSTERING = 87) (*≧ω≦*)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpectralClustering {
    pub n_clusters: usize,
    pub affinity: String,
    pub gamma: f64,
    pub n_neighbors: usize,
    pub n_init: usize,
    pub random_state: Option<u64>,
    pub labels_: Option<Vec<usize>>,
    pub affinity_matrix_: Option<Matrix>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl SpectralClustering {
    pub fn new(
        n_clusters: usize,
        affinity: String,
        gamma: f64,
        n_neighbors: usize,
        n_init: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_clusters == 0 {
            return Err(OchreError::InvalidParameter(
                "n_clusters must be >= 1 ( >_< )".to_string(),
            ));
        }
        if affinity != "rbf" && affinity != "nearest_neighbors" {
            return Err(OchreError::InvalidParameter(format!(
                "affinity must be 'rbf' or 'nearest_neighbors', got '{}' (・`ω´・)",
                affinity
            )));
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "gamma must be positive and finite, got {} ( >_< )",
                gamma
            )));
        }
        if n_neighbors == 0 {
            return Err(OchreError::InvalidParameter(
                "n_neighbors must be >= 1 ( >_< )".to_string(),
            ));
        }
        if n_init == 0 {
            return Err(OchreError::InvalidParameter(
                "n_init must be >= 1 ( >_< )".to_string(),
            ));
        }
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            n_clusters,
            affinity,
            gamma,
            n_neighbors,
            n_init,
            random_state,
            labels_: None,
            affinity_matrix_: None,
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

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        let n = x.rows;
        let p = x.cols;
        if self.n_clusters > n {
            return Err(OchreError::InvalidParameter(format!(
                "n_clusters={} must be <= n_samples={} ( >_< )",
                self.n_clusters, n
            )));
        }
        let backend = get_backend(&self.device_type);

        // 1. Build symmetric affinity matrix W (n x n)
        let w = if self.affinity == "rbf" {
            let mut dist_sq = backend.pairwise_l2_sq(x, x)?;
            for v in dist_sq.data.iter_mut() {
                *v = (-self.gamma * v.max(0.0)).exp();
            }
            dist_sq
        } else {
            let k = self.n_neighbors.min(n);
            let (nn_idx, _) = chunked_kneighbors(&backend, x, x, k, DistMetric::Euclidean, false)?;
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..k {
                    let nb = nn_idx[i * k + j];
                    a[i * n + nb] = 1.0;
                }
            }
            let mut sym = vec![0.0; n * n];
            for i in 0..n {
                for j in i..n {
                    let val = 0.5 * (a[i * n + j] + a[j * n + i]);
                    sym[i * n + j] = val;
                    sym[j * n + i] = val;
                }
            }
            Matrix::from_vec(n, n, sym)?
        };

        // 2. Compute normalized affinity M = D^{-1/2} W D^{-1/2}
        let mut d_inv_sqrt = vec![0.0; n];
        for (i, slot) in d_inv_sqrt.iter_mut().enumerate() {
            let row_sum: f64 = w.data[i * n..(i + 1) * n].iter().sum();
            if row_sum > 0.0 {
                *slot = 1.0 / row_sum.sqrt();
            }
        }

        let mut m_data = vec![0.0; n * n];
        for i in 0..n {
            let di = d_inv_sqrt[i];
            for j in i..n {
                let val = w.data[i * n + j] * di * d_inv_sqrt[j];
                m_data[i * n + j] = val;
                m_data[j * n + i] = val;
            }
        }
        let m_mat = Matrix::from_vec(n, n, m_data)?;

        // 3. Top k eigenvectors of D^{-1/2} W D^{-1/2} (smallest eigenvectors of L_sym) via N5
        let k = self.n_clusters;
        let eig = if n <= 48 || k.saturating_mul(3) >= n {
            backend.small_sym_eig_svd(&m_mat)?
        } else {
            crate::primitives::eig_svd::cpu_top_k_sym_eig(&m_mat, k)?
        };
        let eig_cols = eig.eigenvectors.cols;
        let mut u_data = vec![0.0; n * k];
        for i in 0..n {
            let mut norm_sq = 0.0;
            for c in 0..k {
                let val = eig.eigenvectors.data[i * eig_cols + c];
                u_data[i * k + c] = val;
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for c in 0..k {
                    u_data[i * k + c] *= inv;
                }
            }
        }
        let u_mat = Matrix::from_vec(n, k, u_data)?;

        // 4. Cluster rows of U with KMeans (stay on CPU for small n * k to avoid GPU launch overhead)
        let kmeans_device = if n.saturating_mul(k) < 131_072 {
            Some("cpu".to_string())
        } else {
            self.device_.clone()
        };
        let mut kmeans = KMeans::new(
            k,
            "k-means++".to_string(),
            self.n_init,
            300,
            1e-4,
            self.random_state,
            kmeans_device,
        )?;
        kmeans.fit(&u_mat, None)?;

        self.labels_ = kmeans.labels_;
        self.affinity_matrix_ = Some(w);
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<usize>> {
        self.fit(x)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SPECTRAL_CLUSTERING);
        write_usize(&mut buf, self.n_clusters);
        write_str(&mut buf, &self.affinity);
        write_f64(&mut buf, self.gamma);
        write_usize(&mut buf, self.n_neighbors);
        write_usize(&mut buf, self.n_init);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_usize_vec(&mut buf, &self.labels_);
        write_opt_matrix(&mut buf, &self.affinity_matrix_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SPECTRAL_CLUSTERING)?;
        let n_clusters = r.read_usize()?;
        let affinity = r.read_str()?;
        let gamma = r.read_f64()?;
        let n_neighbors = r.read_usize()?;
        let n_init = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let labels_ = r.read_opt_usize_vec()?;
        let affinity_matrix_ = r.read_opt_matrix()?;
        let n_features_in_ = r.read_opt_usize()?;
        let saved_dev = r.read_opt_str()?;
        let (device_type, device_) =
            resolve_loaded_device(saved_dev, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_clusters,
            affinity,
            gamma,
            n_neighbors,
            n_init,
            random_state,
            labels_,
            affinity_matrix_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// 4. AgglomerativeClustering (TAG_AGGLOMERATIVE_CLUSTERING = 88) (*≧ω≦*)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Linkage {
    Ward,
    Complete,
    Average,
    Single,
}

impl Linkage {
    fn parse(s: &str) -> OchreResult<Self> {
        match s {
            "ward" => Ok(Self::Ward),
            "complete" => Ok(Self::Complete),
            "average" => Ok(Self::Average),
            "single" => Ok(Self::Single),
            _ => Err(OchreError::InvalidParameter(format!(
                "linkage must be 'ward', 'complete', 'average', or 'single', got '{}' (・`ω´・)",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    pub n_clusters: usize,
    pub metric: String,
    pub linkage: String,
    pub labels_: Option<Vec<usize>>,
    pub n_leaves_: usize,
    pub n_connected_components_: usize,
    pub children_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl AgglomerativeClustering {
    pub fn new(
        n_clusters: usize,
        metric: String,
        linkage: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_clusters == 0 {
            return Err(OchreError::InvalidParameter(
                "n_clusters must be >= 1 ( >_< )".to_string(),
            ));
        }
        let link = Linkage::parse(&linkage)?;
        let met = DistMetric::parse(&metric)?;
        if link == Linkage::Ward && met != DistMetric::Euclidean {
            return Err(OchreError::InvalidParameter(format!(
                "Ward linkage only supports 'euclidean' metric, got '{}' (・`ω´・)",
                metric
            )));
        }
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            n_clusters,
            metric,
            linkage,
            labels_: None,
            n_leaves_: 0,
            n_connected_components_: 1,
            children_: None,
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

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        let n = x.rows;
        let p = x.cols;
        if self.n_clusters > n {
            return Err(OchreError::InvalidParameter(format!(
                "n_clusters={} must be <= n_samples={} ( >_< )",
                self.n_clusters, n
            )));
        }
        let link = Linkage::parse(&self.linkage)?;
        let met = DistMetric::parse(&self.metric)?;
        if link == Linkage::Ward && met != DistMetric::Euclidean {
            return Err(OchreError::InvalidParameter(format!(
                "Ward linkage only supports 'euclidean' metric, got '{}' (・`ω´・)",
                self.metric
            )));
        }

        if n == 1 {
            self.labels_ = Some(vec![0]);
            self.n_leaves_ = 1;
            self.n_connected_components_ = 1;
            self.children_ = Some(Vec::new());
            self.n_features_in_ = Some(p);
            return Ok(());
        }

        let backend = get_backend(&self.device_type);
        let mut dist = compute_full_pairwise_distances(&backend, x, met)?.data;

        let mut active = vec![true; n];
        let mut size = vec![1usize; n];
        let mut node_id: Vec<usize> = (0..n).collect();
        let mut next_id = n;
        let mut children = Vec::with_capacity(2 * (n - 1));

        for _step in 0..(n - 1) {
            let mut best_a = 0usize;
            let mut best_b = 0usize;
            let mut best_d = f64::INFINITY;

            for a in 0..n {
                if !active[a] {
                    continue;
                }
                let row = &dist[a * n..(a + 1) * n];
                for b in (a + 1)..n {
                    if active[b] && row[b] < best_d {
                        best_d = row[b];
                        best_a = a;
                        best_b = b;
                    }
                }
            }

            let id_a = node_id[best_a];
            let id_b = node_id[best_b];
            children.push(id_a.min(id_b));
            children.push(id_a.max(id_b));

            node_id[best_a] = next_id;
            next_id += 1;
            active[best_b] = false;

            let n_a = size[best_a] as f64;
            let n_b = size[best_b] as f64;
            let d_ab = best_d;

            for k in 0..n {
                if !active[k] || k == best_a {
                    continue;
                }
                let d_ak = dist[best_a * n + k];
                let d_bk = dist[best_b * n + k];
                let d_new = match link {
                    Linkage::Single => d_ak.min(d_bk),
                    Linkage::Complete => d_ak.max(d_bk),
                    Linkage::Average => (n_a * d_ak + n_b * d_bk) / (n_a + n_b),
                    Linkage::Ward => {
                        let n_k = size[k] as f64;
                        let sq = ((n_a + n_k) * d_ak * d_ak + (n_b + n_k) * d_bk * d_bk
                            - n_k * d_ab * d_ab)
                            / (n_a + n_b + n_k);
                        sq.max(0.0).sqrt()
                    }
                };
                dist[best_a * n + k] = d_new;
                dist[k * n + best_a] = d_new;
            }
            size[best_a] += size[best_b];
        }

        // Reconstruct labels from the first (n - n_clusters) merges via Union-Find
        let total_nodes = 2 * n - 1;
        let mut parent: Vec<usize> = (0..total_nodes).collect();
        let find_root = |parent: &mut [usize], mut x: usize| -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };

        let merges = n - self.n_clusters;
        for step in 0..merges {
            let c0 = children[step * 2];
            let c1 = children[step * 2 + 1];
            let merge_node = n + step;
            let r0 = find_root(&mut parent, c0);
            let r1 = find_root(&mut parent, c1);
            parent[r0] = merge_node;
            parent[r1] = merge_node;
        }

        let mut root_to_label = vec![usize::MAX; total_nodes];
        let mut next_lbl = 0usize;
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let root = find_root(&mut parent, i);
            if root_to_label[root] == usize::MAX {
                root_to_label[root] = next_lbl;
                next_lbl += 1;
            }
            labels.push(root_to_label[root]);
        }

        self.labels_ = Some(labels);
        self.n_leaves_ = n;
        self.n_connected_components_ = 1;
        self.children_ = Some(children);
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<usize>> {
        self.fit(x)?;
        self.labels_.clone().ok_or(OchreError::NotFitted)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_AGGLOMERATIVE_CLUSTERING);
        write_usize(&mut buf, self.n_clusters);
        write_str(&mut buf, &self.metric);
        write_str(&mut buf, &self.linkage);
        write_opt_usize_vec(&mut buf, &self.labels_);
        write_usize(&mut buf, self.n_leaves_);
        write_usize(&mut buf, self.n_connected_components_);
        write_opt_usize_vec(&mut buf, &self.children_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_AGGLOMERATIVE_CLUSTERING)?;
        let n_clusters = r.read_usize()?;
        let metric = r.read_str()?;
        let linkage = r.read_str()?;
        let labels_ = r.read_opt_usize_vec()?;
        let n_leaves_ = r.read_usize()?;
        let n_connected_components_ = r.read_usize()?;
        let children_ = r.read_opt_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let saved_dev = r.read_opt_str()?;
        let (device_type, device_) =
            resolve_loaded_device(saved_dev, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_clusters,
            metric,
            linkage,
            labels_,
            n_leaves_,
            n_connected_components_,
            children_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// 5. LocalOutlierFactor (TAG_LOCAL_OUTLIER_FACTOR = 89) (*≧ω≦*)
// ============================================================================

fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return -1.5;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    pub n_neighbors: usize,
    pub contamination: Option<f64>,
    pub novelty: bool,
    pub metric: String,
    pub negative_outlier_factor_: Option<Vec<f64>>,
    pub offset_: f64,
    pub n_neighbors_: usize,
    pub x_train_: Option<Matrix>,
    pub k_distance_: Option<Vec<f64>>,
    pub lrd_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub device_: Option<String>,
    device_type: DeviceType,
}

impl LocalOutlierFactor {
    pub fn new(
        n_neighbors: usize,
        contamination: Option<f64>,
        novelty: bool,
        metric: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_neighbors == 0 {
            return Err(OchreError::InvalidParameter(
                "n_neighbors must be >= 1 ( >_< )".to_string(),
            ));
        }
        if let Some(c) = contamination
            && (!c.is_finite() || c <= 0.0 || c > 0.5)
        {
            return Err(OchreError::InvalidParameter(format!(
                "contamination must be in (0.0, 0.5], got {} ( >_< )",
                c
            )));
        }
        DistMetric::parse(&metric)?;
        let dev = parse_device(device.as_deref().or(Some("auto")))?;
        Ok(Self {
            n_neighbors,
            contamination,
            novelty,
            metric,
            negative_outlier_factor_: None,
            offset_: -1.5,
            n_neighbors_: 0,
            x_train_: None,
            k_distance_: None,
            lrd_: None,
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

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x(x)?;
        let n = x.rows;
        let p = x.cols;
        if n < 2 {
            return Err(OchreError::InvalidParameter(
                "LocalOutlierFactor requires at least 2 samples ( >_< )".to_string(),
            ));
        }
        let dist_metric = DistMetric::parse(&self.metric)?;
        let backend = get_backend(&self.device_type);

        let k = self.n_neighbors.min(n - 1);
        let (nn_idx, nn_dists) = chunked_kneighbors(&backend, x, x, k, dist_metric, true)?;

        let mut k_distance = Vec::with_capacity(n);
        for i in 0..n {
            k_distance.push(nn_dists.data[i * k + (k - 1)]);
        }

        let mut lrd = Vec::with_capacity(n);
        for i in 0..n {
            let mut reach_sum = 0.0;
            for j in 0..k {
                let nb = nn_idx[i * k + j];
                let d = nn_dists.data[i * k + j];
                reach_sum += k_distance[nb].max(d);
            }
            let mean_reach = reach_sum / (k as f64);
            lrd.push(1.0 / (mean_reach + 1e-10));
        }

        let mut nof = Vec::with_capacity(n);
        for i in 0..n {
            let mut lrd_ratio_sum = 0.0;
            for j in 0..k {
                let nb = nn_idx[i * k + j];
                lrd_ratio_sum += lrd[nb] / lrd[i];
            }
            let lof = lrd_ratio_sum / (k as f64);
            nof.push(-lof);
        }

        let offset = match self.contamination {
            None => -1.5,
            Some(c) => {
                let mut sorted = nof.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                quantile_linear(&sorted, c)
            }
        };

        self.negative_outlier_factor_ = Some(nof);
        self.offset_ = offset;
        self.n_neighbors_ = k;
        self.x_train_ = Some(x.clone());
        self.k_distance_ = Some(k_distance);
        self.lrd_ = Some(lrd);
        self.n_features_in_ = Some(p);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> OchreResult<Vec<i64>> {
        if self.novelty {
            return Err(OchreError::InvalidParameter(
                "fit_predict is not available when novelty=True. Use fit() followed by predict() (・`ω´・)".to_string(),
            ));
        }
        self.fit(x)?;
        let nof = self
            .negative_outlier_factor_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        Ok(nof
            .iter()
            .map(|&v| if v >= self.offset_ { 1 } else { -1 })
            .collect())
    }

    pub fn score_samples(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if !self.novelty {
            return Err(OchreError::InvalidParameter(
                "score_samples is not available when novelty=False. Set novelty=True (・`ω´・)"
                    .to_string(),
            ));
        }
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let x_train = self.x_train_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_dist_train = self.k_distance_.as_ref().ok_or(OchreError::NotFitted)?;
        let lrd_train = self.lrd_.as_ref().ok_or(OchreError::NotFitted)?;

        let dist_metric = DistMetric::parse(&self.metric)?;
        let backend = get_backend(&self.device_type);
        let k = self.n_neighbors_;
        let (nn_idx, nn_dists) = chunked_kneighbors(&backend, x, x_train, k, dist_metric, false)?;

        let n_q = x.rows;
        let mut scores = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let mut reach_sum = 0.0;
            for j in 0..k {
                let nb = nn_idx[i * k + j];
                let d = nn_dists.data[i * k + j];
                reach_sum += k_dist_train[nb].max(d);
            }
            let lrd_q = 1.0 / (reach_sum / (k as f64) + 1e-10);
            let mut ratio_sum = 0.0;
            for j in 0..k {
                let nb = nn_idx[i * k + j];
                ratio_sum += lrd_train[nb] / lrd_q;
            }
            scores.push(-(ratio_sum / (k as f64)));
        }
        Ok(scores)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if !self.novelty {
            return Err(OchreError::InvalidParameter(
                "decision_function is not available when novelty=False. Set novelty=True (・`ω´・)"
                    .to_string(),
            ));
        }
        let scores = self.score_samples(x)?;
        Ok(scores.into_iter().map(|s| s - self.offset_).collect())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<i64>> {
        if !self.novelty {
            return Err(OchreError::InvalidParameter(
                "predict is not available when novelty=False. Set novelty=True (・`ω´・)"
                    .to_string(),
            ));
        }
        let df = self.decision_function(x)?;
        Ok(df
            .into_iter()
            .map(|v| if v >= 0.0 { 1 } else { -1 })
            .collect())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LOCAL_OUTLIER_FACTOR);
        write_usize(&mut buf, self.n_neighbors);
        write_opt_f64(&mut buf, self.contamination);
        write_bool(&mut buf, self.novelty);
        write_str(&mut buf, &self.metric);
        write_opt_f64_vec(&mut buf, &self.negative_outlier_factor_);
        write_f64(&mut buf, self.offset_);
        write_usize(&mut buf, self.n_neighbors_);
        write_opt_matrix(&mut buf, &self.x_train_);
        write_opt_f64_vec(&mut buf, &self.k_distance_);
        write_opt_f64_vec(&mut buf, &self.lrd_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.device_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LOCAL_OUTLIER_FACTOR)?;
        let n_neighbors = r.read_usize()?;
        let contamination = r.read_opt_f64()?;
        let novelty = r.read_bool()?;
        let metric = r.read_str()?;
        let negative_outlier_factor_ = r.read_opt_f64_vec()?;
        let offset_ = r.read_f64()?;
        let n_neighbors_ = r.read_usize()?;
        let x_train_ = r.read_opt_matrix()?;
        let k_distance_ = r.read_opt_f64_vec()?;
        let lrd_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let saved_dev = r.read_opt_str()?;
        let (device_type, device_) =
            resolve_loaded_device(saved_dev, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_neighbors,
            contamination,
            novelty,
            metric,
            negative_outlier_factor_,
            offset_,
            n_neighbors_,
            x_train_,
            k_distance_,
            lrd_,
            n_features_in_,
            device_,
            device_type,
        })
    }
}

// ============================================================================
// PyO3 Bindings (*≧ω≦*)
// ============================================================================

#[pyclass(from_py_object, name = "DBSCAN")]
#[derive(Clone)]
pub struct PyDBSCAN {
    inner: DBSCAN,
}

crate::pymethods_with_dit!(PyDBSCAN, DBSCAN, {
    #[new]
    #[pyo3(signature = (
        eps = 0.5,
        min_samples = 5,
        metric = "euclidean".to_string(),
        device = None
    ))]
    pub fn new(
        eps: f64,
        min_samples: usize,
        metric: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DBSCAN::new(eps, min_samples, metric, device)?,
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
        i64_slice_to_pybytes(py, &labels)
    }

    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(l) => Ok(Some(i64_slice_to_pybytes(py, l)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn core_sample_indices_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.core_sample_indices_ {
            Some(idx) => Ok(Some(usize_slice_to_i64_pybytes(py, idx)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn components_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.components_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    pub fn min_samples(&self) -> usize {
        self.inner.min_samples
    }

    #[getter]
    pub fn metric(&self) -> String {
        self.inner.metric.clone()
    }
});

#[pyclass(from_py_object, name = "OPTICS")]
#[derive(Clone)]
pub struct PyOPTICS {
    inner: OPTICS,
}

crate::pymethods_with_dit!(PyOPTICS, OPTICS, {
    #[new]
    #[pyo3(signature = (
        min_samples = 5,
        max_eps = f64::INFINITY,
        metric = "euclidean".to_string(),
        xi = 0.05,
        min_cluster_size = None,
        device = None
    ))]
    pub fn new(
        min_samples: usize,
        max_eps: f64,
        metric: String,
        xi: f64,
        min_cluster_size: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OPTICS::new(min_samples, max_eps, metric, xi, min_cluster_size, device)?,
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
        i64_slice_to_pybytes(py, &labels)
    }

    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(l) => Ok(Some(i64_slice_to_pybytes(py, l)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn reachability_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.reachability_ {
            Some(r) => Ok(Some(f64_slice_to_pybytes(py, r)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn core_distances_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.core_distances_ {
            Some(c) => Ok(Some(f64_slice_to_pybytes(py, c)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn ordering_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.ordering_ {
            Some(o) => Ok(Some(usize_slice_to_i64_pybytes(py, o)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn min_samples(&self) -> usize {
        self.inner.min_samples
    }

    #[getter]
    pub fn max_eps(&self) -> f64 {
        self.inner.max_eps
    }

    #[getter]
    pub fn metric(&self) -> String {
        self.inner.metric.clone()
    }

    #[getter]
    pub fn xi(&self) -> f64 {
        self.inner.xi
    }

    #[getter]
    pub fn min_cluster_size(&self) -> Option<usize> {
        self.inner.min_cluster_size
    }
});

#[pyclass(from_py_object, name = "SpectralClustering")]
#[derive(Clone)]
pub struct PySpectralClustering {
    inner: SpectralClustering,
}

crate::pymethods_with_dit!(PySpectralClustering, SpectralClustering, {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8,
        affinity = "rbf".to_string(),
        gamma = 1.0,
        n_neighbors = 10,
        n_init = 10,
        random_state = None,
        device = None
    ))]
    pub fn new(
        n_clusters: usize,
        affinity: String,
        gamma: f64,
        n_neighbors: usize,
        n_init: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralClustering::new(
                n_clusters,
                affinity,
                gamma,
                n_neighbors,
                n_init,
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

    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(l) => Ok(Some(usize_slice_to_i64_pybytes(py, l)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn affinity_matrix_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.affinity_matrix_ {
            Some(mat) => Ok(Some(matrix_to_pybytes(py, mat)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    #[getter]
    pub fn affinity(&self) -> String {
        self.inner.affinity.clone()
    }

    #[getter]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    pub fn n_neighbors(&self) -> usize {
        self.inner.n_neighbors
    }

    #[getter]
    pub fn n_init(&self) -> usize {
        self.inner.n_init
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
});

#[pyclass(from_py_object, name = "AgglomerativeClustering")]
#[derive(Clone)]
pub struct PyAgglomerativeClustering {
    inner: AgglomerativeClustering,
}

crate::pymethods_with_dit!(PyAgglomerativeClustering, AgglomerativeClustering, {
    #[new]
    #[pyo3(signature = (
        n_clusters = 2,
        metric = "euclidean".to_string(),
        linkage = "ward".to_string(),
        device = None
    ))]
    pub fn new(
        n_clusters: usize,
        metric: String,
        linkage: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AgglomerativeClustering::new(n_clusters, metric, linkage, device)?,
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

    #[getter]
    pub fn labels_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.labels_ {
            Some(l) => Ok(Some(usize_slice_to_i64_pybytes(py, l)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn children_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.children_ {
            Some(c) => {
                let n_merges = c.len() / 2;
                Ok(Some((n_merges, 2, usize_slice_to_i64_pybytes(py, c)?)))
            }
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_leaves_(&self) -> usize {
        self.inner.n_leaves_
    }

    #[getter]
    pub fn n_connected_components_(&self) -> usize {
        self.inner.n_connected_components_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    #[getter]
    pub fn metric(&self) -> String {
        self.inner.metric.clone()
    }

    #[getter]
    pub fn linkage(&self) -> String {
        self.inner.linkage.clone()
    }
});

#[pyclass(from_py_object, name = "LocalOutlierFactor")]
#[derive(Clone)]
pub struct PyLocalOutlierFactor {
    inner: LocalOutlierFactor,
}

crate::pymethods_with_dit!(PyLocalOutlierFactor, LocalOutlierFactor, {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 20,
        contamination = None,
        novelty = false,
        metric = "euclidean".to_string(),
        device = None
    ))]
    pub fn new(
        n_neighbors: usize,
        contamination: Option<f64>,
        novelty: bool,
        metric: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LocalOutlierFactor::new(n_neighbors, contamination, novelty, metric, device)?,
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
        i64_slice_to_pybytes(py, &labels)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let labels = self.inner.predict(&mat)?;
        i64_slice_to_pybytes(py, &labels)
    }

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let df = self.inner.decision_function(&mat)?;
        f64_slice_to_pybytes(py, &df)
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

    #[getter]
    pub fn negative_outlier_factor_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.negative_outlier_factor_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn offset_(&self) -> f64 {
        self.inner.offset_
    }

    #[getter]
    pub fn n_neighbors_(&self) -> usize {
        self.inner.n_neighbors_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_neighbors(&self) -> usize {
        self.inner.n_neighbors
    }

    #[getter]
    pub fn contamination(&self) -> Option<f64> {
        self.inner.contamination
    }

    #[getter]
    pub fn novelty(&self) -> bool {
        self.inner.novelty
    }

    #[getter]
    pub fn metric(&self) -> String {
        self.inner.metric.clone()
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDBSCAN>()?;
    m.add_class::<PyOPTICS>()?;
    m.add_class::<PySpectralClustering>()?;
    m.add_class::<PyAgglomerativeClustering>()?;
    m.add_class::<PyLocalOutlierFactor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_blobs() -> Matrix {
        Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.1, 10.1],
        ])
        .expect("valid matrix")
    }

    #[test]
    fn test_dbscan_blobs_and_noise() {
        let mut data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.1, 10.1],
            vec![50.0, 50.0],
        ];
        let x = Matrix::from_2d(&data).expect("valid");
        data.clear();
        let mut db =
            DBSCAN::new(0.5, 3, "euclidean".to_string(), Some("cpu".to_string())).expect("ok");
        let labels = db.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
        assert_eq!(labels[8], -1);

        let bytes = db.to_bytes();
        let loaded = DBSCAN::from_bytes(&bytes, Some("cpu")).expect("load");
        assert_eq!(loaded.labels_, db.labels_);
    }

    #[test]
    fn test_optics_blobs() {
        let x = two_blobs();
        let mut opt = OPTICS::new(
            3,
            f64::INFINITY,
            "euclidean".to_string(),
            0.05,
            None,
            Some("cpu".to_string()),
        )
        .expect("ok");
        let labels = opt.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels.len(), 8);
        assert_eq!(opt.ordering_.as_ref().map(|v| v.len()), Some(8));
    }

    #[test]
    fn test_spectral_blobs() {
        let x = two_blobs();
        let mut sp = SpectralClustering::new(
            2,
            "rbf".to_string(),
            1.0,
            3,
            5,
            Some(42),
            Some("cpu".to_string()),
        )
        .expect("ok");
        let labels = sp.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_spectral_topk_eig_large_blobs() {
        let n = 120usize;
        let k = 3usize;
        let mut data = Vec::with_capacity(n * 3);
        for i in 0..n {
            let c = i % k;
            let offset = (c as f64) * 10.0;
            let jitter = ((i % 7) as f64 - 3.0) * 0.05;
            data.push(offset + jitter);
            data.push(offset - jitter);
            data.push(jitter);
        }
        let x = Matrix::from_vec(n, 3, data).expect("valid");
        let mut sp = SpectralClustering::new(
            k,
            "rbf".to_string(),
            0.5,
            5,
            5,
            Some(42),
            Some("auto".to_string()),
        )
        .expect("ok");
        let labels = sp.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels.len(), n);
        for c in 0..k {
            let expected = labels[c];
            for i in (c..n).step_by(k) {
                assert_eq!(labels[i], expected);
            }
        }
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
    }

    #[test]
    fn test_agglomerative_ward() {
        let x = two_blobs();
        let mut agg = AgglomerativeClustering::new(
            2,
            "euclidean".to_string(),
            "ward".to_string(),
            Some("cpu".to_string()),
        )
        .expect("ok");
        let labels = agg.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
        assert_eq!(agg.children_.as_ref().map(|c| c.len()), Some(14));
    }

    #[test]
    fn test_lof_novelty_and_outlier() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![-0.1, 0.0],
            vec![0.0, -0.1],
            vec![20.0, 20.0],
        ])
        .expect("valid");
        let mut lof = LocalOutlierFactor::new(
            3,
            None,
            false,
            "euclidean".to_string(),
            Some("cpu".to_string()),
        )
        .expect("ok");
        let labels = lof.fit_predict(&x).expect("fit_predict");
        assert_eq!(labels[0], 1);
        assert_eq!(labels[6], -1);

        let mut lof_nov = LocalOutlierFactor::new(
            3,
            None,
            true,
            "euclidean".to_string(),
            Some("cpu".to_string()),
        )
        .expect("ok");
        lof_nov.fit(&x).expect("fit");
        let q = Matrix::from_2d(&[vec![0.05, 0.05], vec![30.0, 30.0]]).expect("q");
        let preds = lof_nov.predict(&q).expect("predict");
        assert_eq!(preds, vec![1, -1]);
    }
}
