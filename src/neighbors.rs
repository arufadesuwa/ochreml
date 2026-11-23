// OchreML Neighbors Module (＾▽＾)
// Implements KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier,
// and RadiusNeighborsRegressor backed by chunked distance computation and
// Backend Primitives N3 (pairwise_l2_sq) & N4 (topk_argmin) (*≧ω≦*).

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_matrix, write_opt_str, write_opt_usize, write_str, write_usize,
    write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, r2_score, r2_score_multi};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
    usize_slice_to_i64_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_KNEIGHBORS_CLASSIFIER: u8 = 56;
pub const TAG_KNEIGHBORS_REGRESSOR: u8 = 57;
pub const TAG_RADIUS_NEIGHBORS_CLASSIFIER: u8 = 58;
pub const TAG_RADIUS_NEIGHBORS_REGRESSOR: u8 = 59;

const QUERY_BATCH_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Metric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine,
    Minkowski(f64),
}

impl Metric {
    fn parse(metric: &str, p: f64) -> OchreResult<Self> {
        match metric {
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            "manhattan" | "l1" | "cityblock" => Ok(Metric::Manhattan),
            "chebyshev" => Ok(Metric::Chebyshev),
            "cosine" => Ok(Metric::Cosine),
            "minkowski" => {
                if p.is_nan() || p < 1.0 {
                    return Err(OchreError::InvalidParameter(format!(
                        "Minkowski power p ({}) must be >= 1.0 ( >_< )",
                        p
                    )));
                }
                if (p - 2.0).abs() < 1e-12 {
                    Ok(Metric::Euclidean)
                } else if (p - 1.0).abs() < 1e-12 {
                    Ok(Metric::Manhattan)
                } else if p.is_infinite() && p.is_sign_positive() {
                    Ok(Metric::Chebyshev)
                } else {
                    Ok(Metric::Minkowski(p))
                }
            }
            _ => Err(OchreError::InvalidParameter(format!(
                "Unsupported metric '{}'. Supported: 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski' (・`ω´・)",
                metric
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Weights {
    Uniform,
    Distance,
}

impl Weights {
    fn parse(weights: &str) -> OchreResult<Self> {
        match weights {
            "uniform" => Ok(Weights::Uniform),
            "distance" => Ok(Weights::Distance),
            _ => Err(OchreError::InvalidParameter(format!(
                "weights must be 'uniform' or 'distance', got '{}' (・`ω´・)",
                weights
            ))),
        }
    }
}

fn validate_matrix_finite(x: &Matrix, name: &str) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{} contains NaN or Inf values ( >_< )",
            name
        )));
    }
    Ok(())
}

fn validate_slice_finite(v: &[f64], name: &str) -> OchreResult<()> {
    if v.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{} contains NaN or Inf values ( >_< )",
            name
        )));
    }
    Ok(())
}

fn slice_rows_matrix(x: &Matrix, start_row: usize, end_row: usize) -> OchreResult<Matrix> {
    let n_rows = end_row - start_row;
    let start = start_row * x.cols;
    let end = end_row * x.cols;
    Matrix::from_slice(n_rows, x.cols, &x.data[start..end])
}

#[inline]
fn row_distance(a: &[f64], b: &[f64], metric: Metric) -> f64 {
    match metric {
        Metric::Euclidean => {
            let mut sum = 0.0;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = x - y;
                sum += d * d;
            }
            sum.max(0.0).sqrt()
        }
        Metric::Manhattan => {
            let mut sum = 0.0;
            for (&x, &y) in a.iter().zip(b.iter()) {
                sum += (x - y).abs();
            }
            sum
        }
        Metric::Chebyshev => {
            let mut max_d = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = (x - y).abs();
                if d > max_d {
                    max_d = d;
                }
            }
            max_d
        }
        Metric::Cosine => {
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
        Metric::Minkowski(p) => {
            let mut sum = 0.0;
            for (&x, &y) in a.iter().zip(b.iter()) {
                sum += (x - y).abs().powf(p);
            }
            sum.powf(1.0 / p)
        }
    }
}

fn compute_chunk_distances(
    q_chunk: &Matrix,
    x_train: &Matrix,
    metric: Metric,
    device_type: &DeviceType,
) -> OchreResult<(Matrix, bool)> {
    if metric == Metric::Euclidean {
        let backend = get_backend(device_type);
        let dist_sq = backend.pairwise_l2_sq(q_chunk, x_train)?;
        return Ok((dist_sq, true));
    }

    let n_q = q_chunk.rows;
    let n_t = x_train.rows;
    let cols = q_chunk.cols;
    let mut out = vec![0.0; n_q * n_t];

    let fill_row = |(qi, out_row): (usize, &mut [f64])| {
        let q_row = &q_chunk.data[qi * cols..(qi + 1) * cols];
        for (ti, slot) in out_row.iter_mut().enumerate() {
            let t_row = &x_train.data[ti * cols..(ti + 1) * cols];
            *slot = row_distance(q_row, t_row, metric);
        }
    };

    if n_q * n_t < 16_384 {
        out.chunks_exact_mut(n_t).enumerate().for_each(fill_row);
    } else {
        out.par_chunks_mut(n_t).enumerate().for_each(fill_row);
    }

    let dist_mat = Matrix::from_vec(n_q, n_t, out)?;
    Ok((dist_mat, false))
}

fn chunked_kneighbors(
    x_query: &Matrix,
    x_train: &Matrix,
    k: usize,
    metric: Metric,
    device_type: &DeviceType,
    exclude_self: bool,
) -> OchreResult<(Matrix, Vec<usize>)> {
    let n_queries = x_query.rows;
    let n_train = x_train.rows;
    let max_k = if exclude_self {
        n_train.saturating_sub(1)
    } else {
        n_train
    };
    if k == 0 || k > max_k {
        return Err(OchreError::InvalidParameter(format!(
            "n_neighbors={} must be within 1..={} ( >_< )",
            k, max_k
        )));
    }

    let k_fetch = if exclude_self { k + 1 } else { k };
    let backend = get_backend(device_type);
    let mut all_dists = Vec::with_capacity(n_queries * k);
    let mut all_indices = Vec::with_capacity(n_queries * k);

    let mut q_start = 0;
    while q_start < n_queries {
        let q_end = (q_start + QUERY_BATCH_SIZE).min(n_queries);
        let q_chunk = slice_rows_matrix(x_query, q_start, q_end)?;
        let (dist_mat, is_sq) = compute_chunk_distances(&q_chunk, x_train, metric, device_type)?;
        let (topk_idx, topk_vals) = backend.topk_argmin(&dist_mat, k_fetch)?;

        let q_len = q_end - q_start;
        for local_r in 0..q_len {
            let global_r = q_start + local_r;
            let row_start = local_r * k_fetch;
            let mut taken = 0;
            for j in 0..k_fetch {
                let idx = topk_idx[row_start + j];
                if exclude_self && idx == global_r {
                    continue;
                }
                let raw_d = topk_vals.data[row_start + j];
                let d = if is_sq { raw_d.max(0.0).sqrt() } else { raw_d };
                all_dists.push(d);
                all_indices.push(idx);
                taken += 1;
                if taken == k {
                    break;
                }
            }
            // Fallback if ties pushed self past k_fetch (unlikely since self dist == 0)
            while taken < k {
                all_dists.push(0.0);
                all_indices.push(0);
                taken += 1;
            }
        }
        q_start = q_end;
    }

    let dist_matrix = Matrix::from_vec(n_queries, k, all_dists)?;
    Ok((dist_matrix, all_indices))
}

fn chunked_radius_neighbors(
    x_query: &Matrix,
    x_train: &Matrix,
    radius: f64,
    metric: Metric,
    device_type: &DeviceType,
    exclude_self: bool,
    sort_results: bool,
) -> OchreResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let n_queries = x_query.rows;
    let n_train = x_train.rows;
    let radius_sq = radius * radius;
    let mut offsets = Vec::with_capacity(n_queries + 1);
    offsets.push(0);
    let mut indices = Vec::new();
    let mut distances = Vec::new();

    let mut q_start = 0;
    while q_start < n_queries {
        let q_end = (q_start + QUERY_BATCH_SIZE).min(n_queries);
        let q_chunk = slice_rows_matrix(x_query, q_start, q_end)?;
        let (dist_mat, is_sq) = compute_chunk_distances(&q_chunk, x_train, metric, device_type)?;

        let q_len = q_end - q_start;
        for local_r in 0..q_len {
            let global_r = q_start + local_r;
            let row = &dist_mat.data[local_r * n_train..(local_r + 1) * n_train];
            let mut row_hits: Vec<(f64, usize)> = Vec::new();
            for (ti, &raw_d) in row.iter().enumerate() {
                if exclude_self && ti == global_r {
                    continue;
                }
                if is_sq {
                    if raw_d <= radius_sq {
                        let d = raw_d.max(0.0).sqrt();
                        if d <= radius {
                            row_hits.push((d, ti));
                        }
                    }
                } else if raw_d <= radius {
                    row_hits.push((raw_d, ti));
                }
            }
            if sort_results {
                row_hits.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            }
            for (d, ti) in row_hits {
                distances.push(d);
                indices.push(ti);
            }
            offsets.push(indices.len());
        }
        q_start = q_end;
    }

    Ok((offsets, indices, distances))
}

fn compute_neighbor_weights(dists: &[f64], weights: Weights) -> Vec<f64> {
    let n = dists.len();
    if n == 0 {
        return Vec::new();
    }
    match weights {
        Weights::Uniform => vec![1.0; n],
        Weights::Distance => {
            let zero_count = dists.iter().filter(|&&d| d == 0.0).count();
            if zero_count > 0 {
                dists
                    .iter()
                    .map(|&d| if d == 0.0 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                dists.iter().map(|&d| 1.0 / d).collect()
            }
        }
    }
}

fn extract_classes_and_indices(y: &[f64]) -> (Vec<f64>, Vec<usize>, usize) {
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.total_cmp(b));
    classes.dedup();

    let mut counts = vec![0usize; classes.len()];
    let mut y_idx = Vec::with_capacity(y.len());
    for &val in y {
        let idx = classes.binary_search_by(|c| c.total_cmp(&val)).unwrap_or(0);
        y_idx.push(idx);
        counts[idx] += 1;
    }

    let mut best_cls_idx = 0;
    let mut best_count = 0;
    for (idx, &cnt) in counts.iter().enumerate() {
        if cnt > best_count {
            best_count = cnt;
            best_cls_idx = idx;
        }
    }

    (classes, y_idx, best_cls_idx)
}

fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> OchreResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_pred.len(),
            targets: y_true.len(),
        });
    }
    if y_true.is_empty() {
        return Err(OchreError::EmptyData);
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(a, b)| (a - b).abs() < 1e-12)
        .count();
    Ok(correct as f64 / y_true.len() as f64)
}

// ============================================================================
// 1. KNeighborsClassifier
// ============================================================================

#[derive(Debug, Clone)]
pub struct KNeighborsClassifier {
    pub n_neighbors: usize,
    pub weights: String,
    pub metric: String,
    pub p: f64,
    pub device_: Option<String>,
    device_type: DeviceType,
    x_train: Option<Matrix>,
    y_train: Vec<usize>,
    pub classes_: Vec<f64>,
    pub n_features_in_: Option<usize>,
    pub n_samples_fit_: Option<usize>,
}

impl KNeighborsClassifier {
    pub fn new(
        n_neighbors: usize,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_neighbors == 0 {
            return Err(OchreError::InvalidParameter(
                "n_neighbors must be >= 1 ( >_< )".to_string(),
            ));
        }
        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let device_type = parse_device(device.as_deref())?;
        let device_ = Some(device_type.to_string());
        Ok(Self {
            n_neighbors,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train: None,
            y_train: Vec::new(),
            classes_: Vec::new(),
            n_features_in_: None,
            n_samples_fit_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        if self.n_neighbors > x.rows {
            return Err(OchreError::InvalidParameter(format!(
                "n_neighbors ({}) cannot exceed n_samples ({}) ( >_< )",
                self.n_neighbors, x.rows
            )));
        }
        let (classes, y_idx, _) = extract_classes_and_indices(y);
        self.x_train = Some(x.clone());
        self.y_train = y_idx;
        self.classes_ = classes;
        self.n_features_in_ = Some(x.cols);
        self.n_samples_fit_ = Some(x.rows);
        Ok(())
    }

    fn check_fitted_and_x(&self, x: &Matrix) -> OchreResult<(&Matrix, usize)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        Ok((x_train, n_features))
    }

    pub fn kneighbors(
        &self,
        x: Option<&Matrix>,
        n_neighbors: Option<usize>,
    ) -> OchreResult<(Matrix, Vec<usize>)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let metric = Metric::parse(&self.metric, self.p)?;
        match x {
            Some(q) => {
                let (_, _) = self.check_fitted_and_x(q)?;
                chunked_kneighbors(q, x_train, k, metric, &self.device_type, false)
            }
            None => chunked_kneighbors(x_train, x_train, k, metric, &self.device_type, true),
        }
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let (x_train, _) = self.check_fitted_and_x(x)?;
        let metric = Metric::parse(&self.metric, self.p)?;
        let weights_mode = Weights::parse(&self.weights)?;
        let n_classes = self.classes_.len();
        let k = self.n_neighbors;

        let (dists, indices) = chunked_kneighbors(x, x_train, k, metric, &self.device_type, false)?;
        let mut probs = vec![0.0; x.rows * n_classes];

        for i in 0..x.rows {
            let d_row = &dists.data[i * k..(i + 1) * k];
            let idx_row = &indices[i * k..(i + 1) * k];
            let w_row = compute_neighbor_weights(d_row, weights_mode);
            let mut w_sum = 0.0;
            for (&ti, &w) in idx_row.iter().zip(w_row.iter()) {
                let cls_idx = self.y_train[ti];
                probs[i * n_classes + cls_idx] += w;
                w_sum += w;
            }
            if w_sum > 0.0 {
                for c in 0..n_classes {
                    probs[i * n_classes + c] /= w_sum;
                }
            }
        }

        Matrix::from_vec(x.rows, n_classes, probs)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let probs = self.predict_proba(x)?;
        let n_classes = self.classes_.len();
        let mut preds = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let row = &probs.data[i * n_classes..(i + 1) * n_classes];
            let mut best_c = 0;
            let mut best_p = f64::NEG_INFINITY;
            for (c, &p) in row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            preds.push(self.classes_[best_c]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KNEIGHBORS_CLASSIFIER);
        write_usize(&mut buf, self.n_neighbors);
        write_str(&mut buf, &self.weights);
        write_str(&mut buf, &self.metric);
        write_f64(&mut buf, self.p);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_train);
        write_usize_vec(&mut buf, &self.y_train);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_KNEIGHBORS_CLASSIFIER)?;
        let n_neighbors = reader.read_usize()?;
        let weights = reader.read_str()?;
        let metric = reader.read_str()?;
        let p = reader.read_f64()?;
        let saved_device = reader.read_opt_str()?;
        let x_train = reader.read_opt_matrix()?;
        let y_train = reader.read_usize_vec()?;
        let classes_ = reader.read_f64_vec()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let n_samples_fit_ = reader.read_opt_usize()?;

        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let is_fitted = x_train.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            n_neighbors,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train,
            y_train,
            classes_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// 2. KNeighborsRegressor
// ============================================================================

#[derive(Debug, Clone)]
pub struct KNeighborsRegressor {
    pub n_neighbors: usize,
    pub weights: String,
    pub metric: String,
    pub p: f64,
    pub device_: Option<String>,
    device_type: DeviceType,
    x_train: Option<Matrix>,
    y_train: Option<Matrix>,
    pub is_multi_output_: bool,
    pub n_features_in_: Option<usize>,
    pub n_samples_fit_: Option<usize>,
}

impl KNeighborsRegressor {
    pub fn new(
        n_neighbors: usize,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_neighbors == 0 {
            return Err(OchreError::InvalidParameter(
                "n_neighbors must be >= 1 ( >_< )".to_string(),
            ));
        }
        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let device_type = parse_device(device.as_deref())?;
        let device_ = Some(device_type.to_string());
        Ok(Self {
            n_neighbors,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train: None,
            y_train: None,
            is_multi_output_: false,
            n_features_in_: None,
            n_samples_fit_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_slice_finite(y, "y")?;
        let y_mat = Matrix::from_slice(y.len(), 1, y)?;
        self.fit_internal(x, &y_mat, false)
    }

    pub fn fit_multi(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_matrix_finite(y, "y")?;
        self.fit_internal(x, y, true)
    }

    fn fit_internal(&mut self, x: &Matrix, y: &Matrix, is_multi: bool) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        if self.n_neighbors > x.rows {
            return Err(OchreError::InvalidParameter(format!(
                "n_neighbors ({}) cannot exceed n_samples ({}) ( >_< )",
                self.n_neighbors, x.rows
            )));
        }
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.is_multi_output_ = is_multi;
        self.n_features_in_ = Some(x.cols);
        self.n_samples_fit_ = Some(x.rows);
        Ok(())
    }

    fn check_fitted_and_x(&self, x: &Matrix) -> OchreResult<(&Matrix, &Matrix)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let y_train = self.y_train.as_ref().ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        Ok((x_train, y_train))
    }

    pub fn kneighbors(
        &self,
        x: Option<&Matrix>,
        n_neighbors: Option<usize>,
    ) -> OchreResult<(Matrix, Vec<usize>)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let metric = Metric::parse(&self.metric, self.p)?;
        match x {
            Some(q) => {
                let _ = self.check_fitted_and_x(q)?;
                chunked_kneighbors(q, x_train, k, metric, &self.device_type, false)
            }
            None => chunked_kneighbors(x_train, x_train, k, metric, &self.device_type, true),
        }
    }

    pub fn predict_mat(&self, x: &Matrix) -> OchreResult<Matrix> {
        let (x_train, y_train) = self.check_fitted_and_x(x)?;
        let metric = Metric::parse(&self.metric, self.p)?;
        let weights_mode = Weights::parse(&self.weights)?;
        let k = self.n_neighbors;
        let n_outputs = y_train.cols;

        let (dists, indices) = chunked_kneighbors(x, x_train, k, metric, &self.device_type, false)?;
        let mut out = vec![0.0; x.rows * n_outputs];

        for i in 0..x.rows {
            let d_row = &dists.data[i * k..(i + 1) * k];
            let idx_row = &indices[i * k..(i + 1) * k];
            let w_row = compute_neighbor_weights(d_row, weights_mode);
            let w_sum: f64 = w_row.iter().sum();
            for (&ti, &w) in idx_row.iter().zip(w_row.iter()) {
                for o in 0..n_outputs {
                    out[i * n_outputs + o] += w * y_train.data[ti * n_outputs + o];
                }
            }
            if w_sum > 0.0 {
                for o in 0..n_outputs {
                    out[i * n_outputs + o] /= w_sum;
                }
            }
        }

        Matrix::from_vec(x.rows, n_outputs, out)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict_mat(x)?;
        Ok(r2_score(y, &preds.data))
    }

    pub fn score_multi(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        validate_matrix_finite(y, "y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let preds = self.predict_mat(x)?;
        Ok(r2_score_multi(y, &preds, None))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KNEIGHBORS_REGRESSOR);
        write_usize(&mut buf, self.n_neighbors);
        write_str(&mut buf, &self.weights);
        write_str(&mut buf, &self.metric);
        write_f64(&mut buf, self.p);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_train);
        write_opt_matrix(&mut buf, &self.y_train);
        write_bool(&mut buf, self.is_multi_output_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_KNEIGHBORS_REGRESSOR)?;
        let n_neighbors = reader.read_usize()?;
        let weights = reader.read_str()?;
        let metric = reader.read_str()?;
        let p = reader.read_f64()?;
        let saved_device = reader.read_opt_str()?;
        let x_train = reader.read_opt_matrix()?;
        let y_train = reader.read_opt_matrix()?;
        let is_multi_output_ = reader.read_bool()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let n_samples_fit_ = reader.read_opt_usize()?;

        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let is_fitted = x_train.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            n_neighbors,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train,
            y_train,
            is_multi_output_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// 3. RadiusNeighborsClassifier
// ============================================================================

#[derive(Debug, Clone)]
pub struct RadiusNeighborsClassifier {
    pub radius: f64,
    pub weights: String,
    pub metric: String,
    pub p: f64,
    pub outlier_label: Option<f64>,
    pub outlier_most_frequent: bool,
    pub device_: Option<String>,
    device_type: DeviceType,
    x_train: Option<Matrix>,
    y_train: Vec<usize>,
    pub classes_: Vec<f64>,
    most_frequent_class_idx: usize,
    pub outlier_label_: Option<f64>,
    pub n_features_in_: Option<usize>,
    pub n_samples_fit_: Option<usize>,
}

impl RadiusNeighborsClassifier {
    pub fn new(
        radius: f64,
        weights: String,
        metric: String,
        p: f64,
        outlier_label: Option<f64>,
        outlier_most_frequent: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "radius ({}) must be a strictly positive finite number ( >_< )",
                radius
            )));
        }
        if let Some(lbl) = outlier_label
            && !lbl.is_finite()
        {
            return Err(OchreError::InvalidParameter(
                "outlier_label must be finite ( >_< )".to_string(),
            ));
        }
        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let device_type = parse_device(device.as_deref())?;
        let device_ = Some(device_type.to_string());
        Ok(Self {
            radius,
            weights,
            metric,
            p,
            outlier_label,
            outlier_most_frequent,
            device_,
            device_type,
            x_train: None,
            y_train: Vec::new(),
            classes_: Vec::new(),
            most_frequent_class_idx: 0,
            outlier_label_: None,
            n_features_in_: None,
            n_samples_fit_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let (classes, y_idx, most_freq_idx) = extract_classes_and_indices(y);
        self.x_train = Some(x.clone());
        self.y_train = y_idx;
        self.classes_ = classes;
        self.most_frequent_class_idx = most_freq_idx;
        self.outlier_label_ = if self.outlier_most_frequent {
            Some(self.classes_[most_freq_idx])
        } else {
            self.outlier_label
        };
        self.n_features_in_ = Some(x.cols);
        self.n_samples_fit_ = Some(x.rows);
        Ok(())
    }

    fn check_fitted_and_x(&self, x: &Matrix) -> OchreResult<&Matrix> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        Ok(x_train)
    }

    pub fn radius_neighbors(
        &self,
        x: Option<&Matrix>,
        radius: Option<f64>,
        sort_results: bool,
    ) -> OchreResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let r = radius.unwrap_or(self.radius);
        if !r.is_finite() || r <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "radius ({}) must be a strictly positive finite number ( >_< )",
                r
            )));
        }
        let metric = Metric::parse(&self.metric, self.p)?;
        match x {
            Some(q) => {
                let _ = self.check_fitted_and_x(q)?;
                chunked_radius_neighbors(
                    q,
                    x_train,
                    r,
                    metric,
                    &self.device_type,
                    false,
                    sort_results,
                )
            }
            None => chunked_radius_neighbors(
                x_train,
                x_train,
                r,
                metric,
                &self.device_type,
                true,
                sort_results,
            ),
        }
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let x_train = self.check_fitted_and_x(x)?;
        let metric = Metric::parse(&self.metric, self.p)?;
        let weights_mode = Weights::parse(&self.weights)?;
        let n_classes = self.classes_.len();

        let (offsets, indices, dists) = chunked_radius_neighbors(
            x,
            x_train,
            self.radius,
            metric,
            &self.device_type,
            false,
            false,
        )?;
        let mut probs = vec![0.0; x.rows * n_classes];

        for i in 0..x.rows {
            let start = offsets[i];
            let end = offsets[i + 1];
            if start == end {
                if self.outlier_most_frequent {
                    probs[i * n_classes + self.most_frequent_class_idx] = 1.0;
                } else if let Some(lbl) = self.outlier_label {
                    if let Ok(cls_idx) = self.classes_.binary_search_by(|c| c.total_cmp(&lbl)) {
                        probs[i * n_classes + cls_idx] = 1.0;
                    }
                } else {
                    return Err(OchreError::InvalidParameter(format!(
                        "No neighbors found for test sample {} within radius {} and outlier_label is None ( >_< )",
                        i, self.radius
                    )));
                }
                continue;
            }

            let d_row = &dists[start..end];
            let idx_row = &indices[start..end];
            let w_row = compute_neighbor_weights(d_row, weights_mode);
            let mut w_sum = 0.0;
            for (&ti, &w) in idx_row.iter().zip(w_row.iter()) {
                let cls_idx = self.y_train[ti];
                probs[i * n_classes + cls_idx] += w;
                w_sum += w;
            }
            if w_sum > 0.0 {
                for c in 0..n_classes {
                    probs[i * n_classes + c] /= w_sum;
                }
            }
        }

        Matrix::from_vec(x.rows, n_classes, probs)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let probs = self.predict_proba(x)?;
        let n_classes = self.classes_.len();
        let mut preds = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let row = &probs.data[i * n_classes..(i + 1) * n_classes];
            let row_sum: f64 = row.iter().sum();
            if row_sum == 0.0
                && let Some(lbl) = self.outlier_label_
            {
                preds.push(lbl);
                continue;
            }
            let mut best_c = 0;
            let mut best_p = f64::NEG_INFINITY;
            for (c, &p) in row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            preds.push(self.classes_[best_c]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_RADIUS_NEIGHBORS_CLASSIFIER);
        write_f64(&mut buf, self.radius);
        write_str(&mut buf, &self.weights);
        write_str(&mut buf, &self.metric);
        write_f64(&mut buf, self.p);
        write_opt_f64(&mut buf, self.outlier_label);
        write_bool(&mut buf, self.outlier_most_frequent);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_train);
        write_usize_vec(&mut buf, &self.y_train);
        write_f64_vec(&mut buf, &self.classes_);
        write_usize(&mut buf, self.most_frequent_class_idx);
        write_opt_f64(&mut buf, self.outlier_label_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_RADIUS_NEIGHBORS_CLASSIFIER)?;
        let radius = reader.read_f64()?;
        let weights = reader.read_str()?;
        let metric = reader.read_str()?;
        let p = reader.read_f64()?;
        let outlier_label = reader.read_opt_f64()?;
        let outlier_most_frequent = reader.read_bool()?;
        let saved_device = reader.read_opt_str()?;
        let x_train = reader.read_opt_matrix()?;
        let y_train = reader.read_usize_vec()?;
        let classes_ = reader.read_f64_vec()?;
        let most_frequent_class_idx = reader.read_usize()?;
        let outlier_label_ = reader.read_opt_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let n_samples_fit_ = reader.read_opt_usize()?;

        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let is_fitted = x_train.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            radius,
            weights,
            metric,
            p,
            outlier_label,
            outlier_most_frequent,
            device_,
            device_type,
            x_train,
            y_train,
            classes_,
            most_frequent_class_idx,
            outlier_label_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// 4. RadiusNeighborsRegressor
// ============================================================================

#[derive(Debug, Clone)]
pub struct RadiusNeighborsRegressor {
    pub radius: f64,
    pub weights: String,
    pub metric: String,
    pub p: f64,
    pub device_: Option<String>,
    device_type: DeviceType,
    x_train: Option<Matrix>,
    y_train: Option<Matrix>,
    pub is_multi_output_: bool,
    pub n_features_in_: Option<usize>,
    pub n_samples_fit_: Option<usize>,
}

impl RadiusNeighborsRegressor {
    pub fn new(
        radius: f64,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "radius ({}) must be a strictly positive finite number ( >_< )",
                radius
            )));
        }
        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let device_type = parse_device(device.as_deref())?;
        let device_ = Some(device_type.to_string());
        Ok(Self {
            radius,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train: None,
            y_train: None,
            is_multi_output_: false,
            n_features_in_: None,
            n_samples_fit_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_slice_finite(y, "y")?;
        let y_mat = Matrix::from_slice(y.len(), 1, y)?;
        self.fit_internal(x, &y_mat, false)
    }

    pub fn fit_multi(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_matrix_finite(y, "y")?;
        self.fit_internal(x, y, true)
    }

    fn fit_internal(&mut self, x: &Matrix, y: &Matrix, is_multi: bool) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.is_multi_output_ = is_multi;
        self.n_features_in_ = Some(x.cols);
        self.n_samples_fit_ = Some(x.rows);
        Ok(())
    }

    fn check_fitted_and_x(&self, x: &Matrix) -> OchreResult<(&Matrix, &Matrix)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let y_train = self.y_train.as_ref().ok_or(OchreError::NotFitted)?;
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        Ok((x_train, y_train))
    }

    pub fn radius_neighbors(
        &self,
        x: Option<&Matrix>,
        radius: Option<f64>,
        sort_results: bool,
    ) -> OchreResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
        let x_train = self.x_train.as_ref().ok_or(OchreError::NotFitted)?;
        let r = radius.unwrap_or(self.radius);
        if !r.is_finite() || r <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "radius ({}) must be a strictly positive finite number ( >_< )",
                r
            )));
        }
        let metric = Metric::parse(&self.metric, self.p)?;
        match x {
            Some(q) => {
                let _ = self.check_fitted_and_x(q)?;
                chunked_radius_neighbors(
                    q,
                    x_train,
                    r,
                    metric,
                    &self.device_type,
                    false,
                    sort_results,
                )
            }
            None => chunked_radius_neighbors(
                x_train,
                x_train,
                r,
                metric,
                &self.device_type,
                true,
                sort_results,
            ),
        }
    }

    pub fn predict_mat(&self, x: &Matrix) -> OchreResult<Matrix> {
        let (x_train, y_train) = self.check_fitted_and_x(x)?;
        let metric = Metric::parse(&self.metric, self.p)?;
        let weights_mode = Weights::parse(&self.weights)?;
        let n_outputs = y_train.cols;

        let (offsets, indices, dists) = chunked_radius_neighbors(
            x,
            x_train,
            self.radius,
            metric,
            &self.device_type,
            false,
            false,
        )?;
        let mut out = vec![0.0; x.rows * n_outputs];

        for i in 0..x.rows {
            let start = offsets[i];
            let end = offsets[i + 1];
            if start == end {
                for o in 0..n_outputs {
                    out[i * n_outputs + o] = f64::NAN;
                }
                continue;
            }
            let d_row = &dists[start..end];
            let idx_row = &indices[start..end];
            let w_row = compute_neighbor_weights(d_row, weights_mode);
            let w_sum: f64 = w_row.iter().sum();
            for (&ti, &w) in idx_row.iter().zip(w_row.iter()) {
                for o in 0..n_outputs {
                    out[i * n_outputs + o] += w * y_train.data[ti * n_outputs + o];
                }
            }
            if w_sum > 0.0 {
                for o in 0..n_outputs {
                    out[i * n_outputs + o] /= w_sum;
                }
            }
        }

        Matrix::from_vec(x.rows, n_outputs, out)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict_mat(x)?;
        Ok(r2_score(y, &preds.data))
    }

    pub fn score_multi(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        validate_matrix_finite(y, "y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let preds = self.predict_mat(x)?;
        Ok(r2_score_multi(y, &preds, None))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_RADIUS_NEIGHBORS_REGRESSOR);
        write_f64(&mut buf, self.radius);
        write_str(&mut buf, &self.weights);
        write_str(&mut buf, &self.metric);
        write_f64(&mut buf, self.p);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_train);
        write_opt_matrix(&mut buf, &self.y_train);
        write_bool(&mut buf, self.is_multi_output_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_samples_fit_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_RADIUS_NEIGHBORS_REGRESSOR)?;
        let radius = reader.read_f64()?;
        let weights = reader.read_str()?;
        let metric = reader.read_str()?;
        let p = reader.read_f64()?;
        let saved_device = reader.read_opt_str()?;
        let x_train = reader.read_opt_matrix()?;
        let y_train = reader.read_opt_matrix()?;
        let is_multi_output_ = reader.read_bool()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let n_samples_fit_ = reader.read_opt_usize()?;

        let _ = Weights::parse(&weights)?;
        let _ = Metric::parse(&metric, p)?;
        let is_fitted = x_train.is_some();
        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            radius,
            weights,
            metric,
            p,
            device_,
            device_type,
            x_train,
            y_train,
            is_multi_output_,
            n_features_in_,
            n_samples_fit_,
        })
    }
}

// ============================================================================
// PyO3 Bindings
// ============================================================================

type PyKNeighborsTuple<'py> = (usize, usize, Bound<'py, PyBytes>, Bound<'py, PyBytes>);
type PyRadiusNeighborsTuple<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

#[pyclass(from_py_object, name = "KNeighborsClassifier")]
#[derive(Clone)]
pub struct PyKNeighborsClassifier {
    inner: KNeighborsClassifier,
}

crate::pymethods_with_dit!(PyKNeighborsClassifier, KNeighborsClassifier, {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 5,
        weights = "uniform".to_string(),
        metric = "minkowski".to_string(),
        p = 2.0,
        device = None
    ))]
    pub fn new(
        n_neighbors: usize,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KNeighborsClassifier::new(n_neighbors, weights, metric, p, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
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

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &probs)
    }

    #[pyo3(signature = (x = None, n_neighbors = None))]
    pub fn kneighbors_buffer<'py>(
        &self,
        py: Python<'py>,
        x: Option<PyBuffer<f64>>,
        n_neighbors: Option<usize>,
    ) -> PyResult<PyKNeighborsTuple<'py>> {
        let (dists, indices) = match x {
            Some(buf) => {
                let x_mat = buffer_to_pooled_matrix(py, &buf)?;
                self.inner.kneighbors(Some(&x_mat), n_neighbors)?
            }
            None => self.inner.kneighbors(None, n_neighbors)?,
        };
        let d_bytes = f64_slice_to_pybytes(py, &dists.data)?;
        let i_bytes = usize_slice_to_i64_pybytes(py, &indices)?;
        Ok((dists.rows, dists.cols, d_bytes, i_bytes))
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
    pub fn classes_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        if self.inner.classes_.is_empty() {
            return Ok(None);
        }
        Ok(Some(f64_slice_to_pybytes(py, &self.inner.classes_)?))
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_samples_fit_(&self) -> Option<usize> {
        self.inner.n_samples_fit_
    }
});

#[pyclass(from_py_object, name = "KNeighborsRegressor")]
#[derive(Clone)]
pub struct PyKNeighborsRegressor {
    inner: KNeighborsRegressor,
}

crate::pymethods_with_dit!(PyKNeighborsRegressor, KNeighborsRegressor, {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 5,
        weights = "uniform".to_string(),
        metric = "minkowski".to_string(),
        p = 2.0,
        device = None
    ))]
    pub fn new(
        n_neighbors: usize,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KNeighborsRegressor::new(n_neighbors, weights, metric, p, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
        Ok(())
    }

    pub fn fit_multi_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        self.inner.fit_multi(&x_mat, &y_mat)?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_mat(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x = None, n_neighbors = None))]
    pub fn kneighbors_buffer<'py>(
        &self,
        py: Python<'py>,
        x: Option<PyBuffer<f64>>,
        n_neighbors: Option<usize>,
    ) -> PyResult<PyKNeighborsTuple<'py>> {
        let (dists, indices) = match x {
            Some(buf) => {
                let x_mat = buffer_to_pooled_matrix(py, &buf)?;
                self.inner.kneighbors(Some(&x_mat), n_neighbors)?
            }
            None => self.inner.kneighbors(None, n_neighbors)?,
        };
        let d_bytes = f64_slice_to_pybytes(py, &dists.data)?;
        let i_bytes = usize_slice_to_i64_pybytes(py, &indices)?;
        Ok((dists.rows, dists.cols, d_bytes, i_bytes))
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

    pub fn score_multi_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        Ok(self.inner.score_multi(&x_mat, &y_mat)?)
    }

    #[getter]
    pub fn is_multi_output_(&self) -> bool {
        self.inner.is_multi_output_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_samples_fit_(&self) -> Option<usize> {
        self.inner.n_samples_fit_
    }
});

#[pyclass(from_py_object, name = "RadiusNeighborsClassifier")]
#[derive(Clone)]
pub struct PyRadiusNeighborsClassifier {
    inner: RadiusNeighborsClassifier,
}

crate::pymethods_with_dit!(PyRadiusNeighborsClassifier, RadiusNeighborsClassifier, {
    #[new]
    #[pyo3(signature = (
        radius = 1.0,
        weights = "uniform".to_string(),
        metric = "minkowski".to_string(),
        p = 2.0,
        outlier_label = None,
        outlier_most_frequent = false,
        device = None
    ))]
    pub fn new(
        radius: f64,
        weights: String,
        metric: String,
        p: f64,
        outlier_label: Option<f64>,
        outlier_most_frequent: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RadiusNeighborsClassifier::new(
                radius,
                weights,
                metric,
                p,
                outlier_label,
                outlier_most_frequent,
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
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
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

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &probs)
    }

    #[pyo3(signature = (x = None, radius = None, sort_results = false))]
    pub fn radius_neighbors_buffer<'py>(
        &self,
        py: Python<'py>,
        x: Option<PyBuffer<f64>>,
        radius: Option<f64>,
        sort_results: bool,
    ) -> PyResult<PyRadiusNeighborsTuple<'py>> {
        let (offsets, indices, dists) = match x {
            Some(buf) => {
                let x_mat = buffer_to_pooled_matrix(py, &buf)?;
                self.inner
                    .radius_neighbors(Some(&x_mat), radius, sort_results)?
            }
            None => self.inner.radius_neighbors(None, radius, sort_results)?,
        };
        let off_bytes = usize_slice_to_i64_pybytes(py, &offsets)?;
        let idx_bytes = usize_slice_to_i64_pybytes(py, &indices)?;
        let dist_bytes = f64_slice_to_pybytes(py, &dists)?;
        Ok((off_bytes, idx_bytes, dist_bytes))
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
    pub fn classes_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        if self.inner.classes_.is_empty() {
            return Ok(None);
        }
        Ok(Some(f64_slice_to_pybytes(py, &self.inner.classes_)?))
    }

    #[getter]
    pub fn outlier_label_(&self) -> Option<f64> {
        self.inner.outlier_label_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_samples_fit_(&self) -> Option<usize> {
        self.inner.n_samples_fit_
    }
});

#[pyclass(from_py_object, name = "RadiusNeighborsRegressor")]
#[derive(Clone)]
pub struct PyRadiusNeighborsRegressor {
    inner: RadiusNeighborsRegressor,
}

crate::pymethods_with_dit!(PyRadiusNeighborsRegressor, RadiusNeighborsRegressor, {
    #[new]
    #[pyo3(signature = (
        radius = 1.0,
        weights = "uniform".to_string(),
        metric = "minkowski".to_string(),
        p = 2.0,
        device = None
    ))]
    pub fn new(
        radius: f64,
        weights: String,
        metric: String,
        p: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RadiusNeighborsRegressor::new(radius, weights, metric, p, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&x_mat, &y_vec)?;
        Ok(())
    }

    pub fn fit_multi_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        self.inner.fit_multi(&x_mat, &y_mat)?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_mat(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (x = None, radius = None, sort_results = false))]
    pub fn radius_neighbors_buffer<'py>(
        &self,
        py: Python<'py>,
        x: Option<PyBuffer<f64>>,
        radius: Option<f64>,
        sort_results: bool,
    ) -> PyResult<PyRadiusNeighborsTuple<'py>> {
        let (offsets, indices, dists) = match x {
            Some(buf) => {
                let x_mat = buffer_to_pooled_matrix(py, &buf)?;
                self.inner
                    .radius_neighbors(Some(&x_mat), radius, sort_results)?
            }
            None => self.inner.radius_neighbors(None, radius, sort_results)?,
        };
        let off_bytes = usize_slice_to_i64_pybytes(py, &offsets)?;
        let idx_bytes = usize_slice_to_i64_pybytes(py, &indices)?;
        let dist_bytes = f64_slice_to_pybytes(py, &dists)?;
        Ok((off_bytes, idx_bytes, dist_bytes))
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

    pub fn score_multi_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        Ok(self.inner.score_multi(&x_mat, &y_mat)?)
    }

    #[getter]
    pub fn is_multi_output_(&self) -> bool {
        self.inner.is_multi_output_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_samples_fit_(&self) -> Option<usize> {
        self.inner.n_samples_fit_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKNeighborsClassifier>()?;
    m.add_class::<PyKNeighborsRegressor>()?;
    m.add_class::<PyRadiusNeighborsClassifier>()?;
    m.add_class::<PyRadiusNeighborsRegressor>()?;
    Ok(())
}
