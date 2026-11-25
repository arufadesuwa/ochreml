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
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub const TAG_KMEANS: u8 = 47;
pub const TAG_MINI_BATCH_KMEANS: u8 = 48;

const MAX_DIST_ELEMENTS: usize = 4_000_000;
const FUSED_CPU_OPS_THRESHOLD: usize = 8_000_000;
const FUSED_CPU_DIST_THRESHOLD: usize = 8_000_000;
const PAR_ROW_THRESHOLD: usize = 4_096;
const ASSIGN_PAR_ROW_THRESHOLD: usize = 1_024;
const PREDICT_CHUNK_ROWS: usize = 4_096;

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

struct VarAcc {
    w_sum: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
    tmp_mean: Vec<f64>,
    tmp_m2: Vec<f64>,
}

impl VarAcc {
    fn new(p: usize) -> Self {
        Self {
            w_sum: 0.0,
            mean: vec![0.0f64; p],
            m2: vec![0.0f64; p],
            tmp_mean: vec![0.0f64; p],
            tmp_m2: vec![0.0f64; p],
        }
    }

    /// Accumulate an L2-resident chunk into `self` via single-DRAM-pass Chan variance merge.
    fn add_chunk(&mut self, x_chunk: &[f64], w_chunk: Option<&[f64]>, p: usize) {
        let rows = x_chunk.len() / p;
        if rows == 0 {
            return;
        }
        self.tmp_mean.fill(0.0);
        self.tmp_m2.fill(0.0);

        let wb = match w_chunk {
            Some(wc) => {
                let mut ws = 0.0;
                for (i, &w) in wc.iter().enumerate().take(rows) {
                    ws += w;
                    let row = &x_chunk[i * p..(i + 1) * p];
                    for (m, &xj) in self.tmp_mean.iter_mut().zip(row.iter()) {
                        *m += w * xj;
                    }
                }
                ws
            }
            None => {
                for i in 0..rows {
                    let row = &x_chunk[i * p..(i + 1) * p];
                    for (m, &xj) in self.tmp_mean.iter_mut().zip(row.iter()) {
                        *m += xj;
                    }
                }
                rows as f64
            }
        };
        if wb <= 0.0 {
            return;
        }
        let inv_wb = 1.0 / wb;
        for m in &mut self.tmp_mean {
            *m *= inv_wb;
        }
        match w_chunk {
            Some(wc) => {
                for (i, &w) in wc.iter().enumerate().take(rows) {
                    let row = &x_chunk[i * p..(i + 1) * p];
                    for ((m2, &xj), &mj) in self
                        .tmp_m2
                        .iter_mut()
                        .zip(row.iter())
                        .zip(self.tmp_mean.iter())
                    {
                        let d = xj - mj;
                        *m2 += w * d * d;
                    }
                }
            }
            None => {
                for i in 0..rows {
                    let row = &x_chunk[i * p..(i + 1) * p];
                    for ((m2, &xj), &mj) in self
                        .tmp_m2
                        .iter_mut()
                        .zip(row.iter())
                        .zip(self.tmp_mean.iter())
                    {
                        let d = xj - mj;
                        *m2 += d * d;
                    }
                }
            }
        }

        if self.w_sum <= 0.0 {
            self.w_sum = wb;
            self.mean.copy_from_slice(&self.tmp_mean);
            self.m2.copy_from_slice(&self.tmp_m2);
            return;
        }
        let w_ab = self.w_sum + wb;
        let w_ratio = wb / w_ab;
        let w_prod = (self.w_sum * wb) / w_ab;
        for j in 0..p {
            let delta = self.tmp_mean[j] - self.mean[j];
            self.mean[j] += delta * w_ratio;
            self.m2[j] += self.tmp_m2[j] + delta * delta * w_prod;
        }
        self.w_sum = w_ab;
    }

    fn merge(mut self, other: Self) -> Self {
        if other.w_sum <= 0.0 {
            return self;
        }
        if self.w_sum <= 0.0 {
            return other;
        }
        let w_ab = self.w_sum + other.w_sum;
        let w_ratio = other.w_sum / w_ab;
        let w_prod = (self.w_sum * other.w_sum) / w_ab;
        for j in 0..self.mean.len() {
            let delta = other.mean[j] - self.mean[j];
            self.mean[j] += delta * w_ratio;
            self.m2[j] += other.m2[j] + delta * delta * w_prod;
        }
        self.w_sum = w_ab;
        self
    }
}

fn compute_mean_variance(x: &Matrix, sample_weight: Option<&[f64]>) -> f64 {
    let n = x.rows;
    let p = x.cols;
    if n == 0 || p == 0 {
        return 0.0;
    }

    let acc = if n >= PAR_ROW_THRESHOLD {
        const CHUNK_ROWS: usize = 2048;
        match sample_weight {
            Some(sw) => x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .zip(sw.par_chunks(CHUNK_ROWS))
                .fold(
                    || VarAcc::new(p),
                    |mut acc, (x_chunk, w_chunk)| {
                        acc.add_chunk(x_chunk, Some(w_chunk), p);
                        acc
                    },
                )
                .reduce(|| VarAcc::new(p), VarAcc::merge),
            None => x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .fold(
                    || VarAcc::new(p),
                    |mut acc, x_chunk| {
                        acc.add_chunk(x_chunk, None, p);
                        acc
                    },
                )
                .reduce(|| VarAcc::new(p), VarAcc::merge),
        }
    } else {
        let mut acc = VarAcc::new(p);
        acc.add_chunk(&x.data, sample_weight, p);
        acc
    };

    if acc.w_sum <= 0.0 {
        return 0.0;
    }
    let var_sum: f64 = acc.m2.iter().sum();
    var_sum / (acc.w_sum * p as f64)
}

#[inline(always)]
fn sq_dist_row(a: &[f64], b: &[f64]) -> f64 {
    let p = a.len();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut j = 0;
    while j + 4 <= p {
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        j += 4;
    }
    let mut dist = (s0 + s1) + (s2 + s3);
    while j < p {
        let d = a[j] - b[j];
        dist += d * d;
        j += 1;
    }
    dist
}

#[inline(always)]
fn nearest_center_row(row: &[f64], centers_data: &[f64], k: usize, p: usize) -> (usize, f64) {
    let mut best_k = 0usize;
    let mut best_dist = f64::INFINITY;
    let mut c = 0usize;

    while c + 4 <= k {
        let c0 = &centers_data[c * p..(c + 1) * p];
        let c1 = &centers_data[(c + 1) * p..(c + 2) * p];
        let c2 = &centers_data[(c + 2) * p..(c + 3) * p];
        let c3 = &centers_data[(c + 3) * p..(c + 4) * p];
        let mut d0 = 0.0;
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        let mut d3 = 0.0;
        for j in 0..p {
            let xj = row[j];
            let t0 = xj - c0[j];
            let t1 = xj - c1[j];
            let t2 = xj - c2[j];
            let t3 = xj - c3[j];
            d0 += t0 * t0;
            d1 += t1 * t1;
            d2 += t2 * t2;
            d3 += t3 * t3;
        }
        if d0 < best_dist {
            best_dist = d0;
            best_k = c;
        }
        if d1 < best_dist {
            best_dist = d1;
            best_k = c + 1;
        }
        if d2 < best_dist {
            best_dist = d2;
            best_k = c + 2;
        }
        if d3 < best_dist {
            best_dist = d3;
            best_k = c + 3;
        }
        c += 4;
    }

    while c < k {
        let c_row = &centers_data[c * p..(c + 1) * p];
        let d = sq_dist_row(row, c_row);
        if d < best_dist {
            best_dist = d;
            best_k = c;
        }
        c += 1;
    }

    (best_k, best_dist)
}

#[inline(always)]
fn should_use_fused_cpu(backend: &Backend<'_>, n: usize, k: usize, p: usize) -> bool {
    matches!(backend, Backend::Cpu)
        || n.saturating_mul(k).saturating_mul(p) < FUSED_CPU_OPS_THRESHOLD
        || n.saturating_mul(k) < MAX_DIST_ELEMENTS
}

#[inline(always)]
fn should_use_cpu_predict(backend: &Backend<'_>, n: usize, k: usize, p: usize) -> bool {
    matches!(backend, Backend::Cpu)
        || n.saturating_mul(k) < FUSED_CPU_DIST_THRESHOLD
        || should_use_fused_cpu(backend, n, k, p)
}

/// Zero-allocation fused nearest-center kernel filling pre-allocated label and squared-distance buffers.
fn cpu_assign_labels_fused_into(
    x: &Matrix,
    centers: &Matrix,
    labels: &mut [usize],
    sq_dists: &mut [f64],
) {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    let centers_data = &centers.data;

    if n < PAR_ROW_THRESHOLD {
        for i in 0..n {
            let row = &x.data[i * p..(i + 1) * p];
            let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
            labels[i] = best_k;
            sq_dists[i] = best_dist;
        }
    } else {
        const CHUNK_ROWS: usize = 1024;
        labels
            .par_chunks_mut(CHUNK_ROWS)
            .zip(sq_dists.par_chunks_mut(CHUNK_ROWS))
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .for_each(|((lbl_chunk, dist_chunk), x_chunk)| {
                let rows = lbl_chunk.len();
                for i in 0..rows {
                    let row = &x_chunk[i * p..(i + 1) * p];
                    let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
                    lbl_chunk[i] = best_k;
                    dist_chunk[i] = best_dist;
                }
            });
    }
}

/// Zero-allocation fused nearest-center kernel returning `(labels, sq_dists)`.
fn cpu_assign_labels_fused(x: &Matrix, centers: &Matrix) -> (Vec<usize>, Vec<f64>) {
    let mut labels = vec![0usize; x.rows];
    let mut sq_dists = vec![0.0f64; x.rows];
    cpu_assign_labels_fused_into(x, centers, &mut labels, &mut sq_dists);
    (labels, sq_dists)
}

/// Pre-packed centers for register-friendly SIMD distance-score evaluation (`0.5 * ||c||^2 - x.c`).
enum PackedCenters {
    K4 {
        c_t: Vec<[f64; 4]>,
        hn: [f64; 4],
        k: usize,
        p: usize,
    },
    K8 {
        c_t: Vec<[f64; 8]>,
        hn: [f64; 8],
        k: usize,
        p: usize,
    },
    General {
        half_norms: Vec<f64>,
        k: usize,
        p: usize,
    },
}

impl PackedCenters {
    fn new(centers_data: &[f64], k: usize, p: usize) -> Self {
        let mut half_norms = Vec::with_capacity(k);
        for c in 0..k {
            let cr = &centers_data[c * p..(c + 1) * p];
            let mut norm_sq = 0.0;
            for &v in cr {
                norm_sq += v * v;
            }
            half_norms.push(0.5 * norm_sq);
        }

        if k <= 4 && p <= 512 {
            let mut c_t = vec![[0.0f64; 4]; p];
            let mut hn = [f64::INFINITY; 4];
            for c in 0..k {
                hn[c] = half_norms[c];
                let cr = &centers_data[c * p..(c + 1) * p];
                for j in 0..p {
                    c_t[j][c] = cr[j];
                }
            }
            Self::K4 { c_t, hn, k, p }
        } else if k <= 8 && p <= 512 {
            let mut c_t = vec![[0.0f64; 8]; p];
            let mut hn = [f64::INFINITY; 8];
            for c in 0..k {
                hn[c] = half_norms[c];
                let cr = &centers_data[c * p..(c + 1) * p];
                for j in 0..p {
                    c_t[j][c] = cr[j];
                }
            }
            Self::K8 { c_t, hn, k, p }
        } else {
            Self::General { half_norms, k, p }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn assign_4rows_k4<const COMPUTE_NORM: bool>(
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
    c_t: &[[f64; 4]],
    hn: &[f64; 4],
    k: usize,
    p: usize,
) -> ([usize; 4], [f64; 4]) {
    let r0 = &r0[..p];
    let r1 = &r1[..p];
    let r2 = &r2[..p];
    let r3 = &r3[..p];
    let c_t = &c_t[..p];

    let mut d0 = [0.0f64; 4];
    let mut d1 = [0.0f64; 4];
    let mut d2 = [0.0f64; 4];
    let mut d3 = [0.0f64; 4];
    let mut nx = [0.0f64; 4];

    for j in 0..p {
        let ct = c_t[j];
        let x0 = r0[j];
        let x1 = r1[j];
        let x2 = r2[j];
        let x3 = r3[j];
        if COMPUTE_NORM {
            nx[0] += x0 * x0;
            nx[1] += x1 * x1;
            nx[2] += x2 * x2;
            nx[3] += x3 * x3;
        }
        for c in 0..4 {
            d0[c] += x0 * ct[c];
            d1[c] += x1 * ct[c];
            d2[c] += x2 * ct[c];
            d3[c] += x3 * ct[c];
        }
    }

    let mut best_k = [0usize; 4];
    let mut best_s = [hn[0] - d0[0], hn[0] - d1[0], hn[0] - d2[0], hn[0] - d3[0]];
    for c in 1..k {
        let hnc = hn[c];
        let s0 = hnc - d0[c];
        let s1 = hnc - d1[c];
        let s2 = hnc - d2[c];
        let s3 = hnc - d3[c];
        if s0 < best_s[0] {
            best_s[0] = s0;
            best_k[0] = c;
        }
        if s1 < best_s[1] {
            best_s[1] = s1;
            best_k[1] = c;
        }
        if s2 < best_s[2] {
            best_s[2] = s2;
            best_k[2] = c;
        }
        if s3 < best_s[3] {
            best_s[3] = s3;
            best_k[3] = c;
        }
    }

    let sq = if COMPUTE_NORM {
        [
            (nx[0] + 2.0 * best_s[0]).max(0.0),
            (nx[1] + 2.0 * best_s[1]).max(0.0),
            (nx[2] + 2.0 * best_s[2]).max(0.0),
            (nx[3] + 2.0 * best_s[3]).max(0.0),
        ]
    } else {
        [0.0; 4]
    };
    (best_k, sq)
}

#[inline(always)]
fn assign_1row_k4<const COMPUTE_NORM: bool>(
    row: &[f64],
    c_t: &[[f64; 4]],
    hn: &[f64; 4],
    k: usize,
    p: usize,
) -> (usize, f64) {
    let row = &row[..p];
    let c_t = &c_t[..p];
    let mut d = [0.0f64; 4];
    let mut nx = 0.0f64;
    for j in 0..p {
        let xj = row[j];
        let ct = c_t[j];
        if COMPUTE_NORM {
            nx += xj * xj;
        }
        for c in 0..4 {
            d[c] += xj * ct[c];
        }
    }
    let mut best_k = 0usize;
    let mut best_s = hn[0] - d[0];
    for c in 1..k {
        let s = hn[c] - d[c];
        if s < best_s {
            best_s = s;
            best_k = c;
        }
    }
    let sq = if COMPUTE_NORM {
        (nx + 2.0 * best_s).max(0.0)
    } else {
        0.0
    };
    (best_k, sq)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn assign_4rows_k8<const COMPUTE_NORM: bool>(
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
    c_t: &[[f64; 8]],
    hn: &[f64; 8],
    k: usize,
    p: usize,
) -> ([usize; 4], [f64; 4]) {
    let r0 = &r0[..p];
    let r1 = &r1[..p];
    let r2 = &r2[..p];
    let r3 = &r3[..p];
    let c_t = &c_t[..p];

    let mut d0 = [0.0f64; 8];
    let mut d1 = [0.0f64; 8];
    let mut d2 = [0.0f64; 8];
    let mut d3 = [0.0f64; 8];
    let mut nx = [0.0f64; 4];

    for j in 0..p {
        let ct = &c_t[j];
        let x0 = r0[j];
        let x1 = r1[j];
        let x2 = r2[j];
        let x3 = r3[j];
        if COMPUTE_NORM {
            nx[0] += x0 * x0;
            nx[1] += x1 * x1;
            nx[2] += x2 * x2;
            nx[3] += x3 * x3;
        }
        for c in 0..8 {
            d0[c] += x0 * ct[c];
            d1[c] += x1 * ct[c];
            d2[c] += x2 * ct[c];
            d3[c] += x3 * ct[c];
        }
    }

    let mut best_k = [0usize; 4];
    let mut best_s = [hn[0] - d0[0], hn[0] - d1[0], hn[0] - d2[0], hn[0] - d3[0]];
    for c in 1..k {
        let hnc = hn[c];
        let s0 = hnc - d0[c];
        let s1 = hnc - d1[c];
        let s2 = hnc - d2[c];
        let s3 = hnc - d3[c];
        if s0 < best_s[0] {
            best_s[0] = s0;
            best_k[0] = c;
        }
        if s1 < best_s[1] {
            best_s[1] = s1;
            best_k[1] = c;
        }
        if s2 < best_s[2] {
            best_s[2] = s2;
            best_k[2] = c;
        }
        if s3 < best_s[3] {
            best_s[3] = s3;
            best_k[3] = c;
        }
    }

    let sq = if COMPUTE_NORM {
        [
            (nx[0] + 2.0 * best_s[0]).max(0.0),
            (nx[1] + 2.0 * best_s[1]).max(0.0),
            (nx[2] + 2.0 * best_s[2]).max(0.0),
            (nx[3] + 2.0 * best_s[3]).max(0.0),
        ]
    } else {
        [0.0; 4]
    };
    (best_k, sq)
}

#[inline(always)]
fn assign_1row_k8<const COMPUTE_NORM: bool>(
    row: &[f64],
    c_t: &[[f64; 8]],
    hn: &[f64; 8],
    k: usize,
    p: usize,
) -> (usize, f64) {
    let row = &row[..p];
    let c_t = &c_t[..p];
    let mut d = [0.0f64; 8];
    let mut nx = 0.0f64;
    for j in 0..p {
        let xj = row[j];
        let ct = &c_t[j];
        if COMPUTE_NORM {
            nx += xj * xj;
        }
        for c in 0..8 {
            d[c] += xj * ct[c];
        }
    }
    let mut best_k = 0usize;
    let mut best_s = hn[0] - d[0];
    for c in 1..k {
        let s = hn[c] - d[c];
        if s < best_s {
            best_s = s;
            best_k = c;
        }
    }
    let sq = if COMPUTE_NORM {
        (nx + 2.0 * best_s).max(0.0)
    } else {
        0.0
    };
    (best_k, sq)
}

/// 4-row-unrolled Euclidean argmin kernel minimizing `0.5 * ||c_j||^2 - dot(x_i, c_j)`.
/// Processes 4 sample rows against 2 centers at a time with 4-wide AVX2-friendly inner loops.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn assign_4rows_general<const COMPUTE_NORM: bool>(
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) -> ([usize; 4], [f64; 4]) {
    let r0 = &r0[..p];
    let r1 = &r1[..p];
    let r2 = &r2[..p];
    let r3 = &r3[..p];
    let mut best_k = [0usize; 4];
    let mut best_score = [f64::INFINITY; 4];

    let mut nx = [0.0f64; 4];
    if COMPUTE_NORM {
        let mut nx0 = [0.0f64; 4];
        let mut nx1 = [0.0f64; 4];
        let mut nx2 = [0.0f64; 4];
        let mut nx3 = [0.0f64; 4];
        let (r0c, r0r) = r0.as_chunks::<4>();
        let (r1c, r1r) = r1.as_chunks::<4>();
        let (r2c, r2r) = r2.as_chunks::<4>();
        let (r3c, r3r) = r3.as_chunks::<4>();
        for (((&v0, &v1), &v2), &v3) in r0c.iter().zip(r1c.iter()).zip(r2c.iter()).zip(r3c.iter()) {
            for l in 0..4 {
                nx0[l] += v0[l] * v0[l];
                nx1[l] += v1[l] * v1[l];
                nx2[l] += v2[l] * v2[l];
                nx3[l] += v3[l] * v3[l];
            }
        }
        nx[0] = (nx0[0] + nx0[1]) + (nx0[2] + nx0[3]);
        nx[1] = (nx1[0] + nx1[1]) + (nx1[2] + nx1[3]);
        nx[2] = (nx2[0] + nx2[1]) + (nx2[2] + nx2[3]);
        nx[3] = (nx3[0] + nx3[1]) + (nx3[2] + nx3[3]);
        for j in 0..r0r.len() {
            nx[0] += r0r[j] * r0r[j];
            nx[1] += r1r[j] * r1r[j];
            nx[2] += r2r[j] * r2r[j];
            nx[3] += r3r[j] * r3r[j];
        }
    }

    let mut c = 0usize;
    while c + 2 <= k {
        let c0 = &centers_data[c * p..(c + 1) * p];
        let c1 = &centers_data[(c + 1) * p..(c + 2) * p];

        let mut d00 = [0.0f64; 4];
        let mut d10 = [0.0f64; 4];
        let mut d20 = [0.0f64; 4];
        let mut d30 = [0.0f64; 4];
        let mut d01 = [0.0f64; 4];
        let mut d11 = [0.0f64; 4];
        let mut d21 = [0.0f64; 4];
        let mut d31 = [0.0f64; 4];

        let (c0c, c0r) = c0.as_chunks::<4>();
        let (c1c, c1r) = c1.as_chunks::<4>();
        let (r0c, r0r) = r0.as_chunks::<4>();
        let (r1c, r1r) = r1.as_chunks::<4>();
        let (r2c, r2r) = r2.as_chunks::<4>();
        let (r3c, r3r) = r3.as_chunks::<4>();

        for (((((&vc0, &vc1), &vx0), &vx1), &vx2), &vx3) in c0c
            .iter()
            .zip(c1c.iter())
            .zip(r0c.iter())
            .zip(r1c.iter())
            .zip(r2c.iter())
            .zip(r3c.iter())
        {
            for l in 0..4 {
                d00[l] += vx0[l] * vc0[l];
                d01[l] += vx0[l] * vc1[l];
                d10[l] += vx1[l] * vc0[l];
                d11[l] += vx1[l] * vc1[l];
                d20[l] += vx2[l] * vc0[l];
                d21[l] += vx2[l] * vc1[l];
                d30[l] += vx3[l] * vc0[l];
                d31[l] += vx3[l] * vc1[l];
            }
        }

        let mut s00 = (d00[0] + d00[1]) + (d00[2] + d00[3]);
        let mut s10 = (d10[0] + d10[1]) + (d10[2] + d10[3]);
        let mut s20 = (d20[0] + d20[1]) + (d20[2] + d20[3]);
        let mut s30 = (d30[0] + d30[1]) + (d30[2] + d30[3]);
        let mut s01 = (d01[0] + d01[1]) + (d01[2] + d01[3]);
        let mut s11 = (d11[0] + d11[1]) + (d11[2] + d11[3]);
        let mut s21 = (d21[0] + d21[1]) + (d21[2] + d21[3]);
        let mut s31 = (d31[0] + d31[1]) + (d31[2] + d31[3]);

        for j in 0..c0r.len() {
            let vc0 = c0r[j];
            let vc1 = c1r[j];
            s00 += r0r[j] * vc0;
            s01 += r0r[j] * vc1;
            s10 += r1r[j] * vc0;
            s11 += r1r[j] * vc1;
            s20 += r2r[j] * vc0;
            s21 += r2r[j] * vc1;
            s30 += r3r[j] * vc0;
            s31 += r3r[j] * vc1;
        }

        let hn0 = half_norms[c];
        let hn1 = half_norms[c + 1];

        let sc00 = hn0 - s00;
        let sc01 = hn1 - s01;
        if sc00 < best_score[0] {
            best_score[0] = sc00;
            best_k[0] = c;
        }
        if sc01 < best_score[0] {
            best_score[0] = sc01;
            best_k[0] = c + 1;
        }

        let sc10 = hn0 - s10;
        let sc11 = hn1 - s11;
        if sc10 < best_score[1] {
            best_score[1] = sc10;
            best_k[1] = c;
        }
        if sc11 < best_score[1] {
            best_score[1] = sc11;
            best_k[1] = c + 1;
        }

        let sc20 = hn0 - s20;
        let sc21 = hn1 - s21;
        if sc20 < best_score[2] {
            best_score[2] = sc20;
            best_k[2] = c;
        }
        if sc21 < best_score[2] {
            best_score[2] = sc21;
            best_k[2] = c + 1;
        }

        let sc30 = hn0 - s30;
        let sc31 = hn1 - s31;
        if sc30 < best_score[3] {
            best_score[3] = sc30;
            best_k[3] = c;
        }
        if sc31 < best_score[3] {
            best_score[3] = sc31;
            best_k[3] = c + 1;
        }

        c += 2;
    }

    if c < k {
        let cr = &centers_data[c * p..(c + 1) * p];
        let mut d0 = [0.0f64; 4];
        let mut d1 = [0.0f64; 4];
        let mut d2 = [0.0f64; 4];
        let mut d3 = [0.0f64; 4];
        let (crc, crr) = cr.as_chunks::<4>();
        let (r0c, r0r) = r0.as_chunks::<4>();
        let (r1c, r1r) = r1.as_chunks::<4>();
        let (r2c, r2r) = r2.as_chunks::<4>();
        let (r3c, r3r) = r3.as_chunks::<4>();
        for ((((&vc, &vx0), &vx1), &vx2), &vx3) in crc
            .iter()
            .zip(r0c.iter())
            .zip(r1c.iter())
            .zip(r2c.iter())
            .zip(r3c.iter())
        {
            for l in 0..4 {
                d0[l] += vx0[l] * vc[l];
                d1[l] += vx1[l] * vc[l];
                d2[l] += vx2[l] * vc[l];
                d3[l] += vx3[l] * vc[l];
            }
        }
        let mut s0 = (d0[0] + d0[1]) + (d0[2] + d0[3]);
        let mut s1 = (d1[0] + d1[1]) + (d1[2] + d1[3]);
        let mut s2 = (d2[0] + d2[1]) + (d2[2] + d2[3]);
        let mut s3 = (d3[0] + d3[1]) + (d3[2] + d3[3]);
        for j in 0..crr.len() {
            let vc = crr[j];
            s0 += r0r[j] * vc;
            s1 += r1r[j] * vc;
            s2 += r2r[j] * vc;
            s3 += r3r[j] * vc;
        }
        let hn = half_norms[c];
        if hn - s0 < best_score[0] {
            best_score[0] = hn - s0;
            best_k[0] = c;
        }
        if hn - s1 < best_score[1] {
            best_score[1] = hn - s1;
            best_k[1] = c;
        }
        if hn - s2 < best_score[2] {
            best_score[2] = hn - s2;
            best_k[2] = c;
        }
        if hn - s3 < best_score[3] {
            best_score[3] = hn - s3;
            best_k[3] = c;
        }
    }

    let sq = if COMPUTE_NORM {
        [
            (nx[0] + 2.0 * best_score[0]).max(0.0),
            (nx[1] + 2.0 * best_score[1]).max(0.0),
            (nx[2] + 2.0 * best_score[2]).max(0.0),
            (nx[3] + 2.0 * best_score[3]).max(0.0),
        ]
    } else {
        [0.0; 4]
    };
    (best_k, sq)
}

#[inline(always)]
fn assign_1row_general<const COMPUTE_NORM: bool>(
    row: &[f64],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) -> (usize, f64) {
    let row = &row[..p];
    let mut nx = 0.0f64;
    if COMPUTE_NORM {
        let mut nxa = [0.0f64; 4];
        let (rc, rr) = row.as_chunks::<4>();
        for &v in rc {
            for l in 0..4 {
                nxa[l] += v[l] * v[l];
            }
        }
        nx = (nxa[0] + nxa[1]) + (nxa[2] + nxa[3]);
        for &v in rr {
            nx += v * v;
        }
    }

    let mut best_k = 0usize;
    let mut best_score = f64::INFINITY;
    let mut c = 0usize;

    while c + 4 <= k {
        let c0 = &centers_data[c * p..(c + 1) * p];
        let c1 = &centers_data[(c + 1) * p..(c + 2) * p];
        let c2 = &centers_data[(c + 2) * p..(c + 3) * p];
        let c3 = &centers_data[(c + 3) * p..(c + 4) * p];
        let mut acc0 = [0.0f64; 4];
        let mut acc1 = [0.0f64; 4];
        let mut acc2 = [0.0f64; 4];
        let mut acc3 = [0.0f64; 4];

        let (rc, rr) = row.as_chunks::<4>();
        let (c0c, c0r) = c0.as_chunks::<4>();
        let (c1c, c1r) = c1.as_chunks::<4>();
        let (c2c, c2r) = c2.as_chunks::<4>();
        let (c3c, c3r) = c3.as_chunks::<4>();

        for ((((&vx, &vc0), &vc1), &vc2), &vc3) in rc
            .iter()
            .zip(c0c.iter())
            .zip(c1c.iter())
            .zip(c2c.iter())
            .zip(c3c.iter())
        {
            for l in 0..4 {
                acc0[l] += vx[l] * vc0[l];
                acc1[l] += vx[l] * vc1[l];
                acc2[l] += vx[l] * vc2[l];
                acc3[l] += vx[l] * vc3[l];
            }
        }
        let mut s0 = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
        let mut s1 = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
        let mut s2 = (acc2[0] + acc2[1]) + (acc2[2] + acc2[3]);
        let mut s3 = (acc3[0] + acc3[1]) + (acc3[2] + acc3[3]);
        for j in 0..rr.len() {
            let xj = rr[j];
            s0 += xj * c0r[j];
            s1 += xj * c1r[j];
            s2 += xj * c2r[j];
            s3 += xj * c3r[j];
        }
        let v0 = half_norms[c] - s0;
        let v1 = half_norms[c + 1] - s1;
        let v2 = half_norms[c + 2] - s2;
        let v3 = half_norms[c + 3] - s3;

        if v0 < best_score {
            best_score = v0;
            best_k = c;
        }
        if v1 < best_score {
            best_score = v1;
            best_k = c + 1;
        }
        if v2 < best_score {
            best_score = v2;
            best_k = c + 2;
        }
        if v3 < best_score {
            best_score = v3;
            best_k = c + 3;
        }
        c += 4;
    }

    while c < k {
        let cr = &centers_data[c * p..(c + 1) * p];
        let mut acc = [0.0f64; 4];
        let (rc, rr) = row.as_chunks::<4>();
        let (crc, crr) = cr.as_chunks::<4>();
        for (&vx, &vc) in rc.iter().zip(crc.iter()) {
            for l in 0..4 {
                acc[l] += vx[l] * vc[l];
            }
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for j in 0..rr.len() {
            s += rr[j] * crr[j];
        }
        let v = half_norms[c] - s;
        if v < best_score {
            best_score = v;
            best_k = c;
        }
        c += 1;
    }

    let sq = if COMPUTE_NORM {
        (nx + 2.0 * best_score).max(0.0)
    } else {
        0.0
    };
    (best_k, sq)
}

#[inline]
fn assign_subchunk_packed<const COMPUTE_NORM: bool, const WRITE_LABELS: bool>(
    x_sub: &[f64],
    mut lbl_sub: Option<&mut [usize]>,
    w_sub: Option<&[f64]>,
    packed: &PackedCenters,
    centers_data: &[f64],
    rows: usize,
) -> f64 {
    let mut inertia = 0.0f64;
    let mut i = 0usize;

    match packed {
        PackedCenters::K4 { c_t, hn, k, p } => {
            let p = *p;
            let k = *k;
            if p == 0 || k == 0 {
                return 0.0;
            }
            while i + 4 <= rows {
                let base = i * p;
                let (best, sq) = assign_4rows_k4::<COMPUTE_NORM>(
                    &x_sub[base..base + p],
                    &x_sub[base + p..base + 2 * p],
                    &x_sub[base + 2 * p..base + 3 * p],
                    &x_sub[base + 3 * p..base + 4 * p],
                    c_t,
                    hn,
                    k,
                    p,
                );
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best[0];
                    lbl[i + 1] = best[1];
                    lbl[i + 2] = best[2];
                    lbl[i + 3] = best[3];
                }
                if COMPUTE_NORM {
                    match w_sub {
                        Some(w) => {
                            inertia += w[i] * sq[0]
                                + w[i + 1] * sq[1]
                                + w[i + 2] * sq[2]
                                + w[i + 3] * sq[3];
                        }
                        None => inertia += (sq[0] + sq[1]) + (sq[2] + sq[3]),
                    }
                }
                i += 4;
            }
            while i < rows {
                let row = &x_sub[i * p..(i + 1) * p];
                let (best, sq) = assign_1row_k4::<COMPUTE_NORM>(row, c_t, hn, k, p);
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best;
                }
                if COMPUTE_NORM {
                    inertia += w_sub.map_or(sq, |w| w[i] * sq);
                }
                i += 1;
            }
        }
        PackedCenters::K8 { c_t, hn, k, p } => {
            let p = *p;
            let k = *k;
            if p == 0 || k == 0 {
                return 0.0;
            }
            while i + 4 <= rows {
                let base = i * p;
                let (best, sq) = assign_4rows_k8::<COMPUTE_NORM>(
                    &x_sub[base..base + p],
                    &x_sub[base + p..base + 2 * p],
                    &x_sub[base + 2 * p..base + 3 * p],
                    &x_sub[base + 3 * p..base + 4 * p],
                    c_t,
                    hn,
                    k,
                    p,
                );
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best[0];
                    lbl[i + 1] = best[1];
                    lbl[i + 2] = best[2];
                    lbl[i + 3] = best[3];
                }
                if COMPUTE_NORM {
                    match w_sub {
                        Some(w) => {
                            inertia += w[i] * sq[0]
                                + w[i + 1] * sq[1]
                                + w[i + 2] * sq[2]
                                + w[i + 3] * sq[3];
                        }
                        None => inertia += (sq[0] + sq[1]) + (sq[2] + sq[3]),
                    }
                }
                i += 4;
            }
            while i < rows {
                let row = &x_sub[i * p..(i + 1) * p];
                let (best, sq) = assign_1row_k8::<COMPUTE_NORM>(row, c_t, hn, k, p);
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best;
                }
                if COMPUTE_NORM {
                    inertia += w_sub.map_or(sq, |w| w[i] * sq);
                }
                i += 1;
            }
        }
        PackedCenters::General { half_norms, k, p } => {
            let p = *p;
            let k = *k;
            if p == 0 || k == 0 {
                return 0.0;
            }
            while i + 4 <= rows {
                let base = i * p;
                let (best, sq) = assign_4rows_general::<COMPUTE_NORM>(
                    &x_sub[base..base + p],
                    &x_sub[base + p..base + 2 * p],
                    &x_sub[base + 2 * p..base + 3 * p],
                    &x_sub[base + 3 * p..base + 4 * p],
                    centers_data,
                    half_norms,
                    k,
                    p,
                );
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best[0];
                    lbl[i + 1] = best[1];
                    lbl[i + 2] = best[2];
                    lbl[i + 3] = best[3];
                }
                if COMPUTE_NORM {
                    match w_sub {
                        Some(w) => {
                            inertia += w[i] * sq[0]
                                + w[i + 1] * sq[1]
                                + w[i + 2] * sq[2]
                                + w[i + 3] * sq[3];
                        }
                        None => inertia += (sq[0] + sq[1]) + (sq[2] + sq[3]),
                    }
                }
                i += 4;
            }
            while i < rows {
                let row = &x_sub[i * p..(i + 1) * p];
                let (best, sq) =
                    assign_1row_general::<COMPUTE_NORM>(row, centers_data, half_norms, k, p);
                if WRITE_LABELS && let Some(ref mut lbl) = lbl_sub {
                    lbl[i] = best;
                }
                if COMPUTE_NORM {
                    inertia += w_sub.map_or(sq, |w| w[i] * sq);
                }
                i += 1;
            }
        }
    }
    inertia
}

#[inline]
fn assign_chunk_l1_checked<const COMPUTE_NORM: bool, const WRITE_LABELS: bool>(
    x_chunk: &[f64],
    mut lbl_chunk: Option<&mut [usize]>,
    w_chunk: Option<&[f64]>,
    packed: &PackedCenters,
    centers_data: &[f64],
    rows: usize,
    has_non_finite: Option<&AtomicBool>,
) -> f64 {
    if let Some(flag) = has_non_finite
        && flag.load(Ordering::Relaxed)
    {
        return 0.0;
    }
    let p = match packed {
        PackedCenters::K4 { p, .. }
        | PackedCenters::K8 { p, .. }
        | PackedCenters::General { p, .. } => *p,
    };
    if rows == 0 || p == 0 {
        return 0.0;
    }
    let sub_rows = ((2048 / p.max(1)).clamp(4, 64)) & !3;
    let sub_rows = sub_rows.max(4);
    let mut total_inertia = 0.0f64;
    let mut start = 0usize;
    while start < rows {
        let end = (start + sub_rows).min(rows);
        let x_sub = &x_chunk[start * p..end * p];
        if let Some(flag) = has_non_finite
            && !slice_all_finite(x_sub)
        {
            flag.store(true, Ordering::Relaxed);
            return 0.0;
        }
        let lbl_sub = lbl_chunk.as_deref_mut().map(|l| &mut l[start..end]);
        let w_sub = w_chunk.map(|w| &w[start..end]);
        total_inertia += assign_subchunk_packed::<COMPUTE_NORM, WRITE_LABELS>(
            x_sub,
            lbl_sub,
            w_sub,
            packed,
            centers_data,
            end - start,
        );
        start = end;
    }
    total_inertia
}

fn cpu_assign_labels_impl(
    x: &Matrix,
    centers: &Matrix,
    check_finite: bool,
) -> OchreResult<Vec<usize>> {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    let mut labels = vec![0usize; n];
    if n == 0 || p == 0 || k == 0 {
        return Ok(labels);
    }
    let packed = PackedCenters::new(&centers.data, k, p);
    let flag = AtomicBool::new(false);
    let flag_opt = if check_finite { Some(&flag) } else { None };

    if n < ASSIGN_PAR_ROW_THRESHOLD {
        assign_chunk_l1_checked::<false, true>(
            &x.data,
            Some(&mut labels),
            None,
            &packed,
            &centers.data,
            n,
            flag_opt,
        );
    } else {
        let chunk_rows = if n >= 16_384 {
            PREDICT_CHUNK_ROWS
        } else {
            1024
        };
        labels
            .par_chunks_mut(chunk_rows)
            .zip(x.data.par_chunks(chunk_rows * p))
            .for_each(|(lbl_chunk, x_chunk)| {
                let rows = lbl_chunk.len();
                assign_chunk_l1_checked::<false, true>(
                    x_chunk,
                    Some(lbl_chunk),
                    None,
                    &packed,
                    &centers.data,
                    rows,
                    flag_opt,
                );
            });
    }

    if check_finite && flag.load(Ordering::Relaxed) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or infinite values ( >_< )".to_string(),
        ));
    }
    Ok(labels)
}

/// Zero-allocation fused nearest-center label-only kernel for `predict`.
/// Minimizes `0.5 * ||c_j||^2 - dot(x_i, c_j)` with register-friendly center packing.
#[cfg_attr(not(test), allow(dead_code))]
fn cpu_assign_labels_only(x: &Matrix, centers: &Matrix) -> Vec<usize> {
    match cpu_assign_labels_impl(x, centers, false) {
        Ok(labels) => labels,
        Err(_) => Vec::new(),
    }
}

/// Zero-allocation fused nearest-center label-only kernel with fused finite check.
fn cpu_assign_labels_checked(x: &Matrix, centers: &Matrix) -> OchreResult<Vec<usize>> {
    cpu_assign_labels_impl(x, centers, true)
}

fn cpu_assign_labels_and_inertia_impl(
    x: &Matrix,
    centers: &Matrix,
    sw: Option<&[f64]>,
    labels: &mut [usize],
    check_finite: bool,
) -> OchreResult<f64> {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    if n == 0 || p == 0 || k == 0 {
        return Ok(0.0);
    }
    let packed = PackedCenters::new(&centers.data, k, p);
    let flag = AtomicBool::new(false);
    let flag_opt = if check_finite { Some(&flag) } else { None };

    let inertia = if n < ASSIGN_PAR_ROW_THRESHOLD {
        assign_chunk_l1_checked::<true, true>(
            &x.data,
            Some(labels),
            sw,
            &packed,
            &centers.data,
            n,
            flag_opt,
        )
    } else {
        let chunk_rows = if n >= 16_384 {
            PREDICT_CHUNK_ROWS
        } else {
            1024
        };
        match sw {
            Some(weights) => labels
                .par_chunks_mut(chunk_rows)
                .zip(x.data.par_chunks(chunk_rows * p))
                .zip(weights.par_chunks(chunk_rows))
                .map(|((lbl_chunk, x_chunk), w_chunk)| {
                    let rows = lbl_chunk.len();
                    assign_chunk_l1_checked::<true, true>(
                        x_chunk,
                        Some(lbl_chunk),
                        Some(w_chunk),
                        &packed,
                        &centers.data,
                        rows,
                        flag_opt,
                    )
                })
                .sum(),
            None => labels
                .par_chunks_mut(chunk_rows)
                .zip(x.data.par_chunks(chunk_rows * p))
                .map(|(lbl_chunk, x_chunk)| {
                    let rows = lbl_chunk.len();
                    assign_chunk_l1_checked::<true, true>(
                        x_chunk,
                        Some(lbl_chunk),
                        None,
                        &packed,
                        &centers.data,
                        rows,
                        flag_opt,
                    )
                })
                .sum(),
        }
    };

    if check_finite && flag.load(Ordering::Relaxed) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or infinite values ( >_< )".to_string(),
        ));
    }
    Ok(inertia)
}

/// Zero-extra-alloc fused nearest-center label + inertia kernel computing
/// `sum_i w_i * (||x_i||^2 + 2 * (0.5 * ||c_best||^2 - x_i . c_best))` inside Rayon chunks.
fn cpu_assign_labels_and_inertia(
    x: &Matrix,
    centers: &Matrix,
    sw: Option<&[f64]>,
    labels: &mut [usize],
) -> f64 {
    cpu_assign_labels_and_inertia_impl(x, centers, sw, labels, false).unwrap_or(0.0)
}

/// Zero-extra-alloc fused nearest-center label + inertia kernel with fused finite check.
fn cpu_assign_labels_and_inertia_checked(
    x: &Matrix,
    centers: &Matrix,
    sw: Option<&[f64]>,
    labels: &mut [usize],
) -> OchreResult<f64> {
    cpu_assign_labels_and_inertia_impl(x, centers, sw, labels, true)
}

/// Zero-allocation fused inertia-only kernel with fused finite check for `score`.
fn cpu_inertia_only_checked(x: &Matrix, centers: &Matrix, sw: Option<&[f64]>) -> OchreResult<f64> {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    if n == 0 || p == 0 || k == 0 {
        return Ok(0.0);
    }
    let packed = PackedCenters::new(&centers.data, k, p);
    let flag = AtomicBool::new(false);

    let inertia = if n < ASSIGN_PAR_ROW_THRESHOLD {
        assign_chunk_l1_checked::<true, false>(
            &x.data,
            None,
            sw,
            &packed,
            &centers.data,
            n,
            Some(&flag),
        )
    } else {
        let chunk_rows = if n >= 16_384 {
            PREDICT_CHUNK_ROWS
        } else {
            1024
        };
        match sw {
            Some(weights) => x
                .data
                .par_chunks(chunk_rows * p)
                .zip(weights.par_chunks(chunk_rows))
                .map(|(x_chunk, w_chunk)| {
                    let rows = w_chunk.len();
                    assign_chunk_l1_checked::<true, false>(
                        x_chunk,
                        None,
                        Some(w_chunk),
                        &packed,
                        &centers.data,
                        rows,
                        Some(&flag),
                    )
                })
                .sum(),
            None => x
                .data
                .par_chunks(chunk_rows * p)
                .map(|x_chunk| {
                    let rows = x_chunk.len() / p;
                    assign_chunk_l1_checked::<true, false>(
                        x_chunk,
                        None,
                        None,
                        &packed,
                        &centers.data,
                        rows,
                        Some(&flag),
                    )
                })
                .sum(),
        }
    };

    if flag.load(Ordering::Relaxed) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or infinite values ( >_< )".to_string(),
        ));
    }
    Ok(inertia)
}

/// Fused Lloyd assignment + parallel centroid accumulation in a single memory pass over `X`.
fn cpu_lloyd_step_fused(
    x: &Matrix,
    centers: &Matrix,
    sw: Option<&[f64]>,
    labels: &mut [usize],
    sq_dists: &mut [f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    let centers_data = &centers.data;

    if n < PAR_ROW_THRESHOLD {
        let mut new_centers = vec![0.0f64; k * p];
        let mut cluster_weights = vec![0.0f64; k];
        match sw {
            Some(weights) => {
                for (i, &w) in weights.iter().enumerate().take(n) {
                    let row = &x.data[i * p..(i + 1) * p];
                    let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
                    labels[i] = best_k;
                    sq_dists[i] = best_dist;
                    cluster_weights[best_k] += w;
                    let dst = &mut new_centers[best_k * p..(best_k + 1) * p];
                    for j in 0..p {
                        dst[j] += w * row[j];
                    }
                }
            }
            None => {
                for i in 0..n {
                    let row = &x.data[i * p..(i + 1) * p];
                    let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
                    labels[i] = best_k;
                    sq_dists[i] = best_dist;
                    cluster_weights[best_k] += 1.0;
                    let dst = &mut new_centers[best_k * p..(best_k + 1) * p];
                    for j in 0..p {
                        dst[j] += row[j];
                    }
                }
            }
        }
        return (new_centers, cluster_weights);
    }

    const CHUNK_ROWS: usize = 2048;
    match sw {
        Some(weights) => labels
            .par_chunks_mut(CHUNK_ROWS)
            .zip(sq_dists.par_chunks_mut(CHUNK_ROWS))
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .zip(weights.par_chunks(CHUNK_ROWS))
            .fold(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut local_c, mut local_w), (((lbl_chunk, dist_chunk), x_chunk), w_chunk)| {
                    for (i, &w) in w_chunk.iter().enumerate() {
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
                        lbl_chunk[i] = best_k;
                        dist_chunk[i] = best_dist;
                        local_w[best_k] += w;
                        let dst = &mut local_c[best_k * p..(best_k + 1) * p];
                        for j in 0..p {
                            dst[j] += w * row[j];
                        }
                    }
                    (local_c, local_w)
                },
            )
            .reduce(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut ac, mut aw), (bc, bw)| {
                    for idx in 0..ac.len() {
                        ac[idx] += bc[idx];
                    }
                    for idx in 0..aw.len() {
                        aw[idx] += bw[idx];
                    }
                    (ac, aw)
                },
            ),
        None => labels
            .par_chunks_mut(CHUNK_ROWS)
            .zip(sq_dists.par_chunks_mut(CHUNK_ROWS))
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .fold(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut local_c, mut local_w), ((lbl_chunk, dist_chunk), x_chunk)| {
                    let rows = lbl_chunk.len();
                    for i in 0..rows {
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let (best_k, best_dist) = nearest_center_row(row, centers_data, k, p);
                        lbl_chunk[i] = best_k;
                        dist_chunk[i] = best_dist;
                        local_w[best_k] += 1.0;
                        let dst = &mut local_c[best_k * p..(best_k + 1) * p];
                        for j in 0..p {
                            dst[j] += row[j];
                        }
                    }
                    (local_c, local_w)
                },
            )
            .reduce(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut ac, mut aw), (bc, bw)| {
                    for idx in 0..ac.len() {
                        ac[idx] += bc[idx];
                    }
                    for idx in 0..aw.len() {
                        aw[idx] += bw[idx];
                    }
                    (ac, aw)
                },
            ),
    }
}

/// Parallel centroid accumulation from pre-computed `labels` via `par_chunks(2048)` + thread-local reduction.
fn accumulate_centers_parallel(
    x: &Matrix,
    labels: &[usize],
    sw: Option<&[f64]>,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.rows;
    let p = x.cols;
    if n < PAR_ROW_THRESHOLD {
        let mut new_centers = vec![0.0f64; k * p];
        let mut cluster_weights = vec![0.0f64; k];
        match sw {
            Some(weights) => {
                for i in 0..n {
                    let c = labels[i];
                    let w = weights[i];
                    cluster_weights[c] += w;
                    let row = &x.data[i * p..(i + 1) * p];
                    let dst = &mut new_centers[c * p..(c + 1) * p];
                    for j in 0..p {
                        dst[j] += w * row[j];
                    }
                }
            }
            None => {
                for (i, &c) in labels.iter().enumerate().take(n) {
                    cluster_weights[c] += 1.0;
                    let row = &x.data[i * p..(i + 1) * p];
                    let dst = &mut new_centers[c * p..(c + 1) * p];
                    for j in 0..p {
                        dst[j] += row[j];
                    }
                }
            }
        }
        return (new_centers, cluster_weights);
    }

    const CHUNK_ROWS: usize = 2048;
    match sw {
        Some(weights) => x
            .data
            .par_chunks(CHUNK_ROWS * p)
            .zip(labels.par_chunks(CHUNK_ROWS))
            .zip(weights.par_chunks(CHUNK_ROWS))
            .fold(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut local_c, mut local_w), ((x_chunk, lbl_chunk), w_chunk)| {
                    for i in 0..lbl_chunk.len() {
                        let c = lbl_chunk[i];
                        let w = w_chunk[i];
                        local_w[c] += w;
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let dst = &mut local_c[c * p..(c + 1) * p];
                        for j in 0..p {
                            dst[j] += w * row[j];
                        }
                    }
                    (local_c, local_w)
                },
            )
            .reduce(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut ac, mut aw), (bc, bw)| {
                    for idx in 0..ac.len() {
                        ac[idx] += bc[idx];
                    }
                    for idx in 0..aw.len() {
                        aw[idx] += bw[idx];
                    }
                    (ac, aw)
                },
            ),
        None => x
            .data
            .par_chunks(CHUNK_ROWS * p)
            .zip(labels.par_chunks(CHUNK_ROWS))
            .fold(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut local_c, mut local_w), (x_chunk, lbl_chunk)| {
                    for (i, &c) in lbl_chunk.iter().enumerate() {
                        local_w[c] += 1.0;
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let dst = &mut local_c[c * p..(c + 1) * p];
                        for j in 0..p {
                            dst[j] += row[j];
                        }
                    }
                    (local_c, local_w)
                },
            )
            .reduce(
                || (vec![0.0f64; k * p], vec![0.0f64; k]),
                |(mut ac, mut aw), (bc, bw)| {
                    for idx in 0..ac.len() {
                        ac[idx] += bc[idx];
                    }
                    for idx in 0..aw.len() {
                        aw[idx] += bw[idx];
                    }
                    (ac, aw)
                },
            ),
    }
}

/// Compute nearest center labels and squared distances into pre-allocated buffers.
fn assign_labels_and_sq_dists_into(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
    labels: &mut [usize],
    sq_dists: &mut [f64],
) -> OchreResult<()> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if should_use_fused_cpu(backend, n, k, p) {
        cpu_assign_labels_fused_into(x, centers, labels, sq_dists);
        return Ok(());
    }

    let (l, d) = assign_labels_and_sq_dists(backend, x, centers)?;
    labels.copy_from_slice(&l);
    sq_dists.copy_from_slice(&d);
    Ok(())
}

/// Compute nearest center labels and squared distances using fused CPU kernel or
/// Backend Primitives N3 (`pairwise_l2_sq`) + N4 (`row_argmin`), chunking large inputs.
fn assign_labels_and_sq_dists(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<(Vec<usize>, Vec<f64>)> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if should_use_fused_cpu(backend, n, k, p) {
        return Ok(cpu_assign_labels_fused(x, centers));
    }

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

fn assign_labels_only_checked(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<Vec<usize>> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if should_use_cpu_predict(backend, n, k, p) {
        return cpu_assign_labels_checked(x, centers);
    }
    validate_x(x)?;
    let (labels, _) = assign_labels_and_sq_dists(backend, x, centers)?;
    Ok(labels)
}

/// Assign nearest center labels and compute weighted inertia without allocating a `sq_dists` buffer on CPU.
fn assign_labels_and_inertia_into(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
    sw: Option<&[f64]>,
    labels: &mut [usize],
) -> OchreResult<f64> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if should_use_cpu_predict(backend, n, k, p) {
        return Ok(cpu_assign_labels_and_inertia(x, centers, sw, labels));
    }
    let mut sq_dists = vec![0.0f64; n];
    assign_labels_and_sq_dists_into(backend, x, centers, labels, &mut sq_dists)?;
    Ok(compute_weighted_inertia(&sq_dists, sw))
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
    if should_use_fused_cpu(backend, n, k, p) {
        let mut out = vec![0.0f64; n * k];
        let centers_data = &centers.data;
        if n < PAR_ROW_THRESHOLD {
            for i in 0..n {
                let row = &x.data[i * p..(i + 1) * p];
                let out_row = &mut out[i * k..(i + 1) * k];
                for c in 0..k {
                    let c_row = &centers_data[c * p..(c + 1) * p];
                    out_row[c] = sq_dist_row(row, c_row).sqrt();
                }
            }
        } else {
            const CHUNK_ROWS: usize = 1024;
            out.par_chunks_mut(CHUNK_ROWS * k)
                .zip(x.data.par_chunks(CHUNK_ROWS * p))
                .for_each(|(out_chunk, x_chunk)| {
                    let rows = x_chunk.len() / p;
                    for i in 0..rows {
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let out_row = &mut out_chunk[i * k..(i + 1) * k];
                        for c in 0..k {
                            let c_row = &centers_data[c * p..(c + 1) * p];
                            out_row[c] = sq_dist_row(row, c_row).sqrt();
                        }
                    }
                });
        }
        return Matrix::from_vec(n, k, out);
    }

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

/// Update `min_sq_dist[i] = min(min_sq_dist[i], ||x_i - center||^2)` and return per-chunk potential sums.
fn update_min_sq_dist_and_chunk_sums(
    x: &Matrix,
    center: &[f64],
    sample_weight: Option<&[f64]>,
    min_sq_dist: &mut [f64],
    is_first: bool,
) -> (Vec<f64>, f64) {
    let n = x.rows;
    let p = x.cols;
    const CHUNK_ROWS: usize = 2048;

    if n < PAR_ROW_THRESHOLD {
        let mut total = 0.0;
        match sample_weight {
            Some(sw) => {
                for i in 0..n {
                    let d = sq_dist_row(&x.data[i * p..(i + 1) * p], center);
                    let m = if is_first || d < min_sq_dist[i] {
                        min_sq_dist[i] = d;
                        d
                    } else {
                        min_sq_dist[i]
                    };
                    total += m * sw[i];
                }
            }
            None => {
                for (i, m_slot) in min_sq_dist.iter_mut().enumerate().take(n) {
                    let d = sq_dist_row(&x.data[i * p..(i + 1) * p], center);
                    let m = if is_first || d < *m_slot {
                        *m_slot = d;
                        d
                    } else {
                        *m_slot
                    };
                    total += m;
                }
            }
        }
        return (vec![total], total);
    }

    let chunk_sums: Vec<f64> = match sample_weight {
        Some(sw) => min_sq_dist
            .par_chunks_mut(CHUNK_ROWS)
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .zip(sw.par_chunks(CHUNK_ROWS))
            .map(|((m_chunk, x_chunk), w_chunk)| {
                let mut sum = 0.0;
                for i in 0..m_chunk.len() {
                    let d = sq_dist_row(&x_chunk[i * p..(i + 1) * p], center);
                    let m = if is_first || d < m_chunk[i] {
                        m_chunk[i] = d;
                        d
                    } else {
                        m_chunk[i]
                    };
                    sum += m * w_chunk[i];
                }
                sum
            })
            .collect(),
        None => min_sq_dist
            .par_chunks_mut(CHUNK_ROWS)
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .map(|(m_chunk, x_chunk)| {
                let mut sum = 0.0;
                for i in 0..m_chunk.len() {
                    let d = sq_dist_row(&x_chunk[i * p..(i + 1) * p], center);
                    let m = if is_first || d < m_chunk[i] {
                        m_chunk[i] = d;
                        d
                    } else {
                        m_chunk[i]
                    };
                    sum += m;
                }
                sum
            })
            .collect(),
    };
    let total: f64 = chunk_sums.iter().sum();
    (chunk_sums, total)
}

fn init_centers_kmeans_pp(
    _backend: &Backend<'_>,
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

    let mut min_sq_dist = vec![0.0f64; n];
    let (mut chunk_sums, mut total) = update_min_sq_dist_and_chunk_sums(
        x,
        &centers_data[0..p],
        sample_weight,
        &mut min_sq_dist,
        true,
    );

    const CHUNK_ROWS: usize = 2048;

    for c in 1..n_clusters {
        let next_idx = if total <= 0.0 || !total.is_finite() {
            rng.gen_range(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cum = 0.0;
            let mut chosen = n - 1;
            if chunk_sums.len() == 1 {
                for i in 0..n {
                    let pot = min_sq_dist[i] * sample_weight.map_or(1.0, |sw| sw[i]);
                    cum += pot;
                    if cum >= target {
                        chosen = i;
                        break;
                    }
                }
            } else {
                for (chunk_idx, &cs) in chunk_sums.iter().enumerate() {
                    if cum + cs >= target {
                        let start = chunk_idx * CHUNK_ROWS;
                        let end = (start + CHUNK_ROWS).min(n);
                        for i in start..end {
                            let pot = min_sq_dist[i] * sample_weight.map_or(1.0, |sw| sw[i]);
                            cum += pot;
                            if cum >= target {
                                chosen = i;
                                break;
                            }
                        }
                        break;
                    }
                    cum += cs;
                }
            }
            chosen
        };

        centers_data[c * p..(c + 1) * p].copy_from_slice(&x.data[next_idx * p..(next_idx + 1) * p]);

        if c + 1 < n_clusters {
            let (cs, tot) = update_min_sq_dist_and_chunk_sums(
                x,
                &centers_data[c * p..(c + 1) * p],
                sample_weight,
                &mut min_sq_dist,
                false,
            );
            chunk_sums = cs;
            total = tot;
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
    if sq_dists.len() >= 16_384 {
        match sample_weight {
            Some(sw) => sq_dists
                .par_chunks(4096)
                .zip(sw.par_chunks(4096))
                .map(|(dc, wc)| {
                    dc.iter()
                        .zip(wc.iter())
                        .map(|(&d, &w)| d.max(0.0) * w)
                        .sum::<f64>()
                })
                .sum(),
            None => sq_dists
                .par_chunks(4096)
                .map(|dc| dc.iter().map(|&d| d.max(0.0)).sum::<f64>())
                .sum(),
        }
    } else {
        match sample_weight {
            Some(sw) => sq_dists
                .iter()
                .zip(sw.iter())
                .map(|(&d, &w)| d.max(0.0) * w)
                .sum(),
            None => sq_dists.iter().map(|&d| d.max(0.0)).sum(),
        }
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
        let tol_scaled = if self.tol > 0.0 {
            self.tol * compute_mean_variance(x, sw)
        } else {
            0.0
        };
        let mut rng = DeterministicRng::new(self.random_state);

        let n = x.rows;
        let p = x.cols;
        let k = self.n_clusters;
        let use_fused = should_use_fused_cpu(&backend, n, k, p);

        let mut best_centers: Option<Matrix> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0usize;

        let mut labels = vec![0usize; n];
        let mut sq_dists = vec![0.0f64; n];

        for _run in 0..self.n_init {
            let mut centers = init_centers(&backend, x, sw, k, &self.init, &mut rng)?;
            let mut run_iters = 0usize;
            let mut last_shift_sq = f64::INFINITY;

            for iter in 0..self.max_iter {
                let (mut new_centers_data, mut cluster_weights) = if use_fused {
                    cpu_lloyd_step_fused(x, &centers, sw, &mut labels, &mut sq_dists)
                } else {
                    assign_labels_and_sq_dists_into(
                        &backend,
                        x,
                        &centers,
                        &mut labels,
                        &mut sq_dists,
                    )?;
                    accumulate_centers_parallel(x, &labels, sw, k)
                };

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
                last_shift_sq = shift_sq;

                if shift_sq <= tol_scaled {
                    break;
                }
            }

            if last_shift_sq > 0.0 {
                assign_labels_and_sq_dists_into(&backend, x, &centers, &mut labels, &mut sq_dists)?;
            }
            let inertia = compute_weighted_inertia(&sq_dists, sw);

            if inertia < best_inertia || best_centers.is_none() {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels.clone());
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
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        assign_labels_only_checked(&backend, x, centers)
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

    #[allow(clippy::too_many_arguments)]
    fn minibatch_step_into(
        backend: &Backend<'_>,
        x_batch: &Matrix,
        w_batch: Option<&[f64]>,
        centers: &mut Matrix,
        counts: &mut [f64],
        reassignment_ratio: f64,
        rng: &mut DeterministicRng,
        labels: &mut [usize],
        sq_dists: &mut [f64],
    ) -> OchreResult<f64> {
        assign_labels_and_sq_dists_into(backend, x_batch, centers, labels, sq_dists)?;
        let n = x_batch.rows;
        let p = x_batch.cols;
        let k = centers.rows;

        let batch_inertia = compute_weighted_inertia(sq_dists, w_batch);

        match w_batch {
            Some(weights) => {
                for i in 0..n {
                    let w = weights[i];
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
            }
            None => {
                for (i, &c) in labels.iter().enumerate().take(n) {
                    counts[c] += 1.0;
                    let eta = 1.0 / counts[c];
                    let row = &x_batch.data[i * p..(i + 1) * p];
                    let center = &mut centers.data[c * p..(c + 1) * p];
                    for j in 0..p {
                        center[j] += eta * (row[j] - center[j]);
                    }
                }
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
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        if self.n_clusters > x.rows {
            return Err(OchreError::InvalidParameter(format!(
                "n_clusters={} should be <= n_samples={} ( >_< )",
                self.n_clusters, x.rows
            )));
        }

        let backend = get_backend(&self.device_type);
        let mut rng = DeterministicRng::new(self.random_state);

        let n = x.rows;
        let p = x.cols;
        let k = self.n_clusters;
        let eff_batch = self.batch_size.min(n).max(1);
        let ewma_alpha = ((2.0 * eff_batch as f64) / (n as f64 + 1.0)).clamp(0.01, 1.0);
        let init_size = (3 * eff_batch).max(3 * k).min(n);
        let defer_full_validate = init_size < n && should_use_cpu_predict(&backend, n, k, p);
        if !defer_full_validate {
            validate_x(x)?;
        }

        let mut best_centers: Option<Matrix> = None;
        let mut best_counts: Option<Vec<f64>> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_val_inertia = f64::INFINITY;
        let mut best_n_iter = 0usize;
        let mut best_n_steps = 0usize;

        let mut init_mat = if init_size < n {
            Some(Matrix::zeros(init_size, p))
        } else {
            None
        };
        let mut init_weights = if init_size < n && sw.is_some() {
            Some(vec![0.0f64; init_size])
        } else {
            None
        };

        // Fixed validation subsample of size `init_size` created once before `n_init` loop
        // so candidate runs are scored without full `n`-row assignment on every `n_init` iteration.
        let use_val_subsample = self.n_init > 1 && init_size < n;
        let (val_mat, val_weights) = if use_val_subsample {
            let mut v_mat = Matrix::zeros(init_size, p);
            let mut v_weights = sw.map(|_| vec![0.0f64; init_size]);
            let mut v_w_sum = 0.0;
            for b in 0..init_size {
                let idx = rng.gen_range(n);
                v_mat.data[b * p..(b + 1) * p].copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
                if let (Some(weights), Some(vw)) = (sw, v_weights.as_mut()) {
                    let w = weights[idx];
                    vw[b] = w;
                    v_w_sum += w;
                }
            }
            if v_w_sum <= 0.0 {
                v_weights = None;
            }
            (Some(v_mat), v_weights)
        } else {
            (None, None)
        };

        // Estimate feature variance on `val_mat` (or full `x` only when `init_size >= n`)
        // so large datasets are not scanned upfront before minibatching.
        let precomputed_tol_scaled = if self.tol <= 0.0 {
            Some(0.0)
        } else if let Some(ref vm) = val_mat {
            Some(self.tol * compute_mean_variance(vm, val_weights.as_deref()))
        } else if init_size >= n {
            Some(self.tol * compute_mean_variance(x, sw))
        } else {
            None
        };

        let mut x_batch = Matrix::zeros(eff_batch, p);
        let mut batch_weights = vec![0.0f64; eff_batch];
        let mut batch_labels = vec![0usize; eff_batch];
        let mut batch_sq_dists = vec![0.0f64; eff_batch];
        let mut val_labels = if use_val_subsample {
            vec![0usize; init_size]
        } else {
            Vec::new()
        };
        let mut val_sq_dists = if use_val_subsample {
            vec![0.0f64; init_size]
        } else {
            Vec::new()
        };
        let mut cand_labels = if self.n_init > 1 && !use_val_subsample {
            vec![0usize; n]
        } else {
            Vec::new()
        };

        for _run in 0..self.n_init {
            let mut sub_w_sum = 0.0;
            let mut centers = if let Some(ref mut sub_mat) = init_mat {
                for b in 0..init_size {
                    let idx = rng.gen_range(n);
                    sub_mat.data[b * p..(b + 1) * p]
                        .copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
                    if let (Some(weights), Some(sub_w)) = (sw, init_weights.as_mut()) {
                        let w = weights[idx];
                        sub_w[b] = w;
                        sub_w_sum += w;
                    }
                }
                if sw.is_some() && sub_w_sum <= 0.0 {
                    init_centers(&backend, x, sw, k, &self.init, &mut rng)?
                } else {
                    init_centers(
                        &backend,
                        sub_mat,
                        init_weights.as_deref(),
                        k,
                        &self.init,
                        &mut rng,
                    )?
                }
            } else {
                init_centers(&backend, x, sw, k, &self.init, &mut rng)?
            };

            let tol_scaled = match precomputed_tol_scaled {
                Some(t) => t,
                None => {
                    if let Some(ref sub_mat) = init_mat
                        && !(sw.is_some() && sub_w_sum <= 0.0)
                    {
                        self.tol * compute_mean_variance(sub_mat, init_weights.as_deref())
                    } else {
                        self.tol * compute_mean_variance(x, sw)
                    }
                }
            };

            let mut counts = vec![0.0; k];
            let mut ewma_inertia: Option<f64> = None;
            let mut min_ewma = f64::INFINITY;
            let mut no_improvement = 0usize;
            let mut steps = 0usize;

            for _step in 0..self.max_iter {
                let w_sum = match sw {
                    Some(weights) => {
                        let mut sum = 0.0;
                        for (b, bw) in batch_weights.iter_mut().enumerate().take(eff_batch) {
                            let idx = rng.gen_range(n);
                            x_batch.data[b * p..(b + 1) * p]
                                .copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
                            let w = weights[idx];
                            *bw = w;
                            sum += w;
                        }
                        sum
                    }
                    None => {
                        for b in 0..eff_batch {
                            let idx = rng.gen_range(n);
                            x_batch.data[b * p..(b + 1) * p]
                                .copy_from_slice(&x.data[idx * p..(idx + 1) * p]);
                        }
                        eff_batch as f64
                    }
                };

                let w_opt = if sw.is_some() {
                    Some(batch_weights.as_slice())
                } else {
                    None
                };

                let batch_inertia = Self::minibatch_step_into(
                    &backend,
                    &x_batch,
                    w_opt,
                    &mut centers,
                    &mut counts,
                    self.reassignment_ratio,
                    &mut rng,
                    &mut batch_labels,
                    &mut batch_sq_dists,
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

            if self.n_init == 1 {
                best_centers = Some(centers);
                best_counts = Some(counts);
                best_n_iter = steps.div_ceil((n.div_ceil(eff_batch)).max(1));
                best_n_steps = steps;
            } else if let Some(ref v_mat) = val_mat {
                assign_labels_and_sq_dists_into(
                    &backend,
                    v_mat,
                    &centers,
                    &mut val_labels,
                    &mut val_sq_dists,
                )?;
                let val_inertia = compute_weighted_inertia(&val_sq_dists, val_weights.as_deref());
                if val_inertia < best_val_inertia || best_centers.is_none() {
                    best_val_inertia = val_inertia;
                    best_centers = Some(centers);
                    best_counts = Some(counts);
                    best_n_iter = steps.div_ceil((n.div_ceil(eff_batch)).max(1));
                    best_n_steps = steps;
                }
            } else {
                let inertia =
                    assign_labels_and_inertia_into(&backend, x, &centers, sw, &mut cand_labels)?;
                if inertia < best_inertia || best_centers.is_none() {
                    best_inertia = inertia;
                    best_centers = Some(centers);
                    best_counts = Some(counts);
                    best_labels = Some(cand_labels.clone());
                    best_n_iter = steps.div_ceil((n.div_ceil(eff_batch)).max(1));
                    best_n_steps = steps;
                }
            }
        }

        if best_labels.is_none()
            && let Some(ref centers) = best_centers
        {
            let mut final_labels = vec![0usize; n];
            best_inertia = if defer_full_validate {
                cpu_assign_labels_and_inertia_checked(x, centers, sw, &mut final_labels)?
            } else {
                assign_labels_and_inertia_into(&backend, x, centers, sw, &mut final_labels)?
            };
            best_labels = Some(final_labels);
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

        let mut labels = vec![0usize; x.rows];
        let mut sq_dists = vec![0.0f64; x.rows];

        let _ = Self::minibatch_step_into(
            &backend,
            x,
            sw,
            centers,
            counts,
            self.reassignment_ratio,
            &mut rng,
            &mut labels,
            &mut sq_dists,
        )?;

        self.inertia_ = Some(assign_labels_and_inertia_into(
            &backend,
            x,
            centers,
            sw,
            &mut labels,
        )?);
        self.labels_ = Some(labels);
        self.n_steps_ += 1;
        self.n_iter_ += 1;
        self.device_ = Some(self.device_type.to_string());
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<usize>> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        assign_labels_only_checked(&backend, x, centers)
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
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        if should_use_cpu_predict(&backend, x.rows, centers.rows, x.cols) {
            return Ok(-cpu_inertia_only_checked(x, centers, sw)?);
        }
        validate_x(x)?;
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

    #[test]
    fn test_parallel_paths_large_dataset_parity() {
        let n = 5_000usize;
        let p = 12usize;
        let k = 8usize;
        let mut rng = DeterministicRng::new(Some(12345));
        let mut data = vec![0.0f64; n * p];
        let mut weights = vec![0.0f64; n];
        for i in 0..n {
            let cluster = i % k;
            let base = (cluster as f64) * 10.0;
            weights[i] = 0.5 + (i % 5) as f64 * 0.25;
            for j in 0..p {
                data[i * p + j] = base + (j as f64) * 0.1 + (rng.next_f64() - 0.5) * 0.4;
            }
        }
        let x = Matrix::from_vec(n, p, data).unwrap();

        let mut km = KMeans::new(
            k,
            "k-means++".to_string(),
            3,
            100,
            1e-4,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        let labels = km.fit_predict(&x, None).unwrap();
        assert_eq!(labels.len(), n);
        assert_eq!(km.predict(&x).unwrap(), labels);
        // All samples from the same synthetic cluster should share a label
        for c in 0..k {
            let expected = labels[c];
            for i in (c..n).step_by(k) {
                assert_eq!(labels[i], expected);
            }
        }
        let dists = km.transform(&x).unwrap();
        assert_eq!(dists.rows, n);
        assert_eq!(dists.cols, k);
        assert!((km.score(&x, None).unwrap() + km.inertia_.unwrap()).abs() < 1e-9);

        // Weighted parallel KMeans
        let mut km_w = KMeans::new(
            k,
            "k-means++".to_string(),
            2,
            100,
            1e-4,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        km_w.fit(&x, Some(&weights)).unwrap();
        assert!((km_w.score(&x, Some(&weights)).unwrap() + km_w.inertia_.unwrap()).abs() < 1e-9);

        // Parallel MiniBatchKMeans with subsampled init
        let mut mbk = MiniBatchKMeans::new(
            k,
            "k-means++".to_string(),
            100,
            512,
            0.0,
            10,
            3,
            0.01,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        let mb_labels = mbk.fit_predict(&x, None).unwrap();
        assert_eq!(mb_labels.len(), n);
        assert_eq!(mbk.predict(&x).unwrap(), mb_labels);
        for c in 0..k {
            let expected = mb_labels[c];
            for i in (c..n).step_by(k) {
                assert_eq!(mb_labels[i], expected);
            }
        }
    }

    #[test]
    fn test_predict_fma_identity_matches_sq_dist_all_shapes() {
        let mut rng = DeterministicRng::new(Some(98765));
        for &(n, k, p) in &[
            (64, 3, 5),
            (100, 5, 11),
            (1023, 1, 1),
            (1024, 2, 3),
            (1027, 9, 12),
            (4500, 10, 7),
        ] {
            let x_data: Vec<f64> = (0..n * p).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let c_data: Vec<f64> = (0..k * p).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let x = Matrix::from_vec(n, p, x_data).unwrap();
            let centers = Matrix::from_vec(k, p, c_data).unwrap();
            let fast_labels = cpu_assign_labels_only(&x, &centers);
            let (exact_labels, _) = cpu_assign_labels_fused(&x, &centers);
            assert_eq!(
                fast_labels, exact_labels,
                "mismatch for n={n}, k={k}, p={p}"
            );
        }
    }

    #[test]
    fn test_minibatch_kmeans_subsampled_n_init_and_weighted_score() {
        let n = 6_000usize;
        let p = 12usize;
        let k = 8usize;
        let mut rng = DeterministicRng::new(Some(2026));
        let mut data = vec![0.0f64; n * p];
        let mut weights = vec![0.0f64; n];
        for i in 0..n {
            let cluster = i % k;
            let base = (cluster as f64) * 12.0;
            weights[i] = 0.5 + (i % 4) as f64 * 0.5;
            for j in 0..p {
                data[i * p + j] = base + (rng.next_f64() - 0.5) * 0.5;
            }
        }
        let x = Matrix::from_vec(n, p, data).unwrap();
        let mut mbk = MiniBatchKMeans::new(
            k,
            "k-means++".to_string(),
            100,
            256,
            0.0,
            10,
            3,
            0.01,
            Some(42),
            Some("auto".to_string()),
        )
        .unwrap();
        let labels = mbk.fit_predict(&x, Some(&weights)).unwrap();
        assert_eq!(labels.len(), n);
        assert_eq!(mbk.predict(&x).unwrap(), labels);
        assert!((mbk.score(&x, Some(&weights)).unwrap() + mbk.inertia_.unwrap()).abs() < 1e-8);
        for c in 0..k {
            let expected = labels[c];
            for i in (c..n).step_by(k) {
                assert_eq!(labels[i], expected);
            }
        }
    }
}
