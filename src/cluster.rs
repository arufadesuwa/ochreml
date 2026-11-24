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

pub const TAG_KMEANS: u8 = 47;
pub const TAG_MINI_BATCH_KMEANS: u8 = 48;

const MAX_DIST_ELEMENTS: usize = 4_000_000;
const FUSED_CPU_OPS_THRESHOLD: usize = 4_000_000;
const PAR_ROW_THRESHOLD: usize = 4_096;

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
    let has_non_finite = if x.data.len() >= 65_536 {
        x.data
            .par_chunks(16_384)
            .any(|chunk| chunk.iter().any(|v| !v.is_finite()))
    } else {
        x.data.iter().any(|v| !v.is_finite())
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

fn compute_mean_variance(x: &Matrix, sample_weight: Option<&[f64]>) -> f64 {
    let n = x.rows;
    let p = x.cols;
    if n == 0 || p == 0 {
        return 0.0;
    }

    let (mut means, w_sum) = if n >= PAR_ROW_THRESHOLD {
        const CHUNK_ROWS: usize = 2048;
        match sample_weight {
            Some(sw) => x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .zip(sw.par_chunks(CHUNK_ROWS))
                .fold(
                    || (vec![0.0f64; p], 0.0f64),
                    |(mut acc_m, mut acc_w), (x_chunk, w_chunk)| {
                        for (i, &w) in w_chunk.iter().enumerate() {
                            acc_w += w;
                            let row = &x_chunk[i * p..(i + 1) * p];
                            for j in 0..p {
                                acc_m[j] += w * row[j];
                            }
                        }
                        (acc_m, acc_w)
                    },
                )
                .reduce(
                    || (vec![0.0f64; p], 0.0f64),
                    |(mut a_m, a_w), (b_m, b_w)| {
                        for j in 0..p {
                            a_m[j] += b_m[j];
                        }
                        (a_m, a_w + b_w)
                    },
                ),
            None => {
                let sum_vec = x
                    .data
                    .par_chunks(CHUNK_ROWS * p)
                    .fold(
                        || vec![0.0f64; p],
                        |mut acc_m, x_chunk| {
                            let rows = x_chunk.len() / p;
                            for i in 0..rows {
                                let row = &x_chunk[i * p..(i + 1) * p];
                                for j in 0..p {
                                    acc_m[j] += row[j];
                                }
                            }
                            acc_m
                        },
                    )
                    .reduce(
                        || vec![0.0f64; p],
                        |mut a_m, b_m| {
                            for j in 0..p {
                                a_m[j] += b_m[j];
                            }
                            a_m
                        },
                    );
                (sum_vec, n as f64)
            }
        }
    } else {
        let mut means = vec![0.0; p];
        let mut w_sum = 0.0;
        match sample_weight {
            Some(sw) => {
                for (i, &w) in sw.iter().enumerate().take(n) {
                    w_sum += w;
                    let row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        means[j] += w * row[j];
                    }
                }
            }
            None => {
                w_sum = n as f64;
                for i in 0..n {
                    let row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        means[j] += row[j];
                    }
                }
            }
        }
        (means, w_sum)
    };

    if w_sum <= 0.0 {
        return 0.0;
    }
    let inv_w = 1.0 / w_sum;
    for m in &mut means {
        *m *= inv_w;
    }

    let var_sum: f64 = if n >= PAR_ROW_THRESHOLD {
        const CHUNK_ROWS: usize = 2048;
        match sample_weight {
            Some(sw) => x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .zip(sw.par_chunks(CHUNK_ROWS))
                .map(|(x_chunk, w_chunk)| {
                    let mut local = 0.0;
                    for (i, &w) in w_chunk.iter().enumerate() {
                        let row = &x_chunk[i * p..(i + 1) * p];
                        let mut s = 0.0;
                        for j in 0..p {
                            let d = row[j] - means[j];
                            s += d * d;
                        }
                        local += w * s;
                    }
                    local
                })
                .sum(),
            None => x
                .data
                .par_chunks(CHUNK_ROWS * p)
                .map(|x_chunk| {
                    let rows = x_chunk.len() / p;
                    let mut local = 0.0;
                    for i in 0..rows {
                        let row = &x_chunk[i * p..(i + 1) * p];
                        for j in 0..p {
                            let d = row[j] - means[j];
                            local += d * d;
                        }
                    }
                    local
                })
                .sum(),
        }
    } else {
        let mut var_sum = 0.0;
        match sample_weight {
            Some(sw) => {
                for (i, &w) in sw.iter().enumerate().take(n) {
                    let row = &x.data[i * p..(i + 1) * p];
                    let mut s = 0.0;
                    for j in 0..p {
                        let d = row[j] - means[j];
                        s += d * d;
                    }
                    var_sum += w * s;
                }
            }
            None => {
                for i in 0..n {
                    let row = &x.data[i * p..(i + 1) * p];
                    for j in 0..p {
                        let d = row[j] - means[j];
                        var_sum += d * d;
                    }
                }
            }
        }
        var_sum
    };

    var_sum / (w_sum * p as f64)
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

#[inline(always)]
fn assign_chunk_4centers<const USE_FMA: bool>(
    row: &[f64],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) -> usize {
    let mut best_k = 0usize;
    let mut best_score = f64::NEG_INFINITY;
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

        let mut j = 0usize;
        while j + 4 <= p {
            let x0 = row[j];
            let x1 = row[j + 1];
            let x2 = row[j + 2];
            let x3 = row[j + 3];
            if USE_FMA {
                acc0[0] = x0.mul_add(c0[j], acc0[0]);
                acc0[1] = x1.mul_add(c0[j + 1], acc0[1]);
                acc0[2] = x2.mul_add(c0[j + 2], acc0[2]);
                acc0[3] = x3.mul_add(c0[j + 3], acc0[3]);

                acc1[0] = x0.mul_add(c1[j], acc1[0]);
                acc1[1] = x1.mul_add(c1[j + 1], acc1[1]);
                acc1[2] = x2.mul_add(c1[j + 2], acc1[2]);
                acc1[3] = x3.mul_add(c1[j + 3], acc1[3]);

                acc2[0] = x0.mul_add(c2[j], acc2[0]);
                acc2[1] = x1.mul_add(c2[j + 1], acc2[1]);
                acc2[2] = x2.mul_add(c2[j + 2], acc2[2]);
                acc2[3] = x3.mul_add(c2[j + 3], acc2[3]);

                acc3[0] = x0.mul_add(c3[j], acc3[0]);
                acc3[1] = x1.mul_add(c3[j + 1], acc3[1]);
                acc3[2] = x2.mul_add(c3[j + 2], acc3[2]);
                acc3[3] = x3.mul_add(c3[j + 3], acc3[3]);
            } else {
                acc0[0] += x0 * c0[j];
                acc0[1] += x1 * c0[j + 1];
                acc0[2] += x2 * c0[j + 2];
                acc0[3] += x3 * c0[j + 3];

                acc1[0] += x0 * c1[j];
                acc1[1] += x1 * c1[j + 1];
                acc1[2] += x2 * c1[j + 2];
                acc1[3] += x3 * c1[j + 3];

                acc2[0] += x0 * c2[j];
                acc2[1] += x1 * c2[j + 1];
                acc2[2] += x2 * c2[j + 2];
                acc2[3] += x3 * c2[j + 3];

                acc3[0] += x0 * c3[j];
                acc3[1] += x1 * c3[j + 1];
                acc3[2] += x2 * c3[j + 2];
                acc3[3] += x3 * c3[j + 3];
            }
            j += 4;
        }
        let mut s0 = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
        let mut s1 = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
        let mut s2 = (acc2[0] + acc2[1]) + (acc2[2] + acc2[3]);
        let mut s3 = (acc3[0] + acc3[1]) + (acc3[2] + acc3[3]);
        while j < p {
            let xj = row[j];
            if USE_FMA {
                s0 = xj.mul_add(c0[j], s0);
                s1 = xj.mul_add(c1[j], s1);
                s2 = xj.mul_add(c2[j], s2);
                s3 = xj.mul_add(c3[j], s3);
            } else {
                s0 += xj * c0[j];
                s1 += xj * c1[j];
                s2 += xj * c2[j];
                s3 += xj * c3[j];
            }
            j += 1;
        }
        s0 -= half_norms[c];
        s1 -= half_norms[c + 1];
        s2 -= half_norms[c + 2];
        s3 -= half_norms[c + 3];

        if s0 > best_score {
            best_score = s0;
            best_k = c;
        }
        if s1 > best_score {
            best_score = s1;
            best_k = c + 1;
        }
        if s2 > best_score {
            best_score = s2;
            best_k = c + 2;
        }
        if s3 > best_score {
            best_score = s3;
            best_k = c + 3;
        }
        c += 4;
    }

    while c < k {
        let cr = &centers_data[c * p..(c + 1) * p];
        let mut acc = [0.0f64; 4];
        let mut j = 0usize;
        while j + 4 <= p {
            if USE_FMA {
                acc[0] = row[j].mul_add(cr[j], acc[0]);
                acc[1] = row[j + 1].mul_add(cr[j + 1], acc[1]);
                acc[2] = row[j + 2].mul_add(cr[j + 2], acc[2]);
                acc[3] = row[j + 3].mul_add(cr[j + 3], acc[3]);
            } else {
                acc[0] += row[j] * cr[j];
                acc[1] += row[j + 1] * cr[j + 1];
                acc[2] += row[j + 2] * cr[j + 2];
                acc[3] += row[j + 3] * cr[j + 3];
            }
            j += 4;
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        while j < p {
            if USE_FMA {
                s = row[j].mul_add(cr[j], s);
            } else {
                s += row[j] * cr[j];
            }
            j += 1;
        }
        s -= half_norms[c];
        if s > best_score {
            best_score = s;
            best_k = c;
        }
        c += 1;
    }

    best_k
}

#[inline(always)]
fn assign_labels_chunk_impl<const USE_FMA: bool>(
    x_chunk: &[f64],
    lbl_chunk: &mut [usize],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) {
    if p == 0 || k == 0 {
        return;
    }
    for (row, lbl) in x_chunk.chunks_exact(p).zip(lbl_chunk.iter_mut()) {
        *lbl = assign_chunk_4centers::<USE_FMA>(row, centers_data, half_norms, k, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn assign_labels_chunk_fma(
    x_chunk: &[f64],
    lbl_chunk: &mut [usize],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) {
    assign_labels_chunk_impl::<true>(x_chunk, lbl_chunk, centers_data, half_norms, k, p);
}

#[inline]
fn assign_labels_chunk(
    x_chunk: &[f64],
    lbl_chunk: &mut [usize],
    centers_data: &[f64],
    half_norms: &[f64],
    k: usize,
    p: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: guarded by runtime AVX2 + FMA feature detection.
        unsafe {
            assign_labels_chunk_fma(x_chunk, lbl_chunk, centers_data, half_norms, k, p);
        }
        return;
    }
    assign_labels_chunk_impl::<false>(x_chunk, lbl_chunk, centers_data, half_norms, k, p);
}

/// Zero-allocation fused nearest-center label-only kernel for `predict`.
/// Uses `argmax_k (dot(x_i, c_k) - 0.5 * ||c_k||^2)` with 4-center FMA blocking.
fn cpu_assign_labels_only(x: &Matrix, centers: &Matrix) -> Vec<usize> {
    let n = x.rows;
    let p = x.cols;
    let k = centers.rows;
    let centers_data = &centers.data;
    let mut half_norms = Vec::with_capacity(k);
    for c in 0..k {
        let cr = &centers_data[c * p..(c + 1) * p];
        let mut norm_sq = 0.0;
        for &v in cr {
            norm_sq += v * v;
        }
        half_norms.push(0.5 * norm_sq);
    }
    let mut labels = vec![0usize; n];

    if n < PAR_ROW_THRESHOLD {
        assign_labels_chunk(&x.data, &mut labels, centers_data, &half_norms, k, p);
    } else {
        const CHUNK_ROWS: usize = 2048;
        labels
            .par_chunks_mut(CHUNK_ROWS)
            .zip(x.data.par_chunks(CHUNK_ROWS * p))
            .for_each(|(lbl_chunk, x_chunk)| {
                assign_labels_chunk(x_chunk, lbl_chunk, centers_data, &half_norms, k, p);
            });
    }
    labels
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

fn assign_labels_only(
    backend: &Backend<'_>,
    x: &Matrix,
    centers: &Matrix,
) -> OchreResult<Vec<usize>> {
    let n = x.rows;
    let k = centers.rows;
    let p = x.cols;
    if should_use_fused_cpu(backend, n, k, p) {
        return Ok(cpu_assign_labels_only(x, centers));
    }
    let (labels, _) = assign_labels_and_sq_dists(backend, x, centers)?;
    Ok(labels)
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
        validate_x(x)?;
        validate_fitted_cols(self.n_features_in_, x.cols)?;
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        let backend = get_backend(&self.device_type);
        assign_labels_only(&backend, x, centers)
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
        let eff_batch = self.batch_size.min(n).max(1);
        let ewma_alpha = ((2.0 * eff_batch as f64) / (n as f64 + 1.0)).clamp(0.01, 1.0);
        let init_size = (3 * eff_batch).max(3 * k).min(n);

        let mut best_centers: Option<Matrix> = None;
        let mut best_counts: Option<Vec<f64>> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
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

        let mut x_batch = Matrix::zeros(eff_batch, p);
        let mut batch_weights = vec![0.0f64; eff_batch];
        let mut batch_labels = vec![0usize; eff_batch];
        let mut batch_sq_dists = vec![0.0f64; eff_batch];
        let mut full_labels = vec![0usize; n];
        let mut full_sq_dists = vec![0.0f64; n];

        for _run in 0..self.n_init {
            let mut centers = if let Some(ref mut sub_mat) = init_mat {
                let mut sub_w_sum = 0.0;
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

            assign_labels_and_sq_dists_into(
                &backend,
                x,
                &centers,
                &mut full_labels,
                &mut full_sq_dists,
            )?;
            let inertia = compute_weighted_inertia(&full_sq_dists, sw);

            if inertia < best_inertia || best_centers.is_none() {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_counts = Some(counts);
                best_labels = Some(full_labels.clone());
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

        assign_labels_and_sq_dists_into(&backend, x, centers, &mut labels, &mut sq_dists)?;
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
        assign_labels_only(&backend, x, centers)
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
        for &(n, k, p) in &[(64, 3, 5), (100, 5, 11), (4500, 10, 7)] {
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
}
