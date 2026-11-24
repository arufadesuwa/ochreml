// OchreML Primitive N6: batched_histogram_split (*≧ω≦*)
// Quantized u8 feature binning, parallel gradient/hessian histogram accumulation,
// sibling histogram subtraction, and regularized Newton split finding.

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Quantized `u8` feature matrix for histogram-based Tree & Boosting models.
#[derive(Clone, Debug)]
pub struct BinnedMatrix {
    pub data: Vec<u8>,
    /// Feature-major layout: `col_data[f * rows + r]`.
    pub col_data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub num_bins: usize,
    /// Per-feature sorted split thresholds (length <= `num_bins - 1`).
    pub bin_edges: Vec<Vec<f64>>,
}

/// Accumulated gradient, hessian, and sample count for a single `(feature, bin)` cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GradHessBin {
    pub sum_grad: f64,
    pub sum_hess: f64,
    pub count: u32,
}

/// Optimal split found across all features and histogram bins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramSplit {
    pub feature_idx: usize,
    pub bin_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_grad: f64,
    pub left_hess: f64,
    pub left_count: u32,
}

const MAX_QUANTILE_SAMPLES: usize = 65_536;

fn compute_feature_edges(col_vals: &mut [f64], max_bins: usize) -> Vec<f64> {
    col_vals.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = col_vals.len();
    if n <= 1 || max_bins <= 1 {
        return Vec::new();
    }

    let mut edges = Vec::with_capacity(max_bins - 1);
    let step = n as f64 / max_bins as f64;
    for b in 1..max_bins {
        let idx = ((b as f64 * step) as usize).min(n - 1);
        let candidate = col_vals[idx];
        if edges
            .last()
            .is_none_or(|&prev: &f64| candidate > prev + 1e-12)
        {
            edges.push(candidate);
        }
    }
    edges
}

/// Quantize continuous `f64` features into `u8` bins (`0..max_bins`, capped at 256).
pub fn cpu_quantize_features(x: &Matrix, max_bins: usize) -> OchreResult<BinnedMatrix> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if !(2..=256).contains(&max_bins) {
        return Err(OchreError::InvalidParameter(format!(
            "max_bins={} must be within 2..=256 ( >_< )",
            max_bins
        )));
    }

    let rows = x.rows;
    let cols = x.cols;
    let sample_n = rows.min(MAX_QUANTILE_SAMPLES);
    let bin_edges: Vec<Vec<f64>> = (0..cols)
        .into_par_iter()
        .map(|c| {
            let mut col = Vec::with_capacity(sample_n);
            if sample_n == rows {
                for r in 0..rows {
                    col.push(x.data[r * cols + c]);
                }
            } else {
                for i in 0..sample_n {
                    let r = (i * rows) / sample_n;
                    col.push(x.data[r * cols + c]);
                }
            }
            compute_feature_edges(&mut col, max_bins)
        })
        .collect();

    let mut data = vec![0u8; rows * cols];
    let block_rows = (4096 / cols).clamp(16, 512);
    let block_elems = block_rows * cols;
    if rows * cols >= 16_384 {
        // Cache-block by rows and iterate column-outer so each feature's bin_edges stays L1-hot
        data.par_chunks_mut(block_elems)
            .zip(x.data.par_chunks(block_elems))
            .for_each(|(dst_block, src_block)| {
                let n_r = dst_block.len() / cols;
                for (c, edges) in bin_edges.iter().enumerate() {
                    for r in 0..n_r {
                        let val = src_block[r * cols + c];
                        dst_block[r * cols + c] = edges.partition_point(|&e| val > e) as u8;
                    }
                }
            });
    } else {
        for (c, edges) in bin_edges.iter().enumerate() {
            for r in 0..rows {
                let val = x.data[r * cols + c];
                data[r * cols + c] = edges.partition_point(|&e| val > e) as u8;
            }
        }
    }

    let mut col_data = vec![0u8; rows * cols];
    if rows * cols >= 16_384 {
        col_data
            .par_chunks_exact_mut(rows)
            .enumerate()
            .for_each(|(c, col_dst)| {
                for (r, dst) in col_dst.iter_mut().enumerate() {
                    *dst = data[r * cols + c];
                }
            });
    } else {
        for (c, col_dst) in col_data.chunks_exact_mut(rows).enumerate() {
            for (r, dst) in col_dst.iter_mut().enumerate() {
                *dst = data[r * cols + c];
            }
        }
    }

    Ok(BinnedMatrix {
        data,
        col_data,
        rows,
        cols,
        num_bins: max_bins,
        bin_edges,
    })
}

#[inline]
fn accumulate_row_hist(row_bins: &[u8], g: f64, h: f64, num_bins: usize, hist: &mut [GradHessBin]) {
    for (hist_f, &b) in hist.chunks_exact_mut(num_bins).zip(row_bins.iter()) {
        let b_idx = b as usize;
        if b_idx < num_bins {
            let cell = &mut hist_f[b_idx];
            cell.sum_grad += g;
            cell.sum_hess += h;
            cell.count += 1;
        }
    }
}

/// Accumulate `(sum_grad, sum_hess, count)` histograms over `sample_indices` on CPU.
pub fn cpu_batched_histogram_grad(
    binned: &BinnedMatrix,
    sample_indices: &[u32],
    grad: &[f64],
    hess: &[f64],
) -> OchreResult<Vec<GradHessBin>> {
    if grad.len() != binned.rows || hess.len() != binned.rows {
        return Err(OchreError::LengthMismatch {
            features: binned.rows,
            targets: grad.len().min(hess.len()),
        });
    }

    let cols = binned.cols;
    let num_bins = binned.num_bins;
    let hist_len = cols * num_bins;

    if sample_indices.len() >= 16_384 {
        let out = sample_indices
            .par_chunks(4096)
            .fold(
                || vec![GradHessBin::default(); hist_len],
                |mut local_hist, chunk| {
                    for &idx in chunk {
                        let r = idx as usize;
                        let row_bins = &binned.data[r * cols..(r + 1) * cols];
                        accumulate_row_hist(row_bins, grad[r], hess[r], num_bins, &mut local_hist);
                    }
                    local_hist
                },
            )
            .reduce(
                || vec![GradHessBin::default(); hist_len],
                |mut a, b| {
                    for (dst, src) in a.iter_mut().zip(b.iter()) {
                        dst.sum_grad += src.sum_grad;
                        dst.sum_hess += src.sum_hess;
                        dst.count += src.count;
                    }
                    a
                },
            );
        Ok(out)
    } else {
        let mut out = vec![GradHessBin::default(); hist_len];
        for &idx in sample_indices {
            let r = idx as usize;
            let row_bins = &binned.data[r * cols..(r + 1) * cols];
            accumulate_row_hist(row_bins, grad[r], hess[r], num_bins, &mut out);
        }
        Ok(out)
    }
}

/// Subtract child histogram from parent histogram to obtain sibling histogram in `O(P * B)`.
pub fn subtract_histogram(
    parent: &[GradHessBin],
    child: &[GradHessBin],
) -> OchreResult<Vec<GradHessBin>> {
    if parent.len() != child.len() {
        return Err(OchreError::LengthMismatch {
            features: parent.len(),
            targets: child.len(),
        });
    }
    Ok(parent
        .iter()
        .zip(child.iter())
        .map(|(p, c)| GradHessBin {
            sum_grad: p.sum_grad - c.sum_grad,
            sum_hess: p.sum_hess - c.sum_hess,
            count: p.count.saturating_sub(c.count),
        })
        .collect())
}

#[inline]
fn find_best_feature_split(
    f: usize,
    edges: &[f64],
    hist: &[GradHessBin],
    l2_reg: f64,
    min_samples_leaf: u32,
) -> Option<HistogramSplit> {
    if edges.is_empty() {
        return None;
    }
    let mut total_g = 0.0;
    let mut total_h = 0.0;
    let mut total_n = 0u32;
    for bin in hist {
        total_g += bin.sum_grad;
        total_h += bin.sum_hess;
        total_n += bin.count;
    }
    if total_n < 2 * min_samples_leaf {
        return None;
    }
    let parent_score = (total_g * total_g) / (total_h + l2_reg).max(1e-12);

    let mut left_g = 0.0;
    let mut left_h = 0.0;
    let mut left_n = 0u32;
    let mut best_feat: Option<HistogramSplit> = None;

    for (b, &threshold) in edges.iter().enumerate() {
        left_g += hist[b].sum_grad;
        left_h += hist[b].sum_hess;
        left_n += hist[b].count;
        let right_n = total_n.saturating_sub(left_n);
        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }
        let right_g = total_g - left_g;
        let right_h = total_h - left_h;
        let gain = 0.5
            * ((left_g * left_g) / (left_h + l2_reg).max(1e-12)
                + (right_g * right_g) / (right_h + l2_reg).max(1e-12)
                - parent_score);
        if best_feat.is_none_or(|cur| gain > cur.gain) {
            best_feat = Some(HistogramSplit {
                feature_idx: f,
                bin_idx: b,
                threshold,
                gain,
                left_grad: left_g,
                left_hess: left_h,
                left_count: left_n,
            });
        }
    }
    best_feat
}

/// Find the highest-gain split across all features and histogram bins.
pub fn find_best_histogram_split(
    binned: &BinnedMatrix,
    histograms: &[GradHessBin],
    l2_reg: f64,
    min_samples_leaf: u32,
) -> OchreResult<Option<HistogramSplit>> {
    let expected = binned.cols * binned.num_bins;
    if histograms.len() != expected {
        return Err(OchreError::LengthMismatch {
            features: expected,
            targets: histograms.len(),
        });
    }

    let num_bins = binned.num_bins;
    let best = if expected >= 16_384 {
        (0..binned.cols)
            .into_par_iter()
            .filter_map(|f| {
                let hist = &histograms[f * num_bins..(f + 1) * num_bins];
                find_best_feature_split(f, &binned.bin_edges[f], hist, l2_reg, min_samples_leaf)
            })
            .reduce_with(|a, b| if a.gain >= b.gain { a } else { b })
    } else {
        (0..binned.cols)
            .filter_map(|f| {
                let hist = &histograms[f * num_bins..(f + 1) * num_bins];
                find_best_feature_split(f, &binned.bin_edges[f], hist, l2_reg, min_samples_leaf)
            })
            .reduce(|a, b| if a.gain >= b.gain { a } else { b })
    };

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_histogram_large_and_small_parity() {
        let rows = 20_000usize;
        let cols = 6usize;
        let mut data = Vec::with_capacity(rows * cols);
        let mut grad = Vec::with_capacity(rows);
        let mut hess = Vec::with_capacity(rows);
        for r in 0..rows {
            for c in 0..cols {
                data.push((((r * 37 + c * 13) % 997) as f64) / 997.0);
            }
            grad.push(((r % 17) as f64) - 8.0);
            hess.push(1.0 + ((r % 5) as f64) * 0.25);
        }
        let x = Matrix::from_vec(rows, cols, data).unwrap();
        let binned = cpu_quantize_features(&x, 64).unwrap();

        let large_idx: Vec<u32> = (0..rows as u32).step_by(1).collect();
        let hist_large = cpu_batched_histogram_grad(&binned, &large_idx, &grad, &hess).unwrap();

        let split_point = 10_000usize;
        let small_a: Vec<u32> = (0..split_point as u32).collect();
        let small_b: Vec<u32> = (split_point as u32..rows as u32).collect();
        let hist_a = cpu_batched_histogram_grad(&binned, &small_a, &grad, &hess).unwrap();
        let hist_b = cpu_batched_histogram_grad(&binned, &small_b, &grad, &hess).unwrap();

        assert_eq!(hist_large.len(), cols * 64);
        for ((hl, ha), hb) in hist_large.iter().zip(hist_a.iter()).zip(hist_b.iter()) {
            assert_eq!(hl.count, ha.count + hb.count);
            assert!((hl.sum_grad - (ha.sum_grad + hb.sum_grad)).abs() < 1e-9);
            assert!((hl.sum_hess - (ha.sum_hess + hb.sum_hess)).abs() < 1e-9);
        }
    }
}
