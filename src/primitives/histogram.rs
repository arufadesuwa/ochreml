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

fn compute_feature_edges(col_vals: &mut [f64], max_bins: usize) -> Vec<f64> {
    col_vals.sort_by(|a, b| a.total_cmp(b));
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
    let bin_edges: Vec<Vec<f64>> = (0..cols)
        .into_par_iter()
        .map(|c| {
            let mut col = Vec::with_capacity(rows);
            for r in 0..rows {
                col.push(x.data[r * cols + c]);
            }
            compute_feature_edges(&mut col, max_bins)
        })
        .collect();

    let mut data = vec![0u8; rows * cols];
    data.par_chunks_mut(cols)
        .zip(x.data.par_chunks(cols))
        .for_each(|(dst_row, src_row)| {
            for c in 0..cols {
                let edges = &bin_edges[c];
                let val = src_row[c];
                dst_row[c] = edges.partition_point(|&e| val > e) as u8;
            }
        });

    Ok(BinnedMatrix {
        data,
        rows,
        cols,
        num_bins: max_bins,
        bin_edges,
    })
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

    let total_bins = binned.cols * binned.num_bins;
    let cols = binned.cols;
    let num_bins = binned.num_bins;

    let accumulate_chunk = |indices: &[u32]| -> Vec<GradHessBin> {
        let mut local = vec![GradHessBin::default(); total_bins];
        for &idx in indices {
            let r = idx as usize;
            let g = grad[r];
            let h = hess[r];
            let row = &binned.data[r * cols..(r + 1) * cols];
            for c in 0..cols {
                let b = row[c] as usize;
                let cell = &mut local[c * num_bins + b];
                cell.sum_grad += g;
                cell.sum_hess += h;
                cell.count += 1;
            }
        }
        local
    };

    if sample_indices.len() < 4096 {
        return Ok(accumulate_chunk(sample_indices));
    }

    let merged = sample_indices
        .par_chunks(2048)
        .map(accumulate_chunk)
        .reduce(
            || vec![GradHessBin::default(); total_bins],
            |mut acc, part| {
                for (a, p) in acc.iter_mut().zip(part.iter()) {
                    a.sum_grad += p.sum_grad;
                    a.sum_hess += p.sum_hess;
                    a.count += p.count;
                }
                acc
            },
        );

    Ok(merged)
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
    let best = (0..binned.cols)
        .into_par_iter()
        .filter_map(|f| {
            let edges = &binned.bin_edges[f];
            if edges.is_empty() {
                return None;
            }
            let hist = &histograms[f * num_bins..(f + 1) * num_bins];
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
        })
        .reduce_with(|a, b| if a.gain >= b.gain { a } else { b });

    Ok(best)
}
