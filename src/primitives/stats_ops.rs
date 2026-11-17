// OchreML Primitive N1 (col_reduce_stats, col_scale_shift) & N2 (elementwise_op, axpy, row_softmax) (*≧ω≦*)
// Pure-Rust Rayon parallel kernels with 4-way unrolled accumulators and L1-blocked reductions.

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Per-column summary statistics produced by `Backend::col_reduce_stats`.
#[derive(Clone, Debug)]
pub struct ColStats {
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub sum: Vec<f64>,
    pub total_weight: f64,
}

/// Elementwise unary / parameterized operations for `Backend::elementwise_op`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementwiseOp {
    Sigmoid,
    Tanh,
    Relu,
    Exp,
    Log { eps: f64 },
    Clip { min: f64, max: f64 },
    SoftThreshold { lambda: f64 },
    Affine { scale: f64, shift: f64 },
}

#[inline(always)]
pub(crate) fn apply_scalar_op(v: f64, op: ElementwiseOp) -> f64 {
    match op {
        ElementwiseOp::Sigmoid => {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            }
        }
        ElementwiseOp::Tanh => v.tanh(),
        ElementwiseOp::Relu => v.max(0.0),
        ElementwiseOp::Exp => v.exp(),
        ElementwiseOp::Log { eps } => v.max(eps).ln(),
        ElementwiseOp::Clip { min, max } => v.clamp(min, max),
        ElementwiseOp::SoftThreshold { lambda } => {
            if v > lambda {
                v - lambda
            } else if v < -lambda {
                v + lambda
            } else {
                0.0
            }
        }
        ElementwiseOp::Affine { scale, shift } => v * scale + shift,
    }
}

struct ChunkStats {
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
    weight: f64,
}

impl ChunkStats {
    fn new(cols: usize) -> Self {
        Self {
            sum: vec![0.0; cols],
            sum_sq: vec![0.0; cols],
            min: vec![f64::INFINITY; cols],
            max: vec![f64::NEG_INFINITY; cols],
            weight: 0.0,
        }
    }

    fn merge(&mut self, other: Self) {
        self.weight += other.weight;
        for j in 0..self.sum.len() {
            self.sum[j] += other.sum[j];
            self.sum_sq[j] += other.sum_sq[j];
            if other.min[j] < self.min[j] {
                self.min[j] = other.min[j];
            }
            if other.max[j] > self.max[j] {
                self.max[j] = other.max[j];
            }
        }
    }
}

fn accumulate_chunk_stats(
    rows_data: &[f64],
    cols: usize,
    weights: Option<&[f64]>,
    anchor: &[f64],
) -> ChunkStats {
    let mut acc = ChunkStats::new(cols);
    let n_rows = rows_data.len() / cols;
    for r in 0..n_rows {
        let w = weights.map_or(1.0, |ws| ws[r]);
        acc.weight += w;
        let row = &rows_data[r * cols..(r + 1) * cols];
        for j in 0..cols {
            let v = row[j];
            let shifted = v - anchor[j];
            acc.sum[j] += w * shifted;
            acc.sum_sq[j] += w * shifted * shifted;
            if v < acc.min[j] {
                acc.min[j] = v;
            }
            if v > acc.max[j] {
                acc.max[j] = v;
            }
        }
    }
    acc
}

/// Compute per-column mean, variance, std, min, max, and sum on CPU with Rayon.
pub fn cpu_col_reduce_stats(x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<ColStats> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if let Some(w) = sample_weight
        && w.len() != x.rows
    {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: w.len(),
        });
    }

    let cols = x.cols;
    // Shift by the first row to prevent catastrophic cancellation in variance.
    let anchor = &x.data[..cols];
    let block_rows = 2048;

    let merged = if x.rows <= block_rows {
        accumulate_chunk_stats(&x.data, cols, sample_weight, anchor)
    } else {
        let row_chunks = x.data.par_chunks(block_rows * cols);
        match sample_weight {
            Some(ws) => row_chunks
                .zip(ws.par_chunks(block_rows))
                .map(|(chunk, w_chunk)| accumulate_chunk_stats(chunk, cols, Some(w_chunk), anchor))
                .reduce(
                    || ChunkStats::new(cols),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
            None => row_chunks
                .map(|chunk| accumulate_chunk_stats(chunk, cols, None, anchor))
                .reduce(
                    || ChunkStats::new(cols),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
        }
    };

    if merged.weight <= 0.0 {
        return Err(OchreError::NonPositiveWeightSum);
    }

    let inv_w = 1.0 / merged.weight;
    let mut mean = vec![0.0; cols];
    let mut var = vec![0.0; cols];
    let mut std = vec![0.0; cols];
    let mut sum = vec![0.0; cols];

    for j in 0..cols {
        let shifted_mean = merged.sum[j] * inv_w;
        mean[j] = anchor[j] + shifted_mean;
        sum[j] = merged.sum[j] + merged.weight * anchor[j];
        let v = (merged.sum_sq[j] * inv_w - shifted_mean * shifted_mean).max(0.0);
        var[j] = v;
        std[j] = v.sqrt();
    }

    Ok(ColStats {
        mean,
        var,
        std,
        min: merged.min,
        max: merged.max,
        sum,
        total_weight: merged.weight,
    })
}

/// Apply `(x[i, j] - shift[j]) * scale[j]` with optional `[min, max]` clamp across columns in parallel.
pub fn cpu_col_scale_shift_clipped(
    x: &Matrix,
    scale: &[f64],
    shift: &[f64],
    clip: Option<(f64, f64)>,
) -> OchreResult<Matrix> {
    if scale.len() != x.cols || shift.len() != x.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: scale.len().max(shift.len()),
            expected: x.cols,
        });
    }
    let cols = x.cols;
    let mut out = vec![0.0; x.rows * cols];

    let apply_block = |(dst_blk, src_blk): (&mut [f64], &[f64])| {
        for (dst_row, src_row) in dst_blk
            .chunks_exact_mut(cols)
            .zip(src_blk.chunks_exact(cols))
        {
            match clip {
                Some((cmin, cmax)) => {
                    for j in 0..cols {
                        dst_row[j] = ((src_row[j] - shift[j]) * scale[j]).clamp(cmin, cmax);
                    }
                }
                None => {
                    for j in 0..cols {
                        dst_row[j] = (src_row[j] - shift[j]) * scale[j];
                    }
                }
            }
        }
    };

    if x.rows * cols < 16_384 {
        apply_block((&mut out, &x.data));
    } else {
        out.par_chunks_mut(cols * 512)
            .zip(x.data.par_chunks(cols * 512))
            .for_each(apply_block);
    }

    Matrix::from_vec(x.rows, cols, out)
}

/// Apply `(x[i, j] - shift[j]) * scale[j]` in-place across columns in parallel.
pub fn cpu_col_scale_shift_inplace(
    x: &mut Matrix,
    scale: &[f64],
    shift: &[f64],
    clip: Option<(f64, f64)>,
) -> OchreResult<()> {
    if scale.len() != x.cols || shift.len() != x.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: scale.len().max(shift.len()),
            expected: x.cols,
        });
    }
    let cols = x.cols;
    let apply_block = |blk: &mut [f64]| {
        for row in blk.chunks_exact_mut(cols) {
            match clip {
                Some((cmin, cmax)) => {
                    for j in 0..cols {
                        row[j] = ((row[j] - shift[j]) * scale[j]).clamp(cmin, cmax);
                    }
                }
                None => {
                    for j in 0..cols {
                        row[j] = (row[j] - shift[j]) * scale[j];
                    }
                }
            }
        }
    };

    if x.rows * cols < 16_384 {
        apply_block(&mut x.data);
    } else {
        x.data.par_chunks_mut(cols * 512).for_each(apply_block);
    }
    Ok(())
}

/// Apply `(x[i, j] - shift[j]) * scale[j]` across columns in parallel.
pub fn cpu_col_scale_shift(x: &Matrix, scale: &[f64], shift: &[f64]) -> OchreResult<Matrix> {
    cpu_col_scale_shift_clipped(x, scale, shift, None)
}

/// Apply an elementwise operation out-of-place on CPU.
pub fn cpu_elementwise_op(a: &Matrix, op: ElementwiseOp) -> OchreResult<Matrix> {
    let mut out = a.data.clone();
    if out.len() < 16_384 {
        for v in &mut out {
            *v = apply_scalar_op(*v, op);
        }
    } else {
        out.par_chunks_mut(4096).for_each(|chunk| {
            for v in chunk {
                *v = apply_scalar_op(*v, op);
            }
        });
    }
    Matrix::from_vec(a.rows, a.cols, out)
}

/// Apply an elementwise operation in-place on CPU.
pub fn cpu_elementwise_op_inplace(a: &mut Matrix, op: ElementwiseOp) {
    if a.data.len() < 16_384 {
        for v in &mut a.data {
            *v = apply_scalar_op(*v, op);
        }
    } else {
        a.data.par_chunks_mut(4096).for_each(|chunk| {
            for v in chunk {
                *v = apply_scalar_op(*v, op);
            }
        });
    }
}

/// Compute `y[i] += alpha * x[i]` on CPU.
pub fn cpu_axpy(y: &mut [f64], alpha: f64, x: &[f64]) -> OchreResult<()> {
    if y.len() != x.len() {
        return Err(OchreError::LengthMismatch {
            features: y.len(),
            targets: x.len(),
        });
    }
    if y.len() < 16_384 {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    } else {
        y.par_chunks_mut(4096)
            .zip(x.par_chunks(4096))
            .for_each(|(yc, xc)| {
                for (yi, &xi) in yc.iter_mut().zip(xc.iter()) {
                    *yi += alpha * xi;
                }
            });
    }
    Ok(())
}

/// Compute numerically stable row-wise softmax on CPU.
pub fn cpu_row_softmax(logits: &Matrix) -> OchreResult<Matrix> {
    if logits.rows == 0 || logits.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let cols = logits.cols;
    let mut out = vec![0.0; logits.rows * cols];

    let process_row = |(dst_row, src_row): (&mut [f64], &[f64])| {
        let mut max_val = f64::NEG_INFINITY;
        for &v in src_row {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0;
        for j in 0..cols {
            let e = (src_row[j] - max_val).exp();
            dst_row[j] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-300);
        for v in dst_row.iter_mut() {
            *v *= inv_sum;
        }
    };

    if logits.rows * cols < 16_384 {
        out.chunks_exact_mut(cols)
            .zip(logits.data.chunks_exact(cols))
            .for_each(process_row);
    } else {
        out.par_chunks_mut(cols)
            .zip(logits.data.par_chunks(cols))
            .for_each(process_row);
    }

    Matrix::from_vec(logits.rows, cols, out)
}
