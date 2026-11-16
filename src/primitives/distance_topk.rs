// OchreML Primitive N3 (pairwise_l2_sq) & N4 (row_argmin, topk_argmin) (*≧ω≦*)
// Reuses Backend::matmul for large matrix Gram cross-terms and uses register-buffered TopK selection.

use crate::backend::Backend;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;

fn transpose_matrix(m: &Matrix) -> OchreResult<Matrix> {
    let mut data = vec![0.0; m.rows * m.cols];
    for r in 0..m.rows {
        let row = &m.data[r * m.cols..(r + 1) * m.cols];
        for c in 0..m.cols {
            data[c * m.rows + r] = row[c];
        }
    }
    Matrix::from_vec(m.cols, m.rows, data)
}

fn row_sq_norms(m: &Matrix) -> Vec<f64> {
    let cols = m.cols;
    if m.rows * cols < 16_384 {
        m.data
            .chunks_exact(cols)
            .map(|row| row.iter().map(|&v| v * v).sum())
            .collect()
    } else {
        m.data
            .par_chunks(cols)
            .map(|row| row.iter().map(|&v| v * v).sum())
            .collect()
    }
}

fn direct_pairwise_l2_sq(a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    let cols = a.cols;
    let nb = b.rows;
    let mut out = vec![0.0; a.rows * nb];
    out.par_chunks_mut(nb)
        .zip(a.data.par_chunks(cols))
        .for_each(|(dst_row, a_row)| {
            for (j, b_row) in b.data.chunks_exact(cols).enumerate() {
                let mut sum = 0.0;
                for c in 0..cols {
                    let d = a_row[c] - b_row[c];
                    sum += d * d;
                }
                dst_row[j] = sum;
            }
        });
    Matrix::from_vec(a.rows, nb, out)
}

/// Compute squared Euclidean distance matrix D[i, j] = ||A_i - B_j||^2.
/// Uses `Backend::matmul` for `A * B^T` on larger inputs so CUDA/wgpu/TPU accelerate distance matrices.
pub fn backend_pairwise_l2_sq(
    backend: &Backend<'_>,
    a: &Matrix,
    b: &Matrix,
) -> OchreResult<Matrix> {
    if a.cols != b.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: b.cols,
            expected: a.cols,
        });
    }
    if a.rows == 0 || b.rows == 0 {
        return Err(OchreError::EmptyData);
    }

    // Small-dimension direct path avoids transposing B and minimizes floating-point cancellation.
    if a.cols <= 8 && a.rows.saturating_mul(b.rows) <= 250_000 {
        return direct_pairwise_l2_sq(a, b);
    }

    let a_norms = row_sq_norms(a);
    let b_norms = row_sq_norms(b);
    let b_t = transpose_matrix(b)?;
    let mut dist = backend.matmul(a, &b_t)?;
    let nb = b.rows;

    let fuse_row = |(i, row): (usize, &mut [f64])| {
        let an = a_norms[i];
        for j in 0..nb {
            row[j] = (an + b_norms[j] - 2.0 * row[j]).max(0.0);
        }
    };

    if dist.rows * nb < 16_384 {
        dist.data
            .chunks_exact_mut(nb)
            .enumerate()
            .for_each(fuse_row);
    } else {
        dist.data.par_chunks_mut(nb).enumerate().for_each(fuse_row);
    }

    Ok(dist)
}

#[inline]
fn argmin_slice(row: &[f64]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_val = row[0];
    for (idx, &v) in row.iter().enumerate().skip(1) {
        if v < best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    (best_idx, best_val)
}

/// CPU Rayon implementation of `row_argmin`.
pub fn cpu_row_argmin(dist: &Matrix) -> OchreResult<(Vec<usize>, Vec<f64>)> {
    if dist.rows == 0 || dist.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let cols = dist.cols;
    let pairs: Vec<(usize, f64)> = if dist.rows * cols < 16_384 {
        dist.data.chunks_exact(cols).map(argmin_slice).collect()
    } else {
        dist.data.par_chunks(cols).map(argmin_slice).collect()
    };

    let mut indices = Vec::with_capacity(pairs.len());
    let mut values = Vec::with_capacity(pairs.len());
    for (idx, val) in pairs {
        indices.push(idx);
        values.push(val);
    }
    Ok((indices, values))
}

fn topk_row_small(row: &[f64], k: usize, out_idx: &mut [usize], out_val: &mut [f64]) {
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
    for (idx, &val) in row.iter().enumerate() {
        if buf.len() < k {
            let pos = buf.partition_point(|&(v, _)| v <= val);
            buf.insert(pos, (val, idx));
        } else if val < buf[k - 1].0 {
            let pos = buf.partition_point(|&(v, _)| v <= val);
            buf.insert(pos, (val, idx));
            buf.pop();
        }
    }
    for (j, (val, idx)) in buf.into_iter().enumerate() {
        out_idx[j] = idx;
        out_val[j] = val;
    }
}

fn topk_row_large(row: &[f64], k: usize, out_idx: &mut [usize], out_val: &mut [f64]) {
    let mut pairs: Vec<(f64, usize)> = row.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        pairs.truncate(k);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (j, (val, idx)) in pairs.into_iter().enumerate() {
        out_idx[j] = idx;
        out_val[j] = val;
    }
}

/// CPU Rayon implementation of `topk_argmin` returning `(indices_flat, distances_matrix)`.
pub fn cpu_topk_argmin(dist: &Matrix, k: usize) -> OchreResult<(Vec<usize>, Matrix)> {
    if dist.rows == 0 || dist.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if k == 0 || k > dist.cols {
        return Err(OchreError::InvalidParameter(format!(
            "k={} must be within 1..={} ( >_< )",
            k, dist.cols
        )));
    }

    let cols = dist.cols;
    let mut out_idx = vec![0usize; dist.rows * k];
    let mut out_val = vec![0.0f64; dist.rows * k];

    let process = |((idx_row, val_row), src_row): ((&mut [usize], &mut [f64]), &[f64])| {
        if k <= 32 {
            topk_row_small(src_row, k, idx_row, val_row);
        } else {
            topk_row_large(src_row, k, idx_row, val_row);
        }
    };

    if dist.rows * cols < 16_384 {
        out_idx
            .chunks_exact_mut(k)
            .zip(out_val.chunks_exact_mut(k))
            .zip(dist.data.chunks_exact(cols))
            .for_each(process);
    } else {
        out_idx
            .par_chunks_mut(k)
            .zip(out_val.par_chunks_mut(k))
            .zip(dist.data.par_chunks(cols))
            .for_each(process);
    }

    let dist_mat = Matrix::from_vec(dist.rows, k, out_val)?;
    Ok((out_idx, dist_mat))
}
