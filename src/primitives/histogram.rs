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

fn compute_bin_edges(x: &Matrix, max_bins: usize) -> OchreResult<Vec<Vec<f64>>> {
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
    Ok((0..cols)
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
        .collect())
}

/// Quantize continuous `f64` features directly into feature-major `col_data` (`u8`) without row-major copy.
pub fn cpu_quantize_features_col_only(x: &Matrix, max_bins: usize) -> OchreResult<BinnedMatrix> {
    let bin_edges = compute_bin_edges(x, max_bins)?;
    let rows = x.rows;
    let cols = x.cols;
    let mut col_data = vec![0u8; rows * cols];

    let quantize_col = |c: usize, col_dst: &mut [u8]| {
        let edges = &bin_edges[c];
        let (dst_chunks, dst_rem) = col_dst.as_chunks_mut::<4>();
        let mut r = 0usize;
        for dst4 in dst_chunks {
            let v0 = x.data[r * cols + c];
            let v1 = x.data[(r + 1) * cols + c];
            let v2 = x.data[(r + 2) * cols + c];
            let v3 = x.data[(r + 3) * cols + c];
            dst4[0] = edges.partition_point(|&e| v0 > e) as u8;
            dst4[1] = edges.partition_point(|&e| v1 > e) as u8;
            dst4[2] = edges.partition_point(|&e| v2 > e) as u8;
            dst4[3] = edges.partition_point(|&e| v3 > e) as u8;
            r += 4;
        }
        for dst in dst_rem {
            let val = x.data[r * cols + c];
            *dst = edges.partition_point(|&e| val > e) as u8;
            r += 1;
        }
    };

    if rows * cols >= 16_384 {
        col_data
            .par_chunks_exact_mut(rows)
            .enumerate()
            .for_each(|(c, col_dst)| quantize_col(c, col_dst));
    } else {
        for (c, col_dst) in col_data.chunks_exact_mut(rows).enumerate() {
            quantize_col(c, col_dst);
        }
    }

    Ok(BinnedMatrix {
        data: Vec::new(),
        col_data,
        rows,
        cols,
        num_bins: max_bins,
        bin_edges,
    })
}

/// Quantize continuous `f64` features into `u8` bins (`0..max_bins`, capped at 256).
pub fn cpu_quantize_features(x: &Matrix, max_bins: usize) -> OchreResult<BinnedMatrix> {
    let bin_edges = compute_bin_edges(x, max_bins)?;
    let rows = x.rows;
    let cols = x.cols;

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
                let (dst_chunks, dst_rem) = col_dst.as_chunks_mut::<4>();
                let mut r = 0usize;
                for dst4 in dst_chunks {
                    dst4[0] = data[r * cols + c];
                    dst4[1] = data[(r + 1) * cols + c];
                    dst4[2] = data[(r + 2) * cols + c];
                    dst4[3] = data[(r + 3) * cols + c];
                    r += 4;
                }
                for dst in dst_rem {
                    *dst = data[r * cols + c];
                    r += 1;
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

trait GhSource: Sync {
    fn get(&self, r: usize) -> (f64, f64);
    fn get4(&self, r0: usize, r1: usize, r2: usize, r3: usize) -> [(f64, f64); 4];
    fn accum_col_range(&self, col: &[u8], start: usize, end: usize, local: &mut [GradHessBin; 256]);
}

struct InterleavedGh<'a>(&'a [(f64, f64)]);

impl GhSource for InterleavedGh<'_> {
    #[inline(always)]
    fn get(&self, r: usize) -> (f64, f64) {
        self.0[r]
    }

    #[inline(always)]
    fn get4(&self, r0: usize, r1: usize, r2: usize, r3: usize) -> [(f64, f64); 4] {
        [self.0[r0], self.0[r1], self.0[r2], self.0[r3]]
    }

    #[inline]
    fn accum_col_range(
        &self,
        col: &[u8],
        start: usize,
        end: usize,
        local: &mut [GradHessBin; 256],
    ) {
        let (c_chunks, c_rem) = col[start..end].as_chunks::<4>();
        let (gh_chunks, gh_rem) = self.0[start..end].as_chunks::<4>();
        for (&c4, &gh4) in c_chunks.iter().zip(gh_chunks.iter()) {
            accum4(local, c4, gh4);
        }
        for (&b, &(g, h)) in c_rem.iter().zip(gh_rem.iter()) {
            let cell = &mut local[b as usize];
            cell.sum_grad += g;
            cell.sum_hess += h;
            cell.count += 1;
        }
    }
}

struct SplitGh<'a>(&'a [f64], &'a [f64]);

impl GhSource for SplitGh<'_> {
    #[inline(always)]
    fn get(&self, r: usize) -> (f64, f64) {
        (self.0[r], self.1[r])
    }

    #[inline(always)]
    fn get4(&self, r0: usize, r1: usize, r2: usize, r3: usize) -> [(f64, f64); 4] {
        [
            (self.0[r0], self.1[r0]),
            (self.0[r1], self.1[r1]),
            (self.0[r2], self.1[r2]),
            (self.0[r3], self.1[r3]),
        ]
    }

    #[inline]
    fn accum_col_range(
        &self,
        col: &[u8],
        start: usize,
        end: usize,
        local: &mut [GradHessBin; 256],
    ) {
        let (c_chunks, c_rem) = col[start..end].as_chunks::<4>();
        let (g_chunks, g_rem) = self.0[start..end].as_chunks::<4>();
        let (h_chunks, h_rem) = self.1[start..end].as_chunks::<4>();
        for ((&c4, &g4), &h4) in c_chunks.iter().zip(g_chunks.iter()).zip(h_chunks.iter()) {
            accum4(
                local,
                c4,
                [
                    (g4[0], h4[0]),
                    (g4[1], h4[1]),
                    (g4[2], h4[2]),
                    (g4[3], h4[3]),
                ],
            );
        }
        for ((&b, &g), &h) in c_rem.iter().zip(g_rem.iter()).zip(h_rem.iter()) {
            let cell = &mut local[b as usize];
            cell.sum_grad += g;
            cell.sum_hess += h;
            cell.count += 1;
        }
    }
}

#[inline(always)]
fn accum4(local: &mut [GradHessBin; 256], b: [u8; 4], gh: [(f64, f64); 4]) {
    let b0 = b[0] as usize;
    let b1 = b[1] as usize;
    let b2 = b[2] as usize;
    let b3 = b[3] as usize;
    local[b0].sum_grad += gh[0].0;
    local[b0].sum_hess += gh[0].1;
    local[b0].count += 1;
    local[b1].sum_grad += gh[1].0;
    local[b1].sum_hess += gh[1].1;
    local[b1].count += 1;
    local[b2].sum_grad += gh[2].0;
    local[b2].sum_hess += gh[2].1;
    local[b2].count += 1;
    local[b3].sum_grad += gh[3].0;
    local[b3].sum_hess += gh[3].1;
    local[b3].count += 1;
}

#[inline]
fn accum_col_indexed<S: GhSource>(
    col: &[u8],
    indices: &[u32],
    src: &S,
    local: &mut [GradHessBin; 256],
) {
    let n_rows = col.len();
    let (idx_chunks, idx_rem) = indices.as_chunks::<4>();
    for &idx4 in idx_chunks {
        let r0 = idx4[0] as usize;
        let r1 = idx4[1] as usize;
        let r2 = idx4[2] as usize;
        let r3 = idx4[3] as usize;
        if r0 < n_rows && r1 < n_rows && r2 < n_rows && r3 < n_rows {
            let b = [col[r0], col[r1], col[r2], col[r3]];
            let gh = src.get4(r0, r1, r2, r3);
            accum4(local, b, gh);
        } else {
            for r in [r0, r1, r2, r3] {
                if r < n_rows {
                    let cell = &mut local[col[r] as usize];
                    let (g, h) = src.get(r);
                    cell.sum_grad += g;
                    cell.sum_hess += h;
                    cell.count += 1;
                }
            }
        }
    }
    for &idx in idx_rem {
        let r = idx as usize;
        if r < n_rows {
            let cell = &mut local[col[r] as usize];
            let (g, h) = src.get(r);
            cell.sum_grad += g;
            cell.sum_hess += h;
            cell.count += 1;
        }
    }
}

#[inline]
fn copy_local_hist(local: &[GradHessBin; 256], hist_f: &mut [GradHessBin]) {
    let n = hist_f.len().min(256);
    hist_f[..n].copy_from_slice(&local[..n]);
    if hist_f.len() > 256 {
        hist_f[256..].fill(GradHessBin::default());
    }
}

#[inline]
fn merge_local_hist(dst: &mut [GradHessBin; 256], src: &[GradHessBin; 256], num_bins: usize) {
    let n = num_bins.min(256);
    for (d, s) in dst[..n].iter_mut().zip(src[..n].iter()) {
        d.sum_grad += s.sum_grad;
        d.sum_hess += s.sum_hess;
        d.count += s.count;
    }
}

#[inline]
fn is_identity_indices(indices: &[u32], n_rows: usize) -> bool {
    if indices.len() != n_rows || n_rows == 0 {
        return false;
    }
    if indices[0] != 0 || indices[n_rows - 1] != (n_rows - 1) as u32 {
        return false;
    }
    indices
        .iter()
        .enumerate()
        .all(|(i, &idx)| idx as usize == i)
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

fn cpu_batched_histogram_impl<S: GhSource>(
    binned: &BinnedMatrix,
    sample_indices: Option<&[u32]>,
    src: &S,
    out: &mut [GradHessBin],
) -> OchreResult<()> {
    let n_rows = binned.rows;
    let cols = binned.cols;
    let num_bins = binned.num_bins;
    let hist_len = cols * num_bins;

    if out.len() != hist_len {
        return Err(OchreError::LengthMismatch {
            features: hist_len,
            targets: out.len(),
        });
    }

    let n_samples = match sample_indices {
        None => n_rows,
        Some(indices) => indices.len(),
    };
    if n_samples == 0 || cols == 0 || n_rows == 0 {
        out.fill(GradHessBin::default());
        return Ok(());
    }

    let is_root = match sample_indices {
        None => true,
        Some(indices) => is_identity_indices(indices, n_rows),
    };
    let has_col_data = binned.col_data.len() == n_rows * cols;

    if has_col_data && cols >= 4 && ((is_root && n_rows >= 2048) || n_samples >= 1024) {
        if cols >= 8 || n_samples < 32_768 {
            out.par_chunks_exact_mut(num_bins)
                .enumerate()
                .for_each(|(f, hist_f)| {
                    let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
                    let mut local = [GradHessBin::default(); 256];
                    if is_root {
                        src.accum_col_range(col, 0, n_rows, &mut local);
                    } else if let Some(indices) = sample_indices {
                        accum_col_indexed(col, indices, src, &mut local);
                    }
                    copy_local_hist(&local, hist_f);
                });
        } else {
            const CHUNK_ROWS: usize = 16_384;
            out.par_chunks_exact_mut(num_bins)
                .enumerate()
                .for_each(|(f, hist_f)| {
                    let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
                    let merged = if is_root {
                        let n_chunks = n_rows.div_ceil(CHUNK_ROWS);
                        (0..n_chunks)
                            .into_par_iter()
                            .map(|c| {
                                let start = c * CHUNK_ROWS;
                                let end = (start + CHUNK_ROWS).min(n_rows);
                                let mut local = [GradHessBin::default(); 256];
                                src.accum_col_range(col, start, end, &mut local);
                                local
                            })
                            .reduce(
                                || [GradHessBin::default(); 256],
                                |mut a, b| {
                                    merge_local_hist(&mut a, &b, num_bins);
                                    a
                                },
                            )
                    } else {
                        let indices = sample_indices.unwrap_or(&[]);
                        indices
                            .par_chunks(CHUNK_ROWS)
                            .map(|chunk| {
                                let mut local = [GradHessBin::default(); 256];
                                accum_col_indexed(col, chunk, src, &mut local);
                                local
                            })
                            .reduce(
                                || [GradHessBin::default(); 256],
                                |mut a, b| {
                                    merge_local_hist(&mut a, &b, num_bins);
                                    a
                                },
                            )
                    };
                    copy_local_hist(&merged, hist_f);
                });
        }
        return Ok(());
    }

    if has_col_data && cols < 4 && n_samples >= 16_384 {
        const CHUNK_ROWS: usize = 16_384;
        out.par_chunks_exact_mut(num_bins)
            .enumerate()
            .for_each(|(f, hist_f)| {
                let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
                let merged = if is_root {
                    let n_chunks = n_rows.div_ceil(CHUNK_ROWS);
                    (0..n_chunks)
                        .into_par_iter()
                        .map(|c| {
                            let start = c * CHUNK_ROWS;
                            let end = (start + CHUNK_ROWS).min(n_rows);
                            let mut local = [GradHessBin::default(); 256];
                            src.accum_col_range(col, start, end, &mut local);
                            local
                        })
                        .reduce(
                            || [GradHessBin::default(); 256],
                            |mut a, b| {
                                merge_local_hist(&mut a, &b, num_bins);
                                a
                            },
                        )
                } else {
                    let indices = sample_indices.unwrap_or(&[]);
                    indices
                        .par_chunks(CHUNK_ROWS)
                        .map(|chunk| {
                            let mut local = [GradHessBin::default(); 256];
                            accum_col_indexed(col, chunk, src, &mut local);
                            local
                        })
                        .reduce(
                            || [GradHessBin::default(); 256],
                            |mut a, b| {
                                merge_local_hist(&mut a, &b, num_bins);
                                a
                            },
                        )
                };
                copy_local_hist(&merged, hist_f);
            });
        return Ok(());
    }

    if has_col_data && (is_root || n_samples >= 256 || binned.data.is_empty()) {
        for (f, hist_f) in out.chunks_exact_mut(num_bins).enumerate() {
            let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
            let mut local = [GradHessBin::default(); 256];
            if is_root {
                src.accum_col_range(col, 0, n_rows, &mut local);
            } else if let Some(indices) = sample_indices {
                accum_col_indexed(col, indices, src, &mut local);
            }
            copy_local_hist(&local, hist_f);
        }
        return Ok(());
    }

    if let Some(indices) = sample_indices
        && indices.len() >= 16_384
    {
        let merged = indices
            .par_chunks(4096)
            .fold(
                || vec![GradHessBin::default(); hist_len],
                |mut local_hist, chunk| {
                    for &idx in chunk {
                        let r = idx as usize;
                        if r < n_rows {
                            let row_bins = &binned.data[r * cols..(r + 1) * cols];
                            let (g, h) = src.get(r);
                            accumulate_row_hist(row_bins, g, h, num_bins, &mut local_hist);
                        }
                    }
                    local_hist
                },
            )
            .reduce(
                || vec![GradHessBin::default(); hist_len],
                |mut a, b| {
                    for (dst, src_b) in a.iter_mut().zip(b.iter()) {
                        dst.sum_grad += src_b.sum_grad;
                        dst.sum_hess += src_b.sum_hess;
                        dst.count += src_b.count;
                    }
                    a
                },
            );
        out.copy_from_slice(&merged);
        return Ok(());
    }

    out.fill(GradHessBin::default());
    if is_root {
        for r in 0..n_rows {
            let row_bins = &binned.data[r * cols..(r + 1) * cols];
            let (g, h) = src.get(r);
            accumulate_row_hist(row_bins, g, h, num_bins, out);
        }
    } else if let Some(indices) = sample_indices {
        for &idx in indices {
            let r = idx as usize;
            if r < n_rows {
                let row_bins = &binned.data[r * cols..(r + 1) * cols];
                let (g, h) = src.get(r);
                accumulate_row_hist(row_bins, g, h, num_bins, out);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct UnitGradBin {
    sum_grad: f64,
    count: u32,
}

#[inline(always)]
fn accum_unit4(local: &mut [UnitGradBin; 256], b: [u8; 4], g: [f64; 4]) {
    let b0 = b[0] as usize;
    let b1 = b[1] as usize;
    let b2 = b[2] as usize;
    let b3 = b[3] as usize;
    local[b0].sum_grad += g[0];
    local[b0].count += 1;
    local[b1].sum_grad += g[1];
    local[b1].count += 1;
    local[b2].sum_grad += g[2];
    local[b2].count += 1;
    local[b3].sum_grad += g[3];
    local[b3].count += 1;
}

#[inline]
fn accum_unit_range(
    col: &[u8],
    grad: &[f64],
    start: usize,
    end: usize,
    local: &mut [UnitGradBin; 256],
) {
    let (c_chunks, c_rem) = col[start..end].as_chunks::<4>();
    let (g_chunks, g_rem) = grad[start..end].as_chunks::<4>();
    for (&c4, &g4) in c_chunks.iter().zip(g_chunks.iter()) {
        accum_unit4(local, c4, g4);
    }
    for (&b, &g) in c_rem.iter().zip(g_rem.iter()) {
        let cell = &mut local[b as usize];
        cell.sum_grad += g;
        cell.count += 1;
    }
}

#[inline]
fn accum_unit_indexed(col: &[u8], indices: &[u32], grad: &[f64], local: &mut [UnitGradBin; 256]) {
    let n_rows = col.len();
    let (idx_chunks, idx_rem) = indices.as_chunks::<4>();
    for &idx4 in idx_chunks {
        let r0 = idx4[0] as usize;
        let r1 = idx4[1] as usize;
        let r2 = idx4[2] as usize;
        let r3 = idx4[3] as usize;
        if r0 < n_rows && r1 < n_rows && r2 < n_rows && r3 < n_rows {
            let b = [col[r0], col[r1], col[r2], col[r3]];
            let g = [grad[r0], grad[r1], grad[r2], grad[r3]];
            accum_unit4(local, b, g);
        } else {
            for r in [r0, r1, r2, r3] {
                if r < n_rows {
                    let cell = &mut local[col[r] as usize];
                    cell.sum_grad += grad[r];
                    cell.count += 1;
                }
            }
        }
    }
    for &idx in idx_rem {
        let r = idx as usize;
        if r < n_rows {
            let cell = &mut local[col[r] as usize];
            cell.sum_grad += grad[r];
            cell.count += 1;
        }
    }
}

#[inline]
fn merge_unit_hist(dst: &mut [UnitGradBin; 256], src: &[UnitGradBin; 256], num_bins: usize) {
    let n = num_bins.min(256);
    for (d, s) in dst[..n].iter_mut().zip(src[..n].iter()) {
        d.sum_grad += s.sum_grad;
        d.count += s.count;
    }
}

#[inline]
fn copy_unit_hist(local: &[UnitGradBin; 256], hist_f: &mut [GradHessBin]) {
    let n = hist_f.len().min(256);
    for (dst, u) in hist_f[..n].iter_mut().zip(local[..n].iter()) {
        dst.sum_grad = u.sum_grad;
        dst.sum_hess = u.count as f64;
        dst.count = u.count;
    }
    if hist_f.len() > 256 {
        hist_f[256..].fill(GradHessBin::default());
    }
}

/// Accumulate `(sum_grad, sum_hess=count, count)` histograms for unit-hessian problems (`hess = 1.0`)
/// using 16-byte L1-resident bins and reading only `grad: &[f64]`.
pub fn cpu_batched_histogram_unit_hess_into(
    binned: &BinnedMatrix,
    sample_indices: Option<&[u32]>,
    grad: &[f64],
    out: &mut [GradHessBin],
) -> OchreResult<()> {
    let n_rows = binned.rows;
    let cols = binned.cols;
    let num_bins = binned.num_bins;
    let hist_len = cols * num_bins;

    if grad.len() != n_rows {
        return Err(OchreError::LengthMismatch {
            features: n_rows,
            targets: grad.len(),
        });
    }
    if out.len() != hist_len {
        return Err(OchreError::LengthMismatch {
            features: hist_len,
            targets: out.len(),
        });
    }

    let n_samples = sample_indices.map_or(n_rows, |s| s.len());
    if n_samples == 0 || cols == 0 || n_rows == 0 {
        out.fill(GradHessBin::default());
        return Ok(());
    }

    let is_root = sample_indices.is_none_or(|indices| is_identity_indices(indices, n_rows));
    let has_col_data = binned.col_data.len() == n_rows * cols;

    if has_col_data && cols >= 4 && ((is_root && n_rows >= 2048) || n_samples >= 1024) {
        if cols >= 8 || n_samples < 32_768 {
            out.par_chunks_exact_mut(num_bins)
                .enumerate()
                .for_each(|(f, hist_f)| {
                    let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
                    let mut local = [UnitGradBin::default(); 256];
                    if is_root {
                        accum_unit_range(col, grad, 0, n_rows, &mut local);
                    } else if let Some(indices) = sample_indices {
                        accum_unit_indexed(col, indices, grad, &mut local);
                    }
                    copy_unit_hist(&local, hist_f);
                });
        } else {
            const CHUNK_ROWS: usize = 16_384;
            out.par_chunks_exact_mut(num_bins)
                .enumerate()
                .for_each(|(f, hist_f)| {
                    let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
                    let merged = if is_root {
                        let n_chunks = n_rows.div_ceil(CHUNK_ROWS);
                        (0..n_chunks)
                            .into_par_iter()
                            .map(|c| {
                                let start = c * CHUNK_ROWS;
                                let end = (start + CHUNK_ROWS).min(n_rows);
                                let mut local = [UnitGradBin::default(); 256];
                                accum_unit_range(col, grad, start, end, &mut local);
                                local
                            })
                            .reduce(
                                || [UnitGradBin::default(); 256],
                                |mut a, b| {
                                    merge_unit_hist(&mut a, &b, num_bins);
                                    a
                                },
                            )
                    } else {
                        let indices = sample_indices.unwrap_or(&[]);
                        indices
                            .par_chunks(CHUNK_ROWS)
                            .map(|chunk| {
                                let mut local = [UnitGradBin::default(); 256];
                                accum_unit_indexed(col, chunk, grad, &mut local);
                                local
                            })
                            .reduce(
                                || [UnitGradBin::default(); 256],
                                |mut a, b| {
                                    merge_unit_hist(&mut a, &b, num_bins);
                                    a
                                },
                            )
                    };
                    copy_unit_hist(&merged, hist_f);
                });
        }
        return Ok(());
    }

    if has_col_data {
        for (f, hist_f) in out.chunks_exact_mut(num_bins).enumerate() {
            let col = &binned.col_data[f * n_rows..(f + 1) * n_rows];
            let mut local = [UnitGradBin::default(); 256];
            if is_root {
                accum_unit_range(col, grad, 0, n_rows, &mut local);
            } else if let Some(indices) = sample_indices {
                accum_unit_indexed(col, indices, grad, &mut local);
            }
            copy_unit_hist(&local, hist_f);
        }
        return Ok(());
    }

    out.fill(GradHessBin::default());
    if is_root {
        for (r, &g) in grad.iter().enumerate() {
            accumulate_row_hist(
                &binned.data[r * cols..(r + 1) * cols],
                g,
                1.0,
                num_bins,
                out,
            );
        }
    } else if let Some(indices) = sample_indices {
        for &idx in indices {
            let r = idx as usize;
            if r < n_rows {
                accumulate_row_hist(
                    &binned.data[r * cols..(r + 1) * cols],
                    grad[r],
                    1.0,
                    num_bins,
                    out,
                );
            }
        }
    }
    Ok(())
}

/// Accumulate `(sum_grad, sum_hess, count)` histograms over interleaved `(grad, hess)` pairs into `out`.
/// When `sample_indices` is `None`, iterates sequentially over `0..binned.rows` without indirection.
pub fn cpu_batched_histogram_gh_into(
    binned: &BinnedMatrix,
    sample_indices: Option<&[u32]>,
    gh: &[(f64, f64)],
    out: &mut [GradHessBin],
) -> OchreResult<()> {
    if gh.len() != binned.rows {
        return Err(OchreError::LengthMismatch {
            features: binned.rows,
            targets: gh.len(),
        });
    }
    let src = InterleavedGh(gh);
    cpu_batched_histogram_impl(binned, sample_indices, &src, out)
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
    let mut out = vec![GradHessBin::default(); binned.cols * binned.num_bins];
    let src = SplitGh(grad, hess);
    cpu_batched_histogram_impl(binned, Some(sample_indices), &src, &mut out)?;
    Ok(out)
}

/// Subtract child histogram from parent histogram in place to obtain sibling histogram in `O(P * B)`.
pub fn subtract_histogram_inplace(
    parent: &mut [GradHessBin],
    child: &[GradHessBin],
) -> OchreResult<()> {
    if parent.len() != child.len() {
        return Err(OchreError::LengthMismatch {
            features: parent.len(),
            targets: child.len(),
        });
    }
    for (p, c) in parent.iter_mut().zip(child.iter()) {
        p.sum_grad -= c.sum_grad;
        p.sum_hess -= c.sum_hess;
        p.count = p.count.saturating_sub(c.count);
    }
    Ok(())
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
    let active_bins = (edges.len() + 1).min(hist.len());
    let mut total_g = 0.0;
    let mut total_h = 0.0;
    let mut total_n = 0u32;
    for bin in &hist[..active_bins] {
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
    let mut best_score = f64::NEG_INFINITY;
    let mut best_bin = 0usize;
    let mut best_thresh = 0.0;
    let mut best_left_g = 0.0;
    let mut best_left_h = 0.0;
    let mut best_left_n = 0u32;

    for (b, &threshold) in edges.iter().enumerate() {
        let bin = hist[b];
        if bin.count == 0 {
            continue;
        }
        left_g += bin.sum_grad;
        left_h += bin.sum_hess;
        left_n += bin.count;
        let right_n = total_n.saturating_sub(left_n);
        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }
        let right_g = total_g - left_g;
        let right_h = total_h - left_h;
        let score = (left_g * left_g) / (left_h + l2_reg).max(1e-12)
            + (right_g * right_g) / (right_h + l2_reg).max(1e-12);
        if score > best_score {
            best_score = score;
            best_bin = b;
            best_thresh = threshold;
            best_left_g = left_g;
            best_left_h = left_h;
            best_left_n = left_n;
        }
    }

    if best_score > f64::NEG_INFINITY {
        Some(HistogramSplit {
            feature_idx: f,
            bin_idx: best_bin,
            threshold: best_thresh,
            gain: 0.5 * (best_score - parent_score),
            left_grad: best_left_g,
            left_hess: best_left_h,
            left_count: best_left_n,
        })
    } else {
        None
    }
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

    #[test]
    fn test_gh_into_root_indexed_and_few_features_parity() {
        for &cols in &[2usize, 12usize] {
            let rows = 18_005usize;
            let num_bins = 64usize;
            let mut data = Vec::with_capacity(rows * cols);
            let mut grad = Vec::with_capacity(rows);
            let mut hess = Vec::with_capacity(rows);
            let mut gh = Vec::with_capacity(rows);
            for r in 0..rows {
                for c in 0..cols {
                    data.push((((r * 31 + c * 17) % 991) as f64) / 991.0);
                }
                let g = ((r % 19) as f64) - 9.0;
                let h = 1.0 + ((r % 7) as f64) * 0.1;
                grad.push(g);
                hess.push(h);
                gh.push((g, h));
            }
            let x = Matrix::from_vec(rows, cols, data).unwrap();
            let binned = cpu_quantize_features(&x, num_bins).unwrap();

            let mut hist_root = vec![GradHessBin::default(); cols * num_bins];
            cpu_batched_histogram_gh_into(&binned, None, &gh, &mut hist_root).unwrap();

            let all_idx: Vec<u32> = (0..rows as u32).collect();
            let hist_split = cpu_batched_histogram_grad(&binned, &all_idx, &grad, &hess).unwrap();
            for (hr, hs) in hist_root.iter().zip(hist_split.iter()) {
                assert_eq!(hr.count, hs.count);
                assert!((hr.sum_grad - hs.sum_grad).abs() < 1e-9);
                assert!((hr.sum_hess - hs.sum_hess).abs() < 1e-9);
            }

            // Test non-identity permutation of length rows (e.g. reversed)
            let rev_idx: Vec<u32> = (0..rows as u32).rev().collect();
            let mut hist_rev = vec![GradHessBin::default(); cols * num_bins];
            cpu_batched_histogram_gh_into(&binned, Some(&rev_idx), &gh, &mut hist_rev).unwrap();
            for (hr, hv) in hist_root.iter().zip(hist_rev.iter()) {
                assert_eq!(hr.count, hv.count);
                assert!((hr.sum_grad - hv.sum_grad).abs() < 1e-9);
                assert!((hr.sum_hess - hv.sum_hess).abs() < 1e-9);
            }

            // Test partial indexed + inplace subtraction
            let left_idx: Vec<u32> = (0..rows as u32).step_by(2).collect();
            let right_idx: Vec<u32> = (1..rows as u32).step_by(2).collect();
            let mut hist_left = vec![GradHessBin::default(); cols * num_bins];
            let mut hist_right = vec![GradHessBin::default(); cols * num_bins];
            cpu_batched_histogram_gh_into(&binned, Some(&left_idx), &gh, &mut hist_left).unwrap();
            cpu_batched_histogram_gh_into(&binned, Some(&right_idx), &gh, &mut hist_right).unwrap();
            let mut hist_sub = hist_root.clone();
            subtract_histogram_inplace(&mut hist_sub, &hist_left).unwrap();
            for (hs, hr) in hist_sub.iter().zip(hist_right.iter()) {
                assert_eq!(hs.count, hr.count);
                assert!((hs.sum_grad - hr.sum_grad).abs() < 1e-9);
                assert!((hs.sum_hess - hr.sum_hess).abs() < 1e-9);
            }

            // Test col-only quantizer + 16B unit-hessian histogram parity
            let binned_col = cpu_quantize_features_col_only(&x, num_bins).unwrap();
            assert!(binned_col.data.is_empty());
            assert_eq!(binned_col.col_data, binned.col_data);
            assert_eq!(binned_col.bin_edges, binned.bin_edges);

            let unit_hess = vec![1.0; rows];
            let hist_unit_expected =
                cpu_batched_histogram_grad(&binned_col, &left_idx, &grad, &unit_hess).unwrap();
            let mut hist_unit_fast = vec![GradHessBin::default(); cols * num_bins];
            cpu_batched_histogram_unit_hess_into(
                &binned_col,
                Some(&left_idx),
                &grad,
                &mut hist_unit_fast,
            )
            .unwrap();
            for (hf, he) in hist_unit_fast.iter().zip(hist_unit_expected.iter()) {
                assert_eq!(hf.count, he.count);
                assert!((hf.sum_grad - he.sum_grad).abs() < 1e-9);
                assert!((hf.sum_hess - he.sum_hess).abs() < 1e-9);
            }
        }
    }
}
