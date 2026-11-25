// OchreML Discriminant Analysis Module (*≧ω≦*)
// Implements LinearDiscriminantAnalysis (LDA) and QuadraticDiscriminantAnalysis (QDA)
// accelerated by Backend Primitives (N1, N5, E1, E2, E4).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize, write_str,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::logistic_regression::{extract_sorted_classes, map_labels_to_indices};
use crate::matrix::{Matrix, solve_cholesky_multi};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::borrow::Cow;

pub const TAG_LDA: u8 = 45;
pub const TAG_QDA: u8 = 46;

const PARALLEL_ROW_THRESHOLD: usize = 512;
const PREDICT_GPU_THRESHOLD: usize = 2_000_000;
const LDA_TILE: usize = 64;
const QDA_CHUNK_ROWS: usize = 64;

#[inline]
fn effective_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < 524_288 {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn effective_predict_backend<'a>(dev: &'a DeviceType, elems: usize) -> Backend<'a> {
    if elems < PREDICT_GPU_THRESHOLD {
        Backend::Cpu
    } else {
        get_backend(dev)
    }
}

#[inline]
fn compute_predict_chunk_rows(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (n / (threads * 2)).clamp(256, 2048).next_multiple_of(4)
}

#[inline(always)]
fn slice_is_finite(data: &[f64]) -> bool {
    let (chunks, rem) = data.as_chunks::<8>();
    for c in chunks {
        let ok = c[0].is_finite()
            & c[1].is_finite()
            & c[2].is_finite()
            & c[3].is_finite()
            & c[4].is_finite()
            & c[5].is_finite()
            & c[6].is_finite()
            & c[7].is_finite();
        if !ok {
            return false;
        }
    }
    rem.iter().all(|v| v.is_finite())
}

fn pack_coef_4(coef_data: &[f64], k: usize, p: usize) -> Vec<f64> {
    let n_blocks = k / 4;
    let mut packed = vec![0.0; n_blocks * p * 4];
    for blk in 0..n_blocks {
        let c_base = blk * 4;
        let c0 = &coef_data[c_base * p..(c_base + 1) * p];
        let c1 = &coef_data[(c_base + 1) * p..(c_base + 2) * p];
        let c2 = &coef_data[(c_base + 2) * p..(c_base + 3) * p];
        let c3 = &coef_data[(c_base + 3) * p..(c_base + 4) * p];
        let dst = &mut packed[blk * p * 4..(blk + 1) * p * 4];
        for f in 0..p {
            let off = f * 4;
            dst[off] = c0[f];
            dst[off + 1] = c1[f];
            dst[off + 2] = c2[f];
            dst[off + 3] = c3[f];
        }
    }
    packed
}

fn cpu_lda_linear_1d_chunk(
    x_chunk: &[f64],
    p: usize,
    w: &[f64],
    b: f64,
    out_chunk: &mut [f64],
) -> bool {
    let n_rows = out_chunk.len();
    let mut r = 0usize;
    while r + 4 <= n_rows {
        if !slice_is_finite(&x_chunk[r * p..(r + 4) * p]) {
            return false;
        }
        let (d0, d1, d2, d3) = dot4_4rows(
            &x_chunk[r * p..(r + 1) * p],
            &x_chunk[(r + 1) * p..(r + 2) * p],
            &x_chunk[(r + 2) * p..(r + 3) * p],
            &x_chunk[(r + 3) * p..(r + 4) * p],
            w,
        );
        out_chunk[r] = d0 + b;
        out_chunk[r + 1] = d1 + b;
        out_chunk[r + 2] = d2 + b;
        out_chunk[r + 3] = d3 + b;
        r += 4;
    }
    while r < n_rows {
        let x_row = &x_chunk[r * p..(r + 1) * p];
        if !slice_is_finite(x_row) {
            return false;
        }
        out_chunk[r] = dot4(x_row, w) + b;
        r += 1;
    }
    true
}

fn cpu_lda_linear_1d(x: &Matrix, w: &[f64], b: f64) -> OchreResult<Vec<f64>> {
    let n = x.rows;
    let p = x.cols;
    let mut out = vec![0.0; n];
    if n >= PARALLEL_ROW_THRESHOLD {
        let chunk_rows = compute_predict_chunk_rows(n);
        out.par_chunks_mut(chunk_rows)
            .zip(x.data.par_chunks(chunk_rows * p))
            .try_for_each(|(out_chunk, x_chunk)| -> OchreResult<()> {
                if !cpu_lda_linear_1d_chunk(x_chunk, p, w, b, out_chunk) {
                    return Err(OchreError::InvalidParameter(
                        "Input X contains NaN or Inf values ( >_< )".to_string(),
                    ));
                }
                Ok(())
            })?;
    } else if !cpu_lda_linear_1d_chunk(&x.data, p, w, b, &mut out) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn cpu_lda_linear_multi_chunk(
    x_chunk: &[f64],
    n_rows: usize,
    p: usize,
    coef_packed: &[f64],
    coef_data: &[f64],
    intercept: &[f64],
    k: usize,
    out_chunk: &mut [f64],
) -> bool {
    let n_blocks = k / 4;
    let k_rem_start = n_blocks * 4;
    let mut r = 0usize;

    while r + 4 <= n_rows {
        if !slice_is_finite(&x_chunk[r * p..(r + 4) * p]) {
            return false;
        }
        let x0 = &x_chunk[r * p..(r + 1) * p];
        let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
        let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
        let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
        let o0_off = r * k;
        let o1_off = (r + 1) * k;
        let o2_off = (r + 2) * k;
        let o3_off = (r + 3) * k;

        for blk in 0..n_blocks {
            let c = blk * 4;
            let c_blk = &coef_packed[blk * p * 4..(blk + 1) * p * 4];
            let mut acc0 = [0.0f64; 4];
            let mut acc1 = [0.0f64; 4];
            let mut acc2 = [0.0f64; 4];
            let mut acc3 = [0.0f64; 4];
            for (f, cv) in c_blk.as_chunks::<4>().0.iter().enumerate() {
                let xv0 = x0[f];
                let xv1 = x1[f];
                let xv2 = x2[f];
                let xv3 = x3[f];
                for lane in 0..4 {
                    let cval = cv[lane];
                    acc0[lane] += xv0 * cval;
                    acc1[lane] += xv1 * cval;
                    acc2[lane] += xv2 * cval;
                    acc3[lane] += xv3 * cval;
                }
            }
            let b = &intercept[c..c + 4];
            for lane in 0..4 {
                acc0[lane] += b[lane];
                acc1[lane] += b[lane];
                acc2[lane] += b[lane];
                acc3[lane] += b[lane];
            }
            out_chunk[o0_off + c..o0_off + c + 4].copy_from_slice(&acc0);
            out_chunk[o1_off + c..o1_off + c + 4].copy_from_slice(&acc1);
            out_chunk[o2_off + c..o2_off + c + 4].copy_from_slice(&acc2);
            out_chunk[o3_off + c..o3_off + c + 4].copy_from_slice(&acc3);
        }

        for c in k_rem_start..k {
            let (d0, d1, d2, d3) = dot4_4rows(x0, x1, x2, x3, &coef_data[c * p..(c + 1) * p]);
            let bc = intercept[c];
            out_chunk[o0_off + c] = d0 + bc;
            out_chunk[o1_off + c] = d1 + bc;
            out_chunk[o2_off + c] = d2 + bc;
            out_chunk[o3_off + c] = d3 + bc;
        }
        r += 4;
    }

    while r < n_rows {
        let x_row = &x_chunk[r * p..(r + 1) * p];
        if !slice_is_finite(x_row) {
            return false;
        }
        let dst = &mut out_chunk[r * k..(r + 1) * k];
        for c in 0..k {
            dst[c] = dot4(x_row, &coef_data[c * p..(c + 1) * p]) + intercept[c];
        }
        r += 1;
    }
    true
}

fn cpu_lda_linear_multi(
    x: &Matrix,
    coef_packed: &[f64],
    coef: &Matrix,
    intercept: &[f64],
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let k = coef.rows;
    let coef_data = &coef.data;
    let mut out = vec![0.0; n * k];
    if n >= PARALLEL_ROW_THRESHOLD {
        let chunk_rows = compute_predict_chunk_rows(n);
        out.par_chunks_mut(chunk_rows * k)
            .zip(x.data.par_chunks(chunk_rows * p))
            .try_for_each(|(out_chunk, x_chunk)| -> OchreResult<()> {
                let n_rows = out_chunk.len() / k;
                if !cpu_lda_linear_multi_chunk(
                    x_chunk,
                    n_rows,
                    p,
                    coef_packed,
                    coef_data,
                    intercept,
                    k,
                    out_chunk,
                ) {
                    return Err(OchreError::InvalidParameter(
                        "Input X contains NaN or Inf values ( >_< )".to_string(),
                    ));
                }
                Ok(())
            })?;
    } else if !cpu_lda_linear_multi_chunk(
        &x.data,
        n,
        p,
        coef_packed,
        coef_data,
        intercept,
        k,
        &mut out,
    ) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Matrix::from_vec(n, k, out)
}

fn cpu_lda_predict_binary_chunk(
    x_chunk: &[f64],
    p: usize,
    w: &[f64],
    b: f64,
    c0: f64,
    c1: f64,
    out_chunk: &mut [f64],
) -> bool {
    let n_rows = out_chunk.len();
    let mut r = 0usize;
    while r + 4 <= n_rows {
        if !slice_is_finite(&x_chunk[r * p..(r + 4) * p]) {
            return false;
        }
        let (d0, d1, d2, d3) = dot4_4rows(
            &x_chunk[r * p..(r + 1) * p],
            &x_chunk[(r + 1) * p..(r + 2) * p],
            &x_chunk[(r + 2) * p..(r + 3) * p],
            &x_chunk[(r + 3) * p..(r + 4) * p],
            w,
        );
        out_chunk[r] = if d0 + b > 0.0 { c1 } else { c0 };
        out_chunk[r + 1] = if d1 + b > 0.0 { c1 } else { c0 };
        out_chunk[r + 2] = if d2 + b > 0.0 { c1 } else { c0 };
        out_chunk[r + 3] = if d3 + b > 0.0 { c1 } else { c0 };
        r += 4;
    }
    while r < n_rows {
        let x_row = &x_chunk[r * p..(r + 1) * p];
        if !slice_is_finite(x_row) {
            return false;
        }
        out_chunk[r] = if dot4(x_row, w) + b > 0.0 { c1 } else { c0 };
        r += 1;
    }
    true
}

fn cpu_lda_predict_binary(
    x: &Matrix,
    w: &[f64],
    b: f64,
    c0: f64,
    c1: f64,
) -> OchreResult<Vec<f64>> {
    let n = x.rows;
    let p = x.cols;
    let mut out = vec![0.0; n];
    if n >= PARALLEL_ROW_THRESHOLD {
        let chunk_rows = compute_predict_chunk_rows(n);
        out.par_chunks_mut(chunk_rows)
            .zip(x.data.par_chunks(chunk_rows * p))
            .try_for_each(|(out_chunk, x_chunk)| -> OchreResult<()> {
                if !cpu_lda_predict_binary_chunk(x_chunk, p, w, b, c0, c1, out_chunk) {
                    return Err(OchreError::InvalidParameter(
                        "Input X contains NaN or Inf values ( >_< )".to_string(),
                    ));
                }
                Ok(())
            })?;
    } else if !cpu_lda_predict_binary_chunk(&x.data, p, w, b, c0, c1, &mut out) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn cpu_lda_predict_multi_chunk(
    x_chunk: &[f64],
    p: usize,
    coef_packed: &[f64],
    coef_data: &[f64],
    intercept: &[f64],
    classes: &[f64],
    k: usize,
    preds_chunk: &mut [f64],
) -> bool {
    let n_rows = preds_chunk.len();
    let n_blocks = k / 4;
    let k_rem_start = n_blocks * 4;
    let k_rem = k - k_rem_start;
    let mut r = 0usize;

    while r + 4 <= n_rows {
        if !slice_is_finite(&x_chunk[r * p..(r + 4) * p]) {
            return false;
        }
        let x0 = &x_chunk[r * p..(r + 1) * p];
        let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
        let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
        let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];

        let mut best_c = [0usize; 4];
        let mut best_v = [f64::NEG_INFINITY; 4];

        for blk in 0..n_blocks {
            let c = blk * 4;
            let c_blk = &coef_packed[blk * p * 4..(blk + 1) * p * 4];
            let mut acc0 = [0.0f64; 4];
            let mut acc1 = [0.0f64; 4];
            let mut acc2 = [0.0f64; 4];
            let mut acc3 = [0.0f64; 4];
            for (f, cv) in c_blk.as_chunks::<4>().0.iter().enumerate() {
                let xv0 = x0[f];
                let xv1 = x1[f];
                let xv2 = x2[f];
                let xv3 = x3[f];
                for lane in 0..4 {
                    let cval = cv[lane];
                    acc0[lane] += xv0 * cval;
                    acc1[lane] += xv1 * cval;
                    acc2[lane] += xv2 * cval;
                    acc3[lane] += xv3 * cval;
                }
            }
            let b = &intercept[c..c + 4];
            for lane in 0..4 {
                let v0 = acc0[lane] + b[lane];
                if v0 > best_v[0] {
                    best_v[0] = v0;
                    best_c[0] = c + lane;
                }
                let v1 = acc1[lane] + b[lane];
                if v1 > best_v[1] {
                    best_v[1] = v1;
                    best_c[1] = c + lane;
                }
                let v2 = acc2[lane] + b[lane];
                if v2 > best_v[2] {
                    best_v[2] = v2;
                    best_c[2] = c + lane;
                }
                let v3 = acc3[lane] + b[lane];
                if v3 > best_v[3] {
                    best_v[3] = v3;
                    best_c[3] = c + lane;
                }
            }
        }

        if k_rem >= 2 {
            let c_a = k_rem_start;
            let ca_row = &coef_data[c_a * p..(c_a + 1) * p];
            let cb_row = &coef_data[(c_a + 1) * p..(c_a + 2) * p];
            let mut a0 = 0.0;
            let mut a1 = 0.0;
            let mut a2 = 0.0;
            let mut a3 = 0.0;
            let mut b0 = 0.0;
            let mut b1 = 0.0;
            let mut b2 = 0.0;
            let mut b3 = 0.0;
            if k_rem == 3 {
                let cc_row = &coef_data[(c_a + 2) * p..(c_a + 3) * p];
                let mut c0 = 0.0;
                let mut c1 = 0.0;
                let mut c2 = 0.0;
                let mut c3 = 0.0;
                for f in 0..p {
                    let xv0 = x0[f];
                    let xv1 = x1[f];
                    let xv2 = x2[f];
                    let xv3 = x3[f];
                    let wa = ca_row[f];
                    let wb = cb_row[f];
                    let wc = cc_row[f];
                    a0 += xv0 * wa;
                    a1 += xv1 * wa;
                    a2 += xv2 * wa;
                    a3 += xv3 * wa;
                    b0 += xv0 * wb;
                    b1 += xv1 * wb;
                    b2 += xv2 * wb;
                    b3 += xv3 * wb;
                    c0 += xv0 * wc;
                    c1 += xv1 * wc;
                    c2 += xv2 * wc;
                    c3 += xv3 * wc;
                }
                let vals = [
                    [
                        a0 + intercept[c_a],
                        b0 + intercept[c_a + 1],
                        c0 + intercept[c_a + 2],
                    ],
                    [
                        a1 + intercept[c_a],
                        b1 + intercept[c_a + 1],
                        c1 + intercept[c_a + 2],
                    ],
                    [
                        a2 + intercept[c_a],
                        b2 + intercept[c_a + 1],
                        c2 + intercept[c_a + 2],
                    ],
                    [
                        a3 + intercept[c_a],
                        b3 + intercept[c_a + 1],
                        c3 + intercept[c_a + 2],
                    ],
                ];
                for lane in 0..3 {
                    for row in 0..4 {
                        if vals[row][lane] > best_v[row] {
                            best_v[row] = vals[row][lane];
                            best_c[row] = c_a + lane;
                        }
                    }
                }
            } else {
                for f in 0..p {
                    let xv0 = x0[f];
                    let xv1 = x1[f];
                    let xv2 = x2[f];
                    let xv3 = x3[f];
                    let wa = ca_row[f];
                    let wb = cb_row[f];
                    a0 += xv0 * wa;
                    a1 += xv1 * wa;
                    a2 += xv2 * wa;
                    a3 += xv3 * wa;
                    b0 += xv0 * wb;
                    b1 += xv1 * wb;
                    b2 += xv2 * wb;
                    b3 += xv3 * wb;
                }
                let vals = [
                    [a0 + intercept[c_a], b0 + intercept[c_a + 1]],
                    [a1 + intercept[c_a], b1 + intercept[c_a + 1]],
                    [a2 + intercept[c_a], b2 + intercept[c_a + 1]],
                    [a3 + intercept[c_a], b3 + intercept[c_a + 1]],
                ];
                for lane in 0..2 {
                    for row in 0..4 {
                        if vals[row][lane] > best_v[row] {
                            best_v[row] = vals[row][lane];
                            best_c[row] = c_a + lane;
                        }
                    }
                }
            }
        } else if k_rem == 1 {
            let c = k_rem_start;
            let (d0, d1, d2, d3) = dot4_4rows(x0, x1, x2, x3, &coef_data[c * p..(c + 1) * p]);
            let bc = intercept[c];
            let vals = [d0 + bc, d1 + bc, d2 + bc, d3 + bc];
            for row in 0..4 {
                if vals[row] > best_v[row] {
                    best_v[row] = vals[row];
                    best_c[row] = c;
                }
            }
        }

        preds_chunk[r] = classes[best_c[0]];
        preds_chunk[r + 1] = classes[best_c[1]];
        preds_chunk[r + 2] = classes[best_c[2]];
        preds_chunk[r + 3] = classes[best_c[3]];
        r += 4;
    }

    while r < n_rows {
        let x_row = &x_chunk[r * p..(r + 1) * p];
        if !slice_is_finite(x_row) {
            return false;
        }
        let mut best_c = 0usize;
        let mut best_v = dot4(x_row, &coef_data[..p]) + intercept[0];
        for c in 1..k {
            let v = dot4(x_row, &coef_data[c * p..(c + 1) * p]) + intercept[c];
            if v > best_v {
                best_v = v;
                best_c = c;
            }
        }
        preds_chunk[r] = classes[best_c];
        r += 1;
    }
    true
}

fn cpu_lda_predict_multi(
    x: &Matrix,
    coef_packed: &[f64],
    coef: &Matrix,
    intercept: &[f64],
    classes: &[f64],
) -> OchreResult<Vec<f64>> {
    let n = x.rows;
    let p = x.cols;
    let k = classes.len();
    let coef_data = &coef.data;
    let mut preds = vec![0.0; n];
    if n >= PARALLEL_ROW_THRESHOLD {
        let chunk_rows = compute_predict_chunk_rows(n);
        preds
            .par_chunks_mut(chunk_rows)
            .zip(x.data.par_chunks(chunk_rows * p))
            .try_for_each(|(preds_chunk, x_chunk)| -> OchreResult<()> {
                if !cpu_lda_predict_multi_chunk(
                    x_chunk,
                    p,
                    coef_packed,
                    coef_data,
                    intercept,
                    classes,
                    k,
                    preds_chunk,
                ) {
                    return Err(OchreError::InvalidParameter(
                        "Input X contains NaN or Inf values ( >_< )".to_string(),
                    ));
                }
                Ok(())
            })?;
    } else if !cpu_lda_predict_multi_chunk(
        &x.data,
        p,
        coef_packed,
        coef_data,
        intercept,
        classes,
        k,
        &mut preds,
    ) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(preds)
}

#[inline]
fn dot4(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[inline(always)]
fn dot4_4rows(a0: &[f64], a1: &[f64], a2: &[f64], a3: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let mut s0 = [0.0f64; 4];
    let mut s1 = [0.0f64; 4];
    let mut s2 = [0.0f64; 4];
    let mut s3 = [0.0f64; 4];
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    let (a0_chunks, a0_rem) = a0.as_chunks::<4>();
    let (a1_chunks, a1_rem) = a1.as_chunks::<4>();
    let (a2_chunks, a2_rem) = a2.as_chunks::<4>();
    let (a3_chunks, a3_rem) = a3.as_chunks::<4>();
    for ((((&bc, &c0), &c1), &c2), &c3) in b_chunks
        .iter()
        .zip(a0_chunks.iter())
        .zip(a1_chunks.iter())
        .zip(a2_chunks.iter())
        .zip(a3_chunks.iter())
    {
        for l in 0..4 {
            let bv = bc[l];
            s0[l] += c0[l] * bv;
            s1[l] += c1[l] * bv;
            s2[l] += c2[l] * bv;
            s3[l] += c3[l] * bv;
        }
    }
    let mut r0 = (s0[0] + s0[1]) + (s0[2] + s0[3]);
    let mut r1 = (s1[0] + s1[1]) + (s1[2] + s1[3]);
    let mut r2 = (s2[0] + s2[1]) + (s2[2] + s2[3]);
    let mut r3 = (s3[0] + s3[1]) + (s3[2] + s3[3]);
    for j in 0..b_rem.len() {
        let bv = b_rem[j];
        r0 += a0_rem[j] * bv;
        r1 += a1_rem[j] * bv;
        r2 += a2_rem[j] * bv;
        r3 += a3_rem[j] * bv;
    }
    (r0, r1, r2, r3)
}

#[inline(always)]
fn dot64_tile(a: &[f64; LDA_TILE], b: &[f64; LDA_TILE]) -> f64 {
    let mut acc = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            acc[lane] += a[base + lane] * b[base + lane];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

#[inline(always)]
fn dot64_tile_1x4(
    a: &[f64; LDA_TILE],
    b0: &[f64; LDA_TILE],
    b1: &[f64; LDA_TILE],
    b2: &[f64; LDA_TILE],
    b3: &[f64; LDA_TILE],
) -> (f64, f64, f64, f64) {
    let mut acc0 = [0.0f64; 4];
    let mut acc1 = [0.0f64; 4];
    let mut acc2 = [0.0f64; 4];
    let mut acc3 = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            let ai = a[base + lane];
            acc0[lane] += ai * b0[base + lane];
            acc1[lane] += ai * b1[base + lane];
            acc2[lane] += ai * b2[base + lane];
            acc3[lane] += ai * b3[base + lane];
        }
    }
    (
        (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]),
        (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]),
        (acc2[0] + acc2[1]) + (acc2[2] + acc2[3]),
        (acc3[0] + acc3[1]) + (acc3[2] + acc3[3]),
    )
}

type Tile2x4Result = ((f64, f64, f64, f64), (f64, f64, f64, f64));

#[inline(always)]
fn dot64_tile_2x4(
    a0: &[f64; LDA_TILE],
    a1: &[f64; LDA_TILE],
    b0: &[f64; LDA_TILE],
    b1: &[f64; LDA_TILE],
    b2: &[f64; LDA_TILE],
    b3: &[f64; LDA_TILE],
) -> Tile2x4Result {
    let mut acc00 = [0.0f64; 4];
    let mut acc01 = [0.0f64; 4];
    let mut acc02 = [0.0f64; 4];
    let mut acc03 = [0.0f64; 4];
    let mut acc10 = [0.0f64; 4];
    let mut acc11 = [0.0f64; 4];
    let mut acc12 = [0.0f64; 4];
    let mut acc13 = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            let v_a0 = a0[base + lane];
            let v_a1 = a1[base + lane];
            let v_b0 = b0[base + lane];
            let v_b1 = b1[base + lane];
            let v_b2 = b2[base + lane];
            let v_b3 = b3[base + lane];
            acc00[lane] += v_a0 * v_b0;
            acc01[lane] += v_a0 * v_b1;
            acc02[lane] += v_a0 * v_b2;
            acc03[lane] += v_a0 * v_b3;
            acc10[lane] += v_a1 * v_b0;
            acc11[lane] += v_a1 * v_b1;
            acc12[lane] += v_a1 * v_b2;
            acc13[lane] += v_a1 * v_b3;
        }
    }
    (
        (
            (acc00[0] + acc00[1]) + (acc00[2] + acc00[3]),
            (acc01[0] + acc01[1]) + (acc01[2] + acc01[3]),
            (acc02[0] + acc02[1]) + (acc02[2] + acc02[3]),
            (acc03[0] + acc03[1]) + (acc03[2] + acc03[3]),
        ),
        (
            (acc10[0] + acc10[1]) + (acc10[2] + acc10[3]),
            (acc11[0] + acc11[1]) + (acc11[2] + acc11[3]),
            (acc12[0] + acc12[1]) + (acc12[2] + acc12[3]),
            (acc13[0] + acc13[1]) + (acc13[2] + acc13[3]),
        ),
    )
}

fn accumulate_centered_gram_chunk(
    x_chunk: &[f64],
    y_chunk: &[usize],
    means_data: &[f64],
    p: usize,
    x_t: &mut [[f64; LDA_TILE]],
    gram: &mut [f64],
) {
    let n_chunk = y_chunk.len();
    let mut r_block = 0usize;
    while r_block < n_chunk {
        let block_end = (r_block + LDA_TILE).min(n_chunk);
        let b_len = block_end - r_block;
        if b_len < LDA_TILE {
            for col in x_t.iter_mut() {
                col[b_len..LDA_TILE].fill(0.0);
            }
        }

        let mut b = 0usize;
        while b + 4 <= b_len {
            let r0 = r_block + b;
            let row0 = &x_chunk[r0 * p..(r0 + 1) * p];
            let row1 = &x_chunk[(r0 + 1) * p..(r0 + 2) * p];
            let row2 = &x_chunk[(r0 + 2) * p..(r0 + 3) * p];
            let row3 = &x_chunk[(r0 + 3) * p..(r0 + 4) * p];
            let c0 = y_chunk[r0];
            let c1 = y_chunk[r0 + 1];
            let c2 = y_chunk[r0 + 2];
            let c3 = y_chunk[r0 + 3];
            let m0 = &means_data[c0 * p..(c0 + 1) * p];
            let m1 = &means_data[c1 * p..(c1 + 1) * p];
            let m2 = &means_data[c2 * p..(c2 + 1) * p];
            let m3 = &means_data[c3 * p..(c3 + 1) * p];
            for (j, col) in x_t.iter_mut().enumerate().take(p) {
                col[b] = row0[j] - m0[j];
                col[b + 1] = row1[j] - m1[j];
                col[b + 2] = row2[j] - m2[j];
                col[b + 3] = row3[j] - m3[j];
            }
            b += 4;
        }
        while b < b_len {
            let r0 = r_block + b;
            let row = &x_chunk[r0 * p..(r0 + 1) * p];
            let c0 = y_chunk[r0];
            let m0 = &means_data[c0 * p..(c0 + 1) * p];
            for (j, col) in x_t.iter_mut().enumerate().take(p) {
                col[b] = row[j] - m0[j];
            }
            b += 1;
        }

        let mut j = 0usize;
        while j + 2 <= p {
            let lj0 = &x_t[j];
            let lj1 = &x_t[j + 1];
            let off0 = j * p;
            let off1 = (j + 1) * p;
            gram[off0 + j] += dot64_tile(lj0, lj0);
            gram[off0 + j + 1] += dot64_tile(lj0, lj1);
            gram[off1 + j + 1] += dot64_tile(lj1, lj1);

            let mut k = j + 2;
            while k + 4 <= p {
                let ((s00, s01, s02, s03), (s10, s11, s12, s13)) =
                    dot64_tile_2x4(lj0, lj1, &x_t[k], &x_t[k + 1], &x_t[k + 2], &x_t[k + 3]);
                gram[off0 + k] += s00;
                gram[off0 + k + 1] += s01;
                gram[off0 + k + 2] += s02;
                gram[off0 + k + 3] += s03;
                gram[off1 + k] += s10;
                gram[off1 + k + 1] += s11;
                gram[off1 + k + 2] += s12;
                gram[off1 + k + 3] += s13;
                k += 4;
            }
            while k < p {
                gram[off0 + k] += dot64_tile(lj0, &x_t[k]);
                gram[off1 + k] += dot64_tile(lj1, &x_t[k]);
                k += 1;
            }
            j += 2;
        }
        while j < p {
            let lj = &x_t[j];
            let off = j * p;
            let mut k = j;
            while k + 4 <= p {
                let (s0, s1, s2, s3) =
                    dot64_tile_1x4(lj, &x_t[k], &x_t[k + 1], &x_t[k + 2], &x_t[k + 3]);
                gram[off + k] += s0;
                gram[off + k + 1] += s1;
                gram[off + k + 2] += s2;
                gram[off + k + 3] += s3;
                k += 4;
            }
            while k < p {
                gram[off + k] += dot64_tile(lj, &x_t[k]);
                k += 1;
            }
            j += 1;
        }

        r_block = block_end;
    }
}

fn compute_centered_gram_cpu(x: &Matrix, y_idx: &[usize], means: &Matrix) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let means_data = &means.data;
    let gram_data = if n >= PARALLEL_ROW_THRESHOLD {
        const CHUNK_ROWS: usize = 1024;
        x.data
            .par_chunks(CHUNK_ROWS * p)
            .zip(y_idx.par_chunks(CHUNK_ROWS))
            .fold(
                || (vec![0.0; p * p], vec![[0.0f64; LDA_TILE]; p]),
                |(mut local_gram, mut x_t), (x_chunk, y_chunk)| {
                    accumulate_centered_gram_chunk(
                        x_chunk,
                        y_chunk,
                        means_data,
                        p,
                        &mut x_t,
                        &mut local_gram,
                    );
                    (local_gram, x_t)
                },
            )
            .map(|(local_gram, _)| local_gram)
            .reduce(
                || vec![0.0; p * p],
                |mut a, b| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        let mut gram = vec![0.0; p * p];
        let mut x_t = vec![[0.0f64; LDA_TILE]; p];
        accumulate_centered_gram_chunk(&x.data, y_idx, means_data, p, &mut x_t, &mut gram);
        gram
    };

    let mut gram_mat = Matrix::from_vec(p, p, gram_data)?;
    for r in 0..p {
        for c in (r + 1)..p {
            let v = gram_mat.data[r * p + c];
            gram_mat.data[c * p + r] = v;
        }
    }
    Ok(gram_mat)
}

fn validate_matrix_finite(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_y_finite(y: &[f64], expected_rows: usize) -> OchreResult<()> {
    if y.len() != expected_rows {
        return Err(OchreError::LengthMismatch {
            features: expected_rows,
            targets: y.len(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Target y contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_tol(tol: f64) -> OchreResult<()> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "tol ({tol}) must be non-negative and finite (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_priors(priors: &[f64], n_classes: usize) -> OchreResult<Vec<f64>> {
    if priors.len() != n_classes {
        return Err(OchreError::InvalidParameter(format!(
            "Number of priors ({}) must match number of classes ({n_classes}) ( >_< )",
            priors.len()
        )));
    }
    let mut sum = 0.0;
    for &p in priors {
        if !p.is_finite() || p < 0.0 {
            return Err(OchreError::InvalidParameter(
                "priors must be non-negative finite numbers (・`ω´・)".to_string(),
            ));
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-6 {
        return Err(OchreError::InvalidParameter(format!(
            "priors must sum to 1.0, got {sum} ( >_< )"
        )));
    }
    Ok(priors.iter().map(|&p| p / sum).collect())
}

type ClassStatsResult = (Vec<usize>, Vec<f64>, Matrix, Vec<f64>);

fn compute_class_stats(
    x: &Matrix,
    y_idx: &[usize],
    n_classes: usize,
    priors_param: Option<&[f64]>,
) -> OchreResult<ClassStatsResult> {
    let n_samples = x.rows;
    let n_features = x.cols;
    let mut counts = vec![0usize; n_classes];
    let mut means_data = vec![0.0; n_classes * n_features];

    for (i, &cls) in y_idx.iter().enumerate() {
        counts[cls] += 1;
        let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
        let m_row = &mut means_data[(cls * n_features)..((cls + 1) * n_features)];
        for j in 0..n_features {
            m_row[j] += x_row[j];
        }
    }

    for k in 0..n_classes {
        if counts[k] == 0 {
            return Err(OchreError::InvalidParameter(format!(
                "Class index {k} has zero samples ( >_< )"
            )));
        }
        let inv = 1.0 / (counts[k] as f64);
        let m_row = &mut means_data[(k * n_features)..((k + 1) * n_features)];
        for val in m_row.iter_mut() {
            *val *= inv;
        }
    }

    let priors = match priors_param {
        Some(p) => validate_priors(p, n_classes)?,
        None => counts
            .iter()
            .map(|&c| (c as f64) / (n_samples as f64))
            .collect(),
    };

    let mut xbar = vec![0.0; n_features];
    for k in 0..n_classes {
        let pk = priors[k];
        let m_row = &means_data[(k * n_features)..((k + 1) * n_features)];
        for j in 0..n_features {
            xbar[j] += pk * m_row[j];
        }
    }

    let means = Matrix::from_vec(n_classes, n_features, means_data)?;
    Ok((counts, priors, means, xbar))
}

fn ledoit_wolf_shrinkage(xc: &Matrix, s_emp: &Matrix) -> f64 {
    let n = xc.rows as f64;
    let p = xc.cols;
    if p == 0 || xc.rows == 0 {
        return 0.0;
    }
    let p_f = p as f64;
    let mut trace = 0.0;
    for i in 0..p {
        trace += s_emp.data[i * p + i];
    }
    let mu = trace / p_f;

    let mut delta = 0.0;
    let mut frob_sq_s = 0.0;
    for r in 0..p {
        for c in 0..p {
            let v = s_emp.data[r * p + c];
            frob_sq_s += v * v;
            let diff = if r == c { v - mu } else { v };
            delta += diff * diff;
        }
    }
    delta /= p_f;
    if delta <= 1e-30 {
        return 0.0;
    }

    let sum_norm4: f64 = if xc.rows >= PARALLEL_ROW_THRESHOLD {
        xc.data
            .par_chunks_exact(p)
            .map(|row| {
                let sq: f64 = row.iter().map(|&v| v * v).sum();
                sq * sq
            })
            .sum()
    } else {
        xc.data
            .chunks_exact(p)
            .map(|row| {
                let sq: f64 = row.iter().map(|&v| v * v).sum();
                sq * sq
            })
            .sum()
    };

    let beta_bar = (sum_norm4 / (n * n) - frob_sq_s / n) / p_f;
    let beta = beta_bar.max(0.0).min(delta);
    (beta / delta).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn compute_within_class_cov(
    backend: &Backend<'_>,
    x: &Matrix,
    y_idx: &[usize],
    means: &Matrix,
    priors: &[f64],
    counts: &[usize],
    shrinkage: Option<f64>,
    shrinkage_auto: bool,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let n_classes = means.rows;

    let mut xc_data = vec![0.0; n * p];
    for (i, &cls) in y_idx.iter().enumerate() {
        let x_row = &x.data[(i * p)..((i + 1) * p)];
        let m_row = &means.data[(cls * p)..((cls + 1) * p)];
        let out_row = &mut xc_data[(i * p)..((i + 1) * p)];
        for j in 0..p {
            out_row[j] = x_row[j] - m_row[j];
        }
    }
    let xc = Matrix::from_vec(n, p, xc_data)?;

    // Compute S_w = sum_k priors[k] * (1 / n_k) * X_k_c^T X_k_c
    let mut s_w = Matrix::zeros(p, p);
    for k in 0..n_classes {
        let nk = counts[k];
        if nk == 0 {
            continue;
        }
        let mut xk_data = Vec::with_capacity(nk * p);
        for (i, &cls) in y_idx.iter().enumerate() {
            if cls == k {
                xk_data.extend_from_slice(&xc.data[(i * p)..((i + 1) * p)]);
            }
        }
        let xk = Matrix::from_vec(nk, p, xk_data)?;
        let dummy_y = vec![0.0; nk];
        let (gram_k, _) = backend.compute_normal_equation(&xk, &dummy_y, false)?;
        let scale = priors[k] / (nk as f64);
        for idx in 0..(p * p) {
            s_w.data[idx] += scale * gram_k.data[idx];
        }
    }

    let shrink = if shrinkage_auto {
        let dummy_y = vec![0.0; n];
        let (gram_all, _) = backend.compute_normal_equation(&xc, &dummy_y, false)?;
        let mut s_emp = gram_all;
        let inv_n = 1.0 / (n as f64);
        for v in s_emp.data.iter_mut() {
            *v *= inv_n;
        }
        Some(ledoit_wolf_shrinkage(&xc, &s_emp))
    } else {
        shrinkage
    };

    if let Some(alpha) = shrink {
        let mut trace = 0.0;
        for i in 0..p {
            trace += s_w.data[i * p + i];
        }
        let mu = trace / (p as f64);
        let one_minus = 1.0 - alpha;
        for r in 0..p {
            for c in 0..p {
                let idx = r * p + c;
                s_w.data[idx] *= one_minus;
                if r == c {
                    s_w.data[idx] += alpha * mu;
                }
            }
        }
    }

    Ok(s_w)
}

fn log_softmax_matrix(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let k = scores.cols;
    let mut out = vec![0.0; n * k];
    let compute_row = |(dst, src): (&mut [f64], &[f64])| {
        let max_v = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = src.iter().map(|&v| (v - max_v).exp()).sum();
        let log_norm = max_v + sum_exp.ln();
        for j in 0..k {
            dst[j] = src[j] - log_norm;
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(k)
            .zip(scores.data.par_chunks_exact(k))
            .for_each(compute_row);
    } else {
        out.chunks_exact_mut(k)
            .zip(scores.data.chunks_exact(k))
            .for_each(compute_row);
    }
    Matrix::from_vec(n, k, out)
}

fn softmax_matrix(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let k = scores.cols;
    let mut out = vec![0.0; n * k];
    let compute_row = |(dst, src): (&mut [f64], &[f64])| {
        let max_v = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0;
        for j in 0..k {
            let e = (src[j] - max_v).exp();
            dst[j] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for v in dst.iter_mut() {
            *v *= inv_sum;
        }
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(k)
            .zip(scores.data.par_chunks_exact(k))
            .for_each(compute_row);
    } else {
        out.chunks_exact_mut(k)
            .zip(scores.data.chunks_exact(k))
            .for_each(compute_row);
    }
    Matrix::from_vec(n, k, out)
}

fn binary_log_proba_from_decision(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let mut out = vec![0.0; n * 2];
    let compute_row = |(dst, &z): (&mut [f64], &f64)| {
        // log P(y=1) = -softplus(-z), log P(y=0) = -softplus(z)
        let (lp0, lp1) = if z >= 0.0 {
            let e = (-z).exp();
            let sp = e.ln_1p();
            (-z - sp, -sp)
        } else {
            let e = z.exp();
            let sp = e.ln_1p();
            (-sp, z - sp)
        };
        dst[0] = lp0;
        dst[1] = lp1;
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(2)
            .zip(scores.data.par_iter())
            .for_each(compute_row);
    } else {
        for (dst, &z) in out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(scores.data.iter())
        {
            compute_row((dst, &z));
        }
    }
    Matrix::from_vec(n, 2, out)
}

fn binary_proba_from_decision(scores: &Matrix) -> OchreResult<Matrix> {
    let n = scores.rows;
    let mut out = vec![0.0; n * 2];
    let compute_row = |(dst, &z): (&mut [f64], &f64)| {
        let p1 = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        };
        dst[0] = 1.0 - p1;
        dst[1] = p1;
    };
    if n >= PARALLEL_ROW_THRESHOLD {
        out.par_chunks_exact_mut(2)
            .zip(scores.data.par_iter())
            .for_each(compute_row);
    } else {
        for (dst, &z) in out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(scores.data.iter())
        {
            compute_row((dst, &z));
        }
    }
    Matrix::from_vec(n, 2, out)
}

fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> OchreResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(OchreError::LengthMismatch {
            features: y_true.len(),
            targets: y_pred.len(),
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
    Ok((correct as f64) / (y_true.len() as f64))
}

/// Linear Discriminant Analysis (LDA) classifier & dimensionality reducer (＾▽＾)
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    pub solver: String,
    pub shrinkage: Option<f64>,
    pub shrinkage_auto: bool,
    pub priors: Option<Vec<f64>>,
    pub n_components: Option<usize>,
    pub tol: f64,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<f64>>,
    pub priors_: Option<Vec<f64>>,
    pub means_: Option<Matrix>,
    pub xbar_: Option<Vec<f64>>,
    pub coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub scalings_: Option<Matrix>,
    pub explained_variance_ratio_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub n_components_: Option<usize>,
    coef_packed_: Option<Vec<f64>>,
}

impl LinearDiscriminantAnalysis {
    pub fn new(
        solver: String,
        shrinkage: Option<f64>,
        shrinkage_auto: bool,
        priors: Option<Vec<f64>>,
        n_components: Option<usize>,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if solver != "svd" && solver != "lsqr" && solver != "eigen" {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid solver '{solver}'. Supported solvers are 'svd', 'lsqr', and 'eigen' (・`ω´・)"
            )));
        }
        if solver == "svd" && (shrinkage.is_some() || shrinkage_auto) {
            return Err(OchreError::InvalidParameter(
                "shrinkage is not supported with solver='svd'; use 'lsqr' or 'eigen' (・`ω´・)"
                    .to_string(),
            ));
        }
        if let Some(s) = shrinkage
            && (!s.is_finite() || !(0.0..=1.0).contains(&s))
        {
            return Err(OchreError::InvalidParameter(format!(
                "shrinkage ({s}) must be a finite float in [0.0, 1.0] (・`ω´・)"
            )));
        }
        if let Some(nc) = n_components
            && nc == 0
        {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        validate_tol(tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            solver,
            shrinkage,
            shrinkage_auto,
            priors,
            n_components,
            tol,
            device: dev,
            device_: None,
            classes_: None,
            priors_: None,
            means_: None,
            xbar_: None,
            coef_: None,
            intercept_: None,
            scalings_: None,
            explained_variance_ratio_: None,
            n_features_in_: None,
            n_components_: None,
            coef_packed_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.n_features_in_.is_some() || self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    fn reset_fitted(&mut self) {
        self.classes_ = None;
        self.priors_ = None;
        self.means_ = None;
        self.xbar_ = None;
        self.coef_ = None;
        self.intercept_ = None;
        self.scalings_ = None;
        self.explained_variance_ratio_ = None;
        self.n_features_in_ = None;
        self.n_components_ = None;
        self.coef_packed_ = None;
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.reset_fitted();
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;

        let n_samples = x.rows;
        let n_features = x.cols;
        let classes = extract_sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(OchreError::InsufficientClasses(n_classes));
        }
        if n_samples <= n_classes {
            return Err(OchreError::InvalidParameter(
                "The number of samples must be more than the number of classes ( >_< )".to_string(),
            ));
        }

        let max_components = (n_classes - 1).min(n_features);
        let resolved_n_comp = match self.n_components {
            Some(nc) => {
                if nc > max_components {
                    return Err(OchreError::InvalidParameter(format!(
                        "n_components ({nc}) cannot be larger than min(n_features, n_classes - 1) = {max_components} ( >_< )"
                    )));
                }
                nc
            }
            None => max_components,
        };

        let y_idx = map_labels_to_indices(y, &classes)?;
        let (counts, priors, means, xbar) =
            compute_class_stats(x, &y_idx, n_classes, self.priors.as_deref())?;

        let backend = effective_backend(&self.device, n_samples * n_features);

        // Centered class means: M_c = means - xbar (n_classes x n_features)
        let mut means_centered_data = vec![0.0; n_classes * n_features];
        for k in 0..n_classes {
            for j in 0..n_features {
                means_centered_data[k * n_features + j] = means.data[k * n_features + j] - xbar[j];
            }
        }
        let means_centered = Matrix::from_vec(n_classes, n_features, means_centered_data)?;

        let (coef_full, scalings_opt, evr_opt) = match self.solver.as_str() {
            "svd" => self.solve_svd(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                n_samples,
                n_features,
                n_classes,
            )?,
            "eigen" => self.solve_eigen(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                &counts,
                n_features,
                n_classes,
            )?,
            "lsqr" => self.solve_lsqr(
                &backend,
                x,
                &y_idx,
                &means,
                &means_centered,
                &priors,
                &counts,
                n_features,
                n_classes,
            )?,
            _ => unreachable!(),
        };

        // Compute intercepts: b_k = -0.5 * (mu_k + xbar)^T w_k + ln(priors[k])
        let mut intercept_full = vec![0.0; n_classes];
        for (k, b_out) in intercept_full.iter_mut().enumerate() {
            let mut dot = 0.0;
            for (j, &xb) in xbar.iter().enumerate() {
                let mu_plus_xbar = means.data[k * n_features + j] + xb;
                dot += mu_plus_xbar * coef_full.data[k * n_features + j];
            }
            let log_p = if priors[k] > 0.0 {
                priors[k].ln()
            } else {
                f64::NEG_INFINITY
            };
            *b_out = -0.5 * dot + log_p;
        }

        // Binary sklearn convention: coef_ is (1, n_features) = w_1 - w_0, intercept_ is (1,) = b_1 - b_0
        let (coef_stored, intercept_stored) = if n_classes == 2 {
            let mut w_bin = vec![0.0; n_features];
            for (j, dst) in w_bin.iter_mut().enumerate() {
                *dst = coef_full.data[n_features + j] - coef_full.data[j];
            }
            let b_bin = vec![intercept_full[1] - intercept_full[0]];
            (Matrix::from_vec(1, n_features, w_bin)?, b_bin)
        } else {
            (coef_full, intercept_full)
        };

        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
        self.means_ = Some(means);
        self.xbar_ = Some(xbar);
        self.coef_packed_ = if coef_stored.rows >= 4 {
            Some(pack_coef_4(&coef_stored.data, coef_stored.rows, n_features))
        } else {
            None
        };
        self.coef_ = Some(coef_stored);
        self.intercept_ = Some(intercept_stored);
        self.scalings_ = scalings_opt;
        self.explained_variance_ratio_ = evr_opt;
        self.n_features_in_ = Some(n_features);
        self.n_components_ = Some(resolved_n_comp);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_svd(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let mut gram = if matches!(backend, Backend::Cpu) {
            compute_centered_gram_cpu(x, y_idx, means)?
        } else {
            let mut xc_data = vec![0.0; n_samples * n_features];
            for (i, &cls) in y_idx.iter().enumerate() {
                let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
                let m_row = &means.data[(cls * n_features)..((cls + 1) * n_features)];
                let out_row = &mut xc_data[(i * n_features)..((i + 1) * n_features)];
                for j in 0..n_features {
                    out_row[j] = x_row[j] - m_row[j];
                }
            }
            let xc = Matrix::from_vec(n_samples, n_features, xc_data)?;
            let dummy_y = vec![0.0; n_samples];
            let (gram_gpu, _) = backend.compute_normal_equation(&xc, &dummy_y, false)?;
            gram_gpu
        };

        let inv_n = 1.0 / (n_samples as f64);
        let mut inv_stds = Vec::with_capacity(n_features);
        for r in 0..n_features {
            let v = (gram.data[r * n_features + r] * inv_n).max(0.0).sqrt();
            let std_r = if v == 0.0 { 1.0 } else { v };
            inv_stds.push(1.0 / std_r);
        }

        for (r, &inv_std_r) in inv_stds.iter().enumerate().take(n_features) {
            let row_scale = inv_n * inv_std_r;
            let row = &mut gram.data[r * n_features..(r + 1) * n_features];
            for (dst, &inv_std_c) in row.iter_mut().zip(inv_stds.iter()) {
                *dst *= row_scale * inv_std_c;
            }
        }

        let eig_w = backend.small_sym_eig_svd(&gram)?;
        let s_vals: Vec<f64> = eig_w
            .eigenvalues
            .iter()
            .map(|&lam| lam.max(0.0).sqrt())
            .collect();

        let rank = s_vals.iter().filter(|&&s| s > self.tol).count();
        if rank == 0 {
            return Err(OchreError::SingularMatrix(
                "All within-class singular values are below tol".to_string(),
            ));
        }

        // scalings_w has shape (n_features, rank)
        let mut scalings_w_data = vec![0.0; n_features * rank];
        for r in 0..n_features {
            let inv_std_r = inv_stds[r];
            for k in 0..rank {
                scalings_w_data[r * rank + k] =
                    (eig_w.eigenvectors.data[r * n_features + k] * inv_std_r) / s_vals[k];
            }
        }
        let scalings_w = Matrix::from_vec(n_features, rank, scalings_w_data)?;

        // Project centered class means onto whitened space once and reuse for coef_full:
        // X_b = diag(sqrt(n_samples * priors)) * means_centered * scalings_w  (n_classes x rank)
        let proj = backend.matmul(means_centered, &scalings_w)?;
        let mut xb_raw = proj.clone();
        for (k, &pk) in priors.iter().enumerate() {
            let w = (n_samples as f64 * pk).sqrt();
            for j in 0..rank {
                xb_raw.data[k * rank + j] *= w;
            }
        }
        let dummy_yb = vec![0.0; n_classes];
        let (gram_b, _) = backend.compute_normal_equation(&xb_raw, &dummy_yb, false)?;
        let eig_b = backend.small_sym_eig_svd(&gram_b)?;

        let n_comp_avail = rank.min(n_classes - 1);
        let scalings_full = backend.matmul(&scalings_w, &eig_b.eigenvectors)?;

        let total_ev: f64 = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| v.max(0.0))
            .sum();
        let evr: Vec<f64> = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| {
                if total_ev > 0.0 {
                    v.max(0.0) / total_ev
                } else {
                    0.0
                }
            })
            .collect();

        // coef_full = means_centered * scalings_w * scalings_w^T (n_classes x n_features)
        let scalings_w_t = scalings_w.transpose();
        let coef_full = backend.matmul(&proj, &scalings_w_t)?;

        Ok((coef_full, Some(scalings_full), Some(evr)))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_eigen(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        counts: &[usize],
        n_features: usize,
        n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let s_w = compute_within_class_cov(
            backend,
            x,
            y_idx,
            means,
            priors,
            counts,
            self.shrinkage,
            self.shrinkage_auto,
        )?;

        let eig_w = backend.small_sym_eig_svd(&s_w)?;
        let max_lam = eig_w.eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
        let thresh = self.tol * max_lam.max(1.0);
        let rank = eig_w
            .eigenvalues
            .iter()
            .filter(|&&lam| lam > thresh && lam > 0.0)
            .count();
        if rank == 0 {
            return Err(OchreError::SingularMatrix(
                "Within-class covariance matrix is zero or singular".to_string(),
            ));
        }

        // S_w^{-1/2} = V_r * diag(1 / sqrt(lam_r))  (n_features x rank)
        let mut whiten_data = vec![0.0; n_features * rank];
        for r in 0..n_features {
            for k in 0..rank {
                whiten_data[r * rank + k] =
                    eig_w.eigenvectors.data[r * n_features + k] / eig_w.eigenvalues[k].sqrt();
            }
        }
        let whiten = Matrix::from_vec(n_features, rank, whiten_data)?;

        // S_b = sum_k priors[k] * (mu_k - xbar)(mu_k - xbar)^T
        let mut sb_data = vec![0.0; n_features * n_features];
        for (k, &pk) in priors.iter().enumerate() {
            let row = &means_centered.data[(k * n_features)..((k + 1) * n_features)];
            for r in 0..n_features {
                for c in 0..n_features {
                    sb_data[r * n_features + c] += pk * row[r] * row[c];
                }
            }
        }
        let s_b = Matrix::from_vec(n_features, n_features, sb_data)?;

        // M = whiten^T * S_b * whiten (rank x rank)
        let sb_whiten = backend.matmul(&s_b, &whiten)?;
        let m_mat = backend.matmul_at_b(&whiten, &sb_whiten)?;
        let eig_b = backend.small_sym_eig_svd(&m_mat)?;

        let n_comp_avail = rank.min(n_classes - 1);
        let scalings_full = backend.matmul(&whiten, &eig_b.eigenvectors)?;

        let total_ev: f64 = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| v.max(0.0))
            .sum();
        let evr: Vec<f64> = eig_b
            .eigenvalues
            .iter()
            .take(n_comp_avail)
            .map(|&v| {
                if total_ev > 0.0 {
                    v.max(0.0) / total_ev
                } else {
                    0.0
                }
            })
            .collect();

        // coef_full = means_centered * whiten * whiten^T
        let proj = backend.matmul(means_centered, &whiten)?;
        let whiten_t = whiten.transpose();
        let coef_full = backend.matmul(&proj, &whiten_t)?;

        Ok((coef_full, Some(scalings_full), Some(evr)))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_lsqr(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        y_idx: &[usize],
        means: &Matrix,
        means_centered: &Matrix,
        priors: &[f64],
        counts: &[usize],
        n_features: usize,
        _n_classes: usize,
    ) -> OchreResult<(Matrix, Option<Matrix>, Option<Vec<f64>>)> {
        let s_w = compute_within_class_cov(
            backend,
            x,
            y_idx,
            means,
            priors,
            counts,
            self.shrinkage,
            self.shrinkage_auto,
        )?;

        let rhs = means_centered.transpose();
        let coef_full = match solve_cholesky_multi(&s_w, &rhs) {
            Ok(sol) => sol.transpose(),
            Err(_) => {
                // Fallback to eigendecomposition pseudo-inverse for rank-deficient S_w
                let eig = backend.small_sym_eig_svd(&s_w)?;
                let max_lam = eig.eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
                let thresh = self.tol * max_lam.max(1.0);
                let rank = eig
                    .eigenvalues
                    .iter()
                    .filter(|&&lam| lam > thresh && lam > 0.0)
                    .count();
                if rank == 0 {
                    return Err(OchreError::SingularMatrix(
                        "Within-class covariance matrix is zero or singular".to_string(),
                    ));
                }
                let mut whiten_data = vec![0.0; n_features * rank];
                for r in 0..n_features {
                    for k in 0..rank {
                        whiten_data[r * rank + k] =
                            eig.eigenvectors.data[r * n_features + k] / eig.eigenvalues[k].sqrt();
                    }
                }
                let whiten = Matrix::from_vec(n_features, rank, whiten_data)?;
                let proj = backend.matmul(means_centered, &whiten)?;
                let whiten_t = whiten.transpose();
                backend.matmul(&proj, &whiten_t)?
            }
        };

        Ok((coef_full, None, None))
    }

    fn validate_predict_input(&self, x: &Matrix) -> OchreResult<usize> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        Ok(expected)
    }

    fn validate_predict_shape(&self, x: &Matrix) -> OchreResult<usize> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        Ok(expected)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let p = self.validate_predict_shape(x)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = x.rows;
        let k = coef.rows;
        let backend = effective_predict_backend(&self.device, n.saturating_mul(p));

        if matches!(backend, Backend::Cpu) {
            if k == 1 {
                let out = cpu_lda_linear_1d(x, &coef.data, intercept[0])?;
                return Matrix::from_vec(n, 1, out);
            }
            let coef_packed = match self.coef_packed_.as_deref() {
                Some(cp) => Cow::Borrowed(cp),
                None => Cow::Owned(pack_coef_4(&coef.data, k, p)),
            };
            return cpu_lda_linear_multi(x, &coef_packed, coef, intercept);
        }

        validate_matrix_finite(x)?;
        let coef_t = coef.transpose();
        let mut scores = backend.matmul(x, &coef_t)?;
        for r in 0..scores.rows {
            for (c, &b) in intercept.iter().enumerate().take(k) {
                scores.data[r * k + c] += b;
            }
        }
        Ok(scores)
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let scores = self.decision_function(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            binary_log_proba_from_decision(&scores)
        } else {
            log_softmax_matrix(&scores)
        }
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let scores = self.decision_function(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            binary_proba_from_decision(&scores)
        } else {
            softmax_matrix(&scores)
        }
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let p = self.validate_predict_shape(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = x.rows;
        let backend = effective_predict_backend(&self.device, n.saturating_mul(p));

        if matches!(backend, Backend::Cpu) {
            if classes.len() == 2 {
                return cpu_lda_predict_binary(x, &coef.data, intercept[0], classes[0], classes[1]);
            }
            let coef_packed = match self.coef_packed_.as_deref() {
                Some(cp) => Cow::Borrowed(cp),
                None => Cow::Owned(pack_coef_4(&coef.data, classes.len(), p)),
            };
            return cpu_lda_predict_multi(x, &coef_packed, coef, intercept, classes);
        }

        let scores = self.decision_function(x)?;
        if classes.len() == 2 {
            Ok((0..n)
                .map(|i| {
                    if scores.data[i] > 0.0 {
                        classes[1]
                    } else {
                        classes[0]
                    }
                })
                .collect())
        } else {
            let k = classes.len();
            let mut preds = Vec::with_capacity(n);
            for i in 0..n {
                let row = &scores.data[(i * k)..((i + 1) * k)];
                let mut best_c = 0;
                let mut best_v = row[0];
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best_v {
                        best_v = v;
                        best_c = c;
                    }
                }
                preds.push(classes[best_c]);
            }
            Ok(preds)
        }
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.validate_predict_input(x)?;
        if self.solver == "lsqr" {
            return Err(OchreError::InvalidParameter(
                "transform is not implemented for solver='lsqr'; use 'svd' or 'eigen' (・`ω´・)"
                    .to_string(),
            ));
        }
        let scalings = self.scalings_.as_ref().ok_or(OchreError::NotFitted)?;
        let xbar = self.xbar_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_comp = self.n_components_.ok_or(OchreError::NotFitted)?;
        let p = x.cols;
        let n = x.rows;
        let cols_to_take = n_comp.min(scalings.cols);
        let backend = effective_predict_backend(&self.device, n.saturating_mul(p));

        if matches!(backend, Backend::Cpu) {
            // Transpose first cols_to_take columns of scalings into contiguous rows (cols_to_take x p)
            let mut s_t = vec![0.0; cols_to_take * p];
            for c in 0..cols_to_take {
                for r in 0..p {
                    s_t[c * p + r] = scalings.data[r * scalings.cols + c];
                }
            }
            let offset: Vec<f64> = if self.solver == "eigen" {
                vec![0.0; cols_to_take]
            } else {
                (0..cols_to_take)
                    .map(|c| dot4(xbar, &s_t[c * p..(c + 1) * p]))
                    .collect()
            };
            let mut out = vec![0.0; n * cols_to_take];
            let x_data = &x.data;
            let compute_row = |(r, dst): (usize, &mut [f64])| {
                let x_row = &x_data[r * p..(r + 1) * p];
                for c in 0..cols_to_take {
                    dst[c] = dot4(x_row, &s_t[c * p..(c + 1) * p]) - offset[c];
                }
            };
            if n >= PARALLEL_ROW_THRESHOLD {
                out.par_chunks_exact_mut(cols_to_take)
                    .enumerate()
                    .for_each(compute_row);
            } else {
                out.chunks_exact_mut(cols_to_take)
                    .enumerate()
                    .for_each(compute_row);
            }
            return Matrix::from_vec(n, cols_to_take, out);
        }

        // Slice first n_comp columns of scalings (p x n_comp)
        let mut s_sub_data = Vec::with_capacity(p * cols_to_take);
        for r in 0..p {
            for c in 0..cols_to_take {
                s_sub_data.push(scalings.data[r * scalings.cols + c]);
            }
        }
        let s_sub = Matrix::from_vec(p, cols_to_take, s_sub_data)?;

        let x_in = if self.solver == "eigen" {
            x.clone()
        } else {
            let mut x_centered_data = vec![0.0; n * p];
            for r in 0..n {
                for c in 0..p {
                    x_centered_data[r * p + c] = x.data[r * p + c] - xbar[c];
                }
            }
            Matrix::from_vec(n, p, x_centered_data)?
        };
        backend.matmul(&x_in, &s_sub)
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LDA);
        write_str(&mut buf, &self.solver);
        write_opt_f64(&mut buf, self.shrinkage);
        write_bool(&mut buf, self.shrinkage_auto);
        write_opt_f64_vec(&mut buf, &self.priors);
        write_opt_usize(&mut buf, self.n_components);
        write_f64(&mut buf, self.tol);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.priors_);
        write_opt_matrix(&mut buf, &self.means_);
        write_opt_f64_vec(&mut buf, &self.xbar_);
        write_opt_matrix(&mut buf, &self.coef_);
        write_opt_f64_vec(&mut buf, &self.intercept_);
        write_opt_matrix(&mut buf, &self.scalings_);
        write_opt_f64_vec(&mut buf, &self.explained_variance_ratio_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_components_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LDA)?;
        let solver = r.read_str()?;
        let shrinkage = r.read_opt_f64()?;
        let shrinkage_auto = r.read_bool()?;
        let priors = r.read_opt_f64_vec()?;
        let n_components = r.read_opt_usize()?;
        let tol = r.read_f64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let priors_ = r.read_opt_f64_vec()?;
        let means_ = r.read_opt_matrix()?;
        let xbar_ = r.read_opt_f64_vec()?;
        let coef_ = r.read_opt_matrix()?;
        let intercept_ = r.read_opt_f64_vec()?;
        let scalings_ = r.read_opt_matrix()?;
        let explained_variance_ratio_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_components_ = r.read_opt_usize()?;

        let is_fitted = n_features_in_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;
        let coef_packed_ = match (&coef_, n_features_in_) {
            (Some(c), Some(p)) if c.rows >= 4 => Some(pack_coef_4(&c.data, c.rows, p)),
            _ => None,
        };

        Ok(Self {
            solver,
            shrinkage,
            shrinkage_auto,
            priors,
            n_components,
            tol,
            device: dev,
            device_,
            classes_,
            priors_,
            means_,
            xbar_,
            coef_,
            intercept_,
            scalings_,
            explained_variance_ratio_,
            n_features_in_,
            n_components_,
            coef_packed_,
        })
    }
}

/// Quadratic Discriminant Analysis (QDA) classifier (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    pub priors: Option<Vec<f64>>,
    pub reg_param: f64,
    pub tol: f64,
    pub store_covariance: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<f64>>,
    pub priors_: Option<Vec<f64>>,
    pub means_: Option<Matrix>,
    pub scalings_: Option<Vec<Vec<f64>>>,
    pub rotations_: Option<Vec<Matrix>>,
    pub covariance_: Option<Vec<Matrix>>,
    pub n_features_in_: Option<usize>,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new(
        priors: Option<Vec<f64>>,
        reg_param: f64,
        tol: f64,
        store_covariance: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !reg_param.is_finite() || !(0.0..=1.0).contains(&reg_param) {
            return Err(OchreError::InvalidParameter(format!(
                "reg_param ({reg_param}) must be a finite float in [0.0, 1.0] (・`ω´・)"
            )));
        }
        validate_tol(tol)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            priors,
            reg_param,
            tol,
            store_covariance,
            device: dev,
            device_: None,
            classes_: None,
            priors_: None,
            means_: None,
            scalings_: None,
            rotations_: None,
            covariance_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.n_features_in_.is_some() || self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    fn reset_fitted(&mut self) {
        self.classes_ = None;
        self.priors_ = None;
        self.means_ = None;
        self.scalings_ = None;
        self.rotations_ = None;
        self.covariance_ = None;
        self.n_features_in_ = None;
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.reset_fitted();
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;

        let n_features = x.cols;
        let classes = extract_sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(OchreError::InsufficientClasses(n_classes));
        }

        let y_idx = map_labels_to_indices(y, &classes)?;
        let (counts, priors, means, _xbar) =
            compute_class_stats(x, &y_idx, n_classes, self.priors.as_deref())?;

        for (k, &cnt) in counts.iter().enumerate() {
            if cnt < 2 {
                return Err(OchreError::InvalidParameter(format!(
                    "y has a class ({}) with fewer than 2 samples (got {cnt}) ( >_< )",
                    classes[k]
                )));
            }
        }

        let backend = effective_backend(&self.device, x.rows * n_features);
        let mut scalings_list = Vec::with_capacity(n_classes);
        let mut rotations_list = Vec::with_capacity(n_classes);
        let mut cov_list = if self.store_covariance {
            Some(Vec::with_capacity(n_classes))
        } else {
            None
        };

        for (k, &nk) in counts.iter().enumerate() {
            let mut xk_data = Vec::with_capacity(nk * n_features);
            let m_row = &means.data[(k * n_features)..((k + 1) * n_features)];
            for (i, &cls) in y_idx.iter().enumerate() {
                if cls == k {
                    let x_row = &x.data[(i * n_features)..((i + 1) * n_features)];
                    for j in 0..n_features {
                        xk_data.push(x_row[j] - m_row[j]);
                    }
                }
            }
            let xk = Matrix::from_vec(nk, n_features, xk_data)?;
            let dummy_y = vec![0.0; nk];
            let (mut cov_k, _) = backend.compute_normal_equation(&xk, &dummy_y, false)?;

            let inv_n = 1.0 / (nk as f64);
            let one_minus = 1.0 - self.reg_param;
            for r in 0..n_features {
                for c in 0..n_features {
                    let idx = r * n_features + c;
                    cov_k.data[idx] = one_minus * (cov_k.data[idx] * inv_n);
                    if r == c {
                        cov_k.data[idx] += self.reg_param;
                    }
                }
            }

            if let Some(ref mut covs) = cov_list {
                covs.push(cov_k.clone());
            }

            let eig = backend.small_sym_eig_svd(&cov_k)?;
            let floor = self.tol.max(1e-12);
            let s2: Vec<f64> = eig.eigenvalues.iter().map(|&v| v.max(floor)).collect();
            scalings_list.push(s2);
            rotations_list.push(eig.eigenvectors);
        }

        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
        self.means_ = Some(means);
        self.scalings_ = Some(scalings_list);
        self.rotations_ = Some(rotations_list);
        self.covariance_ = cov_list;
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn validate_predict_input(&self, x: &Matrix) -> OchreResult<usize> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        Ok(expected)
    }

    fn decision_scores_all(&self, x: &Matrix) -> OchreResult<Matrix> {
        let p = self.validate_predict_input(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let priors = self.priors_.as_ref().ok_or(OchreError::NotFitted)?;
        let means = self.means_.as_ref().ok_or(OchreError::NotFitted)?;
        let scalings = self.scalings_.as_ref().ok_or(OchreError::NotFitted)?;
        let rotations = self.rotations_.as_ref().ok_or(OchreError::NotFitted)?;

        let n_classes = classes.len();
        let n_samples = x.rows;
        let backend = effective_backend(&self.device, n_samples * p);

        if matches!(backend, Backend::Cpu) {
            // Precompute per-class pre-scaled transposed rotations (rot_t[c, j] *= inv_s[c]),
            // pre-scaled projected means (pm[c] *= inv_s[c]), and class constants.
            let mut rot_ts = Vec::with_capacity(n_classes);
            let mut proj_means = Vec::with_capacity(n_classes);
            let mut class_consts = Vec::with_capacity(n_classes);

            for k in 0..n_classes {
                let m_row = &means.data[(k * p)..((k + 1) * p)];
                let rot = &rotations[k];
                let s2 = &scalings[k];
                let mut rot_t = vec![0.0; p * p];
                let mut pm = Vec::with_capacity(p);
                let mut logdet = 0.0;
                for c in 0..p {
                    let s2c = s2[c];
                    logdet += s2c.ln();
                    let inv_s = if s2c > 0.0 { 1.0 / s2c.sqrt() } else { 0.0 };
                    let dst = &mut rot_t[c * p..(c + 1) * p];
                    for (r, dst_val) in dst.iter_mut().enumerate().take(p) {
                        *dst_val = rot.data[r * p + c] * inv_s;
                    }
                    pm.push(dot4(m_row, dst));
                }
                let log_prior = if priors[k] > 0.0 {
                    priors[k].ln()
                } else {
                    f64::NEG_INFINITY
                };
                rot_ts.push(rot_t);
                proj_means.push(pm);
                class_consts.push(-0.5 * logdet + log_prior);
            }

            let mut scores = vec![0.0; n_samples * n_classes];
            let x_data = &x.data;
            let compute_chunk = |(chunk_idx, out_chunk): (usize, &mut [f64])| {
                let start_r = chunk_idx * QDA_CHUNK_ROWS;
                let chunk_rows = out_chunk.len() / n_classes;
                let x_chunk = &x_data[start_r * p..(start_r + chunk_rows) * p];
                let mut r = 0usize;
                while r + 4 <= chunk_rows {
                    let x0 = &x_chunk[r * p..(r + 1) * p];
                    let x1 = &x_chunk[(r + 1) * p..(r + 2) * p];
                    let x2 = &x_chunk[(r + 2) * p..(r + 3) * p];
                    let x3 = &x_chunk[(r + 3) * p..(r + 4) * p];
                    for k in 0..n_classes {
                        let rot_t = &rot_ts[k];
                        let pm = &proj_means[k];
                        let mut n0 = 0.0;
                        let mut n1 = 0.0;
                        let mut n2 = 0.0;
                        let mut n3 = 0.0;
                        for c in 0..p {
                            let rt_row = &rot_t[c * p..(c + 1) * p];
                            let (d0, d1, d2, d3) = dot4_4rows(x0, x1, x2, x3, rt_row);
                            let pm_c = pm[c];
                            let p0 = d0 - pm_c;
                            let p1 = d1 - pm_c;
                            let p2 = d2 - pm_c;
                            let p3 = d3 - pm_c;
                            n0 += p0 * p0;
                            n1 += p1 * p1;
                            n2 += p2 * p2;
                            n3 += p3 * p3;
                        }
                        let cc = class_consts[k];
                        out_chunk[r * n_classes + k] = cc - 0.5 * n0;
                        out_chunk[(r + 1) * n_classes + k] = cc - 0.5 * n1;
                        out_chunk[(r + 2) * n_classes + k] = cc - 0.5 * n2;
                        out_chunk[(r + 3) * n_classes + k] = cc - 0.5 * n3;
                    }
                    r += 4;
                }
                while r < chunk_rows {
                    let x_row = &x_chunk[r * p..(r + 1) * p];
                    let dst = &mut out_chunk[r * n_classes..(r + 1) * n_classes];
                    for k in 0..n_classes {
                        let rot_t = &rot_ts[k];
                        let pm = &proj_means[k];
                        let mut norm2 = 0.0;
                        for c in 0..p {
                            let proj_c = dot4(x_row, &rot_t[c * p..(c + 1) * p]) - pm[c];
                            norm2 += proj_c * proj_c;
                        }
                        dst[k] = class_consts[k] - 0.5 * norm2;
                    }
                    r += 1;
                }
            };
            if n_samples >= PARALLEL_ROW_THRESHOLD {
                scores
                    .par_chunks_mut(QDA_CHUNK_ROWS * n_classes)
                    .enumerate()
                    .for_each(compute_chunk);
            } else {
                compute_chunk((0, &mut scores));
            }
            return Matrix::from_vec(n_samples, n_classes, scores);
        }

        let mut scores = vec![0.0; n_samples * n_classes];

        for k in 0..n_classes {
            let m_row = &means.data[(k * p)..((k + 1) * p)];
            let mut xc_data = vec![0.0; n_samples * p];
            for r in 0..n_samples {
                let x_row = &x.data[(r * p)..((r + 1) * p)];
                let out_row = &mut xc_data[(r * p)..((r + 1) * p)];
                for j in 0..p {
                    out_row[j] = x_row[j] - m_row[j];
                }
            }
            let xc = Matrix::from_vec(n_samples, p, xc_data)?;
            let proj = backend.matmul(&xc, &rotations[k])?;
            let s2 = &scalings[k];
            let logdet: f64 = s2.iter().map(|&v| v.ln()).sum();
            let log_prior = if priors[k] > 0.0 {
                priors[k].ln()
            } else {
                f64::NEG_INFINITY
            };

            for r in 0..n_samples {
                let p_row = &proj.data[(r * p)..((r + 1) * p)];
                let mut norm2 = 0.0;
                for j in 0..p {
                    norm2 += (p_row[j] * p_row[j]) / s2[j];
                }
                scores[r * n_classes + k] = -0.5 * (norm2 + logdet) + log_prior;
            }
        }

        Matrix::from_vec(n_samples, n_classes, scores)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        let n_classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?.len();
        if n_classes == 2 {
            let n = all_scores.rows;
            let mut diff = Vec::with_capacity(n);
            for i in 0..n {
                diff.push(all_scores.data[2 * i + 1] - all_scores.data[2 * i]);
            }
            Matrix::from_vec(n, 1, diff)
        } else {
            Ok(all_scores)
        }
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        log_softmax_matrix(&all_scores)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let all_scores = self.decision_scores_all(x)?;
        softmax_matrix(&all_scores)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let all_scores = self.decision_scores_all(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n = all_scores.rows;
        let k = classes.len();
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let row = &all_scores.data[(i * k)..((i + 1) * k)];
            let mut best_c = 0;
            let mut best_v = row[0];
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            preds.push(classes[best_c]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        accuracy_score(y, &preds)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_QDA);
        write_opt_f64_vec(&mut buf, &self.priors);
        write_f64(&mut buf, self.reg_param);
        write_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.store_covariance);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_opt_f64_vec(&mut buf, &self.priors_);
        write_opt_matrix(&mut buf, &self.means_);
        match &self.scalings_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for v in list {
                    write_f64_vec(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.rotations_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for m in list {
                    crate::dit_io::write_matrix(&mut buf, m);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.covariance_ {
            Some(list) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, list.len());
                for m in list {
                    crate::dit_io::write_matrix(&mut buf, m);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_QDA)?;
        let priors = r.read_opt_f64_vec()?;
        let reg_param = r.read_f64()?;
        let tol = r.read_f64()?;
        let store_covariance = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let priors_ = r.read_opt_f64_vec()?;
        let means_ = r.read_opt_matrix()?;

        let scalings_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_f64_vec()?);
            }
            Some(list)
        } else {
            None
        };

        let rotations_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_matrix()?);
            }
            Some(list)
        } else {
            None
        };

        let covariance_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.read_matrix()?);
            }
            Some(list)
        } else {
            None
        };

        let n_features_in_ = r.read_opt_usize()?;
        let is_fitted = n_features_in_.is_some();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            priors,
            reg_param,
            tol,
            store_covariance,
            device: dev,
            device_,
            classes_,
            priors_,
            means_,
            scalings_,
            rotations_,
            covariance_,
            n_features_in_,
        })
    }
}

#[pyclass(from_py_object, name = "LinearDiscriminantAnalysis")]
#[derive(Clone)]
pub struct PyLinearDiscriminantAnalysis {
    inner: LinearDiscriminantAnalysis,
}

crate::pymethods_with_dit!(PyLinearDiscriminantAnalysis, LinearDiscriminantAnalysis, {
    #[new]
    #[pyo3(signature = (
        solver = "svd".to_string(),
        shrinkage = None,
        shrinkage_auto = false,
        priors = None,
        n_components = None,
        tol = 1e-4,
        device = None
    ))]
    pub fn new(
        solver: String,
        shrinkage: Option<f64>,
        shrinkage_auto: bool,
        priors: Option<Vec<f64>>,
        n_components: Option<usize>,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LinearDiscriminantAnalysis::new(
                solver,
                shrinkage,
                shrinkage_auto,
                priors,
                n_components,
                tol,
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

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.decision_function(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict_log_proba(&x_mat)?;
        matrix_to_pybytes(py, &out)
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

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&x_mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let out = self.inner.fit_transform(&x_mat, &y_vec)?;
        matrix_to_pybytes(py, &out)
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
    pub fn solver(&self) -> String {
        self.inner.solver.clone()
    }

    #[getter]
    pub fn shrinkage(&self) -> Option<f64> {
        self.inner.shrinkage
    }

    #[getter]
    pub fn shrinkage_auto(&self) -> bool {
        self.inner.shrinkage_auto
    }

    #[getter]
    pub fn n_components(&self) -> Option<usize> {
        self.inner.n_components
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn priors_(&self) -> Option<Vec<f64>> {
        self.inner.priors_.clone()
    }

    #[getter]
    pub fn xbar_(&self) -> Option<Vec<f64>> {
        self.inner.xbar_.clone()
    }

    #[getter]
    pub fn intercept_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_.clone()
    }

    #[getter]
    pub fn explained_variance_ratio_(&self) -> Option<Vec<f64>> {
        self.inner.explained_variance_ratio_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn n_components_(&self) -> Option<usize> {
        self.inner.n_components_
    }

    pub fn means_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.means_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    pub fn coef_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.coef_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    pub fn scalings_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.scalings_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }
});

#[pyclass(from_py_object, name = "QuadraticDiscriminantAnalysis")]
#[derive(Clone)]
pub struct PyQuadraticDiscriminantAnalysis {
    inner: QuadraticDiscriminantAnalysis,
}

crate::pymethods_with_dit!(
    PyQuadraticDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    {
        #[new]
        #[pyo3(signature = (
            priors = None,
            reg_param = 0.0,
            tol = 1e-4,
            store_covariance = false,
            device = None
        ))]
        pub fn new(
            priors: Option<Vec<f64>>,
            reg_param: f64,
            tol: f64,
            store_covariance: bool,
            device: Option<String>,
        ) -> PyResult<Self> {
            Ok(Self {
                inner: QuadraticDiscriminantAnalysis::new(
                    priors,
                    reg_param,
                    tol,
                    store_covariance,
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

        pub fn decision_function_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let out = self.inner.decision_function(&x_mat)?;
            matrix_to_pybytes(py, &out)
        }

        pub fn predict_proba_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let out = self.inner.predict_proba(&x_mat)?;
            matrix_to_pybytes(py, &out)
        }

        pub fn predict_log_proba_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let x_mat = buffer_to_pooled_matrix(py, &x)?;
            let out = self.inner.predict_log_proba(&x_mat)?;
            matrix_to_pybytes(py, &out)
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
        pub fn reg_param(&self) -> f64 {
            self.inner.reg_param
        }

        #[getter]
        pub fn tol(&self) -> f64 {
            self.inner.tol
        }

        #[getter]
        pub fn store_covariance(&self) -> bool {
            self.inner.store_covariance
        }

        #[getter]
        pub fn classes_(&self) -> Option<Vec<f64>> {
            self.inner.classes_.clone()
        }

        #[getter]
        pub fn priors_(&self) -> Option<Vec<f64>> {
            self.inner.priors_.clone()
        }

        #[getter]
        pub fn scalings_(&self) -> Option<Vec<Vec<f64>>> {
            self.inner.scalings_.clone()
        }

        #[getter]
        pub fn n_features_in_(&self) -> Option<usize> {
            self.inner.n_features_in_
        }

        pub fn means_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
            match &self.inner.means_ {
                Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
                None => Ok(None),
            }
        }

        #[allow(clippy::type_complexity)]
        pub fn rotations_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
            match &self.inner.rotations_ {
                Some(list) => {
                    let mut out = Vec::with_capacity(list.len());
                    for m in list {
                        out.push(matrix_to_pybytes(py, m)?);
                    }
                    Ok(Some(out))
                }
                None => Ok(None),
            }
        }

        #[allow(clippy::type_complexity)]
        pub fn covariance_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
            match &self.inner.covariance_ {
                Some(list) => {
                    let mut out = Vec::with_capacity(list.len());
                    for m in list {
                        out.push(matrix_to_pybytes(py, m)?);
                    }
                    Ok(Some(out))
                }
                None => Ok(None),
            }
        }
    }
);

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearDiscriminantAnalysis>()?;
    m.add_class::<PyQuadraticDiscriminantAnalysis>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda_binary_and_multiclass_solvers() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.2],
            vec![0.2, 0.0],
            vec![-0.1, -0.1],
            vec![0.1, 0.1],
            vec![5.0, 5.2],
            vec![5.2, 5.0],
            vec![4.9, 4.9],
            vec![5.1, 5.1],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        for solver in ["svd", "lsqr", "eigen"] {
            let mut lda = LinearDiscriminantAnalysis::new(
                solver.to_string(),
                None,
                false,
                None,
                None,
                1e-4,
                Some("cpu".to_string()),
            )
            .unwrap();
            lda.fit(&x, &y).unwrap();
            assert_eq!(lda.score(&x, &y).unwrap(), 1.0);
            let probs = lda.predict_proba(&x).unwrap();
            assert_eq!(probs.rows, 8);
            assert_eq!(probs.cols, 2);
            for r in 0..8 {
                let sum = probs.data[2 * r] + probs.data[2 * r + 1];
                assert!((sum - 1.0).abs() < 1e-12);
            }
            let bytes = lda.to_bytes();
            let loaded = LinearDiscriminantAnalysis::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), y);
        }
    }

    #[test]
    fn test_qda_fit_predict_and_dit() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![-0.2, 0.1],
            vec![0.2, -0.1],
            vec![4.0, 4.0],
            vec![4.2, 3.8],
            vec![3.9, 4.1],
            vec![4.1, 4.2],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut qda =
            QuadraticDiscriminantAnalysis::new(None, 0.1, 1e-4, true, Some("cpu".to_string()))
                .unwrap();
        qda.fit(&x, &y).unwrap();
        assert_eq!(qda.score(&x, &y).unwrap(), 1.0);
        assert!(qda.covariance_.is_some());
        let bytes = qda.to_bytes();
        let loaded = QuadraticDiscriminantAnalysis::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), y);
    }

    #[test]
    fn test_lda_qda_large_parallel_parity() {
        // Exercise parallel threshold (>= 512) and fused solve_svd / blocked QDA predict with non-multiple-of-4 cols
        let n = 650;
        let p = 13;
        let mut x_data = vec![0.0; n * p];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let cls = i % 3;
            y[i] = cls as f64;
            let shift = (cls as f64) * 4.0;
            for j in 0..p {
                x_data[i * p + j] = shift + (((i * 7 + j * 13) % 29) as f64) * 0.05;
            }
        }
        let x = Matrix::from_vec(n, p, x_data).unwrap();

        let mut lda = LinearDiscriminantAnalysis::new(
            "svd".to_string(),
            None,
            false,
            None,
            None,
            1e-4,
            Some("cpu".to_string()),
        )
        .unwrap();
        lda.fit(&x, &y).unwrap();
        assert!(lda.score(&x, &y).unwrap() > 0.99);
        let x_t = lda.transform(&x).unwrap();
        assert_eq!(x_t.rows, n);
        assert_eq!(x_t.cols, 2);

        let mut qda =
            QuadraticDiscriminantAnalysis::new(None, 0.05, 1e-4, false, Some("cpu".to_string()))
                .unwrap();
        qda.fit(&x, &y).unwrap();
        assert!(qda.score(&x, &y).unwrap() > 0.99);
        let probs = qda.predict_proba(&x).unwrap();
        for r in 0..n {
            let s = probs[(r, 0)] + probs[(r, 1)] + probs[(r, 2)];
            assert!((s - 1.0).abs() < 1e-10);
        }

        // Verify LDA zero-alloc 4-row SIMD predict parity across k=2, 3, 5, 6 and fused NaN/Inf check
        for k in [2usize, 3, 5, 6] {
            let mut xk_data = vec![0.0; n * p];
            let mut yk = vec![0.0; n];
            for i in 0..n {
                let cls = i % k;
                yk[i] = cls as f64;
                for j in 0..p {
                    let sig = if j == cls { 5.0 } else { 0.0 };
                    xk_data[i * p + j] = sig + (((i * 7 + j * 13) % 23) as f64) * 0.05;
                }
            }
            let xk = Matrix::from_vec(n, p, xk_data).unwrap();
            let mut lda_k = LinearDiscriminantAnalysis::new(
                "svd".to_string(),
                None,
                false,
                None,
                None,
                1e-4,
                Some("cpu".to_string()),
            )
            .unwrap();
            lda_k.fit(&xk, &yk).unwrap();
            let loaded =
                LinearDiscriminantAnalysis::from_bytes(&lda_k.to_bytes(), Some("cpu")).unwrap();
            let df = loaded.decision_function(&xk).unwrap();
            let preds = loaded.predict(&xk).unwrap();
            let classes = loaded.classes_.as_ref().unwrap();
            for r in [0, 3, 4, 511, 512, 649] {
                if k == 2 {
                    let expected = if df[(r, 0)] > 0.0 {
                        classes[1]
                    } else {
                        classes[0]
                    };
                    assert_eq!(preds[r], expected);
                } else {
                    let mut best_c = 0;
                    let mut best_v = f64::NEG_INFINITY;
                    for c in 0..k {
                        if df[(r, c)] > best_v {
                            best_v = df[(r, c)];
                            best_c = c;
                        }
                    }
                    assert_eq!(preds[r], classes[best_c]);
                }
            }
            let mut x_nan = xk.clone();
            x_nan[(514, 3)] = f64::NAN;
            assert!(loaded.predict(&x_nan).is_err());
            assert!(loaded.decision_function(&x_nan).is_err());
        }
    }
}
