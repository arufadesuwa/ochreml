// OchreML Linear Algebra Module (＾▽＾)
// Provides fundamental matrix structures, trait-based indexing, and linear equation solvers.

use crate::error::{OchreError, OchreResult};
use rayon::prelude::*;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline(always)]
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline(always)]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data[r * self.cols + c]
    }
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = 1.0;
        }
        mat
    }

    // Convert 2D vector data into a contiguous Matrix structure
    pub fn from_2d(data_2d: &[Vec<f64>]) -> OchreResult<Self> {
        let rows = data_2d.len();
        if rows == 0 {
            return Err(OchreError::EmptyData);
        }
        let cols = data_2d[0].len();
        if cols == 0 {
            return Err(OchreError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: 1,
                actual_rows: rows,
                actual_cols: 0,
            });
        }

        let mut flat = Vec::with_capacity(rows * cols);
        for (r, row) in data_2d.iter().enumerate() {
            if row.len() != cols {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: r,
                    expected_cols: cols,
                    actual_rows: r,
                    actual_cols: row.len(),
                });
            }
            flat.extend_from_slice(row);
        }

        Ok(Self {
            rows,
            cols,
            data: flat,
        })
    }

    // Construct Matrix directly from contiguous flat Vec (Zero-Copy) (*≧ω≦*)
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> OchreResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if data.len() != rows * cols {
            return Err(OchreError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: cols,
                actual_rows: data.len(),
                actual_cols: 1,
            });
        }
        Ok(Self { rows, cols, data })
    }

    // Construct Matrix directly from contiguous slice (o´∀｀o)
    pub fn from_slice(rows: usize, cols: usize, slice: &[f64]) -> OchreResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if slice.len() != rows * cols {
            return Err(OchreError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: cols,
                actual_rows: slice.len(),
                actual_cols: 1,
            });
        }
        Ok(Self {
            rows,
            cols,
            data: slice.to_vec(),
        })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self[(r, c)]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self[(r, c)] = val;
    }

    // Matrix transpose: swap rows and columns
    pub fn transpose(&self) -> Self {
        let mut res = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                res[(c, r)] = self[(r, c)];
            }
        }
        res
    }

    // Matrix multiplication: A * B
    pub fn matmul(&self, other: &Matrix) -> OchreResult<Self> {
        if self.cols != other.rows {
            return Err(OchreError::MatmulDimensionMismatch {
                rows_a: self.rows,
                cols_a: self.cols,
                rows_b: other.rows,
                cols_b: other.cols,
            });
        }

        let mut res = Self::zeros(self.rows, other.cols);
        let other_cols = other.cols;
        let inner_k = self.cols;
        let mul_row = |(r, out_row): (usize, &mut [f64])| {
            let a_row = &self.data[r * inner_k..(r + 1) * inner_k];
            for (k, &a_rk) in a_row.iter().enumerate() {
                let b_row = &other.data[k * other_cols..(k + 1) * other_cols];
                for (dst, &b_kc) in out_row.iter_mut().zip(b_row.iter()) {
                    *dst += a_rk * b_kc;
                }
            }
        };
        if self.rows >= 64 && self.rows * other_cols * inner_k >= 65_536 {
            res.data
                .par_chunks_mut(other_cols)
                .enumerate()
                .for_each(mul_row);
        } else {
            res.data
                .chunks_exact_mut(other_cols)
                .enumerate()
                .for_each(mul_row);
        }
        Ok(res)
    }

    // Matrix-vector multiplication: A * v
    pub fn matvec(&self, vec: &[f64]) -> OchreResult<Vec<f64>> {
        if self.cols != vec.len() {
            return Err(OchreError::MatvecDimensionMismatch {
                cols: self.cols,
                len: vec.len(),
            });
        }
        if self.rows == 0 {
            return Ok(Vec::new());
        }
        if self.cols == 0 {
            return Ok(vec![0.0; self.rows]);
        }

        let cols = self.cols;
        let res = if self.rows >= 128 && self.rows.saturating_mul(cols) >= 16_384 {
            self.data
                .par_chunks(cols)
                .map(|row| dot_slice_unrolled(row, vec))
                .collect()
        } else {
            self.data
                .chunks_exact(cols)
                .map(|row| dot_slice_unrolled(row, vec))
                .collect()
        };
        Ok(res)
    }

    // Add regularization lambda to main diagonal for Tikhonov (Ridge) regularization
    pub fn add_ridge_diagonal(&mut self, lambda: f64) {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            self[(i, i)] += lambda;
        }
    }

    /// Add L2 regularization penalty `alpha` only to the first `n_features` diagonal entries,
    /// leaving the intercept row/column (`index == n_features`) unpenalized.
    pub fn add_ridge_feature_diagonal(&mut self, alpha: f64, n_features: usize) {
        let limit = n_features.min(self.rows).min(self.cols);
        for i in 0..limit {
            self[(i, i)] += alpha;
        }
    }
}

// Compute (X^T * X) and (X^T * y) in a single pass without allocating transpose matrices.
pub fn compute_normal_equation_mats(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    device: &crate::device::DeviceType,
) -> OchreResult<(Matrix, Vec<f64>)> {
    if x.rows != y.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y.len(),
        });
    }

    let backend = crate::backend::get_backend(device);
    backend.compute_normal_equation(x, y, fit_intercept)
}

const NORMAL_EQ_TILE: usize = 64;

#[inline(always)]
fn dot64(a: &[f64; NORMAL_EQ_TILE], b: &[f64; NORMAL_EQ_TILE]) -> f64 {
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
fn sum64(a: &[f64; NORMAL_EQ_TILE]) -> f64 {
    let mut acc = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            acc[lane] += a[base + lane];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

#[inline(always)]
fn dot64_1x4(
    a: &[f64; NORMAL_EQ_TILE],
    b0: &[f64; NORMAL_EQ_TILE],
    b1: &[f64; NORMAL_EQ_TILE],
    b2: &[f64; NORMAL_EQ_TILE],
    b3: &[f64; NORMAL_EQ_TILE],
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

#[inline(always)]
fn dot64_2x4(
    a0: &[f64; NORMAL_EQ_TILE],
    a1: &[f64; NORMAL_EQ_TILE],
    b0: &[f64; NORMAL_EQ_TILE],
    b1: &[f64; NORMAL_EQ_TILE],
    b2: &[f64; NORMAL_EQ_TILE],
    b3: &[f64; NORMAL_EQ_TILE],
) -> ((f64, f64, f64, f64), (f64, f64, f64, f64)) {
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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dot64_4x4(
    a0: &[f64; NORMAL_EQ_TILE],
    a1: &[f64; NORMAL_EQ_TILE],
    a2: &[f64; NORMAL_EQ_TILE],
    a3: &[f64; NORMAL_EQ_TILE],
    b0: &[f64; NORMAL_EQ_TILE],
    b1: &[f64; NORMAL_EQ_TILE],
    b2: &[f64; NORMAL_EQ_TILE],
    b3: &[f64; NORMAL_EQ_TILE],
) -> [[f64; 4]; 4] {
    let mut acc00 = [0.0f64; 4];
    let mut acc01 = [0.0f64; 4];
    let mut acc02 = [0.0f64; 4];
    let mut acc03 = [0.0f64; 4];
    let mut acc10 = [0.0f64; 4];
    let mut acc11 = [0.0f64; 4];
    let mut acc12 = [0.0f64; 4];
    let mut acc13 = [0.0f64; 4];
    let mut acc20 = [0.0f64; 4];
    let mut acc21 = [0.0f64; 4];
    let mut acc22 = [0.0f64; 4];
    let mut acc23 = [0.0f64; 4];
    let mut acc30 = [0.0f64; 4];
    let mut acc31 = [0.0f64; 4];
    let mut acc32 = [0.0f64; 4];
    let mut acc33 = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            let v_a0 = a0[base + lane];
            let v_a1 = a1[base + lane];
            let v_a2 = a2[base + lane];
            let v_a3 = a3[base + lane];
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
            acc20[lane] += v_a2 * v_b0;
            acc21[lane] += v_a2 * v_b1;
            acc22[lane] += v_a2 * v_b2;
            acc23[lane] += v_a2 * v_b3;
            acc30[lane] += v_a3 * v_b0;
            acc31[lane] += v_a3 * v_b1;
            acc32[lane] += v_a3 * v_b2;
            acc33[lane] += v_a3 * v_b3;
        }
    }
    let sum4 = |v: [f64; 4]| (v[0] + v[1]) + (v[2] + v[3]);
    [
        [sum4(acc00), sum4(acc01), sum4(acc02), sum4(acc03)],
        [sum4(acc10), sum4(acc11), sum4(acc12), sum4(acc13)],
        [sum4(acc20), sum4(acc21), sum4(acc22), sum4(acc23)],
        [sum4(acc30), sum4(acc31), sum4(acc32), sum4(acc33)],
    ]
}

#[inline(always)]
fn dot64_and_sum64_4(
    x0: &[f64; NORMAL_EQ_TILE],
    x1: &[f64; NORMAL_EQ_TILE],
    x2: &[f64; NORMAL_EQ_TILE],
    x3: &[f64; NORMAL_EQ_TILE],
    y: &[f64; NORMAL_EQ_TILE],
) -> ((f64, f64, f64, f64), (f64, f64, f64, f64)) {
    let mut d0 = [0.0f64; 4];
    let mut d1 = [0.0f64; 4];
    let mut d2 = [0.0f64; 4];
    let mut d3 = [0.0f64; 4];
    let mut s0 = [0.0f64; 4];
    let mut s1 = [0.0f64; 4];
    let mut s2 = [0.0f64; 4];
    let mut s3 = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            let yv = y[base + lane];
            let v0 = x0[base + lane];
            let v1 = x1[base + lane];
            let v2 = x2[base + lane];
            let v3 = x3[base + lane];
            d0[lane] += v0 * yv;
            d1[lane] += v1 * yv;
            d2[lane] += v2 * yv;
            d3[lane] += v3 * yv;
            s0[lane] += v0;
            s1[lane] += v1;
            s2[lane] += v2;
            s3[lane] += v3;
        }
    }
    let sum4 = |v: [f64; 4]| (v[0] + v[1]) + (v[2] + v[3]);
    (
        (sum4(d0), sum4(d1), sum4(d2), sum4(d3)),
        (sum4(s0), sum4(s1), sum4(s2), sum4(s3)),
    )
}

#[inline(always)]
fn dot64_and_sum64(a: &[f64; NORMAL_EQ_TILE], y: &[f64; NORMAL_EQ_TILE]) -> (f64, f64) {
    let mut d = [0.0f64; 4];
    let mut s = [0.0f64; 4];
    for chunk in 0..16 {
        let base = chunk * 4;
        for lane in 0..4 {
            let av = a[base + lane];
            d[lane] += av * y[base + lane];
            s[lane] += av;
        }
    }
    ((d[0] + d[1]) + (d[2] + d[3]), (s[0] + s[1]) + (s[2] + s[3]))
}

// Accumulate upper triangle of left_t * right_t^T into xt_x (stride p) using 4x4 SIMD micro-kernel.
#[inline]
fn accumulate_gram_upper_tile(
    left_t: &[[f64; NORMAL_EQ_TILE]],
    right_t: &[[f64; NORMAL_EQ_TILE]],
    xt_x: &mut Matrix,
    p: usize,
) {
    let d = left_t.len();
    let mut j = 0;
    while j + 4 <= d {
        let lj0 = &left_t[j];
        let lj1 = &left_t[j + 1];
        let lj2 = &left_t[j + 2];
        let lj3 = &left_t[j + 3];
        let rj0 = &right_t[j];
        let rj1 = &right_t[j + 1];
        let rj2 = &right_t[j + 2];
        let rj3 = &right_t[j + 3];
        let off0 = j * p;
        let off1 = (j + 1) * p;
        let off2 = (j + 2) * p;
        let off3 = (j + 3) * p;

        let (d00, d01, d02, d03) = dot64_1x4(lj0, rj0, rj1, rj2, rj3);
        xt_x.data[off0 + j] += d00;
        xt_x.data[off0 + j + 1] += d01;
        xt_x.data[off0 + j + 2] += d02;
        xt_x.data[off0 + j + 3] += d03;
        xt_x.data[off1 + j + 1] += dot64(lj1, rj1);
        xt_x.data[off1 + j + 2] += dot64(lj1, rj2);
        xt_x.data[off1 + j + 3] += dot64(lj1, rj3);
        xt_x.data[off2 + j + 2] += dot64(lj2, rj2);
        xt_x.data[off2 + j + 3] += dot64(lj2, rj3);
        xt_x.data[off3 + j + 3] += dot64(lj3, rj3);

        let mut k = j + 4;
        while k + 4 <= d {
            let res = dot64_4x4(
                lj0,
                lj1,
                lj2,
                lj3,
                &right_t[k],
                &right_t[k + 1],
                &right_t[k + 2],
                &right_t[k + 3],
            );
            for c in 0..4 {
                xt_x.data[off0 + k + c] += res[0][c];
                xt_x.data[off1 + k + c] += res[1][c];
                xt_x.data[off2 + k + c] += res[2][c];
                xt_x.data[off3 + k + c] += res[3][c];
            }
            k += 4;
        }
        while k < d {
            let (s0, s1, s2, s3) = dot64_1x4(&right_t[k], lj0, lj1, lj2, lj3);
            xt_x.data[off0 + k] += s0;
            xt_x.data[off1 + k] += s1;
            xt_x.data[off2 + k] += s2;
            xt_x.data[off3 + k] += s3;
            k += 1;
        }
        j += 4;
    }
    while j + 2 <= d {
        let lj0 = &left_t[j];
        let lj1 = &left_t[j + 1];
        let off0 = j * p;
        let off1 = (j + 1) * p;
        xt_x.data[off0 + j] += dot64(lj0, &right_t[j]);
        xt_x.data[off0 + j + 1] += dot64(lj0, &right_t[j + 1]);
        xt_x.data[off1 + j + 1] += dot64(lj1, &right_t[j + 1]);

        let mut k = j + 2;
        while k + 4 <= d {
            let ((r00, r01, r02, r03), (r10, r11, r12, r13)) = dot64_2x4(
                lj0,
                lj1,
                &right_t[k],
                &right_t[k + 1],
                &right_t[k + 2],
                &right_t[k + 3],
            );
            xt_x.data[off0 + k] += r00;
            xt_x.data[off0 + k + 1] += r01;
            xt_x.data[off0 + k + 2] += r02;
            xt_x.data[off0 + k + 3] += r03;
            xt_x.data[off1 + k] += r10;
            xt_x.data[off1 + k + 1] += r11;
            xt_x.data[off1 + k + 2] += r12;
            xt_x.data[off1 + k + 3] += r13;
            k += 4;
        }
        while k < d {
            xt_x.data[off0 + k] += dot64(lj0, &right_t[k]);
            xt_x.data[off1 + k] += dot64(lj1, &right_t[k]);
            k += 1;
        }
        j += 2;
    }
    while j < d {
        let lj = &left_t[j];
        let row_off = j * p;
        let mut k = j;
        while k + 4 <= d {
            let (s0, s1, s2, s3) = dot64_1x4(
                lj,
                &right_t[k],
                &right_t[k + 1],
                &right_t[k + 2],
                &right_t[k + 3],
            );
            xt_x.data[row_off + k] += s0;
            xt_x.data[row_off + k + 1] += s1;
            xt_x.data[row_off + k + 2] += s2;
            xt_x.data[row_off + k + 3] += s3;
            k += 4;
        }
        while k < d {
            xt_x.data[row_off + k] += dot64(lj, &right_t[k]);
            k += 1;
        }
        j += 1;
    }
}

fn mirror_upper_to_lower(xt_x: &mut Matrix, p: usize) {
    for j in 0..p {
        for k in (j + 1)..p {
            let val = xt_x.data[j * p + k];
            xt_x.data[k * p + j] = val;
        }
    }
}

// Compute (X^T * X) and (X^T * y) over a slice range [start_row..end_row] using L1-blocked 64-row tiles.
#[inline]
pub fn compute_normal_equation_range(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let d = x.cols;
    let p = if fit_intercept { d + 1 } else { d };
    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = vec![0.0; p];
    let mut sum_x = if fit_intercept {
        vec![0.0; d]
    } else {
        Vec::new()
    };
    let mut x_t = vec![[0.0f64; NORMAL_EQ_TILE]; d];
    let mut y_t = [0.0f64; NORMAL_EQ_TILE];

    let mut r_block = start_row;
    while r_block < end_row {
        let block_end = (r_block + NORMAL_EQ_TILE).min(end_row);
        let b_len = block_end - r_block;
        if b_len < NORMAL_EQ_TILE {
            for col in &mut x_t {
                col[b_len..NORMAL_EQ_TILE].fill(0.0);
            }
            y_t[b_len..NORMAL_EQ_TILE].fill(0.0);
        }

        y_t[0..b_len].copy_from_slice(&y[r_block..block_end]);
        let mut b = 0;
        while b + 4 <= b_len {
            let r0 = (r_block + b) * d;
            let row0 = &x.data[r0..(r0 + d)];
            let row1 = &x.data[(r0 + d)..(r0 + 2 * d)];
            let row2 = &x.data[(r0 + 2 * d)..(r0 + 3 * d)];
            let row3 = &x.data[(r0 + 3 * d)..(r0 + 4 * d)];
            for (j, col) in x_t.iter_mut().enumerate() {
                col[b] = row0[j];
                col[b + 1] = row1[j];
                col[b + 2] = row2[j];
                col[b + 3] = row3[j];
            }
            b += 4;
        }
        while b < b_len {
            let row = &x.data[((r_block + b) * d)..((r_block + b) * d + d)];
            for (j, &val) in row.iter().enumerate() {
                x_t[j][b] = val;
            }
            b += 1;
        }

        accumulate_gram_upper_tile(&x_t, &x_t, &mut xt_x, p);
        let mut j = 0;
        if fit_intercept {
            while j + 4 <= d {
                let ((d0, d1, d2, d3), (s0, s1, s2, s3)) =
                    dot64_and_sum64_4(&x_t[j], &x_t[j + 1], &x_t[j + 2], &x_t[j + 3], &y_t);
                xt_y[j] += d0;
                xt_y[j + 1] += d1;
                xt_y[j + 2] += d2;
                xt_y[j + 3] += d3;
                sum_x[j] += s0;
                sum_x[j + 1] += s1;
                sum_x[j + 2] += s2;
                sum_x[j + 3] += s3;
                j += 4;
            }
            while j < d {
                let (dj, sj) = dot64_and_sum64(&x_t[j], &y_t);
                xt_y[j] += dj;
                sum_x[j] += sj;
                j += 1;
            }
            xt_y[d] += sum64(&y_t);
        } else {
            while j + 4 <= d {
                let (d0, d1, d2, d3) =
                    dot64_1x4(&y_t, &x_t[j], &x_t[j + 1], &x_t[j + 2], &x_t[j + 3]);
                xt_y[j] += d0;
                xt_y[j + 1] += d1;
                xt_y[j + 2] += d2;
                xt_y[j + 3] += d3;
                j += 4;
            }
            while j < d {
                xt_y[j] += dot64(&x_t[j], &y_t);
                j += 1;
            }
        }
        r_block = block_end;
    }

    if fit_intercept {
        for (j, &sx) in sum_x.iter().enumerate() {
            xt_x.data[j * p + d] = sx;
        }
        xt_x.data[d * p + d] = (end_row.saturating_sub(start_row)) as f64;
    }
    mirror_upper_to_lower(&mut xt_x, p);
    Ok((xt_x, xt_y))
}

// Compute weighted multi-target (X^T * W * X) and (X^T * W * Y) in a single streaming pass (*≧ω≦*)
// Dispatches to trait-based ComputeBackend (CpuBackend, MultiGpuBackend, TpuBackend)
pub fn compute_normal_equation_mats_multi(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    device: &crate::device::DeviceType,
) -> OchreResult<(Matrix, Matrix)> {
    if x.rows != y.rows {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y.rows,
        });
    }
    if let Some(sw) = sample_weight {
        if sw.len() != x.rows {
            return Err(OchreError::WeightLengthMismatch {
                weights: sw.len(),
                samples: x.rows,
            });
        }
        let mut weight_sum = 0.0;
        for &w in sw {
            if w < 0.0 || w.is_nan() {
                return Err(OchreError::InvalidWeights);
            }
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
    }

    let backend = crate::backend::get_backend(device);
    backend.compute_normal_equation_multi(x, y, sample_weight, fit_intercept)
}

// Compute (X^T * W * X) and (X^T * W * Y) over row slice range [start_row..end_row] using L1-blocked 64-row tiles
pub fn compute_normal_equation_multi_range(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Matrix)> {
    let d = x.cols;
    let p = if fit_intercept { d + 1 } else { d };
    let k_targets = y.cols;
    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = Matrix::zeros(p, k_targets);
    let mut x_t = vec![[0.0f64; NORMAL_EQ_TILE]; d];
    let mut y_t = vec![[0.0f64; NORMAL_EQ_TILE]; k_targets];

    match sample_weight {
        None => {
            let mut r_block = start_row;
            while r_block < end_row {
                let block_end = (r_block + NORMAL_EQ_TILE).min(end_row);
                let b_len = block_end - r_block;
                if b_len < NORMAL_EQ_TILE {
                    for col in &mut x_t {
                        col[b_len..NORMAL_EQ_TILE].fill(0.0);
                    }
                    for col in &mut y_t {
                        col[b_len..NORMAL_EQ_TILE].fill(0.0);
                    }
                }
                for (b, r) in (r_block..block_end).enumerate() {
                    let x_row = &x.data[(r * d)..(r * d + d)];
                    for (j, &val) in x_row.iter().enumerate() {
                        x_t[j][b] = val;
                    }
                    let y_row = &y.data[(r * k_targets)..(r * k_targets + k_targets)];
                    for (t, &val) in y_row.iter().enumerate() {
                        y_t[t][b] = val;
                    }
                }
                accumulate_gram_upper_tile(&x_t, &x_t, &mut xt_x, p);
                for j in 0..d {
                    let xj = &x_t[j];
                    let xty_off = j * k_targets;
                    for (t, yt) in y_t.iter().enumerate() {
                        xt_y.data[xty_off + t] += dot64(xj, yt);
                    }
                    if fit_intercept {
                        xt_x.data[j * p + d] += sum64(xj);
                    }
                }
                if fit_intercept {
                    let xty_last = d * k_targets;
                    for (t, yt) in y_t.iter().enumerate() {
                        xt_y.data[xty_last + t] += sum64(yt);
                    }
                }
                r_block = block_end;
            }
            if fit_intercept {
                xt_x.data[d * p + d] = (end_row.saturating_sub(start_row)) as f64;
            }
        }
        Some(sw) => {
            let mut wx_t = vec![[0.0f64; NORMAL_EQ_TILE]; d];
            let mut w_t = [0.0f64; NORMAL_EQ_TILE];
            let mut r_block = start_row;
            while r_block < end_row {
                let block_end = (r_block + NORMAL_EQ_TILE).min(end_row);
                let b_len = block_end - r_block;
                if b_len < NORMAL_EQ_TILE {
                    for col in &mut x_t {
                        col[b_len..NORMAL_EQ_TILE].fill(0.0);
                    }
                    for col in &mut wx_t {
                        col[b_len..NORMAL_EQ_TILE].fill(0.0);
                    }
                    for col in &mut y_t {
                        col[b_len..NORMAL_EQ_TILE].fill(0.0);
                    }
                    w_t[b_len..NORMAL_EQ_TILE].fill(0.0);
                }
                for (b, r) in (r_block..block_end).enumerate() {
                    let w = sw[r];
                    w_t[b] = w;
                    let x_row = &x.data[(r * d)..(r * d + d)];
                    for (j, &val) in x_row.iter().enumerate() {
                        x_t[j][b] = val;
                        wx_t[j][b] = w * val;
                    }
                    let y_row = &y.data[(r * k_targets)..(r * k_targets + k_targets)];
                    for (t, &val) in y_row.iter().enumerate() {
                        y_t[t][b] = val;
                    }
                }
                accumulate_gram_upper_tile(&wx_t, &x_t, &mut xt_x, p);
                for j in 0..d {
                    let wxj = &wx_t[j];
                    let xty_off = j * k_targets;
                    for (t, yt) in y_t.iter().enumerate() {
                        xt_y.data[xty_off + t] += dot64(wxj, yt);
                    }
                    if fit_intercept {
                        xt_x.data[j * p + d] += sum64(wxj);
                    }
                }
                if fit_intercept {
                    let xty_last = d * k_targets;
                    for (t, yt) in y_t.iter().enumerate() {
                        xt_y.data[xty_last + t] += dot64(&w_t, yt);
                    }
                    xt_x.data[d * p + d] += sum64(&w_t);
                }
                r_block = block_end;
            }
        }
    }

    mirror_upper_to_lower(&mut xt_x, p);
    Ok((xt_x, xt_y))
}

#[inline(always)]
pub(crate) fn dot_slice_unrolled(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a_chunks, a_rem) = a[..n].as_chunks::<4>();
    let (b_chunks, b_rem) = b[..n].as_chunks::<4>();
    let mut acc = [0.0f64; 4];
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        for lane in 0..4 {
            acc[lane] += ac[lane] * bc[lane];
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (&av, &bv) in a_rem.iter().zip(b_rem.iter()) {
        sum += av * bv;
    }
    sum
}

#[inline(always)]
fn dot4_4rows(a0: &[f64], a1: &[f64], a2: &[f64], a3: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    let (a0_chunks, a0_rem) = a0.as_chunks::<4>();
    let (a1_chunks, a1_rem) = a1.as_chunks::<4>();
    let (a2_chunks, a2_rem) = a2.as_chunks::<4>();
    let (a3_chunks, a3_rem) = a3.as_chunks::<4>();
    let mut s0 = [0.0f64; 4];
    let mut s1 = [0.0f64; 4];
    let mut s2 = [0.0f64; 4];
    let mut s3 = [0.0f64; 4];
    for ((((&bc, &c0), &c1), &c2), &c3) in b_chunks
        .iter()
        .zip(a0_chunks.iter())
        .zip(a1_chunks.iter())
        .zip(a2_chunks.iter())
        .zip(a3_chunks.iter())
    {
        for lane in 0..4 {
            let bv = bc[lane];
            s0[lane] += c0[lane] * bv;
            s1[lane] += c1[lane] * bv;
            s2[lane] += c2[lane] * bv;
            s3[lane] += c3[lane] * bv;
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

struct CholeskyScratch {
    l_data: Vec<f64>,
    panel: Vec<f64>,
}

const MAX_CHOLESKY_SCRATCH_F64: usize = 2048 * 2048;

thread_local! {
    static CHOLESKY_SCRATCH: std::cell::RefCell<Option<CholeskyScratch>> =
        const { std::cell::RefCell::new(None) };
}

struct CholeskyScratchGuard {
    scratch: Option<CholeskyScratch>,
}

impl CholeskyScratchGuard {
    fn acquire(n: usize) -> Self {
        let taken = CHOLESKY_SCRATCH
            .try_with(|cell| cell.try_borrow_mut().ok().and_then(|mut opt| opt.take()))
            .ok()
            .flatten();
        let mut scratch = taken.unwrap_or_else(|| CholeskyScratch {
            l_data: Vec::new(),
            panel: Vec::new(),
        });
        let need_l = n * n;
        if scratch.l_data.len() < need_l {
            scratch.l_data.resize(need_l, 0.0);
        }
        let need_p = 4 * n;
        if scratch.panel.len() < need_p {
            scratch.panel.resize(need_p, 0.0);
        }
        Self {
            scratch: Some(scratch),
        }
    }
}

impl Drop for CholeskyScratchGuard {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take()
            && scratch.l_data.capacity() <= MAX_CHOLESKY_SCRATCH_F64
        {
            let _ = CHOLESKY_SCRATCH.try_with(|cell| {
                if let Ok(mut opt) = cell.try_borrow_mut() {
                    *opt = Some(scratch);
                }
            });
        }
    }
}

fn cholesky_lower_with_diag_shift_core(
    a: &Matrix,
    lambda: f64,
    diag_tol: f64,
    l: &mut Matrix,
    panel_buf: &mut Vec<f64>,
    zero_upper: bool,
) -> bool {
    let n = a.rows;
    if n == 0 {
        return true;
    }
    debug_assert_eq!(a.cols, n);
    debug_assert_eq!(l.rows, n);
    debug_assert_eq!(l.cols, n);

    if panel_buf.len() < 4 * n {
        panel_buf.resize(4 * n, 0.0);
    }

    for i in 0..n {
        let row_off = i * n;
        let src_row = &a.data[row_off..row_off + n];
        let dst_row = &mut l.data[row_off..row_off + n];
        dst_row[..=i].copy_from_slice(&src_row[..=i]);
        dst_row[i] += lambda;
        if zero_upper {
            dst_row[i + 1..].fill(0.0);
        }
    }

    let mut k = 0;
    while k + 4 <= n {
        let k0 = k;
        let k1 = k + 1;
        let k2 = k + 2;
        let k3 = k + 3;
        let r0 = k0 * n;
        let r1 = k1 * n;
        let r2 = k2 * n;
        let r3 = k3 * n;

        let d0 = l.data[r0 + k0];
        if !d0.is_finite() || d0 <= diag_tol {
            return false;
        }
        let l00 = d0.sqrt();
        let inv0 = 1.0 / l00;
        l.data[r0 + k0] = l00;

        let l10 = l.data[r1 + k0] * inv0;
        let l20 = l.data[r2 + k0] * inv0;
        let l30 = l.data[r3 + k0] * inv0;
        l.data[r1 + k0] = l10;
        l.data[r2 + k0] = l20;
        l.data[r3 + k0] = l30;

        let d1 = l.data[r1 + k1] - l10 * l10;
        if !d1.is_finite() || d1 <= diag_tol {
            return false;
        }
        let l11 = d1.sqrt();
        let inv1 = 1.0 / l11;
        l.data[r1 + k1] = l11;

        let l21 = (l.data[r2 + k1] - l20 * l10) * inv1;
        let l31 = (l.data[r3 + k1] - l30 * l10) * inv1;
        l.data[r2 + k1] = l21;
        l.data[r3 + k1] = l31;

        let d2 = l.data[r2 + k2] - l20 * l20 - l21 * l21;
        if !d2.is_finite() || d2 <= diag_tol {
            return false;
        }
        let l22 = d2.sqrt();
        let inv2 = 1.0 / l22;
        l.data[r2 + k2] = l22;

        let l32 = (l.data[r3 + k2] - l30 * l20 - l31 * l21) * inv2;
        l.data[r3 + k2] = l32;

        let d3 = l.data[r3 + k3] - l30 * l30 - l31 * l31 - l32 * l32;
        if !d3.is_finite() || d3 <= diag_tol {
            return false;
        }
        let l33 = d3.sqrt();
        let inv3 = 1.0 / l33;
        l.data[r3 + k3] = l33;

        let rem_start = k + 4;
        let rem = n - rem_start;
        if rem > 0 {
            let (p0, rest) = panel_buf[..4 * rem].split_at_mut(rem);
            let (p1, rest) = rest.split_at_mut(rem);
            let (p2, p3) = rest.split_at_mut(rem);

            for idx in 0..rem {
                let row_off = (rem_start + idx) * n + k;
                let a0 = l.data[row_off];
                let a1 = l.data[row_off + 1];
                let a2 = l.data[row_off + 2];
                let a3 = l.data[row_off + 3];

                let x0 = a0 * inv0;
                let x1 = (a1 - l10 * x0) * inv1;
                let x2 = (a2 - l20 * x0 - l21 * x1) * inv2;
                let x3 = (a3 - l30 * x0 - l31 * x1 - l32 * x2) * inv3;

                l.data[row_off] = x0;
                l.data[row_off + 1] = x1;
                l.data[row_off + 2] = x2;
                l.data[row_off + 3] = x3;

                p0[idx] = x0;
                p1[idx] = x1;
                p2[idx] = x2;
                p3[idx] = x3;
            }

            let trailing = &mut l.data[rem_start * n..n * n];
            let update_chunk = |(chunk_idx, rows_slice): (usize, &mut [f64])| {
                let base_idx = chunk_idx * 4;
                let n_rows_chunk = rows_slice.len() / n;
                if n_rows_chunk == 4 {
                    let (row0, rest) = rows_slice.split_at_mut(n);
                    let (row1, rest) = rest.split_at_mut(n);
                    let (row2, row3) = rest.split_at_mut(n);

                    let x00 = p0[base_idx];
                    let x01 = p1[base_idx];
                    let x02 = p2[base_idx];
                    let x03 = p3[base_idx];
                    let x10 = p0[base_idx + 1];
                    let x11 = p1[base_idx + 1];
                    let x12 = p2[base_idx + 1];
                    let x13 = p3[base_idx + 1];
                    let x20 = p0[base_idx + 2];
                    let x21 = p1[base_idx + 2];
                    let x22 = p2[base_idx + 2];
                    let x23 = p3[base_idx + 2];
                    let x30 = p0[base_idx + 3];
                    let x31 = p1[base_idx + 3];
                    let x32 = p2[base_idx + 3];
                    let x33 = p3[base_idx + 3];

                    let (r0_c, r0_r) = row0[rem_start..rem_start + base_idx].as_chunks_mut::<4>();
                    let (r1_c, r1_r) = row1[rem_start..rem_start + base_idx].as_chunks_mut::<4>();
                    let (r2_c, r2_r) = row2[rem_start..rem_start + base_idx].as_chunks_mut::<4>();
                    let (r3_c, r3_r) = row3[rem_start..rem_start + base_idx].as_chunks_mut::<4>();
                    let (p0_c, p0_r) = p0[..base_idx].as_chunks::<4>();
                    let (p1_c, p1_r) = p1[..base_idx].as_chunks::<4>();
                    let (p2_c, p2_r) = p2[..base_idx].as_chunks::<4>();
                    let (p3_c, p3_r) = p3[..base_idx].as_chunks::<4>();

                    for (((((((rc0, rc1), rc2), rc3), pc0), pc1), pc2), pc3) in r0_c
                        .iter_mut()
                        .zip(r1_c.iter_mut())
                        .zip(r2_c.iter_mut())
                        .zip(r3_c.iter_mut())
                        .zip(p0_c.iter())
                        .zip(p1_c.iter())
                        .zip(p2_c.iter())
                        .zip(p3_c.iter())
                    {
                        for lane in 0..4 {
                            let c0 = pc0[lane];
                            let c1 = pc1[lane];
                            let c2 = pc2[lane];
                            let c3 = pc3[lane];
                            rc0[lane] -= x00 * c0 + x01 * c1 + x02 * c2 + x03 * c3;
                            rc1[lane] -= x10 * c0 + x11 * c1 + x12 * c2 + x13 * c3;
                            rc2[lane] -= x20 * c0 + x21 * c1 + x22 * c2 + x23 * c3;
                            rc3[lane] -= x30 * c0 + x31 * c1 + x32 * c2 + x33 * c3;
                        }
                    }
                    for j in 0..r0_r.len() {
                        let c0 = p0_r[j];
                        let c1 = p1_r[j];
                        let c2 = p2_r[j];
                        let c3 = p3_r[j];
                        r0_r[j] -= x00 * c0 + x01 * c1 + x02 * c2 + x03 * c3;
                        r1_r[j] -= x10 * c0 + x11 * c1 + x12 * c2 + x13 * c3;
                        r2_r[j] -= x20 * c0 + x21 * c1 + x22 * c2 + x23 * c3;
                        r3_r[j] -= x30 * c0 + x31 * c1 + x32 * c2 + x33 * c3;
                    }

                    let d_off = rem_start + base_idx;
                    row0[d_off] -= x00 * x00 + x01 * x01 + x02 * x02 + x03 * x03;
                    row1[d_off] -= x10 * x00 + x11 * x01 + x12 * x02 + x13 * x03;
                    row1[d_off + 1] -= x10 * x10 + x11 * x11 + x12 * x12 + x13 * x13;
                    row2[d_off] -= x20 * x00 + x21 * x01 + x22 * x02 + x23 * x03;
                    row2[d_off + 1] -= x20 * x10 + x21 * x11 + x22 * x12 + x23 * x13;
                    row2[d_off + 2] -= x20 * x20 + x21 * x21 + x22 * x22 + x23 * x23;
                    row3[d_off] -= x30 * x00 + x31 * x01 + x32 * x02 + x33 * x03;
                    row3[d_off + 1] -= x30 * x10 + x31 * x11 + x32 * x12 + x33 * x13;
                    row3[d_off + 2] -= x30 * x20 + x31 * x21 + x32 * x22 + x33 * x23;
                    row3[d_off + 3] -= x30 * x30 + x31 * x31 + x32 * x32 + x33 * x33;
                } else {
                    for r_local in 0..n_rows_chunk {
                        let idx = base_idx + r_local;
                        let xr0 = p0[idx];
                        let xr1 = p1[idx];
                        let xr2 = p2[idx];
                        let xr3 = p3[idx];
                        let (row_c, row_r) = rows_slice
                            [r_local * n + rem_start..r_local * n + rem_start + idx + 1]
                            .as_chunks_mut::<4>();
                        let (p0_c, p0_r) = p0[..=idx].as_chunks::<4>();
                        let (p1_c, p1_r) = p1[..=idx].as_chunks::<4>();
                        let (p2_c, p2_r) = p2[..=idx].as_chunks::<4>();
                        let (p3_c, p3_r) = p3[..=idx].as_chunks::<4>();
                        for ((((rc, c0), c1), c2), c3) in row_c
                            .iter_mut()
                            .zip(p0_c.iter())
                            .zip(p1_c.iter())
                            .zip(p2_c.iter())
                            .zip(p3_c.iter())
                        {
                            for lane in 0..4 {
                                rc[lane] -= xr0 * c0[lane]
                                    + xr1 * c1[lane]
                                    + xr2 * c2[lane]
                                    + xr3 * c3[lane];
                            }
                        }
                        for j in 0..row_r.len() {
                            row_r[j] -=
                                xr0 * p0_r[j] + xr1 * p1_r[j] + xr2 * p2_r[j] + xr3 * p3_r[j];
                        }
                    }
                }
            };

            if n >= 512 && rem >= 256 {
                trailing
                    .par_chunks_mut(4 * n)
                    .enumerate()
                    .for_each(update_chunk);
            } else {
                trailing
                    .chunks_mut(4 * n)
                    .enumerate()
                    .for_each(update_chunk);
            }
        }
        k += 4;
    }

    for i in k..n {
        let r_i = i * n;
        for j in k..=i {
            let sum = dot_slice_unrolled(&l.data[r_i + k..r_i + j], &l.data[j * n + k..j * n + j]);
            if i == j {
                let diag = l.data[r_i + i] - sum;
                if !diag.is_finite() || diag <= diag_tol {
                    return false;
                }
                l.data[r_i + i] = diag.sqrt();
            } else {
                let ljj = l.data[j * n + j];
                if !ljj.is_finite() || ljj.abs() <= diag_tol {
                    return false;
                }
                l.data[r_i + j] = (l.data[r_i + j] - sum) / ljj;
            }
        }
    }

    true
}

/// 4x4 register-blocked right-looking Cholesky factorization into preallocated `l` (`n x n`).
/// Reads only the lower triangle (`j <= i`) of `a`, adds `lambda` to the diagonal, and writes
/// lower-triangular `L` such that `L * L^T = A + lambda * I` (upper triangle of `l` is zeroed).
pub(crate) fn cholesky_lower_with_diag_shift_into(
    a: &Matrix,
    lambda: f64,
    diag_tol: f64,
    l: &mut Matrix,
    panel_buf: &mut Vec<f64>,
) -> bool {
    cholesky_lower_with_diag_shift_core(a, lambda, diag_tol, l, panel_buf, true)
}

/// 4-row register-blocked forward substitution: solve `L * z = b`.
#[inline]
pub(crate) fn solve_lower_triangular_4row_into(l: &Matrix, b: &[f64], z: &mut [f64]) {
    let n = l.rows;
    let mut i = 0;
    while i + 4 <= n {
        let l0 = &l.data[i * n..(i + 1) * n];
        let l1 = &l.data[(i + 1) * n..(i + 2) * n];
        let l2 = &l.data[(i + 2) * n..(i + 3) * n];
        let l3 = &l.data[(i + 3) * n..(i + 4) * n];
        let (s0, s1, s2, s3) = dot4_4rows(&l0[..i], &l1[..i], &l2[..i], &l3[..i], &z[..i]);
        let inv0 = 1.0 / l0[i];
        let inv1 = 1.0 / l1[i + 1];
        let inv2 = 1.0 / l2[i + 2];
        let inv3 = 1.0 / l3[i + 3];
        let z0 = (b[i] - s0) * inv0;
        let z1 = (b[i + 1] - s1 - l1[i] * z0) * inv1;
        let z2 = (b[i + 2] - s2 - l2[i] * z0 - l2[i + 1] * z1) * inv2;
        let z3 = (b[i + 3] - s3 - l3[i] * z0 - l3[i + 1] * z1 - l3[i + 2] * z2) * inv3;
        z[i] = z0;
        z[i + 1] = z1;
        z[i + 2] = z2;
        z[i + 3] = z3;
        i += 4;
    }
    while i < n {
        let row_off = i * n;
        let sum = dot_slice_unrolled(&l.data[row_off..row_off + i], &z[..i]);
        z[i] = (b[i] - sum) / l.data[row_off + i];
        i += 1;
    }
}

/// 4-row register-blocked backward substitution in-place: solve `L^T * x = z`.
#[inline]
fn solve_upper_triangular_t_4row_inplace(l: &Matrix, x: &mut [f64]) {
    let n = l.rows;
    let mut rem = n;
    while rem >= 4 {
        let i = rem - 4;
        let l0 = &l.data[i * n..(i + 1) * n];
        let l1 = &l.data[(i + 1) * n..(i + 2) * n];
        let l2 = &l.data[(i + 2) * n..(i + 3) * n];
        let l3 = &l.data[(i + 3) * n..(i + 4) * n];

        let inv0 = 1.0 / l0[i];
        let inv1 = 1.0 / l1[i + 1];
        let inv2 = 1.0 / l2[i + 2];
        let inv3 = 1.0 / l3[i + 3];

        let x3 = x[i + 3] * inv3;
        let x2 = (x[i + 2] - l3[i + 2] * x3) * inv2;
        let x1 = (x[i + 1] - l3[i + 1] * x3 - l2[i + 1] * x2) * inv1;
        let x0 = (x[i] - l3[i] * x3 - l2[i] * x2 - l1[i] * x1) * inv0;
        x[i] = x0;
        x[i + 1] = x1;
        x[i + 2] = x2;
        x[i + 3] = x3;

        let (x_chunks, x_rem) = x[..i].as_chunks_mut::<4>();
        let (l0_c, l0_r) = l0[..i].as_chunks::<4>();
        let (l1_c, l1_r) = l1[..i].as_chunks::<4>();
        let (l2_c, l2_r) = l2[..i].as_chunks::<4>();
        let (l3_c, l3_r) = l3[..i].as_chunks::<4>();
        for ((((xc, c0), c1), c2), c3) in x_chunks
            .iter_mut()
            .zip(l0_c.iter())
            .zip(l1_c.iter())
            .zip(l2_c.iter())
            .zip(l3_c.iter())
        {
            for lane in 0..4 {
                xc[lane] -= c0[lane] * x0 + c1[lane] * x1 + c2[lane] * x2 + c3[lane] * x3;
            }
        }
        for k in 0..x_rem.len() {
            x_rem[k] -= l0_r[k] * x0 + l1_r[k] * x1 + l2_r[k] * x2 + l3_r[k] * x3;
        }
        rem = i;
    }
    while rem > 0 {
        let i = rem - 1;
        let row_off = i * n;
        let xi = x[i] / l.data[row_off + i];
        x[i] = xi;
        let (x_chunks, x_rem) = x[..i].as_chunks_mut::<4>();
        let (r_chunks, r_rem) = l.data[row_off..row_off + i].as_chunks::<4>();
        for (xc, rc) in x_chunks.iter_mut().zip(r_chunks.iter()) {
            for lane in 0..4 {
                xc[lane] -= rc[lane] * xi;
            }
        }
        for (xp, &rk) in x_rem.iter_mut().zip(r_rem.iter()) {
            *xp -= rk * xi;
        }
        rem = i;
    }
}

/// 4-row register-blocked backward substitution: solve `L^T * x = z` (row-major scatter-subtract).
#[inline]
#[allow(dead_code)]
pub(crate) fn solve_upper_triangular_t_4row_into(l: &Matrix, z: &[f64], x: &mut [f64]) {
    let n = l.rows;
    x[..n].copy_from_slice(&z[..n]);
    solve_upper_triangular_t_4row_inplace(l, x);
}

/// 4-row x 4-column register-blocked forward triangular solve `L * V = B` in-place on `v_block`
/// (`n x 4` row-major), returning `[||V[:, 0]||^2, ..., ||V[:, 3]||^2]`.
#[inline]
pub(crate) fn solve_lower_triangular_4col_4row_into(
    l: &Matrix,
    v_block: &mut [f64],
    n: usize,
) -> [f64; 4] {
    let mut norm_sq = [0.0f64; 4];
    let mut i = 0;
    while i + 4 <= n {
        let l0 = &l.data[i * n..(i + 1) * n];
        let l1 = &l.data[(i + 1) * n..(i + 2) * n];
        let l2 = &l.data[(i + 2) * n..(i + 3) * n];
        let l3 = &l.data[(i + 3) * n..(i + 4) * n];

        let mut s0 = [0.0f64; 4];
        let mut s1 = [0.0f64; 4];
        let mut s2 = [0.0f64; 4];
        let mut s3 = [0.0f64; 4];

        for j in 0..i {
            let vj = &v_block[j * 4..j * 4 + 4];
            let v0 = vj[0];
            let v1 = vj[1];
            let v2 = vj[2];
            let v3 = vj[3];
            let a0 = l0[j];
            let a1 = l1[j];
            let a2 = l2[j];
            let a3 = l3[j];
            s0[0] += a0 * v0;
            s0[1] += a0 * v1;
            s0[2] += a0 * v2;
            s0[3] += a0 * v3;
            s1[0] += a1 * v0;
            s1[1] += a1 * v1;
            s1[2] += a1 * v2;
            s1[3] += a1 * v3;
            s2[0] += a2 * v0;
            s2[1] += a2 * v1;
            s2[2] += a2 * v2;
            s2[3] += a2 * v3;
            s3[0] += a3 * v0;
            s3[1] += a3 * v1;
            s3[2] += a3 * v2;
            s3[3] += a3 * v3;
        }

        let inv0 = 1.0 / l0[i];
        let inv1 = 1.0 / l1[i + 1];
        let inv2 = 1.0 / l2[i + 2];
        let inv3 = 1.0 / l3[i + 3];
        let l10 = l1[i];
        let l20 = l2[i];
        let l21 = l2[i + 1];
        let l30 = l3[i];
        let l31 = l3[i + 1];
        let l32 = l3[i + 2];

        let vr = &mut v_block[i * 4..(i + 4) * 4];
        for c in 0..4 {
            let x0 = (vr[c] - s0[c]) * inv0;
            let x1 = (vr[4 + c] - s1[c] - l10 * x0) * inv1;
            let x2 = (vr[8 + c] - s2[c] - l20 * x0 - l21 * x1) * inv2;
            let x3 = (vr[12 + c] - s3[c] - l30 * x0 - l31 * x1 - l32 * x2) * inv3;
            vr[c] = x0;
            vr[4 + c] = x1;
            vr[8 + c] = x2;
            vr[12 + c] = x3;
            norm_sq[c] += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        }
        i += 4;
    }

    while i < n {
        let row_off = i * n;
        let l_row = &l.data[row_off..row_off + i];
        let mut s = [0.0f64; 4];
        for (j, &lij) in l_row.iter().enumerate() {
            let vj = &v_block[j * 4..j * 4 + 4];
            s[0] += lij * vj[0];
            s[1] += lij * vj[1];
            s[2] += lij * vj[2];
            s[3] += lij * vj[3];
        }
        let inv_diag = 1.0 / l.data[row_off + i];
        let vi = &mut v_block[i * 4..i * 4 + 4];
        for c in 0..4 {
            let x = (vi[c] - s[c]) * inv_diag;
            vi[c] = x;
            norm_sq[c] += x * x;
        }
        i += 1;
    }
    norm_sq
}

/// 4-row x 4-column register-blocked backward triangular solve `L^T * X = Z` in-place on `v_block` (`n x 4` row-major).
#[inline]
fn solve_upper_triangular_t_4col_4row_inplace(l: &Matrix, v_block: &mut [f64], n: usize) {
    let mut rem = n;
    while rem >= 4 {
        let i = rem - 4;
        let l0 = &l.data[i * n..(i + 1) * n];
        let l1 = &l.data[(i + 1) * n..(i + 2) * n];
        let l2 = &l.data[(i + 2) * n..(i + 3) * n];
        let l3 = &l.data[(i + 3) * n..(i + 4) * n];

        let inv0 = 1.0 / l0[i];
        let inv1 = 1.0 / l1[i + 1];
        let inv2 = 1.0 / l2[i + 2];
        let inv3 = 1.0 / l3[i + 3];
        let l10 = l1[i];
        let l20 = l2[i];
        let l21 = l2[i + 1];
        let l30 = l3[i];
        let l31 = l3[i + 1];
        let l32 = l3[i + 2];

        let mut x0 = [0.0f64; 4];
        let mut x1 = [0.0f64; 4];
        let mut x2 = [0.0f64; 4];
        let mut x3 = [0.0f64; 4];
        {
            let vr = &mut v_block[i * 4..(i + 4) * 4];
            for c in 0..4 {
                let v3 = vr[12 + c] * inv3;
                let v2 = (vr[8 + c] - l32 * v3) * inv2;
                let v1 = (vr[4 + c] - l31 * v3 - l21 * v2) * inv1;
                let v0 = (vr[c] - l30 * v3 - l20 * v2 - l10 * v1) * inv0;
                vr[c] = v0;
                vr[4 + c] = v1;
                vr[8 + c] = v2;
                vr[12 + c] = v3;
                x0[c] = v0;
                x1[c] = v1;
                x2[c] = v2;
                x3[c] = v3;
            }
        }

        let l0_p = &l0[..i];
        let l1_p = &l1[..i];
        let l2_p = &l2[..i];
        let l3_p = &l3[..i];
        let vp = &mut v_block[..i * 4];
        for k in 0..i {
            let a0 = l0_p[k];
            let a1 = l1_p[k];
            let a2 = l2_p[k];
            let a3 = l3_p[k];
            let vk = &mut vp[k * 4..k * 4 + 4];
            vk[0] -= a0 * x0[0] + a1 * x1[0] + a2 * x2[0] + a3 * x3[0];
            vk[1] -= a0 * x0[1] + a1 * x1[1] + a2 * x2[1] + a3 * x3[1];
            vk[2] -= a0 * x0[2] + a1 * x1[2] + a2 * x2[2] + a3 * x3[2];
            vk[3] -= a0 * x0[3] + a1 * x1[3] + a2 * x2[3] + a3 * x3[3];
        }
        rem = i;
    }
    while rem > 0 {
        let i = rem - 1;
        let row_off = i * n;
        let inv_diag = 1.0 / l.data[row_off + i];
        let mut xi = [0.0f64; 4];
        {
            let vi = &mut v_block[i * 4..i * 4 + 4];
            for c in 0..4 {
                let val = vi[c] * inv_diag;
                vi[c] = val;
                xi[c] = val;
            }
        }
        let row_i = &l.data[row_off..row_off + i];
        let vp = &mut v_block[..i * 4];
        for (k, &lik) in row_i.iter().enumerate() {
            let vk = &mut vp[k * 4..k * 4 + 4];
            vk[0] -= lik * xi[0];
            vk[1] -= lik * xi[1];
            vk[2] -= lik * xi[2];
            vk[3] -= lik * xi[3];
        }
        rem = i;
    }
}

/// Solve `(L * L^T) * Theta = B` using 4-column blocked forward/backward triangular solves.
pub(crate) fn solve_cholesky_multi_from_l(l: &Matrix, b: &Matrix) -> Matrix {
    let n = l.rows;
    let k_targets = b.cols;
    if n == 0 || k_targets == 0 {
        return Matrix::zeros(n, k_targets);
    }

    if k_targets == 1 {
        let mut theta = Matrix::zeros(n, 1);
        solve_lower_triangular_4row_into(l, &b.data, &mut theta.data);
        solve_upper_triangular_t_4row_inplace(l, &mut theta.data);
        return theta;
    }

    let mut theta = Matrix::zeros(n, k_targets);
    let n_blocks = k_targets.div_ceil(4);
    let solve_block = |blk: usize, v_block: &mut [f64]| {
        let t_start = blk * 4;
        let bs = (k_targets - t_start).min(4);
        if bs == 4 {
            for r in 0..n {
                let src = &b.data[r * k_targets + t_start..r * k_targets + t_start + 4];
                v_block[r * 4..r * 4 + 4].copy_from_slice(src);
            }
        } else {
            for r in 0..n {
                let src = &b.data[r * k_targets + t_start..r * k_targets + t_start + bs];
                let dst = &mut v_block[r * 4..r * 4 + 4];
                dst[..bs].copy_from_slice(src);
                dst[bs..].fill(0.0);
            }
        }
        solve_lower_triangular_4col_4row_into(l, v_block, n);
        solve_upper_triangular_t_4col_4row_inplace(l, v_block, n);
    };

    if n_blocks >= 2 && n >= 128 {
        let solved_blocks: Vec<(usize, Vec<f64>)> = (0..n_blocks)
            .into_par_iter()
            .map(|blk| {
                let mut v_block = vec![0.0; n * 4];
                solve_block(blk, &mut v_block);
                (blk, v_block)
            })
            .collect();
        for (blk, v_block) in solved_blocks {
            let t_start = blk * 4;
            let bs = (k_targets - t_start).min(4);
            for r in 0..n {
                let dst = &mut theta.data[r * k_targets + t_start..r * k_targets + t_start + bs];
                dst.copy_from_slice(&v_block[r * 4..r * 4 + bs]);
            }
        }
    } else {
        let mut v_block = vec![0.0; n * 4];
        for blk in 0..n_blocks {
            solve_block(blk, &mut v_block);
            let t_start = blk * 4;
            let bs = (k_targets - t_start).min(4);
            for r in 0..n {
                let dst = &mut theta.data[r * k_targets + t_start..r * k_targets + t_start + bs];
                dst.copy_from_slice(&v_block[r * 4..r * 4 + bs]);
            }
        }
    }

    theta
}

// Solve symmetric positive semi-definite linear system (A * theta = b) using Cholesky Decomposition (L * L^T)
pub fn solve_cholesky(a: &Matrix, b: &[f64]) -> OchreResult<Vec<f64>> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    if a.rows != b.len() {
        return Err(OchreError::VectorLengthMismatch);
    }
    let n = a.rows;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut guard = CholeskyScratchGuard::acquire(n);
    if let Some(scratch) = guard.scratch.as_mut() {
        let l_data = std::mem::take(&mut scratch.l_data);
        let mut l = Matrix {
            rows: n,
            cols: n,
            data: l_data,
        };
        let mut solved = false;
        for &lambda in &[0.0, 1e-12, 1e-9, 1e-6, 1e-3] {
            if cholesky_lower_with_diag_shift_core(
                a,
                lambda,
                1e-14,
                &mut l,
                &mut scratch.panel,
                false,
            ) {
                solved = true;
                break;
            }
        }
        let res = if solved {
            let mut theta = vec![0.0; n];
            solve_lower_triangular_4row_into(&l, b, &mut theta);
            solve_upper_triangular_t_4row_inplace(&l, &mut theta);
            Some(theta)
        } else {
            None
        };
        scratch.l_data = std::mem::take(&mut l.data);
        if let Some(theta) = res {
            return Ok(theta);
        }
    }

    // Fallback to Gauss-Jordan elimination if matrix is severely rank-deficient
    solve_linear_system(a, b)
}

// Solve multi-target symmetric positive semi-definite linear system (A * Theta = B)
// Factorizes Cholesky L * L^T ONCE in O(1/3 * p^3) and solves K targets in O(K * p^2) (*≧ω≦*)
pub fn solve_cholesky_multi(a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    if a.rows != b.rows {
        return Err(OchreError::MatrixRowMismatch);
    }
    let n = a.rows;
    let k_targets = b.cols;
    if n == 0 || k_targets == 0 {
        return Ok(Matrix::zeros(n, k_targets));
    }

    let mut guard = CholeskyScratchGuard::acquire(n);
    if let Some(scratch) = guard.scratch.as_mut() {
        let l_data = std::mem::take(&mut scratch.l_data);
        let mut l = Matrix {
            rows: n,
            cols: n,
            data: l_data,
        };
        let mut solved = false;
        for &lambda in &[0.0, 1e-12, 1e-9, 1e-6, 1e-3] {
            if cholesky_lower_with_diag_shift_core(
                a,
                lambda,
                1e-14,
                &mut l,
                &mut scratch.panel,
                false,
            ) {
                solved = true;
                break;
            }
        }
        let res = if solved {
            Some(solve_cholesky_multi_from_l(&l, b))
        } else {
            None
        };
        scratch.l_data = std::mem::take(&mut l.data);
        if let Some(theta) = res {
            return Ok(theta);
        }
    }

    // Fallback: solve each column via Gauss-Jordan elimination
    let mut theta = Matrix::zeros(n, k_targets);
    let mut col_b = vec![0.0; n];
    for t in 0..k_targets {
        for i in 0..n {
            col_b[i] = b[(i, t)];
        }
        let col_sol = solve_linear_system(a, &col_b)?;
        for i in 0..n {
            theta[(i, t)] = col_sol[i];
        }
    }
    Ok(theta)
}

/// Invert a symmetric positive semi-definite matrix A (P x P) using multi-target Cholesky solver.
pub fn invert_spd_cholesky(a: &Matrix) -> OchreResult<Matrix> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    let id = Matrix::identity(a.rows);
    solve_cholesky_multi(a, &id)
}

// Solve linear system A * x = b using Gauss-Jordan elimination with partial pivoting
pub fn solve_linear_system(a: &Matrix, b: &[f64]) -> OchreResult<Vec<f64>> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    if a.rows != b.len() {
        return Err(OchreError::VectorLengthMismatch);
    }

    let n = a.rows;
    if n == 0 {
        return Ok(Vec::new());
    }

    match try_gauss_jordan(a, b) {
        Ok(solution) => Ok(solution),
        Err(_) => {
            // Apply Tikhonov ridge regularization if determinant is close to zero
            let mut regularized_a = a.clone();
            regularized_a.add_ridge_diagonal(1e-7);
            try_gauss_jordan(&regularized_a, b)
                .map_err(|e| OchreError::SolverRegularizationFailed(e.to_string()))
        }
    }
}

fn try_gauss_jordan(a: &Matrix, b: &[f64]) -> OchreResult<Vec<f64>> {
    let n = a.rows;
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[(r, c)];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut pivot_row = col;
        for r in (col + 1)..n {
            let val = aug[r * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = r;
            }
        }

        if max_val < 1e-12 {
            return Err(OchreError::SingularMatrix(
                "pivot absolute value < 1e-12".to_string(),
            ));
        }

        if pivot_row != col {
            for c in 0..(n + 1) {
                let temp = aug[col * (n + 1) + c];
                aug[col * (n + 1) + c] = aug[pivot_row * (n + 1) + c];
                aug[pivot_row * (n + 1) + c] = temp;
            }
        }

        let pivot = aug[col * (n + 1) + col];
        for c in col..(n + 1) {
            aug[col * (n + 1) + c] /= pivot;
        }

        for r in 0..n {
            if r != col {
                let factor = aug[r * (n + 1) + col];
                if factor.abs() > 1e-15 {
                    for c in col..(n + 1) {
                        let sub = factor * aug[col * (n + 1) + c];
                        aug[r * (n + 1) + c] -= sub;
                    }
                }
            }
        }
    }

    let mut result = vec![0.0; n];
    for r in 0..n {
        result[r] = aug[r * (n + 1) + n];
    }
    Ok(result)
}

// Calculate mean value of a slice
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sum: f64 = v.iter().sum();
    sum / (v.len() as f64)
}

// Calculate coefficient of determination R^2
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }
    let y_mean = mean(y_true);
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        ss_tot += (yt - y_mean).powi(2);
        ss_res += (yt - yp).powi(2);
    }
    if ss_tot.abs() < 1e-12 {
        return 1.0;
    }
    1.0 - (ss_res / ss_tot)
}

// Calculate multi-output weighted R^2 score (uniform_average matching Scikit-Learn) (＾▽＾)
pub fn r2_score_multi(y_true: &Matrix, y_pred: &Matrix, sample_weight: Option<&[f64]>) -> f64 {
    if y_true.rows != y_pred.rows
        || y_true.cols != y_pred.cols
        || y_true.rows == 0
        || y_true.cols == 0
    {
        return 0.0;
    }
    let rows = y_true.rows;
    let cols = y_true.cols;

    let mut total_r2 = 0.0;

    for t in 0..cols {
        let mut w_sum = 0.0;
        let mut wy_sum = 0.0;

        match sample_weight {
            Some(sw) => {
                for r in 0..rows {
                    let w = sw[r];
                    w_sum += w;
                    wy_sum += w * y_true[(r, t)];
                }
            }
            None => {
                w_sum = rows as f64;
                for r in 0..rows {
                    wy_sum += y_true[(r, t)];
                }
            }
        }

        if w_sum <= 0.0 {
            continue;
        }
        let y_mean = wy_sum / w_sum;

        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        match sample_weight {
            Some(sw) => {
                for r in 0..rows {
                    let w = sw[r];
                    let yt = y_true[(r, t)];
                    let yp = y_pred[(r, t)];
                    ss_tot += w * (yt - y_mean).powi(2);
                    ss_res += w * (yt - yp).powi(2);
                }
            }
            None => {
                for r in 0..rows {
                    let yt = y_true[(r, t)];
                    let yp = y_pred[(r, t)];
                    ss_tot += (yt - y_mean).powi(2);
                    ss_res += (yt - yp).powi(2);
                }
            }
        }

        let r2_t = if ss_tot.abs() < 1e-12 {
            if ss_res.abs() < 1e-12 { 1.0 } else { 0.0 }
        } else {
            1.0 - (ss_res / ss_tot)
        };
        total_r2 += r2_t;
    }

    total_r2 / (cols as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_operations() {
        let m1 = Matrix::from_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let m2 = Matrix::from_2d(&vec![vec![2.0, 0.0], vec![1.0, 2.0]]).unwrap();

        let mul = m1.matmul(&m2).unwrap();
        assert_eq!(mul[(0, 0)], 4.0);
        assert_eq!(mul[(0, 1)], 4.0);
        assert_eq!(mul[(1, 0)], 10.0);
        assert_eq!(mul[(1, 1)], 8.0);
    }

    #[test]
    fn test_linear_solver() {
        let a = Matrix::from_2d(&vec![vec![2.0, 1.0], vec![1.0, 3.0]]).unwrap();
        let b = vec![5.0, 5.0];
        let sol = solve_linear_system(&a, &b).unwrap();
        assert!((sol[0] - 2.0).abs() < 1e-6);
        assert!((sol[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky_solver() {
        // Symmetric positive definite matrix: [4, 2; 2, 3] * [x; y] = [8, 7] -> sol = [1.25, 1.5]
        let a = Matrix::from_2d(&vec![vec![4.0, 2.0], vec![2.0, 3.0]]).unwrap();
        let b = vec![8.0, 7.0];
        let sol = solve_cholesky(&a, &b).unwrap();
        assert!((sol[0] - 1.25).abs() < 1e-6);
        assert!((sol[1] - 1.50).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_from_vec_and_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = Matrix::from_slice(2, 2, &data).unwrap();
        let m2 = Matrix::from_vec(2, 2, data).unwrap();
        assert_eq!(m1, m2);
        assert_eq!(m1[(0, 1)], 2.0);
        assert_eq!(m1[(1, 0)], 3.0);
    }

    #[test]
    fn test_matrix_indexing() {
        let mut m = Matrix::zeros(2, 2);
        m[(0, 0)] = 5.0;
        m[(0, 1)] = 3.0;
        m[(1, 0)] = 2.0;
        m[(1, 1)] = 8.0;

        assert_eq!(m[(0, 0)], 5.0);
        assert_eq!(m.get(0, 1), 3.0);
        m[(0, 0)] += 10.0;
        assert_eq!(m[(0, 0)], 15.0);
    }

    #[test]
    fn test_compute_normal_equation_mats() {
        let x = Matrix::from_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        // 1. CPU execution
        let (xt_x_cpu, xt_y_cpu) =
            compute_normal_equation_mats(&x, &y, true, &crate::device::DeviceType::Cpu).unwrap();
        assert_eq!(xt_x_cpu.rows, 3);
        assert_eq!(xt_x_cpu.cols, 3);
        assert_eq!(xt_y_cpu.len(), 3);

        // Verify with manual calculation
        assert_eq!(xt_x_cpu[(0, 0)], 35.0);
        assert_eq!(xt_x_cpu[(0, 1)], 44.0);
        assert_eq!(xt_x_cpu[(1, 0)], 44.0);
        assert_eq!(xt_x_cpu[(0, 2)], 9.0);
        assert_eq!(xt_x_cpu[(2, 0)], 9.0);
        assert_eq!(xt_x_cpu[(2, 2)], 3.0);
        assert_eq!(xt_y_cpu[0], 22.0);
        assert_eq!(xt_y_cpu[2], 6.0);

        // 2. Multi-GPU data-parallel execution
        let multi_gpu = crate::device::DeviceType::MultiGpu(vec![0, 1]);
        let (xt_x_mgpu, xt_y_mgpu) =
            compute_normal_equation_mats(&x, &y, true, &multi_gpu).unwrap();
        assert_eq!(xt_x_mgpu, xt_x_cpu);
        assert_eq!(xt_y_mgpu, xt_y_cpu);

        // 3. TPU block systolic execution
        let tpu = crate::device::DeviceType::Tpu(0);
        let (xt_x_tpu, xt_y_tpu) = compute_normal_equation_mats(&x, &y, true, &tpu).unwrap();
        assert_eq!(xt_x_tpu, xt_x_cpu);
        assert_eq!(xt_y_tpu, xt_y_cpu);
    }

    #[test]
    fn test_blocked4_cholesky_and_multi_280() {
        let n = 280usize;
        let p = 16usize;
        let mut x = Matrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x[(i, j)] = (((i * 7 + j * 13) % 97) as f64) * 0.03 - 1.2;
            }
        }
        // Build SPD matrix A = K_rbf + 0.1 * I
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 1.1;
            for j in 0..i {
                let mut d2 = 0.0;
                for c in 0..p {
                    let d = x[(i, c)] - x[(j, c)];
                    d2 += d * d;
                }
                let val = (-0.2 * d2).exp();
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        // Single RHS solve_cholesky
        let b1: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.17).sin()).collect();
        let sol1 = solve_cholesky(&a, &b1).unwrap();
        let ax1 = a.matvec(&sol1).unwrap();
        let max_err1 = ax1
            .iter()
            .zip(b1.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err1 < 1e-10, "max_err1 = {max_err1}");

        // Multi-target solve_cholesky_multi with 7 columns (exercises 4-col + 3-col remainder)
        let k_cols = 7usize;
        let mut b_multi = Matrix::zeros(n, k_cols);
        for i in 0..n {
            for t in 0..k_cols {
                b_multi[(i, t)] = (((i + 1) * (t + 3)) as f64 * 0.11).cos();
            }
        }
        let theta = solve_cholesky_multi(&a, &b_multi).unwrap();
        let a_theta = a.matmul(&theta).unwrap();
        let max_err_m = a_theta
            .data
            .iter()
            .zip(b_multi.data.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err_m < 1e-10, "max_err_m = {max_err_m}");
    }
}
