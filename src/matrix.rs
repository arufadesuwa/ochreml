// OchreML Linear Algebra Module (＾▽＾)
// Provides fundamental matrix structures, trait-based indexing, and linear equation solvers.

use crate::error::{OchreError, OchreResult};
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
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a_rk = self[(r, k)];
                for c in 0..other.cols {
                    res[(r, c)] += a_rk * other[(k, c)];
                }
            }
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

        let mut res = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self[(r, c)] * vec[c];
            }
            res[r] = sum;
        }
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

// Accumulate upper triangle of left_t * right_t^T into xt_x (stride p) using 2x4 SIMD micro-kernel.
#[inline]
fn accumulate_gram_upper_tile(
    left_t: &[[f64; NORMAL_EQ_TILE]],
    right_t: &[[f64; NORMAL_EQ_TILE]],
    xt_x: &mut Matrix,
    p: usize,
) {
    let d = left_t.len();
    let mut j = 0;
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
            for (j, col) in x_t.iter_mut().enumerate().take(d) {
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
        for (j, col) in x_t.iter().enumerate().take(d) {
            xt_y[j] += dot64(col, &y_t);
            if fit_intercept {
                xt_x.data[j * p + d] += sum64(col);
            }
        }
        if fit_intercept {
            xt_y[d] += sum64(&y_t);
        }
        r_block = block_end;
    }

    if fit_intercept {
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

    // Try Cholesky decomposition directly or with subtle Ridge regularizer if ill-conditioned
    for &lambda in &[0.0, 1e-12, 1e-9, 1e-6, 1e-3] {
        let mut l = Matrix::zeros(n, n);
        let mut success = true;

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    let diag = a[(i, i)] + lambda - sum;
                    if diag <= 1e-14 {
                        success = false;
                        break;
                    }
                    l[(i, i)] = diag.sqrt();
                } else {
                    let lj_j = l[(j, j)];
                    if lj_j.abs() <= 1e-14 {
                        success = false;
                        break;
                    }
                    l[(i, j)] = (a[(i, j)] - sum) / lj_j;
                }
            }
            if !success {
                break;
            }
        }

        if success {
            // Forward substitution: L * z = b
            let mut z = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l[(i, k)] * z[k];
                }
                z[i] = (b[i] - sum) / l[(i, i)];
            }

            // Backward substitution: L^T * theta = z
            let mut theta = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = 0.0;
                for k in (i + 1)..n {
                    sum += l[(k, i)] * theta[k];
                }
                theta[i] = (z[i] - sum) / l[(i, i)];
            }

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

    for &lambda in &[0.0, 1e-12, 1e-9, 1e-6, 1e-3] {
        let mut l = Matrix::zeros(n, n);
        let mut success = true;

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    let diag = a[(i, i)] + lambda - sum;
                    if diag <= 1e-14 {
                        success = false;
                        break;
                    }
                    l[(i, i)] = diag.sqrt();
                } else {
                    let lj_j = l[(j, j)];
                    if lj_j.abs() <= 1e-14 {
                        success = false;
                        break;
                    }
                    l[(i, j)] = (a[(i, j)] - sum) / lj_j;
                }
            }
            if !success {
                break;
            }
        }

        if success {
            let mut theta = Matrix::zeros(n, k_targets);
            let mut z = vec![0.0; n];

            for t in 0..k_targets {
                // Forward substitution: L * z = B[:, t]
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..i {
                        sum += l[(i, k)] * z[k];
                    }
                    z[i] = (b[(i, t)] - sum) / l[(i, i)];
                }

                // Backward substitution: L^T * theta[:, t] = z
                for i in (0..n).rev() {
                    let mut sum = 0.0;
                    for k in (i + 1)..n {
                        sum += l[(k, i)] * theta[(k, t)];
                    }
                    theta[(i, t)] = (z[i] - sum) / l[(i, i)];
                }
            }

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
}
