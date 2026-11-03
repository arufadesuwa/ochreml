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

// Compute (X^T * X) and (X^T * y) over a slice range [start_row..end_row] for 1D target (Zero-allocation scalar loop)
#[inline]
pub fn compute_normal_equation_range(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = vec![0.0; p];

    for r in start_row..end_row {
        let yr = y[r];
        let row_offset = r * x.cols;
        let row_slice = &x.data[row_offset..row_offset + x.cols];

        if fit_intercept {
            for j in 0..x.cols {
                let xj = row_slice[j];
                xt_y[j] += xj * yr;
                let xt_x_row = j * p;
                for k in j..x.cols {
                    xt_x.data[xt_x_row + k] += xj * row_slice[k];
                }
                xt_x.data[xt_x_row + x.cols] += xj;
            }
            xt_y[x.cols] += yr;
            xt_x.data[x.cols * p + x.cols] += 1.0;
        } else {
            for j in 0..x.cols {
                let xj = row_slice[j];
                xt_y[j] += xj * yr;
                let xt_x_row = j * p;
                for k in j..x.cols {
                    xt_x.data[xt_x_row + k] += xj * row_slice[k];
                }
            }
        }
    }

    // Mirror upper triangle to lower triangle
    for j in 0..p {
        for k in (j + 1)..p {
            let val = xt_x.data[j * p + k];
            xt_x.data[k * p + j] = val;
        }
    }

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

// Compute (X^T * W * X) and (X^T * W * Y) over row slice range [start_row..end_row]
pub fn compute_normal_equation_multi_range(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Matrix)> {
    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let k_targets = y.cols;
    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = Matrix::zeros(p, k_targets);

    match sample_weight {
        Some(sw) => {
            for r in start_row..end_row {
                let w = sw[r];
                if w == 0.0 {
                    continue;
                }
                let row_offset = r * x.cols;
                let row_slice = &x.data[row_offset..row_offset + x.cols];
                let y_offset = r * k_targets;
                let y_slice = &y.data[y_offset..y_offset + k_targets];

                for j in 0..x.cols {
                    let w_xj = w * row_slice[j];
                    let xt_y_row = j * k_targets;
                    for t in 0..k_targets {
                        xt_y.data[xt_y_row + t] += w_xj * y_slice[t];
                    }
                    let xt_x_row = j * p;
                    for k in j..x.cols {
                        xt_x.data[xt_x_row + k] += w_xj * row_slice[k];
                    }
                    if fit_intercept {
                        xt_x.data[xt_x_row + x.cols] += w_xj;
                    }
                }
                if fit_intercept {
                    let xt_y_last = x.cols * k_targets;
                    for t in 0..k_targets {
                        xt_y.data[xt_y_last + t] += w * y_slice[t];
                    }
                    xt_x[(x.cols, x.cols)] += w;
                }
            }
        }
        None => {
            for r in start_row..end_row {
                let row_offset = r * x.cols;
                let row_slice = &x.data[row_offset..row_offset + x.cols];
                let y_offset = r * k_targets;
                let y_slice = &y.data[y_offset..y_offset + k_targets];

                for j in 0..x.cols {
                    let xj = row_slice[j];
                    let xt_y_row = j * k_targets;
                    for t in 0..k_targets {
                        xt_y.data[xt_y_row + t] += xj * y_slice[t];
                    }
                    let xt_x_row = j * p;
                    for k in j..x.cols {
                        xt_x.data[xt_x_row + k] += xj * row_slice[k];
                    }
                    if fit_intercept {
                        xt_x.data[xt_x_row + x.cols] += xj;
                    }
                }
                if fit_intercept {
                    let xt_y_last = x.cols * k_targets;
                    for t in 0..k_targets {
                        xt_y.data[xt_y_last + t] += y_slice[t];
                    }
                    xt_x[(x.cols, x.cols)] += 1.0;
                }
            }
        }
    }

    // Mirror upper triangle to lower triangle
    for j in 0..p {
        for k in (j + 1)..p {
            let val = xt_x[(j, k)];
            xt_x[(k, j)] = val;
        }
    }

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
            try_gauss_jordan(&regularized_a, b).map_err(|e| {
                OchreError::SolverRegularizationFailed(e.to_string())
            })
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
            return Err(OchreError::SingularMatrix("pivot absolute value < 1e-12".to_string()));
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
    if y_true.rows != y_pred.rows || y_true.cols != y_pred.cols || y_true.rows == 0 || y_true.cols == 0 {
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
        let m1 = Matrix::from_2d(&vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        let m2 = Matrix::from_2d(&vec![
            vec![2.0, 0.0],
            vec![1.0, 2.0],
        ]).unwrap();

        let mul = m1.matmul(&m2).unwrap();
        assert_eq!(mul[(0, 0)], 4.0);
        assert_eq!(mul[(0, 1)], 4.0);
        assert_eq!(mul[(1, 0)], 10.0);
        assert_eq!(mul[(1, 1)], 8.0);
    }

    #[test]
    fn test_linear_solver() {
        let a = Matrix::from_2d(&vec![
            vec![2.0, 1.0],
            vec![1.0, 3.0],
        ]).unwrap();
        let b = vec![5.0, 5.0];
        let sol = solve_linear_system(&a, &b).unwrap();
        assert!((sol[0] - 2.0).abs() < 1e-6);
        assert!((sol[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky_solver() {
        // Symmetric positive definite matrix: [4, 2; 2, 3] * [x; y] = [8, 7] -> sol = [1.25, 1.5]
        let a = Matrix::from_2d(&vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0],
        ]).unwrap();
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
        let x = Matrix::from_2d(&vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        // 1. CPU execution
        let (xt_x_cpu, xt_y_cpu) = compute_normal_equation_mats(&x, &y, true, &crate::device::DeviceType::Cpu).unwrap();
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
        let (xt_x_mgpu, xt_y_mgpu) = compute_normal_equation_mats(&x, &y, true, &multi_gpu).unwrap();
        assert_eq!(xt_x_mgpu, xt_x_cpu);
        assert_eq!(xt_y_mgpu, xt_y_cpu);

        // 3. TPU block systolic execution
        let tpu = crate::device::DeviceType::Tpu(0);
        let (xt_x_tpu, xt_y_tpu) = compute_normal_equation_mats(&x, &y, true, &tpu).unwrap();
        assert_eq!(xt_x_tpu, xt_x_cpu);
        assert_eq!(xt_y_tpu, xt_y_cpu);
    }
}
