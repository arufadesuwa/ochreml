// OchreML Linear Algebra Module (＾▽＾)
// Provides fundamental matrix structures and linear equation solvers.

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
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
            mat.set(i, i, 1.0);
        }
        mat
    }

    // Convert 2D vector data into a contiguous Matrix structure
    pub fn from_2d(data_2d: &[Vec<f64>]) -> Result<Self, String> {
        let rows = data_2d.len();
        if rows == 0 {
            return Err("Matrix cannot be empty (´；ω；`)".to_string());
        }
        let cols = data_2d[0].len();
        if cols == 0 {
            return Err("Matrix column count cannot be zero ( >_< )".to_string());
        }

        let mut flat = Vec::with_capacity(rows * cols);
        for (r, row) in data_2d.iter().enumerate() {
            if row.len() != cols {
                return Err(format!(
                    "Row {} length ({}) does not match first column count ({}) (・`ω´・)",
                    r,
                    row.len(),
                    cols
                ));
            }
            flat.extend_from_slice(row);
        }

        Ok(Self {
            rows,
            cols,
            data: flat,
        })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    // Matrix transpose: swap rows and columns
    pub fn transpose(&self) -> Self {
        let mut res = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                res.set(c, r, self.get(r, c));
            }
        }
        res
    }

    // Matrix multiplication: A * B
    pub fn matmul(&self, other: &Matrix) -> Result<Self, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Incompatible matrix dimensions for multiplication: ({}x{}) and ({}x{}) (´-ω-`)",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        let mut res = Self::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a_rk = self.get(r, k);
                for c in 0..other.cols {
                    let old_val = res.get(r, c);
                    res.set(r, c, old_val + a_rk * other.get(k, c));
                }
            }
        }
        Ok(res)
    }

    // Matrix-vector multiplication: A * v
    pub fn matvec(&self, vec: &[f64]) -> Result<Vec<f64>, String> {
        if self.cols != vec.len() {
            return Err(format!(
                "Matrix column count ({}) must match vector length ({}) (・`ω´・)",
                self.cols,
                vec.len()
            ));
        }

        let mut res = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c) * vec[c];
            }
            res[r] = sum;
        }
        Ok(res)
    }

    // Add regularization lambda to main diagonal for Tikhonov (Ridge) regularization
    pub fn add_ridge_diagonal(&mut self, lambda: f64) {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            let val = self.get(i, i);
            self.set(i, i, val + lambda);
        }
    }
}

// Directly compute (X^T * X) and (X^T * y) in a single pass without allocating transpose matrices.
// Exploits symmetry of the Gram matrix and contiguous row memory layout (*≧ω≦*)
pub fn compute_normal_equation_mats(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
) -> Result<(Matrix, Vec<f64>), String> {
    if x.rows != y.len() {
        return Err(format!(
            "Row count ({}) must match target length ({}) ( >_< )",
            x.rows,
            y.len()
        ));
    }
    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = vec![0.0; p];

    for r in 0..x.rows {
        let yr = y[r];
        let row_offset = r * x.cols;
        let row_slice = &x.data[row_offset..row_offset + x.cols];

        if fit_intercept {
            for j in 0..x.cols {
                let xj = row_slice[j];
                xt_y[j] += xj * yr;
                for k in j..x.cols {
                    xt_x.data[j * p + k] += xj * row_slice[k];
                }
                xt_x.data[j * p + x.cols] += xj;
            }
            xt_y[x.cols] += yr;
            xt_x.data[x.cols * p + x.cols] += 1.0;
        } else {
            for j in 0..x.cols {
                let xj = row_slice[j];
                xt_y[j] += xj * yr;
                for k in j..x.cols {
                    xt_x.data[j * p + k] += xj * row_slice[k];
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

// Solve linear system A * x = b using Gauss-Jordan elimination with partial pivoting
// Equipped with Tikhonov (Ridge) regularizer fallback when near-singular (*≧ω≦*)
pub fn solve_linear_system(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    if a.rows != a.cols {
        return Err("Coefficient matrix A must be square (n x n) (´；ω；`)".to_string());
    }
    if a.rows != b.len() {
        return Err("Constant vector b length must match row count of matrix A ( >_< )".to_string());
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
                format!("Failed to solve linear system after numerical regularization: {} (T_T)", e)
            })
        }
    }
}

fn try_gauss_jordan(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    let n = a.rows;
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a.get(r, c);
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
            return Err("Matrix is singular or near-singular at pivot (´-ω-`)".to_string());
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
        assert_eq!(mul.get(0, 0), 4.0);
        assert_eq!(mul.get(0, 1), 4.0);
        assert_eq!(mul.get(1, 0), 10.0);
        assert_eq!(mul.get(1, 1), 8.0);
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
    fn test_compute_normal_equation_mats() {
        let x = Matrix::from_2d(&vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        let (xt_x, xt_y) = compute_normal_equation_mats(&x, &y, true).unwrap();
        assert_eq!(xt_x.rows, 3);
        assert_eq!(xt_x.cols, 3);
        assert_eq!(xt_y.len(), 3);

        // Verify with manual calculation
        // Column 0: [1, 3, 5] -> norm^2 = 1+9+25 = 35
        assert_eq!(xt_x.get(0, 0), 35.0);
        // Column 0 dot Col 1: 1*2 + 3*4 + 5*6 = 2 + 12 + 30 = 44
        assert_eq!(xt_x.get(0, 1), 44.0);
        assert_eq!(xt_x.get(1, 0), 44.0);
        // Col 0 dot bias: 1 + 3 + 5 = 9
        assert_eq!(xt_x.get(0, 2), 9.0);
        assert_eq!(xt_x.get(2, 0), 9.0);
        // Bias dot bias: 3.0
        assert_eq!(xt_x.get(2, 2), 3.0);
        // Col 0 dot y: 1*1 + 3*2 + 5*3 = 1 + 6 + 15 = 22
        assert_eq!(xt_y[0], 22.0);
        // Bias dot y: 1 + 2 + 3 = 6
        assert_eq!(xt_y[2], 6.0);
    }
}
