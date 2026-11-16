// OchreML Primitive N5: small_sym_eig_svd & gram_svd (*≧ω≦*)
// Cyclic Jacobi eigensolver for symmetric P x P matrices + hardware-accelerated Gram SVD.

use crate::backend::Backend;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;

/// Result of symmetric eigendecomposition `M = V * diag(eigenvalues) * V^T`.
#[derive(Clone, Debug)]
pub struct SymEigResult {
    /// Eigenvalues sorted in descending order.
    pub eigenvalues: Vec<f64>,
    /// Orthonormal eigenvectors stored column-major inside row-major `(p x p)` matrix
    /// (column `k` is eigenvector `k`).
    pub eigenvectors: Matrix,
}

fn jacobi_rotate(a: &mut [f64], v: &mut [f64], p: usize, row: usize, col: usize) {
    let a_pp = a[row * p + row];
    let a_qq = a[col * p + col];
    let a_pq = a[row * p + col];
    if a_pq.abs() <= 1e-15 * (a_pp.abs() + a_qq.abs() + 1e-30) {
        return;
    }

    let tau = (a_qq - a_pp) / (2.0 * a_pq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    let rho = s / (1.0 + c);

    a[row * p + row] = a_pp - t * a_pq;
    a[col * p + col] = a_qq + t * a_pq;
    a[row * p + col] = 0.0;
    a[col * p + row] = 0.0;

    for r in 0..p {
        if r == row || r == col {
            continue;
        }
        let a_rp = a[r * p + row];
        let a_rq = a[r * p + col];
        let new_rp = a_rp - s * (a_rq + rho * a_rp);
        let new_rq = a_rq + s * (a_rp - rho * a_rq);
        a[r * p + row] = new_rp;
        a[row * p + r] = new_rp;
        a[r * p + col] = new_rq;
        a[col * p + r] = new_rq;
    }

    for r in 0..p {
        let v_rp = v[r * p + row];
        let v_rq = v[r * p + col];
        v[r * p + row] = c * v_rp - s * v_rq;
        v[r * p + col] = s * v_rp + c * v_rq;
    }
}

/// Compute symmetric eigendecomposition `M = V * diag(lambda) * V^T` via Cyclic Jacobi.
pub fn cpu_small_sym_eig_svd(sym: &Matrix) -> OchreResult<SymEigResult> {
    if sym.rows == 0 || sym.rows != sym.cols {
        return Err(OchreError::InvalidParameter(format!(
            "small_sym_eig_svd requires a non-empty square matrix, got {}x{} ( >_< )",
            sym.rows, sym.cols
        )));
    }
    let p = sym.rows;
    let mut a = vec![0.0; p * p];
    let mut max_diag = 0.0f64;
    for r in 0..p {
        for c in r..p {
            let val = 0.5 * (sym.data[r * p + c] + sym.data[c * p + r]);
            a[r * p + c] = val;
            a[c * p + r] = val;
        }
        max_diag = max_diag.max(a[r * p + r].abs());
    }

    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }

    let tol = 1e-13 * max_diag.max(1.0);
    let max_sweeps = 64;
    for _sweep in 0..max_sweeps {
        let mut off_max = 0.0f64;
        for r in 0..p {
            for c in (r + 1)..p {
                let val = a[r * p + c].abs();
                if val > off_max {
                    off_max = val;
                }
            }
        }
        if off_max <= tol {
            break;
        }
        for r in 0..p {
            for c in (r + 1)..p {
                jacobi_rotate(&mut a, &mut v, p, r, c);
            }
        }
    }

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| a[j * p + j].total_cmp(&a[i * p + i]));

    let mut eigenvalues = Vec::with_capacity(p);
    let mut eigvecs = vec![0.0; p * p];
    for (new_col, &old_col) in order.iter().enumerate() {
        eigenvalues.push(a[old_col * p + old_col]);
        for r in 0..p {
            eigvecs[r * p + new_col] = v[r * p + old_col];
        }
    }

    Ok(SymEigResult {
        eigenvalues,
        eigenvectors: Matrix::from_vec(p, p, eigvecs)?,
    })
}

/// Compute thin right singular values/vectors of `X` (optionally mean-centered)
/// by accelerating `X_c^T * X_c` on the active hardware backend (`E4` + `N1`) and
/// decomposing the resulting `p x p` Gram matrix (`N5`).
/// Returns `(singular_values, v_right_columns, mean)`.
pub fn backend_gram_svd(
    backend: &Backend<'_>,
    x: &Matrix,
    center: bool,
) -> OchreResult<(Vec<f64>, Matrix, Vec<f64>)> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }

    let (mean, x_work) = if center {
        let stats = backend.col_reduce_stats(x, None)?;
        let ones = vec![1.0; x.cols];
        let centered = backend.col_scale_shift(x, &ones, &stats.mean)?;
        (stats.mean, Some(centered))
    } else {
        (vec![0.0; x.cols], None)
    };

    let target_ref = x_work.as_ref().unwrap_or(x);
    let dummy_y = vec![0.0; target_ref.rows];
    let (gram, _) = backend.compute_normal_equation(target_ref, &dummy_y, false)?;
    let eig = backend.small_sym_eig_svd(&gram)?;
    let singular_values: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|&lam| lam.max(0.0).sqrt())
        .collect();

    Ok((singular_values, eig.eigenvectors, mean))
}
