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

/// Compute symmetric eigendecomposition `M = V * diag(lambda) * V^T` via Cyclic Jacobi
/// (for small matrices `p < 32`).
fn cpu_jacobi_sym_eig(sym: &Matrix) -> OchreResult<SymEigResult> {
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

/// Householder reduction of a real symmetric matrix to tridiagonal form (`tred2`, Golub & Van Loan / EISPACK).
/// Operates on the lower triangle of `z` (row-major `p x p`) and overwrites `z` with the orthogonal
/// transformation matrix `Q` such that `M = Q * T * Q^T`.
/// On exit, `d` holds the diagonal of `T` and `e[1..p]` holds the subdiagonal (`e[0] = 0.0`).
fn householder_tred2(z: &mut [f64], d: &mut [f64], e: &mut [f64], p: usize) {
    let mut u = vec![0.0; p];
    let mut w = vec![0.0; p];

    // Step 1: Reduce lower triangle of z to tridiagonal form
    for i in (1..p).rev() {
        let l = i - 1;
        let mut scale = 0.0f64;
        let row_i_off = i * p;
        for k in 0..=l {
            scale += z[row_i_off + k].abs();
        }

        if scale == 0.0 || l == 0 {
            e[i] = z[row_i_off + l];
            d[i] = 0.0;
            continue;
        }

        let inv_scale = 1.0 / scale;
        let mut h = 0.0f64;
        for k in 0..=l {
            let val = z[row_i_off + k] * inv_scale;
            z[row_i_off + k] = val;
            u[k] = val;
            h += val * val;
        }

        let f = u[l];
        let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
        e[i] = scale * g;
        h -= f * g;
        u[l] = f - g;
        z[row_i_off + l] = u[l];

        // Compute p_vec = A_{0..=l, 0..=l} * u into e[0..=l]
        for j in 0..=l {
            let u_j = u[j];
            let row_j = &z[j * p..j * p + j];
            let mut acc = 0.0f64;
            for k in 0..j {
                let a_jk = row_j[k];
                acc += a_jk * u[k];
                e[k] += a_jk * u_j;
            }
            e[j] = acc + z[j * p + j] * u_j;
        }

        let inv_h = 1.0 / h;
        let mut f_dot = 0.0f64;
        for j in 0..=l {
            let p_j = e[j] * inv_h;
            e[j] = p_j;
            f_dot += p_j * u[j];
        }

        let hh = f_dot * (0.5 * inv_h);
        for j in 0..=l {
            e[j] -= hh * u[j];
        }

        // Symmetric rank-2 update of lower triangle: A -= u * q^T + q * u^T
        for j in 0..=l {
            let u_j = u[j];
            let q_j = e[j];
            let row_j = &mut z[j * p..j * p + j + 1];
            for k in 0..=j {
                row_j[k] -= u_j * e[k] + q_j * u[k];
            }
        }

        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;

    // Step 2: Accumulate orthogonal transformation matrix Q = H_{p-1} * ... * H_2
    for i in 0..p {
        let h = d[i];
        if h != 0.0 {
            let row_i_off = i * p;
            u[..i].copy_from_slice(&z[row_i_off..row_i_off + i]);

            // w[0..i] = u^T * Q[0..i, 0..i]
            w[..i].fill(0.0);
            for m in 0..i {
                let u_m = u[m];
                if u_m != 0.0 {
                    let row_m = &z[m * p..m * p + i];
                    for j in 0..i {
                        w[j] += u_m * row_m[j];
                    }
                }
            }

            // Q[0..i, 0..i] -= (u / h) * w^T
            let inv_h = 1.0 / h;
            for m in 0..i {
                let factor = u[m] * inv_h;
                if factor != 0.0 {
                    let row_m = &mut z[m * p..m * p + i];
                    for j in 0..i {
                        row_m[j] -= factor * w[j];
                    }
                }
            }
        }

        d[i] = z[i * p + i];
        z[i * p + i] = 1.0;
        for j in 0..i {
            z[i * p + j] = 0.0;
            z[j * p + i] = 0.0;
        }
    }
}

/// Implicit Wilkinson-shifted QL algorithm for symmetric tridiagonal matrices (`tql2`, EISPACK / Golub & Van Loan).
/// `zt` holds `Q^T` (each row `k` of `zt` is eigenvector `k` of length `p`) so Givens rotations
/// operate on contiguous rows for cache locality and SIMD auto-vectorization.
fn implicit_ql_tql2(d: &mut [f64], e: &mut [f64], zt: &mut [f64], p: usize) -> OchreResult<()> {
    for i in 1..p {
        e[i - 1] = e[i];
    }
    e[p - 1] = 0.0;

    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    let eps = f64::EPSILON;

    for l in 0..p {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < p - 1 {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }

        if m > l {
            let mut iter = 0usize;
            loop {
                iter += 1;
                if iter > 60 {
                    return Err(OchreError::InvalidParameter(
                        "tql2 failed to converge within 60 iterations per eigenvalue".to_string(),
                    ));
                }

                // Compute implicit Wilkinson shift
                let mut g = d[l];
                let mut p_val = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p_val.hypot(1.0);
                let shift_denom = p_val + if p_val >= 0.0 { r } else { -r };
                d[l] = e[l] / shift_denom;
                d[l + 1] = e[l] * shift_denom;
                let dl1 = d[l + 1];
                let h = g - d[l];
                for d_i in d.iter_mut().take(p).skip(l + 2) {
                    *d_i -= h;
                }
                f += h;

                // Implicit QL transformation
                p_val = d[m];
                let mut c = 1.0f64;
                let mut c2 = 1.0f64;
                let mut c3 = 1.0f64;
                let el1 = e[l + 1];
                let mut s = 0.0f64;
                let mut s2 = 0.0f64;

                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    let h_val = c * p_val;
                    r = p_val.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p_val / r;
                    p_val = c * d[i] - s * g;
                    d[i + 1] = h_val + s * (c * g + s * d[i]);

                    // Apply Givens rotation to contiguous eigenvector rows i and i + 1
                    let (left, right) = zt.split_at_mut((i + 1) * p);
                    let row_i = &mut left[i * p..(i + 1) * p];
                    let row_i1 = &mut right[..p];
                    for k in 0..p {
                        let v_i = row_i[k];
                        let v_i1 = row_i1[k];
                        row_i1[k] = s * v_i + c * v_i1;
                        row_i[k] = c * v_i - s * v_i1;
                    }
                }

                p_val = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p_val;
                d[l] = c * p_val;

                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }

        d[l] += f;
        e[l] = 0.0;
    }

    Ok(())
}

/// Compute symmetric eigendecomposition `M = V * diag(lambda) * V^T` via Householder
/// tridiagonalization (`tred2`) + implicit Wilkinson-shifted QL (`tql2`) for `p >= 32`.
fn cpu_householder_ql_sym_eig(sym: &Matrix) -> OchreResult<SymEigResult> {
    let p = sym.rows;
    let mut z = vec![0.0; p * p];
    for r in 0..p {
        for c in 0..=r {
            let val = 0.5 * (sym.data[r * p + c] + sym.data[c * p + r]);
            z[r * p + c] = val;
            z[c * p + r] = val;
        }
    }

    let mut d = vec![0.0; p];
    let mut e = vec![0.0; p];
    householder_tred2(&mut z, &mut d, &mut e, p);

    // Transpose Q into zt so each eigenvector is a contiguous row during QL rotations
    let mut zt = vec![0.0; p * p];
    for r in 0..p {
        for c in 0..p {
            zt[c * p + r] = z[r * p + c];
        }
    }

    implicit_ql_tql2(&mut d, &mut e, &mut zt, p)?;

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));

    let mut eigenvalues = Vec::with_capacity(p);
    let mut eigvecs = vec![0.0; p * p];
    for (new_col, &old_col) in order.iter().enumerate() {
        eigenvalues.push(d[old_col]);
        let src_row = &zt[old_col * p..(old_col + 1) * p];
        for r in 0..p {
            eigvecs[r * p + new_col] = src_row[r];
        }
    }

    Ok(SymEigResult {
        eigenvalues,
        eigenvectors: Matrix::from_vec(p, p, eigvecs)?,
    })
}

/// Compute symmetric eigendecomposition `M = V * diag(lambda) * V^T`.
/// Uses Cyclic Jacobi for `p < 32` and Householder tridiagonalization (`tred2`) +
/// implicit Wilkinson-shifted QL (`tql2`) for `p >= 32`.
pub fn cpu_small_sym_eig_svd(sym: &Matrix) -> OchreResult<SymEigResult> {
    if sym.rows == 0 || sym.rows != sym.cols {
        return Err(OchreError::InvalidParameter(format!(
            "small_sym_eig_svd requires a non-empty square matrix, got {}x{} ( >_< )",
            sym.rows, sym.cols
        )));
    }
    if sym.rows < 32 {
        cpu_jacobi_sym_eig(sym)
    } else {
        cpu_householder_ql_sym_eig(sym)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_symmetric(p: usize, seed: u64) -> Matrix {
        let mut state = seed;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = (z ^ (z >> 31)) >> 11;
            ((bits as f64) * (1.0 / ((1u64 << 53) as f64))) * 2.0 - 1.0
        };
        let mut data = vec![0.0; p * p];
        for r in 0..p {
            for c in r..p {
                let v = next_f64();
                data[r * p + c] = v;
                data[c * p + r] = v;
            }
        }
        Matrix::from_vec(p, p, data).expect("valid square matrix")
    }

    fn verify_sym_eig(m: &Matrix, tol: f64) {
        let p = m.rows;
        let res = cpu_small_sym_eig_svd(m).expect("eigendecomposition should succeed");
        assert_eq!(res.eigenvalues.len(), p);
        assert_eq!(res.eigenvectors.rows, p);
        assert_eq!(res.eigenvectors.cols, p);

        // Verify descending order
        for i in 1..p {
            assert!(
                res.eigenvalues[i - 1] >= res.eigenvalues[i],
                "eigenvalues not sorted descending at {i}: {} < {}",
                res.eigenvalues[i - 1],
                res.eigenvalues[i]
            );
        }

        let v = &res.eigenvectors.data;

        // Verify ||V^T V - I||_F < tol
        let mut ortho_sq = 0.0f64;
        for c1 in 0..p {
            for c2 in 0..p {
                let mut dot = 0.0;
                for r in 0..p {
                    dot += v[r * p + c1] * v[r * p + c2];
                }
                let target = if c1 == c2 { 1.0 } else { 0.0 };
                let diff = dot - target;
                ortho_sq += diff * diff;
            }
        }
        let ortho_err = ortho_sq.sqrt();
        assert!(
            ortho_err < tol,
            "p={p}: ||V^T V - I||_F = {ortho_err:.3e} >= {tol:.3e}"
        );

        // Verify ||M - V diag(lam) V^T||_F < tol
        let mut recon_sq = 0.0f64;
        for r in 0..p {
            for c in 0..p {
                let mut val = 0.0;
                for k in 0..p {
                    val += v[r * p + k] * res.eigenvalues[k] * v[c * p + k];
                }
                let diff = m.data[r * p + c] - val;
                recon_sq += diff * diff;
            }
        }
        let recon_err = recon_sq.sqrt();
        assert!(
            recon_err < tol,
            "p={p}: ||M - V diag(lam) V^T||_F = {recon_err:.3e} >= {tol:.3e}"
        );
    }

    #[test]
    fn test_sym_eig_jacobi_and_householder_ql() {
        for &p in &[8usize, 32, 64, 128, 300] {
            let m = make_random_symmetric(p, 0x1234_5678 + p as u64);
            verify_sym_eig(&m, 1e-10);
        }
    }

    #[test]
    fn test_sym_eig_edge_cases_householder() {
        // Identity matrix (repeated eigenvalues)
        let eye = Matrix::identity(40);
        verify_sym_eig(&eye, 1e-10);

        // Zero matrix
        let zero = Matrix::zeros(36, 36);
        verify_sym_eig(&zero, 1e-10);

        // Diagonal matrix with negative, zero, and positive entries
        let mut diag = Matrix::zeros(48, 48);
        for i in 0..48 {
            diag[(i, i)] = (i as f64) - 24.0;
        }
        verify_sym_eig(&diag, 1e-10);
    }
}
