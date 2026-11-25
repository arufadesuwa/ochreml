// OchreML Primitive N5: small_sym_eig_svd & gram_svd (*≧ω≦*)
// Cyclic Jacobi eigensolver for symmetric P x P matrices + hardware-accelerated Gram SVD.

use crate::backend::Backend;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;

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

#[inline(always)]
fn dot_contig(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut idx = 0;
    while idx + 4 <= n {
        s0 += a[idx] * b[idx];
        s1 += a[idx + 1] * b[idx + 1];
        s2 += a[idx + 2] * b[idx + 2];
        s3 += a[idx + 3] * b[idx + 3];
        idx += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while idx < n {
        sum += a[idx] * b[idx];
        idx += 1;
    }
    sum
}

#[inline(always)]
fn axpy_sub_contig(dst: &mut [f64], alpha: f64, src: &[f64]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] -= alpha * src[i];
    }
}

#[inline]
fn next_gaussian_pair(state: &mut u64) -> (f64, f64) {
    let mut next_open01 = || -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let bits = (z ^ (z >> 31)) >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    };
    let u1 = next_open01().max(1e-15);
    let u2 = next_open01();
    let mag = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * std::f64::consts::PI * u2;
    (mag * angle.cos(), mag * angle.sin())
}

fn fill_random_unit_ortho(
    target: &mut [f64],
    basis_a: &[f64],
    n_a: usize,
    basis_b: &[f64],
    n_b: usize,
    n: usize,
    rng_state: &mut u64,
) {
    for _attempt in 0..4 {
        let mut i = 0;
        while i + 1 < n {
            let (g0, g1) = next_gaussian_pair(rng_state);
            target[i] = g0;
            target[i + 1] = g1;
            i += 2;
        }
        if i < n {
            target[i] = next_gaussian_pair(rng_state).0;
        }
        for _pass in 0..2 {
            for p in 0..n_a {
                let qp = &basis_a[p * n..(p + 1) * n];
                let h = dot_contig(qp, target);
                axpy_sub_contig(target, h, qp);
            }
            for p in 0..n_b {
                let qp = &basis_b[p * n..(p + 1) * n];
                let h = dot_contig(qp, target);
                axpy_sub_contig(target, h, qp);
            }
        }
        let norm = dot_contig(target, target).sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for v in target.iter_mut() {
                *v *= inv;
            }
            return;
        }
    }
    target.fill(0.0);
}

/// Compute `W = sym * Q_cur` where `Q_cur` has `b` contiguous vectors of length `n`.
fn sym_block_matvec(sym: &Matrix, q_cur: &[f64], b: usize, w_out: &mut [f64]) {
    let n = sym.rows;
    if n < 512 {
        for i in 0..n {
            let row = &sym.data[i * n..(i + 1) * n];
            for c in 0..b {
                let qc = &q_cur[c * n..(c + 1) * n];
                w_out[c * n + i] = dot_contig(row, qc);
            }
        }
        return;
    }

    // Parallelize row-wise for large n, writing contiguous row-major (n x b) then scattering to w_out.
    let mut row_major = vec![0.0; n * b];
    row_major
        .par_chunks_mut(128 * b)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let r_start = chunk_idx * 128;
            let r_count = out_chunk.len() / b;
            for dr in 0..r_count {
                let i = r_start + dr;
                let row = &sym.data[i * n..(i + 1) * n];
                let dst = &mut out_chunk[dr * b..(dr + 1) * b];
                for c in 0..b {
                    let qc = &q_cur[c * n..(c + 1) * n];
                    dst[c] = dot_contig(row, qc);
                }
            }
        });
    for i in 0..n {
        for c in 0..b {
            w_out[c * n + i] = row_major[i * b + c];
        }
    }
}

/// Compute top `k` eigenvalues (descending) and eigenvectors `(p x k)` of symmetric matrix `sym`
/// using Thick-Restart Block Symmetric Lanczos with full 2-pass MGS reorthogonalization.
pub fn cpu_top_k_sym_eig(sym: &Matrix, k: usize) -> OchreResult<SymEigResult> {
    if sym.rows == 0 || sym.rows != sym.cols {
        return Err(OchreError::InvalidParameter(format!(
            "cpu_top_k_sym_eig requires a non-empty square matrix, got {}x{} ( >_< )",
            sym.rows, sym.cols
        )));
    }
    if k == 0 {
        return Err(OchreError::InvalidParameter(
            "cpu_top_k_sym_eig requires k >= 1 ( >_< )".to_string(),
        ));
    }

    let n = sym.rows;
    let k_eff = k.min(n);
    let b = if k_eff <= 6 { k_eff } else { 4 };
    let m_target = (2 * k_eff + 20).clamp(16, 40).min(n);
    let n_blocks = (m_target / b).max(6);
    let m_cap = n_blocks * b;

    if m_cap >= n || n < 32 {
        let full = cpu_small_sym_eig_svd(sym)?;
        if k_eff == n {
            return Ok(full);
        }
        let eigenvalues = full.eigenvalues[..k_eff].to_vec();
        let mut vecs = vec![0.0; n * k_eff];
        for row in 0..n {
            let src = &full.eigenvectors.data[row * n..row * n + k_eff];
            vecs[row * k_eff..(row + 1) * k_eff].copy_from_slice(src);
        }
        return Ok(SymEigResult {
            eigenvalues,
            eigenvectors: Matrix::from_vec(n, k_eff, vecs)?,
        });
    }

    let mut q_vecs = vec![0.0; m_cap * n];
    let mut w_vecs = vec![0.0; b * n];
    let mut q_next = vec![0.0; b * n];
    let mut t_data = vec![0.0; m_cap * m_cap];
    let mut a_blk = vec![0.0; b * b];
    let mut r_blk = vec![0.0; b * b];
    let mut q_keep = Vec::new();
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;

    // Seed initial orthonormal block Q_0 (b vectors of length n)
    for c in 0..b {
        let (prev_basis, tail) = q_vecs.split_at_mut(c * n);
        let qc = &mut tail[..n];
        fill_random_unit_ortho(qc, prev_basis, c, &[], 0, n, &mut rng_state);
    }

    let mut cur_start = 0usize;
    let max_restarts = 2usize;

    for restart in 0..=max_restarts {
        let mut m_cur = cur_start;
        while cur_start + b <= m_cap {
            let cur_end = cur_start + b;
            sym_block_matvec(sym, &q_vecs[cur_start * n..cur_end * n], b, &mut w_vecs);

            // Compute diagonal block A = Q_cur^T * W and symmetrize into T
            for r in 0..b {
                let qr = &q_vecs[(cur_start + r) * n..(cur_start + r + 1) * n];
                for c in 0..b {
                    let wc = &w_vecs[c * n..(c + 1) * n];
                    a_blk[r * b + c] = dot_contig(qr, wc);
                }
            }
            for r in 0..b {
                for c in r..b {
                    let val = 0.5 * (a_blk[r * b + c] + a_blk[c * b + r]);
                    t_data[(cur_start + r) * m_cap + (cur_start + c)] = val;
                    t_data[(cur_start + c) * m_cap + (cur_start + r)] = val;
                }
            }

            // Full 2-pass MGS reorthogonalization of W against all Q[0..cur_end]
            for c in 0..b {
                let wc = &mut w_vecs[c * n..(c + 1) * n];
                for _pass in 0..2 {
                    for p in 0..cur_end {
                        let qp = &q_vecs[p * n..(p + 1) * n];
                        let h = dot_contig(qp, wc);
                        axpy_sub_contig(wc, h, qp);
                    }
                }
            }

            // Orthonormalize residual W -> Q_next * R (b x b upper triangular)
            r_blk.fill(0.0);
            let mut scale_w = 1.0f64;
            for c in 0..b {
                let wc = &w_vecs[c * n..(c + 1) * n];
                scale_w = scale_w.max(dot_contig(wc, wc).sqrt());
            }
            for c in 0..b {
                let (prev_next, tail_next) = q_next.split_at_mut(c * n);
                let qn_c = &mut tail_next[..n];
                qn_c.copy_from_slice(&w_vecs[c * n..(c + 1) * n]);
                for _pass in 0..2 {
                    for prev in 0..c {
                        let qp = &prev_next[prev * n..(prev + 1) * n];
                        let h = dot_contig(qp, qn_c);
                        r_blk[prev * b + c] += h;
                        axpy_sub_contig(qn_c, h, qp);
                    }
                }
                let norm = dot_contig(qn_c, qn_c).sqrt();
                if norm > 1e-12 * scale_w {
                    r_blk[c * b + c] = norm;
                    let inv = 1.0 / norm;
                    for v in qn_c.iter_mut() {
                        *v *= inv;
                    }
                } else {
                    r_blk[c * b + c] = 0.0;
                    fill_random_unit_ortho(
                        qn_c,
                        &q_vecs[..cur_end * n],
                        cur_end,
                        prev_next,
                        c,
                        n,
                        &mut rng_state,
                    );
                }
            }

            m_cur = cur_end;
            if cur_end + b <= m_cap {
                q_vecs[cur_end * n..(cur_end + b) * n].copy_from_slice(&q_next);
                for r in 0..b {
                    for c in 0..b {
                        let val = r_blk[r * b + c];
                        t_data[(cur_end + r) * m_cap + (cur_start + c)] = val;
                        t_data[(cur_start + c) * m_cap + (cur_end + r)] = val;
                    }
                }
                cur_start = cur_end;
            } else {
                break;
            }
        }

        // Extract active (m_cur x m_cur) symmetric tridiagonal/arrow matrix and solve
        let mut t_sub = vec![0.0; m_cur * m_cur];
        for r in 0..m_cur {
            t_sub[r * m_cur..(r + 1) * m_cur]
                .copy_from_slice(&t_data[r * m_cap..r * m_cap + m_cur]);
        }
        let t_mat = Matrix::from_vec(m_cur, m_cur, t_sub)?;
        let small_eig = cpu_small_sym_eig_svd(&t_mat)?;
        let u_data = &small_eig.eigenvectors.data;

        // Compute exact Lanczos residual norm ||M v_i - theta_i v_i||_2 = ||R * U[m_cur-b..m_cur, i]||_2
        let tail_start = m_cur - b;
        let mut max_res = 0.0f64;
        for col in 0..k_eff {
            let mut res_sq = 0.0f64;
            for r in 0..b {
                let mut acc = 0.0;
                for c in 0..b {
                    acc += r_blk[r * b + c] * u_data[(tail_start + c) * m_cur + col];
                }
                res_sq += acc * acc;
            }
            max_res = max_res.max(res_sq.sqrt());
        }
        let scale = small_eig.eigenvalues[0]
            .abs()
            .max(small_eig.eigenvalues[m_cur - 1].abs())
            .max(1.0);

        if max_res <= 1e-10 * scale || restart == max_restarts {
            let eigenvalues = small_eig.eigenvalues[..k_eff].to_vec();
            let mut v_data = vec![0.0; n * k_eff];
            for col in 0..k_eff {
                for j in 0..m_cur {
                    let u_jc = u_data[j * m_cur + col];
                    if u_jc == 0.0 {
                        continue;
                    }
                    let qj = &q_vecs[j * n..(j + 1) * n];
                    for row in 0..n {
                        v_data[row * k_eff + col] += u_jc * qj[row];
                    }
                }
            }
            return Ok(SymEigResult {
                eigenvalues,
                eigenvectors: Matrix::from_vec(n, k_eff, v_data)?,
            });
        }

        // Thick Restart: retain top k_keep Ritz vectors + Q_next residual block
        let k_keep = (k_eff + 4)
            .min(m_cur.saturating_sub(b))
            .max(k_eff)
            .min(m_cap - b);
        q_keep.resize(k_keep * n, 0.0);
        q_keep.fill(0.0);
        for col in 0..k_keep {
            let dst = &mut q_keep[col * n..(col + 1) * n];
            for j in 0..m_cur {
                let u_jc = u_data[j * m_cur + col];
                if u_jc == 0.0 {
                    continue;
                }
                let qj = &q_vecs[j * n..(j + 1) * n];
                for row in 0..n {
                    dst[row] += u_jc * qj[row];
                }
            }
        }
        q_vecs[..k_keep * n].copy_from_slice(&q_keep[..k_keep * n]);
        q_vecs[k_keep * n..(k_keep + b) * n].copy_from_slice(&q_next);

        t_data.fill(0.0);
        for col in 0..k_keep {
            t_data[col * m_cap + col] = small_eig.eigenvalues[col];
            for r in 0..b {
                let mut val = 0.0;
                for c in 0..b {
                    val += r_blk[r * b + c] * u_data[(tail_start + c) * m_cur + col];
                }
                t_data[(k_keep + r) * m_cap + col] = val;
                t_data[col * m_cap + (k_keep + r)] = val;
            }
        }
        cur_start = k_keep;
    }

    Err(OchreError::InvalidParameter(
        "cpu_top_k_sym_eig failed to converge ( >_< )".to_string(),
    ))
}

const GRAM_TILE: usize = 64;

#[inline(always)]
fn dot64_tile(a: &[f64; GRAM_TILE], b: &[f64; GRAM_TILE]) -> f64 {
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
fn sum64_tile(a: &[f64; GRAM_TILE]) -> f64 {
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
fn dot64_tile_1x4(
    a: &[f64; GRAM_TILE],
    b0: &[f64; GRAM_TILE],
    b1: &[f64; GRAM_TILE],
    b2: &[f64; GRAM_TILE],
    b3: &[f64; GRAM_TILE],
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
fn dot64_tile_2x4(
    a0: &[f64; GRAM_TILE],
    a1: &[f64; GRAM_TILE],
    b0: &[f64; GRAM_TILE],
    b1: &[f64; GRAM_TILE],
    b2: &[f64; GRAM_TILE],
    b3: &[f64; GRAM_TILE],
) -> [[f64; 4]; 2] {
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
    [
        [
            (acc00[0] + acc00[1]) + (acc00[2] + acc00[3]),
            (acc01[0] + acc01[1]) + (acc01[2] + acc01[3]),
            (acc02[0] + acc02[1]) + (acc02[2] + acc02[3]),
            (acc03[0] + acc03[1]) + (acc03[2] + acc03[3]),
        ],
        [
            (acc10[0] + acc10[1]) + (acc10[2] + acc10[3]),
            (acc11[0] + acc11[1]) + (acc11[2] + acc11[3]),
            (acc12[0] + acc12[1]) + (acc12[2] + acc12[3]),
            (acc13[0] + acc13[1]) + (acc13[2] + acc13[3]),
        ],
    ]
}

#[inline]
fn accumulate_centered_gram_upper(x_t: &[[f64; GRAM_TILE]], gram: &mut [f64], p: usize) {
    let mut j = 0;
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
            let [r0, r1] = dot64_tile_2x4(lj0, lj1, &x_t[k], &x_t[k + 1], &x_t[k + 2], &x_t[k + 3]);
            for lane in 0..4 {
                gram[off0 + k + lane] += r0[lane];
                gram[off1 + k + lane] += r1[lane];
            }
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
        let row_off = j * p;
        let mut k = j;
        while k + 4 <= p {
            let (s0, s1, s2, s3) =
                dot64_tile_1x4(lj, &x_t[k], &x_t[k + 1], &x_t[k + 2], &x_t[k + 3]);
            gram[row_off + k] += s0;
            gram[row_off + k + 1] += s1;
            gram[row_off + k + 2] += s2;
            gram[row_off + k + 3] += s3;
            k += 4;
        }
        while k < p {
            gram[row_off + k] += dot64_tile(lj, &x_t[k]);
            k += 1;
        }
        j += 1;
    }
}

fn accumulate_centered_gram_slice(
    rows_data: &[f64],
    p: usize,
    anchor: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n_rows = rows_data.len() / p;
    let mut gram = vec![0.0; p * p];
    let mut sum_y = vec![0.0; p];
    let mut x_t = vec![[0.0f64; GRAM_TILE]; p];

    let mut r_block = 0;
    while r_block < n_rows {
        let block_end = (r_block + GRAM_TILE).min(n_rows);
        let b_len = block_end - r_block;
        if b_len < GRAM_TILE {
            for col in &mut x_t {
                col[b_len..GRAM_TILE].fill(0.0);
            }
        }

        let mut b = 0;
        while b + 4 <= b_len {
            let r0 = (r_block + b) * p;
            let row0 = &rows_data[r0..(r0 + p)];
            let row1 = &rows_data[(r0 + p)..(r0 + 2 * p)];
            let row2 = &rows_data[(r0 + 2 * p)..(r0 + 3 * p)];
            let row3 = &rows_data[(r0 + 3 * p)..(r0 + 4 * p)];
            for (j, col) in x_t.iter_mut().enumerate().take(p) {
                let a_j = anchor[j];
                col[b] = row0[j] - a_j;
                col[b + 1] = row1[j] - a_j;
                col[b + 2] = row2[j] - a_j;
                col[b + 3] = row3[j] - a_j;
            }
            b += 4;
        }
        while b < b_len {
            let row = &rows_data[((r_block + b) * p)..((r_block + b) * p + p)];
            for (j, &val) in row.iter().enumerate() {
                x_t[j][b] = val - anchor[j];
            }
            b += 1;
        }

        accumulate_centered_gram_upper(&x_t, &mut gram, p);
        for (j, col) in x_t.iter().enumerate().take(p) {
            sum_y[j] += sum64_tile(col);
        }
        r_block = block_end;
    }

    (gram, sum_y)
}

/// Compute centered Gram matrix `X_c^T X_c` and feature `mean` directly from `X`
/// in a single fused Rayon `par_chunks(1024)` pass without allocating an `n x p` matrix.
/// Uses the first row as a shift anchor to avoid catastrophic cancellation.
fn cpu_centered_gram_and_mean(x: &Matrix) -> OchreResult<(Matrix, Vec<f64>)> {
    let n = x.rows;
    let p = x.cols;
    let anchor = &x.data[..p];

    let (mut gram_data, sum_y) = if n >= 2048 {
        x.data
            .par_chunks(1024 * p)
            .map(|chunk| accumulate_centered_gram_slice(chunk, p, anchor))
            .reduce(
                || (vec![0.0; p * p], vec![0.0; p]),
                |(mut g_acc, mut s_acc), (g_chunk, s_chunk)| {
                    for j in 0..p {
                        let off = j * p;
                        for k in j..p {
                            g_acc[off + k] += g_chunk[off + k];
                        }
                        s_acc[j] += s_chunk[j];
                    }
                    (g_acc, s_acc)
                },
            )
    } else {
        accumulate_centered_gram_slice(&x.data, p, anchor)
    };

    let n_f = n as f64;
    let mut y_mean = vec![0.0; p];
    let mut mean = vec![0.0; p];
    for j in 0..p {
        y_mean[j] = sum_y[j] / n_f;
        mean[j] = anchor[j] + y_mean[j];
    }

    for j in 0..p {
        let nj = n_f * y_mean[j];
        gram_data[j * p + j] = (gram_data[j * p + j] - nj * y_mean[j]).max(0.0);
        for k in (j + 1)..p {
            let val = gram_data[j * p + k] - nj * y_mean[k];
            gram_data[j * p + k] = val;
            gram_data[k * p + j] = val;
        }
    }

    Ok((Matrix::from_vec(p, p, gram_data)?, mean))
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

    let (gram, mean) = if center {
        if matches!(backend, Backend::Cpu) {
            cpu_centered_gram_and_mean(x)?
        } else {
            let stats = backend.col_reduce_stats(x, None)?;
            let ones = vec![1.0; x.cols];
            let centered = backend.col_scale_shift(x, &ones, &stats.mean)?;
            let dummy_y = vec![0.0; centered.rows];
            let (g, _) = backend.compute_normal_equation(&centered, &dummy_y, false)?;
            (g, stats.mean)
        }
    } else {
        let dummy_y = vec![0.0; x.rows];
        let (g, _) = backend.compute_normal_equation(x, &dummy_y, false)?;
        (g, vec![0.0; x.cols])
    };

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

    #[test]
    fn test_top_k_sym_eig_psd_parity() {
        // Low-rank + small noise PSD matrix (like a kernel Gram matrix)
        let p = 96usize;
        let rank = 6usize;
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = (z ^ (z >> 31)) >> 11;
            ((bits as f64) * (1.0 / ((1u64 << 53) as f64))) * 2.0 - 1.0
        };
        let mut a = Matrix::zeros(p, rank);
        for r in 0..p {
            for c in 0..rank {
                a[(r, c)] = next_f64() * ((rank - c) as f64);
            }
        }
        let a_t = a.transpose();
        let mut m = a.matmul(&a_t).unwrap();
        for i in 0..p {
            m[(i, i)] += 1e-3;
        }

        let k = 6usize;
        let exact = cpu_small_sym_eig_svd(&m).unwrap();
        let topk = cpu_top_k_sym_eig(&m, k).unwrap();
        assert_eq!(topk.eigenvalues.len(), k);
        assert_eq!(topk.eigenvectors.rows, p);
        assert_eq!(topk.eigenvectors.cols, k);

        for i in 0..k {
            assert!(
                (topk.eigenvalues[i] - exact.eigenvalues[i]).abs() < 1e-7,
                "eigenvalue mismatch at {i}: {} vs {}",
                topk.eigenvalues[i],
                exact.eigenvalues[i]
            );
        }

        // Verify ||V^T V - I_k||_F < 1e-10
        let v = &topk.eigenvectors;
        let mut ortho_sq = 0.0;
        for c1 in 0..k {
            for c2 in 0..k {
                let mut dot = 0.0;
                for r in 0..p {
                    dot += v[(r, c1)] * v[(r, c2)];
                }
                let diff = dot - if c1 == c2 { 1.0 } else { 0.0 };
                ortho_sq += diff * diff;
            }
        }
        assert!(ortho_sq.sqrt() < 1e-10);
    }

    #[test]
    fn test_top_k_sym_eig_near_degenerate_laplacian_250() {
        // 250x250 normalized affinity matrix with two weakly-coupled clusters:
        // top eigenvalues are ~1.0 and ~0.999 (near-degenerate).
        let n = 250usize;
        let half = n / 2;
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            let ci = if i < half { 0 } else { 1 };
            for j in i..n {
                let cj = if j < half { 0 } else { 1 };
                let val = if ci == cj {
                    let diff = ((i as f64) - (j as f64)) * 0.05;
                    (-diff * diff).exp()
                } else {
                    1e-3 * (((i + j) % 5) as f64 + 1.0) / 5.0
                };
                m[(i, j)] = val;
                m[(j, i)] = val;
            }
        }
        let mut d_inv_sqrt = vec![0.0; n];
        for i in 0..n {
            let s: f64 = (0..n).map(|j| m[(i, j)]).sum();
            d_inv_sqrt[i] = 1.0 / s.sqrt();
        }
        for i in 0..n {
            for j in i..n {
                let v = m[(i, j)] * d_inv_sqrt[i] * d_inv_sqrt[j];
                m[(i, j)] = v;
                m[(j, i)] = v;
            }
        }

        let exact = cpu_small_sym_eig_svd(&m).unwrap();
        let topk = cpu_top_k_sym_eig(&m, 2).unwrap();
        assert_eq!(topk.eigenvalues.len(), 2);
        for idx in 0..2 {
            assert!(
                (topk.eigenvalues[idx] - exact.eigenvalues[idx]).abs() < 1e-9,
                "near-degenerate eig mismatch at {idx}: {} vs {}",
                topk.eigenvalues[idx],
                exact.eigenvalues[idx]
            );
        }

        // Verify exact residual ||M * v_i - lam_i * v_i||_2 < 1e-8
        for col in 0..2 {
            let lam = topk.eigenvalues[col];
            let mut res_sq = 0.0;
            for r in 0..n {
                let mut mv = 0.0;
                for c in 0..n {
                    mv += m[(r, c)] * topk.eigenvectors[(c, col)];
                }
                let diff = mv - lam * topk.eigenvectors[(r, col)];
                res_sq += diff * diff;
            }
            assert!(
                res_sq.sqrt() < 1e-8,
                "Lanczos residual at col {col}: {}",
                res_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_fused_centered_gram_svd_parity() {
        let n = 2500usize;
        let p = 16usize;
        let mut state: u64 = 0xABCD_1234_5678_90EF;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = (z ^ (z >> 31)) >> 11;
            ((bits as f64) * (1.0 / ((1u64 << 53) as f64))) * 2.0 - 1.0
        };
        let mut x_data = vec![0.0; n * p];
        for r in 0..n {
            for c in 0..p {
                // Large offset to test cancellation resistance
                x_data[r * p + c] = 1e6 + (c as f64) * 10.0 + next_f64();
            }
        }
        let x = Matrix::from_vec(n, p, x_data).unwrap();

        let (svals, v_right, mean) = backend_gram_svd(&Backend::Cpu, &x, true).unwrap();

        // Compare against explicit centering
        let stats = Backend::Cpu.col_reduce_stats(&x, None).unwrap();
        let ones = vec![1.0; p];
        let centered = Backend::Cpu
            .col_scale_shift(&x, &ones, &stats.mean)
            .unwrap();
        let (svals_ref, v_ref, _) = backend_gram_svd(&Backend::Cpu, &centered, false).unwrap();

        for c in 0..p {
            assert!((mean[c] - stats.mean[c]).abs() < 1e-9);
            assert!(
                (svals[c] - svals_ref[c]).abs() < 1e-7,
                "sval mismatch at {c}: {} vs {}",
                svals[c],
                svals_ref[c]
            );
            let mut dot = 0.0;
            for r in 0..p {
                dot += v_right[(r, c)] * v_ref[(r, c)];
            }
            assert!(dot.abs() > 0.999999);
        }
    }
}
