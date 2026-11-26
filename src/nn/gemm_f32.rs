// OchreML High-Performance f32 SGEMM Kernel (*≧ω≦*)
// BLIS-style cache-blocked panel packing (MC=64, KC=256, NC=128) + runtime-detected
// x86_64 AVX2+FMA 6x16 register micro-kernel (_mm256_fmadd_ps) + Rayon parallelization
// + portable scalar fallback, with zero heap allocations inside inner loops.

use crate::error::{OchreError, OchreResult};
use rayon::prelude::*;

pub const MC: usize = 64;
pub const KC: usize = 256;
pub const NC: usize = 128;
const MR: usize = 6;
const NR: usize = 16;

// Minimum FLOPs (2*m*n*k / 2 = m*n*k) to amortize Rayon thread-pool overhead.
const RAYON_PAR_THRESHOLD: usize = 262_144;

/// General single-precision matrix multiplication:
/// `C = alpha * op(A) * op(B) + beta * C` in row-major order.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) -> OchreResult<()> {
    validate_sgemm_dims(m, n, k, a.len(), b.len(), c.len(), trans_a, trans_b)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 || alpha == 0.0 {
        scale_c_only(c, m * n, beta);
        return Ok(());
    }

    let flops = m.saturating_mul(n).saturating_mul(k);
    let use_parallel = flops >= RAYON_PAR_THRESHOLD && m >= MC && rayon::current_num_threads() > 1;
    sgemm_core(
        m,
        n,
        k,
        a,
        b,
        c,
        trans_a,
        trans_b,
        alpha,
        beta,
        use_parallel,
    );
    Ok(())
}

/// Batched single-precision matrix multiplication:
/// `C[b] = op(A[b]) * op(B[b])` for `b in 0..batch`.
#[allow(clippy::too_many_arguments)]
pub fn batched_sgemm(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
) -> OchreResult<()> {
    if batch == 0 {
        return Ok(());
    }
    let a_stride = m.checked_mul(k).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm A stride overflow ( >_< )".to_string())
    })?;
    let b_stride = k.checked_mul(n).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm B stride overflow ( >_< )".to_string())
    })?;
    let c_stride = m.checked_mul(n).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm C stride overflow ( >_< )".to_string())
    })?;

    let req_a = batch.checked_mul(a_stride).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm A size overflow ( >_< )".to_string())
    })?;
    let req_b = batch.checked_mul(b_stride).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm B size overflow ( >_< )".to_string())
    })?;
    let req_c = batch.checked_mul(c_stride).ok_or_else(|| {
        OchreError::InvalidParameter("batched_sgemm C size overflow ( >_< )".to_string())
    })?;

    if a.len() < req_a {
        return Err(OchreError::InvalidParameter(format!(
            "batched_sgemm A slice too small: expected at least {req_a}, got {} ( >_< )",
            a.len()
        )));
    }
    if b.len() < req_b {
        return Err(OchreError::InvalidParameter(format!(
            "batched_sgemm B slice too small: expected at least {req_b}, got {} ( >_< )",
            b.len()
        )));
    }
    if c.len() < req_c {
        return Err(OchreError::InvalidParameter(format!(
            "batched_sgemm C slice too small: expected at least {req_c}, got {} ( >_< )",
            c.len()
        )));
    }

    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        c[..req_c].fill(0.0);
        return Ok(());
    }

    let total_flops = batch.saturating_mul(m).saturating_mul(n).saturating_mul(k);
    if batch >= 2 && total_flops >= RAYON_PAR_THRESHOLD {
        c[..req_c]
            .par_chunks_mut(c_stride)
            .enumerate()
            .for_each(|(bi, c_slice)| {
                let a_slice = &a[bi * a_stride..(bi + 1) * a_stride];
                let b_slice = &b[bi * b_stride..(bi + 1) * b_stride];
                sgemm_core(
                    m, n, k, a_slice, b_slice, c_slice, trans_a, trans_b, 1.0, 0.0, false,
                );
            });
    } else {
        let single_parallel = batch == 1 && total_flops >= RAYON_PAR_THRESHOLD && m >= MC;
        for bi in 0..batch {
            let a_slice = &a[bi * a_stride..(bi + 1) * a_stride];
            let b_slice = &b[bi * b_stride..(bi + 1) * b_stride];
            let c_slice = &mut c[bi * c_stride..(bi + 1) * c_stride];
            sgemm_core(
                m,
                n,
                k,
                a_slice,
                b_slice,
                c_slice,
                trans_a,
                trans_b,
                1.0,
                0.0,
                single_parallel,
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_sgemm_dims(
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
    trans_a: bool,
    trans_b: bool,
) -> OchreResult<()> {
    let req_a = m.checked_mul(k).ok_or_else(|| {
        OchreError::InvalidParameter("sgemm A dimensions overflow ( >_< )".to_string())
    })?;
    let req_b = k.checked_mul(n).ok_or_else(|| {
        OchreError::InvalidParameter("sgemm B dimensions overflow ( >_< )".to_string())
    })?;
    let req_c = m.checked_mul(n).ok_or_else(|| {
        OchreError::InvalidParameter("sgemm C dimensions overflow ( >_< )".to_string())
    })?;

    if a_len < req_a {
        let (rows_a, cols_a) = if trans_a { (k, m) } else { (m, k) };
        return Err(OchreError::InvalidParameter(format!(
            "sgemm A ({rows_a}x{cols_a}) slice too small: need {req_a}, got {a_len} ( >_< )"
        )));
    }
    if b_len < req_b {
        let (rows_b, cols_b) = if trans_b { (n, k) } else { (k, n) };
        return Err(OchreError::InvalidParameter(format!(
            "sgemm B ({rows_b}x{cols_b}) slice too small: need {req_b}, got {b_len} ( >_< )"
        )));
    }
    if c_len < req_c {
        return Err(OchreError::InvalidParameter(format!(
            "sgemm C ({m}x{n}) slice too small: need {req_c}, got {c_len} ( >_< )"
        )));
    }
    Ok(())
}

#[inline]
fn scale_c_only(c: &mut [f32], len: usize, beta: f32) {
    if beta == 0.0 {
        c[..len].fill(0.0);
    } else if beta != 1.0 {
        for v in &mut c[..len] {
            *v *= beta;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm_core(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    use_parallel: bool,
) {
    let use_avx2 = has_avx2_fma();

    // Pre-allocate B packing buffer ONCE outside all blocking loops.
    let max_nr_panels = NC.div_ceil(NR);
    let mut packed_b = vec![0.0f32; max_nr_panels * NR * KC];

    // For sequential path, pre-allocate A packing buffer ONCE outside all blocking loops.
    let max_mr_panels = MC.div_ceil(MR);
    let mut packed_a_seq = if use_parallel {
        Vec::new()
    } else {
        vec![0.0f32; max_mr_panels * MR * KC]
    };

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_beta = if pc == 0 { beta } else { 1.0 };

            pack_panel_b(b, n, k, pc, jc, kc, nc, trans_b, &mut packed_b);

            if use_parallel {
                let b_ref = &packed_b[..nc.div_ceil(NR) * NR * kc];
                c[..m * n].par_chunks_mut(MC * n).enumerate().for_each_init(
                    || vec![0.0f32; max_mr_panels * MR * KC],
                    |packed_a, (block_idx, c_block)| {
                        let ic = block_idx * MC;
                        let mc = c_block.len() / n;
                        pack_panel_a(a, m, k, ic, pc, mc, kc, trans_a, packed_a);
                        macro_kernel(
                            mc,
                            nc,
                            kc,
                            n,
                            alpha,
                            eff_beta,
                            packed_a,
                            b_ref,
                            &mut c_block[jc..],
                            use_avx2,
                        );
                    },
                );
            } else {
                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    pack_panel_a(a, m, k, ic, pc, mc, kc, trans_a, &mut packed_a_seq);
                    macro_kernel(
                        mc,
                        nc,
                        kc,
                        n,
                        alpha,
                        eff_beta,
                        &packed_a_seq,
                        &packed_b,
                        &mut c[ic * n + jc..],
                        use_avx2,
                    );
                    ic += MC;
                }
            }

            pc += KC;
        }
        jc += NC;
    }
}

#[inline]
fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Pack `A[ic..ic+mc, pc..pc+kc]` into contiguous `MR x kc` strips (zero-padded for `mr < MR`).
#[allow(clippy::too_many_arguments)]
fn pack_panel_a(
    a: &[f32],
    m: usize,
    k: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    trans_a: bool,
    packed_a: &mut [f32],
) {
    let mut dst_off = 0;
    let mut ir = 0;
    while ir < mc {
        let mr = (mc - ir).min(MR);
        let row0 = ic + ir;
        if !trans_a {
            if mr == MR {
                let r0 = &a[row0 * k + pc..row0 * k + pc + kc];
                let r1 = &a[(row0 + 1) * k + pc..(row0 + 1) * k + pc + kc];
                let r2 = &a[(row0 + 2) * k + pc..(row0 + 2) * k + pc + kc];
                let r3 = &a[(row0 + 3) * k + pc..(row0 + 3) * k + pc + kc];
                let r4 = &a[(row0 + 4) * k + pc..(row0 + 4) * k + pc + kc];
                let r5 = &a[(row0 + 5) * k + pc..(row0 + 5) * k + pc + kc];
                for p in 0..kc {
                    let base = dst_off + p * MR;
                    packed_a[base] = r0[p];
                    packed_a[base + 1] = r1[p];
                    packed_a[base + 2] = r2[p];
                    packed_a[base + 3] = r3[p];
                    packed_a[base + 4] = r4[p];
                    packed_a[base + 5] = r5[p];
                }
            } else {
                for p in 0..kc {
                    let base = dst_off + p * MR;
                    for r in 0..mr {
                        packed_a[base + r] = a[(row0 + r) * k + pc + p];
                    }
                    for r in mr..MR {
                        packed_a[base + r] = 0.0;
                    }
                }
            }
        } else {
            // A is stored as (k x m) row-major: op(A)[row, col] = a[col * m + row]
            for p in 0..kc {
                let src_row = &a[(pc + p) * m + row0..(pc + p) * m + row0 + mr];
                let base = dst_off + p * MR;
                packed_a[base..base + mr].copy_from_slice(src_row);
                for r in mr..MR {
                    packed_a[base + r] = 0.0;
                }
            }
        }
        dst_off += MR * kc;
        ir += MR;
    }
}

/// Pack `B[pc..pc+kc, jc..jc+nc]` into contiguous `kc x NR` strips (zero-padded for `nr < NR`).
#[allow(clippy::too_many_arguments)]
fn pack_panel_b(
    b: &[f32],
    n: usize,
    k: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    trans_b: bool,
    packed_b: &mut [f32],
) {
    let mut dst_off = 0;
    let mut jr = 0;
    while jr < nc {
        let nr = (nc - jr).min(NR);
        let col0 = jc + jr;
        if !trans_b {
            if nr == NR {
                for p in 0..kc {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + NR];
                    let base = dst_off + p * NR;
                    packed_b[base..base + NR].copy_from_slice(src);
                }
            } else {
                for p in 0..kc {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + nr];
                    let base = dst_off + p * NR;
                    packed_b[base..base + nr].copy_from_slice(src);
                    for c in nr..NR {
                        packed_b[base + c] = 0.0;
                    }
                }
            }
        } else {
            // B is stored as (n x k) row-major: op(B)[p, col] = b[col * k + p]
            for p in 0..kc {
                let base = dst_off + p * NR;
                for c in 0..nr {
                    packed_b[base + c] = b[(col0 + c) * k + pc + p];
                }
                for c in nr..NR {
                    packed_b[base + c] = 0.0;
                }
            }
        }
        dst_off += kc * NR;
        jr += NR;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn macro_kernel(
    mc: usize,
    nc: usize,
    kc: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    use_avx2: bool,
) {
    let mut jr = 0;
    let mut b_off = 0;
    while jr < nc {
        let nr = (nc - jr).min(NR);
        let pb = &packed_b[b_off..b_off + kc * NR];

        let mut ir = 0;
        let mut a_off = 0;
        while ir < mc {
            let mr = (mc - ir).min(MR);
            let pa = &packed_a[a_off..a_off + MR * kc];
            let c_tile = &mut c[ir * ldc + jr..];

            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: `has_avx2_fma()` verified runtime AVX2 and FMA support.
                unsafe {
                    microkernel_6x16_avx2_fma(kc, mr, nr, ldc, alpha, beta, pa, pb, c_tile);
                }
                a_off += MR * kc;
                ir += MR;
                continue;
            }
            let _ = use_avx2;
            microkernel_6x16_scalar(kc, mr, nr, ldc, alpha, beta, pa, pb, c_tile);

            a_off += MR * kc;
            ir += MR;
        }

        b_off += kc * NR;
        jr += NR;
    }
}

/// Portable scalar 6x16 micro-kernel operating on packed A (`MR x kc`) and B (`kc x NR`).
#[allow(clippy::too_many_arguments)]
fn microkernel_6x16_scalar(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c: &mut [f32],
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..kc {
        let a_base = p * MR;
        let b_base = p * NR;
        for r in 0..MR {
            let av = pa[a_base + r];
            for col in 0..NR {
                acc[r][col] += av * pb[b_base + col];
            }
        }
    }

    if beta == 0.0 {
        for r in 0..mr {
            let row = &mut c[r * ldc..r * ldc + nr];
            for col in 0..nr {
                row[col] = alpha * acc[r][col];
            }
        }
    } else {
        for r in 0..mr {
            let row = &mut c[r * ldc..r * ldc + nr];
            for col in 0..nr {
                row[col] = alpha * acc[r][col] + beta * row[col];
            }
        }
    }
}

/// x86_64 AVX2 + FMA 6x16 register micro-kernel using `_mm256_fmadd_ps`.
///
/// # Safety
/// Caller must ensure CPU supports `avx2` and `fma` target features, `pa.len() >= kc * 6`,
/// `pb.len() >= kc * 16`, and `c` has valid capacity for `mr` rows of stride `ldc` with `nr` cols.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn microkernel_6x16_avx2_fma(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c: &mut [f32],
) {
    use std::arch::x86_64::*;

    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c41 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c51 = _mm256_setzero_ps();

    let mut a_ptr = pa.as_ptr();
    let mut b_ptr = pb.as_ptr();

    let k_unroll = kc / 4;
    let k_rem = kc % 4;

    // SAFETY: `pa` has `kc * 6` elements and `pb` has `kc * 16` elements, so pointer
    // increments across `4 * k_unroll + k_rem == kc` steps remain strictly in bounds.
    unsafe {
        for _ in 0..k_unroll {
            for step in 0..4 {
                let ap = a_ptr.add(step * MR);
                let bp = b_ptr.add(step * NR);
                let b0 = _mm256_loadu_ps(bp);
                let b1 = _mm256_loadu_ps(bp.add(8));

                let a0 = _mm256_broadcast_ss(&*ap);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);

                let a1 = _mm256_broadcast_ss(&*ap.add(1));
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);

                let a2 = _mm256_broadcast_ss(&*ap.add(2));
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);

                let a3 = _mm256_broadcast_ss(&*ap.add(3));
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);

                let a4 = _mm256_broadcast_ss(&*ap.add(4));
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);

                let a5 = _mm256_broadcast_ss(&*ap.add(5));
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
            }
            a_ptr = a_ptr.add(4 * MR);
            b_ptr = b_ptr.add(4 * NR);
        }

        for _ in 0..k_rem {
            let b0 = _mm256_loadu_ps(b_ptr);
            let b1 = _mm256_loadu_ps(b_ptr.add(8));

            let a0 = _mm256_broadcast_ss(&*a_ptr);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            let a1 = _mm256_broadcast_ss(&*a_ptr.add(1));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            let a2 = _mm256_broadcast_ss(&*a_ptr.add(2));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            let a3 = _mm256_broadcast_ss(&*a_ptr.add(3));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            let a4 = _mm256_broadcast_ss(&*a_ptr.add(4));
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            let a5 = _mm256_broadcast_ss(&*a_ptr.add(5));
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);

            a_ptr = a_ptr.add(MR);
            b_ptr = b_ptr.add(NR);
        }

        let valpha = _mm256_set1_ps(alpha);
        c00 = _mm256_mul_ps(c00, valpha);
        c01 = _mm256_mul_ps(c01, valpha);
        c10 = _mm256_mul_ps(c10, valpha);
        c11 = _mm256_mul_ps(c11, valpha);
        c20 = _mm256_mul_ps(c20, valpha);
        c21 = _mm256_mul_ps(c21, valpha);
        c30 = _mm256_mul_ps(c30, valpha);
        c31 = _mm256_mul_ps(c31, valpha);
        c40 = _mm256_mul_ps(c40, valpha);
        c41 = _mm256_mul_ps(c41, valpha);
        c50 = _mm256_mul_ps(c50, valpha);
        c51 = _mm256_mul_ps(c51, valpha);

        if mr == MR && nr == NR {
            let c_ptr = c.as_mut_ptr();
            if beta == 0.0 {
                _mm256_storeu_ps(c_ptr, c00);
                _mm256_storeu_ps(c_ptr.add(8), c01);
                _mm256_storeu_ps(c_ptr.add(ldc), c10);
                _mm256_storeu_ps(c_ptr.add(ldc + 8), c11);
                _mm256_storeu_ps(c_ptr.add(2 * ldc), c20);
                _mm256_storeu_ps(c_ptr.add(2 * ldc + 8), c21);
                _mm256_storeu_ps(c_ptr.add(3 * ldc), c30);
                _mm256_storeu_ps(c_ptr.add(3 * ldc + 8), c31);
                _mm256_storeu_ps(c_ptr.add(4 * ldc), c40);
                _mm256_storeu_ps(c_ptr.add(4 * ldc + 8), c41);
                _mm256_storeu_ps(c_ptr.add(5 * ldc), c50);
                _mm256_storeu_ps(c_ptr.add(5 * ldc + 8), c51);
            } else {
                let vbeta = _mm256_set1_ps(beta);
                let rows = [
                    (c00, c01),
                    (c10, c11),
                    (c20, c21),
                    (c30, c31),
                    (c40, c41),
                    (c50, c51),
                ];
                for (r, (r0, r1)) in rows.into_iter().enumerate() {
                    let row_ptr = c_ptr.add(r * ldc);
                    let o0 = _mm256_loadu_ps(row_ptr);
                    let o1 = _mm256_loadu_ps(row_ptr.add(8));
                    _mm256_storeu_ps(row_ptr, _mm256_fmadd_ps(vbeta, o0, r0));
                    _mm256_storeu_ps(row_ptr.add(8), _mm256_fmadd_ps(vbeta, o1, r1));
                }
            }
        } else {
            let mut tmp = [0.0f32; MR * NR];
            let tp = tmp.as_mut_ptr();
            _mm256_storeu_ps(tp, c00);
            _mm256_storeu_ps(tp.add(8), c01);
            _mm256_storeu_ps(tp.add(NR), c10);
            _mm256_storeu_ps(tp.add(NR + 8), c11);
            _mm256_storeu_ps(tp.add(2 * NR), c20);
            _mm256_storeu_ps(tp.add(2 * NR + 8), c21);
            _mm256_storeu_ps(tp.add(3 * NR), c30);
            _mm256_storeu_ps(tp.add(3 * NR + 8), c31);
            _mm256_storeu_ps(tp.add(4 * NR), c40);
            _mm256_storeu_ps(tp.add(4 * NR + 8), c41);
            _mm256_storeu_ps(tp.add(5 * NR), c50);
            _mm256_storeu_ps(tp.add(5 * NR + 8), c51);

            if beta == 0.0 {
                for r in 0..mr {
                    let dst = &mut c[r * ldc..r * ldc + nr];
                    let src = &tmp[r * NR..r * NR + nr];
                    dst.copy_from_slice(src);
                }
            } else {
                for r in 0..mr {
                    let dst = &mut c[r * ldc..r * ldc + nr];
                    let src = &tmp[r * NR..r * NR + nr];
                    for col in 0..nr {
                        dst[col] = src[col] + beta * dst[col];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn naive_sgemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = if trans_a { a[p * m + i] } else { a[i * k + p] };
                    let bv = if trans_b { b[j * k + p] } else { b[p * n + j] };
                    acc += av * bv;
                }
                let prev = if beta == 0.0 {
                    0.0
                } else {
                    beta * c[i * n + j]
                };
                c[i * n + j] = alpha * acc + prev;
            }
        }
    }

    #[test]
    fn test_sgemm_avx2_vs_scalar_and_naive_parity() {
        let shapes = [
            (1, 1, 1),
            (5, 13, 7),
            (6, 16, 17),
            (64, 128, 256),
            (73, 145, 270),
        ];
        for &(m, n, k) in &shapes {
            for trans_a in [false, true] {
                for trans_b in [false, true] {
                    let a: Vec<f32> = (0..m * k)
                        .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.1)
                        .collect();
                    let b: Vec<f32> = (0..k * n)
                        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.1)
                        .collect();
                    let mut c_fast = vec![1.5f32; m * n];
                    let mut c_scalar = vec![1.5f32; m * n];
                    let mut c_ref = vec![1.5f32; m * n];

                    sgemm(m, n, k, &a, &b, &mut c_fast, trans_a, trans_b, 1.25, 0.5).unwrap();
                    naive_sgemm(m, n, k, &a, &b, &mut c_ref, trans_a, trans_b, 1.25, 0.5);

                    // Explicitly verify portable scalar kernel matches naive reference too
                    let max_mr_panels = MC.div_ceil(MR);
                    let max_nr_panels = NC.div_ceil(NR);
                    let mut pa = vec![0.0f32; max_mr_panels * MR * KC];
                    let mut pb = vec![0.0f32; max_nr_panels * NR * KC];
                    let mut jc = 0;
                    while jc < n {
                        let nc = (n - jc).min(NC);
                        let mut pc = 0;
                        while pc < k {
                            let kc = (k - pc).min(KC);
                            let eff_beta = if pc == 0 { 0.5 } else { 1.0 };
                            pack_panel_b(&b, n, k, pc, jc, kc, nc, trans_b, &mut pb);
                            let mut ic = 0;
                            while ic < m {
                                let mc = (m - ic).min(MC);
                                pack_panel_a(&a, m, k, ic, pc, mc, kc, trans_a, &mut pa);
                                macro_kernel(
                                    mc,
                                    nc,
                                    kc,
                                    n,
                                    1.25,
                                    eff_beta,
                                    &pa,
                                    &pb,
                                    &mut c_scalar[ic * n + jc..],
                                    false,
                                );
                                ic += MC;
                            }
                            pc += KC;
                        }
                        jc += NC;
                    }

                    for idx in 0..m * n {
                        assert!(
                            (c_fast[idx] - c_ref[idx]).abs() < 1e-3,
                            "fast vs ref mismatch at ({m},{n},{k}) ta={trans_a} tb={trans_b} idx={idx}: {} vs {}",
                            c_fast[idx],
                            c_ref[idx]
                        );
                        assert!(
                            (c_scalar[idx] - c_ref[idx]).abs() < 1e-3,
                            "scalar vs ref mismatch at idx={idx}: {} vs {}",
                            c_scalar[idx],
                            c_ref[idx]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_batched_sgemm_and_parallel_threshold() {
        let batch = 4;
        let m = 70;
        let n = 65;
        let k = 80;
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let mut c = vec![f32::NAN; batch * m * n];
        batched_sgemm(batch, m, n, k, &a, &b, &mut c, false, true).unwrap();

        let mut c_ref = vec![0.0f32; m * n];
        for bi in 0..batch {
            naive_sgemm(
                m,
                n,
                k,
                &a[bi * m * k..(bi + 1) * m * k],
                &b[bi * k * n..(bi + 1) * k * n],
                &mut c_ref,
                false,
                true,
                1.0,
                0.0,
            );
            for idx in 0..m * n {
                assert!((c[bi * m * n + idx] - c_ref[idx]).abs() < 1e-3);
            }
        }
    }
}
