// OchreML High-Performance f32 SGEMM Kernel (*≧ω≦*)
// BLIS-style cache-blocked panel packing (MC=72, KC=256, NC=128) + thread-local scratchpads
// + runtime-detected x86_64 AVX2+FMA 6x16 register micro-kernel (_mm256_fmadd_ps)
// + AVX2 8x8 SIMD transpose packing for trans_b + 2D Rayon parallel tiling
// + portable scalar fallback, with zero per-call heap allocations.

use crate::error::{OchreError, OchreResult};
use rayon::prelude::*;
use std::cell::RefCell;

pub const MC: usize = 72;
pub const KC: usize = 256;
pub const NC: usize = 128;
const MR: usize = 6;
const NR: usize = 16;

// Minimum FLOPs proxy (m*n*k) to amortize Rayon thread-pool overhead.
const RAYON_PAR_THRESHOLD: usize = 262_144;

thread_local! {
    static PACK_A_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static PACK_B_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);

// SAFETY: `SendMutPtr` is only used inside `sgemm_core` to write to strictly disjoint
// 2D `(ic0..ic1, jc0..jc1)` sub-tiles of `C` owned exclusively by the caller for the
// duration of the parallel scope.
unsafe impl Send for SendMutPtr {}
// SAFETY: Each parallel task accesses a disjoint rectangular tile of `C`; no two tasks
// read or write the same memory address.
unsafe impl Sync for SendMutPtr {}

#[inline]
fn with_scratch<R>(
    tls: &'static std::thread::LocalKey<RefCell<Vec<f32>>>,
    needed: usize,
    f: impl FnOnce(&mut [f32]) -> R,
) -> R {
    tls.with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            }
            f(&mut buf[..needed])
        } else {
            let mut fallback = vec![0.0f32; needed];
            f(&mut fallback)
        }
    })
}

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
    let nthreads = rayon::current_num_threads();
    let use_parallel = flops >= RAYON_PAR_THRESHOLD && nthreads > 1 && (m >= 2 * MR || n >= 2 * NR);
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

/// Single-threaded single-precision matrix multiplication (no Rayon dispatch).
/// Intended for callers that already parallelize over an outer dimension (e.g. `Conv2d` batch).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_serial(
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

    sgemm_core(m, n, k, a, b, c, trans_a, trans_b, alpha, beta, false);
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
        let nthreads = rayon::current_num_threads();
        let single_parallel = batch == 1
            && total_flops >= RAYON_PAR_THRESHOLD
            && nthreads > 1
            && (m >= 2 * MR || n >= 2 * NR);
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

/// Choose a 2D tile grid `(nt_m, nt_n)` that maximizes core utilization while minimizing
/// total A+B panel packing perimeter across threads.
#[inline]
fn choose_2d_grid(m: usize, n: usize, nthreads: usize) -> (usize, usize) {
    let max_tm = m.div_ceil(MR).max(1).min(nthreads);
    let max_tn = n.div_ceil(NR).max(1).min(nthreads);

    let mut best_tm = 1usize;
    let mut best_tn = 1usize;
    let mut best_active = 1usize;
    let mut best_perim = usize::MAX;

    for tm in 1..=max_tm {
        let tn = (nthreads / tm).min(max_tn).max(1);
        let active = tm * tn;
        let perim = m.div_ceil(tm) + n.div_ceil(tn);
        if active > best_active || (active == best_active && perim < best_perim) {
            best_active = active;
            best_perim = perim;
            best_tm = tm;
            best_tn = tn;
        }
    }

    (best_tm, best_tn)
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

    if !use_parallel {
        // SAFETY: `c` has at least `m * n` elements, and the single tile covers `[0..m) x [0..n)`.
        unsafe {
            sgemm_tile_serial(
                m,
                n,
                k,
                0,
                m,
                0,
                n,
                a,
                b,
                c.as_mut_ptr(),
                trans_a,
                trans_b,
                alpha,
                beta,
                use_avx2,
            );
        }
        return;
    }

    let nthreads = rayon::current_num_threads();
    let (nt_m, nt_n) = choose_2d_grid(m, n, nthreads);
    let total_tiles = nt_m * nt_n;

    if total_tiles <= 1 {
        // SAFETY: `c` has at least `m * n` elements, and the single tile covers `[0..m) x [0..n)`.
        unsafe {
            sgemm_tile_serial(
                m,
                n,
                k,
                0,
                m,
                0,
                n,
                a,
                b,
                c.as_mut_ptr(),
                trans_a,
                trans_b,
                alpha,
                beta,
                use_avx2,
            );
        }
        return;
    }

    let tile_m = m.div_ceil(nt_m).div_ceil(MR) * MR;
    let tile_n = n.div_ceil(nt_n).div_ceil(NR) * NR;
    let c_ptr = SendMutPtr(c.as_mut_ptr());

    (0..total_tiles).into_par_iter().for_each(|tile_idx| {
        let im = tile_idx / nt_n;
        let jn = tile_idx % nt_n;
        let ic0 = im * tile_m;
        let jc0 = jn * tile_n;
        if ic0 >= m || jc0 >= n {
            return;
        }
        let tm = (m - ic0).min(tile_m);
        let tn = (n - jc0).min(tile_n);
        let ptr = c_ptr;
        // SAFETY: `tile_idx` uniquely identifies `(im, jn)`, partitioning `[0..m) x [0..n)`
        // into strictly disjoint rectangular tiles `[ic0..ic0+tm) x [jc0..jc0+tn)`.
        // `c` has at least `m * n` elements and remains exclusively borrowed for this scope.
        unsafe {
            sgemm_tile_serial(
                m, n, k, ic0, tm, jc0, tn, a, b, ptr.0, trans_a, trans_b, alpha, beta, use_avx2,
            );
        }
    });
}

/// Computes the rectangular sub-tile `C[ic0..ic0+tm, jc0..jc0+tn]` on the current thread
/// using thread-local `PACK_A_SCRATCH` and `PACK_B_SCRATCH` buffers.
///
/// # Safety
/// `c_base` must point to a valid buffer of at least `m * n` `f32` elements, and no other
/// thread may concurrently read or write any element in `[ic0..ic0+tm) x [jc0..jc0+tn)`.
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_tile_serial(
    m: usize,
    n: usize,
    k: usize,
    ic0: usize,
    tm: usize,
    jc0: usize,
    tn: usize,
    a: &[f32],
    b: &[f32],
    c_base: *mut f32,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    use_avx2: bool,
) {
    let max_mc = tm.min(MC);
    let max_nc = tn.min(NC);
    let max_kc = k.min(KC);
    let need_a = max_mc.div_ceil(MR) * MR * max_kc;
    let need_b = max_nc.div_ceil(NR) * NR * max_kc;

    with_scratch(&PACK_B_SCRATCH, need_b, |packed_b| {
        with_scratch(&PACK_A_SCRATCH, need_a, |packed_a| {
            let mut dj = 0;
            while dj < tn {
                let nc = (tn - dj).min(NC);
                let jc = jc0 + dj;
                let mut pc = 0;
                while pc < k {
                    let kc = (k - pc).min(KC);
                    let eff_beta = if pc == 0 { beta } else { 1.0 };

                    pack_panel_b(b, n, k, pc, jc, kc, nc, trans_b, use_avx2, packed_b);

                    let mut di = 0;
                    while di < tm {
                        let mc = (tm - di).min(MC);
                        let ic = ic0 + di;
                        pack_panel_a(a, m, k, ic, pc, mc, kc, trans_a, packed_a);
                        // SAFETY: `(ic..ic+mc, jc..jc+nc)` lies strictly inside
                        // `[ic0..ic0+tm) x [jc0..jc0+tn)`, which is exclusively owned by this call.
                        unsafe {
                            let c_tile = c_base.add(ic * n + jc);
                            macro_kernel(
                                mc, nc, kc, n, alpha, eff_beta, packed_a, packed_b, c_tile,
                                use_avx2,
                            );
                        }
                        di += MC;
                    }

                    pc += KC;
                }
                dj += NC;
            }
        });
    });
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
        let dst_panel = &mut packed_a[dst_off..dst_off + MR * kc];

        if !trans_a {
            if mr == MR {
                let r0 = &a[row0 * k + pc..row0 * k + pc + kc];
                let r1 = &a[(row0 + 1) * k + pc..(row0 + 1) * k + pc + kc];
                let r2 = &a[(row0 + 2) * k + pc..(row0 + 2) * k + pc + kc];
                let r3 = &a[(row0 + 3) * k + pc..(row0 + 3) * k + pc + kc];
                let r4 = &a[(row0 + 4) * k + pc..(row0 + 4) * k + pc + kc];
                let r5 = &a[(row0 + 5) * k + pc..(row0 + 5) * k + pc + kc];
                for (p, chunk) in dst_panel.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    chunk[0] = r0[p];
                    chunk[1] = r1[p];
                    chunk[2] = r2[p];
                    chunk[3] = r3[p];
                    chunk[4] = r4[p];
                    chunk[5] = r5[p];
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    for (r, slot) in chunk.iter_mut().enumerate().take(mr) {
                        *slot = a[(row0 + r) * k + pc + p];
                    }
                    chunk[mr..MR].fill(0.0);
                }
            }
        } else {
            // A is stored as (k x m) row-major: op(A)[row, col] = a[col * m + row]
            if mr == MR {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    let src_row = &a[(pc + p) * m + row0..(pc + p) * m + row0 + MR];
                    chunk.copy_from_slice(src_row);
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    let src_row = &a[(pc + p) * m + row0..(pc + p) * m + row0 + mr];
                    chunk[..mr].copy_from_slice(src_row);
                    chunk[mr..MR].fill(0.0);
                }
            }
        }
        dst_off += MR * kc;
        ir += MR;
    }
}

/// Transposes an 8x8 f32 block from 8 contiguous source rows (stride `src_stride`)
/// into 8 destination rows of stride `NR` (16), offset by `dst_col_off` (0 or 8).
///
/// # Safety
/// Caller must ensure AVX2 is supported, `src_base` has at least `7 * src_stride + 8` valid
/// `f32` elements, and `dst_base` has at least `7 * NR + dst_col_off + 8` valid `f32` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8_to_nr_avx2(
    src_base: *const f32,
    src_stride: usize,
    dst_base: *mut f32,
    dst_col_off: usize,
) {
    use std::arch::x86_64::*;

    // SAFETY: Caller guarantees `src_base` and `dst_base` are valid for the 8x8 block.
    unsafe {
        let r0 = _mm256_loadu_ps(src_base);
        let r1 = _mm256_loadu_ps(src_base.add(src_stride));
        let r2 = _mm256_loadu_ps(src_base.add(2 * src_stride));
        let r3 = _mm256_loadu_ps(src_base.add(3 * src_stride));
        let r4 = _mm256_loadu_ps(src_base.add(4 * src_stride));
        let r5 = _mm256_loadu_ps(src_base.add(5 * src_stride));
        let r6 = _mm256_loadu_ps(src_base.add(6 * src_stride));
        let r7 = _mm256_loadu_ps(src_base.add(7 * src_stride));

        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        let s4 = _mm256_shuffle_ps(t4, t6, 0x44);
        let s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        let s6 = _mm256_shuffle_ps(t5, t7, 0x44);
        let s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

        let o0 = _mm256_permute2f128_ps(s0, s4, 0x20);
        let o1 = _mm256_permute2f128_ps(s1, s5, 0x20);
        let o2 = _mm256_permute2f128_ps(s2, s6, 0x20);
        let o3 = _mm256_permute2f128_ps(s3, s7, 0x20);
        let o4 = _mm256_permute2f128_ps(s0, s4, 0x31);
        let o5 = _mm256_permute2f128_ps(s1, s5, 0x31);
        let o6 = _mm256_permute2f128_ps(s2, s6, 0x31);
        let o7 = _mm256_permute2f128_ps(s3, s7, 0x31);

        let d = dst_base.add(dst_col_off);
        _mm256_storeu_ps(d, o0);
        _mm256_storeu_ps(d.add(NR), o1);
        _mm256_storeu_ps(d.add(2 * NR), o2);
        _mm256_storeu_ps(d.add(3 * NR), o3);
        _mm256_storeu_ps(d.add(4 * NR), o4);
        _mm256_storeu_ps(d.add(5 * NR), o5);
        _mm256_storeu_ps(d.add(6 * NR), o6);
        _mm256_storeu_ps(d.add(7 * NR), o7);
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
    use_avx2: bool,
    packed_b: &mut [f32],
) {
    let mut dst_off = 0;
    let mut jr = 0;
    while jr < nc {
        let nr = (nc - jr).min(NR);
        let col0 = jc + jr;
        let dst_panel = &mut packed_b[dst_off..dst_off + kc * NR];

        if !trans_b {
            if nr == NR {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + NR];
                    chunk.copy_from_slice(src);
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + nr];
                    chunk[..nr].copy_from_slice(src);
                    chunk[nr..NR].fill(0.0);
                }
            }
        } else {
            // B is stored as (n x k) row-major: op(B)[p, col] = b[col * k + p]
            let mut p_done = 0usize;
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && nr == NR && kc >= 8 {
                let p_blocks = kc / 8;
                let src_top = b[(col0 * k + pc)..].as_ptr();
                let src_bot = b[((col0 + 8) * k + pc)..].as_ptr();
                let dst_ptr = dst_panel.as_mut_ptr();
                for pb in 0..p_blocks {
                    let p0 = pb * 8;
                    // SAFETY: `use_avx2` verified AVX2 support; `col0 + 16 <= n` and
                    // `pc + p0 + 8 <= k`, so both 8x8 source blocks and destination rows are in bounds.
                    unsafe {
                        transpose_8x8_to_nr_avx2(src_top.add(p0), k, dst_ptr.add(p0 * NR), 0);
                        transpose_8x8_to_nr_avx2(src_bot.add(p0), k, dst_ptr.add(p0 * NR), 8);
                    }
                }
                p_done = p_blocks * 8;
            }
            let _ = use_avx2;

            if nr == NR {
                for p in p_done..kc {
                    let chunk = &mut dst_panel[p * NR..(p + 1) * NR];
                    let p_idx = pc + p;
                    for (c, dst_val) in chunk.iter_mut().enumerate() {
                        *dst_val = b[(col0 + c) * k + p_idx];
                    }
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                    let p_idx = pc + p;
                    for (c, slot) in chunk.iter_mut().enumerate().take(nr) {
                        *slot = b[(col0 + c) * k + p_idx];
                    }
                    chunk[nr..NR].fill(0.0);
                }
            }
        }
        dst_off += kc * NR;
        jr += NR;
    }
}

/// # Safety
/// `c_ptr` must point to the top-left of a valid `(mc x nc)` sub-matrix with row stride `ldc`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn macro_kernel(
    mc: usize,
    nc: usize,
    kc: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c_ptr: *mut f32,
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
            // SAFETY: `ir + mr <= mc` and `jr + nr <= nc`, so `c_tile` is in bounds of `c_ptr`.
            let c_tile = unsafe { c_ptr.add(ir * ldc + jr) };

            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: `has_avx2_fma()` verified runtime AVX2 and FMA support, and `c_tile`
                // has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
                unsafe {
                    microkernel_6x16_avx2_fma(kc, mr, nr, ldc, alpha, beta, pa, pb, c_tile);
                }
                a_off += MR * kc;
                ir += MR;
                continue;
            }
            let _ = use_avx2;
            // SAFETY: `c_tile` has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
            unsafe {
                microkernel_6x16_scalar(kc, mr, nr, ldc, alpha, beta, pa, pb, c_tile);
            }

            a_off += MR * kc;
            ir += MR;
        }

        b_off += kc * NR;
        jr += NR;
    }
}

/// Portable scalar 6x16 micro-kernel operating on packed A (`MR x kc`) and B (`kc x NR`).
///
/// # Safety
/// `c_ptr` must have valid writable capacity for `mr` rows of stride `ldc` with `nr` cols.
#[allow(clippy::too_many_arguments)]
unsafe fn microkernel_6x16_scalar(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c_ptr: *mut f32,
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

    // SAFETY: Caller guarantees `c_ptr` has `mr` rows of stride `ldc` with at least `nr` cols.
    unsafe {
        if beta == 0.0 {
            for (r, acc_row) in acc.iter().enumerate().take(mr) {
                let row_ptr = c_ptr.add(r * ldc);
                for (col, &val) in acc_row.iter().enumerate().take(nr) {
                    *row_ptr.add(col) = alpha * val;
                }
            }
        } else {
            for (r, acc_row) in acc.iter().enumerate().take(mr) {
                let row_ptr = c_ptr.add(r * ldc);
                for (col, &val) in acc_row.iter().enumerate().take(nr) {
                    let dst = row_ptr.add(col);
                    *dst = alpha * val + beta * (*dst);
                }
            }
        }
    }
}

/// x86_64 AVX2 + FMA 6x16 register micro-kernel using `_mm256_fmadd_ps`.
///
/// # Safety
/// Caller must ensure CPU supports `avx2` and `fma` target features, `pa.len() >= kc * 6`,
/// `pb.len() >= kc * 16`, and `c_ptr` has valid capacity for `mr` rows of stride `ldc` with `nr` cols.
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
    c_ptr: *mut f32,
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

        if alpha != 1.0 {
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
        }

        let rows = [
            (c00, c01),
            (c10, c11),
            (c20, c21),
            (c30, c31),
            (c40, c41),
            (c50, c51),
        ];

        if nr == NR {
            if beta == 0.0 {
                for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
                    let row_ptr = c_ptr.add(r * ldc);
                    _mm256_storeu_ps(row_ptr, r0);
                    _mm256_storeu_ps(row_ptr.add(8), r1);
                }
            } else if beta == 1.0 {
                for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
                    let row_ptr = c_ptr.add(r * ldc);
                    let o0 = _mm256_loadu_ps(row_ptr);
                    let o1 = _mm256_loadu_ps(row_ptr.add(8));
                    _mm256_storeu_ps(row_ptr, _mm256_add_ps(o0, r0));
                    _mm256_storeu_ps(row_ptr.add(8), _mm256_add_ps(o1, r1));
                }
            } else {
                let vbeta = _mm256_set1_ps(beta);
                for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
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
            for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
                _mm256_storeu_ps(tp.add(r * NR), r0);
                _mm256_storeu_ps(tp.add(r * NR + 8), r1);
            }

            if beta == 0.0 {
                for r in 0..mr {
                    let row_ptr = c_ptr.add(r * ldc);
                    let src = &tmp[r * NR..r * NR + nr];
                    std::ptr::copy_nonoverlapping(src.as_ptr(), row_ptr, nr);
                }
            } else {
                for r in 0..mr {
                    let row_ptr = c_ptr.add(r * ldc);
                    let src = &tmp[r * NR..r * NR + nr];
                    for (col, &sv) in src.iter().enumerate() {
                        let dst = row_ptr.add(col);
                        *dst = sv + beta * (*dst);
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
            (16, 256, 128),
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
                    let mut c_serial = vec![1.5f32; m * n];
                    let mut c_scalar = vec![1.5f32; m * n];
                    let mut c_ref = vec![1.5f32; m * n];

                    sgemm(m, n, k, &a, &b, &mut c_fast, trans_a, trans_b, 1.25, 0.5).unwrap();
                    sgemm_serial(m, n, k, &a, &b, &mut c_serial, trans_a, trans_b, 1.25, 0.5)
                        .unwrap();
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
                            pack_panel_b(&b, n, k, pc, jc, kc, nc, trans_b, false, &mut pb);
                            let mut ic = 0;
                            while ic < m {
                                let mc = (m - ic).min(MC);
                                pack_panel_a(&a, m, k, ic, pc, mc, kc, trans_a, &mut pa);
                                unsafe {
                                    macro_kernel(
                                        mc,
                                        nc,
                                        kc,
                                        n,
                                        1.25,
                                        eff_beta,
                                        &pa,
                                        &pb,
                                        c_scalar.as_mut_ptr().add(ic * n + jc),
                                        false,
                                    );
                                }
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
                            (c_serial[idx] - c_ref[idx]).abs() < 1e-3,
                            "serial vs ref mismatch at ({m},{n},{k}) ta={trans_a} tb={trans_b} idx={idx}: {} vs {}",
                            c_serial[idx],
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
