// OchreML High-Performance f32 SGEMM Kernel (*≧ω≦*)
// BLIS-style cache-blocked panel packing (MC=72, KC=256, NC=128) + thread-local scratchpads
// + runtime-detected x86_64 AVX-512F 12x32 / 12x16 (_mm512_fmadd_ps)
// + x86_64 AVX2+FMA 6x16 (_mm256_fmadd_ps) + AVX2 8x8 SIMD transpose packing for trans_b
// + register/L1-fused bias + activation epilogue (sgemm_bias_act) + 2D Rayon parallel tiling
// + portable scalar fallback, with zero per-call heap allocations.

use crate::error::{OchreError, OchreResult};
use crate::nn::simd_math::{
    EpilogueAct, SimdIsa, add_bias_act_row_inplace, simd_add_bias_act_inplace,
};
use rayon::prelude::*;
use std::cell::RefCell;

pub const MC: usize = 72;
pub const KC: usize = 256;
pub const NC: usize = 128;
const MR: usize = 6;
const NR: usize = 16;
const MR_512: usize = 12;
const NR_512: usize = 32;

// Minimum FLOPs proxy (m*n*k) to amortize Rayon thread-pool overhead.
const RAYON_PAR_THRESHOLD: usize = 1_500_000;

thread_local! {
    static PACK_A_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static PACK_B_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmIsa {
    Scalar,
    Avx2Fma,
    Avx512F,
}

#[derive(Clone, Copy)]
struct TileEpilogue<'a> {
    bias: Option<&'a [f32]>,
    act: EpilogueAct,
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

#[inline]
fn detect_gemm_isa() -> GemmIsa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return GemmIsa::Avx512F;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return GemmIsa::Avx2Fma;
        }
    }
    GemmIsa::Scalar
}

#[inline]
fn has_avx2_for_pack() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn isa_to_simd(isa: GemmIsa) -> SimdIsa {
    match isa {
        GemmIsa::Avx512F => SimdIsa::Avx512F,
        GemmIsa::Avx2Fma => SimdIsa::Avx2Fma,
        GemmIsa::Scalar => SimdIsa::Scalar,
    }
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
        None,
        use_parallel,
    );
    Ok(())
}

/// Fused single-precision matrix multiplication with per-tile L1/register bias addition
/// and activation epilogue: `C = act(op(A) @ op(B) + bias)` in row-major order.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_bias_act(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    act: EpilogueAct,
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
) -> OchreResult<()> {
    validate_sgemm_dims(m, n, k, a.len(), b.len(), c.len(), trans_a, trans_b)?;
    let bias_checked = if let Some(bs) = bias {
        if bs.len() < n {
            return Err(OchreError::InvalidParameter(format!(
                "sgemm_bias_act bias slice too small: need at least {n}, got {} ( >_< )",
                bs.len()
            )));
        }
        Some(&bs[..n])
    } else {
        None
    };

    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        c[..m * n].fill(0.0);
        simd_add_bias_act_inplace(&mut c[..m * n], bias_checked, m, n, act);
        return Ok(());
    }

    let epilogue = if bias_checked.is_some() || act != EpilogueAct::None {
        Some(TileEpilogue {
            bias: bias_checked,
            act,
        })
    } else {
        None
    };

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
        1.0,
        0.0,
        epilogue,
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

    sgemm_core(m, n, k, a, b, c, trans_a, trans_b, alpha, beta, None, false);
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
                    m, n, k, a_slice, b_slice, c_slice, trans_a, trans_b, 1.0, 0.0, None, false,
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
                None,
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
/// total A+B panel packing perimeter across threads. Fast-paths narrow `n <= nr_blk` shapes.
#[inline]
fn choose_2d_grid(
    m: usize,
    n: usize,
    mr_blk: usize,
    nr_blk: usize,
    nthreads: usize,
) -> (usize, usize) {
    if n <= nr_blk {
        return (m.div_ceil(mr_blk).max(1).min(nthreads), 1);
    }
    if m <= mr_blk {
        return (1, n.div_ceil(nr_blk).max(1).min(nthreads));
    }

    let max_tm = m.div_ceil(mr_blk).max(1).min(nthreads);
    let max_tn = n.div_ceil(nr_blk).max(1).min(nthreads);

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
    epilogue: Option<TileEpilogue<'_>>,
    use_parallel: bool,
) {
    let isa = detect_gemm_isa();
    let use_avx2_pack = has_avx2_for_pack();
    let (mr_blk, nr_blk) = match isa {
        GemmIsa::Avx512F => {
            if n <= NR {
                (MR_512, NR)
            } else {
                (MR_512, NR_512)
            }
        }
        GemmIsa::Avx2Fma | GemmIsa::Scalar => (MR, NR),
    };

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
                epilogue,
                isa,
                use_avx2_pack,
            );
        }
        return;
    }

    let nthreads = rayon::current_num_threads();
    let (nt_m, nt_n) = choose_2d_grid(m, n, mr_blk, nr_blk, nthreads);
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
                epilogue,
                isa,
                use_avx2_pack,
            );
        }
        return;
    }

    let tile_m = m.div_ceil(nt_m).div_ceil(mr_blk) * mr_blk;
    let tile_n = n.div_ceil(nt_n).div_ceil(nr_blk) * nr_blk;
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
                m,
                n,
                k,
                ic0,
                tm,
                jc0,
                tn,
                a,
                b,
                ptr.0,
                trans_a,
                trans_b,
                alpha,
                beta,
                epilogue,
                isa,
                use_avx2_pack,
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
    epilogue: Option<TileEpilogue<'_>>,
    isa: GemmIsa,
    use_avx2_pack: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if isa == GemmIsa::Avx512F {
        if n <= NR {
            // SAFETY: Caller guarantees `c_base` tile ownership and `isa` verified AVX-512F.
            unsafe {
                sgemm_tile_serial_specialized::<MR_512, NR>(
                    m,
                    n,
                    k,
                    ic0,
                    tm,
                    jc0,
                    tn,
                    a,
                    b,
                    c_base,
                    trans_a,
                    trans_b,
                    alpha,
                    beta,
                    epilogue,
                    isa,
                    use_avx2_pack,
                );
            }
        } else {
            // SAFETY: Caller guarantees `c_base` tile ownership and `isa` verified AVX-512F.
            unsafe {
                sgemm_tile_serial_specialized::<MR_512, NR_512>(
                    m,
                    n,
                    k,
                    ic0,
                    tm,
                    jc0,
                    tn,
                    a,
                    b,
                    c_base,
                    trans_a,
                    trans_b,
                    alpha,
                    beta,
                    epilogue,
                    isa,
                    use_avx2_pack,
                );
            }
        }
        return;
    }

    // SAFETY: Caller guarantees `c_base` tile ownership.
    unsafe {
        sgemm_tile_serial_specialized::<MR, NR>(
            m,
            n,
            k,
            ic0,
            tm,
            jc0,
            tn,
            a,
            b,
            c_base,
            trans_a,
            trans_b,
            alpha,
            beta,
            epilogue,
            isa,
            use_avx2_pack,
        );
    }
}

/// Const-generic cache-blocked tile driver parameterized by micro-kernel shape `(MR_K, NR_K)`.
///
/// # Safety
/// `c_base` must point to a valid buffer of at least `m * n` `f32` elements, and no other
/// thread may concurrently read or write any element in `[ic0..ic0+tm) x [jc0..jc0+tn)`.
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_tile_serial_specialized<const MR_K: usize, const NR_K: usize>(
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
    epilogue: Option<TileEpilogue<'_>>,
    isa: GemmIsa,
    use_avx2_pack: bool,
) {
    let max_mc = tm.min(MC);
    let max_nc = tn.min(NC);
    let max_kc = k.min(KC);
    let need_a = max_mc.div_ceil(MR_K) * MR_K * max_kc;
    let need_b = max_nc.div_ceil(NR_K) * NR_K * max_kc;

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
                    let is_last_k = pc + kc >= k;
                    let panel_epilogue = if is_last_k {
                        epilogue.map(|ep| TileEpilogue {
                            bias: ep.bias.map(|bs| &bs[jc..jc + nc]),
                            act: ep.act,
                        })
                    } else {
                        None
                    };

                    pack_panel_b::<NR_K>(b, n, k, pc, jc, kc, nc, trans_b, use_avx2_pack, packed_b);

                    let mut di = 0;
                    while di < tm {
                        let mc = (tm - di).min(MC);
                        let ic = ic0 + di;
                        pack_panel_a::<MR_K>(a, m, k, ic, pc, mc, kc, trans_a, packed_a);
                        // SAFETY: `(ic..ic+mc, jc..jc+nc)` lies strictly inside
                        // `[ic0..ic0+tm) x [jc0..jc0+tn)`, which is exclusively owned by this call.
                        unsafe {
                            let c_tile = c_base.add(ic * n + jc);
                            macro_kernel::<MR_K, NR_K>(
                                mc,
                                nc,
                                kc,
                                n,
                                alpha,
                                eff_beta,
                                packed_a,
                                packed_b,
                                c_tile,
                                panel_epilogue,
                                isa,
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

/// Pack `A[ic..ic+mc, pc..pc+kc]` into contiguous `MR_K x kc` strips (zero-padded for `mr < MR_K`).
#[allow(clippy::too_many_arguments)]
fn pack_panel_a<const MR_K: usize>(
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
        let mr = (mc - ir).min(MR_K);
        let row0 = ic + ir;
        let dst_panel = &mut packed_a[dst_off..dst_off + MR_K * kc];

        if !trans_a {
            if mr == MR_K {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR_K>().0.iter_mut().enumerate() {
                    let p_idx = pc + p;
                    for (r, slot) in chunk.iter_mut().enumerate() {
                        *slot = a[(row0 + r) * k + p_idx];
                    }
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR_K>().0.iter_mut().enumerate() {
                    let p_idx = pc + p;
                    for (r, slot) in chunk.iter_mut().enumerate().take(mr) {
                        *slot = a[(row0 + r) * k + p_idx];
                    }
                    chunk[mr..MR_K].fill(0.0);
                }
            }
        } else {
            // A is stored as (k x m) row-major: op(A)[row, col] = a[col * m + row]
            if mr == MR_K {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR_K>().0.iter_mut().enumerate() {
                    let src_row = &a[(pc + p) * m + row0..(pc + p) * m + row0 + MR_K];
                    chunk.copy_from_slice(src_row);
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<MR_K>().0.iter_mut().enumerate() {
                    let src_row = &a[(pc + p) * m + row0..(pc + p) * m + row0 + mr];
                    chunk[..mr].copy_from_slice(src_row);
                    chunk[mr..MR_K].fill(0.0);
                }
            }
        }
        dst_off += MR_K * kc;
        ir += MR_K;
    }
}

/// Transposes an 8x8 f32 block from 8 contiguous source rows (stride `src_stride`)
/// into 8 destination rows of stride `dst_stride`, offset by `dst_col_off`.
///
/// # Safety
/// Caller must ensure AVX2 is supported, `src_base` has at least `7 * src_stride + 8` valid
/// `f32` elements, and `dst_base` has at least `7 * dst_stride + dst_col_off + 8` valid `f32` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8_to_stride_avx2(
    src_base: *const f32,
    src_stride: usize,
    dst_base: *mut f32,
    dst_col_off: usize,
    dst_stride: usize,
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
        _mm256_storeu_ps(d.add(dst_stride), o1);
        _mm256_storeu_ps(d.add(2 * dst_stride), o2);
        _mm256_storeu_ps(d.add(3 * dst_stride), o3);
        _mm256_storeu_ps(d.add(4 * dst_stride), o4);
        _mm256_storeu_ps(d.add(5 * dst_stride), o5);
        _mm256_storeu_ps(d.add(6 * dst_stride), o6);
        _mm256_storeu_ps(d.add(7 * dst_stride), o7);
    }
}

/// Pack `B[pc..pc+kc, jc..jc+nc]` into contiguous `kc x NR_K` strips (zero-padded for `nr < NR_K`).
#[allow(clippy::too_many_arguments)]
fn pack_panel_b<const NR_K: usize>(
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
        let nr = (nc - jr).min(NR_K);
        let col0 = jc + jr;
        let dst_panel = &mut packed_b[dst_off..dst_off + kc * NR_K];

        if !trans_b {
            if nr == NR_K {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR_K>().0.iter_mut().enumerate() {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + NR_K];
                    chunk.copy_from_slice(src);
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR_K>().0.iter_mut().enumerate() {
                    let src = &b[(pc + p) * n + col0..(pc + p) * n + col0 + nr];
                    chunk[..nr].copy_from_slice(src);
                    chunk[nr..NR_K].fill(0.0);
                }
            }
        } else {
            // B is stored as (n x k) row-major: op(B)[p, col] = b[col * k + p]
            let mut p_done = 0usize;
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && nr == NR_K && kc >= 8 && NR_K.is_multiple_of(8) {
                let p_blocks = kc / 8;
                let c_blocks = NR_K / 8;
                let dst_ptr = dst_panel.as_mut_ptr();
                for pb in 0..p_blocks {
                    let p0 = pb * 8;
                    for cb in 0..c_blocks {
                        let col_off = cb * 8;
                        let src_ptr = b[((col0 + col_off) * k + pc + p0)..].as_ptr();
                        // SAFETY: `use_avx2` verified AVX2 support; `col0 + col_off + 8 <= n` and
                        // `pc + p0 + 8 <= k`, so both 8x8 source block and destination rows are in bounds.
                        unsafe {
                            transpose_8x8_to_stride_avx2(
                                src_ptr,
                                k,
                                dst_ptr.add(p0 * NR_K),
                                col_off,
                                NR_K,
                            );
                        }
                    }
                }
                p_done = p_blocks * 8;
            }
            let _ = use_avx2;

            if nr == NR_K {
                for p in p_done..kc {
                    let chunk = &mut dst_panel[p * NR_K..(p + 1) * NR_K];
                    let p_idx = pc + p;
                    for (c, dst_val) in chunk.iter_mut().enumerate() {
                        *dst_val = b[(col0 + c) * k + p_idx];
                    }
                }
            } else {
                for (p, chunk) in dst_panel.as_chunks_mut::<NR_K>().0.iter_mut().enumerate() {
                    let p_idx = pc + p;
                    for (c, slot) in chunk.iter_mut().enumerate().take(nr) {
                        *slot = b[(col0 + c) * k + p_idx];
                    }
                    chunk[nr..NR_K].fill(0.0);
                }
            }
        }
        dst_off += kc * NR_K;
        jr += NR_K;
    }
}

/// # Safety
/// `c_ptr` must point to the top-left of a valid `(mc x nc)` sub-matrix with row stride `ldc`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn macro_kernel<const MR_K: usize, const NR_K: usize>(
    mc: usize,
    nc: usize,
    kc: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c_ptr: *mut f32,
    epilogue: Option<TileEpilogue<'_>>,
    isa: GemmIsa,
) {
    let mut jr = 0;
    let mut b_off = 0;
    while jr < nc {
        let nr = (nc - jr).min(NR_K);
        let pb = &packed_b[b_off..b_off + kc * NR_K];
        let strip_epilogue = epilogue.map(|ep| TileEpilogue {
            bias: ep.bias.map(|bs| &bs[jr..jr + nr]),
            act: ep.act,
        });

        let mut ir = 0;
        let mut a_off = 0;
        while ir < mc {
            let mr = (mc - ir).min(MR_K);
            let pa = &packed_a[a_off..a_off + MR_K * kc];
            // SAFETY: `ir + mr <= mc` and `jr + nr <= nc`, so `c_tile` is in bounds of `c_ptr`.
            let c_tile = unsafe { c_ptr.add(ir * ldc + jr) };

            #[cfg(target_arch = "x86_64")]
            match isa {
                GemmIsa::Avx512F if MR_K == MR_512 && NR_K == NR_512 => {
                    // SAFETY: `detect_gemm_isa()` verified runtime AVX-512F support, and `c_tile`
                    // has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
                    unsafe {
                        microkernel_12x32_avx512(
                            kc,
                            mr,
                            nr,
                            ldc,
                            alpha,
                            beta,
                            pa,
                            pb,
                            c_tile,
                            strip_epilogue,
                        );
                    }
                    a_off += MR_K * kc;
                    ir += MR_K;
                    continue;
                }
                GemmIsa::Avx512F if MR_K == MR_512 && NR_K == NR => {
                    // SAFETY: `detect_gemm_isa()` verified runtime AVX-512F support, and `c_tile`
                    // has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
                    unsafe {
                        microkernel_12x16_avx512(
                            kc,
                            mr,
                            nr,
                            ldc,
                            alpha,
                            beta,
                            pa,
                            pb,
                            c_tile,
                            strip_epilogue,
                        );
                    }
                    a_off += MR_K * kc;
                    ir += MR_K;
                    continue;
                }
                GemmIsa::Avx2Fma if MR_K == MR && NR_K == NR => {
                    // SAFETY: `detect_gemm_isa()` verified runtime AVX2+FMA support, and `c_tile`
                    // has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
                    unsafe {
                        microkernel_6x16_avx2_fma(
                            kc,
                            mr,
                            nr,
                            ldc,
                            alpha,
                            beta,
                            pa,
                            pb,
                            c_tile,
                            strip_epilogue,
                        );
                    }
                    a_off += MR_K * kc;
                    ir += MR_K;
                    continue;
                }
                _ => {}
            }
            let _ = isa;
            // SAFETY: `c_tile` has valid capacity for `mr` rows of stride `ldc` with `nr` columns.
            unsafe {
                microkernel_scalar::<MR_K, NR_K>(
                    kc,
                    mr,
                    nr,
                    ldc,
                    alpha,
                    beta,
                    pa,
                    pb,
                    c_tile,
                    strip_epilogue,
                    isa,
                );
            }

            a_off += MR_K * kc;
            ir += MR_K;
        }

        b_off += kc * NR_K;
        jr += NR_K;
    }
}

/// Portable scalar `MR_K x NR_K` micro-kernel operating on packed A (`MR_K x kc`) and B (`kc x NR_K`).
///
/// # Safety
/// `c_ptr` must have valid writable capacity for `mr` rows of stride `ldc` with `nr` cols.
#[allow(clippy::too_many_arguments)]
unsafe fn microkernel_scalar<const MR_K: usize, const NR_K: usize>(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c_ptr: *mut f32,
    epilogue: Option<TileEpilogue<'_>>,
    isa: GemmIsa,
) {
    let mut acc = [[0.0f32; NR_K]; MR_K];
    for p in 0..kc {
        let a_base = p * MR_K;
        let b_base = p * NR_K;
        for (r, acc_row) in acc.iter_mut().enumerate() {
            let av = pa[a_base + r];
            for (col, slot) in acc_row.iter_mut().enumerate() {
                *slot += av * pb[b_base + col];
            }
        }
    }

    let simd_isa = isa_to_simd(isa);

    // SAFETY: Caller guarantees `c_ptr` has `mr` rows of stride `ldc` with at least `nr` cols.
    unsafe {
        for (r, acc_row) in acc.iter_mut().enumerate().take(mr) {
            let row_ptr = c_ptr.add(r * ldc);
            if beta == 0.0 {
                if alpha != 1.0 {
                    for slot in acc_row.iter_mut().take(nr) {
                        *slot *= alpha;
                    }
                }
            } else {
                for (col, slot) in acc_row.iter_mut().enumerate().take(nr) {
                    *slot = alpha * (*slot) + beta * (*row_ptr.add(col));
                }
            }
            if let Some(ep) = epilogue {
                add_bias_act_row_inplace(&mut acc_row[..nr], ep.bias, ep.act, simd_isa);
            }
            std::ptr::copy_nonoverlapping(acc_row.as_ptr(), row_ptr, nr);
        }
    }
}

/// x86_64 AVX2 + FMA 6x16 register micro-kernel using `_mm256_fmadd_ps`,
/// with optional register-fused bias + activation epilogue.
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
    epilogue: Option<TileEpilogue<'_>>,
) {
    use crate::nn::simd_math::avx2::apply_act_ps;
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
            let bias_vecs = epilogue.and_then(|ep| {
                ep.bias.map(|bs| {
                    let bp = bs.as_ptr();
                    (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(8)))
                })
            });
            let act = epilogue.map_or(EpilogueAct::None, |ep| ep.act);

            for (r, &(mut r0, mut r1)) in rows.iter().enumerate().take(mr) {
                let row_ptr = c_ptr.add(r * ldc);
                if beta == 1.0 {
                    r0 = _mm256_add_ps(_mm256_loadu_ps(row_ptr), r0);
                    r1 = _mm256_add_ps(_mm256_loadu_ps(row_ptr.add(8)), r1);
                } else if beta != 0.0 {
                    let vbeta = _mm256_set1_ps(beta);
                    r0 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(row_ptr), r0);
                    r1 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(row_ptr.add(8)), r1);
                }
                if let Some((vb0, vb1)) = bias_vecs {
                    r0 = _mm256_add_ps(r0, vb0);
                    r1 = _mm256_add_ps(r1, vb1);
                }
                if act != EpilogueAct::None {
                    r0 = apply_act_ps(r0, act);
                    r1 = apply_act_ps(r1, act);
                }
                _mm256_storeu_ps(row_ptr, r0);
                _mm256_storeu_ps(row_ptr.add(8), r1);
            }
        } else {
            let mut tmp = [0.0f32; MR * NR];
            let tp = tmp.as_mut_ptr();
            for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
                _mm256_storeu_ps(tp.add(r * NR), r0);
                _mm256_storeu_ps(tp.add(r * NR + 8), r1);
            }

            for r in 0..mr {
                let row_ptr = c_ptr.add(r * ldc);
                let row_buf = &mut tmp[r * NR..r * NR + nr];
                if beta != 0.0 {
                    for (col, slot) in row_buf.iter_mut().enumerate() {
                        *slot += beta * (*row_ptr.add(col));
                    }
                }
                if let Some(ep) = epilogue {
                    add_bias_act_row_inplace(row_buf, ep.bias, ep.act, SimdIsa::Avx2Fma);
                }
                std::ptr::copy_nonoverlapping(row_buf.as_ptr(), row_ptr, nr);
            }
        }
    }
}

/// x86_64 AVX-512F 12x32 register micro-kernel using 24 `zmm` accumulators (`_mm512_fmadd_ps`),
/// with optional register-fused bias + activation epilogue.
///
/// # Safety
/// Caller must ensure CPU supports `avx512f`, `pa.len() >= kc * 12`, `pb.len() >= kc * 32`,
/// and `c_ptr` has valid capacity for `mr` rows of stride `ldc` with `nr` cols.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn microkernel_12x32_avx512(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c_ptr: *mut f32,
    epilogue: Option<TileEpilogue<'_>>,
) {
    use crate::nn::simd_math::avx512::apply_act_ps;
    use std::arch::x86_64::*;

    let mut c00 = _mm512_setzero_ps();
    let mut c01 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c20 = _mm512_setzero_ps();
    let mut c21 = _mm512_setzero_ps();
    let mut c30 = _mm512_setzero_ps();
    let mut c31 = _mm512_setzero_ps();
    let mut c40 = _mm512_setzero_ps();
    let mut c41 = _mm512_setzero_ps();
    let mut c50 = _mm512_setzero_ps();
    let mut c51 = _mm512_setzero_ps();
    let mut c60 = _mm512_setzero_ps();
    let mut c61 = _mm512_setzero_ps();
    let mut c70 = _mm512_setzero_ps();
    let mut c71 = _mm512_setzero_ps();
    let mut c80 = _mm512_setzero_ps();
    let mut c81 = _mm512_setzero_ps();
    let mut c90 = _mm512_setzero_ps();
    let mut c91 = _mm512_setzero_ps();
    let mut c10_0 = _mm512_setzero_ps();
    let mut c10_1 = _mm512_setzero_ps();
    let mut c11_0 = _mm512_setzero_ps();
    let mut c11_1 = _mm512_setzero_ps();

    let mut a_ptr = pa.as_ptr();
    let mut b_ptr = pb.as_ptr();

    let k_unroll = kc / 4;
    let k_rem = kc % 4;

    macro_rules! fma_step_12x32 {
        ($ap:expr, $bp:expr) => {{
            let b0 = _mm512_loadu_ps($bp);
            let b1 = _mm512_loadu_ps($bp.add(16));

            let a0 = _mm512_set1_ps(*$ap);
            c00 = _mm512_fmadd_ps(a0, b0, c00);
            c01 = _mm512_fmadd_ps(a0, b1, c01);

            let a1 = _mm512_set1_ps(*$ap.add(1));
            c10 = _mm512_fmadd_ps(a1, b0, c10);
            c11 = _mm512_fmadd_ps(a1, b1, c11);

            let a2 = _mm512_set1_ps(*$ap.add(2));
            c20 = _mm512_fmadd_ps(a2, b0, c20);
            c21 = _mm512_fmadd_ps(a2, b1, c21);

            let a3 = _mm512_set1_ps(*$ap.add(3));
            c30 = _mm512_fmadd_ps(a3, b0, c30);
            c31 = _mm512_fmadd_ps(a3, b1, c31);

            let a4 = _mm512_set1_ps(*$ap.add(4));
            c40 = _mm512_fmadd_ps(a4, b0, c40);
            c41 = _mm512_fmadd_ps(a4, b1, c41);

            let a5 = _mm512_set1_ps(*$ap.add(5));
            c50 = _mm512_fmadd_ps(a5, b0, c50);
            c51 = _mm512_fmadd_ps(a5, b1, c51);

            let a6 = _mm512_set1_ps(*$ap.add(6));
            c60 = _mm512_fmadd_ps(a6, b0, c60);
            c61 = _mm512_fmadd_ps(a6, b1, c61);

            let a7 = _mm512_set1_ps(*$ap.add(7));
            c70 = _mm512_fmadd_ps(a7, b0, c70);
            c71 = _mm512_fmadd_ps(a7, b1, c71);

            let a8 = _mm512_set1_ps(*$ap.add(8));
            c80 = _mm512_fmadd_ps(a8, b0, c80);
            c81 = _mm512_fmadd_ps(a8, b1, c81);

            let a9 = _mm512_set1_ps(*$ap.add(9));
            c90 = _mm512_fmadd_ps(a9, b0, c90);
            c91 = _mm512_fmadd_ps(a9, b1, c91);

            let a10 = _mm512_set1_ps(*$ap.add(10));
            c10_0 = _mm512_fmadd_ps(a10, b0, c10_0);
            c10_1 = _mm512_fmadd_ps(a10, b1, c10_1);

            let a11 = _mm512_set1_ps(*$ap.add(11));
            c11_0 = _mm512_fmadd_ps(a11, b0, c11_0);
            c11_1 = _mm512_fmadd_ps(a11, b1, c11_1);
        }};
    }

    // SAFETY: `pa` has `kc * 12` elements and `pb` has `kc * 32` elements, so pointer
    // increments across `4 * k_unroll + k_rem == kc` steps remain strictly in bounds.
    unsafe {
        for _ in 0..k_unroll {
            for step in 0..4 {
                fma_step_12x32!(a_ptr.add(step * MR_512), b_ptr.add(step * NR_512));
            }
            a_ptr = a_ptr.add(4 * MR_512);
            b_ptr = b_ptr.add(4 * NR_512);
        }

        for _ in 0..k_rem {
            fma_step_12x32!(a_ptr, b_ptr);
            a_ptr = a_ptr.add(MR_512);
            b_ptr = b_ptr.add(NR_512);
        }

        let mut rows = [
            (c00, c01),
            (c10, c11),
            (c20, c21),
            (c30, c31),
            (c40, c41),
            (c50, c51),
            (c60, c61),
            (c70, c71),
            (c80, c81),
            (c90, c91),
            (c10_0, c10_1),
            (c11_0, c11_1),
        ];

        if alpha != 1.0 {
            let valpha = _mm512_set1_ps(alpha);
            for (r0, r1) in rows.iter_mut().take(mr) {
                *r0 = _mm512_mul_ps(*r0, valpha);
                *r1 = _mm512_mul_ps(*r1, valpha);
            }
        }

        if nr == NR_512 {
            let bias_vecs = epilogue.and_then(|ep| {
                ep.bias.map(|bs| {
                    let bp = bs.as_ptr();
                    (_mm512_loadu_ps(bp), _mm512_loadu_ps(bp.add(16)))
                })
            });
            let act = epilogue.map_or(EpilogueAct::None, |ep| ep.act);

            for (r, &(mut r0, mut r1)) in rows.iter().enumerate().take(mr) {
                let row_ptr = c_ptr.add(r * ldc);
                if beta == 1.0 {
                    r0 = _mm512_add_ps(_mm512_loadu_ps(row_ptr), r0);
                    r1 = _mm512_add_ps(_mm512_loadu_ps(row_ptr.add(16)), r1);
                } else if beta != 0.0 {
                    let vbeta = _mm512_set1_ps(beta);
                    r0 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(row_ptr), r0);
                    r1 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(row_ptr.add(16)), r1);
                }
                if let Some((vb0, vb1)) = bias_vecs {
                    r0 = _mm512_add_ps(r0, vb0);
                    r1 = _mm512_add_ps(r1, vb1);
                }
                if act != EpilogueAct::None {
                    r0 = apply_act_ps(r0, act);
                    r1 = apply_act_ps(r1, act);
                }
                _mm512_storeu_ps(row_ptr, r0);
                _mm512_storeu_ps(row_ptr.add(16), r1);
            }
        } else {
            let mut tmp = [0.0f32; MR_512 * NR_512];
            let tp = tmp.as_mut_ptr();
            for (r, &(r0, r1)) in rows.iter().enumerate().take(mr) {
                _mm512_storeu_ps(tp.add(r * NR_512), r0);
                _mm512_storeu_ps(tp.add(r * NR_512 + 16), r1);
            }

            for r in 0..mr {
                let row_ptr = c_ptr.add(r * ldc);
                let row_buf = &mut tmp[r * NR_512..r * NR_512 + nr];
                if beta != 0.0 {
                    for (col, slot) in row_buf.iter_mut().enumerate() {
                        *slot += beta * (*row_ptr.add(col));
                    }
                }
                if let Some(ep) = epilogue {
                    add_bias_act_row_inplace(row_buf, ep.bias, ep.act, SimdIsa::Avx512F);
                }
                std::ptr::copy_nonoverlapping(row_buf.as_ptr(), row_ptr, nr);
            }
        }
    }
}

/// x86_64 AVX-512F 12x16 register micro-kernel for narrow `n <= 16` matrices (e.g. final classifier
/// heads like `N=10`), avoiding wasted work on padded columns `16..32`.
///
/// # Safety
/// Caller must ensure CPU supports `avx512f`, `pa.len() >= kc * 12`, `pb.len() >= kc * 16`,
/// and `c_ptr` has valid capacity for `mr` rows of stride `ldc` with `nr` cols.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn microkernel_12x16_avx512(
    kc: usize,
    mr: usize,
    nr: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
    pa: &[f32],
    pb: &[f32],
    c_ptr: *mut f32,
    epilogue: Option<TileEpilogue<'_>>,
) {
    use crate::nn::simd_math::avx512::apply_act_ps;
    use std::arch::x86_64::*;

    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();
    let mut c6 = _mm512_setzero_ps();
    let mut c7 = _mm512_setzero_ps();
    let mut c8 = _mm512_setzero_ps();
    let mut c9 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();

    let mut a_ptr = pa.as_ptr();
    let mut b_ptr = pb.as_ptr();

    let k_unroll = kc / 4;
    let k_rem = kc % 4;

    macro_rules! fma_step_12x16 {
        ($ap:expr, $bp:expr) => {{
            let b0 = _mm512_loadu_ps($bp);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap), b0, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(1)), b0, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(2)), b0, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(3)), b0, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(4)), b0, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(5)), b0, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(6)), b0, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(7)), b0, c7);
            c8 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(8)), b0, c8);
            c9 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(9)), b0, c9);
            c10 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(10)), b0, c10);
            c11 = _mm512_fmadd_ps(_mm512_set1_ps(*$ap.add(11)), b0, c11);
        }};
    }

    // SAFETY: `pa` has `kc * 12` elements and `pb` has `kc * 16` elements.
    unsafe {
        for _ in 0..k_unroll {
            for step in 0..4 {
                fma_step_12x16!(a_ptr.add(step * MR_512), b_ptr.add(step * NR));
            }
            a_ptr = a_ptr.add(4 * MR_512);
            b_ptr = b_ptr.add(4 * NR);
        }

        for _ in 0..k_rem {
            fma_step_12x16!(a_ptr, b_ptr);
            a_ptr = a_ptr.add(MR_512);
            b_ptr = b_ptr.add(NR);
        }

        let mut rows = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11];
        if alpha != 1.0 {
            let valpha = _mm512_set1_ps(alpha);
            for r0 in rows.iter_mut().take(mr) {
                *r0 = _mm512_mul_ps(*r0, valpha);
            }
        }

        if nr == NR {
            let bias_vec = epilogue.and_then(|ep| ep.bias.map(|bs| _mm512_loadu_ps(bs.as_ptr())));
            let act = epilogue.map_or(EpilogueAct::None, |ep| ep.act);

            for (r, &mut mut r0) in rows.iter_mut().enumerate().take(mr) {
                let row_ptr = c_ptr.add(r * ldc);
                if beta == 1.0 {
                    r0 = _mm512_add_ps(_mm512_loadu_ps(row_ptr), r0);
                } else if beta != 0.0 {
                    let vbeta = _mm512_set1_ps(beta);
                    r0 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(row_ptr), r0);
                }
                if let Some(vb) = bias_vec {
                    r0 = _mm512_add_ps(r0, vb);
                }
                if act != EpilogueAct::None {
                    r0 = apply_act_ps(r0, act);
                }
                _mm512_storeu_ps(row_ptr, r0);
            }
        } else {
            let mut tmp = [0.0f32; MR_512 * NR];
            let tp = tmp.as_mut_ptr();
            for (r, &r0) in rows.iter().enumerate().take(mr) {
                _mm512_storeu_ps(tp.add(r * NR), r0);
            }
            for r in 0..mr {
                let row_ptr = c_ptr.add(r * ldc);
                let row_buf = &mut tmp[r * NR..r * NR + nr];
                if beta != 0.0 {
                    for (col, slot) in row_buf.iter_mut().enumerate() {
                        *slot += beta * (*row_ptr.add(col));
                    }
                }
                if let Some(ep) = epilogue {
                    add_bias_act_row_inplace(row_buf, ep.bias, ep.act, SimdIsa::Avx512F);
                }
                std::ptr::copy_nonoverlapping(row_buf.as_ptr(), row_ptr, nr);
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
            (256, 10, 128),
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

                    // Explicitly verify portable scalar kernel (both 6x16 and 12x32 packing!)
                    unsafe {
                        sgemm_tile_serial_specialized::<MR, NR>(
                            m,
                            n,
                            k,
                            0,
                            m,
                            0,
                            n,
                            &a,
                            &b,
                            c_scalar.as_mut_ptr(),
                            trans_a,
                            trans_b,
                            1.25,
                            0.5,
                            None,
                            GemmIsa::Scalar,
                            false,
                        );
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
    fn test_sgemm_12x32_packing_parity() {
        // Exercises MR_512=12, NR_512=32 panel packing (including AVX2 8x8 transpose packing for trans_b)
        // on all CPUs, even without AVX-512F hardware.
        let use_avx2_pack = has_avx2_for_pack();
        for &(m, n, k) in &[(13, 35, 19), (25, 64, 33), (73, 131, 260)] {
            for trans_a in [false, true] {
                for trans_b in [false, true] {
                    let a: Vec<f32> = (0..m * k)
                        .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.1)
                        .collect();
                    let b: Vec<f32> = (0..k * n)
                        .map(|i| ((i * 11 % 17) as f32 - 8.0) * 0.1)
                        .collect();
                    let mut c_12x32 = vec![0.0f32; m * n];
                    let mut c_ref = vec![0.0f32; m * n];
                    unsafe {
                        sgemm_tile_serial_specialized::<MR_512, NR_512>(
                            m,
                            n,
                            k,
                            0,
                            m,
                            0,
                            n,
                            &a,
                            &b,
                            c_12x32.as_mut_ptr(),
                            trans_a,
                            trans_b,
                            1.0,
                            0.0,
                            None,
                            GemmIsa::Scalar,
                            use_avx2_pack,
                        );
                    }
                    naive_sgemm(m, n, k, &a, &b, &mut c_ref, trans_a, trans_b, 1.0, 0.0);
                    for idx in 0..m * n {
                        assert!(
                            (c_12x32[idx] - c_ref[idx]).abs() < 1e-3,
                            "12x32 pack mismatch at ({m},{n},{k}) ta={trans_a} tb={trans_b} idx={idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_sgemm_bias_act_fused_epilogue_parity() {
        // Includes k > KC (300 > 256) to verify epilogue runs ONLY on the final K block,
        // narrow N=10, and large parallel shape (> 1.5M FLOPs).
        let shapes = [(7, 10, 19), (32, 48, 300), (128, 128, 128)];
        let acts = [
            EpilogueAct::None,
            EpilogueAct::ReLU,
            EpilogueAct::Sigmoid,
            EpilogueAct::Tanh,
            EpilogueAct::GELU,
            EpilogueAct::SiLU,
        ];

        for &(m, n, k) in &shapes {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.04)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.04)
                .collect();
            let bias: Vec<f32> = (0..n).map(|j| ((j % 11) as f32 - 5.0) * 0.1).collect();

            for &act in &acts {
                for trans_b in [false, true] {
                    let mut c_fused = vec![f32::NAN; m * n];
                    sgemm_bias_act(
                        m,
                        n,
                        k,
                        &a,
                        &b,
                        Some(&bias),
                        act,
                        &mut c_fused,
                        false,
                        trans_b,
                    )
                    .unwrap();

                    let mut c_ref = vec![0.0f32; m * n];
                    naive_sgemm(m, n, k, &a, &b, &mut c_ref, false, trans_b, 1.0, 0.0);
                    simd_add_bias_act_inplace(&mut c_ref, Some(&bias), m, n, act);

                    for idx in 0..m * n {
                        assert!(
                            (c_fused[idx] - c_ref[idx]).abs() < 2e-4,
                            "sgemm_bias_act({act:?}) mismatch at ({m},{n},{k}) tb={trans_b} idx={idx}: {} vs {}",
                            c_fused[idx],
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
        let m = 80;
        let n = 80;
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
