// OchreML CPU SIMD Transcendentals & Vectorized NN Kernels (*≧ω≦*)
// Runtime-dispatched x86_64 AVX-512F (_mm512_*), AVX2+FMA (_mm256_*), and portable scalar
// fallbacks for exp, sigmoid, tanh, GELU, SiLU, ReLU, Softmax, fused bias+activation, and AdamW.

use rayon::prelude::*;

/// Minimum element count before Rayon multi-threading is enabled for elementwise/row SIMD kernels.
const SIMD_RAYON_THRESHOLD: usize = 131_072;

// Math constants for fast exp / GELU approximations.
const EXP_CLAMP_LO: f32 = -88.0;
const EXP_CLAMP_HI: f32 = 88.0;
const LOG2E: f32 = core::f32::consts::LOG2_E;
// Cody-Waite split of ln(2): LN2_HI has trailing mantissa zeros so k * LN2_HI is exact for |k| <= 127.
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;

// Degree-6 Taylor polynomial coefficients for exp(r) on r in [-ln(2)/2, ln(2)/2].
const EXP_C2: f32 = 0.5;
const EXP_C3: f32 = 0.166_666_67;
const EXP_C4: f32 = 0.041_666_668;
const EXP_C5: f32 = 0.008_333_334;
const EXP_C6: f32 = 0.001_388_888_9;

// Clamp |x| for tanh so -2*|x| never overflows and tanh(±9.5) saturates cleanly to ±1.0 in f32.
const TANH_CLAMP: f32 = 9.5;

// Tanh-approximated GELU constants: sqrt(2/pi) and 0.044715.
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_K: f32 = 0.044_715;
const GELU_3K: f32 = 3.0 * 0.044_715;

/// Activation function fused into GEMM / bias-add epilogues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpilogueAct {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimdIsa {
    Scalar,
    Avx2Fma,
    Avx512F,
}

#[inline]
pub(crate) fn detect_simd_isa() -> SimdIsa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return SimdIsa::Avx512F;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdIsa::Avx2Fma;
        }
    }
    SimdIsa::Scalar
}

// ============================================================================
// Portable Scalar Fast Math (exact same polynomial structure as SIMD lanes)
// ============================================================================

#[inline]
pub(crate) fn exp_f32_scalar(x: f32) -> f32 {
    let xc = x.clamp(EXP_CLAMP_LO, EXP_CLAMP_HI);
    let kf = (xc * LOG2E).round_ties_even();
    let ki = kf as i32;
    let r = (-kf).mul_add(LN2_HI, xc);
    let r = (-kf).mul_add(LN2_LO, r);

    let r2 = r * r;
    let r4 = r2 * r2;
    let p01 = 1.0 + r;
    let p23 = EXP_C3.mul_add(r, EXP_C2);
    let p45 = EXP_C5.mul_add(r, EXP_C4);
    let p03 = p23.mul_add(r2, p01);
    let p46 = EXP_C6.mul_add(r2, p45);
    let poly = p46.mul_add(r4, p03);

    let scale = f32::from_bits(((ki + 127) as u32) << 23);
    poly * scale
}

#[inline]
pub(crate) fn sigmoid_f32_scalar(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32_scalar(-x))
}

#[inline]
pub(crate) fn tanh_f32_scalar(x: f32) -> f32 {
    let abs_x = x.abs().min(TANH_CLAMP);
    let e = exp_f32_scalar(-2.0 * abs_x);
    let y = (1.0 - e) / (1.0 + e);
    y.copysign(x)
}

#[inline]
pub(crate) fn gelu_fwd_f32_scalar(x: f32) -> f32 {
    let x2 = x * x;
    let inner = (GELU_K * x2).mul_add(x, x);
    let u = GELU_SQRT_2_OVER_PI * inner;
    0.5 * x * (1.0 + tanh_f32_scalar(u))
}

#[inline]
pub(crate) fn gelu_bwd_f32_scalar(dy: f32, x: f32) -> f32 {
    let x2 = x * x;
    let inner = (GELU_K * x2).mul_add(x, x);
    let u = GELU_SQRT_2_OVER_PI * inner;
    let t = tanh_f32_scalar(u);
    let sech2 = (-t).mul_add(t, 1.0);
    let du = GELU_SQRT_2_OVER_PI * GELU_3K.mul_add(x2, 1.0);
    let deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
    dy * deriv
}

#[inline]
pub(crate) fn silu_fwd_f32_scalar(x: f32) -> f32 {
    x * sigmoid_f32_scalar(x)
}

#[inline]
pub(crate) fn silu_bwd_f32_scalar(dy: f32, x: f32) -> f32 {
    let s = sigmoid_f32_scalar(x);
    let deriv = s * (1.0 + x * (1.0 - s));
    dy * deriv
}

#[inline]
pub(crate) fn apply_act_scalar(x: f32, act: EpilogueAct) -> f32 {
    match act {
        EpilogueAct::None => x,
        EpilogueAct::ReLU => x.max(0.0),
        EpilogueAct::Sigmoid => sigmoid_f32_scalar(x),
        EpilogueAct::Tanh => tanh_f32_scalar(x),
        EpilogueAct::GELU => gelu_fwd_f32_scalar(x),
        EpilogueAct::SiLU => silu_fwd_f32_scalar(x),
    }
}

// ============================================================================
// x86_64 AVX2 + FMA Vector Primitives (8 x f32)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Fast Cody-Waite + degree-6 Estrin `exp(x)` on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exp_ps(x: __m256) -> __m256 {
        let xc = _mm256_max_ps(
            _mm256_set1_ps(EXP_CLAMP_LO),
            _mm256_min_ps(x, _mm256_set1_ps(EXP_CLAMP_HI)),
        );
        let t = _mm256_mul_ps(xc, _mm256_set1_ps(LOG2E));
        let kf = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(t);
        let ki = _mm256_cvtps_epi32(kf);

        // Cody-Waite range reduction: r = xc - kf * LN2_HI - kf * LN2_LO
        let r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_HI), xc);
        let r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_LO), r);

        // Degree-6 Estrin polynomial
        let r2 = _mm256_mul_ps(r, r);
        let r4 = _mm256_mul_ps(r2, r2);
        let p01 = _mm256_add_ps(_mm256_set1_ps(1.0), r);
        let p23 = _mm256_fmadd_ps(_mm256_set1_ps(EXP_C3), r, _mm256_set1_ps(EXP_C2));
        let p45 = _mm256_fmadd_ps(_mm256_set1_ps(EXP_C5), r, _mm256_set1_ps(EXP_C4));
        let p03 = _mm256_fmadd_ps(p23, r2, p01);
        let p46 = _mm256_fmadd_ps(_mm256_set1_ps(EXP_C6), r2, p45);
        let poly = _mm256_fmadd_ps(p46, r4, p03);

        // 2^ki via IEEE-754 exponent bias
        let exp_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(ki, _mm256_set1_epi32(127)));
        let scale = _mm256_castsi256_ps(exp_bits);
        _mm256_mul_ps(poly, scale)
    }

    /// Fast `sigmoid(x)` on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn sigmoid_ps(x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
        // SAFETY: AVX2+FMA verified by caller.
        let e = unsafe { exp_ps(neg_x) };
        _mm256_div_ps(one, _mm256_add_ps(one, e))
    }

    /// Fast `tanh(x)` on 8 `f32` lanes via `sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))`.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn tanh_ps(x: __m256) -> __m256 {
        let sign_mask = _mm256_set1_ps(-0.0);
        let sign = _mm256_and_ps(x, sign_mask);
        let abs_x = _mm256_andnot_ps(sign_mask, x);
        let clamped = _mm256_min_ps(abs_x, _mm256_set1_ps(TANH_CLAMP));
        let neg2x = _mm256_mul_ps(clamped, _mm256_set1_ps(-2.0));
        // SAFETY: AVX2+FMA verified by caller.
        let e = unsafe { exp_ps(neg2x) };
        let one = _mm256_set1_ps(1.0);
        let y = _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
        _mm256_or_ps(y, sign)
    }

    /// Fast tanh-approximated `GELU(x)` on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn gelu_fwd_ps(x: __m256) -> __m256 {
        let x2 = _mm256_mul_ps(x, x);
        let kx2 = _mm256_mul_ps(_mm256_set1_ps(GELU_K), x2);
        let inner = _mm256_fmadd_ps(kx2, x, x);
        let u = _mm256_mul_ps(_mm256_set1_ps(GELU_SQRT_2_OVER_PI), inner);
        // SAFETY: AVX2+FMA verified by caller.
        let t = unsafe { tanh_ps(u) };
        let half_x = _mm256_mul_ps(_mm256_set1_ps(0.5), x);
        _mm256_mul_ps(half_x, _mm256_add_ps(_mm256_set1_ps(1.0), t))
    }

    /// Fast tanh-approximated GELU backward on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn gelu_bwd_ps(dy: __m256, x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let c = _mm256_set1_ps(GELU_SQRT_2_OVER_PI);
        let x2 = _mm256_mul_ps(x, x);
        let inner = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(GELU_K), x2), x, x);
        let u = _mm256_mul_ps(c, inner);
        // SAFETY: AVX2+FMA verified by caller.
        let t = unsafe { tanh_ps(u) };
        let sech2 = _mm256_fnmadd_ps(t, t, one);
        let du = _mm256_mul_ps(c, _mm256_fmadd_ps(_mm256_set1_ps(GELU_3K), x2, one));
        let term1 = _mm256_mul_ps(half, _mm256_add_ps(one, t));
        let term2 = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_mul_ps(sech2, du));
        _mm256_mul_ps(dy, _mm256_add_ps(term1, term2))
    }

    /// Fast `SiLU(x) = x * sigmoid(x)` on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn silu_fwd_ps(x: __m256) -> __m256 {
        // SAFETY: AVX2+FMA verified by caller.
        let s = unsafe { sigmoid_ps(x) };
        _mm256_mul_ps(x, s)
    }

    /// Fast SiLU backward on 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn silu_bwd_ps(dy: __m256, x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        // SAFETY: AVX2+FMA verified by caller.
        let s = unsafe { sigmoid_ps(x) };
        let one_minus_s = _mm256_sub_ps(one, s);
        let deriv = _mm256_mul_ps(s, _mm256_fmadd_ps(x, one_minus_s, one));
        _mm256_mul_ps(dy, deriv)
    }

    /// Apply `EpilogueAct` to an 8-lane `f32` vector register.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn apply_act_ps(x: __m256, act: EpilogueAct) -> __m256 {
        // SAFETY: AVX2+FMA verified by caller.
        unsafe {
            match act {
                EpilogueAct::None => x,
                EpilogueAct::ReLU => _mm256_max_ps(x, _mm256_setzero_ps()),
                EpilogueAct::Sigmoid => sigmoid_ps(x),
                EpilogueAct::Tanh => tanh_ps(x),
                EpilogueAct::GELU => gelu_fwd_ps(x),
                EpilogueAct::SiLU => silu_fwd_ps(x),
            }
        }
    }

    /// Horizontal max of 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn hmax_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let m128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehl_ps(m128, m128);
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps::<0x55>(m64, m64);
        let m32 = _mm_max_ss(m64, shuf2);
        _mm_cvtss_f32(m32)
    }

    /// Horizontal sum of 8 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehl_ps(s128, s128);
        let s64 = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_shuffle_ps::<0x55>(s64, s64);
        let s32 = _mm_add_ss(s64, shuf2);
        _mm_cvtss_f32(s32)
    }
}

// ============================================================================
// x86_64 AVX-512F Vector Primitives (16 x f32)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// Fast Cody-Waite + degree-6 Estrin `exp(x)` on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn exp_ps(x: __m512) -> __m512 {
        let xc = _mm512_max_ps(
            _mm512_set1_ps(EXP_CLAMP_LO),
            _mm512_min_ps(x, _mm512_set1_ps(EXP_CLAMP_HI)),
        );
        let t = _mm512_mul_ps(xc, _mm512_set1_ps(LOG2E));
        let kf = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(t);
        let ki = _mm512_cvtps_epi32(kf);

        // Cody-Waite range reduction
        let r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_HI), xc);
        let r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_LO), r);

        // Degree-6 Estrin polynomial
        let r2 = _mm512_mul_ps(r, r);
        let r4 = _mm512_mul_ps(r2, r2);
        let p01 = _mm512_add_ps(_mm512_set1_ps(1.0), r);
        let p23 = _mm512_fmadd_ps(_mm512_set1_ps(EXP_C3), r, _mm512_set1_ps(EXP_C2));
        let p45 = _mm512_fmadd_ps(_mm512_set1_ps(EXP_C5), r, _mm512_set1_ps(EXP_C4));
        let p03 = _mm512_fmadd_ps(p23, r2, p01);
        let p46 = _mm512_fmadd_ps(_mm512_set1_ps(EXP_C6), r2, p45);
        let poly = _mm512_fmadd_ps(p46, r4, p03);

        let exp_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(ki, _mm512_set1_epi32(127)));
        let scale = _mm512_castsi512_ps(exp_bits);
        _mm512_mul_ps(poly, scale)
    }

    /// Fast `sigmoid(x)` on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn sigmoid_ps(x: __m512) -> __m512 {
        let one = _mm512_set1_ps(1.0);
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        // SAFETY: AVX-512F verified by caller.
        let e = unsafe { exp_ps(neg_x) };
        _mm512_div_ps(one, _mm512_add_ps(one, e))
    }

    /// Fast `tanh(x)` on 16 `f32` lanes via `sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))`.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn tanh_ps(x: __m512) -> __m512 {
        let xi = _mm512_castps_si512(x);
        let sign_mask = _mm512_set1_epi32(i32::MIN);
        let abs_mask = _mm512_set1_epi32(i32::MAX);
        let sign = _mm512_and_epi32(xi, sign_mask);
        let abs_x = _mm512_castsi512_ps(_mm512_and_epi32(xi, abs_mask));
        let clamped = _mm512_min_ps(abs_x, _mm512_set1_ps(TANH_CLAMP));
        let neg2x = _mm512_mul_ps(clamped, _mm512_set1_ps(-2.0));
        // SAFETY: AVX-512F verified by caller.
        let e = unsafe { exp_ps(neg2x) };
        let one = _mm512_set1_ps(1.0);
        let y = _mm512_div_ps(_mm512_sub_ps(one, e), _mm512_add_ps(one, e));
        _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(y), sign))
    }

    /// Fast tanh-approximated `GELU(x)` on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn gelu_fwd_ps(x: __m512) -> __m512 {
        let x2 = _mm512_mul_ps(x, x);
        let kx2 = _mm512_mul_ps(_mm512_set1_ps(GELU_K), x2);
        let inner = _mm512_fmadd_ps(kx2, x, x);
        let u = _mm512_mul_ps(_mm512_set1_ps(GELU_SQRT_2_OVER_PI), inner);
        // SAFETY: AVX-512F verified by caller.
        let t = unsafe { tanh_ps(u) };
        let half_x = _mm512_mul_ps(_mm512_set1_ps(0.5), x);
        _mm512_mul_ps(half_x, _mm512_add_ps(_mm512_set1_ps(1.0), t))
    }

    /// Fast tanh-approximated GELU backward on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn gelu_bwd_ps(dy: __m512, x: __m512) -> __m512 {
        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        let c = _mm512_set1_ps(GELU_SQRT_2_OVER_PI);
        let x2 = _mm512_mul_ps(x, x);
        let inner = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(GELU_K), x2), x, x);
        let u = _mm512_mul_ps(c, inner);
        // SAFETY: AVX-512F verified by caller.
        let t = unsafe { tanh_ps(u) };
        let sech2 = _mm512_fnmadd_ps(t, t, one);
        let du = _mm512_mul_ps(c, _mm512_fmadd_ps(_mm512_set1_ps(GELU_3K), x2, one));
        let term1 = _mm512_mul_ps(half, _mm512_add_ps(one, t));
        let term2 = _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_mul_ps(sech2, du));
        _mm512_mul_ps(dy, _mm512_add_ps(term1, term2))
    }

    /// Fast `SiLU(x) = x * sigmoid(x)` on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn silu_fwd_ps(x: __m512) -> __m512 {
        // SAFETY: AVX-512F verified by caller.
        let s = unsafe { sigmoid_ps(x) };
        _mm512_mul_ps(x, s)
    }

    /// Fast SiLU backward on 16 `f32` lanes.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn silu_bwd_ps(dy: __m512, x: __m512) -> __m512 {
        let one = _mm512_set1_ps(1.0);
        // SAFETY: AVX-512F verified by caller.
        let s = unsafe { sigmoid_ps(x) };
        let one_minus_s = _mm512_sub_ps(one, s);
        let deriv = _mm512_mul_ps(s, _mm512_fmadd_ps(x, one_minus_s, one));
        _mm512_mul_ps(dy, deriv)
    }

    /// Apply `EpilogueAct` to a 16-lane `f32` vector register.
    ///
    /// # Safety
    /// Caller must ensure CPU supports `avx512f`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn apply_act_ps(x: __m512, act: EpilogueAct) -> __m512 {
        // SAFETY: AVX-512F verified by caller.
        unsafe {
            match act {
                EpilogueAct::None => x,
                EpilogueAct::ReLU => _mm512_max_ps(x, _mm512_setzero_ps()),
                EpilogueAct::Sigmoid => sigmoid_ps(x),
                EpilogueAct::Tanh => tanh_ps(x),
                EpilogueAct::GELU => gelu_fwd_ps(x),
                EpilogueAct::SiLU => silu_fwd_ps(x),
            }
        }
    }
}

// ============================================================================
// Single-Threaded Chunk Kernels (AVX-512F / AVX2+FMA / Scalar)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_inplace_avx512(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_ps();
    let n = x.len();
    let mut i = 0;
    let ptr = x.as_mut_ptr();
    // SAFETY: `i + 16 <= n`, so loads/stores stay within `x`.
    unsafe {
        while i + 16 <= n {
            let v = _mm512_loadu_ps(ptr.add(i));
            _mm512_storeu_ps(ptr.add(i), _mm512_max_ps(v, zero));
            i += 16;
        }
    }
    for v in &mut x[i..] {
        *v = v.max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_inplace_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    let ptr = x.as_mut_ptr();
    // SAFETY: `i + 8 <= n`, so loads/stores stay within `x`.
    unsafe {
        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(v, zero));
            i += 8;
        }
    }
    for v in &mut x[i..] {
        *v = v.max(0.0);
    }
}

fn relu_inplace_chunk(x: &mut [f32], isa: SimdIsa) {
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            unsafe { relu_inplace_avx512(x) };
            return;
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            unsafe { relu_inplace_avx2(x) };
            return;
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    for v in x {
        *v = v.max(0.0);
    }
}

macro_rules! impl_unary_into_kernels {
    ($avx512_fn:ident, $avx2_fn:ident, $chunk_fn:ident, $v512_op:path, $v256_op:path, $scalar_op:path) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        unsafe fn $avx512_fn(x: &[f32], out: &mut [f32]) {
            use std::arch::x86_64::*;
            let n = x.len().min(out.len());
            let xp = x.as_ptr();
            let op = out.as_mut_ptr();
            let mut i = 0;
            // SAFETY: `i + 16 <= n <= x.len(), out.len()`, and AVX-512F is active.
            unsafe {
                while i + 16 <= n {
                    let v = _mm512_loadu_ps(xp.add(i));
                    _mm512_storeu_ps(op.add(i), $v512_op(v));
                    i += 16;
                }
            }
            for idx in i..n {
                out[idx] = $scalar_op(x[idx]);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn $avx2_fn(x: &[f32], out: &mut [f32]) {
            use std::arch::x86_64::*;
            let n = x.len().min(out.len());
            let xp = x.as_ptr();
            let op = out.as_mut_ptr();
            let mut i = 0;
            // SAFETY: `i + 8 <= n <= x.len(), out.len()`, and AVX2+FMA are active.
            unsafe {
                while i + 8 <= n {
                    let v = _mm256_loadu_ps(xp.add(i));
                    _mm256_storeu_ps(op.add(i), $v256_op(v));
                    i += 8;
                }
            }
            for idx in i..n {
                out[idx] = $scalar_op(x[idx]);
            }
        }

        fn $chunk_fn(x: &[f32], out: &mut [f32], isa: SimdIsa) {
            #[cfg(target_arch = "x86_64")]
            match isa {
                SimdIsa::Avx512F => {
                    // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
                    unsafe { $avx512_fn(x, out) };
                    return;
                }
                SimdIsa::Avx2Fma => {
                    // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
                    unsafe { $avx2_fn(x, out) };
                    return;
                }
                SimdIsa::Scalar => {}
            }
            let _ = isa;
            let n = x.len().min(out.len());
            for idx in 0..n {
                out[idx] = $scalar_op(x[idx]);
            }
        }
    };
}

impl_unary_into_kernels!(
    sigmoid_into_avx512,
    sigmoid_into_avx2,
    sigmoid_into_chunk,
    avx512::sigmoid_ps,
    avx2::sigmoid_ps,
    sigmoid_f32_scalar
);
impl_unary_into_kernels!(
    tanh_into_avx512,
    tanh_into_avx2,
    tanh_into_chunk,
    avx512::tanh_ps,
    avx2::tanh_ps,
    tanh_f32_scalar
);
impl_unary_into_kernels!(
    gelu_fwd_into_avx512,
    gelu_fwd_into_avx2,
    gelu_fwd_into_chunk,
    avx512::gelu_fwd_ps,
    avx2::gelu_fwd_ps,
    gelu_fwd_f32_scalar
);
impl_unary_into_kernels!(
    silu_fwd_into_avx512,
    silu_fwd_into_avx2,
    silu_fwd_into_chunk,
    avx512::silu_fwd_ps,
    avx2::silu_fwd_ps,
    silu_fwd_f32_scalar
);

macro_rules! impl_binary_into_kernels {
    ($avx512_fn:ident, $avx2_fn:ident, $chunk_fn:ident, $v512_op:path, $v256_op:path, $scalar_op:path) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        unsafe fn $avx512_fn(a: &[f32], b: &[f32], out: &mut [f32]) {
            use std::arch::x86_64::*;
            let n = a.len().min(b.len()).min(out.len());
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let op = out.as_mut_ptr();
            let mut i = 0;
            // SAFETY: `i + 16 <= n <= a.len(), b.len(), out.len()`, and AVX-512F is active.
            unsafe {
                while i + 16 <= n {
                    let va = _mm512_loadu_ps(ap.add(i));
                    let vb = _mm512_loadu_ps(bp.add(i));
                    _mm512_storeu_ps(op.add(i), $v512_op(va, vb));
                    i += 16;
                }
            }
            for idx in i..n {
                out[idx] = $scalar_op(a[idx], b[idx]);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn $avx2_fn(a: &[f32], b: &[f32], out: &mut [f32]) {
            use std::arch::x86_64::*;
            let n = a.len().min(b.len()).min(out.len());
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let op = out.as_mut_ptr();
            let mut i = 0;
            // SAFETY: `i + 8 <= n <= a.len(), b.len(), out.len()`, and AVX2+FMA are active.
            unsafe {
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(ap.add(i));
                    let vb = _mm256_loadu_ps(bp.add(i));
                    _mm256_storeu_ps(op.add(i), $v256_op(va, vb));
                    i += 8;
                }
            }
            for idx in i..n {
                out[idx] = $scalar_op(a[idx], b[idx]);
            }
        }

        fn $chunk_fn(a: &[f32], b: &[f32], out: &mut [f32], isa: SimdIsa) {
            #[cfg(target_arch = "x86_64")]
            match isa {
                SimdIsa::Avx512F => {
                    // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
                    unsafe { $avx512_fn(a, b, out) };
                    return;
                }
                SimdIsa::Avx2Fma => {
                    // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
                    unsafe { $avx2_fn(a, b, out) };
                    return;
                }
                SimdIsa::Scalar => {}
            }
            let _ = isa;
            let n = a.len().min(b.len()).min(out.len());
            for idx in 0..n {
                out[idx] = $scalar_op(a[idx], b[idx]);
            }
        }
    };
}

impl_binary_into_kernels!(
    gelu_bwd_into_avx512,
    gelu_bwd_into_avx2,
    gelu_bwd_into_chunk,
    avx512::gelu_bwd_ps,
    avx2::gelu_bwd_ps,
    gelu_bwd_f32_scalar
);
impl_binary_into_kernels!(
    silu_bwd_into_avx512,
    silu_bwd_into_avx2,
    silu_bwd_into_chunk,
    avx512::silu_bwd_ps,
    avx2::silu_bwd_ps,
    silu_bwd_f32_scalar
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_bias_act_row_avx512(row: &mut [f32], bias: Option<&[f32]>, act: EpilogueAct) {
    use std::arch::x86_64::*;
    let n = row.len();
    let rp = row.as_mut_ptr();
    let mut i = 0;
    if let Some(b) = bias {
        let bp = b.as_ptr();
        // SAFETY: `i + 16 <= n <= row.len(), b.len()`, and AVX-512F is active.
        unsafe {
            while i + 16 <= n {
                let v = _mm512_add_ps(_mm512_loadu_ps(rp.add(i)), _mm512_loadu_ps(bp.add(i)));
                _mm512_storeu_ps(rp.add(i), avx512::apply_act_ps(v, act));
                i += 16;
            }
        }
        for idx in i..n {
            row[idx] = apply_act_scalar(row[idx] + b[idx], act);
        }
    } else {
        // SAFETY: `i + 16 <= n <= row.len()`, and AVX-512F is active.
        unsafe {
            while i + 16 <= n {
                let v = _mm512_loadu_ps(rp.add(i));
                _mm512_storeu_ps(rp.add(i), avx512::apply_act_ps(v, act));
                i += 16;
            }
        }
        for v in &mut row[i..] {
            *v = apply_act_scalar(*v, act);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn add_bias_act_row_avx2(row: &mut [f32], bias: Option<&[f32]>, act: EpilogueAct) {
    use std::arch::x86_64::*;
    let n = row.len();
    let rp = row.as_mut_ptr();
    let mut i = 0;
    if let Some(b) = bias {
        let bp = b.as_ptr();
        // SAFETY: `i + 8 <= n <= row.len(), b.len()`, and AVX2+FMA are active.
        unsafe {
            while i + 8 <= n {
                let v = _mm256_add_ps(_mm256_loadu_ps(rp.add(i)), _mm256_loadu_ps(bp.add(i)));
                _mm256_storeu_ps(rp.add(i), avx2::apply_act_ps(v, act));
                i += 8;
            }
        }
        for idx in i..n {
            row[idx] = apply_act_scalar(row[idx] + b[idx], act);
        }
    } else {
        // SAFETY: `i + 8 <= n <= row.len()`, and AVX2+FMA are active.
        unsafe {
            while i + 8 <= n {
                let v = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(rp.add(i), avx2::apply_act_ps(v, act));
                i += 8;
            }
        }
        for v in &mut row[i..] {
            *v = apply_act_scalar(*v, act);
        }
    }
}

#[inline]
pub(crate) fn add_bias_act_row_inplace(
    row: &mut [f32],
    bias: Option<&[f32]>,
    act: EpilogueAct,
    isa: SimdIsa,
) {
    if bias.is_none() && act == EpilogueAct::None {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            unsafe { add_bias_act_row_avx512(row, bias, act) };
            return;
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            unsafe { add_bias_act_row_avx2(row, bias, act) };
            return;
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    if let Some(b) = bias {
        for (dst, &bv) in row.iter_mut().zip(b.iter()) {
            *dst = apply_act_scalar(*dst + bv, act);
        }
    } else {
        for dst in row.iter_mut() {
            *dst = apply_act_scalar(*dst, act);
        }
    }
}

// ============================================================================
// Row Softmax & Exp-Sub-Max Helpers (reused by CrossEntropy)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn exp_sub_max_sum_avx512(src: &[f32], dst: &mut [f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    // SAFETY: `i + 16 <= n <= src.len(), dst.len()`, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(sp.add(i)));
            i += 16;
        }
    }
    let mut max_v = _mm512_reduce_max_ps(vmax);
    for &v in &src[i..n] {
        if v > max_v {
            max_v = v;
        }
    }

    let vmax_bcast = _mm512_set1_ps(max_v);
    let mut vsum = _mm512_setzero_ps();
    i = 0;
    // SAFETY: `i + 16 <= n <= src.len(), dst.len()`, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            let shifted = _mm512_sub_ps(_mm512_loadu_ps(sp.add(i)), vmax_bcast);
            let e = avx512::exp_ps(shifted);
            _mm512_storeu_ps(dp.add(i), e);
            vsum = _mm512_add_ps(vsum, e);
            i += 16;
        }
    }
    let mut sum = _mm512_reduce_add_ps(vsum);
    for idx in i..n {
        let e = exp_f32_scalar(src[idx] - max_v);
        dst[idx] = e;
        sum += e;
    }
    (max_v, sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn exp_sub_max_sum_avx2(src: &[f32], dst: &mut [f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    // SAFETY: `i + 8 <= n <= src.len(), dst.len()`, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(sp.add(i)));
            i += 8;
        }
    }
    // SAFETY: AVX2 is active.
    let mut max_v = unsafe { avx2::hmax_ps(vmax) };
    for &v in &src[i..n] {
        if v > max_v {
            max_v = v;
        }
    }

    let vmax_bcast = _mm256_set1_ps(max_v);
    let mut vsum = _mm256_setzero_ps();
    i = 0;
    // SAFETY: `i + 8 <= n <= src.len(), dst.len()`, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            let shifted = _mm256_sub_ps(_mm256_loadu_ps(sp.add(i)), vmax_bcast);
            let e = avx2::exp_ps(shifted);
            _mm256_storeu_ps(dp.add(i), e);
            vsum = _mm256_add_ps(vsum, e);
            i += 8;
        }
    }
    // SAFETY: AVX2 is active.
    let mut sum = unsafe { avx2::hsum_ps(vsum) };
    for idx in i..n {
        let e = exp_f32_scalar(src[idx] - max_v);
        dst[idx] = e;
        sum += e;
    }
    (max_v, sum)
}

/// Computes `max_v = max(src)`, writes `dst[j] = exp(src[j] - max_v)`, and returns `(max_v, sum(dst))`.
#[inline]
pub(crate) fn simd_exp_sub_max_sum(src: &[f32], dst: &mut [f32]) -> (f32, f32) {
    let n = src.len().min(dst.len());
    if n == 0 {
        return (f32::NEG_INFINITY, 0.0);
    }
    let isa = detect_simd_isa();
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            return unsafe { exp_sub_max_sum_avx512(&src[..n], &mut dst[..n]) };
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            return unsafe { exp_sub_max_sum_avx2(&src[..n], &mut dst[..n]) };
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &src[..n] {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for idx in 0..n {
        let e = exp_f32_scalar(src[idx] - max_v);
        dst[idx] = e;
        sum += e;
    }
    (max_v, sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
unsafe fn exp_sub_scalar_into_avx512(src: &[f32], shift: f32, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let vshift = _mm512_set1_ps(shift);
    let mut i = 0;
    // SAFETY: `i + 16 <= n <= src.len(), dst.len()`, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            let shifted = _mm512_sub_ps(_mm512_loadu_ps(sp.add(i)), vshift);
            _mm512_storeu_ps(dp.add(i), avx512::exp_ps(shifted));
            i += 16;
        }
    }
    for idx in i..n {
        dst[idx] = exp_f32_scalar(src[idx] - shift);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(dead_code)]
unsafe fn exp_sub_scalar_into_avx2(src: &[f32], shift: f32, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let vshift = _mm256_set1_ps(shift);
    let mut i = 0;
    // SAFETY: `i + 8 <= n <= src.len(), dst.len()`, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            let shifted = _mm256_sub_ps(_mm256_loadu_ps(sp.add(i)), vshift);
            _mm256_storeu_ps(dp.add(i), avx2::exp_ps(shifted));
            i += 8;
        }
    }
    for idx in i..n {
        dst[idx] = exp_f32_scalar(src[idx] - shift);
    }
}

#[allow(dead_code)]
fn exp_sub_scalar_into_chunk(src: &[f32], shift: f32, dst: &mut [f32], isa: SimdIsa) {
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            unsafe { exp_sub_scalar_into_avx512(src, shift, dst) };
            return;
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            unsafe { exp_sub_scalar_into_avx2(src, shift, dst) };
            return;
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    let n = src.len().min(dst.len());
    for idx in 0..n {
        dst[idx] = exp_f32_scalar(src[idx] - shift);
    }
}

/// Vectorized `dst[i] = exp(src[i] - shift)` (AVX-512F / AVX2+FMA / scalar fallback).
///
/// Handles `f32::NEG_INFINITY` entries cleanly (`exp(-inf - shift) -> 0.0`) and
/// safely writes `0.0` when `!shift.is_finite()`.
#[allow(dead_code)]
pub(crate) fn simd_exp_sub_scalar_into(src: &[f32], shift: f32, dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    if n == 0 {
        return;
    }
    if !shift.is_finite() {
        dst[..n].fill(0.0);
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        src[..n]
            .par_chunks(chunk)
            .zip(dst[..n].par_chunks_mut(chunk))
            .for_each(|(sc, dc)| exp_sub_scalar_into_chunk(sc, shift, dc, isa));
    } else {
        exp_sub_scalar_into_chunk(&src[..n], shift, &mut dst[..n], isa);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_row_avx512(row: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = row.len();
    let ptr = row.as_mut_ptr();

    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    // SAFETY: `i + 16 <= n <= row.len()`, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(ptr.add(i)));
            i += 16;
        }
    }
    let mut max_v = _mm512_reduce_max_ps(vmax);
    for &v in &row[i..] {
        if v > max_v {
            max_v = v;
        }
    }

    let vmax_bcast = _mm512_set1_ps(max_v);
    let mut vsum = _mm512_setzero_ps();
    i = 0;
    // SAFETY: `i + 16 <= n <= row.len()`, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            let shifted = _mm512_sub_ps(_mm512_loadu_ps(ptr.add(i)), vmax_bcast);
            let e = avx512::exp_ps(shifted);
            _mm512_storeu_ps(ptr.add(i), e);
            vsum = _mm512_add_ps(vsum, e);
            i += 16;
        }
    }
    let mut sum = _mm512_reduce_add_ps(vsum);
    for v in &mut row[i..] {
        let e = exp_f32_scalar(*v - max_v);
        *v = e;
        sum += e;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        let vinv = _mm512_set1_ps(inv_sum);
        i = 0;
        // SAFETY: `i + 16 <= n <= row.len()`, and AVX-512F is active.
        unsafe {
            while i + 16 <= n {
                let v = _mm512_loadu_ps(ptr.add(i));
                _mm512_storeu_ps(ptr.add(i), _mm512_mul_ps(v, vinv));
                i += 16;
            }
        }
        for v in &mut row[i..] {
            *v *= inv_sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_row_avx2(row: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = row.len();
    let ptr = row.as_mut_ptr();

    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    // SAFETY: `i + 8 <= n <= row.len()`, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(ptr.add(i)));
            i += 8;
        }
    }
    // SAFETY: AVX2 is active.
    let mut max_v = unsafe { avx2::hmax_ps(vmax) };
    for &v in &row[i..] {
        if v > max_v {
            max_v = v;
        }
    }

    let vmax_bcast = _mm256_set1_ps(max_v);
    let mut vsum = _mm256_setzero_ps();
    i = 0;
    // SAFETY: `i + 8 <= n <= row.len()`, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            let shifted = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(i)), vmax_bcast);
            let e = avx2::exp_ps(shifted);
            _mm256_storeu_ps(ptr.add(i), e);
            vsum = _mm256_add_ps(vsum, e);
            i += 8;
        }
    }
    // SAFETY: AVX2 is active.
    let mut sum = unsafe { avx2::hsum_ps(vsum) };
    for v in &mut row[i..] {
        let e = exp_f32_scalar(*v - max_v);
        *v = e;
        sum += e;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv_sum);
        i = 0;
        // SAFETY: `i + 8 <= n <= row.len()`, and AVX2+FMA are active.
        unsafe {
            while i + 8 <= n {
                let v = _mm256_loadu_ps(ptr.add(i));
                _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, vinv));
                i += 8;
            }
        }
        for v in &mut row[i..] {
            *v *= inv_sum;
        }
    }
}

fn softmax_row_inplace(row: &mut [f32], isa: SimdIsa) {
    if row.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            unsafe { softmax_row_avx512(row) };
            return;
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            unsafe { softmax_row_avx2(row) };
            return;
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    let mut max_v = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        let e = exp_f32_scalar(*v - max_v);
        *v = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ============================================================================
// Pure-f32 SIMD AdamW Update Kernels
// ============================================================================

#[derive(Clone, Copy)]
struct AdamWParams {
    beta1: f32,
    omb1: f32,
    beta2: f32,
    omb2: f32,
    decay: f32,
    step_size: f32,
    inv_bc2: f32,
    eps: f32,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn adamw_chunk_avx512(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    p: AdamWParams,
) {
    use std::arch::x86_64::*;
    let n = param.len().min(grad.len()).min(m.len()).min(v.len());
    let pp = param.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = m.as_mut_ptr();
    let vp = v.as_mut_ptr();

    let vb1 = _mm512_set1_ps(p.beta1);
    let vomb1 = _mm512_set1_ps(p.omb1);
    let vb2 = _mm512_set1_ps(p.beta2);
    let vomb2 = _mm512_set1_ps(p.omb2);
    let vdecay = _mm512_set1_ps(p.decay);
    let vstep = _mm512_set1_ps(p.step_size);
    let vinv_bc2 = _mm512_set1_ps(p.inv_bc2);
    let veps = _mm512_set1_ps(p.eps);

    let mut i = 0;
    // SAFETY: `i + 16 <= n` across all four slices, and AVX-512F is active.
    unsafe {
        while i + 16 <= n {
            let gv = _mm512_loadu_ps(gp.add(i));
            let mv = _mm512_loadu_ps(mp.add(i));
            let vv = _mm512_loadu_ps(vp.add(i));
            let pv = _mm512_loadu_ps(pp.add(i));

            let m_new = _mm512_fmadd_ps(vb1, mv, _mm512_mul_ps(vomb1, gv));
            let g2 = _mm512_mul_ps(gv, gv);
            let v_new = _mm512_fmadd_ps(vb2, vv, _mm512_mul_ps(vomb2, g2));

            _mm512_storeu_ps(mp.add(i), m_new);
            _mm512_storeu_ps(vp.add(i), v_new);

            let v_hat = _mm512_mul_ps(v_new, vinv_bc2);
            let denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), veps);
            let update = _mm512_div_ps(_mm512_mul_ps(vstep, m_new), denom);
            let p_new = _mm512_fmsub_ps(pv, vdecay, update);
            _mm512_storeu_ps(pp.add(i), p_new);

            i += 16;
        }
    }
    adamw_chunk_scalar(&mut param[i..n], &grad[i..n], &mut m[i..n], &mut v[i..n], p);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn adamw_chunk_avx2(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    p: AdamWParams,
) {
    use std::arch::x86_64::*;
    let n = param.len().min(grad.len()).min(m.len()).min(v.len());
    let pp = param.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = m.as_mut_ptr();
    let vp = v.as_mut_ptr();

    let vb1 = _mm256_set1_ps(p.beta1);
    let vomb1 = _mm256_set1_ps(p.omb1);
    let vb2 = _mm256_set1_ps(p.beta2);
    let vomb2 = _mm256_set1_ps(p.omb2);
    let vdecay = _mm256_set1_ps(p.decay);
    let vstep = _mm256_set1_ps(p.step_size);
    let vinv_bc2 = _mm256_set1_ps(p.inv_bc2);
    let veps = _mm256_set1_ps(p.eps);

    let mut i = 0;
    // SAFETY: `i + 8 <= n` across all four slices, and AVX2+FMA are active.
    unsafe {
        while i + 8 <= n {
            let gv = _mm256_loadu_ps(gp.add(i));
            let mv = _mm256_loadu_ps(mp.add(i));
            let vv = _mm256_loadu_ps(vp.add(i));
            let pv = _mm256_loadu_ps(pp.add(i));

            let m_new = _mm256_fmadd_ps(vb1, mv, _mm256_mul_ps(vomb1, gv));
            let g2 = _mm256_mul_ps(gv, gv);
            let v_new = _mm256_fmadd_ps(vb2, vv, _mm256_mul_ps(vomb2, g2));

            _mm256_storeu_ps(mp.add(i), m_new);
            _mm256_storeu_ps(vp.add(i), v_new);

            let v_hat = _mm256_mul_ps(v_new, vinv_bc2);
            let denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), veps);
            let update = _mm256_div_ps(_mm256_mul_ps(vstep, m_new), denom);
            let p_new = _mm256_fmsub_ps(pv, vdecay, update);
            _mm256_storeu_ps(pp.add(i), p_new);

            i += 8;
        }
    }
    adamw_chunk_scalar(&mut param[i..n], &grad[i..n], &mut m[i..n], &mut v[i..n], p);
}

#[inline]
fn adamw_chunk_scalar(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    p: AdamWParams,
) {
    let n = param.len().min(grad.len()).min(m.len()).min(v.len());
    for i in 0..n {
        let g = grad[i];
        let m_new = p.beta1.mul_add(m[i], p.omb1 * g);
        let v_new = p.beta2.mul_add(v[i], p.omb2 * g * g);
        m[i] = m_new;
        v[i] = v_new;
        let denom = (v_new * p.inv_bc2).sqrt() + p.eps;
        param[i] = param[i].mul_add(p.decay, -(p.step_size * m_new / denom));
    }
}

fn adamw_chunk(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    p: AdamWParams,
    isa: SimdIsa,
) {
    #[cfg(target_arch = "x86_64")]
    match isa {
        SimdIsa::Avx512F => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX-512F support.
            unsafe { adamw_chunk_avx512(param, grad, m, v, p) };
            return;
        }
        SimdIsa::Avx2Fma => {
            // SAFETY: `detect_simd_isa()` verified runtime AVX2+FMA support.
            unsafe { adamw_chunk_avx2(param, grad, m, v, p) };
            return;
        }
        SimdIsa::Scalar => {}
    }
    let _ = isa;
    adamw_chunk_scalar(param, grad, m, v, p);
}

// ============================================================================
// Public Slice APIs (Auto-Parallelized via Rayon Only When len >= 131_072)
// ============================================================================

/// Vectorized in-place ReLU (`x[i] = max(x[i], 0.0)`).
pub fn simd_relu_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let isa = detect_simd_isa();
    if x.len() >= SIMD_RAYON_THRESHOLD {
        let chunk = (x.len() / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        x.par_chunks_mut(chunk)
            .for_each(|c| relu_inplace_chunk(c, isa));
    } else {
        relu_inplace_chunk(x, isa);
    }
}

/// Vectorized Sigmoid (`out[i] = 1 / (1 + exp(-x[i]))`).
pub fn simd_sigmoid_into(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        x[..n]
            .par_chunks(chunk)
            .zip(out[..n].par_chunks_mut(chunk))
            .for_each(|(xc, oc)| sigmoid_into_chunk(xc, oc, isa));
    } else {
        sigmoid_into_chunk(&x[..n], &mut out[..n], isa);
    }
}

/// Vectorized Tanh (`out[i] = tanh(x[i])`).
pub fn simd_tanh_into(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        x[..n]
            .par_chunks(chunk)
            .zip(out[..n].par_chunks_mut(chunk))
            .for_each(|(xc, oc)| tanh_into_chunk(xc, oc, isa));
    } else {
        tanh_into_chunk(&x[..n], &mut out[..n], isa);
    }
}

/// Vectorized tanh-approximated GELU forward.
pub fn simd_gelu_fwd_into(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        x[..n]
            .par_chunks(chunk)
            .zip(out[..n].par_chunks_mut(chunk))
            .for_each(|(xc, oc)| gelu_fwd_into_chunk(xc, oc, isa));
    } else {
        gelu_fwd_into_chunk(&x[..n], &mut out[..n], isa);
    }
}

/// Vectorized tanh-approximated GELU backward.
pub fn simd_gelu_bwd_into(dy: &[f32], x: &[f32], dx: &mut [f32]) {
    let n = dy.len().min(x.len()).min(dx.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        dy[..n]
            .par_chunks(chunk)
            .zip(x[..n].par_chunks(chunk))
            .zip(dx[..n].par_chunks_mut(chunk))
            .for_each(|((dyc, xc), dxc)| gelu_bwd_into_chunk(dyc, xc, dxc, isa));
    } else {
        gelu_bwd_into_chunk(&dy[..n], &x[..n], &mut dx[..n], isa);
    }
}

/// Vectorized SiLU / Swish forward (`out[i] = x[i] * sigmoid(x[i])`).
pub fn simd_silu_fwd_into(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        x[..n]
            .par_chunks(chunk)
            .zip(out[..n].par_chunks_mut(chunk))
            .for_each(|(xc, oc)| silu_fwd_into_chunk(xc, oc, isa));
    } else {
        silu_fwd_into_chunk(&x[..n], &mut out[..n], isa);
    }
}

/// Vectorized SiLU / Swish backward.
pub fn simd_silu_bwd_into(dy: &[f32], x: &[f32], dx: &mut [f32]) {
    let n = dy.len().min(x.len()).min(dx.len());
    if n == 0 {
        return;
    }
    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        dy[..n]
            .par_chunks(chunk)
            .zip(x[..n].par_chunks(chunk))
            .zip(dx[..n].par_chunks_mut(chunk))
            .for_each(|((dyc, xc), dxc)| silu_bwd_into_chunk(dyc, xc, dxc, isa));
    } else {
        silu_bwd_into_chunk(&dy[..n], &x[..n], &mut dx[..n], isa);
    }
}

/// Vectorized in-place row-wise bias addition + activation epilogue on a `[rows, cols]` matrix.
pub fn simd_add_bias_act_inplace(
    out: &mut [f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
    act: EpilogueAct,
) {
    let total = rows.saturating_mul(cols);
    if total == 0 || out.len() < total {
        return;
    }
    let bias = bias.filter(|b| b.len() >= cols).map(|b| &b[..cols]);
    if bias.is_none() && act == EpilogueAct::None {
        return;
    }

    let isa = detect_simd_isa();
    if bias.is_none() {
        let slice = &mut out[..total];
        if total >= SIMD_RAYON_THRESHOLD {
            let chunk = (total / rayon::current_num_threads().max(1))
                .max(16_384)
                .div_ceil(16)
                * 16;
            slice
                .par_chunks_mut(chunk)
                .for_each(|c| add_bias_act_row_inplace(c, None, act, isa));
        } else {
            add_bias_act_row_inplace(slice, None, act, isa);
        }
        return;
    }

    if total >= SIMD_RAYON_THRESHOLD && rows >= 2 {
        out[..total]
            .par_chunks_mut(cols)
            .for_each(|row| add_bias_act_row_inplace(row, bias, act, isa));
    } else {
        for row in out[..total].chunks_exact_mut(cols) {
            add_bias_act_row_inplace(row, bias, act, isa);
        }
    }
}

/// Vectorized in-place row-wise Softmax on a `[rows, cols]` row-major matrix.
pub fn simd_softmax_rows_inplace(x: &mut [f32], rows: usize, cols: usize) {
    let total = rows.saturating_mul(cols);
    if total == 0 || x.len() < total {
        return;
    }
    let isa = detect_simd_isa();
    if total >= SIMD_RAYON_THRESHOLD && rows >= 2 {
        x[..total]
            .par_chunks_mut(cols)
            .for_each(|row| softmax_row_inplace(row, isa));
    } else {
        for row in x[..total].chunks_exact_mut(cols) {
            softmax_row_inplace(row, isa);
        }
    }
}

/// Pure-`f32` SIMD AdamW optimizer step (AVX-512F / AVX2+FMA / scalar fallback).
#[allow(clippy::too_many_arguments)]
pub fn simd_adamw_step_f32(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
) {
    let n = param.len().min(grad.len()).min(m.len()).min(v.len());
    if n == 0 || step == 0 {
        return;
    }

    let b1 = beta1 as f64;
    let b2 = beta2 as f64;
    let step_i32 = step.min(i32::MAX as usize) as i32;
    let inv_bc1 = 1.0 / (1.0 - b1.powi(step_i32));
    let inv_bc2 = 1.0 / (1.0 - b2.powi(step_i32));

    let params = AdamWParams {
        beta1,
        omb1: (1.0 - b1) as f32,
        beta2,
        omb2: (1.0 - b2) as f32,
        decay: (1.0 - (lr as f64) * (weight_decay as f64)) as f32,
        step_size: ((lr as f64) * inv_bc1) as f32,
        inv_bc2: inv_bc2 as f32,
        eps,
    };

    let isa = detect_simd_isa();
    if n >= SIMD_RAYON_THRESHOLD {
        let chunk = (n / rayon::current_num_threads().max(1))
            .max(16_384)
            .div_ceil(16)
            * 16;
        param[..n]
            .par_chunks_mut(chunk)
            .zip(grad[..n].par_chunks(chunk))
            .zip(m[..n].par_chunks_mut(chunk))
            .zip(v[..n].par_chunks_mut(chunk))
            .for_each(|(((pc, gc), mc), vc)| adamw_chunk(pc, gc, mc, vc, params, isa));
    } else {
        adamw_chunk(
            &mut param[..n],
            &grad[..n],
            &mut m[..n],
            &mut v[..n],
            params,
            isa,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_sigmoid_f64(x: f32) -> f32 {
        let xd = x as f64;
        if xd >= 0.0 {
            (1.0 / (1.0 + (-xd).exp())) as f32
        } else {
            let e = xd.exp();
            (e / (1.0 + e)) as f32
        }
    }

    fn ref_tanh_f64(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    fn ref_gelu_fwd_f64(x: f32) -> f32 {
        let xd = x as f64;
        let c = 0.797_884_560_802_865_4_f64;
        let k = 0.044_715_f64;
        let u = c * (xd + k * xd * xd * xd);
        (0.5 * xd * (1.0 + u.tanh())) as f32
    }

    fn ref_gelu_bwd_f64(dy: f32, x: f32) -> f32 {
        let xd = x as f64;
        let c = 0.797_884_560_802_865_4_f64;
        let k = 0.044_715_f64;
        let x2 = xd * xd;
        let u = c * (xd + k * x2 * xd);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let du = c * (1.0 + 3.0 * k * x2);
        let deriv = 0.5 * (1.0 + t) + 0.5 * xd * sech2 * du;
        ((dy as f64) * deriv) as f32
    }

    fn ref_silu_fwd_f64(x: f32) -> f32 {
        let xd = x as f64;
        let s = ref_sigmoid_f64(x) as f64;
        (xd * s) as f32
    }

    fn ref_silu_bwd_f64(dy: f32, x: f32) -> f32 {
        let xd = x as f64;
        let s = ref_sigmoid_f64(x) as f64;
        let deriv = s * (1.0 + xd * (1.0 - s));
        ((dy as f64) * deriv) as f32
    }

    #[test]
    fn test_simd_transcendentals_parity_and_edges() {
        // Non-multiple-of-16 length covering [-20.0, 20.0] plus extreme edges.
        let mut x: Vec<f32> = (-2000..=2000).map(|i| (i as f32) * 0.01).collect();
        x.extend_from_slice(&[-88.0, -50.0, -9.5, -1e-6, 0.0, 1e-6, 9.5, 50.0, 88.0]);
        let n = x.len();
        assert_ne!(n % 16, 0);
        let dy: Vec<f32> = (0..n)
            .map(|i| (((i * 17) % 31) as f32 - 15.0) * 0.1)
            .collect();

        let mut out_sig = vec![0.0f32; n];
        let mut out_tanh = vec![0.0f32; n];
        let mut out_gelu = vec![0.0f32; n];
        let mut out_gelu_bwd = vec![0.0f32; n];
        let mut out_silu = vec![0.0f32; n];
        let mut out_silu_bwd = vec![0.0f32; n];

        simd_sigmoid_into(&x, &mut out_sig);
        simd_tanh_into(&x, &mut out_tanh);
        simd_gelu_fwd_into(&x, &mut out_gelu);
        simd_gelu_bwd_into(&dy, &x, &mut out_gelu_bwd);
        simd_silu_fwd_into(&x, &mut out_silu);
        simd_silu_bwd_into(&dy, &x, &mut out_silu_bwd);

        for i in 0..n {
            let xv = x[i];
            let err_sig = (out_sig[i] - ref_sigmoid_f64(xv)).abs();
            assert!(
                err_sig < 2e-5,
                "sigmoid err {err_sig} at x={xv}: got {} vs {}",
                out_sig[i],
                ref_sigmoid_f64(xv)
            );
            let err_tanh = (out_tanh[i] - ref_tanh_f64(xv)).abs();
            assert!(
                err_tanh < 2e-5,
                "tanh err {err_tanh} at x={xv}: got {} vs {}",
                out_tanh[i],
                ref_tanh_f64(xv)
            );
            let err_gelu = (out_gelu[i] - ref_gelu_fwd_f64(xv)).abs();
            assert!(
                err_gelu < 2e-5,
                "gelu_fwd err {err_gelu} at x={xv}: got {} vs {}",
                out_gelu[i],
                ref_gelu_fwd_f64(xv)
            );
            let err_gelu_bwd = (out_gelu_bwd[i] - ref_gelu_bwd_f64(dy[i], xv)).abs();
            assert!(
                err_gelu_bwd < 2e-5,
                "gelu_bwd err {err_gelu_bwd} at x={xv}: got {} vs {}",
                out_gelu_bwd[i],
                ref_gelu_bwd_f64(dy[i], xv)
            );
            let err_silu = (out_silu[i] - ref_silu_fwd_f64(xv)).abs();
            assert!(
                err_silu < 2e-5,
                "silu_fwd err {err_silu} at x={xv}: got {} vs {}",
                out_silu[i],
                ref_silu_fwd_f64(xv)
            );
            let err_silu_bwd = (out_silu_bwd[i] - ref_silu_bwd_f64(dy[i], xv)).abs();
            assert!(
                err_silu_bwd < 2e-5,
                "silu_bwd err {err_silu_bwd} at x={xv}: got {} vs {}",
                out_silu_bwd[i],
                ref_silu_bwd_f64(dy[i], xv)
            );
        }

        // Also check scalar fallback directly.
        for &xv in &[-20.0f32, -3.5, -0.1, 0.0, 0.1, 3.5, 20.0] {
            assert!((sigmoid_f32_scalar(xv) - ref_sigmoid_f64(xv)).abs() < 2e-5);
            assert!((tanh_f32_scalar(xv) - ref_tanh_f64(xv)).abs() < 2e-5);
            assert!((gelu_fwd_f32_scalar(xv) - ref_gelu_fwd_f64(xv)).abs() < 2e-5);
            assert!((silu_fwd_f32_scalar(xv) - ref_silu_fwd_f64(xv)).abs() < 2e-5);
        }
    }

    #[test]
    fn test_simd_relu_bias_act_and_softmax() {
        let mut r = vec![-3.0, -0.0, 0.0, 2.5, -1.0, 4.0, -0.1, 0.2, 7.0, -8.0, 1.5];
        simd_relu_inplace(&mut r);
        assert_eq!(
            r,
            vec![0.0, 0.0, 0.0, 2.5, 0.0, 4.0, 0.0, 0.2, 7.0, 0.0, 1.5]
        );

        let rows = 5;
        let cols = 19; // non-multiple-of-16
        let base: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.25)
            .collect();
        let bias: Vec<f32> = (0..cols).map(|j| (j as f32 - 9.0) * 0.1).collect();

        for act in [
            EpilogueAct::None,
            EpilogueAct::ReLU,
            EpilogueAct::Sigmoid,
            EpilogueAct::Tanh,
            EpilogueAct::GELU,
            EpilogueAct::SiLU,
        ] {
            let mut got = base.clone();
            simd_add_bias_act_inplace(&mut got, Some(&bias), rows, cols, act);
            for ri in 0..rows {
                for ci in 0..cols {
                    let v = base[ri * cols + ci] + bias[ci];
                    let expected = match act {
                        EpilogueAct::None => v,
                        EpilogueAct::ReLU => v.max(0.0),
                        EpilogueAct::Sigmoid => ref_sigmoid_f64(v),
                        EpilogueAct::Tanh => ref_tanh_f64(v),
                        EpilogueAct::GELU => ref_gelu_fwd_f64(v),
                        EpilogueAct::SiLU => ref_silu_fwd_f64(v),
                    };
                    assert!(
                        (got[ri * cols + ci] - expected).abs() < 2e-5,
                        "bias_act({act:?}) mismatch at ({ri},{ci})"
                    );
                }
            }
        }

        // Row softmax parity
        let mut sm = base.clone();
        simd_softmax_rows_inplace(&mut sm, rows, cols);
        for ri in 0..rows {
            let src_row = &base[ri * cols..(ri + 1) * cols];
            let max_v = src_row
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
            let exps: Vec<f64> = src_row
                .iter()
                .map(|&v| ((v as f64) - max_v).exp())
                .collect();
            let sum: f64 = exps.iter().sum();
            let mut got_sum = 0.0f32;
            for ci in 0..cols {
                let expected = (exps[ci] / sum) as f32;
                let g = sm[ri * cols + ci];
                got_sum += g;
                assert!(
                    (g - expected).abs() < 2e-5,
                    "softmax mismatch at ({ri},{ci}): {g} vs {expected}"
                );
            }
            assert!((got_sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_adamw_parity_and_convergence() {
        let n = 37; // non-multiple-of-16
        let mut p_simd: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.25).collect();
        let mut p_ref = p_simd.clone();
        let mut m_simd = vec![0.0f32; n];
        let mut v_simd = vec![0.0f32; n];
        let mut m_ref = vec![0.0f64; n];
        let mut v_ref = vec![0.0f64; n];

        let lr = 0.05f32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let wd = 0.01f32;

        // Minimize quadratic 0.5 * sum((p - 1.0)^2) for 50 steps
        for step in 1..=50 {
            let grad_simd: Vec<f32> = p_simd.iter().map(|&x| x - 1.0).collect();
            let grad_ref: Vec<f64> = p_ref.iter().map(|&x| (x as f64) - 1.0).collect();

            simd_adamw_step_f32(
                &mut p_simd,
                &grad_simd,
                &mut m_simd,
                &mut v_simd,
                lr,
                b1,
                b2,
                eps,
                wd,
                step,
            );

            let inv_bc1 = 1.0 / (1.0 - (b1 as f64).powi(step as i32));
            let inv_bc2 = 1.0 / (1.0 - (b2 as f64).powi(step as i32));
            let decay = 1.0 - (lr as f64) * (wd as f64);
            for i in 0..n {
                m_ref[i] = (b1 as f64) * m_ref[i] + (1.0 - b1 as f64) * grad_ref[i];
                v_ref[i] = (b2 as f64) * v_ref[i] + (1.0 - b2 as f64) * grad_ref[i] * grad_ref[i];
                let m_hat = m_ref[i] * inv_bc1;
                let v_hat = v_ref[i] * inv_bc2;
                p_ref[i] = ((p_ref[i] as f64) * decay
                    - (lr as f64) * m_hat / (v_hat.sqrt() + eps as f64))
                    as f32;
            }
        }

        for i in 0..n {
            assert!(
                (p_simd[i] - p_ref[i]).abs() < 2e-5,
                "adamw mismatch at idx={i}: {} vs {}",
                p_simd[i],
                p_ref[i]
            );
            // Must converge significantly closer to 1.0 than initial values.
            let init_dist = ((i as f32 - 18.0) * 0.25 - 1.0).abs();
            let final_dist = (p_simd[i] - 1.0).abs();
            assert!(final_dist < init_dist || final_dist < 0.05);
        }
    }

    #[test]
    fn test_simd_parallel_threshold_paths() {
        // Exercise Rayon parallel branches (>= 131_072 elements).
        let n = SIMD_RAYON_THRESHOLD + 19;
        let x: Vec<f32> = (0..n).map(|i| ((i % 401) as f32 - 200.0) * 0.05).collect();
        let mut out = vec![0.0f32; n];
        simd_gelu_fwd_into(&x, &mut out);
        for &idx in &[0, 17, 65_537, n - 1] {
            assert!((out[idx] - ref_gelu_fwd_f64(x[idx])).abs() < 2e-5);
        }
    }

    #[test]
    fn test_simd_exp_sub_scalar_into() {
        // Non-multiple-of-16 length with finite values, extreme negatives, and NEG_INFINITY.
        let mut src: Vec<f32> = (-25..=25).map(|i| (i as f32) * 0.3).collect();
        src[0] = f32::NEG_INFINITY;
        src[7] = f32::NEG_INFINITY;
        src[16] = f32::NEG_INFINITY;
        let last = src.len() - 1;
        src[last] = f32::NEG_INFINITY;
        src.push(-120.0);
        src.push(2.5);
        let n = src.len();
        assert_ne!(n % 16, 0);

        let shift = 1.75f32;
        let mut dst = vec![f32::NAN; n];
        let mut dst_scalar = vec![f32::NAN; n];
        simd_exp_sub_scalar_into(&src, shift, &mut dst);
        exp_sub_scalar_into_chunk(&src, shift, &mut dst_scalar, SimdIsa::Scalar);

        for i in 0..n {
            if src[i] == f32::NEG_INFINITY || src[i] - shift <= EXP_CLAMP_LO {
                assert_eq!(dst[i], 0.0, "expected 0.0 at idx={i} for src={}", src[i]);
                assert_eq!(dst_scalar[i], 0.0);
            } else {
                let expected = ((src[i] - shift) as f64).exp() as f32;
                let rel_err = (dst[i] - expected).abs() / expected.max(1e-6);
                assert!(
                    rel_err < 2e-5,
                    "exp_sub_scalar mismatch at idx={i}: got {} vs {expected}",
                    dst[i]
                );
                assert!((dst[i] - dst_scalar[i]).abs() <= 1e-5 * expected.max(1.0));
            }
        }

        // Non-finite shifts must safely produce 0.0 without NaNs.
        for &bad_shift in &[f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let mut out_bad = vec![1.0f32; n];
            simd_exp_sub_scalar_into(&src, bad_shift, &mut out_bad);
            assert!(out_bad.iter().all(|&v| v == 0.0));
        }

        // Exercise parallel path above SIMD_RAYON_THRESHOLD.
        let big_n = SIMD_RAYON_THRESHOLD + 23;
        let mut big_src: Vec<f32> = (0..big_n)
            .map(|i| ((i % 41) as f32 - 20.0) * 0.25)
            .collect();
        big_src[13] = f32::NEG_INFINITY;
        big_src[big_n - 1] = f32::NEG_INFINITY;
        let mut big_dst = vec![0.0f32; big_n];
        simd_exp_sub_scalar_into(&big_src, 0.5, &mut big_dst);
        assert_eq!(big_dst[13], 0.0);
        assert_eq!(big_dst[big_n - 1], 0.0);
        assert!((big_dst[100] - ((big_src[100] - 0.5) as f64).exp() as f32).abs() < 2e-4);
    }
}
