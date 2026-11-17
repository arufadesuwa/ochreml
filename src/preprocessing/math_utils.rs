// OchreML Preprocessing Mathematical Utilities (*≧ω≦*)
// High-precision normal CDF/PPF (erfc + Acklam + Halley), Brent 1D optimizer,
// linear/quantile estimators, and NaN-aware column statistics.

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Match `sklearn.preprocessing._data._handle_zeros_in_scale` (`scale < 10 * eps`).
#[inline]
pub fn handle_zeros_in_scale(scale: f64) -> f64 {
    if scale == 0.0 || !scale.is_finite() || scale < 10.0 * f64::EPSILON {
        1.0
    } else {
        scale
    }
}

/// Fast check whether any value in a slice is NaN.
#[inline]
pub fn slice_has_nan(data: &[f64]) -> bool {
    if data.len() < 16_384 {
        data.iter().any(|v| v.is_nan())
    } else {
        data.par_chunks(4096)
            .any(|chunk| chunk.iter().any(|v| v.is_nan()))
    }
}

/// Cody's rational Chebyshev approximation for `erfc(x)` to full IEEE-754 `f64` precision (~1e-16).
pub fn erfc_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= 0.46875 {
        const A: [f64; 5] = [
            3.161_123_743_870_565_5,
            1.138_641_541_510_501_6e2,
            3.774_852_376_853_02e2,
            3.209_377_589_138_469_4e3,
            1.857_777_061_846_031_6e-1,
        ];
        const B: [f64; 4] = [
            2.360_129_095_234_412_2e1,
            2.440_246_379_344_441_7e2,
            1.282_616_526_077_372_3e3,
            2.844_236_833_439_171e3,
        ];
        let ysq = y * y;
        let xnum = (((A[4] * ysq + A[0]) * ysq + A[1]) * ysq + A[2]) * ysq + A[3];
        let xden = (((ysq + B[0]) * ysq + B[1]) * ysq + B[2]) * ysq + B[3];
        let erf_val = x * (xnum / xden);
        return 1.0 - erf_val;
    }

    let res = if y <= 4.0 {
        #[allow(clippy::excessive_precision)]
        const C: [f64; 9] = [
            5.641_884_969_886_701e-1,
            8.883_149_794_388_376,
            6.611_919_063_714_163e1,
            2.986_351_381_974_001_3e2,
            8.819_522_212_417_691e2,
            1.712_047_612_634_070_7e3,
            2.051_078_377_826_071_6e3,
            1.230_339_354_797_997_2e3,
            2.153_115_354_744_038_3e-8,
        ];
        const D: [f64; 8] = [
            1.574_492_611_070_983_5e1,
            1.176_939_508_913_125e2,
            5.371_811_018_620_099e2,
            1.621_389_574_566_690_3e3,
            3.290_799_235_733_459_7e3,
            4.362_619_090_143_247e3,
            3.439_367_674_143_721_6e3,
            1.230_339_354_803_749_4e3,
        ];
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        xnum += C[7];
        xden += D[7];
        let ysq = (y * 16.0).floor() / 16.0;
        let del = (y - ysq) * (y + ysq);
        (-ysq * ysq).exp() * (-del).exp() * (xnum / xden)
    } else if y < 27.3 {
        const P: [f64; 6] = [
            3.053_266_349_612_323_6e-1,
            3.603_448_999_498_044_5e-1,
            1.257_817_261_112_292_5e-1,
            1.608_378_514_874_227_5e-2,
            6.587_491_615_298_378e-4,
            1.631_538_713_730_209_8e-2,
        ];
        const Q: [f64; 5] = [
            2.568_520_192_289_822,
            1.872_952_849_923_460_4,
            5.279_051_029_514_285e-1,
            6.051_834_131_244_132e-2,
            2.335_204_976_268_691_8e-3,
        ];
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        xnum += P[4];
        xden += Q[4];
        let tail = (ysq * xnum / xden) / y;
        let rsqrtpi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
        let ysq_f = (y * 16.0).floor() / 16.0;
        let del = (y - ysq_f) * (y + ysq_f);
        (-ysq_f * ysq_f).exp() * (-del).exp() * (rsqrtpi - tail) / y
    } else {
        0.0
    };

    if x < 0.0 { 2.0 - res } else { res }
}

/// Standard normal cumulative distribution function `Phi(x)`.
#[inline]
pub fn ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc_f64(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// Inverse standard normal CDF `Phi^{-1}(p)` using Acklam's approximation + Halley refinement (~1e-15).
pub fn ndtri(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // Two Halley refinement steps achieve full IEEE-754 f64 accuracy.
    let inv_sqrt_2pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5 * std::f64::consts::FRAC_1_SQRT_2;
    for _ in 0..2 {
        let err = ndtr(x) - p;
        let pdf = inv_sqrt_2pi * (-0.5 * x * x).exp();
        if pdf > 0.0 {
            let u = err / pdf;
            x -= u / (1.0 + 0.5 * x * u);
        }
    }
    x
}

/// Brent's 1D minimization over bracket `(ax, bx, cx)` or `[a, b]` matching `scipy.optimize.brent`.
pub fn brent_minimize<F: FnMut(f64) -> f64>(mut f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    const CGOLD: f64 = 0.381_966_011_250_105_1;
    const ZEPS: f64 = 1e-15;

    // First find a valid 3-point bracket `(xa, xb, xc)` with `f(xb) < f(xa)` and `f(xb) < f(xc)`
    // starting from `[a, b]` (default `[-2.0, 2.0]` or `[0.0, 2.0]`), expanding downhill if needed.
    let mut xa = a;
    let mut xb = b;
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fb > fa {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let gold = 1.618_033_988_749_895;
    let mut xc = xb + gold * (xb - xa);
    let mut fc = f(xc);
    for _ in 0..50 {
        if fb <= fc {
            break;
        }
        let r = (xb - xa) * (fb - fc);
        let q = (xb - xc) * (fb - fa);
        let denom = 2.0 * (q - r).abs().max(1e-20) * (q - r).signum();
        let u_lim = xb + 100.0 * (xc - xb);
        let mut u = xb - ((xb - xc) * q - (xb - xa) * r) / denom;
        let mut fu;
        if (xb - u) * (u - xc) > 0.0 {
            fu = f(u);
            if fu < fc {
                xa = xb;
                xb = u;
                break;
            } else if fu > fb {
                xc = u;
                break;
            }
            u = xc + gold * (xc - xb);
            fu = f(u);
        } else if (xc - u) * (u - u_lim) > 0.0 {
            fu = f(u);
            if fu < fc {
                xb = xc;
                xc = u;
                u = xc + gold * (xc - xb);
                fb = fc;
                fc = fu;
                let fu_next = f(u);
                xa = xb;
                xb = xc;
                xc = u;
                fa = fb;
                fb = fc;
                fc = fu_next;
                continue;
            }
        } else {
            u = xc + gold * (xc - xb);
            fu = f(u);
        }
        xa = xb;
        xb = xc;
        xc = u;
        fa = fb;
        fb = fc;
        fc = fu;
    }

    a = xa.min(xc);
    b = xa.max(xc);
    let mut x = xb;
    let mut w = xb;
    let mut v = xb;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;

    for _ in 0..500 {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= (tol2 - 0.5 * (b - a)) {
            return x;
        }
        let mut use_golden = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            e = if x >= xm { a - x } else { b - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1 * if d >= 0.0 { 1.0 } else { -1.0 }
        };
        let fu = f(u);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

/// Linear-interpolated percentile on a sorted non-empty slice (`q_percent` in `[0, 100]`).
#[inline]
pub fn percentile_sorted(sorted: &[f64], q_percent: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = (q_percent / 100.0).clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// `averaged_inverted_cdf` quantile matching NumPy (`method='averaged_inverted_cdf'`) and `KBinsDiscretizer`.
#[inline]
pub fn averaged_inverted_cdf_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 || q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return sorted[n - 1];
    }
    let nq = (n as f64) * q;
    let floor_nq = nq.floor();
    let j = (floor_nq as usize).saturating_sub(1);
    let k = (floor_nq as usize).min(n - 1);
    let g = nq - floor_nq;
    if g == 0.0 {
        0.5 * (sorted[j] + sorted[k])
    } else {
        sorted[k]
    }
}

/// NaN-skipping per-column summary statistics (used when input contains NaNs).
pub struct NanColStats {
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub count: Vec<f64>,
}

pub fn compute_nan_col_stats(
    x: &Matrix,
    sample_weight: Option<&[f64]>,
) -> OchreResult<NanColStats> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if let Some(w) = sample_weight
        && w.len() != x.rows
    {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: w.len(),
        });
    }

    let cols = x.cols;
    let rows = x.rows;
    let per_col: Vec<(f64, f64, f64, f64, f64)> = (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut w_sum = 0.0;
            let mut mean = 0.0;
            let mut m2 = 0.0;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for r in 0..rows {
                let v = x.data[r * cols + j];
                if v.is_nan() {
                    continue;
                }
                let w = sample_weight.map_or(1.0, |ws| ws[r]);
                if w == 0.0 {
                    continue;
                }
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
                let new_w = w_sum + w;
                let delta = v - mean;
                let r_step = delta * (w / new_w);
                mean += r_step;
                m2 += w_sum * delta * r_step;
                w_sum = new_w;
            }
            if w_sum <= 0.0 {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.0)
            } else {
                let var = (m2 / w_sum).max(0.0);
                (mean, var, min_v, max_v, w_sum)
            }
        })
        .collect();

    let mut mean = Vec::with_capacity(cols);
    let mut var = Vec::with_capacity(cols);
    let mut std = Vec::with_capacity(cols);
    let mut min = Vec::with_capacity(cols);
    let mut max = Vec::with_capacity(cols);
    let mut count = Vec::with_capacity(cols);
    for (m, v, mn, mx, c) in per_col {
        mean.push(m);
        var.push(v);
        std.push(v.sqrt());
        min.push(mn);
        max.push(mx);
        count.push(c);
    }
    Ok(NanColStats {
        mean,
        var,
        std,
        min,
        max,
        count,
    })
}
