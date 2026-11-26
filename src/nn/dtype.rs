// OchreML Mixed-Precision Data Types & Tensor Storage (*≧ω≦*)
// Pure-Rust IEEE-754 f32 <-> bf16 and f32 <-> f16 conversions (zero unsafe)
// and polymorphic TensorStorage backing f32, bfloat16, and float16 tensors.

/// Supported tensor element scalar data types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    F16,
}

impl DType {
    /// Size in bytes of a single scalar element.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::BF16 => write!(f, "bf16"),
            DType::F16 => write!(f, "f16"),
        }
    }
}

/// Convert IEEE-754 `f32` to `bfloat16` (`u16`) using round-to-nearest-even (RNE).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Preserve NaN by forcing quiet NaN bit so payload cannot collapse into Inf.
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Convert `bfloat16` (`u16`) to IEEE-754 `f32`.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Convert IEEE-754 `f32` to IEEE-754 `float16` (`u16`) using round-to-nearest-even (RNE).
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7E00 | ((mant >> 13) as u16);
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00;
    }

    if unbiased >= -14 {
        let h_exp = ((unbiased + 15) as u32) << 10;
        let round_bit = 1u32 << 12;
        let sticky = mant & 0x0FFF;
        let lsb = (mant >> 13) & 1;
        let mut h_mant = mant >> 13;
        if (mant & round_bit) != 0 && (sticky != 0 || lsb != 0) {
            h_mant += 1;
        }
        return sign | ((h_exp + h_mant) as u16);
    }

    if unbiased >= -25 && exp > 0 {
        // Subnormal binary16 (note: f32 subnormals have exp == 0 and are smaller than 2^-126,
        // so they always underflow to 0 in f16).
        let m = mant | 0x0080_0000;
        let shift = ((-14 - unbiased) + 13) as u32;
        let half = 1u32 << (shift - 1);
        let rem_mask = (1u32 << shift) - 1;
        let rem = m & rem_mask;
        let mut h_mant = m >> shift;
        if rem > half || (rem == half && (h_mant & 1) != 0) {
            h_mant += 1;
        }
        return sign | (h_mant as u16);
    }

    sign
}

/// Convert IEEE-754 `float16` (`u16`) to IEEE-754 `f32`.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f_exp | (m << 13));
    }

    if exp == 31 {
        if mant == 0 {
            return f32::from_bits(sign | 0x7F80_0000);
        }
        return f32::from_bits(sign | 0x7FC0_0000 | (mant << 13));
    }

    let f_exp = (exp + 112) << 23;
    f32::from_bits(sign | f_exp | (mant << 13))
}

/// Polymorphic contiguous tensor buffer supporting `f32`, `bf16`, and `f16`.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorStorage {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    F16(Vec<u16>),
}

impl TensorStorage {
    /// Number of scalar elements stored.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TensorStorage::F32(v) => v.len(),
            TensorStorage::BF16(v) | TensorStorage::F16(v) => v.len(),
        }
    }

    /// Returns true if the storage contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element `DType` of this storage.
    #[inline]
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::F32(_) => DType::F32,
            TensorStorage::BF16(_) => DType::BF16,
            TensorStorage::F16(_) => DType::F16,
        }
    }

    /// Converts stored elements to a contiguous `Vec<f32>`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            TensorStorage::F32(v) => v.clone(),
            TensorStorage::BF16(v) => v.iter().map(|&b| bf16_to_f32(b)).collect(),
            TensorStorage::F16(v) => v.iter().map(|&h| f16_to_f32(h)).collect(),
        }
    }

    /// Constructs `TensorStorage` from a `&[f32]` slice quantized into `dtype`.
    pub fn from_f32_slice(data: &[f32], dtype: DType) -> Self {
        match dtype {
            DType::F32 => TensorStorage::F32(data.to_vec()),
            DType::BF16 => TensorStorage::BF16(data.iter().map(|&x| f32_to_bf16(x)).collect()),
            DType::F16 => TensorStorage::F16(data.iter().map(|&x| f32_to_f16(x)).collect()),
        }
    }

    /// Borrows the underlying `&[f32]` slice without copying when `dtype == DType::F32`.
    #[inline]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorStorage::F32(v) => Some(v.as_slice()),
            TensorStorage::BF16(_) | TensorStorage::F16(_) => None,
        }
    }

    /// Allocates zero-initialized storage of length `len` for `dtype`.
    pub fn zeros(len: usize, dtype: DType) -> Self {
        match dtype {
            DType::F32 => TensorStorage::F32(vec![0.0; len]),
            DType::BF16 => TensorStorage::BF16(vec![0u16; len]),
            DType::F16 => TensorStorage::F16(vec![0u16; len]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip_accuracy_and_specials() {
        let vals = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            std::f32::consts::PI,
            -42.25,
            1e-30,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for &x in &vals {
            let b = f32_to_bf16(x);
            let y = bf16_to_f32(b);
            if x.is_infinite() {
                assert_eq!(y, x);
            } else if x == 0.0 {
                assert_eq!(y.to_bits(), x.to_bits());
            } else {
                let rel_err = ((y - x) / x).abs();
                assert!(
                    rel_err < 0.005,
                    "bf16 rel_err too large for {x}: got {y} ({rel_err})"
                );
            }
        }
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        // RNE tie-to-even test: 1.0 + 0.5 ulp_bf16 (0x3F80_8000) -> even 0x3F80
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 1.0 + 1.5 ulp_bf16 (0x3F81_8000) -> even 0x3F82
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
    }

    #[test]
    fn test_f16_roundtrip_accuracy_subnormals_and_specials() {
        let vals = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            65504.0,
            -65504.0,
            6.1035156e-5, // smallest normal f16
            5.9604645e-8, // smallest subnormal f16 (2^-24)
            std::f32::consts::PI,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for &x in &vals {
            let h = f32_to_f16(x);
            let y = f16_to_f32(h);
            if x.is_infinite() {
                assert_eq!(y, x);
            } else if x == 0.0 {
                assert_eq!(y.to_bits(), x.to_bits());
            } else {
                let rel_err = ((y - x) / x).abs();
                assert!(
                    rel_err < 0.002,
                    "f16 rel_err too large for {x}: got {y} ({rel_err})"
                );
            }
        }
        // Overflow to Inf
        assert_eq!(f16_to_f32(f32_to_f16(70000.0)), f32::INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(-70000.0)), f32::NEG_INFINITY);
        // Underflow to signed zero
        assert_eq!(
            f16_to_f32(f32_to_f16(-1e-10)).to_bits(),
            (-0.0f32).to_bits()
        );
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_tensor_storage_methods() {
        let src = vec![1.0f32, -2.5, 0.125, 16.0];
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let st = TensorStorage::from_f32_slice(&src, dtype);
            assert_eq!(st.len(), 4);
            assert!(!st.is_empty());
            assert_eq!(st.dtype(), dtype);
            if dtype == DType::F32 {
                assert_eq!(st.as_f32_slice(), Some(src.as_slice()));
            } else {
                assert!(st.as_f32_slice().is_none());
            }
            let back = st.to_f32_vec();
            for (a, b) in src.iter().zip(back.iter()) {
                assert!((a - b).abs() < 0.02);
            }

            let z = TensorStorage::zeros(5, dtype);
            assert_eq!(z.len(), 5);
            assert_eq!(z.to_f32_vec(), vec![0.0; 5]);
        }
        let empty = TensorStorage::zeros(0, DType::F32);
        assert!(empty.is_empty());
    }
}
