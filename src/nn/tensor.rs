// OchreML Tensor Core Module (＾▽＾)
// Contiguous f32 compute tensors with BF16/F16 storage simulation, Backend dispatch, and autograd hooks.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, GradStore, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::sgemm;

/// Deterministic fast PRNG (SplitMix64 + Xoshiro256++) for weight init, dropout, and shuffling.
#[derive(Clone, Debug)]
pub struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    /// Seeds a `Xoshiro256++` generator via `SplitMix64`.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        if s == [0; 4] {
            s[0] = 0x1234_5678_9ABC_DEF0;
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Returns a uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Returns a uniform `f32` in `[low, high)`.
    #[inline]
    pub fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Returns a pair of standard normal `f32` samples via Box-Muller.
    #[inline]
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Fisher-Yates shuffle on an index slice.
    pub fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Core multi-dimensional tensor backed by contiguous `f32` compute storage.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<f32>,
    pub dtype: DType,
    pub device: DeviceType,
    pub requires_grad: bool,
    pub node_id: Option<usize>,
}

/// Computes row-major contiguous strides for `shape`.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

#[inline]
fn shape_numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

#[inline]
fn maybe_quantize(data: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => data,
        DType::BF16 | DType::F16 => TensorStorage::from_f32_slice(&data, dtype).to_f32_vec(),
    }
}

impl Tensor {
    /// Creates a new `Tensor`, validating element count and optionally allocating a leaf node.
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: DType,
        device: DeviceType,
        requires_grad: bool,
    ) -> OchreResult<Self> {
        let expected = shape_numel(&shape);
        if data.len() != expected {
            return Err(OchreError::InvalidParameter(format!(
                "Tensor data length {} does not match shape {:?} (expected {expected}) ( >_< )",
                data.len(),
                shape
            )));
        }
        let strides = compute_strides(&shape);
        let data = maybe_quantize(data, dtype);
        let node_id = if requires_grad {
            Some(AutogradTape::alloc_leaf())
        } else {
            None
        };
        Ok(Self {
            shape,
            strides,
            data,
            dtype,
            device,
            requires_grad,
            node_id,
        })
    }

    /// Creates a zero-filled `Tensor`.
    pub fn zeros(shape: Vec<usize>, dtype: DType, device: DeviceType) -> Self {
        let numel = shape_numel(&shape);
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data: vec![0.0f32; numel],
            dtype,
            device,
            requires_grad: false,
            node_id: None,
        }
    }

    /// Creates a one-filled `Tensor`.
    pub fn ones(shape: Vec<usize>, dtype: DType, device: DeviceType) -> Self {
        let numel = shape_numel(&shape);
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data: vec![1.0f32; numel],
            dtype,
            device,
            requires_grad: false,
            node_id: None,
        }
    }

    /// Creates a `Tensor` from an `f64` slice, casting to `f32` (and `dtype` precision).
    pub fn from_f64_slice(
        data: &[f64],
        shape: Vec<usize>,
        dtype: DType,
        device: DeviceType,
        requires_grad: bool,
    ) -> OchreResult<Self> {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        Self::new(f32_data, shape, dtype, device, requires_grad)
    }

    /// Creates a standard normal `N(0, 1)` tensor using a deterministic `Xoshiro256++` RNG.
    pub fn randn(shape: Vec<usize>, dtype: DType, device: DeviceType, seed: u64) -> Self {
        Self::randn_with(shape, 0.0, 1.0, dtype, device, seed, false)
    }

    /// Creates a normal `N(mean, std^2)` tensor using a deterministic `Xoshiro256++` RNG.
    pub fn randn_with(
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        dtype: DType,
        device: DeviceType,
        seed: u64,
        requires_grad: bool,
    ) -> Self {
        let numel = shape_numel(&shape);
        let mut rng = FastRng::new(seed);
        let mut data = Vec::with_capacity(numel);
        while data.len() < numel {
            let (z0, z1) = rng.next_normal_pair();
            data.push(mean + std * z0);
            if data.len() < numel {
                data.push(mean + std * z1);
            }
        }
        let data = maybe_quantize(data, dtype);
        let strides = compute_strides(&shape);
        let node_id = if requires_grad {
            Some(AutogradTape::alloc_leaf())
        } else {
            None
        };
        Self {
            shape,
            strides,
            data,
            dtype,
            device,
            requires_grad,
            node_id,
        }
    }

    /// Creates a uniform `U(low, high)` tensor using a deterministic `Xoshiro256++` RNG.
    pub fn uniform(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        dtype: DType,
        device: DeviceType,
        seed: u64,
        requires_grad: bool,
    ) -> Self {
        let numel = shape_numel(&shape);
        let mut rng = FastRng::new(seed);
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            data.push(rng.uniform_f32(low, high));
        }
        let data = maybe_quantize(data, dtype);
        let strides = compute_strides(&shape);
        let node_id = if requires_grad {
            Some(AutogradTape::alloc_leaf())
        } else {
            None
        };
        Self {
            shape,
            strides,
            data,
            dtype,
            device,
            requires_grad,
            node_id,
        }
    }

    /// Creates a trainable leaf parameter (`requires_grad = true`) with a fresh leaf node ID.
    pub fn parameter(
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: DType,
        device: DeviceType,
    ) -> OchreResult<Self> {
        Self::new(data, shape, dtype, device, true)
    }

    /// Creates a `Tensor` from packed `TensorStorage`.
    pub fn from_storage(
        storage: &TensorStorage,
        shape: Vec<usize>,
        device: DeviceType,
        requires_grad: bool,
    ) -> OchreResult<Self> {
        let dtype = storage.dtype();
        let data = storage.to_f32_vec();
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Converts the tensor's compute values to packed `TensorStorage` in `self.dtype`.
    pub fn to_storage(&self) -> TensorStorage {
        TensorStorage::from_f32_slice(&self.data, self.dtype)
    }

    /// Quantizes/casts this tensor to `dtype` via `TensorStorage` round-trip.
    pub fn to_dtype(&self, dtype: DType) -> Self {
        let data = if self.dtype == dtype {
            self.data.clone()
        } else {
            TensorStorage::from_f32_slice(&self.data, dtype).to_f32_vec()
        };
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data,
            dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            node_id: if self.requires_grad {
                Some(AutogradTape::alloc_leaf())
            } else {
                None
            },
        }
    }

    /// Moves/clones this tensor to `device`.
    pub fn to_device(&self, device: DeviceType) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.clone(),
            dtype: self.dtype,
            device,
            requires_grad: self.requires_grad,
            node_id: self.node_id,
        }
    }

    /// Returns a detached clone with `requires_grad = false` and `node_id = None`.
    pub fn detach(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: false,
            node_id: None,
        }
    }

    /// Enables or disables `requires_grad` on this tensor, allocating a leaf node ID if needed.
    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.node_id.is_none() {
            self.node_id = Some(AutogradTape::alloc_leaf());
        } else if !requires_grad {
            self.node_id = None;
        }
        self
    }

    /// Total number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns true if strides match row-major contiguous layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Reshapes the tensor, preserving data and recording `Op::Reshape` when grad is enabled.
    pub fn reshape(&self, new_shape: Vec<usize>) -> OchreResult<Self> {
        let new_numel = shape_numel(&new_shape);
        if new_numel != self.data.len() {
            return Err(OchreError::InvalidParameter(format!(
                "Cannot reshape tensor of shape {:?} ({} elems) into {:?} ({} elems) ( >_< )",
                self.shape,
                self.data.len(),
                new_shape,
                new_numel
            )));
        }
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Reshape {
                input_id: self.node_id,
                input_shape: self.shape.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&new_shape),
            shape: new_shape,
            data: self.data.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Transposes the last two dimensions of a rank >= 2 tensor via `Backend::transpose_last2_f32`.
    pub fn transpose_last2(&self) -> OchreResult<Self> {
        let ndim = self.shape.len();
        if ndim < 2 {
            return Err(OchreError::InvalidParameter(format!(
                "transpose_last2 requires rank >= 2, got shape {:?} ( >_< )",
                self.shape
            )));
        }
        let rows = self.shape[ndim - 2];
        let cols = self.shape[ndim - 1];
        let batch: usize = self.shape[..ndim - 2]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let out_data =
            get_backend(&self.device).transpose_last2_f32(batch, rows, cols, &self.data)?;

        let mut out_shape = self.shape.clone();
        out_shape[ndim - 2] = cols;
        out_shape[ndim - 1] = rows;

        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::TransposeLast2 {
                input_id: self.node_id,
                batch,
                rows,
                cols,
                device: self.device.clone(),
            }))
        } else {
            None
        };

        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Elementwise addition with bias/row and general NumPy-style broadcasting.
    pub fn add(&self, other: &Self) -> OchreResult<Self> {
        let (out_shape, mut out_data) = binary_broadcast_op(self, other, |a, b| a + b)?;
        out_data = maybe_quantize(out_data, self.dtype);
        let grad_active =
            AutogradTape::is_grad_enabled() && (self.requires_grad || other.requires_grad);
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Add {
                a_id: self.node_id,
                b_id: other.node_id,
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
                out_shape: out_shape.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Elementwise subtraction with broadcasting.
    pub fn sub(&self, other: &Self) -> OchreResult<Self> {
        let (out_shape, mut out_data) = binary_broadcast_op(self, other, |a, b| a - b)?;
        out_data = maybe_quantize(out_data, self.dtype);
        let grad_active =
            AutogradTape::is_grad_enabled() && (self.requires_grad || other.requires_grad);
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Sub {
                a_id: self.node_id,
                b_id: other.node_id,
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
                out_shape: out_shape.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Elementwise multiplication with broadcasting.
    pub fn mul(&self, other: &Self) -> OchreResult<Self> {
        let (out_shape, mut out_data) = binary_broadcast_op(self, other, |a, b| a * b)?;
        out_data = maybe_quantize(out_data, self.dtype);
        let grad_active =
            AutogradTape::is_grad_enabled() && (self.requires_grad || other.requires_grad);
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Mul {
                a_id: self.node_id,
                b_id: other.node_id,
                a_data: if other.requires_grad {
                    self.data.clone()
                } else {
                    Vec::new()
                },
                b_data: if self.requires_grad {
                    other.data.clone()
                } else {
                    Vec::new()
                },
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
                out_shape: out_shape.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Scales all elements by a constant scalar `f32`.
    pub fn scale(&self, scalar: f32) -> OchreResult<Self> {
        let out_data = maybe_quantize(self.data.iter().map(|&x| x * scalar).collect(), self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Scale {
                input_id: self.node_id,
                scalar,
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Matrix multiplication supporting 2D `[m, k] @ [k, n]`, ND-by-2D `[..., m, k] @ [k, n]`,
    /// and batched ND `[b..., m, k] @ [b..., k, n]`.
    pub fn matmul(&self, other: &Self) -> OchreResult<Self> {
        self.matmul_trans(other, false, false)
    }

    /// Generalized matrix multiplication with optional transpose of `self` and/or `other`.
    pub fn matmul_trans(&self, other: &Self, trans_a: bool, trans_b: bool) -> OchreResult<Self> {
        let a_ndim = self.shape.len();
        let b_ndim = other.shape.len();
        if a_ndim < 2 || b_ndim < 2 {
            return Err(OchreError::InvalidParameter(format!(
                "matmul requires rank >= 2 tensors, got {:?} and {:?} ( >_< )",
                self.shape, other.shape
            )));
        }

        // Case 1: ND [batch..., m, k] @ 2D [k, n] (or transposed 2D [n, k]) when !trans_a
        if b_ndim == 2 && !trans_a {
            let k_a = self.shape[a_ndim - 1];
            let (k_b, n) = if trans_b {
                (other.shape[1], other.shape[0])
            } else {
                (other.shape[0], other.shape[1])
            };
            let m_flat: usize = self.shape[..a_ndim - 1]
                .iter()
                .copied()
                .product::<usize>()
                .max(1);
            if k_a != k_b {
                return Err(OchreError::MatmulDimensionMismatch {
                    rows_a: m_flat,
                    cols_a: k_a,
                    rows_b: k_b,
                    cols_b: n,
                });
            }

            let mut out_data = if matches!(self.device, DeviceType::Cpu) {
                let mut buf = vec![0.0f32; m_flat * n];
                sgemm(
                    m_flat,
                    n,
                    k_a,
                    &self.data,
                    &other.data,
                    &mut buf,
                    false,
                    trans_b,
                    1.0,
                    0.0,
                )?;
                buf
            } else {
                get_backend(&self.device).batched_matmul_f32(
                    1,
                    m_flat,
                    n,
                    k_a,
                    &self.data,
                    &other.data,
                    false,
                    trans_b,
                )?
            };
            out_data = maybe_quantize(out_data, self.dtype);

            let mut out_shape = self.shape[..a_ndim - 1].to_vec();
            out_shape.push(n);

            let grad_active =
                AutogradTape::is_grad_enabled() && (self.requires_grad || other.requires_grad);
            let node_id = if grad_active {
                let a_saved = if other.requires_grad {
                    self.data.clone()
                } else {
                    Vec::new()
                };
                let b_saved = if self.requires_grad {
                    other.data.clone()
                } else {
                    Vec::new()
                };
                let op = if !trans_b {
                    Op::Matmul {
                        a_id: self.node_id,
                        b_id: other.node_id,
                        a_data: a_saved,
                        b_data: b_saved,
                        m: m_flat,
                        k: k_a,
                        n,
                        device: self.device.clone(),
                    }
                } else {
                    Op::BatchedMatmul {
                        a_id: self.node_id,
                        b_id: other.node_id,
                        a_data: a_saved,
                        b_data: b_saved,
                        batch: 1,
                        m: m_flat,
                        k: k_a,
                        n,
                        trans_a: false,
                        trans_b: true,
                        device: self.device.clone(),
                    }
                };
                Some(AutogradTape::record(op))
            } else {
                None
            };

            return Ok(Self {
                strides: compute_strides(&out_shape),
                shape: out_shape,
                data: out_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: grad_active,
                node_id,
            });
        }

        // Case 2: Equal-rank 2D or batched ND matmul
        if a_ndim != b_ndim || self.shape[..a_ndim - 2] != other.shape[..b_ndim - 2] {
            return Err(OchreError::InvalidParameter(format!(
                "Batched matmul batch dimensions mismatch: {:?} vs {:?} ( >_< )",
                self.shape, other.shape
            )));
        }

        let (m, k_a) = if trans_a {
            (self.shape[a_ndim - 1], self.shape[a_ndim - 2])
        } else {
            (self.shape[a_ndim - 2], self.shape[a_ndim - 1])
        };
        let (k_b, n) = if trans_b {
            (other.shape[b_ndim - 1], other.shape[b_ndim - 2])
        } else {
            (other.shape[b_ndim - 2], other.shape[b_ndim - 1])
        };
        if k_a != k_b {
            return Err(OchreError::MatmulDimensionMismatch {
                rows_a: m,
                cols_a: k_a,
                rows_b: k_b,
                cols_b: n,
            });
        }

        let batch: usize = self.shape[..a_ndim - 2]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let mut out_data = if batch == 1 && matches!(self.device, DeviceType::Cpu) {
            let mut buf = vec![0.0f32; m * n];
            sgemm(
                m,
                n,
                k_a,
                &self.data,
                &other.data,
                &mut buf,
                trans_a,
                trans_b,
                1.0,
                0.0,
            )?;
            buf
        } else {
            get_backend(&self.device).batched_matmul_f32(
                batch,
                m,
                n,
                k_a,
                &self.data,
                &other.data,
                trans_a,
                trans_b,
            )?
        };
        out_data = maybe_quantize(out_data, self.dtype);

        let mut out_shape = self.shape[..a_ndim - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let grad_active =
            AutogradTape::is_grad_enabled() && (self.requires_grad || other.requires_grad);
        let node_id = if grad_active {
            let a_saved = if other.requires_grad {
                self.data.clone()
            } else {
                Vec::new()
            };
            let b_saved = if self.requires_grad {
                other.data.clone()
            } else {
                Vec::new()
            };
            let op = if batch == 1 && !trans_a && !trans_b {
                Op::Matmul {
                    a_id: self.node_id,
                    b_id: other.node_id,
                    a_data: a_saved,
                    b_data: b_saved,
                    m,
                    k: k_a,
                    n,
                    device: self.device.clone(),
                }
            } else {
                Op::BatchedMatmul {
                    a_id: self.node_id,
                    b_id: other.node_id,
                    a_data: a_saved,
                    b_data: b_saved,
                    batch,
                    m,
                    k: k_a,
                    n,
                    trans_a,
                    trans_b,
                    device: self.device.clone(),
                }
            };
            Some(AutogradTape::record(op))
        } else {
            None
        };

        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Sums elements across all axes (`axis = None`) or a single `axis`.
    pub fn sum(&self, axis: Option<usize>) -> OchreResult<Self> {
        let (out_shape, raw_out) = reduce_sum(&self.data, &self.shape, axis)?;
        let out_data = maybe_quantize(raw_out, self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Sum {
                input_id: self.node_id,
                input_shape: self.shape.clone(),
                axis,
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Mean of elements across all axes (`axis = None`) or a single `axis`.
    pub fn mean(&self, axis: Option<usize>) -> OchreResult<Self> {
        let (out_shape, mut raw_out) = reduce_sum(&self.data, &self.shape, axis)?;
        let denom = match axis {
            Some(ax) => self.shape[ax].max(1) as f32,
            None => self.data.len().max(1) as f32,
        };
        for v in &mut raw_out {
            *v /= denom;
        }
        let out_data = maybe_quantize(raw_out, self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Mean {
                input_id: self.node_id,
                input_shape: self.shape.clone(),
                axis,
            }))
        } else {
            None
        };
        Ok(Self {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Rectified Linear Unit: `max(0, x)`.
    pub fn relu(&self) -> OchreResult<Self> {
        let out_data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::ReLU {
                input_id: self.node_id,
                input_data: self.data.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self) -> OchreResult<Self> {
        let out_data = maybe_quantize(
            self.data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect(),
            self.dtype,
        );
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Sigmoid {
                input_id: self.node_id,
                out_data: out_data.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Hyperbolic tangent: `tanh(x)`.
    pub fn tanh(&self) -> OchreResult<Self> {
        let out_data = maybe_quantize(self.data.iter().map(|&x| x.tanh()).collect(), self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Tanh {
                input_id: self.node_id,
                out_data: out_data.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Gaussian Error Linear Unit via `Backend::gelu_f32`.
    pub fn gelu(&self) -> OchreResult<Self> {
        let raw_out = get_backend(&self.device).gelu_f32(&self.data);
        let out_data = maybe_quantize(raw_out, self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::GELU {
                input_id: self.node_id,
                input_data: self.data.clone(),
                device: self.device.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Sigmoid Linear Unit (`x * sigmoid(x)`) via `Backend::silu_f32`.
    pub fn silu(&self) -> OchreResult<Self> {
        let raw_out = get_backend(&self.device).silu_f32(&self.data);
        let out_data = maybe_quantize(raw_out, self.dtype);
        let grad_active = AutogradTape::is_grad_enabled() && self.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::SiLU {
                input_id: self.node_id,
                input_data: self.data.clone(),
                device: self.device.clone(),
            }))
        } else {
            None
        };
        Ok(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Runs reverse-mode autodiff from this tensor and returns the accumulated `GradStore`.
    pub fn backward(&self) -> OchreResult<GradStore> {
        if !self.requires_grad {
            return Err(OchreError::InvalidParameter(
                "Cannot call backward() on a tensor with requires_grad=false ( >_< )".to_string(),
            ));
        }
        let root_id = self.node_id.ok_or_else(|| {
            OchreError::InvalidParameter(
                "Tensor requires_grad=true but has no autograd node_id ( >_< )".to_string(),
            )
        })?;
        AutogradTape::backward(root_id, self.data.len())
    }
}

fn binary_broadcast_op<F>(a: &Tensor, b: &Tensor, op: F) -> OchreResult<(Vec<usize>, Vec<f32>)>
where
    F: Fn(f32, f32) -> f32,
{
    if a.shape == b.shape {
        let out: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return Ok((a.shape.clone(), out));
    }
    if b.data.len() == 1 {
        let bv = b.data[0];
        let out: Vec<f32> = a.data.iter().map(|&x| op(x, bv)).collect();
        return Ok((a.shape.clone(), out));
    }
    if a.data.len() == 1 {
        let av = a.data[0];
        let out: Vec<f32> = b.data.iter().map(|&y| op(av, y)).collect();
        return Ok((b.shape.clone(), out));
    }
    if a.shape.len() == 2 && (b.shape == [a.shape[1]] || b.shape == [1, a.shape[1]]) {
        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut out = vec![0.0f32; a.data.len()];
        for r in 0..rows {
            let off = r * cols;
            for c in 0..cols {
                out[off + c] = op(a.data[off + c], b.data[c]);
            }
        }
        return Ok((a.shape.clone(), out));
    }

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_ndim = out_shape.len();
    let out_numel = shape_numel(&out_shape);
    let a_strides = padded_broadcast_strides(&a.shape, out_ndim);
    let b_strides = padded_broadcast_strides(&b.shape, out_ndim);

    let mut out = Vec::with_capacity(out_numel);
    let mut coords = vec![0usize; out_ndim];
    for flat in 0..out_numel {
        let mut rem = flat;
        for i in (0..out_ndim).rev() {
            let dim = out_shape[i].max(1);
            coords[i] = rem % dim;
            rem /= dim;
        }
        let mut ia = 0usize;
        let mut ib = 0usize;
        for i in 0..out_ndim {
            ia += coords[i] * a_strides[i];
            ib += coords[i] * b_strides[i];
        }
        out.push(op(a.data[ia], b.data[ib]));
    }
    Ok((out_shape, out))
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> OchreResult<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];
    for i in 0..ndim {
        let da = if i < ndim - a.len() {
            1
        } else {
            a[i - (ndim - a.len())]
        };
        let db = if i < ndim - b.len() {
            1
        } else {
            b[i - (ndim - b.len())]
        };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return Err(OchreError::InvalidParameter(format!(
                "Incompatible shapes for broadcasting: {:?} and {:?} ( >_< )",
                a, b
            )));
        }
    }
    Ok(out)
}

fn padded_broadcast_strides(shape: &[usize], out_ndim: usize) -> Vec<usize> {
    let pad = out_ndim - shape.len();
    let mut padded = vec![1usize; out_ndim];
    for (i, &d) in shape.iter().enumerate() {
        padded[pad + i] = d;
    }
    let mut strides = vec![0usize; out_ndim];
    let mut stride = 1usize;
    for i in (0..out_ndim).rev() {
        if padded[i] == 1 {
            strides[i] = 0;
        } else {
            strides[i] = stride;
            stride *= padded[i];
        }
    }
    strides
}

fn reduce_sum(
    data: &[f32],
    shape: &[usize],
    axis: Option<usize>,
) -> OchreResult<(Vec<usize>, Vec<f32>)> {
    match axis {
        None => {
            let total: f32 = data.iter().copied().sum();
            Ok((vec![1], vec![total]))
        }
        Some(ax) => {
            if ax >= shape.len() {
                return Err(OchreError::InvalidParameter(format!(
                    "Reduction axis {ax} out of bounds for shape {:?} ( >_< )",
                    shape
                )));
            }
            let outer: usize = shape[..ax].iter().copied().product::<usize>().max(1);
            let ax_len = shape[ax];
            let inner: usize = shape[ax + 1..].iter().copied().product::<usize>().max(1);
            let mut out = vec![0.0f32; outer * inner];
            for o in 0..outer {
                for a in 0..ax_len {
                    let src_base = (o * ax_len + a) * inner;
                    let dst_base = o * inner;
                    for i in 0..inner {
                        out[dst_base + i] += data[src_base + i];
                    }
                }
            }
            let mut out_shape = shape.to_vec();
            out_shape.remove(ax);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            Ok((out_shape, out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::with_no_grad;

    #[test]
    fn test_tensor_constructors_and_dtype_roundtrip() {
        let z = Tensor::zeros(vec![2, 3], DType::F32, DeviceType::Cpu);
        assert_eq!(z.numel(), 6);
        assert_eq!(z.strides, vec![3, 1]);

        let r = Tensor::randn(vec![4, 5], DType::F32, DeviceType::Cpu, 42);
        assert_eq!(r.numel(), 20);

        let bf = r.to_dtype(DType::BF16);
        assert_eq!(bf.dtype, DType::BF16);
        for (&a, &b) in r.data.iter().zip(bf.data.iter()) {
            assert!((a - b).abs() < 0.05);
        }

        let f16 = r.to_dtype(DType::F16);
        assert_eq!(f16.dtype, DType::F16);
        for (&a, &b) in r.data.iter().zip(f16.data.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_tensor_matmul_add_relu_backward() {
        AutogradTape::clear();
        let x = Tensor::parameter(
            vec![1.0, 2.0, -1.0, 0.5],
            vec![2, 2],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::parameter(
            vec![2.0, -1.0, 0.5, 3.0],
            vec![2, 2],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let b = Tensor::parameter(vec![0.1, -0.2], vec![2], DType::F32, DeviceType::Cpu).unwrap();

        let h = x.matmul(&w).unwrap().add(&b).unwrap().relu().unwrap();
        let loss = h.sum(None).unwrap();
        let grads = loss.backward().unwrap();

        assert_eq!(AutogradTape::len(), 0);
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dw = grads.get(w.node_id.unwrap()).unwrap();
        let db = grads.get(b.node_id.unwrap()).unwrap();

        // Finite-difference check
        let eps = 1e-3f32;
        for idx in 0..4 {
            let mut x_plus = x.data.clone();
            let mut x_minus = x.data.clone();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(x_plus, vec![2, 2], DType::F32, DeviceType::Cpu, false)
                    .unwrap()
                    .matmul(&w)
                    .unwrap()
                    .add(&b)
                    .unwrap()
                    .relu()
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let tm = Tensor::new(x_minus, vec![2, 2], DType::F32, DeviceType::Cpu, false)
                    .unwrap()
                    .matmul(&w)
                    .unwrap()
                    .add(&b)
                    .unwrap()
                    .relu()
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (tp.data[0] - tm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx[idx] - num).abs() < 1e-2,
                "dx[{idx}]: {} vs {num}",
                dx[idx]
            );
        }
        assert_eq!(dw.len(), 4);
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_tensor_activations_and_transpose_backward() {
        AutogradTape::clear();
        let x = Tensor::parameter(
            vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.8],
            vec![2, 3],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let xt = x.transpose_last2().unwrap();
        assert_eq!(xt.shape, vec![3, 2]);
        let y = xt
            .gelu()
            .unwrap()
            .silu()
            .unwrap()
            .sigmoid()
            .unwrap()
            .tanh()
            .unwrap();
        let loss = y.mean(None).unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        assert_eq!(dx.len(), 6);
        assert!(dx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_tensor_broadcast_and_batched_matmul_backward() {
        AutogradTape::clear();
        let a = Tensor::parameter(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let b =
            Tensor::parameter(vec![0.5, -1.0, 2.0], vec![3], DType::F32, DeviceType::Cpu).unwrap();
        let c = a.mul(&b).unwrap().sub(&b).unwrap().scale(0.5).unwrap();
        let loss = c.sum(Some(1)).unwrap().mean(None).unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(grads.get(a.node_id.unwrap()).unwrap().len(), 6);
        assert_eq!(grads.get(b.node_id.unwrap()).unwrap().len(), 3);

        // Batched 3D matmul: [2, 2, 3] @ [2, 3, 2]
        let x3 = Tensor::parameter(
            (0..12).map(|i| (i as f32) * 0.1).collect(),
            vec![2, 2, 3],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w3 = Tensor::parameter(
            (0..12).map(|i| (i as f32) * 0.05 - 0.2).collect(),
            vec![2, 3, 2],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let y3 = x3.matmul(&w3).unwrap();
        assert_eq!(y3.shape, vec![2, 2, 2]);
        let g3 = y3.sum(None).unwrap().backward().unwrap();
        assert_eq!(g3.get(x3.node_id.unwrap()).unwrap().len(), 12);
        assert_eq!(g3.get(w3.node_id.unwrap()).unwrap().len(), 12);
    }
}
