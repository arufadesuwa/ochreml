// OchreML Neural Network Layers Module (*≧ω≦*)
// Linear, Activation (ReLU, Sigmoid, Tanh, GELU, SiLU), Dropout, and Embedding layers.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, CustomBackward, Op};
use crate::nn::dtype::DType;
use crate::nn::gemm_f32::sgemm;
use crate::nn::tensor::{FastRng, Tensor, compute_strides};
use std::sync::Arc;

/// Fully connected linear layer: `y = x @ W^T + b` with `weight` of shape `[out_features, in_features]`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub dtype: DType,
    pub device: DeviceType,
}

impl Linear {
    /// Creates a `Linear` layer with PyTorch-style Kaiming uniform initialization (`bound = 1 / sqrt(in_features)`).
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(OchreError::InvalidParameter(
                "Linear in_features and out_features must be > 0 ( >_< )".to_string(),
            ));
        }
        let bound = 1.0f32 / (in_features as f32).sqrt();
        let weight = Tensor::uniform(
            vec![out_features, in_features],
            -bound,
            bound,
            dtype,
            device.clone(),
            seed,
            true,
        );
        let bias = if use_bias {
            Some(Tensor::uniform(
                vec![out_features],
                -bound,
                bound,
                dtype,
                device.clone(),
                seed.wrapping_add(1),
                true,
            ))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            dtype,
            device,
        })
    }

    /// Creates a `Linear` layer with Kaiming He normal initialization (`std = sqrt(2 / in_features)`).
    pub fn new_kaiming(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(OchreError::InvalidParameter(
                "Linear in_features and out_features must be > 0 ( >_< )".to_string(),
            ));
        }
        let std = (2.0f32 / in_features as f32).sqrt();
        let weight = Tensor::randn_with(
            vec![out_features, in_features],
            0.0,
            std,
            dtype,
            device.clone(),
            seed,
            true,
        );
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_features], dtype, device.clone()).with_requires_grad(true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            dtype,
            device,
        })
    }

    /// Creates a `Linear` layer with Xavier/Glorot uniform initialization.
    pub fn new_xavier(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(OchreError::InvalidParameter(
                "Linear in_features and out_features must be > 0 ( >_< )".to_string(),
            ));
        }
        let bound = (6.0f32 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::uniform(
            vec![out_features, in_features],
            -bound,
            bound,
            dtype,
            device.clone(),
            seed,
            true,
        );
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_features], dtype, device.clone()).with_requires_grad(true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            dtype,
            device,
        })
    }

    /// Wraps pre-existing `weight` `[out_features, in_features]` and optional `bias` `[out_features]`.
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> OchreResult<Self> {
        if weight.shape.len() != 2 {
            return Err(OchreError::InvalidParameter(format!(
                "Linear weight must be 2D [out_features, in_features], got {:?} ( >_< )",
                weight.shape
            )));
        }
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        if let Some(ref b) = bias
            && b.shape != [out_features]
        {
            return Err(OchreError::InvalidParameter(format!(
                "Linear bias must have shape [{out_features}], got {:?} ( >_< )",
                b.shape
            )));
        }
        let dtype = weight.dtype;
        let device = weight.device.clone();
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            dtype,
            device,
        })
    }

    /// Autograd-tracked forward pass on 2D `[batch, in_features]` or ND `[..., in_features]` tensors.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let out = x.matmul_trans(&self.weight, false, true)?;
        match &self.bias {
            Some(b) => out.add(b),
            None => Ok(out),
        }
    }

    /// Zero-allocation forward pass writing `[batch, out_features]` into pre-allocated `out`.
    pub fn forward_raw(&self, x: &[f32], batch: usize, out: &mut [f32]) -> OchreResult<()> {
        let expected_x = batch * self.in_features;
        let expected_out = batch * self.out_features;
        if x.len() != expected_x || out.len() != expected_out {
            return Err(OchreError::DimensionMismatch {
                expected_rows: batch,
                expected_cols: self.out_features,
                actual_rows: x.len(),
                actual_cols: out.len(),
            });
        }
        if matches!(self.device, DeviceType::Cpu) {
            sgemm(
                batch,
                self.out_features,
                self.in_features,
                x,
                &self.weight.data,
                out,
                false,
                true,
                1.0,
                0.0,
            )?;
        } else {
            let res = get_backend(&self.device).batched_matmul_f32(
                1,
                batch,
                self.out_features,
                self.in_features,
                x,
                &self.weight.data,
                false,
                true,
            )?;
            out.copy_from_slice(&res);
        }
        if let Some(ref b) = self.bias {
            let bias = &b.data;
            for r in 0..batch {
                let row = &mut out[r * self.out_features..(r + 1) * self.out_features];
                for c in 0..self.out_features {
                    row[c] += bias[c];
                }
            }
        }
        Ok(())
    }

    /// Zero-allocation backward pass writing `dx` (optional), `dw`, and `db` (optional).
    pub fn backward_raw(
        &self,
        x: &[f32],
        dy: &[f32],
        batch: usize,
        dx: Option<&mut [f32]>,
        dw: &mut [f32],
        db: Option<&mut [f32]>,
    ) -> OchreResult<()> {
        if let Some(dx_buf) = dx {
            if matches!(self.device, DeviceType::Cpu) {
                sgemm(
                    batch,
                    self.in_features,
                    self.out_features,
                    dy,
                    &self.weight.data,
                    dx_buf,
                    false,
                    false,
                    1.0,
                    0.0,
                )?;
            } else {
                let res = get_backend(&self.device).batched_matmul_f32(
                    1,
                    batch,
                    self.in_features,
                    self.out_features,
                    dy,
                    &self.weight.data,
                    false,
                    false,
                )?;
                dx_buf.copy_from_slice(&res);
            }
        }

        if matches!(self.device, DeviceType::Cpu) {
            sgemm(
                self.out_features,
                self.in_features,
                batch,
                dy,
                x,
                dw,
                true,
                false,
                1.0,
                0.0,
            )?;
        } else {
            let res = get_backend(&self.device).batched_matmul_f32(
                1,
                self.out_features,
                self.in_features,
                batch,
                dy,
                x,
                true,
                false,
            )?;
            dw.copy_from_slice(&res);
        }

        if let (Some(db_buf), Some(_)) = (db, &self.bias) {
            db_buf.fill(0.0);
            for r in 0..batch {
                let row = &dy[r * self.out_features..(r + 1) * self.out_features];
                for c in 0..self.out_features {
                    db_buf[c] += row[c];
                }
            }
        }
        Ok(())
    }
}

/// Activation function enum supporting both autograd `Tensor` and zero-alloc raw slice passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
}

impl Activation {
    /// Autograd-compatible tensor forward pass.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::GELU => x.gelu(),
            Activation::SiLU => x.silu(),
        }
    }

    /// Raw forward pass writing into pre-allocated `out` slice.
    pub fn forward_raw(&self, x: &[f32], out: &mut [f32], device: &DeviceType) -> OchreResult<()> {
        if x.len() != out.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: 1,
                expected_cols: x.len(),
                actual_rows: 1,
                actual_cols: out.len(),
            });
        }
        match self {
            Activation::ReLU => {
                for (o, &v) in out.iter_mut().zip(x.iter()) {
                    *o = v.max(0.0);
                }
            }
            Activation::Sigmoid => {
                for (o, &v) in out.iter_mut().zip(x.iter()) {
                    *o = 1.0 / (1.0 + (-v).exp());
                }
            }
            Activation::Tanh => {
                for (o, &v) in out.iter_mut().zip(x.iter()) {
                    *o = v.tanh();
                }
            }
            Activation::GELU => {
                let res = get_backend(device).gelu_f32(x);
                out.copy_from_slice(&res);
            }
            Activation::SiLU => {
                let res = get_backend(device).silu_f32(x);
                out.copy_from_slice(&res);
            }
        }
        Ok(())
    }

    /// Raw backward pass writing `dx` given `x`, `y` (layer output), and `dy`.
    pub fn backward_raw(
        &self,
        x: &[f32],
        y: &[f32],
        dy: &[f32],
        dx: &mut [f32],
        device: &DeviceType,
    ) -> OchreResult<()> {
        if x.len() != dy.len() || dx.len() != dy.len() || y.len() != dy.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: 1,
                expected_cols: dy.len(),
                actual_rows: 1,
                actual_cols: dx.len(),
            });
        }
        match self {
            Activation::ReLU => {
                for ((dst, &g), &xv) in dx.iter_mut().zip(dy.iter()).zip(x.iter()) {
                    *dst = if xv > 0.0 { g } else { 0.0 };
                }
            }
            Activation::Sigmoid => {
                for ((dst, &g), &yv) in dx.iter_mut().zip(dy.iter()).zip(y.iter()) {
                    *dst = g * yv * (1.0 - yv);
                }
            }
            Activation::Tanh => {
                for ((dst, &g), &yv) in dx.iter_mut().zip(dy.iter()).zip(y.iter()) {
                    *dst = g * (1.0 - yv * yv);
                }
            }
            Activation::GELU => {
                let res = get_backend(device).gelu_bwd_f32(dy, x);
                dx.copy_from_slice(&res);
            }
            Activation::SiLU => {
                let res = get_backend(device).silu_bwd_f32(dy, x);
                dx.copy_from_slice(&res);
            }
        }
        Ok(())
    }
}

/// Unit struct for ReLU activation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU;
impl ReLU {
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        x.relu()
    }
}

/// Unit struct for Sigmoid activation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;
impl Sigmoid {
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        x.sigmoid()
    }
}

/// Unit struct for Tanh activation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tanh;
impl Tanh {
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        x.tanh()
    }
}

/// Unit struct for GELU activation.
#[derive(Clone, Copy, Debug, Default)]
pub struct GELU;
impl GELU {
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        x.gelu()
    }
}

/// Unit struct for SiLU activation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiLU;
impl SiLU {
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        x.silu()
    }
}

/// Inverted dropout layer (`scale = 1 / (1 - p)` during training).
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
    pub training: bool,
    pub seed: u64,
}

impl Dropout {
    /// Creates a `Dropout` layer with drop probability `p` in `[0.0, 1.0)`.
    pub fn new(p: f32, seed: u64) -> OchreResult<Self> {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(OchreError::InvalidParameter(format!(
                "Dropout probability p={p} must be in [0.0, 1.0) ( >_< )"
            )));
        }
        Ok(Self {
            p,
            training: true,
            seed,
        })
    }

    /// Autograd-compatible inverted dropout forward pass.
    pub fn forward(&mut self, x: &Tensor) -> OchreResult<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }
        let scale = 1.0f32 / (1.0f32 - self.p);
        let mut rng = FastRng::new(self.seed);
        self.seed = rng.next_u64();

        let mut mask = Vec::with_capacity(x.data.len());
        let mut out_data = Vec::with_capacity(x.data.len());
        for &v in &x.data {
            let m = if rng.next_f32() < self.p { 0.0 } else { scale };
            mask.push(m);
            out_data.push(v * m);
        }

        let grad_active = AutogradTape::is_grad_enabled() && x.requires_grad;
        let node_id = if grad_active {
            let in_id = x.node_id;
            let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
                if let Some(id) = in_id {
                    let dx: Vec<f32> = d_out
                        .iter()
                        .zip(mask.iter())
                        .map(|(&g, &m)| g * m)
                        .collect();
                    Ok(vec![(id, dx)])
                } else {
                    Ok(Vec::new())
                }
            }));
            Some(AutogradTape::record(Op::Custom {
                inputs: vec![in_id],
                backward: bwd,
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: x.shape.clone(),
            strides: x.strides.clone(),
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Zero-allocation raw forward pass writing outputs and scaled binary mask.
    pub fn forward_raw(
        &mut self,
        x: &[f32],
        out: &mut [f32],
        mask: &mut [f32],
        training: bool,
    ) -> OchreResult<()> {
        if x.len() != out.len() || x.len() != mask.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: 1,
                expected_cols: x.len(),
                actual_rows: 1,
                actual_cols: out.len(),
            });
        }
        if !training || self.p == 0.0 {
            out.copy_from_slice(x);
            mask.fill(1.0);
            return Ok(());
        }
        let scale = 1.0f32 / (1.0f32 - self.p);
        let mut rng = FastRng::new(self.seed);
        self.seed = rng.next_u64();
        for ((dst, m_dst), &xv) in out.iter_mut().zip(mask.iter_mut()).zip(x.iter()) {
            let m = if rng.next_f32() < self.p { 0.0 } else { scale };
            *m_dst = m;
            *dst = xv * m;
        }
        Ok(())
    }

    /// Zero-allocation raw backward pass using saved scaled `mask`.
    pub fn backward_raw(&self, dy: &[f32], mask: &[f32], dx: &mut [f32]) -> OchreResult<()> {
        if dy.len() != mask.len() || dy.len() != dx.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: 1,
                expected_cols: dy.len(),
                actual_rows: 1,
                actual_cols: dx.len(),
            });
        }
        for ((dst, &g), &m) in dx.iter_mut().zip(dy.iter()).zip(mask.iter()) {
            *dst = g * m;
        }
        Ok(())
    }
}

/// Token/categorical lookup table with gather forward and scatter-add backward.
#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// Creates an `Embedding` table initialized with `N(0, 1)`.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if num_embeddings == 0 || embedding_dim == 0 {
            return Err(OchreError::InvalidParameter(
                "Embedding num_embeddings and embedding_dim must be > 0 ( >_< )".to_string(),
            ));
        }
        let weight = Tensor::randn_with(
            vec![num_embeddings, embedding_dim],
            0.0,
            1.0,
            dtype,
            device,
            seed,
            true,
        );
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Wraps an existing `[num_embeddings, embedding_dim]` parameter tensor.
    pub fn from_weight(weight: Tensor) -> OchreResult<Self> {
        if weight.shape.len() != 2 {
            return Err(OchreError::InvalidParameter(format!(
                "Embedding weight must be 2D [num_embeddings, embedding_dim], got {:?} ( >_< )",
                weight.shape
            )));
        }
        let num_embeddings = weight.shape[0];
        let embedding_dim = weight.shape[1];
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Gathers embeddings for `indices` into shape `[indices.len(), embedding_dim]`.
    pub fn forward(&self, indices: &[usize]) -> OchreResult<Tensor> {
        let mut out_data = vec![0.0f32; indices.len() * self.embedding_dim];
        self.forward_raw(indices, &mut out_data)?;
        let out_shape = vec![indices.len(), self.embedding_dim];
        let grad_active = AutogradTape::is_grad_enabled() && self.weight.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::Embedding {
                weight_id: self.weight.node_id,
                indices: indices.to_vec(),
                num_embeddings: self.num_embeddings,
                embedding_dim: self.embedding_dim,
            }))
        } else {
            None
        };
        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.weight.dtype,
            device: self.weight.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Gathers embeddings for `[batch, seq_len]` token indices into `[batch, seq_len, embedding_dim]`.
    pub fn forward_2d(
        &self,
        indices: &[usize],
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<Tensor> {
        if indices.len() != batch * seq_len {
            return Err(OchreError::InvalidParameter(format!(
                "Embedding indices length {} does not match {batch}x{seq_len} ( >_< )",
                indices.len()
            )));
        }
        let flat = self.forward(indices)?;
        flat.reshape(vec![batch, seq_len, self.embedding_dim])
    }

    /// Zero-allocation gather forward into pre-allocated `out` (`[indices.len() * embedding_dim]`).
    pub fn forward_raw(&self, indices: &[usize], out: &mut [f32]) -> OchreResult<()> {
        let expected = indices.len() * self.embedding_dim;
        if out.len() != expected {
            return Err(OchreError::DimensionMismatch {
                expected_rows: indices.len(),
                expected_cols: self.embedding_dim,
                actual_rows: out.len(),
                actual_cols: 1,
            });
        }
        for (pos, &idx) in indices.iter().enumerate() {
            if idx >= self.num_embeddings {
                return Err(OchreError::InvalidParameter(format!(
                    "Embedding index {idx} out of bounds for num_embeddings={} ( >_< )",
                    self.num_embeddings
                )));
            }
            let src = &self.weight.data[idx * self.embedding_dim..(idx + 1) * self.embedding_dim];
            let dst = &mut out[pos * self.embedding_dim..(pos + 1) * self.embedding_dim];
            dst.copy_from_slice(src);
        }
        Ok(())
    }

    /// Zero-allocation scatter-add backward into `dw` (`[num_embeddings * embedding_dim]`).
    pub fn backward_raw(&self, indices: &[usize], dy: &[f32], dw: &mut [f32]) -> OchreResult<()> {
        if dy.len() != indices.len() * self.embedding_dim || dw.len() != self.weight.data.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: indices.len(),
                expected_cols: self.embedding_dim,
                actual_rows: dy.len(),
                actual_cols: dw.len(),
            });
        }
        for (pos, &idx) in indices.iter().enumerate() {
            if idx >= self.num_embeddings {
                return Err(OchreError::InvalidParameter(format!(
                    "Embedding index {idx} out of bounds for num_embeddings={} ( >_< )",
                    self.num_embeddings
                )));
            }
            let src = &dy[pos * self.embedding_dim..(pos + 1) * self.embedding_dim];
            let dst = &mut dw[idx * self.embedding_dim..(idx + 1) * self.embedding_dim];
            for (d, &g) in dst.iter_mut().zip(src.iter()) {
                *d += g;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_and_backward() {
        AutogradTape::clear();
        let lin = Linear::new(3, 2, true, DType::F32, DeviceType::Cpu, 7).unwrap();
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0],
            vec![2, 3],
            DType::F32,
            DeviceType::Cpu,
            true,
        )
        .unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 2]);

        let mut raw_y = vec![0.0f32; 4];
        lin.forward_raw(&x.data, 2, &mut raw_y).unwrap();
        for (&a, &b) in y.data.iter().zip(raw_y.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let loss = y.sum(None).unwrap();
        let grads = loss.backward().unwrap();
        let dw = grads.get(lin.weight.node_id.unwrap()).unwrap();
        let db = grads
            .get(lin.bias.as_ref().unwrap().node_id.unwrap())
            .unwrap();
        assert_eq!(dw.len(), 6);
        assert_eq!(db, &[2.0, 2.0]);
    }

    #[test]
    fn test_dropout_inverted_scale_and_eval() {
        AutogradTape::clear();
        let mut drop = Dropout::new(0.5, 123).unwrap();
        let x = Tensor::ones(vec![100], DType::F32, DeviceType::Cpu).with_requires_grad(true);
        let y = drop.forward(&x).unwrap();
        let nonzero = y.data.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero > 20 && nonzero < 80);
        for &v in &y.data {
            assert!(v == 0.0 || (v - 2.0).abs() < 1e-6);
        }
        let grads = y.sum(None).unwrap().backward().unwrap();
        assert_eq!(grads.get(x.node_id.unwrap()).unwrap(), y.data.as_slice());

        drop.training = false;
        let eval_y = drop.forward(&x).unwrap();
        assert_eq!(eval_y.data, x.data);
    }

    #[test]
    fn test_embedding_gather_and_scatter_add_backward() {
        AutogradTape::clear();
        let emb = Embedding::new(5, 3, DType::F32, DeviceType::Cpu, 99).unwrap();
        let out = emb.forward(&[1, 3, 1]).unwrap();
        assert_eq!(out.shape, vec![3, 3]);
        assert_eq!(&out.data[0..3], &out.data[6..9]);

        let grads = out.sum(None).unwrap().backward().unwrap();
        let dw = grads.get(emb.weight.node_id.unwrap()).unwrap();
        assert_eq!(&dw[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&dw[3..6], &[2.0, 2.0, 2.0]);
        assert_eq!(&dw[9..12], &[1.0, 1.0, 1.0]);
    }
}
