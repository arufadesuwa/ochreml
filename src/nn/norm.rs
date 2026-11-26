// OchreML Normalization Layers Module (＾▽＾)
// LayerNorm and RMSNorm backed by N8 Backend::row_layernorm_f32 / row_rmsnorm_f32 with autograd.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::tensor::{Tensor, compute_strides};

#[inline]
fn maybe_quantize(data: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => data,
        DType::BF16 | DType::F16 => TensorStorage::from_f32_slice(&data, dtype).to_f32_vec(),
    }
}

/// Layer Normalization over the last dimension (`normalized_shape`).
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub normalized_shape: usize,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub gamma: Option<Tensor>,
    pub beta: Option<Tensor>,
    pub dtype: DType,
    pub device: DeviceType,
}

impl LayerNorm {
    /// Creates a `LayerNorm` layer over `normalized_shape` features.
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        elementwise_affine: bool,
        dtype: DType,
        device: DeviceType,
    ) -> OchreResult<Self> {
        if normalized_shape == 0 {
            return Err(OchreError::InvalidParameter(
                "LayerNorm normalized_shape must be > 0 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "LayerNorm eps={eps} must be finite and non-negative ( >_< )"
            )));
        }
        let (gamma, beta) = if elementwise_affine {
            (
                Some(
                    Tensor::ones(vec![normalized_shape], dtype, device.clone())
                        .with_requires_grad(true),
                ),
                Some(
                    Tensor::zeros(vec![normalized_shape], dtype, device.clone())
                        .with_requires_grad(true),
                ),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            gamma,
            beta,
            dtype,
            device,
        })
    }

    /// Constructs a `LayerNorm` from pre-existing optional `gamma` and `beta` tensors.
    pub fn from_weights(
        normalized_shape: usize,
        eps: f32,
        gamma: Option<Tensor>,
        beta: Option<Tensor>,
        dtype: DType,
        device: DeviceType,
    ) -> OchreResult<Self> {
        if normalized_shape == 0 {
            return Err(OchreError::InvalidParameter(
                "LayerNorm normalized_shape must be > 0 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "LayerNorm eps={eps} must be finite and non-negative ( >_< )"
            )));
        }
        if let Some(ref g) = gamma
            && g.shape != [normalized_shape]
        {
            return Err(OchreError::InvalidParameter(format!(
                "LayerNorm gamma shape {:?} does not match [{normalized_shape}] ( >_< )",
                g.shape
            )));
        }
        if let Some(ref b) = beta
            && b.shape != [normalized_shape]
        {
            return Err(OchreError::InvalidParameter(format!(
                "LayerNorm beta shape {:?} does not match [{normalized_shape}] ( >_< )",
                b.shape
            )));
        }
        let elementwise_affine = gamma.is_some() || beta.is_some();
        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            gamma,
            beta,
            dtype,
            device,
        })
    }

    /// Raw forward pass over `[rows, normalized_shape]`, returning `(out, mean, rstd)`.
    pub fn forward_raw(
        &self,
        x: &[f32],
        rows: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        get_backend(&self.device).row_layernorm_f32(
            x,
            self.gamma.as_ref().map(|g| g.data.as_slice()),
            self.beta.as_ref().map(|b| b.data.as_slice()),
            rows,
            self.normalized_shape,
            self.eps,
        )
    }

    /// Raw backward pass over `[rows, normalized_shape]`, returning `(dx, dgamma, dbeta)`.
    pub fn backward_raw(
        &self,
        dy: &[f32],
        x: &[f32],
        mean: &[f32],
        rstd: &[f32],
        rows: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        get_backend(&self.device).row_layernorm_bwd_f32(
            dy,
            x,
            self.gamma.as_ref().map(|g| g.data.as_slice()),
            mean,
            rstd,
            rows,
            self.normalized_shape,
        )
    }

    /// Autograd-tracked forward pass normalizing over the last axis of `x`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let Some(&last_dim) = x.shape.last() else {
            return Err(OchreError::InvalidParameter(
                "LayerNorm requires input rank >= 1 ( >_< )".to_string(),
            ));
        };
        if last_dim != self.normalized_shape {
            return Err(OchreError::InvalidParameter(format!(
                "LayerNorm expected last dim {}, got {last_dim} ( >_< )",
                self.normalized_shape
            )));
        }
        if x.data.is_empty() {
            return Err(OchreError::EmptyData);
        }
        let cols = self.normalized_shape;
        let rows = x.data.len() / cols;

        let g_slice = self.gamma.as_ref().map(|g| g.data.as_slice());
        let b_slice = self.beta.as_ref().map(|b| b.data.as_slice());
        let (raw_out, mean, rstd) = get_backend(&x.device)
            .row_layernorm_f32(&x.data, g_slice, b_slice, rows, cols, self.eps)?;
        let out_data = maybe_quantize(raw_out, x.dtype);

        let input_id = if x.requires_grad { x.node_id } else { None };
        let gamma_id = self
            .gamma
            .as_ref()
            .and_then(|g| if g.requires_grad { g.node_id } else { None });
        let beta_id = self
            .beta
            .as_ref()
            .and_then(|b| if b.requires_grad { b.node_id } else { None });
        let grad_active = AutogradTape::is_grad_enabled()
            && (input_id.is_some() || gamma_id.is_some() || beta_id.is_some());

        let node_id = if grad_active {
            Some(AutogradTape::record(Op::LayerNorm {
                input_id,
                gamma_id,
                beta_id,
                x: x.data.clone(),
                gamma: self.gamma.as_ref().map(|g| g.data.clone()),
                mean,
                rstd,
                rows,
                cols,
                device: x.device.clone(),
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: x.shape.clone(),
            strides: compute_strides(&x.shape),
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Returns references to trainable affine parameters (`gamma`, `beta`).
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref g) = self.gamma {
            params.push(g);
        }
        if let Some(ref b) = self.beta {
            params.push(b);
        }
        params
    }

    /// Returns mutable references to trainable affine parameters (`gamma`, `beta`).
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut g) = self.gamma {
            params.push(g);
        }
        if let Some(ref mut b) = self.beta {
            params.push(b);
        }
        params
    }
}

/// Root Mean Square Layer Normalization (`x / rms(x) * gamma`).
#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub normalized_shape: usize,
    pub eps: f32,
    pub gamma: Tensor,
    pub dtype: DType,
    pub device: DeviceType,
}

impl RMSNorm {
    /// Creates an `RMSNorm` layer with `gamma` initialized to ones.
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        dtype: DType,
        device: DeviceType,
    ) -> OchreResult<Self> {
        if normalized_shape == 0 {
            return Err(OchreError::InvalidParameter(
                "RMSNorm normalized_shape must be > 0 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "RMSNorm eps={eps} must be finite and non-negative ( >_< )"
            )));
        }
        let gamma =
            Tensor::ones(vec![normalized_shape], dtype, device.clone()).with_requires_grad(true);
        Ok(Self {
            normalized_shape,
            eps,
            gamma,
            dtype,
            device,
        })
    }

    /// Constructs an `RMSNorm` from a pre-existing `gamma` tensor of shape `[normalized_shape]`.
    pub fn from_weight(normalized_shape: usize, eps: f32, gamma: Tensor) -> OchreResult<Self> {
        if normalized_shape == 0 {
            return Err(OchreError::InvalidParameter(
                "RMSNorm normalized_shape must be > 0 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "RMSNorm eps={eps} must be finite and non-negative ( >_< )"
            )));
        }
        if gamma.shape != [normalized_shape] {
            return Err(OchreError::InvalidParameter(format!(
                "RMSNorm gamma shape {:?} does not match [{normalized_shape}] ( >_< )",
                gamma.shape
            )));
        }
        let dtype = gamma.dtype;
        let device = gamma.device.clone();
        Ok(Self {
            normalized_shape,
            eps,
            gamma,
            dtype,
            device,
        })
    }

    /// Raw forward pass over `[rows, normalized_shape]`, returning `(out, rstd)`.
    pub fn forward_raw(&self, x: &[f32], rows: usize) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        get_backend(&self.device).row_rmsnorm_f32(
            x,
            Some(self.gamma.data.as_slice()),
            rows,
            self.normalized_shape,
            self.eps,
        )
    }

    /// Raw backward pass over `[rows, normalized_shape]`, returning `(dx, dgamma)`.
    pub fn backward_raw(
        &self,
        dy: &[f32],
        x: &[f32],
        rstd: &[f32],
        rows: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        get_backend(&self.device).row_rmsnorm_bwd_f32(
            dy,
            x,
            Some(self.gamma.data.as_slice()),
            rstd,
            rows,
            self.normalized_shape,
        )
    }

    /// Autograd-tracked forward pass normalizing over the last axis of `x`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let Some(&last_dim) = x.shape.last() else {
            return Err(OchreError::InvalidParameter(
                "RMSNorm requires input rank >= 1 ( >_< )".to_string(),
            ));
        };
        if last_dim != self.normalized_shape {
            return Err(OchreError::InvalidParameter(format!(
                "RMSNorm expected last dim {}, got {last_dim} ( >_< )",
                self.normalized_shape
            )));
        }
        if x.data.is_empty() {
            return Err(OchreError::EmptyData);
        }
        let cols = self.normalized_shape;
        let rows = x.data.len() / cols;

        let (raw_out, rstd) = get_backend(&x.device).row_rmsnorm_f32(
            &x.data,
            Some(self.gamma.data.as_slice()),
            rows,
            cols,
            self.eps,
        )?;
        let out_data = maybe_quantize(raw_out, x.dtype);

        let input_id = if x.requires_grad { x.node_id } else { None };
        let gamma_id = if self.gamma.requires_grad {
            self.gamma.node_id
        } else {
            None
        };
        let grad_active =
            AutogradTape::is_grad_enabled() && (input_id.is_some() || gamma_id.is_some());

        let node_id = if grad_active {
            Some(AutogradTape::record(Op::RMSNorm {
                input_id,
                gamma_id,
                x: x.data.clone(),
                gamma: Some(self.gamma.data.clone()),
                rstd,
                rows,
                cols,
                device: x.device.clone(),
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: x.shape.clone(),
            strides: compute_strides(&x.shape),
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }

    /// Returns a reference to the scale parameter `gamma`.
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma]
    }

    /// Returns a mutable reference to the scale parameter `gamma`.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::with_no_grad;

    #[test]
    fn test_layernorm_finite_diff_grad_check() {
        AutogradTape::clear();
        let mut ln = LayerNorm::new(4, 1e-5, true, DType::F32, DeviceType::Cpu).unwrap();
        ln.gamma = Some(
            Tensor::parameter(
                vec![1.1, 0.9, 1.3, 0.7],
                vec![4],
                DType::F32,
                DeviceType::Cpu,
            )
            .unwrap(),
        );
        ln.beta = Some(
            Tensor::parameter(
                vec![0.1, -0.2, 0.3, -0.05],
                vec![4],
                DType::F32,
                DeviceType::Cpu,
            )
            .unwrap(),
        );

        let x = Tensor::parameter(
            vec![0.5, -1.2, 0.8, 2.0, -0.3, 1.4, -0.7, 0.2],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::new(
            vec![0.4, -0.6, 1.1, 0.3, -0.5, 0.9, 0.2, -0.8],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let y = ln.forward(&x).unwrap();
        let (raw_y, _, _) = ln.forward_raw(&x.data, 2).unwrap();
        for (&a, &b) in y.data.iter().zip(raw_y.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dg = grads
            .get(ln.gamma.as_ref().unwrap().node_id.unwrap())
            .unwrap();
        let db = grads
            .get(ln.beta.as_ref().unwrap().node_id.unwrap())
            .unwrap();

        let eps = 1e-3f32;
        for i in 0..x.numel() {
            let mut xp = x.data.clone();
            let mut xm = x.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(xp, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm = Tensor::new(xm, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = ln.forward(&tp).unwrap().mul(&w).unwrap().sum(None).unwrap();
                let lm = ln.forward(&tm).unwrap().mul(&w).unwrap().sum(None).unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx[i] - num).abs() < 5e-3,
                "LayerNorm dx[{i}] mismatch: {} vs {num}",
                dx[i]
            );
        }

        for j in 0..4 {
            let g_orig = ln.gamma.as_ref().unwrap().data.clone();
            let mut gp = g_orig.clone();
            let mut gm = g_orig.clone();
            gp[j] += eps;
            gm[j] -= eps;
            let num_g = with_no_grad(|| {
                let mut ln_p = ln.clone();
                ln_p.gamma =
                    Some(Tensor::new(gp, vec![4], DType::F32, DeviceType::Cpu, false).unwrap());
                let mut ln_m = ln.clone();
                ln_m.gamma =
                    Some(Tensor::new(gm, vec![4], DType::F32, DeviceType::Cpu, false).unwrap());
                let lp = ln_p
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = ln_m
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dg[j] - num_g).abs() < 5e-3,
                "LayerNorm dgamma[{j}] mismatch: {} vs {num_g}",
                dg[j]
            );
            assert!((db[j] - (w.data[j] + w.data[j + 4])).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rmsnorm_finite_diff_grad_check() {
        AutogradTape::clear();
        let mut rn = RMSNorm::new(4, 1e-5, DType::F32, DeviceType::Cpu).unwrap();
        rn.gamma = Tensor::parameter(
            vec![1.2, 0.8, 1.5, 0.6],
            vec![4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        let x = Tensor::parameter(
            vec![0.4, -0.9, 1.3, -0.2, 0.7, 0.1, -1.1, 0.5],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::new(
            vec![0.3, 0.7, -0.4, 1.2, -0.8, 0.5, 0.9, -0.2],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let y = rn.forward(&x).unwrap();
        let (raw_y, _) = rn.forward_raw(&x.data, 2).unwrap();
        for (&a, &b) in y.data.iter().zip(raw_y.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dg = grads.get(rn.gamma.node_id.unwrap()).unwrap();

        let eps = 1e-3f32;
        for i in 0..x.numel() {
            let mut xp = x.data.clone();
            let mut xm = x.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(xp, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm = Tensor::new(xm, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = rn.forward(&tp).unwrap().mul(&w).unwrap().sum(None).unwrap();
                let lm = rn.forward(&tm).unwrap().mul(&w).unwrap().sum(None).unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx[i] - num).abs() < 5e-3,
                "RMSNorm dx[{i}] mismatch: {} vs {num}",
                dx[i]
            );
        }

        for j in 0..4 {
            let mut gp = rn.gamma.data.clone();
            let mut gm = rn.gamma.data.clone();
            gp[j] += eps;
            gm[j] -= eps;
            let num_g = with_no_grad(|| {
                let mut rn_p = rn.clone();
                rn_p.gamma = Tensor::new(gp, vec![4], DType::F32, DeviceType::Cpu, false).unwrap();
                let mut rn_m = rn.clone();
                rn_m.gamma = Tensor::new(gm, vec![4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = rn_p
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = rn_m
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dg[j] - num_g).abs() < 5e-3,
                "RMSNorm dgamma[{j}] mismatch: {} vs {num_g}",
                dg[j]
            );
        }
    }
}
