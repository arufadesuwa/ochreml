// OchreML Optimizers & LR Schedulers Module (＾▽＾)
// Implements SGD, Adam, fused AdamW (with f32 master weights for BF16/F16), grad clipping, and schedulers.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::GradStore;
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::tensor::Tensor;

fn quantize_inplace(data: &mut [f32], dtype: DType) {
    if dtype != DType::F32 {
        let q = TensorStorage::from_f32_slice(data, dtype).to_f32_vec();
        data.copy_from_slice(&q);
    }
}

/// Stochastic Gradient Descent optimizer with momentum, Nesterov, and weight decay.
#[derive(Clone, Debug)]
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub weight_decay: f32,
    pub dampening: f32,
    velocities: Vec<Vec<f32>>,
    master_weights: Vec<Option<Vec<f32>>>,
}

impl SGD {
    /// Creates an `SGD` optimizer.
    pub fn new(lr: f32, momentum: f32, nesterov: bool, weight_decay: f32) -> OchreResult<Self> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "SGD lr={lr} must be non-negative and finite ( >_< )"
            )));
        }
        if !momentum.is_finite() || momentum < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "SGD momentum={momentum} must be non-negative and finite ( >_< )"
            )));
        }
        if nesterov && momentum <= 0.0 {
            return Err(OchreError::InvalidParameter(
                "SGD Nesterov momentum requires momentum > 0 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            lr,
            momentum,
            nesterov,
            weight_decay,
            dampening: 0.0,
            velocities: Vec::new(),
            master_weights: Vec::new(),
        })
    }

    fn ensure_state(&mut self, n_params: usize, lengths: &[usize], params: &[&Tensor]) {
        if self.velocities.len() < n_params {
            self.velocities.resize_with(n_params, Vec::new);
            self.master_weights.resize_with(n_params, || None);
        }
        for i in 0..n_params {
            if self.momentum > 0.0 && self.velocities[i].len() != lengths[i] {
                self.velocities[i] = vec![0.0f32; lengths[i]];
            }
            if params[i].dtype != DType::F32 && self.master_weights[i].is_none() {
                self.master_weights[i] = Some(params[i].data.clone());
            }
        }
    }

    /// Updates `params` in-place using gradients stored in `grads`.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &GradStore) -> OchreResult<()> {
        let lengths: Vec<usize> = params.iter().map(|p| p.data.len()).collect();
        let immut: Vec<&Tensor> = params.iter().map(|p| &**p).collect();
        self.ensure_state(params.len(), &lengths, &immut);

        for (i, param) in params.iter_mut().enumerate() {
            let Some(node_id) = param.node_id else {
                continue;
            };
            let Some(grad) = grads.get(node_id) else {
                continue;
            };
            if grad.len() != param.data.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: param.data.len(),
                    actual_rows: 1,
                    actual_cols: grad.len(),
                });
            }

            let target_slice: &mut [f32] = if let Some(ref mut master) = self.master_weights[i] {
                master.as_mut_slice()
            } else {
                param.data.as_mut_slice()
            };

            sgd_update_slice(
                target_slice,
                grad,
                if self.momentum > 0.0 {
                    Some(self.velocities[i].as_mut_slice())
                } else {
                    None
                },
                self.lr,
                self.momentum,
                self.dampening,
                self.nesterov,
                self.weight_decay,
            );

            if let Some(ref master) = self.master_weights[i] {
                param.data.copy_from_slice(master);
                quantize_inplace(&mut param.data, param.dtype);
            }
        }
        Ok(())
    }

    /// Zero-allocation slice update for `Sequential::fit`.
    pub fn step_slices(&mut self, params: &mut [&mut [f32]], grads: &[&[f32]]) -> OchreResult<()> {
        if params.len() != grads.len() {
            return Err(OchreError::LengthMismatch {
                features: params.len(),
                targets: grads.len(),
            });
        }
        if self.velocities.len() < params.len() {
            self.velocities.resize_with(params.len(), Vec::new);
        }
        for (i, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if p.len() != g.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: p.len(),
                    actual_rows: 1,
                    actual_cols: g.len(),
                });
            }
            if self.momentum > 0.0 && self.velocities[i].len() != p.len() {
                self.velocities[i] = vec![0.0f32; p.len()];
            }
            sgd_update_slice(
                p,
                g,
                if self.momentum > 0.0 {
                    Some(self.velocities[i].as_mut_slice())
                } else {
                    None
                },
                self.lr,
                self.momentum,
                self.dampening,
                self.nesterov,
                self.weight_decay,
            );
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn sgd_update_slice(
    param: &mut [f32],
    grad: &[f32],
    mut velocity: Option<&mut [f32]>,
    lr: f32,
    momentum: f32,
    dampening: f32,
    nesterov: bool,
    weight_decay: f32,
) {
    for j in 0..param.len() {
        let mut g = grad[j];
        if weight_decay != 0.0 {
            g += weight_decay * param[j];
        }
        let step_dir = if let Some(ref mut buf) = velocity {
            let v = buf[j] * momentum + (1.0 - dampening) * g;
            buf[j] = v;
            if nesterov { g + momentum * v } else { v }
        } else {
            g
        };
        param[j] -= lr * step_dir;
    }
}

/// Adam optimizer (coupled L2 weight decay) with master `f32` weights for `BF16`/`F16`.
#[derive(Clone, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    master_weights: Vec<Option<Vec<f32>>>,
}

impl Adam {
    /// Creates an `Adam` optimizer.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> OchreResult<Self> {
        validate_adam_params(lr, beta1, beta2, eps, weight_decay)?;
        Ok(Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            master_weights: Vec::new(),
        })
    }

    /// Creates an `Adam` optimizer with standard PyTorch defaults (`beta1=0.9, beta2=0.999, eps=1e-8`).
    pub fn with_lr(lr: f32, weight_decay: f32) -> OchreResult<Self> {
        Self::new(lr, 0.9, 0.999, 1e-8, weight_decay)
    }

    /// Updates `params` in-place using gradients stored in `grads`.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &GradStore) -> OchreResult<()> {
        self.step_count = self.step_count.saturating_add(1);
        if self.m.len() < params.len() {
            self.m.resize_with(params.len(), Vec::new);
            self.v.resize_with(params.len(), Vec::new);
            self.master_weights.resize_with(params.len(), || None);
        }
        let bc1 = 1.0f32 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0f32 - self.beta2.powi(self.step_count as i32);

        for (i, param) in params.iter_mut().enumerate() {
            let Some(node_id) = param.node_id else {
                continue;
            };
            let Some(grad) = grads.get(node_id) else {
                continue;
            };
            if grad.len() != param.data.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: param.data.len(),
                    actual_rows: 1,
                    actual_cols: grad.len(),
                });
            }
            if self.m[i].len() != param.data.len() {
                self.m[i] = vec![0.0f32; param.data.len()];
                self.v[i] = vec![0.0f32; param.data.len()];
            }
            if param.dtype != DType::F32 && self.master_weights[i].is_none() {
                self.master_weights[i] = Some(param.data.clone());
            }

            let target: &mut [f32] = if let Some(ref mut master) = self.master_weights[i] {
                master.as_mut_slice()
            } else {
                param.data.as_mut_slice()
            };
            adam_update_slice(
                target,
                grad,
                &mut self.m[i],
                &mut self.v[i],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1,
                bc2,
            );
            if let Some(ref master) = self.master_weights[i] {
                param.data.copy_from_slice(master);
                quantize_inplace(&mut param.data, param.dtype);
            }
        }
        Ok(())
    }

    /// Zero-allocation slice update for `Sequential::fit`.
    pub fn step_slices(&mut self, params: &mut [&mut [f32]], grads: &[&[f32]]) -> OchreResult<()> {
        if params.len() != grads.len() {
            return Err(OchreError::LengthMismatch {
                features: params.len(),
                targets: grads.len(),
            });
        }
        self.step_count = self.step_count.saturating_add(1);
        if self.m.len() < params.len() {
            self.m.resize_with(params.len(), Vec::new);
            self.v.resize_with(params.len(), Vec::new);
        }
        let bc1 = 1.0f32 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0f32 - self.beta2.powi(self.step_count as i32);

        for (i, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if p.len() != g.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: p.len(),
                    actual_rows: 1,
                    actual_cols: g.len(),
                });
            }
            if self.m[i].len() != p.len() {
                self.m[i] = vec![0.0f32; p.len()];
                self.v[i] = vec![0.0f32; p.len()];
            }
            adam_update_slice(
                p,
                g,
                &mut self.m[i],
                &mut self.v[i],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1,
                bc2,
            );
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_update_slice(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32,
    bc2: f32,
) {
    let one_m_b1 = 1.0 - beta1;
    let one_m_b2 = 1.0 - beta2;
    for j in 0..param.len() {
        let mut g = grad[j];
        if weight_decay != 0.0 {
            g += weight_decay * param[j];
        }
        m[j] = beta1 * m[j] + one_m_b1 * g;
        v[j] = beta2 * v[j] + one_m_b2 * g * g;
        let m_hat = m[j] / bc1;
        let v_hat = v[j] / bc2;
        param[j] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Fused AdamW optimizer calling `Backend::adamw_step_f32` (N9) with `f32` master weights for `BF16`/`F16`.
#[derive(Clone, Debug)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub device: DeviceType,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    master_weights: Vec<Option<Vec<f32>>>,
}

impl AdamW {
    /// Creates an `AdamW` optimizer targeting `device`.
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        device: DeviceType,
    ) -> OchreResult<Self> {
        validate_adam_params(lr, beta1, beta2, eps, weight_decay)?;
        Ok(Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            device,
            m: Vec::new(),
            v: Vec::new(),
            master_weights: Vec::new(),
        })
    }

    /// Creates an `AdamW` optimizer with standard defaults (`beta1=0.9, beta2=0.999, eps=1e-8`).
    pub fn with_lr(lr: f32, weight_decay: f32, device: DeviceType) -> OchreResult<Self> {
        Self::new(lr, 0.9, 0.999, 1e-8, weight_decay, device)
    }

    /// Returns the stored `f32` master weight slice for parameter index `idx`, if mixed-precision is active.
    pub fn master_weight(&self, idx: usize) -> Option<&[f32]> {
        self.master_weights.get(idx).and_then(|o| o.as_deref())
    }

    /// Updates `params` using `Backend::adamw_step_f32` and master `f32` weights for `BF16`/`F16`.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &GradStore) -> OchreResult<()> {
        self.step_count = self.step_count.saturating_add(1);
        if self.m.len() < params.len() {
            self.m.resize_with(params.len(), Vec::new);
            self.v.resize_with(params.len(), Vec::new);
            self.master_weights.resize_with(params.len(), || None);
        }
        let backend = get_backend(&self.device);

        for (i, param) in params.iter_mut().enumerate() {
            let Some(node_id) = param.node_id else {
                continue;
            };
            let Some(grad) = grads.get(node_id) else {
                continue;
            };
            if grad.len() != param.data.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: param.data.len(),
                    actual_rows: 1,
                    actual_cols: grad.len(),
                });
            }
            if self.m[i].len() != param.data.len() {
                self.m[i] = vec![0.0f32; param.data.len()];
                self.v[i] = vec![0.0f32; param.data.len()];
            }
            if param.dtype != DType::F32 && self.master_weights[i].is_none() {
                self.master_weights[i] = Some(param.data.clone());
            }

            let target: &mut [f32] = if let Some(ref mut master) = self.master_weights[i] {
                master.as_mut_slice()
            } else {
                param.data.as_mut_slice()
            };

            backend.adamw_step_f32(
                target,
                grad,
                &mut self.m[i],
                &mut self.v[i],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                self.step_count,
            )?;

            if let Some(ref master) = self.master_weights[i] {
                param.data.copy_from_slice(master);
                quantize_inplace(&mut param.data, param.dtype);
            }
        }
        Ok(())
    }

    /// Zero-allocation slice update using `Backend::adamw_step_f32` for `Sequential::fit`.
    pub fn step_slices(&mut self, params: &mut [&mut [f32]], grads: &[&[f32]]) -> OchreResult<()> {
        if params.len() != grads.len() {
            return Err(OchreError::LengthMismatch {
                features: params.len(),
                targets: grads.len(),
            });
        }
        self.step_count = self.step_count.saturating_add(1);
        if self.m.len() < params.len() {
            self.m.resize_with(params.len(), Vec::new);
            self.v.resize_with(params.len(), Vec::new);
        }
        let backend = get_backend(&self.device);

        for (i, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m[i].len() != p.len() {
                self.m[i] = vec![0.0f32; p.len()];
                self.v[i] = vec![0.0f32; p.len()];
            }
            backend.adamw_step_f32(
                p,
                g,
                &mut self.m[i],
                &mut self.v[i],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                self.step_count,
            )?;
        }
        Ok(())
    }
}

fn validate_adam_params(
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) -> OchreResult<()> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "Adam lr={lr} must be non-negative and finite ( >_< )"
        )));
    }
    if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
        return Err(OchreError::InvalidParameter(format!(
            "Adam betas ({beta1}, {beta2}) must be in [0.0, 1.0) ( >_< )"
        )));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "Adam eps={eps} must be strictly positive ( >_< )"
        )));
    }
    if !weight_decay.is_finite() || weight_decay < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "Adam weight_decay={weight_decay} must be non-negative ( >_< )"
        )));
    }
    Ok(())
}

/// Clips global L2 gradient norm across `grads` slices in-place and returns the pre-clip norm.
pub fn clip_grad_norm_(grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    let mut sum_sq = 0.0f64;
    for g in grads.iter() {
        for &v in g.iter() {
            let vf = v as f64;
            sum_sq += vf * vf;
        }
    }
    let total_norm = sum_sq.sqrt() as f32;
    if max_norm > 0.0 && total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in grads.iter_mut() {
            for v in g.iter_mut() {
                *v *= scale;
            }
        }
    }
    total_norm
}

/// Clips global L2 gradient norm inside a `GradStore` for the given `node_ids`.
pub fn clip_grad_store_norm_(grads: &mut GradStore, node_ids: &[usize], max_norm: f32) -> f32 {
    let mut sum_sq = 0.0f64;
    for &id in node_ids {
        if let Some(g) = grads.get(id) {
            for &v in g {
                let vf = v as f64;
                sum_sq += vf * vf;
            }
        }
    }
    let total_norm = sum_sq.sqrt() as f32;
    if max_norm > 0.0 && total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for &id in node_ids {
            if let Some(g) = grads.get_mut(id) {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
    total_norm
}

/// Decays learning rate by `gamma` every `step_size` epochs.
#[derive(Clone, Debug)]
pub struct StepLR {
    pub base_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
    pub current_epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> OchreResult<Self> {
        if step_size == 0 {
            return Err(OchreError::InvalidParameter(
                "StepLR step_size must be > 0 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            base_lr,
            step_size,
            gamma,
            current_epoch: 0,
        })
    }

    pub fn get_lr(&self) -> f32 {
        let exp = (self.current_epoch / self.step_size) as i32;
        self.base_lr * self.gamma.powi(exp)
    }

    pub fn step(&mut self) -> f32 {
        self.current_epoch += 1;
        self.get_lr()
    }
}

/// Cosine annealing learning rate schedule from `base_lr` down to `eta_min`.
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    pub base_lr: f32,
    pub t_max: usize,
    pub eta_min: f32,
    pub current_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> OchreResult<Self> {
        if t_max == 0 {
            return Err(OchreError::InvalidParameter(
                "CosineAnnealingLR t_max must be > 0 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            base_lr,
            t_max,
            eta_min,
            current_epoch: 0,
        })
    }

    pub fn get_lr(&self) -> f32 {
        let t = self.current_epoch.min(self.t_max) as f32;
        let tmax = self.t_max as f32;
        self.eta_min
            + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (std::f32::consts::PI * t / tmax).cos())
    }

    pub fn step(&mut self) -> f32 {
        self.current_epoch += 1;
        self.get_lr()
    }
}

/// Linear warmup followed by cosine decay down to `min_lr`.
#[derive(Clone, Debug)]
pub struct WarmupCosineLR {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub min_lr: f32,
    pub current_step: usize,
}

impl WarmupCosineLR {
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f32,
    ) -> OchreResult<Self> {
        if total_steps == 0 {
            return Err(OchreError::InvalidParameter(
                "WarmupCosineLR total_steps must be > 0 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            base_lr,
            warmup_steps,
            total_steps,
            min_lr,
            current_step: 0,
        })
    }

    pub fn get_lr(&self) -> f32 {
        if self.warmup_steps > 0 && self.current_step < self.warmup_steps {
            return self.base_lr * ((self.current_step + 1) as f32 / self.warmup_steps as f32);
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1) as f32;
        let progress = ((self.current_step.saturating_sub(self.warmup_steps)) as f32 / decay_steps)
            .clamp(0.0, 1.0);
        self.min_lr
            + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    pub fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_and_clip_grad_norm() {
        let mut p =
            Tensor::parameter(vec![2.0, -4.0], vec![2], DType::F32, DeviceType::Cpu).unwrap();
        let mut g_buf = vec![3.0f32, 4.0f32];
        let norm = clip_grad_norm_(&mut [g_buf.as_mut_slice()], 2.5);
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((g_buf[0] - 1.5).abs() < 1e-4);

        let mut store = GradStore::new();
        store.insert(p.node_id.unwrap(), g_buf);

        let mut sgd = SGD::new(0.1, 0.9, true, 0.0).unwrap();
        sgd.step(&mut [&mut p], &store).unwrap();
        assert!(p.data[0] < 2.0);
        assert!(p.data[1] < -4.0);
    }

    #[test]
    fn test_adamw_bf16_master_weights() {
        let mut p =
            Tensor::parameter(vec![1.0, -1.0], vec![2], DType::BF16, DeviceType::Cpu).unwrap();
        let mut store = GradStore::new();
        store.insert(p.node_id.unwrap(), vec![0.01, -0.01]);

        let mut adamw = AdamW::with_lr(1e-3, 0.01, DeviceType::Cpu).unwrap();
        for _ in 0..5 {
            adamw.step(&mut [&mut p], &store).unwrap();
        }
        let master = adamw.master_weight(0).unwrap();
        assert!((master[0] - 1.0).abs() > 1e-4);
        assert!(master[0] < 1.0);
    }

    #[test]
    fn test_lr_schedulers() {
        let mut step = StepLR::new(0.1, 2, 0.5).unwrap();
        assert!((step.get_lr() - 0.1).abs() < 1e-6);
        step.step();
        assert!((step.get_lr() - 0.1).abs() < 1e-6);
        step.step();
        assert!((step.get_lr() - 0.05).abs() < 1e-6);

        let mut cos = CosineAnnealingLR::new(0.1, 10, 0.0).unwrap();
        for _ in 0..10 {
            cos.step();
        }
        assert!(cos.get_lr() < 1e-6);

        let mut wcos = WarmupCosineLR::new(0.1, 2, 10, 0.001).unwrap();
        assert!((wcos.get_lr() - 0.05).abs() < 1e-6);
        wcos.step();
        assert!((wcos.get_lr() - 0.1).abs() < 1e-6);
    }
}
