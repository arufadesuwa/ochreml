// OchreML Loss Functions Module (*≧ω≦*)
// Provides MSELoss and fused CrossEntropyLoss (Backend::cross_entropy_grad_f32) with autograd support.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, Op};
use crate::nn::dtype::DType;
use crate::nn::tensor::Tensor;

/// Loss reduction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

/// Mean Squared Error loss with forward/backward raw slices and autograd `Tensor` interface.
#[derive(Clone, Copy, Debug)]
pub struct MSELoss {
    pub reduction: Reduction,
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

impl MSELoss {
    /// Creates an `MSELoss` with the requested `reduction`.
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes `(loss_value, d_pred)` for flat `pred` and `target` slices.
    pub fn forward(&self, pred: &[f32], target: &[f32]) -> OchreResult<(f32, Vec<f32>)> {
        let mut d_pred = vec![0.0f32; pred.len()];
        let loss = self.forward_into(pred, target, &mut d_pred)?;
        Ok((loss, d_pred))
    }

    /// Zero-allocation MSE forward+backward writing `d_pred` into a pre-allocated slice.
    pub fn forward_into(
        &self,
        pred: &[f32],
        target: &[f32],
        d_pred: &mut [f32],
    ) -> OchreResult<f32> {
        if pred.is_empty() {
            return Err(OchreError::EmptyData);
        }
        if pred.len() != target.len() || pred.len() != d_pred.len() {
            return Err(OchreError::DimensionMismatch {
                expected_rows: 1,
                expected_cols: pred.len(),
                actual_rows: 1,
                actual_cols: target.len(),
            });
        }
        let scale = match self.reduction {
            Reduction::Mean => 2.0f32 / (pred.len() as f32),
            Reduction::Sum | Reduction::None => 2.0f32,
        };
        let mut sum_sq = 0.0f32;
        for ((dst, &p), &t) in d_pred.iter_mut().zip(pred.iter()).zip(target.iter()) {
            let diff = p - t;
            sum_sq += diff * diff;
            *dst = scale * diff;
        }
        let loss = match self.reduction {
            Reduction::Mean => sum_sq / (pred.len() as f32),
            Reduction::Sum | Reduction::None => sum_sq,
        };
        Ok(loss)
    }

    /// Autograd-compatible `MSELoss` returning a scalar `[1]` `Tensor` (or elementwise for `Reduction::None`).
    pub fn forward_tensor(&self, pred: &Tensor, target: &Tensor) -> OchreResult<Tensor> {
        if pred.shape != target.shape {
            return Err(OchreError::InvalidParameter(format!(
                "MSELoss shape mismatch: pred {:?} vs target {:?} ( >_< )",
                pred.shape, target.shape
            )));
        }
        if pred.data.is_empty() {
            return Err(OchreError::EmptyData);
        }

        if self.reduction == Reduction::None {
            let diff = pred.sub(target)?;
            return diff.mul(&diff);
        }

        let n = pred.data.len() as f32;
        let scale = match self.reduction {
            Reduction::Mean => 2.0 / n,
            Reduction::Sum | Reduction::None => 2.0,
        };
        let mut diff = Vec::with_capacity(pred.data.len());
        let mut sum_sq = 0.0f32;
        for (&p, &t) in pred.data.iter().zip(target.data.iter()) {
            let d = p - t;
            sum_sq += d * d;
            diff.push(d);
        }
        let loss = match self.reduction {
            Reduction::Mean => sum_sq / n,
            Reduction::Sum | Reduction::None => sum_sq,
        };

        let grad_active = AutogradTape::is_grad_enabled() && pred.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::MSE {
                pred_id: pred.node_id,
                diff,
                scale,
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: vec![1],
            strides: vec![1],
            data: vec![loss],
            dtype: DType::F32,
            device: pred.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }
}

/// Fused Cross-Entropy Loss backed by `Backend::cross_entropy_grad_f32` (N9 kernel).
#[derive(Clone, Copy, Debug)]
pub struct CrossEntropyLoss {
    pub label_smoothing: f32,
    pub ignore_index: Option<usize>,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {
            label_smoothing: 0.0,
            ignore_index: None,
        }
    }
}

impl CrossEntropyLoss {
    /// Creates a `CrossEntropyLoss` validating `0.0 <= label_smoothing < 1.0`.
    pub fn new(label_smoothing: f32, ignore_index: Option<usize>) -> OchreResult<Self> {
        if !label_smoothing.is_finite() || !(0.0..1.0).contains(&label_smoothing) {
            return Err(OchreError::InvalidParameter(format!(
                "label_smoothing={label_smoothing} must be in [0.0, 1.0) ( >_< )"
            )));
        }
        Ok(Self {
            label_smoothing,
            ignore_index,
        })
    }

    /// Computes `(mean_loss, d_logits)` via `Backend::cross_entropy_grad_f32`.
    pub fn forward(
        &self,
        logits: &[f32],
        targets: &[usize],
        rows: usize,
        cols: usize,
        device: &DeviceType,
    ) -> OchreResult<(f32, Vec<f32>)> {
        get_backend(device).cross_entropy_grad_f32(
            logits,
            targets,
            rows,
            cols,
            self.label_smoothing,
            self.ignore_index,
        )
    }

    /// Autograd-compatible `CrossEntropyLoss` returning a scalar `[1]` `Tensor`.
    pub fn forward_tensor(&self, logits: &Tensor, targets: &[usize]) -> OchreResult<Tensor> {
        if logits.shape.len() < 2 {
            return Err(OchreError::InvalidParameter(format!(
                "CrossEntropyLoss expects logits rank >= 2, got {:?} ( >_< )",
                logits.shape
            )));
        }
        let cols = logits.shape[logits.shape.len() - 1];
        let rows: usize = logits.shape[..logits.shape.len() - 1]
            .iter()
            .copied()
            .product();
        if targets.len() != rows {
            return Err(OchreError::LengthMismatch {
                features: rows,
                targets: targets.len(),
            });
        }

        let (loss, d_logits) = get_backend(&logits.device).cross_entropy_grad_f32(
            &logits.data,
            targets,
            rows,
            cols,
            self.label_smoothing,
            self.ignore_index,
        )?;

        let grad_active = AutogradTape::is_grad_enabled() && logits.requires_grad;
        let node_id = if grad_active {
            Some(AutogradTape::record(Op::CrossEntropy {
                logits_id: logits.node_id,
                d_logits,
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: vec![1],
            strides: vec![1],
            data: vec![loss],
            dtype: DType::F32,
            device: logits.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_forward_and_backward() {
        AutogradTape::clear();
        let mse = MSELoss::new(Reduction::Mean);
        let pred =
            Tensor::parameter(vec![2.0, 0.0, -1.0], vec![3], DType::F32, DeviceType::Cpu).unwrap();
        let target = Tensor::new(
            vec![1.0, 1.0, 1.0],
            vec![3],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let loss = mse.forward_tensor(&pred, &target).unwrap();
        // ((1)^2 + (-1)^2 + (-2)^2) / 3 = 6/3 = 2.0
        assert!((loss.data[0] - 2.0).abs() < 1e-6);

        let grads = loss.backward().unwrap();
        let dp = grads.get(pred.node_id.unwrap()).unwrap();
        // d/dp = 2/3 * (p - t) = [2/3, -2/3, -4/3]
        assert!((dp[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((dp[1] + 2.0 / 3.0).abs() < 1e-5);
        assert!((dp[2] + 4.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss_tensor_backward() {
        AutogradTape::clear();
        let ce = CrossEntropyLoss::new(0.0, None).unwrap();
        let logits = Tensor::parameter(
            vec![2.0, 0.0, -1.0, 0.1, 1.5, -0.5],
            vec![2, 3],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = [0usize, 1usize];
        let loss = ce.forward_tensor(&logits, &targets).unwrap();
        assert!(loss.data[0] > 0.0);

        let grads = loss.backward().unwrap();
        let dl = grads.get(logits.node_id.unwrap()).unwrap();
        assert_eq!(dl.len(), 6);
        // Sum of gradients across each row should be ~0 for standard cross-entropy
        assert!((dl[0] + dl[1] + dl[2]).abs() < 1e-5);
        assert!((dl[3] + dl[4] + dl[5]).abs() < 1e-5);
    }
}
