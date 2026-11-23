// OchreML Multi-Layer Perceptron Module (*≧ω≦*)
// Provides hardware-accelerated MLPClassifier and MLPRegressor using Backend E1/E2/N2 primitives.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_str, write_usize,
    write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::primitives::stats_ops::ElementwiseOp;
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub const TAG_MLP_CLASSIFIER: u8 = 81;
pub const TAG_MLP_REGRESSOR: u8 = 82;

const GPU_MATMUL_WORK_THRESHOLD: usize = 65_536;
const GPU_ELEM_WORK_THRESHOLD: usize = 4_096;

// =========================================================================
// DETERMINISTIC PRNG (SplitMix64)
// =========================================================================

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        Self {
            state: seed.unwrap_or(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// =========================================================================
// HARDWARE PRIMITIVE DISPATCHERS (E1, E2, N2 WITH CPU GUARD)
// =========================================================================

#[inline]
fn dispatch_matmul(backend: &Backend<'_>, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    let work = a.rows.saturating_mul(a.cols).saturating_mul(b.cols);
    if !matches!(backend, Backend::Cpu) && work >= GPU_MATMUL_WORK_THRESHOLD {
        backend.matmul(a, b)
    } else {
        Backend::Cpu.matmul(a, b)
    }
}

#[inline]
fn dispatch_matmul_at_b(backend: &Backend<'_>, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    let work = a.rows.saturating_mul(a.cols).saturating_mul(b.cols);
    if !matches!(backend, Backend::Cpu) && work >= GPU_MATMUL_WORK_THRESHOLD {
        backend.matmul_at_b(a, b)
    } else {
        Backend::Cpu.matmul_at_b(a, b)
    }
}

#[inline]
fn dispatch_matmul_a_bt(backend: &Backend<'_>, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    if a.cols != b.cols {
        return Err(OchreError::MatmulDimensionMismatch {
            rows_a: a.rows,
            cols_a: a.cols,
            rows_b: b.cols,
            cols_b: b.rows,
        });
    }
    let mut b_t = Matrix::zeros(b.cols, b.rows);
    for r in 0..b.rows {
        let src = &b.data[(r * b.cols)..(r * b.cols + b.cols)];
        for (c, &src_c) in src.iter().enumerate().take(b.cols) {
            b_t.data[c * b.rows + r] = src_c;
        }
    }
    dispatch_matmul(backend, a, &b_t)
}

#[inline]
fn dispatch_elementwise(
    backend: &Backend<'_>,
    a: &Matrix,
    op: ElementwiseOp,
) -> OchreResult<Matrix> {
    let work = a.rows.saturating_mul(a.cols);
    if !matches!(backend, Backend::Cpu) && work >= GPU_ELEM_WORK_THRESHOLD {
        backend.elementwise_op(a, op)
    } else {
        Backend::Cpu.elementwise_op(a, op)
    }
}

#[inline]
fn dispatch_row_softmax(backend: &Backend<'_>, a: &Matrix) -> OchreResult<Matrix> {
    let work = a.rows.saturating_mul(a.cols);
    if !matches!(backend, Backend::Cpu) && work >= GPU_ELEM_WORK_THRESHOLD {
        backend.row_softmax(a)
    } else {
        Backend::Cpu.row_softmax(a)
    }
}

// =========================================================================
// VALIDATION & SHARED MLP CORE
// =========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Identity,
    Logistic,
    Tanh,
    Relu,
}

impl Activation {
    fn from_str(s: &str) -> OchreResult<Self> {
        match s {
            "identity" => Ok(Self::Identity),
            "logistic" => Ok(Self::Logistic),
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            _ => Err(OchreError::InvalidParameter(format!(
                "Invalid activation '{}'. Supported: 'identity', 'logistic', 'tanh', 'relu'",
                s
            ))),
        }
    }

    fn apply(&self, backend: &Backend<'_>, z: &Matrix) -> OchreResult<Matrix> {
        match self {
            Self::Identity => Ok(z.clone()),
            Self::Logistic => dispatch_elementwise(backend, z, ElementwiseOp::Sigmoid),
            Self::Tanh => dispatch_elementwise(backend, z, ElementwiseOp::Tanh),
            Self::Relu => dispatch_elementwise(backend, z, ElementwiseOp::Relu),
        }
    }

    fn apply_grad_inplace(&self, delta: &mut Matrix, post_act: &Matrix) {
        match self {
            Self::Identity => {}
            Self::Relu => {
                for (d, &a) in delta.data.iter_mut().zip(post_act.data.iter()) {
                    if a <= 0.0 {
                        *d = 0.0;
                    }
                }
            }
            Self::Tanh => {
                for (d, &a) in delta.data.iter_mut().zip(post_act.data.iter()) {
                    *d *= 1.0 - a * a;
                }
            }
            Self::Logistic => {
                for (d, &a) in delta.data.iter_mut().zip(post_act.data.iter()) {
                    *d *= a * (1.0 - a);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Solver {
    Adam,
    Sgd,
    Lbfgs,
}

impl Solver {
    fn from_str(s: &str) -> OchreResult<Self> {
        match s {
            "adam" => Ok(Self::Adam),
            "sgd" => Ok(Self::Sgd),
            "lbfgs" => Ok(Self::Lbfgs),
            _ => Err(OchreError::InvalidParameter(format!(
                "Invalid solver '{}'. Supported: 'adam', 'sgd', 'lbfgs'",
                s
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LearningRateSchedule {
    Constant,
    Invscaling,
    Adaptive,
}

impl LearningRateSchedule {
    fn from_str(s: &str) -> OchreResult<Self> {
        match s {
            "constant" => Ok(Self::Constant),
            "invscaling" => Ok(Self::Invscaling),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(OchreError::InvalidParameter(format!(
                "Invalid learning_rate '{}'. Supported: 'constant', 'invscaling', 'adaptive'",
                s
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutLoss {
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SquaredError,
}

#[derive(Clone, Debug)]
struct MlpConfig {
    hidden_layer_sizes: Vec<usize>,
    activation: String,
    solver: String,
    alpha: f64,
    batch_size: Option<usize>,
    learning_rate: String,
    learning_rate_init: f64,
    power_t: f64,
    max_iter: usize,
    shuffle: bool,
    random_state: Option<u64>,
    tol: f64,
    momentum: f64,
    nesterovs_momentum: bool,
    early_stopping: bool,
    validation_fraction: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    n_iter_no_change: usize,
}

impl MlpConfig {
    fn validate(&self) -> OchreResult<()> {
        if self.hidden_layer_sizes.is_empty() || self.hidden_layer_sizes.contains(&0) {
            return Err(OchreError::InvalidParameter(
                "hidden_layer_sizes must be non-empty with all layer sizes > 0".to_string(),
            ));
        }
        Activation::from_str(&self.activation)?;
        Solver::from_str(&self.solver)?;
        LearningRateSchedule::from_str(&self.learning_rate)?;

        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(OchreError::InvalidAlpha(self.alpha));
        }
        if let Some(bs) = self.batch_size
            && bs == 0
        {
            return Err(OchreError::InvalidParameter(
                "batch_size must be >= 1".to_string(),
            ));
        }
        if !self.learning_rate_init.is_finite() || self.learning_rate_init <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "learning_rate_init ({}) must be > 0",
                self.learning_rate_init
            )));
        }
        if !self.power_t.is_finite() {
            return Err(OchreError::InvalidParameter(
                "power_t must be finite".to_string(),
            ));
        }
        if self.max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1".to_string(),
            ));
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(OchreError::InvalidParameter(
                "tol must be finite and >= 0".to_string(),
            ));
        }
        if !self.momentum.is_finite() || !(0.0..=1.0).contains(&self.momentum) {
            return Err(OchreError::InvalidParameter(
                "momentum must be in [0.0, 1.0]".to_string(),
            ));
        }
        if !self.validation_fraction.is_finite()
            || self.validation_fraction <= 0.0
            || self.validation_fraction >= 1.0
        {
            return Err(OchreError::InvalidParameter(
                "validation_fraction must be in (0.0, 1.0)".to_string(),
            ));
        }
        if !self.beta_1.is_finite() || self.beta_1 < 0.0 || self.beta_1 >= 1.0 {
            return Err(OchreError::InvalidParameter(
                "beta_1 must be in [0.0, 1.0)".to_string(),
            ));
        }
        if !self.beta_2.is_finite() || self.beta_2 < 0.0 || self.beta_2 >= 1.0 {
            return Err(OchreError::InvalidParameter(
                "beta_2 must be in [0.0, 1.0)".to_string(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(OchreError::InvalidParameter(
                "epsilon must be > 0".to_string(),
            ));
        }
        if self.n_iter_no_change == 0 {
            return Err(OchreError::InvalidParameter(
                "n_iter_no_change must be >= 1".to_string(),
            ));
        }
        Ok(())
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        write_usize_vec(buf, &self.hidden_layer_sizes);
        write_str(buf, &self.activation);
        write_str(buf, &self.solver);
        write_f64(buf, self.alpha);
        write_opt_usize(buf, self.batch_size);
        write_str(buf, &self.learning_rate);
        write_f64(buf, self.learning_rate_init);
        write_f64(buf, self.power_t);
        write_usize(buf, self.max_iter);
        write_bool(buf, self.shuffle);
        write_opt_u64(buf, self.random_state);
        write_f64(buf, self.tol);
        write_f64(buf, self.momentum);
        write_bool(buf, self.nesterovs_momentum);
        write_bool(buf, self.early_stopping);
        write_f64(buf, self.validation_fraction);
        write_f64(buf, self.beta_1);
        write_f64(buf, self.beta_2);
        write_f64(buf, self.epsilon);
        write_usize(buf, self.n_iter_no_change);
    }

    fn read_from(reader: &mut ByteReader<'_>) -> OchreResult<Self> {
        Ok(Self {
            hidden_layer_sizes: reader.read_usize_vec()?,
            activation: reader.read_str()?,
            solver: reader.read_str()?,
            alpha: reader.read_f64()?,
            batch_size: reader.read_opt_usize()?,
            learning_rate: reader.read_str()?,
            learning_rate_init: reader.read_f64()?,
            power_t: reader.read_f64()?,
            max_iter: reader.read_usize()?,
            shuffle: reader.read_bool()?,
            random_state: reader.read_opt_u64()?,
            tol: reader.read_f64()?,
            momentum: reader.read_f64()?,
            nesterovs_momentum: reader.read_bool()?,
            early_stopping: reader.read_bool()?,
            validation_fraction: reader.read_f64()?,
            beta_1: reader.read_f64()?,
            beta_2: reader.read_f64()?,
            epsilon: reader.read_f64()?,
            n_iter_no_change: reader.read_usize()?,
        })
    }
}

fn validate_matrix_finite(m: &Matrix, name: &str) -> OchreResult<()> {
    if m.rows == 0 || m.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if m.data.iter().any(|&v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{} contains NaN or Infinity",
            name
        )));
    }
    Ok(())
}

fn validate_slice_finite(s: &[f64], name: &str) -> OchreResult<()> {
    if s.is_empty() {
        return Err(OchreError::EmptyData);
    }
    if s.iter().any(|&v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "{} contains NaN or Infinity",
            name
        )));
    }
    Ok(())
}

fn add_bias_inplace(z: &mut Matrix, bias: &[f64]) {
    let cols = z.cols;
    for r in 0..z.rows {
        let row = &mut z.data[(r * cols)..(r * cols + cols)];
        for (dst, &b) in row.iter_mut().zip(bias.iter()) {
            *dst += b;
        }
    }
}

fn slice_rows(m: &Matrix, indices: &[usize]) -> Matrix {
    let cols = m.cols;
    let mut data = Vec::with_capacity(indices.len() * cols);
    for &idx in indices {
        let start = idx * cols;
        data.extend_from_slice(&m.data[start..(start + cols)]);
    }
    Matrix {
        rows: indices.len(),
        cols,
        data,
    }
}

#[derive(Clone, Debug)]
struct MlpState {
    coefs: Vec<Matrix>,
    intercepts: Vec<Vec<f64>>,
    n_features_in: usize,
    n_outputs: usize,
    out_loss: OutLoss,
    // Optimizer state for partial_fit
    m_w: Vec<Vec<f64>>,
    v_w: Vec<Vec<f64>>,
    m_b: Vec<Vec<f64>>,
    v_b: Vec<Vec<f64>>,
    vel_w: Vec<Vec<f64>>,
    vel_b: Vec<Vec<f64>>,
    adam_t: usize,
    sgd_t: usize,
    current_lr: f64,
    n_iter: usize,
    loss: f64,
    loss_curve: Vec<f64>,
}

impl MlpState {
    fn init(
        cfg: &MlpConfig,
        n_features: usize,
        n_outputs: usize,
        out_loss: OutLoss,
        rng: &mut SplitMix64,
    ) -> OchreResult<Self> {
        let act = Activation::from_str(&cfg.activation)?;
        let factor = if act == Activation::Logistic {
            2.0
        } else {
            6.0
        };

        let mut layer_sizes = Vec::with_capacity(cfg.hidden_layer_sizes.len() + 2);
        layer_sizes.push(n_features);
        layer_sizes.extend_from_slice(&cfg.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

        let n_weight_layers = layer_sizes.len() - 1;
        let mut coefs = Vec::with_capacity(n_weight_layers);
        let mut intercepts = Vec::with_capacity(n_weight_layers);
        let mut m_w = Vec::with_capacity(n_weight_layers);
        let mut v_w = Vec::with_capacity(n_weight_layers);
        let mut m_b = Vec::with_capacity(n_weight_layers);
        let mut v_b = Vec::with_capacity(n_weight_layers);
        let mut vel_w = Vec::with_capacity(n_weight_layers);
        let mut vel_b = Vec::with_capacity(n_weight_layers);

        for l in 0..n_weight_layers {
            let fan_in = layer_sizes[l];
            let fan_out = layer_sizes[l + 1];
            let bound = (factor / ((fan_in + fan_out) as f64)).sqrt();

            let mut w_data = Vec::with_capacity(fan_in * fan_out);
            for _ in 0..(fan_in * fan_out) {
                w_data.push(rng.uniform(-bound, bound));
            }
            let mut b_data = Vec::with_capacity(fan_out);
            for _ in 0..fan_out {
                b_data.push(rng.uniform(-bound, bound));
            }

            coefs.push(Matrix {
                rows: fan_in,
                cols: fan_out,
                data: w_data,
            });
            intercepts.push(b_data);
            m_w.push(vec![0.0; fan_in * fan_out]);
            v_w.push(vec![0.0; fan_in * fan_out]);
            m_b.push(vec![0.0; fan_out]);
            v_b.push(vec![0.0; fan_out]);
            vel_w.push(vec![0.0; fan_in * fan_out]);
            vel_b.push(vec![0.0; fan_out]);
        }

        Ok(Self {
            coefs,
            intercepts,
            n_features_in: n_features,
            n_outputs,
            out_loss,
            m_w,
            v_w,
            m_b,
            v_b,
            vel_w,
            vel_b,
            adam_t: 0,
            sgd_t: 0,
            current_lr: cfg.learning_rate_init,
            n_iter: 0,
            loss: 0.0,
            loss_curve: Vec::new(),
        })
    }

    fn forward_predict(
        &self,
        backend: &Backend<'_>,
        cfg: &MlpConfig,
        x: &Matrix,
    ) -> OchreResult<Matrix> {
        let act = Activation::from_str(&cfg.activation)?;
        let n_layers = self.coefs.len();
        let mut a = x.clone();
        for l in 0..n_layers {
            let mut z = dispatch_matmul(backend, &a, &self.coefs[l])?;
            add_bias_inplace(&mut z, &self.intercepts[l]);
            if l + 1 < n_layers {
                a = act.apply(backend, &z)?;
            } else {
                a = match self.out_loss {
                    OutLoss::BinaryCrossEntropy => {
                        dispatch_elementwise(backend, &z, ElementwiseOp::Sigmoid)?
                    }
                    OutLoss::CategoricalCrossEntropy => dispatch_row_softmax(backend, &z)?,
                    OutLoss::SquaredError => z,
                };
            }
        }
        Ok(a)
    }

    fn forward_and_backprop(
        &self,
        backend: &Backend<'_>,
        act: Activation,
        alpha: f64,
        x_batch: &Matrix,
        y_batch: &Matrix,
    ) -> OchreResult<(f64, Vec<Matrix>, Vec<Vec<f64>>)> {
        let n_layers = self.coefs.len();
        let batch_n = x_batch.rows as f64;
        let mut activations = Vec::with_capacity(n_layers + 1);
        activations.push(x_batch.clone());

        for l in 0..n_layers {
            let mut z = dispatch_matmul(backend, &activations[l], &self.coefs[l])?;
            add_bias_inplace(&mut z, &self.intercepts[l]);
            let a_next = if l + 1 < n_layers {
                act.apply(backend, &z)?
            } else {
                match self.out_loss {
                    OutLoss::BinaryCrossEntropy => {
                        dispatch_elementwise(backend, &z, ElementwiseOp::Sigmoid)?
                    }
                    OutLoss::CategoricalCrossEntropy => dispatch_row_softmax(backend, &z)?,
                    OutLoss::SquaredError => z,
                }
            };
            activations.push(a_next);
        }

        let a_out = &activations[n_layers];
        let mut loss = match self.out_loss {
            OutLoss::BinaryCrossEntropy => {
                let mut sum = 0.0;
                for (&p, &y) in a_out.data.iter().zip(y_batch.data.iter()) {
                    let pc = p.clamp(1e-15, 1.0 - 1e-15);
                    sum -= y * pc.ln() + (1.0 - y) * (1.0 - pc).ln();
                }
                sum / batch_n
            }
            OutLoss::CategoricalCrossEntropy => {
                let mut sum = 0.0;
                for (&p, &y) in a_out.data.iter().zip(y_batch.data.iter()) {
                    if y > 0.0 {
                        let pc = p.clamp(1e-15, 1.0);
                        sum -= y * pc.ln();
                    }
                }
                sum / batch_n
            }
            OutLoss::SquaredError => {
                let mut sum = 0.0;
                for (&p, &y) in a_out.data.iter().zip(y_batch.data.iter()) {
                    let diff = p - y;
                    sum += diff * diff;
                }
                0.5 * sum / batch_n
            }
        };

        // L2 regularization loss
        if alpha > 0.0 {
            let mut l2 = 0.0;
            for w in &self.coefs {
                for &v in &w.data {
                    l2 += v * v;
                }
            }
            loss += 0.5 * (alpha / batch_n) * l2;
        }

        // Output delta = A_out - Y
        let mut delta_data = Vec::with_capacity(a_out.data.len());
        for (&p, &y) in a_out.data.iter().zip(y_batch.data.iter()) {
            delta_data.push(p - y);
        }
        let mut delta = Matrix {
            rows: a_out.rows,
            cols: a_out.cols,
            data: delta_data,
        };

        let mut grad_w = vec![Matrix::zeros(1, 1); n_layers];
        let mut grad_b = vec![Vec::new(); n_layers];
        let inv_n = 1.0 / batch_n;
        let l2_scale = alpha * inv_n;

        for l in (0..n_layers).rev() {
            let mut gw = dispatch_matmul_at_b(backend, &activations[l], &delta)?;
            for (g, &w) in gw.data.iter_mut().zip(self.coefs[l].data.iter()) {
                *g = (*g) * inv_n + l2_scale * w;
            }
            grad_w[l] = gw;

            let mut gb = vec![0.0; delta.cols];
            for r in 0..delta.rows {
                let row = &delta.data[(r * delta.cols)..(r * delta.cols + delta.cols)];
                for (dst, &val) in gb.iter_mut().zip(row.iter()) {
                    *dst += val;
                }
            }
            for g in &mut gb {
                *g *= inv_n;
            }
            grad_b[l] = gb;

            if l > 0 {
                let mut delta_prev = dispatch_matmul_a_bt(backend, &delta, &self.coefs[l])?;
                act.apply_grad_inplace(&mut delta_prev, &activations[l]);
                delta = delta_prev;
            }
        }

        Ok((loss, grad_w, grad_b))
    }

    fn step_adam(&mut self, cfg: &MlpConfig, grad_w: &[Matrix], grad_b: &[Vec<f64>]) {
        self.adam_t += 1;
        let t_f = self.adam_t as f64;
        let b1 = cfg.beta_1;
        let b2 = cfg.beta_2;
        let eps = cfg.epsilon;
        let lr_t = self.current_lr * (1.0 - b2.powf(t_f)).sqrt() / (1.0 - b1.powf(t_f)).max(1e-15);

        for l in 0..self.coefs.len() {
            let w = &mut self.coefs[l].data;
            let gw = &grad_w[l].data;
            let mw = &mut self.m_w[l];
            let vw = &mut self.v_w[l];
            for i in 0..w.len() {
                mw[i] = b1 * mw[i] + (1.0 - b1) * gw[i];
                vw[i] = b2 * vw[i] + (1.0 - b2) * gw[i] * gw[i];
                w[i] -= lr_t * mw[i] / (vw[i].sqrt() + eps);
            }

            let b = &mut self.intercepts[l];
            let gb = &grad_b[l];
            let mb = &mut self.m_b[l];
            let vb = &mut self.v_b[l];
            for i in 0..b.len() {
                mb[i] = b1 * mb[i] + (1.0 - b1) * gb[i];
                vb[i] = b2 * vb[i] + (1.0 - b2) * gb[i] * gb[i];
                b[i] -= lr_t * mb[i] / (vb[i].sqrt() + eps);
            }
        }
    }

    fn step_sgd(
        &mut self,
        cfg: &MlpConfig,
        lr_sched: LearningRateSchedule,
        batch_len: usize,
        grad_w: &[Matrix],
        grad_b: &[Vec<f64>],
    ) {
        self.sgd_t += batch_len;
        let lr = match lr_sched {
            LearningRateSchedule::Invscaling => {
                cfg.learning_rate_init / (self.sgd_t as f64).powf(cfg.power_t)
            }
            LearningRateSchedule::Constant | LearningRateSchedule::Adaptive => self.current_lr,
        };
        let mom = cfg.momentum;
        let nesterov = cfg.nesterovs_momentum;

        for l in 0..self.coefs.len() {
            let w = &mut self.coefs[l].data;
            let gw = &grad_w[l].data;
            let vw = &mut self.vel_w[l];
            for i in 0..w.len() {
                vw[i] = mom * vw[i] - lr * gw[i];
                if nesterov {
                    w[i] += mom * vw[i] - lr * gw[i];
                } else {
                    w[i] += vw[i];
                }
            }

            let b = &mut self.intercepts[l];
            let gb = &grad_b[l];
            let vb = &mut self.vel_b[l];
            for i in 0..b.len() {
                vb[i] = mom * vb[i] - lr * gb[i];
                if nesterov {
                    b[i] += mom * vb[i] - lr * gb[i];
                } else {
                    b[i] += vb[i];
                }
            }
        }
    }

    fn run_one_epoch(
        &mut self,
        backend: &Backend<'_>,
        cfg: &MlpConfig,
        x: &Matrix,
        y: &Matrix,
        rng: &mut SplitMix64,
    ) -> OchreResult<f64> {
        let act = Activation::from_str(&cfg.activation)?;
        let solver = Solver::from_str(&cfg.solver)?;
        let lr_sched = LearningRateSchedule::from_str(&cfg.learning_rate)?;

        let n_samples = x.rows;
        let batch_size = cfg
            .batch_size
            .unwrap_or_else(|| n_samples.min(200))
            .min(n_samples)
            .max(1);

        let mut indices: Vec<usize> = (0..n_samples).collect();
        if cfg.shuffle {
            rng.shuffle(&mut indices);
        }

        let mut total_loss = 0.0;
        let mut start = 0;
        while start < n_samples {
            let end = (start + batch_size).min(n_samples);
            let batch_idx = &indices[start..end];
            let x_b = slice_rows(x, batch_idx);
            let y_b = slice_rows(y, batch_idx);

            let (b_loss, gw, gb) =
                self.forward_and_backprop(backend, act, cfg.alpha, &x_b, &y_b)?;
            total_loss += b_loss * (batch_idx.len() as f64);

            match solver {
                Solver::Adam => self.step_adam(cfg, &gw, &gb),
                Solver::Sgd => self.step_sgd(cfg, lr_sched, batch_idx.len(), &gw, &gb),
                Solver::Lbfgs => {
                    return Err(OchreError::InvalidParameter(
                        "partial_fit is not supported for solver='lbfgs'".to_string(),
                    ));
                }
            }
            start = end;
        }

        let epoch_loss = total_loss / (n_samples as f64);
        self.loss = epoch_loss;
        self.loss_curve.push(epoch_loss);
        self.n_iter += 1;
        Ok(epoch_loss)
    }

    fn pack_params(&self) -> Vec<f64> {
        let mut total = 0;
        for l in 0..self.coefs.len() {
            total += self.coefs[l].data.len() + self.intercepts[l].len();
        }
        let mut out = Vec::with_capacity(total);
        for l in 0..self.coefs.len() {
            out.extend_from_slice(&self.coefs[l].data);
            out.extend_from_slice(&self.intercepts[l]);
        }
        out
    }

    fn unpack_params(&mut self, theta: &[f64]) {
        let mut offset = 0;
        for l in 0..self.coefs.len() {
            let w_len = self.coefs[l].data.len();
            self.coefs[l]
                .data
                .copy_from_slice(&theta[offset..(offset + w_len)]);
            offset += w_len;
            let b_len = self.intercepts[l].len();
            self.intercepts[l].copy_from_slice(&theta[offset..(offset + b_len)]);
            offset += b_len;
        }
    }

    fn pack_grads(&self, grad_w: &[Matrix], grad_b: &[Vec<f64>]) -> Vec<f64> {
        let mut out = Vec::new();
        for l in 0..grad_w.len() {
            out.extend_from_slice(&grad_w[l].data);
            out.extend_from_slice(&grad_b[l]);
        }
        out
    }

    fn fit_lbfgs(
        &mut self,
        backend: &Backend<'_>,
        cfg: &MlpConfig,
        x: &Matrix,
        y: &Matrix,
    ) -> OchreResult<()> {
        let act = Activation::from_str(&cfg.activation)?;
        let m_hist = 10usize;
        let mut theta = self.pack_params();
        let dim = theta.len();

        let (mut f_val, gw, gb) = self.forward_and_backprop(backend, act, cfg.alpha, x, y)?;
        let mut grad = self.pack_grads(&gw, &gb);

        self.loss = f_val;
        self.loss_curve.push(f_val);

        let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
        let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(m_hist);

        for iter in 0..cfg.max_iter {
            let g_inf = grad.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            if g_inf <= cfg.tol {
                break;
            }

            // Two-loop L-BFGS direction
            let mut q = grad.clone();
            let k_len = s_hist.len();
            let mut alpha_l = vec![0.0; k_len];
            for i in (0..k_len).rev() {
                let mut dot = 0.0;
                for j in 0..dim {
                    dot += s_hist[i][j] * q[j];
                }
                alpha_l[i] = rho_hist[i] * dot;
                for j in 0..dim {
                    q[j] -= alpha_l[i] * y_hist[i][j];
                }
            }

            let gamma = if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
                let mut sy = 0.0;
                let mut yy = 0.0;
                for j in 0..dim {
                    sy += s_last[j] * y_last[j];
                    yy += y_last[j] * y_last[j];
                }
                if yy > 1e-30 { sy / yy } else { 1.0 }
            } else {
                1.0
            };

            let mut r: Vec<f64> = q.iter().map(|&v| v * gamma).collect();
            for i in 0..k_len {
                let mut dot = 0.0;
                for j in 0..dim {
                    dot += y_hist[i][j] * r[j];
                }
                let beta = rho_hist[i] * dot;
                let diff = alpha_l[i] - beta;
                for j in 0..dim {
                    r[j] += s_hist[i][j] * diff;
                }
            }

            // Descent direction p = -r
            let mut p: Vec<f64> = r.into_iter().map(|v| -v).collect();
            let mut slope: f64 = p.iter().zip(grad.iter()).map(|(&pi, &gi)| pi * gi).sum();
            if slope >= 0.0 {
                // Fallback to steepest descent
                p = grad.iter().map(|&g| -g).collect();
                slope = p.iter().zip(grad.iter()).map(|(&pi, &gi)| pi * gi).sum();
            }

            // Armijo backtracking line search
            let mut step = 1.0;
            let c1 = 1e-4;
            let mut accepted = false;
            let mut next_theta = theta.clone();
            let mut next_f = f_val;
            let mut next_grad = grad.clone();

            for _ in 0..25 {
                for j in 0..dim {
                    next_theta[j] = theta[j] + step * p[j];
                }
                self.unpack_params(&next_theta);
                let (nf, ngw, ngb) = self.forward_and_backprop(backend, act, cfg.alpha, x, y)?;
                if nf.is_finite() && nf <= f_val + c1 * step * slope {
                    next_f = nf;
                    next_grad = self.pack_grads(&ngw, &ngb);
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }

            if !accepted {
                self.unpack_params(&theta);
                break;
            }

            let s_vec: Vec<f64> = (0..dim).map(|j| next_theta[j] - theta[j]).collect();
            let y_vec: Vec<f64> = (0..dim).map(|j| next_grad[j] - grad[j]).collect();
            let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(&a, &b)| a * b).sum();
            if sy > 1e-12 {
                if s_hist.len() == m_hist {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                s_hist.push(s_vec);
                y_hist.push(y_vec);
                rho_hist.push(1.0 / sy);
            }

            let f_diff = (f_val - next_f).abs();
            theta = next_theta;
            f_val = next_f;
            grad = next_grad;

            self.loss = f_val;
            self.loss_curve.push(f_val);
            self.n_iter = iter + 1;

            if f_diff <= cfg.tol {
                break;
            }
        }

        Ok(())
    }

    fn fit(
        &mut self,
        backend: &Backend<'_>,
        cfg: &MlpConfig,
        x: &Matrix,
        y: &Matrix,
        rng: &mut SplitMix64,
    ) -> OchreResult<()> {
        let solver = Solver::from_str(&cfg.solver)?;
        if solver == Solver::Lbfgs {
            return self.fit_lbfgs(backend, cfg, x, y);
        }

        let lr_sched = LearningRateSchedule::from_str(&cfg.learning_rate)?;

        // Optional early stopping validation split
        let (x_tr, y_tr, x_val, y_val) = if cfg.early_stopping && x.rows >= 10 {
            let n = x.rows;
            let n_val = ((n as f64) * cfg.validation_fraction).round() as usize;
            let n_val = n_val.max(1).min(n - 1);
            let mut all_idx: Vec<usize> = (0..n).collect();
            if cfg.shuffle {
                rng.shuffle(&mut all_idx);
            }
            let val_idx = &all_idx[..n_val];
            let tr_idx = &all_idx[n_val..];
            (
                slice_rows(x, tr_idx),
                slice_rows(y, tr_idx),
                Some(slice_rows(x, val_idx)),
                Some(slice_rows(y, val_idx)),
            )
        } else {
            (x.clone(), y.clone(), None, None)
        };

        let mut best_loss = f64::INFINITY;
        let mut best_val_score = f64::NEG_INFINITY;
        let mut best_coefs = self.coefs.clone();
        let mut best_intercepts = self.intercepts.clone();
        let mut no_improve = 0usize;

        for _ in 0..cfg.max_iter {
            let epoch_loss = self.run_one_epoch(backend, cfg, &x_tr, &y_tr, rng)?;

            if let (Some(xv), Some(yv)) = (&x_val, &y_val) {
                let preds = self.forward_predict(backend, cfg, xv)?;
                let val_score = match self.out_loss {
                    OutLoss::BinaryCrossEntropy => {
                        let mut correct = 0usize;
                        for r in 0..yv.rows {
                            let p_cls = if preds.data[r] >= 0.5 { 1.0 } else { 0.0 };
                            if (p_cls - yv.data[r]).abs() < 0.5 {
                                correct += 1;
                            }
                        }
                        (correct as f64) / (yv.rows as f64)
                    }
                    OutLoss::CategoricalCrossEntropy => {
                        let mut correct = 0usize;
                        let k = yv.cols;
                        for r in 0..yv.rows {
                            let mut best_p = 0usize;
                            let mut best_pv = f64::NEG_INFINITY;
                            let mut best_y = 0usize;
                            let mut best_yv = f64::NEG_INFINITY;
                            for c in 0..k {
                                let pv = preds.data[r * k + c];
                                if pv > best_pv {
                                    best_pv = pv;
                                    best_p = c;
                                }
                                let yval = yv.data[r * k + c];
                                if yval > best_yv {
                                    best_yv = yval;
                                    best_y = c;
                                }
                            }
                            if best_p == best_y {
                                correct += 1;
                            }
                        }
                        (correct as f64) / (yv.rows as f64)
                    }
                    OutLoss::SquaredError => compute_r2_matrix(yv, &preds),
                };

                if val_score > best_val_score + cfg.tol {
                    best_val_score = val_score;
                    best_coefs = self.coefs.clone();
                    best_intercepts = self.intercepts.clone();
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else if epoch_loss < best_loss - cfg.tol {
                best_loss = epoch_loss;
                no_improve = 0;
            } else {
                no_improve += 1;
            }

            if no_improve >= cfg.n_iter_no_change {
                if solver == Solver::Sgd && lr_sched == LearningRateSchedule::Adaptive {
                    if self.current_lr <= 1e-6 {
                        break;
                    }
                    self.current_lr /= 5.0;
                    no_improve = 0;
                } else {
                    break;
                }
            }
        }

        if x_val.is_some() {
            self.coefs = best_coefs;
            self.intercepts = best_intercepts;
        }

        Ok(())
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        write_usize(buf, self.coefs.len());
        for w in &self.coefs {
            write_matrix(buf, w);
        }
        for b in &self.intercepts {
            write_f64_vec(buf, b);
        }
        write_usize(buf, self.n_features_in);
        write_usize(buf, self.n_outputs);
        let loss_tag = match self.out_loss {
            OutLoss::BinaryCrossEntropy => 0u8,
            OutLoss::CategoricalCrossEntropy => 1u8,
            OutLoss::SquaredError => 2u8,
        };
        buf.push(loss_tag);
        write_usize(buf, self.adam_t);
        write_usize(buf, self.sgd_t);
        write_f64(buf, self.current_lr);
        write_usize(buf, self.n_iter);
        write_f64(buf, self.loss);
        write_f64_vec(buf, &self.loss_curve);
    }

    fn read_from(reader: &mut ByteReader<'_>) -> OchreResult<Self> {
        let n_layers = reader.read_usize()?;
        let mut coefs = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            coefs.push(reader.read_matrix()?);
        }
        let mut intercepts = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            intercepts.push(reader.read_f64_vec()?);
        }
        let n_features_in = reader.read_usize()?;
        let n_outputs = reader.read_usize()?;
        let out_loss = match reader.read_u8()? {
            0 => OutLoss::BinaryCrossEntropy,
            1 => OutLoss::CategoricalCrossEntropy,
            2 => OutLoss::SquaredError,
            _ => return Err(OchreError::CorruptedDitFile),
        };
        let adam_t = reader.read_usize()?;
        let sgd_t = reader.read_usize()?;
        let current_lr = reader.read_f64()?;
        let n_iter = reader.read_usize()?;
        let loss = reader.read_f64()?;
        let loss_curve = reader.read_f64_vec()?;

        let mut m_w = Vec::with_capacity(n_layers);
        let mut v_w = Vec::with_capacity(n_layers);
        let mut m_b = Vec::with_capacity(n_layers);
        let mut v_b = Vec::with_capacity(n_layers);
        let mut vel_w = Vec::with_capacity(n_layers);
        let mut vel_b = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            m_w.push(vec![0.0; coefs[l].data.len()]);
            v_w.push(vec![0.0; coefs[l].data.len()]);
            m_b.push(vec![0.0; intercepts[l].len()]);
            v_b.push(vec![0.0; intercepts[l].len()]);
            vel_w.push(vec![0.0; coefs[l].data.len()]);
            vel_b.push(vec![0.0; intercepts[l].len()]);
        }

        Ok(Self {
            coefs,
            intercepts,
            n_features_in,
            n_outputs,
            out_loss,
            m_w,
            v_w,
            m_b,
            v_b,
            vel_w,
            vel_b,
            adam_t,
            sgd_t,
            current_lr,
            n_iter,
            loss,
            loss_curve,
        })
    }
}

fn compute_r2_matrix(y_true: &Matrix, y_pred: &Matrix) -> f64 {
    let n = y_true.rows;
    let k = y_true.cols;
    if n == 0 || k == 0 {
        return 0.0;
    }
    let mut r2_sum = 0.0;
    for c in 0..k {
        let mut mean = 0.0;
        for r in 0..n {
            mean += y_true.data[r * k + c];
        }
        mean /= n as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for r in 0..n {
            let yt = y_true.data[r * k + c];
            let yp = y_pred.data[r * k + c];
            ss_res += (yt - yp) * (yt - yp);
            ss_tot += (yt - mean) * (yt - mean);
        }
        if ss_tot == 0.0 {
            r2_sum += if ss_res == 0.0 { 1.0 } else { 0.0 };
        } else {
            r2_sum += 1.0 - ss_res / ss_tot;
        }
    }
    r2_sum / (k as f64)
}

fn extract_unique_sorted_classes(y: &[f64]) -> Vec<f64> {
    let mut classes = y.to_vec();
    classes.sort_by(|a, b| a.total_cmp(b));
    classes.dedup();
    classes
}

fn encode_classifier_targets(y: &[f64], classes: &[f64]) -> OchreResult<Matrix> {
    let n = y.len();
    let k = classes.len();
    if k == 2 {
        let mut data = Vec::with_capacity(n);
        for &val in y {
            if val == classes[1] {
                data.push(1.0);
            } else if val == classes[0] {
                data.push(0.0);
            } else {
                return Err(OchreError::InvalidParameter(format!(
                    "Unknown label {} encountered in y",
                    val
                )));
            }
        }
        Ok(Matrix {
            rows: n,
            cols: 1,
            data,
        })
    } else {
        let mut data = vec![0.0; n * k];
        for (i, &val) in y.iter().enumerate() {
            let pos = classes.iter().position(|&c| c == val).ok_or_else(|| {
                OchreError::InvalidParameter(format!("Unknown label {} encountered in y", val))
            })?;
            data[i * k + pos] = 1.0;
        }
        Ok(Matrix {
            rows: n,
            cols: k,
            data,
        })
    }
}

// =========================================================================
// MLPClassifier
// =========================================================================

#[derive(Clone, Debug)]
pub struct MLPClassifier {
    config: MlpConfig,
    device: DeviceType,
    pub device_: Option<String>,
    state: Option<MlpState>,
    classes_: Option<Vec<f64>>,
}

impl MLPClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: Option<String>,
        solver: Option<String>,
        alpha: Option<f64>,
        batch_size: Option<usize>,
        learning_rate: Option<String>,
        learning_rate_init: Option<f64>,
        power_t: Option<f64>,
        max_iter: Option<usize>,
        shuffle: Option<bool>,
        random_state: Option<u64>,
        tol: Option<f64>,
        momentum: Option<f64>,
        nesterovs_momentum: Option<bool>,
        early_stopping: Option<bool>,
        validation_fraction: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
        n_iter_no_change: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let config = MlpConfig {
            hidden_layer_sizes: hidden_layer_sizes.unwrap_or_else(|| vec![100]),
            activation: activation.unwrap_or_else(|| "relu".to_string()),
            solver: solver.unwrap_or_else(|| "adam".to_string()),
            alpha: alpha.unwrap_or(0.0001),
            batch_size,
            learning_rate: learning_rate.unwrap_or_else(|| "constant".to_string()),
            learning_rate_init: learning_rate_init.unwrap_or(0.001),
            power_t: power_t.unwrap_or(0.5),
            max_iter: max_iter.unwrap_or(200),
            shuffle: shuffle.unwrap_or(true),
            random_state,
            tol: tol.unwrap_or(1e-4),
            momentum: momentum.unwrap_or(0.9),
            nesterovs_momentum: nesterovs_momentum.unwrap_or(true),
            early_stopping: early_stopping.unwrap_or(false),
            validation_fraction: validation_fraction.unwrap_or(0.1),
            beta_1: beta_1.unwrap_or(0.9),
            beta_2: beta_2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            n_iter_no_change: n_iter_no_change.unwrap_or(10),
        };
        config.validate()?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            config,
            device: dev,
            device_: device,
            state: None,
            classes_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        self.device_ = device;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let classes = extract_unique_sorted_classes(y);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }
        let (n_outputs, out_loss) = if classes.len() == 2 {
            (1, OutLoss::BinaryCrossEntropy)
        } else {
            (classes.len(), OutLoss::CategoricalCrossEntropy)
        };
        let y_mat = encode_classifier_targets(y, &classes)?;

        let mut rng = SplitMix64::new(self.config.random_state);
        let mut state = MlpState::init(&self.config, x.cols, n_outputs, out_loss, &mut rng)?;
        let backend = get_backend(&self.device);
        state.fit(&backend, &self.config, x, &y_mat, &mut rng)?;

        self.classes_ = Some(classes);
        self.state = Some(state);
        Ok(())
    }

    pub fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &[f64],
        classes: Option<&[f64]>,
    ) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        if self.config.solver == "lbfgs" {
            return Err(OchreError::InvalidParameter(
                "partial_fit is not supported for solver='lbfgs'".to_string(),
            ));
        }

        let mut rng = SplitMix64::new(
            self.config
                .random_state
                .map(|s| s.wrapping_add(self.state.as_ref().map_or(0, |st| st.n_iter as u64))),
        );

        if self.state.is_none() {
            let cls = match classes {
                Some(c) => {
                    validate_slice_finite(c, "classes")?;
                    extract_unique_sorted_classes(c)
                }
                None => extract_unique_sorted_classes(y),
            };
            if cls.len() < 2 {
                return Err(OchreError::InsufficientClasses(cls.len()));
            }
            let (n_outputs, out_loss) = if cls.len() == 2 {
                (1, OutLoss::BinaryCrossEntropy)
            } else {
                (cls.len(), OutLoss::CategoricalCrossEntropy)
            };
            let state = MlpState::init(&self.config, x.cols, n_outputs, out_loss, &mut rng)?;
            self.classes_ = Some(cls);
            self.state = Some(state);
        }

        let cls = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let state = self.state.as_mut().ok_or(OchreError::NotFitted)?;
        if x.cols != state.n_features_in {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: state.n_features_in,
            });
        }

        let y_mat = encode_classifier_targets(y, cls)?;
        let backend = get_backend(&self.device);
        state.run_one_epoch(&backend, &self.config, x, &y_mat, &mut rng)?;
        Ok(())
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let state = self.state.as_ref().ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != state.n_features_in {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: state.n_features_in,
            });
        }
        let backend = get_backend(&self.device);
        let raw = state.forward_predict(&backend, &self.config, x)?;
        if state.out_loss == OutLoss::BinaryCrossEntropy {
            let mut data = Vec::with_capacity(x.rows * 2);
            for &p1 in &raw.data {
                let p1c = p1.clamp(0.0, 1.0);
                data.push(1.0 - p1c);
                data.push(p1c);
            }
            Ok(Matrix {
                rows: x.rows,
                cols: 2,
                data,
            })
        } else {
            Ok(raw)
        }
    }

    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mut proba = self.predict_proba(x)?;
        for v in &mut proba.data {
            *v = v.max(1e-15).ln();
        }
        Ok(proba)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let proba = self.predict_proba(x)?;
        let k = proba.cols;
        let mut out = Vec::with_capacity(proba.rows);
        for r in 0..proba.rows {
            let mut best_c = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for c in 0..k {
                let v = proba.data[r * k + c];
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            out.push(classes[best_c]);
        }
        Ok(out)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_slice_finite(y, "y")?;
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|&(&p, &t)| (p - t).abs() < 1e-12)
            .count();
        Ok((correct as f64) / (y.len() as f64))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MLP_CLASSIFIER);
        self.config.write_to(&mut buf);
        write_opt_str(&mut buf, &self.device_);
        match (&self.state, &self.classes_) {
            (Some(st), Some(cls)) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, cls);
                st.write_to(&mut buf);
            }
            _ => {
                write_bool(&mut buf, false);
            }
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_MLP_CLASSIFIER)?;
        let config = MlpConfig::read_from(&mut reader)?;
        config.validate()?;
        let saved_device = reader.read_opt_str()?;
        let is_fitted = reader.read_bool()?;
        let (state, classes_) = if is_fitted {
            let cls = reader.read_f64_vec()?;
            let st = MlpState::read_from(&mut reader)?;
            (Some(st), Some(cls))
        } else {
            (None, None)
        };
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;
        Ok(Self {
            config,
            device: dev,
            device_,
            state,
            classes_,
        })
    }
}

// =========================================================================
// MLPRegressor
// =========================================================================

#[derive(Clone, Debug)]
pub struct MLPRegressor {
    config: MlpConfig,
    device: DeviceType,
    pub device_: Option<String>,
    state: Option<MlpState>,
}

impl MLPRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: Option<String>,
        solver: Option<String>,
        alpha: Option<f64>,
        batch_size: Option<usize>,
        learning_rate: Option<String>,
        learning_rate_init: Option<f64>,
        power_t: Option<f64>,
        max_iter: Option<usize>,
        shuffle: Option<bool>,
        random_state: Option<u64>,
        tol: Option<f64>,
        momentum: Option<f64>,
        nesterovs_momentum: Option<bool>,
        early_stopping: Option<bool>,
        validation_fraction: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
        n_iter_no_change: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let config = MlpConfig {
            hidden_layer_sizes: hidden_layer_sizes.unwrap_or_else(|| vec![100]),
            activation: activation.unwrap_or_else(|| "relu".to_string()),
            solver: solver.unwrap_or_else(|| "adam".to_string()),
            alpha: alpha.unwrap_or(0.0001),
            batch_size,
            learning_rate: learning_rate.unwrap_or_else(|| "constant".to_string()),
            learning_rate_init: learning_rate_init.unwrap_or(0.001),
            power_t: power_t.unwrap_or(0.5),
            max_iter: max_iter.unwrap_or(200),
            shuffle: shuffle.unwrap_or(true),
            random_state,
            tol: tol.unwrap_or(1e-4),
            momentum: momentum.unwrap_or(0.9),
            nesterovs_momentum: nesterovs_momentum.unwrap_or(true),
            early_stopping: early_stopping.unwrap_or(false),
            validation_fraction: validation_fraction.unwrap_or(0.1),
            beta_1: beta_1.unwrap_or(0.9),
            beta_2: beta_2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            n_iter_no_change: n_iter_no_change.unwrap_or(10),
        };
        config.validate()?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            config,
            device: dev,
            device_: device,
            state: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        self.device_ = device;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_matrix_finite(y, "y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let mut rng = SplitMix64::new(self.config.random_state);
        let mut state = MlpState::init(
            &self.config,
            x.cols,
            y.cols,
            OutLoss::SquaredError,
            &mut rng,
        )?;
        let backend = get_backend(&self.device);
        state.fit(&backend, &self.config, x, y, &mut rng)?;
        self.state = Some(state);
        Ok(())
    }

    pub fn partial_fit(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_matrix_finite(x, "X")?;
        validate_matrix_finite(y, "y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        if self.config.solver == "lbfgs" {
            return Err(OchreError::InvalidParameter(
                "partial_fit is not supported for solver='lbfgs'".to_string(),
            ));
        }

        let mut rng = SplitMix64::new(
            self.config
                .random_state
                .map(|s| s.wrapping_add(self.state.as_ref().map_or(0, |st| st.n_iter as u64))),
        );

        if self.state.is_none() {
            let state = MlpState::init(
                &self.config,
                x.cols,
                y.cols,
                OutLoss::SquaredError,
                &mut rng,
            )?;
            self.state = Some(state);
        }

        let state = self.state.as_mut().ok_or(OchreError::NotFitted)?;
        if x.cols != state.n_features_in {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: state.n_features_in,
            });
        }
        if y.cols != state.n_outputs {
            return Err(OchreError::DimensionMismatch {
                expected_rows: y.rows,
                expected_cols: state.n_outputs,
                actual_rows: y.rows,
                actual_cols: y.cols,
            });
        }

        let backend = get_backend(&self.device);
        state.run_one_epoch(&backend, &self.config, x, y, &mut rng)?;
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Matrix> {
        let state = self.state.as_ref().ok_or(OchreError::NotFitted)?;
        validate_matrix_finite(x, "X")?;
        if x.cols != state.n_features_in {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: state.n_features_in,
            });
        }
        let backend = get_backend(&self.device);
        state.forward_predict(&backend, &self.config, x)
    }

    pub fn score(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        validate_matrix_finite(y, "y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let preds = self.predict(x)?;
        if y.cols != preds.cols {
            return Err(OchreError::DimensionMismatch {
                expected_rows: preds.rows,
                expected_cols: preds.cols,
                actual_rows: y.rows,
                actual_cols: y.cols,
            });
        }
        Ok(compute_r2_matrix(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MLP_REGRESSOR);
        self.config.write_to(&mut buf);
        write_opt_str(&mut buf, &self.device_);
        match &self.state {
            Some(st) => {
                write_bool(&mut buf, true);
                st.write_to(&mut buf);
            }
            None => {
                write_bool(&mut buf, false);
            }
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_MLP_REGRESSOR)?;
        let config = MlpConfig::read_from(&mut reader)?;
        config.validate()?;
        let saved_device = reader.read_opt_str()?;
        let is_fitted = reader.read_bool()?;
        let state = if is_fitted {
            Some(MlpState::read_from(&mut reader)?)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;
        Ok(Self {
            config,
            device: dev,
            device_,
            state,
        })
    }
}

// =========================================================================
// PYO3 WRAPPERS
// =========================================================================

fn y_any_to_matrix(py: Python<'_>, y: &Bound<'_, PyAny>) -> PyResult<Matrix> {
    let buf = PyBuffer::<f64>::get(y)?;
    let shape = buf.shape();
    match shape.len() {
        1 => {
            let data = buffer_to_f64_1d(py, &buf)?;
            let n = data.len();
            if n == 0 {
                return Err(OchreError::EmptyData.into());
            }
            Ok(Matrix::from_vec(n, 1, data)?)
        }
        2 => {
            let mat = buffer_to_pooled_matrix(py, &buf)?;
            Ok(mat.mat.clone())
        }
        d => Err(PyValueError::new_err(format!(
            "Target array y must be 1D or 2D, got {}D ( >_< )",
            d
        ))),
    }
}

#[pyclass(from_py_object, name = "MLPClassifier")]
#[derive(Clone)]
pub struct PyMLPClassifier {
    inner: MLPClassifier,
}

pymethods_with_dit!(PyMLPClassifier, MLPClassifier, {
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes = None,
        activation = None,
        solver = None,
        alpha = None,
        batch_size = None,
        learning_rate = None,
        learning_rate_init = None,
        power_t = None,
        max_iter = None,
        shuffle = None,
        random_state = None,
        tol = None,
        momentum = None,
        nesterovs_momentum = None,
        early_stopping = None,
        validation_fraction = None,
        beta_1 = None,
        beta_2 = None,
        epsilon = None,
        n_iter_no_change = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: Option<String>,
        solver: Option<String>,
        alpha: Option<f64>,
        batch_size: Option<usize>,
        learning_rate: Option<String>,
        learning_rate_init: Option<f64>,
        power_t: Option<f64>,
        max_iter: Option<usize>,
        shuffle: Option<bool>,
        random_state: Option<u64>,
        tol: Option<f64>,
        momentum: Option<f64>,
        nesterovs_momentum: Option<bool>,
        early_stopping: Option<bool>,
        validation_fraction: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
        n_iter_no_change: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = MLPClassifier::new(
            hidden_layer_sizes,
            activation,
            solver,
            alpha,
            batch_size,
            learning_rate,
            learning_rate_init,
            power_t,
            max_iter,
            shuffle,
            random_state,
            tol,
            momentum,
            nesterovs_momentum,
            early_stopping,
            validation_fraction,
            beta_1,
            beta_2,
            epsilon,
            n_iter_no_change,
            device,
        )?;
        Ok(Self { inner })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let target = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&mat, &target)?;
        Ok(())
    }

    #[pyo3(signature = (x, y, classes = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        classes: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let target = buffer_to_f64_1d(py, &y)?;
        self.inner.partial_fit(&mat, &target, classes.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&mat)?;
        matrix_to_pybytes(py, &probs)
    }

    pub fn predict_log_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let log_probs = self.inner.predict_log_proba(&mat)?;
        matrix_to_pybytes(py, &log_probs)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let target = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&mat, &target)?)
    }

    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }

    #[getter]
    #[allow(clippy::type_complexity)]
    pub fn coefs_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
        match &self.inner.state {
            Some(st) => {
                let mut out = Vec::with_capacity(st.coefs.len());
                for w in &st.coefs {
                    out.push(matrix_to_pybytes(py, w)?);
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    #[getter]
    pub fn intercepts_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.state.as_ref().map(|st| st.intercepts.clone())
    }

    #[getter]
    pub fn loss_(&self) -> Option<f64> {
        self.inner.state.as_ref().map(|st| st.loss)
    }

    #[getter]
    pub fn loss_curve_(&self) -> Vec<f64> {
        self.inner
            .state
            .as_ref()
            .map_or_else(Vec::new, |st| st.loss_curve.clone())
    }

    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.state.as_ref().map_or(0, |st| st.n_iter)
    }

    #[getter]
    pub fn n_layers_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.coefs.len() + 1)
    }

    #[getter]
    pub fn n_outputs_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.n_outputs)
    }

    #[getter]
    pub fn out_activation_(&self) -> Option<String> {
        self.inner.state.as_ref().map(|st| match st.out_loss {
            OutLoss::BinaryCrossEntropy => "logistic".to_string(),
            OutLoss::CategoricalCrossEntropy => "softmax".to_string(),
            OutLoss::SquaredError => "identity".to_string(),
        })
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.n_features_in)
    }

    #[getter]
    pub fn hidden_layer_sizes(&self) -> Vec<usize> {
        self.inner.config.hidden_layer_sizes.clone()
    }
    #[getter]
    pub fn activation(&self) -> String {
        self.inner.config.activation.clone()
    }
    #[getter]
    pub fn solver(&self) -> String {
        self.inner.config.solver.clone()
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.config.alpha
    }
    #[getter]
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.config.batch_size
    }
    #[getter]
    pub fn learning_rate(&self) -> String {
        self.inner.config.learning_rate.clone()
    }
    #[getter]
    pub fn learning_rate_init(&self) -> f64 {
        self.inner.config.learning_rate_init
    }
    #[getter]
    pub fn power_t(&self) -> f64 {
        self.inner.config.power_t
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.config.max_iter
    }
    #[getter]
    pub fn shuffle(&self) -> bool {
        self.inner.config.shuffle
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.config.random_state
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.config.tol
    }
    #[getter]
    pub fn momentum(&self) -> f64 {
        self.inner.config.momentum
    }
    #[getter]
    pub fn nesterovs_momentum(&self) -> bool {
        self.inner.config.nesterovs_momentum
    }
    #[getter]
    pub fn early_stopping(&self) -> bool {
        self.inner.config.early_stopping
    }
    #[getter]
    pub fn validation_fraction(&self) -> f64 {
        self.inner.config.validation_fraction
    }
    #[getter]
    pub fn beta_1(&self) -> f64 {
        self.inner.config.beta_1
    }
    #[getter]
    pub fn beta_2(&self) -> f64 {
        self.inner.config.beta_2
    }
    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.config.epsilon
    }
    #[getter]
    pub fn n_iter_no_change(&self) -> usize {
        self.inner.config.n_iter_no_change
    }
});

#[pyclass(from_py_object, name = "MLPRegressor")]
#[derive(Clone)]
pub struct PyMLPRegressor {
    inner: MLPRegressor,
}

pymethods_with_dit!(PyMLPRegressor, MLPRegressor, {
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes = None,
        activation = None,
        solver = None,
        alpha = None,
        batch_size = None,
        learning_rate = None,
        learning_rate_init = None,
        power_t = None,
        max_iter = None,
        shuffle = None,
        random_state = None,
        tol = None,
        momentum = None,
        nesterovs_momentum = None,
        early_stopping = None,
        validation_fraction = None,
        beta_1 = None,
        beta_2 = None,
        epsilon = None,
        n_iter_no_change = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: Option<String>,
        solver: Option<String>,
        alpha: Option<f64>,
        batch_size: Option<usize>,
        learning_rate: Option<String>,
        learning_rate_init: Option<f64>,
        power_t: Option<f64>,
        max_iter: Option<usize>,
        shuffle: Option<bool>,
        random_state: Option<u64>,
        tol: Option<f64>,
        momentum: Option<f64>,
        nesterovs_momentum: Option<bool>,
        early_stopping: Option<bool>,
        validation_fraction: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
        n_iter_no_change: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = MLPRegressor::new(
            hidden_layer_sizes,
            activation,
            solver,
            alpha,
            batch_size,
            learning_rate,
            learning_rate_init,
            power_t,
            max_iter,
            shuffle,
            random_state,
            tol,
            momentum,
            nesterovs_momentum,
            early_stopping,
            validation_fraction,
            beta_1,
            beta_2,
            epsilon,
            n_iter_no_change,
            device,
        )?;
        Ok(Self { inner })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = y_any_to_matrix(py, y)?;
        self.inner.fit(&mat, &y_mat)?;
        Ok(())
    }

    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = y_any_to_matrix(py, y)?;
        self.inner.partial_fit(&mat, &y_mat)?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&mat)?;
        matrix_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = y_any_to_matrix(py, y)?;
        Ok(self.inner.score(&mat, &y_mat)?)
    }

    #[getter]
    #[allow(clippy::type_complexity)]
    pub fn coefs_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<(usize, usize, Bound<'py, PyBytes>)>>> {
        match &self.inner.state {
            Some(st) => {
                let mut out = Vec::with_capacity(st.coefs.len());
                for w in &st.coefs {
                    out.push(matrix_to_pybytes(py, w)?);
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    #[getter]
    pub fn intercepts_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.state.as_ref().map(|st| st.intercepts.clone())
    }

    #[getter]
    pub fn loss_(&self) -> Option<f64> {
        self.inner.state.as_ref().map(|st| st.loss)
    }

    #[getter]
    pub fn loss_curve_(&self) -> Vec<f64> {
        self.inner
            .state
            .as_ref()
            .map_or_else(Vec::new, |st| st.loss_curve.clone())
    }

    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.state.as_ref().map_or(0, |st| st.n_iter)
    }

    #[getter]
    pub fn n_layers_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.coefs.len() + 1)
    }

    #[getter]
    pub fn n_outputs_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.n_outputs)
    }

    #[getter]
    pub fn out_activation_(&self) -> Option<String> {
        self.inner.state.as_ref().map(|_| "identity".to_string())
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.state.as_ref().map(|st| st.n_features_in)
    }

    #[getter]
    pub fn hidden_layer_sizes(&self) -> Vec<usize> {
        self.inner.config.hidden_layer_sizes.clone()
    }
    #[getter]
    pub fn activation(&self) -> String {
        self.inner.config.activation.clone()
    }
    #[getter]
    pub fn solver(&self) -> String {
        self.inner.config.solver.clone()
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.config.alpha
    }
    #[getter]
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.config.batch_size
    }
    #[getter]
    pub fn learning_rate(&self) -> String {
        self.inner.config.learning_rate.clone()
    }
    #[getter]
    pub fn learning_rate_init(&self) -> f64 {
        self.inner.config.learning_rate_init
    }
    #[getter]
    pub fn power_t(&self) -> f64 {
        self.inner.config.power_t
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.config.max_iter
    }
    #[getter]
    pub fn shuffle(&self) -> bool {
        self.inner.config.shuffle
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.config.random_state
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.config.tol
    }
    #[getter]
    pub fn momentum(&self) -> f64 {
        self.inner.config.momentum
    }
    #[getter]
    pub fn nesterovs_momentum(&self) -> bool {
        self.inner.config.nesterovs_momentum
    }
    #[getter]
    pub fn early_stopping(&self) -> bool {
        self.inner.config.early_stopping
    }
    #[getter]
    pub fn validation_fraction(&self) -> f64 {
        self.inner.config.validation_fraction
    }
    #[getter]
    pub fn beta_1(&self) -> f64 {
        self.inner.config.beta_1
    }
    #[getter]
    pub fn beta_2(&self) -> f64 {
        self.inner.config.beta_2
    }
    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.config.epsilon
    }
    #[getter]
    pub fn n_iter_no_change(&self) -> usize {
        self.inner.config.n_iter_no_change
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMLPClassifier>()?;
    m.add_class::<PyMLPRegressor>()?;
    Ok(())
}

// =========================================================================
// RUST UNIT TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_classifier_binary_xor_and_dit() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        let y = vec![0.0, 1.0, 1.0, 0.0];

        let mut clf = MLPClassifier::new(
            Some(vec![16]),
            Some("tanh".to_string()),
            Some("adam".to_string()),
            Some(1e-4),
            None,
            None,
            Some(0.05),
            None,
            Some(500),
            Some(false),
            Some(42),
            Some(1e-6),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("cpu".to_string()),
        )
        .unwrap();
        clf.fit(&x, &y).unwrap();
        assert_eq!(clf.score(&x, &y).unwrap(), 1.0);

        let bytes = clf.to_bytes();
        let loaded = MLPClassifier::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), clf.predict(&x).unwrap());
    }

    #[test]
    fn test_mlp_classifier_multiclass_lbfgs() {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..30 {
            let cls = (i % 3) as f64;
            let offset = cls * 4.0;
            rows.push(vec![offset + 0.1 * ((i % 5) as f64), offset - 0.1]);
            y.push(cls);
        }
        let x = Matrix::from_2d(&rows).unwrap();
        let mut clf = MLPClassifier::new(
            Some(vec![12]),
            Some("relu".to_string()),
            Some("lbfgs".to_string()),
            Some(1e-4),
            None,
            None,
            None,
            None,
            Some(200),
            None,
            Some(7),
            Some(1e-5),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("cpu".to_string()),
        )
        .unwrap();
        clf.fit(&x, &y).unwrap();
        assert!(clf.score(&x, &y).unwrap() > 0.95);
    }

    #[test]
    fn test_mlp_regressor_1d_and_multioutput() {
        let mut x_rows = Vec::new();
        let mut y_rows = Vec::new();
        for i in 0..40 {
            let t = (i as f64) * 0.1 - 2.0;
            x_rows.push(vec![t, t * 0.5]);
            y_rows.push(vec![2.0 * t + 1.0, -t + 3.0]);
        }
        let x = Matrix::from_2d(&x_rows).unwrap();
        let y = Matrix::from_2d(&y_rows).unwrap();

        let mut reg = MLPRegressor::new(
            Some(vec![16, 8]),
            Some("relu".to_string()),
            Some("lbfgs".to_string()),
            Some(1e-4),
            None,
            None,
            None,
            None,
            Some(300),
            None,
            Some(123),
            Some(1e-5),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("cpu".to_string()),
        )
        .unwrap();
        reg.fit(&x, &y).unwrap();
        assert!(reg.score(&x, &y).unwrap() > 0.98);
    }
}
