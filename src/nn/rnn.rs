// OchreML Recurrent & Sequence Neural Network Layers (*≧ω≦*)
// Implements RNN (Tanh/ReLU), GRU (3*H fused gates), and LSTM (4*H fused gates)
// with multi-layer stacking, optional bidirectional execution, full BPTT,
// gradient norm clipping helper, and Tensor autograd support.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, CustomBackward, CustomBackwardFn, GradStore, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::sgemm;
use crate::nn::optim::{clip_grad_norm_, clip_grad_store_norm_};
use crate::nn::tensor::{Tensor, compute_strides};
use std::sync::Arc;

#[inline]
fn quantize_vec(data: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => data,
        DType::BF16 | DType::F16 => TensorStorage::from_f32_slice(&data, dtype).to_f32_vec(),
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn gemm_dispatch(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
    beta: f32,
    device: &DeviceType,
) -> OchreResult<()> {
    if matches!(device, DeviceType::Cpu) {
        sgemm(m, n, k, a, b, c, trans_a, trans_b, 1.0, beta)
    } else {
        let res = get_backend(device).batched_matmul_f32(1, m, n, k, a, b, trans_a, trans_b)?;
        if beta == 0.0 {
            c.copy_from_slice(&res);
        } else {
            for (dst, &src) in c.iter_mut().zip(res.iter()) {
                *dst = beta * (*dst) + src;
            }
        }
        Ok(())
    }
}

/// Clips global L2 norm of flat gradient slices in-place and returns the pre-clip norm.
pub fn clip_rnn_grad_norm_(grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    clip_grad_norm_(grads, max_norm)
}

/// Nonlinearity choice for Elman `RNN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnNonlinearity {
    Tanh,
    Relu,
}

#[derive(Clone, Debug)]
struct SeqCoreConfig {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    gate_mult: usize,
    bias: bool,
    bidirectional: bool,
    dtype: DType,
    device: DeviceType,
}

#[derive(Clone, Debug)]
struct SeqWeights {
    weight_ih: Vec<Tensor>,
    weight_hh: Vec<Tensor>,
    bias_ih: Vec<Option<Tensor>>,
    bias_hh: Vec<Option<Tensor>>,
}

impl SeqWeights {
    fn new(cfg: &SeqCoreConfig, seed: u64) -> OchreResult<Self> {
        if cfg.input_size == 0 || cfg.hidden_size == 0 || cfg.num_layers == 0 {
            return Err(OchreError::InvalidParameter(
                "Sequence layer input_size, hidden_size, and num_layers must be > 0 ( >_< )"
                    .to_string(),
            ));
        }
        let dirs = if cfg.bidirectional { 2 } else { 1 };
        let total_units = cfg.num_layers * dirs;
        let bound = 1.0f32 / (cfg.hidden_size as f32).sqrt();
        let gates_dim = cfg.gate_mult * cfg.hidden_size;

        let mut weight_ih = Vec::with_capacity(total_units);
        let mut weight_hh = Vec::with_capacity(total_units);
        let mut bias_ih = Vec::with_capacity(total_units);
        let mut bias_hh = Vec::with_capacity(total_units);

        let mut s = seed;
        for layer in 0..cfg.num_layers {
            let in_dim = if layer == 0 {
                cfg.input_size
            } else {
                cfg.hidden_size * dirs
            };
            for _dir in 0..dirs {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                weight_ih.push(Tensor::uniform(
                    vec![gates_dim, in_dim],
                    -bound,
                    bound,
                    cfg.dtype,
                    cfg.device.clone(),
                    s,
                    true,
                ));
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                weight_hh.push(Tensor::uniform(
                    vec![gates_dim, cfg.hidden_size],
                    -bound,
                    bound,
                    cfg.dtype,
                    cfg.device.clone(),
                    s,
                    true,
                ));
                if cfg.bias {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    bias_ih.push(Some(Tensor::uniform(
                        vec![gates_dim],
                        -bound,
                        bound,
                        cfg.dtype,
                        cfg.device.clone(),
                        s,
                        true,
                    )));
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    bias_hh.push(Some(Tensor::uniform(
                        vec![gates_dim],
                        -bound,
                        bound,
                        cfg.dtype,
                        cfg.device.clone(),
                        s,
                        true,
                    )));
                } else {
                    bias_ih.push(None);
                    bias_hh.push(None);
                }
            }
        }

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut out = Vec::with_capacity(self.weight_ih.len() * 4);
        for i in 0..self.weight_ih.len() {
            out.push(&self.weight_ih[i]);
            out.push(&self.weight_hh[i]);
            if let Some(ref b) = self.bias_ih[i] {
                out.push(b);
            }
            if let Some(ref b) = self.bias_hh[i] {
                out.push(b);
            }
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::with_capacity(self.weight_ih.len() * 4);
        for (w_ih, (w_hh, (b_ih, b_hh))) in self.weight_ih.iter_mut().zip(
            self.weight_hh
                .iter_mut()
                .zip(self.bias_ih.iter_mut().zip(self.bias_hh.iter_mut())),
        ) {
            out.push(w_ih);
            out.push(w_hh);
            if let Some(b) = b_ih {
                out.push(b);
            }
            if let Some(b) = b_hh {
                out.push(b);
            }
        }
        out
    }

    fn param_node_ids(&self) -> Vec<usize> {
        let mut ids = Vec::new();
        for p in self.parameters() {
            if let Some(id) = p.node_id {
                ids.push(id);
            }
        }
        ids
    }

    fn snapshot(&self) -> WeightsSnapshot {
        WeightsSnapshot {
            weight_ih: self.weight_ih.iter().map(|t| t.data.clone()).collect(),
            weight_hh: self.weight_hh.iter().map(|t| t.data.clone()).collect(),
            bias_ih: self
                .bias_ih
                .iter()
                .map(|o| o.as_ref().map(|t| t.data.clone()))
                .collect(),
            bias_hh: self
                .bias_hh
                .iter()
                .map(|o| o.as_ref().map(|t| t.data.clone()))
                .collect(),
            w_ih_ids: self
                .weight_ih
                .iter()
                .map(|t| if t.requires_grad { t.node_id } else { None })
                .collect(),
            w_hh_ids: self
                .weight_hh
                .iter()
                .map(|t| if t.requires_grad { t.node_id } else { None })
                .collect(),
            b_ih_ids: self
                .bias_ih
                .iter()
                .map(|o| {
                    o.as_ref()
                        .and_then(|t| if t.requires_grad { t.node_id } else { None })
                })
                .collect(),
            b_hh_ids: self
                .bias_hh
                .iter()
                .map(|o| {
                    o.as_ref()
                        .and_then(|t| if t.requires_grad { t.node_id } else { None })
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct WeightsSnapshot {
    weight_ih: Vec<Vec<f32>>,
    weight_hh: Vec<Vec<f32>>,
    bias_ih: Vec<Option<Vec<f32>>>,
    bias_hh: Vec<Option<Vec<f32>>>,
    w_ih_ids: Vec<Option<usize>>,
    w_hh_ids: Vec<Option<usize>>,
    b_ih_ids: Vec<Option<usize>>,
    b_hh_ids: Vec<Option<usize>>,
}

impl WeightsSnapshot {
    fn any_requires_grad(&self) -> bool {
        self.w_ih_ids.iter().any(Option::is_some)
            || self.w_hh_ids.iter().any(Option::is_some)
            || self.b_ih_ids.iter().any(Option::is_some)
            || self.b_hh_ids.iter().any(Option::is_some)
    }
}

/// Raw gradient buffers returned by `backward_raw` for `RNN`, `GRU`, and `LSTM`.
#[derive(Clone, Debug)]
pub struct SeqGradOutput {
    pub d_input: Vec<f32>,
    pub d_h0: Vec<f32>,
    pub d_c0: Option<Vec<f32>>,
    pub d_weight_ih: Vec<Vec<f32>>,
    pub d_weight_hh: Vec<Vec<f32>>,
    pub d_bias_ih: Vec<Option<Vec<f32>>>,
    pub d_bias_hh: Vec<Option<Vec<f32>>>,
}

fn validate_seq_input(x: &Tensor, input_size: usize, name: &str) -> OchreResult<(usize, usize)> {
    if x.shape.len() != 3 {
        return Err(OchreError::InvalidParameter(format!(
            "{name} expects 3D [Batch, SeqLen, InputSize] tensor, got shape {:?} ( >_< )",
            x.shape
        )));
    }
    let (batch, seq_len, feat) = (x.shape[0], x.shape[1], x.shape[2]);
    if batch == 0 || seq_len == 0 {
        return Err(OchreError::EmptyData);
    }
    if feat != input_size {
        return Err(OchreError::InvalidParameter(format!(
            "{name} expected input_size={input_size}, got {feat} ( >_< )"
        )));
    }
    Ok((batch, seq_len))
}

fn validate_state_tensor(
    state: &Tensor,
    expected_units: usize,
    batch: usize,
    hidden_size: usize,
    label: &str,
) -> OchreResult<()> {
    if state.shape != [expected_units, batch, hidden_size] {
        return Err(OchreError::InvalidParameter(format!(
            "{label} expected shape [{expected_units}, {batch}, {hidden_size}], got {:?} ( >_< )",
            state.shape
        )));
    }
    Ok(())
}

/// Records one joint BPTT backward node and thin output/state slice nodes that route gradients into it.
fn record_multi_output_bptt(
    input_ids: Vec<Option<usize>>,
    out_len: usize,
    hn_len: usize,
    cn_len: Option<usize>,
    bptt_fn: CustomBackwardFn,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    let total_joint = out_len + hn_len + cn_len.unwrap_or(0);
    let joint_id = AutogradTape::record(Op::Custom {
        inputs: input_ids,
        backward: CustomBackward(bptt_fn),
    });

    let out_id = AutogradTape::record(Op::Custom {
        inputs: vec![Some(joint_id)],
        backward: CustomBackward(Arc::new(move |d_out: &[f32]| {
            let mut buf = vec![0.0f32; total_joint];
            buf[..out_len].copy_from_slice(d_out);
            Ok(vec![(joint_id, buf)])
        })),
    });

    let hn_id = AutogradTape::record(Op::Custom {
        inputs: vec![Some(joint_id)],
        backward: CustomBackward(Arc::new(move |d_hn: &[f32]| {
            let mut buf = vec![0.0f32; total_joint];
            buf[out_len..out_len + hn_len].copy_from_slice(d_hn);
            Ok(vec![(joint_id, buf)])
        })),
    });

    let cn_id = cn_len.map(|clen| {
        AutogradTape::record(Op::Custom {
            inputs: vec![Some(joint_id)],
            backward: CustomBackward(Arc::new(move |d_cn: &[f32]| {
                let mut buf = vec![0.0f32; total_joint];
                buf[out_len + hn_len..out_len + hn_len + clen].copy_from_slice(d_cn);
                Ok(vec![(joint_id, buf)])
            })),
        })
    });

    (Some(out_id), Some(hn_id), cn_id)
}

#[allow(clippy::too_many_arguments)]
fn write_step_to_batch_seq(
    src: &[f32],
    batch: usize,
    seq_len: usize,
    out_feat: usize,
    dir_offset: usize,
    hidden_size: usize,
    t: usize,
    dst: &mut [f32],
) {
    for b in 0..batch {
        let s_off = b * hidden_size;
        let d_off = (b * seq_len + t) * out_feat + dir_offset;
        dst[d_off..d_off + hidden_size].copy_from_slice(&src[s_off..s_off + hidden_size]);
    }
}

/// Multi-layer Elman RNN with Tanh/ReLU nonlinearity, optional bidirectional execution, and BPTT.
#[derive(Clone, Debug)]
pub struct RNN {
    cfg: SeqCoreConfig,
    weights: SeqWeights,
    pub nonlinearity: RnnNonlinearity,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bias: bool,
    pub bidirectional: bool,
    pub dtype: DType,
    pub device: DeviceType,
}

impl RNN {
    /// Creates an `RNN` with PyTorch-style uniform initialization `U(-1/sqrt(H), 1/sqrt(H))`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        nonlinearity: RnnNonlinearity,
        bias: bool,
        bidirectional: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let cfg = SeqCoreConfig {
            input_size,
            hidden_size,
            num_layers,
            gate_mult: 1,
            bias,
            bidirectional,
            dtype,
            device: device.clone(),
        };
        let weights = SeqWeights::new(&cfg, seed)?;
        Ok(Self {
            cfg,
            weights,
            nonlinearity,
            input_size,
            hidden_size,
            num_layers,
            bias,
            bidirectional,
            dtype,
            device,
        })
    }

    pub fn weight_ih(&self) -> &[Tensor] {
        &self.weights.weight_ih
    }

    pub fn weight_hh(&self) -> &[Tensor] {
        &self.weights.weight_hh
    }

    pub fn bias_ih(&self) -> &[Option<Tensor>] {
        &self.weights.bias_ih
    }

    pub fn bias_hh(&self) -> &[Option<Tensor>] {
        &self.weights.bias_hh
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.weights.parameters()
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.weights.parameters_mut()
    }

    /// Clips global L2 gradient norm of all RNN parameters inside `grads`.
    pub fn clip_grad_store_norm(&self, grads: &mut GradStore, max_norm: f32) -> f32 {
        let ids = self.weights.param_node_ids();
        clip_grad_store_norm_(grads, &ids, max_norm)
    }

    /// Raw forward pass returning `(output [B, S, H*dirs], h_n [L*dirs, B, H])`.
    pub fn forward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        let snap = self.weights.snapshot();
        let (out, hn, _) = rnn_forward_core(
            &self.cfg,
            self.nonlinearity,
            &snap,
            x,
            h0,
            batch,
            seq_len,
            false,
        )?;
        Ok((out, hn))
    }

    /// Raw BPTT backward pass returning input, initial-state, and parameter gradients.
    pub fn backward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        d_output: &[f32],
        d_hn: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<SeqGradOutput> {
        let snap = self.weights.snapshot();
        let (_, _, cache) = rnn_forward_core(
            &self.cfg,
            self.nonlinearity,
            &snap,
            x,
            h0,
            batch,
            seq_len,
            true,
        )?;
        rnn_backward_core(&self.cfg, self.nonlinearity, &snap, &cache, d_output, d_hn)
    }

    /// Autograd-compatible forward pass returning `(output, h_n)`.
    pub fn forward(&self, input: &Tensor, h0: Option<&Tensor>) -> OchreResult<(Tensor, Tensor)> {
        let (batch, seq_len) = validate_seq_input(input, self.input_size, "RNN")?;
        let dirs = if self.bidirectional { 2 } else { 1 };
        let units = self.num_layers * dirs;
        if let Some(h) = h0 {
            validate_state_tensor(h, units, batch, self.hidden_size, "RNN h0")?;
        }

        let snap = self.weights.snapshot();
        let x_id = if input.requires_grad {
            input.node_id
        } else {
            None
        };
        let h0_id = h0.and_then(|h| if h.requires_grad { h.node_id } else { None });
        let need_grad = AutogradTape::is_grad_enabled()
            && (x_id.is_some() || h0_id.is_some() || snap.any_requires_grad());

        let (raw_out, raw_hn, cache) = rnn_forward_core(
            &self.cfg,
            self.nonlinearity,
            &snap,
            &input.data,
            h0.map(|h| h.data.as_slice()),
            batch,
            seq_len,
            need_grad,
        )?;

        let out_shape = vec![batch, seq_len, self.hidden_size * dirs];
        let hn_shape = vec![units, batch, self.hidden_size];
        let out_data = quantize_vec(raw_out, self.dtype);
        let hn_data = quantize_vec(raw_hn, self.dtype);

        let (out_id, hn_id) = if need_grad {
            let out_len = out_data.len();
            let hn_len = hn_data.len();
            let cfg = self.cfg.clone();
            let nonlin = self.nonlinearity;

            let mut all_inputs = vec![x_id, h0_id];
            for u in 0..units {
                all_inputs.push(snap.w_ih_ids[u]);
                all_inputs.push(snap.w_hh_ids[u]);
                all_inputs.push(snap.b_ih_ids[u]);
                all_inputs.push(snap.b_hh_ids[u]);
            }

            let snap_bwd = snap.clone();
            let bptt_fn = Arc::new(move |d_joint: &[f32]| {
                let d_output = &d_joint[..out_len];
                let d_hn = &d_joint[out_len..out_len + hn_len];
                let grad_out = rnn_backward_core(&cfg, nonlin, &snap_bwd, &cache, d_output, d_hn)?;
                let mut res = Vec::new();
                if let Some(id) = x_id {
                    res.push((id, grad_out.d_input));
                }
                if let Some(id) = h0_id {
                    res.push((id, grad_out.d_h0));
                }
                for u in 0..units {
                    if let Some(id) = snap_bwd.w_ih_ids[u] {
                        res.push((id, grad_out.d_weight_ih[u].clone()));
                    }
                    if let Some(id) = snap_bwd.w_hh_ids[u] {
                        res.push((id, grad_out.d_weight_hh[u].clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_ih_ids[u], &grad_out.d_bias_ih[u]) {
                        res.push((id, db.clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_hh_ids[u], &grad_out.d_bias_hh[u]) {
                        res.push((id, db.clone()));
                    }
                }
                Ok(res)
            });

            let (oid, hid, _) =
                record_multi_output_bptt(all_inputs, out_len, hn_len, None, bptt_fn);
            (oid, hid)
        } else {
            (None, None)
        };

        Ok((
            Tensor {
                strides: compute_strides(&out_shape),
                shape: out_shape,
                data: out_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: need_grad,
                node_id: out_id,
            },
            Tensor {
                strides: compute_strides(&hn_shape),
                shape: hn_shape,
                data: hn_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: need_grad,
                node_id: hn_id,
            },
        ))
    }
}

struct RnnCache {
    batch: usize,
    seq_len: usize,
    layer_inputs: Vec<Vec<f32>>,
    unit_h_prev: Vec<Vec<f32>>,
    unit_h_states: Vec<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn rnn_forward_core(
    cfg: &SeqCoreConfig,
    nonlin: RnnNonlinearity,
    snap: &WeightsSnapshot,
    x: &[f32],
    h0: Option<&[f32]>,
    batch: usize,
    seq_len: usize,
    save_cache: bool,
) -> OchreResult<(Vec<f32>, Vec<f32>, RnnCache)> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let bh = batch * h;

    if x.len() != batch * seq_len * cfg.input_size {
        return Err(OchreError::InvalidParameter(
            "RNN input length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(h_init) = h0
        && h_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(
            "RNN h0 length mismatch ( >_< )".to_string(),
        ));
    }

    let mut cur_input = x.to_vec();
    let mut hn = vec![0.0f32; units * bh];
    let mut layer_inputs = Vec::with_capacity(if save_cache { cfg.num_layers } else { 0 });
    let mut unit_h_prev = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_h_states = Vec::with_capacity(if save_cache { units } else { 0 });

    for layer in 0..cfg.num_layers {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut layer_out = vec![0.0f32; batch * seq_len * out_dim];

        if save_cache {
            layer_inputs.push(cur_input.clone());
        }

        let mut x_proj = vec![0.0f32; batch * seq_len * h];
        let mut h_proj = vec![0.0f32; bh];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            gemm_dispatch(
                batch * seq_len,
                h,
                in_dim,
                &cur_input,
                &snap.weight_ih[u],
                &mut x_proj,
                false,
                true,
                0.0,
                &cfg.device,
            )?;

            let mut h_prev = if let Some(h_init) = h0 {
                h_init[u * bh..(u + 1) * bh].to_vec()
            } else {
                vec![0.0f32; bh]
            };

            let mut all_prev = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_states = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };

            let b_ih = snap.bias_ih[u].as_deref();
            let b_hh = snap.bias_hh[u].as_deref();
            let mut b_total = vec![0.0f32; h];
            if let Some(v) = b_ih {
                for (d, &s) in b_total.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }
            if let Some(v) = b_hh {
                for (d, &s) in b_total.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }

            let mut h_cur = vec![0.0f32; bh];
            for step in 0..seq_len {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                if save_cache {
                    all_prev[step * bh..(step + 1) * bh].copy_from_slice(&h_prev);
                }

                gemm_dispatch(
                    batch,
                    h,
                    h,
                    &h_prev,
                    &snap.weight_hh[u],
                    &mut h_proj,
                    false,
                    true,
                    0.0,
                    &cfg.device,
                )?;

                for b in 0..batch {
                    let xp_row = &x_proj[(b * seq_len + t) * h..(b * seq_len + t + 1) * h];
                    let hp_row = &h_proj[b * h..(b + 1) * h];
                    let hc_row = &mut h_cur[b * h..(b + 1) * h];
                    match nonlin {
                        RnnNonlinearity::Tanh => {
                            for j in 0..h {
                                hc_row[j] = (xp_row[j] + hp_row[j] + b_total[j]).tanh();
                            }
                        }
                        RnnNonlinearity::Relu => {
                            for j in 0..h {
                                hc_row[j] = (xp_row[j] + hp_row[j] + b_total[j]).max(0.0);
                            }
                        }
                    }
                }

                write_step_to_batch_seq(
                    &h_cur,
                    batch,
                    seq_len,
                    out_dim,
                    dir * h,
                    h,
                    t,
                    &mut layer_out,
                );
                if save_cache {
                    all_states[step * bh..(step + 1) * bh].copy_from_slice(&h_cur);
                }
                std::mem::swap(&mut h_prev, &mut h_cur);
            }

            hn[u * bh..(u + 1) * bh].copy_from_slice(&h_prev);
            if save_cache {
                unit_h_prev.push(all_prev);
                unit_h_states.push(all_states);
            }
        }

        cur_input = layer_out;
    }

    Ok((
        cur_input,
        hn,
        RnnCache {
            batch,
            seq_len,
            layer_inputs,
            unit_h_prev,
            unit_h_states,
        },
    ))
}

fn rnn_backward_core(
    cfg: &SeqCoreConfig,
    nonlin: RnnNonlinearity,
    snap: &WeightsSnapshot,
    cache: &RnnCache,
    d_output: &[f32],
    d_hn: &[f32],
) -> OchreResult<SeqGradOutput> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;

    let mut d_weight_ih: Vec<Vec<f32>> = snap
        .weight_ih
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_weight_hh: Vec<Vec<f32>> = snap
        .weight_hh
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_bias_ih: Vec<Option<Vec<f32>>> = snap
        .bias_ih
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_bias_hh: Vec<Option<Vec<f32>>> = snap
        .bias_hh
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_h0 = vec![0.0f32; units * bh];
    let mut d_cur_out = d_output.to_vec();

    for layer in (0..cfg.num_layers).rev() {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut d_layer_in = vec![0.0f32; batch * seq_len * in_dim];
        let x_layer = &cache.layer_inputs[layer];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            let mut dh_next = d_hn[u * bh..(u + 1) * bh].to_vec();
            let mut da_all = vec![0.0f32; batch * seq_len * h];
            let mut da_steps = vec![0.0f32; seq_len * bh];

            for step in (0..seq_len).rev() {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                let h_cur = &cache.unit_h_states[u][step * bh..(step + 1) * bh];
                let da_step = &mut da_steps[step * bh..(step + 1) * bh];

                for b in 0..batch {
                    let out_off = (b * seq_len + t) * out_dim + dir * h;
                    let hp_off = b * h;
                    let da_all_row = &mut da_all[(b * seq_len + t) * h..(b * seq_len + t + 1) * h];
                    let da_step_row = &mut da_step[hp_off..hp_off + h];
                    let h_cur_row = &h_cur[hp_off..hp_off + h];
                    let dh_next_row = &dh_next[hp_off..hp_off + h];
                    let d_out_row = &d_cur_out[out_off..out_off + h];

                    match nonlin {
                        RnnNonlinearity::Tanh => {
                            for j in 0..h {
                                let dh = d_out_row[j] + dh_next_row[j];
                                let y = h_cur_row[j];
                                let g = dh * (1.0 - y * y);
                                da_step_row[j] = g;
                                da_all_row[j] = g;
                            }
                        }
                        RnnNonlinearity::Relu => {
                            for j in 0..h {
                                let dh = d_out_row[j] + dh_next_row[j];
                                let g = if h_cur_row[j] > 0.0 { dh } else { 0.0 };
                                da_step_row[j] = g;
                                da_all_row[j] = g;
                            }
                        }
                    }
                }

                gemm_dispatch(
                    batch,
                    h,
                    h,
                    da_step,
                    &snap.weight_hh[u],
                    &mut dh_next,
                    false,
                    false,
                    0.0,
                    &cfg.device,
                )?;
            }

            d_h0[u * bh..(u + 1) * bh].copy_from_slice(&dh_next);

            if d_bias_ih[u].is_some() || d_bias_hh[u].is_some() {
                let mut db_acc = vec![0.0f32; h];
                for row in da_steps.chunks_exact(h) {
                    for (d, &v) in db_acc.iter_mut().zip(row.iter()) {
                        *d += v;
                    }
                }
                if let Some(ref mut db) = d_bias_ih[u] {
                    for (d, &v) in db.iter_mut().zip(db_acc.iter()) {
                        *d += v;
                    }
                }
                if let Some(ref mut db) = d_bias_hh[u] {
                    for (d, &v) in db.iter_mut().zip(db_acc.iter()) {
                        *d += v;
                    }
                }
            }

            gemm_dispatch(
                h,
                h,
                seq_len * batch,
                &da_steps,
                &cache.unit_h_prev[u],
                &mut d_weight_hh[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                h,
                in_dim,
                batch * seq_len,
                &da_all,
                x_layer,
                &mut d_weight_ih[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                batch * seq_len,
                in_dim,
                h,
                &da_all,
                &snap.weight_ih[u],
                &mut d_layer_in,
                false,
                false,
                1.0,
                &cfg.device,
            )?;
        }

        d_cur_out = d_layer_in;
    }

    Ok(SeqGradOutput {
        d_input: d_cur_out,
        d_h0,
        d_c0: None,
        d_weight_ih,
        d_weight_hh,
        d_bias_ih,
        d_bias_hh,
    })
}

/// Multi-layer Gated Recurrent Unit (`GRU`) with fused `3 * hidden_size` gate projections `(r, z, n)`,
/// optional bidirectional execution, and full BPTT.
#[derive(Clone, Debug)]
pub struct GRU {
    cfg: SeqCoreConfig,
    weights: SeqWeights,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bias: bool,
    pub bidirectional: bool,
    pub dtype: DType,
    pub device: DeviceType,
}

impl GRU {
    /// Creates a `GRU` with uniform initialization `U(-1/sqrt(H), 1/sqrt(H))`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        bidirectional: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let cfg = SeqCoreConfig {
            input_size,
            hidden_size,
            num_layers,
            gate_mult: 3,
            bias,
            bidirectional,
            dtype,
            device: device.clone(),
        };
        let weights = SeqWeights::new(&cfg, seed)?;
        Ok(Self {
            cfg,
            weights,
            input_size,
            hidden_size,
            num_layers,
            bias,
            bidirectional,
            dtype,
            device,
        })
    }

    pub fn weight_ih(&self) -> &[Tensor] {
        &self.weights.weight_ih
    }

    pub fn weight_hh(&self) -> &[Tensor] {
        &self.weights.weight_hh
    }

    pub fn bias_ih(&self) -> &[Option<Tensor>] {
        &self.weights.bias_ih
    }

    pub fn bias_hh(&self) -> &[Option<Tensor>] {
        &self.weights.bias_hh
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.weights.parameters()
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.weights.parameters_mut()
    }

    /// Clips global L2 gradient norm of all GRU parameters inside `grads`.
    pub fn clip_grad_store_norm(&self, grads: &mut GradStore, max_norm: f32) -> f32 {
        let ids = self.weights.param_node_ids();
        clip_grad_store_norm_(grads, &ids, max_norm)
    }

    /// Raw forward pass returning `(output [B, S, H*dirs], h_n [L*dirs, B, H])`.
    pub fn forward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        let snap = self.weights.snapshot();
        let (out, hn, _) = gru_forward_core(&self.cfg, &snap, x, h0, batch, seq_len, false)?;
        Ok((out, hn))
    }

    /// Raw BPTT backward pass returning input, initial-state, and parameter gradients.
    pub fn backward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        d_output: &[f32],
        d_hn: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<SeqGradOutput> {
        let snap = self.weights.snapshot();
        let (_, _, cache) = gru_forward_core(&self.cfg, &snap, x, h0, batch, seq_len, true)?;
        gru_backward_core(&self.cfg, &snap, &cache, d_output, d_hn)
    }

    /// Autograd-compatible forward pass returning `(output, h_n)`.
    pub fn forward(&self, input: &Tensor, h0: Option<&Tensor>) -> OchreResult<(Tensor, Tensor)> {
        let (batch, seq_len) = validate_seq_input(input, self.input_size, "GRU")?;
        let dirs = if self.bidirectional { 2 } else { 1 };
        let units = self.num_layers * dirs;
        if let Some(h) = h0 {
            validate_state_tensor(h, units, batch, self.hidden_size, "GRU h0")?;
        }

        let snap = self.weights.snapshot();
        let x_id = if input.requires_grad {
            input.node_id
        } else {
            None
        };
        let h0_id = h0.and_then(|h| if h.requires_grad { h.node_id } else { None });
        let need_grad = AutogradTape::is_grad_enabled()
            && (x_id.is_some() || h0_id.is_some() || snap.any_requires_grad());

        let (raw_out, raw_hn, cache) = gru_forward_core(
            &self.cfg,
            &snap,
            &input.data,
            h0.map(|h| h.data.as_slice()),
            batch,
            seq_len,
            need_grad,
        )?;

        let out_shape = vec![batch, seq_len, self.hidden_size * dirs];
        let hn_shape = vec![units, batch, self.hidden_size];
        let out_data = quantize_vec(raw_out, self.dtype);
        let hn_data = quantize_vec(raw_hn, self.dtype);

        let (out_id, hn_id) = if need_grad {
            let out_len = out_data.len();
            let hn_len = hn_data.len();
            let cfg = self.cfg.clone();

            let mut all_inputs = vec![x_id, h0_id];
            for u in 0..units {
                all_inputs.push(snap.w_ih_ids[u]);
                all_inputs.push(snap.w_hh_ids[u]);
                all_inputs.push(snap.b_ih_ids[u]);
                all_inputs.push(snap.b_hh_ids[u]);
            }

            let snap_bwd = snap.clone();
            let bptt_fn = Arc::new(move |d_joint: &[f32]| {
                let d_output = &d_joint[..out_len];
                let d_hn = &d_joint[out_len..out_len + hn_len];
                let grad_out = gru_backward_core(&cfg, &snap_bwd, &cache, d_output, d_hn)?;
                let mut res = Vec::new();
                if let Some(id) = x_id {
                    res.push((id, grad_out.d_input));
                }
                if let Some(id) = h0_id {
                    res.push((id, grad_out.d_h0));
                }
                for u in 0..units {
                    if let Some(id) = snap_bwd.w_ih_ids[u] {
                        res.push((id, grad_out.d_weight_ih[u].clone()));
                    }
                    if let Some(id) = snap_bwd.w_hh_ids[u] {
                        res.push((id, grad_out.d_weight_hh[u].clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_ih_ids[u], &grad_out.d_bias_ih[u]) {
                        res.push((id, db.clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_hh_ids[u], &grad_out.d_bias_hh[u]) {
                        res.push((id, db.clone()));
                    }
                }
                Ok(res)
            });

            let (oid, hid, _) =
                record_multi_output_bptt(all_inputs, out_len, hn_len, None, bptt_fn);
            (oid, hid)
        } else {
            (None, None)
        };

        Ok((
            Tensor {
                strides: compute_strides(&out_shape),
                shape: out_shape,
                data: out_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: need_grad,
                node_id: out_id,
            },
            Tensor {
                strides: compute_strides(&hn_shape),
                shape: hn_shape,
                data: hn_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: need_grad,
                node_id: hn_id,
            },
        ))
    }
}

struct GruCache {
    batch: usize,
    seq_len: usize,
    layer_inputs: Vec<Vec<f32>>,
    unit_h_prev: Vec<Vec<f32>>,
    unit_r: Vec<Vec<f32>>,
    unit_z: Vec<Vec<f32>>,
    unit_n: Vec<Vec<f32>>,
    unit_gh_n: Vec<Vec<f32>>,
}

fn gru_forward_core(
    cfg: &SeqCoreConfig,
    snap: &WeightsSnapshot,
    x: &[f32],
    h0: Option<&[f32]>,
    batch: usize,
    seq_len: usize,
    save_cache: bool,
) -> OchreResult<(Vec<f32>, Vec<f32>, GruCache)> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let h3 = 3 * h;
    let bh = batch * h;
    let bh3 = batch * h3;

    if x.len() != batch * seq_len * cfg.input_size {
        return Err(OchreError::InvalidParameter(
            "GRU input length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(h_init) = h0
        && h_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(
            "GRU h0 length mismatch ( >_< )".to_string(),
        ));
    }

    let mut cur_input = x.to_vec();
    let mut hn = vec![0.0f32; units * bh];
    let mut layer_inputs = Vec::with_capacity(if save_cache { cfg.num_layers } else { 0 });
    let mut unit_h_prev = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_r = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_z = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_n = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_gh_n = Vec::with_capacity(if save_cache { units } else { 0 });

    for layer in 0..cfg.num_layers {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut layer_out = vec![0.0f32; batch * seq_len * out_dim];

        if save_cache {
            layer_inputs.push(cur_input.clone());
        }

        let mut gi_all = vec![0.0f32; batch * seq_len * h3];
        let mut gh = vec![0.0f32; bh3];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            gemm_dispatch(
                batch * seq_len,
                h3,
                in_dim,
                &cur_input,
                &snap.weight_ih[u],
                &mut gi_all,
                false,
                true,
                0.0,
                &cfg.device,
            )?;

            let mut h_prev = if let Some(h_init) = h0 {
                h_init[u * bh..(u + 1) * bh].to_vec()
            } else {
                vec![0.0f32; bh]
            };

            let mut all_prev = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_r = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_z = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_n = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_gh_n = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };

            let b_ih = snap.bias_ih[u].as_deref();
            let b_hh = snap.bias_hh[u].as_deref();
            let mut br_sum = vec![0.0f32; h];
            let mut bz_sum = vec![0.0f32; h];
            let mut bi_n = vec![0.0f32; h];
            let mut bh_n = vec![0.0f32; h];
            if let Some(v) = b_ih {
                for j in 0..h {
                    br_sum[j] += v[j];
                    bz_sum[j] += v[h + j];
                    bi_n[j] = v[2 * h + j];
                }
            }
            if let Some(v) = b_hh {
                for j in 0..h {
                    br_sum[j] += v[j];
                    bz_sum[j] += v[h + j];
                    bh_n[j] = v[2 * h + j];
                }
            }

            let mut h_cur = vec![0.0f32; bh];
            for step in 0..seq_len {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                if save_cache {
                    all_prev[step * bh..(step + 1) * bh].copy_from_slice(&h_prev);
                }

                gemm_dispatch(
                    batch,
                    h3,
                    h,
                    &h_prev,
                    &snap.weight_hh[u],
                    &mut gh,
                    false,
                    true,
                    0.0,
                    &cfg.device,
                )?;

                for b in 0..batch {
                    let gi_row = &gi_all[(b * seq_len + t) * h3..(b * seq_len + t + 1) * h3];
                    let gh_row = &gh[b * h3..(b + 1) * h3];
                    let hp_row = &h_prev[b * h..(b + 1) * h];
                    let hc_row = &mut h_cur[b * h..(b + 1) * h];

                    if save_cache {
                        let base = step * bh + b * h;
                        let r_dst = &mut all_r[base..base + h];
                        let z_dst = &mut all_z[base..base + h];
                        let n_dst = &mut all_n[base..base + h];
                        let ghn_dst = &mut all_gh_n[base..base + h];
                        for j in 0..h {
                            let r = sigmoid_f32(gi_row[j] + gh_row[j] + br_sum[j]);
                            let z = sigmoid_f32(gi_row[h + j] + gh_row[h + j] + bz_sum[j]);
                            let gh_n_val = gh_row[2 * h + j] + bh_n[j];
                            let n = (gi_row[2 * h + j] + bi_n[j] + r * gh_n_val).tanh();
                            hc_row[j] = (1.0 - z) * n + z * hp_row[j];
                            r_dst[j] = r;
                            z_dst[j] = z;
                            n_dst[j] = n;
                            ghn_dst[j] = gh_n_val;
                        }
                    } else {
                        for j in 0..h {
                            let r = sigmoid_f32(gi_row[j] + gh_row[j] + br_sum[j]);
                            let z = sigmoid_f32(gi_row[h + j] + gh_row[h + j] + bz_sum[j]);
                            let gh_n_val = gh_row[2 * h + j] + bh_n[j];
                            let n = (gi_row[2 * h + j] + bi_n[j] + r * gh_n_val).tanh();
                            hc_row[j] = (1.0 - z) * n + z * hp_row[j];
                        }
                    }
                }

                write_step_to_batch_seq(
                    &h_cur,
                    batch,
                    seq_len,
                    out_dim,
                    dir * h,
                    h,
                    t,
                    &mut layer_out,
                );
                std::mem::swap(&mut h_prev, &mut h_cur);
            }

            hn[u * bh..(u + 1) * bh].copy_from_slice(&h_prev);
            if save_cache {
                unit_h_prev.push(all_prev);
                unit_r.push(all_r);
                unit_z.push(all_z);
                unit_n.push(all_n);
                unit_gh_n.push(all_gh_n);
            }
        }

        cur_input = layer_out;
    }

    Ok((
        cur_input,
        hn,
        GruCache {
            batch,
            seq_len,
            layer_inputs,
            unit_h_prev,
            unit_r,
            unit_z,
            unit_n,
            unit_gh_n,
        },
    ))
}

fn gru_backward_core(
    cfg: &SeqCoreConfig,
    snap: &WeightsSnapshot,
    cache: &GruCache,
    d_output: &[f32],
    d_hn: &[f32],
) -> OchreResult<SeqGradOutput> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let h3 = 3 * h;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;
    let bh3 = batch * h3;

    let mut d_weight_ih: Vec<Vec<f32>> = snap
        .weight_ih
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_weight_hh: Vec<Vec<f32>> = snap
        .weight_hh
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_bias_ih: Vec<Option<Vec<f32>>> = snap
        .bias_ih
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_bias_hh: Vec<Option<Vec<f32>>> = snap
        .bias_hh
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_h0 = vec![0.0f32; units * bh];
    let mut d_cur_out = d_output.to_vec();

    for layer in (0..cfg.num_layers).rev() {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut d_layer_in = vec![0.0f32; batch * seq_len * in_dim];
        let x_layer = &cache.layer_inputs[layer];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            let mut dh_next = d_hn[u * bh..(u + 1) * bh].to_vec();
            let mut dgi_all = vec![0.0f32; batch * seq_len * h3];
            let mut dgh_steps = vec![0.0f32; seq_len * bh3];
            let mut dh_from_gh = vec![0.0f32; bh];

            for step in (0..seq_len).rev() {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                let off = step * bh;
                let h_prev = &cache.unit_h_prev[u][off..off + bh];
                let r_s = &cache.unit_r[u][off..off + bh];
                let z_s = &cache.unit_z[u][off..off + bh];
                let n_s = &cache.unit_n[u][off..off + bh];
                let ghn_s = &cache.unit_gh_n[u][off..off + bh];
                let dgh = &mut dgh_steps[step * bh3..(step + 1) * bh3];

                for b in 0..batch {
                    let out_off = (b * seq_len + t) * out_dim + dir * h;
                    let hp_off = b * h;
                    let g_off = b * h3;
                    let dgi_row = &mut dgi_all[(b * seq_len + t) * h3..(b * seq_len + t + 1) * h3];
                    let dgh_row = &mut dgh[g_off..g_off + h3];

                    for j in 0..h {
                        let dh = d_cur_out[out_off + j] + dh_next[hp_off + j];
                        let r = r_s[hp_off + j];
                        let z = z_s[hp_off + j];
                        let n = n_s[hp_off + j];
                        let hp = h_prev[hp_off + j];
                        let ghn = ghn_s[hp_off + j];

                        let dn = dh * (1.0 - z);
                        let dz = dh * (hp - n);
                        let da_n = dn * (1.0 - n * n);
                        let dr = da_n * ghn;
                        let da_z = dz * z * (1.0 - z);
                        let da_r = dr * r * (1.0 - r);

                        dgi_row[j] = da_r;
                        dgi_row[h + j] = da_z;
                        dgi_row[2 * h + j] = da_n;

                        dgh_row[j] = da_r;
                        dgh_row[h + j] = da_z;
                        dgh_row[2 * h + j] = da_n * r;

                        dh_next[hp_off + j] = dh * z;
                    }
                }

                gemm_dispatch(
                    batch,
                    h,
                    h3,
                    dgh,
                    &snap.weight_hh[u],
                    &mut dh_from_gh,
                    false,
                    false,
                    0.0,
                    &cfg.device,
                )?;
                for (d, &s) in dh_next.iter_mut().zip(dh_from_gh.iter()) {
                    *d += s;
                }
            }

            d_h0[u * bh..(u + 1) * bh].copy_from_slice(&dh_next);

            if let Some(ref mut db) = d_bias_ih[u] {
                for row in dgi_all.chunks_exact(h3) {
                    for (d, &v) in db.iter_mut().zip(row.iter()) {
                        *d += v;
                    }
                }
            }
            if let Some(ref mut db) = d_bias_hh[u] {
                for row in dgh_steps.chunks_exact(h3) {
                    for (d, &v) in db.iter_mut().zip(row.iter()) {
                        *d += v;
                    }
                }
            }

            gemm_dispatch(
                h3,
                h,
                seq_len * batch,
                &dgh_steps,
                &cache.unit_h_prev[u],
                &mut d_weight_hh[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                h3,
                in_dim,
                batch * seq_len,
                &dgi_all,
                x_layer,
                &mut d_weight_ih[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                batch * seq_len,
                in_dim,
                h3,
                &dgi_all,
                &snap.weight_ih[u],
                &mut d_layer_in,
                false,
                false,
                1.0,
                &cfg.device,
            )?;
        }

        d_cur_out = d_layer_in;
    }

    Ok(SeqGradOutput {
        d_input: d_cur_out,
        d_h0,
        d_c0: None,
        d_weight_ih,
        d_weight_hh,
        d_bias_ih,
        d_bias_hh,
    })
}

/// Multi-layer Long Short-Term Memory (`LSTM`) with fused `4 * hidden_size` gate projections `(i, f, g, o)`,
/// optional bidirectional execution, and full BPTT.
#[derive(Clone, Debug)]
pub struct LSTM {
    cfg: SeqCoreConfig,
    weights: SeqWeights,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bias: bool,
    pub bidirectional: bool,
    pub dtype: DType,
    pub device: DeviceType,
}

impl LSTM {
    /// Creates an `LSTM` with uniform initialization `U(-1/sqrt(H), 1/sqrt(H))`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        bidirectional: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let cfg = SeqCoreConfig {
            input_size,
            hidden_size,
            num_layers,
            gate_mult: 4,
            bias,
            bidirectional,
            dtype,
            device: device.clone(),
        };
        let weights = SeqWeights::new(&cfg, seed)?;
        Ok(Self {
            cfg,
            weights,
            input_size,
            hidden_size,
            num_layers,
            bias,
            bidirectional,
            dtype,
            device,
        })
    }

    pub fn weight_ih(&self) -> &[Tensor] {
        &self.weights.weight_ih
    }

    pub fn weight_hh(&self) -> &[Tensor] {
        &self.weights.weight_hh
    }

    pub fn bias_ih(&self) -> &[Option<Tensor>] {
        &self.weights.bias_ih
    }

    pub fn bias_hh(&self) -> &[Option<Tensor>] {
        &self.weights.bias_hh
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.weights.parameters()
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.weights.parameters_mut()
    }

    /// Clips global L2 gradient norm of all LSTM parameters inside `grads`.
    pub fn clip_grad_store_norm(&self, grads: &mut GradStore, max_norm: f32) -> f32 {
        let ids = self.weights.param_node_ids();
        clip_grad_store_norm_(grads, &ids, max_norm)
    }

    /// Raw forward pass returning `(output [B, S, H*dirs], h_n [L*dirs, B, H], c_n [L*dirs, B, H])`.
    pub fn forward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        c0: Option<&[f32]>,
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let snap = self.weights.snapshot();
        let (out, hn, cn, _) =
            lstm_forward_core(&self.cfg, &snap, x, h0, c0, batch, seq_len, false)?;
        Ok((out, hn, cn))
    }

    /// Raw BPTT backward pass returning input, initial-state `(h0, c0)`, and parameter gradients.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_raw(
        &self,
        x: &[f32],
        h0: Option<&[f32]>,
        c0: Option<&[f32]>,
        d_output: &[f32],
        d_hn: &[f32],
        d_cn: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> OchreResult<SeqGradOutput> {
        let snap = self.weights.snapshot();
        let (_, _, _, cache) =
            lstm_forward_core(&self.cfg, &snap, x, h0, c0, batch, seq_len, true)?;
        lstm_backward_core(&self.cfg, &snap, &cache, d_output, d_hn, d_cn)
    }

    /// Autograd-compatible forward pass returning `(output, (h_n, c_n))`.
    pub fn forward(
        &self,
        input: &Tensor,
        hx: Option<(&Tensor, &Tensor)>,
    ) -> OchreResult<(Tensor, (Tensor, Tensor))> {
        let (batch, seq_len) = validate_seq_input(input, self.input_size, "LSTM")?;
        let dirs = if self.bidirectional { 2 } else { 1 };
        let units = self.num_layers * dirs;
        if let Some((h, c)) = hx {
            validate_state_tensor(h, units, batch, self.hidden_size, "LSTM h0")?;
            validate_state_tensor(c, units, batch, self.hidden_size, "LSTM c0")?;
        }

        let snap = self.weights.snapshot();
        let x_id = if input.requires_grad {
            input.node_id
        } else {
            None
        };
        let h0_id = hx.and_then(|(h, _)| if h.requires_grad { h.node_id } else { None });
        let c0_id = hx.and_then(|(_, c)| if c.requires_grad { c.node_id } else { None });
        let need_grad = AutogradTape::is_grad_enabled()
            && (x_id.is_some() || h0_id.is_some() || c0_id.is_some() || snap.any_requires_grad());

        let (raw_out, raw_hn, raw_cn, cache) = lstm_forward_core(
            &self.cfg,
            &snap,
            &input.data,
            hx.map(|(h, _)| h.data.as_slice()),
            hx.map(|(_, c)| c.data.as_slice()),
            batch,
            seq_len,
            need_grad,
        )?;

        let out_shape = vec![batch, seq_len, self.hidden_size * dirs];
        let state_shape = vec![units, batch, self.hidden_size];
        let out_data = quantize_vec(raw_out, self.dtype);
        let hn_data = quantize_vec(raw_hn, self.dtype);
        let cn_data = quantize_vec(raw_cn, self.dtype);

        let (out_id, hn_id, cn_id) = if need_grad {
            let out_len = out_data.len();
            let hn_len = hn_data.len();
            let cn_len = cn_data.len();
            let cfg = self.cfg.clone();

            let mut all_inputs = vec![x_id, h0_id, c0_id];
            for u in 0..units {
                all_inputs.push(snap.w_ih_ids[u]);
                all_inputs.push(snap.w_hh_ids[u]);
                all_inputs.push(snap.b_ih_ids[u]);
                all_inputs.push(snap.b_hh_ids[u]);
            }

            let snap_bwd = snap.clone();
            let bptt_fn = Arc::new(move |d_joint: &[f32]| {
                let d_output = &d_joint[..out_len];
                let d_hn = &d_joint[out_len..out_len + hn_len];
                let d_cn = &d_joint[out_len + hn_len..out_len + hn_len + cn_len];
                let grad_out = lstm_backward_core(&cfg, &snap_bwd, &cache, d_output, d_hn, d_cn)?;
                let mut res = Vec::new();
                if let Some(id) = x_id {
                    res.push((id, grad_out.d_input));
                }
                if let Some(id) = h0_id {
                    res.push((id, grad_out.d_h0));
                }
                if let (Some(id), Some(dc0)) = (c0_id, grad_out.d_c0) {
                    res.push((id, dc0));
                }
                for u in 0..units {
                    if let Some(id) = snap_bwd.w_ih_ids[u] {
                        res.push((id, grad_out.d_weight_ih[u].clone()));
                    }
                    if let Some(id) = snap_bwd.w_hh_ids[u] {
                        res.push((id, grad_out.d_weight_hh[u].clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_ih_ids[u], &grad_out.d_bias_ih[u]) {
                        res.push((id, db.clone()));
                    }
                    if let (Some(id), Some(db)) = (snap_bwd.b_hh_ids[u], &grad_out.d_bias_hh[u]) {
                        res.push((id, db.clone()));
                    }
                }
                Ok(res)
            });

            record_multi_output_bptt(all_inputs, out_len, hn_len, Some(cn_len), bptt_fn)
        } else {
            (None, None, None)
        };

        Ok((
            Tensor {
                strides: compute_strides(&out_shape),
                shape: out_shape,
                data: out_data,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: need_grad,
                node_id: out_id,
            },
            (
                Tensor {
                    strides: compute_strides(&state_shape),
                    shape: state_shape.clone(),
                    data: hn_data,
                    dtype: self.dtype,
                    device: self.device.clone(),
                    requires_grad: need_grad,
                    node_id: hn_id,
                },
                Tensor {
                    strides: compute_strides(&state_shape),
                    shape: state_shape,
                    data: cn_data,
                    dtype: self.dtype,
                    device: self.device.clone(),
                    requires_grad: need_grad,
                    node_id: cn_id,
                },
            ),
        ))
    }
}

struct LstmCache {
    batch: usize,
    seq_len: usize,
    layer_inputs: Vec<Vec<f32>>,
    unit_h_prev: Vec<Vec<f32>>,
    unit_c_prev: Vec<Vec<f32>>,
    unit_i: Vec<Vec<f32>>,
    unit_f: Vec<Vec<f32>>,
    unit_g: Vec<Vec<f32>>,
    unit_o: Vec<Vec<f32>>,
    unit_tc: Vec<Vec<f32>>,
}

type LstmForwardOut = (Vec<f32>, Vec<f32>, Vec<f32>, LstmCache);

#[allow(clippy::too_many_arguments)]
fn lstm_forward_core(
    cfg: &SeqCoreConfig,
    snap: &WeightsSnapshot,
    x: &[f32],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
    batch: usize,
    seq_len: usize,
    save_cache: bool,
) -> OchreResult<LstmForwardOut> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let h4 = 4 * h;
    let bh = batch * h;
    let bh4 = batch * h4;

    if x.len() != batch * seq_len * cfg.input_size {
        return Err(OchreError::InvalidParameter(
            "LSTM input length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(h_init) = h0
        && h_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(
            "LSTM h0 length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(c_init) = c0
        && c_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(
            "LSTM c0 length mismatch ( >_< )".to_string(),
        ));
    }

    let mut cur_input = x.to_vec();
    let mut hn = vec![0.0f32; units * bh];
    let mut cn = vec![0.0f32; units * bh];

    let mut layer_inputs = Vec::with_capacity(if save_cache { cfg.num_layers } else { 0 });
    let mut unit_h_prev = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_c_prev = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_i = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_f = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_g = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_o = Vec::with_capacity(if save_cache { units } else { 0 });
    let mut unit_tc = Vec::with_capacity(if save_cache { units } else { 0 });

    for layer in 0..cfg.num_layers {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut layer_out = vec![0.0f32; batch * seq_len * out_dim];

        if save_cache {
            layer_inputs.push(cur_input.clone());
        }

        let mut gi_all = vec![0.0f32; batch * seq_len * h4];
        let mut gh = vec![0.0f32; bh4];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            gemm_dispatch(
                batch * seq_len,
                h4,
                in_dim,
                &cur_input,
                &snap.weight_ih[u],
                &mut gi_all,
                false,
                true,
                0.0,
                &cfg.device,
            )?;

            let mut h_prev = if let Some(h_init) = h0 {
                h_init[u * bh..(u + 1) * bh].to_vec()
            } else {
                vec![0.0f32; bh]
            };
            let mut c_prev = if let Some(c_init) = c0 {
                c_init[u * bh..(u + 1) * bh].to_vec()
            } else {
                vec![0.0f32; bh]
            };

            let mut all_hp = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_cp = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_i = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_f = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_g = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_o = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };
            let mut all_tc = if save_cache {
                vec![0.0f32; seq_len * bh]
            } else {
                Vec::new()
            };

            let b_ih = snap.bias_ih[u].as_deref();
            let b_hh = snap.bias_hh[u].as_deref();
            let mut b_total = vec![0.0f32; h4];
            if let Some(v) = b_ih {
                for (d, &s) in b_total.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }
            if let Some(v) = b_hh {
                for (d, &s) in b_total.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }

            let mut h_cur = vec![0.0f32; bh];
            let mut c_cur = vec![0.0f32; bh];
            for step in 0..seq_len {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                if save_cache {
                    all_hp[step * bh..(step + 1) * bh].copy_from_slice(&h_prev);
                    all_cp[step * bh..(step + 1) * bh].copy_from_slice(&c_prev);
                }

                gemm_dispatch(
                    batch,
                    h4,
                    h,
                    &h_prev,
                    &snap.weight_hh[u],
                    &mut gh,
                    false,
                    true,
                    0.0,
                    &cfg.device,
                )?;

                for b in 0..batch {
                    let gi_row = &gi_all[(b * seq_len + t) * h4..(b * seq_len + t + 1) * h4];
                    let gh_row = &gh[b * h4..(b + 1) * h4];
                    let cp_row = &c_prev[b * h..(b + 1) * h];
                    let hc_row = &mut h_cur[b * h..(b + 1) * h];
                    let cc_row = &mut c_cur[b * h..(b + 1) * h];

                    if save_cache {
                        let base = step * bh + b * h;
                        let i_dst = &mut all_i[base..base + h];
                        let f_dst = &mut all_f[base..base + h];
                        let g_dst = &mut all_g[base..base + h];
                        let o_dst = &mut all_o[base..base + h];
                        let tc_dst = &mut all_tc[base..base + h];
                        for j in 0..h {
                            let iv = sigmoid_f32(gi_row[j] + gh_row[j] + b_total[j]);
                            let fv = sigmoid_f32(gi_row[h + j] + gh_row[h + j] + b_total[h + j]);
                            let gv =
                                (gi_row[2 * h + j] + gh_row[2 * h + j] + b_total[2 * h + j]).tanh();
                            let ov = sigmoid_f32(
                                gi_row[3 * h + j] + gh_row[3 * h + j] + b_total[3 * h + j],
                            );

                            let c_new = fv * cp_row[j] + iv * gv;
                            let tc = c_new.tanh();
                            cc_row[j] = c_new;
                            hc_row[j] = ov * tc;

                            i_dst[j] = iv;
                            f_dst[j] = fv;
                            g_dst[j] = gv;
                            o_dst[j] = ov;
                            tc_dst[j] = tc;
                        }
                    } else {
                        for j in 0..h {
                            let iv = sigmoid_f32(gi_row[j] + gh_row[j] + b_total[j]);
                            let fv = sigmoid_f32(gi_row[h + j] + gh_row[h + j] + b_total[h + j]);
                            let gv =
                                (gi_row[2 * h + j] + gh_row[2 * h + j] + b_total[2 * h + j]).tanh();
                            let ov = sigmoid_f32(
                                gi_row[3 * h + j] + gh_row[3 * h + j] + b_total[3 * h + j],
                            );

                            let c_new = fv * cp_row[j] + iv * gv;
                            let tc = c_new.tanh();
                            cc_row[j] = c_new;
                            hc_row[j] = ov * tc;
                        }
                    }
                }

                write_step_to_batch_seq(
                    &h_cur,
                    batch,
                    seq_len,
                    out_dim,
                    dir * h,
                    h,
                    t,
                    &mut layer_out,
                );
                std::mem::swap(&mut h_prev, &mut h_cur);
                std::mem::swap(&mut c_prev, &mut c_cur);
            }

            hn[u * bh..(u + 1) * bh].copy_from_slice(&h_prev);
            cn[u * bh..(u + 1) * bh].copy_from_slice(&c_prev);
            if save_cache {
                unit_h_prev.push(all_hp);
                unit_c_prev.push(all_cp);
                unit_i.push(all_i);
                unit_f.push(all_f);
                unit_g.push(all_g);
                unit_o.push(all_o);
                unit_tc.push(all_tc);
            }
        }

        cur_input = layer_out;
    }

    Ok((
        cur_input,
        hn,
        cn,
        LstmCache {
            batch,
            seq_len,
            layer_inputs,
            unit_h_prev,
            unit_c_prev,
            unit_i,
            unit_f,
            unit_g,
            unit_o,
            unit_tc,
        },
    ))
}

fn lstm_backward_core(
    cfg: &SeqCoreConfig,
    snap: &WeightsSnapshot,
    cache: &LstmCache,
    d_output: &[f32],
    d_hn: &[f32],
    d_cn: &[f32],
) -> OchreResult<SeqGradOutput> {
    let dirs = if cfg.bidirectional { 2 } else { 1 };
    let units = cfg.num_layers * dirs;
    let h = cfg.hidden_size;
    let h4 = 4 * h;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;
    let bh4 = batch * h4;

    let mut d_weight_ih: Vec<Vec<f32>> = snap
        .weight_ih
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_weight_hh: Vec<Vec<f32>> = snap
        .weight_hh
        .iter()
        .map(|w| vec![0.0f32; w.len()])
        .collect();
    let mut d_bias_ih: Vec<Option<Vec<f32>>> = snap
        .bias_ih
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_bias_hh: Vec<Option<Vec<f32>>> = snap
        .bias_hh
        .iter()
        .map(|o| o.as_ref().map(|b| vec![0.0f32; b.len()]))
        .collect();
    let mut d_h0 = vec![0.0f32; units * bh];
    let mut d_c0 = vec![0.0f32; units * bh];
    let mut d_cur_out = d_output.to_vec();

    for layer in (0..cfg.num_layers).rev() {
        let in_dim = if layer == 0 { cfg.input_size } else { h * dirs };
        let out_dim = h * dirs;
        let mut d_layer_in = vec![0.0f32; batch * seq_len * in_dim];
        let x_layer = &cache.layer_inputs[layer];

        for dir in 0..dirs {
            let u = layer * dirs + dir;
            let mut dh_next = d_hn[u * bh..(u + 1) * bh].to_vec();
            let mut dc_next = d_cn[u * bh..(u + 1) * bh].to_vec();
            let mut dgates_all = vec![0.0f32; batch * seq_len * h4];
            let mut dgates_steps = vec![0.0f32; seq_len * bh4];

            for step in (0..seq_len).rev() {
                let t = if dir == 0 { step } else { seq_len - 1 - step };
                let off = step * bh;
                let c_prev = &cache.unit_c_prev[u][off..off + bh];
                let i_s = &cache.unit_i[u][off..off + bh];
                let f_s = &cache.unit_f[u][off..off + bh];
                let g_s = &cache.unit_g[u][off..off + bh];
                let o_s = &cache.unit_o[u][off..off + bh];
                let tc_s = &cache.unit_tc[u][off..off + bh];
                let dgates_step = &mut dgates_steps[step * bh4..(step + 1) * bh4];

                for b in 0..batch {
                    let out_off = (b * seq_len + t) * out_dim + dir * h;
                    let hp_off = b * h;
                    let g_off = b * h4;
                    let dg_row =
                        &mut dgates_all[(b * seq_len + t) * h4..(b * seq_len + t + 1) * h4];
                    let dg_step_row = &mut dgates_step[g_off..g_off + h4];

                    for j in 0..h {
                        let dh = d_cur_out[out_off + j] + dh_next[hp_off + j];
                        let iv = i_s[hp_off + j];
                        let fv = f_s[hp_off + j];
                        let gv = g_s[hp_off + j];
                        let ov = o_s[hp_off + j];
                        let tc = tc_s[hp_off + j];
                        let cp = c_prev[hp_off + j];

                        let do_v = dh * tc;
                        let dc = dh * ov * (1.0 - tc * tc) + dc_next[hp_off + j];

                        let da_i = dc * gv * iv * (1.0 - iv);
                        let da_f = dc * cp * fv * (1.0 - fv);
                        let da_g = dc * iv * (1.0 - gv * gv);
                        let da_o = do_v * ov * (1.0 - ov);

                        dg_row[j] = da_i;
                        dg_row[h + j] = da_f;
                        dg_row[2 * h + j] = da_g;
                        dg_row[3 * h + j] = da_o;

                        dg_step_row[j] = da_i;
                        dg_step_row[h + j] = da_f;
                        dg_step_row[2 * h + j] = da_g;
                        dg_step_row[3 * h + j] = da_o;

                        dc_next[hp_off + j] = dc * fv;
                    }
                }

                gemm_dispatch(
                    batch,
                    h,
                    h4,
                    dgates_step,
                    &snap.weight_hh[u],
                    &mut dh_next,
                    false,
                    false,
                    0.0,
                    &cfg.device,
                )?;
            }

            d_h0[u * bh..(u + 1) * bh].copy_from_slice(&dh_next);
            d_c0[u * bh..(u + 1) * bh].copy_from_slice(&dc_next);

            if d_bias_ih[u].is_some() || d_bias_hh[u].is_some() {
                let mut db_acc = vec![0.0f32; h4];
                for row in dgates_steps.chunks_exact(h4) {
                    for (d, &v) in db_acc.iter_mut().zip(row.iter()) {
                        *d += v;
                    }
                }
                if let Some(ref mut db) = d_bias_ih[u] {
                    for (d, &v) in db.iter_mut().zip(db_acc.iter()) {
                        *d += v;
                    }
                }
                if let Some(ref mut db) = d_bias_hh[u] {
                    for (d, &v) in db.iter_mut().zip(db_acc.iter()) {
                        *d += v;
                    }
                }
            }

            gemm_dispatch(
                h4,
                h,
                seq_len * batch,
                &dgates_steps,
                &cache.unit_h_prev[u],
                &mut d_weight_hh[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                h4,
                in_dim,
                batch * seq_len,
                &dgates_all,
                x_layer,
                &mut d_weight_ih[u],
                true,
                false,
                1.0,
                &cfg.device,
            )?;
            gemm_dispatch(
                batch * seq_len,
                in_dim,
                h4,
                &dgates_all,
                &snap.weight_ih[u],
                &mut d_layer_in,
                false,
                false,
                1.0,
                &cfg.device,
            )?;
        }

        d_cur_out = d_layer_in;
    }

    Ok(SeqGradOutput {
        d_input: d_cur_out,
        d_h0,
        d_c0: Some(d_c0),
        d_weight_ih,
        d_weight_hh,
        d_bias_ih,
        d_bias_hh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::with_no_grad;

    fn check_slice_fd<F>(base: &[f32], analytic: &[f32], f: F, tol: f32, label: &str)
    where
        F: Fn(&[f32]) -> f32,
    {
        let eps = 1e-3f32;
        for idx in 0..base.len() {
            let mut p = base.to_vec();
            let mut m = base.to_vec();
            p[idx] += eps;
            m[idx] -= eps;
            let num = (f(&p) - f(&m)) / (2.0 * eps);
            assert!(
                (analytic[idx] - num).abs() < tol,
                "{label}[{idx}] mismatch: analytic={} vs numeric={num}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn test_rnn_multilayer_bidirectional_tanh_and_relu_finite_diff() {
        for nonlin in [RnnNonlinearity::Tanh, RnnNonlinearity::Relu] {
            AutogradTape::clear();
            let rnn =
                RNN::new(3, 4, 2, nonlin, true, true, DType::F32, DeviceType::Cpu, 11).unwrap();
            let x = Tensor::randn_with(
                vec![2, 3, 3],
                0.2,
                0.5,
                DType::F32,
                DeviceType::Cpu,
                12,
                true,
            );
            let h0 = Tensor::randn_with(
                vec![4, 2, 4],
                0.1,
                0.4,
                DType::F32,
                DeviceType::Cpu,
                13,
                true,
            );

            let (out, hn) = rnn.forward(&x, Some(&h0)).unwrap();
            assert_eq!(out.shape, vec![2, 3, 8]);
            assert_eq!(hn.shape, vec![4, 2, 4]);

            // Combine both output and h_n in the loss to test joint multi-output BPTT
            let co: Vec<f32> = (0..out.numel())
                .map(|i| ((i as f32) * 0.23).sin())
                .collect();
            let ch: Vec<f32> = (0..hn.numel()).map(|i| ((i as f32) * 0.31).cos()).collect();
            let cot = Tensor::new(
                co.clone(),
                out.shape.clone(),
                DType::F32,
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let cht = Tensor::new(
                ch.clone(),
                hn.shape.clone(),
                DType::F32,
                DeviceType::Cpu,
                false,
            )
            .unwrap();

            let loss = out
                .mul(&cot)
                .unwrap()
                .sum(None)
                .unwrap()
                .add(&hn.mul(&cht).unwrap().sum(None).unwrap())
                .unwrap();
            let grads = loss.backward().unwrap();

            let dx = grads.get(x.node_id.unwrap()).unwrap();
            let dh0 = grads.get(h0.node_id.unwrap()).unwrap();
            let dw0 = grads.get(rnn.weight_ih()[0].node_id.unwrap()).unwrap();
            let dwh0 = grads.get(rnn.weight_hh()[0].node_id.unwrap()).unwrap();
            let dbi0 = grads
                .get(rnn.bias_ih()[0].as_ref().unwrap().node_id.unwrap())
                .unwrap();

            let tol = if nonlin == RnnNonlinearity::Relu {
                3e-2
            } else {
                2e-2
            };
            check_slice_fd(
                &x.data,
                dx,
                |xd| {
                    with_no_grad(|| {
                        let xt = Tensor::new(
                            xd.to_vec(),
                            x.shape.clone(),
                            DType::F32,
                            DeviceType::Cpu,
                            false,
                        )
                        .unwrap();
                        let (o, h) = rnn.forward(&xt, Some(&h0)).unwrap();
                        let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                        let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                        lo + lh
                    })
                },
                tol,
                "RNN dx",
            );

            check_slice_fd(
                &h0.data,
                dh0,
                |hd| {
                    with_no_grad(|| {
                        let ht = Tensor::new(
                            hd.to_vec(),
                            h0.shape.clone(),
                            DType::F32,
                            DeviceType::Cpu,
                            false,
                        )
                        .unwrap();
                        let (o, h) = rnn.forward(&x, Some(&ht)).unwrap();
                        let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                        let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                        lo + lh
                    })
                },
                tol,
                "RNN dh0",
            );

            check_slice_fd(
                &rnn.weight_ih()[0].data,
                dw0,
                |wd| {
                    with_no_grad(|| {
                        let mut r2 = rnn.clone();
                        r2.weights.weight_ih[0].data = wd.to_vec();
                        let (o, h) = r2.forward(&x, Some(&h0)).unwrap();
                        let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                        let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                        lo + lh
                    })
                },
                tol,
                "RNN weight_ih[0]",
            );

            check_slice_fd(
                &rnn.weight_hh()[0].data,
                dwh0,
                |wd| {
                    with_no_grad(|| {
                        let mut r2 = rnn.clone();
                        r2.weights.weight_hh[0].data = wd.to_vec();
                        let (o, h) = r2.forward(&x, Some(&h0)).unwrap();
                        let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                        let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                        lo + lh
                    })
                },
                tol,
                "RNN weight_hh[0]",
            );

            check_slice_fd(
                &rnn.bias_ih()[0].as_ref().unwrap().data,
                dbi0,
                |bd| {
                    with_no_grad(|| {
                        let mut r2 = rnn.clone();
                        r2.weights.bias_ih[0].as_mut().unwrap().data = bd.to_vec();
                        let (o, h) = r2.forward(&x, Some(&h0)).unwrap();
                        let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                        let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                        lo + lh
                    })
                },
                tol,
                "RNN bias_ih[0]",
            );
        }
    }

    #[test]
    fn test_gru_multilayer_bidirectional_finite_diff() {
        AutogradTape::clear();
        let gru = GRU::new(3, 3, 2, true, true, DType::F32, DeviceType::Cpu, 31).unwrap();
        let x = Tensor::randn_with(
            vec![2, 3, 3],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            32,
            true,
        );
        let h0 = Tensor::randn_with(
            vec![4, 2, 3],
            0.0,
            0.4,
            DType::F32,
            DeviceType::Cpu,
            33,
            true,
        );

        let (out, hn) = gru.forward(&x, Some(&h0)).unwrap();
        assert_eq!(out.shape, vec![2, 3, 6]);
        assert_eq!(hn.shape, vec![4, 2, 3]);

        let co: Vec<f32> = (0..out.numel())
            .map(|i| ((i as f32) * 0.17).sin())
            .collect();
        let ch: Vec<f32> = (0..hn.numel()).map(|i| ((i as f32) * 0.29).cos()).collect();
        let cot = Tensor::new(
            co.clone(),
            out.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let cht = Tensor::new(
            ch.clone(),
            hn.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let loss = out
            .mul(&cot)
            .unwrap()
            .sum(None)
            .unwrap()
            .add(&hn.mul(&cht).unwrap().sum(None).unwrap())
            .unwrap();
        let mut grads = loss.backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap().to_vec();
        let dh0 = grads.get(h0.node_id.unwrap()).unwrap().to_vec();
        let dw_ih2 = grads
            .get(gru.weight_ih()[2].node_id.unwrap())
            .unwrap()
            .to_vec();
        let dw_hh1 = grads
            .get(gru.weight_hh()[1].node_id.unwrap())
            .unwrap()
            .to_vec();
        let db_hh0 = grads
            .get(gru.bias_hh()[0].as_ref().unwrap().node_id.unwrap())
            .unwrap()
            .to_vec();

        check_slice_fd(
            &x.data,
            &dx,
            |xd| {
                with_no_grad(|| {
                    let xt = Tensor::new(
                        xd.to_vec(),
                        x.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let (o, h) = gru.forward(&xt, Some(&h0)).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh
                })
            },
            2e-2,
            "GRU dx",
        );

        check_slice_fd(
            &h0.data,
            &dh0,
            |hd| {
                with_no_grad(|| {
                    let ht = Tensor::new(
                        hd.to_vec(),
                        h0.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let (o, h) = gru.forward(&x, Some(&ht)).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh
                })
            },
            2e-2,
            "GRU dh0",
        );

        check_slice_fd(
            &gru.weight_ih()[2].data,
            &dw_ih2,
            |wd| {
                with_no_grad(|| {
                    let mut g2 = gru.clone();
                    g2.weights.weight_ih[2].data = wd.to_vec();
                    let (o, h) = g2.forward(&x, Some(&h0)).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh
                })
            },
            2e-2,
            "GRU weight_ih[2]",
        );

        check_slice_fd(
            &gru.weight_hh()[1].data,
            &dw_hh1,
            |wd| {
                with_no_grad(|| {
                    let mut g2 = gru.clone();
                    g2.weights.weight_hh[1].data = wd.to_vec();
                    let (o, h) = g2.forward(&x, Some(&h0)).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh
                })
            },
            2e-2,
            "GRU weight_hh[1]",
        );

        check_slice_fd(
            &gru.bias_hh()[0].as_ref().unwrap().data,
            &db_hh0,
            |bd| {
                with_no_grad(|| {
                    let mut g2 = gru.clone();
                    g2.weights.bias_hh[0].as_mut().unwrap().data = bd.to_vec();
                    let (o, h) = g2.forward(&x, Some(&h0)).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh
                })
            },
            2e-2,
            "GRU bias_hh[0]",
        );

        // Also verify grad norm clipping helper
        let pre_norm = gru.clip_grad_store_norm(&mut grads, 0.1);
        assert!(pre_norm > 0.1);
    }

    #[test]
    fn test_lstm_multilayer_bidirectional_and_raw_parity_finite_diff() {
        AutogradTape::clear();
        let lstm = LSTM::new(3, 3, 2, true, true, DType::F32, DeviceType::Cpu, 51).unwrap();
        let x = Tensor::randn_with(
            vec![2, 3, 3],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            52,
            true,
        );
        let h0 = Tensor::randn_with(
            vec![4, 2, 3],
            0.0,
            0.4,
            DType::F32,
            DeviceType::Cpu,
            53,
            true,
        );
        let c0 = Tensor::randn_with(
            vec![4, 2, 3],
            0.0,
            0.4,
            DType::F32,
            DeviceType::Cpu,
            54,
            true,
        );

        let (out, (hn, cn)) = lstm.forward(&x, Some((&h0, &c0))).unwrap();
        assert_eq!(out.shape, vec![2, 3, 6]);
        assert_eq!(hn.shape, vec![4, 2, 3]);
        assert_eq!(cn.shape, vec![4, 2, 3]);

        // Raw forward parity
        let (ro, rh, rc) = lstm
            .forward_raw(&x.data, Some(&h0.data), Some(&c0.data), 2, 3)
            .unwrap();
        assert_eq!(ro, out.data);
        assert_eq!(rh, hn.data);
        assert_eq!(rc, cn.data);

        let co: Vec<f32> = (0..out.numel())
            .map(|i| ((i as f32) * 0.19).sin())
            .collect();
        let ch: Vec<f32> = (0..hn.numel()).map(|i| ((i as f32) * 0.27).cos()).collect();
        let cc: Vec<f32> = (0..cn.numel()).map(|i| ((i as f32) * 0.33).sin()).collect();
        let cot = Tensor::new(
            co.clone(),
            out.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let cht = Tensor::new(
            ch.clone(),
            hn.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let cct = Tensor::new(
            cc.clone(),
            cn.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let loss = out
            .mul(&cot)
            .unwrap()
            .sum(None)
            .unwrap()
            .add(&hn.mul(&cht).unwrap().sum(None).unwrap())
            .unwrap()
            .add(&cn.mul(&cct).unwrap().sum(None).unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dh0 = grads.get(h0.node_id.unwrap()).unwrap();
        let dc0 = grads.get(c0.node_id.unwrap()).unwrap();
        let dw_ih3 = grads.get(lstm.weight_ih()[3].node_id.unwrap()).unwrap();
        let dw_hh0 = grads.get(lstm.weight_hh()[0].node_id.unwrap()).unwrap();
        let db_ih1 = grads
            .get(lstm.bias_ih()[1].as_ref().unwrap().node_id.unwrap())
            .unwrap();

        // Raw backward parity
        let raw_g = lstm
            .backward_raw(&x.data, Some(&h0.data), Some(&c0.data), &co, &ch, &cc, 2, 3)
            .unwrap();
        for (&a, &b) in dx.iter().zip(raw_g.d_input.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        check_slice_fd(
            &x.data,
            dx,
            |xd| {
                with_no_grad(|| {
                    let xt = Tensor::new(
                        xd.to_vec(),
                        x.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let (o, (h, c)) = lstm.forward(&xt, Some((&h0, &c0))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM dx",
        );

        check_slice_fd(
            &h0.data,
            dh0,
            |hd| {
                with_no_grad(|| {
                    let ht = Tensor::new(
                        hd.to_vec(),
                        h0.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let (o, (h, c)) = lstm.forward(&x, Some((&ht, &c0))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM dh0",
        );

        check_slice_fd(
            &c0.data,
            dc0,
            |cd| {
                with_no_grad(|| {
                    let ct = Tensor::new(
                        cd.to_vec(),
                        c0.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let (o, (h, c)) = lstm.forward(&x, Some((&h0, &ct))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM dc0",
        );

        check_slice_fd(
            &lstm.weight_ih()[3].data,
            dw_ih3,
            |wd| {
                with_no_grad(|| {
                    let mut l2 = lstm.clone();
                    l2.weights.weight_ih[3].data = wd.to_vec();
                    let (o, (h, c)) = l2.forward(&x, Some((&h0, &c0))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM weight_ih[3]",
        );

        check_slice_fd(
            &lstm.weight_hh()[0].data,
            dw_hh0,
            |wd| {
                with_no_grad(|| {
                    let mut l2 = lstm.clone();
                    l2.weights.weight_hh[0].data = wd.to_vec();
                    let (o, (h, c)) = l2.forward(&x, Some((&h0, &c0))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM weight_hh[0]",
        );

        check_slice_fd(
            &lstm.bias_ih()[1].as_ref().unwrap().data,
            db_ih1,
            |bd| {
                with_no_grad(|| {
                    let mut l2 = lstm.clone();
                    l2.weights.bias_ih[1].as_mut().unwrap().data = bd.to_vec();
                    let (o, (h, c)) = l2.forward(&x, Some((&h0, &c0))).unwrap();
                    let lo: f32 = o.data.iter().zip(co.iter()).map(|(&a, &b)| a * b).sum();
                    let lh: f32 = h.data.iter().zip(ch.iter()).map(|(&a, &b)| a * b).sum();
                    let lc: f32 = c.data.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    lo + lh + lc
                })
            },
            2e-2,
            "LSTM bias_ih[1]",
        );
    }

    #[test]
    fn test_clip_rnn_grad_norm_helper() {
        let mut g1 = vec![3.0f32, 0.0];
        let mut g2 = vec![0.0f32, 4.0];
        let norm = clip_rnn_grad_norm_(&mut [g1.as_mut_slice(), g2.as_mut_slice()], 2.5);
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((g1[0] - 1.5).abs() < 1e-4);
        assert!((g2[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_seq_models_default_zero_state_and_training() {
        use crate::nn::losses::{MSELoss, Reduction};
        use crate::nn::optim::Adam;

        let mut lstm = LSTM::new(2, 4, 1, true, false, DType::F32, DeviceType::Cpu, 71).unwrap();
        let gru = GRU::new(2, 4, 1, true, false, DType::F32, DeviceType::Cpu, 72).unwrap();
        let rnn = RNN::new(
            2,
            4,
            1,
            RnnNonlinearity::Tanh,
            true,
            false,
            DType::F32,
            DeviceType::Cpu,
            73,
        )
        .unwrap();

        let x = Tensor::randn_with(
            vec![2, 4, 2],
            0.0,
            0.8,
            DType::F32,
            DeviceType::Cpu,
            74,
            false,
        );
        let target = Tensor::ones(vec![2, 4, 4], DType::F32, DeviceType::Cpu);

        let (go, _) = gru.forward(&x, None).unwrap();
        let (ro, _) = rnn.forward(&x, None).unwrap();
        assert_eq!(go.shape, vec![2, 4, 4]);
        assert_eq!(ro.shape, vec![2, 4, 4]);

        let mse = MSELoss::new(Reduction::Mean);
        let mut opt = Adam::with_lr(0.05, 0.0).unwrap();
        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;
        for step in 0..25 {
            AutogradTape::clear();
            let (out, _) = lstm.forward(&x, None).unwrap();
            let loss = mse.forward_tensor(&out, &target).unwrap();
            if step == 0 {
                first_loss = loss.data[0];
            }
            last_loss = loss.data[0];
            let grads = loss.backward().unwrap();
            let mut params = lstm.parameters_mut();
            opt.step(&mut params, &grads).unwrap();
        }
        assert!(
            last_loss < first_loss * 0.5,
            "LSTM training loss should decrease: first={first_loss}, last={last_loss}"
        );
    }
}
