// OchreML Transformer, SwiGLU, FeedForward & CausalLM Module (*≧ω≦*)
// Implements SwiGLU, GELU FeedForward, TransformerEncoder/Decoder, and LLaMA-style CausalLM
// with GQA, RoPE, KVCache generation, and DIT\x01 tag 101 serialization.

use crate::backend::get_backend;
#[cfg(feature = "cuda-dynamic")]
use crate::backends::cuda_nn::{self, CudaDeviceBuf};
use crate::device::DeviceType;
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::attention::{
    KVCache, MultiHeadAttention, RoPE, flash_attention_backward_into,
    flash_attention_forward_strided_into, transpose_bhsd_to_bshd_into, transpose_bshd_to_bhsd_into,
};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::sgemm;
use crate::nn::layers::{Embedding, Linear};
use crate::nn::norm::{LayerNorm, RMSNorm};
use crate::nn::simd_math::{
    simd_adamw_step_f32, simd_exp_sub_max_sum, simd_silu_bwd_into, simd_silu_fwd_into,
};
use crate::nn::tensor::{FastRng, Tensor};
use std::cell::RefCell;

#[derive(Default)]
struct SwiGluScratch {
    gate: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Default)]
struct FfnScratch {
    hidden: Vec<f32>,
}

thread_local! {
    static SWIGLU_SCRATCH: RefCell<SwiGluScratch> = RefCell::new(SwiGluScratch::default());
    static FFN_SCRATCH: RefCell<FfnScratch> = RefCell::new(FfnScratch::default());
}

#[inline]
fn ensure_len(buf: &mut Vec<f32>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
}

#[inline]
fn rmsnorm_into_cpu(x: &[f32], gamma: &[f32], rows: usize, cols: usize, eps: f32, out: &mut [f32]) {
    let inv_cols = 1.0f32 / (cols as f32);
    for r in 0..rows {
        let x_row = &x[r * cols..(r + 1) * cols];
        let o_row = &mut out[r * cols..(r + 1) * cols];
        let chunks = cols / 4;
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        for i in 0..chunks {
            let off = i * 4;
            let a = x_row[off];
            let b = x_row[off + 1];
            let c = x_row[off + 2];
            let d = x_row[off + 3];
            s0 += a * a;
            s1 += b * b;
            s2 += c * c;
            s3 += d * d;
        }
        let mut sum_sq = (s0 + s1) + (s2 + s3);
        for &v in &x_row[chunks * 4..] {
            sum_sq += v * v;
        }
        let inv_rms = 1.0f32 / (sum_sq * inv_cols + eps).sqrt();
        for ((dst, &xv), &gv) in o_row.iter_mut().zip(x_row.iter()).zip(gamma.iter()) {
            *dst = xv * inv_rms * gv;
        }
    }
}

#[inline]
fn rmsnorm_fwd_with_rstd_into_cpu(
    x: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    out: &mut [f32],
    rstd: &mut [f32],
) {
    let inv_cols = 1.0f32 / (cols as f32);
    for r in 0..rows {
        let x_row = &x[r * cols..(r + 1) * cols];
        let o_row = &mut out[r * cols..(r + 1) * cols];
        let chunks = cols / 4;
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        for i in 0..chunks {
            let off = i * 4;
            let a = x_row[off];
            let b = x_row[off + 1];
            let c = x_row[off + 2];
            let d = x_row[off + 3];
            s0 += a * a;
            s1 += b * b;
            s2 += c * c;
            s3 += d * d;
        }
        let mut sum_sq = (s0 + s1) + (s2 + s3);
        for &v in &x_row[chunks * 4..] {
            sum_sq += v * v;
        }
        let inv_rms = 1.0f32 / (sum_sq * inv_cols + eps).sqrt();
        rstd[r] = inv_rms;
        for ((dst, &xv), &gv) in o_row.iter_mut().zip(x_row.iter()).zip(gamma.iter()) {
            *dst = xv * inv_rms * gv;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm_bwd_into_cpu(
    dy: &[f32],
    x: &[f32],
    gamma: &[f32],
    rstd: &[f32],
    rows: usize,
    cols: usize,
    dx: &mut [f32],
    dgamma: &mut [f32],
    dgamma_accum: &mut [f64],
) {
    let inv_cols = 1.0f64 / (cols as f64);
    dgamma_accum.fill(0.0);
    for r in 0..rows {
        let dy_row = &dy[r * cols..(r + 1) * cols];
        let x_row = &x[r * cols..(r + 1) * cols];
        let dx_row = &mut dx[r * cols..(r + 1) * cols];
        let rs = rstd[r] as f64;
        let mut dot = 0.0f64;
        for j in 0..cols {
            let dyv = dy_row[j] as f64;
            let g = gamma[j] as f64;
            let x_hat = (x_row[j] as f64) * rs;
            dot += (dyv * g) * x_hat;
            dgamma_accum[j] += dyv * x_hat;
        }
        for j in 0..cols {
            let g = gamma[j] as f64;
            let dy_g = (dy_row[j] as f64) * g;
            let x_hat = (x_row[j] as f64) * rs;
            dx_row[j] = (rs * (dy_g - x_hat * dot * inv_cols)) as f32;
        }
    }
    for (dst, &acc) in dgamma.iter_mut().zip(dgamma_accum.iter()) {
        *dst = acc as f32;
    }
}

fn cross_entropy_grad_into_cpu(
    logits: &[f32],
    targets: &[usize],
    rows: usize,
    cols: usize,
    d_logits: &mut [f32],
) -> f32 {
    let inv_rows = 1.0f64 / (rows as f64);
    let inv_rows_f32 = inv_rows as f32;
    let mut total_loss = 0.0f64;
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let drow = &mut d_logits[r * cols..(r + 1) * cols];
        let tgt = targets[r];
        let (max_v, sum_exp) = simd_exp_sub_max_sum(row, drow);
        let sum_exp_d = sum_exp as f64;
        let log_z = (max_v as f64) + sum_exp_d.ln();
        total_loss += log_z - (row[tgt] as f64);
        let scale = (inv_rows / sum_exp_d) as f32;
        for dst in drow.iter_mut() {
            *dst *= scale;
        }
        drow[tgt] -= inv_rows_f32;
    }
    (total_loss * inv_rows) as f32
}

fn clip_grad_vecs_inplace(grads: &mut [Vec<f32>], max_norm: f32) {
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
}

/// Binary serialization tag for `CausalLM` in `.dit` files.
pub const TAG_NN_TRANSFORMER_LM: u8 = 101;

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
#[derive(Clone, Debug)]
pub struct SwiGLU {
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub d_model: usize,
    pub hidden_dim: usize,
}

impl SwiGLU {
    /// Creates a `SwiGLU` block with projections `[d_model -> hidden_dim -> d_model]`.
    pub fn new(
        d_model: usize,
        hidden_dim: usize,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if d_model == 0 || hidden_dim == 0 {
            return Err(OchreError::InvalidParameter(
                "SwiGLU d_model and hidden_dim must be > 0 ( >_< )".to_string(),
            ));
        }
        let w_gate = Linear::new(d_model, hidden_dim, use_bias, dtype, device.clone(), seed)?;
        let w_up = Linear::new(
            d_model,
            hidden_dim,
            use_bias,
            dtype,
            device.clone(),
            seed.wrapping_add(13),
        )?;
        let w_down = Linear::new(
            hidden_dim,
            d_model,
            use_bias,
            dtype,
            device,
            seed.wrapping_add(29),
        )?;
        Ok(Self {
            w_gate,
            w_up,
            w_down,
            d_model,
            hidden_dim,
        })
    }

    /// Autograd-tracked forward pass using N8 `SiLU` (`silu_f32` / `silu_bwd_f32`).
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let gate = self.w_gate.forward(x)?.silu()?;
        let up = self.w_up.forward(x)?;
        let fused = gate.mul(&up)?;
        self.w_down.forward(&fused)
    }

    /// Raw forward pass over `[rows, d_model]` without autograd tape recording.
    pub fn forward_raw(&self, x: &[f32], rows: usize, out: &mut [f32]) -> OchreResult<()> {
        if x.len() != rows * self.d_model || out.len() != rows * self.d_model {
            return Err(OchreError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: self.d_model,
                actual_rows: x.len(),
                actual_cols: out.len(),
            });
        }
        let hidden_len = rows * self.hidden_dim;
        SWIGLU_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            ensure_len(&mut scratch.gate, hidden_len);
            ensure_len(&mut scratch.up, hidden_len);
            let SwiGluScratch { gate, up } = &mut *scratch;
            let gate_slice = &mut gate[..hidden_len];
            let up_slice = &mut up[..hidden_len];

            self.w_gate.forward_raw(x, rows, gate_slice)?;
            self.w_up.forward_raw(x, rows, up_slice)?;
            if self.w_gate.device == DeviceType::Cpu {
                for (g, &u) in gate_slice.iter_mut().zip(up_slice.iter()) {
                    let gv = *g;
                    *g = (gv / (1.0 + (-gv).exp())) * u;
                }
            } else {
                let silu_gate = get_backend(&self.w_gate.device).silu_f32(gate_slice);
                for ((dst, &s), &u) in gate_slice
                    .iter_mut()
                    .zip(silu_gate.iter())
                    .zip(up_slice.iter())
                {
                    *dst = s * u;
                }
            }
            self.w_down.forward_raw(gate_slice, rows, out)
        })
    }

    /// Returns references to all parameters in `w_gate`, `w_up`, `w_down`.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::with_capacity(6);
        for lin in [&self.w_gate, &self.w_up, &self.w_down] {
            params.push(&lin.weight);
            if let Some(ref b) = lin.bias {
                params.push(b);
            }
        }
        params
    }

    /// Returns mutable references to all parameters in `w_gate`, `w_up`, `w_down`.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::with_capacity(6);
        for lin in [&mut self.w_gate, &mut self.w_up, &mut self.w_down] {
            params.push(&mut lin.weight);
            if let Some(ref mut b) = lin.bias {
                params.push(b);
            }
        }
        params
    }
}

/// Standard Transformer FeedForward network: `fc1 -> GELU -> fc2`.
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    /// Creates a `FeedForward` block `[d_model -> d_ff -> d_model]`.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if d_model == 0 || d_ff == 0 {
            return Err(OchreError::InvalidParameter(
                "FeedForward d_model and d_ff must be > 0 ( >_< )".to_string(),
            ));
        }
        let fc1 = Linear::new(d_model, d_ff, use_bias, dtype, device.clone(), seed)?;
        let fc2 = Linear::new(
            d_ff,
            d_model,
            use_bias,
            dtype,
            device,
            seed.wrapping_add(19),
        )?;
        Ok(Self {
            fc1,
            fc2,
            d_model,
            d_ff,
        })
    }

    /// Autograd-tracked forward pass using N8 `GELU` (`gelu_f32` / `gelu_bwd_f32`).
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let h = self.fc1.forward(x)?.gelu()?;
        self.fc2.forward(&h)
    }

    /// Raw forward pass over `[rows, d_model]` without autograd tape recording.
    pub fn forward_raw(&self, x: &[f32], rows: usize, out: &mut [f32]) -> OchreResult<()> {
        if x.len() != rows * self.d_model || out.len() != rows * self.d_model {
            return Err(OchreError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: self.d_model,
                actual_rows: x.len(),
                actual_cols: out.len(),
            });
        }
        let hidden_len = rows * self.d_ff;
        FFN_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            ensure_len(&mut scratch.hidden, hidden_len);
            let h = &mut scratch.hidden[..hidden_len];
            self.fc1.forward_raw(x, rows, h)?;
            let act = get_backend(&self.fc1.device).gelu_f32(h);
            self.fc2.forward_raw(&act, rows, out)
        })
    }

    /// Returns references to all parameters in `fc1` and `fc2`.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::with_capacity(4);
        for lin in [&self.fc1, &self.fc2] {
            params.push(&lin.weight);
            if let Some(ref b) = lin.bias {
                params.push(b);
            }
        }
        params
    }

    /// Returns mutable references to all parameters in `fc1` and `fc2`.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::with_capacity(4);
        for lin in [&mut self.fc1, &mut self.fc2] {
            params.push(&mut lin.weight);
            if let Some(ref mut b) = lin.bias {
                params.push(b);
            }
        }
        params
    }
}

/// Transformer Encoder Layer supporting pre-norm and post-norm configurations.
#[derive(Clone, Debug)]
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub pre_norm: bool,
}

impl TransformerEncoderLayer {
    /// Creates a `TransformerEncoderLayer`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        pre_norm: bool,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let self_attn = MultiHeadAttention::new(
            d_model,
            num_heads,
            num_heads,
            use_bias,
            false,
            None,
            dtype,
            device.clone(),
            seed,
        )?;
        let ffn = FeedForward::new(
            d_model,
            d_ff,
            use_bias,
            dtype,
            device.clone(),
            seed.wrapping_add(101),
        )?;
        let norm1 = LayerNorm::new(d_model, eps, true, dtype, device.clone())?;
        let norm2 = LayerNorm::new(d_model, eps, true, dtype, device)?;
        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
            pre_norm,
        })
    }

    /// Autograd-tracked encoder layer forward pass.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        if self.pre_norm {
            let n1 = self.norm1.forward(x)?;
            let attn = self.self_attn.forward(&n1, &n1, &n1, None)?;
            let h = x.add(&attn)?;
            let n2 = self.norm2.forward(&h)?;
            let ff = self.ffn.forward(&n2)?;
            h.add(&ff)
        } else {
            let attn = self.self_attn.forward(x, x, x, None)?;
            let h = self.norm1.forward(&x.add(&attn)?)?;
            let ff = self.ffn.forward(&h)?;
            self.norm2.forward(&h.add(&ff)?)
        }
    }

    /// Returns references to all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    /// Returns mutable references to all trainable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        params.extend(self.ffn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }
}

/// Stack of `TransformerEncoderLayer` blocks with optional final `LayerNorm`.
#[derive(Clone, Debug)]
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub final_norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    /// Creates a `TransformerEncoder` of `num_layers` encoder layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        pre_norm: bool,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if num_layers == 0 {
            return Err(OchreError::InvalidParameter(
                "TransformerEncoder num_layers must be > 0 ( >_< )".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                eps,
                pre_norm,
                use_bias,
                dtype,
                device.clone(),
                seed.wrapping_add((i as u64) * 997),
            )?);
        }
        let final_norm = if pre_norm {
            Some(LayerNorm::new(d_model, eps, true, dtype, device)?)
        } else {
            None
        };
        Ok(Self { layers, final_norm })
    }

    /// Autograd-tracked forward pass across all encoder layers.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let mut h = self.layers[0].forward(x)?;
        for layer in &self.layers[1..] {
            h = layer.forward(&h)?;
        }
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h)?;
        }
        Ok(h)
    }

    /// Returns references to all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params
    }

    /// Returns mutable references to all trainable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(ref mut norm) = self.final_norm {
            params.extend(norm.parameters_mut());
        }
        params
    }
}

/// FFN variant inside `TransformerDecoderLayer`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum DecoderFfn {
    FeedForward(FeedForward),
    SwiGLU(SwiGLU),
}

impl DecoderFfn {
    fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        match self {
            DecoderFfn::FeedForward(ff) => ff.forward(x),
            DecoderFfn::SwiGLU(sg) => sg.forward(x),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match self {
            DecoderFfn::FeedForward(ff) => ff.parameters(),
            DecoderFfn::SwiGLU(sg) => sg.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        match self {
            DecoderFfn::FeedForward(ff) => ff.parameters_mut(),
            DecoderFfn::SwiGLU(sg) => sg.parameters_mut(),
        }
    }
}

/// Transformer Decoder Layer with causal self-attention, optional cross-attention, and FFN/SwiGLU.
#[derive(Clone, Debug)]
pub struct TransformerDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: Option<MultiHeadAttention>,
    pub ffn: DecoderFfn,
    pub norm1: LayerNorm,
    pub norm_cross: Option<LayerNorm>,
    pub norm2: LayerNorm,
    pub pre_norm: bool,
}

impl TransformerDecoderLayer {
    /// Creates a `TransformerDecoderLayer`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        with_cross_attn: bool,
        use_swiglu: bool,
        pre_norm: bool,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let self_attn = MultiHeadAttention::new(
            d_model,
            num_heads,
            num_heads,
            use_bias,
            true,
            None,
            dtype,
            device.clone(),
            seed,
        )?;
        let (cross_attn, norm_cross) = if with_cross_attn {
            (
                Some(MultiHeadAttention::new(
                    d_model,
                    num_heads,
                    num_heads,
                    use_bias,
                    false,
                    None,
                    dtype,
                    device.clone(),
                    seed.wrapping_add(53),
                )?),
                Some(LayerNorm::new(d_model, eps, true, dtype, device.clone())?),
            )
        } else {
            (None, None)
        };
        let ffn = if use_swiglu {
            DecoderFfn::SwiGLU(SwiGLU::new(
                d_model,
                d_ff,
                use_bias,
                dtype,
                device.clone(),
                seed.wrapping_add(107),
            )?)
        } else {
            DecoderFfn::FeedForward(FeedForward::new(
                d_model,
                d_ff,
                use_bias,
                dtype,
                device.clone(),
                seed.wrapping_add(107),
            )?)
        };
        let norm1 = LayerNorm::new(d_model, eps, true, dtype, device.clone())?;
        let norm2 = LayerNorm::new(d_model, eps, true, dtype, device)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ffn,
            norm1,
            norm_cross,
            norm2,
            pre_norm,
        })
    }

    /// Autograd-tracked decoder layer forward pass with optional `memory` and `kv_cache`.
    pub fn forward(
        &self,
        x: &Tensor,
        memory: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
    ) -> OchreResult<Tensor> {
        if self.cross_attn.is_some() && memory.is_none() {
            return Err(OchreError::InvalidParameter(
                "TransformerDecoderLayer has cross_attn enabled but memory is None ( >_< )"
                    .to_string(),
            ));
        }
        if self.pre_norm {
            let n1 = self.norm1.forward(x)?;
            let sa = self.self_attn.forward(&n1, &n1, &n1, kv_cache)?;
            let mut h = x.add(&sa)?;
            if let (Some(ca), Some(nc), Some(mem)) = (&self.cross_attn, &self.norm_cross, memory) {
                let n_ca = nc.forward(&h)?;
                let ca_out = ca.forward(&n_ca, mem, mem, None)?;
                h = h.add(&ca_out)?;
            }
            let n2 = self.norm2.forward(&h)?;
            let ff = self.ffn.forward(&n2)?;
            h.add(&ff)
        } else {
            let sa = self.self_attn.forward(x, x, x, kv_cache)?;
            let mut h = self.norm1.forward(&x.add(&sa)?)?;
            if let (Some(ca), Some(nc), Some(mem)) = (&self.cross_attn, &self.norm_cross, memory) {
                let ca_out = ca.forward(&h, mem, mem, None)?;
                h = nc.forward(&h.add(&ca_out)?)?;
            }
            let ff = self.ffn.forward(&h)?;
            self.norm2.forward(&h.add(&ff)?)
        }
    }

    /// Returns references to all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        if let Some(ref ca) = self.cross_attn {
            params.extend(ca.parameters());
        }
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        if let Some(ref nc) = self.norm_cross {
            params.extend(nc.parameters());
        }
        params.extend(self.norm2.parameters());
        params
    }

    /// Returns mutable references to all trainable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        if let Some(ref mut ca) = self.cross_attn {
            params.extend(ca.parameters_mut());
        }
        params.extend(self.ffn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        if let Some(ref mut nc) = self.norm_cross {
            params.extend(nc.parameters_mut());
        }
        params.extend(self.norm2.parameters_mut());
        params
    }
}

/// Stack of `TransformerDecoderLayer` blocks with optional final `LayerNorm`.
#[derive(Clone, Debug)]
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
    pub final_norm: Option<LayerNorm>,
}

impl TransformerDecoder {
    /// Creates a `TransformerDecoder` of `num_layers` decoder layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        with_cross_attn: bool,
        use_swiglu: bool,
        pre_norm: bool,
        use_bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if num_layers == 0 {
            return Err(OchreError::InvalidParameter(
                "TransformerDecoder num_layers must be > 0 ( >_< )".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                eps,
                with_cross_attn,
                use_swiglu,
                pre_norm,
                use_bias,
                dtype,
                device.clone(),
                seed.wrapping_add((i as u64) * 997),
            )?);
        }
        let final_norm = if pre_norm {
            Some(LayerNorm::new(d_model, eps, true, dtype, device)?)
        } else {
            None
        };
        Ok(Self { layers, final_norm })
    }

    /// Autograd-tracked forward pass across all decoder layers.
    pub fn forward(
        &self,
        x: &Tensor,
        memory: Option<&Tensor>,
        mut kv_caches: Option<&mut [KVCache]>,
    ) -> OchreResult<Tensor> {
        if let Some(ref caches) = kv_caches
            && caches.len() != self.layers.len()
        {
            return Err(OchreError::InvalidParameter(format!(
                "TransformerDecoder kv_caches len {} != num_layers {} ( >_< )",
                caches.len(),
                self.layers.len()
            )));
        }
        let first_cache = kv_caches.as_deref_mut().map(|c| &mut c[0]);
        let mut h = self.layers[0].forward(x, memory, first_cache)?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            let cache_i = kv_caches.as_deref_mut().map(|c| &mut c[i]);
            h = layer.forward(&h, memory, cache_i)?;
        }
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h)?;
        }
        Ok(h)
    }

    /// Returns references to all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params
    }

    /// Returns mutable references to all trainable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(ref mut norm) = self.final_norm {
            params.extend(norm.parameters_mut());
        }
        params
    }
}

/// Configuration for LLaMA-style `CausalLM`.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalLMConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub ffn_hidden_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub dtype: DType,
}

impl CausalLMConfig {
    /// Validates architectural invariants for `CausalLMConfig`.
    pub fn validate(&self) -> OchreResult<()> {
        if self.vocab_size == 0
            || self.d_model == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.ffn_hidden_dim == 0
            || self.max_seq_len == 0
        {
            return Err(OchreError::InvalidParameter(
                "CausalLMConfig dimensions must all be > 0 ( >_< )".to_string(),
            ));
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "d_model ({}) must be divisible by num_heads ({}) ( >_< )",
                self.d_model, self.num_heads
            )));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({}) ( >_< )",
                self.num_heads, self.num_kv_heads
            )));
        }
        let head_dim = self.d_model / self.num_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM head_dim ({head_dim}) must be even for RoPE ( >_< )"
            )));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLMConfig rope_theta={} must be finite and > 0 ( >_< )",
                self.rope_theta
            )));
        }
        if !self.norm_eps.is_finite() || self.norm_eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLMConfig norm_eps={} must be finite and >= 0 ( >_< )",
                self.norm_eps
            )));
        }
        Ok(())
    }
}

/// Single decoder block inside `CausalLM`: `RMSNorm -> GQA+RoPE -> RMSNorm -> SwiGLU`.
#[derive(Clone, Debug)]
pub struct CausalLMBlock {
    pub attn_norm: RMSNorm,
    pub attn: MultiHeadAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: SwiGLU,
}

impl CausalLMBlock {
    /// Creates a `CausalLMBlock` with shared `RoPE` tables.
    pub fn new(
        config: &CausalLMConfig,
        rope: RoPE,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let attn_norm = RMSNorm::new(
            config.d_model,
            config.norm_eps,
            config.dtype,
            device.clone(),
        )?;
        let attn = MultiHeadAttention::new(
            config.d_model,
            config.num_heads,
            config.num_kv_heads,
            false,
            true,
            Some(rope),
            config.dtype,
            device.clone(),
            seed,
        )?;
        let ffn_norm = RMSNorm::new(
            config.d_model,
            config.norm_eps,
            config.dtype,
            device.clone(),
        )?;
        let ffn = SwiGLU::new(
            config.d_model,
            config.ffn_hidden_dim,
            false,
            config.dtype,
            device,
            seed.wrapping_add(211),
        )?;
        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        })
    }

    /// Autograd-tracked block forward pass.
    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut KVCache>) -> OchreResult<Tensor> {
        let xn = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&xn, &xn, &xn, kv_cache)?;
        let h = x.add(&attn_out)?;
        let fn_in = self.ffn_norm.forward(&h)?;
        let ff_out = self.ffn.forward(&fn_in)?;
        h.add(&ff_out)
    }

    /// Fast raw forward pass over `[batch, seq_len, d_model]` without autograd tape recording.
    pub fn forward_raw(
        &self,
        h: &mut [f32],
        norm_buf: &mut [f32],
        sub_buf: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_cache: Option<&mut KVCache>,
    ) -> OchreResult<()> {
        let rows = batch * seq_len;
        let d_model = self.attn.d_model;
        if self.attn_norm.device == DeviceType::Cpu {
            rmsnorm_into_cpu(
                h,
                &self.attn_norm.gamma.data,
                rows,
                d_model,
                self.attn_norm.eps,
                norm_buf,
            );
            self.attn.forward_raw(
                norm_buf, norm_buf, norm_buf, batch, seq_len, seq_len, kv_cache, sub_buf,
            )?;
        } else {
            let (norm1, _) = self.attn_norm.forward_raw(h, rows)?;
            self.attn.forward_raw(
                &norm1, &norm1, &norm1, batch, seq_len, seq_len, kv_cache, sub_buf,
            )?;
        }
        for (hv, &av) in h.iter_mut().zip(sub_buf.iter()) {
            *hv += av;
        }
        if self.ffn_norm.device == DeviceType::Cpu {
            rmsnorm_into_cpu(
                h,
                &self.ffn_norm.gamma.data,
                rows,
                d_model,
                self.ffn_norm.eps,
                norm_buf,
            );
            self.ffn.forward_raw(norm_buf, rows, sub_buf)?;
        } else {
            let (norm2, _) = self.ffn_norm.forward_raw(h, rows)?;
            self.ffn.forward_raw(&norm2, rows, sub_buf)?;
        }
        for (hv, &fv) in h.iter_mut().zip(sub_buf.iter()) {
            *hv += fv;
        }
        Ok(())
    }

    /// Returns references to all trainable parameters in this block.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.attn_norm.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ffn_norm.parameters());
        params.extend(self.ffn.parameters());
        params
    }

    /// Returns mutable references to all trainable parameters in this block.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.attn_norm.parameters_mut();
        params.extend(self.attn.parameters_mut());
        params.extend(self.ffn_norm.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params
    }
}

/// LLaMA-style Causal Language Model with RMSNorm, GQA+RoPE Attention, SwiGLU, and `.dit` persistence.
#[derive(Clone, Debug)]
pub struct CausalLM {
    pub config: CausalLMConfig,
    pub tok_embeddings: Embedding,
    pub layers: Vec<CausalLMBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
    pub device: DeviceType,
}

impl CausalLM {
    /// Builds a `CausalLM` from `config` on `device`.
    pub fn new(config: CausalLMConfig, device: DeviceType, seed: u64) -> OchreResult<Self> {
        config.validate()?;
        let head_dim = config.d_model / config.num_heads;
        let rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta)?;
        let tok_embeddings = Embedding::new(
            config.vocab_size,
            config.d_model,
            config.dtype,
            device.clone(),
            seed,
        )?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(CausalLMBlock::new(
                &config,
                rope.clone(),
                device.clone(),
                seed.wrapping_add(((i + 1) as u64) * 1009),
            )?);
        }
        let final_norm = RMSNorm::new(
            config.d_model,
            config.norm_eps,
            config.dtype,
            device.clone(),
        )?;
        let lm_head = Linear::new(
            config.d_model,
            config.vocab_size,
            false,
            config.dtype,
            device.clone(),
            seed.wrapping_add(999_983),
        )?;
        Ok(Self {
            config,
            tok_embeddings,
            layers,
            final_norm,
            lm_head,
            device,
        })
    }

    /// Pre-allocates one `KVCache` per transformer layer for `batch` sequences.
    pub fn new_kv_caches(&self, batch: usize) -> OchreResult<Vec<KVCache>> {
        let head_dim = self.config.d_model / self.config.num_heads;
        let mut caches = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            caches.push(KVCache::new(
                batch,
                self.config.num_kv_heads,
                self.config.max_seq_len,
                head_dim,
            )?);
        }
        Ok(caches)
    }

    /// Autograd-compatible forward pass on token IDs `[batch, seq_len]`, returning logits `[batch, seq_len, vocab_size]`.
    pub fn forward_ids(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        mut kv_cache: Option<&mut [KVCache]>,
    ) -> OchreResult<Tensor> {
        if batch == 0 || seq_len == 0 {
            return Err(OchreError::EmptyData);
        }
        if input_ids.len() != batch * seq_len {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::forward_ids expected {} tokens ({batch}x{seq_len}), got {} ( >_< )",
                batch * seq_len,
                input_ids.len()
            )));
        }
        if let Some(ref caches) = kv_cache
            && caches.len() != self.layers.len()
        {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM kv_cache len {} != num_layers {} ( >_< )",
                caches.len(),
                self.layers.len()
            )));
        }

        let mut h = self.tok_embeddings.forward_2d(input_ids, batch, seq_len)?;
        for (i, block) in self.layers.iter().enumerate() {
            let cache_i = kv_cache.as_deref_mut().map(|c| &mut c[i]);
            h = block.forward(&h, cache_i)?;
        }
        h = self.final_norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Autoregressive token generation using pre-allocated `KVCache` buffers.
    /// Returns `prompt_ids` followed by `max_new_tokens` newly generated token IDs.
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> OchreResult<Vec<usize>> {
        if prompt_ids.is_empty() {
            return Err(OchreError::EmptyData);
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "generate temperature={temperature} must be finite and >= 0 ( >_< )"
            )));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(OchreError::InvalidParameter(format!(
                "generate top_p={top_p} must be in (0.0, 1.0] ( >_< )"
            )));
        }
        let total_len = prompt_ids.len().saturating_add(max_new_tokens);
        if total_len > self.config.max_seq_len {
            return Err(OchreError::InvalidParameter(format!(
                "generate total length {total_len} exceeds max_seq_len {} ( >_< )",
                self.config.max_seq_len
            )));
        }

        let mut output = Vec::with_capacity(total_len);
        output.extend_from_slice(prompt_ids);
        if max_new_tokens == 0 {
            return Ok(output);
        }

        let d_model = self.config.d_model;
        let vocab_size = self.config.vocab_size;
        let mut caches = self.new_kv_caches(1)?;

        // Pre-allocate buffers once for prefill and all O(1)-alloc decode steps.
        let p_len = prompt_ids.len();
        let max_buf = p_len.max(1);
        let mut h = vec![0.0f32; max_buf * d_model];
        let mut norm_buf = vec![0.0f32; max_buf * d_model];
        let mut sub_buf = vec![0.0f32; max_buf * d_model];
        let mut logits = vec![0.0f32; vocab_size];
        let mut sample_buf: Vec<(usize, f32)> = if temperature > 1e-6 {
            Vec::with_capacity(vocab_size)
        } else {
            Vec::new()
        };

        // Prefill prompt
        self.tok_embeddings
            .forward_raw(prompt_ids, &mut h[..p_len * d_model])?;
        for (i, block) in self.layers.iter().enumerate() {
            block.forward_raw(
                &mut h[..p_len * d_model],
                &mut norm_buf[..p_len * d_model],
                &mut sub_buf[..p_len * d_model],
                1,
                p_len,
                Some(&mut caches[i]),
            )?;
        }
        // Only normalize and project the last prompt token to logits.
        let last_h = &h[(p_len - 1) * d_model..p_len * d_model];
        if self.final_norm.device == DeviceType::Cpu {
            rmsnorm_into_cpu(
                last_h,
                &self.final_norm.gamma.data,
                1,
                d_model,
                self.final_norm.eps,
                &mut norm_buf[..d_model],
            );
            self.lm_head
                .forward_raw(&norm_buf[..d_model], 1, &mut logits)?;
        } else {
            let (final_h, _) = self.final_norm.forward_raw(last_h, 1)?;
            self.lm_head.forward_raw(&final_h, 1, &mut logits)?;
        }

        let mut rng = FastRng::new(seed);
        let mut next_tok = sample_next_token(
            &logits,
            temperature,
            top_k,
            top_p,
            &mut rng,
            &mut sample_buf,
        );
        output.push(next_tok);

        // Single-token decode steps reusing pre-allocated buffers
        let mut tok_slice = [0usize; 1];
        for _ in 1..max_new_tokens {
            tok_slice[0] = next_tok;
            self.tok_embeddings
                .forward_raw(&tok_slice, &mut h[..d_model])?;
            for (i, block) in self.layers.iter().enumerate() {
                block.forward_raw(
                    &mut h[..d_model],
                    &mut norm_buf[..d_model],
                    &mut sub_buf[..d_model],
                    1,
                    1,
                    Some(&mut caches[i]),
                )?;
            }
            if self.final_norm.device == DeviceType::Cpu {
                rmsnorm_into_cpu(
                    &h[..d_model],
                    &self.final_norm.gamma.data,
                    1,
                    d_model,
                    self.final_norm.eps,
                    &mut norm_buf[..d_model],
                );
                self.lm_head
                    .forward_raw(&norm_buf[..d_model], 1, &mut logits)?;
            } else {
                let (final_step, _) = self.final_norm.forward_raw(&h[..d_model], 1)?;
                self.lm_head.forward_raw(&final_step, 1, &mut logits)?;
            }
            next_tok = sample_next_token(
                &logits,
                temperature,
                top_k,
                top_p,
                &mut rng,
                &mut sample_buf,
            );
            output.push(next_tok);
        }

        Ok(output)
    }

    /// Zero-GIL, pre-allocated multi-step training loop for `CausalLM` using AdamW + CrossEntropy.
    ///
    /// On `DeviceType::Gpu(_)` / `MultiGpu(_)` (with `cuda-dynamic`), runs 100% resident in VRAM
    /// via `CudaVramSession` (uploading weights/tokens once before the loop, downloading only 1
    /// scalar loss per step, and downloading weights once at the end), falling back cleanly to
    /// the zero-allocation CPU path if CUDA is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn train_steps(
        &mut self,
        input_ids: &[usize],
        targets: &[usize],
        batch: usize,
        seq_len: usize,
        steps: usize,
        lr: f32,
        weight_decay: f32,
        grad_clip: Option<f32>,
    ) -> OchreResult<Vec<f32>> {
        if batch == 0 || seq_len == 0 {
            return Err(OchreError::EmptyData);
        }
        if steps == 0 {
            return Err(OchreError::InvalidParameter(
                "CausalLM::train_steps steps must be > 0 ( >_< )".to_string(),
            ));
        }
        if !lr.is_finite() || lr < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::train_steps lr={lr} must be finite and >= 0 ( >_< )"
            )));
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::train_steps weight_decay={weight_decay} must be finite and >= 0 ( >_< )"
            )));
        }
        if let Some(gc) = grad_clip
            && (!gc.is_finite() || gc < 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::train_steps grad_clip={gc} must be finite and >= 0 ( >_< )"
            )));
        }
        if seq_len > self.config.max_seq_len {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::train_steps seq_len={seq_len} exceeds max_seq_len={} ( >_< )",
                self.config.max_seq_len
            )));
        }
        let n_tokens = batch.checked_mul(seq_len).ok_or_else(|| {
            OchreError::InvalidParameter(
                "CausalLM::train_steps token count overflow ( >_< )".to_string(),
            )
        })?;
        if input_ids.len() != n_tokens || targets.len() != n_tokens {
            return Err(OchreError::InvalidParameter(format!(
                "CausalLM::train_steps expected {n_tokens} tokens ({batch}x{seq_len}), got inputs={} targets={} ( >_< )",
                input_ids.len(),
                targets.len()
            )));
        }
        for &id in input_ids {
            if id >= self.config.vocab_size {
                return Err(OchreError::InvalidParameter(format!(
                    "CausalLM::train_steps input token ID {id} >= vocab_size {} ( >_< )",
                    self.config.vocab_size
                )));
            }
        }
        for &id in targets {
            if id >= self.config.vocab_size {
                return Err(OchreError::InvalidParameter(format!(
                    "CausalLM::train_steps target token ID {id} >= vocab_size {} ( >_< )",
                    self.config.vocab_size
                )));
            }
        }
        let has_unexpected_bias = self.lm_head.bias.is_some()
            || self.layers.iter().any(|b| {
                b.attn.q_proj.bias.is_some()
                    || b.attn.k_proj.bias.is_some()
                    || b.attn.v_proj.bias.is_some()
                    || b.attn.o_proj.bias.is_some()
                    || b.ffn.w_gate.bias.is_some()
                    || b.ffn.w_up.bias.is_some()
                    || b.ffn.w_down.bias.is_some()
            });
        if has_unexpected_bias {
            return Err(OchreError::InvalidParameter(
                "CausalLM::train_steps does not support biased Linear layers ( >_< )".to_string(),
            ));
        }

        #[cfg(feature = "cuda-dynamic")]
        if matches!(self.device, DeviceType::Gpu(_) | DeviceType::MultiGpu(_)) {
            let gpu_id = match &self.device {
                DeviceType::Gpu(id) => *id,
                DeviceType::MultiGpu(ids) => ids.first().copied().unwrap_or(0),
                _ => 0,
            };
            if let Some(res) = self.train_steps_cuda_vram(
                gpu_id,
                input_ids,
                targets,
                batch,
                seq_len,
                steps,
                lr,
                weight_decay,
                grad_clip,
            ) {
                return res;
            }
        }

        self.train_steps_cpu(
            input_ids,
            targets,
            batch,
            seq_len,
            steps,
            lr,
            weight_decay,
            grad_clip,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn train_steps_cpu(
        &mut self,
        input_ids: &[usize],
        targets: &[usize],
        batch: usize,
        seq_len: usize,
        steps: usize,
        lr: f32,
        weight_decay: f32,
        grad_clip: Option<f32>,
    ) -> OchreResult<Vec<f32>> {
        let rows = batch * seq_len;
        let d_model = self.config.d_model;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let hidden_dim = self.config.ffn_hidden_dim;
        let vocab_size = self.config.vocab_size;
        let num_layers = self.layers.len();
        let dtype = self.config.dtype;

        let n_params = 3 + 9 * num_layers;
        let mut grads: Vec<Vec<f32>> = Vec::with_capacity(n_params);
        let mut opt_m: Vec<Vec<f32>> = Vec::with_capacity(n_params);
        let mut opt_v: Vec<Vec<f32>> = Vec::with_capacity(n_params);
        let mut master_weights: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_params);
        {
            let params = self.parameters();
            for p in params {
                let len = p.data.len();
                grads.push(vec![0.0f32; len]);
                opt_m.push(vec![0.0f32; len]);
                opt_v.push(vec![0.0f32; len]);
                master_weights.push(if dtype != DType::F32 {
                    Some(p.data.clone())
                } else {
                    None
                });
            }
        }

        // Pre-allocate all per-layer activation & scratch buffers ONCE outside the step loop.
        let mut layer_x_in: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_xn1: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_rstd1: Vec<Vec<f32>> = (0..num_layers).map(|_| vec![0.0f32; rows]).collect();
        let mut layer_q_bhsd: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_k_bhsd: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * kv_dim])
            .collect();
        let mut layer_v_bhsd: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * kv_dim])
            .collect();
        let mut layer_attn_bhsd: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_lse: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; batch * num_heads * seq_len])
            .collect();
        let mut layer_attn_merged: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_h_mid: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_xn2: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * d_model])
            .collect();
        let mut layer_rstd2: Vec<Vec<f32>> = (0..num_layers).map(|_| vec![0.0f32; rows]).collect();
        let mut layer_gate: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * hidden_dim])
            .collect();
        let mut layer_up: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * hidden_dim])
            .collect();
        let mut layer_swiglu_out: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; rows * hidden_dim])
            .collect();

        let mut x_after_layers = vec![0.0f32; rows * d_model];
        let mut final_norm_out = vec![0.0f32; rows * d_model];
        let mut final_rstd = vec![0.0f32; rows];
        let mut logits = vec![0.0f32; rows * vocab_size];
        let mut d_logits = vec![0.0f32; rows * vocab_size];

        let mut q_bshd_tmp = vec![0.0f32; rows * d_model];
        let mut k_bshd_tmp = vec![0.0f32; rows * kv_dim];
        let mut v_bshd_tmp = vec![0.0f32; rows * kv_dim];

        let mut d_h = vec![0.0f32; rows * d_model];
        let mut d_norm_tmp = vec![0.0f32; rows * d_model];
        let mut d_norm_dx = vec![0.0f32; rows * d_model];
        let mut d_attn_merged = vec![0.0f32; rows * d_model];
        let mut d_attn_bhsd = vec![0.0f32; rows * d_model];
        let mut d_q_bhsd = vec![0.0f32; rows * d_model];
        let mut d_k_bhsd = vec![0.0f32; rows * kv_dim];
        let mut d_v_bhsd = vec![0.0f32; rows * kv_dim];
        let mut d_swiglu_out = vec![0.0f32; rows * hidden_dim];
        let mut d_gate = vec![0.0f32; rows * hidden_dim];
        let mut d_up = vec![0.0f32; rows * hidden_dim];
        let mut ffn_tmp = vec![0.0f32; rows * hidden_dim];
        let mut dgamma_accum = vec![0.0f64; d_model];

        let final_norm_g_idx = 1 + 9 * num_layers;
        let lm_head_w_idx = final_norm_g_idx + 1;
        let mut loss_history = Vec::with_capacity(steps);

        for step in 0..steps {
            // 1. Token embedding lookup
            self.tok_embeddings
                .forward_raw(input_ids, &mut layer_x_in[0])?;

            // 2. Transformer blocks forward
            for l in 0..num_layers {
                let block = &self.layers[l];
                rmsnorm_fwd_with_rstd_into_cpu(
                    &layer_x_in[l],
                    &block.attn_norm.gamma.data,
                    rows,
                    d_model,
                    block.attn_norm.eps,
                    &mut layer_xn1[l],
                    &mut layer_rstd1[l],
                );

                sgemm(
                    rows,
                    d_model,
                    d_model,
                    &layer_xn1[l],
                    &block.attn.q_proj.weight.data,
                    &mut q_bshd_tmp,
                    false,
                    true,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    kv_dim,
                    d_model,
                    &layer_xn1[l],
                    &block.attn.k_proj.weight.data,
                    &mut k_bshd_tmp,
                    false,
                    true,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    kv_dim,
                    d_model,
                    &layer_xn1[l],
                    &block.attn.v_proj.weight.data,
                    &mut v_bshd_tmp,
                    false,
                    true,
                    1.0,
                    0.0,
                )?;

                transpose_bshd_to_bhsd_into(
                    &q_bshd_tmp,
                    &mut layer_q_bhsd[l],
                    batch,
                    seq_len,
                    num_heads,
                    head_dim,
                );
                transpose_bshd_to_bhsd_into(
                    &k_bshd_tmp,
                    &mut layer_k_bhsd[l],
                    batch,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                );
                transpose_bshd_to_bhsd_into(
                    &v_bshd_tmp,
                    &mut layer_v_bhsd[l],
                    batch,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                );

                if let Some(ref rope) = block.attn.rope {
                    rope.apply_inplace(&mut layer_q_bhsd[l], batch, num_heads, seq_len, 0)?;
                    rope.apply_inplace(&mut layer_k_bhsd[l], batch, num_kv_heads, seq_len, 0)?;
                }

                flash_attention_forward_strided_into(
                    &layer_q_bhsd[l],
                    &layer_k_bhsd[l],
                    &layer_v_bhsd[l],
                    batch,
                    num_heads,
                    num_kv_heads,
                    seq_len,
                    seq_len,
                    seq_len,
                    head_dim,
                    true,
                    0,
                    &mut layer_attn_bhsd[l],
                    Some(&mut layer_lse[l]),
                )?;

                transpose_bhsd_to_bshd_into(
                    &layer_attn_bhsd[l],
                    &mut layer_attn_merged[l],
                    batch,
                    num_heads,
                    seq_len,
                    head_dim,
                );

                sgemm(
                    rows,
                    d_model,
                    d_model,
                    &layer_attn_merged[l],
                    &block.attn.o_proj.weight.data,
                    &mut layer_h_mid[l],
                    false,
                    true,
                    1.0,
                    0.0,
                )?;
                for (hm, &xi) in layer_h_mid[l].iter_mut().zip(layer_x_in[l].iter()) {
                    *hm += xi;
                }

                rmsnorm_fwd_with_rstd_into_cpu(
                    &layer_h_mid[l],
                    &block.ffn_norm.gamma.data,
                    rows,
                    d_model,
                    block.ffn_norm.eps,
                    &mut layer_xn2[l],
                    &mut layer_rstd2[l],
                );

                sgemm(
                    rows,
                    hidden_dim,
                    d_model,
                    &layer_xn2[l],
                    &block.ffn.w_gate.weight.data,
                    &mut layer_gate[l],
                    false,
                    true,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    hidden_dim,
                    d_model,
                    &layer_xn2[l],
                    &block.ffn.w_up.weight.data,
                    &mut layer_up[l],
                    false,
                    true,
                    1.0,
                    0.0,
                )?;

                simd_silu_fwd_into(&layer_gate[l], &mut layer_swiglu_out[l]);
                for (sg, &uv) in layer_swiglu_out[l].iter_mut().zip(layer_up[l].iter()) {
                    *sg *= uv;
                }

                let next_x: &mut [f32] = if l + 1 < num_layers {
                    &mut layer_x_in[l + 1]
                } else {
                    &mut x_after_layers
                };
                sgemm(
                    rows,
                    d_model,
                    hidden_dim,
                    &layer_swiglu_out[l],
                    &block.ffn.w_down.weight.data,
                    next_x,
                    false,
                    true,
                    1.0,
                    0.0,
                )?;
                for (nx, &hm) in next_x.iter_mut().zip(layer_h_mid[l].iter()) {
                    *nx += hm;
                }
            }

            // 3. Final norm + LM head + CrossEntropy
            rmsnorm_fwd_with_rstd_into_cpu(
                &x_after_layers,
                &self.final_norm.gamma.data,
                rows,
                d_model,
                self.final_norm.eps,
                &mut final_norm_out,
                &mut final_rstd,
            );
            sgemm(
                rows,
                vocab_size,
                d_model,
                &final_norm_out,
                &self.lm_head.weight.data,
                &mut logits,
                false,
                true,
                1.0,
                0.0,
            )?;

            let loss =
                cross_entropy_grad_into_cpu(&logits, targets, rows, vocab_size, &mut d_logits);
            loss_history.push(loss);

            // 4. Backward pass
            sgemm(
                vocab_size,
                d_model,
                rows,
                &d_logits,
                &final_norm_out,
                &mut grads[lm_head_w_idx],
                true,
                false,
                1.0,
                0.0,
            )?;
            sgemm(
                rows,
                d_model,
                vocab_size,
                &d_logits,
                &self.lm_head.weight.data,
                &mut d_norm_tmp,
                false,
                false,
                1.0,
                0.0,
            )?;
            rmsnorm_bwd_into_cpu(
                &d_norm_tmp,
                &x_after_layers,
                &self.final_norm.gamma.data,
                &final_rstd,
                rows,
                d_model,
                &mut d_h,
                &mut grads[final_norm_g_idx],
                &mut dgamma_accum,
            );

            for l in (0..num_layers).rev() {
                let block = &self.layers[l];
                let base = 1 + 9 * l;
                let an_g_idx = base;
                let wq_idx = base + 1;
                let wk_idx = base + 2;
                let wv_idx = base + 3;
                let wo_idx = base + 4;
                let fn_g_idx = base + 5;
                let wg_idx = base + 6;
                let wu_idx = base + 7;
                let wd_idx = base + 8;

                // w_down backward
                sgemm(
                    d_model,
                    hidden_dim,
                    rows,
                    &d_h,
                    &layer_swiglu_out[l],
                    &mut grads[wd_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    hidden_dim,
                    d_model,
                    &d_h,
                    &block.ffn.w_down.weight.data,
                    &mut d_swiglu_out,
                    false,
                    false,
                    1.0,
                    0.0,
                )?;

                // SwiGLU backward
                simd_silu_fwd_into(&layer_gate[l], &mut d_up);
                for (du, &dy) in d_up.iter_mut().zip(d_swiglu_out.iter()) {
                    *du *= dy;
                }
                for (tmp, (&dy, &uv)) in ffn_tmp
                    .iter_mut()
                    .zip(d_swiglu_out.iter().zip(layer_up[l].iter()))
                {
                    *tmp = dy * uv;
                }
                simd_silu_bwd_into(&ffn_tmp, &layer_gate[l], &mut d_gate);

                // w_gate + w_up backward
                sgemm(
                    hidden_dim,
                    d_model,
                    rows,
                    &d_gate,
                    &layer_xn2[l],
                    &mut grads[wg_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    hidden_dim,
                    d_model,
                    rows,
                    &d_up,
                    &layer_xn2[l],
                    &mut grads[wu_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    d_model,
                    hidden_dim,
                    &d_gate,
                    &block.ffn.w_gate.weight.data,
                    &mut d_norm_tmp,
                    false,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    d_model,
                    hidden_dim,
                    &d_up,
                    &block.ffn.w_up.weight.data,
                    &mut d_norm_tmp,
                    false,
                    false,
                    1.0,
                    1.0,
                )?;

                // ffn_norm backward + residual add into d_h
                rmsnorm_bwd_into_cpu(
                    &d_norm_tmp,
                    &layer_h_mid[l],
                    &block.ffn_norm.gamma.data,
                    &layer_rstd2[l],
                    rows,
                    d_model,
                    &mut d_norm_dx,
                    &mut grads[fn_g_idx],
                    &mut dgamma_accum,
                );
                for (dhv, &dxv) in d_h.iter_mut().zip(d_norm_dx.iter()) {
                    *dhv += dxv;
                }

                // o_proj backward
                sgemm(
                    d_model,
                    d_model,
                    rows,
                    &d_h,
                    &layer_attn_merged[l],
                    &mut grads[wo_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    d_model,
                    d_model,
                    &d_h,
                    &block.attn.o_proj.weight.data,
                    &mut d_attn_merged,
                    false,
                    false,
                    1.0,
                    0.0,
                )?;

                transpose_bshd_to_bhsd_into(
                    &d_attn_merged,
                    &mut d_attn_bhsd,
                    batch,
                    seq_len,
                    num_heads,
                    head_dim,
                );
                flash_attention_backward_into(
                    &d_attn_bhsd,
                    &layer_q_bhsd[l],
                    &layer_k_bhsd[l],
                    &layer_v_bhsd[l],
                    &layer_attn_bhsd[l],
                    &layer_lse[l],
                    batch,
                    num_heads,
                    num_kv_heads,
                    seq_len,
                    seq_len,
                    head_dim,
                    true,
                    0,
                    &mut d_q_bhsd,
                    &mut d_k_bhsd,
                    &mut d_v_bhsd,
                )?;

                if let Some(ref rope) = block.attn.rope {
                    rope.apply_backward_inplace(&mut d_q_bhsd, batch, num_heads, seq_len, 0)?;
                    rope.apply_backward_inplace(&mut d_k_bhsd, batch, num_kv_heads, seq_len, 0)?;
                }

                transpose_bhsd_to_bshd_into(
                    &d_q_bhsd,
                    &mut q_bshd_tmp,
                    batch,
                    num_heads,
                    seq_len,
                    head_dim,
                );
                transpose_bhsd_to_bshd_into(
                    &d_k_bhsd,
                    &mut k_bshd_tmp,
                    batch,
                    num_kv_heads,
                    seq_len,
                    head_dim,
                );
                transpose_bhsd_to_bshd_into(
                    &d_v_bhsd,
                    &mut v_bshd_tmp,
                    batch,
                    num_kv_heads,
                    seq_len,
                    head_dim,
                );

                // Q, K, V projection backward
                sgemm(
                    d_model,
                    d_model,
                    rows,
                    &q_bshd_tmp,
                    &layer_xn1[l],
                    &mut grads[wq_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    kv_dim,
                    d_model,
                    rows,
                    &k_bshd_tmp,
                    &layer_xn1[l],
                    &mut grads[wk_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    kv_dim,
                    d_model,
                    rows,
                    &v_bshd_tmp,
                    &layer_xn1[l],
                    &mut grads[wv_idx],
                    true,
                    false,
                    1.0,
                    0.0,
                )?;

                sgemm(
                    rows,
                    d_model,
                    d_model,
                    &q_bshd_tmp,
                    &block.attn.q_proj.weight.data,
                    &mut d_norm_tmp,
                    false,
                    false,
                    1.0,
                    0.0,
                )?;
                sgemm(
                    rows,
                    d_model,
                    kv_dim,
                    &k_bshd_tmp,
                    &block.attn.k_proj.weight.data,
                    &mut d_norm_tmp,
                    false,
                    false,
                    1.0,
                    1.0,
                )?;
                sgemm(
                    rows,
                    d_model,
                    kv_dim,
                    &v_bshd_tmp,
                    &block.attn.v_proj.weight.data,
                    &mut d_norm_tmp,
                    false,
                    false,
                    1.0,
                    1.0,
                )?;

                // attn_norm backward + residual add into d_h
                rmsnorm_bwd_into_cpu(
                    &d_norm_tmp,
                    &layer_x_in[l],
                    &block.attn_norm.gamma.data,
                    &layer_rstd1[l],
                    rows,
                    d_model,
                    &mut d_norm_dx,
                    &mut grads[an_g_idx],
                    &mut dgamma_accum,
                );
                for (dhv, &dxv) in d_h.iter_mut().zip(d_norm_dx.iter()) {
                    *dhv += dxv;
                }
            }

            // Embedding scatter-add backward
            grads[0].fill(0.0);
            self.tok_embeddings
                .backward_raw(input_ids, &d_h, &mut grads[0])?;

            if let Some(max_norm) = grad_clip {
                clip_grad_vecs_inplace(&mut grads, max_norm);
            }

            // 5. Vectorized SIMD AdamW step
            let mut params_mut = self.parameters_mut();
            for (i, p) in params_mut.iter_mut().enumerate() {
                if let Some(ref mut mw) = master_weights[i] {
                    simd_adamw_step_f32(
                        mw,
                        &grads[i],
                        &mut opt_m[i],
                        &mut opt_v[i],
                        lr,
                        0.9,
                        0.999,
                        1e-8,
                        weight_decay,
                        step + 1,
                    );
                    let q = TensorStorage::from_f32_slice(mw, dtype).to_f32_vec();
                    p.data.copy_from_slice(&q);
                } else {
                    simd_adamw_step_f32(
                        &mut p.data,
                        &grads[i],
                        &mut opt_m[i],
                        &mut opt_v[i],
                        lr,
                        0.9,
                        0.999,
                        1e-8,
                        weight_decay,
                        step + 1,
                    );
                }
            }
        }

        Ok(loss_history)
    }

    #[cfg(feature = "cuda-dynamic")]
    #[allow(clippy::too_many_arguments)]
    fn train_steps_cuda_vram(
        &mut self,
        gpu_id: usize,
        input_ids: &[usize],
        targets: &[usize],
        batch: usize,
        seq_len: usize,
        steps: usize,
        lr: f32,
        weight_decay: f32,
        grad_clip: Option<f32>,
    ) -> Option<OchreResult<Vec<f32>>> {
        let rows = batch * seq_len;
        let d_model = self.config.d_model;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let hidden_dim = self.config.ffn_hidden_dim;
        let vocab_size = self.config.vocab_size;
        let num_layers = self.layers.len();
        let dtype = self.config.dtype;

        cuda_nn::with_vram_session(gpu_id, |sess| {
            // 1. Upload token IDs, targets, and RoPE tables ONCE.
            let ids_u64: Vec<u64> = input_ids.iter().map(|&v| v as u64).collect();
            let tgt_u64: Vec<u64> = targets.iter().map(|&v| v as u64).collect();
            let d_input_ids = sess.upload_u64(&ids_u64)?;
            let d_targets = sess.upload_u64(&tgt_u64)?;

            let mut d_rope_tables: Vec<Option<(CudaDeviceBuf, CudaDeviceBuf)>> =
                Vec::with_capacity(num_layers);
            for layer in &self.layers {
                if let Some(ref rope) = layer.attn.rope {
                    d_rope_tables.push(Some((
                        sess.upload_f32(&rope.cos_table)?,
                        sess.upload_f32(&rope.sin_table)?,
                    )));
                } else {
                    d_rope_tables.push(None);
                }
            }

            // 2. Upload all trainable parameters & allocate grads/AdamW moments ONCE.
            let params_host = self.parameters();
            let n_params = params_host.len();
            let mut param_lens = Vec::with_capacity(n_params);
            let mut d_params = Vec::with_capacity(n_params);
            let mut d_grads = Vec::with_capacity(n_params);
            let mut d_opt_m = Vec::with_capacity(n_params);
            let mut d_opt_v = Vec::with_capacity(n_params);
            for p in params_host {
                let len = p.data.len();
                param_lens.push(len);
                d_params.push(sess.upload_f32(&p.data)?);
                d_grads.push(sess.alloc_zeros_f32(len)?);
                d_opt_m.push(sess.alloc_zeros_f32(len)?);
                d_opt_v.push(sess.alloc_zeros_f32(len)?);
            }

            // 3. Pre-allocate per-layer activations and shared scratch buffers ONCE.
            struct CudaLayerActs {
                x_in: CudaDeviceBuf,
                xn1: CudaDeviceBuf,
                rstd1: CudaDeviceBuf,
                q_bhsd: CudaDeviceBuf,
                k_bhsd: CudaDeviceBuf,
                v_bhsd: CudaDeviceBuf,
                attn_bhsd: CudaDeviceBuf,
                lse: CudaDeviceBuf,
                attn_merged: CudaDeviceBuf,
                h_mid: CudaDeviceBuf,
                xn2: CudaDeviceBuf,
                rstd2: CudaDeviceBuf,
                gate: CudaDeviceBuf,
                up: CudaDeviceBuf,
                swiglu_out: CudaDeviceBuf,
            }

            let mut layer_acts = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                layer_acts.push(CudaLayerActs {
                    x_in: sess.alloc_f32(rows * d_model)?,
                    xn1: sess.alloc_f32(rows * d_model)?,
                    rstd1: sess.alloc_f32(rows)?,
                    q_bhsd: sess.alloc_f32(rows * d_model)?,
                    k_bhsd: sess.alloc_f32(rows * kv_dim)?,
                    v_bhsd: sess.alloc_f32(rows * kv_dim)?,
                    attn_bhsd: sess.alloc_f32(rows * d_model)?,
                    lse: sess.alloc_f32(batch * num_heads * seq_len)?,
                    attn_merged: sess.alloc_f32(rows * d_model)?,
                    h_mid: sess.alloc_f32(rows * d_model)?,
                    xn2: sess.alloc_f32(rows * d_model)?,
                    rstd2: sess.alloc_f32(rows)?,
                    gate: sess.alloc_f32(rows * hidden_dim)?,
                    up: sess.alloc_f32(rows * hidden_dim)?,
                    swiglu_out: sess.alloc_f32(rows * hidden_dim)?,
                });
            }

            let d_x_after_layers = sess.alloc_f32(rows * d_model)?;
            let d_final_norm_out = sess.alloc_f32(rows * d_model)?;
            let d_final_rstd = sess.alloc_f32(rows)?;
            let d_logits = sess.alloc_f32(rows * vocab_size)?;
            let d_dlogits = sess.alloc_f32(rows * vocab_size)?;
            let d_losses = sess.alloc_f32(rows)?;
            let d_step_loss = sess.alloc_zeros_f32(1)?;
            let d_norm_sq = sess.alloc_zeros_f32(1)?;

            let d_q_bshd_tmp = sess.alloc_f32(rows * d_model)?;
            let d_k_bshd_tmp = sess.alloc_f32(rows * kv_dim)?;
            let d_v_bshd_tmp = sess.alloc_f32(rows * kv_dim)?;

            let d_h = sess.alloc_f32(rows * d_model)?;
            let d_norm_tmp = sess.alloc_f32(rows * d_model)?;
            let d_norm_dx = sess.alloc_f32(rows * d_model)?;
            let d_attn_merged = sess.alloc_f32(rows * d_model)?;
            let d_attn_bhsd = sess.alloc_f32(rows * d_model)?;
            let d_dq_bhsd = sess.alloc_f32(rows * d_model)?;
            let d_dk_bhsd = sess.alloc_f32(rows * kv_dim)?;
            let d_dv_bhsd = sess.alloc_f32(rows * kv_dim)?;
            let d_row_scratch = sess.alloc_f32(batch * num_heads * seq_len)?;
            let d_swiglu_out = sess.alloc_f32(rows * hidden_dim)?;
            let d_dgate = sess.alloc_f32(rows * hidden_dim)?;
            let d_dup = sess.alloc_f32(rows * hidden_dim)?;

            let final_norm_g_idx = 1 + 9 * num_layers;
            let lm_head_w_idx = final_norm_g_idx + 1;
            let mut loss_history = Vec::with_capacity(steps);
            let mut host_step_loss = [0.0f32; 1];

            // 4. Training loop 100% resident in VRAM (downloading only 1 scalar loss per step).
            for step in 0..steps {
                sess.embedding_gather_dev(
                    d_params[0].ptr(),
                    d_input_ids.ptr(),
                    layer_acts[0].x_in.ptr(),
                    rows,
                    d_model,
                )?;

                for l in 0..num_layers {
                    let base = 1 + 9 * l;
                    let an_g_idx = base;
                    let wq_idx = base + 1;
                    let wk_idx = base + 2;
                    let wv_idx = base + 3;
                    let wo_idx = base + 4;
                    let fn_g_idx = base + 5;
                    let wg_idx = base + 6;
                    let wu_idx = base + 7;
                    let wd_idx = base + 8;

                    let an_eps = self.layers[l].attn_norm.eps;
                    let fn_eps = self.layers[l].ffn_norm.eps;

                    sess.rmsnorm_fwd_dev(
                        layer_acts[l].x_in.ptr(),
                        d_params[an_g_idx].ptr(),
                        layer_acts[l].xn1.ptr(),
                        layer_acts[l].rstd1.ptr(),
                        rows,
                        d_model,
                        an_eps,
                    )?;

                    sess.sgemm_dev(
                        rows,
                        d_model,
                        d_model,
                        layer_acts[l].xn1.ptr(),
                        d_params[wq_idx].ptr(),
                        d_q_bshd_tmp.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        kv_dim,
                        d_model,
                        layer_acts[l].xn1.ptr(),
                        d_params[wk_idx].ptr(),
                        d_k_bshd_tmp.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        kv_dim,
                        d_model,
                        layer_acts[l].xn1.ptr(),
                        d_params[wv_idx].ptr(),
                        d_v_bshd_tmp.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;

                    sess.bshd_to_bhsd_dev(
                        d_q_bshd_tmp.ptr(),
                        layer_acts[l].q_bhsd.ptr(),
                        batch,
                        seq_len,
                        num_heads,
                        head_dim,
                    )?;
                    sess.bshd_to_bhsd_dev(
                        d_k_bshd_tmp.ptr(),
                        layer_acts[l].k_bhsd.ptr(),
                        batch,
                        seq_len,
                        num_kv_heads,
                        head_dim,
                    )?;
                    sess.bshd_to_bhsd_dev(
                        d_v_bshd_tmp.ptr(),
                        layer_acts[l].v_bhsd.ptr(),
                        batch,
                        seq_len,
                        num_kv_heads,
                        head_dim,
                    )?;

                    if let Some((d_cos, d_sin)) = &d_rope_tables[l] {
                        sess.rope(
                            layer_acts[l].q_bhsd.ptr(),
                            d_cos.ptr(),
                            d_sin.ptr(),
                            batch,
                            num_heads,
                            seq_len,
                            head_dim,
                            0,
                            false,
                        )?;
                        sess.rope(
                            layer_acts[l].k_bhsd.ptr(),
                            d_cos.ptr(),
                            d_sin.ptr(),
                            batch,
                            num_kv_heads,
                            seq_len,
                            head_dim,
                            0,
                            false,
                        )?;
                    }

                    sess.flash_sdpa_fwd(
                        layer_acts[l].q_bhsd.ptr(),
                        layer_acts[l].k_bhsd.ptr(),
                        layer_acts[l].v_bhsd.ptr(),
                        layer_acts[l].attn_bhsd.ptr(),
                        layer_acts[l].lse.ptr(),
                        batch,
                        num_heads,
                        num_kv_heads,
                        seq_len,
                        seq_len,
                        head_dim,
                        true,
                        0,
                    )?;

                    sess.bhsd_to_bshd_dev(
                        layer_acts[l].attn_bhsd.ptr(),
                        layer_acts[l].attn_merged.ptr(),
                        batch,
                        num_heads,
                        seq_len,
                        head_dim,
                    )?;

                    sess.sgemm_dev(
                        rows,
                        d_model,
                        d_model,
                        layer_acts[l].attn_merged.ptr(),
                        d_params[wo_idx].ptr(),
                        layer_acts[l].h_mid.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.add_inplace_dev_f32(
                        layer_acts[l].h_mid.ptr(),
                        layer_acts[l].x_in.ptr(),
                        rows * d_model,
                    )?;

                    sess.rmsnorm_fwd_dev(
                        layer_acts[l].h_mid.ptr(),
                        d_params[fn_g_idx].ptr(),
                        layer_acts[l].xn2.ptr(),
                        layer_acts[l].rstd2.ptr(),
                        rows,
                        d_model,
                        fn_eps,
                    )?;

                    sess.sgemm_dev(
                        rows,
                        hidden_dim,
                        d_model,
                        layer_acts[l].xn2.ptr(),
                        d_params[wg_idx].ptr(),
                        layer_acts[l].gate.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        hidden_dim,
                        d_model,
                        layer_acts[l].xn2.ptr(),
                        d_params[wu_idx].ptr(),
                        layer_acts[l].up.ptr(),
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.swiglu_fwd(
                        layer_acts[l].gate.ptr(),
                        layer_acts[l].up.ptr(),
                        layer_acts[l].swiglu_out.ptr(),
                        rows * hidden_dim,
                    )?;

                    let next_ptr = if l + 1 < num_layers {
                        layer_acts[l + 1].x_in.ptr()
                    } else {
                        d_x_after_layers.ptr()
                    };
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        hidden_dim,
                        layer_acts[l].swiglu_out.ptr(),
                        d_params[wd_idx].ptr(),
                        next_ptr,
                        false,
                        true,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.add_inplace_dev_f32(next_ptr, layer_acts[l].h_mid.ptr(), rows * d_model)?;
                }

                sess.rmsnorm_fwd_dev(
                    d_x_after_layers.ptr(),
                    d_params[final_norm_g_idx].ptr(),
                    d_final_norm_out.ptr(),
                    d_final_rstd.ptr(),
                    rows,
                    d_model,
                    self.final_norm.eps,
                )?;
                sess.sgemm_dev(
                    rows,
                    vocab_size,
                    d_model,
                    d_final_norm_out.ptr(),
                    d_params[lm_head_w_idx].ptr(),
                    d_logits.ptr(),
                    false,
                    true,
                    1.0,
                    0.0,
                    dtype,
                )?;

                sess.fill_f32(d_step_loss.ptr(), 0.0, 1)?;
                sess.cross_entropy_grad_dev_f32(
                    d_logits.ptr(),
                    d_targets.ptr(),
                    d_dlogits.ptr(),
                    d_losses.ptr(),
                    d_step_loss.ptr(),
                    rows,
                    vocab_size,
                )?;

                // Backward in VRAM
                sess.sgemm_dev(
                    vocab_size,
                    d_model,
                    rows,
                    d_dlogits.ptr(),
                    d_final_norm_out.ptr(),
                    d_grads[lm_head_w_idx].ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    dtype,
                )?;
                sess.sgemm_dev(
                    rows,
                    d_model,
                    vocab_size,
                    d_dlogits.ptr(),
                    d_params[lm_head_w_idx].ptr(),
                    d_norm_tmp.ptr(),
                    false,
                    false,
                    1.0,
                    0.0,
                    dtype,
                )?;
                sess.rmsnorm_bwd_dev(
                    d_norm_tmp.ptr(),
                    d_x_after_layers.ptr(),
                    d_params[final_norm_g_idx].ptr(),
                    d_final_rstd.ptr(),
                    d_h.ptr(),
                    d_grads[final_norm_g_idx].ptr(),
                    rows,
                    d_model,
                    true,
                )?;

                for l in (0..num_layers).rev() {
                    let base = 1 + 9 * l;
                    let an_g_idx = base;
                    let wq_idx = base + 1;
                    let wk_idx = base + 2;
                    let wv_idx = base + 3;
                    let wo_idx = base + 4;
                    let fn_g_idx = base + 5;
                    let wg_idx = base + 6;
                    let wu_idx = base + 7;
                    let wd_idx = base + 8;

                    sess.sgemm_dev(
                        d_model,
                        hidden_dim,
                        rows,
                        d_h.ptr(),
                        layer_acts[l].swiglu_out.ptr(),
                        d_grads[wd_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        hidden_dim,
                        d_model,
                        d_h.ptr(),
                        d_params[wd_idx].ptr(),
                        d_swiglu_out.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;

                    sess.swiglu_bwd(
                        d_swiglu_out.ptr(),
                        layer_acts[l].gate.ptr(),
                        layer_acts[l].up.ptr(),
                        d_dgate.ptr(),
                        d_dup.ptr(),
                        rows * hidden_dim,
                    )?;

                    sess.sgemm_dev(
                        hidden_dim,
                        d_model,
                        rows,
                        d_dgate.ptr(),
                        layer_acts[l].xn2.ptr(),
                        d_grads[wg_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        hidden_dim,
                        d_model,
                        rows,
                        d_dup.ptr(),
                        layer_acts[l].xn2.ptr(),
                        d_grads[wu_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        hidden_dim,
                        d_dgate.ptr(),
                        d_params[wg_idx].ptr(),
                        d_norm_tmp.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        hidden_dim,
                        d_dup.ptr(),
                        d_params[wu_idx].ptr(),
                        d_norm_tmp.ptr(),
                        false,
                        false,
                        1.0,
                        1.0,
                        dtype,
                    )?;

                    sess.rmsnorm_bwd_dev(
                        d_norm_tmp.ptr(),
                        layer_acts[l].h_mid.ptr(),
                        d_params[fn_g_idx].ptr(),
                        layer_acts[l].rstd2.ptr(),
                        d_norm_dx.ptr(),
                        d_grads[fn_g_idx].ptr(),
                        rows,
                        d_model,
                        true,
                    )?;
                    sess.add_inplace_dev_f32(d_h.ptr(), d_norm_dx.ptr(), rows * d_model)?;

                    sess.sgemm_dev(
                        d_model,
                        d_model,
                        rows,
                        d_h.ptr(),
                        layer_acts[l].attn_merged.ptr(),
                        d_grads[wo_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        d_model,
                        d_h.ptr(),
                        d_params[wo_idx].ptr(),
                        d_attn_merged.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;

                    sess.bshd_to_bhsd_dev(
                        d_attn_merged.ptr(),
                        d_attn_bhsd.ptr(),
                        batch,
                        seq_len,
                        num_heads,
                        head_dim,
                    )?;
                    sess.flash_sdpa_bwd(
                        d_attn_bhsd.ptr(),
                        layer_acts[l].q_bhsd.ptr(),
                        layer_acts[l].k_bhsd.ptr(),
                        layer_acts[l].v_bhsd.ptr(),
                        layer_acts[l].attn_bhsd.ptr(),
                        layer_acts[l].lse.ptr(),
                        d_row_scratch.ptr(),
                        d_dq_bhsd.ptr(),
                        d_dk_bhsd.ptr(),
                        d_dv_bhsd.ptr(),
                        batch,
                        num_heads,
                        num_kv_heads,
                        seq_len,
                        seq_len,
                        head_dim,
                        true,
                        0,
                    )?;

                    if let Some((d_cos, d_sin)) = &d_rope_tables[l] {
                        sess.rope(
                            d_dq_bhsd.ptr(),
                            d_cos.ptr(),
                            d_sin.ptr(),
                            batch,
                            num_heads,
                            seq_len,
                            head_dim,
                            0,
                            true,
                        )?;
                        sess.rope(
                            d_dk_bhsd.ptr(),
                            d_cos.ptr(),
                            d_sin.ptr(),
                            batch,
                            num_kv_heads,
                            seq_len,
                            head_dim,
                            0,
                            true,
                        )?;
                    }

                    sess.bhsd_to_bshd_dev(
                        d_dq_bhsd.ptr(),
                        d_q_bshd_tmp.ptr(),
                        batch,
                        num_heads,
                        seq_len,
                        head_dim,
                    )?;
                    sess.bhsd_to_bshd_dev(
                        d_dk_bhsd.ptr(),
                        d_k_bshd_tmp.ptr(),
                        batch,
                        num_kv_heads,
                        seq_len,
                        head_dim,
                    )?;
                    sess.bhsd_to_bshd_dev(
                        d_dv_bhsd.ptr(),
                        d_v_bshd_tmp.ptr(),
                        batch,
                        num_kv_heads,
                        seq_len,
                        head_dim,
                    )?;

                    sess.sgemm_dev(
                        d_model,
                        d_model,
                        rows,
                        d_q_bshd_tmp.ptr(),
                        layer_acts[l].xn1.ptr(),
                        d_grads[wq_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        kv_dim,
                        d_model,
                        rows,
                        d_k_bshd_tmp.ptr(),
                        layer_acts[l].xn1.ptr(),
                        d_grads[wk_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        kv_dim,
                        d_model,
                        rows,
                        d_v_bshd_tmp.ptr(),
                        layer_acts[l].xn1.ptr(),
                        d_grads[wv_idx].ptr(),
                        true,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;

                    sess.sgemm_dev(
                        rows,
                        d_model,
                        d_model,
                        d_q_bshd_tmp.ptr(),
                        d_params[wq_idx].ptr(),
                        d_norm_tmp.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        kv_dim,
                        d_k_bshd_tmp.ptr(),
                        d_params[wk_idx].ptr(),
                        d_norm_tmp.ptr(),
                        false,
                        false,
                        1.0,
                        1.0,
                        dtype,
                    )?;
                    sess.sgemm_dev(
                        rows,
                        d_model,
                        kv_dim,
                        d_v_bshd_tmp.ptr(),
                        d_params[wv_idx].ptr(),
                        d_norm_tmp.ptr(),
                        false,
                        false,
                        1.0,
                        1.0,
                        dtype,
                    )?;

                    sess.rmsnorm_bwd_dev(
                        d_norm_tmp.ptr(),
                        layer_acts[l].x_in.ptr(),
                        d_params[an_g_idx].ptr(),
                        layer_acts[l].rstd1.ptr(),
                        d_norm_dx.ptr(),
                        d_grads[an_g_idx].ptr(),
                        rows,
                        d_model,
                        true,
                    )?;
                    sess.add_inplace_dev_f32(d_h.ptr(), d_norm_dx.ptr(), rows * d_model)?;
                }

                sess.fill_f32(d_grads[0].ptr(), 0.0, vocab_size * d_model)?;
                sess.embedding_scatter_add_dev(
                    d_h.ptr(),
                    d_input_ids.ptr(),
                    d_grads[0].ptr(),
                    rows,
                    d_model,
                )?;

                if let Some(max_norm) = grad_clip
                    && max_norm > 0.0
                {
                    sess.fill_f32(d_norm_sq.ptr(), 0.0, 1)?;
                    for (dg, &len) in d_grads.iter().zip(param_lens.iter()) {
                        sess.reduce_sum_sq_add_f32(dg.ptr(), d_norm_sq.ptr(), len)?;
                    }
                    for (dg, &len) in d_grads.iter().zip(param_lens.iter()) {
                        sess.clip_scale_dev_f32(dg.ptr(), d_norm_sq.ptr(), max_norm, len)?;
                    }
                }

                for i in 0..n_params {
                    sess.adamw_step_dev_f32(
                        d_params[i].ptr(),
                        d_grads[i].ptr(),
                        d_opt_m[i].ptr(),
                        d_opt_v[i].ptr(),
                        param_lens[i],
                        lr,
                        0.9,
                        0.999,
                        1e-8,
                        weight_decay,
                        step + 1,
                    )?;
                }

                sess.dtoh_f32(&mut host_step_loss, &d_step_loss)?;
                loss_history.push(host_step_loss[0] / (rows as f32));
            }

            // 5. Download updated weights back into `self` ONCE at the end.
            let mut params_mut = self.parameters_mut();
            for (i, p) in params_mut.iter_mut().enumerate() {
                let mut host_p = vec![0.0f32; param_lens[i]];
                sess.dtoh_f32(&mut host_p, &d_params[i])?;
                let final_p = if dtype != DType::F32 {
                    TensorStorage::from_f32_slice(&host_p, dtype).to_f32_vec()
                } else {
                    host_p
                };
                p.data.copy_from_slice(&final_p);
            }

            Ok(loss_history)
        })
    }

    /// Returns references to all trainable parameters in the `CausalLM`.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.tok_embeddings.weight];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.push(&self.lm_head.weight);
        if let Some(ref b) = self.lm_head.bias {
            params.push(b);
        }
        params
    }

    /// Returns mutable references to all trainable parameters in the `CausalLM`.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.tok_embeddings.weight];
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.final_norm.parameters_mut());
        params.push(&mut self.lm_head.weight);
        if let Some(ref mut b) = self.lm_head.bias {
            params.push(b);
        }
        params
    }

    /// Serializes the `CausalLM` into `DIT\x01` format with `TAG_NN_TRANSFORMER_LM` (`101`).
    pub fn to_dit_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_NN_TRANSFORMER_LM);
        write_usize(&mut buf, self.config.vocab_size);
        write_usize(&mut buf, self.config.d_model);
        write_usize(&mut buf, self.config.num_layers);
        write_usize(&mut buf, self.config.num_heads);
        write_usize(&mut buf, self.config.num_kv_heads);
        write_usize(&mut buf, self.config.ffn_hidden_dim);
        write_usize(&mut buf, self.config.max_seq_len);
        write_f32(&mut buf, self.config.rope_theta);
        write_f32(&mut buf, self.config.norm_eps);
        write_u8(&mut buf, dtype_to_tag(self.config.dtype));

        write_f32_vec(&mut buf, &self.tok_embeddings.weight.data);
        for block in &self.layers {
            write_f32_vec(&mut buf, &block.attn_norm.gamma.data);
            write_linear(&mut buf, &block.attn.q_proj);
            write_linear(&mut buf, &block.attn.k_proj);
            write_linear(&mut buf, &block.attn.v_proj);
            write_linear(&mut buf, &block.attn.o_proj);
            write_f32_vec(&mut buf, &block.ffn_norm.gamma.data);
            write_linear(&mut buf, &block.ffn.w_gate);
            write_linear(&mut buf, &block.ffn.w_up);
            write_linear(&mut buf, &block.ffn.w_down);
        }
        write_f32_vec(&mut buf, &self.final_norm.gamma.data);
        write_linear(&mut buf, &self.lm_head);
        buf
    }

    /// Deserializes a `CausalLM` from `DIT\x01` bytes onto `device`.
    pub fn from_dit_bytes(bytes: &[u8], device: DeviceType) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_NN_TRANSFORMER_LM)?;

        let vocab_size = r.read_usize()?;
        let d_model = r.read_usize()?;
        let num_layers = r.read_usize()?;
        let num_heads = r.read_usize()?;
        let num_kv_heads = r.read_usize()?;
        let ffn_hidden_dim = r.read_usize()?;
        let max_seq_len = r.read_usize()?;
        let rope_theta = read_f32(&mut r)?;
        let norm_eps = read_f32(&mut r)?;
        let dtype = tag_to_dtype(r.read_u8()?)?;

        let config = CausalLMConfig {
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            num_kv_heads,
            ffn_hidden_dim,
            max_seq_len,
            rope_theta,
            norm_eps,
            dtype,
        };
        config.validate()?;

        let emb_data = read_f32_vec(&mut r)?;
        let emb_weight =
            Tensor::parameter(emb_data, vec![vocab_size, d_model], dtype, device.clone())?;
        let tok_embeddings = Embedding::from_weight(emb_weight)?;

        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let rope = RoPE::new(head_dim, max_seq_len, rope_theta)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let an_g =
                Tensor::parameter(read_f32_vec(&mut r)?, vec![d_model], dtype, device.clone())?;
            let attn_norm = RMSNorm::from_weight(d_model, norm_eps, an_g)?;

            let q_proj = read_linear(&mut r, d_model, d_model, dtype, &device)?;
            let k_proj = read_linear(&mut r, d_model, kv_dim, dtype, &device)?;
            let v_proj = read_linear(&mut r, d_model, kv_dim, dtype, &device)?;
            let o_proj = read_linear(&mut r, d_model, d_model, dtype, &device)?;

            let attn = MultiHeadAttention {
                d_model,
                num_heads,
                num_kv_heads,
                head_dim,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rope: Some(rope.clone()),
                causal: true,
                dtype,
                device: device.clone(),
            };

            let fn_g =
                Tensor::parameter(read_f32_vec(&mut r)?, vec![d_model], dtype, device.clone())?;
            let ffn_norm = RMSNorm::from_weight(d_model, norm_eps, fn_g)?;

            let w_gate = read_linear(&mut r, d_model, ffn_hidden_dim, dtype, &device)?;
            let w_up = read_linear(&mut r, d_model, ffn_hidden_dim, dtype, &device)?;
            let w_down = read_linear(&mut r, ffn_hidden_dim, d_model, dtype, &device)?;
            let ffn = SwiGLU {
                w_gate,
                w_up,
                w_down,
                d_model,
                hidden_dim: ffn_hidden_dim,
            };

            layers.push(CausalLMBlock {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let final_g =
            Tensor::parameter(read_f32_vec(&mut r)?, vec![d_model], dtype, device.clone())?;
        let final_norm = RMSNorm::from_weight(d_model, norm_eps, final_g)?;
        let lm_head = read_linear(&mut r, d_model, vocab_size, dtype, &device)?;

        Ok(Self {
            config,
            tok_embeddings,
            layers,
            final_norm,
            lm_head,
            device,
        })
    }
}

fn sample_next_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut FastRng,
    candidates: &mut Vec<(usize, f32)>,
) -> usize {
    if temperature <= 1e-6 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx;
    }

    let inv_t = 1.0f32 / temperature;
    candidates.clear();
    candidates.extend(logits.iter().enumerate().map(|(i, &v)| (i, v * inv_t)));

    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max_l = candidates[0].1;
    let mut sum_exp = 0.0f64;
    for (_, val) in candidates.iter_mut() {
        let e = ((*val - max_l) as f64).exp();
        *val = e as f32;
        sum_exp += e;
    }
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return candidates[0].0;
    }
    let inv_sum = 1.0f64 / sum_exp;
    for (_, val) in candidates.iter_mut() {
        *val = ((*val as f64) * inv_sum) as f32;
    }

    if top_p < 1.0 {
        let mut cum = 0.0f64;
        let mut keep = 0usize;
        for (_, p) in candidates.iter() {
            cum += *p as f64;
            keep += 1;
            if cum >= top_p as f64 {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        let renorm: f64 = candidates.iter().map(|(_, p)| *p as f64).sum();
        if renorm > 0.0 {
            let inv_renorm = 1.0f64 / renorm;
            for (_, p) in candidates.iter_mut() {
                *p = ((*p as f64) * inv_renorm) as f32;
            }
        }
    }

    let u = rng.next_f32() as f64;
    let mut acc = 0.0f64;
    for &(idx, p) in candidates.iter() {
        acc += p as f64;
        if u <= acc {
            return idx;
        }
    }
    candidates[candidates.len() - 1].0
}

fn dtype_to_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
    }
}

fn tag_to_dtype(tag: u8) -> OchreResult<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::BF16),
        2 => Ok(DType::F16),
        _ => Err(OchreError::CorruptedDitFile),
    }
}

fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f32(buf, x);
    }
}

fn read_f32(r: &mut ByteReader<'_>) -> OchreResult<f32> {
    let b = r.read_exact(4)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(b);
    Ok(f32::from_le_bytes(arr))
}

fn read_f32_vec(r: &mut ByteReader<'_>) -> OchreResult<Vec<f32>> {
    let len = r.read_usize()?;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(read_f32(r)?);
    }
    Ok(vec)
}

fn write_linear(buf: &mut Vec<u8>, lin: &Linear) {
    write_f32_vec(buf, &lin.weight.data);
    match &lin.bias {
        Some(b) => {
            write_bool(buf, true);
            write_f32_vec(buf, &b.data);
        }
        None => write_bool(buf, false),
    }
}

fn read_linear(
    r: &mut ByteReader<'_>,
    in_features: usize,
    out_features: usize,
    dtype: DType,
    device: &DeviceType,
) -> OchreResult<Linear> {
    let w_data = read_f32_vec(r)?;
    let weight = Tensor::parameter(
        w_data,
        vec![out_features, in_features],
        dtype,
        device.clone(),
    )?;
    let bias = if r.read_bool()? {
        let b_data = read_f32_vec(r)?;
        Some(Tensor::parameter(
            b_data,
            vec![out_features],
            dtype,
            device.clone(),
        )?)
    } else {
        None
    };
    Linear::from_weights(weight, bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::{AutogradTape, with_no_grad};

    #[test]
    fn test_swiglu_and_feedforward_finite_diff_grad_check() {
        AutogradTape::clear();
        let swiglu = SwiGLU::new(4, 6, true, DType::F32, DeviceType::Cpu, 11).unwrap();
        let x = Tensor::parameter(
            vec![0.4, -0.8, 1.1, -0.2, 0.3, 0.9, -0.5, 0.7],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::new(
            vec![0.6, -0.4, 0.9, 0.2, -0.3, 0.8, -0.7, 0.5],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let y_sg = swiglu.forward(&x).unwrap();
        let mut raw_sg = vec![0.0f32; 8];
        swiglu.forward_raw(&x.data, 2, &mut raw_sg).unwrap();
        for (&a, &b) in y_sg.data.iter().zip(raw_sg.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let grads_sg = y_sg.mul(&w).unwrap().sum(None).unwrap().backward().unwrap();
        let dx_sg = grads_sg.get(x.node_id.unwrap()).unwrap();
        let eps = 1e-3f32;
        for i in 0..x.numel() {
            let mut xp = x.data.clone();
            let mut xm = x.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(xp, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm = Tensor::new(xm, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = swiglu
                    .forward(&tp)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = swiglu
                    .forward(&tm)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx_sg[i] - num).abs() < 5e-3,
                "SwiGLU dx[{i}] mismatch: {} vs {num}",
                dx_sg[i]
            );
        }

        AutogradTape::clear();
        let ffn = FeedForward::new(4, 6, true, DType::F32, DeviceType::Cpu, 23).unwrap();
        let x2 =
            Tensor::parameter(x.data.clone(), vec![2, 4], DType::F32, DeviceType::Cpu).unwrap();
        let y_ff = ffn.forward(&x2).unwrap();
        let mut raw_ff = vec![0.0f32; 8];
        ffn.forward_raw(&x2.data, 2, &mut raw_ff).unwrap();
        for (&a, &b) in y_ff.data.iter().zip(raw_ff.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let grads_ff = y_ff.mul(&w).unwrap().sum(None).unwrap().backward().unwrap();
        let dx_ff = grads_ff.get(x2.node_id.unwrap()).unwrap();
        for i in 0..x2.numel() {
            let mut xp = x2.data.clone();
            let mut xm = x2.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(xp, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm = Tensor::new(xm, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = ffn
                    .forward(&tp)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = ffn
                    .forward(&tm)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx_ff[i] - num).abs() < 5e-3,
                "FeedForward dx[{i}] mismatch: {} vs {num}",
                dx_ff[i]
            );
        }
    }

    #[test]
    fn test_encoder_and_decoder_autograd() {
        AutogradTape::clear();
        let enc = TransformerEncoder::new(
            2,
            8,
            2,
            16,
            1e-5,
            true,
            true,
            DType::F32,
            DeviceType::Cpu,
            7,
        )
        .unwrap();
        let src = Tensor::randn_with(
            vec![1, 4, 8],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            3,
            true,
        );
        let mem = enc.forward(&src).unwrap();
        assert_eq!(mem.shape, vec![1, 4, 8]);

        let dec = TransformerDecoder::new(
            2,
            8,
            2,
            16,
            1e-5,
            true,
            true,
            false,
            true,
            DType::F32,
            DeviceType::Cpu,
            19,
        )
        .unwrap();
        let tgt = Tensor::randn_with(
            vec![1, 3, 8],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            5,
            true,
        );
        let out = dec.forward(&tgt, Some(&mem), None).unwrap();
        assert_eq!(out.shape, vec![1, 3, 8]);
        let grads = out.mean(None).unwrap().backward().unwrap();
        assert!(grads.get(src.node_id.unwrap()).is_some());
        assert!(grads.get(tgt.node_id.unwrap()).is_some());
    }

    #[test]
    fn test_causallm_kv_cache_generate_and_dit_roundtrip() {
        let cfg = CausalLMConfig {
            vocab_size: 32,
            d_model: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            ffn_hidden_dim: 32,
            max_seq_len: 32,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
        };
        let lm = CausalLM::new(cfg.clone(), DeviceType::Cpu, 42).unwrap();

        let prompt = [1usize, 5, 12, 3, 9];
        let full_logits = with_no_grad(|| lm.forward_ids(&prompt, 1, prompt.len(), None).unwrap());
        assert_eq!(full_logits.shape, vec![1, prompt.len(), cfg.vocab_size]);

        let mut caches = lm.new_kv_caches(1).unwrap();
        let prefill_logits = with_no_grad(|| {
            lm.forward_ids(&prompt[..3], 1, 3, Some(&mut caches))
                .unwrap()
        });
        let step1_logits = with_no_grad(|| {
            lm.forward_ids(&prompt[3..4], 1, 1, Some(&mut caches))
                .unwrap()
        });
        let step2_logits = with_no_grad(|| {
            lm.forward_ids(&prompt[4..5], 1, 1, Some(&mut caches))
                .unwrap()
        });
        let mut combined = prefill_logits.data.clone();
        combined.extend_from_slice(&step1_logits.data);
        combined.extend_from_slice(&step2_logits.data);
        let max_err = full_logits
            .data
            .iter()
            .zip(combined.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "CausalLM KVCache parity max_err={max_err}");

        // Generation test (greedy and top-k/top-p)
        let gen_greedy = lm.generate(&[1, 5, 12], 5, 0.0, 0, 1.0, 7).unwrap();
        assert_eq!(gen_greedy.len(), 8);
        assert_eq!(&gen_greedy[..3], &[1, 5, 12]);
        assert!(gen_greedy.iter().all(|&id| id < cfg.vocab_size));

        let gen_sampled = lm.generate(&[1, 5, 12], 4, 0.8, 8, 0.9, 99).unwrap();
        assert_eq!(gen_sampled.len(), 7);
        assert!(gen_sampled.iter().all(|&id| id < cfg.vocab_size));

        // .dit save/load round-trip
        let bytes = lm.to_dit_bytes();
        assert_eq!(&bytes[0..4], b"DIT\x01");
        assert_eq!(bytes[4], TAG_NN_TRANSFORMER_LM);

        let loaded = CausalLM::from_dit_bytes(&bytes, DeviceType::Cpu).unwrap();
        assert_eq!(loaded.config, lm.config);
        let loaded_logits =
            with_no_grad(|| loaded.forward_ids(&prompt, 1, prompt.len(), None).unwrap());
        let roundtrip_err = full_logits
            .data
            .iter()
            .zip(loaded_logits.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(roundtrip_err < 1e-6);

        let loaded_greedy = loaded.generate(&[1, 5, 12], 5, 0.0, 0, 1.0, 7).unwrap();
        assert_eq!(gen_greedy, loaded_greedy);
    }

    #[test]
    fn test_causallm_train_steps_parity_and_loss_reduction() {
        use crate::nn::losses::CrossEntropyLoss;
        use crate::nn::optim::{AdamW, clip_grad_store_norm_};

        let cfg = CausalLMConfig {
            vocab_size: 16,
            d_model: 8,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            ffn_hidden_dim: 16,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
        };

        let mut lm_fast = CausalLM::new(cfg.clone(), DeviceType::Cpu, 123).unwrap();
        let mut lm_ref = lm_fast.clone();

        let batch = 2usize;
        let seq_len = 5usize;
        let input_ids = [1usize, 3, 5, 7, 9, 2, 4, 6, 8, 10];
        let targets = [3usize, 5, 7, 9, 11, 4, 6, 8, 10, 12];
        let steps = 8usize;
        let lr = 0.03f32;
        let wd = 0.01f32;

        let fast_losses = lm_fast
            .train_steps(
                &input_ids,
                &targets,
                batch,
                seq_len,
                steps,
                lr,
                wd,
                Some(1.0),
            )
            .unwrap();
        assert_eq!(fast_losses.len(), steps);

        let ce = CrossEntropyLoss::default();
        let mut adamw = AdamW::with_lr(lr, wd, DeviceType::Cpu).unwrap();
        let mut ref_losses = Vec::with_capacity(steps);
        for _ in 0..steps {
            AutogradTape::clear();
            let logits = lm_ref
                .forward_ids(&input_ids, batch, seq_len, None)
                .unwrap();
            let loss_t = ce.forward_tensor(&logits, &targets).unwrap();
            ref_losses.push(loss_t.data[0]);
            let mut grads = loss_t.backward().unwrap();
            let mut params = lm_ref.parameters_mut();
            let node_ids: Vec<usize> = params.iter().filter_map(|p| p.node_id).collect();
            clip_grad_store_norm_(&mut grads, &node_ids, 1.0);
            adamw.step(&mut params, &grads).unwrap();
        }

        for (step, (&fl, &rl)) in fast_losses.iter().zip(ref_losses.iter()).enumerate() {
            assert!(
                (fl - rl).abs() < 2e-4,
                "CausalLM::train_steps loss mismatch at step {step}: {fl} vs {rl}"
            );
        }

        let fast_params = lm_fast.parameters();
        let ref_params = lm_ref.parameters();
        assert_eq!(fast_params.len(), ref_params.len());
        for (pi, (fp, rp)) in fast_params.iter().zip(ref_params.iter()).enumerate() {
            let max_diff = fp
                .data
                .iter()
                .zip(rp.data.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 2e-4,
                "CausalLM::train_steps param[{pi}] max_diff={max_diff}"
            );
        }

        // Loss reduction + GPU fallback test
        let mut lm_gpu = CausalLM::new(cfg, DeviceType::Gpu(0), 123).unwrap();
        let gpu_losses = lm_gpu
            .train_steps(
                &input_ids,
                &targets,
                batch,
                seq_len,
                25,
                0.05,
                0.0,
                Some(1.0),
            )
            .unwrap();
        assert_eq!(gpu_losses.len(), 25);
        assert!(gpu_losses.iter().all(|v| v.is_finite()));
        assert!(
            gpu_losses[24] < gpu_losses[0],
            "expected loss to drop: {} -> {}",
            gpu_losses[0],
            gpu_losses[24]
        );
    }

    #[test]
    fn test_causallm_train_steps_validation_errors() {
        let cfg = CausalLMConfig {
            vocab_size: 8,
            d_model: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            ffn_hidden_dim: 8,
            max_seq_len: 8,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
        };
        let mut lm = CausalLM::new(cfg, DeviceType::Cpu, 1).unwrap();
        assert!(
            lm.train_steps(&[0, 1], &[1, 2], 1, 2, 0, 0.01, 0.0, None)
                .is_err()
        );
        assert!(
            lm.train_steps(&[0, 1], &[1, 2], 1, 2, 2, -0.1, 0.0, None)
                .is_err()
        );
        assert!(
            lm.train_steps(&[0, 1], &[1], 1, 2, 2, 0.01, 0.0, None)
                .is_err()
        );
        assert!(
            lm.train_steps(&[0, 8], &[1, 2], 1, 2, 2, 0.01, 0.0, None)
                .is_err()
        );
        assert!(
            lm.train_steps(&[0, 1], &[1, 8], 1, 2, 2, 0.01, 0.0, None)
                .is_err()
        );
    }
}
