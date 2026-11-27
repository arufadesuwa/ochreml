// OchreML Transformer, SwiGLU, FeedForward & CausalLM Module (*≧ω≦*)
// Implements SwiGLU, GELU FeedForward, TransformerEncoder/Decoder, and LLaMA-style CausalLM
// with GQA, RoPE, KVCache generation, and DIT\x01 tag 101 serialization.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::attention::{KVCache, MultiHeadAttention, RoPE};
use crate::nn::dtype::DType;
use crate::nn::layers::{Embedding, Linear};
use crate::nn::norm::{LayerNorm, RMSNorm};
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
}
