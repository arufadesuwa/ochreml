// OchreML Attention, Positional Encoding, RoPE, KVCache & FlashAttention Module (*≧ω≦*)
// Implements Sinusoidal PositionalEncoding, Rotary Positional Embeddings (RoPE),
// Flash-Style O(S)-memory tiled ScaledDotProductAttention (GQA/MQA + causal), KVCache, and MHA.

use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, CustomBackward, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::{sgemm, sgemm_serial};
use crate::nn::layers::Linear;
use crate::nn::simd_math::simd_exp_sub_scalar_into;
use crate::nn::tensor::{Tensor, compute_strides};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

// 48 is an exact multiple of SGEMM MR=6 and NR=16, and < MC=64 so tile sgemm
// never spawns nested Rayon tasks inside head-parallel FlashAttention loops.
const FLASH_BR: usize = 48;
const FLASH_BC: usize = 48;
const DECODE_BC: usize = 64;

#[derive(Default)]
struct FlashScratch {
    o_tile: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    dp_tile: Vec<f32>,
    ds_tile: Vec<f32>,
    d_row: Vec<f32>,
}

#[derive(Default)]
struct MhaRawScratch {
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_heads: Vec<f32>,
    k_heads: Vec<f32>,
    v_heads: Vec<f32>,
    attn_out: Vec<f32>,
    merged: Vec<f32>,
}

thread_local! {
    static FLASH_SCRATCH: RefCell<FlashScratch> = RefCell::new(FlashScratch::default());
    static MHA_RAW_SCRATCH: RefCell<MhaRawScratch> = RefCell::new(MhaRawScratch::default());
}

#[inline]
fn ensure_len(buf: &mut Vec<f32>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for i in 0..chunks {
        let off = i * 4;
        s0 += a[off] * b[off];
        s1 += a[off + 1] * b[off + 1];
        s2 += a[off + 2] * b[off + 2];
        s3 += a[off + 3] * b[off + 3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn axpy_f32(y: &mut [f32], x: &[f32], alpha: f32) {
    let n = y.len().min(x.len());
    for (yv, &xv) in y[..n].iter_mut().zip(x[..n].iter()) {
        *yv += alpha * xv;
    }
}

#[cfg(feature = "cuda-dynamic")]
const CUDA_FLASH_DISPATCH_THRESHOLD: usize = 16_384;

#[cfg(feature = "cuda-dynamic")]
#[inline]
fn should_dispatch_cuda_flash(batch: usize, num_heads: usize, q_len: usize, kv_len: usize) -> bool {
    batch
        .saturating_mul(num_heads)
        .saturating_mul(q_len)
        .saturating_mul(kv_len)
        >= CUDA_FLASH_DISPATCH_THRESHOLD
}

#[cfg(feature = "cuda-dynamic")]
#[inline]
fn cuda_gpu_id(device: &DeviceType) -> Option<usize> {
    match device {
        DeviceType::Gpu(id) => Some(*id),
        DeviceType::MultiGpu(ids) => ids.first().copied(),
        _ => None,
    }
}

#[inline]
fn maybe_quantize(data: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => data,
        DType::BF16 | DType::F16 => TensorStorage::from_f32_slice(&data, dtype).to_f32_vec(),
    }
}

/// Sinusoidal positional encoding table of shape `[max_len, d_model]`.
#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    pub max_len: usize,
    pub d_model: usize,
    pub table: Vec<f32>,
}

impl PositionalEncoding {
    /// Precomputes sinusoidal positional encodings `[max_len, d_model]`.
    pub fn new(max_len: usize, d_model: usize) -> OchreResult<Self> {
        if max_len == 0 || d_model == 0 {
            return Err(OchreError::InvalidParameter(
                "PositionalEncoding max_len and d_model must be > 0 ( >_< )".to_string(),
            ));
        }
        let mut table = vec![0.0f32; max_len * d_model];
        let denom_base = 10_000.0f64;
        for pos in 0..max_len {
            let row = &mut table[pos * d_model..(pos + 1) * d_model];
            for (i, slot) in row.iter_mut().enumerate() {
                let pair_idx = (i / 2) * 2;
                let div = denom_base.powf((pair_idx as f64) / (d_model as f64));
                let angle = (pos as f64) / div;
                *slot = if i % 2 == 0 {
                    angle.sin() as f32
                } else {
                    angle.cos() as f32
                };
            }
        }
        Ok(Self {
            max_len,
            d_model,
            table,
        })
    }

    /// Adds sinusoidal positional encoding in-place to raw `[batch, seq_len, d_model]`.
    pub fn forward_raw(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        offset: usize,
    ) -> OchreResult<()> {
        if offset.saturating_add(seq_len) > self.max_len {
            return Err(OchreError::InvalidParameter(format!(
                "PositionalEncoding offset {offset} + seq_len {seq_len} exceeds max_len {} ( >_< )",
                self.max_len
            )));
        }
        let expected = batch * seq_len * self.d_model;
        if x.len() != expected {
            return Err(OchreError::DimensionMismatch {
                expected_rows: batch * seq_len,
                expected_cols: self.d_model,
                actual_rows: x.len(),
                actual_cols: 1,
            });
        }
        for b in 0..batch {
            for s in 0..seq_len {
                let pe_row =
                    &self.table[(offset + s) * self.d_model..(offset + s + 1) * self.d_model];
                let x_row =
                    &mut x[(b * seq_len + s) * self.d_model..(b * seq_len + s + 1) * self.d_model];
                for (dst, &pe) in x_row.iter_mut().zip(pe_row.iter()) {
                    *dst += pe;
                }
            }
        }
        Ok(())
    }

    /// Autograd-compatible positional encoding addition for `[seq_len, d_model]` or `[batch, seq_len, d_model]`.
    pub fn forward(&self, x: &Tensor, offset: usize) -> OchreResult<Tensor> {
        let (seq_len, d_model) = match x.shape.as_slice() {
            [s, d] => (*s, *d),
            [_, s, d] => (*s, *d),
            _ => {
                return Err(OchreError::InvalidParameter(format!(
                    "PositionalEncoding expects [seq, d_model] or [batch, seq, d_model], got {:?} ( >_< )",
                    x.shape
                )));
            }
        };
        if d_model != self.d_model {
            return Err(OchreError::InvalidParameter(format!(
                "PositionalEncoding expected d_model={}, got {d_model} ( >_< )",
                self.d_model
            )));
        }
        if offset.saturating_add(seq_len) > self.max_len {
            return Err(OchreError::InvalidParameter(format!(
                "PositionalEncoding offset {offset} + seq_len {seq_len} exceeds max_len {} ( >_< )",
                self.max_len
            )));
        }
        let start = offset * self.d_model;
        let end = (offset + seq_len) * self.d_model;
        let pe = Tensor::new(
            self.table[start..end].to_vec(),
            vec![seq_len, self.d_model],
            x.dtype,
            x.device.clone(),
            false,
        )?;
        x.add(&pe)
    }
}

/// Rotary Positional Embeddings (RoPE) with cached cosine/sine tables.
#[derive(Clone, Debug)]
pub struct RoPE {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
    pub cos_table: Arc<Vec<f32>>,
    pub sin_table: Arc<Vec<f32>>,
}

impl RoPE {
    /// Precomputes RoPE cos/sin tables `[max_seq_len, head_dim / 2]` for `theta` (default `10000.0`).
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> OchreResult<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(OchreError::InvalidParameter(format!(
                "RoPE head_dim={head_dim} must be a positive even integer ( >_< )"
            )));
        }
        if max_seq_len == 0 {
            return Err(OchreError::InvalidParameter(
                "RoPE max_seq_len must be > 0 ( >_< )".to_string(),
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "RoPE theta={theta} must be finite and > 0 ( >_< )"
            )));
        }
        let half = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * half];
        let mut sin_table = vec![0.0f32; max_seq_len * half];
        let theta_d = theta as f64;
        let dim_d = head_dim as f64;

        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta_d.powf((2 * i) as f64 / dim_d))
            .collect();

        for pos in 0..max_seq_len {
            let p = pos as f64;
            let row_off = pos * half;
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = p * freq;
                cos_table[row_off + i] = angle.cos() as f32;
                sin_table[row_off + i] = angle.sin() as f32;
            }
        }

        Ok(Self {
            head_dim,
            max_seq_len,
            theta,
            cos_table: Arc::new(cos_table),
            sin_table: Arc::new(sin_table),
        })
    }

    /// Applies RoPE in-place to `[batch, num_heads, seq_len, head_dim]` starting at `offset`.
    pub fn apply_inplace(
        &self,
        x: &mut [f32],
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> OchreResult<()> {
        apply_rope_slice(
            x,
            &self.cos_table,
            &self.sin_table,
            self.head_dim,
            self.max_seq_len,
            batch,
            num_heads,
            seq_len,
            offset,
            false,
        )
    }

    /// Exact backward pass for RoPE given upstream gradient `dy` of shape `[batch, num_heads, seq_len, head_dim]`.
    pub fn apply_backward(
        &self,
        dy: &[f32],
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> OchreResult<Vec<f32>> {
        let mut dx = dy.to_vec();
        self.apply_backward_inplace(&mut dx, batch, num_heads, seq_len, offset)?;
        Ok(dx)
    }

    /// Zero-allocation in-place backward pass for RoPE on `[batch, num_heads, seq_len, head_dim]`.
    pub fn apply_backward_inplace(
        &self,
        dy: &mut [f32],
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> OchreResult<()> {
        apply_rope_slice(
            dy,
            &self.cos_table,
            &self.sin_table,
            self.head_dim,
            self.max_seq_len,
            batch,
            num_heads,
            seq_len,
            offset,
            true,
        )
    }

    /// Autograd-compatible RoPE forward on a 4D tensor `[batch, num_heads, seq_len, head_dim]`.
    pub fn forward(&self, x: &Tensor, offset: usize) -> OchreResult<Tensor> {
        if x.shape.len() != 4 {
            return Err(OchreError::InvalidParameter(format!(
                "RoPE forward expects 4D tensor [B, H, S, D], got {:?} ( >_< )",
                x.shape
            )));
        }
        let (batch, num_heads, seq_len, head_dim) =
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        if head_dim != self.head_dim {
            return Err(OchreError::InvalidParameter(format!(
                "RoPE expected head_dim={}, got {head_dim} ( >_< )",
                self.head_dim
            )));
        }

        let mut out_data = x.data.clone();
        self.apply_inplace(&mut out_data, batch, num_heads, seq_len, offset)?;
        out_data = maybe_quantize(out_data, x.dtype);

        let grad_active = AutogradTape::is_grad_enabled() && x.requires_grad;
        let node_id = if grad_active {
            let in_id = x.node_id;
            let cos_tbl = Arc::clone(&self.cos_table);
            let sin_tbl = Arc::clone(&self.sin_table);
            let max_seq_len = self.max_seq_len;
            let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
                if let Some(id) = in_id {
                    let mut dx = d_out.to_vec();
                    apply_rope_slice(
                        &mut dx,
                        &cos_tbl,
                        &sin_tbl,
                        head_dim,
                        max_seq_len,
                        batch,
                        num_heads,
                        seq_len,
                        offset,
                        true,
                    )?;
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
            strides: compute_strides(&x.shape),
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_rope_slice(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    max_seq_len: usize,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    offset: usize,
    inverse: bool,
) -> OchreResult<()> {
    let expected = batch
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(seq_len))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            OchreError::InvalidParameter("RoPE dimensions overflow ( >_< )".to_string())
        })?;
    if data.len() != expected {
        return Err(OchreError::InvalidParameter(format!(
            "RoPE data length {} does not match [{batch}, {num_heads}, {seq_len}, {head_dim}] ({expected}) ( >_< )",
            data.len()
        )));
    }
    if offset.saturating_add(seq_len) > max_seq_len {
        return Err(OchreError::InvalidParameter(format!(
            "RoPE offset {offset} + seq_len {seq_len} exceeds max_seq_len {max_seq_len} ( >_< )"
        )));
    }
    if expected == 0 {
        return Ok(());
    }

    let half = head_dim / 2;
    let total_tokens = batch * num_heads * seq_len;

    let apply_token = |token_idx: usize, token_row: &mut [f32]| {
        let s = token_idx % seq_len;
        let pos = offset + s;
        let cos_row = &cos_table[pos * half..(pos + 1) * half];
        let sin_row = &sin_table[pos * half..(pos + 1) * half];
        let (first_half, second_half) = token_row.split_at_mut(half);
        if inverse {
            for d in 0..half {
                let x0 = first_half[d];
                let x1 = second_half[d];
                let c = cos_row[d];
                let sn = sin_row[d];
                first_half[d] = x0 * c + x1 * sn;
                second_half[d] = -x0 * sn + x1 * c;
            }
        } else {
            for d in 0..half {
                let x0 = first_half[d];
                let x1 = second_half[d];
                let c = cos_row[d];
                let sn = sin_row[d];
                first_half[d] = x0 * c - x1 * sn;
                second_half[d] = x0 * sn + x1 * c;
            }
        }
    };

    if expected >= 8_192 && total_tokens >= 64 {
        data.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(token_idx, token_row)| apply_token(token_idx, token_row));
    } else {
        for (token_idx, token_row) in data.chunks_mut(head_dim).enumerate() {
            apply_token(token_idx, token_row);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_flash_dims(
    q_len_data: usize,
    k_len_data: usize,
    v_len_data: usize,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> OchreResult<()> {
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(OchreError::InvalidParameter(
            "FlashAttention batch, num_heads, num_kv_heads, head_dim must be > 0 ( >_< )"
                .to_string(),
        ));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) ( >_< )"
        )));
    }
    let exp_q = batch * num_heads * q_len * head_dim;
    let exp_kv = batch * num_kv_heads * kv_len * head_dim;
    if q_len_data != exp_q {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention q len {} != expected {exp_q} ( >_< )",
            q_len_data
        )));
    }
    if k_len_data != exp_kv || v_len_data != exp_kv {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention k/v len ({}/{}) != expected {exp_kv} ( >_< )",
            k_len_data, v_len_data
        )));
    }
    Ok(())
}

/// Flash-Style tiled scaled dot-product attention forward (O(S) memory, GQA/MQA, causal).
/// `q`: `[B, H, q_len, D]`, `k`/`v`: `[B, H_kv, kv_len, D]`.
/// Returns `(output [B, H, q_len, D], logsumexp [B, H, q_len])`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>)> {
    validate_flash_dims(
        q.len(),
        k.len(),
        v.len(),
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
    )?;

    flash_attention_forward_strided(
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_len,
        head_dim,
        causal,
        kv_offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_forward_strided(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    kv_stride_seq: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>)> {
    let total_heads = batch * num_heads;
    let q_head_stride = q_len * head_dim;
    let mut out = vec![0.0f32; total_heads * q_head_stride];
    let mut lse = vec![f32::NEG_INFINITY; total_heads * q_len];

    flash_attention_forward_strided_into(
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride_seq,
        head_dim,
        causal,
        kv_offset,
        &mut out,
        Some(&mut lse),
    )?;

    Ok((out, lse))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_forward_strided_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    kv_stride_seq: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
    out: &mut [f32],
    mut lse: Option<&mut [f32]>,
) -> OchreResult<()> {
    let total_heads = batch * num_heads;
    let q_head_stride = q_len * head_dim;
    let kv_head_stride = kv_stride_seq * head_dim;
    let group_size = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if q_len == 0 || kv_len == 0 {
        out.fill(0.0);
        if let Some(lse_buf) = lse.as_deref_mut() {
            lse_buf.fill(f32::NEG_INFINITY);
        }
        return Ok(());
    }

    let tile_gemm = if total_heads >= 2 {
        sgemm_serial
    } else {
        sgemm
    };

    let run_head =
        |bh: usize, out_head: &mut [f32], mut lse_head: Option<&mut [f32]>| -> OchreResult<()> {
            let b = bh / num_heads;
            let h = bh % num_heads;
            let kv_h = h / group_size;
            let q_head = &q[bh * q_head_stride..(bh + 1) * q_head_stride];
            let kv_base = (b * num_kv_heads + kv_h) * kv_head_stride;
            let k_head = &k[kv_base..kv_base + kv_len * head_dim];
            let v_head = &v[kv_base..kv_base + kv_len * head_dim];

            // Fast path for single-token decode (q_len == 1): zero SGEMM packing overhead,
            // 4-way unrolled dot/axpy with online softmax tiled over DECODE_BC keys.
            if q_len == 1 {
                let valid_kv = if causal {
                    (kv_offset + 1).min(kv_len)
                } else {
                    kv_len
                };
                if valid_kv == 0 {
                    out_head.fill(0.0);
                    if let Some(lse_h) = lse_head {
                        lse_h[0] = f32::NEG_INFINITY;
                    }
                    return Ok(());
                }
                out_head.fill(0.0);
                let mut m_i = f32::NEG_INFINITY;
                let mut l_i = 0.0f32;
                let mut scores = [0.0f32; DECODE_BC];
                let mut probs = [0.0f32; DECODE_BC];

                let mut j_start = 0usize;
                while j_start < valid_kv {
                    let bc = (valid_kv - j_start).min(DECODE_BC);
                    let mut row_max = f32::NEG_INFINITY;
                    for (c, score_slot) in scores[..bc].iter_mut().enumerate() {
                        let kj = j_start + c;
                        let k_row = &k_head[kj * head_dim..(kj + 1) * head_dim];
                        let s = dot_f32(q_head, k_row) * scale;
                        *score_slot = s;
                        if s > row_max {
                            row_max = s;
                        }
                    }
                    let m_new = m_i.max(row_max);
                    let alpha = if m_i.is_finite() {
                        (m_i - m_new).exp()
                    } else {
                        0.0
                    };
                    if alpha == 0.0 {
                        out_head.fill(0.0);
                    } else if alpha != 1.0 {
                        for ov in out_head.iter_mut() {
                            *ov *= alpha;
                        }
                    }
                    simd_exp_sub_scalar_into(&scores[..bc], m_new, &mut probs[..bc]);
                    let row_sum: f32 = probs[..bc].iter().copied().sum();
                    for (c, &p) in probs[..bc].iter().enumerate() {
                        if p != 0.0 {
                            let kj = j_start + c;
                            let v_row = &v_head[kj * head_dim..(kj + 1) * head_dim];
                            axpy_f32(out_head, v_row, p);
                        }
                    }
                    l_i = l_i * alpha + row_sum;
                    m_i = m_new;
                    j_start += DECODE_BC;
                }

                if l_i > 0.0 {
                    let inv_l = 1.0f32 / l_i;
                    for ov in out_head.iter_mut() {
                        *ov *= inv_l;
                    }
                    if let Some(lse_h) = lse_head {
                        lse_h[0] = m_i + l_i.ln();
                    }
                } else {
                    out_head.fill(0.0);
                    if let Some(lse_h) = lse_head {
                        lse_h[0] = f32::NEG_INFINITY;
                    }
                }
                return Ok(());
            }

            // GEMM-backed FlashAttention-2 tiled forward with thread-local scratchpad
            // (zero heap allocations inside Rayon tasks, O(Br*Bc + Br*D) memory).
            FLASH_SCRATCH.with(|cell| -> OchreResult<()> {
                let mut scratch = cell.borrow_mut();
                ensure_len(&mut scratch.o_tile, FLASH_BR * head_dim);
                ensure_len(&mut scratch.scores, FLASH_BR * FLASH_BC);
                ensure_len(&mut scratch.probs, FLASH_BR * FLASH_BC);

                let FlashScratch {
                    o_tile,
                    scores,
                    probs,
                    ..
                } = &mut *scratch;

                let mut m_i = [f32::NEG_INFINITY; FLASH_BR];
                let mut l_i = [0.0f32; FLASH_BR];

                let mut i_start = 0usize;
                while i_start < q_len {
                    let br = (q_len - i_start).min(FLASH_BR);
                    m_i[..br].fill(f32::NEG_INFINITY);
                    l_i[..br].fill(0.0);
                    o_tile[..br * head_dim].fill(0.0);

                    let q_tile = &q_head[i_start * head_dim..(i_start + br) * head_dim];
                    let max_allowed_k = if causal {
                        kv_offset + i_start + br - 1
                    } else {
                        usize::MAX
                    };

                    let mut j_start = 0usize;
                    while j_start < kv_len {
                        if causal && j_start > max_allowed_k {
                            break;
                        }
                        let bc = (kv_len - j_start).min(FLASH_BC);
                        let k_tile = &k_head[j_start * head_dim..(j_start + bc) * head_dim];
                        let v_tile = &v_head[j_start * head_dim..(j_start + bc) * head_dim];

                        // S_tile[br x bc] = scale * Q_tile[br x D] @ K_tile[bc x D]^T
                        tile_gemm(
                            br,
                            bc,
                            head_dim,
                            q_tile,
                            k_tile,
                            &mut scores[..br * bc],
                            false,
                            true,
                            scale,
                            0.0,
                        )?;

                        // Apply causal mask only on tiles that cross the diagonal.
                        if causal && j_start + bc > kv_offset + i_start + 1 {
                            for r in 0..br {
                                let qi = i_start + r;
                                let max_valid_c = (kv_offset + qi + 1).saturating_sub(j_start);
                                if max_valid_c < bc {
                                    scores[r * bc + max_valid_c..(r + 1) * bc]
                                        .fill(f32::NEG_INFINITY);
                                }
                            }
                        }

                        // Online softmax update per query row in tile
                        for r in 0..br {
                            let score_row = &scores[r * bc..(r + 1) * bc];
                            let mut row_max = f32::NEG_INFINITY;
                            for &s in score_row {
                                if s > row_max {
                                    row_max = s;
                                }
                            }
                            let prob_row = &mut probs[r * bc..(r + 1) * bc];
                            if row_max == f32::NEG_INFINITY {
                                prob_row.fill(0.0);
                                continue;
                            }

                            let m_prev = m_i[r];
                            let m_new = m_prev.max(row_max);
                            let alpha = if m_prev.is_finite() {
                                (m_prev - m_new).exp()
                            } else {
                                0.0
                            };

                            simd_exp_sub_scalar_into(score_row, m_new, prob_row);
                            let row_sum: f32 = prob_row.iter().copied().sum();

                            let o_row = &mut o_tile[r * head_dim..(r + 1) * head_dim];
                            if alpha == 0.0 {
                                o_row.fill(0.0);
                            } else if alpha != 1.0 {
                                for ov in o_row.iter_mut() {
                                    *ov *= alpha;
                                }
                            }

                            l_i[r] = l_i[r] * alpha + row_sum;
                            m_i[r] = m_new;
                        }

                        // O_tile[br x D] += P_tile[br x bc] @ V_tile[bc x D]
                        tile_gemm(
                            br,
                            head_dim,
                            bc,
                            &probs[..br * bc],
                            v_tile,
                            &mut o_tile[..br * head_dim],
                            false,
                            false,
                            1.0,
                            1.0,
                        )?;

                        j_start += FLASH_BC;
                    }

                    // Normalize output tile and store logsumexp L_i
                    for r in 0..br {
                        let qi = i_start + r;
                        let dst_row = &mut out_head[qi * head_dim..(qi + 1) * head_dim];
                        let li = l_i[r];
                        if li > 0.0 {
                            let inv_l = 1.0f32 / li;
                            let src_row = &o_tile[r * head_dim..(r + 1) * head_dim];
                            for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                                *dst = src * inv_l;
                            }
                            if let Some(ref mut lse_h) = lse_head {
                                lse_h[qi] = m_i[r] + li.ln();
                            }
                        } else {
                            dst_row.fill(0.0);
                            if let Some(ref mut lse_h) = lse_head {
                                lse_h[qi] = f32::NEG_INFINITY;
                            }
                        }
                    }

                    i_start += FLASH_BR;
                }
                Ok(())
            })
        };

    if total_heads >= 2 {
        match lse {
            Some(lse_buf) => out
                .par_chunks_mut(q_head_stride)
                .zip(lse_buf.par_chunks_mut(q_len))
                .enumerate()
                .try_for_each(|(bh, (out_head, lse_head))| {
                    run_head(bh, out_head, Some(lse_head))
                })?,
            None => out
                .par_chunks_mut(q_head_stride)
                .enumerate()
                .try_for_each(|(bh, out_head)| run_head(bh, out_head, None))?,
        }
    } else {
        run_head(0, out, lse)?;
    }

    Ok(())
}

/// Device-dispatched FlashAttention forward (uses CUDA `flash_sdpa_fwd` on GPU with CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_forward_on_device(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
    device: &DeviceType,
) -> OchreResult<(Vec<f32>, Vec<f32>)> {
    validate_flash_dims(
        q.len(),
        k.len(),
        v.len(),
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
    )?;

    #[cfg(feature = "cuda-dynamic")]
    if should_dispatch_cuda_flash(batch, num_heads, q_len, kv_len)
        && let Some(gpu_id) = cuda_gpu_id(device)
        && let Some(Ok(res)) = crate::backends::cuda_nn::cuda_flash_attention_fwd(
            gpu_id,
            q,
            k,
            v,
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
        )
    {
        return Ok(res);
    }
    let _ = device;

    flash_attention_forward(
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
        causal,
        kv_offset,
    )
}

/// Device-dispatched FlashAttention backward (uses CUDA `flash_sdpa_bwd` on GPU with CPU fallback).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn flash_attention_backward_on_device(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    lse: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
    device: &DeviceType,
) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    #[cfg(feature = "cuda-dynamic")]
    if should_dispatch_cuda_flash(batch, num_heads, q_len, kv_len)
        && let Some(gpu_id) = cuda_gpu_id(device)
        && d_out.len() == q.len()
        && out.len() == q.len()
        && lse.len() == batch * num_heads * q_len
        && let Some(Ok(grads)) = crate::backends::cuda_nn::cuda_flash_attention_bwd(
            gpu_id,
            d_out,
            q,
            k,
            v,
            out,
            lse,
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
        )
    {
        return Ok(grads);
    }
    let _ = device;

    flash_attention_backward(
        d_out,
        q,
        k,
        v,
        out,
        lse,
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
        causal,
        kv_offset,
    )
}

/// FlashAttention-2 style tiled backward pass recomputing attention weights from `lse` in O(S) memory.
/// Returns `(dq [B, H, q_len, D], dk [B, H_kv, kv_len, D], dv [B, H_kv, kv_len, D])`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    lse: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    flash_attention_backward_into(
        d_out,
        q,
        k,
        v,
        out,
        lse,
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
        causal,
        kv_offset,
        &mut dq,
        &mut dk,
        &mut dv,
    )?;
    Ok((dq, dk, dv))
}

/// Zero-allocation FlashAttention-2 tiled backward pass writing into pre-allocated `dq`, `dk`, `dv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_backward_into(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    lse: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) -> OchreResult<()> {
    validate_flash_dims(
        q.len(),
        k.len(),
        v.len(),
        batch,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
    )?;
    if d_out.len() != q.len() || out.len() != q.len() {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention backward d_out/out len ({}/{}) != q len ({}) ( >_< )",
            d_out.len(),
            out.len(),
            q.len()
        )));
    }
    if dq.len() != q.len() || dk.len() != k.len() || dv.len() != v.len() {
        return Err(OchreError::InvalidParameter(
            "FlashAttention backward output gradient buffer size mismatch ( >_< )".to_string(),
        ));
    }
    let expected_lse = batch * num_heads * q_len;
    if lse.len() != expected_lse {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention backward lse len {} != expected {expected_lse} ( >_< )",
            lse.len()
        )));
    }

    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    if q_len == 0 || kv_len == 0 {
        return Ok(());
    }

    let group_size = num_heads / num_kv_heads;
    let q_head_stride = q_len * head_dim;
    let kv_head_stride = kv_len * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let use_serial = (group_size == 1 && batch * num_heads >= 2) || (batch * num_kv_heads >= 2);
    let tile_gemm = if use_serial { sgemm_serial } else { sgemm };

    let backward_one_head = |bh: usize,
                             dq_head: &mut [f32],
                             dk_head: &mut [f32],
                             dv_head: &mut [f32]|
     -> OchreResult<()> {
        let b = bh / num_heads;
        let h = bh % num_heads;
        let kv_h = h / group_size;

        let q_head = &q[bh * q_head_stride..(bh + 1) * q_head_stride];
        let do_head = &d_out[bh * q_head_stride..(bh + 1) * q_head_stride];
        let out_head = &out[bh * q_head_stride..(bh + 1) * q_head_stride];
        let lse_head = &lse[bh * q_len..(bh + 1) * q_len];

        let kv_off = (b * num_kv_heads + kv_h) * kv_head_stride;
        let k_head = &k[kv_off..kv_off + kv_head_stride];
        let v_head = &v[kv_off..kv_off + kv_head_stride];

        FLASH_SCRATCH.with(|cell| -> OchreResult<()> {
            let mut scratch = cell.borrow_mut();
            ensure_len(&mut scratch.d_row, q_len);
            ensure_len(&mut scratch.scores, FLASH_BR * FLASH_BC);
            ensure_len(&mut scratch.dp_tile, FLASH_BR * FLASH_BC);
            ensure_len(&mut scratch.probs, FLASH_BR * FLASH_BC);
            ensure_len(&mut scratch.ds_tile, FLASH_BR * FLASH_BC);

            let FlashScratch {
                scores,
                dp_tile,
                probs,
                ds_tile,
                d_row,
                ..
            } = &mut *scratch;

            // D_i = sum_d dO[i, d] * O[i, d] (O(S) per head)
            for (i, d_slot) in d_row[..q_len].iter_mut().enumerate() {
                let o_r = &out_head[i * head_dim..(i + 1) * head_dim];
                let do_r = &do_head[i * head_dim..(i + 1) * head_dim];
                *d_slot = dot_f32(o_r, do_r);
            }

            let mut j_start = 0usize;
            while j_start < kv_len {
                let bc = (kv_len - j_start).min(FLASH_BC);
                let k_tile = &k_head[j_start * head_dim..(j_start + bc) * head_dim];
                let v_tile = &v_head[j_start * head_dim..(j_start + bc) * head_dim];
                let dk_tile = &mut dk_head[j_start * head_dim..(j_start + bc) * head_dim];
                let dv_tile = &mut dv_head[j_start * head_dim..(j_start + bc) * head_dim];

                let mut i_start = if causal {
                    j_start.saturating_sub(kv_offset)
                } else {
                    0
                };
                while i_start < q_len {
                    let br = (q_len - i_start).min(FLASH_BR);
                    let q_tile = &q_head[i_start * head_dim..(i_start + br) * head_dim];
                    let do_tile = &do_head[i_start * head_dim..(i_start + br) * head_dim];
                    let dq_tile = &mut dq_head[i_start * head_dim..(i_start + br) * head_dim];

                    // S_tile[br x bc] = scale * Q_tile[br x D] @ K_tile[bc x D]^T
                    tile_gemm(
                        br,
                        bc,
                        head_dim,
                        q_tile,
                        k_tile,
                        &mut scores[..br * bc],
                        false,
                        true,
                        scale,
                        0.0,
                    )?;

                    // dP_tile[br x bc] = dO_tile[br x D] @ V_tile[bc x D]^T
                    tile_gemm(
                        br,
                        bc,
                        head_dim,
                        do_tile,
                        v_tile,
                        &mut dp_tile[..br * bc],
                        false,
                        true,
                        1.0,
                        0.0,
                    )?;

                    // Elementwise P = exp(S - lse) and dS_scaled = P * (dP - D) * scale
                    let check_causal = causal && j_start + bc > kv_offset + i_start + 1;
                    for r in 0..br {
                        let qi = i_start + r;
                        let lse_i = lse_head[qi];
                        let di = d_row[qi];
                        let row_off = r * bc;
                        let p_row = &mut probs[row_off..row_off + bc];
                        let ds_row = &mut ds_tile[row_off..row_off + bc];
                        if !lse_i.is_finite() {
                            p_row.fill(0.0);
                            ds_row.fill(0.0);
                            continue;
                        }
                        let valid_c = if check_causal {
                            (kv_offset + qi + 1).saturating_sub(j_start).min(bc)
                        } else {
                            bc
                        };
                        let s_row = &scores[row_off..row_off + valid_c];
                        let dp_row = &dp_tile[row_off..row_off + valid_c];
                        simd_exp_sub_scalar_into(s_row, lse_i, &mut p_row[..valid_c]);
                        for c in 0..valid_c {
                            ds_row[c] = p_row[c] * (dp_row[c] - di) * scale;
                        }
                        if valid_c < bc {
                            p_row[valid_c..].fill(0.0);
                            ds_row[valid_c..].fill(0.0);
                        }
                    }

                    // dV_tile[bc x D] += P_tile[br x bc]^T @ dO_tile[br x D]
                    tile_gemm(
                        bc,
                        head_dim,
                        br,
                        &probs[..br * bc],
                        do_tile,
                        dv_tile,
                        true,
                        false,
                        1.0,
                        1.0,
                    )?;

                    // dK_tile[bc x D] += dS_scaled[br x bc]^T @ Q_tile[br x D]
                    tile_gemm(
                        bc,
                        head_dim,
                        br,
                        &ds_tile[..br * bc],
                        q_tile,
                        dk_tile,
                        true,
                        false,
                        1.0,
                        1.0,
                    )?;

                    // dQ_tile[br x D] += dS_scaled[br x bc] @ K_tile[bc x D]
                    tile_gemm(
                        br,
                        head_dim,
                        bc,
                        &ds_tile[..br * bc],
                        k_tile,
                        dq_tile,
                        false,
                        false,
                        1.0,
                        1.0,
                    )?;

                    i_start += FLASH_BR;
                }
                j_start += FLASH_BC;
            }
            Ok(())
        })
    };

    if group_size == 1 && batch * num_heads >= 2 {
        dq.par_chunks_mut(q_head_stride)
            .zip(dk.par_chunks_mut(kv_head_stride))
            .zip(dv.par_chunks_mut(kv_head_stride))
            .enumerate()
            .try_for_each(|(bh, ((dq_h, dk_h), dv_h))| backward_one_head(bh, dq_h, dk_h, dv_h))?;
    } else if batch * num_kv_heads >= 2 {
        let q_group_stride = group_size * q_head_stride;
        dq.par_chunks_mut(q_group_stride)
            .zip(dk.par_chunks_mut(kv_head_stride))
            .zip(dv.par_chunks_mut(kv_head_stride))
            .enumerate()
            .try_for_each(|(b_kv, ((dq_grp, dk_h), dv_h))| -> OchreResult<()> {
                let b = b_kv / num_kv_heads;
                let kv_h = b_kv % num_kv_heads;
                for g in 0..group_size {
                    let h = kv_h * group_size + g;
                    let bh = b * num_heads + h;
                    let dq_h = &mut dq_grp[g * q_head_stride..(g + 1) * q_head_stride];
                    backward_one_head(bh, dq_h, dk_h, dv_h)?;
                }
                Ok(())
            })?;
    } else {
        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let kv_off = (b * num_kv_heads + kv_h) * kv_head_stride;
                let dk_h = &mut dk[kv_off..kv_off + kv_head_stride];
                let dv_h = &mut dv[kv_off..kv_off + kv_head_stride];
                for g in 0..group_size {
                    let h = kv_h * group_size + g;
                    let bh = b * num_heads + h;
                    let dq_h = &mut dq[bh * q_head_stride..(bh + 1) * q_head_stride];
                    backward_one_head(bh, dq_h, dk_h, dv_h)?;
                }
            }
        }
    }

    Ok(())
}

/// Flash-Style Scaled Dot-Product Attention layer with `Tensor` autograd.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScaledDotProductAttention;

impl ScaledDotProductAttention {
    /// Creates a `ScaledDotProductAttention` module.
    pub fn new() -> Self {
        Self
    }

    /// Computes tiled FlashAttention over `q [B, H, q_len, D]`, `k [B, H_kv, kv_len, D]`,
    /// and `v [B, H_kv, kv_len, D]`.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        kv_offset: usize,
    ) -> OchreResult<Tensor> {
        if q.shape.len() != 4 || k.shape.len() != 4 || v.shape.len() != 4 {
            return Err(OchreError::InvalidParameter(format!(
                "ScaledDotProductAttention expects 4D [B, H, S, D] tensors, got q={:?}, k={:?}, v={:?} ( >_< )",
                q.shape, k.shape, v.shape
            )));
        }
        let (batch, num_heads, q_len, head_dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let (kb, num_kv_heads, kv_len, kd) = (k.shape[0], k.shape[1], k.shape[2], k.shape[3]);
        if batch != kb || head_dim != kd || v.shape != [batch, num_kv_heads, kv_len, head_dim] {
            return Err(OchreError::InvalidParameter(format!(
                "ScaledDotProductAttention incompatible shapes q={:?}, k={:?}, v={:?} ( >_< )",
                q.shape, k.shape, v.shape
            )));
        }

        let out_shape = vec![batch, num_heads, q_len, head_dim];
        let q_id = if q.requires_grad { q.node_id } else { None };
        let k_id = if k.requires_grad { k.node_id } else { None };
        let v_id = if v.requires_grad { v.node_id } else { None };
        let grad_active =
            AutogradTape::is_grad_enabled() && (q_id.is_some() || k_id.is_some() || v_id.is_some());

        #[cfg(feature = "cuda-dynamic")]
        if grad_active
            && should_dispatch_cuda_flash(batch, num_heads, q_len, kv_len)
            && let Some(gpu_id) = cuda_gpu_id(&q.device)
            && let Some(cuda_res) = crate::backends::cuda_nn::cuda_flash_attention_fwd_cached(
                gpu_id,
                &q.data,
                &k.data,
                &v.data,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                causal,
                kv_offset,
            )
        {
            let (raw_out, cache) = cuda_res?;
            let out_data = maybe_quantize(raw_out, q.dtype);
            let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
                let (dq, dk, dv) = crate::backends::cuda_nn::cuda_flash_attention_bwd_cached(
                    gpu_id, &cache, d_out,
                )
                .ok_or_else(|| {
                    OchreError::DeviceError(
                        "CUDA FlashAttention cached backward session unavailable ( >_< )"
                            .to_string(),
                    )
                })??;
                let mut pairs = Vec::with_capacity(3);
                if let Some(id) = q_id {
                    pairs.push((id, dq));
                }
                if let Some(id) = k_id {
                    pairs.push((id, dk));
                }
                if let Some(id) = v_id {
                    pairs.push((id, dv));
                }
                Ok(pairs)
            }));
            let id = AutogradTape::record(Op::Custom {
                inputs: vec![q_id, k_id, v_id],
                backward: bwd,
            });
            return Ok(Tensor {
                strides: compute_strides(&out_shape),
                shape: out_shape,
                data: out_data,
                dtype: q.dtype,
                device: q.device.clone(),
                requires_grad: true,
                node_id: Some(id),
            });
        }

        let (raw_out, lse) = flash_attention_forward_on_device(
            &q.data,
            &k.data,
            &v.data,
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
            &q.device,
        )?;

        let (out_data, node_id) = if grad_active {
            let out_data = maybe_quantize(raw_out.clone(), q.dtype);
            let q_saved = q.data.clone();
            let k_saved = k.data.clone();
            let v_saved = v.data.clone();
            let dev_saved = q.device.clone();
            let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
                let (dq, dk, dv) = flash_attention_backward_on_device(
                    d_out,
                    &q_saved,
                    &k_saved,
                    &v_saved,
                    &raw_out,
                    &lse,
                    batch,
                    num_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    causal,
                    kv_offset,
                    &dev_saved,
                )?;
                let mut pairs = Vec::with_capacity(3);
                if let Some(id) = q_id {
                    pairs.push((id, dq));
                }
                if let Some(id) = k_id {
                    pairs.push((id, dk));
                }
                if let Some(id) = v_id {
                    pairs.push((id, dv));
                }
                Ok(pairs)
            }));
            let id = AutogradTape::record(Op::Custom {
                inputs: vec![q_id, k_id, v_id],
                backward: bwd,
            });
            (out_data, Some(id))
        } else {
            (maybe_quantize(raw_out, q.dtype), None)
        };

        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: q.dtype,
            device: q.device.clone(),
            requires_grad: grad_active,
            node_id,
        })
    }
}

/// Pre-allocated Key-Value Cache `[batch, num_kv_heads, max_seq_len, head_dim]` for O(1)-alloc decoding.
#[derive(Clone, Debug)]
pub struct KVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub batch: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub cur_len: usize,
}

impl KVCache {
    /// Allocates a zeroed `KVCache` of capacity `[batch, num_kv_heads, max_seq_len, head_dim]`.
    pub fn new(
        batch: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> OchreResult<Self> {
        if batch == 0 || num_kv_heads == 0 || max_seq_len == 0 || head_dim == 0 {
            return Err(OchreError::InvalidParameter(
                "KVCache dimensions must all be > 0 ( >_< )".to_string(),
            ));
        }
        let total = batch
            .checked_mul(num_kv_heads)
            .and_then(|v| v.checked_mul(max_seq_len))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| {
                OchreError::InvalidParameter("KVCache capacity overflow ( >_< )".to_string())
            })?;
        Ok(Self {
            k: vec![0.0f32; total],
            v: vec![0.0f32; total],
            batch,
            num_kv_heads,
            max_seq_len,
            head_dim,
            cur_len: 0,
        })
    }

    /// Appends `k_new` and `v_new` of shape `[batch, num_kv_heads, seq_len, head_dim]` into the cache.
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32], seq_len: usize) -> OchreResult<()> {
        let expected = self.batch * self.num_kv_heads * seq_len * self.head_dim;
        if k_new.len() != expected || v_new.len() != expected {
            return Err(OchreError::InvalidParameter(format!(
                "KVCache::append expected {expected} elements, got k={} v={} ( >_< )",
                k_new.len(),
                v_new.len()
            )));
        }
        if self.cur_len.saturating_add(seq_len) > self.max_seq_len {
            return Err(OchreError::InvalidParameter(format!(
                "KVCache overflow: cur_len={} + seq_len={} > max_seq_len={} ( >_< )",
                self.cur_len, seq_len, self.max_seq_len
            )));
        }
        if seq_len == 0 {
            return Ok(());
        }

        let src_head_stride = seq_len * self.head_dim;
        let dst_head_stride = self.max_seq_len * self.head_dim;
        let dst_pos_offset = self.cur_len * self.head_dim;

        for bh in 0..(self.batch * self.num_kv_heads) {
            let src_off = bh * src_head_stride;
            let dst_off = bh * dst_head_stride + dst_pos_offset;
            self.k[dst_off..dst_off + src_head_stride]
                .copy_from_slice(&k_new[src_off..src_off + src_head_stride]);
            self.v[dst_off..dst_off + src_head_stride]
                .copy_from_slice(&v_new[src_off..src_off + src_head_stride]);
        }

        self.cur_len += seq_len;
        Ok(())
    }

    /// Resets cached sequence length to 0 without deallocating backing storage.
    pub fn reset(&mut self) {
        self.cur_len = 0;
    }

    /// Returns contiguous `[batch, num_kv_heads, cur_len, head_dim]` copies of cached `k` and `v`.
    pub fn slice_kv(&self) -> (Vec<f32>, Vec<f32>) {
        if self.cur_len == 0 {
            return (Vec::new(), Vec::new());
        }
        let total = self.batch * self.num_kv_heads * self.cur_len * self.head_dim;
        let mut k_out = vec![0.0f32; total];
        let mut v_out = vec![0.0f32; total];
        let src_head_stride = self.max_seq_len * self.head_dim;
        let dst_head_stride = self.cur_len * self.head_dim;
        for bh in 0..(self.batch * self.num_kv_heads) {
            let src_off = bh * src_head_stride;
            let dst_off = bh * dst_head_stride;
            k_out[dst_off..dst_off + dst_head_stride]
                .copy_from_slice(&self.k[src_off..src_off + dst_head_stride]);
            v_out[dst_off..dst_off + dst_head_stride]
                .copy_from_slice(&self.v[src_off..src_off + dst_head_stride]);
        }
        (k_out, v_out)
    }
}

/// Multi-Head / Grouped-Query / Multi-Query Attention layer with optional RoPE and KVCache.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rope: Option<RoPE>,
    pub causal: bool,
    pub dtype: DType,
    pub device: DeviceType,
}

impl MultiHeadAttention {
    /// Creates a `MultiHeadAttention` layer (supports MHA when `num_kv_heads == num_heads`,
    /// GQA when `num_heads % num_kv_heads == 0`, and MQA when `num_kv_heads == 1`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        use_bias: bool,
        causal: bool,
        rope: Option<RoPE>,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if d_model == 0 || num_heads == 0 || num_kv_heads == 0 {
            return Err(OchreError::InvalidParameter(
                "MultiHeadAttention d_model, num_heads, num_kv_heads must be > 0 ( >_< )"
                    .to_string(),
            ));
        }
        if !d_model.is_multiple_of(num_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "d_model ({d_model}) must be divisible by num_heads ({num_heads}) ( >_< )"
            )));
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) ( >_< )"
            )));
        }
        let head_dim = d_model / num_heads;
        if let Some(ref r) = rope
            && r.head_dim != head_dim
        {
            return Err(OchreError::InvalidParameter(format!(
                "RoPE head_dim ({}) does not match attention head_dim ({head_dim}) ( >_< )",
                r.head_dim
            )));
        }

        let kv_dim = num_kv_heads * head_dim;
        let q_proj = Linear::new(d_model, d_model, use_bias, dtype, device.clone(), seed)?;
        let k_proj = Linear::new(
            d_model,
            kv_dim,
            use_bias,
            dtype,
            device.clone(),
            seed.wrapping_add(17),
        )?;
        let v_proj = Linear::new(
            d_model,
            kv_dim,
            use_bias,
            dtype,
            device.clone(),
            seed.wrapping_add(31),
        )?;
        let o_proj = Linear::new(
            d_model,
            d_model,
            use_bias,
            dtype,
            device.clone(),
            seed.wrapping_add(47),
        )?;

        Ok(Self {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            causal,
            dtype,
            device,
        })
    }

    /// Autograd-tracked forward pass for self-attention or cross-attention.
    /// Accepts `[batch, seq_len, d_model]` (or 2D `[seq_len, d_model]` treated as `batch = 1`).
    pub fn forward(
        &self,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        kv_cache: Option<&mut KVCache>,
    ) -> OchreResult<Tensor> {
        let was_2d = q_in.shape.len() == 2;
        let (batch, q_len, q_dim) = extract_bsd(q_in)?;
        let (kb, kv_len, k_dim) = extract_bsd(k_in)?;
        let (vb, v_len, v_dim) = extract_bsd(v_in)?;
        if batch != kb || batch != vb || kv_len != v_len {
            return Err(OchreError::InvalidParameter(format!(
                "MHA batch/kv_len mismatch: q={:?}, k={:?}, v={:?} ( >_< )",
                q_in.shape, k_in.shape, v_in.shape
            )));
        }
        if q_dim != self.d_model || k_dim != self.d_model || v_dim != self.d_model {
            return Err(OchreError::InvalidParameter(format!(
                "MHA expected d_model={}, got ({q_dim}, {k_dim}, {v_dim}) ( >_< )",
                self.d_model
            )));
        }

        let q_proj_out = self.q_proj.forward(q_in)?;
        let k_proj_out = self.k_proj.forward(k_in)?;
        let v_proj_out = self.v_proj.forward(v_in)?;

        let mut q_heads =
            permute_bshd_to_bhsd(&q_proj_out, batch, q_len, self.num_heads, self.head_dim)?;
        let mut k_heads =
            permute_bshd_to_bhsd(&k_proj_out, batch, kv_len, self.num_kv_heads, self.head_dim)?;
        let v_heads =
            permute_bshd_to_bhsd(&v_proj_out, batch, kv_len, self.num_kv_heads, self.head_dim)?;

        let rope_offset = kv_cache.as_ref().map_or(0, |c| c.cur_len);
        if let Some(ref rope) = self.rope {
            q_heads = rope.forward(&q_heads, rope_offset)?;
            k_heads = rope.forward(&k_heads, rope_offset)?;
        }

        let attn = if let Some(cache) = kv_cache {
            cache.append(&k_heads.data, &v_heads.data, kv_len)?;
            let total_kv_len = cache.cur_len;
            let attn_kv_offset = total_kv_len.saturating_sub(q_len);
            if !AutogradTape::is_grad_enabled() || !q_heads.requires_grad {
                // Direct strided view over pre-allocated KVCache without copying K/V history.
                let mut raw_out = vec![0.0f32; batch * self.num_heads * q_len * self.head_dim];
                flash_attention_forward_strided_into(
                    &q_heads.data,
                    &cache.k,
                    &cache.v,
                    batch,
                    self.num_heads,
                    self.num_kv_heads,
                    q_len,
                    total_kv_len,
                    cache.max_seq_len,
                    self.head_dim,
                    self.causal,
                    attn_kv_offset,
                    &mut raw_out,
                    None,
                )?;
                let out_data = maybe_quantize(raw_out, q_heads.dtype);
                let out_shape = vec![batch, self.num_heads, q_len, self.head_dim];
                Tensor {
                    strides: compute_strides(&out_shape),
                    shape: out_shape,
                    data: out_data,
                    dtype: q_heads.dtype,
                    device: q_heads.device.clone(),
                    requires_grad: false,
                    node_id: None,
                }
            } else {
                let (k_all, v_all) = cache.slice_kv();
                let k_t = Tensor::new(
                    k_all,
                    vec![batch, self.num_kv_heads, total_kv_len, self.head_dim],
                    k_heads.dtype,
                    k_heads.device.clone(),
                    false,
                )?;
                let v_t = Tensor::new(
                    v_all,
                    vec![batch, self.num_kv_heads, total_kv_len, self.head_dim],
                    v_heads.dtype,
                    v_heads.device.clone(),
                    false,
                )?;
                ScaledDotProductAttention::new().forward(
                    &q_heads,
                    &k_t,
                    &v_t,
                    self.causal,
                    attn_kv_offset,
                )?
            }
        } else {
            ScaledDotProductAttention::new().forward(
                &q_heads,
                &k_heads,
                &v_heads,
                self.causal,
                0,
            )?
        };

        let merged = permute_bhsd_to_bshd(&attn, batch, self.num_heads, q_len, self.head_dim)?;
        let merged = if was_2d {
            merged.reshape(vec![q_len, self.d_model])?
        } else {
            merged
        };
        self.o_proj.forward(&merged)
    }

    /// Convenience self-attention forward pass (`q = k = v = x`).
    pub fn forward_self(&self, x: &Tensor, kv_cache: Option<&mut KVCache>) -> OchreResult<Tensor> {
        self.forward(x, x, x, kv_cache)
    }

    /// Fast raw forward pass without autograd tape recording.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_raw(
        &self,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        batch: usize,
        q_len: usize,
        kv_len: usize,
        kv_cache: Option<&mut KVCache>,
        out: &mut [f32],
    ) -> OchreResult<()> {
        let q_rows = batch * q_len;
        let kv_rows = batch * kv_len;
        let kv_dim = self.num_kv_heads * self.head_dim;

        if q_in.len() != q_rows * self.d_model
            || k_in.len() != kv_rows * self.d_model
            || v_in.len() != kv_rows * self.d_model
            || out.len() != q_rows * self.d_model
        {
            return Err(OchreError::InvalidParameter(
                "MultiHeadAttention::forward_raw buffer size mismatch ( >_< )".to_string(),
            ));
        }

        let q_len_elems = q_rows * self.d_model;
        let kv_len_elems = kv_rows * kv_dim;
        let q_no_transpose = q_len == 1 || self.num_heads == 1;
        let kv_no_transpose = kv_len == 1 || self.num_kv_heads == 1;

        MHA_RAW_SCRATCH.with(|cell| -> OchreResult<()> {
            let mut scratch = cell.borrow_mut();
            ensure_len(&mut scratch.q_proj, q_len_elems);
            ensure_len(&mut scratch.k_proj, kv_len_elems);
            ensure_len(&mut scratch.v_proj, kv_len_elems);
            ensure_len(&mut scratch.attn_out, q_len_elems);
            if !q_no_transpose {
                ensure_len(&mut scratch.q_heads, q_len_elems);
                ensure_len(&mut scratch.merged, q_len_elems);
            }
            if !kv_no_transpose {
                ensure_len(&mut scratch.k_heads, kv_len_elems);
                ensure_len(&mut scratch.v_heads, kv_len_elems);
            }

            let MhaRawScratch {
                q_proj,
                k_proj,
                v_proj,
                q_heads,
                k_heads,
                v_heads,
                attn_out,
                merged,
            } = &mut *scratch;

            self.q_proj
                .forward_raw(q_in, q_rows, &mut q_proj[..q_len_elems])?;
            self.k_proj
                .forward_raw(k_in, kv_rows, &mut k_proj[..kv_len_elems])?;
            self.v_proj
                .forward_raw(v_in, kv_rows, &mut v_proj[..kv_len_elems])?;

            let q_slice: &mut [f32] = if q_no_transpose {
                &mut q_proj[..q_len_elems]
            } else {
                transpose_bshd_to_bhsd_into(
                    &q_proj[..q_len_elems],
                    &mut q_heads[..q_len_elems],
                    batch,
                    q_len,
                    self.num_heads,
                    self.head_dim,
                );
                &mut q_heads[..q_len_elems]
            };

            let (k_slice, v_slice): (&mut [f32], &[f32]) = if kv_no_transpose {
                (&mut k_proj[..kv_len_elems], &v_proj[..kv_len_elems])
            } else {
                transpose_bshd_to_bhsd_into(
                    &k_proj[..kv_len_elems],
                    &mut k_heads[..kv_len_elems],
                    batch,
                    kv_len,
                    self.num_kv_heads,
                    self.head_dim,
                );
                transpose_bshd_to_bhsd_into(
                    &v_proj[..kv_len_elems],
                    &mut v_heads[..kv_len_elems],
                    batch,
                    kv_len,
                    self.num_kv_heads,
                    self.head_dim,
                );
                (&mut k_heads[..kv_len_elems], &v_heads[..kv_len_elems])
            };

            let rope_offset = kv_cache.as_ref().map_or(0, |c| c.cur_len);
            if let Some(ref rope) = self.rope {
                rope.apply_inplace(q_slice, batch, self.num_heads, q_len, rope_offset)?;
                rope.apply_inplace(k_slice, batch, self.num_kv_heads, kv_len, rope_offset)?;
            }

            if let Some(cache) = kv_cache {
                cache.append(k_slice, v_slice, kv_len)?;
                let total_kv = cache.cur_len;
                let attn_offset = total_kv.saturating_sub(q_len);
                flash_attention_forward_strided_into(
                    q_slice,
                    &cache.k,
                    &cache.v,
                    batch,
                    self.num_heads,
                    self.num_kv_heads,
                    q_len,
                    total_kv,
                    cache.max_seq_len,
                    self.head_dim,
                    self.causal,
                    attn_offset,
                    &mut attn_out[..q_len_elems],
                    None,
                )?;
            } else {
                flash_attention_forward_strided_into(
                    q_slice,
                    k_slice,
                    v_slice,
                    batch,
                    self.num_heads,
                    self.num_kv_heads,
                    q_len,
                    kv_len,
                    kv_len,
                    self.head_dim,
                    self.causal,
                    0,
                    &mut attn_out[..q_len_elems],
                    None,
                )?;
            }

            let o_in: &[f32] = if q_no_transpose {
                &attn_out[..q_len_elems]
            } else {
                transpose_bhsd_to_bshd_into(
                    &attn_out[..q_len_elems],
                    &mut merged[..q_len_elems],
                    batch,
                    self.num_heads,
                    q_len,
                    self.head_dim,
                );
                &merged[..q_len_elems]
            };
            self.o_proj.forward_raw(o_in, q_rows, out)
        })
    }

    /// Returns references to all trainable parameters in `q, k, v, o` projections.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::with_capacity(8);
        for lin in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
            params.push(&lin.weight);
            if let Some(ref b) = lin.bias {
                params.push(b);
            }
        }
        params
    }

    /// Returns mutable references to all trainable parameters in `q, k, v, o` projections.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::with_capacity(8);
        for lin in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            params.push(&mut lin.weight);
            if let Some(ref mut b) = lin.bias {
                params.push(b);
            }
        }
        params
    }
}

fn extract_bsd(t: &Tensor) -> OchreResult<(usize, usize, usize)> {
    match t.shape.as_slice() {
        [s, d] => Ok((1, *s, *d)),
        [b, s, d] => Ok((*b, *s, *d)),
        _ => Err(OchreError::InvalidParameter(format!(
            "MultiHeadAttention expects [seq, d_model] or [batch, seq, d_model], got {:?} ( >_< )",
            t.shape
        ))),
    }
}

pub(crate) fn transpose_bshd_to_bhsd_into(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) {
    if seq_len <= 1 || heads <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    let head_stride = seq_len * head_dim;
    let total_heads = batch * heads;
    if src.len() >= 16_384 && total_heads >= 2 {
        dst.par_chunks_mut(head_stride)
            .enumerate()
            .for_each(|(bh, dst_head)| {
                let b = bh / heads;
                let h = bh % heads;
                for s in 0..seq_len {
                    let s_off = ((b * seq_len + s) * heads + h) * head_dim;
                    let d_off = s * head_dim;
                    dst_head[d_off..d_off + head_dim]
                        .copy_from_slice(&src[s_off..s_off + head_dim]);
                }
            });
    } else {
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..heads {
                    let s_off = ((b * seq_len + s) * heads + h) * head_dim;
                    let d_off = ((b * heads + h) * seq_len + s) * head_dim;
                    dst[d_off..d_off + head_dim].copy_from_slice(&src[s_off..s_off + head_dim]);
                }
            }
        }
    }
}

pub(crate) fn transpose_bhsd_to_bshd_into(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    if seq_len <= 1 || heads <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    let token_stride = heads * head_dim;
    let total_tokens = batch * seq_len;
    if src.len() >= 16_384 && total_tokens >= 2 {
        dst.par_chunks_mut(token_stride)
            .enumerate()
            .for_each(|(bs, dst_tok)| {
                let b = bs / seq_len;
                let s = bs % seq_len;
                for h in 0..heads {
                    let s_off = ((b * heads + h) * seq_len + s) * head_dim;
                    let d_off = h * head_dim;
                    dst_tok[d_off..d_off + head_dim].copy_from_slice(&src[s_off..s_off + head_dim]);
                }
            });
    } else {
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq_len {
                    let s_off = ((b * heads + h) * seq_len + s) * head_dim;
                    let d_off = ((b * seq_len + s) * heads + h) * head_dim;
                    dst[d_off..d_off + head_dim].copy_from_slice(&src[s_off..s_off + head_dim]);
                }
            }
        }
    }
}

pub(crate) fn transpose_bshd_to_bhsd_raw(
    src: &[f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    if seq_len <= 1 || heads <= 1 {
        return src.to_vec();
    }
    let mut dst = vec![0.0f32; src.len()];
    transpose_bshd_to_bhsd_into(src, &mut dst, batch, seq_len, heads, head_dim);
    dst
}

pub(crate) fn transpose_bhsd_to_bshd_raw(
    src: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    if seq_len <= 1 || heads <= 1 {
        return src.to_vec();
    }
    let mut dst = vec![0.0f32; src.len()];
    transpose_bhsd_to_bshd_into(src, &mut dst, batch, heads, seq_len, head_dim);
    dst
}

fn permute_bshd_to_bhsd(
    t: &Tensor,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> OchreResult<Tensor> {
    let out_data = transpose_bshd_to_bhsd_raw(&t.data, batch, seq_len, heads, head_dim);
    let out_shape = vec![batch, heads, seq_len, head_dim];
    let grad_active = AutogradTape::is_grad_enabled() && t.requires_grad;
    let node_id = if grad_active {
        let in_id = t.node_id;
        let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
            if let Some(id) = in_id {
                let dx = transpose_bhsd_to_bshd_raw(d_out, batch, heads, seq_len, head_dim);
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
        strides: compute_strides(&out_shape),
        shape: out_shape,
        data: out_data,
        dtype: t.dtype,
        device: t.device.clone(),
        requires_grad: grad_active,
        node_id,
    })
}

fn permute_bhsd_to_bshd(
    t: &Tensor,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> OchreResult<Tensor> {
    let out_data = transpose_bhsd_to_bshd_raw(&t.data, batch, heads, seq_len, head_dim);
    let out_shape = vec![batch, seq_len, heads * head_dim];
    let grad_active = AutogradTape::is_grad_enabled() && t.requires_grad;
    let node_id = if grad_active {
        let in_id = t.node_id;
        let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
            if let Some(id) = in_id {
                let dx = transpose_bshd_to_bhsd_raw(d_out, batch, seq_len, heads, head_dim);
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
        strides: compute_strides(&out_shape),
        shape: out_shape,
        data: out_data,
        dtype: t.dtype,
        device: t.device.clone(),
        requires_grad: grad_active,
        node_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::with_no_grad;
    use crate::nn::tensor::FastRng;

    #[allow(clippy::too_many_arguments)]
    fn naive_reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        do_grad: &[f32],
        batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        causal: bool,
        kv_offset: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let group_size = num_heads / num_kv_heads;
        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f32; q.len()];
        let mut dq = vec![0.0f32; q.len()];
        let mut dk = vec![0.0f32; k.len()];
        let mut dv = vec![0.0f32; v.len()];

        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / group_size;
                let q_off = (b * num_heads + h) * q_len * head_dim;
                let kv_off = (b * num_kv_heads + kv_h) * kv_len * head_dim;

                let mut probs = vec![0.0f64; q_len * kv_len];
                for qi in 0..q_len {
                    let mut row_scores = vec![f64::NEG_INFINITY; kv_len];
                    let mut max_s = f64::NEG_INFINITY;
                    for kj in 0..kv_len {
                        if causal && kj > kv_offset + qi {
                            continue;
                        }
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            dot += (q[q_off + qi * head_dim + d] as f64)
                                * (k[kv_off + kj * head_dim + d] as f64);
                        }
                        let s = dot * scale;
                        row_scores[kj] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut sum_e = 0.0f64;
                    for kj in 0..kv_len {
                        if row_scores[kj].is_finite() {
                            let e = (row_scores[kj] - max_s).exp();
                            probs[qi * kv_len + kj] = e;
                            sum_e += e;
                        }
                    }
                    if sum_e > 0.0 {
                        for kj in 0..kv_len {
                            probs[qi * kv_len + kj] /= sum_e;
                        }
                    }
                    for d in 0..head_dim {
                        let mut acc = 0.0f64;
                        for kj in 0..kv_len {
                            acc += probs[qi * kv_len + kj] * (v[kv_off + kj * head_dim + d] as f64);
                        }
                        out[q_off + qi * head_dim + d] = acc as f32;
                    }
                }

                // Backward
                for qi in 0..q_len {
                    let mut dp_row = vec![0.0f64; kv_len];
                    let mut d_sum = 0.0f64;
                    for kj in 0..kv_len {
                        let p = probs[qi * kv_len + kj];
                        let mut dp = 0.0f64;
                        for d in 0..head_dim {
                            let do_val = do_grad[q_off + qi * head_dim + d] as f64;
                            dv[kv_off + kj * head_dim + d] += (p * do_val) as f32;
                            dp += do_val * (v[kv_off + kj * head_dim + d] as f64);
                        }
                        dp_row[kj] = dp;
                        d_sum += p * dp;
                    }
                    for kj in 0..kv_len {
                        let ds = probs[qi * kv_len + kj] * (dp_row[kj] - d_sum) * scale;
                        for d in 0..head_dim {
                            dq[q_off + qi * head_dim + d] +=
                                (ds * (k[kv_off + kj * head_dim + d] as f64)) as f32;
                            dk[kv_off + kj * head_dim + d] +=
                                (ds * (q[q_off + qi * head_dim + d] as f64)) as f32;
                        }
                    }
                }
            }
        }
        (out, dq, dk, dv)
    }

    #[test]
    fn test_flash_attention_parity_gqa_and_causal_fwd_bwd() {
        let mut rng = FastRng::new(1337);
        // Test sizes > BR/BC=48 (e.g. q_len=100, kv_len=105) plus q_len=1 decode path!
        for &(batch, num_heads, num_kv_heads, q_len, kv_len, head_dim, causal, kv_offset) in &[
            (2, 4, 2, 40, 40, 16, true, 0),
            (1, 6, 2, 35, 42, 12, true, 7),
            (2, 4, 4, 33, 37, 8, false, 0),
            (1, 4, 2, 100, 105, 64, true, 5),
            (1, 4, 2, 1, 75, 32, true, 74),
        ] {
            let q: Vec<f32> = (0..batch * num_heads * q_len * head_dim)
                .map(|_| rng.uniform_f32(-0.8, 0.8))
                .collect();
            let k: Vec<f32> = (0..batch * num_kv_heads * kv_len * head_dim)
                .map(|_| rng.uniform_f32(-0.8, 0.8))
                .collect();
            let v: Vec<f32> = (0..batch * num_kv_heads * kv_len * head_dim)
                .map(|_| rng.uniform_f32(-0.8, 0.8))
                .collect();
            let do_grad: Vec<f32> = (0..q.len()).map(|_| rng.uniform_f32(-0.5, 0.5)).collect();

            let (ref_out, ref_dq, ref_dk, ref_dv) = naive_reference_attention(
                &q,
                &k,
                &v,
                &do_grad,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                causal,
                kv_offset,
            );
            let (flash_out, lse) = flash_attention_forward(
                &q,
                &k,
                &v,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                causal,
                kv_offset,
            )
            .unwrap();
            let (flash_dq, flash_dk, flash_dv) = flash_attention_backward(
                &do_grad,
                &q,
                &k,
                &v,
                &flash_out,
                &lse,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                causal,
                kv_offset,
            )
            .unwrap();

            for (name, got, exp) in [
                ("out", &flash_out, &ref_out),
                ("dq", &flash_dq, &ref_dq),
                ("dk", &flash_dk, &ref_dk),
                ("dv", &flash_dv, &ref_dv),
            ] {
                let max_err = got
                    .iter()
                    .zip(exp.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-5,
                    "FlashAttention {name} max_abs_err={max_err} >= 1e-5"
                );
            }
        }
    }

    #[test]
    fn test_rope_finite_diff_grad_check() {
        AutogradTape::clear();
        let rope = RoPE::new(6, 16, 10_000.0).unwrap();
        let x = Tensor::parameter(
            (0..2 * 2 * 3 * 6)
                .map(|i| ((i as f32) * 0.13).sin())
                .collect(),
            vec![2, 2, 3, 6],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::new(
            (0..2 * 2 * 3 * 6)
                .map(|i| ((i as f32) * 0.27).cos())
                .collect(),
            vec![2, 2, 3, 6],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let y = rope.forward(&x, 2).unwrap();
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();

        let direct_dx = rope.apply_backward(&w.data, 2, 2, 3, 2).unwrap();
        for (&a, &b) in dx.iter().zip(direct_dx.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let eps = 1e-3f32;
        for i in 0..x.numel() {
            let mut xp = x.data.clone();
            let mut xm = x.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp =
                    Tensor::new(xp, vec![2, 2, 3, 6], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm =
                    Tensor::new(xm, vec![2, 2, 3, 6], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = rope
                    .forward(&tp, 2)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = rope
                    .forward(&tm, 2)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!(
                (dx[i] - num).abs() < 5e-3,
                "RoPE dx[{i}] mismatch: {} vs {num}",
                dx[i]
            );
        }
    }

    #[test]
    fn test_kv_cache_incremental_decoding_parity() {
        let rope = RoPE::new(8, 32, 10_000.0).unwrap();
        let mha = MultiHeadAttention::new(
            16,
            2,
            1,
            true,
            true,
            Some(rope),
            DType::F32,
            DeviceType::Cpu,
            42,
        )
        .unwrap();

        let batch = 1;
        let seq_len = 6;
        let x = Tensor::randn(vec![batch, seq_len, 16], DType::F32, DeviceType::Cpu, 77);
        let full_out = with_no_grad(|| mha.forward_self(&x, None).unwrap());

        let mut cache = KVCache::new(batch, 1, 32, 8).unwrap();
        // Prefill first 3 tokens, then decode 1 token at a time
        let prefill_x = Tensor::new(
            x.data[..3 * 16].to_vec(),
            vec![batch, 3, 16],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let mut incremental = Vec::new();
        let p_out = with_no_grad(|| mha.forward_self(&prefill_x, Some(&mut cache)).unwrap());
        incremental.extend_from_slice(&p_out.data);

        for t in 3..seq_len {
            let step_x = Tensor::new(
                x.data[t * 16..(t + 1) * 16].to_vec(),
                vec![batch, 1, 16],
                DType::F32,
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let s_out = with_no_grad(|| mha.forward_self(&step_x, Some(&mut cache)).unwrap());
            incremental.extend_from_slice(&s_out.data);
        }

        let max_err = full_out
            .data
            .iter()
            .zip(incremental.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "KVCache parity max_err={max_err}");

        cache.reset();
        assert_eq!(cache.cur_len, 0);
    }

    #[test]
    fn test_positional_encoding_and_sdpa_autograd() {
        AutogradTape::clear();
        let pe = PositionalEncoding::new(16, 8).unwrap();
        let x = Tensor::parameter(
            vec![0.1; 2 * 4 * 8],
            vec![2, 4, 8],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let y = pe.forward(&x, 2).unwrap();
        let grads = y.sum(None).unwrap().backward().unwrap();
        assert!(
            grads
                .get(x.node_id.unwrap())
                .unwrap()
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-6)
        );

        let q = Tensor::parameter(vec![0.2; 32], vec![1, 2, 4, 4], DType::F32, DeviceType::Cpu)
            .unwrap();
        let k = Tensor::parameter(
            vec![-0.1; 16],
            vec![1, 1, 4, 4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let v = Tensor::parameter(vec![0.3; 16], vec![1, 1, 4, 4], DType::F32, DeviceType::Cpu)
            .unwrap();
        let attn = ScaledDotProductAttention::new()
            .forward(&q, &k, &v, true, 0)
            .unwrap();
        let g = attn.sum(None).unwrap().backward().unwrap();
        assert_eq!(g.get(q.node_id.unwrap()).unwrap().len(), 32);
        assert_eq!(g.get(k.node_id.unwrap()).unwrap().len(), 16);
        assert_eq!(g.get(v.node_id.unwrap()).unwrap().len(), 16);
    }

    #[test]
    fn test_sdpa_gpu_dispatch_with_cpu_fallback_parity() {
        AutogradTape::clear();
        let q_data: Vec<f32> = (0..2 * 4 * 6 * 8)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();
        let k_data: Vec<f32> = (0..2 * 2 * 6 * 8)
            .map(|i| ((i as f32) * 0.17).cos())
            .collect();
        let v_data: Vec<f32> = (0..2 * 2 * 6 * 8)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5)
            .collect();

        let q_cpu = Tensor::parameter(
            q_data.clone(),
            vec![2, 4, 6, 8],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let k_cpu = Tensor::parameter(
            k_data.clone(),
            vec![2, 2, 6, 8],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let v_cpu = Tensor::parameter(
            v_data.clone(),
            vec![2, 2, 6, 8],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let out_cpu = ScaledDotProductAttention::new()
            .forward(&q_cpu, &k_cpu, &v_cpu, true, 0)
            .unwrap();
        let g_cpu = out_cpu.sum(None).unwrap().backward().unwrap();

        AutogradTape::clear();
        let q_gpu =
            Tensor::parameter(q_data, vec![2, 4, 6, 8], DType::F32, DeviceType::Gpu(0)).unwrap();
        let k_gpu =
            Tensor::parameter(k_data, vec![2, 2, 6, 8], DType::F32, DeviceType::Gpu(0)).unwrap();
        let v_gpu =
            Tensor::parameter(v_data, vec![2, 2, 6, 8], DType::F32, DeviceType::Gpu(0)).unwrap();
        let out_gpu = ScaledDotProductAttention::new()
            .forward(&q_gpu, &k_gpu, &v_gpu, true, 0)
            .unwrap();
        let g_gpu = out_gpu.sum(None).unwrap().backward().unwrap();

        for (&a, &b) in out_cpu.data.iter().zip(out_gpu.data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "SDPA GPU/CPU fwd mismatch: {a} vs {b}"
            );
        }
        for (cid, gid) in [
            (q_cpu.node_id.unwrap(), q_gpu.node_id.unwrap()),
            (k_cpu.node_id.unwrap(), k_gpu.node_id.unwrap()),
            (v_cpu.node_id.unwrap(), v_gpu.node_id.unwrap()),
        ] {
            for (&a, &b) in g_cpu
                .get(cid)
                .unwrap()
                .iter()
                .zip(g_gpu.get(gid).unwrap().iter())
            {
                assert!(
                    (a - b).abs() < 1e-5,
                    "SDPA GPU/CPU bwd mismatch: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_sdpa_large_gpu_cached_fallback_parity() {
        // batch*num_heads*q_len*kv_len = 1*4*64*64 = 16_384 exercises the cached GPU dispatch branch
        // and verifies clean CPU fallback when CUDA runtime is unavailable.
        AutogradTape::clear();
        let (batch, num_heads, num_kv_heads, seq_len, head_dim) = (1, 4, 2, 64, 8);
        let q_data: Vec<f32> = (0..batch * num_heads * seq_len * head_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.4)
            .collect();
        let k_data: Vec<f32> = (0..batch * num_kv_heads * seq_len * head_dim)
            .map(|i| ((i as f32) * 0.09).cos() * 0.4)
            .collect();
        let v_data: Vec<f32> = (0..batch * num_kv_heads * seq_len * head_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.4)
            .collect();

        let q_cpu = Tensor::parameter(
            q_data.clone(),
            vec![batch, num_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let k_cpu = Tensor::parameter(
            k_data.clone(),
            vec![batch, num_kv_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let v_cpu = Tensor::parameter(
            v_data.clone(),
            vec![batch, num_kv_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let out_cpu = ScaledDotProductAttention::new()
            .forward(&q_cpu, &k_cpu, &v_cpu, true, 0)
            .unwrap();
        let g_cpu = out_cpu.sum(None).unwrap().backward().unwrap();

        AutogradTape::clear();
        let q_gpu = Tensor::parameter(
            q_data,
            vec![batch, num_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Gpu(0),
        )
        .unwrap();
        let k_gpu = Tensor::parameter(
            k_data,
            vec![batch, num_kv_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Gpu(0),
        )
        .unwrap();
        let v_gpu = Tensor::parameter(
            v_data,
            vec![batch, num_kv_heads, seq_len, head_dim],
            DType::F32,
            DeviceType::Gpu(0),
        )
        .unwrap();
        let out_gpu = ScaledDotProductAttention::new()
            .forward(&q_gpu, &k_gpu, &v_gpu, true, 0)
            .unwrap();
        let g_gpu = out_gpu.sum(None).unwrap().backward().unwrap();

        for (&a, &b) in out_cpu.data.iter().zip(out_gpu.data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(
            g_cpu.get(q_cpu.node_id.unwrap()).unwrap().len(),
            g_gpu.get(q_gpu.node_id.unwrap()).unwrap().len()
        );
    }
}
