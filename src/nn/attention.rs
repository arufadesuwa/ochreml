// OchreML Attention, Positional Encoding, RoPE, KVCache & FlashAttention Module (*≧ω≦*)
// Implements Sinusoidal PositionalEncoding, Rotary Positional Embeddings (RoPE),
// Flash-Style O(S)-memory tiled ScaledDotProductAttention (GQA/MQA + causal), KVCache, and MHA.

use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, CustomBackward, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::layers::Linear;
use crate::nn::tensor::{Tensor, compute_strides};
use rayon::prelude::*;
use std::sync::Arc;

const FLASH_BR: usize = 32;
const FLASH_BC: usize = 32;

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
        apply_rope_slice(
            &mut dx,
            &self.cos_table,
            &self.sin_table,
            self.head_dim,
            self.max_seq_len,
            batch,
            num_heads,
            seq_len,
            offset,
            true,
        )?;
        Ok(dx)
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
    let seq_stride = seq_len * head_dim;

    for bh in 0..(batch * num_heads) {
        let head_slice = &mut data[bh * seq_stride..(bh + 1) * seq_stride];
        for s in 0..seq_len {
            let pos = offset + s;
            let cos_row = &cos_table[pos * half..(pos + 1) * half];
            let sin_row = &sin_table[pos * half..(pos + 1) * half];
            let token_row = &mut head_slice[s * head_dim..(s + 1) * head_dim];
            let (first_half, second_half) = token_row.split_at_mut(half);
            for d in 0..half {
                let x0 = first_half[d];
                let x1 = second_half[d];
                let c = cos_row[d];
                let sn = sin_row[d];
                if inverse {
                    first_half[d] = x0 * c + x1 * sn;
                    second_half[d] = -x0 * sn + x1 * c;
                } else {
                    first_half[d] = x0 * c - x1 * sn;
                    second_half[d] = x0 * sn + x1 * c;
                }
            }
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
    let kv_head_stride = kv_stride_seq * head_dim;
    let group_size = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut out = vec![0.0f32; total_heads * q_head_stride];
    let mut lse = vec![f32::NEG_INFINITY; total_heads * q_len];

    if q_len == 0 {
        return Ok((out, lse));
    }
    if kv_len == 0 {
        return Ok((out, lse));
    }

    let run_head = |bh: usize, out_head: &mut [f32], lse_head: &mut [f32]| {
        let b = bh / num_heads;
        let h = bh % num_heads;
        let kv_h = h / group_size;
        let q_head = &q[bh * q_head_stride..(bh + 1) * q_head_stride];
        let kv_base = (b * num_kv_heads + kv_h) * kv_head_stride;
        let k_head = &k[kv_base..kv_base + kv_len * head_dim];
        let v_head = &v[kv_base..kv_base + kv_len * head_dim];

        // Reusable tile scratch buffers (O(Br*Bc + Br*D) per head, zero O(S^2) matrix).
        let mut m_i = [f32::NEG_INFINITY; FLASH_BR];
        let mut l_i = [0.0f32; FLASH_BR];
        let mut o_tile = vec![0.0f32; FLASH_BR * head_dim];
        let mut scores = [0.0f32; FLASH_BR * FLASH_BC];
        let mut probs = [0.0f32; FLASH_BR * FLASH_BC];

        let mut i_start = 0usize;
        while i_start < q_len {
            let br = (q_len - i_start).min(FLASH_BR);
            m_i[..br].fill(f32::NEG_INFINITY);
            l_i[..br].fill(0.0);
            o_tile[..br * head_dim].fill(0.0);

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

                // Compute S_tile = scale * Q_tile @ K_tile^T
                for r in 0..br {
                    let qi = i_start + r;
                    let q_row = &q_head[qi * head_dim..(qi + 1) * head_dim];
                    let score_row = &mut scores[r * bc..(r + 1) * bc];
                    for (c, score_slot) in score_row.iter_mut().enumerate() {
                        let kj = j_start + c;
                        if causal && kj > kv_offset + qi {
                            *score_slot = f32::NEG_INFINITY;
                        } else {
                            let k_row = &k_head[kj * head_dim..(kj + 1) * head_dim];
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_row[d] * k_row[d];
                            }
                            *score_slot = dot * scale;
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
                    if row_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let m_prev = m_i[r];
                    let m_new = m_prev.max(row_max);
                    let alpha = if m_prev.is_finite() {
                        (m_prev - m_new).exp()
                    } else {
                        0.0
                    };

                    let prob_row = &mut probs[r * bc..(r + 1) * bc];
                    let mut row_sum = 0.0f32;
                    for c in 0..bc {
                        let s = score_row[c];
                        let p = if s == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (s - m_new).exp()
                        };
                        prob_row[c] = p;
                        row_sum += p;
                    }

                    let o_row = &mut o_tile[r * head_dim..(r + 1) * head_dim];
                    if alpha != 1.0 {
                        for ov in o_row.iter_mut() {
                            *ov *= alpha;
                        }
                    }
                    for (c, &p) in prob_row.iter().enumerate() {
                        if p != 0.0 {
                            let v_row =
                                &v_head[(j_start + c) * head_dim..(j_start + c + 1) * head_dim];
                            for d in 0..head_dim {
                                o_row[d] += p * v_row[d];
                            }
                        }
                    }

                    l_i[r] = l_i[r] * alpha + row_sum;
                    m_i[r] = m_new;
                }

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
                    for d in 0..head_dim {
                        dst_row[d] = src_row[d] * inv_l;
                    }
                    lse_head[qi] = m_i[r] + li.ln();
                } else {
                    dst_row.fill(0.0);
                    lse_head[qi] = f32::NEG_INFINITY;
                }
            }

            i_start += FLASH_BR;
        }
    };

    if total_heads >= 2 {
        out.par_chunks_mut(q_head_stride)
            .zip(lse.par_chunks_mut(q_len))
            .enumerate()
            .for_each(|(bh, (out_head, lse_head))| run_head(bh, out_head, lse_head));
    } else {
        run_head(0, &mut out, &mut lse);
    }

    Ok((out, lse))
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
    let expected_lse = batch * num_heads * q_len;
    if lse.len() != expected_lse {
        return Err(OchreError::InvalidParameter(format!(
            "FlashAttention backward lse len {} != expected {expected_lse} ( >_< )",
            lse.len()
        )));
    }

    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    if q_len == 0 || kv_len == 0 {
        return Ok((dq, dk, dv));
    }

    let group_size = num_heads / num_kv_heads;
    let q_head_stride = q_len * head_dim;
    let kv_head_stride = kv_len * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let backward_one_head =
        |bh: usize, dq_head: &mut [f32], dk_head: &mut [f32], dv_head: &mut [f32]| {
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

            // D_i = sum_d dO[i, d] * O[i, d] (O(S) per head)
            let mut d_row = vec![0.0f32; q_len];
            for i in 0..q_len {
                let mut dot = 0.0f32;
                let o_r = &out_head[i * head_dim..(i + 1) * head_dim];
                let do_r = &do_head[i * head_dim..(i + 1) * head_dim];
                for d in 0..head_dim {
                    dot += o_r[d] * do_r[d];
                }
                d_row[i] = dot;
            }

            let mut p_tile = [0.0f32; FLASH_BR * FLASH_BC];
            let mut ds_tile = [0.0f32; FLASH_BR * FLASH_BC];

            let mut j_start = 0usize;
            while j_start < kv_len {
                let bc = (kv_len - j_start).min(FLASH_BC);
                let mut i_start = 0usize;
                while i_start < q_len {
                    let br = (q_len - i_start).min(FLASH_BR);
                    if causal && j_start > kv_offset + i_start + br - 1 {
                        i_start += FLASH_BR;
                        continue;
                    }

                    // Recompute P_ij = exp(scale * Q_i @ K_j^T - lse_i) and dP_ij = dO_i @ V_j^T
                    for r in 0..br {
                        let qi = i_start + r;
                        let lse_i = lse_head[qi];
                        let q_r = &q_head[qi * head_dim..(qi + 1) * head_dim];
                        let do_r = &do_head[qi * head_dim..(qi + 1) * head_dim];
                        let di = d_row[qi];

                        for c in 0..bc {
                            let kj = j_start + c;
                            if (causal && kj > kv_offset + qi) || !lse_i.is_finite() {
                                p_tile[r * bc + c] = 0.0;
                                ds_tile[r * bc + c] = 0.0;
                            } else {
                                let k_c = &k_head[kj * head_dim..(kj + 1) * head_dim];
                                let v_c = &v_head[kj * head_dim..(kj + 1) * head_dim];
                                let mut s = 0.0f32;
                                let mut dp = 0.0f32;
                                for d in 0..head_dim {
                                    s += q_r[d] * k_c[d];
                                    dp += do_r[d] * v_c[d];
                                }
                                let p = (s * scale - lse_i).exp();
                                p_tile[r * bc + c] = p;
                                ds_tile[r * bc + c] = p * (dp - di);
                            }
                        }
                    }

                    // Accumulate dV_j += P_ij^T @ dO_i and dK_j += scale * (dS_ij^T @ Q_i)
                    for c in 0..bc {
                        let kj = j_start + c;
                        let dv_c = &mut dv_head[kj * head_dim..(kj + 1) * head_dim];
                        let dk_c = &mut dk_head[kj * head_dim..(kj + 1) * head_dim];
                        for r in 0..br {
                            let qi = i_start + r;
                            let p = p_tile[r * bc + c];
                            let ds = ds_tile[r * bc + c] * scale;
                            if p != 0.0 {
                                let do_r = &do_head[qi * head_dim..(qi + 1) * head_dim];
                                for d in 0..head_dim {
                                    dv_c[d] += p * do_r[d];
                                }
                            }
                            if ds != 0.0 {
                                let q_r = &q_head[qi * head_dim..(qi + 1) * head_dim];
                                for d in 0..head_dim {
                                    dk_c[d] += ds * q_r[d];
                                }
                            }
                        }
                    }

                    // Accumulate dQ_i += scale * (dS_ij @ K_j)
                    for r in 0..br {
                        let qi = i_start + r;
                        let dq_r = &mut dq_head[qi * head_dim..(qi + 1) * head_dim];
                        for c in 0..bc {
                            let ds = ds_tile[r * bc + c] * scale;
                            if ds != 0.0 {
                                let kj = j_start + c;
                                let k_c = &k_head[kj * head_dim..(kj + 1) * head_dim];
                                for d in 0..head_dim {
                                    dq_r[d] += ds * k_c[d];
                                }
                            }
                        }
                    }

                    i_start += FLASH_BR;
                }
                j_start += FLASH_BC;
            }
        };

    if group_size == 1 && batch * num_heads >= 2 {
        dq.par_chunks_mut(q_head_stride)
            .zip(dk.par_chunks_mut(kv_head_stride))
            .zip(dv.par_chunks_mut(kv_head_stride))
            .enumerate()
            .for_each(|(bh, ((dq_h, dk_h), dv_h))| {
                backward_one_head(bh, dq_h, dk_h, dv_h);
            });
    } else if batch * num_kv_heads >= 2 {
        let q_group_stride = group_size * q_head_stride;
        dq.par_chunks_mut(q_group_stride)
            .zip(dk.par_chunks_mut(kv_head_stride))
            .zip(dv.par_chunks_mut(kv_head_stride))
            .enumerate()
            .for_each(|(b_kv, ((dq_grp, dk_h), dv_h))| {
                let b = b_kv / num_kv_heads;
                let kv_h = b_kv % num_kv_heads;
                for g in 0..group_size {
                    let h = kv_h * group_size + g;
                    let bh = b * num_heads + h;
                    let dq_h = &mut dq_grp[g * q_head_stride..(g + 1) * q_head_stride];
                    backward_one_head(bh, dq_h, dk_h, dv_h);
                }
            });
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
                    backward_one_head(bh, dq_h, dk_h, dv_h);
                }
            }
        }
    }

    Ok((dq, dk, dv))
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

        let (raw_out, lse) = flash_attention_forward(
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
        )?;
        let out_data = maybe_quantize(raw_out.clone(), q.dtype);
        let out_shape = vec![batch, num_heads, q_len, head_dim];

        let q_id = if q.requires_grad { q.node_id } else { None };
        let k_id = if k.requires_grad { k.node_id } else { None };
        let v_id = if v.requires_grad { v.node_id } else { None };
        let grad_active =
            AutogradTape::is_grad_enabled() && (q_id.is_some() || k_id.is_some() || v_id.is_some());

        let node_id = if grad_active {
            let q_saved = q.data.clone();
            let k_saved = k.data.clone();
            let v_saved = v.data.clone();
            let bwd = CustomBackward(Arc::new(move |d_out: &[f32]| {
                let (dq, dk, dv) = flash_attention_backward(
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
            Some(AutogradTape::record(Op::Custom {
                inputs: vec![q_id, k_id, v_id],
                backward: bwd,
            }))
        } else {
            None
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

        let (k_attn, v_attn, attn_kv_offset) = if let Some(cache) = kv_cache {
            cache.append(&k_heads.data, &v_heads.data, kv_len)?;
            let (k_all, v_all) = cache.slice_kv();
            let total_kv_len = cache.cur_len;
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
            let offset = total_kv_len.saturating_sub(q_len);
            (k_t, v_t, offset)
        } else {
            (k_heads, v_heads, 0usize)
        };

        let attn = ScaledDotProductAttention::new().forward(
            &q_heads,
            &k_attn,
            &v_attn,
            self.causal,
            attn_kv_offset,
        )?;

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

        let mut q_proj_buf = vec![0.0f32; q_rows * self.d_model];
        let mut k_proj_buf = vec![0.0f32; kv_rows * kv_dim];
        let mut v_proj_buf = vec![0.0f32; kv_rows * kv_dim];
        self.q_proj.forward_raw(q_in, q_rows, &mut q_proj_buf)?;
        self.k_proj.forward_raw(k_in, kv_rows, &mut k_proj_buf)?;
        self.v_proj.forward_raw(v_in, kv_rows, &mut v_proj_buf)?;

        let mut q_heads =
            transpose_bshd_to_bhsd_raw(&q_proj_buf, batch, q_len, self.num_heads, self.head_dim);
        let mut k_heads = transpose_bshd_to_bhsd_raw(
            &k_proj_buf,
            batch,
            kv_len,
            self.num_kv_heads,
            self.head_dim,
        );
        let v_heads = transpose_bshd_to_bhsd_raw(
            &v_proj_buf,
            batch,
            kv_len,
            self.num_kv_heads,
            self.head_dim,
        );

        let rope_offset = kv_cache.as_ref().map_or(0, |c| c.cur_len);
        if let Some(ref rope) = self.rope {
            rope.apply_inplace(&mut q_heads, batch, self.num_heads, q_len, rope_offset)?;
            rope.apply_inplace(&mut k_heads, batch, self.num_kv_heads, kv_len, rope_offset)?;
        }

        let attn_out = if let Some(cache) = kv_cache {
            cache.append(&k_heads, &v_heads, kv_len)?;
            let total_kv = cache.cur_len;
            let attn_offset = total_kv.saturating_sub(q_len);
            let (out_heads, _) = flash_attention_forward_strided(
                &q_heads,
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
            )?;
            out_heads
        } else {
            let (out_heads, _) = flash_attention_forward(
                &q_heads,
                &k_heads,
                &v_heads,
                batch,
                self.num_heads,
                self.num_kv_heads,
                q_len,
                kv_len,
                self.head_dim,
                self.causal,
                0,
            )?;
            out_heads
        };

        let merged =
            transpose_bhsd_to_bshd_raw(&attn_out, batch, self.num_heads, q_len, self.head_dim);
        self.o_proj.forward_raw(&merged, q_rows, out)
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

pub(crate) fn transpose_bshd_to_bhsd_raw(
    src: &[f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..heads {
                let s_off = ((b * seq_len + s) * heads + h) * head_dim;
                let d_off = ((b * heads + h) * seq_len + s) * head_dim;
                dst[d_off..d_off + head_dim].copy_from_slice(&src[s_off..s_off + head_dim]);
            }
        }
    }
    dst
}

pub(crate) fn transpose_bhsd_to_bshd_raw(
    src: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_len {
                let s_off = ((b * heads + h) * seq_len + s) * head_dim;
                let d_off = ((b * seq_len + s) * heads + h) * head_dim;
                dst[d_off..d_off + head_dim].copy_from_slice(&src[s_off..s_off + head_dim]);
            }
        }
    }
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
        // Test sizes > BR/BC=32 (e.g. q_len=40, kv_len=45) to exercise multi-tile online softmax!
        for &(batch, num_heads, num_kv_heads, q_len, kv_len, head_dim, causal, kv_offset) in &[
            (2, 4, 2, 40, 40, 16, true, 0),
            (1, 6, 2, 35, 42, 12, true, 7),
            (2, 4, 4, 33, 37, 8, false, 0),
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
}
