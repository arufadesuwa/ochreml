// OchreML Neural Network Hardware Primitives (N7..N10) (*≧ω≦*)
// N7: Batched f32 MatMul + TransposeLast2
// N8: LayerNorm, RMSNorm, GELU, SiLU (fwd + bwd) & Causal Mask
// N9: Fused Cross-Entropy + Grad & Fused AdamW Optimizer Step
// N10: im2col, col2im, MaxPool2d & AvgPool2d (fwd + bwd)

use crate::backend::Backend;
use crate::error::{OchreError, OchreResult};
use crate::nn::gemm_f32::batched_sgemm;
use crate::nn::simd_math::{
    simd_adamw_step_f32, simd_exp_sub_max_sum, simd_gelu_bwd_into, simd_gelu_fwd_into,
    simd_silu_bwd_into, simd_silu_fwd_into,
};
use rayon::prelude::*;

const PAR_ELEMS_THRESHOLD: usize = 16_384;
const PAR_ROWS_THRESHOLD: usize = 64;

#[allow(clippy::too_many_arguments)]
fn validate_conv_out_dims(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
) -> OchreResult<(usize, usize)> {
    if kh == 0 || kw == 0 || stride_h == 0 || stride_w == 0 || dil_h == 0 || dil_w == 0 {
        return Err(OchreError::InvalidParameter(
            "Kernel, stride, and dilation sizes must be >= 1 ( >_< )".to_string(),
        ));
    }
    let eff_kh = dil_h
        .checked_mul(kh - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| OchreError::InvalidParameter("Effective kh overflow ( >_< )".to_string()))?;
    let eff_kw = dil_w
        .checked_mul(kw - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| OchreError::InvalidParameter("Effective kw overflow ( >_< )".to_string()))?;

    let padded_h = h + 2 * pad_h;
    let padded_w = w + 2 * pad_w;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(OchreError::InvalidParameter(format!(
            "Padded input ({padded_h}x{padded_w}) smaller than effective kernel ({eff_kh}x{eff_kw}) ( >_< )"
        )));
    }
    let out_h = (padded_h - eff_kh) / stride_h + 1;
    let out_w = (padded_w - eff_kw) / stride_w + 1;
    Ok((out_h, out_w))
}

// ============================================================================
// CPU Reference & Rayon-Parallel Implementations (N7..N10)
// ============================================================================

/// N7: CPU batched `f32` matrix multiplication via cache-blocked AVX2+FMA SGEMM.
#[allow(clippy::too_many_arguments)]
pub fn cpu_batched_matmul_f32(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: bool,
    trans_b: bool,
) -> OchreResult<Vec<f32>> {
    let out_len = batch
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n))
        .ok_or_else(|| {
            OchreError::InvalidParameter(
                "batched_matmul_f32 output size overflow ( >_< )".to_string(),
            )
        })?;
    let mut out = vec![0.0f32; out_len];
    batched_sgemm(batch, m, n, k, a, b, &mut out, trans_a, trans_b)?;
    Ok(out)
}

/// N7: CPU batched last-2-dimension transpose `[batch, rows, cols] -> [batch, cols, rows]`.
pub fn cpu_transpose_last2_f32(
    batch: usize,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> OchreResult<Vec<f32>> {
    let total = batch
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(cols))
        .ok_or_else(|| {
            OchreError::InvalidParameter("transpose_last2_f32 size overflow ( >_< )".to_string())
        })?;
    if x.len() != total {
        return Err(OchreError::InvalidParameter(format!(
            "transpose_last2_f32 expected {total} elements, got {} ( >_< )",
            x.len()
        )));
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    let mat_stride = rows * cols;
    let mut out = vec![0.0f32; total];
    let tile = 32usize;

    let transpose_plane = |src: &[f32], dst: &mut [f32]| {
        for r0 in (0..rows).step_by(tile) {
            let r_end = (r0 + tile).min(rows);
            for c0 in (0..cols).step_by(tile) {
                let c_end = (c0 + tile).min(cols);
                for r in r0..r_end {
                    let s_row = &src[r * cols..r * cols + cols];
                    for c in c0..c_end {
                        dst[c * rows + r] = s_row[c];
                    }
                }
            }
        }
    };

    if batch >= 2 && total >= PAR_ELEMS_THRESHOLD {
        out.par_chunks_mut(mat_stride)
            .enumerate()
            .for_each(|(b, dst)| {
                let src = &x[b * mat_stride..(b + 1) * mat_stride];
                transpose_plane(src, dst);
            });
    } else {
        for b in 0..batch {
            let src = &x[b * mat_stride..(b + 1) * mat_stride];
            let dst = &mut out[b * mat_stride..(b + 1) * mat_stride];
            transpose_plane(src, dst);
        }
    }
    Ok(out)
}

/// N8: CPU row-wise LayerNorm forward returning `(out, mean, rstd)`.
pub fn cpu_row_layernorm_f32(
    x: &[f32],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let total = rows.checked_mul(cols).ok_or_else(|| {
        OchreError::InvalidParameter("row_layernorm_f32 dimensions overflow ( >_< )".to_string())
    })?;
    if x.len() != total {
        return Err(OchreError::InvalidParameter(format!(
            "row_layernorm_f32 expected x len {total}, got {} ( >_< )",
            x.len()
        )));
    }
    if let Some(g) = gamma
        && g.len() != cols
    {
        return Err(OchreError::InvalidParameter(format!(
            "row_layernorm_f32 gamma len {} != cols {cols} ( >_< )",
            g.len()
        )));
    }
    if let Some(b) = beta
        && b.len() != cols
    {
        return Err(OchreError::InvalidParameter(format!(
            "row_layernorm_f32 beta len {} != cols {cols} ( >_< )",
            b.len()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "row_layernorm_f32 invalid eps {eps} ( >_< )"
        )));
    }

    let mut out = vec![0.0f32; total];
    let mut mean = vec![0.0f32; rows];
    let mut rstd = vec![0.0f32; rows];
    let inv_cols = 1.0 / (cols as f64);
    let eps_d = eps as f64;

    let process_row = |r: usize, out_row: &mut [f32], m_slot: &mut f32, rs_slot: &mut f32| {
        let row = &x[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in row {
            sum += v as f64;
        }
        let mu = sum * inv_cols;

        let mut var_sum = 0.0f64;
        for &v in row {
            let d = (v as f64) - mu;
            var_sum += d * d;
        }
        let rs = 1.0 / (var_sum * inv_cols + eps_d).sqrt();
        *m_slot = mu as f32;
        *rs_slot = rs as f32;

        for j in 0..cols {
            let x_hat = ((row[j] as f64) - mu) * rs;
            let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
            let b = beta.map_or(0.0f64, |bv| bv[j] as f64);
            out_row[j] = (x_hat * g + b) as f32;
        }
    };

    if rows >= PAR_ROWS_THRESHOLD && total >= PAR_ELEMS_THRESHOLD {
        out.par_chunks_mut(cols)
            .zip(mean.par_iter_mut())
            .zip(rstd.par_iter_mut())
            .enumerate()
            .for_each(|(r, ((out_row, m_slot), rs_slot))| {
                process_row(r, out_row, m_slot, rs_slot);
            });
    } else {
        for r in 0..rows {
            process_row(
                r,
                &mut out[r * cols..(r + 1) * cols],
                &mut mean[r],
                &mut rstd[r],
            );
        }
    }

    Ok((out, mean, rstd))
}

/// N8: CPU row-wise LayerNorm backward returning `(dx, dgamma, dbeta)`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_row_layernorm_bwd_f32(
    dy: &[f32],
    x: &[f32],
    gamma: Option<&[f32]>,
    mean: &[f32],
    rstd: &[f32],
    rows: usize,
    cols: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let total = rows * cols;
    if dy.len() != total || x.len() != total || mean.len() != rows || rstd.len() != rows {
        return Err(OchreError::InvalidParameter(
            "row_layernorm_bwd_f32 input length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(g) = gamma
        && g.len() != cols
    {
        return Err(OchreError::InvalidParameter(
            "row_layernorm_bwd_f32 gamma len mismatch ( >_< )".to_string(),
        ));
    }

    let mut dx = vec![0.0f32; total];
    let inv_cols = 1.0 / (cols as f64);

    if rows >= PAR_ROWS_THRESHOLD && total >= PAR_ELEMS_THRESHOLD {
        let (dgamma_acc, dbeta_acc) = dx
            .par_chunks_mut(cols)
            .enumerate()
            .fold(
                || (vec![0.0f64; cols], vec![0.0f64; cols]),
                |(mut dg, mut db), (r, dx_row)| {
                    let dy_row = &dy[r * cols..(r + 1) * cols];
                    let x_row = &x[r * cols..(r + 1) * cols];
                    let mu = mean[r] as f64;
                    let rs = rstd[r] as f64;

                    let mut sum1 = 0.0f64;
                    let mut sum2 = 0.0f64;
                    for j in 0..cols {
                        let dyv = dy_row[j] as f64;
                        let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                        let dy_g = dyv * g;
                        let x_hat = ((x_row[j] as f64) - mu) * rs;
                        sum1 += dy_g;
                        sum2 += dy_g * x_hat;
                        dg[j] += dyv * x_hat;
                        db[j] += dyv;
                    }
                    for j in 0..cols {
                        let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                        let dy_g = (dy_row[j] as f64) * g;
                        let x_hat = ((x_row[j] as f64) - mu) * rs;
                        dx_row[j] = (rs * (dy_g - inv_cols * (sum1 + x_hat * sum2))) as f32;
                    }
                    (dg, db)
                },
            )
            .reduce(
                || (vec![0.0f64; cols], vec![0.0f64; cols]),
                |(mut dg_a, mut db_a), (dg_b, db_b)| {
                    for j in 0..cols {
                        dg_a[j] += dg_b[j];
                        db_a[j] += db_b[j];
                    }
                    (dg_a, db_a)
                },
            );
        let dgamma = dgamma_acc.into_iter().map(|v| v as f32).collect();
        let dbeta = dbeta_acc.into_iter().map(|v| v as f32).collect();
        Ok((dx, dgamma, dbeta))
    } else {
        let mut dgamma_acc = vec![0.0f64; cols];
        let mut dbeta_acc = vec![0.0f64; cols];
        for r in 0..rows {
            let dy_row = &dy[r * cols..(r + 1) * cols];
            let x_row = &x[r * cols..(r + 1) * cols];
            let dx_row = &mut dx[r * cols..(r + 1) * cols];
            let mu = mean[r] as f64;
            let rs = rstd[r] as f64;

            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            for j in 0..cols {
                let dyv = dy_row[j] as f64;
                let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                let dy_g = dyv * g;
                let x_hat = ((x_row[j] as f64) - mu) * rs;
                sum1 += dy_g;
                sum2 += dy_g * x_hat;
                dgamma_acc[j] += dyv * x_hat;
                dbeta_acc[j] += dyv;
            }
            for j in 0..cols {
                let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                let dy_g = (dy_row[j] as f64) * g;
                let x_hat = ((x_row[j] as f64) - mu) * rs;
                dx_row[j] = (rs * (dy_g - inv_cols * (sum1 + x_hat * sum2))) as f32;
            }
        }
        let dgamma = dgamma_acc.into_iter().map(|v| v as f32).collect();
        let dbeta = dbeta_acc.into_iter().map(|v| v as f32).collect();
        Ok((dx, dgamma, dbeta))
    }
}

/// N8: CPU row-wise RMSNorm forward returning `(out, rstd)`.
pub fn cpu_row_rmsnorm_f32(
    x: &[f32],
    gamma: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> OchreResult<(Vec<f32>, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let total = rows.checked_mul(cols).ok_or_else(|| {
        OchreError::InvalidParameter("row_rmsnorm_f32 dimensions overflow ( >_< )".to_string())
    })?;
    if x.len() != total {
        return Err(OchreError::InvalidParameter(format!(
            "row_rmsnorm_f32 expected x len {total}, got {} ( >_< )",
            x.len()
        )));
    }
    if let Some(g) = gamma
        && g.len() != cols
    {
        return Err(OchreError::InvalidParameter(format!(
            "row_rmsnorm_f32 gamma len {} != cols {cols} ( >_< )",
            g.len()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "row_rmsnorm_f32 invalid eps {eps} ( >_< )"
        )));
    }

    let mut out = vec![0.0f32; total];
    let mut rstd = vec![0.0f32; rows];
    let inv_cols = 1.0 / (cols as f64);
    let eps_d = eps as f64;

    let process_row = |r: usize, out_row: &mut [f32], rs_slot: &mut f32| {
        let row = &x[r * cols..(r + 1) * cols];
        let mut sq = 0.0f64;
        for &v in row {
            let vd = v as f64;
            sq += vd * vd;
        }
        let rs = 1.0 / (sq * inv_cols + eps_d).sqrt();
        *rs_slot = rs as f32;
        for j in 0..cols {
            let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
            out_row[j] = ((row[j] as f64) * rs * g) as f32;
        }
    };

    if rows >= PAR_ROWS_THRESHOLD && total >= PAR_ELEMS_THRESHOLD {
        out.par_chunks_mut(cols)
            .zip(rstd.par_iter_mut())
            .enumerate()
            .for_each(|(r, (out_row, rs_slot))| process_row(r, out_row, rs_slot));
    } else {
        for r in 0..rows {
            process_row(r, &mut out[r * cols..(r + 1) * cols], &mut rstd[r]);
        }
    }
    Ok((out, rstd))
}

/// N8: CPU row-wise RMSNorm backward returning `(dx, dgamma)`.
pub fn cpu_row_rmsnorm_bwd_f32(
    dy: &[f32],
    x: &[f32],
    gamma: Option<&[f32]>,
    rstd: &[f32],
    rows: usize,
    cols: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let total = rows * cols;
    if dy.len() != total || x.len() != total || rstd.len() != rows {
        return Err(OchreError::InvalidParameter(
            "row_rmsnorm_bwd_f32 input length mismatch ( >_< )".to_string(),
        ));
    }
    if let Some(g) = gamma
        && g.len() != cols
    {
        return Err(OchreError::InvalidParameter(
            "row_rmsnorm_bwd_f32 gamma len mismatch ( >_< )".to_string(),
        ));
    }

    let mut dx = vec![0.0f32; total];
    let inv_cols = 1.0 / (cols as f64);

    if rows >= PAR_ROWS_THRESHOLD && total >= PAR_ELEMS_THRESHOLD {
        let dgamma_acc = dx
            .par_chunks_mut(cols)
            .enumerate()
            .fold(
                || vec![0.0f64; cols],
                |mut dg, (r, dx_row)| {
                    let dy_row = &dy[r * cols..(r + 1) * cols];
                    let x_row = &x[r * cols..(r + 1) * cols];
                    let rs = rstd[r] as f64;
                    let mut dot = 0.0f64;
                    for j in 0..cols {
                        let dyv = dy_row[j] as f64;
                        let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                        let x_hat = (x_row[j] as f64) * rs;
                        dot += (dyv * g) * x_hat;
                        dg[j] += dyv * x_hat;
                    }
                    for j in 0..cols {
                        let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                        let dy_g = (dy_row[j] as f64) * g;
                        let x_hat = (x_row[j] as f64) * rs;
                        dx_row[j] = (rs * (dy_g - x_hat * dot * inv_cols)) as f32;
                    }
                    dg
                },
            )
            .reduce(
                || vec![0.0f64; cols],
                |mut a, b| {
                    for j in 0..cols {
                        a[j] += b[j];
                    }
                    a
                },
            );
        let dgamma = dgamma_acc.into_iter().map(|v| v as f32).collect();
        Ok((dx, dgamma))
    } else {
        let mut dgamma_acc = vec![0.0f64; cols];
        for r in 0..rows {
            let dy_row = &dy[r * cols..(r + 1) * cols];
            let x_row = &x[r * cols..(r + 1) * cols];
            let dx_row = &mut dx[r * cols..(r + 1) * cols];
            let rs = rstd[r] as f64;
            let mut dot = 0.0f64;
            for j in 0..cols {
                let dyv = dy_row[j] as f64;
                let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                let x_hat = (x_row[j] as f64) * rs;
                dot += (dyv * g) * x_hat;
                dgamma_acc[j] += dyv * x_hat;
            }
            for j in 0..cols {
                let g = gamma.map_or(1.0f64, |gv| gv[j] as f64);
                let dy_g = (dy_row[j] as f64) * g;
                let x_hat = (x_row[j] as f64) * rs;
                dx_row[j] = (rs * (dy_g - x_hat * dot * inv_cols)) as f32;
            }
        }
        let dgamma = dgamma_acc.into_iter().map(|v| v as f32).collect();
        Ok((dx, dgamma))
    }
}

/// N8: CPU GELU forward (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`).
pub fn cpu_gelu_f32(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    simd_gelu_fwd_into(x, &mut out);
    out
}

/// N8: CPU GELU backward.
pub fn cpu_gelu_bwd_f32(dy: &[f32], x: &[f32]) -> Vec<f32> {
    let n = dy.len().min(x.len());
    let mut dx = vec![0.0f32; n];
    simd_gelu_bwd_into(&dy[..n], &x[..n], &mut dx);
    dx
}

/// N8: CPU SiLU / Swish forward (`x * sigmoid(x)`).
pub fn cpu_silu_f32(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    simd_silu_fwd_into(x, &mut out);
    out
}

/// N8: CPU SiLU backward.
pub fn cpu_silu_bwd_f32(dy: &[f32], x: &[f32]) -> Vec<f32> {
    let n = dy.len().min(x.len());
    let mut dx = vec![0.0f32; n];
    simd_silu_bwd_into(&dy[..n], &x[..n], &mut dx);
    dx
}

/// N8: CPU causal attention mask in-place (`kj > offset + qi` set to `-inf`).
pub fn cpu_causal_mask_f32(
    scores: &mut [f32],
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    offset: usize,
) -> OchreResult<()> {
    let total = batch_heads
        .checked_mul(q_len)
        .and_then(|v| v.checked_mul(k_len))
        .ok_or_else(|| {
            OchreError::InvalidParameter("causal_mask_f32 dimensions overflow ( >_< )".to_string())
        })?;
    if scores.len() != total {
        return Err(OchreError::InvalidParameter(format!(
            "causal_mask_f32 expected {total} elements, got {} ( >_< )",
            scores.len()
        )));
    }
    if total == 0 {
        return Ok(());
    }
    let plane = q_len * k_len;
    let mask_plane = |mat: &mut [f32]| {
        for qi in 0..q_len {
            let start_k = (offset + qi + 1).min(k_len);
            if start_k < k_len {
                mat[qi * k_len + start_k..qi * k_len + k_len].fill(f32::NEG_INFINITY);
            }
        }
    };
    if batch_heads >= 2 && total >= PAR_ELEMS_THRESHOLD {
        scores.par_chunks_mut(plane).for_each(mask_plane);
    } else {
        for bh in 0..batch_heads {
            mask_plane(&mut scores[bh * plane..(bh + 1) * plane]);
        }
    }
    Ok(())
}

/// N9: CPU fused Cross-Entropy + Softmax gradient returning `(mean_loss, d_logits)`.
pub fn cpu_cross_entropy_grad_f32(
    logits: &[f32],
    targets: &[usize],
    rows: usize,
    cols: usize,
    label_smoothing: f32,
    ignore_index: Option<usize>,
) -> OchreResult<(f32, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let total = rows.checked_mul(cols).ok_or_else(|| {
        OchreError::InvalidParameter("cross_entropy_grad_f32 size overflow ( >_< )".to_string())
    })?;
    if logits.len() != total || targets.len() != rows {
        return Err(OchreError::InvalidParameter(
            "cross_entropy_grad_f32 input length mismatch ( >_< )".to_string(),
        ));
    }
    if !label_smoothing.is_finite() || !(0.0..=1.0).contains(&label_smoothing) {
        return Err(OchreError::InvalidParameter(format!(
            "label_smoothing ({label_smoothing}) must be in [0.0, 1.0] ( >_< )"
        )));
    }

    let mut valid_count = 0usize;
    for &t in targets {
        if Some(t) == ignore_index {
            continue;
        }
        if t >= cols {
            return Err(OchreError::InvalidParameter(format!(
                "Target class index {t} out of range for {cols} classes ( >_< )"
            )));
        }
        valid_count += 1;
    }
    if valid_count == 0 {
        return Ok((0.0, vec![0.0; total]));
    }

    let mut d_logits = vec![0.0f32; total];
    let inv_valid = 1.0 / (valid_count as f64);
    let ls = label_smoothing as f64;
    let smooth_base = ls / (cols as f64);
    let one_minus_ls = 1.0 - ls;

    let compute_row = |r: usize, grad_row: &mut [f32]| -> f64 {
        let tgt = targets[r];
        if Some(tgt) == ignore_index {
            grad_row.fill(0.0);
            return 0.0;
        }
        let row = &logits[r * cols..(r + 1) * cols];
        let (max_v, sum_exp) = simd_exp_sub_max_sum(row, grad_row);
        let sum_exp_d = sum_exp as f64;
        let log_z = (max_v as f64) + sum_exp_d.ln();
        let inv_sum_exp = 1.0 / sum_exp_d;
        if label_smoothing == 0.0 {
            let scale = (inv_sum_exp * inv_valid) as f32;
            for g in grad_row.iter_mut() {
                *g *= scale;
            }
            grad_row[tgt] -= inv_valid as f32;
            log_z - (row[tgt] as f64)
        } else {
            let mut row_loss = 0.0f64;
            for (j, &v) in row.iter().enumerate() {
                let prob = (grad_row[j] as f64) * inv_sum_exp;
                let q = if j == tgt {
                    smooth_base + one_minus_ls
                } else {
                    smooth_base
                };
                row_loss -= q * ((v as f64) - log_z);
                grad_row[j] = ((prob - q) * inv_valid) as f32;
            }
            row_loss
        }
    };

    let total_loss: f64 = if rows >= PAR_ROWS_THRESHOLD && total >= PAR_ELEMS_THRESHOLD {
        d_logits
            .par_chunks_mut(cols)
            .enumerate()
            .map(|(r, grad_row)| compute_row(r, grad_row))
            .sum()
    } else {
        let mut acc = 0.0f64;
        for r in 0..rows {
            acc += compute_row(r, &mut d_logits[r * cols..(r + 1) * cols]);
        }
        acc
    };

    Ok(((total_loss * inv_valid) as f32, d_logits))
}

/// N9: CPU fused AdamW optimizer update step.
#[allow(clippy::too_many_arguments)]
pub fn cpu_adamw_step_f32(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
) -> OchreResult<()> {
    let n = param.len();
    if grad.len() != n || m.len() != n || v.len() != n {
        return Err(OchreError::InvalidParameter(
            "adamw_step_f32 slice lengths must all match ( >_< )".to_string(),
        ));
    }
    if step == 0 {
        return Err(OchreError::InvalidParameter(
            "adamw_step_f32 step must be >= 1 ( >_< )".to_string(),
        ));
    }
    if !lr.is_finite()
        || lr < 0.0
        || !beta1.is_finite()
        || !(0.0..1.0).contains(&beta1)
        || !beta2.is_finite()
        || !(0.0..1.0).contains(&beta2)
        || !eps.is_finite()
        || eps <= 0.0
        || !weight_decay.is_finite()
        || weight_decay < 0.0
    {
        return Err(OchreError::InvalidParameter(
            "adamw_step_f32 invalid hyperparameters ( >_< )".to_string(),
        ));
    }
    if n == 0 {
        return Ok(());
    }

    simd_adamw_step_f32(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step);
    Ok(())
}

/// N10: CPU `im2col` lowering `[batch, c, h, w]` into `[(batch*out_h*out_w), (c*kh*kw)]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_im2col_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
) -> OchreResult<(Vec<f32>, usize, usize)> {
    let total_in = batch
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| {
            OchreError::InvalidParameter("im2col_f32 input size overflow ( >_< )".to_string())
        })?;
    if input.len() != total_in {
        return Err(OchreError::InvalidParameter(format!(
            "im2col_f32 expected {total_in} input elements, got {} ( >_< )",
            input.len()
        )));
    }
    let (out_h, out_w) =
        validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)?;
    let n_rows = batch * out_h * out_w;
    let col_w = c * kh * kw;
    let mut col = vec![0.0f32; n_rows * col_w];

    let fill_row = |row_idx: usize, dst_row: &mut [f32]| {
        let ow_i = row_idx % out_w;
        let tmp = row_idx / out_w;
        let oh_i = tmp % out_h;
        let b = tmp / out_h;
        let in_b_base = b * c * h * w;

        let mut col_idx = 0;
        for ch in 0..c {
            let ch_base = in_b_base + ch * h * w;
            for ky in 0..kh {
                let iy = (oh_i * stride_h + ky * dil_h) as isize - (pad_h as isize);
                if iy < 0 || iy >= h as isize {
                    dst_row[col_idx..col_idx + kw].fill(0.0);
                    col_idx += kw;
                    continue;
                }
                let row_base = ch_base + (iy as usize) * w;
                for kx in 0..kw {
                    let ix = (ow_i * stride_w + kx * dil_w) as isize - (pad_w as isize);
                    dst_row[col_idx] = if ix >= 0 && ix < w as isize {
                        input[row_base + (ix as usize)]
                    } else {
                        0.0
                    };
                    col_idx += 1;
                }
            }
        }
    };

    if n_rows * col_w >= PAR_ELEMS_THRESHOLD {
        col.par_chunks_mut(col_w)
            .enumerate()
            .for_each(|(r, dst)| fill_row(r, dst));
    } else {
        for r in 0..n_rows {
            fill_row(r, &mut col[r * col_w..(r + 1) * col_w]);
        }
    }
    Ok((col, out_h, out_w))
}

/// N10: CPU `col2im` accumulating `[(batch*out_h*out_w), (c*kh*kw)]` back into `[batch, c, h, w]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_col2im_f32(
    col: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
    out_h: usize,
    out_w: usize,
) -> OchreResult<Vec<f32>> {
    let (exp_oh, exp_ow) =
        validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)?;
    if out_h != exp_oh || out_w != exp_ow {
        return Err(OchreError::InvalidParameter(format!(
            "col2im_f32 out_h/out_w ({out_h}x{out_w}) mismatch expected ({exp_oh}x{exp_ow}) ( >_< )"
        )));
    }
    let col_w = c * kh * kw;
    let n_rows = batch * out_h * out_w;
    if col.len() != n_rows * col_w {
        return Err(OchreError::InvalidParameter(format!(
            "col2im_f32 expected {} elements, got {} ( >_< )",
            n_rows * col_w,
            col.len()
        )));
    }

    let plane_in = c * h * w;
    let plane_rows = out_h * out_w;
    let mut img = vec![0.0f32; batch * plane_in];

    let scatter_batch = |b: usize, img_b: &mut [f32]| {
        let col_b = &col[b * plane_rows * col_w..(b + 1) * plane_rows * col_w];
        for oh_i in 0..out_h {
            for ow_i in 0..out_w {
                let src_row = &col_b[(oh_i * out_w + ow_i) * col_w..];
                let mut col_idx = 0;
                for ch in 0..c {
                    let ch_base = ch * h * w;
                    for ky in 0..kh {
                        let iy = (oh_i * stride_h + ky * dil_h) as isize - (pad_h as isize);
                        if iy < 0 || iy >= h as isize {
                            col_idx += kw;
                            continue;
                        }
                        let row_base = ch_base + (iy as usize) * w;
                        for kx in 0..kw {
                            let ix = (ow_i * stride_w + kx * dil_w) as isize - (pad_w as isize);
                            if ix >= 0 && ix < w as isize {
                                img_b[row_base + (ix as usize)] += src_row[col_idx];
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    };

    if batch >= 2 && col.len() >= PAR_ELEMS_THRESHOLD {
        img.par_chunks_mut(plane_in)
            .enumerate()
            .for_each(|(b, img_b)| scatter_batch(b, img_b));
    } else {
        for b in 0..batch {
            scatter_batch(b, &mut img[b * plane_in..(b + 1) * plane_in]);
        }
    }
    Ok(img)
}

/// N10: CPU MaxPool2D forward returning `(out, argmax_indices, out_h, out_w)`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_max_pool2d_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> OchreResult<(Vec<f32>, Vec<usize>, usize, usize)> {
    let total_in = batch
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| {
            OchreError::InvalidParameter("max_pool2d_f32 size overflow ( >_< )".to_string())
        })?;
    if input.len() != total_in {
        return Err(OchreError::InvalidParameter(format!(
            "max_pool2d_f32 expected {total_in} elements, got {} ( >_< )",
            input.len()
        )));
    }
    let (out_h, out_w) =
        validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1)?;
    let out_plane = out_h * out_w;
    let total_out = batch * c * out_plane;
    let mut out = vec![0.0f32; total_out];
    let mut argmax = vec![usize::MAX; total_out];

    let pool_channel = |bc: usize, out_ch: &mut [f32], arg_ch: &mut [usize]| {
        let in_base = bc * h * w;
        for oh_i in 0..out_h {
            for ow_i in 0..out_w {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = usize::MAX;
                for ky in 0..kh {
                    let iy = (oh_i * stride_h + ky) as isize - (pad_h as isize);
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    let row_off = in_base + (iy as usize) * w;
                    for kx in 0..kw {
                        let ix = (ow_i * stride_w + kx) as isize - (pad_w as isize);
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let idx = row_off + (ix as usize);
                        let val = input[idx];
                        if val > best_val || best_idx == usize::MAX {
                            best_val = val;
                            best_idx = idx;
                        }
                    }
                }
                let out_idx = oh_i * out_w + ow_i;
                out_ch[out_idx] = best_val;
                arg_ch[out_idx] = best_idx;
            }
        }
    };

    if total_out >= PAR_ELEMS_THRESHOLD && batch * c >= 4 {
        out.par_chunks_mut(out_plane)
            .zip(argmax.par_chunks_mut(out_plane))
            .enumerate()
            .for_each(|(bc, (och, ach))| pool_channel(bc, och, ach));
    } else {
        for bc in 0..(batch * c) {
            pool_channel(
                bc,
                &mut out[bc * out_plane..(bc + 1) * out_plane],
                &mut argmax[bc * out_plane..(bc + 1) * out_plane],
            );
        }
    }

    Ok((out, argmax, out_h, out_w))
}

/// N10: CPU MaxPool2D backward scattering `dy` via saved `argmax` indices.
pub fn cpu_max_pool2d_bwd_f32(dy: &[f32], argmax: &[usize], total_in_len: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; total_in_len];
    for (&g, &idx) in dy.iter().zip(argmax.iter()) {
        if idx < total_in_len {
            dx[idx] += g;
        }
    }
    dx
}

/// N10: CPU AvgPool2D forward (`count_include_pad = true`, divisor `kh * kw`).
#[allow(clippy::too_many_arguments)]
pub fn cpu_avg_pool2d_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> OchreResult<(Vec<f32>, usize, usize)> {
    let total_in = batch
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| {
            OchreError::InvalidParameter("avg_pool2d_f32 size overflow ( >_< )".to_string())
        })?;
    if input.len() != total_in {
        return Err(OchreError::InvalidParameter(format!(
            "avg_pool2d_f32 expected {total_in} elements, got {} ( >_< )",
            input.len()
        )));
    }
    let (out_h, out_w) =
        validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1)?;
    let out_plane = out_h * out_w;
    let total_out = batch * c * out_plane;
    let mut out = vec![0.0f32; total_out];
    let inv_area = 1.0 / ((kh * kw) as f32);

    let pool_channel = |bc: usize, out_ch: &mut [f32]| {
        let in_base = bc * h * w;
        for oh_i in 0..out_h {
            for ow_i in 0..out_w {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let iy = (oh_i * stride_h + ky) as isize - (pad_h as isize);
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    let row_off = in_base + (iy as usize) * w;
                    for kx in 0..kw {
                        let ix = (ow_i * stride_w + kx) as isize - (pad_w as isize);
                        if ix >= 0 && ix < w as isize {
                            sum += input[row_off + (ix as usize)];
                        }
                    }
                }
                out_ch[oh_i * out_w + ow_i] = sum * inv_area;
            }
        }
    };

    if total_out >= PAR_ELEMS_THRESHOLD && batch * c >= 4 {
        out.par_chunks_mut(out_plane)
            .enumerate()
            .for_each(|(bc, och)| pool_channel(bc, och));
    } else {
        for bc in 0..(batch * c) {
            pool_channel(bc, &mut out[bc * out_plane..(bc + 1) * out_plane]);
        }
    }

    Ok((out, out_h, out_w))
}

/// N10: CPU AvgPool2D backward distributing `dy / (kh * kw)` into `[batch, c, h, w]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_avg_pool2d_bwd_f32(
    dy: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
) -> OchreResult<Vec<f32>> {
    let (exp_oh, exp_ow) =
        validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1)?;
    if out_h != exp_oh || out_w != exp_ow {
        return Err(OchreError::InvalidParameter(format!(
            "avg_pool2d_bwd_f32 out_h/out_w ({out_h}x{out_w}) mismatch expected ({exp_oh}x{exp_ow}) ( >_< )"
        )));
    }
    let out_plane = out_h * out_w;
    let total_out = batch * c * out_plane;
    if dy.len() != total_out {
        return Err(OchreError::InvalidParameter(format!(
            "avg_pool2d_bwd_f32 expected {total_out} dy elements, got {} ( >_< )",
            dy.len()
        )));
    }
    let in_plane = h * w;
    let mut dx = vec![0.0f32; batch * c * in_plane];
    let inv_area = 1.0 / ((kh * kw) as f32);

    let bwd_channel = |bc: usize, dx_ch: &mut [f32]| {
        let dy_ch = &dy[bc * out_plane..(bc + 1) * out_plane];
        for oh_i in 0..out_h {
            for ow_i in 0..out_w {
                let g = dy_ch[oh_i * out_w + ow_i] * inv_area;
                for ky in 0..kh {
                    let iy = (oh_i * stride_h + ky) as isize - (pad_h as isize);
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    let row_off = (iy as usize) * w;
                    for kx in 0..kw {
                        let ix = (ow_i * stride_w + kx) as isize - (pad_w as isize);
                        if ix >= 0 && ix < w as isize {
                            dx_ch[row_off + (ix as usize)] += g;
                        }
                    }
                }
            }
        }
    };

    if total_out >= PAR_ELEMS_THRESHOLD && batch * c >= 4 {
        dx.par_chunks_mut(in_plane)
            .enumerate()
            .for_each(|(bc, dx_ch)| bwd_channel(bc, dx_ch));
    } else {
        for bc in 0..(batch * c) {
            bwd_channel(bc, &mut dx[bc * in_plane..(bc + 1) * in_plane]);
        }
    }

    Ok(dx)
}

// ============================================================================
// Backend<'a> Hardware Dispatch Methods (N7..N10)
// ============================================================================

impl<'a> Backend<'a> {
    /// N7: Batched `f32` matrix multiplication across CPU AVX2+FMA, CUDA, wgpu, and TPU.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_matmul_f32(
        &self,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        trans_a: bool,
        trans_b: bool,
    ) -> OchreResult<Vec<f32>> {
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            return cpu_batched_matmul_f32(batch, m, n, k, a, b, trans_a, trans_b);
        }
        let req_a = batch.saturating_mul(m).saturating_mul(k);
        let req_b = batch.saturating_mul(k).saturating_mul(n);
        if a.len() < req_a || b.len() < req_b {
            return cpu_batched_matmul_f32(batch, m, n, k, a, b, trans_a, trans_b);
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_batched_matmul_f32(
                    *id, batch, m, n, k, a, b, trans_a, trans_b,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_batched_matmul_f32(
                    ids[0], batch, m, n, k, a, b, trans_a, trans_b,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_batched_matmul_f32(
                batch, m, n, k, a, b, trans_a, trans_b,
            )
        {
            return res;
        }

        #[cfg(feature = "tpu-pjrt")]
        if let Backend::Tpu(id) = self
            && let Some(res) = crate::backends::tpu_nn::tpu_batched_matmul_f32(
                *id, batch, m, n, k, a, b, trans_a, trans_b,
            )
        {
            return res;
        }

        cpu_batched_matmul_f32(batch, m, n, k, a, b, trans_a, trans_b)
    }

    /// N7: Batched transpose of the last two dimensions `[batch, rows, cols] -> [batch, cols, rows]`.
    pub fn transpose_last2_f32(
        &self,
        batch: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> OchreResult<Vec<f32>> {
        cpu_transpose_last2_f32(batch, rows, cols, x)
    }

    /// N8: Row-wise LayerNorm forward returning `(out, mean, rstd)`.
    pub fn row_layernorm_f32(
        &self,
        x: &[f32],
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if rows == 0
            || cols == 0
            || x.len() != rows * cols
            || gamma.is_some_and(|g| g.len() != cols)
            || beta.is_some_and(|b| b.len() != cols)
            || !eps.is_finite()
            || eps < 0.0
        {
            return cpu_row_layernorm_f32(x, gamma, beta, rows, cols, eps);
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_row_layernorm_f32(
                    *id, x, gamma, beta, rows, cols, eps,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_row_layernorm_f32(
                    ids[0], x, gamma, beta, rows, cols, eps,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) =
                crate::backends::wgpu_nn::wgpu_row_layernorm_f32(x, gamma, beta, rows, cols, eps)
        {
            return res;
        }

        cpu_row_layernorm_f32(x, gamma, beta, rows, cols, eps)
    }

    /// N8: Row-wise LayerNorm backward returning `(dx, dgamma, dbeta)`.
    #[allow(clippy::too_many_arguments)]
    pub fn row_layernorm_bwd_f32(
        &self,
        dy: &[f32],
        x: &[f32],
        gamma: Option<&[f32]>,
        mean: &[f32],
        rstd: &[f32],
        rows: usize,
        cols: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        cpu_row_layernorm_bwd_f32(dy, x, gamma, mean, rstd, rows, cols)
    }

    /// N8: Row-wise RMSNorm forward returning `(out, rstd)`.
    pub fn row_rmsnorm_f32(
        &self,
        x: &[f32],
        gamma: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        if rows == 0
            || cols == 0
            || x.len() != rows * cols
            || gamma.is_some_and(|g| g.len() != cols)
            || !eps.is_finite()
            || eps < 0.0
        {
            return cpu_row_rmsnorm_f32(x, gamma, rows, cols, eps);
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) =
                    crate::backends::cuda_nn::cuda_row_rmsnorm_f32(*id, x, gamma, rows, cols, eps)
                {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_row_rmsnorm_f32(
                    ids[0], x, gamma, rows, cols, eps,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) =
                crate::backends::wgpu_nn::wgpu_row_rmsnorm_f32(x, gamma, rows, cols, eps)
        {
            return res;
        }

        cpu_row_rmsnorm_f32(x, gamma, rows, cols, eps)
    }

    /// N8: Row-wise RMSNorm backward returning `(dx, dgamma)`.
    pub fn row_rmsnorm_bwd_f32(
        &self,
        dy: &[f32],
        x: &[f32],
        gamma: Option<&[f32]>,
        rstd: &[f32],
        rows: usize,
        cols: usize,
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        cpu_row_rmsnorm_bwd_f32(dy, x, gamma, rstd, rows, cols)
    }

    /// N8: Elementwise GELU forward.
    pub fn gelu_f32(&self, x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_gelu_f32(*id, x) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_gelu_f32(ids[0], x) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(Ok(res)) = crate::backends::wgpu_nn::wgpu_gelu_f32(x)
        {
            return res;
        }

        cpu_gelu_f32(x)
    }

    /// N8: Elementwise GELU backward.
    pub fn gelu_bwd_f32(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_gelu_bwd_f32(*id, dy, x) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_gelu_bwd_f32(ids[0], dy, x) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(Ok(res)) = crate::backends::wgpu_nn::wgpu_gelu_bwd_f32(dy, x)
        {
            return res;
        }

        cpu_gelu_bwd_f32(dy, x)
    }

    /// N8: Elementwise SiLU forward.
    pub fn silu_f32(&self, x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_silu_f32(*id, x) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_silu_f32(ids[0], x) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(Ok(res)) = crate::backends::wgpu_nn::wgpu_silu_f32(x)
        {
            return res;
        }

        cpu_silu_f32(x)
    }

    /// N8: Elementwise SiLU backward.
    pub fn silu_bwd_f32(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_silu_bwd_f32(*id, dy, x) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(Ok(res)) = crate::backends::cuda_nn::cuda_silu_bwd_f32(ids[0], dy, x) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(Ok(res)) = crate::backends::wgpu_nn::wgpu_silu_bwd_f32(dy, x)
        {
            return res;
        }

        cpu_silu_bwd_f32(dy, x)
    }

    /// N8: Causal attention mask applied in-place to `[batch_heads, q_len, k_len]`.
    pub fn causal_mask_f32(
        &self,
        scores: &mut [f32],
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        offset: usize,
    ) -> OchreResult<()> {
        let Some(total) = batch_heads
            .checked_mul(q_len)
            .and_then(|v| v.checked_mul(k_len))
        else {
            return cpu_causal_mask_f32(scores, batch_heads, q_len, k_len, offset);
        };
        if scores.len() != total {
            return cpu_causal_mask_f32(scores, batch_heads, q_len, k_len, offset);
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_causal_mask_f32(
                    *id,
                    scores,
                    batch_heads,
                    q_len,
                    k_len,
                    offset,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_causal_mask_f32(
                    ids[0],
                    scores,
                    batch_heads,
                    q_len,
                    k_len,
                    offset,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_causal_mask_f32(
                scores,
                batch_heads,
                q_len,
                k_len,
                offset,
            )
        {
            return res;
        }

        cpu_causal_mask_f32(scores, batch_heads, q_len, k_len, offset)
    }

    /// N9: Fused Cross-Entropy loss + softmax gradient returning `(mean_loss, d_logits)`.
    pub fn cross_entropy_grad_f32(
        &self,
        logits: &[f32],
        targets: &[usize],
        rows: usize,
        cols: usize,
        label_smoothing: f32,
        ignore_index: Option<usize>,
    ) -> OchreResult<(f32, Vec<f32>)> {
        if rows == 0
            || cols == 0
            || logits.len() != rows * cols
            || targets.len() != rows
            || !label_smoothing.is_finite()
            || !(0.0..=1.0).contains(&label_smoothing)
            || targets
                .iter()
                .any(|&t| Some(t) != ignore_index && t >= cols)
        {
            return cpu_cross_entropy_grad_f32(
                logits,
                targets,
                rows,
                cols,
                label_smoothing,
                ignore_index,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_cross_entropy_grad_f32(
                    *id,
                    logits,
                    targets,
                    rows,
                    cols,
                    label_smoothing,
                    ignore_index,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_cross_entropy_grad_f32(
                    ids[0],
                    logits,
                    targets,
                    rows,
                    cols,
                    label_smoothing,
                    ignore_index,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_cross_entropy_grad_f32(
                logits,
                targets,
                rows,
                cols,
                label_smoothing,
                ignore_index,
            )
        {
            return res;
        }

        cpu_cross_entropy_grad_f32(logits, targets, rows, cols, label_smoothing, ignore_index)
    }

    /// N9: Fused AdamW parameter & moment update step.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_f32(
        &self,
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: usize,
    ) -> OchreResult<()> {
        let n = param.len();
        if grad.len() != n
            || m.len() != n
            || v.len() != n
            || step == 0
            || !lr.is_finite()
            || lr < 0.0
            || !beta1.is_finite()
            || !(0.0..1.0).contains(&beta1)
            || !beta2.is_finite()
            || !(0.0..1.0).contains(&beta2)
            || !eps.is_finite()
            || eps <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return cpu_adamw_step_f32(
                param,
                grad,
                m,
                v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_adamw_step_f32(
                    *id,
                    param,
                    grad,
                    m,
                    v,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_adamw_step_f32(
                    ids[0],
                    param,
                    grad,
                    m,
                    v,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_adamw_step_f32(
                param,
                grad,
                m,
                v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )
        {
            return res;
        }

        cpu_adamw_step_f32(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step)
    }

    /// N10: im2col lowering `[batch, c, h, w] -> (col_mat [(batch*out_h*out_w), (c*kh*kw)], out_h, out_w)`.
    #[allow(clippy::too_many_arguments)]
    pub fn im2col_f32(
        &self,
        input: &[f32],
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dil_h: usize,
        dil_w: usize,
    ) -> OchreResult<(Vec<f32>, usize, usize)> {
        let Ok((out_h, out_w)) =
            validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)
        else {
            return cpu_im2col_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            );
        };
        if input.len() != batch * c * h * w {
            return cpu_im2col_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_im2col_f32(
                    *id, input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h,
                    dil_w, out_h, out_w,
                ) {
                    return res.map(|col| (col, out_h, out_w));
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_im2col_f32(
                    ids[0], input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h,
                    dil_w, out_h, out_w,
                ) {
                    return res.map(|col| (col, out_h, out_w));
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_im2col_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                out_h, out_w,
            )
        {
            return res.map(|col| (col, out_h, out_w));
        }

        cpu_im2col_f32(
            input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        )
    }

    /// N10: col2im scatter-add from `[(batch*out_h*out_w), (c*kh*kw)]` to `[batch, c, h, w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn col2im_f32(
        &self,
        col: &[f32],
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dil_h: usize,
        dil_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> OchreResult<Vec<f32>> {
        let Ok((exp_oh, exp_ow)) =
            validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)
        else {
            return cpu_col2im_f32(
                col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h,
                out_w,
            );
        };
        if out_h != exp_oh || out_w != exp_ow || col.len() != batch * out_h * out_w * c * kh * kw {
            return cpu_col2im_f32(
                col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h,
                out_w,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_col2im_f32(
                    *id, col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h,
                    dil_w, out_h, out_w,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_col2im_f32(
                    ids[0], col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h,
                    dil_w, out_h, out_w,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_col2im_f32(
                col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h,
                out_w,
            )
        {
            return res;
        }

        cpu_col2im_f32(
            col, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h,
            out_w,
        )
    }

    /// N10: MaxPool2D forward returning `(out, argmax_indices, out_h, out_w)`.
    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_f32(
        &self,
        input: &[f32],
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> OchreResult<(Vec<f32>, Vec<usize>, usize, usize)> {
        let Ok((out_h, out_w)) =
            validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1)
        else {
            return cpu_max_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
            );
        };
        if input.len() != batch * c * h * w {
            return cpu_max_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_max_pool2d_f32(
                    *id, input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h,
                    out_w,
                ) {
                    return res.map(|(out, argmax)| (out, argmax, out_h, out_w));
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_max_pool2d_f32(
                    ids[0], input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h,
                    out_w,
                ) {
                    return res.map(|(out, argmax)| (out, argmax, out_h, out_w));
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_max_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w,
            )
        {
            return res.map(|(out, argmax)| (out, argmax, out_h, out_w));
        }

        cpu_max_pool2d_f32(
            input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
        )
    }

    /// N10: MaxPool2D backward scattering `dy` via saved `argmax` indices.
    pub fn max_pool2d_bwd_f32(
        &self,
        dy: &[f32],
        argmax: &[usize],
        total_in_len: usize,
    ) -> Vec<f32> {
        cpu_max_pool2d_bwd_f32(dy, argmax, total_in_len)
    }

    /// N10: AvgPool2D forward returning `(out, out_h, out_w)`.
    #[allow(clippy::too_many_arguments)]
    pub fn avg_pool2d_f32(
        &self,
        input: &[f32],
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> OchreResult<(Vec<f32>, usize, usize)> {
        let Ok((out_h, out_w)) =
            validate_conv_out_dims(h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, 1, 1)
        else {
            return cpu_avg_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
            );
        };
        if input.len() != batch * c * h * w {
            return cpu_avg_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
            );
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_nn::cuda_avg_pool2d_f32(
                    *id, input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h,
                    out_w,
                ) {
                    return res.map(|out| (out, out_h, out_w));
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_nn::cuda_avg_pool2d_f32(
                    ids[0], input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h,
                    out_w,
                ) {
                    return res.map(|out| (out, out_h, out_w));
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_nn::wgpu_avg_pool2d_f32(
                input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w,
            )
        {
            return res.map(|out| (out, out_h, out_w));
        }

        cpu_avg_pool2d_f32(
            input, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
        )
    }

    /// N10: AvgPool2D backward distributing `dy / (kh * kw)` into `[batch, c, h, w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn avg_pool2d_bwd_f32(
        &self,
        dy: &[f32],
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> OchreResult<Vec<f32>> {
        cpu_avg_pool2d_bwd_f32(
            dy, batch, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::get_backend;
    use crate::device::DeviceType;

    #[test]
    fn test_n7_batched_matmul_and_transpose_last2() {
        let b = get_backend(&DeviceType::Cpu);
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let t = b.transpose_last2_f32(2, 2, 3, &a).unwrap();
        assert_eq!(
            t,
            vec![
                1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0
            ]
        );

        let rhs = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0];
        let mm = b
            .batched_matmul_f32(2, 2, 2, 3, &a, &rhs, false, false)
            .unwrap();
        assert_eq!(mm.len(), 8);
        assert!((mm[0] - 4.0).abs() < 1e-5);
        assert!((mm[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_n8_layernorm_and_rmsnorm_finite_diff() {
        let b = get_backend(&DeviceType::Cpu);
        let rows = 3;
        let cols = 4;
        let x = vec![
            0.3, -1.2, 0.7, 2.1, -0.5, 0.9, 1.1, -0.3, 1.5, 0.2, -0.8, 0.4,
        ];
        let gamma = vec![1.2, 0.8, 1.5, 0.9];
        let beta = vec![0.1, -0.2, 0.3, -0.1];
        let dy = vec![
            0.5, -0.3, 0.2, 0.8, -0.4, 0.6, -0.1, 0.9, 0.2, 0.7, -0.5, 0.3,
        ];

        let (out, mean, rstd) = b
            .row_layernorm_f32(&x, Some(&gamma), Some(&beta), rows, cols, 1e-5)
            .unwrap();
        let (dx, dgamma, dbeta) = b
            .row_layernorm_bwd_f32(&dy, &x, Some(&gamma), &mean, &rstd, rows, cols)
            .unwrap();

        let h = 1e-3f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let (op, _, _) = b
                .row_layernorm_f32(&xp, Some(&gamma), Some(&beta), rows, cols, 1e-5)
                .unwrap();
            let (om, _, _) = b
                .row_layernorm_f32(&xm, Some(&gamma), Some(&beta), rows, cols, 1e-5)
                .unwrap();
            let num: f32 = op
                .iter()
                .zip(om.iter())
                .zip(dy.iter())
                .map(|((&p, &m), &g)| (p - m) * g / (2.0 * h))
                .sum();
            assert!(
                (dx[i] - num).abs() < 2e-2,
                "layernorm dx[{i}]: {} vs {num}",
                dx[i]
            );
        }
        for j in 0..cols {
            let mut gp = gamma.clone();
            let mut gm = gamma.clone();
            gp[j] += h;
            gm[j] -= h;
            let (op, _, _) = b
                .row_layernorm_f32(&x, Some(&gp), Some(&beta), rows, cols, 1e-5)
                .unwrap();
            let (om, _, _) = b
                .row_layernorm_f32(&x, Some(&gm), Some(&beta), rows, cols, 1e-5)
                .unwrap();
            let num: f32 = op
                .iter()
                .zip(om.iter())
                .zip(dy.iter())
                .map(|((&p, &m), &g)| (p - m) * g / (2.0 * h))
                .sum();
            assert!((dgamma[j] - num).abs() < 1e-2);
            let exp_db: f32 = (0..rows).map(|r| dy[r * cols + j]).sum();
            assert!((dbeta[j] - exp_db).abs() < 1e-5);
        }
        assert_eq!(out.len(), x.len());

        // RMSNorm finite-diff
        let (_, r_rstd) = b
            .row_rmsnorm_f32(&x, Some(&gamma), rows, cols, 1e-5)
            .unwrap();
        let (r_dx, r_dg) = b
            .row_rmsnorm_bwd_f32(&dy, &x, Some(&gamma), &r_rstd, rows, cols)
            .unwrap();
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let (op, _) = b
                .row_rmsnorm_f32(&xp, Some(&gamma), rows, cols, 1e-5)
                .unwrap();
            let (om, _) = b
                .row_rmsnorm_f32(&xm, Some(&gamma), rows, cols, 1e-5)
                .unwrap();
            let num: f32 = op
                .iter()
                .zip(om.iter())
                .zip(dy.iter())
                .map(|((&p, &m), &g)| (p - m) * g / (2.0 * h))
                .sum();
            assert!(
                (r_dx[i] - num).abs() < 2e-2,
                "rmsnorm dx[{i}]: {} vs {num}",
                r_dx[i]
            );
        }
        assert_eq!(r_dg.len(), cols);
    }

    #[test]
    fn test_n8_activations_and_causal_mask() {
        let b = get_backend(&DeviceType::Cpu);
        let x = vec![-2.0, -0.5, 0.0, 0.7, 2.5];
        let dy = vec![1.0, -0.5, 0.3, 1.2, -0.8];
        let h = 1e-3f32;

        let gelu_dx = b.gelu_bwd_f32(&dy, &x);
        let silu_dx = b.silu_bwd_f32(&dy, &x);
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let gp = b.gelu_f32(&xp)[i];
            let gm = b.gelu_f32(&xm)[i];
            let g_num = dy[i] * (gp - gm) / (2.0 * h);
            assert!((gelu_dx[i] - g_num).abs() < 1e-2);

            let sp = b.silu_f32(&xp)[i];
            let sm = b.silu_f32(&xm)[i];
            let s_num = dy[i] * (sp - sm) / (2.0 * h);
            assert!((silu_dx[i] - s_num).abs() < 1e-2);
        }

        let mut scores = vec![1.0; 2 * 3 * 4];
        b.causal_mask_f32(&mut scores, 2, 3, 4, 1).unwrap();
        // row 0 (pos 1): cols 0,1 unmasked, 2,3 -inf
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 1.0);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[3], f32::NEG_INFINITY);
        // row 2 (pos 3): all 4 cols unmasked
        assert_eq!(scores[2 * 4 + 3], 1.0);
    }

    #[test]
    fn test_n9_cross_entropy_finite_diff_and_adamw() {
        let b = get_backend(&DeviceType::Cpu);
        let logits = vec![1.2, -0.4, 0.3, -1.0, 0.5, 2.1, 0.1, 0.2, 0.3];
        let targets = vec![0, 2, 1];
        let (loss, d_logits) = b
            .cross_entropy_grad_f32(&logits, &targets, 3, 3, 0.1, Some(1))
            .unwrap();
        assert!(loss > 0.0);
        // Ignored row (row 2, target 1 == ignore_index) must have 0 grad
        assert_eq!(&d_logits[6..9], &[0.0, 0.0, 0.0]);

        let h = 1e-3f32;
        for i in 0..6 {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[i] += h;
            lm[i] -= h;
            let (loss_p, _) = b
                .cross_entropy_grad_f32(&lp, &targets, 3, 3, 0.1, Some(1))
                .unwrap();
            let (loss_m, _) = b
                .cross_entropy_grad_f32(&lm, &targets, 3, 3, 0.1, Some(1))
                .unwrap();
            let num = (loss_p - loss_m) / (2.0 * h);
            assert!(
                (d_logits[i] - num).abs() < 1e-2,
                "ce grad[{i}]: {} vs {num}",
                d_logits[i]
            );
        }

        let mut param = vec![1.0, -2.0];
        let grad = vec![0.5, -0.25];
        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];
        b.adamw_step_f32(
            &mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 0.01, 1,
        )
        .unwrap();
        assert!((param[0] - (1.0 * 0.999 - 0.1)).abs() < 1e-4);
        assert!((param[1] - (-2.0 * 0.999 + 0.1)).abs() < 1e-4);
    }

    #[test]
    fn test_n10_im2col_col2im_and_pooling_finite_diff() {
        let b = get_backend(&DeviceType::Cpu);
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let (col, oh, ow) = b
            .im2col_f32(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1)
            .unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(col.len(), 4 * 4);
        // Stride=2 non-overlapping 2x2 patches -> col2im recovers exact input!
        let rec = b
            .col2im_f32(&col, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, oh, ow)
            .unwrap();
        assert_eq!(rec, input);

        // MaxPool2D
        let (mp, argmax, moh, mow) = b
            .max_pool2d_f32(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0)
            .unwrap();
        assert_eq!((moh, mow), (2, 2));
        assert_eq!(mp, vec![6.0, 8.0, 14.0, 16.0]);
        assert_eq!(argmax, vec![5, 7, 13, 15]);
        let mp_dx = b.max_pool2d_bwd_f32(&[1.0, 2.0, 3.0, 4.0], &argmax, input.len());
        assert_eq!(mp_dx[5], 1.0);
        assert_eq!(mp_dx[15], 4.0);
        assert_eq!(mp_dx[0], 0.0);

        // AvgPool2D fwd + bwd parity
        let (ap, aoh, aow) = b
            .avg_pool2d_f32(&input, 1, 1, 4, 4, 2, 2, 1, 1, 1, 1)
            .unwrap();
        let dy = vec![1.0; ap.len()];
        let ap_dx = b
            .avg_pool2d_bwd_f32(&dy, 1, 1, 4, 4, 2, 2, 1, 1, 1, 1, aoh, aow)
            .unwrap();
        let h = 1e-2f32;
        for i in 0..input.len() {
            let mut xp = input.clone();
            let mut xm = input.clone();
            xp[i] += h;
            xm[i] -= h;
            let (op, _, _) = b.avg_pool2d_f32(&xp, 1, 1, 4, 4, 2, 2, 1, 1, 1, 1).unwrap();
            let (om, _, _) = b.avg_pool2d_f32(&xm, 1, 1, 4, 4, 2, 2, 1, 1, 1, 1).unwrap();
            let num: f32 = op
                .iter()
                .zip(om.iter())
                .zip(dy.iter())
                .map(|((&p, &m), &g)| (p - m) * g / (2.0 * h))
                .sum();
            assert!((ap_dx[i] - num).abs() < 1e-2);
        }
    }
}
