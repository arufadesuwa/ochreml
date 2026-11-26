// OchreML Convolutional Neural Network Layers (*≧ω≦*)
// Implements Conv2d (im2col/col2im + SGEMM, groups >= 1), DepthwiseConv2d,
// MaxPool2d, AvgPool2d, GlobalAvgPool2d, BatchNorm2d, Flatten, and ResidualBlock.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::autograd::{AutogradTape, CustomBackward, Op};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::sgemm;
use crate::nn::tensor::{Tensor, compute_strides};
use rayon::prelude::*;
use std::sync::Arc;

const PAR_ELEMS_THRESHOLD: usize = 16_384;

#[inline]
fn quantize_vec(data: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => data,
        DType::BF16 | DType::F16 => TensorStorage::from_f32_slice(&data, dtype).to_f32_vec(),
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_conv_out_dims(
    in_h: usize,
    in_w: usize,
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
            "Kernel, stride, and dilation must be >= 1 ( >_< )".to_string(),
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
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(OchreError::InvalidParameter(format!(
            "Padded input ({padded_h}x{padded_w}) smaller than effective kernel ({eff_kh}x{eff_kw}) ( >_< )"
        )));
    }
    Ok((
        (padded_h - eff_kh) / stride_h + 1,
        (padded_w - eff_kw) / stride_w + 1,
    ))
}

fn validate_nchw_tensor(
    x: &Tensor,
    expected_channels: Option<usize>,
    op_name: &str,
) -> OchreResult<(usize, usize, usize, usize)> {
    if x.shape.len() != 4 {
        return Err(OchreError::InvalidParameter(format!(
            "{op_name} expects 4D NCHW input, got shape {:?} ( >_< )",
            x.shape
        )));
    }
    let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    if n == 0 || c == 0 || h == 0 || w == 0 {
        return Err(OchreError::EmptyData);
    }
    if let Some(exp_c) = expected_channels
        && c != exp_c
    {
        return Err(OchreError::InvalidParameter(format!(
            "{op_name} expected {exp_c} input channels, got {c} ( >_< )"
        )));
    }
    Ok((n, c, h, w))
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
    device: &DeviceType,
) -> OchreResult<()> {
    if matches!(device, DeviceType::Cpu) {
        sgemm(m, n, k, a, b, c, trans_a, trans_b, 1.0, 0.0)
    } else {
        let res = get_backend(device).batched_matmul_f32(1, m, n, k, a, b, trans_a, trans_b)?;
        c.copy_from_slice(&res);
        Ok(())
    }
}

/// 2D spatial convolution over NCHW tensors with `im2col`/`col2im` + SGEMM and `groups >= 1`.
#[derive(Clone, Debug)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub dtype: DType,
    pub device: DeviceType,
}

impl Conv2d {
    /// Creates a `Conv2d` layer with Kaiming uniform initialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if in_channels == 0 || out_channels == 0 || groups == 0 {
            return Err(OchreError::InvalidParameter(
                "Conv2d in_channels, out_channels, and groups must be > 0 ( >_< )".to_string(),
            ));
        }
        if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
            return Err(OchreError::InvalidParameter(format!(
                "Conv2d in_channels ({in_channels}) and out_channels ({out_channels}) must be divisible by groups ({groups}) ( >_< )"
            )));
        }
        if kernel_size.0 == 0
            || kernel_size.1 == 0
            || stride.0 == 0
            || stride.1 == 0
            || dilation.0 == 0
            || dilation.1 == 0
        {
            return Err(OchreError::InvalidParameter(
                "Conv2d kernel_size, stride, and dilation must be >= 1 ( >_< )".to_string(),
            ));
        }
        let in_per_group = in_channels / groups;
        let fan_in = (in_per_group * kernel_size.0 * kernel_size.1) as f32;
        let bound = 1.0f32 / fan_in.sqrt();
        let weight = Tensor::uniform(
            vec![out_channels, in_per_group, kernel_size.0, kernel_size.1],
            -bound,
            bound,
            dtype,
            device.clone(),
            seed,
            true,
        );
        let bias_tensor = if bias {
            Some(Tensor::uniform(
                vec![out_channels],
                -bound,
                bound,
                dtype,
                device.clone(),
                seed.wrapping_add(1),
                true,
            ))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
            device,
        })
    }

    /// Wraps pre-existing `[out_channels, in_channels / groups, kH, kW]` weight and optional `[out_channels]` bias.
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> OchreResult<Self> {
        if weight.shape.len() != 4 || groups == 0 {
            return Err(OchreError::InvalidParameter(format!(
                "Conv2d weight must be 4D and groups > 0, got shape {:?}, groups={groups} ( >_< )",
                weight.shape
            )));
        }
        let out_channels = weight.shape[0];
        let in_per_group = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];
        if out_channels == 0
            || in_per_group == 0
            || kh == 0
            || kw == 0
            || !out_channels.is_multiple_of(groups)
            || stride.0 == 0
            || stride.1 == 0
            || dilation.0 == 0
            || dilation.1 == 0
        {
            return Err(OchreError::InvalidParameter(
                "Invalid Conv2d weight dimensions or hyperparameters ( >_< )".to_string(),
            ));
        }
        if let Some(ref b) = bias
            && b.shape != [out_channels]
        {
            return Err(OchreError::InvalidParameter(format!(
                "Conv2d bias must have shape [{out_channels}], got {:?} ( >_< )",
                b.shape
            )));
        }
        let dtype = weight.dtype;
        let device = weight.device.clone();
        Ok(Self {
            weight,
            bias,
            in_channels: in_per_group * groups,
            out_channels,
            kernel_size: (kh, kw),
            stride,
            padding,
            dilation,
            groups,
            dtype,
            device,
        })
    }

    /// Computes `(out_h, out_w)` for an input spatial size `(in_h, in_w)`.
    pub fn output_shape(&self, in_h: usize, in_w: usize) -> OchreResult<(usize, usize)> {
        compute_conv_out_dims(
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
        )
    }

    /// Returns references to all trainable parameter tensors (`[weight]` or `[weight, bias]`).
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Returns mutable references to all trainable parameter tensors.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Forward pass returning `(output, col_buffers_per_group, out_h, out_w)` for raw/autograd use.
    #[allow(clippy::type_complexity)]
    fn forward_with_cols(
        &self,
        x: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        save_cols: bool,
    ) -> OchreResult<(Vec<f32>, Vec<Vec<f32>>, usize, usize)> {
        let expected_in = batch * self.in_channels * in_h * in_w;
        if x.len() != expected_in {
            return Err(OchreError::InvalidParameter(format!(
                "Conv2d expected {expected_in} input elements, got {} ( >_< )",
                x.len()
            )));
        }
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let spatial_out = out_h * out_w;
        let total_rows = batch * spatial_out;
        let (kh, kw) = self.kernel_size;
        let in_per_g = self.in_channels / self.groups;
        let out_per_g = self.out_channels / self.groups;
        let patch_per_g = in_per_g * kh * kw;
        let backend = get_backend(&self.device);

        let mut out = vec![0.0f32; batch * self.out_channels * spatial_out];
        let mut saved_cols = if save_cols {
            Vec::with_capacity(self.groups)
        } else {
            Vec::new()
        };

        let mut y_mat = vec![0.0f32; total_rows * out_per_g];
        let mut x_group_buf = if self.groups > 1 {
            vec![0.0f32; batch * in_per_g * in_h * in_w]
        } else {
            Vec::new()
        };

        for g in 0..self.groups {
            let x_slice: &[f32] = if self.groups == 1 {
                x
            } else {
                let hw = in_h * in_w;
                let c_start = g * in_per_g;
                for b in 0..batch {
                    let src_off = (b * self.in_channels + c_start) * hw;
                    let dst_off = b * in_per_g * hw;
                    x_group_buf[dst_off..dst_off + in_per_g * hw]
                        .copy_from_slice(&x[src_off..src_off + in_per_g * hw]);
                }
                &x_group_buf
            };

            let (col_g, _, _) = backend.im2col_f32(
                x_slice,
                batch,
                in_per_g,
                in_h,
                in_w,
                kh,
                kw,
                self.stride.0,
                self.stride.1,
                self.padding.0,
                self.padding.1,
                self.dilation.0,
                self.dilation.1,
            )?;

            let w_off = g * out_per_g * patch_per_g;
            let w_g = &self.weight.data[w_off..w_off + out_per_g * patch_per_g];
            gemm_dispatch(
                total_rows,
                out_per_g,
                patch_per_g,
                &col_g,
                w_g,
                &mut y_mat,
                false,
                true,
                &self.device,
            )?;

            let oc_start = g * out_per_g;
            for b in 0..batch {
                for oc_local in 0..out_per_g {
                    let oc = oc_start + oc_local;
                    let bias_val = self.bias.as_ref().map_or(0.0f32, |bias| bias.data[oc]);
                    let dst_base = (b * self.out_channels + oc) * spatial_out;
                    for s in 0..spatial_out {
                        out[dst_base + s] =
                            y_mat[(b * spatial_out + s) * out_per_g + oc_local] + bias_val;
                    }
                }
            }

            if save_cols {
                saved_cols.push(col_g);
            }
        }

        Ok((out, saved_cols, out_h, out_w))
    }

    /// Raw slice forward pass writing `[batch, out_channels, out_h, out_w]` into `out`.
    pub fn forward_raw(
        &self,
        x: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        out: &mut [f32],
    ) -> OchreResult<()> {
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let expected_out = batch * self.out_channels * out_h * out_w;
        if out.len() != expected_out {
            return Err(OchreError::InvalidParameter(format!(
                "Conv2d::forward_raw output buffer len {} != expected {expected_out} ( >_< )",
                out.len()
            )));
        }
        let (res, _, _, _) = self.forward_with_cols(x, batch, in_h, in_w, false)?;
        out.copy_from_slice(&res);
        Ok(())
    }

    /// Raw slice backward pass writing `dx` (optional), `dw`, and `db` (optional).
    #[allow(clippy::too_many_arguments)]
    pub fn backward_raw(
        &self,
        x: &[f32],
        dy: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        dx: Option<&mut [f32]>,
        dw: &mut [f32],
        db: Option<&mut [f32]>,
    ) -> OchreResult<()> {
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let expected_in = batch * self.in_channels * in_h * in_w;
        let expected_out = batch * self.out_channels * out_h * out_w;
        if x.len() != expected_in || dy.len() != expected_out || dw.len() != self.weight.data.len()
        {
            return Err(OchreError::InvalidParameter(
                "Conv2d::backward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }

        let (kh, kw) = self.kernel_size;
        let in_per_g = self.in_channels / self.groups;
        let patch_per_g = in_per_g * kh * kw;
        let backend = get_backend(&self.device);

        let mut cols = Vec::with_capacity(self.groups);
        let mut x_group_buf = if self.groups > 1 {
            vec![0.0f32; batch * in_per_g * in_h * in_w]
        } else {
            Vec::new()
        };
        for g in 0..self.groups {
            let x_slice: &[f32] = if self.groups == 1 {
                x
            } else {
                let hw = in_h * in_w;
                let c_start = g * in_per_g;
                for b in 0..batch {
                    let src_off = (b * self.in_channels + c_start) * hw;
                    let dst_off = b * in_per_g * hw;
                    x_group_buf[dst_off..dst_off + in_per_g * hw]
                        .copy_from_slice(&x[src_off..src_off + in_per_g * hw]);
                }
                &x_group_buf
            };
            let (col_g, _, _) = backend.im2col_f32(
                x_slice,
                batch,
                in_per_g,
                in_h,
                in_w,
                kh,
                kw,
                self.stride.0,
                self.stride.1,
                self.padding.0,
                self.padding.1,
                self.dilation.0,
                self.dilation.1,
            )?;
            cols.push(col_g);
        }

        let (dx_vec, dw_vec, db_vec) = conv2d_backward_core(
            dy,
            &cols,
            &self.weight.data,
            batch,
            self.in_channels,
            in_h,
            in_w,
            self.out_channels,
            kh,
            kw,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_h,
            out_w,
            &self.device,
            dx.is_some(),
            true,
            db.is_some() && self.bias.is_some(),
            patch_per_g,
        )?;

        if let Some(dx_buf) = dx {
            if dx_buf.len() != expected_in {
                return Err(OchreError::InvalidParameter(
                    "Conv2d::backward_raw dx length mismatch ( >_< )".to_string(),
                ));
            }
            dx_buf.copy_from_slice(&dx_vec);
        }
        dw.copy_from_slice(&dw_vec);
        if let (Some(db_buf), Some(_)) = (db, &self.bias) {
            if db_buf.len() != self.out_channels {
                return Err(OchreError::InvalidParameter(
                    "Conv2d::backward_raw db length mismatch ( >_< )".to_string(),
                ));
            }
            db_buf.copy_from_slice(&db_vec);
        }
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, _, in_h, in_w) = validate_nchw_tensor(x, Some(self.in_channels), "Conv2d")?;
        let need_grad = AutogradTape::is_grad_enabled()
            && (x.requires_grad
                || self.weight.requires_grad
                || self.bias.as_ref().is_some_and(|b| b.requires_grad));

        let (raw_out, cols, out_h, out_w) = self.forward_with_cols(
            &x.data,
            batch,
            in_h,
            in_w,
            need_grad && (x.requires_grad || self.weight.requires_grad),
        )?;
        let out_data = quantize_vec(raw_out, self.dtype);
        let out_shape = vec![batch, self.out_channels, out_h, out_w];

        let node_id = if need_grad {
            let x_id = if x.requires_grad { x.node_id } else { None };
            let w_id = if self.weight.requires_grad {
                self.weight.node_id
            } else {
                None
            };
            let b_id = self
                .bias
                .as_ref()
                .and_then(|b| if b.requires_grad { b.node_id } else { None });
            let w_data = if x_id.is_some() {
                self.weight.data.clone()
            } else {
                Vec::new()
            };
            let in_c = self.in_channels;
            let out_c = self.out_channels;
            let (kh, kw) = self.kernel_size;
            let stride = self.stride;
            let padding = self.padding;
            let dilation = self.dilation;
            let groups = self.groups;
            let patch_per_g = (in_c / groups) * kh * kw;
            let dev = self.device.clone();

            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                let (dx, dw, db) = conv2d_backward_core(
                    dy,
                    &cols,
                    &w_data,
                    batch,
                    in_c,
                    in_h,
                    in_w,
                    out_c,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    groups,
                    out_h,
                    out_w,
                    &dev,
                    x_id.is_some(),
                    w_id.is_some(),
                    b_id.is_some(),
                    patch_per_g,
                )?;
                let mut res = Vec::with_capacity(3);
                if let Some(id) = x_id {
                    res.push((id, dx));
                }
                if let Some(id) = w_id {
                    res.push((id, dw));
                }
                if let Some(id) = b_id {
                    res.push((id, db));
                }
                Ok(res)
            }));

            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id, w_id, b_id],
                backward: bwd,
            }))
        } else {
            None
        };

        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv2d_backward_core(
    dy: &[f32],
    cols: &[Vec<f32>],
    weight_data: &[f32],
    batch: usize,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    out_h: usize,
    out_w: usize,
    device: &DeviceType,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
    patch_per_g: usize,
) -> OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let spatial_out = out_h * out_w;
    let total_rows = batch * spatial_out;
    let in_per_g = in_c / groups;
    let out_per_g = out_c / groups;
    let backend = get_backend(device);

    let db = if need_db {
        let mut acc = vec![0.0f32; out_c];
        for b in 0..batch {
            for oc in 0..out_c {
                let src = &dy[(b * out_c + oc) * spatial_out..(b * out_c + oc + 1) * spatial_out];
                let sum: f32 = src.iter().copied().sum();
                acc[oc] += sum;
            }
        }
        acc
    } else {
        Vec::new()
    };

    let mut dx = if need_dx {
        vec![0.0f32; batch * in_c * in_h * in_w]
    } else {
        Vec::new()
    };
    let mut dw = if need_dw {
        vec![0.0f32; out_c * patch_per_g]
    } else {
        Vec::new()
    };

    if !need_dx && !need_dw {
        return Ok((dx, dw, db));
    }

    let mut dy_mat = vec![0.0f32; total_rows * out_per_g];
    let mut dcol_g = if need_dx {
        vec![0.0f32; total_rows * patch_per_g]
    } else {
        Vec::new()
    };

    #[allow(clippy::needless_range_loop)]
    for g in 0..groups {
        let oc_start = g * out_per_g;
        for b in 0..batch {
            for oc_local in 0..out_per_g {
                let oc = oc_start + oc_local;
                let src_base = (b * out_c + oc) * spatial_out;
                for s in 0..spatial_out {
                    dy_mat[(b * spatial_out + s) * out_per_g + oc_local] = dy[src_base + s];
                }
            }
        }

        if need_dw {
            let dw_off = g * out_per_g * patch_per_g;
            gemm_dispatch(
                out_per_g,
                patch_per_g,
                total_rows,
                &dy_mat,
                &cols[g],
                &mut dw[dw_off..dw_off + out_per_g * patch_per_g],
                true,
                false,
                device,
            )?;
        }

        if need_dx {
            let w_off = g * out_per_g * patch_per_g;
            let w_g = &weight_data[w_off..w_off + out_per_g * patch_per_g];
            gemm_dispatch(
                total_rows,
                patch_per_g,
                out_per_g,
                &dy_mat,
                w_g,
                &mut dcol_g,
                false,
                false,
                device,
            )?;
            let dx_g = backend.col2im_f32(
                &dcol_g, batch, in_per_g, in_h, in_w, kh, kw, stride.0, stride.1, padding.0,
                padding.1, dilation.0, dilation.1, out_h, out_w,
            )?;
            if groups == 1 {
                dx = dx_g;
            } else {
                let hw = in_h * in_w;
                let c_start = g * in_per_g;
                for b in 0..batch {
                    let src_off = b * in_per_g * hw;
                    let dst_off = (b * in_c + c_start) * hw;
                    dx[dst_off..dst_off + in_per_g * hw]
                        .copy_from_slice(&dx_g[src_off..src_off + in_per_g * hw]);
                }
            }
        }
    }

    Ok((dx, dw, db))
}

/// Specialized fast depthwise spatial convolution (`groups == in_channels`,
/// `out_channels = in_channels * depth_multiplier`) without im2col allocations.
#[derive(Clone, Debug)]
pub struct DepthwiseConv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub depth_multiplier: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub dtype: DType,
    pub device: DeviceType,
}

impl DepthwiseConv2d {
    /// Creates a `DepthwiseConv2d` layer with Kaiming uniform initialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        depth_multiplier: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if in_channels == 0 || depth_multiplier == 0 {
            return Err(OchreError::InvalidParameter(
                "DepthwiseConv2d in_channels and depth_multiplier must be > 0 ( >_< )".to_string(),
            ));
        }
        if kernel_size.0 == 0
            || kernel_size.1 == 0
            || stride.0 == 0
            || stride.1 == 0
            || dilation.0 == 0
            || dilation.1 == 0
        {
            return Err(OchreError::InvalidParameter(
                "DepthwiseConv2d kernel_size, stride, and dilation must be >= 1 ( >_< )"
                    .to_string(),
            ));
        }
        let out_channels = in_channels * depth_multiplier;
        let fan_in = (kernel_size.0 * kernel_size.1) as f32;
        let bound = 1.0f32 / fan_in.sqrt();
        let weight = Tensor::uniform(
            vec![out_channels, 1, kernel_size.0, kernel_size.1],
            -bound,
            bound,
            dtype,
            device.clone(),
            seed,
            true,
        );
        let bias_tensor = if bias {
            Some(Tensor::uniform(
                vec![out_channels],
                -bound,
                bound,
                dtype,
                device.clone(),
                seed.wrapping_add(1),
                true,
            ))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: bias_tensor,
            in_channels,
            depth_multiplier,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            dtype,
            device,
        })
    }

    /// Computes `(out_h, out_w)` for an input spatial size `(in_h, in_w)`.
    pub fn output_shape(&self, in_h: usize, in_w: usize) -> OchreResult<(usize, usize)> {
        compute_conv_out_dims(
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
        )
    }

    /// Returns references to trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Returns mutable references to trainable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Direct depthwise forward pass without im2col allocation.
    pub fn forward_raw(
        &self,
        x: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        out: &mut [f32],
    ) -> OchreResult<()> {
        let expected_in = batch * self.in_channels * in_h * in_w;
        if x.len() != expected_in {
            return Err(OchreError::InvalidParameter(format!(
                "DepthwiseConv2d expected {expected_in} input elements, got {} ( >_< )",
                x.len()
            )));
        }
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let out_plane = out_h * out_w;
        let expected_out = batch * self.out_channels * out_plane;
        if out.len() != expected_out {
            return Err(OchreError::InvalidParameter(format!(
                "DepthwiseConv2d output buffer len {} != expected {expected_out} ( >_< )",
                out.len()
            )));
        }
        let (kh, kw) = self.kernel_size;
        let k_area = kh * kw;
        let in_plane = in_h * in_w;
        let w_data = &self.weight.data;
        let b_data = self.bias.as_ref().map(|b| b.data.as_slice());

        let compute_oc_plane = |boc: usize, dst_plane: &mut [f32]| {
            let b = boc / self.out_channels;
            let oc = boc % self.out_channels;
            let ic = oc / self.depth_multiplier;
            let x_plane = &x[(b * self.in_channels + ic) * in_plane..];
            let w_k = &w_data[oc * k_area..(oc + 1) * k_area];
            let bias_v = b_data.map_or(0.0f32, |bd| bd[oc]);

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_v;
                    for ky in 0..kh {
                        let iy = (oh * self.stride.0 + ky * self.dilation.0) as isize
                            - self.padding.0 as isize;
                        if iy < 0 || iy >= in_h as isize {
                            continue;
                        }
                        let x_row = (iy as usize) * in_w;
                        let w_row = ky * kw;
                        for kx in 0..kw {
                            let ix = (ow * self.stride.1 + kx * self.dilation.1) as isize
                                - self.padding.1 as isize;
                            if ix >= 0 && ix < in_w as isize {
                                acc += x_plane[x_row + (ix as usize)] * w_k[w_row + kx];
                            }
                        }
                    }
                    dst_plane[oh * out_w + ow] = acc;
                }
            }
        };

        if expected_out >= PAR_ELEMS_THRESHOLD && batch * self.out_channels >= 4 {
            out.par_chunks_mut(out_plane)
                .enumerate()
                .for_each(|(boc, dst_plane)| compute_oc_plane(boc, dst_plane));
        } else {
            for boc in 0..(batch * self.out_channels) {
                compute_oc_plane(boc, &mut out[boc * out_plane..(boc + 1) * out_plane]);
            }
        }
        Ok(())
    }

    /// Direct depthwise backward pass writing `dx` (optional), `dw`, and `db` (optional).
    #[allow(clippy::too_many_arguments)]
    pub fn backward_raw(
        &self,
        x: &[f32],
        dy: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        dx: Option<&mut [f32]>,
        dw: &mut [f32],
        db: Option<&mut [f32]>,
    ) -> OchreResult<()> {
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let expected_in = batch * self.in_channels * in_h * in_w;
        let expected_out = batch * self.out_channels * out_h * out_w;
        if x.len() != expected_in || dy.len() != expected_out || dw.len() != self.weight.data.len()
        {
            return Err(OchreError::InvalidParameter(
                "DepthwiseConv2d::backward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        let need_dx = dx.is_some();
        let need_db = db.is_some() && self.bias.is_some();
        let (dx_vec, dw_vec, db_vec) = depthwise_backward_core(
            x,
            dy,
            &self.weight.data,
            batch,
            self.in_channels,
            self.depth_multiplier,
            in_h,
            in_w,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_h,
            out_w,
            need_dx,
            true,
            need_db,
        );
        if let Some(dx_buf) = dx {
            if dx_buf.len() != expected_in {
                return Err(OchreError::InvalidParameter(
                    "DepthwiseConv2d::backward_raw dx length mismatch ( >_< )".to_string(),
                ));
            }
            dx_buf.copy_from_slice(&dx_vec);
        }
        dw.copy_from_slice(&dw_vec);
        if let (Some(db_buf), Some(_)) = (db, &self.bias) {
            if db_buf.len() != self.out_channels {
                return Err(OchreError::InvalidParameter(
                    "DepthwiseConv2d::backward_raw db length mismatch ( >_< )".to_string(),
                ));
            }
            db_buf.copy_from_slice(&db_vec);
        }
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, _, in_h, in_w) =
            validate_nchw_tensor(x, Some(self.in_channels), "DepthwiseConv2d")?;
        let (out_h, out_w) = self.output_shape(in_h, in_w)?;
        let mut raw_out = vec![0.0f32; batch * self.out_channels * out_h * out_w];
        self.forward_raw(&x.data, batch, in_h, in_w, &mut raw_out)?;
        let out_data = quantize_vec(raw_out, self.dtype);
        let out_shape = vec![batch, self.out_channels, out_h, out_w];

        let need_grad = AutogradTape::is_grad_enabled()
            && (x.requires_grad
                || self.weight.requires_grad
                || self.bias.as_ref().is_some_and(|b| b.requires_grad));

        let node_id = if need_grad {
            let x_id = if x.requires_grad { x.node_id } else { None };
            let w_id = if self.weight.requires_grad {
                self.weight.node_id
            } else {
                None
            };
            let b_id = self
                .bias
                .as_ref()
                .and_then(|b| if b.requires_grad { b.node_id } else { None });
            let x_saved = if w_id.is_some() {
                x.data.clone()
            } else {
                Vec::new()
            };
            let w_saved = if x_id.is_some() {
                self.weight.data.clone()
            } else {
                Vec::new()
            };
            let in_c = self.in_channels;
            let mult = self.depth_multiplier;
            let kernel_size = self.kernel_size;
            let stride = self.stride;
            let padding = self.padding;
            let dilation = self.dilation;

            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                let (dx, dw, db) = depthwise_backward_core(
                    &x_saved,
                    dy,
                    &w_saved,
                    batch,
                    in_c,
                    mult,
                    in_h,
                    in_w,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    out_h,
                    out_w,
                    x_id.is_some(),
                    w_id.is_some(),
                    b_id.is_some(),
                );
                let mut res = Vec::with_capacity(3);
                if let Some(id) = x_id {
                    res.push((id, dx));
                }
                if let Some(id) = w_id {
                    res.push((id, dw));
                }
                if let Some(id) = b_id {
                    res.push((id, db));
                }
                Ok(res)
            }));

            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id, w_id, b_id],
                backward: bwd,
            }))
        } else {
            None
        };

        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn depthwise_backward_core(
    x: &[f32],
    dy: &[f32],
    w: &[f32],
    batch: usize,
    in_c: usize,
    depth_multiplier: usize,
    in_h: usize,
    in_w: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    out_h: usize,
    out_w: usize,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let out_c = in_c * depth_multiplier;
    let (kh, kw) = kernel_size;
    let k_area = kh * kw;
    let in_plane = in_h * in_w;
    let out_plane = out_h * out_w;

    let db = if need_db {
        let mut acc = vec![0.0f32; out_c];
        for b in 0..batch {
            for oc in 0..out_c {
                let s: f32 = dy[(b * out_c + oc) * out_plane..(b * out_c + oc + 1) * out_plane]
                    .iter()
                    .copied()
                    .sum();
                acc[oc] += s;
            }
        }
        acc
    } else {
        Vec::new()
    };

    let mut dw = if need_dw {
        vec![0.0f32; out_c * k_area]
    } else {
        Vec::new()
    };

    if need_dw {
        let compute_dw_oc = |oc: usize, dw_k: &mut [f32]| {
            let ic = oc / depth_multiplier;
            for b in 0..batch {
                let x_plane = &x[(b * in_c + ic) * in_plane..];
                let dy_plane = &dy[(b * out_c + oc) * out_plane..];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = dy_plane[oh * out_w + ow];
                        for ky in 0..kh {
                            let iy =
                                (oh * stride.0 + ky * dilation.0) as isize - padding.0 as isize;
                            if iy < 0 || iy >= in_h as isize {
                                continue;
                            }
                            let x_row = (iy as usize) * in_w;
                            let w_row = ky * kw;
                            for kx in 0..kw {
                                let ix =
                                    (ow * stride.1 + kx * dilation.1) as isize - padding.1 as isize;
                                if ix >= 0 && ix < in_w as isize {
                                    dw_k[w_row + kx] += g * x_plane[x_row + (ix as usize)];
                                }
                            }
                        }
                    }
                }
            }
        };

        if batch * out_c * out_plane >= PAR_ELEMS_THRESHOLD && out_c >= 4 {
            dw.par_chunks_mut(k_area)
                .enumerate()
                .for_each(|(oc, dw_k)| compute_dw_oc(oc, dw_k));
        } else {
            for oc in 0..out_c {
                compute_dw_oc(oc, &mut dw[oc * k_area..(oc + 1) * k_area]);
            }
        }
    }

    let mut dx = if need_dx {
        vec![0.0f32; batch * in_c * in_plane]
    } else {
        Vec::new()
    };

    if need_dx {
        let compute_dx_bic = |bic: usize, dx_plane: &mut [f32]| {
            let b = bic / in_c;
            let ic = bic % in_c;
            for m in 0..depth_multiplier {
                let oc = ic * depth_multiplier + m;
                let dy_plane = &dy[(b * out_c + oc) * out_plane..];
                let w_k = &w[oc * k_area..(oc + 1) * k_area];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = dy_plane[oh * out_w + ow];
                        for ky in 0..kh {
                            let iy =
                                (oh * stride.0 + ky * dilation.0) as isize - padding.0 as isize;
                            if iy < 0 || iy >= in_h as isize {
                                continue;
                            }
                            let dx_row = (iy as usize) * in_w;
                            let w_row = ky * kw;
                            for kx in 0..kw {
                                let ix =
                                    (ow * stride.1 + kx * dilation.1) as isize - padding.1 as isize;
                                if ix >= 0 && ix < in_w as isize {
                                    dx_plane[dx_row + (ix as usize)] += g * w_k[w_row + kx];
                                }
                            }
                        }
                    }
                }
            }
        };

        if batch * in_c * in_plane >= PAR_ELEMS_THRESHOLD && batch * in_c >= 4 {
            dx.par_chunks_mut(in_plane)
                .enumerate()
                .for_each(|(bic, dx_plane)| compute_dx_bic(bic, dx_plane));
        } else {
            for bic in 0..(batch * in_c) {
                compute_dx_bic(bic, &mut dx[bic * in_plane..(bic + 1) * in_plane]);
            }
        }
    }

    (dx, dw, db)
}

/// 2D Max Pooling layer over NCHW tensors backed by `Backend::max_pool2d_f32`.
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    /// Creates a `MaxPool2d` layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> OchreResult<Self> {
        if kernel_size.0 == 0 || kernel_size.1 == 0 || stride.0 == 0 || stride.1 == 0 {
            return Err(OchreError::InvalidParameter(
                "MaxPool2d kernel_size and stride must be >= 1 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            kernel_size,
            stride,
            padding,
        })
    }

    /// Computes `(out_h, out_w)` for an input spatial size `(in_h, in_w)`.
    pub fn output_shape(&self, in_h: usize, in_w: usize) -> OchreResult<(usize, usize)> {
        compute_conv_out_dims(
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            1,
            1,
        )
    }

    /// Raw slice forward pass returning `argmax` indices and writing pooled values into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_raw(
        &self,
        x: &[f32],
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out: &mut [f32],
        device: &DeviceType,
    ) -> OchreResult<Vec<usize>> {
        let (res, argmax, _, _) = get_backend(device).max_pool2d_f32(
            x,
            batch,
            channels,
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
        )?;
        if out.len() != res.len() {
            return Err(OchreError::InvalidParameter(format!(
                "MaxPool2d::forward_raw output buffer len {} != expected {} ( >_< )",
                out.len(),
                res.len()
            )));
        }
        out.copy_from_slice(&res);
        Ok(argmax)
    }

    /// Raw slice backward pass scattering `dy` via saved `argmax` indices.
    pub fn backward_raw(
        &self,
        dy: &[f32],
        argmax: &[usize],
        dx: &mut [f32],
        device: &DeviceType,
    ) -> OchreResult<()> {
        if dy.len() != argmax.len() {
            return Err(OchreError::InvalidParameter(
                "MaxPool2d::backward_raw dy and argmax length mismatch ( >_< )".to_string(),
            ));
        }
        let res = get_backend(device).max_pool2d_bwd_f32(dy, argmax, dx.len());
        dx.copy_from_slice(&res);
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, c, h, w) = validate_nchw_tensor(x, None, "MaxPool2d")?;
        let (raw_out, argmax, out_h, out_w) = get_backend(&x.device).max_pool2d_f32(
            &x.data,
            batch,
            c,
            h,
            w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
        )?;
        let out_data = quantize_vec(raw_out, x.dtype);
        let out_shape = vec![batch, c, out_h, out_w];
        let need_grad = AutogradTape::is_grad_enabled() && x.requires_grad;
        let node_id = if need_grad {
            let x_id = x.node_id;
            let total_in = x.data.len();
            let dev = x.device.clone();
            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                if let Some(id) = x_id {
                    let dx = get_backend(&dev).max_pool2d_bwd_f32(dy, &argmax, total_in);
                    Ok(vec![(id, dx)])
                } else {
                    Ok(Vec::new())
                }
            }));
            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id],
                backward: bwd,
            }))
        } else {
            None
        };
        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

/// 2D Average Pooling layer over NCHW tensors backed by `Backend::avg_pool2d_f32`.
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    /// Creates an `AvgPool2d` layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> OchreResult<Self> {
        if kernel_size.0 == 0 || kernel_size.1 == 0 || stride.0 == 0 || stride.1 == 0 {
            return Err(OchreError::InvalidParameter(
                "AvgPool2d kernel_size and stride must be >= 1 ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            kernel_size,
            stride,
            padding,
        })
    }

    /// Computes `(out_h, out_w)` for an input spatial size `(in_h, in_w)`.
    pub fn output_shape(&self, in_h: usize, in_w: usize) -> OchreResult<(usize, usize)> {
        compute_conv_out_dims(
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            1,
            1,
        )
    }

    /// Raw slice forward pass writing pooled values into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_raw(
        &self,
        x: &[f32],
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out: &mut [f32],
        device: &DeviceType,
    ) -> OchreResult<()> {
        let (res, _, _) = get_backend(device).avg_pool2d_f32(
            x,
            batch,
            channels,
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
        )?;
        if out.len() != res.len() {
            return Err(OchreError::InvalidParameter(format!(
                "AvgPool2d::forward_raw output buffer len {} != expected {} ( >_< )",
                out.len(),
                res.len()
            )));
        }
        out.copy_from_slice(&res);
        Ok(())
    }

    /// Raw slice backward pass writing `dx`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_raw(
        &self,
        dy: &[f32],
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        dx: &mut [f32],
        device: &DeviceType,
    ) -> OchreResult<()> {
        let res = get_backend(device).avg_pool2d_bwd_f32(
            dy,
            batch,
            channels,
            in_h,
            in_w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            out_h,
            out_w,
        )?;
        if dx.len() != res.len() {
            return Err(OchreError::InvalidParameter(
                "AvgPool2d::backward_raw dx length mismatch ( >_< )".to_string(),
            ));
        }
        dx.copy_from_slice(&res);
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, c, h, w) = validate_nchw_tensor(x, None, "AvgPool2d")?;
        let (raw_out, out_h, out_w) = get_backend(&x.device).avg_pool2d_f32(
            &x.data,
            batch,
            c,
            h,
            w,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
        )?;
        let out_data = quantize_vec(raw_out, x.dtype);
        let out_shape = vec![batch, c, out_h, out_w];
        let need_grad = AutogradTape::is_grad_enabled() && x.requires_grad;
        let node_id = if need_grad {
            let x_id = x.node_id;
            let (kh, kw) = self.kernel_size;
            let (sh, sw) = self.stride;
            let (ph, pw) = self.padding;
            let dev = x.device.clone();
            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                if let Some(id) = x_id {
                    let dx = get_backend(&dev).avg_pool2d_bwd_f32(
                        dy, batch, c, h, w, kh, kw, sh, sw, ph, pw, out_h, out_w,
                    )?;
                    Ok(vec![(id, dx)])
                } else {
                    Ok(Vec::new())
                }
            }));
            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id],
                backward: bwd,
            }))
        } else {
            None
        };
        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

/// Global Average Pooling over spatial `(H, W)` dimensions: `[N, C, H, W] -> [N, C, 1, 1]` or `[N, C]`.
#[derive(Clone, Copy, Debug)]
pub struct GlobalAvgPool2d {
    pub keep_dims: bool,
}

impl Default for GlobalAvgPool2d {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalAvgPool2d {
    /// Creates a `GlobalAvgPool2d` returning `[N, C, 1, 1]` by default.
    pub fn new() -> Self {
        Self { keep_dims: true }
    }

    /// Creates a `GlobalAvgPool2d` with configurable `keep_dims` (`[N, C, 1, 1]` vs `[N, C]`).
    pub fn with_keep_dims(keep_dims: bool) -> Self {
        Self { keep_dims }
    }

    /// Raw slice forward pass averaging each `(n, c)` spatial plane into `out` (`[N * C]`).
    pub fn forward_raw(
        &self,
        x: &[f32],
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out: &mut [f32],
    ) -> OchreResult<()> {
        let plane = in_h * in_w;
        if plane == 0 || batch == 0 || channels == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.len() != batch * channels * plane || out.len() != batch * channels {
            return Err(OchreError::InvalidParameter(
                "GlobalAvgPool2d::forward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        let inv = 1.0f64 / (plane as f64);
        for (bc, dst) in out.iter_mut().enumerate() {
            let slice = &x[bc * plane..(bc + 1) * plane];
            let sum: f64 = slice.iter().map(|&v| v as f64).sum();
            *dst = (sum * inv) as f32;
        }
        Ok(())
    }

    /// Raw slice backward pass distributing `dy[n, c] / (H * W)` into `dx`.
    pub fn backward_raw(
        &self,
        dy: &[f32],
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        dx: &mut [f32],
    ) -> OchreResult<()> {
        let plane = in_h * in_w;
        if plane == 0 || batch == 0 || channels == 0 {
            return Err(OchreError::EmptyData);
        }
        if dy.len() != batch * channels || dx.len() != batch * channels * plane {
            return Err(OchreError::InvalidParameter(
                "GlobalAvgPool2d::backward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        let inv = 1.0f32 / (plane as f32);
        for (bc, &g) in dy.iter().enumerate() {
            dx[bc * plane..(bc + 1) * plane].fill(g * inv);
        }
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, c, h, w) = validate_nchw_tensor(x, None, "GlobalAvgPool2d")?;
        let mut raw_out = vec![0.0f32; batch * c];
        self.forward_raw(&x.data, batch, c, h, w, &mut raw_out)?;
        let out_data = quantize_vec(raw_out, x.dtype);
        let out_shape = if self.keep_dims {
            vec![batch, c, 1, 1]
        } else {
            vec![batch, c]
        };

        let need_grad = AutogradTape::is_grad_enabled() && x.requires_grad;
        let node_id = if need_grad {
            let x_id = x.node_id;
            let plane = h * w;
            let inv = 1.0f32 / (plane as f32);
            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                if let Some(id) = x_id {
                    let mut dx = vec![0.0f32; batch * c * plane];
                    for (bc, &g) in dy.iter().enumerate() {
                        dx[bc * plane..(bc + 1) * plane].fill(g * inv);
                    }
                    Ok(vec![(id, dx)])
                } else {
                    Ok(Vec::new())
                }
            }));
            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id],
                backward: bwd,
            }))
        } else {
            None
        };

        Ok(Tensor {
            strides: compute_strides(&out_shape),
            shape: out_shape,
            data: out_data,
            dtype: x.dtype,
            device: x.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

/// Per-channel 2D Batch Normalization across `(N, H, W)` with running statistics and train/eval modes.
#[derive(Clone, Debug)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub training: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub dtype: DType,
    pub device: DeviceType,
}

impl BatchNorm2d {
    /// Creates a `BatchNorm2d` layer (`weight=1`, `bias=0`, `running_mean=0`, `running_var=1`).
    pub fn new(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        dtype: DType,
        device: DeviceType,
    ) -> OchreResult<Self> {
        if num_features == 0 {
            return Err(OchreError::InvalidParameter(
                "BatchNorm2d num_features must be > 0 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite()
            || eps < 0.0
            || !momentum.is_finite()
            || !(0.0..=1.0).contains(&momentum)
        {
            return Err(OchreError::InvalidParameter(format!(
                "BatchNorm2d invalid eps={eps} or momentum={momentum} ( >_< )"
            )));
        }
        let weight = if affine {
            Some(Tensor::ones(vec![num_features], dtype, device.clone()).with_requires_grad(true))
        } else {
            None
        };
        let bias = if affine {
            Some(Tensor::zeros(vec![num_features], dtype, device.clone()).with_requires_grad(true))
        } else {
            None
        };
        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            training: true,
            weight,
            bias,
            running_mean: vec![0.0f32; num_features],
            running_var: vec![1.0f32; num_features],
            dtype,
            device,
        })
    }

    /// Sets training mode (`training = true`).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Sets evaluation mode (`training = false`).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns references to trainable affine parameters (`gamma`, `beta`).
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(w);
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Returns mutable references to trainable affine parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut w) = self.weight {
            params.push(w);
        }
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Raw forward pass returning `(batch_mean, batch_inv_std)` per channel and writing `out`.
    pub fn forward_raw(
        &mut self,
        x: &[f32],
        batch: usize,
        h: usize,
        w: usize,
        out: &mut [f32],
    ) -> OchreResult<(Vec<f32>, Vec<f32>)> {
        let c = self.num_features;
        let plane = h * w;
        let m = batch * plane;
        if m == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.len() != batch * c * plane || out.len() != x.len() {
            return Err(OchreError::InvalidParameter(
                "BatchNorm2d::forward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }

        let mut mean = vec![0.0f32; c];
        let mut inv_std = vec![0.0f32; c];
        let gamma = self.weight.as_ref().map(|t| t.data.as_slice());
        let beta = self.bias.as_ref().map(|t| t.data.as_slice());

        if self.training {
            let inv_m = 1.0f64 / (m as f64);
            let eps_d = self.eps as f64;
            let mom = self.momentum as f64;

            for ch in 0..c {
                let mut sum = 0.0f64;
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for &v in &x[base..base + plane] {
                        sum += v as f64;
                    }
                }
                let mu = sum * inv_m;

                let mut sq = 0.0f64;
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for &v in &x[base..base + plane] {
                        let d = (v as f64) - mu;
                        sq += d * d;
                    }
                }
                let var = sq * inv_m;
                let istd = 1.0 / (var + eps_d).sqrt();

                mean[ch] = mu as f32;
                inv_std[ch] = istd as f32;

                let unbiased_var = if m > 1 { sq / ((m - 1) as f64) } else { var };
                self.running_mean[ch] =
                    ((1.0 - mom) * (self.running_mean[ch] as f64) + mom * mu) as f32;
                self.running_var[ch] =
                    ((1.0 - mom) * (self.running_var[ch] as f64) + mom * unbiased_var) as f32;

                let g = gamma.map_or(1.0f64, |gd| gd[ch] as f64);
                let bt = beta.map_or(0.0f64, |bd| bd[ch] as f64);
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for idx in 0..plane {
                        let x_hat = ((x[base + idx] as f64) - mu) * istd;
                        out[base + idx] = (g * x_hat + bt) as f32;
                    }
                }
            }
        } else {
            let eps_d = self.eps as f64;
            for ch in 0..c {
                let mu = self.running_mean[ch] as f64;
                let var = self.running_var[ch] as f64;
                let istd = 1.0 / (var + eps_d).sqrt();
                mean[ch] = mu as f32;
                inv_std[ch] = istd as f32;
                let g = gamma.map_or(1.0f64, |gd| gd[ch] as f64);
                let bt = beta.map_or(0.0f64, |bd| bd[ch] as f64);
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for idx in 0..plane {
                        let x_hat = ((x[base + idx] as f64) - mu) * istd;
                        out[base + idx] = (g * x_hat + bt) as f32;
                    }
                }
            }
        }

        Ok((mean, inv_std))
    }

    /// Raw backward pass computing `dx` (optional), `dgamma` (optional), `dbeta` (optional).
    #[allow(clippy::too_many_arguments)]
    pub fn backward_raw(
        &self,
        x: &[f32],
        dy: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        batch: usize,
        h: usize,
        w: usize,
        dx: Option<&mut [f32]>,
        dgamma: Option<&mut [f32]>,
        dbeta: Option<&mut [f32]>,
    ) -> OchreResult<()> {
        let c = self.num_features;
        let expected = batch * c * h * w;
        if x.len() != expected || dy.len() != expected || mean.len() != c || inv_std.len() != c {
            return Err(OchreError::InvalidParameter(
                "BatchNorm2d::backward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        let gamma = self.weight.as_ref().map(|t| t.data.as_slice());
        let (dx_vec, dg_vec, db_vec) = batchnorm2d_backward_core(
            x,
            dy,
            gamma,
            mean,
            inv_std,
            batch,
            c,
            h,
            w,
            self.training,
            dx.is_some(),
            dgamma.is_some() && self.affine,
            dbeta.is_some() && self.affine,
        );
        if let Some(dx_buf) = dx {
            if dx_buf.len() != expected {
                return Err(OchreError::InvalidParameter(
                    "BatchNorm2d::backward_raw dx length mismatch ( >_< )".to_string(),
                ));
            }
            dx_buf.copy_from_slice(&dx_vec);
        }
        if let Some(dg_buf) = dgamma
            && self.affine
        {
            if dg_buf.len() != c {
                return Err(OchreError::InvalidParameter(
                    "BatchNorm2d::backward_raw dgamma length mismatch ( >_< )".to_string(),
                ));
            }
            dg_buf.copy_from_slice(&dg_vec);
        }
        if let Some(db_buf) = dbeta
            && self.affine
        {
            if db_buf.len() != c {
                return Err(OchreError::InvalidParameter(
                    "BatchNorm2d::backward_raw dbeta length mismatch ( >_< )".to_string(),
                ));
            }
            db_buf.copy_from_slice(&db_vec);
        }
        Ok(())
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&mut self, x: &Tensor) -> OchreResult<Tensor> {
        let (batch, c, h, w) = validate_nchw_tensor(x, Some(self.num_features), "BatchNorm2d")?;
        let mut raw_out = vec![0.0f32; x.data.len()];
        let (mean, inv_std) = self.forward_raw(&x.data, batch, h, w, &mut raw_out)?;
        let out_data = quantize_vec(raw_out, self.dtype);

        let w_id = self
            .weight
            .as_ref()
            .and_then(|t| if t.requires_grad { t.node_id } else { None });
        let b_id = self
            .bias
            .as_ref()
            .and_then(|t| if t.requires_grad { t.node_id } else { None });
        let x_id = if x.requires_grad { x.node_id } else { None };
        let need_grad =
            AutogradTape::is_grad_enabled() && (x_id.is_some() || w_id.is_some() || b_id.is_some());

        let node_id = if need_grad {
            let x_saved = if x_id.is_some() || w_id.is_some() {
                x.data.clone()
            } else {
                Vec::new()
            };
            let gamma_saved = if x_id.is_some() {
                self.weight.as_ref().map(|t| t.data.clone())
            } else {
                None
            };
            let is_training = self.training;

            let bwd = CustomBackward(Arc::new(move |dy: &[f32]| {
                let (dx, dg, db) = batchnorm2d_backward_core(
                    &x_saved,
                    dy,
                    gamma_saved.as_deref(),
                    &mean,
                    &inv_std,
                    batch,
                    c,
                    h,
                    w,
                    is_training,
                    x_id.is_some(),
                    w_id.is_some(),
                    b_id.is_some(),
                );
                let mut res = Vec::with_capacity(3);
                if let Some(id) = x_id {
                    res.push((id, dx));
                }
                if let Some(id) = w_id {
                    res.push((id, dg));
                }
                if let Some(id) = b_id {
                    res.push((id, db));
                }
                Ok(res)
            }));

            Some(AutogradTape::record(Op::Custom {
                inputs: vec![x_id, w_id, b_id],
                backward: bwd,
            }))
        } else {
            None
        };

        Ok(Tensor {
            shape: x.shape.clone(),
            strides: compute_strides(&x.shape),
            data: out_data,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: need_grad,
            node_id,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn batchnorm2d_backward_core(
    x: &[f32],
    dy: &[f32],
    gamma: Option<&[f32]>,
    mean: &[f32],
    inv_std: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    is_training: bool,
    need_dx: bool,
    need_dg: bool,
    need_db: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let plane = h * w;
    let m = (batch * plane) as f64;
    let mut dx = if need_dx {
        vec![0.0f32; batch * c * plane]
    } else {
        Vec::new()
    };
    let mut dgamma = if need_dg { vec![0.0f32; c] } else { Vec::new() };
    let mut dbeta = if need_db { vec![0.0f32; c] } else { Vec::new() };

    for ch in 0..c {
        let mu = mean[ch] as f64;
        let istd = inv_std[ch] as f64;
        let g = gamma.map_or(1.0f64, |gd| gd[ch] as f64);

        let mut sum_dy = 0.0f64;
        let mut sum_dy_xhat = 0.0f64;
        for b in 0..batch {
            let base = (b * c + ch) * plane;
            for idx in 0..plane {
                let dyv = dy[base + idx] as f64;
                sum_dy += dyv;
                if need_dg || (need_dx && is_training) {
                    let x_hat = ((x[base + idx] as f64) - mu) * istd;
                    sum_dy_xhat += dyv * x_hat;
                }
            }
        }
        if need_dg {
            dgamma[ch] = sum_dy_xhat as f32;
        }
        if need_db {
            dbeta[ch] = sum_dy as f32;
        }
        if need_dx {
            if is_training {
                let scale = g * istd / m;
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for idx in 0..plane {
                        let dyv = dy[base + idx] as f64;
                        let x_hat = ((x[base + idx] as f64) - mu) * istd;
                        dx[base + idx] = (scale * (m * dyv - sum_dy - x_hat * sum_dy_xhat)) as f32;
                    }
                }
            } else {
                let scale = g * istd;
                for b in 0..batch {
                    let base = (b * c + ch) * plane;
                    for idx in 0..plane {
                        dx[base + idx] = ((dy[base + idx] as f64) * scale) as f32;
                    }
                }
            }
        }
    }

    (dx, dgamma, dbeta)
}

/// Flattens dimensions from `start_dim` onwards: `[N, C, H, W] -> [N, C*H*W]`.
#[derive(Clone, Copy, Debug)]
pub struct Flatten {
    pub start_dim: usize,
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Flatten {
    /// Creates a `Flatten` layer preserving batch axis 0 (`start_dim = 1`).
    pub fn new() -> Self {
        Self { start_dim: 1 }
    }

    /// Creates a `Flatten` layer flattening from `start_dim` onwards.
    pub fn with_start_dim(start_dim: usize) -> Self {
        Self { start_dim }
    }

    /// Autograd-compatible reshape from `[..., d_k, ...]` into `[..., prod(d_{>=start_dim})]`.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        if x.shape.is_empty() {
            return Err(OchreError::InvalidParameter(
                "Flatten cannot flatten a 0-D scalar tensor ( >_< )".to_string(),
            ));
        }
        if self.start_dim >= x.shape.len() {
            return Err(OchreError::InvalidParameter(format!(
                "Flatten start_dim={} out of bounds for shape {:?} ( >_< )",
                self.start_dim, x.shape
            )));
        }
        let mut new_shape = x.shape[..self.start_dim].to_vec();
        let tail: usize = x.shape[self.start_dim..].iter().copied().product();
        new_shape.push(tail);
        x.reshape(new_shape)
    }

    /// Raw forward copy (`x` -> `out`).
    pub fn forward_raw(&self, x: &[f32], out: &mut [f32]) -> OchreResult<()> {
        if x.len() != out.len() {
            return Err(OchreError::InvalidParameter(
                "Flatten::forward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        out.copy_from_slice(x);
        Ok(())
    }

    /// Raw backward copy (`dy` -> `dx`).
    pub fn backward_raw(&self, dy: &[f32], dx: &mut [f32]) -> OchreResult<()> {
        if dy.len() != dx.len() {
            return Err(OchreError::InvalidParameter(
                "Flatten::backward_raw slice length mismatch ( >_< )".to_string(),
            ));
        }
        dx.copy_from_slice(dy);
        Ok(())
    }
}

/// ResNet-style basic residual block:
/// `Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d + optional 1x1 downsample -> ReLU`.
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub downsample: Option<(Conv2d, BatchNorm2d)>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: (usize, usize),
}

impl ResidualBlock {
    /// Creates a `ResidualBlock` with optional 1x1 downsample shortcut when shape/stride changes.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        let conv1 = Conv2d::new(
            in_channels,
            out_channels,
            (3, 3),
            stride,
            (1, 1),
            (1, 1),
            1,
            false,
            dtype,
            device.clone(),
            seed,
        )?;
        let bn1 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, dtype, device.clone())?;
        let conv2 = Conv2d::new(
            out_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            false,
            dtype,
            device.clone(),
            seed.wrapping_add(17),
        )?;
        let bn2 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, dtype, device.clone())?;

        let downsample = if in_channels != out_channels || stride != (1, 1) {
            let ds_conv = Conv2d::new(
                in_channels,
                out_channels,
                (1, 1),
                stride,
                (0, 0),
                (1, 1),
                1,
                false,
                dtype,
                device.clone(),
                seed.wrapping_add(31),
            )?;
            let ds_bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, dtype, device)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            in_channels,
            out_channels,
            stride,
        })
    }

    /// Sets training mode across all `BatchNorm2d` submodules.
    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.train();
        }
    }

    /// Sets evaluation mode across all `BatchNorm2d` submodules.
    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.eval();
        }
    }

    /// Returns references to all trainable parameters in the block.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    /// Returns mutable references to all trainable parameters in the block.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.conv1.parameters_mut();
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        if let Some((ref mut ds_conv, ref mut ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters_mut());
            params.extend(ds_bn.parameters_mut());
        }
        params
    }

    /// Autograd-compatible forward pass on an NCHW `Tensor`.
    pub fn forward(&mut self, x: &Tensor) -> OchreResult<Tensor> {
        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(&out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        let identity = match &mut self.downsample {
            Some((ds_conv, ds_bn)) => {
                let d = ds_conv.forward(x)?;
                ds_bn.forward(&d)?
            }
            None => x.clone(),
        };
        let sum = out.add(&identity)?;
        sum.relu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::with_no_grad;
    use crate::nn::losses::{MSELoss, Reduction};
    use crate::nn::optim::SGD;

    fn check_grad_fd<F>(param: &Tensor, analytic: &[f32], f: F, tol: f32, label: &str)
    where
        F: Fn(&[f32]) -> f32,
    {
        let eps = 1e-3f32;
        for idx in 0..param.data.len() {
            let mut p_plus = param.data.clone();
            let mut p_minus = param.data.clone();
            p_plus[idx] += eps;
            p_minus[idx] -= eps;
            let num = (f(&p_plus) - f(&p_minus)) / (2.0 * eps);
            assert!(
                (analytic[idx] - num).abs() < tol,
                "{label}[{idx}] mismatch: analytic={} vs numeric={num}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn test_conv2d_grouped_and_raw_parity_finite_diff() {
        AutogradTape::clear();
        let conv = Conv2d::new(
            4,
            6,
            (3, 2),
            (2, 1),
            (1, 1),
            (1, 1),
            2,
            true,
            DType::F32,
            DeviceType::Cpu,
            11,
        )
        .unwrap();
        let x = Tensor::randn_with(
            vec![2, 4, 5, 4],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            22,
            true,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 6, 3, 5]);

        // Raw parity check
        let mut raw_y = vec![0.0f32; y.numel()];
        conv.forward_raw(&x.data, 2, 5, 4, &mut raw_y).unwrap();
        for (&a, &b) in y.data.iter().zip(raw_y.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let target = Tensor::randn(y.shape.clone(), DType::F32, DeviceType::Cpu, 33);
        let weights: Vec<f32> = target.data.iter().map(|&v| v * 0.3 + 0.1).collect();
        let w_tensor = Tensor::new(
            weights.clone(),
            y.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let loss = y.mul(&w_tensor).unwrap().sum(None).unwrap();
        let grads = loss.backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dw = grads.get(conv.weight.node_id.unwrap()).unwrap();
        let db = grads
            .get(conv.bias.as_ref().unwrap().node_id.unwrap())
            .unwrap();

        // Check raw backward matches autograd backward
        let mut raw_dx = vec![0.0f32; x.numel()];
        let mut raw_dw = vec![0.0f32; conv.weight.numel()];
        let mut raw_db = vec![0.0f32; conv.out_channels];
        conv.backward_raw(
            &x.data,
            &weights,
            2,
            5,
            4,
            Some(&mut raw_dx),
            &mut raw_dw,
            Some(&mut raw_db),
        )
        .unwrap();
        for (&a, &b) in dx.iter().zip(raw_dx.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (&a, &b) in dw.iter().zip(raw_dw.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (&a, &b) in db.iter().zip(raw_db.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        check_grad_fd(
            &x,
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
                    let yt = conv.forward(&xt).unwrap();
                    yt.data
                        .iter()
                        .zip(weights.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "Conv2d dx",
        );

        check_grad_fd(
            &conv.weight,
            dw,
            |wd| {
                with_no_grad(|| {
                    let wt = Tensor::new(
                        wd.to_vec(),
                        conv.weight.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let c2 = Conv2d::from_weights(
                        wt,
                        conv.bias.clone(),
                        conv.stride,
                        conv.padding,
                        conv.dilation,
                        conv.groups,
                    )
                    .unwrap();
                    let yt = c2.forward(&x).unwrap();
                    yt.data
                        .iter()
                        .zip(weights.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "Conv2d dw",
        );
    }

    #[test]
    fn test_depthwise_conv2d_finite_diff() {
        AutogradTape::clear();
        let dwconv = DepthwiseConv2d::new(
            3,
            2,
            (3, 3),
            (1, 2),
            (1, 1),
            (1, 1),
            true,
            DType::F32,
            DeviceType::Cpu,
            41,
        )
        .unwrap();
        let x = Tensor::randn_with(
            vec![2, 3, 4, 5],
            0.0,
            0.5,
            DType::F32,
            DeviceType::Cpu,
            42,
            true,
        );
        let y = dwconv.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 6, 4, 3]);

        let coeffs: Vec<f32> = (0..y.numel()).map(|i| ((i as f32) * 0.13).sin()).collect();
        let ct = Tensor::new(
            coeffs.clone(),
            y.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let grads = y.mul(&ct).unwrap().sum(None).unwrap().backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dw = grads.get(dwconv.weight.node_id.unwrap()).unwrap();
        let db = grads
            .get(dwconv.bias.as_ref().unwrap().node_id.unwrap())
            .unwrap();
        assert_eq!(db.len(), 6);

        check_grad_fd(
            &x,
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
                    let yt = dwconv.forward(&xt).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "DepthwiseConv2d dx",
        );

        check_grad_fd(
            &dwconv.weight,
            dw,
            |wd| {
                with_no_grad(|| {
                    let mut clone = dwconv.clone();
                    clone.weight = Tensor::new(
                        wd.to_vec(),
                        dwconv.weight.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let yt = clone.forward(&x).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "DepthwiseConv2d dw",
        );
    }

    #[test]
    fn test_pooling_layers_and_flatten_finite_diff() {
        AutogradTape::clear();
        // Distinct values avoid ties in MaxPool2d finite difference
        let x_vals: Vec<f32> = (0..2 * 2 * 4 * 4)
            .map(|i| (i as f32) * 0.17 - 2.3 + ((i * 7 % 5) as f32) * 0.31)
            .collect();
        let x = Tensor::new(x_vals, vec![2, 2, 4, 4], DType::F32, DeviceType::Cpu, true).unwrap();

        // MaxPool2d
        let mp = MaxPool2d::new((2, 2), (2, 2), (0, 0)).unwrap();
        let ym = mp.forward(&x).unwrap();
        assert_eq!(ym.shape, vec![2, 2, 2, 2]);
        let cm: Vec<f32> = (0..ym.numel()).map(|i| ((i as f32) + 1.0) * 0.25).collect();
        let cmt = Tensor::new(
            cm.clone(),
            ym.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let gm = ym.mul(&cmt).unwrap().sum(None).unwrap().backward().unwrap();
        let dxm = gm.get(x.node_id.unwrap()).unwrap();
        check_grad_fd(
            &x,
            dxm,
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
                    let yt = mp.forward(&xt).unwrap();
                    yt.data.iter().zip(cm.iter()).map(|(&a, &b)| a * b).sum()
                })
            },
            2e-2,
            "MaxPool2d dx",
        );

        // AvgPool2d
        let ap = AvgPool2d::new((2, 2), (1, 1), (1, 1)).unwrap();
        let ya = ap.forward(&x).unwrap();
        assert_eq!(ya.shape, vec![2, 2, 5, 5]);
        let ca: Vec<f32> = (0..ya.numel()).map(|i| ((i as f32) * 0.19).cos()).collect();
        let cat = Tensor::new(
            ca.clone(),
            ya.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let ga = ya.mul(&cat).unwrap().sum(None).unwrap().backward().unwrap();
        let dxa = ga.get(x.node_id.unwrap()).unwrap();
        check_grad_fd(
            &x,
            dxa,
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
                    let yt = ap.forward(&xt).unwrap();
                    yt.data.iter().zip(ca.iter()).map(|(&a, &b)| a * b).sum()
                })
            },
            2e-2,
            "AvgPool2d dx",
        );

        // GlobalAvgPool2d + Flatten
        let gap = GlobalAvgPool2d::new();
        let flat = Flatten::new();
        let yg = flat.forward(&gap.forward(&x).unwrap()).unwrap();
        assert_eq!(yg.shape, vec![2, 2]);
        let cg = vec![0.4f32, -0.7, 1.2, 0.3];
        let cgt = Tensor::new(
            cg.clone(),
            yg.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let gg = yg.mul(&cgt).unwrap().sum(None).unwrap().backward().unwrap();
        let dxg = gg.get(x.node_id.unwrap()).unwrap();
        check_grad_fd(
            &x,
            dxg,
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
                    let yt = flat.forward(&gap.forward(&xt).unwrap()).unwrap();
                    yt.data.iter().zip(cg.iter()).map(|(&a, &b)| a * b).sum()
                })
            },
            2e-2,
            "GlobalAvgPool2d+Flatten dx",
        );
    }

    #[test]
    fn test_batchnorm2d_train_finite_diff_and_eval_running_stats() {
        AutogradTape::clear();
        let mut bn = BatchNorm2d::new(2, 1e-5, 0.2, true, DType::F32, DeviceType::Cpu).unwrap();
        bn.weight.as_mut().unwrap().data = vec![1.3, 0.7];
        bn.bias.as_mut().unwrap().data = vec![0.2, -0.4];

        let x = Tensor::randn_with(
            vec![2, 2, 3, 3],
            1.5,
            0.8,
            DType::F32,
            DeviceType::Cpu,
            77,
            true,
        );
        // Snapshot running stats before forward so finite diff uses fixed forward behavior
        let rmean_init = bn.running_mean.clone();
        let rvar_init = bn.running_var.clone();

        let y = bn.forward(&x).unwrap();
        assert!(bn.running_mean[0] > 0.1);
        assert!((bn.running_var[0] - 1.0).abs() > 1e-3);

        let coeffs: Vec<f32> = (0..y.numel()).map(|i| ((i as f32) * 0.21).sin()).collect();
        let ct = Tensor::new(
            coeffs.clone(),
            y.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let grads = y.mul(&ct).unwrap().sum(None).unwrap().backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let dg = grads
            .get(bn.weight.as_ref().unwrap().node_id.unwrap())
            .unwrap();
        let db = grads
            .get(bn.bias.as_ref().unwrap().node_id.unwrap())
            .unwrap();

        check_grad_fd(
            &x,
            dx,
            |xd| {
                with_no_grad(|| {
                    let mut bn_copy = bn.clone();
                    bn_copy.running_mean = rmean_init.clone();
                    bn_copy.running_var = rvar_init.clone();
                    let xt = Tensor::new(
                        xd.to_vec(),
                        x.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let yt = bn_copy.forward(&xt).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "BatchNorm2d dx",
        );

        check_grad_fd(
            bn.weight.as_ref().unwrap(),
            dg,
            |gd| {
                with_no_grad(|| {
                    let mut bn_copy = bn.clone();
                    bn_copy.running_mean = rmean_init.clone();
                    bn_copy.running_var = rvar_init.clone();
                    bn_copy.weight.as_mut().unwrap().data = gd.to_vec();
                    let yt = bn_copy.forward(&x).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "BatchNorm2d dgamma",
        );

        check_grad_fd(
            bn.bias.as_ref().unwrap(),
            db,
            |bd| {
                with_no_grad(|| {
                    let mut bn_copy = bn.clone();
                    bn_copy.running_mean = rmean_init.clone();
                    bn_copy.running_var = rvar_init.clone();
                    bn_copy.bias.as_mut().unwrap().data = bd.to_vec();
                    let yt = bn_copy.forward(&x).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            2e-2,
            "BatchNorm2d dbeta",
        );

        // Eval mode must use frozen running stats regardless of batch content
        bn.eval();
        let saved_rm = bn.running_mean.clone();
        let saved_rv = bn.running_var.clone();
        let x_single = Tensor::new(
            vec![2.0; 2 * 2 * 2],
            vec![1, 2, 2, 2],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let y_eval = bn.forward(&x_single).unwrap();
        assert_eq!(bn.running_mean, saved_rm);
        assert_eq!(bn.running_var, saved_rv);
        let expected_c0 = 1.3 * (2.0 - saved_rm[0]) / (saved_rv[0] + 1e-5).sqrt() + 0.2;
        assert!((y_eval.data[0] - expected_c0).abs() < 1e-4);
    }

    #[test]
    fn test_residual_block_finite_diff_and_training_loss_decreases() {
        AutogradTape::clear();
        let mut block = ResidualBlock::new(2, 3, (2, 2), DType::F32, DeviceType::Cpu, 101).unwrap();
        assert!(block.downsample.is_some());
        let x = Tensor::randn_with(
            vec![2, 2, 4, 4],
            0.5,
            0.7,
            DType::F32,
            DeviceType::Cpu,
            102,
            true,
        );
        // Eval mode for deterministic finite-diff (train mode also tested in training loop below)
        block.eval();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 3, 2, 2]);

        let coeffs: Vec<f32> = (0..y.numel())
            .map(|i| ((i as f32) * 0.37).sin() + 0.5)
            .collect();
        let ct = Tensor::new(
            coeffs.clone(),
            y.shape.clone(),
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let grads = y.mul(&ct).unwrap().sum(None).unwrap().backward().unwrap();

        let dx = grads.get(x.node_id.unwrap()).unwrap();
        check_grad_fd(
            &x,
            dx,
            |xd| {
                with_no_grad(|| {
                    let mut b_clone = block.clone();
                    let xt = Tensor::new(
                        xd.to_vec(),
                        x.shape.clone(),
                        DType::F32,
                        DeviceType::Cpu,
                        false,
                    )
                    .unwrap();
                    let yt = b_clone.forward(&xt).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            3e-2,
            "ResidualBlock dx",
        );

        let dw_c1 = grads.get(block.conv1.weight.node_id.unwrap()).unwrap();
        check_grad_fd(
            &block.conv1.weight,
            dw_c1,
            |wd| {
                with_no_grad(|| {
                    let mut b_clone = block.clone();
                    b_clone.conv1.weight.data = wd.to_vec();
                    let yt = b_clone.forward(&x).unwrap();
                    yt.data
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
            },
            3e-2,
            "ResidualBlock conv1.weight",
        );

        // Training convergence test: Conv2d -> ResidualBlock -> GlobalAvgPool2d -> Flatten -> MSE
        let mut stem = Conv2d::new(
            2,
            4,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            true,
            DType::F32,
            DeviceType::Cpu,
            201,
        )
        .unwrap();
        let mut res = ResidualBlock::new(4, 4, (1, 1), DType::F32, DeviceType::Cpu, 202).unwrap();
        res.train();
        let gap = GlobalAvgPool2d::with_keep_dims(false);
        let mse = MSELoss::new(Reduction::Mean);
        let mut sgd = SGD::new(0.05, 0.9, false, 0.0).unwrap();

        let train_x = Tensor::randn_with(
            vec![4, 2, 4, 4],
            0.0,
            1.0,
            DType::F32,
            DeviceType::Cpu,
            203,
            false,
        );
        let train_y = Tensor::ones(vec![4, 4], DType::F32, DeviceType::Cpu);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;
        for step in 0..12 {
            AutogradTape::clear();
            let h = stem.forward(&train_x).unwrap();
            let h = res.forward(&h).unwrap();
            let pred = gap.forward(&h).unwrap();
            let loss = mse.forward_tensor(&pred, &train_y).unwrap();
            if step == 0 {
                first_loss = loss.data[0];
            }
            last_loss = loss.data[0];
            let grads = loss.backward().unwrap();

            let mut params: Vec<&mut Tensor> = stem.parameters_mut();
            params.extend(res.parameters_mut());
            sgd.step(&mut params, &grads).unwrap();
        }
        assert!(
            last_loss < first_loss * 0.5,
            "Expected CNN+ResidualBlock loss to drop significantly: first={first_loss}, last={last_loss}"
        );
    }

    #[test]
    fn test_conv2d_dilated_and_standalone_training() {
        AutogradTape::clear();
        let mut conv = Conv2d::new(
            2,
            2,
            (3, 3),
            (1, 1),
            (2, 2),
            (2, 2),
            1,
            true,
            DType::F32,
            DeviceType::Cpu,
            301,
        )
        .unwrap();
        let x = Tensor::randn_with(
            vec![2, 2, 5, 5],
            0.0,
            0.7,
            DType::F32,
            DeviceType::Cpu,
            302,
            true,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 2, 5, 5]);
        let grads = y.sum(None).unwrap().backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        check_grad_fd(
            &x,
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
                    conv.forward(&xt).unwrap().sum(None).unwrap().data[0]
                })
            },
            2e-2,
            "Dilated Conv2d dx",
        );

        let target = Tensor::ones(vec![2, 2, 5, 5], DType::F32, DeviceType::Cpu);
        let mse = MSELoss::new(Reduction::Mean);
        let mut sgd = SGD::new(0.05, 0.9, false, 0.0).unwrap();
        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;
        for step in 0..8 {
            AutogradTape::clear();
            let pred = conv.forward(&x).unwrap();
            let loss = mse.forward_tensor(&pred, &target).unwrap();
            if step == 0 {
                first_loss = loss.data[0];
            }
            last_loss = loss.data[0];
            let g = loss.backward().unwrap();
            let mut params = conv.parameters_mut();
            sgd.step(&mut params, &g).unwrap();
        }
        assert!(last_loss < first_loss * 0.5);
    }
}
