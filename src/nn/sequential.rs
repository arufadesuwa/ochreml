// OchreML Zero-GIL Sequential Trainer & Serialization Module (*≧ω≦*)
// Pre-allocated buffer training loop using fused N8/N9 kernels and DIT\x01 tag 100 persistence.

use crate::backend::{Backend, get_backend};
use crate::device::DeviceType;
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f32, write_f32_vec, write_header, write_u8,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::cnn::{
    AvgPool2d, BatchNorm2d, Conv2d, DepthwiseConv2d, Flatten, GlobalAvgPool2d, MaxPool2d,
    ResidualBlock,
};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::layers::{Activation, Dropout, Embedding, Linear};
use crate::nn::losses::{MSELoss, Reduction};
use crate::nn::norm::{LayerNorm, RMSNorm};
use crate::nn::optim::{Adam, AdamW, SGD, clip_grad_norm_};
use crate::nn::tensor::{FastRng, Tensor};

/// Binary serialization tag for `Sequential` in `.dit` files.
pub const TAG_NN_SEQUENTIAL: u8 = 100;

const LAYER_TAG_LINEAR: u8 = 1;
const LAYER_TAG_RELU: u8 = 2;
const LAYER_TAG_SIGMOID: u8 = 3;
const LAYER_TAG_TANH: u8 = 4;
const LAYER_TAG_GELU: u8 = 5;
const LAYER_TAG_SILU: u8 = 6;
const LAYER_TAG_DROPOUT: u8 = 7;
const LAYER_TAG_EMBEDDING: u8 = 8;
const LAYER_TAG_LAYERNORM: u8 = 9;
const LAYER_TAG_RMSNORM: u8 = 10;
const LAYER_TAG_CONV2D: u8 = 11;
const LAYER_TAG_DEPTHWISE_CONV2D: u8 = 12;
const LAYER_TAG_MAXPOOL2D: u8 = 13;
const LAYER_TAG_AVGPOOL2D: u8 = 14;
const LAYER_TAG_GLOBAL_AVGPOOL2D: u8 = 15;
const LAYER_TAG_BATCHNORM2D: u8 = 16;
const LAYER_TAG_FLATTEN: u8 = 17;
const LAYER_TAG_RESIDUAL_BLOCK: u8 = 18;

/// Optimizer choice for `Sequential::fit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimKind {
    SGD,
    Adam,
    AdamW,
}

/// Loss function choice for `Sequential::fit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    CrossEntropy,
    MSE,
}

/// Declarative layer specification for constructing a `Sequential` model.
#[derive(Clone, Debug)]
pub enum LayerSpec {
    Linear {
        in_features: usize,
        out_features: usize,
        bias: bool,
    },
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Dropout {
        p: f32,
    },
    Embedding {
        num_embeddings: usize,
        embedding_dim: usize,
    },
    LayerNorm {
        normalized_shape: usize,
        eps: f32,
        elementwise_affine: bool,
    },
    RMSNorm {
        normalized_shape: usize,
        eps: f32,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    },
    DepthwiseConv2d {
        in_channels: usize,
        depth_multiplier: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
    },
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    AvgPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    GlobalAvgPool2d {
        keep_dims: bool,
    },
    BatchNorm2d {
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
    },
    Flatten {
        start_dim: usize,
    },
    ResidualBlock {
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
    },
}

/// Concrete layer enum for `Sequential`, supporting MLP, CNN, and Normalization layers.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum SequentialLayer {
    Linear(Linear),
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Dropout(Dropout),
    Embedding(Embedding),
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
    Conv2d(Conv2d),
    DepthwiseConv2d(DepthwiseConv2d),
    MaxPool2d(MaxPool2d),
    AvgPool2d(AvgPool2d),
    GlobalAvgPool2d(GlobalAvgPool2d),
    BatchNorm2d(BatchNorm2d),
    Flatten(Flatten),
    ResidualBlock(ResidualBlock),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DimState {
    Flat(usize),
    Nchw { c: usize, h: usize, w: usize },
}

impl DimState {
    #[inline]
    fn elems(&self) -> usize {
        match *self {
            DimState::Flat(d) => d,
            DimState::Nchw { c, h, w } => c * h * w,
        }
    }

    fn require_nchw(&self, layer_name: &str) -> OchreResult<(usize, usize, usize)> {
        match *self {
            DimState::Nchw { c, h, w } => Ok((c, h, w)),
            DimState::Flat(d) => Err(OchreError::InvalidParameter(format!(
                "{layer_name} expects 4D NCHW spatial input [N, C, H, W], got flat dim {d} ( >_< )"
            ))),
        }
    }
}

impl SequentialLayer {
    fn output_state(&self, current: DimState) -> OchreResult<DimState> {
        match self {
            SequentialLayer::Linear(lin) => {
                let d = current.elems();
                if d != lin.in_features {
                    return Err(OchreError::InvalidParameter(format!(
                        "Layer dimension mismatch: input has {d}, Linear expects {} ( >_< )",
                        lin.in_features
                    )));
                }
                Ok(DimState::Flat(lin.out_features))
            }
            SequentialLayer::ReLU
            | SequentialLayer::Sigmoid
            | SequentialLayer::Tanh
            | SequentialLayer::GELU
            | SequentialLayer::SiLU
            | SequentialLayer::Dropout(_) => Ok(current),
            SequentialLayer::Embedding(emb) => {
                if current != DimState::Flat(1) {
                    return Err(OchreError::InvalidParameter(format!(
                        "Embedding in Sequential expects scalar token index per sample (in_dim=1), got {current:?} ( >_< )"
                    )));
                }
                Ok(DimState::Flat(emb.embedding_dim))
            }
            SequentialLayer::LayerNorm(ln) => {
                let last_dim = match current {
                    DimState::Flat(d) => d,
                    DimState::Nchw { w, .. } => w,
                };
                if last_dim != ln.normalized_shape {
                    return Err(OchreError::InvalidParameter(format!(
                        "LayerNorm expected last dim {}, got {last_dim} ( >_< )",
                        ln.normalized_shape
                    )));
                }
                Ok(current)
            }
            SequentialLayer::RMSNorm(rn) => {
                let last_dim = match current {
                    DimState::Flat(d) => d,
                    DimState::Nchw { w, .. } => w,
                };
                if last_dim != rn.normalized_shape {
                    return Err(OchreError::InvalidParameter(format!(
                        "RMSNorm expected last dim {}, got {last_dim} ( >_< )",
                        rn.normalized_shape
                    )));
                }
                Ok(current)
            }
            SequentialLayer::Conv2d(conv) => {
                let (c, h, w) = current.require_nchw("Conv2d")?;
                if c != conv.in_channels {
                    return Err(OchreError::InvalidParameter(format!(
                        "Conv2d expected {} input channels, got {c} ( >_< )",
                        conv.in_channels
                    )));
                }
                let (oh, ow) = conv.output_shape(h, w)?;
                Ok(DimState::Nchw {
                    c: conv.out_channels,
                    h: oh,
                    w: ow,
                })
            }
            SequentialLayer::DepthwiseConv2d(dw) => {
                let (c, h, w) = current.require_nchw("DepthwiseConv2d")?;
                if c != dw.in_channels {
                    return Err(OchreError::InvalidParameter(format!(
                        "DepthwiseConv2d expected {} input channels, got {c} ( >_< )",
                        dw.in_channels
                    )));
                }
                let (oh, ow) = dw.output_shape(h, w)?;
                Ok(DimState::Nchw {
                    c: dw.out_channels,
                    h: oh,
                    w: ow,
                })
            }
            SequentialLayer::MaxPool2d(pool) => {
                let (c, h, w) = current.require_nchw("MaxPool2d")?;
                let (oh, ow) = pool.output_shape(h, w)?;
                Ok(DimState::Nchw { c, h: oh, w: ow })
            }
            SequentialLayer::AvgPool2d(pool) => {
                let (c, h, w) = current.require_nchw("AvgPool2d")?;
                let (oh, ow) = pool.output_shape(h, w)?;
                Ok(DimState::Nchw { c, h: oh, w: ow })
            }
            SequentialLayer::GlobalAvgPool2d(gap) => {
                let (c, _, _) = current.require_nchw("GlobalAvgPool2d")?;
                if gap.keep_dims {
                    Ok(DimState::Nchw { c, h: 1, w: 1 })
                } else {
                    Ok(DimState::Flat(c))
                }
            }
            SequentialLayer::BatchNorm2d(bn) => {
                let (c, h, w) = current.require_nchw("BatchNorm2d")?;
                if c != bn.num_features {
                    return Err(OchreError::InvalidParameter(format!(
                        "BatchNorm2d expected {} channels, got {c} ( >_< )",
                        bn.num_features
                    )));
                }
                Ok(DimState::Nchw { c, h, w })
            }
            SequentialLayer::Flatten(fl) => {
                if fl.start_dim == 0 || fl.start_dim > 3 {
                    return Err(OchreError::InvalidParameter(format!(
                        "Flatten in Sequential requires start_dim in 1..=3, got {} ( >_< )",
                        fl.start_dim
                    )));
                }
                Ok(DimState::Flat(current.elems()))
            }
            SequentialLayer::ResidualBlock(rb) => {
                let (c, h, w) = current.require_nchw("ResidualBlock")?;
                if c != rb.in_channels {
                    return Err(OchreError::InvalidParameter(format!(
                        "ResidualBlock expected {} input channels, got {c} ( >_< )",
                        rb.in_channels
                    )));
                }
                let (oh, ow) = rb.conv1.output_shape(h, w)?;
                Ok(DimState::Nchw {
                    c: rb.out_channels,
                    h: oh,
                    w: ow,
                })
            }
        }
    }

    fn as_activation(&self) -> Option<Activation> {
        match self {
            SequentialLayer::ReLU => Some(Activation::ReLU),
            SequentialLayer::Sigmoid => Some(Activation::Sigmoid),
            SequentialLayer::Tanh => Some(Activation::Tanh),
            SequentialLayer::GELU => Some(Activation::GELU),
            SequentialLayer::SiLU => Some(Activation::SiLU),
            _ => None,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match self {
            SequentialLayer::Linear(lin) => {
                let mut v = vec![&lin.weight];
                if let Some(ref b) = lin.bias {
                    v.push(b);
                }
                v
            }
            SequentialLayer::Embedding(emb) => vec![&emb.weight],
            SequentialLayer::LayerNorm(ln) => ln.parameters(),
            SequentialLayer::RMSNorm(rn) => rn.parameters(),
            SequentialLayer::Conv2d(c) => c.parameters(),
            SequentialLayer::DepthwiseConv2d(dw) => dw.parameters(),
            SequentialLayer::BatchNorm2d(bn) => bn.parameters(),
            SequentialLayer::ResidualBlock(rb) => rb.parameters(),
            _ => Vec::new(),
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        match self {
            SequentialLayer::Linear(lin) => {
                let mut v = vec![&mut lin.weight];
                if let Some(ref mut b) = lin.bias {
                    v.push(b);
                }
                v
            }
            SequentialLayer::Embedding(emb) => vec![&mut emb.weight],
            SequentialLayer::LayerNorm(ln) => ln.parameters_mut(),
            SequentialLayer::RMSNorm(rn) => rn.parameters_mut(),
            SequentialLayer::Conv2d(c) => c.parameters_mut(),
            SequentialLayer::DepthwiseConv2d(dw) => dw.parameters_mut(),
            SequentialLayer::BatchNorm2d(bn) => bn.parameters_mut(),
            SequentialLayer::ResidualBlock(rb) => rb.parameters_mut(),
            _ => Vec::new(),
        }
    }
}

/// Sequential neural network container with zero-GIL Rust `fit` and `predict`.
#[derive(Clone, Debug)]
pub struct Sequential {
    pub layers: Vec<SequentialLayer>,
    pub device: DeviceType,
    pub dtype: DType,
    pub training: bool,
}

struct SequentialPlan {
    in_state: DimState,
    layer_out_states: Vec<DimState>,
    max_dim: usize,
    out_dim: usize,
    param_ranges: Vec<(usize, usize, usize)>, // (layer_idx, sub_param_idx, n_elems)
    dropout_indices: Vec<usize>,
}

#[derive(Default)]
struct LayerScratch {
    argmax: Vec<usize>,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    // ResidualBlock saved activations/stats
    c1_out: Vec<f32>,
    b1_out: Vec<f32>,
    b1_mean: Vec<f32>,
    b1_inv: Vec<f32>,
    r1_out: Vec<f32>,
    c2_out: Vec<f32>,
    b2_mean: Vec<f32>,
    b2_inv: Vec<f32>,
    ds_c_out: Vec<f32>,
    ds_b_mean: Vec<f32>,
    ds_b_inv: Vec<f32>,
    sum_out: Vec<f32>,
    d_sum: Vec<f32>,
    d_tmp1: Vec<f32>,
    d_tmp2: Vec<f32>,
}

enum OptimizerRunner {
    Sgd(SGD),
    Adam(Adam),
    AdamW(AdamW),
}

impl OptimizerRunner {
    fn new(kind: OptimKind, lr: f32, weight_decay: f32, device: DeviceType) -> OchreResult<Self> {
        match kind {
            OptimKind::SGD => Ok(Self::Sgd(SGD::new(lr, 0.9, false, weight_decay)?)),
            OptimKind::Adam => Ok(Self::Adam(Adam::with_lr(lr, weight_decay)?)),
            OptimKind::AdamW => Ok(Self::AdamW(AdamW::with_lr(lr, weight_decay, device)?)),
        }
    }

    fn step_slices(&mut self, params: &mut [&mut [f32]], grads: &[&[f32]]) -> OchreResult<()> {
        match self {
            Self::Sgd(opt) => opt.step_slices(params, grads),
            Self::Adam(opt) => opt.step_slices(params, grads),
            Self::AdamW(opt) => opt.step_slices(params, grads),
        }
    }
}

impl Sequential {
    /// Wraps already-constructed layers into a `Sequential` model.
    pub fn new(layers: Vec<SequentialLayer>, dtype: DType, device: DeviceType) -> Self {
        Self {
            layers,
            device,
            dtype,
            training: true,
        }
    }

    /// Builds a `Sequential` model from declarative `LayerSpec` descriptors.
    pub fn from_specs(
        specs: &[LayerSpec],
        dtype: DType,
        device: DeviceType,
        seed: u64,
    ) -> OchreResult<Self> {
        if specs.is_empty() {
            return Err(OchreError::InvalidParameter(
                "Sequential requires at least one layer ( >_< )".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(specs.len());
        let mut layer_seed = seed;
        for spec in specs {
            layer_seed = layer_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let layer = match spec {
                LayerSpec::Linear {
                    in_features,
                    out_features,
                    bias,
                } => SequentialLayer::Linear(Linear::new(
                    *in_features,
                    *out_features,
                    *bias,
                    dtype,
                    device.clone(),
                    layer_seed,
                )?),
                LayerSpec::ReLU => SequentialLayer::ReLU,
                LayerSpec::Sigmoid => SequentialLayer::Sigmoid,
                LayerSpec::Tanh => SequentialLayer::Tanh,
                LayerSpec::GELU => SequentialLayer::GELU,
                LayerSpec::SiLU => SequentialLayer::SiLU,
                LayerSpec::Dropout { p } => SequentialLayer::Dropout(Dropout::new(*p, layer_seed)?),
                LayerSpec::Embedding {
                    num_embeddings,
                    embedding_dim,
                } => SequentialLayer::Embedding(Embedding::new(
                    *num_embeddings,
                    *embedding_dim,
                    dtype,
                    device.clone(),
                    layer_seed,
                )?),
                LayerSpec::LayerNorm {
                    normalized_shape,
                    eps,
                    elementwise_affine,
                } => SequentialLayer::LayerNorm(LayerNorm::new(
                    *normalized_shape,
                    *eps,
                    *elementwise_affine,
                    dtype,
                    device.clone(),
                )?),
                LayerSpec::RMSNorm {
                    normalized_shape,
                    eps,
                } => SequentialLayer::RMSNorm(RMSNorm::new(
                    *normalized_shape,
                    *eps,
                    dtype,
                    device.clone(),
                )?),
                LayerSpec::Conv2d {
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                } => SequentialLayer::Conv2d(Conv2d::new(
                    *in_channels,
                    *out_channels,
                    *kernel_size,
                    *stride,
                    *padding,
                    *dilation,
                    *groups,
                    *bias,
                    dtype,
                    device.clone(),
                    layer_seed,
                )?),
                LayerSpec::DepthwiseConv2d {
                    in_channels,
                    depth_multiplier,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias,
                } => SequentialLayer::DepthwiseConv2d(DepthwiseConv2d::new(
                    *in_channels,
                    *depth_multiplier,
                    *kernel_size,
                    *stride,
                    *padding,
                    *dilation,
                    *bias,
                    dtype,
                    device.clone(),
                    layer_seed,
                )?),
                LayerSpec::MaxPool2d {
                    kernel_size,
                    stride,
                    padding,
                } => SequentialLayer::MaxPool2d(MaxPool2d::new(*kernel_size, *stride, *padding)?),
                LayerSpec::AvgPool2d {
                    kernel_size,
                    stride,
                    padding,
                } => SequentialLayer::AvgPool2d(AvgPool2d::new(*kernel_size, *stride, *padding)?),
                LayerSpec::GlobalAvgPool2d { keep_dims } => {
                    SequentialLayer::GlobalAvgPool2d(GlobalAvgPool2d::with_keep_dims(*keep_dims))
                }
                LayerSpec::BatchNorm2d {
                    num_features,
                    eps,
                    momentum,
                    affine,
                } => SequentialLayer::BatchNorm2d(BatchNorm2d::new(
                    *num_features,
                    *eps,
                    *momentum,
                    *affine,
                    dtype,
                    device.clone(),
                )?),
                LayerSpec::Flatten { start_dim } => {
                    SequentialLayer::Flatten(Flatten::with_start_dim(*start_dim))
                }
                LayerSpec::ResidualBlock {
                    in_channels,
                    out_channels,
                    stride,
                } => SequentialLayer::ResidualBlock(ResidualBlock::new(
                    *in_channels,
                    *out_channels,
                    *stride,
                    dtype,
                    device.clone(),
                    layer_seed,
                )?),
            };
            layers.push(layer);
        }
        Ok(Self::new(layers, dtype, device))
    }

    /// Returns references to all trainable parameter tensors across layers.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Returns mutable references to all trainable parameter tensors across layers.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    /// Total number of scalar trainable parameters across all layers.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|t| t.numel()).sum()
    }

    /// Sets training mode (`training = true`) across the model, Dropout, BatchNorm2d, and ResidualBlock layers.
    pub fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            match layer {
                SequentialLayer::Dropout(d) => d.training = true,
                SequentialLayer::BatchNorm2d(bn) => bn.train(),
                SequentialLayer::ResidualBlock(rb) => rb.train(),
                _ => {}
            }
        }
    }

    /// Sets evaluation mode (`training = false`) across the model, Dropout, BatchNorm2d, and ResidualBlock layers.
    pub fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            match layer {
                SequentialLayer::Dropout(d) => d.training = false,
                SequentialLayer::BatchNorm2d(bn) => bn.eval(),
                SequentialLayer::ResidualBlock(rb) => rb.eval(),
                _ => {}
            }
        }
    }

    /// Autograd-compatible forward pass on a `Tensor`.
    pub fn forward(&mut self, x: &Tensor) -> OchreResult<Tensor> {
        let mut cur = x.clone();
        for layer in &mut self.layers {
            cur = match layer {
                SequentialLayer::Linear(lin) => lin.forward(&cur)?,
                SequentialLayer::ReLU => cur.relu()?,
                SequentialLayer::Sigmoid => cur.sigmoid()?,
                SequentialLayer::Tanh => cur.tanh()?,
                SequentialLayer::GELU => cur.gelu()?,
                SequentialLayer::SiLU => cur.silu()?,
                SequentialLayer::Dropout(d) => d.forward(&cur)?,
                SequentialLayer::Embedding(emb) => {
                    let indices: Vec<usize> =
                        cur.data.iter().map(|&v| v.max(0.0) as usize).collect();
                    emb.forward(&indices)?
                }
                SequentialLayer::LayerNorm(ln) => ln.forward(&cur)?,
                SequentialLayer::RMSNorm(rn) => rn.forward(&cur)?,
                SequentialLayer::Conv2d(c) => c.forward(&cur)?,
                SequentialLayer::DepthwiseConv2d(dw) => dw.forward(&cur)?,
                SequentialLayer::MaxPool2d(p) => p.forward(&cur)?,
                SequentialLayer::AvgPool2d(p) => p.forward(&cur)?,
                SequentialLayer::GlobalAvgPool2d(gap) => gap.forward(&cur)?,
                SequentialLayer::BatchNorm2d(bn) => bn.forward(&cur)?,
                SequentialLayer::Flatten(fl) => fl.forward(&cur)?,
                SequentialLayer::ResidualBlock(rb) => rb.forward(&cur)?,
            };
        }
        Ok(cur)
    }

    /// Zero-GIL, zero-per-batch-allocation Rust training loop for flat `[n_samples, in_dim]` inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        x: &[f32],
        n_samples: usize,
        in_dim: usize,
        y_class: Option<&[usize]>,
        y_reg: Option<&[f32]>,
        out_dim: usize,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        optimizer: OptimKind,
        loss: LossKind,
        weight_decay: f32,
        grad_clip: Option<f32>,
        shuffle: bool,
        seed: u64,
    ) -> OchreResult<Vec<f32>> {
        self.fit_nd(
            x,
            n_samples,
            &[in_dim],
            y_class,
            y_reg,
            out_dim,
            epochs,
            batch_size,
            lr,
            optimizer,
            loss,
            weight_decay,
            grad_clip,
            shuffle,
            seed,
        )
    }

    /// Zero-GIL Rust training loop supporting both `[in_dim]` and `[C, H, W]` per-sample input shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_nd(
        &mut self,
        x: &[f32],
        n_samples: usize,
        sample_shape: &[usize],
        y_class: Option<&[usize]>,
        y_reg: Option<&[f32]>,
        out_dim: usize,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        optimizer: OptimKind,
        loss: LossKind,
        weight_decay: f32,
        grad_clip: Option<f32>,
        shuffle: bool,
        seed: u64,
    ) -> OchreResult<Vec<f32>> {
        let in_state = parse_sample_shape(sample_shape)?;
        let in_dim = in_state.elems();
        validate_fit_inputs(
            &self.layers,
            x,
            n_samples,
            in_dim,
            y_class,
            y_reg,
            out_dim,
            epochs,
            batch_size,
            loss,
        )?;

        let plan = build_plan(&self.layers, in_state)?;
        if plan.out_dim != out_dim {
            return Err(OchreError::InvalidParameter(format!(
                "Model output dimension {} does not match out_dim={out_dim} ( >_< )",
                plan.out_dim
            )));
        }

        self.train();
        let max_bs = batch_size.min(n_samples);
        let backend = get_backend(&self.device);

        // Pre-allocate all batch/activation/gradient/optimizer buffers ONCE outside epoch loop.
        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len() + 1);
        acts.push(vec![0.0f32; max_bs * in_dim]);
        for st in &plan.layer_out_states {
            acts.push(vec![0.0f32; max_bs * st.elems()]);
        }

        let mut grad_a = vec![0.0f32; max_bs * plan.max_dim];
        let mut grad_b = vec![0.0f32; max_bs * plan.max_dim];
        let mut batch_y_class = vec![0usize; max_bs];
        let mut batch_y_reg = vec![0.0f32; max_bs * out_dim];
        let mut batch_indices = vec![0usize; max_bs];

        let mut dropout_masks: Vec<Vec<f32>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(idx, l)| match l {
                SequentialLayer::Dropout(_) => {
                    let dim = if idx == 0 {
                        in_dim
                    } else {
                        plan.layer_out_states[idx - 1].elems()
                    };
                    vec![1.0f32; max_bs * dim]
                }
                _ => Vec::new(),
            })
            .collect();

        let mut scratches: Vec<LayerScratch> = (0..self.layers.len())
            .map(|_| LayerScratch::default())
            .collect();

        let mut param_grads: Vec<Vec<f32>> = plan
            .param_ranges
            .iter()
            .map(|&(_, _, len)| vec![0.0f32; len])
            .collect();
        let mut master_weights: Vec<Option<Vec<f32>>> = if self.dtype != DType::F32 {
            plan.param_ranges
                .iter()
                .map(|&(layer_idx, sub_idx, _)| {
                    Some(layer_param_slices(&self.layers[layer_idx])[sub_idx].to_vec())
                })
                .collect()
        } else {
            vec![None; plan.param_ranges.len()]
        };

        let mut opt_runner =
            OptimizerRunner::new(optimizer, lr, weight_decay, self.device.clone())?;
        let mse = MSELoss::new(Reduction::Mean);
        let mut perm: Vec<usize> = (0..n_samples).collect();
        let mut rng = FastRng::new(seed);
        let mut loss_history = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            if shuffle {
                rng.shuffle(&mut perm);
            }
            let mut epoch_loss = 0.0f64;

            for start in (0..n_samples).step_by(max_bs) {
                let end = (start + max_bs).min(n_samples);
                let bs = end - start;
                let batch_perm = &perm[start..end];

                gather_batch_x(
                    x,
                    in_dim,
                    batch_perm,
                    shuffle,
                    start,
                    &mut acts[0][..bs * in_dim],
                );
                if let Some(yc) = y_class {
                    gather_batch_class(yc, batch_perm, shuffle, start, &mut batch_y_class[..bs]);
                }
                if let Some(yr) = y_reg {
                    gather_batch_x(
                        yr,
                        out_dim,
                        batch_perm,
                        shuffle,
                        start,
                        &mut batch_y_reg[..bs * out_dim],
                    );
                }

                for (li, layer) in self.layers.iter_mut().enumerate() {
                    let in_st = if li == 0 {
                        plan.in_state
                    } else {
                        plan.layer_out_states[li - 1]
                    };
                    let out_st = plan.layer_out_states[li];
                    let in_d = in_st.elems();
                    let out_d = out_st.elems();
                    let (left, right) = acts.split_at_mut(li + 1);
                    let in_slice = &left[li][..bs * in_d];
                    let out_slice = &mut right[0][..bs * out_d];

                    let mask_len = if dropout_masks[li].is_empty() {
                        0
                    } else {
                        bs * in_d
                    };
                    forward_layer_raw(
                        layer,
                        in_st,
                        out_st,
                        in_slice,
                        out_slice,
                        &mut dropout_masks[li][..mask_len],
                        &mut batch_indices[..bs],
                        &mut scratches[li],
                        bs,
                        true,
                        &backend,
                    )?;
                }

                let logits = &acts[self.layers.len()][..bs * out_dim];
                let d_loss_slice = &mut grad_a[..bs * out_dim];
                let batch_loss = match loss {
                    LossKind::CrossEntropy => {
                        let (l, dl) = backend.cross_entropy_grad_f32(
                            logits,
                            &batch_y_class[..bs],
                            bs,
                            out_dim,
                            0.0,
                            None,
                        )?;
                        d_loss_slice.copy_from_slice(&dl);
                        l
                    }
                    LossKind::MSE => {
                        mse.forward_into(logits, &batch_y_reg[..bs * out_dim], d_loss_slice)?
                    }
                };
                epoch_loss += (batch_loss as f64) * (bs as f64);

                for g in &mut param_grads {
                    g.fill(0.0);
                }

                let mut cur_is_a = true;
                let mut param_cursor = plan.param_ranges.len();

                for li in (0..self.layers.len()).rev() {
                    let in_st = if li == 0 {
                        plan.in_state
                    } else {
                        plan.layer_out_states[li - 1]
                    };
                    let out_st = plan.layer_out_states[li];
                    let in_d = in_st.elems();
                    let out_d = out_st.elems();
                    let (dy_slice, dx_slice) = if cur_is_a {
                        (&grad_a[..bs * out_d], &mut grad_b[..bs * in_d])
                    } else {
                        (&grad_b[..bs * out_d], &mut grad_a[..bs * in_d])
                    };

                    let need_dx = li > 0;
                    let n_params = layer_param_count(&self.layers[li]);
                    param_cursor -= n_params;
                    let p_grads_slice = &mut param_grads[param_cursor..param_cursor + n_params];

                    backward_layer_raw(
                        &self.layers[li],
                        in_st,
                        out_st,
                        &acts[li][..bs * in_d],
                        &acts[li + 1][..bs * out_d],
                        dy_slice,
                        if need_dx { Some(dx_slice) } else { None },
                        p_grads_slice,
                        &dropout_masks[li][..if dropout_masks[li].is_empty() {
                            0
                        } else {
                            bs * in_d
                        }],
                        &batch_indices[..bs],
                        &mut scratches[li],
                        bs,
                        &self.device,
                    )?;
                    cur_is_a = !cur_is_a;
                }

                if let Some(max_norm) = grad_clip {
                    let mut grad_refs: Vec<&mut [f32]> =
                        param_grads.iter_mut().map(Vec::as_mut_slice).collect();
                    clip_grad_norm_(&mut grad_refs, max_norm);
                }

                let grad_refs: Vec<&[f32]> = param_grads.iter().map(Vec::as_slice).collect();
                if self.dtype != DType::F32 {
                    let mut p_refs: Vec<&mut [f32]> = master_weights
                        .iter_mut()
                        .filter_map(|m| m.as_mut().map(Vec::as_mut_slice))
                        .collect();
                    opt_runner.step_slices(&mut p_refs, &grad_refs)?;
                    for (pi, &(layer_idx, sub_idx, _)) in plan.param_ranges.iter().enumerate() {
                        if let Some(ref master) = master_weights[pi] {
                            let q = TensorStorage::from_f32_slice(master, self.dtype).to_f32_vec();
                            let dst = layer_param_slices_mut(&mut self.layers[layer_idx])
                                .swap_remove(sub_idx);
                            dst.copy_from_slice(&q);
                        }
                    }
                } else {
                    let mut p_refs = collect_param_slices_mut(&mut self.layers);
                    opt_runner.step_slices(&mut p_refs, &grad_refs)?;
                }
            }

            loss_history.push((epoch_loss / n_samples as f64) as f32);
        }

        let _ = plan.dropout_indices.len();
        Ok(loss_history)
    }

    /// Runs batched inference in eval mode (no dropout) for flat `[n_samples, in_dim]` inputs.
    pub fn predict(
        &self,
        x: &[f32],
        n_samples: usize,
        in_dim: usize,
        batch_size: usize,
    ) -> OchreResult<(Vec<f32>, usize)> {
        self.predict_nd(x, n_samples, &[in_dim], batch_size)
    }

    /// Runs batched inference in eval mode supporting `[in_dim]` and `[C, H, W]` per-sample shapes.
    pub fn predict_nd(
        &self,
        x: &[f32],
        n_samples: usize,
        sample_shape: &[usize],
        batch_size: usize,
    ) -> OchreResult<(Vec<f32>, usize)> {
        let in_state = parse_sample_shape(sample_shape)?;
        let in_dim = in_state.elems();
        if n_samples == 0 || in_dim == 0 || batch_size == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.len() != n_samples * in_dim {
            return Err(OchreError::DimensionMismatch {
                expected_rows: n_samples,
                expected_cols: in_dim,
                actual_rows: x.len(),
                actual_cols: 1,
            });
        }
        let plan = build_plan(&self.layers, in_state)?;
        let out_dim = plan.out_dim;
        let max_bs = batch_size.min(n_samples);
        let backend = get_backend(&self.device);

        let mut buf_a = vec![0.0f32; max_bs * plan.max_dim.max(in_dim)];
        let mut buf_b = vec![0.0f32; max_bs * plan.max_dim.max(in_dim)];
        let mut batch_indices = vec![0usize; max_bs];
        let mut scratch = LayerScratch::default();
        let mut outputs = vec![0.0f32; n_samples * out_dim];
        let mut cloned_layers = self.layers.clone();
        for l in &mut cloned_layers {
            match l {
                SequentialLayer::Dropout(d) => d.training = false,
                SequentialLayer::BatchNorm2d(bn) => bn.eval(),
                SequentialLayer::ResidualBlock(rb) => rb.eval(),
                _ => {}
            }
        }

        for start in (0..n_samples).step_by(max_bs) {
            let end = (start + max_bs).min(n_samples);
            let bs = end - start;
            buf_a[..bs * in_dim].copy_from_slice(&x[start * in_dim..end * in_dim]);
            let mut cur_is_a = true;

            for (li, layer) in cloned_layers.iter_mut().enumerate() {
                let in_st = if li == 0 {
                    plan.in_state
                } else {
                    plan.layer_out_states[li - 1]
                };
                let out_st = plan.layer_out_states[li];
                let in_d = in_st.elems();
                let out_d = out_st.elems();
                let (in_slice, out_slice) = if cur_is_a {
                    (&buf_a[..bs * in_d], &mut buf_b[..bs * out_d])
                } else {
                    (&buf_b[..bs * in_d], &mut buf_a[..bs * out_d])
                };
                forward_layer_raw(
                    layer,
                    in_st,
                    out_st,
                    in_slice,
                    out_slice,
                    &mut [],
                    &mut batch_indices[..bs],
                    &mut scratch,
                    bs,
                    false,
                    &backend,
                )?;
                cur_is_a = !cur_is_a;
            }

            let final_slice = if cur_is_a {
                &buf_a[..bs * out_dim]
            } else {
                &buf_b[..bs * out_dim]
            };
            outputs[start * out_dim..end * out_dim].copy_from_slice(final_slice);
        }

        Ok((outputs, out_dim))
    }

    /// Serializes the `Sequential` model into `DIT\x01` format with `TAG_NN_SEQUENTIAL` (100).
    pub fn to_dit_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_NN_SEQUENTIAL);
        write_u8(&mut buf, dtype_to_tag(self.dtype));
        write_bool(&mut buf, self.training);
        write_usize(&mut buf, self.layers.len());
        for layer in &self.layers {
            match layer {
                SequentialLayer::Linear(lin) => {
                    write_u8(&mut buf, LAYER_TAG_LINEAR);
                    write_usize(&mut buf, lin.in_features);
                    write_usize(&mut buf, lin.out_features);
                    write_f32_vec(&mut buf, &lin.weight.data);
                    write_opt_tensor_data(&mut buf, &lin.bias);
                }
                SequentialLayer::ReLU => write_u8(&mut buf, LAYER_TAG_RELU),
                SequentialLayer::Sigmoid => write_u8(&mut buf, LAYER_TAG_SIGMOID),
                SequentialLayer::Tanh => write_u8(&mut buf, LAYER_TAG_TANH),
                SequentialLayer::GELU => write_u8(&mut buf, LAYER_TAG_GELU),
                SequentialLayer::SiLU => write_u8(&mut buf, LAYER_TAG_SILU),
                SequentialLayer::Dropout(d) => {
                    write_u8(&mut buf, LAYER_TAG_DROPOUT);
                    write_f32(&mut buf, d.p);
                    write_bool(&mut buf, d.training);
                    buf.extend_from_slice(&d.seed.to_le_bytes());
                }
                SequentialLayer::Embedding(emb) => {
                    write_u8(&mut buf, LAYER_TAG_EMBEDDING);
                    write_usize(&mut buf, emb.num_embeddings);
                    write_usize(&mut buf, emb.embedding_dim);
                    write_f32_vec(&mut buf, &emb.weight.data);
                }
                SequentialLayer::LayerNorm(ln) => {
                    write_u8(&mut buf, LAYER_TAG_LAYERNORM);
                    write_usize(&mut buf, ln.normalized_shape);
                    write_f32(&mut buf, ln.eps);
                    write_opt_tensor_data(&mut buf, &ln.gamma);
                    write_opt_tensor_data(&mut buf, &ln.beta);
                }
                SequentialLayer::RMSNorm(rn) => {
                    write_u8(&mut buf, LAYER_TAG_RMSNORM);
                    write_usize(&mut buf, rn.normalized_shape);
                    write_f32(&mut buf, rn.eps);
                    write_f32_vec(&mut buf, &rn.gamma.data);
                }
                SequentialLayer::Conv2d(c) => {
                    write_u8(&mut buf, LAYER_TAG_CONV2D);
                    write_conv2d(&mut buf, c);
                }
                SequentialLayer::DepthwiseConv2d(dw) => {
                    write_u8(&mut buf, LAYER_TAG_DEPTHWISE_CONV2D);
                    write_usize(&mut buf, dw.in_channels);
                    write_usize(&mut buf, dw.depth_multiplier);
                    write_usize(&mut buf, dw.kernel_size.0);
                    write_usize(&mut buf, dw.kernel_size.1);
                    write_usize(&mut buf, dw.stride.0);
                    write_usize(&mut buf, dw.stride.1);
                    write_usize(&mut buf, dw.padding.0);
                    write_usize(&mut buf, dw.padding.1);
                    write_usize(&mut buf, dw.dilation.0);
                    write_usize(&mut buf, dw.dilation.1);
                    write_f32_vec(&mut buf, &dw.weight.data);
                    write_opt_tensor_data(&mut buf, &dw.bias);
                }
                SequentialLayer::MaxPool2d(p) => {
                    write_u8(&mut buf, LAYER_TAG_MAXPOOL2D);
                    write_usize(&mut buf, p.kernel_size.0);
                    write_usize(&mut buf, p.kernel_size.1);
                    write_usize(&mut buf, p.stride.0);
                    write_usize(&mut buf, p.stride.1);
                    write_usize(&mut buf, p.padding.0);
                    write_usize(&mut buf, p.padding.1);
                }
                SequentialLayer::AvgPool2d(p) => {
                    write_u8(&mut buf, LAYER_TAG_AVGPOOL2D);
                    write_usize(&mut buf, p.kernel_size.0);
                    write_usize(&mut buf, p.kernel_size.1);
                    write_usize(&mut buf, p.stride.0);
                    write_usize(&mut buf, p.stride.1);
                    write_usize(&mut buf, p.padding.0);
                    write_usize(&mut buf, p.padding.1);
                }
                SequentialLayer::GlobalAvgPool2d(gap) => {
                    write_u8(&mut buf, LAYER_TAG_GLOBAL_AVGPOOL2D);
                    write_bool(&mut buf, gap.keep_dims);
                }
                SequentialLayer::BatchNorm2d(bn) => {
                    write_u8(&mut buf, LAYER_TAG_BATCHNORM2D);
                    write_batchnorm2d(&mut buf, bn);
                }
                SequentialLayer::Flatten(fl) => {
                    write_u8(&mut buf, LAYER_TAG_FLATTEN);
                    write_usize(&mut buf, fl.start_dim);
                }
                SequentialLayer::ResidualBlock(rb) => {
                    write_u8(&mut buf, LAYER_TAG_RESIDUAL_BLOCK);
                    write_usize(&mut buf, rb.in_channels);
                    write_usize(&mut buf, rb.out_channels);
                    write_usize(&mut buf, rb.stride.0);
                    write_usize(&mut buf, rb.stride.1);
                    write_conv2d(&mut buf, &rb.conv1);
                    write_batchnorm2d(&mut buf, &rb.bn1);
                    write_conv2d(&mut buf, &rb.conv2);
                    write_batchnorm2d(&mut buf, &rb.bn2);
                    match &rb.downsample {
                        Some((ds_c, ds_bn)) => {
                            write_bool(&mut buf, true);
                            write_conv2d(&mut buf, ds_c);
                            write_batchnorm2d(&mut buf, ds_bn);
                        }
                        None => write_bool(&mut buf, false),
                    }
                }
            }
        }
        buf
    }

    /// Deserializes a `Sequential` model from `DIT\x01` bytes onto `device`.
    pub fn from_dit_bytes(bytes: &[u8], device: DeviceType) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_NN_SEQUENTIAL)?;
        let dtype = tag_to_dtype(r.read_u8()?)?;
        let training = r.read_bool()?;
        let num_layers = r.read_usize()?;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let ltag = r.read_u8()?;
            let layer = match ltag {
                LAYER_TAG_LINEAR => {
                    let in_features = r.read_usize()?;
                    let out_features = r.read_usize()?;
                    let w_data = r.read_f32_vec()?;
                    let weight = Tensor::parameter(
                        w_data,
                        vec![out_features, in_features],
                        dtype,
                        device.clone(),
                    )?;
                    let bias = read_opt_tensor(&mut r, vec![out_features], dtype, &device)?;
                    SequentialLayer::Linear(Linear::from_weights(weight, bias)?)
                }
                LAYER_TAG_RELU => SequentialLayer::ReLU,
                LAYER_TAG_SIGMOID => SequentialLayer::Sigmoid,
                LAYER_TAG_TANH => SequentialLayer::Tanh,
                LAYER_TAG_GELU => SequentialLayer::GELU,
                LAYER_TAG_SILU => SequentialLayer::SiLU,
                LAYER_TAG_DROPOUT => {
                    let p = r.read_f32()?;
                    let d_training = r.read_bool()?;
                    let seed_bytes = r.read_exact(8)?;
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(seed_bytes);
                    let seed = u64::from_le_bytes(arr);
                    let mut d = Dropout::new(p, seed)?;
                    d.training = d_training;
                    SequentialLayer::Dropout(d)
                }
                LAYER_TAG_EMBEDDING => {
                    let num_embeddings = r.read_usize()?;
                    let embedding_dim = r.read_usize()?;
                    let w_data = r.read_f32_vec()?;
                    let weight = Tensor::parameter(
                        w_data,
                        vec![num_embeddings, embedding_dim],
                        dtype,
                        device.clone(),
                    )?;
                    SequentialLayer::Embedding(Embedding::from_weight(weight)?)
                }
                LAYER_TAG_LAYERNORM => {
                    let normalized_shape = r.read_usize()?;
                    let eps = r.read_f32()?;
                    let gamma = read_opt_tensor(&mut r, vec![normalized_shape], dtype, &device)?;
                    let beta = read_opt_tensor(&mut r, vec![normalized_shape], dtype, &device)?;
                    SequentialLayer::LayerNorm(LayerNorm::from_weights(
                        normalized_shape,
                        eps,
                        gamma,
                        beta,
                        dtype,
                        device.clone(),
                    )?)
                }
                LAYER_TAG_RMSNORM => {
                    let normalized_shape = r.read_usize()?;
                    let eps = r.read_f32()?;
                    let g_data = r.read_f32_vec()?;
                    let gamma =
                        Tensor::parameter(g_data, vec![normalized_shape], dtype, device.clone())?;
                    SequentialLayer::RMSNorm(RMSNorm::from_weight(normalized_shape, eps, gamma)?)
                }
                LAYER_TAG_CONV2D => SequentialLayer::Conv2d(read_conv2d(&mut r, dtype, &device)?),
                LAYER_TAG_DEPTHWISE_CONV2D => {
                    let in_channels = r.read_usize()?;
                    let depth_multiplier = r.read_usize()?;
                    let kh = r.read_usize()?;
                    let kw = r.read_usize()?;
                    let sh = r.read_usize()?;
                    let sw = r.read_usize()?;
                    let ph = r.read_usize()?;
                    let pw = r.read_usize()?;
                    let dh = r.read_usize()?;
                    let dw_dil = r.read_usize()?;
                    let out_channels = in_channels
                        .checked_mul(depth_multiplier)
                        .ok_or(OchreError::CorruptedDitFile)?;
                    let w_data = r.read_f32_vec()?;
                    let weight = Tensor::parameter(
                        w_data,
                        vec![out_channels, 1, kh, kw],
                        dtype,
                        device.clone(),
                    )?;
                    let bias = read_opt_tensor(&mut r, vec![out_channels], dtype, &device)?;
                    if in_channels == 0
                        || depth_multiplier == 0
                        || kh == 0
                        || kw == 0
                        || sh == 0
                        || sw == 0
                        || dh == 0
                        || dw_dil == 0
                    {
                        return Err(OchreError::CorruptedDitFile);
                    }
                    SequentialLayer::DepthwiseConv2d(DepthwiseConv2d {
                        weight,
                        bias,
                        in_channels,
                        depth_multiplier,
                        out_channels,
                        kernel_size: (kh, kw),
                        stride: (sh, sw),
                        padding: (ph, pw),
                        dilation: (dh, dw_dil),
                        dtype,
                        device: device.clone(),
                    })
                }
                LAYER_TAG_MAXPOOL2D => {
                    let kh = r.read_usize()?;
                    let kw = r.read_usize()?;
                    let sh = r.read_usize()?;
                    let sw = r.read_usize()?;
                    let ph = r.read_usize()?;
                    let pw = r.read_usize()?;
                    SequentialLayer::MaxPool2d(MaxPool2d::new((kh, kw), (sh, sw), (ph, pw))?)
                }
                LAYER_TAG_AVGPOOL2D => {
                    let kh = r.read_usize()?;
                    let kw = r.read_usize()?;
                    let sh = r.read_usize()?;
                    let sw = r.read_usize()?;
                    let ph = r.read_usize()?;
                    let pw = r.read_usize()?;
                    SequentialLayer::AvgPool2d(AvgPool2d::new((kh, kw), (sh, sw), (ph, pw))?)
                }
                LAYER_TAG_GLOBAL_AVGPOOL2D => {
                    let keep_dims = r.read_bool()?;
                    SequentialLayer::GlobalAvgPool2d(GlobalAvgPool2d::with_keep_dims(keep_dims))
                }
                LAYER_TAG_BATCHNORM2D => {
                    SequentialLayer::BatchNorm2d(read_batchnorm2d(&mut r, dtype, &device)?)
                }
                LAYER_TAG_FLATTEN => {
                    let start_dim = r.read_usize()?;
                    SequentialLayer::Flatten(Flatten::with_start_dim(start_dim))
                }
                LAYER_TAG_RESIDUAL_BLOCK => {
                    let in_channels = r.read_usize()?;
                    let out_channels = r.read_usize()?;
                    let sh = r.read_usize()?;
                    let sw = r.read_usize()?;
                    let conv1 = read_conv2d(&mut r, dtype, &device)?;
                    let bn1 = read_batchnorm2d(&mut r, dtype, &device)?;
                    let conv2 = read_conv2d(&mut r, dtype, &device)?;
                    let bn2 = read_batchnorm2d(&mut r, dtype, &device)?;
                    let has_ds = r.read_bool()?;
                    let downsample = if has_ds {
                        let ds_c = read_conv2d(&mut r, dtype, &device)?;
                        let ds_bn = read_batchnorm2d(&mut r, dtype, &device)?;
                        Some((ds_c, ds_bn))
                    } else {
                        None
                    };
                    SequentialLayer::ResidualBlock(ResidualBlock {
                        conv1,
                        bn1,
                        conv2,
                        bn2,
                        downsample,
                        in_channels,
                        out_channels,
                        stride: (sh, sw),
                    })
                }
                _ => return Err(OchreError::CorruptedDitFile),
            };
            layers.push(layer);
        }

        Ok(Self {
            layers,
            device,
            dtype,
            training,
        })
    }
}

fn parse_sample_shape(sample_shape: &[usize]) -> OchreResult<DimState> {
    match sample_shape {
        [d] if *d > 0 => Ok(DimState::Flat(*d)),
        [c, h, w] if *c > 0 && *h > 0 && *w > 0 => Ok(DimState::Nchw {
            c: *c,
            h: *h,
            w: *w,
        }),
        _ => Err(OchreError::InvalidParameter(format!(
            "Sequential expects per-sample input shape [D] or [C, H, W] with positive dims, got {:?} ( >_< )",
            sample_shape
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_fit_inputs(
    layers: &[SequentialLayer],
    x: &[f32],
    n_samples: usize,
    in_dim: usize,
    y_class: Option<&[usize]>,
    y_reg: Option<&[f32]>,
    out_dim: usize,
    epochs: usize,
    batch_size: usize,
    loss: LossKind,
) -> OchreResult<()> {
    if layers.is_empty() || n_samples == 0 || in_dim == 0 || out_dim == 0 {
        return Err(OchreError::EmptyData);
    }
    if epochs == 0 || batch_size == 0 {
        return Err(OchreError::InvalidParameter(
            "epochs and batch_size must be > 0 ( >_< )".to_string(),
        ));
    }
    if x.len() != n_samples * in_dim {
        return Err(OchreError::DimensionMismatch {
            expected_rows: n_samples,
            expected_cols: in_dim,
            actual_rows: x.len(),
            actual_cols: 1,
        });
    }
    match loss {
        LossKind::CrossEntropy => {
            let yc = y_class.ok_or_else(|| {
                OchreError::InvalidParameter(
                    "CrossEntropy loss requires y_class targets ( >_< )".to_string(),
                )
            })?;
            if yc.len() != n_samples {
                return Err(OchreError::LengthMismatch {
                    features: n_samples,
                    targets: yc.len(),
                });
            }
            for &cls in yc {
                if cls >= out_dim {
                    return Err(OchreError::InvalidParameter(format!(
                        "Class label {cls} exceeds out_dim={out_dim} ( >_< )"
                    )));
                }
            }
        }
        LossKind::MSE => {
            let yr = y_reg.ok_or_else(|| {
                OchreError::InvalidParameter("MSE loss requires y_reg targets ( >_< )".to_string())
            })?;
            if yr.len() != n_samples * out_dim {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: n_samples,
                    expected_cols: out_dim,
                    actual_rows: yr.len(),
                    actual_cols: 1,
                });
            }
        }
    }
    Ok(())
}

fn build_plan(layers: &[SequentialLayer], in_state: DimState) -> OchreResult<SequentialPlan> {
    let mut cur = in_state;
    let mut max_dim = in_state.elems();
    let mut layer_out_states = Vec::with_capacity(layers.len());
    let mut param_ranges = Vec::new();
    let mut dropout_indices = Vec::new();

    for (idx, layer) in layers.iter().enumerate() {
        cur = layer.output_state(cur)?;
        max_dim = max_dim.max(cur.elems());
        layer_out_states.push(cur);
        for (sub_idx, p) in layer_param_slices(layer).into_iter().enumerate() {
            param_ranges.push((idx, sub_idx, p.len()));
        }
        if matches!(layer, SequentialLayer::Dropout(_)) {
            dropout_indices.push(idx);
        }
    }

    Ok(SequentialPlan {
        in_state,
        layer_out_states,
        max_dim,
        out_dim: cur.elems(),
        param_ranges,
        dropout_indices,
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_layer_raw(
    layer: &mut SequentialLayer,
    in_st: DimState,
    out_st: DimState,
    in_slice: &[f32],
    out_slice: &mut [f32],
    mask_slice: &mut [f32],
    indices_buf: &mut [usize],
    scratch: &mut LayerScratch,
    bs: usize,
    training: bool,
    backend: &Backend<'_>,
) -> OchreResult<()> {
    match layer {
        SequentialLayer::Linear(lin) => lin.forward_raw(in_slice, bs, out_slice),
        SequentialLayer::Dropout(drop) => {
            drop.forward_raw(in_slice, out_slice, mask_slice, training)
        }
        SequentialLayer::Embedding(emb) => {
            for (dst, &val) in indices_buf.iter_mut().zip(in_slice.iter()) {
                *dst = val.max(0.0) as usize;
            }
            emb.forward_raw(indices_buf, out_slice)
        }
        SequentialLayer::ReLU => {
            Activation::ReLU.forward_raw(in_slice, out_slice, &DeviceType::Cpu)
        }
        SequentialLayer::Sigmoid => {
            Activation::Sigmoid.forward_raw(in_slice, out_slice, &DeviceType::Cpu)
        }
        SequentialLayer::Tanh => {
            Activation::Tanh.forward_raw(in_slice, out_slice, &DeviceType::Cpu)
        }
        SequentialLayer::GELU => {
            let res = backend.gelu_f32(in_slice);
            out_slice.copy_from_slice(&res);
            Ok(())
        }
        SequentialLayer::SiLU => {
            let res = backend.silu_f32(in_slice);
            out_slice.copy_from_slice(&res);
            Ok(())
        }
        SequentialLayer::LayerNorm(ln) => {
            let rows = (bs * in_st.elems()) / ln.normalized_shape;
            let (out, mean, rstd) = ln.forward_raw(in_slice, rows)?;
            out_slice.copy_from_slice(&out);
            scratch.mean = mean;
            scratch.inv_std = rstd;
            Ok(())
        }
        SequentialLayer::RMSNorm(rn) => {
            let rows = (bs * in_st.elems()) / rn.normalized_shape;
            let (out, rstd) = rn.forward_raw(in_slice, rows)?;
            out_slice.copy_from_slice(&out);
            scratch.inv_std = rstd;
            Ok(())
        }
        SequentialLayer::Conv2d(conv) => {
            let (_, h, w) = in_st.require_nchw("Conv2d")?;
            conv.forward_raw(in_slice, bs, h, w, out_slice)
        }
        SequentialLayer::DepthwiseConv2d(dw) => {
            let (_, h, w) = in_st.require_nchw("DepthwiseConv2d")?;
            dw.forward_raw(in_slice, bs, h, w, out_slice)
        }
        SequentialLayer::MaxPool2d(pool) => {
            let (c, h, w) = in_st.require_nchw("MaxPool2d")?;
            let argmax =
                pool.forward_raw(in_slice, bs, c, h, w, out_slice, &backend_dev(backend))?;
            scratch.argmax = argmax;
            Ok(())
        }
        SequentialLayer::AvgPool2d(pool) => {
            let (c, h, w) = in_st.require_nchw("AvgPool2d")?;
            pool.forward_raw(in_slice, bs, c, h, w, out_slice, &backend_dev(backend))
        }
        SequentialLayer::GlobalAvgPool2d(gap) => {
            let (c, h, w) = in_st.require_nchw("GlobalAvgPool2d")?;
            gap.forward_raw(in_slice, bs, c, h, w, out_slice)
        }
        SequentialLayer::BatchNorm2d(bn) => {
            let (_, h, w) = in_st.require_nchw("BatchNorm2d")?;
            let (mean, inv_std) = bn.forward_raw(in_slice, bs, h, w, out_slice)?;
            scratch.mean = mean;
            scratch.inv_std = inv_std;
            Ok(())
        }
        SequentialLayer::Flatten(_) => {
            out_slice.copy_from_slice(in_slice);
            Ok(())
        }
        SequentialLayer::ResidualBlock(rb) => {
            let (_, in_h, in_w) = in_st.require_nchw("ResidualBlock")?;
            let (_, out_h, out_w) = out_st.require_nchw("ResidualBlock")?;
            let mid_len = bs * rb.out_channels * out_h * out_w;
            scratch.c1_out.resize(mid_len, 0.0);
            scratch.b1_out.resize(mid_len, 0.0);
            scratch.r1_out.resize(mid_len, 0.0);
            scratch.c2_out.resize(mid_len, 0.0);
            scratch.sum_out.resize(mid_len, 0.0);

            rb.conv1
                .forward_raw(in_slice, bs, in_h, in_w, &mut scratch.c1_out)?;
            let (m1, i1) =
                rb.bn1
                    .forward_raw(&scratch.c1_out, bs, out_h, out_w, &mut scratch.b1_out)?;
            scratch.b1_mean = m1;
            scratch.b1_inv = i1;
            for (dst, &v) in scratch.r1_out.iter_mut().zip(scratch.b1_out.iter()) {
                *dst = v.max(0.0);
            }
            rb.conv2
                .forward_raw(&scratch.r1_out, bs, out_h, out_w, &mut scratch.c2_out)?;
            let (m2, i2) =
                rb.bn2
                    .forward_raw(&scratch.c2_out, bs, out_h, out_w, &mut scratch.sum_out)?;
            scratch.b2_mean = m2;
            scratch.b2_inv = i2;

            if let Some((ref ds_c, ref mut ds_bn)) = rb.downsample {
                scratch.ds_c_out.resize(mid_len, 0.0);
                scratch.d_tmp1.resize(mid_len, 0.0);
                ds_c.forward_raw(in_slice, bs, in_h, in_w, &mut scratch.ds_c_out)?;
                let (md, id) =
                    ds_bn.forward_raw(&scratch.ds_c_out, bs, out_h, out_w, &mut scratch.d_tmp1)?;
                scratch.ds_b_mean = md;
                scratch.ds_b_inv = id;
                for (s, &idv) in scratch.sum_out.iter_mut().zip(scratch.d_tmp1.iter()) {
                    *s += idv;
                }
            } else {
                for (s, &idv) in scratch.sum_out.iter_mut().zip(in_slice.iter()) {
                    *s += idv;
                }
            }

            for (o, &s) in out_slice.iter_mut().zip(scratch.sum_out.iter()) {
                *o = s.max(0.0);
            }
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_layer_raw(
    layer: &SequentialLayer,
    in_st: DimState,
    out_st: DimState,
    x_act: &[f32],
    y_act: &[f32],
    dy_slice: &[f32],
    mut dx_slice: Option<&mut [f32]>,
    param_grads: &mut [Vec<f32>],
    dropout_mask: &[f32],
    batch_indices: &[usize],
    scratch: &mut LayerScratch,
    bs: usize,
    device: &DeviceType,
) -> OchreResult<()> {
    match layer {
        SequentialLayer::Linear(lin) => {
            let (dw, db) = if lin.bias.is_some() {
                let (w_part, b_part) = param_grads.split_at_mut(1);
                (&mut w_part[0][..], Some(&mut b_part[0][..]))
            } else {
                (&mut param_grads[0][..], None)
            };
            lin.backward_raw(x_act, dy_slice, bs, dx_slice, dw, db)
        }
        SequentialLayer::Embedding(emb) => {
            emb.backward_raw(batch_indices, dy_slice, &mut param_grads[0])
        }
        SequentialLayer::Dropout(drop) => {
            if let Some(dx) = dx_slice {
                drop.backward_raw(dy_slice, dropout_mask, dx)?;
            }
            Ok(())
        }
        SequentialLayer::ReLU
        | SequentialLayer::Sigmoid
        | SequentialLayer::Tanh
        | SequentialLayer::GELU
        | SequentialLayer::SiLU => {
            if let Some(dx) = dx_slice
                && let Some(act) = layer.as_activation()
            {
                act.backward_raw(x_act, y_act, dy_slice, dx, device)?;
            }
            Ok(())
        }
        SequentialLayer::LayerNorm(ln) => {
            let rows = (bs * in_st.elems()) / ln.normalized_shape;
            let (dx, dgamma, dbeta) =
                ln.backward_raw(dy_slice, x_act, &scratch.mean, &scratch.inv_std, rows)?;
            if let Some(dx_buf) = dx_slice {
                dx_buf.copy_from_slice(&dx);
            }
            let mut cursor = 0usize;
            if ln.gamma.is_some() {
                param_grads[cursor].copy_from_slice(&dgamma);
                cursor += 1;
            }
            if ln.beta.is_some() {
                param_grads[cursor].copy_from_slice(&dbeta);
            }
            Ok(())
        }
        SequentialLayer::RMSNorm(rn) => {
            let rows = (bs * in_st.elems()) / rn.normalized_shape;
            let (dx, dgamma) = rn.backward_raw(dy_slice, x_act, &scratch.inv_std, rows)?;
            if let Some(dx_buf) = dx_slice {
                dx_buf.copy_from_slice(&dx);
            }
            param_grads[0].copy_from_slice(&dgamma);
            Ok(())
        }
        SequentialLayer::Conv2d(conv) => {
            let (_, h, w) = in_st.require_nchw("Conv2d")?;
            let (dw, db) = if conv.bias.is_some() {
                let (w_part, b_part) = param_grads.split_at_mut(1);
                (&mut w_part[0][..], Some(&mut b_part[0][..]))
            } else {
                (&mut param_grads[0][..], None)
            };
            conv.backward_raw(x_act, dy_slice, bs, h, w, dx_slice, dw, db)
        }
        SequentialLayer::DepthwiseConv2d(dw_layer) => {
            let (_, h, w) = in_st.require_nchw("DepthwiseConv2d")?;
            let (dw, db) = if dw_layer.bias.is_some() {
                let (w_part, b_part) = param_grads.split_at_mut(1);
                (&mut w_part[0][..], Some(&mut b_part[0][..]))
            } else {
                (&mut param_grads[0][..], None)
            };
            dw_layer.backward_raw(x_act, dy_slice, bs, h, w, dx_slice, dw, db)
        }
        SequentialLayer::MaxPool2d(pool) => {
            if let Some(dx) = dx_slice {
                pool.backward_raw(dy_slice, &scratch.argmax, dx, device)?;
            }
            Ok(())
        }
        SequentialLayer::AvgPool2d(pool) => {
            if let Some(dx) = dx_slice {
                let (c, in_h, in_w) = in_st.require_nchw("AvgPool2d")?;
                let (_, out_h, out_w) = out_st.require_nchw("AvgPool2d")?;
                pool.backward_raw(dy_slice, bs, c, in_h, in_w, out_h, out_w, dx, device)?;
            }
            Ok(())
        }
        SequentialLayer::GlobalAvgPool2d(gap) => {
            if let Some(dx) = dx_slice {
                let (c, in_h, in_w) = in_st.require_nchw("GlobalAvgPool2d")?;
                gap.backward_raw(dy_slice, bs, c, in_h, in_w, dx)?;
            }
            Ok(())
        }
        SequentialLayer::BatchNorm2d(bn) => {
            let (_, h, w) = in_st.require_nchw("BatchNorm2d")?;
            let (dg, db) = if bn.affine {
                let (w_part, b_part) = param_grads.split_at_mut(1);
                (Some(&mut w_part[0][..]), Some(&mut b_part[0][..]))
            } else {
                (None, None)
            };
            bn.backward_raw(
                x_act,
                dy_slice,
                &scratch.mean,
                &scratch.inv_std,
                bs,
                h,
                w,
                dx_slice,
                dg,
                db,
            )
        }
        SequentialLayer::Flatten(_) => {
            if let Some(dx) = dx_slice {
                dx.copy_from_slice(dy_slice);
            }
            Ok(())
        }
        SequentialLayer::ResidualBlock(rb) => {
            let (_, in_h, in_w) = in_st.require_nchw("ResidualBlock")?;
            let (_, out_h, out_w) = out_st.require_nchw("ResidualBlock")?;
            let mid_len = bs * rb.out_channels * out_h * out_w;
            let in_len = bs * rb.in_channels * in_h * in_w;
            scratch.d_sum.resize(mid_len, 0.0);
            scratch.d_tmp1.resize(mid_len, 0.0);
            scratch.d_tmp2.resize(mid_len, 0.0);

            for ((dst, &g), &s) in scratch
                .d_sum
                .iter_mut()
                .zip(dy_slice.iter())
                .zip(scratch.sum_out.iter())
            {
                *dst = if s > 0.0 { g } else { 0.0 };
            }

            // Parameter layout matches ResidualBlock::parameters_mut:
            // conv1(w[,b]), bn1([w,b]), conv2(w[,b]), bn2([w,b]), downsample?(ds_conv(w[,b]), ds_bn([w,b]))
            let mut p_idx = 0usize;
            let c1_w_idx = p_idx;
            p_idx += 1;
            let c1_b_idx = if rb.conv1.bias.is_some() {
                p_idx += 1;
                Some(p_idx - 1)
            } else {
                None
            };
            let bn1_w_idx = if rb.bn1.affine {
                p_idx += 2;
                Some((p_idx - 2, p_idx - 1))
            } else {
                None
            };
            let c2_w_idx = p_idx;
            p_idx += 1;
            let c2_b_idx = if rb.conv2.bias.is_some() {
                p_idx += 1;
                Some(p_idx - 1)
            } else {
                None
            };
            let bn2_w_idx = if rb.bn2.affine {
                p_idx += 2;
                Some((p_idx - 2, p_idx - 1))
            } else {
                None
            };

            // bn2 backward: d_sum -> d_tmp1
            {
                let mut dg_buf = bn2_w_idx.map(|(gi, _)| vec![0.0f32; param_grads[gi].len()]);
                let mut db_buf = bn2_w_idx.map(|(_, bi)| vec![0.0f32; param_grads[bi].len()]);
                rb.bn2.backward_raw(
                    &scratch.c2_out,
                    &scratch.d_sum,
                    &scratch.b2_mean,
                    &scratch.b2_inv,
                    bs,
                    out_h,
                    out_w,
                    Some(&mut scratch.d_tmp1),
                    dg_buf.as_deref_mut(),
                    db_buf.as_deref_mut(),
                )?;
                if let (Some((gi, bi)), Some(dg), Some(db)) = (bn2_w_idx, dg_buf, db_buf) {
                    param_grads[gi].copy_from_slice(&dg);
                    param_grads[bi].copy_from_slice(&db);
                }
            }

            // conv2 backward: d_tmp1 -> d_tmp2
            {
                let mut dw_buf = vec![0.0f32; param_grads[c2_w_idx].len()];
                let mut db_buf = c2_b_idx.map(|bi| vec![0.0f32; param_grads[bi].len()]);
                rb.conv2.backward_raw(
                    &scratch.r1_out,
                    &scratch.d_tmp1,
                    bs,
                    out_h,
                    out_w,
                    Some(&mut scratch.d_tmp2),
                    &mut dw_buf,
                    db_buf.as_deref_mut(),
                )?;
                param_grads[c2_w_idx].copy_from_slice(&dw_buf);
                if let (Some(bi), Some(db)) = (c2_b_idx, db_buf) {
                    param_grads[bi].copy_from_slice(&db);
                }
            }

            // relu1 backward in-place into d_tmp2
            for (d, &b1v) in scratch.d_tmp2.iter_mut().zip(scratch.b1_out.iter()) {
                if b1v <= 0.0 {
                    *d = 0.0;
                }
            }

            // bn1 backward: d_tmp2 -> d_tmp1
            {
                let mut dg_buf = bn1_w_idx.map(|(gi, _)| vec![0.0f32; param_grads[gi].len()]);
                let mut db_buf = bn1_w_idx.map(|(_, bi)| vec![0.0f32; param_grads[bi].len()]);
                rb.bn1.backward_raw(
                    &scratch.c1_out,
                    &scratch.d_tmp2,
                    &scratch.b1_mean,
                    &scratch.b1_inv,
                    bs,
                    out_h,
                    out_w,
                    Some(&mut scratch.d_tmp1),
                    dg_buf.as_deref_mut(),
                    db_buf.as_deref_mut(),
                )?;
                if let (Some((gi, bi)), Some(dg), Some(db)) = (bn1_w_idx, dg_buf, db_buf) {
                    param_grads[gi].copy_from_slice(&dg);
                    param_grads[bi].copy_from_slice(&db);
                }
            }

            // conv1 backward: d_tmp1 -> dx_main (dx_slice)
            {
                let mut dw_buf = vec![0.0f32; param_grads[c1_w_idx].len()];
                let mut db_buf = c1_b_idx.map(|bi| vec![0.0f32; param_grads[bi].len()]);
                rb.conv1.backward_raw(
                    x_act,
                    &scratch.d_tmp1,
                    bs,
                    in_h,
                    in_w,
                    dx_slice.as_deref_mut(),
                    &mut dw_buf,
                    db_buf.as_deref_mut(),
                )?;
                param_grads[c1_w_idx].copy_from_slice(&dw_buf);
                if let (Some(bi), Some(db)) = (c1_b_idx, db_buf) {
                    param_grads[bi].copy_from_slice(&db);
                }
            }

            // shortcut branch
            if let Some((ref ds_c, ref ds_bn)) = rb.downsample {
                let dsc_w_idx = p_idx;
                p_idx += 1;
                let dsc_b_idx = if ds_c.bias.is_some() {
                    p_idx += 1;
                    Some(p_idx - 1)
                } else {
                    None
                };
                let dsbn_w_idx = if ds_bn.affine {
                    Some((p_idx, p_idx + 1))
                } else {
                    None
                };

                let mut dg_buf = dsbn_w_idx.map(|(gi, _)| vec![0.0f32; param_grads[gi].len()]);
                let mut db_buf = dsbn_w_idx.map(|(_, bi)| vec![0.0f32; param_grads[bi].len()]);
                ds_bn.backward_raw(
                    &scratch.ds_c_out,
                    &scratch.d_sum,
                    &scratch.ds_b_mean,
                    &scratch.ds_b_inv,
                    bs,
                    out_h,
                    out_w,
                    Some(&mut scratch.d_tmp1),
                    dg_buf.as_deref_mut(),
                    db_buf.as_deref_mut(),
                )?;
                if let (Some((gi, bi)), Some(dg), Some(db)) = (dsbn_w_idx, dg_buf, db_buf) {
                    param_grads[gi].copy_from_slice(&dg);
                    param_grads[bi].copy_from_slice(&db);
                }

                let mut dw_buf = vec![0.0f32; param_grads[dsc_w_idx].len()];
                let mut db_buf = dsc_b_idx.map(|bi| vec![0.0f32; param_grads[bi].len()]);
                let mut dx_skip = if dx_slice.is_some() {
                    vec![0.0f32; in_len]
                } else {
                    Vec::new()
                };
                ds_c.backward_raw(
                    x_act,
                    &scratch.d_tmp1,
                    bs,
                    in_h,
                    in_w,
                    if dx_slice.is_some() {
                        Some(&mut dx_skip)
                    } else {
                        None
                    },
                    &mut dw_buf,
                    db_buf.as_deref_mut(),
                )?;
                param_grads[dsc_w_idx].copy_from_slice(&dw_buf);
                if let (Some(bi), Some(db)) = (dsc_b_idx, db_buf) {
                    param_grads[bi].copy_from_slice(&db);
                }
                if let Some(dx_buf) = dx_slice {
                    for (dst, &s) in dx_buf.iter_mut().zip(dx_skip.iter()) {
                        *dst += s;
                    }
                }
            } else if let Some(dx_buf) = dx_slice {
                for (dst, &s) in dx_buf.iter_mut().zip(scratch.d_sum.iter()) {
                    *dst += s;
                }
            }
            Ok(())
        }
    }
}

fn backend_dev(backend: &Backend<'_>) -> DeviceType {
    match backend {
        Backend::Cpu => DeviceType::Cpu,
        Backend::CudaSingle(id) => DeviceType::Gpu(*id),
        Backend::CudaMulti(ids) => DeviceType::MultiGpu(ids.to_vec()),
        Backend::Wgpu => DeviceType::Wgpu,
        Backend::Tpu(id) => DeviceType::Tpu(*id),
    }
}

fn gather_batch_x(
    src: &[f32],
    cols: usize,
    perm: &[usize],
    shuffle: bool,
    start: usize,
    dst: &mut [f32],
) {
    if !shuffle {
        let len = perm.len() * cols;
        dst.copy_from_slice(&src[start * cols..start * cols + len]);
    } else {
        for (row_i, &idx) in perm.iter().enumerate() {
            let s = &src[idx * cols..(idx + 1) * cols];
            let d = &mut dst[row_i * cols..(row_i + 1) * cols];
            d.copy_from_slice(s);
        }
    }
}

fn gather_batch_class(
    src: &[usize],
    perm: &[usize],
    shuffle: bool,
    start: usize,
    dst: &mut [usize],
) {
    if !shuffle {
        dst.copy_from_slice(&src[start..start + perm.len()]);
    } else {
        for (i, &idx) in perm.iter().enumerate() {
            dst[i] = src[idx];
        }
    }
}

fn layer_param_count(layer: &SequentialLayer) -> usize {
    layer.parameters().len()
}

fn layer_param_slices(layer: &SequentialLayer) -> Vec<&[f32]> {
    layer
        .parameters()
        .into_iter()
        .map(|t| t.data.as_slice())
        .collect()
}

fn layer_param_slices_mut(layer: &mut SequentialLayer) -> Vec<&mut [f32]> {
    layer
        .parameters_mut()
        .into_iter()
        .map(|t| t.data.as_mut_slice())
        .collect()
}

fn collect_param_slices_mut(layers: &mut [SequentialLayer]) -> Vec<&mut [f32]> {
    let mut out = Vec::new();
    for layer in layers.iter_mut() {
        out.extend(layer_param_slices_mut(layer));
    }
    out
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

fn write_opt_tensor_data(buf: &mut Vec<u8>, t: &Option<Tensor>) {
    match t {
        Some(tensor) => {
            write_bool(buf, true);
            write_f32_vec(buf, &tensor.data);
        }
        None => write_bool(buf, false),
    }
}

fn read_opt_tensor(
    r: &mut ByteReader<'_>,
    shape: Vec<usize>,
    dtype: DType,
    device: &DeviceType,
) -> OchreResult<Option<Tensor>> {
    if r.read_bool()? {
        let data = r.read_f32_vec()?;
        Ok(Some(Tensor::parameter(data, shape, dtype, device.clone())?))
    } else {
        Ok(None)
    }
}

fn write_conv2d(buf: &mut Vec<u8>, c: &Conv2d) {
    write_usize(buf, c.in_channels);
    write_usize(buf, c.out_channels);
    write_usize(buf, c.kernel_size.0);
    write_usize(buf, c.kernel_size.1);
    write_usize(buf, c.stride.0);
    write_usize(buf, c.stride.1);
    write_usize(buf, c.padding.0);
    write_usize(buf, c.padding.1);
    write_usize(buf, c.dilation.0);
    write_usize(buf, c.dilation.1);
    write_usize(buf, c.groups);
    write_f32_vec(buf, &c.weight.data);
    write_opt_tensor_data(buf, &c.bias);
}

fn read_conv2d(r: &mut ByteReader<'_>, dtype: DType, device: &DeviceType) -> OchreResult<Conv2d> {
    let in_channels = r.read_usize()?;
    let out_channels = r.read_usize()?;
    let kh = r.read_usize()?;
    let kw = r.read_usize()?;
    let sh = r.read_usize()?;
    let sw = r.read_usize()?;
    let ph = r.read_usize()?;
    let pw = r.read_usize()?;
    let dh = r.read_usize()?;
    let dw = r.read_usize()?;
    let groups = r.read_usize()?;
    if groups == 0 || !in_channels.is_multiple_of(groups) {
        return Err(OchreError::CorruptedDitFile);
    }
    let in_per_g = in_channels / groups;
    let w_data = r.read_f32_vec()?;
    let weight = Tensor::parameter(
        w_data,
        vec![out_channels, in_per_g, kh, kw],
        dtype,
        device.clone(),
    )?;
    let bias = read_opt_tensor(r, vec![out_channels], dtype, device)?;
    Conv2d::from_weights(weight, bias, (sh, sw), (ph, pw), (dh, dw), groups)
}

fn write_batchnorm2d(buf: &mut Vec<u8>, bn: &BatchNorm2d) {
    write_usize(buf, bn.num_features);
    write_f32(buf, bn.eps);
    write_f32(buf, bn.momentum);
    write_bool(buf, bn.affine);
    write_bool(buf, bn.training);
    write_opt_tensor_data(buf, &bn.weight);
    write_opt_tensor_data(buf, &bn.bias);
    write_f32_vec(buf, &bn.running_mean);
    write_f32_vec(buf, &bn.running_var);
}

fn read_batchnorm2d(
    r: &mut ByteReader<'_>,
    dtype: DType,
    device: &DeviceType,
) -> OchreResult<BatchNorm2d> {
    let num_features = r.read_usize()?;
    let eps = r.read_f32()?;
    let momentum = r.read_f32()?;
    let affine = r.read_bool()?;
    let training = r.read_bool()?;
    let weight = read_opt_tensor(r, vec![num_features], dtype, device)?;
    let bias = read_opt_tensor(r, vec![num_features], dtype, device)?;
    let running_mean = r.read_f32_vec()?;
    let running_var = r.read_f32_vec()?;
    if num_features == 0
        || running_mean.len() != num_features
        || running_var.len() != num_features
        || !eps.is_finite()
        || eps < 0.0
        || !momentum.is_finite()
        || !(0.0..=1.0).contains(&momentum)
    {
        return Err(OchreError::CorruptedDitFile);
    }
    Ok(BatchNorm2d {
        num_features,
        eps,
        momentum,
        affine,
        training,
        weight,
        bias,
        running_mean,
        running_var,
        dtype,
        device: device.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_xor_fit_predict_and_dit_roundtrip() {
        let specs = [
            LayerSpec::Linear {
                in_features: 2,
                out_features: 16,
                bias: true,
            },
            LayerSpec::GELU,
            LayerSpec::Linear {
                in_features: 16,
                out_features: 2,
                bias: true,
            },
        ];
        let mut model = Sequential::from_specs(&specs, DType::F32, DeviceType::Cpu, 42).unwrap();

        let x = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let y_class = [0usize, 1, 1, 0];

        let history = model
            .fit(
                &x,
                4,
                2,
                Some(&y_class),
                None,
                2,
                250,
                4,
                0.05,
                OptimKind::AdamW,
                LossKind::CrossEntropy,
                0.0,
                Some(5.0),
                true,
                7,
            )
            .unwrap();
        assert_eq!(history.len(), 250);
        assert!(history.last().unwrap() < history.first().unwrap());
        assert!(*history.last().unwrap() < 0.2);

        let (preds, out_dim) = model.predict(&x, 4, 2, 2).unwrap();
        assert_eq!(out_dim, 2);
        for (i, &cls) in y_class.iter().enumerate() {
            let p0 = preds[i * 2];
            let p1 = preds[i * 2 + 1];
            let pred_cls = if p1 > p0 { 1 } else { 0 };
            assert_eq!(pred_cls, cls);
        }

        let bytes = model.to_dit_bytes();
        let loaded = Sequential::from_dit_bytes(&bytes, DeviceType::Cpu).unwrap();
        let (preds2, _) = loaded.predict(&x, 4, 2, 4).unwrap();
        for (&a, &b) in preds.iter().zip(preds2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sequential_mse_regression_bf16() {
        let specs = [
            LayerSpec::Linear {
                in_features: 1,
                out_features: 8,
                bias: true,
            },
            LayerSpec::SiLU,
            LayerSpec::Linear {
                in_features: 8,
                out_features: 1,
                bias: true,
            },
        ];
        let mut model = Sequential::from_specs(&specs, DType::BF16, DeviceType::Cpu, 11).unwrap();
        let x = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let y_reg: Vec<f32> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

        let history = model
            .fit(
                &x,
                5,
                1,
                None,
                Some(&y_reg),
                1,
                120,
                5,
                0.05,
                OptimKind::AdamW,
                LossKind::MSE,
                0.0,
                None,
                false,
                1,
            )
            .unwrap();
        assert!(history.last().unwrap() < &0.1);
    }

    #[test]
    fn test_sequential_dropout_embedding_sgd_and_corrupted_dit() {
        let specs = [
            LayerSpec::Embedding {
                num_embeddings: 4,
                embedding_dim: 6,
            },
            LayerSpec::Dropout { p: 0.2 },
            LayerSpec::ReLU,
            LayerSpec::Linear {
                in_features: 6,
                out_features: 2,
                bias: false,
            },
        ];
        let mut model = Sequential::from_specs(&specs, DType::F32, DeviceType::Cpu, 9).unwrap();
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let y = [0usize, 1, 0, 1];
        let hist = model
            .fit(
                &x,
                4,
                1,
                Some(&y),
                None,
                2,
                30,
                2,
                0.1,
                OptimKind::SGD,
                LossKind::CrossEntropy,
                0.0,
                None,
                true,
                3,
            )
            .unwrap();
        assert_eq!(hist.len(), 30);
        assert!(hist.iter().all(|v| v.is_finite()));

        assert!(Sequential::from_dit_bytes(b"BAD!", DeviceType::Cpu).is_err());
    }

    #[test]
    fn test_sequential_cnn_and_norm_fit_predict_dit_roundtrip() {
        let specs = [
            LayerSpec::Conv2d {
                in_channels: 1,
                out_channels: 4,
                kernel_size: (3, 3),
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 1,
                bias: true,
            },
            LayerSpec::BatchNorm2d {
                num_features: 4,
                eps: 1e-5,
                momentum: 0.1,
                affine: true,
            },
            LayerSpec::ReLU,
            LayerSpec::MaxPool2d {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            },
            LayerSpec::ResidualBlock {
                in_channels: 4,
                out_channels: 6,
                stride: (1, 1),
            },
            LayerSpec::GlobalAvgPool2d { keep_dims: false },
            LayerSpec::LayerNorm {
                normalized_shape: 6,
                eps: 1e-5,
                elementwise_affine: true,
            },
            LayerSpec::Linear {
                in_features: 6,
                out_features: 2,
                bias: true,
            },
        ];
        let mut model = Sequential::from_specs(&specs, DType::F32, DeviceType::Cpu, 123).unwrap();
        // 4 images of [1, 4, 4]
        let mut x = Vec::with_capacity(4 * 16);
        let y_class = [0usize, 1, 0, 1];
        for &yc in &y_class {
            let base = if yc == 0 { -1.0f32 } else { 1.0f32 };
            for j in 0..16 {
                x.push(base + (j as f32) * 0.05);
            }
        }

        let hist = model
            .fit_nd(
                &x,
                4,
                &[1, 4, 4],
                Some(&y_class),
                None,
                2,
                25,
                2,
                0.05,
                OptimKind::AdamW,
                LossKind::CrossEntropy,
                0.0,
                Some(2.0),
                true,
                7,
            )
            .unwrap();
        assert_eq!(hist.len(), 25);
        assert!(hist.iter().all(|v| v.is_finite()));
        assert!(hist.last().unwrap() < hist.first().unwrap());

        let (preds, out_dim) = model.predict_nd(&x, 4, &[1, 4, 4], 2).unwrap();
        assert_eq!(out_dim, 2);

        let bytes = model.to_dit_bytes();
        let loaded = Sequential::from_dit_bytes(&bytes, DeviceType::Cpu).unwrap();
        let (preds2, _) = loaded.predict_nd(&x, 4, &[1, 4, 4], 4).unwrap();
        for (&a, &b) in preds.iter().zip(preds2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
