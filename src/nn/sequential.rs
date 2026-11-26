// OchreML Zero-GIL Sequential Trainer & Serialization Module (*≧ω≦*)
// Pre-allocated buffer training loop using fused N8/N9 kernels and DIT\x01 tag 100 persistence.

use crate::backend::{Backend, get_backend};
use crate::device::DeviceType;
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::layers::{Activation, Dropout, Embedding, Linear};
use crate::nn::losses::{MSELoss, Reduction};
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
}

/// Concrete layer enum for `Sequential`, cleanly extensible for Wave 2 CNN/RNN/Transformer layers.
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
}

impl SequentialLayer {
    fn output_dim(&self, current_dim: usize) -> OchreResult<usize> {
        match self {
            SequentialLayer::Linear(lin) => {
                if current_dim != lin.in_features {
                    return Err(OchreError::InvalidParameter(format!(
                        "Layer dimension mismatch: input has {current_dim}, Linear expects {} ( >_< )",
                        lin.in_features
                    )));
                }
                Ok(lin.out_features)
            }
            SequentialLayer::ReLU
            | SequentialLayer::Sigmoid
            | SequentialLayer::Tanh
            | SequentialLayer::GELU
            | SequentialLayer::SiLU
            | SequentialLayer::Dropout(_) => Ok(current_dim),
            SequentialLayer::Embedding(emb) => {
                if current_dim != 1 {
                    return Err(OchreError::InvalidParameter(format!(
                        "Embedding in Sequential expects scalar token index per sample (in_dim=1), got {current_dim} ( >_< )"
                    )));
                }
                Ok(emb.embedding_dim)
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
    layer_out_dims: Vec<usize>,
    max_dim: usize,
    out_dim: usize,
    param_ranges: Vec<(usize, usize, bool)>, // (layer_idx, n_elems, is_bias)
    dropout_indices: Vec<usize>,
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
            };
            layers.push(layer);
        }
        Ok(Self::new(layers, dtype, device))
    }

    /// Sets training mode (`training = true`) across the model and all Dropout layers.
    pub fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            if let SequentialLayer::Dropout(d) = layer {
                d.training = true;
            }
        }
    }

    /// Sets evaluation mode (`training = false`) across the model and all Dropout layers.
    pub fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            if let SequentialLayer::Dropout(d) = layer {
                d.training = false;
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
            };
        }
        Ok(cur)
    }

    /// Zero-GIL, zero-per-batch-allocation Rust training loop returning per-epoch mean loss history.
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

        let plan = build_plan(&self.layers, in_dim)?;
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
        for &d in &plan.layer_out_dims {
            acts.push(vec![0.0f32; max_bs * d]);
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
                        plan.layer_out_dims[idx - 1]
                    };
                    vec![1.0f32; max_bs * dim]
                }
                _ => Vec::new(),
            })
            .collect();

        let mut param_grads: Vec<Vec<f32>> = plan
            .param_ranges
            .iter()
            .map(|&(_, len, _)| vec![0.0f32; len])
            .collect();
        let mut master_weights: Vec<Option<Vec<f32>>> = if self.dtype != DType::F32 {
            plan.param_ranges
                .iter()
                .map(|&(layer_idx, _, is_bias)| {
                    Some(extract_param_slice(&self.layers[layer_idx], is_bias).to_vec())
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

                // Forward pass through pre-allocated activation buffers
                for (li, layer) in self.layers.iter_mut().enumerate() {
                    let in_d = if li == 0 {
                        in_dim
                    } else {
                        plan.layer_out_dims[li - 1]
                    };
                    let out_d = plan.layer_out_dims[li];
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
                        in_slice,
                        out_slice,
                        &mut dropout_masks[li][..mask_len],
                        &mut batch_indices[..bs],
                        bs,
                        true,
                        &backend,
                    )?;
                }

                // Loss + d_logits into grad_a[..bs * out_dim]
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

                // Zero param grad buffers
                for g in &mut param_grads {
                    g.fill(0.0);
                }

                // Backward pass using ping-pong grad buffers (grad_a <-> grad_b)
                let mut cur_is_a = true;
                let mut param_cursor = plan.param_ranges.len();

                for li in (0..self.layers.len()).rev() {
                    let in_d = if li == 0 {
                        in_dim
                    } else {
                        plan.layer_out_dims[li - 1]
                    };
                    let out_d = plan.layer_out_dims[li];
                    let (dy_slice, dx_slice) = if cur_is_a {
                        (&grad_a[..bs * out_d], &mut grad_b[..bs * in_d])
                    } else {
                        (&grad_b[..bs * out_d], &mut grad_a[..bs * in_d])
                    };

                    let need_dx = li > 0;
                    match &self.layers[li] {
                        SequentialLayer::Linear(lin) => {
                            let (dw, db) = if lin.bias.is_some() {
                                param_cursor -= 2;
                                let (w_part, b_part) = param_grads.split_at_mut(param_cursor + 1);
                                (&mut w_part[param_cursor][..], Some(&mut b_part[0][..]))
                            } else {
                                param_cursor -= 1;
                                (&mut param_grads[param_cursor][..], None)
                            };
                            lin.backward_raw(
                                &acts[li][..bs * in_d],
                                dy_slice,
                                bs,
                                if need_dx { Some(dx_slice) } else { None },
                                dw,
                                db,
                            )?;
                        }
                        SequentialLayer::Embedding(emb) => {
                            param_cursor -= 1;
                            emb.backward_raw(
                                &batch_indices[..bs],
                                dy_slice,
                                &mut param_grads[param_cursor],
                            )?;
                        }
                        SequentialLayer::Dropout(drop) => {
                            if need_dx {
                                drop.backward_raw(
                                    dy_slice,
                                    &dropout_masks[li][..bs * in_d],
                                    dx_slice,
                                )?;
                            }
                        }
                        _ => {
                            if need_dx && let Some(act) = self.layers[li].as_activation() {
                                act.backward_raw(
                                    &acts[li][..bs * in_d],
                                    &acts[li + 1][..bs * out_d],
                                    dy_slice,
                                    dx_slice,
                                    &self.device,
                                )?;
                            }
                        }
                    }
                    cur_is_a = !cur_is_a;
                }

                if let Some(max_norm) = grad_clip {
                    let mut grad_refs: Vec<&mut [f32]> =
                        param_grads.iter_mut().map(Vec::as_mut_slice).collect();
                    clip_grad_norm_(&mut grad_refs, max_norm);
                }

                // Optimizer step (on f32 master weights when dtype != F32, else directly on layer data)
                let grad_refs: Vec<&[f32]> = param_grads.iter().map(Vec::as_slice).collect();
                if self.dtype != DType::F32 {
                    let mut p_refs: Vec<&mut [f32]> = master_weights
                        .iter_mut()
                        .filter_map(|m| m.as_mut().map(Vec::as_mut_slice))
                        .collect();
                    opt_runner.step_slices(&mut p_refs, &grad_refs)?;
                    for (pi, &(layer_idx, _, is_bias)) in plan.param_ranges.iter().enumerate() {
                        if let Some(ref master) = master_weights[pi] {
                            let q = TensorStorage::from_f32_slice(master, self.dtype).to_f32_vec();
                            let dst = extract_param_slice_mut(&mut self.layers[layer_idx], is_bias);
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

    /// Runs batched inference in eval mode (no dropout), returning `(flat_outputs, out_dim)`.
    pub fn predict(
        &self,
        x: &[f32],
        n_samples: usize,
        in_dim: usize,
        batch_size: usize,
    ) -> OchreResult<(Vec<f32>, usize)> {
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
        let plan = build_plan(&self.layers, in_dim)?;
        let out_dim = plan.out_dim;
        let max_bs = batch_size.min(n_samples);
        let backend = get_backend(&self.device);

        let mut buf_a = vec![0.0f32; max_bs * plan.max_dim.max(in_dim)];
        let mut buf_b = vec![0.0f32; max_bs * plan.max_dim.max(in_dim)];
        let mut batch_indices = vec![0usize; max_bs];
        let mut outputs = vec![0.0f32; n_samples * out_dim];
        let mut cloned_layers = self.layers.clone();

        for start in (0..n_samples).step_by(max_bs) {
            let end = (start + max_bs).min(n_samples);
            let bs = end - start;
            buf_a[..bs * in_dim].copy_from_slice(&x[start * in_dim..end * in_dim]);
            let mut cur_is_a = true;

            for (li, layer) in cloned_layers.iter_mut().enumerate() {
                let in_d = if li == 0 {
                    in_dim
                } else {
                    plan.layer_out_dims[li - 1]
                };
                let out_d = plan.layer_out_dims[li];
                let (in_slice, out_slice) = if cur_is_a {
                    (&buf_a[..bs * in_d], &mut buf_b[..bs * out_d])
                } else {
                    (&buf_b[..bs * in_d], &mut buf_a[..bs * out_d])
                };
                forward_layer_raw(
                    layer,
                    in_slice,
                    out_slice,
                    &mut [],
                    &mut batch_indices[..bs],
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
                    match &lin.bias {
                        Some(b) => {
                            write_bool(&mut buf, true);
                            write_f32_vec(&mut buf, &b.data);
                        }
                        None => write_bool(&mut buf, false),
                    }
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
                    let w_data = read_f32_vec(&mut r)?;
                    let weight = Tensor::parameter(
                        w_data,
                        vec![out_features, in_features],
                        dtype,
                        device.clone(),
                    )?;
                    let has_bias = r.read_bool()?;
                    let bias = if has_bias {
                        let b_data = read_f32_vec(&mut r)?;
                        Some(Tensor::parameter(
                            b_data,
                            vec![out_features],
                            dtype,
                            device.clone(),
                        )?)
                    } else {
                        None
                    };
                    SequentialLayer::Linear(Linear::from_weights(weight, bias)?)
                }
                LAYER_TAG_RELU => SequentialLayer::ReLU,
                LAYER_TAG_SIGMOID => SequentialLayer::Sigmoid,
                LAYER_TAG_TANH => SequentialLayer::Tanh,
                LAYER_TAG_GELU => SequentialLayer::GELU,
                LAYER_TAG_SILU => SequentialLayer::SiLU,
                LAYER_TAG_DROPOUT => {
                    let p = read_f32(&mut r)?;
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
                    let w_data = read_f32_vec(&mut r)?;
                    let weight = Tensor::parameter(
                        w_data,
                        vec![num_embeddings, embedding_dim],
                        dtype,
                        device.clone(),
                    )?;
                    SequentialLayer::Embedding(Embedding::from_weight(weight)?)
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

fn build_plan(layers: &[SequentialLayer], in_dim: usize) -> OchreResult<SequentialPlan> {
    let mut cur_dim = in_dim;
    let mut max_dim = in_dim;
    let mut layer_out_dims = Vec::with_capacity(layers.len());
    let mut param_ranges = Vec::new();
    let mut dropout_indices = Vec::new();

    for (idx, layer) in layers.iter().enumerate() {
        cur_dim = layer.output_dim(cur_dim)?;
        max_dim = max_dim.max(cur_dim);
        layer_out_dims.push(cur_dim);
        match layer {
            SequentialLayer::Linear(lin) => {
                param_ranges.push((idx, lin.weight.data.len(), false));
                if let Some(ref b) = lin.bias {
                    param_ranges.push((idx, b.data.len(), true));
                }
            }
            SequentialLayer::Embedding(emb) => {
                param_ranges.push((idx, emb.weight.data.len(), false));
            }
            SequentialLayer::Dropout(_) => dropout_indices.push(idx),
            _ => {}
        }
    }

    Ok(SequentialPlan {
        layer_out_dims,
        max_dim,
        out_dim: cur_dim,
        param_ranges,
        dropout_indices,
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_layer_raw(
    layer: &mut SequentialLayer,
    in_slice: &[f32],
    out_slice: &mut [f32],
    mask_slice: &mut [f32],
    indices_buf: &mut [usize],
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

fn extract_param_slice(layer: &SequentialLayer, is_bias: bool) -> &[f32] {
    match layer {
        SequentialLayer::Linear(lin) => {
            if is_bias {
                lin.bias.as_ref().map(|b| b.data.as_slice()).unwrap_or(&[])
            } else {
                &lin.weight.data
            }
        }
        SequentialLayer::Embedding(emb) => &emb.weight.data,
        _ => &[],
    }
}

fn extract_param_slice_mut(layer: &mut SequentialLayer, is_bias: bool) -> &mut [f32] {
    match layer {
        SequentialLayer::Linear(lin) => {
            if is_bias {
                lin.bias
                    .as_mut()
                    .map(|b| b.data.as_mut_slice())
                    .unwrap_or(&mut [])
            } else {
                &mut lin.weight.data
            }
        }
        SequentialLayer::Embedding(emb) => &mut emb.weight.data,
        _ => &mut [],
    }
}

fn collect_param_slices_mut(layers: &mut [SequentialLayer]) -> Vec<&mut [f32]> {
    let mut out = Vec::new();
    for layer in layers.iter_mut() {
        match layer {
            SequentialLayer::Linear(lin) => {
                out.push(lin.weight.data.as_mut_slice());
                if let Some(ref mut b) = lin.bias {
                    out.push(b.data.as_mut_slice());
                }
            }
            SequentialLayer::Embedding(emb) => {
                out.push(emb.weight.data.as_mut_slice());
            }
            _ => {}
        }
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
}
