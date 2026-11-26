// OchreML Arena-Based Autograd Engine (*≧ω≦*)
// Thread-local integer-indexed computation tape with RAII NoGradGuard and eager intermediate cleanup.

use crate::backend::get_backend;
use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::nn::gemm_f32::sgemm;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Thread-safe custom backward closure returning `Vec<(input_node_id, grad_input)>`.
pub type CustomBackwardFn =
    Arc<dyn Fn(&[f32]) -> OchreResult<Vec<(usize, Vec<f32>)>> + Send + Sync>;

/// Wrapper for custom backward closures supporting `Clone` and `Debug`.
#[derive(Clone)]
pub struct CustomBackward(pub CustomBackwardFn);

impl fmt::Debug for CustomBackward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CustomBackward(<fn>)")
    }
}

/// Operations recorded onto the thread-local autograd tape.
#[derive(Clone, Debug)]
pub enum Op {
    Leaf,
    Add {
        a_id: Option<usize>,
        b_id: Option<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        out_shape: Vec<usize>,
    },
    Sub {
        a_id: Option<usize>,
        b_id: Option<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        out_shape: Vec<usize>,
    },
    Mul {
        a_id: Option<usize>,
        b_id: Option<usize>,
        a_data: Vec<f32>,
        b_data: Vec<f32>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        out_shape: Vec<usize>,
    },
    Scale {
        input_id: Option<usize>,
        scalar: f32,
    },
    Matmul {
        a_id: Option<usize>,
        b_id: Option<usize>,
        a_data: Vec<f32>,
        b_data: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        device: DeviceType,
    },
    BatchedMatmul {
        a_id: Option<usize>,
        b_id: Option<usize>,
        a_data: Vec<f32>,
        b_data: Vec<f32>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        device: DeviceType,
    },
    TransposeLast2 {
        input_id: Option<usize>,
        batch: usize,
        rows: usize,
        cols: usize,
        device: DeviceType,
    },
    Reshape {
        input_id: Option<usize>,
        input_shape: Vec<usize>,
    },
    Sum {
        input_id: Option<usize>,
        input_shape: Vec<usize>,
        axis: Option<usize>,
    },
    Mean {
        input_id: Option<usize>,
        input_shape: Vec<usize>,
        axis: Option<usize>,
    },
    ReLU {
        input_id: Option<usize>,
        input_data: Vec<f32>,
    },
    Sigmoid {
        input_id: Option<usize>,
        out_data: Vec<f32>,
    },
    Tanh {
        input_id: Option<usize>,
        out_data: Vec<f32>,
    },
    GELU {
        input_id: Option<usize>,
        input_data: Vec<f32>,
        device: DeviceType,
    },
    SiLU {
        input_id: Option<usize>,
        input_data: Vec<f32>,
        device: DeviceType,
    },
    CrossEntropy {
        logits_id: Option<usize>,
        d_logits: Vec<f32>,
    },
    MSE {
        pred_id: Option<usize>,
        diff: Vec<f32>,
        scale: f32,
    },
    LayerNorm {
        input_id: Option<usize>,
        gamma_id: Option<usize>,
        beta_id: Option<usize>,
        x: Vec<f32>,
        gamma: Option<Vec<f32>>,
        mean: Vec<f32>,
        rstd: Vec<f32>,
        rows: usize,
        cols: usize,
        device: DeviceType,
    },
    RMSNorm {
        input_id: Option<usize>,
        gamma_id: Option<usize>,
        x: Vec<f32>,
        gamma: Option<Vec<f32>>,
        rstd: Vec<f32>,
        rows: usize,
        cols: usize,
        device: DeviceType,
    },
    Conv2d {
        input_id: Option<usize>,
        weight_id: Option<usize>,
        bias_id: Option<usize>,
        batch: usize,
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
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
        col_data: Vec<f32>,
        weight_data: Vec<f32>,
        device: DeviceType,
    },
    Embedding {
        weight_id: Option<usize>,
        indices: Vec<usize>,
        num_embeddings: usize,
        embedding_dim: usize,
    },
    Custom {
        inputs: Vec<Option<usize>>,
        backward: CustomBackward,
    },
}

impl Op {
    fn collect_inputs(&self, out: &mut Vec<usize>) {
        match self {
            Op::Leaf => {}
            Op::Add { a_id, b_id, .. }
            | Op::Sub { a_id, b_id, .. }
            | Op::Mul { a_id, b_id, .. }
            | Op::Matmul { a_id, b_id, .. }
            | Op::BatchedMatmul { a_id, b_id, .. } => {
                if let Some(id) = a_id {
                    out.push(*id);
                }
                if let Some(id) = b_id {
                    out.push(*id);
                }
            }
            Op::Scale { input_id, .. }
            | Op::TransposeLast2 { input_id, .. }
            | Op::Reshape { input_id, .. }
            | Op::Sum { input_id, .. }
            | Op::Mean { input_id, .. }
            | Op::ReLU { input_id, .. }
            | Op::Sigmoid { input_id, .. }
            | Op::Tanh { input_id, .. }
            | Op::GELU { input_id, .. }
            | Op::SiLU { input_id, .. } => {
                if let Some(id) = input_id {
                    out.push(*id);
                }
            }
            Op::CrossEntropy { logits_id, .. } => {
                if let Some(id) = logits_id {
                    out.push(*id);
                }
            }
            Op::MSE { pred_id, .. } => {
                if let Some(id) = pred_id {
                    out.push(*id);
                }
            }
            Op::LayerNorm {
                input_id,
                gamma_id,
                beta_id,
                ..
            } => {
                if let Some(id) = input_id {
                    out.push(*id);
                }
                if let Some(id) = gamma_id {
                    out.push(*id);
                }
                if let Some(id) = beta_id {
                    out.push(*id);
                }
            }
            Op::RMSNorm {
                input_id, gamma_id, ..
            } => {
                if let Some(id) = input_id {
                    out.push(*id);
                }
                if let Some(id) = gamma_id {
                    out.push(*id);
                }
            }
            Op::Conv2d {
                input_id,
                weight_id,
                bias_id,
                ..
            } => {
                if let Some(id) = input_id {
                    out.push(*id);
                }
                if let Some(id) = weight_id {
                    out.push(*id);
                }
                if let Some(id) = bias_id {
                    out.push(*id);
                }
            }
            Op::Embedding { weight_id, .. } => {
                if let Some(id) = weight_id {
                    out.push(*id);
                }
            }
            Op::Custom { inputs, .. } => {
                for id in inputs.iter().flatten() {
                    out.push(*id);
                }
            }
        }
    }
}

/// Gradient storage mapping tensor `node_id` -> flat `Vec<f32>` gradient.
#[derive(Clone, Debug, Default)]
pub struct GradStore {
    grads: HashMap<usize, Vec<f32>>,
}

impl GradStore {
    /// Creates an empty `GradStore`.
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Returns the gradient slice for `node_id`, if present.
    pub fn get(&self, node_id: usize) -> Option<&[f32]> {
        self.grads.get(&node_id).map(Vec::as_slice)
    }

    /// Returns a mutable gradient vector for `node_id`, if present.
    pub fn get_mut(&mut self, node_id: usize) -> Option<&mut Vec<f32>> {
        self.grads.get_mut(&node_id)
    }

    /// Inserts or overwrites the gradient for `node_id`.
    pub fn insert(&mut self, node_id: usize, grad: Vec<f32>) {
        self.grads.insert(node_id, grad);
    }

    /// Accumulates `grad` into `node_id` (or inserts a copy if absent).
    pub fn accumulate(&mut self, node_id: usize, grad: &[f32]) -> OchreResult<()> {
        if let Some(existing) = self.grads.get_mut(&node_id) {
            if existing.len() != grad.len() {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: 1,
                    expected_cols: existing.len(),
                    actual_rows: 1,
                    actual_cols: grad.len(),
                });
            }
            for (dst, &src) in existing.iter_mut().zip(grad.iter()) {
                *dst += src;
            }
        } else {
            self.grads.insert(node_id, grad.to_vec());
        }
        Ok(())
    }

    /// Removes and returns the gradient vector for `node_id`.
    pub fn remove(&mut self, node_id: usize) -> Option<Vec<f32>> {
        self.grads.remove(&node_id)
    }

    /// Number of stored tensor gradients.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Returns true if no gradients are stored.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Iterates over `(node_id, grad_slice)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        self.grads.iter().map(|(&id, v)| (id, v.as_slice()))
    }
}

struct TapeNode {
    op: Op,
}

struct AutogradState {
    next_id: usize,
    base_id: usize,
    nodes: Vec<Option<TapeNode>>,
    overflow_nodes: HashMap<usize, TapeNode>,
    no_grad_depth: usize,
}

impl AutogradState {
    fn new() -> Self {
        Self {
            next_id: 1,
            base_id: 1,
            nodes: Vec::new(),
            overflow_nodes: HashMap::new(),
            no_grad_depth: 0,
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn record(&mut self, op: Op) -> usize {
        let id = self.alloc_id();
        if id >= self.base_id {
            let idx = id - self.base_id;
            if idx >= self.nodes.len() {
                self.nodes.resize_with(idx + 1, || None);
            }
            self.nodes[idx] = Some(TapeNode { op });
        } else {
            self.overflow_nodes.insert(id, TapeNode { op });
        }
        id
    }

    fn get_node(&self, id: usize) -> Option<&TapeNode> {
        if id >= self.base_id {
            self.nodes.get(id - self.base_id).and_then(|o| o.as_ref())
        } else {
            self.overflow_nodes.get(&id)
        }
    }

    fn take_node(&mut self, id: usize) -> Option<TapeNode> {
        let taken = if id >= self.base_id {
            let idx = id - self.base_id;
            self.nodes.get_mut(idx).and_then(Option::take)
        } else {
            self.overflow_nodes.remove(&id)
        };
        while matches!(self.nodes.last(), Some(None)) {
            self.nodes.pop();
        }
        if self.nodes.is_empty() {
            self.base_id = self.next_id;
        }
        taken
    }

    fn active_len(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count() + self.overflow_nodes.len()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.overflow_nodes.clear();
        self.base_id = self.next_id;
    }
}

thread_local! {
    static AUTOGRAD_STATE: RefCell<AutogradState> = RefCell::new(AutogradState::new());
}

/// Enters a `no_grad` scope on the current thread (nestable).
pub fn no_grad_enter() {
    AUTOGRAD_STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.no_grad_depth = state.no_grad_depth.saturating_add(1);
    });
}

/// Exits a `no_grad` scope on the current thread.
pub fn no_grad_exit() {
    AUTOGRAD_STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.no_grad_depth = state.no_grad_depth.saturating_sub(1);
    });
}

/// Returns `true` if gradient recording is currently enabled on this thread.
pub fn is_grad_enabled() -> bool {
    AUTOGRAD_STATE.with(|cell| cell.borrow().no_grad_depth == 0)
}

/// RAII guard that disables autograd tape recording while in scope.
#[derive(Debug)]
pub struct NoGradGuard {
    _private: (),
}

impl NoGradGuard {
    /// Enters a `no_grad` scope until this guard is dropped.
    pub fn new() -> Self {
        no_grad_enter();
        Self { _private: () }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        no_grad_exit();
    }
}

/// Executes `f` with autograd recording disabled, restoring the previous state on return.
pub fn with_no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

/// Thread-local arena-based autograd tape controller.
pub struct AutogradTape;

impl AutogradTape {
    /// Allocates a unique node ID for a leaf tensor/parameter without inserting an intermediate node.
    pub fn alloc_leaf() -> usize {
        AUTOGRAD_STATE.with(|cell| cell.borrow_mut().alloc_id())
    }

    /// Records an intermediate operation onto the thread-local tape and returns its node ID.
    pub fn record(op: Op) -> usize {
        AUTOGRAD_STATE.with(|cell| cell.borrow_mut().record(op))
    }

    /// Returns true if gradient recording is currently enabled.
    pub fn is_grad_enabled() -> bool {
        is_grad_enabled()
    }

    /// Returns the number of currently active intermediate nodes on the thread-local tape.
    pub fn len() -> usize {
        AUTOGRAD_STATE.with(|cell| cell.borrow().active_len())
    }

    /// Returns true if no intermediate nodes are stored on the tape.
    pub fn is_empty() -> bool {
        Self::len() == 0
    }

    /// Clears all intermediate nodes from the thread-local tape.
    pub fn clear() {
        AUTOGRAD_STATE.with(|cell| cell.borrow_mut().clear());
    }

    /// Runs reverse-mode autodiff from `root_id`, eagerly dropping visited tape nodes.
    pub fn backward(root_id: usize, root_numel: usize) -> OchreResult<GradStore> {
        let mut grads = GradStore::new();
        grads.insert(root_id, vec![1.0; root_numel]);

        let reachable = collect_reachable_nodes(root_id);
        if reachable.is_empty() {
            return Ok(grads);
        }

        // Take all reachable nodes up front in descending order (reverse topological order).
        let mut nodes_to_run = Vec::with_capacity(reachable.len());
        AUTOGRAD_STATE.with(|cell| {
            let mut state = cell.borrow_mut();
            for id in reachable {
                if let Some(node) = state.take_node(id) {
                    nodes_to_run.push((id, node));
                }
            }
        });

        for (id, node) in nodes_to_run {
            let d_out = match grads.get(id) {
                Some(g) => g.to_vec(),
                None => continue,
            };
            backward_op(&node.op, &d_out, &mut grads)?;
        }

        Ok(grads)
    }
}

fn collect_reachable_nodes(root_id: usize) -> Vec<usize> {
    AUTOGRAD_STATE.with(|cell| {
        let state = cell.borrow();
        let mut visited = HashSet::new();
        let mut stack = vec![root_id];
        let mut order = Vec::new();
        let mut inputs = Vec::with_capacity(4);

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(node) = state.get_node(id) {
                order.push(id);
                inputs.clear();
                node.op.collect_inputs(&mut inputs);
                for &in_id in &inputs {
                    if !visited.contains(&in_id) {
                        stack.push(in_id);
                    }
                }
            }
        }
        order.sort_unstable_by(|a, b| b.cmp(a));
        order
    })
}

fn backward_op(op: &Op, d_out: &[f32], grads: &mut GradStore) -> OchreResult<()> {
    match op {
        Op::Leaf => Ok(()),
        Op::Add {
            a_id,
            b_id,
            a_shape,
            b_shape,
            out_shape,
        } => {
            if let Some(id) = a_id {
                let da = reduce_grad_for_shape(d_out, out_shape, a_shape)?;
                grads.accumulate(*id, &da)?;
            }
            if let Some(id) = b_id {
                let db = reduce_grad_for_shape(d_out, out_shape, b_shape)?;
                grads.accumulate(*id, &db)?;
            }
            Ok(())
        }
        Op::Sub {
            a_id,
            b_id,
            a_shape,
            b_shape,
            out_shape,
        } => {
            if let Some(id) = a_id {
                let da = reduce_grad_for_shape(d_out, out_shape, a_shape)?;
                grads.accumulate(*id, &da)?;
            }
            if let Some(id) = b_id {
                let neg_out: Vec<f32> = d_out.iter().map(|&v| -v).collect();
                let db = reduce_grad_for_shape(&neg_out, out_shape, b_shape)?;
                grads.accumulate(*id, &db)?;
            }
            Ok(())
        }
        Op::Mul {
            a_id,
            b_id,
            a_data,
            b_data,
            a_shape,
            b_shape,
            out_shape,
        } => {
            if let Some(id) = a_id {
                let full_da = mul_broadcast_grad(d_out, out_shape, b_data, b_shape)?;
                let da = reduce_grad_for_shape(&full_da, out_shape, a_shape)?;
                grads.accumulate(*id, &da)?;
            }
            if let Some(id) = b_id {
                let full_db = mul_broadcast_grad(d_out, out_shape, a_data, a_shape)?;
                let db = reduce_grad_for_shape(&full_db, out_shape, b_shape)?;
                grads.accumulate(*id, &db)?;
            }
            Ok(())
        }
        Op::Scale { input_id, scalar } => {
            if let Some(id) = input_id {
                let dx: Vec<f32> = d_out.iter().map(|&g| g * *scalar).collect();
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::Matmul {
            a_id,
            b_id,
            a_data,
            b_data,
            m,
            k,
            n,
            device,
        } => backward_matmul(
            *a_id, *b_id, a_data, b_data, *m, *k, *n, device, d_out, grads,
        ),
        Op::BatchedMatmul {
            a_id,
            b_id,
            a_data,
            b_data,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            device,
        } => backward_batched_matmul(
            *a_id, *b_id, a_data, b_data, *batch, *m, *k, *n, *trans_a, *trans_b, device, d_out,
            grads,
        ),
        Op::TransposeLast2 {
            input_id,
            batch,
            rows,
            cols,
            device,
        } => {
            if let Some(id) = input_id {
                let backend = get_backend(device);
                let dx = backend.transpose_last2_f32(*batch, *cols, *rows, d_out)?;
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::Reshape { input_id, .. } => {
            if let Some(id) = input_id {
                grads.accumulate(*id, d_out)?;
            }
            Ok(())
        }
        Op::Sum {
            input_id,
            input_shape,
            axis,
        } => {
            if let Some(id) = input_id {
                let dx = expand_grad_for_reduction(d_out, input_shape, *axis, 1.0)?;
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::Mean {
            input_id,
            input_shape,
            axis,
        } => {
            if let Some(id) = input_id {
                let denom = match axis {
                    Some(ax) => *input_shape.get(*ax).unwrap_or(&1) as f32,
                    None => input_shape.iter().copied().product::<usize>().max(1) as f32,
                };
                let dx = expand_grad_for_reduction(d_out, input_shape, *axis, 1.0 / denom)?;
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::ReLU {
            input_id,
            input_data,
        } => {
            if let Some(id) = input_id {
                let dx: Vec<f32> = d_out
                    .iter()
                    .zip(input_data.iter())
                    .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                    .collect();
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::Sigmoid { input_id, out_data } => {
            if let Some(id) = input_id {
                let dx: Vec<f32> = d_out
                    .iter()
                    .zip(out_data.iter())
                    .map(|(&g, &y)| g * y * (1.0 - y))
                    .collect();
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::Tanh { input_id, out_data } => {
            if let Some(id) = input_id {
                let dx: Vec<f32> = d_out
                    .iter()
                    .zip(out_data.iter())
                    .map(|(&g, &y)| g * (1.0 - y * y))
                    .collect();
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::GELU {
            input_id,
            input_data,
            device,
        } => {
            if let Some(id) = input_id {
                let dx = get_backend(device).gelu_bwd_f32(d_out, input_data);
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::SiLU {
            input_id,
            input_data,
            device,
        } => {
            if let Some(id) = input_id {
                let dx = get_backend(device).silu_bwd_f32(d_out, input_data);
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::CrossEntropy {
            logits_id,
            d_logits,
        } => {
            if let Some(id) = logits_id {
                let scale = d_out.first().copied().unwrap_or(1.0);
                if (scale - 1.0).abs() <= f32::EPSILON {
                    grads.accumulate(*id, d_logits)?;
                } else {
                    let scaled: Vec<f32> = d_logits.iter().map(|&g| g * scale).collect();
                    grads.accumulate(*id, &scaled)?;
                }
            }
            Ok(())
        }
        Op::MSE {
            pred_id,
            diff,
            scale,
        } => {
            if let Some(id) = pred_id {
                let factor = d_out.first().copied().unwrap_or(1.0) * *scale;
                let dx: Vec<f32> = diff.iter().map(|&d| d * factor).collect();
                grads.accumulate(*id, &dx)?;
            }
            Ok(())
        }
        Op::LayerNorm {
            input_id,
            gamma_id,
            beta_id,
            x,
            gamma,
            mean,
            rstd,
            rows,
            cols,
            device,
        } => {
            if input_id.is_none() && gamma_id.is_none() && beta_id.is_none() {
                return Ok(());
            }
            let (dx, dgamma, dbeta) = get_backend(device).row_layernorm_bwd_f32(
                d_out,
                x,
                gamma.as_deref(),
                mean,
                rstd,
                *rows,
                *cols,
            )?;
            if let Some(id) = input_id {
                grads.accumulate(*id, &dx)?;
            }
            if let Some(id) = gamma_id {
                grads.accumulate(*id, &dgamma)?;
            }
            if let Some(id) = beta_id {
                grads.accumulate(*id, &dbeta)?;
            }
            Ok(())
        }
        Op::RMSNorm {
            input_id,
            gamma_id,
            x,
            gamma,
            rstd,
            rows,
            cols,
            device,
        } => {
            if input_id.is_none() && gamma_id.is_none() {
                return Ok(());
            }
            let (dx, dgamma) = get_backend(device).row_rmsnorm_bwd_f32(
                d_out,
                x,
                gamma.as_deref(),
                rstd,
                *rows,
                *cols,
            )?;
            if let Some(id) = input_id {
                grads.accumulate(*id, &dx)?;
            }
            if let Some(id) = gamma_id {
                grads.accumulate(*id, &dgamma)?;
            }
            Ok(())
        }
        Op::Conv2d {
            input_id,
            weight_id,
            bias_id,
            batch,
            in_c,
            in_h,
            in_w,
            out_c,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            out_h,
            out_w,
            col_data,
            weight_data,
            device,
        } => backward_conv2d(
            *input_id,
            *weight_id,
            *bias_id,
            *batch,
            *in_c,
            *in_h,
            *in_w,
            *out_c,
            *kh,
            *kw,
            *stride_h,
            *stride_w,
            *pad_h,
            *pad_w,
            *dil_h,
            *dil_w,
            *out_h,
            *out_w,
            col_data,
            weight_data,
            device,
            d_out,
            grads,
        ),
        Op::Embedding {
            weight_id,
            indices,
            num_embeddings,
            embedding_dim,
        } => {
            if let Some(id) = weight_id {
                let mut dw = vec![0.0f32; (*num_embeddings) * (*embedding_dim)];
                for (pos, &idx) in indices.iter().enumerate() {
                    if idx >= *num_embeddings {
                        return Err(OchreError::InvalidParameter(format!(
                            "Embedding index {idx} out of range for num_embeddings={num_embeddings} ( >_< )"
                        )));
                    }
                    let src_start = pos * (*embedding_dim);
                    let dst_start = idx * (*embedding_dim);
                    for d in 0..(*embedding_dim) {
                        dw[dst_start + d] += d_out[src_start + d];
                    }
                }
                grads.accumulate(*id, &dw)?;
            }
            Ok(())
        }
        Op::Custom { backward, .. } => {
            let pairs = (backward.0)(d_out)?;
            for (in_id, grad) in pairs {
                grads.accumulate(in_id, &grad)?;
            }
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_matmul(
    a_id: Option<usize>,
    b_id: Option<usize>,
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
    device: &DeviceType,
    d_out: &[f32],
    grads: &mut GradStore,
) -> OchreResult<()> {
    let backend = get_backend(device);
    if let Some(id) = a_id {
        let da = if matches!(device, DeviceType::Cpu) {
            let mut buf = vec![0.0f32; m * k];
            sgemm(m, k, n, d_out, b_data, &mut buf, false, true, 1.0, 0.0)?;
            buf
        } else {
            backend.batched_matmul_f32(1, m, k, n, d_out, b_data, false, true)?
        };
        grads.accumulate(id, &da)?;
    }
    if let Some(id) = b_id {
        let db = if matches!(device, DeviceType::Cpu) {
            let mut buf = vec![0.0f32; k * n];
            sgemm(k, n, m, a_data, d_out, &mut buf, true, false, 1.0, 0.0)?;
            buf
        } else {
            backend.batched_matmul_f32(1, k, n, m, a_data, d_out, true, false)?
        };
        grads.accumulate(id, &db)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward_batched_matmul(
    a_id: Option<usize>,
    b_id: Option<usize>,
    a_data: &[f32],
    b_data: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    device: &DeviceType,
    d_out: &[f32],
    grads: &mut GradStore,
) -> OchreResult<()> {
    let backend = get_backend(device);
    if let Some(id) = a_id {
        let da = match (trans_a, trans_b) {
            (false, false) => {
                backend.batched_matmul_f32(batch, m, k, n, d_out, b_data, false, true)?
            }
            (false, true) => {
                backend.batched_matmul_f32(batch, m, k, n, d_out, b_data, false, false)?
            }
            (true, false) => {
                backend.batched_matmul_f32(batch, k, m, n, b_data, d_out, false, true)?
            }
            (true, true) => {
                backend.batched_matmul_f32(batch, k, m, n, b_data, d_out, true, true)?
            }
        };
        grads.accumulate(id, &da)?;
    }
    if let Some(id) = b_id {
        let db = match (trans_a, trans_b) {
            (false, false) => {
                backend.batched_matmul_f32(batch, k, n, m, a_data, d_out, true, false)?
            }
            (false, true) => {
                backend.batched_matmul_f32(batch, n, k, m, d_out, a_data, true, false)?
            }
            (true, false) => {
                backend.batched_matmul_f32(batch, k, n, m, a_data, d_out, false, false)?
            }
            (true, true) => {
                backend.batched_matmul_f32(batch, n, k, m, d_out, a_data, true, true)?
            }
        };
        grads.accumulate(id, &db)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward_conv2d(
    input_id: Option<usize>,
    weight_id: Option<usize>,
    bias_id: Option<usize>,
    batch: usize,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
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
    col_data: &[f32],
    weight_data: &[f32],
    device: &DeviceType,
    d_out: &[f32],
    grads: &mut GradStore,
) -> OchreResult<()> {
    let spatial_out = out_h * out_w;
    let total_rows = batch * spatial_out;
    let patch_dim = in_c * kh * kw;

    // Reshape d_out from NCHW [batch, out_c, out_h, out_w] into [batch*out_h*out_w, out_c]
    let mut dy_mat = vec![0.0f32; total_rows * out_c];
    for b in 0..batch {
        for oc in 0..out_c {
            let src_base = (b * out_c + oc) * spatial_out;
            for s in 0..spatial_out {
                dy_mat[(b * spatial_out + s) * out_c + oc] = d_out[src_base + s];
            }
        }
    }

    if let Some(id) = bias_id {
        let mut db = vec![0.0f32; out_c];
        for row in 0..total_rows {
            let off = row * out_c;
            for oc in 0..out_c {
                db[oc] += dy_mat[off + oc];
            }
        }
        grads.accumulate(id, &db)?;
    }

    if let Some(id) = weight_id {
        // weight is [out_c, patch_dim]: dW = dy_mat^T * col_data
        let mut dw = vec![0.0f32; out_c * patch_dim];
        sgemm(
            out_c, patch_dim, total_rows, &dy_mat, col_data, &mut dw, true, false, 1.0, 0.0,
        )?;
        grads.accumulate(id, &dw)?;
    }

    if let Some(id) = input_id {
        let mut dcol = vec![0.0f32; total_rows * patch_dim];
        sgemm(
            total_rows,
            patch_dim,
            out_c,
            &dy_mat,
            weight_data,
            &mut dcol,
            false,
            false,
            1.0,
            0.0,
        )?;
        let dx = get_backend(device).col2im_f32(
            &dcol, batch, in_c, in_h, in_w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            out_h, out_w,
        )?;
        grads.accumulate(id, &dx)?;
    }

    Ok(())
}

/// Computes `d_out * other` with NumPy-style trailing broadcasting.
pub(crate) fn mul_broadcast_grad(
    d_out: &[f32],
    out_shape: &[usize],
    other: &[f32],
    other_shape: &[usize],
) -> OchreResult<Vec<f32>> {
    if out_shape == other_shape {
        return Ok(d_out
            .iter()
            .zip(other.iter())
            .map(|(&g, &x)| g * x)
            .collect());
    }
    if other.len() == 1 {
        let scalar = other[0];
        return Ok(d_out.iter().map(|&g| g * scalar).collect());
    }
    if out_shape.len() == 2 && (other_shape == [out_shape[1]] || other_shape == [1, out_shape[1]]) {
        let rows = out_shape[0];
        let cols = out_shape[1];
        let mut out = vec![0.0f32; d_out.len()];
        for r in 0..rows {
            let off = r * cols;
            for c in 0..cols {
                out[off + c] = d_out[off + c] * other[c];
            }
        }
        return Ok(out);
    }

    let out_ndim = out_shape.len();
    let pad = out_ndim.saturating_sub(other_shape.len());
    let mut padded = vec![1usize; out_ndim];
    for (i, &dim) in other_shape.iter().enumerate() {
        padded[pad + i] = dim;
    }
    let other_strides = compute_broadcast_strides(&padded);
    let mut out = vec![0.0f32; d_out.len()];
    let mut coords = vec![0usize; out_ndim];
    for (flat, &g) in d_out.iter().enumerate() {
        flat_to_coords(flat, out_shape, &mut coords);
        let o_idx = coords_to_flat(&coords, &other_strides);
        out[flat] = g * other[o_idx];
    }
    Ok(out)
}

/// Reduces `grad` from `out_shape` back down to `target_shape` by summing over broadcast axes.
pub(crate) fn reduce_grad_for_shape(
    grad: &[f32],
    out_shape: &[usize],
    target_shape: &[usize],
) -> OchreResult<Vec<f32>> {
    if out_shape == target_shape {
        return Ok(grad.to_vec());
    }
    let target_numel: usize = target_shape.iter().copied().product::<usize>().max(1);
    if target_numel == 1 {
        let s: f32 = grad.iter().copied().sum();
        return Ok(vec![s]);
    }
    if out_shape.len() == 2 && (target_shape == [out_shape[1]] || target_shape == [1, out_shape[1]])
    {
        let rows = out_shape[0];
        let cols = out_shape[1];
        let mut acc = vec![0.0f32; cols];
        for r in 0..rows {
            let row = &grad[r * cols..(r + 1) * cols];
            for c in 0..cols {
                acc[c] += row[c];
            }
        }
        return Ok(acc);
    }

    let out_ndim = out_shape.len();
    if target_shape.len() > out_ndim {
        return Err(OchreError::InvalidParameter(format!(
            "Cannot reduce shape {:?} to larger rank shape {:?} ( >_< )",
            out_shape, target_shape
        )));
    }
    let pad = out_ndim - target_shape.len();
    let mut padded = vec![1usize; out_ndim];
    for (i, &dim) in target_shape.iter().enumerate() {
        padded[pad + i] = dim;
    }
    let target_strides = compute_broadcast_strides(&padded);
    let mut acc = vec![0.0f32; target_numel];
    let mut coords = vec![0usize; out_ndim];
    for (flat, &g) in grad.iter().enumerate() {
        flat_to_coords(flat, out_shape, &mut coords);
        let t_idx = coords_to_flat(&coords, &target_strides);
        acc[t_idx] += g;
    }
    Ok(acc)
}

fn expand_grad_for_reduction(
    d_out: &[f32],
    input_shape: &[usize],
    axis: Option<usize>,
    scale: f32,
) -> OchreResult<Vec<f32>> {
    let total: usize = input_shape.iter().copied().product::<usize>().max(1);
    match axis {
        None => {
            let val = d_out.first().copied().unwrap_or(0.0) * scale;
            Ok(vec![val; total])
        }
        Some(ax) => {
            if ax >= input_shape.len() {
                return Err(OchreError::InvalidParameter(format!(
                    "Reduction axis {ax} out of bounds for shape {:?} ( >_< )",
                    input_shape
                )));
            }
            let outer: usize = input_shape[..ax].iter().copied().product::<usize>().max(1);
            let ax_len = input_shape[ax];
            let inner: usize = input_shape[ax + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1);
            let mut dx = vec![0.0f32; total];
            for o in 0..outer {
                for a in 0..ax_len {
                    let dst_base = (o * ax_len + a) * inner;
                    let src_base = o * inner;
                    for i in 0..inner {
                        dx[dst_base + i] = d_out[src_base + i] * scale;
                    }
                }
            }
            Ok(dx)
        }
    }
}

fn compute_broadcast_strides(padded_shape: &[usize]) -> Vec<usize> {
    let ndim = padded_shape.len();
    let mut strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        if padded_shape[i] == 1 {
            strides[i] = 0;
        } else {
            strides[i] = stride;
            stride *= padded_shape[i];
        }
    }
    strides
}

#[inline]
fn flat_to_coords(mut flat: usize, shape: &[usize], coords: &mut [usize]) {
    for i in (0..shape.len()).rev() {
        let dim = shape[i].max(1);
        coords[i] = flat % dim;
        flat /= dim;
    }
}

#[inline]
fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
    let mut idx = 0usize;
    for (&c, &s) in coords.iter().zip(strides.iter()) {
        idx += c * s;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad_guard_and_nesting() {
        assert!(is_grad_enabled());
        {
            let _g1 = NoGradGuard::new();
            assert!(!is_grad_enabled());
            with_no_grad(|| {
                assert!(!is_grad_enabled());
            });
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_grad_store_accumulate() {
        let mut store = GradStore::new();
        let id = AutogradTape::alloc_leaf();
        store.accumulate(id, &[1.0, 2.0]).unwrap();
        store.accumulate(id, &[0.5, -1.0]).unwrap();
        assert_eq!(store.get(id).unwrap(), &[1.5, 1.0]);
    }

    #[test]
    fn test_tape_eager_cleanup_on_backward() {
        AutogradTape::clear();
        let x = AutogradTape::alloc_leaf();
        let y = AutogradTape::record(Op::Scale {
            input_id: Some(x),
            scalar: 3.0,
        });
        let z = AutogradTape::record(Op::Sum {
            input_id: Some(y),
            input_shape: vec![2],
            axis: None,
        });
        assert_eq!(AutogradTape::len(), 2);
        let grads = AutogradTape::backward(z, 1).unwrap();
        assert_eq!(AutogradTape::len(), 0);
        assert_eq!(grads.get(x).unwrap(), &[3.0, 3.0]);
    }
}
