// OchreML PyO3 Bindings for Deep Learning Engine (_ochreml.nn) (*≧ω≦*)
// Exposes Tensor, Autograd, MLP/CNN/RNN/Norm/Attention/Transformer/CausalLM/LoRA, Optimizers & Sequential.

use crate::device::parse_device;
use crate::dit_io::validate_dit_path;
use crate::error::{OchreError, OchreResult};
use crate::nn::attention::{
    KVCache, MultiHeadAttention, PositionalEncoding, RoPE, ScaledDotProductAttention,
};
use crate::nn::autograd::{AutogradTape, GradStore, is_grad_enabled, no_grad_enter, no_grad_exit};
use crate::nn::cnn::{
    AvgPool2d, BatchNorm2d, Conv2d, DepthwiseConv2d, Flatten, GlobalAvgPool2d, MaxPool2d,
    ResidualBlock,
};
use crate::nn::dtype::DType;
use crate::nn::layers::{Dropout, Embedding, Linear};
use crate::nn::lora::{LoRAAdapterState, LoRALinear};
use crate::nn::losses::{CrossEntropyLoss, MSELoss, Reduction};
use crate::nn::norm::{LayerNorm, RMSNorm};
use crate::nn::optim::{
    Adam, AdamW, CosineAnnealingLR, SGD, StepLR, WarmupCosineLR, clip_grad_store_norm_,
};
use crate::nn::rnn::{GRU, LSTM, RNN, RnnNonlinearity};
use crate::nn::sequential::{LossKind, OptimKind, Sequential, SequentialLayer};
use crate::nn::tensor::Tensor;
use crate::nn::transformer::{
    CausalLM, CausalLMConfig, FeedForward, SwiGLU, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList, PyTuple};
use std::cell::RefCell;
use std::fs;
use std::sync::{Arc, RwLock};

thread_local! {
    static THREAD_GRADS: RefCell<GradStore> = RefCell::new(GradStore::new());
}

fn accumulate_thread_grads(store: &GradStore) -> OchreResult<()> {
    THREAD_GRADS.with(|cell| {
        let mut dst = cell.borrow_mut();
        for (id, g) in store.iter() {
            dst.accumulate(id, g)?;
        }
        Ok(())
    })
}

fn parse_dtype(s: &str) -> OchreResult<DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "fp32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" | "fp16" => Ok(DType::F16),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported dtype '{other}'. Expected 'f32', 'bf16', or 'f16' ( >_< )"
        ))),
    }
}

fn dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
    }
}

fn parse_reduction(s: &str) -> OchreResult<Reduction> {
    match s.trim().to_ascii_lowercase().as_str() {
        "mean" => Ok(Reduction::Mean),
        "sum" => Ok(Reduction::Sum),
        "none" => Ok(Reduction::None),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported reduction '{other}'. Expected 'mean', 'sum', or 'none' ( >_< )"
        ))),
    }
}

fn parse_optim_kind(s: &str) -> OchreResult<OptimKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "sgd" => Ok(OptimKind::SGD),
        "adam" => Ok(OptimKind::Adam),
        "adamw" => Ok(OptimKind::AdamW),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported optimizer '{other}'. Expected 'sgd', 'adam', or 'adamw' ( >_< )"
        ))),
    }
}

fn parse_loss_kind(s: &str) -> OchreResult<LossKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cross_entropy" | "crossentropy" | "ce" => Ok(LossKind::CrossEntropy),
        "mse" | "mse_loss" => Ok(LossKind::MSE),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported loss '{other}'. Expected 'cross_entropy' or 'mse' ( >_< )"
        ))),
    }
}

fn parse_nonlinearity(s: &str) -> OchreResult<RnnNonlinearity> {
    match s.trim().to_ascii_lowercase().as_str() {
        "tanh" => Ok(RnnNonlinearity::Tanh),
        "relu" => Ok(RnnNonlinearity::Relu),
        other => Err(OchreError::InvalidParameter(format!(
            "Unsupported RNN nonlinearity '{other}'. Expected 'tanh' or 'relu' ( >_< )"
        ))),
    }
}

fn f32_slice_to_pybytes<'py>(py: Python<'py>, data: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    let n_bytes = std::mem::size_of_val(data);
    PyBytes::new_with(py, n_bytes, |buf| {
        let (chunks, _) = buf.as_chunks_mut::<4>();
        for (dst, &val) in chunks.iter_mut().zip(data.iter()) {
            *dst = val.to_le_bytes();
        }
        Ok(())
    })
}

fn ingest_f32_data(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    explicit_shape: Option<Vec<usize>>,
) -> PyResult<(Vec<f32>, Vec<usize>)> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        let shape = explicit_shape.unwrap_or_else(|| buf.shape().to_vec());
        let data = buf.to_vec(py).map_err(|e| {
            PyValueError::new_err(format!("Failed to read float32 buffer: {e} ( >_< )"))
        })?;
        return Ok((data, shape));
    }
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        let shape = explicit_shape.unwrap_or_else(|| buf.shape().to_vec());
        let raw = buf.to_vec(py).map_err(|e| {
            PyValueError::new_err(format!("Failed to read float64 buffer: {e} ( >_< )"))
        })?;
        let data: Vec<f32> = raw.into_iter().map(|v| v as f32).collect();
        return Ok((data, shape));
    }
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        let shape = explicit_shape.unwrap_or_else(|| buf.shape().to_vec());
        let raw = buf.to_vec(py).map_err(|e| {
            PyValueError::new_err(format!("Failed to read int64 buffer: {e} ( >_< )"))
        })?;
        let data: Vec<f32> = raw.into_iter().map(|v| v as f32).collect();
        return Ok((data, shape));
    }
    if let Ok(buf) = PyBuffer::<i32>::get(obj) {
        let shape = explicit_shape.unwrap_or_else(|| buf.shape().to_vec());
        let raw = buf.to_vec(py).map_err(|e| {
            PyValueError::new_err(format!("Failed to read int32 buffer: {e} ( >_< )"))
        })?;
        let data: Vec<f32> = raw.into_iter().map(|v| v as f32).collect();
        return Ok((data, shape));
    }

    if let Ok(scalar) = obj.extract::<f32>() {
        let shape = explicit_shape.unwrap_or_else(|| vec![1]);
        return Ok((vec![scalar], shape));
    }

    let mut data = Vec::new();
    let inferred = flatten_py_sequence(obj, &mut data)?;
    let shape = explicit_shape.unwrap_or(inferred);
    Ok((data, shape))
}

fn flatten_py_sequence(obj: &Bound<'_, PyAny>, out: &mut Vec<f32>) -> PyResult<Vec<usize>> {
    if let Ok(v) = obj.extract::<f32>() {
        out.push(v);
        return Ok(Vec::new());
    }
    if let Ok(seq) = obj.cast::<PyList>() {
        let len = seq.len();
        if len == 0 {
            return Err(PyValueError::new_err(
                "Cannot create Tensor from empty list ( >_< )",
            ));
        }
        let mut sub_shape: Option<Vec<usize>> = None;
        for item in seq.iter() {
            let s = flatten_py_sequence(&item, out)?;
            match &sub_shape {
                None => sub_shape = Some(s),
                Some(prev) if prev != &s => {
                    return Err(PyValueError::new_err(
                        "Ragged nested sequences cannot form a Tensor ( >_< )",
                    ));
                }
                _ => {}
            }
        }
        let mut shape = vec![len];
        if let Some(sub) = sub_shape {
            shape.extend(sub);
        }
        return Ok(shape);
    }
    if let Ok(seq) = obj.cast::<PyTuple>() {
        let len = seq.len();
        if len == 0 {
            return Err(PyValueError::new_err(
                "Cannot create Tensor from empty tuple ( >_< )",
            ));
        }
        let mut sub_shape: Option<Vec<usize>> = None;
        for item in seq.iter() {
            let s = flatten_py_sequence(&item, out)?;
            match &sub_shape {
                None => sub_shape = Some(s),
                Some(prev) if prev != &s => {
                    return Err(PyValueError::new_err(
                        "Ragged nested tuples cannot form a Tensor ( >_< )",
                    ));
                }
                _ => {}
            }
        }
        let mut shape = vec![len];
        if let Some(sub) = sub_shape {
            shape.extend(sub);
        }
        return Ok(shape);
    }
    Err(PyValueError::new_err(
        "Expected buffer, nested list/tuple, or float scalar for Tensor data ( >_< )",
    ))
}

fn ingest_usize_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        let raw = buf.to_vec(py)?;
        return raw
            .into_iter()
            .map(|v| {
                usize::try_from(v)
                    .map_err(|_| PyValueError::new_err("Target index cannot be negative ( >_< )"))
            })
            .collect();
    }
    if let Ok(buf) = PyBuffer::<i32>::get(obj) {
        let raw = buf.to_vec(py)?;
        return raw
            .into_iter()
            .map(|v| {
                usize::try_from(v)
                    .map_err(|_| PyValueError::new_err("Target index cannot be negative ( >_< )"))
            })
            .collect();
    }
    if let Ok(buf) = PyBuffer::<u64>::get(obj) {
        let raw = buf.to_vec(py)?;
        return Ok(raw.into_iter().map(|v| v as usize).collect());
    }
    if let Ok(v) = obj.extract::<Vec<usize>>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<Vec<Vec<usize>>>() {
        return Ok(v.into_iter().flatten().collect());
    }
    Err(PyValueError::new_err(
        "Expected integer sequence or buffer for indices/targets ( >_< )",
    ))
}

/// Trait enabling in-place optimizer steps on `PyTensor` views backed by stateful Rust layers.
pub trait ParamOwner: Send + Sync {
    fn update_param(&self, idx: usize, data: &[f32]) -> OchreResult<()>;
}

macro_rules! impl_param_owner {
    ($ty:ty) => {
        impl ParamOwner for RwLock<$ty> {
            fn update_param(&self, idx: usize, data: &[f32]) -> OchreResult<()> {
                let mut guard = self
                    .write()
                    .map_err(|_| OchreError::General("Parameter lock poisoned ( >_< )".into()))?;
                let mut params = guard.parameters_mut();
                if let Some(p) = params.get_mut(idx) {
                    if p.data.len() != data.len() {
                        return Err(OchreError::DimensionMismatch {
                            expected_rows: 1,
                            expected_cols: p.data.len(),
                            actual_rows: 1,
                            actual_cols: data.len(),
                        });
                    }
                    p.data.copy_from_slice(data);
                }
                Ok(())
            }
        }
    };
}

impl_param_owner!(CausalLM);
impl_param_owner!(LoRALinear);
impl_param_owner!(RNN);
impl_param_owner!(GRU);
impl_param_owner!(LSTM);
impl_param_owner!(MultiHeadAttention);
impl_param_owner!(SwiGLU);
impl_param_owner!(FeedForward);
impl_param_owner!(TransformerEncoderLayer);
impl_param_owner!(TransformerEncoder);
impl_param_owner!(TransformerDecoderLayer);
impl_param_owner!(TransformerDecoder);
impl_param_owner!(ResidualBlock);

#[derive(Clone)]
struct ParamLink {
    owner: Arc<dyn ParamOwner>,
    index: usize,
}

fn read_tensor(t: &Arc<RwLock<Tensor>>) -> PyResult<Tensor> {
    t.read()
        .map(|g| g.clone())
        .map_err(|_| PyValueError::new_err("Tensor lock poisoned ( >_< )"))
}

fn export_owned_params(owner: Arc<dyn ParamOwner>, params: Vec<&Tensor>) -> Vec<PyTensor> {
    params
        .into_iter()
        .enumerate()
        .map(|(index, t)| PyTensor {
            inner: Arc::new(RwLock::new(t.clone())),
            link: Some(ParamLink {
                owner: Arc::clone(&owner),
                index,
            }),
        })
        .collect()
}

/// Autograd gradient store wrapper returned by `Tensor.backward()`.
#[pyclass(
    name = "GradStore",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyGradStore {
    inner: GradStore,
}

#[pymethods]
impl PyGradStore {
    pub fn get(&self, node_id: usize) -> Option<Vec<f32>> {
        self.inner.get(node_id).map(|s| s.to_vec())
    }

    pub fn node_ids(&self) -> Vec<usize> {
        self.inner.iter().map(|(id, _)| id).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn clip_grad_norm_(&mut self, node_ids: Vec<usize>, max_norm: f32) -> f32 {
        clip_grad_store_norm_(&mut self.inner, &node_ids, max_norm)
    }
}

/// Core OchreML Tensor exposed to Python (`ochreml._ochreml.nn.Tensor`).
#[pyclass(name = "Tensor", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTensor {
    inner: Arc<RwLock<Tensor>>,
    link: Option<ParamLink>,
}

impl PyTensor {
    fn from_tensor(t: Tensor) -> Self {
        Self {
            inner: Arc::new(RwLock::new(t)),
            link: None,
        }
    }

    fn tensor(&self) -> PyResult<Tensor> {
        read_tensor(&self.inner)
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape = None, dtype = "f32".to_string(), device = "auto".to_string(), requires_grad = false))]
    pub fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: String,
        device: String,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let (flat, inferred_shape) = ingest_f32_data(py, data, shape)?;
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        let t = Tensor::new(flat, inferred_shape, dt, dev, requires_grad)?;
        Ok(Self::from_tensor(t))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32".to_string(), device = "auto".to_string(), requires_grad = false))]
    pub fn zeros(
        shape: Vec<usize>,
        dtype: String,
        device: String,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        let mut t = Tensor::zeros(shape, dt, dev);
        if requires_grad {
            t = t.with_requires_grad(true);
        }
        Ok(Self::from_tensor(t))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32".to_string(), device = "auto".to_string(), requires_grad = false))]
    pub fn ones(
        shape: Vec<usize>,
        dtype: String,
        device: String,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        let mut t = Tensor::ones(shape, dt, dev);
        if requires_grad {
            t = t.with_requires_grad(true);
        }
        Ok(Self::from_tensor(t))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, mean = 0.0, std = 1.0, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42, requires_grad = false))]
    pub fn randn(
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        dtype: String,
        device: String,
        seed: u64,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        let t = Tensor::randn_with(shape, mean, std, dt, dev, seed, requires_grad);
        Ok(Self::from_tensor(t))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, low = -1.0, high = 1.0, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42, requires_grad = false))]
    pub fn uniform(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        dtype: String,
        device: String,
        seed: u64,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        let t = Tensor::uniform(shape, low, high, dt, dev, seed, requires_grad);
        Ok(Self::from_tensor(t))
    }

    #[getter]
    pub fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.tensor()?.shape)
    }

    #[getter]
    pub fn ndim(&self) -> PyResult<usize> {
        Ok(self.tensor()?.ndim())
    }

    #[getter]
    pub fn numel(&self) -> PyResult<usize> {
        Ok(self.tensor()?.numel())
    }

    #[getter]
    pub fn dtype(&self) -> PyResult<String> {
        Ok(dtype_str(self.tensor()?.dtype).to_string())
    }

    #[getter]
    pub fn device(&self) -> PyResult<String> {
        Ok(self.tensor()?.device.to_string())
    }

    #[getter]
    pub fn requires_grad(&self) -> PyResult<bool> {
        Ok(self.tensor()?.requires_grad)
    }

    #[setter]
    pub fn set_requires_grad(&mut self, val: bool) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Tensor lock poisoned ( >_< )"))?;
        let updated = guard.clone().with_requires_grad(val);
        *guard = updated;
        Ok(())
    }

    #[getter]
    pub fn node_id(&self) -> PyResult<Option<usize>> {
        Ok(self.tensor()?.node_id)
    }

    #[getter]
    pub fn grad(&self) -> PyResult<Option<PyTensor>> {
        let t = self.tensor()?;
        let Some(nid) = t.node_id else {
            return Ok(None);
        };
        let maybe_vec = THREAD_GRADS.with(|cell| cell.borrow().get(nid).map(|s| s.to_vec()));
        match maybe_vec {
            Some(g_data) => {
                let gt = Tensor::new(g_data, t.shape.clone(), t.dtype, t.device.clone(), false)?;
                Ok(Some(PyTensor::from_tensor(gt)))
            }
            None => Ok(None),
        }
    }

    pub fn zero_grad(&self) -> PyResult<()> {
        let t = self.tensor()?;
        if let Some(nid) = t.node_id {
            THREAD_GRADS.with(|cell| {
                cell.borrow_mut().remove(nid);
            });
        }
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<(Vec<usize>, Bound<'py, PyBytes>)> {
        let t = self.tensor()?;
        let bytes = f32_slice_to_pybytes(py, &t.data)?;
        Ok((t.shape, bytes))
    }

    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let t = self.tensor()?;
        let bytes = f32_slice_to_pybytes(py, &t.data)?;
        let np = py.import("numpy")?;
        let arr = np
            .getattr("frombuffer")?
            .call1((bytes, "float32"))?
            .call_method1("reshape", (t.shape,))?
            .call_method0("copy")?;
        Ok(arr)
    }

    pub fn tolist(&self) -> PyResult<Vec<f32>> {
        Ok(self.tensor()?.data)
    }

    pub fn to_dtype(&self, dtype: String) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        Ok(Self::from_tensor(self.tensor()?.to_dtype(dt)))
    }

    pub fn to_device(&self, device: String) -> PyResult<Self> {
        let dev = parse_device(Some(&device))?;
        Ok(Self::from_tensor(self.tensor()?.to_device(dev)))
    }

    pub fn detach(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.detach()))
    }

    pub fn copy_from_(&self, other: &PyTensor) -> PyResult<()> {
        let src = other.tensor()?;
        {
            let mut dst = self
                .inner
                .write()
                .map_err(|_| PyValueError::new_err("Tensor lock poisoned ( >_< )"))?;
            if dst.shape != src.shape {
                return Err(PyValueError::new_err(format!(
                    "Shape mismatch in copy_from_: {:?} vs {:?} ( >_< )",
                    dst.shape, src.shape
                )));
            }
            dst.data.copy_from_slice(&src.data);
        }
        if let Some(ref link) = self.link {
            link.owner.update_param(link.index, &src.data)?;
        }
        Ok(())
    }

    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.reshape(shape)?))
    }

    pub fn transpose_last2(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.transpose_last2()?))
    }

    pub fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        let a = self.tensor()?;
        let b = other.tensor()?;
        Ok(Self::from_tensor(a.matmul(&b)?))
    }

    pub fn relu(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.relu()?))
    }

    pub fn sigmoid(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.sigmoid()?))
    }

    pub fn tanh(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.tanh()?))
    }

    pub fn gelu(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.gelu()?))
    }

    pub fn silu(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.silu()?))
    }

    #[pyo3(signature = (axis = None))]
    pub fn sum(&self, axis: Option<usize>) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.sum(axis)?))
    }

    #[pyo3(signature = (axis = None))]
    pub fn mean(&self, axis: Option<usize>) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.mean(axis)?))
    }

    pub fn backward(&self) -> PyResult<PyGradStore> {
        let gs = self.tensor()?.backward()?;
        accumulate_thread_grads(&gs)?;
        Ok(PyGradStore { inner: gs })
    }

    pub fn item(&self) -> PyResult<f32> {
        let t = self.tensor()?;
        if t.data.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() only works on 1-element tensors, got {} elements ( >_< )",
                t.data.len()
            )));
        }
        Ok(t.data[0])
    }

    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let a = self.tensor()?;
        if let Ok(b_py) = other.extract::<PyRef<'_, PyTensor>>() {
            let b = b_py.tensor()?;
            return Ok(Self::from_tensor(a.add(&b)?));
        }
        if let Ok(scalar) = other.extract::<f32>() {
            let b = Tensor::new(vec![scalar], vec![1], a.dtype, a.device.clone(), false)?;
            return Ok(Self::from_tensor(a.add(&b)?));
        }
        Err(PyValueError::new_err(
            "Tensor addition requires a Tensor or float scalar ( >_< )",
        ))
    }

    pub fn __radd__(&self, other: f32) -> PyResult<Self> {
        let a = self.tensor()?;
        let b = Tensor::new(vec![other], vec![1], a.dtype, a.device.clone(), false)?;
        Ok(Self::from_tensor(b.add(&a)?))
    }

    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let a = self.tensor()?;
        if let Ok(b_py) = other.extract::<PyRef<'_, PyTensor>>() {
            let b = b_py.tensor()?;
            return Ok(Self::from_tensor(a.sub(&b)?));
        }
        if let Ok(scalar) = other.extract::<f32>() {
            let b = Tensor::new(vec![scalar], vec![1], a.dtype, a.device.clone(), false)?;
            return Ok(Self::from_tensor(a.sub(&b)?));
        }
        Err(PyValueError::new_err(
            "Tensor subtraction requires a Tensor or float scalar ( >_< )",
        ))
    }

    pub fn __rsub__(&self, other: f32) -> PyResult<Self> {
        let a = self.tensor()?;
        let b = Tensor::new(vec![other], vec![1], a.dtype, a.device.clone(), false)?;
        Ok(Self::from_tensor(b.sub(&a)?))
    }

    pub fn __neg__(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.scale(-1.0)?))
    }

    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let a = self.tensor()?;
        if let Ok(b_py) = other.extract::<PyRef<'_, PyTensor>>() {
            let b = b_py.tensor()?;
            return Ok(Self::from_tensor(a.mul(&b)?));
        }
        if let Ok(scalar) = other.extract::<f32>() {
            return Ok(Self::from_tensor(a.scale(scalar)?));
        }
        Err(PyValueError::new_err(
            "Tensor multiplication requires a Tensor or float scalar ( >_< )",
        ))
    }

    pub fn __rmul__(&self, other: f32) -> PyResult<Self> {
        Ok(Self::from_tensor(self.tensor()?.scale(other)?))
    }

    pub fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        self.matmul(other)
    }

    pub fn __repr__(&self) -> PyResult<String> {
        let t = self.tensor()?;
        Ok(format!(
            "Tensor(shape={:?}, dtype='{}', device='{}', requires_grad={})",
            t.shape,
            dtype_str(t.dtype),
            t.device,
            t.requires_grad
        ))
    }
}

// =========================================================================
// AUTOGRAD NO_GRAD CONTROLS
// =========================================================================

#[pyfunction]
pub fn enter_no_grad() {
    no_grad_enter();
}

#[pyfunction]
pub fn exit_no_grad() {
    no_grad_exit();
}

#[pyfunction]
#[pyo3(name = "is_grad_enabled")]
pub fn py_is_grad_enabled() -> bool {
    is_grad_enabled()
}

#[pyfunction]
pub fn clear_autograd_tape() {
    AutogradTape::clear();
}

#[pyfunction]
pub fn clear_grad_store() {
    THREAD_GRADS.with(|cell| {
        *cell.borrow_mut() = GradStore::new();
    });
}

// =========================================================================
// FUNCTIONAL LAYER & LOSS BINDINGS (100% RUST MATH & AUTOGRAD)
// =========================================================================

#[pyfunction]
#[pyo3(signature = (in_features, out_features, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
pub fn init_linear_params(
    in_features: usize,
    out_features: usize,
    bias: bool,
    dtype: String,
    device: String,
    seed: u64,
) -> PyResult<(PyTensor, Option<PyTensor>)> {
    let lin = Linear::new(
        in_features,
        out_features,
        bias,
        parse_dtype(&dtype)?,
        parse_device(Some(&device))?,
        seed,
    )?;
    Ok((
        PyTensor::from_tensor(lin.weight),
        lin.bias.map(PyTensor::from_tensor),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias = None))]
pub fn linear_forward(
    x: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
) -> PyResult<PyTensor> {
    let w = weight.tensor()?;
    let b = bias.map(PyTensor::tensor).transpose()?;
    let lin = Linear::from_weights(w, b)?;
    Ok(PyTensor::from_tensor(lin.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (in_channels, out_channels, kernel_size, stride = (1, 1), padding = (0, 0), dilation = (1, 1), groups = 1, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
#[allow(clippy::too_many_arguments)]
pub fn init_conv2d_params(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    bias: bool,
    dtype: String,
    device: String,
    seed: u64,
) -> PyResult<(PyTensor, Option<PyTensor>)> {
    let conv = Conv2d::new(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        parse_dtype(&dtype)?,
        parse_device(Some(&device))?,
        seed,
    )?;
    Ok((
        PyTensor::from_tensor(conv.weight),
        conv.bias.map(PyTensor::from_tensor),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias = None, stride = (1, 1), padding = (0, 0), dilation = (1, 1), groups = 1))]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward(
    x: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyTensor> {
    let w = weight.tensor()?;
    let b = bias.map(PyTensor::tensor).transpose()?;
    let conv = Conv2d::from_weights(w, b, stride, padding, dilation, groups)?;
    Ok(PyTensor::from_tensor(conv.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (in_channels, depth_multiplier, kernel_size, stride = (1, 1), padding = (0, 0), dilation = (1, 1), bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
#[allow(clippy::too_many_arguments)]
pub fn init_depthwise_conv2d_params(
    in_channels: usize,
    depth_multiplier: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    bias: bool,
    dtype: String,
    device: String,
    seed: u64,
) -> PyResult<(PyTensor, Option<PyTensor>)> {
    let dw = DepthwiseConv2d::new(
        in_channels,
        depth_multiplier,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        parse_dtype(&dtype)?,
        parse_device(Some(&device))?,
        seed,
    )?;
    Ok((
        PyTensor::from_tensor(dw.weight),
        dw.bias.map(PyTensor::from_tensor),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias = None, depth_multiplier = 1, stride = (1, 1), padding = (0, 0), dilation = (1, 1)))]
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_forward(
    x: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    depth_multiplier: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> PyResult<PyTensor> {
    let w = weight.tensor()?;
    let b = bias.map(PyTensor::tensor).transpose()?;
    if w.shape.len() != 4
        || w.shape[1] != 1
        || depth_multiplier == 0
        || !w.shape[0].is_multiple_of(depth_multiplier)
    {
        return Err(PyValueError::new_err(
            "Invalid DepthwiseConv2d weight shape or depth_multiplier ( >_< )",
        ));
    }
    let out_channels = w.shape[0];
    let in_channels = out_channels / depth_multiplier;
    let dw = DepthwiseConv2d {
        dtype: w.dtype,
        device: w.device.clone(),
        kernel_size: (w.shape[2], w.shape[3]),
        weight: w,
        bias: b,
        in_channels,
        depth_multiplier,
        out_channels,
        stride,
        padding,
        dilation,
    };
    Ok(PyTensor::from_tensor(dw.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (x, kernel_size, stride = None, padding = (0, 0)))]
pub fn max_pool2d_forward(
    x: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> PyResult<PyTensor> {
    let st = stride.unwrap_or(kernel_size);
    let pool = MaxPool2d::new(kernel_size, st, padding)?;
    Ok(PyTensor::from_tensor(pool.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (x, kernel_size, stride = None, padding = (0, 0)))]
pub fn avg_pool2d_forward(
    x: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> PyResult<PyTensor> {
    let st = stride.unwrap_or(kernel_size);
    let pool = AvgPool2d::new(kernel_size, st, padding)?;
    Ok(PyTensor::from_tensor(pool.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (x, keep_dims = true))]
pub fn global_avg_pool2d_forward(x: &PyTensor, keep_dims: bool) -> PyResult<PyTensor> {
    let gap = GlobalAvgPool2d::with_keep_dims(keep_dims);
    Ok(PyTensor::from_tensor(gap.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (x, num_features, weight, bias, running_mean, running_var, training = true, eps = 1e-5, momentum = 0.1))]
#[allow(clippy::too_many_arguments)]
pub fn batch_norm2d_forward(
    x: &PyTensor,
    num_features: usize,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    training: bool,
    eps: f32,
    momentum: f32,
) -> PyResult<(PyTensor, Vec<f32>, Vec<f32>)> {
    let xt = x.tensor()?;
    let w = weight.map(PyTensor::tensor).transpose()?;
    let b = bias.map(PyTensor::tensor).transpose()?;
    let mut bn = BatchNorm2d::new(
        num_features,
        eps,
        momentum,
        w.is_some() || b.is_some(),
        xt.dtype,
        xt.device.clone(),
    )?;
    bn.weight = w;
    bn.bias = b;
    bn.training = training;
    if running_mean.len() == num_features && running_var.len() == num_features {
        bn.running_mean = running_mean;
        bn.running_var = running_var;
    }
    let out = bn.forward(&xt)?;
    Ok((PyTensor::from_tensor(out), bn.running_mean, bn.running_var))
}

#[pyfunction]
#[pyo3(signature = (x, start_dim = 1))]
pub fn flatten_forward(x: &PyTensor, start_dim: usize) -> PyResult<PyTensor> {
    let fl = Flatten::with_start_dim(start_dim);
    Ok(PyTensor::from_tensor(fl.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (x, normalized_shape, eps = 1e-5, gamma = None, beta = None))]
pub fn layer_norm_forward(
    x: &PyTensor,
    normalized_shape: usize,
    eps: f32,
    gamma: Option<&PyTensor>,
    beta: Option<&PyTensor>,
) -> PyResult<PyTensor> {
    let xt = x.tensor()?;
    let g = gamma.map(PyTensor::tensor).transpose()?;
    let b = beta.map(PyTensor::tensor).transpose()?;
    let ln = LayerNorm::from_weights(normalized_shape, eps, g, b, xt.dtype, xt.device.clone())?;
    Ok(PyTensor::from_tensor(ln.forward(&xt)?))
}

#[pyfunction]
#[pyo3(signature = (x, normalized_shape, eps, gamma))]
pub fn rms_norm_forward(
    x: &PyTensor,
    normalized_shape: usize,
    eps: f32,
    gamma: &PyTensor,
) -> PyResult<PyTensor> {
    let rn = RMSNorm::from_weight(normalized_shape, eps, gamma.tensor()?)?;
    Ok(PyTensor::from_tensor(rn.forward(&x.tensor()?)?))
}

#[pyfunction]
#[pyo3(signature = (num_embeddings, embedding_dim, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
pub fn init_embedding_params(
    num_embeddings: usize,
    embedding_dim: usize,
    dtype: String,
    device: String,
    seed: u64,
) -> PyResult<PyTensor> {
    let emb = Embedding::new(
        num_embeddings,
        embedding_dim,
        parse_dtype(&dtype)?,
        parse_device(Some(&device))?,
        seed,
    )?;
    Ok(PyTensor::from_tensor(emb.weight))
}

#[pyfunction]
#[pyo3(signature = (indices, weight, batch = None, seq_len = None))]
pub fn embedding_forward(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    weight: &PyTensor,
    batch: Option<usize>,
    seq_len: Option<usize>,
) -> PyResult<PyTensor> {
    let idx = ingest_usize_vec(py, indices)?;
    let emb = Embedding::from_weight(weight.tensor()?)?;
    let out = match (batch, seq_len) {
        (Some(b), Some(s)) => emb.forward_2d(&idx, b, s)?,
        _ => emb.forward(&idx)?,
    };
    Ok(PyTensor::from_tensor(out))
}

#[pyfunction]
#[pyo3(signature = (x, p = 0.5, training = true, seed = 42))]
pub fn dropout_forward(
    x: &PyTensor,
    p: f32,
    training: bool,
    seed: u64,
) -> PyResult<(PyTensor, u64)> {
    let mut drop = Dropout::new(p, seed)?;
    drop.training = training;
    let out = drop.forward(&x.tensor()?)?;
    Ok((PyTensor::from_tensor(out), drop.seed))
}

#[pyfunction]
#[pyo3(signature = (x, max_len, d_model, offset = 0))]
pub fn positional_encoding_forward(
    x: &PyTensor,
    max_len: usize,
    d_model: usize,
    offset: usize,
) -> PyResult<PyTensor> {
    let pe = PositionalEncoding::new(max_len, d_model)?;
    Ok(PyTensor::from_tensor(pe.forward(&x.tensor()?, offset)?))
}

#[pyfunction]
#[pyo3(signature = (x, head_dim, max_seq_len, theta = 10000.0, offset = 0))]
pub fn rope_forward(
    x: &PyTensor,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    offset: usize,
) -> PyResult<PyTensor> {
    let rope = RoPE::new(head_dim, max_seq_len, theta)?;
    Ok(PyTensor::from_tensor(rope.forward(&x.tensor()?, offset)?))
}

#[pyfunction]
#[pyo3(signature = (q, k, v, causal = false, kv_offset = 0))]
pub fn sdpa_forward(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    causal: bool,
    kv_offset: usize,
) -> PyResult<PyTensor> {
    let sdpa = ScaledDotProductAttention::new();
    Ok(PyTensor::from_tensor(sdpa.forward(
        &q.tensor()?,
        &k.tensor()?,
        &v.tensor()?,
        causal,
        kv_offset,
    )?))
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = "mean".to_string()))]
pub fn mse_loss(pred: &PyTensor, target: &PyTensor, reduction: String) -> PyResult<PyTensor> {
    let loss_fn = MSELoss::new(parse_reduction(&reduction)?);
    Ok(PyTensor::from_tensor(
        loss_fn.forward_tensor(&pred.tensor()?, &target.tensor()?)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (logits, targets, label_smoothing = 0.0, ignore_index = None))]
pub fn cross_entropy_loss(
    py: Python<'_>,
    logits: &PyTensor,
    targets: &Bound<'_, PyAny>,
    label_smoothing: f32,
    ignore_index: Option<usize>,
) -> PyResult<PyTensor> {
    let tgt = ingest_usize_vec(py, targets)?;
    let loss_fn = CrossEntropyLoss::new(label_smoothing, ignore_index)?;
    Ok(PyTensor::from_tensor(
        loss_fn.forward_tensor(&logits.tensor()?, &tgt)?,
    ))
}

// =========================================================================
// STATEFUL LAYER & SEQUENCE BINDINGS
// =========================================================================

#[pyclass(
    name = "ResidualBlock",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyResidualBlock {
    inner: Arc<RwLock<ResidualBlock>>,
}

#[pymethods]
impl PyResidualBlock {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, stride = (1, 1), dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let rb = ResidualBlock::new(
            in_channels,
            out_channels,
            stride,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(rb)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("ResidualBlock lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn train(&self) -> PyResult<()> {
        self.inner
            .write()
            .map_err(|_| PyValueError::new_err("ResidualBlock lock poisoned ( >_< )"))?
            .train();
        Ok(())
    }

    pub fn eval(&self) -> PyResult<()> {
        self.inner
            .write()
            .map_err(|_| PyValueError::new_err("ResidualBlock lock poisoned ( >_< )"))?
            .eval();
        Ok(())
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("ResidualBlock lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(name = "RNN", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyRNN {
    inner: Arc<RwLock<RNN>>,
}

#[pymethods]
impl PyRNN {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers = 1, nonlinearity = "tanh".to_string(), bias = true, bidirectional = false, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        nonlinearity: String,
        bias: bool,
        bidirectional: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let rnn = RNN::new(
            input_size,
            hidden_size,
            num_layers,
            parse_nonlinearity(&nonlinearity)?,
            bias,
            bidirectional,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(rnn)),
        })
    }

    #[pyo3(signature = (input, h0 = None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        h0: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("RNN lock poisoned ( >_< )"))?;
        let h0_t = h0.map(PyTensor::tensor).transpose()?;
        let (out, hn) = guard.forward(&input.tensor()?, h0_t.as_ref())?;
        Ok((PyTensor::from_tensor(out), PyTensor::from_tensor(hn)))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("RNN lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(name = "GRU", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGRU {
    inner: Arc<RwLock<GRU>>,
}

#[pymethods]
impl PyGRU {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers = 1, bias = true, bidirectional = false, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        bidirectional: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let gru = GRU::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            bidirectional,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(gru)),
        })
    }

    #[pyo3(signature = (input, h0 = None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        h0: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("GRU lock poisoned ( >_< )"))?;
        let h0_t = h0.map(PyTensor::tensor).transpose()?;
        let (out, hn) = guard.forward(&input.tensor()?, h0_t.as_ref())?;
        Ok((PyTensor::from_tensor(out), PyTensor::from_tensor(hn)))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("GRU lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(name = "LSTM", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLSTM {
    inner: Arc<RwLock<LSTM>>,
}

#[pymethods]
impl PyLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers = 1, bias = true, bidirectional = false, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        bidirectional: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let lstm = LSTM::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            bidirectional,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(lstm)),
        })
    }

    #[pyo3(signature = (input, h0 = None, c0 = None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        h0: Option<&PyTensor>,
        c0: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LSTM lock poisoned ( >_< )"))?;
        let hx_owned = match (h0, c0) {
            (Some(h), Some(c)) => Some((h.tensor()?, c.tensor()?)),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "LSTM forward requires both h0 and c0 when state is provided ( >_< )",
                ));
            }
        };
        let hx_ref = hx_owned.as_ref().map(|(h, c)| (h, c));
        let (out, (hn, cn)) = guard.forward(&input.tensor()?, hx_ref)?;
        Ok((
            PyTensor::from_tensor(out),
            (PyTensor::from_tensor(hn), PyTensor::from_tensor(cn)),
        ))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LSTM lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(name = "KVCache", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyKVCache {
    inner: KVCache,
}

#[pymethods]
impl PyKVCache {
    #[new]
    pub fn new(
        batch: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KVCache::new(batch, num_kv_heads, max_seq_len, head_dim)?,
        })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn cur_len(&self) -> usize {
        self.inner.cur_len
    }

    #[getter]
    pub fn batch(&self) -> usize {
        self.inner.batch
    }

    #[getter]
    pub fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads
    }

    #[getter]
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    #[getter]
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim
    }
}

#[pyclass(
    name = "MultiHeadAttention",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyMultiHeadAttention {
    inner: Arc<RwLock<MultiHeadAttention>>,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads, num_kv_heads = None, bias = true, causal = false, use_rope = false, max_seq_len = 2048, rope_theta = 10000.0, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        bias: bool,
        causal: bool,
        use_rope: bool,
        max_seq_len: usize,
        rope_theta: f32,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(
                "d_model must be divisible by num_heads > 0 ( >_< )",
            ));
        }
        let rope = if use_rope {
            Some(RoPE::new(d_model / num_heads, max_seq_len, rope_theta)?)
        } else {
            None
        };
        let mha = MultiHeadAttention::new(
            d_model,
            num_heads,
            num_kv_heads.unwrap_or(num_heads),
            bias,
            causal,
            rope,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(mha)),
        })
    }

    #[pyo3(signature = (q, k, v, kv_cache = None))]
    pub fn forward(
        &self,
        q: &PyTensor,
        k: &PyTensor,
        v: &PyTensor,
        mut kv_cache: Option<PyRefMut<'_, PyKVCache>>,
    ) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("MultiHeadAttention lock poisoned ( >_< )"))?;
        let cache_ref = kv_cache.as_deref_mut().map(|c| &mut c.inner);
        let out = guard.forward(&q.tensor()?, &k.tensor()?, &v.tensor()?, cache_ref)?;
        Ok(PyTensor::from_tensor(out))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("MultiHeadAttention lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(name = "SwiGLU", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PySwiGLU {
    inner: Arc<RwLock<SwiGLU>>,
}

#[pymethods]
impl PySwiGLU {
    #[new]
    #[pyo3(signature = (d_model, hidden_dim, bias = false, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    pub fn new(
        d_model: usize,
        hidden_dim: usize,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let sg = SwiGLU::new(
            d_model,
            hidden_dim,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(sg)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("SwiGLU lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("SwiGLU lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(
    name = "FeedForward",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFeedForward {
    inner: Arc<RwLock<FeedForward>>,
}

#[pymethods]
impl PyFeedForward {
    #[new]
    #[pyo3(signature = (d_model, d_ff, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    pub fn new(
        d_model: usize,
        d_ff: usize,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let ff = FeedForward::new(
            d_model,
            d_ff,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(ff)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("FeedForward lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("FeedForward lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(
    name = "TransformerEncoderLayer",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTransformerEncoderLayer {
    inner: Arc<RwLock<TransformerEncoderLayer>>,
}

#[pymethods]
impl PyTransformerEncoderLayer {
    #[new]
    #[pyo3(signature = (d_model, num_heads, d_ff, eps = 1e-5, pre_norm = true, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        pre_norm: bool,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let layer = TransformerEncoderLayer::new(
            d_model,
            num_heads,
            d_ff,
            eps,
            pre_norm,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(layer)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerEncoderLayer lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerEncoderLayer lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(
    name = "TransformerEncoder",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTransformerEncoder {
    inner: Arc<RwLock<TransformerEncoder>>,
}

#[pymethods]
impl PyTransformerEncoder {
    #[new]
    #[pyo3(signature = (num_layers, d_model, num_heads, d_ff, eps = 1e-5, pre_norm = true, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        pre_norm: bool,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let enc = TransformerEncoder::new(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            eps,
            pre_norm,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(enc)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerEncoder lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerEncoder lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(
    name = "TransformerDecoderLayer",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTransformerDecoderLayer {
    inner: Arc<RwLock<TransformerDecoderLayer>>,
}

#[pymethods]
impl PyTransformerDecoderLayer {
    #[new]
    #[pyo3(signature = (d_model, num_heads, d_ff, eps = 1e-5, with_cross_attn = true, use_swiglu = false, pre_norm = true, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        eps: f32,
        with_cross_attn: bool,
        use_swiglu: bool,
        pre_norm: bool,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let layer = TransformerDecoderLayer::new(
            d_model,
            num_heads,
            d_ff,
            eps,
            with_cross_attn,
            use_swiglu,
            pre_norm,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(layer)),
        })
    }

    #[pyo3(signature = (x, memory = None, kv_cache = None))]
    pub fn forward(
        &self,
        x: &PyTensor,
        memory: Option<&PyTensor>,
        mut kv_cache: Option<PyRefMut<'_, PyKVCache>>,
    ) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerDecoderLayer lock poisoned ( >_< )"))?;
        let mem_t = memory.map(PyTensor::tensor).transpose()?;
        let cache_ref = kv_cache.as_deref_mut().map(|c| &mut c.inner);
        Ok(PyTensor::from_tensor(guard.forward(
            &x.tensor()?,
            mem_t.as_ref(),
            cache_ref,
        )?))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerDecoderLayer lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

#[pyclass(
    name = "TransformerDecoder",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTransformerDecoder {
    inner: Arc<RwLock<TransformerDecoder>>,
}

#[pymethods]
impl PyTransformerDecoder {
    #[new]
    #[pyo3(signature = (num_layers, d_model, num_heads, d_ff, eps = 1e-5, with_cross_attn = true, use_swiglu = false, pre_norm = true, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
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
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let dec = TransformerDecoder::new(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            eps,
            with_cross_attn,
            use_swiglu,
            pre_norm,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        Ok(Self {
            inner: Arc::new(RwLock::new(dec)),
        })
    }

    #[pyo3(signature = (x, memory = None, kv_caches = None))]
    pub fn forward(
        &self,
        x: &PyTensor,
        memory: Option<&PyTensor>,
        kv_caches: Option<Vec<PyRefMut<'_, PyKVCache>>>,
    ) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerDecoder lock poisoned ( >_< )"))?;
        let mem_t = memory.map(PyTensor::tensor).transpose()?;
        let out = match kv_caches {
            Some(mut caches) => {
                let mut raw_caches: Vec<KVCache> = caches.iter().map(|c| c.inner.clone()).collect();
                let res = guard.forward(&x.tensor()?, mem_t.as_ref(), Some(&mut raw_caches))?;
                for (py_c, updated) in caches.iter_mut().zip(raw_caches) {
                    py_c.inner = updated;
                }
                res
            }
            None => guard.forward(&x.tensor()?, mem_t.as_ref(), None)?,
        };
        Ok(PyTensor::from_tensor(out))
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("TransformerDecoder lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }
}

// =========================================================================
// SEQUENTIAL TRAINER & PERSISTENCE (TAG 100)
// =========================================================================

#[pyclass(
    name = "Sequential",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySequential {
    inner: Sequential,
}

#[pymethods]
impl PySequential {
    #[new]
    #[pyo3(signature = (dtype = "f32".to_string(), device = "auto".to_string()))]
    pub fn new(dtype: String, device: String) -> PyResult<Self> {
        let dt = parse_dtype(&dtype)?;
        let dev = parse_device(Some(&device))?;
        Ok(Self {
            inner: Sequential::new(Vec::new(), dt, dev),
        })
    }

    #[pyo3(signature = (in_features, out_features, bias = true, seed = None))]
    pub fn add_linear(
        &mut self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let s = seed.unwrap_or(42 + self.inner.layers.len() as u64);
        let lin = Linear::new(
            in_features,
            out_features,
            bias,
            self.inner.dtype,
            self.inner.device.clone(),
            s,
        )?;
        self.inner.layers.push(SequentialLayer::Linear(lin));
        Ok(())
    }

    pub fn add_relu(&mut self) {
        self.inner.layers.push(SequentialLayer::ReLU);
    }

    pub fn add_sigmoid(&mut self) {
        self.inner.layers.push(SequentialLayer::Sigmoid);
    }

    pub fn add_tanh(&mut self) {
        self.inner.layers.push(SequentialLayer::Tanh);
    }

    pub fn add_gelu(&mut self) {
        self.inner.layers.push(SequentialLayer::GELU);
    }

    pub fn add_silu(&mut self) {
        self.inner.layers.push(SequentialLayer::SiLU);
    }

    #[pyo3(signature = (p = 0.5, seed = None))]
    pub fn add_dropout(&mut self, p: f32, seed: Option<u64>) -> PyResult<()> {
        let s = seed.unwrap_or(101 + self.inner.layers.len() as u64);
        self.inner
            .layers
            .push(SequentialLayer::Dropout(Dropout::new(p, s)?));
        Ok(())
    }

    #[pyo3(signature = (num_embeddings, embedding_dim, seed = None))]
    pub fn add_embedding(
        &mut self,
        num_embeddings: usize,
        embedding_dim: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let s = seed.unwrap_or(202 + self.inner.layers.len() as u64);
        let emb = Embedding::new(
            num_embeddings,
            embedding_dim,
            self.inner.dtype,
            self.inner.device.clone(),
            s,
        )?;
        self.inner.layers.push(SequentialLayer::Embedding(emb));
        Ok(())
    }

    #[pyo3(signature = (normalized_shape, eps = 1e-5, elementwise_affine = true))]
    pub fn add_layernorm(
        &mut self,
        normalized_shape: usize,
        eps: f32,
        elementwise_affine: bool,
    ) -> PyResult<()> {
        let ln = LayerNorm::new(
            normalized_shape,
            eps,
            elementwise_affine,
            self.inner.dtype,
            self.inner.device.clone(),
        )?;
        self.inner.layers.push(SequentialLayer::LayerNorm(ln));
        Ok(())
    }

    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    pub fn add_rmsnorm(&mut self, normalized_shape: usize, eps: f32) -> PyResult<()> {
        let rn = RMSNorm::new(
            normalized_shape,
            eps,
            self.inner.dtype,
            self.inner.device.clone(),
        )?;
        self.inner.layers.push(SequentialLayer::RMSNorm(rn));
        Ok(())
    }

    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = (1, 1), padding = (0, 0), dilation = (1, 1), groups = 1, bias = true, seed = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let s = seed.unwrap_or(303 + self.inner.layers.len() as u64);
        let conv = Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            self.inner.dtype,
            self.inner.device.clone(),
            s,
        )?;
        self.inner.layers.push(SequentialLayer::Conv2d(conv));
        Ok(())
    }

    #[pyo3(signature = (in_channels, depth_multiplier, kernel_size, stride = (1, 1), padding = (0, 0), dilation = (1, 1), bias = true, seed = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_depthwise_conv2d(
        &mut self,
        in_channels: usize,
        depth_multiplier: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let s = seed.unwrap_or(404 + self.inner.layers.len() as u64);
        let dw = DepthwiseConv2d::new(
            in_channels,
            depth_multiplier,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            self.inner.dtype,
            self.inner.device.clone(),
            s,
        )?;
        self.inner.layers.push(SequentialLayer::DepthwiseConv2d(dw));
        Ok(())
    }

    #[pyo3(signature = (kernel_size, stride = None, padding = (0, 0)))]
    pub fn add_max_pool2d(
        &mut self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> PyResult<()> {
        let pool = MaxPool2d::new(kernel_size, stride.unwrap_or(kernel_size), padding)?;
        self.inner.layers.push(SequentialLayer::MaxPool2d(pool));
        Ok(())
    }

    #[pyo3(signature = (kernel_size, stride = None, padding = (0, 0)))]
    pub fn add_avg_pool2d(
        &mut self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> PyResult<()> {
        let pool = AvgPool2d::new(kernel_size, stride.unwrap_or(kernel_size), padding)?;
        self.inner.layers.push(SequentialLayer::AvgPool2d(pool));
        Ok(())
    }

    #[pyo3(signature = (keep_dims = true))]
    pub fn add_global_avg_pool2d(&mut self, keep_dims: bool) {
        self.inner.layers.push(SequentialLayer::GlobalAvgPool2d(
            GlobalAvgPool2d::with_keep_dims(keep_dims),
        ));
    }

    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1, affine = true))]
    pub fn add_batch_norm2d(
        &mut self,
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
    ) -> PyResult<()> {
        let bn = BatchNorm2d::new(
            num_features,
            eps,
            momentum,
            affine,
            self.inner.dtype,
            self.inner.device.clone(),
        )?;
        self.inner.layers.push(SequentialLayer::BatchNorm2d(bn));
        Ok(())
    }

    #[pyo3(signature = (start_dim = 1))]
    pub fn add_flatten(&mut self, start_dim: usize) {
        self.inner
            .layers
            .push(SequentialLayer::Flatten(Flatten::with_start_dim(start_dim)));
    }

    #[pyo3(signature = (in_channels, out_channels, stride = (1, 1), seed = None))]
    pub fn add_residual_block(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        seed: Option<u64>,
    ) -> PyResult<()> {
        let s = seed.unwrap_or(505 + self.inner.layers.len() as u64);
        let rb = ResidualBlock::new(
            in_channels,
            out_channels,
            stride,
            self.inner.dtype,
            self.inner.device.clone(),
            s,
        )?;
        self.inner.layers.push(SequentialLayer::ResidualBlock(rb));
        Ok(())
    }

    pub fn add_spec(&mut self, spec_kind: &str, args: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = (spec_kind, args);
        Err(PyValueError::new_err(
            "Use typed add_* methods on _ochreml.nn.Sequential ( >_< )",
        ))
    }

    pub fn forward(&mut self, x: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.forward(&x.tensor()?)?))
    }

    #[pyo3(signature = (x, y, out_dim = None, epochs = 10, batch_size = 32, lr = 1e-3, optimizer = "adamw".to_string(), loss = "cross_entropy".to_string(), weight_decay = 0.0, grad_clip = None, shuffle = true, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        out_dim: Option<usize>,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        optimizer: String,
        loss: String,
        weight_decay: f32,
        grad_clip: Option<f32>,
        shuffle: bool,
        seed: u64,
    ) -> PyResult<Vec<f32>> {
        let (x_data, x_shape) = ingest_f32_data(py, x, None)?;
        if x_shape.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "Sequential.fit expects at least 2D input [N, ...], got shape {:?} ( >_< )",
                x_shape
            )));
        }
        let n_samples = x_shape[0];
        let sample_shape = x_shape[1..].to_vec();
        let opt_kind = parse_optim_kind(&optimizer)?;
        let loss_kind = parse_loss_kind(&loss)?;

        let (y_class, y_reg, resolved_out_dim) = match loss_kind {
            LossKind::CrossEntropy => {
                let classes = ingest_usize_vec(py, y)?;
                let inferred_out = classes.iter().copied().max().map_or(1, |m| m + 1);
                let od = out_dim
                    .unwrap_or_else(|| infer_last_linear_out(&self.inner).unwrap_or(inferred_out));
                (Some(classes), None, od)
            }
            LossKind::MSE => {
                let (yr, y_shape) = ingest_f32_data(py, y, None)?;
                let od = out_dim.unwrap_or_else(|| {
                    if y_shape.len() >= 2 {
                        y_shape[1..].iter().copied().product()
                    } else {
                        1
                    }
                });
                (None, Some(yr), od)
            }
        };

        let model = &mut self.inner;
        let history = py.detach(|| {
            model.fit_nd(
                &x_data,
                n_samples,
                &sample_shape,
                y_class.as_deref(),
                y_reg.as_deref(),
                resolved_out_dim,
                epochs,
                batch_size,
                lr,
                opt_kind,
                loss_kind,
                weight_decay,
                grad_clip,
                shuffle,
                seed,
            )
        })?;
        Ok(history)
    }

    #[pyo3(signature = (x, batch_size = 64))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<(Vec<usize>, Bound<'py, PyBytes>)> {
        let (x_data, x_shape) = ingest_f32_data(py, x, None)?;
        if x_shape.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "Sequential.predict expects at least 2D input [N, ...], got shape {:?} ( >_< )",
                x_shape
            )));
        }
        let n_samples = x_shape[0];
        let sample_shape = x_shape[1..].to_vec();
        let (preds, out_dim) = py.detach(|| {
            self.inner
                .predict_nd(&x_data, n_samples, &sample_shape, batch_size)
        })?;
        let bytes = f32_slice_to_pybytes(py, &preds)?;
        Ok((vec![n_samples, out_dim], bytes))
    }

    #[pyo3(signature = (x, batch_size = 64))]
    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<(Vec<usize>, Bound<'py, PyBytes>)> {
        let (x_data, x_shape) = ingest_f32_data(py, x, None)?;
        if x_shape.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "Sequential.predict_proba expects at least 2D input [N, ...], got shape {:?} ( >_< )",
                x_shape
            )));
        }
        let n_samples = x_shape[0];
        let sample_shape = x_shape[1..].to_vec();
        let (mut logits, out_dim) = py.detach(|| {
            let (mut out, od) =
                self.inner
                    .predict_nd(&x_data, n_samples, &sample_shape, batch_size)?;
            for row in out.chunks_exact_mut(od) {
                let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_v).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }
            }
            OchreResult::Ok((out, od))
        })?;
        let _ = &mut logits;
        let bytes = f32_slice_to_pybytes(py, &logits)?;
        Ok((vec![n_samples, out_dim], bytes))
    }

    pub fn train(&mut self) {
        self.inner.train();
    }

    pub fn eval(&mut self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    pub fn to_dit_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_dit_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = "auto".to_string()))]
    pub fn from_dit_bytes(bytes: &[u8], device: String) -> PyResult<Self> {
        let dev = parse_device(Some(&device))?;
        Ok(Self {
            inner: Sequential::from_dit_bytes(bytes, dev)?,
        })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_dit_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {e} (´；ω；`)")))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = "auto".to_string()))]
    pub fn load(path: &str, device: String) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {e} (´；ω；`)"))
        })?;
        Self::from_dit_bytes(&bytes, device)
    }

    #[getter]
    pub fn dtype(&self) -> &'static str {
        dtype_str(self.inner.dtype)
    }

    #[getter]
    pub fn device(&self) -> String {
        self.inner.device.to_string()
    }

    #[getter]
    pub fn training(&self) -> bool {
        self.inner.training
    }

    pub fn __len__(&self) -> usize {
        self.inner.layers.len()
    }
}

fn infer_last_linear_out(seq: &Sequential) -> Option<usize> {
    for layer in seq.layers.iter().rev() {
        if let SequentialLayer::Linear(lin) = layer {
            return Some(lin.out_features);
        }
    }
    None
}

// =========================================================================
// CAUSAL LM & LORA BINDINGS (TAGS 101 & 102)
// =========================================================================

#[pyclass(name = "CausalLM", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCausalLM {
    inner: Arc<RwLock<CausalLM>>,
}

#[pymethods]
impl PyCausalLM {
    #[new]
    #[pyo3(signature = (vocab_size, d_model, num_layers, num_heads, num_kv_heads = None, ffn_hidden_dim = None, max_seq_len = 2048, rope_theta = 10000.0, norm_eps = 1e-5, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        ffn_hidden_dim: Option<usize>,
        max_seq_len: usize,
        rope_theta: f32,
        norm_eps: f32,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let cfg = CausalLMConfig {
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            num_kv_heads: num_kv_heads.unwrap_or(num_heads),
            ffn_hidden_dim: ffn_hidden_dim.unwrap_or(4 * d_model),
            max_seq_len,
            rope_theta,
            norm_eps,
            dtype: parse_dtype(&dtype)?,
        };
        let lm = CausalLM::new(cfg, parse_device(Some(&device))?, seed)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(lm)),
        })
    }

    pub fn new_kv_caches(&self, batch: usize) -> PyResult<Vec<PyKVCache>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        let caches = guard.new_kv_caches(batch)?;
        Ok(caches.into_iter().map(|c| PyKVCache { inner: c }).collect())
    }

    #[pyo3(signature = (input_ids, batch = None, seq_len = None, kv_caches = None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        batch: Option<usize>,
        seq_len: Option<usize>,
        kv_caches: Option<Vec<PyRefMut<'_, PyKVCache>>>,
    ) -> PyResult<PyTensor> {
        let (ids, b, s) = if let (Some(b), Some(s)) = (batch, seq_len) {
            (ingest_usize_vec(py, input_ids)?, b, s)
        } else if let Ok(rows) = input_ids.extract::<Vec<Vec<usize>>>() {
            let b = rows.len();
            let s = rows.first().map_or(0, Vec::len);
            (rows.into_iter().flatten().collect(), b, s)
        } else {
            let flat = ingest_usize_vec(py, input_ids)?;
            let s = flat.len();
            (flat, 1, s)
        };

        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        let out = match kv_caches {
            Some(mut caches) => {
                let mut raw_caches: Vec<KVCache> = caches.iter().map(|c| c.inner.clone()).collect();
                let res = guard.forward_ids(&ids, b, s, Some(&mut raw_caches))?;
                for (py_c, updated) in caches.iter_mut().zip(raw_caches) {
                    py_c.inner = updated;
                }
                res
            }
            None => guard.forward_ids(&ids, b, s, None)?,
        };
        Ok(PyTensor::from_tensor(out))
    }

    #[pyo3(signature = (prompt_ids, max_new_tokens = 32, temperature = 1.0, top_k = 0, top_p = 1.0, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        py: Python<'_>,
        prompt_ids: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> PyResult<Vec<usize>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        let out = py.detach(|| {
            guard.generate(&prompt_ids, max_new_tokens, temperature, top_k, top_p, seed)
        })?;
        Ok(out)
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }

    pub fn num_parameters(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        Ok(guard.parameters().iter().map(|t| t.numel()).sum())
    }

    pub fn to_dit_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        Ok(PyBytes::new(py, &guard.to_dit_bytes()))
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = "auto".to_string()))]
    pub fn from_dit_bytes(bytes: &[u8], device: String) -> PyResult<Self> {
        let dev = parse_device(Some(&device))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(CausalLM::from_dit_bytes(bytes, dev)?)),
        })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?;
        fs::write(path, guard.to_dit_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {e} (´；ω；`)")))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = "auto".to_string()))]
    pub fn load(path: &str, device: String) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {e} (´；ω；`)"))
        })?;
        Self::from_dit_bytes(&bytes, device)
    }

    #[getter]
    pub fn vocab_size(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .vocab_size)
    }

    #[getter]
    pub fn d_model(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .d_model)
    }

    #[getter]
    pub fn num_layers(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .num_layers)
    }

    #[getter]
    pub fn num_heads(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .num_heads)
    }

    #[getter]
    pub fn num_kv_heads(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .num_kv_heads)
    }

    #[getter]
    pub fn ffn_hidden_dim(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .ffn_hidden_dim)
    }

    #[getter]
    pub fn max_seq_len(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .max_seq_len)
    }

    #[getter]
    pub fn rope_theta(&self) -> PyResult<f32> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .rope_theta)
    }

    #[getter]
    pub fn norm_eps(&self) -> PyResult<f32> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .config
            .norm_eps)
    }

    #[getter]
    pub fn dtype(&self) -> PyResult<&'static str> {
        Ok(dtype_str(
            self.inner
                .read()
                .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
                .config
                .dtype,
        ))
    }

    #[getter]
    pub fn device(&self) -> PyResult<String> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("CausalLM lock poisoned ( >_< )"))?
            .device
            .to_string())
    }
}

#[pyclass(
    name = "LoRALinear",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLoRALinear {
    inner: Arc<RwLock<LoRALinear>>,
}

#[pymethods]
impl PyLoRALinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, rank = 4, alpha = 8.0, bias = true, dtype = "f32".to_string(), device = "auto".to_string(), seed = 42))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        bias: bool,
        dtype: String,
        device: String,
        seed: u64,
    ) -> PyResult<Self> {
        let base = Linear::new(
            in_features,
            out_features,
            bias,
            parse_dtype(&dtype)?,
            parse_device(Some(&device))?,
            seed,
        )?;
        let lora = LoRALinear::new(base, rank, alpha, seed.wrapping_add(1009))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(lora)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (weight, bias = None, rank = 4, alpha = 8.0, seed = 42))]
    pub fn from_linear(
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let w = weight.tensor()?;
        let b = bias.map(PyTensor::tensor).transpose()?;
        let base = Linear::from_weights(w, b)?;
        let lora = LoRALinear::new(base, rank, alpha, seed)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(lora)),
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.forward(&x.tensor()?)?))
    }

    pub fn merge_weights(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        Ok(guard.merge_weights()?)
    }

    pub fn unmerge_weights(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        Ok(guard.unmerge_weights()?)
    }

    #[pyo3(signature = (name = "default".to_string()))]
    pub fn save_adapter_bytes<'py>(
        &self,
        py: Python<'py>,
        name: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        let mut state = LoRAAdapterState::new();
        state.insert(name, &guard)?;
        Ok(PyBytes::new(py, &state.to_dit_bytes()))
    }

    #[pyo3(signature = (bytes, name = "default".to_string()))]
    pub fn load_adapter_bytes(&self, bytes: &[u8], name: String) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        let dev = guard.base.device.clone();
        let state = LoRAAdapterState::from_dit_bytes(bytes, dev)?;
        state.apply_to(&name, &mut guard)?;
        Ok(())
    }

    pub fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        let owner: Arc<dyn ParamOwner> = Arc::clone(&self.inner) as Arc<dyn ParamOwner>;
        Ok(export_owned_params(owner, guard.parameters()))
    }

    #[getter]
    pub fn lora_a(&self) -> PyResult<PyTensor> {
        let params = self.parameters()?;
        Ok(params[0].clone())
    }

    #[getter]
    pub fn lora_b(&self) -> PyResult<PyTensor> {
        let params = self.parameters()?;
        Ok(params[1].clone())
    }

    #[getter]
    pub fn base_weight(&self) -> PyResult<PyTensor> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        Ok(PyTensor::from_tensor(guard.base.weight.clone()))
    }

    #[getter]
    pub fn base_bias(&self) -> PyResult<Option<PyTensor>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        Ok(guard.base.bias.clone().map(PyTensor::from_tensor))
    }

    #[getter]
    pub fn merged(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?
            .merged)
    }

    #[getter]
    pub fn rank(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?
            .rank)
    }

    #[getter]
    pub fn alpha(&self) -> PyResult<f32> {
        Ok(self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?
            .alpha)
    }
}

#[pyclass(
    name = "LoRAAdapterState",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLoRAAdapterState {
    inner: LoRAAdapterState,
}

#[pymethods]
impl PyLoRAAdapterState {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: LoRAAdapterState::new(),
        }
    }

    pub fn insert(&mut self, name: String, layer: &PyLoRALinear) -> PyResult<()> {
        let guard = layer
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        self.inner.insert(name, &guard)?;
        Ok(())
    }

    pub fn apply_to(&self, name: &str, target: &PyLoRALinear) -> PyResult<()> {
        let mut guard = target
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("LoRALinear lock poisoned ( >_< )"))?;
        self.inner.apply_to(name, &mut guard)?;
        Ok(())
    }

    pub fn names(&self) -> Vec<String> {
        self.inner.adapters.iter().map(|a| a.name.clone()).collect()
    }

    pub fn to_dit_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_dit_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = "auto".to_string()))]
    pub fn from_dit_bytes(bytes: &[u8], device: String) -> PyResult<Self> {
        let dev = parse_device(Some(&device))?;
        Ok(Self {
            inner: LoRAAdapterState::from_dit_bytes(bytes, dev)?,
        })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_dit_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {e} (´；ω；`)")))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = "auto".to_string()))]
    pub fn load(path: &str, device: String) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {e} (´；ω；`)"))
        })?;
        Self::from_dit_bytes(&bytes, device)
    }
}

impl Default for PyLoRAAdapterState {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// OPTIMIZERS, GRAD CLIPPING & LR SCHEDULERS
// =========================================================================

fn optimizer_step_params(
    params: &[PyRef<'_, PyTensor>],
    grads: Option<&PyGradStore>,
    step_fn: impl FnOnce(&mut [&mut Tensor], &GradStore) -> OchreResult<()>,
) -> PyResult<()> {
    let mut owned: Vec<Tensor> = Vec::with_capacity(params.len());
    for p in params {
        owned.push(p.tensor()?);
    }
    let mut refs: Vec<&mut Tensor> = owned.iter_mut().collect();
    match grads {
        Some(gs) => step_fn(&mut refs, &gs.inner)?,
        None => THREAD_GRADS.with(|cell| {
            let store = cell.borrow();
            step_fn(&mut refs, &store)
        })?,
    }
    for (p, updated) in params.iter().zip(owned) {
        if let Ok(mut guard) = p.inner.write() {
            *guard = updated.clone();
        }
        if let Some(ref link) = p.link {
            link.owner.update_param(link.index, &updated.data)?;
        }
    }
    Ok(())
}

fn zero_grad_params(params: &[PyRef<'_, PyTensor>]) -> PyResult<()> {
    for p in params {
        p.zero_grad()?;
    }
    Ok(())
}

#[pyclass(name = "SGD", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PySGD {
    inner: SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (lr = 1e-2, momentum = 0.0, nesterov = false, weight_decay = 0.0))]
    pub fn new(lr: f32, momentum: f32, nesterov: bool, weight_decay: f32) -> PyResult<Self> {
        Ok(Self {
            inner: SGD::new(lr, momentum, nesterov, weight_decay)?,
        })
    }

    #[pyo3(signature = (params, grads = None))]
    pub fn step(
        &mut self,
        params: Vec<PyRef<'_, PyTensor>>,
        grads: Option<&PyGradStore>,
    ) -> PyResult<()> {
        optimizer_step_params(&params, grads, |p, g| self.inner.step(p, g))
    }

    pub fn zero_grad(&self, params: Vec<PyRef<'_, PyTensor>>) -> PyResult<()> {
        zero_grad_params(&params)
    }

    #[getter]
    pub fn lr(&self) -> f32 {
        self.inner.lr
    }

    #[setter]
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.lr = lr;
    }
}

#[pyclass(name = "Adam", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAdam {
    inner: Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0))]
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> PyResult<Self> {
        Ok(Self {
            inner: Adam::new(lr, beta1, beta2, eps, weight_decay)?,
        })
    }

    #[pyo3(signature = (params, grads = None))]
    pub fn step(
        &mut self,
        params: Vec<PyRef<'_, PyTensor>>,
        grads: Option<&PyGradStore>,
    ) -> PyResult<()> {
        optimizer_step_params(&params, grads, |p, g| self.inner.step(p, g))
    }

    pub fn zero_grad(&self, params: Vec<PyRef<'_, PyTensor>>) -> PyResult<()> {
        zero_grad_params(&params)
    }

    #[getter]
    pub fn lr(&self) -> f32 {
        self.inner.lr
    }

    #[setter]
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.lr = lr;
    }
}

#[pyclass(name = "AdamW", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAdamW {
    inner: AdamW,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01, device = "auto".to_string()))]
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        device: String,
    ) -> PyResult<Self> {
        let dev = parse_device(Some(&device))?;
        Ok(Self {
            inner: AdamW::new(lr, beta1, beta2, eps, weight_decay, dev)?,
        })
    }

    #[pyo3(signature = (params, grads = None))]
    pub fn step(
        &mut self,
        params: Vec<PyRef<'_, PyTensor>>,
        grads: Option<&PyGradStore>,
    ) -> PyResult<()> {
        optimizer_step_params(&params, grads, |p, g| self.inner.step(p, g))
    }

    pub fn zero_grad(&self, params: Vec<PyRef<'_, PyTensor>>) -> PyResult<()> {
        zero_grad_params(&params)
    }

    #[getter]
    pub fn lr(&self) -> f32 {
        self.inner.lr
    }

    #[setter]
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.lr = lr;
    }
}

#[pyfunction]
#[pyo3(signature = (params, max_norm, grads = None))]
pub fn clip_grad_norm_(
    params: Vec<PyRef<'_, PyTensor>>,
    max_norm: f32,
    mut grads: Option<PyRefMut<'_, PyGradStore>>,
) -> PyResult<f32> {
    let mut ids = Vec::with_capacity(params.len());
    for p in &params {
        if let Some(nid) = p.tensor()?.node_id {
            ids.push(nid);
        }
    }
    if let Some(ref mut gs) = grads {
        return Ok(clip_grad_store_norm_(&mut gs.inner, &ids, max_norm));
    }
    Ok(THREAD_GRADS.with(|cell| {
        let mut store = cell.borrow_mut();
        clip_grad_store_norm_(&mut store, &ids, max_norm)
    }))
}

#[pyclass(name = "StepLR", module = "ochreml._ochreml.nn", skip_from_py_object)]
#[derive(Clone)]
pub struct PyStepLR {
    inner: StepLR,
}

#[pymethods]
impl PyStepLR {
    #[new]
    #[pyo3(signature = (base_lr, step_size, gamma = 0.1))]
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> PyResult<Self> {
        Ok(Self {
            inner: StepLR::new(base_lr, step_size, gamma)?,
        })
    }

    pub fn get_lr(&self) -> f32 {
        self.inner.get_lr()
    }

    pub fn step(&mut self) -> f32 {
        self.inner.step()
    }

    #[getter]
    pub fn current_epoch(&self) -> usize {
        self.inner.current_epoch
    }
}

#[pyclass(
    name = "CosineAnnealingLR",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCosineAnnealingLR {
    inner: CosineAnnealingLR,
}

#[pymethods]
impl PyCosineAnnealingLR {
    #[new]
    #[pyo3(signature = (base_lr, t_max, eta_min = 0.0))]
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> PyResult<Self> {
        Ok(Self {
            inner: CosineAnnealingLR::new(base_lr, t_max, eta_min)?,
        })
    }

    pub fn get_lr(&self) -> f32 {
        self.inner.get_lr()
    }

    pub fn step(&mut self) -> f32 {
        self.inner.step()
    }

    #[getter]
    pub fn current_epoch(&self) -> usize {
        self.inner.current_epoch
    }
}

#[pyclass(
    name = "WarmupCosineLR",
    module = "ochreml._ochreml.nn",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyWarmupCosineLR {
    inner: WarmupCosineLR,
}

#[pymethods]
impl PyWarmupCosineLR {
    #[new]
    #[pyo3(signature = (base_lr, warmup_steps, total_steps, min_lr = 0.0))]
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: WarmupCosineLR::new(base_lr, warmup_steps, total_steps, min_lr)?,
        })
    }

    pub fn get_lr(&self) -> f32 {
        self.inner.get_lr()
    }

    pub fn step(&mut self) -> f32 {
        self.inner.step()
    }

    #[getter]
    pub fn current_step(&self) -> usize {
        self.inner.current_step
    }
}

/// Registers all `_ochreml.nn` classes and functions on the `nn` PyModule.
pub fn register_nn_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyGradStore>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PyCausalLM>()?;
    m.add_class::<PyLoRALinear>()?;
    m.add_class::<PyLoRAAdapterState>()?;
    m.add_class::<PyKVCache>()?;
    m.add_class::<PyResidualBlock>()?;
    m.add_class::<PyRNN>()?;
    m.add_class::<PyGRU>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyMultiHeadAttention>()?;
    m.add_class::<PySwiGLU>()?;
    m.add_class::<PyFeedForward>()?;
    m.add_class::<PyTransformerEncoderLayer>()?;
    m.add_class::<PyTransformerEncoder>()?;
    m.add_class::<PyTransformerDecoderLayer>()?;
    m.add_class::<PyTransformerDecoder>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyWarmupCosineLR>()?;

    m.add_function(wrap_pyfunction!(enter_no_grad, m)?)?;
    m.add_function(wrap_pyfunction!(exit_no_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(clear_autograd_tape, m)?)?;
    m.add_function(wrap_pyfunction!(clear_grad_store, m)?)?;
    m.add_function(wrap_pyfunction!(init_linear_params, m)?)?;
    m.add_function(wrap_pyfunction!(linear_forward, m)?)?;
    m.add_function(wrap_pyfunction!(init_conv2d_params, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(init_depthwise_conv2d_params, m)?)?;
    m.add_function(wrap_pyfunction!(depthwise_conv2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(max_pool2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(avg_pool2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(global_avg_pool2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm2d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(rms_norm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(init_embedding_params, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_forward, m)?)?;
    m.add_function(wrap_pyfunction!(dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(positional_encoding_forward, m)?)?;
    m.add_function(wrap_pyfunction!(rope_forward, m)?)?;
    m.add_function(wrap_pyfunction!(sdpa_forward, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(clip_grad_norm_, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TAG_NN_SEQUENTIAL, TAG_NN_STATE_DICT, TAG_NN_TRANSFORMER_LM};

    #[test]
    fn test_nn_py_binding_parsers_and_dit_tags() {
        assert_eq!(TAG_NN_SEQUENTIAL, 100);
        assert_eq!(TAG_NN_TRANSFORMER_LM, 101);
        assert_eq!(TAG_NN_STATE_DICT, 102);

        assert_eq!(parse_dtype("float32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("f32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("bfloat16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("bf16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("float16").unwrap(), DType::F16);
        assert_eq!(parse_dtype("f16").unwrap(), DType::F16);
        assert!(parse_dtype("int8").is_err());

        assert_eq!(parse_optim_kind("adamw").unwrap(), OptimKind::AdamW);
        assert_eq!(parse_optim_kind("Adam").unwrap(), OptimKind::Adam);
        assert_eq!(parse_optim_kind("SGD").unwrap(), OptimKind::SGD);
        assert!(parse_optim_kind("rmsprop").is_err());

        assert_eq!(
            parse_loss_kind("cross_entropy").unwrap(),
            LossKind::CrossEntropy
        );
        assert_eq!(parse_loss_kind("mse").unwrap(), LossKind::MSE);
        assert_eq!(parse_reduction("mean").unwrap(), Reduction::Mean);
        assert_eq!(parse_reduction("sum").unwrap(), Reduction::Sum);
        assert_eq!(parse_reduction("none").unwrap(), Reduction::None);
        assert_eq!(parse_nonlinearity("tanh").unwrap(), RnnNonlinearity::Tanh);
        assert_eq!(parse_nonlinearity("relu").unwrap(), RnnNonlinearity::Relu);
    }
}
