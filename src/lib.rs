// PyO3 bindings module for OchreML (＾▽＾)
// Exposes Rust Linear Regression and Decision Tree implementations with Hardware Acceleration to Python.

pub mod backend;
pub mod decision_tree;
pub mod device;
pub mod error;
pub mod linear_regression;
pub mod matrix;

use decision_tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeNode};
use device::parse_device;
use error::{OchreError, OchreResult};
use linear_regression::LinearRegression;
use matrix::Matrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::fs;

// Auto-convert OchreError to PyValueError with kaomoji preserved (*≧ω≦*)
impl From<OchreError> for pyo3::PyErr {
    fn from(err: OchreError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

const DIT_MAGIC: &[u8; 4] = b"DIT\x01";
const TAG_LINEAR_REGRESSION: u8 = 1;
const TAG_DT_CLASSIFIER: u8 = 2;
const TAG_DT_REGRESSOR: u8 = 3;

fn validate_dit_path(path: &str) -> OchreResult<()> {
    if !path.trim().to_lowercase().ends_with(".dit") {
        return Err(OchreError::InvalidModelExtension(path.to_string()));
    }
    Ok(())
}

// Minimal binary writer/reader helpers (ponytail: stdlib byte cursor, zero external crate deps)
fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn write_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(u8::from(v));
}
fn write_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u64).to_le_bytes());
}
fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_usize(buf, s.len());
    buf.extend_from_slice(s.as_bytes());
}
fn write_opt_str(buf: &mut Vec<u8>, s: &Option<String>) {
    match s {
        Some(val) => {
            write_u8(buf, 1);
            write_str(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_opt_usize(buf: &mut Vec<u8>, v: Option<usize>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_usize(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_opt_f64(buf: &mut Vec<u8>, v: Option<f64>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_f64(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_f64_vec(buf: &mut Vec<u8>, v: &[f64]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f64(buf, x);
    }
}
fn write_opt_f64_vec(buf: &mut Vec<u8>, v: &Option<Vec<f64>>) {
    match v {
        Some(vec) => {
            write_u8(buf, 1);
            write_f64_vec(buf, vec);
        }
        None => write_u8(buf, 0),
    }
}
fn write_tree_node(buf: &mut Vec<u8>, node: &TreeNode) {
    match node {
        TreeNode::Leaf {
            value,
            probabilities,
        } => {
            write_u8(buf, 0);
            write_f64(buf, *value);
            write_usize(buf, probabilities.len());
            for &(cls, prob) in probabilities {
                write_f64(buf, cls);
                write_f64(buf, prob);
            }
        }
        TreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            write_u8(buf, 1);
            write_usize(buf, *feature_index);
            write_f64(buf, *threshold);
            write_tree_node(buf, left);
            write_tree_node(buf, right);
        }
    }
}
fn write_opt_tree_node(buf: &mut Vec<u8>, node: &Option<TreeNode>) {
    match node {
        Some(n) => {
            write_u8(buf, 1);
            write_tree_node(buf, n);
        }
        None => write_u8(buf, 0),
    }
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_exact(&mut self, len: usize) -> OchreResult<&'a [u8]> {
        if self.pos + len > self.data.len() {
            return Err(OchreError::CorruptedDitFile);
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }
    fn read_u8(&mut self) -> OchreResult<u8> {
        Ok(self.read_exact(1)?[0])
    }
    fn read_bool(&mut self) -> OchreResult<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_usize(&mut self) -> OchreResult<usize> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()) as usize)
    }
    fn read_f64(&mut self) -> OchreResult<f64> {
        let b = self.read_exact(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_str(&mut self) -> OchreResult<String> {
        let len = self.read_usize()?;
        let b = self.read_exact(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| OchreError::InvalidUtf8InDit)
    }
    fn read_opt_str(&mut self) -> OchreResult<Option<String>> {
        if self.read_bool()? {
            Ok(Some(self.read_str()?))
        } else {
            Ok(None)
        }
    }
    fn read_opt_usize(&mut self) -> OchreResult<Option<usize>> {
        if self.read_bool()? {
            Ok(Some(self.read_usize()?))
        } else {
            Ok(None)
        }
    }
    fn read_opt_f64(&mut self) -> OchreResult<Option<f64>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64()?))
        } else {
            Ok(None)
        }
    }
    fn read_f64_vec(&mut self) -> OchreResult<Vec<f64>> {
        let len = self.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f64()?);
        }
        Ok(vec)
    }
    fn read_opt_f64_vec(&mut self) -> OchreResult<Option<Vec<f64>>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64_vec()?))
        } else {
            Ok(None)
        }
    }
    fn read_tree_node(&mut self) -> OchreResult<TreeNode> {
        match self.read_u8()? {
            0 => {
                let value = self.read_f64()?;
                let len = self.read_usize()?;
                let mut probabilities = Vec::with_capacity(len);
                for _ in 0..len {
                    probabilities.push((self.read_f64()?, self.read_f64()?));
                }
                Ok(TreeNode::Leaf {
                    value,
                    probabilities,
                })
            }
            1 => {
                let feature_index = self.read_usize()?;
                let threshold = self.read_f64()?;
                let left = Box::new(self.read_tree_node()?);
                let right = Box::new(self.read_tree_node()?);
                Ok(TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
    fn read_opt_tree_node(&mut self) -> OchreResult<Option<TreeNode>> {
        if self.read_bool()? {
            Ok(Some(self.read_tree_node()?))
        } else {
            Ok(None)
        }
    }
}

fn check_magic_and_tag(reader: &mut ByteReader<'_>, expected_tag: u8) -> OchreResult<()> {
    let magic = reader.read_exact(4)?;
    if magic != DIT_MAGIC {
        return Err(OchreError::InvalidDitHeader);
    }
    let tag = reader.read_u8()?;
    if tag != expected_tag {
        return Err(OchreError::DitTagMismatch {
            expected: expected_tag,
            actual: tag,
        });
    }
    Ok(())
}

impl LinearRegression {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_LINEAR_REGRESSION);
        write_bool(&mut buf, self.fit_intercept);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        match &self.coef_multi_ {
            Some(m) => {
                write_u8(&mut buf, 1);
                write_usize(&mut buf, m.rows);
                write_usize(&mut buf, m.cols);
                write_f64_vec(&mut buf, &m.data);
            }
            None => write_u8(&mut buf, 0),
        }
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LINEAR_REGRESSION)?;
        let fit_intercept = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let dev = parse_device(device.or(Some("auto")))?;
        let device_ = if saved_device.is_some() || coef_.is_some() {
            Some(dev.to_string())
        } else {
            None
        };

        let (is_multi_target_, n_targets_, coef_multi_, intercept_multi_) = if r.pos < r.data.len() {
            let is_multi = r.read_bool()?;
            let n_targets = r.read_usize()?;
            let coef_multi = if r.read_bool()? {
                let rows = r.read_usize()?;
                let cols = r.read_usize()?;
                let data = r.read_f64_vec()?;
                Some(Matrix::from_vec(rows, cols, data)?)
            } else {
                None
            };
            let intercept_multi = r.read_opt_f64_vec()?;
            (is_multi, n_targets, coef_multi, intercept_multi)
        } else {
            let coef_multi = match &coef_ {
                Some(c) => Some(Matrix::from_vec(1, c.len(), c.clone())?),
                None => None,
            };
            let intercept_multi = intercept_.map(|v| vec![v]);
            (false, 1, coef_multi, intercept_multi)
        };

        Ok(Self {
            fit_intercept,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl DecisionTreeClassifier {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_DT_CLASSIFIER);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.device_);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_tree_node(&mut buf, &self.root);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DT_CLASSIFIER)?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let root = r.read_opt_tree_node()?;
        let dev = parse_device(device.or(Some("auto")))?;
        let device_ = if saved_device.is_some() || root.is_some() {
            Some(dev.to_string())
        } else {
            None
        };
        Ok(Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device: dev,
            device_,
            root,
            classes_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl DecisionTreeRegressor {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_DT_REGRESSOR);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_tree_node(&mut buf, &self.root);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DT_REGRESSOR)?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let root = r.read_opt_tree_node()?;
        let dev = parse_device(device.or(Some("auto")))?;
        let device_ = if saved_device.is_some() || root.is_some() {
            Some(dev.to_string())
        } else {
            None
        };
        Ok(Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device: dev,
            device_,
            root,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

#[pyfunction]
pub fn get_available_devices() -> Vec<String> {
    device::get_available_devices_list()
}

#[pyfunction]
pub fn get_device_info() -> HashMap<String, String> {
    device::get_device_info_map()
}

fn buffer_to_matrix(py: Python<'_>, x: &pyo3::buffer::PyBuffer<f64>) -> PyResult<Matrix> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Feature matrix X must be a 2D array (n_samples, n_features), got {}D ( >_< )",
            shape.len()
        )));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut data = vec![0.0; rows * cols];
    x.copy_to_slice(py, &mut data)
        .map_err(|e| PyValueError::new_err(format!("Failed to copy buffer data: {} (´；ω；`)", e)))?;
    Ok(Matrix::from_vec(rows, cols, data)?)
}

fn buffer_to_target_vector(py: Python<'_>, y: &pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
    let shape = y.shape();
    if shape.len() > 2 || (shape.len() == 2 && shape[1] != 1) {
        return Err(PyValueError::new_err(format!(
            "Target vector y must be a 1D vector (n_samples,) or 2D column (n_samples, 1) (・`ω´・)"
        )));
    }
    let count = y.item_count();
    let mut data = vec![0.0; count];
    y.copy_to_slice(py, &mut data)
        .map_err(|e| PyValueError::new_err(format!("Failed to copy target buffer: {} (´；ω；`)", e)))?;
    Ok(data)
}

fn buffer_to_target_matrix(py: Python<'_>, y: &pyo3::buffer::PyBuffer<f64>) -> PyResult<(Matrix, bool)> {
    let shape = y.shape();
    let (rows, cols, is_2d) = if shape.len() == 1 {
        (shape[0], 1, false)
    } else if shape.len() == 2 {
        (shape[0], shape[1], true)
    } else {
        return Err(PyValueError::new_err(format!(
            "Target y must be a 1D vector or 2D matrix (n_samples, n_targets), got {}D (・`ω´・)",
            shape.len()
        )));
    };
    let mut data = vec![0.0; rows * cols];
    y.copy_to_slice(py, &mut data)
        .map_err(|e| PyValueError::new_err(format!("Failed to copy target buffer: {} (´；ω；`)", e)))?;
    let mat = Matrix::from_vec(rows, cols, data)?;
    Ok((mat, is_2d))
}

fn buffer_to_sample_weight(py: Python<'_>, sw: &pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
    let shape = sw.shape();
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "sample_weight must be a 1D vector (n_samples,), got {}D ( >_< )",
            shape.len()
        )));
    }
    let count = sw.item_count();
    let mut data = vec![0.0; count];
    sw.copy_to_slice(py, &mut data)
        .map_err(|e| PyValueError::new_err(format!("Failed to copy sample_weight buffer: {} (´；ω；`)", e)))?;
    Ok(data)
}

#[pyclass(from_py_object, name = "LinearRegression")]
#[derive(Clone)]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept = true, device = None))]
    pub fn new(fit_intercept: bool, device: Option<String>) -> PyResult<Self> {
        let inner = LinearRegression::new(fit_intercept, device)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted_1d(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        self.inner
            .fit_multi(&x_mat, &y_mat, sample_weight.as_deref(), true)?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let (target_mat, is_2d) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_multi(&mat, &target_mat, sw_vec.as_deref(), is_2d)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let mat = Matrix::from_2d(&x)?;
        let preds = self.inner.predict_multi(&mat)?;
        let mut out = Vec::with_capacity(preds.rows);
        for r in 0..preds.rows {
            let offset = r * preds.cols;
            out.push(preds.data[offset..offset + preds.cols].to_vec());
        }
        Ok(out)
    }

    pub fn predict_buffer(&self, py: Python<'_>, x: pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        if sample_weight.is_none() && !self.inner.is_multi_target_ {
            return Ok(self.inner.score(&x_mat, &y)?);
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y)?;
        Ok(self.inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_multi(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        Ok(self.inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_matrix(py, &x)?;
        let (y_mat, _) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner
            .score_multi(&x_mat, &y_mat, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e)))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e)))?;
        let inner = LinearRegression::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = LinearRegression::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = LinearRegression::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(|m| {
            let mut out = Vec::with_capacity(m.rows);
            for r in 0..m.rows {
                let offset = r * m.cols;
                out.push(m.data[offset..offset + m.cols].to_vec());
            }
            out
        })
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }

    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }

    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "DecisionTreeClassifier")]
#[derive(Clone)]
pub struct PyDecisionTreeClassifier {
    inner: DecisionTreeClassifier,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None, device = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = DecisionTreeClassifier::new(criterion, max_depth, min_samples_split, min_samples_leaf, device)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner.fit(&mat, &y)?;
        Ok(())
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        self.inner.fit(&mat, &target)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_buffer(&self, py: Python<'_>, x: pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.score(&mat, &y)?)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        Ok(self.inner.score(&mat, &target)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e)))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e)))?;
        let inner = DecisionTreeClassifier::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = DecisionTreeClassifier::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = DecisionTreeClassifier::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn criterion(&self) -> String {
        self.inner.criterion.clone()
    }

    #[getter]
    pub fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth
    }

    #[getter]
    pub fn min_samples_split(&self) -> usize {
        self.inner.min_samples_split
    }

    #[getter]
    pub fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }

    #[getter]
    pub fn classes_(&self) -> Vec<f64> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "DecisionTreeRegressor")]
#[derive(Clone)]
pub struct PyDecisionTreeRegressor {
    inner: DecisionTreeRegressor,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None, device = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = DecisionTreeRegressor::new(criterion, max_depth, min_samples_split, min_samples_leaf, device)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner.fit(&mat, &y)?;
        Ok(())
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        self.inner.fit(&mat, &target)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_buffer(&self, py: Python<'_>, x: pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.score(&mat, &y)?)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        Ok(self.inner.score(&mat, &target)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes())
            .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e)))
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e)))?;
        let inner = DecisionTreeRegressor::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = DecisionTreeRegressor::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = DecisionTreeRegressor::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn criterion(&self) -> String {
        self.inner.criterion.clone()
    }

    #[getter]
    pub fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth
    }

    #[getter]
    pub fn min_samples_split(&self) -> usize {
        self.inner.min_samples_split
    }

    #[getter]
    pub fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pymodule]
fn _ochreml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyDecisionTreeClassifier>()?;
    m.add_class::<PyDecisionTreeRegressor>()?;
    Ok(())
}
