// OchreML PyO3 Zero-Copy Buffer & PyBytes Conversion Helpers (*≧ω≦*)
// Shared across preprocessing, imputation, encoding, and text feature extraction bindings.

use crate::dit_io::validate_dit_path;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::encoders::CatValue;
use crate::preprocessing::text_fe::CsrParts;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyFloat, PyInt, PyNone, PyString};
use std::cell::RefCell;
use std::fs;
use std::ops::{Deref, DerefMut};

const MAX_POOLED_VEC_CAP: usize = 48 * 1024 * 1024;

thread_local! {
    static PREPROC_BUF_POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

fn pop_pooled_vec(min_cap: usize) -> Vec<f64> {
    PREPROC_BUF_POOL
        .with(|pool| {
            let mut slots = pool.borrow_mut();
            if let Some(pos) = slots.iter().position(|v| v.capacity() >= min_cap) {
                Some(slots.swap_remove(pos))
            } else {
                slots.pop()
            }
        })
        .unwrap_or_default()
}

fn pop_pooled_vec_for_copy(min_len: usize) -> Option<Vec<f64>> {
    PREPROC_BUF_POOL.with(|pool| {
        let mut slots = pool.borrow_mut();
        let pos = slots
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v.len() >= min_len && (v.len() <= min_len.saturating_mul(2) || v.len() < 1_048_576)
            })
            .min_by_key(|(_, v)| v.capacity())
            .map(|(idx, _)| idx);
        pos.map(|idx| slots.swap_remove(idx))
    })
}

pub struct PooledMatrix {
    pub mat: Matrix,
}

impl PooledMatrix {
    pub fn with_len(rows: usize, cols: usize) -> PyResult<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| PyValueError::new_err("Matrix dimensions overflow ( >_< )"))?;
        let mut data = pop_pooled_vec(len);
        data.resize(len, 0.0);
        Ok(Self {
            mat: Matrix::from_vec(rows, cols, data)?,
        })
    }
}

impl Deref for PooledMatrix {
    type Target = Matrix;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl DerefMut for PooledMatrix {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mat
    }
}

impl Drop for PooledMatrix {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.mat.data);
        if buf.capacity() <= MAX_POOLED_VEC_CAP {
            PREPROC_BUF_POOL.with(|pool| {
                if let Ok(mut slots) = pool.try_borrow_mut() {
                    if slots.len() < 3 {
                        slots.push(buf);
                    } else if let Some(min_slot) = slots.iter_mut().min_by_key(|v| v.capacity())
                        && buf.capacity() > min_slot.capacity()
                    {
                        *min_slot = buf;
                    }
                }
            });
        }
    }
}

pub fn buffer_to_pooled_matrix(py: Python<'_>, x: &PyBuffer<f64>) -> PyResult<PooledMatrix> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Feature matrix X must be a 2D array (n_samples, n_features), got {}D ( >_< )",
            shape.len()
        )));
    }
    let rows = shape[0];
    let cols = shape[1];
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("Matrix dimensions overflow ( >_< )"))?;

    if x.is_c_contiguous()
        && let Some(mut data) = pop_pooled_vec_for_copy(len)
    {
        data.truncate(len);
        x.copy_to_slice(py, &mut data[..len]).map_err(|e| {
            PyValueError::new_err(format!("Failed to copy buffer data: {} (´；ω；`)", e))
        })?;
        return Ok(PooledMatrix {
            mat: Matrix::from_vec(rows, cols, data)?,
        });
    }

    let data = x.to_vec(py).map_err(|e| {
        PyValueError::new_err(format!("Failed to copy buffer data: {} (´；ω；`)", e))
    })?;
    Ok(PooledMatrix {
        mat: Matrix::from_vec(rows, cols, data)?,
    })
}

pub fn buffer_to_matrix(py: Python<'_>, x: &PyBuffer<f64>) -> PyResult<Matrix> {
    let mut pooled = buffer_to_pooled_matrix(py, x)?;
    Ok(std::mem::replace(
        &mut pooled.mat,
        Matrix {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        },
    ))
}

pub fn write_slice_to_pybuffer<T: pyo3::buffer::Element + Copy>(
    py: Python<'_>,
    data: &[T],
    out: &PyBuffer<T>,
) -> PyResult<()> {
    out.copy_from_slice(py, data).map_err(|e| {
        PyValueError::new_err(format!("Failed to write output buffer: {} (´；ω；`)", e))
    })
}

pub fn buffer_to_typed_1d<T: pyo3::buffer::Element + Copy + Default>(
    py: Python<'_>,
    v: &PyBuffer<T>,
) -> PyResult<Vec<T>> {
    let shape = v.shape();
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "Expected 1D buffer, got {}D ( >_< )",
            shape.len()
        )));
    }
    v.to_vec(py)
        .map_err(|e| PyValueError::new_err(format!("Failed to copy 1D buffer: {} (´；ω；`)", e)))
}

pub fn buffer_to_f64_1d(py: Python<'_>, v: &PyBuffer<f64>) -> PyResult<Vec<f64>> {
    buffer_to_typed_1d(py, v)
}

pub fn buffer_to_u32_1d(py: Python<'_>, v: &PyBuffer<u32>) -> PyResult<Vec<u32>> {
    buffer_to_typed_1d(py, v)
}

pub fn buffer_to_usize_1d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(b64) = PyBuffer::<i64>::get(obj) {
        let tmp = b64.to_vec(py)?;
        return Ok(tmp.into_iter().map(|v| v as usize).collect());
    }
    if let Ok(b32) = PyBuffer::<i32>::get(obj) {
        let tmp = b32.to_vec(py)?;
        return Ok(tmp.into_iter().map(|v| v as usize).collect());
    }
    Err(PyValueError::new_err(
        "Expected 1D int32 or int64 buffer for CSR indices/indptr ( >_< )",
    ))
}

pub fn buffer_to_vec(py: Python<'_>, v: &PyBuffer<f64>) -> PyResult<Vec<f64>> {
    buffer_to_f64_1d(py, v)
}

pub fn matrix_to_pybytes<'py>(
    py: Python<'py>,
    mat: &Matrix,
) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
    Ok((mat.rows, mat.cols, f64_slice_to_pybytes(py, &mat.data)?))
}

pub fn f64_slice_to_pybytes<'py>(py: Python<'py>, data: &[f64]) -> PyResult<Bound<'py, PyBytes>> {
    let n_bytes = std::mem::size_of_val(data);
    PyBytes::new_with(py, n_bytes, |buf| {
        let (chunks, _) = buf.as_chunks_mut::<8>();
        for (dst, &val) in chunks.iter_mut().zip(data.iter()) {
            *dst = val.to_le_bytes();
        }
        Ok(())
    })
}

pub fn i64_slice_to_pybytes<'py>(py: Python<'py>, data: &[i64]) -> PyResult<Bound<'py, PyBytes>> {
    let n_bytes = std::mem::size_of_val(data);
    PyBytes::new_with(py, n_bytes, |buf| {
        let (chunks, _) = buf.as_chunks_mut::<8>();
        for (dst, &val) in chunks.iter_mut().zip(data.iter()) {
            *dst = val.to_le_bytes();
        }
        Ok(())
    })
}

pub fn usize_slice_to_i64_pybytes<'py>(
    py: Python<'py>,
    data: &[usize],
) -> PyResult<Bound<'py, PyBytes>> {
    let n_bytes = std::mem::size_of_val(data);
    PyBytes::new_with(py, n_bytes, |buf| {
        let (chunks, _) = buf.as_chunks_mut::<8>();
        for (dst, &val) in chunks.iter_mut().zip(data.iter()) {
            *dst = (val as i64).to_le_bytes();
        }
        Ok(())
    })
}

pub type PyCsrTuple<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    usize,
    usize,
);

pub fn csr_parts_to_py<'py>(py: Python<'py>, parts: CsrParts) -> PyResult<PyCsrTuple<'py>> {
    let (data, indices, indptr, n_rows, n_cols) = parts;
    let d_bytes = f64_slice_to_pybytes(py, &data)?;
    let idx_bytes = usize_slice_to_i64_pybytes(py, &indices)?;
    let ptr_bytes = usize_slice_to_i64_pybytes(py, &indptr)?;
    Ok((d_bytes, idx_bytes, ptr_bytes, n_rows, n_cols))
}

pub fn save_dit_bytes(path: &str, bytes: &[u8]) -> PyResult<()> {
    validate_dit_path(path)?;
    fs::write(path, bytes)
        .map_err(|e| PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e)))
}

pub fn load_dit_bytes(path: &str) -> PyResult<Vec<u8>> {
    validate_dit_path(path)?;
    fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e)))
}

pub fn pyany_to_catvalue(obj: &Bound<'_, PyAny>) -> PyResult<CatValue> {
    if obj.is_instance_of::<PyNone>() {
        return Ok(CatValue::Missing);
    }
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(CatValue::Int(i64::from(b)));
    }
    if obj.is_instance_of::<PyInt>()
        && let Ok(v) = obj.extract::<i64>()
    {
        return Ok(CatValue::Int(v));
    }
    if obj.is_instance_of::<PyFloat>() {
        let v: f64 = obj.extract()?;
        return Ok(CatValue::from_f64(v));
    }
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        return Ok(CatValue::Str(s));
    }
    // NumPy scalar types (`np.int64`, `np.float64`, `np.str_`, etc.)
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(CatValue::Int(v));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(CatValue::from_f64(v));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(CatValue::Str(s));
    }
    Err(PyValueError::new_err(
        "Unsupported category value type; expected str, int, float, or None ( >_< )",
    ))
}

pub fn catvalue_to_py<'py>(py: Python<'py>, val: &CatValue) -> PyResult<Bound<'py, PyAny>> {
    match val {
        CatValue::Missing => Ok(f64::NAN.into_pyobject(py)?.into_any()),
        CatValue::Int(i) => Ok((*i).into_pyobject(py)?.into_any()),
        CatValue::Float(bits) => Ok(f64::from_bits(*bits).into_pyobject(py)?.into_any()),
        CatValue::Str(s) => Ok(s.into_pyobject(py)?.into_any()),
    }
}

pub fn opt_catvalue_to_py<'py>(
    py: Python<'py>,
    val: &Option<CatValue>,
) -> PyResult<Bound<'py, PyAny>> {
    match val {
        Some(v) => catvalue_to_py(py, v),
        None => Ok(py.None().into_bound(py)),
    }
}

pub fn py_rows_to_cols(rows: Vec<Vec<Bound<'_, PyAny>>>) -> PyResult<Vec<Vec<CatValue>>> {
    if rows.is_empty() {
        return Err(OchreError::EmptyData.into());
    }
    let n_samples = rows.len();
    let n_features = rows[0].len();
    if n_features == 0 {
        return Err(OchreError::EmptyData.into());
    }
    let mut cols = vec![Vec::with_capacity(n_samples); n_features];
    for (r_idx, row) in rows.iter().enumerate() {
        if row.len() != n_features {
            return Err(PyValueError::new_err(format!(
                "Row {} has {} columns, expected {} ( >_< )",
                r_idx,
                row.len(),
                n_features
            )));
        }
        for (j, item) in row.iter().enumerate() {
            cols[j].push(pyany_to_catvalue(item)?);
        }
    }
    Ok(cols)
}

pub fn f64_matrix_to_cols(mat: &Matrix) -> OchreResult<Vec<Vec<CatValue>>> {
    if mat.rows == 0 || mat.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    let mut cols = vec![Vec::with_capacity(mat.rows); mat.cols];
    for r in 0..mat.rows {
        for j in 0..mat.cols {
            cols[j].push(CatValue::from_f64(mat.data[r * mat.cols + j]));
        }
    }
    Ok(cols)
}

#[macro_export]
macro_rules! pymethods_with_dit {
    ($py_ty:ident, $rust_ty:ty, { $($body:tt)* }) => {
        #[pymethods]
        impl $py_ty {
            $($body)*

            pub fn save(&self, path: &str) -> PyResult<()> {
                $crate::preprocessing::py_helpers::save_dit_bytes(path, &self.inner.to_bytes())
            }

            #[staticmethod]
            #[pyo3(signature = (path, device = None))]
            pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
                let bytes = $crate::preprocessing::py_helpers::load_dit_bytes(path)?;
                let inner = <$rust_ty>::from_bytes(&bytes, device.as_deref())?;
                Ok(Self { inner })
            }

            #[pyo3(signature = (device = None))]
            pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
                self.inner.set_device(device)?;
                Ok(())
            }

            pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
                pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
            }

            #[staticmethod]
            #[pyo3(signature = (bytes, device = None))]
            pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
                let inner = <$rust_ty>::from_bytes(bytes, device.as_deref())?;
                Ok(Self { inner })
            }

            pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
                pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
            }

            pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
                self.inner = <$rust_ty>::from_bytes(state, None)?;
                Ok(())
            }

            #[getter]
            pub fn device_(&self) -> Option<String> {
                self.inner.device_.clone()
            }
        }
    };
}
