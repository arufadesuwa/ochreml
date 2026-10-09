// PyO3 bindings module for OchreML (＾▽＾)
// Exposes Rust Linear Regression and Decision Tree implementations with Hardware Acceleration to Python.

pub mod decision_tree;
pub mod device;
pub mod linear_regression;
pub mod matrix;

use decision_tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use linear_regression::LinearRegression;
use matrix::Matrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn get_available_devices() -> Vec<String> {
    device::get_available_devices_list()
}

#[pyfunction]
pub fn get_device_info() -> HashMap<String, String> {
    device::get_device_info_map()
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
        let inner = LinearRegression::new(fit_intercept, device).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.fit(&mat, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.predict(&mat).map_err(PyValueError::new_err)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.score(&mat, &y).map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
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
        let inner = DecisionTreeClassifier::new(criterion, max_depth, min_samples_split, min_samples_leaf, device)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.fit(&mat, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.predict(&mat).map_err(PyValueError::new_err)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.score(&mat, &y).map_err(PyValueError::new_err)
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
        let inner = DecisionTreeRegressor::new(criterion, max_depth, min_samples_split, min_samples_leaf, device)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.fit(&mat, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.predict(&mat).map_err(PyValueError::new_err)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x).map_err(PyValueError::new_err)?;
        self.inner.score(&mat, &y).map_err(PyValueError::new_err)
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
