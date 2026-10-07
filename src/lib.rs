// PyO3 bindings module for OchreML (＾▽＾)
// Exposes Rust Linear Regression and Decision Tree implementations to Python.

pub mod decision_tree;
pub mod linear_regression;
pub mod matrix;

use decision_tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use linear_regression::LinearRegression;
use matrix::Matrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(from_py_object, name = "LinearRegression")]
#[derive(Clone)]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept = true))]
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            inner: LinearRegression::new(fit_intercept),
        }
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
}

#[pyclass(from_py_object, name = "DecisionTreeClassifier")]
#[derive(Clone)]
pub struct PyDecisionTreeClassifier {
    inner: DecisionTreeClassifier,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
    ) -> Self {
        Self {
            inner: DecisionTreeClassifier::new(criterion, max_depth, min_samples_split, min_samples_leaf),
        }
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
}

#[pyclass(from_py_object, name = "DecisionTreeRegressor")]
#[derive(Clone)]
pub struct PyDecisionTreeRegressor {
    inner: DecisionTreeRegressor,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
    ) -> Self {
        Self {
            inner: DecisionTreeRegressor::new(criterion, max_depth, min_samples_split, min_samples_leaf),
        }
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
}

#[pymodule]
fn _ochreml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyDecisionTreeClassifier>()?;
    m.add_class::<PyDecisionTreeRegressor>()?;
    Ok(())
}
