// OchreML PyO3 Bindings for Scalers, Transforms & Feature Engineering (＾▽＾)

use crate::preprocessing::features::{KernelCenterer, PolynomialFeatures, SplineTransformer};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_matrix, buffer_to_pooled_matrix, matrix_to_pybytes,
    write_slice_to_pybuffer,
};
use crate::preprocessing::scalers::{
    MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler, normalize_matrix,
    normalize_matrix_inplace_opt,
};
use crate::preprocessing::transforms::{
    Binarizer, KBinsDiscretizer, PowerTransformer, QuantileTransformer, binarize_matrix,
    binarize_matrix_inplace,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(from_py_object, name = "StandardScaler")]
#[derive(Clone)]
pub struct PyStandardScaler {
    inner: StandardScaler,
}

crate::pymethods_with_dit!(PyStandardScaler, StandardScaler, {
    #[new]
    #[pyo3(signature = (with_mean = true, with_std = true, device = None))]
    pub fn new(with_mean: bool, with_std: bool, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: StandardScaler::new(with_mean, with_std, device)?,
        })
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.fit(&mat, sw.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, sample_weight = None))]
    pub fn partial_fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.partial_fit(&mat, sw.as_deref())?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    #[pyo3(signature = (x, out, sample_weight = None))]
    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        let sw = match sample_weight {
            Some(ref b) => Some(buffer_to_f64_1d(py, b)?),
            None => None,
        };
        self.inner.fit(&mat, sw.as_deref())?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn with_mean(&self) -> bool {
        self.inner.with_mean
    }
    #[getter]
    pub fn with_std(&self) -> bool {
        self.inner.with_std
    }
    #[getter]
    pub fn mean_(&self) -> Option<Vec<f64>> {
        self.inner.mean_.clone()
    }
    #[getter]
    pub fn var_(&self) -> Option<Vec<f64>> {
        self.inner.var_.clone()
    }
    #[getter]
    pub fn scale_(&self) -> Option<Vec<f64>> {
        self.inner.scale_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_samples_seen_(&self) -> usize {
        self.inner.n_samples_seen_
    }
});

#[pyclass(from_py_object, name = "MinMaxScaler")]
#[derive(Clone)]
pub struct PyMinMaxScaler {
    inner: MinMaxScaler,
}

crate::pymethods_with_dit!(PyMinMaxScaler, MinMaxScaler, {
    #[new]
    #[pyo3(signature = (feature_range = (0.0, 1.0), clip = false, device = None))]
    pub fn new(feature_range: (f64, f64), clip: bool, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: MinMaxScaler::new(feature_range, clip, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn partial_fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.partial_fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn feature_range(&self) -> (f64, f64) {
        self.inner.feature_range
    }
    #[getter]
    pub fn clip(&self) -> bool {
        self.inner.clip
    }
    #[getter]
    pub fn min_(&self) -> Option<Vec<f64>> {
        self.inner.min_.clone()
    }
    #[getter]
    pub fn scale_(&self) -> Option<Vec<f64>> {
        self.inner.scale_.clone()
    }
    #[getter]
    pub fn data_min_(&self) -> Option<Vec<f64>> {
        self.inner.data_min_.clone()
    }
    #[getter]
    pub fn data_max_(&self) -> Option<Vec<f64>> {
        self.inner.data_max_.clone()
    }
    #[getter]
    pub fn data_range_(&self) -> Option<Vec<f64>> {
        self.inner.data_range_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_samples_seen_(&self) -> usize {
        self.inner.n_samples_seen_
    }
});

#[pyclass(from_py_object, name = "MaxAbsScaler")]
#[derive(Clone)]
pub struct PyMaxAbsScaler {
    inner: MaxAbsScaler,
}

crate::pymethods_with_dit!(PyMaxAbsScaler, MaxAbsScaler, {
    #[new]
    #[pyo3(signature = (device = None))]
    pub fn new(device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: MaxAbsScaler::new(device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn partial_fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.partial_fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn max_abs_(&self) -> Option<Vec<f64>> {
        self.inner.max_abs_.clone()
    }
    #[getter]
    pub fn scale_(&self) -> Option<Vec<f64>> {
        self.inner.scale_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_samples_seen_(&self) -> usize {
        self.inner.n_samples_seen_
    }
});

#[pyclass(from_py_object, name = "RobustScaler")]
#[derive(Clone)]
pub struct PyRobustScaler {
    inner: RobustScaler,
}

crate::pymethods_with_dit!(PyRobustScaler, RobustScaler, {
    #[new]
    #[pyo3(signature = (with_centering = true, with_scaling = true, quantile_range = (25.0, 75.0), unit_variance = false, device = None))]
    pub fn new(
        with_centering: bool,
        with_scaling: bool,
        quantile_range: (f64, f64),
        unit_variance: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RobustScaler::new(
                with_centering,
                with_scaling,
                quantile_range,
                unit_variance,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn with_centering(&self) -> bool {
        self.inner.with_centering
    }
    #[getter]
    pub fn with_scaling(&self) -> bool {
        self.inner.with_scaling
    }
    #[getter]
    pub fn quantile_range(&self) -> (f64, f64) {
        self.inner.quantile_range
    }
    #[getter]
    pub fn unit_variance(&self) -> bool {
        self.inner.unit_variance
    }
    #[getter]
    pub fn center_(&self) -> Option<Vec<f64>> {
        self.inner.center_.clone()
    }
    #[getter]
    pub fn scale_(&self) -> Option<Vec<f64>> {
        self.inner.scale_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "Normalizer")]
#[derive(Clone)]
pub struct PyNormalizer {
    inner: Normalizer,
}

crate::pymethods_with_dit!(PyNormalizer, Normalizer, {
    #[new]
    #[pyo3(signature = (norm = "l2".to_string(), device = None))]
    pub fn new(norm: String, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Normalizer::new(norm, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    #[getter]
    pub fn norm(&self) -> String {
        self.inner.norm.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "Binarizer")]
#[derive(Clone)]
pub struct PyBinarizer {
    inner: Binarizer,
}

crate::pymethods_with_dit!(PyBinarizer, Binarizer, {
    #[new]
    #[pyo3(signature = (threshold = 0.0, device = None))]
    pub fn new(threshold: f64, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Binarizer::new(threshold, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    #[getter]
    pub fn threshold(&self) -> f64 {
        self.inner.threshold
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "KernelCenterer")]
#[derive(Clone)]
pub struct PyKernelCenterer {
    inner: KernelCenterer,
}

crate::pymethods_with_dit!(PyKernelCenterer, KernelCenterer, {
    #[new]
    #[pyo3(signature = (device = None))]
    pub fn new(device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: KernelCenterer::new(device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, k: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &k)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        k: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &k)?;
        self.inner.transform_inplace(&mut mat)?;
        matrix_to_pybytes(py, &mat)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        k: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &k)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        k: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut mat = buffer_to_pooled_matrix(py, &k)?;
        self.inner.fit(&mat)?;
        self.inner.transform_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    #[getter]
    pub fn k_fit_rows_(&self) -> Option<Vec<f64>> {
        self.inner.k_fit_rows_.clone()
    }
    #[getter]
    pub fn k_fit_all_(&self) -> Option<f64> {
        self.inner.k_fit_all_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "PolynomialFeatures")]
#[derive(Clone)]
pub struct PyPolynomialFeatures {
    inner: PolynomialFeatures,
}

crate::pymethods_with_dit!(PyPolynomialFeatures, PolynomialFeatures, {
    #[new]
    #[pyo3(signature = (min_degree = 0, max_degree = 2, interaction_only = false, include_bias = true, order = "C".to_string(), device = None))]
    pub fn new(
        min_degree: usize,
        max_degree: usize,
        interaction_only: bool,
        include_bias: bool,
        order: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PolynomialFeatures::new(
                min_degree,
                max_degree,
                interaction_only,
                include_bias,
                order,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn fit_dims(&mut self, n_features: usize) -> PyResult<()> {
        self.inner.fit_dims(n_features)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let n_out = self
            .inner
            .n_output_features_
            .ok_or(crate::error::OchreError::NotFitted)?;
        let mut buf = vec![0.0; mat.rows * n_out];
        self.inner.transform_to_slice(&mat, &mut buf)?;
        write_slice_to_pybuffer(py, &buf, &out)
    }

    #[getter]
    pub fn min_degree(&self) -> usize {
        self.inner.min_degree
    }
    #[getter]
    pub fn max_degree(&self) -> usize {
        self.inner.max_degree
    }
    #[getter]
    pub fn interaction_only(&self) -> bool {
        self.inner.interaction_only
    }
    #[getter]
    pub fn include_bias(&self) -> bool {
        self.inner.include_bias
    }
    #[getter]
    pub fn order(&self) -> String {
        self.inner.order.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_output_features_(&self) -> Option<usize> {
        self.inner.n_output_features_
    }
    #[getter]
    pub fn powers_(&self) -> Option<Vec<usize>> {
        self.inner.powers_.clone()
    }
});

#[pyclass(from_py_object, name = "SplineTransformer")]
#[derive(Clone)]
pub struct PySplineTransformer {
    inner: SplineTransformer,
}

crate::pymethods_with_dit!(PySplineTransformer, SplineTransformer, {
    #[new]
    #[pyo3(signature = (n_knots = 5, degree = 3, knots_mode = "uniform".to_string(), extrapolation = "constant".to_string(), include_bias = true, order = "C".to_string(), device = None))]
    pub fn new(
        n_knots: usize,
        degree: usize,
        knots_mode: String,
        extrapolation: String,
        include_bias: bool,
        order: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SplineTransformer::new(
                n_knots,
                degree,
                knots_mode,
                extrapolation,
                include_bias,
                order,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, custom_knots = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        custom_knots: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let ck = match custom_knots {
            Some(ref b) => Some(buffer_to_pooled_matrix(py, b)?),
            None => None,
        };
        self.inner.fit(&mat, ck.as_deref())?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let res = self.inner.transform(&mat)?;
        write_slice_to_pybuffer(py, &res.data, &out)
    }

    #[getter]
    pub fn n_knots(&self) -> usize {
        self.inner.n_knots
    }
    #[getter]
    pub fn degree(&self) -> usize {
        self.inner.degree
    }
    #[getter]
    pub fn knots_mode(&self) -> String {
        self.inner.knots_mode.clone()
    }
    #[getter]
    pub fn extrapolation(&self) -> String {
        self.inner.extrapolation.clone()
    }
    #[getter]
    pub fn include_bias(&self) -> bool {
        self.inner.include_bias
    }
    #[getter]
    pub fn order(&self) -> String {
        self.inner.order.clone()
    }
    #[getter]
    pub fn knots_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.knots_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn n_features_out_(&self) -> Option<usize> {
        self.inner.n_features_out_
    }
});

#[pyclass(from_py_object, name = "PowerTransformer")]
#[derive(Clone)]
pub struct PyPowerTransformer {
    inner: PowerTransformer,
}

crate::pymethods_with_dit!(PyPowerTransformer, PowerTransformer, {
    #[new]
    #[pyo3(signature = (method = "yeo-johnson".to_string(), standardize = true, device = None))]
    pub fn new(method: String, standardize: bool, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: PowerTransformer::new(method, standardize, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let res = self.inner.transform(&mat)?;
        write_slice_to_pybuffer(py, &res.data, &out)
    }

    pub fn fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let res = self.inner.fit_transform(&mat)?;
        write_slice_to_pybuffer(py, &res.data, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn method(&self) -> String {
        self.inner.method.clone()
    }
    #[getter]
    pub fn standardize(&self) -> bool {
        self.inner.standardize
    }
    #[getter]
    pub fn lambdas_(&self) -> Option<Vec<f64>> {
        self.inner.lambdas_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "QuantileTransformer")]
#[derive(Clone)]
pub struct PyQuantileTransformer {
    inner: QuantileTransformer,
}

crate::pymethods_with_dit!(PyQuantileTransformer, QuantileTransformer, {
    #[new]
    #[pyo3(signature = (n_quantiles = 1000, output_distribution = "uniform".to_string(), ignore_implicit_zeros = false, subsample = 10000, random_state = None, device = None))]
    pub fn new(
        n_quantiles: usize,
        output_distribution: String,
        ignore_implicit_zeros: bool,
        subsample: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: QuantileTransformer::new(
                n_quantiles,
                output_distribution,
                ignore_implicit_zeros,
                subsample,
                random_state,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_quantiles(&self) -> usize {
        self.inner.n_quantiles
    }
    #[getter]
    pub fn output_distribution(&self) -> String {
        self.inner.output_distribution.clone()
    }
    #[getter]
    pub fn n_quantiles_(&self) -> Option<usize> {
        self.inner.n_quantiles_
    }
    #[getter]
    pub fn references_(&self) -> Option<Vec<f64>> {
        self.inner.references_.clone()
    }
    #[getter]
    pub fn quantiles_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.quantiles_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "KBinsDiscretizer")]
#[derive(Clone)]
pub struct PyKBinsDiscretizer {
    inner: KBinsDiscretizer,
}

crate::pymethods_with_dit!(PyKBinsDiscretizer, KBinsDiscretizer, {
    #[new]
    #[pyo3(signature = (n_bins_param, encode = "onehot".to_string(), strategy = "quantile".to_string(), subsample = Some(200_000), device = None))]
    pub fn new(
        n_bins_param: Vec<usize>,
        encode: String,
        strategy: String,
        subsample: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KBinsDiscretizer::new(n_bins_param, encode, strategy, subsample, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_matrix(py, &x)?;
        let out = self.inner.transform_dense(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_matrix(py, &x)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn encode(&self) -> String {
        self.inner.encode.clone()
    }
    #[getter]
    pub fn strategy(&self) -> String {
        self.inner.strategy.clone()
    }
    #[getter]
    pub fn bin_edges_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.bin_edges_.clone()
    }
    #[getter]
    pub fn n_bins_(&self) -> Option<Vec<usize>> {
        self.inner.n_bins_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyfunction]
#[pyo3(signature = (x, norm = "l2", axis = 1))]
pub fn _rust_normalize_buffer<'py>(
    py: Python<'py>,
    x: PyBuffer<f64>,
    norm: &str,
    axis: usize,
) -> PyResult<(usize, usize, Bound<'py, PyBytes>, Vec<f64>)> {
    let mat = buffer_to_pooled_matrix(py, &x)?;
    let (out, norms) = normalize_matrix(&mat, norm, axis)?;
    let (rows, cols, bytes) = matrix_to_pybytes(py, &out)?;
    Ok((rows, cols, bytes, norms))
}

#[pyfunction]
#[pyo3(signature = (x, out, norm = "l2", axis = 1, return_norms = true))]
pub fn _rust_normalize_into(
    py: Python<'_>,
    x: PyBuffer<f64>,
    out: PyBuffer<f64>,
    norm: &str,
    axis: usize,
    return_norms: bool,
) -> PyResult<Vec<f64>> {
    let mut mat = buffer_to_pooled_matrix(py, &x)?;
    let norms = normalize_matrix_inplace_opt(&mut mat, norm, axis, return_norms)?;
    write_slice_to_pybuffer(py, &mat.data, &out)?;
    Ok(norms)
}

#[pyfunction]
#[pyo3(signature = (x, threshold = 0.0))]
pub fn _rust_binarize_buffer<'py>(
    py: Python<'py>,
    x: PyBuffer<f64>,
    threshold: f64,
) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
    let mat = buffer_to_pooled_matrix(py, &x)?;
    let out = binarize_matrix(&mat, threshold)?;
    matrix_to_pybytes(py, &out)
}

#[pyfunction]
#[pyo3(signature = (x, out, threshold = 0.0))]
pub fn _rust_binarize_into(
    py: Python<'_>,
    x: PyBuffer<f64>,
    out: PyBuffer<f64>,
    threshold: f64,
) -> PyResult<()> {
    let mut mat = buffer_to_pooled_matrix(py, &x)?;
    binarize_matrix_inplace(&mut mat, threshold)?;
    write_slice_to_pybuffer(py, &mat.data, &out)
}
