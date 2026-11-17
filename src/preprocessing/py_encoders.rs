// OchreML PyO3 Bindings for Categorical & Target Encoders (*≧ω≦*)

use crate::preprocessing::encoders::{
    CatValue, LabelBinarizer, LabelEncoder, MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder,
    TargetEncoder,
};
use crate::preprocessing::py_helpers::{
    PyCsrTuple, buffer_to_matrix, catvalue_to_py, csr_parts_to_py, f64_matrix_to_cols,
    i64_slice_to_pybytes, matrix_to_pybytes, opt_catvalue_to_py, py_rows_to_cols,
    pyany_to_catvalue,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

fn parse_explicit_cats(
    explicit_cats: Option<Vec<Vec<Bound<'_, PyAny>>>>,
) -> PyResult<Option<Vec<Vec<CatValue>>>> {
    match explicit_cats {
        Some(vecs) => {
            let mut out = Vec::with_capacity(vecs.len());
            for v in vecs {
                let mut cv = Vec::with_capacity(v.len());
                for item in &v {
                    cv.push(pyany_to_catvalue(item)?);
                }
                out.push(cv);
            }
            Ok(Some(out))
        }
        None => Ok(None),
    }
}

fn categories_to_py<'py>(
    py: Python<'py>,
    cats: &Option<Vec<Vec<CatValue>>>,
) -> PyResult<Option<Vec<Vec<Bound<'py, PyAny>>>>> {
    match cats {
        Some(all) => {
            let mut out = Vec::with_capacity(all.len());
            for col in all {
                let mut pycol = Vec::with_capacity(col.len());
                for item in col {
                    pycol.push(catvalue_to_py(py, item)?);
                }
                out.push(pycol);
            }
            Ok(Some(out))
        }
        None => Ok(None),
    }
}

#[pyclass(from_py_object, name = "OneHotEncoder")]
#[derive(Clone)]
pub struct PyOneHotEncoder {
    inner: OneHotEncoder,
}

crate::pymethods_with_dit!(PyOneHotEncoder, OneHotEncoder, {
    #[new]
    #[pyo3(signature = (handle_unknown = "error".to_string(), drop_mode = None, min_frequency = None, max_categories = None, device = None))]
    pub fn new(
        handle_unknown: String,
        drop_mode: Option<String>,
        min_frequency: Option<f64>,
        max_categories: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OneHotEncoder::new(
                handle_unknown,
                drop_mode,
                min_frequency,
                max_categories,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, explicit_categories = None, explicit_drop = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let cols = f64_matrix_to_cols(&mat)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        self.inner.fit(&cols, ec.as_deref(), ed.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (rows, explicit_categories = None, explicit_drop = None))]
    pub fn fit_objects(
        &mut self,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let cols = py_rows_to_cols(rows)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        self.inner.fit(&cols, ec.as_deref(), ed.as_deref())?;
        Ok(())
    }

    pub fn transform_buffer_csr<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let mat = buffer_to_matrix(py, &x)?;
        let cols = f64_matrix_to_cols(&mat)?;
        let (indices, indptr, n_rows, n_cols) = self.inner.transform_csr(&cols)?;
        let data = vec![1.0; indices.len()];
        csr_parts_to_py(py, (data, indices, indptr, n_rows, n_cols))
    }

    pub fn transform_objects_csr<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let cols = py_rows_to_cols(rows)?;
        let (indices, indptr, n_rows, n_cols) = self.inner.transform_csr(&cols)?;
        let data = vec![1.0; indices.len()];
        csr_parts_to_py(py, (data, indices, indptr, n_rows, n_cols))
    }

    pub fn transform_buffer_dense<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_matrix(py, &x)?;
        let cols = f64_matrix_to_cols(&mat)?;
        let out = self.inner.transform_dense(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_objects_dense<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_rows_to_cols(rows)?;
        let out = self.inner.transform_dense(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let mat = buffer_to_matrix(py, &x)?;
        let res = self.inner.inverse_transform(&mat)?;
        res.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|opt| opt_catvalue_to_py(py, &opt))
                    .collect()
            })
            .collect()
    }

    #[getter]
    pub fn categories_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<Vec<Bound<'py, PyAny>>>>> {
        categories_to_py(py, &self.inner.categories_)
    }
    #[getter]
    pub fn drop_idx_(&self) -> Option<Vec<Option<usize>>> {
        self.inner.drop_idx_.clone()
    }
    #[getter]
    pub fn infrequent_indices_(&self) -> Option<Vec<Vec<usize>>> {
        self.inner.infrequent_indices_.clone()
    }
    #[getter]
    pub fn n_out_per_feat_(&self) -> Vec<usize> {
        self.inner.n_out_per_feat_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "OrdinalEncoder")]
#[derive(Clone)]
pub struct PyOrdinalEncoder {
    inner: OrdinalEncoder,
}

crate::pymethods_with_dit!(PyOrdinalEncoder, OrdinalEncoder, {
    #[new]
    #[pyo3(signature = (handle_unknown = "error".to_string(), unknown_value = None, encoded_missing_value = f64::NAN, min_frequency = None, max_categories = None, device = None))]
    pub fn new(
        handle_unknown: String,
        unknown_value: Option<f64>,
        encoded_missing_value: f64,
        min_frequency: Option<f64>,
        max_categories: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OrdinalEncoder::new(
                handle_unknown,
                unknown_value,
                encoded_missing_value,
                min_frequency,
                max_categories,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, explicit_categories = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let cols = f64_matrix_to_cols(&mat)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        self.inner.fit(&cols, ec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (rows, explicit_categories = None))]
    pub fn fit_objects(
        &mut self,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let cols = py_rows_to_cols(rows)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        self.inner.fit(&cols, ec.as_deref())?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_matrix(py, &x)?;
        let cols = f64_matrix_to_cols(&mat)?;
        let out = self.inner.transform(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_rows_to_cols(rows)?;
        let out = self.inner.transform(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let mat = buffer_to_matrix(py, &x)?;
        let res = self.inner.inverse_transform(&mat)?;
        res.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|opt| opt_catvalue_to_py(py, &opt))
                    .collect()
            })
            .collect()
    }

    #[getter]
    pub fn categories_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<Vec<Bound<'py, PyAny>>>>> {
        categories_to_py(py, &self.inner.categories_)
    }
    #[getter]
    pub fn infrequent_indices_(&self) -> Option<Vec<Vec<usize>>> {
        self.inner.infrequent_indices_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "LabelEncoder")]
#[derive(Clone)]
pub struct PyLabelEncoder {
    inner: LabelEncoder,
}

crate::pymethods_with_dit!(PyLabelEncoder, LabelEncoder, {
    #[new]
    #[pyo3(signature = (device = None))]
    pub fn new(device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: LabelEncoder::new(device)?,
        })
    }

    pub fn fit_objects(&mut self, y: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let vals: Vec<CatValue> = y
            .iter()
            .map(pyany_to_catvalue)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.fit(&vals)?;
        Ok(())
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        y: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let vals: Vec<CatValue> = y
            .iter()
            .map(pyany_to_catvalue)
            .collect::<PyResult<Vec<_>>>()?;
        let out: Vec<i64> = self
            .inner
            .transform(&vals)?
            .into_iter()
            .map(|i| i as i64)
            .collect();
        i64_slice_to_pybytes(py, &out)
    }

    pub fn inverse_transform_indices<'py>(
        &self,
        py: Python<'py>,
        y: Vec<usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let vals = self.inner.inverse_transform(&y)?;
        vals.iter().map(|v| catvalue_to_py(py, v)).collect()
    }

    #[getter]
    pub fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        match &self.inner.classes_ {
            Some(cls) => cls
                .iter()
                .map(|v| catvalue_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map(Some),
            None => Ok(None),
        }
    }
});

#[pyclass(from_py_object, name = "LabelBinarizer")]
#[derive(Clone)]
pub struct PyLabelBinarizer {
    inner: LabelBinarizer,
}

crate::pymethods_with_dit!(PyLabelBinarizer, LabelBinarizer, {
    #[new]
    #[pyo3(signature = (neg_label = 0, pos_label = 1, device = None))]
    pub fn new(neg_label: i64, pos_label: i64, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: LabelBinarizer::new(neg_label, pos_label, device)?,
        })
    }

    pub fn fit_objects(&mut self, y: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let vals: Vec<CatValue> = y
            .iter()
            .map(pyany_to_catvalue)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.fit(&vals)?;
        Ok(())
    }

    pub fn fit_classes(&mut self, classes: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let vals: Vec<CatValue> = classes
            .iter()
            .map(pyany_to_catvalue)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.fit_with_classes(vals)?;
        Ok(())
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        y: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let vals: Vec<CatValue> = y
            .iter()
            .map(pyany_to_catvalue)
            .collect::<PyResult<Vec<_>>>()?;
        let (rows, cols, out) = self.inner.transform(&vals)?;
        let bytes = i64_slice_to_pybytes(py, &out)?;
        Ok((rows, cols, bytes))
    }

    #[pyo3(signature = (y, threshold = None))]
    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        y: PyBuffer<f64>,
        threshold: Option<f64>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mat = buffer_to_matrix(py, &y)?;
        let res = self.inner.inverse_transform(&mat, threshold)?;
        res.iter().map(|v| catvalue_to_py(py, v)).collect()
    }

    #[getter]
    pub fn neg_label(&self) -> i64 {
        self.inner.neg_label
    }
    #[getter]
    pub fn pos_label(&self) -> i64 {
        self.inner.pos_label
    }
    #[getter]
    pub fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        match &self.inner.classes_ {
            Some(cls) => cls
                .iter()
                .map(|v| catvalue_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map(Some),
            None => Ok(None),
        }
    }
});

#[pyclass(from_py_object, name = "MultiLabelBinarizer")]
#[derive(Clone)]
pub struct PyMultiLabelBinarizer {
    inner: MultiLabelBinarizer,
}

crate::pymethods_with_dit!(PyMultiLabelBinarizer, MultiLabelBinarizer, {
    #[new]
    #[pyo3(signature = (device = None))]
    pub fn new(device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: MultiLabelBinarizer::new(device)?,
        })
    }

    #[pyo3(signature = (y, explicit_classes = None))]
    pub fn fit_objects(
        &mut self,
        y: Vec<Vec<Bound<'_, PyAny>>>,
        explicit_classes: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<()> {
        let rows: Vec<Vec<CatValue>> = y
            .iter()
            .map(|r| {
                r.iter()
                    .map(pyany_to_catvalue)
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let ec = explicit_classes
            .map(|c| {
                c.iter()
                    .map(pyany_to_catvalue)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        self.inner.fit(&rows, ec)?;
        Ok(())
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        y: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let rows: Vec<Vec<CatValue>> = y
            .iter()
            .map(|r| {
                r.iter()
                    .map(pyany_to_catvalue)
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (n_rows, n_cols, out) = self.inner.transform(&rows)?;
        let bytes = i64_slice_to_pybytes(py, &out)?;
        Ok((n_rows, n_cols, bytes))
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        yt: PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let mat = buffer_to_matrix(py, &yt)?;
        let res = self.inner.inverse_transform(&mat)?;
        res.iter()
            .map(|row| row.iter().map(|v| catvalue_to_py(py, v)).collect())
            .collect()
    }

    #[getter]
    pub fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        match &self.inner.classes_ {
            Some(cls) => cls
                .iter()
                .map(|v| catvalue_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map(Some),
            None => Ok(None),
        }
    }
});

#[pyclass(from_py_object, name = "TargetEncoder")]
#[derive(Clone)]
pub struct PyTargetEncoder {
    inner: TargetEncoder,
}

crate::pymethods_with_dit!(PyTargetEncoder, TargetEncoder, {
    #[new]
    #[pyo3(signature = (smooth = None, cv = 5, shuffle = true, random_state = None, target_type = "auto".to_string(), device = None))]
    pub fn new(
        smooth: Option<f64>,
        cv: usize,
        shuffle: bool,
        random_state: Option<u64>,
        target_type: String,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TargetEncoder::new(smooth, cv, shuffle, random_state, target_type, device)?,
        })
    }

    #[pyo3(signature = (rows, y_cols, classes, inferred_target_type, explicit_categories = None))]
    pub fn fit_objects(
        &mut self,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let cols = py_rows_to_cols(rows)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        let cls = classes
            .map(|c| {
                c.iter()
                    .map(pyany_to_catvalue)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        self.inner.fit_from_precomputed(
            &cols,
            ec.as_deref(),
            &y_cols,
            cls,
            inferred_target_type,
        )?;
        Ok(())
    }

    #[pyo3(signature = (rows, y_cols, classes, inferred_target_type, fold_ids, explicit_categories = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_objects<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        fold_ids: Vec<usize>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_rows_to_cols(rows)?;
        let ec = parse_explicit_cats(explicit_categories)?;
        let cls = classes
            .map(|c| {
                c.iter()
                    .map(pyany_to_catvalue)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let out = self.inner.fit_transform_with_folds(
            &cols,
            ec.as_deref(),
            &y_cols,
            cls,
            inferred_target_type,
            &fold_ids,
        )?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_rows_to_cols(rows)?;
        let out = self.inner.transform(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn categories_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<Vec<Bound<'py, PyAny>>>>> {
        categories_to_py(py, &self.inner.categories_)
    }
    #[getter]
    pub fn encodings_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.encodings_.clone()
    }
    #[getter]
    pub fn target_mean_(&self) -> Option<Vec<f64>> {
        self.inner.target_mean_.clone()
    }
    #[getter]
    pub fn target_type_(&self) -> Option<String> {
        self.inner.target_type_.clone()
    }
    #[getter]
    pub fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        match &self.inner.classes_ {
            Some(cls) => cls
                .iter()
                .map(|v| catvalue_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map(Some),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});
