// OchreML PyO3 Bindings for Imputers & Feature/Text Extractors (o´∀｀o)

use crate::preprocessing::impute::{KNNImputer, MissingIndicator, SimpleImputer};
use crate::preprocessing::py_helpers::{
    PyCsrTuple, buffer_to_f64_1d, buffer_to_pooled_matrix, buffer_to_usize_1d, catvalue_to_py,
    csr_parts_to_py, matrix_to_pybytes, py_rows_to_cols, pyany_to_catvalue,
    write_slice_to_pybuffer,
};
use crate::preprocessing::text_fe::{
    CountVectorizer, DictVectorizer, FeatureHasher, HashingVectorizer, TfidfTransformer,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyString};
use std::collections::{HashMap, HashSet};

fn extract_py_dict_pairs(
    rows: &Bound<'_, PyAny>,
    separator: &str,
) -> PyResult<Vec<Vec<(String, f64)>>> {
    let iter = rows.try_iter()?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let dict = item.cast::<PyDict>()?;
        let mut row = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key_str: String = if let Ok(s) = k.cast::<PyString>() {
                s.to_str()?.to_owned()
            } else {
                k.str()?.to_str()?.to_owned()
            };
            if let Ok(sv) = v.cast::<PyString>() {
                let mut combined =
                    String::with_capacity(key_str.len() + separator.len() + sv.len()?);
                combined.push_str(&key_str);
                combined.push_str(separator);
                combined.push_str(sv.to_str()?);
                row.push((combined, 1.0));
            } else if v.is_instance_of::<PyBool>() {
                let b: bool = v.extract()?;
                row.push((key_str, if b { 1.0 } else { 0.0 }));
            } else {
                let val: f64 = v.extract()?;
                if val != 0.0 {
                    row.push((key_str, val));
                }
            }
        }
        out.push(row);
    }
    Ok(out)
}

#[pyclass(from_py_object, name = "MissingIndicator")]
#[derive(Clone)]
pub struct PyMissingIndicator {
    inner: MissingIndicator,
}

crate::pymethods_with_dit!(PyMissingIndicator, MissingIndicator, {
    #[new]
    #[pyo3(signature = (missing_values = f64::NAN, features_mode = "missing-only".to_string(), error_on_new = true, device = None))]
    pub fn new(
        missing_values: f64,
        features_mode: String,
        error_on_new: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MissingIndicator::new(missing_values, features_mode, error_on_new, device)?,
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

    pub fn transform_u8_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform_u8(&mat)?;
        let n_out = self.inner.features_.as_ref().map_or(0, |f| f.len());
        let bytes = PyBytes::new(py, &out);
        Ok((mat.rows, n_out, bytes))
    }

    pub fn fit_transform_u8_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        let out = self.inner.transform_u8(&mat)?;
        let n_out = self.inner.features_.as_ref().map_or(0, |f| f.len());
        let bytes = PyBytes::new(py, &out);
        Ok((mat.rows, n_out, bytes))
    }

    #[getter]
    pub fn features_(&self) -> Option<Vec<usize>> {
        self.inner.features_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "SimpleImputer")]
#[derive(Clone)]
pub struct PySimpleImputer {
    inner: SimpleImputer,
}

crate::pymethods_with_dit!(PySimpleImputer, SimpleImputer, {
    #[new]
    #[pyo3(signature = (missing_values, strategy = "mean".to_string(), fill_value = None, add_indicator = false, keep_empty_features = false, device = None))]
    pub fn new(
        missing_values: Bound<'_, PyAny>,
        strategy: String,
        fill_value: Option<Bound<'_, PyAny>>,
        add_indicator: bool,
        keep_empty_features: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        let mv = pyany_to_catvalue(&missing_values)?;
        let fv = fill_value.as_ref().map(pyany_to_catvalue).transpose()?;
        Ok(Self {
            inner: SimpleImputer::new(
                mv,
                strategy,
                fv,
                add_indicator,
                keep_empty_features,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit_numeric(&mat)?;
        Ok(())
    }

    pub fn fit_objects(&mut self, rows: Vec<Vec<Bound<'_, PyAny>>>) -> PyResult<()> {
        let cols = py_rows_to_cols(rows)?;
        self.inner.fit_categorical(&cols)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        if self.inner.can_transform_numeric_inplace(mat.cols) {
            self.inner.transform_numeric_inplace(&mut mat)?;
            return matrix_to_pybytes(py, &mat);
        }
        let out = self.inner.transform_numeric(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn try_transform_into(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<bool> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        if !self.inner.can_transform_numeric_inplace(mat.cols) {
            return Ok(false);
        }
        self.inner.transform_numeric_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)?;
        Ok(true)
    }

    pub fn try_fit_transform_into(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
    ) -> PyResult<bool> {
        let mut mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit_numeric(&mat)?;
        if !self.inner.can_transform_numeric_inplace(mat.cols) {
            return Ok(false);
        }
        self.inner.transform_numeric_inplace(&mut mat)?;
        write_slice_to_pybuffer(py, &mat.data, &out)?;
        Ok(true)
    }

    pub fn transform_objects<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let cols = py_rows_to_cols(rows)?;
        let out_cols = self.inner.transform_categorical(&cols)?;
        let n_cols = out_cols.len();
        let n_rows = if n_cols > 0 { out_cols[0].len() } else { 0 };
        let mut out_rows = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let mut row = Vec::with_capacity(n_cols);
            for c in &out_cols {
                row.push(catvalue_to_py(py, &c[r])?);
            }
            out_rows.push(row);
        }
        Ok(out_rows)
    }

    #[getter]
    pub fn statistics_<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        match &self.inner.statistics_ {
            Some(st) => st
                .iter()
                .map(|v| catvalue_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map(Some),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn keep_cols_(&self) -> Vec<usize> {
        self.inner.keep_cols_.clone()
    }
    #[getter]
    pub fn indicator_features_(&self) -> Option<Vec<usize>> {
        self.inner
            .indicator_
            .as_ref()
            .and_then(|ind| ind.features_.clone())
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "KNNImputer")]
#[derive(Clone)]
pub struct PyKNNImputer {
    inner: KNNImputer,
}

crate::pymethods_with_dit!(PyKNNImputer, KNNImputer, {
    #[new]
    #[pyo3(signature = (missing_values = f64::NAN, n_neighbors = 5, weights = "uniform".to_string(), add_indicator = false, keep_empty_features = false, device = None))]
    pub fn new(
        missing_values: f64,
        n_neighbors: usize,
        weights: String,
        add_indicator: bool,
        keep_empty_features: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KNNImputer::new(
                missing_values,
                n_neighbors,
                weights,
                add_indicator,
                keep_empty_features,
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

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn indicator_features_(&self) -> Option<Vec<usize>> {
        self.inner
            .indicator_
            .as_ref()
            .and_then(|ind| ind.features_.clone())
    }
});

#[pyclass(from_py_object, name = "FeatureHasher")]
#[derive(Clone)]
pub struct PyFeatureHasher {
    inner: FeatureHasher,
}

crate::pymethods_with_dit!(PyFeatureHasher, FeatureHasher, {
    #[new]
    #[pyo3(signature = (n_features = 1_048_576, input_type = "dict".to_string(), alternate_sign = true, device = None))]
    pub fn new(
        n_features: usize,
        input_type: String,
        alternate_sign: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FeatureHasher::new(n_features, input_type, alternate_sign, device)?,
        })
    }

    pub fn transform_pairs_csr<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<(String, f64)>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let parts = self.inner.transform_pairs(&rows)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_dicts_csr<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyAny>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let pairs = extract_py_dict_pairs(rows, "=")?;
        let parts = self.inner.transform_pairs(&pairs)?;
        csr_parts_to_py(py, parts)
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.inner.n_features
    }
    #[getter]
    pub fn input_type(&self) -> String {
        self.inner.input_type.clone()
    }
    #[getter]
    pub fn alternate_sign(&self) -> bool {
        self.inner.alternate_sign
    }
});

#[pyclass(from_py_object, name = "DictVectorizer")]
#[derive(Clone)]
pub struct PyDictVectorizer {
    inner: DictVectorizer,
}

crate::pymethods_with_dit!(PyDictVectorizer, DictVectorizer, {
    #[new]
    #[pyo3(signature = (separator = "=".to_string(), sort = true, device = None))]
    pub fn new(separator: String, sort: bool, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: DictVectorizer::new(separator, sort, device)?,
        })
    }

    pub fn fit_pairs(&mut self, rows: Vec<Vec<(String, f64)>>) -> PyResult<()> {
        self.inner.fit(&rows)?;
        Ok(())
    }

    pub fn fit_dicts(&mut self, rows: &Bound<'_, PyAny>) -> PyResult<()> {
        let pairs = extract_py_dict_pairs(rows, &self.inner.separator)?;
        self.inner.fit(&pairs)?;
        Ok(())
    }

    pub fn transform_pairs_csr<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<(String, f64)>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let parts = self.inner.transform(&rows)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_dicts_csr<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyAny>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let pairs = extract_py_dict_pairs(rows, &self.inner.separator)?;
        let parts = self.inner.transform(&pairs)?;
        csr_parts_to_py(py, parts)
    }

    pub fn fit_transform_dicts_csr<'py>(
        &mut self,
        py: Python<'py>,
        rows: &Bound<'py, PyAny>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let pairs = extract_py_dict_pairs(rows, &self.inner.separator)?;
        self.inner.fit(&pairs)?;
        let parts = self.inner.transform(&pairs)?;
        csr_parts_to_py(py, parts)
    }

    pub fn restrict(&mut self, support: Vec<bool>) -> PyResult<()> {
        self.inner.restrict(&support)?;
        Ok(())
    }

    #[getter]
    pub fn feature_names_(&self) -> Option<Vec<String>> {
        self.inner.feature_names_.clone()
    }
    #[getter]
    pub fn vocabulary_(&self) -> Option<HashMap<String, usize>> {
        self.inner.vocabulary_.clone()
    }
});

#[pyclass(from_py_object, name = "CountVectorizer")]
#[derive(Clone)]
pub struct PyCountVectorizer {
    inner: CountVectorizer,
}

crate::pymethods_with_dit!(PyCountVectorizer, CountVectorizer, {
    #[new]
    #[pyo3(signature = (lowercase = true, analyzer = "word".to_string(), ngram_range = (1, 1), max_df = 1.0, min_df = 1.0, max_features = None, binary = false, stop_words = Vec::new(), fixed_vocab = None, device = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lowercase: bool,
        analyzer: String,
        ngram_range: (usize, usize),
        max_df: f64,
        min_df: f64,
        max_features: Option<usize>,
        binary: bool,
        stop_words: Vec<String>,
        fixed_vocab: Option<HashMap<String, usize>>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let sw: HashSet<String> = stop_words.into_iter().collect();
        Ok(Self {
            inner: CountVectorizer::new(
                lowercase,
                analyzer,
                ngram_range,
                max_df,
                min_df,
                max_features,
                binary,
                sw,
                fixed_vocab,
                device,
            )?,
        })
    }

    pub fn fit_docs(&mut self, docs: Vec<String>) -> PyResult<()> {
        let tokens = self.inner.analyze_docs(&docs);
        self.inner.fit_from_tokenized(&tokens)?;
        Ok(())
    }

    pub fn fit_tokenized(&mut self, docs_tokens: Vec<Vec<String>>) -> PyResult<()> {
        self.inner.fit_from_tokenized(&docs_tokens)?;
        Ok(())
    }

    pub fn transform_docs_csr<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<String>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let tokens = self.inner.analyze_docs(&docs);
        let parts = self.inner.transform_from_tokenized(&tokens)?;
        csr_parts_to_py(py, parts)
    }

    pub fn fit_transform_docs_csr<'py>(
        &mut self,
        py: Python<'py>,
        docs: Vec<String>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let tokens = self.inner.analyze_docs(&docs);
        self.inner.fit_from_tokenized(&tokens)?;
        let parts = self.inner.transform_from_tokenized(&tokens)?;
        csr_parts_to_py(py, parts)
    }

    pub fn fit_transform_docs_tfidf_csr<'py>(
        &mut self,
        py: Python<'py>,
        docs: Vec<String>,
        tfidf: &mut PyTfidfTransformer,
    ) -> PyResult<PyCsrTuple<'py>> {
        let tokens = self.inner.analyze_docs(&docs);
        self.inner.fit_from_tokenized(&tokens)?;
        let (data, indices, indptr, n_rows, n_cols) =
            self.inner.transform_from_tokenized(&tokens)?;
        tfidf
            .inner
            .fit_csr(&data, &indices, &indptr, n_rows, n_cols)?;
        let parts = tfidf
            .inner
            .transform_csr(&data, &indices, &indptr, n_rows, n_cols)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_docs_tfidf_csr<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<String>,
        tfidf: &PyTfidfTransformer,
    ) -> PyResult<PyCsrTuple<'py>> {
        let tokens = self.inner.analyze_docs(&docs);
        let (data, indices, indptr, n_rows, n_cols) =
            self.inner.transform_from_tokenized(&tokens)?;
        let parts = tfidf
            .inner
            .transform_csr(&data, &indices, &indptr, n_rows, n_cols)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_tokenized_csr<'py>(
        &self,
        py: Python<'py>,
        docs_tokens: Vec<Vec<String>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let parts = self.inner.transform_from_tokenized(&docs_tokens)?;
        csr_parts_to_py(py, parts)
    }

    #[getter]
    pub fn vocabulary_(&self) -> Option<HashMap<String, usize>> {
        self.inner.vocabulary_.clone()
    }
    #[getter]
    pub fn fixed_vocabulary_(&self) -> bool {
        self.inner.fixed_vocabulary_
    }
    #[getter]
    pub fn stop_words_(&self) -> Vec<String> {
        self.inner.stop_words_.iter().cloned().collect()
    }
});

#[pyclass(from_py_object, name = "TfidfTransformer")]
#[derive(Clone)]
pub struct PyTfidfTransformer {
    inner: TfidfTransformer,
}

crate::pymethods_with_dit!(PyTfidfTransformer, TfidfTransformer, {
    #[new]
    #[pyo3(signature = (norm = Some("l2".to_string()), use_idf = true, smooth_idf = true, sublinear_tf = false, device = None))]
    pub fn new(
        norm: Option<String>,
        use_idf: bool,
        smooth_idf: bool,
        sublinear_tf: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TfidfTransformer::new(norm, use_idf, smooth_idf, sublinear_tf, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit_dense(&mat)?;
        Ok(())
    }

    pub fn fit_csr(
        &mut self,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<()> {
        self.inner
            .fit_csr(&data, &indices, &indptr, n_rows, n_cols)?;
        Ok(())
    }

    pub fn fit_csr_buffer(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<f64>,
        indices: Bound<'_, PyAny>,
        indptr: Bound<'_, PyAny>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<()> {
        let d = buffer_to_f64_1d(py, &data)?;
        let idx = buffer_to_usize_1d(py, &indices)?;
        let ptr = buffer_to_usize_1d(py, &indptr)?;
        self.inner.fit_csr(&d, &idx, &ptr, n_rows, n_cols)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform_dense(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn transform_csr<'py>(
        &self,
        py: Python<'py>,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<PyCsrTuple<'py>> {
        let parts = self
            .inner
            .transform_csr(&data, &indices, &indptr, n_rows, n_cols)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_csr_buffer<'py>(
        &self,
        py: Python<'py>,
        data: PyBuffer<f64>,
        indices: Bound<'py, PyAny>,
        indptr: Bound<'py, PyAny>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<PyCsrTuple<'py>> {
        let d = buffer_to_f64_1d(py, &data)?;
        let idx = buffer_to_usize_1d(py, &indices)?;
        let ptr = buffer_to_usize_1d(py, &indptr)?;
        let parts = self.inner.transform_csr(&d, &idx, &ptr, n_rows, n_cols)?;
        csr_parts_to_py(py, parts)
    }

    pub fn fit_transform_csr_buffer<'py>(
        &mut self,
        py: Python<'py>,
        data: PyBuffer<f64>,
        indices: Bound<'py, PyAny>,
        indptr: Bound<'py, PyAny>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<PyCsrTuple<'py>> {
        let d = buffer_to_f64_1d(py, &data)?;
        let idx = buffer_to_usize_1d(py, &indices)?;
        let ptr = buffer_to_usize_1d(py, &indptr)?;
        self.inner.fit_csr(&d, &idx, &ptr, n_rows, n_cols)?;
        let parts = self.inner.transform_csr(&d, &idx, &ptr, n_rows, n_cols)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_csr_data_into(
        &self,
        py: Python<'_>,
        data: PyBuffer<f64>,
        indices: Bound<'_, PyAny>,
        indptr: Bound<'_, PyAny>,
        n_cols: usize,
        out_data: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut d = buffer_to_f64_1d(py, &data)?;
        let idx = buffer_to_usize_1d(py, &indices)?;
        let ptr = buffer_to_usize_1d(py, &indptr)?;
        self.inner
            .transform_csr_data_inplace(&mut d, &idx, &ptr, n_cols)?;
        write_slice_to_pybuffer(py, &d, &out_data)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_csr_data_into(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<f64>,
        indices: Bound<'_, PyAny>,
        indptr: Bound<'_, PyAny>,
        n_rows: usize,
        n_cols: usize,
        out_data: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mut d = buffer_to_f64_1d(py, &data)?;
        let idx = buffer_to_usize_1d(py, &indices)?;
        let ptr = buffer_to_usize_1d(py, &indptr)?;
        self.inner.fit_csr(&d, &idx, &ptr, n_rows, n_cols)?;
        self.inner
            .transform_csr_data_inplace(&mut d, &idx, &ptr, n_cols)?;
        write_slice_to_pybuffer(py, &d, &out_data)
    }

    #[getter]
    pub fn idf_(&self) -> Option<Vec<f64>> {
        self.inner.idf_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "HashingVectorizer")]
#[derive(Clone)]
pub struct PyHashingVectorizer {
    inner: HashingVectorizer,
}

crate::pymethods_with_dit!(PyHashingVectorizer, HashingVectorizer, {
    #[new]
    #[pyo3(signature = (lowercase = true, analyzer = "word".to_string(), ngram_range = (1, 1), n_features = 1_048_576, binary = false, norm = Some("l2".to_string()), alternate_sign = true, stop_words = Vec::new(), device = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lowercase: bool,
        analyzer: String,
        ngram_range: (usize, usize),
        n_features: usize,
        binary: bool,
        norm: Option<String>,
        alternate_sign: bool,
        stop_words: Vec<String>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let sw: HashSet<String> = stop_words.into_iter().collect();
        Ok(Self {
            inner: HashingVectorizer::new(
                lowercase,
                analyzer,
                ngram_range,
                n_features,
                binary,
                norm,
                alternate_sign,
                sw,
                device,
            )?,
        })
    }

    pub fn transform_docs_csr<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<String>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let tokens = self.inner.analyze_docs(&docs);
        let parts = self.inner.transform_from_tokenized(&tokens)?;
        csr_parts_to_py(py, parts)
    }

    pub fn transform_tokenized_csr<'py>(
        &self,
        py: Python<'py>,
        docs_tokens: Vec<Vec<String>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let parts = self.inner.transform_from_tokenized(&docs_tokens)?;
        csr_parts_to_py(py, parts)
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.inner.n_features
    }
});
