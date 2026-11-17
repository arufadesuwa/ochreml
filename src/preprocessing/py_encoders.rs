// OchreML PyO3 Bindings for Categorical & Target Encoders (*≧ω≦*)

use crate::preprocessing::encoders::{
    CatValue, LabelBinarizer, LabelEncoder, MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder,
    TargetEncoder, u32_buf_to_cat_cols,
};
use crate::preprocessing::py_helpers::{
    PyCsrTuple, buffer_to_matrix, buffer_to_u32_1d, catvalue_to_py, csr_parts_to_py,
    f64_matrix_to_cols, i64_slice_to_pybytes, matrix_to_pybytes, opt_catvalue_to_py,
    py_rows_to_cols, pyany_to_catvalue, write_slice_to_pybuffer,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList, PyString};
use std::collections::HashMap;

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

fn intern_py_col_list(
    py_col: &Bound<'_, PyList>,
) -> PyResult<(Vec<CatValue>, Vec<usize>, Vec<usize>)> {
    let n_rows = py_col.len();
    let mut str_map: HashMap<String, usize> = HashMap::new();
    let mut cat_map: HashMap<CatValue, usize> = HashMap::new();
    let mut raw_cats: Vec<CatValue> = Vec::new();
    let mut raw_counts: Vec<usize> = Vec::new();
    let mut raw_ids: Vec<usize> = Vec::with_capacity(n_rows);

    for item in py_col.iter() {
        let id = if let Ok(ps) = item.cast::<PyString>() {
            let s = ps.to_str()?;
            if let Some(&existing) = str_map.get(s) {
                raw_counts[existing] += 1;
                existing
            } else {
                let owned = s.to_owned();
                let new_id = raw_cats.len();
                str_map.insert(owned.clone(), new_id);
                raw_cats.push(CatValue::Str(owned));
                raw_counts.push(1);
                new_id
            }
        } else {
            let cv = pyany_to_catvalue(&item)?;
            if let Some(&existing) = cat_map.get(&cv) {
                raw_counts[existing] += 1;
                existing
            } else {
                let new_id = raw_cats.len();
                cat_map.insert(cv.clone(), new_id);
                raw_cats.push(cv);
                raw_counts.push(1);
                new_id
            }
        };
        raw_ids.push(id);
    }

    let n_uniq = raw_cats.len();
    let mut order: Vec<usize> = (0..n_uniq).collect();
    order.sort_by(|&a, &b| raw_cats[a].sort_cmp(&raw_cats[b]));

    let mut sorted_cats = Vec::with_capacity(n_uniq);
    let mut sorted_counts = Vec::with_capacity(n_uniq);
    let mut raw_to_sorted = vec![0usize; n_uniq];
    for (sorted_pos, &orig_idx) in order.iter().enumerate() {
        sorted_cats.push(raw_cats[orig_idx].clone());
        sorted_counts.push(raw_counts[orig_idx]);
        raw_to_sorted[orig_idx] = sorted_pos;
    }
    for id in &mut raw_ids {
        *id = raw_to_sorted[*id];
    }
    Ok((sorted_cats, sorted_counts, raw_ids))
}

fn build_py_col_lookup_maps(
    cats: &[CatValue],
) -> (HashMap<String, usize>, HashMap<CatValue, usize>) {
    let mut str_map = HashMap::with_capacity(cats.len());
    let mut other_map = HashMap::new();
    for (idx, c) in cats.iter().enumerate() {
        match c {
            CatValue::Str(s) => {
                str_map.insert(s.clone(), idx);
            }
            _ => {
                other_map.insert(c.clone(), idx);
            }
        }
    }
    (str_map, other_map)
}

fn lookup_py_col_item(
    item: &Bound<'_, PyAny>,
    str_map: &HashMap<String, usize>,
    other_map: &HashMap<CatValue, usize>,
) -> PyResult<Result<usize, CatValue>> {
    if let Ok(ps) = item.cast::<PyString>() {
        let s = ps.to_str()?;
        if let Some(&idx) = str_map.get(s) {
            return Ok(Ok(idx));
        }
        return Ok(Err(CatValue::Str(s.to_owned())));
    }
    let cv = pyany_to_catvalue(item)?;
    if let Some(&idx) = other_map.get(&cv) {
        return Ok(Ok(idx));
    }
    Ok(Err(cv))
}

fn py_col_lists_to_cat_cols(col_lists: &[Bound<'_, PyList>]) -> PyResult<Vec<Vec<CatValue>>> {
    if col_lists.is_empty() || col_lists[0].is_empty() {
        return Err(crate::error::OchreError::EmptyData.into());
    }
    let mut out = Vec::with_capacity(col_lists.len());
    for py_col in col_lists {
        let (cats, _, row_ids) = intern_py_col_list(py_col)?;
        out.push(row_ids.into_iter().map(|id| cats[id].clone()).collect());
    }
    Ok(out)
}

fn encode_ohe_col_lists(
    inner: &OneHotEncoder,
    col_lists: &[Bound<'_, PyList>],
) -> PyResult<(Vec<usize>, Vec<usize>, usize, usize)> {
    let n_features = inner
        .n_features_in_
        .ok_or(crate::error::OchreError::NotFitted)?;
    if col_lists.len() != n_features {
        return Err(crate::error::OchreError::FeatureCountMismatch {
            features: col_lists.len(),
            expected: n_features,
        }
        .into());
    }
    let n_rows = col_lists[0].len();
    let categories = inner
        .categories_
        .as_ref()
        .ok_or(crate::error::OchreError::NotFitted)?;
    let infreq = inner
        .infrequent_indices_
        .as_ref()
        .ok_or(crate::error::OchreError::NotFitted)?;
    let mut total_cols = 0usize;
    let mut col_encoded = Vec::with_capacity(n_features);
    for (j, py_col) in col_lists.iter().enumerate() {
        let off = total_cols;
        total_cols += inner.n_out_per_feat_[j];
        let out_map = &inner.feature_out_map_[j];
        let inf_out = infreq[j]
            .first()
            .and_then(|&cat_i| out_map[cat_i])
            .map(|l| off + l);
        let (str_map, other_map) = build_py_col_lookup_maps(&categories[j]);
        let mut res = Vec::with_capacity(n_rows);
        for item in py_col.iter() {
            match lookup_py_col_item(&item, &str_map, &other_map)? {
                Ok(cat_idx) => res.push(out_map[cat_idx].map(|l| off + l)),
                Err(_) if inner.handle_unknown == "ignore" => res.push(None),
                Err(_) if inner.handle_unknown == "infrequent_if_exist" => res.push(inf_out),
                Err(cv) => {
                    return Err(crate::error::OchreError::InvalidParameter(format!(
                        "Found unknown category '{}' in column {} during transform ( >_< )",
                        cv.display_key(),
                        j
                    ))
                    .into());
                }
            }
        }
        col_encoded.push(res);
    }
    Ok(OneHotEncoder::assemble_csr_from_encoded_cols(
        &col_encoded,
        n_rows,
        total_cols,
    ))
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

    #[pyo3(signature = (data, n_rows, n_cols, width, explicit_drop = None))]
    pub fn fit_utf32(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        self.inner
            .fit_utf32(&buf, n_rows, n_cols, width, ed.as_deref())?;
        Ok(())
    }

    pub fn transform_utf32_csr<'py>(
        &self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> PyResult<PyCsrTuple<'py>> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let (indices, indptr, nr, nc) = self
            .inner
            .transform_utf32_csr(&buf, n_rows, n_cols, width)?;
        let vals = vec![1.0; indices.len()];
        csr_parts_to_py(py, (vals, indices, indptr, nr, nc))
    }

    #[pyo3(signature = (data, n_rows, n_cols, width, explicit_drop = None))]
    pub fn fit_transform_utf32_csr<'py>(
        &mut self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        let (indices, indptr, nr, nc) =
            self.inner
                .fit_transform_utf32_csr(&buf, n_rows, n_cols, width, ed.as_deref())?;
        let vals = vec![1.0; indices.len()];
        csr_parts_to_py(py, (vals, indices, indptr, nr, nc))
    }

    pub fn transform_utf32_dense<'py>(
        &self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let (indices, indptr, nr, nc) = self
            .inner
            .transform_utf32_csr(&buf, n_rows, n_cols, width)?;
        let out = OneHotEncoder::csr_to_dense(&indices, &indptr, nr, nc)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (data, n_rows, n_cols, width, explicit_drop = None))]
    pub fn fit_transform_utf32_dense<'py>(
        &mut self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        let (indices, indptr, nr, nc) =
            self.inner
                .fit_transform_utf32_csr(&buf, n_rows, n_cols, width, ed.as_deref())?;
        let out = OneHotEncoder::csr_to_dense(&indices, &indptr, nr, nc)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (col_lists, explicit_drop = None))]
    pub fn fit_col_lists(
        &mut self,
        col_lists: Vec<Bound<'_, PyList>>,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        if col_lists.is_empty() || col_lists[0].is_empty() {
            return Err(crate::error::OchreError::EmptyData.into());
        }
        let n_rows = col_lists[0].len();
        let mut interned = Vec::with_capacity(col_lists.len());
        for col in &col_lists {
            let (cats, counts, _) = intern_py_col_list(col)?;
            interned.push((cats, counts));
        }
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        self.inner
            .fit_interned_cols(interned, n_rows, ed.as_deref())?;
        Ok(())
    }

    pub fn transform_col_lists_csr<'py>(
        &self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        let (indices, indptr, nr, nc) = encode_ohe_col_lists(&self.inner, &col_lists)?;
        let vals = vec![1.0; indices.len()];
        csr_parts_to_py(py, (vals, indices, indptr, nr, nc))
    }

    pub fn transform_col_lists_dense<'py>(
        &self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let (indices, indptr, nr, nc) = encode_ohe_col_lists(&self.inner, &col_lists)?;
        let out = OneHotEncoder::csr_to_dense(&indices, &indptr, nr, nc)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (col_lists, explicit_drop = None))]
    pub fn fit_transform_col_lists_csr<'py>(
        &mut self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<PyCsrTuple<'py>> {
        if col_lists.is_empty() || col_lists[0].is_empty() {
            return Err(crate::error::OchreError::EmptyData.into());
        }
        let n_rows = col_lists[0].len();
        let mut interned = Vec::with_capacity(col_lists.len());
        for col in &col_lists {
            interned.push(intern_py_col_list(col)?);
        }
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        let (indices, indptr, nr, nc) =
            self.inner
                .fit_transform_interned_csr(interned, n_rows, ed.as_deref())?;
        let vals = vec![1.0; indices.len()];
        csr_parts_to_py(py, (vals, indices, indptr, nr, nc))
    }

    #[pyo3(signature = (col_lists, explicit_drop = None))]
    pub fn fit_transform_col_lists_dense<'py>(
        &mut self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
        explicit_drop: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        if col_lists.is_empty() || col_lists[0].is_empty() {
            return Err(crate::error::OchreError::EmptyData.into());
        }
        let n_rows = col_lists[0].len();
        let mut interned = Vec::with_capacity(col_lists.len());
        for col in &col_lists {
            interned.push(intern_py_col_list(col)?);
        }
        let ed = match explicit_drop {
            Some(v) => Some(
                v.iter()
                    .map(|opt| opt.as_ref().map(pyany_to_catvalue).transpose())
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            None => None,
        };
        let (indices, indptr, nr, nc) =
            self.inner
                .fit_transform_interned_csr(interned, n_rows, ed.as_deref())?;
        let out = OneHotEncoder::csr_to_dense(&indices, &indptr, nr, nc)?;
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

    pub fn fit_utf32(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        self.inner.fit_utf32(&buf, n_rows, n_cols, width)?;
        Ok(())
    }

    pub fn transform_utf32_into(
        &self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let mat = self.inner.transform_utf32(&buf, n_rows, n_cols, width)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_transform_utf32_into(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let mat = self
            .inner
            .fit_transform_utf32(&buf, n_rows, n_cols, width)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
    }

    pub fn fit_col_lists(&mut self, col_lists: Vec<Bound<'_, PyList>>) -> PyResult<()> {
        if col_lists.is_empty() || col_lists[0].is_empty() {
            return Err(crate::error::OchreError::EmptyData.into());
        }
        let n_rows = col_lists[0].len();
        let mut interned = Vec::with_capacity(col_lists.len());
        for col in &col_lists {
            let (cats, counts, _) = intern_py_col_list(col)?;
            interned.push((cats, counts));
        }
        self.inner.fit_interned_cols(interned, n_rows)?;
        Ok(())
    }

    pub fn transform_col_lists_into(
        &self,
        py: Python<'_>,
        col_lists: Vec<Bound<'_, PyList>>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        let n_features = self
            .inner
            .n_features_in_
            .ok_or(crate::error::OchreError::NotFitted)?;
        if col_lists.len() != n_features {
            return Err(crate::error::OchreError::FeatureCountMismatch {
                features: col_lists.len(),
                expected: n_features,
            }
            .into());
        }
        let n_rows = col_lists[0].len();
        let categories = self
            .inner
            .categories_
            .as_ref()
            .ok_or(crate::error::OchreError::NotFitted)?;
        let mut out_data = vec![0.0; n_rows * n_features];
        for (j, py_col) in col_lists.iter().enumerate() {
            let (str_map, other_map) = build_py_col_lookup_maps(&categories[j]);
            let codes = &self.inner.cat_to_code_[j];
            for (r, item) in py_col.iter().enumerate() {
                let val = match lookup_py_col_item(&item, &str_map, &other_map)? {
                    Ok(cat_idx) => codes[cat_idx] as f64,
                    Err(cv) if cv.is_missing() => self.inner.encoded_missing_value,
                    Err(_) if self.inner.handle_unknown == "use_encoded_value" => {
                        self.inner.unknown_value.unwrap_or(f64::NAN)
                    }
                    Err(cv) => {
                        return Err(crate::error::OchreError::InvalidParameter(format!(
                            "Found unknown category '{}' in column {} during transform ( >_< )",
                            cv.display_key(),
                            j
                        ))
                        .into());
                    }
                };
                out_data[r * n_features + j] = val;
            }
        }
        write_slice_to_pybuffer(py, &out_data, &out)
    }

    pub fn fit_transform_col_lists_into(
        &mut self,
        py: Python<'_>,
        col_lists: Vec<Bound<'_, PyList>>,
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        if col_lists.is_empty() || col_lists[0].is_empty() {
            return Err(crate::error::OchreError::EmptyData.into());
        }
        let n_rows = col_lists[0].len();
        let mut interned = Vec::with_capacity(col_lists.len());
        for col in &col_lists {
            interned.push(intern_py_col_list(col)?);
        }
        let mat = self.inner.fit_transform_interned(interned, n_rows)?;
        write_slice_to_pybuffer(py, &mat.data, &out)
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

    pub fn fit_utf32(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        self.inner.fit_utf32(&buf, n_rows, width)?;
        Ok(())
    }

    pub fn transform_utf32_into(
        &self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
        out: PyBuffer<i64>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let res = self.inner.transform_utf32(&buf, n_rows, width)?;
        write_slice_to_pybuffer(py, &res, &out)
    }

    pub fn fit_transform_utf32_into(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
        out: PyBuffer<i64>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let res = self.inner.fit_transform_utf32(&buf, n_rows, width)?;
        write_slice_to_pybuffer(py, &res, &out)
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

    pub fn fit_utf32(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        self.inner.fit_utf32(&buf, n_rows, width)?;
        Ok(())
    }

    pub fn transform_utf32<'py>(
        &self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let (rows, cols, out) = self.inner.transform_utf32(&buf, n_rows, width)?;
        let bytes = i64_slice_to_pybytes(py, &out)?;
        Ok((rows, cols, bytes))
    }

    pub fn fit_transform_utf32<'py>(
        &mut self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        width: usize,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let (rows, cols, out) = self.inner.fit_transform_utf32(&buf, n_rows, width)?;
        let bytes = i64_slice_to_pybytes(py, &out)?;
        Ok((rows, cols, bytes))
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

    #[pyo3(signature = (col_lists, y_cols, classes, inferred_target_type, explicit_categories = None))]
    pub fn fit_col_lists(
        &mut self,
        col_lists: Vec<Bound<'_, PyList>>,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let cols = py_col_lists_to_cat_cols(&col_lists)?;
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

    #[pyo3(signature = (col_lists, y_cols, classes, inferred_target_type, fold_ids, explicit_categories = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_col_lists<'py>(
        &mut self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        fold_ids: Vec<usize>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_col_lists_to_cat_cols(&col_lists)?;
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

    pub fn transform_col_lists<'py>(
        &self,
        py: Python<'py>,
        col_lists: Vec<Bound<'_, PyList>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let cols = py_col_lists_to_cat_cols(&col_lists)?;
        let out = self.inner.transform(&cols)?;
        matrix_to_pybytes(py, &out)
    }

    #[pyo3(signature = (data, n_rows, n_cols, width, y_cols, classes, inferred_target_type, explicit_categories = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit_utf32(
        &mut self,
        py: Python<'_>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<()> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let cols = u32_buf_to_cat_cols(&buf, n_rows, n_cols, width)?;
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

    #[pyo3(signature = (data, n_rows, n_cols, width, y_cols, classes, inferred_target_type, fold_ids, explicit_categories = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_utf32<'py>(
        &mut self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
        y_cols: Vec<Vec<f64>>,
        classes: Option<Vec<Bound<'_, PyAny>>>,
        inferred_target_type: String,
        fold_ids: Vec<usize>,
        explicit_categories: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let cols = u32_buf_to_cat_cols(&buf, n_rows, n_cols, width)?;
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

    pub fn transform_utf32<'py>(
        &self,
        py: Python<'py>,
        data: PyBuffer<u32>,
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let buf = buffer_to_u32_1d(py, &data)?;
        let cols = u32_buf_to_cat_cols(&buf, n_rows, n_cols, width)?;
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
