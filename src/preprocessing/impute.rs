// OchreML Missing Value Imputation Engine (*≧ω≦*)
// Implements SimpleImputer (N1 col_reduce_stats + median/most_frequent/constant),
// MissingIndicator, and KNNImputer (N3 pairwise_l2_sq / masked matmul + N4 topk_argmin).

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_matrix,
    write_opt_str, write_opt_usize, write_opt_usize_vec, write_str, write_usize, write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::encoders::CatValue;
use crate::preprocessing::math_utils::{percentile_sorted, slice_has_nan};
use rayon::prelude::*;
use std::collections::HashMap;

pub const TAG_SIMPLE_IMPUTER: u8 = 25;
pub const TAG_MISSING_INDICATOR: u8 = 26;
pub const TAG_KNN_IMPUTER: u8 = 27;

fn resolve_loaded_device(
    saved_device: Option<String>,
    is_fitted: bool,
    device: Option<&str>,
) -> OchreResult<(DeviceType, Option<String>)> {
    let dev = parse_device(device.or(Some("auto")))?;
    let device_ = if saved_device.is_some() || is_fitted {
        Some(dev.to_string())
    } else {
        None
    };
    Ok((dev, device_))
}

#[inline]
fn is_missing_f64(v: f64, missing_val: f64) -> bool {
    if missing_val.is_nan() {
        v.is_nan()
    } else {
        v == missing_val
    }
}

/// Binary indicator for missing values (＾▽＾)
#[derive(Debug, Clone)]
pub struct MissingIndicator {
    pub missing_values: f64,
    pub features_mode: String, // "missing-only" or "all"
    pub error_on_new: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub features_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
}

impl MissingIndicator {
    pub fn new(
        missing_values: f64,
        features_mode: String,
        error_on_new: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if features_mode != "missing-only" && features_mode != "all" {
            return Err(OchreError::InvalidParameter(format!(
                "features must be 'missing-only' or 'all', got '{}' ( >_< )",
                features_mode
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            missing_values,
            features_mode,
            error_on_new,
            device: dev,
            device_: None,
            features_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let cols = x.cols;
        let rows = x.rows;
        let mv = self.missing_values;
        let features: Vec<usize> = if self.features_mode == "all" {
            (0..cols).collect()
        } else {
            (0..cols)
                .into_par_iter()
                .filter(|&j| (0..rows).any(|r| is_missing_f64(x.data[r * cols + j], mv)))
                .collect()
        };
        self.features_ = Some(features);
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let features = self.features_.as_ref().ok_or(OchreError::NotFitted)?;
        let mv = self.missing_values;

        if self.error_on_new && self.features_mode == "missing-only" {
            let mut is_selected = vec![false; expected];
            for &j in features {
                is_selected[j] = true;
            }
            for (j, &selected) in is_selected.iter().enumerate() {
                if !selected && (0..x.rows).any(|r| is_missing_f64(x.data[r * expected + j], mv)) {
                    return Err(OchreError::InvalidParameter(format!(
                        "MissingIndicator: column {} had no missing values during fit but has missing values during transform ( >_< )",
                        j
                    )));
                }
            }
        }

        let n_out = features.len();
        let mut out = vec![0.0; x.rows * n_out];
        for r in 0..x.rows {
            for (k, &j) in features.iter().enumerate() {
                out[r * n_out + k] = if is_missing_f64(x.data[r * expected + j], mv) {
                    1.0
                } else {
                    0.0
                };
            }
        }
        Matrix::from_vec(x.rows, n_out, out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MISSING_INDICATOR);
        write_f64(&mut buf, self.missing_values);
        write_str(&mut buf, &self.features_mode);
        write_bool(&mut buf, self.error_on_new);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize_vec(&mut buf, &self.features_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_MISSING_INDICATOR)?;
        let missing_values = r.read_f64()?;
        let features_mode = r.read_str()?;
        let error_on_new = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let features_ = r.read_opt_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            missing_values,
            features_mode,
            error_on_new,
            device: dev,
            device_,
            features_,
            n_features_in_,
        })
    }
}

/// Univariate imputer for completing missing values (o´∀｀o)
#[derive(Debug, Clone)]
pub struct SimpleImputer {
    pub missing_values: CatValue,
    pub strategy: String, // "mean", "median", "most_frequent", "constant"
    pub fill_value: Option<CatValue>,
    pub add_indicator: bool,
    pub keep_empty_features: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub statistics_: Option<Vec<CatValue>>,
    pub keep_cols_: Vec<usize>,
    pub indicator_: Option<MissingIndicator>,
    pub n_features_in_: Option<usize>,
}

impl SimpleImputer {
    pub fn new(
        missing_values: CatValue,
        strategy: String,
        fill_value: Option<CatValue>,
        add_indicator: bool,
        keep_empty_features: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !matches!(
            strategy.as_str(),
            "mean" | "median" | "most_frequent" | "constant"
        ) {
            return Err(OchreError::InvalidParameter(format!(
                "SimpleImputer strategy must be 'mean', 'median', 'most_frequent', or 'constant', got '{}' ( >_< )",
                strategy
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            missing_values,
            strategy,
            fill_value,
            add_indicator,
            keep_empty_features,
            device: dev,
            device_: None,
            statistics_: None,
            keep_cols_: Vec::new(),
            indicator_: None,
            n_features_in_: None,
        })
    }

    pub fn fit_numeric(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let rows = x.rows;
        let cols = x.cols;
        let mv = match self.missing_values {
            CatValue::Missing => f64::NAN,
            CatValue::Float(bits) => f64::from_bits(bits),
            CatValue::Int(i) => i as f64,
            _ => f64::NAN,
        };

        // Fast N1 path when strategy == "mean" and no missing values exist
        let stats_vec: Vec<CatValue> =
            if self.strategy == "mean" && mv.is_nan() && !slice_has_nan(&x.data) {
                let stats = get_backend(&self.device).col_reduce_stats(x, None)?;
                stats.mean.into_iter().map(CatValue::from_f64).collect()
            } else {
                let fill_num = match &self.fill_value {
                    Some(CatValue::Float(b)) => f64::from_bits(*b),
                    Some(CatValue::Int(i)) => *i as f64,
                    _ => 0.0,
                };
                let strategy = self.strategy.as_str();
                (0..cols)
                    .into_par_iter()
                    .map(|j| {
                        let mut vals: Vec<f64> = (0..rows)
                            .map(|r| x.data[r * cols + j])
                            .filter(|&v| !is_missing_f64(v, mv))
                            .collect();
                        if vals.is_empty() {
                            if strategy == "constant" {
                                return Ok(CatValue::from_f64(fill_num));
                            }
                            return Ok(CatValue::Missing);
                        }
                        match strategy {
                            "mean" => {
                                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                                Ok(CatValue::from_f64(m))
                            }
                            "median" => {
                                vals.sort_by(|a, b| a.total_cmp(b));
                                Ok(CatValue::from_f64(percentile_sorted(&vals, 50.0)))
                            }
                            "most_frequent" => {
                                vals.sort_by(|a, b| a.total_cmp(b));
                                let mut best_val = vals[0];
                                let mut best_cnt = 1usize;
                                let mut cur_val = vals[0];
                                let mut cur_cnt = 1usize;
                                for &v in vals.iter().skip(1) {
                                    if v == cur_val {
                                        cur_cnt += 1;
                                    } else {
                                        if cur_cnt > best_cnt {
                                            best_cnt = cur_cnt;
                                            best_val = cur_val;
                                        }
                                        cur_val = v;
                                        cur_cnt = 1;
                                    }
                                }
                                if cur_cnt > best_cnt {
                                    best_val = cur_val;
                                }
                                Ok(CatValue::from_f64(best_val))
                            }
                            _ => Ok(CatValue::from_f64(fill_num)),
                        }
                    })
                    .collect::<OchreResult<Vec<CatValue>>>()?
            };

        let mut keep_cols = Vec::with_capacity(cols);
        let mut final_stats = stats_vec;
        for (j, st) in final_stats.iter_mut().enumerate() {
            if st.is_missing() {
                if self.keep_empty_features {
                    *st = CatValue::from_f64(0.0);
                    keep_cols.push(j);
                }
            } else {
                keep_cols.push(j);
            }
        }

        if self.add_indicator {
            let mut ind = MissingIndicator::new(
                mv,
                "missing-only".to_string(),
                false,
                Some(self.device.to_string()),
            )?;
            ind.fit(x)?;
            self.indicator_ = Some(ind);
        } else {
            self.indicator_ = None;
        }

        self.statistics_ = Some(final_stats);
        self.keep_cols_ = keep_cols;
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_categorical(&mut self, cols_data: &[Vec<CatValue>]) -> OchreResult<()> {
        if cols_data.is_empty() || cols_data[0].is_empty() {
            return Err(OchreError::EmptyData);
        }
        if self.strategy == "mean" || self.strategy == "median" {
            return Err(OchreError::InvalidParameter(format!(
                "Cannot use strategy '{}' with non-numeric data ( >_< )",
                self.strategy
            )));
        }
        let cols = cols_data.len();
        let mut final_stats = Vec::with_capacity(cols);
        let mut keep_cols = Vec::with_capacity(cols);

        for (j, col) in cols_data.iter().enumerate() {
            let valid: Vec<&CatValue> = col
                .iter()
                .filter(|v| !v.is_missing() && **v != self.missing_values)
                .collect();
            let stat = if valid.is_empty() {
                if self.strategy == "constant" {
                    self.fill_value
                        .clone()
                        .unwrap_or_else(|| CatValue::Str("missing_value".to_string()))
                } else {
                    CatValue::Missing
                }
            } else if self.strategy == "constant" {
                self.fill_value
                    .clone()
                    .unwrap_or_else(|| CatValue::Str("missing_value".to_string()))
            } else {
                let mut counts: HashMap<CatValue, usize> = HashMap::new();
                for v in valid {
                    *counts.entry(v.clone()).or_insert(0) += 1;
                }
                let mut pairs: Vec<(CatValue, usize)> = counts.into_iter().collect();
                pairs.sort_by(|(a, ca), (b, cb)| cb.cmp(ca).then_with(|| a.sort_cmp(b)));
                pairs
                    .into_iter()
                    .next()
                    .map(|(v, _)| v)
                    .unwrap_or(CatValue::Missing)
            };

            if stat.is_missing() {
                if self.keep_empty_features {
                    final_stats.push(CatValue::Str("missing_value".to_string()));
                    keep_cols.push(j);
                } else {
                    final_stats.push(CatValue::Missing);
                }
            } else {
                final_stats.push(stat);
                keep_cols.push(j);
            }
        }

        self.statistics_ = Some(final_stats);
        self.keep_cols_ = keep_cols;
        self.indicator_ = None;
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform_numeric(&self, x: &Matrix) -> OchreResult<Matrix> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let stats = self.statistics_.as_ref().ok_or(OchreError::NotFitted)?;
        let mv = match self.missing_values {
            CatValue::Missing => f64::NAN,
            CatValue::Float(bits) => f64::from_bits(bits),
            CatValue::Int(i) => i as f64,
            _ => f64::NAN,
        };
        let fill_vals: Vec<f64> = self
            .keep_cols_
            .iter()
            .map(|&j| match stats[j] {
                CatValue::Float(b) => f64::from_bits(b),
                CatValue::Int(i) => i as f64,
                _ => f64::NAN,
            })
            .collect();

        let ind_mat = if let Some(ref ind) = self.indicator_ {
            Some(ind.transform(x)?)
        } else {
            None
        };
        let ind_cols = ind_mat.as_ref().map_or(0, |m| m.cols);
        let n_keep = self.keep_cols_.len();
        let out_cols = n_keep + ind_cols;
        let mut out = vec![0.0; x.rows * out_cols];

        for r in 0..x.rows {
            for (k, &j) in self.keep_cols_.iter().enumerate() {
                let v = x.data[r * expected + j];
                out[r * out_cols + k] = if is_missing_f64(v, mv) {
                    fill_vals[k]
                } else {
                    v
                };
            }
            if let Some(ref im) = ind_mat {
                for k in 0..ind_cols {
                    out[r * out_cols + n_keep + k] = im.data[r * ind_cols + k];
                }
            }
        }
        Matrix::from_vec(x.rows, out_cols, out)
    }

    pub fn transform_categorical(
        &self,
        cols_data: &[Vec<CatValue>],
    ) -> OchreResult<Vec<Vec<CatValue>>> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if cols_data.len() != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: cols_data.len(),
                expected,
            });
        }
        let stats = self.statistics_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_samples = cols_data[0].len();
        let mut out = Vec::with_capacity(self.keep_cols_.len());
        for &j in &self.keep_cols_ {
            let fill = &stats[j];
            let mut col_out = Vec::with_capacity(n_samples);
            for v in &cols_data[j] {
                if v.is_missing() || *v == self.missing_values {
                    col_out.push(fill.clone());
                } else {
                    col_out.push(v.clone());
                }
            }
            out.push(col_out);
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        if let Some(ref mut ind) = self.indicator_ {
            ind.set_device(Some(dev.to_string()))?;
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SIMPLE_IMPUTER);
        self.missing_values.write_to(&mut buf);
        write_str(&mut buf, &self.strategy);
        match &self.fill_value {
            Some(fv) => {
                write_bool(&mut buf, true);
                fv.write_to(&mut buf);
            }
            None => write_bool(&mut buf, false),
        }
        write_bool(&mut buf, self.add_indicator);
        write_bool(&mut buf, self.keep_empty_features);
        write_opt_str(&mut buf, &self.device_);
        match &self.statistics_ {
            Some(st) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, st.len());
                for s in st {
                    s.write_to(&mut buf);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_usize_vec(&mut buf, &self.keep_cols_);
        match &self.indicator_ {
            Some(ind) => {
                write_bool(&mut buf, true);
                let ib = ind.to_bytes();
                write_usize(&mut buf, ib.len());
                buf.extend_from_slice(&ib);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SIMPLE_IMPUTER)?;
        let missing_values = CatValue::read_from(&mut r)?;
        let strategy = r.read_str()?;
        let fill_value = if r.read_bool()? {
            Some(CatValue::read_from(&mut r)?)
        } else {
            None
        };
        let add_indicator = r.read_bool()?;
        let keep_empty_features = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let statistics_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut st = Vec::with_capacity(len);
            for _ in 0..len {
                st.push(CatValue::read_from(&mut r)?);
            }
            Some(st)
        } else {
            None
        };
        let keep_cols_ = r.read_usize_vec()?;
        let indicator_ = if r.read_bool()? {
            let ilen = r.read_usize()?;
            let ib = r.read_exact(ilen)?;
            Some(MissingIndicator::from_bytes(ib, device)?)
        } else {
            None
        };
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            missing_values,
            strategy,
            fill_value,
            add_indicator,
            keep_empty_features,
            device: dev,
            device_,
            statistics_,
            keep_cols_,
            indicator_,
            n_features_in_,
        })
    }
}

/// Imputation using k-Nearest Neighbors accelerated by Backend N3 (pairwise_l2_sq / matmul)
/// and N4 (topk_argmin) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct KNNImputer {
    pub missing_values: f64,
    pub n_neighbors: usize,
    pub weights: String, // "uniform" or "distance"
    pub add_indicator: bool,
    pub keep_empty_features: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub fit_x_: Option<Matrix>,
    pub valid_cols_: Vec<usize>,
    pub indicator_: Option<MissingIndicator>,
    pub n_features_in_: Option<usize>,
}

impl KNNImputer {
    pub fn new(
        missing_values: f64,
        n_neighbors: usize,
        weights: String,
        add_indicator: bool,
        keep_empty_features: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_neighbors == 0 {
            return Err(OchreError::InvalidParameter(
                "n_neighbors must be >= 1 ( >_< )".to_string(),
            ));
        }
        if weights != "uniform" && weights != "distance" {
            return Err(OchreError::InvalidParameter(format!(
                "weights must be 'uniform' or 'distance', got '{}' ( >_< )",
                weights
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            missing_values,
            n_neighbors,
            weights,
            add_indicator,
            keep_empty_features,
            device: dev,
            device_: None,
            fit_x_: None,
            valid_cols_: Vec::new(),
            indicator_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let cols = x.cols;
        let mv = self.missing_values;
        let valid_cols: Vec<usize> = (0..cols)
            .filter(|&j| {
                self.keep_empty_features
                    || (0..x.rows).any(|r| !is_missing_f64(x.data[r * cols + j], mv))
            })
            .collect();

        if self.add_indicator {
            let mut ind = MissingIndicator::new(
                mv,
                "missing-only".to_string(),
                false,
                Some(self.device.to_string()),
            )?;
            ind.fit(x)?;
            self.indicator_ = Some(ind);
        } else {
            self.indicator_ = None;
        }

        self.fit_x_ = Some(x.clone());
        self.valid_cols_ = valid_cols;
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Compute `(n_test x n_train)` nan-Euclidean squared distance matrix using Backend N3 / matmul.
    fn compute_nan_l2_sq(&self, x: &Matrix, fit_x: &Matrix) -> OchreResult<Matrix> {
        let mv = self.missing_values;
        let x_has_miss = if mv.is_nan() {
            slice_has_nan(&x.data)
        } else {
            x.data.contains(&mv)
        };
        let f_has_miss = if mv.is_nan() {
            slice_has_nan(&fit_x.data)
        } else {
            fit_x.data.contains(&mv)
        };
        let backend = get_backend(&self.device);

        // Fast exact N3 path when neither matrix has missing values
        if !x_has_miss && !f_has_miss {
            return backend.pairwise_l2_sq(x, fit_x);
        }

        let p = x.cols as f64;
        let nx = x.rows;
        let nf = fit_x.rows;
        let cols = x.cols;

        // Build zero-filled X0, X0^2, mask M_x, and transposed B0^T, (B0^2)^T, M_b^T for Backend::matmul
        let mut x0 = vec![0.0; nx * cols];
        let mut x0_sq = vec![0.0; nx * cols];
        let mut mx = vec![0.0; nx * cols];
        for i in 0..nx * cols {
            let v = x.data[i];
            if !is_missing_f64(v, mv) {
                x0[i] = v;
                x0_sq[i] = v * v;
                mx[i] = 1.0;
            }
        }

        let mut b0_t = vec![0.0; cols * nf];
        let mut b0_sq_t = vec![0.0; cols * nf];
        let mut mb_t = vec![0.0; cols * nf];
        for r in 0..nf {
            for c in 0..cols {
                let v = fit_x.data[r * cols + c];
                if !is_missing_f64(v, mv) {
                    let idx = c * nf + r;
                    b0_t[idx] = v;
                    b0_sq_t[idx] = v * v;
                    mb_t[idx] = 1.0;
                }
            }
        }

        let x0_m = Matrix::from_vec(nx, cols, x0)?;
        let x0_sq_m = Matrix::from_vec(nx, cols, x0_sq)?;
        let mx_m = Matrix::from_vec(nx, cols, mx)?;
        let b0_tm = Matrix::from_vec(cols, nf, b0_t)?;
        let b0_sq_tm = Matrix::from_vec(cols, nf, b0_sq_t)?;
        let mb_tm = Matrix::from_vec(cols, nf, mb_t)?;

        let cross = backend.matmul(&x0_m, &b0_tm)?;
        let term_a = backend.matmul(&x0_sq_m, &mb_tm)?;
        let term_b = backend.matmul(&mx_m, &b0_sq_tm)?;
        let counts = backend.matmul(&mx_m, &mb_tm)?;

        let mut dist_sq = vec![f64::INFINITY; nx * nf];
        for (idx, dst) in dist_sq.iter_mut().enumerate() {
            let cnt = counts.data[idx];
            if cnt > 0.0 {
                let sq = (term_a.data[idx] + term_b.data[idx] - 2.0 * cross.data[idx]).max(0.0);
                *dst = (p / cnt) * sq;
            }
        }
        Matrix::from_vec(nx, nf, dist_sq)
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let fit_x = self.fit_x_.as_ref().ok_or(OchreError::NotFitted)?;
        let mv = self.missing_values;
        let dist_sq_mat = self.compute_nan_l2_sq(x, fit_x)?;
        let backend = get_backend(&self.device);
        let nf = fit_x.rows;

        // Start with a copy of `x` (filtering out all-NaN columns if !keep_empty_features)
        let mut imputed = x.data.clone();
        let is_dist_weight = self.weights == "distance";

        for &j in &self.valid_cols_ {
            // Find rows in `x` that need imputation in column `j`
            let receiver_rows: Vec<usize> = (0..x.rows)
                .filter(|&r| is_missing_f64(x.data[r * expected + j], mv))
                .collect();
            if receiver_rows.is_empty() {
                continue;
            }

            // Potential donors in `fit_x` that are NOT missing column `j`
            let donors: Vec<usize> = (0..nf)
                .filter(|&r| !is_missing_f64(fit_x.data[r * expected + j], mv))
                .collect();

            if donors.is_empty() {
                for &r in &receiver_rows {
                    imputed[r * expected + j] = 0.0;
                }
                continue;
            }

            let k_eff = self.n_neighbors.min(donors.len());
            // Build `(n_receivers x n_donors)` distance submatrix and call Backend::topk_argmin (N4)
            let mut sub_d = Vec::with_capacity(receiver_rows.len() * donors.len());
            for &ri in &receiver_rows {
                let row_off = ri * nf;
                for &di in &donors {
                    sub_d.push(dist_sq_mat.data[row_off + di]);
                }
            }
            let sub_mat = Matrix::from_vec(receiver_rows.len(), donors.len(), sub_d)?;
            let (topk_local_idx, topk_dist_sq) = backend.topk_argmin(&sub_mat, k_eff)?;

            for (r_pos, &ri) in receiver_rows.iter().enumerate() {
                let idx_slice = &topk_local_idx[r_pos * k_eff..(r_pos + 1) * k_eff];
                let dsq_slice = &topk_dist_sq.data[r_pos * k_eff..(r_pos + 1) * k_eff];

                // Filter out unreachable donors (`INFINITY`, no overlapping non-missing coords)
                let mut valid_pairs: Vec<(f64, f64)> = Vec::with_capacity(k_eff);
                for (&loc_d, &dsq) in idx_slice.iter().zip(dsq_slice.iter()) {
                    if dsq.is_finite() {
                        let donor_row = donors[loc_d];
                        valid_pairs.push((dsq.sqrt(), fit_x.data[donor_row * expected + j]));
                    }
                }

                if valid_pairs.is_empty() {
                    // Fallback: mean of all potential donors for column `j`
                    let mean_donor: f64 = donors
                        .iter()
                        .map(|&di| fit_x.data[di * expected + j])
                        .sum::<f64>()
                        / (donors.len() as f64);
                    imputed[ri * expected + j] = mean_donor;
                    continue;
                }

                let val = if !is_dist_weight {
                    valid_pairs.iter().map(|&(_, v)| v).sum::<f64>() / (valid_pairs.len() as f64)
                } else {
                    let zero_vals: Vec<f64> = valid_pairs
                        .iter()
                        .filter(|&&(d, _)| d == 0.0)
                        .map(|&(_, v)| v)
                        .collect();
                    if !zero_vals.is_empty() {
                        zero_vals.iter().sum::<f64>() / (zero_vals.len() as f64)
                    } else {
                        let mut w_sum = 0.0;
                        let mut vw_sum = 0.0;
                        for (d, v) in valid_pairs {
                            let w = 1.0 / d;
                            w_sum += w;
                            vw_sum += w * v;
                        }
                        vw_sum / w_sum
                    }
                };
                imputed[ri * expected + j] = val;
            }
        }

        let ind_mat = if let Some(ref ind) = self.indicator_ {
            Some(ind.transform(x)?)
        } else {
            None
        };
        let ind_cols = ind_mat.as_ref().map_or(0, |m| m.cols);
        let n_keep = self.valid_cols_.len();
        let out_cols = n_keep + ind_cols;

        if n_keep == expected && ind_cols == 0 {
            return Matrix::from_vec(x.rows, expected, imputed);
        }

        let mut out = vec![0.0; x.rows * out_cols];
        for r in 0..x.rows {
            for (k, &j) in self.valid_cols_.iter().enumerate() {
                out[r * out_cols + k] = imputed[r * expected + j];
            }
            if let Some(ref im) = ind_mat {
                for k in 0..ind_cols {
                    out[r * out_cols + n_keep + k] = im.data[r * ind_cols + k];
                }
            }
        }
        Matrix::from_vec(x.rows, out_cols, out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        if let Some(ref mut ind) = self.indicator_ {
            ind.set_device(Some(dev.to_string()))?;
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_KNN_IMPUTER);
        write_f64(&mut buf, self.missing_values);
        write_usize(&mut buf, self.n_neighbors);
        write_str(&mut buf, &self.weights);
        write_bool(&mut buf, self.add_indicator);
        write_bool(&mut buf, self.keep_empty_features);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.fit_x_);
        write_usize_vec(&mut buf, &self.valid_cols_);
        match &self.indicator_ {
            Some(ind) => {
                write_bool(&mut buf, true);
                let ib = ind.to_bytes();
                write_usize(&mut buf, ib.len());
                buf.extend_from_slice(&ib);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_KNN_IMPUTER)?;
        let missing_values = r.read_f64()?;
        let n_neighbors = r.read_usize()?;
        let weights = r.read_str()?;
        let add_indicator = r.read_bool()?;
        let keep_empty_features = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let fit_x_ = r.read_opt_matrix()?;
        let valid_cols_ = r.read_usize_vec()?;
        let indicator_ = if r.read_bool()? {
            let ilen = r.read_usize()?;
            let ib = r.read_exact(ilen)?;
            Some(MissingIndicator::from_bytes(ib, device)?)
        } else {
            None
        };
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            missing_values,
            n_neighbors,
            weights,
            add_indicator,
            keep_empty_features,
            device: dev,
            device_,
            fit_x_,
            valid_cols_,
            indicator_,
            n_features_in_,
        })
    }
}
