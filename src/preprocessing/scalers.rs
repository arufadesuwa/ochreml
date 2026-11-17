// OchreML Scaling & Normalization Engine (*≧ω≦*)
// Implements StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, and Normalizer
// accelerated by Backend Primitives N1 (col_reduce_stats, col_scale_shift) and N2 (elementwise_op).

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64_vec, write_opt_str, write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::math_utils::{
    compute_nan_col_stats, handle_zeros_in_scale, ndtri, percentile_sorted, slice_has_nan,
};
use crate::primitives::ElementwiseOp;
use rayon::prelude::*;

pub const TAG_STANDARD_SCALER: u8 = 7;
pub const TAG_MINMAX_SCALER: u8 = 8;
pub const TAG_MAXABS_SCALER: u8 = 9;
pub const TAG_ROBUST_SCALER: u8 = 10;
pub const TAG_NORMALIZER: u8 = 11;

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

fn validate_fitted_cols(n_features_in: Option<usize>, cols: usize) -> OchreResult<usize> {
    let expected = n_features_in.ok_or(OchreError::NotFitted)?;
    if cols != expected {
        return Err(OchreError::FeatureCountMismatch {
            features: cols,
            expected,
        });
    }
    Ok(expected)
}

fn validate_partial_fit_cols(n_features_in: &mut Option<usize>, cols: usize) -> OchreResult<()> {
    if let Some(expected) = *n_features_in {
        if cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: cols,
                expected,
            });
        }
    } else {
        *n_features_in = Some(cols);
    }
    Ok(())
}

/// Standardize features by removing the mean and scaling to unit variance (＾▽＾)
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub with_mean: bool,
    pub with_std: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub mean_: Option<Vec<f64>>,
    pub var_: Option<Vec<f64>>,
    pub scale_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub n_samples_seen_: usize,
    pub col_weights_: Vec<f64>,
}

impl StandardScaler {
    pub fn new(with_mean: bool, with_std: bool, device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            with_mean,
            with_std,
            device: dev,
            device_: None,
            mean_: None,
            var_: None,
            scale_: None,
            n_features_in_: None,
            n_samples_seen_: 0,
            col_weights_: Vec::new(),
        })
    }

    pub fn fit(&mut self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.mean_ = None;
        self.var_ = None;
        self.scale_ = None;
        self.n_features_in_ = None;
        self.n_samples_seen_ = 0;
        self.col_weights_.clear();
        self.partial_fit(x, sample_weight)
    }

    pub fn partial_fit(&mut self, x: &Matrix, sample_weight: Option<&[f64]>) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_partial_fit_cols(&mut self.n_features_in_, x.cols)?;
        self.device_ = Some(self.device.to_string());
        self.n_samples_seen_ += x.rows;

        if !self.with_mean && !self.with_std {
            return Ok(());
        }

        let (batch_mean, batch_var, batch_w): (Vec<f64>, Vec<f64>, Vec<f64>) =
            if slice_has_nan(&x.data) {
                let ns = compute_nan_col_stats(x, sample_weight)?;
                (ns.mean, ns.var, ns.count)
            } else {
                let stats = get_backend(&self.device).col_reduce_stats(x, sample_weight)?;
                let w = vec![stats.total_weight; x.cols];
                (stats.mean, stats.var, w)
            };

        if self.mean_.is_none() {
            self.col_weights_ = batch_w;
            self.mean_ = Some(batch_mean);
            if self.with_std {
                let scale: Vec<f64> = batch_var
                    .iter()
                    .map(|&v| handle_zeros_in_scale(v.sqrt()))
                    .collect();
                self.var_ = Some(batch_var);
                self.scale_ = Some(scale);
            }
            return Ok(());
        }

        let prev_mean = self.mean_.as_mut().ok_or(OchreError::NotFitted)?;
        let mut prev_var = self.var_.as_mut();
        for j in 0..x.cols {
            let wa = self.col_weights_[j];
            let wb = batch_w[j];
            if wb <= 0.0 {
                continue;
            }
            if wa <= 0.0 {
                prev_mean[j] = batch_mean[j];
                if let Some(ref mut pv) = prev_var {
                    pv[j] = batch_var[j];
                }
                self.col_weights_[j] = wb;
                continue;
            }
            let w_tot = wa + wb;
            let delta = batch_mean[j] - prev_mean[j];
            prev_mean[j] += delta * (wb / w_tot);
            if let Some(ref mut pv) = prev_var {
                let m2 = pv[j] * wa + batch_var[j] * wb + delta * delta * (wa * wb / w_tot);
                pv[j] = (m2 / w_tot).max(0.0);
            }
            self.col_weights_[j] = w_tot;
        }

        if let Some(ref var_vec) = self.var_ {
            self.scale_ = Some(
                var_vec
                    .iter()
                    .map(|&v| handle_zeros_in_scale(v.sqrt()))
                    .collect(),
            );
        }
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let shift = if self.with_mean {
            self.mean_.as_deref().ok_or(OchreError::NotFitted)?
        } else {
            &vec![0.0; cols]
        };
        let inv_scale: Vec<f64> = if self.with_std {
            self.scale_
                .as_ref()
                .ok_or(OchreError::NotFitted)?
                .iter()
                .map(|&s| 1.0 / s)
                .collect()
        } else {
            vec![1.0; cols]
        };
        get_backend(&self.device).col_scale_shift(x, &inv_scale, shift)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale: Vec<f64> = if self.with_std {
            self.scale_.clone().ok_or(OchreError::NotFitted)?
        } else {
            vec![1.0; cols]
        };
        let shift: Vec<f64> = if self.with_mean {
            let mean = self.mean_.as_ref().ok_or(OchreError::NotFitted)?;
            (0..cols).map(|j| -mean[j] / scale[j]).collect()
        } else {
            vec![0.0; cols]
        };
        get_backend(&self.device).col_scale_shift(x, &scale, &shift)
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
        let mut buf = write_header(TAG_STANDARD_SCALER);
        write_bool(&mut buf, self.with_mean);
        write_bool(&mut buf, self.with_std);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.mean_);
        write_opt_f64_vec(&mut buf, &self.var_);
        write_opt_f64_vec(&mut buf, &self.scale_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_seen_);
        write_f64_vec(&mut buf, &self.col_weights_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_STANDARD_SCALER)?;
        let with_mean = r.read_bool()?;
        let with_std = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let mean_ = r.read_opt_f64_vec()?;
        let var_ = r.read_opt_f64_vec()?;
        let scale_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_samples_seen_ = r.read_usize()?;
        let col_weights_ = r.read_f64_vec()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            with_mean,
            with_std,
            device: dev,
            device_,
            mean_,
            var_,
            scale_,
            n_features_in_,
            n_samples_seen_,
            col_weights_,
        })
    }
}

/// Transform features by scaling each feature to a given range (o´∀｀o)
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub feature_range: (f64, f64),
    pub clip: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub min_: Option<Vec<f64>>,
    pub scale_: Option<Vec<f64>>,
    pub data_min_: Option<Vec<f64>>,
    pub data_max_: Option<Vec<f64>>,
    pub data_range_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub n_samples_seen_: usize,
}

impl MinMaxScaler {
    pub fn new(feature_range: (f64, f64), clip: bool, device: Option<String>) -> OchreResult<Self> {
        if feature_range.0 >= feature_range.1
            || !feature_range.0.is_finite()
            || !feature_range.1.is_finite()
        {
            return Err(OchreError::InvalidParameter(format!(
                "Minimum of desired feature range must be smaller than maximum. Got ({}, {}) ( >_< )",
                feature_range.0, feature_range.1
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            feature_range,
            clip,
            device: dev,
            device_: None,
            min_: None,
            scale_: None,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            n_features_in_: None,
            n_samples_seen_: 0,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        self.min_ = None;
        self.scale_ = None;
        self.data_min_ = None;
        self.data_max_ = None;
        self.data_range_ = None;
        self.n_features_in_ = None;
        self.n_samples_seen_ = 0;
        self.partial_fit(x)
    }

    pub fn partial_fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_partial_fit_cols(&mut self.n_features_in_, x.cols)?;
        self.device_ = Some(self.device.to_string());
        self.n_samples_seen_ += x.rows;

        let (batch_min, batch_max) = if slice_has_nan(&x.data) {
            let ns = compute_nan_col_stats(x, None)?;
            (ns.min, ns.max)
        } else {
            let stats = get_backend(&self.device).col_reduce_stats(x, None)?;
            (stats.min, stats.max)
        };

        let (data_min, data_max) = match (self.data_min_.take(), self.data_max_.take()) {
            (Some(mut dmin), Some(mut dmax)) => {
                for j in 0..x.cols {
                    if !batch_min[j].is_nan() && (dmin[j].is_nan() || batch_min[j] < dmin[j]) {
                        dmin[j] = batch_min[j];
                    }
                    if !batch_max[j].is_nan() && (dmax[j].is_nan() || batch_max[j] > dmax[j]) {
                        dmax[j] = batch_max[j];
                    }
                }
                (dmin, dmax)
            }
            _ => (batch_min, batch_max),
        };

        let span = self.feature_range.1 - self.feature_range.0;
        let mut data_range = Vec::with_capacity(x.cols);
        let mut scale = Vec::with_capacity(x.cols);
        let mut min_offset = Vec::with_capacity(x.cols);
        for j in 0..x.cols {
            let r = data_max[j] - data_min[j];
            let s = span / handle_zeros_in_scale(r);
            data_range.push(r);
            scale.push(s);
            min_offset.push(self.feature_range.0 - data_min[j] * s);
        }

        self.data_min_ = Some(data_min);
        self.data_max_ = Some(data_max);
        self.data_range_ = Some(data_range);
        self.scale_ = Some(scale);
        self.min_ = Some(min_offset);
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale = self.scale_.as_ref().ok_or(OchreError::NotFitted)?;
        let min_offset = self.min_.as_ref().ok_or(OchreError::NotFitted)?;
        let shift: Vec<f64> = (0..cols).map(|j| -min_offset[j] / scale[j]).collect();
        let backend = get_backend(&self.device);
        let out = backend.col_scale_shift(x, scale, &shift)?;
        if self.clip {
            return backend.elementwise_op(
                &out,
                ElementwiseOp::Clip {
                    min: self.feature_range.0,
                    max: self.feature_range.1,
                },
            );
        }
        Ok(out)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale = self.scale_.as_ref().ok_or(OchreError::NotFitted)?;
        let min_offset = self.min_.as_ref().ok_or(OchreError::NotFitted)?;
        let inv_scale: Vec<f64> = scale.iter().map(|&s| 1.0 / s).collect();
        get_backend(&self.device).col_scale_shift(x, &inv_scale, min_offset)
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
        let mut buf = write_header(TAG_MINMAX_SCALER);
        write_f64(&mut buf, self.feature_range.0);
        write_f64(&mut buf, self.feature_range.1);
        write_bool(&mut buf, self.clip);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.min_);
        write_opt_f64_vec(&mut buf, &self.scale_);
        write_opt_f64_vec(&mut buf, &self.data_min_);
        write_opt_f64_vec(&mut buf, &self.data_max_);
        write_opt_f64_vec(&mut buf, &self.data_range_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_seen_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_MINMAX_SCALER)?;
        let fmin = r.read_f64()?;
        let fmax = r.read_f64()?;
        let clip = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let min_ = r.read_opt_f64_vec()?;
        let scale_ = r.read_opt_f64_vec()?;
        let data_min_ = r.read_opt_f64_vec()?;
        let data_max_ = r.read_opt_f64_vec()?;
        let data_range_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_samples_seen_ = r.read_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            feature_range: (fmin, fmax),
            clip,
            device: dev,
            device_,
            min_,
            scale_,
            data_min_,
            data_max_,
            data_range_,
            n_features_in_,
            n_samples_seen_,
        })
    }
}

/// Scale each feature by its maximum absolute value (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct MaxAbsScaler {
    pub device: DeviceType,
    pub device_: Option<String>,
    pub max_abs_: Option<Vec<f64>>,
    pub scale_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub n_samples_seen_: usize,
}

impl MaxAbsScaler {
    pub fn new(device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            device: dev,
            device_: None,
            max_abs_: None,
            scale_: None,
            n_features_in_: None,
            n_samples_seen_: 0,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        self.max_abs_ = None;
        self.scale_ = None;
        self.n_features_in_ = None;
        self.n_samples_seen_ = 0;
        self.partial_fit(x)
    }

    pub fn partial_fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        validate_partial_fit_cols(&mut self.n_features_in_, x.cols)?;
        self.device_ = Some(self.device.to_string());
        self.n_samples_seen_ += x.rows;

        let (bmin, bmax) = if slice_has_nan(&x.data) {
            let ns = compute_nan_col_stats(x, None)?;
            (ns.min, ns.max)
        } else {
            let stats = get_backend(&self.device).col_reduce_stats(x, None)?;
            (stats.min, stats.max)
        };

        let batch_abs: Vec<f64> = (0..x.cols)
            .map(|j| bmin[j].abs().max(bmax[j].abs()))
            .collect();
        let max_abs = match self.max_abs_.take() {
            Some(mut prev) => {
                for j in 0..x.cols {
                    if !batch_abs[j].is_nan() && (prev[j].is_nan() || batch_abs[j] > prev[j]) {
                        prev[j] = batch_abs[j];
                    }
                }
                prev
            }
            None => batch_abs,
        };
        let scale: Vec<f64> = max_abs.iter().map(|&v| handle_zeros_in_scale(v)).collect();
        self.max_abs_ = Some(max_abs);
        self.scale_ = Some(scale);
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale = self.scale_.as_ref().ok_or(OchreError::NotFitted)?;
        let inv_scale: Vec<f64> = scale.iter().map(|&s| 1.0 / s).collect();
        let zeros = vec![0.0; cols];
        get_backend(&self.device).col_scale_shift(x, &inv_scale, &zeros)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale = self.scale_.as_ref().ok_or(OchreError::NotFitted)?;
        let zeros = vec![0.0; cols];
        get_backend(&self.device).col_scale_shift(x, scale, &zeros)
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
        let mut buf = write_header(TAG_MAXABS_SCALER);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.max_abs_);
        write_opt_f64_vec(&mut buf, &self.scale_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_samples_seen_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_MAXABS_SCALER)?;
        let saved_device = r.read_opt_str()?;
        let max_abs_ = r.read_opt_f64_vec()?;
        let scale_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_samples_seen_ = r.read_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            device: dev,
            device_,
            max_abs_,
            scale_,
            n_features_in_,
            n_samples_seen_,
        })
    }
}

/// Scale features using statistics that are robust to outliers (＾▽＾)
#[derive(Debug, Clone)]
pub struct RobustScaler {
    pub with_centering: bool,
    pub with_scaling: bool,
    pub quantile_range: (f64, f64),
    pub unit_variance: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub center_: Option<Vec<f64>>,
    pub scale_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl RobustScaler {
    pub fn new(
        with_centering: bool,
        with_scaling: bool,
        quantile_range: (f64, f64),
        unit_variance: bool,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !(0.0..=100.0).contains(&quantile_range.0)
            || !(0.0..=100.0).contains(&quantile_range.1)
            || quantile_range.0 >= quantile_range.1
        {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid quantile range: ({}, {}) ( >_< )",
                quantile_range.0, quantile_range.1
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            with_centering,
            with_scaling,
            quantile_range,
            unit_variance,
            device: dev,
            device_: None,
            center_: None,
            scale_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let cols = x.cols;
        let rows = x.rows;
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());

        let (q_min, q_max) = self.quantile_range;
        let per_col: Vec<(f64, f64)> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut vals: Vec<f64> = (0..rows)
                    .map(|r| x.data[r * cols + j])
                    .filter(|v| !v.is_nan())
                    .collect();
                if vals.is_empty() {
                    return (f64::NAN, 1.0);
                }
                vals.sort_by(|a, b| a.total_cmp(b));
                let med = percentile_sorted(&vals, 50.0);
                let ql = percentile_sorted(&vals, q_min);
                let qh = percentile_sorted(&vals, q_max);
                (med, handle_zeros_in_scale(qh - ql))
            })
            .collect();

        if self.with_centering {
            self.center_ = Some(per_col.iter().map(|&(c, _)| c).collect());
        } else {
            self.center_ = None;
        }

        if self.with_scaling {
            let norm_iqr = if self.unit_variance {
                let adjust = ndtri(q_max / 100.0) - ndtri(q_min / 100.0);
                if adjust > 0.0 && adjust.is_finite() {
                    adjust
                } else {
                    1.0
                }
            } else {
                1.0
            };
            self.scale_ = Some(per_col.iter().map(|&(_, s)| s / norm_iqr).collect());
        } else {
            self.scale_ = None;
        }
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let shift = if self.with_centering {
            self.center_.as_deref().ok_or(OchreError::NotFitted)?
        } else {
            &vec![0.0; cols]
        };
        let inv_scale: Vec<f64> = if self.with_scaling {
            self.scale_
                .as_ref()
                .ok_or(OchreError::NotFitted)?
                .iter()
                .map(|&s| 1.0 / s)
                .collect()
        } else {
            vec![1.0; cols]
        };
        get_backend(&self.device).col_scale_shift(x, &inv_scale, shift)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_fitted_cols(self.n_features_in_, x.cols)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let scale: Vec<f64> = if self.with_scaling {
            self.scale_.clone().ok_or(OchreError::NotFitted)?
        } else {
            vec![1.0; cols]
        };
        let shift: Vec<f64> = if self.with_centering {
            let center = self.center_.as_ref().ok_or(OchreError::NotFitted)?;
            (0..cols).map(|j| -center[j] / scale[j]).collect()
        } else {
            vec![0.0; cols]
        };
        get_backend(&self.device).col_scale_shift(x, &scale, &shift)
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
        let mut buf = write_header(TAG_ROBUST_SCALER);
        write_bool(&mut buf, self.with_centering);
        write_bool(&mut buf, self.with_scaling);
        write_f64(&mut buf, self.quantile_range.0);
        write_f64(&mut buf, self.quantile_range.1);
        write_bool(&mut buf, self.unit_variance);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.center_);
        write_opt_f64_vec(&mut buf, &self.scale_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ROBUST_SCALER)?;
        let with_centering = r.read_bool()?;
        let with_scaling = r.read_bool()?;
        let q0 = r.read_f64()?;
        let q1 = r.read_f64()?;
        let unit_variance = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let center_ = r.read_opt_f64_vec()?;
        let scale_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            with_centering,
            with_scaling,
            quantile_range: (q0, q1),
            unit_variance,
            device: dev,
            device_,
            center_,
            scale_,
            n_features_in_,
        })
    }
}

/// Normalize samples individually to unit norm ('l1', 'l2', or 'max') (o´∀｀o)
#[derive(Debug, Clone)]
pub struct Normalizer {
    pub norm: String,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub n_features_in_: Option<usize>,
}

fn validate_norm_kind(norm: &str) -> OchreResult<()> {
    match norm {
        "l1" | "l2" | "max" => Ok(()),
        _ => Err(OchreError::InvalidParameter(format!(
            "Unsupported norm '{}'. Expected 'l1', 'l2', or 'max' ( >_< )",
            norm
        ))),
    }
}

/// Stateless row-wise (`axis=1`) or column-wise (`axis=0`) normalization returning `(normalized, norms)`.
pub fn normalize_matrix(x: &Matrix, norm: &str, axis: usize) -> OchreResult<(Matrix, Vec<f64>)> {
    validate_norm_kind(norm)?;
    if axis > 1 {
        return Err(OchreError::InvalidParameter(format!(
            "axis must be 0 or 1, got {} ( >_< )",
            axis
        )));
    }
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }

    if axis == 1 {
        let cols = x.cols;
        let mut out = vec![0.0; x.rows * cols];
        let mut norms = vec![0.0; x.rows];
        let process = |((dst, src), norm_out): ((&mut [f64], &[f64]), &mut f64)| {
            let n = match norm {
                "l1" => src.iter().map(|v| v.abs()).sum::<f64>(),
                "l2" => src.iter().map(|v| v * v).sum::<f64>().sqrt(),
                _ => src.iter().fold(0.0f64, |acc, v| acc.max(v.abs())),
            };
            *norm_out = n;
            if n == 0.0 {
                dst.copy_from_slice(src);
            } else {
                let inv = 1.0 / n;
                for j in 0..cols {
                    dst[j] = src[j] * inv;
                }
            }
        };

        if x.rows * cols < 16_384 {
            out.chunks_exact_mut(cols)
                .zip(x.data.chunks_exact(cols))
                .zip(norms.iter_mut())
                .for_each(process);
        } else {
            out.par_chunks_mut(cols)
                .zip(x.data.par_chunks(cols))
                .zip(norms.par_iter_mut())
                .for_each(process);
        }
        return Ok((Matrix::from_vec(x.rows, cols, out)?, norms));
    }

    // axis == 0: column-wise normalization
    let rows = x.rows;
    let cols = x.cols;
    let norms: Vec<f64> = (0..cols)
        .into_par_iter()
        .map(|j| match norm {
            "l1" => (0..rows).map(|r| x.data[r * cols + j].abs()).sum::<f64>(),
            "l2" => (0..rows)
                .map(|r| {
                    let v = x.data[r * cols + j];
                    v * v
                })
                .sum::<f64>()
                .sqrt(),
            _ => (0..rows).fold(0.0f64, |acc, r| acc.max(x.data[r * cols + j].abs())),
        })
        .collect();
    let inv_scale: Vec<f64> = norms
        .iter()
        .map(|&n| if n == 0.0 { 1.0 } else { 1.0 / n })
        .collect();
    let zeros = vec![0.0; cols];
    let out = crate::primitives::cpu_col_scale_shift(x, &inv_scale, &zeros)?;
    Ok((out, norms))
}

impl Normalizer {
    pub fn new(norm: String, device: Option<String>) -> OchreResult<Self> {
        validate_norm_kind(&norm)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            norm,
            device: dev,
            device_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        if let Some(expected) = self.n_features_in_
            && x.cols != expected
        {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let (out, _) = normalize_matrix(x, &self.norm, 1)?;
        Ok(out)
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
        let mut buf = write_header(TAG_NORMALIZER);
        write_str(&mut buf, &self.norm);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_NORMALIZER)?;
        let norm = r.read_str()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            norm,
            device: dev,
            device_,
            n_features_in_,
        })
    }
}
