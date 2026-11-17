// OchreML Non-Linear Distributions & Discretization Transforms (*≧ω≦*)
// Implements PowerTransformer (Yeo-Johnson & Box-Cox), QuantileTransformer,
// KBinsDiscretizer (uniform / quantile / 1D k-means), and Binarizer.

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_usize, write_str, write_usize,
    write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::math_utils::{
    averaged_inverted_cdf_sorted, brent_minimize, ndtr, ndtri, percentile_sorted,
};
use crate::preprocessing::scalers::StandardScaler;
use rayon::prelude::*;

pub const TAG_BINARIZER: u8 = 12;
pub const TAG_KBINS_DISCRETIZER: u8 = 15;
pub const TAG_POWER_TRANSFORMER: u8 = 16;
pub const TAG_QUANTILE_TRANSFORMER: u8 = 17;

const BOUNDS_THRESHOLD: f64 = 1e-7;

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

fn validate_cols(n_features_in: Option<usize>, cols: usize) -> OchreResult<usize> {
    let expected = n_features_in.ok_or(OchreError::NotFitted)?;
    if cols != expected {
        return Err(OchreError::FeatureCountMismatch {
            features: cols,
            expected,
        });
    }
    Ok(expected)
}

#[inline]
fn yeo_johnson_transform_scalar(x: f64, lmbda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        if lmbda.abs() < f64::EPSILON {
            x.ln_1p()
        } else {
            ((lmbda * x.ln_1p()).exp_m1()) / lmbda
        }
    } else if (lmbda - 2.0).abs() < f64::EPSILON {
        -(-x).ln_1p()
    } else {
        -(((2.0 - lmbda) * (-x).ln_1p()).exp_m1()) / (2.0 - lmbda)
    }
}

#[inline]
fn yeo_johnson_inverse_scalar(x: f64, lmbda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        if lmbda.abs() < f64::EPSILON {
            x.exp_m1()
        } else {
            ((lmbda * x).ln_1p() / lmbda).exp_m1()
        }
    } else if (lmbda - 2.0).abs() < f64::EPSILON {
        -(-x).exp_m1()
    } else {
        -(((-(2.0 - lmbda) * x).ln_1p()) / (2.0 - lmbda)).exp_m1()
    }
}

#[inline]
fn box_cox_transform_scalar(x: f64, lmbda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if lmbda.abs() < 1e-15 {
        x.ln()
    } else {
        (lmbda * x.ln()).exp_m1() / lmbda
    }
}

#[inline]
fn box_cox_inverse_scalar(x: f64, lmbda: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if lmbda.abs() < 1e-15 {
        x.exp()
    } else {
        ((lmbda * x).ln_1p() / lmbda).exp()
    }
}

fn yeo_johnson_neg_llf(vals: &[f64], lmbda: f64) -> f64 {
    let n = vals.len() as f64;
    let mut ys = Vec::with_capacity(vals.len());
    let mut sum = 0.0;
    let mut log_jacobian = 0.0;
    for &x in vals {
        let y = yeo_johnson_transform_scalar(x, lmbda);
        if !y.is_finite() {
            return f64::INFINITY;
        }
        ys.push(y);
        sum += y;
        let sgn_log = if x >= 0.0 { x.ln_1p() } else { -(-x).ln_1p() };
        log_jacobian += sgn_log;
    }
    let mean = sum / n;
    let var = (ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f64>() / n).max(1e-300);
    0.5 * n * var.ln() - (lmbda - 1.0) * log_jacobian
}

fn box_cox_neg_llf(vals: &[f64], lmbda: f64) -> f64 {
    let n = vals.len() as f64;
    let mut ys = Vec::with_capacity(vals.len());
    let mut sum = 0.0;
    let mut log_sum = 0.0;
    for &x in vals {
        let lx = x.ln();
        let y = if lmbda.abs() < 1e-15 {
            lx
        } else {
            (lmbda * lx).exp_m1() / lmbda
        };
        if !y.is_finite() {
            return f64::INFINITY;
        }
        ys.push(y);
        sum += y;
        log_sum += lx;
    }
    let mean = sum / n;
    let var = (ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f64>() / n).max(1e-300);
    0.5 * n * var.ln() - (lmbda - 1.0) * log_sum
}

/// Apply a power transform featurewise to make data more Gaussian-like (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct PowerTransformer {
    pub method: String,
    pub standardize: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub lambdas_: Option<Vec<f64>>,
    pub scaler_: Option<StandardScaler>,
    pub n_features_in_: Option<usize>,
}

impl PowerTransformer {
    pub fn new(method: String, standardize: bool, device: Option<String>) -> OchreResult<Self> {
        if method != "yeo-johnson" && method != "box-cox" {
            return Err(OchreError::InvalidParameter(format!(
                "PowerTransformer method must be 'yeo-johnson' or 'box-cox', got '{}' ( >_< )",
                method
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            method,
            standardize,
            device: dev,
            device_: None,
            lambdas_: None,
            scaler_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        let _ = self.fit_transform_internal(x, false)?;
        Ok(())
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit_transform_internal(x, true)?
            .ok_or(OchreError::EmptyData)
    }

    fn fit_transform_internal(
        &mut self,
        x: &Matrix,
        return_out: bool,
    ) -> OchreResult<Option<Matrix>> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let rows = x.rows;
        let cols = x.cols;
        let is_box_cox = self.method == "box-cox";

        let lambdas: Vec<f64> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut col_vals = Vec::with_capacity(rows);
                for r in 0..rows {
                    let v = x.data[r * cols + j];
                    if !v.is_nan() {
                        if is_box_cox && v <= 0.0 {
                            return Err(OchreError::InvalidParameter(
                                "The Box-Cox transformation can only be applied to strictly positive data ( >_< )".to_string(),
                            ));
                        }
                        col_vals.push(v);
                    }
                }
                if col_vals.is_empty() {
                    return Err(OchreError::InvalidParameter(
                        "Column must not be all NaN ( >_< )".to_string(),
                    ));
                }
                let first = col_vals[0];
                if col_vals.iter().all(|&v| v == first) {
                    return Ok(1.0);
                }
                let opt = if is_box_cox {
                    brent_minimize(|lam| box_cox_neg_llf(&col_vals, lam), -2.0, 2.0, 1e-11)
                } else {
                    brent_minimize(|lam| yeo_johnson_neg_llf(&col_vals, lam), -2.0, 2.0, 1e-11)
                };
                Ok(opt)
            })
            .collect::<OchreResult<Vec<f64>>>()?;

        self.lambdas_ = Some(lambdas.clone());
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());

        if !self.standardize && !return_out {
            self.scaler_ = None;
            return Ok(None);
        }

        let transformed = self.apply_power_only(x, &lambdas)?;
        if self.standardize {
            let mut scaler = StandardScaler::new(true, true, Some(self.device.to_string()))?;
            scaler.fit(&transformed, None)?;
            let out = if return_out {
                Some(scaler.transform(&transformed)?)
            } else {
                None
            };
            self.scaler_ = Some(scaler);
            Ok(out)
        } else {
            self.scaler_ = None;
            Ok(if return_out { Some(transformed) } else { None })
        }
    }

    fn apply_power_only(&self, x: &Matrix, lambdas: &[f64]) -> OchreResult<Matrix> {
        let cols = x.cols;
        let is_box_cox = self.method == "box-cox";
        let mut out = vec![0.0; x.rows * cols];
        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            for j in 0..cols {
                let v = src[j];
                if is_box_cox && !v.is_nan() && v <= 0.0 {
                    dst[j] = f64::NAN;
                } else {
                    dst[j] = if is_box_cox {
                        box_cox_transform_scalar(v, lambdas[j])
                    } else {
                        yeo_johnson_transform_scalar(v, lambdas[j])
                    };
                }
            }
        };
        if x.rows * cols < 16_384 {
            out.chunks_exact_mut(cols)
                .zip(x.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(cols)
                .zip(x.data.par_chunks(cols))
                .for_each(process_row);
        }
        Matrix::from_vec(x.rows, cols, out)
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let lambdas = self.lambdas_.as_ref().ok_or(OchreError::NotFitted)?;
        if self.method == "box-cox" && x.data.iter().any(|&v| !v.is_nan() && v <= 0.0) {
            return Err(OchreError::InvalidParameter(
                "The Box-Cox transformation can only be applied to strictly positive data ( >_< )"
                    .to_string(),
            ));
        }
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let transformed = self.apply_power_only(x, lambdas)?;
        if let Some(ref scaler) = self.scaler_ {
            scaler.transform(&transformed)
        } else {
            Ok(transformed)
        }
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let lambdas = self.lambdas_.as_ref().ok_or(OchreError::NotFitted)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let unscaled = if let Some(ref scaler) = self.scaler_ {
            scaler.inverse_transform(x)?
        } else {
            x.clone()
        };
        let is_box_cox = self.method == "box-cox";
        let mut out = vec![0.0; unscaled.rows * cols];
        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            for j in 0..cols {
                dst[j] = if is_box_cox {
                    box_cox_inverse_scalar(src[j], lambdas[j])
                } else {
                    yeo_johnson_inverse_scalar(src[j], lambdas[j])
                };
            }
        };
        if unscaled.rows * cols < 16_384 {
            out.chunks_exact_mut(cols)
                .zip(unscaled.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(cols)
                .zip(unscaled.data.par_chunks(cols))
                .for_each(process_row);
        }
        Matrix::from_vec(unscaled.rows, cols, out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        if let Some(ref mut sc) = self.scaler_ {
            sc.set_device(Some(dev.to_string()))?;
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_POWER_TRANSFORMER);
        write_str(&mut buf, &self.method);
        write_bool(&mut buf, self.standardize);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.lambdas_);
        match &self.scaler_ {
            Some(sc) => {
                write_bool(&mut buf, true);
                let sb = sc.to_bytes();
                write_usize(&mut buf, sb.len());
                buf.extend_from_slice(&sb);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_POWER_TRANSFORMER)?;
        let method = r.read_str()?;
        let standardize = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let lambdas_ = r.read_opt_f64_vec()?;
        let has_scaler = r.read_bool()?;
        let scaler_ = if has_scaler {
            let slen = r.read_usize()?;
            let sbytes = r.read_exact(slen)?;
            Some(StandardScaler::from_bytes(sbytes, device)?)
        } else {
            None
        };
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            method,
            standardize,
            device: dev,
            device_,
            lambdas_,
            scaler_,
            n_features_in_,
        })
    }
}

/// Piecewise linear interpolation matching `np.interp(x, xp, fp)` where `xp` is non-decreasing.
#[inline]
fn interp_1d(val: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if val <= xp[0] {
        return fp[0];
    }
    if val >= xp[n - 1] {
        return fp[n - 1];
    }
    // Rightmost index `idx` such that `xp[idx] <= val`
    let idx = xp.partition_point(|&p| p <= val).saturating_sub(1);
    let x0 = xp[idx];
    let x1 = xp[idx + 1];
    if x1 == x0 {
        fp[idx]
    } else {
        fp[idx] + (val - x0) * (fp[idx + 1] - fp[idx]) / (x1 - x0)
    }
}

/// Piecewise linear interpolation matching `-np.interp(-val, -xp[::-1], -fp[::-1])`.
#[inline]
fn interp_1d_rev(val: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if val >= xp[n - 1] {
        return fp[n - 1];
    }
    if val <= xp[0] {
        return fp[0];
    }
    // Equivalent to finding leftmost `k` with `xp[k] >= val`
    let k = xp.partition_point(|&p| p < val);
    if xp[k] == val {
        return fp[k];
    }
    let idx = k - 1;
    let x0 = xp[idx];
    let x1 = xp[idx + 1];
    if x1 == x0 {
        fp[idx + 1]
    } else {
        fp[idx + 1] - (x1 - val) * (fp[idx + 1] - fp[idx]) / (x1 - x0)
    }
}

/// Transform features using quantiles information (＾▽＾)
#[derive(Debug, Clone)]
pub struct QuantileTransformer {
    pub n_quantiles: usize,
    pub output_distribution: String,
    pub ignore_implicit_zeros: bool,
    pub subsample: usize,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub n_quantiles_: Option<usize>,
    pub quantiles_: Option<Matrix>,
    pub references_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl QuantileTransformer {
    pub fn new(
        n_quantiles: usize,
        output_distribution: String,
        ignore_implicit_zeros: bool,
        subsample: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_quantiles == 0 {
            return Err(OchreError::InvalidParameter(
                "n_quantiles must be >= 1 ( >_< )".to_string(),
            ));
        }
        if output_distribution != "uniform" && output_distribution != "normal" {
            return Err(OchreError::InvalidParameter(format!(
                "output_distribution must be 'uniform' or 'normal', got '{}' ( >_< )",
                output_distribution
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_quantiles,
            output_distribution,
            ignore_implicit_zeros,
            subsample,
            random_state,
            device: dev,
            device_: None,
            n_quantiles_: None,
            quantiles_: None,
            references_: None,
            n_features_in_: None,
        })
    }

    fn subsample_rows(&self, n_rows: usize) -> Vec<usize> {
        if n_rows <= self.subsample {
            return (0..n_rows).collect();
        }
        let mut seed = self.random_state.unwrap_or(0) ^ 0x9E37_79B9_7F4A_7C15;
        let mut indices: Vec<usize> = (0..n_rows).collect();
        // Partial Fisher-Yates shuffle for `self.subsample` indices
        for i in 0..self.subsample {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let j = i + (z as usize) % (n_rows - i);
            indices.swap(i, j);
        }
        indices.truncate(self.subsample);
        indices
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let row_idx = self.subsample_rows(x.rows);
        let n_q = self.n_quantiles.min(row_idx.len());
        let references: Vec<f64> = if n_q == 1 {
            vec![0.5]
        } else {
            (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect()
        };

        let cols = x.cols;
        let per_col: Vec<Vec<f64>> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut vals: Vec<f64> = row_idx
                    .iter()
                    .map(|&r| x.data[r * cols + j])
                    .filter(|v| !v.is_nan())
                    .collect();
                if vals.is_empty() {
                    return vec![f64::NAN; n_q];
                }
                vals.sort_by(|a, b| a.total_cmp(b));
                references
                    .iter()
                    .map(|&q| percentile_sorted(&vals, q * 100.0))
                    .collect()
            })
            .collect();

        // Store quantiles_ as (n_quantiles_, n_features)
        let mut q_mat = vec![0.0; n_q * cols];
        for j in 0..cols {
            for i in 0..n_q {
                q_mat[i * cols + j] = per_col[j][i];
            }
        }
        self.n_quantiles_ = Some(n_q);
        self.references_ = Some(references);
        self.quantiles_ = Some(Matrix::from_vec(n_q, cols, q_mat)?);
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.transform_impl(x, false)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.transform_impl(x, true)
    }

    fn transform_impl(&self, x: &Matrix, inverse: bool) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let q_mat = self.quantiles_.as_ref().ok_or(OchreError::NotFitted)?;
        let refs = self.references_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_q = q_mat.rows;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let is_normal = self.output_distribution == "normal";
        let clip_min = ndtri(BOUNDS_THRESHOLD - f64::EPSILON);
        let clip_max = ndtri(1.0 - (BOUNDS_THRESHOLD - f64::EPSILON));

        let col_q: Vec<Vec<f64>> = (0..cols)
            .map(|j| (0..n_q).map(|i| q_mat.data[i * cols + j]).collect())
            .collect();

        let mut out = vec![0.0; x.rows * cols];
        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            for j in 0..cols {
                let v = src[j];
                if v.is_nan() {
                    dst[j] = f64::NAN;
                    continue;
                }
                let qcol = &col_q[j];
                if !inverse {
                    if is_normal {
                        if v + BOUNDS_THRESHOLD > qcol[n_q - 1] {
                            dst[j] = clip_max;
                        } else if v - BOUNDS_THRESHOLD < qcol[0] {
                            dst[j] = clip_min;
                        } else {
                            let u = 0.5 * (interp_1d(v, qcol, refs) + interp_1d_rev(v, qcol, refs));
                            dst[j] = ndtri(u).clamp(clip_min, clip_max);
                        }
                    } else if v >= qcol[n_q - 1] {
                        dst[j] = 1.0;
                    } else if v <= qcol[0] {
                        dst[j] = 0.0;
                    } else {
                        dst[j] = 0.5 * (interp_1d(v, qcol, refs) + interp_1d_rev(v, qcol, refs));
                    }
                } else if is_normal {
                    let u_cdf = ndtr(v);
                    if u_cdf - BOUNDS_THRESHOLD < 0.0 {
                        dst[j] = qcol[0];
                    } else if u_cdf + BOUNDS_THRESHOLD > 1.0 {
                        dst[j] = qcol[n_q - 1];
                    } else {
                        dst[j] = interp_1d(u_cdf, refs, qcol);
                    }
                } else {
                    let u = v.clamp(0.0, 1.0);
                    dst[j] = interp_1d(u, refs, qcol);
                }
            }
        };

        if x.rows * cols < 16_384 {
            out.chunks_exact_mut(cols)
                .zip(x.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(cols)
                .zip(x.data.par_chunks(cols))
                .for_each(process_row);
        }
        Matrix::from_vec(x.rows, cols, out)
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
        let mut buf = write_header(TAG_QUANTILE_TRANSFORMER);
        write_usize(&mut buf, self.n_quantiles);
        write_str(&mut buf, &self.output_distribution);
        write_bool(&mut buf, self.ignore_implicit_zeros);
        write_usize(&mut buf, self.subsample);
        write_opt_usize(&mut buf, self.random_state.map(|v| v as usize));
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_quantiles_);
        write_opt_matrix(&mut buf, &self.quantiles_);
        write_opt_f64_vec(&mut buf, &self.references_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_QUANTILE_TRANSFORMER)?;
        let n_quantiles = r.read_usize()?;
        let output_distribution = r.read_str()?;
        let ignore_implicit_zeros = r.read_bool()?;
        let subsample = r.read_usize()?;
        let random_state = r.read_opt_usize()?.map(|v| v as u64);
        let saved_device = r.read_opt_str()?;
        let n_quantiles_ = r.read_opt_usize()?;
        let quantiles_ = r.read_opt_matrix()?;
        let references_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_quantiles,
            output_distribution,
            ignore_implicit_zeros,
            subsample,
            random_state,
            device: dev,
            device_,
            n_quantiles_,
            quantiles_,
            references_,
            n_features_in_,
        })
    }
}

/// Bin continuous data into intervals (`uniform`, `quantile`, or `kmeans`) (o´∀｀o)
#[derive(Debug, Clone)]
pub struct KBinsDiscretizer {
    pub n_bins_param: Vec<usize>,
    pub encode: String,
    pub strategy: String,
    pub subsample: Option<usize>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub bin_edges_: Option<Vec<Vec<f64>>>,
    pub n_bins_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
}

fn fit_1d_kmeans_edges(col: &[f64], k: usize) -> Vec<f64> {
    let mut col_min = f64::INFINITY;
    let mut col_max = f64::NEG_INFINITY;
    for &v in col {
        if v < col_min {
            col_min = v;
        }
        if v > col_max {
            col_max = v;
        }
    }
    if col_min == col_max || k <= 1 {
        return vec![col_min, col_max];
    }

    let step = (col_max - col_min) / k as f64;
    let mut centers: Vec<f64> = (0..k).map(|b| col_min + (b as f64 + 0.5) * step).collect();
    let mut sums = vec![0.0; k];
    let mut counts = vec![0usize; k];

    for _ in 0..300 {
        sums.fill(0.0);
        counts.fill(0);
        for &v in col {
            let mut best_c = 0;
            let mut best_d = (v - centers[0]).abs();
            for (c_idx, &c_val) in centers.iter().enumerate().skip(1) {
                let d = (v - c_val).abs();
                if d < best_d {
                    best_d = d;
                    best_c = c_idx;
                }
            }
            sums[best_c] += v;
            counts[best_c] += 1;
        }
        let mut max_shift = 0.0f64;
        for c_idx in 0..k {
            if counts[c_idx] > 0 {
                let new_c = sums[c_idx] / counts[c_idx] as f64;
                max_shift = max_shift.max((new_c - centers[c_idx]).abs());
                centers[c_idx] = new_c;
            }
        }
        if max_shift < 1e-8 * (col_max - col_min).max(1.0) {
            break;
        }
    }

    centers.sort_by(|a, b| a.total_cmp(b));
    let mut edges = Vec::with_capacity(k + 1);
    edges.push(col_min);
    for i in 0..(k - 1) {
        edges.push(0.5 * (centers[i] + centers[i + 1]));
    }
    edges.push(col_max);
    edges
}

impl KBinsDiscretizer {
    pub fn new(
        n_bins_param: Vec<usize>,
        encode: String,
        strategy: String,
        subsample: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_bins_param.is_empty() || n_bins_param.iter().any(|&b| b < 2) {
            return Err(OchreError::InvalidParameter(
                "n_bins must be >= 2 for all features ( >_< )".to_string(),
            ));
        }
        if encode != "onehot" && encode != "onehot-dense" && encode != "ordinal" {
            return Err(OchreError::InvalidParameter(format!(
                "encode must be 'onehot', 'onehot-dense', or 'ordinal', got '{}' ( >_< )",
                encode
            )));
        }
        if strategy != "uniform" && strategy != "quantile" && strategy != "kmeans" {
            return Err(OchreError::InvalidParameter(format!(
                "strategy must be 'uniform', 'quantile', or 'kmeans', got '{}' ( >_< )",
                strategy
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_bins_param,
            encode,
            strategy,
            subsample,
            device: dev,
            device_: None,
            bin_edges_: None,
            n_bins_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let cols = x.cols;
        let rows = x.rows;
        let bins_per_col: Vec<usize> = if self.n_bins_param.len() == 1 {
            vec![self.n_bins_param[0]; cols]
        } else if self.n_bins_param.len() == cols {
            self.n_bins_param.clone()
        } else {
            return Err(OchreError::FeatureCountMismatch {
                features: self.n_bins_param.len(),
                expected: cols,
            });
        };

        let stats = get_backend(&self.device).col_reduce_stats(x, None)?;
        let strategy = self.strategy.as_str();
        let max_rows = self.subsample.unwrap_or(rows).min(rows);

        let edges_and_bins: Vec<(Vec<f64>, usize)> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let k = bins_per_col[j];
                let col_min = stats.min[j];
                let col_max = stats.max[j];
                if col_min == col_max {
                    return (vec![col_min - 0.5, col_max + 0.5], 1);
                }
                let raw_edges = match strategy {
                    "uniform" => {
                        let step = (col_max - col_min) / k as f64;
                        (0..=k)
                            .map(|b| {
                                if b == k {
                                    col_max
                                } else {
                                    col_min + b as f64 * step
                                }
                            })
                            .collect::<Vec<f64>>()
                    }
                    "quantile" => {
                        let mut col_vals: Vec<f64> =
                            (0..max_rows).map(|r| x.data[r * cols + j]).collect();
                        col_vals.sort_by(|a, b| a.total_cmp(b));
                        (0..=k)
                            .map(|b| averaged_inverted_cdf_sorted(&col_vals, b as f64 / k as f64))
                            .collect::<Vec<f64>>()
                    }
                    _ => {
                        let col_vals: Vec<f64> =
                            (0..max_rows).map(|r| x.data[r * cols + j]).collect();
                        fit_1d_kmeans_edges(&col_vals, k)
                    }
                };
                // Deduplicate tiny flat bin edges matching sklearn
                let mut clean_edges: Vec<f64> = Vec::with_capacity(raw_edges.len());
                for e in raw_edges {
                    if let Some(&last) = clean_edges.last()
                        && (e - last).abs() <= 1e-8
                    {
                        continue;
                    }
                    clean_edges.push(e);
                }
                if clean_edges.len() < 2 {
                    clean_edges = vec![col_min - 0.5, col_max + 0.5];
                }
                let n_b = clean_edges.len() - 1;
                (clean_edges, n_b)
            })
            .collect();

        let mut bin_edges = Vec::with_capacity(cols);
        let mut n_bins = Vec::with_capacity(cols);
        for (e, nb) in edges_and_bins {
            bin_edges.push(e);
            n_bins.push(nb);
        }

        self.bin_edges_ = Some(bin_edges);
        self.n_bins_ = Some(n_bins);
        self.n_features_in_ = Some(cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform_ordinal(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let edges = self.bin_edges_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_bins = self.n_bins_.as_ref().ok_or(OchreError::NotFitted)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, cols, Vec::new());
        }
        let mut out = vec![0.0; x.rows * cols];
        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            for j in 0..cols {
                let e = &edges[j];
                let nb = n_bins[j];
                let v = src[j];
                // np.digitize right=False with interior edges `e[1..nb]`, clipped to `0..nb-1`
                let bin = e[1..nb].partition_point(|&edge| edge <= v);
                dst[j] = bin.min(nb - 1) as f64;
            }
        };
        if x.rows * cols < 16_384 {
            out.chunks_exact_mut(cols)
                .zip(x.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(cols)
                .zip(x.data.par_chunks(cols))
                .for_each(process_row);
        }
        Matrix::from_vec(x.rows, cols, out)
    }

    pub fn transform_dense(&self, x: &Matrix) -> OchreResult<Matrix> {
        if self.encode == "ordinal" {
            return self.transform_ordinal(x);
        }
        let ord = self.transform_ordinal(x)?;
        let n_bins = self.n_bins_.as_ref().ok_or(OchreError::NotFitted)?;
        let total_cols: usize = n_bins.iter().sum();
        let mut offsets = Vec::with_capacity(n_bins.len());
        let mut acc = 0usize;
        for &nb in n_bins {
            offsets.push(acc);
            acc += nb;
        }
        let mut out = vec![0.0; x.rows * total_cols];
        for r in 0..x.rows {
            for j in 0..x.cols {
                let bin = ord.data[r * x.cols + j] as usize;
                out[r * total_cols + offsets[j] + bin] = 1.0;
            }
        }
        Matrix::from_vec(x.rows, total_cols, out)
    }

    pub fn inverse_transform_from_ordinal(&self, x_ord: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x_ord.cols)?;
        let edges = self.bin_edges_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_bins = self.n_bins_.as_ref().ok_or(OchreError::NotFitted)?;
        let mut out = vec![0.0; x_ord.rows * cols];
        for r in 0..x_ord.rows {
            for j in 0..cols {
                let b = (x_ord.data[r * cols + j].round() as isize)
                    .clamp(0, (n_bins[j] - 1) as isize) as usize;
                out[r * cols + j] = 0.5 * (edges[j][b] + edges[j][b + 1]);
            }
        }
        Matrix::from_vec(x_ord.rows, cols, out)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        if self.encode == "ordinal" {
            return self.inverse_transform_from_ordinal(x);
        }
        let n_bins = self.n_bins_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_cols: usize = n_bins.iter().sum();
        if x.cols != expected_cols {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }
        let edges = self.bin_edges_.as_ref().ok_or(OchreError::NotFitted)?;
        let cols = n_bins.len();
        let mut out = vec![0.0; x.rows * cols];
        for r in 0..x.rows {
            let mut offset = 0usize;
            for j in 0..cols {
                let nb = n_bins[j];
                let slice = &x.data[r * expected_cols + offset..r * expected_cols + offset + nb];
                let mut best_b = 0;
                let mut best_v = slice[0];
                for (b, &v) in slice.iter().enumerate().skip(1) {
                    if v > best_v {
                        best_v = v;
                        best_b = b;
                    }
                }
                out[r * cols + j] = 0.5 * (edges[j][best_b] + edges[j][best_b + 1]);
                offset += nb;
            }
        }
        Matrix::from_vec(x.rows, cols, out)
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
        let mut buf = write_header(TAG_KBINS_DISCRETIZER);
        write_usize_vec(&mut buf, &self.n_bins_param);
        write_str(&mut buf, &self.encode);
        write_str(&mut buf, &self.strategy);
        write_opt_usize(&mut buf, self.subsample);
        write_opt_str(&mut buf, &self.device_);
        match &self.bin_edges_ {
            Some(edges) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, edges.len());
                for e in edges {
                    write_f64_vec(&mut buf, e);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.n_bins_ {
            Some(nb) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, nb);
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_KBINS_DISCRETIZER)?;
        let n_bins_param = r.read_usize_vec()?;
        let encode = r.read_str()?;
        let strategy = r.read_str()?;
        let subsample = r.read_opt_usize()?;
        let saved_device = r.read_opt_str()?;
        let bin_edges_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut edges = Vec::with_capacity(len);
            for _ in 0..len {
                edges.push(r.read_f64_vec()?);
            }
            Some(edges)
        } else {
            None
        };
        let n_bins_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_bins_param,
            encode,
            strategy,
            subsample,
            device: dev,
            device_,
            bin_edges_,
            n_bins_,
            n_features_in_,
        })
    }
}

/// Binarize data (set feature values to 0 or 1) according to a threshold (＾▽＾)
#[derive(Debug, Clone)]
pub struct Binarizer {
    pub threshold: f64,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub n_features_in_: Option<usize>,
}

impl Binarizer {
    pub fn new(threshold: f64, device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            threshold,
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
        binarize_matrix(x, self.threshold)
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
        let mut buf = write_header(TAG_BINARIZER);
        write_f64(&mut buf, self.threshold);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_BINARIZER)?;
        let threshold = r.read_f64()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            threshold,
            device: dev,
            device_,
            n_features_in_,
        })
    }
}

pub fn binarize_matrix(x: &Matrix, threshold: f64) -> OchreResult<Matrix> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| v.is_nan()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN values which Binarizer cannot threshold ( >_< )".to_string(),
        ));
    }
    let mut out = vec![0.0; x.data.len()];
    if out.len() < 16_384 {
        for (dst, &src) in out.iter_mut().zip(x.data.iter()) {
            *dst = if src > threshold { 1.0 } else { 0.0 };
        }
    } else {
        out.par_chunks_mut(4096)
            .zip(x.data.par_chunks(4096))
            .for_each(|(dst, src)| {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = if s > threshold { 1.0 } else { 0.0 };
                }
            });
    }
    Matrix::from_vec(x.rows, x.cols, out)
}
