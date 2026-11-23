// OchreML Feature Selection Module
// Implements f_regression, f_classif, chi2 (with pure-Rust incomplete beta/gamma survival functions)
// and VarianceThreshold (TAG 96), SelectKBest (TAG 97), SelectPercentile (TAG 98).

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64_vec,
    write_opt_str, write_opt_usize, write_str,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
    usize_slice_to_i64_pybytes,
};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::HashMap;

pub const TAG_VARIANCE_THRESHOLD: u8 = 96;
pub const TAG_SELECT_K_BEST: u8 = 97;
pub const TAG_SELECT_PERCENTILE: u8 = 98;

// ============================================================================
// Pure-Rust Special Functions (Lanczos ln_gamma, Incomplete Beta, Incomplete Gamma)
// ============================================================================

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        let sin_pi_z = (std::f64::consts::PI * z).sin();
        return std::f64::consts::PI.ln() - sin_pi_z.abs().ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + (i as f64));
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 300;
    const EPS: f64 = 3.0e-15;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_IT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    h
}

pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * betacf(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * betacf(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

pub fn f_survival(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat.is_nan() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f_stat <= 0.0 {
        return 1.0;
    }
    if f_stat.is_infinite() {
        return 0.0;
    }
    let x = df2 / (df2 + df1 * f_stat);
    regularized_incomplete_beta(x, 0.5 * df2, 0.5 * df1)
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    const MAX_IT: usize = 500;
    const EPS: f64 = 3.0e-15;

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_IT {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    let ln_prefactor = -x + a * x.ln() - ln_gamma(a);
    (sum * ln_prefactor.exp()).clamp(0.0, 1.0)
}

fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const MAX_IT: usize = 500;
    const EPS: f64 = 3.0e-15;
    const FPMIN: f64 = 1.0e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_IT {
        let fi = i as f64;
        let an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    let ln_prefactor = -x + a * x.ln() - ln_gamma(a);
    (ln_prefactor.exp() * h).clamp(0.0, 1.0)
}

pub fn regularized_upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        (1.0 - gamma_p_series(a, x)).clamp(0.0, 1.0)
    } else {
        gamma_q_cf(a, x)
    }
}

pub fn chi2_survival(chi2_stat: f64, df: f64) -> f64 {
    if chi2_stat.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if chi2_stat <= 0.0 {
        return 1.0;
    }
    if chi2_stat.is_infinite() {
        return 0.0;
    }
    regularized_upper_incomplete_gamma(0.5 * df, 0.5 * chi2_stat)
}

// ============================================================================
// Statistical Feature Scoring Tests
// ============================================================================

fn validate_xy(x: &Matrix, y_len: usize) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != y_len {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y_len,
        });
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf".to_string(),
        ));
    }
    Ok(())
}

pub fn f_regression(
    x: &Matrix,
    y: &[f64],
    center: bool,
    force_finite: bool,
) -> OchreResult<(Vec<f64>, Vec<f64>)> {
    validate_xy(x, y.len())?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Target y contains NaN or Inf".to_string(),
        ));
    }

    let n = x.rows;
    let p = x.cols;
    let min_samples = if center { 3 } else { 2 };
    if n < min_samples {
        return Err(OchreError::InvalidParameter(format!(
            "f_regression requires at least {} samples when center={}, got {}",
            min_samples, center, n
        )));
    }

    let dof = (if center { n - 2 } else { n - 1 }) as f64;
    let nf = n as f64;

    let y_mean = if center {
        y.iter().sum::<f64>() / nf
    } else {
        0.0
    };
    let y_norm_sq: f64 = y
        .iter()
        .map(|&yi| {
            let dy = yi - y_mean;
            dy * dy
        })
        .sum();
    let y_norm = y_norm_sq.sqrt();

    let results: Vec<(f64, f64)> = (0..p)
        .into_par_iter()
        .map(|j| {
            if y_norm == 0.0 {
                return if force_finite {
                    (0.0, 1.0)
                } else {
                    (f64::NAN, f64::NAN)
                };
            }

            let x_mean = if center {
                let mut sum = 0.0;
                for r in 0..n {
                    sum += x.data[r * p + j];
                }
                sum / nf
            } else {
                0.0
            };

            let mut dot = 0.0;
            let mut x_norm_sq = 0.0;
            for (r, &yv) in y.iter().enumerate().take(n) {
                let dx = x.data[r * p + j] - x_mean;
                let dy = yv - y_mean;
                dot += dx * dy;
                x_norm_sq += dx * dx;
            }
            let x_norm = x_norm_sq.sqrt();
            if x_norm == 0.0 {
                return if force_finite {
                    (0.0, 1.0)
                } else {
                    (f64::NAN, f64::NAN)
                };
            }

            let r = (dot / (x_norm * y_norm)).clamp(-1.0, 1.0);
            let r2 = r * r;
            if r2 >= 1.0 {
                return if force_finite {
                    (f64::MAX, 0.0)
                } else {
                    (f64::INFINITY, 0.0)
                };
            }
            let f_val = (r2 / (1.0 - r2)) * dof;
            let p_val = f_survival(f_val, 1.0, dof);
            (f_val, p_val)
        })
        .collect();

    let mut scores = Vec::with_capacity(p);
    let mut pvalues = Vec::with_capacity(p);
    for (f_val, p_val) in results {
        scores.push(f_val);
        pvalues.push(p_val);
    }
    Ok((scores, pvalues))
}

pub fn f_classif(x: &Matrix, y: &[i64]) -> OchreResult<(Vec<f64>, Vec<f64>)> {
    validate_xy(x, y.len())?;
    let n = x.rows;
    let p = x.cols;

    let mut class_map: HashMap<i64, usize> = HashMap::new();
    let mut class_indices: Vec<Vec<usize>> = Vec::new();
    for (i, &lbl) in y.iter().enumerate() {
        let cid = match class_map.get(&lbl) {
            Some(&id) => id,
            None => {
                let id = class_indices.len();
                class_map.insert(lbl, id);
                class_indices.push(Vec::new());
                id
            }
        };
        class_indices[cid].push(i);
    }

    let k = class_indices.len();
    if k < 2 {
        return Err(OchreError::InsufficientClasses(k));
    }
    if n <= k {
        return Err(OchreError::InvalidParameter(format!(
            "f_classif requires n_samples ({}) > n_classes ({})",
            n, k
        )));
    }

    let df_between = (k - 1) as f64;
    let df_within = (n - k) as f64;
    let nf = n as f64;

    let results: Vec<(f64, f64)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let mut total_sum = 0.0;
            for r in 0..n {
                total_sum += x.data[r * p + j];
            }
            let grand_mean = total_sum / nf;

            let mut ss_between = 0.0;
            let mut ss_within = 0.0;
            for indices in &class_indices {
                let nk = indices.len() as f64;
                let mut c_sum = 0.0;
                for &r in indices {
                    c_sum += x.data[r * p + j];
                }
                let c_mean = c_sum / nk;
                let diff_mean = c_mean - grand_mean;
                ss_between += nk * diff_mean * diff_mean;
                for &r in indices {
                    let d = x.data[r * p + j] - c_mean;
                    ss_within += d * d;
                }
            }

            let ms_between = ss_between / df_between;
            let ms_within = ss_within / df_within;
            if ms_within == 0.0 {
                if ms_between == 0.0 {
                    (f64::NAN, f64::NAN)
                } else {
                    (f64::INFINITY, 0.0)
                }
            } else {
                let f_val = ms_between / ms_within;
                let p_val = f_survival(f_val, df_between, df_within);
                (f_val, p_val)
            }
        })
        .collect();

    let mut scores = Vec::with_capacity(p);
    let mut pvalues = Vec::with_capacity(p);
    for (f_val, p_val) in results {
        scores.push(f_val);
        pvalues.push(p_val);
    }
    Ok((scores, pvalues))
}

pub fn chi2(x: &Matrix, y: &[i64]) -> OchreResult<(Vec<f64>, Vec<f64>)> {
    validate_xy(x, y.len())?;
    if x.data.iter().any(|&v| v < 0.0) {
        return Err(OchreError::InvalidParameter(
            "Input X must be non-negative for chi2".to_string(),
        ));
    }

    let n = x.rows;
    let p = x.cols;

    let mut class_map: HashMap<i64, usize> = HashMap::new();
    let mut class_indices: Vec<Vec<usize>> = Vec::new();
    for (i, &lbl) in y.iter().enumerate() {
        let cid = match class_map.get(&lbl) {
            Some(&id) => id,
            None => {
                let id = class_indices.len();
                class_map.insert(lbl, id);
                class_indices.push(Vec::new());
                id
            }
        };
        class_indices[cid].push(i);
    }

    let k = class_indices.len();
    if k < 2 {
        return Err(OchreError::InsufficientClasses(k));
    }

    let df = (k - 1) as f64;
    let nf = n as f64;

    let results: Vec<(f64, f64)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let mut feature_sum = 0.0;
            for r in 0..n {
                feature_sum += x.data[r * p + j];
            }
            if feature_sum == 0.0 {
                return (f64::NAN, f64::NAN);
            }

            let mut chi2_val = 0.0;
            for indices in &class_indices {
                let class_prob = (indices.len() as f64) / nf;
                let expected = class_prob * feature_sum;
                let mut observed = 0.0;
                for &r in indices {
                    observed += x.data[r * p + j];
                }
                if expected > 0.0 {
                    let diff = observed - expected;
                    chi2_val += (diff * diff) / expected;
                }
            }
            let p_val = chi2_survival(chi2_val, df);
            (chi2_val, p_val)
        })
        .collect();

    let mut scores = Vec::with_capacity(p);
    let mut pvalues = Vec::with_capacity(p);
    for (s, pv) in results {
        scores.push(s);
        pvalues.push(pv);
    }
    Ok((scores, pvalues))
}

// ============================================================================
// Shared Column Selection / Inverse Helpers
// ============================================================================

fn select_columns_by_mask(x: &Matrix, mask: &[bool]) -> OchreResult<Matrix> {
    if x.cols != mask.len() {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: mask.len(),
        });
    }
    let selected_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(j, &keep)| if keep { Some(j) } else { None })
        .collect();

    if selected_indices.is_empty() {
        return Err(OchreError::InvalidParameter(
            "No features were selected (all support mask entries are false)".to_string(),
        ));
    }

    let n = x.rows;
    let p_in = x.cols;
    let p_out = selected_indices.len();
    let mut out = vec![0.0; n * p_out];
    out.par_chunks_mut(p_out)
        .enumerate()
        .for_each(|(r, dst_row)| {
            let src_row = &x.data[r * p_in..(r + 1) * p_in];
            for (dst_c, &src_c) in selected_indices.iter().enumerate() {
                dst_row[dst_c] = src_row[src_c];
            }
        });
    Matrix::from_vec(n, p_out, out)
}

fn inverse_select_columns_by_mask(x: &Matrix, mask: &[bool]) -> OchreResult<Matrix> {
    let selected_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(j, &keep)| if keep { Some(j) } else { None })
        .collect();
    if x.cols != selected_indices.len() {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: selected_indices.len(),
        });
    }
    let n = x.rows;
    let p_in = x.cols;
    let p_out = mask.len();
    let mut out = vec![0.0; n * p_out];
    out.par_chunks_mut(p_out)
        .enumerate()
        .for_each(|(r, dst_row)| {
            let src_row = &x.data[r * p_in..(r + 1) * p_in];
            for (src_c, &dst_c) in selected_indices.iter().enumerate() {
                dst_row[dst_c] = src_row[src_c];
            }
        });
    Matrix::from_vec(n, p_out, out)
}

fn mask_to_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(j, &keep)| if keep { Some(j) } else { None })
        .collect()
}

// ============================================================================
// VarianceThreshold (TAG 96)
// ============================================================================

#[derive(Clone, Debug)]
pub struct VarianceThreshold {
    pub threshold: f64,
    pub device_type: DeviceType,
    pub device_: Option<String>,
    pub variances_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub is_fitted: bool,
}

impl VarianceThreshold {
    pub fn new(threshold: f64, device: Option<String>) -> OchreResult<Self> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "threshold must be a finite non-negative float, got {}",
                threshold
            )));
        }
        let dev = parse_device(device.as_deref())?;
        let device_ = device.map(|_| dev.to_string());
        Ok(Self {
            threshold,
            device_type: dev,
            device_,
            variances_: None,
            n_features_in_: None,
            is_fitted: false,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.data.iter().any(|v| !v.is_finite()) {
            return Err(OchreError::InvalidParameter(
                "Input X contains NaN or Inf".to_string(),
            ));
        }
        let backend = get_backend(&self.device_type);
        let stats = backend.col_reduce_stats(x, None)?;
        let any_kept = stats.var.iter().any(|&v| v > self.threshold);
        if !any_kept {
            return Err(OchreError::InvalidParameter(format!(
                "No feature in X meets the variance threshold {:.5}",
                self.threshold
            )));
        }
        self.variances_ = Some(stats.var);
        self.n_features_in_ = Some(x.cols);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn get_support_mask(&self) -> OchreResult<Vec<bool>> {
        if !self.is_fitted {
            return Err(OchreError::NotFitted);
        }
        let vars = self.variances_.as_ref().ok_or(OchreError::NotFitted)?;
        Ok(vars.iter().map(|&v| v > self.threshold).collect())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        select_columns_by_mask(x, &mask)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        inverse_select_columns_by_mask(x, &mask)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_VARIANCE_THRESHOLD);
        write_f64(&mut buf, self.threshold);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.variances_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_fitted);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_VARIANCE_THRESHOLD)?;

        let threshold = reader.read_f64()?;
        let saved_device = reader.read_opt_str()?;
        let variances_ = reader.read_opt_f64_vec()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let is_fitted = reader.read_bool()?;

        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            threshold,
            device_type,
            device_,
            variances_,
            n_features_in_,
            is_fitted,
        })
    }
}

#[pyclass(from_py_object, name = "VarianceThreshold")]
#[derive(Clone)]
pub struct PyVarianceThreshold {
    inner: VarianceThreshold,
}

pymethods_with_dit!(PyVarianceThreshold, VarianceThreshold, {
    #[new]
    #[pyo3(signature = (threshold = 0.0, device = None))]
    pub fn new(threshold: f64, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: VarianceThreshold::new(threshold, device)?,
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

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
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

    pub fn get_support_mask(&self) -> PyResult<Vec<bool>> {
        Ok(self.inner.get_support_mask()?)
    }

    pub fn get_support_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mask = self.inner.get_support_mask()?;
        let indices = mask_to_indices(&mask);
        usize_slice_to_i64_pybytes(py, &indices)
    }

    #[getter]
    pub fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn variances_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.variances_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }
});

// ============================================================================
// SelectKBest (TAG 97) & SelectPercentile (TAG 98)
// ============================================================================

fn compute_builtin_scores(
    score_func_name: &str,
    x: &Matrix,
    y_f64: &[f64],
) -> OchreResult<(Vec<f64>, Vec<f64>)> {
    match score_func_name {
        "f_regression" => f_regression(x, y_f64, true, true),
        "f_classif" => {
            let y_i64: Vec<i64> = y_f64.iter().map(|&v| v.round() as i64).collect();
            f_classif(x, &y_i64)
        }
        "chi2" => {
            let y_i64: Vec<i64> = y_f64.iter().map(|&v| v.round() as i64).collect();
            chi2(x, &y_i64)
        }
        other => Err(OchreError::InvalidParameter(format!(
            "Unknown builtin score_func '{}'; expected 'f_classif', 'f_regression', or 'chi2'",
            other
        ))),
    }
}

fn top_k_mask(scores: &[f64], k: Option<usize>) -> Vec<bool> {
    let p = scores.len();
    let Some(k_val) = k else {
        return vec![true; p];
    };
    if k_val == 0 {
        return vec![false; p];
    }
    if k_val >= p {
        return vec![true; p];
    }
    let mut order: Vec<usize> = (0..p).collect();
    // Sort ascending by score (NaN treated as smallest), then take the last k_val indices
    order.sort_by(|&a, &b| {
        let sa = if scores[a].is_nan() {
            f64::NEG_INFINITY
        } else {
            scores[a]
        };
        let sb = if scores[b].is_nan() {
            f64::NEG_INFINITY
        } else {
            scores[b]
        };
        sa.total_cmp(&sb).then_with(|| a.cmp(&b))
    });
    let mut mask = vec![false; p];
    for &idx in &order[p - k_val..] {
        mask[idx] = true;
    }
    mask
}

fn percentile_mask(scores: &[f64], percentile: f64) -> Vec<bool> {
    let p = scores.len();
    if percentile >= 100.0 {
        return vec![true; p];
    }
    if percentile <= 0.0 || p == 0 {
        return vec![false; p];
    }
    let k_val = ((p as f64) * percentile / 100.0).round() as usize;
    top_k_mask(scores, Some(k_val))
}

#[derive(Clone, Debug)]
pub struct SelectKBest {
    pub score_func_name: String,
    pub k: Option<usize>,
    pub device_type: DeviceType,
    pub device_: Option<String>,
    pub scores_: Option<Vec<f64>>,
    pub pvalues_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub is_fitted: bool,
}

impl SelectKBest {
    pub fn new(
        score_func_name: String,
        k: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        let device_ = device.map(|_| dev.to_string());
        Ok(Self {
            score_func_name,
            k,
            device_type: dev,
            device_,
            scores_: None,
            pvalues_: None,
            n_features_in_: None,
            is_fitted: false,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        let (scores, pvalues) = compute_builtin_scores(&self.score_func_name, x, y)?;
        self.scores_ = Some(scores);
        self.pvalues_ = Some(pvalues);
        self.n_features_in_ = Some(x.cols);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn fit_with_scores(
        &mut self,
        n_features: usize,
        scores: Vec<f64>,
        pvalues: Option<Vec<f64>>,
    ) -> OchreResult<()> {
        if scores.len() != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: scores.len(),
                expected: n_features,
            });
        }
        if let Some(ref pv) = pvalues
            && pv.len() != n_features
        {
            return Err(OchreError::FeatureCountMismatch {
                features: pv.len(),
                expected: n_features,
            });
        }
        self.scores_ = Some(scores);
        self.pvalues_ = pvalues;
        self.n_features_in_ = Some(n_features);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn get_support_mask(&self) -> OchreResult<Vec<bool>> {
        if !self.is_fitted {
            return Err(OchreError::NotFitted);
        }
        let scores = self.scores_.as_ref().ok_or(OchreError::NotFitted)?;
        Ok(top_k_mask(scores, self.k))
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        select_columns_by_mask(x, &mask)
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        inverse_select_columns_by_mask(x, &mask)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SELECT_K_BEST);
        write_str(&mut buf, &self.score_func_name);
        write_opt_usize(&mut buf, self.k);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.scores_);
        write_opt_f64_vec(&mut buf, &self.pvalues_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_fitted);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_SELECT_K_BEST)?;

        let score_func_name = reader.read_str()?;
        let k = reader.read_opt_usize()?;
        let saved_device = reader.read_opt_str()?;
        let scores_ = reader.read_opt_f64_vec()?;
        let pvalues_ = reader.read_opt_f64_vec()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let is_fitted = reader.read_bool()?;

        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            score_func_name,
            k,
            device_type,
            device_,
            scores_,
            pvalues_,
            n_features_in_,
            is_fitted,
        })
    }
}

#[pyclass(from_py_object, name = "SelectKBest")]
#[derive(Clone)]
pub struct PySelectKBest {
    inner: SelectKBest,
}

pymethods_with_dit!(PySelectKBest, SelectKBest, {
    #[new]
    #[pyo3(signature = (score_func_name = "f_classif".to_string(), k = Some(10), device = None))]
    pub fn new(
        score_func_name: String,
        k: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SelectKBest::new(score_func_name, k, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&mat, &y_vec)?;
        Ok(())
    }

    #[pyo3(signature = (n_features, scores, pvalues = None))]
    pub fn fit_with_scores(
        &mut self,
        n_features: usize,
        scores: Vec<f64>,
        pvalues: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner.fit_with_scores(n_features, scores, pvalues)?;
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

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let out = self.inner.fit_transform(&mat, &y_vec)?;
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

    pub fn get_support_mask(&self) -> PyResult<Vec<bool>> {
        Ok(self.inner.get_support_mask()?)
    }

    pub fn get_support_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mask = self.inner.get_support_mask()?;
        let indices = mask_to_indices(&mask);
        usize_slice_to_i64_pybytes(py, &indices)
    }

    #[getter]
    pub fn score_func_name(&self) -> String {
        self.inner.score_func_name.clone()
    }

    #[getter]
    pub fn k(&self) -> Option<usize> {
        self.inner.k
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.scores_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn pvalues_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.pvalues_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }
});

#[derive(Clone, Debug)]
pub struct SelectPercentile {
    pub score_func_name: String,
    pub percentile: f64,
    pub device_type: DeviceType,
    pub device_: Option<String>,
    pub scores_: Option<Vec<f64>>,
    pub pvalues_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
    pub is_fitted: bool,
}

impl SelectPercentile {
    pub fn new(
        score_func_name: String,
        percentile: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
            return Err(OchreError::InvalidParameter(format!(
                "percentile must be in [0, 100], got {}",
                percentile
            )));
        }
        let dev = parse_device(device.as_deref())?;
        let device_ = device.map(|_| dev.to_string());
        Ok(Self {
            score_func_name,
            percentile,
            device_type: dev,
            device_,
            scores_: None,
            pvalues_: None,
            n_features_in_: None,
            is_fitted: false,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        let (scores, pvalues) = compute_builtin_scores(&self.score_func_name, x, y)?;
        self.scores_ = Some(scores);
        self.pvalues_ = Some(pvalues);
        self.n_features_in_ = Some(x.cols);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn fit_with_scores(
        &mut self,
        n_features: usize,
        scores: Vec<f64>,
        pvalues: Option<Vec<f64>>,
    ) -> OchreResult<()> {
        if scores.len() != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: scores.len(),
                expected: n_features,
            });
        }
        if let Some(ref pv) = pvalues
            && pv.len() != n_features
        {
            return Err(OchreError::FeatureCountMismatch {
                features: pv.len(),
                expected: n_features,
            });
        }
        self.scores_ = Some(scores);
        self.pvalues_ = pvalues;
        self.n_features_in_ = Some(n_features);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn get_support_mask(&self) -> OchreResult<Vec<bool>> {
        if !self.is_fitted {
            return Err(OchreError::NotFitted);
        }
        let scores = self.scores_.as_ref().ok_or(OchreError::NotFitted)?;
        Ok(percentile_mask(scores, self.percentile))
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        select_columns_by_mask(x, &mask)
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let mask = self.get_support_mask()?;
        inverse_select_columns_by_mask(x, &mask)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SELECT_PERCENTILE);
        write_str(&mut buf, &self.score_func_name);
        write_f64(&mut buf, self.percentile);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.scores_);
        write_opt_f64_vec(&mut buf, &self.pvalues_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_fitted);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_SELECT_PERCENTILE)?;

        let score_func_name = reader.read_str()?;
        let percentile = reader.read_f64()?;
        let saved_device = reader.read_opt_str()?;
        let scores_ = reader.read_opt_f64_vec()?;
        let pvalues_ = reader.read_opt_f64_vec()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let is_fitted = reader.read_bool()?;

        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            score_func_name,
            percentile,
            device_type,
            device_,
            scores_,
            pvalues_,
            n_features_in_,
            is_fitted,
        })
    }
}

#[pyclass(from_py_object, name = "SelectPercentile")]
#[derive(Clone)]
pub struct PySelectPercentile {
    inner: SelectPercentile,
}

pymethods_with_dit!(PySelectPercentile, SelectPercentile, {
    #[new]
    #[pyo3(signature = (score_func_name = "f_classif".to_string(), percentile = 10.0, device = None))]
    pub fn new(score_func_name: String, percentile: f64, device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: SelectPercentile::new(score_func_name, percentile, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        self.inner.fit(&mat, &y_vec)?;
        Ok(())
    }

    #[pyo3(signature = (n_features, scores, pvalues = None))]
    pub fn fit_with_scores(
        &mut self,
        n_features: usize,
        scores: Vec<f64>,
        pvalues: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner.fit_with_scores(n_features, scores, pvalues)?;
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

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let out = self.inner.fit_transform(&mat, &y_vec)?;
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

    pub fn get_support_mask(&self) -> PyResult<Vec<bool>> {
        Ok(self.inner.get_support_mask()?)
    }

    pub fn get_support_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mask = self.inner.get_support_mask()?;
        let indices = mask_to_indices(&mask);
        usize_slice_to_i64_pybytes(py, &indices)
    }

    #[getter]
    pub fn score_func_name(&self) -> String {
        self.inner.score_func_name.clone()
    }

    #[getter]
    pub fn percentile(&self) -> f64 {
        self.inner.percentile
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.scores_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn pvalues_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.pvalues_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }
});

// ============================================================================
// PyO3 Statistical Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (x, y, center = true, force_finite = true))]
pub fn f_regression_buffer<'py>(
    py: Python<'py>,
    x: PyBuffer<f64>,
    y: PyBuffer<f64>,
    center: bool,
    force_finite: bool,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let mat = buffer_to_pooled_matrix(py, &x)?;
    let y_vec = buffer_to_f64_1d(py, &y)?;
    let (scores, pvalues) = f_regression(&mat, &y_vec, center, force_finite)?;
    Ok((
        f64_slice_to_pybytes(py, &scores)?,
        f64_slice_to_pybytes(py, &pvalues)?,
    ))
}

#[pyfunction]
pub fn f_classif_buffer<'py>(
    py: Python<'py>,
    x: PyBuffer<f64>,
    y: Vec<i64>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let mat = buffer_to_pooled_matrix(py, &x)?;
    let (scores, pvalues) = f_classif(&mat, &y)?;
    Ok((
        f64_slice_to_pybytes(py, &scores)?,
        f64_slice_to_pybytes(py, &pvalues)?,
    ))
}

#[pyfunction]
pub fn chi2_buffer<'py>(
    py: Python<'py>,
    x: PyBuffer<f64>,
    y: Vec<i64>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let mat = buffer_to_pooled_matrix(py, &x)?;
    let (scores, pvalues) = chi2(&mat, &y)?;
    Ok((
        f64_slice_to_pybytes(py, &scores)?,
        f64_slice_to_pybytes(py, &pvalues)?,
    ))
}

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVarianceThreshold>()?;
    m.add_class::<PySelectKBest>()?;
    m.add_class::<PySelectPercentile>()?;
    m.add_function(wrap_pyfunction!(f_regression_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(f_classif_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_buffer, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_survival_functions() {
        // chi2(3.841458820694124, df=1) -> p ~ 0.05
        let p_chi = chi2_survival(3.841458820694124, 1.0);
        assert!((p_chi - 0.05).abs() < 1e-6, "p_chi={}", p_chi);

        // F(4.0, df1=2, df2=10) -> (10/18)^5 = 3125/59049 = 0.05292214940134492
        let p_f = f_survival(4.0, 2.0, 10.0);
        assert!(
            (p_f - 0.052_922_149_401_344_92).abs() < 1e-12,
            "p_f={}",
            p_f
        );
    }

    #[test]
    fn test_f_regression_classif_chi2_and_selectors() {
        let x = match Matrix::from_2d(&[
            vec![1.0, 0.0, 10.0],
            vec![2.0, 0.0, 20.0],
            vec![3.0, 0.0, 30.0],
            vec![4.0, 0.0, 40.0],
            vec![5.0, 0.0, 50.0],
            vec![6.0, 0.0, 60.0],
        ]) {
            Ok(m) => m,
            Err(e) => panic!("matrix err: {}", e),
        };
        let y_reg = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let (f_reg, p_reg) = match f_regression(&x, &y_reg, true, true) {
            Ok(res) => res,
            Err(e) => panic!("f_regression err: {}", e),
        };
        assert!(f_reg[0] > 1e6);
        assert_eq!(f_reg[1], 0.0);
        assert_eq!(p_reg[1], 1.0);

        let y_cls = vec![0, 0, 0, 1, 1, 1];
        let (f_cls, p_cls) = match f_classif(&x, &y_cls) {
            Ok(res) => res,
            Err(e) => panic!("f_classif err: {}", e),
        };
        assert!(f_cls[0] > 10.0);
        assert!(p_cls[0] < 0.05);

        let (c2, p_c2) = match chi2(&x, &y_cls) {
            Ok(res) => res,
            Err(e) => panic!("chi2 err: {}", e),
        };
        assert!(c2[0] > 1.0);
        assert!(p_c2[0] < 0.2);

        // VarianceThreshold drops col 1 (var=0)
        let mut vt = match VarianceThreshold::new(0.0, Some("cpu".to_string())) {
            Ok(v) => v,
            Err(e) => panic!("vt err: {}", e),
        };
        let x_vt = match vt.fit_transform(&x) {
            Ok(m) => m,
            Err(e) => panic!("vt fit_transform err: {}", e),
        };
        assert_eq!(x_vt.cols, 2);

        // SelectKBest k=1
        let mut skb =
            match SelectKBest::new("f_classif".to_string(), Some(1), Some("cpu".to_string())) {
                Ok(s) => s,
                Err(e) => panic!("skb err: {}", e),
            };
        let y_f64: Vec<f64> = y_cls.iter().map(|&v| v as f64).collect();
        let x_skb = match skb.fit_transform(&x, &y_f64) {
            Ok(m) => m,
            Err(e) => panic!("skb fit_transform err: {}", e),
        };
        assert_eq!(x_skb.cols, 1);
    }
}
