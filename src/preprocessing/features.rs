// OchreML Feature Engineering Kernels (*≧ω≦*)
// Implements PolynomialFeatures, SplineTransformer (Cox-de Boor B-splines),
// and KernelCenterer (accelerated by Backend::col_reduce_stats N1).

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64_vec, write_header, write_opt_f64,
    write_opt_f64_vec, write_opt_str, write_opt_usize, write_str, write_usize, write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::math_utils::percentile_sorted;
use rayon::prelude::*;

pub const TAG_KERNEL_CENTERER: u8 = 13;
pub const TAG_POLYNOMIAL_FEATURES: u8 = 14;
pub const TAG_SPLINE_TRANSFORMER: u8 = 18;

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

fn gen_combinations_recursive(
    n_features: usize,
    degree: usize,
    start: usize,
    with_replacement: bool,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == degree {
        let mut powers = vec![0usize; n_features];
        for &idx in current.iter() {
            powers[idx] += 1;
        }
        out.push(powers);
        return;
    }
    for i in start..n_features {
        current.push(i);
        let next_start = if with_replacement { i } else { i + 1 };
        gen_combinations_recursive(
            n_features,
            degree,
            next_start,
            with_replacement,
            current,
            out,
        );
        current.pop();
    }
}

/// Generate polynomial and interaction features (＾▽＾)
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    pub min_degree: usize,
    pub max_degree: usize,
    pub interaction_only: bool,
    pub include_bias: bool,
    pub order: String,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub n_features_in_: Option<usize>,
    pub n_output_features_: Option<usize>,
    pub powers_: Option<Vec<usize>>, // flattened (n_output_features_, n_features_in_)
}

impl PolynomialFeatures {
    pub fn new(
        min_degree: usize,
        max_degree: usize,
        interaction_only: bool,
        include_bias: bool,
        order: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if min_degree > max_degree {
            return Err(OchreError::InvalidParameter(format!(
                "min_degree ({}) cannot exceed max_degree ({}) ( >_< )",
                min_degree, max_degree
            )));
        }
        if order != "C" && order != "F" {
            return Err(OchreError::InvalidParameter(format!(
                "order must be 'C' or 'F', got '{}' ( >_< )",
                order
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            min_degree,
            max_degree,
            interaction_only,
            include_bias,
            order,
            device: dev,
            device_: None,
            n_features_in_: None,
            n_output_features_: None,
            powers_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        self.fit_dims(x.cols)
    }

    pub fn fit_dims(&mut self, n_features: usize) -> OchreResult<()> {
        if n_features == 0 {
            return Err(OchreError::EmptyData);
        }
        let start_deg = if self.include_bias && self.min_degree == 0 {
            0
        } else {
            self.min_degree.max(1)
        };
        let mut all_powers: Vec<Vec<usize>> = Vec::new();
        let mut cur = Vec::new();
        for d in start_deg..=self.max_degree {
            if d == 0 {
                all_powers.push(vec![0usize; n_features]);
            } else {
                gen_combinations_recursive(
                    n_features,
                    d,
                    0,
                    !self.interaction_only,
                    &mut cur,
                    &mut all_powers,
                );
            }
        }
        let n_out = all_powers.len();
        let mut flat = Vec::with_capacity(n_out * n_features);
        for p in all_powers {
            flat.extend_from_slice(&p);
        }

        self.n_features_in_ = Some(n_features);
        self.n_output_features_ = Some(n_out);
        self.powers_ = Some(flat);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let _ = validate_cols(self.n_features_in_, x.cols)?;
        let n_out = self.n_output_features_.ok_or(OchreError::NotFitted)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, n_out, Vec::new());
        }
        let mut out = vec![0.0; x.rows * n_out];
        self.transform_to_slice(x, &mut out)?;
        Matrix::from_vec(x.rows, n_out, out)
    }

    pub fn transform_to_slice(&self, x: &Matrix, out: &mut [f64]) -> OchreResult<()> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let n_out = self.n_output_features_.ok_or(OchreError::NotFitted)?;
        let powers = self.powers_.as_ref().ok_or(OchreError::NotFitted)?;
        if x.rows == 0 {
            return Ok(());
        }

        let term_specs: Vec<Vec<(usize, usize)>> = (0..n_out)
            .map(|k| {
                let row = &powers[k * cols..(k + 1) * cols];
                row.iter()
                    .enumerate()
                    .filter(|&(_, &exp)| exp > 0)
                    .map(|(j, &exp)| (j, exp))
                    .collect()
            })
            .collect();

        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            for (k, spec) in term_specs.iter().enumerate() {
                let mut prod = 1.0;
                for &(j, exp) in spec {
                    let v = src[j];
                    if exp == 1 {
                        prod *= v;
                    } else if exp == 2 {
                        prod *= v * v;
                    } else {
                        prod *= v.powi(exp as i32);
                    }
                }
                dst[k] = prod;
            }
        };

        if x.rows * n_out < 16_384 {
            out.chunks_exact_mut(n_out)
                .zip(x.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(n_out)
                .zip(x.data.par_chunks(cols))
                .for_each(process_row);
        }
        Ok(())
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
        let mut buf = write_header(TAG_POLYNOMIAL_FEATURES);
        write_usize(&mut buf, self.min_degree);
        write_usize(&mut buf, self.max_degree);
        write_bool(&mut buf, self.interaction_only);
        write_bool(&mut buf, self.include_bias);
        write_str(&mut buf, &self.order);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_output_features_);
        match &self.powers_ {
            Some(p) => {
                write_bool(&mut buf, true);
                write_usize_vec(&mut buf, p);
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_POLYNOMIAL_FEATURES)?;
        let min_degree = r.read_usize()?;
        let max_degree = r.read_usize()?;
        let interaction_only = r.read_bool()?;
        let include_bias = r.read_bool()?;
        let order = r.read_str()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_output_features_ = r.read_opt_usize()?;
        let powers_ = if r.read_bool()? {
            Some(r.read_usize_vec()?)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            min_degree,
            max_degree,
            interaction_only,
            include_bias,
            order,
            device: dev,
            device_,
            n_features_in_,
            n_output_features_,
            powers_,
        })
    }
}

/// Evaluate all `(degree + 1)` non-zero B-spline basis functions on interval `[t[span], t[span+1])`
/// using Cox-de Boor recurrence (`left`/`right` triangular table).
fn cox_de_boor_basis(x: f64, span: usize, degree: usize, t: &[f64], out_b: &mut [f64]) {
    out_b[0] = 1.0;
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    for j in 1..=degree {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom == 0.0 { 0.0 } else { out_b[r] / denom };
            out_b[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out_b[j] = saved;
    }
}

/// Evaluate derivatives `dB/dx` at `x` for the `(degree + 1)` active basis functions at `span`.
fn cox_de_boor_deriv(x: f64, span: usize, degree: usize, t: &[f64], out_db: &mut [f64]) {
    if degree == 0 {
        out_db[0] = 0.0;
        return;
    }
    let mut b_prev = vec![0.0; degree];
    cox_de_boor_basis(x, span, degree - 1, t, &mut b_prev);
    let d_f = degree as f64;
    for r in 0..=degree {
        let left_term = if r == 0 {
            0.0
        } else {
            let denom = t[span + r] - t[span + r - degree];
            if denom == 0.0 {
                0.0
            } else {
                d_f * b_prev[r - 1] / denom
            }
        };
        let right_term = if r == degree {
            0.0
        } else {
            let denom = t[span + r + 1] - t[span + r + 1 - degree];
            if denom == 0.0 {
                0.0
            } else {
                d_f * b_prev[r] / denom
            }
        };
        out_db[r] = left_term - right_term;
    }
}

fn find_spline_span(x: f64, degree: usize, n_basis: usize, t: &[f64]) -> usize {
    if x >= t[n_basis] {
        return n_basis - 1;
    }
    if x <= t[degree] {
        return degree;
    }
    // Binary search for `span` such that `t[span] <= x < t[span + 1]`
    let mut low = degree;
    let mut high = n_basis;
    let mut mid = (low + high) / 2;
    while x < t[mid] || x >= t[mid + 1] {
        if x < t[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

fn build_extended_knots(base: &[f64], degree: usize, periodic: bool) -> Vec<f64> {
    let k = base.len();
    let mut t = Vec::with_capacity(k + 2 * degree);
    if periodic {
        let period = base[k - 1] - base[0];
        for i in 0..degree {
            t.push(base[k - 1 - degree + i] - period);
        }
        t.extend_from_slice(base);
        for i in 0..degree {
            t.push(base[i + 1] + period);
        }
    } else {
        let step_l = if k >= 2 { base[1] - base[0] } else { 1.0 };
        let step_r = if k >= 2 {
            base[k - 1] - base[k - 2]
        } else {
            1.0
        };
        for i in 0..degree {
            t.push(base[0] - ((degree - i) as f64) * step_l);
        }
        t.extend_from_slice(base);
        for i in 0..degree {
            t.push(base[k - 1] + ((i + 1) as f64) * step_r);
        }
    }
    t
}

/// Generate univariate B-spline bases for features (o´∀｀o)
#[derive(Debug, Clone)]
pub struct SplineTransformer {
    pub n_knots: usize,
    pub degree: usize,
    pub knots_mode: String,
    pub extrapolation: String,
    pub include_bias: bool,
    pub order: String,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub knots_: Option<Vec<Vec<f64>>>, // extended knot vector per feature
    pub n_features_in_: Option<usize>,
    pub n_features_out_: Option<usize>,
}

impl SplineTransformer {
    pub fn new(
        n_knots: usize,
        degree: usize,
        knots_mode: String,
        extrapolation: String,
        include_bias: bool,
        order: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_knots < 2 {
            return Err(OchreError::InvalidParameter(
                "n_knots must be >= 2 ( >_< )".to_string(),
            ));
        }
        if !matches!(
            extrapolation.as_str(),
            "error" | "constant" | "linear" | "continue" | "periodic"
        ) {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid extrapolation '{}' ( >_< )",
                extrapolation
            )));
        }
        if order != "C" && order != "F" {
            return Err(OchreError::InvalidParameter(format!(
                "order must be 'C' or 'F', got '{}' ( >_< )",
                order
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_knots,
            degree,
            knots_mode,
            extrapolation,
            include_bias,
            order,
            device: dev,
            device_: None,
            knots_: None,
            n_features_in_: None,
            n_features_out_: None,
        })
    }

    pub fn fit(&mut self, x: &Matrix, custom_knots: Option<&Matrix>) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let rows = x.rows;
        let cols = x.cols;
        let periodic = self.extrapolation == "periodic";

        let extended_knots: Vec<Vec<f64>> = if let Some(ck) = custom_knots {
            if ck.rows != self.n_knots || ck.cols != cols {
                return Err(OchreError::DimensionMismatch {
                    expected_rows: self.n_knots,
                    expected_cols: cols,
                    actual_rows: ck.rows,
                    actual_cols: ck.cols,
                });
            }
            (0..cols)
                .map(|j| {
                    let base: Vec<f64> = (0..ck.rows).map(|i| ck.data[i * cols + j]).collect();
                    build_extended_knots(&base, self.degree, periodic)
                })
                .collect()
        } else if self.knots_mode == "uniform" {
            let stats = get_backend(&self.device).col_reduce_stats(x, None)?;
            (0..cols)
                .map(|j| {
                    let cmin = stats.min[j];
                    let cmax = stats.max[j];
                    let base: Vec<f64> = (0..self.n_knots)
                        .map(|i| {
                            if i == self.n_knots - 1 {
                                cmax
                            } else {
                                cmin + (i as f64) * (cmax - cmin) / ((self.n_knots - 1) as f64)
                            }
                        })
                        .collect();
                    build_extended_knots(&base, self.degree, periodic)
                })
                .collect()
        } else if self.knots_mode == "quantile" {
            (0..cols)
                .into_par_iter()
                .map(|j| {
                    let mut col_vals: Vec<f64> = (0..rows).map(|r| x.data[r * cols + j]).collect();
                    col_vals.sort_by(|a, b| a.total_cmp(b));
                    let base: Vec<f64> = (0..self.n_knots)
                        .map(|i| {
                            let q = 100.0 * (i as f64) / ((self.n_knots - 1) as f64);
                            percentile_sorted(&col_vals, q)
                        })
                        .collect();
                    build_extended_knots(&base, self.degree, periodic)
                })
                .collect()
        } else {
            return Err(OchreError::InvalidParameter(format!(
                "knots must be 'uniform', 'quantile', or array-like, got '{}' ( >_< )",
                self.knots_mode
            )));
        };

        let n_splines_per_feat = if periodic {
            self.n_knots - 1
        } else {
            self.n_knots + self.degree - 1
        };
        let n_out_per_feat = if self.include_bias {
            n_splines_per_feat
        } else {
            n_splines_per_feat.saturating_sub(1)
        };

        self.knots_ = Some(extended_knots);
        self.n_features_in_ = Some(cols);
        self.n_features_out_ = Some(cols * n_out_per_feat);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let cols = validate_cols(self.n_features_in_, x.cols)?;
        let knots = self.knots_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_out_total = self.n_features_out_.ok_or(OchreError::NotFitted)?;
        if x.rows == 0 {
            return Matrix::from_vec(0, n_out_total, Vec::new());
        }

        let d = self.degree;
        let periodic = self.extrapolation == "periodic";
        let n_basis_full = self.n_knots + d - 1;
        let n_splines_per_feat = if periodic {
            self.n_knots - 1
        } else {
            n_basis_full
        };
        let n_out_per_feat = n_out_total / cols;
        let skip_bias = usize::from(!self.include_bias);
        let ext = self.extrapolation.as_str();

        if ext == "error" {
            for r in 0..x.rows {
                for (j, t) in knots.iter().enumerate().take(cols) {
                    let v = x.data[r * cols + j];
                    if v < t[d] || v > t[n_basis_full] {
                        return Err(OchreError::InvalidParameter(format!(
                            "Value {} outside fitted spline domain [{}, {}] with extrapolation='error' ( >_< )",
                            v, t[d], t[n_basis_full]
                        )));
                    }
                }
            }
        }

        let mut out = vec![0.0; x.rows * n_out_total];
        let process_row = |(dst, src): (&mut [f64], &[f64])| {
            let mut b_local = vec![0.0; d + 1];
            let mut db_local = vec![0.0; d + 1];
            let mut full_row = vec![0.0; n_basis_full];
            for j in 0..cols {
                let t = &knots[j];
                let t_min = t[d];
                let t_max = t[n_basis_full];
                let mut v = src[j];
                full_row.fill(0.0);

                if periodic {
                    let period = t_max - t_min;
                    if period > 0.0 {
                        v = t_min + ((v - t_min).rem_euclid(period));
                    } else {
                        v = t_min;
                    }
                    let span = find_spline_span(v, d, n_basis_full, t);
                    cox_de_boor_basis(v, span, d, t, &mut b_local);
                    for r in 0..=d {
                        full_row[span - d + r] = b_local[r];
                    }
                    // Fold wrapped boundary basis columns into `[0..n_knots - 1]`
                    for r in 0..d {
                        full_row[r] += full_row[n_splines_per_feat + r];
                    }
                } else if v < t_min {
                    match ext {
                        "constant" => {
                            let span = d;
                            cox_de_boor_basis(t_min, span, d, t, &mut b_local);
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r];
                            }
                        }
                        "linear" => {
                            let span = d;
                            cox_de_boor_basis(t_min, span, d, t, &mut b_local);
                            cox_de_boor_deriv(t_min, span, d, t, &mut db_local);
                            let dx = v - t_min;
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r] + db_local[r] * dx;
                            }
                        }
                        _ => {
                            // "continue"
                            let span = d;
                            cox_de_boor_basis(v, span, d, t, &mut b_local);
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r];
                            }
                        }
                    }
                } else if v > t_max {
                    match ext {
                        "constant" => {
                            let span = n_basis_full - 1;
                            cox_de_boor_basis(t_max, span, d, t, &mut b_local);
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r];
                            }
                        }
                        "linear" => {
                            let span = n_basis_full - 1;
                            cox_de_boor_basis(t_max, span, d, t, &mut b_local);
                            cox_de_boor_deriv(t_max, span, d, t, &mut db_local);
                            let dx = v - t_max;
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r] + db_local[r] * dx;
                            }
                        }
                        _ => {
                            let span = n_basis_full - 1;
                            cox_de_boor_basis(v, span, d, t, &mut b_local);
                            for r in 0..=d {
                                full_row[span - d + r] = b_local[r];
                            }
                        }
                    }
                } else {
                    let span = find_spline_span(v, d, n_basis_full, t);
                    cox_de_boor_basis(v, span, d, t, &mut b_local);
                    for r in 0..=d {
                        full_row[span - d + r] = b_local[r];
                    }
                }

                let dst_slice = &mut dst[j * n_out_per_feat..(j + 1) * n_out_per_feat];
                dst_slice.copy_from_slice(&full_row[skip_bias..skip_bias + n_out_per_feat]);
            }
        };

        if x.rows * n_out_total < 16_384 {
            out.chunks_exact_mut(n_out_total)
                .zip(x.data.chunks_exact(cols))
                .for_each(process_row);
        } else {
            out.par_chunks_mut(n_out_total)
                .zip(x.data.par_chunks(cols))
                .for_each(process_row);
        }

        Matrix::from_vec(x.rows, n_out_total, out)
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
        let mut buf = write_header(TAG_SPLINE_TRANSFORMER);
        write_usize(&mut buf, self.n_knots);
        write_usize(&mut buf, self.degree);
        write_str(&mut buf, &self.knots_mode);
        write_str(&mut buf, &self.extrapolation);
        write_bool(&mut buf, self.include_bias);
        write_str(&mut buf, &self.order);
        write_opt_str(&mut buf, &self.device_);
        match &self.knots_ {
            Some(ks) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, ks.len());
                for k in ks {
                    write_f64_vec(&mut buf, k);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_features_out_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SPLINE_TRANSFORMER)?;
        let n_knots = r.read_usize()?;
        let degree = r.read_usize()?;
        let knots_mode = r.read_str()?;
        let extrapolation = r.read_str()?;
        let include_bias = r.read_bool()?;
        let order = r.read_str()?;
        let saved_device = r.read_opt_str()?;
        let knots_ = if r.read_bool()? {
            let len = r.read_usize()?;
            let mut ks = Vec::with_capacity(len);
            for _ in 0..len {
                ks.push(r.read_f64_vec()?);
            }
            Some(ks)
        } else {
            None
        };
        let n_features_in_ = r.read_opt_usize()?;
        let n_features_out_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_knots,
            degree,
            knots_mode,
            extrapolation,
            include_bias,
            order,
            device: dev,
            device_,
            knots_,
            n_features_in_,
            n_features_out_,
        })
    }
}

/// Center an arbitrary kernel matrix in the feature space (*≧ω≦*)
/// Uses Backend::col_reduce_stats (N1) for column means.
#[derive(Debug, Clone)]
pub struct KernelCenterer {
    pub device: DeviceType,
    pub device_: Option<String>,
    pub k_fit_rows_: Option<Vec<f64>>,
    pub k_fit_all_: Option<f64>,
    pub n_features_in_: Option<usize>,
}

impl KernelCenterer {
    pub fn new(device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            device: dev,
            device_: None,
            k_fit_rows_: None,
            k_fit_all_: None,
            n_features_in_: None,
        })
    }

    pub fn fit(&mut self, k: &Matrix) -> OchreResult<()> {
        if k.rows == 0 || k.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if k.rows != k.cols {
            return Err(OchreError::NonSquareMatrix);
        }
        let stats = get_backend(&self.device).col_reduce_stats(k, None)?;
        let mean_all: f64 = stats.mean.iter().sum::<f64>() / stats.mean.len() as f64;
        self.k_fit_rows_ = Some(stats.mean);
        self.k_fit_all_ = Some(mean_all);
        self.n_features_in_ = Some(k.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, k: &Matrix) -> OchreResult<Matrix> {
        let mut out = Matrix::from_vec(k.rows, k.cols, k.data.clone())?;
        self.transform_inplace(&mut out)?;
        Ok(out)
    }

    pub fn transform_inplace(&self, k: &mut Matrix) -> OchreResult<()> {
        let cols = validate_cols(self.n_features_in_, k.cols)?;
        let fit_rows = self.k_fit_rows_.as_ref().ok_or(OchreError::NotFitted)?;
        let fit_all = self.k_fit_all_.ok_or(OchreError::NotFitted)?;
        if k.rows == 0 {
            return Ok(());
        }
        let process_row = |row: &mut [f64]| {
            let row_mean: f64 = row.iter().sum::<f64>() / cols as f64;
            let shift = fit_all - row_mean;
            for (dst, &fr) in row.iter_mut().zip(fit_rows.iter()) {
                *dst = *dst - fr + shift;
            }
        };
        if k.rows * cols < 16_384 {
            k.data.chunks_exact_mut(cols).for_each(process_row);
        } else {
            k.data.par_chunks_mut(cols).for_each(process_row);
        }
        Ok(())
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
        let mut buf = write_header(TAG_KERNEL_CENTERER);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.k_fit_rows_);
        write_opt_f64(&mut buf, self.k_fit_all_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_KERNEL_CENTERER)?;
        let saved_device = r.read_opt_str()?;
        let k_fit_rows_ = r.read_opt_f64_vec()?;
        let k_fit_all_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            device: dev,
            device_,
            k_fit_rows_,
            k_fit_all_,
            n_features_in_,
        })
    }
}
