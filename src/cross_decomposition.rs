// OchreML Matrix & Cross-Decomposition Engine (*≧ω≦*)
// Implements NMF, PLSRegression, and CCA accelerated by Hardware Primitives
// E1 (matmul), E2 (matmul_at_b), E4 (compute_normal_equation), N1 (col_reduce_stats, col_scale_shift),
// N2 (elementwise_op), and N5 (gram_svd, small_sym_eig_svd).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_opt_f64, write_opt_f64_vec,
    write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_opt_usize_vec,
    write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, solve_cholesky_multi};
use crate::preprocessing::py_helpers::{
    buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub const TAG_NMF: u8 = 90;
pub const TAG_CCA: u8 = 91;
pub const TAG_PLS_REGRESSION: u8 = 92;

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        Self {
            state: seed.unwrap_or(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01().max(1e-15);
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn validate_finite_matrix(x: &Matrix, name: &str) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(format!(
            "Input {name} contains NaN or Inf ( >_< )"
        )));
    }
    Ok(())
}

fn validate_non_negative_matrix(x: &Matrix) -> OchreResult<()> {
    validate_finite_matrix(x, "X")?;
    if x.data.iter().any(|&v| v < 0.0) {
        return Err(OchreError::InvalidParameter(
            "Negative values in data passed to NMF ( >_< )".to_string(),
        ));
    }
    Ok(())
}

/// Invert a small square matrix `A (k x k)` using Gauss-Jordan elimination with partial pivoting
/// and gentle Tikhonov fallback if near-singular.
fn invert_square_matrix(a: &Matrix) -> OchreResult<Matrix> {
    if a.rows != a.cols {
        return Err(OchreError::NonSquareMatrix);
    }
    let n = a.rows;
    if n == 0 {
        return Err(OchreError::EmptyData);
    }
    for &reg in &[0.0, 1e-12, 1e-9, 1e-6] {
        let mut aug = vec![0.0; n * 2 * n];
        for r in 0..n {
            for c in 0..n {
                aug[r * 2 * n + c] = a[(r, c)] + if r == c { reg } else { 0.0 };
            }
            aug[r * 2 * n + n + r] = 1.0;
        }
        let mut ok = true;
        for col in 0..n {
            let mut pivot_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for r in (col + 1)..n {
                let val = aug[r * 2 * n + col].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = r;
                }
            }
            if max_val <= 1e-14 {
                ok = false;
                break;
            }
            if pivot_row != col {
                for c in 0..(2 * n) {
                    aug.swap(col * 2 * n + c, pivot_row * 2 * n + c);
                }
            }
            let inv_piv = 1.0 / aug[col * 2 * n + col];
            for c in 0..(2 * n) {
                aug[col * 2 * n + c] *= inv_piv;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug[r * 2 * n + col];
                if factor != 0.0 {
                    for c in 0..(2 * n) {
                        aug[r * 2 * n + c] -= factor * aug[col * 2 * n + c];
                    }
                }
            }
        }
        if ok {
            let mut out = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    out[(r, c)] = aug[r * 2 * n + n + c];
                }
            }
            return Ok(out);
        }
    }
    Err(OchreError::SingularMatrix(
        "Failed to invert rotation matrix in cross-decomposition (´-ω-`)".to_string(),
    ))
}

/// Center and optionally scale matrix `X` using sample std (`ddof = 1`).
/// Returns `(scaled_matrix, mean, std)`.
fn center_and_scale_ddof1(
    backend: &Backend<'_>,
    x: &Matrix,
    scale: bool,
) -> OchreResult<(Matrix, Vec<f64>, Vec<f64>)> {
    let n = x.rows;
    let p = x.cols;
    let stats = backend.col_reduce_stats(x, None)?;
    let mean = stats.mean;
    let std = if scale && n > 1 {
        let factor = (n as f64) / ((n - 1) as f64);
        stats
            .var
            .iter()
            .map(|&v| {
                let s = (v.max(0.0) * factor).sqrt();
                if s <= 1e-15 { 1.0 } else { s }
            })
            .collect()
    } else {
        vec![1.0; p]
    };
    let inv_std: Vec<f64> = std.iter().map(|&s| 1.0 / s).collect();
    let scaled = backend.col_scale_shift(x, &inv_std, &mean)?;
    Ok((scaled, mean, std))
}

/// Apply stored `(X - mean) / std` transform using `backend.col_scale_shift`.
fn apply_center_scale(
    backend: &Backend<'_>,
    x: &Matrix,
    mean: &[f64],
    std: &[f64],
) -> OchreResult<Matrix> {
    let inv_std: Vec<f64> = std.iter().map(|&s| 1.0 / s).collect();
    backend.col_scale_shift(x, &inv_std, mean)
}

/// Compute R^2 score between `y_true` and `y_pred` (uniform average across targets).
fn compute_r2_score(y_true: &Matrix, y_pred: &Matrix) -> OchreResult<f64> {
    if y_true.rows != y_pred.rows || y_true.cols != y_pred.cols {
        return Err(OchreError::DimensionMismatch {
            expected_rows: y_true.rows,
            expected_cols: y_true.cols,
            actual_rows: y_pred.rows,
            actual_cols: y_pred.cols,
        });
    }
    let n = y_true.rows;
    let q = y_true.cols;
    let mut r2_sum = 0.0;
    for col in 0..q {
        let mut mean_y = 0.0;
        for r in 0..n {
            mean_y += y_true[(r, col)];
        }
        mean_y /= n as f64;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for r in 0..n {
            let diff = y_true[(r, col)] - y_pred[(r, col)];
            let centered = y_true[(r, col)] - mean_y;
            ss_res += diff * diff;
            ss_tot += centered * centered;
        }
        if ss_tot <= 1e-15 {
            r2_sum += if ss_res <= 1e-15 { 1.0 } else { 0.0 };
        } else {
            r2_sum += 1.0 - ss_res / ss_tot;
        }
    }
    Ok(r2_sum / (q as f64))
}

#[inline]
fn frobenius_diff(a: &Matrix, b: &Matrix) -> f64 {
    let mut sq = 0.0;
    for (&va, &vb) in a.data.iter().zip(b.data.iter()) {
        let d = va - vb;
        sq += d * d;
    }
    sq.sqrt()
}

// ============================================================================
// 1. Non-negative Matrix Factorization (NMF) (＾▽＾)
// ============================================================================

#[derive(Clone, Debug)]
pub struct NMF {
    pub n_components: Option<usize>,
    pub init: Option<String>,
    pub max_iter: usize,
    pub tol: f64,
    pub l1_ratio: f64,
    pub alpha_w: f64,
    pub alpha_h: Option<f64>,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,

    pub components_: Option<Matrix>,
    pub n_components_: Option<usize>,
    pub reconstruction_err_: Option<f64>,
    pub n_iter_: Option<usize>,
    pub n_features_in_: Option<usize>,
}

impl NMF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: Option<usize>,
        init: Option<String>,
        max_iter: usize,
        tol: f64,
        l1_ratio: f64,
        alpha_w: f64,
        alpha_h: Option<f64>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if let Some(k) = n_components
            && k == 0
        {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if let Some(ref init_str) = init {
            match init_str.as_str() {
                "nndsvd" | "nndsvda" | "random" => {}
                _ => {
                    return Err(OchreError::InvalidParameter(format!(
                        "Invalid init '{init_str}'. Supported: 'nndsvd', 'nndsvda', 'random' ( >_< )"
                    )));
                }
            }
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(OchreError::InvalidParameter(
                "tol must be a non-negative finite number ( >_< )".to_string(),
            ));
        }
        if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) {
            return Err(OchreError::InvalidL1Ratio(l1_ratio));
        }
        if !alpha_w.is_finite() || alpha_w < 0.0 {
            return Err(OchreError::InvalidAlpha(alpha_w));
        }
        if let Some(ah) = alpha_h
            && (!ah.is_finite() || ah < 0.0)
        {
            return Err(OchreError::InvalidAlpha(ah));
        }

        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            init,
            max_iter,
            tol,
            l1_ratio,
            alpha_w,
            alpha_h,
            random_state,
            device: dev,
            device_: None,
            components_: None,
            n_components_: None,
            reconstruction_err_: None,
            n_iter_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_ = Some(dev.to_string());
        self.device = dev;
        Ok(())
    }

    fn initialize_wh(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        k: usize,
        init_mode: &str,
    ) -> OchreResult<(Matrix, Matrix)> {
        let n = x.rows;
        let p = x.cols;
        let mean_x = x.data.iter().sum::<f64>() / ((n * p) as f64);

        if init_mode == "random" {
            let avg = (mean_x / (k as f64)).sqrt();
            let mut rng = SplitMix64::new(self.random_state);
            let mut w = Matrix::zeros(n, k);
            let mut h = Matrix::zeros(k, p);
            for val in &mut w.data {
                *val = avg * rng.next_standard_normal().abs();
            }
            for val in &mut h.data {
                *val = avg * rng.next_standard_normal().abs();
            }
            return Ok((w, h));
        }

        // NNDSVD / NNDSVDA using N5 (gram_svd / small_sym_eig_svd)
        let mut w = Matrix::zeros(n, k);
        let mut h = Matrix::zeros(k, p);

        if p <= n || n > 4096 {
            let (svals, v_right, _) = backend.gram_svd(x, false)?;
            for j in 0..k {
                let sigma = svals.get(j).copied().unwrap_or(0.0);
                if sigma <= 1e-12 {
                    continue;
                }
                let mut v_j = vec![0.0; p];
                for r in 0..p {
                    v_j[r] = v_right[(r, j)];
                }
                let xv = backend.matvec(x, &v_j)?;
                let inv_s = 1.0 / sigma;
                let u_j: Vec<f64> = xv.iter().map(|&val| val * inv_s).collect();
                fill_nndsvd_component(&mut w, &mut h, j, sigma, &u_j, &v_j);
            }
        } else {
            // Dual Gram SVD when n < p and n <= 4096
            let x_t = x.transpose();
            let gram_nn = backend.matmul(x, &x_t)?;
            let eig = backend.small_sym_eig_svd(&gram_nn)?;
            for j in 0..k {
                let sigma = eig
                    .eigenvalues
                    .get(j)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    .sqrt();
                if sigma <= 1e-12 {
                    continue;
                }
                let mut u_j = vec![0.0; n];
                for (r, slot) in u_j.iter_mut().enumerate().take(n) {
                    *slot = eig.eigenvectors[(r, j)];
                }
                let xtu = backend.matvec(&x_t, &u_j)?;
                let inv_s = 1.0 / sigma;
                let v_j: Vec<f64> = xtu.iter().map(|&val| val * inv_s).collect();
                fill_nndsvd_component(&mut w, &mut h, j, sigma, &u_j, &v_j);
            }
        }

        if init_mode == "nndsvda" {
            for val in &mut w.data {
                if *val <= 1e-12 {
                    *val = mean_x;
                }
            }
            for val in &mut h.data {
                if *val <= 1e-12 {
                    *val = mean_x;
                }
            }
        }

        Ok((w, h))
    }

    fn update_h(
        backend: &Backend<'_>,
        x: &Matrix,
        w: &Matrix,
        h: &mut Matrix,
        l1_h: f64,
        l2_h: f64,
    ) -> OchreResult<()> {
        let num = backend.matmul_at_b(w, x)?;
        let wt_w = backend.matmul_at_b(w, w)?;
        let denom = backend.matmul(&wt_w, h)?;
        for i in 0..h.data.len() {
            let d = denom.data[i] + l1_h + l2_h * h.data[i] + 1e-12;
            h.data[i] = (h.data[i] * num.data[i] / d).max(0.0);
        }
        Ok(())
    }

    fn update_w(
        backend: &Backend<'_>,
        x: &Matrix,
        w: &mut Matrix,
        h: &Matrix,
        l1_w: f64,
        l2_w: f64,
    ) -> OchreResult<()> {
        let h_t = h.transpose();
        let num = backend.matmul(x, &h_t)?;
        let h_ht = backend.matmul(h, &h_t)?;
        let denom = backend.matmul(w, &h_ht)?;
        for i in 0..w.data.len() {
            let d = denom.data[i] + l1_w + l2_w * w.data[i] + 1e-12;
            w.data[i] = (w.data[i] * num.data[i] / d).max(0.0);
        }
        Ok(())
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        validate_non_negative_matrix(x)?;
        let n = x.rows;
        let p = x.cols;
        let min_dim = n.min(p);
        let k = self.n_components.unwrap_or(min_dim);

        let init_mode = match self.init.as_deref() {
            Some("nndsvd") => {
                if k > min_dim {
                    return Err(OchreError::InvalidParameter(format!(
                        "init='nndsvd' requires n_components ({k}) <= min(n_samples, n_features) ({min_dim}) ( >_< )"
                    )));
                }
                "nndsvd"
            }
            Some("nndsvda") => {
                if k > min_dim {
                    return Err(OchreError::InvalidParameter(format!(
                        "init='nndsvda' requires n_components ({k}) <= min(n_samples, n_features) ({min_dim}) ( >_< )"
                    )));
                }
                "nndsvda"
            }
            Some("random") => "random",
            None => {
                if k <= min_dim {
                    "nndsvda"
                } else {
                    "random"
                }
            }
            Some(other) => {
                return Err(OchreError::InvalidParameter(format!(
                    "Invalid init '{other}' ( >_< )"
                )));
            }
        };

        let backend = get_backend(&self.device);
        let (mut w, mut h) = self.initialize_wh(&backend, x, k, init_mode)?;

        let alpha_h_val = self.alpha_h.unwrap_or(self.alpha_w);
        let l1_h = (n as f64) * alpha_h_val * self.l1_ratio;
        let l2_h = (n as f64) * alpha_h_val * (1.0 - self.l1_ratio);
        let l1_w = (p as f64) * self.alpha_w * self.l1_ratio;
        let l2_w = (p as f64) * self.alpha_w * (1.0 - self.l1_ratio);

        let recon_init = backend.matmul(&w, &h)?;
        let err_init = frobenius_diff(x, &recon_init);
        let mut prev_err = err_init;
        let mut final_err = err_init;
        let mut n_iter = 0usize;

        for it in 1..=self.max_iter {
            Self::update_w(&backend, x, &mut w, &h, l1_w, l2_w)?;
            Self::update_h(&backend, x, &w, &mut h, l1_h, l2_h)?;
            n_iter = it;

            if it == self.max_iter || it % 10 == 0 {
                let recon = backend.matmul(&w, &h)?;
                final_err = frobenius_diff(x, &recon);
                if it > 1 && (prev_err - final_err).abs() / err_init.max(1e-12) < self.tol {
                    break;
                }
                prev_err = final_err;
            }
        }

        self.components_ = Some(h);
        self.n_components_ = Some(k);
        self.reconstruction_err_ = Some(final_err);
        self.n_iter_ = Some(n_iter);
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());

        Ok(w)
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        let _ = self.fit_transform(x)?;
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        let h = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let k = self.n_components_.ok_or(OchreError::NotFitted)?;
        validate_non_negative_matrix(x)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let backend = get_backend(&self.device);
        let mean_x = x.data.iter().sum::<f64>() / ((x.rows * x.cols) as f64);
        let init_val = (mean_x / (k as f64)).sqrt().max(1e-6);
        let mut w = Matrix::from_vec(x.rows, k, vec![init_val; x.rows * k])?;

        let l1_w = (expected_p as f64) * self.alpha_w * self.l1_ratio;
        let l2_w = (expected_p as f64) * self.alpha_w * (1.0 - self.l1_ratio);

        let h_t = h.transpose();
        let h_ht = backend.matmul(h, &h_t)?;
        let x_ht = backend.matmul(x, &h_t)?;

        let mut prev_err = f64::INFINITY;
        for it in 1..=self.max_iter {
            let denom = backend.matmul(&w, &h_ht)?;
            for i in 0..w.data.len() {
                let d = denom.data[i] + l1_w + l2_w * w.data[i] + 1e-12;
                w.data[i] = (w.data[i] * x_ht.data[i] / d).max(0.0);
            }
            if it % 10 == 0 {
                let recon = backend.matmul(&w, h)?;
                let err = frobenius_diff(x, &recon);
                if (prev_err - err).abs() / prev_err.max(1e-12) < self.tol {
                    break;
                }
                prev_err = err;
            }
        }
        Ok(w)
    }

    pub fn inverse_transform(&self, w: &Matrix) -> OchreResult<Matrix> {
        let h = self.components_.as_ref().ok_or(OchreError::NotFitted)?;
        let k = self.n_components_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(w, "W")?;
        if w.cols != k {
            return Err(OchreError::FeatureCountMismatch {
                features: w.cols,
                expected: k,
            });
        }
        let backend = get_backend(&self.device);
        backend.matmul(w, h)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_NMF);
        write_opt_usize(&mut buf, self.n_components);
        write_opt_str(&mut buf, &self.init);
        write_usize(&mut buf, self.max_iter);
        buf.extend_from_slice(&self.tol.to_le_bytes());
        buf.extend_from_slice(&self.l1_ratio.to_le_bytes());
        buf.extend_from_slice(&self.alpha_w.to_le_bytes());
        write_opt_f64(&mut buf, self.alpha_h);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.components_);
        write_opt_usize(&mut buf, self.n_components_);
        write_opt_f64(&mut buf, self.reconstruction_err_);
        write_opt_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_NMF)?;
        let n_components = r.read_opt_usize()?;
        let init = r.read_opt_str()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let l1_ratio = r.read_f64()?;
        let alpha_w = r.read_f64()?;
        let alpha_h = r.read_opt_f64()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let components_ = r.read_opt_matrix()?;
        let n_components_ = r.read_opt_usize()?;
        let reconstruction_err_ = r.read_opt_f64()?;
        let n_iter_ = r.read_opt_usize()?;
        let n_features_in_ = r.read_opt_usize()?;

        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;

        Ok(Self {
            n_components,
            init,
            max_iter,
            tol,
            l1_ratio,
            alpha_w,
            alpha_h,
            random_state,
            device: dev,
            device_,
            components_,
            n_components_,
            reconstruction_err_,
            n_iter_,
            n_features_in_,
        })
    }
}

fn fill_nndsvd_component(
    w: &mut Matrix,
    h: &mut Matrix,
    j: usize,
    sigma: f64,
    u_j: &[f64],
    v_j: &[f64],
) {
    let mut u_p_sq = 0.0;
    let mut u_n_sq = 0.0;
    for &u in u_j {
        if u >= 0.0 {
            u_p_sq += u * u;
        } else {
            u_n_sq += u * u;
        }
    }
    let mut v_p_sq = 0.0;
    let mut v_n_sq = 0.0;
    for &v in v_j {
        if v >= 0.0 {
            v_p_sq += v * v;
        } else {
            v_n_sq += v * v;
        }
    }
    let u_p_norm = u_p_sq.sqrt();
    let u_n_norm = u_n_sq.sqrt();
    let v_p_norm = v_p_sq.sqrt();
    let v_n_norm = v_n_sq.sqrt();

    let m_p = u_p_norm * v_p_norm;
    let m_n = u_n_norm * v_n_norm;

    if m_p >= m_n {
        if m_p <= 1e-15 {
            return;
        }
        let scale = (sigma * m_p).sqrt();
        let u_scale = scale / u_p_norm.max(1e-15);
        let v_scale = scale / v_p_norm.max(1e-15);
        for (r, &u) in u_j.iter().enumerate() {
            if u > 0.0 {
                w[(r, j)] = u * u_scale;
            }
        }
        for (c, &v) in v_j.iter().enumerate() {
            if v > 0.0 {
                h[(j, c)] = v * v_scale;
            }
        }
    } else {
        let scale = (sigma * m_n).sqrt();
        let u_scale = scale / u_n_norm.max(1e-15);
        let v_scale = scale / v_n_norm.max(1e-15);
        for (r, &u) in u_j.iter().enumerate() {
            if u < 0.0 {
                w[(r, j)] = (-u) * u_scale;
            }
        }
        for (c, &v) in v_j.iter().enumerate() {
            if v < 0.0 {
                h[(j, c)] = (-v) * v_scale;
            }
        }
    }
}

/// ============================================================================
/// 2. Shared Cross-Decomposition NIPALS Engine (PLSRegression & CCA) (*≧ω≦*)
/// ============================================================================

#[derive(Clone, Debug)]
struct NipalsFitResult {
    x_weights_: Matrix,
    y_weights_: Matrix,
    x_loadings_: Matrix,
    y_loadings_: Matrix,
    x_scores_: Matrix,
    y_scores_: Matrix,
    x_rotations_: Matrix,
    y_rotations_: Matrix,
    coef_: Matrix,
    intercept_: Vec<f64>,
    x_mean_: Vec<f64>,
    y_mean_: Vec<f64>,
    x_std_: Vec<f64>,
    y_std_: Vec<f64>,
    n_iter_: Vec<usize>,
}

fn vec_norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[allow(clippy::too_many_arguments)]
fn run_nipals(
    backend: &Backend<'_>,
    x: &Matrix,
    y: &Matrix,
    n_components: usize,
    scale: bool,
    max_iter: usize,
    tol: f64,
    mode_b: bool,
) -> OchreResult<NipalsFitResult> {
    let n = x.rows;
    let p = x.cols;
    let q = y.cols;

    let (mut xk, x_mean, x_std) = center_and_scale_ddof1(backend, x, scale)?;
    let (mut yk, y_mean, y_std) = center_and_scale_ddof1(backend, y, scale)?;

    let mut x_weights = Matrix::zeros(p, n_components);
    let mut y_weights = Matrix::zeros(q, n_components);
    let mut x_loadings = Matrix::zeros(p, n_components);
    let mut y_loadings = Matrix::zeros(q, n_components);
    let mut x_scores = Matrix::zeros(n, n_components);
    let mut y_scores = Matrix::zeros(n, n_components);
    let mut n_iter = Vec::with_capacity(n_components);

    for k in 0..n_components {
        // Initialize u as first non-zero column of Yk
        let mut u = vec![0.0; n];
        for col in 0..q {
            let mut col_sq = 0.0;
            for r in 0..n {
                let v = yk[(r, col)];
                col_sq += v * v;
            }
            if col_sq > 1e-24 {
                for r in 0..n {
                    u[r] = yk[(r, col)];
                }
                break;
            }
        }

        let mut w = vec![0.0; p];
        let mut c = vec![0.0; q];
        let mut t = vec![0.0; n];
        let mut iters_done = 0usize;

        // Precompute Gram matrices if Mode B (CCA)
        let xk_gram = if mode_b {
            Some(backend.matmul_at_b(&xk, &xk)?)
        } else {
            None
        };
        let yk_gram = if mode_b {
            Some(backend.matmul_at_b(&yk, &yk)?)
        } else {
            None
        };

        for it in 1..=max_iter {
            iters_done = it;
            let w_old = w.clone();

            let u_mat = Matrix::from_vec(n, 1, u.clone())?;
            let xk_t_u = backend.matmul_at_b(&xk, &u_mat)?;
            if let Some(ref gram_x) = xk_gram {
                let sol = solve_cholesky_multi(gram_x, &xk_t_u)?;
                w = sol.data;
            } else {
                let u_norm_sq = dot_slice(&u, &u).max(1e-15);
                w = xk_t_u.data.iter().map(|&v| v / u_norm_sq).collect();
            }
            let w_norm = vec_norm2(&w);
            if w_norm > 1e-15 {
                let inv = 1.0 / w_norm;
                for val in &mut w {
                    *val *= inv;
                }
            } else {
                w[0] = 1.0;
            }

            t = backend.matvec(&xk, &w)?;

            let t_mat = Matrix::from_vec(n, 1, t.clone())?;
            let yk_t_t = backend.matmul_at_b(&yk, &t_mat)?;
            if let Some(ref gram_y) = yk_gram {
                let sol = solve_cholesky_multi(gram_y, &yk_t_t)?;
                c = sol.data;
                let c_norm = vec_norm2(&c);
                if c_norm > 1e-15 {
                    let inv = 1.0 / c_norm;
                    for val in &mut c {
                        *val *= inv;
                    }
                } else {
                    c[0] = 1.0;
                }
                u = backend.matvec(&yk, &c)?;
            } else {
                let t_norm_sq = dot_slice(&t, &t).max(1e-15);
                c = yk_t_t.data.iter().map(|&v| v / t_norm_sq).collect();
                let c_norm_sq = dot_slice(&c, &c);
                if c_norm_sq > 1e-15 {
                    let yk_c = backend.matvec(&yk, &c)?;
                    let inv = 1.0 / c_norm_sq;
                    u = yk_c.iter().map(|&v| v * inv).collect();
                } else {
                    u.fill(0.0);
                }
            }

            let mut diff_sq = 0.0;
            for (&wi, &wo) in w.iter().zip(w_old.iter()) {
                let d = wi - wo;
                diff_sq += d * d;
            }
            if diff_sq < tol || (!mode_b && q == 1) {
                break;
            }
        }
        n_iter.push(iters_done);

        // Compute loadings and deflate
        let t_mat = Matrix::from_vec(n, 1, t.clone())?;
        let t_norm_sq = dot_slice(&t, &t).max(1e-15);
        let p_vec: Vec<f64> = backend
            .matmul_at_b(&xk, &t_mat)?
            .data
            .into_iter()
            .map(|v| v / t_norm_sq)
            .collect();

        let q_vec: Vec<f64> = if mode_b {
            let u_mat = Matrix::from_vec(n, 1, u.clone())?;
            let u_norm_sq = dot_slice(&u, &u).max(1e-15);
            backend
                .matmul_at_b(&yk, &u_mat)?
                .data
                .into_iter()
                .map(|v| v / u_norm_sq)
                .collect()
        } else {
            backend
                .matmul_at_b(&yk, &t_mat)?
                .data
                .into_iter()
                .map(|v| v / t_norm_sq)
                .collect()
        };

        // Deflate Xk -= t * p_vec^T
        for r in 0..n {
            let tr = t[r];
            for col in 0..p {
                xk[(r, col)] -= tr * p_vec[col];
            }
        }
        // Deflate Yk -= (u if mode_b else t) * q_vec^T
        let y_deflate_score = if mode_b { &u } else { &t };
        for r in 0..n {
            let sr = y_deflate_score[r];
            for col in 0..q {
                yk[(r, col)] -= sr * q_vec[col];
            }
        }

        for r in 0..p {
            x_weights[(r, k)] = w[r];
            x_loadings[(r, k)] = p_vec[r];
        }
        for r in 0..q {
            y_weights[(r, k)] = c[r];
            y_loadings[(r, k)] = q_vec[r];
        }
        for r in 0..n {
            x_scores[(r, k)] = t[r];
            y_scores[(r, k)] = u[r];
        }
    }

    // x_rotations_ = W * (P^T * W)^{-1}
    let pt_w = backend.matmul_at_b(&x_loadings, &x_weights)?;
    let pt_w_inv = invert_square_matrix(&pt_w)?;
    let x_rotations = backend.matmul(&x_weights, &pt_w_inv)?;

    // y_rotations_ = C * (Q^T * C)^{-1}
    let qt_c = backend.matmul_at_b(&y_loadings, &y_weights)?;
    let qt_c_inv = invert_square_matrix(&qt_c)?;
    let y_rotations = backend.matmul(&y_weights, &qt_c_inv)?;

    // Regression coefficients: B_scaled = x_rotations * B_y where B_y = (T^T T)^{-1} T^T Y_scaled
    let (y_scaled, _, _) = center_and_scale_ddof1(backend, y, scale)?;
    let tt_t = backend.matmul_at_b(&x_scores, &x_scores)?;
    let tt_y = backend.matmul_at_b(&x_scores, &y_scaled)?;
    let b_y = solve_cholesky_multi(&tt_t, &tt_y)?;
    let b_scaled = backend.matmul(&x_rotations, &b_y)?; // (p x q)

    // coef_ (q x p): coef_[j, i] = b_scaled[i, j] * y_std[j] / x_std[i]
    let mut coef = Matrix::zeros(q, p);
    for j in 0..q {
        for i in 0..p {
            coef[(j, i)] = b_scaled[(i, j)] * y_std[j] / x_std[i];
        }
    }
    // intercept_ = y_mean - coef_ * x_mean
    let coef_xmean = backend.matvec(&coef, &x_mean)?;
    let intercept: Vec<f64> = y_mean
        .iter()
        .zip(coef_xmean.iter())
        .map(|(&ym, &cx)| ym - cx)
        .collect();

    Ok(NipalsFitResult {
        x_weights_: x_weights,
        y_weights_: y_weights,
        x_loadings_: x_loadings,
        y_loadings_: y_loadings,
        x_scores_: x_scores,
        y_scores_: y_scores,
        x_rotations_: x_rotations,
        y_rotations_: y_rotations,
        coef_: coef,
        intercept_: intercept,
        x_mean_: x_mean,
        y_mean_: y_mean,
        x_std_: x_std,
        y_std_: y_std,
        n_iter_: n_iter,
    })
}

/// ============================================================================
/// 3. PLSRegression (*≧ω≦*)
/// ============================================================================

#[derive(Clone, Debug)]
pub struct PLSRegression {
    pub n_components: usize,
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub device: DeviceType,
    pub device_: Option<String>,

    pub x_weights_: Option<Matrix>,
    pub y_weights_: Option<Matrix>,
    pub x_loadings_: Option<Matrix>,
    pub y_loadings_: Option<Matrix>,
    pub x_scores_: Option<Matrix>,
    pub y_scores_: Option<Matrix>,
    pub x_rotations_: Option<Matrix>,
    pub y_rotations_: Option<Matrix>,
    pub coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub x_mean_: Option<Vec<f64>>,
    pub y_mean_: Option<Vec<f64>>,
    pub x_std_: Option<Vec<f64>>,
    pub y_std_: Option<Vec<f64>>,
    pub n_iter_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
    pub n_targets_: Option<usize>,
}

impl PLSRegression {
    pub fn new(
        n_components: usize,
        scale: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(OchreError::InvalidParameter(
                "tol must be a non-negative finite number ( >_< )".to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            scale,
            max_iter,
            tol,
            device: dev,
            device_: None,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            coef_: None,
            intercept_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            n_iter_: None,
            n_features_in_: None,
            n_targets_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_ = Some(dev.to_string());
        self.device = dev;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x, "X")?;
        validate_finite_matrix(y, "Y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let max_comp = x.rows.min(x.cols);
        if self.n_components > max_comp {
            return Err(OchreError::InvalidParameter(format!(
                "n_components ({}) must be <= min(n_samples, n_features) ({max_comp}) ( >_< )",
                self.n_components
            )));
        }

        let backend = get_backend(&self.device);
        let res = run_nipals(
            &backend,
            x,
            y,
            self.n_components,
            self.scale,
            self.max_iter,
            self.tol,
            false,
        )?;

        self.x_weights_ = Some(res.x_weights_);
        self.y_weights_ = Some(res.y_weights_);
        self.x_loadings_ = Some(res.x_loadings_);
        self.y_loadings_ = Some(res.y_loadings_);
        self.x_scores_ = Some(res.x_scores_);
        self.y_scores_ = Some(res.y_scores_);
        self.x_rotations_ = Some(res.x_rotations_);
        self.y_rotations_ = Some(res.y_rotations_);
        self.coef_ = Some(res.coef_);
        self.intercept_ = Some(res.intercept_);
        self.x_mean_ = Some(res.x_mean_);
        self.y_mean_ = Some(res.y_mean_);
        self.x_std_ = Some(res.x_std_);
        self.y_std_ = Some(res.y_std_);
        self.n_iter_ = Some(res.n_iter_);
        self.n_features_in_ = Some(x.cols);
        self.n_targets_ = Some(y.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(
        &self,
        x: &Matrix,
        y: Option<&Matrix>,
    ) -> OchreResult<(Matrix, Option<Matrix>)> {
        let x_rot = self.x_rotations_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_mean = self.x_mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_std = self.x_std_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x, "X")?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let backend = get_backend(&self.device);
        let x_scaled = apply_center_scale(&backend, x, x_mean, x_std)?;
        let x_scores = backend.matmul(&x_scaled, x_rot)?;

        let y_scores = if let Some(y_mat) = y {
            let y_rot = self.y_rotations_.as_ref().ok_or(OchreError::NotFitted)?;
            let y_mean = self.y_mean_.as_ref().ok_or(OchreError::NotFitted)?;
            let y_std = self.y_std_.as_ref().ok_or(OchreError::NotFitted)?;
            let expected_q = self.n_targets_.ok_or(OchreError::NotFitted)?;
            validate_finite_matrix(y_mat, "Y")?;
            if y_mat.rows != x.rows {
                return Err(OchreError::LengthMismatch {
                    features: x.rows,
                    targets: y_mat.rows,
                });
            }
            if y_mat.cols != expected_q {
                return Err(OchreError::FeatureCountMismatch {
                    features: y_mat.cols,
                    expected: expected_q,
                });
            }
            let y_scaled = apply_center_scale(&backend, y_mat, y_mean, y_std)?;
            Some(backend.matmul(&y_scaled, y_rot)?)
        } else {
            None
        };

        Ok((x_scores, y_scores))
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<(Matrix, Matrix)> {
        self.fit(x, y)?;
        let x_scores = self.x_scores_.clone().ok_or(OchreError::NotFitted)?;
        let y_scores = self.y_scores_.clone().ok_or(OchreError::NotFitted)?;
        Ok((x_scores, y_scores))
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x, "X")?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }
        let backend = get_backend(&self.device);
        let coef_t = coef.transpose();
        let mut pred = backend.matmul(x, &coef_t)?;
        for r in 0..pred.rows {
            for (c, &b) in intercept.iter().enumerate() {
                pred[(r, c)] += b;
            }
        }
        Ok(pred)
    }

    pub fn inverse_transform(&self, x_scores: &Matrix) -> OchreResult<Matrix> {
        let x_load = self.x_loadings_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_mean = self.x_mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_std = self.x_std_.as_ref().ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x_scores, "X")?;
        if x_scores.cols != self.n_components {
            return Err(OchreError::FeatureCountMismatch {
                features: x_scores.cols,
                expected: self.n_components,
            });
        }
        let backend = get_backend(&self.device);
        let x_load_t = x_load.transpose();
        let mut x_rec = backend.matmul(x_scores, &x_load_t)?;
        for r in 0..x_rec.rows {
            for c in 0..x_rec.cols {
                x_rec[(r, c)] = x_rec[(r, c)] * x_std[c] + x_mean[c];
            }
        }
        Ok(x_rec)
    }

    pub fn score(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        validate_finite_matrix(y, "Y")?;
        let y_pred = self.predict(x)?;
        compute_r2_score(y, &y_pred)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_PLS_REGRESSION);
        write_usize(&mut buf, self.n_components);
        write_bool(&mut buf, self.scale);
        write_usize(&mut buf, self.max_iter);
        buf.extend_from_slice(&self.tol.to_le_bytes());
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_weights_);
        write_opt_matrix(&mut buf, &self.y_weights_);
        write_opt_matrix(&mut buf, &self.x_loadings_);
        write_opt_matrix(&mut buf, &self.y_loadings_);
        write_opt_matrix(&mut buf, &self.x_scores_);
        write_opt_matrix(&mut buf, &self.y_scores_);
        write_opt_matrix(&mut buf, &self.x_rotations_);
        write_opt_matrix(&mut buf, &self.y_rotations_);
        write_opt_matrix(&mut buf, &self.coef_);
        write_opt_f64_vec(&mut buf, &self.intercept_);
        write_opt_f64_vec(&mut buf, &self.x_mean_);
        write_opt_f64_vec(&mut buf, &self.y_mean_);
        write_opt_f64_vec(&mut buf, &self.x_std_);
        write_opt_f64_vec(&mut buf, &self.y_std_);
        write_opt_usize_vec(&mut buf, &self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_targets_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_PLS_REGRESSION)?;
        let n_components = r.read_usize()?;
        let scale = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let saved_device = r.read_opt_str()?;
        let x_weights_ = r.read_opt_matrix()?;
        let y_weights_ = r.read_opt_matrix()?;
        let x_loadings_ = r.read_opt_matrix()?;
        let y_loadings_ = r.read_opt_matrix()?;
        let x_scores_ = r.read_opt_matrix()?;
        let y_scores_ = r.read_opt_matrix()?;
        let x_rotations_ = r.read_opt_matrix()?;
        let y_rotations_ = r.read_opt_matrix()?;
        let coef_ = r.read_opt_matrix()?;
        let intercept_ = r.read_opt_f64_vec()?;
        let x_mean_ = r.read_opt_f64_vec()?;
        let y_mean_ = r.read_opt_f64_vec()?;
        let x_std_ = r.read_opt_f64_vec()?;
        let y_std_ = r.read_opt_f64_vec()?;
        let n_iter_ = r.read_opt_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_targets_ = r.read_opt_usize()?;

        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;

        Ok(Self {
            n_components,
            scale,
            max_iter,
            tol,
            device: dev,
            device_,
            x_weights_,
            y_weights_,
            x_loadings_,
            y_loadings_,
            x_scores_,
            y_scores_,
            x_rotations_,
            y_rotations_,
            coef_,
            intercept_,
            x_mean_,
            y_mean_,
            x_std_,
            y_std_,
            n_iter_,
            n_features_in_,
            n_targets_,
        })
    }
}

/// ============================================================================
/// 4. Canonical Correlation Analysis (CCA) (*≧ω≦*)
/// ============================================================================

#[derive(Clone, Debug)]
pub struct CCA {
    pub n_components: usize,
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub device: DeviceType,
    pub device_: Option<String>,

    pub x_weights_: Option<Matrix>,
    pub y_weights_: Option<Matrix>,
    pub x_loadings_: Option<Matrix>,
    pub y_loadings_: Option<Matrix>,
    pub x_scores_: Option<Matrix>,
    pub y_scores_: Option<Matrix>,
    pub x_rotations_: Option<Matrix>,
    pub y_rotations_: Option<Matrix>,
    pub coef_: Option<Matrix>,
    pub intercept_: Option<Vec<f64>>,
    pub x_mean_: Option<Vec<f64>>,
    pub y_mean_: Option<Vec<f64>>,
    pub x_std_: Option<Vec<f64>>,
    pub y_std_: Option<Vec<f64>>,
    pub n_iter_: Option<Vec<usize>>,
    pub n_features_in_: Option<usize>,
    pub n_targets_: Option<usize>,
}

impl CCA {
    pub fn new(
        n_components: usize,
        scale: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(OchreError::InvalidParameter(
                "tol must be a non-negative finite number ( >_< )".to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            scale,
            max_iter,
            tol,
            device: dev,
            device_: None,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            coef_: None,
            intercept_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            n_iter_: None,
            n_features_in_: None,
            n_targets_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_ = Some(dev.to_string());
        self.device = dev;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x, "X")?;
        validate_finite_matrix(y, "Y")?;
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        let max_comp = x.rows.min(x.cols).min(y.cols);
        if self.n_components > max_comp {
            return Err(OchreError::InvalidParameter(format!(
                "n_components ({}) must be <= min(n_samples, n_features, n_targets) ({max_comp}) ( >_< )",
                self.n_components
            )));
        }

        let backend = get_backend(&self.device);
        let res = run_nipals(
            &backend,
            x,
            y,
            self.n_components,
            self.scale,
            self.max_iter,
            self.tol,
            true,
        )?;

        self.x_weights_ = Some(res.x_weights_);
        self.y_weights_ = Some(res.y_weights_);
        self.x_loadings_ = Some(res.x_loadings_);
        self.y_loadings_ = Some(res.y_loadings_);
        self.x_scores_ = Some(res.x_scores_);
        self.y_scores_ = Some(res.y_scores_);
        self.x_rotations_ = Some(res.x_rotations_);
        self.y_rotations_ = Some(res.y_rotations_);
        self.coef_ = Some(res.coef_);
        self.intercept_ = Some(res.intercept_);
        self.x_mean_ = Some(res.x_mean_);
        self.y_mean_ = Some(res.y_mean_);
        self.x_std_ = Some(res.x_std_);
        self.y_std_ = Some(res.y_std_);
        self.n_iter_ = Some(res.n_iter_);
        self.n_features_in_ = Some(x.cols);
        self.n_targets_ = Some(y.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(
        &self,
        x: &Matrix,
        y: Option<&Matrix>,
    ) -> OchreResult<(Matrix, Option<Matrix>)> {
        let x_rot = self.x_rotations_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_mean = self.x_mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let x_std = self.x_std_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x, "X")?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }

        let backend = get_backend(&self.device);
        let x_scaled = apply_center_scale(&backend, x, x_mean, x_std)?;
        let x_scores = backend.matmul(&x_scaled, x_rot)?;

        let y_scores = if let Some(y_mat) = y {
            let y_rot = self.y_rotations_.as_ref().ok_or(OchreError::NotFitted)?;
            let y_mean = self.y_mean_.as_ref().ok_or(OchreError::NotFitted)?;
            let y_std = self.y_std_.as_ref().ok_or(OchreError::NotFitted)?;
            let expected_q = self.n_targets_.ok_or(OchreError::NotFitted)?;
            validate_finite_matrix(y_mat, "Y")?;
            if y_mat.rows != x.rows {
                return Err(OchreError::LengthMismatch {
                    features: x.rows,
                    targets: y_mat.rows,
                });
            }
            if y_mat.cols != expected_q {
                return Err(OchreError::FeatureCountMismatch {
                    features: y_mat.cols,
                    expected: expected_q,
                });
            }
            let y_scaled = apply_center_scale(&backend, y_mat, y_mean, y_std)?;
            Some(backend.matmul(&y_scaled, y_rot)?)
        } else {
            None
        };

        Ok((x_scores, y_scores))
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &Matrix) -> OchreResult<(Matrix, Matrix)> {
        self.fit(x, y)?;
        let x_scores = self.x_scores_.clone().ok_or(OchreError::NotFitted)?;
        let y_scores = self.y_scores_.clone().ok_or(OchreError::NotFitted)?;
        Ok((x_scores, y_scores))
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.as_ref().ok_or(OchreError::NotFitted)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_finite_matrix(x, "X")?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }
        let backend = get_backend(&self.device);
        let coef_t = coef.transpose();
        let mut pred = backend.matmul(x, &coef_t)?;
        for r in 0..pred.rows {
            for (c, &b) in intercept.iter().enumerate() {
                pred[(r, c)] += b;
            }
        }
        Ok(pred)
    }

    pub fn score(&self, x: &Matrix, y: &Matrix) -> OchreResult<f64> {
        validate_finite_matrix(y, "Y")?;
        let y_pred = self.predict(x)?;
        compute_r2_score(y, &y_pred)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_CCA);
        write_usize(&mut buf, self.n_components);
        write_bool(&mut buf, self.scale);
        write_usize(&mut buf, self.max_iter);
        buf.extend_from_slice(&self.tol.to_le_bytes());
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.x_weights_);
        write_opt_matrix(&mut buf, &self.y_weights_);
        write_opt_matrix(&mut buf, &self.x_loadings_);
        write_opt_matrix(&mut buf, &self.y_loadings_);
        write_opt_matrix(&mut buf, &self.x_scores_);
        write_opt_matrix(&mut buf, &self.y_scores_);
        write_opt_matrix(&mut buf, &self.x_rotations_);
        write_opt_matrix(&mut buf, &self.y_rotations_);
        write_opt_matrix(&mut buf, &self.coef_);
        write_opt_f64_vec(&mut buf, &self.intercept_);
        write_opt_f64_vec(&mut buf, &self.x_mean_);
        write_opt_f64_vec(&mut buf, &self.y_mean_);
        write_opt_f64_vec(&mut buf, &self.x_std_);
        write_opt_f64_vec(&mut buf, &self.y_std_);
        write_opt_usize_vec(&mut buf, &self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.n_targets_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_CCA)?;
        let n_components = r.read_usize()?;
        let scale = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_f64()?;
        let saved_device = r.read_opt_str()?;
        let x_weights_ = r.read_opt_matrix()?;
        let y_weights_ = r.read_opt_matrix()?;
        let x_loadings_ = r.read_opt_matrix()?;
        let y_loadings_ = r.read_opt_matrix()?;
        let x_scores_ = r.read_opt_matrix()?;
        let y_scores_ = r.read_opt_matrix()?;
        let x_rotations_ = r.read_opt_matrix()?;
        let y_rotations_ = r.read_opt_matrix()?;
        let coef_ = r.read_opt_matrix()?;
        let intercept_ = r.read_opt_f64_vec()?;
        let x_mean_ = r.read_opt_f64_vec()?;
        let y_mean_ = r.read_opt_f64_vec()?;
        let x_std_ = r.read_opt_f64_vec()?;
        let y_std_ = r.read_opt_f64_vec()?;
        let n_iter_ = r.read_opt_usize_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_targets_ = r.read_opt_usize()?;

        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;

        Ok(Self {
            n_components,
            scale,
            max_iter,
            tol,
            device: dev,
            device_,
            x_weights_,
            y_weights_,
            x_loadings_,
            y_loadings_,
            x_scores_,
            y_scores_,
            x_rotations_,
            y_rotations_,
            coef_,
            intercept_,
            x_mean_,
            y_mean_,
            x_std_,
            y_std_,
            n_iter_,
            n_features_in_,
            n_targets_,
        })
    }
}

// ============================================================================
// 5. PyO3 Bindings (*≧ω≦*)
// ============================================================================

type PyMatrixReturn<'py> = (usize, usize, Bound<'py, PyBytes>);
type PyOptMatrixReturn<'py> = Option<PyMatrixReturn<'py>>;

#[pyclass(from_py_object, name = "NMF")]
#[derive(Clone)]
pub struct PyNMF {
    inner: NMF,
}

pymethods_with_dit!(PyNMF, NMF, {
    #[new]
    #[pyo3(signature = (
        n_components = None,
        init = None,
        max_iter = 200,
        tol = 1e-4,
        l1_ratio = 0.0,
        alpha_w = 0.0,
        alpha_h = None,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: Option<usize>,
        init: Option<String>,
        max_iter: usize,
        tol: f64,
        l1_ratio: f64,
        alpha_w: f64,
        alpha_h: Option<f64>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NMF::new(
                n_components,
                init,
                max_iter,
                tol,
                l1_ratio,
                alpha_w,
                alpha_h,
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
    ) -> PyResult<PyMatrixReturn<'py>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<PyMatrixReturn<'py>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        w: PyBuffer<f64>,
    ) -> PyResult<PyMatrixReturn<'py>> {
        let mat = buffer_to_pooled_matrix(py, &w)?;
        let out = self.inner.inverse_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_components(&self) -> Option<usize> {
        self.inner.n_components
    }
    #[getter]
    pub fn init(&self) -> Option<String> {
        self.inner.init.clone()
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn l1_ratio(&self) -> f64 {
        self.inner.l1_ratio
    }
    #[getter]
    pub fn alpha_w(&self) -> f64 {
        self.inner.alpha_w
    }
    #[getter]
    pub fn alpha_h(&self) -> Option<f64> {
        self.inner.alpha_h
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn components_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        match &self.inner.components_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }
    #[getter]
    pub fn n_components_(&self) -> Option<usize> {
        self.inner.n_components_
    }
    #[getter]
    pub fn reconstruction_err_(&self) -> Option<f64> {
        self.inner.reconstruction_err_
    }
    #[getter]
    pub fn n_iter_(&self) -> Option<usize> {
        self.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "PLSRegression")]
#[derive(Clone)]
pub struct PyPLSRegression {
    inner: PLSRegression,
}

pymethods_with_dit!(PyPLSRegression, PLSRegression, {
    #[new]
    #[pyo3(signature = (n_components = 2, scale = true, max_iter = 500, tol = 1e-6, device = None))]
    pub fn new(
        n_components: usize,
        scale: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PLSRegression::new(n_components, scale, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        self.inner.fit(&x_mat, &y_mat)?;
        Ok(())
    }

    #[pyo3(signature = (x, y = None))]
    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: Option<PyBuffer<f64>>,
    ) -> PyResult<(PyMatrixReturn<'py>, PyOptMatrixReturn<'py>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = match y {
            Some(ref yb) => Some(buffer_to_pooled_matrix(py, yb)?),
            None => None,
        };
        let (xs, ys) = self.inner.transform(&x_mat, y_mat.as_deref())?;
        let xs_bytes = matrix_to_pybytes(py, &xs)?;
        let ys_bytes = match ys {
            Some(ref m) => Some(matrix_to_pybytes(py, m)?),
            None => None,
        };
        Ok((xs_bytes, ys_bytes))
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(PyMatrixReturn<'py>, PyMatrixReturn<'py>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        let (xs, ys) = self.inner.fit_transform(&x_mat, &y_mat)?;
        Ok((matrix_to_pybytes(py, &xs)?, matrix_to_pybytes(py, &ys)?))
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<PyMatrixReturn<'py>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let pred = self.inner.predict(&x_mat)?;
        matrix_to_pybytes(py, &pred)
    }

    pub fn inverse_transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x_scores: PyBuffer<f64>,
    ) -> PyResult<PyMatrixReturn<'py>> {
        let x_mat = buffer_to_pooled_matrix(py, &x_scores)?;
        let rec = self.inner.inverse_transform(&x_mat)?;
        matrix_to_pybytes(py, &rec)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_mat)?)
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }
    #[getter]
    pub fn scale(&self) -> bool {
        self.inner.scale
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn x_weights_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_weights_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_weights_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_weights_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn x_loadings_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_loadings_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_loadings_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_loadings_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn x_rotations_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_rotations_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_rotations_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_rotations_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn coef_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .coef_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.inner
            .intercept_
            .as_ref()
            .map(|v| f64_slice_to_pybytes(py, v))
            .transpose()
    }
    #[getter]
    pub fn n_iter_(&self) -> Option<Vec<usize>> {
        self.inner.n_iter_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(from_py_object, name = "CCA")]
#[derive(Clone)]
pub struct PyCCA {
    inner: CCA,
}

pymethods_with_dit!(PyCCA, CCA, {
    #[new]
    #[pyo3(signature = (n_components = 2, scale = true, max_iter = 500, tol = 1e-6, device = None))]
    pub fn new(
        n_components: usize,
        scale: bool,
        max_iter: usize,
        tol: f64,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CCA::new(n_components, scale, max_iter, tol, device)?,
        })
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        self.inner.fit(&x_mat, &y_mat)?;
        Ok(())
    }

    #[pyo3(signature = (x, y = None))]
    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: Option<PyBuffer<f64>>,
    ) -> PyResult<(PyMatrixReturn<'py>, PyOptMatrixReturn<'py>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = match y {
            Some(ref yb) => Some(buffer_to_pooled_matrix(py, yb)?),
            None => None,
        };
        let (xs, ys) = self.inner.transform(&x_mat, y_mat.as_deref())?;
        let xs_bytes = matrix_to_pybytes(py, &xs)?;
        let ys_bytes = match ys {
            Some(ref m) => Some(matrix_to_pybytes(py, m)?),
            None => None,
        };
        Ok((xs_bytes, ys_bytes))
    }

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<(PyMatrixReturn<'py>, PyMatrixReturn<'py>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        let (xs, ys) = self.inner.fit_transform(&x_mat, &y_mat)?;
        Ok((matrix_to_pybytes(py, &xs)?, matrix_to_pybytes(py, &ys)?))
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<PyMatrixReturn<'py>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let pred = self.inner.predict(&x_mat)?;
        matrix_to_pybytes(py, &pred)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_mat = buffer_to_pooled_matrix(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_mat)?)
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }
    #[getter]
    pub fn scale(&self) -> bool {
        self.inner.scale
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn x_weights_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_weights_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_weights_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_weights_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn x_loadings_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_loadings_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_loadings_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_loadings_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn x_rotations_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .x_rotations_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn y_rotations_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .y_rotations_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn coef_<'py>(&self, py: Python<'py>) -> PyResult<PyOptMatrixReturn<'py>> {
        self.inner
            .coef_
            .as_ref()
            .map(|m| matrix_to_pybytes(py, m))
            .transpose()
    }
    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.inner
            .intercept_
            .as_ref()
            .map(|v| f64_slice_to_pybytes(py, v))
            .transpose()
    }
    #[getter]
    pub fn n_iter_(&self) -> Option<Vec<usize>> {
        self.inner.n_iter_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNMF>()?;
    m.add_class::<PyCCA>()?;
    m.add_class::<PyPLSRegression>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmf_basic_and_dit_roundtrip() {
        let x = Matrix::from_2d(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ])
        .unwrap();
        let mut nmf = NMF::new(
            Some(2),
            Some("nndsvda".to_string()),
            200,
            1e-4,
            0.0,
            0.0,
            None,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        let w = nmf.fit_transform(&x).unwrap();
        assert_eq!(w.rows, 4);
        assert_eq!(w.cols, 2);
        assert!(nmf.reconstruction_err_.unwrap() < 0.5);

        let recon = nmf.inverse_transform(&w).unwrap();
        assert_eq!(recon.rows, 4);
        assert_eq!(recon.cols, 3);

        let bytes = nmf.to_bytes();
        let loaded = NMF::from_bytes(&bytes, Some("cpu")).unwrap();
        let w_loaded = loaded.transform(&x).unwrap();
        assert_eq!(w_loaded.rows, 4);
        assert_eq!(w_loaded.cols, 2);
    }

    #[test]
    fn test_pls_and_cca_basic() {
        let x = Matrix::from_2d(&[
            vec![1.0, 2.0, 0.5],
            vec![2.0, 4.0, 1.0],
            vec![3.0, 6.0, 1.5],
            vec![4.0, 8.0, 2.0],
            vec![5.0, 10.0, 2.5],
        ])
        .unwrap();
        let y = Matrix::from_2d(&[
            vec![2.0, 1.0],
            vec![4.0, 2.0],
            vec![6.0, 3.0],
            vec![8.0, 4.0],
            vec![10.0, 5.0],
        ])
        .unwrap();

        let mut pls = PLSRegression::new(2, true, 500, 1e-6, Some("cpu".to_string())).unwrap();
        pls.fit(&x, &y).unwrap();
        let r2 = pls.score(&x, &y).unwrap();
        assert!(r2 > 0.99);

        let mut cca = CCA::new(2, true, 500, 1e-6, Some("cpu".to_string())).unwrap();
        cca.fit(&x, &y).unwrap();
        let (xs, ys) = cca.transform(&x, Some(&y)).unwrap();
        assert_eq!(xs.cols, 2);
        assert_eq!(ys.unwrap().cols, 2);
    }
}
