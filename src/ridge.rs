// OchreML Ridge Regression, Classification & Cross-Validation Module (＾▽＾)
// Implements Ridge, RidgeClassifier, and RidgeCV reusing Backend normal-equation primitives.

use crate::device::{DeviceType, parse_device};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{
    compute_and_solve_normal_1d, compute_and_solve_normal_multi, dot_unrolled, predict_linear_1d,
    predict_linear_multi, solve_from_normal_mats_multi,
};
use crate::matrix::{
    Matrix, compute_normal_equation_mats, compute_normal_equation_mats_multi,
    compute_normal_equation_multi_range, r2_score, r2_score_multi,
};

fn validate_alpha(alpha: f64) -> OchreResult<()> {
    if alpha < 0.0 || !alpha.is_finite() {
        return Err(OchreError::InvalidAlpha(alpha));
    }
    Ok(())
}

fn validate_ridge_cv_params(alphas: &[f64], cv: Option<usize>) -> OchreResult<()> {
    if alphas.is_empty() {
        return Err(OchreError::InvalidRidgeCVAlphas);
    }
    if let Some(k) = cv {
        if k < 2 {
            return Err(OchreError::InvalidCvFolds(k));
        }
        for &a in alphas {
            if a < 0.0 || !a.is_finite() {
                return Err(OchreError::InvalidRidgeCVAlphas);
            }
        }
    } else {
        for &a in alphas {
            if a <= 0.0 || !a.is_finite() {
                return Err(OchreError::InvalidRidgeCVAlphas);
            }
        }
    }
    Ok(())
}

fn validate_sample_weight_len(sample_weight: Option<&[f64]>, n_samples: usize) -> OchreResult<()> {
    if let Some(sw) = sample_weight {
        if sw.len() != n_samples {
            return Err(OchreError::WeightLengthMismatch {
                weights: sw.len(),
                samples: n_samples,
            });
        }
        let mut weight_sum = 0.0;
        for &w in sw {
            if w < 0.0 || w.is_nan() {
                return Err(OchreError::InvalidWeights);
            }
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
    }
    Ok(())
}

/// L2-regularized Linear Regression (Tikhonov regularization) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct Ridge {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub is_multi_target_: bool,
    pub n_targets_: usize,
    pub n_features_in_: Option<usize>,
}

impl Ridge {
    /// Create a new Ridge regression estimator.
    pub fn new(alpha: f64, fit_intercept: bool, device: Option<String>) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            alpha,
            fit_intercept,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            is_multi_target_: false,
            n_targets_: 1,
            n_features_in_: None,
        })
    }

    /// Fit 1D Ridge regression model.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted_1d(x, y, None)
    }

    /// Fit 1D Ridge regression model with optional sample_weight.
    pub fn fit_weighted_1d(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }

        if sample_weight.is_some() {
            let y_mat = Matrix::from_slice(x.rows, 1, y)?;
            return self.fit_multi(x, &y_mat, sample_weight, false);
        }

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        self.is_multi_target_ = false;
        self.n_targets_ = 1;

        let (coef_vec, intercept_val) =
            compute_and_solve_normal_1d(x, y, self.fit_intercept, self.alpha, &self.device)?;

        self.coef_multi_ = Some(Matrix::from_vec(1, x.cols, coef_vec.clone())?);
        self.intercept_multi_ = Some(vec![intercept_val]);
        self.coef_ = Some(coef_vec);
        self.intercept_ = Some(intercept_val);

        Ok(())
    }

    /// Fit multi-output or weighted Ridge regression model.
    pub fn fit_multi(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        is_2d_input: bool,
    ) -> OchreResult<()> {
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        if x.rows == 0 || x.cols == 0 || y.cols == 0 {
            return Err(OchreError::EmptyData);
        }

        if y.cols == 1 && !is_2d_input && sample_weight.is_none() {
            return self.fit_weighted_1d(x, &y.data, None);
        }

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        self.is_multi_target_ = is_2d_input;
        self.n_targets_ = y.cols;

        let (coef_mat, intercept_vec) = compute_and_solve_normal_multi(
            x,
            y,
            sample_weight,
            self.fit_intercept,
            self.alpha,
            &self.device,
        )?;

        self.coef_ = Some(coef_mat.data[0..x.cols].to_vec());
        self.intercept_ = Some(intercept_vec[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercept_vec);

        Ok(())
    }

    /// Predict 1D target values for X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Predict multi-target values of shape (n_samples, n_targets).
    pub fn predict_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept_vec = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercept_vec)
    }

    /// Compute R^2 score on 1D test data.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    /// Compute weighted multi-target R^2 score on test data.
    pub fn score_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<f64> {
        validate_sample_weight_len(sample_weight, x.rows)?;
        let preds = self.predict_multi(x)?;
        Ok(r2_score_multi(y, &preds, sample_weight))
    }
}

/// Linear Classifier using L2-regularized Ridge Regression (o´∀｀o)
#[derive(Debug, Clone)]
pub struct RidgeClassifier {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Vec<f64>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub is_multiclass_: bool,
    pub n_features_in_: Option<usize>,
}

impl RidgeClassifier {
    /// Create a new RidgeClassifier estimator.
    pub fn new(alpha: f64, fit_intercept: bool, device: Option<String>) -> OchreResult<Self> {
        validate_alpha(alpha)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            alpha,
            fit_intercept,
            device: dev,
            device_: None,
            classes_: Vec::new(),
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            is_multiclass_: false,
            n_features_in_: None,
        })
    }

    /// Fit RidgeClassifier without sample weights.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted(x, y, None)
    }

    /// Fit RidgeClassifier with optional sample_weight.
    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        self.classes_ = classes;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());

        if self.classes_.len() <= 2 {
            self.fit_binary(x, y, sample_weight)
        } else {
            self.fit_multiclass(x, y, sample_weight)
        }
    }

    fn fit_binary(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        self.is_multiclass_ = false;
        let first_class = self.classes_[0];
        let y_bin: Vec<f64> = y
            .iter()
            .map(|&v| {
                if (v - first_class).abs() < 1e-9 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();

        let (coef_vec, intercept_val) = if sample_weight.is_some() {
            let y_mat = Matrix::from_vec(x.rows, 1, y_bin)?;
            let (cm, iv) = compute_and_solve_normal_multi(
                x,
                &y_mat,
                sample_weight,
                self.fit_intercept,
                self.alpha,
                &self.device,
            )?;
            (cm.data, iv[0])
        } else {
            compute_and_solve_normal_1d(x, &y_bin, self.fit_intercept, self.alpha, &self.device)?
        };

        self.coef_multi_ = Some(Matrix::from_vec(1, x.cols, coef_vec.clone())?);
        self.intercept_multi_ = Some(vec![intercept_val]);
        self.coef_ = Some(coef_vec);
        self.intercept_ = Some(intercept_val);
        Ok(())
    }

    fn fit_multiclass(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        self.is_multiclass_ = true;
        let n_classes = self.classes_.len();
        let d = x.cols;
        let p = if self.fit_intercept { d + 1 } else { d };

        let (coef_mat, intercept_vec) = if sample_weight.is_none() {
            let (xt_x, _) = compute_normal_equation_mats(x, y, self.fit_intercept, &self.device)?;
            let xt_y = build_ovr_xt_y_unweighted(x, y, &self.classes_, self.fit_intercept, p)?;
            match solve_from_normal_mats_multi(&xt_x, &xt_y, d, self.fit_intercept, self.alpha) {
                Ok(res) => res,
                Err(_) if self.device == DeviceType::Wgpu => {
                    let (cpu_xtx, _) =
                        compute_normal_equation_mats(x, y, self.fit_intercept, &DeviceType::Cpu)?;
                    solve_from_normal_mats_multi(
                        &cpu_xtx,
                        &xt_y,
                        d,
                        self.fit_intercept,
                        self.alpha,
                    )?
                }
                Err(e) => return Err(e),
            }
        } else {
            let mut y_ovr = Matrix {
                rows: x.rows,
                cols: n_classes,
                data: vec![-1.0; x.rows * n_classes],
            };
            for (r, &label) in y.iter().enumerate() {
                if let Some(cls_idx) = self.classes_.iter().position(|&c| (c - label).abs() < 1e-9)
                {
                    y_ovr[(r, cls_idx)] = 1.0;
                }
            }
            compute_and_solve_normal_multi(
                x,
                &y_ovr,
                sample_weight,
                self.fit_intercept,
                self.alpha,
                &self.device,
            )?
        };

        self.coef_ = Some(coef_mat.data[0..x.cols].to_vec());
        self.intercept_ = Some(intercept_vec[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercept_vec);
        Ok(())
    }

    /// Compute signed confidence scores for binary classification.
    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Compute one-vs-rest confidence scores of shape (n_samples, n_classes).
    pub fn decision_function_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept_vec = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercept_vec)
    }

    /// Predict class labels for samples in X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if self.classes_.is_empty() {
            return Err(OchreError::NotFitted);
        }
        if !self.is_multiclass_ {
            let scores = self.decision_function(x)?;
            if self.classes_.len() == 1 {
                return Ok(vec![self.classes_[0]; x.rows]);
            }
            let c0 = self.classes_[0];
            let c1 = self.classes_[1];
            return Ok(scores
                .into_iter()
                .map(|s| if s > 0.0 { c1 } else { c0 })
                .collect());
        }

        let scores = self.decision_function_multi(x)?;
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut best_idx = 0;
            let mut best_score = scores[(r, 0)];
            for c in 1..scores.cols {
                let s = scores[(r, c)];
                if s > best_score {
                    best_score = s;
                    best_idx = c;
                }
            }
            preds.push(self.classes_[best_idx]);
        }
        Ok(preds)
    }

    /// Compute classification accuracy, with optional sample_weight.
    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weight_len(sample_weight, x.rows)?;
        let preds = self.predict(x)?;
        if preds.is_empty() {
            return Ok(0.0);
        }

        match sample_weight {
            Some(sw) => {
                let mut weighted_correct = 0.0;
                let mut total_weight = 0.0;
                for ((p, actual), &w) in preds.iter().zip(y.iter()).zip(sw.iter()) {
                    total_weight += w;
                    if (*p - *actual).abs() < 1e-9 {
                        weighted_correct += w;
                    }
                }
                Ok(weighted_correct / total_weight)
            }
            None => {
                let correct = preds
                    .iter()
                    .zip(y.iter())
                    .filter(|(p, actual)| (**p - **actual).abs() < 1e-9)
                    .count();
                Ok((correct as f64) / (y.len() as f64))
            }
        }
    }
}

/// Ridge Regression with built-in Cross-Validation (LOO-CV / K-Fold) (*^▽^*)
#[derive(Debug, Clone)]
pub struct RidgeCV {
    pub alphas: Vec<f64>,
    pub fit_intercept: bool,
    pub cv: Option<usize>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub alpha_: Option<f64>,
    pub best_score_: Option<f64>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub is_multi_target_: bool,
    pub n_targets_: usize,
    pub n_features_in_: Option<usize>,
}

fn build_ovr_xt_y_unweighted(
    x: &Matrix,
    y: &[f64],
    classes: &[f64],
    fit_intercept: bool,
    p: usize,
) -> OchreResult<Matrix> {
    let d = x.cols;
    let k = classes.len();
    let (class_x_sum, class_count) = if x.rows >= 2048 {
        use rayon::prelude::*;
        x.data
            .par_chunks(512 * d)
            .enumerate()
            .map(|(chunk_idx, x_chunk)| {
                let start_r = chunk_idx * 512;
                let n_r = x_chunk.len() / d;
                let mut local_sum = vec![0.0; k * d];
                let mut local_cnt = vec![0.0; k];
                for i in 0..n_r {
                    let label = y[start_r + i];
                    if let Some(c) = classes.iter().position(|&cl| (cl - label).abs() < 1e-9) {
                        local_cnt[c] += 1.0;
                        let dst = &mut local_sum[(c * d)..(c * d + d)];
                        let src = &x_chunk[(i * d)..(i * d + d)];
                        for (dv, &sv) in dst.iter_mut().zip(src.iter()) {
                            *dv += sv;
                        }
                    }
                }
                (local_sum, local_cnt)
            })
            .reduce(
                || (vec![0.0; k * d], vec![0.0; k]),
                |(mut s_a, mut c_a), (s_b, c_b)| {
                    for (a, b) in s_a.iter_mut().zip(s_b.iter()) {
                        *a += *b;
                    }
                    for (a, b) in c_a.iter_mut().zip(c_b.iter()) {
                        *a += *b;
                    }
                    (s_a, c_a)
                },
            )
    } else {
        let mut local_sum = vec![0.0; k * d];
        let mut local_cnt = vec![0.0; k];
        for (r, &label) in y.iter().enumerate() {
            if let Some(c) = classes.iter().position(|&cl| (cl - label).abs() < 1e-9) {
                local_cnt[c] += 1.0;
                let dst = &mut local_sum[(c * d)..(c * d + d)];
                let src = &x.data[(r * d)..(r * d + d)];
                for (dv, &sv) in dst.iter_mut().zip(src.iter()) {
                    *dv += sv;
                }
            }
        }
        (local_sum, local_cnt)
    };

    let mut total_col_sum = vec![0.0; d];
    for c in 0..k {
        let c_slice = &class_x_sum[(c * d)..(c * d + d)];
        for (j, &v) in c_slice.iter().enumerate() {
            total_col_sum[j] += v;
        }
    }

    let mut xt_y = Matrix::zeros(p, k);
    for j in 0..d {
        let col_tot = total_col_sum[j];
        for c in 0..k {
            xt_y[(j, c)] = 2.0 * class_x_sum[c * d + j] - col_tot;
        }
    }
    if fit_intercept {
        let n_f = x.rows as f64;
        for c in 0..k {
            xt_y[(d, c)] = 2.0 * class_count[c] - n_f;
        }
    }
    Ok(xt_y)
}

// Cyclic Jacobi eigendecomposition of symmetric d x d matrix G_c = Q * diag(eig) * Q^T.
// Returns (eigenvalues, Q^T) where each row of Q^T is an orthonormal eigenvector.
fn sym_eigen_jacobi_transpose(g: &Matrix) -> (Vec<f64>, Matrix) {
    let d = g.rows;
    let mut a = g.data.clone();
    let mut q_t = vec![0.0; d * d];
    for i in 0..d {
        q_t[i * d + i] = 1.0;
    }

    for _sweep in 0..50 {
        let mut max_off = 0.0f64;
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                max_off = max_off.max(apq.abs());
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                if apq.abs() <= 1e-15 * (app.abs() + aqq.abs()) {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * d + p] = app - t * apq;
                a[q * d + q] = aqq + t * apq;
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;

                for k in 0..d {
                    if k != p && k != q {
                        let akp = a[k * d + p];
                        let akq = a[k * d + q];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;
                        a[k * d + p] = new_kp;
                        a[p * d + k] = new_kp;
                        a[k * d + q] = new_kq;
                        a[q * d + k] = new_kq;
                    }
                }
                for k in 0..d {
                    let qp = q_t[p * d + k];
                    let qq = q_t[q * d + k];
                    q_t[p * d + k] = c * qp - s * qq;
                    q_t[q * d + k] = s * qp + c * qq;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
    }

    let eig: Vec<f64> = (0..d).map(|i| a[i * d + i].max(0.0)).collect();
    (
        eig,
        Matrix {
            rows: d,
            cols: d,
            data: q_t,
        },
    )
}

#[inline]
fn project_z_sq_row(q_t: &[f64], q_mu: &[f64], x_row: &[f64], z_sq: &mut [f64]) {
    let d = x_row.len();
    let mut i = 0;
    while i + 4 <= d {
        let q0 = &q_t[(i * d)..(i * d + d)];
        let q1 = &q_t[((i + 1) * d)..((i + 1) * d + d)];
        let q2 = &q_t[((i + 2) * d)..((i + 2) * d + d)];
        let q3 = &q_t[((i + 3) * d)..((i + 3) * d + d)];
        let mut s0 = -q_mu[i];
        let mut s1 = -q_mu[i + 1];
        let mut s2 = -q_mu[i + 2];
        let mut s3 = -q_mu[i + 3];
        for j in 0..d {
            let xj = x_row[j];
            s0 += xj * q0[j];
            s1 += xj * q1[j];
            s2 += xj * q2[j];
            s3 += xj * q3[j];
        }
        z_sq[i] = s0 * s0;
        z_sq[i + 1] = s1 * s1;
        z_sq[i + 2] = s2 * s2;
        z_sq[i + 3] = s3 * s3;
        i += 4;
    }
    while i < d {
        let zi = dot_unrolled(&q_t[(i * d)..(i * d + d)], x_row) - q_mu[i];
        z_sq[i] = zi * zi;
        i += 1;
    }
}

fn slice_matrix_rows(m: &Matrix, start_row: usize, end_row: usize) -> Matrix {
    let rows = end_row - start_row;
    let cols = m.cols;
    let data = m.data[(start_row * cols)..(end_row * cols)].to_vec();
    Matrix { rows, cols, data }
}

fn subtract_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&va, &vb)| va - vb)
        .collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

fn evaluate_loo_batched_eigen(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    total_xt_x: &Matrix,
    total_xt_y: &Matrix,
    alphas: &[f64],
) -> OchreResult<(f64, f64, Matrix, Vec<f64>)> {
    let d = x.cols;
    let m_count = alphas.len();
    let k_targets = y.cols;

    let mut models = Vec::with_capacity(m_count);
    for &alpha in alphas {
        let mut reg_xtx = total_xt_x.clone();
        reg_xtx.add_ridge_feature_diagonal(alpha, d);
        let (coef_mat, intercept_vec) =
            solve_from_normal_mats_multi(&reg_xtx, total_xt_y, d, fit_intercept, 0.0)?;
        models.push((alpha, coef_mat, intercept_vec));
    }

    let (mu, g_c, c0) = if fit_intercept {
        let w_sum = total_xt_x[(d, d)];
        if w_sum <= 1e-12 {
            return Err(OchreError::NonPositiveWeightSum);
        }
        let mu_vec: Vec<f64> = (0..d).map(|j| total_xt_x[(j, d)] / w_sum).collect();
        let mut g_mat = Matrix::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                g_mat[(i, j)] = total_xt_x[(i, j)] - w_sum * mu_vec[i] * mu_vec[j];
            }
        }
        (mu_vec, g_mat, 1.0 / w_sum)
    } else {
        let mut g_mat = Matrix::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                g_mat[(i, j)] = total_xt_x[(i, j)];
            }
        }
        (vec![0.0; d], g_mat, 0.0)
    };

    let (eig, q_t) = sym_eigen_jacobi_transpose(&g_c);
    let q_mu: Vec<f64> = (0..d)
        .map(|i| dot_unrolled(&q_t.data[(i * d)..(i * d + d)], &mu))
        .collect();
    let inv_eigs: Vec<Vec<f64>> = alphas
        .iter()
        .map(|&a| (0..d).map(|i| 1.0 / (eig[i] + a)).collect())
        .collect();

    let eval_chunk = |start_r: usize, x_chunk: &[f64]| -> Vec<f64> {
        let n_r = x_chunk.len() / d;
        let mut local_err = vec![0.0; m_count];
        let mut z_sq = vec![0.0; d];
        for idx in 0..n_r {
            let r = start_r + idx;
            let w = sample_weight.map_or(1.0, |sw| sw[r]);
            if w == 0.0 {
                continue;
            }
            let x_row = &x_chunk[(idx * d)..(idx * d + d)];
            project_z_sq_row(&q_t.data, &q_mu, x_row, &mut z_sq);
            let sqrt_w = w.sqrt();
            for m in 0..m_count {
                let lev = w * (c0 + dot_unrolled(&z_sq, &inv_eigs[m]));
                let raw_denom = 1.0 - lev;
                let denom = if raw_denom.abs() < 1e-12 {
                    if raw_denom < 0.0 { -1e-12 } else { 1e-12 }
                } else {
                    raw_denom
                };
                let (_, ref coef_mat, ref intercept_vec) = models[m];
                for t in 0..k_targets {
                    let c_row = &coef_mat.data[(t * d)..(t * d + d)];
                    let pred = intercept_vec[t] + dot_unrolled(x_row, c_row);
                    let err = sqrt_w * (y[(r, t)] - pred) / denom;
                    local_err[m] += err * err;
                }
            }
        }
        local_err
    };

    let sum_sq_looe = if x.rows >= 1024 {
        use rayon::prelude::*;
        let block = 512;
        x.data
            .par_chunks(block * d)
            .enumerate()
            .map(|(c_idx, x_chunk)| eval_chunk(c_idx * block, x_chunk))
            .reduce(
                || vec![0.0; m_count],
                |mut a, b| {
                    for (va, vb) in a.iter_mut().zip(b.iter()) {
                        *va += *vb;
                    }
                    a
                },
            )
    } else {
        eval_chunk(0, &x.data)
    };

    let denom_scale = (x.rows * k_targets) as f64;
    let mut best: Option<(f64, f64, Matrix, Vec<f64>)> = None;
    for (m, (alpha, coef_mat, intercept_vec)) in models.into_iter().enumerate() {
        let score = -sum_sq_looe[m] / denom_scale;
        if best.as_ref().is_none_or(|cur| score > cur.1) {
            best = Some((alpha, score, coef_mat, intercept_vec));
        }
    }
    best.ok_or(OchreError::InvalidRidgeCVAlphas)
}

impl RidgeCV {
    /// Create a new RidgeCV estimator.
    pub fn new(
        alphas: Vec<f64>,
        fit_intercept: bool,
        cv: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        validate_ridge_cv_params(&alphas, cv)?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            alphas,
            fit_intercept,
            cv,
            device: dev,
            device_: None,
            alpha_: None,
            best_score_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            is_multi_target_: false,
            n_targets_: 1,
            n_features_in_: None,
        })
    }

    /// Fit 1D RidgeCV model.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted_1d(x, y, None)
    }

    /// Fit 1D RidgeCV model with optional sample_weight.
    pub fn fit_weighted_1d(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        let y_mat = Matrix::from_slice(x.rows, 1, y)?;
        self.fit_multi(x, &y_mat, sample_weight, false)
    }

    /// Fit multi-output or weighted RidgeCV model computing (X^T W X, X^T W Y) once in Backend.
    pub fn fit_multi(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        is_2d_input: bool,
    ) -> OchreResult<()> {
        if x.rows != y.rows {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.rows,
            });
        }
        if x.rows == 0 || x.cols == 0 || y.cols == 0 {
            return Err(OchreError::EmptyData);
        }

        let (total_xt_x, total_xt_y) = compute_normal_equation_mats_multi(
            x,
            y,
            sample_weight,
            self.fit_intercept,
            &self.device,
        )?;

        let eval_res = self.evaluate_candidates(x, y, sample_weight, &total_xt_x, &total_xt_y);
        let (best_alpha, best_score, coef_mat, intercept_vec) = match eval_res {
            Ok(res) => res,
            Err(_) if self.device == DeviceType::Wgpu => {
                let (cpu_xtx, cpu_xty) = compute_normal_equation_mats_multi(
                    x,
                    y,
                    sample_weight,
                    self.fit_intercept,
                    &DeviceType::Cpu,
                )?;
                self.evaluate_candidates(x, y, sample_weight, &cpu_xtx, &cpu_xty)?
            }
            Err(e) => return Err(e),
        };

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        self.is_multi_target_ = is_2d_input;
        self.n_targets_ = y.cols;
        self.alpha_ = Some(best_alpha);
        self.best_score_ = Some(best_score);
        self.coef_ = Some(coef_mat.data[0..x.cols].to_vec());
        self.intercept_ = Some(intercept_vec[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercept_vec);

        Ok(())
    }

    fn evaluate_candidates(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        total_xt_x: &Matrix,
        total_xt_y: &Matrix,
    ) -> OchreResult<(f64, f64, Matrix, Vec<f64>)> {
        match self.cv {
            None => self.evaluate_loo_candidates(x, y, sample_weight, total_xt_x, total_xt_y),
            Some(k) => {
                self.evaluate_kfold_candidates(x, y, sample_weight, total_xt_x, total_xt_y, k)
            }
        }
    }

    fn evaluate_loo_candidates(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        total_xt_x: &Matrix,
        total_xt_y: &Matrix,
    ) -> OchreResult<(f64, f64, Matrix, Vec<f64>)> {
        evaluate_loo_batched_eigen(
            x,
            y,
            sample_weight,
            self.fit_intercept,
            total_xt_x,
            total_xt_y,
            &self.alphas,
        )
    }

    fn evaluate_kfold_candidates(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        total_xt_x: &Matrix,
        total_xt_y: &Matrix,
        k: usize,
    ) -> OchreResult<(f64, f64, Matrix, Vec<f64>)> {
        if k > x.rows {
            return Err(OchreError::InvalidCvFolds(k));
        }

        let base = x.rows / k;
        let rem = x.rows % k;
        let mut start = 0;
        let mut folds = Vec::with_capacity(k);

        for f in 0..k {
            let len = base + usize::from(f < rem);
            let end = start + len;
            let (fold_xtx, fold_xty) = compute_normal_equation_multi_range(
                x,
                y,
                sample_weight,
                self.fit_intercept,
                start,
                end,
            )?;
            let x_val = slice_matrix_rows(x, start, end);
            let y_val = slice_matrix_rows(y, start, end);
            folds.push((x_val, y_val, fold_xtx, fold_xty));
            start = end;
        }

        let mut best_alpha_score: Option<(f64, f64)> = None;
        for &alpha in &self.alphas {
            let mut score_sum = 0.0;
            for (x_val, y_val, fold_xtx, fold_xty) in &folds {
                let train_xtx = subtract_matrix(total_xt_x, fold_xtx);
                let train_xty = subtract_matrix(total_xt_y, fold_xty);
                let (coef_mat, intercept_vec) = solve_from_normal_mats_multi(
                    &train_xtx,
                    &train_xty,
                    x.cols,
                    self.fit_intercept,
                    alpha,
                )?;
                let val_preds = predict_linear_multi(x_val, &coef_mat, &intercept_vec)?;
                score_sum += r2_score_multi(y_val, &val_preds, None);
            }
            let mean_score = score_sum / (k as f64);
            let is_better = match best_alpha_score {
                None => true,
                Some((_, cur_best)) => mean_score > cur_best,
            };
            if is_better {
                best_alpha_score = Some((alpha, mean_score));
            }
        }

        let (best_alpha, best_score) = best_alpha_score.ok_or(OchreError::InvalidRidgeCVAlphas)?;
        let (coef_mat, intercept_vec) = solve_from_normal_mats_multi(
            total_xt_x,
            total_xt_y,
            x.cols,
            self.fit_intercept,
            best_alpha,
        )?;
        Ok((best_alpha, best_score, coef_mat, intercept_vec))
    }

    /// Predict 1D target values for X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    /// Predict multi-target values of shape (n_samples, n_targets).
    pub fn predict_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept_vec = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercept_vec)
    }

    /// Compute R^2 score on 1D test data.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    /// Compute weighted multi-target R^2 score on test data.
    pub fn score_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<f64> {
        validate_sample_weight_len(sample_weight, x.rows)?;
        let preds = self.predict_multi(x)?;
        Ok(r2_score_multi(y, &preds, sample_weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_unpenalized_intercept() {
        // Zero-mean features with non-zero target mean: intercept must remain exact even for huge alpha.
        let x =
            Matrix::from_2d(&[vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]]).unwrap();
        let y = vec![7.0, 9.0, 11.0, 13.0, 15.0]; // y = 2*x + 11

        let mut model = Ridge::new(1000.0, true, Some("cpu".to_string())).unwrap();
        model.fit(&x, &y).unwrap();

        let intercept = model.intercept_.unwrap();
        let coef = model.coef_.as_ref().unwrap();
        assert!((intercept - 11.0).abs() < 1e-10);
        assert!(coef[0] > 0.0 && coef[0] < 0.05);
    }

    #[test]
    fn test_ridge_1d_and_multi_output_weighted() {
        let x = Matrix::from_2d(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ])
        .unwrap();
        let y = Matrix::from_2d(&[
            vec![3.0, 9.0],
            vec![4.0, 4.0],
            vec![6.0, 8.0],
            vec![14.0, 10.0],
        ])
        .unwrap();
        let weights = vec![1.0, 2.0, 1.5, 3.0];

        // With alpha = 0.0, Ridge must match exact unregularized solution.
        let mut model = Ridge::new(0.0, true, None).unwrap();
        model.fit_multi(&x, &y, Some(&weights), true).unwrap();

        let coef_mat = model.coef_multi_.as_ref().unwrap();
        let intercept_vec = model.intercept_multi_.as_ref().unwrap();
        assert!((coef_mat[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((coef_mat[(0, 1)] - 3.0).abs() < 1e-6);
        assert!((intercept_vec[0] - 1.0).abs() < 1e-6);
        assert!((coef_mat[(1, 0)] - 4.0).abs() < 1e-6);
        assert!((coef_mat[(1, 1)] + 1.0).abs() < 1e-6);
        assert!((intercept_vec[1] - 5.0).abs() < 1e-6);

        let r2 = model.score_multi(&x, &y, Some(&weights)).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_classifier_binary_and_multiclass() {
        let x_bin = Matrix::from_2d(&[
            vec![-3.0, -1.0],
            vec![-2.0, -2.0],
            vec![2.0, 1.0],
            vec![3.0, 2.0],
        ])
        .unwrap();
        let y_bin = vec![0.0, 0.0, 1.0, 1.0];

        let mut clf_bin = RidgeClassifier::new(1.0, true, Some("cpu".to_string())).unwrap();
        clf_bin.fit(&x_bin, &y_bin).unwrap();
        assert_eq!(clf_bin.classes_, vec![0.0, 1.0]);
        assert!(!clf_bin.is_multiclass_);
        assert!((clf_bin.score(&x_bin, &y_bin, None).unwrap() - 1.0).abs() < 1e-9);
        let df_bin = clf_bin.decision_function(&x_bin).unwrap();
        assert!(df_bin[0] < 0.0 && df_bin[2] > 0.0);

        let x_mc = Matrix::from_2d(&[
            vec![3.0, 0.0],
            vec![2.5, 0.5],
            vec![0.0, 3.0],
            vec![0.5, 2.5],
            vec![-3.0, -3.0],
            vec![-2.5, -2.5],
        ])
        .unwrap();
        let y_mc = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let mut clf_mc = RidgeClassifier::new(0.5, true, Some("cpu".to_string())).unwrap();
        clf_mc.fit(&x_mc, &y_mc).unwrap();
        assert_eq!(clf_mc.classes_, vec![0.0, 1.0, 2.0]);
        assert!(clf_mc.is_multiclass_);
        assert_eq!(clf_mc.predict(&x_mc).unwrap(), y_mc);
        assert!((clf_mc.score(&x_mc, &y_mc, None).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ridge_cv_loo_and_kfold() {
        // Noisy collinear-free linear relationship where small alpha is optimal.
        let x = Matrix::from_2d(&[
            vec![1.0, 0.2],
            vec![2.0, 0.5],
            vec![3.0, 0.1],
            vec![4.0, 0.9],
            vec![5.0, 0.4],
            vec![6.0, 1.2],
            vec![7.0, 0.7],
            vec![8.0, 1.5],
        ])
        .unwrap();
        let y: Vec<f64> = (0..8)
            .map(|r| 3.0 * x[(r, 0)] - 2.0 * x[(r, 1)] + 1.0)
            .collect();

        let mut rcv_loo =
            RidgeCV::new(vec![1e-4, 1.0, 100.0], true, None, Some("cpu".to_string())).unwrap();
        rcv_loo.fit(&x, &y).unwrap();
        assert_eq!(rcv_loo.alpha_, Some(1e-4));
        assert!(rcv_loo.best_score_.unwrap() > -1e-4);
        assert!((rcv_loo.intercept_.unwrap() - 1.0).abs() < 1e-2);

        let mut rcv_kfold = RidgeCV::new(
            vec![1e-4, 1.0, 100.0],
            true,
            Some(4),
            Some("cpu".to_string()),
        )
        .unwrap();
        rcv_kfold.fit(&x, &y).unwrap();
        assert_eq!(rcv_kfold.alpha_, Some(1e-4));
        assert!(rcv_kfold.best_score_.unwrap() > 0.999);
    }
}
