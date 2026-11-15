// OchreML Ridge Regression, Classification & Cross-Validation Module (＾▽＾)
// Implements Ridge, RidgeClassifier, and RidgeCV reusing Backend normal-equation primitives.

use crate::device::{DeviceType, parse_device};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{
    compute_and_solve_normal_1d, compute_and_solve_normal_multi, predict_linear_1d,
    predict_linear_multi, solve_from_normal_mats_multi,
};
use crate::matrix::{
    Matrix, compute_normal_equation_mats_multi, compute_normal_equation_multi_range,
    invert_spd_cholesky, r2_score, r2_score_multi,
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
        let mut y_ovr = Matrix {
            rows: x.rows,
            cols: n_classes,
            data: vec![-1.0; x.rows * n_classes],
        };
        for (r, &label) in y.iter().enumerate() {
            if let Some(cls_idx) = self.classes_.iter().position(|&c| (c - label).abs() < 1e-9) {
                y_ovr[(r, cls_idx)] = 1.0;
            }
        }

        let (coef_mat, intercept_vec) = compute_and_solve_normal_multi(
            x,
            &y_ovr,
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

#[inline]
fn quad_form_row(a_inv: &Matrix, x_row: &[f64], fit_intercept: bool) -> f64 {
    let d = x_row.len();
    let p = a_inv.cols;
    let mut sum = 0.0;
    for (i, &xi) in x_row.iter().enumerate() {
        let row_off = i * p;
        sum += xi * xi * a_inv.data[row_off + i];
        let mut off_sum = 0.0;
        for (&xj, &aij) in x_row[(i + 1)..d]
            .iter()
            .zip(&a_inv.data[(row_off + i + 1)..(row_off + d)])
        {
            off_sum += xj * aij;
        }
        sum += 2.0 * xi * off_sum;
        if fit_intercept {
            sum += 2.0 * xi * a_inv.data[row_off + d];
        }
    }
    if fit_intercept {
        sum += a_inv.data[d * p + d];
    }
    sum
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

fn compute_loo_score(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    xt_x: &Matrix,
    xt_y: &Matrix,
    alpha: f64,
) -> OchreResult<(f64, Matrix, Vec<f64>)> {
    let mut reg_xtx = xt_x.clone();
    reg_xtx.add_ridge_feature_diagonal(alpha, x.cols);

    let (coef_mat, intercept_vec) =
        solve_from_normal_mats_multi(&reg_xtx, xt_y, x.cols, fit_intercept, 0.0)?;
    let a_inv = invert_spd_cholesky(&reg_xtx)?;
    let preds = predict_linear_multi(x, &coef_mat, &intercept_vec)?;

    let mut sum_sq_looe = 0.0;
    for r in 0..x.rows {
        let w = sample_weight.map_or(1.0, |sw| sw[r]);
        if w == 0.0 {
            continue;
        }
        let x_off = r * x.cols;
        let x_row = &x.data[x_off..x_off + x.cols];
        let leverage = w * quad_form_row(&a_inv, x_row, fit_intercept);
        let raw_denom = 1.0 - leverage;
        let denom = if raw_denom.abs() < 1e-12 {
            if raw_denom < 0.0 { -1e-12 } else { 1e-12 }
        } else {
            raw_denom
        };
        let sqrt_w = w.sqrt();
        for t in 0..y.cols {
            let err = sqrt_w * (y[(r, t)] - preds[(r, t)]) / denom;
            sum_sq_looe += err * err;
        }
    }

    let score = -sum_sq_looe / ((x.rows * y.cols) as f64);
    Ok((score, coef_mat, intercept_vec))
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
        let mut best: Option<(f64, f64, Matrix, Vec<f64>)> = None;
        for &alpha in &self.alphas {
            let (score, coef_mat, intercept_vec) = compute_loo_score(
                x,
                y,
                sample_weight,
                self.fit_intercept,
                total_xt_x,
                total_xt_y,
                alpha,
            )?;
            let is_better = match &best {
                None => true,
                Some(cur) => score > cur.1,
            };
            if is_better {
                best = Some((alpha, score, coef_mat, intercept_vec));
            }
        }
        best.ok_or(OchreError::InvalidRidgeCVAlphas)
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
