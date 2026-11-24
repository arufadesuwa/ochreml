// OchreML Linear Regression Module (＾▽＾)
// Implements OLS LinearRegression and shared normal-equation solvers/predictors.

use crate::device::{DeviceType, parse_device};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{
    Matrix, compute_normal_equation_mats, compute_normal_equation_mats_multi, r2_score,
    r2_score_multi, solve_cholesky, solve_cholesky_multi,
};
use std::borrow::Cow;

pub(crate) fn solve_from_normal_mats_1d(
    xt_x: &Matrix,
    xt_y: &[f64],
    n_features: usize,
    fit_intercept: bool,
    alpha: f64,
) -> OchreResult<(Vec<f64>, f64)> {
    let mat: Cow<'_, Matrix> = if alpha > 0.0 {
        let mut reg_xtx = xt_x.clone();
        reg_xtx.add_ridge_feature_diagonal(alpha, n_features);
        Cow::Owned(reg_xtx)
    } else {
        Cow::Borrowed(xt_x)
    };

    let theta = solve_cholesky(&mat, xt_y)?;
    if fit_intercept {
        Ok((theta[0..n_features].to_vec(), theta[n_features]))
    } else {
        Ok((theta, 0.0))
    }
}

pub(crate) fn solve_from_normal_mats_multi(
    xt_x: &Matrix,
    xt_y: &Matrix,
    n_features: usize,
    fit_intercept: bool,
    alpha: f64,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let mat: Cow<'_, Matrix> = if alpha > 0.0 {
        let mut reg_xtx = xt_x.clone();
        reg_xtx.add_ridge_feature_diagonal(alpha, n_features);
        Cow::Owned(reg_xtx)
    } else {
        Cow::Borrowed(xt_x)
    };

    let theta = solve_cholesky_multi(&mat, xt_y)?;
    let k_targets = xt_y.cols;
    let mut coef_mat = Matrix::zeros(k_targets, n_features);
    let mut intercept_vec = vec![0.0; k_targets];

    for t in 0..k_targets {
        for c in 0..n_features {
            coef_mat[(t, c)] = theta[(c, t)];
        }
        if fit_intercept {
            intercept_vec[t] = theta[(n_features, t)];
        }
    }

    Ok((coef_mat, intercept_vec))
}

pub(crate) fn compute_and_solve_normal_1d(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    alpha: f64,
    device: &DeviceType,
) -> OchreResult<(Vec<f64>, f64)> {
    let (xt_x, xt_y) = compute_normal_equation_mats(x, y, fit_intercept, device)?;
    match solve_from_normal_mats_1d(&xt_x, &xt_y, x.cols, fit_intercept, alpha) {
        Ok(res) => Ok(res),
        Err(_) if *device == DeviceType::Wgpu => {
            let (cpu_xtx, cpu_xty) =
                compute_normal_equation_mats(x, y, fit_intercept, &DeviceType::Cpu)?;
            solve_from_normal_mats_1d(&cpu_xtx, &cpu_xty, x.cols, fit_intercept, alpha)
        }
        Err(e) => Err(e),
    }
}

pub(crate) fn compute_and_solve_normal_multi(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    alpha: f64,
    device: &DeviceType,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let (xt_x, xt_y) =
        compute_normal_equation_mats_multi(x, y, sample_weight, fit_intercept, device)?;
    match solve_from_normal_mats_multi(&xt_x, &xt_y, x.cols, fit_intercept, alpha) {
        Ok(res) => Ok(res),
        Err(_) if *device == DeviceType::Wgpu => {
            let (cpu_xtx, cpu_xty) = compute_normal_equation_mats_multi(
                x,
                y,
                sample_weight,
                fit_intercept,
                &DeviceType::Cpu,
            )?;
            solve_from_normal_mats_multi(&cpu_xtx, &cpu_xty, x.cols, fit_intercept, alpha)
        }
        Err(e) => Err(e),
    }
}

#[inline(always)]
pub(crate) fn dot_unrolled(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f64; 4];
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        for lane in 0..4 {
            acc[lane] += a[base + lane] * b[base + lane];
        }
    }
    let mut tail = 0.0;
    for i in (chunks * 4)..n {
        tail += a[i] * b[i];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

pub(crate) fn predict_linear_1d(x: &Matrix, coef: &[f64], intercept: f64) -> OchreResult<Vec<f64>> {
    if x.cols != coef.len() {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: coef.len(),
        });
    }

    let d = x.cols;
    if x.rows >= 2048 {
        use rayon::prelude::*;
        let mut predictions = vec![0.0; x.rows];
        predictions
            .par_chunks_mut(512)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_r = chunk_idx * 512;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let r = start_r + i;
                    let row_slice = &x.data[(r * d)..(r * d + d)];
                    *out_val = intercept + dot_unrolled(row_slice, coef);
                }
            });
        return Ok(predictions);
    }

    let mut predictions = Vec::with_capacity(x.rows);
    for r in 0..x.rows {
        let row_slice = &x.data[(r * d)..(r * d + d)];
        predictions.push(intercept + dot_unrolled(row_slice, coef));
    }

    Ok(predictions)
}

pub(crate) fn predict_linear_multi(
    x: &Matrix,
    coef_mat: &Matrix,
    intercept_vec: &[f64],
) -> OchreResult<Matrix> {
    if x.cols != coef_mat.cols {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: coef_mat.cols,
        });
    }
    if intercept_vec.len() != coef_mat.rows {
        return Err(OchreError::VectorLengthMismatch);
    }

    let d = x.cols;
    let k_targets = coef_mat.rows;
    let mut preds = Matrix::zeros(x.rows, k_targets);

    if x.rows >= 2048 {
        use rayon::prelude::*;
        let row_block = 512;
        preds
            .data
            .par_chunks_mut(row_block * k_targets)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_r = chunk_idx * row_block;
                let n_rows_chunk = out_chunk.len() / k_targets;
                for i in 0..n_rows_chunk {
                    let r = start_r + i;
                    let x_row = &x.data[(r * d)..(r * d + d)];
                    let pred_off = i * k_targets;
                    for (t, &intercept_t) in intercept_vec.iter().enumerate() {
                        let c_row = &coef_mat.data[(t * d)..(t * d + d)];
                        out_chunk[pred_off + t] = intercept_t + dot_unrolled(x_row, c_row);
                    }
                }
            });
        return Ok(preds);
    }

    for r in 0..x.rows {
        let x_row = &x.data[(r * d)..(r * d + d)];
        let pred_offset = r * k_targets;
        for (t, &intercept_t) in intercept_vec.iter().enumerate() {
            let c_row = &coef_mat.data[(t * d)..(t * d + d)];
            preds.data[pred_offset + t] = intercept_t + dot_unrolled(x_row, c_row);
        }
    }

    Ok(preds)
}

#[derive(Debug, Clone)]
pub struct LinearRegression {
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

impl LinearRegression {
    pub fn new(fit_intercept: bool, device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
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

    // Fit 1D linear model using Normal Equation: (X^T * X) * theta = X^T * y
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        self.fit_weighted_1d(x, y, None)
    }

    // Fit 1D linear model with optional sample_weight
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
        if x.data.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
            return Err(OchreError::InvalidParameter(
                "Input X or y contains NaN or Inf values ( >_< )".to_string(),
            ));
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
            compute_and_solve_normal_1d(x, y, self.fit_intercept, 0.0, &self.device)?;

        self.coef_multi_ = Some(Matrix::from_vec(1, x.cols, coef_vec.clone())?);
        self.intercept_multi_ = Some(vec![intercept_val]);
        self.coef_ = Some(coef_vec);
        self.intercept_ = Some(intercept_val);

        Ok(())
    }

    // Fit multi-output / weighted linear model: (X^T * W * X) * Theta = X^T * W * Y (*≧ω≦*)
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
        if x.data.iter().any(|v| !v.is_finite()) || y.data.iter().any(|v| !v.is_finite()) {
            return Err(OchreError::InvalidParameter(
                "Input X or y contains NaN or Inf values ( >_< )".to_string(),
            ));
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
            0.0,
            &self.device,
        )?;

        self.coef_ = Some(coef_mat.data[0..x.cols].to_vec());
        self.intercept_ = Some(intercept_vec[0]);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercept_vec);

        Ok(())
    }

    // Predict 1D continuous target values for new feature data X
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        predict_linear_1d(x, coef, intercept)
    }

    // Predict multi-target continuous values of shape (n_samples, n_targets) (＾▽＾)
    pub fn predict_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept_vec = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercept_vec)
    }

    // Compute R^2 score on 1D test data
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    // Compute weighted multi-target R^2 score on test data
    pub fn score_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<f64> {
        if let Some(sw) = sample_weight {
            if sw.len() != x.rows {
                return Err(OchreError::WeightLengthMismatch {
                    weights: sw.len(),
                    samples: x.rows,
                });
            }
        }
        let preds = self.predict_multi(x)?;
        Ok(r2_score_multi(y, &preds, sample_weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_simple() {
        let x =
            Matrix::from_2d(&vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut model = LinearRegression::new(true, None).unwrap();
        model.fit(&x, &y).unwrap();
        assert!(model.device_.is_some());

        let coef = model.coef_.as_ref().unwrap();
        let intercept = model.intercept_.unwrap();

        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!((intercept - 1.0).abs() < 1e-4);

        let test_x = Matrix::from_2d(&vec![vec![6.0], vec![7.0]]).unwrap();
        let preds = model.predict(&test_x).unwrap();
        assert!((preds[0] - 13.0).abs() < 1e-4);
        assert!((preds[1] - 15.0).abs() < 1e-4);

        let score = model.score(&x, &y).unwrap();
        assert!((score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_linear_regression_multi_output_and_weights() {
        let x = Matrix::from_2d(&vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ])
        .unwrap();
        // Target 0: y0 = 2*x0 + 3*x1 + 1
        // Target 1: y1 = 4*x0 - 1*x1 + 5
        let y = Matrix::from_2d(&vec![
            vec![3.0, 9.0],
            vec![4.0, 4.0],
            vec![6.0, 8.0],
            vec![14.0, 10.0],
        ])
        .unwrap();
        let weights = vec![1.0, 2.0, 1.5, 3.0];

        let mut model = LinearRegression::new(true, None).unwrap();
        model.fit_multi(&x, &y, Some(&weights), true).unwrap();

        let coef_mat = model.coef_multi_.as_ref().unwrap();
        let intercept_vec = model.intercept_multi_.as_ref().unwrap();

        assert_eq!(coef_mat.rows, 2);
        assert_eq!(coef_mat.cols, 2);
        assert!((coef_mat[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((coef_mat[(0, 1)] - 3.0).abs() < 1e-6);
        assert!((intercept_vec[0] - 1.0).abs() < 1e-6);

        assert!((coef_mat[(1, 0)] - 4.0).abs() < 1e-6);
        assert!((coef_mat[(1, 1)] - (-1.0)).abs() < 1e-6);
        assert!((intercept_vec[1] - 5.0).abs() < 1e-6);

        let r2 = model.score_multi(&x, &y, Some(&weights)).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);
    }
}
