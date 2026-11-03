use crate::device::{parse_device, DeviceType};
use crate::matrix::{
    compute_normal_equation_mats, compute_normal_equation_mats_multi, r2_score, r2_score_multi,
    solve_cholesky, solve_cholesky_multi, Matrix,
};

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
    pub fn new(fit_intercept: bool, device: Option<String>) -> Result<Self, String> {
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
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<(), String> {
        self.fit_weighted_1d(x, y, None)
    }

    // Fit 1D linear model with optional sample_weight
    pub fn fit_weighted_1d(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), String> {
        if x.rows != y.len() {
            return Err(format!(
                "Number of feature rows ({}) does not match target length ({}) ( >_< )",
                x.rows,
                y.len()
            ));
        }
        if x.rows == 0 || x.cols == 0 {
            return Err("Training data cannot be empty (´；ω；`)".to_string());
        }

        if sample_weight.is_some() {
            let y_mat = Matrix::from_slice(x.rows, 1, y)?;
            return self.fit_multi(x, &y_mat, sample_weight, false);
        }

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        self.is_multi_target_ = false;
        self.n_targets_ = 1;

        let (xt_x, xt_y) = compute_normal_equation_mats(x, y, self.fit_intercept, &self.device)?;
        let theta = solve_cholesky(&xt_x, &xt_y)?;

        let (coef_vec, intercept_val) = if self.fit_intercept {
            (theta[0..x.cols].to_vec(), theta[x.cols])
        } else {
            (theta, 0.0)
        };

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
    ) -> Result<(), String> {
        if x.rows != y.rows {
            return Err(format!(
                "Number of feature rows ({}) does not match target rows ({}) ( >_< )",
                x.rows, y.rows
            ));
        }
        if x.rows == 0 || x.cols == 0 || y.cols == 0 {
            return Err("Training data cannot be empty (´；ω；`)".to_string());
        }

        if y.cols == 1 && !is_2d_input && sample_weight.is_none() {
            return self.fit_weighted_1d(x, &y.data, None);
        }

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        self.is_multi_target_ = is_2d_input;
        self.n_targets_ = y.cols;

        let (xt_x, xt_y) = compute_normal_equation_mats_multi(
            x,
            y,
            sample_weight,
            self.fit_intercept,
            &self.device,
        )?;
        let theta = solve_cholesky_multi(&xt_x, &xt_y)?;

        let k_targets = y.cols;
        let mut coef_mat = Matrix::zeros(k_targets, x.cols);
        let mut intercept_vec = vec![0.0; k_targets];

        for t in 0..k_targets {
            for c in 0..x.cols {
                coef_mat.set(t, c, theta.get(c, t));
            }
            if self.fit_intercept {
                intercept_vec[t] = theta.get(x.cols, t);
            }
        }

        let first_coef = coef_mat.data[0..x.cols].to_vec();
        let first_intercept = intercept_vec[0];

        self.coef_ = Some(first_coef);
        self.intercept_ = Some(first_intercept);
        self.coef_multi_ = Some(coef_mat);
        self.intercept_multi_ = Some(intercept_vec);

        Ok(())
    }

    // Predict 1D continuous target values for new feature data X
    pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, String> {
        let coef = match &self.coef_ {
            Some(c) => c,
            None => return Err("Model is not fitted yet. Call fit() first (´-ω-`)".to_string()),
        };
        let intercept = self.intercept_.unwrap_or(0.0);

        if x.cols != coef.len() {
            return Err(format!(
                "Number of features in test data ({}) does not match model ({}) (・`ω´・)",
                x.cols,
                coef.len()
            ));
        }

        let mut predictions = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let offset = r * x.cols;
            let row_slice = &x.data[offset..offset + x.cols];
            let mut val = intercept;
            for c in 0..x.cols {
                val += row_slice[c] * coef[c];
            }
            predictions.push(val);
        }

        Ok(predictions)
    }

    // Predict multi-target continuous values of shape (n_samples, n_targets) (＾▽＾)
    pub fn predict_multi(&self, x: &Matrix) -> Result<Matrix, String> {
        let coef_mat = match &self.coef_multi_ {
            Some(c) => c,
            None => return Err("Model is not fitted yet. Call fit() first (´-ω-`)".to_string()),
        };
        let intercept_vec = match &self.intercept_multi_ {
            Some(v) => v,
            None => return Err("Model is not fitted yet. Call fit() first (´-ω-`)".to_string()),
        };

        if x.cols != coef_mat.cols {
            return Err(format!(
                "Number of features in test data ({}) does not match model ({}) (・`ω´・)",
                x.cols, coef_mat.cols
            ));
        }

        let k_targets = coef_mat.rows;
        let mut preds = Matrix::zeros(x.rows, k_targets);

        for r in 0..x.rows {
            let x_offset = r * x.cols;
            let x_row = &x.data[x_offset..x_offset + x.cols];
            let pred_offset = r * k_targets;
            for t in 0..k_targets {
                let c_offset = t * x.cols;
                let c_row = &coef_mat.data[c_offset..c_offset + x.cols];
                let mut val = intercept_vec[t];
                for c in 0..x.cols {
                    val += x_row[c] * c_row[c];
                }
                preds.data[pred_offset + t] = val;
            }
        }

        Ok(preds)
    }

    // Compute R^2 score on 1D test data
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, String> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    // Compute weighted multi-target R^2 score on test data
    pub fn score_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64, String> {
        if let Some(sw) = sample_weight {
            if sw.len() != x.rows {
                return Err(format!(
                    "Sample weight length ({}) must match test sample count ({}) ( >_< )",
                    sw.len(),
                    x.rows
                ));
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
        let x = Matrix::from_2d(&vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ]).unwrap();
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
        ]).unwrap();
        // Target 0: y0 = 2*x0 + 3*x1 + 1
        // Target 1: y1 = 4*x0 - 1*x1 + 5
        let y = Matrix::from_2d(&vec![
            vec![3.0, 9.0],
            vec![4.0, 4.0],
            vec![6.0, 8.0],
            vec![14.0, 10.0],
        ]).unwrap();
        let weights = vec![1.0, 2.0, 1.5, 3.0];

        let mut model = LinearRegression::new(true, None).unwrap();
        model.fit_multi(&x, &y, Some(&weights), true).unwrap();

        let coef_mat = model.coef_multi_.as_ref().unwrap();
        let intercept_vec = model.intercept_multi_.as_ref().unwrap();

        assert_eq!(coef_mat.rows, 2);
        assert_eq!(coef_mat.cols, 2);
        assert!((coef_mat.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((coef_mat.get(0, 1) - 3.0).abs() < 1e-6);
        assert!((intercept_vec[0] - 1.0).abs() < 1e-6);

        assert!((coef_mat.get(1, 0) - 4.0).abs() < 1e-6);
        assert!((coef_mat.get(1, 1) - (-1.0)).abs() < 1e-6);
        assert!((intercept_vec[1] - 5.0).abs() < 1e-6);

        let r2 = model.score_multi(&x, &y, Some(&weights)).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);
    }
}
