// OchreML Linear Regression Module (＾▽＾)
// Implementation of Ordinary Least Squares (OLS) via the Normal Equation.

use crate::matrix::{r2_score, solve_linear_system, Matrix};

#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub fit_intercept: bool,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub n_features_in_: Option<usize>,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
        }
    }

    // Fit linear model using Normal Equation: (X^T * X) * theta = X^T * y
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<(), String> {
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

        self.n_features_in_ = Some(x.cols);

        // Construct Design Matrix, augmenting with a bias column if fit_intercept is true
        let x_design = if self.fit_intercept {
            let mut aug = Matrix::zeros(x.rows, x.cols + 1);
            for r in 0..x.rows {
                for c in 0..x.cols {
                    aug.set(r, c, x.get(r, c));
                }
                aug.set(r, x.cols, 1.0);
            }
            aug
        } else {
            x.clone()
        };

        let xt = x_design.transpose();
        let xt_x = xt.matmul(&x_design)?;
        let xt_y = xt.matvec(y)?;

        let theta = solve_linear_system(&xt_x, &xt_y)?;

        if self.fit_intercept {
            self.coef_ = Some(theta[0..x.cols].to_vec());
            self.intercept_ = Some(theta[x.cols]);
        } else {
            self.coef_ = Some(theta);
            self.intercept_ = Some(0.0);
        }

        Ok(())
    }

    // Predict continuous target values for new feature data X
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
            let mut val = intercept;
            for c in 0..x.cols {
                val += x.get(r, c) * coef[c];
            }
            predictions.push(val);
        }

        Ok(predictions)
    }

    // Compute R^2 score on test data
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, String> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
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

        let mut model = LinearRegression::new(true);
        model.fit(&x, &y).unwrap();

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
}
