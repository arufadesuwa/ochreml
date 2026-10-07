// OchreML Decision Tree Module (＾▽＾)
// Implementation of CART algorithm for classification and regression trees.

use crate::matrix::{mean, r2_score, Matrix};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        value: f64,
        probabilities: Vec<(f64, f64)>, // (class_label, probability)
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if row[*feature_index] <= *threshold {
                    left.predict_row(row)
                } else {
                    right.predict_row(row)
                }
            }
        }
    }
}

// =========================================================================
// DECISION TREE CLASSIFIER
// =========================================================================

#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    pub criterion: String, // "gini" or "entropy"
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub root: Option<TreeNode>,
    pub classes_: Vec<f64>,
    pub n_features_in_: Option<usize>,
}

impl DecisionTreeClassifier {
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
    ) -> Self {
        Self {
            criterion: criterion.unwrap_or_else(|| "gini".to_string()).to_lowercase(),
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            root: None,
            classes_: Vec::new(),
            n_features_in_: None,
        }
    }

    // Compute impurity (Gini or Entropy) for a set of target labels
    fn compute_impurity(&self, labels: &[f64]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }

        let total = labels.len() as f64;
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &val in labels {
            *counts.entry((val * 1000.0).round() as i64).or_insert(0) += 1;
        }

        if self.criterion == "entropy" {
            let mut entropy = 0.0;
            for &count in counts.values() {
                let p = (count as f64) / total;
                if p > 0.0 {
                    entropy -= p * (p + 1e-15).log2();
                }
            }
            entropy
        } else {
            // Default: Gini Impurity = 1 - sum(p_i^2)
            let mut sum_sq = 0.0;
            for &count in counts.values() {
                let p = (count as f64) / total;
                sum_sq += p * p;
            }
            1.0 - sum_sq
        }
    }

    // Construct a leaf node by majority voting
    fn make_leaf(&self, labels: &[f64]) -> TreeNode {
        let mut counts: HashMap<i64, (f64, usize)> = HashMap::new();
        for &val in labels {
            let key = (val * 1000.0).round() as i64;
            let entry = counts.entry(key).or_insert((val, 0));
            entry.1 += 1;
        }

        let mut majority_val = if !labels.is_empty() { labels[0] } else { 0.0 };
        let mut max_count = 0;
        let total = labels.len() as f64;

        let mut probabilities = Vec::new();
        for (_, (val, count)) in counts {
            if count > max_count {
                max_count = count;
                majority_val = val;
            }
            if total > 0.0 {
                probabilities.push((val, (count as f64) / total));
            }
        }

        TreeNode::Leaf {
            value: majority_val,
            probabilities,
        }
    }

    // Recursively build decision tree structure
    fn build_tree(
        &self,
        x: &Matrix,
        y: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        let labels: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

        let max_depth_reached = self.max_depth.map_or(false, |md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;

        let current_impurity = self.compute_impurity(&labels);
        if current_impurity < 1e-9 || max_depth_reached || too_few_samples {
            return self.make_leaf(&labels);
        }

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left: Vec<usize> = Vec::new();
        let mut best_right: Vec<usize> = Vec::new();

        for f in 0..x.cols {
            let mut feature_vals: Vec<f64> = indices.iter().map(|&i| x.get(i, f)).collect();
            feature_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            feature_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

            if feature_vals.len() < 2 {
                continue;
            }

            for w in feature_vals.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;
                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();

                for &i in indices {
                    if x.get(i, f) <= threshold {
                        left_idx.push(i);
                    } else {
                        right_idx.push(i);
                    }
                }

                if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                    continue;
                }

                let left_labels: Vec<f64> = left_idx.iter().map(|&i| y[i]).collect();
                let right_labels: Vec<f64> = right_idx.iter().map(|&i| y[i]).collect();

                let imp_left = self.compute_impurity(&left_labels);
                let imp_right = self.compute_impurity(&right_labels);

                let p_left = (left_idx.len() as f64) / (n_samples as f64);
                let p_right = (right_idx.len() as f64) / (n_samples as f64);

                let gain = current_impurity - (p_left * imp_left + p_right * imp_right);
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = f;
                    best_threshold = threshold;
                    best_left = left_idx;
                    best_right = right_idx;
                }
            }
        }

        if best_gain <= 1e-9 || best_left.is_empty() || best_right.is_empty() {
            return self.make_leaf(&labels);
        }

        let left_child = self.build_tree(x, y, &best_left, depth + 1);
        let right_child = self.build_tree(x, y, &best_right, depth + 1);

        TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    // Fit DecisionTreeClassifier on feature matrix X and target labels y
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<(), String> {
        if x.rows != y.len() {
            return Err(format!(
                "Number of feature rows ({}) does not match target labels length ({}) ( >_< )",
                x.rows,
                y.len()
            ));
        }
        if x.rows == 0 || x.cols == 0 {
            return Err("Training dataset cannot be empty (´；ω；`)".to_string());
        }

        self.n_features_in_ = Some(x.cols);

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        self.classes_ = classes;

        let indices: Vec<usize> = (0..x.rows).collect();
        let root = self.build_tree(x, y, &indices, 0);
        self.root = Some(root);

        Ok(())
    }

    // Predict class label for new feature data X
    pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, String> {
        let root = match &self.root {
            Some(r) => r,
            None => return Err("Model is not fitted yet. Call fit() first (´-ω-`)".to_string()),
        };

        if let Some(expected_cols) = self.n_features_in_ {
            if x.cols != expected_cols {
                return Err(format!(
                    "Number of features ({}) does not match training features ({}) (・`ω´・)",
                    x.cols, expected_cols
                ));
            }
        }

        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row: Vec<f64> = (0..x.cols).map(|c| x.get(r, c)).collect();
            preds.push(root.predict_row(&row));
        }
        Ok(preds)
    }

    // Compute classification accuracy
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, String> {
        let preds = self.predict(x)?;
        if preds.len() != y.len() || preds.is_empty() {
            return Ok(0.0);
        }
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, actual)| (**p - **actual).abs() < 1e-9)
            .count();
        Ok((correct as f64) / (y.len() as f64))
    }
}

// =========================================================================
// DECISION TREE REGRESSOR
// =========================================================================

#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    pub criterion: String, // "squared_error" or "mse"
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub root: Option<TreeNode>,
    pub n_features_in_: Option<usize>,
}

impl DecisionTreeRegressor {
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
    ) -> Self {
        Self {
            criterion: criterion.unwrap_or_else(|| "squared_error".to_string()).to_lowercase(),
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            root: None,
            n_features_in_: None,
        }
    }

    // Compute variance / MSE of continuous target values
    fn compute_variance(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        let m = mean(values);
        let mut sum_sq = 0.0;
        for &v in values {
            sum_sq += (v - m).powi(2);
        }
        sum_sq / (values.len() as f64)
    }

    // Recursively build regression tree by minimizing variance
    fn build_tree(
        &self,
        x: &Matrix,
        y: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        let targets: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let current_variance = self.compute_variance(&targets);
        let node_mean = mean(&targets);

        let max_depth_reached = self.max_depth.map_or(false, |md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;

        if current_variance < 1e-9 || max_depth_reached || too_few_samples {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left: Vec<usize> = Vec::new();
        let mut best_right: Vec<usize> = Vec::new();

        for f in 0..x.cols {
            let mut feature_vals: Vec<f64> = indices.iter().map(|&i| x.get(i, f)).collect();
            feature_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            feature_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

            if feature_vals.len() < 2 {
                continue;
            }

            for w in feature_vals.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;
                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();

                for &i in indices {
                    if x.get(i, f) <= threshold {
                        left_idx.push(i);
                    } else {
                        right_idx.push(i);
                    }
                }

                if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                    continue;
                }

                let left_targets: Vec<f64> = left_idx.iter().map(|&i| y[i]).collect();
                let right_targets: Vec<f64> = right_idx.iter().map(|&i| y[i]).collect();

                let var_left = self.compute_variance(&left_targets);
                let var_right = self.compute_variance(&right_targets);

                let p_left = (left_idx.len() as f64) / (n_samples as f64);
                let p_right = (right_idx.len() as f64) / (n_samples as f64);

                let gain = current_variance - (p_left * var_left + p_right * var_right);
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = f;
                    best_threshold = threshold;
                    best_left = left_idx;
                    best_right = right_idx;
                }
            }
        }

        if best_gain <= 1e-9 || best_left.is_empty() || best_right.is_empty() {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let left_child = self.build_tree(x, y, &best_left, depth + 1);
        let right_child = self.build_tree(x, y, &best_right, depth + 1);

        TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    // Fit DecisionTreeRegressor on continuous dataset (X, y)
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<(), String> {
        if x.rows != y.len() {
            return Err(format!(
                "Number of feature rows ({}) does not match target length ({}) ( >_< )",
                x.rows,
                y.len()
            ));
        }
        if x.rows == 0 || x.cols == 0 {
            return Err("Training dataset cannot be empty (´；ω；`)".to_string());
        }

        self.n_features_in_ = Some(x.cols);
        let indices: Vec<usize> = (0..x.rows).collect();
        let root = self.build_tree(x, y, &indices, 0);
        self.root = Some(root);

        Ok(())
    }

    // Predict continuous target values for new data X
    pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, String> {
        let root = match &self.root {
            Some(r) => r,
            None => return Err("Model is not fitted yet. Call fit() first (´-ω-`)".to_string()),
        };

        if let Some(expected_cols) = self.n_features_in_ {
            if x.cols != expected_cols {
                return Err(format!(
                    "Number of features ({}) does not match training features ({}) (・`ω´・)",
                    x.cols, expected_cols
                ));
            }
        }

        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row: Vec<f64> = (0..x.cols).map(|c| x.get(r, c)).collect();
            preds.push(root.predict_row(&row));
        }
        Ok(preds)
    }

    // Compute R^2 score for regression performance
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, String> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_classifier() {
        let x = Matrix::from_2d(&vec![
            vec![0.5],
            vec![1.0],
            vec![1.5],
            vec![2.5],
            vec![3.0],
            vec![3.5],
        ]).unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut clf = DecisionTreeClassifier::new(Some("gini".to_string()), Some(3), None, None);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        assert_eq!(preds, y);

        let test_x = Matrix::from_2d(&vec![vec![0.8], vec![3.2]]).unwrap();
        let test_preds = clf.predict(&test_x).unwrap();
        assert_eq!(test_preds, vec![0.0, 1.0]);

        let acc = clf.score(&x, &y).unwrap();
        assert!((acc - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x = Matrix::from_2d(&vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
        ]).unwrap();
        let y = vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0];

        let mut reg = DecisionTreeRegressor::new(None, Some(3), None, None);
        reg.fit(&x, &y).unwrap();

        let preds = reg.predict(&x).unwrap();
        for (p, actual) in preds.iter().zip(y.iter()) {
            assert!((p - actual).abs() < 1e-4);
        }

        let test_x = Matrix::from_2d(&vec![vec![2.5], vec![7.5]]).unwrap();
        let test_preds = reg.predict(&test_x).unwrap();
        assert!((test_preds[0] - 10.0).abs() < 1e-4);
        assert!((test_preds[1] - 20.0).abs() < 1e-4);
    }
}
