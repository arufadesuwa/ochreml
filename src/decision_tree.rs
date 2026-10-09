// OchreML Decision Tree Module (＾▽＾)
// Implementation of CART algorithm for classification and regression trees.

use crate::device::{parse_device, DeviceType};
use crate::matrix::{r2_score, Matrix};

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
    pub device: DeviceType,
    pub device_: Option<String>,
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
        device: Option<String>,
    ) -> Result<Self, String> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            criterion: criterion.unwrap_or_else(|| "gini".to_string()).to_lowercase(),
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            device: dev,
            device_: None,
            root: None,
            classes_: Vec::new(),
            n_features_in_: None,
        })
    }

    // Compute impurity (Gini or Entropy) from class frequency counts in O(C)
    fn compute_impurity_from_counts(&self, counts: &[usize], total_samples: usize) -> f64 {
        if total_samples == 0 {
            return 0.0;
        }

        let total = total_samples as f64;
        if self.criterion == "entropy" {
            let mut entropy = 0.0;
            for &count in counts {
                if count > 0 {
                    let p = (count as f64) / total;
                    entropy -= p * (p + 1e-15).log2();
                }
            }
            entropy
        } else {
            // Default: Gini Impurity = 1 - sum(p_i^2)
            let mut sum_sq = 0.0;
            for &count in counts {
                let p = (count as f64) / total;
                sum_sq += p * p;
            }
            1.0 - sum_sq
        }
    }

    // Construct a leaf node from class frequency counts
    fn make_leaf_from_counts(&self, counts: &[usize], total_samples: usize) -> TreeNode {
        let mut majority_val = self.classes_.first().copied().unwrap_or(0.0);
        let mut max_count = 0;
        let total = total_samples as f64;

        let mut probabilities = Vec::with_capacity(counts.len());
        for (i, &count) in counts.iter().enumerate() {
            if count > max_count {
                max_count = count;
                majority_val = self.classes_[i];
            }
            let prob = if total > 0.0 {
                (count as f64) / total
            } else {
                0.0
            };
            probabilities.push((self.classes_[i], prob));
        }

        TreeNode::Leaf {
            value: majority_val,
            probabilities,
        }
    }

    // Recursively build decision tree structure with O(D * N log N) sliding split search
    fn build_tree(
        &self,
        x: &Matrix,
        class_indices: &[usize],
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        let n_classes = self.classes_.len();
        let mut total_counts = vec![0usize; n_classes];
        for &i in indices {
            total_counts[class_indices[i]] += 1;
        }

        let max_depth_reached = self.max_depth.map_or(false, |md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;
        let current_impurity = self.compute_impurity_from_counts(&total_counts, n_samples);

        if current_impurity < 1e-9 || max_depth_reached || too_few_samples {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(n_samples);

        for f in 0..x.cols {
            pairs.clear();
            for &i in indices {
                pairs.push((x.get(i, f), class_indices[i]));
            }
            pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if (pairs[n_samples - 1].0 - pairs[0].0).abs() < 1e-9 {
                continue;
            }

            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = total_counts.clone();
            let mut n_left = 0usize;
            let mut n_right = n_samples;

            for i in 0..(n_samples - 1) {
                let cls = pairs[i].1;
                left_counts[cls] += 1;
                right_counts[cls] -= 1;
                n_left += 1;
                n_right -= 1;

                let cur_val = pairs[i].0;
                let next_val = pairs[i + 1].0;
                if (next_val - cur_val).abs() > 1e-9 {
                    if n_left >= self.min_samples_leaf && n_right >= self.min_samples_leaf {
                        let imp_left = self.compute_impurity_from_counts(&left_counts, n_left);
                        let imp_right = self.compute_impurity_from_counts(&right_counts, n_right);

                        let p_left = (n_left as f64) / (n_samples as f64);
                        let p_right = (n_right as f64) / (n_samples as f64);

                        let gain = current_impurity - (p_left * imp_left + p_right * imp_right);
                        if gain > best_gain {
                            best_gain = gain;
                            best_feature = f;
                            best_threshold = (cur_val + next_val) / 2.0;
                        }
                    }
                }
            }
        }

        if best_gain <= 1e-9 {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let mut best_left = Vec::with_capacity(n_samples / 2);
        let mut best_right = Vec::with_capacity(n_samples / 2);
        for &i in indices {
            if x.get(i, best_feature) <= best_threshold {
                best_left.push(i);
            } else {
                best_right.push(i);
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let left_child = self.build_tree(x, class_indices, &best_left, depth + 1);
        let right_child = self.build_tree(x, class_indices, &best_right, depth + 1);

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
        self.device_ = Some(self.device.to_string());

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        self.classes_ = classes;

        let class_indices: Vec<usize> = y
            .iter()
            .map(|&val| {
                self.classes_
                    .iter()
                    .position(|&c| (c - val).abs() < 1e-9)
                    .unwrap_or(0)
            })
            .collect();

        let indices: Vec<usize> = (0..x.rows).collect();
        let root = self.build_tree(x, &class_indices, &indices, 0);
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
            let offset = r * x.cols;
            let row_slice = &x.data[offset..offset + x.cols];
            preds.push(root.predict_row(row_slice));
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
    pub device: DeviceType,
    pub device_: Option<String>,
    pub root: Option<TreeNode>,
    pub n_features_in_: Option<usize>,
}

impl DecisionTreeRegressor {
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> Result<Self, String> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            criterion: criterion.unwrap_or_else(|| "squared_error".to_string()).to_lowercase(),
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            device: dev,
            device_: None,
            root: None,
            n_features_in_: None,
        })
    }

    // Recursively build regression tree with O(D * N log N) sliding split search and O(1) MSE updates
    fn build_tree(
        &self,
        x: &Matrix,
        y: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        let mut total_sum = 0.0;
        let mut total_sum_sq = 0.0;
        for &i in indices {
            let yi = y[i];
            total_sum += yi;
            total_sum_sq += yi * yi;
        }

        let node_mean = if n_samples > 0 {
            total_sum / (n_samples as f64)
        } else {
            0.0
        };

        let current_variance = if n_samples > 1 {
            let mean_sq = node_mean * node_mean;
            (total_sum_sq / (n_samples as f64) - mean_sq).max(0.0)
        } else {
            0.0
        };

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

        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_samples);
        let n_total_f = n_samples as f64;

        for f in 0..x.cols {
            pairs.clear();
            for &i in indices {
                pairs.push((x.get(i, f), y[i]));
            }
            pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if (pairs[n_samples - 1].0 - pairs[0].0).abs() < 1e-9 {
                continue;
            }

            let mut sum_left = 0.0;
            let mut sum_sq_left = 0.0;
            let mut n_left = 0usize;

            let mut sum_right = total_sum;
            let mut sum_sq_right = total_sum_sq;
            let mut n_right = n_samples;

            for i in 0..(n_samples - 1) {
                let val = pairs[i].1;
                sum_left += val;
                sum_sq_left += val * val;
                n_left += 1;

                sum_right -= val;
                sum_sq_right -= val * val;
                n_right -= 1;

                let cur_val = pairs[i].0;
                let next_val = pairs[i + 1].0;
                if (next_val - cur_val).abs() > 1e-9 {
                    if n_left >= self.min_samples_leaf && n_right >= self.min_samples_leaf {
                        let mean_l = sum_left / (n_left as f64);
                        let var_l = (sum_sq_left / (n_left as f64) - mean_l * mean_l).max(0.0);

                        let mean_r = sum_right / (n_right as f64);
                        let var_r = (sum_sq_right / (n_right as f64) - mean_r * mean_r).max(0.0);

                        let p_left = (n_left as f64) / n_total_f;
                        let p_right = (n_right as f64) / n_total_f;

                        let gain = current_variance - (p_left * var_l + p_right * var_r);
                        if gain > best_gain {
                            best_gain = gain;
                            best_feature = f;
                            best_threshold = (cur_val + next_val) / 2.0;
                        }
                    }
                }
            }
        }

        if best_gain <= 1e-9 {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let mut best_left = Vec::with_capacity(n_samples / 2);
        let mut best_right = Vec::with_capacity(n_samples / 2);
        for &i in indices {
            if x.get(i, best_feature) <= best_threshold {
                best_left.push(i);
            } else {
                best_right.push(i);
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
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
        self.device_ = Some(self.device.to_string());
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
            let offset = r * x.cols;
            let row_slice = &x.data[offset..offset + x.cols];
            preds.push(root.predict_row(row_slice));
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

        let mut clf = DecisionTreeClassifier::new(Some("gini".to_string()), Some(3), None, None, None).unwrap();
        clf.fit(&x, &y).unwrap();
        assert!(clf.device_.is_some());

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

        let mut reg = DecisionTreeRegressor::new(None, Some(3), None, None, None).unwrap();
        reg.fit(&x, &y).unwrap();
        assert!(reg.device_.is_some());

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
