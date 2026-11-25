// OchreML Decision Tree Module (＾▽＾)
// Implementation of CART algorithm for classification and regression trees,
// with Strict Accelerator-First N6 GPU histogram acceleration on CUDA and wgpu.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, r2_score};
use crate::primitives::{BinnedMatrix, GradHessBin, subtract_histogram};
use rayon::prelude::*;
use std::borrow::Cow;

const N6_MAX_BINS: usize = 256;
const N6_MIN_ROWS: usize = 64;

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

/// Contiguous flat node for `DecisionTreeClassifier` inference.
#[derive(Debug, Clone, Copy)]
pub struct FlatClfNode {
    pub feature_or_offset: u32,
    pub left: u32,
    pub right: u32,
    pub value_or_threshold: f64,
}

/// Contiguous flat node for `DecisionTreeRegressor` inference.
#[derive(Debug, Clone, Copy)]
pub struct FlatRegNode {
    pub feature_index: u32,
    pub left: u32,
    pub right: u32,
    pub value_or_threshold: f64,
}

#[inline]
fn validate_row_finite(row: &[f64]) -> OchreResult<()> {
    let mut ok = true;
    for &v in row {
        ok &= v.is_finite();
    }
    if !ok {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn validate_4rows_finite(r0: &[f64], r1: &[f64], r2: &[f64], r3: &[f64]) -> OchreResult<()> {
    let mut ok = true;
    for (((&a, &b), &c), &d) in r0.iter().zip(r1.iter()).zip(r2.iter()).zip(r3.iter()) {
        ok &= a.is_finite() & b.is_finite() & c.is_finite() & d.is_finite();
    }
    if !ok {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn flatten_reg_tree(node: &TreeNode, out: &mut Vec<FlatRegNode>) -> usize {
    let idx = out.len();
    match node {
        TreeNode::Leaf { value, .. } => {
            out.push(FlatRegNode {
                feature_index: 0,
                left: 0,
                right: u32::MAX,
                value_or_threshold: *value,
            });
        }
        TreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            out.push(FlatRegNode {
                feature_index: *feature_index as u32,
                left: 0,
                right: 0,
                value_or_threshold: *threshold,
            });
            let left_idx = flatten_reg_tree(left, out);
            let right_idx = flatten_reg_tree(right, out);
            out[idx].left = left_idx as u32;
            out[idx].right = right_idx as u32;
        }
    }
    idx
}

#[inline]
fn traverse_flat_reg_from(nodes: &[FlatRegNode], mut idx: usize, row: &[f64]) -> f64 {
    loop {
        let node = nodes[idx];
        if node.right == u32::MAX {
            return node.value_or_threshold;
        }
        idx = if row[node.feature_index as usize] <= node.value_or_threshold {
            node.left as usize
        } else {
            node.right as usize
        };
    }
}

#[inline]
fn predict_4rows(
    nodes: &[FlatRegNode],
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
) -> [f64; 4] {
    let mut i0 = 0usize;
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut i3 = 0usize;
    loop {
        let n0 = nodes[i0];
        let n1 = nodes[i1];
        let n2 = nodes[i2];
        let n3 = nodes[i3];
        if n0.right == u32::MAX
            || n1.right == u32::MAX
            || n2.right == u32::MAX
            || n3.right == u32::MAX
        {
            break;
        }
        i0 = if r0[n0.feature_index as usize] <= n0.value_or_threshold {
            n0.left as usize
        } else {
            n0.right as usize
        };
        i1 = if r1[n1.feature_index as usize] <= n1.value_or_threshold {
            n1.left as usize
        } else {
            n1.right as usize
        };
        i2 = if r2[n2.feature_index as usize] <= n2.value_or_threshold {
            n2.left as usize
        } else {
            n2.right as usize
        };
        i3 = if r3[n3.feature_index as usize] <= n3.value_or_threshold {
            n3.left as usize
        } else {
            n3.right as usize
        };
    }
    [
        traverse_flat_reg_from(nodes, i0, r0),
        traverse_flat_reg_from(nodes, i1, r1),
        traverse_flat_reg_from(nodes, i2, r2),
        traverse_flat_reg_from(nodes, i3, r3),
    ]
}

fn predict_reg_chunk(
    nodes: &[FlatRegNode],
    x_chunk: &[f64],
    cols: usize,
    out: &mut [f64],
) -> OchreResult<()> {
    let n_rows = out.len();
    let full_quads = n_rows / 4;
    for q in 0..full_quads {
        let base = q * 4 * cols;
        let r0 = &x_chunk[base..base + cols];
        let r1 = &x_chunk[base + cols..base + 2 * cols];
        let r2 = &x_chunk[base + 2 * cols..base + 3 * cols];
        let r3 = &x_chunk[base + 3 * cols..base + 4 * cols];
        validate_4rows_finite(r0, r1, r2, r3)?;
        let preds = predict_4rows(nodes, r0, r1, r2, r3);
        let o = q * 4;
        out[o] = preds[0];
        out[o + 1] = preds[1];
        out[o + 2] = preds[2];
        out[o + 3] = preds[3];
    }
    for r in (full_quads * 4)..n_rows {
        let row = &x_chunk[r * cols..(r + 1) * cols];
        validate_row_finite(row)?;
        out[r] = traverse_flat_reg_from(nodes, 0, row);
    }
    Ok(())
}

fn flatten_clf_tree(
    node: &TreeNode,
    classes: &[f64],
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let idx = nodes.len();
    match node {
        TreeNode::Leaf {
            value,
            probabilities,
        } => {
            let offset = probs_out.len() as u32;
            let start = probs_out.len();
            probs_out.resize(start + classes.len(), 0.0);
            for &(cls, p) in probabilities {
                if let Some(pos) = classes.iter().position(|&c| (c - cls).abs() < 1e-9) {
                    probs_out[start + pos] = p;
                }
            }
            nodes.push(FlatClfNode {
                feature_or_offset: offset,
                left: 0,
                right: u32::MAX,
                value_or_threshold: *value,
            });
        }
        TreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            nodes.push(FlatClfNode {
                feature_or_offset: *feature_index as u32,
                left: 0,
                right: 0,
                value_or_threshold: *threshold,
            });
            let left_idx = flatten_clf_tree(left, classes, nodes, probs_out);
            let right_idx = flatten_clf_tree(right, classes, nodes, probs_out);
            nodes[idx].left = left_idx as u32;
            nodes[idx].right = right_idx as u32;
        }
    }
    idx
}

#[inline]
fn traverse_flat_clf_from(nodes: &[FlatClfNode], mut idx: usize, row: &[f64]) -> FlatClfNode {
    loop {
        let node = nodes[idx];
        if node.right == u32::MAX {
            return node;
        }
        idx = if row[node.feature_or_offset as usize] <= node.value_or_threshold {
            node.left as usize
        } else {
            node.right as usize
        };
    }
}

#[inline]
fn predict_proba_4rows(
    nodes: &[FlatClfNode],
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
) -> [FlatClfNode; 4] {
    let mut i0 = 0usize;
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut i3 = 0usize;
    loop {
        let n0 = nodes[i0];
        let n1 = nodes[i1];
        let n2 = nodes[i2];
        let n3 = nodes[i3];
        if n0.right == u32::MAX
            || n1.right == u32::MAX
            || n2.right == u32::MAX
            || n3.right == u32::MAX
        {
            break;
        }
        i0 = if r0[n0.feature_or_offset as usize] <= n0.value_or_threshold {
            n0.left as usize
        } else {
            n0.right as usize
        };
        i1 = if r1[n1.feature_or_offset as usize] <= n1.value_or_threshold {
            n1.left as usize
        } else {
            n1.right as usize
        };
        i2 = if r2[n2.feature_or_offset as usize] <= n2.value_or_threshold {
            n2.left as usize
        } else {
            n2.right as usize
        };
        i3 = if r3[n3.feature_or_offset as usize] <= n3.value_or_threshold {
            n3.left as usize
        } else {
            n3.right as usize
        };
    }
    [
        traverse_flat_clf_from(nodes, i0, r0),
        traverse_flat_clf_from(nodes, i1, r1),
        traverse_flat_clf_from(nodes, i2, r2),
        traverse_flat_clf_from(nodes, i3, r3),
    ]
}

fn predict_clf_chunk(
    nodes: &[FlatClfNode],
    x_chunk: &[f64],
    cols: usize,
    out: &mut [f64],
) -> OchreResult<()> {
    let n_rows = out.len();
    let full_quads = n_rows / 4;
    for q in 0..full_quads {
        let base = q * 4 * cols;
        let r0 = &x_chunk[base..base + cols];
        let r1 = &x_chunk[base + cols..base + 2 * cols];
        let r2 = &x_chunk[base + 2 * cols..base + 3 * cols];
        let r3 = &x_chunk[base + 3 * cols..base + 4 * cols];
        validate_4rows_finite(r0, r1, r2, r3)?;
        let leaves = predict_proba_4rows(nodes, r0, r1, r2, r3);
        let o = q * 4;
        out[o] = leaves[0].value_or_threshold;
        out[o + 1] = leaves[1].value_or_threshold;
        out[o + 2] = leaves[2].value_or_threshold;
        out[o + 3] = leaves[3].value_or_threshold;
    }
    for r in (full_quads * 4)..n_rows {
        let row = &x_chunk[r * cols..(r + 1) * cols];
        validate_row_finite(row)?;
        out[r] = traverse_flat_clf_from(nodes, 0, row).value_or_threshold;
    }
    Ok(())
}

fn predict_clf_proba_chunk(
    nodes: &[FlatClfNode],
    probs: &[f64],
    n_classes: usize,
    x_chunk: &[f64],
    cols: usize,
    out: &mut [f64],
) -> OchreResult<()> {
    let n_rows = out.len() / n_classes;
    let full_quads = n_rows / 4;
    for q in 0..full_quads {
        let base = q * 4 * cols;
        let r0 = &x_chunk[base..base + cols];
        let r1 = &x_chunk[base + cols..base + 2 * cols];
        let r2 = &x_chunk[base + 2 * cols..base + 3 * cols];
        let r3 = &x_chunk[base + 3 * cols..base + 4 * cols];
        validate_4rows_finite(r0, r1, r2, r3)?;
        let leaves = predict_proba_4rows(nodes, r0, r1, r2, r3);
        let o = q * 4 * n_classes;
        for (k, leaf) in leaves.iter().enumerate() {
            let off = leaf.feature_or_offset as usize;
            out[o + k * n_classes..o + (k + 1) * n_classes]
                .copy_from_slice(&probs[off..off + n_classes]);
        }
    }
    for r in (full_quads * 4)..n_rows {
        let row = &x_chunk[r * cols..(r + 1) * cols];
        validate_row_finite(row)?;
        let off = traverse_flat_clf_from(nodes, 0, row).feature_or_offset as usize;
        out[r * n_classes..(r + 1) * n_classes].copy_from_slice(&probs[off..off + n_classes]);
    }
    Ok(())
}

fn transpose_to_col_major(x: &Matrix) -> Vec<f64> {
    let n_samples = x.rows;
    let n_features = x.cols;
    let mut x_cols = vec![0.0; n_features * n_samples];
    x_cols
        .par_chunks_exact_mut(n_samples)
        .enumerate()
        .for_each(|(f, col)| {
            for (r, dst) in col.iter_mut().enumerate() {
                *dst = x.data[r * n_features + f];
            }
        });
    x_cols
}

impl TreeNode {
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { value, .. } => return *value,
                TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    if row[*feature_index] <= *threshold {
                        node = left;
                    } else {
                        node = right;
                    }
                }
            }
        }
    }

    pub fn predict_proba_row(&self, row: &[f64], classes: &[f64]) -> Vec<f64> {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { probabilities, .. } => {
                    let mut probs = vec![0.0; classes.len()];
                    for &(cls, p) in probabilities {
                        if let Some(pos) = classes.iter().position(|&c| (c - cls).abs() < 1e-9) {
                            probs[pos] = p;
                        }
                    }
                    return probs;
                }
                TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    if row[*feature_index] <= *threshold {
                        node = left;
                    } else {
                        node = right;
                    }
                }
            }
        }
    }
}

#[inline]
fn is_gpu_histogram_device(device: &DeviceType, n_rows: usize) -> bool {
    n_rows >= N6_MIN_ROWS
        && matches!(
            device,
            DeviceType::Gpu(_) | DeviceType::MultiGpu(_) | DeviceType::Wgpu
        )
}

fn validate_fit_inputs(x: &Matrix, y: &[f64]) -> OchreResult<()> {
    if x.rows != y.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y.len(),
        });
    }
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Target y contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(())
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
    pub(crate) flat_nodes_: Vec<FlatClfNode>,
    pub(crate) flat_probs_: Vec<f64>,
}

impl DecisionTreeClassifier {
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        let crit = criterion
            .unwrap_or_else(|| "gini".to_string())
            .to_lowercase();
        if crit != "gini" && crit != "entropy" {
            return Err(OchreError::InvalidCriterion(crit));
        }
        Ok(Self {
            criterion: crit,
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            device: dev,
            device_: None,
            root: None,
            classes_: Vec::new(),
            n_features_in_: None,
            flat_nodes_: Vec::new(),
            flat_probs_: Vec::new(),
        })
    }

    /// Construct a `DecisionTreeClassifier` from `.dit` deserialized fields and populate flat cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_loaded(
        criterion: String,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        device: DeviceType,
        device_: Option<String>,
        root: Option<TreeNode>,
        classes_: Vec<f64>,
        n_features_in_: Option<usize>,
    ) -> Self {
        let mut clf = Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device,
            device_,
            root,
            classes_,
            n_features_in_,
            flat_nodes_: Vec::new(),
            flat_probs_: Vec::new(),
        };
        clf.rebuild_flat_cache();
        clf
    }

    /// Rebuild contiguous in-memory flat nodes and probability table from `self.root`.
    pub(crate) fn rebuild_flat_cache(&mut self) {
        self.flat_nodes_.clear();
        self.flat_probs_.clear();
        if let Some(ref root) = self.root {
            flatten_clf_tree(
                root,
                &self.classes_,
                &mut self.flat_nodes_,
                &mut self.flat_probs_,
            );
        }
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

    // Recursively build decision tree structure with O(D * N log N) sliding split search (CPU/TPU)
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        &self,
        x_cols: &[f64],
        n_total: usize,
        n_features: usize,
        class_indices: &[usize],
        indices: &mut [usize],
        depth: usize,
        pairs: &mut Vec<(f64, usize)>,
        left_counts: &mut Vec<usize>,
        right_counts: &mut Vec<usize>,
    ) -> TreeNode {
        let n_samples = indices.len();
        let n_classes = self.classes_.len();
        let mut total_counts = vec![0usize; n_classes];
        for &i in indices.iter() {
            total_counts[class_indices[i]] += 1;
        }

        let max_depth_reached = self.max_depth.is_some_and(|md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;
        let current_impurity = self.compute_impurity_from_counts(&total_counts, n_samples);

        if current_impurity < 1e-9 || max_depth_reached || too_few_samples {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let (best_gain, best_feature, best_threshold) = if n_samples >= 512 && n_features > 1 {
            (0..n_features)
                .into_par_iter()
                .filter_map(|f| {
                    let col = &x_cols[f * n_total..(f + 1) * n_total];
                    let mut local_pairs: Vec<(f64, usize)> = Vec::with_capacity(n_samples);
                    for &i in indices.iter() {
                        local_pairs.push((col[i], class_indices[i]));
                    }
                    local_pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

                    if (local_pairs[n_samples - 1].0 - local_pairs[0].0).abs() < 1e-9 {
                        return None;
                    }

                    let mut l_counts = vec![0usize; n_classes];
                    let mut r_counts = total_counts.clone();
                    let mut n_left = 0usize;
                    let mut n_right = n_samples;
                    let mut f_best_gain = 0.0;
                    let mut f_best_thresh = 0.0;

                    for i in 0..(n_samples - 1) {
                        let cls = local_pairs[i].1;
                        l_counts[cls] += 1;
                        r_counts[cls] -= 1;
                        n_left += 1;
                        n_right -= 1;

                        let cur_val = local_pairs[i].0;
                        let next_val = local_pairs[i + 1].0;
                        if (next_val - cur_val).abs() > 1e-9
                            && n_left >= self.min_samples_leaf
                            && n_right >= self.min_samples_leaf
                        {
                            let imp_left = self.compute_impurity_from_counts(&l_counts, n_left);
                            let imp_right = self.compute_impurity_from_counts(&r_counts, n_right);

                            let p_left = (n_left as f64) / (n_samples as f64);
                            let p_right = (n_right as f64) / (n_samples as f64);

                            let gain = current_impurity - (p_left * imp_left + p_right * imp_right);
                            if gain > f_best_gain {
                                f_best_gain = gain;
                                f_best_thresh = (cur_val + next_val) / 2.0;
                            }
                        }
                    }
                    if f_best_gain > 0.0 {
                        Some((f_best_gain, f, f_best_thresh))
                    } else {
                        None
                    }
                })
                .reduce_with(|a, b| {
                    if a.0 > b.0 || (a.0 == b.0 && a.1 < b.1) {
                        a
                    } else {
                        b
                    }
                })
                .unwrap_or((0.0, 0, 0.0))
        } else {
            let mut best_gain = 0.0;
            let mut best_feature = 0;
            let mut best_threshold = 0.0;

            for f in 0..n_features {
                let col = &x_cols[f * n_total..(f + 1) * n_total];
                pairs.clear();
                for &i in indices.iter() {
                    pairs.push((col[i], class_indices[i]));
                }
                pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

                if (pairs[n_samples - 1].0 - pairs[0].0).abs() < 1e-9 {
                    continue;
                }

                left_counts.fill(0);
                right_counts.copy_from_slice(&total_counts);
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
                    if (next_val - cur_val).abs() > 1e-9
                        && n_left >= self.min_samples_leaf
                        && n_right >= self.min_samples_leaf
                    {
                        let imp_left = self.compute_impurity_from_counts(left_counts, n_left);
                        let imp_right = self.compute_impurity_from_counts(right_counts, n_right);

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
            (best_gain, best_feature, best_threshold)
        };

        if best_gain <= 1e-9 {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let best_col = &x_cols[best_feature * n_total..(best_feature + 1) * n_total];
        let mut lo = 0usize;
        let mut hi = n_samples;
        while lo < hi {
            if best_col[indices[lo]] <= best_threshold {
                lo += 1;
            } else {
                hi -= 1;
                indices.swap(lo, hi);
            }
        }
        let left_len = lo;

        if left_len == 0 || left_len == n_samples {
            return self.make_leaf_from_counts(&total_counts, n_samples);
        }

        let (best_left, best_right) = indices.split_at_mut(left_len);
        let left_child = self.build_tree(
            x_cols,
            n_total,
            n_features,
            class_indices,
            best_left,
            depth + 1,
            pairs,
            left_counts,
            right_counts,
        );
        let right_child = self.build_tree(
            x_cols,
            n_total,
            n_features,
            class_indices,
            best_right,
            depth + 1,
            pairs,
            left_counts,
            right_counts,
        );

        TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree_n6(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        binned: &BinnedMatrix,
        class_indices: &[usize],
        one_hot: &[Vec<f64>],
        ones: &[f64],
        indices: &[u32],
        hists: &[Vec<GradHessBin>],
        depth: usize,
    ) -> OchreResult<TreeNode> {
        let n_samples = indices.len();
        let n_classes = self.classes_.len();
        let mut total_counts = vec![0usize; n_classes];
        for &i in indices {
            total_counts[class_indices[i as usize]] += 1;
        }

        let max_depth_reached = self.max_depth.is_some_and(|md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;
        let current_impurity = self.compute_impurity_from_counts(&total_counts, n_samples);

        if current_impurity < 1e-9 || max_depth_reached || too_few_samples {
            return Ok(self.make_leaf_from_counts(&total_counts, n_samples));
        }

        let num_bins = binned.num_bins;
        let k_tracked = one_hot.len();

        let best = (0..binned.cols)
            .into_par_iter()
            .filter_map(|f| {
                let edges = &binned.bin_edges[f];
                if edges.is_empty() {
                    return None;
                }
                let mut left_counts = vec![0usize; n_classes];
                let mut right_counts = total_counts.clone();
                let mut n_left = 0usize;
                let mut best_feat: Option<(f64, usize, f64)> = None;

                for (b, &threshold) in edges.iter().enumerate() {
                    let bin_idx = f * num_bins + b;
                    let bin_total = hists[0][bin_idx].count as usize;
                    n_left += bin_total;

                    let mut tracked_sum = 0usize;
                    for k in 0..k_tracked {
                        let c_k = hists[k][bin_idx].sum_grad.round().max(0.0) as usize;
                        left_counts[k] += c_k;
                        tracked_sum += left_counts[k];
                    }
                    if n_classes > k_tracked {
                        left_counts[n_classes - 1] = n_left.saturating_sub(tracked_sum);
                    }
                    for k in 0..n_classes {
                        right_counts[k] = total_counts[k].saturating_sub(left_counts[k]);
                    }

                    let n_right = n_samples.saturating_sub(n_left);
                    if n_left >= self.min_samples_leaf && n_right >= self.min_samples_leaf {
                        let imp_left = self.compute_impurity_from_counts(&left_counts, n_left);
                        let imp_right = self.compute_impurity_from_counts(&right_counts, n_right);
                        let p_left = (n_left as f64) / (n_samples as f64);
                        let p_right = (n_right as f64) / (n_samples as f64);
                        let gain = current_impurity - (p_left * imp_left + p_right * imp_right);
                        if best_feat.is_none_or(|(bg, _, _)| gain > bg) {
                            best_feat = Some((gain, f, threshold));
                        }
                    }
                }
                best_feat
            })
            .reduce_with(|a, b| if a.0 >= b.0 { a } else { b });

        let Some((best_gain, best_feature, best_threshold)) = best else {
            return Ok(self.make_leaf_from_counts(&total_counts, n_samples));
        };

        if best_gain <= 1e-9 {
            return Ok(self.make_leaf_from_counts(&total_counts, n_samples));
        }

        let mut best_left = Vec::with_capacity(n_samples / 2);
        let mut best_right = Vec::with_capacity(n_samples / 2);
        for &i in indices {
            if x.get(i as usize, best_feature) <= best_threshold {
                best_left.push(i);
            } else {
                best_right.push(i);
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return Ok(self.make_leaf_from_counts(&total_counts, n_samples));
        }

        let left_is_smaller = best_left.len() <= best_right.len();
        let small_idx = if left_is_smaller {
            &best_left
        } else {
            &best_right
        };

        let mut small_hists = Vec::with_capacity(k_tracked);
        let mut large_hists = Vec::with_capacity(k_tracked);
        for k in 0..k_tracked {
            let sh = backend.batched_histogram_grad(binned, small_idx, &one_hot[k], ones)?;
            let lh = subtract_histogram(&hists[k], &sh)?;
            small_hists.push(sh);
            large_hists.push(lh);
        }

        let (left_hists, right_hists) = if left_is_smaller {
            (small_hists, large_hists)
        } else {
            (large_hists, small_hists)
        };

        let left_child = self.build_tree_n6(
            backend,
            x,
            binned,
            class_indices,
            one_hot,
            ones,
            &best_left,
            &left_hists,
            depth + 1,
        )?;
        let right_child = self.build_tree_n6(
            backend,
            x,
            binned,
            class_indices,
            one_hot,
            ones,
            &best_right,
            &right_hists,
            depth + 1,
        )?;

        Ok(TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        })
    }

    // Fit DecisionTreeClassifier on feature matrix X and target labels y
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_fit_inputs(x, y)?;

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.total_cmp(b));
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

        if self.classes_.len() > 1 && is_gpu_histogram_device(&self.device, x.rows) {
            let backend = get_backend(&self.device);
            let binned = backend.quantize_features(x, N6_MAX_BINS)?;
            let k_tracked = self.classes_.len() - 1;
            let one_hot: Vec<Vec<f64>> = (0..k_tracked)
                .map(|k| {
                    class_indices
                        .iter()
                        .map(|&ci| if ci == k { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            let ones = vec![1.0f64; x.rows];
            let indices_u32: Vec<u32> = (0..x.rows as u32).collect();
            let mut root_hists = Vec::with_capacity(k_tracked);
            for oh in &one_hot {
                root_hists.push(backend.batched_histogram_grad(
                    &binned,
                    &indices_u32,
                    oh,
                    &ones,
                )?);
            }
            let root = self.build_tree_n6(
                &backend,
                x,
                &binned,
                &class_indices,
                &one_hot,
                &ones,
                &indices_u32,
                &root_hists,
                0,
            )?;
            self.root = Some(root);
        } else {
            let x_cols = transpose_to_col_major(x);
            let mut indices: Vec<usize> = (0..x.rows).collect();
            let n_classes = self.classes_.len();
            let mut pairs = Vec::with_capacity(x.rows.min(512));
            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = vec![0usize; n_classes];
            let root = self.build_tree(
                &x_cols,
                x.rows,
                x.cols,
                &class_indices,
                &mut indices,
                0,
                &mut pairs,
                &mut left_counts,
                &mut right_counts,
            );
            self.root = Some(root);
        }

        self.rebuild_flat_cache();

        Ok(())
    }

    // Predict class label for new feature data X
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let root = match &self.root {
            Some(r) => r,
            None => return Err(OchreError::NotFitted),
        };

        if let Some(expected_cols) = self.n_features_in_
            && x.cols != expected_cols
        {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }

        if x.rows == 0 {
            return Ok(Vec::new());
        }

        let nodes: Cow<'_, [FlatClfNode]> = if self.flat_nodes_.is_empty() {
            let mut flat = Vec::with_capacity(64);
            let mut probs = Vec::with_capacity(32 * self.classes_.len().max(1));
            flatten_clf_tree(root, &self.classes_, &mut flat, &mut probs);
            Cow::Owned(flat)
        } else {
            Cow::Borrowed(&self.flat_nodes_)
        };

        let cols = x.cols;
        let mut preds = vec![0.0; x.rows];

        if x.rows >= 2048 {
            preds
                .par_chunks_mut(2048)
                .zip(x.data.par_chunks(2048 * cols))
                .try_for_each(|(out_chunk, x_chunk)| {
                    predict_clf_chunk(&nodes, x_chunk, cols, out_chunk)
                })?;
        } else {
            predict_clf_chunk(&nodes, &x.data, cols, &mut preds)?;
        }
        Ok(preds)
    }

    // Predict class probabilities for new feature data X
    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let root = match &self.root {
            Some(r) => r,
            None => return Err(OchreError::NotFitted),
        };

        if let Some(expected_cols) = self.n_features_in_
            && x.cols != expected_cols
        {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }

        let n_classes = self.classes_.len();
        if n_classes == 0 {
            return Err(OchreError::NotFitted);
        }
        if x.rows == 0 {
            return Matrix::from_vec(0, n_classes, Vec::new());
        }

        let (nodes, probs): (Cow<'_, [FlatClfNode]>, Cow<'_, [f64]>) =
            if self.flat_nodes_.is_empty() {
                let mut flat = Vec::with_capacity(64);
                let mut flat_probs = Vec::with_capacity(32 * n_classes);
                flatten_clf_tree(root, &self.classes_, &mut flat, &mut flat_probs);
                (Cow::Owned(flat), Cow::Owned(flat_probs))
            } else {
                (
                    Cow::Borrowed(&self.flat_nodes_),
                    Cow::Borrowed(&self.flat_probs_),
                )
            };

        let cols = x.cols;
        let mut data = vec![0.0; x.rows * n_classes];
        if x.rows >= 2048 {
            data.par_chunks_mut(2048 * n_classes)
                .zip(x.data.par_chunks(2048 * cols))
                .try_for_each(|(out_chunk, x_chunk)| {
                    predict_clf_proba_chunk(&nodes, &probs, n_classes, x_chunk, cols, out_chunk)
                })?;
        } else {
            predict_clf_proba_chunk(&nodes, &probs, n_classes, &x.data, cols, &mut data)?;
        }
        Matrix::from_vec(x.rows, n_classes, data)
    }

    // Compute classification accuracy
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
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
    pub(crate) flat_nodes_: Vec<FlatRegNode>,
}

impl DecisionTreeRegressor {
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        let crit = criterion
            .unwrap_or_else(|| "squared_error".to_string())
            .to_lowercase();
        if crit != "squared_error" {
            return Err(OchreError::InvalidRegressorCriterion(crit));
        }
        Ok(Self {
            criterion: crit,
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2).max(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1).max(1),
            device: dev,
            device_: None,
            root: None,
            n_features_in_: None,
            flat_nodes_: Vec::new(),
        })
    }

    /// Construct a `DecisionTreeRegressor` from `.dit` deserialized fields and populate flat cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_loaded(
        criterion: String,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        device: DeviceType,
        device_: Option<String>,
        root: Option<TreeNode>,
        n_features_in_: Option<usize>,
    ) -> Self {
        let mut reg = Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device,
            device_,
            root,
            n_features_in_,
            flat_nodes_: Vec::new(),
        };
        reg.rebuild_flat_cache();
        reg
    }

    /// Rebuild contiguous in-memory flat nodes from `self.root`.
    pub(crate) fn rebuild_flat_cache(&mut self) {
        self.flat_nodes_.clear();
        if let Some(ref root) = self.root {
            flatten_reg_tree(root, &mut self.flat_nodes_);
        }
    }

    // Recursively build regression tree with O(D * N log N) sliding split search and O(1) MSE updates
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        &self,
        x_cols: &[f64],
        n_total: usize,
        n_features: usize,
        y: &[f64],
        indices: &mut [usize],
        depth: usize,
        pairs: &mut Vec<(f64, f64)>,
    ) -> TreeNode {
        let n_samples = indices.len();
        let mut total_sum = 0.0;
        let mut total_sum_sq = 0.0;
        for &i in indices.iter() {
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

        let max_depth_reached = self.max_depth.is_some_and(|md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;

        if current_variance < 1e-9 || max_depth_reached || too_few_samples {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let n_total_f = n_samples as f64;
        let (best_gain, best_feature, best_threshold) = if n_samples >= 512 && n_features > 1 {
            (0..n_features)
                .into_par_iter()
                .filter_map(|f| {
                    let col = &x_cols[f * n_total..(f + 1) * n_total];
                    let mut local_pairs: Vec<(f64, f64)> = Vec::with_capacity(n_samples);
                    for &i in indices.iter() {
                        local_pairs.push((col[i], y[i]));
                    }
                    local_pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

                    if (local_pairs[n_samples - 1].0 - local_pairs[0].0).abs() < 1e-9 {
                        return None;
                    }

                    let mut sum_left = 0.0;
                    let mut sum_sq_left = 0.0;
                    let mut n_left = 0usize;

                    let mut sum_right = total_sum;
                    let mut sum_sq_right = total_sum_sq;
                    let mut n_right = n_samples;
                    let mut f_best_gain = 0.0;
                    let mut f_best_thresh = 0.0;

                    for i in 0..(n_samples - 1) {
                        let val = local_pairs[i].1;
                        sum_left += val;
                        sum_sq_left += val * val;
                        n_left += 1;

                        sum_right -= val;
                        sum_sq_right -= val * val;
                        n_right -= 1;

                        let cur_val = local_pairs[i].0;
                        let next_val = local_pairs[i + 1].0;
                        if (next_val - cur_val).abs() > 1e-9
                            && n_left >= self.min_samples_leaf
                            && n_right >= self.min_samples_leaf
                        {
                            let mean_l = sum_left / (n_left as f64);
                            let var_l = (sum_sq_left / (n_left as f64) - mean_l * mean_l).max(0.0);

                            let mean_r = sum_right / (n_right as f64);
                            let var_r =
                                (sum_sq_right / (n_right as f64) - mean_r * mean_r).max(0.0);

                            let p_left = (n_left as f64) / n_total_f;
                            let p_right = (n_right as f64) / n_total_f;

                            let gain = current_variance - (p_left * var_l + p_right * var_r);
                            if gain > f_best_gain {
                                f_best_gain = gain;
                                f_best_thresh = (cur_val + next_val) / 2.0;
                            }
                        }
                    }
                    if f_best_gain > 0.0 {
                        Some((f_best_gain, f, f_best_thresh))
                    } else {
                        None
                    }
                })
                .reduce_with(|a, b| {
                    if a.0 > b.0 || (a.0 == b.0 && a.1 < b.1) {
                        a
                    } else {
                        b
                    }
                })
                .unwrap_or((0.0, 0, 0.0))
        } else {
            let mut best_gain = 0.0;
            let mut best_feature = 0;
            let mut best_threshold = 0.0;

            for f in 0..n_features {
                let col = &x_cols[f * n_total..(f + 1) * n_total];
                pairs.clear();
                for &i in indices.iter() {
                    pairs.push((col[i], y[i]));
                }
                pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

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
                    if (next_val - cur_val).abs() > 1e-9
                        && n_left >= self.min_samples_leaf
                        && n_right >= self.min_samples_leaf
                    {
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
            (best_gain, best_feature, best_threshold)
        };

        if best_gain <= 1e-9 {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let best_col = &x_cols[best_feature * n_total..(best_feature + 1) * n_total];
        let mut lo = 0usize;
        let mut hi = n_samples;
        while lo < hi {
            if best_col[indices[lo]] <= best_threshold {
                lo += 1;
            } else {
                hi -= 1;
                indices.swap(lo, hi);
            }
        }
        let left_len = lo;

        if left_len == 0 || left_len == n_samples {
            return TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            };
        }

        let (best_left, best_right) = indices.split_at_mut(left_len);
        let left_child =
            self.build_tree(x_cols, n_total, n_features, y, best_left, depth + 1, pairs);
        let right_child =
            self.build_tree(x_cols, n_total, n_features, y, best_right, depth + 1, pairs);

        TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree_n6(
        &self,
        backend: &Backend<'_>,
        x: &Matrix,
        binned: &BinnedMatrix,
        y: &[f64],
        grad: &[f64],
        hess: &[f64],
        indices: &[u32],
        hist: &[GradHessBin],
        depth: usize,
    ) -> OchreResult<TreeNode> {
        let n_samples = indices.len();
        let mut total_sum = 0.0;
        let mut total_sum_sq = 0.0;
        for &i in indices {
            let yi = y[i as usize];
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

        let max_depth_reached = self.max_depth.is_some_and(|md| depth >= md);
        let too_few_samples = n_samples < self.min_samples_split;

        if current_variance < 1e-9 || max_depth_reached || too_few_samples {
            return Ok(TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            });
        }

        let split_opt =
            backend.find_best_histogram_split(binned, hist, 0.0, self.min_samples_leaf as u32)?;
        let Some(split) = split_opt else {
            return Ok(TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            });
        };

        if split.gain <= 1e-9 * (n_samples as f64) {
            return Ok(TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            });
        }

        let mut best_left = Vec::with_capacity(n_samples / 2);
        let mut best_right = Vec::with_capacity(n_samples / 2);
        for &i in indices {
            if x.get(i as usize, split.feature_idx) <= split.threshold {
                best_left.push(i);
            } else {
                best_right.push(i);
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return Ok(TreeNode::Leaf {
                value: node_mean,
                probabilities: Vec::new(),
            });
        }

        let left_is_smaller = best_left.len() <= best_right.len();
        let small_idx = if left_is_smaller {
            &best_left
        } else {
            &best_right
        };
        let small_hist = backend.batched_histogram_grad(binned, small_idx, grad, hess)?;
        let large_hist = subtract_histogram(hist, &small_hist)?;
        let (left_hist, right_hist) = if left_is_smaller {
            (small_hist, large_hist)
        } else {
            (large_hist, small_hist)
        };

        let left_child = self.build_tree_n6(
            backend,
            x,
            binned,
            y,
            grad,
            hess,
            &best_left,
            &left_hist,
            depth + 1,
        )?;
        let right_child = self.build_tree_n6(
            backend,
            x,
            binned,
            y,
            grad,
            hess,
            &best_right,
            &right_hist,
            depth + 1,
        )?;

        Ok(TreeNode::Internal {
            feature_index: split.feature_idx,
            threshold: split.threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        })
    }

    // Fit DecisionTreeRegressor on continuous dataset (X, y)
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> OchreResult<()> {
        validate_fit_inputs(x, y)?;

        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());

        if is_gpu_histogram_device(&self.device, x.rows) {
            let backend = get_backend(&self.device);
            let binned = backend.quantize_features(x, N6_MAX_BINS)?;
            let grad: Vec<f64> = y.iter().map(|&v| -v).collect();
            let hess = vec![1.0f64; x.rows];
            let indices_u32: Vec<u32> = (0..x.rows as u32).collect();
            let root_hist = backend.batched_histogram_grad(&binned, &indices_u32, &grad, &hess)?;
            let root = self.build_tree_n6(
                &backend,
                x,
                &binned,
                y,
                &grad,
                &hess,
                &indices_u32,
                &root_hist,
                0,
            )?;
            self.root = Some(root);
        } else {
            let x_cols = transpose_to_col_major(x);
            let mut indices: Vec<usize> = (0..x.rows).collect();
            let mut pairs = Vec::with_capacity(x.rows.min(512));
            let root = self.build_tree(&x_cols, x.rows, x.cols, y, &mut indices, 0, &mut pairs);
            self.root = Some(root);
        }

        self.rebuild_flat_cache();

        Ok(())
    }

    // Predict continuous target values for new data X
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let root = match &self.root {
            Some(r) => r,
            None => return Err(OchreError::NotFitted),
        };

        if let Some(expected_cols) = self.n_features_in_
            && x.cols != expected_cols
        {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_cols,
            });
        }

        if x.rows == 0 {
            return Ok(Vec::new());
        }

        let nodes: Cow<'_, [FlatRegNode]> = if self.flat_nodes_.is_empty() {
            let mut flat = Vec::with_capacity(64);
            flatten_reg_tree(root, &mut flat);
            Cow::Owned(flat)
        } else {
            Cow::Borrowed(&self.flat_nodes_)
        };

        let cols = x.cols;
        let mut preds = vec![0.0; x.rows];

        if x.rows >= 2048 {
            preds
                .par_chunks_mut(2048)
                .zip(x.data.par_chunks(2048 * cols))
                .try_for_each(|(out_chunk, x_chunk)| {
                    predict_reg_chunk(&nodes, x_chunk, cols, out_chunk)
                })?;
        } else {
            predict_reg_chunk(&nodes, &x.data, cols, &mut preds)?;
        }
        Ok(preds)
    }

    // Compute R^2 score for regression performance
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_classifier() {
        let x = Matrix::from_2d(&[
            vec![0.5],
            vec![1.0],
            vec![1.5],
            vec![2.5],
            vec![3.0],
            vec![3.5],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut clf =
            DecisionTreeClassifier::new(Some("gini".to_string()), Some(3), None, None, None)
                .unwrap();
        clf.fit(&x, &y).unwrap();
        assert!(clf.device_.is_some());

        let preds = clf.predict(&x).unwrap();
        assert_eq!(preds, y);

        let test_x = Matrix::from_2d(&[vec![0.8], vec![3.2]]).unwrap();
        let test_preds = clf.predict(&test_x).unwrap();
        assert_eq!(test_preds, vec![0.0, 1.0]);

        let acc = clf.score(&x, &y).unwrap();
        assert!((acc - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x = Matrix::from_2d(&[
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
        ])
        .unwrap();
        let y = vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0];

        let mut reg = DecisionTreeRegressor::new(None, Some(3), None, None, None).unwrap();
        reg.fit(&x, &y).unwrap();
        assert!(reg.device_.is_some());

        let preds = reg.predict(&x).unwrap();
        for (p, actual) in preds.iter().zip(y.iter()) {
            assert!((p - actual).abs() < 1e-4);
        }

        let test_x = Matrix::from_2d(&[vec![2.5], vec![7.5]]).unwrap();
        let test_preds = reg.predict(&test_x).unwrap();
        assert!((test_preds[0] - 10.0).abs() < 1e-4);
        assert!((test_preds[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_decision_tree_n6_wgpu_path() {
        let mut rows = Vec::with_capacity(128);
        let mut y_cls = Vec::with_capacity(128);
        let mut y_reg = Vec::with_capacity(128);
        for i in 0..128 {
            let x0 = (i as f64) / 128.0;
            let x1 = ((i * 7) % 19) as f64 / 19.0;
            rows.push(vec![x0, x1]);
            y_cls.push(if x0 >= 0.5 { 1.0 } else { 0.0 });
            y_reg.push(if x0 >= 0.5 { 10.0 } else { 2.0 });
        }
        let x = Matrix::from_2d(&rows).unwrap();

        let mut clf = DecisionTreeClassifier::new(
            Some("gini".to_string()),
            Some(3),
            None,
            None,
            Some("wgpu".to_string()),
        )
        .unwrap();
        clf.fit(&x, &y_cls).unwrap();
        assert!(clf.score(&x, &y_cls).unwrap() > 0.98);

        let mut reg =
            DecisionTreeRegressor::new(None, Some(3), None, None, Some("wgpu".to_string()))
                .unwrap();
        reg.fit(&x, &y_reg).unwrap();
        assert!(reg.score(&x, &y_reg).unwrap() > 0.98);
    }

    #[test]
    fn test_decision_tree_flat_parallel_predict_parity() {
        let n = 4100usize;
        let mut data = Vec::with_capacity(n * 2);
        let mut y_cls = Vec::with_capacity(n);
        let mut y_reg = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = (i as f64) / (n as f64);
            let x1 = ((i * 13) % 97) as f64 / 97.0;
            data.push(x0);
            data.push(x1);
            y_cls.push(if x0 >= 0.5 { 1.0 } else { 0.0 });
            y_reg.push(if x0 >= 0.5 { 10.0 } else { 2.0 });
        }
        let x = Matrix::from_vec(n, 2, data).unwrap();

        let mut clf = DecisionTreeClassifier::new(None, Some(4), None, None, None).unwrap();
        clf.fit(&x, &y_cls).unwrap();
        let preds = clf.predict(&x).unwrap();
        assert_eq!(preds.len(), n);
        assert_eq!(preds, y_cls);
        let proba = clf.predict_proba(&x).unwrap();
        assert_eq!(proba.rows, n);
        assert_eq!(proba.cols, 2);

        let mut reg = DecisionTreeRegressor::new(None, Some(4), None, None, None).unwrap();
        reg.fit(&x, &y_reg).unwrap();
        assert!(!reg.flat_nodes_.is_empty());
        let rpreds = reg.predict(&x).unwrap();
        assert_eq!(rpreds.len(), n);
        for (p, &t) in rpreds.iter().zip(y_reg.iter()) {
            assert!((p - t).abs() < 1e-6);
        }

        // Verify NaN/Inf fused validation in both parallel (>= 2048) and serial paths
        let mut bad_x = x.clone();
        bad_x.data[(n - 3) * 2 + 1] = f64::NAN;
        assert!(clf.predict(&bad_x).is_err());
        assert!(clf.predict_proba(&bad_x).is_err());
        assert!(clf.score(&bad_x, &y_cls).is_err());
        assert!(reg.predict(&bad_x).is_err());
        assert!(reg.score(&bad_x, &y_reg).is_err());

        let small_bad = Matrix::from_vec(
            5,
            2,
            vec![0.1, 0.2, 0.3, f64::INFINITY, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        .unwrap();
        assert!(clf.predict(&small_bad).is_err());
        assert!(clf.predict_proba(&small_bad).is_err());
        assert!(reg.predict(&small_bad).is_err());

        // Verify from_loaded populates flat cache without changing serialization
        let clf_loaded = DecisionTreeClassifier::from_loaded(
            clf.criterion.clone(),
            clf.max_depth,
            clf.min_samples_split,
            clf.min_samples_leaf,
            clf.device.clone(),
            clf.device_.clone(),
            clf.root.clone(),
            clf.classes_.clone(),
            clf.n_features_in_,
        );
        assert!(!clf_loaded.flat_nodes_.is_empty());
        assert_eq!(clf_loaded.predict(&x).unwrap(), y_cls);

        let reg_loaded = DecisionTreeRegressor::from_loaded(
            reg.criterion.clone(),
            reg.max_depth,
            reg.min_samples_split,
            reg.min_samples_leaf,
            reg.device.clone(),
            reg.device_.clone(),
            reg.root.clone(),
            reg.n_features_in_,
        );
        assert!(!reg_loaded.flat_nodes_.is_empty());
        assert_eq!(reg_loaded.predict(&x).unwrap(), rpreds);
    }
}
