// OchreML Histogram-based Gradient Boosting Engine (*≧ω≦*)
// Implements HistGradientBoostingClassifier and HistGradientBoostingRegressor
// accelerated by Backend Primitive N6 (quantize_features, batched_histogram_grad,
// subtract_histogram, find_best_histogram_split).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_f64, write_f64_vec, write_header, write_opt_str,
    write_opt_u64, write_opt_usize, write_str, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, r2_score};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::primitives::{BinnedMatrix, GradHessBin, HistogramSplit, subtract_histogram};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub const TAG_HIST_GB_CLASSIFIER: u8 = 65;
pub const TAG_HIST_GB_REGRESSOR: u8 = 66;

#[derive(Debug, Clone)]
pub enum HgbNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<HgbNode>,
        right: Box<HgbNode>,
    },
}

impl HgbNode {
    #[inline]
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let mut node = self;
        loop {
            match node {
                HgbNode::Leaf { value } => return *value,
                HgbNode::Internal {
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

    fn write_to_buf(&self, buf: &mut Vec<u8>) {
        match self {
            HgbNode::Leaf { value } => {
                write_u8(buf, 0);
                write_f64(buf, *value);
            }
            HgbNode::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => {
                write_u8(buf, 1);
                write_usize(buf, *feature_index);
                write_f64(buf, *threshold);
                left.write_to_buf(buf);
                right.write_to_buf(buf);
            }
        }
    }

    fn read_from_reader(r: &mut ByteReader<'_>, depth: usize) -> OchreResult<Self> {
        if depth > 4096 {
            return Err(OchreError::CorruptedDitFile);
        }
        let tag = r.read_u8()?;
        match tag {
            0 => {
                let value = r.read_f64()?;
                Ok(HgbNode::Leaf { value })
            }
            1 => {
                let feature_index = r.read_usize()?;
                let threshold = r.read_f64()?;
                let left = Self::read_from_reader(r, depth + 1)?;
                let right = Self::read_from_reader(r, depth + 1)?;
                Ok(HgbNode::Internal {
                    feature_index,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
}

#[derive(Debug, Clone)]
enum ArenaNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

impl ArenaNode {
    fn into_hgb_node(arena: &[ArenaNode], idx: usize) -> HgbNode {
        match &arena[idx] {
            ArenaNode::Leaf { value } => HgbNode::Leaf { value: *value },
            ArenaNode::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => HgbNode::Internal {
                feature_index: *feature_index,
                threshold: *threshold,
                left: Box::new(Self::into_hgb_node(arena, *left)),
                right: Box::new(Self::into_hgb_node(arena, *right)),
            },
        }
    }
}

struct SplitCandidate {
    node_idx: usize,
    depth: usize,
    sample_indices: Vec<u32>,
    hist: Vec<GradHessBin>,
    sum_grad: f64,
    sum_hess: f64,
    split: HistogramSplit,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx && self.split.gain == other.split.gain
    }
}

impl Eq for SplitCandidate {}

impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.split
            .gain
            .total_cmp(&other.split.gain)
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

#[inline]
fn newton_leaf_value(sum_grad: f64, sum_hess: f64, l2_reg: f64, learning_rate: f64) -> f64 {
    let denom = sum_hess + l2_reg;
    if denom <= 1e-12 {
        0.0
    } else {
        (-learning_rate * (sum_grad / denom)).clamp(-1e6, 1e6)
    }
}

fn weighted_median(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_w: f64 = pairs.iter().map(|&(_, w)| w).sum();
    if total_w <= 0.0 {
        return pairs[pairs.len() / 2].0;
    }
    let half = 0.5 * total_w;
    let mut cum = 0.0;
    for &(val, w) in &pairs {
        cum += w;
        if cum >= half {
            return val;
        }
    }
    pairs.last().map_or(0.0, |&(v, _)| v)
}

#[allow(clippy::too_many_arguments)]
fn validate_common_params(
    learning_rate: f64,
    max_iter: usize,
    max_leaf_nodes: Option<usize>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f64,
    max_bins: usize,
) -> OchreResult<()> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "learning_rate={} must be strictly positive and finite ( >_< )",
            learning_rate
        )));
    }
    if max_iter == 0 {
        return Err(OchreError::InvalidParameter(
            "max_iter must be >= 1 ( >_< )".to_string(),
        ));
    }
    if let Some(m) = max_leaf_nodes
        && m < 2
    {
        return Err(OchreError::InvalidParameter(format!(
            "max_leaf_nodes={} must be >= 2 when specified ( >_< )",
            m
        )));
    }
    if let Some(d) = max_depth
        && d == 0
    {
        return Err(OchreError::InvalidParameter(
            "max_depth must be >= 1 when specified ( >_< )".to_string(),
        ));
    }
    if min_samples_leaf == 0 {
        return Err(OchreError::InvalidParameter(
            "min_samples_leaf must be >= 1 ( >_< )".to_string(),
        ));
    }
    if !l2_regularization.is_finite() || l2_regularization < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "l2_regularization={} must be non-negative and finite ( >_< )",
            l2_regularization
        )));
    }
    if !(2..=255).contains(&max_bins) {
        return Err(OchreError::InvalidParameter(format!(
            "max_bins={} must be within 2..=255 ( >_< )",
            max_bins
        )));
    }
    Ok(())
}

fn validate_fit_inputs(
    x: &Matrix,
    y: &[f64],
    sample_weight: Option<&[f64]>,
) -> OchreResult<Vec<f64>> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.rows != y.len() {
        return Err(OchreError::LengthMismatch {
            features: x.rows,
            targets: y.len(),
        });
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
    if let Some(sw) = sample_weight {
        if sw.len() != x.rows {
            return Err(OchreError::WeightLengthMismatch {
                weights: sw.len(),
                samples: x.rows,
            });
        }
        if sw.iter().any(|&w| !w.is_finite() || w < 0.0) {
            return Err(OchreError::InvalidWeights);
        }
        let sum_w: f64 = sw.iter().sum();
        if sum_w <= 0.0 {
            return Err(OchreError::NonPositiveWeightSum);
        }
        Ok(sw.to_vec())
    } else {
        Ok(vec![1.0; x.rows])
    }
}

fn validate_predict_matrix(x: &Matrix, n_features_in: Option<usize>) -> OchreResult<usize> {
    let expected_cols = n_features_in.ok_or(OchreError::NotFitted)?;
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.cols != expected_cols {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: expected_cols,
        });
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(expected_cols)
}

struct TreeBuildConfig {
    learning_rate: f64,
    max_leaf_nodes: Option<usize>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_hgb_tree(
    backend: &Backend<'_>,
    binned: &BinnedMatrix,
    grad: &[f64],
    hess: &[f64],
    cfg: &TreeBuildConfig,
    raw: &mut [f64],
    abs_loss_targets: Option<(&[f64], &[f64])>,
) -> OchreResult<HgbNode> {
    let n_rows = binned.rows;
    let cols = binned.cols;
    let all_indices: Vec<u32> = (0..n_rows as u32).collect();
    let root_sum_g: f64 = grad.iter().sum();
    let root_sum_h: f64 = hess.iter().sum();

    let root_val = newton_leaf_value(
        root_sum_g,
        root_sum_h,
        cfg.l2_regularization,
        cfg.learning_rate,
    );
    let mut arena = vec![ArenaNode::Leaf { value: root_val }];
    let mut leaf_samples: Vec<Option<Vec<u32>>> = vec![Some(all_indices.clone())];
    let mut n_leaves = 1usize;

    let can_split_root = cfg.max_leaf_nodes.is_none_or(|m| n_leaves < m)
        && cfg.max_depth.is_none_or(|d| d > 0)
        && n_rows >= 2 * cfg.min_samples_leaf;

    if can_split_root {
        let root_hist = backend.batched_histogram_grad(binned, &all_indices, grad, hess)?;
        if let Some(root_split) = backend.find_best_histogram_split(
            binned,
            &root_hist,
            cfg.l2_regularization,
            cfg.min_samples_leaf as u32,
        )? && root_split.gain > 1e-12
        {
            let mut heap = BinaryHeap::new();
            heap.push(SplitCandidate {
                node_idx: 0,
                depth: 0,
                sample_indices: all_indices,
                hist: root_hist,
                sum_grad: root_sum_g,
                sum_hess: root_sum_h,
                split: root_split,
            });

            while cfg.max_leaf_nodes.is_none_or(|m| n_leaves < m) {
                let Some(cand) = heap.pop() else {
                    break;
                };
                if cand.split.gain <= 1e-12 {
                    break;
                }

                let feat = cand.split.feature_idx;
                let bin_thresh = cand.split.bin_idx as u8;
                let mut left_indices = Vec::with_capacity(cand.sample_indices.len() / 2);
                let mut right_indices = Vec::with_capacity(cand.sample_indices.len() / 2);
                for &idx in &cand.sample_indices {
                    if binned.data[idx as usize * cols + feat] <= bin_thresh {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.len() < cfg.min_samples_leaf
                    || right_indices.len() < cfg.min_samples_leaf
                {
                    continue;
                }

                let left_is_smaller = left_indices.len() <= right_indices.len();
                let small_idx = if left_is_smaller {
                    &left_indices
                } else {
                    &right_indices
                };
                let small_hist = backend.batched_histogram_grad(binned, small_idx, grad, hess)?;
                let large_hist = subtract_histogram(&cand.hist, &small_hist)?;
                let (left_hist, right_hist) = if left_is_smaller {
                    (small_hist, large_hist)
                } else {
                    (large_hist, small_hist)
                };

                let left_g = cand.split.left_grad;
                let left_h = cand.split.left_hess;
                let right_g = cand.sum_grad - left_g;
                let right_h = cand.sum_hess - left_h;

                let left_val =
                    newton_leaf_value(left_g, left_h, cfg.l2_regularization, cfg.learning_rate);
                let right_val =
                    newton_leaf_value(right_g, right_h, cfg.l2_regularization, cfg.learning_rate);

                let left_node_idx = arena.len();
                arena.push(ArenaNode::Leaf { value: left_val });
                let right_node_idx = arena.len();
                arena.push(ArenaNode::Leaf { value: right_val });

                arena[cand.node_idx] = ArenaNode::Internal {
                    feature_index: feat,
                    threshold: cand.split.threshold,
                    left: left_node_idx,
                    right: right_node_idx,
                };

                leaf_samples[cand.node_idx] = None;
                let child_depth = cand.depth + 1;
                let can_expand_depth = cfg.max_depth.is_none_or(|md| child_depth < md);

                let left_can_split =
                    can_expand_depth && left_indices.len() >= 2 * cfg.min_samples_leaf;
                let right_can_split =
                    can_expand_depth && right_indices.len() >= 2 * cfg.min_samples_leaf;

                if left_can_split
                    && let Some(lsplit) = backend.find_best_histogram_split(
                        binned,
                        &left_hist,
                        cfg.l2_regularization,
                        cfg.min_samples_leaf as u32,
                    )?
                    && lsplit.gain > 1e-12
                {
                    leaf_samples.push(Some(left_indices.clone()));
                    heap.push(SplitCandidate {
                        node_idx: left_node_idx,
                        depth: child_depth,
                        sample_indices: left_indices,
                        hist: left_hist,
                        sum_grad: left_g,
                        sum_hess: left_h,
                        split: lsplit,
                    });
                } else {
                    leaf_samples.push(Some(left_indices));
                }

                if right_can_split
                    && let Some(rsplit) = backend.find_best_histogram_split(
                        binned,
                        &right_hist,
                        cfg.l2_regularization,
                        cfg.min_samples_leaf as u32,
                    )?
                    && rsplit.gain > 1e-12
                {
                    leaf_samples.push(Some(right_indices.clone()));
                    heap.push(SplitCandidate {
                        node_idx: right_node_idx,
                        depth: child_depth,
                        sample_indices: right_indices,
                        hist: right_hist,
                        sum_grad: right_g,
                        sum_hess: right_h,
                        split: rsplit,
                    });
                } else {
                    leaf_samples.push(Some(right_indices));
                }

                n_leaves += 1;
            }
        }
    }

    for (node_idx, maybe_indices) in leaf_samples.into_iter().enumerate() {
        let Some(indices) = maybe_indices else {
            continue;
        };
        let leaf_val = if let Some((y, weights)) = abs_loss_targets {
            let mut resids = Vec::with_capacity(indices.len());
            let mut ws = Vec::with_capacity(indices.len());
            for &idx in &indices {
                let r = idx as usize;
                resids.push(y[r] - raw[r]);
                ws.push(weights[r]);
            }
            let val = cfg.learning_rate * weighted_median(&resids, &ws);
            arena[node_idx] = ArenaNode::Leaf { value: val };
            val
        } else if let ArenaNode::Leaf { value } = arena[node_idx] {
            value
        } else {
            0.0
        };
        for &idx in &indices {
            raw[idx as usize] += leaf_val;
        }
    }

    Ok(ArenaNode::into_hgb_node(&arena, 0))
}

fn predict_trees_sum(trees: &[HgbNode], baseline: f64, x: &Matrix) -> Vec<f64> {
    let cols = x.cols;
    if x.rows >= 256 {
        (0..x.rows)
            .into_par_iter()
            .map(|r| {
                let row = &x.data[r * cols..(r + 1) * cols];
                let mut acc = baseline;
                for tree in trees {
                    acc += tree.predict_row(row);
                }
                acc
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row = &x.data[r * cols..(r + 1) * cols];
            let mut acc = baseline;
            for tree in trees {
                acc += tree.predict_row(row);
            }
            out.push(acc);
        }
        out
    }
}

// =========================================================================
// HIST GRADIENT BOOSTING REGRESSOR
// =========================================================================

#[derive(Debug, Clone)]
pub struct HistGradientBoostingRegressor {
    pub loss: String,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub max_leaf_nodes: Option<usize>,
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub n_features_in_: Option<usize>,
    pub n_iter_: usize,
    pub baseline_: f64,
    pub trees_: Vec<HgbNode>,
}

impl HistGradientBoostingRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Option<String>,
        learning_rate: Option<f64>,
        max_iter: Option<usize>,
        max_leaf_nodes: Option<usize>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        l2_regularization: Option<f64>,
        max_bins: Option<usize>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let loss_str = loss
            .unwrap_or_else(|| "squared_error".to_string())
            .to_lowercase();
        if loss_str != "squared_error" && loss_str != "absolute_error" {
            return Err(OchreError::InvalidParameter(format!(
                "loss='{}' is not supported; use 'squared_error' or 'absolute_error' ( >_< )",
                loss_str
            )));
        }
        let lr = learning_rate.unwrap_or(0.1);
        let m_iter = max_iter.unwrap_or(100);
        let min_leaf = min_samples_leaf.unwrap_or(20);
        let l2 = l2_regularization.unwrap_or(0.0);
        let bins = max_bins.unwrap_or(255);
        validate_common_params(lr, m_iter, max_leaf_nodes, max_depth, min_leaf, l2, bins)?;
        let dev = parse_device(device.as_deref())?;

        Ok(Self {
            loss: loss_str,
            learning_rate: lr,
            max_iter: m_iter,
            max_leaf_nodes,
            max_depth,
            min_samples_leaf: min_leaf,
            l2_regularization: l2,
            max_bins: bins,
            random_state,
            device: dev,
            device_: None,
            n_features_in_: None,
            n_iter_: 0,
            baseline_: 0.0,
            trees_: Vec::new(),
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        let weights = validate_fit_inputs(x, y, sample_weight)?;
        let backend = get_backend(&self.device);
        let binned = backend.quantize_features(x, self.max_bins)?;

        let is_abs = self.loss == "absolute_error";
        let baseline = if is_abs {
            weighted_median(y, &weights)
        } else {
            let sum_wy: f64 = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();
            let sum_w: f64 = weights.iter().sum();
            sum_wy / sum_w
        };

        let mut raw = vec![baseline; x.rows];
        let mut grad = vec![0.0; x.rows];
        let hess = weights.clone();
        let cfg = TreeBuildConfig {
            learning_rate: self.learning_rate,
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
        };

        let mut trees = Vec::with_capacity(self.max_iter);
        for _ in 0..self.max_iter {
            if is_abs {
                for i in 0..x.rows {
                    let diff = raw[i] - y[i];
                    grad[i] = if diff > 0.0 {
                        weights[i]
                    } else if diff < 0.0 {
                        -weights[i]
                    } else {
                        0.0
                    };
                }
            } else {
                for i in 0..x.rows {
                    grad[i] = weights[i] * (raw[i] - y[i]);
                }
            }

            let tree = build_hgb_tree(
                &backend,
                &binned,
                &grad,
                &hess,
                &cfg,
                &mut raw,
                if is_abs { Some((y, &weights)) } else { None },
            )?;
            trees.push(tree);
        }

        self.baseline_ = baseline;
        self.trees_ = trees;
        self.n_iter_ = self.trees_.len();
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        validate_predict_matrix(x, self.n_features_in_)?;
        Ok(predict_trees_sum(&self.trees_, self.baseline_, x))
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_HIST_GB_REGRESSOR);
        write_str(&mut buf, &self.loss);
        write_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.max_iter);
        write_opt_usize(&mut buf, self.max_leaf_nodes);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_leaf);
        write_f64(&mut buf, self.l2_regularization);
        write_usize(&mut buf, self.max_bins);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_iter_);
        write_f64(&mut buf, self.baseline_);
        write_usize(&mut buf, self.trees_.len());
        for tree in &self.trees_ {
            tree.write_to_buf(&mut buf);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_HIST_GB_REGRESSOR)?;
        let loss = r.read_str()?;
        let learning_rate = r.read_f64()?;
        let max_iter = r.read_usize()?;
        let max_leaf_nodes = r.read_opt_usize()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let l2_regularization = r.read_f64()?;
        let max_bins = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_iter_ = r.read_usize()?;
        let baseline_ = r.read_f64()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(HgbNode::read_from_reader(&mut r, 0)?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            loss,
            learning_rate,
            max_iter,
            max_leaf_nodes,
            max_depth,
            min_samples_leaf,
            l2_regularization,
            max_bins,
            random_state,
            device: dev,
            device_,
            n_features_in_,
            n_iter_,
            baseline_,
            trees_,
        })
    }
}

// =========================================================================
// HIST GRADIENT BOOSTING CLASSIFIER
// =========================================================================

#[derive(Debug, Clone)]
pub struct HistGradientBoostingClassifier {
    pub learning_rate: f64,
    pub max_iter: usize,
    pub max_leaf_nodes: Option<usize>,
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Vec<f64>,
    pub n_features_in_: Option<usize>,
    pub n_iter_: usize,
    pub baseline_: Vec<f64>,
    /// Trees stored in iteration-major order:
    /// binary -> `max_iter` trees; multiclass -> `max_iter * n_classes` trees.
    pub trees_: Vec<HgbNode>,
}

impl HistGradientBoostingClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        learning_rate: Option<f64>,
        max_iter: Option<usize>,
        max_leaf_nodes: Option<usize>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        l2_regularization: Option<f64>,
        max_bins: Option<usize>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let lr = learning_rate.unwrap_or(0.1);
        let m_iter = max_iter.unwrap_or(100);
        let min_leaf = min_samples_leaf.unwrap_or(20);
        let l2 = l2_regularization.unwrap_or(0.0);
        let bins = max_bins.unwrap_or(255);
        validate_common_params(lr, m_iter, max_leaf_nodes, max_depth, min_leaf, l2, bins)?;
        let dev = parse_device(device.as_deref())?;

        Ok(Self {
            learning_rate: lr,
            max_iter: m_iter,
            max_leaf_nodes,
            max_depth,
            min_samples_leaf: min_leaf,
            l2_regularization: l2,
            max_bins: bins,
            random_state,
            device: dev,
            device_: None,
            classes_: Vec::new(),
            n_features_in_: None,
            n_iter_: 0,
            baseline_: Vec::new(),
            trees_: Vec::new(),
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        let weights = validate_fit_inputs(x, y, sample_weight)?;

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }

        let class_indices: Vec<usize> = y
            .iter()
            .map(|&val| {
                classes
                    .iter()
                    .position(|&c| (c - val).abs() < 1e-9)
                    .unwrap_or(0)
            })
            .collect();

        let backend = get_backend(&self.device);
        let binned = backend.quantize_features(x, self.max_bins)?;
        let cfg = TreeBuildConfig {
            learning_rate: self.learning_rate,
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
        };

        let n_classes = classes.len();
        let total_w: f64 = weights.iter().sum();

        if n_classes == 2 {
            let mut w_pos = 0.0;
            for i in 0..x.rows {
                if class_indices[i] == 1 {
                    w_pos += weights[i];
                }
            }
            let p1 = (w_pos / total_w).clamp(1e-15, 1.0 - 1e-15);
            let baseline_val = (p1 / (1.0 - p1)).ln();

            let mut raw = vec![baseline_val; x.rows];
            let mut grad = vec![0.0; x.rows];
            let mut hess = vec![0.0; x.rows];
            let mut trees = Vec::with_capacity(self.max_iter);

            for _ in 0..self.max_iter {
                for i in 0..x.rows {
                    let p = 1.0 / (1.0 + (-raw[i]).exp());
                    let yi = if class_indices[i] == 1 { 1.0 } else { 0.0 };
                    grad[i] = weights[i] * (p - yi);
                    hess[i] = weights[i] * (p * (1.0 - p)).max(1e-15);
                }
                let tree = build_hgb_tree(&backend, &binned, &grad, &hess, &cfg, &mut raw, None)?;
                trees.push(tree);
            }

            self.baseline_ = vec![baseline_val];
            self.trees_ = trees;
        } else {
            let mut class_w = vec![0.0; n_classes];
            for i in 0..x.rows {
                class_w[class_indices[i]] += weights[i];
            }
            let baseline: Vec<f64> = class_w
                .iter()
                .map(|&cw| (cw / total_w).max(1e-15).ln())
                .collect();

            let mut raws: Vec<Vec<f64>> = baseline.iter().map(|&b| vec![b; x.rows]).collect();
            let mut probs = vec![0.0; x.rows * n_classes];
            let mut grad = vec![0.0; x.rows];
            let mut hess = vec![0.0; x.rows];
            let mut trees = Vec::with_capacity(self.max_iter * n_classes);

            for _ in 0..self.max_iter {
                for i in 0..x.rows {
                    let mut max_r = f64::NEG_INFINITY;
                    for raw_k in raws.iter().take(n_classes) {
                        if raw_k[i] > max_r {
                            max_r = raw_k[i];
                        }
                    }
                    let mut sum_exp = 0.0;
                    for k in 0..n_classes {
                        let e = (raws[k][i] - max_r).exp();
                        probs[i * n_classes + k] = e;
                        sum_exp += e;
                    }
                    let inv = 1.0 / sum_exp;
                    for k in 0..n_classes {
                        probs[i * n_classes + k] *= inv;
                    }
                }

                for k in 0..n_classes {
                    for i in 0..x.rows {
                        let p = probs[i * n_classes + k];
                        let yk = if class_indices[i] == k { 1.0 } else { 0.0 };
                        grad[i] = weights[i] * (p - yk);
                        hess[i] = weights[i] * (p * (1.0 - p)).max(1e-15);
                    }
                    let tree =
                        build_hgb_tree(&backend, &binned, &grad, &hess, &cfg, &mut raws[k], None)?;
                    trees.push(tree);
                }
            }

            self.baseline_ = baseline;
            self.trees_ = trees;
        }

        self.classes_ = classes;
        self.n_iter_ = self.max_iter;
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        validate_predict_matrix(x, self.n_features_in_)?;
        let n_classes = self.classes_.len();
        if n_classes < 2 {
            return Err(OchreError::NotFitted);
        }

        let cols = x.cols;
        if n_classes == 2 {
            let scores = predict_trees_sum(&self.trees_, self.baseline_[0], x);
            return Matrix::from_vec(x.rows, 1, scores);
        }

        let mut data = vec![0.0; x.rows * n_classes];
        for r in 0..x.rows {
            let row = &x.data[r * cols..(r + 1) * cols];
            for k in 0..n_classes {
                data[r * n_classes + k] = self.baseline_[k];
            }
            for iter in 0..self.n_iter_ {
                let base = iter * n_classes;
                for k in 0..n_classes {
                    data[r * n_classes + k] += self.trees_[base + k].predict_row(row);
                }
            }
        }
        Matrix::from_vec(x.rows, n_classes, data)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let df = self.decision_function(x)?;
        let n_classes = self.classes_.len();
        let mut data = Vec::with_capacity(x.rows * n_classes);
        if n_classes == 2 {
            for &s in &df.data {
                let p1 = 1.0 / (1.0 + (-s).exp());
                data.push(1.0 - p1);
                data.push(p1);
            }
        } else {
            for r in 0..x.rows {
                let row = &df.data[r * n_classes..(r + 1) * n_classes];
                let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_e = 0.0;
                let start = data.len();
                for &v in row {
                    let e = (v - max_v).exp();
                    data.push(e);
                    sum_e += e;
                }
                let inv = 1.0 / sum_e;
                for v in &mut data[start..start + n_classes] {
                    *v *= inv;
                }
            }
        }
        Matrix::from_vec(x.rows, n_classes, data)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let proba = self.predict_proba(x)?;
        let n_classes = self.classes_.len();
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row = &proba.data[r * n_classes..(r + 1) * n_classes];
            let mut best_k = 0;
            let mut best_p = f64::NEG_INFINITY;
            for (k, &p) in row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_k = k;
                }
            }
            preds.push(self.classes_[best_k]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        let preds = self.predict(x)?;
        if preds.is_empty() {
            return Ok(0.0);
        }
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, actual)| (**p - **actual).abs() < 1e-9)
            .count();
        Ok((correct as f64) / (y.len() as f64))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_HIST_GB_CLASSIFIER);
        write_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.max_iter);
        write_opt_usize(&mut buf, self.max_leaf_nodes);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_leaf);
        write_f64(&mut buf, self.l2_regularization);
        write_usize(&mut buf, self.max_bins);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_usize(&mut buf, self.n_iter_);
        write_f64_vec(&mut buf, &self.baseline_);
        write_usize(&mut buf, self.trees_.len());
        for tree in &self.trees_ {
            tree.write_to_buf(&mut buf);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_HIST_GB_CLASSIFIER)?;
        let learning_rate = r.read_f64()?;
        let max_iter = r.read_usize()?;
        let max_leaf_nodes = r.read_opt_usize()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let l2_regularization = r.read_f64()?;
        let max_bins = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let n_iter_ = r.read_usize()?;
        let baseline_ = r.read_f64_vec()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(HgbNode::read_from_reader(&mut r, 0)?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            learning_rate,
            max_iter,
            max_leaf_nodes,
            max_depth,
            min_samples_leaf,
            l2_regularization,
            max_bins,
            random_state,
            device: dev,
            device_,
            classes_,
            n_features_in_,
            n_iter_,
            baseline_,
            trees_,
        })
    }
}

// =========================================================================
// PYO3 WRAPPERS
// =========================================================================

#[pyclass(from_py_object, name = "HistGradientBoostingRegressor")]
#[derive(Clone)]
pub struct PyHistGradientBoostingRegressor {
    inner: HistGradientBoostingRegressor,
}

pymethods_with_dit!(
    PyHistGradientBoostingRegressor,
    HistGradientBoostingRegressor,
    {
        #[new]
        #[pyo3(signature = (
            loss = None,
            learning_rate = None,
            max_iter = None,
            max_leaf_nodes = Some(31),
            max_depth = None,
            min_samples_leaf = None,
            l2_regularization = None,
            max_bins = None,
            random_state = None,
            device = None
        ))]
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            loss: Option<String>,
            learning_rate: Option<f64>,
            max_iter: Option<usize>,
            max_leaf_nodes: Option<usize>,
            max_depth: Option<usize>,
            min_samples_leaf: Option<usize>,
            l2_regularization: Option<f64>,
            max_bins: Option<usize>,
            random_state: Option<u64>,
            device: Option<String>,
        ) -> PyResult<Self> {
            let inner = HistGradientBoostingRegressor::new(
                loss,
                learning_rate,
                max_iter,
                max_leaf_nodes,
                max_depth,
                min_samples_leaf,
                l2_regularization,
                max_bins,
                random_state,
                device,
            )?;
            Ok(Self { inner })
        }

        #[pyo3(signature = (x, y, sample_weight = None))]
        pub fn fit_buffer(
            &mut self,
            py: Python<'_>,
            x: PyBuffer<f64>,
            y: PyBuffer<f64>,
            sample_weight: Option<PyBuffer<f64>>,
        ) -> PyResult<()> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let target = buffer_to_f64_1d(py, &y)?;
            let sw = match sample_weight {
                Some(ref buf) => Some(buffer_to_f64_1d(py, buf)?),
                None => None,
            };
            self.inner.fit(&mat, &target, sw.as_deref())?;
            Ok(())
        }

        pub fn predict_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let preds = self.inner.predict(&mat)?;
            f64_slice_to_pybytes(py, &preds)
        }

        pub fn score_buffer(
            &self,
            py: Python<'_>,
            x: PyBuffer<f64>,
            y: PyBuffer<f64>,
        ) -> PyResult<f64> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let target = buffer_to_f64_1d(py, &y)?;
            Ok(self.inner.score(&mat, &target)?)
        }

        #[getter]
        pub fn loss(&self) -> String {
            self.inner.loss.clone()
        }

        #[getter]
        pub fn learning_rate(&self) -> f64 {
            self.inner.learning_rate
        }

        #[getter]
        pub fn max_iter(&self) -> usize {
            self.inner.max_iter
        }

        #[getter]
        pub fn max_leaf_nodes(&self) -> Option<usize> {
            self.inner.max_leaf_nodes
        }

        #[getter]
        pub fn max_depth(&self) -> Option<usize> {
            self.inner.max_depth
        }

        #[getter]
        pub fn min_samples_leaf(&self) -> usize {
            self.inner.min_samples_leaf
        }

        #[getter]
        pub fn l2_regularization(&self) -> f64 {
            self.inner.l2_regularization
        }

        #[getter]
        pub fn max_bins(&self) -> usize {
            self.inner.max_bins
        }

        #[getter]
        pub fn random_state(&self) -> Option<u64> {
            self.inner.random_state
        }

        #[getter]
        pub fn n_features_in_(&self) -> Option<usize> {
            self.inner.n_features_in_
        }

        #[getter]
        pub fn n_iter_(&self) -> usize {
            self.inner.n_iter_
        }
    }
);

#[pyclass(from_py_object, name = "HistGradientBoostingClassifier")]
#[derive(Clone)]
pub struct PyHistGradientBoostingClassifier {
    inner: HistGradientBoostingClassifier,
}

pymethods_with_dit!(
    PyHistGradientBoostingClassifier,
    HistGradientBoostingClassifier,
    {
        #[new]
        #[pyo3(signature = (
            learning_rate = None,
            max_iter = None,
            max_leaf_nodes = Some(31),
            max_depth = None,
            min_samples_leaf = None,
            l2_regularization = None,
            max_bins = None,
            random_state = None,
            device = None
        ))]
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            learning_rate: Option<f64>,
            max_iter: Option<usize>,
            max_leaf_nodes: Option<usize>,
            max_depth: Option<usize>,
            min_samples_leaf: Option<usize>,
            l2_regularization: Option<f64>,
            max_bins: Option<usize>,
            random_state: Option<u64>,
            device: Option<String>,
        ) -> PyResult<Self> {
            let inner = HistGradientBoostingClassifier::new(
                learning_rate,
                max_iter,
                max_leaf_nodes,
                max_depth,
                min_samples_leaf,
                l2_regularization,
                max_bins,
                random_state,
                device,
            )?;
            Ok(Self { inner })
        }

        #[pyo3(signature = (x, y, sample_weight = None))]
        pub fn fit_buffer(
            &mut self,
            py: Python<'_>,
            x: PyBuffer<f64>,
            y: PyBuffer<f64>,
            sample_weight: Option<PyBuffer<f64>>,
        ) -> PyResult<()> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let target = buffer_to_f64_1d(py, &y)?;
            let sw = match sample_weight {
                Some(ref buf) => Some(buffer_to_f64_1d(py, buf)?),
                None => None,
            };
            self.inner.fit(&mat, &target, sw.as_deref())?;
            Ok(())
        }

        pub fn predict_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let preds = self.inner.predict(&mat)?;
            f64_slice_to_pybytes(py, &preds)
        }

        pub fn predict_proba_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let proba = self.inner.predict_proba(&mat)?;
            matrix_to_pybytes(py, &proba)
        }

        pub fn decision_function_buffer<'py>(
            &self,
            py: Python<'py>,
            x: PyBuffer<f64>,
        ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let df = self.inner.decision_function(&mat)?;
            matrix_to_pybytes(py, &df)
        }

        pub fn score_buffer(
            &self,
            py: Python<'_>,
            x: PyBuffer<f64>,
            y: PyBuffer<f64>,
        ) -> PyResult<f64> {
            let mat = buffer_to_pooled_matrix(py, &x)?;
            let target = buffer_to_f64_1d(py, &y)?;
            Ok(self.inner.score(&mat, &target)?)
        }

        #[getter]
        pub fn classes_(&self) -> Vec<f64> {
            self.inner.classes_.clone()
        }

        #[getter]
        pub fn learning_rate(&self) -> f64 {
            self.inner.learning_rate
        }

        #[getter]
        pub fn max_iter(&self) -> usize {
            self.inner.max_iter
        }

        #[getter]
        pub fn max_leaf_nodes(&self) -> Option<usize> {
            self.inner.max_leaf_nodes
        }

        #[getter]
        pub fn max_depth(&self) -> Option<usize> {
            self.inner.max_depth
        }

        #[getter]
        pub fn min_samples_leaf(&self) -> usize {
            self.inner.min_samples_leaf
        }

        #[getter]
        pub fn l2_regularization(&self) -> f64 {
            self.inner.l2_regularization
        }

        #[getter]
        pub fn max_bins(&self) -> usize {
            self.inner.max_bins
        }

        #[getter]
        pub fn random_state(&self) -> Option<u64> {
            self.inner.random_state
        }

        #[getter]
        pub fn n_features_in_(&self) -> Option<usize> {
            self.inner.n_features_in_
        }

        #[getter]
        pub fn n_iter_(&self) -> usize {
            self.inner.n_iter_
        }
    }
);

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHistGradientBoostingClassifier>()?;
    m.add_class::<PyHistGradientBoostingRegressor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hgb_regressor_squared_and_absolute() {
        let mut rows = Vec::with_capacity(80);
        let mut y = Vec::with_capacity(80);
        for i in 0..80 {
            let x0 = (i as f64) / 20.0;
            let x1 = ((i * 3) % 11) as f64 / 11.0;
            rows.push(vec![x0, x1]);
            y.push(2.5 * x0 + 1.0);
        }
        let x = Matrix::from_2d(&rows).unwrap();

        let mut reg = HistGradientBoostingRegressor::new(
            Some("squared_error".to_string()),
            Some(0.15),
            Some(40),
            Some(15),
            None,
            Some(5),
            Some(0.0),
            Some(64),
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        reg.fit(&x, &y, None).unwrap();
        assert!(reg.score(&x, &y).unwrap() > 0.95);

        let bytes = reg.to_bytes();
        let loaded = HistGradientBoostingRegressor::from_bytes(&bytes, None).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), reg.predict(&x).unwrap());

        let mut reg_abs = HistGradientBoostingRegressor::new(
            Some("absolute_error".to_string()),
            Some(0.15),
            Some(40),
            Some(15),
            None,
            Some(5),
            Some(0.0),
            Some(64),
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        reg_abs.fit(&x, &y, None).unwrap();
        assert!(reg_abs.score(&x, &y).unwrap() > 0.90);
    }

    #[test]
    fn test_hgb_classifier_binary_and_multiclass() {
        let mut rows = Vec::with_capacity(90);
        let mut y_bin = Vec::with_capacity(90);
        let mut y_multi = Vec::with_capacity(90);
        for i in 0..90 {
            let x0 = (i as f64) / 90.0;
            let x1 = ((i * 5) % 13) as f64 / 13.0;
            rows.push(vec![x0, x1]);
            y_bin.push(if x0 >= 0.5 { 1.0 } else { 0.0 });
            y_multi.push(if x0 < 0.33 {
                0.0
            } else if x0 < 0.66 {
                1.0
            } else {
                2.0
            });
        }
        let x = Matrix::from_2d(&rows).unwrap();

        let mut clf_bin = HistGradientBoostingClassifier::new(
            Some(0.15),
            Some(30),
            Some(15),
            None,
            Some(5),
            Some(0.0),
            Some(64),
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        clf_bin.fit(&x, &y_bin, None).unwrap();
        assert!(clf_bin.score(&x, &y_bin).unwrap() > 0.95);

        let mut clf_multi = HistGradientBoostingClassifier::new(
            Some(0.15),
            Some(30),
            Some(15),
            None,
            Some(5),
            Some(0.0),
            Some(64),
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        clf_multi.fit(&x, &y_multi, None).unwrap();
        assert!(clf_multi.score(&x, &y_multi).unwrap() > 0.95);

        let bytes = clf_multi.to_bytes();
        let loaded = HistGradientBoostingClassifier::from_bytes(&bytes, None).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), clf_multi.predict(&x).unwrap());
    }
}
