// OchreML Wave 4 Boosting Ensembles Module (*≧ω≦*)
// Implements GradientBoostingRegressor, GradientBoostingClassifier,
// AdaBoostClassifier (SAMME), and AdaBoostRegressor (AdaBoost.R2)
// accelerated with N6 (histogram split), N2 (elementwise/softmax), and Rayon.

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_f64, write_f64_vec, write_header, write_opt_f64_vec,
    write_opt_str, write_opt_u64, write_opt_usize, write_str, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::logistic_regression::{extract_sorted_classes, map_labels_to_indices};
use crate::matrix::{Matrix, r2_score};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::primitives::{BinnedMatrix, ElementwiseOp, GradHessBin, subtract_histogram};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub const TAG_GB_CLASSIFIER: u8 = 71;
pub const TAG_GB_REGRESSOR: u8 = 72;
pub const TAG_ADABOOST_CLASSIFIER: u8 = 73;
pub const TAG_ADABOOST_REGRESSOR: u8 = 74;

const N2_THRESHOLD: usize = 4096;
const HIST_MAX_BINS: usize = 256;

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) & Shared Validation Helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: Option<u64>) -> Self {
        let s = match seed {
            Some(v) => v,
            None => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15),
        };
        Self {
            state: s.wrapping_add(0x9E37_79B9_7F4A_7C15),
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
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Partial Fisher-Yates sample without replacement of `k` items from `0..n`.
    fn sample_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let take = k.min(n);
        for i in 0..take {
            let rem = n - i;
            let j = i + (self.next_u64() as usize % rem);
            pool.swap(i, j);
        }
        pool.truncate(take);
        pool
    }
}

fn validate_matrix_finite(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_y_finite(y: &[f64], expected_rows: usize) -> OchreResult<()> {
    if y.len() != expected_rows {
        return Err(OchreError::LengthMismatch {
            features: expected_rows,
            targets: y.len(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Target y contains NaN or Inf values ( >_< )".to_string(),
        ));
    }
    Ok(())
}

fn validate_and_prepare_weights(sw: Option<&[f64]>, n_samples: usize) -> OchreResult<Vec<f64>> {
    match sw {
        Some(w) => {
            if w.len() != n_samples {
                return Err(OchreError::WeightLengthMismatch {
                    weights: w.len(),
                    samples: n_samples,
                });
            }
            let mut sum = 0.0;
            for &val in w {
                if !val.is_finite() || val < 0.0 {
                    return Err(OchreError::InvalidWeights);
                }
                sum += val;
            }
            if sum <= 0.0 {
                return Err(OchreError::NonPositiveWeightSum);
            }
            Ok(w.to_vec())
        }
        None => Ok(vec![1.0; n_samples]),
    }
}

fn validate_positive_f64(val: f64, name: &str) -> OchreResult<()> {
    if !val.is_finite() || val <= 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "{name} ({val}) must be a strictly positive finite number (・`ω´・)"
        )));
    }
    Ok(())
}

fn validate_gb_common(
    learning_rate: f64,
    n_estimators: usize,
    subsample: f64,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: usize,
) -> OchreResult<()> {
    validate_positive_f64(learning_rate, "learning_rate")?;
    if n_estimators == 0 {
        return Err(OchreError::InvalidParameter(
            "n_estimators must be >= 1 ( >_< )".to_string(),
        ));
    }
    if !subsample.is_finite() || subsample <= 0.0 || subsample > 1.0 {
        return Err(OchreError::InvalidParameter(format!(
            "subsample ({subsample}) must be in (0.0, 1.0] (・`ω´・)"
        )));
    }
    if min_samples_split < 2 {
        return Err(OchreError::InvalidParameter(format!(
            "min_samples_split ({min_samples_split}) must be >= 2 ( >_< )"
        )));
    }
    if min_samples_leaf < 1 {
        return Err(OchreError::InvalidParameter(format!(
            "min_samples_leaf ({min_samples_leaf}) must be >= 1 ( >_< )"
        )));
    }
    if max_depth < 1 {
        return Err(OchreError::InvalidParameter(format!(
            "max_depth ({max_depth}) must be >= 1 ( >_< )"
        )));
    }
    Ok(())
}

fn weighted_mean(vals: &[f64], weights: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&v, &w) in vals.iter().zip(weights.iter()) {
        num += w * v;
        den += w;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

fn weighted_quantile(vals: &[f64], weights: &[f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut pairs: Vec<(f64, f64)> = vals
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(_, w)| *w > 0.0)
        .collect();
    if pairs.is_empty() {
        return vals[0];
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_w: f64 = pairs.iter().map(|(_, w)| *w).sum();
    let target = q.clamp(0.0, 1.0) * total_w;
    let mut cum = 0.0;
    for (v, w) in &pairs {
        cum += *w;
        if cum >= target {
            return *v;
        }
    }
    pairs.last().map_or(0.0, |(v, _)| *v)
}

#[inline]
fn weighted_median(vals: &[f64], weights: &[f64]) -> f64 {
    weighted_quantile(vals, weights, 0.5)
}

fn normalize_importances(imp: &mut [f64]) {
    let sum: f64 = imp.iter().sum();
    if sum > 0.0 {
        for v in imp.iter_mut() {
            *v /= sum;
        }
    }
}

fn check_predict_input(x: &Matrix, n_features_in: Option<usize>) -> OchreResult<usize> {
    let n_feat = n_features_in.ok_or(OchreError::NotFitted)?;
    validate_matrix_finite(x)?;
    if x.cols != n_feat {
        return Err(OchreError::FeatureCountMismatch {
            features: x.cols,
            expected: n_feat,
        });
    }
    Ok(n_feat)
}

// ---------------------------------------------------------------------------
// Regression Tree (for GradientBoosting & AdaBoostRegressor)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum RegTreeNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<RegTreeNode>,
        right: Box<RegTreeNode>,
    },
}

impl RegTreeNode {
    #[inline]
    fn predict_row(&self, row: &[f64]) -> f64 {
        let mut node = self;
        loop {
            match node {
                RegTreeNode::Leaf { value } => return *value,
                RegTreeNode::Internal {
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

    fn predict_matrix(&self, x: &Matrix) -> Vec<f64> {
        let cols = x.cols;
        if x.rows >= 1024 {
            x.data
                .par_chunks_exact(cols)
                .map(|row| self.predict_row(row))
                .collect()
        } else {
            x.data
                .chunks_exact(cols)
                .map(|row| self.predict_row(row))
                .collect()
        }
    }
}

fn write_reg_tree(buf: &mut Vec<u8>, node: &RegTreeNode) {
    match node {
        RegTreeNode::Leaf { value } => {
            write_u8(buf, 0);
            write_f64(buf, *value);
        }
        RegTreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            write_u8(buf, 1);
            write_usize(buf, *feature_index);
            write_f64(buf, *threshold);
            write_reg_tree(buf, left);
            write_reg_tree(buf, right);
        }
    }
}

fn read_reg_tree(reader: &mut ByteReader<'_>) -> OchreResult<RegTreeNode> {
    let tag = reader.read_u8()?;
    match tag {
        0 => {
            let value = reader.read_f64()?;
            Ok(RegTreeNode::Leaf { value })
        }
        1 => {
            let feature_index = reader.read_usize()?;
            let threshold = reader.read_f64()?;
            let left = read_reg_tree(reader)?;
            let right = read_reg_tree(reader)?;
            Ok(RegTreeNode::Internal {
                feature_index,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        _ => Err(OchreError::InvalidTreeNodeTag),
    }
}

#[derive(Clone, Copy)]
enum LeafLossRule<'a> {
    MeanTarget,
    BinaryLogLoss { probs: &'a [f64] },
    MulticlassLogLoss { k_classes: usize },
    AbsoluteError { residuals: &'a [f64] },
    Huber { residuals: &'a [f64], delta: f64 },
}

fn compute_leaf_value(
    indices: &[usize],
    target: &[f64],
    weights: &[f64],
    rule: LeafLossRule<'_>,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    match rule {
        LeafLossRule::MeanTarget => {
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in indices {
                num += weights[i] * target[i];
                den += weights[i];
            }
            if den > 0.0 { num / den } else { 0.0 }
        }
        LeafLossRule::BinaryLogLoss { probs } => {
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in indices {
                num += weights[i] * target[i];
                let p = probs[i];
                den += weights[i] * p * (1.0 - p);
            }
            if den > 1e-15 { num / den } else { 0.0 }
        }
        LeafLossRule::MulticlassLogLoss { k_classes } => {
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in indices {
                let r = target[i];
                let abs_r = r.abs();
                num += weights[i] * r;
                den += weights[i] * abs_r * (1.0 - abs_r);
            }
            if den > 1e-15 {
                let factor = (k_classes as f64 - 1.0) / (k_classes as f64);
                factor * (num / den)
            } else {
                0.0
            }
        }
        LeafLossRule::AbsoluteError { residuals } => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut ws = Vec::with_capacity(indices.len());
            for &i in indices {
                vals.push(residuals[i]);
                ws.push(weights[i]);
            }
            weighted_median(&vals, &ws)
        }
        LeafLossRule::Huber { residuals, delta } => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut ws = Vec::with_capacity(indices.len());
            let mut w_sum = 0.0;
            for &i in indices {
                vals.push(residuals[i]);
                ws.push(weights[i]);
                w_sum += weights[i];
            }
            if w_sum <= 0.0 {
                return 0.0;
            }
            let med = weighted_median(&vals, &ws);
            let mut diff_sum = 0.0;
            for (&r, &w) in vals.iter().zip(ws.iter()) {
                let d = r - med;
                diff_sum += w * d.signum() * d.abs().min(delta);
            }
            med + diff_sum / w_sum
        }
    }
}

struct RegTreeConfig {
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_reg_tree_cpu(
    x: &Matrix,
    target: &[f64],
    weights: &[f64],
    indices: &[usize],
    depth: usize,
    cfg: &RegTreeConfig,
    rule: LeafLossRule<'_>,
    importances: &mut [f64],
) -> RegTreeNode {
    let n = indices.len();
    if depth >= cfg.max_depth || n < cfg.min_samples_split || n < 2 * cfg.min_samples_leaf {
        return RegTreeNode::Leaf {
            value: compute_leaf_value(indices, target, weights, rule),
        };
    }

    let mut total_w = 0.0;
    let mut total_sum = 0.0;
    for &i in indices {
        let w = weights[i];
        total_w += w;
        total_sum += w * target[i];
    }
    if total_w <= 1e-15 {
        return RegTreeNode::Leaf {
            value: compute_leaf_value(indices, target, weights, rule),
        };
    }
    let parent_score = (total_sum * total_sum) / total_w;

    // Parallel feature split search via Rayon
    let best = (0..x.cols)
        .into_par_iter()
        .filter_map(|f| {
            let mut pairs: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
            for &i in indices {
                pairs.push((x.data[i * x.cols + f], target[i], weights[i]));
            }
            pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            if (pairs[n - 1].0 - pairs[0].0).abs() <= 1e-12 {
                return None;
            }

            let mut left_w = 0.0;
            let mut left_sum = 0.0;
            let mut best_gain = 1e-12;
            let mut best_thresh = 0.0;
            let mut found = false;

            for i in 0..(n - 1) {
                left_w += pairs[i].2;
                left_sum += pairs[i].2 * pairs[i].1;

                let cur_x = pairs[i].0;
                let next_x = pairs[i + 1].0;
                if (next_x - cur_x).abs() <= 1e-12 {
                    continue;
                }

                let left_n = i + 1;
                let right_n = n - left_n;
                if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                    continue;
                }

                let right_w = total_w - left_w;
                if left_w <= 1e-15 || right_w <= 1e-15 {
                    continue;
                }
                let right_sum = total_sum - left_sum;
                let gain = (left_sum * left_sum) / left_w + (right_sum * right_sum) / right_w
                    - parent_score;

                if gain > best_gain {
                    best_gain = gain;
                    best_thresh = 0.5 * (cur_x + next_x);
                    found = true;
                }
            }

            if found {
                Some((best_gain, f, best_thresh))
            } else {
                None
            }
        })
        .reduce_with(|a, b| if a.0 >= b.0 { a } else { b });

    let Some((best_gain, best_f, best_thresh)) = best else {
        return RegTreeNode::Leaf {
            value: compute_leaf_value(indices, target, weights, rule),
        };
    };

    let mut left_idx = Vec::with_capacity(n / 2 + 1);
    let mut right_idx = Vec::with_capacity(n / 2 + 1);
    for &i in indices {
        if x.data[i * x.cols + best_f] <= best_thresh {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    if left_idx.is_empty() || right_idx.is_empty() {
        return RegTreeNode::Leaf {
            value: compute_leaf_value(indices, target, weights, rule),
        };
    }

    importances[best_f] += best_gain;

    let left = build_reg_tree_cpu(
        x,
        target,
        weights,
        &left_idx,
        depth + 1,
        cfg,
        rule,
        importances,
    );
    let right = build_reg_tree_cpu(
        x,
        target,
        weights,
        &right_idx,
        depth + 1,
        cfg,
        rule,
        importances,
    );

    RegTreeNode::Internal {
        feature_index: best_f,
        threshold: best_thresh,
        left: Box::new(left),
        right: Box::new(right),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_reg_tree_n6(
    backend: &Backend<'_>,
    binned: &BinnedMatrix,
    x: &Matrix,
    target: &[f64],
    weights: &[f64],
    grad: &[f64],
    hess: &[f64],
    indices_u32: &[u32],
    parent_hist: Option<Vec<GradHessBin>>,
    depth: usize,
    cfg: &RegTreeConfig,
    rule: LeafLossRule<'_>,
    importances: &mut [f64],
) -> OchreResult<RegTreeNode> {
    let n = indices_u32.len();
    if depth >= cfg.max_depth || n < cfg.min_samples_split || n < 2 * cfg.min_samples_leaf {
        let idx_usize: Vec<usize> = indices_u32.iter().map(|&i| i as usize).collect();
        return Ok(RegTreeNode::Leaf {
            value: compute_leaf_value(&idx_usize, target, weights, rule),
        });
    }

    let hist = match parent_hist {
        Some(h) => h,
        None => backend.batched_histogram_grad(binned, indices_u32, grad, hess)?,
    };

    let split_opt =
        backend.find_best_histogram_split(binned, &hist, 0.0, cfg.min_samples_leaf as u32)?;

    let Some(split) = split_opt else {
        let idx_usize: Vec<usize> = indices_u32.iter().map(|&i| i as usize).collect();
        return Ok(RegTreeNode::Leaf {
            value: compute_leaf_value(&idx_usize, target, weights, rule),
        });
    };

    if split.gain <= 1e-12 {
        let idx_usize: Vec<usize> = indices_u32.iter().map(|&i| i as usize).collect();
        return Ok(RegTreeNode::Leaf {
            value: compute_leaf_value(&idx_usize, target, weights, rule),
        });
    }

    let mut left_idx = Vec::with_capacity(split.left_count as usize);
    let mut right_idx = Vec::with_capacity(n.saturating_sub(split.left_count as usize));
    for &idx in indices_u32 {
        let r = idx as usize;
        if x.data[r * x.cols + split.feature_idx] <= split.threshold {
            left_idx.push(idx);
        } else {
            right_idx.push(idx);
        }
    }

    if left_idx.is_empty() || right_idx.is_empty() {
        let idx_usize: Vec<usize> = indices_u32.iter().map(|&i| i as usize).collect();
        return Ok(RegTreeNode::Leaf {
            value: compute_leaf_value(&idx_usize, target, weights, rule),
        });
    }

    importances[split.feature_idx] += split.gain;

    // Histogram subtraction trick: compute smaller child, subtract from parent for larger child
    let (left_hist, right_hist) = if left_idx.len() <= right_idx.len() {
        let lh = backend.batched_histogram_grad(binned, &left_idx, grad, hess)?;
        let rh = subtract_histogram(&hist, &lh)?;
        (lh, rh)
    } else {
        let rh = backend.batched_histogram_grad(binned, &right_idx, grad, hess)?;
        let lh = subtract_histogram(&hist, &rh)?;
        (lh, rh)
    };

    let left = build_reg_tree_n6(
        backend,
        binned,
        x,
        target,
        weights,
        grad,
        hess,
        &left_idx,
        Some(left_hist),
        depth + 1,
        cfg,
        rule,
        importances,
    )?;
    let right = build_reg_tree_n6(
        backend,
        binned,
        x,
        target,
        weights,
        grad,
        hess,
        &right_idx,
        Some(right_hist),
        depth + 1,
        cfg,
        rule,
        importances,
    )?;

    Ok(RegTreeNode::Internal {
        feature_index: split.feature_idx,
        threshold: split.threshold,
        left: Box::new(left),
        right: Box::new(right),
    })
}

fn use_n6_histogram(backend: &Backend<'_>) -> bool {
    matches!(
        backend,
        Backend::CudaSingle(_) | Backend::CudaMulti(_) | Backend::Wgpu
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_stage_reg_tree(
    backend: &Backend<'_>,
    binned_opt: Option<&BinnedMatrix>,
    x: &Matrix,
    target: &[f64],
    weights: &[f64],
    indices: &[usize],
    cfg: &RegTreeConfig,
    rule: LeafLossRule<'_>,
    importances: &mut [f64],
) -> OchreResult<RegTreeNode> {
    if let Some(binned) = binned_opt {
        let n = x.rows;
        let mut grad = vec![0.0; n];
        let mut hess = vec![0.0; n];
        for &i in indices {
            grad[i] = weights[i] * target[i];
            hess[i] = weights[i];
        }
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        build_reg_tree_n6(
            backend,
            binned,
            x,
            target,
            weights,
            &grad,
            &hess,
            &indices_u32,
            None,
            0,
            cfg,
            rule,
            importances,
        )
    } else {
        Ok(build_reg_tree_cpu(
            x,
            target,
            weights,
            indices,
            0,
            cfg,
            rule,
            importances,
        ))
    }
}

// ---------------------------------------------------------------------------
// 1. GradientBoostingRegressor
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GradientBoostingRegressor {
    pub loss: String,
    pub learning_rate: f64,
    pub n_estimators: usize,
    pub subsample: f64,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
    pub alpha: f64,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    device: DeviceType,
    pub init_pred_: f64,
    trees_: Vec<RegTreeNode>,
    pub feature_importances_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl GradientBoostingRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        learning_rate: f64,
        n_estimators: usize,
        subsample: f64,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_depth: usize,
        alpha: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let loss_lc = loss.to_lowercase();
        if loss_lc != "squared_error" && loss_lc != "absolute_error" && loss_lc != "huber" {
            return Err(OchreError::InvalidParameter(format!(
                "loss '{loss}' must be 'squared_error', 'absolute_error', or 'huber' (・`ω´・)"
            )));
        }
        validate_gb_common(
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
        )?;
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(OchreError::InvalidParameter(format!(
                "alpha ({alpha}) must be in (0.0, 1.0) (・`ω´・)"
            )));
        }
        let resolved = device.or_else(|| Some("auto".to_string()));
        let dev = parse_device(resolved.as_deref())?;
        Ok(Self {
            loss: loss_lc,
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            alpha,
            random_state,
            device_: resolved,
            device: dev,
            init_pred_: 0.0,
            trees_: Vec::new(),
            feature_importances_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.or_else(|| Some("auto".to_string()));
        self.device = parse_device(resolved.as_deref())?;
        self.device_ = resolved;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let weights = validate_and_prepare_weights(sample_weight, x.rows)?;
        let n_samples = x.rows;
        let n_features = x.cols;

        let backend = get_backend(&self.device);
        let binned = if use_n6_histogram(&backend) {
            Some(backend.quantize_features(x, HIST_MAX_BINS)?)
        } else {
            None
        };

        let init = if self.loss == "squared_error" {
            weighted_mean(y, &weights)
        } else {
            weighted_median(y, &weights)
        };

        let mut f_pred = vec![init; n_samples];
        let mut residuals = vec![0.0; n_samples];
        let mut pseudo_res = vec![0.0; n_samples];
        let mut importances = vec![0.0; n_features];
        let mut trees = Vec::with_capacity(self.n_estimators);

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let n_sub = if self.subsample < 1.0 {
            ((n_samples as f64) * self.subsample).round().max(1.0) as usize
        } else {
            n_samples
        };
        let mut rng = Rng64::new(self.random_state);
        let cfg = RegTreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        for _stage in 0..self.n_estimators {
            for i in 0..n_samples {
                residuals[i] = y[i] - f_pred[i];
            }

            let stage_indices_owned;
            let stage_indices: &[usize] = if self.subsample < 1.0 {
                stage_indices_owned = rng.sample_without_replacement(n_samples, n_sub);
                &stage_indices_owned
            } else {
                &all_indices
            };

            let (target_slice, rule) = match self.loss.as_str() {
                "squared_error" => (&residuals[..], LeafLossRule::MeanTarget),
                "absolute_error" => {
                    for i in 0..n_samples {
                        let e = residuals[i];
                        pseudo_res[i] = if e > 0.0 {
                            1.0
                        } else if e < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                    (
                        &pseudo_res[..],
                        LeafLossRule::AbsoluteError {
                            residuals: &residuals,
                        },
                    )
                }
                _ => {
                    // Huber loss
                    let mut abs_res = Vec::with_capacity(stage_indices.len());
                    let mut sub_w = Vec::with_capacity(stage_indices.len());
                    for &i in stage_indices {
                        abs_res.push(residuals[i].abs());
                        sub_w.push(weights[i]);
                    }
                    let delta = weighted_quantile(&abs_res, &sub_w, self.alpha).max(1e-12);
                    for i in 0..n_samples {
                        let e = residuals[i];
                        if e.abs() <= delta {
                            pseudo_res[i] = e;
                        } else {
                            pseudo_res[i] = delta * e.signum();
                        }
                    }
                    (
                        &pseudo_res[..],
                        LeafLossRule::Huber {
                            residuals: &residuals,
                            delta,
                        },
                    )
                }
            };

            let tree = fit_stage_reg_tree(
                &backend,
                binned.as_ref(),
                x,
                target_slice,
                &weights,
                stage_indices,
                &cfg,
                rule,
                &mut importances,
            )?;

            let tree_preds = tree.predict_matrix(x);
            for i in 0..n_samples {
                f_pred[i] += self.learning_rate * tree_preds[i];
            }
            trees.push(tree);
        }

        normalize_importances(&mut importances);
        self.init_pred_ = init;
        self.trees_ = trees;
        self.feature_importances_ = Some(importances);
        self.n_features_in_ = Some(n_features);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        check_predict_input(x, self.n_features_in_)?;
        let cols = x.cols;
        let init = self.init_pred_;
        let lr = self.learning_rate;
        let preds: Vec<f64> = if x.rows >= 512 {
            x.data
                .par_chunks_exact(cols)
                .map(|row| {
                    let mut sum = init;
                    for tree in &self.trees_ {
                        sum += lr * tree.predict_row(row);
                    }
                    sum
                })
                .collect()
        } else {
            x.data
                .chunks_exact(cols)
                .map(|row| {
                    let mut sum = init;
                    for tree in &self.trees_ {
                        sum += lr * tree.predict_row(row);
                    }
                    sum
                })
                .collect()
        };
        Ok(preds)
    }

    pub fn staged_predict(&self, x: &Matrix) -> OchreResult<Matrix> {
        check_predict_input(x, self.n_features_in_)?;
        let n_stages = self.trees_.len();
        let n_samples = x.rows;
        let mut out = vec![0.0; n_stages * n_samples];
        let mut current = vec![self.init_pred_; n_samples];
        for (m, tree) in self.trees_.iter().enumerate() {
            let tree_preds = tree.predict_matrix(x);
            let row_off = m * n_samples;
            for (i, &tp) in tree_preds.iter().enumerate().take(n_samples) {
                current[i] += self.learning_rate * tp;
                out[row_off + i] = current[i];
            }
        }
        Matrix::from_vec(n_stages, n_samples, out)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GB_REGRESSOR);
        write_str(&mut buf, &self.loss);
        write_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.n_estimators);
        write_f64(&mut buf, self.subsample);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_usize(&mut buf, self.max_depth);
        write_f64(&mut buf, self.alpha);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_f64(&mut buf, self.init_pred_);
        write_usize(&mut buf, self.trees_.len());
        for t in &self.trees_ {
            write_reg_tree(&mut buf, t);
        }
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_GB_REGRESSOR)?;
        let loss = r.read_str()?;
        let learning_rate = r.read_f64()?;
        let n_estimators = r.read_usize()?;
        let subsample = r.read_f64()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let max_depth = r.read_usize()?;
        let alpha = r.read_f64()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let init_pred_ = r.read_f64()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(read_reg_tree(&mut r)?);
        }
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            loss,
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            alpha,
            random_state,
            device_,
            device: dev,
            init_pred_,
            trees_,
            feature_importances_,
            n_features_in_,
        })
    }
}

// ---------------------------------------------------------------------------
// 2. GradientBoostingClassifier
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GradientBoostingClassifier {
    pub loss: String,
    pub learning_rate: f64,
    pub n_estimators: usize,
    pub subsample: f64,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    init_pred_: Vec<f64>,
    trees_: Vec<RegTreeNode>,
    pub feature_importances_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl GradientBoostingClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        learning_rate: f64,
        n_estimators: usize,
        subsample: f64,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let loss_lc = loss.to_lowercase();
        if loss_lc != "log_loss" {
            return Err(OchreError::InvalidParameter(format!(
                "loss '{loss}' must be 'log_loss' (・`ω´・)"
            )));
        }
        validate_gb_common(
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
        )?;
        let resolved = device.or_else(|| Some("auto".to_string()));
        let dev = parse_device(resolved.as_deref())?;
        Ok(Self {
            loss: loss_lc,
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            random_state,
            device_: resolved,
            device: dev,
            classes_: None,
            init_pred_: Vec::new(),
            trees_: Vec::new(),
            feature_importances_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.or_else(|| Some("auto".to_string()));
        self.device = parse_device(resolved.as_deref())?;
        self.device_ = resolved;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let weights = validate_and_prepare_weights(sample_weight, x.rows)?;
        let classes = extract_sorted_classes(y);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }
        let class_indices = map_labels_to_indices(y, &classes)?;
        let n_samples = x.rows;
        let n_features = x.cols;
        let k_classes = classes.len();

        let backend = get_backend(&self.device);
        let binned = if use_n6_histogram(&backend) {
            Some(backend.quantize_features(x, HIST_MAX_BINS)?)
        } else {
            None
        };

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let n_sub = if self.subsample < 1.0 {
            ((n_samples as f64) * self.subsample).round().max(1.0) as usize
        } else {
            n_samples
        };
        let mut rng = Rng64::new(self.random_state);
        let cfg = RegTreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };
        let mut importances = vec![0.0; n_features];

        if k_classes == 2 {
            let mut y_bin = vec![0.0; n_samples];
            let mut w_pos = 0.0;
            let mut w_tot = 0.0;
            for i in 0..n_samples {
                let yi = if class_indices[i] == 1 { 1.0 } else { 0.0 };
                y_bin[i] = yi;
                w_pos += weights[i] * yi;
                w_tot += weights[i];
            }
            let p_pos = (w_pos / w_tot).clamp(1e-15, 1.0 - 1e-15);
            let init = (p_pos / (1.0 - p_pos)).ln();
            let mut f_scores = vec![init; n_samples];
            let mut probs = vec![0.0; n_samples];
            let mut res = vec![0.0; n_samples];
            let mut trees = Vec::with_capacity(self.n_estimators);

            for _stage in 0..self.n_estimators {
                if n_samples >= N2_THRESHOLD {
                    let f_mat = Matrix::from_slice(n_samples, 1, &f_scores)?;
                    let sig = backend.elementwise_op(&f_mat, ElementwiseOp::Sigmoid)?;
                    probs.copy_from_slice(&sig.data);
                } else {
                    for i in 0..n_samples {
                        probs[i] = 1.0 / (1.0 + (-f_scores[i]).exp());
                    }
                }
                for i in 0..n_samples {
                    res[i] = y_bin[i] - probs[i];
                }

                let stage_indices_owned;
                let stage_indices: &[usize] = if self.subsample < 1.0 {
                    stage_indices_owned = rng.sample_without_replacement(n_samples, n_sub);
                    &stage_indices_owned
                } else {
                    &all_indices
                };

                let tree = fit_stage_reg_tree(
                    &backend,
                    binned.as_ref(),
                    x,
                    &res,
                    &weights,
                    stage_indices,
                    &cfg,
                    LeafLossRule::BinaryLogLoss { probs: &probs },
                    &mut importances,
                )?;
                let preds = tree.predict_matrix(x);
                for i in 0..n_samples {
                    f_scores[i] += self.learning_rate * preds[i];
                }
                trees.push(tree);
            }

            normalize_importances(&mut importances);
            self.classes_ = Some(classes);
            self.init_pred_ = vec![init];
            self.trees_ = trees;
            self.feature_importances_ = Some(importances);
            self.n_features_in_ = Some(n_features);
            return Ok(());
        }

        // Multiclass (K >= 3)
        let mut class_w = vec![0.0; k_classes];
        let mut total_w = 0.0;
        for i in 0..n_samples {
            class_w[class_indices[i]] += weights[i];
            total_w += weights[i];
        }
        let init_vec: Vec<f64> = class_w
            .iter()
            .map(|&cw| (cw / total_w).max(1e-15).ln())
            .collect();

        let mut f_mat = Matrix::zeros(n_samples, k_classes);
        for i in 0..n_samples {
            let row = &mut f_mat.data[i * k_classes..(i + 1) * k_classes];
            row.copy_from_slice(&init_vec);
        }

        let mut trees = Vec::with_capacity(self.n_estimators * k_classes);
        let mut res_k = vec![0.0; n_samples];

        for _stage in 0..self.n_estimators {
            let probs = backend.row_softmax(&f_mat)?;
            let stage_indices_owned;
            let stage_indices: &[usize] = if self.subsample < 1.0 {
                stage_indices_owned = rng.sample_without_replacement(n_samples, n_sub);
                &stage_indices_owned
            } else {
                &all_indices
            };

            for k in 0..k_classes {
                for i in 0..n_samples {
                    let y_ik = if class_indices[i] == k { 1.0 } else { 0.0 };
                    res_k[i] = y_ik - probs.data[i * k_classes + k];
                }
                let tree = fit_stage_reg_tree(
                    &backend,
                    binned.as_ref(),
                    x,
                    &res_k,
                    &weights,
                    stage_indices,
                    &cfg,
                    LeafLossRule::MulticlassLogLoss { k_classes },
                    &mut importances,
                )?;
                let preds = tree.predict_matrix(x);
                for (i, &p) in preds.iter().enumerate().take(n_samples) {
                    f_mat.data[i * k_classes + k] += self.learning_rate * p;
                }
                trees.push(tree);
            }
        }

        normalize_importances(&mut importances);
        self.classes_ = Some(classes);
        self.init_pred_ = init_vec;
        self.trees_ = trees;
        self.feature_importances_ = Some(importances);
        self.n_features_in_ = Some(n_features);
        Ok(())
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        check_predict_input(x, self.n_features_in_)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_classes = classes.len();
        let n_samples = x.rows;
        let cols = x.cols;
        let lr = self.learning_rate;

        if k_classes == 2 {
            let init = self.init_pred_[0];
            let mut out = vec![0.0; n_samples];
            for (r, row) in x.data.chunks_exact(cols).enumerate() {
                let mut sum = init;
                for tree in &self.trees_ {
                    sum += lr * tree.predict_row(row);
                }
                out[r] = sum;
            }
            return Matrix::from_vec(n_samples, 1, out);
        }

        let mut out = vec![0.0; n_samples * k_classes];
        let n_stages = self.trees_.len() / k_classes;
        for (r, row) in x.data.chunks_exact(cols).enumerate() {
            let dst = &mut out[r * k_classes..(r + 1) * k_classes];
            dst.copy_from_slice(&self.init_pred_);
            for m in 0..n_stages {
                let base = m * k_classes;
                for (k, dst_k) in dst.iter_mut().enumerate().take(k_classes) {
                    *dst_k += lr * self.trees_[base + k].predict_row(row);
                }
            }
        }
        Matrix::from_vec(n_samples, k_classes, out)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let df = self.decision_function(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_samples = df.rows;
        if classes.len() == 2 {
            let mut out = vec![0.0; n_samples * 2];
            for i in 0..n_samples {
                let p1 = 1.0 / (1.0 + (-df.data[i]).exp());
                out[i * 2] = 1.0 - p1;
                out[i * 2 + 1] = p1;
            }
            return Matrix::from_vec(n_samples, 2, out);
        }
        let backend = get_backend(&self.device);
        backend.row_softmax(&df)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let df = self.decision_function(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_classes = classes.len();
        let n_samples = df.rows;
        let mut preds = Vec::with_capacity(n_samples);
        if k_classes == 2 {
            for i in 0..n_samples {
                preds.push(if df.data[i] >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                });
            }
            return Ok(preds);
        }
        for i in 0..n_samples {
            let row = &df.data[i * k_classes..(i + 1) * k_classes];
            let mut best_k = 0;
            let mut best_v = row[0];
            for (k, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best_k = k;
                }
            }
            preds.push(classes[best_k]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-12)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_GB_CLASSIFIER);
        write_str(&mut buf, &self.loss);
        write_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.n_estimators);
        write_f64(&mut buf, self.subsample);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_usize(&mut buf, self.max_depth);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_f64_vec(&mut buf, &self.init_pred_);
        write_usize(&mut buf, self.trees_.len());
        for t in &self.trees_ {
            write_reg_tree(&mut buf, t);
        }
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_GB_CLASSIFIER)?;
        let loss = r.read_str()?;
        let learning_rate = r.read_f64()?;
        let n_estimators = r.read_usize()?;
        let subsample = r.read_f64()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let max_depth = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let init_pred_ = r.read_f64_vec()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(read_reg_tree(&mut r)?);
        }
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            loss,
            learning_rate,
            n_estimators,
            subsample,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            random_state,
            device_,
            device: dev,
            classes_,
            init_pred_,
            trees_,
            feature_importances_,
            n_features_in_,
        })
    }
}

// ---------------------------------------------------------------------------
// Classification Tree (Weighted Gini for AdaBoostClassifier)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum ClfTreeNode {
    Leaf {
        class_idx: usize,
        proba: Vec<f64>,
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<ClfTreeNode>,
        right: Box<ClfTreeNode>,
    },
}

impl ClfTreeNode {
    #[inline]
    fn predict_leaf(&self, row: &[f64]) -> (usize, &[f64]) {
        let mut node = self;
        loop {
            match node {
                ClfTreeNode::Leaf { class_idx, proba } => return (*class_idx, proba.as_slice()),
                ClfTreeNode::Internal {
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

fn write_clf_tree(buf: &mut Vec<u8>, node: &ClfTreeNode) {
    match node {
        ClfTreeNode::Leaf { class_idx, proba } => {
            write_u8(buf, 0);
            write_usize(buf, *class_idx);
            write_f64_vec(buf, proba);
        }
        ClfTreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            write_u8(buf, 1);
            write_usize(buf, *feature_index);
            write_f64(buf, *threshold);
            write_clf_tree(buf, left);
            write_clf_tree(buf, right);
        }
    }
}

fn read_clf_tree(reader: &mut ByteReader<'_>) -> OchreResult<ClfTreeNode> {
    let tag = reader.read_u8()?;
    match tag {
        0 => {
            let class_idx = reader.read_usize()?;
            let proba = reader.read_f64_vec()?;
            Ok(ClfTreeNode::Leaf { class_idx, proba })
        }
        1 => {
            let feature_index = reader.read_usize()?;
            let threshold = reader.read_f64()?;
            let left = read_clf_tree(reader)?;
            let right = read_clf_tree(reader)?;
            Ok(ClfTreeNode::Internal {
                feature_index,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        _ => Err(OchreError::InvalidTreeNodeTag),
    }
}

fn weighted_gini(class_w: &[f64], total_w: f64) -> f64 {
    if total_w <= 1e-15 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    let inv = 1.0 / total_w;
    for &w in class_w {
        let p = w * inv;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

fn make_clf_leaf(class_w: &[f64], total_w: f64) -> ClfTreeNode {
    let mut best_k = 0;
    let mut best_w = -1.0;
    let mut proba = vec![0.0; class_w.len()];
    let inv = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };
    for (k, &w) in class_w.iter().enumerate() {
        proba[k] = w * inv;
        if w > best_w {
            best_w = w;
            best_k = k;
        }
    }
    ClfTreeNode::Leaf {
        class_idx: best_k,
        proba,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_weighted_clf_tree(
    x: &Matrix,
    class_indices: &[usize],
    weights: &[f64],
    indices: &[usize],
    n_classes: usize,
    depth: usize,
    max_depth: usize,
    importances: &mut [f64],
) -> ClfTreeNode {
    let n = indices.len();
    let mut total_cw = vec![0.0; n_classes];
    let mut total_w = 0.0;
    for &i in indices {
        let w = weights[i];
        total_cw[class_indices[i]] += w;
        total_w += w;
    }
    let parent_gini = weighted_gini(&total_cw, total_w);
    if depth >= max_depth || n < 2 || parent_gini <= 1e-12 || total_w <= 1e-15 {
        return make_clf_leaf(&total_cw, total_w);
    }

    let best = (0..x.cols)
        .into_par_iter()
        .filter_map(|f| {
            let mut pairs: Vec<(f64, usize, f64)> = Vec::with_capacity(n);
            for &i in indices {
                pairs.push((x.data[i * x.cols + f], class_indices[i], weights[i]));
            }
            pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            if (pairs[n - 1].0 - pairs[0].0).abs() <= 1e-12 {
                return None;
            }

            let mut left_cw = vec![0.0; n_classes];
            let mut right_cw = total_cw.clone();
            let mut left_w = 0.0;
            let mut right_w = total_w;

            let mut best_gain = 1e-12;
            let mut best_thresh = 0.0;
            let mut found = false;

            for i in 0..(n - 1) {
                let c = pairs[i].1;
                let w = pairs[i].2;
                left_cw[c] += w;
                right_cw[c] = (right_cw[c] - w).max(0.0);
                left_w += w;
                right_w = (right_w - w).max(0.0);

                let cur_x = pairs[i].0;
                let next_x = pairs[i + 1].0;
                if (next_x - cur_x).abs() <= 1e-12 {
                    continue;
                }
                if left_w <= 1e-15 || right_w <= 1e-15 {
                    continue;
                }

                let g_l = weighted_gini(&left_cw, left_w);
                let g_r = weighted_gini(&right_cw, right_w);
                let gain = total_w * parent_gini - left_w * g_l - right_w * g_r;
                if gain > best_gain {
                    best_gain = gain;
                    best_thresh = 0.5 * (cur_x + next_x);
                    found = true;
                }
            }

            if found {
                Some((best_gain, f, best_thresh))
            } else {
                None
            }
        })
        .reduce_with(|a, b| if a.0 >= b.0 { a } else { b });

    let Some((best_gain, best_f, best_thresh)) = best else {
        return make_clf_leaf(&total_cw, total_w);
    };

    let mut left_idx = Vec::with_capacity(n / 2 + 1);
    let mut right_idx = Vec::with_capacity(n / 2 + 1);
    for &i in indices {
        if x.data[i * x.cols + best_f] <= best_thresh {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    if left_idx.is_empty() || right_idx.is_empty() {
        return make_clf_leaf(&total_cw, total_w);
    }

    importances[best_f] += best_gain;

    let left = build_weighted_clf_tree(
        x,
        class_indices,
        weights,
        &left_idx,
        n_classes,
        depth + 1,
        max_depth,
        importances,
    );
    let right = build_weighted_clf_tree(
        x,
        class_indices,
        weights,
        &right_idx,
        n_classes,
        depth + 1,
        max_depth,
        importances,
    );

    ClfTreeNode::Internal {
        feature_index: best_f,
        threshold: best_thresh,
        left: Box::new(left),
        right: Box::new(right),
    }
}

// ---------------------------------------------------------------------------
// 3. AdaBoostClassifier (SAMME)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdaBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    device: DeviceType,
    pub classes_: Option<Vec<f64>>,
    trees_: Vec<ClfTreeNode>,
    pub estimator_weights_: Vec<f64>,
    pub estimator_errors_: Vec<f64>,
    pub feature_importances_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl AdaBoostClassifier {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_estimators == 0 {
            return Err(OchreError::InvalidParameter(
                "n_estimators must be >= 1 ( >_< )".to_string(),
            ));
        }
        validate_positive_f64(learning_rate, "learning_rate")?;
        if max_depth < 1 {
            return Err(OchreError::InvalidParameter(format!(
                "max_depth ({max_depth}) must be >= 1 ( >_< )"
            )));
        }
        let resolved = device.or_else(|| Some("auto".to_string()));
        let dev = parse_device(resolved.as_deref())?;
        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            device_: resolved,
            device: dev,
            classes_: None,
            trees_: Vec::new(),
            estimator_weights_: Vec::new(),
            estimator_errors_: Vec::new(),
            feature_importances_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.or_else(|| Some("auto".to_string()));
        self.device = parse_device(resolved.as_deref())?;
        self.device_ = resolved;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let mut weights = validate_and_prepare_weights(sample_weight, x.rows)?;
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }

        let classes = extract_sorted_classes(y);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }
        let class_indices = map_labels_to_indices(y, &classes)?;
        let n_samples = x.rows;
        let n_features = x.cols;
        let k_classes = classes.len();
        let all_indices: Vec<usize> = (0..n_samples).collect();

        let mut trees = Vec::new();
        let mut est_weights = Vec::new();
        let mut est_errors = Vec::new();
        let mut total_importances = vec![0.0; n_features];

        for m in 0..self.n_estimators {
            let mut tree_imp = vec![0.0; n_features];
            let tree = build_weighted_clf_tree(
                x,
                &class_indices,
                &weights,
                &all_indices,
                k_classes,
                0,
                self.max_depth,
                &mut tree_imp,
            );
            normalize_importances(&mut tree_imp);

            let mut err = 0.0;
            let mut incorrect = vec![false; n_samples];
            for (i, row) in x.data.chunks_exact(n_features).enumerate() {
                let (pred_k, _) = tree.predict_leaf(row);
                if pred_k != class_indices[i] {
                    incorrect[i] = true;
                    err += weights[i];
                }
            }

            if err <= 0.0 {
                let w_est = 1.0;
                for f in 0..n_features {
                    total_importances[f] += w_est * tree_imp[f];
                }
                trees.push(tree);
                est_weights.push(w_est);
                est_errors.push(0.0);
                break;
            }

            let max_err = 1.0 - 1.0 / (k_classes as f64);
            if err >= max_err {
                if m == 0 {
                    trees.push(tree);
                    est_weights.push(0.0);
                    est_errors.push(err);
                }
                break;
            }

            let alpha_m =
                self.learning_rate * (((1.0 - err) / err).ln() + ((k_classes as f64) - 1.0).ln());

            for f in 0..n_features {
                total_importances[f] += alpha_m * tree_imp[f];
            }
            trees.push(tree);
            est_weights.push(alpha_m);
            est_errors.push(err);

            let mut new_sum = 0.0;
            for i in 0..n_samples {
                if incorrect[i] {
                    weights[i] *= alpha_m.exp();
                }
                new_sum += weights[i];
            }
            if new_sum <= 0.0 || !new_sum.is_finite() {
                break;
            }
            for w in &mut weights {
                *w /= new_sum;
            }
        }

        normalize_importances(&mut total_importances);
        self.classes_ = Some(classes);
        self.trees_ = trees;
        self.estimator_weights_ = est_weights;
        self.estimator_errors_ = est_errors;
        self.feature_importances_ = Some(total_importances);
        self.n_features_in_ = Some(n_features);
        Ok(())
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Matrix> {
        check_predict_input(x, self.n_features_in_)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_classes = classes.len();
        let n_samples = x.rows;
        let cols = x.cols;
        let norm: f64 = self.estimator_weights_.iter().sum();

        let mut scores = vec![0.0; n_samples * k_classes];
        for (i, row) in x.data.chunks_exact(cols).enumerate() {
            let dst = &mut scores[i * k_classes..(i + 1) * k_classes];
            for (tree, &w) in self.trees_.iter().zip(self.estimator_weights_.iter()) {
                let (pred_k, _) = tree.predict_leaf(row);
                dst[pred_k] += w;
            }
            if norm > 0.0 {
                for v in dst.iter_mut() {
                    *v /= norm;
                }
            }
        }

        if k_classes == 2 {
            let mut bin = vec![0.0; n_samples];
            for i in 0..n_samples {
                bin[i] = scores[i * 2 + 1] - scores[i * 2];
            }
            return Matrix::from_vec(n_samples, 1, bin);
        }
        Matrix::from_vec(n_samples, k_classes, scores)
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        check_predict_input(x, self.n_features_in_)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_classes = classes.len();
        let n_samples = x.rows;
        let cols = x.cols;
        let norm: f64 = self.estimator_weights_.iter().sum();

        let mut raw_scores = vec![0.0; n_samples * k_classes];
        for (i, row) in x.data.chunks_exact(cols).enumerate() {
            let dst = &mut raw_scores[i * k_classes..(i + 1) * k_classes];
            for (tree, &w) in self.trees_.iter().zip(self.estimator_weights_.iter()) {
                let (pred_k, _) = tree.predict_leaf(row);
                dst[pred_k] += w;
            }
            if norm > 0.0 {
                let scale = 1.0 / (norm * (k_classes as f64 - 1.0).max(1.0));
                for v in dst.iter_mut() {
                    *v *= scale;
                }
            }
        }
        let mat = Matrix::from_vec(n_samples, k_classes, raw_scores)?;
        let backend = get_backend(&self.device);
        backend.row_softmax(&mat)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let df = self.decision_function(x)?;
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let k_classes = classes.len();
        let n_samples = df.rows;
        let mut preds = Vec::with_capacity(n_samples);
        if k_classes == 2 {
            for i in 0..n_samples {
                preds.push(if df.data[i] >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                });
            }
            return Ok(preds);
        }
        for i in 0..n_samples {
            let row = &df.data[i * k_classes..(i + 1) * k_classes];
            let mut best_k = 0;
            let mut best_v = row[0];
            for (k, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best_k = k;
                }
            }
            preds.push(classes[best_k]);
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-12)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ADABOOST_CLASSIFIER);
        write_usize(&mut buf, self.n_estimators);
        write_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.max_depth);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.classes_);
        write_usize(&mut buf, self.trees_.len());
        for t in &self.trees_ {
            write_clf_tree(&mut buf, t);
        }
        write_f64_vec(&mut buf, &self.estimator_weights_);
        write_f64_vec(&mut buf, &self.estimator_errors_);
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ADABOOST_CLASSIFIER)?;
        let n_estimators = r.read_usize()?;
        let learning_rate = r.read_f64()?;
        let max_depth = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_opt_f64_vec()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(read_clf_tree(&mut r)?);
        }
        let estimator_weights_ = r.read_f64_vec()?;
        let estimator_errors_ = r.read_f64_vec()?;
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            device_,
            device: dev,
            classes_,
            trees_,
            estimator_weights_,
            estimator_errors_,
            feature_importances_,
            n_features_in_,
        })
    }
}

// ---------------------------------------------------------------------------
// 4. AdaBoostRegressor (AdaBoost.R2)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdaBoostRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub loss: String,
    pub max_depth: usize,
    pub random_state: Option<u64>,
    pub device_: Option<String>,
    device: DeviceType,
    trees_: Vec<RegTreeNode>,
    pub estimator_weights_: Vec<f64>,
    pub estimator_errors_: Vec<f64>,
    pub feature_importances_: Option<Vec<f64>>,
    pub n_features_in_: Option<usize>,
}

impl AdaBoostRegressor {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        loss: String,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_estimators == 0 {
            return Err(OchreError::InvalidParameter(
                "n_estimators must be >= 1 ( >_< )".to_string(),
            ));
        }
        validate_positive_f64(learning_rate, "learning_rate")?;
        let loss_lc = loss.to_lowercase();
        if loss_lc != "linear" && loss_lc != "square" && loss_lc != "exponential" {
            return Err(OchreError::InvalidParameter(format!(
                "loss '{loss}' must be 'linear', 'square', or 'exponential' (・`ω´・)"
            )));
        }
        if max_depth < 1 {
            return Err(OchreError::InvalidParameter(format!(
                "max_depth ({max_depth}) must be >= 1 ( >_< )"
            )));
        }
        let resolved = device.or_else(|| Some("auto".to_string()));
        let dev = parse_device(resolved.as_deref())?;
        Ok(Self {
            n_estimators,
            learning_rate,
            loss: loss_lc,
            max_depth,
            random_state,
            device_: resolved,
            device: dev,
            trees_: Vec::new(),
            estimator_weights_: Vec::new(),
            estimator_errors_: Vec::new(),
            feature_importances_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let resolved = device.or_else(|| Some("auto".to_string()));
        self.device = parse_device(resolved.as_deref())?;
        self.device_ = resolved;
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_matrix_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let mut weights = validate_and_prepare_weights(sample_weight, x.rows)?;
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }

        let n_samples = x.rows;
        let n_features = x.cols;
        let unit_weights = vec![1.0; n_samples];
        let mut rng = Rng64::new(self.random_state);
        let cfg = RegTreeConfig {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut trees = Vec::new();
        let mut est_weights = Vec::new();
        let mut est_errors = Vec::new();
        let mut total_importances = vec![0.0; n_features];

        for m in 0..self.n_estimators {
            // Weighted bootstrap sample via CDF binary search
            let mut cdf = Vec::with_capacity(n_samples);
            let mut acc = 0.0;
            for &w in &weights {
                acc += w;
                cdf.push(acc);
            }
            if let Some(last) = cdf.last_mut() {
                *last = 1.0;
            }
            let mut boot_idx = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                let u = rng.next_f64();
                let idx = cdf.partition_point(|&c| c < u).min(n_samples - 1);
                boot_idx.push(idx);
            }

            let mut tree_imp = vec![0.0; n_features];
            let tree = build_reg_tree_cpu(
                x,
                y,
                &unit_weights,
                &boot_idx,
                0,
                &cfg,
                LeafLossRule::MeanTarget,
                &mut tree_imp,
            );
            normalize_importances(&mut tree_imp);

            let y_pred = tree.predict_matrix(x);
            let mut err_vec = vec![0.0; n_samples];
            let mut err_max = 0.0_f64;
            for i in 0..n_samples {
                let e = (y[i] - y_pred[i]).abs();
                err_vec[i] = e;
                if e > err_max {
                    err_max = e;
                }
            }

            if err_max <= 0.0 {
                if m == 0 {
                    for f in 0..n_features {
                        total_importances[f] += tree_imp[f];
                    }
                    trees.push(tree);
                    est_weights.push(1.0);
                    est_errors.push(0.0);
                }
                break;
            }

            for e in &mut err_vec {
                *e /= err_max;
                if self.loss == "square" {
                    *e = (*e) * (*e);
                } else if self.loss == "exponential" {
                    *e = 1.0 - (-*e).exp();
                }
            }

            let mut est_err = 0.0;
            for i in 0..n_samples {
                est_err += weights[i] * err_vec[i];
            }

            if est_err >= 0.5 {
                if m == 0 {
                    for f in 0..n_features {
                        total_importances[f] += tree_imp[f];
                    }
                    trees.push(tree);
                    est_weights.push(1.0);
                    est_errors.push(est_err);
                }
                break;
            }

            let beta = est_err / (1.0 - est_err);
            let est_w = if beta <= 1e-15 {
                1.0
            } else {
                self.learning_rate * (1.0 / beta).ln()
            };

            for f in 0..n_features {
                total_importances[f] += est_w * tree_imp[f];
            }
            trees.push(tree);
            est_weights.push(est_w);
            est_errors.push(est_err);

            if beta > 0.0 {
                let mut new_sum = 0.0;
                for i in 0..n_samples {
                    weights[i] *= beta.powf((1.0 - err_vec[i]) * self.learning_rate);
                    new_sum += weights[i];
                }
                if new_sum <= 0.0 || !new_sum.is_finite() {
                    break;
                }
                for w in &mut weights {
                    *w /= new_sum;
                }
            } else {
                break;
            }
        }

        normalize_importances(&mut total_importances);
        self.trees_ = trees;
        self.estimator_weights_ = est_weights;
        self.estimator_errors_ = est_errors;
        self.feature_importances_ = Some(total_importances);
        self.n_features_in_ = Some(n_features);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        check_predict_input(x, self.n_features_in_)?;
        let n_samples = x.rows;
        let cols = x.cols;
        let total_w: f64 = self.estimator_weights_.iter().sum();
        let half_w = 0.5 * total_w;

        let predict_row = |row: &[f64]| -> f64 {
            let mut pairs: Vec<(f64, f64)> = self
                .trees_
                .iter()
                .zip(self.estimator_weights_.iter())
                .map(|(t, &w)| (t.predict_row(row), w))
                .collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut cum = 0.0;
            for (val, w) in &pairs {
                cum += *w;
                if cum >= half_w {
                    return *val;
                }
            }
            pairs.last().map_or(0.0, |(v, _)| *v)
        };

        let preds: Vec<f64> = if n_samples >= 512 {
            x.data.par_chunks_exact(cols).map(predict_row).collect()
        } else {
            x.data.chunks_exact(cols).map(predict_row).collect()
        };
        Ok(preds)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ADABOOST_REGRESSOR);
        write_usize(&mut buf, self.n_estimators);
        write_f64(&mut buf, self.learning_rate);
        write_str(&mut buf, &self.loss);
        write_usize(&mut buf, self.max_depth);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_usize(&mut buf, self.trees_.len());
        for t in &self.trees_ {
            write_reg_tree(&mut buf, t);
        }
        write_f64_vec(&mut buf, &self.estimator_weights_);
        write_f64_vec(&mut buf, &self.estimator_errors_);
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ADABOOST_REGRESSOR)?;
        let n_estimators = r.read_usize()?;
        let learning_rate = r.read_f64()?;
        let loss = r.read_str()?;
        let max_depth = r.read_usize()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let n_trees = r.read_usize()?;
        let mut trees_ = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees_.push(read_reg_tree(&mut r)?);
        }
        let estimator_weights_ = r.read_f64_vec()?;
        let estimator_errors_ = r.read_f64_vec()?;
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_estimators,
            learning_rate,
            loss,
            max_depth,
            random_state,
            device_,
            device: dev,
            trees_,
            estimator_weights_,
            estimator_errors_,
            feature_importances_,
            n_features_in_,
        })
    }
}

// ---------------------------------------------------------------------------
// PyO3 Bindings
// ---------------------------------------------------------------------------

#[pyclass(name = "GradientBoostingRegressor")]
pub struct PyGradientBoostingRegressor {
    inner: GradientBoostingRegressor,
}

crate::pymethods_with_dit!(PyGradientBoostingRegressor, GradientBoostingRegressor, {
    #[new]
    #[pyo3(signature = (
        loss = "squared_error".to_string(),
        learning_rate = 0.1,
        n_estimators = 100,
        subsample = 1.0,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_depth = 3,
        alpha = 0.9,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        learning_rate: f64,
        n_estimators: usize,
        subsample: f64,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_depth: usize,
        alpha: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GradientBoostingRegressor::new(
                loss,
                learning_rate,
                n_estimators,
                subsample,
                min_samples_split,
                min_samples_leaf,
                max_depth,
                alpha,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|sw| buffer_to_f64_1d(py, sw))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn staged_predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let staged = self.inner.staged_predict(&x_mat)?;
        matrix_to_pybytes(py, &staged)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
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
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators
    }
    #[getter]
    pub fn subsample(&self) -> f64 {
        self.inner.subsample
    }
    #[getter]
    pub fn min_samples_split(&self) -> usize {
        self.inner.min_samples_split
    }
    #[getter]
    pub fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }
    #[getter]
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn feature_importances_(&self) -> Option<Vec<f64>> {
        self.inner.feature_importances_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "GradientBoostingClassifier")]
pub struct PyGradientBoostingClassifier {
    inner: GradientBoostingClassifier,
}

crate::pymethods_with_dit!(PyGradientBoostingClassifier, GradientBoostingClassifier, {
    #[new]
    #[pyo3(signature = (
        loss = "log_loss".to_string(),
        learning_rate = 0.1,
        n_estimators = 100,
        subsample = 1.0,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_depth = 3,
        random_state = None,
        device = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        learning_rate: f64,
        n_estimators: usize,
        subsample: f64,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GradientBoostingClassifier::new(
                loss,
                learning_rate,
                n_estimators,
                subsample,
                min_samples_split,
                min_samples_leaf,
                max_depth,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|sw| buffer_to_f64_1d(py, sw))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &probs)
    }

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let df = self.inner.decision_function(&x_mat)?;
        matrix_to_pybytes(py, &df)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
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
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators
    }
    #[getter]
    pub fn subsample(&self) -> f64 {
        self.inner.subsample
    }
    #[getter]
    pub fn min_samples_split(&self) -> usize {
        self.inner.min_samples_split
    }
    #[getter]
    pub fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }
    #[getter]
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }
    #[getter]
    pub fn feature_importances_(&self) -> Option<Vec<f64>> {
        self.inner.feature_importances_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "AdaBoostClassifier")]
pub struct PyAdaBoostClassifier {
    inner: AdaBoostClassifier,
}

crate::pymethods_with_dit!(PyAdaBoostClassifier, AdaBoostClassifier, {
    #[new]
    #[pyo3(signature = (
        n_estimators = 50,
        learning_rate = 1.0,
        max_depth = 1,
        random_state = None,
        device = None
    ))]
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaBoostClassifier::new(
                n_estimators,
                learning_rate,
                max_depth,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|sw| buffer_to_f64_1d(py, sw))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn predict_proba_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let probs = self.inner.predict_proba(&x_mat)?;
        matrix_to_pybytes(py, &probs)
    }

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let df = self.inner.decision_function(&x_mat)?;
        matrix_to_pybytes(py, &df)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators
    }
    #[getter]
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }
    #[getter]
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn classes_(&self) -> Option<Vec<f64>> {
        self.inner.classes_.clone()
    }
    #[getter]
    pub fn estimator_weights_(&self) -> Vec<f64> {
        self.inner.estimator_weights_.clone()
    }
    #[getter]
    pub fn estimator_errors_(&self) -> Vec<f64> {
        self.inner.estimator_errors_.clone()
    }
    #[getter]
    pub fn feature_importances_(&self) -> Option<Vec<f64>> {
        self.inner.feature_importances_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

#[pyclass(name = "AdaBoostRegressor")]
pub struct PyAdaBoostRegressor {
    inner: AdaBoostRegressor,
}

crate::pymethods_with_dit!(PyAdaBoostRegressor, AdaBoostRegressor, {
    #[new]
    #[pyo3(signature = (
        n_estimators = 50,
        learning_rate = 1.0,
        loss = "linear".to_string(),
        max_depth = 3,
        random_state = None,
        device = None
    ))]
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        loss: String,
        max_depth: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaBoostRegressor::new(
                n_estimators,
                learning_rate,
                loss,
                max_depth,
                random_state,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
        sample_weight: Option<PyBuffer<f64>>,
    ) -> PyResult<()> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        let sw_vec = sample_weight
            .as_ref()
            .map(|sw| buffer_to_f64_1d(py, sw))
            .transpose()?;
        self.inner.fit(&x_mat, &y_vec, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let preds = self.inner.predict(&x_mat)?;
        f64_slice_to_pybytes(py, &preds)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: PyBuffer<f64>,
        y: PyBuffer<f64>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_pooled_matrix(py, &x)?;
        let y_vec = buffer_to_f64_1d(py, &y)?;
        Ok(self.inner.score(&x_mat, &y_vec)?)
    }

    #[getter]
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators
    }
    #[getter]
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }
    #[getter]
    pub fn loss(&self) -> String {
        self.inner.loss.clone()
    }
    #[getter]
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn estimator_weights_(&self) -> Vec<f64> {
        self.inner.estimator_weights_.clone()
    }
    #[getter]
    pub fn estimator_errors_(&self) -> Vec<f64> {
        self.inner.estimator_errors_.clone()
    }
    #[getter]
    pub fn feature_importances_(&self) -> Option<Vec<f64>> {
        self.inner.feature_importances_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGradientBoostingClassifier>()?;
    m.add_class::<PyGradientBoostingRegressor>()?;
    m.add_class::<PyAdaBoostClassifier>()?;
    m.add_class::<PyAdaBoostRegressor>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Rust Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gb_regressor_losses_and_dit() {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let x0 = i as f64 / 10.0;
            let x1 = ((i * 7) % 10) as f64 / 10.0;
            rows.push(vec![x0, x1]);
            y.push(2.0 * x0 + 0.5 * x1);
        }
        let x = Matrix::from_2d(&rows).unwrap();
        for loss in ["squared_error", "absolute_error", "huber"] {
            let mut gbr = GradientBoostingRegressor::new(
                loss.to_string(),
                0.1,
                40,
                1.0,
                2,
                1,
                3,
                0.9,
                Some(42),
                Some("cpu".to_string()),
            )
            .unwrap();
            gbr.fit(&x, &y, None).unwrap();
            let r2 = gbr.score(&x, &y).unwrap();
            assert!(r2 > 0.9, "loss={loss} r2={r2}");

            let staged = gbr.staged_predict(&x).unwrap();
            assert_eq!(staged.rows, 40);
            assert_eq!(staged.cols, 40);

            let bytes = gbr.to_bytes();
            let loaded = GradientBoostingRegressor::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(gbr.predict(&x).unwrap(), loaded.predict(&x).unwrap());
        }
    }

    #[test]
    fn test_gb_classifier_binary_and_multiclass() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.1],
            vec![0.1, 0.2],
            vec![0.2, 0.0],
            vec![1.0, 1.1],
            vec![1.1, 1.2],
            vec![1.2, 1.0],
        ])
        .unwrap();
        let y_bin = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut gbc = GradientBoostingClassifier::new(
            "log_loss".to_string(),
            0.2,
            25,
            1.0,
            2,
            1,
            2,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        gbc.fit(&x, &y_bin, None).unwrap();
        assert!((gbc.score(&x, &y_bin).unwrap() - 1.0).abs() < 1e-12);

        let x_mc = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![2.0, 0.0],
            vec![2.1, 0.1],
            vec![0.0, 2.0],
            vec![0.1, 2.1],
        ])
        .unwrap();
        let y_mc = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let mut gbc_mc = GradientBoostingClassifier::new(
            "log_loss".to_string(),
            0.2,
            25,
            1.0,
            2,
            1,
            2,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        gbc_mc.fit(&x_mc, &y_mc, None).unwrap();
        assert!((gbc_mc.score(&x_mc, &y_mc).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_adaboost_classifier_and_regressor() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.0],
            vec![1.2, 1.1],
            vec![2.0, 0.0],
            vec![2.1, 0.2],
        ])
        .unwrap();
        let y_clf = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let mut abc =
            AdaBoostClassifier::new(20, 1.0, 1, Some(42), Some("cpu".to_string())).unwrap();
        abc.fit(&x, &y_clf, None).unwrap();
        assert!(abc.score(&x, &y_clf).unwrap() >= 0.8);

        let y_reg = vec![0.0, 0.3, 2.0, 2.3, 2.0, 2.3];
        let mut abr = AdaBoostRegressor::new(
            20,
            1.0,
            "linear".to_string(),
            3,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        abr.fit(&x, &y_reg, None).unwrap();
        assert!(abr.score(&x, &y_reg).unwrap() > 0.8);
    }
}
