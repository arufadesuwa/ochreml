// OchreML Forest & Isolation Ensembles Module (＾▽＾)
// Implements RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
// ExtraTreesRegressor, and IsolationForest with N6 accelerator-first histogram splits
// on CUDA/wgpu and Rayon-parallel tree building on CPU/TPU (*≧ω≦*).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_str, write_opt_u64, write_opt_usize, write_str,
    write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, r2_score};
use crate::preprocessing::py_helpers::{
    buffer_to_f64_1d, buffer_to_pooled_matrix, f64_slice_to_pybytes, i64_slice_to_pybytes,
    matrix_to_pybytes,
};
use crate::primitives::{BinnedMatrix, GradHessBin, subtract_histogram};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_RANDOM_FOREST_CLASSIFIER: u8 = 67;
pub const TAG_RANDOM_FOREST_REGRESSOR: u8 = 68;
pub const TAG_EXTRA_TREES_CLASSIFIER: u8 = 69;
pub const TAG_EXTRA_TREES_REGRESSOR: u8 = 70;
pub const TAG_ISOLATION_FOREST: u8 = 75;

// =========================================================================
// SHARED PRNG, VALIDATION, AND TREE REPRESENTATION
// =========================================================================

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
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
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Uniform in [0.0, 1.0)
        const INV_2_53: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * INV_2_53
    }
}

fn resolve_base_seed(random_state: Option<u64>) -> u64 {
    match random_state {
        Some(s) => s,
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5EED_CAFE_BABE_1234),
    }
}

fn validate_x_finite(x: &Matrix) -> OchreResult<()> {
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

fn validate_y_finite(y: &[f64], n_samples: usize) -> OchreResult<()> {
    if y.len() != n_samples {
        return Err(OchreError::LengthMismatch {
            features: n_samples,
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

fn validate_sample_weight(
    sample_weight: Option<&[f64]>,
    n_samples: usize,
) -> OchreResult<Vec<f64>> {
    match sample_weight {
        None => Ok(vec![1.0; n_samples]),
        Some(sw) => {
            if sw.len() != n_samples {
                return Err(OchreError::WeightLengthMismatch {
                    weights: sw.len(),
                    samples: n_samples,
                });
            }
            let mut sum = 0.0;
            for &w in sw {
                if !w.is_finite() || w < 0.0 {
                    return Err(OchreError::InvalidWeights);
                }
                sum += w;
            }
            if sum <= 0.0 {
                return Err(OchreError::NonPositiveWeightSum);
            }
            Ok(sw.to_vec())
        }
    }
}

fn validate_max_features_spec(spec: Option<&str>) -> OchreResult<Option<String>> {
    match spec {
        None => Ok(None),
        Some(raw) => {
            let s = raw.trim().to_lowercase();
            if s == "sqrt" || s == "log2" || s == "all" {
                return Ok(Some(s));
            }
            if s.contains('.') || s.contains('e') {
                let frac: f64 = s.parse().map_err(|_| {
                    OchreError::InvalidParameter(format!("Invalid max_features='{}' ( >_< )", raw))
                })?;
                if !frac.is_finite() || frac <= 0.0 || frac > 1.0 {
                    return Err(OchreError::InvalidParameter(format!(
                        "Float max_features={} must be in (0.0, 1.0] ( >_< )",
                        frac
                    )));
                }
                return Ok(Some(s));
            }
            let cnt: usize = s.parse().map_err(|_| {
                OchreError::InvalidParameter(format!("Invalid max_features='{}' ( >_< )", raw))
            })?;
            if cnt == 0 {
                return Err(OchreError::InvalidParameter(
                    "Integer max_features must be >= 1 ( >_< )".to_string(),
                ));
            }
            Ok(Some(s))
        }
    }
}

fn resolve_max_features(
    spec: Option<&str>,
    n_features: usize,
    default_sqrt: bool,
) -> OchreResult<usize> {
    match spec {
        None => {
            if default_sqrt {
                Ok(((n_features as f64).sqrt().floor() as usize).max(1))
            } else {
                Ok(n_features)
            }
        }
        Some(s) => {
            let lower = s.trim().to_lowercase();
            if lower == "sqrt" {
                Ok(((n_features as f64).sqrt().floor() as usize).max(1))
            } else if lower == "log2" {
                Ok(((n_features as f64).log2().floor() as usize).max(1))
            } else if lower == "all" {
                Ok(n_features)
            } else if lower.contains('.') || lower.contains('e') {
                let frac: f64 = lower.parse().map_err(|_| {
                    OchreError::InvalidParameter(format!("Invalid max_features='{}' ( >_< )", s))
                })?;
                if !frac.is_finite() || frac <= 0.0 || frac > 1.0 {
                    return Err(OchreError::InvalidParameter(format!(
                        "Float max_features={} must be in (0.0, 1.0] ( >_< )",
                        frac
                    )));
                }
                Ok(((frac * n_features as f64).ceil() as usize).clamp(1, n_features))
            } else {
                let cnt: usize = lower.parse().map_err(|_| {
                    OchreError::InvalidParameter(format!("Invalid max_features='{}' ( >_< )", s))
                })?;
                if cnt == 0 || cnt > n_features {
                    return Err(OchreError::InvalidParameter(format!(
                        "max_features={} must be in 1..={} ( >_< )",
                        cnt, n_features
                    )));
                }
                Ok(cnt)
            }
        }
    }
}

fn sample_feature_subset(
    pool: &mut Vec<usize>,
    n_features: usize,
    max_feat: usize,
    rng: &mut SplitMix64,
) {
    pool.clear();
    pool.extend(0..n_features);
    let k = max_feat.min(n_features);
    for i in 0..k {
        let j = i + rng.next_usize(n_features - i);
        pool.swap(i, j);
    }
    pool.truncate(k);
}

#[derive(Debug, Clone)]
enum ClfTreeNode {
    Leaf {
        probs: Vec<f64>,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<ClfTreeNode>,
        right: Box<ClfTreeNode>,
    },
}

impl ClfTreeNode {
    fn predict_proba_row(&self, row: &[f64]) -> &[f64] {
        match self {
            ClfTreeNode::Leaf { probs } => probs,
            ClfTreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.predict_proba_row(row)
                } else {
                    right.predict_proba_row(row)
                }
            }
        }
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            ClfTreeNode::Leaf { probs } => {
                write_u8(buf, 0);
                write_f64_vec(buf, probs);
            }
            ClfTreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                write_u8(buf, 1);
                write_usize(buf, *feature);
                write_f64(buf, *threshold);
                left.write_to(buf);
                right.write_to(buf);
            }
        }
    }

    fn read_from(r: &mut ByteReader<'_>) -> OchreResult<Self> {
        match r.read_u8()? {
            0 => Ok(ClfTreeNode::Leaf {
                probs: r.read_f64_vec()?,
            }),
            1 => {
                let feature = r.read_usize()?;
                let threshold = r.read_f64()?;
                let left = Box::new(Self::read_from(r)?);
                let right = Box::new(Self::read_from(r)?);
                Ok(ClfTreeNode::Internal {
                    feature,
                    threshold,
                    left,
                    right,
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
}

#[derive(Debug, Clone)]
enum RegTreeNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<RegTreeNode>,
        right: Box<RegTreeNode>,
    },
}

impl RegTreeNode {
    fn predict_row(&self, row: &[f64]) -> f64 {
        match self {
            RegTreeNode::Leaf { value } => *value,
            RegTreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.predict_row(row)
                } else {
                    right.predict_row(row)
                }
            }
        }
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            RegTreeNode::Leaf { value } => {
                write_u8(buf, 0);
                write_f64(buf, *value);
            }
            RegTreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                write_u8(buf, 1);
                write_usize(buf, *feature);
                write_f64(buf, *threshold);
                left.write_to(buf);
                right.write_to(buf);
            }
        }
    }

    fn read_from(r: &mut ByteReader<'_>) -> OchreResult<Self> {
        match r.read_u8()? {
            0 => Ok(RegTreeNode::Leaf {
                value: r.read_f64()?,
            }),
            1 => {
                let feature = r.read_usize()?;
                let threshold = r.read_f64()?;
                let left = Box::new(Self::read_from(r)?);
                let right = Box::new(Self::read_from(r)?);
                Ok(RegTreeNode::Internal {
                    feature,
                    threshold,
                    left,
                    right,
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
}

// =========================================================================
// CLASSIFICATION TREE BUILDER (RF & EXTRA TREES)
// =========================================================================

#[derive(Clone, Copy)]
struct ClfBuildCfg<'a> {
    criterion_entropy: bool,
    extra_trees: bool,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    n_classes: usize,
    total_tree_weight: f64,
    x: &'a Matrix,
    class_idx: &'a [usize],
}

fn impurity_weighted(counts: &[f64], total_w: f64, entropy: bool) -> f64 {
    if total_w <= 0.0 {
        return 0.0;
    }
    if entropy {
        let mut ent = 0.0;
        for &c in counts {
            if c > 0.0 {
                let p = c / total_w;
                ent -= p * p.log2();
            }
        }
        ent
    } else {
        let mut sum_sq = 0.0;
        for &c in counts {
            let p = c / total_w;
            sum_sq += p * p;
        }
        1.0 - sum_sq
    }
}

fn build_clf_tree(
    cfg: &ClfBuildCfg<'_>,
    samples: &[(usize, f64)],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
) -> ClfTreeNode {
    let n = samples.len();
    let mut counts = vec![0.0; cfg.n_classes];
    let mut total_w = 0.0;
    for &(idx, w) in samples {
        counts[cfg.class_idx[idx]] += w;
        total_w += w;
    }

    let make_leaf = || {
        let probs = if total_w > 0.0 {
            counts.iter().map(|&c| c / total_w).collect()
        } else {
            vec![1.0 / cfg.n_classes.max(1) as f64; cfg.n_classes]
        };
        ClfTreeNode::Leaf { probs }
    };

    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| depth >= d)
    {
        return make_leaf();
    }

    let parent_imp = impurity_weighted(&counts, total_w, cfg.criterion_entropy);
    if parent_imp <= 1e-12 {
        return make_leaf();
    }

    let mut feat_pool = Vec::with_capacity(cfg.x.cols);
    sample_feature_subset(&mut feat_pool, cfg.x.cols, cfg.max_features, rng);

    let mut best_gain = 0.0;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;

    if cfg.extra_trees {
        let mut left_counts = vec![0.0; cfg.n_classes];
        let mut right_counts = vec![0.0; cfg.n_classes];
        for &f in &feat_pool {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &(idx, _) in samples {
                let v = cfg.x.data[idx * cfg.x.cols + f];
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            if max_v <= min_v + 1e-12 {
                continue;
            }
            let thresh = min_v + rng.next_f64() * (max_v - min_v);
            let thresh = if thresh >= max_v {
                0.5 * (min_v + max_v)
            } else {
                thresh
            };

            left_counts.fill(0.0);
            right_counts.fill(0.0);
            let mut left_w = 0.0;
            let mut left_n = 0usize;
            for &(idx, w) in samples {
                let v = cfg.x.data[idx * cfg.x.cols + f];
                let c = cfg.class_idx[idx];
                if v <= thresh {
                    left_counts[c] += w;
                    left_w += w;
                    left_n += 1;
                } else {
                    right_counts[c] += w;
                }
            }
            let right_n = n - left_n;
            if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                continue;
            }
            let right_w = total_w - left_w;
            if left_w <= 0.0 || right_w <= 0.0 {
                continue;
            }
            let left_imp = impurity_weighted(&left_counts, left_w, cfg.criterion_entropy);
            let right_imp = impurity_weighted(&right_counts, right_w, cfg.criterion_entropy);
            let gain = parent_imp - (left_w / total_w) * left_imp - (right_w / total_w) * right_imp;
            if gain > best_gain {
                best_gain = gain;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    } else {
        let mut sorted: Vec<(f64, usize, f64)> = Vec::with_capacity(n);
        let mut left_counts = vec![0.0; cfg.n_classes];
        let mut right_counts = vec![0.0; cfg.n_classes];

        for &f in &feat_pool {
            sorted.clear();
            for &(idx, w) in samples {
                sorted.push((cfg.x.data[idx * cfg.x.cols + f], cfg.class_idx[idx], w));
            }
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

            if sorted[n - 1].0 <= sorted[0].0 + 1e-12 {
                continue;
            }

            left_counts.fill(0.0);
            right_counts.copy_from_slice(&counts);
            let mut left_w = 0.0;

            for i in 0..(n - 1) {
                let (val_i, c_i, w_i) = sorted[i];
                left_counts[c_i] += w_i;
                right_counts[c_i] -= w_i;
                left_w += w_i;

                let left_n = i + 1;
                let right_n = n - left_n;
                if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                    continue;
                }
                let val_next = sorted[i + 1].0;
                if val_i + 1e-12 >= val_next {
                    continue;
                }

                let right_w = total_w - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let left_imp = impurity_weighted(&left_counts, left_w, cfg.criterion_entropy);
                let right_imp = impurity_weighted(&right_counts, right_w, cfg.criterion_entropy);
                let gain =
                    parent_imp - (left_w / total_w) * left_imp - (right_w / total_w) * right_imp;
                if gain > best_gain {
                    best_gain = gain;
                    best_feat = f;
                    best_thresh = 0.5 * (val_i + val_next);
                }
            }
        }
    }

    if best_gain <= 1e-12 {
        return make_leaf();
    }

    let mut left_samples = Vec::with_capacity(n / 2);
    let mut right_samples = Vec::with_capacity(n / 2);
    for &item in samples {
        if cfg.x.data[item.0 * cfg.x.cols + best_feat] <= best_thresh {
            left_samples.push(item);
        } else {
            right_samples.push(item);
        }
    }

    if left_samples.len() < cfg.min_samples_leaf || right_samples.len() < cfg.min_samples_leaf {
        return make_leaf();
    }

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += (total_w / cfg.total_tree_weight) * best_gain;
    }

    let left = Box::new(build_clf_tree(
        cfg,
        &left_samples,
        depth + 1,
        rng,
        importances,
    ));
    let right = Box::new(build_clf_tree(
        cfg,
        &right_samples,
        depth + 1,
        rng,
        importances,
    ));

    ClfTreeNode::Internal {
        feature: best_feat,
        threshold: best_thresh,
        left,
        right,
    }
}

// =========================================================================
// REGRESSION TREE BUILDER (RF & EXTRA TREES, WITH N6 ACCELERATOR PATH)
// =========================================================================

#[derive(Clone, Copy)]
struct RegBuildCfg<'a> {
    extra_trees: bool,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    total_tree_weight: f64,
    x: &'a Matrix,
    y: &'a [f64],
}

fn build_reg_tree(
    cfg: &RegBuildCfg<'_>,
    samples: &[(usize, f64)],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
) -> RegTreeNode {
    let n = samples.len();
    let mut sum_y = 0.0;
    let mut sum_y2 = 0.0;
    let mut total_w = 0.0;
    for &(idx, w) in samples {
        let yi = cfg.y[idx];
        sum_y += w * yi;
        sum_y2 += w * yi * yi;
        total_w += w;
    }

    let mean_val = if total_w > 0.0 { sum_y / total_w } else { 0.0 };
    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| depth >= d)
        || total_w <= 0.0
    {
        return RegTreeNode::Leaf { value: mean_val };
    }

    let parent_var = (sum_y2 / total_w) - mean_val * mean_val;
    if parent_var <= 1e-12 {
        return RegTreeNode::Leaf { value: mean_val };
    }
    let parent_score = (sum_y * sum_y) / total_w;

    let mut feat_pool = Vec::with_capacity(cfg.x.cols);
    sample_feature_subset(&mut feat_pool, cfg.x.cols, cfg.max_features, rng);

    let mut best_score_gain = 0.0;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;

    if cfg.extra_trees {
        for &f in &feat_pool {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &(idx, _) in samples {
                let v = cfg.x.data[idx * cfg.x.cols + f];
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            if max_v <= min_v + 1e-12 {
                continue;
            }
            let thresh = min_v + rng.next_f64() * (max_v - min_v);
            let thresh = if thresh >= max_v {
                0.5 * (min_v + max_v)
            } else {
                thresh
            };

            let mut left_sum = 0.0;
            let mut left_w = 0.0;
            let mut left_n = 0usize;
            for &(idx, w) in samples {
                if cfg.x.data[idx * cfg.x.cols + f] <= thresh {
                    left_sum += w * cfg.y[idx];
                    left_w += w;
                    left_n += 1;
                }
            }
            let right_n = n - left_n;
            if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                continue;
            }
            let right_w = total_w - left_w;
            if left_w <= 0.0 || right_w <= 0.0 {
                continue;
            }
            let right_sum = sum_y - left_sum;
            let gain =
                (left_sum * left_sum) / left_w + (right_sum * right_sum) / right_w - parent_score;
            if gain > best_score_gain {
                best_score_gain = gain;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    } else {
        let mut sorted: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        for &f in &feat_pool {
            sorted.clear();
            for &(idx, w) in samples {
                sorted.push((cfg.x.data[idx * cfg.x.cols + f], cfg.y[idx], w));
            }
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

            if sorted[n - 1].0 <= sorted[0].0 + 1e-12 {
                continue;
            }

            let mut left_sum = 0.0;
            let mut left_w = 0.0;
            for i in 0..(n - 1) {
                let (val_i, yi, wi) = sorted[i];
                left_sum += wi * yi;
                left_w += wi;

                let left_n = i + 1;
                let right_n = n - left_n;
                if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                    continue;
                }
                let val_next = sorted[i + 1].0;
                if val_i + 1e-12 >= val_next {
                    continue;
                }

                let right_w = total_w - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let right_sum = sum_y - left_sum;
                let gain = (left_sum * left_sum) / left_w + (right_sum * right_sum) / right_w
                    - parent_score;
                if gain > best_score_gain {
                    best_score_gain = gain;
                    best_feat = f;
                    best_thresh = 0.5 * (val_i + val_next);
                }
            }
        }
    }

    if best_score_gain <= 1e-12 {
        return RegTreeNode::Leaf { value: mean_val };
    }

    let mut left_samples = Vec::with_capacity(n / 2);
    let mut right_samples = Vec::with_capacity(n / 2);
    for &item in samples {
        if cfg.x.data[item.0 * cfg.x.cols + best_feat] <= best_thresh {
            left_samples.push(item);
        } else {
            right_samples.push(item);
        }
    }

    if left_samples.len() < cfg.min_samples_leaf || right_samples.len() < cfg.min_samples_leaf {
        return RegTreeNode::Leaf { value: mean_val };
    }

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += best_score_gain / cfg.total_tree_weight;
    }

    let left = Box::new(build_reg_tree(
        cfg,
        &left_samples,
        depth + 1,
        rng,
        importances,
    ));
    let right = Box::new(build_reg_tree(
        cfg,
        &right_samples,
        depth + 1,
        rng,
        importances,
    ));

    RegTreeNode::Internal {
        feature: best_feat,
        threshold: best_thresh,
        left,
        right,
    }
}

/// N6 Hardware-Accelerated Histogram Regression Tree Builder for CUDA / wgpu.
#[allow(clippy::too_many_arguments)]
fn build_reg_tree_n6(
    backend: &Backend<'_>,
    binned: &BinnedMatrix,
    grad: &[f64],
    hess: &[f64],
    sample_indices: &[u32],
    parent_hist: &[GradHessBin],
    depth: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    total_tree_weight: f64,
    rng: &mut SplitMix64,
    importances: &mut [f64],
) -> OchreResult<RegTreeNode> {
    let n = sample_indices.len();
    let mut sum_g = 0.0;
    let mut sum_h = 0.0;
    for &idx in sample_indices {
        let r = idx as usize;
        sum_g += grad[r];
        sum_h += hess[r];
    }
    let leaf_val = if sum_h > 0.0 { sum_g / sum_h } else { 0.0 };

    if n < min_samples_split
        || n < 2 * min_samples_leaf
        || max_depth.is_some_and(|d| depth >= d)
        || sum_h <= 0.0
    {
        return Ok(RegTreeNode::Leaf { value: leaf_val });
    }

    // Use N6 find_best_histogram_split when all features are considered, or feature-subsampled scan
    let best_split = if max_features >= binned.cols {
        backend.find_best_histogram_split(binned, parent_hist, 0.0, min_samples_leaf as u32)?
    } else {
        let mut feat_pool = Vec::with_capacity(binned.cols);
        sample_feature_subset(&mut feat_pool, binned.cols, max_features, rng);
        let num_bins = binned.num_bins;
        let parent_score = (sum_g * sum_g) / sum_h.max(1e-12);
        let mut best: Option<(usize, f64, f64)> = None;
        for &f in &feat_pool {
            let edges = &binned.bin_edges[f];
            if edges.is_empty() {
                continue;
            }
            let hist = &parent_hist[f * num_bins..(f + 1) * num_bins];
            let mut left_g = 0.0;
            let mut left_h = 0.0;
            let mut left_n = 0u32;
            for (b, &threshold) in edges.iter().enumerate() {
                left_g += hist[b].sum_grad;
                left_h += hist[b].sum_hess;
                left_n += hist[b].count;
                let right_n = (n as u32).saturating_sub(left_n);
                if left_n < min_samples_leaf as u32 || right_n < min_samples_leaf as u32 {
                    continue;
                }
                let right_g = sum_g - left_g;
                let right_h = sum_h - left_h;
                if left_h <= 0.0 || right_h <= 0.0 {
                    continue;
                }
                let gain = 0.5
                    * ((left_g * left_g) / left_h.max(1e-12)
                        + (right_g * right_g) / right_h.max(1e-12)
                        - parent_score);
                if best.is_none_or(|cur| gain > cur.2) {
                    best = Some((f, threshold, gain));
                }
            }
        }
        best.map(
            |(feature_idx, threshold, gain)| crate::primitives::HistogramSplit {
                feature_idx,
                bin_idx: 0,
                threshold,
                gain,
                left_grad: 0.0,
                left_hess: 0.0,
                left_count: 0,
            },
        )
    };

    let Some(split) = best_split else {
        return Ok(RegTreeNode::Leaf { value: leaf_val });
    };
    if split.gain <= 1e-12 {
        return Ok(RegTreeNode::Leaf { value: leaf_val });
    }

    let mut left_idx = Vec::with_capacity(n / 2);
    let mut right_idx = Vec::with_capacity(n / 2);
    let f = split.feature_idx;
    let bin_cutoff = binned.bin_edges[f]
        .iter()
        .position(|&e| (e - split.threshold).abs() <= 1e-15)
        .unwrap_or(split.bin_idx) as u8;

    for &idx in sample_indices {
        let r = idx as usize;
        if binned.data[r * binned.cols + f] <= bin_cutoff {
            left_idx.push(idx);
        } else {
            right_idx.push(idx);
        }
    }

    if left_idx.len() < min_samples_leaf || right_idx.len() < min_samples_leaf {
        return Ok(RegTreeNode::Leaf { value: leaf_val });
    }

    if total_tree_weight > 0.0 {
        importances[f] += (2.0 * split.gain) / total_tree_weight;
    }

    // Sibling histogram subtraction optimization (N6)
    let (left_hist, right_hist) = if left_idx.len() <= right_idx.len() {
        let lh = backend.batched_histogram_grad(binned, &left_idx, grad, hess)?;
        let rh = subtract_histogram(parent_hist, &lh)?;
        (lh, rh)
    } else {
        let rh = backend.batched_histogram_grad(binned, &right_idx, grad, hess)?;
        let lh = subtract_histogram(parent_hist, &rh)?;
        (lh, rh)
    };

    let left = Box::new(build_reg_tree_n6(
        backend,
        binned,
        grad,
        hess,
        &left_idx,
        &left_hist,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        total_tree_weight,
        rng,
        importances,
    )?);
    let right = Box::new(build_reg_tree_n6(
        backend,
        binned,
        grad,
        hess,
        &right_idx,
        &right_hist,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        total_tree_weight,
        rng,
        importances,
    )?);

    Ok(RegTreeNode::Internal {
        feature: f,
        threshold: split.threshold,
        left,
        right,
    })
}

fn build_bootstrap_samples(
    n_samples: usize,
    bootstrap: bool,
    sw: &[f64],
    rng: &mut SplitMix64,
) -> (Vec<(usize, f64)>, f64) {
    if !bootstrap {
        let samples: Vec<(usize, f64)> = (0..n_samples).map(|i| (i, sw[i])).collect();
        let total_w: f64 = sw.iter().sum();
        return (samples, total_w);
    }
    let mut counts = vec![0usize; n_samples];
    for _ in 0..n_samples {
        counts[rng.next_usize(n_samples)] += 1;
    }
    let mut samples = Vec::with_capacity(n_samples);
    let mut total_w = 0.0;
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 && sw[i] > 0.0 {
            let w = sw[i] * (c as f64);
            samples.push((i, w));
            total_w += w;
        }
    }
    if samples.is_empty() {
        for (i, &w) in sw.iter().enumerate() {
            if w > 0.0 {
                samples.push((i, w));
                total_w += w;
            }
        }
    }
    (samples, total_w)
}

fn normalize_and_average_importances(per_tree: Vec<Vec<f64>>, n_features: usize) -> Vec<f64> {
    if per_tree.is_empty() {
        return vec![0.0; n_features];
    }
    let mut avg = vec![0.0; n_features];
    let n_trees = per_tree.len() as f64;
    for imp in per_tree {
        let sum: f64 = imp.iter().sum();
        if sum > 0.0 {
            for (dst, &val) in avg.iter_mut().zip(imp.iter()) {
                *dst += val / sum;
            }
        }
    }
    for v in &mut avg {
        *v /= n_trees;
    }
    let total: f64 = avg.iter().sum();
    if total > 0.0 {
        for v in &mut avg {
            *v /= total;
        }
    }
    avg
}

// =========================================================================
// SHARED FOREST CLASSIFIER CORE (RF & EXTRA TREES)
// =========================================================================

#[derive(Debug, Clone)]
struct ForestClassifierCore {
    extra_trees: bool,
    n_estimators: usize,
    criterion: String,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    device: DeviceType,
    device_: Option<String>,
    trees: Vec<ClfTreeNode>,
    classes_: Vec<f64>,
    n_features_in_: Option<usize>,
    feature_importances_: Option<Vec<f64>>,
}

impl ForestClassifierCore {
    #[allow(clippy::too_many_arguments)]
    fn new(
        extra_trees: bool,
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let n_est = n_estimators.unwrap_or(100);
        if n_est == 0 {
            return Err(OchreError::InvalidParameter(
                "n_estimators must be >= 1 ( >_< )".to_string(),
            ));
        }
        let crit = criterion
            .unwrap_or_else(|| "gini".to_string())
            .to_lowercase();
        if crit != "gini" && crit != "entropy" {
            return Err(OchreError::InvalidCriterion(crit));
        }
        let mss = min_samples_split.unwrap_or(2);
        if mss < 2 {
            return Err(OchreError::InvalidParameter(format!(
                "min_samples_split={} must be >= 2 ( >_< )",
                mss
            )));
        }
        let msl = min_samples_leaf.unwrap_or(1);
        if msl < 1 {
            return Err(OchreError::InvalidParameter(format!(
                "min_samples_leaf={} must be >= 1 ( >_< )",
                msl
            )));
        }
        let mf = validate_max_features_spec(max_features.as_deref())?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            extra_trees,
            n_estimators: n_est,
            criterion: crit,
            max_depth,
            min_samples_split: mss,
            min_samples_leaf: msl,
            max_features: mf,
            bootstrap: bootstrap.unwrap_or(!extra_trees),
            random_state,
            device: dev,
            device_: None,
            trees: Vec::new(),
            classes_: Vec::new(),
            n_features_in_: None,
            feature_importances_: None,
        })
    }

    fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev.clone();
        if self.n_features_in_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        Ok(())
    }

    fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_x_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let max_feat = resolve_max_features(self.max_features.as_deref(), x.cols, true)?;

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

        let class_idx: Vec<usize> = y
            .iter()
            .map(|&val| {
                classes
                    .iter()
                    .position(|&c| (c - val).abs() <= 1e-12)
                    .unwrap_or(0)
            })
            .collect();

        // Check backend availability; CPU/TPU uses Rayon parallel tree building
        let _backend = get_backend(&self.device);

        let base_seed = resolve_base_seed(self.random_state);
        let criterion_entropy = self.criterion == "entropy";
        let n_classes = classes.len();
        let n_samples = x.rows;
        let n_features = x.cols;

        let results: Vec<(ClfTreeNode, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let seed = base_seed
                    .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ 0xA5A5_5A5A_1234_5678;
                let mut rng = SplitMix64::new(seed);
                let (samples, total_tree_weight) =
                    build_bootstrap_samples(n_samples, self.bootstrap, &sw, &mut rng);
                let cfg = ClfBuildCfg {
                    criterion_entropy,
                    extra_trees: self.extra_trees,
                    max_depth: self.max_depth,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: self.min_samples_leaf,
                    max_features: max_feat,
                    n_classes,
                    total_tree_weight,
                    x,
                    class_idx: &class_idx,
                };
                let mut imp = vec![0.0; n_features];
                let root = build_clf_tree(&cfg, &samples, 0, &mut rng, &mut imp);
                (root, imp)
            })
            .collect();

        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut all_imp = Vec::with_capacity(self.n_estimators);
        for (t, imp) in results {
            trees.push(t);
            all_imp.push(imp);
        }

        self.trees = trees;
        self.classes_ = classes;
        self.n_features_in_ = Some(n_features);
        self.feature_importances_ = Some(normalize_and_average_importances(all_imp, n_features));
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_x_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let n_classes = self.classes_.len();
        let n_trees = self.trees.len() as f64;
        let mut out = vec![0.0; x.rows * n_classes];

        out.par_chunks_mut(n_classes)
            .enumerate()
            .for_each(|(r, dst)| {
                let row = &x.data[r * expected..(r + 1) * expected];
                for tree in &self.trees {
                    let probs = tree.predict_proba_row(row);
                    for (d, &p) in dst.iter_mut().zip(probs.iter()) {
                        *d += p;
                    }
                }
                for d in dst.iter_mut() {
                    *d /= n_trees;
                }
            });

        Matrix::from_vec(x.rows, n_classes, out)
    }

    fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let proba = self.predict_proba(x)?;
        let n_classes = self.classes_.len();
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row = &proba.data[r * n_classes..(r + 1) * n_classes];
            let mut best_c = 0usize;
            let mut best_p = f64::NEG_INFINITY;
            for (c, &p) in row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            preds.push(self.classes_[best_c]);
        }
        Ok(preds)
    }

    fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|&(&p, &t)| (p - t).abs() <= 1e-12)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    fn to_bytes(&self, tag: u8) -> Vec<u8> {
        let mut buf = write_header(tag);
        write_bool(&mut buf, self.extra_trees);
        write_usize(&mut buf, self.n_estimators);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.max_features);
        write_bool(&mut buf, self.bootstrap);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_usize(&mut buf, self.trees.len());
        for tree in &self.trees {
            tree.write_to(&mut buf);
        }
        buf
    }

    fn from_bytes(bytes: &[u8], expected_tag: u8, device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, expected_tag)?;
        let extra_trees = r.read_bool()?;
        let n_estimators = r.read_usize()?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let max_features = r.read_opt_str()?;
        let bootstrap = r.read_bool()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_trees = r.read_usize()?;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees.push(ClfTreeNode::read_from(&mut r)?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            extra_trees,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device: dev,
            device_,
            trees,
            classes_,
            n_features_in_,
            feature_importances_,
        })
    }
}

// =========================================================================
// SHARED FOREST REGRESSOR CORE (RF & EXTRA TREES)
// =========================================================================

#[derive(Debug, Clone)]
struct ForestRegressorCore {
    extra_trees: bool,
    n_estimators: usize,
    criterion: String,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    device: DeviceType,
    device_: Option<String>,
    trees: Vec<RegTreeNode>,
    n_features_in_: Option<usize>,
    feature_importances_: Option<Vec<f64>>,
}

impl ForestRegressorCore {
    #[allow(clippy::too_many_arguments)]
    fn new(
        extra_trees: bool,
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let n_est = n_estimators.unwrap_or(100);
        if n_est == 0 {
            return Err(OchreError::InvalidParameter(
                "n_estimators must be >= 1 ( >_< )".to_string(),
            ));
        }
        let crit = criterion
            .unwrap_or_else(|| "squared_error".to_string())
            .to_lowercase();
        if crit != "squared_error" {
            return Err(OchreError::InvalidRegressorCriterion(crit));
        }
        let mss = min_samples_split.unwrap_or(2);
        if mss < 2 {
            return Err(OchreError::InvalidParameter(format!(
                "min_samples_split={} must be >= 2 ( >_< )",
                mss
            )));
        }
        let msl = min_samples_leaf.unwrap_or(1);
        if msl < 1 {
            return Err(OchreError::InvalidParameter(format!(
                "min_samples_leaf={} must be >= 1 ( >_< )",
                msl
            )));
        }
        let mf = validate_max_features_spec(max_features.as_deref())?;
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            extra_trees,
            n_estimators: n_est,
            criterion: crit,
            max_depth,
            min_samples_split: mss,
            min_samples_leaf: msl,
            max_features: mf,
            bootstrap: bootstrap.unwrap_or(!extra_trees),
            random_state,
            device: dev,
            device_: None,
            trees: Vec::new(),
            n_features_in_: None,
            feature_importances_: None,
        })
    }

    fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev.clone();
        if self.n_features_in_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        Ok(())
    }

    fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        validate_x_finite(x)?;
        validate_y_finite(y, x.rows)?;
        let sw = validate_sample_weight(sample_weight, x.rows)?;
        let max_feat = resolve_max_features(self.max_features.as_deref(), x.cols, false)?;

        let backend = get_backend(&self.device);
        let base_seed = resolve_base_seed(self.random_state);
        let n_samples = x.rows;
        let n_features = x.cols;

        // N6 Accelerator-First Path on CUDA / wgpu for RandomForestRegressor
        let use_n6_gpu = !self.extra_trees
            && matches!(
                backend,
                Backend::CudaSingle(_) | Backend::CudaMulti(_) | Backend::Wgpu
            );

        let (trees, all_imp) = if use_n6_gpu {
            let binned = backend.quantize_features(x, 256)?;
            let mut trees = Vec::with_capacity(self.n_estimators);
            let mut all_imp = Vec::with_capacity(self.n_estimators);
            for tree_idx in 0..self.n_estimators {
                let seed = base_seed
                    .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ 0x5A5A_A5A5_8765_4321;
                let mut rng = SplitMix64::new(seed);
                let (samples, total_tree_weight) =
                    build_bootstrap_samples(n_samples, self.bootstrap, &sw, &mut rng);

                let mut grad = vec![0.0; n_samples];
                let mut hess = vec![0.0; n_samples];
                let mut active_indices = Vec::with_capacity(samples.len());
                for &(idx, w) in &samples {
                    grad[idx] = w * y[idx];
                    hess[idx] = w;
                    active_indices.push(idx as u32);
                }

                let root_hist =
                    backend.batched_histogram_grad(&binned, &active_indices, &grad, &hess)?;
                let mut imp = vec![0.0; n_features];
                let tree = build_reg_tree_n6(
                    &backend,
                    &binned,
                    &grad,
                    &hess,
                    &active_indices,
                    &root_hist,
                    0,
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    max_feat,
                    total_tree_weight,
                    &mut rng,
                    &mut imp,
                )?;
                trees.push(tree);
                all_imp.push(imp);
            }
            (trees, all_imp)
        } else {
            let results: Vec<(RegTreeNode, Vec<f64>)> = (0..self.n_estimators)
                .into_par_iter()
                .map(|tree_idx| {
                    let seed = base_seed
                        .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ 0x5A5A_A5A5_8765_4321;
                    let mut rng = SplitMix64::new(seed);
                    let (samples, total_tree_weight) =
                        build_bootstrap_samples(n_samples, self.bootstrap, &sw, &mut rng);
                    let cfg = RegBuildCfg {
                        extra_trees: self.extra_trees,
                        max_depth: self.max_depth,
                        min_samples_split: self.min_samples_split,
                        min_samples_leaf: self.min_samples_leaf,
                        max_features: max_feat,
                        total_tree_weight,
                        x,
                        y,
                    };
                    let mut imp = vec![0.0; n_features];
                    let root = build_reg_tree(&cfg, &samples, 0, &mut rng, &mut imp);
                    (root, imp)
                })
                .collect();

            let mut trees = Vec::with_capacity(self.n_estimators);
            let mut all_imp = Vec::with_capacity(self.n_estimators);
            for (t, imp) in results {
                trees.push(t);
                all_imp.push(imp);
            }
            (trees, all_imp)
        };

        self.trees = trees;
        self.n_features_in_ = Some(n_features);
        self.feature_importances_ = Some(normalize_and_average_importances(all_imp, n_features));
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        validate_x_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let n_trees = self.trees.len() as f64;
        let preds: Vec<f64> = (0..x.rows)
            .into_par_iter()
            .map(|r| {
                let row = &x.data[r * expected..(r + 1) * expected];
                let mut sum = 0.0;
                for tree in &self.trees {
                    sum += tree.predict_row(row);
                }
                sum / n_trees
            })
            .collect();
        Ok(preds)
    }

    fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        validate_y_finite(y, x.rows)?;
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    fn to_bytes(&self, tag: u8) -> Vec<u8> {
        let mut buf = write_header(tag);
        write_bool(&mut buf, self.extra_trees);
        write_usize(&mut buf, self.n_estimators);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.max_features);
        write_bool(&mut buf, self.bootstrap);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_f64_vec(&mut buf, &self.feature_importances_);
        write_usize(&mut buf, self.trees.len());
        for tree in &self.trees {
            tree.write_to(&mut buf);
        }
        buf
    }

    fn from_bytes(bytes: &[u8], expected_tag: u8, device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, expected_tag)?;
        let extra_trees = r.read_bool()?;
        let n_estimators = r.read_usize()?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let max_features = r.read_opt_str()?;
        let bootstrap = r.read_bool()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let feature_importances_ = r.read_opt_f64_vec()?;
        let n_trees = r.read_usize()?;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees.push(RegTreeNode::read_from(&mut r)?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            extra_trees,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device: dev,
            device_,
            trees,
            n_features_in_,
            feature_importances_,
        })
    }
}

// =========================================================================
// PUBLIC RUST STRUCTS: RF & EXTRA TREES
// =========================================================================

#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    core: ForestClassifierCore,
    pub device_: Option<String>,
}

impl RandomForestClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let core = ForestClassifierCore::new(
            false,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device,
        )?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.core.set_device(device)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.core.fit(x, y, sample_weight)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.core.predict_proba(x)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.core.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        self.core.score(x, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.core.to_bytes(TAG_RANDOM_FOREST_CLASSIFIER)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let core = ForestClassifierCore::from_bytes(bytes, TAG_RANDOM_FOREST_CLASSIFIER, device)?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExtraTreesClassifier {
    core: ForestClassifierCore,
    pub device_: Option<String>,
}

impl ExtraTreesClassifier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let core = ForestClassifierCore::new(
            true,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device,
        )?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.core.set_device(device)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.core.fit(x, y, sample_weight)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        self.core.predict_proba(x)
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.core.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        self.core.score(x, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.core.to_bytes(TAG_EXTRA_TREES_CLASSIFIER)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let core = ForestClassifierCore::from_bytes(bytes, TAG_EXTRA_TREES_CLASSIFIER, device)?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    core: ForestRegressorCore,
    pub device_: Option<String>,
}

impl RandomForestRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let core = ForestRegressorCore::new(
            false,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device,
        )?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.core.set_device(device)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.core.fit(x, y, sample_weight)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.core.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        self.core.score(x, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.core.to_bytes(TAG_RANDOM_FOREST_REGRESSOR)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let core = ForestRegressorCore::from_bytes(bytes, TAG_RANDOM_FOREST_REGRESSOR, device)?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExtraTreesRegressor {
    core: ForestRegressorCore,
    pub device_: Option<String>,
}

impl ExtraTreesRegressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_estimators: Option<usize>,
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        max_features: Option<String>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let core = ForestRegressorCore::new(
            true,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            device,
        )?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        self.core.set_device(device)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<()> {
        self.core.fit(x, y, sample_weight)?;
        self.device_ = self.core.device_.clone();
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        self.core.predict(x)
    }

    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        self.core.score(x, y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.core.to_bytes(TAG_EXTRA_TREES_REGRESSOR)
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let core = ForestRegressorCore::from_bytes(bytes, TAG_EXTRA_TREES_REGRESSOR, device)?;
        Ok(Self {
            device_: core.device_.clone(),
            core,
        })
    }
}

// =========================================================================
// ISOLATION FOREST (LIU ET AL. 2008)
// =========================================================================

#[derive(Debug, Clone)]
enum IsoTreeNode {
    Leaf {
        size: usize,
    },
    Internal {
        feature: usize,
        split_val: f64,
        left: Box<IsoTreeNode>,
        right: Box<IsoTreeNode>,
    },
}

#[inline]
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else if n == 2 {
        1.0
    } else {
        const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
        let nf = n as f64;
        let h = (nf - 1.0).ln() + EULER_MASCHERONI;
        2.0 * h - 2.0 * (nf - 1.0) / nf
    }
}

impl IsoTreeNode {
    fn path_length(&self, row: &[f64], depth: usize) -> f64 {
        match self {
            IsoTreeNode::Leaf { size } => depth as f64 + c_factor(*size),
            IsoTreeNode::Internal {
                feature,
                split_val,
                left,
                right,
            } => {
                if row[*feature] < *split_val {
                    left.path_length(row, depth + 1)
                } else {
                    right.path_length(row, depth + 1)
                }
            }
        }
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            IsoTreeNode::Leaf { size } => {
                write_u8(buf, 0);
                write_usize(buf, *size);
            }
            IsoTreeNode::Internal {
                feature,
                split_val,
                left,
                right,
            } => {
                write_u8(buf, 1);
                write_usize(buf, *feature);
                write_f64(buf, *split_val);
                left.write_to(buf);
                right.write_to(buf);
            }
        }
    }

    fn read_from(r: &mut ByteReader<'_>) -> OchreResult<Self> {
        match r.read_u8()? {
            0 => Ok(IsoTreeNode::Leaf {
                size: r.read_usize()?,
            }),
            1 => {
                let feature = r.read_usize()?;
                let split_val = r.read_f64()?;
                let left = Box::new(Self::read_from(r)?);
                let right = Box::new(Self::read_from(r)?);
                Ok(IsoTreeNode::Internal {
                    feature,
                    split_val,
                    left,
                    right,
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
}

fn build_iso_tree(
    x: &Matrix,
    indices: &[usize],
    features: &[usize],
    depth: usize,
    height_limit: usize,
    rng: &mut SplitMix64,
) -> IsoTreeNode {
    let n = indices.len();
    if depth >= height_limit || n <= 1 {
        return IsoTreeNode::Leaf { size: n };
    }

    // Shuffle candidate feature order and pick first non-constant feature
    let mut cand = features.to_vec();
    for i in 0..cand.len() {
        let j = i + rng.next_usize(cand.len() - i);
        cand.swap(i, j);
        let f = cand[i];
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &r in indices {
            let v = x.data[r * x.cols + f];
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        if max_v > min_v + 1e-12 {
            let split_val = min_v + rng.next_f64() * (max_v - min_v);
            let split_val = if split_val <= min_v || split_val >= max_v {
                0.5 * (min_v + max_v)
            } else {
                split_val
            };
            let mut left_idx = Vec::with_capacity(n / 2);
            let mut right_idx = Vec::with_capacity(n / 2);
            for &r in indices {
                if x.data[r * x.cols + f] < split_val {
                    left_idx.push(r);
                } else {
                    right_idx.push(r);
                }
            }
            if left_idx.is_empty() || right_idx.is_empty() {
                return IsoTreeNode::Leaf { size: n };
            }
            let left = Box::new(build_iso_tree(
                x,
                &left_idx,
                features,
                depth + 1,
                height_limit,
                rng,
            ));
            let right = Box::new(build_iso_tree(
                x,
                &right_idx,
                features,
                depth + 1,
                height_limit,
                rng,
            ));
            return IsoTreeNode::Internal {
                feature: f,
                split_val,
                left,
                right,
            };
        }
    }

    IsoTreeNode::Leaf { size: n }
}

fn percentile_linear(vals: &[f64], q: f64) -> f64 {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n == 0 {
        return -0.5;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[derive(Debug, Clone)]
pub struct IsolationForest {
    pub n_estimators: usize,
    pub max_samples: Option<usize>,
    pub contamination: Option<f64>,
    pub max_features: f64,
    pub bootstrap: bool,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    trees: Vec<IsoTreeNode>,
    pub n_features_in_: Option<usize>,
    pub max_samples_: Option<usize>,
    pub offset_: Option<f64>,
}

impl IsolationForest {
    pub fn new(
        n_estimators: Option<usize>,
        max_samples: Option<usize>,
        contamination: Option<f64>,
        max_features: Option<f64>,
        bootstrap: Option<bool>,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        let n_est = n_estimators.unwrap_or(100);
        if n_est == 0 {
            return Err(OchreError::InvalidParameter(
                "n_estimators must be >= 1 ( >_< )".to_string(),
            ));
        }
        if let Some(ms) = max_samples
            && ms == 0
        {
            return Err(OchreError::InvalidParameter(
                "max_samples must be >= 1 ( >_< )".to_string(),
            ));
        }
        if let Some(c) = contamination
            && (!c.is_finite() || c <= 0.0 || c > 0.5)
        {
            return Err(OchreError::InvalidParameter(format!(
                "contamination={} must be in (0.0, 0.5] or 'auto' ( >_< )",
                c
            )));
        }
        let mf = max_features.unwrap_or(1.0);
        if !mf.is_finite() || mf <= 0.0 || mf > 1.0 {
            return Err(OchreError::InvalidParameter(format!(
                "max_features={} must be in (0.0, 1.0] ( >_< )",
                mf
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_estimators: n_est,
            max_samples,
            contamination,
            max_features: mf,
            bootstrap: bootstrap.unwrap_or(false),
            random_state,
            device: dev,
            device_: None,
            trees: Vec::new(),
            n_features_in_: None,
            max_samples_: None,
            offset_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev.clone();
        if self.n_features_in_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_x_finite(x)?;
        let _backend = get_backend(&self.device);

        let n_samples = x.rows;
        let n_features = x.cols;
        let max_s = self
            .max_samples
            .unwrap_or_else(|| n_samples.min(256))
            .min(n_samples)
            .max(1);
        let max_f = ((self.max_features * n_features as f64).ceil() as usize).clamp(1, n_features);
        let height_limit = if max_s <= 1 {
            0
        } else {
            (max_s as f64).log2().ceil() as usize
        };

        let base_seed = resolve_base_seed(self.random_state);

        let trees: Vec<IsoTreeNode> = (0..self.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let seed = base_seed
                    .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ 0x150F_0F15_0F15_0F15;
                let mut rng = SplitMix64::new(seed);

                let row_indices: Vec<usize> = if self.bootstrap {
                    (0..max_s).map(|_| rng.next_usize(n_samples)).collect()
                } else {
                    let mut pool: Vec<usize> = (0..n_samples).collect();
                    for i in 0..max_s {
                        let j = i + rng.next_usize(n_samples - i);
                        pool.swap(i, j);
                    }
                    pool.truncate(max_s);
                    pool
                };

                let mut feat_pool = Vec::with_capacity(n_features);
                sample_feature_subset(&mut feat_pool, n_features, max_f, &mut rng);

                build_iso_tree(x, &row_indices, &feat_pool, 0, height_limit, &mut rng)
            })
            .collect();

        self.trees = trees;
        self.n_features_in_ = Some(n_features);
        self.max_samples_ = Some(max_s);
        self.device_ = Some(self.device.to_string());

        let offset = match self.contamination {
            None => -0.5,
            Some(c) => {
                let scores = self.score_samples(x)?;
                percentile_linear(&scores, c)
            }
        };
        self.offset_ = Some(offset);
        Ok(())
    }

    pub fn score_samples(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let expected = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let max_s = self.max_samples_.ok_or(OchreError::NotFitted)?;
        validate_x_finite(x)?;
        if x.cols != expected {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected,
            });
        }
        let n_trees = self.trees.len() as f64;
        let c_n = c_factor(max_s);

        let scores: Vec<f64> = (0..x.rows)
            .into_par_iter()
            .map(|r| {
                if c_n <= 0.0 {
                    return -0.5;
                }
                let row = &x.data[r * expected..(r + 1) * expected];
                let mut depth_sum = 0.0;
                for tree in &self.trees {
                    depth_sum += tree.path_length(row, 0);
                }
                let mean_depth = depth_sum / n_trees;
                -(2.0_f64.powf(-mean_depth / c_n))
            })
            .collect();

        Ok(scores)
    }

    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let offset = self.offset_.ok_or(OchreError::NotFitted)?;
        let scores = self.score_samples(x)?;
        Ok(scores.into_iter().map(|s| s - offset).collect())
    }

    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<i64>> {
        let dec = self.decision_function(x)?;
        Ok(dec
            .into_iter()
            .map(|d| if d >= 0.0 { 1 } else { -1 })
            .collect())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ISOLATION_FOREST);
        write_usize(&mut buf, self.n_estimators);
        write_opt_usize(&mut buf, self.max_samples);
        write_opt_f64(&mut buf, self.contamination);
        write_f64(&mut buf, self.max_features);
        write_bool(&mut buf, self.bootstrap);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_usize(&mut buf, self.max_samples_);
        write_opt_f64(&mut buf, self.offset_);
        write_usize(&mut buf, self.trees.len());
        for tree in &self.trees {
            tree.write_to(&mut buf);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ISOLATION_FOREST)?;
        let n_estimators = r.read_usize()?;
        let max_samples = r.read_opt_usize()?;
        let contamination = r.read_opt_f64()?;
        let max_features = r.read_f64()?;
        let bootstrap = r.read_bool()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let max_samples_ = r.read_opt_usize()?;
        let offset_ = r.read_opt_f64()?;
        let n_trees = r.read_usize()?;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees.push(IsoTreeNode::read_from(&mut r)?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            n_estimators,
            max_samples,
            contamination,
            max_features,
            bootstrap,
            random_state,
            device: dev,
            device_,
            trees,
            n_features_in_,
            max_samples_,
            offset_,
        })
    }
}

// =========================================================================
// PYO3 WRAPPERS & MODULE REGISTRATION
// =========================================================================

macro_rules! impl_py_forest_classifier {
    ($py_name:ident, $rust_ty:ty, $py_str:literal) => {
        #[pyclass(from_py_object, name = $py_str)]
        #[derive(Clone)]
        pub struct $py_name {
            inner: $rust_ty,
        }

        pymethods_with_dit!($py_name, $rust_ty, {
            #[new]
            #[pyo3(signature = (n_estimators = 100, criterion = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = None, bootstrap = None, random_state = None, device = None))]
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                n_estimators: usize,
                criterion: Option<String>,
                max_depth: Option<usize>,
                min_samples_split: usize,
                min_samples_leaf: usize,
                max_features: Option<String>,
                bootstrap: Option<bool>,
                random_state: Option<u64>,
                device: Option<String>,
            ) -> PyResult<Self> {
                Ok(Self {
                    inner: <$rust_ty>::new(
                        Some(n_estimators),
                        criterion,
                        max_depth,
                        Some(min_samples_split),
                        Some(min_samples_leaf),
                        max_features,
                        bootstrap,
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
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let y_vec = buffer_to_f64_1d(py, &y)?;
                let sw_vec = match sample_weight {
                    Some(ref sw) => Some(buffer_to_f64_1d(py, sw)?),
                    None => None,
                };
                self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
                Ok(())
            }

            pub fn predict_proba_buffer<'py>(
                &self,
                py: Python<'py>,
                x: PyBuffer<f64>,
            ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let out = self.inner.predict_proba(&mat)?;
                matrix_to_pybytes(py, &out)
            }

            pub fn predict_buffer<'py>(
                &self,
                py: Python<'py>,
                x: PyBuffer<f64>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let out = self.inner.predict(&mat)?;
                f64_slice_to_pybytes(py, &out)
            }

            pub fn score_buffer(
                &self,
                py: Python<'_>,
                x: PyBuffer<f64>,
                y: PyBuffer<f64>,
            ) -> PyResult<f64> {
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let y_vec = buffer_to_f64_1d(py, &y)?;
                Ok(self.inner.score(&mat, &y_vec)?)
            }

            #[getter]
            pub fn classes_bytes<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Option<Bound<'py, PyBytes>>> {
                if self.inner.core.n_features_in_.is_none() {
                    return Ok(None);
                }
                Ok(Some(f64_slice_to_pybytes(py, &self.inner.core.classes_)?))
            }

            #[getter]
            pub fn n_features_in_(&self) -> Option<usize> {
                self.inner.core.n_features_in_
            }

            #[getter]
            pub fn feature_importances_bytes<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Option<Bound<'py, PyBytes>>> {
                match &self.inner.core.feature_importances_ {
                    Some(imp) => Ok(Some(f64_slice_to_pybytes(py, imp)?)),
                    None => Ok(None),
                }
            }
        });
    };
}

macro_rules! impl_py_forest_regressor {
    ($py_name:ident, $rust_ty:ty, $py_str:literal) => {
        #[pyclass(from_py_object, name = $py_str)]
        #[derive(Clone)]
        pub struct $py_name {
            inner: $rust_ty,
        }

        pymethods_with_dit!($py_name, $rust_ty, {
            #[new]
            #[pyo3(signature = (n_estimators = 100, criterion = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = None, bootstrap = None, random_state = None, device = None))]
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                n_estimators: usize,
                criterion: Option<String>,
                max_depth: Option<usize>,
                min_samples_split: usize,
                min_samples_leaf: usize,
                max_features: Option<String>,
                bootstrap: Option<bool>,
                random_state: Option<u64>,
                device: Option<String>,
            ) -> PyResult<Self> {
                Ok(Self {
                    inner: <$rust_ty>::new(
                        Some(n_estimators),
                        criterion,
                        max_depth,
                        Some(min_samples_split),
                        Some(min_samples_leaf),
                        max_features,
                        bootstrap,
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
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let y_vec = buffer_to_f64_1d(py, &y)?;
                let sw_vec = match sample_weight {
                    Some(ref sw) => Some(buffer_to_f64_1d(py, sw)?),
                    None => None,
                };
                self.inner.fit(&mat, &y_vec, sw_vec.as_deref())?;
                Ok(())
            }

            pub fn predict_buffer<'py>(
                &self,
                py: Python<'py>,
                x: PyBuffer<f64>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let out = self.inner.predict(&mat)?;
                f64_slice_to_pybytes(py, &out)
            }

            pub fn score_buffer(
                &self,
                py: Python<'_>,
                x: PyBuffer<f64>,
                y: PyBuffer<f64>,
            ) -> PyResult<f64> {
                let mat = buffer_to_pooled_matrix(py, &x)?;
                let y_vec = buffer_to_f64_1d(py, &y)?;
                Ok(self.inner.score(&mat, &y_vec)?)
            }

            #[getter]
            pub fn n_features_in_(&self) -> Option<usize> {
                self.inner.core.n_features_in_
            }

            #[getter]
            pub fn feature_importances_bytes<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Option<Bound<'py, PyBytes>>> {
                match &self.inner.core.feature_importances_ {
                    Some(imp) => Ok(Some(f64_slice_to_pybytes(py, imp)?)),
                    None => Ok(None),
                }
            }
        });
    };
}

impl_py_forest_classifier!(
    PyRandomForestClassifier,
    RandomForestClassifier,
    "RandomForestClassifier"
);
impl_py_forest_classifier!(
    PyExtraTreesClassifier,
    ExtraTreesClassifier,
    "ExtraTreesClassifier"
);
impl_py_forest_regressor!(
    PyRandomForestRegressor,
    RandomForestRegressor,
    "RandomForestRegressor"
);
impl_py_forest_regressor!(
    PyExtraTreesRegressor,
    ExtraTreesRegressor,
    "ExtraTreesRegressor"
);

#[pyclass(from_py_object, name = "IsolationForest")]
#[derive(Clone)]
pub struct PyIsolationForest {
    inner: IsolationForest,
}

pymethods_with_dit!(PyIsolationForest, IsolationForest, {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_samples = None,
        contamination = None,
        max_features = 1.0,
        bootstrap = false,
        random_state = None,
        device = None
    ))]
    pub fn new(
        n_estimators: usize,
        max_samples: Option<usize>,
        contamination: Option<f64>,
        max_features: f64,
        bootstrap: bool,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IsolationForest::new(
                Some(n_estimators),
                max_samples,
                contamination,
                Some(max_features),
                Some(bootstrap),
                random_state,
                device,
            )?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn score_samples_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.score_samples(&mat)?;
        f64_slice_to_pybytes(py, &out)
    }

    pub fn decision_function_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.decision_function(&mat)?;
        f64_slice_to_pybytes(py, &out)
    }

    pub fn predict_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.predict(&mat)?;
        i64_slice_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn max_samples_(&self) -> Option<usize> {
        self.inner.max_samples_
    }

    #[getter]
    pub fn offset_(&self) -> Option<f64> {
        self.inner.offset_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRandomForestClassifier>()?;
    m.add_class::<PyRandomForestRegressor>()?;
    m.add_class::<PyExtraTreesClassifier>()?;
    m.add_class::<PyExtraTreesRegressor>()?;
    m.add_class::<PyIsolationForest>()?;
    Ok(())
}
