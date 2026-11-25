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

#[derive(Debug, Clone, Copy)]
struct FlatClfNode {
    feature_or_offset: u32,
    left: u32,
    right: u32,
    threshold: f64,
}

#[inline]
fn push_flat_clf_leaf_probs(
    probs: &[f64],
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let idx = nodes.len();
    let offset = probs_out.len() as u32;
    probs_out.extend_from_slice(probs);
    nodes.push(FlatClfNode {
        feature_or_offset: offset,
        left: 0,
        right: u32::MAX,
        threshold: 0.0,
    });
    idx
}

#[inline]
fn push_flat_clf_leaf_weighted(
    counts: &[f64],
    total_w: f64,
    n_classes: usize,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let idx = nodes.len();
    let offset = probs_out.len() as u32;
    if total_w > 0.0 {
        let inv_w = 1.0 / total_w;
        for &c in counts {
            probs_out.push(c * inv_w);
        }
    } else {
        let uniform = 1.0 / n_classes.max(1) as f64;
        probs_out.resize(probs_out.len() + n_classes, uniform);
    }
    nodes.push(FlatClfNode {
        feature_or_offset: offset,
        left: 0,
        right: u32::MAX,
        threshold: 0.0,
    });
    idx
}

#[inline]
fn push_flat_clf_leaf_u32(
    counts: &[u32],
    n: usize,
    n_classes: usize,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let idx = nodes.len();
    let offset = probs_out.len() as u32;
    if n > 0 {
        let inv_n = 1.0 / (n as f64);
        for &c in counts {
            probs_out.push((c as f64) * inv_n);
        }
    } else {
        let uniform = 1.0 / n_classes.max(1) as f64;
        probs_out.resize(probs_out.len() + n_classes, uniform);
    }
    nodes.push(FlatClfNode {
        feature_or_offset: offset,
        left: 0,
        right: u32::MAX,
        threshold: 0.0,
    });
    idx
}

fn write_flat_clf_node(
    nodes: &[FlatClfNode],
    probs: &[f64],
    n_classes: usize,
    idx: usize,
    buf: &mut Vec<u8>,
) {
    let node = nodes[idx];
    if node.right == u32::MAX {
        write_u8(buf, 0);
        let off = node.feature_or_offset as usize;
        write_f64_vec(buf, &probs[off..off + n_classes]);
    } else {
        write_u8(buf, 1);
        write_usize(buf, node.feature_or_offset as usize);
        write_f64(buf, node.threshold);
        write_flat_clf_node(nodes, probs, n_classes, node.left as usize, buf);
        write_flat_clf_node(nodes, probs, n_classes, node.right as usize, buf);
    }
}

fn read_flat_clf_node(
    r: &mut ByteReader<'_>,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> OchreResult<usize> {
    match r.read_u8()? {
        0 => {
            let probs = r.read_f64_vec()?;
            Ok(push_flat_clf_leaf_probs(&probs, nodes, probs_out))
        }
        1 => {
            let feature = r.read_usize()?;
            let threshold = r.read_f64()?;
            let idx = nodes.len();
            nodes.push(FlatClfNode {
                feature_or_offset: feature as u32,
                left: 0,
                right: 0,
                threshold,
            });
            let left_idx = read_flat_clf_node(r, nodes, probs_out)?;
            let right_idx = read_flat_clf_node(r, nodes, probs_out)?;
            nodes[idx].left = left_idx as u32;
            nodes[idx].right = right_idx as u32;
            Ok(idx)
        }
        _ => Err(OchreError::InvalidTreeNodeTag),
    }
}

fn merge_flat_clf_trees(
    per_tree: Vec<(Vec<FlatClfNode>, Vec<f64>, Vec<f64>)>,
) -> (Vec<FlatClfNode>, Vec<usize>, Vec<f64>, Vec<Vec<f64>>) {
    let total_nodes: usize = per_tree.iter().map(|(n, _, _)| n.len()).sum();
    let total_probs: usize = per_tree.iter().map(|(_, p, _)| p.len()).sum();
    let mut flat_nodes = Vec::with_capacity(total_nodes);
    let mut flat_roots = Vec::with_capacity(per_tree.len());
    let mut flat_probs = Vec::with_capacity(total_probs);
    let mut all_imp = Vec::with_capacity(per_tree.len());
    for (mut t_nodes, t_probs, imp) in per_tree {
        let node_base = flat_nodes.len() as u32;
        let prob_base = flat_probs.len() as u32;
        flat_roots.push(node_base as usize);
        for node in &mut t_nodes {
            if node.right == u32::MAX {
                node.feature_or_offset += prob_base;
            } else {
                node.left += node_base;
                node.right += node_base;
            }
        }
        flat_nodes.extend_from_slice(&t_nodes);
        flat_probs.extend_from_slice(&t_probs);
        all_imp.push(imp);
    }
    (flat_nodes, flat_roots, flat_probs, all_imp)
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

fn compute_root_min_max(x_rows: &[f64], n_samples: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    if n_samples >= 4096 {
        x_rows
            .par_chunks(4096 * p)
            .map(|chunk| {
                let mut mn = vec![f64::INFINITY; p];
                let mut mx = vec![f64::NEG_INFINITY; p];
                for row in chunk.chunks_exact(p) {
                    for f in 0..p {
                        let v = row[f];
                        if v < mn[f] {
                            mn[f] = v;
                        }
                        if v > mx[f] {
                            mx[f] = v;
                        }
                    }
                }
                (mn, mx)
            })
            .reduce(
                || (vec![f64::INFINITY; p], vec![f64::NEG_INFINITY; p]),
                |(mut mn_a, mut mx_a), (mn_b, mx_b)| {
                    for f in 0..p {
                        if mn_b[f] < mn_a[f] {
                            mn_a[f] = mn_b[f];
                        }
                        if mx_b[f] > mx_a[f] {
                            mx_a[f] = mx_b[f];
                        }
                    }
                    (mn_a, mx_a)
                },
            )
    } else {
        let mut mn = vec![f64::INFINITY; p];
        let mut mx = vec![f64::NEG_INFINITY; p];
        for row in x_rows.chunks_exact(p) {
            for f in 0..p {
                let v = row[f];
                if v < mn[f] {
                    mn[f] = v;
                }
                if v > mx[f] {
                    mx[f] = v;
                }
            }
        }
        (mn, mx)
    }
}

#[inline]
fn predict_flat_clf_proba<'a>(
    nodes: &[FlatClfNode],
    probs: &'a [f64],
    n_classes: usize,
    root: usize,
    row: &[f64],
) -> &'a [f64] {
    let mut idx = root;
    loop {
        let node = nodes[idx];
        if node.right == u32::MAX {
            let off = node.feature_or_offset as usize;
            return &probs[off..off + n_classes];
        }
        if row[node.feature_or_offset as usize] <= node.threshold {
            idx = node.left as usize;
        } else {
            idx = node.right as usize;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FlatRegNode {
    feature: u32,
    left: u32,
    right: u32,
    value_or_threshold: f64,
}

#[inline]
fn push_flat_reg_leaf(value: f64, nodes: &mut Vec<FlatRegNode>) -> usize {
    let idx = nodes.len();
    nodes.push(FlatRegNode {
        feature: 0,
        left: 0,
        right: u32::MAX,
        value_or_threshold: value,
    });
    idx
}

fn write_flat_reg_node(nodes: &[FlatRegNode], idx: usize, buf: &mut Vec<u8>) {
    let node = nodes[idx];
    if node.right == u32::MAX {
        write_u8(buf, 0);
        write_f64(buf, node.value_or_threshold);
    } else {
        write_u8(buf, 1);
        write_usize(buf, node.feature as usize);
        write_f64(buf, node.value_or_threshold);
        write_flat_reg_node(nodes, node.left as usize, buf);
        write_flat_reg_node(nodes, node.right as usize, buf);
    }
}

fn read_flat_reg_node(r: &mut ByteReader<'_>, nodes: &mut Vec<FlatRegNode>) -> OchreResult<usize> {
    match r.read_u8()? {
        0 => {
            let value = r.read_f64()?;
            Ok(push_flat_reg_leaf(value, nodes))
        }
        1 => {
            let feature = r.read_usize()?;
            let threshold = r.read_f64()?;
            let idx = nodes.len();
            nodes.push(FlatRegNode {
                feature: feature as u32,
                left: 0,
                right: 0,
                value_or_threshold: threshold,
            });
            let left_idx = read_flat_reg_node(r, nodes)?;
            let right_idx = read_flat_reg_node(r, nodes)?;
            nodes[idx].left = left_idx as u32;
            nodes[idx].right = right_idx as u32;
            Ok(idx)
        }
        _ => Err(OchreError::InvalidTreeNodeTag),
    }
}

fn merge_flat_reg_trees(
    per_tree: Vec<(Vec<FlatRegNode>, Vec<f64>)>,
) -> (Vec<FlatRegNode>, Vec<usize>, Vec<Vec<f64>>) {
    let total_nodes: usize = per_tree.iter().map(|(n, _)| n.len()).sum();
    let mut flat_nodes = Vec::with_capacity(total_nodes);
    let mut flat_roots = Vec::with_capacity(per_tree.len());
    let mut all_imp = Vec::with_capacity(per_tree.len());
    for (mut t_nodes, imp) in per_tree {
        let node_base = flat_nodes.len() as u32;
        flat_roots.push(node_base as usize);
        for node in &mut t_nodes {
            if node.right != u32::MAX {
                node.left += node_base;
                node.right += node_base;
            }
        }
        flat_nodes.extend_from_slice(&t_nodes);
        all_imp.push(imp);
    }
    (flat_nodes, flat_roots, all_imp)
}

#[inline]
fn predict_flat_reg_tree(nodes: &[FlatRegNode], root: usize, row: &[f64]) -> f64 {
    let mut idx = root;
    loop {
        let node = nodes[idx];
        if node.right == u32::MAX {
            return node.value_or_threshold;
        }
        if row[node.feature as usize] <= node.value_or_threshold {
            idx = node.left as usize;
        } else {
            idx = node.right as usize;
        }
    }
}

// =========================================================================
// SHARED EXTRATREES HELPERS (THRESHOLD GENERATION & ORDER-PRESERVING PARTITION)
// =========================================================================

#[inline]
fn draw_et_thresholds(
    feat_pool: &[usize],
    min_v: &[f64],
    max_v: &[f64],
    indexed_by_pool: bool,
    rng: &mut SplitMix64,
    active_feats: &mut Vec<usize>,
    active_thresh: &mut Vec<f64>,
) {
    active_feats.clear();
    active_thresh.clear();
    for (fi, &f) in feat_pool.iter().enumerate() {
        let slot = if indexed_by_pool { fi } else { f };
        let mn = min_v[slot];
        let mx = max_v[slot];
        if mx <= mn + 1e-12 {
            continue;
        }
        let thresh = mn + rng.next_f64() * (mx - mn);
        let thresh = if thresh >= mx {
            0.5 * (mn + mx)
        } else {
            thresh
        };
        active_feats.push(f);
        active_thresh.push(thresh);
    }
}

#[inline]
fn partition_u32_with_buf(
    x_rows: &[f64],
    p: usize,
    best_feat: usize,
    best_thresh: f64,
    left_len: usize,
    samples: &mut [u32],
    part_buf: &mut Vec<u32>,
) {
    let n = samples.len();
    if part_buf.len() < n {
        part_buf.resize(n, 0);
    }
    let buf = &mut part_buf[..n];
    let last = n.saturating_sub(1);
    let mut l_pos = 0usize;
    let mut r_pos = left_len;
    for &idx in samples.iter() {
        let go_left = x_rows[idx as usize * p + best_feat] <= best_thresh;
        let dst = if go_left { l_pos } else { r_pos };
        buf[dst.min(last)] = idx;
        l_pos += usize::from(go_left);
        r_pos += usize::from(!go_left);
    }
    samples.copy_from_slice(buf);
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
    n_samples: usize,
    n_features: usize,
    total_tree_weight: f64,
    x_rows: &'a [f64],
    x_cols: &'a [f64],
    class_idx: &'a [usize],
}

struct ClfTreeScratch {
    counts: Vec<f64>,
    feat_pool: Vec<usize>,
    left_counts: Vec<f64>,
    right_counts: Vec<f64>,
    sorted: Vec<(f64, usize, f64)>,
    et_min: Vec<f64>,
    et_max: Vec<f64>,
    et_active_feats: Vec<usize>,
    et_active_thresh: Vec<f64>,
    et_left_counts: Vec<f64>,
    et_left_w: Vec<f64>,
    et_left_n: Vec<usize>,
}

impl ClfTreeScratch {
    fn new(
        n_features: usize,
        max_feat: usize,
        n_classes: usize,
        sorted_cap: usize,
        extra_trees: bool,
    ) -> Self {
        let et_cap = if extra_trees { max_feat } else { 0 };
        Self {
            counts: vec![0.0; n_classes],
            feat_pool: Vec::with_capacity(n_features),
            left_counts: vec![0.0; n_classes],
            right_counts: vec![0.0; n_classes],
            sorted: Vec::with_capacity(sorted_cap),
            et_min: vec![0.0; et_cap],
            et_max: vec![0.0; et_cap],
            et_active_feats: Vec::with_capacity(et_cap),
            et_active_thresh: Vec::with_capacity(et_cap),
            et_left_counts: vec![0.0; et_cap * n_classes],
            et_left_w: vec![0.0; et_cap],
            et_left_n: vec![0; et_cap],
        }
    }
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

#[allow(clippy::too_many_arguments)]
fn build_clf_tree(
    cfg: &ClfBuildCfg<'_>,
    samples: &mut [(usize, f64)],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut ClfTreeScratch,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let n = samples.len();
    scratch.counts.fill(0.0);
    let mut total_w = 0.0;
    for &(idx, w) in samples.iter() {
        scratch.counts[cfg.class_idx[idx]] += w;
        total_w += w;
    }

    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| depth >= d)
    {
        return push_flat_clf_leaf_weighted(
            &scratch.counts,
            total_w,
            cfg.n_classes,
            nodes,
            probs_out,
        );
    }

    let parent_imp = impurity_weighted(&scratch.counts, total_w, cfg.criterion_entropy);
    if parent_imp <= 1e-12 {
        return push_flat_clf_leaf_weighted(
            &scratch.counts,
            total_w,
            cfg.n_classes,
            nodes,
            probs_out,
        );
    }

    sample_feature_subset(
        &mut scratch.feat_pool,
        cfg.n_features,
        cfg.max_features,
        rng,
    );

    let mut best_gain = 0.0;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;
    let n_samples = cfg.n_samples;
    let p = cfg.n_features;

    if cfg.extra_trees {
        let k = scratch.feat_pool.len();
        let n_classes = cfg.n_classes;
        let feat_pool = &scratch.feat_pool[..k];
        let min_v = &mut scratch.et_min[..k];
        let max_v = &mut scratch.et_max[..k];
        min_v.fill(f64::INFINITY);
        max_v.fill(f64::NEG_INFINITY);

        for &(idx, _) in samples.iter() {
            let row = &cfg.x_rows[idx * p..(idx + 1) * p];
            for fi in 0..k {
                let v = row[feat_pool[fi]];
                if v < min_v[fi] {
                    min_v[fi] = v;
                }
                if v > max_v[fi] {
                    max_v[fi] = v;
                }
            }
        }

        draw_et_thresholds(
            feat_pool,
            min_v,
            max_v,
            true,
            rng,
            &mut scratch.et_active_feats,
            &mut scratch.et_active_thresh,
        );

        let m = scratch.et_active_feats.len();
        if m > 0 {
            let active_feats = &scratch.et_active_feats[..m];
            let active_thresh = &scratch.et_active_thresh[..m];
            let left_counts = &mut scratch.et_left_counts[..m * n_classes];
            let left_w = &mut scratch.et_left_w[..m];
            let left_n = &mut scratch.et_left_n[..m];
            left_counts.fill(0.0);
            left_w.fill(0.0);
            left_n.fill(0);

            for &(idx, w) in samples.iter() {
                let row = &cfg.x_rows[idx * p..(idx + 1) * p];
                let c = cfg.class_idx[idx];
                for ai in 0..m {
                    if row[active_feats[ai]] <= active_thresh[ai] {
                        left_counts[ai * n_classes + c] += w;
                        left_w[ai] += w;
                        left_n[ai] += 1;
                    }
                }
            }

            for ai in 0..m {
                let ln = left_n[ai];
                let rn = n - ln;
                if ln < cfg.min_samples_leaf || rn < cfg.min_samples_leaf {
                    continue;
                }
                let lw = left_w[ai];
                let rw = total_w - lw;
                if lw <= 0.0 || rw <= 0.0 {
                    continue;
                }
                let lc = &left_counts[ai * n_classes..(ai + 1) * n_classes];
                for ((dst, &tot), &left) in scratch
                    .right_counts
                    .iter_mut()
                    .zip(scratch.counts.iter())
                    .zip(lc.iter())
                {
                    *dst = (tot - left).max(0.0);
                }
                let left_imp = impurity_weighted(lc, lw, cfg.criterion_entropy);
                let right_imp = impurity_weighted(&scratch.right_counts, rw, cfg.criterion_entropy);
                let gain = parent_imp - (lw / total_w) * left_imp - (rw / total_w) * right_imp;
                if gain > best_gain {
                    best_gain = gain;
                    best_feat = active_feats[ai];
                    best_thresh = active_thresh[ai];
                }
            }
        }
    } else {
        for fi in 0..scratch.feat_pool.len() {
            let f = scratch.feat_pool[fi];
            let col = &cfg.x_cols[f * n_samples..(f + 1) * n_samples];
            scratch.sorted.clear();
            for &(idx, w) in samples.iter() {
                scratch.sorted.push((col[idx], cfg.class_idx[idx], w));
            }
            scratch.sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            if scratch.sorted[n - 1].0 <= scratch.sorted[0].0 + 1e-12 {
                continue;
            }

            scratch.left_counts.fill(0.0);
            scratch.right_counts.copy_from_slice(&scratch.counts);
            let mut left_w = 0.0;

            for i in 0..(n - 1) {
                let (val_i, c_i, w_i) = scratch.sorted[i];
                scratch.left_counts[c_i] += w_i;
                scratch.right_counts[c_i] -= w_i;
                left_w += w_i;

                let left_n = i + 1;
                let right_n = n - left_n;
                if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                    continue;
                }
                let val_next = scratch.sorted[i + 1].0;
                if val_i + 1e-12 >= val_next {
                    continue;
                }

                let right_w = total_w - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let left_imp =
                    impurity_weighted(&scratch.left_counts, left_w, cfg.criterion_entropy);
                let right_imp =
                    impurity_weighted(&scratch.right_counts, right_w, cfg.criterion_entropy);
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
        return push_flat_clf_leaf_weighted(
            &scratch.counts,
            total_w,
            cfg.n_classes,
            nodes,
            probs_out,
        );
    }

    let mut lo = 0usize;
    let mut hi = n;
    if cfg.extra_trees {
        loop {
            while lo < hi && cfg.x_rows[samples[lo].0 * p + best_feat] <= best_thresh {
                lo += 1;
            }
            while lo < hi && cfg.x_rows[samples[hi - 1].0 * p + best_feat] > best_thresh {
                hi -= 1;
            }
            if lo >= hi {
                break;
            }
            hi -= 1;
            samples.swap(lo, hi);
            lo += 1;
        }
    } else {
        let best_col = &cfg.x_cols[best_feat * n_samples..(best_feat + 1) * n_samples];
        loop {
            while lo < hi && best_col[samples[lo].0] <= best_thresh {
                lo += 1;
            }
            while lo < hi && best_col[samples[hi - 1].0] > best_thresh {
                hi -= 1;
            }
            if lo >= hi {
                break;
            }
            hi -= 1;
            samples.swap(lo, hi);
            lo += 1;
        }
    }
    let left_len = lo;
    let right_len = n - left_len;

    if left_len < cfg.min_samples_leaf || right_len < cfg.min_samples_leaf {
        return push_flat_clf_leaf_weighted(
            &scratch.counts,
            total_w,
            cfg.n_classes,
            nodes,
            probs_out,
        );
    }

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += (total_w / cfg.total_tree_weight) * best_gain;
    }

    let node_idx = nodes.len();
    nodes.push(FlatClfNode {
        feature_or_offset: best_feat as u32,
        left: 0,
        right: 0,
        threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(left_len);
    let left_idx = build_clf_tree(
        cfg,
        left_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    let right_idx = build_clf_tree(
        cfg,
        right_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
}

// =========================================================================
// UNWEIGHTED U32/U64 FAST-PATH: EXTRATREES CLASSIFIER
// =========================================================================

trait EtClfSample: Copy + Default + Send + Sync {
    fn pack(idx: usize, cls: u32) -> Self;
    fn idx(self) -> usize;
    fn cls(self) -> usize;
}

impl EtClfSample for u32 {
    #[inline(always)]
    fn pack(idx: usize, cls: u32) -> Self {
        (idx as u32) | (cls << 24)
    }

    #[inline(always)]
    fn idx(self) -> usize {
        (self & 0x00FF_FFFF) as usize
    }

    #[inline(always)]
    fn cls(self) -> usize {
        (self >> 24) as usize
    }
}

impl EtClfSample for u64 {
    #[inline(always)]
    fn pack(idx: usize, cls: u32) -> Self {
        (idx as u64) | ((cls as u64) << 32)
    }

    #[inline(always)]
    fn idx(self) -> usize {
        (self as u32) as usize
    }

    #[inline(always)]
    fn cls(self) -> usize {
        (self >> 32) as usize
    }
}

#[derive(Clone, Copy)]
struct EtClfCfg<'a> {
    criterion_entropy: bool,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    n_classes: usize,
    n_samples: usize,
    n_features: usize,
    total_tree_weight: f64,
    x_rows: &'a [f64],
    class_idx: &'a [u32],
    root_min: &'a [f64],
    root_max: &'a [f64],
    root_counts: &'a [u32],
}

struct EtClfScratch<S: EtClfSample> {
    feat_pool: Vec<usize>,
    et_min: Vec<f64>,
    et_max: Vec<f64>,
    et_active_feats: Vec<usize>,
    et_active_thresh: Vec<f64>,
    et_active_slots: Vec<usize>,
    et_thresh_by_feat: Vec<f64>,
    et_left_counts: Vec<u32>,
    counts_stack: Vec<u32>,
    part_buf: Vec<S>,
    cand_vals: Vec<f64>,
}

impl<S: EtClfSample> EtClfScratch<S> {
    fn new(n_features: usize, max_feat: usize, n_classes: usize) -> Self {
        let thresh_by_feat_cap = if max_feat == n_features {
            n_features
        } else {
            0
        };
        Self {
            feat_pool: Vec::with_capacity(n_features),
            et_min: vec![0.0; max_feat],
            et_max: vec![0.0; max_feat],
            et_active_feats: Vec::with_capacity(max_feat),
            et_active_thresh: Vec::with_capacity(max_feat),
            et_active_slots: Vec::with_capacity(max_feat),
            et_thresh_by_feat: vec![0.0; thresh_by_feat_cap],
            et_left_counts: vec![0; max_feat * n_classes],
            counts_stack: vec![0; 32 * 2 * n_classes],
            part_buf: Vec::new(),
            cand_vals: Vec::new(),
        }
    }
}

#[inline]
fn draw_et_thresholds_with_slots(
    feat_pool: &[usize],
    min_v: &[f64],
    max_v: &[f64],
    rng: &mut SplitMix64,
    active_feats: &mut Vec<usize>,
    active_thresh: &mut Vec<f64>,
    active_slots: &mut Vec<usize>,
) {
    active_feats.clear();
    active_thresh.clear();
    active_slots.clear();
    for (fi, &f) in feat_pool.iter().enumerate() {
        let mn = min_v[fi];
        let mx = max_v[fi];
        if mx <= mn + 1e-12 {
            continue;
        }
        let thresh = mn + rng.next_f64() * (mx - mn);
        let thresh = if thresh >= mx {
            0.5 * (mn + mx)
        } else {
            thresh
        };
        active_feats.push(f);
        active_thresh.push(thresh);
        active_slots.push(fi);
    }
}

#[inline]
fn gather_et_cand_minmax<S: EtClfSample>(
    x_rows: &[f64],
    p: usize,
    samples: &[S],
    feat_pool: &[usize],
    min_v: &mut [f64],
    max_v: &mut [f64],
    cand: &mut [f64],
) {
    let n = samples.len();
    let k = feat_pool.len();
    if k == 4 {
        let f0 = feat_pool[0];
        let f1 = feat_pool[1];
        let f2 = feat_pool[2];
        let f3 = feat_pool[3];
        let mut mn0 = f64::INFINITY;
        let mut mx0 = f64::NEG_INFINITY;
        let mut mn1 = f64::INFINITY;
        let mut mx1 = f64::NEG_INFINITY;
        let mut mn2 = f64::INFINITY;
        let mut mx2 = f64::NEG_INFINITY;
        let mut mn3 = f64::INFINITY;
        let mut mx3 = f64::NEG_INFINITY;
        let (c0, rest) = cand.split_at_mut(n);
        let (c1, rest) = rest.split_at_mut(n);
        let (c2, c3) = rest.split_at_mut(n);
        for (i, &s) in samples.iter().enumerate() {
            let base = s.idx() * p;
            let row = &x_rows[base..base + p];
            let v0 = row[f0];
            let v1 = row[f1];
            let v2 = row[f2];
            let v3 = row[f3];
            if v0 < mn0 {
                mn0 = v0;
            }
            if v0 > mx0 {
                mx0 = v0;
            }
            if v1 < mn1 {
                mn1 = v1;
            }
            if v1 > mx1 {
                mx1 = v1;
            }
            if v2 < mn2 {
                mn2 = v2;
            }
            if v2 > mx2 {
                mx2 = v2;
            }
            if v3 < mn3 {
                mn3 = v3;
            }
            if v3 > mx3 {
                mx3 = v3;
            }
            c0[i] = v0;
            c1[i] = v1;
            c2[i] = v2;
            c3[i] = v3;
        }
        min_v[0] = mn0;
        max_v[0] = mx0;
        min_v[1] = mn1;
        max_v[1] = mx1;
        min_v[2] = mn2;
        max_v[2] = mx2;
        min_v[3] = mn3;
        max_v[3] = mx3;
    } else {
        min_v[..k].fill(f64::INFINITY);
        max_v[..k].fill(f64::NEG_INFINITY);
        for (i, &s) in samples.iter().enumerate() {
            let base = s.idx() * p;
            let row = &x_rows[base..base + p];
            for fi in 0..k {
                let v = row[feat_pool[fi]];
                if v < min_v[fi] {
                    min_v[fi] = v;
                }
                if v > max_v[fi] {
                    max_v[fi] = v;
                }
                cand[fi * n + i] = v;
            }
        }
    }
}

#[inline]
fn count_left_col_packed<S: EtClfSample>(
    col: &[f64],
    samples: &[S],
    thresh: f64,
    n_classes: usize,
    lc: &mut [u32],
) {
    if n_classes == 2 {
        let mut c0 = 0u32;
        let mut c1 = 0u32;
        for (&val, &s) in col.iter().zip(samples.iter()) {
            let go_left = u32::from(val <= thresh);
            let cls = s.cls() as u32;
            c1 += go_left & cls;
            c0 += go_left & (cls ^ 1);
        }
        lc[0] = c0;
        lc[1] = c1;
    } else {
        let len = col.len();
        let mut i = 0usize;
        while i + 4 <= len {
            let g0 = u32::from(col[i] <= thresh);
            let g1 = u32::from(col[i + 1] <= thresh);
            let g2 = u32::from(col[i + 2] <= thresh);
            let g3 = u32::from(col[i + 3] <= thresh);
            lc[samples[i].cls()] += g0;
            lc[samples[i + 1].cls()] += g1;
            lc[samples[i + 2].cls()] += g2;
            lc[samples[i + 3].cls()] += g3;
            i += 4;
        }
        while i < len {
            lc[samples[i].cls()] += u32::from(col[i] <= thresh);
            i += 1;
        }
    }
}

#[inline]
fn partition_packed_with_vals<S: EtClfSample>(
    col: &[f64],
    best_thresh: f64,
    left_len: usize,
    samples: &mut [S],
    part_buf: &mut Vec<S>,
) {
    let n = samples.len();
    if part_buf.len() < n {
        part_buf.resize(n, S::default());
    }
    let buf = &mut part_buf[..n];
    let last = n.saturating_sub(1);
    let mut l_pos = 0usize;
    let mut r_pos = left_len;
    for (&val, &s) in col.iter().zip(samples.iter()) {
        let go_left = val <= best_thresh;
        let dst = if go_left { l_pos } else { r_pos };
        buf[dst.min(last)] = s;
        l_pos += usize::from(go_left);
        r_pos += usize::from(!go_left);
    }
    samples.copy_from_slice(buf);
}

#[inline]
fn partition_packed_from_x<S: EtClfSample>(
    x_rows: &[f64],
    p: usize,
    best_feat: usize,
    best_thresh: f64,
    left_len: usize,
    samples: &mut [S],
    part_buf: &mut Vec<S>,
) {
    let n = samples.len();
    if part_buf.len() < n {
        part_buf.resize(n, S::default());
    }
    let buf = &mut part_buf[..n];
    let last = n.saturating_sub(1);
    let mut l_pos = 0usize;
    let mut r_pos = left_len;
    for &s in samples.iter() {
        let go_left = x_rows[s.idx() * p + best_feat] <= best_thresh;
        let dst = if go_left { l_pos } else { r_pos };
        buf[dst.min(last)] = s;
        l_pos += usize::from(go_left);
        r_pos += usize::from(!go_left);
    }
    samples.copy_from_slice(buf);
}

#[inline]
fn clf_impurity_u32(counts: &[u32], n: usize, entropy: bool) -> (f64, u64) {
    if n == 0 {
        return (0.0, 0);
    }
    let nf = n as f64;
    if entropy {
        let mut sum_c_log = 0.0;
        for &c in counts {
            if c > 0 {
                let cf = c as f64;
                sum_c_log += cf * cf.log2();
            }
        }
        (nf.log2() - sum_c_log / nf, 0)
    } else {
        let mut sum_sq = 0u64;
        for &c in counts {
            let cu = c as u64;
            sum_sq += cu * cu;
        }
        (1.0 - (sum_sq as f64) / (nf * nf), sum_sq)
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_et_clf_splits_u32(
    active_feats: &[usize],
    active_thresh: &[f64],
    left_counts: &[u32],
    col_stride: usize,
    class_stride: usize,
    dense_by_feat: bool,
    counts: &[u32],
    n: usize,
    min_samples_leaf: usize,
    criterion_entropy: bool,
    parent_imp: f64,
    parent_sum_sq: u64,
) -> Option<(usize, f64, f64, usize, usize)> {
    let n_classes = counts.len();
    let nf = n as f64;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;
    let mut best_col_idx = 0usize;
    let mut best_left_n = 0usize;
    let mut found = false;

    if criterion_entropy {
        let mut best_gain = 0.0;
        for (ai, (&feat, &thresh)) in active_feats.iter().zip(active_thresh.iter()).enumerate() {
            let col_idx = if dense_by_feat { feat } else { ai };
            let col_base = col_idx * col_stride;
            let mut ln = 0u32;
            let mut l_ent_sum = 0.0;
            let mut r_ent_sum = 0.0;
            for c in 0..n_classes {
                let lc = left_counts[col_base + c * class_stride];
                let rc = counts[c] - lc;
                ln += lc;
                if lc > 0 {
                    let lf = lc as f64;
                    l_ent_sum += lf * lf.log2();
                }
                if rc > 0 {
                    let rf = rc as f64;
                    r_ent_sum += rf * rf.log2();
                }
            }
            let lnu = ln as usize;
            let rnu = n - lnu;
            if lnu < min_samples_leaf || rnu < min_samples_leaf {
                continue;
            }
            let lnf = lnu as f64;
            let rnf = rnu as f64;
            let child_ent = (lnf * lnf.log2() - l_ent_sum + rnf * rnf.log2() - r_ent_sum) / nf;
            let gain = parent_imp - child_ent;
            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_thresh = thresh;
                best_col_idx = col_idx;
                best_left_n = lnu;
                found = true;
            }
        }
        if !found || best_gain <= 1e-12 {
            return None;
        }
        Some((best_feat, best_thresh, best_gain, best_col_idx, best_left_n))
    } else {
        let mut best_score = f64::NEG_INFINITY;
        for (ai, (&feat, &thresh)) in active_feats.iter().zip(active_thresh.iter()).enumerate() {
            let col_idx = if dense_by_feat { feat } else { ai };
            let col_base = col_idx * col_stride;
            let mut ln = 0u64;
            let mut l_sum_sq = 0u64;
            let mut r_sum_sq = 0u64;
            for c in 0..n_classes {
                let lc = left_counts[col_base + c * class_stride] as u64;
                let rc = (counts[c] as u64) - lc;
                ln += lc;
                l_sum_sq += lc * lc;
                r_sum_sq += rc * rc;
            }
            let lnu = ln as usize;
            let rnu = n - lnu;
            if lnu < min_samples_leaf || rnu < min_samples_leaf {
                continue;
            }
            let score = (l_sum_sq as f64) / (lnu as f64) + (r_sum_sq as f64) / (rnu as f64);
            if score > best_score {
                best_score = score;
                best_feat = feat;
                best_thresh = thresh;
                best_col_idx = col_idx;
                best_left_n = lnu;
                found = true;
            }
        }
        if !found {
            return None;
        }
        let best_gain = (best_score - (parent_sum_sq as f64) / nf) / nf;
        if best_gain <= 1e-12 {
            return None;
        }
        Some((best_feat, best_thresh, best_gain, best_col_idx, best_left_n))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_clf_tree_root_et_u32<S: EtClfSample>(
    cfg: &EtClfCfg<'_>,
    samples: &mut [S],
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut EtClfScratch<S>,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let n = cfg.n_samples;
    let p = cfg.n_features;
    let n_classes = cfg.n_classes;
    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| d == 0)
    {
        return push_flat_clf_leaf_u32(cfg.root_counts, n, n_classes, nodes, probs_out);
    }

    let (parent_imp, parent_sum_sq) = clf_impurity_u32(cfg.root_counts, n, cfg.criterion_entropy);
    if parent_imp <= 1e-12 {
        return push_flat_clf_leaf_u32(cfg.root_counts, n, n_classes, nodes, probs_out);
    }

    sample_feature_subset(&mut scratch.feat_pool, p, cfg.max_features, rng);
    draw_et_thresholds(
        &scratch.feat_pool,
        cfg.root_min,
        cfg.root_max,
        false,
        rng,
        &mut scratch.et_active_feats,
        &mut scratch.et_active_thresh,
    );

    let m = scratch.et_active_feats.len();
    if m == 0 {
        return push_flat_clf_leaf_u32(cfg.root_counts, n, n_classes, nodes, probs_out);
    }

    let use_cand = n >= 128 && m <= 8;
    let dense_by_feat = m == p && !use_cand;
    let stride = if dense_by_feat { p } else { m };
    let left_counts = &mut scratch.et_left_counts[..n_classes * stride];
    left_counts.fill(0);

    if use_cand {
        let cand_len = m * n;
        if scratch.cand_vals.len() < cand_len {
            scratch.cand_vals.resize(cand_len, 0.0);
        }
        let cand = &mut scratch.cand_vals[..cand_len];
        let active_feats = &scratch.et_active_feats[..m];
        let active_thresh = &scratch.et_active_thresh[..m];
        if m == 4 {
            let f0 = active_feats[0];
            let f1 = active_feats[1];
            let f2 = active_feats[2];
            let f3 = active_feats[3];
            let t0 = active_thresh[0];
            let t1 = active_thresh[1];
            let t2 = active_thresh[2];
            let t3 = active_thresh[3];
            let (c0, rest) = cand.split_at_mut(n);
            let (c1, rest) = rest.split_at_mut(n);
            let (c2, c3) = rest.split_at_mut(n);
            for (i, (row, &c)) in cfg
                .x_rows
                .chunks_exact(p)
                .zip(cfg.class_idx.iter())
                .enumerate()
            {
                let cu = c as usize;
                let lc_c = &mut left_counts[cu * 4..cu * 4 + 4];
                let v0 = row[f0];
                let v1 = row[f1];
                let v2 = row[f2];
                let v3 = row[f3];
                lc_c[0] += u32::from(v0 <= t0);
                lc_c[1] += u32::from(v1 <= t1);
                lc_c[2] += u32::from(v2 <= t2);
                lc_c[3] += u32::from(v3 <= t3);
                c0[i] = v0;
                c1[i] = v1;
                c2[i] = v2;
                c3[i] = v3;
            }
        } else {
            for (i, (row, &c)) in cfg
                .x_rows
                .chunks_exact(p)
                .zip(cfg.class_idx.iter())
                .enumerate()
            {
                let cu = c as usize;
                let lc_c = &mut left_counts[cu * m..cu * m + m];
                for ai in 0..m {
                    let v = row[active_feats[ai]];
                    lc_c[ai] += u32::from(v <= active_thresh[ai]);
                    cand[ai * n + i] = v;
                }
            }
        }
    } else if dense_by_feat {
        let thresh_s = &mut scratch.et_thresh_by_feat[..p];
        for (&f, &t) in scratch
            .et_active_feats
            .iter()
            .zip(scratch.et_active_thresh.iter())
        {
            thresh_s[f] = t;
        }
        for (row, &c) in cfg.x_rows.chunks_exact(p).zip(cfg.class_idx.iter()) {
            let cu = c as usize;
            let lc_c = &mut left_counts[cu * p..cu * p + p];
            for f in 0..p {
                lc_c[f] += u32::from(row[f] <= thresh_s[f]);
            }
        }
    } else {
        let active_feats = &scratch.et_active_feats[..m];
        let active_thresh = &scratch.et_active_thresh[..m];
        for (row, &c) in cfg.x_rows.chunks_exact(p).zip(cfg.class_idx.iter()) {
            let cu = c as usize;
            let lc_c = &mut left_counts[cu * m..cu * m + m];
            for ai in 0..m {
                lc_c[ai] += u32::from(row[active_feats[ai]] <= active_thresh[ai]);
            }
        }
    }

    let Some((best_feat, best_thresh, best_gain, best_col_idx, best_left_n)) =
        eval_et_clf_splits_u32(
            &scratch.et_active_feats[..m],
            &scratch.et_active_thresh[..m],
            left_counts,
            1,
            stride,
            dense_by_feat,
            cfg.root_counts,
            n,
            cfg.min_samples_leaf,
            cfg.criterion_entropy,
            parent_imp,
            parent_sum_sq,
        )
    else {
        return push_flat_clf_leaf_u32(cfg.root_counts, n, n_classes, nodes, probs_out);
    };

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += ((n as f64) / cfg.total_tree_weight) * best_gain;
    }

    let need = 4 * n_classes;
    if scratch.counts_stack.len() < need {
        scratch.counts_stack.resize(need, 0);
    }
    let r_off = n_classes;
    let l_off = 2 * n_classes;
    for c in 0..n_classes {
        let lc = left_counts[c * stride + best_col_idx];
        scratch.counts_stack[l_off + c] = lc;
        scratch.counts_stack[r_off + c] = cfg.root_counts[c] - lc;
    }

    let last = n.saturating_sub(1);
    let mut l_pos = 0usize;
    let mut r_pos = best_left_n;
    if use_cand {
        let best_col = &scratch.cand_vals[best_col_idx * n..(best_col_idx + 1) * n];
        for (idx, (&val, &c)) in best_col.iter().zip(cfg.class_idx.iter()).enumerate() {
            let go_left = val <= best_thresh;
            let dst = if go_left { l_pos } else { r_pos };
            samples[dst.min(last)] = S::pack(idx, c);
            l_pos += usize::from(go_left);
            r_pos += usize::from(!go_left);
        }
    } else {
        for (idx, (row, &c)) in cfg
            .x_rows
            .chunks_exact(p)
            .zip(cfg.class_idx.iter())
            .enumerate()
        {
            let go_left = row[best_feat] <= best_thresh;
            let dst = if go_left { l_pos } else { r_pos };
            samples[dst.min(last)] = S::pack(idx, c);
            l_pos += usize::from(go_left);
            r_pos += usize::from(!go_left);
        }
    }

    let node_idx = nodes.len();
    nodes.push(FlatClfNode {
        feature_or_offset: best_feat as u32,
        left: 0,
        right: 0,
        threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(best_left_n);
    let left_idx = build_clf_tree_et_u32(
        cfg,
        left_samples,
        1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    scratch
        .counts_stack
        .copy_within(r_off..r_off + n_classes, l_off);
    let right_idx = build_clf_tree_et_u32(
        cfg,
        right_samples,
        1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
}

#[allow(clippy::too_many_arguments)]
fn build_clf_tree_et_u32<S: EtClfSample>(
    cfg: &EtClfCfg<'_>,
    samples: &mut [S],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut EtClfScratch<S>,
    nodes: &mut Vec<FlatClfNode>,
    probs_out: &mut Vec<f64>,
) -> usize {
    let n = samples.len();
    let n_classes = cfg.n_classes;
    let cur_off = depth * 2 * n_classes;
    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| depth >= d)
    {
        return push_flat_clf_leaf_u32(
            &scratch.counts_stack[cur_off..cur_off + n_classes],
            n,
            n_classes,
            nodes,
            probs_out,
        );
    }

    let (parent_imp, parent_sum_sq) = clf_impurity_u32(
        &scratch.counts_stack[cur_off..cur_off + n_classes],
        n,
        cfg.criterion_entropy,
    );
    if parent_imp <= 1e-12 {
        return push_flat_clf_leaf_u32(
            &scratch.counts_stack[cur_off..cur_off + n_classes],
            n,
            n_classes,
            nodes,
            probs_out,
        );
    }

    let p = cfg.n_features;
    sample_feature_subset(&mut scratch.feat_pool, p, cfg.max_features, rng);
    let k = scratch.feat_pool.len();
    let use_cand = n >= 128 && k <= 8;

    let (best_feat, best_thresh, best_gain, best_col_idx, best_left_n, col_stride, class_stride) =
        if use_cand {
            let cand_len = k * n;
            if scratch.cand_vals.len() < cand_len {
                scratch.cand_vals.resize(cand_len, 0.0);
            }
            gather_et_cand_minmax(
                cfg.x_rows,
                p,
                samples,
                &scratch.feat_pool[..k],
                &mut scratch.et_min[..k],
                &mut scratch.et_max[..k],
                &mut scratch.cand_vals[..cand_len],
            );
            draw_et_thresholds_with_slots(
                &scratch.feat_pool[..k],
                &scratch.et_min[..k],
                &scratch.et_max[..k],
                rng,
                &mut scratch.et_active_feats,
                &mut scratch.et_active_thresh,
                &mut scratch.et_active_slots,
            );
            let m = scratch.et_active_feats.len();
            if m == 0 {
                return push_flat_clf_leaf_u32(
                    &scratch.counts_stack[cur_off..cur_off + n_classes],
                    n,
                    n_classes,
                    nodes,
                    probs_out,
                );
            }
            let left_counts = &mut scratch.et_left_counts[..m * n_classes];
            left_counts.fill(0);
            for (ai, (&thresh, &slot)) in scratch.et_active_thresh[..m]
                .iter()
                .zip(scratch.et_active_slots[..m].iter())
                .enumerate()
            {
                let col = &scratch.cand_vals[slot * n..(slot + 1) * n];
                let lc = &mut left_counts[ai * n_classes..(ai + 1) * n_classes];
                count_left_col_packed(col, samples, thresh, n_classes, lc);
            }
            let Some((bf, bt, bg, bci, bln)) = eval_et_clf_splits_u32(
                &scratch.et_active_feats[..m],
                &scratch.et_active_thresh[..m],
                left_counts,
                n_classes,
                1,
                false,
                &scratch.counts_stack[cur_off..cur_off + n_classes],
                n,
                cfg.min_samples_leaf,
                cfg.criterion_entropy,
                parent_imp,
                parent_sum_sq,
            ) else {
                return push_flat_clf_leaf_u32(
                    &scratch.counts_stack[cur_off..cur_off + n_classes],
                    n,
                    n_classes,
                    nodes,
                    probs_out,
                );
            };
            (bf, bt, bg, bci, bln, n_classes, 1)
        } else {
            let all_feats = k == p;
            if all_feats {
                let min_v = &mut scratch.et_min[..p];
                let max_v = &mut scratch.et_max[..p];
                min_v.fill(f64::INFINITY);
                max_v.fill(f64::NEG_INFINITY);
                for &s in samples.iter() {
                    let uidx = s.idx();
                    let row = &cfg.x_rows[uidx * p..uidx * p + p];
                    for f in 0..p {
                        let v = row[f];
                        if v < min_v[f] {
                            min_v[f] = v;
                        }
                        if v > max_v[f] {
                            max_v[f] = v;
                        }
                    }
                }
                draw_et_thresholds(
                    &scratch.feat_pool[..k],
                    min_v,
                    max_v,
                    false,
                    rng,
                    &mut scratch.et_active_feats,
                    &mut scratch.et_active_thresh,
                );
            } else {
                let feat_pool = &scratch.feat_pool[..k];
                let min_v = &mut scratch.et_min[..k];
                let max_v = &mut scratch.et_max[..k];
                min_v.fill(f64::INFINITY);
                max_v.fill(f64::NEG_INFINITY);
                for &s in samples.iter() {
                    let uidx = s.idx();
                    let row = &cfg.x_rows[uidx * p..uidx * p + p];
                    for fi in 0..k {
                        let v = row[feat_pool[fi]];
                        if v < min_v[fi] {
                            min_v[fi] = v;
                        }
                        if v > max_v[fi] {
                            max_v[fi] = v;
                        }
                    }
                }
                draw_et_thresholds(
                    feat_pool,
                    min_v,
                    max_v,
                    true,
                    rng,
                    &mut scratch.et_active_feats,
                    &mut scratch.et_active_thresh,
                );
            }

            let m = scratch.et_active_feats.len();
            if m == 0 {
                return push_flat_clf_leaf_u32(
                    &scratch.counts_stack[cur_off..cur_off + n_classes],
                    n,
                    n_classes,
                    nodes,
                    probs_out,
                );
            }

            let dense_by_feat = m == p;
            let stride = if dense_by_feat { p } else { m };
            let left_counts = &mut scratch.et_left_counts[..n_classes * stride];
            left_counts.fill(0);

            if dense_by_feat {
                let thresh_s = &mut scratch.et_thresh_by_feat[..p];
                for (&f, &t) in scratch
                    .et_active_feats
                    .iter()
                    .zip(scratch.et_active_thresh.iter())
                {
                    thresh_s[f] = t;
                }
                for &s in samples.iter() {
                    let uidx = s.idx();
                    let row = &cfg.x_rows[uidx * p..uidx * p + p];
                    let cu = s.cls();
                    let lc_c = &mut left_counts[cu * p..cu * p + p];
                    for f in 0..p {
                        lc_c[f] += u32::from(row[f] <= thresh_s[f]);
                    }
                }
            } else {
                let active_feats = &scratch.et_active_feats[..m];
                let active_thresh = &scratch.et_active_thresh[..m];
                for &s in samples.iter() {
                    let uidx = s.idx();
                    let row = &cfg.x_rows[uidx * p..uidx * p + p];
                    let cu = s.cls();
                    let lc_c = &mut left_counts[cu * m..cu * m + m];
                    for ai in 0..m {
                        lc_c[ai] += u32::from(row[active_feats[ai]] <= active_thresh[ai]);
                    }
                }
            }

            let Some((bf, bt, bg, bci, bln)) = eval_et_clf_splits_u32(
                &scratch.et_active_feats[..m],
                &scratch.et_active_thresh[..m],
                left_counts,
                1,
                stride,
                dense_by_feat,
                &scratch.counts_stack[cur_off..cur_off + n_classes],
                n,
                cfg.min_samples_leaf,
                cfg.criterion_entropy,
                parent_imp,
                parent_sum_sq,
            ) else {
                return push_flat_clf_leaf_u32(
                    &scratch.counts_stack[cur_off..cur_off + n_classes],
                    n,
                    n_classes,
                    nodes,
                    probs_out,
                );
            };
            (bf, bt, bg, bci, bln, 1, stride)
        };

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += ((n as f64) / cfg.total_tree_weight) * best_gain;
    }

    let need = (depth + 2) * 2 * n_classes;
    if scratch.counts_stack.len() < need {
        scratch.counts_stack.resize(need, 0);
    }
    let r_off = cur_off + n_classes;
    let l_off = (depth + 1) * 2 * n_classes;
    let col_base = best_col_idx * col_stride;
    for c in 0..n_classes {
        let lc = scratch.et_left_counts[col_base + c * class_stride];
        let tot = scratch.counts_stack[cur_off + c];
        scratch.counts_stack[l_off + c] = lc;
        scratch.counts_stack[r_off + c] = tot - lc;
    }

    if use_cand {
        let best_slot = scratch.et_active_slots[best_col_idx];
        let best_col = &scratch.cand_vals[best_slot * n..(best_slot + 1) * n];
        partition_packed_with_vals(
            best_col,
            best_thresh,
            best_left_n,
            samples,
            &mut scratch.part_buf,
        );
    } else {
        partition_packed_from_x(
            cfg.x_rows,
            p,
            best_feat,
            best_thresh,
            best_left_n,
            samples,
            &mut scratch.part_buf,
        );
    }

    let node_idx = nodes.len();
    nodes.push(FlatClfNode {
        feature_or_offset: best_feat as u32,
        left: 0,
        right: 0,
        threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(best_left_n);
    let left_idx = build_clf_tree_et_u32(
        cfg,
        left_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    scratch
        .counts_stack
        .copy_within(r_off..r_off + n_classes, l_off);
    let right_idx = build_clf_tree_et_u32(
        cfg,
        right_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
        probs_out,
    );
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
}

fn run_et_clf_trees_packed<S: EtClfSample>(
    cfg: &EtClfCfg<'_>,
    n_estimators: usize,
    base_seed: u64,
) -> Vec<(Vec<FlatClfNode>, Vec<f64>, Vec<f64>)> {
    (0..n_estimators)
        .into_par_iter()
        .map_init(
            || {
                (
                    EtClfScratch::<S>::new(cfg.n_features, cfg.max_features, cfg.n_classes),
                    vec![S::default(); cfg.n_samples],
                )
            },
            |(scratch, root_samples), tree_idx| {
                let seed = base_seed
                    .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ 0xA5A5_5A5A_1234_5678;
                let mut rng = SplitMix64::new(seed);
                let mut imp = vec![0.0; cfg.n_features];
                let mut t_nodes = Vec::with_capacity(256);
                let mut t_probs = Vec::with_capacity(128 * cfg.n_classes);
                build_clf_tree_root_et_u32(
                    cfg,
                    root_samples,
                    &mut rng,
                    &mut imp,
                    scratch,
                    &mut t_nodes,
                    &mut t_probs,
                );
                (t_nodes, t_probs, imp)
            },
        )
        .collect()
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
    n_samples: usize,
    n_features: usize,
    total_tree_weight: f64,
    x_rows: &'a [f64],
    x_cols: &'a [f64],
    y: &'a [f64],
}

struct RegTreeScratch {
    feat_pool: Vec<usize>,
    sorted: Vec<(f64, f64, f64)>,
    et_min: Vec<f64>,
    et_max: Vec<f64>,
    et_active_feats: Vec<usize>,
    et_active_thresh: Vec<f64>,
    et_left_sum: Vec<f64>,
    et_left_w: Vec<f64>,
    et_left_n: Vec<usize>,
}

impl RegTreeScratch {
    fn new(n_features: usize, max_feat: usize, sorted_cap: usize, extra_trees: bool) -> Self {
        let et_cap = if extra_trees { max_feat } else { 0 };
        Self {
            feat_pool: Vec::with_capacity(n_features),
            sorted: Vec::with_capacity(sorted_cap),
            et_min: vec![0.0; et_cap],
            et_max: vec![0.0; et_cap],
            et_active_feats: Vec::with_capacity(et_cap),
            et_active_thresh: Vec::with_capacity(et_cap),
            et_left_sum: vec![0.0; et_cap],
            et_left_w: vec![0.0; et_cap],
            et_left_n: vec![0; et_cap],
        }
    }
}

fn build_reg_tree(
    cfg: &RegBuildCfg<'_>,
    samples: &mut [(usize, f64)],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut RegTreeScratch,
    nodes: &mut Vec<FlatRegNode>,
) -> usize {
    let n = samples.len();
    let mut sum_y = 0.0;
    let mut sum_y2 = 0.0;
    let mut total_w = 0.0;
    for &(idx, w) in samples.iter() {
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
        return push_flat_reg_leaf(mean_val, nodes);
    }

    let parent_var = (sum_y2 / total_w) - mean_val * mean_val;
    if parent_var <= 1e-12 {
        return push_flat_reg_leaf(mean_val, nodes);
    }
    let parent_score = (sum_y * sum_y) / total_w;

    sample_feature_subset(
        &mut scratch.feat_pool,
        cfg.n_features,
        cfg.max_features,
        rng,
    );

    let mut best_score_gain = 0.0;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;
    let n_samples = cfg.n_samples;
    let p = cfg.n_features;

    if cfg.extra_trees {
        let k = scratch.feat_pool.len();
        let feat_pool = &scratch.feat_pool[..k];
        let min_v = &mut scratch.et_min[..k];
        let max_v = &mut scratch.et_max[..k];
        min_v.fill(f64::INFINITY);
        max_v.fill(f64::NEG_INFINITY);

        for &(idx, _) in samples.iter() {
            let row = &cfg.x_rows[idx * p..(idx + 1) * p];
            for fi in 0..k {
                let v = row[feat_pool[fi]];
                if v < min_v[fi] {
                    min_v[fi] = v;
                }
                if v > max_v[fi] {
                    max_v[fi] = v;
                }
            }
        }

        draw_et_thresholds(
            feat_pool,
            min_v,
            max_v,
            true,
            rng,
            &mut scratch.et_active_feats,
            &mut scratch.et_active_thresh,
        );

        let m = scratch.et_active_feats.len();
        if m > 0 {
            let active_feats = &scratch.et_active_feats[..m];
            let active_thresh = &scratch.et_active_thresh[..m];
            let left_sum = &mut scratch.et_left_sum[..m];
            let left_w = &mut scratch.et_left_w[..m];
            let left_n = &mut scratch.et_left_n[..m];
            left_sum.fill(0.0);
            left_w.fill(0.0);
            left_n.fill(0);

            for &(idx, w) in samples.iter() {
                let row = &cfg.x_rows[idx * p..(idx + 1) * p];
                let wy = w * cfg.y[idx];
                for ai in 0..m {
                    let go_left = row[active_feats[ai]] <= active_thresh[ai];
                    let mask = if go_left { 1.0 } else { 0.0 };
                    left_sum[ai] += mask * wy;
                    left_w[ai] += mask * w;
                    left_n[ai] += usize::from(go_left);
                }
            }

            for ai in 0..m {
                let ln = left_n[ai];
                let rn = n - ln;
                if ln < cfg.min_samples_leaf || rn < cfg.min_samples_leaf {
                    continue;
                }
                let lw = left_w[ai];
                let rw = total_w - lw;
                if lw <= 0.0 || rw <= 0.0 {
                    continue;
                }
                let ls = left_sum[ai];
                let rs = sum_y - ls;
                let gain = (ls * ls) / lw + (rs * rs) / rw - parent_score;
                if gain > best_score_gain {
                    best_score_gain = gain;
                    best_feat = active_feats[ai];
                    best_thresh = active_thresh[ai];
                }
            }
        }
    } else {
        for fi in 0..scratch.feat_pool.len() {
            let f = scratch.feat_pool[fi];
            let col = &cfg.x_cols[f * n_samples..(f + 1) * n_samples];
            scratch.sorted.clear();
            for &(idx, w) in samples.iter() {
                scratch.sorted.push((col[idx], cfg.y[idx], w));
            }
            scratch.sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            if scratch.sorted[n - 1].0 <= scratch.sorted[0].0 + 1e-12 {
                continue;
            }

            let mut left_sum = 0.0;
            let mut left_w = 0.0;
            for i in 0..(n - 1) {
                let (val_i, yi, wi) = scratch.sorted[i];
                left_sum += wi * yi;
                left_w += wi;

                let left_n = i + 1;
                let right_n = n - left_n;
                if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                    continue;
                }
                let val_next = scratch.sorted[i + 1].0;
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
        return push_flat_reg_leaf(mean_val, nodes);
    }

    let mut lo = 0usize;
    let mut hi = n;
    if cfg.extra_trees {
        loop {
            while lo < hi && cfg.x_rows[samples[lo].0 * p + best_feat] <= best_thresh {
                lo += 1;
            }
            while lo < hi && cfg.x_rows[samples[hi - 1].0 * p + best_feat] > best_thresh {
                hi -= 1;
            }
            if lo >= hi {
                break;
            }
            hi -= 1;
            samples.swap(lo, hi);
            lo += 1;
        }
    } else {
        let best_col = &cfg.x_cols[best_feat * n_samples..(best_feat + 1) * n_samples];
        loop {
            while lo < hi && best_col[samples[lo].0] <= best_thresh {
                lo += 1;
            }
            while lo < hi && best_col[samples[hi - 1].0] > best_thresh {
                hi -= 1;
            }
            if lo >= hi {
                break;
            }
            hi -= 1;
            samples.swap(lo, hi);
            lo += 1;
        }
    }
    let left_len = lo;
    let right_len = n - left_len;

    if left_len < cfg.min_samples_leaf || right_len < cfg.min_samples_leaf {
        return push_flat_reg_leaf(mean_val, nodes);
    }

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += best_score_gain / cfg.total_tree_weight;
    }

    let node_idx = nodes.len();
    nodes.push(FlatRegNode {
        feature: best_feat as u32,
        left: 0,
        right: 0,
        value_or_threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(left_len);
    let left_idx = build_reg_tree(
        cfg,
        left_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
    );
    let right_idx = build_reg_tree(
        cfg,
        right_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
    );
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
}

// =========================================================================
// UNWEIGHTED U32 FAST-PATH: EXTRATREES REGRESSOR
// =========================================================================

#[derive(Clone, Copy)]
struct EtRegCfg<'a> {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    n_samples: usize,
    n_features: usize,
    total_tree_weight: f64,
    x_rows: &'a [f64],
    y: &'a [f64],
    root_min: &'a [f64],
    root_max: &'a [f64],
    root_sum_y: f64,
    root_sum_y2: f64,
}

struct EtRegScratch {
    feat_pool: Vec<usize>,
    et_min: Vec<f64>,
    et_max: Vec<f64>,
    et_active_feats: Vec<usize>,
    et_active_thresh: Vec<f64>,
    et_thresh_by_feat: Vec<f64>,
    et_left_sum: Vec<f64>,
    et_left_n: Vec<u32>,
    part_buf: Vec<u32>,
}

impl EtRegScratch {
    fn new(n_features: usize, max_feat: usize) -> Self {
        let thresh_by_feat_cap = if max_feat == n_features {
            n_features
        } else {
            0
        };
        Self {
            feat_pool: Vec::with_capacity(n_features),
            et_min: vec![0.0; max_feat],
            et_max: vec![0.0; max_feat],
            et_active_feats: Vec::with_capacity(max_feat),
            et_active_thresh: Vec::with_capacity(max_feat),
            et_thresh_by_feat: vec![0.0; thresh_by_feat_cap],
            et_left_sum: vec![0.0; max_feat],
            et_left_n: vec![0; max_feat],
            part_buf: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_et_reg_splits_u32(
    active_feats: &[usize],
    active_thresh: &[f64],
    left_sum: &[f64],
    left_n: &[u32],
    dense_by_feat: bool,
    sum_y: f64,
    n: usize,
    min_samples_leaf: usize,
    parent_score: f64,
) -> Option<(usize, f64, f64, usize)> {
    let mut best_score_gain = 0.0;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0;
    let mut best_left_n = 0usize;
    let mut found = false;

    for (ai, (&feat, &thresh)) in active_feats.iter().zip(active_thresh.iter()).enumerate() {
        let col_idx = if dense_by_feat { feat } else { ai };
        let ln = left_n[col_idx] as usize;
        let rn = n - ln;
        if ln < min_samples_leaf || rn < min_samples_leaf {
            continue;
        }
        let ls = left_sum[col_idx];
        let rs = sum_y - ls;
        let gain = (ls * ls) / (ln as f64) + (rs * rs) / (rn as f64) - parent_score;
        if gain > best_score_gain {
            best_score_gain = gain;
            best_feat = feat;
            best_thresh = thresh;
            best_left_n = ln;
            found = true;
        }
    }

    if !found || best_score_gain <= 1e-12 {
        None
    } else {
        Some((best_feat, best_thresh, best_score_gain, best_left_n))
    }
}

fn build_reg_tree_root_et_u32(
    cfg: &EtRegCfg<'_>,
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut EtRegScratch,
    nodes: &mut Vec<FlatRegNode>,
) -> usize {
    let n = cfg.n_samples;
    let p = cfg.n_features;
    let nf = n as f64;
    let mean_val = if n > 0 { cfg.root_sum_y / nf } else { 0.0 };
    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| d == 0)
    {
        return push_flat_reg_leaf(mean_val, nodes);
    }

    let parent_var = (cfg.root_sum_y2 / nf) - mean_val * mean_val;
    if parent_var <= 1e-12 {
        return push_flat_reg_leaf(mean_val, nodes);
    }
    let parent_score = (cfg.root_sum_y * cfg.root_sum_y) / nf;

    sample_feature_subset(&mut scratch.feat_pool, p, cfg.max_features, rng);
    draw_et_thresholds(
        &scratch.feat_pool,
        cfg.root_min,
        cfg.root_max,
        false,
        rng,
        &mut scratch.et_active_feats,
        &mut scratch.et_active_thresh,
    );

    let m = scratch.et_active_feats.len();
    if m == 0 {
        return push_flat_reg_leaf(mean_val, nodes);
    }

    let dense_by_feat = m == p;
    let left_sum = &mut scratch.et_left_sum[..m];
    let left_n = &mut scratch.et_left_n[..m];
    left_sum.fill(0.0);
    left_n.fill(0);

    if dense_by_feat {
        let thresh_s = &mut scratch.et_thresh_by_feat[..p];
        for (&f, &t) in scratch
            .et_active_feats
            .iter()
            .zip(scratch.et_active_thresh.iter())
        {
            thresh_s[f] = t;
        }
        for (row, &yi) in cfg.x_rows.chunks_exact(p).zip(cfg.y.iter()) {
            for f in 0..p {
                let go_left = row[f] <= thresh_s[f];
                left_sum[f] += if go_left { yi } else { 0.0 };
                left_n[f] += u32::from(go_left);
            }
        }
    } else {
        let active_feats = &scratch.et_active_feats[..m];
        let active_thresh = &scratch.et_active_thresh[..m];
        for (row, &yi) in cfg.x_rows.chunks_exact(p).zip(cfg.y.iter()) {
            for ai in 0..m {
                let go_left = row[active_feats[ai]] <= active_thresh[ai];
                left_sum[ai] += if go_left { yi } else { 0.0 };
                left_n[ai] += u32::from(go_left);
            }
        }
    }

    let Some((best_feat, best_thresh, best_score_gain, best_left_n)) = eval_et_reg_splits_u32(
        &scratch.et_active_feats[..m],
        &scratch.et_active_thresh[..m],
        left_sum,
        left_n,
        dense_by_feat,
        cfg.root_sum_y,
        n,
        cfg.min_samples_leaf,
        parent_score,
    ) else {
        return push_flat_reg_leaf(mean_val, nodes);
    };

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += best_score_gain / cfg.total_tree_weight;
    }

    let mut samples = vec![0u32; n];
    let last = n.saturating_sub(1);
    let mut l_pos = 0usize;
    let mut r_pos = best_left_n;
    for (idx, row) in cfg.x_rows.chunks_exact(p).enumerate() {
        let go_left = row[best_feat] <= best_thresh;
        let dst = if go_left { l_pos } else { r_pos };
        samples[dst.min(last)] = idx as u32;
        l_pos += usize::from(go_left);
        r_pos += usize::from(!go_left);
    }

    let node_idx = nodes.len();
    nodes.push(FlatRegNode {
        feature: best_feat as u32,
        left: 0,
        right: 0,
        value_or_threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(best_left_n);
    let left_idx = build_reg_tree_et_u32(cfg, left_samples, 1, rng, importances, scratch, nodes);
    let right_idx = build_reg_tree_et_u32(cfg, right_samples, 1, rng, importances, scratch, nodes);
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
}

fn build_reg_tree_et_u32(
    cfg: &EtRegCfg<'_>,
    samples: &mut [u32],
    depth: usize,
    rng: &mut SplitMix64,
    importances: &mut [f64],
    scratch: &mut EtRegScratch,
    nodes: &mut Vec<FlatRegNode>,
) -> usize {
    let n = samples.len();
    if n == 1 {
        return push_flat_reg_leaf(cfg.y[samples[0] as usize], nodes);
    }
    if n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|d| depth >= d)
    {
        let mut sum_y = 0.0;
        for &idx in samples.iter() {
            sum_y += cfg.y[idx as usize];
        }
        return push_flat_reg_leaf(sum_y / (n as f64), nodes);
    }

    let mut sum_y = 0.0;
    let mut sum_y2 = 0.0;
    for &idx in samples.iter() {
        let yi = cfg.y[idx as usize];
        sum_y += yi;
        sum_y2 += yi * yi;
    }
    let nf = n as f64;
    let mean_val = sum_y / nf;
    let parent_var = (sum_y2 / nf) - mean_val * mean_val;
    if parent_var <= 1e-12 {
        return push_flat_reg_leaf(mean_val, nodes);
    }
    let parent_score = (sum_y * sum_y) / nf;

    let p = cfg.n_features;
    sample_feature_subset(&mut scratch.feat_pool, p, cfg.max_features, rng);
    let k = scratch.feat_pool.len();
    let all_feats = k == p;

    if all_feats {
        let min_v = &mut scratch.et_min[..p];
        let max_v = &mut scratch.et_max[..p];
        min_v.fill(f64::INFINITY);
        max_v.fill(f64::NEG_INFINITY);
        for &idx in samples.iter() {
            let uidx = idx as usize;
            let row = &cfg.x_rows[uidx * p..uidx * p + p];
            for f in 0..p {
                let v = row[f];
                if v < min_v[f] {
                    min_v[f] = v;
                }
                if v > max_v[f] {
                    max_v[f] = v;
                }
            }
        }
        draw_et_thresholds(
            &scratch.feat_pool[..k],
            min_v,
            max_v,
            false,
            rng,
            &mut scratch.et_active_feats,
            &mut scratch.et_active_thresh,
        );
    } else {
        let feat_pool = &scratch.feat_pool[..k];
        let min_v = &mut scratch.et_min[..k];
        let max_v = &mut scratch.et_max[..k];
        min_v.fill(f64::INFINITY);
        max_v.fill(f64::NEG_INFINITY);
        for &idx in samples.iter() {
            let uidx = idx as usize;
            let row = &cfg.x_rows[uidx * p..uidx * p + p];
            for fi in 0..k {
                let v = row[feat_pool[fi]];
                if v < min_v[fi] {
                    min_v[fi] = v;
                }
                if v > max_v[fi] {
                    max_v[fi] = v;
                }
            }
        }
        draw_et_thresholds(
            feat_pool,
            min_v,
            max_v,
            true,
            rng,
            &mut scratch.et_active_feats,
            &mut scratch.et_active_thresh,
        );
    }

    let m = scratch.et_active_feats.len();
    if m == 0 {
        return push_flat_reg_leaf(mean_val, nodes);
    }

    let dense_by_feat = m == p;
    let left_sum = &mut scratch.et_left_sum[..m];
    let left_n = &mut scratch.et_left_n[..m];
    left_sum.fill(0.0);
    left_n.fill(0);

    if dense_by_feat {
        let thresh_s = &mut scratch.et_thresh_by_feat[..p];
        for (&f, &t) in scratch
            .et_active_feats
            .iter()
            .zip(scratch.et_active_thresh.iter())
        {
            thresh_s[f] = t;
        }
        for &idx in samples.iter() {
            let uidx = idx as usize;
            let row = &cfg.x_rows[uidx * p..uidx * p + p];
            let yi = cfg.y[uidx];
            for f in 0..p {
                let go_left = row[f] <= thresh_s[f];
                left_sum[f] += if go_left { yi } else { 0.0 };
                left_n[f] += u32::from(go_left);
            }
        }
    } else {
        let active_feats = &scratch.et_active_feats[..m];
        let active_thresh = &scratch.et_active_thresh[..m];
        for &idx in samples.iter() {
            let uidx = idx as usize;
            let row = &cfg.x_rows[uidx * p..uidx * p + p];
            let yi = cfg.y[uidx];
            for ai in 0..m {
                let go_left = row[active_feats[ai]] <= active_thresh[ai];
                left_sum[ai] += if go_left { yi } else { 0.0 };
                left_n[ai] += u32::from(go_left);
            }
        }
    }

    let Some((best_feat, best_thresh, best_score_gain, best_left_n)) = eval_et_reg_splits_u32(
        &scratch.et_active_feats[..m],
        &scratch.et_active_thresh[..m],
        left_sum,
        left_n,
        dense_by_feat,
        sum_y,
        n,
        cfg.min_samples_leaf,
        parent_score,
    ) else {
        return push_flat_reg_leaf(mean_val, nodes);
    };

    if cfg.total_tree_weight > 0.0 {
        importances[best_feat] += best_score_gain / cfg.total_tree_weight;
    }

    partition_u32_with_buf(
        cfg.x_rows,
        p,
        best_feat,
        best_thresh,
        best_left_n,
        samples,
        &mut scratch.part_buf,
    );

    let node_idx = nodes.len();
    nodes.push(FlatRegNode {
        feature: best_feat as u32,
        left: 0,
        right: 0,
        value_or_threshold: best_thresh,
    });
    let (left_samples, right_samples) = samples.split_at_mut(best_left_n);
    let left_idx = build_reg_tree_et_u32(
        cfg,
        left_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
    );
    let right_idx = build_reg_tree_et_u32(
        cfg,
        right_samples,
        depth + 1,
        rng,
        importances,
        scratch,
        nodes,
    );
    nodes[node_idx].left = left_idx as u32;
    nodes[node_idx].right = right_idx as u32;
    node_idx
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
    nodes: &mut Vec<FlatRegNode>,
) -> OchreResult<usize> {
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
        return Ok(push_flat_reg_leaf(leaf_val, nodes));
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
        return Ok(push_flat_reg_leaf(leaf_val, nodes));
    };
    if split.gain <= 1e-12 {
        return Ok(push_flat_reg_leaf(leaf_val, nodes));
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
        return Ok(push_flat_reg_leaf(leaf_val, nodes));
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

    let node_idx = nodes.len();
    nodes.push(FlatRegNode {
        feature: f as u32,
        left: 0,
        right: 0,
        value_or_threshold: split.threshold,
    });
    let left_node = build_reg_tree_n6(
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
        nodes,
    )?;
    let right_node = build_reg_tree_n6(
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
        nodes,
    )?;
    nodes[node_idx].left = left_node as u32;
    nodes[node_idx].right = right_node as u32;
    Ok(node_idx)
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
    flat_nodes: Vec<FlatClfNode>,
    flat_roots: Vec<usize>,
    flat_probs: Vec<f64>,
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
            flat_nodes: Vec::new(),
            flat_roots: Vec::new(),
            flat_probs: Vec::new(),
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
        let max_feat = resolve_max_features(self.max_features.as_deref(), x.cols, true)?;

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

        // Check backend availability; CPU/TPU uses Rayon parallel tree building
        let _backend = get_backend(&self.device);

        let base_seed = resolve_base_seed(self.random_state);
        let criterion_entropy = self.criterion == "entropy";
        let n_classes = classes.len();
        let n_samples = x.rows;
        let n_features = x.cols;

        let use_et_u32 = self.extra_trees
            && !self.bootstrap
            && sample_weight.is_none()
            && n_samples <= u32::MAX as usize;

        let (flat_nodes, flat_roots, flat_probs, all_imp) = if use_et_u32 {
            let mut root_counts = vec![0u32; n_classes];
            let class_idx_u32: Vec<u32> = y
                .iter()
                .map(|&val| {
                    let c = classes
                        .iter()
                        .position(|&cls| (cls - val).abs() <= 1e-12)
                        .unwrap_or(0);
                    root_counts[c] += 1;
                    c as u32
                })
                .collect();
            let (root_min, root_max) = compute_root_min_max(&x.data, n_samples, n_features);
            let total_tree_weight = n_samples as f64;
            let cfg = EtClfCfg {
                criterion_entropy,
                max_depth: self.max_depth,
                min_samples_split: self.min_samples_split,
                min_samples_leaf: self.min_samples_leaf,
                max_features: max_feat,
                n_classes,
                n_samples,
                n_features,
                total_tree_weight,
                x_rows: &x.data,
                class_idx: &class_idx_u32,
                root_min: &root_min,
                root_max: &root_max,
                root_counts: &root_counts,
            };

            let results = if n_samples < (1usize << 24) && n_classes <= 256 {
                run_et_clf_trees_packed::<u32>(&cfg, self.n_estimators, base_seed)
            } else {
                run_et_clf_trees_packed::<u64>(&cfg, self.n_estimators, base_seed)
            };
            merge_flat_clf_trees(results)
        } else {
            let sw = validate_sample_weight(sample_weight, x.rows)?;
            let class_idx: Vec<usize> = y
                .iter()
                .map(|&val| {
                    classes
                        .iter()
                        .position(|&c| (c - val).abs() <= 1e-12)
                        .unwrap_or(0)
                })
                .collect();
            let x_cols = if self.extra_trees {
                Vec::new()
            } else {
                transpose_to_col_major(x)
            };

            let results: Vec<(Vec<FlatClfNode>, Vec<f64>, Vec<f64>)> = (0..self.n_estimators)
                .into_par_iter()
                .map(|tree_idx| {
                    let seed = base_seed
                        .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ 0xA5A5_5A5A_1234_5678;
                    let mut rng = SplitMix64::new(seed);
                    let (mut samples, total_tree_weight) =
                        build_bootstrap_samples(n_samples, self.bootstrap, &sw, &mut rng);
                    let cfg = ClfBuildCfg {
                        criterion_entropy,
                        extra_trees: self.extra_trees,
                        max_depth: self.max_depth,
                        min_samples_split: self.min_samples_split,
                        min_samples_leaf: self.min_samples_leaf,
                        max_features: max_feat,
                        n_classes,
                        n_samples,
                        n_features,
                        total_tree_weight,
                        x_rows: &x.data,
                        x_cols: &x_cols,
                        class_idx: &class_idx,
                    };
                    let mut imp = vec![0.0; n_features];
                    let sorted_cap = if self.extra_trees { 0 } else { samples.len() };
                    let mut scratch = ClfTreeScratch::new(
                        n_features,
                        max_feat,
                        n_classes,
                        sorted_cap,
                        self.extra_trees,
                    );
                    let mut t_nodes = Vec::with_capacity(256);
                    let mut t_probs = Vec::with_capacity(128 * n_classes);
                    build_clf_tree(
                        &cfg,
                        &mut samples,
                        0,
                        &mut rng,
                        &mut imp,
                        &mut scratch,
                        &mut t_nodes,
                        &mut t_probs,
                    );
                    (t_nodes, t_probs, imp)
                })
                .collect();
            merge_flat_clf_trees(results)
        };

        self.flat_nodes = flat_nodes;
        self.flat_roots = flat_roots;
        self.flat_probs = flat_probs;
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
        let inv_trees = 1.0 / (self.flat_roots.len() as f64);
        let mut out = vec![0.0; x.rows * n_classes];

        if x.rows >= 256 {
            out.par_chunks_mut(256 * n_classes)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_r = chunk_idx * 256;
                    for (offset, dst) in chunk.chunks_exact_mut(n_classes).enumerate() {
                        let r = base_r + offset;
                        let row = &x.data[r * expected..(r + 1) * expected];
                        for &root in &self.flat_roots {
                            let probs = predict_flat_clf_proba(
                                &self.flat_nodes,
                                &self.flat_probs,
                                n_classes,
                                root,
                                row,
                            );
                            for (d, &p) in dst.iter_mut().zip(probs.iter()) {
                                *d += p;
                            }
                        }
                        for d in dst.iter_mut() {
                            *d *= inv_trees;
                        }
                    }
                });
        } else {
            for (r, dst) in out.chunks_exact_mut(n_classes).enumerate() {
                let row = &x.data[r * expected..(r + 1) * expected];
                for &root in &self.flat_roots {
                    let probs = predict_flat_clf_proba(
                        &self.flat_nodes,
                        &self.flat_probs,
                        n_classes,
                        root,
                        row,
                    );
                    for (d, &p) in dst.iter_mut().zip(probs.iter()) {
                        *d += p;
                    }
                }
                for d in dst.iter_mut() {
                    *d *= inv_trees;
                }
            }
        }

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
        let n_classes = self.classes_.len();
        write_usize(&mut buf, self.flat_roots.len());
        for &root in &self.flat_roots {
            write_flat_clf_node(
                &self.flat_nodes,
                &self.flat_probs,
                n_classes,
                root,
                &mut buf,
            );
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
        let mut flat_nodes = Vec::with_capacity(n_trees * 64);
        let mut flat_roots = Vec::with_capacity(n_trees);
        let mut flat_probs = Vec::with_capacity(n_trees * 32 * classes_.len().max(1));
        for _ in 0..n_trees {
            flat_roots.push(read_flat_clf_node(
                &mut r,
                &mut flat_nodes,
                &mut flat_probs,
            )?);
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
            flat_nodes,
            flat_roots,
            flat_probs,
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
    flat_nodes: Vec<FlatRegNode>,
    flat_roots: Vec<usize>,
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
            flat_nodes: Vec::new(),
            flat_roots: Vec::new(),
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

        let use_et_u32 = self.extra_trees
            && !self.bootstrap
            && sample_weight.is_none()
            && n_samples <= u32::MAX as usize;

        let (flat_nodes, flat_roots, all_imp) = if use_n6_gpu {
            let sw = validate_sample_weight(sample_weight, x.rows)?;
            let binned = backend.quantize_features(x, 256)?;
            let mut per_tree = Vec::with_capacity(self.n_estimators);
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
                let mut t_nodes = Vec::with_capacity(256);
                build_reg_tree_n6(
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
                    &mut t_nodes,
                )?;
                per_tree.push((t_nodes, imp));
            }
            merge_flat_reg_trees(per_tree)
        } else if use_et_u32 {
            let (root_sum_y, root_sum_y2) = if n_samples >= 4096 {
                y.par_chunks(4096)
                    .map(|chunk| {
                        let mut sy = 0.0;
                        let mut sy2 = 0.0;
                        for &yi in chunk {
                            sy += yi;
                            sy2 += yi * yi;
                        }
                        (sy, sy2)
                    })
                    .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
            } else {
                let mut sy = 0.0;
                let mut sy2 = 0.0;
                for &yi in y {
                    sy += yi;
                    sy2 += yi * yi;
                }
                (sy, sy2)
            };
            let (root_min, root_max) = compute_root_min_max(&x.data, n_samples, n_features);
            let total_tree_weight = n_samples as f64;
            let cfg = EtRegCfg {
                max_depth: self.max_depth,
                min_samples_split: self.min_samples_split,
                min_samples_leaf: self.min_samples_leaf,
                max_features: max_feat,
                n_samples,
                n_features,
                total_tree_weight,
                x_rows: &x.data,
                y,
                root_min: &root_min,
                root_max: &root_max,
                root_sum_y,
                root_sum_y2,
            };

            let results: Vec<(Vec<FlatRegNode>, Vec<f64>)> = (0..self.n_estimators)
                .into_par_iter()
                .map(|tree_idx| {
                    let seed = base_seed
                        .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ 0x5A5A_A5A5_8765_4321;
                    let mut rng = SplitMix64::new(seed);
                    let mut imp = vec![0.0; n_features];
                    let mut scratch = EtRegScratch::new(n_features, max_feat);
                    let mut t_nodes = Vec::with_capacity(256);
                    build_reg_tree_root_et_u32(
                        &cfg,
                        &mut rng,
                        &mut imp,
                        &mut scratch,
                        &mut t_nodes,
                    );
                    (t_nodes, imp)
                })
                .collect();
            merge_flat_reg_trees(results)
        } else {
            let sw = validate_sample_weight(sample_weight, x.rows)?;
            let x_cols = if self.extra_trees {
                Vec::new()
            } else {
                transpose_to_col_major(x)
            };
            let results: Vec<(Vec<FlatRegNode>, Vec<f64>)> = (0..self.n_estimators)
                .into_par_iter()
                .map(|tree_idx| {
                    let seed = base_seed
                        .wrapping_add((tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ 0x5A5A_A5A5_8765_4321;
                    let mut rng = SplitMix64::new(seed);
                    let (mut samples, total_tree_weight) =
                        build_bootstrap_samples(n_samples, self.bootstrap, &sw, &mut rng);
                    let cfg = RegBuildCfg {
                        extra_trees: self.extra_trees,
                        max_depth: self.max_depth,
                        min_samples_split: self.min_samples_split,
                        min_samples_leaf: self.min_samples_leaf,
                        max_features: max_feat,
                        n_samples,
                        n_features,
                        total_tree_weight,
                        x_rows: &x.data,
                        x_cols: &x_cols,
                        y,
                    };
                    let mut imp = vec![0.0; n_features];
                    let sorted_cap = if self.extra_trees { 0 } else { samples.len() };
                    let mut scratch =
                        RegTreeScratch::new(n_features, max_feat, sorted_cap, self.extra_trees);
                    let mut t_nodes = Vec::with_capacity(256);
                    build_reg_tree(
                        &cfg,
                        &mut samples,
                        0,
                        &mut rng,
                        &mut imp,
                        &mut scratch,
                        &mut t_nodes,
                    );
                    (t_nodes, imp)
                })
                .collect();
            merge_flat_reg_trees(results)
        };

        self.flat_nodes = flat_nodes;
        self.flat_roots = flat_roots;
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
        let inv_trees = 1.0 / (self.flat_roots.len() as f64);
        let mut preds = vec![0.0; x.rows];
        if x.rows >= 256 {
            preds
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_r = chunk_idx * 256;
                    for (offset, dst) in chunk.iter_mut().enumerate() {
                        let r = base_r + offset;
                        let row = &x.data[r * expected..(r + 1) * expected];
                        let mut sum = 0.0;
                        for &root in &self.flat_roots {
                            sum += predict_flat_reg_tree(&self.flat_nodes, root, row);
                        }
                        *dst = sum * inv_trees;
                    }
                });
        } else {
            for (r, dst) in preds.iter_mut().enumerate() {
                let row = &x.data[r * expected..(r + 1) * expected];
                let mut sum = 0.0;
                for &root in &self.flat_roots {
                    sum += predict_flat_reg_tree(&self.flat_nodes, root, row);
                }
                *dst = sum * inv_trees;
            }
        }
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
        write_usize(&mut buf, self.flat_roots.len());
        for &root in &self.flat_roots {
            write_flat_reg_node(&self.flat_nodes, root, &mut buf);
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
        let mut flat_nodes = Vec::with_capacity(n_trees * 64);
        let mut flat_roots = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            flat_roots.push(read_flat_reg_node(&mut r, &mut flat_nodes)?);
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
            flat_nodes,
            flat_roots,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extra_trees_and_rf_fit_predict_dit_roundtrip() {
        let n = 300usize;
        let mut data = Vec::with_capacity(n * 4);
        let mut y_cls = Vec::with_capacity(n);
        let mut y_reg = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = (i as f64) / (n as f64);
            let x1 = ((i * 7) % 31) as f64 / 31.0;
            let x2 = ((i * 11) % 19) as f64 / 19.0;
            let x3 = x0 * 2.0 - x1;
            data.extend_from_slice(&[x0, x1, x2, x3]);
            y_cls.push(if x0 >= 0.5 { 1.0 } else { 0.0 });
            y_reg.push(3.0 * x0 + 0.5 * x1);
        }
        let x = Matrix::from_vec(n, 4, data).unwrap();

        let mut etc = ExtraTreesClassifier::new(
            Some(20),
            None,
            Some(8),
            None,
            None,
            None,
            None,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        etc.fit(&x, &y_cls, None).unwrap();
        assert!(etc.score(&x, &y_cls).unwrap() > 0.95);
        let etc_loaded = ExtraTreesClassifier::from_bytes(&etc.to_bytes(), None).unwrap();
        assert_eq!(etc.predict(&x).unwrap(), etc_loaded.predict(&x).unwrap());

        let mut etr = ExtraTreesRegressor::new(
            Some(20),
            None,
            Some(8),
            None,
            None,
            None,
            None,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        etr.fit(&x, &y_reg, None).unwrap();
        assert!(etr.score(&x, &y_reg).unwrap() > 0.95);
        let etr_loaded = ExtraTreesRegressor::from_bytes(&etr.to_bytes(), None).unwrap();
        assert_eq!(etr.predict(&x).unwrap(), etr_loaded.predict(&x).unwrap());

        let mut rfc = RandomForestClassifier::new(
            Some(10),
            None,
            Some(6),
            None,
            None,
            None,
            None,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        rfc.fit(&x, &y_cls, None).unwrap();
        assert!(rfc.score(&x, &y_cls).unwrap() > 0.95);

        let mut rfr = RandomForestRegressor::new(
            Some(10),
            None,
            Some(6),
            None,
            None,
            None,
            None,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        rfr.fit(&x, &y_reg, None).unwrap();
        assert!(rfr.score(&x, &y_reg).unwrap() > 0.95);
    }

    #[test]
    fn test_extra_trees_u32_fast_path_parity_and_edge_cases() {
        let n = 5000usize;
        let p = 8usize;
        let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_BABE);
        let mut data = Vec::with_capacity(n * p);
        let mut y_cls = Vec::with_capacity(n);
        let mut y_reg = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.next_f64();
            let x1 = rng.next_f64();
            let x2 = 3.5; // constant feature to verify constant-feature skipping
            let x3 = rng.next_f64();
            let x4 = x0 - 0.5 * x1;
            let x5 = rng.next_f64();
            let x6 = rng.next_f64();
            let x7 = x3 * x5;
            data.extend_from_slice(&[x0, x1, x2, x3, x4, x5, x6, x7]);
            let cls = if x0 + x1 < 0.7 {
                0.0
            } else if x0 + x1 < 1.3 {
                1.0
            } else {
                2.0
            };
            y_cls.push(cls);
            y_reg.push(2.0 * x0 - 1.5 * x1 + 0.8 * x3 + 0.4 * x7);
        }
        let x = Matrix::from_vec(n, p, data).unwrap();
        let unit_w = vec![1.0; n];

        // Classifier parity: unweighted u32 fast-path vs unit-weighted fallback (gini & entropy, sqrt & all)
        for crit in ["gini", "entropy"] {
            for mf in [None, Some("all".to_string())] {
                let mut etc_fast = ExtraTreesClassifier::new(
                    Some(12),
                    Some(crit.to_string()),
                    Some(7),
                    Some(4),
                    Some(2),
                    mf.clone(),
                    Some(false),
                    Some(123),
                    Some("cpu".to_string()),
                )
                .unwrap();
                etc_fast.fit(&x, &y_cls, None).unwrap();

                let mut etc_ref = ExtraTreesClassifier::new(
                    Some(12),
                    Some(crit.to_string()),
                    Some(7),
                    Some(4),
                    Some(2),
                    mf,
                    Some(false),
                    Some(123),
                    Some("cpu".to_string()),
                )
                .unwrap();
                etc_ref.fit(&x, &y_cls, Some(&unit_w)).unwrap();

                let p_fast = etc_fast.predict_proba(&x).unwrap();
                let p_ref = etc_ref.predict_proba(&x).unwrap();
                for (a, b) in p_fast.data.iter().zip(p_ref.data.iter()) {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "clf proba mismatch ({crit}): {a} vs {b}"
                    );
                }
                assert_eq!(etc_fast.predict(&x).unwrap(), etc_ref.predict(&x).unwrap());
                let imp_fast = etc_fast.core.feature_importances_.as_ref().unwrap();
                let imp_ref = etc_ref.core.feature_importances_.as_ref().unwrap();
                for (a, b) in imp_fast.iter().zip(imp_ref.iter()) {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "clf imp mismatch ({crit}): {a} vs {b}"
                    );
                }
                let roundtrip =
                    ExtraTreesClassifier::from_bytes(&etc_fast.to_bytes(), None).unwrap();
                assert_eq!(
                    etc_fast.predict(&x).unwrap(),
                    roundtrip.predict(&x).unwrap()
                );
            }
        }

        // Regressor parity: unweighted u32 fast-path vs unit-weighted fallback (all & sqrt)
        for mf in [None, Some("sqrt".to_string())] {
            let mut etr_fast = ExtraTreesRegressor::new(
                Some(12),
                None,
                Some(7),
                Some(4),
                Some(2),
                mf.clone(),
                Some(false),
                Some(123),
                Some("cpu".to_string()),
            )
            .unwrap();
            etr_fast.fit(&x, &y_reg, None).unwrap();

            let mut etr_ref = ExtraTreesRegressor::new(
                Some(12),
                None,
                Some(7),
                Some(4),
                Some(2),
                mf,
                Some(false),
                Some(123),
                Some("cpu".to_string()),
            )
            .unwrap();
            etr_ref.fit(&x, &y_reg, Some(&unit_w)).unwrap();

            let y_fast = etr_fast.predict(&x).unwrap();
            let y_ref = etr_ref.predict(&x).unwrap();
            for (a, b) in y_fast.iter().zip(y_ref.iter()) {
                assert!((a - b).abs() < 1e-10, "reg pred mismatch: {a} vs {b}");
            }
            let imp_fast = etr_fast.core.feature_importances_.as_ref().unwrap();
            let imp_ref = etr_ref.core.feature_importances_.as_ref().unwrap();
            for (a, b) in imp_fast.iter().zip(imp_ref.iter()) {
                assert!((a - b).abs() < 1e-10, "reg imp mismatch: {a} vs {b}");
            }
            let roundtrip = ExtraTreesRegressor::from_bytes(&etr_fast.to_bytes(), None).unwrap();
            assert_eq!(
                etr_fast.predict(&x).unwrap(),
                roundtrip.predict(&x).unwrap()
            );
        }

        // Bootstrapped ExtraTrees with skipped x_cols must also work cleanly
        let mut etc_boot = ExtraTreesClassifier::new(
            Some(10),
            None,
            Some(6),
            None,
            None,
            None,
            Some(true),
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        etc_boot.fit(&x, &y_cls, None).unwrap();
        assert!(etc_boot.score(&x, &y_cls).unwrap() > 0.85);

        let mut etr_boot = ExtraTreesRegressor::new(
            Some(10),
            None,
            Some(6),
            None,
            None,
            None,
            Some(true),
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        etr_boot.fit(&x, &y_reg, None).unwrap();
        assert!(etr_boot.score(&x, &y_reg).unwrap() > 0.85);
    }
}
