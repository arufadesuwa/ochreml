// OchreML Categorical & Target Encoding Engine (*≧ω≦*)
// Rayon-parallel hash/sort encoders: OneHotEncoder, OrdinalEncoder, TargetEncoder,
// LabelEncoder, LabelBinarizer, and MultiLabelBinarizer.

use crate::device::{DeviceType, parse_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header, write_i64,
    write_opt_f64, write_opt_str, write_opt_usize, write_str, write_u8, write_usize,
    write_usize_vec,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub const TAG_ONE_HOT_ENCODER: u8 = 19;
pub const TAG_ORDINAL_ENCODER: u8 = 20;
pub const TAG_LABEL_ENCODER: u8 = 21;
pub const TAG_LABEL_BINARIZER: u8 = 22;
pub const TAG_MULTILABEL_BINARIZER: u8 = 23;
pub const TAG_TARGET_ENCODER: u8 = 24;

/// Canonical category value supporting integers, floats (NaN-safe), strings, and missing values.
#[derive(Debug, Clone)]
pub enum CatValue {
    Int(i64),
    Float(u64), // IEEE-754 bits
    Str(String),
    Missing,
}

impl CatValue {
    pub fn from_f64(v: f64) -> Self {
        if v.is_nan() {
            CatValue::Missing
        } else {
            // Canonicalize -0.0 to +0.0
            let clean = if v == 0.0 { 0.0 } else { v };
            CatValue::Float(clean.to_bits())
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, CatValue::Missing)
    }

    pub fn sort_cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (CatValue::Missing, CatValue::Missing) => Ordering::Equal,
            (CatValue::Missing, _) => Ordering::Greater,
            (_, CatValue::Missing) => Ordering::Less,
            (CatValue::Int(a), CatValue::Int(b)) => a.cmp(b),
            (CatValue::Float(a), CatValue::Float(b)) => {
                f64::from_bits(*a).total_cmp(&f64::from_bits(*b))
            }
            (CatValue::Int(a), CatValue::Float(b)) => (*a as f64).total_cmp(&f64::from_bits(*b)),
            (CatValue::Float(a), CatValue::Int(b)) => f64::from_bits(*a).total_cmp(&(*b as f64)),
            (CatValue::Str(a), CatValue::Str(b)) => a.cmp(b),
            (CatValue::Str(a), _) => a.cmp(&other.display_key()),
            (_, CatValue::Str(b)) => self.display_key().cmp(b),
        }
    }

    pub fn display_key(&self) -> String {
        match self {
            CatValue::Int(v) => v.to_string(),
            CatValue::Float(bits) => format!("{}", f64::from_bits(*bits)),
            CatValue::Str(s) => s.clone(),
            CatValue::Missing => "nan".to_string(),
        }
    }

    pub fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            CatValue::Missing => write_u8(buf, 0),
            CatValue::Int(v) => {
                write_u8(buf, 1);
                write_i64(buf, *v);
            }
            CatValue::Float(bits) => {
                write_u8(buf, 2);
                write_f64(buf, f64::from_bits(*bits));
            }
            CatValue::Str(s) => {
                write_u8(buf, 3);
                write_str(buf, s);
            }
        }
    }

    pub fn read_from(r: &mut ByteReader<'_>) -> OchreResult<Self> {
        match r.read_u8()? {
            0 => Ok(CatValue::Missing),
            1 => Ok(CatValue::Int(r.read_i64()?)),
            2 => Ok(CatValue::Float(r.read_f64()?.to_bits())),
            3 => Ok(CatValue::Str(r.read_str()?)),
            _ => Err(OchreError::CorruptedDitFile),
        }
    }
}

impl PartialEq for CatValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CatValue::Missing, CatValue::Missing) => true,
            (CatValue::Int(a), CatValue::Int(b)) => a == b,
            (CatValue::Float(a), CatValue::Float(b)) => a == b,
            (CatValue::Str(a), CatValue::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CatValue {}

impl Hash for CatValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            CatValue::Missing => {}
            CatValue::Int(v) => v.hash(state),
            CatValue::Float(bits) => bits.hash(state),
            CatValue::Str(s) => s.hash(state),
        }
    }
}

fn resolve_loaded_device(
    saved_device: Option<String>,
    is_fitted: bool,
    device: Option<&str>,
) -> OchreResult<(DeviceType, Option<String>)> {
    let dev = parse_device(device.or(Some("auto")))?;
    let device_ = if saved_device.is_some() || is_fitted {
        Some(dev.to_string())
    } else {
        None
    };
    Ok((dev, device_))
}

/// Extract sorted unique categories and their counts for a column.
fn unique_sorted_with_counts(col: &[CatValue]) -> (Vec<CatValue>, Vec<usize>) {
    let mut counts: HashMap<CatValue, usize> = HashMap::new();
    for v in col {
        *counts.entry(v.clone()).or_insert(0) += 1;
    }
    let mut pairs: Vec<(CatValue, usize)> = counts.into_iter().collect();
    pairs.sort_by(|(a, _), (b, _)| a.sort_cmp(b));
    let cats = pairs.iter().map(|(c, _)| c.clone()).collect();
    let cnts = pairs.iter().map(|(_, n)| *n).collect();
    (cats, cnts)
}

/// Compute infrequent category mask based on `min_frequency` and `max_categories`.
fn identify_infrequent_mask(
    counts: &[usize],
    n_samples: usize,
    min_frequency: Option<f64>,
    max_categories: Option<usize>,
) -> Vec<bool> {
    let n_cats = counts.len();
    let mut is_infreq = vec![false; n_cats];
    if let Some(mf) = min_frequency {
        let thresh = if mf < 1.0 {
            mf * (n_samples as f64)
        } else {
            mf
        };
        for (i, &c) in counts.iter().enumerate() {
            if (c as f64) < thresh {
                is_infreq[i] = true;
            }
        }
    }
    if let Some(mc) = max_categories {
        let remaining: usize = is_infreq.iter().filter(|&&f| !f).count();
        if remaining > mc && mc >= 1 {
            // Keep top `mc - 1` frequent categories, mark rest infrequent
            let mut active: Vec<(usize, usize)> = counts
                .iter()
                .enumerate()
                .filter(|(i, _)| !is_infreq[*i])
                .map(|(i, &c)| (i, c))
                .collect();
            active.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let keep = mc - 1;
            for &(idx, _) in active.iter().skip(keep) {
                is_infreq[idx] = true;
            }
        }
    }
    is_infreq
}

fn build_cat_map(cats: &[CatValue]) -> HashMap<CatValue, usize> {
    let mut map = HashMap::with_capacity(cats.len());
    for (idx, c) in cats.iter().enumerate() {
        map.insert(c.clone(), idx);
    }
    map
}

fn trim_u32_cell(cell: &[u32]) -> &[u32] {
    let mut end = cell.len();
    while end > 0 && cell[end - 1] == 0 {
        end -= 1;
    }
    &cell[..end]
}

fn u32_cell_to_string(cell: &[u32]) -> OchreResult<String> {
    let mut s = String::with_capacity(cell.len() * 2);
    for &cp in cell {
        let ch = char::from_u32(cp).ok_or_else(|| {
            OchreError::InvalidParameter(format!(
                "Invalid Unicode code point {} in string array ( >_< )",
                cp
            ))
        })?;
        s.push(ch);
    }
    Ok(s)
}

fn intern_u32_column(
    data: &[u32],
    n_rows: usize,
    n_cols: usize,
    width: usize,
    col_idx: usize,
) -> OchreResult<(Vec<CatValue>, Vec<usize>, Vec<usize>)> {
    let mut map: HashMap<&[u32], usize> = HashMap::new();
    let mut raw_keys: Vec<&[u32]> = Vec::new();
    let mut raw_counts: Vec<usize> = Vec::new();
    let mut raw_ids = Vec::with_capacity(n_rows);
    let stride = n_cols * width;
    let col_off = col_idx * width;

    for r in 0..n_rows {
        let start = r * stride + col_off;
        let cell = trim_u32_cell(&data[start..start + width]);
        let id = if let Some(&existing) = map.get(cell) {
            raw_counts[existing] += 1;
            existing
        } else {
            let new_id = raw_keys.len();
            map.insert(cell, new_id);
            raw_keys.push(cell);
            raw_counts.push(1);
            new_id
        };
        raw_ids.push(id);
    }

    let n_uniq = raw_keys.len();
    let mut cats_raw = Vec::with_capacity(n_uniq);
    for key in raw_keys {
        cats_raw.push(CatValue::Str(u32_cell_to_string(key)?));
    }

    let mut order: Vec<usize> = (0..n_uniq).collect();
    order.sort_by(|&a, &b| cats_raw[a].sort_cmp(&cats_raw[b]));

    let mut sorted_cats = Vec::with_capacity(n_uniq);
    let mut sorted_counts = Vec::with_capacity(n_uniq);
    let mut raw_to_sorted = vec![0usize; n_uniq];
    for (sorted_pos, &orig_idx) in order.iter().enumerate() {
        sorted_cats.push(cats_raw[orig_idx].clone());
        sorted_counts.push(raw_counts[orig_idx]);
        raw_to_sorted[orig_idx] = sorted_pos;
    }
    for id in &mut raw_ids {
        *id = raw_to_sorted[*id];
    }
    Ok((sorted_cats, sorted_counts, raw_ids))
}

fn build_u32_patterns(cats: &[CatValue]) -> Vec<Option<Vec<u32>>> {
    cats.iter()
        .map(|c| match c {
            CatValue::Str(s) => Some(s.chars().map(|ch| ch as u32).collect()),
            _ => None,
        })
        .collect()
}

pub fn u32_buf_to_cat_cols(
    data: &[u32],
    n_rows: usize,
    n_cols: usize,
    width: usize,
) -> OchreResult<Vec<Vec<CatValue>>> {
    if n_rows == 0 || n_cols == 0 || width == 0 {
        return Err(OchreError::EmptyData);
    }
    (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let (cats, _, row_ids) = intern_u32_column(data, n_rows, n_cols, width, j)?;
            Ok(row_ids.into_iter().map(|id| cats[id].clone()).collect())
        })
        .collect()
}

/// Encode categorical features as a one-hot numeric array (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    pub handle_unknown: String,
    pub drop_mode: Option<String>,
    pub min_frequency: Option<f64>,
    pub max_categories: Option<usize>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub categories_: Option<Vec<Vec<CatValue>>>,
    pub drop_idx_: Option<Vec<Option<usize>>>,
    pub infrequent_indices_: Option<Vec<Vec<usize>>>,
    /// Mapping from category index to output column offset within feature `j` (None if dropped)
    pub feature_out_map_: Vec<Vec<Option<usize>>>,
    pub n_out_per_feat_: Vec<usize>,
    pub n_features_in_: Option<usize>,
}

impl OneHotEncoder {
    pub fn new(
        handle_unknown: String,
        drop_mode: Option<String>,
        min_frequency: Option<f64>,
        max_categories: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !matches!(
            handle_unknown.as_str(),
            "error" | "ignore" | "infrequent_if_exist"
        ) {
            return Err(OchreError::InvalidParameter(format!(
                "handle_unknown must be 'error', 'ignore', or 'infrequent_if_exist', got '{}' ( >_< )",
                handle_unknown
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            handle_unknown,
            drop_mode,
            min_frequency,
            max_categories,
            device: dev,
            device_: None,
            categories_: None,
            drop_idx_: None,
            infrequent_indices_: None,
            feature_out_map_: Vec::new(),
            n_out_per_feat_: Vec::new(),
            n_features_in_: None,
        })
    }

    pub fn fit(
        &mut self,
        cols_data: &[Vec<CatValue>],
        explicit_categories: Option<&[Vec<CatValue>]>,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<()> {
        if cols_data.is_empty() || cols_data[0].is_empty() {
            return Err(OchreError::EmptyData);
        }
        let n_features = cols_data.len();
        let n_samples = cols_data[0].len();
        if let Some(ec) = explicit_categories
            && ec.len() != n_features
        {
            return Err(OchreError::FeatureCountMismatch {
                features: ec.len(),
                expected: n_features,
            });
        }

        let per_col: Vec<(Vec<CatValue>, Vec<usize>, Vec<bool>)> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let col = &cols_data[j];
                if let Some(ec) = explicit_categories {
                    let cats = ec[j].clone();
                    let mut counts = vec![0usize; cats.len()];
                    let cmap = build_cat_map(&cats);
                    for v in col {
                        if let Some(&idx) = cmap.get(v) {
                            counts[idx] += 1;
                        }
                    }
                    let mask = identify_infrequent_mask(
                        &counts,
                        n_samples,
                        self.min_frequency,
                        self.max_categories,
                    );
                    (cats, counts, mask)
                } else {
                    let (cats, counts) = unique_sorted_with_counts(col);
                    let mask = identify_infrequent_mask(
                        &counts,
                        n_samples,
                        self.min_frequency,
                        self.max_categories,
                    );
                    (cats, counts, mask)
                }
            })
            .collect();

        self.apply_fit_per_col(per_col, explicit_drop)
    }

    fn apply_fit_per_col(
        &mut self,
        per_col: Vec<(Vec<CatValue>, Vec<usize>, Vec<bool>)>,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<()> {
        let n_features = per_col.len();
        let mut categories = Vec::with_capacity(n_features);
        let mut infreq_indices = Vec::with_capacity(n_features);
        let mut drop_indices: Vec<Option<usize>> = Vec::with_capacity(n_features);
        let mut out_maps = Vec::with_capacity(n_features);
        let mut n_out_per_feat = Vec::with_capacity(n_features);

        for (j, (cats, _counts, mask)) in per_col.into_iter().enumerate() {
            let infreq_idxs: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_inf)| is_inf)
                .map(|(i, _)| i)
                .collect();
            let has_infreq = !infreq_idxs.is_empty();

            // Build pre-drop category -> local output index mapping
            let mut n_active = 0usize;
            let mut local_idx = vec![0usize; cats.len()];
            for (i, &is_inf) in mask.iter().enumerate() {
                if !is_inf {
                    local_idx[i] = n_active;
                    n_active += 1;
                }
            }
            if has_infreq {
                let infreq_slot = n_active;
                for (i, &is_inf) in mask.iter().enumerate() {
                    if is_inf {
                        local_idx[i] = infreq_slot;
                    }
                }
                n_active += 1;
            }

            let drop_pos: Option<usize> = if let Some(ed) = explicit_drop {
                match &ed[j] {
                    Some(target) => {
                        let pos = cats.iter().position(|c| c == target).ok_or_else(|| {
                            OchreError::InvalidParameter(format!(
                                "Drop category '{}' not found in feature {} ( >_< )",
                                target.display_key(),
                                j
                            ))
                        })?;
                        if mask[pos] {
                            return Err(OchreError::InvalidParameter(
                                "Cannot drop an infrequent category ( >_< )".to_string(),
                            ));
                        }
                        Some(local_idx[pos])
                    }
                    None => None,
                }
            } else {
                match self.drop_mode.as_deref() {
                    Some("first") => {
                        if n_active > 0 {
                            Some(0)
                        } else {
                            None
                        }
                    }
                    Some("if_binary") => {
                        if n_active == 2 {
                            Some(0)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };

            let mut out_map = vec![None; cats.len()];
            let final_n_out = if let Some(dp) = drop_pos {
                for (i, &loc) in local_idx.iter().enumerate() {
                    if loc < dp {
                        out_map[i] = Some(loc);
                    } else if loc > dp {
                        out_map[i] = Some(loc - 1);
                    }
                }
                n_active.saturating_sub(1)
            } else {
                for (i, &loc) in local_idx.iter().enumerate() {
                    out_map[i] = Some(loc);
                }
                n_active
            };

            // Store original category index for drop_idx_ (matching sklearn)
            let drop_cat_idx = drop_pos.and_then(|dp| local_idx.iter().position(|&l| l == dp));

            categories.push(cats);
            infreq_indices.push(infreq_idxs);
            drop_indices.push(drop_cat_idx);
            out_maps.push(out_map);
            n_out_per_feat.push(final_n_out);
        }

        let has_any_drop = drop_indices.iter().any(|d| d.is_some());
        self.categories_ = Some(categories);
        self.infrequent_indices_ = Some(infreq_indices);
        self.drop_idx_ = if has_any_drop {
            Some(drop_indices)
        } else {
            None
        };
        self.feature_out_map_ = out_maps;
        self.n_out_per_feat_ = n_out_per_feat;
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Transform columns into CSR `(indices, indptr, total_cols)` in Rayon parallel.
    pub fn transform_csr(
        &self,
        cols_data: &[Vec<CatValue>],
    ) -> OchreResult<(Vec<usize>, Vec<usize>, usize, usize)> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if cols_data.len() != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: cols_data.len(),
                expected: n_features,
            });
        }
        let n_samples = cols_data[0].len();
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let infreq = self
            .infrequent_indices_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;

        let mut offsets = Vec::with_capacity(n_features);
        let mut total_cols = 0usize;
        for &nout in &self.n_out_per_feat_ {
            offsets.push(total_cols);
            total_cols += nout;
        }

        let maps: Vec<HashMap<CatValue, usize>> =
            categories.iter().map(|c| build_cat_map(c)).collect();

        // Map each column in parallel to `Option<usize>` output column index
        let col_encoded: Vec<Vec<Option<usize>>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let map = &maps[j];
                let out_map = &self.feature_out_map_[j];
                let off = offsets[j];
                let inf_out: Option<usize> = infreq[j]
                    .first()
                    .and_then(|&cat_i| out_map[cat_i])
                    .map(|l| off + l);

                let mut res = Vec::with_capacity(n_samples);
                for v in &cols_data[j] {
                    if let Some(&cat_idx) = map.get(v) {
                        res.push(out_map[cat_idx].map(|l| off + l));
                    } else if self.handle_unknown == "ignore" {
                        res.push(None);
                    } else if self.handle_unknown == "infrequent_if_exist" {
                        res.push(inf_out);
                    } else {
                        return Err(OchreError::InvalidParameter(format!(
                            "Found unknown category '{}' in column {} during transform ( >_< )",
                            v.display_key(),
                            j
                        )));
                    }
                }
                Ok(res)
            })
            .collect::<OchreResult<Vec<Vec<Option<usize>>>>>()?;
        Ok(Self::assemble_csr_from_encoded_cols(
            &col_encoded,
            n_samples,
            total_cols,
        ))
    }

    pub fn assemble_csr_from_encoded_cols(
        col_encoded: &[Vec<Option<usize>>],
        n_samples: usize,
        total_cols: usize,
    ) -> (Vec<usize>, Vec<usize>, usize, usize) {
        let n_features = col_encoded.len();
        let mut indices = Vec::with_capacity(n_samples * n_features);
        let mut indptr = Vec::with_capacity(n_samples + 1);
        indptr.push(0);
        for r in 0..n_samples {
            for col in col_encoded {
                if let Some(c_idx) = col[r] {
                    indices.push(c_idx);
                }
            }
            indptr.push(indices.len());
        }
        (indices, indptr, n_samples, total_cols)
    }

    pub fn fit_utf32(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<()> {
        if n_rows == 0 || n_cols == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let per_col = (0..n_cols)
            .into_par_iter()
            .map(|j| {
                let (cats, counts, _) = intern_u32_column(data, n_rows, n_cols, width, j)?;
                let mask = identify_infrequent_mask(
                    &counts,
                    n_rows,
                    self.min_frequency,
                    self.max_categories,
                );
                Ok((cats, counts, mask))
            })
            .collect::<OchreResult<Vec<_>>>()?;
        self.apply_fit_per_col(per_col, explicit_drop)
    }

    pub fn fit_interned_cols(
        &mut self,
        interned: Vec<(Vec<CatValue>, Vec<usize>)>,
        n_rows: usize,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<()> {
        let per_col = interned
            .into_iter()
            .map(|(cats, counts)| {
                let mask = identify_infrequent_mask(
                    &counts,
                    n_rows,
                    self.min_frequency,
                    self.max_categories,
                );
                (cats, counts, mask)
            })
            .collect();
        self.apply_fit_per_col(per_col, explicit_drop)
    }

    pub fn fit_transform_interned_csr(
        &mut self,
        interned: Vec<(Vec<CatValue>, Vec<usize>, Vec<usize>)>,
        n_rows: usize,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<(Vec<usize>, Vec<usize>, usize, usize)> {
        let n_cols = interned.len();
        let mut per_col = Vec::with_capacity(n_cols);
        let mut row_ids_cols = Vec::with_capacity(n_cols);
        for (cats, counts, rids) in interned {
            let mask =
                identify_infrequent_mask(&counts, n_rows, self.min_frequency, self.max_categories);
            per_col.push((cats, counts, mask));
            row_ids_cols.push(rids);
        }
        self.apply_fit_per_col(per_col, explicit_drop)?;

        let mut total_cols = 0usize;
        let col_encoded: Vec<Vec<Option<usize>>> = (0..n_cols)
            .map(|j| {
                let off = total_cols;
                total_cols += self.n_out_per_feat_[j];
                let out_map = &self.feature_out_map_[j];
                row_ids_cols[j]
                    .iter()
                    .map(|&cat_idx| out_map[cat_idx].map(|l| off + l))
                    .collect()
            })
            .collect();
        Ok(Self::assemble_csr_from_encoded_cols(
            &col_encoded,
            n_rows,
            total_cols,
        ))
    }

    pub fn fit_transform_utf32_csr(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
        explicit_drop: Option<&[Option<CatValue>]>,
    ) -> OchreResult<(Vec<usize>, Vec<usize>, usize, usize)> {
        if n_rows == 0 || n_cols == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let interned = (0..n_cols)
            .into_par_iter()
            .map(|j| intern_u32_column(data, n_rows, n_cols, width, j))
            .collect::<OchreResult<Vec<_>>>()?;
        self.fit_transform_interned_csr(interned, n_rows, explicit_drop)
    }

    pub fn transform_utf32_csr(
        &self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> OchreResult<(Vec<usize>, Vec<usize>, usize, usize)> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if n_cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: n_cols,
                expected: n_features,
            });
        }
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let infreq = self
            .infrequent_indices_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;

        let mut offsets = Vec::with_capacity(n_features);
        let mut total_cols = 0usize;
        for &nout in &self.n_out_per_feat_ {
            offsets.push(total_cols);
            total_cols += nout;
        }

        let stride = n_cols * width;
        let col_encoded: Vec<Vec<Option<usize>>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let patterns = build_u32_patterns(&categories[j]);
                let mut map: HashMap<&[u32], usize> = HashMap::with_capacity(patterns.len());
                for (idx, opt_p) in patterns.iter().enumerate() {
                    if let Some(p) = opt_p {
                        map.insert(p.as_slice(), idx);
                    }
                }
                let out_map = &self.feature_out_map_[j];
                let off = offsets[j];
                let inf_out: Option<usize> = infreq[j]
                    .first()
                    .and_then(|&cat_i| out_map[cat_i])
                    .map(|l| off + l);

                let col_off = j * width;
                let mut res = Vec::with_capacity(n_rows);
                for r in 0..n_rows {
                    let start = r * stride + col_off;
                    let cell = trim_u32_cell(&data[start..start + width]);
                    if let Some(&cat_idx) = map.get(cell) {
                        res.push(out_map[cat_idx].map(|l| off + l));
                    } else if self.handle_unknown == "ignore" {
                        res.push(None);
                    } else if self.handle_unknown == "infrequent_if_exist" {
                        res.push(inf_out);
                    } else {
                        let key = u32_cell_to_string(cell)?;
                        return Err(OchreError::InvalidParameter(format!(
                            "Found unknown category '{}' in column {} during transform ( >_< )",
                            key, j
                        )));
                    }
                }
                Ok(res)
            })
            .collect::<OchreResult<Vec<_>>>()?;
        Ok(Self::assemble_csr_from_encoded_cols(
            &col_encoded,
            n_rows,
            total_cols,
        ))
    }

    pub fn csr_to_dense(
        indices: &[usize],
        indptr: &[usize],
        n_samples: usize,
        total_cols: usize,
    ) -> OchreResult<Matrix> {
        let mut data = vec![0.0; n_samples * total_cols];
        for r in 0..n_samples {
            for &c_idx in &indices[indptr[r]..indptr[r + 1]] {
                data[r * total_cols + c_idx] = 1.0;
            }
        }
        Matrix::from_vec(n_samples, total_cols, data)
    }

    pub fn transform_dense(&self, cols_data: &[Vec<CatValue>]) -> OchreResult<Matrix> {
        let (indices, indptr, n_samples, total_cols) = self.transform_csr(cols_data)?;
        Self::csr_to_dense(&indices, &indptr, n_samples, total_cols)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Vec<Vec<Option<CatValue>>>> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let total_cols: usize = self.n_out_per_feat_.iter().sum();
        if x.cols != total_cols {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: total_cols,
            });
        }

        // Build reverse map `(feature_j, local_col_idx) -> CatValue`
        let mut rev_maps: Vec<Vec<CatValue>> = Vec::with_capacity(n_features);
        for (j, cat_col) in categories.iter().enumerate().take(n_features) {
            let nout = self.n_out_per_feat_[j];
            let mut rmap = vec![CatValue::Missing; nout];
            for (cat_i, opt_loc) in self.feature_out_map_[j].iter().enumerate() {
                if let Some(loc) = *opt_loc {
                    rmap[loc] = cat_col[cat_i].clone();
                }
            }
            rev_maps.push(rmap);
        }

        let mut out = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut row_vals = Vec::with_capacity(n_features);
            let mut offset = 0usize;
            for j in 0..n_features {
                let nout = self.n_out_per_feat_[j];
                let slice = &x.data[r * total_cols + offset..r * total_cols + offset + nout];
                let found = slice.iter().position(|&v| v == 1.0 || v > 0.5);
                if let Some(loc) = found {
                    row_vals.push(Some(rev_maps[j][loc].clone()));
                } else if let Some(ref drops) = self.drop_idx_
                    && let Some(d_cat) = drops[j]
                {
                    row_vals.push(Some(categories[j][d_cat].clone()));
                } else {
                    row_vals.push(None);
                }
                offset += nout;
            }
            out.push(row_vals);
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ONE_HOT_ENCODER);
        write_str(&mut buf, &self.handle_unknown);
        write_opt_str(&mut buf, &self.drop_mode);
        write_opt_f64(&mut buf, self.min_frequency);
        write_opt_usize(&mut buf, self.max_categories);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        match &self.categories_ {
            Some(cats) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cats.len());
                for cvec in cats {
                    write_usize(&mut buf, cvec.len());
                    for c in cvec {
                        c.write_to(&mut buf);
                    }
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.drop_idx_ {
            Some(drops) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, drops.len());
                for d in drops {
                    write_opt_usize(&mut buf, *d);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.infrequent_indices_ {
            Some(inf) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, inf.len());
                for v in inf {
                    write_usize_vec(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_usize(&mut buf, self.feature_out_map_.len());
        for fmap in &self.feature_out_map_ {
            write_usize(&mut buf, fmap.len());
            for opt in fmap {
                write_opt_usize(&mut buf, *opt);
            }
        }
        write_usize_vec(&mut buf, &self.n_out_per_feat_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ONE_HOT_ENCODER)?;
        let handle_unknown = r.read_str()?;
        let drop_mode = r.read_opt_str()?;
        let min_frequency = r.read_opt_f64()?;
        let max_categories = r.read_opt_usize()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let categories_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cats = Vec::with_capacity(n);
            for _ in 0..n {
                let m = r.read_usize()?;
                let mut cv = Vec::with_capacity(m);
                for _ in 0..m {
                    cv.push(CatValue::read_from(&mut r)?);
                }
                cats.push(cv);
            }
            Some(cats)
        } else {
            None
        };
        let drop_idx_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut d = Vec::with_capacity(n);
            for _ in 0..n {
                d.push(r.read_opt_usize()?);
            }
            Some(d)
        } else {
            None
        };
        let infrequent_indices_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut inf = Vec::with_capacity(n);
            for _ in 0..n {
                inf.push(r.read_usize_vec()?);
            }
            Some(inf)
        } else {
            None
        };
        let fmap_len = r.read_usize()?;
        let mut feature_out_map_ = Vec::with_capacity(fmap_len);
        for _ in 0..fmap_len {
            let m = r.read_usize()?;
            let mut fmap = Vec::with_capacity(m);
            for _ in 0..m {
                fmap.push(r.read_opt_usize()?);
            }
            feature_out_map_.push(fmap);
        }
        let n_out_per_feat_ = r.read_usize_vec()?;
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            handle_unknown,
            drop_mode,
            min_frequency,
            max_categories,
            device: dev,
            device_,
            categories_,
            drop_idx_,
            infrequent_indices_,
            feature_out_map_,
            n_out_per_feat_,
            n_features_in_,
        })
    }
}

/// Encode categorical features as an integer array (＾▽＾)
#[derive(Debug, Clone)]
pub struct OrdinalEncoder {
    pub handle_unknown: String,
    pub unknown_value: Option<f64>,
    pub encoded_missing_value: f64,
    pub min_frequency: Option<f64>,
    pub max_categories: Option<usize>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub categories_: Option<Vec<Vec<CatValue>>>,
    pub infrequent_indices_: Option<Vec<Vec<usize>>>,
    pub cat_to_code_: Vec<Vec<usize>>,
    pub infreq_code_: Vec<Option<usize>>,
    pub n_features_in_: Option<usize>,
}

impl OrdinalEncoder {
    pub fn new(
        handle_unknown: String,
        unknown_value: Option<f64>,
        encoded_missing_value: f64,
        min_frequency: Option<f64>,
        max_categories: Option<usize>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if handle_unknown != "error" && handle_unknown != "use_encoded_value" {
            return Err(OchreError::InvalidParameter(format!(
                "handle_unknown must be 'error' or 'use_encoded_value', got '{}' ( >_< )",
                handle_unknown
            )));
        }
        if handle_unknown == "use_encoded_value" && unknown_value.is_none() {
            return Err(OchreError::InvalidParameter(
                "unknown_value must be specified when handle_unknown='use_encoded_value' ( >_< )"
                    .to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            handle_unknown,
            unknown_value,
            encoded_missing_value,
            min_frequency,
            max_categories,
            device: dev,
            device_: None,
            categories_: None,
            infrequent_indices_: None,
            cat_to_code_: Vec::new(),
            infreq_code_: Vec::new(),
            n_features_in_: None,
        })
    }

    pub fn fit(
        &mut self,
        cols_data: &[Vec<CatValue>],
        explicit_categories: Option<&[Vec<CatValue>]>,
    ) -> OchreResult<()> {
        if cols_data.is_empty() || cols_data[0].is_empty() {
            return Err(OchreError::EmptyData);
        }
        let n_features = cols_data.len();
        let n_samples = cols_data[0].len();
        if let Some(ec) = explicit_categories
            && ec.len() != n_features
        {
            return Err(OchreError::FeatureCountMismatch {
                features: ec.len(),
                expected: n_features,
            });
        }

        let per_col: Vec<(Vec<CatValue>, Vec<bool>)> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let col = &cols_data[j];
                if let Some(ec) = explicit_categories {
                    let cats = ec[j].clone();
                    let mut counts = vec![0usize; cats.len()];
                    let cmap = build_cat_map(&cats);
                    for v in col {
                        if let Some(&idx) = cmap.get(v) {
                            counts[idx] += 1;
                        }
                    }
                    let mask = identify_infrequent_mask(
                        &counts,
                        n_samples,
                        self.min_frequency,
                        self.max_categories,
                    );
                    (cats, mask)
                } else {
                    let (cats, counts) = unique_sorted_with_counts(col);
                    let mask = identify_infrequent_mask(
                        &counts,
                        n_samples,
                        self.min_frequency,
                        self.max_categories,
                    );
                    (cats, mask)
                }
            })
            .collect();

        self.apply_fit_per_col(per_col)
    }

    fn apply_fit_per_col(&mut self, per_col: Vec<(Vec<CatValue>, Vec<bool>)>) -> OchreResult<()> {
        let n_features = per_col.len();
        let mut categories = Vec::with_capacity(n_features);
        let mut infreq_indices = Vec::with_capacity(n_features);
        let mut cat_to_code = Vec::with_capacity(n_features);
        let mut infreq_code = Vec::with_capacity(n_features);

        for (cats, mask) in per_col {
            let infreq_idxs: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_inf)| is_inf)
                .map(|(i, _)| i)
                .collect();
            let has_infreq = !infreq_idxs.is_empty();
            let mut codes = vec![0usize; cats.len()];
            let mut next_code = 0usize;
            for (i, &is_inf) in mask.iter().enumerate() {
                if !is_inf {
                    codes[i] = next_code;
                    next_code += 1;
                }
            }
            let inf_c = if has_infreq {
                let slot = next_code;
                for (i, &is_inf) in mask.iter().enumerate() {
                    if is_inf {
                        codes[i] = slot;
                    }
                }
                Some(slot)
            } else {
                None
            };
            categories.push(cats);
            infreq_indices.push(infreq_idxs);
            cat_to_code.push(codes);
            infreq_code.push(inf_c);
        }

        self.categories_ = Some(categories);
        self.infrequent_indices_ = Some(infreq_indices);
        self.cat_to_code_ = cat_to_code;
        self.infreq_code_ = infreq_code;
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, cols_data: &[Vec<CatValue>]) -> OchreResult<Matrix> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if cols_data.len() != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: cols_data.len(),
                expected: n_features,
            });
        }
        let n_samples = cols_data[0].len();
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let maps: Vec<HashMap<CatValue, usize>> =
            categories.iter().map(|c| build_cat_map(c)).collect();

        let encoded_cols: Vec<Vec<f64>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let map = &maps[j];
                let codes = &self.cat_to_code_[j];
                let mut out_col = Vec::with_capacity(n_samples);
                for v in &cols_data[j] {
                    if let Some(&cat_idx) = map.get(v) {
                        out_col.push(codes[cat_idx] as f64);
                    } else if v.is_missing() {
                        out_col.push(self.encoded_missing_value);
                    } else if self.handle_unknown == "use_encoded_value" {
                        out_col.push(self.unknown_value.unwrap_or(f64::NAN));
                    } else {
                        return Err(OchreError::InvalidParameter(format!(
                            "Found unknown category '{}' in column {} during transform ( >_< )",
                            v.display_key(),
                            j
                        )));
                    }
                }
                Ok(out_col)
            })
            .collect::<OchreResult<Vec<Vec<f64>>>>()?;

        let mut out = vec![0.0; n_samples * n_features];
        for r in 0..n_samples {
            for j in 0..n_features {
                out[r * n_features + j] = encoded_cols[j][r];
            }
        }
        Matrix::from_vec(n_samples, n_features, out)
    }

    pub fn fit_utf32(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> OchreResult<()> {
        if n_rows == 0 || n_cols == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let per_col = (0..n_cols)
            .into_par_iter()
            .map(|j| {
                let (cats, counts, _) = intern_u32_column(data, n_rows, n_cols, width, j)?;
                let mask = identify_infrequent_mask(
                    &counts,
                    n_rows,
                    self.min_frequency,
                    self.max_categories,
                );
                Ok((cats, mask))
            })
            .collect::<OchreResult<Vec<_>>>()?;
        self.apply_fit_per_col(per_col)
    }

    pub fn fit_interned_cols(
        &mut self,
        interned: Vec<(Vec<CatValue>, Vec<usize>)>,
        n_rows: usize,
    ) -> OchreResult<()> {
        let per_col = interned
            .into_iter()
            .map(|(cats, counts)| {
                let mask = identify_infrequent_mask(
                    &counts,
                    n_rows,
                    self.min_frequency,
                    self.max_categories,
                );
                (cats, mask)
            })
            .collect();
        self.apply_fit_per_col(per_col)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_interned(
        &mut self,
        interned: Vec<(Vec<CatValue>, Vec<usize>, Vec<usize>)>,
        n_rows: usize,
    ) -> OchreResult<Matrix> {
        let n_cols = interned.len();
        let mut per_col = Vec::with_capacity(n_cols);
        let mut row_ids_cols = Vec::with_capacity(n_cols);
        for (cats, counts, rids) in interned {
            let mask =
                identify_infrequent_mask(&counts, n_rows, self.min_frequency, self.max_categories);
            per_col.push((cats, mask));
            row_ids_cols.push(rids);
        }
        self.apply_fit_per_col(per_col)?;

        let mut out = vec![0.0; n_rows * n_cols];
        for j in 0..n_cols {
            let codes = &self.cat_to_code_[j];
            let rids = &row_ids_cols[j];
            for r in 0..n_rows {
                out[r * n_cols + j] = codes[rids[r]] as f64;
            }
        }
        Matrix::from_vec(n_rows, n_cols, out)
    }

    pub fn fit_transform_utf32(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> OchreResult<Matrix> {
        if n_rows == 0 || n_cols == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let interned = (0..n_cols)
            .into_par_iter()
            .map(|j| intern_u32_column(data, n_rows, n_cols, width, j))
            .collect::<OchreResult<Vec<_>>>()?;
        self.fit_transform_interned(interned, n_rows)
    }

    pub fn transform_utf32(
        &self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> OchreResult<Matrix> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if n_cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: n_cols,
                expected: n_features,
            });
        }
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let stride = n_cols * width;
        let encoded_cols: Vec<Vec<f64>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let patterns = build_u32_patterns(&categories[j]);
                let mut map: HashMap<&[u32], usize> = HashMap::with_capacity(patterns.len());
                for (idx, opt_p) in patterns.iter().enumerate() {
                    if let Some(p) = opt_p {
                        map.insert(p.as_slice(), idx);
                    }
                }
                let codes = &self.cat_to_code_[j];
                let col_off = j * width;
                let mut out_col = Vec::with_capacity(n_rows);
                for r in 0..n_rows {
                    let start = r * stride + col_off;
                    let cell = trim_u32_cell(&data[start..start + width]);
                    if let Some(&cat_idx) = map.get(cell) {
                        out_col.push(codes[cat_idx] as f64);
                    } else if self.handle_unknown == "use_encoded_value" {
                        out_col.push(self.unknown_value.unwrap_or(f64::NAN));
                    } else {
                        let key = u32_cell_to_string(cell)?;
                        return Err(OchreError::InvalidParameter(format!(
                            "Found unknown category '{}' in column {} during transform ( >_< )",
                            key, j
                        )));
                    }
                }
                Ok(out_col)
            })
            .collect::<OchreResult<Vec<_>>>()?;

        let mut out = vec![0.0; n_rows * n_features];
        for r in 0..n_rows {
            for j in 0..n_features {
                out[r * n_features + j] = encoded_cols[j][r];
            }
        }
        Matrix::from_vec(n_rows, n_features, out)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> OchreResult<Vec<Vec<Option<CatValue>>>> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: n_features,
            });
        }
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let mut rev: Vec<Vec<Option<CatValue>>> = Vec::with_capacity(n_features);
        for (j, cat_col) in categories.iter().enumerate().take(n_features) {
            let max_code = self.cat_to_code_[j].iter().copied().max().unwrap_or(0);
            let mut rmap = vec![None; max_code + 1];
            for (cat_i, &code) in self.cat_to_code_[j].iter().enumerate() {
                if self.infreq_code_[j] == Some(code) {
                    rmap[code] = None;
                } else {
                    rmap[code] = Some(cat_col[cat_i].clone());
                }
            }
            rev.push(rmap);
        }

        let mut out = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut row = Vec::with_capacity(n_features);
            for (j, rev_col) in rev.iter().enumerate().take(n_features) {
                let v = x.data[r * n_features + j];
                if v.is_nan()
                    || (self.encoded_missing_value.is_finite() && v == self.encoded_missing_value)
                {
                    row.push(None);
                    continue;
                }
                if let Some(uv) = self.unknown_value
                    && !uv.is_nan()
                    && v == uv
                {
                    row.push(None);
                    continue;
                }
                let idx = v.round() as isize;
                if idx >= 0 && (idx as usize) < rev_col.len() {
                    row.push(rev_col[idx as usize].clone());
                } else {
                    row.push(None);
                }
            }
            out.push(row);
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_ORDINAL_ENCODER);
        write_str(&mut buf, &self.handle_unknown);
        write_opt_f64(&mut buf, self.unknown_value);
        write_f64(&mut buf, self.encoded_missing_value);
        write_opt_f64(&mut buf, self.min_frequency);
        write_opt_usize(&mut buf, self.max_categories);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        match &self.categories_ {
            Some(cats) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cats.len());
                for cvec in cats {
                    write_usize(&mut buf, cvec.len());
                    for c in cvec {
                        c.write_to(&mut buf);
                    }
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.infrequent_indices_ {
            Some(inf) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, inf.len());
                for v in inf {
                    write_usize_vec(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_usize(&mut buf, self.cat_to_code_.len());
        for c in &self.cat_to_code_ {
            write_usize_vec(&mut buf, c);
        }
        write_usize(&mut buf, self.infreq_code_.len());
        for ic in &self.infreq_code_ {
            write_opt_usize(&mut buf, *ic);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_ORDINAL_ENCODER)?;
        let handle_unknown = r.read_str()?;
        let unknown_value = r.read_opt_f64()?;
        let encoded_missing_value = r.read_f64()?;
        let min_frequency = r.read_opt_f64()?;
        let max_categories = r.read_opt_usize()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let categories_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cats = Vec::with_capacity(n);
            for _ in 0..n {
                let m = r.read_usize()?;
                let mut cv = Vec::with_capacity(m);
                for _ in 0..m {
                    cv.push(CatValue::read_from(&mut r)?);
                }
                cats.push(cv);
            }
            Some(cats)
        } else {
            None
        };
        let infrequent_indices_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut inf = Vec::with_capacity(n);
            for _ in 0..n {
                inf.push(r.read_usize_vec()?);
            }
            Some(inf)
        } else {
            None
        };
        let c2c_len = r.read_usize()?;
        let mut cat_to_code_ = Vec::with_capacity(c2c_len);
        for _ in 0..c2c_len {
            cat_to_code_.push(r.read_usize_vec()?);
        }
        let ic_len = r.read_usize()?;
        let mut infreq_code_ = Vec::with_capacity(ic_len);
        for _ in 0..ic_len {
            infreq_code_.push(r.read_opt_usize()?);
        }
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            handle_unknown,
            unknown_value,
            encoded_missing_value,
            min_frequency,
            max_categories,
            device: dev,
            device_,
            categories_,
            infrequent_indices_,
            cat_to_code_,
            infreq_code_,
            n_features_in_,
        })
    }
}

/// Encode target labels with value between 0 and n_classes - 1 (o´∀｀o)
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<CatValue>>,
}

impl LabelEncoder {
    pub fn new(device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            device: dev,
            device_: None,
            classes_: None,
        })
    }

    pub fn fit(&mut self, y: &[CatValue]) -> OchreResult<()> {
        if y.is_empty() {
            return Err(OchreError::EmptyData);
        }
        let (classes, _) = unique_sorted_with_counts(y);
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_utf32(&mut self, data: &[u32], n_rows: usize, width: usize) -> OchreResult<()> {
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let (classes, _, _) = intern_u32_column(data, n_rows, 1, width, 0)?;
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_transform_utf32(
        &mut self,
        data: &[u32],
        n_rows: usize,
        width: usize,
    ) -> OchreResult<Vec<i64>> {
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let (classes, _, ids) = intern_u32_column(data, n_rows, 1, width, 0)?;
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(ids.into_iter().map(|i| i as i64).collect())
    }

    pub fn transform_utf32(
        &self,
        data: &[u32],
        n_rows: usize,
        width: usize,
    ) -> OchreResult<Vec<i64>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let patterns = build_u32_patterns(classes);
        let mut map: HashMap<&[u32], usize> = HashMap::with_capacity(patterns.len());
        for (idx, opt_p) in patterns.iter().enumerate() {
            if let Some(p) = opt_p {
                map.insert(p.as_slice(), idx);
            }
        }
        let mut out = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let start = r * width;
            let cell = trim_u32_cell(&data[start..start + width]);
            if let Some(&idx) = map.get(cell) {
                out.push(idx as i64);
            } else {
                let key = u32_cell_to_string(cell)?;
                return Err(OchreError::InvalidParameter(format!(
                    "y contains previously unseen label: '{}' ( >_< )",
                    key
                )));
            }
        }
        Ok(out)
    }

    pub fn transform(&self, y: &[CatValue]) -> OchreResult<Vec<usize>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let map = build_cat_map(classes);
        let mut out = Vec::with_capacity(y.len());
        for v in y {
            let idx = map.get(v).ok_or_else(|| {
                OchreError::InvalidParameter(format!(
                    "y contains previously unseen label: '{}' ( >_< )",
                    v.display_key()
                ))
            })?;
            out.push(*idx);
        }
        Ok(out)
    }

    pub fn inverse_transform(&self, y: &[usize]) -> OchreResult<Vec<CatValue>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let mut out = Vec::with_capacity(y.len());
        for &idx in y {
            let val = classes.get(idx).ok_or_else(|| {
                OchreError::InvalidParameter(format!(
                    "y contains previously unseen encoded label: {} ( >_< )",
                    idx
                ))
            })?;
            out.push(val.clone());
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LABEL_ENCODER);
        write_opt_str(&mut buf, &self.device_);
        match &self.classes_ {
            Some(cls) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cls.len());
                for c in cls {
                    c.write_to(&mut buf);
                }
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LABEL_ENCODER)?;
        let saved_device = r.read_opt_str()?;
        let classes_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cls = Vec::with_capacity(n);
            for _ in 0..n {
                cls.push(CatValue::read_from(&mut r)?);
            }
            Some(cls)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, classes_.is_some(), device)?;
        Ok(Self {
            device: dev,
            device_,
            classes_,
        })
    }
}

/// Binarize labels in a one-vs-all fashion (＾▽＾)
#[derive(Debug, Clone)]
pub struct LabelBinarizer {
    pub neg_label: i64,
    pub pos_label: i64,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<CatValue>>,
}

impl LabelBinarizer {
    pub fn new(neg_label: i64, pos_label: i64, device: Option<String>) -> OchreResult<Self> {
        if neg_label >= pos_label {
            return Err(OchreError::InvalidParameter(format!(
                "neg_label={} must be strictly less than pos_label={} ( >_< )",
                neg_label, pos_label
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            neg_label,
            pos_label,
            device: dev,
            device_: None,
            classes_: None,
        })
    }

    pub fn fit(&mut self, y: &[CatValue]) -> OchreResult<()> {
        if y.is_empty() {
            return Err(OchreError::EmptyData);
        }
        let (classes, _) = unique_sorted_with_counts(y);
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_with_classes(&mut self, classes: Vec<CatValue>) -> OchreResult<()> {
        if classes.is_empty() {
            return Err(OchreError::EmptyData);
        }
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    fn binarize_class_ids(
        &self,
        ids: &[Option<usize>],
        n_classes: usize,
    ) -> (usize, usize, Vec<i64>) {
        let n_cols = if n_classes <= 2 { 1 } else { n_classes };
        let n_rows = ids.len();
        let mut out = vec![self.neg_label; n_rows * n_cols];
        for (r, opt_idx) in ids.iter().enumerate() {
            if let Some(idx) = opt_idx {
                if n_classes == 1 {
                    out[r] = self.pos_label;
                } else if n_classes == 2 {
                    out[r] = if *idx == 1 {
                        self.pos_label
                    } else {
                        self.neg_label
                    };
                } else {
                    out[r * n_cols + *idx] = self.pos_label;
                }
            }
        }
        (n_rows, n_cols, out)
    }

    pub fn fit_utf32(&mut self, data: &[u32], n_rows: usize, width: usize) -> OchreResult<()> {
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let (classes, _, _) = intern_u32_column(data, n_rows, 1, width, 0)?;
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_transform_utf32(
        &mut self,
        data: &[u32],
        n_rows: usize,
        width: usize,
    ) -> OchreResult<(usize, usize, Vec<i64>)> {
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let (classes, _, row_ids) = intern_u32_column(data, n_rows, 1, width, 0)?;
        let n_classes = classes.len();
        self.classes_ = Some(classes);
        self.device_ = Some(self.device.to_string());
        let opt_ids: Vec<Option<usize>> = row_ids.into_iter().map(Some).collect();
        Ok(self.binarize_class_ids(&opt_ids, n_classes))
    }

    pub fn transform_utf32(
        &self,
        data: &[u32],
        n_rows: usize,
        width: usize,
    ) -> OchreResult<(usize, usize, Vec<i64>)> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        if n_rows == 0 || width == 0 {
            return Err(OchreError::EmptyData);
        }
        let patterns = build_u32_patterns(classes);
        let mut map: HashMap<&[u32], usize> = HashMap::with_capacity(patterns.len());
        for (idx, opt_p) in patterns.iter().enumerate() {
            if let Some(p) = opt_p {
                map.insert(p.as_slice(), idx);
            }
        }
        let ids: Vec<Option<usize>> = (0..n_rows)
            .map(|r| {
                let start = r * width;
                let cell = trim_u32_cell(&data[start..start + width]);
                map.get(cell).copied()
            })
            .collect();
        Ok(self.binarize_class_ids(&ids, classes.len()))
    }

    pub fn transform(&self, y: &[CatValue]) -> OchreResult<(usize, usize, Vec<i64>)> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes.len();
        let n_cols = if n_classes <= 2 { 1 } else { n_classes };
        let n_rows = y.len();
        let map = build_cat_map(classes);
        let mut out = vec![self.neg_label; n_rows * n_cols];
        for (r, v) in y.iter().enumerate() {
            if let Some(&idx) = map.get(v) {
                if n_classes == 1 {
                    out[r] = self.pos_label;
                } else if n_classes == 2 {
                    out[r] = if idx == 1 {
                        self.pos_label
                    } else {
                        self.neg_label
                    };
                } else {
                    out[r * n_cols + idx] = self.pos_label;
                }
            }
        }
        Ok((n_rows, n_cols, out))
    }

    pub fn inverse_transform(
        &self,
        y: &Matrix,
        threshold: Option<f64>,
    ) -> OchreResult<Vec<CatValue>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_classes = classes.len();
        let thresh = threshold.unwrap_or_else(|| 0.5 * ((self.pos_label + self.neg_label) as f64));
        let mut out = Vec::with_capacity(y.rows);

        if n_classes <= 2 {
            if y.cols != 1 {
                return Err(OchreError::FeatureCountMismatch {
                    features: y.cols,
                    expected: 1,
                });
            }
            for r in 0..y.rows {
                let v = y.data[r];
                if n_classes == 1 {
                    out.push(classes[0].clone());
                } else {
                    let idx = usize::from(v >= thresh);
                    out.push(classes[idx].clone());
                }
            }
        } else {
            if y.cols != n_classes {
                return Err(OchreError::FeatureCountMismatch {
                    features: y.cols,
                    expected: n_classes,
                });
            }
            for r in 0..y.rows {
                let row = &y.data[r * n_classes..(r + 1) * n_classes];
                let mut best_i = 0;
                let mut best_v = row[0];
                for (i, &val) in row.iter().enumerate().skip(1) {
                    if val > best_v {
                        best_v = val;
                        best_i = i;
                    }
                }
                out.push(classes[best_i].clone());
            }
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_LABEL_BINARIZER);
        write_i64(&mut buf, self.neg_label);
        write_i64(&mut buf, self.pos_label);
        write_opt_str(&mut buf, &self.device_);
        match &self.classes_ {
            Some(cls) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cls.len());
                for c in cls {
                    c.write_to(&mut buf);
                }
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LABEL_BINARIZER)?;
        let neg_label = r.read_i64()?;
        let pos_label = r.read_i64()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cls = Vec::with_capacity(n);
            for _ in 0..n {
                cls.push(CatValue::read_from(&mut r)?);
            }
            Some(cls)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, classes_.is_some(), device)?;
        Ok(Self {
            neg_label,
            pos_label,
            device: dev,
            device_,
            classes_,
        })
    }
}

/// Transform between iterable of iterables and a multilabel format (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct MultiLabelBinarizer {
    pub device: DeviceType,
    pub device_: Option<String>,
    pub classes_: Option<Vec<CatValue>>,
}

impl MultiLabelBinarizer {
    pub fn new(device: Option<String>) -> OchreResult<Self> {
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            device: dev,
            device_: None,
            classes_: None,
        })
    }

    pub fn fit(
        &mut self,
        y: &[Vec<CatValue>],
        explicit_classes: Option<Vec<CatValue>>,
    ) -> OchreResult<()> {
        if let Some(cls) = explicit_classes {
            self.classes_ = Some(cls);
        } else {
            let mut uniq: HashMap<CatValue, ()> = HashMap::new();
            for row in y {
                for item in row {
                    uniq.insert(item.clone(), ());
                }
            }
            let mut cls: Vec<CatValue> = uniq.into_keys().collect();
            cls.sort_by(|a, b| a.sort_cmp(b));
            self.classes_ = Some(cls);
        }
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn transform(&self, y: &[Vec<CatValue>]) -> OchreResult<(usize, usize, Vec<i64>)> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_cols = classes.len();
        let n_rows = y.len();
        let map = build_cat_map(classes);
        let mut out = vec![0i64; n_rows * n_cols];
        for (r, row) in y.iter().enumerate() {
            for item in row {
                if let Some(&idx) = map.get(item) {
                    out[r * n_cols + idx] = 1;
                }
            }
        }
        Ok((n_rows, n_cols, out))
    }

    pub fn inverse_transform(&self, yt: &Matrix) -> OchreResult<Vec<Vec<CatValue>>> {
        let classes = self.classes_.as_ref().ok_or(OchreError::NotFitted)?;
        if yt.cols != classes.len() {
            return Err(OchreError::FeatureCountMismatch {
                features: yt.cols,
                expected: classes.len(),
            });
        }
        let mut out = Vec::with_capacity(yt.rows);
        for r in 0..yt.rows {
            let mut labels = Vec::new();
            for (c, cls_val) in classes.iter().enumerate() {
                if yt.data[r * yt.cols + c] > 0.0 {
                    labels.push(cls_val.clone());
                }
            }
            out.push(labels);
        }
        Ok(out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MULTILABEL_BINARIZER);
        write_opt_str(&mut buf, &self.device_);
        match &self.classes_ {
            Some(cls) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cls.len());
                for c in cls {
                    c.write_to(&mut buf);
                }
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_MULTILABEL_BINARIZER)?;
        let saved_device = r.read_opt_str()?;
        let classes_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cls = Vec::with_capacity(n);
            for _ in 0..n {
                cls.push(CatValue::read_from(&mut r)?);
            }
            Some(cls)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, classes_.is_some(), device)?;
        Ok(Self {
            device: dev,
            device_,
            classes_,
        })
    }
}

/// Target encoding for categorical features using empirical Bayes shrinkage (＾▽＾)
#[derive(Debug, Clone)]
pub struct TargetEncoder {
    pub smooth: Option<f64>, // None means "auto"
    pub cv: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
    pub target_type: String,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub categories_: Option<Vec<Vec<CatValue>>>,
    pub encodings_: Option<Vec<Vec<f64>>>,
    pub target_mean_: Option<Vec<f64>>,
    pub classes_: Option<Vec<CatValue>>,
    pub target_type_: Option<String>,
    pub n_features_in_: Option<usize>,
}

fn fit_single_col_encoding(
    ord_col: &[Option<usize>],
    n_cats: usize,
    y_col: &[f64],
    y_mean: f64,
    y_var: f64,
    smooth: Option<f64>,
) -> Vec<f64> {
    let mut counts = vec![0usize; n_cats];
    let mut sums = vec![0.0; n_cats];
    for (i, &opt_c) in ord_col.iter().enumerate() {
        if let Some(c) = opt_c {
            counts[c] += 1;
            sums[c] += y_col[i];
        }
    }
    let mut means = vec![y_mean; n_cats];
    for c in 0..n_cats {
        if counts[c] > 0 {
            means[c] = sums[c] / (counts[c] as f64);
        }
    }

    match smooth {
        Some(m) => (0..n_cats)
            .map(|c| {
                let nc = counts[c] as f64;
                if nc == 0.0 {
                    y_mean
                } else {
                    (nc * means[c] + m * y_mean) / (nc + m)
                }
            })
            .collect(),
        None => {
            let mut ss = vec![0.0; n_cats];
            for (i, &opt_c) in ord_col.iter().enumerate() {
                if let Some(c) = opt_c {
                    let d = y_col[i] - means[c];
                    ss[c] += d * d;
                }
            }
            (0..n_cats)
                .map(|c| {
                    let nc = counts[c] as f64;
                    if nc == 0.0 || y_var <= 0.0 {
                        y_mean
                    } else {
                        let vc = ss[c] / nc;
                        let lam = (nc * y_var) / (nc * y_var + vc);
                        lam * means[c] + (1.0 - lam) * y_mean
                    }
                })
                .collect()
        }
    }
}

impl TargetEncoder {
    pub fn new(
        smooth: Option<f64>,
        cv: usize,
        shuffle: bool,
        random_state: Option<u64>,
        target_type: String,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if cv < 2 {
            return Err(OchreError::InvalidCvFolds(cv));
        }
        if let Some(s) = smooth
            && (s < 0.0 || !s.is_finite())
        {
            return Err(OchreError::InvalidParameter(
                "smooth must be 'auto' or a non-negative finite float ( >_< )".to_string(),
            ));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            smooth,
            cv,
            shuffle,
            random_state,
            target_type,
            device: dev,
            device_: None,
            categories_: None,
            encodings_: None,
            target_mean_: None,
            classes_: None,
            target_type_: None,
            n_features_in_: None,
        })
    }

    pub fn fit_interned(
        &mut self,
        categories: Vec<Vec<CatValue>>,
        ord_cols: &[Vec<Option<usize>>],
        n_samples: usize,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
    ) -> OchreResult<()> {
        let n_features = categories.len();
        let n_targets = y_cols.len();

        let mut target_means = Vec::with_capacity(n_targets);
        let mut target_vars = Vec::with_capacity(n_targets);
        for yc in y_cols {
            let mean = yc.iter().sum::<f64>() / (n_samples as f64);
            let var = yc.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n_samples as f64);
            target_means.push(mean);
            target_vars.push(var);
        }

        let mut encodings = Vec::with_capacity(n_features * n_targets);
        for j in 0..n_features {
            let n_cats = categories[j].len();
            for t in 0..n_targets {
                encodings.push(fit_single_col_encoding(
                    &ord_cols[j],
                    n_cats,
                    &y_cols[t],
                    target_means[t],
                    target_vars[t],
                    self.smooth,
                ));
            }
        }

        self.categories_ = Some(categories);
        self.encodings_ = Some(encodings);
        self.target_mean_ = Some(target_means);
        self.classes_ = classes;
        self.target_type_ = Some(inferred_target_type);
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_interned(
        &mut self,
        categories: Vec<Vec<CatValue>>,
        ord_cols: Vec<Vec<Option<usize>>>,
        n_samples: usize,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
        fold_ids: &[usize],
    ) -> OchreResult<Matrix> {
        let n_features = categories.len();
        self.fit_interned(
            categories,
            &ord_cols,
            n_samples,
            y_cols,
            classes,
            inferred_target_type,
        )?;
        let n_targets = y_cols.len();
        let categories_ref = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_folds = fold_ids.iter().copied().max().unwrap_or(0) + 1;
        let out_cols = n_features * n_targets;
        let mut out = vec![0.0; n_samples * out_cols];

        for fold in 0..n_folds {
            let train_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] != fold).collect();
            let val_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] == fold).collect();
            if train_idx.is_empty() || val_idx.is_empty() {
                continue;
            }
            for (t, y_col) in y_cols.iter().enumerate().take(n_targets) {
                let y_tr: Vec<f64> = train_idx.iter().map(|&i| y_col[i]).collect();
                let y_mean = y_tr.iter().sum::<f64>() / (y_tr.len() as f64);
                let y_var = y_tr
                    .iter()
                    .map(|&v| (v - y_mean) * (v - y_mean))
                    .sum::<f64>()
                    / (y_tr.len() as f64);
                for j in 0..n_features {
                    let n_cats = categories_ref[j].len();
                    let ord_tr: Vec<Option<usize>> =
                        train_idx.iter().map(|&i| ord_cols[j][i]).collect();
                    let enc =
                        fit_single_col_encoding(&ord_tr, n_cats, &y_tr, y_mean, y_var, self.smooth);
                    let col_out_idx = j * n_targets + t;
                    for &vi in &val_idx {
                        out[vi * out_cols + col_out_idx] = match ord_cols[j][vi] {
                            Some(cat_i) => enc[cat_i],
                            None => y_mean,
                        };
                    }
                }
            }
        }
        Matrix::from_vec(n_samples, out_cols, out)
    }

    pub fn transform_ord_cols(
        &self,
        ord_cols: &[Vec<Option<usize>>],
        n_samples: usize,
    ) -> OchreResult<Matrix> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        let encodings = self.encodings_.as_ref().ok_or(OchreError::NotFitted)?;
        let target_means = self.target_mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_targets = target_means.len();
        let out_cols = n_features * n_targets;
        let mut out = vec![0.0; n_samples * out_cols];
        for j in 0..n_features {
            for (r, &opt_c) in ord_cols[j].iter().enumerate() {
                for t in 0..n_targets {
                    let enc_vec = &encodings[j * n_targets + t];
                    out[r * out_cols + j * n_targets + t] = match opt_c {
                        Some(cat_i) => enc_vec[cat_i],
                        None => target_means[t],
                    };
                }
            }
        }
        Matrix::from_vec(n_samples, out_cols, out)
    }

    pub fn transform_utf32(
        &self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
    ) -> OchreResult<Matrix> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if n_cols != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: n_cols,
                expected: n_features,
            });
        }
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let stride = n_cols * width;
        let ord_cols: Vec<Vec<Option<usize>>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let patterns = build_u32_patterns(&categories[j]);
                let mut map: HashMap<&[u32], usize> = HashMap::with_capacity(patterns.len());
                for (idx, opt_p) in patterns.iter().enumerate() {
                    if let Some(p) = opt_p {
                        map.insert(p.as_slice(), idx);
                    }
                }
                let col_off = j * width;
                (0..n_rows)
                    .map(|r| {
                        let start = r * stride + col_off;
                        let cell = trim_u32_cell(&data[start..start + width]);
                        map.get(cell).copied()
                    })
                    .collect()
            })
            .collect();
        self.transform_ord_cols(&ord_cols, n_rows)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_utf32_fast(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
    ) -> OchreResult<()> {
        let per_col = (0..n_cols)
            .into_par_iter()
            .map(|j| intern_u32_column(data, n_rows, n_cols, width, j))
            .collect::<OchreResult<Vec<_>>>()?;
        let mut categories = Vec::with_capacity(n_cols);
        let mut ord_cols = Vec::with_capacity(n_cols);
        for (cats, _, rids) in per_col {
            categories.push(cats);
            ord_cols.push(rids.into_iter().map(Some).collect());
        }
        self.fit_interned(
            categories,
            &ord_cols,
            n_rows,
            y_cols,
            classes,
            inferred_target_type,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_utf32_fast(
        &mut self,
        data: &[u32],
        n_rows: usize,
        n_cols: usize,
        width: usize,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
        fold_ids: &[usize],
    ) -> OchreResult<Matrix> {
        let per_col = (0..n_cols)
            .into_par_iter()
            .map(|j| intern_u32_column(data, n_rows, n_cols, width, j))
            .collect::<OchreResult<Vec<_>>>()?;
        let mut categories = Vec::with_capacity(n_cols);
        let mut ord_cols = Vec::with_capacity(n_cols);
        for (cats, _, rids) in per_col {
            categories.push(cats);
            ord_cols.push(rids.into_iter().map(Some).collect());
        }
        self.fit_transform_interned(
            categories,
            ord_cols,
            n_rows,
            y_cols,
            classes,
            inferred_target_type,
            fold_ids,
        )
    }

    pub fn fit_from_precomputed(
        &mut self,
        cols_data: &[Vec<CatValue>],
        explicit_categories: Option<&[Vec<CatValue>]>,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
    ) -> OchreResult<()> {
        if cols_data.is_empty() || cols_data[0].is_empty() {
            return Err(OchreError::EmptyData);
        }
        let n_features = cols_data.len();
        let n_samples = cols_data[0].len();
        let n_targets = y_cols.len();

        let categories: Vec<Vec<CatValue>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                if let Some(ec) = explicit_categories {
                    ec[j].clone()
                } else {
                    unique_sorted_with_counts(&cols_data[j]).0
                }
            })
            .collect();

        let maps: Vec<HashMap<CatValue, usize>> =
            categories.iter().map(|c| build_cat_map(c)).collect();

        let ord_cols: Vec<Vec<Option<usize>>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                cols_data[j]
                    .iter()
                    .map(|v| maps[j].get(v).copied())
                    .collect()
            })
            .collect();

        let mut target_means = Vec::with_capacity(n_targets);
        let mut target_vars = Vec::with_capacity(n_targets);
        for yc in y_cols {
            let mean = yc.iter().sum::<f64>() / (n_samples as f64);
            let var = yc.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n_samples as f64);
            target_means.push(mean);
            target_vars.push(var);
        }

        let mut encodings = Vec::with_capacity(n_features * n_targets);
        for j in 0..n_features {
            let n_cats = categories[j].len();
            for t in 0..n_targets {
                encodings.push(fit_single_col_encoding(
                    &ord_cols[j],
                    n_cats,
                    &y_cols[t],
                    target_means[t],
                    target_vars[t],
                    self.smooth,
                ));
            }
        }

        self.categories_ = Some(categories);
        self.encodings_ = Some(encodings);
        self.target_mean_ = Some(target_means);
        self.classes_ = classes;
        self.target_type_ = Some(inferred_target_type);
        self.n_features_in_ = Some(n_features);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    pub fn fit_transform_with_folds(
        &mut self,
        cols_data: &[Vec<CatValue>],
        explicit_categories: Option<&[Vec<CatValue>]>,
        y_cols: &[Vec<f64>],
        classes: Option<Vec<CatValue>>,
        inferred_target_type: String,
        fold_ids: &[usize],
    ) -> OchreResult<Matrix> {
        self.fit_from_precomputed(
            cols_data,
            explicit_categories,
            y_cols,
            classes,
            inferred_target_type,
        )?;
        let n_features = cols_data.len();
        let n_samples = cols_data[0].len();
        let n_targets = y_cols.len();
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let maps: Vec<HashMap<CatValue, usize>> =
            categories.iter().map(|c| build_cat_map(c)).collect();
        let ord_cols: Vec<Vec<Option<usize>>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                cols_data[j]
                    .iter()
                    .map(|v| maps[j].get(v).copied())
                    .collect()
            })
            .collect();

        let n_folds = fold_ids.iter().copied().max().unwrap_or(0) + 1;
        let out_cols = n_features * n_targets;
        let mut out = vec![0.0; n_samples * out_cols];

        for fold in 0..n_folds {
            let train_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] != fold).collect();
            let val_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] == fold).collect();
            if train_idx.is_empty() || val_idx.is_empty() {
                continue;
            }
            for (t, y_col) in y_cols.iter().enumerate().take(n_targets) {
                let y_tr: Vec<f64> = train_idx.iter().map(|&i| y_col[i]).collect();
                let y_mean = y_tr.iter().sum::<f64>() / (y_tr.len() as f64);
                let y_var = y_tr
                    .iter()
                    .map(|&v| (v - y_mean) * (v - y_mean))
                    .sum::<f64>()
                    / (y_tr.len() as f64);
                for j in 0..n_features {
                    let n_cats = categories[j].len();
                    let ord_tr: Vec<Option<usize>> =
                        train_idx.iter().map(|&i| ord_cols[j][i]).collect();
                    let enc =
                        fit_single_col_encoding(&ord_tr, n_cats, &y_tr, y_mean, y_var, self.smooth);
                    let col_out_idx = j * n_targets + t;
                    for &vi in &val_idx {
                        out[vi * out_cols + col_out_idx] = match ord_cols[j][vi] {
                            Some(cat_i) => enc[cat_i],
                            None => y_mean,
                        };
                    }
                }
            }
        }
        Matrix::from_vec(n_samples, out_cols, out)
    }

    pub fn transform(&self, cols_data: &[Vec<CatValue>]) -> OchreResult<Matrix> {
        let n_features = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if cols_data.len() != n_features {
            return Err(OchreError::FeatureCountMismatch {
                features: cols_data.len(),
                expected: n_features,
            });
        }
        let categories = self.categories_.as_ref().ok_or(OchreError::NotFitted)?;
        let encodings = self.encodings_.as_ref().ok_or(OchreError::NotFitted)?;
        let target_means = self.target_mean_.as_ref().ok_or(OchreError::NotFitted)?;
        let n_targets = target_means.len();
        let n_samples = cols_data[0].len();
        let out_cols = n_features * n_targets;

        let maps: Vec<HashMap<CatValue, usize>> =
            categories.iter().map(|c| build_cat_map(c)).collect();

        let mut out = vec![0.0; n_samples * out_cols];
        for j in 0..n_features {
            let map = &maps[j];
            for (r, v) in cols_data[j].iter().enumerate() {
                let opt_c = map.get(v).copied();
                for t in 0..n_targets {
                    let enc_vec = &encodings[j * n_targets + t];
                    out[r * out_cols + j * n_targets + t] = match opt_c {
                        Some(cat_i) => enc_vec[cat_i],
                        None => target_means[t],
                    };
                }
            }
        }
        Matrix::from_vec(n_samples, out_cols, out)
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_TARGET_ENCODER);
        write_opt_f64(&mut buf, self.smooth);
        write_usize(&mut buf, self.cv);
        write_bool(&mut buf, self.shuffle);
        write_opt_usize(&mut buf, self.random_state.map(|v| v as usize));
        write_str(&mut buf, &self.target_type);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_str(&mut buf, &self.target_type_);
        match &self.target_mean_ {
            Some(tm) => {
                write_bool(&mut buf, true);
                write_f64_vec(&mut buf, tm);
            }
            None => write_bool(&mut buf, false),
        }
        match &self.categories_ {
            Some(cats) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cats.len());
                for cvec in cats {
                    write_usize(&mut buf, cvec.len());
                    for c in cvec {
                        c.write_to(&mut buf);
                    }
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.encodings_ {
            Some(encs) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, encs.len());
                for e in encs {
                    write_f64_vec(&mut buf, e);
                }
            }
            None => write_bool(&mut buf, false),
        }
        match &self.classes_ {
            Some(cls) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, cls.len());
                for c in cls {
                    c.write_to(&mut buf);
                }
            }
            None => write_bool(&mut buf, false),
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_TARGET_ENCODER)?;
        let smooth = r.read_opt_f64()?;
        let cv = r.read_usize()?;
        let shuffle = r.read_bool()?;
        let random_state = r.read_opt_usize()?.map(|v| v as u64);
        let target_type = r.read_str()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let target_type_ = r.read_opt_str()?;
        let target_mean_ = if r.read_bool()? {
            Some(r.read_f64_vec()?)
        } else {
            None
        };
        let categories_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cats = Vec::with_capacity(n);
            for _ in 0..n {
                let m = r.read_usize()?;
                let mut cv_vec = Vec::with_capacity(m);
                for _ in 0..m {
                    cv_vec.push(CatValue::read_from(&mut r)?);
                }
                cats.push(cv_vec);
            }
            Some(cats)
        } else {
            None
        };
        let encodings_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut encs = Vec::with_capacity(n);
            for _ in 0..n {
                encs.push(r.read_f64_vec()?);
            }
            Some(encs)
        } else {
            None
        };
        let classes_ = if r.read_bool()? {
            let n = r.read_usize()?;
            let mut cls = Vec::with_capacity(n);
            for _ in 0..n {
                cls.push(CatValue::read_from(&mut r)?);
            }
            Some(cls)
        } else {
            None
        };
        let (dev, device_) = resolve_loaded_device(saved_device, n_features_in_.is_some(), device)?;
        Ok(Self {
            smooth,
            cv,
            shuffle,
            random_state,
            target_type,
            device: dev,
            device_,
            categories_,
            encodings_,
            target_mean_,
            classes_,
            target_type_,
            n_features_in_,
        })
    }
}
