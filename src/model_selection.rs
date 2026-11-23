// OchreML Model Selection Splitters Module
// Implements KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, and train_test_split index generation.

use crate::error::{OchreError, OchreResult};
use crate::preprocessing::py_helpers::usize_slice_to_i64_pybytes;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        Self {
            state: seed.unwrap_or(0x9E37_79B9_7F4A_7C15),
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

    fn shuffle_slice<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = (self.next_u64() % ((i + 1) as u64)) as usize;
            slice.swap(i, j);
        }
    }
}

pub fn kfold_splits(
    n_samples: usize,
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
) -> OchreResult<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_splits < 2 {
        return Err(OchreError::InvalidParameter(format!(
            "n_splits must be at least 2, got {}",
            n_splits
        )));
    }
    if n_splits > n_samples {
        return Err(OchreError::InvalidParameter(format!(
            "Cannot have number of splits n_splits={} greater than the number of samples: n_samples={}",
            n_splits, n_samples
        )));
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        let mut rng = SplitMix64::new(random_state);
        rng.shuffle_slice(&mut indices);
    }

    let base_size = n_samples / n_splits;
    let rem = n_samples % n_splits;
    let mut folds = Vec::with_capacity(n_splits);
    let mut current = 0usize;

    for fold_idx in 0..n_splits {
        let fold_size = base_size + usize::from(fold_idx < rem);
        let start = current;
        let stop = current + fold_size;
        current = stop;

        let test_idx = indices[start..stop].to_vec();
        let mut train_idx = Vec::with_capacity(n_samples - fold_size);
        train_idx.extend_from_slice(&indices[..start]);
        train_idx.extend_from_slice(&indices[stop..]);
        folds.push((train_idx, test_idx));
    }

    Ok(folds)
}

pub fn stratified_kfold_splits(
    y: &[i64],
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
) -> OchreResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let n_samples = y.len();
    if n_splits < 2 {
        return Err(OchreError::InvalidParameter(format!(
            "n_splits must be at least 2, got {}",
            n_splits
        )));
    }
    if n_splits > n_samples {
        return Err(OchreError::InvalidParameter(format!(
            "n_splits={} cannot be greater than n_samples={}",
            n_splits, n_samples
        )));
    }

    let mut class_order: Vec<i64> = Vec::new();
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, &label) in y.iter().enumerate() {
        class_indices
            .entry(label)
            .or_insert_with(|| {
                class_order.push(label);
                Vec::new()
            })
            .push(idx);
    }
    class_order.sort_unstable();

    let mut rng = SplitMix64::new(random_state);
    let mut test_folds = vec![0usize; n_samples];

    for label in &class_order {
        if let Some(indices) = class_indices.get_mut(label) {
            if shuffle {
                rng.shuffle_slice(indices);
            }
            for (pos, &sample_idx) in indices.iter().enumerate() {
                test_folds[sample_idx] = pos % n_splits;
            }
        }
    }

    let mut folds = Vec::with_capacity(n_splits);
    for fold_idx in 0..n_splits {
        let mut train_idx = Vec::new();
        let mut test_idx = Vec::new();
        for (sample_idx, &f) in test_folds.iter().enumerate() {
            if f == fold_idx {
                test_idx.push(sample_idx);
            } else {
                train_idx.push(sample_idx);
            }
        }
        folds.push((train_idx, test_idx));
    }

    Ok(folds)
}

pub fn group_kfold_splits(
    groups: &[i64],
    n_splits: usize,
) -> OchreResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let n_samples = groups.len();
    if n_splits < 2 {
        return Err(OchreError::InvalidParameter(format!(
            "n_splits must be at least 2, got {}",
            n_splits
        )));
    }
    if n_samples == 0 {
        return Err(OchreError::EmptyData);
    }

    let mut group_order: Vec<i64> = Vec::new();
    let mut group_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, &g) in groups.iter().enumerate() {
        group_map
            .entry(g)
            .or_insert_with(|| {
                group_order.push(g);
                Vec::new()
            })
            .push(idx);
    }

    let n_groups = group_order.len();
    if n_splits > n_groups {
        return Err(OchreError::InvalidParameter(format!(
            "Cannot have number of splits n_splits={} greater than the number of groups: {}",
            n_splits, n_groups
        )));
    }

    // Sort groups by descending size (stable tie-breaker by group id)
    group_order.sort_by(|a, b| {
        let len_a = group_map.get(a).map_or(0, |v| v.len());
        let len_b = group_map.get(b).map_or(0, |v| v.len());
        len_b.cmp(&len_a).then_with(|| a.cmp(b))
    });

    let mut fold_counts = vec![0usize; n_splits];
    let mut sample_to_fold = vec![0usize; n_samples];

    for g in group_order {
        let mut best_fold = 0usize;
        let mut min_count = usize::MAX;
        for (f, &cnt) in fold_counts.iter().enumerate() {
            if cnt < min_count {
                min_count = cnt;
                best_fold = f;
            }
        }
        if let Some(indices) = group_map.get(&g) {
            fold_counts[best_fold] += indices.len();
            for &idx in indices {
                sample_to_fold[idx] = best_fold;
            }
        }
    }

    let mut folds = Vec::with_capacity(n_splits);
    for fold_idx in 0..n_splits {
        let mut train_idx = Vec::new();
        let mut test_idx = Vec::new();
        for (idx, &f) in sample_to_fold.iter().enumerate() {
            if f == fold_idx {
                test_idx.push(idx);
            } else {
                train_idx.push(idx);
            }
        }
        folds.push((train_idx, test_idx));
    }

    Ok(folds)
}

pub fn time_series_splits(
    n_samples: usize,
    n_splits: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
    gap: usize,
) -> OchreResult<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_splits < 1 {
        return Err(OchreError::InvalidParameter(format!(
            "n_splits must be at least 1, got {}",
            n_splits
        )));
    }
    let n_folds = n_splits + 1;
    if n_folds > n_samples {
        return Err(OchreError::InvalidParameter(format!(
            "Cannot have number of folds={} greater than the number of samples: {}",
            n_folds, n_samples
        )));
    }

    let resolved_test_size = match test_size {
        Some(ts) => {
            if ts == 0 {
                return Err(OchreError::InvalidParameter(
                    "test_size must be >= 1".to_string(),
                ));
            }
            ts
        }
        None => n_samples / n_folds,
    };

    if resolved_test_size == 0 {
        return Err(OchreError::InvalidParameter(
            "Computed test_size is 0".to_string(),
        ));
    }
    let needed = n_splits * resolved_test_size;
    if needed >= n_samples {
        return Err(OchreError::InvalidParameter(format!(
            "Too many splits={} for number of samples={} with test_size={}",
            n_splits, n_samples, resolved_test_size
        )));
    }

    let first_test_start = n_samples - needed;
    let mut folds = Vec::with_capacity(n_splits);
    for split_idx in 0..n_splits {
        let test_start = first_test_start + split_idx * resolved_test_size;
        let test_end = test_start + resolved_test_size;
        if test_start <= gap {
            return Err(OchreError::InvalidParameter(format!(
                "Too many splits={} for n_samples={} and gap={}",
                n_splits, n_samples, gap
            )));
        }
        let train_end = test_start - gap;
        let train_start = match max_train_size {
            Some(mts) if train_end > mts => train_end - mts,
            _ => 0,
        };
        let train_idx: Vec<usize> = (train_start..train_end).collect();
        let test_idx: Vec<usize> = (test_start..test_end).collect();
        folds.push((train_idx, test_idx));
    }

    Ok(folds)
}

pub fn compute_train_test_split_indices(
    n_samples: usize,
    n_test: usize,
    n_train: usize,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<&[i64]>,
) -> OchreResult<(Vec<usize>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(OchreError::EmptyData);
    }
    if n_test + n_train > n_samples {
        return Err(OchreError::InvalidParameter(format!(
            "The sum of test_size ({}) and train_size ({}) cannot exceed n_samples ({})",
            n_test, n_train, n_samples
        )));
    }
    if n_test == 0 || n_train == 0 {
        return Err(OchreError::InvalidParameter(
            "Both train and test sets must be non-empty".to_string(),
        ));
    }

    if let Some(labels) = stratify {
        if labels.len() != n_samples {
            return Err(OchreError::LengthMismatch {
                features: n_samples,
                targets: labels.len(),
            });
        }
        if !shuffle {
            return Err(OchreError::InvalidParameter(
                "Stratified train/test split is not supported when shuffle=False".to_string(),
            ));
        }
        let mut class_order: Vec<i64> = Vec::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
        for (idx, &lbl) in labels.iter().enumerate() {
            class_indices
                .entry(lbl)
                .or_insert_with(|| {
                    class_order.push(lbl);
                    Vec::new()
                })
                .push(idx);
        }
        class_order.sort_unstable();

        for &lbl in &class_order {
            let cnt = class_indices.get(&lbl).map_or(0, |v| v.len());
            if cnt < 2 {
                return Err(OchreError::InvalidParameter(
                    "The least populated class in y has only 1 member, which is too few for stratify"
                        .to_string(),
                ));
            }
        }

        let n_classes = class_order.len();
        if n_test < n_classes || n_train < n_classes {
            return Err(OchreError::InvalidParameter(format!(
                "test_size={} and train_size={} must be >= n_classes={}",
                n_test, n_train, n_classes
            )));
        }

        let mut rng = SplitMix64::new(random_state);
        // Compute per-class test allocations proportional to class size
        let mut class_test_counts = Vec::with_capacity(n_classes);
        let mut total_assigned_test = 0usize;
        for &lbl in &class_order {
            let cnt = class_indices.get(&lbl).map_or(0, |v| v.len());
            let exact = (cnt as f64) * (n_test as f64) / (n_samples as f64);
            let mut k = exact.round() as usize;
            if k == 0 {
                k = 1;
            }
            if k >= cnt {
                k = cnt - 1;
            }
            class_test_counts.push((k, exact - (k as f64), cnt));
            total_assigned_test += k;
        }

        // Adjust rounding differences
        while total_assigned_test < n_test {
            let mut best_i = None;
            let mut best_diff = f64::NEG_INFINITY;
            for (i, &(k, diff, cnt)) in class_test_counts.iter().enumerate() {
                if k + 1 < cnt && diff > best_diff {
                    best_diff = diff;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i {
                class_test_counts[i].0 += 1;
                class_test_counts[i].1 -= 1.0;
                total_assigned_test += 1;
            } else {
                break;
            }
        }
        while total_assigned_test > n_test {
            let mut best_i = None;
            let mut best_diff = f64::INFINITY;
            for (i, &(k, diff, _cnt)) in class_test_counts.iter().enumerate() {
                if k > 1 && diff < best_diff {
                    best_diff = diff;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i {
                class_test_counts[i].0 -= 1;
                class_test_counts[i].1 += 1.0;
                total_assigned_test -= 1;
            } else {
                break;
            }
        }

        let mut test_idx = Vec::with_capacity(n_test);
        let mut train_idx = Vec::with_capacity(n_samples - n_test);
        for (i, &lbl) in class_order.iter().enumerate() {
            if let Some(indices) = class_indices.get_mut(&lbl) {
                rng.shuffle_slice(indices);
                let k_test = class_test_counts[i].0;
                test_idx.extend_from_slice(&indices[..k_test]);
                train_idx.extend_from_slice(&indices[k_test..]);
            }
        }
        rng.shuffle_slice(&mut test_idx);
        rng.shuffle_slice(&mut train_idx);
        train_idx.truncate(n_train);
        return Ok((train_idx, test_idx));
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        let mut rng = SplitMix64::new(random_state);
        rng.shuffle_slice(&mut indices);
        let test_idx = indices[..n_test].to_vec();
        let train_idx = indices[n_test..(n_test + n_train)].to_vec();
        Ok((train_idx, test_idx))
    } else {
        let train_idx = indices[..n_train].to_vec();
        let test_idx = indices[n_train..(n_train + n_test)].to_vec();
        Ok((train_idx, test_idx))
    }
}

fn folds_to_py<'py>(
    py: Python<'py>,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
    let mut out = Vec::with_capacity(folds.len());
    for (tr, te) in folds {
        let tr_b = usize_slice_to_i64_pybytes(py, &tr)?;
        let te_b = usize_slice_to_i64_pybytes(py, &te)?;
        out.push((tr_b, te_b));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits = 5, shuffle = false, random_state = None))]
pub fn kfold_split<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
    let folds = kfold_splits(n_samples, n_splits, shuffle, random_state)?;
    folds_to_py(py, folds)
}

#[pyfunction]
#[pyo3(signature = (y, n_splits = 5, shuffle = false, random_state = None))]
pub fn stratified_kfold_split<'py>(
    py: Python<'py>,
    y: Vec<i64>,
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
    let folds = stratified_kfold_splits(&y, n_splits, shuffle, random_state)?;
    folds_to_py(py, folds)
}

#[pyfunction]
#[pyo3(signature = (groups, n_splits = 5))]
pub fn group_kfold_split<'py>(
    py: Python<'py>,
    groups: Vec<i64>,
    n_splits: usize,
) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
    let folds = group_kfold_splits(&groups, n_splits)?;
    folds_to_py(py, folds)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits = 5, max_train_size = None, test_size = None, gap = 0))]
pub fn time_series_split<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_splits: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
    gap: usize,
) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
    let folds = time_series_splits(n_samples, n_splits, max_train_size, test_size, gap)?;
    folds_to_py(py, folds)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_test, n_train, shuffle = true, random_state = None, stratify = None))]
pub fn train_test_split_indices<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_test: usize,
    n_train: usize,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<Vec<i64>>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let (tr, te) = compute_train_test_split_indices(
        n_samples,
        n_test,
        n_train,
        shuffle,
        random_state,
        stratify.as_deref(),
    )?;
    let tr_b = usize_slice_to_i64_pybytes(py, &tr)?;
    let te_b = usize_slice_to_i64_pybytes(py, &te)?;
    Ok((tr_b, te_b))
}

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kfold_split, m)?)?;
    m.add_function(wrap_pyfunction!(stratified_kfold_split, m)?)?;
    m.add_function(wrap_pyfunction!(group_kfold_split, m)?)?;
    m.add_function(wrap_pyfunction!(time_series_split, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split_indices, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_and_stratified_kfold() {
        let folds = match kfold_splits(10, 3, true, Some(42)) {
            Ok(f) => f,
            Err(e) => panic!("kfold_splits failed: {}", e),
        };
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0].1.len(), 4);
        assert_eq!(folds[1].1.len(), 3);
        assert_eq!(folds[2].1.len(), 3);

        let y = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let sfolds = match stratified_kfold_splits(&y, 3, false, None) {
            Ok(f) => f,
            Err(e) => panic!("stratified_kfold_splits failed: {}", e),
        };
        assert_eq!(sfolds.len(), 3);
        for (tr, te) in &sfolds {
            assert_eq!(tr.len(), 8);
            assert_eq!(te.len(), 4);
            let ones_te = te.iter().filter(|&&i| y[i] == 1).count();
            assert_eq!(ones_te, 2);
        }
    }

    #[test]
    fn test_group_kfold_and_time_series() {
        let groups = vec![1, 1, 1, 2, 2, 3, 3, 4];
        let gfolds = match group_kfold_splits(&groups, 2) {
            Ok(f) => f,
            Err(e) => panic!("group_kfold_splits failed: {}", e),
        };
        assert_eq!(gfolds.len(), 2);
        for (tr, te) in &gfolds {
            let tr_groups: std::collections::HashSet<i64> = tr.iter().map(|&i| groups[i]).collect();
            let te_groups: std::collections::HashSet<i64> = te.iter().map(|&i| groups[i]).collect();
            assert!(tr_groups.is_disjoint(&te_groups));
        }

        let ts_folds = match time_series_splits(10, 3, Some(4), Some(2), 1) {
            Ok(f) => f,
            Err(e) => panic!("time_series_splits failed: {}", e),
        };
        assert_eq!(ts_folds.len(), 3);
        assert_eq!(ts_folds[0], (vec![0, 1, 2], vec![4, 5]));
        assert_eq!(ts_folds[1], (vec![1, 2, 3, 4], vec![6, 7]));
        assert_eq!(ts_folds[2], (vec![3, 4, 5, 6], vec![8, 9]));
    }

    #[test]
    fn test_train_test_split_stratified() {
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let (tr, te) = match compute_train_test_split_indices(12, 3, 9, true, Some(7), Some(&y)) {
            Ok(res) => res,
            Err(e) => panic!("train_test_split failed: {}", e),
        };
        assert_eq!(tr.len(), 9);
        assert_eq!(te.len(), 3);
        let mut te_classes = [0usize; 3];
        for &idx in &te {
            te_classes[y[idx] as usize] += 1;
        }
        assert_eq!(te_classes, [1, 1, 1]);
    }
}
