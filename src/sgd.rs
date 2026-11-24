// OchreML Stochastic Gradient Descent (SGD) Module (＾▽＾)
// Implements SGDClassifier (hinge, log_loss, modified_huber) and
// SGDRegressor (squared_error, huber, epsilon_insensitive) with L1/L2/ElasticNet penalties.

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_f64_vec, write_header,
    write_opt_f64, write_opt_f64_vec, write_opt_matrix, write_opt_str, write_opt_u64,
    write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::linear_regression::{predict_linear_1d, predict_linear_multi};
use crate::logistic_regression::{
    build_effective_weights, extract_sorted_classes, map_labels_to_indices, validate_sample_weights,
};
use crate::matrix::{Matrix, r2_score};
use crate::primitives::ElementwiseOp;
use crate::primitives::stats_ops::apply_scalar_op;

pub const TAG_SGD_CLASSIFIER: u8 = 41;
pub const TAG_SGD_REGRESSOR: u8 = 42;

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

fn validate_penalty(penalty: &str, l1_ratio: f64) -> OchreResult<(f64, f64)> {
    if !(0.0..=1.0).contains(&l1_ratio) || !l1_ratio.is_finite() {
        return Err(OchreError::InvalidL1Ratio(l1_ratio));
    }
    match penalty {
        "l2" => Ok((0.0, 1.0)),
        "l1" => Ok((1.0, 0.0)),
        "elasticnet" => Ok((l1_ratio, 1.0 - l1_ratio)),
        "none" => Ok((0.0, 0.0)),
        _ => Err(OchreError::InvalidParameter(format!(
            "Invalid penalty '{penalty}'. Expected 'l2', 'l1', 'elasticnet', or 'none'"
        ))),
    }
}

fn validate_learning_rate(lr: &str, eta0: f64, power_t: f64) -> OchreResult<()> {
    match lr {
        "optimal" => {}
        "constant" | "invscaling" => {
            if eta0 <= 0.0 || !eta0.is_finite() {
                return Err(OchreError::InvalidParameter(format!(
                    "eta0 ({eta0}) must be > 0.0 for learning_rate='{lr}'"
                )));
            }
        }
        _ => {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid learning_rate '{lr}'. Expected 'optimal', 'constant', or 'invscaling'"
            )));
        }
    }
    if !power_t.is_finite() {
        return Err(OchreError::InvalidParameter(
            "power_t must be finite".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn compute_t0_optimal(alpha: f64, loss_name: &str) -> f64 {
    if alpha <= 0.0 {
        return 1.0;
    }
    let typw = (1.0 / alpha.sqrt()).sqrt();
    let dloss_bound: f64 = if loss_name == "modified_huber" {
        4.0
    } else {
        1.0
    };
    let init_eta0 = typw / dloss_bound;
    1.0 / (alpha * init_eta0)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LrSchedule {
    Optimal,
    Constant,
    InvScaling,
}

impl LrSchedule {
    fn from_str(lr: &str) -> Self {
        match lr {
            "constant" => Self::Constant,
            "invscaling" => Self::InvScaling,
            _ => Self::Optimal,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClfLoss {
    Hinge,
    LogLoss,
    ModifiedHuber,
}

impl ClfLoss {
    fn from_str(loss: &str) -> Self {
        match loss {
            "log_loss" => Self::LogLoss,
            "modified_huber" => Self::ModifiedHuber,
            _ => Self::Hinge,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RegLoss {
    SquaredError,
    Huber,
    EpsilonInsensitive,
}

impl RegLoss {
    fn from_str(loss: &str) -> Self {
        match loss {
            "huber" => Self::Huber,
            "epsilon_insensitive" => Self::EpsilonInsensitive,
            _ => Self::SquaredError,
        }
    }
}

#[inline(always)]
fn dot_unrolled_chunks(a: &[f64], b: &[f64]) -> f64 {
    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    let mut acc = [0.0f64; 4];
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        acc[0] += ac[0] * bc[0];
        acc[1] += ac[1] * bc[1];
        acc[2] += ac[2] * bc[2];
        acc[3] += ac[3] * bc[3];
    }
    let mut tail = 0.0;
    for (&aj, &bj) in a_rem.iter().zip(b_rem.iter()) {
        tail += aj * bj;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[inline(always)]
fn scale_slice_unrolled(w: &mut [f64], scale: f64) {
    let (w_chunks, w_rem) = w.as_chunks_mut::<4>();
    for wc in w_chunks {
        wc[0] *= scale;
        wc[1] *= scale;
        wc[2] *= scale;
        wc[3] *= scale;
    }
    for wj in w_rem {
        *wj *= scale;
    }
}

#[inline(always)]
fn axpy_sub_unrolled(w: &mut [f64], row: &[f64], scale: f64) {
    let (w_chunks, w_rem) = w.as_chunks_mut::<4>();
    let (r_chunks, r_rem) = row.as_chunks::<4>();
    for (wc, rc) in w_chunks.iter_mut().zip(r_chunks.iter()) {
        wc[0] -= scale * rc[0];
        wc[1] -= scale * rc[1];
        wc[2] -= scale * rc[2];
        wc[3] -= scale * rc[3];
    }
    for (wj, &rj) in w_rem.iter_mut().zip(r_rem.iter()) {
        *wj -= scale * rj;
    }
}

#[inline(always)]
fn step_size_sched(lr: LrSchedule, eta0: f64, power_t: f64, alpha: f64, t: f64, t0: f64) -> f64 {
    match lr {
        LrSchedule::Constant => eta0,
        LrSchedule::InvScaling => eta0 / t.powf(power_t),
        LrSchedule::Optimal => {
            if alpha <= 0.0 {
                1.0 / (1.0 + t)
            } else {
                1.0 / (alpha * (t0 + t - 1.0))
            }
        }
    }
}

#[inline(always)]
fn classifier_loss_and_dloss_enum(loss: ClfLoss, p: f64, y_pm1: f64) -> (f64, f64) {
    let z = p * y_pm1;
    match loss {
        ClfLoss::Hinge => {
            if z < 1.0 {
                (1.0 - z, -y_pm1)
            } else {
                (0.0, 0.0)
            }
        }
        ClfLoss::LogLoss => {
            if z >= 0.0 {
                let e = (-z).exp();
                (e.ln_1p(), -y_pm1 * (e / (1.0 + e)))
            } else {
                let e = z.exp();
                (-z + e.ln_1p(), -y_pm1 / (1.0 + e))
            }
        }
        ClfLoss::ModifiedHuber => {
            if z >= 1.0 {
                (0.0, 0.0)
            } else if z >= -1.0 {
                let d = 1.0 - z;
                (d * d, -2.0 * d * y_pm1)
            } else {
                (-4.0 * z, -4.0 * y_pm1)
            }
        }
    }
}

#[inline(always)]
fn regressor_loss_and_dloss_enum(loss: RegLoss, pred: f64, y: f64, epsilon: f64) -> (f64, f64) {
    let diff = pred - y;
    match loss {
        RegLoss::SquaredError => (0.5 * diff * diff, diff),
        RegLoss::Huber => {
            let abs_d = diff.abs();
            if abs_d <= epsilon {
                (0.5 * diff * diff, diff)
            } else {
                (epsilon * (abs_d - 0.5 * epsilon), epsilon * diff.signum())
            }
        }
        RegLoss::EpsilonInsensitive => {
            let abs_d = diff.abs();
            if abs_d <= epsilon {
                (0.0, 0.0)
            } else {
                (abs_d - epsilon, diff.signum())
            }
        }
    }
}

const LARGE_N_BLOCK_SHUFFLE_THRESHOLD: usize = 100_000;
const SHUFFLE_BLOCK_SIZE: usize = 512;

enum EpochOrder {
    Rows(Vec<u32>),
    Blocks {
        block_ids: Vec<u32>,
        n: usize,
        block_size: usize,
    },
}

impl EpochOrder {
    fn new(n: usize) -> Self {
        if n >= LARGE_N_BLOCK_SHUFFLE_THRESHOLD {
            let n_blocks = n.div_ceil(SHUFFLE_BLOCK_SIZE);
            Self::Blocks {
                block_ids: (0..n_blocks as u32).collect(),
                n,
                block_size: SHUFFLE_BLOCK_SIZE,
            }
        } else {
            Self::Rows((0..n as u32).collect())
        }
    }

    fn shuffle(&mut self, rng: &mut SplitMix64) {
        match self {
            Self::Rows(rows) => rng.shuffle(rows),
            Self::Blocks { block_ids, .. } => rng.shuffle(block_ids),
        }
    }

    #[inline(always)]
    fn for_each<F: FnMut(usize)>(&self, mut f: F) {
        match self {
            Self::Rows(rows) => {
                for &idx in rows {
                    f(idx as usize);
                }
            }
            Self::Blocks {
                block_ids,
                n,
                block_size,
            } => {
                for &bid in block_ids {
                    let start = (bid as usize) * block_size;
                    let end = (start + block_size).min(*n);
                    for idx in start..end {
                        f(idx);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn sgd_update_l1(
    w: &mut [f64],
    w_scale: &mut f64,
    row: &[f64],
    eta: f64,
    step_g: f64,
    l1_reg: f64,
    l2_reg: f64,
) {
    let decay = (1.0 - eta * l2_reg).max(0.0);
    if (*w_scale - 1.0).abs() > 1e-12 {
        scale_slice_unrolled(w, *w_scale);
        *w_scale = 1.0;
    }
    let thr = eta * l1_reg;
    for (wj, &rj) in w.iter_mut().zip(row.iter()) {
        let updated = *wj * decay - step_g * rj;
        *wj = apply_scalar_op(updated, ElementwiseOp::SoftThreshold { lambda: thr });
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_sgd_binary_classifier(
    x: &Matrix,
    y_pm1: &[f64],
    weights: &[f64],
    loss_name: &str,
    l1_reg: f64,
    l2_reg: f64,
    alpha: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: Option<f64>,
    shuffle: bool,
    lr: &str,
    eta0: f64,
    power_t: f64,
    rng: &mut SplitMix64,
) -> (Vec<f64>, f64, usize, f64) {
    let n = x.rows;
    let p = x.cols;
    let mut w = vec![0.0; p];
    let mut w_scale = 1.0_f64;
    let mut b = 0.0_f64;
    let mut order = EpochOrder::new(n);
    let loss_kind = ClfLoss::from_str(loss_name);
    let lr_sched = LrSchedule::from_str(lr);
    let intercept_eta_scale = if lr_sched == LrSchedule::Optimal {
        0.5
    } else {
        1.0
    };
    let t0 = compute_t0_optimal(alpha, loss_name);
    let mut t = 1.0_f64;
    let mut best_loss = f64::INFINITY;
    let mut no_improve = 0usize;
    let mut n_iter = 0usize;
    let no_l1 = l1_reg == 0.0;

    for epoch in 0..max_iter {
        n_iter = epoch + 1;
        if shuffle {
            order.shuffle(rng);
        }
        let mut epoch_loss = 0.0;

        if no_l1 {
            order.for_each(|idx| {
                let row = &x.data[idx * p..(idx + 1) * p];
                let pred = dot_unrolled_chunks(row, &w) * w_scale + b;
                let (l_val, dloss) = classifier_loss_and_dloss_enum(loss_kind, pred, y_pm1[idx]);
                let eta = step_size_sched(lr_sched, eta0, power_t, alpha, t, t0);

                let decay = (1.0 - eta * l2_reg).max(0.0);
                if decay > 1e-12 {
                    w_scale *= decay;
                    if w_scale < 1e-9 {
                        scale_slice_unrolled(&mut w, w_scale);
                        w_scale = 1.0;
                    }
                } else {
                    w.fill(0.0);
                    w_scale = 1.0;
                }

                if dloss != 0.0 {
                    let sw = weights[idx];
                    epoch_loss += sw * l_val;
                    let step_g = eta * sw * dloss;
                    axpy_sub_unrolled(&mut w, row, step_g / w_scale);
                    if fit_intercept {
                        b -= (eta * intercept_eta_scale) * sw * dloss;
                    }
                }

                t += 1.0;
            });
        } else {
            order.for_each(|idx| {
                let row = &x.data[idx * p..(idx + 1) * p];
                let pred = dot_unrolled_chunks(row, &w) * w_scale + b;
                let (l_val, dloss) = classifier_loss_and_dloss_enum(loss_kind, pred, y_pm1[idx]);
                let eta = step_size_sched(lr_sched, eta0, power_t, alpha, t, t0);

                if dloss == 0.0 {
                    sgd_update_l1(&mut w, &mut w_scale, row, eta, 0.0, l1_reg, l2_reg);
                } else {
                    let sw = weights[idx];
                    epoch_loss += sw * l_val;
                    let step_g = eta * sw * dloss;
                    sgd_update_l1(&mut w, &mut w_scale, row, eta, step_g, l1_reg, l2_reg);
                    if fit_intercept {
                        b -= (eta * intercept_eta_scale) * sw * dloss;
                    }
                }

                t += 1.0;
            });
        }

        let avg_loss = epoch_loss / (n as f64);
        if let Some(tol_val) = tol {
            if avg_loss > best_loss - tol_val {
                no_improve += 1;
                if no_improve >= 5 {
                    break;
                }
            } else {
                best_loss = avg_loss;
                no_improve = 0;
            }
        }
    }

    if (w_scale - 1.0).abs() > 1e-12 {
        scale_slice_unrolled(&mut w, w_scale);
    }
    (w, b, n_iter, t - 1.0)
}

#[inline]
fn stable_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Linear Classifier trained with Stochastic Gradient Descent (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct SGDClassifier {
    pub loss: String,
    pub penalty: String,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: Option<f64>,
    pub shuffle: bool,
    pub learning_rate: String,
    pub eta0: f64,
    pub power_t: f64,
    pub random_state: Option<u64>,
    pub class_weight_mode: Option<String>,
    pub class_weight_map: Option<Vec<(f64, f64)>>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub coef_multi_: Option<Matrix>,
    pub intercept_multi_: Option<Vec<f64>>,
    pub classes_: Vec<f64>,
    pub is_multiclass_: bool,
    pub n_iter_: usize,
    pub t_: f64,
    pub n_features_in_: Option<usize>,
}

impl SGDClassifier {
    /// Create a new SGDClassifier estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        penalty: String,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: Option<f64>,
        shuffle: bool,
        learning_rate: String,
        eta0: f64,
        power_t: f64,
        random_state: Option<u64>,
        class_weight_mode: Option<String>,
        class_weight_map: Option<Vec<(f64, f64)>>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !matches!(loss.as_str(), "hinge" | "log_loss" | "modified_huber") {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid loss '{loss}'. Expected 'hinge', 'log_loss', or 'modified_huber'"
            )));
        }
        if alpha < 0.0 || !alpha.is_finite() {
            return Err(OchreError::InvalidAlpha(alpha));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1".to_string(),
            ));
        }
        validate_penalty(&penalty, l1_ratio)?;
        validate_learning_rate(&learning_rate, eta0, power_t)?;
        let dev = parse_device(device.as_deref())?;

        Ok(Self {
            loss,
            penalty,
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            shuffle,
            learning_rate,
            eta0,
            power_t,
            random_state,
            class_weight_mode,
            class_weight_map,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            coef_multi_: None,
            intercept_multi_: None,
            classes_: Vec::new(),
            is_multiclass_: false,
            n_iter_: 0,
            t_: 0.0,
            n_features_in_: None,
        })
    }

    /// Fit SGDClassifier on X and y with optional sample_weight.
    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weights(sample_weight, x.rows)?;

        let classes = extract_sorted_classes(y);
        if classes.len() < 2 {
            return Err(OchreError::InsufficientClasses(classes.len()));
        }
        let y_idx = map_labels_to_indices(y, &classes)?;
        let n_classes = classes.len();
        let weights = build_effective_weights(
            &y_idx,
            n_classes,
            &classes,
            sample_weight,
            self.class_weight_mode.as_deref(),
            self.class_weight_map.as_deref(),
        )?;

        let (l1_frac, l2_frac) = validate_penalty(&self.penalty, self.l1_ratio)?;
        let l1_reg = self.alpha * l1_frac;
        let l2_reg = self.alpha * l2_frac;
        let mut rng = SplitMix64::new(self.random_state);
        let p = x.cols;

        if n_classes == 2 {
            let y_pm1: Vec<f64> = y_idx
                .iter()
                .map(|&i| if i == 1 { 1.0 } else { -1.0 })
                .collect();
            let (w, b, n_iter, t_total) = fit_sgd_binary_classifier(
                x,
                &y_pm1,
                &weights,
                &self.loss,
                l1_reg,
                l2_reg,
                self.alpha,
                self.fit_intercept,
                self.max_iter,
                self.tol,
                self.shuffle,
                &self.learning_rate,
                self.eta0,
                self.power_t,
                &mut rng,
            );
            self.coef_ = Some(w.clone());
            self.intercept_ = Some(b);
            self.coef_multi_ = Some(Matrix::from_vec(1, p, w)?);
            self.intercept_multi_ = Some(vec![b]);
            self.is_multiclass_ = false;
            self.n_iter_ = n_iter;
            self.t_ = t_total;
        } else {
            let mut coef_data = vec![0.0; n_classes * p];
            let mut intercepts = vec![0.0; n_classes];
            let mut max_n_iter = 0usize;
            let mut total_t = 0.0;

            for c in 0..n_classes {
                let y_pm1: Vec<f64> = y_idx
                    .iter()
                    .map(|&i| if i == c { 1.0 } else { -1.0 })
                    .collect();
                let (w, b, n_it, t_c) = fit_sgd_binary_classifier(
                    x,
                    &y_pm1,
                    &weights,
                    &self.loss,
                    l1_reg,
                    l2_reg,
                    self.alpha,
                    self.fit_intercept,
                    self.max_iter,
                    self.tol,
                    self.shuffle,
                    &self.learning_rate,
                    self.eta0,
                    self.power_t,
                    &mut rng,
                );
                coef_data[c * p..(c + 1) * p].copy_from_slice(&w);
                intercepts[c] = b;
                max_n_iter = max_n_iter.max(n_it);
                total_t += t_c;
            }
            self.coef_ = None;
            self.intercept_ = None;
            self.coef_multi_ = Some(Matrix::from_vec(n_classes, p, coef_data)?);
            self.intercept_multi_ = Some(intercepts);
            self.is_multiclass_ = true;
            self.n_iter_ = max_n_iter;
            self.t_ = total_t;
        }

        self.classes_ = classes;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Compute decision function for binary classification.
    pub fn decision_function(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if self.is_multiclass_ {
            return Err(OchreError::InvalidParameter(
                "Use decision_function_multi for multiclass SGDClassifier".to_string(),
            ));
        }
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        if !matches!(self.device, DeviceType::Cpu) && x.rows.saturating_mul(x.cols) >= 262_144 {
            let mut out = get_backend(&self.device).matvec(x, coef)?;
            if intercept != 0.0 {
                for v in &mut out {
                    *v += intercept;
                }
            }
            return Ok(out);
        }
        predict_linear_1d(x, coef, intercept)
    }

    /// Compute OvR decision function for multiclass classification.
    pub fn decision_function_multi(&self, x: &Matrix) -> OchreResult<Matrix> {
        let coef_mat = self.coef_multi_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercepts = self
            .intercept_multi_
            .as_ref()
            .ok_or(OchreError::NotFitted)?;
        predict_linear_multi(x, coef_mat, intercepts)
    }

    /// Predict class probabilities (supported for 'log_loss' and 'modified_huber').
    pub fn predict_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        if self.classes_.is_empty() {
            return Err(OchreError::NotFitted);
        }
        if self.loss == "hinge" {
            return Err(OchreError::UnsupportedPredictProbaLoss(self.loss.clone()));
        }

        if !self.is_multiclass_ {
            let scores = self.decision_function(x)?;
            let mut data = Vec::with_capacity(x.rows * 2);
            for s in scores {
                let p1 = if self.loss == "log_loss" {
                    stable_sigmoid(s)
                } else {
                    ((s + 1.0) * 0.5).clamp(0.0, 1.0)
                };
                data.push(1.0 - p1);
                data.push(p1);
            }
            return Matrix::from_vec(x.rows, 2, data);
        }

        let scores = self.decision_function_multi(x)?;
        let k = scores.cols;
        let mut data = Vec::with_capacity(x.rows * k);
        for r in 0..x.rows {
            let mut row_p = Vec::with_capacity(k);
            let mut sum = 0.0;
            for c in 0..k {
                let s = scores[(r, c)];
                let p = if self.loss == "log_loss" {
                    stable_sigmoid(s)
                } else {
                    ((s + 1.0) * 0.5).clamp(0.0, 1.0)
                };
                sum += p;
                row_p.push(p);
            }
            if sum > 0.0 {
                for p in row_p {
                    data.push(p / sum);
                }
            } else {
                let u = 1.0 / (k as f64);
                data.extend(std::iter::repeat_n(u, k));
            }
        }
        Matrix::from_vec(x.rows, k, data)
    }

    /// Predict log-probabilities.
    pub fn predict_log_proba(&self, x: &Matrix) -> OchreResult<Matrix> {
        let probs = self.predict_proba(x)?;
        let data = probs.data.into_iter().map(|p| p.max(1e-15).ln()).collect();
        Matrix::from_vec(probs.rows, probs.cols, data)
    }

    /// Predict class labels for samples in X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        if self.classes_.is_empty() {
            return Err(OchreError::NotFitted);
        }
        if !self.is_multiclass_ {
            let scores = self.decision_function(x)?;
            let c0 = self.classes_[0];
            let c1 = self.classes_[1];
            return Ok(scores
                .into_iter()
                .map(|s| if s > 0.0 { c1 } else { c0 })
                .collect());
        }

        let scores = self.decision_function_multi(x)?;
        let mut preds = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let mut best_idx = 0;
            let mut best_score = scores[(r, 0)];
            for c in 1..scores.cols {
                let s = scores[(r, c)];
                if s > best_score {
                    best_score = s;
                    best_idx = c;
                }
            }
            preds.push(self.classes_[best_idx]);
        }
        Ok(preds)
    }

    /// Compute classification accuracy with optional sample_weight.
    pub fn score(&self, x: &Matrix, y: &[f64], sample_weight: Option<&[f64]>) -> OchreResult<f64> {
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weights(sample_weight, x.rows)?;
        let preds = self.predict(x)?;
        if preds.is_empty() {
            return Ok(0.0);
        }
        match sample_weight {
            Some(sw) => {
                let mut weighted_correct = 0.0;
                let mut total_weight = 0.0;
                for ((p, actual), &w) in preds.iter().zip(y.iter()).zip(sw.iter()) {
                    total_weight += w;
                    if (*p - *actual).abs() < 1e-9 {
                        weighted_correct += w;
                    }
                }
                Ok(weighted_correct / total_weight)
            }
            None => {
                let correct = preds
                    .iter()
                    .zip(y.iter())
                    .filter(|(p, actual)| (**p - **actual).abs() < 1e-9)
                    .count();
                Ok(correct as f64 / y.len() as f64)
            }
        }
    }

    /// Update target computation device.
    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    /// Serialize model to `.dit` binary bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SGD_CLASSIFIER);
        write_str(&mut buf, &self.loss);
        write_str(&mut buf, &self.penalty);
        write_f64(&mut buf, self.alpha);
        write_f64(&mut buf, self.l1_ratio);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_opt_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.shuffle);
        write_str(&mut buf, &self.learning_rate);
        write_f64(&mut buf, self.eta0);
        write_f64(&mut buf, self.power_t);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.class_weight_mode);
        match &self.class_weight_map {
            Some(pairs) => {
                write_bool(&mut buf, true);
                write_usize(&mut buf, pairs.len());
                for &(k, v) in pairs {
                    write_f64(&mut buf, k);
                    write_f64(&mut buf, v);
                }
            }
            None => write_bool(&mut buf, false),
        }
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        write_f64_vec(&mut buf, &self.classes_);
        write_bool(&mut buf, self.is_multiclass_);
        write_usize(&mut buf, self.n_iter_);
        write_f64(&mut buf, self.t_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    /// Deserialize model from `.dit` binary bytes.
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SGD_CLASSIFIER)?;
        let loss = r.read_str()?;
        let penalty = r.read_str()?;
        let alpha = r.read_f64()?;
        let l1_ratio = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_opt_f64()?;
        let shuffle = r.read_bool()?;
        let learning_rate = r.read_str()?;
        let eta0 = r.read_f64()?;
        let power_t = r.read_f64()?;
        let random_state = r.read_opt_u64()?;
        let class_weight_mode = r.read_opt_str()?;
        let has_map = r.read_bool()?;
        let class_weight_map = if has_map {
            let len = r.read_usize()?;
            let mut pairs = Vec::with_capacity(len);
            for _ in 0..len {
                pairs.push((r.read_f64()?, r.read_f64()?));
            }
            Some(pairs)
        } else {
            None
        };
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let classes_ = r.read_f64_vec()?;
        let is_multiclass_ = r.read_bool()?;
        let n_iter_ = r.read_usize()?;
        let t_ = r.read_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let is_fitted = !classes_.is_empty();
        let (dev, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            loss,
            penalty,
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            shuffle,
            learning_rate,
            eta0,
            power_t,
            random_state,
            class_weight_mode,
            class_weight_map,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            classes_,
            is_multiclass_,
            n_iter_,
            t_,
            n_features_in_,
        })
    }
}

/// Linear Regressor trained with Stochastic Gradient Descent (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct SGDRegressor {
    pub loss: String,
    pub penalty: String,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: Option<f64>,
    pub shuffle: bool,
    pub learning_rate: String,
    pub eta0: f64,
    pub power_t: f64,
    pub epsilon: f64,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub coef_: Option<Vec<f64>>,
    pub intercept_: Option<f64>,
    pub n_iter_: usize,
    pub t_: f64,
    pub n_features_in_: Option<usize>,
}

impl SGDRegressor {
    /// Create a new SGDRegressor estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: String,
        penalty: String,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: Option<f64>,
        shuffle: bool,
        learning_rate: String,
        eta0: f64,
        power_t: f64,
        epsilon: f64,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if !matches!(
            loss.as_str(),
            "squared_error" | "huber" | "epsilon_insensitive"
        ) {
            return Err(OchreError::InvalidParameter(format!(
                "Invalid loss '{loss}'. Expected 'squared_error', 'huber', or 'epsilon_insensitive'"
            )));
        }
        if alpha < 0.0 || !alpha.is_finite() {
            return Err(OchreError::InvalidAlpha(alpha));
        }
        if epsilon < 0.0 || !epsilon.is_finite() {
            return Err(OchreError::InvalidParameter(format!(
                "epsilon ({epsilon}) must be non-negative and finite"
            )));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1".to_string(),
            ));
        }
        validate_penalty(&penalty, l1_ratio)?;
        validate_learning_rate(&learning_rate, eta0, power_t)?;
        let dev = parse_device(device.as_deref())?;

        Ok(Self {
            loss,
            penalty,
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            shuffle,
            learning_rate,
            eta0,
            power_t,
            epsilon,
            random_state,
            device: dev,
            device_: None,
            coef_: None,
            intercept_: None,
            n_iter_: 0,
            t_: 0.0,
            n_features_in_: None,
        })
    }

    /// Fit SGDRegressor on X and y with optional sample_weight.
    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<()> {
        if x.rows == 0 || x.cols == 0 {
            return Err(OchreError::EmptyData);
        }
        if x.rows != y.len() {
            return Err(OchreError::LengthMismatch {
                features: x.rows,
                targets: y.len(),
            });
        }
        validate_sample_weights(sample_weight, x.rows)?;

        let (l1_frac, l2_frac) = validate_penalty(&self.penalty, self.l1_ratio)?;
        let l1_reg = self.alpha * l1_frac;
        let l2_reg = self.alpha * l2_frac;
        let mut rng = SplitMix64::new(self.random_state);

        let n = x.rows;
        let p = x.cols;
        let mut w = vec![0.0; p];
        let mut w_scale = 1.0_f64;
        let mut b = 0.0_f64;
        let mut order = EpochOrder::new(n);
        let loss_kind = RegLoss::from_str(&self.loss);
        let lr_sched = LrSchedule::from_str(&self.learning_rate);
        let intercept_eta_scale = if lr_sched == LrSchedule::Optimal {
            0.5
        } else {
            1.0
        };
        let t0 = compute_t0_optimal(self.alpha, &self.loss);
        let mut t = 1.0_f64;
        let mut best_loss = f64::INFINITY;
        let mut no_improve = 0usize;
        let mut n_iter = 0usize;

        let no_l1 = l1_reg == 0.0;

        for epoch in 0..self.max_iter {
            n_iter = epoch + 1;
            if self.shuffle {
                order.shuffle(&mut rng);
            }
            let mut epoch_loss = 0.0;

            if no_l1 {
                order.for_each(|idx| {
                    let row = &x.data[idx * p..(idx + 1) * p];
                    let pred = dot_unrolled_chunks(row, &w) * w_scale + b;
                    let (l_val, dloss) =
                        regressor_loss_and_dloss_enum(loss_kind, pred, y[idx], self.epsilon);
                    let eta = step_size_sched(lr_sched, self.eta0, self.power_t, self.alpha, t, t0);

                    let decay = (1.0 - eta * l2_reg).max(0.0);
                    if decay > 1e-12 {
                        w_scale *= decay;
                        if w_scale < 1e-9 {
                            scale_slice_unrolled(&mut w, w_scale);
                            w_scale = 1.0;
                        }
                    } else {
                        w.fill(0.0);
                        w_scale = 1.0;
                    }

                    if dloss != 0.0 {
                        let sw = sample_weight.map_or(1.0, |s| s[idx]);
                        epoch_loss += sw * l_val;
                        let step_g = eta * sw * dloss;
                        axpy_sub_unrolled(&mut w, row, step_g / w_scale);
                        if self.fit_intercept {
                            b -= (eta * intercept_eta_scale) * sw * dloss;
                        }
                    }

                    t += 1.0;
                });
            } else {
                order.for_each(|idx| {
                    let row = &x.data[idx * p..(idx + 1) * p];
                    let pred = dot_unrolled_chunks(row, &w) * w_scale + b;
                    let (l_val, dloss) =
                        regressor_loss_and_dloss_enum(loss_kind, pred, y[idx], self.epsilon);
                    let eta = step_size_sched(lr_sched, self.eta0, self.power_t, self.alpha, t, t0);

                    if dloss == 0.0 {
                        sgd_update_l1(&mut w, &mut w_scale, row, eta, 0.0, l1_reg, l2_reg);
                    } else {
                        let sw = sample_weight.map_or(1.0, |s| s[idx]);
                        epoch_loss += sw * l_val;
                        let step_g = eta * sw * dloss;
                        sgd_update_l1(&mut w, &mut w_scale, row, eta, step_g, l1_reg, l2_reg);
                        if self.fit_intercept {
                            b -= (eta * intercept_eta_scale) * sw * dloss;
                        }
                    }

                    t += 1.0;
                });
            }

            let avg_loss = epoch_loss / (n as f64);
            if let Some(tol_val) = self.tol {
                if avg_loss > best_loss - tol_val {
                    no_improve += 1;
                    if no_improve >= 5 {
                        break;
                    }
                } else {
                    best_loss = avg_loss;
                    no_improve = 0;
                }
            }
        }

        if (w_scale - 1.0).abs() > 1e-12 {
            scale_slice_unrolled(&mut w, w_scale);
        }
        self.coef_ = Some(w);
        self.intercept_ = Some(b);
        self.n_iter_ = n_iter;
        self.t_ = t - 1.0;
        self.n_features_in_ = Some(p);
        self.device_ = Some(self.device.to_string());
        Ok(())
    }

    /// Predict target values for samples in X.
    pub fn predict(&self, x: &Matrix) -> OchreResult<Vec<f64>> {
        let coef = self.coef_.as_ref().ok_or(OchreError::NotFitted)?;
        let intercept = self.intercept_.unwrap_or(0.0);
        if !matches!(self.device, DeviceType::Cpu) && x.rows.saturating_mul(x.cols) >= 262_144 {
            let mut out = get_backend(&self.device).matvec(x, coef)?;
            if intercept != 0.0 {
                for v in &mut out {
                    *v += intercept;
                }
            }
            return Ok(out);
        }
        predict_linear_1d(x, coef, intercept)
    }

    /// Compute R^2 score on X and y.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> OchreResult<f64> {
        let preds = self.predict(x)?;
        Ok(r2_score(y, &preds))
    }

    /// Update target computation device.
    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }

    /// Serialize model to `.dit` binary bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_SGD_REGRESSOR);
        write_str(&mut buf, &self.loss);
        write_str(&mut buf, &self.penalty);
        write_f64(&mut buf, self.alpha);
        write_f64(&mut buf, self.l1_ratio);
        write_bool(&mut buf, self.fit_intercept);
        write_usize(&mut buf, self.max_iter);
        write_opt_f64(&mut buf, self.tol);
        write_bool(&mut buf, self.shuffle);
        write_str(&mut buf, &self.learning_rate);
        write_f64(&mut buf, self.eta0);
        write_f64(&mut buf, self.power_t);
        write_f64(&mut buf, self.epsilon);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_usize(&mut buf, self.n_iter_);
        write_f64(&mut buf, self.t_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    /// Deserialize model from `.dit` binary bytes.
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_SGD_REGRESSOR)?;
        let loss = r.read_str()?;
        let penalty = r.read_str()?;
        let alpha = r.read_f64()?;
        let l1_ratio = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_opt_f64()?;
        let shuffle = r.read_bool()?;
        let learning_rate = r.read_str()?;
        let eta0 = r.read_f64()?;
        let power_t = r.read_f64()?;
        let epsilon = r.read_f64()?;
        let random_state = r.read_opt_u64()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_iter_ = r.read_usize()?;
        let t_ = r.read_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            loss,
            penalty,
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            tol,
            shuffle,
            learning_rate,
            eta0,
            power_t,
            epsilon,
            random_state,
            device: dev,
            device_,
            coef_,
            intercept_,
            n_iter_,
            t_,
            n_features_in_,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_classifier_losses_and_dit() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                -2.0, -1.0, -1.5, -2.0, -1.0, -1.0, 1.0, 1.5, 2.0, 1.0, 2.5, 2.0,
            ],
        )
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        for loss in ["hinge", "log_loss", "modified_huber"] {
            let mut clf = SGDClassifier::new(
                loss.to_string(),
                "l2".to_string(),
                0.001,
                0.15,
                true,
                500,
                Some(1e-4),
                true,
                "optimal".to_string(),
                0.0,
                0.5,
                Some(42),
                None,
                None,
                Some("cpu".to_string()),
            )
            .unwrap();
            clf.fit_weighted(&x, &y, None).unwrap();
            assert!(clf.score(&x, &y, None).unwrap() > 0.99);

            if loss == "hinge" {
                assert!(clf.predict_proba(&x).is_err());
            } else {
                let probs = clf.predict_proba(&x).unwrap();
                assert_eq!(probs.rows, 6);
                assert_eq!(probs.cols, 2);
            }

            let bytes = clf.to_bytes();
            let loaded = SGDClassifier::from_bytes(&bytes, Some("cpu")).unwrap();
            assert_eq!(loaded.predict(&x).unwrap(), clf.predict(&x).unwrap());
        }
    }

    #[test]
    fn test_sgd_regressor_and_dit() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut reg = SGDRegressor::new(
            "squared_error".to_string(),
            "elasticnet".to_string(),
            0.0001,
            0.15,
            true,
            1000,
            Some(1e-5),
            true,
            "invscaling".to_string(),
            0.02,
            0.25,
            0.1,
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        reg.fit_weighted(&x, &y, None).unwrap();
        assert!(reg.score(&x, &y).unwrap() > 0.98);

        let bytes = reg.to_bytes();
        let loaded = SGDRegressor::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.predict(&x).unwrap(), reg.predict(&x).unwrap());
    }
}
