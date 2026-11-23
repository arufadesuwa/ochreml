// OchreML Manifold Learning Module (*≧ω≦*)
// Implements TSNE (exact Barnes-Hut-free gradient descent with Jacobs' adaptive gains)
// and MDS (metric & non-metric SMACOF with Kruskal isotonic regression),
// accelerated by Backend Primitives N3 (pairwise_l2_sq), N5 (gram_svd), and E1 (matmul).

use crate::backend::{Backend, get_backend};
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_f64, write_header, write_opt_f64,
    write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_str, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{buffer_to_pooled_matrix, matrix_to_pybytes};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_TSNE: u8 = 93;
pub const TAG_MDS: u8 = 94;

// Deterministic SplitMix64 PRNG with Box-Muller Gaussian sampling
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

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01().max(1e-15);
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn validate_finite_matrix(x: &Matrix) -> OchreResult<()> {
    if x.rows == 0 || x.cols == 0 {
        return Err(OchreError::EmptyData);
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(OchreError::InvalidParameter(
            "Input X contains NaN or Inf ( >_< )".to_string(),
        ));
    }
    Ok(())
}

/// Center columns of a matrix in-place.
fn center_columns_in_place(mat: &mut Matrix) {
    let n = mat.rows;
    let k = mat.cols;
    if n == 0 || k == 0 {
        return;
    }
    let mut means = vec![0.0; k];
    for r in 0..n {
        let row = &mat.data[r * k..(r + 1) * k];
        for (c, &val) in row.iter().enumerate() {
            means[c] += val;
        }
    }
    let inv_n = 1.0 / (n as f64);
    for m in &mut means {
        *m *= inv_n;
    }
    for r in 0..n {
        let row = &mut mat.data[r * k..(r + 1) * k];
        for (c, val) in row.iter_mut().enumerate() {
            *val -= means[c];
        }
    }
}

/// Binary search for per-row precision `beta_i` to match target perplexity,
/// returning conditional probabilities `P_{j|i}` flattened as `(n x n)`.
fn compute_joint_probabilities(d_sq: &Matrix, perplexity: f64) -> OchreResult<Vec<f64>> {
    let n = d_sq.rows;
    let target_entropy = perplexity.ln();
    let mut p_cond = vec![0.0; n * n];

    p_cond.par_chunks_mut(n).enumerate().for_each(|(i, p_row)| {
        let d_row = &d_sq.data[i * n..(i + 1) * n];
        let mut d_min = f64::INFINITY;
        for (j, &d) in d_row.iter().enumerate() {
            if j != i && d < d_min {
                d_min = d;
            }
        }
        if !d_min.is_finite() {
            d_min = 0.0;
        }

        let mut beta = 1.0_f64;
        let mut beta_min = 0.0_f64;
        let mut beta_max = f64::INFINITY;

        for _ in 0..50 {
            let mut sum_p = 0.0_f64;
            let mut weighted_d = 0.0_f64;
            for (j, &d) in d_row.iter().enumerate() {
                if j == i {
                    p_row[j] = 0.0;
                    continue;
                }
                let shifted = (d - d_min).max(0.0);
                let p_j = (-beta * shifted).exp();
                p_row[j] = p_j;
                sum_p += p_j;
                weighted_d += shifted * p_j;
            }

            let entropy = if sum_p > 1e-300 {
                sum_p.ln() + beta * (weighted_d / sum_p)
            } else {
                0.0
            };
            let h_diff = entropy - target_entropy;
            if h_diff.abs() < 1e-5 {
                break;
            }
            if h_diff > 0.0 {
                beta_min = beta;
                if beta_max.is_finite() {
                    beta = 0.5 * (beta + beta_max);
                } else {
                    beta *= 2.0;
                }
            } else {
                beta_max = beta;
                beta = 0.5 * (beta + beta_min);
            }
        }

        let sum_p: f64 = p_row.iter().sum();
        if sum_p > 1e-300 {
            let inv_sum = 1.0 / sum_p;
            for v in p_row.iter_mut() {
                *v *= inv_sum;
            }
        } else {
            let uniform = 1.0 / ((n - 1).max(1) as f64);
            for (j, v) in p_row.iter_mut().enumerate() {
                *v = if j == i { 0.0 } else { uniform };
            }
        }
    });

    // Symmetrize: P_ij = max((P_{j|i} + P_{i|j}) / (2n), 1e-12) for i != j
    let mut p_sym = vec![0.0; n * n];
    let inv_2n = 1.0 / (2.0 * (n as f64));
    for i in 0..n {
        for j in (i + 1)..n {
            let val = ((p_cond[i * n + j] + p_cond[j * n + i]) * inv_2n).max(1e-12);
            p_sym[i * n + j] = val;
            p_sym[j * n + i] = val;
        }
    }

    Ok(p_sym)
}

/// Initialize t-SNE embedding `Y` (`n x k`) via PCA (`gram_svd`) or Gaussian noise.
fn init_tsne_embedding(
    backend: &Backend<'_>,
    x: &Matrix,
    n_components: usize,
    init: &str,
    random_state: Option<u64>,
) -> OchreResult<Matrix> {
    let n = x.rows;
    let p = x.cols;
    let mut rng = SplitMix64::new(random_state);

    if init == "random" {
        let mut data = Vec::with_capacity(n * n_components);
        for _ in 0..(n * n_components) {
            data.push(1e-4 * rng.next_standard_normal());
        }
        return Matrix::from_vec(n, n_components, data);
    }

    // PCA initialization via Backend N5 (gram_svd)
    let (_svals, v_cols, mean) = backend.gram_svd(x, true)?;
    let k_pca = n_components.min(p);

    // Build centered X (n x p) and V_k (p x k_pca)
    let mut x_centered = Vec::with_capacity(n * p);
    for r in 0..n {
        let row = &x.data[r * p..(r + 1) * p];
        for (c, &val) in row.iter().enumerate() {
            x_centered.push(val - mean[c]);
        }
    }
    let x_c_mat = Matrix::from_vec(n, p, x_centered)?;

    let mut v_k_data = Vec::with_capacity(p * k_pca);
    for r in 0..p {
        for c in 0..k_pca {
            v_k_data.push(v_cols.data[r * v_cols.cols + c]);
        }
    }
    let v_k_mat = Matrix::from_vec(p, k_pca, v_k_data)?;
    let y_pca = backend.matmul(&x_c_mat, &v_k_mat)?;

    let mut y_data = vec![0.0; n * n_components];
    for r in 0..n {
        for c in 0..k_pca {
            y_data[r * n_components + c] = y_pca.data[r * k_pca + c];
        }
        for c in k_pca..n_components {
            y_data[r * n_components + c] = 1e-4 * rng.next_standard_normal();
        }
    }

    // Scale Y so column 0 has standard deviation 1e-4 (matching scikit-learn)
    let mut mean0 = 0.0;
    for r in 0..n {
        mean0 += y_data[r * n_components];
    }
    mean0 /= n as f64;
    let mut var0 = 0.0;
    for r in 0..n {
        let diff = y_data[r * n_components] - mean0;
        var0 += diff * diff;
    }
    let std0 = (var0 / (n as f64)).sqrt();
    if std0 > 1e-12 {
        let scale = 1e-4 / std0;
        for r in 0..n {
            for c in 0..k_pca {
                y_data[r * n_components + c] *= scale;
            }
        }
    }

    Matrix::from_vec(n, n_components, y_data)
}

/// t-distributed Stochastic Neighbor Embedding (TSNE) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct TSNE {
    pub n_components: usize,
    pub perplexity: f64,
    pub early_exaggeration: f64,
    pub learning_rate: Option<f64>,
    pub max_iter: usize,
    pub init: String,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub embedding_: Option<Matrix>,
    pub kl_divergence_: Option<f64>,
    pub n_iter_: Option<usize>,
    pub learning_rate_: Option<f64>,
    pub n_features_in_: Option<usize>,
}

impl TSNE {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: usize,
        perplexity: f64,
        early_exaggeration: f64,
        learning_rate: Option<f64>,
        max_iter: usize,
        init: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !perplexity.is_finite() || perplexity <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "perplexity ({perplexity}) must be a positive finite number ( >_< )"
            )));
        }
        if !early_exaggeration.is_finite() || early_exaggeration < 1.0 {
            return Err(OchreError::InvalidParameter(format!(
                "early_exaggeration ({early_exaggeration}) must be >= 1.0 ( >_< )"
            )));
        }
        if let Some(lr) = learning_rate
            && (!lr.is_finite() || lr <= 0.0)
        {
            return Err(OchreError::InvalidParameter(format!(
                "learning_rate ({lr}) must be positive and finite ( >_< )"
            )));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1 ( >_< )".to_string(),
            ));
        }
        let init_norm = init.to_lowercase();
        if init_norm != "pca" && init_norm != "random" {
            return Err(OchreError::InvalidParameter(format!(
                "init '{init}' is invalid; expected 'pca' or 'random' ( >_< )"
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            perplexity,
            early_exaggeration,
            learning_rate,
            max_iter,
            init: init_norm,
            random_state,
            device: dev,
            device_: None,
            embedding_: None,
            kl_divergence_: None,
            n_iter_: None,
            learning_rate_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        if self.n_features_in_.is_some() {
            self.device_ = Some(self.device.to_string());
        }
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        let _ = self.fit_transform(x)?;
        Ok(())
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let n = x.rows;
        let k = self.n_components;
        if n < 2 {
            return Err(OchreError::InvalidParameter(
                "TSNE requires at least 2 samples ( >_< )".to_string(),
            ));
        }
        if self.perplexity >= (n as f64) {
            return Err(OchreError::InvalidParameter(format!(
                "perplexity ({}) must be strictly less than n_samples ({n}) ( >_< )",
                self.perplexity
            )));
        }

        let backend = get_backend(&self.device);
        let lr = self
            .learning_rate
            .unwrap_or_else(|| ((n as f64) / self.early_exaggeration / 4.0).max(50.0));

        // 1. Pairwise squared distances via Backend N3
        let d_sq = backend.pairwise_l2_sq(x, x)?;
        // 2. Binary search per-row beta_i & symmetrize P
        let p_sym = compute_joint_probabilities(&d_sq, self.perplexity)?;
        // 3. Initialize Y via PCA (N5) or random SplitMix64
        let mut y = init_tsne_embedding(&backend, x, k, &self.init, self.random_state)?;

        let mut gains = vec![1.0_f64; n * k];
        let mut update = vec![0.0_f64; n * k];
        let mut grad = vec![0.0_f64; n * k];
        let mut num = vec![0.0_f64; n * n];

        let early_exag_iters = 250usize.min(self.max_iter);
        let mut final_iter = 0usize;

        for iter in 0..self.max_iter {
            final_iter = iter + 1;
            let exag = if iter < early_exag_iters {
                self.early_exaggeration
            } else {
                1.0
            };
            let momentum = if iter < early_exag_iters { 0.5 } else { 0.8 };

            // Compute Q via Backend N3 pairwise_l2_sq(&y, &y) + 1 / (1 + d^2)
            let y_d_sq = backend.pairwise_l2_sq(&y, &y)?;
            let mut sum_num = 0.0_f64;
            for i in 0..n {
                num[i * n + i] = 0.0;
                for j in (i + 1)..n {
                    let val = 1.0 / (1.0 + y_d_sq.data[i * n + j].max(0.0));
                    num[i * n + j] = val;
                    num[j * n + i] = val;
                    sum_num += 2.0 * val;
                }
            }
            let inv_sum_num = 1.0 / sum_num.max(1e-12);

            // Exact gradient computation in parallel across samples
            let y_slice = &y.data;
            grad.par_chunks_mut(k).enumerate().for_each(|(i, g_row)| {
                g_row.fill(0.0);
                let yi = &y_slice[i * k..(i + 1) * k];
                let p_row = &p_sym[i * n..(i + 1) * n];
                let num_row = &num[i * n..(i + 1) * n];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let num_ij = num_row[j];
                    let q_ij = (num_ij * inv_sum_num).max(1e-12);
                    let w = 4.0 * (exag * p_row[j] - q_ij) * num_ij;
                    let yj = &y_slice[j * k..(j + 1) * k];
                    for c in 0..k {
                        g_row[c] += w * (yi[c] - yj[c]);
                    }
                }
            });

            // Jacobs' delta-bar-delta adaptive gains + momentum update
            let mut grad_norm_sq = 0.0_f64;
            for idx in 0..(n * k) {
                let g = grad[idx];
                grad_norm_sq += g * g;
                let same_sign = (g > 0.0) == (update[idx] > 0.0);
                let gain = if same_sign {
                    (gains[idx] * 0.8).max(0.01)
                } else {
                    gains[idx] + 0.2
                };
                gains[idx] = gain;
                let u = momentum * update[idx] - lr * gain * g;
                update[idx] = u;
                y.data[idx] += u;
            }

            center_columns_in_place(&mut y);

            if iter >= early_exag_iters && grad_norm_sq.sqrt() < 1e-7 {
                break;
            }
        }

        // Compute final KL divergence KL(P || Q)
        let y_d_sq = backend.pairwise_l2_sq(&y, &y)?;
        let mut sum_num = 0.0_f64;
        for i in 0..n {
            num[i * n + i] = 0.0;
            for j in (i + 1)..n {
                let val = 1.0 / (1.0 + y_d_sq.data[i * n + j].max(0.0));
                num[i * n + j] = val;
                num[j * n + i] = val;
                sum_num += 2.0 * val;
            }
        }
        let inv_sum_num = 1.0 / sum_num.max(1e-12);
        let mut kl = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p_ij = p_sym[i * n + j].max(1e-12);
                let q_ij = (num[i * n + j] * inv_sum_num).max(1e-12);
                kl += p_ij * (p_ij / q_ij).ln();
            }
        }

        self.embedding_ = Some(y.clone());
        self.kl_divergence_ = Some(kl);
        self.n_iter_ = Some(final_iter);
        self.learning_rate_ = Some(lr);
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());

        Ok(y)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_TSNE);
        write_usize(&mut buf, self.n_components);
        write_f64(&mut buf, self.perplexity);
        write_f64(&mut buf, self.early_exaggeration);
        write_opt_f64(&mut buf, self.learning_rate);
        write_usize(&mut buf, self.max_iter);
        write_str(&mut buf, &self.init);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.embedding_);
        write_opt_f64(&mut buf, self.kl_divergence_);
        write_opt_usize(&mut buf, self.n_iter_);
        write_opt_f64(&mut buf, self.learning_rate_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(data: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(data);
        check_magic_and_tag(&mut r, TAG_TSNE)?;
        let n_components = r.read_usize()?;
        let perplexity = r.read_f64()?;
        let early_exaggeration = r.read_f64()?;
        let learning_rate = r.read_opt_f64()?;
        let max_iter = r.read_usize()?;
        let init = r.read_str()?;
        let random_state = r.read_opt_u64()?;
        let stored_dev = r.read_opt_str()?;
        let embedding_ = r.read_opt_matrix()?;
        let kl_divergence_ = r.read_opt_f64()?;
        let n_iter_ = r.read_opt_usize()?;
        let learning_rate_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;

        let (resolved_dev, resolved_label) =
            resolve_loaded_device(stored_dev, n_features_in_.is_some(), device)?;
        let mut model = Self::new(
            n_components,
            perplexity,
            early_exaggeration,
            learning_rate,
            max_iter,
            init,
            random_state,
            Some(resolved_dev.to_string()),
        )?;
        model.embedding_ = embedding_;
        model.kl_divergence_ = kl_divergence_;
        model.n_iter_ = n_iter_;
        model.learning_rate_ = learning_rate_;
        model.n_features_in_ = n_features_in_;
        model.device_ = resolved_label;
        Ok(model)
    }
}

#[pyclass(from_py_object, name = "TSNE")]
#[derive(Clone)]
pub struct PyTSNE {
    inner: TSNE,
}

pymethods_with_dit!(PyTSNE, TSNE, {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_components = 2, perplexity = 30.0, early_exaggeration = 12.0, learning_rate = None, max_iter = 1000, init = "pca".to_string(), random_state = None, device = None))]
    pub fn new(
        n_components: usize,
        perplexity: f64,
        early_exaggeration: f64,
        learning_rate: Option<f64>,
        max_iter: usize,
        init: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TSNE::new(
                n_components,
                perplexity,
                early_exaggeration,
                learning_rate,
                max_iter,
                init,
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

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }

    #[getter]
    pub fn perplexity(&self) -> f64 {
        self.inner.perplexity
    }

    #[getter]
    pub fn early_exaggeration(&self) -> f64 {
        self.inner.early_exaggeration
    }

    #[getter]
    pub fn learning_rate(&self) -> Option<f64> {
        self.inner.learning_rate
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn init(&self) -> String {
        self.inner.init.clone()
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn embedding_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.embedding_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn kl_divergence_(&self) -> Option<f64> {
        self.inner.kl_divergence_
    }

    #[getter]
    pub fn n_iter_(&self) -> Option<usize> {
        self.inner.n_iter_
    }

    #[getter]
    pub fn learning_rate_(&self) -> Option<f64> {
        self.inner.learning_rate_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

// ============================================================================
// Multidimensional Scaling (MDS) - SMACOF + Kruskal Isotonic Regression
// ============================================================================

/// Pool Adjacent Violators Algorithm (PAVA) for non-decreasing isotonic regression.
fn pava_non_decreasing(y: &[f64]) -> Vec<f64> {
    let m = y.len();
    if m == 0 {
        return Vec::new();
    }
    // Stack of (sum, count) blocks
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(m);
    for &val in y {
        blocks.push((val, 1));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let (s_last, c_last) = blocks[len - 1];
            let (s_prev, c_prev) = blocks[len - 2];
            if s_prev / (c_prev as f64) > s_last / (c_last as f64) {
                blocks.pop();
                if let Some(prev) = blocks.last_mut() {
                    prev.0 += s_last;
                    prev.1 += c_last;
                }
            } else {
                break;
            }
        }
    }
    let mut out = Vec::with_capacity(m);
    for (s, c) in blocks {
        let avg = s / (c as f64);
        for _ in 0..c {
            out.push(avg);
        }
    }
    out
}

/// Compute Euclidean distance matrix `D(Z)` (`n x n`) from embedding `Z` (`n x k`).
fn embedding_distances(backend: &Backend<'_>, z: &Matrix) -> OchreResult<Matrix> {
    let mut d_sq = backend.pairwise_l2_sq(z, z)?;
    let n = z.rows;
    for i in 0..n {
        d_sq.data[i * n + i] = 0.0;
        for j in (i + 1)..n {
            let d = d_sq.data[i * n + j].max(0.0).sqrt();
            d_sq.data[i * n + j] = d;
            d_sq.data[j * n + i] = d;
        }
    }
    Ok(d_sq)
}

/// Compute `Z_new = (1 / n) * B * Z` using `Backend::matmul` (E1) on large matrices
/// and Rayon row-wise parallelism on small matrices.
fn guttman_transform(backend: &Backend<'_>, b_mat: &Matrix, z: &Matrix) -> OchreResult<Matrix> {
    let n = z.rows;
    let k = z.cols;
    let inv_n = 1.0 / (n as f64);

    if n >= 64 {
        let mut z_new = backend.matmul(b_mat, z)?;
        for v in &mut z_new.data {
            *v *= inv_n;
        }
        return Ok(z_new);
    }

    let mut out = vec![0.0_f64; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(i, out_row)| {
        let b_row = &b_mat.data[i * n..(i + 1) * n];
        for (j, &b_ij) in b_row.iter().enumerate() {
            if b_ij != 0.0 {
                let z_row = &z.data[j * k..(j + 1) * k];
                for c in 0..k {
                    out_row[c] += b_ij * z_row[c];
                }
            }
        }
        for v in out_row.iter_mut() {
            *v *= inv_n;
        }
    });
    Matrix::from_vec(n, k, out)
}

/// Run a single SMACOF restart.
#[allow(clippy::too_many_arguments)]
fn smacof_single(
    backend: &Backend<'_>,
    dissim: &Matrix,
    upper_order: &[(usize, usize)],
    n_components: usize,
    metric: bool,
    max_iter: usize,
    eps: f64,
    rng: &mut SplitMix64,
) -> OchreResult<(Matrix, f64, usize)> {
    let n = dissim.rows;
    let k = n_components;

    let mut z_data = Vec::with_capacity(n * k);
    for _ in 0..(n * k) {
        z_data.push(rng.next_standard_normal());
    }
    let mut z = Matrix::from_vec(n, k, z_data)?;
    center_columns_in_place(&mut z);

    let mut disparities = dissim.data.clone();
    let mut b_mat = Matrix::zeros(n, n);
    let mut old_stress: Option<f64> = None;
    let mut final_stress = 0.0_f64;
    let mut final_iter = 0usize;

    let m_pairs = upper_order.len();
    let norm_target = (m_pairs as f64).max(1.0);

    for iter in 0..max_iter {
        final_iter = iter + 1;
        let d_z = embedding_distances(backend, &z)?;

        if !metric && m_pairs > 0 {
            // Sort ties in d_z for pairs with identical dissimilarity (Kruskal secondary/primary)
            let mut d_z_ordered: Vec<f64> = upper_order
                .iter()
                .map(|&(i, j)| d_z.data[i * n + j])
                .collect();
            let iso = pava_non_decreasing(&d_z_ordered);
            let sum_sq: f64 = iso.iter().map(|&v| v * v).sum();
            let scale = if sum_sq > 1e-12 {
                (norm_target / sum_sq).sqrt()
            } else {
                1.0
            };
            for (idx, &(i, j)) in upper_order.iter().enumerate() {
                let val = iso[idx] * scale;
                disparities[i * n + j] = val;
                disparities[j * n + i] = val;
            }
            d_z_ordered.clear();
        }

        // Compute stress = sum_{i < j} (d_Z[i, j] - disparities[i, j])^2
        let mut stress = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = d_z.data[i * n + j] - disparities[i * n + j];
                stress += diff * diff;
            }
        }
        final_stress = stress;

        // Build B(Z)
        for i in 0..n {
            let mut diag = 0.0_f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d_ij = d_z.data[i * n + j];
                let b_ij = if d_ij > 1e-12 {
                    -disparities[i * n + j] / d_ij
                } else {
                    0.0
                };
                b_mat.data[i * n + j] = b_ij;
                diag -= b_ij;
            }
            b_mat.data[i * n + i] = diag;
        }

        z = guttman_transform(backend, &b_mat, &z)?;
        center_columns_in_place(&mut z);

        if let Some(prev) = old_stress {
            let rel_tol = eps * prev.max(1.0);
            if (prev - stress).abs() <= rel_tol {
                break;
            }
        }
        old_stress = Some(stress);
    }

    // Recompute exact stress for final Z
    let d_z_final = embedding_distances(backend, &z)?;
    if !metric && m_pairs > 0 {
        let d_z_ordered: Vec<f64> = upper_order
            .iter()
            .map(|&(i, j)| d_z_final.data[i * n + j])
            .collect();
        let iso = pava_non_decreasing(&d_z_ordered);
        let sum_sq: f64 = iso.iter().map(|&v| v * v).sum();
        let scale = if sum_sq > 1e-12 {
            (norm_target / sum_sq).sqrt()
        } else {
            1.0
        };
        for (idx, &(i, j)) in upper_order.iter().enumerate() {
            let val = iso[idx] * scale;
            disparities[i * n + j] = val;
            disparities[j * n + i] = val;
        }
    }
    let mut exact_stress = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = d_z_final.data[i * n + j] - disparities[i * n + j];
            exact_stress += diff * diff;
        }
    }
    if exact_stress.is_finite() {
        final_stress = exact_stress;
    }

    Ok((z, final_stress, final_iter))
}

/// Multidimensional Scaling (MDS) (*≧ω≦*)
#[derive(Debug, Clone)]
pub struct MDS {
    pub n_components: usize,
    pub metric: bool,
    pub n_init: usize,
    pub max_iter: usize,
    pub eps: f64,
    pub dissimilarity: String,
    pub random_state: Option<u64>,
    pub device: DeviceType,
    pub device_: Option<String>,
    pub embedding_: Option<Matrix>,
    pub stress_: Option<f64>,
    pub dissimilarity_matrix_: Option<Matrix>,
    pub n_iter_: Option<usize>,
    pub n_features_in_: Option<usize>,
}

impl MDS {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: usize,
        metric: bool,
        n_init: usize,
        max_iter: usize,
        eps: f64,
        dissimilarity: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1 ( >_< )".to_string(),
            ));
        }
        if n_init == 0 {
            return Err(OchreError::InvalidParameter(
                "n_init must be >= 1 ( >_< )".to_string(),
            ));
        }
        if max_iter == 0 {
            return Err(OchreError::InvalidParameter(
                "max_iter must be >= 1 ( >_< )".to_string(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "eps ({eps}) must be non-negative and finite ( >_< )"
            )));
        }
        let dissim_norm = dissimilarity.to_lowercase();
        if dissim_norm != "euclidean" && dissim_norm != "precomputed" {
            return Err(OchreError::InvalidParameter(format!(
                "dissimilarity '{dissimilarity}' is invalid; expected 'euclidean' or 'precomputed' ( >_< )"
            )));
        }
        let dev = parse_device(device.as_deref())?;
        Ok(Self {
            n_components,
            metric,
            n_init,
            max_iter,
            eps,
            dissimilarity: dissim_norm,
            random_state,
            device: dev,
            device_: None,
            embedding_: None,
            stress_: None,
            dissimilarity_matrix_: None,
            n_iter_: None,
            n_features_in_: None,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device = dev;
        if self.n_features_in_.is_some() {
            self.device_ = Some(self.device.to_string());
        }
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        let _ = self.fit_transform(x)?;
        Ok(())
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        validate_finite_matrix(x)?;
        let n = x.rows;
        let backend = get_backend(&self.device);

        let dissim = if self.dissimilarity == "precomputed" {
            if x.rows != x.cols {
                return Err(OchreError::NonSquareMatrix);
            }
            let mut d = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let v = x.data[i * n + j];
                    if v < 0.0 {
                        return Err(OchreError::InvalidParameter(
                            "Precomputed dissimilarity matrix must be non-negative ( >_< )"
                                .to_string(),
                        ));
                    }
                    let vt = x.data[j * n + i];
                    if (v - vt).abs() > 1e-8 * (1.0 + v.abs()) {
                        return Err(OchreError::InvalidParameter(
                            "Precomputed dissimilarity matrix must be symmetric ( >_< )"
                                .to_string(),
                        ));
                    }
                    d.data[i * n + j] = if i == j { 0.0 } else { 0.5 * (v + vt) };
                }
            }
            d
        } else {
            let mut d_sq = backend.pairwise_l2_sq(x, x)?;
            for i in 0..n {
                d_sq.data[i * n + i] = 0.0;
                for j in (i + 1)..n {
                    let dist = d_sq.data[i * n + j].max(0.0).sqrt();
                    d_sq.data[i * n + j] = dist;
                    d_sq.data[j * n + i] = dist;
                }
            }
            d_sq
        };

        // Precompute upper-triangular pairs sorted by dissimilarity for non-metric PAVA
        let mut upper_order: Vec<(usize, usize)> = if !self.metric {
            let mut pairs = Vec::with_capacity(n * n.saturating_sub(1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    pairs.push((i, j));
                }
            }
            pairs.sort_by(|&(i1, j1), &(i2, j2)| {
                let d1 = dissim.data[i1 * n + j1];
                let d2 = dissim.data[i2 * n + j2];
                d1.total_cmp(&d2)
            });
            pairs
        } else {
            Vec::new()
        };

        let mut rng = SplitMix64::new(self.random_state);
        let mut best_z: Option<Matrix> = None;
        let mut best_stress = f64::INFINITY;
        let mut best_iter = 0usize;

        for _ in 0..self.n_init {
            let (z, stress, iters) = smacof_single(
                &backend,
                &dissim,
                &upper_order,
                self.n_components,
                self.metric,
                self.max_iter,
                self.eps,
                &mut rng,
            )?;
            if stress < best_stress || best_z.is_none() {
                best_stress = stress;
                best_z = Some(z);
                best_iter = iters;
            }
        }
        upper_order.clear();

        let embedding = best_z.ok_or(OchreError::EmptyData)?;
        self.embedding_ = Some(embedding.clone());
        self.stress_ = Some(best_stress);
        self.dissimilarity_matrix_ = Some(dissim);
        self.n_iter_ = Some(best_iter);
        self.n_features_in_ = Some(x.cols);
        self.device_ = Some(self.device.to_string());

        Ok(embedding)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_MDS);
        write_usize(&mut buf, self.n_components);
        write_bool(&mut buf, self.metric);
        write_usize(&mut buf, self.n_init);
        write_usize(&mut buf, self.max_iter);
        write_f64(&mut buf, self.eps);
        write_str(&mut buf, &self.dissimilarity);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.embedding_);
        write_opt_f64(&mut buf, self.stress_);
        write_opt_matrix(&mut buf, &self.dissimilarity_matrix_);
        write_opt_usize(&mut buf, self.n_iter_);
        write_opt_usize(&mut buf, self.n_features_in_);
        buf
    }

    pub fn from_bytes(data: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(data);
        check_magic_and_tag(&mut r, TAG_MDS)?;
        let n_components = r.read_usize()?;
        let metric = r.read_bool()?;
        let n_init = r.read_usize()?;
        let max_iter = r.read_usize()?;
        let eps = r.read_f64()?;
        let dissimilarity = r.read_str()?;
        let random_state = r.read_opt_u64()?;
        let stored_dev = r.read_opt_str()?;
        let embedding_ = r.read_opt_matrix()?;
        let stress_ = r.read_opt_f64()?;
        let dissimilarity_matrix_ = r.read_opt_matrix()?;
        let n_iter_ = r.read_opt_usize()?;
        let n_features_in_ = r.read_opt_usize()?;

        let (resolved_dev, resolved_label) =
            resolve_loaded_device(stored_dev, n_features_in_.is_some(), device)?;
        let mut model = Self::new(
            n_components,
            metric,
            n_init,
            max_iter,
            eps,
            dissimilarity,
            random_state,
            Some(resolved_dev.to_string()),
        )?;
        model.embedding_ = embedding_;
        model.stress_ = stress_;
        model.dissimilarity_matrix_ = dissimilarity_matrix_;
        model.n_iter_ = n_iter_;
        model.n_features_in_ = n_features_in_;
        model.device_ = resolved_label;
        Ok(model)
    }
}

#[pyclass(from_py_object, name = "MDS")]
#[derive(Clone)]
pub struct PyMDS {
    inner: MDS,
}

pymethods_with_dit!(PyMDS, MDS, {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_components = 2, metric = true, n_init = 4, max_iter = 300, eps = 1e-3, dissimilarity = "euclidean".to_string(), random_state = None, device = None))]
    pub fn new(
        n_components: usize,
        metric: bool,
        n_init: usize,
        max_iter: usize,
        eps: f64,
        dissimilarity: String,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MDS::new(
                n_components,
                metric,
                n_init,
                max_iter,
                eps,
                dissimilarity,
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

    pub fn fit_transform_buffer<'py>(
        &mut self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.fit_transform(&mat)?;
        matrix_to_pybytes(py, &out)
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }

    #[getter]
    pub fn metric(&self) -> bool {
        self.inner.metric
    }

    #[getter]
    pub fn n_init(&self) -> usize {
        self.inner.n_init
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    pub fn dissimilarity(&self) -> String {
        self.inner.dissimilarity.clone()
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn embedding_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.embedding_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn stress_(&self) -> Option<f64> {
        self.inner.stress_
    }

    #[getter]
    pub fn dissimilarity_matrix_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.dissimilarity_matrix_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn n_iter_(&self) -> Option<usize> {
        self.inner.n_iter_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTSNE>()?;
    m.add_class::<PyMDS>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsne_fit_and_dit_roundtrip() {
        let mut rows = Vec::new();
        for i in 0..8 {
            rows.push(vec![i as f64 * 0.1, 0.0, 0.0]);
        }
        for i in 0..8 {
            rows.push(vec![10.0 + i as f64 * 0.1, 10.0, 10.0]);
        }
        let x = Matrix::from_2d(&rows).unwrap();
        let mut tsne = TSNE::new(
            2,
            5.0,
            12.0,
            None,
            200,
            "pca".to_string(),
            Some(42),
            Some("cpu".to_string()),
        )
        .unwrap();
        let emb = tsne.fit_transform(&x).unwrap();
        assert_eq!(emb.rows, 16);
        assert_eq!(emb.cols, 2);
        assert!(tsne.kl_divergence_.unwrap().is_finite());

        let bytes = tsne.to_bytes();
        let loaded = TSNE::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.embedding_.unwrap(), emb);
    }

    #[test]
    fn test_mds_metric_and_nonmetric_dit_roundtrip() {
        let x = Matrix::from_2d(&[
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![1.0, 2.0, 0.0],
            vec![0.5, 1.0, 1.5],
        ])
        .unwrap();

        let mut mds = MDS::new(
            2,
            true,
            2,
            100,
            1e-4,
            "euclidean".to_string(),
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        let emb = mds.fit_transform(&x).unwrap();
        assert_eq!(emb.rows, 5);
        assert_eq!(emb.cols, 2);
        assert!(mds.stress_.unwrap() >= 0.0);

        let bytes = mds.to_bytes();
        let loaded = MDS::from_bytes(&bytes, Some("cpu")).unwrap();
        assert_eq!(loaded.embedding_.unwrap(), emb);

        let mut nm_mds = MDS::new(
            2,
            false,
            2,
            100,
            1e-4,
            "euclidean".to_string(),
            Some(7),
            Some("cpu".to_string()),
        )
        .unwrap();
        let nm_emb = nm_mds.fit_transform(&x).unwrap();
        assert_eq!(nm_emb.rows, 5);
        assert_eq!(nm_emb.cols, 2);
        assert!(nm_mds.stress_.unwrap().is_finite());
    }
}
