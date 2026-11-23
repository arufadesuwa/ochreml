// PyO3 bindings module for OchreML (＾▽＾)
// Exposes Rust Linear, Ridge, and Decision Tree implementations with Hardware Acceleration to Python.

pub mod backend;
pub mod backends;
pub mod boosting;
pub mod cluster;
pub mod cluster_advanced;
pub mod cross_decomposition;
pub mod decision_tree;
pub mod decomposition;
pub mod device;
pub mod discriminant;
pub mod dit_io;
pub mod error;
pub mod forest;
pub mod gaussian_process;
pub mod glm;
pub mod hist_gradient_boosting;
pub mod kernel_models;
pub mod linear_regression;
pub mod linear_svm;
pub mod logistic_regression;
pub mod manifold;
pub mod matrix;
pub mod mixture;
pub mod naive_bayes;
pub mod neighbors;
pub mod neural_network;
pub mod preprocessing;
pub mod primitives;
pub mod ridge;
pub mod sgd;
pub mod sparse_linear;

use decision_tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeNode};
use device::parse_device;
use error::{OchreError, OchreResult};
use linear_regression::LinearRegression;
use logistic_regression::LogisticRegression;
use matrix::Matrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ridge::{Ridge, RidgeCV, RidgeClassifier};
use sgd::{SGDClassifier, SGDRegressor};
use sparse_linear::{ElasticNet, Lasso};
use std::collections::HashMap;
use std::fs;

// Auto-convert OchreError to PyValueError with kaomoji preserved (*≧ω≦*)
impl From<OchreError> for pyo3::PyErr {
    fn from(err: OchreError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

const DIT_MAGIC: &[u8; 4] = b"DIT\x01";
const TAG_LINEAR_REGRESSION: u8 = 1;
const TAG_DT_CLASSIFIER: u8 = 2;
const TAG_DT_REGRESSOR: u8 = 3;
const TAG_RIDGE: u8 = 4;
const TAG_RIDGE_CLASSIFIER: u8 = 5;
const TAG_RIDGE_CV: u8 = 6;
pub const TAG_STANDARD_SCALER: u8 = preprocessing::scalers::TAG_STANDARD_SCALER;
pub const TAG_MINMAX_SCALER: u8 = preprocessing::scalers::TAG_MINMAX_SCALER;
pub const TAG_MAXABS_SCALER: u8 = preprocessing::scalers::TAG_MAXABS_SCALER;
pub const TAG_ROBUST_SCALER: u8 = preprocessing::scalers::TAG_ROBUST_SCALER;
pub const TAG_NORMALIZER: u8 = preprocessing::scalers::TAG_NORMALIZER;
pub const TAG_BINARIZER: u8 = preprocessing::transforms::TAG_BINARIZER;
pub const TAG_KERNEL_CENTERER: u8 = preprocessing::features::TAG_KERNEL_CENTERER;
pub const TAG_POLYNOMIAL_FEATURES: u8 = preprocessing::features::TAG_POLYNOMIAL_FEATURES;
pub const TAG_KBINS_DISCRETIZER: u8 = preprocessing::transforms::TAG_KBINS_DISCRETIZER;
pub const TAG_POWER_TRANSFORMER: u8 = preprocessing::transforms::TAG_POWER_TRANSFORMER;
pub const TAG_QUANTILE_TRANSFORMER: u8 = preprocessing::transforms::TAG_QUANTILE_TRANSFORMER;
pub const TAG_SPLINE_TRANSFORMER: u8 = preprocessing::features::TAG_SPLINE_TRANSFORMER;
pub const TAG_ONE_HOT_ENCODER: u8 = preprocessing::encoders::TAG_ONE_HOT_ENCODER;
pub const TAG_ORDINAL_ENCODER: u8 = preprocessing::encoders::TAG_ORDINAL_ENCODER;
pub const TAG_LABEL_ENCODER: u8 = preprocessing::encoders::TAG_LABEL_ENCODER;
pub const TAG_LABEL_BINARIZER: u8 = preprocessing::encoders::TAG_LABEL_BINARIZER;
pub const TAG_MULTILABEL_BINARIZER: u8 = preprocessing::encoders::TAG_MULTILABEL_BINARIZER;
pub const TAG_TARGET_ENCODER: u8 = preprocessing::encoders::TAG_TARGET_ENCODER;
pub const TAG_SIMPLE_IMPUTER: u8 = preprocessing::impute::TAG_SIMPLE_IMPUTER;
pub const TAG_MISSING_INDICATOR: u8 = preprocessing::impute::TAG_MISSING_INDICATOR;
pub const TAG_KNN_IMPUTER: u8 = preprocessing::impute::TAG_KNN_IMPUTER;
pub const TAG_DICT_VECTORIZER: u8 = preprocessing::text_fe::TAG_DICT_VECTORIZER;
pub const TAG_FEATURE_HASHER: u8 = preprocessing::text_fe::TAG_FEATURE_HASHER;
pub const TAG_COUNT_VECTORIZER: u8 = preprocessing::text_fe::TAG_COUNT_VECTORIZER;
pub const TAG_TFIDF_TRANSFORMER: u8 = preprocessing::text_fe::TAG_TFIDF_TRANSFORMER;
pub const TAG_TFIDF_VECTORIZER: u8 = preprocessing::text_fe::TAG_TFIDF_VECTORIZER;
pub const TAG_HASHING_VECTORIZER: u8 = preprocessing::text_fe::TAG_HASHING_VECTORIZER;
pub const TAG_ITERATIVE_IMPUTER: u8 = 34;
pub const TAG_FUNCTION_TRANSFORMER: u8 = 35;
pub const TAG_COLUMN_TRANSFORMER: u8 = 36;
pub const TAG_TRANSFORMED_TARGET_REGRESSOR: u8 = 37;
pub const TAG_LOGISTIC_REGRESSION: u8 = logistic_regression::TAG_LOGISTIC_REGRESSION;
pub const TAG_LASSO: u8 = sparse_linear::TAG_LASSO;
pub const TAG_ELASTIC_NET: u8 = sparse_linear::TAG_ELASTIC_NET;
pub const TAG_SGD_CLASSIFIER: u8 = sgd::TAG_SGD_CLASSIFIER;
pub const TAG_SGD_REGRESSOR: u8 = sgd::TAG_SGD_REGRESSOR;
pub const TAG_PCA: u8 = decomposition::TAG_PCA;
pub const TAG_TRUNCATED_SVD: u8 = decomposition::TAG_TRUNCATED_SVD;
pub const TAG_LDA: u8 = discriminant::TAG_LDA;
pub const TAG_QDA: u8 = discriminant::TAG_QDA;
pub const TAG_KMEANS: u8 = cluster::TAG_KMEANS;
pub const TAG_MINI_BATCH_KMEANS: u8 = cluster::TAG_MINI_BATCH_KMEANS;
pub const TAG_HUBER_REGRESSOR: u8 = glm::TAG_HUBER_REGRESSOR;
pub const TAG_QUANTILE_REGRESSOR: u8 = glm::TAG_QUANTILE_REGRESSOR;
pub const TAG_POISSON_REGRESSOR: u8 = glm::TAG_POISSON_REGRESSOR;
pub const TAG_GAMMA_REGRESSOR: u8 = glm::TAG_GAMMA_REGRESSOR;
pub const TAG_TWEEDIE_REGRESSOR: u8 = glm::TAG_TWEEDIE_REGRESSOR;
pub const TAG_LINEAR_SVC: u8 = linear_svm::TAG_LINEAR_SVC;
pub const TAG_LINEAR_SVR: u8 = linear_svm::TAG_LINEAR_SVR;
pub const TAG_KNEIGHBORS_CLASSIFIER: u8 = neighbors::TAG_KNEIGHBORS_CLASSIFIER;
pub const TAG_KNEIGHBORS_REGRESSOR: u8 = neighbors::TAG_KNEIGHBORS_REGRESSOR;
pub const TAG_RADIUS_NEIGHBORS_CLASSIFIER: u8 = neighbors::TAG_RADIUS_NEIGHBORS_CLASSIFIER;
pub const TAG_RADIUS_NEIGHBORS_REGRESSOR: u8 = neighbors::TAG_RADIUS_NEIGHBORS_REGRESSOR;
pub const TAG_GAUSSIAN_NB: u8 = naive_bayes::TAG_GAUSSIAN_NB;
pub const TAG_MULTINOMIAL_NB: u8 = naive_bayes::TAG_MULTINOMIAL_NB;
pub const TAG_BERNOULLI_NB: u8 = naive_bayes::TAG_BERNOULLI_NB;
pub const TAG_COMPLEMENT_NB: u8 = naive_bayes::TAG_COMPLEMENT_NB;
pub const TAG_GAUSSIAN_MIXTURE: u8 = mixture::TAG_GAUSSIAN_MIXTURE;
pub const TAG_HIST_GB_CLASSIFIER: u8 = hist_gradient_boosting::TAG_HIST_GB_CLASSIFIER;
pub const TAG_HIST_GB_REGRESSOR: u8 = hist_gradient_boosting::TAG_HIST_GB_REGRESSOR;
pub const TAG_RANDOM_FOREST_CLASSIFIER: u8 = forest::TAG_RANDOM_FOREST_CLASSIFIER;
pub const TAG_RANDOM_FOREST_REGRESSOR: u8 = forest::TAG_RANDOM_FOREST_REGRESSOR;
pub const TAG_EXTRA_TREES_CLASSIFIER: u8 = forest::TAG_EXTRA_TREES_CLASSIFIER;
pub const TAG_EXTRA_TREES_REGRESSOR: u8 = forest::TAG_EXTRA_TREES_REGRESSOR;
pub const TAG_GB_CLASSIFIER: u8 = boosting::TAG_GB_CLASSIFIER;
pub const TAG_GB_REGRESSOR: u8 = boosting::TAG_GB_REGRESSOR;
pub const TAG_ADABOOST_CLASSIFIER: u8 = boosting::TAG_ADABOOST_CLASSIFIER;
pub const TAG_ADABOOST_REGRESSOR: u8 = boosting::TAG_ADABOOST_REGRESSOR;
pub const TAG_ISOLATION_FOREST: u8 = forest::TAG_ISOLATION_FOREST;
pub const TAG_SVC: u8 = kernel_models::TAG_SVC;
pub const TAG_SVR: u8 = kernel_models::TAG_SVR;
pub const TAG_ONE_CLASS_SVM: u8 = kernel_models::TAG_ONE_CLASS_SVM;
pub const TAG_KERNEL_RIDGE: u8 = kernel_models::TAG_KERNEL_RIDGE;
pub const TAG_KERNEL_PCA: u8 = kernel_models::TAG_KERNEL_PCA;
pub const TAG_MLP_CLASSIFIER: u8 = neural_network::TAG_MLP_CLASSIFIER;
pub const TAG_MLP_REGRESSOR: u8 = neural_network::TAG_MLP_REGRESSOR;
pub const TAG_GP_REGRESSOR: u8 = gaussian_process::TAG_GP_REGRESSOR;
pub const TAG_GP_CLASSIFIER: u8 = gaussian_process::TAG_GP_CLASSIFIER;
pub const TAG_DBSCAN: u8 = cluster_advanced::TAG_DBSCAN;
pub const TAG_OPTICS: u8 = cluster_advanced::TAG_OPTICS;
pub const TAG_SPECTRAL_CLUSTERING: u8 = cluster_advanced::TAG_SPECTRAL_CLUSTERING;
pub const TAG_AGGLOMERATIVE_CLUSTERING: u8 = cluster_advanced::TAG_AGGLOMERATIVE_CLUSTERING;
pub const TAG_LOCAL_OUTLIER_FACTOR: u8 = cluster_advanced::TAG_LOCAL_OUTLIER_FACTOR;
pub const TAG_NMF: u8 = cross_decomposition::TAG_NMF;
pub const TAG_CCA: u8 = cross_decomposition::TAG_CCA;
pub const TAG_PLS_REGRESSION: u8 = cross_decomposition::TAG_PLS_REGRESSION;
pub const TAG_TSNE: u8 = manifold::TAG_TSNE;
pub const TAG_MDS: u8 = manifold::TAG_MDS;

fn validate_dit_path(path: &str) -> OchreResult<()> {
    if !path.trim().to_lowercase().ends_with(".dit") {
        return Err(OchreError::InvalidModelExtension(path.to_string()));
    }
    Ok(())
}

// Minimal binary writer/reader helpers (ponytail: stdlib byte cursor, zero external crate deps)
fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn write_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(u8::from(v));
}
fn write_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u64).to_le_bytes());
}
fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_usize(buf, s.len());
    buf.extend_from_slice(s.as_bytes());
}
fn write_opt_str(buf: &mut Vec<u8>, s: &Option<String>) {
    match s {
        Some(val) => {
            write_u8(buf, 1);
            write_str(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_opt_usize(buf: &mut Vec<u8>, v: Option<usize>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_usize(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_opt_f64(buf: &mut Vec<u8>, v: Option<f64>) {
    match v {
        Some(val) => {
            write_u8(buf, 1);
            write_f64(buf, val);
        }
        None => write_u8(buf, 0),
    }
}
fn write_f64_vec(buf: &mut Vec<u8>, v: &[f64]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f64(buf, x);
    }
}
fn write_opt_f64_vec(buf: &mut Vec<u8>, v: &Option<Vec<f64>>) {
    match v {
        Some(vec) => {
            write_u8(buf, 1);
            write_f64_vec(buf, vec);
        }
        None => write_u8(buf, 0),
    }
}
fn write_opt_matrix(buf: &mut Vec<u8>, m: &Option<Matrix>) {
    match m {
        Some(mat) => {
            write_u8(buf, 1);
            write_usize(buf, mat.rows);
            write_usize(buf, mat.cols);
            write_f64_vec(buf, &mat.data);
        }
        None => write_u8(buf, 0),
    }
}
fn write_tree_node(buf: &mut Vec<u8>, node: &TreeNode) {
    match node {
        TreeNode::Leaf {
            value,
            probabilities,
        } => {
            write_u8(buf, 0);
            write_f64(buf, *value);
            write_usize(buf, probabilities.len());
            for &(cls, prob) in probabilities {
                write_f64(buf, cls);
                write_f64(buf, prob);
            }
        }
        TreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            write_u8(buf, 1);
            write_usize(buf, *feature_index);
            write_f64(buf, *threshold);
            write_tree_node(buf, left);
            write_tree_node(buf, right);
        }
    }
}
fn write_opt_tree_node(buf: &mut Vec<u8>, node: &Option<TreeNode>) {
    match node {
        Some(n) => {
            write_u8(buf, 1);
            write_tree_node(buf, n);
        }
        None => write_u8(buf, 0),
    }
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_exact(&mut self, len: usize) -> OchreResult<&'a [u8]> {
        if self.pos + len > self.data.len() {
            return Err(OchreError::CorruptedDitFile);
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }
    fn read_u8(&mut self) -> OchreResult<u8> {
        Ok(self.read_exact(1)?[0])
    }
    fn read_bool(&mut self) -> OchreResult<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_usize(&mut self) -> OchreResult<usize> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()) as usize)
    }
    fn read_f64(&mut self) -> OchreResult<f64> {
        let b = self.read_exact(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_str(&mut self) -> OchreResult<String> {
        let len = self.read_usize()?;
        let b = self.read_exact(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| OchreError::InvalidUtf8InDit)
    }
    fn read_opt_str(&mut self) -> OchreResult<Option<String>> {
        if self.read_bool()? {
            Ok(Some(self.read_str()?))
        } else {
            Ok(None)
        }
    }
    fn read_opt_usize(&mut self) -> OchreResult<Option<usize>> {
        if self.read_bool()? {
            Ok(Some(self.read_usize()?))
        } else {
            Ok(None)
        }
    }
    fn read_opt_f64(&mut self) -> OchreResult<Option<f64>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64()?))
        } else {
            Ok(None)
        }
    }
    fn read_f64_vec(&mut self) -> OchreResult<Vec<f64>> {
        let len = self.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f64()?);
        }
        Ok(vec)
    }
    fn read_opt_f64_vec(&mut self) -> OchreResult<Option<Vec<f64>>> {
        if self.read_bool()? {
            Ok(Some(self.read_f64_vec()?))
        } else {
            Ok(None)
        }
    }
    fn read_opt_matrix(&mut self) -> OchreResult<Option<Matrix>> {
        if self.read_bool()? {
            let rows = self.read_usize()?;
            let cols = self.read_usize()?;
            let data = self.read_f64_vec()?;
            Ok(Some(Matrix::from_vec(rows, cols, data)?))
        } else {
            Ok(None)
        }
    }
    fn read_tree_node(&mut self) -> OchreResult<TreeNode> {
        match self.read_u8()? {
            0 => {
                let value = self.read_f64()?;
                let len = self.read_usize()?;
                let mut probabilities = Vec::with_capacity(len);
                for _ in 0..len {
                    probabilities.push((self.read_f64()?, self.read_f64()?));
                }
                Ok(TreeNode::Leaf {
                    value,
                    probabilities,
                })
            }
            1 => {
                let feature_index = self.read_usize()?;
                let threshold = self.read_f64()?;
                let left = Box::new(self.read_tree_node()?);
                let right = Box::new(self.read_tree_node()?);
                Ok(TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                })
            }
            _ => Err(OchreError::InvalidTreeNodeTag),
        }
    }
    fn read_opt_tree_node(&mut self) -> OchreResult<Option<TreeNode>> {
        if self.read_bool()? {
            Ok(Some(self.read_tree_node()?))
        } else {
            Ok(None)
        }
    }
}

fn check_magic_and_tag(reader: &mut ByteReader<'_>, expected_tag: u8) -> OchreResult<()> {
    let magic = reader.read_exact(4)?;
    if magic != DIT_MAGIC {
        return Err(OchreError::InvalidDitHeader);
    }
    let tag = reader.read_u8()?;
    if tag != expected_tag {
        return Err(OchreError::DitTagMismatch {
            expected: expected_tag,
            actual: tag,
        });
    }
    Ok(())
}

fn resolve_loaded_device(
    saved_device: Option<String>,
    is_fitted: bool,
    device: Option<&str>,
) -> OchreResult<(device::DeviceType, Option<String>)> {
    let dev = parse_device(device.or(Some("auto")))?;
    let device_ = if saved_device.is_some() || is_fitted {
        Some(dev.to_string())
    } else {
        None
    };
    Ok((dev, device_))
}

fn matrix_to_nested_vec(m: &Matrix) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let offset = r * m.cols;
        out.push(m.data[offset..offset + m.cols].to_vec());
    }
    out
}

impl LinearRegression {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_LINEAR_REGRESSION);
        write_bool(&mut buf, self.fit_intercept);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_LINEAR_REGRESSION)?;
        let fit_intercept = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let (dev, device_) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        let (is_multi_target_, n_targets_, coef_multi_, intercept_multi_) = if r.pos < r.data.len()
        {
            let is_multi = r.read_bool()?;
            let n_targets = r.read_usize()?;
            let coef_multi = r.read_opt_matrix()?;
            let intercept_multi = r.read_opt_f64_vec()?;
            (is_multi, n_targets, coef_multi, intercept_multi)
        } else {
            let coef_multi = match &coef_ {
                Some(c) => Some(Matrix::from_vec(1, c.len(), c.clone())?),
                None => None,
            };
            let intercept_multi = intercept_.map(|v| vec![v]);
            (false, 1, coef_multi, intercept_multi)
        };

        Ok(Self {
            fit_intercept,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl Ridge {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_RIDGE);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_RIDGE)?;
        let alpha = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let is_multi_target_ = r.read_bool()?;
        let n_targets_ = r.read_usize()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let (dev, device_) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            alpha,
            fit_intercept,
            device: dev,
            device_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl RidgeClassifier {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_RIDGE_CLASSIFIER);
        write_f64(&mut buf, self.alpha);
        write_bool(&mut buf, self.fit_intercept);
        write_opt_str(&mut buf, &self.device_);
        write_f64_vec(&mut buf, &self.classes_);
        write_bool(&mut buf, self.is_multiclass_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_RIDGE_CLASSIFIER)?;
        let alpha = r.read_f64()?;
        let fit_intercept = r.read_bool()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_f64_vec()?;
        let is_multiclass_ = r.read_bool()?;
        let n_features_in_ = r.read_opt_usize()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let (dev, device_) = resolve_loaded_device(saved_device, !classes_.is_empty(), device)?;

        Ok(Self {
            alpha,
            fit_intercept,
            device: dev,
            device_,
            classes_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multiclass_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl RidgeCV {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_RIDGE_CV);
        write_f64_vec(&mut buf, &self.alphas);
        write_bool(&mut buf, self.fit_intercept);
        write_opt_usize(&mut buf, self.cv);
        write_opt_str(&mut buf, &self.device_);
        write_opt_f64(&mut buf, self.alpha_);
        write_opt_f64(&mut buf, self.best_score_);
        write_opt_f64_vec(&mut buf, &self.coef_);
        write_opt_f64(&mut buf, self.intercept_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_multi_target_);
        write_usize(&mut buf, self.n_targets_);
        write_opt_matrix(&mut buf, &self.coef_multi_);
        write_opt_f64_vec(&mut buf, &self.intercept_multi_);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_RIDGE_CV)?;
        let alphas = r.read_f64_vec()?;
        let fit_intercept = r.read_bool()?;
        let cv = r.read_opt_usize()?;
        let saved_device = r.read_opt_str()?;
        let alpha_ = r.read_opt_f64()?;
        let best_score_ = r.read_opt_f64()?;
        let coef_ = r.read_opt_f64_vec()?;
        let intercept_ = r.read_opt_f64()?;
        let n_features_in_ = r.read_opt_usize()?;
        let is_multi_target_ = r.read_bool()?;
        let n_targets_ = r.read_usize()?;
        let coef_multi_ = r.read_opt_matrix()?;
        let intercept_multi_ = r.read_opt_f64_vec()?;
        let (dev, device_) = resolve_loaded_device(saved_device, coef_.is_some(), device)?;

        Ok(Self {
            alphas,
            fit_intercept,
            cv,
            device: dev,
            device_,
            alpha_,
            best_score_,
            coef_,
            intercept_,
            coef_multi_,
            intercept_multi_,
            is_multi_target_,
            n_targets_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl DecisionTreeClassifier {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_DT_CLASSIFIER);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.device_);
        write_f64_vec(&mut buf, &self.classes_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_tree_node(&mut buf, &self.root);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DT_CLASSIFIER)?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let saved_device = r.read_opt_str()?;
        let classes_ = r.read_f64_vec()?;
        let n_features_in_ = r.read_opt_usize()?;
        let root = r.read_opt_tree_node()?;
        let dev = parse_device(device.or(Some("auto")))?;
        let device_ = if saved_device.is_some() || root.is_some() {
            Some(dev.to_string())
        } else {
            None
        };
        Ok(Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device: dev,
            device_,
            root,
            classes_,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

impl DecisionTreeRegressor {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DIT_MAGIC);
        write_u8(&mut buf, TAG_DT_REGRESSOR);
        write_str(&mut buf, &self.criterion);
        write_opt_usize(&mut buf, self.max_depth);
        write_usize(&mut buf, self.min_samples_split);
        write_usize(&mut buf, self.min_samples_leaf);
        write_opt_str(&mut buf, &self.device_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_opt_tree_node(&mut buf, &self.root);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_DT_REGRESSOR)?;
        let criterion = r.read_str()?;
        let max_depth = r.read_opt_usize()?;
        let min_samples_split = r.read_usize()?;
        let min_samples_leaf = r.read_usize()?;
        let saved_device = r.read_opt_str()?;
        let n_features_in_ = r.read_opt_usize()?;
        let root = r.read_opt_tree_node()?;
        let dev = parse_device(device.or(Some("auto")))?;
        let device_ = if saved_device.is_some() || root.is_some() {
            Some(dev.to_string())
        } else {
            None
        };
        Ok(Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device: dev,
            device_,
            root,
            n_features_in_,
        })
    }

    pub fn set_device(&mut self, device: Option<&str>) -> OchreResult<()> {
        let dev = parse_device(device)?;
        if self.device_.is_some() {
            self.device_ = Some(dev.to_string());
        }
        self.device = dev;
        Ok(())
    }
}

#[pyfunction]
pub fn get_available_devices() -> Vec<String> {
    device::get_available_devices_list()
}

#[pyfunction]
pub fn get_device_info() -> HashMap<String, String> {
    device::get_device_info_map()
}

thread_local! {
    static MATRIX_BUF_POOL: std::cell::RefCell<Vec<Vec<f64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn pop_pooled_vec(min_cap: usize) -> Vec<f64> {
    MATRIX_BUF_POOL
        .with(|pool| {
            let mut slots = pool.borrow_mut();
            if let Some(pos) = slots.iter().position(|v| v.capacity() >= min_cap) {
                Some(slots.swap_remove(pos))
            } else {
                slots.pop()
            }
        })
        .unwrap_or_default()
}

struct PooledMatrix {
    mat: Matrix,
}

impl std::ops::Deref for PooledMatrix {
    type Target = Matrix;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl Drop for PooledMatrix {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.mat.data);
        if buf.capacity() <= 16 * 1024 * 1024 {
            MATRIX_BUF_POOL.with(|pool| {
                if let Ok(mut slots) = pool.try_borrow_mut() {
                    if slots.len() < 2 {
                        slots.push(buf);
                    } else if let Some(min_slot) = slots.iter_mut().min_by_key(|v| v.capacity())
                        && buf.capacity() > min_slot.capacity()
                    {
                        *min_slot = buf;
                    }
                }
            });
        }
    }
}

fn buffer_to_matrix(py: Python<'_>, x: &pyo3::buffer::PyBuffer<f64>) -> PyResult<PooledMatrix> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Feature matrix X must be a 2D array (n_samples, n_features), got {}D ( >_< )",
            shape.len()
        )));
    }
    let rows = shape[0];
    let cols = shape[1];
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("Matrix dimensions overflow ( >_< )"))?;

    if x.is_c_contiguous() {
        let mut data = pop_pooled_vec(len);
        data.resize(len, 0.0);
        x.copy_to_slice(py, &mut data[..len]).map_err(|e| {
            PyValueError::new_err(format!("Failed to copy buffer data: {} (´；ω；`)", e))
        })?;
        return Ok(PooledMatrix {
            mat: Matrix::from_vec(rows, cols, data)?,
        });
    }

    let data = x.to_vec(py).map_err(|e| {
        PyValueError::new_err(format!("Failed to copy buffer data: {} (´；ω；`)", e))
    })?;
    Ok(PooledMatrix {
        mat: Matrix::from_vec(rows, cols, data)?,
    })
}

fn buffer_to_target_vector(py: Python<'_>, y: &pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
    let shape = y.shape();
    if shape.len() > 2 || (shape.len() == 2 && shape[1] != 1) {
        return Err(PyValueError::new_err(format!(
            "Target vector y must be a 1D vector (n_samples,) or 2D column (n_samples, 1) (・`ω´・)"
        )));
    }
    y.to_vec(py).map_err(|e| {
        PyValueError::new_err(format!("Failed to copy target buffer: {} (´；ω；`)", e))
    })
}

fn buffer_to_target_matrix(
    py: Python<'_>,
    y: &pyo3::buffer::PyBuffer<f64>,
) -> PyResult<(PooledMatrix, bool)> {
    let shape = y.shape();
    let (rows, cols, is_2d) = if shape.len() == 1 {
        (shape[0], 1, false)
    } else if shape.len() == 2 {
        (shape[0], shape[1], true)
    } else {
        return Err(PyValueError::new_err(format!(
            "Target y must be a 1D vector or 2D matrix (n_samples, n_targets), got {}D (・`ω´・)",
            shape.len()
        )));
    };
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("Target dimensions overflow ( >_< )"))?;
    if y.is_c_contiguous() {
        let mut data = pop_pooled_vec(len);
        data.resize(len, 0.0);
        y.copy_to_slice(py, &mut data[..len]).map_err(|e| {
            PyValueError::new_err(format!("Failed to copy target buffer: {} (´；ω；`)", e))
        })?;
        return Ok((
            PooledMatrix {
                mat: Matrix::from_vec(rows, cols, data)?,
            },
            is_2d,
        ));
    }
    let data = y.to_vec(py).map_err(|e| {
        PyValueError::new_err(format!("Failed to copy target buffer: {} (´；ω；`)", e))
    })?;
    let mat = Matrix::from_vec(rows, cols, data)?;
    Ok((PooledMatrix { mat }, is_2d))
}

fn buffer_to_sample_weight(py: Python<'_>, sw: &pyo3::buffer::PyBuffer<f64>) -> PyResult<Vec<f64>> {
    let shape = sw.shape();
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "sample_weight must be a 1D vector (n_samples,), got {}D ( >_< )",
            shape.len()
        )));
    }
    sw.to_vec(py).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to copy sample_weight buffer: {} (´；ω；`)",
            e
        ))
    })
}

#[pyclass(from_py_object, name = "LinearRegression")]
#[derive(Clone)]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept = true, device = None))]
    pub fn new(fit_intercept: bool, device: Option<String>) -> PyResult<Self> {
        let inner = LinearRegression::new(fit_intercept, device)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted_1d(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        self.inner
            .fit_multi(&x_mat, &y_mat, sample_weight.as_deref(), true)?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let (target_mat, is_2d) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_multi(&mat, &target_mat, sw_vec.as_deref(), is_2d)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let mat = Matrix::from_2d(&x)?;
        let preds = self.inner.predict_multi(&mat)?;
        let mut out = Vec::with_capacity(preds.rows);
        for r in 0..preds.rows {
            let offset = r * preds.cols;
            out.push(preds.data[offset..offset + preds.cols].to_vec());
        }
        Ok(out)
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        if sample_weight.is_none() && !self.inner.is_multi_target_ {
            return Ok(self.inner.score(&x_mat, &y)?);
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_multi(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_matrix(py, &x)?;
        let (y_mat, _) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score_multi(&x_mat, &y_mat, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = LinearRegression::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = LinearRegression::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = LinearRegression::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(|m| {
            let mut out = Vec::with_capacity(m.rows);
            for r in 0..m.rows {
                let offset = r * m.cols;
                out.push(m.data[offset..offset + m.cols].to_vec());
            }
            out
        })
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }

    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }

    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "DecisionTreeClassifier")]
#[derive(Clone)]
pub struct PyDecisionTreeClassifier {
    inner: DecisionTreeClassifier,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None, device = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = DecisionTreeClassifier::new(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device,
        )?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner.fit(&mat, &y)?;
        Ok(())
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        self.inner.fit(&mat, &target)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.score(&mat, &y)?)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        Ok(self.inner.score(&mat, &target)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = DecisionTreeClassifier::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = DecisionTreeClassifier::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = DecisionTreeClassifier::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn criterion(&self) -> String {
        self.inner.criterion.clone()
    }

    #[getter]
    pub fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth
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
    pub fn classes_(&self) -> Vec<f64> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "DecisionTreeRegressor")]
#[derive(Clone)]
pub struct PyDecisionTreeRegressor {
    inner: DecisionTreeRegressor,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (criterion = None, max_depth = None, min_samples_split = None, min_samples_leaf = None, device = None))]
    pub fn new(
        criterion: Option<String>,
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = DecisionTreeRegressor::new(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            device,
        )?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner.fit(&mat, &y)?;
        Ok(())
    }

    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        self.inner.fit(&mat, &target)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.score(&mat, &y)?)
    }

    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        Ok(self.inner.score(&mat, &target)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = DecisionTreeRegressor::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = DecisionTreeRegressor::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = DecisionTreeRegressor::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn criterion(&self) -> String {
        self.inner.criterion.clone()
    }

    #[getter]
    pub fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth
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
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "Ridge")]
#[derive(Clone)]
pub struct PyRidge {
    inner: Ridge,
}

#[pymethods]
impl PyRidge {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, device = None))]
    pub fn new(alpha: f64, fit_intercept: bool, device: Option<String>) -> PyResult<Self> {
        let inner = Ridge::new(alpha, fit_intercept, device)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted_1d(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        self.inner
            .fit_multi(&x_mat, &y_mat, sample_weight.as_deref(), true)?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let (target_mat, is_2d) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_multi(&mat, &target_mat, sw_vec.as_deref(), is_2d)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let mat = Matrix::from_2d(&x)?;
        let preds = self.inner.predict_multi(&mat)?;
        Ok(matrix_to_nested_vec(&preds))
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        if sample_weight.is_none() && !self.inner.is_multi_target_ {
            return Ok(self.inner.score(&x_mat, &y)?);
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_multi(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_matrix(py, &x)?;
        let (y_mat, _) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score_multi(&x_mat, &y_mat, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = Ridge::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = Ridge::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = Ridge::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }

    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }

    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "RidgeClassifier")]
#[derive(Clone)]
pub struct PyRidgeClassifier {
    inner: RidgeClassifier,
}

#[pymethods]
impl PyRidgeClassifier {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, device = None))]
    pub fn new(alpha: f64, fit_intercept: bool, device: Option<String>) -> PyResult<Self> {
        let inner = RidgeClassifier::new(alpha, fit_intercept, device)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner.fit_weighted(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.decision_function(&mat)?)
    }

    pub fn decision_function_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let mat = Matrix::from_2d(&x)?;
        let scores = self.inner.decision_function_multi(&mat)?;
        Ok(matrix_to_nested_vec(&scores))
    }

    pub fn decision_function_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.decision_function(&mat)?)
    }

    pub fn decision_function_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mat = buffer_to_matrix(py, &x)?;
        let scores = self.inner.decision_function_multi(&mat)?;
        Ok(matrix_to_nested_vec(&scores))
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.score(&mat, &y, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &target, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = RidgeClassifier::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = RidgeClassifier::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = RidgeClassifier::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn classes_(&self) -> Vec<f64> {
        self.inner.classes_.clone()
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }

    #[getter]
    pub fn is_multiclass_(&self) -> bool {
        self.inner.is_multiclass_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(from_py_object, name = "RidgeCV")]
#[derive(Clone)]
pub struct PyRidgeCV {
    inner: RidgeCV,
}

#[pymethods]
impl PyRidgeCV {
    #[new]
    #[pyo3(signature = (alphas = None, fit_intercept = true, cv = None, device = None))]
    pub fn new(
        alphas: Option<Vec<f64>>,
        fit_intercept: bool,
        cv: Option<usize>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let alpha_list = alphas.unwrap_or_else(|| vec![0.1, 1.0, 10.0]);
        let inner = RidgeCV::new(alpha_list, fit_intercept, cv, device)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted_1d(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        self.inner
            .fit_multi(&x_mat, &y_mat, sample_weight.as_deref(), true)?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let (target_mat, is_2d) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_multi(&mat, &target_mat, sw_vec.as_deref(), is_2d)?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mat = Matrix::from_2d(&x)?;
        Ok(self.inner.predict(&mat)?)
    }

    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let mat = Matrix::from_2d(&x)?;
        let preds = self.inner.predict_multi(&mat)?;
        Ok(matrix_to_nested_vec(&preds))
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        let mat = buffer_to_matrix(py, &x)?;
        Ok(self.inner.predict(&mat)?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        if sample_weight.is_none() && !self.inner.is_multi_target_ {
            return Ok(self.inner.score(&x_mat, &y)?);
        }
        let y_mat = Matrix::from_vec(y.len(), 1, y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_multi(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        let x_mat = Matrix::from_2d(&x)?;
        let y_mat = Matrix::from_2d(&y)?;
        Ok(self
            .inner
            .score_multi(&x_mat, &y_mat, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let x_mat = buffer_to_matrix(py, &x)?;
        let (y_mat, _) = buffer_to_target_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score_multi(&x_mat, &y_mat, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        let inner = RidgeCV::from_bytes(&bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        self.inner.set_device(device.as_deref())?;
        Ok(())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        let inner = RidgeCV::from_bytes(bytes, device.as_deref())?;
        Ok(Self { inner })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = RidgeCV::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn alphas(&self) -> Vec<f64> {
        self.inner.alphas.clone()
    }

    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }

    #[getter]
    pub fn cv(&self) -> Option<usize> {
        self.inner.cv
    }

    #[getter]
    pub fn alpha_(&self) -> Option<f64> {
        self.inner.alpha_
    }

    #[getter]
    pub fn best_score_(&self) -> Option<f64> {
        self.inner.best_score_
    }

    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }

    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }

    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }

    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }

    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }

    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

// --- Thin PyO3 Diagnostic & Benchmark Wrappers for Primitives N1..N6 (*≧ω≦*) ---

#[pyfunction]
#[pyo3(signature = (x_buf, device = "cpu"))]
fn _prim_n1_col_stats(
    py: Python<'_>,
    x_buf: &Bound<'_, PyAny>,
    device: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let dev = parse_device(Some(device))?;
    let buf = pyo3::buffer::PyBuffer::<f64>::get(x_buf)?;
    let x = buffer_to_matrix(py, &buf)?;
    let stats = backend::get_backend(&dev).col_reduce_stats(&x, None)?;
    Ok((stats.mean, stats.std, stats.min, stats.max))
}

fn matrix_to_pybytes<'py>(py: Python<'py>, mat: &Matrix) -> (usize, usize, Bound<'py, PyBytes>) {
    let mut raw = Vec::with_capacity(mat.data.len() * std::mem::size_of::<f64>());
    for &v in &mat.data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    (mat.rows, mat.cols, PyBytes::new(py, &raw))
}

#[pyfunction]
#[pyo3(signature = (x_buf, op = "relu", param = 0.0, device = "cpu"))]
fn _prim_n2_elementwise<'py>(
    py: Python<'py>,
    x_buf: &Bound<'py, PyAny>,
    op: &str,
    param: f64,
    device: &str,
) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
    use primitives::ElementwiseOp;
    let dev = parse_device(Some(device))?;
    let buf = pyo3::buffer::PyBuffer::<f64>::get(x_buf)?;
    let x = buffer_to_matrix(py, &buf)?;
    let ew_op = match op {
        "sigmoid" => ElementwiseOp::Sigmoid,
        "tanh" => ElementwiseOp::Tanh,
        "relu" => ElementwiseOp::Relu,
        "exp" => ElementwiseOp::Exp,
        "log" => ElementwiseOp::Log {
            eps: param.max(1e-12),
        },
        "soft_threshold" => ElementwiseOp::SoftThreshold { lambda: param },
        "affine" => ElementwiseOp::Affine {
            scale: param,
            shift: 1.0,
        },
        _ => ElementwiseOp::Clip {
            min: -param.abs(),
            max: param.abs(),
        },
    };
    let out = backend::get_backend(&dev).elementwise_op(&x, ew_op)?;
    Ok(matrix_to_pybytes(py, &out))
}

#[pyfunction]
#[pyo3(signature = (a_buf, b_buf, device = "cpu"))]
fn _prim_n3_pairwise_l2<'py>(
    py: Python<'py>,
    a_buf: &Bound<'py, PyAny>,
    b_buf: &Bound<'py, PyAny>,
    device: &str,
) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
    let dev = parse_device(Some(device))?;
    let a_py = pyo3::buffer::PyBuffer::<f64>::get(a_buf)?;
    let b_py = pyo3::buffer::PyBuffer::<f64>::get(b_buf)?;
    let a = buffer_to_matrix(py, &a_py)?;
    let b = buffer_to_matrix(py, &b_py)?;
    let out = backend::get_backend(&dev).pairwise_l2_sq(&a, &b)?;
    Ok(matrix_to_pybytes(py, &out))
}

#[pyfunction]
#[pyo3(signature = (d_buf, k = 1, device = "cpu"))]
fn _prim_n4_topk_argmin(
    py: Python<'_>,
    d_buf: &Bound<'_, PyAny>,
    k: usize,
    device: &str,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let dev = parse_device(Some(device))?;
    let d_py = pyo3::buffer::PyBuffer::<f64>::get(d_buf)?;
    let d = buffer_to_matrix(py, &d_py)?;
    let b = backend::get_backend(&dev);
    if k == 1 {
        return Ok(b.row_argmin(&d)?);
    }
    let (idx, dists) = b.topk_argmin(&d, k)?;
    Ok((idx, dists.data))
}

#[pyfunction]
#[pyo3(signature = (x_buf, center = true, device = "cpu"))]
fn _prim_n5_gram_svd(
    py: Python<'_>,
    x_buf: &Bound<'_, PyAny>,
    center: bool,
    device: &str,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
    let dev = parse_device(Some(device))?;
    let x_py = pyo3::buffer::PyBuffer::<f64>::get(x_buf)?;
    let x = buffer_to_matrix(py, &x_py)?;
    let (svals, v, mean) = backend::get_backend(&dev).gram_svd(&x, center)?;
    Ok((svals, matrix_to_nested_vec(&v), mean))
}

#[pyfunction]
#[pyo3(signature = (x_buf, grad, hess, max_bins = 64, device = "cpu"))]
fn _prim_n6_histogram_split(
    py: Python<'_>,
    x_buf: &Bound<'_, PyAny>,
    grad: Vec<f64>,
    hess: Vec<f64>,
    max_bins: usize,
    device: &str,
) -> PyResult<Option<(usize, usize, f64, f64)>> {
    let dev = parse_device(Some(device))?;
    let x_py = pyo3::buffer::PyBuffer::<f64>::get(x_buf)?;
    let x = buffer_to_matrix(py, &x_py)?;
    let b = backend::get_backend(&dev);
    let binned = b.quantize_features(&x, max_bins)?;
    let all_idx: Vec<u32> = (0..x.rows as u32).collect();
    let hist = b.batched_histogram_grad(&binned, &all_idx, &grad, &hess)?;
    let split = b.find_best_histogram_split(&binned, &hist, 1.0, 1)?;
    Ok(split.map(|s| (s.feature_idx, s.bin_idx, s.threshold, s.gain)))
}

#[pyclass(
    name = "LogisticRegression",
    module = "ochreml._ochreml",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLogisticRegression {
    inner: LogisticRegression,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (c = 1.0, fit_intercept = true, max_iter = 100, tol = 1e-4, class_weight_mode = None, class_weight_map = None, device = None))]
    pub fn new(
        c: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        class_weight_mode: Option<String>,
        class_weight_map: Option<Vec<(f64, f64)>>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let inner = LogisticRegression::new(
            c,
            fit_intercept,
            max_iter,
            tol,
            class_weight_mode,
            class_weight_map,
            device,
        )?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mat = Matrix::from_2d(&x)?;
        self.inner
            .fit_weighted(&mat, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner.fit_weighted(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&Matrix::from_2d(&x)?)?)
    }

    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&*buffer_to_matrix(py, &x)?)?)
    }

    pub fn predict_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_proba(&Matrix::from_2d(&x)?)?,
        ))
    }

    pub fn predict_proba_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_proba(&*buffer_to_matrix(py, &x)?)?,
        ))
    }

    pub fn predict_log_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_log_proba(&Matrix::from_2d(&x)?)?,
        ))
    }

    pub fn predict_log_proba_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_log_proba(&*buffer_to_matrix(py, &x)?)?,
        ))
    }

    pub fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.decision_function(&Matrix::from_2d(&x)?)?)
    }

    pub fn decision_function_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.decision_function(&*buffer_to_matrix(py, &x)?)?)
    }

    pub fn decision_function_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.decision_function_multi(&Matrix::from_2d(&x)?)?,
        ))
    }

    pub fn decision_function_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self
                .inner
                .decision_function_multi(&*buffer_to_matrix(py, &x)?)?,
        ))
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        Ok(self
            .inner
            .score(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &target, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        Ok(Self {
            inner: LogisticRegression::from_bytes(&bytes, device.as_deref())?,
        })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        Ok(self.inner.set_device(device.as_deref())?)
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: LogisticRegression::from_bytes(bytes, device.as_deref())?,
        })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = LogisticRegression::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn c(&self) -> f64 {
        self.inner.c
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn classes_(&self) -> Vec<f64> {
        self.inner.classes_.clone()
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }
    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }
    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }
    #[getter]
    pub fn is_multiclass_(&self) -> bool {
        self.inner.is_multiclass_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(name = "Lasso", module = "ochreml._ochreml", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLasso {
    inner: Lasso,
}

#[pymethods]
impl PyLasso {
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, max_iter = 1000, tol = 1e-4, positive = false, device = None))]
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        positive: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Lasso::new(alpha, fit_intercept, max_iter, tol, positive, device)?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .fit_weighted_1d(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner.fit_weighted_multi(
            &Matrix::from_2d(&x)?,
            &Matrix::from_2d(&y)?,
            sample_weight.as_deref(),
        )?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_weighted_1d(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_weighted_multi(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&Matrix::from_2d(&x)?)?)
    }
    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_multi(&Matrix::from_2d(&x)?)?,
        ))
    }
    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&*buffer_to_matrix(py, &x)?)?)
    }
    pub fn predict_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_multi(&*buffer_to_matrix(py, &x)?)?,
        ))
    }
    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        Ok(self.inner.score(&Matrix::from_2d(&x)?, &y)?)
    }
    pub fn score_multi(&self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> PyResult<f64> {
        Ok(self
            .inner
            .score_multi(&Matrix::from_2d(&x)?, &Matrix::from_2d(&y)?)?)
    }
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        Ok(self.inner.score(
            &*buffer_to_matrix(py, &x)?,
            &buffer_to_target_vector(py, &y)?,
        )?)
    }
    pub fn score_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        Ok(self
            .inner
            .score_multi(&*buffer_to_matrix(py, &x)?, &*buffer_to_matrix(py, &y)?)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        Ok(Self {
            inner: Lasso::from_bytes(&bytes, device.as_deref())?,
        })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        Ok(self.inner.set_device(device.as_deref())?)
    }
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Lasso::from_bytes(bytes, device.as_deref())?,
        })
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = Lasso::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn positive(&self) -> bool {
        self.inner.positive
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }
    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }
    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }
    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }
    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }
    #[getter]
    pub fn n_iter_(&self) -> Vec<usize> {
        self.inner.n_iter_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(name = "ElasticNet", module = "ochreml._ochreml", skip_from_py_object)]
#[derive(Clone)]
pub struct PyElasticNet {
    inner: ElasticNet,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[pyo3(signature = (alpha = 1.0, l1_ratio = 0.5, fit_intercept = true, max_iter = 1000, tol = 1e-4, positive = false, device = None))]
    pub fn new(
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        positive: bool,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ElasticNet::new(
                alpha,
                l1_ratio,
                fit_intercept,
                max_iter,
                tol,
                positive,
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .fit_weighted_1d(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner.fit_weighted_multi(
            &Matrix::from_2d(&x)?,
            &Matrix::from_2d(&y)?,
            sample_weight.as_deref(),
        )?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_weighted_1d(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_multi_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_matrix(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner
            .fit_weighted_multi(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&Matrix::from_2d(&x)?)?)
    }
    pub fn predict_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_multi(&Matrix::from_2d(&x)?)?,
        ))
    }
    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&*buffer_to_matrix(py, &x)?)?)
    }
    pub fn predict_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_multi(&*buffer_to_matrix(py, &x)?)?,
        ))
    }
    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        Ok(self.inner.score(&Matrix::from_2d(&x)?, &y)?)
    }
    pub fn score_multi(&self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> PyResult<f64> {
        Ok(self
            .inner
            .score_multi(&Matrix::from_2d(&x)?, &Matrix::from_2d(&y)?)?)
    }
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        Ok(self.inner.score(
            &*buffer_to_matrix(py, &x)?,
            &buffer_to_target_vector(py, &y)?,
        )?)
    }
    pub fn score_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        Ok(self
            .inner
            .score_multi(&*buffer_to_matrix(py, &x)?, &*buffer_to_matrix(py, &y)?)?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        Ok(Self {
            inner: ElasticNet::from_bytes(&bytes, device.as_deref())?,
        })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        Ok(self.inner.set_device(device.as_deref())?)
    }
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: ElasticNet::from_bytes(bytes, device.as_deref())?,
        })
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = ElasticNet::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn l1_ratio(&self) -> f64 {
        self.inner.l1_ratio
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> f64 {
        self.inner.tol
    }
    #[getter]
    pub fn positive(&self) -> bool {
        self.inner.positive
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }
    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }
    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }
    #[getter]
    pub fn is_multi_target_(&self) -> bool {
        self.inner.is_multi_target_
    }
    #[getter]
    pub fn n_targets_(&self) -> usize {
        self.inner.n_targets_
    }
    #[getter]
    pub fn n_iter_(&self) -> Vec<usize> {
        self.inner.n_iter_.clone()
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(
    name = "SGDClassifier",
    module = "ochreml._ochreml",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySGDClassifier {
    inner: SGDClassifier,
}

#[pymethods]
impl PySGDClassifier {
    #[new]
    #[pyo3(signature = (loss = "hinge".to_string(), penalty = "l2".to_string(), alpha = 0.0001, l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = Some(1e-3), shuffle = true, learning_rate = "optimal".to_string(), eta0 = 0.0, power_t = 0.5, random_state = None, class_weight_mode = None, class_weight_map = None, device = None))]
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
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SGDClassifier::new(
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
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .fit_weighted(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner.fit_weighted(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&Matrix::from_2d(&x)?)?)
    }
    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&*buffer_to_matrix(py, &x)?)?)
    }
    pub fn predict_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_proba(&Matrix::from_2d(&x)?)?,
        ))
    }
    pub fn predict_proba_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_proba(&*buffer_to_matrix(py, &x)?)?,
        ))
    }
    pub fn predict_log_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_log_proba(&Matrix::from_2d(&x)?)?,
        ))
    }
    pub fn predict_log_proba_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.predict_log_proba(&*buffer_to_matrix(py, &x)?)?,
        ))
    }
    pub fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.decision_function(&Matrix::from_2d(&x)?)?)
    }
    pub fn decision_function_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.decision_function(&*buffer_to_matrix(py, &x)?)?)
    }
    pub fn decision_function_multi(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self.inner.decision_function_multi(&Matrix::from_2d(&x)?)?,
        ))
    }
    pub fn decision_function_multi_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        Ok(matrix_to_nested_vec(
            &self
                .inner
                .decision_function_multi(&*buffer_to_matrix(py, &x)?)?,
        ))
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score(
        &self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        Ok(self
            .inner
            .score(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?)
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<f64> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        Ok(self.inner.score(&mat, &target, sw_vec.as_deref())?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        Ok(Self {
            inner: SGDClassifier::from_bytes(&bytes, device.as_deref())?,
        })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        Ok(self.inner.set_device(device.as_deref())?)
    }
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: SGDClassifier::from_bytes(bytes, device.as_deref())?,
        })
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = SGDClassifier::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn loss(&self) -> String {
        self.inner.loss.clone()
    }
    #[getter]
    pub fn penalty(&self) -> String {
        self.inner.penalty.clone()
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn l1_ratio(&self) -> f64 {
        self.inner.l1_ratio
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> Option<f64> {
        self.inner.tol
    }
    #[getter]
    pub fn shuffle(&self) -> bool {
        self.inner.shuffle
    }
    #[getter]
    pub fn learning_rate(&self) -> String {
        self.inner.learning_rate.clone()
    }
    #[getter]
    pub fn eta0(&self) -> f64 {
        self.inner.eta0
    }
    #[getter]
    pub fn power_t(&self) -> f64 {
        self.inner.power_t
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn classes_(&self) -> Vec<f64> {
        self.inner.classes_.clone()
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn coef_multi_(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coef_multi_.as_ref().map(matrix_to_nested_vec)
    }
    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }
    #[getter]
    pub fn intercept_multi_(&self) -> Option<Vec<f64>> {
        self.inner.intercept_multi_.clone()
    }
    #[getter]
    pub fn is_multiclass_(&self) -> bool {
        self.inner.is_multiclass_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn t_(&self) -> f64 {
        self.inner.t_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pyclass(
    name = "SGDRegressor",
    module = "ochreml._ochreml",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySGDRegressor {
    inner: SGDRegressor,
}

#[pymethods]
impl PySGDRegressor {
    #[new]
    #[pyo3(signature = (loss = "squared_error".to_string(), penalty = "l2".to_string(), alpha = 0.0001, l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = Some(1e-3), shuffle = true, learning_rate = "invscaling".to_string(), eta0 = 0.01, power_t = 0.25, epsilon = 0.1, random_state = None, device = None))]
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
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SGDRegressor::new(
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
                device,
            )?,
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .fit_weighted(&Matrix::from_2d(&x)?, &y, sample_weight.as_deref())?;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn fit_buffer(
        &mut self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
        sample_weight: Option<pyo3::buffer::PyBuffer<f64>>,
    ) -> PyResult<()> {
        let mat = buffer_to_matrix(py, &x)?;
        let target = buffer_to_target_vector(py, &y)?;
        let sw_vec = match sample_weight {
            Some(ref sw_buf) => Some(buffer_to_sample_weight(py, sw_buf)?),
            None => None,
        };
        self.inner.fit_weighted(&mat, &target, sw_vec.as_deref())?;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&Matrix::from_2d(&x)?)?)
    }
    pub fn predict_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(&*buffer_to_matrix(py, &x)?)?)
    }
    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        Ok(self.inner.score(&Matrix::from_2d(&x)?, &y)?)
    }
    pub fn score_buffer(
        &self,
        py: Python<'_>,
        x: pyo3::buffer::PyBuffer<f64>,
        y: pyo3::buffer::PyBuffer<f64>,
    ) -> PyResult<f64> {
        Ok(self.inner.score(
            &*buffer_to_matrix(py, &x)?,
            &buffer_to_target_vector(py, &y)?,
        )?)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        validate_dit_path(path)?;
        fs::write(path, self.inner.to_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Failed to write .dit file: {} (´；ω；`)", e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, device = None))]
    pub fn load(path: &str, device: Option<String>) -> PyResult<Self> {
        validate_dit_path(path)?;
        let bytes = fs::read(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to read .dit file: {} (´；ω；`)", e))
        })?;
        Ok(Self {
            inner: SGDRegressor::from_bytes(&bytes, device.as_deref())?,
        })
    }

    #[pyo3(signature = (device = None))]
    pub fn set_device(&mut self, device: Option<String>) -> PyResult<()> {
        Ok(self.inner.set_device(device.as_deref())?)
    }
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    #[staticmethod]
    #[pyo3(signature = (bytes, device = None))]
    pub fn from_bytes(bytes: &[u8], device: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: SGDRegressor::from_bytes(bytes, device.as_deref())?,
        })
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = SGDRegressor::from_bytes(state, None)?;
        Ok(())
    }

    #[getter]
    pub fn loss(&self) -> String {
        self.inner.loss.clone()
    }
    #[getter]
    pub fn penalty(&self) -> String {
        self.inner.penalty.clone()
    }
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    pub fn l1_ratio(&self) -> f64 {
        self.inner.l1_ratio
    }
    #[getter]
    pub fn fit_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }
    #[getter]
    pub fn tol(&self) -> Option<f64> {
        self.inner.tol
    }
    #[getter]
    pub fn shuffle(&self) -> bool {
        self.inner.shuffle
    }
    #[getter]
    pub fn learning_rate(&self) -> String {
        self.inner.learning_rate.clone()
    }
    #[getter]
    pub fn eta0(&self) -> f64 {
        self.inner.eta0
    }
    #[getter]
    pub fn power_t(&self) -> f64 {
        self.inner.power_t
    }
    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }
    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }
    #[getter]
    pub fn coef_(&self) -> Option<Vec<f64>> {
        self.inner.coef_.clone()
    }
    #[getter]
    pub fn intercept_(&self) -> Option<f64> {
        self.inner.intercept_
    }
    #[getter]
    pub fn n_iter_(&self) -> usize {
        self.inner.n_iter_
    }
    #[getter]
    pub fn t_(&self) -> f64 {
        self.inner.t_
    }
    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }
    #[getter]
    pub fn device_(&self) -> Option<String> {
        self.inner.device_.clone()
    }
}

#[pymodule]
fn _ochreml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n1_col_stats, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n2_elementwise, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n3_pairwise_l2, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n4_topk_argmin, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n5_gram_svd, m)?)?;
    m.add_function(wrap_pyfunction!(_prim_n6_histogram_split, m)?)?;
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyRidge>()?;
    m.add_class::<PyRidgeClassifier>()?;
    m.add_class::<PyRidgeCV>()?;
    m.add_class::<PyDecisionTreeClassifier>()?;
    m.add_class::<PyDecisionTreeRegressor>()?;
    m.add_class::<PyLogisticRegression>()?;
    m.add_class::<PyLasso>()?;
    m.add_class::<PyElasticNet>()?;
    m.add_class::<PySGDClassifier>()?;
    m.add_class::<PySGDRegressor>()?;
    preprocessing::register_py_module(m)?;
    decomposition::register_py_classes(m)?;
    discriminant::register_py_classes(m)?;
    cluster::register_py_classes(m)?;
    glm::register_py_classes(m)?;
    linear_svm::register_py_classes(m)?;
    neighbors::register_py_classes(m)?;
    naive_bayes::register_py_classes(m)?;
    mixture::register_py_classes(m)?;
    hist_gradient_boosting::register_py_classes(m)?;
    forest::register_py_classes(m)?;
    boosting::register_py_classes(m)?;
    kernel_models::register_py_classes(m)?;
    neural_network::register_py_classes(m)?;
    gaussian_process::register_py_classes(m)?;
    cluster_advanced::register_py_classes(m)?;
    cross_decomposition::register_py_classes(m)?;
    manifold::register_py_classes(m)?;
    Ok(())
}
