// OchreML Kernel Approximation Module
// Implements RBFSampler (Random Fourier Features) accelerated by Backend E1 matmul and Rayon.

use crate::backend::get_backend;
use crate::device::{DeviceType, parse_device, resolve_loaded_device};
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_bool, write_header, write_opt_f64, write_opt_f64_vec,
    write_opt_matrix, write_opt_str, write_opt_u64, write_opt_usize, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::preprocessing::py_helpers::{
    buffer_to_pooled_matrix, f64_slice_to_pybytes, matrix_to_pybytes,
};
use crate::pymethods_with_dit;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub const TAG_RBF_SAMPLER: u8 = 95;

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
    fn next_f64_01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
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
            "Input X contains NaN or Inf".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RBFSampler {
    pub gamma_val: Option<f64>,
    pub gamma_mode: Option<String>,
    pub n_components: usize,
    pub random_state: Option<u64>,
    pub device_type: DeviceType,
    pub device_: Option<String>,
    pub random_weights_: Option<Matrix>,
    pub random_offset_: Option<Vec<f64>>,
    pub gamma_: Option<f64>,
    pub n_features_in_: Option<usize>,
    pub is_fitted: bool,
}

impl RBFSampler {
    pub fn new(
        gamma_val: Option<f64>,
        gamma_mode: Option<String>,
        n_components: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> OchreResult<Self> {
        if n_components == 0 {
            return Err(OchreError::InvalidParameter(
                "n_components must be >= 1".to_string(),
            ));
        }
        match (&gamma_val, &gamma_mode) {
            (Some(g), None) => {
                if !g.is_finite() || *g <= 0.0 {
                    return Err(OchreError::InvalidParameter(format!(
                        "gamma must be a strictly positive finite float, got {}",
                        g
                    )));
                }
            }
            (None, Some(mode)) => {
                if mode != "scale" {
                    return Err(OchreError::InvalidParameter(format!(
                        "Unsupported gamma string '{}'; expected 'scale' or positive float",
                        mode
                    )));
                }
            }
            (None, None) => {}
            (Some(_), Some(_)) => {
                return Err(OchreError::InvalidParameter(
                    "Specify either gamma float or gamma mode, not both".to_string(),
                ));
            }
        }

        let dev = parse_device(device.as_deref())?;
        let device_ = device.map(|_| dev.to_string());

        Ok(Self {
            gamma_val,
            gamma_mode: if gamma_val.is_none() && gamma_mode.is_none() {
                Some("scale".to_string())
            } else {
                gamma_mode
            },
            n_components,
            random_state,
            device_type: dev,
            device_,
            random_weights_: None,
            random_offset_: None,
            gamma_: None,
            n_features_in_: None,
            is_fitted: false,
        })
    }

    pub fn set_device(&mut self, device: Option<String>) -> OchreResult<()> {
        let dev = parse_device(device.as_deref())?;
        self.device_type = dev.clone();
        self.device_ = Some(dev.to_string());
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix) -> OchreResult<()> {
        validate_finite_matrix(x)?;
        let p = x.cols;
        let backend = get_backend(&self.device_type);

        let resolved_gamma = match (&self.gamma_val, self.gamma_mode.as_deref()) {
            (Some(g), _) => *g,
            (None, Some("scale")) | (None, None) => {
                let stats = backend.col_reduce_stats(x, None)?;
                let pf = p as f64;
                let global_mean = stats.mean.iter().sum::<f64>() / pf;
                let global_var = stats
                    .var
                    .iter()
                    .zip(stats.mean.iter())
                    .map(|(&v, &m)| v + (m - global_mean) * (m - global_mean))
                    .sum::<f64>()
                    / pf;
                if global_var <= 0.0 || !global_var.is_finite() {
                    1.0
                } else {
                    1.0 / (pf * global_var)
                }
            }
            (None, Some(other)) => {
                return Err(OchreError::InvalidParameter(format!(
                    "Unsupported gamma mode '{}'",
                    other
                )));
            }
        };

        let mut rng = SplitMix64::new(self.random_state);
        let std_dev = (2.0 * resolved_gamma).sqrt();
        let mut weights = Vec::with_capacity(p * self.n_components);
        for _ in 0..(p * self.n_components) {
            weights.push(std_dev * rng.next_standard_normal());
        }
        let weights_mat = Matrix::from_vec(p, self.n_components, weights)?;

        let two_pi = 2.0 * std::f64::consts::PI;
        let mut offset = Vec::with_capacity(self.n_components);
        for _ in 0..self.n_components {
            offset.push(rng.next_f64_01() * two_pi);
        }

        self.random_weights_ = Some(weights_mat);
        self.random_offset_ = Some(offset);
        self.gamma_ = Some(resolved_gamma);
        self.n_features_in_ = Some(p);
        self.is_fitted = true;
        if self.device_.is_none() {
            self.device_ = Some(self.device_type.to_string());
        }
        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> OchreResult<Matrix> {
        if !self.is_fitted {
            return Err(OchreError::NotFitted);
        }
        validate_finite_matrix(x)?;
        let expected_p = self.n_features_in_.ok_or(OchreError::NotFitted)?;
        if x.cols != expected_p {
            return Err(OchreError::FeatureCountMismatch {
                features: x.cols,
                expected: expected_p,
            });
        }
        let weights = self.random_weights_.as_ref().ok_or(OchreError::NotFitted)?;
        let offset = self.random_offset_.as_ref().ok_or(OchreError::NotFitted)?;

        let backend = get_backend(&self.device_type);
        let mut proj = backend.matmul(x, weights)?;
        let n_comp = self.n_components;
        let norm_factor = (2.0 / (n_comp as f64)).sqrt();

        proj.data.par_chunks_mut(n_comp).for_each(|row| {
            for j in 0..n_comp {
                row[j] = (row[j] + offset[j]).cos() * norm_factor;
            }
        });

        Ok(proj)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> OchreResult<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_RBF_SAMPLER);
        write_opt_f64(&mut buf, self.gamma_val);
        write_opt_str(&mut buf, &self.gamma_mode);
        write_usize(&mut buf, self.n_components);
        write_opt_u64(&mut buf, self.random_state);
        write_opt_str(&mut buf, &self.device_);
        write_opt_matrix(&mut buf, &self.random_weights_);
        write_opt_f64_vec(&mut buf, &self.random_offset_);
        write_opt_f64(&mut buf, self.gamma_);
        write_opt_usize(&mut buf, self.n_features_in_);
        write_bool(&mut buf, self.is_fitted);
        buf
    }

    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> OchreResult<Self> {
        let mut reader = ByteReader::new(bytes);
        check_magic_and_tag(&mut reader, TAG_RBF_SAMPLER)?;

        let gamma_val = reader.read_opt_f64()?;
        let gamma_mode = reader.read_opt_str()?;
        let n_components = reader.read_usize()?;
        let random_state = reader.read_opt_u64()?;
        let saved_device = reader.read_opt_str()?;
        let random_weights_ = reader.read_opt_matrix()?;
        let random_offset_ = reader.read_opt_f64_vec()?;
        let gamma_ = reader.read_opt_f64()?;
        let n_features_in_ = reader.read_opt_usize()?;
        let is_fitted = reader.read_bool()?;

        let (device_type, device_) = resolve_loaded_device(saved_device, is_fitted, device)?;

        Ok(Self {
            gamma_val,
            gamma_mode,
            n_components,
            random_state,
            device_type,
            device_,
            random_weights_,
            random_offset_,
            gamma_,
            n_features_in_,
            is_fitted,
        })
    }
}

#[pyclass(from_py_object, name = "RBFSampler")]
#[derive(Clone)]
pub struct PyRBFSampler {
    inner: RBFSampler,
}

pymethods_with_dit!(PyRBFSampler, RBFSampler, {
    #[new]
    #[pyo3(signature = (gamma_val = None, gamma_mode = None, n_components = 100, random_state = None, device = None))]
    pub fn new(
        gamma_val: Option<f64>,
        gamma_mode: Option<String>,
        n_components: usize,
        random_state: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RBFSampler::new(gamma_val, gamma_mode, n_components, random_state, device)?,
        })
    }

    pub fn fit_buffer(&mut self, py: Python<'_>, x: PyBuffer<f64>) -> PyResult<()> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        self.inner.fit(&mat)?;
        Ok(())
    }

    pub fn transform_buffer<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f64>,
    ) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let mat = buffer_to_pooled_matrix(py, &x)?;
        let out = self.inner.transform(&mat)?;
        matrix_to_pybytes(py, &out)
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
    pub fn gamma_val(&self) -> Option<f64> {
        self.inner.gamma_val
    }

    #[getter]
    pub fn gamma_mode(&self) -> Option<String> {
        self.inner.gamma_mode.clone()
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }

    #[getter]
    pub fn random_state(&self) -> Option<u64> {
        self.inner.random_state
    }

    #[getter]
    pub fn gamma_(&self) -> Option<f64> {
        self.inner.gamma_
    }

    #[getter]
    pub fn n_features_in_(&self) -> Option<usize> {
        self.inner.n_features_in_
    }

    #[getter]
    pub fn random_weights_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyBytes>)>> {
        match &self.inner.random_weights_ {
            Some(m) => Ok(Some(matrix_to_pybytes(py, m)?)),
            None => Ok(None),
        }
    }

    #[getter]
    pub fn random_offset_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.inner.random_offset_ {
            Some(v) => Ok(Some(f64_slice_to_pybytes(py, v)?)),
            None => Ok(None),
        }
    }
});

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRBFSampler>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_sampler_basic_and_determinism() {
        let x = Matrix::from_2d(&[
            vec![0.0, 1.0, -1.0],
            vec![1.0, 0.5, 0.2],
            vec![-0.5, 1.5, 0.8],
            vec![0.3, -0.7, 1.1],
        ])
        .ok();
        let x = match x {
            Some(m) => m,
            None => return,
        };

        let mut s1 = match RBFSampler::new(Some(0.5), None, 64, Some(42), Some("cpu".to_string())) {
            Ok(s) => s,
            Err(e) => panic!("unexpected err: {}", e),
        };
        let mut s2 = match RBFSampler::new(Some(0.5), None, 64, Some(42), Some("cpu".to_string())) {
            Ok(s) => s,
            Err(e) => panic!("unexpected err: {}", e),
        };

        let z1 = match s1.fit_transform(&x) {
            Ok(m) => m,
            Err(e) => panic!("fit_transform failed: {}", e),
        };
        let z2 = match s2.fit_transform(&x) {
            Ok(m) => m,
            Err(e) => panic!("fit_transform failed: {}", e),
        };
        assert_eq!(z1.rows, 4);
        assert_eq!(z1.cols, 64);
        assert_eq!(z1.data, z2.data);

        // .dit roundtrip
        let bytes = s1.to_bytes();
        let loaded = match RBFSampler::from_bytes(&bytes, Some("cpu")) {
            Ok(m) => m,
            Err(e) => panic!("from_bytes failed: {}", e),
        };
        let z_loaded = match loaded.transform(&x) {
            Ok(m) => m,
            Err(e) => panic!("transform failed: {}", e),
        };
        assert_eq!(z1.data, z_loaded.data);
    }

    #[test]
    fn test_rbf_sampler_scale_and_kernel_approx() {
        let x = Matrix::from_2d(&[vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]).ok();
        let x = match x {
            Some(m) => m,
            None => return,
        };
        let mut s = match RBFSampler::new(
            None,
            Some("scale".to_string()),
            500,
            Some(7),
            Some("cpu".to_string()),
        ) {
            Ok(s) => s,
            Err(e) => panic!("unexpected err: {}", e),
        };
        let z = match s.fit_transform(&x) {
            Ok(m) => m,
            Err(e) => panic!("fit_transform failed: {}", e),
        };
        // Dot product of identical rows 0 and 1 should be ~1.0
        let mut dot01 = 0.0;
        for j in 0..500 {
            dot01 += z[(0, j)] * z[(1, j)];
        }
        assert!((dot01 - 1.0).abs() < 0.15, "dot01={}", dot01);
    }
}
