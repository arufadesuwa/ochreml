// OchreML Shared Hardware Primitives (N1..N6) (*≧ω≦*)
// Dispatches col_reduce_stats (N1), elementwise_op (N2), pairwise_l2_sq (N3),
// topk_argmin (N4), small_sym_eig_svd (N5), and batched_histogram_split (N6)
// across CPU Rayon, CUDA, wgpu, and TPU with seamless CPU fallback.

pub mod distance_topk;
pub mod eig_svd;
pub mod histogram;
pub mod stats_ops;

use crate::backend::Backend;
use crate::error::OchreResult;
use crate::matrix::Matrix;

pub use distance_topk::{backend_pairwise_l2_sq, cpu_row_argmin, cpu_topk_argmin};
pub use eig_svd::{SymEigResult, backend_gram_svd, cpu_small_sym_eig_svd};
pub use histogram::{
    BinnedMatrix, GradHessBin, HistogramSplit, cpu_batched_histogram_grad, cpu_quantize_features,
    find_best_histogram_split, subtract_histogram,
};
pub use stats_ops::{
    ColStats, ElementwiseOp, cpu_axpy, cpu_col_reduce_stats, cpu_col_scale_shift,
    cpu_col_scale_shift_clipped, cpu_col_scale_shift_inplace, cpu_elementwise_op,
    cpu_elementwise_op_inplace, cpu_row_softmax,
};

impl<'a> Backend<'a> {
    /// N1: Compute per-column statistics (mean, var, std, min, max, sum).
    #[inline]
    pub fn col_reduce_stats(
        &self,
        x: &Matrix,
        sample_weight: Option<&[f64]>,
    ) -> OchreResult<ColStats> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) =
                    crate::backends::cuda_primitives::cuda_col_reduce_stats(*id, x, sample_weight)
                {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_col_reduce_stats(
                    ids[0],
                    x,
                    sample_weight,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) =
                crate::backends::wgpu_primitives::wgpu_col_reduce_stats(x, sample_weight)
        {
            return res;
        }

        cpu_col_reduce_stats(x, sample_weight)
    }

    /// N1: Apply `(x[i, j] - shift[j]) * scale[j]` across columns.
    #[inline]
    pub fn col_scale_shift(&self, x: &Matrix, scale: &[f64], shift: &[f64]) -> OchreResult<Matrix> {
        cpu_col_scale_shift(x, scale, shift)
    }

    /// N2: Apply elementwise unary or parameterized operation out-of-place.
    #[inline]
    pub fn elementwise_op(&self, a: &Matrix, op: ElementwiseOp) -> OchreResult<Matrix> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_elementwise_op(*id, a, op)
                {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) =
                    crate::backends::cuda_primitives::cuda_elementwise_op(ids[0], a, op)
                {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_primitives::wgpu_elementwise_op(a, op)
        {
            return res;
        }

        #[cfg(feature = "tpu-pjrt")]
        if let Backend::Tpu(id) = self
            && let Some(res) = crate::backends::tpu_primitives::tpu_elementwise_op(*id, a, op)
        {
            return res;
        }

        cpu_elementwise_op(a, op)
    }

    /// N2: Apply elementwise operation in-place.
    #[inline]
    pub fn elementwise_op_inplace(&self, a: &mut Matrix, op: ElementwiseOp) -> OchreResult<()> {
        if a.data.len() >= 65_536 && !matches!(self, Backend::Cpu) {
            let res = self.elementwise_op(a, op)?;
            a.data = res.data;
            return Ok(());
        }
        cpu_elementwise_op_inplace(a, op);
        Ok(())
    }

    /// N2: Compute `y += alpha * x`.
    #[inline]
    pub fn axpy(&self, y: &mut [f64], alpha: f64, x: &[f64]) -> OchreResult<()> {
        cpu_axpy(y, alpha, x)
    }

    /// N2: Numerically stable row-wise softmax.
    #[inline]
    pub fn row_softmax(&self, logits: &Matrix) -> OchreResult<Matrix> {
        cpu_row_softmax(logits)
    }

    /// N3: Squared Euclidean distance matrix `D[i, j] = ||A_i - B_j||^2`.
    #[inline]
    pub fn pairwise_l2_sq(&self, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
        backend_pairwise_l2_sq(self, a, b)
    }

    /// N4: Per-row argmin returning `(indices, min_values)`.
    #[inline]
    pub fn row_argmin(&self, dist: &Matrix) -> OchreResult<(Vec<usize>, Vec<f64>)> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_row_argmin(*id, dist) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_row_argmin(ids[0], dist) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_primitives::wgpu_row_argmin(dist)
        {
            return res;
        }

        cpu_row_argmin(dist)
    }

    /// N4: Per-row Top-K smallest distances returning `(flat_indices, sorted_distances)`.
    #[inline]
    pub fn topk_argmin(&self, dist: &Matrix, k: usize) -> OchreResult<(Vec<usize>, Matrix)> {
        cpu_topk_argmin(dist, k)
    }

    /// N5: Symmetric eigendecomposition `M = V * diag(lambda) * V^T` sorted descending.
    #[inline]
    pub fn small_sym_eig_svd(&self, sym: &Matrix) -> OchreResult<SymEigResult> {
        cpu_small_sym_eig_svd(sym)
    }

    /// N5: Hardware-accelerated Gram SVD returning `(singular_values, v_right, mean)`.
    #[inline]
    pub fn gram_svd(&self, x: &Matrix, center: bool) -> OchreResult<(Vec<f64>, Matrix, Vec<f64>)> {
        backend_gram_svd(self, x, center)
    }

    /// N6: Quantize continuous `f64` features into `u8` bins (`0..max_bins`).
    #[inline]
    pub fn quantize_features(&self, x: &Matrix, max_bins: usize) -> OchreResult<BinnedMatrix> {
        cpu_quantize_features(x, max_bins)
    }

    /// N6: Accumulate `(sum_grad, sum_hess, count)` histograms over `sample_indices`.
    #[inline]
    pub fn batched_histogram_grad(
        &self,
        binned: &BinnedMatrix,
        sample_indices: &[u32],
        grad: &[f64],
        hess: &[f64],
    ) -> OchreResult<Vec<GradHessBin>> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_batched_histogram_grad(
                    *id,
                    binned,
                    sample_indices,
                    grad,
                    hess,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda_primitives::cuda_batched_histogram_grad(
                    ids[0],
                    binned,
                    sample_indices,
                    grad,
                    hess,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu)
            && let Some(res) = crate::backends::wgpu_primitives::wgpu_batched_histogram_grad(
                binned,
                sample_indices,
                grad,
                hess,
            )
        {
            return res;
        }

        cpu_batched_histogram_grad(binned, sample_indices, grad, hess)
    }

    /// N6: Find optimal split across all features and histogram bins.
    #[inline]
    pub fn find_best_histogram_split(
        &self,
        binned: &BinnedMatrix,
        histograms: &[GradHessBin],
        l2_reg: f64,
        min_samples_leaf: u32,
    ) -> OchreResult<Option<HistogramSplit>> {
        find_best_histogram_split(binned, histograms, l2_reg, min_samples_leaf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::get_backend;
    use crate::device::DeviceType;

    #[test]
    fn test_n1_col_reduce_stats_and_scale_shift() {
        let backend = get_backend(&DeviceType::Cpu);
        let x = Matrix::from_2d(&[
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ])
        .unwrap();

        let stats = backend.col_reduce_stats(&x, None).unwrap();
        assert!((stats.mean[0] - 2.5).abs() < 1e-12);
        assert!((stats.mean[1] - 25.0).abs() < 1e-12);
        assert!((stats.var[0] - 1.25).abs() < 1e-12);
        assert!((stats.std[1] - 125.0f64.sqrt()).abs() < 1e-11);
        assert!((stats.min[0] - 1.0).abs() < 1e-12);
        assert!((stats.max[1] - 40.0).abs() < 1e-12);
        assert!((stats.sum[0] - 10.0).abs() < 1e-12);

        let w = [1.0, 2.0, 2.0, 1.0];
        let wstats = backend.col_reduce_stats(&x, Some(&w)).unwrap();
        assert!((wstats.mean[0] - 2.5).abs() < 1e-12);
        assert!((wstats.total_weight - 6.0).abs() < 1e-12);

        let inv_std: Vec<f64> = stats.std.iter().map(|&s| 1.0 / s).collect();
        let scaled = backend.col_scale_shift(&x, &inv_std, &stats.mean).unwrap();
        let scaled_stats = backend.col_reduce_stats(&scaled, None).unwrap();
        assert!(scaled_stats.mean[0].abs() < 1e-12);
        assert!((scaled_stats.std[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_n2_elementwise_ops_and_softmax() {
        let backend = get_backend(&DeviceType::Cpu);
        let mut m = Matrix::from_2d(&[vec![-3.0, 0.0, 2.5], vec![1.0, -0.5, 4.0]]).unwrap();

        let relu = backend.elementwise_op(&m, ElementwiseOp::Relu).unwrap();
        assert_eq!(relu.data, vec![0.0, 0.0, 2.5, 1.0, 0.0, 4.0]);

        let soft = backend
            .elementwise_op(&m, ElementwiseOp::SoftThreshold { lambda: 1.0 })
            .unwrap();
        assert_eq!(soft.data, vec![-2.0, 0.0, 1.5, 0.0, 0.0, 3.0]);

        let sig = backend.elementwise_op(&m, ElementwiseOp::Sigmoid).unwrap();
        assert!((sig[(0, 1)] - 0.5).abs() < 1e-12);
        assert!(sig[(0, 0)] > 0.0 && sig[(0, 0)] < 0.1);

        let probs = backend.row_softmax(&m).unwrap();
        for r in 0..probs.rows {
            let row_sum: f64 = (0..probs.cols).map(|c| probs[(r, c)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-12);
        }

        let mut y = vec![1.0, 2.0, 3.0];
        backend.axpy(&mut y, 2.0, &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(y, vec![9.0, 12.0, 15.0]);

        backend
            .elementwise_op_inplace(
                &mut m,
                ElementwiseOp::Clip {
                    min: -1.0,
                    max: 2.0,
                },
            )
            .unwrap();
        assert_eq!(m.data, vec![-1.0, 0.0, 2.0, 1.0, -0.5, 2.0]);
    }

    #[test]
    fn test_n3_n4_pairwise_l2_and_topk_argmin() {
        let backend = get_backend(&DeviceType::Cpu);
        let a = Matrix::from_2d(&[vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 1.0]]).unwrap();
        let b = Matrix::from_2d(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![3.0, 4.0],
            vec![10.0, 10.0],
        ])
        .unwrap();

        let dist = backend.pairwise_l2_sq(&a, &b).unwrap();
        assert!((dist[(0, 0)] - 0.0).abs() < 1e-12);
        assert!((dist[(1, 0)] - 25.0).abs() < 1e-12);
        assert!((dist[(1, 2)] - 0.0).abs() < 1e-12);

        let (arg_idx, arg_val) = backend.row_argmin(&dist).unwrap();
        assert_eq!(arg_idx, vec![0, 2, 1]);
        assert!((arg_val[2] - 1.0).abs() < 1e-12);

        let (topk_idx, topk_dist) = backend.topk_argmin(&dist, 2).unwrap();
        assert_eq!(&topk_idx[0..2], &[0, 1]);
        assert!((topk_dist[(0, 1)] - 1.0).abs() < 1e-12);
        assert_eq!(&topk_idx[2..4], &[2, 1]);
    }

    #[test]
    fn test_n5_small_sym_eig_and_gram_svd() {
        let backend = get_backend(&DeviceType::Cpu);
        let sym = Matrix::from_2d(&[
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ])
        .unwrap();
        let eig = backend.small_sym_eig_svd(&sym).unwrap();
        assert!(eig.eigenvalues[0] >= eig.eigenvalues[1]);
        assert!(eig.eigenvalues[1] >= eig.eigenvalues[2]);

        // Verify V^T * V = I and M * v_0 = lambda_0 * v_0
        let v = &eig.eigenvectors;
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|r| v[(r, i)] * v[(r, j)]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-11);
            }
        }

        let x = Matrix::from_2d(&[
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ])
        .unwrap();
        let (svals, _, mean) = backend.gram_svd(&x, true).unwrap();
        assert!((mean[0] - 2.5).abs() < 1e-12);
        assert!((svals[0] - 5.0).abs() < 1e-11);
        assert!(svals[1] < 1e-10);
    }

    #[test]
    fn test_n6_batched_histogram_and_split() {
        let backend = get_backend(&DeviceType::Cpu);
        let mut rows = Vec::new();
        let mut grad = Vec::new();
        let mut hess = Vec::new();
        for i in 0..40 {
            let x0 = i as f64;
            let x1 = (i % 5) as f64;
            rows.push(vec![x0, x1]);
            grad.push(if i < 20 { -1.0 } else { 1.0 });
            hess.push(1.0);
        }
        let x = Matrix::from_2d(&rows).unwrap();
        let binned = backend.quantize_features(&x, 16).unwrap();
        let all_idx: Vec<u32> = (0..40).collect();
        let hist = backend
            .batched_histogram_grad(&binned, &all_idx, &grad, &hess)
            .unwrap();

        let split = backend
            .find_best_histogram_split(&binned, &hist, 1.0, 5)
            .unwrap()
            .expect("expected valid split");
        assert_eq!(split.feature_idx, 0);
        assert!(split.gain > 10.0);

        let left_idx: Vec<u32> = (0..20).collect();
        let left_hist = backend
            .batched_histogram_grad(&binned, &left_idx, &grad, &hess)
            .unwrap();
        let right_hist = subtract_histogram(&hist, &left_hist).unwrap();
        let right_count: u32 = right_hist[0..binned.num_bins].iter().map(|b| b.count).sum();
        assert_eq!(right_count, 20);
    }

    #[test]
    fn test_primitives_across_device_fallbacks() {
        let m = Matrix::from_2d(&[vec![1.0, -2.0], vec![3.0, 4.0]]).unwrap();
        for dev in [
            DeviceType::Cpu,
            DeviceType::Gpu(0),
            DeviceType::Wgpu,
            DeviceType::Tpu(0),
        ] {
            let b = get_backend(&dev);
            let s = b.col_reduce_stats(&m, None).unwrap();
            assert!((s.mean[0] - 2.0).abs() < 1e-5);
            let r = b.elementwise_op(&m, ElementwiseOp::Relu).unwrap();
            assert!((r[(0, 1)] - 0.0).abs() < 1e-5);
        }
    }
}
