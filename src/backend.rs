// OchreML Hardware Compute Backend Module (*≧ω≦*)
// Decouples hardware acceleration targets (CPU Rayon, Native CUDA/cuBLAS, TPU)
// from core mathematical solvers using zero-cost enum dispatch.

use crate::device::DeviceType;
use crate::error::{OchreError, OchreResult};
use crate::matrix::{Matrix, compute_normal_equation_multi_range, compute_normal_equation_range};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend<'a> {
    Cpu,
    CudaSingle(usize),
    CudaMulti(&'a [usize]),
    Wgpu,
    Tpu(usize),
}

#[inline(always)]
pub fn get_backend(device: &DeviceType) -> Backend<'_> {
    match device {
        DeviceType::Gpu(id) => Backend::CudaSingle(*id),
        DeviceType::MultiGpu(gpu_ids) if gpu_ids.len() > 1 => Backend::CudaMulti(gpu_ids),
        DeviceType::MultiGpu(gpu_ids) if gpu_ids.len() == 1 => Backend::CudaSingle(gpu_ids[0]),
        DeviceType::Wgpu => Backend::Wgpu,
        DeviceType::Tpu(id) => Backend::Tpu(*id),
        _ => Backend::Cpu,
    }
}

impl<'a> Backend<'a> {
    // Reusable Primitive 1: General Matrix Multiplication (C = A * B)
    #[inline]
    pub fn matmul(&self, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
        if a.cols != b.rows {
            return Err(OchreError::MatmulDimensionMismatch {
                rows_a: a.rows,
                cols_a: a.cols,
                rows_b: b.rows,
                cols_b: b.cols,
            });
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda::cuda_matmul(*id, a, b) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda::cuda_matmul(ids[0], a, b) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu) {
            if let Some(res) = crate::backends::wgpu_gpu::wgpu_matmul(a, b) {
                return res;
            }
        }

        #[cfg(feature = "tpu-pjrt")]
        if let Backend::Tpu(id) = self {
            if let Some(res) = crate::backends::tpu::tpu_matmul(*id, a, b) {
                return res;
            }
        }

        a.matmul(b)
    }

    // Reusable Primitive 2: Transposed Matrix Multiplication (C = A^T * B)
    #[inline]
    pub fn matmul_at_b(&self, a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
        if a.rows != b.rows {
            return Err(OchreError::LengthMismatch {
                features: a.rows,
                targets: b.rows,
            });
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda::cuda_matmul_at_b(*id, a, b) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda::cuda_matmul_at_b(ids[0], a, b) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu) {
            if let Some(res) = crate::backends::wgpu_gpu::wgpu_matmul_at_b(a, b) {
                return res;
            }
        }

        #[cfg(feature = "tpu-pjrt")]
        if let Backend::Tpu(id) = self {
            if let Some(res) = crate::backends::tpu::tpu_matmul_at_b(*id, a, b) {
                return res;
            }
        }

        cpu_matmul_at_b(a, b)
    }

    // Reusable Primitive 3: Matrix-Vector Multiplication (y = A * v)
    #[inline]
    pub fn matvec(&self, a: &Matrix, v: &[f64]) -> OchreResult<Vec<f64>> {
        if a.cols != v.len() {
            return Err(OchreError::MatvecDimensionMismatch {
                cols: a.cols,
                len: v.len(),
            });
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda::cuda_matvec(*id, a, v) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) if !ids.is_empty() => {
                if let Some(res) = crate::backends::cuda::cuda_matvec(ids[0], a, v) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu) {
            if let Some(res) = crate::backends::wgpu_gpu::wgpu_matvec(a, v) {
                return res;
            }
        }

        #[cfg(feature = "tpu-pjrt")]
        if let Backend::Tpu(id) = self {
            if let Some(res) = crate::backends::tpu::tpu_matvec(*id, a, v) {
                return res;
            }
        }

        a.matvec(v)
    }

    // Fused Normal Equation (1D single-output): zero intermediate allocations (*≧ω≦*)
    #[inline]
    pub fn compute_normal_equation(
        &self,
        x: &Matrix,
        y: &[f64],
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Vec<f64>)> {
        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) =
                    crate::backends::cuda::cuda_compute_normal_equation(&[*id], x, y, fit_intercept)
                {
                    return res;
                }
            }
            Backend::CudaMulti(ids) => {
                if let Some(res) =
                    crate::backends::cuda::cuda_compute_normal_equation(ids, x, y, fit_intercept)
                {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu) {
            if let Some(res) =
                crate::backends::wgpu_gpu::wgpu_compute_normal_equation(x, y, fit_intercept)
            {
                return res;
            }
        }

        let p = if fit_intercept { x.cols + 1 } else { x.cols };
        match self {
            Backend::Cpu | Backend::CudaSingle(_) | Backend::Wgpu => {
                cpu_normal_equation_1d(x, y, fit_intercept, p)
            }
            Backend::CudaMulti(gpu_ids) => {
                let n_devices = gpu_ids.len().max(1);
                let chunk_size = (x.rows + n_devices - 1) / n_devices;
                let mut total_xt_x = Matrix::zeros(p, p);
                let mut total_xt_y = vec![0.0; p];

                for d in 0..n_devices {
                    let start_row = d * chunk_size;
                    if start_row >= x.rows {
                        break;
                    }
                    let end_row = (start_row + chunk_size).min(x.rows);
                    let (chunk_xt_x, chunk_xt_y) =
                        compute_normal_equation_range(x, y, fit_intercept, start_row, end_row)?;

                    for i in 0..(p * p) {
                        total_xt_x.data[i] += chunk_xt_x.data[i];
                    }
                    for i in 0..p {
                        total_xt_y[i] += chunk_xt_y[i];
                    }
                }
                Ok((total_xt_x, total_xt_y))
            }
            Backend::Tpu(_id) => {
                #[cfg(feature = "tpu-pjrt")]
                if let Some(res) =
                    crate::backends::tpu::tpu_compute_normal_equation(*_id, x, y, fit_intercept)
                {
                    return res;
                }

                let block_size = 64;
                let mut total_xt_x = Matrix::zeros(p, p);
                let mut total_xt_y = vec![0.0; p];

                let mut start_row = 0;
                while start_row < x.rows {
                    let end_row = (start_row + block_size).min(x.rows);
                    let (tile_xt_x, tile_xt_y) =
                        compute_normal_equation_range(x, y, fit_intercept, start_row, end_row)?;
                    for i in 0..(p * p) {
                        total_xt_x.data[i] += tile_xt_x.data[i];
                    }
                    for i in 0..p {
                        total_xt_y[i] += tile_xt_y[i];
                    }
                    start_row = end_row;
                }
                Ok((total_xt_x, total_xt_y))
            }
        }
    }

    // Fused Multi-target & weighted Normal Equation
    #[inline]
    pub fn compute_normal_equation_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Matrix)> {
        if y.cols == 1 && sample_weight.is_none() {
            let (xt_x, xt_y_vec) = self.compute_normal_equation(x, &y.data, fit_intercept)?;
            let p = xt_y_vec.len();
            return Ok((xt_x, Matrix::from_vec(p, 1, xt_y_vec)?));
        }

        #[cfg(feature = "cuda-dynamic")]
        match self {
            Backend::CudaSingle(id) => {
                if let Some(res) = crate::backends::cuda::cuda_compute_normal_equation_multi(
                    &[*id],
                    x,
                    y,
                    sample_weight,
                    fit_intercept,
                ) {
                    return res;
                }
            }
            Backend::CudaMulti(ids) => {
                if let Some(res) = crate::backends::cuda::cuda_compute_normal_equation_multi(
                    ids,
                    x,
                    y,
                    sample_weight,
                    fit_intercept,
                ) {
                    return res;
                }
            }
            _ => {}
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(self, Backend::Wgpu) {
            if let Some(res) = crate::backends::wgpu_gpu::wgpu_compute_normal_equation_multi(
                x,
                y,
                sample_weight,
                fit_intercept,
            ) {
                return res;
            }
        }

        let p = if fit_intercept { x.cols + 1 } else { x.cols };
        let k_targets = y.cols;

        match self {
            Backend::Cpu | Backend::CudaSingle(_) | Backend::Wgpu => {
                cpu_normal_equation_multi(x, y, sample_weight, fit_intercept, p, k_targets)
            }
            Backend::CudaMulti(gpu_ids) => {
                let n_devices = gpu_ids.len().max(1);
                let chunk_size = (x.rows + n_devices - 1) / n_devices;
                let mut total_xt_x = Matrix::zeros(p, p);
                let mut total_xt_y = Matrix::zeros(p, k_targets);

                for d in 0..n_devices {
                    let start_row = d * chunk_size;
                    if start_row >= x.rows {
                        break;
                    }
                    let end_row = (start_row + chunk_size).min(x.rows);
                    let (chunk_xt_x, chunk_xt_y) = compute_normal_equation_multi_range(
                        x,
                        y,
                        sample_weight,
                        fit_intercept,
                        start_row,
                        end_row,
                    )?;

                    for i in 0..(p * p) {
                        total_xt_x.data[i] += chunk_xt_x.data[i];
                    }
                    for i in 0..(p * k_targets) {
                        total_xt_y.data[i] += chunk_xt_y.data[i];
                    }
                }
                Ok((total_xt_x, total_xt_y))
            }
            Backend::Tpu(_id) => {
                #[cfg(feature = "tpu-pjrt")]
                if let Some(res) = crate::backends::tpu::tpu_compute_normal_equation_multi(
                    *_id,
                    x,
                    y,
                    sample_weight,
                    fit_intercept,
                ) {
                    return res;
                }

                let block_size = 64;
                let mut total_xt_x = Matrix::zeros(p, p);
                let mut total_xt_y = Matrix::zeros(p, k_targets);

                let mut start_row = 0;
                while start_row < x.rows {
                    let end_row = (start_row + block_size).min(x.rows);
                    let (tile_xt_x, tile_xt_y) = compute_normal_equation_multi_range(
                        x,
                        y,
                        sample_weight,
                        fit_intercept,
                        start_row,
                        end_row,
                    )?;
                    for i in 0..(p * p) {
                        total_xt_x.data[i] += tile_xt_x.data[i];
                    }
                    for i in 0..(p * k_targets) {
                        total_xt_y.data[i] += tile_xt_y.data[i];
                    }
                    start_row = end_row;
                }
                Ok((total_xt_x, total_xt_y))
            }
        }
    }
}

fn aligned_row_chunks_range(start_row: usize, end_row: usize) -> Vec<(usize, usize)> {
    let n_rows = end_row.saturating_sub(start_row);
    let num_tasks = (rayon::current_num_threads().max(1) * 8).max(8);
    let raw_chunk = n_rows.div_ceil(num_tasks);
    let chunk_size = (raw_chunk.div_ceil(64) * 64).max(64);
    let mut chunks = Vec::with_capacity(num_tasks);
    let mut start = start_row;
    while start < end_row {
        let end = (start + chunk_size).min(end_row);
        chunks.push((start, end));
        start = end;
    }
    chunks
}

pub(crate) fn cpu_normal_equation_1d_slice(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    if end_row.saturating_sub(start_row) >= 1000 {
        use rayon::prelude::*;
        let chunks = aligned_row_chunks_range(start_row, end_row);

        chunks
            .into_par_iter()
            .with_max_len(1)
            .map(|(start, end)| compute_normal_equation_range(x, y, fit_intercept, start, end))
            .reduce(
                || Ok((Matrix::zeros(p, p), vec![0.0; p])),
                |acc, chunk| {
                    let (mut acc_m, mut acc_y) = acc?;
                    let (chunk_m, chunk_y) = chunk?;
                    for i in 0..(p * p) {
                        acc_m.data[i] += chunk_m.data[i];
                    }
                    for i in 0..p {
                        acc_y[i] += chunk_y[i];
                    }
                    Ok((acc_m, acc_y))
                },
            )
    } else {
        compute_normal_equation_range(x, y, fit_intercept, start_row, end_row)
    }
}

fn cpu_normal_equation_1d(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    _p: usize,
) -> OchreResult<(Matrix, Vec<f64>)> {
    cpu_normal_equation_1d_slice(x, y, fit_intercept, 0, x.rows)
}

pub(crate) fn cpu_normal_equation_multi_slice(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Matrix)> {
    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let k_targets = y.cols;
    if end_row.saturating_sub(start_row) >= 1000 {
        use rayon::prelude::*;
        let chunks = aligned_row_chunks_range(start_row, end_row);

        chunks
            .into_par_iter()
            .with_max_len(1)
            .map(|(start, end)| {
                compute_normal_equation_multi_range(x, y, sample_weight, fit_intercept, start, end)
            })
            .reduce(
                || Ok((Matrix::zeros(p, p), Matrix::zeros(p, k_targets))),
                |acc, chunk| {
                    let (mut acc_m, mut acc_y) = acc?;
                    let (chunk_m, chunk_y) = chunk?;
                    for i in 0..(p * p) {
                        acc_m.data[i] += chunk_m.data[i];
                    }
                    for i in 0..(p * k_targets) {
                        acc_y.data[i] += chunk_y.data[i];
                    }
                    Ok((acc_m, acc_y))
                },
            )
    } else {
        compute_normal_equation_multi_range(x, y, sample_weight, fit_intercept, start_row, end_row)
    }
}

fn cpu_normal_equation_multi(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    _p: usize,
    _k_targets: usize,
) -> OchreResult<(Matrix, Matrix)> {
    cpu_normal_equation_multi_slice(x, y, sample_weight, fit_intercept, 0, x.rows)
}

// Cache-friendly A^T * B on CPU without allocating explicit A^T
fn cpu_matmul_at_b(a: &Matrix, b: &Matrix) -> OchreResult<Matrix> {
    let p = a.cols;
    let k = b.cols;
    let mut out = Matrix::zeros(p, k);
    for r in 0..a.rows {
        let a_row = &a.data[(r * p)..(r * p + p)];
        let b_row = &b.data[(r * k)..(r * k + k)];
        for i in 0..p {
            let ai = a_row[i];
            let out_row = &mut out.data[(i * k)..(i * k + k)];
            for j in 0..k {
                out_row[j] += ai * b_row[j];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_dispatch_single_and_multi_gpu() {
        assert_eq!(get_backend(&DeviceType::Cpu), Backend::Cpu);
        assert_eq!(get_backend(&DeviceType::Gpu(0)), Backend::CudaSingle(0));
        let ids = vec![0, 1];
        assert_eq!(
            get_backend(&DeviceType::MultiGpu(ids.clone())),
            Backend::CudaMulti(&ids)
        );
        assert_eq!(get_backend(&DeviceType::Wgpu), Backend::Wgpu);
        assert_eq!(get_backend(&DeviceType::Tpu(0)), Backend::Tpu(0));
    }

    #[test]
    fn test_backend_matrix_primitives() {
        let backend = Backend::Cpu;
        let a = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let b = Matrix::from_2d(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]).unwrap();

        // A^T * B: (2x3) * (3x2) = (2x2)
        // [1*7 + 3*9 + 5*11 = 89,  1*8 + 3*10 + 5*12 = 98]
        // [2*7 + 4*9 + 6*11 = 116, 2*8 + 4*10 + 6*12 = 128]
        let atb = backend.matmul_at_b(&a, &b).unwrap();
        assert_eq!(atb.rows, 2);
        assert_eq!(atb.cols, 2);
        assert_eq!(atb.data, vec![89.0, 98.0, 116.0, 128.0]);

        let v = vec![2.0, 3.0];
        let mv = backend.matvec(&a, &v).unwrap();
        assert_eq!(mv, vec![8.0, 18.0, 28.0]);

        let c = Matrix::from_2d(&[vec![1.0, 0.0], vec![0.0, 2.0]]).unwrap();
        let mm = backend.matmul(&a, &c).unwrap();
        assert_eq!(mm.data, vec![1.0, 4.0, 3.0, 8.0, 5.0, 12.0]);
    }
}
