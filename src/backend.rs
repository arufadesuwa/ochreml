// OchreML Hardware Compute Backend Module (*≧ω≦*)
// Decouples hardware acceleration targets (CPU Rayon, Multi-GPU Data-Parallel, TPU Systolic)
// from core mathematical solvers using trait polymorphism.

use crate::device::DeviceType;
use crate::error::OchreResult;
use crate::matrix::{compute_normal_equation_multi_range, Matrix};

pub trait ComputeBackend: Send + Sync {
    fn compute_normal_equation_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Matrix)>;
}

pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn compute_normal_equation_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Matrix)> {
        let p = if fit_intercept { x.cols + 1 } else { x.cols };
        let k_targets = y.cols;

        if x.rows >= 1000 {
            use rayon::prelude::*;
            let num_threads = rayon::current_num_threads().max(1);
            let chunk_size = (x.rows + num_threads - 1) / num_threads;
            let chunks: Vec<(usize, usize)> = (0..num_threads)
                .map(|i| {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(x.rows);
                    (start, end)
                })
                .filter(|(start, end)| start < end)
                .collect();

            let (total_xt_x, total_xt_y) = chunks
                .into_par_iter()
                .map(|(start, end)| {
                    compute_normal_equation_multi_range(
                        x,
                        y,
                        sample_weight,
                        fit_intercept,
                        start,
                        end,
                    )
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
                )?;

            Ok((total_xt_x, total_xt_y))
        } else {
            compute_normal_equation_multi_range(x, y, sample_weight, fit_intercept, 0, x.rows)
        }
    }
}

pub struct MultiGpuBackend {
    pub gpu_ids: Vec<usize>,
}

impl ComputeBackend for MultiGpuBackend {
    fn compute_normal_equation_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Matrix)> {
        let p = if fit_intercept { x.cols + 1 } else { x.cols };
        let k_targets = y.cols;
        let n_devices = self.gpu_ids.len().max(1);
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
}

pub struct TpuBackend {
    pub device_id: usize,
}

impl ComputeBackend for TpuBackend {
    fn compute_normal_equation_multi(
        &self,
        x: &Matrix,
        y: &Matrix,
        sample_weight: Option<&[f64]>,
        fit_intercept: bool,
    ) -> OchreResult<(Matrix, Matrix)> {
        let p = if fit_intercept { x.cols + 1 } else { x.cols };
        let k_targets = y.cols;
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

pub fn get_backend(device: &DeviceType) -> Box<dyn ComputeBackend> {
    match device {
        DeviceType::MultiGpu(gpu_ids) if gpu_ids.len() > 1 => {
            Box::new(MultiGpuBackend { gpu_ids: gpu_ids.clone() })
        }
        DeviceType::Tpu(id) => Box::new(TpuBackend { device_id: *id }),
        _ => Box::new(CpuBackend),
    }
}

