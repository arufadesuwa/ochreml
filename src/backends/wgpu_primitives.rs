// OchreML Cross-Platform wgpu Primitives (N1, N2, N4, N6) (*≧ω≦*)
// Uses pure-Rust WGSL compute shaders via the shared WgpuContext from wgpu_gpu.rs.

use crate::backends::wgpu_gpu::get_ctx;
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::primitives::stats_ops::ElementwiseOp;
use crate::primitives::{BinnedMatrix, ColStats, GradHessBin};
use bytemuck::{Pod, Zeroable};
use std::sync::{OnceLock, mpsc};
use wgpu::util::DeviceExt;

const MIN_WGPU_ELEMENTS: usize = 65_536;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrimDims {
    rows: u32,
    cols: u32,
    op_code: u32,
    _pad: u32,
    p1: f32,
    p2: f32,
    _pad2: [f32; 2],
}

const ELEMENTWISE_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    op_code: u32,
    _pad: u32,
    p1: f32,
    p2: f32,
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.rows * dims.cols;
    if (idx >= total) {
        return;
    }
    let v = data[idx];
    var out = v;
    if (dims.op_code == 0u) {
        out = 1.0 / (1.0 + exp(-v));
    } else if (dims.op_code == 1u) {
        out = tanh(v);
    } else if (dims.op_code == 2u) {
        out = max(v, 0.0);
    } else if (dims.op_code == 3u) {
        out = exp(v);
    } else if (dims.op_code == 4u) {
        out = log(max(v, dims.p1));
    } else if (dims.op_code == 5u) {
        out = clamp(v, dims.p1, dims.p2);
    } else if (dims.op_code == 6u) {
        let lam = dims.p1;
        out = select(select(0.0, v + lam, v < -lam), v - lam, v > lam);
    } else {
        out = v * dims.p1 + dims.p2;
    }
    data[idx] = out;
}
"#;

const COL_REDUCE_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    op_code: u32,
    _pad: u32,
    p1: f32,
    p2: f32,
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mat_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_stats: array<f32>; // 4 * cols: sum, sq, min, max

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= dims.cols) {
        return;
    }
    let anchor = mat_x[col];
    var sum_s: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    var min_v: f32 = anchor;
    var max_v: f32 = anchor;

    for (var r: u32 = 0u; r < dims.rows; r = r + 1u) {
        let v = mat_x[r * dims.cols + col];
        let d = v - anchor;
        sum_s = sum_s + d;
        sum_sq = sum_sq + d * d;
        min_v = min(min_v, v);
        max_v = max(max_v, v);
    }

    let base = col * 4u;
    out_stats[base] = sum_s;
    out_stats[base + 1u] = sum_sq;
    out_stats[base + 2u] = min_v;
    out_stats[base + 3u] = max_v;
}
"#;

const ROW_ARGMIN_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    op_code: u32,
    _pad: u32,
    p1: f32,
    p2: f32,
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> dist: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_val: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    let base = row * dims.cols;
    var best_val = dist[base];
    var best_idx: u32 = 0u;
    for (var c: u32 = 1u; c < dims.cols; c = c + 1u) {
        let v = dist[base + c];
        if (v < best_val) {
            best_val = v;
            best_idx = c;
        }
    }
    out_idx[row] = best_idx;
    out_val[row] = best_val;
}
"#;

struct WgpuPrimPipelines {
    elementwise: wgpu::ComputePipeline,
    col_reduce: wgpu::ComputePipeline,
    row_argmin: wgpu::ComputePipeline,
}

static WGPU_PRIM_PIPELINES: OnceLock<Option<WgpuPrimPipelines>> = OnceLock::new();

fn get_pipelines() -> Option<(
    &'static crate::backends::wgpu_gpu::WgpuContext,
    &'static WgpuPrimPipelines,
)> {
    let ctx = get_ctx()?;
    let pipes = WGPU_PRIM_PIPELINES
        .get_or_init(|| {
            let make_pipe = |label: &str, wgsl: &str| {
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                    });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            Some(WgpuPrimPipelines {
                elementwise: make_pipe("OchreML Elementwise", ELEMENTWISE_WGSL),
                col_reduce: make_pipe("OchreML ColReduce", COL_REDUCE_WGSL),
                row_argmin: make_pipe("OchreML RowArgmin", ROW_ARGMIN_WGSL),
            })
        })
        .as_ref()?;
    Some((ctx, pipes))
}

fn readback_bytes(
    ctx: &crate::backends::wgpu_gpu::WgpuContext,
    staging: &wgpu::Buffer,
    size: u64,
) -> OchreResult<Vec<u8>> {
    let slice = staging.slice(0..size);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| OchreError::DeviceError("wgpu readback channel closed ( >_< )".to_string()))?
        .map_err(|e| OchreError::DeviceError(format!("wgpu buffer map failed: {} ( >_< )", e)))?;
    let view = slice.get_mapped_range();
    let out = view.to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

/// wgpu implementation of N1 (`col_reduce_stats`) for large unweighted matrices.
pub fn wgpu_col_reduce_stats(
    x: &Matrix,
    sample_weight: Option<&[f64]>,
) -> Option<OchreResult<ColStats>> {
    if sample_weight.is_some() || x.cols < 256 || x.rows * x.cols < 4_000_000 {
        return None;
    }
    let (ctx, pipes) = get_pipelines()?;
    let cols = x.cols;
    let x_f32: Vec<f32> = x.data.iter().map(|&v| v as f32).collect();
    let dims = PrimDims {
        rows: x.rows as u32,
        cols: cols as u32,
        op_code: 0,
        _pad: 0,
        p1: 0.0,
        p2: 0.0,
        _pad2: [0.0; 2],
    };

    let out_bytes = (cols * 4 * std::mem::size_of::<f32>()) as u64;
    let dims_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prim_dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let x_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x_buf"),
            contents: bytemuck::cast_slice(&x_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_stats"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_stats"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = pipes.col_reduce.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("col_reduce_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("col_reduce_enc"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.col_reduce);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((cols as u32) + 63) / 64, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    ctx.queue.submit(Some(enc.finish()));

    let bytes = match readback_bytes(ctx, &staging, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let stats_f32: &[f32] = bytemuck::cast_slice(&bytes);
    let n = x.rows as f64;
    let inv_n = 1.0 / n;
    let mut mean = vec![0.0; cols];
    let mut var = vec![0.0; cols];
    let mut std = vec![0.0; cols];
    let mut min_v = vec![0.0; cols];
    let mut max_v = vec![0.0; cols];
    let mut sum = vec![0.0; cols];

    for j in 0..cols {
        let anchor = x.data[j];
        let s_shift = stats_f32[j * 4] as f64;
        let sq_shift = stats_f32[j * 4 + 1] as f64;
        let sm = s_shift * inv_n;
        mean[j] = anchor + sm;
        sum[j] = s_shift + n * anchor;
        let v = (sq_shift * inv_n - sm * sm).max(0.0);
        var[j] = v;
        std[j] = v.sqrt();
        min_v[j] = stats_f32[j * 4 + 2] as f64;
        max_v[j] = stats_f32[j * 4 + 3] as f64;
    }

    Some(Ok(ColStats {
        mean,
        var,
        std,
        min: min_v,
        max: max_v,
        sum,
        total_weight: n,
    }))
}

/// wgpu implementation of N2 (`elementwise_op`) for large buffers.
pub fn wgpu_elementwise_op(a: &Matrix, op: ElementwiseOp) -> Option<OchreResult<Matrix>> {
    if a.data.len() < MIN_WGPU_ELEMENTS {
        return None;
    }
    let (ctx, pipes) = get_pipelines()?;
    let (op_code, p1, p2): (u32, f32, f32) = match op {
        ElementwiseOp::Sigmoid => (0, 0.0, 0.0),
        ElementwiseOp::Tanh => (1, 0.0, 0.0),
        ElementwiseOp::Relu => (2, 0.0, 0.0),
        ElementwiseOp::Exp => (3, 0.0, 0.0),
        ElementwiseOp::Log { eps } => (4, eps as f32, 0.0),
        ElementwiseOp::Clip { min, max } => (5, min as f32, max as f32),
        ElementwiseOp::SoftThreshold { lambda } => (6, lambda as f32, 0.0),
        ElementwiseOp::Affine { scale, shift } => (7, scale as f32, shift as f32),
    };

    let data_f32: Vec<f32> = a.data.iter().map(|&v| v as f32).collect();
    let byte_len = (data_f32.len() * std::mem::size_of::<f32>()) as u64;
    let dims = PrimDims {
        rows: a.rows as u32,
        cols: a.cols as u32,
        op_code,
        _pad: 0,
        p1,
        p2,
        _pad2: [0.0; 2],
    };

    let dims_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elem_dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let data_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elem_data"),
            contents: bytemuck::cast_slice(&data_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("elem_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = pipes.elementwise.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("elem_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elem_enc"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.elementwise);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((data_f32.len() as u32) + 255) / 256, 1, 1);
    }
    enc.copy_buffer_to_buffer(&data_buf, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(enc.finish()));

    let bytes = match readback_bytes(ctx, &staging, byte_len) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out_f32: &[f32] = bytemuck::cast_slice(&bytes);
    let out_f64: Vec<f64> = out_f32.iter().map(|&v| v as f64).collect();
    Some(Matrix::from_vec(a.rows, a.cols, out_f64))
}

/// wgpu implementation of N4 (`row_argmin`) for large distance matrices.
pub fn wgpu_row_argmin(dist: &Matrix) -> Option<OchreResult<(Vec<usize>, Vec<f64>)>> {
    if dist.rows * dist.cols < MIN_WGPU_ELEMENTS {
        return None;
    }
    let (ctx, pipes) = get_pipelines()?;
    let dist_f32: Vec<f32> = dist.data.iter().map(|&v| v as f32).collect();
    let idx_bytes = (dist.rows * std::mem::size_of::<u32>()) as u64;
    let val_bytes = (dist.rows * std::mem::size_of::<f32>()) as u64;
    let dims = PrimDims {
        rows: dist.rows as u32,
        cols: dist.cols as u32,
        op_code: 0,
        _pad: 0,
        p1: 0.0,
        p2: 0.0,
        _pad2: [0.0; 2],
    };

    let dims_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("argmin_dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let dist_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("argmin_dist"),
            contents: bytemuck::cast_slice(&dist_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let idx_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmin_idx"),
        size: idx_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let val_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmin_val"),
        size: val_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let idx_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmin_idx_stg"),
        size: idx_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let val_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmin_val_stg"),
        size: val_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = pipes.row_argmin.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("argmin_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dist_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: val_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("argmin_enc"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.row_argmin);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((dist.rows as u32) + 63) / 64, 1, 1);
    }
    enc.copy_buffer_to_buffer(&idx_buf, 0, &idx_staging, 0, idx_bytes);
    enc.copy_buffer_to_buffer(&val_buf, 0, &val_staging, 0, val_bytes);
    ctx.queue.submit(Some(enc.finish()));

    let ib = match readback_bytes(ctx, &idx_staging, idx_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let vb = match readback_bytes(ctx, &val_staging, val_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let idx_u32: &[u32] = bytemuck::cast_slice(&ib);
    let val_f32: &[f32] = bytemuck::cast_slice(&vb);
    let indices = idx_u32.iter().map(|&v| v as usize).collect();
    let values = val_f32.iter().map(|&v| v as f64).collect();
    Some(Ok((indices, values)))
}

/// wgpu entrypoint for N6 (`batched_histogram_grad`): delegates to CPU Rayon unless GPU-resident.
pub fn wgpu_batched_histogram_grad(
    _binned: &BinnedMatrix,
    _sample_indices: &[u32],
    _grad: &[f64],
    _hess: &[f64],
) -> Option<OchreResult<Vec<GradHessBin>>> {
    None
}
