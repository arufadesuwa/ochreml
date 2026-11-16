// OchreML Cross-Platform Native GPU Backend via wgpu (Vulkan / Metal / DX12) (*≧ω≦*)
// Explicitly rejects software rasterizers (wgpu::DeviceType::Cpu) and applies
// column-scaling in f64->f32->f64 to protect Normal Equation numerical stability.

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use bytemuck::{Pod, Zeroable};
use std::sync::{OnceLock, mpsc};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatDims {
    m: u32,
    n: u32,
    k: u32,
    split_k: u32,
}

const MATMUL_WGSL: &str = r#"
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let lx = lid.x;
    let ly = lid.y;

    var acc: f32 = 0.0;
    let num_tiles = (dims.k + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + lx;
        let b_row = t * 16u + ly;

        if (row < dims.m && a_col < dims.k) {
            tile_a[ly * 16u + lx] = mat_a[row * dims.k + a_col];
        } else {
            tile_a[ly * 16u + lx] = 0.0;
        }

        if (b_row < dims.k && col < dims.n) {
            tile_b[ly * 16u + lx] = mat_b[b_row * dims.n + col];
        } else {
            tile_b[ly * 16u + lx] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        mat_c[row * dims.n + col] = acc;
    }
}
"#;

// Computes C(z, m x n) = A(k x m)^T * B(k x n) over Split-K slices with Kahan summation
const MATMUL_AT_B_WGSL: &str = r#"
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    split_k: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y; // 0..m (column index of A)
    let col = gid.x; // 0..n (column index of B)
    let z = gid.z;   // Split-K slice index
    let lx = lid.x;
    let ly = lid.y;

    let sk = max(dims.split_k, 1u);
    let chunk = (dims.k + sk - 1u) / sk;
    let k_start = z * chunk;
    let k_end = min(k_start + chunk, dims.k);
    let k_len = select(0u, k_end - k_start, k_end > k_start);

    var acc: f32 = 0.0;
    var comp: f32 = 0.0;
    let num_tiles = (k_len + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_row = k_start + t * 16u + lx;
        let b_row = k_start + t * 16u + ly;

        if (row < dims.m && a_row < k_end) {
            tile_a[ly * 16u + lx] = mat_a[a_row * dims.m + row];
        } else {
            tile_a[ly * 16u + lx] = 0.0;
        }

        if (b_row < k_end && col < dims.n) {
            tile_b[ly * 16u + lx] = mat_b[b_row * dims.n + col];
        } else {
            tile_b[ly * 16u + lx] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let prod = tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
            let y_k = prod - comp;
            let t_k = acc + y_k;
            comp = (t_k - acc) - y_k;
            acc = t_k;
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        mat_c[z * (dims.m * dims.n) + row * dims.n + col] = acc;
    }
}
"#;

const MATVEC_WGSL: &str = r#"
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> vec_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> vec_y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.m) {
        return;
    }
    var acc: f32 = 0.0;
    let offset = row * dims.n;
    for (var j: u32 = 0u; j < dims.n; j = j + 1u) {
        acc = acc + mat_a[offset + j] * vec_x[j];
    }
    vec_y[row] = acc;
}
"#;

pub struct WgpuContext {
    adapter_name: String,
    backend_name: String,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_at_b_pipeline: wgpu::ComputePipeline,
    matvec_pipeline: wgpu::ComputePipeline,
}

static WGPU_CTX: OnceLock<Option<WgpuContext>> = OnceLock::new();

fn init_wgpu_context() -> Option<WgpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    // Explicitly reject software rasterizers (wgpu::DeviceType::Cpu like llvmpipe)
    let adapter = adapters
        .iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .or_else(|| {
            adapters
                .iter()
                .find(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
        })
        .or_else(|| {
            adapters
                .iter()
                .find(|a| a.get_info().device_type == wgpu::DeviceType::VirtualGpu)
        })?;

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("OchreML WGPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("OchreML Matmul Shader"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_WGSL.into()),
    });
    let matmul_at_b_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("OchreML Matmul AtB Shader"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_AT_B_WGSL.into()),
    });
    let matvec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("OchreML Matvec Shader"),
        source: wgpu::ShaderSource::Wgsl(MATVEC_WGSL.into()),
    });

    let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("OchreML Matmul Pipeline"),
        layout: None,
        module: &matmul_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let matmul_at_b_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("OchreML Matmul AtB Pipeline"),
        layout: None,
        module: &matmul_at_b_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("OchreML Matvec Pipeline"),
        layout: None,
        module: &matvec_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Some(WgpuContext {
        adapter_name: info.name,
        backend_name: format!("{:?}", info.backend),
        device,
        queue,
        matmul_pipeline,
        matmul_at_b_pipeline,
        matvec_pipeline,
    })
}

pub(crate) fn get_ctx() -> Option<&'static WgpuContext> {
    WGPU_CTX.get_or_init(init_wgpu_context).as_ref()
}

pub fn is_available() -> bool {
    get_ctx().is_some()
}

pub fn adapter_info_string() -> Option<String> {
    get_ctx().map(|c| format!("{} ({})", c.adapter_name, c.backend_name))
}

fn run_binary_shader(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    dims: MatDims,
    a_f32: &[f32],
    b_f32: &[f32],
    out_len: usize,
    wg_x: u32,
    wg_y: u32,
) -> OchreResult<Vec<f32>> {
    let out_bytes = (out_len.max(1) * std::mem::size_of::<f32>()) as u64;

    let dims_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let a_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buf_a"),
            contents: bytemuck::cast_slice(a_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let b_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buf_b"),
            contents: bytemuck::cast_slice(b_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buf_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buf_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ochreml_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ochreml_enc"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ochreml_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| {
            OchreError::DeviceError("wgpu map_async channel closed (´；ω；`)".to_string())
        })?
        .map_err(|e| {
            OchreError::DeviceError(format!("wgpu buffer map failed: {:?} (´；ω；`)", e))
        })?;

    let view = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&view)[..out_len].to_vec();
    drop(view);
    staging_buf.unmap();
    Ok(result)
}

pub fn wgpu_matmul(a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let ctx = get_ctx()?;
    let a_f32: Vec<f32> = a.data.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b.data.iter().map(|&v| v as f32).collect();
    let dims = MatDims {
        m: a.rows as u32,
        n: b.cols as u32,
        k: a.cols as u32,
        split_k: 1,
    };
    let wg_x = (b.cols as u32 + 15) / 16;
    let wg_y = (a.rows as u32 + 15) / 16;

    let res = run_binary_shader(
        ctx,
        &ctx.matmul_pipeline,
        dims,
        &a_f32,
        &b_f32,
        a.rows * b.cols,
        wg_x,
        wg_y,
    )
    .map(|out_f32| Matrix {
        rows: a.rows,
        cols: b.cols,
        data: out_f32.into_iter().map(|v| v as f64).collect(),
    });
    Some(res)
}

pub fn wgpu_matmul_at_b(a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let ctx = get_ctx()?;
    let a_f32: Vec<f32> = a.data.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b.data.iter().map(|&v| v as f32).collect();
    let dims = MatDims {
        m: a.cols as u32,
        n: b.cols as u32,
        k: a.rows as u32,
        split_k: 1,
    };
    let wg_x = (b.cols as u32 + 15) / 16;
    let wg_y = (a.cols as u32 + 15) / 16;

    let res = run_binary_shader(
        ctx,
        &ctx.matmul_at_b_pipeline,
        dims,
        &a_f32,
        &b_f32,
        a.cols * b.cols,
        wg_x,
        wg_y,
    )
    .map(|out_f32| Matrix {
        rows: a.cols,
        cols: b.cols,
        data: out_f32.into_iter().map(|v| v as f64).collect(),
    });
    Some(res)
}

pub fn wgpu_matvec(a: &Matrix, v: &[f64]) -> Option<OchreResult<Vec<f64>>> {
    let ctx = get_ctx()?;
    let a_f32: Vec<f32> = a.data.iter().map(|&val| val as f32).collect();
    let v_f32: Vec<f32> = v.iter().map(|&val| val as f32).collect();
    let dims = MatDims {
        m: a.rows as u32,
        n: a.cols as u32,
        k: 1,
        split_k: 1,
    };
    let wg_x = (a.rows as u32 + 63) / 64;

    let res = run_binary_shader(
        ctx,
        &ctx.matvec_pipeline,
        dims,
        &a_f32,
        &v_f32,
        a.rows,
        wg_x,
        1,
    )
    .map(|out_f32| out_f32.into_iter().map(|val| val as f64).collect());
    Some(res)
}

// Column-scaled f32 Normal Equation on wgpu with Rayon packing, single-upload buffers,
// Split-K parallel GPU reduction, and f64 host accumulation.
pub fn wgpu_compute_normal_equation_multi(
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Matrix)>> {
    let ctx = get_ctx()?;
    let res = (|| {
        let n = x.rows;
        let p_x = x.cols;
        let k_targets = y.cols;
        let p_total = if fit_intercept { p_x + 1 } else { p_x };

        // 1. Compute column max-abs scales and intercept sums in f64 (parallelized via Rayon)
        let (mut x_scales, mut y_scales, sum_x, sum_y, weight_sum) = if n >= 4096 {
            use rayon::prelude::*;
            let num_threads = rayon::current_num_threads().max(1);
            let chunk_size = (n + num_threads - 1) / num_threads;
            (0..num_threads)
                .into_par_iter()
                .map(|tid| {
                    let start = tid * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mut xs = vec![0.0_f64; p_x];
                    let mut ys = vec![0.0_f64; k_targets];
                    let mut sx = vec![0.0_f64; p_x];
                    let mut sy = vec![0.0_f64; k_targets];
                    let mut ws = 0.0_f64;
                    for r in start..end {
                        let w = sample_weight.map_or(1.0, |sw| sw[r]);
                        ws += w;
                        let x_row = &x.data[(r * p_x)..(r * p_x + p_x)];
                        let y_row = &y.data[(r * k_targets)..(r * k_targets + k_targets)];
                        for j in 0..p_x {
                            let abs_v = x_row[j].abs();
                            if abs_v > xs[j] {
                                xs[j] = abs_v;
                            }
                            if fit_intercept {
                                sx[j] += w * x_row[j];
                            }
                        }
                        for t in 0..k_targets {
                            let abs_v = y_row[t].abs();
                            if abs_v > ys[t] {
                                ys[t] = abs_v;
                            }
                            if fit_intercept {
                                sy[t] += w * y_row[t];
                            }
                        }
                    }
                    (xs, ys, sx, sy, ws)
                })
                .reduce(
                    || {
                        (
                            vec![0.0; p_x],
                            vec![0.0; k_targets],
                            vec![0.0; p_x],
                            vec![0.0; k_targets],
                            0.0,
                        )
                    },
                    |mut a, b| {
                        for j in 0..p_x {
                            if b.0[j] > a.0[j] {
                                a.0[j] = b.0[j];
                            }
                            a.2[j] += b.2[j];
                        }
                        for t in 0..k_targets {
                            if b.1[t] > a.1[t] {
                                a.1[t] = b.1[t];
                            }
                            a.3[t] += b.3[t];
                        }
                        a.4 += b.4;
                        a
                    },
                )
        } else {
            let mut xs = vec![0.0_f64; p_x];
            let mut ys = vec![0.0_f64; k_targets];
            let mut sx = vec![0.0_f64; p_x];
            let mut sy = vec![0.0_f64; k_targets];
            let mut ws = 0.0_f64;
            for r in 0..n {
                let w = sample_weight.map_or(1.0, |sw| sw[r]);
                ws += w;
                let x_row = &x.data[(r * p_x)..(r * p_x + p_x)];
                let y_row = &y.data[(r * k_targets)..(r * k_targets + k_targets)];
                for j in 0..p_x {
                    let abs_v = x_row[j].abs();
                    if abs_v > xs[j] {
                        xs[j] = abs_v;
                    }
                    if fit_intercept {
                        sx[j] += w * x_row[j];
                    }
                }
                for t in 0..k_targets {
                    let abs_v = y_row[t].abs();
                    if abs_v > ys[t] {
                        ys[t] = abs_v;
                    }
                    if fit_intercept {
                        sy[t] += w * y_row[t];
                    }
                }
            }
            (xs, ys, sx, sy, ws)
        };

        for s in &mut x_scales {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        for s in &mut y_scales {
            if *s == 0.0 {
                *s = 1.0;
            }
        }

        // 2. Pack scaled f32 buffers (parallelized via Rayon)
        let mut x_f32 = vec![0.0_f32; n * p_x];
        let mut y_f32 = vec![0.0_f32; n * k_targets];
        if n >= 4096 {
            use rayon::prelude::*;
            x_f32
                .par_chunks_mut(p_x.max(1))
                .enumerate()
                .for_each(|(r, x_dst)| {
                    let sqrt_w = sample_weight.map_or(1.0_f64, |sw| sw[r].sqrt());
                    let x_src = &x.data[(r * p_x)..(r * p_x + p_x)];
                    for j in 0..p_x {
                        x_dst[j] = ((x_src[j] / x_scales[j]) * sqrt_w) as f32;
                    }
                });
            y_f32
                .par_chunks_mut(k_targets.max(1))
                .enumerate()
                .for_each(|(r, y_dst)| {
                    let sqrt_w = sample_weight.map_or(1.0_f64, |sw| sw[r].sqrt());
                    let y_src = &y.data[(r * k_targets)..(r * k_targets + k_targets)];
                    for t in 0..k_targets {
                        y_dst[t] = ((y_src[t] / y_scales[t]) * sqrt_w) as f32;
                    }
                });
        } else {
            for r in 0..n {
                let sqrt_w = sample_weight.map_or(1.0_f64, |sw| sw[r].sqrt());
                let x_off = r * p_x;
                let y_off = r * k_targets;
                for j in 0..p_x {
                    x_f32[x_off + j] = ((x.data[x_off + j] / x_scales[j]) * sqrt_w) as f32;
                }
                for t in 0..k_targets {
                    y_f32[y_off + t] = ((y.data[y_off + t] / y_scales[t]) * sqrt_w) as f32;
                }
            }
        }

        // 3. Upload X and Y ONCE, then dispatch Split-K X^T X and X^T Y in a single CommandEncoder
        let split_k = ((n + 4095) / 4096).clamp(1, 64) as u32;
        let sk_usize = split_k as usize;
        let xtx_elems = sk_usize * p_x * p_x;
        let xty_elems = sk_usize * p_x * k_targets;
        let xtx_bytes = (xtx_elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let xty_bytes = (xty_elems.max(1) * std::mem::size_of::<f32>()) as u64;

        let x_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x_f32"),
                contents: bytemuck::cast_slice(&x_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let y_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("y_f32"),
                contents: bytemuck::cast_slice(&y_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let dims_xtx = MatDims {
            m: p_x as u32,
            n: p_x as u32,
            k: n as u32,
            split_k,
        };
        let dims_xty = MatDims {
            m: p_x as u32,
            n: k_targets as u32,
            k: n as u32,
            split_k,
        };

        let dims_xtx_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims_xtx"),
                contents: bytemuck::bytes_of(&dims_xtx),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dims_xty_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims_xty"),
                contents: bytemuck::bytes_of(&dims_xty),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let out_xtx_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_xtx"),
            size: xtx_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_xty_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_xty"),
            size: xty_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stg_xtx_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg_xtx"),
            size: xtx_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_xty_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg_xty"),
            size: xty_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = ctx.matmul_at_b_pipeline.get_bind_group_layout(0);
        let bg_xtx = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_xtx"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_xtx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_xtx_buf.as_entire_binding(),
                },
            ],
        });
        let bg_xty = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_xty"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_xty_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_xty_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("norm_eq_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("norm_eq_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.matmul_at_b_pipeline);
            pass.set_bind_group(0, &bg_xtx, &[]);
            pass.dispatch_workgroups((p_x as u32 + 15) / 16, (p_x as u32 + 15) / 16, split_k);
            pass.set_bind_group(0, &bg_xty, &[]);
            pass.dispatch_workgroups(
                (k_targets as u32 + 15) / 16,
                (p_x as u32 + 15) / 16,
                split_k,
            );
        }
        encoder.copy_buffer_to_buffer(&out_xtx_buf, 0, &stg_xtx_buf, 0, xtx_bytes);
        encoder.copy_buffer_to_buffer(&out_xty_buf, 0, &stg_xty_buf, 0, xty_bytes);
        ctx.queue.submit(Some(encoder.finish()));

        let s_xtx = stg_xtx_buf.slice(..);
        let s_xty = stg_xty_buf.slice(..);
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        s_xtx.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });
        s_xty.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx1.recv()
            .map_err(|_| {
                OchreError::DeviceError("wgpu xtx map channel closed (´；ω；`)".to_string())
            })?
            .map_err(|e| {
                OchreError::DeviceError(format!("wgpu xtx map failed: {:?} (´；ω；`)", e))
            })?;
        rx2.recv()
            .map_err(|_| {
                OchreError::DeviceError("wgpu xty map channel closed (´；ω；`)".to_string())
            })?
            .map_err(|e| {
                OchreError::DeviceError(format!("wgpu xty map failed: {:?} (´；ω；`)", e))
            })?;

        let v_xtx = s_xtx.get_mapped_range();
        let v_xty = s_xty.get_mapped_range();
        let raw_xtx: &[f32] = &bytemuck::cast_slice(&v_xtx)[..xtx_elems];
        let raw_xty: &[f32] = &bytemuck::cast_slice(&v_xty)[..xty_elems];

        // 4. Reduce Split-K slices in f64 & unscale back to f64
        let mut full_xtx = Matrix::zeros(p_total, p_total);
        let mut full_xty = Matrix::zeros(p_total, k_targets);
        let xtx_stride = p_x * p_x;
        let xty_stride = p_x * k_targets;

        for z in 0..sk_usize {
            let z_xtx = &raw_xtx[(z * xtx_stride)..((z + 1) * xtx_stride)];
            let z_xty = &raw_xty[(z * xty_stride)..((z + 1) * xty_stride)];
            for j in 0..p_x {
                for k in 0..p_x {
                    full_xtx.data[j * p_total + k] += z_xtx[j * p_x + k] as f64;
                }
                for t in 0..k_targets {
                    full_xty.data[j * k_targets + t] += z_xty[j * k_targets + t] as f64;
                }
            }
        }
        drop(v_xtx);
        drop(v_xty);
        stg_xtx_buf.unmap();
        stg_xty_buf.unmap();

        for j in 0..p_x {
            for k in 0..p_x {
                full_xtx.data[j * p_total + k] *= x_scales[j] * x_scales[k];
            }
            for t in 0..k_targets {
                full_xty.data[j * k_targets + t] *= x_scales[j] * y_scales[t];
            }
        }

        if fit_intercept {
            for j in 0..p_x {
                full_xtx.data[j * p_total + p_x] = sum_x[j];
                full_xtx.data[p_x * p_total + j] = sum_x[j];
            }
            full_xtx.data[p_x * p_total + p_x] = weight_sum;
            for t in 0..k_targets {
                full_xty.data[p_x * k_targets + t] = sum_y[t];
            }
        }

        Ok((full_xtx, full_xty))
    })();
    Some(res)
}

pub fn wgpu_compute_normal_equation(
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Vec<f64>)>> {
    let y_mat = Matrix {
        rows: y.len(),
        cols: 1,
        data: y.to_vec(),
    };
    let res = wgpu_compute_normal_equation_multi(x, &y_mat, None, fit_intercept)?;
    Some(res.map(|(xtx, xty_mat)| (xtx, xty_mat.data)))
}
