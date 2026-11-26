// OchreML Cross-Platform wgpu Neural Network Backend (N7..N10) (*≧ω≦*)
// Pure-Rust WGSL compute shaders via crate::backends::wgpu_gpu::get_ctx() for batched SGEMM,
// norms/activations/causal_mask, cross_entropy/adamw, and im2col/pool2d with graceful CPU fallback.

use crate::backends::wgpu_gpu::{WgpuContext, get_ctx};
use crate::error::{OchreError, OchreResult};
use bytemuck::{Pod, Zeroable};
use std::sync::{OnceLock, mpsc};
use wgpu::util::DeviceExt;

const MIN_WGPU_ELEMS: usize = 65_536;
const MIN_WGPU_GEMM_OPS: usize = 262_144;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GemmParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnaryParams {
    n: u32,
    op_code: u32, // 0=gelu, 1=gelu_bwd, 2=silu, 3=silu_bwd
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormParams {
    rows: u32,
    cols: u32,
    has_gamma: u32,
    has_beta: u32,
    eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CausalMaskParams {
    total: u32,
    q_len: u32,
    k_len: u32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrossEntropyParams {
    rows: u32,
    cols: u32,
    has_ignore: u32,
    ignore_index: u32,
    label_smoothing: f32,
    inv_valid: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamwParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    inv_bc1: f32,
    inv_bc2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConvPoolParams {
    total: u32,
    c: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    oh: u32,
    ow: u32,
    mode: u32, // 0=im2col, 1=col2im, 2=max_pool2d, 3=avg_pool2d
    _pad: u32,
}

const BATCHED_GEMM_WGSL: &str = r#"
struct Params {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let batch_idx = gid.z;
    if (batch_idx >= p.batch) {
        return;
    }

    let lx = lid.x;
    let ly = lid.y;
    let a_base = batch_idx * p.m * p.k;
    let b_base = batch_idx * p.k * p.n;
    let c_base = batch_idx * p.m * p.n;

    var acc: f32 = 0.0;
    let tiles = (p.k + 15u) / 16u;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let a_k = t * 16u + lx;
        let b_k = t * 16u + ly;

        if (row < p.m && a_k < p.k) {
            if (p.trans_a != 0u) {
                tile_a[ly * 16u + lx] = a[a_base + a_k * p.m + row];
            } else {
                tile_a[ly * 16u + lx] = a[a_base + row * p.k + a_k];
            }
        } else {
            tile_a[ly * 16u + lx] = 0.0;
        }

        if (b_k < p.k && col < p.n) {
            if (p.trans_b != 0u) {
                tile_b[ly * 16u + lx] = b[b_base + col * p.k + b_k];
            } else {
                tile_b[ly * 16u + lx] = b[b_base + b_k * p.n + col];
            }
        } else {
            tile_b[ly * 16u + lx] = 0.0;
        }

        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }
        workgroupBarrier();
    }

    if (row < p.m && col < p.n) {
        c[c_base + row * p.n + col] = acc;
    }
}
"#;

const NN_ACTIVATION_WGSL: &str = r#"
struct Params {
    n: u32,
    op_code: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n) {
        return;
    }
    let xv = x[idx];
    if (p.op_code == 0u) {
        let u = clamp(0.79788456 * (xv + 0.044715 * xv * xv * xv), -15.0, 15.0);
        out[idx] = 0.5 * xv * (1.0 + tanh(u));
    } else if (p.op_code == 1u) {
        let x2 = xv * xv;
        let u = clamp(0.79788456 * (xv + 0.044715 * xv * x2), -15.0, 15.0);
        let t = tanh(u);
        let sech2 = 1.0 - t * t;
        let du = 0.79788456 * (1.0 + 0.134145 * x2);
        out[idx] = dy[idx] * (0.5 * (1.0 + t) + 0.5 * xv * sech2 * du);
    } else if (p.op_code == 2u) {
        let s = 1.0 / (1.0 + exp(-clamp(xv, -80.0, 80.0)));
        out[idx] = xv * s;
    } else {
        let s = 1.0 / (1.0 + exp(-clamp(xv, -80.0, 80.0)));
        out[idx] = dy[idx] * s * (1.0 + xv * (1.0 - s));
    }
}
"#;

const LAYERNORM_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    has_gamma: u32,
    has_beta: u32,
    eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_mean: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_rstd: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= p.rows) {
        return;
    }
    let base = r * p.cols;
    var s: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        s = s + x[base + j];
    }
    let mean = s / f32(p.cols);

    var vs: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let d = x[base + j] - mean;
        vs = vs + d * d;
    }
    let rstd = inverseSqrt(vs / f32(p.cols) + p.eps);
    out_mean[r] = mean;
    out_rstd[r] = rstd;

    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let xh = (x[base + j] - mean) * rstd;
        let g = select(1.0, gamma[j], p.has_gamma != 0u);
        let b = select(0.0, beta[j], p.has_beta != 0u);
        out[base + j] = xh * g + b;
    }
}
"#;

const RMSNORM_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    has_gamma: u32,
    has_beta: u32,
    eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_rstd: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= p.rows) {
        return;
    }
    let base = r * p.cols;
    var sq: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let v = x[base + j];
        sq = sq + v * v;
    }
    let rstd = inverseSqrt(sq / f32(p.cols) + p.eps);
    out_rstd[r] = rstd;

    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let g = select(1.0, gamma[j], p.has_gamma != 0u);
        out[base + j] = x[base + j] * rstd * g;
    }
}
"#;

const CAUSAL_MASK_WGSL: &str = r#"
struct Params {
    total: u32,
    q_len: u32,
    k_len: u32,
    offset: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) {
        return;
    }
    let kj = idx % p.k_len;
    let qi = (idx / p.k_len) % p.q_len;
    if (kj > p.offset + qi) {
        scores[idx] = -3.4028234e38;
    }
}
"#;

const CROSS_ENTROPY_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    has_ignore: u32,
    ignore_index: u32,
    label_smoothing: f32,
    inv_valid: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> losses: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= p.rows) {
        return;
    }
    let tgt = targets[r];
    let base = r * p.cols;
    if (p.has_ignore != 0u && tgt == p.ignore_index) {
        losses[r] = 0.0;
        for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
            d_logits[base + j] = 0.0;
        }
        return;
    }
    var max_v = logits[base];
    for (var j: u32 = 1u; j < p.cols; j = j + 1u) {
        max_v = max(max_v, logits[base + j]);
    }
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let e = exp(logits[base + j] - max_v);
        d_logits[base + j] = e;
        sum_exp = sum_exp + e;
    }
    let log_z = max_v + log(sum_exp);
    let smooth_base = p.label_smoothing / f32(p.cols);
    let one_minus_ls = 1.0 - p.label_smoothing;
    var row_loss: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        let prob = d_logits[base + j] / sum_exp;
        let q = smooth_base + select(0.0, one_minus_ls, j == tgt);
        row_loss = row_loss - q * (logits[base + j] - log_z);
        d_logits[base + j] = (prob - q) * p.inv_valid;
    }
    losses[r] = row_loss;
}
"#;

const ADAMW_WGSL: &str = r#"
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    inv_bc1: f32,
    inv_bc2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_buf: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n) {
        return;
    }
    let g = grad[idx];
    let m_new = p.beta1 * m_buf[idx] + (1.0 - p.beta1) * g;
    let v_new = p.beta2 * v_buf[idx] + (1.0 - p.beta2) * g * g;
    m_buf[idx] = m_new;
    v_buf[idx] = v_new;
    let decayed = param[idx] * (1.0 - p.lr * p.weight_decay);
    let m_hat = m_new * p.inv_bc1;
    let v_hat = v_new * p.inv_bc2;
    param[idx] = decayed - p.lr * m_hat / (sqrt(v_hat) + p.eps);
}
"#;

const CONV_POOL_WGSL: &str = r#"
struct Params {
    total: u32,
    c: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    oh: u32,
    ow: u32,
    mode: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> argmax_buf: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) {
        return;
    }

    if (p.mode == 0u) {
        let col_w = p.c * p.kh * p.kw;
        let row = idx / col_w;
        let col = idx % col_w;
        let ow_i = row % p.ow;
        let tmp = row / p.ow;
        let oh_i = tmp % p.oh;
        let b = tmp / p.oh;
        let kx = col % p.kw;
        let tmp2 = col / p.kw;
        let ky = tmp2 % p.kh;
        let ch = tmp2 / p.kh;
        let iy = i32(oh_i * p.sh + ky * p.dh) - i32(p.ph);
        let ix = i32(ow_i * p.sw + kx * p.dw) - i32(p.pw);
        if (iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w)) {
            let in_idx = ((b * p.c + ch) * p.h + u32(iy)) * p.w + u32(ix);
            dst[idx] = src[in_idx];
        } else {
            dst[idx] = 0.0;
        }
    } else if (p.mode == 1u) {
        let ix = idx % p.w;
        let tmp = idx / p.w;
        let iy = tmp % p.h;
        let tmp2 = tmp / p.h;
        let ch = tmp2 % p.c;
        let b = tmp2 / p.c;
        let col_w = p.c * p.kh * p.kw;
        var acc: f32 = 0.0;
        for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
            let num_y = i32(iy + p.ph) - i32(ky * p.dh);
            if (num_y < 0 || (u32(num_y) % p.sh) != 0u) {
                continue;
            }
            let oh_i = u32(num_y) / p.sh;
            if (oh_i >= p.oh) {
                continue;
            }
            for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
                let num_x = i32(ix + p.pw) - i32(kx * p.dw);
                if (num_x < 0 || (u32(num_x) % p.sw) != 0u) {
                    continue;
                }
                let ow_i = u32(num_x) / p.sw;
                if (ow_i >= p.ow) {
                    continue;
                }
                let row = (b * p.oh + oh_i) * p.ow + ow_i;
                let col = (ch * p.kh + ky) * p.kw + kx;
                acc = acc + src[row * col_w + col];
            }
        }
        dst[idx] = acc;
    } else if (p.mode == 2u) {
        let ow_i = idx % p.ow;
        let tmp = idx / p.ow;
        let oh_i = tmp % p.oh;
        let tmp2 = tmp / p.oh;
        let ch = tmp2 % p.c;
        let b = tmp2 / p.c;
        var best_val: f32 = -3.4028234e38;
        var best_idx: u32 = 0xffffffffu;
        for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
            let iy = i32(oh_i * p.sh + ky) - i32(p.ph);
            if (iy < 0 || iy >= i32(p.h)) {
                continue;
            }
            for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
                let ix = i32(ow_i * p.sw + kx) - i32(p.pw);
                if (ix < 0 || ix >= i32(p.w)) {
                    continue;
                }
                let in_idx = ((b * p.c + ch) * p.h + u32(iy)) * p.w + u32(ix);
                let v = src[in_idx];
                if (v > best_val || best_idx == 0xffffffffu) {
                    best_val = v;
                    best_idx = in_idx;
                }
            }
        }
        dst[idx] = best_val;
        argmax_buf[idx] = best_idx;
    } else {
        let ow_i = idx % p.ow;
        let tmp = idx / p.ow;
        let oh_i = tmp % p.oh;
        let tmp2 = tmp / p.oh;
        let ch = tmp2 % p.c;
        let b = tmp2 / p.c;
        var acc: f32 = 0.0;
        for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
            let iy = i32(oh_i * p.sh + ky) - i32(p.ph);
            if (iy < 0 || iy >= i32(p.h)) {
                continue;
            }
            for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
                let ix = i32(ow_i * p.sw + kx) - i32(p.pw);
                if (ix < 0 || ix >= i32(p.w)) {
                    continue;
                }
                let in_idx = ((b * p.c + ch) * p.h + u32(iy)) * p.w + u32(ix);
                acc = acc + src[in_idx];
            }
        }
        dst[idx] = acc / f32(p.kh * p.kw);
    }
}
"#;

struct WgpuNnPipelines {
    batched_gemm: wgpu::ComputePipeline,
    activation: wgpu::ComputePipeline,
    layernorm: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    causal_mask: wgpu::ComputePipeline,
    cross_entropy: wgpu::ComputePipeline,
    adamw: wgpu::ComputePipeline,
    conv_pool: wgpu::ComputePipeline,
}

static WGPU_NN_PIPELINES: OnceLock<Option<WgpuNnPipelines>> = OnceLock::new();

fn get_nn_pipelines() -> Option<(&'static WgpuContext, &'static WgpuNnPipelines)> {
    let ctx = get_ctx()?;
    let pipes = WGPU_NN_PIPELINES
        .get_or_init(|| {
            let make = |label: &str, src: &str| {
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
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
            Some(WgpuNnPipelines {
                batched_gemm: make("OchreML NN BatchedGEMM", BATCHED_GEMM_WGSL),
                activation: make("OchreML NN Activation", NN_ACTIVATION_WGSL),
                layernorm: make("OchreML NN LayerNorm", LAYERNORM_WGSL),
                rmsnorm: make("OchreML NN RMSNorm", RMSNORM_WGSL),
                causal_mask: make("OchreML NN CausalMask", CAUSAL_MASK_WGSL),
                cross_entropy: make("OchreML NN CrossEntropy", CROSS_ENTROPY_WGSL),
                adamw: make("OchreML NN AdamW", ADAMW_WGSL),
                conv_pool: make("OchreML NN ConvPool", CONV_POOL_WGSL),
            })
        })
        .as_ref()?;
    Some((ctx, pipes))
}

fn readback_bytes(ctx: &WgpuContext, staging: &wgpu::Buffer, size: u64) -> OchreResult<Vec<u8>> {
    let slice = staging.slice(0..size);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| {
            OchreError::DeviceError("wgpu NN readback channel closed ( >_< )".to_string())
        })?
        .map_err(|e| OchreError::DeviceError(format!("wgpu NN buffer map failed: {e} ( >_< )")))?;
    let view = slice.get_mapped_range();
    let out = view.to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

#[inline]
fn create_f32_ro_buf(ctx: &WgpuContext, label: &str, data: &[f32]) -> wgpu::Buffer {
    let dummy = [0.0f32];
    let slice = if data.is_empty() { &dummy } else { data };
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

#[inline]
fn create_rw_buf_bytes(
    ctx: &WgpuContext,
    label: &str,
    byte_size: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let sz = byte_size.max(4);
    let dev_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nn_stg"),
        size: sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (dev_buf, stg_buf)
}

/// N7: wgpu batched SGEMM.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_batched_matmul_f32(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: bool,
    trans_b: bool,
) -> Option<OchreResult<Vec<f32>>> {
    if batch == 0
        || m == 0
        || n == 0
        || k == 0
        || batch.saturating_mul(m).saturating_mul(n).saturating_mul(k) < MIN_WGPU_GEMM_OPS
    {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let params = GemmParams {
        batch: batch as u32,
        m: m as u32,
        n: n as u32,
        k: k as u32,
        trans_a: u32::from(trans_a),
        trans_b: u32::from(trans_b),
        _pad0: 0,
        _pad1: 0,
    };
    let out_len = batch * m * n;
    let out_bytes = (out_len * 4) as u64;
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let a_buf = create_f32_ro_buf(ctx, "gemm_a", &a[..batch * m * k]);
    let b_buf = create_f32_ro_buf(ctx, "gemm_b", &b[..batch * k * n]);
    let (c_buf, stg) = create_rw_buf_bytes(ctx, "gemm_c", out_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_bg"),
        layout: &pipes.batched_gemm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
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
                resource: c_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.batched_gemm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            (n as u32).div_ceil(16),
            (m as u32).div_ceil(16),
            batch as u32,
        );
    }
    enc.copy_buffer_to_buffer(&c_buf, 0, &stg, 0, out_bytes);
    ctx.queue.submit(Some(enc.finish()));
    let raw = match readback_bytes(ctx, &stg, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out: &[f32] = bytemuck::cast_slice(&raw);
    Some(Ok(out[..out_len].to_vec()))
}

fn wgpu_run_activation(
    x: &[f32],
    dy: Option<&[f32]>,
    op_code: u32,
) -> Option<OchreResult<Vec<f32>>> {
    if x.len() < MIN_WGPU_ELEMS {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let n = x.len();
    let out_bytes = (n * 4) as u64;
    let params = UnaryParams {
        n: n as u32,
        op_code,
        _pad0: 0,
        _pad1: 0,
    };
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("act_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let x_buf = create_f32_ro_buf(ctx, "act_x", x);
    let dy_buf = create_f32_ro_buf(ctx, "act_dy", dy.unwrap_or(&[]));
    let (out_buf, stg) = create_rw_buf_bytes(ctx, "act_out", out_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("act_bg"),
        layout: &pipes.activation.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dy_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.activation);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &stg, 0, out_bytes);
    ctx.queue.submit(Some(enc.finish()));
    let raw = match readback_bytes(ctx, &stg, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out: &[f32] = bytemuck::cast_slice(&raw);
    Some(Ok(out[..n].to_vec()))
}

/// N8: wgpu GELU forward.
pub fn wgpu_gelu_f32(x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    wgpu_run_activation(x, None, 0)
}

/// N8: wgpu GELU backward.
pub fn wgpu_gelu_bwd_f32(dy: &[f32], x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if dy.len() != x.len() {
        return None;
    }
    wgpu_run_activation(x, Some(dy), 1)
}

/// N8: wgpu SiLU forward.
pub fn wgpu_silu_f32(x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    wgpu_run_activation(x, None, 2)
}

/// N8: wgpu SiLU backward.
pub fn wgpu_silu_bwd_f32(dy: &[f32], x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if dy.len() != x.len() {
        return None;
    }
    wgpu_run_activation(x, Some(dy), 3)
}

/// N8: wgpu row LayerNorm forward.
#[allow(clippy::type_complexity)]
pub fn wgpu_row_layernorm_f32(
    x: &[f32],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if rows * cols < MIN_WGPU_ELEMS {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let params = NormParams {
        rows: rows as u32,
        cols: cols as u32,
        has_gamma: u32::from(gamma.is_some()),
        has_beta: u32::from(beta.is_some()),
        eps,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let out_bytes = (x.len() * 4) as u64;
    let stat_bytes = (rows * 4) as u64;
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ln_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let x_buf = create_f32_ro_buf(ctx, "ln_x", x);
    let g_buf = create_f32_ro_buf(ctx, "ln_g", gamma.unwrap_or(&[]));
    let b_buf = create_f32_ro_buf(ctx, "ln_b", beta.unwrap_or(&[]));
    let (out_buf, out_stg) = create_rw_buf_bytes(ctx, "ln_out", out_bytes);
    let (mean_buf, mean_stg) = create_rw_buf_bytes(ctx, "ln_mean", stat_bytes);
    let (rstd_buf, rstd_stg) = create_rw_buf_bytes(ctx, "ln_rstd", stat_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ln_bg"),
        layout: &pipes.layernorm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: g_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: mean_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: rstd_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.layernorm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &out_stg, 0, out_bytes);
    enc.copy_buffer_to_buffer(&mean_buf, 0, &mean_stg, 0, stat_bytes);
    enc.copy_buffer_to_buffer(&rstd_buf, 0, &rstd_stg, 0, stat_bytes);
    ctx.queue.submit(Some(enc.finish()));

    let out_raw = match readback_bytes(ctx, &out_stg, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let mean_raw = match readback_bytes(ctx, &mean_stg, stat_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let rstd_raw = match readback_bytes(ctx, &rstd_stg, stat_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out: &[f32] = bytemuck::cast_slice(&out_raw);
    let mean: &[f32] = bytemuck::cast_slice(&mean_raw);
    let rstd: &[f32] = bytemuck::cast_slice(&rstd_raw);
    Some(Ok((
        out[..x.len()].to_vec(),
        mean[..rows].to_vec(),
        rstd[..rows].to_vec(),
    )))
}

/// N8: wgpu row RMSNorm forward.
pub fn wgpu_row_rmsnorm_f32(
    x: &[f32],
    gamma: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>)>> {
    if rows * cols < MIN_WGPU_ELEMS {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let params = NormParams {
        rows: rows as u32,
        cols: cols as u32,
        has_gamma: u32::from(gamma.is_some()),
        has_beta: 0,
        eps,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let out_bytes = (x.len() * 4) as u64;
    let stat_bytes = (rows * 4) as u64;
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rn_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let x_buf = create_f32_ro_buf(ctx, "rn_x", x);
    let g_buf = create_f32_ro_buf(ctx, "rn_g", gamma.unwrap_or(&[]));
    let (out_buf, out_stg) = create_rw_buf_bytes(ctx, "rn_out", out_bytes);
    let (rstd_buf, rstd_stg) = create_rw_buf_bytes(ctx, "rn_rstd", stat_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rn_bg"),
        layout: &pipes.rmsnorm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: g_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: rstd_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.rmsnorm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &out_stg, 0, out_bytes);
    enc.copy_buffer_to_buffer(&rstd_buf, 0, &rstd_stg, 0, stat_bytes);
    ctx.queue.submit(Some(enc.finish()));

    let out_raw = match readback_bytes(ctx, &out_stg, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let rstd_raw = match readback_bytes(ctx, &rstd_stg, stat_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out: &[f32] = bytemuck::cast_slice(&out_raw);
    let rstd: &[f32] = bytemuck::cast_slice(&rstd_raw);
    Some(Ok((out[..x.len()].to_vec(), rstd[..rows].to_vec())))
}

/// N8: wgpu causal mask in-place.
pub fn wgpu_causal_mask_f32(
    scores: &mut [f32],
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    offset: usize,
) -> Option<OchreResult<()>> {
    let total = batch_heads * q_len * k_len;
    if total < MIN_WGPU_ELEMS {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let params = CausalMaskParams {
        total: total as u32,
        q_len: q_len as u32,
        k_len: k_len as u32,
        offset: offset as u32,
    };
    let byte_len = (total * 4) as u64;
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cm_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let s_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cm_s"),
            contents: bytemuck::cast_slice(scores),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let stg = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cm_stg"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cm_bg"),
        layout: &pipes.causal_mask.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: s_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.causal_mask);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total as u32).div_ceil(256), 1, 1);
    }
    enc.copy_buffer_to_buffer(&s_buf, 0, &stg, 0, byte_len);
    ctx.queue.submit(Some(enc.finish()));
    let raw = match readback_bytes(ctx, &stg, byte_len) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out: &[f32] = bytemuck::cast_slice(&raw);
    scores.copy_from_slice(&out[..total]);
    Some(Ok(()))
}

/// N9: wgpu cross_entropy_grad_f32.
pub fn wgpu_cross_entropy_grad_f32(
    logits: &[f32],
    targets: &[usize],
    rows: usize,
    cols: usize,
    label_smoothing: f32,
    ignore_index: Option<usize>,
) -> Option<OchreResult<(f32, Vec<f32>)>> {
    if rows * cols < MIN_WGPU_ELEMS {
        return None;
    }
    let valid_count = targets.iter().filter(|&&t| Some(t) != ignore_index).count();
    if valid_count == 0 {
        return Some(Ok((0.0, vec![0.0; rows * cols])));
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let tgt_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
    let params = CrossEntropyParams {
        rows: rows as u32,
        cols: cols as u32,
        has_ignore: u32::from(ignore_index.is_some()),
        ignore_index: ignore_index.unwrap_or(usize::MAX) as u32,
        label_smoothing,
        inv_valid: 1.0 / (valid_count as f32),
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let grad_bytes = (logits.len() * 4) as u64;
    let loss_bytes = (rows * 4) as u64;
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ce_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let l_buf = create_f32_ro_buf(ctx, "ce_l", logits);
    let t_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ce_t"),
            contents: bytemuck::cast_slice(&tgt_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let (g_buf, g_stg) = create_rw_buf_bytes(ctx, "ce_g", grad_bytes);
    let (loss_buf, loss_stg) = create_rw_buf_bytes(ctx, "ce_loss", loss_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ce_bg"),
        layout: &pipes.cross_entropy.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: l_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: t_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: g_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: loss_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.cross_entropy);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&g_buf, 0, &g_stg, 0, grad_bytes);
    enc.copy_buffer_to_buffer(&loss_buf, 0, &loss_stg, 0, loss_bytes);
    ctx.queue.submit(Some(enc.finish()));

    let g_raw = match readback_bytes(ctx, &g_stg, grad_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let loss_raw = match readback_bytes(ctx, &loss_stg, loss_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let d_grad: &[f32] = bytemuck::cast_slice(&g_raw);
    let row_losses: &[f32] = bytemuck::cast_slice(&loss_raw);
    let sum_loss: f64 = row_losses[..rows].iter().map(|&v| v as f64).sum();
    Some(Ok((
        (sum_loss / valid_count as f64) as f32,
        d_grad[..logits.len()].to_vec(),
    )))
}

/// N9: wgpu fused AdamW step.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_adamw_step_f32(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
) -> Option<OchreResult<()>> {
    if param.len() < MIN_WGPU_ELEMS {
        return None;
    }
    let (ctx, pipes) = get_nn_pipelines()?;
    let n = param.len();
    let byte_len = (n * 4) as u64;
    let params = AdamwParams {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        inv_bc1: (1.0 / (1.0 - (beta1 as f64).powi(step as i32))) as f32,
        inv_bc2: (1.0 / (1.0 - (beta2 as f64).powi(step as i32))) as f32,
        _pad3: 0.0,
    };
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aw_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let mk_rw = |label: &str, data: &[f32]| {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    };
    let param_buf = mk_rw("aw_param", param);
    let grad_buf = create_f32_ro_buf(ctx, "aw_grad", grad);
    let m_buf = mk_rw("aw_m", m);
    let v_buf = mk_rw("aw_v", v);
    let mk_stg = |label: &str| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let param_stg = mk_stg("aw_param_stg");
    let m_stg = mk_stg("aw_m_stg");
    let v_stg = mk_stg("aw_v_stg");

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aw_bg"),
        layout: &pipes.adamw.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grad_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: m_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: v_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.adamw);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
    }
    enc.copy_buffer_to_buffer(&param_buf, 0, &param_stg, 0, byte_len);
    enc.copy_buffer_to_buffer(&m_buf, 0, &m_stg, 0, byte_len);
    enc.copy_buffer_to_buffer(&v_buf, 0, &v_stg, 0, byte_len);
    ctx.queue.submit(Some(enc.finish()));

    let p_raw = match readback_bytes(ctx, &param_stg, byte_len) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let m_raw = match readback_bytes(ctx, &m_stg, byte_len) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let v_raw = match readback_bytes(ctx, &v_stg, byte_len) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    param.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&p_raw)[..n]);
    m.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&m_raw)[..n]);
    v.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&v_raw)[..n]);
    Some(Ok(()))
}

fn wgpu_run_conv_pool(
    src: &[f32],
    params: ConvPoolParams,
    out_len: usize,
    need_argmax: bool,
) -> Option<OchreResult<(Vec<f32>, Vec<usize>)>> {
    let (ctx, pipes) = get_nn_pipelines()?;
    let out_bytes = (out_len * 4) as u64;
    let arg_bytes = if need_argmax { out_bytes } else { 4u64 };
    let p_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cp_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let src_buf = create_f32_ro_buf(ctx, "cp_src", src);
    let (dst_buf, dst_stg) = create_rw_buf_bytes(ctx, "cp_dst", out_bytes);
    let (arg_buf, arg_stg) = create_rw_buf_bytes(ctx, "cp_arg", arg_bytes);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cp_bg"),
        layout: &pipes.conv_pool.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: arg_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipes.conv_pool);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(params.total.div_ceil(256), 1, 1);
    }
    enc.copy_buffer_to_buffer(&dst_buf, 0, &dst_stg, 0, out_bytes);
    if need_argmax {
        enc.copy_buffer_to_buffer(&arg_buf, 0, &arg_stg, 0, arg_bytes);
    }
    ctx.queue.submit(Some(enc.finish()));

    let dst_raw = match readback_bytes(ctx, &dst_stg, out_bytes) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let out = bytemuck::cast_slice::<u8, f32>(&dst_raw)[..out_len].to_vec();
    let argmax = if need_argmax {
        let arg_raw = match readback_bytes(ctx, &arg_stg, arg_bytes) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        bytemuck::cast_slice::<u8, u32>(&arg_raw)[..out_len]
            .iter()
            .map(|&v| {
                if v == u32::MAX {
                    usize::MAX
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    Some(Ok((out, argmax)))
}

/// N10: wgpu im2col.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_im2col_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
    out_h: usize,
    out_w: usize,
) -> Option<OchreResult<Vec<f32>>> {
    let total = batch * out_h * out_w * c * kh * kw;
    if total < MIN_WGPU_ELEMS {
        return None;
    }
    let params = ConvPoolParams {
        total: total as u32,
        c: c as u32,
        h: h as u32,
        w: w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: stride_h as u32,
        sw: stride_w as u32,
        ph: pad_h as u32,
        pw: pad_w as u32,
        dh: dil_h as u32,
        dw: dil_w as u32,
        oh: out_h as u32,
        ow: out_w as u32,
        mode: 0,
        _pad: 0,
    };
    wgpu_run_conv_pool(input, params, total, false).map(|r| r.map(|(v, _)| v))
}

/// N10: wgpu col2im.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_col2im_f32(
    col: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
    out_h: usize,
    out_w: usize,
) -> Option<OchreResult<Vec<f32>>> {
    if col.len() < MIN_WGPU_ELEMS {
        return None;
    }
    let total = batch * c * h * w;
    let params = ConvPoolParams {
        total: total as u32,
        c: c as u32,
        h: h as u32,
        w: w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: stride_h as u32,
        sw: stride_w as u32,
        ph: pad_h as u32,
        pw: pad_w as u32,
        dh: dil_h as u32,
        dw: dil_w as u32,
        oh: out_h as u32,
        ow: out_w as u32,
        mode: 1,
        _pad: 0,
    };
    wgpu_run_conv_pool(col, params, total, false).map(|r| r.map(|(v, _)| v))
}

/// N10: wgpu max_pool2d.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_max_pool2d_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
) -> Option<OchreResult<(Vec<f32>, Vec<usize>)>> {
    if input.len() < MIN_WGPU_ELEMS {
        return None;
    }
    let total = batch * c * out_h * out_w;
    let params = ConvPoolParams {
        total: total as u32,
        c: c as u32,
        h: h as u32,
        w: w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: stride_h as u32,
        sw: stride_w as u32,
        ph: pad_h as u32,
        pw: pad_w as u32,
        dh: 1,
        dw: 1,
        oh: out_h as u32,
        ow: out_w as u32,
        mode: 2,
        _pad: 0,
    };
    wgpu_run_conv_pool(input, params, total, true)
}

/// N10: wgpu avg_pool2d.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_avg_pool2d_f32(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
) -> Option<OchreResult<Vec<f32>>> {
    if input.len() < MIN_WGPU_ELEMS {
        return None;
    }
    let total = batch * c * out_h * out_w;
    let params = ConvPoolParams {
        total: total as u32,
        c: c as u32,
        h: h as u32,
        w: w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: stride_h as u32,
        sw: stride_w as u32,
        ph: pad_h as u32,
        pw: pad_w as u32,
        dh: 1,
        dw: 1,
        oh: out_h as u32,
        ow: out_w as u32,
        mode: 3,
        _pad: 0,
    };
    wgpu_run_conv_pool(input, params, total, false).map(|r| r.map(|(v, _)| v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_nn_shaders_and_fallback() {
        assert!(BATCHED_GEMM_WGSL.contains("workgroupBarrier"));
        assert!(NN_ACTIVATION_WGSL.contains("tanh"));
        assert!(ADAMW_WGSL.contains("weight_decay"));
        let small = vec![1.0f32; 32];
        assert!(wgpu_gelu_f32(&small).is_none());
        assert!(wgpu_batched_matmul_f32(1, 4, 4, 4, &small, &small, false, false).is_none());
    }
}
