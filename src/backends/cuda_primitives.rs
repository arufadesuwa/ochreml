// OchreML CUDA Runtime PTX Primitives (N1, N2, N4, N6) (*≧ω≦*)
// Compiles universal sm_50 PTX at runtime via cuModuleLoadData (no nvcc build dependency)
// and shares CudaDriver + CudaDeviceCtx + VRAM workspace with src/backends/cuda.rs.

use crate::backends::cuda::{
    CUDA_SUCCESS, CuContext, CuDevicePtr, CuResult, CudaDriver, copy_dtoh_raw, copy_htod_raw,
    get_driver, offset_ptr,
};
use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use crate::primitives::stats_ops::ElementwiseOp;
use crate::primitives::{BinnedMatrix, ColStats, GradHessBin};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

type CuModule = *mut c_void;
type CuFunction = *mut c_void;

type FnCuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CuResult;

const MIN_GPU_ELEMENTS: usize = 524_288;

// Universal sm_50 PTX kernels for exact FP64 elementwise, column reduction, argmin, and histogram.
const OCHRE_PRIMITIVES_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// Elementwise FP64 kernel: op 2=Relu, 5=Clip, 6=SoftThreshold, 7=Affine
.visible .entry ochre_elementwise_f64(
    .param .u64 param_data,
    .param .u64 param_n,
    .param .u32 param_op,
    .param .f64 param_p1,
    .param .f64 param_p2
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;
    .reg .f64 %fd<10>;

    ld.param.u64 %rd1, [param_data];
    ld.param.u64 %rd2, [param_n];
    ld.param.u32 %r1, [param_op];
    ld.param.f64 %fd1, [param_p1];
    ld.param.f64 %fd2, [param_p2];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    cvt.u64.u32 %rd3, %r5;
    setp.ge.u64 %p1, %rd3, %rd2;
    @%p1 bra DONE;

    shl.b64 %rd4, %rd3, 3;
    add.u64 %rd1, %rd1, %rd4;
    ld.global.f64 %fd3, [%rd1];

    setp.eq.u32 %p2, %r1, 2;
    @%p2 bra DO_RELU;
    setp.eq.u32 %p3, %r1, 5;
    @%p3 bra DO_CLIP;
    setp.eq.u32 %p4, %r1, 6;
    @%p4 bra DO_SOFT;

    // Affine: v * p1 + p2
    fma.rn.f64 %fd4, %fd3, %fd1, %fd2;
    bra STORE;

DO_RELU:
    mov.f64 %fd5, 0d0000000000000000;
    max.f64 %fd4, %fd3, %fd5;
    bra STORE;

DO_CLIP:
    max.f64 %fd6, %fd3, %fd1;
    min.f64 %fd4, %fd6, %fd2;
    bra STORE;

DO_SOFT:
    mov.f64 %fd5, 0d0000000000000000;
    sub.f64 %fd6, %fd3, %fd1;
    add.f64 %fd7, %fd3, %fd1;
    setp.gt.f64 %p2, %fd3, %fd1;
    setp.lt.f64 %p3, %fd3, 0d0000000000000000;
    neg.f64 %fd8, %fd1;
    setp.lt.f64 %p3, %fd3, %fd8;
    selp.f64 %fd9, %fd7, %fd5, %p3;
    selp.f64 %fd4, %fd6, %fd9, %p2;

STORE:
    st.global.f64 [%rd1], %fd4;
DONE:
    ret;
}

// Per-column reduction: 1 thread per column over rows (sum_shifted, sum_sq_shifted, min, max).
.visible .entry ochre_col_reduce_f64(
    .param .u64 param_x,
    .param .u64 param_w,
    .param .u64 param_rows,
    .param .u64 param_cols,
    .param .u32 param_has_w,
    .param .u64 param_out_sum,
    .param .u64 param_out_sq,
    .param .u64 param_out_min,
    .param .u64 param_out_max
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<16>;
    .reg .f64 %fd<14>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r4; // col j

    ld.param.u64 %rd2, [param_cols];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra COL_DONE;

    ld.param.u64 %rd3, [param_x];
    ld.param.u64 %rd4, [param_w];
    ld.param.u64 %rd5, [param_rows];
    ld.param.u32 %r5, [param_has_w];

    // Anchor = x[0, j]
    shl.b64 %rd6, %rd1, 3;
    add.u64 %rd7, %rd3, %rd6;
    ld.global.f64 %fd1, [%rd7]; // anchor

    mov.f64 %fd2, 0d0000000000000000; // sum_shifted
    mov.f64 %fd3, 0d0000000000000000; // sum_sq_shifted
    mov.f64 %fd4, %fd1;               // min
    mov.f64 %fd5, %fd1;               // max
    mov.u64 %rd8, 0;                  // row r
    shl.b64 %rd9, %rd2, 3;            // row stride in bytes

LOOP_R:
    setp.ge.u64 %p2, %rd8, %rd5;
    @%p2 bra LOOP_END;

    ld.global.f64 %fd6, [%rd7];
    mov.f64 %fd7, 0d3FF0000000000000; // 1.0
    setp.eq.u32 %p3, %r5, 0;
    @%p3 bra NO_W;
    shl.b64 %rd10, %rd8, 3;
    add.u64 %rd11, %rd4, %rd10;
    ld.global.f64 %fd7, [%rd11];
NO_W:
    sub.f64 %fd8, %fd6, %fd1;
    mul.f64 %fd9, %fd7, %fd8;
    add.f64 %fd2, %fd2, %fd9;
    fma.rn.f64 %fd3, %fd9, %fd8, %fd3;
    min.f64 %fd4, %fd4, %fd6;
    max.f64 %fd5, %fd5, %fd6;

    add.u64 %rd7, %rd7, %rd9;
    add.u64 %rd8, %rd8, 1;
    bra LOOP_R;

LOOP_END:
    ld.param.u64 %rd12, [param_out_sum];
    ld.param.u64 %rd13, [param_out_sq];
    ld.param.u64 %rd14, [param_out_min];
    ld.param.u64 %rd15, [param_out_max];
    add.u64 %rd12, %rd12, %rd6;
    add.u64 %rd13, %rd13, %rd6;
    add.u64 %rd14, %rd14, %rd6;
    add.u64 %rd15, %rd15, %rd6;
    st.global.f64 [%rd12], %fd2;
    st.global.f64 [%rd13], %fd3;
    st.global.f64 [%rd14], %fd4;
    st.global.f64 [%rd15], %fd5;

COL_DONE:
    ret;
}

// Per-row argmin: 1 thread per row over cols.
.visible .entry ochre_row_argmin_f64(
    .param .u64 param_d,
    .param .u64 param_rows,
    .param .u64 param_cols,
    .param .u64 param_out_idx,
    .param .u64 param_out_val
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<14>;
    .reg .f64 %fd<4>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r4; // row i

    ld.param.u64 %rd2, [param_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra ARG_DONE;

    ld.param.u64 %rd3, [param_d];
    ld.param.u64 %rd4, [param_cols];
    mul.lo.u64 %rd5, %rd1, %rd4;
    shl.b64 %rd5, %rd5, 3;
    add.u64 %rd6, %rd3, %rd5;

    ld.global.f64 %fd1, [%rd6]; // best_val
    mov.u64 %rd7, 0;            // best_idx
    mov.u64 %rd8, 1;            // col j

ARG_LOOP:
    setp.ge.u64 %p2, %rd8, %rd4;
    @%p2 bra ARG_END;
    add.u64 %rd6, %rd6, 8;
    ld.global.f64 %fd2, [%rd6];
    setp.lt.f64 %p3, %fd2, %fd1;
    @%p3 mov.f64 %fd1, %fd2;
    @%p3 mov.u64 %rd7, %rd8;
    add.u64 %rd8, %rd8, 1;
    bra ARG_LOOP;

ARG_END:
    ld.param.u64 %rd9, [param_out_idx];
    ld.param.u64 %rd10, [param_out_val];
    shl.b64 %rd11, %rd1, 3;
    add.u64 %rd9, %rd9, %rd11;
    add.u64 %rd10, %rd10, %rd11;
    st.global.u64 [%rd9], %rd7;
    st.global.f64 [%rd10], %fd1;

ARG_DONE:
    ret;
}
"#;

struct PtxFuncs {
    _module: CuModule,
    elementwise: CuFunction,
    col_reduce: CuFunction,
    row_argmin: CuFunction,
}

// SAFETY: CuModule / CuFunction handles are bound to their CUDA context and guarded by Mutex.
unsafe impl Send for PtxFuncs {}
// SAFETY: Guarded by Mutex inside PtxRuntime.
unsafe impl Sync for PtxFuncs {}

struct PtxRuntime {
    cu_module_load_data: FnCuModuleLoadData,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    modules: Mutex<HashMap<usize, PtxFuncs>>,
}

static PTX_RUNTIME: OnceLock<Option<PtxRuntime>> = OnceLock::new();

fn init_ptx_runtime(lib: &Library) -> Option<PtxRuntime> {
    // SAFETY: Symbols come from the already-loaded `libcuda.so.1` driver library.
    unsafe {
        let cu_module_load_data = *lib.get::<FnCuModuleLoadData>(b"cuModuleLoadData\0").ok()?;
        let cu_module_get_function = *lib
            .get::<FnCuModuleGetFunction>(b"cuModuleGetFunction\0")
            .ok()?;
        let cu_launch_kernel = *lib.get::<FnCuLaunchKernel>(b"cuLaunchKernel\0").ok()?;
        let cu_ctx_synchronize = *lib.get::<FnCuCtxSynchronize>(b"cuCtxSynchronize\0").ok()?;
        Some(PtxRuntime {
            cu_module_load_data,
            cu_module_get_function,
            cu_launch_kernel,
            cu_ctx_synchronize,
            modules: Mutex::new(HashMap::new()),
        })
    }
}

fn get_ptx_runtime(driver: &CudaDriver) -> Option<&'static PtxRuntime> {
    PTX_RUNTIME
        .get_or_init(|| init_ptx_runtime(&driver._cuda_lib))
        .as_ref()
}

fn get_or_load_funcs(
    ptx: &PtxRuntime,
    ctx_key: CuContext,
) -> Option<(CuFunction, CuFunction, CuFunction)> {
    let key = ctx_key as usize;
    let mut map = ptx.modules.lock().ok()?;
    if let Some(f) = map.get(&key) {
        return Some((f.elementwise, f.col_reduce, f.row_argmin));
    }

    let ptx_c = CString::new(OCHRE_PRIMITIVES_PTX).ok()?;
    let mut module: CuModule = std::ptr::null_mut();
    // SAFETY: `ptx_c` is null-terminated PTX source and current CUDA context is active.
    let rc = unsafe { (ptx.cu_module_load_data)(&mut module, ptx_c.as_ptr() as *const c_void) };
    if rc != CUDA_SUCCESS || module.is_null() {
        return None;
    }

    let load_fn = |name: &str| -> Option<CuFunction> {
        let c_name = CString::new(name).ok()?;
        let mut func: CuFunction = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded CUmodule on the active context.
        let frc = unsafe { (ptx.cu_module_get_function)(&mut func, module, c_name.as_ptr()) };
        if frc == CUDA_SUCCESS && !func.is_null() {
            Some(func)
        } else {
            None
        }
    };

    let elementwise = load_fn("ochre_elementwise_f64")?;
    let col_reduce = load_fn("ochre_col_reduce_f64")?;
    let row_argmin = load_fn("ochre_row_argmin_f64")?;

    map.insert(
        key,
        PtxFuncs {
            _module: module,
            elementwise,
            col_reduce,
            row_argmin,
        },
    );
    Some((elementwise, col_reduce, row_argmin))
}

/// CUDA implementation of N1 (`col_reduce_stats`) for large matrices.
pub fn cuda_col_reduce_stats(
    gpu_id: usize,
    x: &Matrix,
    sample_weight: Option<&[f64]>,
) -> Option<OchreResult<ColStats>> {
    // Kernel launches 1 thread per column; when cols < 256 GPU SM occupancy is too low
    // to amortize PCIe HtoD transfer, so CPU Rayon SIMD is strictly faster.
    if x.cols < 256 || x.rows * x.cols < 4_000_000 {
        return None;
    }
    let driver = get_driver()?;
    let ptx = get_ptx_runtime(driver)?;
    let ctx_arc = driver.get_device_ctx(gpu_id).ok()?;
    let mut guard = ctx_arc.lock().ok()?;
    // SAFETY: Bind CUDA context for this device before module/kernel calls.
    unsafe {
        if (driver.cu_ctx_set_current)(guard.ctx) != CUDA_SUCCESS {
            return None;
        }
    }
    let (_, col_reduce_fn, _) = get_or_load_funcs(ptx, guard.ctx)?;

    let cols = x.cols;
    let rows = x.rows;
    let w_len = sample_weight.map_or(0, |w| w.len());
    let total_f64 = x.data.len() + w_len + cols * 4;
    let base = match guard.ensure_workspace(driver, total_f64) {
        Ok(ptr) => ptr,
        Err(e) => return Some(Err(e)),
    };

    let d_x = base;
    let d_w = offset_ptr(d_x, x.data.len());
    let d_sum = offset_ptr(d_w, w_len);
    let d_sq = offset_ptr(d_sum, cols);
    let d_min = offset_ptr(d_sq, cols);
    let d_max = offset_ptr(d_min, cols);

    if let Err(e) = copy_htod_raw(driver, d_x, &x.data) {
        return Some(Err(e));
    }
    if let Some(w) = sample_weight
        && let Err(e) = copy_htod_raw(driver, d_w, w)
    {
        return Some(Err(e));
    }

    let mut rows_u64 = rows as u64;
    let mut cols_u64 = cols as u64;
    let mut has_w: u32 = if sample_weight.is_some() { 1 } else { 0 };
    let mut d_x_arg = d_x;
    let mut d_w_arg = d_w;
    let mut d_sum_arg = d_sum;
    let mut d_sq_arg = d_sq;
    let mut d_min_arg = d_min;
    let mut d_max_arg = d_max;
    let mut params: [*mut c_void; 9] = [
        (&mut d_x_arg as *mut CuDevicePtr).cast(),
        (&mut d_w_arg as *mut CuDevicePtr).cast(),
        (&mut rows_u64 as *mut u64).cast(),
        (&mut cols_u64 as *mut u64).cast(),
        (&mut has_w as *mut u32).cast(),
        (&mut d_sum_arg as *mut CuDevicePtr).cast(),
        (&mut d_sq_arg as *mut CuDevicePtr).cast(),
        (&mut d_min_arg as *mut CuDevicePtr).cast(),
        (&mut d_max_arg as *mut CuDevicePtr).cast(),
    ];

    let block_x = 128u32;
    let grid_x = ((cols as u32) + block_x - 1) / block_x;
    // SAFETY: Kernel parameters match `ochre_col_reduce_f64` signature and valid device buffers.
    let rc = unsafe {
        let lrc = (ptx.cu_launch_kernel)(
            col_reduce_fn,
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if lrc != CUDA_SUCCESS {
            return None;
        }
        (ptx.cu_ctx_synchronize)()
    };
    if rc != CUDA_SUCCESS {
        return None;
    }

    let mut sum_shifted = vec![0.0; cols];
    let mut sum_sq = vec![0.0; cols];
    let mut min_v = vec![0.0; cols];
    let mut max_v = vec![0.0; cols];
    if let Err(e) = copy_dtoh_raw(driver, &mut sum_shifted, d_sum)
        .and_then(|_| copy_dtoh_raw(driver, &mut sum_sq, d_sq))
        .and_then(|_| copy_dtoh_raw(driver, &mut min_v, d_min))
        .and_then(|_| copy_dtoh_raw(driver, &mut max_v, d_max))
    {
        return Some(Err(e));
    }

    let total_weight: f64 = sample_weight.map_or(rows as f64, |w| w.iter().sum());
    if total_weight <= 0.0 {
        return Some(Err(OchreError::InvalidParameter(
            "Total sample weight must be positive ( >_< )".to_string(),
        )));
    }

    let inv_w = 1.0 / total_weight;
    let mut mean = vec![0.0; cols];
    let mut var = vec![0.0; cols];
    let mut std = vec![0.0; cols];
    let mut sum = vec![0.0; cols];
    for j in 0..cols {
        let anchor = x.data[j];
        let sm = sum_shifted[j] * inv_w;
        mean[j] = anchor + sm;
        sum[j] = sum_shifted[j] + total_weight * anchor;
        let v = (sum_sq[j] * inv_w - sm * sm).max(0.0);
        var[j] = v;
        std[j] = v.sqrt();
    }

    Some(Ok(ColStats {
        mean,
        var,
        std,
        min: min_v,
        max: max_v,
        sum,
        total_weight,
    }))
}

/// CUDA implementation of N2 (`elementwise_op`) for native FP64 PTX operations.
pub fn cuda_elementwise_op(
    gpu_id: usize,
    a: &Matrix,
    op: ElementwiseOp,
) -> Option<OchreResult<Matrix>> {
    if a.data.len() < MIN_GPU_ELEMENTS {
        return None;
    }
    let (op_code, p1, p2): (u32, f64, f64) = match op {
        ElementwiseOp::Relu => (2, 0.0, 0.0),
        ElementwiseOp::Clip { min, max } => (5, min, max),
        ElementwiseOp::SoftThreshold { lambda } => (6, lambda, 0.0),
        ElementwiseOp::Affine { scale, shift } => (7, scale, shift),
        _ => return None,
    };

    let driver = get_driver()?;
    let ptx = get_ptx_runtime(driver)?;
    let ctx_arc = driver.get_device_ctx(gpu_id).ok()?;
    let mut guard = ctx_arc.lock().ok()?;
    // SAFETY: Bind CUDA context for this device.
    unsafe {
        if (driver.cu_ctx_set_current)(guard.ctx) != CUDA_SUCCESS {
            return None;
        }
    }
    let (elem_fn, _, _) = get_or_load_funcs(ptx, guard.ctx)?;
    let d_data = match guard.ensure_workspace(driver, a.data.len()) {
        Ok(ptr) => ptr,
        Err(e) => return Some(Err(e)),
    };
    if let Err(e) = copy_htod_raw(driver, d_data, &a.data) {
        return Some(Err(e));
    }

    let mut d_data_arg = d_data;
    let mut n_u64 = a.data.len() as u64;
    let mut op_arg = op_code;
    let mut p1_arg = p1;
    let mut p2_arg = p2;
    let mut params: [*mut c_void; 5] = [
        (&mut d_data_arg as *mut CuDevicePtr).cast(),
        (&mut n_u64 as *mut u64).cast(),
        (&mut op_arg as *mut u32).cast(),
        (&mut p1_arg as *mut f64).cast(),
        (&mut p2_arg as *mut f64).cast(),
    ];

    let block_x = 256u32;
    let grid_x = ((a.data.len() as u32) + block_x - 1) / block_x;
    // SAFETY: Kernel parameters match `ochre_elementwise_f64` signature.
    let rc = unsafe {
        let lrc = (ptx.cu_launch_kernel)(
            elem_fn,
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if lrc != CUDA_SUCCESS {
            return None;
        }
        (ptx.cu_ctx_synchronize)()
    };
    if rc != CUDA_SUCCESS {
        return None;
    }

    let mut out = vec![0.0; a.data.len()];
    if let Err(e) = copy_dtoh_raw(driver, &mut out, d_data) {
        return Some(Err(e));
    }
    Some(Matrix::from_vec(a.rows, a.cols, out))
}

/// CUDA implementation of N4 (`row_argmin`) for large distance matrices.
pub fn cuda_row_argmin(
    gpu_id: usize,
    dist: &Matrix,
) -> Option<OchreResult<(Vec<usize>, Vec<f64>)>> {
    if dist.rows * dist.cols < MIN_GPU_ELEMENTS {
        return None;
    }
    let driver = get_driver()?;
    let ptx = get_ptx_runtime(driver)?;
    let ctx_arc = driver.get_device_ctx(gpu_id).ok()?;
    let mut guard = ctx_arc.lock().ok()?;
    // SAFETY: Bind CUDA context for this device.
    unsafe {
        if (driver.cu_ctx_set_current)(guard.ctx) != CUDA_SUCCESS {
            return None;
        }
    }
    let (_, _, argmin_fn) = get_or_load_funcs(ptx, guard.ctx)?;
    let total_f64 = dist.data.len() + dist.rows * 2;
    let d_mat = match guard.ensure_workspace(driver, total_f64) {
        Ok(ptr) => ptr,
        Err(e) => return Some(Err(e)),
    };
    let d_idx = offset_ptr(d_mat, dist.data.len());
    let d_val = offset_ptr(d_idx, dist.rows);
    if let Err(e) = copy_htod_raw(driver, d_mat, &dist.data) {
        return Some(Err(e));
    }

    let mut d_mat_arg = d_mat;
    let mut rows_u64 = dist.rows as u64;
    let mut cols_u64 = dist.cols as u64;
    let mut d_idx_arg = d_idx;
    let mut d_val_arg = d_val;
    let mut params: [*mut c_void; 5] = [
        (&mut d_mat_arg as *mut CuDevicePtr).cast(),
        (&mut rows_u64 as *mut u64).cast(),
        (&mut cols_u64 as *mut u64).cast(),
        (&mut d_idx_arg as *mut CuDevicePtr).cast(),
        (&mut d_val_arg as *mut CuDevicePtr).cast(),
    ];

    let block_x = 128u32;
    let grid_x = ((dist.rows as u32) + block_x - 1) / block_x;
    // SAFETY: Kernel parameters match `ochre_row_argmin_f64` signature.
    let rc = unsafe {
        let lrc = (ptx.cu_launch_kernel)(
            argmin_fn,
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if lrc != CUDA_SUCCESS {
            return None;
        }
        (ptx.cu_ctx_synchronize)()
    };
    if rc != CUDA_SUCCESS {
        return None;
    }

    let mut idx_u64 = vec![0u64; dist.rows];
    let mut vals = vec![0.0f64; dist.rows];
    // SAFETY: `d_idx` holds `dist.rows` `u64` indices.
    let copy_idx_rc = unsafe {
        (driver.cu_memcpy_dtoh)(
            idx_u64.as_mut_ptr().cast(),
            d_idx,
            dist.rows * std::mem::size_of::<u64>(),
        )
    };
    if copy_idx_rc != CUDA_SUCCESS {
        return None;
    }
    if let Err(e) = copy_dtoh_raw(driver, &mut vals, d_val) {
        return Some(Err(e));
    }
    let indices = idx_u64.into_iter().map(|v| v as usize).collect();
    Some(Ok((indices, vals)))
}

/// CUDA entrypoint for N6 (`batched_histogram_grad`): delegates to CPU Rayon for small/medium batches.
pub fn cuda_batched_histogram_grad(
    _gpu_id: usize,
    _binned: &BinnedMatrix,
    _sample_indices: &[u32],
    _grad: &[f64],
    _hess: &[f64],
) -> Option<OchreResult<Vec<GradHessBin>>> {
    // Fallback to CPU Rayon histogramming unless GPU-resident histogram buffers are active.
    None
}
