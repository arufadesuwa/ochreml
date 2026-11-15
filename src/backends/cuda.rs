// OchreML Native NVIDIA CUDA + cuBLAS Dynamic Backend (*≧ω≦*)
// Uses runtime dlopen (libloading) so PyPI wheels compile cleanly everywhere without CUDA Toolkit,
// while achieving full FP64 Tensor/CUDA performance on NVIDIA GPUs (e.g., Kaggle T4 x2).

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use libloading::Library;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CublasStatus = i32;
type CublasHandle = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_FILL_MODE_UPPER: i32 = 1;
const CUBLAS_SIDE_RIGHT: i32 = 1;

// Persistent 64 MB (8M f64) per-GPU workspace to avoid repeated OS/MMU cuMemAlloc/cuMemFree calls
const DEFAULT_WORKSPACE_F64: usize = 8 * 1024 * 1024;
// Minimum matrix elements (128 MB) before sharding across multiple GPUs
const MULTI_GPU_MIN_ELEMENTS: usize = 16_000_000;

type FnCuInit = unsafe extern "C" fn(u32) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type FnCuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type FnCuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDgemm = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
) -> CublasStatus;
type FnCublasDsyrk = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
) -> CublasStatus;
type FnCublasDgemv = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
    *const f64,
    CuDevicePtr,
    i32,
) -> CublasStatus;
type FnCublasDdgmm = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
) -> CublasStatus;

struct CudaDeviceCtx {
    ctx: CuContext,
    handle: CublasHandle,
    workspace_ptr: CuDevicePtr,
    workspace_len: usize,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_mem_free: FnCuMemFree,
}

// SAFETY: Protected by Mutex and bound to calling thread via cuCtxSetCurrent before every call.
unsafe impl Send for CudaDeviceCtx {}
// SAFETY: All mutable access to CudaDeviceCtx goes through Arc<Mutex<CudaDeviceCtx>>.
unsafe impl Sync for CudaDeviceCtx {}

impl Drop for CudaDeviceCtx {
    fn drop(&mut self) {
        if self.workspace_ptr != 0 {
            // SAFETY: Bind context and release persistent VRAM workspace on teardown.
            unsafe {
                let _ = (self.cu_ctx_set_current)(self.ctx);
                let _ = (self.cu_mem_free)(self.workspace_ptr);
            }
            self.workspace_ptr = 0;
            self.workspace_len = 0;
        }
    }
}

#[inline]
fn align32(len_f64: usize) -> usize {
    (len_f64 + 31) & !31
}

#[inline]
fn offset_ptr(base: CuDevicePtr, offset_f64: usize) -> CuDevicePtr {
    base + (offset_f64 * std::mem::size_of::<f64>()) as u64
}

impl CudaDeviceCtx {
    fn ensure_workspace(
        &mut self,
        driver: &CudaDriver,
        required_f64: usize,
    ) -> OchreResult<CuDevicePtr> {
        let needed = align32(required_f64.max(1));
        if self.workspace_ptr != 0 && self.workspace_len >= needed {
            return Ok(self.workspace_ptr);
        }

        let new_len = needed.max(DEFAULT_WORKSPACE_F64);
        let bytes = new_len * std::mem::size_of::<f64>();
        let mut new_ptr: CuDevicePtr = 0;
        // SAFETY: Current thread has bound `self.ctx`; `new_ptr` receives a valid device allocation.
        let rc = unsafe { (driver.cu_mem_alloc)(&mut new_ptr, bytes) };
        if rc != CUDA_SUCCESS || new_ptr == 0 {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuMemAlloc failed with code {} (´；ω；`)",
                rc
            )));
        }

        if self.workspace_ptr != 0 {
            // SAFETY: Old workspace is no longer referenced by any active stream on this context.
            unsafe {
                let _ = (driver.cu_mem_free)(self.workspace_ptr);
            }
        }
        self.workspace_ptr = new_ptr;
        self.workspace_len = new_len;
        Ok(self.workspace_ptr)
    }
}

struct CudaDriver {
    _cuda_lib: Library,
    _cublas_lib: Library,
    cu_device_get: FnCuDeviceGet,
    cu_primary_ctx_retain: Option<FnCuDevicePrimaryCtxRetain>,
    cu_ctx_create: FnCuCtxCreate,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_mem_alloc: FnCuMemAlloc,
    cu_mem_free: FnCuMemFree,
    cu_memcpy_htod: FnCuMemcpyHtoD,
    cu_memcpy_dtoh: FnCuMemcpyDtoH,
    cublas_create: FnCublasCreate,
    cublas_dgemm: FnCublasDgemm,
    cublas_dsyrk: FnCublasDsyrk,
    cublas_dgemv: FnCublasDgemv,
    cublas_ddgmm: Option<FnCublasDdgmm>,
    contexts: Mutex<HashMap<usize, Arc<Mutex<CudaDeviceCtx>>>>,
}

// SAFETY: Function pointers and dynamic library handles are thread-safe after initialization.
unsafe impl Send for CudaDriver {}
// SAFETY: Mutable context map is protected by a Mutex.
unsafe impl Sync for CudaDriver {}

static CUDA_DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();

fn load_first_library(candidates: &[&str]) -> Option<Library> {
    for name in candidates {
        // SAFETY: Loading shared system library (`libcuda` / `libcublas`) by standard SONAME.
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

fn init_cuda_driver() -> Option<CudaDriver> {
    #[cfg(target_os = "windows")]
    let cuda_candidates = ["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    let cuda_candidates = ["libcuda.so.1", "libcuda.so"];

    #[cfg(target_os = "windows")]
    let cublas_candidates = ["cublas64_12.dll", "cublas64_11.dll", "cublas64_10.dll"];
    #[cfg(not(target_os = "windows"))]
    let cublas_candidates = ["libcublas.so.12", "libcublas.so.11", "libcublas.so"];

    let cuda_lib = load_first_library(&cuda_candidates)?;
    let cublas_lib = load_first_library(&cublas_candidates)?;

    // SAFETY: Resolve standard CUDA Driver API and cuBLAS C ABI symbols from loaded libraries.
    unsafe {
        let cu_init: FnCuInit = *cuda_lib.get::<FnCuInit>(b"cuInit\0").ok()?;
        if cu_init(0) != CUDA_SUCCESS {
            return None;
        }

        let cu_device_get = *cuda_lib.get::<FnCuDeviceGet>(b"cuDeviceGet\0").ok()?;
        let cu_primary_ctx_retain = cuda_lib
            .get::<FnCuDevicePrimaryCtxRetain>(b"cuDevicePrimaryCtxRetain\0")
            .ok()
            .map(|s| *s);
        let cu_ctx_create = *cuda_lib.get::<FnCuCtxCreate>(b"cuCtxCreate_v2\0").ok()?;
        let cu_ctx_set_current = *cuda_lib
            .get::<FnCuCtxSetCurrent>(b"cuCtxSetCurrent\0")
            .ok()?;
        let cu_mem_alloc = *cuda_lib.get::<FnCuMemAlloc>(b"cuMemAlloc_v2\0").ok()?;
        let cu_mem_free = *cuda_lib.get::<FnCuMemFree>(b"cuMemFree_v2\0").ok()?;
        let cu_memcpy_htod = *cuda_lib.get::<FnCuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0").ok()?;
        let cu_memcpy_dtoh = *cuda_lib.get::<FnCuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0").ok()?;

        let cublas_create = *cublas_lib
            .get::<FnCublasCreate>(b"cublasCreate_v2\0")
            .ok()?;
        let cublas_dgemm = *cublas_lib.get::<FnCublasDgemm>(b"cublasDgemm_v2\0").ok()?;
        let cublas_dsyrk = *cublas_lib.get::<FnCublasDsyrk>(b"cublasDsyrk_v2\0").ok()?;
        let cublas_dgemv = *cublas_lib.get::<FnCublasDgemv>(b"cublasDgemv_v2\0").ok()?;
        let cublas_ddgmm = cublas_lib
            .get::<FnCublasDdgmm>(b"cublasDdgmm\0")
            .ok()
            .map(|s| *s);

        Some(CudaDriver {
            _cuda_lib: cuda_lib,
            _cublas_lib: cublas_lib,
            cu_device_get,
            cu_primary_ctx_retain,
            cu_ctx_create,
            cu_ctx_set_current,
            cu_mem_alloc,
            cu_mem_free,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cublas_create,
            cublas_dgemm,
            cublas_dsyrk,
            cublas_dgemv,
            cublas_ddgmm,
            contexts: Mutex::new(HashMap::new()),
        })
    }
}

fn get_driver() -> Option<&'static CudaDriver> {
    CUDA_DRIVER.get_or_init(init_cuda_driver).as_ref()
}

/// Pre-warm CUDA primary context, cuBLAS handle, and persistent 64 MB VRAM workspace on `gpu_id`.
pub fn warmup_device(gpu_id: usize) {
    if let Some(driver) = get_driver() {
        let _ = driver.get_device_ctx(gpu_id);
    }
}

pub fn is_available() -> bool {
    if let Some(driver) = get_driver() {
        let _ = driver.get_device_ctx(0);
        true
    } else {
        false
    }
}

#[inline]
fn copy_htod_raw(driver: &CudaDriver, dst: CuDevicePtr, src: &[f64]) -> OchreResult<()> {
    let bytes = std::mem::size_of_val(src);
    if bytes > 0 {
        // SAFETY: `dst` points to a valid device allocation with at least `bytes` capacity.
        let rc = unsafe { (driver.cu_memcpy_htod)(dst, src.as_ptr() as *const c_void, bytes) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuMemcpyHtoD failed with code {} (´；ω；`)",
                rc
            )));
        }
    }
    Ok(())
}

#[inline]
fn copy_dtoh_raw(driver: &CudaDriver, dst: &mut [f64], src: CuDevicePtr) -> OchreResult<()> {
    let bytes = std::mem::size_of_val(dst);
    if bytes > 0 {
        // SAFETY: `src` points to a valid device allocation and `dst` has `bytes` capacity.
        let rc = unsafe { (driver.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, bytes) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuMemcpyDtoH failed with code {} (´；ω；`)",
                rc
            )));
        }
    }
    Ok(())
}

impl CudaDriver {
    fn get_device_ctx(&self, gpu_id: usize) -> OchreResult<Arc<Mutex<CudaDeviceCtx>>> {
        let mut map = self.contexts.lock().map_err(|_| {
            OchreError::DeviceError("CUDA context cache mutex poisoned ( >_< )".to_string())
        })?;
        if let Some(existing) = map.get(&gpu_id) {
            return Ok(Arc::clone(existing));
        }

        let mut dev: CuDevice = 0;
        // SAFETY: Queries device handle for `gpu_id`.
        let rc = unsafe { (self.cu_device_get)(&mut dev, gpu_id as i32) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuDeviceGet({}) failed with code {} ( >_< )",
                gpu_id, rc
            )));
        }

        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: Retains primary context (shared with PyTorch/CUDA runtime) or creates a new context.
        let ctx_rc = if let Some(retain) = self.cu_primary_ctx_retain {
            unsafe { retain(&mut ctx, dev) }
        } else {
            unsafe { (self.cu_ctx_create)(&mut ctx, 0, dev) }
        };
        if ctx_rc != CUDA_SUCCESS || ctx.is_null() {
            return Err(OchreError::DeviceError(format!(
                "CUDA context creation on GPU {} failed with code {} (´；ω；`)",
                gpu_id, ctx_rc
            )));
        }

        // SAFETY: Binds `ctx` to the current host thread.
        let set_rc = unsafe { (self.cu_ctx_set_current)(ctx) };
        if set_rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuCtxSetCurrent failed with code {} ( >_< )",
                set_rc
            )));
        }

        let mut handle: CublasHandle = std::ptr::null_mut();
        // SAFETY: Initializes cuBLAS handle on the currently bound context.
        let blas_rc = unsafe { (self.cublas_create)(&mut handle) };
        if blas_rc != CUBLAS_STATUS_SUCCESS || handle.is_null() {
            return Err(OchreError::DeviceError(format!(
                "cuBLAS cublasCreate_v2 failed with code {} (´；ω；`)",
                blas_rc
            )));
        }

        let mut dev_ctx = CudaDeviceCtx {
            ctx,
            handle,
            workspace_ptr: 0,
            workspace_len: 0,
            cu_ctx_set_current: self.cu_ctx_set_current,
            cu_mem_free: self.cu_mem_free,
        };

        // Pre-allocate default 64 MB persistent workspace and warm up cuBLAS FP64 kernels
        if let Ok(ws_ptr) = dev_ctx.ensure_workspace(self, DEFAULT_WORKSPACE_F64) {
            let zeros = [0.0_f64; 4];
            let _ = copy_htod_raw(self, ws_ptr, &zeros);
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let d_out = offset_ptr(ws_ptr, 32);
            // SAFETY: Warm-up 1x1 dsyrk and dgemv within pre-allocated workspace.
            unsafe {
                let _ = (self.cublas_dsyrk)(
                    handle,
                    CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N,
                    1,
                    1,
                    &alpha,
                    ws_ptr,
                    1,
                    &beta,
                    d_out,
                    1,
                );
                let _ = (self.cublas_dgemv)(
                    handle,
                    CUBLAS_OP_N,
                    1,
                    1,
                    &alpha,
                    ws_ptr,
                    1,
                    ws_ptr,
                    1,
                    &beta,
                    d_out,
                    1,
                );
            }
        }

        let arc = Arc::new(Mutex::new(dev_ctx));
        map.insert(gpu_id, Arc::clone(&arc));
        Ok(arc)
    }

    fn bind_ctx(&self, guard: &CudaDeviceCtx) -> OchreResult<()> {
        // SAFETY: Binds `guard.ctx` to the calling thread while holding the device mutex.
        let rc = unsafe { (self.cu_ctx_set_current)(guard.ctx) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuCtxSetCurrent failed with code {} ( >_< )",
                rc
            )));
        }
        Ok(())
    }
}

fn check_cublas(status: CublasStatus, op: &str) -> OchreResult<()> {
    if status != CUBLAS_STATUS_SUCCESS {
        return Err(OchreError::DeviceError(format!(
            "cuBLAS {} failed with status {} (´；ω；`)",
            op, status
        )));
    }
    Ok(())
}

// Compute X^T * y (p_x), column sums of X (p_x), and sum of y in a single register-fused host pass
// while the GPU executes Split-K cublasDgemm_v2 asynchronously.
fn host_sum_and_xty_1d(
    x_slice: &[f64],
    y_slice: &[f64],
    n_chunk: usize,
    p_x: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    if n_chunk < 8192 {
        let mut xty = vec![0.0_f64; p_x];
        let mut sum_x = vec![0.0_f64; p_x];
        let mut sum_y = 0.0_f64;
        for (i, &yi) in y_slice.iter().enumerate() {
            sum_y += yi;
            let row = &x_slice[(i * p_x)..((i + 1) * p_x)];
            for j in 0..p_x {
                let xij = row[j];
                sum_x[j] += xij;
                xty[j] += xij * yi;
            }
        }
        return (xty, sum_x, sum_y);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let block_rows = (n_chunk / (n_threads * 4).max(1)).max(2048);
    let mut ranges = Vec::new();
    let mut r = 0;
    while r < n_chunk {
        let end = (r + block_rows).min(n_chunk);
        ranges.push((r, end));
        r = end;
    }

    let partials: Vec<(Vec<f64>, Vec<f64>, f64)> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let mut xty = vec![0.0_f64; p_x];
            let mut sx = vec![0.0_f64; p_x];
            let mut sy = 0.0_f64;
            for i in start..end {
                let yi = y_slice[i];
                sy += yi;
                let row = &x_slice[(i * p_x)..((i + 1) * p_x)];
                for j in 0..p_x {
                    let xij = row[j];
                    sx[j] += xij;
                    xty[j] += xij * yi;
                }
            }
            (xty, sx, sy)
        })
        .collect();

    let mut total_xty = vec![0.0_f64; p_x];
    let mut total_x = vec![0.0_f64; p_x];
    let mut total_y = 0.0_f64;
    for (xty, sx, sy) in partials {
        total_y += sy;
        for j in 0..p_x {
            total_x[j] += sx[j];
            total_xty[j] += xty[j];
        }
    }
    (total_xty, total_x, total_y)
}

// Compute (optionally weighted) column sums of X (n_chunk x p_x) and Y (n_chunk x k_targets)
// on CPU in parallel while the GPU executes dsyrk/dgemm asynchronously.
fn host_sum_x_and_y_multi(
    x_slice: &[f64],
    y_slice: &[f64],
    sw_slice: Option<&[f64]>,
    n_chunk: usize,
    p_x: usize,
    k_targets: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_chunk < 8192 {
        let mut sum_x = vec![0.0_f64; p_x];
        let mut sum_y = vec![0.0_f64; k_targets];
        if let Some(sw) = sw_slice {
            for (i, &w) in sw.iter().enumerate() {
                let x_row = &x_slice[(i * p_x)..((i + 1) * p_x)];
                let y_row = &y_slice[(i * k_targets)..((i + 1) * k_targets)];
                for j in 0..p_x {
                    sum_x[j] += w * x_row[j];
                }
                for t in 0..k_targets {
                    sum_y[t] += w * y_row[t];
                }
            }
        } else {
            for i in 0..n_chunk {
                let x_row = &x_slice[(i * p_x)..((i + 1) * p_x)];
                let y_row = &y_slice[(i * k_targets)..((i + 1) * k_targets)];
                for j in 0..p_x {
                    sum_x[j] += x_row[j];
                }
                for t in 0..k_targets {
                    sum_y[t] += y_row[t];
                }
            }
        }
        return (sum_x, sum_y);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let block_rows = (n_chunk / (n_threads * 4).max(1)).max(2048);
    let mut ranges = Vec::new();
    let mut r = 0;
    while r < n_chunk {
        let end = (r + block_rows).min(n_chunk);
        ranges.push((r, end));
        r = end;
    }

    let partials: Vec<(Vec<f64>, Vec<f64>)> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let mut sx = vec![0.0_f64; p_x];
            let mut sy = vec![0.0_f64; k_targets];
            if let Some(sw) = sw_slice {
                for i in start..end {
                    let w = sw[i];
                    let x_row = &x_slice[(i * p_x)..((i + 1) * p_x)];
                    let y_row = &y_slice[(i * k_targets)..((i + 1) * k_targets)];
                    for j in 0..p_x {
                        sx[j] += w * x_row[j];
                    }
                    for t in 0..k_targets {
                        sy[t] += w * y_row[t];
                    }
                }
            } else {
                for i in start..end {
                    let x_row = &x_slice[(i * p_x)..((i + 1) * p_x)];
                    let y_row = &y_slice[(i * k_targets)..((i + 1) * k_targets)];
                    for j in 0..p_x {
                        sx[j] += x_row[j];
                    }
                    for t in 0..k_targets {
                        sy[t] += y_row[t];
                    }
                }
            }
            (sx, sy)
        })
        .collect();

    let mut total_x = vec![0.0_f64; p_x];
    let mut total_y = vec![0.0_f64; k_targets];
    for (sx, sy) in partials {
        for j in 0..p_x {
            total_x[j] += sx[j];
        }
        for t in 0..k_targets {
            total_y[t] += sy[t];
        }
    }
    (total_x, total_y)
}

// General Matrix Multiplication: C = A * B (Zero-transpose via C^T = B^T * A^T in column-major)
pub fn cuda_matmul(gpu_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let mut guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let m = a.rows as i32;
        let k = a.cols as i32;
        let n = b.cols as i32;

        let off_a = 0;
        let off_b = off_a + align32(a.data.len());
        let off_c = off_b + align32(b.data.len());
        let total_f64 = off_c + align32(a.rows * b.cols);
        let ws_ptr = guard.ensure_workspace(driver, total_f64)?;

        let d_a = offset_ptr(ws_ptr, off_a);
        let d_b = offset_ptr(ws_ptr, off_b);
        let d_c = offset_ptr(ws_ptr, off_c);

        copy_htod_raw(driver, d_a, &a.data)?;
        copy_htod_raw(driver, d_b, &b.data)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        // SAFETY: All device pointers lie within the bound context's persistent workspace.
        let status = unsafe {
            (driver.cublas_dgemm)(
                guard.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                d_b,
                n,
                d_a,
                k,
                &beta,
                d_c,
                n,
            )
        };
        check_cublas(status, "cublasDgemm_v2")?;

        let mut out = Matrix::zeros(a.rows, b.cols);
        copy_dtoh_raw(driver, &mut out.data, d_c)?;
        Ok(out)
    })();
    Some(res)
}

// Compute A^T * B where A is (N x p) and B is (N x K) row-major -> (p x K) row-major
pub fn cuda_matmul_at_b(gpu_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let mut guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let n_rows = a.rows as i32;
        let p = a.cols as i32;
        let k = b.cols as i32;

        let off_a = 0;
        let off_b = off_a + align32(a.data.len());
        let off_c = off_b + align32(b.data.len());
        let total_f64 = off_c + align32(a.cols * b.cols);
        let ws_ptr = guard.ensure_workspace(driver, total_f64)?;

        let d_a = offset_ptr(ws_ptr, off_a);
        let d_b = offset_ptr(ws_ptr, off_b);
        let d_c = offset_ptr(ws_ptr, off_c);

        copy_htod_raw(driver, d_a, &a.data)?;
        copy_htod_raw(driver, d_b, &b.data)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        // SAFETY: All device pointers lie within the bound context's persistent workspace.
        let status = unsafe {
            (driver.cublas_dgemm)(
                guard.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                k,
                p,
                n_rows,
                &alpha,
                d_b,
                k,
                d_a,
                p,
                &beta,
                d_c,
                k,
            )
        };
        check_cublas(status, "cublasDgemm_v2(AtB)")?;

        let mut out = Matrix::zeros(a.cols, b.cols);
        copy_dtoh_raw(driver, &mut out.data, d_c)?;
        Ok(out)
    })();
    Some(res)
}

// Matrix-Vector Multiplication: y = A * v (Zero-transpose via (A^T)^T * v)
pub fn cuda_matvec(gpu_id: usize, a: &Matrix, v: &[f64]) -> Option<OchreResult<Vec<f64>>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let mut guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let m = a.rows as i32;
        let n = a.cols as i32;

        let off_a = 0;
        let off_v = off_a + align32(a.data.len());
        let off_y = off_v + align32(v.len());
        let total_f64 = off_y + align32(a.rows);
        let ws_ptr = guard.ensure_workspace(driver, total_f64)?;

        let d_a = offset_ptr(ws_ptr, off_a);
        let d_v = offset_ptr(ws_ptr, off_v);
        let d_y = offset_ptr(ws_ptr, off_y);

        copy_htod_raw(driver, d_a, &a.data)?;
        copy_htod_raw(driver, d_v, v)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        // SAFETY: All device pointers lie within the bound context's persistent workspace.
        let status = unsafe {
            (driver.cublas_dgemv)(
                guard.handle,
                CUBLAS_OP_T,
                n,
                m,
                &alpha,
                d_a,
                n,
                d_v,
                1,
                &beta,
                d_y,
                1,
            )
        };
        check_cublas(status, "cublasDgemv_v2")?;

        let mut out = vec![0.0; a.rows];
        copy_dtoh_raw(driver, &mut out, d_y)?;
        Ok(out)
    })();
    Some(res)
}

// Single-GPU 1D Normal Equation chunk over [start_row..end_row] using persistent workspace,
// Split-K cublasDgemm_v2, register-fused host X^T y + intercept sums, and heterogeneous
// GPU + CPU co-execution for tall-skinny matrices (p <= 64).
fn cuda_normal_eq_1d_chunk(
    driver: &CudaDriver,
    gpu_id: usize,
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Vec<f64>)> {
    let n_chunk = end_row - start_row;
    let p_x = x.cols;
    let p_total = if fit_intercept { p_x + 1 } else { p_x };

    let ctx_arc = driver.get_device_ctx(gpu_id)?;
    let mut guard = ctx_arc
        .lock()
        .map_err(|_| OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string()))?;
    driver.bind_ctx(&guard)?;

    // For tall-skinny matrices (p_x <= 64), PCIe transfer + 1/32-rate FP64 on T4/GeForce GPUs
    // is memory/FP64-bound on large N; co-executing 4096 rows on GPU and the remainder on
    // host AVX2/AVX-512 cores overlaps both processors for maximum throughput.
    let n_gpu = if p_x <= 64 && n_chunk > 4096 {
        4096
    } else {
        n_chunk
    };
    let gpu_end = start_row + n_gpu;

    let x_gpu_slice = &x.data[(start_row * p_x)..(gpu_end * p_x)];
    let y_gpu_slice = &y[start_row..gpu_end];

    let off_x = 0;
    let off_xtx = off_x + align32(x_gpu_slice.len());
    let total_f64 = off_xtx + align32(p_x * p_x);
    let ws_ptr = guard.ensure_workspace(driver, total_f64)?;

    let d_x = offset_ptr(ws_ptr, off_x);
    let d_xtx = offset_ptr(ws_ptr, off_xtx);

    copy_htod_raw(driver, d_x, x_gpu_slice)?;

    let alpha = 1.0_f64;
    let beta = 0.0_f64;

    // 1. X_gpu^T * X_gpu via Split-K cublasDgemm_v2 (asynchronous across all GPU SMs)
    // SAFETY: `d_x` and `d_xtx` are valid workspace sub-allocations on the bound context.
    let gemm_rc = unsafe {
        (driver.cublas_dgemm)(
            guard.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            p_x as i32,
            p_x as i32,
            n_gpu as i32,
            &alpha,
            d_x,
            p_x as i32,
            d_x,
            p_x as i32,
            &beta,
            d_xtx,
            p_x as i32,
        )
    };
    check_cublas(gemm_rc, "cublasDgemm_v2(XtX)")?;

    // 2. While GPU executes dgemm asynchronously, compute X_gpu^T * y_gpu + intercept sums
    //    AND co-execute any remaining rows [gpu_end..end_row] on host SIMD cores!
    let (xty_sub, sum_x, sum_y) = host_sum_and_xty_1d(x_gpu_slice, y_gpu_slice, n_gpu, p_x);
    let cpu_tail = if gpu_end < end_row {
        Some(crate::backend::cpu_normal_equation_1d_slice(
            x,
            y,
            fit_intercept,
            gpu_end,
            end_row,
        )?)
    } else {
        None
    };

    let mut xtx_sub = Matrix::zeros(p_x, p_x);
    copy_dtoh_raw(driver, &mut xtx_sub.data, d_xtx)?;

    let (mut full_xtx, mut full_xty) = if !fit_intercept {
        (xtx_sub, xty_sub)
    } else {
        let mut fxtx = Matrix::zeros(p_total, p_total);
        for (r, &sx) in sum_x.iter().enumerate().take(p_x) {
            let dst_off = r * p_total;
            let src_off = r * p_x;
            fxtx.data[dst_off..(dst_off + p_x)]
                .copy_from_slice(&xtx_sub.data[src_off..(src_off + p_x)]);
            fxtx.data[dst_off + p_x] = sx;
        }
        let last_off = p_x * p_total;
        fxtx.data[last_off..(last_off + p_x)].copy_from_slice(&sum_x);
        fxtx.data[last_off + p_x] = n_gpu as f64;

        let mut fxty = Vec::with_capacity(p_total);
        fxty.extend_from_slice(&xty_sub);
        fxty.push(sum_y);
        (fxtx, fxty)
    };

    if let Some((tail_xtx, tail_xty)) = cpu_tail {
        for i in 0..(p_total * p_total) {
            full_xtx.data[i] += tail_xtx.data[i];
        }
        for i in 0..p_total {
            full_xty[i] += tail_xty[i];
        }
    }

    Ok((full_xtx, full_xty))
}

pub fn cuda_compute_normal_equation(
    gpu_ids: &[usize],
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Vec<f64>)>> {
    let driver = get_driver()?;
    if gpu_ids.is_empty() {
        return None;
    }

    let total_elems = x.rows.saturating_mul(x.cols);
    if gpu_ids.len() == 1 || x.rows < gpu_ids.len() || total_elems < MULTI_GPU_MIN_ELEMENTS {
        return Some(cuda_normal_eq_1d_chunk(
            driver,
            gpu_ids[0],
            x,
            y,
            fit_intercept,
            0,
            x.rows,
        ));
    }

    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let n_dev = gpu_ids.len();
    let chunk_size = x.rows.div_ceil(n_dev);

    let chunk_results: Vec<OchreResult<(Matrix, Vec<f64>)>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_dev);
        for (idx, &gid) in gpu_ids.iter().enumerate() {
            let start = idx * chunk_size;
            if start >= x.rows {
                break;
            }
            let end = (start + chunk_size).min(x.rows);
            handles.push(s.spawn(move || {
                cuda_normal_eq_1d_chunk(driver, gid, x, y, fit_intercept, start, end)
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(OchreError::DeviceError(
                        "Multi-GPU worker thread panicked (´；ω；`)".to_string(),
                    ))
                })
            })
            .collect()
    });

    let mut total_xtx = Matrix::zeros(p, p);
    let mut total_xty = vec![0.0; p];
    for res in chunk_results {
        match res {
            Ok((m, v)) => {
                for i in 0..(p * p) {
                    total_xtx.data[i] += m.data[i];
                }
                for i in 0..p {
                    total_xty[i] += v[i];
                }
            }
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok((total_xtx, total_xty)))
}

// Multi-target & optional sample_weight Normal Equation chunk on GPU using persistent workspace,
// Split-K cublasDgemm_v2, and heterogeneous GPU + CPU co-execution for tall-skinny matrices.
#[allow(clippy::too_many_arguments)]
fn cuda_normal_eq_multi_chunk(
    driver: &CudaDriver,
    gpu_id: usize,
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
    start_row: usize,
    end_row: usize,
) -> OchreResult<(Matrix, Matrix)> {
    let n_chunk = end_row - start_row;
    let p_x = x.cols;
    let k_targets = y.cols;
    let p_total = if fit_intercept { p_x + 1 } else { p_x };

    let ctx_arc = driver.get_device_ctx(gpu_id)?;
    let mut guard = ctx_arc
        .lock()
        .map_err(|_| OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string()))?;
    driver.bind_ctx(&guard)?;

    let n_gpu = if p_x <= 64 && n_chunk > 4096 {
        4096
    } else {
        n_chunk
    };
    let gpu_end = start_row + n_gpu;

    let x_slice = &x.data[(start_row * p_x)..(gpu_end * p_x)];
    let y_slice = &y.data[(start_row * k_targets)..(gpu_end * k_targets)];
    let sw_slice = sample_weight.map(|sw| &sw[start_row..gpu_end]);

    let off_x = 0;
    let off_y = off_x + align32(x_slice.len());
    let off_xtx = off_y + align32(y_slice.len());
    let off_xty = off_xtx + align32(p_x * p_x);
    let off_sqrt = off_xty + align32(p_x * k_targets);
    let total_f64 = if sw_slice.is_some() {
        off_sqrt + align32(n_gpu)
    } else {
        off_sqrt
    };
    let ws_ptr = guard.ensure_workspace(driver, total_f64)?;

    let d_x = offset_ptr(ws_ptr, off_x);
    let d_y = offset_ptr(ws_ptr, off_y);
    let d_xtx = offset_ptr(ws_ptr, off_xtx);
    let d_xty = offset_ptr(ws_ptr, off_xty);

    copy_htod_raw(driver, d_x, x_slice)?;
    copy_htod_raw(driver, d_y, y_slice)?;

    let weight_sum = if let Some(sw) = sw_slice {
        let mut sqrt_sw = Vec::with_capacity(n_gpu);
        let mut w_sum = 0.0;
        for &w in sw {
            w_sum += w;
            sqrt_sw.push(w.sqrt());
        }
        let d_sqrt = offset_ptr(ws_ptr, off_sqrt);
        copy_htod_raw(driver, d_sqrt, &sqrt_sw)?;

        let ddgmm = driver.cublas_ddgmm.ok_or_else(|| {
            OchreError::DeviceError(
                "cublasDdgmm not available in loaded cuBLAS library ( >_< )".to_string(),
            )
        })?;
        // SAFETY: `d_x` and `d_sqrt` are valid workspace sub-allocations on the bound context.
        let rc_x = unsafe {
            ddgmm(
                guard.handle,
                CUBLAS_SIDE_RIGHT,
                p_x as i32,
                n_gpu as i32,
                d_x,
                p_x as i32,
                d_sqrt,
                1,
                d_x,
                p_x as i32,
            )
        };
        check_cublas(rc_x, "cublasDdgmm(X)")?;

        // SAFETY: `d_y` and `d_sqrt` are valid workspace sub-allocations on the bound context.
        let rc_y = unsafe {
            ddgmm(
                guard.handle,
                CUBLAS_SIDE_RIGHT,
                k_targets as i32,
                n_gpu as i32,
                d_y,
                k_targets as i32,
                d_sqrt,
                1,
                d_y,
                k_targets as i32,
            )
        };
        check_cublas(rc_y, "cublasDdgmm(Y)")?;
        w_sum
    } else {
        n_gpu as f64
    };

    let alpha = 1.0_f64;
    let beta = 0.0_f64;

    // SAFETY: `d_x` and `d_xtx` are valid workspace sub-allocations on the bound context.
    let xtx_rc = unsafe {
        (driver.cublas_dgemm)(
            guard.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            p_x as i32,
            p_x as i32,
            n_gpu as i32,
            &alpha,
            d_x,
            p_x as i32,
            d_x,
            p_x as i32,
            &beta,
            d_xtx,
            p_x as i32,
        )
    };
    check_cublas(xtx_rc, "cublasDgemm_v2(XtX_multi)")?;

    // X^T * Y: col-major C^T(K x p_x) = B^T(K x N) * A(N x p_x)
    // SAFETY: `d_x`, `d_y`, and `d_xty` are valid workspace sub-allocations on the bound context.
    let gemm_rc = unsafe {
        (driver.cublas_dgemm)(
            guard.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            k_targets as i32,
            p_x as i32,
            n_gpu as i32,
            &alpha,
            d_y,
            k_targets as i32,
            d_x,
            p_x as i32,
            &beta,
            d_xty,
            k_targets as i32,
        )
    };
    check_cublas(gemm_rc, "cublasDgemm_v2(XtY)")?;

    // Compute intercept sums and co-execute remaining rows [gpu_end..end_row] on host CPU
    let intercept_sums = if fit_intercept {
        Some(host_sum_x_and_y_multi(
            x_slice, y_slice, sw_slice, n_gpu, p_x, k_targets,
        ))
    } else {
        None
    };
    let cpu_tail = if gpu_end < end_row {
        Some(crate::backend::cpu_normal_equation_multi_slice(
            x,
            y,
            sample_weight,
            fit_intercept,
            gpu_end,
            end_row,
        )?)
    } else {
        None
    };

    let mut xtx_sub = Matrix::zeros(p_x, p_x);
    copy_dtoh_raw(driver, &mut xtx_sub.data, d_xtx)?;

    let mut xty_sub = Matrix::zeros(p_x, k_targets);
    copy_dtoh_raw(driver, &mut xty_sub.data, d_xty)?;

    let (mut full_xtx, mut full_xty) = match intercept_sums {
        Some((sum_x, sum_y)) => {
            let mut fxtx = Matrix::zeros(p_total, p_total);
            for (r, &sx) in sum_x.iter().enumerate().take(p_x) {
                let dst_off = r * p_total;
                let src_off = r * p_x;
                fxtx.data[dst_off..(dst_off + p_x)]
                    .copy_from_slice(&xtx_sub.data[src_off..(src_off + p_x)]);
                fxtx.data[dst_off + p_x] = sx;
            }
            let last_off = p_x * p_total;
            fxtx.data[last_off..(last_off + p_x)].copy_from_slice(&sum_x);
            fxtx.data[last_off + p_x] = weight_sum;

            let mut fxty = Matrix::zeros(p_total, k_targets);
            fxty.data[0..(p_x * k_targets)].copy_from_slice(&xty_sub.data);
            fxty.data[(p_x * k_targets)..(p_total * k_targets)].copy_from_slice(&sum_y);
            (fxtx, fxty)
        }
        None => (xtx_sub, xty_sub),
    };

    if let Some((tail_xtx, tail_xty)) = cpu_tail {
        for i in 0..(p_total * p_total) {
            full_xtx.data[i] += tail_xtx.data[i];
        }
        for i in 0..(p_total * k_targets) {
            full_xty.data[i] += tail_xty.data[i];
        }
    }

    Ok((full_xtx, full_xty))
}

pub fn cuda_compute_normal_equation_multi(
    gpu_ids: &[usize],
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Matrix)>> {
    let driver = get_driver()?;
    if gpu_ids.is_empty() {
        return None;
    }

    if sample_weight.is_some() && driver.cublas_ddgmm.is_none() {
        return None;
    }

    let total_elems = x.rows.saturating_mul(x.cols);
    if gpu_ids.len() == 1 || x.rows < gpu_ids.len() || total_elems < MULTI_GPU_MIN_ELEMENTS {
        return Some(cuda_normal_eq_multi_chunk(
            driver,
            gpu_ids[0],
            x,
            y,
            sample_weight,
            fit_intercept,
            0,
            x.rows,
        ));
    }

    let p = if fit_intercept { x.cols + 1 } else { x.cols };
    let k_targets = y.cols;
    let n_dev = gpu_ids.len();
    let chunk_size = x.rows.div_ceil(n_dev);

    let chunk_results: Vec<OchreResult<(Matrix, Matrix)>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_dev);
        for (idx, &gid) in gpu_ids.iter().enumerate() {
            let start = idx * chunk_size;
            if start >= x.rows {
                break;
            }
            let end = (start + chunk_size).min(x.rows);
            handles.push(s.spawn(move || {
                cuda_normal_eq_multi_chunk(
                    driver,
                    gid,
                    x,
                    y,
                    sample_weight,
                    fit_intercept,
                    start,
                    end,
                )
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(OchreError::DeviceError(
                        "Multi-GPU worker thread panicked (´；ω；`)".to_string(),
                    ))
                })
            })
            .collect()
    });

    let mut total_xtx = Matrix::zeros(p, p);
    let mut total_xty = Matrix::zeros(p, k_targets);
    for res in chunk_results {
        match res {
            Ok((m, y_mat)) => {
                for i in 0..(p * p) {
                    total_xtx.data[i] += m.data[i];
                }
                for i in 0..(p * k_targets) {
                    total_xty.data[i] += y_mat.data[i];
                }
            }
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok((total_xtx, total_xty)))
}
