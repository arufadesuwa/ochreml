// OchreML Native NVIDIA CUDA + cuBLAS Dynamic Backend (*≧ω≦*)
// Uses runtime dlopen (libloading) so PyPI wheels compile cleanly everywhere without CUDA Toolkit,
// while achieving full FP64 Tensor/CUDA performance on NVIDIA GPUs (e.g., Kaggle T4 x2).

use crate::error::{OchreError, OchreResult};
use crate::matrix::Matrix;
use libloading::Library;
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
type FnCublasDdot = unsafe extern "C" fn(
    CublasHandle,
    i32,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
    *mut f64,
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
}

// Safety: Protected by Mutex and bound to calling thread via cuCtxSetCurrent before every call.
unsafe impl Send for CudaDeviceCtx {}
unsafe impl Sync for CudaDeviceCtx {}

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
    cublas_ddot: FnCublasDdot,
    cublas_ddgmm: Option<FnCublasDdgmm>,
    contexts: Mutex<HashMap<usize, Arc<Mutex<CudaDeviceCtx>>>>,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

static CUDA_DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();

fn load_first_library(candidates: &[&str]) -> Option<Library> {
    for name in candidates {
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
        let cublas_ddot = *cublas_lib.get::<FnCublasDdot>(b"cublasDdot_v2\0").ok()?;
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
            cublas_ddot,
            cublas_ddgmm,
            contexts: Mutex::new(HashMap::new()),
        })
    }
}

fn get_driver() -> Option<&'static CudaDriver> {
    CUDA_DRIVER.get_or_init(init_cuda_driver).as_ref()
}

pub fn is_available() -> bool {
    get_driver().is_some()
}

// RAII GPU buffer so VRAM is always freed even on early error returns
struct DeviceBuffer<'a> {
    driver: &'a CudaDriver,
    ptr: CuDevicePtr,
}

impl<'a> DeviceBuffer<'a> {
    fn alloc(driver: &'a CudaDriver, len_f64: usize) -> OchreResult<Self> {
        let mut ptr: CuDevicePtr = 0;
        let bytes = len_f64.max(1) * std::mem::size_of::<f64>();
        let rc = unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuMemAlloc failed with code {} (´；ω；`)",
                rc
            )));
        }
        Ok(Self { driver, ptr })
    }

    fn from_slice(driver: &'a CudaDriver, data: &[f64]) -> OchreResult<Self> {
        let buf = Self::alloc(driver, data.len())?;
        let bytes = data.len() * std::mem::size_of::<f64>();
        if bytes > 0 {
            let rc =
                unsafe { (driver.cu_memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes) };
            if rc != CUDA_SUCCESS {
                return Err(OchreError::DeviceError(format!(
                    "CUDA cuMemcpyHtoD failed with code {} (´；ω；`)",
                    rc
                )));
            }
        }
        Ok(buf)
    }

    fn copy_to_host(&self, out: &mut [f64]) -> OchreResult<()> {
        let bytes = out.len() * std::mem::size_of::<f64>();
        if bytes > 0 {
            let rc = unsafe {
                (self.driver.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            };
            if rc != CUDA_SUCCESS {
                return Err(OchreError::DeviceError(format!(
                    "CUDA cuMemcpyDtoH failed with code {} (´；ω；`)",
                    rc
                )));
            }
        }
        Ok(())
    }
}

impl<'a> Drop for DeviceBuffer<'a> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
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
        let rc = unsafe { (self.cu_device_get)(&mut dev, gpu_id as i32) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuDeviceGet({}) failed with code {} ( >_< )",
                gpu_id, rc
            )));
        }

        let mut ctx: CuContext = std::ptr::null_mut();
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

        let set_rc = unsafe { (self.cu_ctx_set_current)(ctx) };
        if set_rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuCtxSetCurrent failed with code {} ( >_< )",
                set_rc
            )));
        }

        let mut handle: CublasHandle = std::ptr::null_mut();
        let blas_rc = unsafe { (self.cublas_create)(&mut handle) };
        if blas_rc != CUBLAS_STATUS_SUCCESS || handle.is_null() {
            return Err(OchreError::DeviceError(format!(
                "cuBLAS cublasCreate_v2 failed with code {} (´；ω；`)",
                blas_rc
            )));
        }

        let arc = Arc::new(Mutex::new(CudaDeviceCtx { ctx, handle }));
        map.insert(gpu_id, Arc::clone(&arc));
        Ok(arc)
    }

    fn bind_ctx(&self, guard: &CudaDeviceCtx) -> OchreResult<()> {
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

// Mirror lower triangle (row-major) produced by column-major CUBLAS_FILL_MODE_UPPER cublasDsyrk_v2
#[inline]
fn mirror_cublas_upper_to_full(mat: &mut Matrix) {
    let p = mat.rows;
    for r in 0..p {
        for c in (r + 1)..p {
            let val = mat.data[c * p + r];
            mat.data[r * p + c] = val;
        }
    }
}

// General Matrix Multiplication: C = A * B (Zero-transpose via C^T = B^T * A^T in column-major)
pub fn cuda_matmul(gpu_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let m = a.rows as i32;
        let k = a.cols as i32;
        let n = b.cols as i32;

        let d_a = DeviceBuffer::from_slice(driver, &a.data)?;
        let d_b = DeviceBuffer::from_slice(driver, &b.data)?;
        let d_c = DeviceBuffer::alloc(driver, a.rows * b.cols)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        let status = unsafe {
            (driver.cublas_dgemm)(
                guard.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                d_b.ptr,
                n,
                d_a.ptr,
                k,
                &beta,
                d_c.ptr,
                n,
            )
        };
        check_cublas(status, "cublasDgemm_v2")?;

        let mut out = Matrix::zeros(a.rows, b.cols);
        d_c.copy_to_host(&mut out.data)?;
        Ok(out)
    })();
    Some(res)
}

// Compute A^T * B where A is (N x p) and B is (N x K) row-major -> (p x K) row-major
pub fn cuda_matmul_at_b(gpu_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let n_rows = a.rows as i32;
        let p = a.cols as i32;
        let k = b.cols as i32;

        let d_a = DeviceBuffer::from_slice(driver, &a.data)?;
        let d_b = DeviceBuffer::from_slice(driver, &b.data)?;
        let d_c = DeviceBuffer::alloc(driver, a.cols * b.cols)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        let status = unsafe {
            (driver.cublas_dgemm)(
                guard.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                k,
                p,
                n_rows,
                &alpha,
                d_b.ptr,
                k,
                d_a.ptr,
                p,
                &beta,
                d_c.ptr,
                k,
            )
        };
        check_cublas(status, "cublasDgemm_v2(AtB)")?;

        let mut out = Matrix::zeros(a.cols, b.cols);
        d_c.copy_to_host(&mut out.data)?;
        Ok(out)
    })();
    Some(res)
}

// Matrix-Vector Multiplication: y = A * v (Zero-transpose via (A^T)^T * v)
pub fn cuda_matvec(gpu_id: usize, a: &Matrix, v: &[f64]) -> Option<OchreResult<Vec<f64>>> {
    let driver = get_driver()?;
    let res = (|| {
        let ctx_arc = driver.get_device_ctx(gpu_id)?;
        let guard = ctx_arc.lock().map_err(|_| {
            OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string())
        })?;
        driver.bind_ctx(&guard)?;

        let m = a.rows as i32;
        let n = a.cols as i32;

        let d_a = DeviceBuffer::from_slice(driver, &a.data)?;
        let d_v = DeviceBuffer::from_slice(driver, v)?;
        let d_y = DeviceBuffer::alloc(driver, a.rows)?;

        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        let status = unsafe {
            (driver.cublas_dgemv)(
                guard.handle,
                CUBLAS_OP_T,
                n,
                m,
                &alpha,
                d_a.ptr,
                n,
                d_v.ptr,
                1,
                &beta,
                d_y.ptr,
                1,
            )
        };
        check_cublas(status, "cublasDgemv_v2")?;

        let mut out = vec![0.0; a.rows];
        d_y.copy_to_host(&mut out)?;
        Ok(out)
    })();
    Some(res)
}

// Single-GPU 1D Normal Equation chunk over [start_row..end_row] without copying X into N x (p+1)
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
    let guard = ctx_arc
        .lock()
        .map_err(|_| OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string()))?;
    driver.bind_ctx(&guard)?;

    let x_slice = &x.data[(start_row * p_x)..(end_row * p_x)];
    let y_slice = &y[start_row..end_row];

    let d_x = DeviceBuffer::from_slice(driver, x_slice)?;
    let d_y = DeviceBuffer::from_slice(driver, y_slice)?;
    let d_xtx = DeviceBuffer::alloc(driver, p_x * p_x)?;
    let d_xty = DeviceBuffer::alloc(driver, p_x)?;

    let alpha = 1.0_f64;
    let beta = 0.0_f64;

    // 1. X^T * X via cublasDsyrk_v2 (row-major X(N x p) == col-major X^T(p x N))
    let syrk_rc = unsafe {
        (driver.cublas_dsyrk)(
            guard.handle,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_x.ptr,
            p_x as i32,
            &beta,
            d_xtx.ptr,
            p_x as i32,
        )
    };
    check_cublas(syrk_rc, "cublasDsyrk_v2")?;

    // 2. X^T * y via cublasDgemv_v2
    let gemv_rc = unsafe {
        (driver.cublas_dgemv)(
            guard.handle,
            CUBLAS_OP_N,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_x.ptr,
            p_x as i32,
            d_y.ptr,
            1,
            &beta,
            d_xty.ptr,
            1,
        )
    };
    check_cublas(gemv_rc, "cublasDgemv_v2(Xty)")?;

    let mut xtx_sub = Matrix::zeros(p_x, p_x);
    d_xtx.copy_to_host(&mut xtx_sub.data)?;
    mirror_cublas_upper_to_full(&mut xtx_sub);

    let mut xty_sub = vec![0.0; p_x];
    d_xty.copy_to_host(&mut xty_sub)?;

    if !fit_intercept {
        return Ok((xtx_sub, xty_sub));
    }

    // 3. Intercept sums via ones vector on GPU (avoids copying X into N x (p+1))
    let ones = vec![1.0_f64; n_chunk];
    let d_ones = DeviceBuffer::from_slice(driver, &ones)?;
    let d_sum_x = DeviceBuffer::alloc(driver, p_x)?;

    let sum_x_rc = unsafe {
        (driver.cublas_dgemv)(
            guard.handle,
            CUBLAS_OP_N,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_x.ptr,
            p_x as i32,
            d_ones.ptr,
            1,
            &beta,
            d_sum_x.ptr,
            1,
        )
    };
    check_cublas(sum_x_rc, "cublasDgemv_v2(sum_x)")?;

    let mut sum_y = 0.0_f64;
    let sum_y_rc = unsafe {
        (driver.cublas_ddot)(
            guard.handle,
            n_chunk as i32,
            d_y.ptr,
            1,
            d_ones.ptr,
            1,
            &mut sum_y,
        )
    };
    check_cublas(sum_y_rc, "cublasDdot_v2(sum_y)")?;

    let mut sum_x = vec![0.0; p_x];
    d_sum_x.copy_to_host(&mut sum_x)?;

    let mut full_xtx = Matrix::zeros(p_total, p_total);
    for r in 0..p_x {
        let dst_off = r * p_total;
        let src_off = r * p_x;
        full_xtx.data[dst_off..(dst_off + p_x)]
            .copy_from_slice(&xtx_sub.data[src_off..(src_off + p_x)]);
        full_xtx.data[dst_off + p_x] = sum_x[r];
    }
    let last_off = p_x * p_total;
    full_xtx.data[last_off..(last_off + p_x)].copy_from_slice(&sum_x);
    full_xtx.data[last_off + p_x] = n_chunk as f64;

    let mut full_xty = Vec::with_capacity(p_total);
    full_xty.extend_from_slice(&xty_sub);
    full_xty.push(sum_y);

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

    if gpu_ids.len() == 1 || x.rows < gpu_ids.len() {
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
    let chunk_size = (x.rows + n_dev - 1) / n_dev;

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

// Multi-target & optional sample_weight Normal Equation chunk on GPU
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
    let guard = ctx_arc
        .lock()
        .map_err(|_| OchreError::DeviceError("CUDA device mutex poisoned ( >_< )".to_string()))?;
    driver.bind_ctx(&guard)?;

    let x_slice = &x.data[(start_row * p_x)..(end_row * p_x)];
    let y_slice = &y.data[(start_row * k_targets)..(end_row * k_targets)];

    let d_x = DeviceBuffer::from_slice(driver, x_slice)?;
    let d_y = DeviceBuffer::from_slice(driver, y_slice)?;

    let (d_weight_vec, weight_sum) = if let Some(sw) = sample_weight {
        let sw_slice = &sw[start_row..end_row];
        let mut sqrt_sw = Vec::with_capacity(n_chunk);
        let mut w_sum = 0.0;
        for &w in sw_slice {
            w_sum += w;
            sqrt_sw.push(w.sqrt());
        }
        let d_sqrt = DeviceBuffer::from_slice(driver, &sqrt_sw)?;
        let ddgmm = driver.cublas_ddgmm.ok_or_else(|| {
            OchreError::DeviceError(
                "cublasDdgmm not available in loaded cuBLAS library ( >_< )".to_string(),
            )
        })?;
        let rc_x = unsafe {
            ddgmm(
                guard.handle,
                CUBLAS_SIDE_RIGHT,
                p_x as i32,
                n_chunk as i32,
                d_x.ptr,
                p_x as i32,
                d_sqrt.ptr,
                1,
                d_x.ptr,
                p_x as i32,
            )
        };
        check_cublas(rc_x, "cublasDdgmm(X)")?;

        let rc_y = unsafe {
            ddgmm(
                guard.handle,
                CUBLAS_SIDE_RIGHT,
                k_targets as i32,
                n_chunk as i32,
                d_y.ptr,
                k_targets as i32,
                d_sqrt.ptr,
                1,
                d_y.ptr,
                k_targets as i32,
            )
        };
        check_cublas(rc_y, "cublasDdgmm(Y)")?;
        (Some(d_sqrt), w_sum)
    } else {
        (None, n_chunk as f64)
    };

    let d_xtx = DeviceBuffer::alloc(driver, p_x * p_x)?;
    let d_xty = DeviceBuffer::alloc(driver, p_x * k_targets)?;

    let alpha = 1.0_f64;
    let beta = 0.0_f64;

    let syrk_rc = unsafe {
        (driver.cublas_dsyrk)(
            guard.handle,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_x.ptr,
            p_x as i32,
            &beta,
            d_xtx.ptr,
            p_x as i32,
        )
    };
    check_cublas(syrk_rc, "cublasDsyrk_v2(multi)")?;

    // X^T * Y: col-major C^T(K x p_x) = B^T(K x N) * A(N x p_x)
    let gemm_rc = unsafe {
        (driver.cublas_dgemm)(
            guard.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            k_targets as i32,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_y.ptr,
            k_targets as i32,
            d_x.ptr,
            p_x as i32,
            &beta,
            d_xty.ptr,
            k_targets as i32,
        )
    };
    check_cublas(gemm_rc, "cublasDgemm_v2(XtY)")?;

    let mut xtx_sub = Matrix::zeros(p_x, p_x);
    d_xtx.copy_to_host(&mut xtx_sub.data)?;
    mirror_cublas_upper_to_full(&mut xtx_sub);

    let mut xty_sub = Matrix::zeros(p_x, k_targets);
    d_xty.copy_to_host(&mut xty_sub.data)?;

    if !fit_intercept {
        return Ok((xtx_sub, xty_sub));
    }

    // Intercept sums: if weighted, d_x and d_y are already scaled by sqrt(w),
    // so multiplying by sqrt(w) gives sum(w * x_j) and sum(w * y_t).
    let d_ones_owned;
    let d_vec_ptr = match d_weight_vec {
        Some(ref buf) => buf.ptr,
        None => {
            let ones = vec![1.0_f64; n_chunk];
            d_ones_owned = DeviceBuffer::from_slice(driver, &ones)?;
            d_ones_owned.ptr
        }
    };

    let d_sum_x = DeviceBuffer::alloc(driver, p_x)?;
    let sum_x_rc = unsafe {
        (driver.cublas_dgemv)(
            guard.handle,
            CUBLAS_OP_N,
            p_x as i32,
            n_chunk as i32,
            &alpha,
            d_x.ptr,
            p_x as i32,
            d_vec_ptr,
            1,
            &beta,
            d_sum_x.ptr,
            1,
        )
    };
    check_cublas(sum_x_rc, "cublasDgemv_v2(sum_x multi)")?;

    let d_sum_y = DeviceBuffer::alloc(driver, k_targets)?;
    let sum_y_rc = unsafe {
        (driver.cublas_dgemv)(
            guard.handle,
            CUBLAS_OP_N,
            k_targets as i32,
            n_chunk as i32,
            &alpha,
            d_y.ptr,
            k_targets as i32,
            d_vec_ptr,
            1,
            &beta,
            d_sum_y.ptr,
            1,
        )
    };
    check_cublas(sum_y_rc, "cublasDgemv_v2(sum_y multi)")?;

    let mut sum_x = vec![0.0; p_x];
    d_sum_x.copy_to_host(&mut sum_x)?;
    let mut sum_y = vec![0.0; k_targets];
    d_sum_y.copy_to_host(&mut sum_y)?;

    let mut full_xtx = Matrix::zeros(p_total, p_total);
    for r in 0..p_x {
        let dst_off = r * p_total;
        let src_off = r * p_x;
        full_xtx.data[dst_off..(dst_off + p_x)]
            .copy_from_slice(&xtx_sub.data[src_off..(src_off + p_x)]);
        full_xtx.data[dst_off + p_x] = sum_x[r];
    }
    let last_off = p_x * p_total;
    full_xtx.data[last_off..(last_off + p_x)].copy_from_slice(&sum_x);
    full_xtx.data[last_off + p_x] = weight_sum;

    let mut full_xty = Matrix::zeros(p_total, k_targets);
    full_xty.data[0..(p_x * k_targets)].copy_from_slice(&xty_sub.data);
    full_xty.data[(p_x * k_targets)..(p_total * k_targets)].copy_from_slice(&sum_y);

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

    if gpu_ids.len() == 1 || x.rows < gpu_ids.len() {
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
    let chunk_size = (x.rows + n_dev - 1) / n_dev;

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
