// OchreML CUDA Neural Network Backend (cuBLAS + sm_50 Runtime PTX) (*≧ω≦*)
// Dynamic libcublas loading (cublasSgemm_v2, cublasSgemmStridedBatched, cublasGemmEx for BF16/F16)
// plus universal sm_50 runtime PTX kernels for N8/N9/N10 and persistent VRAM workspace cache.

use crate::backends::cuda::{
    CUDA_SUCCESS, CuContext, CuDevicePtr, CuResult, CudaDriver, get_driver,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::dtype::DType;
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

type CublasStatus = i32;
type CublasHandle = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;

// cudaDataType_t & cublasComputeType_t ABI constants
const CUDA_R_32F: i32 = 0;
const CUDA_R_16F: i32 = 2;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT_TENSOR_OP: i32 = 99;

const DEFAULT_NN_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const MIN_CUDA_NN_ELEMENTS: usize = 16_384;
const MIN_CUDA_GEMM_OPS: usize = 131_072;

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

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type FnCublasSgemm = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    CuDevicePtr,
    i32,
    CuDevicePtr,
    i32,
    *const f32,
    CuDevicePtr,
    i32,
) -> CublasStatus;
type FnCublasSgemmStridedBatched = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    CuDevicePtr,
    i32,
    i64,
    CuDevicePtr,
    i32,
    i64,
    *const f32,
    CuDevicePtr,
    i32,
    i64,
    i32,
) -> CublasStatus;
type FnCublasGemmEx = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    CuDevicePtr,
    i32,
    i32,
    CuDevicePtr,
    i32,
    i32,
    *const f32,
    CuDevicePtr,
    i32,
    i32,
    i32,
    i32,
) -> CublasStatus;

/// Persistent VRAM workspace cache for NN forward/backward tensors.
pub struct CudaNnWorkspace {
    ptr: CuDevicePtr,
    byte_len: usize,
    ctx: CuContext,
    cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
}

// SAFETY: Access is serialized per CUDA context through a Mutex in CudaNnRuntime.
unsafe impl Send for CudaNnWorkspace {}
// SAFETY: Guarded by Mutex.
unsafe impl Sync for CudaNnWorkspace {}

impl Drop for CudaNnWorkspace {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: Bind owning CUDA context before releasing persistent device buffer.
            unsafe {
                let _ = (self.cu_ctx_set_current)(self.ctx);
                let _ = (self.cu_mem_free)(self.ptr);
            }
            self.ptr = 0;
            self.byte_len = 0;
        }
    }
}

#[inline]
fn align256(bytes: usize) -> usize {
    (bytes + 255) & !255
}

impl CudaNnWorkspace {
    fn new(driver: &CudaDriver, ctx: CuContext) -> Self {
        Self {
            ptr: 0,
            byte_len: 0,
            ctx,
            cu_ctx_set_current: driver.cu_ctx_set_current,
            cu_mem_free: driver.cu_mem_free,
        }
    }

    pub(crate) fn ensure_bytes(
        &mut self,
        driver: &CudaDriver,
        required: usize,
    ) -> OchreResult<CuDevicePtr> {
        let needed = align256(required.max(256));
        if self.ptr != 0 && self.byte_len >= needed {
            return Ok(self.ptr);
        }
        let alloc_bytes = needed.max(DEFAULT_NN_WORKSPACE_BYTES);
        let mut new_ptr: CuDevicePtr = 0;
        // SAFETY: Context is bound by caller; `new_ptr` receives device allocation.
        let rc = unsafe { (driver.cu_mem_alloc)(&mut new_ptr, alloc_bytes) };
        if rc != CUDA_SUCCESS || new_ptr == 0 {
            return Err(OchreError::DeviceError(format!(
                "CUDA NN cuMemAlloc({alloc_bytes}) failed with code {rc} (´；ω；`)"
            )));
        }
        if self.ptr != 0 {
            // SAFETY: Old buffer is idle because context execution is serialized by mutex.
            unsafe {
                let _ = (driver.cu_mem_free)(self.ptr);
            }
        }
        self.ptr = new_ptr;
        self.byte_len = alloc_bytes;
        Ok(self.ptr)
    }
}

struct NnPtxFuncs {
    _module: CuModule,
    gelu: CuFunction,
    gelu_bwd: CuFunction,
    silu: CuFunction,
    silu_bwd: CuFunction,
    row_layernorm: CuFunction,
    row_rmsnorm: CuFunction,
    causal_mask: CuFunction,
    cross_entropy_grad: CuFunction,
    adamw_step: CuFunction,
    im2col: CuFunction,
    col2im: CuFunction,
    max_pool2d: CuFunction,
    avg_pool2d: CuFunction,
}

struct CudaNnCtxState {
    handle: CublasHandle,
    cublas_destroy: FnCublasDestroy,
    workspace: CudaNnWorkspace,
    ptx: Option<NnPtxFuncs>,
}

// SAFETY: Serialized via per-context Mutex inside CudaNnRuntime.
unsafe impl Send for CudaNnCtxState {}
// SAFETY: Guarded by Mutex.
unsafe impl Sync for CudaNnCtxState {}

impl Drop for CudaNnCtxState {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Bind context and destroy cuBLAS handle on teardown.
            unsafe {
                let _ = (self.workspace.cu_ctx_set_current)(self.workspace.ctx);
                let _ = (self.cublas_destroy)(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

struct CudaNnRuntime {
    _cublas_lib: Library,
    cublas_create: FnCublasCreate,
    cublas_destroy: FnCublasDestroy,
    cublas_sgemm: FnCublasSgemm,
    cublas_sgemm_strided_batched: Option<FnCublasSgemmStridedBatched>,
    cublas_gemm_ex: Option<FnCublasGemmEx>,
    cu_module_load_data: FnCuModuleLoadData,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    contexts: Mutex<HashMap<usize, Box<Mutex<CudaNnCtxState>>>>,
}

// SAFETY: Dynamic libraries and resolved symbols are immutable after init; context map is Mutex-protected.
unsafe impl Send for CudaNnRuntime {}
// SAFETY: Guarded by Mutex.
unsafe impl Sync for CudaNnRuntime {}

static CUDA_NN_RUNTIME: OnceLock<Option<CudaNnRuntime>> = OnceLock::new();

// Universal sm_50 PTX kernels for N8, N9, and N10 (no sm_75+ instructions).
const OCHRE_NN_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// Helper macro semantics implemented inline:
// tanh(u) via ex2: clamp 2u in [-40, 40], e = exp(2u), (e - 1)/(e + 1)

.visible .entry ochre_gelu_f32(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<16>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra DONE;

    ld.param.u64 %rd3, [p_x];
    ld.param.u64 %rd4, [p_out];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f1, [%rd3]; // x

    mul.f32 %f2, %f1, %f1;
    mul.f32 %f2, %f2, %f1;                // x^3
    mov.f32 %f3, 0f3D372713;              // 0.044715
    fma.rn.f32 %f4, %f3, %f2, %f1;        // x + 0.044715 * x^3
    mov.f32 %f5, 0f3F4C422A;              // sqrt(2/pi) = 0.79788456
    mul.f32 %f6, %f4, %f5;                // u
    add.f32 %f7, %f6, %f6;                // 2u
    mov.f32 %f8, 0fC2200000;              // -40.0
    mov.f32 %f9, 0f42200000;              // +40.0
    max.f32 %f7, %f7, %f8;
    min.f32 %f7, %f7, %f9;
    mov.f32 %f10, 0f3FB8AA3B;             // log2(e) = 1.44269504
    mul.f32 %f7, %f7, %f10;
    ex2.approx.f32 %f11, %f7;             // exp(2u)
    mov.f32 %f12, 0f3F800000;             // 1.0
    sub.f32 %f13, %f11, %f12;
    add.f32 %f14, %f11, %f12;
    div.rn.f32 %f15, %f13, %f14;          // tanh(u)
    add.f32 %f15, %f15, %f12;             // 1 + tanh(u)
    mov.f32 %f12, 0f3F000000;             // 0.5
    mul.f32 %f15, %f15, %f12;
    mul.f32 %f15, %f15, %f1;
    st.global.f32 [%rd4], %f15;
DONE:
    ret;
}

.visible .entry ochre_gelu_bwd_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_dx,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<22>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra DONE;

    ld.param.u64 %rd3, [p_dy];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_dx];
    shl.b64 %rd6, %rd1, 2;
    add.u64 %rd3, %rd3, %rd6;
    add.u64 %rd4, %rd4, %rd6;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f1, [%rd3]; // dy
    ld.global.f32 %f2, [%rd4]; // x

    mul.f32 %f3, %f2, %f2;                // x^2
    mul.f32 %f4, %f3, %f2;                // x^3
    mov.f32 %f5, 0f3D372713;              // 0.044715
    fma.rn.f32 %f6, %f5, %f4, %f2;        // x + k*x^3
    mov.f32 %f7, 0f3F4C422A;              // c = 0.79788456
    mul.f32 %f8, %f6, %f7;                // u
    add.f32 %f9, %f8, %f8;                // 2u
    mov.f32 %f10, 0fC2200000;
    mov.f32 %f11, 0f42200000;
    max.f32 %f9, %f9, %f10;
    min.f32 %f9, %f9, %f11;
    mov.f32 %f12, 0f3FB8AA3B;
    mul.f32 %f9, %f9, %f12;
    ex2.approx.f32 %f13, %f9;
    mov.f32 %f14, 0f3F800000;             // 1.0
    sub.f32 %f15, %f13, %f14;
    add.f32 %f16, %f13, %f14;
    div.rn.f32 %f17, %f15, %f16;          // t = tanh(u)

    // sech^2 = 1 - t*t
    mul.f32 %f18, %f17, %f17;
    sub.f32 %f18, %f14, %f18;
    // du/dx = c * (1 + 3*k*x^2)
    mov.f32 %f19, 0f3E0972E2;             // 3 * 0.044715 = 0.134145
    fma.rn.f32 %f20, %f19, %f3, %f14;
    mul.f32 %f20, %f20, %f7;
    // dgelu = 0.5*(1+t) + 0.5*x*sech^2*du_dx
    add.f32 %f21, %f14, %f17;
    mul.f32 %f18, %f18, %f20;
    fma.rn.f32 %f21, %f2, %f18, %f21;
    mov.f32 %f12, 0f3F000000;             // 0.5
    mul.f32 %f21, %f21, %f12;
    mul.f32 %f21, %f21, %f1;
    st.global.f32 [%rd5], %f21;
DONE:
    ret;
}

.visible .entry ochre_silu_f32(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra DONE;

    ld.param.u64 %rd3, [p_x];
    ld.param.u64 %rd4, [p_out];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f1, [%rd3];
    neg.f32 %f2, %f1;
    mov.f32 %f3, 0fC2A00000;              // -80.0
    mov.f32 %f4, 0f42A00000;              // +80.0
    max.f32 %f2, %f2, %f3;
    min.f32 %f2, %f2, %f4;
    mov.f32 %f5, 0f3FB8AA3B;              // log2(e)
    mul.f32 %f2, %f2, %f5;
    ex2.approx.f32 %f6, %f2;
    mov.f32 %f7, 0f3F800000;
    add.f32 %f8, %f7, %f6;
    div.rn.f32 %f9, %f1, %f8;
    st.global.f32 [%rd4], %f9;
DONE:
    ret;
}

.visible .entry ochre_silu_bwd_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_dx,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<14>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra DONE;

    ld.param.u64 %rd3, [p_dy];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_dx];
    shl.b64 %rd6, %rd1, 2;
    add.u64 %rd3, %rd3, %rd6;
    add.u64 %rd4, %rd4, %rd6;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f1, [%rd3]; // dy
    ld.global.f32 %f2, [%rd4]; // x

    neg.f32 %f3, %f2;
    mov.f32 %f4, 0fC2A00000;
    mov.f32 %f5, 0f42A00000;
    max.f32 %f3, %f3, %f4;
    min.f32 %f3, %f3, %f5;
    mov.f32 %f6, 0f3FB8AA3B;
    mul.f32 %f3, %f3, %f6;
    ex2.approx.f32 %f7, %f3;
    mov.f32 %f8, 0f3F800000;              // 1.0
    add.f32 %f9, %f8, %f7;
    div.rn.f32 %f10, %f8, %f9;            // sig
    sub.f32 %f11, %f8, %f10;              // 1 - sig
    mul.f32 %f12, %f2, %f11;              // x * (1 - sig)
    add.f32 %f12, %f8, %f12;              // 1 + x*(1-sig)
    mul.f32 %f13, %f10, %f12;
    mul.f32 %f13, %f13, %f1;
    st.global.f32 [%rd5], %f13;
DONE:
    ret;
}

.visible .entry ochre_row_layernorm_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_beta,
    .param .u64 p_out,
    .param .u64 p_mean,
    .param .u64 p_rstd,
    .param .u64 p_rows,
    .param .u64 p_cols,
    .param .u32 p_has_gamma,
    .param .u32 p_has_beta,
    .param .f32 p_eps
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1; // row r
    ld.param.u64 %rd2, [p_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra LN_DONE;

    ld.param.u64 %rd3, [p_cols];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_out];
    mul.lo.u64 %rd6, %rd1, %rd3;
    shl.b64 %rd6, %rd6, 2;
    add.u64 %rd4, %rd4, %rd6; // x_row
    add.u64 %rd5, %rd5, %rd6; // out_row

    mov.f32 %f1, 0f00000000; // sum
    mov.u64 %rd7, 0;
LN_SUM:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra LN_SUM_END;
    shl.b64 %rd8, %rd7, 2;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.f32 %f2, [%rd9];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd7, %rd7, 1;
    bra LN_SUM;
LN_SUM_END:
    cvt.rn.f32.u64 %f3, %rd3;
    div.rn.f32 %f4, %f1, %f3; // mean

    mov.f32 %f5, 0f00000000; // var_sum
    mov.u64 %rd7, 0;
LN_VAR:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra LN_VAR_END;
    shl.b64 %rd8, %rd7, 2;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.f32 %f2, [%rd9];
    sub.f32 %f6, %f2, %f4;
    fma.rn.f32 %f5, %f6, %f6, %f5;
    add.u64 %rd7, %rd7, 1;
    bra LN_VAR;
LN_VAR_END:
    div.rn.f32 %f5, %f5, %f3; // var
    ld.param.f32 %f7, [p_eps];
    add.f32 %f5, %f5, %f7;
    rsqrt.approx.f32 %f8, %f5; // rstd

    ld.param.u64 %rd10, [p_mean];
    ld.param.u64 %rd11, [p_rstd];
    shl.b64 %rd12, %rd1, 2;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd11, %rd11, %rd12;
    st.global.f32 [%rd10], %f4;
    st.global.f32 [%rd11], %f8;

    ld.param.u64 %rd13, [p_gamma];
    ld.param.u64 %rd14, [p_beta];
    ld.param.u32 %r4, [p_has_gamma];
    ld.param.u32 %r5, [p_has_beta];

    mov.u64 %rd7, 0;
LN_NORM:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra LN_DONE;
    shl.b64 %rd8, %rd7, 2;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.f32 %f2, [%rd9];
    sub.f32 %f6, %f2, %f4;
    mul.f32 %f6, %f6, %f8; // x_hat

    mov.f32 %f9, 0f3F800000;
    setp.eq.u32 %p3, %r4, 0;
    @%p3 bra LN_NO_G;
    add.u64 %rd15, %rd13, %rd8;
    ld.global.f32 %f9, [%rd15];
LN_NO_G:
    mov.f32 %f10, 0f00000000;
    setp.eq.u32 %p4, %r5, 0;
    @%p4 bra LN_NO_B;
    add.u64 %rd15, %rd14, %rd8;
    ld.global.f32 %f10, [%rd15];
LN_NO_B:
    fma.rn.f32 %f11, %f6, %f9, %f10;
    add.u64 %rd15, %rd5, %rd8;
    st.global.f32 [%rd15], %f11;
    add.u64 %rd7, %rd7, 1;
    bra LN_NORM;
LN_DONE:
    ret;
}

.visible .entry ochre_row_rmsnorm_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_out,
    .param .u64 p_rstd,
    .param .u64 p_rows,
    .param .u64 p_cols,
    .param .u32 p_has_gamma,
    .param .f32 p_eps
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra RN_DONE;

    ld.param.u64 %rd3, [p_cols];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_out];
    mul.lo.u64 %rd6, %rd1, %rd3;
    shl.b64 %rd6, %rd6, 2;
    add.u64 %rd4, %rd4, %rd6;
    add.u64 %rd5, %rd5, %rd6;

    mov.f32 %f1, 0f00000000;
    mov.u64 %rd7, 0;
RN_SUM:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra RN_END;
    shl.b64 %rd8, %rd7, 2;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.f32 %f2, [%rd9];
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u64 %rd7, %rd7, 1;
    bra RN_SUM;
RN_END:
    cvt.rn.f32.u64 %f3, %rd3;
    div.rn.f32 %f1, %f1, %f3;
    ld.param.f32 %f4, [p_eps];
    add.f32 %f1, %f1, %f4;
    rsqrt.approx.f32 %f5, %f1; // rstd

    ld.param.u64 %rd10, [p_rstd];
    shl.b64 %rd11, %rd1, 2;
    add.u64 %rd10, %rd10, %rd11;
    st.global.f32 [%rd10], %f5;

    ld.param.u64 %rd12, [p_gamma];
    ld.param.u32 %r4, [p_has_gamma];
    mov.u64 %rd7, 0;
RN_OUT:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra RN_DONE;
    shl.b64 %rd8, %rd7, 2;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.f32 %f2, [%rd9];
    mul.f32 %f6, %f2, %f5;
    mov.f32 %f7, 0f3F800000;
    setp.eq.u32 %p3, %r4, 0;
    @%p3 bra RN_NO_G;
    add.u64 %rd13, %rd12, %rd8;
    ld.global.f32 %f7, [%rd13];
RN_NO_G:
    mul.f32 %f8, %f6, %f7;
    add.u64 %rd13, %rd5, %rd8;
    st.global.f32 [%rd13], %f8;
    add.u64 %rd7, %rd7, 1;
    bra RN_OUT;
RN_DONE:
    ret;
}

.visible .entry ochre_causal_mask_f32(
    .param .u64 p_scores,
    .param .u64 p_total,
    .param .u64 p_q_len,
    .param .u64 p_k_len,
    .param .u64 p_offset
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<10>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra CM_DONE;

    ld.param.u64 %rd3, [p_k_len];
    ld.param.u64 %rd4, [p_q_len];
    ld.param.u64 %rd5, [p_offset];

    rem.u64 %rd6, %rd1, %rd3;       // kj
    div.u64 %rd7, %rd1, %rd3;       // rem
    rem.u64 %rd8, %rd7, %rd4;       // qi
    add.u64 %rd8, %rd8, %rd5;       // offset + qi
    setp.le.u64 %p2, %rd6, %rd8;
    @%p2 bra CM_DONE;

    ld.param.u64 %rd9, [p_scores];
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd9, %rd9, %rd1;
    mov.f32 %f1, 0fFF800000;        // -Infinity
    st.global.f32 [%rd9], %f1;
CM_DONE:
    ret;
}

.visible .entry ochre_adamw_step_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_m,
    .param .u64 p_v,
    .param .u64 p_n,
    .param .f32 p_lr,
    .param .f32 p_beta1,
    .param .f32 p_beta2,
    .param .f32 p_eps,
    .param .f32 p_wd,
    .param .f32 p_inv_bc1,
    .param .f32 p_inv_bc2
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<18>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra AW_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_param];
    ld.param.u64 %rd5, [p_grad];
    ld.param.u64 %rd6, [p_m];
    ld.param.u64 %rd7, [p_v];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd7, %rd3;

    ld.global.f32 %f1, [%rd4]; // param
    ld.global.f32 %f2, [%rd5]; // grad
    ld.global.f32 %f3, [%rd6]; // m
    ld.global.f32 %f4, [%rd7]; // v

    ld.param.f32 %f5, [p_lr];
    ld.param.f32 %f6, [p_beta1];
    ld.param.f32 %f7, [p_beta2];
    ld.param.f32 %f8, [p_eps];
    ld.param.f32 %f9, [p_wd];
    ld.param.f32 %f10, [p_inv_bc1];
    ld.param.f32 %f11, [p_inv_bc2];

    mov.f32 %f12, 0f3F800000;
    sub.f32 %f13, %f12, %f6;   // 1 - beta1
    mul.f32 %f3, %f3, %f6;
    fma.rn.f32 %f3, %f13, %f2, %f3; // m_new

    sub.f32 %f14, %f12, %f7;   // 1 - beta2
    mul.f32 %f15, %f2, %f2;    // g^2
    mul.f32 %f4, %f4, %f7;
    fma.rn.f32 %f4, %f14, %f15, %f4; // v_new

    st.global.f32 [%rd6], %f3;
    st.global.f32 [%rd7], %f4;

    // decoupled weight decay: param * (1 - lr * wd)
    mul.f32 %f16, %f5, %f9;
    sub.f32 %f16, %f12, %f16;
    mul.f32 %f1, %f1, %f16;

    // update: param -= lr * (m * inv_bc1) / (sqrt(v * inv_bc2) + eps)
    mul.f32 %f13, %f3, %f10;
    mul.f32 %f14, %f4, %f11;
    sqrt.rn.f32 %f14, %f14;
    add.f32 %f14, %f14, %f8;
    div.rn.f32 %f17, %f13, %f14;
    mul.f32 %f17, %f17, %f5;
    sub.f32 %f1, %f1, %f17;
    st.global.f32 [%rd4], %f1;
AW_DONE:
    ret;
}

.visible .entry ochre_cross_entropy_grad_f32(
    .param .u64 p_logits,
    .param .u64 p_targets,
    .param .u64 p_dlogits,
    .param .u64 p_losses,
    .param .u64 p_rows,
    .param .u64 p_cols,
    .param .f32 p_ls,
    .param .u64 p_ignore,
    .param .u32 p_has_ignore,
    .param .f32 p_inv_valid
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<16>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1; // row r
    ld.param.u64 %rd2, [p_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra CE_DONE;

    ld.param.u64 %rd3, [p_cols];
    ld.param.u64 %rd4, [p_targets];
    shl.b64 %rd5, %rd1, 3;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.u64 %rd6, [%rd4]; // target t

    ld.param.u64 %rd7, [p_logits];
    ld.param.u64 %rd8, [p_dlogits];
    mul.lo.u64 %rd9, %rd1, %rd3;
    shl.b64 %rd9, %rd9, 2;
    add.u64 %rd7, %rd7, %rd9;
    add.u64 %rd8, %rd8, %rd9;

    ld.param.u64 %rd10, [p_losses];
    shl.b64 %rd11, %rd1, 2;
    add.u64 %rd10, %rd10, %rd11;

    ld.param.u32 %r4, [p_has_ignore];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra CE_ACTIVE;
    ld.param.u64 %rd12, [p_ignore];
    setp.ne.u64 %p3, %rd6, %rd12;
    @%p3 bra CE_ACTIVE;

    // Ignored row: zero loss and zero grad
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd10], %f1;
    mov.u64 %rd13, 0;
CE_ZERO:
    setp.ge.u64 %p4, %rd13, %rd3;
    @%p4 bra CE_DONE;
    shl.b64 %rd11, %rd13, 2;
    add.u64 %rd12, %rd8, %rd11;
    st.global.f32 [%rd12], %f1;
    add.u64 %rd13, %rd13, 1;
    bra CE_ZERO;

CE_ACTIVE:
    ld.global.f32 %f1, [%rd7]; // max_v
    mov.u64 %rd13, 1;
CE_MAX:
    setp.ge.u64 %p4, %rd13, %rd3;
    @%p4 bra CE_MAX_END;
    shl.b64 %rd11, %rd13, 2;
    add.u64 %rd12, %rd7, %rd11;
    ld.global.f32 %f2, [%rd12];
    max.f32 %f1, %f1, %f2;
    add.u64 %rd13, %rd13, 1;
    bra CE_MAX;
CE_MAX_END:

    mov.f32 %f3, 0f00000000; // sum_exp
    mov.f32 %f4, 0f3FB8AA3B; // log2(e)
    mov.u64 %rd13, 0;
CE_EXP:
    setp.ge.u64 %p4, %rd13, %rd3;
    @%p4 bra CE_EXP_END;
    shl.b64 %rd11, %rd13, 2;
    add.u64 %rd12, %rd7, %rd11;
    ld.global.f32 %f2, [%rd12];
    sub.f32 %f5, %f2, %f1;
    mul.f32 %f5, %f5, %f4;
    ex2.approx.f32 %f5, %f5;
    add.f32 %f3, %f3, %f5;
    add.u64 %rd12, %rd8, %rd11;
    st.global.f32 [%rd12], %f5; // store unnormalized exp temporarily in dlogits
    add.u64 %rd13, %rd13, 1;
    bra CE_EXP;
CE_EXP_END:

    lg2.approx.f32 %f6, %f3;
    mov.f32 %f7, 0f3F317218; // ln(2) = 0.69314718
    mul.f32 %f6, %f6, %f7;
    add.f32 %f6, %f6, %f1;   // log_z

    ld.param.f32 %f8, [p_ls];
    cvt.rn.f32.u64 %f9, %rd3;
    div.rn.f32 %f10, %f8, %f9; // smooth_base = ls / cols
    mov.f32 %f11, 0f3F800000;
    sub.f32 %f11, %f11, %f8;   // 1 - ls
    ld.param.f32 %f12, [p_inv_valid];

    mov.f32 %f13, 0f00000000; // row_loss
    mov.u64 %rd13, 0;
CE_GRAD:
    setp.ge.u64 %p4, %rd13, %rd3;
    @%p4 bra CE_GRAD_END;
    shl.b64 %rd11, %rd13, 2;
    add.u64 %rd12, %rd8, %rd11;
    ld.global.f32 %f5, [%rd12];
    div.rn.f32 %f5, %f5, %f3; // prob

    mov.f32 %f14, %f10;
    setp.ne.u64 %p3, %rd13, %rd6;
    @%p3 bra CE_NOT_TGT;
    add.f32 %f14, %f14, %f11; // q_j
CE_NOT_TGT:
    // loss -= q_j * (logit - log_z)
    add.u64 %rd9, %rd7, %rd11;
    ld.global.f32 %f2, [%rd9];
    sub.f32 %f15, %f2, %f6;
    mul.f32 %f15, %f15, %f14;
    sub.f32 %f13, %f13, %f15;

    sub.f32 %f5, %f5, %f14;
    mul.f32 %f5, %f5, %f12;
    st.global.f32 [%rd12], %f5;
    add.u64 %rd13, %rd13, 1;
    bra CE_GRAD;
CE_GRAD_END:
    st.global.f32 [%rd10], %f13;
CE_DONE:
    ret;
}

.visible .entry ochre_im2col_f32(
    .param .u64 p_in,
    .param .u64 p_col,
    .param .u64 p_total,
    .param .u64 p_c,
    .param .u64 p_h,
    .param .u64 p_w,
    .param .u64 p_kh,
    .param .u64 p_kw,
    .param .u64 p_sh,
    .param .u64 p_sw,
    .param .u64 p_ph,
    .param .u64 p_pw,
    .param .u64 p_dh,
    .param .u64 p_dw,
    .param .u64 p_oh,
    .param .u64 p_ow
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<28>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1; // flat idx in col_mat
    ld.param.u64 %rd2, [p_total];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra IC_DONE;

    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_kh];
    ld.param.u64 %rd5, [p_kw];
    ld.param.u64 %rd6, [p_oh];
    ld.param.u64 %rd7, [p_ow];

    mul.lo.u64 %rd8, %rd3, %rd4;
    mul.lo.u64 %rd8, %rd8, %rd5;  // col_width = c * kh * kw
    div.u64 %rd9, %rd1, %rd8;     // row
    rem.u64 %rd10, %rd1, %rd8;    // col

    rem.u64 %rd11, %rd9, %rd7;    // ow
    div.u64 %rd12, %rd9, %rd7;
    rem.u64 %rd13, %rd12, %rd6;   // oh
    div.u64 %rd14, %rd12, %rd6;   // b

    rem.u64 %rd15, %rd10, %rd5;   // kx
    div.u64 %rd16, %rd10, %rd5;
    rem.u64 %rd17, %rd16, %rd4;   // ky
    div.u64 %rd18, %rd16, %rd4;   // ch

    ld.param.u64 %rd19, [p_sh];
    ld.param.u64 %rd20, [p_dh];
    ld.param.u64 %rd21, [p_ph];
    mul.lo.u64 %rd22, %rd13, %rd19;
    mul.lo.u64 %rd23, %rd17, %rd20;
    add.u64 %rd22, %rd22, %rd23;
    sub.s64 %rd22, %rd22, %rd21;  // iy (signed)

    ld.param.u64 %rd19, [p_sw];
    ld.param.u64 %rd20, [p_dw];
    ld.param.u64 %rd21, [p_pw];
    mul.lo.u64 %rd23, %rd11, %rd19;
    mul.lo.u64 %rd24, %rd15, %rd20;
    add.u64 %rd23, %rd23, %rd24;
    sub.s64 %rd23, %rd23, %rd21;  // ix (signed)

    ld.param.u64 %rd24, [p_h];
    ld.param.u64 %rd25, [p_w];
    mov.f32 %f1, 0f00000000;
    setp.lt.s64 %p2, %rd22, 0;
    setp.ge.s64 %p3, %rd22, %rd24;
    setp.lt.s64 %p4, %rd23, 0;
    setp.ge.s64 %p5, %rd23, %rd25;
    or.pred %p2, %p2, %p3;
    or.pred %p4, %p4, %p5;
    or.pred %p2, %p2, %p4;
    @%p2 bra IC_STORE;

    // in_idx = ((b * c + ch) * h + iy) * w + ix
    mul.lo.u64 %rd26, %rd14, %rd3;
    add.u64 %rd26, %rd26, %rd18;
    mul.lo.u64 %rd26, %rd26, %rd24;
    add.u64 %rd26, %rd26, %rd22;
    mul.lo.u64 %rd26, %rd26, %rd25;
    add.u64 %rd26, %rd26, %rd23;
    shl.b64 %rd26, %rd26, 2;
    ld.param.u64 %rd27, [p_in];
    add.u64 %rd27, %rd27, %rd26;
    ld.global.f32 %f1, [%rd27];

IC_STORE:
    ld.param.u64 %rd27, [p_col];
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd27, %rd27, %rd1;
    st.global.f32 [%rd27], %f1;
IC_DONE:
    ret;
}

.visible .entry ochre_col2im_f32(
    .param .u64 p_col,
    .param .u64 p_img,
    .param .u64 p_total_in,
    .param .u64 p_c,
    .param .u64 p_h,
    .param .u64 p_w,
    .param .u64 p_kh,
    .param .u64 p_kw,
    .param .u64 p_sh,
    .param .u64 p_sw,
    .param .u64 p_ph,
    .param .u64 p_pw,
    .param .u64 p_dh,
    .param .u64 p_dw,
    .param .u64 p_oh,
    .param .u64 p_ow
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1; // idx in img
    ld.param.u64 %rd2, [p_total_in];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra C2I_DONE;

    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_h];
    ld.param.u64 %rd5, [p_w];
    rem.u64 %rd6, %rd1, %rd5;     // ix
    div.u64 %rd7, %rd1, %rd5;
    rem.u64 %rd8, %rd7, %rd4;     // iy
    div.u64 %rd7, %rd7, %rd4;
    rem.u64 %rd9, %rd7, %rd3;     // ch
    div.u64 %rd10, %rd7, %rd3;    // b

    ld.param.u64 %rd11, [p_kh];
    ld.param.u64 %rd12, [p_kw];
    ld.param.u64 %rd13, [p_sh];
    ld.param.u64 %rd14, [p_sw];
    ld.param.u64 %rd15, [p_ph];
    ld.param.u64 %rd16, [p_pw];
    ld.param.u64 %rd17, [p_dh];
    ld.param.u64 %rd18, [p_dw];
    ld.param.u64 %rd19, [p_oh];
    ld.param.u64 %rd20, [p_ow];
    ld.param.u64 %rd21, [p_col];

    mul.lo.u64 %rd22, %rd3, %rd11;
    mul.lo.u64 %rd22, %rd22, %rd12; // col_width = c*kh*kw

    mov.f32 %f1, 0f00000000;
    mov.u64 %rd23, 0; // ky
C2I_KY:
    setp.ge.u64 %p2, %rd23, %rd11;
    @%p2 bra C2I_END;
    // num_y = iy + ph - ky * dh
    mul.lo.u64 %rd24, %rd23, %rd17;
    add.u64 %rd25, %rd8, %rd15;
    sub.s64 %rd25, %rd25, %rd24;
    setp.lt.s64 %p3, %rd25, 0;
    @%p3 bra C2I_NEXT_KY;
    rem.u64 %rd26, %rd25, %rd13;
    setp.ne.u64 %p4, %rd26, 0;
    @%p4 bra C2I_NEXT_KY;
    div.u64 %rd25, %rd25, %rd13; // oh
    setp.ge.u64 %p5, %rd25, %rd19;
    @%p5 bra C2I_NEXT_KY;

    mov.u64 %rd27, 0; // kx
C2I_KX:
    setp.ge.u64 %p6, %rd27, %rd12;
    @%p6 bra C2I_NEXT_KY;
    mul.lo.u64 %rd24, %rd27, %rd18;
    add.u64 %rd28, %rd6, %rd16;
    sub.s64 %rd28, %rd28, %rd24;
    setp.lt.s64 %p7, %rd28, 0;
    @%p7 bra C2I_NEXT_KX;
    rem.u64 %rd26, %rd28, %rd14;
    setp.ne.u64 %p8, %rd26, 0;
    @%p8 bra C2I_NEXT_KX;
    div.u64 %rd28, %rd28, %rd14; // ow
    setp.ge.u64 %p7, %rd28, %rd20;
    @%p7 bra C2I_NEXT_KX;

    // row = (b * oh_dim + oh) * ow_dim + ow
    mul.lo.u64 %rd29, %rd10, %rd19;
    add.u64 %rd29, %rd29, %rd25;
    mul.lo.u64 %rd29, %rd29, %rd20;
    add.u64 %rd29, %rd29, %rd28;
    // col = (ch * kh + ky) * kw + kx
    mul.lo.u64 %rd30, %rd9, %rd11;
    add.u64 %rd30, %rd30, %rd23;
    mul.lo.u64 %rd30, %rd30, %rd12;
    add.u64 %rd30, %rd30, %rd27;

    mul.lo.u64 %rd31, %rd29, %rd22;
    add.u64 %rd31, %rd31, %rd30;
    shl.b64 %rd31, %rd31, 2;
    add.u64 %rd31, %rd21, %rd31;
    ld.global.f32 %f2, [%rd31];
    add.f32 %f1, %f1, %f2;

C2I_NEXT_KX:
    add.u64 %rd27, %rd27, 1;
    bra C2I_KX;
C2I_NEXT_KY:
    add.u64 %rd23, %rd23, 1;
    bra C2I_KY;

C2I_END:
    ld.param.u64 %rd31, [p_img];
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd31, %rd31, %rd1;
    st.global.f32 [%rd31], %f1;
C2I_DONE:
    ret;
}

.visible .entry ochre_max_pool2d_f32(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u64 p_argmax,
    .param .u64 p_total_out,
    .param .u64 p_c,
    .param .u64 p_h,
    .param .u64 p_w,
    .param .u64 p_kh,
    .param .u64 p_kw,
    .param .u64 p_sh,
    .param .u64 p_sw,
    .param .u64 p_ph,
    .param .u64 p_pw,
    .param .u64 p_oh,
    .param .u64 p_ow
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<26>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total_out];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra MP_DONE;

    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_h];
    ld.param.u64 %rd5, [p_w];
    ld.param.u64 %rd6, [p_oh];
    ld.param.u64 %rd7, [p_ow];

    rem.u64 %rd8, %rd1, %rd7;      // ow
    div.u64 %rd9, %rd1, %rd7;
    rem.u64 %rd10, %rd9, %rd6;     // oh
    div.u64 %rd9, %rd9, %rd6;
    rem.u64 %rd11, %rd9, %rd3;     // ch
    div.u64 %rd12, %rd9, %rd3;     // b

    ld.param.u64 %rd13, [p_kh];
    ld.param.u64 %rd14, [p_kw];
    ld.param.u64 %rd15, [p_sh];
    ld.param.u64 %rd16, [p_sw];
    ld.param.u64 %rd17, [p_ph];
    ld.param.u64 %rd18, [p_pw];
    ld.param.u64 %rd19, [p_in];

    mov.f32 %f1, 0fFF800000;       // best_val = -inf
    mov.u64 %rd20, 18446744073709551615; // u64::MAX sentinel

    mov.u64 %rd21, 0; // ky
MP_KY:
    setp.ge.u64 %p2, %rd21, %rd13;
    @%p2 bra MP_END;
    mul.lo.u64 %rd22, %rd10, %rd15;
    add.u64 %rd22, %rd22, %rd21;
    sub.s64 %rd22, %rd22, %rd17; // iy
    setp.lt.s64 %p3, %rd22, 0;
    setp.ge.s64 %p4, %rd22, %rd4;
    or.pred %p3, %p3, %p4;
    @%p3 bra MP_NEXT_KY;

    mov.u64 %rd23, 0; // kx
MP_KX:
    setp.ge.u64 %p5, %rd23, %rd14;
    @%p5 bra MP_NEXT_KY;
    mul.lo.u64 %rd24, %rd8, %rd16;
    add.u64 %rd24, %rd24, %rd23;
    sub.s64 %rd24, %rd24, %rd18; // ix
    setp.lt.s64 %p6, %rd24, 0;
    setp.ge.s64 %p7, %rd24, %rd5;
    or.pred %p6, %p6, %p7;
    @%p6 bra MP_NEXT_KX;

    mul.lo.u64 %rd25, %rd12, %rd3;
    add.u64 %rd25, %rd25, %rd11;
    mul.lo.u64 %rd25, %rd25, %rd4;
    add.u64 %rd25, %rd25, %rd22;
    mul.lo.u64 %rd25, %rd25, %rd5;
    add.u64 %rd25, %rd25, %rd24; // in_idx

    shl.b64 %rd24, %rd25, 2;
    add.u64 %rd24, %rd19, %rd24;
    ld.global.f32 %f2, [%rd24];
    setp.gt.f32 %p8, %f2, %f1;
    @%p8 mov.f32 %f1, %f2;
    @%p8 mov.u64 %rd20, %rd25;

MP_NEXT_KX:
    add.u64 %rd23, %rd23, 1;
    bra MP_KX;
MP_NEXT_KY:
    add.u64 %rd21, %rd21, 1;
    bra MP_KY;

MP_END:
    ld.param.u64 %rd24, [p_out];
    shl.b64 %rd25, %rd1, 2;
    add.u64 %rd24, %rd24, %rd25;
    st.global.f32 [%rd24], %f1;

    ld.param.u64 %rd24, [p_argmax];
    shl.b64 %rd25, %rd1, 3;
    add.u64 %rd24, %rd24, %rd25;
    st.global.u64 [%rd24], %rd20;
MP_DONE:
    ret;
}

.visible .entry ochre_avg_pool2d_f32(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u64 p_total_out,
    .param .u64 p_c,
    .param .u64 p_h,
    .param .u64 p_w,
    .param .u64 p_kh,
    .param .u64 p_kw,
    .param .u64 p_sh,
    .param .u64 p_sw,
    .param .u64 p_ph,
    .param .u64 p_pw,
    .param .u64 p_oh,
    .param .u64 p_ow
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<26>;
    .reg .f32 %f<4>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total_out];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra AP_DONE;

    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_h];
    ld.param.u64 %rd5, [p_w];
    ld.param.u64 %rd6, [p_oh];
    ld.param.u64 %rd7, [p_ow];

    rem.u64 %rd8, %rd1, %rd7;
    div.u64 %rd9, %rd1, %rd7;
    rem.u64 %rd10, %rd9, %rd6;
    div.u64 %rd9, %rd9, %rd6;
    rem.u64 %rd11, %rd9, %rd3;
    div.u64 %rd12, %rd9, %rd3;

    ld.param.u64 %rd13, [p_kh];
    ld.param.u64 %rd14, [p_kw];
    ld.param.u64 %rd15, [p_sh];
    ld.param.u64 %rd16, [p_sw];
    ld.param.u64 %rd17, [p_ph];
    ld.param.u64 %rd18, [p_pw];
    ld.param.u64 %rd19, [p_in];

    mov.f32 %f1, 0f00000000;
    mov.u64 %rd21, 0;
AP_KY:
    setp.ge.u64 %p2, %rd21, %rd13;
    @%p2 bra AP_END;
    mul.lo.u64 %rd22, %rd10, %rd15;
    add.u64 %rd22, %rd22, %rd21;
    sub.s64 %rd22, %rd22, %rd17;
    setp.lt.s64 %p3, %rd22, 0;
    setp.ge.s64 %p4, %rd22, %rd4;
    or.pred %p3, %p3, %p4;
    @%p3 bra AP_NEXT_KY;

    mov.u64 %rd23, 0;
AP_KX:
    setp.ge.u64 %p5, %rd23, %rd14;
    @%p5 bra AP_NEXT_KY;
    mul.lo.u64 %rd24, %rd8, %rd16;
    add.u64 %rd24, %rd24, %rd23;
    sub.s64 %rd24, %rd24, %rd18;
    setp.lt.s64 %p6, %rd24, 0;
    setp.ge.s64 %p7, %rd24, %rd5;
    or.pred %p6, %p6, %p7;
    @%p6 bra AP_NEXT_KX;

    mul.lo.u64 %rd25, %rd12, %rd3;
    add.u64 %rd25, %rd25, %rd11;
    mul.lo.u64 %rd25, %rd25, %rd4;
    add.u64 %rd25, %rd25, %rd22;
    mul.lo.u64 %rd25, %rd25, %rd5;
    add.u64 %rd25, %rd25, %rd24;
    shl.b64 %rd24, %rd25, 2;
    add.u64 %rd24, %rd19, %rd24;
    ld.global.f32 %f2, [%rd24];
    add.f32 %f1, %f1, %f2;

AP_NEXT_KX:
    add.u64 %rd23, %rd23, 1;
    bra AP_KX;
AP_NEXT_KY:
    add.u64 %rd21, %rd21, 1;
    bra AP_KY;

AP_END:
    mul.lo.u64 %rd24, %rd13, %rd14;
    cvt.rn.f32.u64 %f3, %rd24;
    div.rn.f32 %f1, %f1, %f3;
    ld.param.u64 %rd24, [p_out];
    shl.b64 %rd25, %rd1, 2;
    add.u64 %rd24, %rd24, %rd25;
    st.global.f32 [%rd24], %f1;
AP_DONE:
    ret;
}
"#;

fn load_cublas_library() -> Option<Library> {
    #[cfg(target_os = "windows")]
    let candidates = ["cublas64_12.dll", "cublas64_11.dll", "cublas64_10.dll"];
    #[cfg(not(target_os = "windows"))]
    let candidates = ["libcublas.so.12", "libcublas.so.11", "libcublas.so"];

    for name in candidates {
        // SAFETY: Loading standard system cuBLAS library by SONAME.
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

fn init_nn_runtime() -> Option<CudaNnRuntime> {
    let driver = get_driver()?;
    let cublas_lib = load_cublas_library()?;

    // SAFETY: Resolving standard CUDA Driver PTX and cuBLAS C ABI symbols.
    unsafe {
        let cublas_create = *cublas_lib
            .get::<FnCublasCreate>(b"cublasCreate_v2\0")
            .ok()?;
        let cublas_destroy = *cublas_lib
            .get::<FnCublasDestroy>(b"cublasDestroy_v2\0")
            .ok()?;
        let cublas_sgemm = *cublas_lib.get::<FnCublasSgemm>(b"cublasSgemm_v2\0").ok()?;
        let cublas_sgemm_strided_batched = cublas_lib
            .get::<FnCublasSgemmStridedBatched>(b"cublasSgemmStridedBatched\0")
            .ok()
            .map(|s| *s);
        let cublas_gemm_ex = cublas_lib
            .get::<FnCublasGemmEx>(b"cublasGemmEx\0")
            .ok()
            .map(|s| *s);

        let cu_module_load_data = *driver
            ._cuda_lib
            .get::<FnCuModuleLoadData>(b"cuModuleLoadData\0")
            .ok()?;
        let cu_module_get_function = *driver
            ._cuda_lib
            .get::<FnCuModuleGetFunction>(b"cuModuleGetFunction\0")
            .ok()?;
        let cu_launch_kernel = *driver
            ._cuda_lib
            .get::<FnCuLaunchKernel>(b"cuLaunchKernel\0")
            .ok()?;
        let cu_ctx_synchronize = *driver
            ._cuda_lib
            .get::<FnCuCtxSynchronize>(b"cuCtxSynchronize\0")
            .ok()?;

        Some(CudaNnRuntime {
            _cublas_lib: cublas_lib,
            cublas_create,
            cublas_destroy,
            cublas_sgemm,
            cublas_sgemm_strided_batched,
            cublas_gemm_ex,
            cu_module_load_data,
            cu_module_get_function,
            cu_launch_kernel,
            cu_ctx_synchronize,
            contexts: Mutex::new(HashMap::new()),
        })
    }
}

fn get_nn_runtime() -> Option<&'static CudaNnRuntime> {
    CUDA_NN_RUNTIME.get_or_init(init_nn_runtime).as_ref()
}

fn load_ptx_funcs(rt: &CudaNnRuntime) -> Option<NnPtxFuncs> {
    let ptx_c = CString::new(OCHRE_NN_PTX).ok()?;
    let mut module: CuModule = std::ptr::null_mut();
    // SAFETY: `ptx_c` is null-terminated PTX source and current CUDA context is bound.
    let rc = unsafe { (rt.cu_module_load_data)(&mut module, ptx_c.as_ptr().cast()) };
    if rc != CUDA_SUCCESS || module.is_null() {
        return None;
    }

    let get_fn = |name: &str| -> Option<CuFunction> {
        let cname = CString::new(name).ok()?;
        let mut func: CuFunction = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded CUmodule on the active context.
        let frc = unsafe { (rt.cu_module_get_function)(&mut func, module, cname.as_ptr()) };
        if frc == CUDA_SUCCESS && !func.is_null() {
            Some(func)
        } else {
            None
        }
    };

    Some(NnPtxFuncs {
        _module: module,
        gelu: get_fn("ochre_gelu_f32")?,
        gelu_bwd: get_fn("ochre_gelu_bwd_f32")?,
        silu: get_fn("ochre_silu_f32")?,
        silu_bwd: get_fn("ochre_silu_bwd_f32")?,
        row_layernorm: get_fn("ochre_row_layernorm_f32")?,
        row_rmsnorm: get_fn("ochre_row_rmsnorm_f32")?,
        causal_mask: get_fn("ochre_causal_mask_f32")?,
        cross_entropy_grad: get_fn("ochre_cross_entropy_grad_f32")?,
        adamw_step: get_fn("ochre_adamw_step_f32")?,
        im2col: get_fn("ochre_im2col_f32")?,
        col2im: get_fn("ochre_col2im_f32")?,
        max_pool2d: get_fn("ochre_max_pool2d_f32")?,
        avg_pool2d: get_fn("ochre_avg_pool2d_f32")?,
    })
}

fn with_nn_ctx<R>(
    gpu_id: usize,
    f: impl FnOnce(&CudaDriver, &CudaNnRuntime, &mut CudaNnCtxState) -> OchreResult<R>,
) -> Option<OchreResult<R>> {
    let driver = get_driver()?;
    let rt = get_nn_runtime()?;
    let ctx_arc = driver.get_device_ctx(gpu_id).ok()?;
    let cu_ctx = {
        let guard = ctx_arc.lock().ok()?;
        guard.ctx
    };

    let ctx_mutex: &'static Mutex<CudaNnCtxState> = {
        let mut map = rt.contexts.lock().ok()?;
        let key = cu_ctx as usize;
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(key) {
            // SAFETY: Bind `cu_ctx` before creating cuBLAS handle and loading PTX module.
            let set_rc = unsafe { (driver.cu_ctx_set_current)(cu_ctx) };
            if set_rc != CUDA_SUCCESS {
                return None;
            }
            let mut handle: CublasHandle = std::ptr::null_mut();
            // SAFETY: Initializes cuBLAS handle on bound context.
            let blas_rc = unsafe { (rt.cublas_create)(&mut handle) };
            if blas_rc != CUBLAS_STATUS_SUCCESS || handle.is_null() {
                return None;
            }
            let ptx = load_ptx_funcs(rt);
            let state = CudaNnCtxState {
                handle,
                cublas_destroy: rt.cublas_destroy,
                workspace: CudaNnWorkspace::new(driver, cu_ctx),
                ptx,
            };
            e.insert(Box::new(Mutex::new(state)));
        }
        let boxed = map.get(&key)?;
        // SAFETY: Entries in `rt.contexts` live for `'static` and are never removed.
        unsafe { &*(boxed.as_ref() as *const Mutex<CudaNnCtxState>) }
    };

    let mut state = ctx_mutex.lock().ok()?;
    // SAFETY: Bind CUDA context for this calling thread.
    let rc = unsafe { (driver.cu_ctx_set_current)(state.workspace.ctx) };
    if rc != CUDA_SUCCESS {
        return Some(Err(OchreError::DeviceError(format!(
            "CUDA cuCtxSetCurrent failed with code {rc} ( >_< )"
        ))));
    }
    Some(f(driver, rt, &mut state))
}

#[inline]
fn htod_bytes(driver: &CudaDriver, dst: CuDevicePtr, src: &[u8]) -> OchreResult<()> {
    if !src.is_empty() {
        // SAFETY: `dst` is a valid CUDA device pointer with at least `src.len()` bytes.
        let rc = unsafe { (driver.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len()) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA NN cuMemcpyHtoD failed with code {rc} (´；ω；`)"
            )));
        }
    }
    Ok(())
}

#[inline]
fn dtoh_bytes(driver: &CudaDriver, dst: &mut [u8], src: CuDevicePtr) -> OchreResult<()> {
    if !dst.is_empty() {
        // SAFETY: `src` is a valid CUDA device pointer and `dst` has `dst.len()` capacity.
        let rc = unsafe { (driver.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len()) };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA NN cuMemcpyDtoH failed with code {rc} (´；ω；`)"
            )));
        }
    }
    Ok(())
}

#[inline]
fn htod_f32(driver: &CudaDriver, dst: CuDevicePtr, src: &[f32]) -> OchreResult<()> {
    // SAFETY: `f32` has no padding and valid byte representation for device transfer.
    let bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
    };
    htod_bytes(driver, dst, bytes)
}

#[inline]
fn dtoh_f32(driver: &CudaDriver, dst: &mut [f32], src: CuDevicePtr) -> OchreResult<()> {
    // SAFETY: `dst` is a valid mutable `f32` slice of `dst.len() * 4` bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(dst))
    };
    dtoh_bytes(driver, bytes, src)
}

fn check_cublas(status: CublasStatus, op: &str) -> OchreResult<()> {
    if status != CUBLAS_STATUS_SUCCESS {
        return Err(OchreError::DeviceError(format!(
            "cuBLAS {op} failed with status {status} (´；ω；`)"
        )));
    }
    Ok(())
}

fn launch_1d(
    rt: &CudaNnRuntime,
    func: CuFunction,
    total_threads: usize,
    block_x: u32,
    params: &mut [*mut c_void],
) -> OchreResult<()> {
    if total_threads == 0 {
        return Ok(());
    }
    let grid_x = ((total_threads as u64).div_ceil(block_x as u64)).min(u32::MAX as u64) as u32;
    // SAFETY: Caller supplies kernel parameters matching `func`'s PTX signature.
    let rc = unsafe {
        (rt.cu_launch_kernel)(
            func,
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
        )
    };
    if rc != CUDA_SUCCESS {
        return Err(OchreError::DeviceError(format!(
            "CUDA cuLaunchKernel failed with code {rc} ( >_< )"
        )));
    }
    // SAFETY: Synchronize default stream on bound context before host readback.
    let sync_rc = unsafe { (rt.cu_ctx_synchronize)() };
    if sync_rc != CUDA_SUCCESS {
        return Err(OchreError::DeviceError(format!(
            "CUDA cuCtxSynchronize failed with code {sync_rc} ( >_< )"
        )));
    }
    Ok(())
}

/// N7: CUDA cuBLAS SGEMM (`C = alpha * op(A) * op(B) + beta * C`).
#[allow(clippy::too_many_arguments)]
pub fn cuda_sgemm_f32(
    gpu_id: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) -> Option<OchreResult<()>> {
    if m.saturating_mul(n).saturating_mul(k) < MIN_CUDA_GEMM_OPS || m == 0 || n == 0 || k == 0 {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let a_elems = m * k;
        let b_elems = k * n;
        let c_elems = m * n;
        let off_a = 0usize;
        let off_b = align256(a_elems * 4);
        let off_c = off_b + align256(b_elems * 4);
        let total_bytes = off_c + align256(c_elems * 4);

        let base = state.workspace.ensure_bytes(driver, total_bytes)?;
        let d_a = base + off_a as u64;
        let d_b = base + off_b as u64;
        let d_c = base + off_c as u64;

        htod_f32(driver, d_a, &a[..a_elems])?;
        htod_f32(driver, d_b, &b[..b_elems])?;
        if beta != 0.0 {
            htod_f32(driver, d_c, &c[..c_elems])?;
        }

        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };

        // SAFETY: All device pointers lie within persistent workspace on the bound context.
        let status = unsafe {
            (rt.cublas_sgemm)(
                state.handle,
                op_b,
                op_a,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                d_b,
                ldb,
                d_a,
                lda,
                &beta,
                d_c,
                n as i32,
            )
        };
        check_cublas(status, "cublasSgemm_v2")?;
        dtoh_f32(driver, &mut c[..c_elems], d_c)?;
        Ok(())
    })
}

/// N7: CUDA `cublasSgemmStridedBatched` for batched f32 matmul.
#[allow(clippy::too_many_arguments)]
pub fn cuda_batched_matmul_f32(
    gpu_id: usize,
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
        || batch.saturating_mul(m).saturating_mul(n).saturating_mul(k) < MIN_CUDA_GEMM_OPS
    {
        return None;
    }
    if batch == 1 {
        let mut out = vec![0.0f32; m * n];
        let res = cuda_sgemm_f32(gpu_id, m, n, k, a, b, &mut out, trans_a, trans_b, 1.0, 0.0)?;
        return Some(res.map(|()| out));
    }

    with_nn_ctx(gpu_id, |driver, rt, state| {
        let strided_fn = rt.cublas_sgemm_strided_batched.ok_or_else(|| {
            OchreError::DeviceError("cublasSgemmStridedBatched not available ( >_< )".to_string())
        })?;
        let stride_a = m * k;
        let stride_b = k * n;
        let stride_c = m * n;
        let a_elems = batch * stride_a;
        let b_elems = batch * stride_b;
        let c_elems = batch * stride_c;

        let off_a = 0usize;
        let off_b = align256(a_elems * 4);
        let off_c = off_b + align256(b_elems * 4);
        let total_bytes = off_c + align256(c_elems * 4);

        let base = state.workspace.ensure_bytes(driver, total_bytes)?;
        let d_a = base + off_a as u64;
        let d_b = base + off_b as u64;
        let d_c = base + off_c as u64;

        htod_f32(driver, d_a, &a[..a_elems])?;
        htod_f32(driver, d_b, &b[..b_elems])?;

        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // SAFETY: Strided batched buffers fit inside `state.workspace`.
        let status = unsafe {
            strided_fn(
                state.handle,
                op_b,
                op_a,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                d_b,
                ldb,
                stride_b as i64,
                d_a,
                lda,
                stride_a as i64,
                &beta,
                d_c,
                n as i32,
                stride_c as i64,
                batch as i32,
            )
        };
        check_cublas(status, "cublasSgemmStridedBatched")?;
        let mut out = vec![0.0f32; c_elems];
        dtoh_f32(driver, &mut out, d_c)?;
        Ok(out)
    })
}

/// Tensor-Core `cublasGemmEx` supporting `DType::BF16`, `DType::F16`, and `DType::F32`
/// with `CUBLAS_COMPUTE_32F` accumulation into `Vec<f32>`.
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_ex(
    gpu_id: usize,
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    b_bytes: &[u8],
    dtype: DType,
    trans_a: bool,
    trans_b: bool,
) -> Option<OchreResult<Vec<f32>>> {
    if m.saturating_mul(n).saturating_mul(k) < MIN_CUDA_GEMM_OPS || m == 0 || n == 0 || k == 0 {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let gemm_ex = rt.cublas_gemm_ex.ok_or_else(|| {
            OchreError::DeviceError("cublasGemmEx not available ( >_< )".to_string())
        })?;
        let elem_sz = dtype.size_bytes();
        let req_a = m * k * elem_sz;
        let req_b = k * n * elem_sz;
        if a_bytes.len() < req_a || b_bytes.len() < req_b {
            return Err(OchreError::InvalidParameter(
                "cuda_gemm_ex input byte slice too small ( >_< )".to_string(),
            ));
        }

        let cuda_ty = match dtype {
            DType::F32 => CUDA_R_32F,
            DType::BF16 => CUDA_R_16BF,
            DType::F16 => CUDA_R_16F,
        };
        let off_a = 0usize;
        let off_b = align256(req_a);
        let off_c = off_b + align256(req_b);
        let total_bytes = off_c + align256(m * n * 4);

        let base = state.workspace.ensure_bytes(driver, total_bytes)?;
        let d_a = base + off_a as u64;
        let d_b = base + off_b as u64;
        let d_c = base + off_c as u64;

        htod_bytes(driver, d_a, &a_bytes[..req_a])?;
        htod_bytes(driver, d_b, &b_bytes[..req_b])?;

        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // SAFETY: `d_a`, `d_b`, `d_c` are properly aligned inside persistent VRAM workspace.
        let status = unsafe {
            gemm_ex(
                state.handle,
                op_b,
                op_a,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                d_b,
                cuda_ty,
                ldb,
                d_a,
                cuda_ty,
                lda,
                &beta,
                d_c,
                CUDA_R_32F,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        check_cublas(status, "cublasGemmEx")?;
        let mut out = vec![0.0f32; m * n];
        dtoh_f32(driver, &mut out, d_c)?;
        Ok(out)
    })
}

/// N8: CUDA GELU forward.
pub fn cuda_gelu_f32(gpu_id: usize, x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if x.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .gelu;
        let n = x.len();
        let off_out = align256(n * 4);
        let base = state
            .workspace
            .ensure_bytes(driver, off_out + align256(n * 4))?;
        let mut d_x = base;
        let mut d_out = base + off_out as u64;
        let mut n_u64 = n as u64;
        htod_f32(driver, d_x, x)?;
        let mut params: [*mut c_void; 3] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, n, 256, &mut params)?;
        let mut out = vec![0.0f32; n];
        dtoh_f32(driver, &mut out, d_out)?;
        Ok(out)
    })
}

/// N8: CUDA GELU backward.
pub fn cuda_gelu_bwd_f32(gpu_id: usize, dy: &[f32], x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if x.len() < MIN_CUDA_NN_ELEMENTS || dy.len() != x.len() {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .gelu_bwd;
        let n = x.len();
        let stride = align256(n * 4);
        let base = state.workspace.ensure_bytes(driver, stride * 3)?;
        let mut d_dy = base;
        let mut d_x = base + stride as u64;
        let mut d_dx = base + (stride * 2) as u64;
        let mut n_u64 = n as u64;
        htod_f32(driver, d_dy, dy)?;
        htod_f32(driver, d_x, x)?;
        let mut params: [*mut c_void; 4] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_dx as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, n, 256, &mut params)?;
        let mut out = vec![0.0f32; n];
        dtoh_f32(driver, &mut out, d_dx)?;
        Ok(out)
    })
}

/// N8: CUDA SiLU forward.
pub fn cuda_silu_f32(gpu_id: usize, x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if x.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .silu;
        let n = x.len();
        let stride = align256(n * 4);
        let base = state.workspace.ensure_bytes(driver, stride * 2)?;
        let mut d_x = base;
        let mut d_out = base + stride as u64;
        let mut n_u64 = n as u64;
        htod_f32(driver, d_x, x)?;
        let mut params: [*mut c_void; 3] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, n, 256, &mut params)?;
        let mut out = vec![0.0f32; n];
        dtoh_f32(driver, &mut out, d_out)?;
        Ok(out)
    })
}

/// N8: CUDA SiLU backward.
pub fn cuda_silu_bwd_f32(gpu_id: usize, dy: &[f32], x: &[f32]) -> Option<OchreResult<Vec<f32>>> {
    if x.len() < MIN_CUDA_NN_ELEMENTS || dy.len() != x.len() {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .silu_bwd;
        let n = x.len();
        let stride = align256(n * 4);
        let base = state.workspace.ensure_bytes(driver, stride * 3)?;
        let mut d_dy = base;
        let mut d_x = base + stride as u64;
        let mut d_dx = base + (stride * 2) as u64;
        let mut n_u64 = n as u64;
        htod_f32(driver, d_dy, dy)?;
        htod_f32(driver, d_x, x)?;
        let mut params: [*mut c_void; 4] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_dx as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, n, 256, &mut params)?;
        let mut out = vec![0.0f32; n];
        dtoh_f32(driver, &mut out, d_dx)?;
        Ok(out)
    })
}

/// N8: CUDA row LayerNorm forward.
#[allow(clippy::type_complexity)]
pub fn cuda_row_layernorm_f32(
    gpu_id: usize,
    x: &[f32],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if rows * cols < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .row_layernorm;
        let off_x = 0usize;
        let off_g = off_x + align256(x.len() * 4);
        let off_b = off_g + align256(cols * 4);
        let off_out = off_b + align256(cols * 4);
        let off_mean = off_out + align256(x.len() * 4);
        let off_rstd = off_mean + align256(rows * 4);
        let total = off_rstd + align256(rows * 4);

        let base = state.workspace.ensure_bytes(driver, total)?;
        let mut d_x = base + off_x as u64;
        let mut d_g = base + off_g as u64;
        let mut d_b = base + off_b as u64;
        let mut d_out = base + off_out as u64;
        let mut d_mean = base + off_mean as u64;
        let mut d_rstd = base + off_rstd as u64;

        htod_f32(driver, d_x, x)?;
        if let Some(g) = gamma {
            htod_f32(driver, d_g, g)?;
        }
        if let Some(b) = beta {
            htod_f32(driver, d_b, b)?;
        }

        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut has_g: u32 = u32::from(gamma.is_some());
        let mut has_b: u32 = u32::from(beta.is_some());
        let mut eps_f32 = eps;
        let mut params: [*mut c_void; 11] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_g as *mut CuDevicePtr).cast(),
            (&mut d_b as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_mean as *mut CuDevicePtr).cast(),
            (&mut d_rstd as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
            (&mut has_g as *mut u32).cast(),
            (&mut has_b as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
        ];
        launch_1d(rt, func, rows, 128, &mut params)?;

        let mut out = vec![0.0f32; x.len()];
        let mut mean = vec![0.0f32; rows];
        let mut rstd = vec![0.0f32; rows];
        dtoh_f32(driver, &mut out, d_out)?;
        dtoh_f32(driver, &mut mean, d_mean)?;
        dtoh_f32(driver, &mut rstd, d_rstd)?;
        Ok((out, mean, rstd))
    })
}

/// N8: CUDA row RMSNorm forward.
pub fn cuda_row_rmsnorm_f32(
    gpu_id: usize,
    x: &[f32],
    gamma: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>)>> {
    if rows * cols < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .row_rmsnorm;
        let off_x = 0usize;
        let off_g = off_x + align256(x.len() * 4);
        let off_out = off_g + align256(cols * 4);
        let off_rstd = off_out + align256(x.len() * 4);
        let total = off_rstd + align256(rows * 4);

        let base = state.workspace.ensure_bytes(driver, total)?;
        let mut d_x = base + off_x as u64;
        let mut d_g = base + off_g as u64;
        let mut d_out = base + off_out as u64;
        let mut d_rstd = base + off_rstd as u64;

        htod_f32(driver, d_x, x)?;
        if let Some(g) = gamma {
            htod_f32(driver, d_g, g)?;
        }

        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut has_g: u32 = u32::from(gamma.is_some());
        let mut eps_f32 = eps;
        let mut params: [*mut c_void; 8] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_g as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_rstd as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
            (&mut has_g as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
        ];
        launch_1d(rt, func, rows, 128, &mut params)?;

        let mut out = vec![0.0f32; x.len()];
        let mut rstd = vec![0.0f32; rows];
        dtoh_f32(driver, &mut out, d_out)?;
        dtoh_f32(driver, &mut rstd, d_rstd)?;
        Ok((out, rstd))
    })
}

/// N8: CUDA causal mask in-place.
pub fn cuda_causal_mask_f32(
    gpu_id: usize,
    scores: &mut [f32],
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    offset: usize,
) -> Option<OchreResult<()>> {
    let total = batch_heads * q_len * k_len;
    if total < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .causal_mask;
        let mut d_scores = state.workspace.ensure_bytes(driver, total * 4)?;
        htod_f32(driver, d_scores, scores)?;
        let mut total_u64 = total as u64;
        let mut q_u64 = q_len as u64;
        let mut k_u64 = k_len as u64;
        let mut off_u64 = offset as u64;
        let mut params: [*mut c_void; 5] = [
            (&mut d_scores as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut q_u64 as *mut u64).cast(),
            (&mut k_u64 as *mut u64).cast(),
            (&mut off_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, total, 256, &mut params)?;
        dtoh_f32(driver, scores, d_scores)?;
        Ok(())
    })
}

/// N9: CUDA fused cross-entropy + gradient kernel.
pub fn cuda_cross_entropy_grad_f32(
    gpu_id: usize,
    logits: &[f32],
    targets: &[usize],
    rows: usize,
    cols: usize,
    label_smoothing: f32,
    ignore_index: Option<usize>,
) -> Option<OchreResult<(f32, Vec<f32>)>> {
    if rows * cols < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    let valid_count = targets.iter().filter(|&&t| Some(t) != ignore_index).count();
    if valid_count == 0 {
        return Some(Ok((0.0, vec![0.0; rows * cols])));
    }
    let targets_u64: Vec<u64> = targets.iter().map(|&t| t as u64).collect();
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .cross_entropy_grad;
        let off_logits = 0usize;
        let off_targets = off_logits + align256(logits.len() * 4);
        let off_dlogits = off_targets + align256(targets_u64.len() * 8);
        let off_losses = off_dlogits + align256(logits.len() * 4);
        let total_bytes = off_losses + align256(rows * 4);

        let base = state.workspace.ensure_bytes(driver, total_bytes)?;
        let mut d_logits = base + off_logits as u64;
        let mut d_targets = base + off_targets as u64;
        let mut d_dlogits = base + off_dlogits as u64;
        let mut d_losses = base + off_losses as u64;

        htod_f32(driver, d_logits, logits)?;
        // SAFETY: `targets_u64` is a valid contiguous `u64` slice.
        let t_bytes = unsafe {
            std::slice::from_raw_parts(
                targets_u64.as_ptr().cast::<u8>(),
                std::mem::size_of_val(targets_u64.as_slice()),
            )
        };
        htod_bytes(driver, d_targets, t_bytes)?;

        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut ls = label_smoothing;
        let mut ign_u64 = ignore_index.unwrap_or(usize::MAX) as u64;
        let mut has_ign = u32::from(ignore_index.is_some());
        let mut inv_valid = 1.0f32 / (valid_count as f32);

        let mut params: [*mut c_void; 10] = [
            (&mut d_logits as *mut CuDevicePtr).cast(),
            (&mut d_targets as *mut CuDevicePtr).cast(),
            (&mut d_dlogits as *mut CuDevicePtr).cast(),
            (&mut d_losses as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
            (&mut ls as *mut f32).cast(),
            (&mut ign_u64 as *mut u64).cast(),
            (&mut has_ign as *mut u32).cast(),
            (&mut inv_valid as *mut f32).cast(),
        ];
        launch_1d(rt, func, rows, 128, &mut params)?;

        let mut d_grad = vec![0.0f32; logits.len()];
        let mut losses = vec![0.0f32; rows];
        dtoh_f32(driver, &mut d_grad, d_dlogits)?;
        dtoh_f32(driver, &mut losses, d_losses)?;
        let sum_loss: f64 = losses.iter().map(|&v| v as f64).sum();
        let mean_loss = (sum_loss / valid_count as f64) as f32;
        Ok((mean_loss, d_grad))
    })
}

/// N9: CUDA fused AdamW step.
#[allow(clippy::too_many_arguments)]
pub fn cuda_adamw_step_f32(
    gpu_id: usize,
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
    if param.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .adamw_step;
        let n = param.len();
        let stride = align256(n * 4);
        let base = state.workspace.ensure_bytes(driver, stride * 4)?;
        let mut d_p = base;
        let mut d_g = base + stride as u64;
        let mut d_m = base + (stride * 2) as u64;
        let mut d_v = base + (stride * 3) as u64;

        htod_f32(driver, d_p, param)?;
        htod_f32(driver, d_g, grad)?;
        htod_f32(driver, d_m, m)?;
        htod_f32(driver, d_v, v)?;

        let inv_bc1 = (1.0 / (1.0 - (beta1 as f64).powi(step as i32))) as f32;
        let inv_bc2 = (1.0 / (1.0 - (beta2 as f64).powi(step as i32))) as f32;
        let mut n_u64 = n as u64;
        let mut lr_v = lr;
        let mut b1_v = beta1;
        let mut b2_v = beta2;
        let mut eps_v = eps;
        let mut wd_v = weight_decay;
        let mut ib1_v = inv_bc1;
        let mut ib2_v = inv_bc2;

        let mut params: [*mut c_void; 12] = [
            (&mut d_p as *mut CuDevicePtr).cast(),
            (&mut d_g as *mut CuDevicePtr).cast(),
            (&mut d_m as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut lr_v as *mut f32).cast(),
            (&mut b1_v as *mut f32).cast(),
            (&mut b2_v as *mut f32).cast(),
            (&mut eps_v as *mut f32).cast(),
            (&mut wd_v as *mut f32).cast(),
            (&mut ib1_v as *mut f32).cast(),
            (&mut ib2_v as *mut f32).cast(),
        ];
        launch_1d(rt, func, n, 256, &mut params)?;
        dtoh_f32(driver, param, d_p)?;
        dtoh_f32(driver, m, d_m)?;
        dtoh_f32(driver, v, d_v)?;
        Ok(())
    })
}

/// N10: CUDA im2col.
#[allow(clippy::too_many_arguments)]
pub fn cuda_im2col_f32(
    gpu_id: usize,
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
    let total_col = batch * out_h * out_w * c * kh * kw;
    if total_col < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .im2col;
        let off_col = align256(input.len() * 4);
        let base = state
            .workspace
            .ensure_bytes(driver, off_col + align256(total_col * 4))?;
        let mut d_in = base;
        let mut d_col = base + off_col as u64;
        htod_f32(driver, d_in, input)?;

        let mut total_u64 = total_col as u64;
        let mut c_u64 = c as u64;
        let mut h_u64 = h as u64;
        let mut w_u64 = w as u64;
        let mut kh_u64 = kh as u64;
        let mut kw_u64 = kw as u64;
        let mut sh_u64 = stride_h as u64;
        let mut sw_u64 = stride_w as u64;
        let mut ph_u64 = pad_h as u64;
        let mut pw_u64 = pad_w as u64;
        let mut dh_u64 = dil_h as u64;
        let mut dw_u64 = dil_w as u64;
        let mut oh_u64 = out_h as u64;
        let mut ow_u64 = out_w as u64;
        let mut params: [*mut c_void; 16] = [
            (&mut d_in as *mut CuDevicePtr).cast(),
            (&mut d_col as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut w_u64 as *mut u64).cast(),
            (&mut kh_u64 as *mut u64).cast(),
            (&mut kw_u64 as *mut u64).cast(),
            (&mut sh_u64 as *mut u64).cast(),
            (&mut sw_u64 as *mut u64).cast(),
            (&mut ph_u64 as *mut u64).cast(),
            (&mut pw_u64 as *mut u64).cast(),
            (&mut dh_u64 as *mut u64).cast(),
            (&mut dw_u64 as *mut u64).cast(),
            (&mut oh_u64 as *mut u64).cast(),
            (&mut ow_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, total_col, 256, &mut params)?;
        let mut out = vec![0.0f32; total_col];
        dtoh_f32(driver, &mut out, d_col)?;
        Ok(out)
    })
}

/// N10: CUDA col2im.
#[allow(clippy::too_many_arguments)]
pub fn cuda_col2im_f32(
    gpu_id: usize,
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
    let total_in = batch * c * h * w;
    if col.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .col2im;
        let off_img = align256(col.len() * 4);
        let base = state
            .workspace
            .ensure_bytes(driver, off_img + align256(total_in * 4))?;
        let mut d_col = base;
        let mut d_img = base + off_img as u64;
        htod_f32(driver, d_col, col)?;

        let mut total_u64 = total_in as u64;
        let mut c_u64 = c as u64;
        let mut h_u64 = h as u64;
        let mut w_u64 = w as u64;
        let mut kh_u64 = kh as u64;
        let mut kw_u64 = kw as u64;
        let mut sh_u64 = stride_h as u64;
        let mut sw_u64 = stride_w as u64;
        let mut ph_u64 = pad_h as u64;
        let mut pw_u64 = pad_w as u64;
        let mut dh_u64 = dil_h as u64;
        let mut dw_u64 = dil_w as u64;
        let mut oh_u64 = out_h as u64;
        let mut ow_u64 = out_w as u64;
        let mut params: [*mut c_void; 16] = [
            (&mut d_col as *mut CuDevicePtr).cast(),
            (&mut d_img as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut w_u64 as *mut u64).cast(),
            (&mut kh_u64 as *mut u64).cast(),
            (&mut kw_u64 as *mut u64).cast(),
            (&mut sh_u64 as *mut u64).cast(),
            (&mut sw_u64 as *mut u64).cast(),
            (&mut ph_u64 as *mut u64).cast(),
            (&mut pw_u64 as *mut u64).cast(),
            (&mut dh_u64 as *mut u64).cast(),
            (&mut dw_u64 as *mut u64).cast(),
            (&mut oh_u64 as *mut u64).cast(),
            (&mut ow_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, total_in, 256, &mut params)?;
        let mut out = vec![0.0f32; total_in];
        dtoh_f32(driver, &mut out, d_img)?;
        Ok(out)
    })
}

/// N10: CUDA max_pool2d.
#[allow(clippy::too_many_arguments)]
pub fn cuda_max_pool2d_f32(
    gpu_id: usize,
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
    let total_out = batch * c * out_h * out_w;
    if input.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .max_pool2d;
        let off_out = align256(input.len() * 4);
        let off_arg = off_out + align256(total_out * 4);
        let total_bytes = off_arg + align256(total_out * 8);
        let base = state.workspace.ensure_bytes(driver, total_bytes)?;
        let mut d_in = base;
        let mut d_out = base + off_out as u64;
        let mut d_arg = base + off_arg as u64;
        htod_f32(driver, d_in, input)?;

        let mut total_u64 = total_out as u64;
        let mut c_u64 = c as u64;
        let mut h_u64 = h as u64;
        let mut w_u64 = w as u64;
        let mut kh_u64 = kh as u64;
        let mut kw_u64 = kw as u64;
        let mut sh_u64 = stride_h as u64;
        let mut sw_u64 = stride_w as u64;
        let mut ph_u64 = pad_h as u64;
        let mut pw_u64 = pad_w as u64;
        let mut oh_u64 = out_h as u64;
        let mut ow_u64 = out_w as u64;
        let mut params: [*mut c_void; 15] = [
            (&mut d_in as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_arg as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut w_u64 as *mut u64).cast(),
            (&mut kh_u64 as *mut u64).cast(),
            (&mut kw_u64 as *mut u64).cast(),
            (&mut sh_u64 as *mut u64).cast(),
            (&mut sw_u64 as *mut u64).cast(),
            (&mut ph_u64 as *mut u64).cast(),
            (&mut pw_u64 as *mut u64).cast(),
            (&mut oh_u64 as *mut u64).cast(),
            (&mut ow_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, total_out, 256, &mut params)?;
        let mut out = vec![0.0f32; total_out];
        let mut arg_u64 = vec![0u64; total_out];
        dtoh_f32(driver, &mut out, d_out)?;
        // SAFETY: `arg_u64` has `total_out * 8` bytes.
        let arg_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                arg_u64.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(arg_u64.as_slice()),
            )
        };
        dtoh_bytes(driver, arg_bytes, d_arg)?;
        let argmax = arg_u64.into_iter().map(|v| v as usize).collect();
        Ok((out, argmax))
    })
}

/// N10: CUDA avg_pool2d.
#[allow(clippy::too_many_arguments)]
pub fn cuda_avg_pool2d_f32(
    gpu_id: usize,
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
    let total_out = batch * c * out_h * out_w;
    if input.len() < MIN_CUDA_NN_ELEMENTS {
        return None;
    }
    with_nn_ctx(gpu_id, |driver, rt, state| {
        let func = state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))?
            .avg_pool2d;
        let off_out = align256(input.len() * 4);
        let base = state
            .workspace
            .ensure_bytes(driver, off_out + align256(total_out * 4))?;
        let mut d_in = base;
        let mut d_out = base + off_out as u64;
        htod_f32(driver, d_in, input)?;

        let mut total_u64 = total_out as u64;
        let mut c_u64 = c as u64;
        let mut h_u64 = h as u64;
        let mut w_u64 = w as u64;
        let mut kh_u64 = kh as u64;
        let mut kw_u64 = kw as u64;
        let mut sh_u64 = stride_h as u64;
        let mut sw_u64 = stride_w as u64;
        let mut ph_u64 = pad_h as u64;
        let mut pw_u64 = pad_w as u64;
        let mut oh_u64 = out_h as u64;
        let mut ow_u64 = out_w as u64;
        let mut params: [*mut c_void; 14] = [
            (&mut d_in as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut w_u64 as *mut u64).cast(),
            (&mut kh_u64 as *mut u64).cast(),
            (&mut kw_u64 as *mut u64).cast(),
            (&mut sh_u64 as *mut u64).cast(),
            (&mut sw_u64 as *mut u64).cast(),
            (&mut ph_u64 as *mut u64).cast(),
            (&mut pw_u64 as *mut u64).cast(),
            (&mut oh_u64 as *mut u64).cast(),
            (&mut ow_u64 as *mut u64).cast(),
        ];
        launch_1d(rt, func, total_out, 256, &mut params)?;
        let mut out = vec![0.0f32; total_out];
        dtoh_f32(driver, &mut out, d_out)?;
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_nn_ptx_syntax_and_graceful_fallback() {
        assert!(OCHRE_NN_PTX.contains(".target sm_50"));
        assert!(OCHRE_NN_PTX.contains("ochre_gelu_f32"));
        assert!(OCHRE_NN_PTX.contains("ochre_adamw_step_f32"));
        assert!(OCHRE_NN_PTX.contains("ochre_im2col_f32"));
        assert!(!OCHRE_NN_PTX.contains("tanh.approx"));

        // Below threshold always returns None cleanly for CPU fallback
        let small = vec![1.0f32; 16];
        assert!(cuda_gelu_f32(0, &small).is_none());
        assert!(cuda_batched_matmul_f32(0, 1, 4, 4, 4, &small, &small, false, false).is_none());
    }
}
