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
    // VRAM-resident training & inference kernels
    fill_f32: CuFunction,
    copy_f32: CuFunction,
    add_inplace_f32: CuFunction,
    mul_f32: CuFunction,
    f32_to_bf16: CuFunction,
    gather_rows_f32: CuFunction,
    gather_u64: CuFunction,
    add_bias_row_f32: CuFunction,
    sum_rows_f32: CuFunction,
    act_fwd_f32: CuFunction,
    act_bwd_f32: CuFunction,
    mse_grad_f32: CuFunction,
    reduce_sum_add_f32: CuFunction,
    reduce_sum_sq_add_f32: CuFunction,
    clip_scale_f32: CuFunction,
    sgd_step_f32: CuFunction,
    adam_step_f32: CuFunction,
    nhwc_to_nchw_bias_f32: CuFunction,
    nchw_to_nhwc_f32: CuFunction,
    sum_nchw_channel_f32: CuFunction,
    bn2d_fwd_train_f32: CuFunction,
    bn2d_fwd_eval_f32: CuFunction,
    bn2d_bwd_f32: CuFunction,
    max_pool2d_bwd_f32: CuFunction,
    global_avg_pool2d_fwd_f32: CuFunction,
    global_avg_pool2d_bwd_f32: CuFunction,
    row_layernorm_bwd_dx_f32: CuFunction,
    row_layernorm_bwd_dgb_f32: CuFunction,
    row_rmsnorm_bwd_dx_f32: CuFunction,
    row_rmsnorm_bwd_dg_f32: CuFunction,
    dropout_fwd_f32: CuFunction,
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

.visible .entry ochre_fill_f32(
    .param .u64 p_dst,
    .param .f32 p_val,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<5>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra FILL_DONE;

    ld.param.u64 %rd3, [p_dst];
    ld.param.f32 %f1, [p_val];
    shl.b64 %rd4, %rd1, 2;
    add.u64 %rd3, %rd3, %rd4;
    st.global.f32 [%rd3], %f1;
FILL_DONE:
    ret;
}

.visible .entry ochre_copy_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra CPY_DONE;

    ld.param.u64 %rd3, [p_src];
    ld.param.u64 %rd4, [p_dst];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f1, [%rd3];
    st.global.f32 [%rd4], %f1;
CPY_DONE:
    ret;
}

.visible .entry ochre_add_inplace_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra ADDI_DONE;

    ld.param.u64 %rd3, [p_dst];
    ld.param.u64 %rd4, [p_src];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f1, [%rd3];
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd3], %f1;
ADDI_DONE:
    ret;
}

.visible .entry ochre_mul_f32(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra MUL_DONE;

    ld.param.u64 %rd3, [p_a];
    ld.param.u64 %rd4, [p_b];
    ld.param.u64 %rd5, [p_out];
    shl.b64 %rd6, %rd1, 2;
    add.u64 %rd3, %rd3, %rd6;
    add.u64 %rd4, %rd4, %rd6;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f1, [%rd3];
    ld.global.f32 %f2, [%rd4];
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd5], %f3;
MUL_DONE:
    ret;
}

.visible .entry ochre_f32_to_bf16(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .u16 %rs<2>;
    .reg .b64 %rd<7>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra BF16_DONE;

    ld.param.u64 %rd3, [p_src];
    ld.param.u64 %rd4, [p_dst];
    shl.b64 %rd5, %rd1, 2;
    shl.b64 %rd6, %rd1, 1;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd6;
    ld.global.u32 %r4, [%rd3];
    shr.u32 %r5, %r4, 16;
    and.b32 %r5, %r5, 1;
    add.u32 %r5, %r5, 32767;
    add.u32 %r4, %r4, %r5;
    shr.u32 %r4, %r4, 16;
    cvt.u16.u32 %rs1, %r4;
    st.global.u16 [%rd4], %rs1;
BF16_DONE:
    ret;
}

.visible .entry ochre_gather_rows_f32(
    .param .u64 p_src,
    .param .u64 p_indices,
    .param .u64 p_dst,
    .param .u64 p_batch,
    .param .u64 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<12>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_cols];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra GR_DONE;

    div.u64 %rd5, %rd1, %rd3;      // row
    rem.u64 %rd6, %rd1, %rd3;      // col
    ld.param.u64 %rd7, [p_indices];
    shl.b64 %rd8, %rd5, 3;
    add.u64 %rd7, %rd7, %rd8;
    ld.global.u64 %rd9, [%rd7];    // src_row

    mul.lo.u64 %rd10, %rd9, %rd3;
    add.u64 %rd10, %rd10, %rd6;
    shl.b64 %rd10, %rd10, 2;
    ld.param.u64 %rd11, [p_src];
    add.u64 %rd11, %rd11, %rd10;
    ld.global.f32 %f1, [%rd11];

    ld.param.u64 %rd11, [p_dst];
    shl.b64 %rd10, %rd1, 2;
    add.u64 %rd11, %rd11, %rd10;
    st.global.f32 [%rd11], %f1;
GR_DONE:
    ret;
}

.visible .entry ochre_gather_u64(
    .param .u64 p_src,
    .param .u64 p_indices,
    .param .u64 p_dst,
    .param .u64 p_batch
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<9>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra GU_DONE;

    shl.b64 %rd3, %rd1, 3;
    ld.param.u64 %rd4, [p_indices];
    add.u64 %rd4, %rd4, %rd3;
    ld.global.u64 %rd5, [%rd4];
    shl.b64 %rd5, %rd5, 3;
    ld.param.u64 %rd6, [p_src];
    add.u64 %rd6, %rd6, %rd5;
    ld.global.u64 %rd7, [%rd6];

    ld.param.u64 %rd8, [p_dst];
    add.u64 %rd8, %rd8, %rd3;
    st.global.u64 [%rd8], %rd7;
GU_DONE:
    ret;
}

.visible .entry ochre_add_bias_row_f32(
    .param .u64 p_y,
    .param .u64 p_bias,
    .param .u64 p_rows,
    .param .u64 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<9>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_rows];
    ld.param.u64 %rd3, [p_cols];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra ABR_DONE;

    rem.u64 %rd5, %rd1, %rd3;      // col
    ld.param.u64 %rd6, [p_y];
    ld.param.u64 %rd7, [p_bias];
    shl.b64 %rd8, %rd1, 2;
    add.u64 %rd6, %rd6, %rd8;
    shl.b64 %rd8, %rd5, 2;
    add.u64 %rd7, %rd7, %rd8;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd6], %f1;
ABR_DONE:
    ret;
}

.visible .entry ochre_sum_rows_f32(
    .param .u64 p_dy,
    .param .u64 p_db,
    .param .u64 p_rows,
    .param .u64 p_cols
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // col c
    ld.param.u64 %rd2, [p_cols];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SR_DONE;

    ld.param.u64 %rd3, [p_rows];
    ld.param.u64 %rd4, [p_dy];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd4, %rd4, %rd5;      // &dy[0, c]
    shl.b64 %rd6, %rd2, 2;         // row_stride_bytes

    mov.f32 %f1, 0f00000000;
    mov.u64 %rd7, 0;
SR_LOOP:
    setp.ge.u64 %p2, %rd7, %rd3;
    @%p2 bra SR_END;
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd4, %rd4, %rd6;
    add.u64 %rd7, %rd7, 1;
    bra SR_LOOP;
SR_END:
    ld.param.u64 %rd8, [p_db];
    add.u64 %rd8, %rd8, %rd5;
    st.global.f32 [%rd8], %f1;
SR_DONE:
    ret;
}

.visible .entry ochre_act_fwd_f32(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u64 p_n,
    .param .u32 p_kind
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra AF_DONE;

    ld.param.u64 %rd3, [p_x];
    ld.param.u64 %rd4, [p_out];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd3, %rd3, %rd5;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f1, [%rd3];
    ld.param.u32 %r4, [p_kind];

    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra AF_RELU;
    setp.eq.u32 %p3, %r4, 1;
    @%p3 bra AF_SIG;

    // kind == 2: Tanh via ex2
    add.f32 %f2, %f1, %f1;
    mov.f32 %f3, 0fC2200000;       // -40.0
    mov.f32 %f4, 0f42200000;       // +40.0
    max.f32 %f2, %f2, %f3;
    min.f32 %f2, %f2, %f4;
    mov.f32 %f5, 0f3FB8AA3B;       // log2(e)
    mul.f32 %f2, %f2, %f5;
    ex2.approx.f32 %f6, %f2;
    mov.f32 %f7, 0f3F800000;       // 1.0
    sub.f32 %f8, %f6, %f7;
    add.f32 %f9, %f6, %f7;
    div.rn.f32 %f1, %f8, %f9;
    bra AF_STORE;

AF_SIG:
    neg.f32 %f2, %f1;
    mov.f32 %f3, 0fC2A00000;       // -80.0
    mov.f32 %f4, 0f42A00000;       // +80.0
    max.f32 %f2, %f2, %f3;
    min.f32 %f2, %f2, %f4;
    mov.f32 %f5, 0f3FB8AA3B;
    mul.f32 %f2, %f2, %f5;
    ex2.approx.f32 %f6, %f2;
    mov.f32 %f7, 0f3F800000;
    add.f32 %f8, %f7, %f6;
    div.rn.f32 %f1, %f7, %f8;
    bra AF_STORE;

AF_RELU:
    mov.f32 %f2, 0f00000000;
    max.f32 %f1, %f1, %f2;

AF_STORE:
    st.global.f32 [%rd4], %f1;
AF_DONE:
    ret;
}

.visible .entry ochre_act_bwd_f32(
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dx,
    .param .u64 p_n,
    .param .u32 p_kind
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<7>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra AB_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_y];
    ld.param.u64 %rd6, [p_dy];
    ld.param.u64 %rd7, [p_dx];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd7, %rd3;
    ld.global.f32 %f1, [%rd6];     // dy
    ld.param.u32 %r4, [p_kind];

    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra AB_RELU;
    setp.eq.u32 %p3, %r4, 1;
    @%p3 bra AB_SIG;

    // kind == 2: Tanh bwd: dy * (1 - y*y)
    ld.global.f32 %f2, [%rd5];     // y
    mov.f32 %f3, 0f3F800000;
    mul.f32 %f4, %f2, %f2;
    sub.f32 %f4, %f3, %f4;
    mul.f32 %f1, %f1, %f4;
    bra AB_STORE;

AB_SIG:
    // Sigmoid bwd: dy * y * (1 - y)
    ld.global.f32 %f2, [%rd5];     // y
    mov.f32 %f3, 0f3F800000;
    sub.f32 %f4, %f3, %f2;
    mul.f32 %f4, %f2, %f4;
    mul.f32 %f1, %f1, %f4;
    bra AB_STORE;

AB_RELU:
    ld.global.f32 %f2, [%rd4];     // x
    mov.f32 %f3, 0f00000000;
    setp.gt.f32 %p4, %f2, %f3;
    selp.f32 %f1, %f1, %f3, %p4;

AB_STORE:
    st.global.f32 [%rd7], %f1;
AB_DONE:
    ret;
}

.visible .entry ochre_mse_grad_f32(
    .param .u64 p_pred,
    .param .u64 p_target,
    .param .u64 p_dpred,
    .param .u64 p_sq,
    .param .u64 p_n,
    .param .f32 p_scale
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<6>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra MSE_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_pred];
    ld.param.u64 %rd5, [p_target];
    ld.param.u64 %rd6, [p_dpred];
    ld.param.u64 %rd7, [p_sq];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd7, %rd3;

    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    ld.param.f32 %f3, [p_scale];
    sub.f32 %f4, %f1, %f2;         // diff
    mul.f32 %f5, %f4, %f3;         // scale * diff
    mul.f32 %f4, %f4, %f4;         // diff * diff
    st.global.f32 [%rd6], %f5;
    st.global.f32 [%rd7], %f4;
MSE_DONE:
    ret;
}

.visible .entry ochre_reduce_sum_add_f32(
    .param .u64 p_src,
    .param .u64 p_accum,
    .param .u64 p_n,
    .param .f32 p_scale
)
{
    .shared .f32 smem_rs[256];
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r1;         // idx
    ld.param.u64 %rd2, [p_n];
    ld.param.u64 %rd3, [p_src];

    mov.f32 %f1, 0f00000000;
RS_LOOP:
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra RS_REDUCE;
    shl.b64 %rd4, %rd1, 2;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd1, %rd1, 256;
    bra RS_LOOP;

RS_REDUCE:
    mov.u64 %rd6, smem_rs;
    cvt.u64.u32 %rd7, %r1;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f1;
    bar.sync 0;

    mov.u32 %r2, 128;
RS_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra RS_WRITE;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra RS_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.u64 %rd9, %rd6, %rd9;
    ld.shared.f32 %f2, [%rd9];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd8], %f1;
RS_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra RS_TREE;

RS_WRITE:
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra RS_DONE;
    ld.param.u64 %rd9, [p_accum];
    ld.param.f32 %f3, [p_scale];
    ld.global.f32 %f4, [%rd9];
    fma.rn.f32 %f4, %f1, %f3, %f4;
    st.global.f32 [%rd9], %f4;
RS_DONE:
    ret;
}

.visible .entry ochre_reduce_sum_sq_add_f32(
    .param .u64 p_src,
    .param .u64 p_accum,
    .param .u64 p_n
)
{
    .shared .f32 smem_sq[256];
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    ld.param.u64 %rd3, [p_src];

    mov.f32 %f1, 0f00000000;
RSQ_LOOP:
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra RSQ_REDUCE;
    shl.b64 %rd4, %rd1, 2;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f2, [%rd5];
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u64 %rd1, %rd1, 256;
    bra RSQ_LOOP;

RSQ_REDUCE:
    mov.u64 %rd6, smem_sq;
    cvt.u64.u32 %rd7, %r1;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f1;
    bar.sync 0;

    mov.u32 %r2, 128;
RSQ_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra RSQ_WRITE;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra RSQ_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.u64 %rd9, %rd6, %rd9;
    ld.shared.f32 %f2, [%rd9];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd8], %f1;
RSQ_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra RSQ_TREE;

RSQ_WRITE:
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra RSQ_DONE;
    ld.param.u64 %rd9, [p_accum];
    ld.global.f32 %f3, [%rd9];
    add.f32 %f3, %f3, %f1;
    st.global.f32 [%rd9], %f3;
RSQ_DONE:
    ret;
}

.visible .entry ochre_clip_scale_f32(
    .param .u64 p_grad,
    .param .u64 p_norm_sq,
    .param .f32 p_max_norm,
    .param .u64 p_n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<7>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra CS_DONE;

    ld.param.u64 %rd3, [p_norm_sq];
    ld.global.f32 %f1, [%rd3];
    sqrt.rn.f32 %f2, %f1;          // total_norm
    ld.param.f32 %f3, [p_max_norm];
    setp.le.f32 %p2, %f2, %f3;
    @%p2 bra CS_DONE;

    mov.f32 %f4, 0f358637BD;       // 1e-6
    add.f32 %f5, %f2, %f4;
    div.rn.f32 %f5, %f3, %f5;      // scale = max_norm / (total_norm + 1e-6)

    ld.param.u64 %rd4, [p_grad];
    shl.b64 %rd5, %rd1, 2;
    add.u64 %rd4, %rd4, %rd5;
    ld.global.f32 %f6, [%rd4];
    mul.f32 %f6, %f6, %f5;
    st.global.f32 [%rd4], %f6;
CS_DONE:
    ret;
}

.visible .entry ochre_sgd_step_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_vel,
    .param .u64 p_n,
    .param .f32 p_lr,
    .param .f32 p_mom,
    .param .f32 p_wd,
    .param .u32 p_has_mom
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<9>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SGD_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_param];
    ld.param.u64 %rd5, [p_grad];
    ld.param.u64 %rd6, [p_vel];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;

    ld.global.f32 %f1, [%rd4];     // param
    ld.global.f32 %f2, [%rd5];     // grad
    ld.param.f32 %f3, [p_lr];
    ld.param.f32 %f4, [p_mom];
    ld.param.f32 %f5, [p_wd];
    ld.param.u32 %r4, [p_has_mom];

    fma.rn.f32 %f6, %f5, %f1, %f2; // g = grad + wd * param
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra SGD_NO_MOM;
    ld.global.f32 %f7, [%rd6];     // vel
    fma.rn.f32 %f6, %f7, %f4, %f6; // v = vel * mom + g
    st.global.f32 [%rd6], %f6;
SGD_NO_MOM:
    mul.f32 %f8, %f3, %f6;
    sub.f32 %f1, %f1, %f8;
    st.global.f32 [%rd4], %f1;
SGD_DONE:
    ret;
}

.visible .entry ochre_adam_step_f32(
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
    @%p1 bra AD_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_param];
    ld.param.u64 %rd5, [p_grad];
    ld.param.u64 %rd6, [p_m];
    ld.param.u64 %rd7, [p_v];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd7, %rd3;

    ld.global.f32 %f1, [%rd4];     // param
    ld.global.f32 %f2, [%rd5];     // grad
    ld.global.f32 %f3, [%rd6];     // m
    ld.global.f32 %f4, [%rd7];     // v

    ld.param.f32 %f5, [p_lr];
    ld.param.f32 %f6, [p_beta1];
    ld.param.f32 %f7, [p_beta2];
    ld.param.f32 %f8, [p_eps];
    ld.param.f32 %f9, [p_wd];
    ld.param.f32 %f10, [p_inv_bc1];
    ld.param.f32 %f11, [p_inv_bc2];

    // Coupled L2 weight decay: g = grad + wd * param
    fma.rn.f32 %f2, %f9, %f1, %f2;

    mov.f32 %f12, 0f3F800000;
    sub.f32 %f13, %f12, %f6;
    mul.f32 %f3, %f3, %f6;
    fma.rn.f32 %f3, %f13, %f2, %f3;

    sub.f32 %f14, %f12, %f7;
    mul.f32 %f15, %f2, %f2;
    mul.f32 %f4, %f4, %f7;
    fma.rn.f32 %f4, %f14, %f15, %f4;

    st.global.f32 [%rd6], %f3;
    st.global.f32 [%rd7], %f4;

    mul.f32 %f13, %f3, %f10;
    mul.f32 %f14, %f4, %f11;
    sqrt.rn.f32 %f14, %f14;
    add.f32 %f14, %f14, %f8;
    div.rn.f32 %f17, %f13, %f14;
    mul.f32 %f17, %f17, %f5;
    sub.f32 %f1, %f1, %f17;
    st.global.f32 [%rd4], %f1;
AD_DONE:
    ret;
}

.visible .entry ochre_nhwc_to_nchw_bias_f32(
    .param .u64 p_ymat,
    .param .u64 p_bias,
    .param .u64 p_out,
    .param .u64 p_batch,
    .param .u64 p_c,
    .param .u64 p_spatial,
    .param .u32 p_has_bias
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // nchw idx
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_spatial];
    mul.lo.u64 %rd5, %rd2, %rd3;
    mul.lo.u64 %rd5, %rd5, %rd4;
    setp.ge.u64 %p1, %rd1, %rd5;
    @%p1 bra N2C_DONE;

    rem.u64 %rd6, %rd1, %rd4;      // s
    div.u64 %rd7, %rd1, %rd4;
    rem.u64 %rd8, %rd7, %rd3;      // oc
    div.u64 %rd9, %rd7, %rd3;      // b

    // y_idx = (b * spatial + s) * c + oc
    mul.lo.u64 %rd10, %rd9, %rd4;
    add.u64 %rd10, %rd10, %rd6;
    mul.lo.u64 %rd10, %rd10, %rd3;
    add.u64 %rd10, %rd10, %rd8;
    shl.b64 %rd10, %rd10, 2;
    ld.param.u64 %rd11, [p_ymat];
    add.u64 %rd11, %rd11, %rd10;
    ld.global.f32 %f1, [%rd11];

    ld.param.u32 %r4, [p_has_bias];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra N2C_STORE;
    ld.param.u64 %rd12, [p_bias];
    shl.b64 %rd10, %rd8, 2;
    add.u64 %rd12, %rd12, %rd10;
    ld.global.f32 %f2, [%rd12];
    add.f32 %f1, %f1, %f2;

N2C_STORE:
    ld.param.u64 %rd13, [p_out];
    shl.b64 %rd10, %rd1, 2;
    add.u64 %rd13, %rd13, %rd10;
    st.global.f32 [%rd13], %f1;
N2C_DONE:
    ret;
}

.visible .entry ochre_nchw_to_nhwc_f32(
    .param .u64 p_nchw,
    .param .u64 p_nhwc,
    .param .u64 p_batch,
    .param .u64 p_c,
    .param .u64 p_spatial
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<13>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_spatial];
    mul.lo.u64 %rd5, %rd2, %rd3;
    mul.lo.u64 %rd5, %rd5, %rd4;
    setp.ge.u64 %p1, %rd1, %rd5;
    @%p1 bra C2N_DONE;

    rem.u64 %rd6, %rd1, %rd4;      // s
    div.u64 %rd7, %rd1, %rd4;
    rem.u64 %rd8, %rd7, %rd3;      // oc
    div.u64 %rd9, %rd7, %rd3;      // b

    ld.param.u64 %rd10, [p_nchw];
    shl.b64 %rd11, %rd1, 2;
    add.u64 %rd10, %rd10, %rd11;
    ld.global.f32 %f1, [%rd10];

    mul.lo.u64 %rd11, %rd9, %rd4;
    add.u64 %rd11, %rd11, %rd6;
    mul.lo.u64 %rd11, %rd11, %rd3;
    add.u64 %rd11, %rd11, %rd8;
    shl.b64 %rd11, %rd11, 2;
    ld.param.u64 %rd12, [p_nhwc];
    add.u64 %rd12, %rd12, %rd11;
    st.global.f32 [%rd12], %f1;
C2N_DONE:
    ret;
}

.visible .entry ochre_sum_nchw_channel_f32(
    .param .u64 p_dy,
    .param .u64 p_db,
    .param .u64 p_batch,
    .param .u64 p_c,
    .param .u64 p_spatial
)
{
    .shared .f32 smem_sc[256];
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<3>;

    mov.u32 %r4, %ctaid.x;         // ch
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r4;         // ch u64
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_spatial];
    mul.lo.u64 %rd5, %rd2, %rd4;   // m = batch * spatial
    ld.param.u64 %rd6, [p_dy];

    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd7, %r1;         // idx
SC_LOOP:
    setp.ge.u64 %p1, %rd7, %rd5;
    @%p1 bra SC_REDUCE;
    div.u64 %rd8, %rd7, %rd4;      // b
    rem.u64 %rd9, %rd7, %rd4;      // s
    mul.lo.u64 %rd10, %rd8, %rd3;
    add.u64 %rd10, %rd10, %rd1;
    mul.lo.u64 %rd10, %rd10, %rd4;
    add.u64 %rd10, %rd10, %rd9;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd6, %rd10;
    ld.global.f32 %f2, [%rd10];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd7, %rd7, 256;
    bra SC_LOOP;

SC_REDUCE:
    mov.u64 %rd11, smem_sc;
    cvt.u64.u32 %rd12, %r1;
    shl.b64 %rd12, %rd12, 2;
    add.u64 %rd12, %rd11, %rd12;
    st.shared.f32 [%rd12], %f1;
    bar.sync 0;

    mov.u32 %r2, 128;
SC_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra SC_WRITE;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra SC_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd13, %r3;
    shl.b64 %rd13, %rd13, 2;
    add.u64 %rd13, %rd11, %rd13;
    ld.shared.f32 %f2, [%rd13];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd12], %f1;
SC_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra SC_TREE;

SC_WRITE:
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra SC_DONE;
    ld.param.u64 %rd13, [p_db];
    shl.b64 %rd10, %rd1, 2;
    add.u64 %rd13, %rd13, %rd10;
    st.global.f32 [%rd13], %f1;
SC_DONE:
    ret;
}

.visible .entry ochre_bn2d_fwd_train_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_beta,
    .param .u64 p_rm,
    .param .u64 p_rv,
    .param .u64 p_out,
    .param .u64 p_smean,
    .param .u64 p_sinv,
    .param .u64 p_batch,
    .param .u64 p_c,
    .param .u64 p_plane,
    .param .f32 p_eps,
    .param .f32 p_mom,
    .param .u32 p_has_affine
)
{
    .shared .f32 smem_bn[256];
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;

    mov.u32 %r4, %ctaid.x;         // ch
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r4;         // ch u64
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_plane];
    mul.lo.u64 %rd5, %rd2, %rd4;   // m = batch * plane
    ld.param.u64 %rd6, [p_x];

    // Pass 1: sum
    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd7, %r1;
BNF_SUM:
    setp.ge.u64 %p1, %rd7, %rd5;
    @%p1 bra BNF_SUM_RED;
    div.u64 %rd8, %rd7, %rd4;
    rem.u64 %rd9, %rd7, %rd4;
    mul.lo.u64 %rd10, %rd8, %rd3;
    add.u64 %rd10, %rd10, %rd1;
    mul.lo.u64 %rd10, %rd10, %rd4;
    add.u64 %rd10, %rd10, %rd9;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd6, %rd10;
    ld.global.f32 %f2, [%rd10];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd7, %rd7, 256;
    bra BNF_SUM;

BNF_SUM_RED:
    mov.u64 %rd11, smem_bn;
    cvt.u64.u32 %rd12, %r1;
    shl.b64 %rd12, %rd12, 2;
    add.u64 %rd12, %rd11, %rd12;
    st.shared.f32 [%rd12], %f1;
    bar.sync 0;

    mov.u32 %r2, 128;
BNF_S_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra BNF_MU;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra BNF_S_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd13, %r3;
    shl.b64 %rd13, %rd13, 2;
    add.u64 %rd13, %rd11, %rd13;
    ld.shared.f32 %f2, [%rd13];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd12], %f1;
BNF_S_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra BNF_S_TREE;

BNF_MU:
    ld.shared.f32 %f3, [%rd11];    // total_sum
    cvt.rn.f32.u64 %f4, %rd5;      // m_f32
    div.rn.f32 %f5, %f3, %f4;      // mu
    bar.sync 0;

    // Pass 2: variance sum
    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd7, %r1;
BNF_VAR:
    setp.ge.u64 %p1, %rd7, %rd5;
    @%p1 bra BNF_VAR_RED;
    div.u64 %rd8, %rd7, %rd4;
    rem.u64 %rd9, %rd7, %rd4;
    mul.lo.u64 %rd10, %rd8, %rd3;
    add.u64 %rd10, %rd10, %rd1;
    mul.lo.u64 %rd10, %rd10, %rd4;
    add.u64 %rd10, %rd10, %rd9;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd6, %rd10;
    ld.global.f32 %f2, [%rd10];
    sub.f32 %f6, %f2, %f5;
    fma.rn.f32 %f1, %f6, %f6, %f1;
    add.u64 %rd7, %rd7, 256;
    bra BNF_VAR;

BNF_VAR_RED:
    st.shared.f32 [%rd12], %f1;
    bar.sync 0;

    mov.u32 %r2, 128;
BNF_V_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra BNF_STATS;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra BNF_V_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd13, %r3;
    shl.b64 %rd13, %rd13, 2;
    add.u64 %rd13, %rd11, %rd13;
    ld.shared.f32 %f2, [%rd13];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd12], %f1;
BNF_V_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra BNF_V_TREE;

BNF_STATS:
    ld.shared.f32 %f6, [%rd11];    // sq
    div.rn.f32 %f7, %f6, %f4;      // var = sq / m
    ld.param.f32 %f8, [p_eps];
    add.f32 %f9, %f7, %f8;
    sqrt.rn.f32 %f9, %f9;
    mov.f32 %f10, 0f3F800000;
    div.rn.f32 %f11, %f10, %f9;    // inv_std

    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra BNF_NORM;
    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd15, [p_smean];
    add.u64 %rd15, %rd15, %rd14;
    st.global.f32 [%rd15], %f5;
    ld.param.u64 %rd15, [p_sinv];
    add.u64 %rd15, %rd15, %rd14;
    st.global.f32 [%rd15], %f11;

    // unbiased_var
    mov.f32 %f12, %f7;
    setp.le.u64 %p3, %rd5, 1;
    @%p3 bra BNF_UP_RUN;
    sub.u64 %rd13, %rd5, 1;
    cvt.rn.f32.u64 %f13, %rd13;
    div.rn.f32 %f12, %f6, %f13;
BNF_UP_RUN:
    ld.param.f32 %f13, [p_mom];
    sub.f32 %f14, %f10, %f13;      // 1 - mom
    ld.param.u64 %rd15, [p_rm];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f15, [%rd15];
    mul.f32 %f15, %f15, %f14;
    fma.rn.f32 %f15, %f13, %f5, %f15;
    st.global.f32 [%rd15], %f15;

    ld.param.u64 %rd15, [p_rv];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f15, [%rd15];
    mul.f32 %f15, %f15, %f14;
    fma.rn.f32 %f15, %f13, %f12, %f15;
    st.global.f32 [%rd15], %f15;

BNF_NORM:
    mov.f32 %f12, 0f3F800000;      // gamma = 1
    mov.f32 %f13, 0f00000000;      // beta = 0
    ld.param.u32 %r5, [p_has_affine];
    setp.eq.u32 %p4, %r5, 0;
    @%p4 bra BNF_OUT_LOOP;
    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd15, [p_gamma];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f12, [%rd15];
    ld.param.u64 %rd15, [p_beta];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f13, [%rd15];

BNF_OUT_LOOP:
    ld.param.u64 %rd15, [p_out];
    cvt.u64.u32 %rd7, %r1;
BNF_OUT:
    setp.ge.u64 %p1, %rd7, %rd5;
    @%p1 bra BNF_DONE;
    div.u64 %rd8, %rd7, %rd4;
    rem.u64 %rd9, %rd7, %rd4;
    mul.lo.u64 %rd10, %rd8, %rd3;
    add.u64 %rd10, %rd10, %rd1;
    mul.lo.u64 %rd10, %rd10, %rd4;
    add.u64 %rd10, %rd10, %rd9;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd13, %rd6, %rd10;
    ld.global.f32 %f2, [%rd13];
    sub.f32 %f6, %f2, %f5;
    mul.f32 %f6, %f6, %f11;        // x_hat
    fma.rn.f32 %f6, %f6, %f12, %f13;
    add.u64 %rd13, %rd15, %rd10;
    st.global.f32 [%rd13], %f6;
    add.u64 %rd7, %rd7, 256;
    bra BNF_OUT;
BNF_DONE:
    ret;
}

.visible .entry ochre_bn2d_fwd_eval_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_beta,
    .param .u64 p_rm,
    .param .u64 p_rv,
    .param .u64 p_out,
    .param .u64 p_total,
    .param .u64 p_c,
    .param .u64 p_plane,
    .param .f32 p_eps,
    .param .u32 p_has_affine
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<11>;
    .reg .f32 %f<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra BNE_DONE;

    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_plane];
    div.u64 %rd5, %rd1, %rd4;
    rem.u64 %rd5, %rd5, %rd3;      // ch
    shl.b64 %rd6, %rd5, 2;

    ld.param.u64 %rd7, [p_rm];
    add.u64 %rd7, %rd7, %rd6;
    ld.global.f32 %f1, [%rd7];     // mu
    ld.param.u64 %rd7, [p_rv];
    add.u64 %rd7, %rd7, %rd6;
    ld.global.f32 %f2, [%rd7];     // var
    ld.param.f32 %f3, [p_eps];
    add.f32 %f2, %f2, %f3;
    sqrt.rn.f32 %f2, %f2;
    mov.f32 %f4, 0f3F800000;
    div.rn.f32 %f5, %f4, %f2;      // inv_std

    mov.f32 %f6, 0f3F800000;       // g
    mov.f32 %f7, 0f00000000;       // b
    ld.param.u32 %r4, [p_has_affine];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra BNE_COMP;
    ld.param.u64 %rd8, [p_gamma];
    add.u64 %rd8, %rd8, %rd6;
    ld.global.f32 %f6, [%rd8];
    ld.param.u64 %rd8, [p_beta];
    add.u64 %rd8, %rd8, %rd6;
    ld.global.f32 %f7, [%rd8];

BNE_COMP:
    shl.b64 %rd9, %rd1, 2;
    ld.param.u64 %rd10, [p_x];
    add.u64 %rd10, %rd10, %rd9;
    ld.global.f32 %f8, [%rd10];
    sub.f32 %f9, %f8, %f1;
    mul.f32 %f9, %f9, %f5;
    fma.rn.f32 %f9, %f9, %f6, %f7;
    ld.param.u64 %rd10, [p_out];
    add.u64 %rd10, %rd10, %rd9;
    st.global.f32 [%rd10], %f9;
BNE_DONE:
    ret;
}

.visible .entry ochre_bn2d_bwd_f32(
    .param .u64 p_x,
    .param .u64 p_dy,
    .param .u64 p_gamma,
    .param .u64 p_smean,
    .param .u64 p_sinv,
    .param .u64 p_dx,
    .param .u64 p_dg,
    .param .u64 p_db,
    .param .u64 p_batch,
    .param .u64 p_c,
    .param .u64 p_plane,
    .param .u32 p_has_affine,
    .param .u32 p_need_dx
)
{
    .shared .f32 smem_bdy[256];
    .shared .f32 smem_bdyx[256];
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<16>;

    mov.u32 %r4, %ctaid.x;         // ch
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_c];
    ld.param.u64 %rd4, [p_plane];
    mul.lo.u64 %rd5, %rd2, %rd4;   // m
    ld.param.u64 %rd6, [p_x];
    ld.param.u64 %rd7, [p_dy];

    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd15, [p_smean];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f3, [%rd15];    // mu
    ld.param.u64 %rd15, [p_sinv];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f4, [%rd15];    // inv_std

    mov.f32 %f1, 0f00000000;       // sum_dy
    mov.f32 %f2, 0f00000000;       // sum_dy_xhat
    cvt.u64.u32 %rd8, %r1;
BNB_SUM:
    setp.ge.u64 %p1, %rd8, %rd5;
    @%p1 bra BNB_RED;
    div.u64 %rd9, %rd8, %rd4;
    rem.u64 %rd10, %rd8, %rd4;
    mul.lo.u64 %rd11, %rd9, %rd3;
    add.u64 %rd11, %rd11, %rd1;
    mul.lo.u64 %rd11, %rd11, %rd4;
    add.u64 %rd11, %rd11, %rd10;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd12, %rd6, %rd11;
    add.u64 %rd13, %rd7, %rd11;
    ld.global.f32 %f5, [%rd12];    // x
    ld.global.f32 %f6, [%rd13];    // dy
    sub.f32 %f7, %f5, %f3;
    mul.f32 %f7, %f7, %f4;         // x_hat
    add.f32 %f1, %f1, %f6;
    fma.rn.f32 %f2, %f6, %f7, %f2;
    add.u64 %rd8, %rd8, 256;
    bra BNB_SUM;

BNB_RED:
    mov.u64 %rd12, smem_bdy;
    mov.u64 %rd13, smem_bdyx;
    cvt.u64.u32 %rd16, %r1;
    shl.b64 %rd16, %rd16, 2;
    add.u64 %rd17, %rd12, %rd16;
    st.shared.f32 [%rd17], %f1;
    add.u64 %rd17, %rd13, %rd16;
    st.shared.f32 [%rd17], %f2;
    bar.sync 0;

    mov.u32 %r2, 128;
BNB_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra BNB_WRITE;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra BNB_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd17, %r3;
    shl.b64 %rd17, %rd17, 2;
    add.u64 %rd11, %rd12, %rd17;
    ld.shared.f32 %f5, [%rd11];
    add.f32 %f1, %f1, %f5;
    add.u64 %rd11, %rd12, %rd16;
    st.shared.f32 [%rd11], %f1;

    add.u64 %rd11, %rd13, %rd17;
    ld.shared.f32 %f6, [%rd11];
    add.f32 %f2, %f2, %f6;
    add.u64 %rd11, %rd13, %rd16;
    st.shared.f32 [%rd11], %f2;
BNB_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra BNB_TREE;

BNB_WRITE:
    ld.shared.f32 %f8, [%rd12];    // total_sum_dy
    ld.shared.f32 %f9, [%rd13];    // total_sum_dy_xhat

    ld.param.u32 %r5, [p_has_affine];
    setp.ne.u32 %p1, %r1, 0;
    setp.eq.u32 %p3, %r5, 0;
    or.pred %p1, %p1, %p3;
    @%p1 bra BNB_DX_CHECK;
    ld.param.u64 %rd15, [p_dg];
    add.u64 %rd15, %rd15, %rd14;
    st.global.f32 [%rd15], %f9;
    ld.param.u64 %rd15, [p_db];
    add.u64 %rd15, %rd15, %rd14;
    st.global.f32 [%rd15], %f8;

BNB_DX_CHECK:
    ld.param.u32 %r3, [p_need_dx];
    setp.eq.u32 %p4, %r3, 0;
    @%p4 bra BNB_DONE;

    mov.f32 %f10, 0f3F800000;      // gamma = 1
    setp.eq.u32 %p3, %r5, 0;
    @%p3 bra BNB_DX_INIT;
    ld.param.u64 %rd15, [p_gamma];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f10, [%rd15];

BNB_DX_INIT:
    cvt.rn.f32.u64 %f11, %rd5;     // m_f32
    mul.f32 %f12, %f10, %f4;
    div.rn.f32 %f12, %f12, %f11;   // scale = g * inv_std / m
    ld.param.u64 %rd15, [p_dx];
    cvt.u64.u32 %rd8, %r1;
BNB_DX_LOOP:
    setp.ge.u64 %p1, %rd8, %rd5;
    @%p1 bra BNB_DONE;
    div.u64 %rd9, %rd8, %rd4;
    rem.u64 %rd10, %rd8, %rd4;
    mul.lo.u64 %rd11, %rd9, %rd3;
    add.u64 %rd11, %rd11, %rd1;
    mul.lo.u64 %rd11, %rd11, %rd4;
    add.u64 %rd11, %rd11, %rd10;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd12, %rd6, %rd11;
    add.u64 %rd13, %rd7, %rd11;
    ld.global.f32 %f5, [%rd12];    // x
    ld.global.f32 %f6, [%rd13];    // dy
    sub.f32 %f7, %f5, %f3;
    mul.f32 %f7, %f7, %f4;         // x_hat
    mul.f32 %f13, %f11, %f6;       // m * dy
    sub.f32 %f13, %f13, %f8;       // m * dy - sum_dy
    mul.f32 %f14, %f7, %f9;        // x_hat * sum_dy_xhat
    sub.f32 %f13, %f13, %f14;
    mul.f32 %f15, %f12, %f13;
    add.u64 %rd17, %rd15, %rd11;
    st.global.f32 [%rd17], %f15;
    add.u64 %rd8, %rd8, 256;
    bra BNB_DX_LOOP;
BNB_DONE:
    ret;
}

.visible .entry ochre_max_pool2d_bwd_f32(
    .param .u64 p_dy,
    .param .u64 p_argmax,
    .param .u64 p_dx,
    .param .u64 p_total_out
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<9>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total_out];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra MPB_DONE;

    ld.param.u64 %rd3, [p_argmax];
    shl.b64 %rd4, %rd1, 3;
    add.u64 %rd3, %rd3, %rd4;
    ld.global.u64 %rd5, [%rd3];
    mov.u64 %rd6, 18446744073709551615;
    setp.eq.u64 %p2, %rd5, %rd6;
    @%p2 bra MPB_DONE;

    ld.param.u64 %rd7, [p_dy];
    shl.b64 %rd4, %rd1, 2;
    add.u64 %rd7, %rd7, %rd4;
    ld.global.f32 %f1, [%rd7];

    ld.param.u64 %rd8, [p_dx];
    shl.b64 %rd5, %rd5, 2;
    add.u64 %rd8, %rd8, %rd5;
    atom.global.add.f32 %f2, [%rd8], %f1;
MPB_DONE:
    ret;
}

.visible .entry ochre_global_avg_pool2d_fwd_f32(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u64 p_bc_total,
    .param .u64 p_plane
)
{
    .shared .f32 smem_gap[128];
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<11>;
    .reg .f32 %f<4>;

    mov.u32 %r4, %ctaid.x;         // bc
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_bc_total];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra GAPF_DONE;

    ld.param.u64 %rd3, [p_plane];
    ld.param.u64 %rd4, [p_x];
    mul.lo.u64 %rd5, %rd1, %rd3;
    shl.b64 %rd5, %rd5, 2;
    add.u64 %rd4, %rd4, %rd5;      // &x[bc * plane]

    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd6, %r1;
GAPF_LOOP:
    setp.ge.u64 %p1, %rd6, %rd3;
    @%p1 bra GAPF_RED;
    shl.b64 %rd7, %rd6, 2;
    add.u64 %rd7, %rd4, %rd7;
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    add.u64 %rd6, %rd6, 128;
    bra GAPF_LOOP;

GAPF_RED:
    mov.u64 %rd8, smem_gap;
    cvt.u64.u32 %rd9, %r1;
    shl.b64 %rd9, %rd9, 2;
    add.u64 %rd9, %rd8, %rd9;
    st.shared.f32 [%rd9], %f1;
    bar.sync 0;

    mov.u32 %r2, 64;
GAPF_TREE:
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra GAPF_WRITE;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra GAPF_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd10, %r3;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd8, %rd10;
    ld.shared.f32 %f2, [%rd10];
    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd9], %f1;
GAPF_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra GAPF_TREE;

GAPF_WRITE:
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra GAPF_DONE;
    cvt.rn.f32.u64 %f3, %rd3;
    div.rn.f32 %f1, %f1, %f3;
    ld.param.u64 %rd10, [p_out];
    shl.b64 %rd7, %rd1, 2;
    add.u64 %rd10, %rd10, %rd7;
    st.global.f32 [%rd10], %f1;
GAPF_DONE:
    ret;
}

.visible .entry ochre_global_avg_pool2d_bwd_f32(
    .param .u64 p_dy,
    .param .u64 p_dx,
    .param .u64 p_total_in,
    .param .u64 p_plane
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_total_in];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra GAPB_DONE;

    ld.param.u64 %rd3, [p_plane];
    div.u64 %rd4, %rd1, %rd3;      // bc
    ld.param.u64 %rd5, [p_dy];
    shl.b64 %rd6, %rd4, 2;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f1, [%rd5];
    cvt.rn.f32.u64 %f2, %rd3;
    div.rn.f32 %f1, %f1, %f2;

    ld.param.u64 %rd7, [p_dx];
    shl.b64 %rd6, %rd1, 2;
    add.u64 %rd7, %rd7, %rd6;
    st.global.f32 [%rd7], %f1;
GAPB_DONE:
    ret;
}

.visible .entry ochre_row_layernorm_bwd_dx_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_mean,
    .param .u64 p_rstd,
    .param .u64 p_dx,
    .param .u64 p_rows,
    .param .u64 p_cols,
    .param .u32 p_has_gamma
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<15>;
    .reg .f32 %f<15>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // row
    ld.param.u64 %rd2, [p_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra LNBX_DONE;

    ld.param.u64 %rd3, [p_cols];
    mul.lo.u64 %rd4, %rd1, %rd3;
    shl.b64 %rd4, %rd4, 2;
    ld.param.u64 %rd5, [p_dy];
    ld.param.u64 %rd6, [p_x];
    ld.param.u64 %rd7, [p_dx];
    add.u64 %rd5, %rd5, %rd4;
    add.u64 %rd6, %rd6, %rd4;
    add.u64 %rd7, %rd7, %rd4;

    shl.b64 %rd8, %rd1, 2;
    ld.param.u64 %rd9, [p_mean];
    add.u64 %rd9, %rd9, %rd8;
    ld.global.f32 %f1, [%rd9];     // mean
    ld.param.u64 %rd9, [p_rstd];
    add.u64 %rd9, %rd9, %rd8;
    ld.global.f32 %f2, [%rd9];     // rstd

    ld.param.u64 %rd10, [p_gamma];
    ld.param.u32 %r4, [p_has_gamma];

    mov.f32 %f3, 0f00000000;       // sum_dyg
    mov.f32 %f4, 0f00000000;       // sum_dyg_xhat
    mov.u64 %rd11, 0;
LNBX_S:
    setp.ge.u64 %p2, %rd11, %rd3;
    @%p2 bra LNBX_S_END;
    shl.b64 %rd12, %rd11, 2;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.f32 %f5, [%rd13];    // dy
    add.u64 %rd13, %rd6, %rd12;
    ld.global.f32 %f6, [%rd13];    // x
    mov.f32 %f7, 0f3F800000;
    setp.eq.u32 %p3, %r4, 0;
    @%p3 bra LNBX_NG1;
    add.u64 %rd14, %rd10, %rd12;
    ld.global.f32 %f7, [%rd14];
LNBX_NG1:
    mul.f32 %f8, %f5, %f7;         // dyg
    sub.f32 %f9, %f6, %f1;
    mul.f32 %f9, %f9, %f2;         // xhat
    add.f32 %f3, %f3, %f8;
    fma.rn.f32 %f4, %f8, %f9, %f4;
    add.u64 %rd11, %rd11, 1;
    bra LNBX_S;
LNBX_S_END:

    cvt.rn.f32.u64 %f10, %rd3;     // cols_f
    div.rn.f32 %f11, %f2, %f10;    // rstd / cols
    mov.u64 %rd11, 0;
LNBX_W:
    setp.ge.u64 %p2, %rd11, %rd3;
    @%p2 bra LNBX_DONE;
    shl.b64 %rd12, %rd11, 2;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.f32 %f5, [%rd13];
    add.u64 %rd13, %rd6, %rd12;
    ld.global.f32 %f6, [%rd13];
    mov.f32 %f7, 0f3F800000;
    setp.eq.u32 %p3, %r4, 0;
    @%p3 bra LNBX_NG2;
    add.u64 %rd14, %rd10, %rd12;
    ld.global.f32 %f7, [%rd14];
LNBX_NG2:
    mul.f32 %f8, %f5, %f7;
    sub.f32 %f9, %f6, %f1;
    mul.f32 %f9, %f9, %f2;
    mul.f32 %f12, %f10, %f8;
    sub.f32 %f12, %f12, %f3;
    mul.f32 %f13, %f9, %f4;
    sub.f32 %f12, %f12, %f13;
    mul.f32 %f14, %f11, %f12;
    add.u64 %rd13, %rd7, %rd12;
    st.global.f32 [%rd13], %f14;
    add.u64 %rd11, %rd11, 1;
    bra LNBX_W;
LNBX_DONE:
    ret;
}

.visible .entry ochre_row_layernorm_bwd_dgb_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_mean,
    .param .u64 p_rstd,
    .param .u64 p_dg,
    .param .u64 p_db,
    .param .u64 p_rows,
    .param .u64 p_cols,
    .param .u32 p_has_gamma,
    .param .u32 p_has_beta
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<8>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // col
    ld.param.u64 %rd2, [p_cols];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra LNBG_DONE;

    ld.param.u64 %rd3, [p_rows];
    ld.param.u64 %rd4, [p_dy];
    ld.param.u64 %rd5, [p_x];
    ld.param.u64 %rd6, [p_mean];
    ld.param.u64 %rd7, [p_rstd];

    shl.b64 %rd8, %rd1, 2;
    add.u64 %rd4, %rd4, %rd8;
    add.u64 %rd5, %rd5, %rd8;
    shl.b64 %rd9, %rd2, 2;         // row_stride

    mov.f32 %f1, 0f00000000;       // sum_dg
    mov.f32 %f2, 0f00000000;       // sum_db
    mov.u64 %rd10, 0;
LNBG_LOOP:
    setp.ge.u64 %p2, %rd10, %rd3;
    @%p2 bra LNBG_END;
    ld.global.f32 %f3, [%rd4];     // dy
    ld.global.f32 %f4, [%rd5];     // x
    shl.b64 %rd11, %rd10, 2;
    add.u64 %rd12, %rd6, %rd11;
    ld.global.f32 %f5, [%rd12];    // mean
    add.u64 %rd12, %rd7, %rd11;
    ld.global.f32 %f6, [%rd12];    // rstd
    sub.f32 %f7, %f4, %f5;
    mul.f32 %f7, %f7, %f6;         // xhat
    fma.rn.f32 %f1, %f3, %f7, %f1;
    add.f32 %f2, %f2, %f3;
    add.u64 %rd4, %rd4, %rd9;
    add.u64 %rd5, %rd5, %rd9;
    add.u64 %rd10, %rd10, 1;
    bra LNBG_LOOP;

LNBG_END:
    ld.param.u32 %r4, [p_has_gamma];
    setp.eq.u32 %p3, %r4, 0;
    @%p3 bra LNBG_B;
    ld.param.u64 %rd13, [p_dg];
    add.u64 %rd13, %rd13, %rd8;
    st.global.f32 [%rd13], %f1;
LNBG_B:
    ld.param.u32 %r5, [p_has_beta];
    setp.eq.u32 %p4, %r5, 0;
    @%p4 bra LNBG_DONE;
    ld.param.u64 %rd13, [p_db];
    add.u64 %rd13, %rd13, %rd8;
    st.global.f32 [%rd13], %f2;
LNBG_DONE:
    ret;
}

.visible .entry ochre_row_rmsnorm_bwd_dx_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_rstd,
    .param .u64 p_dx,
    .param .u64 p_rows,
    .param .u64 p_cols
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<11>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra RNBX_DONE;

    ld.param.u64 %rd3, [p_cols];
    mul.lo.u64 %rd4, %rd1, %rd3;
    shl.b64 %rd4, %rd4, 2;
    ld.param.u64 %rd5, [p_dy];
    ld.param.u64 %rd6, [p_x];
    ld.param.u64 %rd7, [p_dx];
    ld.param.u64 %rd8, [p_gamma];
    add.u64 %rd5, %rd5, %rd4;
    add.u64 %rd6, %rd6, %rd4;
    add.u64 %rd7, %rd7, %rd4;

    shl.b64 %rd9, %rd1, 2;
    ld.param.u64 %rd10, [p_rstd];
    add.u64 %rd10, %rd10, %rd9;
    ld.global.f32 %f1, [%rd10];    // rstd

    mov.f32 %f2, 0f00000000;       // dot
    mov.u64 %rd11, 0;
RNBX_S:
    setp.ge.u64 %p2, %rd11, %rd3;
    @%p2 bra RNBX_S_END;
    shl.b64 %rd12, %rd11, 2;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.f32 %f3, [%rd13];
    add.u64 %rd13, %rd6, %rd12;
    ld.global.f32 %f4, [%rd13];
    add.u64 %rd13, %rd8, %rd12;
    ld.global.f32 %f5, [%rd13];
    mul.f32 %f6, %f3, %f5;
    fma.rn.f32 %f2, %f6, %f4, %f2;
    add.u64 %rd11, %rd11, 1;
    bra RNBX_S;
RNBX_S_END:

    cvt.rn.f32.u64 %f7, %rd3;
    mul.f32 %f8, %f1, %f1;
    mul.f32 %f8, %f8, %f1;
    div.rn.f32 %f8, %f8, %f7;
    mul.f32 %f8, %f8, %f2;         // coeff = (rstd^3 / cols) * dot

    mov.u64 %rd11, 0;
RNBX_W:
    setp.ge.u64 %p2, %rd11, %rd3;
    @%p2 bra RNBX_DONE;
    shl.b64 %rd12, %rd11, 2;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.f32 %f3, [%rd13];
    add.u64 %rd13, %rd6, %rd12;
    ld.global.f32 %f4, [%rd13];
    add.u64 %rd13, %rd8, %rd12;
    ld.global.f32 %f5, [%rd13];
    mul.f32 %f9, %f3, %f5;
    mul.f32 %f9, %f9, %f1;
    mul.f32 %f10, %f4, %f8;
    sub.f32 %f9, %f9, %f10;
    add.u64 %rd13, %rd7, %rd12;
    st.global.f32 [%rd13], %f9;
    add.u64 %rd11, %rd11, 1;
    bra RNBX_W;
RNBX_DONE:
    ret;
}

.visible .entry ochre_row_rmsnorm_bwd_dg_f32(
    .param .u64 p_dy,
    .param .u64 p_x,
    .param .u64 p_rstd,
    .param .u64 p_dg,
    .param .u64 p_rows,
    .param .u64 p_cols
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<13>;
    .reg .f32 %f<6>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_cols];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra RNBG_DONE;

    ld.param.u64 %rd3, [p_rows];
    ld.param.u64 %rd4, [p_dy];
    ld.param.u64 %rd5, [p_x];
    ld.param.u64 %rd6, [p_rstd];
    shl.b64 %rd7, %rd1, 2;
    add.u64 %rd4, %rd4, %rd7;
    add.u64 %rd5, %rd5, %rd7;
    shl.b64 %rd8, %rd2, 2;

    mov.f32 %f1, 0f00000000;
    mov.u64 %rd9, 0;
RNBG_LOOP:
    setp.ge.u64 %p2, %rd9, %rd3;
    @%p2 bra RNBG_END;
    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f3, [%rd5];
    shl.b64 %rd10, %rd9, 2;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.f32 %f4, [%rd11];
    mul.f32 %f5, %f3, %f4;
    fma.rn.f32 %f1, %f2, %f5, %f1;
    add.u64 %rd4, %rd4, %rd8;
    add.u64 %rd5, %rd5, %rd8;
    add.u64 %rd9, %rd9, 1;
    bra RNBG_LOOP;
RNBG_END:
    ld.param.u64 %rd12, [p_dg];
    add.u64 %rd12, %rd12, %rd7;
    st.global.f32 [%rd12], %f1;
RNBG_DONE:
    ret;
}

.visible .entry ochre_dropout_fwd_f32(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u64 p_mask,
    .param .u64 p_n,
    .param .f32 p_p,
    .param .f32 p_scale,
    .param .u64 p_seed
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<11>;
    .reg .f32 %f<7>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra DO_DONE;

    // SplitMix64 stateless hash
    ld.param.u64 %rd3, [p_seed];
    xor.b64 %rd4, %rd3, %rd1;
    add.u64 %rd4, %rd4, 11400714819323198485;
    shr.u64 %rd5, %rd4, 30;
    xor.b64 %rd4, %rd4, %rd5;
    mul.lo.u64 %rd4, %rd4, 13787848793156543929;
    shr.u64 %rd5, %rd4, 27;
    xor.b64 %rd4, %rd4, %rd5;
    mul.lo.u64 %rd4, %rd4, 10723151780598845931;
    shr.u64 %rd5, %rd4, 31;
    xor.b64 %rd4, %rd4, %rd5;

    shr.u64 %rd4, %rd4, 40;
    cvt.u32.u64 %r4, %rd4;
    cvt.rn.f32.u32 %f1, %r4;
    mov.f32 %f2, 0f33800000;       // 1.0 / 16777216.0 = 2^-24
    mul.f32 %f1, %f1, %f2;

    ld.param.f32 %f3, [p_p];
    ld.param.f32 %f4, [p_scale];
    mov.f32 %f5, 0f00000000;
    setp.lt.f32 %p2, %f1, %f3;
    selp.f32 %f4, %f5, %f4, %p2;   // mask

    shl.b64 %rd6, %rd1, 2;
    ld.param.u64 %rd7, [p_x];
    ld.param.u64 %rd8, [p_out];
    ld.param.u64 %rd9, [p_mask];
    add.u64 %rd7, %rd7, %rd6;
    add.u64 %rd8, %rd8, %rd6;
    add.u64 %rd9, %rd9, %rd6;
    ld.global.f32 %f6, [%rd7];
    mul.f32 %f6, %f6, %f4;
    st.global.f32 [%rd9], %f4;
    st.global.f32 [%rd8], %f6;
DO_DONE:
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
        fill_f32: get_fn("ochre_fill_f32")?,
        copy_f32: get_fn("ochre_copy_f32")?,
        add_inplace_f32: get_fn("ochre_add_inplace_f32")?,
        mul_f32: get_fn("ochre_mul_f32")?,
        f32_to_bf16: get_fn("ochre_f32_to_bf16")?,
        gather_rows_f32: get_fn("ochre_gather_rows_f32")?,
        gather_u64: get_fn("ochre_gather_u64")?,
        add_bias_row_f32: get_fn("ochre_add_bias_row_f32")?,
        sum_rows_f32: get_fn("ochre_sum_rows_f32")?,
        act_fwd_f32: get_fn("ochre_act_fwd_f32")?,
        act_bwd_f32: get_fn("ochre_act_bwd_f32")?,
        mse_grad_f32: get_fn("ochre_mse_grad_f32")?,
        reduce_sum_add_f32: get_fn("ochre_reduce_sum_add_f32")?,
        reduce_sum_sq_add_f32: get_fn("ochre_reduce_sum_sq_add_f32")?,
        clip_scale_f32: get_fn("ochre_clip_scale_f32")?,
        sgd_step_f32: get_fn("ochre_sgd_step_f32")?,
        adam_step_f32: get_fn("ochre_adam_step_f32")?,
        nhwc_to_nchw_bias_f32: get_fn("ochre_nhwc_to_nchw_bias_f32")?,
        nchw_to_nhwc_f32: get_fn("ochre_nchw_to_nhwc_f32")?,
        sum_nchw_channel_f32: get_fn("ochre_sum_nchw_channel_f32")?,
        bn2d_fwd_train_f32: get_fn("ochre_bn2d_fwd_train_f32")?,
        bn2d_fwd_eval_f32: get_fn("ochre_bn2d_fwd_eval_f32")?,
        bn2d_bwd_f32: get_fn("ochre_bn2d_bwd_f32")?,
        max_pool2d_bwd_f32: get_fn("ochre_max_pool2d_bwd_f32")?,
        global_avg_pool2d_fwd_f32: get_fn("ochre_global_avg_pool2d_fwd_f32")?,
        global_avg_pool2d_bwd_f32: get_fn("ochre_global_avg_pool2d_bwd_f32")?,
        row_layernorm_bwd_dx_f32: get_fn("ochre_row_layernorm_bwd_dx_f32")?,
        row_layernorm_bwd_dgb_f32: get_fn("ochre_row_layernorm_bwd_dgb_f32")?,
        row_rmsnorm_bwd_dx_f32: get_fn("ochre_row_rmsnorm_bwd_dx_f32")?,
        row_rmsnorm_bwd_dg_f32: get_fn("ochre_row_rmsnorm_bwd_dg_f32")?,
        dropout_fwd_f32: get_fn("ochre_dropout_fwd_f32")?,
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

// ============================================================================
// VRAM-Resident CUDA Device Buffers & Session (Zero Per-Op Host Copies / Sync)
// ============================================================================

/// Safe RAII wrapper around a CUDA device memory allocation (`CuDevicePtr`).
/// Automatically binds its owning CUDA context and calls `cuMemFree` on drop.
pub struct CudaDeviceBuf {
    ptr: CuDevicePtr,
    byte_len: usize,
    ctx: CuContext,
    cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
}

// SAFETY: `CudaDeviceBuf` owns a unique device pointer and binds its CUDA context on drop.
unsafe impl Send for CudaDeviceBuf {}

impl Drop for CudaDeviceBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: Bind owning CUDA context before freeing the owned device pointer.
            unsafe {
                let _ = (self.cu_ctx_set_current)(self.ctx);
                let _ = (self.cu_mem_free)(self.ptr);
            }
            self.ptr = 0;
            self.byte_len = 0;
        }
    }
}

impl CudaDeviceBuf {
    /// Returns the raw CUDA device pointer (`u64`).
    #[inline]
    pub fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }

    /// Returns the allocated size in bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Returns the number of `f32` elements that fit in this buffer.
    #[inline]
    pub fn len_f32(&self) -> usize {
        self.byte_len / 4
    }

    /// Returns the number of `u64` elements that fit in this buffer.
    #[inline]
    pub fn len_u64(&self) -> usize {
        self.byte_len / 8
    }

    /// Returns a device pointer offset by `elem_offset` `f32` elements.
    #[inline]
    pub fn ptr_offset_f32(&self, elem_offset: usize) -> CuDevicePtr {
        self.ptr + (elem_offset * 4) as u64
    }

    /// Returns a device pointer offset by `elem_offset` `u64` elements.
    #[inline]
    pub fn ptr_offset_u64(&self, elem_offset: usize) -> CuDevicePtr {
        self.ptr + (elem_offset * 8) as u64
    }
}

/// Device-resident CUDA session holding an active CUDA context, cuBLAS handle, and PTX module.
/// All kernel launches and cuBLAS GEMMs enqueue asynchronously on the default stream (stream 0)
/// in strict FIFO order without intermediate `cuCtxSynchronize()` stalls.
pub struct CudaVramSession<'a> {
    driver: &'a CudaDriver,
    rt: &'a CudaNnRuntime,
    state: &'a mut CudaNnCtxState,
    bf16_gemm_ex_ok: bool,
}

fn launch_grid_async(
    rt: &CudaNnRuntime,
    func: CuFunction,
    grid_x: u32,
    block_x: u32,
    params: &mut [*mut c_void],
) -> OchreResult<()> {
    if grid_x == 0 || block_x == 0 {
        return Ok(());
    }
    // SAFETY: Caller supplies kernel parameters matching `func`'s PTX signature on the bound context.
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
    Ok(())
}

fn launch_1d_async(
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
    launch_grid_async(rt, func, grid_x, block_x, params)
}

/// Runs `f` inside a VRAM-resident CUDA session on `gpu_id`.
/// Returns `None` if CUDA driver, cuBLAS, or universal PTX module is unavailable (clean CPU fallback).
pub fn with_vram_session<R>(
    gpu_id: usize,
    f: impl FnOnce(&mut CudaVramSession<'_>) -> OchreResult<R>,
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
    state.ptx.as_ref()?;
    // SAFETY: Bind CUDA context for this calling thread.
    let rc = unsafe { (driver.cu_ctx_set_current)(state.workspace.ctx) };
    if rc != CUDA_SUCCESS {
        return Some(Err(OchreError::DeviceError(format!(
            "CUDA cuCtxSetCurrent failed with code {rc} ( >_< )"
        ))));
    }
    let bf16_gemm_ex_ok = rt.cublas_gemm_ex.is_some();
    let mut session = CudaVramSession {
        driver,
        rt,
        state: &mut state,
        bf16_gemm_ex_ok,
    };
    Some(f(&mut session))
}

impl<'a> CudaVramSession<'a> {
    #[inline]
    fn ptx(&self) -> OchreResult<&NnPtxFuncs> {
        self.state
            .ptx
            .as_ref()
            .ok_or_else(|| OchreError::DeviceError("CUDA NN PTX not loaded ( >_< )".to_string()))
    }

    /// Allocates `byte_len` bytes in VRAM (minimum 4 bytes so pointer is always non-null).
    pub fn alloc_bytes(&self, byte_len: usize) -> OchreResult<CudaDeviceBuf> {
        let alloc_sz = byte_len.max(4);
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: Context is bound by `CudaVramSession`; `ptr` receives a valid device allocation.
        let rc = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, alloc_sz) };
        if rc != CUDA_SUCCESS || ptr == 0 {
            return Err(OchreError::DeviceError(format!(
                "CUDA VRAM cuMemAlloc({alloc_sz}) failed with code {rc} (´；ω；`)"
            )));
        }
        Ok(CudaDeviceBuf {
            ptr,
            byte_len: alloc_sz,
            ctx: self.state.workspace.ctx,
            cu_ctx_set_current: self.driver.cu_ctx_set_current,
            cu_mem_free: self.driver.cu_mem_free,
        })
    }

    /// Allocates `len` uninitialized `f32` elements in VRAM.
    pub fn alloc_f32(&self, len: usize) -> OchreResult<CudaDeviceBuf> {
        self.alloc_bytes(len.saturating_mul(4))
    }

    /// Allocates `len` zero-initialized `f32` elements in VRAM.
    pub fn alloc_zeros_f32(&self, len: usize) -> OchreResult<CudaDeviceBuf> {
        let buf = self.alloc_f32(len)?;
        if len > 0 {
            self.fill_f32(buf.ptr(), 0.0, len)?;
        }
        Ok(buf)
    }

    /// Allocates `len` uninitialized `u64` elements in VRAM.
    pub fn alloc_u64(&self, len: usize) -> OchreResult<CudaDeviceBuf> {
        self.alloc_bytes(len.saturating_mul(8))
    }

    /// Allocates and uploads an `f32` slice to VRAM.
    pub fn upload_f32(&self, src: &[f32]) -> OchreResult<CudaDeviceBuf> {
        let buf = self.alloc_f32(src.len())?;
        htod_f32(self.driver, buf.ptr(), src)?;
        Ok(buf)
    }

    /// Allocates and uploads a `u64` slice to VRAM.
    pub fn upload_u64(&self, src: &[u64]) -> OchreResult<CudaDeviceBuf> {
        let buf = self.alloc_u64(src.len())?;
        self.htod_u64(&buf, src)?;
        Ok(buf)
    }

    /// Copies a host `f32` slice into an existing device buffer.
    pub fn htod_f32(&self, dst: &CudaDeviceBuf, src: &[f32]) -> OchreResult<()> {
        htod_f32(self.driver, dst.ptr(), src)
    }

    /// Copies a host `u64` slice into an existing device buffer.
    pub fn htod_u64(&self, dst: &CudaDeviceBuf, src: &[u64]) -> OchreResult<()> {
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `u64` has no padding and valid byte representation for device transfer.
        let bytes = unsafe {
            std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
        };
        htod_bytes(self.driver, dst.ptr(), bytes)
    }

    /// Copies a device `f32` buffer back to host `dst` (implicitly synchronizes default stream).
    pub fn dtoh_f32(&self, dst: &mut [f32], src: &CudaDeviceBuf) -> OchreResult<()> {
        dtoh_f32(self.driver, dst, src.ptr())
    }

    /// Synchronizes the default stream on the bound CUDA context.
    pub fn sync(&self) -> OchreResult<()> {
        // SAFETY: Synchronizes bound CUDA context.
        let rc = unsafe { (self.rt.cu_ctx_synchronize)() };
        if rc != CUDA_SUCCESS {
            return Err(OchreError::DeviceError(format!(
                "CUDA cuCtxSynchronize failed with code {rc} ( >_< )"
            )));
        }
        Ok(())
    }

    /// Device-resident cuBLAS GEMM (`C = alpha * op(A) * op(B) + beta * C`) on raw `CuDevicePtr`s.
    /// Uses Tensor Core `cublasGemmEx` (`CUDA_R_16BF`) when `dtype == DType::BF16` and supported,
    /// seamlessly falling back to `cublasSgemm_v2` on pre-Ampere GPUs (e.g. T4 `sm_75`).
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_dev(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        d_a: CuDevicePtr,
        d_b: CuDevicePtr,
        d_c: CuDevicePtr,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
        dtype: DType,
    ) -> OchreResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };

        if dtype == DType::BF16
            && self.bf16_gemm_ex_ok
            && m.saturating_mul(n).saturating_mul(k) >= 4096
            && let Some(gemm_ex) = self.rt.cublas_gemm_ex
        {
            let a_elems = m * k;
            let b_elems = k * n;
            let off_b = align256(a_elems * 2);
            let total_bytes = off_b + align256(b_elems * 2);
            if let Ok(base) = self.state.workspace.ensure_bytes(self.driver, total_bytes) {
                let d_a16 = base;
                let d_b16 = base + off_b as u64;
                self.f32_to_bf16_dev(d_a, d_a16, a_elems)?;
                self.f32_to_bf16_dev(d_b, d_b16, b_elems)?;
                // SAFETY: `d_a16`, `d_b16`, `d_c` are valid device pointers on the bound context.
                let status = unsafe {
                    gemm_ex(
                        self.state.handle,
                        op_b,
                        op_a,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        d_b16,
                        CUDA_R_16BF,
                        ldb,
                        d_a16,
                        CUDA_R_16BF,
                        lda,
                        &beta,
                        d_c,
                        CUDA_R_32F,
                        n as i32,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                };
                if status == CUBLAS_STATUS_SUCCESS {
                    return Ok(());
                }
                // Pre-Ampere GPUs (e.g. T4 sm_75) lack hardware BF16 GemmEx; disable for session.
                self.bf16_gemm_ex_ok = false;
            }
        }

        // SAFETY: All device pointers are valid allocations on the bound CUDA context.
        let status = unsafe {
            (self.rt.cublas_sgemm)(
                self.state.handle,
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
        check_cublas(status, "cublasSgemm_v2(dev)")
    }

    /// Converts `n` f32 elements at `d_src` to bf16 at `d_dst` on GPU.
    pub fn f32_to_bf16_dev(
        &self,
        mut d_src: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.f32_to_bf16;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 3] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Fills `d_dst[0..n]` with `val` on GPU.
    pub fn fill_f32(&self, mut d_dst: CuDevicePtr, mut val: f32, n: usize) -> OchreResult<()> {
        let func = self.ptx()?.fill_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 3] = [
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut val as *mut f32).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Copies `n` f32 elements from `d_src` to `d_dst` on GPU.
    pub fn copy_f32(
        &self,
        mut d_src: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.copy_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 3] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// In-place elementwise `d_dst[i] += d_src[i]` on GPU.
    pub fn add_inplace_dev_f32(
        &self,
        mut d_dst: CuDevicePtr,
        mut d_src: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.add_inplace_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 3] = [
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Elementwise `d_out[i] = d_a[i] * d_b[i]` on GPU.
    pub fn mul_dev_f32(
        &self,
        mut d_a: CuDevicePtr,
        mut d_b: CuDevicePtr,
        mut d_out: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.mul_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_a as *mut CuDevicePtr).cast(),
            (&mut d_b as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Gathers `batch` rows of width `cols` from `d_src` using `d_indices` (u64) into `d_dst`.
    pub fn gather_rows_f32(
        &self,
        mut d_src: CuDevicePtr,
        mut d_indices: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        batch: usize,
        cols: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.gather_rows_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = cols as u64;
        let mut params: [*mut c_void; 5] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_indices as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, batch * cols, 256, &mut params)
    }

    /// Gathers `batch` u64 elements from `d_src` using `d_indices` into `d_dst`.
    pub fn gather_u64(
        &self,
        mut d_src: CuDevicePtr,
        mut d_indices: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        batch: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.gather_u64;
        let mut b_u64 = batch as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_indices as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, batch, 256, &mut params)
    }

    /// In-place `Y[r, c] += bias[c]` on GPU.
    pub fn add_bias_row_f32(
        &self,
        mut d_y: CuDevicePtr,
        mut d_bias: CuDevicePtr,
        rows: usize,
        cols: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.add_bias_row_f32;
        let mut r_u64 = rows as u64;
        let mut c_u64 = cols as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_y as *mut CuDevicePtr).cast(),
            (&mut d_bias as *mut CuDevicePtr).cast(),
            (&mut r_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, rows * cols, 256, &mut params)
    }

    /// Computes `db[c] = sum_r dY[r, c]` on GPU.
    pub fn sum_rows_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_db: CuDevicePtr,
        rows: usize,
        cols: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.sum_rows_f32;
        let mut r_u64 = rows as u64;
        let mut c_u64 = cols as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_db as *mut CuDevicePtr).cast(),
            (&mut r_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, cols, 256, &mut params)
    }

    /// Device-resident activation forward (`kind`: 0=ReLU, 1=Sigmoid, 2=Tanh, 3=GELU, 4=SiLU).
    pub fn act_fwd_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_out: CuDevicePtr,
        n: usize,
        mut kind: u32,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let mut n_u64 = n as u64;
        match kind {
            3 => {
                let mut params: [*mut c_void; 3] = [
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_out as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                ];
                launch_1d_async(self.rt, ptx.gelu, n, 256, &mut params)
            }
            4 => {
                let mut params: [*mut c_void; 3] = [
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_out as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                ];
                launch_1d_async(self.rt, ptx.silu, n, 256, &mut params)
            }
            _ => {
                let mut params: [*mut c_void; 4] = [
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_out as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                    (&mut kind as *mut u32).cast(),
                ];
                launch_1d_async(self.rt, ptx.act_fwd_f32, n, 256, &mut params)
            }
        }
    }

    /// Device-resident activation backward (`kind`: 0=ReLU, 1=Sigmoid, 2=Tanh, 3=GELU, 4=SiLU).
    pub fn act_bwd_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_y: CuDevicePtr,
        mut d_dy: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        n: usize,
        mut kind: u32,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let mut n_u64 = n as u64;
        match kind {
            3 => {
                let mut params: [*mut c_void; 4] = [
                    (&mut d_dy as *mut CuDevicePtr).cast(),
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_dx as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                ];
                launch_1d_async(self.rt, ptx.gelu_bwd, n, 256, &mut params)
            }
            4 => {
                let mut params: [*mut c_void; 4] = [
                    (&mut d_dy as *mut CuDevicePtr).cast(),
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_dx as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                ];
                launch_1d_async(self.rt, ptx.silu_bwd, n, 256, &mut params)
            }
            _ => {
                let mut params: [*mut c_void; 6] = [
                    (&mut d_x as *mut CuDevicePtr).cast(),
                    (&mut d_y as *mut CuDevicePtr).cast(),
                    (&mut d_dy as *mut CuDevicePtr).cast(),
                    (&mut d_dx as *mut CuDevicePtr).cast(),
                    (&mut n_u64 as *mut u64).cast(),
                    (&mut kind as *mut u32).cast(),
                ];
                launch_1d_async(self.rt, ptx.act_bwd_f32, n, 256, &mut params)
            }
        }
    }

    /// Device-resident fused CrossEntropy + gradient; accumulates `batch_loss * rows` into `d_epoch_loss`.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_grad_dev_f32(
        &self,
        mut d_logits: CuDevicePtr,
        mut d_targets: CuDevicePtr,
        mut d_dlogits: CuDevicePtr,
        mut d_losses: CuDevicePtr,
        d_epoch_loss: CuDevicePtr,
        rows: usize,
        cols: usize,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut ls = 0.0f32;
        let mut ign_u64 = u64::MAX;
        let mut has_ign = 0u32;
        let mut inv_valid = 1.0f32 / (rows as f32);
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
        launch_1d_async(self.rt, ptx.cross_entropy_grad, rows, 128, &mut params)?;
        self.reduce_sum_add_f32(d_losses, d_epoch_loss, rows, 1.0)
    }

    /// Device-resident fused MSE loss + gradient; accumulates `batch_loss * rows` into `d_epoch_loss`.
    #[allow(clippy::too_many_arguments)]
    pub fn mse_grad_dev_f32(
        &self,
        mut d_pred: CuDevicePtr,
        mut d_target: CuDevicePtr,
        mut d_dpred: CuDevicePtr,
        mut d_sq: CuDevicePtr,
        d_epoch_loss: CuDevicePtr,
        rows: usize,
        cols: usize,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let n = rows * cols;
        let mut n_u64 = n as u64;
        let mut scale = 2.0f32 / (n as f32);
        let mut params: [*mut c_void; 6] = [
            (&mut d_pred as *mut CuDevicePtr).cast(),
            (&mut d_target as *mut CuDevicePtr).cast(),
            (&mut d_dpred as *mut CuDevicePtr).cast(),
            (&mut d_sq as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        launch_1d_async(self.rt, ptx.mse_grad_f32, n, 256, &mut params)?;
        self.reduce_sum_add_f32(d_sq, d_epoch_loss, n, 1.0f32 / (cols as f32))
    }

    /// Reduces `d_src[0..n]` and adds `sum * scale` into scalar `d_accum[0]` on GPU.
    pub fn reduce_sum_add_f32(
        &self,
        mut d_src: CuDevicePtr,
        mut d_accum: CuDevicePtr,
        n: usize,
        mut scale: f32,
    ) -> OchreResult<()> {
        let func = self.ptx()?.reduce_sum_add_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_accum as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        launch_grid_async(self.rt, func, 1, 256, &mut params)
    }

    /// Reduces sum-of-squares of `d_src[0..n]` and adds into scalar `d_accum[0]` on GPU.
    pub fn reduce_sum_sq_add_f32(
        &self,
        mut d_src: CuDevicePtr,
        mut d_accum: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.reduce_sum_sq_add_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 3] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_accum as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_grid_async(self.rt, func, 1, 256, &mut params)
    }

    /// Scales `d_grad[0..n]` in-place by `max_norm / (sqrt(d_norm_sq[0]) + 1e-6)` if norm > max_norm.
    pub fn clip_scale_dev_f32(
        &self,
        mut d_grad: CuDevicePtr,
        mut d_norm_sq: CuDevicePtr,
        mut max_norm: f32,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.clip_scale_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_grad as *mut CuDevicePtr).cast(),
            (&mut d_norm_sq as *mut CuDevicePtr).cast(),
            (&mut max_norm as *mut f32).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Device-resident in-place AdamW step.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_dev_f32(
        &self,
        mut d_param: CuDevicePtr,
        mut d_grad: CuDevicePtr,
        mut d_m: CuDevicePtr,
        mut d_v: CuDevicePtr,
        n: usize,
        mut lr: f32,
        mut beta1: f32,
        mut beta2: f32,
        mut eps: f32,
        mut weight_decay: f32,
        step: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.adamw_step;
        let mut inv_bc1 = (1.0 / (1.0 - (beta1 as f64).powi(step as i32))) as f32;
        let mut inv_bc2 = (1.0 / (1.0 - (beta2 as f64).powi(step as i32))) as f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 12] = [
            (&mut d_param as *mut CuDevicePtr).cast(),
            (&mut d_grad as *mut CuDevicePtr).cast(),
            (&mut d_m as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut lr as *mut f32).cast(),
            (&mut beta1 as *mut f32).cast(),
            (&mut beta2 as *mut f32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut weight_decay as *mut f32).cast(),
            (&mut inv_bc1 as *mut f32).cast(),
            (&mut inv_bc2 as *mut f32).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Device-resident in-place Adam step (coupled L2 weight decay).
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_dev_f32(
        &self,
        mut d_param: CuDevicePtr,
        mut d_grad: CuDevicePtr,
        mut d_m: CuDevicePtr,
        mut d_v: CuDevicePtr,
        n: usize,
        mut lr: f32,
        mut beta1: f32,
        mut beta2: f32,
        mut eps: f32,
        mut weight_decay: f32,
        step: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.adam_step_f32;
        let mut inv_bc1 = (1.0 / (1.0 - (beta1 as f64).powi(step as i32))) as f32;
        let mut inv_bc2 = (1.0 / (1.0 - (beta2 as f64).powi(step as i32))) as f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 12] = [
            (&mut d_param as *mut CuDevicePtr).cast(),
            (&mut d_grad as *mut CuDevicePtr).cast(),
            (&mut d_m as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut lr as *mut f32).cast(),
            (&mut beta1 as *mut f32).cast(),
            (&mut beta2 as *mut f32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut weight_decay as *mut f32).cast(),
            (&mut inv_bc1 as *mut f32).cast(),
            (&mut inv_bc2 as *mut f32).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Device-resident in-place SGD step with momentum & L2 weight decay.
    #[allow(clippy::too_many_arguments)]
    pub fn sgd_step_dev_f32(
        &self,
        mut d_param: CuDevicePtr,
        mut d_grad: CuDevicePtr,
        mut d_vel: CuDevicePtr,
        n: usize,
        mut lr: f32,
        mut momentum: f32,
        mut weight_decay: f32,
    ) -> OchreResult<()> {
        let func = self.ptx()?.sgd_step_f32;
        let mut n_u64 = n as u64;
        let mut has_mom = u32::from(momentum > 0.0);
        let mut params: [*mut c_void; 8] = [
            (&mut d_param as *mut CuDevicePtr).cast(),
            (&mut d_grad as *mut CuDevicePtr).cast(),
            (&mut d_vel as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut lr as *mut f32).cast(),
            (&mut momentum as *mut f32).cast(),
            (&mut weight_decay as *mut f32).cast(),
            (&mut has_mom as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// Device-resident im2col.
    #[allow(clippy::too_many_arguments)]
    pub fn im2col_dev_f32(
        &self,
        mut d_in: CuDevicePtr,
        mut d_col: CuDevicePtr,
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
    ) -> OchreResult<()> {
        let func = self.ptx()?.im2col;
        let total = batch * out_h * out_w * c * kh * kw;
        let mut total_u64 = total as u64;
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
        launch_1d_async(self.rt, func, total, 256, &mut params)
    }

    /// Device-resident col2im.
    #[allow(clippy::too_many_arguments)]
    pub fn col2im_dev_f32(
        &self,
        mut d_col: CuDevicePtr,
        mut d_img: CuDevicePtr,
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
    ) -> OchreResult<()> {
        let func = self.ptx()?.col2im;
        let total_in = batch * c * h * w;
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
        launch_1d_async(self.rt, func, total_in, 256, &mut params)
    }

    /// Transposes `[batch * spatial, c]` -> `[batch, c, spatial]` and optionally adds per-channel bias.
    #[allow(clippy::too_many_arguments)]
    pub fn nhwc_to_nchw_bias_dev_f32(
        &self,
        mut d_ymat: CuDevicePtr,
        mut d_bias: CuDevicePtr,
        mut d_out: CuDevicePtr,
        batch: usize,
        c: usize,
        spatial: usize,
        has_bias: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.nhwc_to_nchw_bias_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = c as u64;
        let mut s_u64 = spatial as u64;
        let mut hb = u32::from(has_bias);
        let mut params: [*mut c_void; 7] = [
            (&mut d_ymat as *mut CuDevicePtr).cast(),
            (&mut d_bias as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut hb as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, batch * c * spatial, 256, &mut params)
    }

    /// Transposes `[batch, c, spatial]` -> `[batch * spatial, c]` on GPU.
    pub fn nchw_to_nhwc_dev_f32(
        &self,
        mut d_nchw: CuDevicePtr,
        mut d_nhwc: CuDevicePtr,
        batch: usize,
        c: usize,
        spatial: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.nchw_to_nhwc_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = c as u64;
        let mut s_u64 = spatial as u64;
        let mut params: [*mut c_void; 5] = [
            (&mut d_nchw as *mut CuDevicePtr).cast(),
            (&mut d_nhwc as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, batch * c * spatial, 256, &mut params)
    }

    /// Sums NCHW `[batch, c, spatial]` across `(batch, spatial)` into `d_db[c]` on GPU.
    pub fn sum_nchw_channel_dev_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_db: CuDevicePtr,
        batch: usize,
        c: usize,
        spatial: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.sum_nchw_channel_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = c as u64;
        let mut s_u64 = spatial as u64;
        let mut params: [*mut c_void; 5] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_db as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
        ];
        launch_grid_async(self.rt, func, c as u32, 256, &mut params)
    }

    /// Device-resident BatchNorm2d forward (training mode: computes stats, updates running stats, normalizes).
    #[allow(clippy::too_many_arguments)]
    pub fn batchnorm2d_fwd_train_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_beta: CuDevicePtr,
        mut d_rm: CuDevicePtr,
        mut d_rv: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_smean: CuDevicePtr,
        mut d_sinv: CuDevicePtr,
        batch: usize,
        c: usize,
        plane: usize,
        mut eps: f32,
        mut momentum: f32,
        has_affine: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.bn2d_fwd_train_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = c as u64;
        let mut p_u64 = plane as u64;
        let mut ha = u32::from(has_affine);
        let mut params: [*mut c_void; 14] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_gamma as *mut CuDevicePtr).cast(),
            (&mut d_beta as *mut CuDevicePtr).cast(),
            (&mut d_rm as *mut CuDevicePtr).cast(),
            (&mut d_rv as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_smean as *mut CuDevicePtr).cast(),
            (&mut d_sinv as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut p_u64 as *mut u64).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut momentum as *mut f32).cast(),
            (&mut ha as *mut u32).cast(),
        ];
        launch_grid_async(self.rt, func, c as u32, 256, &mut params)
    }

    /// Device-resident BatchNorm2d forward (eval mode: uses running_mean / running_var).
    #[allow(clippy::too_many_arguments)]
    pub fn batchnorm2d_fwd_eval_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_beta: CuDevicePtr,
        mut d_rm: CuDevicePtr,
        mut d_rv: CuDevicePtr,
        mut d_out: CuDevicePtr,
        batch: usize,
        c: usize,
        plane: usize,
        mut eps: f32,
        has_affine: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.bn2d_fwd_eval_f32;
        let total = batch * c * plane;
        let mut total_u64 = total as u64;
        let mut c_u64 = c as u64;
        let mut p_u64 = plane as u64;
        let mut ha = u32::from(has_affine);
        let mut params: [*mut c_void; 11] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_gamma as *mut CuDevicePtr).cast(),
            (&mut d_beta as *mut CuDevicePtr).cast(),
            (&mut d_rm as *mut CuDevicePtr).cast(),
            (&mut d_rv as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut total_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut p_u64 as *mut u64).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut ha as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, total, 256, &mut params)
    }

    /// Device-resident BatchNorm2d backward (training mode).
    #[allow(clippy::too_many_arguments)]
    pub fn batchnorm2d_bwd_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_dy: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_smean: CuDevicePtr,
        mut d_sinv: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        mut d_dg: CuDevicePtr,
        mut d_db: CuDevicePtr,
        batch: usize,
        c: usize,
        plane: usize,
        has_affine: bool,
        need_dx: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.bn2d_bwd_f32;
        let mut b_u64 = batch as u64;
        let mut c_u64 = c as u64;
        let mut p_u64 = plane as u64;
        let mut ha = u32::from(has_affine);
        let mut ndx = u32::from(need_dx);
        let mut params: [*mut c_void; 13] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_gamma as *mut CuDevicePtr).cast(),
            (&mut d_smean as *mut CuDevicePtr).cast(),
            (&mut d_sinv as *mut CuDevicePtr).cast(),
            (&mut d_dx as *mut CuDevicePtr).cast(),
            (&mut d_dg as *mut CuDevicePtr).cast(),
            (&mut d_db as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut c_u64 as *mut u64).cast(),
            (&mut p_u64 as *mut u64).cast(),
            (&mut ha as *mut u32).cast(),
            (&mut ndx as *mut u32).cast(),
        ];
        launch_grid_async(self.rt, func, c as u32, 256, &mut params)
    }

    /// Device-resident MaxPool2d forward.
    #[allow(clippy::too_many_arguments)]
    pub fn maxpool2d_fwd_dev_f32(
        &self,
        mut d_in: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_argmax: CuDevicePtr,
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
    ) -> OchreResult<()> {
        let func = self.ptx()?.max_pool2d;
        let total_out = batch * c * out_h * out_w;
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
            (&mut d_argmax as *mut CuDevicePtr).cast(),
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
        launch_1d_async(self.rt, func, total_out, 256, &mut params)
    }

    /// Device-resident MaxPool2d backward (zeroes `d_dx` then scatters `d_dy` via atomic add).
    pub fn maxpool2d_bwd_dev_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_argmax: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        total_in: usize,
        total_out: usize,
    ) -> OchreResult<()> {
        self.fill_f32(d_dx, 0.0, total_in)?;
        let func = self.ptx()?.max_pool2d_bwd_f32;
        let mut to_u64 = total_out as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_argmax as *mut CuDevicePtr).cast(),
            (&mut d_dx as *mut CuDevicePtr).cast(),
            (&mut to_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, total_out, 256, &mut params)
    }

    /// Device-resident GlobalAvgPool2d forward.
    pub fn global_avg_pool2d_fwd_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_out: CuDevicePtr,
        bc_total: usize,
        plane: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.global_avg_pool2d_fwd_f32;
        let mut bc_u64 = bc_total as u64;
        let mut pl_u64 = plane as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut bc_u64 as *mut u64).cast(),
            (&mut pl_u64 as *mut u64).cast(),
        ];
        launch_grid_async(self.rt, func, bc_total as u32, 128, &mut params)
    }

    /// Device-resident GlobalAvgPool2d backward.
    pub fn global_avg_pool2d_bwd_dev_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        total_in: usize,
        plane: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.global_avg_pool2d_bwd_f32;
        let mut ti_u64 = total_in as u64;
        let mut pl_u64 = plane as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_dx as *mut CuDevicePtr).cast(),
            (&mut ti_u64 as *mut u64).cast(),
            (&mut pl_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, total_in, 256, &mut params)
    }

    /// Device-resident LayerNorm forward.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_fwd_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_beta: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_mean: CuDevicePtr,
        mut d_rstd: CuDevicePtr,
        rows: usize,
        cols: usize,
        has_gamma: bool,
        has_beta: bool,
        mut eps: f32,
    ) -> OchreResult<()> {
        let func = self.ptx()?.row_layernorm;
        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut hg = u32::from(has_gamma);
        let mut hb = u32::from(has_beta);
        let mut params: [*mut c_void; 11] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_gamma as *mut CuDevicePtr).cast(),
            (&mut d_beta as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_mean as *mut CuDevicePtr).cast(),
            (&mut d_rstd as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
            (&mut hg as *mut u32).cast(),
            (&mut hb as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        launch_1d_async(self.rt, func, rows, 128, &mut params)
    }

    /// Device-resident LayerNorm backward.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_bwd_dev_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_mean: CuDevicePtr,
        mut d_rstd: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        mut d_dg: CuDevicePtr,
        mut d_db: CuDevicePtr,
        rows: usize,
        cols: usize,
        has_gamma: bool,
        has_beta: bool,
        need_dx: bool,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut hg = u32::from(has_gamma);
        let mut hb = u32::from(has_beta);
        if need_dx {
            let mut params: [*mut c_void; 9] = [
                (&mut d_dy as *mut CuDevicePtr).cast(),
                (&mut d_x as *mut CuDevicePtr).cast(),
                (&mut d_gamma as *mut CuDevicePtr).cast(),
                (&mut d_mean as *mut CuDevicePtr).cast(),
                (&mut d_rstd as *mut CuDevicePtr).cast(),
                (&mut d_dx as *mut CuDevicePtr).cast(),
                (&mut rows_u64 as *mut u64).cast(),
                (&mut cols_u64 as *mut u64).cast(),
                (&mut hg as *mut u32).cast(),
            ];
            launch_1d_async(
                self.rt,
                ptx.row_layernorm_bwd_dx_f32,
                rows,
                128,
                &mut params,
            )?;
        }
        if has_gamma || has_beta {
            let mut params: [*mut c_void; 10] = [
                (&mut d_dy as *mut CuDevicePtr).cast(),
                (&mut d_x as *mut CuDevicePtr).cast(),
                (&mut d_mean as *mut CuDevicePtr).cast(),
                (&mut d_rstd as *mut CuDevicePtr).cast(),
                (&mut d_dg as *mut CuDevicePtr).cast(),
                (&mut d_db as *mut CuDevicePtr).cast(),
                (&mut rows_u64 as *mut u64).cast(),
                (&mut cols_u64 as *mut u64).cast(),
                (&mut hg as *mut u32).cast(),
                (&mut hb as *mut u32).cast(),
            ];
            launch_1d_async(
                self.rt,
                ptx.row_layernorm_bwd_dgb_f32,
                cols,
                128,
                &mut params,
            )?;
        }
        Ok(())
    }

    /// Device-resident RMSNorm forward.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_rstd: CuDevicePtr,
        rows: usize,
        cols: usize,
        mut eps: f32,
    ) -> OchreResult<()> {
        let func = self.ptx()?.row_rmsnorm;
        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        let mut hg = 1u32;
        let mut params: [*mut c_void; 8] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_gamma as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_rstd as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
            (&mut hg as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        launch_1d_async(self.rt, func, rows, 128, &mut params)
    }

    /// Device-resident RMSNorm backward.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bwd_dev_f32(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_x: CuDevicePtr,
        mut d_gamma: CuDevicePtr,
        mut d_rstd: CuDevicePtr,
        mut d_dx: CuDevicePtr,
        mut d_dg: CuDevicePtr,
        rows: usize,
        cols: usize,
        need_dx: bool,
    ) -> OchreResult<()> {
        let ptx = self.ptx()?;
        let mut rows_u64 = rows as u64;
        let mut cols_u64 = cols as u64;
        if need_dx {
            let mut params: [*mut c_void; 7] = [
                (&mut d_dy as *mut CuDevicePtr).cast(),
                (&mut d_x as *mut CuDevicePtr).cast(),
                (&mut d_gamma as *mut CuDevicePtr).cast(),
                (&mut d_rstd as *mut CuDevicePtr).cast(),
                (&mut d_dx as *mut CuDevicePtr).cast(),
                (&mut rows_u64 as *mut u64).cast(),
                (&mut cols_u64 as *mut u64).cast(),
            ];
            launch_1d_async(self.rt, ptx.row_rmsnorm_bwd_dx_f32, rows, 128, &mut params)?;
        }
        let mut params: [*mut c_void; 6] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_rstd as *mut CuDevicePtr).cast(),
            (&mut d_dg as *mut CuDevicePtr).cast(),
            (&mut rows_u64 as *mut u64).cast(),
            (&mut cols_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, ptx.row_rmsnorm_bwd_dg_f32, cols, 128, &mut params)
    }

    /// Device-resident inverted Dropout forward.
    #[allow(clippy::too_many_arguments)]
    pub fn dropout_fwd_dev_f32(
        &self,
        mut d_x: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_mask: CuDevicePtr,
        n: usize,
        mut p: f32,
        mut seed: u64,
    ) -> OchreResult<()> {
        let func = self.ptx()?.dropout_fwd_f32;
        let mut n_u64 = n as u64;
        let mut scale = 1.0f32 / (1.0f32 - p);
        let mut params: [*mut c_void; 7] = [
            (&mut d_x as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_mask as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
            (&mut p as *mut f32).cast(),
            (&mut scale as *mut f32).cast(),
            (&mut seed as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }
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

    #[test]
    fn test_vram_resident_ptx_entries_and_sm50_compliance() {
        let required_entries = [
            "ochre_fill_f32",
            "ochre_copy_f32",
            "ochre_add_inplace_f32",
            "ochre_mul_f32",
            "ochre_f32_to_bf16",
            "ochre_gather_rows_f32",
            "ochre_gather_u64",
            "ochre_add_bias_row_f32",
            "ochre_sum_rows_f32",
            "ochre_act_fwd_f32",
            "ochre_act_bwd_f32",
            "ochre_mse_grad_f32",
            "ochre_reduce_sum_add_f32",
            "ochre_reduce_sum_sq_add_f32",
            "ochre_clip_scale_f32",
            "ochre_sgd_step_f32",
            "ochre_adam_step_f32",
            "ochre_nhwc_to_nchw_bias_f32",
            "ochre_nchw_to_nhwc_f32",
            "ochre_sum_nchw_channel_f32",
            "ochre_bn2d_fwd_train_f32",
            "ochre_bn2d_fwd_eval_f32",
            "ochre_bn2d_bwd_f32",
            "ochre_max_pool2d_bwd_f32",
            "ochre_global_avg_pool2d_fwd_f32",
            "ochre_global_avg_pool2d_bwd_f32",
            "ochre_row_layernorm_bwd_dx_f32",
            "ochre_row_layernorm_bwd_dgb_f32",
            "ochre_row_rmsnorm_bwd_dx_f32",
            "ochre_row_rmsnorm_bwd_dg_f32",
            "ochre_dropout_fwd_f32",
        ];
        for entry in required_entries {
            assert!(
                OCHRE_NN_PTX.contains(&format!(".visible .entry {entry}")),
                "missing PTX entry: {entry}"
            );
        }
        // Strict sm_50 compliance: no Turing/Ampere+ PTX ops
        assert!(!OCHRE_NN_PTX.contains("tanh.approx"));
        assert!(!OCHRE_NN_PTX.contains("mma.sync"));
        assert!(!OCHRE_NN_PTX.contains("ldmatrix"));

        // Balanced braces across all kernels
        let opens = OCHRE_NN_PTX.matches('{').count();
        let closes = OCHRE_NN_PTX.matches('}').count();
        let entries = OCHRE_NN_PTX.matches(".visible .entry").count();
        assert_eq!(opens, closes);
        assert_eq!(opens, entries);

        // Session either initializes cleanly on CUDA hosts or returns None on CPU-only hosts
        if let Some(res) = with_vram_session(0, |sess| {
            let buf = sess.upload_f32(&[1.0, 2.0, 3.0, 4.0])?;
            let mut out = vec![0.0f32; 4];
            sess.dtoh_f32(&mut out, &buf)?;
            Ok(out)
        }) {
            assert_eq!(res.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        }
    }
}
