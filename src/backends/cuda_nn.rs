// OchreML CUDA Neural Network Backend (cuBLAS + sm_50 Runtime PTX) (*≧ω≦*)
// Dynamic libcublas loading (cublasSgemm_v2, cublasSgemmStridedBatched, cublasGemmEx for BF16/F16)
// plus universal sm_50 runtime PTX kernels for N8/N9/N10 and persistent VRAM workspace cache.

use crate::backends::cuda::{
    CUDA_SUCCESS, CuContext, CuDevicePtr, CuResult, CudaDriver, get_driver,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::dtype::{DType, TensorStorage};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, Mutex, OnceLock};

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

const VRAM_POOL_MIN_BYTES: usize = 256;
const VRAM_POOL_POW2_MAX_BYTES: usize = 16 * 1024 * 1024;
const VRAM_POOL_LARGE_ALIGN_BYTES: usize = 1024 * 1024;
const VRAM_POOL_MAX_CACHED_BYTES: usize = 512 * 1024 * 1024;
const VRAM_POOL_MAX_BUFS_PER_BUCKET: usize = 32;

/// Rounds a VRAM allocation size up to its slab pool bucket size:
/// clamped to at least 256 B, next power of two up to 16 MiB, and 1 MiB-aligned above 16 MiB.
#[inline]
pub(crate) fn pool_bucket_size(byte_len: usize) -> usize {
    let clamped = byte_len.max(VRAM_POOL_MIN_BYTES);
    if clamped <= VRAM_POOL_POW2_MAX_BYTES {
        clamped.next_power_of_two()
    } else {
        let align = VRAM_POOL_LARGE_ALIGN_BYTES;
        clamped.saturating_add(align - 1) & !(align - 1)
    }
}

struct VramPool {
    ctx: CuContext,
    cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    buckets: HashMap<usize, Vec<CuDevicePtr>>,
    cached_bytes: usize,
    max_cached_bytes: usize,
}

// SAFETY: Serialized via Arc<Mutex<VramPool>> and binds `self.ctx` before releasing device pointers.
unsafe impl Send for VramPool {}
// SAFETY: Guarded by Mutex.
unsafe impl Sync for VramPool {}

impl VramPool {
    fn new(driver: &CudaDriver, ctx: CuContext) -> Self {
        Self {
            ctx,
            cu_ctx_set_current: driver.cu_ctx_set_current,
            cu_mem_free: driver.cu_mem_free,
            buckets: HashMap::new(),
            cached_bytes: 0,
            max_cached_bytes: VRAM_POOL_MAX_CACHED_BYTES,
        }
    }

    fn pop(&mut self, bucket_len: usize) -> Option<CuDevicePtr> {
        let ptr = self.buckets.get_mut(&bucket_len)?.pop()?;
        self.cached_bytes = self.cached_bytes.saturating_sub(bucket_len);
        Some(ptr)
    }

    fn recycle(&mut self, bucket_len: usize, ptr: CuDevicePtr) {
        if ptr == 0 {
            return;
        }
        let under_bucket_cap =
            self.buckets.get(&bucket_len).map_or(0, Vec::len) < VRAM_POOL_MAX_BUFS_PER_BUCKET;
        if under_bucket_cap && self.cached_bytes.saturating_add(bucket_len) <= self.max_cached_bytes
        {
            self.buckets.entry(bucket_len).or_default().push(ptr);
            self.cached_bytes += bucket_len;
        } else {
            // SAFETY: Bind owning CUDA context before freeing evicted device buffer.
            unsafe {
                let _ = (self.cu_ctx_set_current)(self.ctx);
                let _ = (self.cu_mem_free)(ptr);
            }
        }
    }

    fn drain_all(&mut self) {
        if self.buckets.is_empty() {
            return;
        }
        // SAFETY: Bind owning CUDA context once before releasing all remaining pooled buffers.
        unsafe {
            let _ = (self.cu_ctx_set_current)(self.ctx);
            for (_, ptrs) in self.buckets.drain() {
                for ptr in ptrs {
                    if ptr != 0 {
                        let _ = (self.cu_mem_free)(ptr);
                    }
                }
            }
        }
        self.cached_bytes = 0;
    }
}

impl Drop for VramPool {
    fn drop(&mut self) {
        self.drain_all();
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
    // Wave 1: RNN/GRU/LSTM fused step kernels & Transformer VRAM kernels
    rnn_step_fwd_f32: CuFunction,
    rnn_step_bwd_f32: CuFunction,
    gru_step_fwd_f32: CuFunction,
    gru_step_bwd_f32: CuFunction,
    lstm_step_fwd_f32: CuFunction,
    lstm_step_bwd_f32: CuFunction,
    rope_fwd_bwd_f32: CuFunction,
    swiglu_fwd_f32: CuFunction,
    swiglu_bwd_f32: CuFunction,
    flash_sdpa_fwd_f32: CuFunction,
    flash_sdpa_bwd_dq_f32: CuFunction,
    flash_sdpa_bwd_dkv_f32: CuFunction,
    sdpa_causal_softmax_f32: CuFunction,
    sdpa_bwd_ds_f32: CuFunction,
    embedding_scatter_add_f32: CuFunction,
    bshd_to_bhsd_f32: CuFunction,
    bhsd_to_bshd_f32: CuFunction,
}

struct CudaNnCtxState {
    handle: CublasHandle,
    cublas_destroy: FnCublasDestroy,
    workspace: CudaNnWorkspace,
    vram_pool: Arc<Mutex<VramPool>>,
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

.visible .entry ochre_rnn_step_fwd_f32(
    .param .u64 p_x_proj,
    .param .u64 p_h_proj,
    .param .u64 p_bias,
    .param .u64 p_h_cur,
    .param .u64 p_layer_out,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off,
    .param .u32 p_nonlin,
    .param .u32 p_has_bias
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // idx = b * h + j
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;   // bh
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra RSF_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // b * seq_len + t

    // x_proj[(b * seq_len + t) * h + j]
    mul.lo.u64 %rd10, %rd9, %rd3;
    add.u64 %rd10, %rd10, %rd6;
    shl.b64 %rd10, %rd10, 2;
    ld.param.u64 %rd11, [p_x_proj];
    add.u64 %rd11, %rd11, %rd10;
    ld.global.f32 %f1, [%rd11];

    // h_proj[idx]
    shl.b64 %rd12, %rd1, 2;
    ld.param.u64 %rd11, [p_h_proj];
    add.u64 %rd11, %rd11, %rd12;
    ld.global.f32 %f2, [%rd11];
    add.f32 %f1, %f1, %f2;

    ld.param.u32 %r4, [p_has_bias];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra RSF_ACT;
    ld.param.u64 %rd11, [p_bias];
    shl.b64 %rd13, %rd6, 2;
    add.u64 %rd11, %rd11, %rd13;
    ld.global.f32 %f3, [%rd11];
    add.f32 %f1, %f1, %f3;

RSF_ACT:
    ld.param.u32 %r5, [p_nonlin];
    setp.ne.u32 %p3, %r5, 0;
    @%p3 bra RSF_RELU;

    // Tanh via clamped ex2(2u * log2(e))
    add.f32 %f4, %f1, %f1;
    mov.f32 %f5, 0fC2200000;       // -40.0
    mov.f32 %f6, 0f42200000;       // +40.0
    max.f32 %f4, %f4, %f5;
    min.f32 %f4, %f4, %f6;
    mov.f32 %f7, 0f3FB8AA3B;       // log2(e)
    mul.f32 %f4, %f4, %f7;
    ex2.approx.f32 %f8, %f4;
    mov.f32 %f9, 0f3F800000;       // 1.0
    sub.f32 %f10, %f8, %f9;
    add.f32 %f11, %f8, %f9;
    div.rn.f32 %f1, %f10, %f11;
    bra RSF_STORE;

RSF_RELU:
    mov.f32 %f4, 0f00000000;
    max.f32 %f1, %f1, %f4;

RSF_STORE:
    ld.param.u64 %rd11, [p_h_cur];
    add.u64 %rd11, %rd11, %rd12;
    st.global.f32 [%rd11], %f1;

    ld.param.u64 %rd14, [p_out_dim];
    ld.param.u64 %rd15, [p_dir_off];
    mul.lo.u64 %rd10, %rd9, %rd14;
    add.u64 %rd10, %rd10, %rd15;
    add.u64 %rd10, %rd10, %rd6;
    shl.b64 %rd10, %rd10, 2;
    ld.param.u64 %rd11, [p_layer_out];
    add.u64 %rd11, %rd11, %rd10;
    st.global.f32 [%rd11], %f1;
RSF_DONE:
    ret;
}

.visible .entry ochre_rnn_step_bwd_f32(
    .param .u64 p_d_out,
    .param .u64 p_dh_next,
    .param .u64 p_h_cur,
    .param .u64 p_da_step,
    .param .u64 p_da_all,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off,
    .param .u32 p_nonlin
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<7>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra RSB_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // b * seq_len + t

    ld.param.u64 %rd10, [p_out_dim];
    ld.param.u64 %rd11, [p_dir_off];
    mul.lo.u64 %rd12, %rd9, %rd10;
    add.u64 %rd12, %rd12, %rd11;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd13, [p_d_out];
    add.u64 %rd13, %rd13, %rd12;
    ld.global.f32 %f1, [%rd13];    // d_out

    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd13, [p_dh_next];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f2, [%rd13];    // dh_next
    add.f32 %f1, %f1, %f2;         // dh

    ld.param.u64 %rd13, [p_h_cur];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f3, [%rd13];    // y = h_cur

    ld.param.u32 %r4, [p_nonlin];
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra RSB_RELU;

    // Tanh derivative: dh * (1 - y*y)
    mov.f32 %f4, 0f3F800000;
    mul.f32 %f5, %f3, %f3;
    sub.f32 %f5, %f4, %f5;
    mul.f32 %f6, %f1, %f5;
    bra RSB_STORE;

RSB_RELU:
    mov.f32 %f4, 0f00000000;
    setp.gt.f32 %p3, %f3, %f4;
    selp.f32 %f6, %f1, %f4, %p3;

RSB_STORE:
    ld.param.u64 %rd13, [p_da_step];
    add.u64 %rd13, %rd13, %rd14;
    st.global.f32 [%rd13], %f6;

    mul.lo.u64 %rd15, %rd9, %rd3;
    add.u64 %rd15, %rd15, %rd6;
    shl.b64 %rd15, %rd15, 2;
    ld.param.u64 %rd13, [p_da_all];
    add.u64 %rd13, %rd13, %rd15;
    st.global.f32 [%rd13], %f6;
RSB_DONE:
    ret;
}

.visible .entry ochre_gru_step_fwd_f32(
    .param .u64 p_gi_all,
    .param .u64 p_gh,
    .param .u64 p_biases,
    .param .u64 p_h_prev,
    .param .u64 p_h_cur,
    .param .u64 p_layer_out,
    .param .u64 p_r_out,
    .param .u64 p_z_out,
    .param .u64 p_n_out,
    .param .u64 p_ghn_out,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off,
    .param .u32 p_save_cache
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<24>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // idx = b * h + j
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra GSF_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // bt = b * seq_len + t

    mul.lo.u64 %rd10, %rd3, 3;     // h3 = 3 * h
    mul.lo.u64 %rd11, %rd9, %rd10;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;       // gi byte offset for gate 0
    mul.lo.u64 %rd12, %rd5, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;       // gh byte offset for gate 0
    shl.b64 %rd13, %rd3, 2;        // h_bytes = h * 4

    ld.param.u64 %rd14, [p_gi_all];
    add.u64 %rd14, %rd14, %rd11;
    ld.param.u64 %rd15, [p_gh];
    add.u64 %rd15, %rd15, %rd12;
    ld.param.u64 %rd16, [p_biases];
    shl.b64 %rd17, %rd6, 2;
    add.u64 %rd16, %rd16, %rd17;   // &biases[j]

    // Constants
    mov.f32 %f16, 0fC2A00000;      // -80.0
    mov.f32 %f17, 0f42A00000;      // +80.0
    mov.f32 %f18, 0f3FB8AA3B;      // log2(e)
    mov.f32 %f19, 0f3F800000;      // 1.0

    // Gate r = sigmoid(gi[0] + gh[0] + br_sum[j])
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    neg.f32 %f4, %f1;
    max.f32 %f4, %f4, %f16;
    min.f32 %f4, %f4, %f17;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f19, %f4;
    div.rn.f32 %f20, %f19, %f4;    // r

    // Gate z = sigmoid(gi[h] + gh[h] + bz_sum[j])
    add.u64 %rd14, %rd14, %rd13;
    add.u64 %rd15, %rd15, %rd13;
    add.u64 %rd16, %rd16, %rd13;
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    neg.f32 %f4, %f1;
    max.f32 %f4, %f4, %f16;
    min.f32 %f4, %f4, %f17;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f19, %f4;
    div.rn.f32 %f21, %f19, %f4;    // z

    // Gate n: ghn = gh[2h] + bh_n[j], n = tanh(gi[2h] + bi_n[j] + r * ghn)
    add.u64 %rd14, %rd14, %rd13;
    add.u64 %rd15, %rd15, %rd13;
    add.u64 %rd16, %rd16, %rd13;   // bi_n[j]
    ld.global.f32 %f1, [%rd14];    // gi[2h]
    ld.global.f32 %f2, [%rd15];    // gh[2h]
    ld.global.f32 %f3, [%rd16];    // bi_n
    add.u64 %rd16, %rd16, %rd13;   // bh_n[j]
    ld.global.f32 %f5, [%rd16];    // bh_n
    add.f32 %f22, %f2, %f5;        // ghn
    add.f32 %f1, %f1, %f3;
    fma.rn.f32 %f1, %f20, %f22, %f1; // pre_n
    add.f32 %f6, %f1, %f1;
    mov.f32 %f7, 0fC2200000;       // -40.0
    mov.f32 %f8, 0f42200000;       // +40.0
    max.f32 %f6, %f6, %f7;
    min.f32 %f6, %f6, %f8;
    mul.f32 %f6, %f6, %f18;
    ex2.approx.f32 %f6, %f6;
    sub.f32 %f9, %f6, %f19;
    add.f32 %f10, %f6, %f19;
    div.rn.f32 %f23, %f9, %f10;    // n

    // h_cur = (1 - z) * n + z * h_prev
    shl.b64 %rd17, %rd1, 2;
    ld.param.u64 %rd14, [p_h_prev];
    add.u64 %rd14, %rd14, %rd17;
    ld.global.f32 %f11, [%rd14];   // h_prev
    sub.f32 %f12, %f19, %f21;      // 1 - z
    mul.f32 %f13, %f12, %f23;
    fma.rn.f32 %f13, %f21, %f11, %f13; // h_cur

    ld.param.u64 %rd14, [p_h_cur];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f13;

    ld.param.u64 %rd14, [p_out_dim];
    ld.param.u64 %rd15, [p_dir_off];
    mul.lo.u64 %rd11, %rd9, %rd14;
    add.u64 %rd11, %rd11, %rd15;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;
    ld.param.u64 %rd16, [p_layer_out];
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f13;

    ld.param.u32 %r4, [p_save_cache];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra GSF_DONE;

    ld.param.u64 %rd14, [p_r_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f20;
    ld.param.u64 %rd14, [p_z_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f21;
    ld.param.u64 %rd14, [p_n_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f23;
    ld.param.u64 %rd14, [p_ghn_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f22;
GSF_DONE:
    ret;
}

.visible .entry ochre_gru_step_bwd_f32(
    .param .u64 p_d_out,
    .param .u64 p_dh_next,
    .param .u64 p_h_prev,
    .param .u64 p_r,
    .param .u64 p_z,
    .param .u64 p_n,
    .param .u64 p_ghn,
    .param .u64 p_dgi_all,
    .param .u64 p_dgh_step,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<19>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra GSB_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // bt

    ld.param.u64 %rd10, [p_out_dim];
    ld.param.u64 %rd11, [p_dir_off];
    mul.lo.u64 %rd12, %rd9, %rd10;
    add.u64 %rd12, %rd12, %rd11;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd13, [p_d_out];
    add.u64 %rd13, %rd13, %rd12;
    ld.global.f32 %f1, [%rd13];    // d_out

    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd15, [p_dh_next];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f2, [%rd15];    // dh_next
    add.f32 %f1, %f1, %f2;         // dh

    ld.param.u64 %rd13, [p_h_prev];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f3, [%rd13];    // hp
    ld.param.u64 %rd13, [p_r];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f4, [%rd13];    // r
    ld.param.u64 %rd13, [p_z];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f5, [%rd13];    // z
    ld.param.u64 %rd13, [p_n];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f6, [%rd13];    // n
    ld.param.u64 %rd13, [p_ghn];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f7, [%rd13];    // ghn

    mov.f32 %f8, 0f3F800000;       // 1.0
    sub.f32 %f9, %f8, %f5;         // 1 - z
    mul.f32 %f10, %f1, %f9;        // dn = dh * (1 - z)
    sub.f32 %f11, %f3, %f6;        // hp - n
    mul.f32 %f11, %f1, %f11;       // dz = dh * (hp - n)
    mul.f32 %f12, %f6, %f6;
    sub.f32 %f12, %f8, %f12;
    mul.f32 %f13, %f10, %f12;      // da_n = dn * (1 - n^2)
    mul.f32 %f14, %f13, %f7;       // dr = da_n * ghn
    mul.f32 %f15, %f11, %f5;
    mul.f32 %f15, %f15, %f9;       // da_z = dz * z * (1 - z)
    sub.f32 %f16, %f8, %f4;        // 1 - r
    mul.f32 %f16, %f4, %f16;
    mul.f32 %f16, %f14, %f16;      // da_r = dr * r * (1 - r)
    mul.f32 %f17, %f13, %f4;       // da_n * r
    mul.f32 %f18, %f1, %f5;        // dh * z

    // Update dh_next in-place with dh * z
    st.global.f32 [%rd15], %f18;

    // Store dgi_all[(bt)*3h + (0,h,2h) + j] and dgh_step[b*3h + (0,h,2h) + j]
    mul.lo.u64 %rd10, %rd3, 3;     // 3h
    shl.b64 %rd11, %rd3, 2;        // h * 4

    mul.lo.u64 %rd12, %rd9, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd16, [p_dgi_all];
    add.u64 %rd16, %rd16, %rd12;
    st.global.f32 [%rd16], %f16;
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f15;
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f13;

    mul.lo.u64 %rd12, %rd5, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd17, [p_dgh_step];
    add.u64 %rd17, %rd17, %rd12;
    st.global.f32 [%rd17], %f16;
    add.u64 %rd17, %rd17, %rd11;
    st.global.f32 [%rd17], %f15;
    add.u64 %rd17, %rd17, %rd11;
    st.global.f32 [%rd17], %f17;
GSB_DONE:
    ret;
}

.visible .entry ochre_lstm_step_fwd_f32(
    .param .u64 p_gi_all,
    .param .u64 p_gh,
    .param .u64 p_b_total,
    .param .u64 p_c_prev,
    .param .u64 p_h_cur,
    .param .u64 p_c_cur,
    .param .u64 p_layer_out,
    .param .u64 p_i_out,
    .param .u64 p_f_out,
    .param .u64 p_g_out,
    .param .u64 p_o_out,
    .param .u64 p_tc_out,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off,
    .param .u32 p_save_cache
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<26>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra LSF_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // bt

    shl.b64 %rd10, %rd3, 2;        // h4 = 4 * h (elements)
    mul.lo.u64 %rd11, %rd9, %rd10;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;
    mul.lo.u64 %rd12, %rd5, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    shl.b64 %rd13, %rd3, 2;        // h_bytes = h * 4

    ld.param.u64 %rd14, [p_gi_all];
    add.u64 %rd14, %rd14, %rd11;
    ld.param.u64 %rd15, [p_gh];
    add.u64 %rd15, %rd15, %rd12;
    ld.param.u64 %rd16, [p_b_total];
    shl.b64 %rd17, %rd6, 2;
    add.u64 %rd16, %rd16, %rd17;

    mov.f32 %f14, 0fC2A00000;      // -80.0
    mov.f32 %f15, 0f42A00000;      // +80.0
    mov.f32 %f16, 0fC2200000;      // -40.0
    mov.f32 %f17, 0f42200000;      // +40.0
    mov.f32 %f18, 0f3FB8AA3B;      // log2(e)
    mov.f32 %f19, 0f3F800000;      // 1.0

    // i = sigmoid(gi[0] + gh[0] + b[0])
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    neg.f32 %f4, %f1;
    max.f32 %f4, %f4, %f14;
    min.f32 %f4, %f4, %f15;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f19, %f4;
    div.rn.f32 %f20, %f19, %f4;    // iv

    // f = sigmoid(gi[h] + gh[h] + b[h])
    add.u64 %rd14, %rd14, %rd13;
    add.u64 %rd15, %rd15, %rd13;
    add.u64 %rd16, %rd16, %rd13;
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    neg.f32 %f4, %f1;
    max.f32 %f4, %f4, %f14;
    min.f32 %f4, %f4, %f15;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f19, %f4;
    div.rn.f32 %f21, %f19, %f4;    // fv

    // g = tanh(gi[2h] + gh[2h] + b[2h])
    add.u64 %rd14, %rd14, %rd13;
    add.u64 %rd15, %rd15, %rd13;
    add.u64 %rd16, %rd16, %rd13;
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    add.f32 %f4, %f1, %f1;
    max.f32 %f4, %f4, %f16;
    min.f32 %f4, %f4, %f17;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    sub.f32 %f5, %f4, %f19;
    add.f32 %f6, %f4, %f19;
    div.rn.f32 %f22, %f5, %f6;     // gv

    // o = sigmoid(gi[3h] + gh[3h] + b[3h])
    add.u64 %rd14, %rd14, %rd13;
    add.u64 %rd15, %rd15, %rd13;
    add.u64 %rd16, %rd16, %rd13;
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    ld.global.f32 %f3, [%rd16];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    neg.f32 %f4, %f1;
    max.f32 %f4, %f4, %f14;
    min.f32 %f4, %f4, %f15;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f19, %f4;
    div.rn.f32 %f23, %f19, %f4;    // ov

    // c_new = fv * c_prev + iv * gv, tc = tanh(c_new), h_new = ov * tc
    shl.b64 %rd17, %rd1, 2;
    ld.param.u64 %rd14, [p_c_prev];
    add.u64 %rd14, %rd14, %rd17;
    ld.global.f32 %f7, [%rd14];    // c_prev
    mul.f32 %f8, %f21, %f7;
    fma.rn.f32 %f8, %f20, %f22, %f8; // c_new

    add.f32 %f4, %f8, %f8;
    max.f32 %f4, %f4, %f16;
    min.f32 %f4, %f4, %f17;
    mul.f32 %f4, %f4, %f18;
    ex2.approx.f32 %f4, %f4;
    sub.f32 %f5, %f4, %f19;
    add.f32 %f6, %f4, %f19;
    div.rn.f32 %f24, %f5, %f6;     // tc = tanh(c_new)
    mul.f32 %f25, %f23, %f24;      // h_new = ov * tc

    ld.param.u64 %rd14, [p_c_cur];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f8;

    ld.param.u64 %rd14, [p_h_cur];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f25;

    ld.param.u64 %rd14, [p_out_dim];
    ld.param.u64 %rd15, [p_dir_off];
    mul.lo.u64 %rd11, %rd9, %rd14;
    add.u64 %rd11, %rd11, %rd15;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;
    ld.param.u64 %rd16, [p_layer_out];
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f25;

    ld.param.u32 %r4, [p_save_cache];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra LSF_DONE;

    ld.param.u64 %rd14, [p_i_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f20;
    ld.param.u64 %rd14, [p_f_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f21;
    ld.param.u64 %rd14, [p_g_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f22;
    ld.param.u64 %rd14, [p_o_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f23;
    ld.param.u64 %rd14, [p_tc_out];
    add.u64 %rd14, %rd14, %rd17;
    st.global.f32 [%rd14], %f24;
LSF_DONE:
    ret;
}

.visible .entry ochre_lstm_step_bwd_f32(
    .param .u64 p_d_out,
    .param .u64 p_dh_next,
    .param .u64 p_dc_next,
    .param .u64 p_c_prev,
    .param .u64 p_i,
    .param .u64 p_f,
    .param .u64 p_g,
    .param .u64 p_o,
    .param .u64 p_tc,
    .param .u64 p_dg_all,
    .param .u64 p_dg_step,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_h,
    .param .u64 p_out_dim,
    .param .u64 p_t,
    .param .u64 p_dir_off
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<20>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_h];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra LSB_DONE;

    div.u64 %rd5, %rd1, %rd3;      // b
    rem.u64 %rd6, %rd1, %rd3;      // j
    ld.param.u64 %rd7, [p_seq_len];
    ld.param.u64 %rd8, [p_t];
    mul.lo.u64 %rd9, %rd5, %rd7;
    add.u64 %rd9, %rd9, %rd8;      // bt

    ld.param.u64 %rd10, [p_out_dim];
    ld.param.u64 %rd11, [p_dir_off];
    mul.lo.u64 %rd12, %rd9, %rd10;
    add.u64 %rd12, %rd12, %rd11;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd13, [p_d_out];
    add.u64 %rd13, %rd13, %rd12;
    ld.global.f32 %f1, [%rd13];    // d_out

    shl.b64 %rd14, %rd1, 2;
    ld.param.u64 %rd13, [p_dh_next];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f2, [%rd13];    // dh_next
    add.f32 %f1, %f1, %f2;         // dh

    ld.param.u64 %rd15, [p_dc_next];
    add.u64 %rd15, %rd15, %rd14;
    ld.global.f32 %f3, [%rd15];    // dc_next

    ld.param.u64 %rd13, [p_c_prev];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f4, [%rd13];    // cp
    ld.param.u64 %rd13, [p_i];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f5, [%rd13];    // iv
    ld.param.u64 %rd13, [p_f];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f6, [%rd13];    // fv
    ld.param.u64 %rd13, [p_g];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f7, [%rd13];    // gv
    ld.param.u64 %rd13, [p_o];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f8, [%rd13];    // ov
    ld.param.u64 %rd13, [p_tc];
    add.u64 %rd13, %rd13, %rd14;
    ld.global.f32 %f9, [%rd13];    // tc

    mov.f32 %f10, 0f3F800000;      // 1.0
    mul.f32 %f11, %f1, %f9;        // do_v = dh * tc
    mul.f32 %f12, %f9, %f9;
    sub.f32 %f12, %f10, %f12;      // 1 - tc^2
    mul.f32 %f12, %f8, %f12;       // ov * (1 - tc^2)
    fma.rn.f32 %f13, %f1, %f12, %f3; // dc = dh * ov * (1 - tc^2) + dc_next

    // da_i = dc * gv * iv * (1 - iv)
    sub.f32 %f14, %f10, %f5;
    mul.f32 %f14, %f5, %f14;
    mul.f32 %f14, %f7, %f14;
    mul.f32 %f14, %f13, %f14;

    // da_f = dc * cp * fv * (1 - fv)
    sub.f32 %f15, %f10, %f6;
    mul.f32 %f15, %f6, %f15;
    mul.f32 %f15, %f4, %f15;
    mul.f32 %f15, %f13, %f15;

    // da_g = dc * iv * (1 - gv^2)
    mul.f32 %f16, %f7, %f7;
    sub.f32 %f16, %f10, %f16;
    mul.f32 %f16, %f5, %f16;
    mul.f32 %f16, %f13, %f16;

    // da_o = do_v * ov * (1 - ov)
    sub.f32 %f17, %f10, %f8;
    mul.f32 %f17, %f8, %f17;
    mul.f32 %f17, %f11, %f17;

    // dc_next = dc * fv
    mul.f32 %f18, %f13, %f6;
    st.global.f32 [%rd15], %f18;

    shl.b64 %rd10, %rd3, 2;        // 4h elements
    shl.b64 %rd11, %rd3, 2;        // h * 4 bytes

    mul.lo.u64 %rd12, %rd9, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd16, [p_dg_all];
    add.u64 %rd16, %rd16, %rd12;
    st.global.f32 [%rd16], %f14;
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f15;
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f16;
    add.u64 %rd16, %rd16, %rd11;
    st.global.f32 [%rd16], %f17;

    mul.lo.u64 %rd12, %rd5, %rd10;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd17, [p_dg_step];
    add.u64 %rd17, %rd17, %rd12;
    st.global.f32 [%rd17], %f14;
    add.u64 %rd17, %rd17, %rd11;
    st.global.f32 [%rd17], %f15;
    add.u64 %rd17, %rd17, %rd11;
    st.global.f32 [%rd17], %f16;
    add.u64 %rd17, %rd17, %rd11;
    st.global.f32 [%rd17], %f17;
LSB_DONE:
    ret;
}

.visible .entry ochre_rope_fwd_bwd_f32(
    .param .u64 p_data,
    .param .u64 p_cos,
    .param .u64 p_sin,
    .param .u64 p_total_pairs,
    .param .u64 p_seq_len,
    .param .u64 p_half_dim,
    .param .u64 p_offset,
    .param .u32 p_inverse
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<15>;
    .reg .f32 %f<8>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // pair idx
    ld.param.u64 %rd2, [p_total_pairs];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra ROPE_DONE;

    ld.param.u64 %rd3, [p_half_dim];
    ld.param.u64 %rd4, [p_seq_len];
    ld.param.u64 %rd5, [p_offset];

    rem.u64 %rd6, %rd1, %rd3;      // d in 0..half
    div.u64 %rd7, %rd1, %rd3;      // token_idx
    rem.u64 %rd8, %rd7, %rd4;      // s in 0..seq_len
    add.u64 %rd8, %rd8, %rd5;      // pos = offset + s

    // cos/sin offset: (pos * half + d) * 4
    mul.lo.u64 %rd9, %rd8, %rd3;
    add.u64 %rd9, %rd9, %rd6;
    shl.b64 %rd9, %rd9, 2;
    ld.param.u64 %rd10, [p_cos];
    add.u64 %rd10, %rd10, %rd9;
    ld.global.f32 %f3, [%rd10];    // c
    ld.param.u64 %rd10, [p_sin];
    add.u64 %rd10, %rd10, %rd9;
    ld.global.f32 %f4, [%rd10];    // sn

    // data offsets: base0 = (token_idx * 2 * half + d) * 4, base1 = base0 + half * 4
    shl.b64 %rd11, %rd3, 1;        // head_dim = 2 * half
    mul.lo.u64 %rd12, %rd7, %rd11;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    ld.param.u64 %rd13, [p_data];
    add.u64 %rd13, %rd13, %rd12;   // &data[base + d]
    shl.b64 %rd14, %rd3, 2;        // half * 4
    add.u64 %rd14, %rd13, %rd14;   // &data[base + half + d]

    ld.global.f32 %f1, [%rd13];    // x0
    ld.global.f32 %f2, [%rd14];    // x1

    ld.param.u32 %r4, [p_inverse];
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra ROPE_INV;

    // Forward: y0 = x0*c - x1*sn, y1 = x0*sn + x1*c
    mul.f32 %f5, %f1, %f3;
    mul.f32 %f6, %f2, %f4;
    sub.f32 %f5, %f5, %f6;
    mul.f32 %f7, %f1, %f4;
    fma.rn.f32 %f7, %f2, %f3, %f7;
    bra ROPE_STORE;

ROPE_INV:
    // Inverse: y0 = x0*c + x1*sn, y1 = -x0*sn + x1*c
    mul.f32 %f5, %f1, %f3;
    fma.rn.f32 %f5, %f2, %f4, %f5;
    mul.f32 %f6, %f2, %f3;
    mul.f32 %f7, %f1, %f4;
    sub.f32 %f7, %f6, %f7;

ROPE_STORE:
    st.global.f32 [%rd13], %f5;
    st.global.f32 [%rd14], %f7;
ROPE_DONE:
    ret;
}

.visible .entry ochre_swiglu_fwd_f32(
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_out,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<11>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SWF_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_gate];
    ld.param.u64 %rd5, [p_up];
    ld.param.u64 %rd6, [p_out];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    ld.global.f32 %f1, [%rd4];     // g
    ld.global.f32 %f2, [%rd5];     // u

    neg.f32 %f3, %f1;
    mov.f32 %f4, 0fC2A00000;       // -80.0
    mov.f32 %f5, 0f42A00000;       // +80.0
    max.f32 %f3, %f3, %f4;
    min.f32 %f3, %f3, %f5;
    mov.f32 %f6, 0f3FB8AA3B;       // log2(e)
    mul.f32 %f3, %f3, %f6;
    ex2.approx.f32 %f7, %f3;
    mov.f32 %f8, 0f3F800000;
    add.f32 %f9, %f8, %f7;
    div.rn.f32 %f10, %f1, %f9;     // silu(g)
    mul.f32 %f10, %f10, %f2;       // silu(g) * u
    st.global.f32 [%rd6], %f10;
SWF_DONE:
    ret;
}

.visible .entry ochre_swiglu_bwd_f32(
    .param .u64 p_dy,
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_dgate,
    .param .u64 p_dup,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<9>;
    .reg .f32 %f<16>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_n];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SWB_DONE;

    shl.b64 %rd3, %rd1, 2;
    ld.param.u64 %rd4, [p_dy];
    ld.param.u64 %rd5, [p_gate];
    ld.param.u64 %rd6, [p_up];
    ld.param.u64 %rd7, [p_dgate];
    ld.param.u64 %rd8, [p_dup];
    add.u64 %rd4, %rd4, %rd3;
    add.u64 %rd5, %rd5, %rd3;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd7, %rd3;
    add.u64 %rd8, %rd8, %rd3;

    ld.global.f32 %f1, [%rd4];     // dy
    ld.global.f32 %f2, [%rd5];     // g
    ld.global.f32 %f3, [%rd6];     // u

    neg.f32 %f4, %f2;
    mov.f32 %f5, 0fC2A00000;
    mov.f32 %f6, 0f42A00000;
    max.f32 %f4, %f4, %f5;
    min.f32 %f4, %f4, %f6;
    mov.f32 %f7, 0f3FB8AA3B;
    mul.f32 %f4, %f4, %f7;
    ex2.approx.f32 %f8, %f4;
    mov.f32 %f9, 0f3F800000;       // 1.0
    add.f32 %f10, %f9, %f8;
    div.rn.f32 %f11, %f9, %f10;    // sig

    // dup = dy * g * sig
    mul.f32 %f12, %f2, %f11;       // silu(g)
    mul.f32 %f13, %f1, %f12;       // dup
    st.global.f32 [%rd8], %f13;

    // dgate = dy * u * sig * (1 + g * (1 - sig))
    sub.f32 %f14, %f9, %f11;       // 1 - sig
    fma.rn.f32 %f14, %f2, %f14, %f9; // 1 + g * (1 - sig)
    mul.f32 %f15, %f11, %f14;      // dsilu
    mul.f32 %f15, %f15, %f3;
    mul.f32 %f15, %f15, %f1;
    st.global.f32 [%rd7], %f15;
SWB_DONE:
    ret;
}

.visible .entry ochre_flash_sdpa_fwd_f32(
    .param .u64 p_q,
    .param .u64 p_k,
    .param .u64 p_v,
    .param .u64 p_out,
    .param .u64 p_lse,
    .param .u64 p_total_q_rows,
    .param .u64 p_num_heads,
    .param .u64 p_num_kv_heads,
    .param .u64 p_q_len,
    .param .u64 p_kv_len,
    .param .u64 p_head_dim,
    .param .u32 p_causal,
    .param .u64 p_kv_offset,
    .param .f32 p_scale
)
{
    .shared .f32 smem_fsf_s[64];
    .shared .f32 smem_fsf_p[64];
    .shared .f32 smem_fsf_r[64];
    .reg .pred %p<6>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<26>;
    .reg .f32 %f<18>;

    mov.u32 %r4, %ctaid.x;         // row_idx = bh * q_len + qi
    mov.u32 %r1, %tid.x;           // tid in 0..63
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_total_q_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra FSF_DONE;

    ld.param.u64 %rd3, [p_q_len];
    ld.param.u64 %rd4, [p_kv_len];
    ld.param.u64 %rd5, [p_head_dim];
    ld.param.u64 %rd6, [p_num_heads];
    ld.param.u64 %rd7, [p_num_kv_heads];

    div.u64 %rd8, %rd1, %rd3;      // bh
    rem.u64 %rd9, %rd1, %rd3;      // qi
    div.u64 %rd10, %rd8, %rd6;     // b
    rem.u64 %rd11, %rd8, %rd6;     // h
    div.u64 %rd12, %rd6, %rd7;     // group_size
    div.u64 %rd11, %rd11, %rd12;   // kv_h
    mul.lo.u64 %rd10, %rd10, %rd7;
    add.u64 %rd10, %rd10, %rd11;   // kv_bh = b * num_kv_heads + kv_h

    // valid_kv = causal ? min(kv_offset + qi + 1, kv_len) : kv_len
    mov.u64 %rd12, %rd4;
    ld.param.u32 %r5, [p_causal];
    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra FSF_VKV_OK;
    ld.param.u64 %rd13, [p_kv_offset];
    add.u64 %rd13, %rd13, %rd9;
    add.u64 %rd13, %rd13, 1;
    min.u64 %rd12, %rd12, %rd13;
FSF_VKV_OK:

    // Base pointers: q_row, out_row, k_head, v_head
    mul.lo.u64 %rd14, %rd1, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd15, [p_q];
    add.u64 %rd15, %rd15, %rd14;   // q_row
    ld.param.u64 %rd16, [p_out];
    add.u64 %rd16, %rd16, %rd14;   // out_row

    mul.lo.u64 %rd14, %rd10, %rd4;
    mul.lo.u64 %rd14, %rd14, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd17, [p_k];
    add.u64 %rd17, %rd17, %rd14;   // k_head
    ld.param.u64 %rd18, [p_v];
    add.u64 %rd18, %rd18, %rd14;   // v_head

    // Handle valid_kv == 0 uniformly
    setp.ne.u64 %p2, %rd12, 0;
    @%p2 bra FSF_INIT;
    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd19, %r1;
FSF_ZERO:
    setp.ge.u64 %p3, %rd19, %rd5;
    @%p3 bra FSF_ZERO_LSE;
    shl.b64 %rd20, %rd19, 2;
    add.u64 %rd21, %rd16, %rd20;
    st.global.f32 [%rd21], %f1;
    add.u64 %rd19, %rd19, 64;
    bra FSF_ZERO;
FSF_ZERO_LSE:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra FSF_DONE;
    ld.param.u64 %rd20, [p_lse];
    shl.b64 %rd21, %rd1, 2;
    add.u64 %rd20, %rd20, %rd21;
    mov.f32 %f2, 0fFF800000;       // -inf
    st.global.f32 [%rd20], %f2;
    bra FSF_DONE;

FSF_INIT:
    mov.f32 %f1, 0fFF800000;       // m_i = -inf
    mov.f32 %f2, 0f00000000;       // l_i = 0.0
    ld.param.f32 %f3, [p_scale];
    mov.u64 %rd19, 0;              // j_start = 0
    cvt.u64.u32 %rd24, %r1;        // tid u64
    shl.b64 %rd25, %rd24, 2;       // tid * 4

FSF_TILE:
    setp.ge.u64 %p2, %rd19, %rd12;
    @%p2 bra FSF_FINAL;

    sub.u64 %rd20, %rd12, %rd19;
    mov.u64 %rd21, 64;
    min.u64 %rd20, %rd20, %rd21;   // bc = min(64, valid_kv - j_start)

    // Each thread tid < bc computes score s_j = dot(q_row, k_head[j_start + tid]) * scale
    mov.f32 %f4, 0fFF800000;       // s_j = -inf
    setp.ge.u64 %p3, %rd24, %rd20;
    @%p3 bra FSF_STORE_S;
    add.u64 %rd21, %rd19, %rd24;   // kj = j_start + tid
    mul.lo.u64 %rd21, %rd21, %rd5;
    shl.b64 %rd21, %rd21, 2;
    add.u64 %rd21, %rd17, %rd21;   // &k_head[kj * D]
    mov.f32 %f4, 0f00000000;
    mov.u64 %rd22, 0;
FSF_DOT:
    setp.ge.u64 %p4, %rd22, %rd5;
    @%p4 bra FSF_DOT_END;
    shl.b64 %rd23, %rd22, 2;
    add.u64 %rd14, %rd15, %rd23;
    ld.global.f32 %f5, [%rd14];
    add.u64 %rd14, %rd21, %rd23;
    ld.global.f32 %f6, [%rd14];
    fma.rn.f32 %f4, %f5, %f6, %f4;
    add.u64 %rd22, %rd22, 1;
    bra FSF_DOT;
FSF_DOT_END:
    mul.f32 %f4, %f4, %f3;         // s_j = dot * scale

FSF_STORE_S:
    mov.u64 %rd21, smem_fsf_s;
    add.u64 %rd21, %rd21, %rd25;
    st.shared.f32 [%rd21], %f4;
    mov.u64 %rd22, smem_fsf_r;
    add.u64 %rd22, %rd22, %rd25;
    st.shared.f32 [%rd22], %f4;
    bar.sync 0;

    // Tree max reduction over 64 threads
    mov.u32 %r2, 32;
FSF_MAX_TREE:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra FSF_MAX_DONE;
    setp.ge.u32 %p3, %r1, %r2;
    @%p3 bra FSF_MAX_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd23, %r3;
    shl.b64 %rd23, %rd23, 2;
    mov.u64 %rd14, smem_fsf_r;
    add.u64 %rd14, %rd14, %rd23;
    ld.shared.f32 %f5, [%rd14];
    ld.shared.f32 %f6, [%rd22];
    max.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd22], %f6;
FSF_MAX_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra FSF_MAX_TREE;

FSF_MAX_DONE:
    mov.u64 %rd14, smem_fsf_r;
    ld.shared.f32 %f7, [%rd14];    // tile_max
    max.f32 %f8, %f1, %f7;         // m_new = max(m_i, tile_max)

    // alpha = (j_start == 0) ? 0.0 : exp(clamp(m_i - m_new, -80, 0))
    mov.f32 %f9, 0f00000000;
    setp.eq.u64 %p4, %rd19, 0;
    @%p4 bra FSF_ALPHA_OK;
    sub.f32 %f10, %f1, %f8;
    mov.f32 %f11, 0fC2A00000;      // -80.0
    max.f32 %f10, %f10, %f11;
    mov.f32 %f12, 0f3FB8AA3B;      // log2(e)
    mul.f32 %f10, %f10, %f12;
    ex2.approx.f32 %f9, %f10;      // alpha
FSF_ALPHA_OK:
    bar.sync 0;

    // Each thread computes p_j = (tid < bc) ? exp(clamp(s_j - m_new, -80, 0)) : 0.0
    mov.f32 %f13, 0f00000000;
    setp.ge.u64 %p3, %rd24, %rd20;
    @%p3 bra FSF_STORE_P;
    ld.shared.f32 %f4, [%rd21];    // s_j
    sub.f32 %f10, %f4, %f8;
    mov.f32 %f11, 0fC2A00000;
    max.f32 %f10, %f10, %f11;
    mov.f32 %f12, 0f3FB8AA3B;
    mul.f32 %f10, %f10, %f12;
    ex2.approx.f32 %f13, %f10;     // p_j
FSF_STORE_P:
    mov.u64 %rd21, smem_fsf_p;
    add.u64 %rd21, %rd21, %rd25;
    st.shared.f32 [%rd21], %f13;
    st.shared.f32 [%rd22], %f13;
    bar.sync 0;

    // Tree sum reduction over 64 threads
    mov.u32 %r2, 32;
FSF_SUM_TREE:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra FSF_SUM_DONE;
    setp.ge.u32 %p3, %r1, %r2;
    @%p3 bra FSF_SUM_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd23, %r3;
    shl.b64 %rd23, %rd23, 2;
    mov.u64 %rd14, smem_fsf_r;
    add.u64 %rd14, %rd14, %rd23;
    ld.shared.f32 %f5, [%rd14];
    ld.shared.f32 %f6, [%rd22];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd22], %f6;
FSF_SUM_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra FSF_SUM_TREE;

FSF_SUM_DONE:
    mov.u64 %rd14, smem_fsf_r;
    ld.shared.f32 %f14, [%rd14];   // tile_sum
    fma.rn.f32 %f2, %f2, %f9, %f14; // l_i = l_i * alpha + tile_sum
    mov.f32 %f1, %f8;              // m_i = m_new

    // Accumulate O[qi, d] = O[qi, d] * alpha + sum(c=0..bc) smem_p[c] * V[j_start+c, d]
    mov.u64 %rd22, %rd24;          // d = tid
FSF_OUT_D:
    setp.ge.u64 %p3, %rd22, %rd5;
    @%p3 bra FSF_OUT_END;
    shl.b64 %rd23, %rd22, 2;
    add.u64 %rd14, %rd16, %rd23;   // &out_row[d]
    mov.f32 %f15, 0f00000000;
    setp.eq.u64 %p4, %rd19, 0;
    @%p4 bra FSF_V_INIT;
    ld.global.f32 %f15, [%rd14];
    mul.f32 %f15, %f15, %f9;
FSF_V_INIT:
    mov.u64 %rd21, 0;              // c = 0
FSF_V_LOOP:
    setp.ge.u64 %p5, %rd21, %rd20;
    @%p5 bra FSF_V_STORE;
    shl.b64 %rd13, %rd21, 2;
    mov.u64 %rd11, smem_fsf_p;
    add.u64 %rd11, %rd11, %rd13;
    ld.shared.f32 %f16, [%rd11];   // p_c
    add.u64 %rd11, %rd19, %rd21;   // j_start + c
    mul.lo.u64 %rd11, %rd11, %rd5;
    add.u64 %rd11, %rd11, %rd22;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd11, %rd18, %rd11;   // &v_head[(j_start+c)*D + d]
    ld.global.f32 %f17, [%rd11];
    fma.rn.f32 %f15, %f16, %f17, %f15;
    add.u64 %rd21, %rd21, 1;
    bra FSF_V_LOOP;
FSF_V_STORE:
    st.global.f32 [%rd14], %f15;
    add.u64 %rd22, %rd22, 64;
    bra FSF_OUT_D;

FSF_OUT_END:
    bar.sync 0;
    add.u64 %rd19, %rd19, 64;
    bra FSF_TILE;

FSF_FINAL:
    mov.f32 %f10, 0f3F800000;
    div.rn.f32 %f11, %f10, %f2;    // inv_l = 1.0 / l_i
    mov.u64 %rd22, %rd24;
FSF_NORM_D:
    setp.ge.u64 %p3, %rd22, %rd5;
    @%p3 bra FSF_LSE;
    shl.b64 %rd23, %rd22, 2;
    add.u64 %rd14, %rd16, %rd23;
    ld.global.f32 %f15, [%rd14];
    mul.f32 %f15, %f15, %f11;
    st.global.f32 [%rd14], %f15;
    add.u64 %rd22, %rd22, 64;
    bra FSF_NORM_D;

FSF_LSE:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra FSF_DONE;
    lg2.approx.f32 %f12, %f2;
    mov.f32 %f13, 0f3F317218;      // ln(2)
    fma.rn.f32 %f12, %f12, %f13, %f1; // m_i + ln(l_i)
    ld.param.u64 %rd20, [p_lse];
    shl.b64 %rd21, %rd1, 2;
    add.u64 %rd20, %rd20, %rd21;
    st.global.f32 [%rd20], %f12;
FSF_DONE:
    ret;
}

.visible .entry ochre_flash_sdpa_bwd_dq_f32(
    .param .u64 p_do,
    .param .u64 p_q,
    .param .u64 p_k,
    .param .u64 p_v,
    .param .u64 p_out,
    .param .u64 p_lse,
    .param .u64 p_d_row,
    .param .u64 p_dq,
    .param .u64 p_total_q_rows,
    .param .u64 p_num_heads,
    .param .u64 p_num_kv_heads,
    .param .u64 p_q_len,
    .param .u64 p_kv_len,
    .param .u64 p_head_dim,
    .param .u32 p_causal,
    .param .u64 p_kv_offset,
    .param .f32 p_scale
)
{
    .shared .f32 smem_fsq_d[64];
    .shared .f32 smem_fsq_ds[64];
    .reg .pred %p<6>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<28>;
    .reg .f32 %f<16>;

    mov.u32 %r4, %ctaid.x;         // row_idx = bh * q_len + qi
    mov.u32 %r1, %tid.x;           // tid in 0..63
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_total_q_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra FSQ_DONE;

    ld.param.u64 %rd3, [p_q_len];
    ld.param.u64 %rd4, [p_kv_len];
    ld.param.u64 %rd5, [p_head_dim];
    ld.param.u64 %rd6, [p_num_heads];
    ld.param.u64 %rd7, [p_num_kv_heads];

    div.u64 %rd8, %rd1, %rd3;      // bh
    rem.u64 %rd9, %rd1, %rd3;      // qi
    div.u64 %rd10, %rd8, %rd6;     // b
    rem.u64 %rd11, %rd8, %rd6;     // h
    div.u64 %rd12, %rd6, %rd7;     // group_size
    div.u64 %rd11, %rd11, %rd12;   // kv_h
    mul.lo.u64 %rd10, %rd10, %rd7;
    add.u64 %rd10, %rd10, %rd11;   // kv_bh

    // Base pointers
    mul.lo.u64 %rd14, %rd1, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd15, [p_q];
    add.u64 %rd15, %rd15, %rd14;   // q_row
    ld.param.u64 %rd16, [p_do];
    add.u64 %rd16, %rd16, %rd14;   // do_row
    ld.param.u64 %rd26, [p_out];
    add.u64 %rd26, %rd26, %rd14;   // out_row
    ld.param.u64 %rd27, [p_dq];
    add.u64 %rd27, %rd27, %rd14;   // dq_row

    cvt.u64.u32 %rd24, %r1;
    shl.b64 %rd25, %rd24, 2;

    // Step 1: Compute D_i = sum_d do_row[d] * out_row[d] and zero dq_row
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.u64 %rd19, %rd24;
FSQ_DI_LOOP:
    setp.ge.u64 %p2, %rd19, %rd5;
    @%p2 bra FSQ_DI_RED;
    shl.b64 %rd20, %rd19, 2;
    add.u64 %rd21, %rd16, %rd20;
    ld.global.f32 %f3, [%rd21];
    add.u64 %rd21, %rd26, %rd20;
    ld.global.f32 %f4, [%rd21];
    fma.rn.f32 %f1, %f3, %f4, %f1;
    add.u64 %rd21, %rd27, %rd20;
    st.global.f32 [%rd21], %f2;    // zero dq_row[d]
    add.u64 %rd19, %rd19, 64;
    bra FSQ_DI_LOOP;

FSQ_DI_RED:
    mov.u64 %rd22, smem_fsq_d;
    add.u64 %rd22, %rd22, %rd25;
    st.shared.f32 [%rd22], %f1;
    bar.sync 0;

    mov.u32 %r2, 32;
FSQ_DI_TREE:
    setp.eq.u32 %p3, %r2, 0;
    @%p3 bra FSQ_DI_DONE;
    setp.ge.u32 %p2, %r1, %r2;
    @%p2 bra FSQ_DI_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd20, %r3;
    shl.b64 %rd20, %rd20, 2;
    mov.u64 %rd21, smem_fsq_d;
    add.u64 %rd21, %rd21, %rd20;
    ld.shared.f32 %f3, [%rd21];
    ld.shared.f32 %f4, [%rd22];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd22], %f4;
FSQ_DI_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra FSQ_DI_TREE;

FSQ_DI_DONE:
    mov.u64 %rd21, smem_fsq_d;
    ld.shared.f32 %f1, [%rd21];    // D_i
    setp.ne.u32 %p2, %r1, 0;
    @%p2 bra FSQ_CHECK_LSE;
    ld.param.u64 %rd20, [p_d_row];
    shl.b64 %rd21, %rd1, 2;
    add.u64 %rd20, %rd20, %rd21;
    st.global.f32 [%rd20], %f1;

FSQ_CHECK_LSE:
    ld.param.u64 %rd20, [p_lse];
    shl.b64 %rd21, %rd1, 2;
    add.u64 %rd20, %rd20, %rd21;
    ld.global.f32 %f2, [%rd20];    // lse_i
    mov.f32 %f3, 0fFC000000;       // -1.06e36 guard for -inf
    setp.le.f32 %p2, %f2, %f3;
    @%p2 bra FSQ_DONE;

    // valid_kv = causal ? min(kv_offset + qi + 1, kv_len) : kv_len
    mov.u64 %rd12, %rd4;
    ld.param.u32 %r5, [p_causal];
    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra FSQ_VKV_OK;
    ld.param.u64 %rd13, [p_kv_offset];
    add.u64 %rd13, %rd13, %rd9;
    add.u64 %rd13, %rd13, 1;
    min.u64 %rd12, %rd12, %rd13;
FSQ_VKV_OK:
    setp.eq.u64 %p2, %rd12, 0;
    @%p2 bra FSQ_DONE;

    mul.lo.u64 %rd14, %rd10, %rd4;
    mul.lo.u64 %rd14, %rd14, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd17, [p_k];
    add.u64 %rd17, %rd17, %rd14;   // k_head
    ld.param.u64 %rd18, [p_v];
    add.u64 %rd18, %rd18, %rd14;   // v_head

    ld.param.f32 %f3, [p_scale];
    mov.u64 %rd19, 0;              // j_start

FSQ_TILE:
    setp.ge.u64 %p2, %rd19, %rd12;
    @%p2 bra FSQ_DONE;

    sub.u64 %rd20, %rd12, %rd19;
    mov.u64 %rd21, 64;
    min.u64 %rd20, %rd20, %rd21;   // bc

    mov.f32 %f10, 0f00000000;      // ds_j = 0.0
    setp.ge.u64 %p3, %rd24, %rd20;
    @%p3 bra FSQ_STORE_DS;

    add.u64 %rd21, %rd19, %rd24;   // kj
    mul.lo.u64 %rd21, %rd21, %rd5;
    shl.b64 %rd21, %rd21, 2;
    add.u64 %rd22, %rd17, %rd21;   // k_kj
    add.u64 %rd23, %rd18, %rd21;   // v_kj
    mov.f32 %f4, 0f00000000;       // s_j
    mov.f32 %f5, 0f00000000;       // dp_j
    mov.u64 %rd13, 0;
FSQ_DOT:
    setp.ge.u64 %p4, %rd13, %rd5;
    @%p4 bra FSQ_DOT_END;
    shl.b64 %rd14, %rd13, 2;
    add.u64 %rd11, %rd15, %rd14;
    ld.global.f32 %f6, [%rd11];    // q
    add.u64 %rd11, %rd22, %rd14;
    ld.global.f32 %f7, [%rd11];    // k
    fma.rn.f32 %f4, %f6, %f7, %f4;
    add.u64 %rd11, %rd16, %rd14;
    ld.global.f32 %f8, [%rd11];    // do
    add.u64 %rd11, %rd23, %rd14;
    ld.global.f32 %f9, [%rd11];    // v
    fma.rn.f32 %f5, %f8, %f9, %f5;
    add.u64 %rd13, %rd13, 1;
    bra FSQ_DOT;
FSQ_DOT_END:
    mul.f32 %f4, %f4, %f3;         // s_j * scale
    sub.f32 %f11, %f4, %f2;        // s_j - lse_i
    mov.f32 %f12, 0fC2A00000;      // -80.0
    mov.f32 %f13, 0f42A00000;      // +80.0
    max.f32 %f11, %f11, %f12;
    min.f32 %f11, %f11, %f13;
    mov.f32 %f14, 0f3FB8AA3B;      // log2(e)
    mul.f32 %f11, %f11, %f14;
    ex2.approx.f32 %f11, %f11;     // p_j
    sub.f32 %f15, %f5, %f1;        // dp_j - D_i
    mul.f32 %f10, %f11, %f15;
    mul.f32 %f10, %f10, %f3;       // ds_j = p_j * (dp_j - D_i) * scale

FSQ_STORE_DS:
    mov.u64 %rd21, smem_fsq_ds;
    add.u64 %rd21, %rd21, %rd25;
    st.shared.f32 [%rd21], %f10;
    bar.sync 0;

    // Accumulate dQ[qi, d] += sum(c=0..bc) smem_ds[c] * K[j_start + c, d]
    mov.u64 %rd22, %rd24;
FSQ_ACC_D:
    setp.ge.u64 %p3, %rd22, %rd5;
    @%p3 bra FSQ_ACC_END;
    shl.b64 %rd23, %rd22, 2;
    add.u64 %rd14, %rd27, %rd23;   // &dq_row[d]
    ld.global.f32 %f15, [%rd14];
    mov.u64 %rd21, 0;
FSQ_K_LOOP:
    setp.ge.u64 %p5, %rd21, %rd20;
    @%p5 bra FSQ_K_STORE;
    shl.b64 %rd13, %rd21, 2;
    mov.u64 %rd11, smem_fsq_ds;
    add.u64 %rd11, %rd11, %rd13;
    ld.shared.f32 %f11, [%rd11];
    add.u64 %rd11, %rd19, %rd21;
    mul.lo.u64 %rd11, %rd11, %rd5;
    add.u64 %rd11, %rd11, %rd22;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd11, %rd17, %rd11;
    ld.global.f32 %f12, [%rd11];
    fma.rn.f32 %f15, %f11, %f12, %f15;
    add.u64 %rd21, %rd21, 1;
    bra FSQ_K_LOOP;
FSQ_K_STORE:
    st.global.f32 [%rd14], %f15;
    add.u64 %rd22, %rd22, 64;
    bra FSQ_ACC_D;

FSQ_ACC_END:
    bar.sync 0;
    add.u64 %rd19, %rd19, 64;
    bra FSQ_TILE;
FSQ_DONE:
    ret;
}

.visible .entry ochre_flash_sdpa_bwd_dkv_f32(
    .param .u64 p_do,
    .param .u64 p_q,
    .param .u64 p_k,
    .param .u64 p_v,
    .param .u64 p_lse,
    .param .u64 p_d_row,
    .param .u64 p_dk,
    .param .u64 p_dv,
    .param .u64 p_total_kv_rows,
    .param .u64 p_num_heads,
    .param .u64 p_num_kv_heads,
    .param .u64 p_q_len,
    .param .u64 p_kv_len,
    .param .u64 p_head_dim,
    .param .u32 p_causal,
    .param .u64 p_kv_offset,
    .param .f32 p_scale
)
{
    .shared .f32 smem_fsk_p[64];
    .shared .f32 smem_fsk_ds[64];
    .reg .pred %p<6>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<30>;
    .reg .f32 %f<18>;

    mov.u32 %r4, %ctaid.x;         // kv_row_idx = kv_bh * kv_len + kj
    mov.u32 %r1, %tid.x;           // tid in 0..63
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_total_kv_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra FSK_DONE;

    ld.param.u64 %rd3, [p_q_len];
    ld.param.u64 %rd4, [p_kv_len];
    ld.param.u64 %rd5, [p_head_dim];
    ld.param.u64 %rd6, [p_num_heads];
    ld.param.u64 %rd7, [p_num_kv_heads];

    div.u64 %rd8, %rd1, %rd4;      // kv_bh
    rem.u64 %rd9, %rd1, %rd4;      // kj
    div.u64 %rd10, %rd8, %rd7;     // b
    rem.u64 %rd11, %rd8, %rd7;     // kv_h
    div.u64 %rd12, %rd6, %rd7;     // group_size

    // Base pointers for k_row, v_row, dk_row, dv_row
    mul.lo.u64 %rd14, %rd1, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd15, [p_k];
    add.u64 %rd15, %rd15, %rd14;   // k_row
    ld.param.u64 %rd16, [p_v];
    add.u64 %rd16, %rd16, %rd14;   // v_row
    ld.param.u64 %rd17, [p_dk];
    add.u64 %rd17, %rd17, %rd14;   // dk_row
    ld.param.u64 %rd18, [p_dv];
    add.u64 %rd18, %rd18, %rd14;   // dv_row

    cvt.u64.u32 %rd24, %r1;
    shl.b64 %rd25, %rd24, 2;

    // Zero dk_row and dv_row
    mov.f32 %f1, 0f00000000;
    mov.u64 %rd19, %rd24;
FSK_ZERO:
    setp.ge.u64 %p2, %rd19, %rd5;
    @%p2 bra FSK_ZERO_DONE;
    shl.b64 %rd20, %rd19, 2;
    add.u64 %rd21, %rd17, %rd20;
    st.global.f32 [%rd21], %f1;
    add.u64 %rd21, %rd18, %rd20;
    st.global.f32 [%rd21], %f1;
    add.u64 %rd19, %rd19, 64;
    bra FSK_ZERO;
FSK_ZERO_DONE:

    // qi_start = causal ? saturating_sub(kj, kv_offset) : 0
    mov.u64 %rd13, 0;
    ld.param.u32 %r5, [p_causal];
    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra FSK_QI_OK;
    ld.param.u64 %rd20, [p_kv_offset];
    setp.le.u64 %p2, %rd9, %rd20;
    @%p2 bra FSK_QI_OK;
    sub.u64 %rd13, %rd9, %rd20;
FSK_QI_OK:
    setp.ge.u64 %p2, %rd13, %rd3;
    @%p2 bra FSK_DONE;

    ld.param.f32 %f3, [p_scale];
    // First query head for this kv_h: bh_base = b * num_heads + kv_h * group_size
    mul.lo.u64 %rd26, %rd10, %rd6;
    mul.lo.u64 %rd20, %rd11, %rd12;
    add.u64 %rd26, %rd26, %rd20;   // bh_base

    mov.u64 %rd27, 0;              // g = 0..group_size
FSK_G_LOOP:
    setp.ge.u64 %p2, %rd27, %rd12;
    @%p2 bra FSK_DONE;

    add.u64 %rd28, %rd26, %rd27;   // bh = bh_base + g
    mul.lo.u64 %rd29, %rd28, %rd3; // bh * q_len

    mov.u64 %rd19, %rd13;          // i_start = qi_start
FSK_I_TILE:
    setp.ge.u64 %p2, %rd19, %rd3;
    @%p2 bra FSK_G_NEXT;

    sub.u64 %rd20, %rd3, %rd19;
    mov.u64 %rd21, 64;
    min.u64 %rd20, %rd20, %rd21;   // br

    mov.f32 %f10, 0f00000000;      // p_i
    mov.f32 %f11, 0f00000000;      // ds_i
    setp.ge.u64 %p3, %rd24, %rd20;
    @%p3 bra FSK_STORE_PDS;

    add.u64 %rd21, %rd19, %rd24;   // qi = i_start + tid
    add.u64 %rd21, %rd29, %rd21;   // row_idx = bh * q_len + qi
    shl.b64 %rd22, %rd21, 2;
    ld.param.u64 %rd23, [p_lse];
    add.u64 %rd23, %rd23, %rd22;
    ld.global.f32 %f2, [%rd23];    // lse_i
    mov.f32 %f4, 0fFC000000;
    setp.le.f32 %p4, %f2, %f4;
    @%p4 bra FSK_STORE_PDS;

    ld.param.u64 %rd23, [p_d_row];
    add.u64 %rd23, %rd23, %rd22;
    ld.global.f32 %f1, [%rd23];    // D_i

    mul.lo.u64 %rd21, %rd21, %rd5;
    shl.b64 %rd21, %rd21, 2;
    ld.param.u64 %rd22, [p_q];
    add.u64 %rd22, %rd22, %rd21;   // q_qi
    ld.param.u64 %rd23, [p_do];
    add.u64 %rd23, %rd23, %rd21;   // do_qi

    mov.f32 %f4, 0f00000000;       // s_i
    mov.f32 %f5, 0f00000000;       // dp_i
    mov.u64 %rd14, 0;
FSK_DOT:
    setp.ge.u64 %p5, %rd14, %rd5;
    @%p5 bra FSK_DOT_END;
    shl.b64 %rd8, %rd14, 2;
    add.u64 %rd10, %rd22, %rd8;
    ld.global.f32 %f6, [%rd10];
    add.u64 %rd10, %rd15, %rd8;
    ld.global.f32 %f7, [%rd10];
    fma.rn.f32 %f4, %f6, %f7, %f4;
    add.u64 %rd10, %rd23, %rd8;
    ld.global.f32 %f8, [%rd10];
    add.u64 %rd10, %rd16, %rd8;
    ld.global.f32 %f9, [%rd10];
    fma.rn.f32 %f5, %f8, %f9, %f5;
    add.u64 %rd14, %rd14, 1;
    bra FSK_DOT;
FSK_DOT_END:
    mul.f32 %f4, %f4, %f3;
    sub.f32 %f12, %f4, %f2;
    mov.f32 %f13, 0fC2A00000;
    mov.f32 %f14, 0f42A00000;
    max.f32 %f12, %f12, %f13;
    min.f32 %f12, %f12, %f14;
    mov.f32 %f15, 0f3FB8AA3B;
    mul.f32 %f12, %f12, %f15;
    ex2.approx.f32 %f10, %f12;     // p_i
    sub.f32 %f16, %f5, %f1;
    mul.f32 %f11, %f10, %f16;
    mul.f32 %f11, %f11, %f3;       // ds_i

FSK_STORE_PDS:
    mov.u64 %rd21, smem_fsk_p;
    add.u64 %rd21, %rd21, %rd25;
    st.shared.f32 [%rd21], %f10;
    mov.u64 %rd21, smem_fsk_ds;
    add.u64 %rd21, %rd21, %rd25;
    st.shared.f32 [%rd21], %f11;
    bar.sync 0;

    // Accumulate dV[kj, d] and dK[kj, d]
    mov.u64 %rd22, %rd24;
FSK_ACC_D:
    setp.ge.u64 %p3, %rd22, %rd5;
    @%p3 bra FSK_ACC_END;
    shl.b64 %rd23, %rd22, 2;
    add.u64 %rd14, %rd17, %rd23;   // &dk_row[d]
    add.u64 %rd8, %rd18, %rd23;    // &dv_row[d]
    ld.global.f32 %f16, [%rd14];   // dk_acc
    ld.global.f32 %f17, [%rd8];    // dv_acc

    mov.u64 %rd21, 0;              // r = 0..br
FSK_R_LOOP:
    setp.ge.u64 %p5, %rd21, %rd20;
    @%p5 bra FSK_R_STORE;
    shl.b64 %rd10, %rd21, 2;
    mov.u64 %rd11, smem_fsk_p;
    add.u64 %rd11, %rd11, %rd10;
    ld.shared.f32 %f10, [%rd11];   // p_r
    mov.u64 %rd11, smem_fsk_ds;
    add.u64 %rd11, %rd11, %rd10;
    ld.shared.f32 %f11, [%rd11];   // ds_r

    add.u64 %rd11, %rd19, %rd21;   // qi = i_start + r
    add.u64 %rd11, %rd29, %rd11;   // bh * q_len + qi
    mul.lo.u64 %rd11, %rd11, %rd5;
    add.u64 %rd11, %rd11, %rd22;
    shl.b64 %rd11, %rd11, 2;
    ld.param.u64 %rd10, [p_q];
    add.u64 %rd10, %rd10, %rd11;
    ld.global.f32 %f12, [%rd10];   // Q[bh, qi, d]
    fma.rn.f32 %f16, %f11, %f12, %f16;
    ld.param.u64 %rd10, [p_do];
    add.u64 %rd10, %rd10, %rd11;
    ld.global.f32 %f13, [%rd10];   // dO[bh, qi, d]
    fma.rn.f32 %f17, %f10, %f13, %f17;

    add.u64 %rd21, %rd21, 1;
    bra FSK_R_LOOP;
FSK_R_STORE:
    st.global.f32 [%rd14], %f16;
    st.global.f32 [%rd8], %f17;
    add.u64 %rd22, %rd22, 64;
    bra FSK_ACC_D;

FSK_ACC_END:
    bar.sync 0;
    add.u64 %rd19, %rd19, 64;
    bra FSK_I_TILE;

FSK_G_NEXT:
    add.u64 %rd27, %rd27, 1;
    bra FSK_G_LOOP;
FSK_DONE:
    ret;
}

.visible .entry ochre_embedding_scatter_add_f32(
    .param .u64 p_dy,
    .param .u64 p_ids,
    .param .u64 p_dweight,
    .param .u64 p_num_tokens,
    .param .u64 p_d_model
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;
    ld.param.u64 %rd2, [p_num_tokens];
    ld.param.u64 %rd3, [p_d_model];
    mul.lo.u64 %rd4, %rd2, %rd3;
    setp.ge.u64 %p1, %rd1, %rd4;
    @%p1 bra ESA_DONE;

    div.u64 %rd5, %rd1, %rd3;      // tok
    rem.u64 %rd6, %rd1, %rd3;      // col

    ld.param.u64 %rd7, [p_ids];
    shl.b64 %rd8, %rd5, 3;
    add.u64 %rd7, %rd7, %rd8;
    ld.global.u64 %rd9, [%rd7];    // vocab_id

    ld.param.u64 %rd10, [p_dy];
    shl.b64 %rd8, %rd1, 2;
    add.u64 %rd10, %rd10, %rd8;
    ld.global.f32 %f1, [%rd10];

    mul.lo.u64 %rd11, %rd9, %rd3;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;
    ld.param.u64 %rd10, [p_dweight];
    add.u64 %rd10, %rd10, %rd11;
    atom.global.add.f32 %f2, [%rd10], %f1;
ESA_DONE:
    ret;
}

.visible .entry ochre_bshd_to_bhsd_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_batch,
    .param .u64 p_seq_len,
    .param .u64 p_num_heads,
    .param .u64 p_head_dim
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<15>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // flat src idx in [B, S, H, D]
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_seq_len];
    ld.param.u64 %rd4, [p_num_heads];
    ld.param.u64 %rd5, [p_head_dim];
    mul.lo.u64 %rd6, %rd2, %rd3;
    mul.lo.u64 %rd6, %rd6, %rd4;
    mul.lo.u64 %rd6, %rd6, %rd5;
    setp.ge.u64 %p1, %rd1, %rd6;
    @%p1 bra BSHD_DONE;

    rem.u64 %rd7, %rd1, %rd5;      // d
    div.u64 %rd8, %rd1, %rd5;
    rem.u64 %rd9, %rd8, %rd4;      // h
    div.u64 %rd8, %rd8, %rd4;
    rem.u64 %rd10, %rd8, %rd3;     // s
    div.u64 %rd11, %rd8, %rd3;     // b

    ld.param.u64 %rd12, [p_src];
    shl.b64 %rd13, %rd1, 2;
    add.u64 %rd12, %rd12, %rd13;
    ld.global.f32 %f1, [%rd12];

    // dst idx = ((b * H + h) * S + s) * D + d
    mul.lo.u64 %rd14, %rd11, %rd4;
    add.u64 %rd14, %rd14, %rd9;
    mul.lo.u64 %rd14, %rd14, %rd3;
    add.u64 %rd14, %rd14, %rd10;
    mul.lo.u64 %rd14, %rd14, %rd5;
    add.u64 %rd14, %rd14, %rd7;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd12, [p_dst];
    add.u64 %rd12, %rd12, %rd14;
    st.global.f32 [%rd12], %f1;
BSHD_DONE:
    ret;
}

.visible .entry ochre_bhsd_to_bshd_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_batch,
    .param .u64 p_num_heads,
    .param .u64 p_seq_len,
    .param .u64 p_head_dim
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<15>;
    .reg .f32 %f1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    cvt.u64.u32 %rd1, %r1;         // flat src idx in [B, H, S, D]
    ld.param.u64 %rd2, [p_batch];
    ld.param.u64 %rd3, [p_num_heads];
    ld.param.u64 %rd4, [p_seq_len];
    ld.param.u64 %rd5, [p_head_dim];
    mul.lo.u64 %rd6, %rd2, %rd3;
    mul.lo.u64 %rd6, %rd6, %rd4;
    mul.lo.u64 %rd6, %rd6, %rd5;
    setp.ge.u64 %p1, %rd1, %rd6;
    @%p1 bra BHSD_DONE;

    rem.u64 %rd7, %rd1, %rd5;      // d
    div.u64 %rd8, %rd1, %rd5;
    rem.u64 %rd9, %rd8, %rd4;      // s
    div.u64 %rd8, %rd8, %rd4;
    rem.u64 %rd10, %rd8, %rd3;     // h
    div.u64 %rd11, %rd8, %rd3;     // b

    ld.param.u64 %rd12, [p_src];
    shl.b64 %rd13, %rd1, 2;
    add.u64 %rd12, %rd12, %rd13;
    ld.global.f32 %f1, [%rd12];

    // dst idx = ((b * S + s) * H + h) * D + d
    mul.lo.u64 %rd14, %rd11, %rd4;
    add.u64 %rd14, %rd14, %rd9;
    mul.lo.u64 %rd14, %rd14, %rd3;
    add.u64 %rd14, %rd14, %rd10;
    mul.lo.u64 %rd14, %rd14, %rd5;
    add.u64 %rd14, %rd14, %rd7;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd12, [p_dst];
    add.u64 %rd12, %rd12, %rd14;
    st.global.f32 [%rd12], %f1;
BHSD_DONE:
    ret;
}

.visible .entry ochre_sdpa_causal_softmax_f32(
    .param .u64 p_s,
    .param .u64 p_lse,
    .param .u64 p_total_rows,
    .param .u64 p_q_len,
    .param .u64 p_kv_len,
    .param .u32 p_causal,
    .param .u64 p_kv_offset
)
{
    .shared .f32 smem_scs[128];
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<10>;

    mov.u32 %r4, %ctaid.x;         // row_idx
    mov.u32 %r1, %tid.x;           // tid in 0..127
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_total_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SCS_DONE;

    ld.param.u64 %rd3, [p_q_len];
    ld.param.u64 %rd4, [p_kv_len];
    rem.u64 %rd5, %rd1, %rd3;      // qi = row_idx % q_len

    // valid_kv = causal ? min(kv_offset + qi + 1, kv_len) : kv_len
    mov.u64 %rd6, %rd4;
    ld.param.u32 %r5, [p_causal];
    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra SCS_VKV_OK;
    ld.param.u64 %rd7, [p_kv_offset];
    add.u64 %rd7, %rd7, %rd5;
    add.u64 %rd7, %rd7, 1;
    min.u64 %rd6, %rd6, %rd7;
SCS_VKV_OK:

    mul.lo.u64 %rd8, %rd1, %rd4;
    shl.b64 %rd8, %rd8, 2;
    ld.param.u64 %rd9, [p_s];
    add.u64 %rd9, %rd9, %rd8;      // s_row

    cvt.u64.u32 %rd10, %r1;        // tid u64
    shl.b64 %rd11, %rd10, 2;       // tid * 4
    mov.u64 %rd12, smem_scs;
    add.u64 %rd12, %rd12, %rd11;   // &smem_scs[tid]

    // Handle valid_kv == 0
    setp.ne.u64 %p2, %rd6, 0;
    @%p2 bra SCS_MAX_INIT;
    mov.f32 %f1, 0f00000000;
    mov.u64 %rd13, %rd10;
SCS_ZERO_ALL:
    setp.ge.u64 %p3, %rd13, %rd4;
    @%p3 bra SCS_ZERO_LSE;
    shl.b64 %rd14, %rd13, 2;
    add.u64 %rd15, %rd9, %rd14;
    st.global.f32 [%rd15], %f1;
    add.u64 %rd13, %rd13, 128;
    bra SCS_ZERO_ALL;
SCS_ZERO_LSE:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra SCS_DONE;
    ld.param.u64 %rd14, [p_lse];
    shl.b64 %rd15, %rd1, 2;
    add.u64 %rd14, %rd14, %rd15;
    mov.f32 %f2, 0fFF800000;       // -inf
    st.global.f32 [%rd14], %f2;
    bra SCS_DONE;

SCS_MAX_INIT:
    mov.f32 %f1, 0fFF800000;       // local_max = -inf
    mov.u64 %rd13, %rd10;
SCS_MAX_LOOP:
    setp.ge.u64 %p3, %rd13, %rd6;
    @%p3 bra SCS_MAX_RED;
    shl.b64 %rd14, %rd13, 2;
    add.u64 %rd15, %rd9, %rd14;
    ld.global.f32 %f2, [%rd15];
    max.f32 %f1, %f1, %f2;
    add.u64 %rd13, %rd13, 128;
    bra SCS_MAX_LOOP;

SCS_MAX_RED:
    st.shared.f32 [%rd12], %f1;
    bar.sync 0;
    mov.u32 %r2, 64;
SCS_MAX_TREE:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra SCS_MAX_DONE;
    setp.ge.u32 %p3, %r1, %r2;
    @%p3 bra SCS_MAX_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd14, %r3;
    shl.b64 %rd14, %rd14, 2;
    mov.u64 %rd15, smem_scs;
    add.u64 %rd15, %rd15, %rd14;
    ld.shared.f32 %f2, [%rd15];
    max.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd12], %f1;
SCS_MAX_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra SCS_MAX_TREE;

SCS_MAX_DONE:
    mov.u64 %rd15, smem_scs;
    ld.shared.f32 %f3, [%rd15];    // row_max
    bar.sync 0;

    // Pass 2: exp(s - row_max) for c < valid_kv, zero for c >= valid_kv
    mov.f32 %f4, 0f00000000;       // local_sum
    mov.f32 %f5, 0fC2A00000;       // -80.0
    mov.f32 %f6, 0f3FB8AA3B;       // log2(e)
    mov.u64 %rd13, %rd10;
SCS_EXP_LOOP:
    setp.ge.u64 %p3, %rd13, %rd4;
    @%p3 bra SCS_SUM_RED;
    shl.b64 %rd14, %rd13, 2;
    add.u64 %rd15, %rd9, %rd14;
    setp.ge.u64 %p4, %rd13, %rd6;
    @%p4 bra SCS_EXP_MASK;
    ld.global.f32 %f2, [%rd15];
    sub.f32 %f7, %f2, %f3;
    max.f32 %f7, %f7, %f5;
    mul.f32 %f7, %f7, %f6;
    ex2.approx.f32 %f8, %f7;
    add.f32 %f4, %f4, %f8;
    st.global.f32 [%rd15], %f8;
    bra SCS_EXP_NEXT;
SCS_EXP_MASK:
    mov.f32 %f8, 0f00000000;
    st.global.f32 [%rd15], %f8;
SCS_EXP_NEXT:
    add.u64 %rd13, %rd13, 128;
    bra SCS_EXP_LOOP;

SCS_SUM_RED:
    st.shared.f32 [%rd12], %f4;
    bar.sync 0;
    mov.u32 %r2, 64;
SCS_SUM_TREE:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra SCS_SUM_DONE;
    setp.ge.u32 %p3, %r1, %r2;
    @%p3 bra SCS_SUM_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd14, %r3;
    shl.b64 %rd14, %rd14, 2;
    mov.u64 %rd15, smem_scs;
    add.u64 %rd15, %rd15, %rd14;
    ld.shared.f32 %f2, [%rd15];
    add.f32 %f4, %f4, %f2;
    st.shared.f32 [%rd12], %f4;
SCS_SUM_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra SCS_SUM_TREE;

SCS_SUM_DONE:
    mov.u64 %rd15, smem_scs;
    ld.shared.f32 %f9, [%rd15];    // row_sum
    mov.f32 %f5, 0f3F800000;       // 1.0
    div.rn.f32 %f6, %f5, %f9;      // inv_sum = 1.0 / row_sum

    mov.u64 %rd13, %rd10;
SCS_NORM_LOOP:
    setp.ge.u64 %p3, %rd13, %rd6;
    @%p3 bra SCS_WRITE_LSE;
    shl.b64 %rd14, %rd13, 2;
    add.u64 %rd15, %rd9, %rd14;
    ld.global.f32 %f7, [%rd15];
    mul.f32 %f7, %f7, %f6;
    st.global.f32 [%rd15], %f7;
    add.u64 %rd13, %rd13, 128;
    bra SCS_NORM_LOOP;

SCS_WRITE_LSE:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra SCS_DONE;
    lg2.approx.f32 %f7, %f9;
    mov.f32 %f8, 0f3F317218;       // ln(2)
    fma.rn.f32 %f7, %f7, %f8, %f3; // row_max + ln(row_sum)
    ld.param.u64 %rd14, [p_lse];
    shl.b64 %rd15, %rd1, 2;
    add.u64 %rd14, %rd14, %rd15;
    st.global.f32 [%rd14], %f7;
SCS_DONE:
    ret;
}

.visible .entry ochre_sdpa_bwd_ds_f32(
    .param .u64 p_do,
    .param .u64 p_o,
    .param .u64 p_s,
    .param .u64 p_dp,
    .param .u64 p_lse,
    .param .u64 p_total_rows,
    .param .u64 p_q_len,
    .param .u64 p_kv_len,
    .param .u64 p_head_dim,
    .param .u32 p_causal,
    .param .u64 p_kv_offset,
    .param .f32 p_scale
)
{
    .shared .f32 smem_sbd[128];
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<13>;

    mov.u32 %r4, %ctaid.x;         // row_idx
    mov.u32 %r1, %tid.x;           // tid in 0..127
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_total_rows];
    setp.ge.u64 %p1, %rd1, %rd2;
    @%p1 bra SBD_DONE;

    ld.param.u64 %rd3, [p_q_len];
    ld.param.u64 %rd4, [p_kv_len];
    ld.param.u64 %rd5, [p_head_dim];
    rem.u64 %rd6, %rd1, %rd3;      // qi = row_idx % q_len

    // valid_kv = causal ? min(kv_offset + qi + 1, kv_len) : kv_len
    mov.u64 %rd7, %rd4;
    ld.param.u32 %r5, [p_causal];
    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra SBD_VKV_OK;
    ld.param.u64 %rd8, [p_kv_offset];
    add.u64 %rd8, %rd8, %rd6;
    add.u64 %rd8, %rd8, 1;
    min.u64 %rd7, %rd7, %rd8;
SBD_VKV_OK:

    mul.lo.u64 %rd9, %rd1, %rd4;
    shl.b64 %rd9, %rd9, 2;
    ld.param.u64 %rd10, [p_s];
    add.u64 %rd10, %rd10, %rd9;    // s_row
    ld.param.u64 %rd11, [p_dp];
    add.u64 %rd11, %rd11, %rd9;    // dp_row

    cvt.u64.u32 %rd12, %r1;        // tid u64
    shl.b64 %rd13, %rd12, 2;       // tid * 4

    ld.param.u64 %rd14, [p_lse];
    shl.b64 %rd15, %rd1, 2;
    add.u64 %rd14, %rd14, %rd15;
    ld.global.f32 %f1, [%rd14];    // lse_i
    setp.eq.u64 %p2, %rd7, 0;
    @%p2 bra SBD_ZERO_ROW;
    mov.f32 %f2, 0fFC000000;
    setp.le.f32 %p2, %f1, %f2;
    @%p2 bra SBD_ZERO_ROW;
    bra SBD_COMPUTE_DI;

SBD_ZERO_ROW:
    mov.f32 %f3, 0f00000000;
    mov.u64 %rd16, %rd12;
SBD_ZERO_LOOP:
    setp.ge.u64 %p3, %rd16, %rd4;
    @%p3 bra SBD_DONE;
    shl.b64 %rd17, %rd16, 2;
    add.u64 %rd18, %rd10, %rd17;
    st.global.f32 [%rd18], %f3;
    add.u64 %rd18, %rd11, %rd17;
    st.global.f32 [%rd18], %f3;
    add.u64 %rd16, %rd16, 128;
    bra SBD_ZERO_LOOP;

SBD_COMPUTE_DI:
    mul.lo.u64 %rd14, %rd1, %rd5;
    shl.b64 %rd14, %rd14, 2;
    ld.param.u64 %rd15, [p_do];
    add.u64 %rd15, %rd15, %rd14;   // do_row
    ld.param.u64 %rd16, [p_o];
    add.u64 %rd16, %rd16, %rd14;   // o_row

    mov.f32 %f2, 0f00000000;       // local D_i
    mov.u64 %rd17, %rd12;
SBD_DI_LOOP:
    setp.ge.u64 %p3, %rd17, %rd5;
    @%p3 bra SBD_DI_RED;
    shl.b64 %rd18, %rd17, 2;
    add.u64 %rd19, %rd15, %rd18;
    ld.global.f32 %f3, [%rd19];
    add.u64 %rd19, %rd16, %rd18;
    ld.global.f32 %f4, [%rd19];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u64 %rd17, %rd17, 128;
    bra SBD_DI_LOOP;

SBD_DI_RED:
    mov.u64 %rd17, smem_sbd;
    add.u64 %rd17, %rd17, %rd13;
    st.shared.f32 [%rd17], %f2;
    bar.sync 0;

    mov.u32 %r2, 64;
SBD_DI_TREE:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra SBD_DI_DONE;
    setp.ge.u32 %p3, %r1, %r2;
    @%p3 bra SBD_DI_SYNC;
    add.u32 %r3, %r1, %r2;
    cvt.u64.u32 %rd18, %r3;
    shl.b64 %rd18, %rd18, 2;
    mov.u64 %rd19, smem_sbd;
    add.u64 %rd19, %rd19, %rd18;
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f2, %f2, %f3;
    st.shared.f32 [%rd17], %f2;
SBD_DI_SYNC:
    bar.sync 0;
    shr.u32 %r2, %r2, 1;
    bra SBD_DI_TREE;

SBD_DI_DONE:
    mov.u64 %rd19, smem_sbd;
    ld.shared.f32 %f2, [%rd19];    // D_i
    ld.param.f32 %f3, [p_scale];
    mov.f32 %f4, 0fC2A00000;       // -80.0
    mov.f32 %f5, 0f42A00000;       // +80.0
    mov.f32 %f6, 0f3FB8AA3B;       // log2(e)

    mov.u64 %rd16, %rd12;
SBD_OUT_LOOP:
    setp.ge.u64 %p3, %rd16, %rd4;
    @%p3 bra SBD_DONE;
    shl.b64 %rd17, %rd16, 2;
    add.u64 %rd18, %rd10, %rd17;   // &s_row[c]
    add.u64 %rd19, %rd11, %rd17;   // &dp_row[c]
    setp.ge.u64 %p4, %rd16, %rd7;
    @%p4 bra SBD_OUT_MASK;

    ld.global.f32 %f7, [%rd18];    // s_val
    ld.global.f32 %f8, [%rd19];    // dp_val
    sub.f32 %f9, %f7, %f1;         // s_val - lse_i
    max.f32 %f9, %f9, %f4;
    min.f32 %f9, %f9, %f5;
    mul.f32 %f9, %f9, %f6;
    ex2.approx.f32 %f10, %f9;      // p_val
    sub.f32 %f11, %f8, %f2;        // dp_val - D_i
    mul.f32 %f12, %f10, %f11;
    mul.f32 %f12, %f12, %f3;       // dS = p * (dp - D_i) * scale
    st.global.f32 [%rd18], %f10;   // overwrite s with P
    st.global.f32 [%rd19], %f12;   // overwrite dp with dS
    bra SBD_OUT_NEXT;

SBD_OUT_MASK:
    mov.f32 %f10, 0f00000000;
    st.global.f32 [%rd18], %f10;
    st.global.f32 [%rd19], %f10;
SBD_OUT_NEXT:
    add.u64 %rd16, %rd16, 128;
    bra SBD_OUT_LOOP;

SBD_DONE:
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
        rnn_step_fwd_f32: get_fn("ochre_rnn_step_fwd_f32")?,
        rnn_step_bwd_f32: get_fn("ochre_rnn_step_bwd_f32")?,
        gru_step_fwd_f32: get_fn("ochre_gru_step_fwd_f32")?,
        gru_step_bwd_f32: get_fn("ochre_gru_step_bwd_f32")?,
        lstm_step_fwd_f32: get_fn("ochre_lstm_step_fwd_f32")?,
        lstm_step_bwd_f32: get_fn("ochre_lstm_step_bwd_f32")?,
        rope_fwd_bwd_f32: get_fn("ochre_rope_fwd_bwd_f32")?,
        swiglu_fwd_f32: get_fn("ochre_swiglu_fwd_f32")?,
        swiglu_bwd_f32: get_fn("ochre_swiglu_bwd_f32")?,
        flash_sdpa_fwd_f32: get_fn("ochre_flash_sdpa_fwd_f32")?,
        flash_sdpa_bwd_dq_f32: get_fn("ochre_flash_sdpa_bwd_dq_f32")?,
        flash_sdpa_bwd_dkv_f32: get_fn("ochre_flash_sdpa_bwd_dkv_f32")?,
        sdpa_causal_softmax_f32: get_fn("ochre_sdpa_causal_softmax_f32")?,
        sdpa_bwd_ds_f32: get_fn("ochre_sdpa_bwd_ds_f32")?,
        embedding_scatter_add_f32: get_fn("ochre_embedding_scatter_add_f32")?,
        bshd_to_bhsd_f32: get_fn("ochre_bshd_to_bhsd_f32")?,
        bhsd_to_bshd_f32: get_fn("ochre_bhsd_to_bshd_f32")?,
    })
}

fn get_nn_ctx_mutex(
    driver: &CudaDriver,
    rt: &CudaNnRuntime,
    gpu_id: usize,
) -> Option<&'static Mutex<CudaNnCtxState>> {
    let ctx_arc = driver.get_device_ctx(gpu_id).ok()?;
    let cu_ctx = {
        let guard = ctx_arc.lock().ok()?;
        guard.ctx
    };

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
            vram_pool: Arc::new(Mutex::new(VramPool::new(driver, cu_ctx))),
            ptx,
        };
        e.insert(Box::new(Mutex::new(state)));
    }
    let boxed = map.get(&key)?;
    // SAFETY: Entries in `rt.contexts` live for `'static` and are never removed.
    Some(unsafe { &*(boxed.as_ref() as *const Mutex<CudaNnCtxState>) })
}

fn with_nn_ctx<R>(
    gpu_id: usize,
    f: impl FnOnce(&CudaDriver, &CudaNnRuntime, &mut CudaNnCtxState) -> OchreResult<R>,
) -> Option<OchreResult<R>> {
    let driver = get_driver()?;
    let rt = get_nn_runtime()?;
    let ctx_mutex = get_nn_ctx_mutex(driver, rt, gpu_id)?;

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

/// Safe RAII wrapper around a pooled CUDA device memory allocation (`CuDevicePtr`).
/// Automatically recycles into its owning `VramPool` on drop (zero `cuMemFree` on hot drops).
pub struct CudaDeviceBuf {
    ptr: CuDevicePtr,
    byte_len: usize,
    bucket_len: usize,
    pool: Arc<Mutex<VramPool>>,
}

// SAFETY: `CudaDeviceBuf` owns a unique device pointer and recycles into a Mutex-guarded `VramPool`.
unsafe impl Send for CudaDeviceBuf {}
// SAFETY: `&CudaDeviceBuf` exposes only immutable pointer/size metadata; actual CUDA device operations
// are serialized through the per-context Mutex inside `with_vram_session`.
unsafe impl Sync for CudaDeviceBuf {}

impl Drop for CudaDeviceBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let ptr = self.ptr;
            let bucket_len = self.bucket_len;
            self.ptr = 0;
            self.byte_len = 0;
            match self.pool.lock() {
                Ok(mut pool) => pool.recycle(bucket_len, ptr),
                Err(poisoned) => poisoned.into_inner().recycle(bucket_len, ptr),
            }
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
    let ctx_mutex = get_nn_ctx_mutex(driver, rt, gpu_id)?;

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

    /// Allocates at least `byte_len` bytes in VRAM from the persistent slab pool (or `cuMemAlloc` on miss).
    pub fn alloc_bytes(&self, byte_len: usize) -> OchreResult<CudaDeviceBuf> {
        let logical_sz = byte_len.max(4);
        let bucket_len = pool_bucket_size(logical_sz);
        if let Ok(mut pool) = self.state.vram_pool.lock()
            && let Some(ptr) = pool.pop(bucket_len)
        {
            return Ok(CudaDeviceBuf {
                ptr,
                byte_len: logical_sz,
                bucket_len,
                pool: Arc::clone(&self.state.vram_pool),
            });
        }
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: Context is bound by `CudaVramSession`; `ptr` receives a valid device allocation.
        let mut rc = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, bucket_len) };
        if (rc != CUDA_SUCCESS || ptr == 0)
            && let Ok(mut pool) = self.state.vram_pool.lock()
        {
            pool.drain_all();
            ptr = 0;
            // SAFETY: Retry allocation after releasing cached slab buffers.
            rc = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, bucket_len) };
        }
        if rc != CUDA_SUCCESS || ptr == 0 {
            return Err(OchreError::DeviceError(format!(
                "CUDA VRAM cuMemAlloc({bucket_len}) failed with code {rc} (´；ω；`)"
            )));
        }
        Ok(CudaDeviceBuf {
            ptr,
            byte_len: logical_sz,
            bucket_len,
            pool: Arc::clone(&self.state.vram_pool),
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

    /// Device-resident strided batched cuBLAS SGEMM (`C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b]`),
    /// falling back to a loop over `sgemm_dev` when `cublasSgemmStridedBatched` is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided_batched_dev(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        d_a: CuDevicePtr,
        stride_a: usize,
        d_b: CuDevicePtr,
        stride_b: usize,
        d_c: CuDevicePtr,
        stride_c: usize,
        batch_count: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) -> OchreResult<()> {
        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        if let Some(strided_fn) = self.rt.cublas_sgemm_strided_batched {
            let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
            let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
            let lda = if trans_a { m as i32 } else { k as i32 };
            let ldb = if trans_b { k as i32 } else { n as i32 };
            // SAFETY: All strided device slices lie within valid allocations on the bound CUDA context.
            let status = unsafe {
                strided_fn(
                    self.state.handle,
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
                    batch_count as i32,
                )
            };
            return check_cublas(status, "cublasSgemmStridedBatched(dev)");
        }
        for b in 0..batch_count {
            self.sgemm_dev(
                m,
                n,
                k,
                d_a + ((b * stride_a * 4) as u64),
                d_b + ((b * stride_b * 4) as u64),
                d_c + ((b * stride_c * 4) as u64),
                trans_a,
                trans_b,
                alpha,
                beta,
                DType::F32,
            )?;
        }
        Ok(())
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

    /// Alias for `rmsnorm_fwd_dev_f32` (VRAM-resident RMSNorm forward).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd_dev(
        &self,
        d_x: CuDevicePtr,
        d_gamma: CuDevicePtr,
        d_out: CuDevicePtr,
        d_rstd: CuDevicePtr,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> OchreResult<()> {
        self.rmsnorm_fwd_dev_f32(d_x, d_gamma, d_out, d_rstd, rows, cols, eps)
    }

    /// Alias for `rmsnorm_bwd_dev_f32` (VRAM-resident RMSNorm backward).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bwd_dev(
        &self,
        d_dy: CuDevicePtr,
        d_x: CuDevicePtr,
        d_gamma: CuDevicePtr,
        d_rstd: CuDevicePtr,
        d_dx: CuDevicePtr,
        d_dg: CuDevicePtr,
        rows: usize,
        cols: usize,
        need_dx: bool,
    ) -> OchreResult<()> {
        self.rmsnorm_bwd_dev_f32(d_dy, d_x, d_gamma, d_rstd, d_dx, d_dg, rows, cols, need_dx)
    }

    /// VRAM-resident embedding gather: `d_out[tok, :] = d_weight[d_ids[tok], :]`.
    pub fn embedding_gather_dev(
        &self,
        d_weight: CuDevicePtr,
        d_ids: CuDevicePtr,
        d_out: CuDevicePtr,
        num_tokens: usize,
        d_model: usize,
    ) -> OchreResult<()> {
        self.gather_rows_f32(d_weight, d_ids, d_out, num_tokens, d_model)
    }

    /// VRAM-resident embedding backward scatter-add: `d_dweight[d_ids[tok], :] += d_dy[tok, :]`.
    /// Caller should zero `d_dweight` before accumulation unless accumulating gradients.
    pub fn embedding_scatter_add_dev(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_ids: CuDevicePtr,
        mut d_dweight: CuDevicePtr,
        num_tokens: usize,
        d_model: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.embedding_scatter_add_f32;
        let mut nt_u64 = num_tokens as u64;
        let mut dm_u64 = d_model as u64;
        let mut params: [*mut c_void; 5] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_ids as *mut CuDevicePtr).cast(),
            (&mut d_dweight as *mut CuDevicePtr).cast(),
            (&mut nt_u64 as *mut u64).cast(),
            (&mut dm_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, num_tokens * d_model, 256, &mut params)
    }

    /// VRAM-resident transpose `[B, S, H, D]` -> `[B, H, S, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn bshd_to_bhsd_dev(
        &self,
        mut d_src: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.bshd_to_bhsd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = num_heads as u64;
        let mut d_u64 = head_dim as u64;
        let mut params: [*mut c_void; 6] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut d_u64 as *mut u64).cast(),
        ];
        launch_1d_async(
            self.rt,
            func,
            batch * seq_len * num_heads * head_dim,
            256,
            &mut params,
        )
    }

    /// VRAM-resident transpose `[B, H, S, D]` -> `[B, S, H, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn bhsd_to_bshd_dev(
        &self,
        mut d_src: CuDevicePtr,
        mut d_dst: CuDevicePtr,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.bhsd_to_bshd_f32;
        let mut b_u64 = batch as u64;
        let mut h_u64 = num_heads as u64;
        let mut s_u64 = seq_len as u64;
        let mut d_u64 = head_dim as u64;
        let mut params: [*mut c_void; 6] = [
            (&mut d_src as *mut CuDevicePtr).cast(),
            (&mut d_dst as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut d_u64 as *mut u64).cast(),
        ];
        launch_1d_async(
            self.rt,
            func,
            batch * num_heads * seq_len * head_dim,
            256,
            &mut params,
        )
    }

    /// In-place VRAM-resident RoPE forward (`inverse=false`) or inverse/backward (`inverse=true`)
    /// on `[batch, num_heads, seq_len, head_dim]` using precomputed `[max_seq_len, head_dim/2]` cos/sin tables.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        mut d_data: CuDevicePtr,
        mut d_cos: CuDevicePtr,
        mut d_sin: CuDevicePtr,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        offset: usize,
        inverse: bool,
    ) -> OchreResult<()> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA RoPE head_dim={head_dim} must be positive and even ( >_< )"
            )));
        }
        let half = head_dim / 2;
        let total_pairs = batch * num_heads * seq_len * half;
        if total_pairs == 0 {
            return Ok(());
        }
        let func = self.ptx()?.rope_fwd_bwd_f32;
        let mut tp_u64 = total_pairs as u64;
        let mut s_u64 = seq_len as u64;
        let mut hd_u64 = half as u64;
        let mut off_u64 = offset as u64;
        let mut inv_u32 = u32::from(inverse);
        let mut params: [*mut c_void; 8] = [
            (&mut d_data as *mut CuDevicePtr).cast(),
            (&mut d_cos as *mut CuDevicePtr).cast(),
            (&mut d_sin as *mut CuDevicePtr).cast(),
            (&mut tp_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut hd_u64 as *mut u64).cast(),
            (&mut off_u64 as *mut u64).cast(),
            (&mut inv_u32 as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, total_pairs, 256, &mut params)
    }

    /// VRAM-resident fused SwiGLU forward: `d_out[i] = silu(d_gate[i]) * d_up[i]`.
    pub fn swiglu_fwd(
        &self,
        mut d_gate: CuDevicePtr,
        mut d_up: CuDevicePtr,
        mut d_out: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.swiglu_fwd_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 4] = [
            (&mut d_gate as *mut CuDevicePtr).cast(),
            (&mut d_up as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// VRAM-resident fused SwiGLU backward computing both `d_dgate` and `d_dup` in one kernel.
    pub fn swiglu_bwd(
        &self,
        mut d_dy: CuDevicePtr,
        mut d_gate: CuDevicePtr,
        mut d_up: CuDevicePtr,
        mut d_dgate: CuDevicePtr,
        mut d_dup: CuDevicePtr,
        n: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.swiglu_bwd_f32;
        let mut n_u64 = n as u64;
        let mut params: [*mut c_void; 6] = [
            (&mut d_dy as *mut CuDevicePtr).cast(),
            (&mut d_gate as *mut CuDevicePtr).cast(),
            (&mut d_up as *mut CuDevicePtr).cast(),
            (&mut d_dgate as *mut CuDevicePtr).cast(),
            (&mut d_dup as *mut CuDevicePtr).cast(),
            (&mut n_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, n, 256, &mut params)
    }

    /// VRAM-resident GQA-native FlashSDPA forward supporting causal masking and `kv_offset`.
    /// Uses batched cuBLAS + `ochre_sdpa_causal_softmax_f32` (0 K/V duplication) when `q_len * kv_len <= 512 * 512`,
    /// and tiled online-softmax PTX `ochre_flash_sdpa_fwd_f32` for larger sequences.
    /// `d_q`/`d_out`: `[B, H, q_len, D]`, `d_k`/`d_v`: `[B, H_kv, kv_len, D]`, `d_lse`: `[B, H, q_len]`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_sdpa_fwd(
        &mut self,
        mut d_q: CuDevicePtr,
        mut d_k: CuDevicePtr,
        mut d_v: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_lse: CuDevicePtr,
        batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        causal: bool,
        kv_offset: usize,
    ) -> OchreResult<()> {
        if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || q_len == 0 {
            return Ok(());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA flash_sdpa_fwd num_heads ({num_heads}) not divisible by num_kv_heads ({num_kv_heads}) ( >_< )"
            )));
        }
        let total_q_rows = batch * num_heads * q_len;
        if kv_len == 0 {
            self.fill_f32(d_out, 0.0, total_q_rows * head_dim)?;
            self.fill_f32(d_lse, f32::NEG_INFINITY, total_q_rows)?;
            return Ok(());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        if q_len.saturating_mul(kv_len) <= 512 * 512 {
            let b_kv = batch * num_kv_heads;
            let g = num_heads / num_kv_heads;
            let m_q = g * q_len;
            let stride_q = m_q * head_dim;
            let stride_kv = kv_len * head_dim;
            let stride_s = m_q * kv_len;
            let d_s = self.alloc_f32(b_kv * stride_s)?;

            self.sgemm_strided_batched_dev(
                m_q,
                kv_len,
                head_dim,
                d_q,
                stride_q,
                d_k,
                stride_kv,
                d_s.ptr(),
                stride_s,
                b_kv,
                false,
                true,
                scale,
                0.0,
            )?;

            let softmax_fn = self.ptx()?.sdpa_causal_softmax_f32;
            let mut d_s_ptr = d_s.ptr();
            let mut tr_u64 = total_q_rows as u64;
            let mut ql_u64 = q_len as u64;
            let mut kvl_u64 = kv_len as u64;
            let mut causal_u32 = u32::from(causal);
            let mut kvo_u64 = kv_offset as u64;
            let mut sm_params: [*mut c_void; 7] = [
                (&mut d_s_ptr as *mut CuDevicePtr).cast(),
                (&mut d_lse as *mut CuDevicePtr).cast(),
                (&mut tr_u64 as *mut u64).cast(),
                (&mut ql_u64 as *mut u64).cast(),
                (&mut kvl_u64 as *mut u64).cast(),
                (&mut causal_u32 as *mut u32).cast(),
                (&mut kvo_u64 as *mut u64).cast(),
            ];
            launch_grid_async(
                self.rt,
                softmax_fn,
                total_q_rows as u32,
                128,
                &mut sm_params,
            )?;

            return self.sgemm_strided_batched_dev(
                m_q,
                head_dim,
                kv_len,
                d_s.ptr(),
                stride_s,
                d_v,
                stride_kv,
                d_out,
                stride_q,
                b_kv,
                false,
                false,
                1.0,
                0.0,
            );
        }

        let func = self.ptx()?.flash_sdpa_fwd_f32;
        let mut tqr_u64 = total_q_rows as u64;
        let mut nh_u64 = num_heads as u64;
        let mut nkvh_u64 = num_kv_heads as u64;
        let mut ql_u64 = q_len as u64;
        let mut kvl_u64 = kv_len as u64;
        let mut hd_u64 = head_dim as u64;
        let mut causal_u32 = u32::from(causal);
        let mut kvo_u64 = kv_offset as u64;
        let mut scale_arg = scale;
        let mut params: [*mut c_void; 14] = [
            (&mut d_q as *mut CuDevicePtr).cast(),
            (&mut d_k as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_lse as *mut CuDevicePtr).cast(),
            (&mut tqr_u64 as *mut u64).cast(),
            (&mut nh_u64 as *mut u64).cast(),
            (&mut nkvh_u64 as *mut u64).cast(),
            (&mut ql_u64 as *mut u64).cast(),
            (&mut kvl_u64 as *mut u64).cast(),
            (&mut hd_u64 as *mut u64).cast(),
            (&mut causal_u32 as *mut u32).cast(),
            (&mut kvo_u64 as *mut u64).cast(),
            (&mut scale_arg as *mut f32).cast(),
        ];
        launch_grid_async(self.rt, func, total_q_rows as u32, 64, &mut params)
    }

    /// VRAM-resident GQA-native FlashSDPA backward supporting causal masking and `kv_offset`.
    /// Uses batched cuBLAS + `ochre_sdpa_bwd_ds_f32` (0 K/V duplication) when `q_len * kv_len <= 512 * 512`,
    /// and tiled PTX `ochre_flash_sdpa_bwd_{dq,dkv}_f32` for larger sequences.
    /// `d_row_scratch` must have capacity for `batch * num_heads * q_len` `f32` elements on the tiled path.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_sdpa_bwd(
        &mut self,
        mut d_do: CuDevicePtr,
        mut d_q: CuDevicePtr,
        mut d_k: CuDevicePtr,
        mut d_v: CuDevicePtr,
        mut d_out: CuDevicePtr,
        mut d_lse: CuDevicePtr,
        mut d_row_scratch: CuDevicePtr,
        mut d_dq: CuDevicePtr,
        mut d_dk: CuDevicePtr,
        mut d_dv: CuDevicePtr,
        batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        causal: bool,
        kv_offset: usize,
    ) -> OchreResult<()> {
        if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Ok(());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA flash_sdpa_bwd num_heads ({num_heads}) not divisible by num_kv_heads ({num_kv_heads}) ( >_< )"
            )));
        }
        if q_len == 0 || kv_len == 0 {
            self.fill_f32(d_dq, 0.0, batch * num_heads * q_len * head_dim)?;
            self.fill_f32(d_dk, 0.0, batch * num_kv_heads * kv_len * head_dim)?;
            self.fill_f32(d_dv, 0.0, batch * num_kv_heads * kv_len * head_dim)?;
            return Ok(());
        }
        let total_q_rows = batch * num_heads * q_len;
        let mut scale = 1.0f32 / (head_dim as f32).sqrt();
        if q_len.saturating_mul(kv_len) <= 512 * 512 {
            let b_kv = batch * num_kv_heads;
            let g = num_heads / num_kv_heads;
            let m_q = g * q_len;
            let stride_q = m_q * head_dim;
            let stride_kv = kv_len * head_dim;
            let stride_s = m_q * kv_len;
            let d_s = self.alloc_f32(b_kv * stride_s)?;
            let d_dp = self.alloc_f32(b_kv * stride_s)?;

            self.sgemm_strided_batched_dev(
                m_q,
                kv_len,
                head_dim,
                d_q,
                stride_q,
                d_k,
                stride_kv,
                d_s.ptr(),
                stride_s,
                b_kv,
                false,
                true,
                scale,
                0.0,
            )?;
            self.sgemm_strided_batched_dev(
                m_q,
                kv_len,
                head_dim,
                d_do,
                stride_q,
                d_v,
                stride_kv,
                d_dp.ptr(),
                stride_s,
                b_kv,
                false,
                true,
                1.0,
                0.0,
            )?;

            let ds_fn = self.ptx()?.sdpa_bwd_ds_f32;
            let mut d_s_ptr = d_s.ptr();
            let mut d_dp_ptr = d_dp.ptr();
            let mut tr_u64 = total_q_rows as u64;
            let mut ql_u64 = q_len as u64;
            let mut kvl_u64 = kv_len as u64;
            let mut hd_u64 = head_dim as u64;
            let mut causal_u32 = u32::from(causal);
            let mut kvo_u64 = kv_offset as u64;
            let mut ds_params: [*mut c_void; 12] = [
                (&mut d_do as *mut CuDevicePtr).cast(),
                (&mut d_out as *mut CuDevicePtr).cast(),
                (&mut d_s_ptr as *mut CuDevicePtr).cast(),
                (&mut d_dp_ptr as *mut CuDevicePtr).cast(),
                (&mut d_lse as *mut CuDevicePtr).cast(),
                (&mut tr_u64 as *mut u64).cast(),
                (&mut ql_u64 as *mut u64).cast(),
                (&mut kvl_u64 as *mut u64).cast(),
                (&mut hd_u64 as *mut u64).cast(),
                (&mut causal_u32 as *mut u32).cast(),
                (&mut kvo_u64 as *mut u64).cast(),
                (&mut scale as *mut f32).cast(),
            ];
            launch_grid_async(self.rt, ds_fn, total_q_rows as u32, 128, &mut ds_params)?;

            self.sgemm_strided_batched_dev(
                kv_len,
                head_dim,
                m_q,
                d_s.ptr(),
                stride_s,
                d_do,
                stride_q,
                d_dv,
                stride_kv,
                b_kv,
                true,
                false,
                1.0,
                0.0,
            )?;
            self.sgemm_strided_batched_dev(
                m_q,
                head_dim,
                kv_len,
                d_dp.ptr(),
                stride_s,
                d_k,
                stride_kv,
                d_dq,
                stride_q,
                b_kv,
                false,
                false,
                1.0,
                0.0,
            )?;
            return self.sgemm_strided_batched_dev(
                kv_len,
                head_dim,
                m_q,
                d_dp.ptr(),
                stride_s,
                d_q,
                stride_q,
                d_dk,
                stride_kv,
                b_kv,
                true,
                false,
                1.0,
                0.0,
            );
        }

        let ptx = self.ptx()?;
        let total_kv_rows = batch * num_kv_heads * kv_len;
        let mut tqr_u64 = total_q_rows as u64;
        let mut tkvr_u64 = total_kv_rows as u64;
        let mut nh_u64 = num_heads as u64;
        let mut nkvh_u64 = num_kv_heads as u64;
        let mut ql_u64 = q_len as u64;
        let mut kvl_u64 = kv_len as u64;
        let mut hd_u64 = head_dim as u64;
        let mut causal_u32 = u32::from(causal);
        let mut kvo_u64 = kv_offset as u64;

        let mut dq_params: [*mut c_void; 17] = [
            (&mut d_do as *mut CuDevicePtr).cast(),
            (&mut d_q as *mut CuDevicePtr).cast(),
            (&mut d_k as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_lse as *mut CuDevicePtr).cast(),
            (&mut d_row_scratch as *mut CuDevicePtr).cast(),
            (&mut d_dq as *mut CuDevicePtr).cast(),
            (&mut tqr_u64 as *mut u64).cast(),
            (&mut nh_u64 as *mut u64).cast(),
            (&mut nkvh_u64 as *mut u64).cast(),
            (&mut ql_u64 as *mut u64).cast(),
            (&mut kvl_u64 as *mut u64).cast(),
            (&mut hd_u64 as *mut u64).cast(),
            (&mut causal_u32 as *mut u32).cast(),
            (&mut kvo_u64 as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        launch_grid_async(
            self.rt,
            ptx.flash_sdpa_bwd_dq_f32,
            total_q_rows as u32,
            64,
            &mut dq_params,
        )?;

        let mut dkv_params: [*mut c_void; 17] = [
            (&mut d_do as *mut CuDevicePtr).cast(),
            (&mut d_q as *mut CuDevicePtr).cast(),
            (&mut d_k as *mut CuDevicePtr).cast(),
            (&mut d_v as *mut CuDevicePtr).cast(),
            (&mut d_lse as *mut CuDevicePtr).cast(),
            (&mut d_row_scratch as *mut CuDevicePtr).cast(),
            (&mut d_dk as *mut CuDevicePtr).cast(),
            (&mut d_dv as *mut CuDevicePtr).cast(),
            (&mut tkvr_u64 as *mut u64).cast(),
            (&mut nh_u64 as *mut u64).cast(),
            (&mut nkvh_u64 as *mut u64).cast(),
            (&mut ql_u64 as *mut u64).cast(),
            (&mut kvl_u64 as *mut u64).cast(),
            (&mut hd_u64 as *mut u64).cast(),
            (&mut causal_u32 as *mut u32).cast(),
            (&mut kvo_u64 as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        launch_grid_async(
            self.rt,
            ptx.flash_sdpa_bwd_dkv_f32,
            total_kv_rows as u32,
            64,
            &mut dkv_params,
        )
    }

    /// VRAM-resident fused Elman RNN timestep forward.
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_step_fwd_dev(
        &self,
        mut d_x_proj: CuDevicePtr,
        mut d_h_proj: CuDevicePtr,
        mut d_bias: CuDevicePtr,
        mut d_h_cur: CuDevicePtr,
        mut d_layer_out: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
        is_relu: bool,
        has_bias: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.rnn_step_fwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut nonlin = u32::from(is_relu);
        let mut hb = u32::from(has_bias);
        let mut params: [*mut c_void; 13] = [
            (&mut d_x_proj as *mut CuDevicePtr).cast(),
            (&mut d_h_proj as *mut CuDevicePtr).cast(),
            (&mut d_bias as *mut CuDevicePtr).cast(),
            (&mut d_h_cur as *mut CuDevicePtr).cast(),
            (&mut d_layer_out as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
            (&mut nonlin as *mut u32).cast(),
            (&mut hb as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }

    /// VRAM-resident fused Elman RNN timestep backward.
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_step_bwd_dev(
        &self,
        mut d_out: CuDevicePtr,
        mut d_dh_next: CuDevicePtr,
        mut d_h_cur: CuDevicePtr,
        mut d_da_step: CuDevicePtr,
        mut d_da_all: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
        is_relu: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.rnn_step_bwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut nonlin = u32::from(is_relu);
        let mut params: [*mut c_void; 12] = [
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_dh_next as *mut CuDevicePtr).cast(),
            (&mut d_h_cur as *mut CuDevicePtr).cast(),
            (&mut d_da_step as *mut CuDevicePtr).cast(),
            (&mut d_da_all as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
            (&mut nonlin as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }

    /// VRAM-resident fused GRU 3-gate timestep forward.
    #[allow(clippy::too_many_arguments)]
    pub fn gru_step_fwd_dev(
        &self,
        mut d_gi_all: CuDevicePtr,
        mut d_gh: CuDevicePtr,
        mut d_biases: CuDevicePtr,
        mut d_h_prev: CuDevicePtr,
        mut d_h_cur: CuDevicePtr,
        mut d_layer_out: CuDevicePtr,
        mut d_r_out: CuDevicePtr,
        mut d_z_out: CuDevicePtr,
        mut d_n_out: CuDevicePtr,
        mut d_ghn_out: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
        save_cache: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.gru_step_fwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut sc = u32::from(save_cache);
        let mut params: [*mut c_void; 17] = [
            (&mut d_gi_all as *mut CuDevicePtr).cast(),
            (&mut d_gh as *mut CuDevicePtr).cast(),
            (&mut d_biases as *mut CuDevicePtr).cast(),
            (&mut d_h_prev as *mut CuDevicePtr).cast(),
            (&mut d_h_cur as *mut CuDevicePtr).cast(),
            (&mut d_layer_out as *mut CuDevicePtr).cast(),
            (&mut d_r_out as *mut CuDevicePtr).cast(),
            (&mut d_z_out as *mut CuDevicePtr).cast(),
            (&mut d_n_out as *mut CuDevicePtr).cast(),
            (&mut d_ghn_out as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
            (&mut sc as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }

    /// VRAM-resident fused GRU 3-gate timestep backward (updates `d_dh_next` in-place with `dh * z`).
    #[allow(clippy::too_many_arguments)]
    pub fn gru_step_bwd_dev(
        &self,
        mut d_out: CuDevicePtr,
        mut d_dh_next: CuDevicePtr,
        mut d_h_prev: CuDevicePtr,
        mut d_r: CuDevicePtr,
        mut d_z: CuDevicePtr,
        mut d_n: CuDevicePtr,
        mut d_ghn: CuDevicePtr,
        mut d_dgi_all: CuDevicePtr,
        mut d_dgh_step: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.gru_step_bwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut params: [*mut c_void; 15] = [
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_dh_next as *mut CuDevicePtr).cast(),
            (&mut d_h_prev as *mut CuDevicePtr).cast(),
            (&mut d_r as *mut CuDevicePtr).cast(),
            (&mut d_z as *mut CuDevicePtr).cast(),
            (&mut d_n as *mut CuDevicePtr).cast(),
            (&mut d_ghn as *mut CuDevicePtr).cast(),
            (&mut d_dgi_all as *mut CuDevicePtr).cast(),
            (&mut d_dgh_step as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }

    /// VRAM-resident fused LSTM 4-gate timestep forward.
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_step_fwd_dev(
        &self,
        mut d_gi_all: CuDevicePtr,
        mut d_gh: CuDevicePtr,
        mut d_b_total: CuDevicePtr,
        mut d_c_prev: CuDevicePtr,
        mut d_h_cur: CuDevicePtr,
        mut d_c_cur: CuDevicePtr,
        mut d_layer_out: CuDevicePtr,
        mut d_i_out: CuDevicePtr,
        mut d_f_out: CuDevicePtr,
        mut d_g_out: CuDevicePtr,
        mut d_o_out: CuDevicePtr,
        mut d_tc_out: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
        save_cache: bool,
    ) -> OchreResult<()> {
        let func = self.ptx()?.lstm_step_fwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut sc = u32::from(save_cache);
        let mut params: [*mut c_void; 19] = [
            (&mut d_gi_all as *mut CuDevicePtr).cast(),
            (&mut d_gh as *mut CuDevicePtr).cast(),
            (&mut d_b_total as *mut CuDevicePtr).cast(),
            (&mut d_c_prev as *mut CuDevicePtr).cast(),
            (&mut d_h_cur as *mut CuDevicePtr).cast(),
            (&mut d_c_cur as *mut CuDevicePtr).cast(),
            (&mut d_layer_out as *mut CuDevicePtr).cast(),
            (&mut d_i_out as *mut CuDevicePtr).cast(),
            (&mut d_f_out as *mut CuDevicePtr).cast(),
            (&mut d_g_out as *mut CuDevicePtr).cast(),
            (&mut d_o_out as *mut CuDevicePtr).cast(),
            (&mut d_tc_out as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
            (&mut sc as *mut u32).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }

    /// VRAM-resident fused LSTM 4-gate timestep backward (updates `d_dc_next` in-place with `dc * fv`).
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_step_bwd_dev(
        &self,
        mut d_out: CuDevicePtr,
        mut d_dh_next: CuDevicePtr,
        mut d_dc_next: CuDevicePtr,
        mut d_c_prev: CuDevicePtr,
        mut d_i: CuDevicePtr,
        mut d_f: CuDevicePtr,
        mut d_g: CuDevicePtr,
        mut d_o: CuDevicePtr,
        mut d_tc: CuDevicePtr,
        mut d_dg_all: CuDevicePtr,
        mut d_dg_step: CuDevicePtr,
        batch: usize,
        seq_len: usize,
        h: usize,
        out_dim: usize,
        t: usize,
        dir_off: usize,
    ) -> OchreResult<()> {
        let func = self.ptx()?.lstm_step_bwd_f32;
        let mut b_u64 = batch as u64;
        let mut s_u64 = seq_len as u64;
        let mut h_u64 = h as u64;
        let mut od_u64 = out_dim as u64;
        let mut t_u64 = t as u64;
        let mut doff_u64 = dir_off as u64;
        let mut params: [*mut c_void; 17] = [
            (&mut d_out as *mut CuDevicePtr).cast(),
            (&mut d_dh_next as *mut CuDevicePtr).cast(),
            (&mut d_dc_next as *mut CuDevicePtr).cast(),
            (&mut d_c_prev as *mut CuDevicePtr).cast(),
            (&mut d_i as *mut CuDevicePtr).cast(),
            (&mut d_f as *mut CuDevicePtr).cast(),
            (&mut d_g as *mut CuDevicePtr).cast(),
            (&mut d_o as *mut CuDevicePtr).cast(),
            (&mut d_tc as *mut CuDevicePtr).cast(),
            (&mut d_dg_all as *mut CuDevicePtr).cast(),
            (&mut d_dg_step as *mut CuDevicePtr).cast(),
            (&mut b_u64 as *mut u64).cast(),
            (&mut s_u64 as *mut u64).cast(),
            (&mut h_u64 as *mut u64).cast(),
            (&mut od_u64 as *mut u64).cast(),
            (&mut t_u64 as *mut u64).cast(),
            (&mut doff_u64 as *mut u64).cast(),
        ];
        launch_1d_async(self.rt, func, batch * h, 256, &mut params)
    }
}

// ============================================================================
// High-Level VRAM-Resident FlashAttention & RNN / GRU / LSTM Runners
// ============================================================================

/// Host wrapper for VRAM-resident CUDA FlashAttention forward.
/// Returns `None` if CUDA is unavailable (graceful CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn cuda_flash_attention_fwd(
    gpu_id: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>)>> {
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || q_len == 0 {
        return None;
    }
    with_vram_session(gpu_id, |sess| {
        let d_q = sess.upload_f32(q)?;
        let d_k = sess.upload_f32(k)?;
        let d_v = sess.upload_f32(v)?;
        let out_len = batch * num_heads * q_len * head_dim;
        let lse_len = batch * num_heads * q_len;
        let d_out = sess.alloc_f32(out_len)?;
        let d_lse = sess.alloc_f32(lse_len)?;
        sess.flash_sdpa_fwd(
            d_q.ptr(),
            d_k.ptr(),
            d_v.ptr(),
            d_out.ptr(),
            d_lse.ptr(),
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
        )?;
        let mut out = vec![0.0f32; out_len];
        let mut lse = vec![0.0f32; lse_len];
        sess.dtoh_f32(&mut out, &d_out)?;
        sess.dtoh_f32(&mut lse, &d_lse)?;
        Ok((out, lse))
    })
}

/// Host wrapper for VRAM-resident CUDA FlashAttention backward.
/// Returns `None` if CUDA is unavailable (graceful CPU fallback).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cuda_flash_attention_bwd(
    gpu_id: usize,
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    lse: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return None;
    }
    with_vram_session(gpu_id, |sess| {
        let d_do = sess.upload_f32(d_out)?;
        let d_q = sess.upload_f32(q)?;
        let d_k = sess.upload_f32(k)?;
        let d_v = sess.upload_f32(v)?;
        let d_o = sess.upload_f32(out)?;
        let d_lse = sess.upload_f32(lse)?;
        let d_row = sess.alloc_f32(batch * num_heads * q_len)?;
        let d_dq = sess.alloc_f32(q.len())?;
        let d_dk = sess.alloc_f32(k.len())?;
        let d_dv = sess.alloc_f32(v.len())?;
        sess.flash_sdpa_bwd(
            d_do.ptr(),
            d_q.ptr(),
            d_k.ptr(),
            d_v.ptr(),
            d_o.ptr(),
            d_lse.ptr(),
            d_row.ptr(),
            d_dq.ptr(),
            d_dk.ptr(),
            d_dv.ptr(),
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
        )?;
        let mut dq = vec![0.0f32; q.len()];
        let mut dk = vec![0.0f32; k.len()];
        let mut dv = vec![0.0f32; v.len()];
        sess.dtoh_f32(&mut dq, &d_dq)?;
        sess.dtoh_f32(&mut dk, &d_dk)?;
        sess.dtoh_f32(&mut dv, &d_dv)?;
        Ok((dq, dk, dv))
    })
}

/// Device-resident FlashAttention activation cache kept in VRAM between
/// `cuda_flash_attention_fwd_cached` and `cuda_flash_attention_bwd_cached`.
pub struct CudaFlashCache {
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
    d_q: CudaDeviceBuf,
    d_k: CudaDeviceBuf,
    d_v: CudaDeviceBuf,
    d_out: CudaDeviceBuf,
    d_lse: CudaDeviceBuf,
}

/// Host wrapper for VRAM-resident CUDA FlashAttention forward that keeps `Q, K, V, O, LSE`
/// resident in VRAM inside `CudaFlashCache` for zero-reupload autograd backward.
#[allow(clippy::too_many_arguments)]
pub fn cuda_flash_attention_fwd_cached(
    gpu_id: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    kv_offset: usize,
) -> Option<OchreResult<(Vec<f32>, CudaFlashCache)>> {
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || q_len == 0 {
        return None;
    }
    with_vram_session(gpu_id, |sess| {
        let d_q = sess.upload_f32(q)?;
        let d_k = sess.upload_f32(k)?;
        let d_v = sess.upload_f32(v)?;
        let out_len = batch * num_heads * q_len * head_dim;
        let lse_len = batch * num_heads * q_len;
        let d_out = sess.alloc_f32(out_len)?;
        let d_lse = sess.alloc_f32(lse_len)?;
        sess.flash_sdpa_fwd(
            d_q.ptr(),
            d_k.ptr(),
            d_v.ptr(),
            d_out.ptr(),
            d_lse.ptr(),
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            kv_offset,
        )?;
        let mut out = vec![0.0f32; out_len];
        sess.dtoh_f32(&mut out, &d_out)?;
        Ok((
            out,
            CudaFlashCache {
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                causal,
                kv_offset,
                d_q,
                d_k,
                d_v,
                d_out,
                d_lse,
            },
        ))
    })
}

/// Host wrapper for VRAM-resident CUDA FlashAttention backward consuming `CudaFlashCache`
/// without re-uploading `Q, K, V, O, LSE`.
#[allow(clippy::type_complexity)]
pub fn cuda_flash_attention_bwd_cached(
    gpu_id: usize,
    cache: &CudaFlashCache,
    d_out: &[f32],
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if cache.batch == 0 || cache.num_heads == 0 || cache.num_kv_heads == 0 || cache.head_dim == 0 {
        return None;
    }
    with_vram_session(gpu_id, |sess| {
        if d_out.len() != cache.d_out.len_f32() {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA flash attention cached backward d_out len {} != expected {} ( >_< )",
                d_out.len(),
                cache.d_out.len_f32()
            )));
        }
        let d_do = sess.upload_f32(d_out)?;
        let d_row_ptr = if cache.q_len.saturating_mul(cache.kv_len) > 512 * 512 {
            Some(sess.alloc_f32(cache.batch * cache.num_heads * cache.q_len)?)
        } else {
            None
        };
        let d_dq = sess.alloc_f32(cache.d_q.len_f32())?;
        let d_dk = sess.alloc_f32(cache.d_k.len_f32())?;
        let d_dv = sess.alloc_f32(cache.d_v.len_f32())?;
        sess.flash_sdpa_bwd(
            d_do.ptr(),
            cache.d_q.ptr(),
            cache.d_k.ptr(),
            cache.d_v.ptr(),
            cache.d_out.ptr(),
            cache.d_lse.ptr(),
            d_row_ptr.as_ref().map_or(0, CudaDeviceBuf::ptr),
            d_dq.ptr(),
            d_dk.ptr(),
            d_dv.ptr(),
            cache.batch,
            cache.num_heads,
            cache.num_kv_heads,
            cache.q_len,
            cache.kv_len,
            cache.head_dim,
            cache.causal,
            cache.kv_offset,
        )?;
        let mut dq = vec![0.0f32; cache.d_q.len_f32()];
        let mut dk = vec![0.0f32; cache.d_k.len_f32()];
        let mut dv = vec![0.0f32; cache.d_v.len_f32()];
        sess.dtoh_f32(&mut dq, &d_dq)?;
        sess.dtoh_f32(&mut dk, &d_dk)?;
        sess.dtoh_f32(&mut dv, &d_dv)?;
        Ok((dq, dk, dv))
    })
}

/// Gradient buffers downloaded once at the end of VRAM-resident RNN/GRU/LSTM backward.
#[derive(Clone, Debug)]
pub struct CudaSeqGradOutput {
    pub d_input: Vec<f32>,
    pub d_h0: Vec<f32>,
    pub d_c0: Option<Vec<f32>>,
    pub d_weight_ih: Vec<Vec<f32>>,
    pub d_weight_hh: Vec<Vec<f32>>,
    pub d_bias_ih: Vec<Option<Vec<f32>>>,
    pub d_bias_hh: Vec<Option<Vec<f32>>>,
}

/// Device-resident activation & weight cache kept in VRAM between `cuda_rnn_forward` and `cuda_rnn_backward`.
pub struct CudaRnnCache {
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    is_relu: bool,
    dtype: DType,
    has_bias_ih: Vec<bool>,
    has_bias_hh: Vec<bool>,
    w_ih_lens: Vec<usize>,
    w_hh_lens: Vec<usize>,
    d_weight_ih: Vec<CudaDeviceBuf>,
    d_weight_hh: Vec<CudaDeviceBuf>,
    d_layer_inputs: Vec<CudaDeviceBuf>,
    /// Per unit: `[(seq_len + 1) * B * H]` holding `[h_0, h_1, ..., h_T]` in VRAM.
    d_unit_all_h: Vec<CudaDeviceBuf>,
}

/// Device-resident activation & weight cache kept in VRAM between `cuda_gru_forward` and `cuda_gru_backward`.
pub struct CudaGruCache {
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    has_bias_ih: Vec<bool>,
    has_bias_hh: Vec<bool>,
    w_ih_lens: Vec<usize>,
    w_hh_lens: Vec<usize>,
    d_weight_ih: Vec<CudaDeviceBuf>,
    d_weight_hh: Vec<CudaDeviceBuf>,
    d_layer_inputs: Vec<CudaDeviceBuf>,
    d_unit_all_h: Vec<CudaDeviceBuf>,
    d_unit_r: Vec<CudaDeviceBuf>,
    d_unit_z: Vec<CudaDeviceBuf>,
    d_unit_n: Vec<CudaDeviceBuf>,
    d_unit_ghn: Vec<CudaDeviceBuf>,
}

/// Device-resident activation & weight cache kept in VRAM between `cuda_lstm_forward` and `cuda_lstm_backward`.
pub struct CudaLstmCache {
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    has_bias_ih: Vec<bool>,
    has_bias_hh: Vec<bool>,
    w_ih_lens: Vec<usize>,
    w_hh_lens: Vec<usize>,
    d_weight_ih: Vec<CudaDeviceBuf>,
    d_weight_hh: Vec<CudaDeviceBuf>,
    d_layer_inputs: Vec<CudaDeviceBuf>,
    d_unit_all_h: Vec<CudaDeviceBuf>,
    d_unit_all_c: Vec<CudaDeviceBuf>,
    d_unit_i: Vec<CudaDeviceBuf>,
    d_unit_f: Vec<CudaDeviceBuf>,
    d_unit_g: Vec<CudaDeviceBuf>,
    d_unit_o: Vec<CudaDeviceBuf>,
    d_unit_tc: Vec<CudaDeviceBuf>,
}

/// 100% VRAM-resident multi-layer (optionally bidirectional) Elman RNN forward pass.
/// Uploads `x`, `h0`, and weights once, runs all layers and timesteps in VRAM with zero per-step host copies,
/// and downloads only `(output, h_n)`. When `save_cache == true`, keeps activations in VRAM inside `CudaRnnCache`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cuda_rnn_forward(
    gpu_id: usize,
    x: &[f32],
    h0: Option<&[f32]>,
    weight_ih: &[&[f32]],
    weight_hh: &[&[f32]],
    bias_ih: &[Option<&[f32]>],
    bias_hh: &[Option<&[f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    is_relu: bool,
    dtype: DType,
    save_cache: bool,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Option<CudaRnnCache>)>> {
    let dirs = if bidirectional { 2 } else { 1 };
    let units = num_layers * dirs;
    let h = hidden_size;
    let bh = batch * h;

    with_vram_session(gpu_id, |sess| {
        if x.len() != batch * seq_len * input_size {
            return Err(OchreError::InvalidParameter(
                "CUDA RNN input length mismatch ( >_< )".to_string(),
            ));
        }
        if let Some(h_init) = h0
            && h_init.len() != units * bh
        {
            return Err(OchreError::InvalidParameter(
                "CUDA RNN h0 length mismatch ( >_< )".to_string(),
            ));
        }
        if weight_ih.len() != units
            || weight_hh.len() != units
            || bias_ih.len() != units
            || bias_hh.len() != units
        {
            return Err(OchreError::InvalidParameter(
                "CUDA RNN weight/bias unit count mismatch ( >_< )".to_string(),
            ));
        }

        let mut d_w_ih = Vec::with_capacity(units);
        let mut d_w_hh = Vec::with_capacity(units);
        let mut d_b_total = Vec::with_capacity(units);
        let mut has_bias_any = Vec::with_capacity(units);
        let mut has_bias_ih = Vec::with_capacity(units);
        let mut has_bias_hh = Vec::with_capacity(units);
        let mut w_ih_lens = Vec::with_capacity(units);
        let mut w_hh_lens = Vec::with_capacity(units);

        for u in 0..units {
            w_ih_lens.push(weight_ih[u].len());
            w_hh_lens.push(weight_hh[u].len());
            d_w_ih.push(sess.upload_f32(weight_ih[u])?);
            d_w_hh.push(sess.upload_f32(weight_hh[u])?);
            has_bias_ih.push(bias_ih[u].is_some());
            has_bias_hh.push(bias_hh[u].is_some());
            let hb = bias_ih[u].is_some() || bias_hh[u].is_some();
            has_bias_any.push(hb);
            let mut bt = vec![0.0f32; h];
            if let Some(b) = bias_ih[u] {
                for (d, &s) in bt.iter_mut().zip(b.iter()) {
                    *d += s;
                }
            }
            if let Some(b) = bias_hh[u] {
                for (d, &s) in bt.iter_mut().zip(b.iter()) {
                    *d += s;
                }
            }
            d_b_total.push(sess.upload_f32(&bt)?);
        }

        let d_h0 = match h0 {
            Some(hi) => Some(sess.upload_f32(hi)?),
            None => None,
        };
        let mut d_cur_input = sess.upload_f32(x)?;
        let d_hn = sess.alloc_f32(units * bh)?;
        let d_x_proj = sess.alloc_f32(batch * seq_len * h)?;
        let d_h_proj = sess.alloc_f32(bh)?;

        // Ping-pong buffers for eval mode (!save_cache)
        let d_ping = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_pong = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };

        let mut d_layer_inputs = Vec::with_capacity(if save_cache { num_layers } else { 0 });
        let mut d_unit_all_h = Vec::with_capacity(if save_cache { units } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 { input_size } else { h * dirs };
            let out_dim = h * dirs;
            let d_layer_out = sess.alloc_f32(batch * seq_len * out_dim)?;

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.sgemm_dev(
                    batch * seq_len,
                    h,
                    in_dim,
                    d_cur_input.ptr(),
                    d_w_ih[u].ptr(),
                    d_x_proj.ptr(),
                    false,
                    true,
                    1.0,
                    0.0,
                    dtype,
                )?;

                if save_cache {
                    let d_all_h = sess.alloc_f32((seq_len + 1) * bh)?;
                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), d_all_h.ptr(), bh)?;
                    } else {
                        sess.fill_f32(d_all_h.ptr(), 0.0, bh)?;
                    }

                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        let p_prev = d_all_h.ptr_offset_f32(step * bh);
                        let p_cur = d_all_h.ptr_offset_f32((step + 1) * bh);
                        sess.sgemm_dev(
                            batch,
                            h,
                            h,
                            p_prev,
                            d_w_hh[u].ptr(),
                            d_h_proj.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.rnn_step_fwd_dev(
                            d_x_proj.ptr(),
                            d_h_proj.ptr(),
                            d_b_total[u].ptr(),
                            p_cur,
                            d_layer_out.ptr(),
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            is_relu,
                            has_bias_any[u],
                        )?;
                    }
                    sess.copy_f32(
                        d_all_h.ptr_offset_f32(seq_len * bh),
                        d_hn.ptr_offset_f32(u * bh),
                        bh,
                    )?;
                    d_unit_all_h.push(d_all_h);
                } else {
                    let (mut p_prev, mut p_cur) = match (&d_ping, &d_pong) {
                        (Some(a), Some(b)) => (a.ptr(), b.ptr()),
                        _ => unreachable!(),
                    };
                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), p_prev, bh)?;
                    } else {
                        sess.fill_f32(p_prev, 0.0, bh)?;
                    }
                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        sess.sgemm_dev(
                            batch,
                            h,
                            h,
                            p_prev,
                            d_w_hh[u].ptr(),
                            d_h_proj.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.rnn_step_fwd_dev(
                            d_x_proj.ptr(),
                            d_h_proj.ptr(),
                            d_b_total[u].ptr(),
                            p_cur,
                            d_layer_out.ptr(),
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            is_relu,
                            has_bias_any[u],
                        )?;
                        std::mem::swap(&mut p_prev, &mut p_cur);
                    }
                    sess.copy_f32(p_prev, d_hn.ptr_offset_f32(u * bh), bh)?;
                }
            }

            if save_cache {
                d_layer_inputs.push(d_cur_input);
            }
            d_cur_input = d_layer_out;
        }

        let mut out = vec![0.0f32; batch * seq_len * h * dirs];
        let mut hn = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut out, &d_cur_input)?;
        sess.dtoh_f32(&mut hn, &d_hn)?;

        let cache = if save_cache {
            Some(CudaRnnCache {
                batch,
                seq_len,
                input_size,
                hidden_size,
                num_layers,
                bidirectional,
                is_relu,
                dtype,
                has_bias_ih,
                has_bias_hh,
                w_ih_lens,
                w_hh_lens,
                d_weight_ih: d_w_ih,
                d_weight_hh: d_w_hh,
                d_layer_inputs,
                d_unit_all_h,
            })
        } else {
            None
        };
        Ok((out, hn, cache))
    })
}

/// 100% VRAM-resident BPTT backward pass for Elman RNN using `CudaRnnCache`.
pub fn cuda_rnn_backward(
    gpu_id: usize,
    cache: &CudaRnnCache,
    d_output: &[f32],
    d_hn: &[f32],
) -> Option<OchreResult<CudaSeqGradOutput>> {
    let dirs = if cache.bidirectional { 2 } else { 1 };
    let units = cache.num_layers * dirs;
    let h = cache.hidden_size;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;

    with_vram_session(gpu_id, |sess| {
        let mut d_cur_out = sess.upload_f32(d_output)?;
        let d_hn_in = sess.upload_f32(d_hn)?;
        let d_h0_out = sess.alloc_f32(units * bh)?;
        let d_dh_next = sess.alloc_f32(bh)?;
        let d_da_all = sess.alloc_f32(batch * seq_len * h)?;
        let d_da_steps = sess.alloc_f32(seq_len * bh)?;
        let d_db_tmp = sess.alloc_f32(h)?;

        let mut d_weight_ih: Vec<Vec<f32>> = cache
            .w_ih_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_weight_hh: Vec<Vec<f32>> = cache
            .w_hh_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_bias_ih: Vec<Option<Vec<f32>>> = vec![None; units];
        let mut d_bias_hh: Vec<Option<Vec<f32>>> = vec![None; units];

        for layer in (0..cache.num_layers).rev() {
            let in_dim = if layer == 0 {
                cache.input_size
            } else {
                h * dirs
            };
            let out_dim = h * dirs;
            let d_layer_in = sess.alloc_f32(batch * seq_len * in_dim)?;
            let d_x_layer = &cache.d_layer_inputs[layer];

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.copy_f32(d_hn_in.ptr_offset_f32(u * bh), d_dh_next.ptr(), bh)?;

                for step in (0..seq_len).rev() {
                    let t = if dir == 0 { step } else { seq_len - 1 - step };
                    let p_h_cur = cache.d_unit_all_h[u].ptr_offset_f32((step + 1) * bh);
                    let p_da_step = d_da_steps.ptr_offset_f32(step * bh);
                    sess.rnn_step_bwd_dev(
                        d_cur_out.ptr(),
                        d_dh_next.ptr(),
                        p_h_cur,
                        p_da_step,
                        d_da_all.ptr(),
                        batch,
                        seq_len,
                        h,
                        out_dim,
                        t,
                        dir * h,
                        cache.is_relu,
                    )?;
                    sess.sgemm_dev(
                        batch,
                        h,
                        h,
                        p_da_step,
                        cache.d_weight_hh[u].ptr(),
                        d_dh_next.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        cache.dtype,
                    )?;
                }

                sess.copy_f32(d_dh_next.ptr(), d_h0_out.ptr_offset_f32(u * bh), bh)?;

                if cache.has_bias_ih[u] || cache.has_bias_hh[u] {
                    sess.sum_rows_f32(d_da_steps.ptr(), d_db_tmp.ptr(), seq_len * batch, h)?;
                    let mut db_host = vec![0.0f32; h];
                    sess.dtoh_f32(&mut db_host, &d_db_tmp)?;
                    if cache.has_bias_ih[u] {
                        d_bias_ih[u] = Some(db_host.clone());
                    }
                    if cache.has_bias_hh[u] {
                        d_bias_hh[u] = Some(db_host);
                    }
                }

                let d_dw_hh = sess.alloc_f32(cache.w_hh_lens[u])?;
                sess.sgemm_dev(
                    h,
                    h,
                    seq_len * batch,
                    d_da_steps.ptr(),
                    cache.d_unit_all_h[u].ptr(),
                    d_dw_hh.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                let d_dw_ih = sess.alloc_f32(cache.w_ih_lens[u])?;
                sess.sgemm_dev(
                    h,
                    in_dim,
                    batch * seq_len,
                    d_da_all.ptr(),
                    d_x_layer.ptr(),
                    d_dw_ih.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                sess.sgemm_dev(
                    batch * seq_len,
                    in_dim,
                    h,
                    d_da_all.ptr(),
                    cache.d_weight_ih[u].ptr(),
                    d_layer_in.ptr(),
                    false,
                    false,
                    1.0,
                    if dir == 0 { 0.0 } else { 1.0 },
                    cache.dtype,
                )?;

                sess.dtoh_f32(&mut d_weight_hh[u], &d_dw_hh)?;
                sess.dtoh_f32(&mut d_weight_ih[u], &d_dw_ih)?;
            }

            d_cur_out = d_layer_in;
        }

        let mut d_input = vec![0.0f32; batch * seq_len * cache.input_size];
        let mut d_h0 = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut d_input, &d_cur_out)?;
        sess.dtoh_f32(&mut d_h0, &d_h0_out)?;

        Ok(CudaSeqGradOutput {
            d_input,
            d_h0,
            d_c0: None,
            d_weight_ih,
            d_weight_hh,
            d_bias_ih,
            d_bias_hh,
        })
    })
}

/// 100% VRAM-resident multi-layer (optionally bidirectional) GRU forward pass.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cuda_gru_forward(
    gpu_id: usize,
    x: &[f32],
    h0: Option<&[f32]>,
    weight_ih: &[&[f32]],
    weight_hh: &[&[f32]],
    bias_ih: &[Option<&[f32]>],
    bias_hh: &[Option<&[f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    save_cache: bool,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Option<CudaGruCache>)>> {
    let dirs = if bidirectional { 2 } else { 1 };
    let units = num_layers * dirs;
    let h = hidden_size;
    let h3 = 3 * h;
    let bh = batch * h;

    with_vram_session(gpu_id, |sess| {
        if x.len() != batch * seq_len * input_size {
            return Err(OchreError::InvalidParameter(
                "CUDA GRU input length mismatch ( >_< )".to_string(),
            ));
        }
        if let Some(h_init) = h0
            && h_init.len() != units * bh
        {
            return Err(OchreError::InvalidParameter(
                "CUDA GRU h0 length mismatch ( >_< )".to_string(),
            ));
        }
        if weight_ih.len() != units
            || weight_hh.len() != units
            || bias_ih.len() != units
            || bias_hh.len() != units
        {
            return Err(OchreError::InvalidParameter(
                "CUDA GRU weight/bias unit count mismatch ( >_< )".to_string(),
            ));
        }

        let mut d_w_ih = Vec::with_capacity(units);
        let mut d_w_hh = Vec::with_capacity(units);
        let mut d_biases = Vec::with_capacity(units);
        let mut has_bias_ih = Vec::with_capacity(units);
        let mut has_bias_hh = Vec::with_capacity(units);
        let mut w_ih_lens = Vec::with_capacity(units);
        let mut w_hh_lens = Vec::with_capacity(units);

        for u in 0..units {
            w_ih_lens.push(weight_ih[u].len());
            w_hh_lens.push(weight_hh[u].len());
            d_w_ih.push(sess.upload_f32(weight_ih[u])?);
            d_w_hh.push(sess.upload_f32(weight_hh[u])?);
            has_bias_ih.push(bias_ih[u].is_some());
            has_bias_hh.push(bias_hh[u].is_some());
            // Pack [br_sum (H) | bz_sum (H) | bi_n (H) | bh_n (H)]
            let mut packed = vec![0.0f32; 4 * h];
            if let Some(v) = bias_ih[u] {
                for j in 0..h {
                    packed[j] += v[j];
                    packed[h + j] += v[h + j];
                    packed[2 * h + j] = v[2 * h + j];
                }
            }
            if let Some(v) = bias_hh[u] {
                for j in 0..h {
                    packed[j] += v[j];
                    packed[h + j] += v[h + j];
                    packed[3 * h + j] = v[2 * h + j];
                }
            }
            d_biases.push(sess.upload_f32(&packed)?);
        }

        let d_h0 = match h0 {
            Some(hi) => Some(sess.upload_f32(hi)?),
            None => None,
        };
        let mut d_cur_input = sess.upload_f32(x)?;
        let d_hn = sess.alloc_f32(units * bh)?;
        let d_gi_all = sess.alloc_f32(batch * seq_len * h3)?;
        let d_gh = sess.alloc_f32(batch * h3)?;

        let d_ping = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_pong = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };

        let mut d_layer_inputs = Vec::with_capacity(if save_cache { num_layers } else { 0 });
        let mut d_unit_all_h = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_r = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_z = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_n = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_ghn = Vec::with_capacity(if save_cache { units } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 { input_size } else { h * dirs };
            let out_dim = h * dirs;
            let d_layer_out = sess.alloc_f32(batch * seq_len * out_dim)?;

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.sgemm_dev(
                    batch * seq_len,
                    h3,
                    in_dim,
                    d_cur_input.ptr(),
                    d_w_ih[u].ptr(),
                    d_gi_all.ptr(),
                    false,
                    true,
                    1.0,
                    0.0,
                    dtype,
                )?;

                if save_cache {
                    let d_all_h = sess.alloc_f32((seq_len + 1) * bh)?;
                    let d_r = sess.alloc_f32(seq_len * bh)?;
                    let d_z = sess.alloc_f32(seq_len * bh)?;
                    let d_n = sess.alloc_f32(seq_len * bh)?;
                    let d_ghn = sess.alloc_f32(seq_len * bh)?;

                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), d_all_h.ptr(), bh)?;
                    } else {
                        sess.fill_f32(d_all_h.ptr(), 0.0, bh)?;
                    }

                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        let p_prev = d_all_h.ptr_offset_f32(step * bh);
                        let p_cur = d_all_h.ptr_offset_f32((step + 1) * bh);
                        sess.sgemm_dev(
                            batch,
                            h3,
                            h,
                            p_prev,
                            d_w_hh[u].ptr(),
                            d_gh.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.gru_step_fwd_dev(
                            d_gi_all.ptr(),
                            d_gh.ptr(),
                            d_biases[u].ptr(),
                            p_prev,
                            p_cur,
                            d_layer_out.ptr(),
                            d_r.ptr_offset_f32(step * bh),
                            d_z.ptr_offset_f32(step * bh),
                            d_n.ptr_offset_f32(step * bh),
                            d_ghn.ptr_offset_f32(step * bh),
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            true,
                        )?;
                    }

                    sess.copy_f32(
                        d_all_h.ptr_offset_f32(seq_len * bh),
                        d_hn.ptr_offset_f32(u * bh),
                        bh,
                    )?;
                    d_unit_all_h.push(d_all_h);
                    d_unit_r.push(d_r);
                    d_unit_z.push(d_z);
                    d_unit_n.push(d_n);
                    d_unit_ghn.push(d_ghn);
                } else {
                    let (mut p_prev, mut p_cur) = match (&d_ping, &d_pong) {
                        (Some(a), Some(b)) => (a.ptr(), b.ptr()),
                        _ => unreachable!(),
                    };
                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), p_prev, bh)?;
                    } else {
                        sess.fill_f32(p_prev, 0.0, bh)?;
                    }
                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        sess.sgemm_dev(
                            batch,
                            h3,
                            h,
                            p_prev,
                            d_w_hh[u].ptr(),
                            d_gh.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.gru_step_fwd_dev(
                            d_gi_all.ptr(),
                            d_gh.ptr(),
                            d_biases[u].ptr(),
                            p_prev,
                            p_cur,
                            d_layer_out.ptr(),
                            0,
                            0,
                            0,
                            0,
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            false,
                        )?;
                        std::mem::swap(&mut p_prev, &mut p_cur);
                    }
                    sess.copy_f32(p_prev, d_hn.ptr_offset_f32(u * bh), bh)?;
                }
            }

            if save_cache {
                d_layer_inputs.push(d_cur_input);
            }
            d_cur_input = d_layer_out;
        }

        let mut out = vec![0.0f32; batch * seq_len * h * dirs];
        let mut hn = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut out, &d_cur_input)?;
        sess.dtoh_f32(&mut hn, &d_hn)?;

        let cache = if save_cache {
            Some(CudaGruCache {
                batch,
                seq_len,
                input_size,
                hidden_size,
                num_layers,
                bidirectional,
                dtype,
                has_bias_ih,
                has_bias_hh,
                w_ih_lens,
                w_hh_lens,
                d_weight_ih: d_w_ih,
                d_weight_hh: d_w_hh,
                d_layer_inputs,
                d_unit_all_h,
                d_unit_r,
                d_unit_z,
                d_unit_n,
                d_unit_ghn,
            })
        } else {
            None
        };
        Ok((out, hn, cache))
    })
}

/// 100% VRAM-resident BPTT backward pass for GRU using `CudaGruCache`.
pub fn cuda_gru_backward(
    gpu_id: usize,
    cache: &CudaGruCache,
    d_output: &[f32],
    d_hn: &[f32],
) -> Option<OchreResult<CudaSeqGradOutput>> {
    let dirs = if cache.bidirectional { 2 } else { 1 };
    let units = cache.num_layers * dirs;
    let h = cache.hidden_size;
    let h3 = 3 * h;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;
    let bh3 = batch * h3;

    with_vram_session(gpu_id, |sess| {
        let mut d_cur_out = sess.upload_f32(d_output)?;
        let d_hn_in = sess.upload_f32(d_hn)?;
        let d_h0_out = sess.alloc_f32(units * bh)?;
        let d_dh_next = sess.alloc_f32(bh)?;
        let d_dgi_all = sess.alloc_f32(batch * seq_len * h3)?;
        let d_dgh_steps = sess.alloc_f32(seq_len * bh3)?;
        let d_db_tmp = sess.alloc_f32(h3)?;

        let mut d_weight_ih: Vec<Vec<f32>> = cache
            .w_ih_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_weight_hh: Vec<Vec<f32>> = cache
            .w_hh_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_bias_ih: Vec<Option<Vec<f32>>> = vec![None; units];
        let mut d_bias_hh: Vec<Option<Vec<f32>>> = vec![None; units];

        for layer in (0..cache.num_layers).rev() {
            let in_dim = if layer == 0 {
                cache.input_size
            } else {
                h * dirs
            };
            let out_dim = h * dirs;
            let d_layer_in = sess.alloc_f32(batch * seq_len * in_dim)?;
            let d_x_layer = &cache.d_layer_inputs[layer];

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.copy_f32(d_hn_in.ptr_offset_f32(u * bh), d_dh_next.ptr(), bh)?;

                for step in (0..seq_len).rev() {
                    let t = if dir == 0 { step } else { seq_len - 1 - step };
                    let off = step * bh;
                    let p_dgh = d_dgh_steps.ptr_offset_f32(step * bh3);
                    sess.gru_step_bwd_dev(
                        d_cur_out.ptr(),
                        d_dh_next.ptr(),
                        cache.d_unit_all_h[u].ptr_offset_f32(off),
                        cache.d_unit_r[u].ptr_offset_f32(off),
                        cache.d_unit_z[u].ptr_offset_f32(off),
                        cache.d_unit_n[u].ptr_offset_f32(off),
                        cache.d_unit_ghn[u].ptr_offset_f32(off),
                        d_dgi_all.ptr(),
                        p_dgh,
                        batch,
                        seq_len,
                        h,
                        out_dim,
                        t,
                        dir * h,
                    )?;
                    // gru_step_bwd_dev already wrote `dh * z` into d_dh_next, so accumulate beta = 1.0!
                    sess.sgemm_dev(
                        batch,
                        h,
                        h3,
                        p_dgh,
                        cache.d_weight_hh[u].ptr(),
                        d_dh_next.ptr(),
                        false,
                        false,
                        1.0,
                        1.0,
                        cache.dtype,
                    )?;
                }

                sess.copy_f32(d_dh_next.ptr(), d_h0_out.ptr_offset_f32(u * bh), bh)?;

                if cache.has_bias_ih[u] {
                    sess.sum_rows_f32(d_dgi_all.ptr(), d_db_tmp.ptr(), batch * seq_len, h3)?;
                    let mut db_host = vec![0.0f32; h3];
                    sess.dtoh_f32(&mut db_host, &d_db_tmp)?;
                    d_bias_ih[u] = Some(db_host);
                }
                if cache.has_bias_hh[u] {
                    sess.sum_rows_f32(d_dgh_steps.ptr(), d_db_tmp.ptr(), seq_len * batch, h3)?;
                    let mut db_host = vec![0.0f32; h3];
                    sess.dtoh_f32(&mut db_host, &d_db_tmp)?;
                    d_bias_hh[u] = Some(db_host);
                }

                let d_dw_hh = sess.alloc_f32(cache.w_hh_lens[u])?;
                sess.sgemm_dev(
                    h3,
                    h,
                    seq_len * batch,
                    d_dgh_steps.ptr(),
                    cache.d_unit_all_h[u].ptr(),
                    d_dw_hh.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                let d_dw_ih = sess.alloc_f32(cache.w_ih_lens[u])?;
                sess.sgemm_dev(
                    h3,
                    in_dim,
                    batch * seq_len,
                    d_dgi_all.ptr(),
                    d_x_layer.ptr(),
                    d_dw_ih.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                sess.sgemm_dev(
                    batch * seq_len,
                    in_dim,
                    h3,
                    d_dgi_all.ptr(),
                    cache.d_weight_ih[u].ptr(),
                    d_layer_in.ptr(),
                    false,
                    false,
                    1.0,
                    if dir == 0 { 0.0 } else { 1.0 },
                    cache.dtype,
                )?;

                sess.dtoh_f32(&mut d_weight_hh[u], &d_dw_hh)?;
                sess.dtoh_f32(&mut d_weight_ih[u], &d_dw_ih)?;
            }

            d_cur_out = d_layer_in;
        }

        let mut d_input = vec![0.0f32; batch * seq_len * cache.input_size];
        let mut d_h0 = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut d_input, &d_cur_out)?;
        sess.dtoh_f32(&mut d_h0, &d_h0_out)?;

        Ok(CudaSeqGradOutput {
            d_input,
            d_h0,
            d_c0: None,
            d_weight_ih,
            d_weight_hh,
            d_bias_ih,
            d_bias_hh,
        })
    })
}

/// 100% VRAM-resident multi-layer (optionally bidirectional) LSTM forward pass.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cuda_lstm_forward(
    gpu_id: usize,
    x: &[f32],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
    weight_ih: &[&[f32]],
    weight_hh: &[&[f32]],
    bias_ih: &[Option<&[f32]>],
    bias_hh: &[Option<&[f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    save_cache: bool,
) -> Option<OchreResult<(Vec<f32>, Vec<f32>, Vec<f32>, Option<CudaLstmCache>)>> {
    let dirs = if bidirectional { 2 } else { 1 };
    let units = num_layers * dirs;
    let h = hidden_size;
    let h4 = 4 * h;
    let bh = batch * h;

    with_vram_session(gpu_id, |sess| {
        if x.len() != batch * seq_len * input_size {
            return Err(OchreError::InvalidParameter(
                "CUDA LSTM input length mismatch ( >_< )".to_string(),
            ));
        }
        if let Some(h_init) = h0
            && h_init.len() != units * bh
        {
            return Err(OchreError::InvalidParameter(
                "CUDA LSTM h0 length mismatch ( >_< )".to_string(),
            ));
        }
        if let Some(c_init) = c0
            && c_init.len() != units * bh
        {
            return Err(OchreError::InvalidParameter(
                "CUDA LSTM c0 length mismatch ( >_< )".to_string(),
            ));
        }
        if weight_ih.len() != units
            || weight_hh.len() != units
            || bias_ih.len() != units
            || bias_hh.len() != units
        {
            return Err(OchreError::InvalidParameter(
                "CUDA LSTM weight/bias unit count mismatch ( >_< )".to_string(),
            ));
        }

        let mut d_w_ih = Vec::with_capacity(units);
        let mut d_w_hh = Vec::with_capacity(units);
        let mut d_b_total = Vec::with_capacity(units);
        let mut has_bias_ih = Vec::with_capacity(units);
        let mut has_bias_hh = Vec::with_capacity(units);
        let mut w_ih_lens = Vec::with_capacity(units);
        let mut w_hh_lens = Vec::with_capacity(units);

        for u in 0..units {
            w_ih_lens.push(weight_ih[u].len());
            w_hh_lens.push(weight_hh[u].len());
            d_w_ih.push(sess.upload_f32(weight_ih[u])?);
            d_w_hh.push(sess.upload_f32(weight_hh[u])?);
            has_bias_ih.push(bias_ih[u].is_some());
            has_bias_hh.push(bias_hh[u].is_some());
            let mut bt = vec![0.0f32; h4];
            if let Some(v) = bias_ih[u] {
                for (d, &s) in bt.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }
            if let Some(v) = bias_hh[u] {
                for (d, &s) in bt.iter_mut().zip(v.iter()) {
                    *d += s;
                }
            }
            d_b_total.push(sess.upload_f32(&bt)?);
        }

        let d_h0 = match h0 {
            Some(hi) => Some(sess.upload_f32(hi)?),
            None => None,
        };
        let d_c0 = match c0 {
            Some(ci) => Some(sess.upload_f32(ci)?),
            None => None,
        };
        let mut d_cur_input = sess.upload_f32(x)?;
        let d_hn = sess.alloc_f32(units * bh)?;
        let d_cn = sess.alloc_f32(units * bh)?;
        let d_gi_all = sess.alloc_f32(batch * seq_len * h4)?;
        let d_gh = sess.alloc_f32(batch * h4)?;

        let d_h_ping = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_h_pong = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_c_ping = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_c_pong = if !save_cache {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };

        let mut d_layer_inputs = Vec::with_capacity(if save_cache { num_layers } else { 0 });
        let mut d_unit_all_h = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_all_c = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_i = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_f = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_g = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_o = Vec::with_capacity(if save_cache { units } else { 0 });
        let mut d_unit_tc = Vec::with_capacity(if save_cache { units } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 { input_size } else { h * dirs };
            let out_dim = h * dirs;
            let d_layer_out = sess.alloc_f32(batch * seq_len * out_dim)?;

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.sgemm_dev(
                    batch * seq_len,
                    h4,
                    in_dim,
                    d_cur_input.ptr(),
                    d_w_ih[u].ptr(),
                    d_gi_all.ptr(),
                    false,
                    true,
                    1.0,
                    0.0,
                    dtype,
                )?;

                if save_cache {
                    let d_all_h = sess.alloc_f32((seq_len + 1) * bh)?;
                    let d_all_c = sess.alloc_f32((seq_len + 1) * bh)?;
                    let d_i = sess.alloc_f32(seq_len * bh)?;
                    let d_f = sess.alloc_f32(seq_len * bh)?;
                    let d_g = sess.alloc_f32(seq_len * bh)?;
                    let d_o = sess.alloc_f32(seq_len * bh)?;
                    let d_tc = sess.alloc_f32(seq_len * bh)?;

                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), d_all_h.ptr(), bh)?;
                    } else {
                        sess.fill_f32(d_all_h.ptr(), 0.0, bh)?;
                    }
                    if let Some(ref c0_buf) = d_c0 {
                        sess.copy_f32(c0_buf.ptr_offset_f32(u * bh), d_all_c.ptr(), bh)?;
                    } else {
                        sess.fill_f32(d_all_c.ptr(), 0.0, bh)?;
                    }

                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        let p_hp = d_all_h.ptr_offset_f32(step * bh);
                        let p_hc = d_all_h.ptr_offset_f32((step + 1) * bh);
                        let p_cp = d_all_c.ptr_offset_f32(step * bh);
                        let p_cc = d_all_c.ptr_offset_f32((step + 1) * bh);
                        sess.sgemm_dev(
                            batch,
                            h4,
                            h,
                            p_hp,
                            d_w_hh[u].ptr(),
                            d_gh.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.lstm_step_fwd_dev(
                            d_gi_all.ptr(),
                            d_gh.ptr(),
                            d_b_total[u].ptr(),
                            p_cp,
                            p_hc,
                            p_cc,
                            d_layer_out.ptr(),
                            d_i.ptr_offset_f32(step * bh),
                            d_f.ptr_offset_f32(step * bh),
                            d_g.ptr_offset_f32(step * bh),
                            d_o.ptr_offset_f32(step * bh),
                            d_tc.ptr_offset_f32(step * bh),
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            true,
                        )?;
                    }

                    sess.copy_f32(
                        d_all_h.ptr_offset_f32(seq_len * bh),
                        d_hn.ptr_offset_f32(u * bh),
                        bh,
                    )?;
                    sess.copy_f32(
                        d_all_c.ptr_offset_f32(seq_len * bh),
                        d_cn.ptr_offset_f32(u * bh),
                        bh,
                    )?;
                    d_unit_all_h.push(d_all_h);
                    d_unit_all_c.push(d_all_c);
                    d_unit_i.push(d_i);
                    d_unit_f.push(d_f);
                    d_unit_g.push(d_g);
                    d_unit_o.push(d_o);
                    d_unit_tc.push(d_tc);
                } else {
                    let (mut p_hp, mut p_hc) = match (&d_h_ping, &d_h_pong) {
                        (Some(a), Some(b)) => (a.ptr(), b.ptr()),
                        _ => unreachable!(),
                    };
                    let (mut p_cp, mut p_cc) = match (&d_c_ping, &d_c_pong) {
                        (Some(a), Some(b)) => (a.ptr(), b.ptr()),
                        _ => unreachable!(),
                    };
                    if let Some(ref h0_buf) = d_h0 {
                        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), p_hp, bh)?;
                    } else {
                        sess.fill_f32(p_hp, 0.0, bh)?;
                    }
                    if let Some(ref c0_buf) = d_c0 {
                        sess.copy_f32(c0_buf.ptr_offset_f32(u * bh), p_cp, bh)?;
                    } else {
                        sess.fill_f32(p_cp, 0.0, bh)?;
                    }

                    for step in 0..seq_len {
                        let t = if dir == 0 { step } else { seq_len - 1 - step };
                        sess.sgemm_dev(
                            batch,
                            h4,
                            h,
                            p_hp,
                            d_w_hh[u].ptr(),
                            d_gh.ptr(),
                            false,
                            true,
                            1.0,
                            0.0,
                            dtype,
                        )?;
                        sess.lstm_step_fwd_dev(
                            d_gi_all.ptr(),
                            d_gh.ptr(),
                            d_b_total[u].ptr(),
                            p_cp,
                            p_hc,
                            p_cc,
                            d_layer_out.ptr(),
                            0,
                            0,
                            0,
                            0,
                            0,
                            batch,
                            seq_len,
                            h,
                            out_dim,
                            t,
                            dir * h,
                            false,
                        )?;
                        std::mem::swap(&mut p_hp, &mut p_hc);
                        std::mem::swap(&mut p_cp, &mut p_cc);
                    }
                    sess.copy_f32(p_hp, d_hn.ptr_offset_f32(u * bh), bh)?;
                    sess.copy_f32(p_cp, d_cn.ptr_offset_f32(u * bh), bh)?;
                }
            }

            if save_cache {
                d_layer_inputs.push(d_cur_input);
            }
            d_cur_input = d_layer_out;
        }

        let mut out = vec![0.0f32; batch * seq_len * h * dirs];
        let mut hn = vec![0.0f32; units * bh];
        let mut cn = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut out, &d_cur_input)?;
        sess.dtoh_f32(&mut hn, &d_hn)?;
        sess.dtoh_f32(&mut cn, &d_cn)?;

        let cache = if save_cache {
            Some(CudaLstmCache {
                batch,
                seq_len,
                input_size,
                hidden_size,
                num_layers,
                bidirectional,
                dtype,
                has_bias_ih,
                has_bias_hh,
                w_ih_lens,
                w_hh_lens,
                d_weight_ih: d_w_ih,
                d_weight_hh: d_w_hh,
                d_layer_inputs,
                d_unit_all_h,
                d_unit_all_c,
                d_unit_i,
                d_unit_f,
                d_unit_g,
                d_unit_o,
                d_unit_tc,
            })
        } else {
            None
        };
        Ok((out, hn, cn, cache))
    })
}

/// 100% VRAM-resident BPTT backward pass for LSTM using `CudaLstmCache`.
pub fn cuda_lstm_backward(
    gpu_id: usize,
    cache: &CudaLstmCache,
    d_output: &[f32],
    d_hn: &[f32],
    d_cn: &[f32],
) -> Option<OchreResult<CudaSeqGradOutput>> {
    let dirs = if cache.bidirectional { 2 } else { 1 };
    let units = cache.num_layers * dirs;
    let h = cache.hidden_size;
    let h4 = 4 * h;
    let batch = cache.batch;
    let seq_len = cache.seq_len;
    let bh = batch * h;
    let bh4 = batch * h4;

    with_vram_session(gpu_id, |sess| {
        let mut d_cur_out = sess.upload_f32(d_output)?;
        let d_hn_in = sess.upload_f32(d_hn)?;
        let d_cn_in = sess.upload_f32(d_cn)?;
        let d_h0_out = sess.alloc_f32(units * bh)?;
        let d_c0_out = sess.alloc_f32(units * bh)?;
        let d_dh_next = sess.alloc_f32(bh)?;
        let d_dc_next = sess.alloc_f32(bh)?;
        let d_dg_all = sess.alloc_f32(batch * seq_len * h4)?;
        let d_dg_steps = sess.alloc_f32(seq_len * bh4)?;
        let d_db_tmp = sess.alloc_f32(h4)?;

        let mut d_weight_ih: Vec<Vec<f32>> = cache
            .w_ih_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_weight_hh: Vec<Vec<f32>> = cache
            .w_hh_lens
            .iter()
            .map(|&len| vec![0.0f32; len])
            .collect();
        let mut d_bias_ih: Vec<Option<Vec<f32>>> = vec![None; units];
        let mut d_bias_hh: Vec<Option<Vec<f32>>> = vec![None; units];

        for layer in (0..cache.num_layers).rev() {
            let in_dim = if layer == 0 {
                cache.input_size
            } else {
                h * dirs
            };
            let out_dim = h * dirs;
            let d_layer_in = sess.alloc_f32(batch * seq_len * in_dim)?;
            let d_x_layer = &cache.d_layer_inputs[layer];

            for dir in 0..dirs {
                let u = layer * dirs + dir;
                sess.copy_f32(d_hn_in.ptr_offset_f32(u * bh), d_dh_next.ptr(), bh)?;
                sess.copy_f32(d_cn_in.ptr_offset_f32(u * bh), d_dc_next.ptr(), bh)?;

                for step in (0..seq_len).rev() {
                    let t = if dir == 0 { step } else { seq_len - 1 - step };
                    let off = step * bh;
                    let p_dg_step = d_dg_steps.ptr_offset_f32(step * bh4);
                    sess.lstm_step_bwd_dev(
                        d_cur_out.ptr(),
                        d_dh_next.ptr(),
                        d_dc_next.ptr(),
                        cache.d_unit_all_c[u].ptr_offset_f32(off),
                        cache.d_unit_i[u].ptr_offset_f32(off),
                        cache.d_unit_f[u].ptr_offset_f32(off),
                        cache.d_unit_g[u].ptr_offset_f32(off),
                        cache.d_unit_o[u].ptr_offset_f32(off),
                        cache.d_unit_tc[u].ptr_offset_f32(off),
                        d_dg_all.ptr(),
                        p_dg_step,
                        batch,
                        seq_len,
                        h,
                        out_dim,
                        t,
                        dir * h,
                    )?;
                    sess.sgemm_dev(
                        batch,
                        h,
                        h4,
                        p_dg_step,
                        cache.d_weight_hh[u].ptr(),
                        d_dh_next.ptr(),
                        false,
                        false,
                        1.0,
                        0.0,
                        cache.dtype,
                    )?;
                }

                sess.copy_f32(d_dh_next.ptr(), d_h0_out.ptr_offset_f32(u * bh), bh)?;
                sess.copy_f32(d_dc_next.ptr(), d_c0_out.ptr_offset_f32(u * bh), bh)?;

                if cache.has_bias_ih[u] || cache.has_bias_hh[u] {
                    sess.sum_rows_f32(d_dg_steps.ptr(), d_db_tmp.ptr(), seq_len * batch, h4)?;
                    let mut db_host = vec![0.0f32; h4];
                    sess.dtoh_f32(&mut db_host, &d_db_tmp)?;
                    if cache.has_bias_ih[u] {
                        d_bias_ih[u] = Some(db_host.clone());
                    }
                    if cache.has_bias_hh[u] {
                        d_bias_hh[u] = Some(db_host);
                    }
                }

                let d_dw_hh = sess.alloc_f32(cache.w_hh_lens[u])?;
                sess.sgemm_dev(
                    h4,
                    h,
                    seq_len * batch,
                    d_dg_steps.ptr(),
                    cache.d_unit_all_h[u].ptr(),
                    d_dw_hh.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                let d_dw_ih = sess.alloc_f32(cache.w_ih_lens[u])?;
                sess.sgemm_dev(
                    h4,
                    in_dim,
                    batch * seq_len,
                    d_dg_all.ptr(),
                    d_x_layer.ptr(),
                    d_dw_ih.ptr(),
                    true,
                    false,
                    1.0,
                    0.0,
                    cache.dtype,
                )?;
                sess.sgemm_dev(
                    batch * seq_len,
                    in_dim,
                    h4,
                    d_dg_all.ptr(),
                    cache.d_weight_ih[u].ptr(),
                    d_layer_in.ptr(),
                    false,
                    false,
                    1.0,
                    if dir == 0 { 0.0 } else { 1.0 },
                    cache.dtype,
                )?;

                sess.dtoh_f32(&mut d_weight_hh[u], &d_dw_hh)?;
                sess.dtoh_f32(&mut d_weight_ih[u], &d_dw_ih)?;
            }

            d_cur_out = d_layer_in;
        }

        let mut d_input = vec![0.0f32; batch * seq_len * cache.input_size];
        let mut d_h0 = vec![0.0f32; units * bh];
        let mut d_c0 = vec![0.0f32; units * bh];
        sess.dtoh_f32(&mut d_input, &d_cur_out)?;
        sess.dtoh_f32(&mut d_h0, &d_h0_out)?;
        sess.dtoh_f32(&mut d_c0, &d_c0_out)?;

        Ok(CudaSeqGradOutput {
            d_input,
            d_h0,
            d_c0: Some(d_c0),
            d_weight_ih,
            d_weight_hh,
            d_bias_ih,
            d_bias_hh,
        })
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeqCellKind {
    Rnn { is_relu: bool },
    Gru,
    Lstm,
}

impl SeqCellKind {
    #[inline]
    fn gate_mult(self) -> usize {
        match self {
            Self::Rnn { .. } => 1,
            Self::Gru => 3,
            Self::Lstm => 4,
        }
    }

    #[inline]
    fn packed_bias_elems(self, h: usize) -> usize {
        match self {
            Self::Rnn { .. } => h,
            Self::Gru => 4 * h,
            Self::Lstm => 4 * h,
        }
    }

    #[inline]
    fn label(self) -> &'static str {
        match self {
            Self::Rnn { .. } => "RNN",
            Self::Gru => "GRU",
            Self::Lstm => "LSTM",
        }
    }
}

struct SeqTrainUnitState {
    w_ih_idx: usize,
    w_hh_idx: usize,
    b_ih_idx: Option<usize>,
    b_hh_idx: Option<usize>,
    has_bias_any: bool,
    d_b_packed: CudaDeviceBuf,
    d_all_h: CudaDeviceBuf,
    d_all_c: Option<CudaDeviceBuf>,
    gate_bufs: Vec<CudaDeviceBuf>,
}

#[allow(clippy::too_many_arguments)]
fn validate_seq_train_args(
    kind: SeqCellKind,
    x: &[f32],
    target: &[f32],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
    weight_ih: &[&mut [f32]],
    weight_hh: &[&mut [f32]],
    bias_ih: &[Option<&mut [f32]>],
    bias_hh: &[Option<&mut [f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    grad_clip: Option<f32>,
) -> OchreResult<()> {
    let label = kind.label();
    if batch == 0 || seq_len == 0 || input_size == 0 || hidden_size == 0 || num_layers == 0 {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} train_steps dimensions must all be > 0 ( >_< )"
        )));
    }
    if steps == 0 {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} train_steps steps must be > 0 ( >_< )"
        )));
    }
    if !lr.is_finite() || lr < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} train_steps lr={lr} must be finite and >= 0 ( >_< )"
        )));
    }
    if !weight_decay.is_finite() || weight_decay < 0.0 {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} train_steps weight_decay={weight_decay} must be finite and >= 0 ( >_< )"
        )));
    }
    if let Some(gc) = grad_clip
        && (!gc.is_finite() || gc < 0.0)
    {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} train_steps grad_clip={gc} must be finite and >= 0 ( >_< )"
        )));
    }
    let dirs = if bidirectional { 2 } else { 1 };
    let units = num_layers * dirs;
    let gates = kind.gate_mult() * hidden_size;
    let bh = batch * hidden_size;
    if x.len() != batch * seq_len * input_size {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} input length {} != expected {} ( >_< )",
            x.len(),
            batch * seq_len * input_size
        )));
    }
    if target.len() != batch * seq_len * hidden_size * dirs {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} target length {} != expected {} ( >_< )",
            target.len(),
            batch * seq_len * hidden_size * dirs
        )));
    }
    if let Some(h_init) = h0
        && h_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} h0 length mismatch ( >_< )"
        )));
    }
    if let Some(c_init) = c0
        && c_init.len() != units * bh
    {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} c0 length mismatch ( >_< )"
        )));
    }
    if weight_ih.len() != units
        || weight_hh.len() != units
        || bias_ih.len() != units
        || bias_hh.len() != units
    {
        return Err(OchreError::InvalidParameter(format!(
            "CUDA {label} weight/bias unit count mismatch ( >_< )"
        )));
    }
    for u in 0..units {
        let layer = u / dirs;
        let in_dim = if layer == 0 {
            input_size
        } else {
            hidden_size * dirs
        };
        if weight_ih[u].len() != gates * in_dim || weight_hh[u].len() != gates * hidden_size {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA {label} unit {u} weight size mismatch ( >_< )"
            )));
        }
        if let Some(b) = bias_ih[u].as_deref()
            && b.len() != gates
        {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA {label} unit {u} bias_ih size mismatch ( >_< )"
            )));
        }
        if let Some(b) = bias_hh[u].as_deref()
            && b.len() != gates
        {
            return Err(OchreError::InvalidParameter(format!(
                "CUDA {label} unit {u} bias_hh size mismatch ( >_< )"
            )));
        }
    }
    Ok(())
}

fn pack_seq_unit_bias_dev(
    sess: &CudaVramSession<'_>,
    kind: SeqCellKind,
    h: usize,
    b_ih_ptr: Option<CuDevicePtr>,
    b_hh_ptr: Option<CuDevicePtr>,
    d_b_packed: CuDevicePtr,
) -> OchreResult<()> {
    match kind {
        SeqCellKind::Rnn { .. } | SeqCellKind::Lstm => {
            let gates = kind.gate_mult() * h;
            match (b_ih_ptr, b_hh_ptr) {
                (Some(bih), Some(bhh)) => {
                    sess.copy_f32(bih, d_b_packed, gates)?;
                    sess.add_inplace_dev_f32(d_b_packed, bhh, gates)?;
                }
                (Some(bih), None) => sess.copy_f32(bih, d_b_packed, gates)?,
                (None, Some(bhh)) => sess.copy_f32(bhh, d_b_packed, gates)?,
                (None, None) => {}
            }
        }
        SeqCellKind::Gru => {
            sess.fill_f32(d_b_packed, 0.0, 4 * h)?;
            if let Some(bih) = b_ih_ptr {
                sess.copy_f32(bih, d_b_packed, 2 * h)?;
                sess.copy_f32(
                    bih + ((2 * h * 4) as u64),
                    d_b_packed + ((2 * h * 4) as u64),
                    h,
                )?;
            }
            if let Some(bhh) = b_hh_ptr {
                sess.add_inplace_dev_f32(d_b_packed, bhh, 2 * h)?;
                sess.copy_f32(
                    bhh + ((2 * h * 4) as u64),
                    d_b_packed + ((3 * h * 4) as u64),
                    h,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn seq_train_unit_fwd_dev(
    sess: &mut CudaVramSession<'_>,
    kind: SeqCellKind,
    unit: &SeqTrainUnitState,
    d_params: &[CudaDeviceBuf],
    d_x_layer: CuDevicePtr,
    d_layer_out: CuDevicePtr,
    d_h0: Option<&CudaDeviceBuf>,
    d_c0: Option<&CudaDeviceBuf>,
    d_gi_all: CuDevicePtr,
    d_gh: CuDevicePtr,
    batch: usize,
    seq_len: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
    u: usize,
    dir: usize,
    dtype: DType,
) -> OchreResult<()> {
    let gates = kind.gate_mult() * h;
    let bh = batch * h;
    let d_w_ih = d_params[unit.w_ih_idx].ptr();
    let d_w_hh = d_params[unit.w_hh_idx].ptr();

    sess.sgemm_dev(
        batch * seq_len,
        gates,
        in_dim,
        d_x_layer,
        d_w_ih,
        d_gi_all,
        false,
        true,
        1.0,
        0.0,
        dtype,
    )?;

    if let Some(h0_buf) = d_h0 {
        sess.copy_f32(h0_buf.ptr_offset_f32(u * bh), unit.d_all_h.ptr(), bh)?;
    } else {
        sess.fill_f32(unit.d_all_h.ptr(), 0.0, bh)?;
    }
    if let Some(ref all_c) = unit.d_all_c {
        if let Some(c0_buf) = d_c0 {
            sess.copy_f32(c0_buf.ptr_offset_f32(u * bh), all_c.ptr(), bh)?;
        } else {
            sess.fill_f32(all_c.ptr(), 0.0, bh)?;
        }
    }

    for step in 0..seq_len {
        let t = if dir == 0 { step } else { seq_len - 1 - step };
        let p_hp = unit.d_all_h.ptr_offset_f32(step * bh);
        let p_hc = unit.d_all_h.ptr_offset_f32((step + 1) * bh);
        sess.sgemm_dev(
            batch, gates, h, p_hp, d_w_hh, d_gh, false, true, 1.0, 0.0, dtype,
        )?;
        match kind {
            SeqCellKind::Rnn { is_relu } => sess.rnn_step_fwd_dev(
                d_gi_all,
                d_gh,
                unit.d_b_packed.ptr(),
                p_hc,
                d_layer_out,
                batch,
                seq_len,
                h,
                out_dim,
                t,
                dir * h,
                is_relu,
                unit.has_bias_any,
            )?,
            SeqCellKind::Gru => sess.gru_step_fwd_dev(
                d_gi_all,
                d_gh,
                unit.d_b_packed.ptr(),
                p_hp,
                p_hc,
                d_layer_out,
                unit.gate_bufs[0].ptr_offset_f32(step * bh),
                unit.gate_bufs[1].ptr_offset_f32(step * bh),
                unit.gate_bufs[2].ptr_offset_f32(step * bh),
                unit.gate_bufs[3].ptr_offset_f32(step * bh),
                batch,
                seq_len,
                h,
                out_dim,
                t,
                dir * h,
                true,
            )?,
            SeqCellKind::Lstm => {
                let all_c = unit.d_all_c.as_ref().ok_or_else(|| {
                    OchreError::DeviceError("Missing LSTM cell state buffer ( >_< )".to_string())
                })?;
                sess.lstm_step_fwd_dev(
                    d_gi_all,
                    d_gh,
                    unit.d_b_packed.ptr(),
                    all_c.ptr_offset_f32(step * bh),
                    p_hc,
                    all_c.ptr_offset_f32((step + 1) * bh),
                    d_layer_out,
                    unit.gate_bufs[0].ptr_offset_f32(step * bh),
                    unit.gate_bufs[1].ptr_offset_f32(step * bh),
                    unit.gate_bufs[2].ptr_offset_f32(step * bh),
                    unit.gate_bufs[3].ptr_offset_f32(step * bh),
                    unit.gate_bufs[4].ptr_offset_f32(step * bh),
                    batch,
                    seq_len,
                    h,
                    out_dim,
                    t,
                    dir * h,
                    true,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn seq_train_unit_bwd_dev(
    sess: &mut CudaVramSession<'_>,
    kind: SeqCellKind,
    unit: &SeqTrainUnitState,
    d_params: &[CudaDeviceBuf],
    d_grads: &[CudaDeviceBuf],
    d_cur_out: CuDevicePtr,
    d_x_layer: CuDevicePtr,
    d_layer_in: Option<CuDevicePtr>,
    d_dh_next: CuDevicePtr,
    d_dc_next: Option<CuDevicePtr>,
    d_dgi_all: CuDevicePtr,
    d_dgh_steps: CuDevicePtr,
    batch: usize,
    seq_len: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
    dir: usize,
    dtype: DType,
) -> OchreResult<()> {
    let gates = kind.gate_mult() * h;
    let bh = batch * h;
    let b_gates = batch * gates;
    let d_w_ih = d_params[unit.w_ih_idx].ptr();
    let d_w_hh = d_params[unit.w_hh_idx].ptr();

    sess.fill_f32(d_dh_next, 0.0, bh)?;
    if let Some(dc_next) = d_dc_next {
        sess.fill_f32(dc_next, 0.0, bh)?;
    }

    for step in (0..seq_len).rev() {
        let t = if dir == 0 { step } else { seq_len - 1 - step };
        let p_dgh = d_dgh_steps + ((step * b_gates * 4) as u64);
        match kind {
            SeqCellKind::Rnn { is_relu } => {
                sess.rnn_step_bwd_dev(
                    d_cur_out,
                    d_dh_next,
                    unit.d_all_h.ptr_offset_f32((step + 1) * bh),
                    p_dgh,
                    d_dgi_all,
                    batch,
                    seq_len,
                    h,
                    out_dim,
                    t,
                    dir * h,
                    is_relu,
                )?;
                sess.sgemm_dev(
                    batch, h, h, p_dgh, d_w_hh, d_dh_next, false, false, 1.0, 0.0, dtype,
                )?;
            }
            SeqCellKind::Gru => {
                let off = step * bh;
                sess.gru_step_bwd_dev(
                    d_cur_out,
                    d_dh_next,
                    unit.d_all_h.ptr_offset_f32(off),
                    unit.gate_bufs[0].ptr_offset_f32(off),
                    unit.gate_bufs[1].ptr_offset_f32(off),
                    unit.gate_bufs[2].ptr_offset_f32(off),
                    unit.gate_bufs[3].ptr_offset_f32(off),
                    d_dgi_all,
                    p_dgh,
                    batch,
                    seq_len,
                    h,
                    out_dim,
                    t,
                    dir * h,
                )?;
                sess.sgemm_dev(
                    batch, h, gates, p_dgh, d_w_hh, d_dh_next, false, false, 1.0, 1.0, dtype,
                )?;
            }
            SeqCellKind::Lstm => {
                let off = step * bh;
                let all_c = unit.d_all_c.as_ref().ok_or_else(|| {
                    OchreError::DeviceError("Missing LSTM cell state buffer ( >_< )".to_string())
                })?;
                let dc_next = d_dc_next.ok_or_else(|| {
                    OchreError::DeviceError("Missing LSTM dc_next buffer ( >_< )".to_string())
                })?;
                sess.lstm_step_bwd_dev(
                    d_cur_out,
                    d_dh_next,
                    dc_next,
                    all_c.ptr_offset_f32(off),
                    unit.gate_bufs[0].ptr_offset_f32(off),
                    unit.gate_bufs[1].ptr_offset_f32(off),
                    unit.gate_bufs[2].ptr_offset_f32(off),
                    unit.gate_bufs[3].ptr_offset_f32(off),
                    unit.gate_bufs[4].ptr_offset_f32(off),
                    d_dgi_all,
                    p_dgh,
                    batch,
                    seq_len,
                    h,
                    out_dim,
                    t,
                    dir * h,
                )?;
                sess.sgemm_dev(
                    batch, h, gates, p_dgh, d_w_hh, d_dh_next, false, false, 1.0, 0.0, dtype,
                )?;
            }
        }
    }

    match kind {
        SeqCellKind::Gru => {
            if let Some(idx) = unit.b_ih_idx {
                sess.sum_rows_f32(d_dgi_all, d_grads[idx].ptr(), batch * seq_len, gates)?;
            }
            if let Some(idx) = unit.b_hh_idx {
                sess.sum_rows_f32(d_dgh_steps, d_grads[idx].ptr(), seq_len * batch, gates)?;
            }
        }
        SeqCellKind::Rnn { .. } | SeqCellKind::Lstm => {
            if let Some(ih_idx) = unit.b_ih_idx {
                sess.sum_rows_f32(d_dgh_steps, d_grads[ih_idx].ptr(), seq_len * batch, gates)?;
                if let Some(hh_idx) = unit.b_hh_idx {
                    sess.copy_f32(d_grads[ih_idx].ptr(), d_grads[hh_idx].ptr(), gates)?;
                }
            } else if let Some(hh_idx) = unit.b_hh_idx {
                sess.sum_rows_f32(d_dgh_steps, d_grads[hh_idx].ptr(), seq_len * batch, gates)?;
            }
        }
    }

    sess.sgemm_dev(
        gates,
        h,
        seq_len * batch,
        d_dgh_steps,
        unit.d_all_h.ptr(),
        d_grads[unit.w_hh_idx].ptr(),
        true,
        false,
        1.0,
        0.0,
        dtype,
    )?;
    sess.sgemm_dev(
        gates,
        in_dim,
        batch * seq_len,
        d_dgi_all,
        d_x_layer,
        d_grads[unit.w_ih_idx].ptr(),
        true,
        false,
        1.0,
        0.0,
        dtype,
    )?;
    if let Some(layer_in) = d_layer_in {
        sess.sgemm_dev(
            batch * seq_len,
            in_dim,
            gates,
            d_dgi_all,
            d_w_ih,
            layer_in,
            false,
            false,
            1.0,
            if dir == 0 { 0.0 } else { 1.0 },
            dtype,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cuda_seq_train_steps_inner(
    gpu_id: usize,
    kind: SeqCellKind,
    x: &[f32],
    target: &[f32],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
    weight_ih: &mut [&mut [f32]],
    weight_hh: &mut [&mut [f32]],
    bias_ih: &mut [Option<&mut [f32]>],
    bias_hh: &mut [Option<&mut [f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    grad_clip: Option<f32>,
) -> Option<OchreResult<Vec<f32>>> {
    if let Err(e) = validate_seq_train_args(
        kind,
        x,
        target,
        h0,
        c0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        steps,
        lr,
        weight_decay,
        grad_clip,
    ) {
        return Some(Err(e));
    }

    let dirs = if bidirectional { 2 } else { 1 };
    let units = num_layers * dirs;
    let h = hidden_size;
    let gates = kind.gate_mult() * h;
    let bh = batch * h;
    let out_dim = h * dirs;
    let rows = batch * seq_len;

    with_vram_session(gpu_id, |sess| {
        // 1. Upload inputs, targets, initial states, and weights ONCE.
        let d_x = sess.upload_f32(x)?;
        let d_target = sess.upload_f32(target)?;
        let d_h0 = match h0 {
            Some(hi) => Some(sess.upload_f32(hi)?),
            None => None,
        };
        let d_c0 = match c0 {
            Some(ci) => Some(sess.upload_f32(ci)?),
            None => None,
        };

        let mut d_params = Vec::with_capacity(units * 4);
        let mut d_grads = Vec::with_capacity(units * 4);
        let mut d_opt_m = Vec::with_capacity(units * 4);
        let mut d_opt_v = Vec::with_capacity(units * 4);
        let mut param_lens = Vec::with_capacity(units * 4);

        let mut push_param = |slice: &[f32]| -> OchreResult<usize> {
            let idx = d_params.len();
            let len = slice.len();
            param_lens.push(len);
            d_params.push(sess.upload_f32(slice)?);
            d_grads.push(sess.alloc_zeros_f32(len)?);
            d_opt_m.push(sess.alloc_zeros_f32(len)?);
            d_opt_v.push(sess.alloc_zeros_f32(len)?);
            Ok(idx)
        };

        let mut unit_states = Vec::with_capacity(units);
        for u in 0..units {
            let w_ih_idx = push_param(weight_ih[u])?;
            let w_hh_idx = push_param(weight_hh[u])?;
            let b_ih_idx = match bias_ih[u].as_deref() {
                Some(b) => Some(push_param(b)?),
                None => None,
            };
            let b_hh_idx = match bias_hh[u].as_deref() {
                Some(b) => Some(push_param(b)?),
                None => None,
            };
            let has_bias_any = b_ih_idx.is_some() || b_hh_idx.is_some();
            let d_b_packed = sess.alloc_zeros_f32(kind.packed_bias_elems(h))?;
            let d_all_h = sess.alloc_f32((seq_len + 1) * bh)?;
            let d_all_c = if kind == SeqCellKind::Lstm {
                Some(sess.alloc_f32((seq_len + 1) * bh)?)
            } else {
                None
            };
            let n_gate_bufs = match kind {
                SeqCellKind::Rnn { .. } => 0,
                SeqCellKind::Gru => 4,
                SeqCellKind::Lstm => 5,
            };
            let mut gate_bufs = Vec::with_capacity(n_gate_bufs);
            for _ in 0..n_gate_bufs {
                gate_bufs.push(sess.alloc_f32(seq_len * bh)?);
            }
            unit_states.push(SeqTrainUnitState {
                w_ih_idx,
                w_hh_idx,
                b_ih_idx,
                b_hh_idx,
                has_bias_any,
                d_b_packed,
                d_all_h,
                d_all_c,
                gate_bufs,
            });
        }

        // 2. Pre-allocate pooled layer activations, loss, and scratch buffers ONCE.
        let mut d_layer_acts = Vec::with_capacity(num_layers.saturating_sub(1));
        for _ in 1..num_layers {
            d_layer_acts.push(sess.alloc_f32(rows * out_dim)?);
        }
        let d_output = sess.alloc_f32(rows * out_dim)?;
        let d_doutput = sess.alloc_f32(rows * out_dim)?;
        let d_sq = sess.alloc_f32(rows * out_dim)?;
        let d_step_loss = sess.alloc_zeros_f32(1)?;
        let d_norm_sq = sess.alloc_zeros_f32(1)?;

        let d_gi_all = sess.alloc_f32(rows * gates)?;
        let d_gh = sess.alloc_f32(batch * gates)?;
        let d_dgi_all = sess.alloc_f32(rows * gates)?;
        let d_dgh_steps = sess.alloc_f32(seq_len * batch * gates)?;
        let d_dh_next = sess.alloc_f32(bh)?;
        let d_dc_next = if kind == SeqCellKind::Lstm {
            Some(sess.alloc_f32(bh)?)
        } else {
            None
        };
        let d_layer_grad_a = if num_layers > 1 {
            Some(sess.alloc_f32(rows * out_dim)?)
        } else {
            None
        };
        let d_layer_grad_b = if num_layers > 1 {
            Some(sess.alloc_f32(rows * out_dim)?)
        } else {
            None
        };

        let mut loss_history = Vec::with_capacity(steps);
        let mut host_step_loss = [0.0f32; 1];

        // 3. Run all training steps 100% resident in VRAM (downloading 1 scalar loss per step).
        for step in 0..steps {
            for unit in &unit_states {
                pack_seq_unit_bias_dev(
                    sess,
                    kind,
                    h,
                    unit.b_ih_idx.map(|i| d_params[i].ptr()),
                    unit.b_hh_idx.map(|i| d_params[i].ptr()),
                    unit.d_b_packed.ptr(),
                )?;
            }

            for layer in 0..num_layers {
                let in_dim = if layer == 0 { input_size } else { out_dim };
                let x_ptr = if layer == 0 {
                    d_x.ptr()
                } else {
                    d_layer_acts[layer - 1].ptr()
                };
                let out_ptr = if layer + 1 < num_layers {
                    d_layer_acts[layer].ptr()
                } else {
                    d_output.ptr()
                };
                for dir in 0..dirs {
                    let u = layer * dirs + dir;
                    seq_train_unit_fwd_dev(
                        sess,
                        kind,
                        &unit_states[u],
                        &d_params,
                        x_ptr,
                        out_ptr,
                        d_h0.as_ref(),
                        d_c0.as_ref(),
                        d_gi_all.ptr(),
                        d_gh.ptr(),
                        batch,
                        seq_len,
                        in_dim,
                        h,
                        out_dim,
                        u,
                        dir,
                        dtype,
                    )?;
                }
            }

            sess.fill_f32(d_step_loss.ptr(), 0.0, 1)?;
            sess.mse_grad_dev_f32(
                d_output.ptr(),
                d_target.ptr(),
                d_doutput.ptr(),
                d_sq.ptr(),
                d_step_loss.ptr(),
                rows,
                out_dim,
            )?;

            let mut cur_out_ptr = d_doutput.ptr();
            for layer in (0..num_layers).rev() {
                let in_dim = if layer == 0 { input_size } else { out_dim };
                let x_ptr = if layer == 0 {
                    d_x.ptr()
                } else {
                    d_layer_acts[layer - 1].ptr()
                };
                let layer_in_ptr = if layer > 0 {
                    match (&d_layer_grad_a, &d_layer_grad_b) {
                        (Some(ga), Some(gb)) => {
                            if cur_out_ptr == ga.ptr() {
                                Some(gb.ptr())
                            } else {
                                Some(ga.ptr())
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                for dir in 0..dirs {
                    let u = layer * dirs + dir;
                    seq_train_unit_bwd_dev(
                        sess,
                        kind,
                        &unit_states[u],
                        &d_params,
                        &d_grads,
                        cur_out_ptr,
                        x_ptr,
                        layer_in_ptr,
                        d_dh_next.ptr(),
                        d_dc_next.as_ref().map(CudaDeviceBuf::ptr),
                        d_dgi_all.ptr(),
                        d_dgh_steps.ptr(),
                        batch,
                        seq_len,
                        in_dim,
                        h,
                        out_dim,
                        dir,
                        dtype,
                    )?;
                }
                if let Some(next_in) = layer_in_ptr {
                    cur_out_ptr = next_in;
                }
            }

            if let Some(max_norm) = grad_clip
                && max_norm > 0.0
            {
                sess.fill_f32(d_norm_sq.ptr(), 0.0, 1)?;
                for (dg, &len) in d_grads.iter().zip(param_lens.iter()) {
                    sess.reduce_sum_sq_add_f32(dg.ptr(), d_norm_sq.ptr(), len)?;
                }
                for (dg, &len) in d_grads.iter().zip(param_lens.iter()) {
                    sess.clip_scale_dev_f32(dg.ptr(), d_norm_sq.ptr(), max_norm, len)?;
                }
            }

            for i in 0..d_params.len() {
                sess.adamw_step_dev_f32(
                    d_params[i].ptr(),
                    d_grads[i].ptr(),
                    d_opt_m[i].ptr(),
                    d_opt_v[i].ptr(),
                    param_lens[i],
                    lr,
                    0.9,
                    0.999,
                    1e-8,
                    weight_decay,
                    step + 1,
                )?;
            }

            sess.dtoh_f32(&mut host_step_loss, &d_step_loss)?;
            loss_history.push(host_step_loss[0] / (rows as f32));
        }

        // 4. Download updated weights/biases back to host slices ONCE at the end.
        let download_param = |idx: usize, dst: &mut [f32]| -> OchreResult<()> {
            sess.dtoh_f32(dst, &d_params[idx])?;
            if dtype != DType::F32 {
                let q = TensorStorage::from_f32_slice(dst, dtype).to_f32_vec();
                dst.copy_from_slice(&q);
            }
            Ok(())
        };

        for (u, unit) in unit_states.iter().enumerate() {
            download_param(unit.w_ih_idx, weight_ih[u])?;
            download_param(unit.w_hh_idx, weight_hh[u])?;
            if let (Some(idx), Some(dst)) = (unit.b_ih_idx, bias_ih[u].as_deref_mut()) {
                download_param(idx, dst)?;
            }
            if let (Some(idx), Some(dst)) = (unit.b_hh_idx, bias_hh[u].as_deref_mut()) {
                download_param(idx, dst)?;
            }
        }

        Ok(loss_history)
    })
}

/// 100% VRAM-resident multi-step AdamW + MSE training loop for Elman `RNN`.
/// Uploads inputs and weights once, runs all `steps` in VRAM (downloading only 1 scalar MSE loss
/// per step), and downloads updated weights back to host slices once at the end.
/// Returns `None` if CUDA is unavailable (graceful CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn cuda_rnn_train_steps(
    gpu_id: usize,
    x: &[f32],
    target: &[f32],
    h0: Option<&[f32]>,
    weight_ih: &mut [&mut [f32]],
    weight_hh: &mut [&mut [f32]],
    bias_ih: &mut [Option<&mut [f32]>],
    bias_hh: &mut [Option<&mut [f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    is_relu: bool,
    dtype: DType,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    grad_clip: Option<f32>,
) -> Option<OchreResult<Vec<f32>>> {
    cuda_seq_train_steps_inner(
        gpu_id,
        SeqCellKind::Rnn { is_relu },
        x,
        target,
        h0,
        None,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        dtype,
        steps,
        lr,
        weight_decay,
        grad_clip,
    )
}

/// 100% VRAM-resident multi-step AdamW + MSE training loop for `GRU`.
/// Uploads inputs and weights once, runs all `steps` in VRAM (downloading only 1 scalar MSE loss
/// per step), and downloads updated weights back to host slices once at the end.
/// Returns `None` if CUDA is unavailable (graceful CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn cuda_gru_train_steps(
    gpu_id: usize,
    x: &[f32],
    target: &[f32],
    h0: Option<&[f32]>,
    weight_ih: &mut [&mut [f32]],
    weight_hh: &mut [&mut [f32]],
    bias_ih: &mut [Option<&mut [f32]>],
    bias_hh: &mut [Option<&mut [f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    grad_clip: Option<f32>,
) -> Option<OchreResult<Vec<f32>>> {
    cuda_seq_train_steps_inner(
        gpu_id,
        SeqCellKind::Gru,
        x,
        target,
        h0,
        None,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        dtype,
        steps,
        lr,
        weight_decay,
        grad_clip,
    )
}

/// 100% VRAM-resident multi-step AdamW + MSE training loop for `LSTM`.
/// Uploads inputs and weights once, runs all `steps` in VRAM (downloading only 1 scalar MSE loss
/// per step), and downloads updated weights back to host slices once at the end.
/// Returns `None` if CUDA is unavailable (graceful CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn cuda_lstm_train_steps(
    gpu_id: usize,
    x: &[f32],
    target: &[f32],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
    weight_ih: &mut [&mut [f32]],
    weight_hh: &mut [&mut [f32]],
    bias_ih: &mut [Option<&mut [f32]>],
    bias_hh: &mut [Option<&mut [f32]>],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dtype: DType,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    grad_clip: Option<f32>,
) -> Option<OchreResult<Vec<f32>>> {
    cuda_seq_train_steps_inner(
        gpu_id,
        SeqCellKind::Lstm,
        x,
        target,
        h0,
        c0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        dtype,
        steps,
        lr,
        weight_decay,
        grad_clip,
    )
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
            "ochre_rnn_step_fwd_f32",
            "ochre_rnn_step_bwd_f32",
            "ochre_gru_step_fwd_f32",
            "ochre_gru_step_bwd_f32",
            "ochre_lstm_step_fwd_f32",
            "ochre_lstm_step_bwd_f32",
            "ochre_rope_fwd_bwd_f32",
            "ochre_swiglu_fwd_f32",
            "ochre_swiglu_bwd_f32",
            "ochre_flash_sdpa_fwd_f32",
            "ochre_flash_sdpa_bwd_dq_f32",
            "ochre_flash_sdpa_bwd_dkv_f32",
            "ochre_sdpa_causal_softmax_f32",
            "ochre_sdpa_bwd_ds_f32",
            "ochre_embedding_scatter_add_f32",
            "ochre_bshd_to_bhsd_f32",
            "ochre_bhsd_to_bshd_f32",
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

    #[test]
    fn test_vram_pool_bucket_rounding() {
        assert_eq!(pool_bucket_size(0), 256);
        assert_eq!(pool_bucket_size(4), 256);
        assert_eq!(pool_bucket_size(255), 256);
        assert_eq!(pool_bucket_size(256), 256);
        assert_eq!(pool_bucket_size(257), 512);
        assert_eq!(pool_bucket_size(1024), 1024);
        assert_eq!(pool_bucket_size(1025), 2048);
        let mib16 = 16 * 1024 * 1024;
        assert_eq!(pool_bucket_size(mib16 - 1), mib16);
        assert_eq!(pool_bucket_size(mib16), mib16);
        assert_eq!(pool_bucket_size(mib16 + 1), 17 * 1024 * 1024);
        assert_eq!(pool_bucket_size(20 * 1024 * 1024 + 123), 21 * 1024 * 1024);
    }

    #[test]
    fn test_cuda_seq_train_steps_validation_and_fallback() {
        let mut w_ih = vec![0.1f32; 8];
        let mut w_hh = vec![0.1f32; 4];
        let mut b_ih = vec![0.0f32; 2];
        let mut b_hh = vec![0.0f32; 2];
        let x = vec![0.5f32; 12];
        let target = vec![0.25f32; 6];

        // Invalid steps = 0 returns Some(Err(...)) even on CPU-only hosts
        let err_res = cuda_rnn_train_steps(
            0,
            &x,
            &target,
            None,
            &mut [w_ih.as_mut_slice()],
            &mut [w_hh.as_mut_slice()],
            &mut [Some(b_ih.as_mut_slice())],
            &mut [Some(b_hh.as_mut_slice())],
            1,
            3,
            4,
            2,
            1,
            false,
            false,
            DType::F32,
            0,
            0.01,
            0.0,
            None,
        );
        assert!(matches!(err_res, Some(Err(_))));

        // Valid args: either None on CPU-only hosts or Ok(losses) on CUDA hosts
        if let Some(res) = cuda_rnn_train_steps(
            0,
            &x,
            &target,
            None,
            &mut [w_ih.as_mut_slice()],
            &mut [w_hh.as_mut_slice()],
            &mut [Some(b_ih.as_mut_slice())],
            &mut [Some(b_hh.as_mut_slice())],
            1,
            3,
            4,
            2,
            1,
            false,
            false,
            DType::F32,
            3,
            0.01,
            0.0,
            Some(1.0),
        ) {
            let losses = res.unwrap();
            assert_eq!(losses.len(), 3);
            assert!(losses.iter().all(|v| v.is_finite()));
        }
    }
}
