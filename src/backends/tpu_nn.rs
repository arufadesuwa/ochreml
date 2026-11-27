// OchreML Google TPU PJRT Neural Network & 1-Graph Whole-Step StableHLO Compiler (*≧ω≦*)
// Supports dynamic PJRT execution for `tpu_batched_matmul_f32` / `tpu_batched_matmul_bf16`
// (`stablehlo.dot_general` f32/bf16) AND lowers forward + CrossEntropy/MSE loss + backward + AdamW
// into a single cached StableHLO MLIR program (`TpuStepGraph` / `emit_mlp_train_step_mlir`).
// Matches all `src/backends/tpu.rs` PJRT C ABI invariants for clean CPU fallback when TPU is absent.

use crate::backends::tpu::{
    PJRT_API_MAJOR, PjrtApiPrefix, PjrtBuffer, PjrtBufferDestroyArgs, PjrtBufferMemoryLayout,
    PjrtBufferMemoryLayoutTiled, PjrtBufferMemoryLayoutUnion, PjrtBufferToHostBufferArgs,
    PjrtClient, PjrtClientAddressableDevicesArgs, PjrtClientBufferFromHostBufferArgs,
    PjrtClientCompileArgs, PjrtClientCreateArgs, PjrtClientDestroyArgs, PjrtDevice, PjrtError,
    PjrtErrorDestroyArgs, PjrtEvent, PjrtEventAwaitArgs, PjrtEventDestroyArgs, PjrtExecuteOptions,
    PjrtLoadedExecutable, PjrtLoadedExecutableExecuteArgs, PjrtPluginInitializeArgs, PjrtProgram,
    next_pow2_dim,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::dtype::{bf16_to_f32, f32_to_bf16};
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt::Write as _;
use std::ptr;
use std::sync::{Mutex, OnceLock};

// PJRT C API constants matching `src/backends/tpu.rs` and OpenXLA `pjrt_c_api.h`
const PJRT_BUFFER_TYPE_F32: c_int = 11;
const PJRT_BUFFER_TYPE_BF16: c_int = 16;
const PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL: c_int = 0;
const PJRT_BUFFER_MEMORY_LAYOUT_TYPE_TILED: c_int = 0;

// Serialized xla::CompileOptionsProto { executable_build_options: { num_replicas: 1, num_partitions: 1 } }
const COMPILE_OPTIONS_PROTO: [u8; 6] = [0x1A, 0x04, 0x20, 0x01, 0x28, 0x01];
const MIN_TPU_NN_OPS: usize = 16_777_216;

struct SendExecutable(*mut PjrtLoadedExecutable);
// SAFETY: PJRT loaded executables are thread-safe across execute calls.
unsafe impl Send for SendExecutable {}
// SAFETY: Guarded by Mutex cache.
unsafe impl Sync for SendExecutable {}

struct SendClient(*mut PjrtClient);
// SAFETY: PJRT_Client is thread-safe per PJRT C API specification.
unsafe impl Send for SendClient {}
// SAFETY: Shared across threads.
unsafe impl Sync for SendClient {}

struct SendDevice(*mut PjrtDevice);
// SAFETY: Device pointer belongs to static PJRT_Client lifetime.
unsafe impl Send for SendDevice {}
// SAFETY: Immutable pointer.
unsafe impl Sync for SendDevice {}

struct SendApi(*const PjrtApiPrefix);
// SAFETY: PJRT API function table lives as long as `_lib`.
unsafe impl Send for SendApi {}
// SAFETY: Immutable function table.
unsafe impl Sync for SendApi {}

struct TpuNnContext {
    _lib: Library,
    api: SendApi,
    client: SendClient,
    devices: Vec<SendDevice>,
    exec_cache: Mutex<HashMap<String, SendExecutable>>,
}

static TPU_NN_CTX: OnceLock<Option<TpuNnContext>> = OnceLock::new();

fn candidate_pjrt_paths() -> Vec<String> {
    let mut paths = Vec::new();
    for var in ["PJRT_PLUGIN_LIBRARY_PATH", "TPU_LIBRARY_PATH"] {
        if let Ok(p) = std::env::var(var)
            && !p.is_empty()
        {
            paths.push(p);
        }
    }
    paths.extend(
        [
            "/usr/local/lib/libpjrt_c_api_tpu.so",
            "/usr/lib/libpjrt_c_api_tpu.so",
            "/usr/local/lib/libtpu.so",
            "/usr/lib/libtpu.so",
            "libpjrt_c_api_tpu.so",
            "libtpu.so",
            "/usr/local/lib/libpjrt_c_api.so",
        ]
        .into_iter()
        .map(String::from),
    );
    paths
}

fn check_pjrt_err(api: &PjrtApiPrefix, err: *mut PjrtError) -> bool {
    if err.is_null() {
        return true;
    }
    if let Some(destroy_fn) = api.error_destroy {
        let mut args = PjrtErrorDestroyArgs {
            struct_size: std::mem::size_of::<PjrtErrorDestroyArgs>(),
            extension_start: ptr::null_mut(),
            error: err,
        };
        // SAFETY: `err` is a valid non-null PJRT_Error returned by `api`.
        unsafe { destroy_fn(&mut args) };
    }
    false
}

fn await_and_destroy_event(api: &PjrtApiPrefix, event: *mut PjrtEvent) -> bool {
    if event.is_null() {
        return true;
    }
    let mut ok = true;
    if let Some(await_fn) = api.event_await {
        let mut args = PjrtEventAwaitArgs {
            struct_size: std::mem::size_of::<PjrtEventAwaitArgs>(),
            extension_start: ptr::null_mut(),
            event,
        };
        // SAFETY: `event` is valid and owned until destroyed below.
        let err = unsafe { await_fn(&mut args) };
        ok = check_pjrt_err(api, err);
    }
    if let Some(destroy_fn) = api.event_destroy {
        let mut args = PjrtEventDestroyArgs {
            struct_size: std::mem::size_of::<PjrtEventDestroyArgs>(),
            extension_start: ptr::null_mut(),
            event,
        };
        // SAFETY: `event` is valid and destroyed once.
        let err = unsafe { destroy_fn(&mut args) };
        let _ = check_pjrt_err(api, err);
    }
    ok
}

fn destroy_buffer(api: &PjrtApiPrefix, buffer: *mut PjrtBuffer) {
    if buffer.is_null() {
        return;
    }
    if let Some(destroy_fn) = api.buffer_destroy {
        let mut args = PjrtBufferDestroyArgs {
            struct_size: std::mem::size_of::<PjrtBufferDestroyArgs>(),
            extension_start: ptr::null_mut(),
            buffer,
        };
        // SAFETY: `buffer` is valid and owned by caller.
        let err = unsafe { destroy_fn(&mut args) };
        let _ = check_pjrt_err(api, err);
    }
}

fn try_init_tpu_nn() -> Option<TpuNnContext> {
    for path in candidate_pjrt_paths() {
        // SAFETY: Dynamically loading standard PJRT TPU plugin shared object.
        let lib = match unsafe { Library::new(&path) } {
            Ok(l) => l,
            Err(_) => continue,
        };
        // SAFETY: Resolving standard `GetPjrtApi` entry point.
        let get_api: Symbol<unsafe extern "C" fn() -> *const PjrtApiPrefix> =
            match unsafe { lib.get(b"GetPjrtApi\0") } {
                Ok(sym) => sym,
                Err(_) => continue,
            };
        // SAFETY: Invokes plugin API table getter.
        let api_ptr = unsafe { get_api() };
        if api_ptr.is_null() {
            continue;
        }
        // SAFETY: `api_ptr` points to static PJRT_Api struct from `lib`.
        let api = unsafe { &*api_ptr };
        if api.struct_size < std::mem::size_of::<PjrtApiPrefix>()
            || api.pjrt_api_version.major_version != PJRT_API_MAJOR
        {
            continue;
        }
        if api.client_create.is_none()
            || api.client_addressable_devices.is_none()
            || api.client_compile.is_none()
            || api.client_buffer_from_host_buffer.is_none()
            || api.loaded_executable_execute.is_none()
            || api.buffer_to_host_buffer.is_none()
            || api.buffer_destroy.is_none()
        {
            continue;
        }

        if let Some(init_fn) = api.plugin_initialize {
            let mut init_args = PjrtPluginInitializeArgs {
                struct_size: std::mem::size_of::<PjrtPluginInitializeArgs>(),
                extension_start: ptr::null_mut(),
            };
            // SAFETY: Standard PJRT plugin initialization call.
            let err = unsafe { init_fn(&mut init_args) };
            if !check_pjrt_err(api, err) {
                continue;
            }
        }

        let create_fn = api.client_create?;
        let mut create_args = PjrtClientCreateArgs {
            struct_size: std::mem::size_of::<PjrtClientCreateArgs>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: ptr::null(),
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        // SAFETY: Creating PJRT client with valid zeroed args struct.
        let err = unsafe { create_fn(&mut create_args) };
        if !check_pjrt_err(api, err) || create_args.client.is_null() {
            continue;
        }

        let dev_fn = api.client_addressable_devices?;
        let mut dev_args = PjrtClientAddressableDevicesArgs {
            struct_size: std::mem::size_of::<PjrtClientAddressableDevicesArgs>(),
            extension_start: ptr::null_mut(),
            client: create_args.client,
            addressable_devices: ptr::null(),
            num_addressable_devices: 0,
        };
        // SAFETY: Querying addressable devices from `create_args.client`.
        let err = unsafe { dev_fn(&mut dev_args) };
        if !check_pjrt_err(api, err)
            || dev_args.addressable_devices.is_null()
            || dev_args.num_addressable_devices == 0
        {
            if let Some(destroy_fn) = api.client_destroy {
                let mut dargs = PjrtClientDestroyArgs {
                    struct_size: std::mem::size_of::<PjrtClientDestroyArgs>(),
                    extension_start: ptr::null_mut(),
                    client: create_args.client,
                };
                // SAFETY: Releasing client when no devices are addressable.
                let _ = check_pjrt_err(api, unsafe { destroy_fn(&mut dargs) });
            }
            continue;
        }

        // SAFETY: PJRT guarantees `addressable_devices` has `num_addressable_devices` elements.
        let devices_slice = unsafe {
            std::slice::from_raw_parts(
                dev_args.addressable_devices,
                dev_args.num_addressable_devices,
            )
        };
        let devices = devices_slice.iter().copied().map(SendDevice).collect();

        return Some(TpuNnContext {
            _lib: lib,
            api: SendApi(api_ptr),
            client: SendClient(create_args.client),
            devices,
            exec_cache: Mutex::new(HashMap::new()),
        });
    }
    None
}

fn get_tpu_nn_ctx() -> Option<&'static TpuNnContext> {
    TPU_NN_CTX.get_or_init(try_init_tpu_nn).as_ref()
}

fn get_or_compile(
    ctx: &TpuNnContext,
    key: &str,
    mlir_src: &str,
) -> Option<*mut PjrtLoadedExecutable> {
    {
        let cache = ctx.exec_cache.lock().ok()?;
        if let Some(exec) = cache.get(key) {
            return Some(exec.0);
        }
    }

    let code_cstr = CString::new(mlir_src).ok()?;
    let format_cstr = CString::new("mlir").ok()?;
    // SAFETY: `ctx.api.0` points to static PJRT C API table.
    let api = unsafe { &*ctx.api.0 };
    let compile_fn = api.client_compile?;

    let program = PjrtProgram {
        struct_size: std::mem::size_of::<PjrtProgram>(),
        extension_start: ptr::null_mut(),
        code: code_cstr.as_ptr(),
        code_size: code_cstr.as_bytes().len(),
        format: format_cstr.as_ptr(),
        format_size: format_cstr.as_bytes().len(),
    };
    let mut compile_args = PjrtClientCompileArgs {
        struct_size: std::mem::size_of::<PjrtClientCompileArgs>(),
        extension_start: ptr::null_mut(),
        client: ctx.client.0,
        program: &program,
        compile_options: COMPILE_OPTIONS_PROTO.as_ptr().cast::<c_char>(),
        compile_options_size: COMPILE_OPTIONS_PROTO.len(),
        executable: ptr::null_mut(),
    };
    // SAFETY: Compiling StableHLO MLIR module with valid PJRT client and options proto.
    let err = unsafe { compile_fn(&mut compile_args) };
    if !check_pjrt_err(api, err) || compile_args.executable.is_null() {
        return None;
    }

    let exec_ptr = compile_args.executable;
    if let Ok(mut cache) = ctx.exec_cache.lock() {
        cache.insert(key.to_string(), SendExecutable(exec_ptr));
    }
    Some(exec_ptr)
}

fn upload_raw_buffer(
    api: &PjrtApiPrefix,
    client: *mut PjrtClient,
    device: *mut PjrtDevice,
    data: *const c_void,
    buffer_type: c_int,
    dims: &[i64],
) -> Option<*mut PjrtBuffer> {
    let upload_fn = api.client_buffer_from_host_buffer?;
    let mut args = PjrtClientBufferFromHostBufferArgs {
        struct_size: std::mem::size_of::<PjrtClientBufferFromHostBufferArgs>(),
        extension_start: ptr::null_mut(),
        client,
        data,
        buffer_type,
        dims: dims.as_ptr(),
        num_dims: dims.len(),
        byte_strides: ptr::null(),
        num_byte_strides: 0,
        host_buffer_semantics: PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL,
        device,
        memory: ptr::null_mut(),
        device_layout: ptr::null_mut(),
        done_with_host_buffer: ptr::null_mut(),
        buffer: ptr::null_mut(),
    };
    // SAFETY: `data` points to contiguous host buffer matching `dims` and `buffer_type`.
    let err = unsafe { upload_fn(&mut args) };
    if !check_pjrt_err(api, err) || args.buffer.is_null() {
        let _ = await_and_destroy_event(api, args.done_with_host_buffer);
        return None;
    }
    if !await_and_destroy_event(api, args.done_with_host_buffer) {
        destroy_buffer(api, args.buffer);
        return None;
    }
    Some(args.buffer)
}

fn download_buffer(
    api: &PjrtApiPrefix,
    src: *mut PjrtBuffer,
    dst: *mut c_void,
    dst_size: usize,
    minor_to_major: &[i64],
) -> bool {
    let Some(to_host_fn) = api.buffer_to_host_buffer else {
        return false;
    };
    let mut host_layout = PjrtBufferMemoryLayout {
        struct_size: std::mem::size_of::<PjrtBufferMemoryLayout>(),
        extension_start: ptr::null_mut(),
        layout: PjrtBufferMemoryLayoutUnion {
            tiled: PjrtBufferMemoryLayoutTiled {
                struct_size: std::mem::size_of::<PjrtBufferMemoryLayoutTiled>(),
                extension_start: ptr::null_mut(),
                minor_to_major: minor_to_major.as_ptr(),
                minor_to_major_size: minor_to_major.len(),
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
                num_tiles: 0,
            },
        },
        layout_type: PJRT_BUFFER_MEMORY_LAYOUT_TYPE_TILED,
    };
    let mut args = PjrtBufferToHostBufferArgs {
        struct_size: std::mem::size_of::<PjrtBufferToHostBufferArgs>(),
        extension_start: ptr::null_mut(),
        src,
        host_layout: (&mut host_layout as *mut PjrtBufferMemoryLayout).cast(),
        dst,
        dst_size,
        event: ptr::null_mut(),
    };
    // SAFETY: `dst` has `dst_size` bytes matching the device buffer's row-major layout.
    let err = unsafe { to_host_fn(&mut args) };
    check_pjrt_err(api, err) && await_and_destroy_event(api, args.event)
}

/// Emit StableHLO MLIR for batched matrix multiplication (`stablehlo.dot_general`) in `f32` or `bf16`.
pub fn emit_batched_dot_general_mlir(
    batch: usize,
    m_pad: usize,
    k_pad: usize,
    n_pad: usize,
    trans_a: bool,
    trans_b: bool,
    bf16: bool,
) -> String {
    let elem = if bf16 { "bf16" } else { "f32" };
    let (a_r, a_c, lhs_contract) = if trans_a {
        (k_pad, m_pad, 1usize)
    } else {
        (m_pad, k_pad, 2usize)
    };
    let (b_r, b_c, rhs_contract) = if trans_b {
        (n_pad, k_pad, 2usize)
    } else {
        (k_pad, n_pad, 1usize)
    };

    format!(
        "module @ochreml_tpu_bmm {{\n\
         \x20 func.func @main(%arg0: tensor<{batch}x{a_r}x{a_c}x{elem}>, %arg1: tensor<{batch}x{b_r}x{b_c}x{elem}>) -> tensor<{batch}x{m_pad}x{n_pad}xf32> {{\n\
         \x20   %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [{lhs_contract}] x [{rhs_contract}], precision = [HIGHEST, HIGHEST] : (tensor<{batch}x{a_r}x{a_c}x{elem}>, tensor<{batch}x{b_r}x{b_c}x{elem}>) -> tensor<{batch}x{m_pad}x{n_pad}xf32>\n\
         \x20   return %0 : tensor<{batch}x{m_pad}x{n_pad}xf32>\n\
         \x20 }}\n\
         }}\n"
    )
}

fn pad_batched_3d_f32(
    src: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * pad_rows * pad_cols];
    for b in 0..batch {
        for r in 0..rows {
            let s = &src[(b * rows + r) * cols..(b * rows + r) * cols + cols];
            let d = &mut out[(b * pad_rows + r) * pad_cols..(b * pad_rows + r) * pad_cols + cols];
            d.copy_from_slice(s);
        }
    }
    out
}

fn unpad_batched_3d_f32(
    src: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * rows * cols];
    for b in 0..batch {
        for r in 0..rows {
            let s = &src[(b * pad_rows + r) * pad_cols..(b * pad_rows + r) * pad_cols + cols];
            let d = &mut out[(b * rows + r) * cols..(b * rows + r) * cols + cols];
            d.copy_from_slice(s);
        }
    }
    out
}

/// N7: Execute batched `f32` matmul via PJRT StableHLO `dot_general` on TPU.
#[allow(clippy::too_many_arguments)]
pub fn tpu_batched_matmul_f32(
    device_id: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: bool,
    trans_b: bool,
) -> Option<OchreResult<Vec<f32>>> {
    tpu_batched_matmul_impl(device_id, batch, m, n, k, a, b, trans_a, trans_b, false)
}

/// N7: Execute batched `bf16` MXU matmul with `f32` accumulation via PJRT StableHLO `dot_general`.
#[allow(clippy::too_many_arguments)]
pub fn tpu_batched_matmul_bf16(
    device_id: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: bool,
    trans_b: bool,
) -> Option<OchreResult<Vec<f32>>> {
    tpu_batched_matmul_impl(device_id, batch, m, n, k, a, b, trans_a, trans_b, true)
}

#[allow(clippy::too_many_arguments)]
fn tpu_batched_matmul_impl(
    device_id: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: bool,
    trans_b: bool,
    bf16: bool,
) -> Option<OchreResult<Vec<f32>>> {
    if batch == 0
        || m == 0
        || n == 0
        || k == 0
        || batch.saturating_mul(m).saturating_mul(n).saturating_mul(k) < MIN_TPU_NN_OPS
    {
        return None;
    }
    let ctx = get_tpu_nn_ctx()?;
    let device = ctx
        .devices
        .get(device_id)
        .or_else(|| ctx.devices.first())?
        .0;
    // SAFETY: `ctx.api.0` points to valid PJRT C API table.
    let api = unsafe { &*ctx.api.0 };

    let m_pad = next_pow2_dim(m, 16);
    let k_pad = next_pow2_dim(k, 16);
    let n_pad = next_pow2_dim(n, 16);
    let key = format!("bmm:{batch}x{m_pad}x{k_pad}x{n_pad}:ta={trans_a}:tb={trans_b}:bf16={bf16}");
    let mlir = emit_batched_dot_general_mlir(batch, m_pad, k_pad, n_pad, trans_a, trans_b, bf16);
    let exec = get_or_compile(ctx, &key, &mlir)?;

    let (a_rows, a_cols, a_pad_r, a_pad_c) = if trans_a {
        (k, m, k_pad, m_pad)
    } else {
        (m, k, m_pad, k_pad)
    };
    let (b_rows, b_cols, b_pad_r, b_pad_c) = if trans_b {
        (n, k, n_pad, k_pad)
    } else {
        (k, n, k_pad, n_pad)
    };

    let a_f32 = pad_batched_3d_f32(a, batch, a_rows, a_cols, a_pad_r, a_pad_c);
    let b_f32 = pad_batched_3d_f32(b, batch, b_rows, b_cols, b_pad_r, b_pad_c);
    let a_dims = [batch as i64, a_pad_r as i64, a_pad_c as i64];
    let b_dims = [batch as i64, b_pad_r as i64, b_pad_c as i64];

    let (buf_a, buf_b) = if bf16 {
        let a_bf16: Vec<u16> = a_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let ba = upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            a_bf16.as_ptr().cast(),
            PJRT_BUFFER_TYPE_BF16,
            &a_dims,
        )?;
        let bb = match upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            b_bf16.as_ptr().cast(),
            PJRT_BUFFER_TYPE_BF16,
            &b_dims,
        ) {
            Some(buf) => buf,
            None => {
                destroy_buffer(api, ba);
                return None;
            }
        };
        (ba, bb)
    } else {
        let ba = upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            a_f32.as_ptr().cast(),
            PJRT_BUFFER_TYPE_F32,
            &a_dims,
        )?;
        let bb = match upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            b_f32.as_ptr().cast(),
            PJRT_BUFFER_TYPE_F32,
            &b_dims,
        ) {
            Some(buf) => buf,
            None => {
                destroy_buffer(api, ba);
                return None;
            }
        };
        (ba, bb)
    };

    let exec_fn = api.loaded_executable_execute?;
    let mut arg_ptrs = [buf_a, buf_b];
    let arg_lists: [*const *mut PjrtBuffer; 1] = [arg_ptrs.as_mut_ptr()];
    let mut out_ptrs: [*mut PjrtBuffer; 1] = [ptr::null_mut()];
    let out_lists: [*mut *mut PjrtBuffer; 1] = [out_ptrs.as_mut_ptr()];
    let mut complete_event: *mut PjrtEvent = ptr::null_mut();

    let options = PjrtExecuteOptions {
        struct_size: std::mem::size_of::<PjrtExecuteOptions>(),
        extension_start: ptr::null_mut(),
        send_callbacks: ptr::null_mut(),
        recv_callbacks: ptr::null_mut(),
        num_send_ops: 0,
        num_recv_ops: 0,
        launch_id: 0,
        non_donatable_input_indices: ptr::null(),
        num_non_donatable_input_indices: 0,
        context: ptr::null_mut(),
        call_location: ptr::null(),
        num_tasks: 0,
        task_ids: ptr::null_mut(),
        incarnation_ids: ptr::null_mut(),
        multi_slice_config: ptr::null_mut(),
        use_major_to_minor_data_layout_for_callbacks: false,
        hlo_output_callbacks: ptr::null_mut(),
        num_hlo_output_callbacks: 0,
        custom_options: ptr::null(),
        num_custom_options: 0,
    };
    let mut exec_args = PjrtLoadedExecutableExecuteArgs {
        struct_size: std::mem::size_of::<PjrtLoadedExecutableExecuteArgs>(),
        extension_start: ptr::null_mut(),
        executable: exec,
        options: &options,
        argument_lists: arg_lists.as_ptr(),
        num_devices: 1,
        num_args: 2,
        output_lists: out_lists.as_ptr(),
        device_complete_events: &mut complete_event,
        execute_device: ptr::null_mut(),
    };

    // SAFETY: `exec`, `buf_a`, and `buf_b` are valid PJRT handles on `ctx.client`.
    let err = unsafe { exec_fn(&mut exec_args) };
    destroy_buffer(api, buf_a);
    destroy_buffer(api, buf_b);

    if !check_pjrt_err(api, err)
        || out_ptrs[0].is_null()
        || !await_and_destroy_event(api, complete_event)
    {
        destroy_buffer(api, out_ptrs[0]);
        return None;
    }

    let mut out_pad = vec![0.0f32; batch * m_pad * n_pad];
    let ok = download_buffer(
        api,
        out_ptrs[0],
        out_pad.as_mut_ptr().cast(),
        out_pad.len() * std::mem::size_of::<f32>(),
        &[2, 1, 0],
    );
    destroy_buffer(api, out_ptrs[0]);
    if !ok {
        return None;
    }

    Some(Ok(unpad_batched_3d_f32(
        &out_pad, batch, m, n, m_pad, n_pad,
    )))
}

// ============================================================================
// 1-Graph Whole-Step StableHLO Compiler (Forward + Loss + Backward + AdamW)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TpuLossKind {
    CrossEntropy,
    Mse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TpuActivation {
    Relu,
    Gelu,
    Silu,
    Tanh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TpuDType {
    F32,
    BF16,
}

/// Whole-step MLP training graph lowered to a single cached StableHLO program:
/// Forward -> CrossEntropy/MSE Loss -> Backward -> AdamW parameter & moment update.
#[derive(Clone, Debug, PartialEq)]
pub struct TpuStepGraph {
    pub batch_size: usize,
    pub layer_dims: Vec<usize>,
    pub activation: TpuActivation,
    pub loss: TpuLossKind,
    pub dtype: TpuDType,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

/// Output buffers returned by a single TPU training step graph execution.
#[derive(Clone, Debug)]
pub struct TpuStepOutput {
    pub loss: f32,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub m_weights: Vec<Vec<f32>>,
    pub v_weights: Vec<Vec<f32>>,
    pub m_biases: Vec<Vec<f32>>,
    pub v_biases: Vec<Vec<f32>>,
}

impl TpuStepGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size: usize,
        layer_dims: Vec<usize>,
        activation: TpuActivation,
        loss: TpuLossKind,
        dtype: TpuDType,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> OchreResult<Self> {
        if batch_size == 0 || layer_dims.len() < 2 || layer_dims.contains(&0) {
            return Err(OchreError::InvalidParameter(
                "TpuStepGraph requires non-zero batch_size and >= 2 positive layer_dims ( >_< )"
                    .to_string(),
            ));
        }
        if !lr.is_finite()
            || lr < 0.0
            || !beta1.is_finite()
            || !(0.0..1.0).contains(&beta1)
            || !beta2.is_finite()
            || !(0.0..1.0).contains(&beta2)
            || !eps.is_finite()
            || eps <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(OchreError::InvalidParameter(
                "TpuStepGraph received invalid AdamW hyperparameters ( >_< )".to_string(),
            ));
        }
        Ok(Self {
            batch_size,
            layer_dims,
            activation,
            loss,
            dtype,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layer_dims.len() - 1
    }

    pub fn cache_key(&self) -> String {
        format!(
            "mlp_step:b={}:dims={:?}:act={:?}:loss={:?}:dt={:?}:lr={:08x}:b1={:08x}:b2={:08x}:eps={:08x}:wd={:08x}",
            self.batch_size,
            self.layer_dims,
            self.activation,
            self.loss,
            self.dtype,
            self.lr.to_bits(),
            self.beta1.to_bits(),
            self.beta2.to_bits(),
            self.eps.to_bits(),
            self.weight_decay.to_bits(),
        )
    }

    pub fn emit_mlir(&self) -> OchreResult<String> {
        emit_mlp_train_step_mlir(self)
    }

    /// Emit a single StableHLO MLIR program wrapping `num_steps` consecutive training steps
    /// inside a `stablehlo.while` loop with zero host round-trips per step.
    pub fn emit_multi_step_while_mlir(&self, num_steps: usize) -> String {
        emit_mlp_multi_step_while_mlir(self, num_steps)
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn emit_mlp_step_ops(
    graph: &TpuStepGraph,
    body: &mut String,
    indent: &str,
    id_counter: &mut usize,
    x_var: &str,
    targets_var: &str,
    inv_bc1_var: &str,
    inv_bc2_var: &str,
    w_vars: &[String],
    b_vars: &[String],
    mw_vars: &[String],
    vw_vars: &[String],
    mb_vars: &[String],
    vb_vars: &[String],
) -> (
    String,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
) {
    let n_layers = graph.num_layers();
    let b = graph.batch_size;
    let in_dim = graph.layer_dims[0];
    let out_dim = graph.layer_dims[n_layers];

    let mut next_id = |prefix: &str| -> String {
        let s = format!("%{prefix}_{}", *id_counter);
        *id_counter += 1;
        s
    };

    // Convert input to f32 if bf16 for unified residual/loss math (dot_general uses BF16 MXU).
    let mut cur_act_f32 = if graph.dtype == TpuDType::BF16 {
        let v = next_id("x_f32");
        let _ = writeln!(
            body,
            "{indent}{v} = stablehlo.convert {x_var} : (tensor<{b}x{in_dim}xbf16>) -> tensor<{b}x{in_dim}xf32>"
        );
        v
    } else {
        x_var.to_string()
    };

    // --- 1. Forward pass ---
    let mut layer_inputs = Vec::with_capacity(n_layers);
    let mut layer_pre_acts = Vec::with_capacity(n_layers);
    let mut layer_post_acts = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        let wl = &w_vars[l];
        let bl = &b_vars[l];
        layer_inputs.push(cur_act_f32.clone());

        let mm = if graph.dtype == TpuDType::BF16 {
            let lhs = next_id("lhs_bf16");
            let rhs = next_id("rhs_bf16");
            let dot = next_id("dot");
            let _ = writeln!(
                body,
                "{indent}{lhs} = stablehlo.convert {cur_act_f32} : (tensor<{b}x{d_in}xf32>) -> tensor<{b}x{d_in}xbf16>"
            );
            let _ = writeln!(
                body,
                "{indent}{rhs} = stablehlo.convert {wl} : (tensor<{d_in}x{d_out}xf32>) -> tensor<{d_in}x{d_out}xbf16>"
            );
            let _ = writeln!(
                body,
                "{indent}{dot} = stablehlo.dot_general {lhs}, {rhs}, contracting_dims = [1] x [0] : (tensor<{b}x{d_in}xbf16>, tensor<{d_in}x{d_out}xbf16>) -> tensor<{b}x{d_out}xf32>"
            );
            dot
        } else {
            let dot = next_id("dot");
            let _ = writeln!(
                body,
                "{indent}{dot} = stablehlo.dot_general {cur_act_f32}, {wl}, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<{b}x{d_in}xf32>, tensor<{d_in}x{d_out}xf32>) -> tensor<{b}x{d_out}xf32>"
            );
            dot
        };

        let b_bcast = next_id("b_bcast");
        let pre = next_id("pre");
        let _ = writeln!(
            body,
            "{indent}{b_bcast} = stablehlo.broadcast_in_dim {bl}, dims = [1] : (tensor<{d_out}xf32>) -> tensor<{b}x{d_out}xf32>"
        );
        let _ = writeln!(
            body,
            "{indent}{pre} = stablehlo.add {mm}, {b_bcast} : tensor<{b}x{d_out}xf32>"
        );
        layer_pre_acts.push(pre.clone());

        if l + 1 < n_layers {
            let ty = format!("tensor<{b}x{d_out}xf32>");
            let post = match graph.activation {
                TpuActivation::Relu => {
                    let z = next_id("zero");
                    let out = next_id("relu");
                    let _ = writeln!(body, "{indent}{z} = stablehlo.constant dense<0.0> : {ty}");
                    let _ = writeln!(body, "{indent}{out} = stablehlo.maximum {pre}, {z} : {ty}");
                    out
                }
                TpuActivation::Tanh => {
                    let out = next_id("tanh");
                    let _ = writeln!(body, "{indent}{out} = stablehlo.tanh {pre} : {ty}");
                    out
                }
                TpuActivation::Silu => {
                    let sig = next_id("sig");
                    let out = next_id("silu");
                    let _ = writeln!(body, "{indent}{sig} = stablehlo.logistic {pre} : {ty}");
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.multiply {pre}, {sig} : {ty}"
                    );
                    out
                }
                TpuActivation::Gelu => {
                    let x2 = next_id("x2");
                    let x3 = next_id("x3");
                    let k = next_id("k");
                    let kx3 = next_id("kx3");
                    let inner = next_id("inner");
                    let c = next_id("c");
                    let u = next_id("u");
                    let th = next_id("th");
                    let one = next_id("one");
                    let half = next_id("half");
                    let one_p = next_id("one_p");
                    let hx = next_id("hx");
                    let out = next_id("gelu");
                    let _ = writeln!(
                        body,
                        "{indent}{x2} = stablehlo.multiply {pre}, {pre} : {ty}"
                    );
                    let _ = writeln!(body, "{indent}{x3} = stablehlo.multiply {x2}, {pre} : {ty}");
                    let _ = writeln!(
                        body,
                        "{indent}{k} = stablehlo.constant dense<4.471500e-02> : {ty}"
                    );
                    let _ = writeln!(body, "{indent}{kx3} = stablehlo.multiply {k}, {x3} : {ty}");
                    let _ = writeln!(body, "{indent}{inner} = stablehlo.add {pre}, {kx3} : {ty}");
                    let _ = writeln!(
                        body,
                        "{indent}{c} = stablehlo.constant dense<7.978846e-01> : {ty}"
                    );
                    let _ = writeln!(body, "{indent}{u} = stablehlo.multiply {c}, {inner} : {ty}");
                    let _ = writeln!(body, "{indent}{th} = stablehlo.tanh {u} : {ty}");
                    let _ = writeln!(body, "{indent}{one} = stablehlo.constant dense<1.0> : {ty}");
                    let _ = writeln!(
                        body,
                        "{indent}{half} = stablehlo.constant dense<0.5> : {ty}"
                    );
                    let _ = writeln!(body, "{indent}{one_p} = stablehlo.add {one}, {th} : {ty}");
                    let _ = writeln!(
                        body,
                        "{indent}{hx} = stablehlo.multiply {half}, {pre} : {ty}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.multiply {hx}, {one_p} : {ty}"
                    );
                    out
                }
            };
            layer_post_acts.push(post.clone());
            cur_act_f32 = post;
        } else {
            layer_post_acts.push(pre.clone());
            cur_act_f32 = pre;
        }
    }

    // --- 2. Loss & d_logits ---
    let logits_ty = format!("tensor<{b}x{out_dim}xf32>");
    let row_ty = format!("tensor<{b}xf32>");
    let (loss_var, mut d_out) = match graph.loss {
        TpuLossKind::Mse => {
            let diff = next_id("mse_diff");
            let sq = next_id("mse_sq");
            let z0 = next_id("z0");
            let sum_all = next_id("mse_sum");
            let denom = next_id("mse_denom");
            let loss = next_id("loss");
            let two_denom = next_id("mse_scale");
            let scale_b = next_id("mse_scale_b");
            let dlogits = next_id("dlogits");
            let n_elems = (b * out_dim) as f32;
            let _ = writeln!(
                body,
                "{indent}{diff} = stablehlo.subtract {cur_act_f32}, {targets_var} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{sq} = stablehlo.multiply {diff}, {diff} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{z0} = stablehlo.constant dense<0.0> : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{sum_all} = stablehlo.reduce({sq} init: {z0}) across dimensions = [0, 1] : ({logits_ty}, tensor<f32>) -> tensor<f32> reducer(%a: tensor<f32>, %b: tensor<f32>) {{ %s = stablehlo.add %a, %b : tensor<f32>; stablehlo.return %s : tensor<f32> }}"
            );
            let _ = writeln!(
                body,
                "{indent}{denom} = stablehlo.constant dense<{n_elems:.8e}> : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{loss} = stablehlo.divide {sum_all}, {denom} : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{two_denom} = stablehlo.constant dense<{:.8e}> : tensor<f32>",
                2.0 / n_elems
            );
            let _ = writeln!(
                body,
                "{indent}{scale_b} = stablehlo.broadcast_in_dim {two_denom}, dims = [] : (tensor<f32>) -> {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{dlogits} = stablehlo.multiply {diff}, {scale_b} : {logits_ty}"
            );
            (loss, dlogits)
        }
        TpuLossKind::CrossEntropy => {
            let ninf = next_id("ninf");
            let max_r = next_id("row_max");
            let max_b = next_id("max_b");
            let shifted = next_id("shifted");
            let exps = next_id("exps");
            let z0 = next_id("z0");
            let sum_exp = next_id("sum_exp");
            let sum_b = next_id("sum_b");
            let probs = next_id("probs");
            let log_sum = next_id("log_sum");
            let log_b = next_id("log_b");
            let log_sm = next_id("log_sm");
            let prod = next_id("ce_prod");
            let sum_loss = next_id("sum_loss");
            let neg_inv_b = next_id("neg_inv_b");
            let loss = next_id("loss");
            let p_minus_t = next_id("p_minus_t");
            let inv_b = next_id("inv_b");
            let inv_b_bcast = next_id("inv_b_bcast");
            let dlogits = next_id("dlogits");

            let _ = writeln!(
                body,
                "{indent}{ninf} = stablehlo.constant dense<-3.4028234e+38> : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{max_r} = stablehlo.reduce({cur_act_f32} init: {ninf}) across dimensions = [1] : ({logits_ty}, tensor<f32>) -> {row_ty} reducer(%a: tensor<f32>, %b: tensor<f32>) {{ %m = stablehlo.maximum %a, %b : tensor<f32>; stablehlo.return %m : tensor<f32> }}"
            );
            let _ = writeln!(
                body,
                "{indent}{max_b} = stablehlo.broadcast_in_dim {max_r}, dims = [0] : ({row_ty}) -> {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{shifted} = stablehlo.subtract {cur_act_f32}, {max_b} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{exps} = stablehlo.exponential {shifted} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{z0} = stablehlo.constant dense<0.0> : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{sum_exp} = stablehlo.reduce({exps} init: {z0}) across dimensions = [1] : ({logits_ty}, tensor<f32>) -> {row_ty} reducer(%a: tensor<f32>, %b: tensor<f32>) {{ %s = stablehlo.add %a, %b : tensor<f32>; stablehlo.return %s : tensor<f32> }}"
            );
            let _ = writeln!(
                body,
                "{indent}{sum_b} = stablehlo.broadcast_in_dim {sum_exp}, dims = [0] : ({row_ty}) -> {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{probs} = stablehlo.divide {exps}, {sum_b} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{log_sum} = stablehlo.log {sum_exp} : {row_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{log_b} = stablehlo.broadcast_in_dim {log_sum}, dims = [0] : ({row_ty}) -> {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{log_sm} = stablehlo.subtract {shifted}, {log_b} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{prod} = stablehlo.multiply {targets_var}, {log_sm} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{sum_loss} = stablehlo.reduce({prod} init: {z0}) across dimensions = [0, 1] : ({logits_ty}, tensor<f32>) -> tensor<f32> reducer(%a: tensor<f32>, %b: tensor<f32>) {{ %s = stablehlo.add %a, %b : tensor<f32>; stablehlo.return %s : tensor<f32> }}"
            );
            let _ = writeln!(
                body,
                "{indent}{neg_inv_b} = stablehlo.constant dense<{:.8e}> : tensor<f32>",
                -1.0 / (b as f32)
            );
            let _ = writeln!(
                body,
                "{indent}{loss} = stablehlo.multiply {sum_loss}, {neg_inv_b} : tensor<f32>"
            );
            let _ = writeln!(
                body,
                "{indent}{p_minus_t} = stablehlo.subtract {probs}, {targets_var} : {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{inv_b} = stablehlo.constant dense<{:.8e}> : tensor<f32>",
                1.0 / (b as f32)
            );
            let _ = writeln!(
                body,
                "{indent}{inv_b_bcast} = stablehlo.broadcast_in_dim {inv_b}, dims = [] : (tensor<f32>) -> {logits_ty}"
            );
            let _ = writeln!(
                body,
                "{indent}{dlogits} = stablehlo.multiply {p_minus_t}, {inv_b_bcast} : {logits_ty}"
            );
            (loss, dlogits)
        }
    };

    // --- 3. Backward pass ---
    let mut dw_vars = vec![String::new(); n_layers];
    let mut db_vars = vec![String::new(); n_layers];

    for l in (0..n_layers).rev() {
        let d_in = graph.layer_dims[l];
        let d_out_dim = graph.layer_dims[l + 1];
        let h_in = &layer_inputs[l];
        let wl = &w_vars[l];

        let dw = next_id("dw");
        let _ = writeln!(
            body,
            "{indent}{dw} = stablehlo.dot_general {h_in}, {d_out}, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<{b}x{d_in}xf32>, tensor<{b}x{d_out_dim}xf32>) -> tensor<{d_in}x{d_out_dim}xf32>"
        );
        dw_vars[l] = dw;

        let zb = next_id("zb");
        let db = next_id("db");
        let _ = writeln!(
            body,
            "{indent}{zb} = stablehlo.constant dense<0.0> : tensor<f32>"
        );
        let _ = writeln!(
            body,
            "{indent}{db} = stablehlo.reduce({d_out} init: {zb}) across dimensions = [0] : (tensor<{b}x{d_out_dim}xf32>, tensor<f32>) -> tensor<{d_out_dim}xf32> reducer(%a: tensor<f32>, %b: tensor<f32>) {{ %s = stablehlo.add %a, %b : tensor<f32>; stablehlo.return %s : tensor<f32> }}"
        );
        db_vars[l] = db;

        if l > 0 {
            let dh = next_id("dh");
            let _ = writeln!(
                body,
                "{indent}{dh} = stablehlo.dot_general {d_out}, {wl}, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<{b}x{d_out_dim}xf32>, tensor<{d_in}x{d_out_dim}xf32>) -> tensor<{b}x{d_in}xf32>"
            );
            let pre_prev = &layer_pre_acts[l - 1];
            let post_prev = &layer_post_acts[l - 1];
            let ty_prev = format!("tensor<{b}x{d_in}xf32>");
            let pred_ty = format!("tensor<{b}x{d_in}xi1>");

            d_out = match graph.activation {
                TpuActivation::Relu => {
                    let z = next_id("z_bwd");
                    let mask = next_id("mask");
                    let out = next_id("dpre");
                    let _ = writeln!(
                        body,
                        "{indent}{z} = stablehlo.constant dense<0.0> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{mask} = stablehlo.compare GT, {pre_prev}, {z}, FLOAT : ({ty_prev}, {ty_prev}) -> {pred_ty}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.select {mask}, {dh}, {z} : {pred_ty}, {ty_prev}"
                    );
                    out
                }
                TpuActivation::Tanh => {
                    let one = next_id("one_bwd");
                    let sq = next_id("tanh_sq");
                    let deriv = next_id("tanh_d");
                    let out = next_id("dpre");
                    let _ = writeln!(
                        body,
                        "{indent}{one} = stablehlo.constant dense<1.0> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{sq} = stablehlo.multiply {post_prev}, {post_prev} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{deriv} = stablehlo.subtract {one}, {sq} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.multiply {dh}, {deriv} : {ty_prev}"
                    );
                    out
                }
                TpuActivation::Silu => {
                    let sig = next_id("sig_bwd");
                    let one = next_id("one_bwd");
                    let oms = next_id("oms");
                    let xoms = next_id("xoms");
                    let inner = next_id("silu_in");
                    let deriv = next_id("silu_d");
                    let out = next_id("dpre");
                    let _ = writeln!(
                        body,
                        "{indent}{sig} = stablehlo.logistic {pre_prev} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{one} = stablehlo.constant dense<1.0> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{oms} = stablehlo.subtract {one}, {sig} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{xoms} = stablehlo.multiply {pre_prev}, {oms} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{inner} = stablehlo.add {one}, {xoms} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{deriv} = stablehlo.multiply {sig}, {inner} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.multiply {dh}, {deriv} : {ty_prev}"
                    );
                    out
                }
                TpuActivation::Gelu => {
                    let x2 = next_id("gx2");
                    let x3 = next_id("gx3");
                    let k = next_id("gk");
                    let kx3 = next_id("gkx3");
                    let inner = next_id("ginner");
                    let c = next_id("gc");
                    let u = next_id("gu");
                    let th = next_id("gth");
                    let one = next_id("gone");
                    let half = next_id("ghalf");
                    let th2 = next_id("gth2");
                    let sech2 = next_id("gsech2");
                    let k3 = next_id("gk3");
                    let k3x2 = next_id("gk3x2");
                    let du1 = next_id("gdu1");
                    let du = next_id("gdu");
                    let term1 = next_id("gt1");
                    let term2 = next_id("gt2");
                    let term2h = next_id("gt2h");
                    let deriv_sum = next_id("gderiv_s");
                    let deriv = next_id("gderiv");
                    let out = next_id("dpre");

                    let _ = writeln!(
                        body,
                        "{indent}{x2} = stablehlo.multiply {pre_prev}, {pre_prev} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{x3} = stablehlo.multiply {x2}, {pre_prev} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{k} = stablehlo.constant dense<4.471500e-02> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{kx3} = stablehlo.multiply {k}, {x3} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{inner} = stablehlo.add {pre_prev}, {kx3} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{c} = stablehlo.constant dense<7.978846e-01> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{u} = stablehlo.multiply {c}, {inner} : {ty_prev}"
                    );
                    let _ = writeln!(body, "{indent}{th} = stablehlo.tanh {u} : {ty_prev}");
                    let _ = writeln!(
                        body,
                        "{indent}{one} = stablehlo.constant dense<1.0> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{half} = stablehlo.constant dense<0.5> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{th2} = stablehlo.multiply {th}, {th} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{sech2} = stablehlo.subtract {one}, {th2} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{k3} = stablehlo.constant dense<1.341450e-01> : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{k3x2} = stablehlo.multiply {k3}, {x2} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{du1} = stablehlo.add {one}, {k3x2} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{du} = stablehlo.multiply {c}, {du1} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{term1} = stablehlo.add {one}, {th} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{term2} = stablehlo.multiply {sech2}, {du} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{term2h} = stablehlo.multiply {pre_prev}, {term2} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{deriv_sum} = stablehlo.add {term1}, {term2h} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{deriv} = stablehlo.multiply {half}, {deriv_sum} : {ty_prev}"
                    );
                    let _ = writeln!(
                        body,
                        "{indent}{out} = stablehlo.multiply {dh}, {deriv} : {ty_prev}"
                    );
                    out
                }
            };
        }
    }

    // --- 4. Fused AdamW updates ---
    let mut emit_adamw = |param: &str,
                          grad: &str,
                          m_var: &str,
                          v_var: &str,
                          ty: &str,
                          wd: f32|
     -> (String, String, String) {
        let b1 = next_id("b1");
        let omb1 = next_id("omb1");
        let b2 = next_id("b2");
        let omb2 = next_id("omb2");
        let m1 = next_id("m1");
        let m2 = next_id("m2");
        let m_new = next_id("m_new");
        let g2 = next_id("g2");
        let v1 = next_id("v1");
        let v2 = next_id("v2");
        let v_new = next_id("v_new");
        let ib1_b = next_id("ib1_b");
        let ib2_b = next_id("ib2_b");
        let m_hat = next_id("m_hat");
        let v_hat = next_id("v_hat");
        let sqv = next_id("sqv");
        let eps_c = next_id("eps");
        let denom = next_id("denom");
        let step_ratio = next_id("ratio");
        let lr_c = next_id("lr");
        let delta = next_id("delta");
        let decay_c = next_id("decay");
        let p_dec = next_id("p_dec");
        let p_new = next_id("p_new");

        let _ = writeln!(
            body,
            "{indent}{b1} = stablehlo.constant dense<{:.8e}> : {ty}",
            graph.beta1
        );
        let _ = writeln!(
            body,
            "{indent}{omb1} = stablehlo.constant dense<{:.8e}> : {ty}",
            1.0 - graph.beta1
        );
        let _ = writeln!(
            body,
            "{indent}{b2} = stablehlo.constant dense<{:.8e}> : {ty}",
            graph.beta2
        );
        let _ = writeln!(
            body,
            "{indent}{omb2} = stablehlo.constant dense<{:.8e}> : {ty}",
            1.0 - graph.beta2
        );
        let _ = writeln!(
            body,
            "{indent}{m1} = stablehlo.multiply {b1}, {m_var} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{m2} = stablehlo.multiply {omb1}, {grad} : {ty}"
        );
        let _ = writeln!(body, "{indent}{m_new} = stablehlo.add {m1}, {m2} : {ty}");
        let _ = writeln!(
            body,
            "{indent}{g2} = stablehlo.multiply {grad}, {grad} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{v1} = stablehlo.multiply {b2}, {v_var} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{v2} = stablehlo.multiply {omb2}, {g2} : {ty}"
        );
        let _ = writeln!(body, "{indent}{v_new} = stablehlo.add {v1}, {v2} : {ty}");

        let _ = writeln!(
            body,
            "{indent}{ib1_b} = stablehlo.broadcast_in_dim {inv_bc1_var}, dims = [] : (tensor<f32>) -> {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{ib2_b} = stablehlo.broadcast_in_dim {inv_bc2_var}, dims = [] : (tensor<f32>) -> {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{m_hat} = stablehlo.multiply {m_new}, {ib1_b} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{v_hat} = stablehlo.multiply {v_new}, {ib2_b} : {ty}"
        );
        let _ = writeln!(body, "{indent}{sqv} = stablehlo.sqrt {v_hat} : {ty}");
        let _ = writeln!(
            body,
            "{indent}{eps_c} = stablehlo.constant dense<{:.8e}> : {ty}",
            graph.eps
        );
        let _ = writeln!(
            body,
            "{indent}{denom} = stablehlo.add {sqv}, {eps_c} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{step_ratio} = stablehlo.divide {m_hat}, {denom} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{lr_c} = stablehlo.constant dense<{:.8e}> : {ty}",
            graph.lr
        );
        let _ = writeln!(
            body,
            "{indent}{delta} = stablehlo.multiply {lr_c}, {step_ratio} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{decay_c} = stablehlo.constant dense<{:.8e}> : {ty}",
            1.0 - graph.lr * wd
        );
        let _ = writeln!(
            body,
            "{indent}{p_dec} = stablehlo.multiply {param}, {decay_c} : {ty}"
        );
        let _ = writeln!(
            body,
            "{indent}{p_new} = stablehlo.subtract {p_dec}, {delta} : {ty}"
        );
        (p_new, m_new, v_new)
    };

    let mut new_w = Vec::with_capacity(n_layers);
    let mut new_b = Vec::with_capacity(n_layers);
    let mut new_mw = Vec::with_capacity(n_layers);
    let mut new_vw = Vec::with_capacity(n_layers);
    let mut new_mb = Vec::with_capacity(n_layers);
    let mut new_vb = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out_dim = graph.layer_dims[l + 1];
        let w_ty = format!("tensor<{d_in}x{d_out_dim}xf32>");
        let b_ty = format!("tensor<{d_out_dim}xf32>");
        let (wn, mwn, vwn) = emit_adamw(
            &w_vars[l],
            &dw_vars[l],
            &mw_vars[l],
            &vw_vars[l],
            &w_ty,
            graph.weight_decay,
        );
        let (bn, mbn, vbn) = emit_adamw(
            &b_vars[l],
            &db_vars[l],
            &mb_vars[l],
            &vb_vars[l],
            &b_ty,
            0.0,
        );
        new_w.push(wn);
        new_b.push(bn);
        new_mw.push(mwn);
        new_vw.push(vwn);
        new_mb.push(mbn);
        new_vb.push(vbn);
    }

    (loss_var, new_w, new_b, new_mw, new_vw, new_mb, new_vb)
}

/// Emit a single fused StableHLO MLIR program implementing Forward + Loss + Backward + AdamW.
///
/// Function signature:
/// `%x`, `%targets` (one-hot `[B, C]` for CE or `[B, C]` for MSE), `%inv_bc1`, `%inv_bc2`,
/// followed by per-layer `%w_i`, `%b_i`, `%mw_i`, `%vw_i`, `%mb_i`, `%vb_i`.
/// Returns `(loss_scalar, updated w_i..., updated b_i..., updated mw_i..., updated vw_i..., updated mb_i..., updated vb_i...)`.
pub fn emit_mlp_train_step_mlir(graph: &TpuStepGraph) -> OchreResult<String> {
    let n_layers = graph.num_layers();
    let b = graph.batch_size;
    let in_dim = graph.layer_dims[0];
    let out_dim = graph.layer_dims[n_layers];
    let elem = match graph.dtype {
        TpuDType::F32 => "f32",
        TpuDType::BF16 => "bf16",
    };

    let mut args = Vec::new();
    args.push(format!("%x: tensor<{b}x{in_dim}x{elem}>"));
    args.push(format!("%targets: tensor<{b}x{out_dim}xf32>"));
    args.push("%inv_bc1: tensor<f32>".to_string());
    args.push("%inv_bc2: tensor<f32>".to_string());

    let mut w_vars = Vec::with_capacity(n_layers);
    let mut b_vars = Vec::with_capacity(n_layers);
    let mut mw_vars = Vec::with_capacity(n_layers);
    let mut vw_vars = Vec::with_capacity(n_layers);
    let mut mb_vars = Vec::with_capacity(n_layers);
    let mut vb_vars = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        args.push(format!("%w{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%b{l}: tensor<{d_out}xf32>"));
        args.push(format!("%mw{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%vw{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%mb{l}: tensor<{d_out}xf32>"));
        args.push(format!("%vb{l}: tensor<{d_out}xf32>"));
        w_vars.push(format!("%w{l}"));
        b_vars.push(format!("%b{l}"));
        mw_vars.push(format!("%mw{l}"));
        vw_vars.push(format!("%vw{l}"));
        mb_vars.push(format!("%mb{l}"));
        vb_vars.push(format!("%vb{l}"));
    }

    let mut ret_types = vec!["tensor<f32>".to_string()];
    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_in}x{d_out}xf32>"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_out}xf32>"));
    }
    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_in}x{d_out}xf32>"));
    }
    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_in}x{d_out}xf32>"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_out}xf32>"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        ret_types.push(format!("tensor<{d_out}xf32>"));
    }

    let mut body = String::new();
    let mut id_counter: usize = 0;
    let (loss_var, new_w, new_b, new_mw, new_vw, new_mb, new_vb) = emit_mlp_step_ops(
        graph,
        &mut body,
        "    ",
        &mut id_counter,
        "%x",
        "%targets",
        "%inv_bc1",
        "%inv_bc2",
        &w_vars,
        &b_vars,
        &mw_vars,
        &vw_vars,
        &mb_vars,
        &vb_vars,
    );

    let mut ret_vals = vec![loss_var];
    ret_vals.extend(new_w);
    ret_vals.extend(new_b);
    ret_vals.extend(new_mw);
    ret_vals.extend(new_vw);
    ret_vals.extend(new_mb);
    ret_vals.extend(new_vb);

    Ok(format!(
        "module @ochreml_tpu_step {{\n  func.func @main({}) -> ({}) {{\n{}    return {} : {}\n  }}\n}}\n",
        args.join(", "),
        ret_types.join(", "),
        body,
        ret_vals.join(", "),
        ret_types.join(", "),
    ))
}

/// Emit a single StableHLO MLIR program wrapping `num_steps` consecutive MLP training steps
/// inside a `stablehlo.while` loop so TPU executes multiple steps per PJRT invocation with
/// zero host round-trips per step.
///
/// Function signature:
/// `%x_steps: tensor<N x B x D_in x {f32|bf16}>`,
/// `%target_steps: tensor<N x B x D_out x f32>`,
/// `%init_b1_pow: tensor<f32>`, `%init_b2_pow: tensor<f32>`,
/// followed by per-layer `%w_i`, `%b_i`, `%mw_i`, `%vw_i`, `%mb_i`, `%vb_i`.
/// Returns `(mean_loss, updated w_i..., updated b_i..., updated mw_i..., updated vw_i..., updated mb_i..., updated vb_i...)`.
pub fn emit_mlp_multi_step_while_mlir(graph: &TpuStepGraph, num_steps: usize) -> String {
    let steps = num_steps.max(1);
    let n_layers = graph.num_layers();
    let b = graph.batch_size;
    let in_dim = graph.layer_dims[0];
    let out_dim = graph.layer_dims[n_layers];
    let elem = match graph.dtype {
        TpuDType::F32 => "f32",
        TpuDType::BF16 => "bf16",
    };

    let mut args = Vec::new();
    args.push(format!("%x_steps: tensor<{steps}x{b}x{in_dim}x{elem}>"));
    args.push(format!("%target_steps: tensor<{steps}x{b}x{out_dim}xf32>"));
    args.push("%init_b1_pow: tensor<f32>".to_string());
    args.push("%init_b2_pow: tensor<f32>".to_string());

    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        args.push(format!("%w{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%b{l}: tensor<{d_out}xf32>"));
        args.push(format!("%mw{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%vw{l}: tensor<{d_in}x{d_out}xf32>"));
        args.push(format!("%mb{l}: tensor<{d_out}xf32>"));
        args.push(format!("%vb{l}: tensor<{d_out}xf32>"));
    }

    // Loop-carried state:
    // (iter: i32, loss_sum: f32, b1_pow: f32, b2_pow: f32, w0.., b0.., mw0.., vw0.., mb0.., vb0..)
    let mut state_bindings = vec![
        "%iter = %c0_i32".to_string(),
        "%loss_sum = %c0_f32".to_string(),
        "%b1_pow = %init_b1_pow".to_string(),
        "%b2_pow = %init_b2_pow".to_string(),
    ];
    let mut state_types = vec![
        "tensor<i32>".to_string(),
        "tensor<f32>".to_string(),
        "tensor<f32>".to_string(),
        "tensor<f32>".to_string(),
    ];
    let mut ret_types = vec!["tensor<f32>".to_string()];

    let mut lw_vars = Vec::with_capacity(n_layers);
    let mut lb_vars = Vec::with_capacity(n_layers);
    let mut lmw_vars = Vec::with_capacity(n_layers);
    let mut lvw_vars = Vec::with_capacity(n_layers);
    let mut lmb_vars = Vec::with_capacity(n_layers);
    let mut lvb_vars = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_in}x{d_out}xf32>");
        state_bindings.push(format!("%lw{l} = %w{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lw_vars.push(format!("%lw{l}"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_out}xf32>");
        state_bindings.push(format!("%lb{l} = %b{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lb_vars.push(format!("%lb{l}"));
    }
    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_in}x{d_out}xf32>");
        state_bindings.push(format!("%lmw{l} = %mw{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lmw_vars.push(format!("%lmw{l}"));
    }
    for l in 0..n_layers {
        let d_in = graph.layer_dims[l];
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_in}x{d_out}xf32>");
        state_bindings.push(format!("%lvw{l} = %vw{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lvw_vars.push(format!("%lvw{l}"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_out}xf32>");
        state_bindings.push(format!("%lmb{l} = %mb{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lmb_vars.push(format!("%lmb{l}"));
    }
    for l in 0..n_layers {
        let d_out = graph.layer_dims[l + 1];
        let ty = format!("tensor<{d_out}xf32>");
        state_bindings.push(format!("%lvb{l} = %vb{l}"));
        state_types.push(ty.clone());
        ret_types.push(ty);
        lvb_vars.push(format!("%lvb{l}"));
    }

    let n_state = state_types.len();
    let mut do_body = String::new();
    let mut id_counter: usize = 0;

    // Slice step batch from 3D [steps, B, D] -> 2D [B, D]
    let _ = writeln!(
        do_body,
        "      %zero_idx = stablehlo.constant dense<0> : tensor<i32>"
    );
    let _ = writeln!(
        do_body,
        "      %x_slice3 = stablehlo.dynamic_slice %x_steps, %iter, %zero_idx, %zero_idx, sizes = [1, {b}, {in_dim}] : (tensor<{steps}x{b}x{in_dim}x{elem}>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{b}x{in_dim}x{elem}>"
    );
    let _ = writeln!(
        do_body,
        "      %x_step = stablehlo.reshape %x_slice3 : (tensor<1x{b}x{in_dim}x{elem}>) -> tensor<{b}x{in_dim}x{elem}>"
    );
    let _ = writeln!(
        do_body,
        "      %t_slice3 = stablehlo.dynamic_slice %target_steps, %iter, %zero_idx, %zero_idx, sizes = [1, {b}, {out_dim}] : (tensor<{steps}x{b}x{out_dim}xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{b}x{out_dim}xf32>"
    );
    let _ = writeln!(
        do_body,
        "      %t_step = stablehlo.reshape %t_slice3 : (tensor<1x{b}x{out_dim}xf32>) -> tensor<{b}x{out_dim}xf32>"
    );

    // Update AdamW bias-correction powers inside the while loop
    let _ = writeln!(
        do_body,
        "      %c_one_f32 = stablehlo.constant dense<1.0> : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %c_beta1 = stablehlo.constant dense<{:.8e}> : tensor<f32>",
        graph.beta1
    );
    let _ = writeln!(
        do_body,
        "      %c_beta2 = stablehlo.constant dense<{:.8e}> : tensor<f32>",
        graph.beta2
    );
    let _ = writeln!(
        do_body,
        "      %b1_pow_next = stablehlo.multiply %b1_pow, %c_beta1 : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %b2_pow_next = stablehlo.multiply %b2_pow, %c_beta2 : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %bc1_den = stablehlo.subtract %c_one_f32, %b1_pow_next : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %bc2_den = stablehlo.subtract %c_one_f32, %b2_pow_next : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %step_inv_bc1 = stablehlo.divide %c_one_f32, %bc1_den : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %step_inv_bc2 = stablehlo.divide %c_one_f32, %bc2_den : tensor<f32>"
    );

    let (step_loss, new_w, new_b, new_mw, new_vw, new_mb, new_vb) = emit_mlp_step_ops(
        graph,
        &mut do_body,
        "      ",
        &mut id_counter,
        "%x_step",
        "%t_step",
        "%step_inv_bc1",
        "%step_inv_bc2",
        &lw_vars,
        &lb_vars,
        &lmw_vars,
        &lvw_vars,
        &lmb_vars,
        &lvb_vars,
    );

    let _ = writeln!(
        do_body,
        "      %loss_sum_next = stablehlo.add %loss_sum, {step_loss} : tensor<f32>"
    );
    let _ = writeln!(
        do_body,
        "      %c1_i32 = stablehlo.constant dense<1> : tensor<i32>"
    );
    let _ = writeln!(
        do_body,
        "      %iter_next = stablehlo.add %iter, %c1_i32 : tensor<i32>"
    );

    let mut yield_vals = vec![
        "%iter_next".to_string(),
        "%loss_sum_next".to_string(),
        "%b1_pow_next".to_string(),
        "%b2_pow_next".to_string(),
    ];
    yield_vals.extend(new_w);
    yield_vals.extend(new_b);
    yield_vals.extend(new_mw);
    yield_vals.extend(new_vw);
    yield_vals.extend(new_mb);
    yield_vals.extend(new_vb);

    let mut final_ret_vals = vec!["%mean_loss".to_string()];
    for idx in 4..n_state {
        final_ret_vals.push(format!("%while_res#{idx}"));
    }

    format!(
        "module @ochreml_tpu_multi_step {{\n\
         \x20 func.func @main({}) -> ({}) {{\n\
         \x20   %c0_i32 = stablehlo.constant dense<0> : tensor<i32>\n\
         \x20   %c0_f32 = stablehlo.constant dense<0.0> : tensor<f32>\n\
         \x20   %while_res:{n_state} = stablehlo.while({}) : {}\n\
         \x20   cond {{\n\
         \x20     %c_steps = stablehlo.constant dense<{steps}> : tensor<i32>\n\
         \x20     %cond = stablehlo.compare LT, %iter, %c_steps, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n\
         \x20     stablehlo.return %cond : tensor<i1>\n\
         \x20   }} do {{\n\
         {}\
         \x20     stablehlo.return {} : {}\n\
         \x20   }}\n\
         \x20   %c_steps_f32 = stablehlo.constant dense<{:.8e}> : tensor<f32>\n\
         \x20   %mean_loss = stablehlo.divide %while_res#1, %c_steps_f32 : tensor<f32>\n\
         \x20   return {} : {}\n\
         \x20 }}\n\
         }}\n",
        args.join(", "),
        ret_types.join(", "),
        state_bindings.join(", "),
        state_types.join(", "),
        do_body,
        yield_vals.join(", "),
        state_types.join(", "),
        steps as f32,
        final_ret_vals.join(", "),
        ret_types.join(", "),
    )
}

/// Execute a 1-graph compiled MLP training step on TPU via PJRT.
/// Returns `None` if PJRT TPU runtime is unavailable on this host.
#[allow(clippy::too_many_arguments)]
pub fn execute_tpu_mlp_train_step(
    device_id: usize,
    graph: &TpuStepGraph,
    x: &[f32],
    targets: &[f32],
    step: usize,
    weights: &[Vec<f32>],
    biases: &[Vec<f32>],
    m_weights: &[Vec<f32>],
    v_weights: &[Vec<f32>],
    m_biases: &[Vec<f32>],
    v_biases: &[Vec<f32>],
) -> Option<OchreResult<TpuStepOutput>> {
    let ctx = get_tpu_nn_ctx()?;
    let device = ctx
        .devices
        .get(device_id)
        .or_else(|| ctx.devices.first())?
        .0;
    // SAFETY: `ctx.api.0` points to valid static PJRT C API table.
    let api = unsafe { &*ctx.api.0 };

    let mlir = match graph.emit_mlir() {
        Ok(m) => m,
        Err(e) => return Some(Err(e)),
    };
    let exec = get_or_compile(ctx, &graph.cache_key(), &mlir)?;
    let n_layers = graph.num_layers();
    if weights.len() != n_layers
        || biases.len() != n_layers
        || m_weights.len() != n_layers
        || v_weights.len() != n_layers
        || m_biases.len() != n_layers
        || v_biases.len() != n_layers
    {
        return Some(Err(OchreError::InvalidParameter(
            "execute_tpu_mlp_train_step layer count mismatch ( >_< )".to_string(),
        )));
    }

    let inv_bc1 = (1.0 / (1.0 - (graph.beta1 as f64).powi(step.max(1) as i32))) as f32;
    let inv_bc2 = (1.0 / (1.0 - (graph.beta2 as f64).powi(step.max(1) as i32))) as f32;
    let in_dim = graph.layer_dims[0];
    let out_dim = graph.layer_dims[n_layers];

    let mut bufs: Vec<*mut PjrtBuffer> = Vec::with_capacity(4 + 6 * n_layers);
    let x_dims = [graph.batch_size as i64, in_dim as i64];
    let x_buf = match graph.dtype {
        TpuDType::F32 => upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            x.as_ptr().cast(),
            PJRT_BUFFER_TYPE_F32,
            &x_dims,
        ),
        TpuDType::BF16 => {
            let x_bf16: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
            upload_raw_buffer(
                api,
                ctx.client.0,
                device,
                x_bf16.as_ptr().cast(),
                PJRT_BUFFER_TYPE_BF16,
                &x_dims,
            )
        }
    };
    let xb = x_buf?;
    bufs.push(xb);

    let t_dims = [graph.batch_size as i64, out_dim as i64];
    let Some(tb) = upload_raw_buffer(
        api,
        ctx.client.0,
        device,
        targets.as_ptr().cast(),
        PJRT_BUFFER_TYPE_F32,
        &t_dims,
    ) else {
        for b in bufs {
            destroy_buffer(api, b);
        }
        return None;
    };
    bufs.push(tb);

    for scalar in [inv_bc1, inv_bc2] {
        let Some(sb) = upload_raw_buffer(
            api,
            ctx.client.0,
            device,
            (&scalar as *const f32).cast(),
            PJRT_BUFFER_TYPE_F32,
            &[],
        ) else {
            for b in bufs {
                destroy_buffer(api, b);
            }
            return None;
        };
        bufs.push(sb);
    }

    for l in 0..n_layers {
        let w_dims = [graph.layer_dims[l] as i64, graph.layer_dims[l + 1] as i64];
        let b_dims = [graph.layer_dims[l + 1] as i64];
        let uploads: [(&[f32], &[i64]); 6] = [
            (&weights[l], &w_dims),
            (&biases[l], &b_dims),
            (&m_weights[l], &w_dims),
            (&v_weights[l], &w_dims),
            (&m_biases[l], &b_dims),
            (&v_biases[l], &b_dims),
        ];
        for (slice, dims) in uploads {
            let Some(ub) = upload_raw_buffer(
                api,
                ctx.client.0,
                device,
                slice.as_ptr().cast(),
                PJRT_BUFFER_TYPE_F32,
                dims,
            ) else {
                for b in bufs {
                    destroy_buffer(api, b);
                }
                return None;
            };
            bufs.push(ub);
        }
    }

    let n_outputs = 1 + 6 * n_layers;
    let mut out_ptrs = vec![ptr::null_mut(); n_outputs];
    let arg_lists: [*const *mut PjrtBuffer; 1] = [bufs.as_ptr()];
    let out_lists: [*mut *mut PjrtBuffer; 1] = [out_ptrs.as_mut_ptr()];
    let mut complete_event: *mut PjrtEvent = ptr::null_mut();
    let options = PjrtExecuteOptions {
        struct_size: std::mem::size_of::<PjrtExecuteOptions>(),
        extension_start: ptr::null_mut(),
        send_callbacks: ptr::null_mut(),
        recv_callbacks: ptr::null_mut(),
        num_send_ops: 0,
        num_recv_ops: 0,
        launch_id: 0,
        non_donatable_input_indices: ptr::null(),
        num_non_donatable_input_indices: 0,
        context: ptr::null_mut(),
        call_location: ptr::null(),
        num_tasks: 0,
        task_ids: ptr::null_mut(),
        incarnation_ids: ptr::null_mut(),
        multi_slice_config: ptr::null_mut(),
        use_major_to_minor_data_layout_for_callbacks: false,
        hlo_output_callbacks: ptr::null_mut(),
        num_hlo_output_callbacks: 0,
        custom_options: ptr::null(),
        num_custom_options: 0,
    };
    let mut exec_args = PjrtLoadedExecutableExecuteArgs {
        struct_size: std::mem::size_of::<PjrtLoadedExecutableExecuteArgs>(),
        extension_start: ptr::null_mut(),
        executable: exec,
        options: &options,
        argument_lists: arg_lists.as_ptr(),
        num_devices: 1,
        num_args: bufs.len(),
        output_lists: out_lists.as_ptr(),
        device_complete_events: &mut complete_event,
        execute_device: ptr::null_mut(),
    };

    let Some(exec_fn) = api.loaded_executable_execute else {
        for b in bufs {
            destroy_buffer(api, b);
        }
        return None;
    };
    // SAFETY: `exec` and all `bufs` belong to `ctx.client` and match compiled signature.
    let err = unsafe { exec_fn(&mut exec_args) };
    for b in bufs {
        destroy_buffer(api, b);
    }

    if !check_pjrt_err(api, err) || !await_and_destroy_event(api, complete_event) {
        for ob in out_ptrs {
            destroy_buffer(api, ob);
        }
        return None;
    }

    let mut loss_val = 0.0f32;
    let mut ok = !out_ptrs[0].is_null()
        && download_buffer(api, out_ptrs[0], (&mut loss_val as *mut f32).cast(), 4, &[]);

    let mut read_vec = |idx: usize, len: usize, minor_to_major: &[i64]| -> Vec<f32> {
        let mut buf = vec![0.0f32; len];
        if ok && !out_ptrs[idx].is_null() {
            ok = download_buffer(
                api,
                out_ptrs[idx],
                buf.as_mut_ptr().cast(),
                len * 4,
                minor_to_major,
            );
        } else {
            ok = false;
        }
        buf
    };

    let mut out_w = Vec::with_capacity(n_layers);
    let mut out_b = Vec::with_capacity(n_layers);
    let mut out_mw = Vec::with_capacity(n_layers);
    let mut out_vw = Vec::with_capacity(n_layers);
    let mut out_mb = Vec::with_capacity(n_layers);
    let mut out_vb = Vec::with_capacity(n_layers);

    let mut cursor = 1;
    for l in 0..n_layers {
        let w_len = graph.layer_dims[l] * graph.layer_dims[l + 1];
        out_w.push(read_vec(cursor, w_len, &[1, 0]));
        cursor += 1;
    }
    for l in 0..n_layers {
        out_b.push(read_vec(cursor, graph.layer_dims[l + 1], &[0]));
        cursor += 1;
    }
    for l in 0..n_layers {
        let w_len = graph.layer_dims[l] * graph.layer_dims[l + 1];
        out_mw.push(read_vec(cursor, w_len, &[1, 0]));
        cursor += 1;
    }
    for l in 0..n_layers {
        let w_len = graph.layer_dims[l] * graph.layer_dims[l + 1];
        out_vw.push(read_vec(cursor, w_len, &[1, 0]));
        cursor += 1;
    }
    for l in 0..n_layers {
        out_mb.push(read_vec(cursor, graph.layer_dims[l + 1], &[0]));
        cursor += 1;
    }
    for l in 0..n_layers {
        out_vb.push(read_vec(cursor, graph.layer_dims[l + 1], &[0]));
        cursor += 1;
    }

    for ob in out_ptrs {
        destroy_buffer(api, ob);
    }
    if !ok {
        return None;
    }

    // Keep bf16 import live even when TPU hardware is absent on host
    let _ = bf16_to_f32(f32_to_bf16(loss_val));

    Some(Ok(TpuStepOutput {
        loss: loss_val,
        weights: out_w,
        biases: out_b,
        m_weights: out_mw,
        v_weights: out_vw,
        m_biases: out_mb,
        v_biases: out_vb,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_batched_dot_general_mlir_f32_and_bf16() {
        let mlir_f32 = emit_batched_dot_general_mlir(4, 32, 64, 16, false, true, false);
        assert!(mlir_f32.contains("func.func @main"));
        assert!(mlir_f32.contains("stablehlo.dot_general"));
        assert!(mlir_f32.contains("batching_dims = [0] x [0]"));
        assert!(mlir_f32.contains("contracting_dims = [2] x [2]"));
        assert!(mlir_f32.contains("tensor<4x32x64xf32>"));
        assert!(mlir_f32.contains("tensor<4x16x64xf32>"));
        assert!(mlir_f32.contains("tensor<4x32x16xf32>"));

        let mlir_bf16 = emit_batched_dot_general_mlir(2, 16, 32, 16, true, false, true);
        assert!(mlir_bf16.contains("tensor<2x32x16xbf16>"));
        assert!(mlir_bf16.contains("contracting_dims = [1] x [1]"));
        assert!(mlir_bf16.contains("-> tensor<2x16x16xf32>"));
    }

    #[test]
    fn test_emit_mlp_train_step_mlir_ce_and_mse() {
        let graph_ce = TpuStepGraph::new(
            64,
            vec![784, 256, 10],
            TpuActivation::Gelu,
            TpuLossKind::CrossEntropy,
            TpuDType::BF16,
            1e-3,
            0.9,
            0.999,
            1e-8,
            0.01,
        )
        .unwrap();
        let mlir_ce = graph_ce.emit_mlir().unwrap();
        assert!(mlir_ce.contains("module @ochreml_tpu_step"));
        assert!(mlir_ce.contains("stablehlo.dot_general"));
        assert!(mlir_ce.contains("stablehlo.exponential"));
        assert!(mlir_ce.contains("stablehlo.tanh"));
        assert!(mlir_ce.contains("stablehlo.sqrt"));
        assert!(mlir_ce.contains("tensor<64x784xbf16>"));
        assert!(mlir_ce.contains("tensor<784x256xf32>"));
        assert!(mlir_ce.contains("tensor<256x10xf32>"));

        let graph_mse = TpuStepGraph::new(
            32,
            vec![16, 32, 4],
            TpuActivation::Relu,
            TpuLossKind::Mse,
            TpuDType::F32,
            1e-2,
            0.9,
            0.999,
            1e-8,
            0.0,
        )
        .unwrap();
        let mlir_mse = graph_mse.emit_mlir().unwrap();
        assert!(mlir_mse.contains("stablehlo.maximum"));
        assert!(mlir_mse.contains("stablehlo.subtract"));
        assert!(mlir_mse.contains("stablehlo.select"));
    }

    #[test]
    fn test_emit_mlp_multi_step_while_mlir() {
        let graph_ce = TpuStepGraph::new(
            64,
            vec![784, 256, 10],
            TpuActivation::Gelu,
            TpuLossKind::CrossEntropy,
            TpuDType::BF16,
            1e-3,
            0.9,
            0.999,
            1e-8,
            0.01,
        )
        .unwrap();
        let mlir = emit_mlp_multi_step_while_mlir(&graph_ce, 8);
        assert!(mlir.contains("module @ochreml_tpu_multi_step"));
        assert!(mlir.contains("stablehlo.while("));
        assert!(mlir.contains("cond {"));
        assert!(mlir.contains("} do {"));
        assert!(mlir.contains("stablehlo.compare LT, %iter, %c_steps, SIGNED"));
        assert!(mlir.contains("dense<8> : tensor<i32>"));
        assert!(mlir.contains("stablehlo.dynamic_slice %x_steps"));
        assert!(mlir.contains("stablehlo.dynamic_slice %target_steps"));
        assert!(mlir.contains("tensor<8x64x784xbf16>"));
        assert!(mlir.contains("tensor<8x64x10xf32>"));
        assert!(mlir.contains("stablehlo.reshape %x_slice3"));
        assert!(mlir.contains("stablehlo.dot_general"));
        assert!(mlir.contains("stablehlo.divide %while_res#1, %c_steps_f32"));
        assert_eq!(mlir.matches('{').count(), mlir.matches('}').count());

        let graph_mse = TpuStepGraph::new(
            32,
            vec![16, 32, 4],
            TpuActivation::Silu,
            TpuLossKind::Mse,
            TpuDType::F32,
            1e-2,
            0.9,
            0.999,
            1e-8,
            0.0,
        )
        .unwrap();
        let mlir_mse = graph_mse.emit_multi_step_while_mlir(4);
        assert!(mlir_mse.contains("stablehlo.while("));
        assert!(mlir_mse.contains("dense<4> : tensor<i32>"));
        assert!(mlir_mse.contains("stablehlo.logistic"));
        assert!(mlir_mse.contains("tensor<4x32x16xf32>"));
        assert_eq!(mlir_mse.matches('{').count(), mlir_mse.matches('}').count());
    }

    #[test]
    fn test_tpu_nn_graceful_fallback_without_hardware() {
        let small = vec![1.0f32; 64];
        assert!(tpu_batched_matmul_f32(0, 2, 4, 4, 4, &small, &small, false, false).is_none());
        assert!(tpu_batched_matmul_bf16(0, 2, 4, 4, 4, &small, &small, false, false).is_none());
    }
}
