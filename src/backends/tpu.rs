// OchreML Native Google TPU Backend via PJRT C API (Runtime Dynamic Loading) (*≧ω≦*)
// Feature-gated behind `tpu-pjrt` so PyPI/local wheels stay tiny and independent of XLA SDK.
// Uses `libloading` to load `libpjrt_c_api_tpu.so` / `libtpu.so` at runtime, caches `PJRT_Client`
// and compiled executables in `OnceLock`, and emits power-of-two padded StableHLO/VHLO programs.
// Any missing library, ABI mismatch, or compile failure returns `None` for seamless CPU fallback.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use crate::error::OchreResult;
use crate::matrix::Matrix;

// --- PJRT C API Minimal ABI Definitions (OpenXLA pjrt_c_api.h) ---

pub const PJRT_API_MAJOR: c_int = 0;

#[repr(C)]
pub struct PjrtExtensionBase {
    pub struct_size: usize,
    pub extension_type: c_int,
    pub next: *mut PjrtExtensionBase,
}

#[repr(C)]
pub struct PjrtApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub major_version: c_int,
    pub minor_version: c_int,
}

#[repr(C)]
pub struct PjrtError {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtClient {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtDevice {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtLoadedExecutable {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtBuffer {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtEvent {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct PjrtErrorDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub error: *mut PjrtError,
}

#[repr(C)]
pub struct PjrtErrorMessageArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub error: *const PjrtError,
    pub message: *const c_char,
    pub message_size: usize,
}

#[repr(C)]
pub struct PjrtPluginInitializeArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
}

#[repr(C)]
pub struct PjrtEventDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PjrtEventAwaitArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PjrtClientCreateArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub create_options: *const c_void,
    pub num_options: usize,
    pub kv_get_callback: *const c_void,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: *const c_void,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PjrtClient,
    pub kv_try_get_callback: *const c_void,
    pub kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
pub struct PjrtClientDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
}

#[repr(C)]
pub struct PjrtClientAddressableDevicesArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub addressable_devices: *const *mut PjrtDevice,
    pub num_addressable_devices: usize,
}

#[repr(C)]
pub struct PjrtProgram {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub code: *const c_char,
    pub code_size: usize,
    pub format: *const c_char,
    pub format_size: usize,
}

#[repr(C)]
pub struct PjrtClientCompileArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub program: *const PjrtProgram,
    pub compile_options: *const c_char,
    pub compile_options_size: usize,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PjrtClientBufferFromHostBufferArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub data: *const c_void,
    pub buffer_type: c_int, // PJRT_Buffer_Type_F32 = 11
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: c_int, // ImmutableOnlyDuringCall = 0
    pub device: *mut PjrtDevice,
    pub memory: *mut c_void,
    pub device_layout: *mut c_void,
    pub done_with_host_buffer: *mut PjrtEvent,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PjrtLoadedExecutableDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PjrtExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub send_callbacks: *mut *mut c_void,
    pub recv_callbacks: *mut *mut c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: c_int,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *mut c_void,
    pub call_location: *const c_char,
    pub num_tasks: usize,
    pub task_ids: *mut c_int,
    pub incarnation_ids: *mut i64,
    pub multi_slice_config: *mut c_void,
    pub use_major_to_minor_data_layout_for_callbacks: bool,
    pub hlo_output_callbacks: *mut c_void,
    pub num_hlo_output_callbacks: usize,
    pub custom_options: *const c_void,
    pub num_custom_options: usize,
}

#[repr(C)]
pub struct PjrtLoadedExecutableExecuteArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtLoadedExecutable,
    pub options: *const PjrtExecuteOptions,
    pub argument_lists: *const *const *mut PjrtBuffer,
    pub num_devices: usize,
    pub num_args: usize,
    pub output_lists: *const *mut *mut PjrtBuffer,
    pub device_complete_events: *mut *mut PjrtEvent,
    pub execute_device: *mut PjrtDevice,
}

#[repr(C)]
pub struct PjrtBufferDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PjrtBufferToHostBufferArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub src: *mut PjrtBuffer,
    pub host_layout: *mut c_void,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PjrtEvent,
}

// PJRT_Buffer_Type_F32 in pjrt_c_api.h
const PJRT_BUFFER_TYPE_F32: c_int = 11;
const PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL: c_int = 0;

// Prefix of PJRT_Api struct matching OpenXLA pjrt_c_api.h field order.
// Any field beyond `buffer_to_host_buffer` is not needed here.
#[repr(C)]
pub struct PjrtApiPrefix {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub pjrt_api_version: PjrtApiVersion,

    pub error_destroy: Option<unsafe extern "C" fn(*mut PjrtErrorDestroyArgs)>,
    pub error_message: Option<unsafe extern "C" fn(*mut PjrtErrorMessageArgs)>,
    pub error_get_code: *const c_void,

    pub plugin_initialize: Option<unsafe extern "C" fn(*mut PjrtPluginInitializeArgs) -> *mut PjrtError>,
    pub plugin_attributes: *const c_void,

    pub event_destroy: Option<unsafe extern "C" fn(*mut PjrtEventDestroyArgs) -> *mut PjrtError>,
    pub event_is_ready: *const c_void,
    pub event_error: *const c_void,
    pub event_await: Option<unsafe extern "C" fn(*mut PjrtEventAwaitArgs) -> *mut PjrtError>,
    pub event_on_ready: *const c_void,

    pub client_create: Option<unsafe extern "C" fn(*mut PjrtClientCreateArgs) -> *mut PjrtError>,
    pub client_destroy: Option<unsafe extern "C" fn(*mut PjrtClientDestroyArgs) -> *mut PjrtError>,
    pub client_platform_name: *const c_void,
    pub client_process_index: *const c_void,
    pub client_platform_version: *const c_void,
    pub client_devices: *const c_void,
    pub client_addressable_devices:
        Option<unsafe extern "C" fn(*mut PjrtClientAddressableDevicesArgs) -> *mut PjrtError>,
    pub client_lookup_device: *const c_void,
    pub client_lookup_addressable_device: *const c_void,
    pub client_addressable_memories: *const c_void,
    pub client_compile: Option<unsafe extern "C" fn(*mut PjrtClientCompileArgs) -> *mut PjrtError>,
    pub client_default_device_assignment: *const c_void,
    pub client_buffer_from_host_buffer:
        Option<unsafe extern "C" fn(*mut PjrtClientBufferFromHostBufferArgs) -> *mut PjrtError>,

    pub device_description_id: *const c_void,
    pub device_description_process_index: *const c_void,
    pub device_description_attributes: *const c_void,
    pub device_description_kind: *const c_void,
    pub device_description_debug_string: *const c_void,
    pub device_description_to_string: *const c_void,

    pub device_get_description: *const c_void,
    pub device_is_addressable: *const c_void,
    pub device_local_hardware_id: *const c_void,
    pub device_addressable_memories: *const c_void,
    pub device_default_memory: *const c_void,
    pub device_memory_stats: *const c_void,

    pub memory_id: *const c_void,
    pub memory_kind: *const c_void,
    pub memory_debug_string: *const c_void,
    pub memory_to_string: *const c_void,
    pub memory_addressable_by_devices: *const c_void,

    pub executable_destroy: *const c_void,
    pub executable_name: *const c_void,
    pub executable_num_replicas: *const c_void,
    pub executable_num_partitions: *const c_void,
    pub executable_num_outputs: *const c_void,
    pub executable_size_of_generated_code_in_bytes: *const c_void,
    pub executable_get_cost_analysis: *const c_void,
    pub executable_output_memory_kinds: *const c_void,
    pub executable_optimized_program: *const c_void,
    pub executable_serialize: *const c_void,

    pub loaded_executable_destroy:
        Option<unsafe extern "C" fn(*mut PjrtLoadedExecutableDestroyArgs) -> *mut PjrtError>,
    pub loaded_executable_get_executable: *const c_void,
    pub loaded_executable_addressable_devices: *const c_void,
    pub loaded_executable_delete: *const c_void,
    pub loaded_executable_is_deleted: *const c_void,
    pub loaded_executable_execute:
        Option<unsafe extern "C" fn(*mut PjrtLoadedExecutableExecuteArgs) -> *mut PjrtError>,
    pub executable_deserialize_and_load: *const c_void,
    pub loaded_executable_fingerprint: *const c_void,

    pub buffer_destroy: Option<unsafe extern "C" fn(*mut PjrtBufferDestroyArgs) -> *mut PjrtError>,
    pub buffer_element_type: *const c_void,
    pub buffer_dimensions: *const c_void,
    pub buffer_unpadded_dimensions: *const c_void,
    pub buffer_dynamic_dimension_indices: *const c_void,
    pub buffer_get_memory_layout: *const c_void,
    pub buffer_on_device_size_in_bytes: *const c_void,
    pub buffer_device: *const c_void,
    pub buffer_memory: *const c_void,
    pub buffer_delete: *const c_void,
    pub buffer_is_deleted: *const c_void,
    pub buffer_copy_to_device: *const c_void,
    pub buffer_to_host_buffer:
        Option<unsafe extern "C" fn(*mut PjrtBufferToHostBufferArgs) -> *mut PjrtError>,
}

// --- Cached Runtime Context ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TpuOpKind {
    MatMul,
    MatMulAtB,
}

struct SendExecutable(*mut PjrtLoadedExecutable);
unsafe impl Send for SendExecutable {}
unsafe impl Sync for SendExecutable {}

struct SendClient(*mut PjrtClient);
unsafe impl Send for SendClient {}
unsafe impl Sync for SendClient {}

struct SendDevice(*mut PjrtDevice);
unsafe impl Send for SendDevice {}
unsafe impl Sync for SendDevice {}

struct SendApi(*const PjrtApiPrefix);
unsafe impl Send for SendApi {}
unsafe impl Sync for SendApi {}

struct TpuContext {
    _lib: Library,
    api: SendApi,
    client: SendClient,
    devices: Vec<SendDevice>,
    lib_path: String,
    exec_cache: Mutex<HashMap<(TpuOpKind, usize, usize, usize), SendExecutable>>,
}

static TPU_CONTEXT: OnceLock<Option<TpuContext>> = OnceLock::new();

fn destroy_pjrt_error(api: &PjrtApiPrefix, err: *mut PjrtError) {
    if err.is_null() {
        return;
    }
    if let Some(destroy_fn) = api.error_destroy {
        let mut args = PjrtErrorDestroyArgs {
            struct_size: std::mem::size_of::<PjrtErrorDestroyArgs>(),
            extension_start: ptr::null_mut(),
            error: err,
        };
        unsafe { destroy_fn(&mut args) };
    }
}

fn tpu_debug_enabled() -> bool {
    matches!(std::env::var("OCHREML_DEBUG").as_deref(), Ok("1") | Ok("true"))
}

fn check_pjrt_error(api: &PjrtApiPrefix, err: *mut PjrtError) -> bool {
    if err.is_null() {
        return true;
    }
    if let Some(msg_fn) = api.error_message {
        let mut msg_args = PjrtErrorMessageArgs {
            struct_size: std::mem::size_of::<PjrtErrorMessageArgs>(),
            extension_start: ptr::null_mut(),
            error: err,
            message: ptr::null(),
            message_size: 0,
        };
        unsafe { msg_fn(&mut msg_args) };
        if !msg_args.message.is_null() && msg_args.message_size > 0 {
            let slice = unsafe {
                std::slice::from_raw_parts(msg_args.message as *const u8, msg_args.message_size)
            };
            let msg = String::from_utf8_lossy(slice);
            if tpu_debug_enabled() {
                eprintln!("[ochreml::tpu] PJRT error: {}", msg);
            }
        }
    }
    destroy_pjrt_error(api, err);
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
        let err = unsafe { await_fn(&mut args) };
        ok = check_pjrt_error(api, err);
    }
    if let Some(destroy_fn) = api.event_destroy {
        let mut args = PjrtEventDestroyArgs {
            struct_size: std::mem::size_of::<PjrtEventDestroyArgs>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let err = unsafe { destroy_fn(&mut args) };
        let _ = check_pjrt_error(api, err);
    }
    ok
}

fn candidate_pjrt_paths() -> Vec<String> {
    let mut paths = Vec::new();
    for var in ["PJRT_PLUGIN_LIBRARY_PATH", "TPU_LIBRARY_PATH"] {
        if let Ok(env_path) = std::env::var(var) {
            if !env_path.is_empty() {
                paths.push(env_path);
            }
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

fn try_init_tpu_context() -> Option<TpuContext> {
    for path in candidate_pjrt_paths() {
        let lib = match unsafe { Library::new(&path) } {
            Ok(l) => l,
            Err(_) => continue,
        };

        let get_api: Symbol<unsafe extern "C" fn() -> *const PjrtApiPrefix> =
            match unsafe { lib.get(b"GetPjrtApi\0") } {
                Ok(sym) => sym,
                Err(_) => continue,
            };

        let api_ptr = unsafe { get_api() };
        if api_ptr.is_null() {
            continue;
        }

        let api = unsafe { &*api_ptr };
        if api.struct_size < std::mem::size_of::<PjrtApiPrefix>() {
            if tpu_debug_enabled() {
                eprintln!(
                    "[ochreml::tpu] PJRT_Api struct_size mismatch in {}: got {}, need >= {}",
                    path,
                    api.struct_size,
                    std::mem::size_of::<PjrtApiPrefix>()
                );
            }
            continue;
        }
        if api.pjrt_api_version.major_version != PJRT_API_MAJOR {
            if tpu_debug_enabled() {
                eprintln!(
                    "[ochreml::tpu] PJRT_Api major_version mismatch in {}: got {}, expected {}",
                    path, api.pjrt_api_version.major_version, PJRT_API_MAJOR
                );
            }
            continue;
        }

        // Verify all required function pointers are present
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
            let err = unsafe { init_fn(&mut init_args) };
            if !check_pjrt_error(api, err) {
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
        let err = unsafe { create_fn(&mut create_args) };
        if !check_pjrt_error(api, err) || create_args.client.is_null() {
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
        let err = unsafe { dev_fn(&mut dev_args) };
        if !check_pjrt_error(api, err)
            || dev_args.addressable_devices.is_null()
            || dev_args.num_addressable_devices == 0
        {
            if let Some(destroy_fn) = api.client_destroy {
                let mut dargs = PjrtClientDestroyArgs {
                    struct_size: std::mem::size_of::<PjrtClientDestroyArgs>(),
                    extension_start: ptr::null_mut(),
                    client: create_args.client,
                };
                let _ = check_pjrt_error(api, unsafe { destroy_fn(&mut dargs) });
            }
            continue;
        }

        let devices_slice = unsafe {
            std::slice::from_raw_parts(dev_args.addressable_devices, dev_args.num_addressable_devices)
        };
        let devices = devices_slice.iter().copied().map(SendDevice).collect();

        return Some(TpuContext {
            _lib: lib,
            api: SendApi(api_ptr),
            client: SendClient(create_args.client),
            devices,
            lib_path: path,
            exec_cache: Mutex::new(HashMap::new()),
        });
    }
    None
}

fn get_tpu_context() -> Option<&'static TpuContext> {
    TPU_CONTEXT.get_or_init(try_init_tpu_context).as_ref()
}

pub fn is_available() -> bool {
    get_tpu_context().is_some()
}

pub fn tpu_info_string() -> Option<String> {
    get_tpu_context().map(|ctx| {
        format!(
            "PJRT TPU ({} addressable device(s), {})",
            ctx.devices.len(),
            ctx.lib_path
        )
    })
}

// --- Power-of-Two Padding & Minimal StableHLO / VHLO Program Emitter ---

#[inline]
pub fn next_pow2_dim(dim: usize, min_dim: usize) -> usize {
    dim.max(min_dim).next_power_of_two()
}

/// Emits a minimal StableHLO MLIR textual module for `dot_general` on f32 tensors
/// with power-of-two padded shapes so the number of compiled executables stays small.
pub fn emit_stablehlo_dot_general_mlir(
    m_pad: usize,
    k_pad: usize,
    n_pad: usize,
    transpose_a: bool,
) -> String {
    let (a_rows, a_cols, lhs_contract, lhs_batch_or_free) = if transpose_a {
        // A is (k_pad x m_pad); A^T * B contracts dim 0 of A with dim 0 of B
        (k_pad, m_pad, 0usize, 1usize)
    } else {
        // A is (m_pad x k_pad); A * B contracts dim 1 of A with dim 0 of B
        (m_pad, k_pad, 1usize, 0usize)
    };
    let _ = lhs_batch_or_free;

    format!(
        "module @ochreml_tpu {{\n\
         \x20 func.func @main(%arg0: tensor<{a_rows}x{a_cols}xf32>, %arg1: tensor<{k_pad}x{n_pad}xf32>) -> tensor<{m_pad}x{n_pad}xf32> {{\n\
         \x20   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [{lhs_contract}] x [0], precision = [HIGHEST, HIGHEST] : (tensor<{a_rows}x{a_cols}xf32>, tensor<{k_pad}x{n_pad}xf32>) -> tensor<{m_pad}x{n_pad}xf32>\n\
         \x20   return %0 : tensor<{m_pad}x{n_pad}xf32>\n\
         \x20 }}\n\
         }}\n"
    )
}

/// Builds a minimal MLIR bytecode container header (`ML\xefR`) wrapping the versioned
/// StableHLO/VHLO payload metadata for diagnostics or bytecode-aware runtimes.
pub fn emit_vhlo_bytecode_header(producer: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + producer.len());
    // Standard MLIR bytecode magic bytes: 'M', 'L', 0xEF, 'R'
    out.extend_from_slice(b"ML\xefR");
    // Bytecode format version (varint 1) + producer string length + bytes
    out.push(0x01);
    let bytes = producer.as_bytes();
    out.push((bytes.len().min(127)) as u8);
    out.extend_from_slice(&bytes[..bytes.len().min(127)]);
    out
}

fn get_or_compile_executable(
    ctx: &TpuContext,
    op: TpuOpKind,
    m_pad: usize,
    k_pad: usize,
    n_pad: usize,
) -> Option<*mut PjrtLoadedExecutable> {
    let key = (op, m_pad, k_pad, n_pad);
    {
        let cache = ctx.exec_cache.lock().ok()?;
        if let Some(exec) = cache.get(&key) {
            return Some(exec.0);
        }
    }

    let transpose_a = matches!(op, TpuOpKind::MatMulAtB);
    let mlir_src = emit_stablehlo_dot_general_mlir(m_pad, k_pad, n_pad, transpose_a);
    let code_cstr = CString::new(mlir_src).ok()?;
    let format_cstr = CString::new("mlir").ok()?;

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

    // Serialized xla::CompileOptionsProto { executable_build_options: { num_replicas: 1, num_partitions: 1 } }
    const COMPILE_OPTIONS_PROTO: [u8; 6] = [0x1A, 0x04, 0x20, 0x01, 0x28, 0x01];

    let mut compile_args = PjrtClientCompileArgs {
        struct_size: std::mem::size_of::<PjrtClientCompileArgs>(),
        extension_start: ptr::null_mut(),
        client: ctx.client.0,
        program: &program,
        compile_options: COMPILE_OPTIONS_PROTO.as_ptr() as *const c_char,
        compile_options_size: COMPILE_OPTIONS_PROTO.len(),
        executable: ptr::null_mut(),
    };

    let err = unsafe { compile_fn(&mut compile_args) };
    if !check_pjrt_error(api, err) || compile_args.executable.is_null() {
        return None;
    }

    let exec_ptr = compile_args.executable;
    if let Ok(mut cache) = ctx.exec_cache.lock() {
        cache.insert(key, SendExecutable(exec_ptr));
    }
    Some(exec_ptr)
}

fn pad_matrix_to_f32(mat: &Matrix, target_rows: usize, target_cols: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; target_rows * target_cols];
    for r in 0..mat.rows {
        let src_row = &mat.data[r * mat.cols..(r + 1) * mat.cols];
        let dst_row = &mut padded[r * target_cols..r * target_cols + mat.cols];
        for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
            *d = s as f32;
        }
    }
    padded
}

fn unpad_f32_to_matrix(buf: &[f32], padded_cols: usize, rows: usize, cols: usize) -> Matrix {
    let mut data = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let src_row = &buf[r * padded_cols..r * padded_cols + cols];
        let dst_row = &mut data[r * cols..(r + 1) * cols];
        for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
            *d = s as f64;
        }
    }
    Matrix::from_vec(rows, cols, data).expect("valid dimensions")
}

fn upload_f32_buffer(
    api: &PjrtApiPrefix,
    client: *mut PjrtClient,
    device: *mut PjrtDevice,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Option<*mut PjrtBuffer> {
    let upload_fn = api.client_buffer_from_host_buffer?;
    let dims = [rows as i64, cols as i64];
    let mut args = PjrtClientBufferFromHostBufferArgs {
        struct_size: std::mem::size_of::<PjrtClientBufferFromHostBufferArgs>(),
        extension_start: ptr::null_mut(),
        client,
        data: data.as_ptr() as *const c_void,
        buffer_type: PJRT_BUFFER_TYPE_F32,
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
    let err = unsafe { upload_fn(&mut args) };
    if !check_pjrt_error(api, err) || args.buffer.is_null() {
        let _ = await_and_destroy_event(api, args.done_with_host_buffer);
        return None;
    }
    if !await_and_destroy_event(api, args.done_with_host_buffer) {
        destroy_pjrt_buffer(api, args.buffer);
        return None;
    }
    Some(args.buffer)
}

fn destroy_pjrt_buffer(api: &PjrtApiPrefix, buffer: *mut PjrtBuffer) {
    if buffer.is_null() {
        return;
    }
    if let Some(destroy_fn) = api.buffer_destroy {
        let mut args = PjrtBufferDestroyArgs {
            struct_size: std::mem::size_of::<PjrtBufferDestroyArgs>(),
            extension_start: ptr::null_mut(),
            buffer,
        };
        let err = unsafe { destroy_fn(&mut args) };
        let _ = check_pjrt_error(api, err);
    }
}

fn execute_tpu_dot(
    device_id: usize,
    op: TpuOpKind,
    a: &Matrix,
    b: &Matrix,
) -> Option<Matrix> {
    let ctx = get_tpu_context()?;
    let device = ctx.devices.get(device_id).or_else(|| ctx.devices.first())?.0;
    let api = unsafe { &*ctx.api.0 };

    let (m, k, n) = match op {
        TpuOpKind::MatMul => (a.rows, a.cols, b.cols),
        TpuOpKind::MatMulAtB => (a.cols, a.rows, b.cols),
    };

    let m_pad = next_pow2_dim(m, 16);
    let k_pad = next_pow2_dim(k, 16);
    let n_pad = next_pow2_dim(n, 16);

    let exec = get_or_compile_executable(ctx, op, m_pad, k_pad, n_pad)?;

    let (a_rows_pad, a_cols_pad) = match op {
        TpuOpKind::MatMul => (m_pad, k_pad),
        TpuOpKind::MatMulAtB => (k_pad, m_pad),
    };
    let a_host = pad_matrix_to_f32(a, a_rows_pad, a_cols_pad);
    let b_host = pad_matrix_to_f32(b, k_pad, n_pad);

    let buf_a = upload_f32_buffer(api, ctx.client.0, device, &a_host, a_rows_pad, a_cols_pad)?;
    let buf_b = match upload_f32_buffer(api, ctx.client.0, device, &b_host, k_pad, n_pad) {
        Some(buf) => buf,
        None => {
            destroy_pjrt_buffer(api, buf_a);
            return None;
        }
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

    let err = unsafe { exec_fn(&mut exec_args) };
    destroy_pjrt_buffer(api, buf_a);
    destroy_pjrt_buffer(api, buf_b);

    if !check_pjrt_error(api, err) || out_ptrs[0].is_null() {
        let _ = await_and_destroy_event(api, complete_event);
        destroy_pjrt_buffer(api, out_ptrs[0]);
        return None;
    }

    if !await_and_destroy_event(api, complete_event) {
        destroy_pjrt_buffer(api, out_ptrs[0]);
        return None;
    }

    let mut out_host = vec![0.0f32; m_pad * n_pad];
    let to_host_fn = api.buffer_to_host_buffer?;
    let mut to_host_args = PjrtBufferToHostBufferArgs {
        struct_size: std::mem::size_of::<PjrtBufferToHostBufferArgs>(),
        extension_start: ptr::null_mut(),
        src: out_ptrs[0],
        host_layout: ptr::null_mut(),
        dst: out_host.as_mut_ptr() as *mut c_void,
        dst_size: out_host.len() * std::mem::size_of::<f32>(),
        event: ptr::null_mut(),
    };
    let err = unsafe { to_host_fn(&mut to_host_args) };
    let transfer_ok = check_pjrt_error(api, err) && await_and_destroy_event(api, to_host_args.event);
    destroy_pjrt_buffer(api, out_ptrs[0]);

    if !transfer_ok {
        return None;
    }

    Some(unpad_f32_to_matrix(&out_host, n_pad, m, n))
}

// --- Public OchreML Backend Entry Points ---

pub fn tpu_matmul(device_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    execute_tpu_dot(device_id, TpuOpKind::MatMul, a, b).map(Ok)
}

pub fn tpu_matmul_at_b(device_id: usize, a: &Matrix, b: &Matrix) -> Option<OchreResult<Matrix>> {
    execute_tpu_dot(device_id, TpuOpKind::MatMulAtB, a, b).map(Ok)
}

pub fn tpu_matvec(device_id: usize, a: &Matrix, v: &[f64]) -> Option<OchreResult<Vec<f64>>> {
    let v_mat = Matrix::from_vec(v.len(), 1, v.to_vec()).ok()?;
    let res = execute_tpu_dot(device_id, TpuOpKind::MatMul, a, &v_mat)?;
    Some(Ok(res.data))
}

/// Column-scaled TPU Normal Equation for 1D target: `(X^T * X, X^T * y)`
pub fn tpu_compute_normal_equation(
    device_id: usize,
    x: &Matrix,
    y: &[f64],
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Vec<f64>)>> {
    if !is_available() {
        return None;
    }
    let y_mat = Matrix::from_vec(y.len(), 1, y.to_vec()).ok()?;
    let (xt_x, xt_y_mat) =
        tpu_compute_normal_equation_multi(device_id, x, &y_mat, None, fit_intercept)?.ok()?;
    Some(Ok((xt_x, xt_y_mat.data)))
}

/// Column-scaled TPU Normal Equation for multi-target & optional `sample_weight`.
/// Scales columns of X by max-abs on host before `f32` TPU `dot_general` to avoid
/// catastrophic cancellation when feature scales differ widely.
pub fn tpu_compute_normal_equation_multi(
    device_id: usize,
    x: &Matrix,
    y: &Matrix,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> Option<OchreResult<(Matrix, Matrix)>> {
    if !is_available() {
        return None;
    }
    let n = x.rows;
    let p_in = x.cols;
    let k = y.cols;
    let p = if fit_intercept { p_in + 1 } else { p_in };

    let mut x_scales = vec![1.0f64; p_in];
    for r in 0..n {
        let w_sqrt = sample_weight.map_or(1.0, |w| w[r].max(0.0).sqrt());
        for c in 0..p_in {
            let v = (x[(r, c)] * w_sqrt).abs();
            if v > x_scales[c] {
                x_scales[c] = v;
            }
        }
    }

    let mut y_scales = vec![1.0f64; k];
    for r in 0..n {
        let w_sqrt = sample_weight.map_or(1.0, |w| w[r].max(0.0).sqrt());
        for t in 0..k {
            let v = (y[(r, t)] * w_sqrt).abs();
            if v > y_scales[t] {
                y_scales[t] = v;
            }
        }
    }

    let mut x_scaled = vec![0.0f64; n * p_in];
    let mut y_scaled = vec![0.0f64; n * k];
    let mut sum_wx = vec![0.0f64; p_in];
    let mut sum_wy = vec![0.0f64; k];
    let mut sum_w = 0.0f64;

    for r in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[r]);
        let w_sqrt = w.max(0.0).sqrt();
        sum_w += w;
        for c in 0..p_in {
            let xv = x[(r, c)];
            sum_wx[c] += w * xv;
            x_scaled[r * p_in + c] = (xv * w_sqrt) / x_scales[c];
        }
        for t in 0..k {
            let yv = y[(r, t)];
            sum_wy[t] += w * yv;
            y_scaled[r * k + t] = (yv * w_sqrt) / y_scales[t];
        }
    }

    let x_mat = Matrix::from_vec(n, p_in, x_scaled).ok()?;
    let y_mat = Matrix::from_vec(n, k, y_scaled).ok()?;

    let xt_x_scaled = execute_tpu_dot(device_id, TpuOpKind::MatMulAtB, &x_mat, &x_mat)?;
    let xt_y_scaled = execute_tpu_dot(device_id, TpuOpKind::MatMulAtB, &x_mat, &y_mat)?;

    let mut xt_x = Matrix::zeros(p, p);
    let mut xt_y = Matrix::zeros(p, k);

    for i in 0..p_in {
        for j in 0..p_in {
            xt_x[(i, j)] = xt_x_scaled[(i, j)] * x_scales[i] * x_scales[j];
        }
        for t in 0..k {
            xt_y[(i, t)] = xt_y_scaled[(i, t)] * x_scales[i] * y_scales[t];
        }
    }

    if fit_intercept {
        for c in 0..p_in {
            xt_x[(c, p_in)] = sum_wx[c];
            xt_x[(p_in, c)] = sum_wx[c];
        }
        xt_x[(p_in, p_in)] = sum_w;
        for t in 0..k {
            xt_y[(p_in, t)] = sum_wy[t];
        }
    }

    Some(Ok((xt_x, xt_y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_pow2_dim() {
        assert_eq!(next_pow2_dim(1, 16), 16);
        assert_eq!(next_pow2_dim(16, 16), 16);
        assert_eq!(next_pow2_dim(17, 16), 32);
        assert_eq!(next_pow2_dim(100, 16), 128);
    }

    #[test]
    fn test_pad_and_unpad_roundtrip() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let padded = pad_matrix_to_f32(&m, 4, 4);
        assert_eq!(padded.len(), 16);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[3], 0.0);
        assert_eq!(padded[4], 4.0);
        assert_eq!(padded[15], 0.0);

        let restored = unpad_f32_to_matrix(&padded, 4, 2, 3);
        assert_eq!(restored, m);
    }

    #[test]
    fn test_emit_stablehlo_and_vhlo_header() {
        let mlir = emit_stablehlo_dot_general_mlir(32, 64, 16, false);
        assert!(mlir.contains("func.func @main"));
        assert!(mlir.contains("stablehlo.dot_general"));
        assert!(mlir.contains("tensor<32x64xf32>"));
        assert!(mlir.contains("tensor<64x16xf32>"));
        assert!(mlir.contains("tensor<32x16xf32>"));

        let mlir_t = emit_stablehlo_dot_general_mlir(32, 64, 16, true);
        assert!(mlir_t.contains("contracting_dims = [0] x [0]"));

        let hdr = emit_vhlo_bytecode_header("OchreML-0.3");
        assert_eq!(&hdr[0..4], b"ML\xefR");
    }

    #[test]
    fn test_tpu_graceful_fallback_when_unavailable() {
        // On hosts without libtpu.so / PJRT TPU, calls cleanly return None for CPU fallback.
        if !is_available() {
            let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
            assert!(tpu_matmul(0, &a, &b).is_none());
            assert!(tpu_matmul_at_b(0, &a, &b).is_none());
            assert!(tpu_matvec(0, &a, &[1.0, 2.0]).is_none());
        }
    }
}
