// OchreML Native Hardware Accelerator Backends (*≧ω≦*)
// Houses runtime-loaded native drivers (CUDA/cuBLAS, and upcoming wgpu & TPU PJRT).

#[cfg(feature = "cuda-dynamic")]
pub mod cuda;
#[cfg(feature = "cuda-dynamic")]
pub mod cuda_primitives;

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_gpu;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu_primitives;

#[cfg(feature = "tpu-pjrt")]
pub mod tpu;
#[cfg(feature = "tpu-pjrt")]
pub mod tpu_primitives;
