// OchreML Native Hardware Accelerator Backends (*≧ω≦*)
// Houses runtime-loaded native drivers (CUDA/cuBLAS, and upcoming wgpu & TPU PJRT).

#[cfg(feature = "cuda-dynamic")]
pub mod cuda;

