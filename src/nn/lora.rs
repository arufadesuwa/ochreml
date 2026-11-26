// OchreML LoRA (Low-Rank Adaptation) Module (＾▽＾)
// Implements frozen-base LoRALinear with weight merging/unmerging and DIT\x01 tag 102 adapter state persistence.

use crate::device::DeviceType;
use crate::dit_io::{
    ByteReader, check_magic_and_tag, write_header, write_str, write_u8, write_usize,
};
use crate::error::{OchreError, OchreResult};
use crate::nn::dtype::{DType, TensorStorage};
use crate::nn::gemm_f32::sgemm;
use crate::nn::layers::Linear;
use crate::nn::tensor::Tensor;

/// Binary serialization tag for `LoRAAdapterState` / state dicts in `.dit` files.
pub const TAG_NN_STATE_DICT: u8 = 102;

/// Low-Rank Adaptation wrapper around a frozen `Linear` layer:
/// `y = x @ W^T (+ b) + (alpha / rank) * (x @ A^T) @ B^T`.
#[derive(Clone, Debug)]
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub rank: usize,
    pub alpha: f32,
    pub merged: bool,
}

impl LoRALinear {
    /// Wraps `base` in a `LoRALinear` adapter, freezing `base` parameters and initializing
    /// `lora_a` `[rank, in_features]` (Kaiming normal) and `lora_b` `[out_features, rank]` (zeros).
    pub fn new(base: Linear, rank: usize, alpha: f32, seed: u64) -> OchreResult<Self> {
        if rank == 0 {
            return Err(OchreError::InvalidParameter(
                "LoRALinear rank must be > 0 ( >_< )".to_string(),
            ));
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "LoRALinear alpha={alpha} must be finite and > 0 ( >_< )"
            )));
        }

        let mut frozen_base = base;
        frozen_base.weight = frozen_base.weight.with_requires_grad(false);
        if let Some(b) = frozen_base.bias.take() {
            frozen_base.bias = Some(b.with_requires_grad(false));
        }

        let std = (2.0f32 / (frozen_base.in_features as f32)).sqrt();
        let lora_a = Tensor::randn_with(
            vec![rank, frozen_base.in_features],
            0.0,
            std,
            frozen_base.dtype,
            frozen_base.device.clone(),
            seed,
            true,
        );
        let lora_b = Tensor::zeros(
            vec![frozen_base.out_features, rank],
            frozen_base.dtype,
            frozen_base.device.clone(),
        )
        .with_requires_grad(true);

        Ok(Self {
            base: frozen_base,
            lora_a,
            lora_b,
            rank,
            alpha,
            merged: false,
        })
    }

    /// Constructs a `LoRALinear` from an existing `base` and pre-initialized `lora_a` / `lora_b` tensors.
    pub fn from_parts(
        base: Linear,
        lora_a: Tensor,
        lora_b: Tensor,
        rank: usize,
        alpha: f32,
    ) -> OchreResult<Self> {
        if rank == 0 {
            return Err(OchreError::InvalidParameter(
                "LoRALinear rank must be > 0 ( >_< )".to_string(),
            ));
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(OchreError::InvalidParameter(format!(
                "LoRALinear alpha={alpha} must be finite and > 0 ( >_< )"
            )));
        }
        if lora_a.shape != [rank, base.in_features] {
            return Err(OchreError::InvalidParameter(format!(
                "LoRALinear lora_a shape {:?} != expected [{rank}, {}] ( >_< )",
                lora_a.shape, base.in_features
            )));
        }
        if lora_b.shape != [base.out_features, rank] {
            return Err(OchreError::InvalidParameter(format!(
                "LoRALinear lora_b shape {:?} != expected [{}, {rank}] ( >_< )",
                lora_b.shape, base.out_features
            )));
        }
        let mut frozen_base = base;
        frozen_base.weight = frozen_base.weight.with_requires_grad(false);
        if let Some(b) = frozen_base.bias.take() {
            frozen_base.bias = Some(b.with_requires_grad(false));
        }
        Ok(Self {
            base: frozen_base,
            lora_a: lora_a.with_requires_grad(true),
            lora_b: lora_b.with_requires_grad(true),
            rank,
            alpha,
            merged: false,
        })
    }

    /// Adapter scaling factor `alpha / rank`.
    #[inline]
    pub fn scaling(&self) -> f32 {
        self.alpha / (self.rank as f32)
    }

    /// Autograd-tracked forward pass. Uses `base.forward(x)` directly when merged,
    /// or `x @ W^T (+ b) + (alpha / r) * (x @ A^T) @ B^T` when unmerged.
    pub fn forward(&self, x: &Tensor) -> OchreResult<Tensor> {
        let base_out = self.base.forward(x)?;
        if self.merged {
            return Ok(base_out);
        }
        let xa = x.matmul_trans(&self.lora_a, false, true)?;
        let xab = xa.matmul_trans(&self.lora_b, false, true)?;
        let scaled = xab.scale(self.scaling())?;
        base_out.add(&scaled)
    }

    /// Folds `(alpha / rank) * (B @ A)` into `base.weight` for zero-overhead inference.
    pub fn merge_weights(&mut self) -> OchreResult<()> {
        if self.merged {
            return Ok(());
        }
        let scale = self.scaling();
        sgemm(
            self.base.out_features,
            self.base.in_features,
            self.rank,
            &self.lora_b.data,
            &self.lora_a.data,
            &mut self.base.weight.data,
            false,
            false,
            scale,
            1.0,
        )?;
        if self.base.dtype != DType::F32 {
            self.base.weight.data =
                TensorStorage::from_f32_slice(&self.base.weight.data, self.base.dtype).to_f32_vec();
        }
        self.merged = true;
        Ok(())
    }

    /// Subtracts `(alpha / rank) * (B @ A)` from `base.weight` to restore unmerged training state.
    pub fn unmerge_weights(&mut self) -> OchreResult<()> {
        if !self.merged {
            return Ok(());
        }
        let scale = self.scaling();
        sgemm(
            self.base.out_features,
            self.base.in_features,
            self.rank,
            &self.lora_b.data,
            &self.lora_a.data,
            &mut self.base.weight.data,
            false,
            false,
            -scale,
            1.0,
        )?;
        if self.base.dtype != DType::F32 {
            self.base.weight.data =
                TensorStorage::from_f32_slice(&self.base.weight.data, self.base.dtype).to_f32_vec();
        }
        self.merged = false;
        Ok(())
    }

    /// Returns references to the trainable LoRA adapter parameters (`lora_a`, `lora_b`).
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Returns mutable references to the trainable LoRA adapter parameters (`lora_a`, `lora_b`).
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }
}

/// Named LoRA adapter entry storing `(name, rank, alpha, lora_a, lora_b)`.
#[derive(Clone, Debug)]
pub struct LoRAAdapterEntry {
    pub name: String,
    pub rank: usize,
    pub alpha: f32,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
}

/// Serializable collection of named LoRA adapter `A`/`B` tensors (`TAG_NN_STATE_DICT = 102`).
#[derive(Clone, Debug, Default)]
pub struct LoRAAdapterState {
    pub adapters: Vec<LoRAAdapterEntry>,
}

impl LoRAAdapterState {
    /// Creates an empty `LoRAAdapterState`.
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Inserts or replaces a named `LoRALinear` adapter in the state dictionary.
    pub fn insert(&mut self, name: impl Into<String>, layer: &LoRALinear) -> OchreResult<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(OchreError::InvalidParameter(
                "LoRAAdapterState adapter name cannot be empty ( >_< )".to_string(),
            ));
        }
        let entry = LoRAAdapterEntry {
            name: name.clone(),
            rank: layer.rank,
            alpha: layer.alpha,
            lora_a: layer.lora_a.detach(),
            lora_b: layer.lora_b.detach(),
        };
        if let Some(existing) = self.adapters.iter_mut().find(|a| a.name == name) {
            *existing = entry;
        } else {
            self.adapters.push(entry);
        }
        Ok(())
    }

    /// Builds a `LoRAAdapterState` from a slice of `(name, &LoRALinear)` pairs.
    pub fn from_named_layers(layers: &[(&str, &LoRALinear)]) -> OchreResult<Self> {
        let mut state = Self::new();
        for &(name, layer) in layers {
            state.insert(name, layer)?;
        }
        Ok(state)
    }

    /// Looks up a named adapter entry.
    pub fn get(&self, name: &str) -> Option<&LoRAAdapterEntry> {
        self.adapters.iter().find(|a| a.name == name)
    }

    /// Loads the named adapter weights into `target`, validating shape, rank, and merged state.
    pub fn apply_to(&self, name: &str, target: &mut LoRALinear) -> OchreResult<()> {
        let entry = self.get(name).ok_or_else(|| {
            OchreError::InvalidParameter(format!(
                "Adapter '{name}' not found in LoRAAdapterState ( >_< )"
            ))
        })?;
        if entry.lora_a.shape != target.lora_a.shape || entry.lora_b.shape != target.lora_b.shape {
            return Err(OchreError::InvalidParameter(format!(
                "Adapter '{name}' shapes ({:?}, {:?}) do not match target ({:?}, {:?}) ( >_< )",
                entry.lora_a.shape, entry.lora_b.shape, target.lora_a.shape, target.lora_b.shape
            )));
        }
        if target.merged {
            target.unmerge_weights()?;
        }
        target.rank = entry.rank;
        target.alpha = entry.alpha;
        target.lora_a.data.copy_from_slice(&entry.lora_a.data);
        target.lora_b.data.copy_from_slice(&entry.lora_b.data);
        Ok(())
    }

    /// Serializes all named LoRA adapters into `DIT\x01` format with `TAG_NN_STATE_DICT` (`102`).
    pub fn to_dit_bytes(&self) -> Vec<u8> {
        let mut buf = write_header(TAG_NN_STATE_DICT);
        write_usize(&mut buf, self.adapters.len());
        for entry in &self.adapters {
            write_str(&mut buf, &entry.name);
            write_usize(&mut buf, entry.rank);
            write_f32(&mut buf, entry.alpha);
            write_u8(&mut buf, dtype_to_tag(entry.lora_a.dtype));
            write_usize(&mut buf, entry.lora_a.shape[0]);
            write_usize(&mut buf, entry.lora_a.shape[1]);
            write_f32_vec(&mut buf, &entry.lora_a.data);
            write_usize(&mut buf, entry.lora_b.shape[0]);
            write_usize(&mut buf, entry.lora_b.shape[1]);
            write_f32_vec(&mut buf, &entry.lora_b.data);
        }
        buf
    }

    /// Deserializes `LoRAAdapterState` from `DIT\x01` bytes onto `device`.
    pub fn from_dit_bytes(bytes: &[u8], device: DeviceType) -> OchreResult<Self> {
        let mut r = ByteReader::new(bytes);
        check_magic_and_tag(&mut r, TAG_NN_STATE_DICT)?;
        let count = r.read_usize()?;
        let mut adapters = Vec::with_capacity(count);
        for _ in 0..count {
            let name = r.read_str()?;
            let rank = r.read_usize()?;
            let alpha = read_f32(&mut r)?;
            if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
                return Err(OchreError::CorruptedDitFile);
            }
            let dtype = tag_to_dtype(r.read_u8()?)?;
            let a_rows = r.read_usize()?;
            let a_cols = r.read_usize()?;
            let a_data = read_f32_vec(&mut r)?;
            let lora_a = Tensor::parameter(a_data, vec![a_rows, a_cols], dtype, device.clone())?;

            let b_rows = r.read_usize()?;
            let b_cols = r.read_usize()?;
            let b_data = read_f32_vec(&mut r)?;
            let lora_b = Tensor::parameter(b_data, vec![b_rows, b_cols], dtype, device.clone())?;

            adapters.push(LoRAAdapterEntry {
                name,
                rank,
                alpha,
                lora_a,
                lora_b,
            });
        }
        Ok(Self { adapters })
    }
}

fn dtype_to_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
    }
}

fn tag_to_dtype(tag: u8) -> OchreResult<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::BF16),
        2 => Ok(DType::F16),
        _ => Err(OchreError::CorruptedDitFile),
    }
}

fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    write_usize(buf, v.len());
    for &x in v {
        write_f32(buf, x);
    }
}

fn read_f32(r: &mut ByteReader<'_>) -> OchreResult<f32> {
    let b = r.read_exact(4)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(b);
    Ok(f32::from_le_bytes(arr))
}

fn read_f32_vec(r: &mut ByteReader<'_>) -> OchreResult<Vec<f32>> {
    let len = r.read_usize()?;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(read_f32(r)?);
    }
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::autograd::{AutogradTape, with_no_grad};

    #[test]
    fn test_lora_merge_unmerge_parity_and_roundtrip() {
        let base = Linear::new(6, 4, true, DType::F32, DeviceType::Cpu, 17).unwrap();
        let orig_w = base.weight.data.clone();
        let mut lora = LoRALinear::new(base, 2, 4.0, 99).unwrap();
        // Set non-zero lora_b so adapter contribution is significant
        lora.lora_b = Tensor::parameter(
            vec![0.2, -0.3, 0.5, 0.1, -0.4, 0.6, 0.25, -0.15],
            vec![4, 2],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        let x = Tensor::randn(vec![3, 6], DType::F32, DeviceType::Cpu, 55);
        let unmerged_y = with_no_grad(|| lora.forward(&x).unwrap());

        lora.merge_weights().unwrap();
        assert!(lora.merged);
        let merged_y = with_no_grad(|| lora.forward(&x).unwrap());

        let max_err = unmerged_y
            .data
            .iter()
            .zip(merged_y.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "LoRALinear merge_weights parity max_err={max_err}"
        );

        lora.unmerge_weights().unwrap();
        assert!(!lora.merged);
        let roundtrip_w_err = orig_w
            .iter()
            .zip(lora.base.weight.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            roundtrip_w_err < 1e-5,
            "LoRALinear unmerge_weights weight roundtrip max_err={roundtrip_w_err}"
        );

        let unmerged_y2 = with_no_grad(|| lora.forward(&x).unwrap());
        let roundtrip_y_err = unmerged_y
            .data
            .iter()
            .zip(unmerged_y2.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(roundtrip_y_err < 1e-5);
    }

    #[test]
    fn test_lora_finite_diff_grad_check_and_frozen_base() {
        AutogradTape::clear();
        let base = Linear::new(4, 3, true, DType::F32, DeviceType::Cpu, 101).unwrap();
        let mut lora = LoRALinear::new(base, 2, 3.0, 202).unwrap();
        lora.lora_b = Tensor::parameter(
            vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.1],
            vec![3, 2],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        assert!(!lora.base.weight.requires_grad);
        assert!(!lora.base.bias.as_ref().unwrap().requires_grad);
        assert!(lora.lora_a.requires_grad);
        assert!(lora.lora_b.requires_grad);

        let x = Tensor::parameter(
            vec![0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.9],
            vec![2, 4],
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();
        let w = Tensor::new(
            vec![0.7, -0.2, 0.5, -0.4, 0.9, 0.3],
            vec![2, 3],
            DType::F32,
            DeviceType::Cpu,
            false,
        )
        .unwrap();

        let y = lora.forward(&x).unwrap();
        let grads = y.mul(&w).unwrap().sum(None).unwrap().backward().unwrap();
        let dx = grads.get(x.node_id.unwrap()).unwrap();
        let da = grads.get(lora.lora_a.node_id.unwrap()).unwrap();
        let db = grads.get(lora.lora_b.node_id.unwrap()).unwrap();

        let eps = 1e-3f32;
        for i in 0..x.numel() {
            let mut xp = x.data.clone();
            let mut xm = x.data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = with_no_grad(|| {
                let tp = Tensor::new(xp, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let tm = Tensor::new(xm, vec![2, 4], DType::F32, DeviceType::Cpu, false).unwrap();
                let lp = lora
                    .forward(&tp)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = lora
                    .forward(&tm)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!((dx[i] - num).abs() < 5e-3, "LoRA dx[{i}]");
        }

        for i in 0..lora.lora_a.numel() {
            let mut ap = lora.lora_a.data.clone();
            let mut am = lora.lora_a.data.clone();
            ap[i] += eps;
            am[i] -= eps;
            let num = with_no_grad(|| {
                let mut lp_layer = lora.clone();
                lp_layer.lora_a.data = ap;
                let mut lm_layer = lora.clone();
                lm_layer.lora_a.data = am;
                let lp = lp_layer
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = lm_layer
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!((da[i] - num).abs() < 5e-3, "LoRA dA[{i}]");
        }

        for i in 0..lora.lora_b.numel() {
            let mut bp = lora.lora_b.data.clone();
            let mut bm = lora.lora_b.data.clone();
            bp[i] += eps;
            bm[i] -= eps;
            let num = with_no_grad(|| {
                let mut lp_layer = lora.clone();
                lp_layer.lora_b.data = bp;
                let mut lm_layer = lora.clone();
                lm_layer.lora_b.data = bm;
                let lp = lp_layer
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                let lm = lm_layer
                    .forward(&x)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(None)
                    .unwrap();
                (lp.data[0] - lm.data[0]) / (2.0 * eps)
            });
            assert!((db[i] - num).abs() < 5e-3, "LoRA dB[{i}]");
        }
    }

    #[test]
    fn test_lora_adapter_state_dit_roundtrip() {
        let base1 = Linear::new(4, 4, false, DType::F32, DeviceType::Cpu, 1).unwrap();
        let base2 = Linear::new(4, 2, true, DType::F32, DeviceType::Cpu, 2).unwrap();
        let mut l1 = LoRALinear::new(base1.clone(), 2, 8.0, 10).unwrap();
        l1.lora_b.data.fill(0.25);
        let mut l2 = LoRALinear::new(base2.clone(), 1, 2.0, 20).unwrap();
        l2.lora_b.data.fill(-0.5);

        let state =
            LoRAAdapterState::from_named_layers(&[("q_proj", &l1), ("v_proj", &l2)]).unwrap();
        let bytes = state.to_dit_bytes();
        assert_eq!(&bytes[0..4], b"DIT\x01");
        assert_eq!(bytes[4], TAG_NN_STATE_DICT);

        let loaded = LoRAAdapterState::from_dit_bytes(&bytes, DeviceType::Cpu).unwrap();
        assert_eq!(loaded.adapters.len(), 2);

        let mut target1 = LoRALinear::new(base1, 2, 1.0, 999).unwrap();
        loaded.apply_to("q_proj", &mut target1).unwrap();
        assert_eq!(target1.rank, 2);
        assert!((target1.alpha - 8.0).abs() < 1e-6);
        assert_eq!(target1.lora_a.data, l1.lora_a.data);
        assert_eq!(target1.lora_b.data, l1.lora_b.data);

        assert!(LoRAAdapterState::from_dit_bytes(b"NOPE!", DeviceType::Cpu).is_err());
    }
}
