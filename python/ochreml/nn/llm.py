"""
OchreML Attention, Transformer, CausalLM & LoRA Modules (＾▽＾)
Provides PositionalEncoding, RoPE, ScaledDotProductAttention, KVCache, MultiHeadAttention,
SwiGLU, FeedForward, TransformerEncoder/Decoder, CausalLM, and LoRALinear backed by Rust.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .._ochreml import nn as _rust_nn
from .._preproc_utils import validate_dit_path
from .modules import (
    Linear,
    Module,
    Parameter,
    Tensor,
    _normalize_dtype,
    _public_dtype,
    _resolve_device,
    _unwrap_tensor,
)


class PositionalEncoding(Module):
    """Sinusoidal Transformer positional encoding (＾▽＾)."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.positional_encoding_forward(
                _unwrap_tensor(x), self.max_len, self.d_model, offset=int(offset)
            )
        )


class RoPE(Module):
    """Rotary Position Embedding (RoPE) for [B, H, S, D] tensors (*≧ω≦*)."""

    def __init__(
        self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.rope_forward(
                _unwrap_tensor(x),
                self.head_dim,
                self.max_seq_len,
                theta=self.theta,
                offset=int(offset),
            )
        )


class ScaledDotProductAttention(Module):
    """Tiled FlashAttention-style scaled dot-product attention with GQA/MQA support (o´∀｀o)."""

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool = False,
        kv_offset: int = 0,
    ) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.sdpa_forward(
                _unwrap_tensor(q),
                _unwrap_tensor(k),
                _unwrap_tensor(v),
                causal=bool(causal),
                kv_offset=int(kv_offset),
            )
        )


class KVCache:
    """Pre-allocated Key-Value cache [batch, num_kv_heads, max_seq_len, head_dim] (＾▽＾)."""

    def __init__(
        self, batch: int, num_kv_heads: int, max_seq_len: int, head_dim: int
    ) -> None:
        self._rust = _rust_nn.KVCache(
            int(batch), int(num_kv_heads), int(max_seq_len), int(head_dim)
        )

    @classmethod
    def _from_rust(cls, rust_cache: _rust_nn.KVCache) -> "KVCache":
        obj = cls.__new__(cls)
        obj._rust = rust_cache
        return obj

    def reset(self) -> None:
        self._rust.reset()

    def cur_len(self) -> int:
        return int(self._rust.cur_len())

    @property
    def max_seq_len(self) -> int:
        return int(self._rust.max_seq_len)


class MultiHeadAttention(Module):
    """MHA / GQA / MQA layer with optional RoPE and KVCache (*≧ω≦*)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = True,
        causal: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.num_heads
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.MultiHeadAttention(
            self.d_model,
            self.num_heads,
            num_kv_heads=self.num_kv_heads,
            bias=bool(bias),
            causal=bool(causal),
            use_rope=bool(use_rope),
            max_seq_len=int(max_seq_len),
            rope_theta=float(rope_theta),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        return Tensor._from_rust(
            self._rust.forward(
                _unwrap_tensor(q),
                _unwrap_tensor(k),
                _unwrap_tensor(v),
                kv_cache=kv_cache._rust if kv_cache is not None else None,
            )
        )


class SwiGLU(Module):
    """SwiGLU feed-forward block: down(silu(gate(x)) * up(x)) (＾▽＾)."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        bias: bool = False,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.SwiGLU(
            int(d_model), int(hidden_dim), bias=bool(bias), dtype=dt, device=dev, seed=int(seed)
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))


class FeedForward(Module):
    """Transformer FeedForward block: fc1 -> GELU -> fc2 (o´∀｀o)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.FeedForward(
            int(d_model), int(d_ff), bias=bool(bias), dtype=dt, device=dev, seed=int(seed)
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        pre_norm: bool = True,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.TransformerEncoderLayer(
            int(d_model),
            int(num_heads),
            int(d_ff),
            eps=float(eps),
            pre_norm=bool(pre_norm),
            bias=bool(bias),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))


class TransformerEncoder(Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        pre_norm: bool = True,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.TransformerEncoder(
            int(num_layers),
            int(d_model),
            int(num_heads),
            int(d_ff),
            eps=float(eps),
            pre_norm=bool(pre_norm),
            bias=bool(bias),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        with_cross_attn: bool = True,
        use_swiglu: bool = False,
        pre_norm: bool = True,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.TransformerDecoderLayer(
            int(d_model),
            int(num_heads),
            int(d_ff),
            eps=float(eps),
            with_cross_attn=bool(with_cross_attn),
            use_swiglu=bool(use_swiglu),
            pre_norm=bool(pre_norm),
            bias=bool(bias),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        return Tensor._from_rust(
            self._rust.forward(
                _unwrap_tensor(x),
                memory=_unwrap_tensor(memory) if memory is not None else None,
                kv_cache=kv_cache._rust if kv_cache is not None else None,
            )
        )


class TransformerDecoder(Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        with_cross_attn: bool = True,
        use_swiglu: bool = False,
        pre_norm: bool = True,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.TransformerDecoder(
            int(num_layers),
            int(d_model),
            int(num_heads),
            int(d_ff),
            eps=float(eps),
            with_cross_attn=bool(with_cross_attn),
            use_swiglu=bool(use_swiglu),
            pre_norm=bool(pre_norm),
            bias=bool(bias),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        kv_caches: Optional[Sequence[KVCache]] = None,
    ) -> Tensor:
        raw_caches = None if kv_caches is None else [c._rust for c in kv_caches]
        return Tensor._from_rust(
            self._rust.forward(
                _unwrap_tensor(x),
                memory=_unwrap_tensor(memory) if memory is not None else None,
                kv_caches=raw_caches,
            )
        )


class CausalLM(Module):
    """
    LLaMA-style Causal Language Model with RMSNorm, GQA+RoPE, SwiGLU,
    zero-GIL `generate()`, and `.dit` Tag 101 persistence (*≧ω≦*).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self._rust = _rust_nn.CausalLM(
            int(vocab_size),
            int(d_model),
            int(num_layers),
            int(num_heads),
            num_kv_heads=num_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            max_seq_len=int(max_seq_len),
            rope_theta=float(rope_theta),
            norm_eps=float(norm_eps),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._sync_from_rust()

    def _sync_from_rust(self) -> None:
        self.vocab_size = int(self._rust.vocab_size)
        self.d_model = int(self._rust.d_model)
        self.num_layers = int(self._rust.num_layers)
        self.num_heads = int(self._rust.num_heads)
        self.num_kv_heads = int(self._rust.num_kv_heads)
        self.ffn_hidden_dim = int(self._rust.ffn_hidden_dim)
        self.max_seq_len = int(self._rust.max_seq_len)
        self.rope_theta = float(self._rust.rope_theta)
        self.norm_eps = float(self._rust.norm_eps)
        self.dtype = _public_dtype(self._rust.dtype)
        self.device = str(self._rust.device)
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    @classmethod
    def _from_rust(
        cls, rust_lm: _rust_nn.CausalLM, device: Optional[str] = None
    ) -> "CausalLM":
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj._rust = rust_lm
        obj._sync_from_rust()
        if device is not None:
            obj.device = device
        return obj

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def new_kv_caches(self, batch: int = 1) -> List[KVCache]:
        return [KVCache._from_rust(c) for c in self._rust.new_kv_caches(int(batch))]

    def forward(
        self,
        input_ids: Any,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
        kv_caches: Optional[Sequence[KVCache]] = None,
    ) -> Tensor:
        raw_caches = None if kv_caches is None else [c._rust for c in kv_caches]
        return Tensor._from_rust(
            self._rust.forward(
                input_ids, batch=batch, seq_len=seq_len, kv_caches=raw_caches
            )
        )

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> List[int]:
        """Generate tokens zero-GIL in Rust using pre-allocated KVCaches (＾▽＾)."""
        return list(
            self._rust.generate(
                [int(t) for t in prompt_ids],
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                seed=int(seed),
            )
        )

    def num_parameters(self) -> int:
        return int(self._rust.num_parameters())

    def to_dit_bytes(self) -> bytes:
        return bytes(self._rust.to_dit_bytes())

    @classmethod
    def from_dit_bytes(
        cls, data: bytes, device: Optional[str] = None
    ) -> "CausalLM":
        dev = _resolve_device(device)
        return cls._from_rust(_rust_nn.CausalLM.from_dit_bytes(data, dev), dev)

    def save(self, path: str) -> None:
        self._rust.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CausalLM":
        dev = _resolve_device(device)
        return cls._from_rust(_rust_nn.CausalLM.load(validate_dit_path(path), dev), dev)


class LoRALinear(Module):
    """Low-Rank Adaptation wrapper around a frozen Linear layer (o´∀｀o)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.LoRALinear(
            int(in_features),
            int(out_features),
            rank=int(rank),
            alpha=float(alpha),
            bias=bool(bias),
            dtype=dt,
            device=dev,
            seed=int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    @classmethod
    def from_linear(
        cls,
        linear: Linear,
        rank: int = 4,
        alpha: float = 8.0,
        seed: int = 42,
    ) -> "LoRALinear":
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.dtype = linear.dtype
        obj.device = linear.device
        obj._rust = _rust_nn.LoRALinear.from_linear(
            linear.weight._tensor,
            bias=linear.bias._tensor if linear.bias is not None else None,
            rank=int(rank),
            alpha=float(alpha),
            seed=int(seed),
        )
        obj._cached_params = [Parameter._from_rust(p) for p in obj._rust.parameters()]
        return obj

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    @property
    def lora_a(self) -> Parameter:
        return self._cached_params[0]

    @property
    def lora_b(self) -> Parameter:
        return self._cached_params[1]

    @property
    def merged(self) -> bool:
        return bool(self._rust.merged)

    @property
    def rank(self) -> int:
        return int(self._rust.rank)

    @property
    def alpha(self) -> float:
        return float(self._rust.alpha)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))

    def merge_weights(self) -> None:
        self._rust.merge_weights()

    def unmerge_weights(self) -> None:
        self._rust.unmerge_weights()

    def save_adapter_bytes(self, name: str = "default") -> bytes:
        return bytes(self._rust.save_adapter_bytes(name=name))

    def load_adapter_bytes(self, data: bytes, name: str = "default") -> None:
        self._rust.load_adapter_bytes(data, name=name)
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def save_adapter(self, path: str, name: str = "default") -> None:
        with open(validate_dit_path(path), "wb") as f:
            f.write(self.save_adapter_bytes(name=name))

    def load_adapter(self, path: str, name: str = "default") -> None:
        with open(validate_dit_path(path), "rb") as f:
            self.load_adapter_bytes(f.read(), name=name)


class LoRAAdapterState:
    """Named LoRA adapter state dictionary persisted via `.dit` Tag 102 (*≧ω≦*)."""

    def __init__(self) -> None:
        self._rust = _rust_nn.LoRAAdapterState()

    @classmethod
    def _from_rust(
        cls, rust_state: _rust_nn.LoRAAdapterState, device: Optional[str] = None
    ) -> "LoRAAdapterState":
        obj = cls.__new__(cls)
        obj._rust = rust_state
        return obj

    def insert(self, name: str, layer: LoRALinear) -> None:
        self._rust.insert(name, layer._rust)

    def apply_to(self, name: str, target: LoRALinear) -> None:
        self._rust.apply_to(name, target._rust)
        target._cached_params = [Parameter._from_rust(p) for p in target._rust.parameters()]

    def names(self) -> List[str]:
        return list(self._rust.names())

    def to_dit_bytes(self) -> bytes:
        return bytes(self._rust.to_dit_bytes())

    @classmethod
    def from_dit_bytes(
        cls, data: bytes, device: Optional[str] = None
    ) -> "LoRAAdapterState":
        dev = _resolve_device(device)
        return cls._from_rust(_rust_nn.LoRAAdapterState.from_dit_bytes(data, dev), dev)

    def save(self, path: str) -> None:
        self._rust.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LoRAAdapterState":
        dev = _resolve_device(device)
        return cls._from_rust(
            _rust_nn.LoRAAdapterState.load(validate_dit_path(path), dev), dev
        )
