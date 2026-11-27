"""
OchreML Deep Learning Core Modules (＾▽＾)
Provides Tensor, Parameter, Module, ModuleList, Sequential, MLP, CNN, RNN, Norm, and Loss wrappers
backed 100% by Rust hardware-accelerated kernels and autograd.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from .._ochreml import nn as _rust_nn
from .._preproc_utils import validate_dit_path

float32 = "float32"
bfloat16 = "bfloat16"
float16 = "float16"

_DTYPE_TO_RUST = {
    "f32": "f32",
    "float32": "f32",
    "fp32": "f32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "float16": "f16",
    "fp16": "f16",
}

_DTYPE_TO_PUBLIC = {
    "f32": "float32",
    "bf16": "bfloat16",
    "f16": "float16",
}


def _normalize_dtype(dtype: Optional[str]) -> str:
    if dtype is None:
        return "f32"
    key = str(dtype).strip().lower()
    if key in _DTYPE_TO_RUST:
        return _DTYPE_TO_RUST[key]
    raise ValueError(
        f"Unsupported dtype {dtype!r}. Expected 'float32'/'f32', 'bfloat16'/'bf16', or 'float16'/'f16' ( >_< )"
    )


def _public_dtype(dtype: str) -> str:
    return _DTYPE_TO_PUBLIC.get(dtype, dtype)


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


def _pair(val: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    if isinstance(val, int):
        return (int(val), int(val))
    seq = tuple(int(x) for x in val)
    if len(seq) != 2:
        raise ValueError(f"Expected int or 2-tuple, got {val!r} ( >_< )")
    return (seq[0], seq[1])


def _unwrap_tensor(x: Any) -> _rust_nn.Tensor:
    if isinstance(x, Tensor):
        return x._tensor
    if isinstance(x, _rust_nn.Tensor):
        return x
    raise TypeError(f"Expected ochreml.nn.Tensor, got {type(x).__name__} ( >_< )")


class no_grad:
    """Context manager and decorator disabling autograd tape recording (*≧ω≦*)."""

    def __enter__(self) -> "no_grad":
        _rust_nn.enter_no_grad()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        _rust_nn.exit_no_grad()

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)

        return wrapped


def is_grad_enabled() -> bool:
    """Return True if autograd recording is enabled on the current thread."""
    return bool(_rust_nn.is_grad_enabled())


class Tensor:
    """Hardware-accelerated OchreML Tensor with Rust autograd and f32/bf16/f16 support (＾▽＾)."""

    def __init__(
        self,
        data: Any,
        shape: Optional[Sequence[int]] = None,
        dtype: str = "float32",
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, _rust_nn.Tensor):
            self._tensor = data
            return
        if isinstance(data, Tensor):
            self._tensor = data._tensor
            return
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        shp = None if shape is None else [int(s) for s in shape]
        self._tensor = _rust_nn.Tensor(
            data, shape=shp, dtype=dt, device=dev, requires_grad=requires_grad
        )

    @classmethod
    def _from_rust(cls, rust_tensor: _rust_nn.Tensor) -> "Tensor":
        obj = cls.__new__(cls)
        obj._tensor = rust_tensor
        return obj

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: str = "float32",
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls._from_rust(
            _rust_nn.Tensor.zeros(
                [int(s) for s in shape],
                dtype=_normalize_dtype(dtype),
                device=_resolve_device(device),
                requires_grad=requires_grad,
            )
        )

    @classmethod
    def ones(
        cls,
        shape: Sequence[int],
        dtype: str = "float32",
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls._from_rust(
            _rust_nn.Tensor.ones(
                [int(s) for s in shape],
                dtype=_normalize_dtype(dtype),
                device=_resolve_device(device),
                requires_grad=requires_grad,
            )
        )

    @classmethod
    def randn(
        cls,
        shape: Sequence[int],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls._from_rust(
            _rust_nn.Tensor.randn(
                [int(s) for s in shape],
                mean=float(mean),
                std=float(std),
                dtype=_normalize_dtype(dtype),
                device=_resolve_device(device),
                seed=int(seed),
                requires_grad=requires_grad,
            )
        )

    @classmethod
    def uniform(
        cls,
        shape: Sequence[int],
        low: float = -1.0,
        high: float = 1.0,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls._from_rust(
            _rust_nn.Tensor.uniform(
                [int(s) for s in shape],
                low=float(low),
                high=float(high),
                dtype=_normalize_dtype(dtype),
                device=_resolve_device(device),
                seed=int(seed),
                requires_grad=requires_grad,
            )
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return int(self._tensor.ndim)

    @property
    def numel(self) -> int:
        return int(self._tensor.numel)

    @property
    def dtype(self) -> str:
        return _public_dtype(self._tensor.dtype)

    @property
    def device(self) -> str:
        return str(self._tensor.device)

    @property
    def requires_grad(self) -> bool:
        return bool(self._tensor.requires_grad)

    @requires_grad.setter
    def requires_grad(self, val: bool) -> None:
        self._tensor.requires_grad = bool(val)

    @property
    def node_id(self) -> Optional[int]:
        return self._tensor.node_id

    @property
    def grad(self) -> Optional["Tensor"]:
        g = self._tensor.grad
        if g is None:
            return None
        return Tensor._from_rust(g)

    def zero_grad(self) -> None:
        self._tensor.zero_grad()

    def numpy(self) -> Any:
        return self._tensor.numpy()

    def tolist(self) -> List[float]:
        return list(self._tensor.tolist())

    def to_dtype(self, dtype: str) -> "Tensor":
        return Tensor._from_rust(self._tensor.to_dtype(_normalize_dtype(dtype)))

    def to_device(self, device: str) -> "Tensor":
        return Tensor._from_rust(self._tensor.to_device(device))

    def to(self, target: str) -> "Tensor":
        low = str(target).strip().lower()
        if low in _DTYPE_TO_RUST:
            return self.to_dtype(low)
        return self.to_device(target)

    def detach(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.detach())

    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shp = [int(s) for s in shape[0]]
        else:
            shp = [int(s) for s in shape]  # type: ignore[arg-type]
        return Tensor._from_rust(self._tensor.reshape(shp))

    def transpose_last2(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.transpose_last2())

    def matmul(self, other: "Tensor") -> "Tensor":
        return Tensor._from_rust(self._tensor.matmul(_unwrap_tensor(other)))

    def relu(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.relu())

    def sigmoid(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.sigmoid())

    def tanh(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.tanh())

    def gelu(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.gelu())

    def silu(self) -> "Tensor":
        return Tensor._from_rust(self._tensor.silu())

    def sum(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor._from_rust(self._tensor.sum(axis))

    def mean(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor._from_rust(self._tensor.mean(axis))

    def backward(self) -> Any:
        return self._tensor.backward()

    def item(self) -> float:
        return float(self._tensor.item())

    def __add__(self, other: Any) -> "Tensor":
        rhs = other._tensor if isinstance(other, Tensor) else other
        return Tensor._from_rust(self._tensor + rhs)

    def __radd__(self, other: float) -> "Tensor":
        return Tensor._from_rust(other + self._tensor)

    def __sub__(self, other: Any) -> "Tensor":
        rhs = other._tensor if isinstance(other, Tensor) else other
        return Tensor._from_rust(self._tensor - rhs)

    def __rsub__(self, other: float) -> "Tensor":
        return Tensor._from_rust(other - self._tensor)

    def __neg__(self) -> "Tensor":
        return Tensor._from_rust(-self._tensor)

    def __mul__(self, other: Any) -> "Tensor":
        rhs = other._tensor if isinstance(other, Tensor) else other
        return Tensor._from_rust(self._tensor * rhs)

    def __rmul__(self, other: float) -> "Tensor":
        return Tensor._from_rust(other * self._tensor)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def __repr__(self) -> str:
        return repr(self._tensor)


class Parameter(Tensor):
    """Trainable parameter Tensor (`requires_grad=True` by default) (*≧ω≦*)."""

    def __init__(
        self,
        data: Any,
        shape: Optional[Sequence[int]] = None,
        dtype: str = "float32",
        device: Optional[str] = None,
        requires_grad: bool = True,
    ) -> None:
        if isinstance(data, Tensor):
            t = data._tensor
            if requires_grad and not t.requires_grad:
                t.requires_grad = True
            self._tensor = t
        elif isinstance(data, _rust_nn.Tensor):
            if requires_grad and not data.requires_grad:
                data.requires_grad = True
            self._tensor = data
        else:
            super().__init__(
                data,
                shape=shape,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )

    @classmethod
    def _from_rust(cls, rust_tensor: _rust_nn.Tensor) -> "Parameter":
        obj = cls.__new__(cls)
        obj._tensor = rust_tensor
        return obj


class Module:
    """PyTorch-style base class for all neural network layers and models (＾▽＾)."""

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        modules = object.__getattribute__(self, "_modules")
        params = object.__getattribute__(self, "_parameters")
        if isinstance(value, Parameter):
            modules.pop(name, None)
            params[name] = value
        elif isinstance(value, Module):
            params.pop(name, None)
            modules[name] = value
        else:
            if name in params and value is None:
                params[name] = None
            elif name in modules and value is None:
                modules[name] = None
        object.__setattr__(self, name, value)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name: str, tensor: Any) -> None:
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def _rust_owned_parameters(self) -> List[Parameter]:
        return []

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> List[Tuple[str, Parameter]]:
        out: List[Tuple[str, Parameter]] = []
        seen_ids: set[int] = set()

        for name, p in self._parameters.items():
            if p is not None:
                key = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
                if id(p._tensor) not in seen_ids:
                    seen_ids.add(id(p._tensor))
                    out.append((key, p))

        for idx, p in enumerate(self._rust_owned_parameters()):
            key = f"{prefix}param_{idx}" if not prefix else f"{prefix}.param_{idx}"
            if id(p._tensor) not in seen_ids:
                seen_ids.add(id(p._tensor))
                out.append((key, p))

        if recurse:
            for m_name, mod in self._modules.items():
                if mod is not None:
                    sub_prefix = m_name if not prefix else f"{prefix}.{m_name}"
                    out.extend(mod.named_parameters(prefix=sub_prefix, recurse=True))
        return out

    def parameters(self, recurse: bool = True) -> List[Parameter]:
        return [p for _, p in self.named_parameters(recurse=recurse)]

    def num_parameters(self) -> int:
        return sum(p.numel for p in self.parameters())

    def children(self) -> Iterator["Module"]:
        for mod in self._modules.values():
            if mod is not None:
                yield mod

    def modules(self) -> Iterator["Module"]:
        yield self
        for mod in self.children():
            yield from mod.modules()

    def _on_mode_change(self, mode: bool) -> None:
        pass

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", bool(mode))
        self._on_mode_change(bool(mode))
        for mod in self.children():
            mod.train(bool(mode))
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for name, p in self.named_parameters():
            state[name] = p.numpy()
        for mod_name, mod in self._named_modules():
            for b_name, buf in mod._buffers.items():
                key = b_name if not mod_name else f"{mod_name}.{b_name}"
                state[key] = list(buf) if isinstance(buf, (list, tuple)) else buf
        return state

    def _named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for name, mod in self._modules.items():
            if mod is not None:
                sub = name if not prefix else f"{prefix}.{name}"
                yield from mod._named_modules(sub)

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        named = dict(self.named_parameters())
        missing: List[str] = []
        unexpected = [k for k in state_dict if k not in named]
        for name, param in named.items():
            if name not in state_dict:
                missing.append(name)
                continue
            arr = state_dict[name]
            new_t = _rust_nn.Tensor(
                arr,
                shape=list(param.shape),
                dtype=_normalize_dtype(param.dtype),
                device=param.device,
                requires_grad=param.requires_grad,
            )
            param._tensor.copy_from_(new_t)

        for mod_name, mod in self._named_modules():
            for b_name in list(mod._buffers.keys()):
                key = b_name if not mod_name else f"{mod_name}.{b_name}"
                if key in state_dict:
                    mod._buffers[b_name] = list(state_dict[key])
                    object.__setattr__(mod, b_name, mod._buffers[b_name])
                    if key in unexpected:
                        unexpected.remove(key)

        if strict and (missing or unexpected):
            raise KeyError(
                f"Error(s) in loading state_dict: missing={missing}, unexpected={unexpected} ( >_< )"
            )
        return missing, unexpected

    def to(self, device: Optional[str] = None, dtype: Optional[str] = None) -> "Module":
        for p in self.parameters():
            if dtype is not None:
                p._tensor = p._tensor.to_dtype(_normalize_dtype(dtype))
            if device is not None:
                p._tensor = p._tensor.to_device(device)
        for mod in self.modules():
            if device is not None and hasattr(mod, "device"):
                object.__setattr__(mod, "device", device)
            if dtype is not None and hasattr(mod, "dtype"):
                object.__setattr__(mod, "dtype", _public_dtype(_normalize_dtype(dtype)))
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward() ( >_< )")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    """Ordered list of Modules with automatic parameter/submodule registration (o´∀｀o)."""

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        self._items: List[Module] = []
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        idx = len(self._items)
        self._items.append(module)
        self.add_module(str(idx), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._items)

    def __getitem__(self, idx: int) -> Module:
        return self._items[idx]


# =========================================================================
# MLP, ACTIVATION, EMBEDDING & NORM LAYERS
# =========================================================================


class Linear(Module):
    """Fully connected linear transformation `y = x @ W^T + b` (＾▽＾)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        w, b = _rust_nn.init_linear_params(
            self.in_features, self.out_features, self.use_bias, dt, dev, int(seed)
        )
        self.weight = Parameter._from_rust(w)
        self.bias = Parameter._from_rust(b) if b is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.linear_forward(
                _unwrap_tensor(x),
                self.weight._tensor,
                self.bias._tensor if self.bias is not None else None,
            )
        )


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class Dropout(Module):
    """Inverted dropout layer (*≧ω≦*)."""

    def __init__(self, p: float = 0.5, seed: int = 42) -> None:
        super().__init__()
        self.p = float(p)
        self.seed = int(seed)

    def forward(self, x: Tensor) -> Tensor:
        out, next_seed = _rust_nn.dropout_forward(
            _unwrap_tensor(x), self.p, self.training, self.seed
        )
        self.seed = next_seed
        return Tensor._from_rust(out)


class Embedding(Module):
    """Lookup table for discrete token/categorical indices (o´∀｀o)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        w = _rust_nn.init_embedding_params(
            self.num_embeddings, self.embedding_dim, dt, dev, int(seed)
        )
        self.weight = Parameter._from_rust(w)

    def forward(self, indices: Any) -> Tensor:
        batch = None
        seq_len = None
        if np is not None and isinstance(indices, np.ndarray) and indices.ndim == 2:
            batch, seq_len = int(indices.shape[0]), int(indices.shape[1])
            indices = indices.reshape(-1)
        elif (
            isinstance(indices, (list, tuple))
            and len(indices) > 0
            and isinstance(indices[0], (list, tuple))
        ):
            batch = len(indices)
            seq_len = len(indices[0])
            indices = [int(v) for row in indices for v in row]
        return Tensor._from_rust(
            _rust_nn.embedding_forward(
                indices, self.weight._tensor, batch=batch, seq_len=seq_len
            )
        )


class LayerNorm(Module):
    """Layer Normalization backed by N8 Rust kernels (＾▽＾)."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        if self.elementwise_affine:
            self.gamma: Optional[Parameter] = Parameter(
                _rust_nn.Tensor.ones(
                    [self.normalized_shape], dtype=dt, device=dev, requires_grad=True
                )
            )
            self.beta: Optional[Parameter] = Parameter(
                _rust_nn.Tensor.zeros(
                    [self.normalized_shape], dtype=dt, device=dev, requires_grad=True
                )
            )
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.layer_norm_forward(
                _unwrap_tensor(x),
                self.normalized_shape,
                self.eps,
                self.gamma._tensor if self.gamma is not None else None,
                self.beta._tensor if self.beta is not None else None,
            )
        )


class RMSNorm(Module):
    """Root Mean Square Layer Normalization backed by N8 Rust kernels (*≧ω≦*)."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        dtype: str = "float32",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self.gamma = Parameter(
            _rust_nn.Tensor.ones(
                [self.normalized_shape], dtype=dt, device=dev, requires_grad=True
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.rms_norm_forward(
                _unwrap_tensor(x),
                self.normalized_shape,
                self.eps,
                self.gamma._tensor,
            )
        )


# =========================================================================
# CNN LAYERS
# =========================================================================


class Conv2d(Module):
    """2D Convolution over NCHW tensors backed by im2col/col2im + SGEMM/GPU kernels (＾▽＾)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.use_bias = bool(bias)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        w, b = _rust_nn.init_conv2d_params(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.use_bias,
            dt,
            dev,
            int(seed),
        )
        self.weight = Parameter._from_rust(w)
        self.bias = Parameter._from_rust(b) if b is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.conv2d_forward(
                _unwrap_tensor(x),
                self.weight._tensor,
                self.bias._tensor if self.bias is not None else None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        )


class DepthwiseConv2d(Module):
    """Depthwise 2D Convolution without im2col overhead (*≧ω≦*)."""

    def __init__(
        self,
        in_channels: int,
        depth_multiplier: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.depth_multiplier = int(depth_multiplier)
        self.out_channels = self.in_channels * self.depth_multiplier
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.use_bias = bool(bias)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        w, b = _rust_nn.init_depthwise_conv2d_params(
            self.in_channels,
            self.depth_multiplier,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.use_bias,
            dt,
            dev,
            int(seed),
        )
        self.weight = Parameter._from_rust(w)
        self.bias = Parameter._from_rust(b) if b is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.depthwise_conv2d_forward(
                _unwrap_tensor(x),
                self.weight._tensor,
                self.bias._tensor if self.bias is not None else None,
                self.depth_multiplier,
                self.stride,
                self.padding,
                self.dilation,
            )
        )


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.max_pool2d_forward(
                _unwrap_tensor(x), self.kernel_size, self.stride, self.padding
            )
        )


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.avg_pool2d_forward(
                _unwrap_tensor(x), self.kernel_size, self.stride, self.padding
            )
        )


class GlobalAvgPool2d(Module):
    def __init__(self, keep_dims: bool = True) -> None:
        super().__init__()
        self.keep_dims = bool(keep_dims)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.global_avg_pool2d_forward(_unwrap_tensor(x), self.keep_dims)
        )


class BatchNorm2d(Module):
    """2D Batch Normalization over NCHW tensors (*≧ω≦*)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype: str = "float32",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        if self.affine:
            self.weight: Optional[Parameter] = Parameter(
                _rust_nn.Tensor.ones(
                    [self.num_features], dtype=dt, device=dev, requires_grad=True
                )
            )
            self.bias: Optional[Parameter] = Parameter(
                _rust_nn.Tensor.zeros(
                    [self.num_features], dtype=dt, device=dev, requires_grad=True
                )
            )
        else:
            self.weight = None
            self.bias = None
        self.register_buffer("running_mean", [0.0] * self.num_features)
        self.register_buffer("running_var", [1.0] * self.num_features)

    def forward(self, x: Tensor) -> Tensor:
        out, rm, rv = _rust_nn.batch_norm2d_forward(
            _unwrap_tensor(x),
            self.num_features,
            self.weight._tensor if self.weight is not None else None,
            self.bias._tensor if self.bias is not None else None,
            list(self.running_mean),
            list(self.running_var),
            self.training,
            self.eps,
            self.momentum,
        )
        self.running_mean = rm
        self.running_var = rv
        self._buffers["running_mean"] = rm
        self._buffers["running_var"] = rv
        return Tensor._from_rust(out)


class Flatten(Module):
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.flatten_forward(_unwrap_tensor(x), self.start_dim)
        )


class ResidualBlock(Module):
    """ResNet-style basic residual block executed in Rust (＾▽＾)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = _pair(stride)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.ResidualBlock(
            self.in_channels, self.out_channels, self.stride, dt, dev, int(seed)
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def _on_mode_change(self, mode: bool) -> None:
        if mode:
            self._rust.train()
        else:
            self._rust.eval()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor._from_rust(self._rust.forward(_unwrap_tensor(x)))


# =========================================================================
# SEQUENCE MODELS (RNN, GRU, LSTM)
# =========================================================================


class RNN(Module):
    """Multi-layer Elman RNN with BPTT in Rust (o´∀｀o)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        bidirectional: bool = False,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.nonlinearity = str(nonlinearity)
        self.bias = bool(bias)
        self.bidirectional = bool(bidirectional)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.RNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.nonlinearity,
            self.bias,
            self.bidirectional,
            dt,
            dev,
            int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self, input: Tensor, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        out, hn = self._rust.forward(
            _unwrap_tensor(input), _unwrap_tensor(h0) if h0 is not None else None
        )
        return Tensor._from_rust(out), Tensor._from_rust(hn)


class GRU(Module):
    """Multi-layer Gated Recurrent Unit (GRU) with BPTT in Rust (*≧ω≦*)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.bidirectional = bool(bidirectional)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.bidirectional,
            dt,
            dev,
            int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self, input: Tensor, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        out, hn = self._rust.forward(
            _unwrap_tensor(input), _unwrap_tensor(h0) if h0 is not None else None
        )
        return Tensor._from_rust(out), Tensor._from_rust(hn)


class LSTM(Module):
    """Multi-layer Long Short-Term Memory (LSTM) with BPTT in Rust (＾▽＾)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        dtype: str = "float32",
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.bidirectional = bool(bidirectional)
        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._rust = _rust_nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.bidirectional,
            dt,
            dev,
            int(seed),
        )
        self._cached_params = [Parameter._from_rust(p) for p in self._rust.parameters()]

    def _rust_owned_parameters(self) -> List[Parameter]:
        return self._cached_params

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, c0 = (None, None) if hx is None else (_unwrap_tensor(hx[0]), _unwrap_tensor(hx[1]))
        out, (hn, cn) = self._rust.forward(_unwrap_tensor(input), h0, c0)
        return Tensor._from_rust(out), (Tensor._from_rust(hn), Tensor._from_rust(cn))


# =========================================================================
# LOSSES
# =========================================================================


class MSELoss(Module):
    """Mean Squared Error loss backed by Rust autograd (＾▽＾)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = str(reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.mse_loss(_unwrap_tensor(pred), _unwrap_tensor(target), self.reduction)
        )


class CrossEntropyLoss(Module):
    """Fused Cross-Entropy Loss backed by N9 Rust kernel (*≧ω≦*)."""

    def __init__(
        self, label_smoothing: float = 0.0, ignore_index: Optional[int] = None
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Any) -> Tensor:
        return Tensor._from_rust(
            _rust_nn.cross_entropy_loss(
                _unwrap_tensor(logits),
                targets,
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index,
            )
        )


# =========================================================================
# SEQUENTIAL TRAINER & PERSISTENCE (TAG 100)
# =========================================================================


def _add_layer_to_rust_seq(rust_seq: _rust_nn.Sequential, layer: Module) -> bool:
    if isinstance(layer, Linear):
        rust_seq.add_linear(layer.in_features, layer.out_features, bias=layer.use_bias)
    elif isinstance(layer, ReLU):
        rust_seq.add_relu()
    elif isinstance(layer, Sigmoid):
        rust_seq.add_sigmoid()
    elif isinstance(layer, Tanh):
        rust_seq.add_tanh()
    elif isinstance(layer, GELU):
        rust_seq.add_gelu()
    elif isinstance(layer, SiLU):
        rust_seq.add_silu()
    elif isinstance(layer, Dropout):
        rust_seq.add_dropout(p=layer.p, seed=layer.seed)
    elif isinstance(layer, Embedding):
        rust_seq.add_embedding(layer.num_embeddings, layer.embedding_dim)
    elif isinstance(layer, LayerNorm):
        rust_seq.add_layernorm(
            layer.normalized_shape,
            eps=layer.eps,
            elementwise_affine=layer.elementwise_affine,
        )
    elif isinstance(layer, RMSNorm):
        rust_seq.add_rmsnorm(layer.normalized_shape, eps=layer.eps)
    elif isinstance(layer, Conv2d):
        rust_seq.add_conv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.use_bias,
        )
    elif isinstance(layer, DepthwiseConv2d):
        rust_seq.add_depthwise_conv2d(
            layer.in_channels,
            layer.depth_multiplier,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            bias=layer.use_bias,
        )
    elif isinstance(layer, MaxPool2d):
        rust_seq.add_max_pool2d(
            layer.kernel_size, stride=layer.stride, padding=layer.padding
        )
    elif isinstance(layer, AvgPool2d):
        rust_seq.add_avg_pool2d(
            layer.kernel_size, stride=layer.stride, padding=layer.padding
        )
    elif isinstance(layer, GlobalAvgPool2d):
        rust_seq.add_global_avg_pool2d(keep_dims=layer.keep_dims)
    elif isinstance(layer, BatchNorm2d):
        rust_seq.add_batch_norm2d(
            layer.num_features,
            eps=layer.eps,
            momentum=layer.momentum,
            affine=layer.affine,
        )
    elif isinstance(layer, Flatten):
        rust_seq.add_flatten(start_dim=layer.start_dim)
    elif isinstance(layer, ResidualBlock):
        rust_seq.add_residual_block(
            layer.in_channels, layer.out_channels, stride=layer.stride
        )
    else:
        return False
    return True


class Sequential(Module):
    """
    Sequential neural network container supporting both PyTorch-style `forward()`
    and zero-GIL Scikit-Learn-ergonomic `fit()` / `predict()` / `predict_proba()` (*≧ω≦*).
    """

    def __init__(
        self,
        *layers: Union[Module, Sequence[Module]],
        dtype: str = "float32",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            flat_layers = list(layers[0])
        else:
            flat_layers = list(layers)  # type: ignore[arg-type]

        dt = _normalize_dtype(dtype)
        dev = _resolve_device(device)
        self.dtype = _public_dtype(dt)
        self.device = dev
        self._layers: List[Module] = []
        self._rust_seq: Optional[_rust_nn.Sequential] = _rust_nn.Sequential(
            dtype=dt, device=dev
        )
        for idx, layer in enumerate(flat_layers):
            self.add_module(str(idx), layer)
            self._layers.append(layer)
            if self._rust_seq is not None and not _add_layer_to_rust_seq(
                self._rust_seq, layer
            ):
                self._rust_seq = None

    @classmethod
    def _from_rust(
        cls, rust_seq: _rust_nn.Sequential, device: Optional[str] = None
    ) -> "Sequential":
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj._rust_seq = rust_seq
        obj._layers = []
        obj.dtype = _public_dtype(rust_seq.dtype)
        obj.device = rust_seq.device or _resolve_device(device)
        object.__setattr__(obj, "training", bool(rust_seq.training))
        return obj

    def _on_mode_change(self, mode: bool) -> None:
        if self._rust_seq is not None:
            if mode:
                self._rust_seq.train()
            else:
                self._rust_seq.eval()

    def forward(self, x: Tensor) -> Tensor:
        if self._rust_seq is not None and len(self._layers) == 0:
            return Tensor._from_rust(self._rust_seq.forward(_unwrap_tensor(x)))
        cur = x
        for layer in self._layers:
            cur = layer(cur)
        return cur

    def fit(
        self,
        X: Any,
        y: Any,
        out_dim: Optional[int] = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        optimizer: str = "adamw",
        loss: str = "cross_entropy",
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[float]:
        """Train zero-GIL in Rust and return per-epoch mean loss history (＾▽＾)."""
        if self._rust_seq is None:
            raise ValueError(
                "Sequential.fit() requires all layers to be built-in Rust Sequential layers ( >_< )"
            )
        return list(
            self._rust_seq.fit(
                X,
                y,
                out_dim=out_dim,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                optimizer=optimizer,
                loss=loss,
                weight_decay=float(weight_decay),
                grad_clip=grad_clip,
                shuffle=bool(shuffle),
                seed=int(seed),
            )
        )

    def predict(self, X: Any, batch_size: int = 8192) -> Any:
        """Run batched inference in eval mode and return a NumPy float32 array (*≧ω≦*)."""
        if self._rust_seq is None:
            with no_grad():
                t = X if isinstance(X, Tensor) else Tensor(X, dtype=self.dtype, device=self.device)
                return self.forward(t).numpy()
        shape, raw = self._rust_seq.predict(X, batch_size=int(batch_size))
        if np is None:
            return raw
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(shape)).copy()

    def predict_proba(self, X: Any, batch_size: int = 8192) -> Any:
        """Return softmax class probabilities as a NumPy float32 array (o´∀｀o)."""
        if self._rust_seq is None:
            raise ValueError(
                "Sequential.predict_proba() requires built-in Rust Sequential layers ( >_< )"
            )
        shape, raw = self._rust_seq.predict_proba(X, batch_size=int(batch_size))
        if np is None:
            return raw
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(shape)).copy()

    def predict_classes(self, X: Any, batch_size: int = 8192) -> Any:
        """Run batched inference in eval mode and return 1D int64 class indices (*≧ω≦*)."""
        if self._rust_seq is None:
            logits = self.predict(X, batch_size=batch_size)
            if np is None:
                return [int(max(range(len(row)), key=lambda idx: row[idx])) for row in logits]
            return np.argmax(logits, axis=-1).astype(np.int64)
        classes = self._rust_seq.predict_classes(X, batch_size=int(batch_size))
        if np is None:
            return list(classes)
        return np.asarray(classes, dtype=np.int64)

    def num_parameters(self) -> int:
        if self._rust_seq is not None:
            return int(self._rust_seq.num_parameters())
        return super().num_parameters()

    def to_dit_bytes(self) -> bytes:
        if self._rust_seq is None:
            raise ValueError(
                "to_dit_bytes() is only supported for built-in Rust Sequential architectures ( >_< )"
            )
        return bytes(self._rust_seq.to_dit_bytes())

    @classmethod
    def from_dit_bytes(
        cls, data: bytes, device: Optional[str] = None
    ) -> "Sequential":
        dev = _resolve_device(device)
        return cls._from_rust(_rust_nn.Sequential.from_dit_bytes(data, dev), dev)

    def save(self, path: str) -> None:
        if self._rust_seq is None:
            raise ValueError(
                "save() is only supported for built-in Rust Sequential architectures ( >_< )"
            )
        self._rust_seq.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "Sequential":
        dev = _resolve_device(device)
        rust_seq = _rust_nn.Sequential.load(validate_dit_path(path), dev)
        return cls._from_rust(rust_seq, dev)

    def __len__(self) -> int:
        if self._rust_seq is not None and len(self._layers) == 0:
            return len(self._rust_seq)
        return len(self._layers)
