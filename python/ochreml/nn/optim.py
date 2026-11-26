"""
OchreML Neural Network Optimizers, Gradient Clipping & LR Schedulers (＾▽＾)
Backed by Rust SGD, Adam, fused AdamW (N9), clip_grad_norm_, StepLR, CosineAnnealingLR, and WarmupCosineLR.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Union

from .._ochreml import nn as _rust_nn
from .modules import Parameter, Tensor, _resolve_device


def _collect_params(params: Iterable[Any]) -> List[Tensor]:
    collected: List[Tensor] = []
    for p in params:
        if isinstance(p, (Tensor, Parameter)):
            collected.append(p)
        else:
            raise TypeError(
                f"Optimizer parameters must be ochreml.nn.Tensor/Parameter, got {type(p).__name__} ( >_< )"
            )
    return collected


class Optimizer:
    """Base optimizer class wrapping Rust parameter updates (*≧ω≦*)."""

    def __init__(self, params: Iterable[Any], rust_opt: Any) -> None:
        self.params: List[Tensor] = _collect_params(params)
        self._rust = rust_opt

    @property
    def lr(self) -> float:
        return float(self._rust.lr)

    @lr.setter
    def lr(self, value: float) -> None:
        self._rust.lr = float(value)

    def step(self, grads: Optional[Any] = None) -> None:
        self._rust.step([p._tensor for p in self.params], grads=grads)

    def zero_grad(self) -> None:
        self._rust.zero_grad([p._tensor for p in self.params])


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum, Nesterov, and weight decay (o´∀｀o)."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        rust_opt = _rust_nn.SGD(
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, rust_opt)


class Adam(Optimizer):
    """Adam optimizer with bias correction and f32 master weights for BF16/F16 (＾▽＾)."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        rust_opt = _rust_nn.Adam(
            lr=float(lr),
            beta1=float(betas[0]),
            beta2=float(betas[1]),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, rust_opt)


class AdamW(Optimizer):
    """Fused AdamW optimizer backed by N9 Rust kernel (*≧ω≦*)."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        rust_opt = _rust_nn.AdamW(
            lr=float(lr),
            beta1=float(betas[0]),
            beta2=float(betas[1]),
            eps=float(eps),
            weight_decay=float(weight_decay),
            device=_resolve_device(device),
        )
        super().__init__(params, rust_opt)


def clip_grad_norm_(
    parameters: Iterable[Any], max_norm: float, grads: Optional[Any] = None
) -> float:
    """Clip global L2 gradient norm across `parameters` in Rust and return pre-clip norm (＾▽＾)."""
    params = _collect_params(parameters)
    return float(
        _rust_nn.clip_grad_norm_(
            [p._tensor for p in params], float(max_norm), grads=grads
        )
    )


class _BaseScheduler:
    def __init__(self, target: Union[Optimizer, float], rust_sched: Any) -> None:
        self.optimizer: Optional[Optimizer] = target if hasattr(target, "lr") else None
        self._rust = rust_sched

    def get_lr(self) -> float:
        return float(self._rust.get_lr())

    def step(self) -> float:
        new_lr = float(self._rust.step())
        if self.optimizer is not None:
            self.optimizer.lr = new_lr
        return new_lr


class StepLR(_BaseScheduler):
    """Decay learning rate by `gamma` every `step_size` epochs (o´∀｀o)."""

    def __init__(
        self,
        optimizer_or_lr: Union[Optimizer, float],
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        base_lr = float(
            optimizer_or_lr.lr
            if hasattr(optimizer_or_lr, "lr")
            else optimizer_or_lr
        )
        super().__init__(
            optimizer_or_lr,
            _rust_nn.StepLR(base_lr, int(step_size), gamma=float(gamma)),
        )


class CosineAnnealingLR(_BaseScheduler):
    """Cosine annealing LR scheduler (*≧ω≦*)."""

    def __init__(
        self,
        optimizer_or_lr: Union[Optimizer, float],
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        base_lr = float(
            optimizer_or_lr.lr
            if hasattr(optimizer_or_lr, "lr")
            else optimizer_or_lr
        )
        super().__init__(
            optimizer_or_lr,
            _rust_nn.CosineAnnealingLR(base_lr, int(T_max), eta_min=float(eta_min)),
        )


class WarmupCosineLR(_BaseScheduler):
    """Linear warmup followed by cosine decay LR scheduler (＾▽＾)."""

    def __init__(
        self,
        optimizer_or_lr: Union[Optimizer, float],
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        base_lr = float(
            optimizer_or_lr.lr
            if hasattr(optimizer_or_lr, "lr")
            else optimizer_or_lr
        )
        super().__init__(
            optimizer_or_lr,
            _rust_nn.WarmupCosineLR(
                base_lr, int(warmup_steps), int(total_steps), min_lr=float(min_lr)
            ),
        )
