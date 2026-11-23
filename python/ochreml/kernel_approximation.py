"""
OchreML Kernel Approximation Module
Provides Scikit-Learn-compatible RBFSampler (Random Fourier Features) accelerated by Rust backends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ._ochreml import RBFSampler as _RustRBFSampler
from ._preproc_utils import bytes_to_f64_2d, validate_dit_path
from ._utils import process_features, to_feature_matrix


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from . import get_default_device

    return get_default_device()


def _to_f64_array(X: Any) -> Any:
    is_buf, X_data = process_features(X)
    if is_buf:
        return X_data
    if np is not None:
        return np.ascontiguousarray(X_data, dtype=np.float64)
    return to_feature_matrix(X)


class RBFSampler:
    """
    Approximate a RBF kernel feature map using its Fourier transform.

    Parameters
    ----------
    gamma : float or {'scale'}, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2).
        If 'scale', uses 1 / (n_features * X.var()).
    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
    random_state : int or None, default=None
        Pseudo-random number generator seed for deterministic sampling.
    device : str or None, default=None
        Target compute device ('auto', 'cpu', 'cuda:0', 'wgpu', 'tpu:0').
    """

    _rust_cls = _RustRBFSampler

    def __init__(
        self,
        gamma: Union[float, str] = 1.0,
        n_components: int = 100,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.device = _resolve_device(device)
        self._init_rust_model()

    def _init_rust_model(self) -> None:
        gamma_val: Optional[float] = None
        gamma_mode: Optional[str] = None
        if isinstance(self.gamma, str):
            gamma_mode = self.gamma
        elif isinstance(self.gamma, (int, float)) and not isinstance(self.gamma, bool):
            gamma_val = float(self.gamma)
        else:
            raise TypeError(
                f"gamma must be 'scale' or a positive float, got {type(self.gamma)}"
            )

        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer")

        self._model = _RustRBFSampler(
            gamma_val=gamma_val,
            gamma_mode=gamma_mode,
            n_components=self.n_components,
            random_state=self.random_state,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustRBFSampler, device: str) -> "RBFSampler":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        if rust_obj.gamma_val is not None:
            inst.gamma = rust_obj.gamma_val
        elif rust_obj.gamma_mode is not None:
            inst.gamma = rust_obj.gamma_mode
        else:
            inst.gamma = "scale"
        inst.n_components = rust_obj.n_components
        inst.random_state = rust_obj.random_state
        inst.device = rust_obj.device_ or device
        return inst

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "RBFSampler":
        valid = {"gamma", "n_components", "random_state", "device"}
        for key, val in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter '{key}' for RBFSampler")
            if key == "device":
                setattr(self, key, _resolve_device(val))
            else:
                setattr(self, key, val)
        self._init_rust_model()
        return self

    def fit(self, X: Any, y: Optional[Any] = None) -> "RBFSampler":
        X_data = _to_f64_array(X)
        self._model.fit_buffer(X_data)
        return self

    def transform(self, X: Any) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        X_data = _to_f64_array(X)
        rows, cols, raw = self._model.fit_transform_buffer(X_data)
        return bytes_to_f64_2d(rows, cols, raw)

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RBFSampler":
        target_dev = _resolve_device(device)
        rust_obj = cls._rust_cls.load(validate_dit_path(path), target_dev)
        return cls._from_rust(rust_obj, target_dev)

    def to(self, device: str) -> "RBFSampler":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        rust_obj = _RustRBFSampler.from_bytes(state, target_dev)
        self._model = rust_obj
        if rust_obj.gamma_val is not None:
            self.gamma = rust_obj.gamma_val
        elif rust_obj.gamma_mode is not None:
            self.gamma = rust_obj.gamma_mode
        else:
            self.gamma = "scale"
        self.n_components = rust_obj.n_components
        self.random_state = rust_obj.random_state
        self.device = rust_obj.device_ or target_dev

    @property
    def random_weights_(self) -> Optional[Any]:
        res = self._model.random_weights_
        if res is None:
            return None
        rows, cols, raw = res
        return bytes_to_f64_2d(rows, cols, raw)

    @property
    def random_offset_(self) -> Optional[Any]:
        raw = self._model.random_offset_
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float64).copy()

    @property
    def gamma_(self) -> Optional[float]:
        return self._model.gamma_

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_


__all__ = ["RBFSampler"]
