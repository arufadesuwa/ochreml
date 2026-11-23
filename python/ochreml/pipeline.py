"""
OchreML Pipeline & Meta-Estimator Protocol Utilities (＾▽＾)
Implements Pipeline, make_pipeline, estimator cloning, parameter routing,
and .dit/pickle persistence helpers (*≧ω≦*).
"""

from __future__ import annotations

import copy
import inspect
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def _is_estimator_like(obj: Any) -> bool:
    """Return True if obj looks like an estimator or transformer."""
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return False
    return hasattr(obj, "fit") or hasattr(obj, "get_params")


def _get_init_param_names(cls: type) -> List[str]:
    """Inspect __init__ signature and return positional/keyword parameter names."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []
    names: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(name)
    return names


def _get_estimator_params(est: Any, deep: bool = True) -> Dict[str, Any]:
    """
    Extract parameters from either a Scikit-Learn or OchreML estimator (o´∀｀o).
    """
    if est is None or isinstance(est, str):
        return {}
    if hasattr(est, "get_params"):
        return dict(est.get_params(deep=deep))

    out: Dict[str, Any] = {}
    for name in _get_init_param_names(est.__class__):
        if hasattr(est, name):
            val = getattr(est, name)
            out[name] = val
            if deep and _is_estimator_like(val):
                for sub_k, sub_v in _get_estimator_params(val, deep=True).items():
                    out[f"{name}__{sub_k}"] = sub_v
    return out


def _set_estimator_params(est: Any, **params: Any) -> Any:
    """
    Set parameters on either a Scikit-Learn or OchreML estimator (*≧ω≦*).
    Reconstructs Rust-backed OchreML wrappers in-place when they lack set_params().
    """
    if not params:
        return est
    if hasattr(est, "set_params"):
        return est.set_params(**params)

    cls = est.__class__
    valid_names = _get_init_param_names(cls)
    current = _get_estimator_params(est, deep=False)

    nested: Dict[str, Dict[str, Any]] = {}
    for key, val in params.items():
        if "__" in key:
            head, tail = key.split("__", 1)
            if head not in valid_names and not hasattr(est, head):
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {cls.__name__} ( >_< )"
                )
            nested.setdefault(head, {})[tail] = val
        else:
            if key not in valid_names and not hasattr(est, key):
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {cls.__name__} ( >_< )"
                )
            current[key] = val

    for head, sub_params in nested.items():
        sub_est = current.get(head, getattr(est, head, None))
        if sub_est is None:
            raise ValueError(
                f"Cannot set nested parameter on None attribute '{head}' ( >_< )"
            )
        _set_estimator_params(sub_est, **sub_params)
        current[head] = sub_est

    # Reconstruct Rust-backed OchreML estimator so internal _model gets new params
    try:
        rebuilt = cls(**current)
        est.__dict__.clear()
        est.__dict__.update(rebuilt.__dict__)
    except Exception:
        for k, v in current.items():
            setattr(est, k, v)
    return est


def clone_estimator(est: Any) -> Any:
    """
    Create an unfitted deep copy of an OchreML or Scikit-Learn estimator (＾▽＾).
    """
    if est is None or isinstance(est, (str, int, float, bool, bytes)):
        return est
    if isinstance(est, (list, tuple)):
        cloned_items = [clone_estimator(item) for item in est]
        return type(est)(cloned_items)

    cls = est.__class__
    if hasattr(est, "get_params"):
        params = est.get_params(deep=False)
        cloned_params = {
            k: (clone_estimator(v) if _is_estimator_like(v) else copy.deepcopy(v))
            for k, v in params.items()
        }
        return cls(**cloned_params)

    param_names = _get_init_param_names(cls)
    if param_names:
        kwargs: Dict[str, Any] = {}
        sig = inspect.signature(cls.__init__)
        for name in param_names:
            if hasattr(est, name):
                val = getattr(est, name)
                kwargs[name] = (
                    clone_estimator(val) if _is_estimator_like(val) else copy.deepcopy(val)
                )
            else:
                default = sig.parameters[name].default
                if default is not inspect.Parameter.empty:
                    kwargs[name] = copy.deepcopy(default)
        try:
            return cls(**kwargs)
        except Exception:
            pass

    return copy.deepcopy(est)


def _check_is_fitted(estimator: Any, attributes: Union[str, Sequence[str]]) -> None:
    """Verify that fitted attributes exist on estimator."""
    attrs = [attributes] if isinstance(attributes, str) else list(attributes)
    for attr in attrs:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            raise ValueError(
                f"This {type(estimator).__name__} instance is not fitted yet. "
                f"Call 'fit' before using this estimator (´-ω-`)"
            )


class _BaseMetaEstimator:
    """
    Shared Scikit-Learn parameter protocol & .dit persistence mixin (＾▽＾).
    """

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        out: Dict[str, Any] = {}
        for name in _get_init_param_names(self.__class__):
            val = getattr(self, name, None)
            out[name] = val
            if deep and _is_estimator_like(val):
                for sub_k, sub_v in _get_estimator_params(val, deep=True).items():
                    out[f"{name}__{sub_k}"] = sub_v
        return out

    def set_params(self, **params: Any) -> Any:
        """Set the parameters of this estimator."""
        if not params:
            return self
        valid_names = set(_get_init_param_names(self.__class__))
        nested: Dict[str, Dict[str, Any]] = {}
        for key, val in params.items():
            if "__" in key:
                head, tail = key.split("__", 1)
                if head not in valid_names:
                    raise ValueError(
                        f"Invalid parameter '{key}' for estimator {type(self).__name__} ( >_< )"
                    )
                nested.setdefault(head, {})[tail] = val
            else:
                if key not in valid_names:
                    raise ValueError(
                        f"Invalid parameter '{key}' for estimator {type(self).__name__} ( >_< )"
                    )
                setattr(self, key, val)

        for head, sub_params in nested.items():
            sub_est = getattr(self, head, None)
            if sub_est is None:
                raise ValueError(
                    f"Cannot set parameter on None attribute '{head}' ( >_< )"
                )
            _set_estimator_params(sub_est, **sub_params)
        return self

    def save(self, path: str) -> None:
        """Save this meta-estimator to a '.dit' file (*≧ω≦*)."""
        from ._preproc_utils import validate_dit_path

        path_str = validate_dit_path(path)
        with open(path_str, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> Any:
        """Load a meta-estimator from a '.dit' file (＾▽＾)."""
        from ._preproc_utils import validate_dit_path

        path_str = validate_dit_path(path)
        with open(path_str, "rb") as f:
            obj = pickle.load(f)
        if device is not None and hasattr(obj, "to"):
            obj.to(device)
        return obj


class Pipeline(_BaseMetaEstimator):
    """
    Pipeline of transforms with a final estimator (＾▽＾).

    Sequentially applies a list of transforms and a final estimator.
    Intermediate steps must implement `fit` and `transform`, or be `'passthrough'`.
    The final estimator only needs to implement `fit`.

    Parameters
    ----------
    steps : list of (str, estimator or 'passthrough') tuples
        List of (name, transform) tuples chained sequentially.
    memory : str or object or None, default=None
        Kept for Scikit-Learn API compatibility.
    verbose : bool, default=False
        If True, prints timing/progress information during fitting.

    Attributes
    ----------
    named_steps : dict
        Dictionary mapping step names to step estimators.
    classes_ : numpy.ndarray
        Class labels from the final estimator (if classifier).
    """

    def __init__(
        self,
        steps: Sequence[Tuple[str, Any]],
        *,
        memory: Optional[Any] = None,
        verbose: bool = False,
    ):
        self.steps = list(steps)
        self.memory = memory
        self.verbose = verbose
        self._validate_steps()

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("Pipeline steps cannot be empty ( >_< )")
        names = [name for name, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Pipeline step names must be unique, got {names} ( >_< )")
        for name, est in self.steps:
            if "__" in name:
                raise ValueError(
                    f"Pipeline step name '{name}' must not contain '__' ( >_< )"
                )
            if est is not None and est != "passthrough":
                if not hasattr(est, "fit"):
                    raise TypeError(
                        f"Step '{name}' ({type(est).__name__}) does not implement fit() (・`ω´・)"
                    )
        for name, est in self.steps[:-1]:
            if est is not None and est != "passthrough":
                if not (hasattr(est, "fit_transform") or hasattr(est, "transform")):
                    raise TypeError(
                        f"Intermediate step '{name}' ({type(est).__name__}) must implement "
                        "fit_transform() or transform() (・`ω´・)"
                    )

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Map step names to their estimators."""
        return dict(self.steps)

    @property
    def _final_estimator(self) -> Any:
        return self.steps[-1][1]

    @property
    def classes_(self) -> Any:
        est = self._final_estimator
        if est is None or est == "passthrough" or not hasattr(est, "classes_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'classes_' (´-ω-`)"
            )
        return est.classes_

    @property
    def n_features_in_(self) -> Optional[int]:
        for _, est in self.steps:
            if est is not None and est != "passthrough":
                return getattr(est, "n_features_in_", None)
        return None

    @property
    def feature_names_in_(self) -> Optional[Any]:
        for _, est in self.steps:
            if est is not None and est != "passthrough":
                return getattr(est, "feature_names_in_", None)
        return None

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str, slice]) -> Any:
        if isinstance(ind, slice):
            return Pipeline(self.steps[ind], memory=self.memory, verbose=self.verbose)
        if isinstance(ind, str):
            return self.named_steps[ind]
        if isinstance(ind, int) or (np is not None and isinstance(ind, np.integer)):
            return self.steps[int(ind)][1]
        raise TypeError(f"Unsupported Pipeline index type: {type(ind)} ( >_< )")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this pipeline and all nested steps (*≧ω≦*)."""
        out: Dict[str, Any] = {
            "memory": self.memory,
            "steps": self.steps,
            "verbose": self.verbose,
        }
        if not deep:
            return out
        for name, est in self.steps:
            out[name] = est
            if est is not None and est != "passthrough":
                for sub_k, sub_v in _get_estimator_params(est, deep=True).items():
                    out[f"{name}__{sub_k}"] = sub_v
        return out

    def set_params(self, **params: Any) -> "Pipeline":
        """Set parameters on the pipeline or on nested steps via 'step__param' (o´∀｀o)."""
        if not params:
            return self
        if "steps" in params:
            self.steps = list(params.pop("steps"))
        if "memory" in params:
            self.memory = params.pop("memory")
        if "verbose" in params:
            self.verbose = params.pop("verbose")

        step_map = {name: idx for idx, (name, _) in enumerate(self.steps)}
        nested: Dict[str, Dict[str, Any]] = {}
        for key, val in params.items():
            if "__" in key:
                step_name, sub_param = key.split("__", 1)
                if step_name not in step_map:
                    raise ValueError(
                        f"Unknown pipeline step '{step_name}' in parameter '{key}' ( >_< )"
                    )
                nested.setdefault(step_name, {})[sub_param] = val
            elif key in step_map:
                idx = step_map[key]
                self.steps[idx] = (key, val)
            else:
                raise ValueError(
                    f"Invalid parameter '{key}' for Pipeline ( >_< )"
                )

        for step_name, sub_params in nested.items():
            idx = step_map[step_name]
            est = self.steps[idx][1]
            if est is None or est == "passthrough":
                raise ValueError(
                    f"Cannot set parameters on '{est}' step '{step_name}' ( >_< )"
                )
            _set_estimator_params(est, **sub_params)

        self._validate_steps()
        return self

    def _split_fit_params(self, fit_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        step_params: Dict[str, Dict[str, Any]] = {name: {} for name, _ in self.steps}
        for key, val in fit_params.items():
            if "__" not in key:
                raise ValueError(
                    f"Pipeline.fit parameter '{key}' must be prefixed with '<step_name>__' ( >_< )"
                )
            step_name, param_name = key.split("__", 1)
            if step_name not in step_params:
                raise ValueError(
                    f"Unknown pipeline step '{step_name}' in fit parameter '{key}' ( >_< )"
                )
            step_params[step_name][param_name] = val
        return step_params

    def _fit_transformers(
        self, X: Any, y: Optional[Any] = None, **fit_params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        step_params = self._split_fit_params(fit_params)
        Xt = X
        for name, trans in self.steps[:-1]:
            if trans is None or trans == "passthrough":
                continue
            kw = step_params[name]
            if hasattr(trans, "fit_transform"):
                Xt = trans.fit_transform(Xt, y, **kw)
            else:
                Xt = trans.fit(Xt, y, **kw).transform(Xt)
        last_name, _ = self.steps[-1]
        return Xt, step_params[last_name]

    def _transform_to_final(self, X: Any) -> Any:
        Xt = X
        for _, trans in self.steps[:-1]:
            if trans is not None and trans != "passthrough":
                Xt = trans.transform(Xt)
        return Xt

    def fit(self, X: Any, y: Optional[Any] = None, **fit_params: Any) -> "Pipeline":
        """Fit all transforms sequentially and fit the final estimator (*≧ω≦*)."""
        Xt, last_params = self._fit_transformers(X, y, **fit_params)
        final_est = self._final_estimator
        if final_est is not None and final_est != "passthrough":
            final_est.fit(Xt, y, **last_params)
        return self

    def fit_transform(
        self, X: Any, y: Optional[Any] = None, **fit_params: Any
    ) -> Any:
        """Fit all transforms and apply final transform (＾▽＾)."""
        Xt, last_params = self._fit_transformers(X, y, **fit_params)
        final_est = self._final_estimator
        if final_est is None or final_est == "passthrough":
            return Xt
        if hasattr(final_est, "fit_transform"):
            return final_est.fit_transform(Xt, y, **last_params)
        return final_est.fit(Xt, y, **last_params).transform(Xt)

    def fit_predict(
        self, X: Any, y: Optional[Any] = None, **fit_params: Any
    ) -> Any:
        """Fit all transforms and predict with the final estimator (o´∀｀o)."""
        Xt, last_params = self._fit_transformers(X, y, **fit_params)
        final_est = self._final_estimator
        if hasattr(final_est, "fit_predict"):
            return final_est.fit_predict(Xt, y, **last_params)
        return final_est.fit(Xt, y, **last_params).predict(Xt)

    def transform(self, X: Any) -> Any:
        """Transform data through all pipeline steps (＾▽＾)."""
        Xt = self._transform_to_final(X)
        final_est = self._final_estimator
        if final_est is None or final_est == "passthrough":
            return Xt
        return final_est.transform(Xt)

    def predict(self, X: Any, **predict_params: Any) -> Any:
        """Transform data and predict with the final estimator (*≧ω≦*)."""
        Xt = self._transform_to_final(X)
        return self._final_estimator.predict(Xt, **predict_params)

    def predict_proba(self, X: Any, **predict_params: Any) -> Any:
        """Transform data and predict class probabilities with the final estimator."""
        Xt = self._transform_to_final(X)
        return self._final_estimator.predict_proba(Xt, **predict_params)

    def decision_function(self, X: Any) -> Any:
        """Transform data and compute decision function with the final estimator."""
        Xt = self._transform_to_final(X)
        return self._final_estimator.decision_function(Xt)

    def score(
        self, X: Any, y: Optional[Any] = None, sample_weight: Optional[Any] = None
    ) -> float:
        """Transform data and score with the final estimator (o´∀｀o)."""
        Xt = self._transform_to_final(X)
        if sample_weight is not None:
            return float(self._final_estimator.score(Xt, y, sample_weight=sample_weight))
        return float(self._final_estimator.score(Xt, y))

    def __repr__(self) -> str:
        return f"Pipeline(steps={self.steps!r}) (＾▽＾)"


def make_pipeline(
    *steps: Any,
    memory: Optional[Any] = None,
    verbose: bool = False,
) -> Pipeline:
    """
    Construct a Pipeline from the given estimators with auto-generated names (＾▽＾).
    """
    named_steps: List[Tuple[str, Any]] = []
    counts: Dict[str, int] = {}
    for est in steps:
        base = "passthrough" if est is None or est == "passthrough" else type(est).__name__.lower()
        cnt = counts.get(base, 0) + 1
        counts[base] = cnt
        name = f"{base}-{cnt}" if cnt > 1 else base
        named_steps.append((name, est))
    return Pipeline(named_steps, memory=memory, verbose=verbose)


__all__ = ["Pipeline", "make_pipeline"]
