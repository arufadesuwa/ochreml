"""
Comprehensive tests for OchreML core neural network primitives (`ochreml.nn`) (＾▽＾).
Covers Tensor ops & broadcasting, no_grad, autograd vs numerical gradients,
MLP/activation/embedding/loss layers, optimizers, LR schedulers, gradient clipping,
Sequential.fit() (classification & regression), bf16/fp16 dtypes, .dit round-trip,
and error handling.
"""

import math
import numpy as np
import pytest

import ochreml
from ochreml import nn


def _numerical_grad(fn, arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite-difference gradient for small float32 tensors."""
    grad = np.zeros_like(arr, dtype=np.float64)
    flat = arr.reshape(-1)
    g_flat = grad.reshape(-1)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        loss_pos = float(fn(arr.copy()))
        flat[i] = orig - eps
        loss_neg = float(fn(arr.copy()))
        flat[i] = orig
        g_flat[i] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad.astype(np.float32)


def test_tensor_creation_properties_and_ops():
    t = nn.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu")
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.numel == 6
    assert t.dtype == "float32"
    assert t.device == "cpu"
    assert not t.requires_grad
    assert t.node_id is None
    assert np.allclose(t.numpy(), [[1, 2, 3], [4, 5, 6]])
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    z = nn.Tensor.zeros((2, 2), device="cpu")
    o = nn.Tensor.ones((2, 2), device="cpu")
    r = nn.Tensor.randn((3, 4), mean=1.0, std=0.5, seed=7, device="cpu")
    u = nn.Tensor.uniform((3, 4), low=-2.0, high=2.0, seed=7, device="cpu")
    assert np.allclose(z.numpy(), 0.0)
    assert np.allclose(o.numpy(), 1.0)
    assert r.shape == (3, 4)
    assert np.all((u.numpy() >= -2.0) & (u.numpy() <= 2.0))

    reshaped = t.reshape(3, 2)
    assert reshaped.shape == (3, 2)
    transposed = t.transpose_last2()
    assert transposed.shape == (3, 2)
    assert np.allclose(transposed.numpy(), t.numpy().T)

    # Scalar & elementwise ops + matmul
    a = nn.Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    b = nn.Tensor([[2.0, 0.5], [1.0, -1.0]], device="cpu")
    assert np.allclose((a + b).numpy(), a.numpy() + b.numpy())
    assert np.allclose((a - b).numpy(), a.numpy() - b.numpy())
    assert np.allclose((a * b).numpy(), a.numpy() * b.numpy())
    assert np.allclose((-a).numpy(), -a.numpy())
    assert np.allclose((1.5 + a).numpy(), 1.5 + a.numpy())
    assert np.allclose((5.0 - a).numpy(), 5.0 - a.numpy())
    assert np.allclose((2.0 * a).numpy(), 2.0 * a.numpy())
    assert np.allclose((a @ b).numpy(), a.numpy() @ b.numpy())
    assert np.isclose(a.sum().item(), 10.0)
    assert np.isclose(a.mean().item(), 2.5)
    assert np.allclose(a.sum(axis=0).numpy(), [4.0, 6.0])
    assert np.allclose(a.mean(axis=1).numpy(), [1.5, 3.5])


def test_tensor_broadcasting_forward_and_backward():
    a_np = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)
    b_np = np.array([0.25, -0.5, 1.5], dtype=np.float32)
    c_np = np.array([[2.0], [-1.5]], dtype=np.float32)

    a = nn.Tensor(a_np, device="cpu", requires_grad=True)
    b = nn.Tensor(b_np, device="cpu", requires_grad=True)
    c = nn.Tensor(c_np, device="cpu", requires_grad=True)

    y = (a + b) * c
    expected = (a_np + b_np) * c_np
    assert y.shape == (2, 3)
    assert np.allclose(y.numpy(), expected, atol=1e-5)

    loss = y.sum()
    loss.backward()
    assert a.grad is not None and b.grad is not None and c.grad is not None

    expected_da = np.broadcast_to(c_np, a_np.shape)
    expected_db = expected_da.sum(axis=0)
    expected_dc = (a_np + b_np).sum(axis=1, keepdims=True)
    assert np.allclose(a.grad.numpy(), expected_da, atol=1e-5)
    assert np.allclose(b.grad.numpy(), expected_db, atol=1e-5)
    assert np.allclose(c.grad.numpy(), expected_dc, atol=1e-5)


def test_no_grad_context_decorator_and_detach():
    assert nn.is_grad_enabled()
    x = nn.Tensor([[1.0, 2.0]], device="cpu", requires_grad=True)

    with nn.no_grad():
        assert not nn.is_grad_enabled()
        y = (x * 3.0 + 1.0).relu()
        assert not y.requires_grad
        assert y.node_id is None
    assert nn.is_grad_enabled()

    @nn.no_grad()
    def infer(inp: nn.Tensor) -> nn.Tensor:
        assert not nn.is_grad_enabled()
        return inp.sigmoid()

    out = infer(x)
    assert not out.requires_grad
    assert nn.is_grad_enabled()

    detached = (x * 2.0).detach()
    assert not detached.requires_grad
    assert detached.node_id is None


@pytest.mark.parametrize(
    "act_cls,np_fn",
    [
        (nn.ReLU, lambda x: np.maximum(x, 0.0)),
        (nn.Sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x))),
        (nn.Tanh, np.tanh),
        (
            nn.GELU,
            lambda x: 0.5
            * x
            * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3)))),
        ),
        (nn.SiLU, lambda x: x / (1.0 + np.exp(-x))),
    ],
)
def test_activations_forward_and_autograd_vs_numerical(act_cls, np_fn):
    rng = np.random.default_rng(123)
    x_np = rng.uniform(-1.5, 1.5, size=(2, 4)).astype(np.float32)
    # Avoid kink near 0 for ReLU finite difference
    x_np[np.abs(x_np) < 0.15] = 0.35
    w_np = rng.uniform(0.2, 1.2, size=(2, 4)).astype(np.float32)

    act = act_cls()
    x = nn.Tensor(x_np, device="cpu", requires_grad=True)
    w = nn.Tensor(w_np, device="cpu", requires_grad=False)
    out = act(x)
    assert np.allclose(out.numpy(), np_fn(x_np), atol=1e-4)

    loss = (out * w).sum()
    loss.backward()
    assert x.grad is not None

    num_grad = _numerical_grad(lambda arr: float(np.sum(np_fn(arr) * w_np)), x_np, eps=1e-3)
    assert np.allclose(x.grad.numpy(), num_grad, atol=2e-2, rtol=2e-2)


def test_linear_and_mse_autograd_vs_numerical():
    rng = np.random.default_rng(7)
    x_np = rng.normal(0.0, 1.0, size=(3, 4)).astype(np.float32)
    target_np = rng.normal(0.0, 1.0, size=(3, 2)).astype(np.float32)

    layer = nn.Linear(4, 2, bias=True, device="cpu", seed=9)
    mse = nn.MSELoss(reduction="mean")

    x = nn.Tensor(x_np, device="cpu", requires_grad=True)
    target = nn.Tensor(target_np, device="cpu", requires_grad=False)
    pred = layer(x)
    loss = mse(pred, target)
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None

    w_np = layer.weight.numpy().copy()
    b_np = layer.bias.numpy().copy()

    def eval_fn(x_arr, w_arr, b_arr):
        p = x_arr @ w_arr.T + b_arr
        return float(np.mean((p - target_np) ** 2))

    num_dx = _numerical_grad(lambda a: eval_fn(a, w_np, b_np), x_np)
    num_dw = _numerical_grad(lambda a: eval_fn(x_np, a, b_np), w_np)
    num_db = _numerical_grad(lambda a: eval_fn(x_np, w_np, a), b_np)

    assert np.allclose(x.grad.numpy(), num_dx, atol=2e-2, rtol=2e-2)
    assert np.allclose(layer.weight.grad.numpy(), num_dw, atol=2e-2, rtol=2e-2)
    assert np.allclose(layer.bias.grad.numpy(), num_db, atol=2e-2, rtol=2e-2)

    # Bias-free linear
    lin_nobias = nn.Linear(4, 3, bias=False, device="cpu", seed=5)
    assert lin_nobias.bias is None
    assert lin_nobias(nn.Tensor(x_np, device="cpu")).shape == (3, 3)


def test_dropout_train_vs_eval_and_embedding():
    x = nn.Tensor.ones((4, 16), device="cpu")
    drop = nn.Dropout(p=0.5, seed=42)
    drop.train()
    y_train = drop(x).numpy()
    assert np.any(y_train == 0.0)
    assert np.any(np.isclose(y_train, 2.0))

    drop.eval()
    y_eval = drop(x).numpy()
    assert np.allclose(y_eval, 1.0)

    # Embedding 1D and 2D + backward update
    emb = nn.Embedding(10, 4, device="cpu", seed=11)
    out_1d = emb([1, 3, 5])
    assert out_1d.shape == (3, 4)

    out_2d = emb([[0, 2, 4], [1, 3, 5]])
    assert out_2d.shape == (2, 3, 4)
    loss = out_2d.sum()
    loss.backward()
    assert emb.weight.grad is not None
    grad_np = emb.weight.grad.numpy()
    assert np.allclose(grad_np[0], 1.0)
    assert np.allclose(grad_np[6], 0.0)


def test_cross_entropy_loss_autograd_smoothing_and_ignore_index():
    rng = np.random.default_rng(42)
    logits_np = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float32)
    targets = [1, 3, 0, 4]

    ce = nn.CrossEntropyLoss()
    logits = nn.Tensor(logits_np, device="cpu", requires_grad=True)
    loss = ce(logits, targets)
    loss.backward()
    assert logits.grad is not None

    def np_ce(arr):
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        log_sm = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return float(-np.mean([log_sm[i, t] for i, t in enumerate(targets)]))

    assert np.isclose(loss.item(), np_ce(logits_np), atol=1e-5)
    num_grad = _numerical_grad(np_ce, logits_np)
    assert np.allclose(logits.grad.numpy(), num_grad, atol=2e-2, rtol=2e-2)

    # label_smoothing and ignore_index
    ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=255)
    logits2 = nn.Tensor(logits_np, device="cpu", requires_grad=True)
    loss2 = ce_smooth(logits2, [1, 255, 0, 255])
    loss2.backward()
    assert math.isfinite(loss2.item()) and loss2.item() > 0.0
    assert np.allclose(logits2.grad.numpy()[1], 0.0)
    assert np.allclose(logits2.grad.numpy()[3], 0.0)


@pytest.mark.parametrize(
    "opt_factory",
    [
        lambda p: nn.SGD(p, lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4),
        lambda p: nn.Adam(p, lr=0.05, betas=(0.9, 0.999), weight_decay=1e-4),
        lambda p: nn.AdamW(p, lr=0.05, betas=(0.9, 0.999), weight_decay=1e-2, device="cpu"),
    ],
)
def test_optimizers_converge_on_linear_problem(opt_factory):
    rng = np.random.default_rng(0)
    x_np = rng.normal(0.0, 1.0, size=(32, 3)).astype(np.float32)
    true_w = np.array([[2.0, -1.5, 0.5]], dtype=np.float32)
    y_np = x_np @ true_w.T + 0.75

    layer = nn.Linear(3, 1, bias=True, device="cpu", seed=17)
    mse = nn.MSELoss()
    opt = opt_factory(layer.parameters())

    x = nn.Tensor(x_np, device="cpu")
    y = nn.Tensor(y_np, device="cpu")

    initial_loss = None
    final_loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = mse(layer(x), y)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.25


def test_schedulers_and_clip_grad_norm():
    p = nn.Parameter([[3.0, 4.0]], device="cpu")
    opt = nn.SGD([p], lr=0.1)

    step_sched = nn.StepLR(opt, step_size=2, gamma=0.5)
    assert np.isclose(step_sched.get_lr(), 0.1)
    step_sched.step()
    assert np.isclose(opt.lr, 0.1)
    step_sched.step()
    assert np.isclose(opt.lr, 0.05)

    cos_sched = nn.CosineAnnealingLR(0.1, T_max=4, eta_min=0.01)
    lrs = [cos_sched.step() for _ in range(4)]
    assert lrs[0] > lrs[1] > lrs[2] >= lrs[3]
    assert np.isclose(lrs[-1], 0.01, atol=1e-5)

    warm_sched = nn.WarmupCosineLR(0.2, warmup_steps=3, total_steps=7, min_lr=0.02)
    w_lrs = [warm_sched.get_lr()] + [warm_sched.step() for _ in range(7)]
    assert w_lrs[0] < w_lrs[1] < w_lrs[2]
    assert np.isclose(w_lrs[2], 0.2, atol=1e-5)
    assert w_lrs[3] > w_lrs[5] > w_lrs[7]
    assert np.isclose(w_lrs[-1], 0.02, atol=1e-5)

    # clip_grad_norm_
    p.zero_grad()
    loss = (p * 1.0).sum()  # grad = [1, 1], norm = sqrt(2)
    loss.backward()
    pre_norm = nn.clip_grad_norm_([p], max_norm=0.5)
    assert np.isclose(pre_norm, math.sqrt(2.0), atol=1e-5)
    clipped_norm = float(np.linalg.norm(p.grad.numpy()))
    assert np.isclose(clipped_norm, 0.5, atol=1e-5)


def test_sequential_fit_classification_and_regression_and_dit(tmp_path):
    rng = np.random.default_rng(42)
    # 3-class linearly separable clusters
    centers = np.array([[2.5, 0.0], [-2.5, 2.0], [-2.5, -2.0]], dtype=np.float32)
    x_list, y_list = [], []
    for cls_idx, c in enumerate(centers):
        pts = c + rng.normal(0.0, 0.35, size=(40, 2)).astype(np.float32)
        x_list.append(pts)
        y_list.extend([cls_idx] * 40)
    X_cls = np.vstack(x_list)
    y_cls = np.array(y_list, dtype=np.int64)

    clf = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        device="cpu",
    )
    history = clf.fit(
        X_cls,
        y_cls,
        epochs=25,
        batch_size=16,
        lr=0.03,
        optimizer="adamw",
        loss="cross_entropy",
        seed=42,
    )
    assert len(history) == 25
    assert history[-1] < history[0] * 0.4

    probs = clf.predict_proba(X_cls)
    preds_logits = clf.predict(X_cls)
    assert probs.shape == (120, 3)
    assert preds_logits.shape == (120, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    acc = float(np.mean(np.argmax(probs, axis=1) == y_cls))
    assert acc >= 0.95

    # .dit round-trip via ochreml.save / ochreml.load
    dit_path = str(tmp_path / "mlp_clf.dit")
    ochreml.save(clf, dit_path)
    loaded = ochreml.load(dit_path, device="cpu")
    assert isinstance(loaded, nn.Sequential)
    assert np.allclose(loaded.predict(X_cls), preds_logits, atol=1e-5)
    assert np.allclose(loaded.predict_proba(X_cls), probs, atol=1e-5)

    # bytes round-trip
    raw_bytes = clf.to_dit_bytes()
    from_bytes = nn.Sequential.from_dit_bytes(raw_bytes, device="cpu")
    assert np.allclose(from_bytes.predict(X_cls), preds_logits, atol=1e-5)

    # Regression convergence
    X_reg = rng.uniform(-1.5, 1.5, size=(96, 3)).astype(np.float32)
    y_reg = (2.0 * X_reg[:, :1] - 1.5 * X_reg[:, 1:2] + 0.5 * X_reg[:, 2:3] + 0.3).astype(
        np.float32
    )
    reg = nn.Sequential(
        nn.Linear(3, 16),
        nn.GELU(),
        nn.Linear(16, 1),
        device="cpu",
    )
    reg_hist = reg.fit(
        X_reg,
        y_reg,
        out_dim=1,
        epochs=35,
        batch_size=16,
        lr=0.03,
        optimizer="adamw",
        loss="mse",
        grad_clip=1.0,
        seed=7,
    )
    assert reg_hist[-1] < reg_hist[0] * 0.2
    reg_pred = reg.predict(X_reg)
    assert reg_pred.shape == (96, 1)
    mse_val = float(np.mean((reg_pred - y_reg) ** 2))
    assert mse_val < 0.1


@pytest.mark.parametrize("dtype_name", [nn.bfloat16, nn.float16])
def test_low_precision_dtype_roundtrip_and_training(dtype_name):
    t = nn.Tensor([[1.25, -0.5, 2.0]], dtype="float32", device="cpu")
    t_lp = t.to_dtype(dtype_name)
    assert t_lp.dtype == dtype_name
    t_back = t_lp.to(nn.float32)
    assert t_back.dtype == "float32"
    assert np.allclose(t_back.numpy(), t.numpy(), atol=2e-2)

    rng = np.random.default_rng(19)
    X = rng.normal(0.0, 1.0, size=(48, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)

    model = nn.Sequential(
        nn.Linear(4, 12, dtype=dtype_name),
        nn.SiLU(),
        nn.Linear(12, 2, dtype=dtype_name),
        dtype=dtype_name,
        device="cpu",
    )
    hist = model.fit(
        X, y, epochs=20, batch_size=16, lr=0.03, optimizer="adamw", loss="cross_entropy", seed=5
    )
    assert len(hist) == 20
    assert hist[-1] < hist[0]
    preds = model.predict(X)
    assert preds.shape == (48, 2)


def test_invalid_inputs_and_dit_extension_errors(tmp_path):
    seq = nn.Sequential(nn.Linear(4, 2), device="cpu")
    bad_ext = str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match=r"\.dit"):
        ochreml.save(seq, bad_ext)
    with pytest.raises(ValueError, match=r"\.dit"):
        ochreml.load(bad_ext)
    with pytest.raises(ValueError, match=r"\.dit"):
        seq.save(str(tmp_path / "model.bin"))

    with pytest.raises(ValueError):
        nn.Tensor([1.0, 2.0], dtype="int8", device="cpu")

    a = nn.Tensor([[1.0, 2.0, 3.0]], device="cpu")
    b = nn.Tensor([[1.0, 2.0]], device="cpu")
    with pytest.raises(ValueError):
        _ = a @ b

    with pytest.raises(ValueError):
        _ = a.item()

    # Non-built-in module inside Sequential cannot call Rust-only fit()
    class CustomAct(nn.Module):
        def forward(self, x):
            return x

    custom_seq = nn.Sequential(nn.Linear(2, 2), CustomAct(), device="cpu")
    with pytest.raises(ValueError):
        custom_seq.fit(np.zeros((4, 2), dtype=np.float32), [0, 1, 0, 1])
