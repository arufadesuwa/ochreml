"""
Comprehensive tests for OchreML CNN and Recurrent Sequence layers (`ochreml.nn`) (＾▽＾).
Covers Conv2d (stride/padding/dilation/groups), DepthwiseConv2d, MaxPool2d, AvgPool2d,
GlobalAvgPool2d, BatchNorm2d (train vs eval running stats), Flatten, ResidualBlock,
CNN Sequential .fit() + .dit round-trip, and RNN/GRU/LSTM (single/multi-layer,
bidirectional, forward + backward gradient updates).
"""

import numpy as np
import pytest

import ochreml
from ochreml import nn


def _numerical_grad(fn, arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(arr, dtype=np.float64)
    flat = arr.reshape(-1)
    g_flat = grad.reshape(-1)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        pos = float(fn(arr.copy()))
        flat[i] = orig - eps
        neg = float(fn(arr.copy()))
        flat[i] = orig
        g_flat[i] = (pos - neg) / (2.0 * eps)
    return grad.astype(np.float32)


def _numpy_conv2d(x, w, b, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    n, c_in, h, wid = x.shape
    c_out, c_per_group, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    eff_kh = dh * (kh - 1) + 1
    eff_kw = dw * (kw - 1) + 1
    h_out = (h + 2 * ph - eff_kh) // sh + 1
    w_out = (wid + 2 * pw - eff_kw) // sw + 1
    out = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)
    out_per_group = c_out // groups
    for ni in range(n):
        for g in range(groups):
            for oc_local in range(out_per_group):
                oc = g * out_per_group + oc_local
                ic_start = g * c_per_group
                for oh in range(h_out):
                    for ow in range(w_out):
                        patch = x_pad[
                            ni,
                            ic_start : ic_start + c_per_group,
                            oh * sh : oh * sh + eff_kh : dh,
                            ow * sw : ow * sw + eff_kw : dw,
                        ]
                        val = float(np.sum(patch * w[oc]))
                        if b is not None:
                            val += float(b[oc])
                        out[ni, oc, oh, ow] = val
    return out


@pytest.mark.parametrize(
    "in_c,out_c,k,stride,padding,dilation,groups,bias",
    [
        (2, 3, 3, 1, 1, 1, 1, True),
        (4, 6, 3, 2, 1, 1, 2, True),
        (2, 4, (2, 3), (1, 2), (1, 0), (2, 1), 1, False),
    ],
)
def test_conv2d_forward_parity_and_autograd(
    in_c, out_c, k, stride, padding, dilation, groups, bias
):
    rng = np.random.default_rng(11)
    x_np = rng.normal(0.0, 0.5, size=(2, in_c, 6, 7)).astype(np.float32)
    conv = nn.Conv2d(
        in_c,
        out_c,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        device="cpu",
        seed=23,
    )
    x = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y = conv(x)

    w_np = conv.weight.numpy()
    b_np = conv.bias.numpy() if conv.bias is not None else None
    ref = _numpy_conv2d(
        x_np,
        w_np,
        b_np,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=groups,
    )
    assert y.shape == ref.shape
    assert np.allclose(y.numpy(), ref, atol=1e-4)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None and conv.weight.grad is not None
    if bias:
        assert conv.bias is not None and conv.bias.grad is not None

    num_dx = _numerical_grad(
        lambda a: float(
            np.sum(
                _numpy_conv2d(
                    a, w_np, b_np, conv.stride, conv.padding, conv.dilation, groups
                )
            )
        ),
        x_np,
    )
    num_dw = _numerical_grad(
        lambda a: float(
            np.sum(
                _numpy_conv2d(
                    x_np, a, b_np, conv.stride, conv.padding, conv.dilation, groups
                )
            )
        ),
        w_np,
    )
    assert np.allclose(x.grad.numpy(), num_dx, atol=2e-2, rtol=2e-2)
    assert np.allclose(conv.weight.grad.numpy(), num_dw, atol=2e-2, rtol=2e-2)


def test_depthwise_conv2d_forward_and_backward():
    rng = np.random.default_rng(31)
    x_np = rng.normal(0.0, 0.6, size=(2, 3, 5, 5)).astype(np.float32)
    dw = nn.DepthwiseConv2d(
        3, depth_multiplier=2, kernel_size=3, stride=1, padding=1, bias=True, device="cpu", seed=9
    )
    x = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y = dw(x)
    assert y.shape == (2, 6, 5, 5)

    loss = y.mean()
    loss.backward()
    assert x.grad is not None and dw.weight.grad is not None
    assert dw.bias is not None and dw.bias.grad is not None
    assert np.linalg.norm(dw.weight.grad.numpy()) > 0.0


def test_pooling_and_flatten_layers():
    x_np = np.arange(1, 1 + 2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4)
    x = nn.Tensor(x_np, device="cpu", requires_grad=True)

    maxp = nn.MaxPool2d(kernel_size=2, stride=2)
    ym = maxp(x)
    assert ym.shape == (2, 2, 2, 2)
    expected_max = x_np.reshape(2, 2, 2, 2, 2, 2).max(axis=(3, 5))
    assert np.allclose(ym.numpy(), expected_max)
    ym.sum().backward()
    assert x.grad is not None
    assert np.isclose(x.grad.numpy().sum(), 2 * 2 * 2 * 2)

    x.zero_grad()
    avgp = nn.AvgPool2d(kernel_size=2, stride=2)
    ya = avgp(x)
    assert ya.shape == (2, 2, 2, 2)
    expected_avg = x_np.reshape(2, 2, 2, 2, 2, 2).mean(axis=(3, 5))
    assert np.allclose(ya.numpy(), expected_avg)
    ya.sum().backward()
    assert np.allclose(x.grad.numpy(), 0.25)

    x.zero_grad()
    gap_keep = nn.GlobalAvgPool2d(keep_dims=True)
    yg1 = gap_keep(x)
    assert yg1.shape == (2, 2, 1, 1)
    assert np.allclose(yg1.numpy(), x_np.mean(axis=(2, 3), keepdims=True))

    gap_flat = nn.GlobalAvgPool2d(keep_dims=False)
    yg2 = gap_flat(x)
    assert yg2.shape == (2, 2)
    assert np.allclose(yg2.numpy(), x_np.mean(axis=(2, 3)))

    flat = nn.Flatten(start_dim=1)
    yf = flat(x)
    assert yf.shape == (2, 32)
    yf.sum().backward()
    assert np.allclose(x.grad.numpy(), 1.0)


def test_batchnorm2d_train_vs_eval_running_stats():
    rng = np.random.default_rng(7)
    bn = nn.BatchNorm2d(3, eps=1e-5, momentum=0.2, affine=True, device="cpu")

    # Feed several training batches with non-zero channel means and non-unit variances
    bn.train()
    x_train_np = rng.normal(loc=4.0, scale=2.5, size=(6, 3, 4, 4)).astype(np.float32)
    x_train = nn.Tensor(x_train_np, device="cpu", requires_grad=True)
    y_train = bn(x_train)
    y_np = y_train.numpy()
    assert np.allclose(y_np.mean(axis=(0, 2, 3)), 0.0, atol=1e-4)
    assert np.allclose(y_np.std(axis=(0, 2, 3)), 1.0, atol=1e-3)

    rm_after_train = list(bn.running_mean)
    rv_after_train = list(bn.running_var)
    # running_mean moved from 0.0 toward ~4.0
    assert all(abs(m) > 0.4 for m in rm_after_train)

    y_train.sum().backward()
    assert x_train.grad is not None
    assert bn.weight is not None and bn.weight.grad is not None
    assert bn.bias is not None and bn.bias.grad is not None

    # Switch to eval(): running stats must remain frozen and be used for normalization
    bn.eval()
    x_eval_np = np.ones((2, 3, 2, 2), dtype=np.float32) * 3.0
    y_eval = bn(nn.Tensor(x_eval_np, device="cpu")).numpy()
    assert list(bn.running_mean) == rm_after_train
    assert list(bn.running_var) == rv_after_train

    expected_eval = np.empty_like(x_eval_np)
    for c in range(3):
        expected_eval[:, c] = (x_eval_np[:, c] - rm_after_train[c]) / np.sqrt(
            rv_after_train[c] + 1e-5
        )
    assert np.allclose(y_eval, expected_eval, atol=1e-4)


def test_residual_block_identity_and_downsample_autograd():
    rng = np.random.default_rng(5)
    x_np = rng.normal(0.0, 0.5, size=(2, 4, 6, 6)).astype(np.float32)

    rb_same = nn.ResidualBlock(4, 4, stride=1, device="cpu", seed=1)
    x1 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y1 = rb_same(x1)
    assert y1.shape == (2, 4, 6, 6)
    y1.mean().backward()
    assert x1.grad is not None
    assert any(p.grad is not None for p in rb_same.parameters())

    rb_down = nn.ResidualBlock(4, 8, stride=2, device="cpu", seed=2)
    rb_down.eval()
    x2 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y2 = rb_down(x2)
    assert y2.shape == (2, 8, 3, 3)
    rb_down.train()
    y2_tr = rb_down(x2)
    y2_tr.sum().backward()
    assert x2.grad is not None


def test_cnn_sequential_fit_and_dit_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    # 2 classes of 8x8 synthetic images: class 0 bright top-left, class 1 bright bottom-right
    n_per_class = 24
    X = rng.normal(0.0, 0.15, size=(2 * n_per_class, 1, 8, 8)).astype(np.float32)
    X[:n_per_class, 0, :4, :4] += 2.0
    X[n_per_class:, 0, 4:, 4:] += 2.0
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)

    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.ResidualBlock(4, 4, stride=1),
        nn.GlobalAvgPool2d(keep_dims=False),
        nn.Linear(4, 2),
        device="cpu",
    )
    hist = model.fit(
        X,
        y,
        epochs=15,
        batch_size=12,
        lr=0.03,
        optimizer="adamw",
        loss="cross_entropy",
        seed=42,
    )
    assert len(hist) == 15
    assert hist[-1] < hist[0] * 0.5

    probs = model.predict_proba(X)
    preds = model.predict(X)
    assert probs.shape == (48, 2)
    assert np.mean(np.argmax(probs, axis=1) == y) >= 0.90

    dit_path = str(tmp_path / "cnn_model.dit")
    ochreml.save(model, dit_path)
    loaded = ochreml.load(dit_path, device="cpu")
    assert isinstance(loaded, nn.Sequential)
    assert np.allclose(loaded.predict(X), preds, atol=1e-5)
    assert np.allclose(loaded.predict_proba(X), probs, atol=1e-5)


@pytest.mark.parametrize("cls_name", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("num_layers,bidirectional", [(1, False), (2, True)])
def test_rnn_gru_lstm_forward_and_bptt_update(cls_name, num_layers, bidirectional):
    rng = np.random.default_rng(101)
    batch, seq_len, in_dim, hid_dim = 3, 5, 4, 6
    dirs = 2 if bidirectional else 1

    x_np = rng.normal(0.0, 0.5, size=(batch, seq_len, in_dim)).astype(np.float32)
    target_np = np.ones((batch, seq_len, hid_dim * dirs), dtype=np.float32) * 0.5

    if cls_name == "RNN":
        layer = nn.RNN(
            in_dim,
            hid_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            bidirectional=bidirectional,
            device="cpu",
            seed=7,
        )
        h0 = nn.Tensor.zeros((num_layers * dirs, batch, hid_dim), device="cpu")
        hx = h0
    elif cls_name == "GRU":
        layer = nn.GRU(
            in_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            device="cpu",
            seed=7,
        )
        h0 = nn.Tensor.zeros((num_layers * dirs, batch, hid_dim), device="cpu")
        hx = h0
    else:
        layer = nn.LSTM(
            in_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            device="cpu",
            seed=7,
        )
        h0 = nn.Tensor.zeros((num_layers * dirs, batch, hid_dim), device="cpu")
        c0 = nn.Tensor.zeros((num_layers * dirs, batch, hid_dim), device="cpu")
        hx = (h0, c0)

    opt = nn.Adam(layer.parameters(), lr=0.05)
    mse = nn.MSELoss()
    x = nn.Tensor(x_np, device="cpu", requires_grad=True)
    target = nn.Tensor(target_np, device="cpu")

    losses = []
    for step_idx in range(6):
        opt.zero_grad()
        x.zero_grad()
        out, state = layer(x, hx if step_idx == 0 else None)
        assert out.shape == (batch, seq_len, hid_dim * dirs)
        if cls_name == "LSTM":
            hn, cn = state
            assert hn.shape == (num_layers * dirs, batch, hid_dim)
            assert cn.shape == (num_layers * dirs, batch, hid_dim)
        else:
            assert state.shape == (num_layers * dirs, batch, hid_dim)
        loss = mse(out, target)
        losses.append(loss.item())
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters())
        opt.step()

    assert losses[-1] < losses[0]
