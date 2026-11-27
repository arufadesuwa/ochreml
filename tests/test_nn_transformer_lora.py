"""
Comprehensive tests for OchreML Normalization, Attention, Transformer, CausalLM, and LoRA (`ochreml.nn`) (＾▽＾).
Covers LayerNorm, RMSNorm, PositionalEncoding, RoPE, ScaledDotProductAttention (Flash-style tiled
vs naive numpy), MultiHeadAttention (MHA & GQA), KVCache incremental decode parity,
SwiGLU, FeedForward, TransformerEncoder/Decoder, CausalLM (forward/train/generate/.dit),
and LoRALinear (adapter training, merge/unmerge parity, adapter-only .dit save/load).
"""

import math
import numpy as np
import pytest

import ochreml
from ochreml import nn


def _naive_attention(q, k, v, causal=False, kv_offset=0):
    """Reference numpy attention supporting GQA/MQA."""
    b, h_q, s_q, d = q.shape
    _, h_kv, s_kv, _ = k.shape
    group = h_q // h_kv
    k_rep = np.repeat(k, group, axis=1)
    v_rep = np.repeat(v, group, axis=1)
    scale = 1.0 / math.sqrt(d)
    scores = np.matmul(q, np.swapaxes(k_rep, -1, -2)) * scale
    if causal:
        q_pos = kv_offset + np.arange(s_q)[:, None]
        k_pos = np.arange(s_kv)[None, :]
        mask = k_pos > q_pos
        scores = np.where(mask[None, None, :, :], -1e9, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, v_rep).astype(np.float32)


def test_layernorm_and_rmsnorm_parity_and_autograd():
    rng = np.random.default_rng(11)
    x_np = rng.normal(1.5, 2.0, size=(2, 4, 8)).astype(np.float32)

    ln = nn.LayerNorm(8, eps=1e-5, elementwise_affine=True, device="cpu")
    x1 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y1 = ln(x1)
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    ref_ln = (x_np - mean) / np.sqrt(var + 1e-5)
    assert np.allclose(y1.numpy(), ref_ln, atol=1e-4)

    (y1 * nn.Tensor.randn((2, 4, 8), seed=3, device="cpu")).sum().backward()
    assert x1.grad is not None
    assert ln.gamma is not None and ln.gamma.grad is not None
    assert ln.beta is not None and ln.beta.grad is not None

    ln_noaff = nn.LayerNorm(8, elementwise_affine=False, device="cpu")
    assert ln_noaff.gamma is None and ln_noaff.beta is None
    assert np.allclose(ln_noaff(nn.Tensor(x_np, device="cpu")).numpy(), ref_ln, atol=1e-4)

    rms = nn.RMSNorm(8, eps=1e-5, device="cpu")
    x2 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y2 = rms(x2)
    ref_rms = x_np / np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + 1e-5)
    assert np.allclose(y2.numpy(), ref_rms, atol=1e-4)
    (y2 * nn.Tensor.randn((2, 4, 8), seed=5, device="cpu")).sum().backward()
    assert x2.grad is not None and rms.gamma.grad is not None


def test_positional_encoding_and_rope():
    pe = nn.PositionalEncoding(max_len=16, d_model=8)
    x_np = np.zeros((2, 5, 8), dtype=np.float32)
    y_pe = pe(nn.Tensor(x_np, device="cpu"), offset=2).numpy()
    for pos_idx in range(5):
        p = pos_idx + 2
        for i in range(4):
            angle = p / (10000.0 ** (2.0 * i / 8.0))
            assert np.isclose(y_pe[0, pos_idx, 2 * i], math.sin(angle), atol=1e-5)
            assert np.isclose(y_pe[0, pos_idx, 2 * i + 1], math.cos(angle), atol=1e-5)

    # RoPE preserves vector L2 norm per head
    rng = np.random.default_rng(19)
    q_np = rng.normal(0.0, 1.0, size=(2, 3, 6, 8)).astype(np.float32)
    rope = nn.RoPE(head_dim=8, max_seq_len=32, theta=10000.0)
    q = nn.Tensor(q_np, device="cpu", requires_grad=True)
    q_rot = rope(q, offset=3)
    assert q_rot.shape == q_np.shape
    norms_before = np.linalg.norm(q_np, axis=-1)
    norms_after = np.linalg.norm(q_rot.numpy(), axis=-1)
    assert np.allclose(norms_before, norms_after, atol=1e-5)
    q_rot.sum().backward()
    assert q.grad is not None


@pytest.mark.parametrize("causal,kv_offset", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize("h_q,h_kv,s_q,s_kv", [(4, 4, 37, 37), (4, 2, 12, 16)])
def test_sdpa_tiled_vs_naive_numpy_and_backward(causal, kv_offset, h_q, h_kv, s_q, s_kv):
    if kv_offset > 0 and s_kv < kv_offset + s_q:
        s_kv = kv_offset + s_q
    rng = np.random.default_rng(77)
    d = 16
    q_np = rng.normal(0.0, 0.5, size=(2, h_q, s_q, d)).astype(np.float32)
    k_np = rng.normal(0.0, 0.5, size=(2, h_kv, s_kv, d)).astype(np.float32)
    v_np = rng.normal(0.0, 0.5, size=(2, h_kv, s_kv, d)).astype(np.float32)

    sdpa = nn.ScaledDotProductAttention()
    q = nn.Tensor(q_np, device="cpu", requires_grad=True)
    k = nn.Tensor(k_np, device="cpu", requires_grad=True)
    v = nn.Tensor(v_np, device="cpu", requires_grad=True)

    out = sdpa(q, k, v, causal=causal, kv_offset=kv_offset)
    ref = _naive_attention(q_np, k_np, v_np, causal=causal, kv_offset=kv_offset)
    assert out.shape == ref.shape
    assert np.allclose(out.numpy(), ref, atol=1e-4, rtol=1e-4)

    out.mean().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None


@pytest.mark.parametrize("num_kv_heads,use_rope", [(4, False), (2, True), (1, True)])
def test_mha_and_gqa_kv_cache_incremental_parity(num_kv_heads, use_rope):
    rng = np.random.default_rng(29)
    b, s, d_model, num_heads = 2, 6, 16, 4
    head_dim = d_model // num_heads

    mha = nn.MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        bias=True,
        causal=True,
        use_rope=use_rope,
        max_seq_len=32,
        device="cpu",
        seed=13,
    )
    x_np = rng.normal(0.0, 0.5, size=(b, s, d_model)).astype(np.float32)
    x_full = nn.Tensor(x_np, device="cpu", requires_grad=True)
    full_out = mha(x_full, x_full, x_full)
    assert full_out.shape == (b, s, d_model)

    full_out.sum().backward()
    assert x_full.grad is not None
    assert all(p.grad is not None for p in mha.parameters())

    # Incremental prefill (first 3 tokens) + 1-by-1 decode (remaining 3 tokens)
    cache = nn.KVCache(batch=b, num_kv_heads=num_kv_heads, max_seq_len=32, head_dim=head_dim)
    with nn.no_grad():
        prefill_in = nn.Tensor(x_np[:, :3, :], device="cpu")
        prefill_out = mha(prefill_in, prefill_in, prefill_in, kv_cache=cache)
        assert cache.cur_len() == 3
        step_outs = [prefill_out.numpy()]
        for pos in range(3, s):
            tok_in = nn.Tensor(x_np[:, pos : pos + 1, :], device="cpu")
            tok_out = mha(tok_in, tok_in, tok_in, kv_cache=cache)
            assert cache.cur_len() == pos + 1
            step_outs.append(tok_out.numpy())

    inc_full = np.concatenate(step_outs, axis=1)
    assert np.allclose(inc_full, full_out.numpy(), atol=1e-4)

    cache.reset()
    assert cache.cur_len() == 0


def test_swiglu_feedforward_and_transformer_encoder_decoder():
    rng = np.random.default_rng(55)
    b, s, d_model, d_ff, n_heads = 2, 5, 16, 32, 4
    x_np = rng.normal(0.0, 0.5, size=(b, s, d_model)).astype(np.float32)

    swiglu = nn.SwiGLU(d_model, d_ff, bias=False, device="cpu", seed=3)
    x1 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y_sg = swiglu(x1)
    assert y_sg.shape == (b, s, d_model)
    y_sg.mean().backward()
    assert x1.grad is not None and all(p.grad is not None for p in swiglu.parameters())

    ff = nn.FeedForward(d_model, d_ff, bias=True, device="cpu", seed=4)
    x2 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y_ff = ff(x2)
    assert y_ff.shape == (b, s, d_model)
    y_ff.mean().backward()
    assert x2.grad is not None and all(p.grad is not None for p in ff.parameters())

    enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, pre_norm=True, device="cpu", seed=5)
    enc = nn.TransformerEncoder(2, d_model, n_heads, d_ff, pre_norm=False, device="cpu", seed=6)
    x3 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y_enc_l = enc_layer(x3)
    y_enc = enc(y_enc_l)
    assert y_enc.shape == (b, s, d_model)
    y_enc.mean().backward()
    assert x3.grad is not None and all(p.grad is not None for p in enc.parameters())

    mem = nn.Tensor(rng.normal(0.0, 0.5, size=(b, 4, d_model)).astype(np.float32), device="cpu")
    dec_layer = nn.TransformerDecoderLayer(
        d_model, n_heads, d_ff, with_cross_attn=True, use_swiglu=True, device="cpu", seed=7
    )
    dec = nn.TransformerDecoder(
        2, d_model, n_heads, d_ff, with_cross_attn=False, use_swiglu=True, device="cpu", seed=8
    )
    x4 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    y_dec_l = dec_layer(x4, memory=mem)
    assert y_dec_l.shape == (b, s, d_model)
    y_dec_l.mean().backward()
    assert x4.grad is not None

    x5 = nn.Tensor(x_np, device="cpu", requires_grad=True)
    caches = [nn.KVCache(b, n_heads, 16, d_model // n_heads) for _ in range(2)]
    y_dec = dec(x5, kv_caches=caches)
    assert y_dec.shape == (b, s, d_model)
    assert all(c.cur_len() == s for c in caches)


def test_causal_lm_forward_train_generate_kv_cache_and_dit(tmp_path):
    lm = nn.CausalLM(
        vocab_size=24,
        d_model=16,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_hidden_dim=32,
        max_seq_len=32,
        device="cpu",
        seed=42,
    )
    assert lm.num_parameters() > 0

    # Full forward vs incremental KV cache forward parity
    prompt = [1, 5, 9, 3, 7]
    with nn.no_grad():
        full_logits = lm(prompt).numpy()
        assert full_logits.shape == (1, 5, 24)

        caches = lm.new_kv_caches(batch=1)
        prefill_logits = lm(prompt[:3], kv_caches=caches).numpy()
        step_logits = [prefill_logits]
        for tok in prompt[3:]:
            step_logits.append(lm([tok], kv_caches=caches).numpy())
        inc_logits = np.concatenate(step_logits, axis=1)
        assert np.allclose(inc_logits, full_logits, atol=1e-4)

    # Training step reduces next-token cross-entropy loss
    inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    targets = [2, 3, 4, 5, 6, 7, 8, 9]
    ce = nn.CrossEntropyLoss()
    opt = nn.AdamW(lm.parameters(), lr=0.02, weight_decay=0.0, device="cpu")
    losses = []
    for _ in range(5):
        opt.zero_grad()
        logits = lm(inputs)  # [2, 4, 24]
        loss = ce(logits.reshape(8, 24), targets)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]

    # Generation: greedy, top-k, top-p
    g_greedy_1 = lm.generate([1, 2, 3], max_new_tokens=6, temperature=0.0, seed=11)
    g_greedy_2 = lm.generate([1, 2, 3], max_new_tokens=6, temperature=0.0, seed=99)
    assert len(g_greedy_1) == 9
    assert g_greedy_1[:3] == [1, 2, 3]
    assert g_greedy_1 == g_greedy_2

    g_topk = lm.generate([1, 2, 3], max_new_tokens=5, temperature=0.8, top_k=5, seed=7)
    g_topp = lm.generate([1, 2, 3], max_new_tokens=5, temperature=0.8, top_p=0.9, seed=7)
    assert len(g_topk) == 8 and len(g_topp) == 8
    assert all(0 <= t < 24 for t in g_topk + g_topp)

    # .dit round-trip via ochreml.save / ochreml.load
    dit_path = str(tmp_path / "causal_lm.dit")
    ochreml.save(lm, dit_path)
    loaded = ochreml.load(dit_path, device="cpu")
    assert isinstance(loaded, nn.CausalLM)
    assert loaded.vocab_size == 24 and loaded.num_kv_heads == 2
    with nn.no_grad():
        assert np.allclose(loaded(inputs).numpy(), lm(inputs).numpy(), atol=1e-5)
    assert loaded.generate([1, 2, 3], max_new_tokens=6, temperature=0.0, seed=11) == g_greedy_1


def test_causal_lm_train_steps():
    lm = nn.CausalLM(
        vocab_size=24,
        d_model=16,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_hidden_dim=32,
        max_seq_len=32,
        device="cpu",
        seed=42,
    )
    params_before = [p.numpy().copy() for p in lm.parameters()]
    inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    targets = [2, 3, 4, 5, 6, 7, 8, 9]

    losses = lm.train_steps(inputs, targets, steps=5, lr=0.02)
    assert len(losses) == 5
    assert all(math.isfinite(val) for val in losses)
    assert losses[-1] < losses[0]

    params_after = [p.numpy() for p in lm.parameters()]
    assert len(params_after) == len(params_before)
    assert any(
        not np.allclose(b, a, atol=1e-7) for b, a in zip(params_before, params_after)
    )


def test_lora_linear_training_merge_unmerge_and_adapter_dit(tmp_path):
    rng = np.random.default_rng(13)
    base_lin = nn.Linear(8, 6, bias=True, device="cpu", seed=5)
    base_weight_before = base_lin.weight.numpy().copy()

    lora = nn.LoRALinear.from_linear(base_lin, rank=2, alpha=4.0, seed=19)
    assert not lora.merged
    assert lora.rank == 2 and np.isclose(lora.alpha, 4.0)
    assert len(lora.parameters()) == 2  # only lora_a and lora_b

    x_np = rng.normal(0.0, 1.0, size=(6, 8)).astype(np.float32)
    y_np = rng.normal(0.5, 0.5, size=(6, 6)).astype(np.float32)
    x = nn.Tensor(x_np, device="cpu")
    y = nn.Tensor(y_np, device="cpu")

    opt = nn.Adam(lora.parameters(), lr=0.05)
    mse = nn.MSELoss()
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = mse(lora(x), y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]

    # Base weight must remain frozen
    assert np.allclose(lora._rust.base_weight.numpy(), base_weight_before, atol=1e-6)

    with nn.no_grad():
        out_unmerged = lora(x).numpy()

    lora.merge_weights()
    assert lora.merged
    with nn.no_grad():
        out_merged = lora(x).numpy()
    assert np.allclose(out_merged, out_unmerged, atol=1e-4)

    lora.unmerge_weights()
    assert not lora.merged
    assert np.allclose(lora._rust.base_weight.numpy(), base_weight_before, atol=1e-5)
    with nn.no_grad():
        assert np.allclose(lora(x).numpy(), out_unmerged, atol=1e-5)

    # Adapter-only .dit persistence (Tag 102) via LoRAAdapterState & ochreml.save/load
    state = nn.LoRAAdapterState()
    state.insert("proj_q", lora)
    assert state.names() == ["proj_q"]

    adapter_path = str(tmp_path / "lora_adapter.dit")
    ochreml.save(state, adapter_path)
    loaded_state = ochreml.load(adapter_path, device="cpu")
    assert isinstance(loaded_state, nn.LoRAAdapterState)

    fresh_lora = nn.LoRALinear.from_linear(base_lin, rank=2, alpha=4.0, seed=99)
    loaded_state.apply_to("proj_q", fresh_lora)
    with nn.no_grad():
        assert np.allclose(fresh_lora(x).numpy(), out_unmerged, atol=1e-5)

    # Direct save_adapter / load_adapter helper on LoRALinear
    single_path = str(tmp_path / "single_adapter.dit")
    lora.save_adapter(single_path, name="task_a")
    fresh_lora_2 = nn.LoRALinear.from_linear(base_lin, rank=2, alpha=4.0, seed=101)
    fresh_lora_2.load_adapter(single_path, name="task_a")
    with nn.no_grad():
        assert np.allclose(fresh_lora_2(x).numpy(), out_unmerged, atol=1e-5)
