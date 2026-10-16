# SPDX-License-Identifier: Apache-2.0
"""Tiny architecture, packed format, context and path selection contracts."""

import json
import struct

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

from rapid_mlx.models.lilicorr import (  # noqa: E402
    LiLiCorr,
    load_nvfp4_safetensors,
    validate_checkpoint,
)

pytestmark = pytest.mark.requires_mlx


@pytest.fixture
def full_precision_reference_stream():
    # Keep strict FP32 reference comparisons independent of M5 TF32 defaults.
    # The separate forward and NVFP4 tests exercise the default GPU stream.
    with mx.stream(mx.cpu):
        yield


def tiny_draft():
    config = {
        "hidden_size": 32,
        "block_size": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "dflash_config": {
            "target_layer_ids": [0, 2],
            "swa_window_size": 8,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "lilicorr_hidden_size": 16,
            "lilicorr_candidate_topk": 2,
            "lilicorr_num_heads": 2,
            "lilicorr_num_layers": 2,
            "lilicorr_mlp_ratio": 2,
            "lilicorr_factor_dim": 16,
            "lilicorr_vector_eps": 1e-4,
            "lilicorr_logit_scale": 8.0,
            "mask_token_id": 3,
        },
    }
    w = {}

    def linear(p, out, inp, bias=False):
        w[p + ".weight"] = mx.random.normal((out, inp)) * 0.04
        if bias:
            w[p + ".bias"] = mx.random.normal((out,)) * 0.01

    def norm(p, size):
        w[p + ".weight"] = mx.ones((size,))

    linear("fc", 32, 64)
    norm("hidden_norm", 32)
    norm("norm", 32)
    for i in range(2):
        p = f"layers.{i}"
        for n in ["input_layernorm", "post_attention_layernorm"]:
            norm(p + "." + n, 32)
        for n in ["attention_conv", "mlp_conv"]:
            w[p + "." + n + ".base_kernel"] = mx.random.normal((2, 2, 32)) * 0.1
            linear(p + "." + n + ".kernel_projection", 16, 32)
        for n, size in [("q_proj", 32), ("k_proj", 16), ("v_proj", 16), ("o_proj", 32)]:
            linear(p + ".self_attn." + n, size, 32)
        for n in ["q_norm", "k_norm"]:
            norm(p + ".self_attn." + n, 8)
        w[p + ".self_attn.attention_sink_bias"] = mx.random.normal((4,))
        for n, out, inp in [
            ("up_proj", 64, 32),
            ("gate_proj", 64, 32),
            ("down_proj", 32, 64),
        ]:
            linear(p + ".mlp." + n, out, inp)
    for n in ["token_proj", "pass_hidden_proj", "context_proj"]:
        linear("lilicorr." + n, 16, 32, True)
    norm("lilicorr.feature_norm", 5)
    w["lilicorr.feature_norm.bias"] = mx.zeros((5,))
    linear("lilicorr.feature_mlp.up_proj", 16, 5, True)
    linear("lilicorr.feature_mlp.down_proj", 16, 16, True)
    for n, shape in {
        "slot_embedding": (1, 1, 3, 1, 16),
        "rank_embedding": (1, 1, 1, 2, 16),
        "relative_slot_bias": (2, 7),
        "same_slot_bias": (2,),
    }.items():
        w["lilicorr." + n] = mx.random.normal(shape) * 0.02
    for i in range(2):
        p = f"lilicorr.layers.{i}"
        norm(p + ".attn_norm", 16)
        norm(p + ".mlp_norm", 16)
        w[p + ".attn.in_proj_weight"] = mx.random.normal((48, 16)) * 0.04
        w[p + ".attn.in_proj_bias"] = mx.zeros((48,))
        linear(p + ".attn.out_proj", 16, 16, True)
        linear(p + ".mlp.up_proj", 32, 16, True)
        linear(p + ".mlp.down_proj", 16, 32, True)
    norm("lilicorr.output_norm", 16)
    norm("lilicorr.anchor_norm", 16)
    linear("lilicorr.factor_input_proj", 16, 48, True)
    for n in ["in_head", "out_head", "anchor_out_head"]:
        linear("lilicorr." + n, 16, 16, True)
    return LiLiCorr(config, w)


def test_tiny_forward_and_context_incremental_parity():
    mx.random.seed(7)
    draft = tiny_draft()
    taps = mx.random.normal((1, 11, 64))
    full = draft.append_context(taps)
    partial = draft.append_context(taps[:, :7])
    split = draft.append_context(taps[:, 7:], partial)
    assert full.length == split.length == 11
    for (ka, va), (kb, vb) in zip(full.kv, split.kv, strict=True):
        assert ka.shape == (1, 2, 8, 8)
        assert bool(mx.allclose(ka, kb, atol=1e-5))
        assert bool(mx.allclose(va, vb, atol=1e-5))
    embed, head = nn.Embedding(64, 32), nn.Linear(32, 64, bias=False)
    before = [k for k, _ in full.kv]
    proposal = draft.propose(5, full, embed, head)
    assert proposal.shape == (1, 3)
    assert bool(mx.all((proposal >= 0) & (proposal < 64)))
    assert bool(mx.all(mx.isfinite(draft(embed(mx.array([[5, 3, 3, 3]])), full))))
    assert all(k is old for (k, _), old in zip(full.kv, before, strict=True))
    with pytest.raises(ValueError, match="complete"):
        draft(mx.zeros((1, 3, 32)), full)
    with pytest.raises(ValueError, match="channels"):
        draft.append_context(mx.zeros((1, 1, 32)))


def write_tensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    metadata = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + b"".join(blobs))


def test_native_nvfp4_adapter_independent_dequant_reference(tmp_path):
    # All E2M1 values/signs and multiple positive E4M3 block scales.
    raw = np.arange(256, dtype=np.uint8).reshape(8, 32)
    scales = np.resize(np.array([0x30, 0x38, 0x40, 0x44], np.uint8), (8, 4))
    global_scale = np.array(0.75, np.float32)
    file = tmp_path / "model.safetensors"
    tensors = {
        "proj.weight": ("U8", raw),
        "proj.weight_scale": ("F8_E4M3", scales),
        "proj.weight_scale_2": ("F32", global_scale),
    }
    write_tensors(file, tensors)
    w = load_nvfp4_safetensors(file)
    assert w["proj.weight"].dtype == mx.uint32
    np.testing.assert_array_equal(np.array(w["proj.weight"].view(mx.uint8)), raw)
    np.testing.assert_array_equal(np.array(w["proj.weight_scale"]), scales)
    table = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    nibbles = np.stack([raw & 15, raw >> 4], axis=-1).reshape(8, 64)
    decoded = table[nibbles & 7] * np.where(nibbles & 8, -1, 1)
    # Positive normal E4M3 scale: mantissa 3 bits, exponent bias 7.
    scale_ref = (1 + (scales & 7) / 8) * 2.0 ** ((scales >> 3).astype(int) - 7)
    decoded = decoded * np.repeat(scale_ref, 16, axis=-1) * global_scale
    x = mx.arange(128).reshape(2, 64).astype(mx.float32) / 128
    draft = LiLiCorr(
        {
            "hidden_size": 32,
            "block_size": 4,
            "rms_norm_eps": 1e-6,
            "head_dim": 8,
            "rope_theta": 10000,
            "dflash_config": {},
        },
        w,
    )
    actual = draft.linear(x, "proj")
    expected = np.array(x) @ decoded.T
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    del tensors["proj.weight_scale_2"]
    write_tensors(file, tensors)
    with pytest.raises(ValueError, match="global"):
        load_nvfp4_safetensors(file)


def test_walk_is_greedy_conditional_not_independent_or_global():
    ids = mx.array([[[10, 11], [20, 21], [30, 31]]])
    start = mx.array([[1.0, 2.0]])
    pairs = mx.array([[[[100.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]]])
    assert LiLiCorr.greedy_walk(ids, start, pairs).tolist() == [[11, 21, 31]]


def test_target_taps_are_block_outputs_in_requested_order():
    from rapid_mlx.models.nemotron_h_puzzle import Model, ModelArgs
    from tests.test_nemotron_h_puzzle_vendored import TestNemotronBolt

    target = Model(ModelArgs.from_dict(TestNemotronBolt.config()))
    ids = mx.array([[1, 2, 3]])
    ordinary = target.backbone(ids)
    hidden, taps = target.backbone(ids, target_layer_ids=[3, 0])
    x = target.backbone.embeddings(ids)
    first = None
    for i, layer in enumerate(target.backbone.layers):
        # Causal mask required for the attention block.
        mask = (
            mx.triu(mx.full((3, 3), -float("inf")), k=1)
            if layer.block_type == "*"
            else None
        )
        x = layer(x, mask=mask)
        if i == 0:
            first = x
    assert bool(mx.allclose(ordinary, hidden))
    assert bool(mx.allclose(taps[..., :32], x))
    assert bool(mx.allclose(taps[..., 32:], first))
    for invalid in [[], [0, 0], [-1], [4]]:
        with pytest.raises(ValueError, match="taps"):
            target.backbone(ids, target_layer_ids=invalid)


def test_backbone_against_independent_dense_numpy_attention(
    full_precision_reference_stream,
):
    """Check context normalization, tap time, RoPE, sinks and grouped conv math."""
    mx.random.seed(19)
    draft = tiny_draft()
    w = {k: np.array(v) for k, v in draft.tensors.items()}
    raw = np.random.default_rng(4).normal(size=(1, 11, 64)).astype(np.float32)
    query = np.random.default_rng(5).normal(size=(1, 4, 32)).astype(np.float32)

    def linear(x, p):
        return x @ w[p + ".weight"].T + w.get(p + ".bias", 0)

    def rms(x, p):
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            * w[p + ".weight"]
        )

    def rotate(x, start):
        n = x.shape[2]
        phase = np.arange(start, start + n)[:, None] * 10000.0 ** (
            -np.arange(0, 8, 2) / 8
        )
        a, b = np.split(x, 2, axis=-1)
        return np.concatenate(
            [
                a * np.cos(phase) - b * np.sin(phase),
                a * np.sin(phase) + b * np.cos(phase),
            ],
            axis=-1,
        )

    def heads(x, n):
        return x.reshape(1, -1, n, 8).transpose(0, 2, 1, 3)

    def convolve(x, p, side, delta):
        y = np.zeros_like(x)
        for row in range(x.shape[1]):
            for channel in range(32):
                for tap in range(min(row + 1, 2)):
                    coeff = (
                        w[p + ".base_kernel"][side, tap, channel]
                        + delta[row, tap, channel // 8]
                    )
                    y[0, row, channel] += coeff * x[0, row - tap, channel]
        return y

    def prepare(x, p):
        delta = linear(x, p + ".kernel_projection").reshape(4, 2, 2, 4)
        return convolve(x, p, 0, delta[:, 0]), delta[:, 1]

    context = rms(linear(raw, "fc"), "hidden_norm")
    x = query.copy()
    for i in range(2):
        p = f"layers.{i}"
        a = p + ".self_attn"
        z, delta = prepare(rms(x, p + ".input_layernorm"), p + ".attention_conv")
        q = rotate(rms(heads(linear(z, a + ".q_proj"), 4), a + ".q_norm"), 11)
        kq = rotate(rms(heads(linear(z, a + ".k_proj"), 2), a + ".k_norm"), 11)
        kc = rotate(rms(heads(linear(context, a + ".k_proj"), 2), a + ".k_norm"), 0)[
            :, :, -8:
        ]
        vc = heads(linear(context, a + ".v_proj"), 2)[:, :, -8:]
        vq = heads(linear(z, a + ".v_proj"), 2)
        keys = np.repeat(np.concatenate([kc, kq], axis=2), 2, axis=1)
        values = np.repeat(np.concatenate([vc, vq], axis=2), 2, axis=1)
        scores = (q @ keys.transpose(0, 1, 3, 2)) / np.sqrt(8)
        visible = np.abs(np.arange(11, 15)[:, None] - np.arange(3, 15)[None, :]) < 8
        scores = np.where(visible, scores, -np.inf)
        sinks = np.broadcast_to(
            w[a + ".attention_sink_bias"][None, :, None, None], (1, 4, 4, 1)
        )
        scores = np.concatenate([scores, sinks], axis=-1)
        prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        y = prob[..., :-1] @ values
        y = y.transpose(0, 2, 1, 3).reshape(1, 4, 32)
        x += convolve(linear(y, a + ".o_proj"), p + ".attention_conv", 1, delta)
        z, delta = prepare(rms(x, p + ".post_attention_layernorm"), p + ".mlp_conv")
        gate = linear(z, p + ".mlp.gate_proj")
        y = linear(
            gate / (1 + np.exp(-gate)) * linear(z, p + ".mlp.up_proj"),
            p + ".mlp.down_proj",
        )
        x += convolve(y, p + ".mlp_conv", 1, delta)
    expected = rms(x, "norm")
    actual = draft(mx.array(query), draft.append_context(mx.array(raw)))
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)


def test_correlator_against_independent_dense_numpy(full_precision_reference_stream):
    mx.random.seed(21)
    draft = tiny_draft()
    w = {k: np.array(v) for k, v in draft.tensors.items()}
    rng = np.random.default_rng(22)
    embedding = rng.normal(size=(1, 3, 2, 32)).astype(np.float32)
    hidden = rng.normal(size=(1, 3, 32)).astype(np.float32)
    anchor = rng.normal(size=(1, 32)).astype(np.float32)
    lp = -rng.uniform(0, 5, size=(1, 3, 2)).astype(np.float32)

    def linear(x, p):
        return x @ w[p + ".weight"].T + w.get(p + ".bias", 0)

    def rms(x, p):
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            * w[p + ".weight"]
        )

    def silu(x):
        return x / (1 + np.exp(-x))

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-4)

    rank = np.broadcast_to(np.arange(2), lp.shape)
    f = np.stack(
        [lp, np.exp(lp), lp - lp.max(axis=-1, keepdims=True), rank, rank == 0], axis=-1
    )
    f = (f - f.mean(axis=-1, keepdims=True)) / np.sqrt(
        f.var(axis=-1, keepdims=True) + 1e-5
    )
    f = f * w["lilicorr.feature_norm.weight"] + w["lilicorr.feature_norm.bias"]
    f = linear(
        silu(linear(f, "lilicorr.feature_mlp.up_proj")),
        "lilicorr.feature_mlp.down_proj",
    )
    z = (
        linear(embedding, "lilicorr.token_proj")
        + linear(hidden, "lilicorr.pass_hidden_proj")[:, :, None]
    )
    z += f
    z += w["lilicorr.slot_embedding"][:, 0]
    z += w["lilicorr.rank_embedding"][:, 0]
    z = z.reshape(1, 6, 16)
    sid = np.repeat(np.arange(3), 2)
    rel = sid[:, None] - sid[None, :]
    bias = w["lilicorr.relative_slot_bias"][:, rel + 3] + w["lilicorr.same_slot_bias"][
        :, None, None
    ] * (rel == 0)
    for i in range(2):
        p = f"lilicorr.layers.{i}"
        a = p + ".attn"
        qkv = (
            rms(z, p + ".attn_norm") @ w[a + ".in_proj_weight"].T
            + w[a + ".in_proj_bias"]
        )
        q, k, v = [
            t.reshape(1, 6, 2, 8).transpose(0, 2, 1, 3)
            for t in np.split(qkv, 3, axis=-1)
        ]
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(8) + bias
        prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        y = (prob @ v).transpose(0, 2, 1, 3).reshape(1, 6, 16)
        z += linear(y, a + ".out_proj")
        z += linear(
            silu(linear(rms(z, p + ".mlp_norm"), p + ".mlp.up_proj")),
            p + ".mlp.down_proj",
        )
    z = rms(z, "lilicorr.output_norm").reshape(1, 3, 2, 16)
    a = rms(linear(anchor, "lilicorr.context_proj"), "lilicorr.anchor_norm")
    broadcast = np.broadcast_to(a[:, None, None], z.shape)
    factors = silu(
        linear(
            np.concatenate([z, broadcast, z * broadcast], axis=-1),
            "lilicorr.factor_input_proj",
        )
    )
    incoming = unit(linear(factors, "lilicorr.in_head"))
    outgoing = unit(linear(factors, "lilicorr.out_head"))
    start = (unit(linear(a, "lilicorr.anchor_out_head"))[:, None] * incoming[:, 0]).sum(
        axis=-1
    ) * 8
    pairs = (outgoing[:, :-1] @ incoming[:, 1:].swapaxes(-1, -2)) * 8
    actual_start, actual_pairs = draft.score(
        mx.array(embedding), mx.array(lp), mx.array(hidden), mx.array(anchor)
    )
    np.testing.assert_allclose(np.array(actual_start), start, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(actual_pairs), pairs, rtol=2e-5, atol=2e-5)


def test_checkpoint_coverage_and_shapes_fail_closed():
    draft = tiny_draft()
    validate_checkpoint(draft.config, draft.tensors)
    broken = dict(draft.tensors)
    del broken["hidden_norm.weight"]
    with pytest.raises(ValueError, match="coverage"):
        validate_checkpoint(draft.config, broken)
    broken = dict(draft.tensors)
    broken["fc.weight"] = mx.zeros((32, 32))
    with pytest.raises(ValueError, match="shape"):
        validate_checkpoint(draft.config, broken)
