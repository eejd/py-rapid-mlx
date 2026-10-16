# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rapid-MLX contributors
# Architecture/checkpoint: NVIDIA LiLiCorr, OpenMDW-1.1;
# see rapid_mlx/models/LICENSE-LILICORR. Weights are downloaded separately.
"""Offline hidden-conditioned block drafter with native ModelOpt NVFP4 weights.

No target cache mutation or speculative serving integration is provided here.
Context contains confirmed target rows only; query blocks are never cached.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class DraftContext:
    kv: list
    length: int
    anchor: mx.array


def load_nvfp4_safetensors(path):
    """Preserve E2M1 nibbles and E4M3 scale bytes; never requantize.

    MLX expects eight packed nibbles per little-endian uint32, while the
    checkpoint stores two per uint8. Scales remain raw uint8 E4M3 encodings.
    """
    tensors = {}
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            if dtype not in {"BF16", "F32", "U8", "F8_E4M3"}:
                raise ValueError(f"Unsupported checkpoint dtype {dtype}")
            start, end = info["data_offsets"]
            stream.seek(8 + size + start)
            data = stream.read(end - start)
            if len(data) != end - start:
                raise ValueError(f"Truncated tensor {name}")
            np_dtype = {"BF16": "<u2", "F32": "<f4"}.get(dtype, "u1")
            array = mx.array(np.frombuffer(data, dtype=np_dtype).copy()).reshape(
                info["shape"]
            )
            if dtype == "BF16":
                array = array.view(mx.bfloat16)
            if dtype == "U8" and name.endswith(".weight"):
                if array.shape[-1] % 4:
                    raise ValueError(f"NVFP4 packed row is not uint32 aligned: {name}")
                array = array.view(mx.uint32)
            tensors[name] = array
    for name, array in tensors.items():
        if array.dtype == mx.uint32 and name.endswith(".weight"):
            prefix = name[:-7]
            if prefix + ".weight_scale_2" not in tensors:
                raise ValueError(f"Missing global NVFP4 scale: {prefix}")
            scale = tensors.get(prefix + ".weight_scale")
            expected = (array.shape[0], array.shape[1] // 2)
            if scale is None or scale.dtype != mx.uint8 or scale.shape != expected:
                raise ValueError(f"Invalid NVFP4 block scales: {prefix}")
    return tensors


class LiLiCorr(nn.Module):
    """Batch-one offline module; embedding and head are shared with the target.

    Tensor names mirror the export to keep packed-weight provenance reviewable.
    The constructor also accepts floating tensors for small reference tests.
    """

    def __init__(self, config, tensors):
        super().__init__()
        self.config = config
        self.tensors = tensors
        self.draft = config["dflash_config"]
        self.width = config["hidden_size"]
        self.block_size = config["block_size"]
        self.eps = config["rms_norm_eps"]
        self.rope = nn.RoPE(
            config["head_dim"], traditional=False, base=config["rope_theta"]
        )
        if self.draft.get("causal", False):
            raise ValueError("Only noncausal block attention is supported")

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        tensors = load_nvfp4_safetensors(path / "model.safetensors")
        validate_checkpoint(config, tensors)
        return cls(config, tensors)

    def linear(self, x, name):
        weights = self.tensors
        if name + ".weight_scale" in weights:
            y = mx.quantized_matmul(
                x,
                weights[name + ".weight"],
                weights[name + ".weight_scale"],
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            y = y * weights[name + ".weight_scale_2"].astype(y.dtype)
        else:
            y = x @ weights[name + ".weight"].T
        if name + ".bias" in weights:
            y = y + weights[name + ".bias"]
        return y

    def norm(self, x, name):
        return mx.fast.rms_norm(x, self.tensors[name + ".weight"], self.eps)

    def _heads(self, x, heads):
        return x.reshape(1, -1, heads, self.config["head_dim"]).transpose(0, 2, 1, 3)

    def append_context(self, taps, context=None):
        """Append consecutive confirmed rows starting at context.length (or zero)."""
        if taps.ndim != 3 or taps.shape[0] != 1 or taps.shape[1] == 0:
            raise ValueError("Expected nonempty batch-one target taps")
        expected = self.width * len(self.draft["target_layer_ids"])
        if taps.shape[-1] != expected:
            raise ValueError(f"Expected {expected} target feature channels")
        start = 0 if context is None else context.length
        hidden = self.norm(self.linear(taps, "fc"), "hidden_norm")
        kv = []
        nh = self.config["num_key_value_heads"]
        window = self.draft["swa_window_size"]
        for i in range(self.config["num_hidden_layers"]):
            prefix = f"layers.{i}.self_attn"
            k = self._heads(self.linear(hidden, prefix + ".k_proj"), nh)
            k = self.rope(self.norm(k, prefix + ".k_norm"), offset=start)
            v = self._heads(self.linear(hidden, prefix + ".v_proj"), nh)
            if context is not None:
                k = mx.concatenate([context.kv[i][0], k], axis=2)
                v = mx.concatenate([context.kv[i][1], v], axis=2)
            kv.append((k[:, :, -window:], v[:, :, -window:]))
        return DraftContext(kv, start + taps.shape[1], hidden[:, -1])

    def _conv(self, x, name, side, delta):
        length = x.shape[1]
        group = self.draft["conv_group_size"]
        taps = self.draft["conv_kernel_size"]
        groups = self.width // group
        z = x.reshape(length, groups, group).astype(mx.float32)
        base = self.tensors[name + ".base_kernel"][side].astype(mx.float32)
        coeff = (
            base.reshape(1, taps, groups, group) + delta.astype(mx.float32)[..., None]
        )
        out = coeff[:, 0] * z
        for tap in range(1, min(taps, length)):
            shifted = mx.concatenate([mx.zeros_like(z[:tap]), z[:-tap]], axis=0)
            out = out + coeff[:, tap] * shifted
        return out.reshape(x.shape).astype(x.dtype)

    def _prepare(self, x, name):
        delta = self.linear(x, name + ".kernel_projection").reshape(
            x.shape[1],
            2,
            self.draft["conv_kernel_size"],
            self.width // self.draft["conv_group_size"],
        )
        return self._conv(x, name, 0, delta[:, 0]), delta[:, 1]

    def __call__(self, query_embeddings, context):
        if query_embeddings.shape != (1, self.block_size, self.width):
            raise ValueError("Expected one complete anchor-plus-mask block")
        x = query_embeddings
        nq, nk = self.config["num_attention_heads"], self.config["num_key_value_heads"]
        dim = self.config["head_dim"]
        # Noncausal SWA is symmetric; every slot can see the full draft block.
        ctx_len = context.kv[0][0].shape[2]
        key_pos = mx.arange(context.length - ctx_len, context.length + self.block_size)
        query_pos = mx.arange(context.length, context.length + self.block_size)
        visible = (
            mx.abs(query_pos[:, None] - key_pos[None, :])
            < self.draft["swa_window_size"]
        )
        for i in range(self.config["num_hidden_layers"]):
            p = f"layers.{i}"
            z, delta = self._prepare(
                self.norm(x, p + ".input_layernorm"), p + ".attention_conv"
            )
            a = p + ".self_attn"
            q = self.rope(
                self.norm(
                    self._heads(self.linear(z, a + ".q_proj"), nq), a + ".q_norm"
                ),
                offset=context.length,
            )
            k = self.rope(
                self.norm(
                    self._heads(self.linear(z, a + ".k_proj"), nk), a + ".k_norm"
                ),
                offset=context.length,
            )
            v = self._heads(self.linear(z, a + ".v_proj"), nk)
            k = mx.concatenate(
                [context.kv[i][0], k, mx.zeros_like(k[:, :, :1])], axis=2
            )
            v = mx.concatenate(
                [context.kv[i][1], v, mx.zeros_like(v[:, :, :1])], axis=2
            )
            mask = mx.where(visible, 0.0, -float("inf"))
            mask = mx.broadcast_to(mask, (nq, self.block_size, key_pos.size))
            sink = self.tensors[a + ".attention_sink_bias"][:, None, None]
            mask = mx.concatenate(
                [mask, mx.broadcast_to(sink, (nq, self.block_size, 1))], axis=-1
            ).astype(q.dtype)
            z = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=dim**-0.5, mask=mask
            )
            z = z.transpose(0, 2, 1, 3).reshape(x.shape)
            x = x + self._conv(
                self.linear(z, a + ".o_proj"), p + ".attention_conv", 1, delta
            )
            z, delta = self._prepare(
                self.norm(x, p + ".post_attention_layernorm"), p + ".mlp_conv"
            )
            z = self.linear(
                nn.silu(self.linear(z, p + ".mlp.gate_proj"))
                * self.linear(z, p + ".mlp.up_proj"),
                p + ".mlp.down_proj",
            )
            x = x + self._conv(z, p + ".mlp_conv", 1, delta)
        return self.norm(x, "norm")

    def candidates(self, logits):
        z = logits.astype(mx.float32)
        topk = self.draft["lilicorr_candidate_topk"]
        ids = mx.argpartition(-z, kth=topk - 1, axis=-1)[..., :topk]
        values = mx.take_along_axis(z, ids, axis=-1)
        order = mx.argsort(-values, axis=-1)
        ids = mx.take_along_axis(ids, order, axis=-1)
        values = mx.take_along_axis(values, order, axis=-1)
        return ids, values - mx.logsumexp(z, axis=-1, keepdims=True)

    def score(self, embeddings, log_probs, pass_hidden, anchor):
        w, d = self.tensors, self.draft
        _, slots, topk = log_probs.shape
        h = d["lilicorr_hidden_size"]
        lp = log_probs.astype(mx.float32)
        rank = mx.broadcast_to(mx.arange(topk) / max(topk - 1, 1), lp.shape)
        features = mx.stack(
            [
                lp,
                mx.exp(lp),
                lp - mx.max(lp, axis=-1, keepdims=True),
                rank,
                (rank == 0).astype(lp.dtype),
            ],
            axis=-1,
        ).astype(embeddings.dtype)
        features = mx.fast.layer_norm(
            features,
            w["lilicorr.feature_norm.weight"],
            w["lilicorr.feature_norm.bias"],
            1e-5,
        )
        features = self.linear(
            nn.silu(self.linear(features, "lilicorr.feature_mlp.up_proj")),
            "lilicorr.feature_mlp.down_proj",
        )
        z = (
            self.linear(embeddings, "lilicorr.token_proj")
            + self.linear(pass_hidden, "lilicorr.pass_hidden_proj")[:, :, None, :]
        )
        z = z + features
        z = z + w["lilicorr.slot_embedding"][:, 0, :slots]
        z = z + w["lilicorr.rank_embedding"][:, 0]
        z = z.reshape(1, slots * topk, h)
        sid = mx.repeat(mx.arange(slots), topk)
        rel = sid[:, None] - sid[None, :]
        bias = w["lilicorr.relative_slot_bias"][:, rel + self.block_size - 1]
        bias = bias + w["lilicorr.same_slot_bias"][:, None, None] * (rel == 0)
        heads = d["lilicorr_num_heads"]
        for i in range(d["lilicorr_num_layers"]):
            p = f"lilicorr.layers.{i}"
            a = p + ".attn"
            qkv = (
                self.norm(z, p + ".attn_norm") @ w[a + ".in_proj_weight"].T
                + w[a + ".in_proj_bias"]
            )
            q, k, v = [
                t.reshape(1, slots * topk, heads, h // heads).transpose(0, 2, 1, 3)
                for t in mx.split(qkv, 3, axis=-1)
            ]
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=(h // heads) ** -0.5, mask=bias
            )
            z = z + self.linear(
                y.transpose(0, 2, 1, 3).reshape(z.shape), a + ".out_proj"
            )
            z = z + self.linear(
                nn.silu(self.linear(self.norm(z, p + ".mlp_norm"), p + ".mlp.up_proj")),
                p + ".mlp.down_proj",
            )
        z = self.norm(z, "lilicorr.output_norm").reshape(1, slots, topk, h)
        anchor = self.norm(
            self.linear(anchor, "lilicorr.context_proj"), "lilicorr.anchor_norm"
        )
        a = mx.broadcast_to(anchor[:, None, None, :], z.shape)
        factor = nn.silu(
            self.linear(
                mx.concatenate([z, a, z * a], axis=-1), "lilicorr.factor_input_proj"
            )
        )

        def unit(x):
            norm = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
            return x / mx.maximum(norm, d["lilicorr_vector_eps"]).astype(x.dtype)

        outgoing = unit(self.linear(factor, "lilicorr.out_head"))
        incoming = unit(self.linear(factor, "lilicorr.in_head"))
        anchor_out = unit(self.linear(anchor, "lilicorr.anchor_out_head"))
        start = mx.sum(anchor_out[:, None] * incoming[:, 0], axis=-1)
        pairs = outgoing[:, :-1] @ mx.swapaxes(incoming[:, 1:], -1, -2)
        return start.astype(mx.float32) * d["lilicorr_logit_scale"], pairs.astype(
            mx.float32
        ) * d["lilicorr_logit_scale"]

    @staticmethod
    def greedy_walk(ids, start, pairs):
        previous = mx.argmax(start, axis=-1)
        selected = [mx.take_along_axis(ids[:, 0], previous[:, None], axis=-1)]
        for slot in range(1, ids.shape[1]):
            scores = mx.take_along_axis(
                pairs[:, slot - 1], previous[:, None, None], axis=1
            )[:, 0]
            previous = mx.argmax(scores, axis=-1)
            selected.append(
                mx.take_along_axis(ids[:, slot], previous[:, None], axis=-1)
            )
        return mx.concatenate(selected, axis=-1)

    def propose(self, anchor_token, context, embedding, head):
        query = mx.array(
            [[anchor_token] + [self.draft["mask_token_id"]] * (self.block_size - 1)]
        )
        hidden = self(embedding(query), context)[:, 1:]
        ids, lp = self.candidates(head(hidden))
        start, pairs = self.score(embedding(ids), lp, hidden, context.anchor)
        return self.greedy_walk(ids, start, pairs)


def validate_checkpoint(config, weights):
    """Fail closed on missing, extra or incorrectly shaped exported parameters."""
    d = config["dflash_config"]
    if config.get("attention_bias", False) or d.get("causal", False):
        raise ValueError("Unsupported biased/causal backbone")
    width, h = config["hidden_size"], d["lilicorr_hidden_size"]
    shapes = {}

    def linear(name, out, inp, bias=False):
        shapes[name + ".weight"] = (out, inp)
        if bias:
            shapes[name + ".bias"] = (out,)

    def norm(name, size):
        shapes[name + ".weight"] = (size,)

    linear("fc", width, width * len(d["target_layer_ids"]))
    norm("hidden_norm", width)
    norm("norm", width)
    dim = config["head_dim"]
    for i in range(config["num_hidden_layers"]):
        p = f"layers.{i}"
        norm(p + ".input_layernorm", width)
        norm(p + ".post_attention_layernorm", width)
        for name in ["attention_conv", "mlp_conv"]:
            c = p + "." + name
            shapes[c + ".base_kernel"] = (2, d["conv_kernel_size"], width)
            linear(
                c + ".kernel_projection",
                2 * d["conv_kernel_size"] * width // d["conv_group_size"],
                width,
            )
        a = p + ".self_attn"
        for name, heads in [
            ("q_proj", config["num_attention_heads"]),
            ("k_proj", config["num_key_value_heads"]),
            ("v_proj", config["num_key_value_heads"]),
        ]:
            linear(a + "." + name, heads * dim, width)
        linear(a + ".o_proj", width, config["num_attention_heads"] * dim)
        norm(a + ".q_norm", dim)
        norm(a + ".k_norm", dim)
        shapes[a + ".attention_sink_bias"] = (config["num_attention_heads"],)
        for name in ["gate_proj", "up_proj"]:
            linear(p + ".mlp." + name, config["intermediate_size"], width)
        linear(p + ".mlp.down_proj", width, config["intermediate_size"])
    for name in ["token_proj", "pass_hidden_proj", "context_proj"]:
        linear("lilicorr." + name, h, width, True)
    norm("lilicorr.feature_norm", 5)
    shapes["lilicorr.feature_norm.bias"] = (5,)
    linear("lilicorr.feature_mlp.up_proj", h, 5, True)
    linear("lilicorr.feature_mlp.down_proj", h, h, True)
    shapes.update(
        {
            "lilicorr.slot_embedding": (1, 1, config["block_size"] - 1, 1, h),
            "lilicorr.rank_embedding": (1, 1, 1, d["lilicorr_candidate_topk"], h),
            "lilicorr.relative_slot_bias": (
                d["lilicorr_num_heads"],
                2 * config["block_size"] - 1,
            ),
            "lilicorr.same_slot_bias": (d["lilicorr_num_heads"],),
        }
    )
    for i in range(d["lilicorr_num_layers"]):
        p = f"lilicorr.layers.{i}"
        norm(p + ".attn_norm", h)
        norm(p + ".mlp_norm", h)
        shapes[p + ".attn.in_proj_weight"] = (3 * h, h)
        shapes[p + ".attn.in_proj_bias"] = (3 * h,)
        linear(p + ".attn.out_proj", h, h, True)
        inner = int(h * d["lilicorr_mlp_ratio"])
        linear(p + ".mlp.up_proj", inner, h, True)
        linear(p + ".mlp.down_proj", h, inner, True)
    norm("lilicorr.output_norm", h)
    norm("lilicorr.anchor_norm", h)
    linear("lilicorr.factor_input_proj", h, 3 * h, True)
    for name in ["in_head", "out_head", "anchor_out_head"]:
        linear("lilicorr." + name, d["lilicorr_factor_dim"], h, True)
    for name, shape in list(shapes.items()):
        weight = weights.get(name)
        if weight is not None and weight.dtype == mx.uint32:
            out, inp = shape
            prefix = name[:-7]
            shapes[name] = (out, inp // 8)
            shapes[prefix + ".weight_scale"] = (out, inp // 16)
            shapes[prefix + ".weight_scale_2"] = ()
    if weights.keys() != shapes.keys():
        raise ValueError(
            f"Checkpoint coverage: missing={sorted(shapes.keys() - weights.keys())}, extra={sorted(weights.keys() - shapes.keys())}"
        )
    for name, shape in shapes.items():
        if weights[name].shape != shape:
            raise ValueError(
                f"Checkpoint shape {name}: expected {shape}, got {weights[name].shape}"
            )
