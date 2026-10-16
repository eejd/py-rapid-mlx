# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Nemotron Super native head adapter for the shared MTP generator."""

import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from rapid_mlx.models import nemotron_h_super as super_h


class NemotronMambaCache(ArraysCache):
    """The existing generator's per-boundary conv/SSM rollback contract."""

    rollback_state = None
    n_confirmed_for_mtp = 0

    def extract(self, idx):
        # Match the base extraction of a finalized request, retaining the
        # concrete recurrent type for prefix reuse and scheduler handoff.
        cache = type(self)(2)
        cache.cache = [None if x is None else x[idx : idx + 1] for x in self.cache]
        return cache

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            return super().merge(caches)
        # The base empty-cache merge adds a left-padding vector even for a
        # singleton. It survives ordinary prefill and is unnecessary here.
        cache = cls(2)
        cache.cache = list(caches[0].cache)
        cache.left_padding = caches[0].left_padding
        cache.lengths = caches[0].lengths
        return cache

    def restore_rollback(self, n_to_drop, verify_size):
        keep = verify_size - n_to_drop
        if not self.rollback_state or not 1 <= keep <= len(self.rollback_state):
            raise ValueError("Invalid Mamba2 accepted-prefix boundary")
        self[0], self[1] = self.rollback_state[keep - 1]
        self.rollback_state = None


class MambaWithHistory(super_h.shared.NemotronHMamba2Mixer):
    """Batched projections/convolution, short sequential recurrence history."""

    def _conv(self, conv_input, cache, mask):
        if cache is not None and getattr(cache, "n_confirmed_for_mtp", 0) > 0:
            if cache.lengths is not None or cache.left_padding is not None:
                raise ValueError("Mamba2 MTP verification requires an unpadded lane")
            if mask is not None:
                conv_input = mx.where(mask[..., None], conv_input, 0)
            initial = cache[0]
            if initial is None:
                initial = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            padded = mx.concatenate([initial, conv_input], axis=1)
            cache.rollback_state = [
                (padded[:, p : p + self.conv_kernel_size - 1], None)
                for p in range(1, conv_input.shape[1])
            ]
        return super()._conv(conv_input, cache, mask)

    def _ssm(self, hidden_states, B, C, dt, cache, mask):
        if cache is None or getattr(cache, "n_confirmed_for_mtp", 0) <= 0:
            return super()._ssm(hidden_states, B, C, dt, cache, mask)
        batch, length, _ = hidden_states.shape
        dtype = hidden_states.dtype
        x = hidden_states.astype(mx.float32).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        B = B.astype(mx.float32).reshape(
            batch, length, self.n_groups, self.ssm_state_size
        )
        C = C.astype(mx.float32).reshape(
            batch, length, self.n_groups, self.ssm_state_size
        )
        state = cache[1]
        outputs = []
        for p in range(length):
            y, state = super_h.shared.ssm_update(
                x[:, p : p + 1],
                self.A_log.astype(mx.float32),
                B[:, p : p + 1],
                C[:, p : p + 1],
                self.D.astype(mx.float32),
                dt[:, p : p + 1],
                self.dt_bias.astype(dt.dtype),
                state,
                self.time_step_limit,
                mask=None if mask is None else mask[:, p : p + 1],
            )
            outputs.append(y)
            if p < length - 1:
                conv, _ = cache.rollback_state[p]
                cache.rollback_state[p] = (conv, state)
        cache[1] = state
        return mx.concatenate(outputs, axis=1).astype(dtype).reshape(batch, length, -1)


class MTPHead(nn.Module):
    """One shared logical MTP module: embedding fusion, attention, MoE."""

    def __init__(self, args):
        super().__init__()
        self.layers = [super_h.shared.NemotronHBlock(args, k) for k in ("*", "E")]
        for layer in self.layers:
            layer.norm = super_h.RMSNorm(args.hidden_size, args.layer_norm_epsilon)
        first, last = self.layers
        first.enorm = super_h.RMSNorm(args.hidden_size, args.layer_norm_epsilon)
        first.hnorm = super_h.RMSNorm(args.hidden_size, args.layer_norm_epsilon)
        first.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        last.final_layernorm = super_h.RMSNorm(
            args.hidden_size, args.layer_norm_epsilon
        )

    def __call__(self, hidden, tokens, embeddings, cache):
        first, last = self.layers
        fused = mx.concatenate(
            [first.enorm(embeddings(tokens)), first.hnorm(hidden)], -1
        )
        x = first.eh_proj(fused)
        x = first(x, mask=create_attention_mask(x, cache[0]), cache=cache[0])
        return last.final_layernorm(last(x))


def quantization_predicate(path, module):
    """Four-bit experts/projections; small sensitive tensors retain BF16."""
    return hasattr(module, "to_quantized") and not any(
        k in path.split(".")
        for k in ("gate", "eh_proj", "fc1_latent_proj", "fc2_latent_proj")
    )


def sidecar_manifest(args):
    return {
        "format_version": 1,
        "model_type": "nemotron_h",
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }


def inject_mtp_support(model, mtp_sidecar=None, allow_random_init=False):
    """Load the additive head strictly before changing the target instance."""
    if not isinstance(model, super_h.Model) or mtp_sidecar is None:
        return False
    from .qwen3_5_inject import _resolve_sidecar_file

    weights_file = _resolve_sidecar_file(mtp_sidecar)
    if weights_file is None:
        return False
    manifest = json.loads((weights_file.parent / "nemotron_mtp.json").read_text())
    if manifest != sidecar_manifest(model.args):
        return False
    head = MTPHead(model.args)
    nn.quantize(head, bits=4, group_size=64, class_predicate=quantization_predicate)
    weights = mx.load(str(weights_file))
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(head.parameters()))
    for name, value in weights.items():
        if name in expected:
            dtype = expected[name].dtype
            if (dtype == mx.uint32 and value.dtype != mx.uint32) or (
                dtype != mx.uint32 and not mx.issubdtype(value.dtype, mx.floating)
            ):
                raise ValueError(f"Invalid native MTP tensor dtype: {name}")
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    original = type(model)

    class WithMTP(original):
        mtp_recursive_draft_depth = 2
        mtp_max_speculative_tokens = 2

        @property
        def mtp_target_cache_count(self):
            # Stateless MoE blocks deliberately own no cache entry.
            return sum(layer.block_type in ("M", "*") for layer in self.layers)

        def make_cache(self):
            return [
                NemotronMambaCache(2) if layer.block_type == "M" else KVCache()
                for layer in self.layers
                if layer.block_type in ("M", "*")
            ]

        def __call__(self, inputs, cache=None, return_hidden=False, n_confirmed=0):
            if not return_hidden and not n_confirmed:
                return super().__call__(inputs, cache=cache)
            backbone = self.backbone
            x = backbone.embeddings(inputs)
            if cache is None:
                cache = [None] * sum(
                    layer.block_type in ("M", "*") for layer in self.layers
                )
            fa_mask = create_attention_mask(x, cache[backbone.fa_idx])
            ssm_mask = create_ssm_mask(x, cache[backbone.ssm_idx])
            entries = iter(cache)
            for layer in self.layers:
                c = next(entries) if layer.block_type in ("M", "*") else None
                if layer.block_type == "M" and c is not None:
                    if not isinstance(c, NemotronMambaCache):
                        raise ValueError(
                            "MTP requires fresh rollback-capable Mamba caches"
                        )
                    c.n_confirmed_for_mtp = n_confirmed
                try:
                    x = layer(
                        x,
                        mask=fa_mask if layer.block_type == "*" else ssm_mask,
                        cache=c,
                    )
                finally:
                    if isinstance(c, NemotronMambaCache):
                        c.n_confirmed_for_mtp = 0
            hidden = backbone.norm_f(x)
            logits = self.lm_head(hidden)
            return (logits, hidden) if return_hidden else logits

        def mtp_forward(
            self, hidden_states, next_token_ids, mtp_cache, return_hidden=False
        ):
            hidden = self.mtp(
                hidden_states, next_token_ids, self.backbone.embeddings, mtp_cache
            )
            logits = self.lm_head(hidden)
            return (logits, hidden) if return_hidden else logits

        def make_mtp_cache(self):
            return [KVCache()]

    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.__class__ = MambaWithHistory
    model.mtp = head
    model.__class__ = WithMTP
    return True


def validate_mtp_support(model):
    return isinstance(getattr(model, "mtp", None), MTPHead)
