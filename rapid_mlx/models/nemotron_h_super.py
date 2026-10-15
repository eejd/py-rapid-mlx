# SPDX-License-Identifier: Apache-2.0
"""Precision adapter for standalone Nemotron-H text extracted from Super VL.

Artifacts retain the native ``nemotron_h`` config and tensor layout. The
``rapid_mlx_nemotron_h`` opt-in selects the reviewed local math in Rapid's
checked loader, without replacing the installed architecture globally.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import nemotron_h as base

from . import nemotron_h_puzzle as shared


def text_config(config):
    """Extract text metadata without executing checkpoint Python files."""
    config = dict(config)
    if config.get("model_type") == "nemotron_h_omni":
        config = dict(config["llm_config"])
    if config.get("model_type") != "nemotron_h":
        raise ValueError("Expected nemotron_h language config")
    for key in ("auto_map", "model_file"):
        config.pop(key, None)
    config["num_hidden_layers"] = len(config["layers_block_type"])
    # JSON's encoded infinity is not a numeric argument to mx.clip. Omitting
    # the default bound is valid JSON and uses the native 0..inf default.
    limit = config.get("time_step_limit")
    if limit == [0.0, {"__float__": "Infinity"}]:
        config.pop("time_step_limit")
    config["rapid_mlx_nemotron_h"] = True
    # No Rapid consumer implements Nemotron-H MTP; its tensors are dropped.
    config.pop("num_nextn_predict_layers", None)
    config.pop("mtp_layers_block_type", None)
    return config


class ModelArgs(base.ModelArgs):
    @classmethod
    def from_dict(cls, params):
        return super().from_dict(text_config(params))

    def for_layer(self, layer_idx):
        return self


class RMSNorm(shared.NemotronHRMSNorm):
    @mx.compile
    def __call__(self, x):
        # The checkpoint's language norm rounds the normalized activation
        # before applying its weight. Keep the BF16 boundary explicit.
        dtype = x.dtype
        value = x.astype(mx.float32)
        variance = mx.mean(mx.square(value), axis=-1, keepdims=True)
        value = (value * mx.rsqrt(variance + self.eps)).astype(dtype)
        return self.weight * value


class GatedRMSNorm(base.MambaRMSNormGated):
    @mx.compile
    def __call__(self, x, gate=None):
        dtype = x.dtype
        value = x.astype(mx.float32)
        if gate is not None:
            value = value * nn.silu(gate.astype(mx.float32))
        grouped = mx.unflatten(value, axis=-1, shape=(-1, self.group_size))
        variance = mx.mean(mx.square(grouped), axis=-1, keepdims=True)
        grouped = grouped * mx.rsqrt(variance + self.eps)
        return self.weight * grouped.flatten(-2).astype(dtype)


class Model(shared.Model):
    def __init__(self, args):
        super().__init__(args)
        for layer in self.layers:
            layer.norm = RMSNorm(args.hidden_size, args.layer_norm_epsilon)
            if layer.block_type == "M":
                mixer = layer.mixer
                mixer.norm = GatedRMSNorm(
                    mixer.intermediate_size,
                    args.layer_norm_epsilon,
                    mixer.intermediate_size // mixer.n_groups,
                )
        self.backbone.norm_f = RMSNorm(args.hidden_size, args.layer_norm_epsilon)

    def sanitize(self, weights):
        return super().sanitize(
            {k: v for k, v in weights.items() if not k.startswith("vision_projector.")}
        )


def get_model_classes(config):
    return Model, ModelArgs
