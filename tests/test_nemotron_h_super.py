# SPDX-License-Identifier: Apache-2.0
"""Super text extraction, precision boundary and real tiny conversion contracts."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.utils import load_model, save_model  # noqa: E402

from rapid_mlx.model_load_errors import load_model_checked  # noqa: E402
from rapid_mlx.models import nemotron_h_super as super_h  # noqa: E402
from scripts import nemotron_super_vl_convert as convert  # noqa: E402


def config():
    return dict(
        model_type="nemotron_h",
        architectures=["NemotronHForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        max_position_embeddings=1048576,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=32,
        mamba_proj_bias=False,
        ssm_state_size=32,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        moe_latent_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=256,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        time_step_limit=[0.0, {"__float__": "Infinity"}],
        layers_block_type=["mamba", "moe", "attention", "moe"],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
    )


@pytest.fixture(autouse=True)
def deterministic():
    device = mx.default_device()
    mx.random.seed(0)
    yield
    mx.set_default_device(device)
    mx.clear_cache()


def test_extraction_config_and_native_compatibility():
    from mlx_lm.models.nemotron_h import ModelArgs as NativeArgs

    inner = config() | {
        "auto_map": {"AutoConfig": "remote.Code"},
        "model_file": "evil.py",
    }
    omni = {"model_type": "nemotron_h_omni", "llm_config": inner, "vision_config": {}}
    before = deepcopy(omni)
    text = super_h.text_config(omni)
    assert omni == before
    assert text["model_type"] == "nemotron_h"
    assert text["num_hidden_layers"] == 4
    assert text["rapid_mlx_nemotron_h"] is True
    assert (
        not {
            "auto_map",
            "model_file",
            "num_nextn_predict_layers",
            "mtp_layers_block_type",
            "vision_config",
        }
        & text.keys()
    )
    args = super_h.ModelArgs.from_dict(omni)
    assert args.for_layer(1) is args
    assert args.time_step_limit == (0, float("inf"))
    assert NativeArgs.from_dict(text).hybrid_override_pattern == ["M", "E", "*", "E"]
    finite = super_h.ModelArgs.from_dict(config() | {"time_step_limit": [0.01, 0.2]})
    assert finite.time_step_limit == [0.01, 0.2]
    with pytest.raises(ValueError, match="Expected nemotron_h"):
        super_h.text_config({"model_type": "other"})
    # Real 88-layer uniform topology can be parsed without allocating weights.
    full = config() | {
        "layers_block_type": ["mamba", "moe"] * 40 + ["attention"] * 8,
        "hidden_size": 4096,
        "moe_latent_size": 1024,
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_intermediate_size": 2688,
    }
    args = super_h.ModelArgs.from_dict(full)
    assert args.num_hidden_layers == 88
    assert args.for_layer(87).n_routed_experts == 512
    assert args.for_layer(1).num_experts_per_tok == 22


def test_norm_rounds_activation_before_weight_and_router_fp32():
    norm = super_h.RMSNorm(128, 1e-5)
    norm.weight = mx.random.uniform(0.5, 1.5, (128,)).astype(mx.bfloat16)
    x = mx.random.normal((2, 3, 128)).astype(mx.bfloat16)
    value = x.astype(mx.float32)
    expected = (
        value * mx.rsqrt(mx.mean(value**2, axis=-1, keepdims=True) + 1e-5)
    ).astype(x.dtype) * norm.weight
    assert bool(mx.array_equal(norm(x), expected))
    assert not bool(
        mx.array_equal(norm(x), super_h.shared.NemotronHRMSNorm.__call__(norm, x))
    )
    model = super_h.Model(super_h.ModelArgs.from_dict(config()))
    model.set_dtype(mx.bfloat16)
    gate = model.layers[1].mixer.gate
    gate.weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
    indices, scores = gate(x)
    logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    raw = mx.sigmoid(logits)
    reference_indices = mx.argsort(-raw, axis=-1)[..., :2]
    # The unordered expert selection must contain the same experts.
    assert bool(mx.array_equal(mx.sort(indices), mx.sort(reference_indices)))
    selected = mx.take_along_axis(raw, indices, axis=-1)
    assert bool(mx.allclose(scores, selected / selected.sum(-1, keepdims=True) * 5))
    assert scores.dtype == mx.float32
    inputs = mx.array([[1, 2, 3, 4, 5]])
    full = model(inputs)
    cache = model.make_cache()
    cached = mx.concatenate(
        [model(inputs[:, :3], cache=cache), model(inputs[:, 3:], cache=cache)], axis=1
    )
    assert bool(
        mx.allclose(
            full.astype(mx.float32), cached.astype(mx.float32), atol=0.04, rtol=0.04
        )
    )
    assert full.shape == (1, 5, 128) and full.dtype == mx.bfloat16


def test_checked_loader_opt_in_and_explicit_callback(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(super_h.text_config(config())))
    captured = {}

    def loader(path, **kwargs):
        captured.update(kwargs)
        return "model", "config"

    assert load_model_checked(loader, tmp_path) == ("model", "config")
    assert captured["get_model_classes"]({}) == (super_h.Model, super_h.ModelArgs)
    assert captured["model_config"] == {"model_file": None, "auto_map": None}
    captured.clear()
    sentinel = object()
    load_model_checked(loader, tmp_path, get_model_classes=sentinel)
    assert captured == {"get_model_classes": sentinel}
    captured.clear()
    load_model_checked(loader, tmp_path, model_config={"rapid_mlx_nemotron_h": False})
    assert "get_model_classes" not in captured
    (tmp_path / "config.json").write_text(json.dumps(config()))
    captured.clear()
    load_model_checked(loader, tmp_path)
    assert captured == {}


@pytest.mark.parametrize("with_gate", [False, True])
def test_gated_norm_fp32_math(with_gate):
    norm = super_h.GatedRMSNorm(128, 1e-5, 32)
    norm.weight = mx.random.uniform(0.5, 1.5, (128,)).astype(mx.bfloat16)
    x = mx.random.normal((2, 3, 128)).astype(mx.bfloat16)
    gate = mx.random.normal(x.shape).astype(mx.bfloat16) if with_gate else None
    value = x.astype(mx.float32)
    if with_gate:
        import mlx.nn as nn

        value = value * nn.silu(gate.astype(mx.float32))
    grouped = value.reshape(2, 3, 4, 32)
    value = grouped * mx.rsqrt(mx.mean(grouped**2, -1, keepdims=True) + 1e-5)
    expected = value.reshape(x.shape).astype(x.dtype) * norm.weight
    assert bool(mx.array_equal(norm(x, gate), expected))


@pytest.mark.parametrize("bits", [3, 4])
def test_real_tiny_conversion_roundtrip(tmp_path, bits):
    source = tmp_path / "source"
    model = super_h.Model(super_h.ModelArgs.from_dict(config()))
    model.set_dtype(mx.bfloat16)
    original = dict(tree_flatten(model.parameters()))
    save_model(source, model)
    # Exercise the actual omni namespace, including Super's additional
    # vision final norm, rather than only a pre-sanitized text artifact.
    original_shard = source / "model.safetensors"
    mx.save_safetensors(
        str(original_shard),
        {f"language_model.{k}": v for k, v in original.items()}
        | {
            "vision_model.unused": mx.ones(1),
            "mlp1.unused": mx.ones(1),
            "vision_projector.vision_final_layernorm.weight": mx.ones(1),
            "language_model.mtp.unused": mx.ones(1),
        },
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h_omni", "llm_config": config()})
    )
    (source / "chat_template.jinja").write_text("{{ messages }}")
    output = tmp_path / "output"
    result = convert.main(
        ["--source", str(source), "--output", str(output), "--bits", str(bits)]
    )
    assert result["bytes"] == sum(f.stat().st_size for f in output.iterdir())
    assert "openmdw-1.1" in (output / "README.md").read_text()
    assert (output / "chat_template.jinja").read_text() == "{{ messages }}"
    loaded, text = load_model_checked(load_model, output)
    assert isinstance(loaded, super_h.Model)
    params = dict(tree_flatten(loaded.parameters()))
    for key, value in original.items():
        if any(k in key for k in ("lm_head", ".gate.", "latent_proj")):
            assert bool(mx.array_equal(params[key], value))
    expert = text["quantization"].get(
        "backbone.layers.1.mixer.switch_mlp.fc1", text["quantization"]
    )
    assert expert["bits"] == bits
    assert expert["group_size"] == (128 if bits == 3 else 64)
    assert bool(mx.all(mx.isfinite(loaded(mx.array([[1, 2, 3]])))))
    with pytest.raises(FileExistsError):
        convert.main(
            ["--source", str(source), "--output", str(output), "--bits", str(bits)]
        )
    index = json.loads((source / "model.safetensors.index.json").read_text())
    (source / next(iter(index["weight_map"].values()))).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        convert.main(
            [
                "--source",
                str(source),
                "--output",
                str(tmp_path / "missing"),
                "--bits",
                str(bits),
            ]
        )
    cache = output / ".cache"
    cache.mkdir()
    (cache / "ignored").write_text("ignore")
    assert convert.artifact_size(output) == result["bytes"]
    assert "96 GB+" in convert.model_card(bits, 70e9)


def test_metadata_recurrent_safety_and_parsers(tmp_path):
    from rapid_mlx.model_auto_config import (
        config_declares_linear_attention,
        detect_model_config,
    )

    text = super_h.text_config(config())
    (tmp_path / "config.json").write_text(json.dumps(text))
    profile = detect_model_config(str(tmp_path))
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid and profile.is_hybrid_explicit
    assert profile.is_moe and profile.is_text_only
    assert profile.supports_spec_decode is False
    assert config_declares_linear_attention(text)
    assert config_declares_linear_attention({"text_config": text})
    assert not config_declares_linear_attention({"layers_block_type": ["attention"]})
