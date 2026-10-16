# SPDX-License-Identifier: Apache-2.0
"""Super text extraction, precision boundary and real tiny conversion contracts."""

import json
import os
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
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "auto_map": {"AutoProcessor": "remote.Processor"},
                "processor_class": "RemoteProcessor",
            }
        )
    )
    output = tmp_path / "output"
    result = convert.main(
        ["--source", str(source), "--output", str(output), "--bits", str(bits)]
    )
    assert result["bytes"] == sum(f.stat().st_size for f in output.iterdir())
    assert (
        not {"auto_map", "model_file"}
        & json.loads((output / "config.json").read_text()).keys()
    )
    assert "openmdw-1.1" in (output / "README.md").read_text()
    assert (output / "chat_template.jinja").read_text() == "{{ messages }}"
    assert json.loads((output / "tokenizer_config.json").read_text()) == {
        "tokenizer_class": "TokenizersBackend"
    }
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
    assert ("128 GB+" if bits == 4 else "45.903 GB") in convert.model_card(bits, 70e9)


@pytest.fixture
def staged_conversion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config()))
    (source / "model.safetensors").write_bytes(b"source weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}})
    )
    (source / "chat_template.jinja").write_text("{{ messages }}")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Backend", "processor_class": "Remote"})
    )
    output = tmp_path / "new-parent" / "output"
    argv = ["--source", str(source), "--output", str(output), "--bits", "4"]
    saved = []
    monkeypatch.setattr(
        convert, "load_model_checked", lambda *a, **kw: (object(), config())
    )
    monkeypatch.setattr(convert, "quantization_predicate", lambda *a: None)
    monkeypatch.setattr(
        convert, "quantize_model", lambda model, cfg, *a, **kw: (model, cfg)
    )

    def save(staging, model, **kwargs):
        assert not output.exists()
        assert staging.parent == output.parent
        assert staging.name.startswith(".output.partial-")
        (staging / "model.safetensors").write_bytes(b"converted weights")
        saved.append(staging)

    monkeypatch.setattr(convert, "save_model", save)
    monkeypatch.setattr(
        convert, "save_config", lambda cfg, path: path.write_text(json.dumps(cfg))
    )
    return argv, output, saved


def test_converter_atomic_success(staged_conversion, capsys):
    argv, output, saved = staged_conversion
    result = convert.main(argv)
    assert output.is_dir()
    assert "openmdw-1.1" in (output / "README.md").read_text()
    assert (output / "chat_template.jinja").read_text() == "{{ messages }}"
    assert json.loads((output / "tokenizer_config.json").read_text()) == {
        "tokenizer_class": "Backend"
    }
    assert (output / "model.safetensors").read_bytes() == b"converted weights"
    assert (output / "config.json").is_file()
    assert result["output"] == str(output)
    assert result["bytes"] == sum(path.stat().st_size for path in output.iterdir())
    assert json.loads(capsys.readouterr().out) == result
    assert not saved[0].exists()
    assert not list(output.parent.glob(".output.partial-*"))


@pytest.mark.parametrize(
    "failure_at,error",
    [
        ("save_model", OSError),
        ("save_model", KeyboardInterrupt),
        ("save_config", ValueError),
        ("copyfile", OSError),
        ("model_card", ValueError),
        ("rename", OSError),
    ],
)
def test_converter_atomic_failure_and_retry(
    staged_conversion, monkeypatch, failure_at, error
):
    argv, output, saved = staged_conversion
    owner = {"copyfile": convert.shutil, "rename": os}.get(failure_at, convert)
    original = getattr(owner, failure_at)

    def fail(*args, **kwargs):
        if failure_at == "save_model":
            original(*args, **kwargs)
        assert saved[0].is_dir()
        assert (saved[0] / "model.safetensors").is_file()
        assert not output.exists()
        raise error("injected conversion failure")

    with monkeypatch.context() as patch:
        patch.setattr(owner, failure_at, fail)
        with pytest.raises(error, match="injected conversion failure"):
            convert.main(argv)
    assert not output.exists()
    assert not saved[0].exists()
    assert not list(output.parent.glob(".output.partial-*"))
    result = convert.main(argv)
    assert result["output"] == str(output)
    assert (output / "README.md").is_file()
    assert not list(output.parent.glob(".output.partial-*"))


def test_converter_preserves_destination_created_during_conversion(
    staged_conversion, monkeypatch
):
    argv, output, saved = staged_conversion
    original = convert.model_card

    def create_destination(*args):
        output.mkdir()
        (output / "keep").write_text("another conversion")
        return original(*args)

    monkeypatch.setattr(convert, "model_card", create_destination)
    with pytest.raises(FileExistsError):
        convert.main(argv)
    assert (output / "keep").read_text() == "another conversion"
    assert list(output.iterdir()) == [output / "keep"]
    assert not saved[0].exists()
    assert not list(output.parent.glob(".output.partial-*"))


def test_mixed_allocation_and_precision_boundaries():
    # Full Super topology, without allocating the full checkpoint.
    layers = (
        1,
        3,
        5,
        8,
        10,
        12,
        14,
        17,
        19,
        21,
        23,
        26,
        28,
        30,
        32,
        34,
        37,
        39,
        41,
        43,
        45,
        48,
        50,
        52,
        54,
        56,
        59,
        61,
        63,
        65,
        67,
        70,
        72,
        74,
        76,
        79,
        81,
        83,
        85,
        87,
    )
    assert "45.903 GB" in convert.model_card(3, 43e9)
    assert "26 interior" in convert.model_card(3, 43e9)
    model = super_h.Model(super_h.ModelArgs.from_dict(config()))
    predicate = convert.quantization_predicate(model, 3)
    allocated = []
    for layer in layers:
        for projection in ("fc1", "fc2"):
            path = f"backbone.layers.{layer}.mixer.switch_mlp.{projection}"
            recipe = predicate(path, None)
            allocated.append(recipe["bits"])
            assert recipe["group_size"] == 128
    assert allocated.count(2) == 52
    assert allocated.count(3) == 28
    assert all(
        predicate(f"backbone.layers.{i}.mixer.switch_mlp.fc1", None)["bits"] == 3
        for i in layers[:7] + layers[-7:]
    )
    for path in (
        "lm_head",
        "backbone.layers.17.mixer.gate",
        "backbone.layers.17.mixer.fc1_latent_proj",
        "backbone.layers.17.mixer.fc2_latent_proj",
    ):
        assert predicate(path, None) is False
    assert predicate("backbone.embeddings", None) == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    assert predicate("backbone.layers.17.mixer.shared_experts.fc1", None) == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }


def test_mixed_two_bit_expert_roundtrip(tmp_path, monkeypatch):
    # Exercise real two-bit packing/loading alongside three-bit experts.
    monkeypatch.setattr(convert, "TWO_BIT_LAYERS", frozenset({3}))
    source = tmp_path / "source"
    model = super_h.Model(super_h.ModelArgs.from_dict(config()))
    model.set_dtype(mx.bfloat16)
    save_model(source, model)
    (source / "config.json").write_text(json.dumps(super_h.text_config(config())))
    output = tmp_path / "mixed"
    convert.main(["--source", str(source), "--output", str(output), "--bits", "3"])
    loaded, metadata = load_model_checked(load_model, output)
    for layer, bits in ((1, 3), (3, 2)):
        for projection in ("fc1", "fc2"):
            path = f"backbone.layers.{layer}.mixer.switch_mlp.{projection}"
            assert metadata["quantization"][path] == {
                "bits": bits,
                "group_size": 128,
                "mode": "affine",
            }
    assert bool(mx.all(mx.isfinite(loaded(mx.array([[1, 2, 3]])))))


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


@pytest.mark.parametrize("bits", [3, 4])
def test_generated_card_published_guidance(bits):
    card = convert.model_card(bits, 43e9 if bits == 3 else 70e9)
    assert "Requires Rapid-MLX 0.16.2 or newer." in card
    for internal in ("handoff", "/private/tmp", "PR #", "Memory qualification pending"):
        assert internal not in card
    if bits == 4:
        assert "Recommend **128 GB+** unified memory" in card
        assert "72.7 GB at 1K context on M3 Ultra" in card
        assert "96 GB Macs may need a raised GPU wired-memory limit" in card
        assert "are not yet verified" in card
    else:
        assert "Targets **64 GB** unified" in card
        assert "for 1,024 prompt tokens plus 64 generated tokens" in card
        assert "prefill chunks of 256" in card
        assert "256 GiB M3 Ultra; a physical 64 GB Mac has not been tested" in card


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.safetensors",
        "nested/../../escaped.safetensors",
        "/absolute.safetensors",
    ],
)
def test_converter_rejects_untrusted_shard_names(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config()))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": name}})
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="shard path escapes"):
        convert.main(["--source", str(source), "--output", str(output), "--bits", "4"])
    assert not output.exists()
    assert not (tmp_path / "escaped.safetensors").exists()


@pytest.mark.parametrize("root_name", ["source", "view"])
def test_shard_path_rejects_symlink_escape(tmp_path, root_name):
    root = tmp_path / root_name
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="shard path escapes"):
        convert.shard_path(root, "linked/shard.safetensors")
    (root / "shard.safetensors").symlink_to(outside / "shard.safetensors")
    with pytest.raises(ValueError, match="shard path escapes"):
        convert.shard_path(root, "shard.safetensors")


def test_shard_path_allows_contained_nested_file(tmp_path):
    assert convert.shard_path(tmp_path, "nested/shard.safetensors") == (
        tmp_path / "nested/shard.safetensors"
    )


def mtp_fixture(tmp_path):
    import mlx.nn as nn

    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import (
        MTPHead,
        quantization_predicate,
        sidecar_manifest,
    )

    args = super_h.ModelArgs.from_dict(config())
    model = super_h.Model(args)
    model.set_dtype(mx.bfloat16 if mx.default_device() == mx.gpu else mx.float32)
    head = MTPHead(args)
    head.set_dtype(mx.bfloat16 if mx.default_device() == mx.gpu else mx.float32)
    nn.quantize(head, bits=4, group_size=64, class_predicate=quantization_predicate)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(head.parameters()))
    )
    (tmp_path / "nemotron_mtp.json").write_text(json.dumps(sidecar_manifest(args)))
    return model


@pytest.mark.parametrize("keep", [1, 2])
def test_mtp_mamba_partial_acceptance(tmp_path, keep):
    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import (
        NemotronMambaCache,
        inject_mtp_support,
    )

    model = mtp_fixture(tmp_path)
    assert inject_mtp_support(model, tmp_path)
    serial, verified = model.make_cache(), model.make_cache()
    prefix = mx.array([[1, 2, 3, 4]])
    model(prefix, cache=serial)
    model(prefix, cache=verified)
    tokens = mx.array([[5, 6, 7]])
    expected = []
    for p in range(keep):
        expected.append(model(tokens[:, p : p + 1], cache=serial))
    logits, _ = model(tokens, cache=verified, return_hidden=True, n_confirmed=1)
    mx.eval(logits, [c.state for c in verified], expected)
    recurrent = verified[0]
    assert isinstance(recurrent, NemotronMambaCache)
    assert len(recurrent.rollback_state) == 2
    with pytest.raises(ValueError, match="boundary"):
        recurrent.restore_rollback(3, 3)
    recurrent.restore_rollback(3 - keep, 3)
    verified[1].trim(3 - keep)
    for a, b in zip(serial[0].state, verified[0].state):
        assert bool(mx.allclose(a, b, atol=2e-3, rtol=2e-3))
    assert recurrent.rollback_state is None
    assert recurrent.n_confirmed_for_mtp == 0
    actual = model(mx.array([[8]]), cache=verified)
    target = model(mx.array([[8]]), cache=serial)
    assert bool(
        mx.allclose(
            actual.astype(mx.float32), target.astype(mx.float32), atol=0.08, rtol=0.08
        )
    )


def test_mtp_sidecar_strict_loading_opt_in_and_generation(tmp_path):
    from rapid_mlx.spec_decode.mtp.detect import MTPEligibility, detect_mtp_eligibility
    from rapid_mlx.spec_decode.mtp.dispatch import (
        dispatch_mtp_inject,
        dispatch_mtp_validate,
    )
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import inject_mtp_support

    model = mtp_fixture(tmp_path)
    assert not hasattr(model, "mtp")
    cfg = super_h.text_config(config())
    assert detect_mtp_eligibility(cfg) is MTPEligibility.NONE
    assert (
        detect_mtp_eligibility(cfg, has_external_sidecar=True) is MTPEligibility.CHAIN
    )
    assert (
        detect_mtp_eligibility(config(), has_external_sidecar=True)
        is MTPEligibility.NONE
    )
    assert not inject_mtp_support(model)
    assert not inject_mtp_support(object(), tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not inject_mtp_support(model, empty)
    assert not dispatch_mtp_validate(model, "nemotron_h")
    manifest = tmp_path / "nemotron_mtp.json"
    good = manifest.read_text()
    manifest.write_text("{}")
    assert not inject_mtp_support(model, tmp_path)
    manifest.write_text(good)
    assert dispatch_mtp_inject(model, "nemotron_h", mtp_sidecar=tmp_path)
    assert dispatch_mtp_validate(model, "nemotron_h")
    assert model.make_mtp_cache()[0].offset == 0
    prompt = mx.array([1, 2, 3, 4])
    for k in (0, 1, 2):
        out = list(
            mtp_generate_step(prompt, model, max_tokens=8, max_k=k, disable_auto_k=True)
        )
        assert len(out) == 8
    # Ordinary and hidden-state target routes agree without a cache.
    ordinary = model(prompt[None])
    hidden_logits, hidden = model(prompt[None], return_hidden=True)
    assert bool(mx.array_equal(ordinary, hidden_logits))
    assert hidden.shape == (1, 4, 128)
    cache = model.make_cache()
    cache[0].lengths = mx.array([3])
    with pytest.raises(ValueError, match="unpadded"):
        model(prompt[None], cache=cache, n_confirmed=1)
    assert cache[0].n_confirmed_for_mtp == 0
    from mlx_lm.models.cache import ArraysCache

    cache = [ArraysCache(2), model.make_cache()[1]]
    with pytest.raises(ValueError, match="fresh"):
        model(prompt[None], cache=cache, n_confirmed=1)


def test_mtp_cli_alias_gates():
    from rapid_mlx.cli import (
        _alias_mtp_declaration,
        _alias_mtp_default_enabled,
        _resolve_mtp_depth_for_model,
    )
    from rapid_mlx.scheduler import _config_vetted_mtp_supports_spec_decode

    for alias in ("nemotron-3.5-super-vl", "nemotron-3.5-super-vl-3bit"):
        assert _alias_mtp_declaration(alias)[1] == 1
        assert not _alias_mtp_default_enabled(alias)
    assert _config_vetted_mtp_supports_spec_decode("nemotron_h")
    assert _resolve_mtp_depth_for_model("nemotron_h", 2, explicit=True) == 2
    assert _resolve_mtp_depth_for_model("nemotron_h", 2, explicit=False) == 1
    assert _resolve_mtp_depth_for_model("nemotron_h", 0, explicit=True) == 0
    with pytest.raises(ValueError, match="Nemotron"):
        _resolve_mtp_depth_for_model("nemotron_h", 3, explicit=True)


def test_additive_mtp_conversion_and_failure_cleanup(tmp_path, monkeypatch):
    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import MTPHead, inject_mtp_support

    source = tmp_path / "source"
    source.mkdir()
    cfg = config()
    (source / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h_omni", "llm_config": cfg})
    )
    args = super_h.ModelArgs.from_dict(cfg)
    head = MTPHead(args)
    head.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(head.parameters()))
    for target, original in (("fc1", "up_proj"), ("fc2", "down_proj")):
        packed = weights.pop(f"layers.1.mixer.switch_mlp.{target}.weight")
        for expert in range(args.n_routed_experts):
            weights[f"layers.1.mixer.experts.{expert}.{original}.weight"] = packed[
                expert
            ]
    weights = {"language_model.mtp." + name: value for name, value in weights.items()}
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    index = {"weight_map": {name: "model.safetensors" for name in weights}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    output = tmp_path / "sidecar"
    result = convert.main(
        ["--source", str(source), "--output", str(output), "--bits", "3", "--mtp-only"]
    )
    assert result["bytes"] > 0
    assert sorted(
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    ) == ["mtp/model.safetensors", "mtp/nemotron_mtp.json"]
    assert inject_mtp_support(super_h.Model(args), output)
    with pytest.raises(FileExistsError):
        convert.convert_mtp_sidecar(source, output)
    bad = tmp_path / "bad"
    (source / "config.json").write_text(
        json.dumps(cfg | {"num_nextn_predict_layers": 0})
    )
    with pytest.raises(ValueError, match="attention/MoE"):
        convert.convert_mtp_sidecar(source, bad)
    (source / "config.json").write_text(json.dumps(cfg))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="no native"):
        convert.convert_mtp_sidecar(source, bad)
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    original_save = mx.save_safetensors

    def fail_save(*args, **kwargs):
        raise OSError("test failure")

    monkeypatch.setattr(mx, "save_safetensors", fail_save)
    with pytest.raises(OSError, match="test failure"):
        convert.convert_mtp_sidecar(source, bad)
    assert not bad.exists()
    assert not list(tmp_path.glob(".bad.partial-*"))
    monkeypatch.setattr(mx, "save_safetensors", original_save)

    def raced_save(*args, **kwargs):
        original_save(*args, **kwargs)
        bad.mkdir()

    monkeypatch.setattr(mx, "save_safetensors", raced_save)
    with pytest.raises(FileExistsError):
        convert.convert_mtp_sidecar(source, bad)
    assert not list(tmp_path.glob(".bad.partial-*"))


def test_mtp_shared_cache_symlink_containment(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--model"
    source = repo / "snapshots" / "revision"
    source.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "hash").write_bytes(b"weights")
    (source / "model.safetensors").symlink_to(blobs / "hash")
    assert (
        convert.mtp_source_shard(source, "model.safetensors").read_bytes() == b"weights"
    )
    outside = tmp_path / "outside"
    outside.write_bytes(b"forbidden")
    (source / "evil.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        convert.mtp_source_shard(source, "evil.safetensors")


@pytest.mark.parametrize("stop", [False, True])
def test_mtp_benchmark_harness_tiny(tmp_path, monkeypatch, stop):
    import sys
    from types import SimpleNamespace

    from scripts import benchmark_nemotron_mtp as bench

    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    model = mtp_fixture(sidecar)

    class Tokenizer:
        eos_token_ids = set(range(128)) if stop else set()

        def encode(self, text):
            return [1, 2, 3]

        def apply_chat_template(self, *args, **kwargs):
            return [4, 5, 6]

    monkeypatch.setattr(
        bench, "load_model_with_fallback", lambda path: (model, Tokenizer())
    )
    monkeypatch.setattr(
        bench,
        "subprocess",
        SimpleNamespace(check_output=lambda *a, **kw: "tiny-test-commit\n"),
    )
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--model",
            "tiny",
            "--sidecar",
            str(sidecar),
            "--output",
            str(output),
            "--contexts",
            "16",
            "--tokens",
            "4",
            "--repeats",
            "2",
        ],
    )
    result = bench.main()
    assert len(result["rows"]) == 24
    assert all(
        len(r["tokens"]) == (1 if stop else 4) and r["tok_s"] >= 0
        for r in result["rows"]
    )


def test_mtp_empty_cache_mask_history_and_corrupt_sidecar(tmp_path):
    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import inject_mtp_support

    model = mtp_fixture(tmp_path)
    weights_file = tmp_path / "model.safetensors"
    good = mx.load(str(weights_file))
    corrupt = dict(good)
    key = next(k for k in good if good[k].dtype == mx.uint32)
    corrupt[key] = good[key].astype(mx.float32)
    mx.save_safetensors(str(weights_file), corrupt)
    with pytest.raises(ValueError, match="dtype"):
        inject_mtp_support(model, tmp_path)
    assert not hasattr(model, "mtp")
    corrupt = dict(good)
    corrupt.pop(key)
    mx.save_safetensors(str(weights_file), corrupt)
    with pytest.raises(ValueError):
        inject_mtp_support(model, tmp_path)
    mx.save_safetensors(str(weights_file), good)
    assert inject_mtp_support(model, tmp_path)
    cache = model.make_cache()[0]
    mixer = model.layers[0].mixer
    x = model.backbone.embeddings(mx.array([[1, 2, 3]]))
    cache.n_confirmed_for_mtp = 1
    y = mixer(x, mask=mx.array([[True, True, False]]), cache=cache)
    mx.eval(y, cache.state)
    assert len(cache.rollback_state) == 2
    assert cache.rollback_state[0][1] is not None
    cache.restore_rollback(2, 3)
    assert cache[0].shape[1] == model.args.conv_kernel - 1


def test_mtp_real_cli_default_and_explicit_preset():
    from rapid_mlx.cli import _normalize_speculative_config_or_exit
    from tests.test_mtp_default_off_3115 import _args

    for alias in ("nemotron-3.5-super-vl", "nemotron-3.5-super-vl-3bit"):
        args = _args(alias)
        _normalize_speculative_config_or_exit(args)
        assert args.spec_decode == "none"
        assert args._speculative_config is None
        args = _args(alias, '{"method":"mtp"}')
        _normalize_speculative_config_or_exit(args)
        assert args.spec_decode == "mtp"
        assert args.mtp_sidecar.endswith(
            "MLX-3bit" if alias.endswith("3bit") else "MLX-4bit"
        )
        assert args.mtp_max_k == 1


def test_mtp_cache_batch_handoff_and_prefix_type(tmp_path):
    from mlx_lm.generate import PromptProcessingBatch

    from rapid_mlx.memory_cache import _class_name_is_recurrent
    from rapid_mlx.spec_decode.mtp.nemotron_h_inject import (
        NemotronMambaCache,
        inject_mtp_support,
    )

    model = mtp_fixture(tmp_path)
    assert inject_mtp_support(model, tmp_path)
    assert _class_name_is_recurrent(NemotronMambaCache.__name__)
    empty = NemotronMambaCache(2)
    assert empty.extract(0).state == [None, None]
    merged = NemotronMambaCache.merge([empty, NemotronMambaCache(2)])
    assert isinstance(merged, NemotronMambaCache)
    batch = PromptProcessingBatch(model, [1], [model.make_cache()], prefill_step_size=2)
    batch.prompt([[1, 2, 3]])
    gb = batch.generate([[4]])
    assert isinstance(gb.prompt_cache[0], NemotronMambaCache)
    assert gb.prompt_cache[0].left_padding is None
    model(mx.array([[5, 6]]), cache=gb.prompt_cache, return_hidden=True, n_confirmed=1)
    extracted = gb.extract_cache(0)
    assert isinstance(extracted[0], NemotronMambaCache)
    assert extracted[0].rollback_state is None
