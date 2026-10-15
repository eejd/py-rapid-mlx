# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored Ling 3.0 (bailing_hybrid) backbone.

Pins the contract that:

1. The vendored module registers into mlx-lm's importlib lookup.
2. The layer schedule puts MLA last in each group (plus trailing
   remainder layers) and layer 0 stays a dense MLP.
3. A tiny synthetic config constructs + runs: logits shape and
   incremental (cache) decode matching full prefill across all three
   cache kinds (conv states + KDA ssm state via ArraysCache, MLA
   KVCache).
4. ``sanitize`` stacks per-expert weights into SwitchGLU layout,
   remaps ``*_conv1d.weight`` into the nested conv module (with torch
   depthwise layout transposed), and drops MTP heads.
5. The KDA safe-gate law matches the fla ``USE_LOWER_BOUND`` kernel
   branch: ``g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``.
6. ``detect_model_config`` routes Ling names to glm47/qwen3 parsers
   with the hybrid flag set.

Numeric fidelity against the authoritative torch reference (HF remote
code on transformers 4.57 with a pure-CPU fla shim implementing the
exact kernel semantics) was verified out-of-band on identical random
weights: max abs logits diff 1.6e-6 full-prefill and 1.9e-6
token-by-token incremental (8-layer tiny config exercising
dense+MoE+KDA+MLA). The reference requires the fla shim harness, so
that check is not repeated here.
"""

import importlib
import sys

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.models import bailing_hybrid as bh  # noqa: E402

TINY = dict(
    model_type="bailing_hybrid",
    hidden_size=64,
    num_hidden_layers=8,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    vocab_size=128,
    layer_group_size=4,
    first_k_dense_replace=1,
    q_lora_rank=32,
    kv_lora_rank=48,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    num_experts=16,
    num_experts_per_tok=4,
    n_group=4,
    topk_group=2,
    moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=32,
)


def tiny_model():
    return bh.Model(bh.ModelArgs.from_dict(dict(TINY)))


@pytest.fixture(autouse=True)
def _clear_vendored_register():
    sys.modules.pop("mlx_lm.models.bailing_hybrid", None)
    yield
    sys.modules.pop("mlx_lm.models.bailing_hybrid", None)


def test_register_vendored_archs_makes_mlx_lm_loader_find_it():
    from rapid_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    assert "mlx_lm.models.bailing_hybrid" not in sys.modules
    _register_vendored_archs()
    assert "mlx_lm.models.bailing_hybrid" in sys.modules
    assert "bailing_hybrid" in _VENDORED_MODEL_TYPES
    mod = importlib.import_module("mlx_lm.models.bailing_hybrid")
    assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")


def test_layer_schedule():
    args = bh.ModelArgs.from_dict(dict(TINY))
    assert [i for i in range(8) if args.is_mla_layer(i)] == [3, 7]
    # Trailing remainder layers are MLA (torch ref).
    args = bh.ModelArgs.from_dict(dict(TINY, num_hidden_layers=6))
    assert [i for i in range(6) if args.is_mla_layer(i)] == [3, 4, 5]
    # Ling-3.0-tiny production shape: 24 layers -> 6 MLA.
    args = bh.ModelArgs.from_dict(dict(TINY, num_hidden_layers=24))
    assert sum(args.is_mla_layer(i) for i in range(24)) == 6


def test_forward_and_cache_parity():
    model = tiny_model()
    ids = mx.random.randint(0, TINY["vocab_size"], (1, 12))
    logits = model(ids)
    assert logits.shape == (1, 12, TINY["vocab_size"])

    cache = model.make_cache()
    from mlx_lm.models.cache import ArraysCache, KVCache

    kinds = [type(c) for c in cache]
    assert kinds.count(KVCache) == 2 and kinds.count(ArraysCache) == 6

    steps = [model(ids[:, i : i + 1], cache=cache) for i in range(12)]
    inc = mx.concatenate(steps, axis=1)
    diff = float(mx.abs(inc - logits).max())
    assert diff < 2e-3, diff


def test_dense_layer_zero():
    model = tiny_model()
    assert isinstance(model.model.layers[0].mlp, bh.BailingMLP)
    assert isinstance(model.model.layers[1].mlp, bh.BailingSparseMoE)


def test_kda_safe_gate_law():
    f = mx.array([[[[0.3, -0.2]]]])
    a_log = mx.array([0.5])
    dt_bias = mx.array([[0.1, 0.0]])
    g = bh._kda_gate(f, a_log, dt_bias, safe_gate=True, lower_bound=-5.0)
    import math

    a = math.exp(0.5)
    expect0 = -5.0 * (1 / (1 + math.exp(-a * 0.4)))
    expect1 = -5.0 * (1 / (1 + math.exp(-a * -0.2)))
    assert abs(float(g[0, 0, 0, 0]) - expect0) < 1e-5
    assert abs(float(g[0, 0, 0, 1]) - expect1) < 1e-5
    # Softplus law when the safe gate is off: -exp(A_log)*softplus(f+bias).
    g2 = bh._kda_gate(f, a_log, dt_bias, safe_gate=False, lower_bound=-5.0)
    expect_sp = -a * math.log1p(math.exp(0.4))
    assert abs(float(g2[0, 0, 0, 0]) - expect_sp) < 1e-5


def test_sanitize_expert_stack_conv_remap_and_mtp_drop():
    model = tiny_model()
    weights = {}
    # Per-expert weights for layer 1 (MoE).
    for e in range(TINY["num_experts"]):
        for m in ("gate_proj", "up_proj", "down_proj"):
            weights[f"model.layers.1.mlp.experts.{e}.{m}.weight"] = mx.zeros((2, 2))
    # Conv weight in torch depthwise layout [C, 1, K].
    weights["model.layers.0.attention.q_conv1d.weight"] = mx.zeros((8, 1, 4))
    # MTP tensors (flash/2.6 exports) must be dropped.
    weights["model.mtp_layers.0.foo.weight"] = mx.zeros((1,))
    out = model.sanitize(weights)
    assert out["model.layers.1.mlp.experts.gate_proj.weight"].shape == (
        TINY["num_experts"],
        2,
        2,
    )
    assert not any(".experts.0." in k for k in out)
    assert out["model.layers.0.attention.q_conv1d.conv.weight"].shape == (8, 4, 1)
    assert not any("mtp" in k for k in out)


def test_detect_model_config_routes_ling():
    from rapid_mlx.model_auto_config import detect_model_config

    for name in (
        "inclusionAI/Ling-3.0-tiny",
        "ling-3.0-tiny-4bit",
        "somewhere/Ling-2.6-flash-MLX-4bit",
    ):
        cfg = detect_model_config(name)
        assert cfg is not None, name
        assert cfg.tool_call_parser == "glm47"
        assert cfg.reasoning_parser == "qwen3"
        assert cfg.is_hybrid is True

    # Nearby names must not be routed to the Ling configuration
    # (codex r1 #2: the old disjunction was vacuously true).
    cfg = detect_model_config("mlx-community/gemma-4-26b-a4b-it-4bit")
    assert cfg is not None and cfg.tool_call_parser == "gemma4"
    # "sterling" contains the substring "ling" but not at a boundary the
    # pattern accepts.
    cfg = detect_model_config("sterling-3b")
    assert cfg is None or cfg.tool_call_parser != "glm47"
    # Version forms outside 3.0 / 2.6 are unknown Ling models — never
    # claim parser/hybrid support for them (codex r5).
    for name in ("acme/ling-3b", "acme/ling-20b", "acme/Ling-3.1-nano"):
        cfg = detect_model_config(name)
        assert cfg is None or cfg.tool_call_parser != "glm47", name
    assert detect_model_config("acme/sterling-3b") is None or (
        detect_model_config("acme/sterling-3b").tool_call_parser != "glm47"
    )


def test_glm47_parser_handles_bailing_wire():
    """The Bailing V3 template renders NAME immediately followed by
    <arg_key> (no newline) — glm47 must still extract the call."""
    from rapid_mlx.tool_parsers.glm47_tool_parser import Glm47ToolParser

    p = Glm47ToolParser(None)
    wire = (
        "<tool_call>get_weather<arg_key>city</arg_key>\n"
        "<arg_value>Tokyo</arg_value>\n</tool_call>"
    )
    r = p.extract_tool_calls(wire)
    assert r.tools_called
    assert r.tool_calls[0]["name"] == "get_weather"
    import json

    assert json.loads(r.tool_calls[0]["arguments"]) == {"city": "Tokyo"}


def test_glm47_parser_accepts_null_tools_from_plain_chat_request():
    """OpenAI request serialization includes ``tools: null`` for plain chat."""
    from rapid_mlx.tool_parsers.glm47_tool_parser import Glm47ToolParser

    parser = Glm47ToolParser(None)
    result = parser.extract_tool_calls("A normal answer", {"tools": None})

    assert not result.tools_called
    assert result.content == "A normal answer"


def test_gate_retain_all_groups():
    """topk_group == n_group keeps every group (codex r3 #1: the drop
    path faulted on argpartition(kth=-1))."""
    args = bh.ModelArgs.from_dict(dict(TINY, n_group=4, topk_group=4))
    model = bh.Model(args)
    out = model(mx.array([[1, 2, 3]]))
    assert out.shape == (1, 3, TINY["vocab_size"])


def test_short_conv_kernel_one_state():
    """kernel_size=1 must keep an EMPTY rolling state (codex r3 #3)."""
    conv = bh.ShortConv1d(4, 1)
    x = mx.ones((1, 5, 4))
    out, state = conv(x)
    assert out.shape == (1, 5, 4)
    assert state.shape == (1, 0, 4)
    out2, state2 = conv(mx.ones((1, 1, 4)), state)
    assert out2.shape == (1, 1, 4) and state2.shape == (1, 0, 4)


def test_gate_group_drop_masks_whole_group():
    """Group dropping must mask EVERY expert slot of a dropped group,
    not just slot 0 (codex r4 asked; mx.put_along_axis broadcasts the
    trailing size-1 index dim, so the implementation is correct — this
    test pins that semantic against regressions)."""
    args = bh.ModelArgs.from_dict(
        dict(
            TINY,
            num_experts=8,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
            moe_router_enable_expert_bias=True,
        )
    )
    gate = bh.BailingGate(args)
    # Bias group 1 (experts 4-7) far above group 0, EXCEPT expert 5,
    # which gets the single highest bias overall. If masking only hit
    # slot 0 of the dropped group, expert 5 (slot 1 of group 0's
    # competitor... construct: make group 0 the DROPPED group but give
    # its slot-1 expert (index 1) the highest selection score. A
    # correct whole-group mask never selects expert 1.
    import numpy as np

    bias = np.zeros(8, dtype=np.float32)
    bias[4:8] = 10.0  # group 1 wins the group competition
    bias[1] = 100.0  # slot 1 of dropped group 0: highest single score
    # But group score = sum of top-2 per group: group0 = 100 + ~0,
    # group1 = 20. Make group1 still win: raise its two best.
    bias[4] = 60.0
    bias[5] = 60.0  # group1 top-2 sum = 120 > group0's ~100
    gate.expert_bias = mx.array(bias)
    gate.weight = mx.zeros_like(gate.weight)

    idx, _ = gate(mx.zeros((1, 3, TINY["hidden_size"])))
    chosen = set(np.array(idx).flatten().tolist())
    assert chosen <= {4, 5, 6, 7}, chosen  # nothing from dropped group 0


def test_alias_pins_ling_configuration():
    import json
    from pathlib import Path

    aliases = json.loads(
        (Path(__file__).parent.parent / "rapid_mlx" / "aliases.json").read_text()
    )
    entry = aliases["ling-3.0-tiny-4bit"]
    assert entry["hf_path"] == "rapid-mlx/Ling-3.0-tiny-MLX-4bit"
    assert entry["tool_call_parser"] == "glm47"
    assert entry["reasoning_parser"] == "qwen3"
    assert entry["is_hybrid"] is True
    assert entry["is_hybrid_explicit"] is True
    assert entry["is_moe"] is True


# Text-only heterogeneous Puzzle contracts share this Apple CI test entry.
class TestNemotronBolt:
    @staticmethod
    def config(**overrides):
        config = dict(
            model_type="nemotron_h_puzzle",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            max_position_embeddings=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            attention_bias=False,
            mamba_num_heads=2,
            mamba_head_dim=16,
            mamba_proj_bias=False,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            moe_latent_size=16,
            moe_shared_expert_intermediate_size=48,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            n_routed_experts=512,
            num_experts_per_tok=9,
            norm_topk_prob=True,
            routed_scaling_factor=5.0,
            time_step_limit=[0.0, {"__float__": "Infinity"}],
            block_configs=[
                {"block_type": "mamba"},
                {
                    "block_type": "moe",
                    "n_routed_experts": 4,
                    "moe_intermediate_size": 16,
                    "num_experts_per_tok": 2,
                },
                {"block_type": "attention"},
                {
                    "block_type": "moe",
                    "n_routed_experts": 6,
                    "moe_intermediate_size": 24,
                    "num_experts_per_tok": 3,
                },
            ],
        )
        return config | overrides

    @staticmethod
    def module():
        from rapid_mlx.models import nemotron_h_puzzle

        return nemotron_h_puzzle

    def model(self, **overrides):
        module = self.module()
        return module.Model(module.ModelArgs.from_dict(self.config(**overrides)))

    def test_config_omni_and_blockwise_truth(self):
        from copy import deepcopy

        module = self.module()
        inner = self.config(auto_map={"AutoConfig": "remote.Config"})
        omni = {
            "model_type": "nemotron_h_omni",
            "llm_config": inner,
            "vision_config": {"hidden_size": 999},
        }
        before = deepcopy(omni)
        text = module.text_config(omni)
        assert text["model_type"] == "nemotron_h_puzzle"
        assert "auto_map" not in text and "vision_config" not in text
        args = module.ModelArgs.from_dict(omni)
        assert args.hybrid_override_pattern == ["M", "E", "*", "E"]
        assert args.time_step_limit == (0, float("inf"))
        assert args.for_layer(1).n_routed_experts == 4
        assert args.for_layer(3).n_routed_experts == 6
        assert args.for_layer(1).moe_intermediate_size == 16
        assert args.for_layer(3).num_experts_per_tok == 3
        assert args.n_routed_experts == 512
        assert omni == before
        inner.pop("num_hidden_layers")
        assert module.ModelArgs.from_dict(inner).num_hidden_layers == 4
        with pytest.raises(ValueError, match="Expected"):
            module.text_config({"model_type": "other"})

    def test_config_validation_and_fallbacks(self):
        module = self.module()
        for changes, message in [
            ({"num_hidden_layers": 3}, "one entry"),
            ({"block_configs": [{"block_type": "bad"}] * 4}, "unsupported"),
            ({"block_configs": [{"block_type": "moe"}] * 4}, "missing"),
            ({"hybrid_override_pattern": ["M"] * 4}, "conflicts"),
        ]:
            with pytest.raises(ValueError, match=message):
                module.ModelArgs.from_dict(self.config(**changes))
        config = self.config()
        config["block_configs"][1]["n_routed_experts"] = 1
        with pytest.raises(ValueError, match="top-k"):
            module.ModelArgs.from_dict(config).for_layer(1)
        args = module.ModelArgs.from_dict(
            self.config(
                block_configs=None,
                layers_block_type=["mamba", "attention"],
                num_hidden_layers=None,
                time_step_limit=None,
                time_step_min=0.002,
            )
        )
        assert args.num_hidden_layers == 2 and args.for_layer(0) is args
        assert args.time_step_limit == (0.002, float("inf"))
        assert (
            module.ModelArgs.from_dict(
                self.config(time_step_limit=None)
            ).time_step_limit[0]
            == 0
        )
        args = module.ModelArgs.from_dict(self.config())
        args.block_configs[1].pop("n_routed_experts")
        assert args.for_layer(1).n_routed_experts == 512

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
    def test_four_layer_forward_and_cached_decode(self, dtype):
        model = self.model()
        model.set_dtype(dtype)
        # Nonzero router weights exercise FP32 matmul rather than tied zero logits.
        for index in (1, 3):
            gate = model.layers[index].mixer.gate
            gate.weight = mx.random.normal(gate.weight.shape).astype(dtype) * 0.1
        tokens = mx.array([[1, 2, 3, 4]])
        expected = model(tokens)
        cache = model.make_cache()
        prefill = model(tokens[:, :3], cache)
        decoded = model(tokens[:, 3:], cache)
        mx.eval(expected, prefill, decoded)
        assert expected.shape == (1, 4, 64)
        assert bool(mx.all(mx.isfinite(expected)))
        atol = 0.08 if dtype == mx.bfloat16 else 1e-4
        assert bool(mx.allclose(expected[:, :3], prefill, atol=atol, rtol=atol))
        assert bool(mx.allclose(expected[:, 3:], decoded, atol=atol, rtol=atol))
        assert len(cache) == 2
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            mixer = model.layers[index].mixer
            assert mixer.gate.weight.shape == (count, 32)
            assert mixer.switch_mlp.fc1.weight.shape == (count, width, 16)
        assert model.cast_predicate("backbone.layers.0.mixer.A_log") is False
        assert (
            model.cast_predicate("backbone.layers.1.mixer.gate.e_score_correction_bias")
            is False
        )
        assert model.cast_predicate("lm_head.weight") is True

    @pytest.mark.parametrize(
        "blocks",
        [
            [{"block_type": "mlp"}],
            [{"block_type": "attention"}],
            [
                {
                    "block_type": "moe",
                    "moe_intermediate_size": 16,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                }
            ],
            [{"block_type": "mamba"}],
        ],
    )
    def test_single_kind_schedules(self, blocks):
        model = self.model(
            block_configs=blocks,
            num_hidden_layers=1,
            moe_latent_size=None,
            n_shared_experts=None,
        )
        logits = model(mx.array([[1, 2]]), model.make_cache())
        mx.eval(logits)
        assert logits.shape == (1, 2, 64) and bool(mx.all(mx.isfinite(logits)))

    def test_sanitize_heterogeneous_experts_and_omni(self):
        model = self.model()
        original = {
            "language_model.backbone.embeddings.weight": mx.ones((64, 32)),
            "language_model.backbone.norm_f.weight": mx.ones(32),
            "language_model.lm_head.weight": mx.ones((64, 32)),
            "language_model.mtp.layers.0.weight": mx.ones(1),
            "vision_model.radio_model.weight": mx.ones(1),
            "mlp1.0.weight": mx.ones(1),
            "language_model.backbone.layers.0.mixer.conv1d.weight": mx.ones((96, 1, 4)),
        }
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            for expert in range(count):
                prefix = (
                    f"language_model.backbone.layers.{index}.mixer.experts.{expert}"
                )
                original[f"{prefix}.up_proj.weight"] = mx.full((width, 16), expert)
                original[f"{prefix}.down_proj.weight"] = mx.full((16, width), expert)
        result = model.sanitize(original)
        assert len(original) == 27
        assert not any(
            key.startswith(("language_model.", "vision_model.", "mlp1.", "mtp."))
            for key in result
        )
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            prefix = f"backbone.layers.{index}.mixer.switch_mlp"
            assert result[f"{prefix}.fc1.weight"].shape == (count, width, 16)
            assert result[f"{prefix}.fc2.weight"].shape == (count, 16, width)
            assert bool(mx.all(result[f"{prefix}.fc1.weight"][-1] == count - 1))
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (96, 4, 1)
        twice = model.sanitize(result)
        assert all(twice[key] is value for key, value in result.items())
        assert "backbone.embeddings.weight" in model.sanitize(
            {"model.embeddings.weight": mx.ones(1)}
        )

    def test_norm_router_and_quantization_boundaries(self):
        module = self.module()
        norm = module.NemotronHRMSNorm(32, 1e-5)
        norm.weight = mx.linspace(0.5, 1.5, 32)
        x = mx.linspace(-2, 3, 32).astype(mx.bfloat16)
        reference = (
            x.astype(mx.float32)
            * mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32))) + 1e-5)
            * norm.weight
        ).astype(x.dtype)
        assert bool(mx.array_equal(norm(x), reference))
        gate = self.model().layers[1].mixer.gate
        gate.weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
        gate.e_score_correction_bias = mx.array([0.1, -0.1, 0, 0.2])
        x = mx.random.normal((2, 32)).astype(mx.bfloat16)
        inds, scores = gate(x)
        logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
        raw = mx.sigmoid(logits)
        expected_inds = mx.argsort(-(raw + gate.e_score_correction_bias), axis=-1)[
            :, :2
        ]
        assert bool(mx.array_equal(mx.sort(inds), mx.sort(expected_inds)))
        selected = mx.take_along_axis(raw, inds, axis=-1)
        assert bool(mx.allclose(scores, selected / selected.sum(-1, keepdims=True) * 5))
        predicate = self.model().quant_predicate
        for path in (
            "lm_head",
            "backbone.layers.1.mixer.gate",
            "backbone.layers.1.mixer.fc1_latent_proj",
            "backbone.layers.1.mixer.fc2_latent_proj",
        ):
            assert predicate(path, None) is False
        assert predicate("backbone.layers.1.mixer.switch_mlp.fc1", None) is True

    def test_ssm_native_dt_and_chunked_reference(self):
        module = self.module()
        dt = mx.array([[-1.17, 0.23]], dtype=mx.bfloat16)
        bias = mx.array([0.1, -0.2], dtype=mx.bfloat16)
        assert bool(
            mx.array_equal(
                module.compute_dt_native(dt, bias, (0, 10)),
                mx.clip(mx.logaddexp(0, dt + bias), 0, 10),
            )
        )
        x = mx.random.normal((2, 5, 2, 16))
        b = mx.random.normal((2, 5, 1, 32))
        c = mx.random.normal((2, 5, 1, 32))
        dt = mx.random.normal((2, 5, 2))
        args = (x, mx.zeros(2), b, c, mx.ones(2), dt, mx.zeros(2))
        full, state = module.ssm_attn(*args)
        chunked, chunk_state = module.ssm_attn(*args, step=2)
        assert bool(mx.allclose(full, chunked, atol=2e-5, rtol=2e-5))
        assert bool(mx.allclose(state, chunk_state, atol=2e-5, rtol=2e-5))
        mask = mx.array([[True] * 5, [True, True, False, False, False]])
        initial = mx.ones((2, 2, 16, 32))
        result, state = module.ssm_attn(
            *args, state=initial, lengths=mx.array([5, -1]), mask=mask, step=2
        )
        mx.eval(result, state)
        assert bool(mx.array_equal(state[1], initial[1]))

    def test_registration_and_text_loader(self, tmp_path, monkeypatch, caplog):
        import builtins
        import importlib.util
        import json

        from rapid_mlx.utils import tokenizer

        name = "mlx_lm.models.nemotron_h_puzzle"
        monkeypatch.delitem(sys.modules, name, raising=False)
        tokenizer._VENDORED_MODEL_TYPES.discard("nemotron_h_puzzle")
        original_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if kwargs.get("fromlist", args[2] if len(args) > 2 else None) == (
                "nemotron_h_puzzle",
            ):
                raise ImportError("injected")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", fail_import)
            tokenizer._register_vendored_archs()
        assert "nemotron_h_puzzle vendored module failed" in caplog.text
        assert "nemotron_h_puzzle" not in tokenizer._VENDORED_MODEL_TYPES
        with monkeypatch.context() as patch:
            original_find = importlib.util.find_spec

            def no_spec(module):
                if module == name:
                    raise ValueError("missing spec")
                return original_find(module)

            patch.setattr(importlib.util, "find_spec", no_spec)
            tokenizer._register_vendored_archs()
        assert sys.modules[name] is self.module()
        tokenizer._register_vendored_archs()
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "nemotron_h_puzzle"})
        )
        assert tokenizer._is_vendored_arch_model(str(tmp_path))
        expected = (object(), object())
        monkeypatch.setattr(
            tokenizer, "_load_with_tokenizer_fallback", lambda *a, **kw: expected
        )
        assert tokenizer._load_model_with_fallback_impl(str(tmp_path)) is expected
        monkeypatch.delitem(sys.modules, name)
        with monkeypatch.context() as patch:
            original_find = importlib.util.find_spec
            patch.setattr(
                importlib.util,
                "find_spec",
                lambda module: object() if module == name else original_find(module),
            )
            tokenizer._register_vendored_archs()
        assert name not in sys.modules
        tokenizer._VENDORED_MODEL_TYPES.discard("nemotron_h_puzzle")

    def test_full_size_config_without_allocating_weights(self):
        module = self.module()
        blocks = [{"block_type": "mamba"}] * 40 + [{"block_type": "attention"}] * 8
        counts = (240, 304, 368, 432, 496)
        widths = (1536, 2048, 2304, 2560)
        blocks += [
            {
                "block_type": "moe",
                "n_routed_experts": counts[i % 5],
                "moe_intermediate_size": widths[i % 4],
                "num_experts_per_tok": 9,
            }
            for i in range(40)
        ]
        config = self.config(
            block_configs=blocks,
            num_hidden_layers=None,
            hidden_size=4096,
            moe_latent_size=1024,
        )
        config.pop("n_routed_experts")
        args = module.ModelArgs.from_dict(
            {"model_type": "nemotron_h_omni", "llm_config": config}
        )
        assert args.num_hidden_layers == 88
        assert args.n_routed_experts is None
        assert {args.for_layer(i).n_routed_experts for i in range(48, 88)} == set(
            counts
        )
        assert {args.for_layer(i).moe_intermediate_size for i in range(48, 88)} == set(
            widths
        )
