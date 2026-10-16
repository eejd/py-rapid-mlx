# SPDX-License-Identifier: Apache-2.0
"""Offline routing, template and fail-closed provenance contracts for 4.7."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jinja2.sandbox import SandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "rapid_mlx/models/mlx_vlm_vendored/models/minicpmv4_7"
FIXTURES = ROOT / "tests/fixtures/minicpmv4_7"
COMMIT = "b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0"


def test_vendored_provenance():
    inventory = json.loads((VENDOR / "provenance.json").read_text())
    assert inventory["upstream_commit"] == COMMIT
    assert set(inventory["files"]) == {p.name for p in VENDOR.glob("*.py")}
    assert "Permission is hereby granted" in (VENDOR / "LICENSE").read_text()
    for name, entry in inventory["files"].items():
        source = (VENDOR / name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == entry["vendored_sha256"]
        assert COMMIT in source
        body = "\n".join(source.split("\n")[3:])
        if name == "_vision_compat.py":
            region = body[body.index("    def get_vision_embedding(") :]
            assert hashlib.sha256(region.encode()).hexdigest() == entry["region_sha256"]
            assert "VENDOR-DEVIATION(subset)" in body
            continue
        if name == "_vision.py":
            region = body[body.index("class VisionModel(nn.Module):") :]
            assert hashlib.sha256(region.encode()).hexdigest() == entry["region_sha256"]
            assert "VENDOR-DEVIATION(subset)" in body
            continue
        if body.startswith("# VENDOR-DEVIATION"):
            assert "import-redirect" in body
            while body.startswith("#"):
                body = body.split("\n", 1)[1]
        for new, old in (
            ("from mlx_vlm.models.base import", "from ..base import"),
            ("from ._config import", "from ..minicpmv4_6.config import"),
            (
                "from ._vision import VisionModel",
                "from ..minicpmv4_6.vision import VisionModel",
            ),
            (
                "from ._vision_compat import Model as MiniCPMVModel",
                "from ..minicpmv4_6.minicpmv4_6 import Model as MiniCPMVModel",
            ),
            ("from mlx_vlm.models.minicpmv4_6.", "from ..minicpmv4_6."),
            ("from mlx_vlm.models.qwen3_5_moe.", "from ..qwen3_5_moe."),
        ):
            body = body.replace(new, old)
        assert hashlib.sha256(body.encode()).hexdigest() == entry["upstream_sha256"], (
            name
        )


@pytest.mark.parametrize("size,moe", [("1B", False), ("35B-A3B", True)])
def test_routing_and_profiles(size, moe, tmp_path):
    from rapid_mlx.api.utils import is_mllm_model
    from rapid_mlx.byom import preflight
    from rapid_mlx.byom._vlm_model_types import MLX_VLM_MODEL_TYPES
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.model_auto_config import detect_model_config

    quant = "mxfp4" if moe else "8bit"
    alias = f"minicpm-v4.7-{size.lower()}-{quant}"
    repo = f"mlx-community/MiniCPM-V-4.7-{size}-{quant}"
    assert resolve_profile(alias).hf_path == repo
    assert resolve_profile(alias).supports_image_input
    assert "minicpmv4_7" in MLX_VLM_MODEL_TYPES
    assert "minicpmv4_7" in preflight.RAPID_MODEL_TYPES
    # Arbitrary local names exercise config-driven detection independently.
    (tmp_path / "config.json").write_bytes(
        (FIXTURES / f"{size}-config.json").read_bytes()
    )
    for name in (alias, repo, str(tmp_path)):
        assert is_mllm_model(name)
        config = detect_model_config(name)
        assert config.tool_call_parser == "qwen3_coder_xml"
        assert config.reasoning_parser == "qwen3"
        assert config.is_moe is moe
        assert config.is_hybrid is moe
        assert config.is_hybrid_explicit
        assert not config.supports_spec_decode


@pytest.mark.parametrize("size", ["1B", "35B-A3B"])
def test_checkpoint_template_and_existing_parser(size):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    template = SandboxedEnvironment().from_string(
        (FIXTURES / f"{size}-chat_template.jinja").read_text()
    )
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "weather", "arguments": {"city": "Paris"}}}
            ],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True)
    call = prompt[prompt.index("<tool_call>", prompt.index("<|im_start|>assistant")) :]
    call = call[: call.index("</tool_call>") + len("</tool_call>")]
    result = Qwen3CoderToolParser().extract_tool_calls(call, request={"tools": tools})
    assert result.tools_called
    assert result.tool_calls[0]["name"] == "weather"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}
    # Raw Jinja leaves the flag undefined; mlx_vlm supplies False when omitted.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
    ).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_canvas_rope_text_and_image():
    # rope.py has only NumPy dependencies; load it without the MLX package init.
    from types import SimpleNamespace

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "minicpm_canvas_test", VENDOR / "rope.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(**json.loads((FIXTURES / "1B-config.json").read_text()))
    rope = module.CanvasRoPE(config)
    tokens = np.array(
        [[1, config.image_start_id, config.image_token_id, config.image_end_id, 2]]
    )
    positions, delta = rope.get_rope_index(
        tokens,
        mm_token_type_ids=(tokens == config.image_token_id).astype(np.int32),
        target_sizes=np.array([[4, 4]]),
    )
    assert positions.shape == (3, 1, 5)
    assert delta.shape == (1, 1)
    assert positions[:, 0, -1].tolist() == [3, 3, 3]
