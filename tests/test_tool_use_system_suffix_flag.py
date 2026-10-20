# SPDX-License-Identifier: Apache-2.0
"""``--tool-use-system-suffix {on,off}``: the default tool-use system suffix is optional.

``on`` (the default) is the upstream behaviour: the chat route appends
``_TOOL_USE_SYSTEM_SUFFIX`` to the system prompt of a tool request while a tool
parser is active. ``off`` removes exactly that text and nothing else.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import cli, server
from rapid_mlx.config import get_config, reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.service.helpers import _TOOL_USE_REQUIRED_SUFFIX, _TOOL_USE_SYSTEM_SUFFIX

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


class _RecordingEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    supports_tool_calls = True

    def __init__(self) -> None:
        self.last_messages: Any = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_messages = copy.deepcopy(messages)
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _post(messages: list[dict], *, suffix: bool | None, **body: Any) -> list:
    """Messages the engine received for one request; ``suffix=None`` leaves the default."""
    engine = _RecordingEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    if suffix is not None:
        cfg.tool_use_system_suffix = suffix
    app = FastAPI()
    app.include_router(chat_router)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": messages, "max_tokens": 8, **body},
    )
    assert response.status_code == 200, response.text
    return engine.last_messages


_USER = {"role": "user", "content": "weather in Paris?"}
_SYSTEM = {"role": "system", "content": "You are terse."}


def test_default_is_on_and_appends_the_suffix_exactly_as_before() -> None:
    assert reset_config().tool_use_system_suffix is True
    # no system message: the suffix (stripped) becomes the system message
    assert _post([_USER], suffix=None, tools=_TOOLS) == [
        {"role": "system", "content": _TOOL_USE_SYSTEM_SUFFIX.strip()},
        _USER,
    ]
    # a system message: the suffix is appended to it, nothing else changes
    assert _post([_SYSTEM, _USER], suffix=None, tools=_TOOLS) == [
        {"role": "system", "content": _SYSTEM["content"] + _TOOL_USE_SYSTEM_SUFFIX},
        _USER,
    ]


def test_off_removes_exactly_the_suffix() -> None:
    assert _post([_USER], suffix=False, tools=_TOOLS) == [_USER]
    assert _post([_SYSTEM, _USER], suffix=False, tools=_TOOLS) == [_SYSTEM, _USER]
    blocks = {"role": "system", "content": [{"type": "text", "text": "Be terse."}]}
    # list-of-blocks system content (the route flattens it to text either way)
    off = _post([blocks, _USER], suffix=False, tools=_TOOLS)
    assert off[0]["content"] == "Be terse."
    assert off[1:] == [_USER]


def test_off_and_on_differ_by_the_suffix_only() -> None:
    on = _post([_SYSTEM, _USER], suffix=True, tools=_TOOLS)
    off = _post([_SYSTEM, _USER], suffix=False, tools=_TOOLS)
    assert on[0]["content"] == off[0]["content"] + _TOOL_USE_SYSTEM_SUFFIX
    assert on[1:] == off[1:]


def test_requests_without_tools_are_the_same_in_both_modes() -> None:
    assert _post([_SYSTEM, _USER], suffix=True) == _post(
        [_SYSTEM, _USER], suffix=False
    )


def test_off_keeps_the_required_and_named_tool_choice_suffixes() -> None:
    required = _post([_USER], suffix=False, tools=_TOOLS, tool_choice="required")
    assert required[0] == {
        "role": "system",
        "content": _TOOL_USE_REQUIRED_SUFFIX.strip(),
    }
    named = _post(
        [_USER],
        suffix=False,
        tools=_TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert "get_weather" in named[0]["content"]
    assert named[0]["content"].startswith("CRITICAL")


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, raising=False)
    return monkeypatch


def test_resolver_precedence_flag_then_env_then_on(_clean_env) -> None:
    resolve = server._resolve_tool_use_system_suffix
    assert resolve(None) is True
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "off")
    assert resolve(None) is False
    assert resolve("on") is True  # the flag wins
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, " ON ")
    assert resolve(None) is True
    assert resolve("off") is False
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "")
    assert resolve(None) is True


def test_resolver_rejects_an_invalid_value(_clean_env) -> None:
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "maybe")
    with pytest.raises(SystemExit, match="RAPID_MLX_TOOL_USE_SYSTEM_SUFFIX='maybe'"):
        server._resolve_tool_use_system_suffix(None)


def test_serve_parser_accepts_on_off_and_defaults_to_unset() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "m"]).tool_use_system_suffix is None
    assert (
        parser.parse_args(
            ["serve", "m", "--tool-use-system-suffix", "off"]
        ).tool_use_system_suffix
        == "off"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "m", "--tool-use-system-suffix", "auto"])


def test_sync_config_carries_the_flag_into_the_live_config(monkeypatch) -> None:
    monkeypatch.setattr(server, "_tool_use_system_suffix", False)
    server._sync_config()
    assert get_config().tool_use_system_suffix is False
    monkeypatch.setattr(server, "_tool_use_system_suffix", True)
    server._sync_config()
    assert get_config().tool_use_system_suffix is True
