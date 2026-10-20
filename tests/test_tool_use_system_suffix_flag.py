# SPDX-License-Identifier: Apache-2.0
"""``--tool-use-system-suffix {on,off}``: the default tool-use system suffix is opt-in.

``off`` is the default of this fork (a deliberate divergence from upstream): the chat
route appends nothing to the system prompt of a tool request. ``on`` reproduces upstream:
``_TOOL_USE_SYSTEM_SUFFIX`` is appended to the system prompt of a tool request while a
tool parser is active (``tool_choice`` auto or unset). The ``tool_choice=required`` /
named-function suffixes are a client's explicit request and are injected in both modes.
"""

from __future__ import annotations

import copy
import sys
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rapid_mlx.cli as cli
from rapid_mlx import server
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
    """Messages the engine received for one request; ``suffix=None`` leaves the default (off)."""
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


def test_default_is_off_and_adds_nothing() -> None:
    assert reset_config().tool_use_system_suffix is False
    assert _post([_USER], suffix=None, tools=_TOOLS) == [_USER]
    assert _post([_SYSTEM, _USER], suffix=None, tools=_TOOLS) == [_SYSTEM, _USER]


def test_on_reproduces_the_upstream_messages() -> None:
    # no system message: the suffix (stripped) becomes the system message
    assert _post([_USER], suffix=True, tools=_TOOLS) == [
        {"role": "system", "content": _TOOL_USE_SYSTEM_SUFFIX.strip()},
        _USER,
    ]
    # a system message: the suffix is appended to it, nothing else changes
    assert _post([_SYSTEM, _USER], suffix=True, tools=_TOOLS) == [
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


@pytest.mark.parametrize("suffix", [False, True])
def test_required_and_named_tool_choice_suffixes_are_injected_in_both_modes(
    suffix: bool,
) -> None:
    required = _post([_USER], suffix=suffix, tools=_TOOLS, tool_choice="required")
    assert required[0] == {
        "role": "system",
        "content": _TOOL_USE_REQUIRED_SUFFIX.strip(),
    }
    named = _post(
        [_USER],
        suffix=suffix,
        tools=_TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert "get_weather" in named[0]["content"]
    assert named[0]["content"].startswith("CRITICAL")


@pytest.mark.parametrize("suffix", [False, True])
def test_nameless_function_tool_choice_is_rejected_before_any_injection(
    suffix: bool,
) -> None:
    """``{'type': 'function', 'function': {}}`` names no tool: the route answers 400 in
    both modes, so the ``elif cfg.tool_use_system_suffix`` fallback for it in the suffix
    selection (``routes/chat.py``) is not reachable through HTTP."""
    engine = _RecordingEngine()
    cfg = reset_config()
    cfg.engine, cfg.model_name, cfg.model_registry = engine, "test-model", None
    cfg.no_thinking, cfg.tool_call_parser = True, "hermes"
    cfg.tool_use_system_suffix = suffix
    app = FastAPI()
    app.include_router(chat_router)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [_USER],
            "tools": _TOOLS,
            "tool_choice": {"type": "function", "function": {}},
            "max_tokens": 8,
        },
    )
    assert response.status_code == 400
    assert "requires function.name" in response.text
    assert engine.last_messages is None


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, raising=False)
    return monkeypatch


def test_resolver_precedence_flag_then_env_then_off(_clean_env) -> None:
    resolve = server._resolve_tool_use_system_suffix
    assert resolve(None) is False
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "on")
    assert resolve(None) is True
    assert resolve("off") is False  # the flag wins
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, " OFF ")
    assert resolve(None) is False
    assert resolve("on") is True
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "")
    assert resolve(None) is False


def test_resolver_rejects_an_invalid_value(_clean_env) -> None:
    _clean_env.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, "maybe")
    with pytest.raises(SystemExit, match="RAPID_MLX_TOOL_USE_SYSTEM_SUFFIX='maybe'"):
        server._resolve_tool_use_system_suffix(None)
    _clean_env.delenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV)
    with pytest.raises(SystemExit, match="--tool-use-system-suffix='auto'"):
        server._resolve_tool_use_system_suffix("auto")


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
    monkeypatch.setattr(server, "_tool_use_system_suffix", True)
    server._sync_config()
    assert get_config().tool_use_system_suffix is True
    monkeypatch.setattr(server, "_tool_use_system_suffix", False)
    server._sync_config()
    assert get_config().tool_use_system_suffix is False


class _StopError(Exception):
    """Raised once the serve path has built SchedulerConfig (after the flag wiring)."""


def _run_serve(argv: list[str]) -> None:
    """Drive the real ``main()`` -> ``serve_command`` up to ``SchedulerConfig(...)``.

    Same boundary mocks as ``tests/test_cli_response_cache_flag.py``; ``--text-only`` keeps
    a vision-capable alias off the optional-runtime guard. The flag wiring in
    ``serve_command`` precedes the ``SchedulerConfig`` construction.
    """

    def _stop(*args, **kwargs):
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch.object(
            cli, "_gather_kv_cache_dtype_inputs", lambda *a, **k: ({}, None)
        ),
        mock.patch(
            "rapid_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch(
            "rapid_mlx.utils.tokenizer.load_model_with_fallback",
            return_value=(object(), object()),
        ),
        mock.patch("rapid_mlx.scheduler.SchedulerConfig", _stop),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises(_StopError),
    ):
        cli.main()


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "extra, env, expected",
    [
        ([], None, False),
        (["--tool-use-system-suffix", "on"], None, True),
        (["--tool-use-system-suffix", "off"], "on", False),
        ([], "on", True),
    ],
)
def test_serve_command_wires_flag_and_env_into_the_server(
    monkeypatch, extra, env, expected
) -> None:
    """MUTATION-KILL: deleting the ``server._tool_use_system_suffix = ...`` block in
    ``serve_command`` (cli.py) makes the two ``True`` rows fail."""
    pytest.importorskip("mlx")
    monkeypatch.delenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, env)
    # the opposite of the expectation first, so an untouched global cannot pass
    monkeypatch.setattr(server, "_tool_use_system_suffix", not expected)
    _run_serve(["serve", "qwen3.5-4b-4bit", "--text-only", *extra])
    assert server._tool_use_system_suffix is expected


@pytest.mark.parametrize(
    "extra, env, expected",
    [
        ([], None, False),
        (["--tool-use-system-suffix", "on"], None, True),
        (["--tool-use-system-suffix", "off"], "on", False),
        ([], "on", True),
    ],
)
def test_server_main_wires_flag_and_env(monkeypatch, extra, env, expected) -> None:
    """``python -m rapid_mlx.server`` has its own parser and globals; the wiring sits
    before ``load_model``, which is stopped here."""

    def _stop(*args, **kwargs):
        raise _StopError

    monkeypatch.delenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(server.TOOL_USE_SYSTEM_SUFFIX_ENV, env)
    monkeypatch.setattr(server, "_tool_use_system_suffix", not expected)
    monkeypatch.setattr(server, "load_model", _stop)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx-server", "--model", "m", *extra])
    with pytest.raises((_StopError, SystemExit)):
        server.main()
    assert server._tool_use_system_suffix is expected
