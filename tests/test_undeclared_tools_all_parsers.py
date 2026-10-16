# SPDX-License-Identifier: Apache-2.0
"""Declared tools authorize parsed calls across buffered and streaming paths."""

import json
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.routes.chat import router
from rapid_mlx.service import helpers
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers.abstract_tool_parser import ExtractedToolCallInformation
from tests.test_empty_rejected_tool_reply import (
    NOTICE,
    REJECTED,
    TOOLS,
    _Engine,
    _Request,
)
from tests.test_postprocessor import _make_cfg, _make_output

GEMMA = '<|tool_call>call:send_email{body:<|"|>hello<|"|>}<tool_call|>'
VALID = '<tool_call><function=write_file>{"path":"x"}</function></tool_call>'


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "parser,text,expected,valid",
    [
        ("nemotron", REJECTED, NOTICE, False),
        ("nemotron", "Before. " + REJECTED + " After.", "Before.  After.", False),
        ("nemotron", VALID + REJECTED, "", True),
        ("gemma4", GEMMA, NOTICE, False),
    ],
)
def test_chat_all_formats(stream, parser, text, expected, valid):
    cfg = reset_config()
    cfg.engine = _Engine(text, deltas=text.splitlines(keepends=True))
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser
    app = FastAPI()
    app.include_router(router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                c
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for c in json.loads(line[6:])["choices"]
            ]
            content = "".join(c["delta"].get("content") or "" for c in choices)
            calls = [tc for c in choices for tc in c["delta"].get("tool_calls", [])]
            finish = choices[-1]["finish_reason"]
        else:
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            calls = choice["message"].get("tool_calls") or []
            finish = choice["finish_reason"]
        assert content.strip() == expected.strip()
        assert bool(calls) == valid
        assert all(
            tc.get("function", {}).get("name", "write_file") == "write_file"
            for tc in calls
        )
        assert finish == ("tool_calls" if valid else "stop")
    finally:
        reset_config()


@pytest.mark.parametrize(
    "parser,text,recognized",
    [
        (
            "hermes",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
            True,
        ),
        (
            "qwen",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
            True,
        ),
        (
            "qwen3_xml",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
            True,
        ),
        (
            "hy_v3",
            '<tool_call:opensource>send_email<tool_sep:opensource>{"x":1}<end_of_tool_call:opensource>',
            False,
        ),
        (
            "k2_horizon",
            "<ifm|tool_calls><ifm|tool_call>send_email<ifm|arg_key>x</ifm|arg_key><ifm|arg_value>1</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
            False,
        ),
        ("gemma4", GEMMA, True),
        ("nemotron", REJECTED, True),
        ("llama", '<function=send_email>{"x":1}</function>', True),
        ("mistral", '[TOOL_CALLS] [{"name":"send_email","arguments":{"x":1}}]', True),
        (
            "harmony",
            '<|start|>assistant to=functions.send_email<|channel|>commentary json<|message|>{"x":1}<|call|>',
            True,
        ),
        (
            "glm47",
            "<tool_call>send_email<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            False,
        ),
    ],
)
def test_buffered_parser_authorization(monkeypatch, parser, text, recognized):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", parser)
    content, calls = helpers._parse_tool_calls_with_parser(text, _Request())
    assert not calls
    if recognized:
        assert helpers._was_undeclared_tool_dropped()
        assert not content.strip()


def processor(parser="hermes"):
    return StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser=parser),
        request={"tools": TOOLS},
    )


def anchor(name, index=0):
    return {
        "index": index,
        "id": f"call_{index}",
        "function": {"name": name, "arguments": "{"},
    }


def test_incremental_drop_and_renumber():
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser._undeclared_dropped = 0
    pp.tool_parser.extract_tool_calls_streaming.side_effect = [
        {"tool_calls": [anchor("send_email")], "content": "Before"},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"x":1}'}}]},
        {"tool_calls": [{"function": {"arguments": "ignored"}}]},
        {"tool_calls": [anchor("write_file", 1)]},
        {"tool_calls": [{"index": 1, "function": {"arguments": '"path":"x"}'}}]},
    ]
    assert pp._detect_tool_calls("<") == {"content": "Before"}
    assert not pp.tool_calls_detected
    assert pp._detect_tool_calls("x") == {"content": ""}
    assert pp._detect_tool_calls("x") == {"content": ""}
    assert pp._detect_tool_calls("x")["tool_calls"][0]["index"] == 0
    assert pp._detect_tool_calls("x")["tool_calls"][0]["index"] == 0
    assert pp.undeclared_tool_dropped
    pp.reset()
    assert not pp.undeclared_tool_dropped


def test_channel_undeclared_finish():
    pp = processor()
    events = pp.process_chunk(
        _make_output(
            "",
            channel="tool_call",
            tool_calls=[{"name": "send_email", "arguments": "{}"}],
            finished=True,
            finish_reason="tool_calls",
        )
    )
    assert not any(e.tool_calls for e in events)
    assert not pp.tool_calls_detected
    assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize("mode", ["extract", "fallback", "raw", "reported"])
def test_finalize_authorizes_every_recovery(mode):
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser._undeclared_dropped = int(mode == "reported")
    pp.tool_parser.finalize_legacy_raw_stream = None
    pp.tool_parser.flush_pending = Mock(return_value=None)
    pp.tool_parser.finish = Mock(return_value=None)
    pp.tool_parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=mode == "extract",
        tool_calls=[{"id": "bad", "name": "send_email", "arguments": "{}"}]
        if mode == "extract"
        else [],
        content=None,
    )
    if mode == "raw":
        pp.tool_parser.finalize_legacy_raw_stream = Mock(
            return_value={"tool_calls": [anchor("send_email")]}
        )
    pp.tool_accumulated_text = (
        '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>'
    )
    events = pp.finalize()
    assert not any(e.tool_calls for e in events)
    assert not pp.tool_calls_detected


@pytest.mark.parametrize(
    "req", [None, {}, {"tools": []}, {"tools": [{"name": "write_file"}]}, _Request()]
)
def test_buffered_structured_shapes(monkeypatch, req):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "gemma4")
    content, calls = helpers._parse_tool_calls_with_parser(
        "Narration",
        req,
        structured_tool_calls=[
            {"name": "send_email", "arguments": "{}"},
            {"name": "write_file", "arguments": "{}"},
        ],
    )
    assert content == "Narration"
    assert [c.function.name for c in calls] == (
        ["write_file"] if req and req != {"tools": []} else ["send_email", "write_file"]
    )


def test_nemotron_boundaries_and_logging(caplog):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    pp = NemotronToolParser()
    pp.reset()
    prefix = REJECTED.split("</tool_call>")[0]
    assert pp.extract_tool_calls_streaming(
        "", prefix, prefix, request={"tools": TOOLS}
    ) in (None, {"content": ""})
    delta = "</tool_call>"
    assert (
        pp.extract_tool_calls_streaming(
            prefix, prefix + delta, delta, request={"tools": TOOLS}
        )
        is None
    )
    assert pp._undeclared_dropped == 1
    assert sum("dropped undeclared" in r.message for r in caplog.records) == 1
    assert not any("unhandled variant" in r.message for r in caplog.records)
    for text in [
        "```xml\n" + REJECTED + "\n```",
        REJECTED.replace("<tool_call>", "").replace("</tool_call>", ""),
        REJECTED.replace("today's price", "x" * 32768),
    ]:
        pp.reset()
        assert pp.extract_tool_calls(text, {"tools": TOOLS}).content == text
        assert pp._undeclared_dropped == 0


def test_nemotron_every_byte_split():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    wire = "Before. " + REJECTED + " After."
    for split in range(1, len(wire)):
        parser = NemotronToolParser()
        parser.reset()
        previous = ""
        visible = ""
        for delta in [wire[:split], wire[split:]]:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous, current, delta, request={"tools": TOOLS}
            )
            if result:
                visible += result.get("content", "")
            previous = current
        assert visible == "Before.  After.", (split, visible)


def test_stream_filter_no_declared_tools():
    pp = processor()
    pp.request = None
    calls = [anchor("send_email")]
    assert pp._filter_declared_calls(calls, incremental=True) == calls


def test_nemotron_partial_prefix_flush():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    assert parser.extract_tool_calls_streaming("", "Hello <too", "Hello <too") == {
        "content": "Hello "
    }
    assert parser.flush_held_content("Hello <too") == "<too"
    assert parser.flush_held_content("Hello <too") == ""
    assert parser.flush_held_content(REJECTED) == ""


def test_declared_pydantic_tool_in_dict():
    from rapid_mlx.api.models import ToolDefinition
    from rapid_mlx.tool_call_authorization import declared_names

    assert declared_names(
        {"tools": [ToolDefinition(**TOOLS[0]), None, {}, {"name": None}]}
    ) == {"write_file"}


@pytest.mark.parametrize("called", [False, True])
def test_inferred_parser_reports_drops(monkeypatch, called):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", False)
    monkeypatch.setattr(cfg, "reasoning_parser_name", "minimax")
    parser = Mock()
    parser._undeclared_dropped = 1
    parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=called,
        content="Kept",
        tool_calls=[{"id": "ok", "name": "write_file", "arguments": "{}"}]
        if called
        else [],
    )
    monkeypatch.setattr(
        helpers.ToolParserManager, "get_tool_parser", lambda _: lambda _: parser
    )
    content, calls = helpers._parse_tool_calls_with_parser(REJECTED, _Request())
    assert content == "Kept"
    assert bool(calls) == called
    assert helpers._was_undeclared_tool_dropped()


@pytest.mark.parametrize("channel", [None, "content"])
def test_dropped_stream_keeps_prose_and_honest_finish(channel):
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser._undeclared_dropped = 0
    pp.tool_parser.extract_tool_calls_streaming.side_effect = [
        {"tool_calls": [anchor("send_email")], "content": "Before. "},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"x":1}'}}]},
        {"content": "After."},
    ]
    events = []
    for i, text in enumerate(["<attempt>", "arguments", "tail"]):
        events += pp.process_chunk(
            _make_output(
                text,
                channel=channel,
                finished=i == 2,
                finish_reason="tool_calls" if i == 2 else None,
            )
        )
    assert not any(e.tool_calls for e in events)
    assert "".join(e.content or "" for e in events) == "Before. After."
    assert events[-1].finish_reason == "stop"
    assert not pp.tool_calls_detected


def test_qwen_structured_forced_undeclared_reports_drop(monkeypatch):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    content, calls = helpers._parse_tool_calls_with_parser(
        "Kept",
        {
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "write_file"}},
        },
        structured_tool_calls=[{"name": "send_email", "arguments": "{}"}],
    )
    assert content == "Kept"
    assert calls is None
    assert helpers._was_undeclared_tool_dropped()


def test_buffered_authorization_ignores_forced_choice(monkeypatch):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "gemma4")
    _, calls = helpers._parse_tool_calls_with_parser(
        "",
        {
            "tools": [*TOOLS, {"name": "send_email"}],
            "tool_choice": {"type": "function", "function": {"name": "write_file"}},
        },
        structured_tool_calls=[{"name": "send_email", "arguments": "{}"}],
    )
    assert calls[0].function.name == "send_email"
    assert not helpers._was_undeclared_tool_dropped()
