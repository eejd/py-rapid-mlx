# SPDX-License-Identifier: Apache-2.0
"""Buffered suffix cleanup with unchanged streaming behavior."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.anthropic import router as messages_router
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.responses import router as responses_router

TOOLS = [
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
TRUNCATED = [
    "<tool_call>",
    "<tool_call>\n<function=get_weather",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
]
COMPLETE = "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, text, finish_reason="length", chunks=None):
        self.text = text
        self.finish_reason = finish_reason
        self.chunks = chunks or [text]

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self.text,
            prompt_tokens=4,
            completion_tokens=4,
            finished=True,
            finish_reason=self.finish_reason,
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for index, chunk in enumerate(self.chunks):
            accumulated += chunk
            last = index == len(self.chunks) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4,
                completion_tokens=index + 1,
                finished=last,
                finish_reason=self.finish_reason if last else None,
            )


@pytest.fixture
def client():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    for router in (chat_router, messages_router, responses_router):
        app.include_router(router)
    yield TestClient(app), cfg
    reset_config()


def _reply(
    client,
    route,
    text,
    *,
    stream=False,
    finish_reason="length",
    choice="auto",
    tools=TOOLS,
    chunks=None,
    budget=16,
    with_reasoning=False,
):
    http, cfg = client
    cfg.engine = _Engine(text, finish_reason, chunks)
    body = {
        "model": "test-model",
        "stream": stream,
        "max_tokens": budget,
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": tools,
        "tool_choice": choice,
    }
    if route == "messages":
        body["tools"] = [
            {"name": t["function"]["name"], "input_schema": t["function"]["parameters"]}
            for t in tools
        ]
        body["tool_choice"] = {"type": "auto"}
    elif route == "responses":
        body["input"] = "Weather?"
        body["max_output_tokens"] = body.pop("max_tokens")
        body.pop("messages")
        body["tools"] = [{"type": "function", **t["function"]} for t in tools]
    path = {
        "chat": "chat/completions",
        "messages": "messages",
        "responses": "responses",
    }[route]
    response = http.post("/v1/" + path, json=body)
    assert response.status_code == 200, response.text
    if stream:
        choices = [
            c
            for line in response.text.splitlines()
            if line.startswith("data: {")
            for c in json.loads(line[6:])["choices"]
        ]
        return (
            "".join(c["delta"].get("content") or "" for c in choices),
            [c["delta"]["tool_calls"] for c in choices if c["delta"].get("tool_calls")],
            choices[-1]["finish_reason"],
        )
    data = response.json()
    if with_reasoning:
        return data
    if route == "chat":
        choice = data["choices"][0]
        return (
            choice["message"].get("content") or "",
            choice["message"].get("tool_calls"),
            choice["finish_reason"],
        )
    if route == "messages":
        return (
            "".join(b["text"] for b in data["content"] if b["type"] == "text"),
            [b for b in data["content"] if b["type"] == "tool_use"],
            data["stop_reason"],
        )
    return (
        "".join(
            c["text"]
            for b in data["output"]
            if b["type"] == "message"
            for c in b["content"]
            if c["type"] == "output_text"
        ),
        [b for b in data["output"] if b["type"] == "function_call"],
        data["status"],
    )


@pytest.mark.parametrize("parser", ["qwen3_coder_xml", "hermes"])
@pytest.mark.parametrize("chunk_size", [1, 7, 200])
@pytest.mark.parametrize("stream", [False, True])
def test_forced_named_truncated_json_4663(client, stream, parser, chunk_size):
    client[1].tool_call_parser = parser
    tools = [
        {
            "type": "function",
            "function": {
                "name": "release_probe",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    raw = '<tool_call> {"name": "release_probe", "'
    content, calls, _ = _reply(
        client,
        "chat",
        raw,
        chunks=[raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)],
        stream=stream,
        choice={"type": "function", "function": {"name": "release_probe"}},
        tools=tools,
    )
    assert calls
    assert content == ""
    if stream:
        names = "".join(
            c.get("function", {}).get("name") or "" for chunk in calls for c in chunk
        )
        arguments = "".join(
            c.get("function", {}).get("arguments") or ""
            for chunk in calls
            for c in chunk
        )
    else:
        assert len(calls) == 1
        names = calls[0]["function"]["name"]
        arguments = calls[0]["function"]["arguments"]
    assert names == "release_probe"
    assert json.loads(arguments) == {}


def _clean_content(text, choice=None):
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    class Request:
        tools = TOOLS

        def model_dump(self):
            return {"tools": self.tools, "tool_choice": choice}

    return _parse_tool_calls_with_parser(text, Request())[0]


PARSERS = ["hermes", "qwen3_coder_xml"]
ROUTES = ["chat", "messages", "responses"]
PARTIAL = TRUNCATED + [
    "<tool_call>{",
    "<tool_call>{ \n\t",
    '<tool_call>{"name":"get_weather","arguments":{"city":"</tool_call>',
    '<tool_call>{"name":"get_weather","arguments":{"city":"<tool_call>',
    r'<tool_call>{"name":"get_weather","arguments":{"city":"escaped \" </tool_call>',
    "<tool_call>\n<function",
    '<tool_call>\n{"name',
    '<tool_call>\n{  "',
    '<tool_call>\n{ "n',
    '<tool_call>\n{ "na',
    '<tool_call>\n{ "nam',
    '<tool_call>\n{ "name"',
    '<tool_call>\n{ "name": "get_weather", "arguments":',
]


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("text", PARTIAL)
@pytest.mark.parametrize("narration", ["", "Let me check.\n"])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_buffered_truncated(client, parser, route, text, narration, finish_reason):
    client[1].tool_call_parser = parser
    content, calls, finish = _reply(
        client, route, narration + text, budget=4, finish_reason=finish_reason
    )
    assert content == narration.strip()
    assert not calls
    expected = {
        "length": {
            "chat": "length",
            "messages": "max_tokens",
            "responses": "incomplete",
        },
        "stop": {"chat": "stop", "messages": "end_turn", "responses": "completed"},
    }[finish_reason][route]
    if route == "responses" and finish_reason == "stop" and not narration:
        expected = "failed"  # Existing guard for a completely empty response.
    assert finish == expected


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "text",
    [
        "Use <tool_call> {json} to describe the format.",
        "Use <tool_call> to introduce a call.",
        "<tool_call> then more prose",
        '<tool_call> {"example": "not a call"',
        "Use `<tool_call>` here",
        "Use ``<tool_call>\n<function`` here",
        "```xml\n<tool_call>\n<function=get_weather\n```",
        '~~~xml\n<tool_call>\n{"name\n~~~',
        "<tool_",
    ],
)
def test_prose_and_code_preserved(client, parser, route, text):
    client[1].tool_call_parser = parser
    assert _reply(client, route, text)[0] == text


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "tail", ["", "\n<tool_call>", "\n<tool_call>\n<function", '\n<tool_call>\n{"name']
)
def test_complete_call_then_truncated(client, parser, route, tail):
    client[1].tool_call_parser = parser
    content, calls, _ = _reply(client, route, COMPLETE + tail)
    assert content == ""
    assert len(calls) == 1
    call = calls[0]
    assert (call.get("function", {}).get("name") or call.get("name")) == "get_weather"
    arguments = call.get("function", {}).get(
        "arguments", call.get("arguments", call.get("input"))
    )
    assert (json.loads(arguments) if isinstance(arguments, str) else arguments) == {
        "city": "Paris"
    }


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("text", ["<tool_call>\n<function", '<tool_call>\n{"name'])
def test_reasoning_fallback(client, parser, route, text):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    assert _reply(client, route, text)[0] == ""


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup(client, parser, record_property):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    raw = "Use <tool_call> for a call.\n" * 800
    start = perf_counter()
    assert cls.strip_trailing_tool_envelope(raw) == raw
    elapsed = perf_counter() - start
    record_property("cleanup_seconds", elapsed)
    assert elapsed < 0.1


@pytest.mark.parametrize("parser", PARSERS)
def test_undeclared_xml_policy(client, parser):
    client[1].tool_call_parser = parser
    raw = "<tool_call>\n<function=unknown>\n<parameter=city>Paris"
    expected = raw if parser == "qwen3_coder_xml" else ""
    assert _reply(client, "chat", raw)[0] == expected
    assert _reply(client, "chat", raw, stream=True)[0] == expected


@pytest.mark.parametrize("parser", PARSERS)
def test_cleanup_guards(client, parser):
    client[1].tool_call_parser = parser
    assert _clean_content("<tool_call>", choice="none") == "<tool_call>"
    assert _reply(client, "chat", "<tool_call>", tools=[])[0] == "<tool_call>"
    assert _clean_content("plain text") == "plain text"
    assert _clean_content("<tool_call>{}</tool_call>") == "<tool_call>{}</tool_call>"
    assert _clean_content("```xml\n<tool_call>") == "```xml\n<tool_call>"
    assert (
        _clean_content("```xml\n<tool_call>\n```\n<tool_call>")
        == "```xml\n<tool_call>\n```\n"
    )


@pytest.mark.parametrize(
    "parser,enabled",
    [("hermes", False), ("missing-parser", True), (None, True), ("qwen", True)],
)
def test_configuration_guard(client, parser, enabled):
    client[1].tool_call_parser = parser
    client[1].enable_auto_tool_choice = enabled
    assert _clean_content("<tool_call>") == "<tool_call>"


def test_structured_content_authoritative(client):
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    class Request:
        tools = TOOLS

        def model_dump(self):
            return {"tools": self.tools}

    content, calls = _parse_tool_calls_with_parser(
        "<tool_call>",
        Request(),
        structured_tool_calls=[{"name": "get_weather", "arguments": "{}"}],
    )
    assert content == "<tool_call>"
    assert len(calls) == 1


def _stream_snapshots():
    """Review reproducers at the shared streaming seam, including EOF events."""
    from dataclasses import asdict

    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.service.postprocessor import StreamingPostProcessor
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager
    from tests.test_postprocessor import _make_cfg, _make_output

    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k != "id"}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    reproducers = [
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>\n<tool_call>\n',
        "Use <tool_call> {json} to describe the format.",
        "Use the <tool_call> tag.\nLet me check.\n<tool_call>\n<function=get_weather",
        "<think>Let me think.</think><tool_call>\n<function=get_weather",
        "<think>I should use <tool_call>\n<function=get_weather> to look this up.</think>Final answer.",
        "Let me check.\n<tool_call>\n<function=get_weather>\n",
        "Hello\n<tool_call>\n<function=get_weather>\n<parameter=city>P",
        "<tool_call>\n<function",
        '<tool_call>\n{"name',
        "<tool_call>\n<function=unknown>\n<parameter=city>Paris",
    ]
    rows = []
    for parser_name in PARSERS:
        for raw in reproducers:
            # Fragmented XML leakage for the Qwen parser is a separate issue.
            sizes = [len(raw)]
            if parser_name == "hermes" and raw in (
                reproducers[0],
                reproducers[5],
                reproducers[6],
                reproducers[9],
            ):
                sizes.append(1)
            for size in sizes:
                parser = ToolParserManager.get_tool_parser(parser_name)(None)
                processor = StreamingPostProcessor(
                    _make_cfg(
                        enable_auto_tool_choice=True,
                        tool_call_parser=parser_name,
                        tool_parser_instance=parser,
                        reasoning_parser=Qwen3ReasoningParser(None)
                        if "<think>" in raw
                        else None,
                    ),
                    tools_requested=True,
                    request={"tools": TOOLS, "tool_choice": "auto"},
                )
                processor.reset()
                events = []
                for i in range(0, len(raw), size):
                    events.extend(
                        processor.process_chunk(_make_output(raw[i : i + size]))
                    )
                events.extend(processor.finalize())
                rows.append(stable([asdict(event) for event in events]))
    return rows


def test_streaming_unchanged_against_base():
    """All streaming definitions are identical; compare review events as well."""
    import ast
    import os
    import subprocess
    import sys
    from pathlib import Path

    import rapid_mlx

    root = Path(rapid_mlx.__file__).resolve().parents[1]
    for name in ("hermes_tool_parser", "qwen3coder_tool_parser"):
        relative = f"rapid_mlx/tool_parsers/{name}.py"
        base = subprocess.check_output(
            ["git", "show", f"origin/main:{relative}"], cwd=root, text=True
        )

        def existing_definitions(source):
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    node.body = [
                        method
                        for method in node.body
                        if not (
                            isinstance(method, ast.FunctionDef)
                            and method.name == "strip_trailing_tool_envelope"
                        )
                    ]
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "_strip_trailing_tool_envelope"
                )
            ]
            return ast.dump(tree, include_attributes=False)

        assert existing_definitions(
            (root / relative).read_text()
        ) == existing_definitions(base)

    # Load the same test helper with base production modules on PYTHONPATH.
    base_root = os.environ.get("TRUNCATED_TOOL_BASE")
    if base_root is None:
        return  # Definitions prove parity in CI; opt in to cross-worktree events.
    script = "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_stream_snapshots']()))"
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(Path(__file__).resolve())],
        cwd=base_root,
        env={**os.environ, "PYTHONPATH": base_root, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
    )
    assert _stream_snapshots() == json.loads(output)


def test_xml_without_declared_tools(client):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    assert (
        Qwen3CoderToolParser.strip_trailing_tool_envelope("<tool_call>", {"tools": []})
        == "<tool_call>"
    )


@pytest.mark.parametrize("route", ROUTES)
def test_hermes_complete_json(client, route):
    client[1].tool_call_parser = "hermes"
    raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    content, calls, status = _reply(client, route, raw, finish_reason="stop")
    assert content == ""
    assert len(calls) == 1
    assert (
        status
        == {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}[
            route
        ]
    )


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup_closed_wrappers(client, parser, record_property):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    raw = ("long narration " * 6 + "<tool_call>{}</tool_call>") * 8600 + "<tool_call>"
    start = perf_counter()
    assert cls.strip_trailing_tool_envelope(raw) == raw[: -len("<tool_call>")]
    elapsed = perf_counter() - start
    record_property("closed_wrappers_seconds", elapsed)
    assert elapsed < 0.15


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
def test_prose_opener_before_real_call(client, parser, route):
    client[1].tool_call_parser = parser
    narration = "Use the <tool_call> tag.\nLet me check.\n"
    raw = narration + "<tool_call>\n<function=get_weather"
    expected = _reply(client, route, narration)[0]
    assert _reply(client, route, raw)[0] == expected


@pytest.mark.parametrize("parser", PARSERS)
def test_exact_whitespace_streaming_parity(client, parser):
    client[1].tool_call_parser = parser
    narration = "Let me check.\n"
    raw = narration + "<tool_call>\n<function=get_weather"
    buffered = _reply(client, "chat", raw)[0]
    streamed = _reply(client, "chat", raw, stream=True)[0]
    # Buffered routes normalize ordinary text too. Check exact parser parity
    # before that normalization, and the route's ordinary-text policy after it.
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cleaned = ToolParserManager.get_tool_parser(parser).strip_trailing_tool_envelope(
        raw
    )
    if parser == "qwen3_coder_xml":
        assert cleaned == streamed == narration
    else:
        assert (
            cleaned
            == _reply(client, "chat", raw, stream=True, chunks=list(raw))[0]
            == narration
        )
    assert buffered == _reply(client, "chat", narration)[0]


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("inside_reasoning", [False, True])
@pytest.mark.parametrize("rescue_enabled", [False, True])
def test_reasoning_boundaries(
    client, parser, route, inside_reasoning, rescue_enabled, monkeypatch
):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on" if rescue_enabled else "off")
    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    raw = (
        "<think>I should use <tool_call>\n"
        "<function=get_weather> to look this up.</think>Final answer."
        if inside_reasoning
        else "<think>Let me think.</think><tool_call>\n<function=get_weather"
    )
    content = _reply(client, route, raw)[0]
    assert content.endswith("Final answer.") if inside_reasoning else content == ""
    data = _reply(client, route, raw, with_reasoning=True)
    if route == "chat":
        reasoning = data["choices"][0]["message"].get("reasoning_content")
    elif route == "messages":
        reasoning = "".join(
            b["thinking"] for b in data["content"] if b["type"] == "thinking"
        )
    else:
        reasoning = "".join(
            c["text"]
            for b in data["output"]
            if b["type"] == "reasoning"
            for c in b["summary"]
        )
    if not inside_reasoning:
        assert reasoning == "Let me think."
    if route == "chat" and not inside_reasoning and not rescue_enabled:
        assert content == _reply(client, route, raw, stream=True)[0]


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup_prose_then_real_call(client, parser):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    narration = "Use <tool_call> for a call.\n" * 20000
    raw = narration + "<tool_call>\n<function=get_weather"
    start = perf_counter()
    assert cls.strip_trailing_tool_envelope(raw) == narration
    assert perf_counter() - start < 0.2


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("closed", [False, True])
def test_prose_json_string_state(client, parser, route, closed):
    client[1].tool_call_parser = parser
    raw = '<tool_call>{"example":"<tool_call>\n<function=get_weather'
    if closed:
        raw += '"}</tool_call>'
    assert _reply(client, route, raw)[0] == raw


@pytest.mark.parametrize("parser", PARSERS)
def test_reasoning_boundary_guard_without_source(client, parser):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.service.helpers import _strip_trailing_tool_envelope

    client[1].tool_call_parser = parser
    client[1].reasoning_parser = Qwen3ReasoningParser(None)
    raw = "<tool_call>\n<function=get_weather</think>Final answer."
    assert _strip_trailing_tool_envelope(raw) == raw
    assert _strip_trailing_tool_envelope("<tool_call>\n<function=get_weather") == ""
