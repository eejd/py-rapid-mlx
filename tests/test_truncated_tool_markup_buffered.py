# SPDX-License-Identifier: Apache-2.0
"""Buffered tool markup must obey the same visibility rules as streaming."""

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


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
@pytest.mark.parametrize("text", TRUNCATED)
@pytest.mark.parametrize("narration", ["", "Let me check the weather.\n"])
@pytest.mark.parametrize("budget", [1, 4, 16])
def test_truncated_envelope(client, route, text, narration, budget):
    content, calls, finish = _reply(client, route, narration + text, budget=budget)
    assert content == narration.rstrip()
    assert not calls
    assert (
        finish
        == {"chat": "length", "messages": "max_tokens", "responses": "incomplete"}[
            route
        ]
    )


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
@pytest.mark.parametrize(
    "text",
    [
        "```xml\n<tool_call>\n<function=get_weather\n```",
        "use the `<tool_call>` tag",
        "Use <tool_call> to introduce a call, then continue writing prose.",
    ],
)
def test_documentation_preserved(client, route, text):
    assert _reply(client, route, text)[0] == text


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
def test_complete_call_preserved(client, route):
    content, calls, _ = _reply(client, route, COMPLETE, finish_reason="stop")
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


@pytest.mark.parametrize("text", TRUNCATED + ["<tool_"])
def test_chat_stream_parity(client, text):
    buffered = _reply(client, "chat", text)
    streamed = _reply(client, "chat", text, stream=True)
    assert buffered[0] == streamed[0]
    assert buffered[2] == streamed[2] == "length"


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


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
def test_partial_prefix_released_at_eof(client, route):
    assert _reply(client, route, "<tool_")[0] == "<tool_"


@pytest.mark.parametrize("text", TRUNCATED)
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_reasoning_finalizer_cannot_restore_markup(client, text, finish_reason):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = client[1]
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    assert _reply(client, "chat", text, finish_reason=finish_reason)[0] == ""


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call> then more prose",
        "```xml\n<tool_call>",
        "use the ``<tool_call>`` tag",
        "<tool_call>{} </tool_call>\n<tool_call>",
        "```xml\n<tool_call>\n```\n<tool_call>",
    ],
)
def test_shared_cleanup_boundaries(client, text):
    expected = {
        "<tool_call>{} </tool_call>\n<tool_call>": "<tool_call>{} </tool_call>",
        "```xml\n<tool_call>\n```\n<tool_call>": "```xml\n<tool_call>\n```",
    }.get(text, text)
    assert _clean_content(text) == expected


@pytest.mark.parametrize(
    "parser,enabled",
    [("qwen3_coder_xml", False), ("missing-parser", True), (None, True)],
)
def test_cleanup_respects_configuration(client, parser, enabled):

    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.enable_auto_tool_choice = enabled
    assert _clean_content(TRUNCATED[0]) == TRUNCATED[0]


def test_no_tools_preserves_text(client):
    assert _reply(client, "chat", TRUNCATED[0], tools=[])[0] == TRUNCATED[0]


@pytest.mark.parametrize("text", TRUNCATED)
def test_chat_narration_stream_parity(client, text):
    raw = "Let me check the weather.\n" + text
    buffered = _reply(client, "chat", raw)
    streamed = _reply(
        client,
        "chat",
        raw,
        stream=True,
        chunks=["Let me check the weather.\n", text],
    )
    assert buffered[0] == streamed[0].rstrip() == "Let me check the weather."
    assert buffered[2] == streamed[2] == "length"


def test_complete_call_before_truncated_envelope(client):
    content, calls, _ = _reply(client, "chat", COMPLETE + "\n<tool_call>")
    assert content == ""
    assert len(calls) == 1


def _clean_content(text, choice=None):
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    class Request:
        tools = TOOLS

        def model_dump(self):
            return {"tools": self.tools, "tool_choice": choice}

    return _parse_tool_calls_with_parser(text, Request())[0]


@pytest.mark.parametrize("text", ["<tool_call>\n<fun", "<tool_call>\n<function="])
@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
def test_partial_function_header(client, text, route):
    assert _reply(client, route, text)[0] == ""


@pytest.mark.parametrize(
    "text",
    [
        "Use <tool_call> {json} to describe the format.",
        '<tool_call> {"example": "not a call"',
    ],
)
def test_json_prose_preserved(client, text):
    assert _reply(client, "chat", text)[0] == text


def test_none_choice_preserves_unfinished_example(client):
    assert _clean_content(TRUNCATED[0], choice="none") == TRUNCATED[0]
    assert _reply(client, "chat", TRUNCATED[0], choice="none")[0] == TRUNCATED[0]


def test_structured_content_remains_authoritative(client):
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


HERMES_TRUNCATED = TRUNCATED + ['<tool_call>\n{"name": "get_weather", "argu']
HERMES_JSON = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
@pytest.mark.parametrize("text", HERMES_TRUNCATED)
@pytest.mark.parametrize("narration", ["", "Let me check the weather.\n"])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_hermes_truncated_envelope(client, route, text, narration, finish_reason):
    client[1].tool_call_parser = "hermes"
    content, calls, finish = _reply(
        client, route, narration + text, finish_reason=finish_reason, budget=1
    )
    assert content == narration.rstrip()
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
        # Existing empty-response guard applies after markup is hidden.
        expected = "failed"
    assert finish == expected


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
@pytest.mark.parametrize("text", HERMES_TRUNCATED)
@pytest.mark.parametrize(
    "template", ["```xml\n{}\n```", "Use `{}` here", "Use ``{}`` here"]
)
def test_hermes_examples_preserved(client, route, text, template):
    client[1].tool_call_parser = "hermes"
    raw = template.format(text)
    assert _reply(client, route, raw)[0] == raw


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
@pytest.mark.parametrize("text", [COMPLETE, HERMES_JSON])
def test_hermes_complete_call_preserved(client, route, text):
    client[1].tool_call_parser = "hermes"
    content, calls, finish = _reply(client, route, text, finish_reason="stop")
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
    assert (
        finish
        == {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}[
            route
        ]
    )


@pytest.mark.parametrize(
    "text,tail",
    [(t, "") for t in HERMES_TRUNCATED]
    + [
        ("<tool_", "<tool_"),
        ("<tool_call>\n<fun", "<fun"),
        ("<tool_call>\n<function", "<function"),
        ("<tool_call>\n<", "<"),
    ],
)
@pytest.mark.parametrize("narration", ["", "Let me check the weather.\n"])
@pytest.mark.parametrize("chunk_size", [1, 200])
def test_hermes_chat_stream_parity(client, text, tail, narration, chunk_size):
    client[1].tool_call_parser = "hermes"
    raw = narration + text
    buffered = _reply(client, "chat", raw)
    streamed = _reply(
        client,
        "chat",
        raw,
        stream=True,
        chunks=[raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)],
    )
    assert buffered[0] == streamed[0].rstrip() == (narration + tail).rstrip()
    assert not buffered[1] and not streamed[1]
    assert buffered[2] == streamed[2] == "length"


@pytest.mark.parametrize(
    "text,tail",
    [(t, "") for t in HERMES_TRUNCATED]
    + [
        ("<tool_", "<tool_"),
        ("<tool_call>\n<fun", "<fun"),
    ],
)
@pytest.mark.parametrize("narration", ["", "Let me check the weather.\n"])
@pytest.mark.parametrize("chunk_size", [1, 200])
def test_hermes_postprocessor_eof(client, text, tail, narration, chunk_size):
    from rapid_mlx.service.postprocessor import StreamingPostProcessor
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
    from tests.test_postprocessor import _make_cfg, _make_output

    client[1].tool_call_parser = "hermes"
    raw = narration + text
    parser = HermesToolParser(None)
    processor = StreamingPostProcessor(
        _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=parser,
        ),
        tools_requested=True,
        request={"tools": TOOLS, "tool_choice": "auto"},
    )
    processor.reset()
    events = []
    for i in range(0, len(raw), chunk_size):
        events.extend(processor.process_chunk(_make_output(raw[i : i + chunk_size])))
    assert parser.flush_held_content(processor.tool_accumulated_text) == tail
    events.extend(processor.finalize())
    content = "".join(e.content or "" for e in events)
    assert content == _clean_content(raw) == narration + tail
    assert not any(e.type == "tool_call" for e in events)


@pytest.mark.parametrize("text", HERMES_TRUNCATED)
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_hermes_reasoning_fallback(client, text, finish_reason):
    client[1].tool_call_parser = "hermes"
    test_reasoning_finalizer_cannot_restore_markup(client, text, finish_reason)


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call> then more prose",
        "```xml\n<tool_call>",
        "<tool_call>{} </tool_call>\n<tool_call>",
        "```xml\n<tool_call>\n```\n<tool_call>",
    ],
)
def test_hermes_hook_boundaries(client, text):
    client[1].tool_call_parser = "hermes"
    expected = (
        text.rsplit("<tool_call>", 1)[0]
        if text
        in {
            "<tool_call>{} </tool_call>\n<tool_call>",
            "```xml\n<tool_call>\n```\n<tool_call>",
        }
        else text
    )
    assert _clean_content(text) == expected


def test_hermes_none_choice_preserves_unfinished_example(client):
    client[1].tool_call_parser = "hermes"
    test_none_choice_preserves_unfinished_example(client)


def test_unfinished_undeclared_xml_is_released(client):
    raw = "<tool_call>\n<function=unknown>\n<parameter=city>Paris"
    buffered = _reply(client, "chat", raw)
    streamed = _reply(client, "chat", raw, stream=True)
    assert buffered[0] == streamed[0] == raw
    assert not buffered[1] and not streamed[1]


def test_unrelated_parser_hook_is_noop(client):
    client[1].tool_call_parser = "qwen"
    assert _clean_content(TRUNCATED[0]) == TRUNCATED[0]


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
def test_hermes_unclosed_documentation_preserved(client, route):
    client[1].tool_call_parser = "hermes"
    raw = "Example:\n<tool_call>\n<function=get_weather>"
    assert _reply(client, route, raw)[0] == raw


@pytest.mark.parametrize("chunk_size", [1, 200])
def test_hermes_header_example_eof_parity(client, chunk_size):
    text = "<tool_call>\n<function=get_weather>"
    narration = "Example:\n"
    test_hermes_postprocessor_eof(client, text, text, narration, chunk_size)
    test_hermes_chat_stream_parity(client, text, text, narration, chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 200])
def test_hermes_header_without_narration_stays_hidden(client, chunk_size):
    text = "<tool_call>\n<function=get_weather>"
    test_hermes_postprocessor_eof(client, text, "", "", chunk_size)
    test_hermes_chat_stream_parity(client, text, "", "", chunk_size)
