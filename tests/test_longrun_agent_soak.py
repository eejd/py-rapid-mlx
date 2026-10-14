"""Response diagnostics for the live-server soak without loading a model."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.longrun_agent_soak import Soak


def make_soak(tmp_path):
    args = SimpleNamespace(
        output=tmp_path,
        duration=1,
        seed=1,
        timeout=5,
        concurrency=1,
        model="test",
        url="http://localhost",
    )
    return Soak(args)


async def close_soak(soak):
    await soak.client.aclose()
    soak.errors.close()
    soak.csv_file.close()


def read_error(tmp_path):
    return json.loads((tmp_path / "errors.jsonl").read_text().strip())


def test_blank_response_records_bounded_details(tmp_path):
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": " " * 600,
                    "reasoning_content": "r" * 2100,
                },
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 80},
    }

    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            )
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)
        return soak

    soak = asyncio.run(run())
    error = read_error(tmp_path)
    assert soak.totals["errors"] == 1
    assert error["error"] == "empty answer"
    assert error["finish_reason"] == "length"
    assert error["usage"] == body["usage"]
    assert error["content_length"] == 600
    assert error["reasoning_content"] == "r" * 2000
    assert error["content_repr"] == repr(" " * 600)[:500]
    assert error["request_message_count"] == 2


def test_no_choices_records_missing_response_fields(tmp_path):
    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert error["error"] == "no choices"
    assert error["finish_reason"] is None
    assert error["usage"] is None
    assert error["content_length"] == 0
    assert error["reasoning_content"] == ""
    assert error["content_repr"] == "''"
    assert error["request_message_count"] == 2


@pytest.mark.parametrize(
    ("followup_body", "expected_error"),
    [
        (
            {
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                "usage": {"completion_tokens": 1},
            },
            "empty tool follow-up answer",
        ),
        ({"choices": []}, "tool follow-up has no choices"),
    ],
)
def test_tool_followup_records_actual_request_message_count(
    tmp_path, followup_body, expected_error
):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "lookup_symbol",
                                            "arguments": '{"name":"parse_request"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json=followup_body)

    async def run():
        soak = make_soak(tmp_path)
        soak.sequence = 4
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert calls == 2
    assert error["error"] == expected_error
    assert error["stage"] == "tool_followup"
    assert error["finish_reason"] == (
        followup_body["choices"][0]["finish_reason"]
        if followup_body["choices"]
        else None
    )
    assert error["usage"] == followup_body.get("usage")
    assert error["request_message_count"] == 4


def test_incomplete_stream_records_received_content(tmp_path):
    async def run():
        soak = make_soak(tmp_path)
        soak.sequence = 1
        chunk = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "delta": {"content": "partial", "reasoning_content": "thinking"},
                }
            ],
            "usage": {"completion_tokens": 2},
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=f"data: {json.dumps(chunk)}\n\n"
                )
            )
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert error["error"] == "stream missing completion marker"
    assert error["finish_reason"] == "stop"
    assert error["content_length"] == 7
    assert error["content_repr"] == "'partial'"
    assert error["reasoning_content"] == "thinking"
    assert error["usage"] == {"completion_tokens": 2}


@pytest.mark.parametrize("failure", ["timeout", "http", "other"])
def test_request_exceptions_do_not_add_response_details(tmp_path, failure):
    async def run():
        soak = make_soak(tmp_path)

        async def fail(*args):
            if failure == "timeout":
                raise TimeoutError
            if failure == "http":
                response = httpx.Response(
                    503, request=httpx.Request("POST", soak.args.url)
                )
                raise httpx.HTTPStatusError(
                    "failure", request=response.request, response=response
                )
            raise RuntimeError("failure")

        soak.send = fail
        await soak.request(0)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert "finish_reason" not in error
