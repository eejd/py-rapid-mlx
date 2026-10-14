# SPDX-License-Identifier: Apache-2.0
"""A guard stop keeps truncation handling internal and reports its actual cause."""

import asyncio

from rapid_mlx.api.anthropic_adapter import (
    _convert_stop_reason,
    _thinking_block_content,
)
from rapid_mlx.api.constants import is_rescue_payload
from rapid_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from rapid_mlx.api.protocol_mapping import public_finish_reason
from rapid_mlx.api.responses_adapter import _convert_status
from rapid_mlx.service.helpers import _apply_reasoning_cutoff_notice


def test_only_repetition_abort_changes_public_finish_reason():
    assert public_finish_reason("length", "repetition") == "repetition"
    assert public_finish_reason("length", None) == "length"
    assert public_finish_reason("tool_calls", "repetition") == "repetition"


def test_other_protocols_keep_interrupted_turn_distinct_from_normal_stop():
    assert _convert_stop_reason("repetition") == "pause_turn"
    assert _convert_status("repetition") == "incomplete"
    assert (
        _thinking_block_content(
            "partial",
            "[truncated — generation stopped after repetition]\n\npartial",
            "repetition",
        )
        is None
    )


def test_reasoning_cutoff_notice_names_repetition_instead_of_token_budget():
    for include_tail, reasoning_text in (
        (False, "partial thought"),
        (True, "partial thought"),
        (True, "<think>"),
    ):
        notice = _apply_reasoning_cutoff_notice(
            None,
            reasoning_text,
            None,
            "length",
            include_reasoning_tail=include_tail,
            error_kind="repetition",
        )
        assert notice.startswith("[truncated — generation stopped after repetition]")
        assert "max_tokens" not in notice
        assert is_rescue_payload(notice)


def test_buffered_chat_replay_restores_internal_repetition_cause(monkeypatch):
    from rapid_mlx.routes import chat as route

    observed = []

    async def fake_stream(engine, *_args, **_kwargs):
        async for output in engine.stream_chat():
            observed.append(output)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(route, "stream_chat_completion", fake_stream)
    response = ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content="partial"),
                finish_reason="repetition",
            )
        ],
        usage=Usage(prompt_tokens=2, completion_tokens=80, total_tokens=82),
    )
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hello"}], stream=True
    )

    async def collect():
        return [
            chunk
            async for chunk in route._stream_buffered_chat_response(response, request)
        ]

    assert asyncio.run(collect()) == ["data: [DONE]\n\n"]
    assert observed[-1].finish_reason == "length"
    assert observed[-1].error_kind == "repetition"
