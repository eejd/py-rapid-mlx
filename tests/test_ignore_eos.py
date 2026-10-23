# SPDX-License-Identifier: Apache-2.0
"""``ignore_eos`` on the public routes (fixed-output-length load tests).

A client such as guidellm sends ``"ignore_eos": true`` with ``max_tokens`` and
requires ``usage.completion_tokens == max_tokens``. The engine already honoured
``SamplingParams.ignore_eos``; these tests pin the pieces that connect it to
the wire and the behaviour of every scheduler lane that receives it:

* request schemas (chat, completions, responses): strict boolean;
* the shared sampling-kwargs builder and the three routes: forwarded only
  when ``true``; absent / ``false`` / ``null`` leave the engine default;
* text scheduler on a tiny real model: continues past EOS to ``max_tokens``,
  adds no EOS text, still stops on caller ``stop_token_ids``, mixed batches;
* multimodal scheduler: an ignored EOS is a counted token with no text;
* serial speculative servers that cannot suppress EOS refuse the field.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rapid_mlx.api.models import ChatCompletionRequest, CompletionRequest
from rapid_mlx.api.responses_adapter import responses_to_openai
from rapid_mlx.api.responses_models import ResponsesRequest
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.service.helpers import build_extended_sampling_kwargs
from tests.test_lane_parity_contract import _route_client, _RouteRecordingEngine

# ------------------------------------------------------------------ schemas


def _chat(**kw: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kw)


def _completion(**kw: Any) -> CompletionRequest:
    return CompletionRequest(prompt="hi", **kw)


def _responses(**kw: Any) -> ResponsesRequest:
    return ResponsesRequest(model="m", input="hi", **kw)


SCHEMAS = [_chat, _completion, _responses]


@pytest.mark.parametrize("make", SCHEMAS, ids=["chat", "completions", "responses"])
def test_schema_default_is_absent(make) -> None:
    assert make().ignore_eos is None


@pytest.mark.parametrize("make", SCHEMAS, ids=["chat", "completions", "responses"])
@pytest.mark.parametrize("value", [True, False, None])
def test_schema_accepts_json_boolean_and_null(make, value) -> None:
    assert make(ignore_eos=value).ignore_eos is value


@pytest.mark.parametrize("make", SCHEMAS, ids=["chat", "completions", "responses"])
@pytest.mark.parametrize("value", ["yes", "true", "false", "", 0, 1, [True], {}])
def test_schema_rejects_everything_that_is_not_a_boolean(make, value) -> None:
    """Pydantic's lax bool would read ``"false"`` as False and ``1`` as True."""
    with pytest.raises(ValidationError, match="ignore_eos must be a boolean"):
        make(ignore_eos=value)


# ------------------------------------------------- sampling kwargs + adapter


def test_sampling_kwargs_forward_only_true() -> None:
    assert build_extended_sampling_kwargs(_chat(ignore_eos=True))["ignore_eos"] is True
    for req in (_chat(), _chat(ignore_eos=False), _chat(ignore_eos=None)):
        assert "ignore_eos" not in build_extended_sampling_kwargs(req)
    assert (
        build_extended_sampling_kwargs(_completion(ignore_eos=True))["ignore_eos"]
        is True
    )
    assert "ignore_eos" not in build_extended_sampling_kwargs(_completion())


def test_responses_request_reaches_the_shared_kwargs_builder() -> None:
    openai_request = responses_to_openai(_responses(ignore_eos=True))
    assert openai_request.ignore_eos is True
    assert build_extended_sampling_kwargs(openai_request)["ignore_eos"] is True
    assert responses_to_openai(_responses()).ignore_eos is None


# ------------------------------------------------------------------- routes


@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("is_mllm", [False, True], ids=["text", "multimodal"])
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize(
    "sent,forwarded",
    [({"ignore_eos": True}, True), ({"ignore_eos": False}, None), ({}, None)],
    ids=["true", "false", "absent"],
)
def test_chat_and_responses_routes_forward_ignore_eos(
    surface, is_mllm, stream, sent, forwarded
) -> None:
    engine = _RouteRecordingEngine(is_mllm=is_mllm)
    client = _route_client(surface, engine)
    if surface == "chat":
        path = "/v1/chat/completions"
        body = {
            "model": "lane-parity-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 23,
            "stream": stream,
            **sent,
        }
    else:
        path = "/v1/responses"
        body = {
            "model": "lane-parity-test",
            "input": "hello",
            "max_output_tokens": 23,
            "stream": stream,
            **sent,
        }
    try:
        if stream:
            with client.stream("POST", path, json=body) as response:
                assert response.status_code == 200, response.read()
                list(response.iter_lines())
        else:
            response = client.post(path, json=body)
            assert response.status_code == 200, response.text
        _method, kwargs = engine.calls[-1]
        assert kwargs["max_tokens"] == 23
        assert kwargs.get("ignore_eos") is forwarded
        if forwarded is None:
            assert "ignore_eos" not in kwargs
    finally:
        reset_config()


def test_chat_route_rejects_non_boolean_ignore_eos_with_http_error() -> None:
    engine = _RouteRecordingEngine(is_mllm=False)
    client = _route_client("chat", engine)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "lane-parity-test",
                "messages": [{"role": "user", "content": "hello"}],
                "ignore_eos": "false",
            },
        )
        assert response.status_code == 400
        assert "ignore_eos" in json.dumps(response.json())
        assert engine.calls == []
    finally:
        reset_config()


class _CompletionsEngine:
    """Records the kwargs the completions route hands to the engine."""

    supports_completion_logprobs = False
    is_mllm = False
    tokenizer = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _out() -> GenerationOutput:
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="length",
        )

    async def generate(self, *_args, **kwargs):
        self.calls.append(("generate", kwargs))
        return self._out()

    async def stream_generate(self, *_args, **kwargs):
        self.calls.append(("stream", kwargs))
        yield self._out()


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize(
    "sent,forwarded",
    [({"ignore_eos": True}, True), ({"ignore_eos": False}, None), ({}, None)],
    ids=["true", "false", "absent"],
)
def test_completions_route_forwards_ignore_eos(
    stream, sent, forwarded, monkeypatch
) -> None:
    from rapid_mlx.routes import completions as route

    cfg = reset_config()
    engine = _CompletionsEngine()
    cfg.engine = engine
    cfg.model_name = "stub-model"
    cfg.model_registry = None
    cfg.reasoning_parser = None
    cfg.tool_call_parser = None
    monkeypatch.setattr(route, "get_engine", lambda *_a, **_kw: engine)
    monkeypatch.setattr(
        route, "enforce_context_length_for_prompt", lambda *_a, **_kw: None
    )
    app = FastAPI()
    app.include_router(route.router)
    client = TestClient(app, raise_server_exceptions=False)
    body = {
        "model": "stub-model",
        "prompt": "hi",
        "max_tokens": 9,
        "stream": stream,
        **sent,
    }
    try:
        if stream:
            with client.stream("POST", "/v1/completions", json=body) as response:
                assert response.status_code == 200, response.read()
                list(response.iter_lines())
        else:
            response = client.post("/v1/completions", json=body)
            assert response.status_code == 200, response.text
        _method, kwargs = engine.calls[-1]
        assert kwargs.get("ignore_eos") is forwarded
        if forwarded is None:
            assert "ignore_eos" not in kwargs
    finally:
        reset_config()


# ----------------------------------------------- text scheduler, real model


def _tiny_model_and_eos():
    """A random 2-layer llama and the token it greedily predicts first.

    Using that token as the EOS makes "ends at the first token" the
    unambiguous default behaviour, and the model then keeps emitting it:
    the exact-length run is visible in ``output_token_ids``.
    """
    import mlx.core as mx
    from mlx_lm.models import llama

    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    first = int(mx.argmax(model(mx.array([_PROMPT]))[:, -1, :], axis=-1).item())
    return model, first


_PROMPT = [1, 2, 3, 4]


class _Tokenizer:
    def __init__(self, eos: int) -> None:
        self.eos_token_id = eos
        self._eos_token_ids = {eos}

    def encode(self, _text, **_kw):
        return list(_PROMPT)

    def decode(self, ids, **_kw):
        return "".join(f"<{i}>" for i in ids)


def _run_scheduler(model, eos, requests, *, max_steps=200):
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    scheduler = Scheduler(model, _Tokenizer(eos), SchedulerConfig(max_num_seqs=4))
    for request in requests:
        request.prompt_token_ids = list(_PROMPT)
        request.num_prompt_tokens = len(_PROMPT)
        scheduler.add_request(request)
    final: dict[str, Any] = {}
    deltas: dict[str, list[str]] = {r.request_id: [] for r in requests}
    for _ in range(max_steps):
        step = scheduler.step()
        for out in step.outputs:
            deltas[out.request_id].append(out.new_text)
            if out.finished:
                final[out.request_id] = out
        if len(final) == len(requests):
            break
    return final, deltas


def _request(rid: str, **params: Any):
    from rapid_mlx.request import Request, SamplingParams

    return Request(
        request_id=rid,
        prompt=_PROMPT,
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0, **params),
    )


def test_text_scheduler_default_ends_at_eos_and_ignore_eos_runs_to_max_tokens() -> None:
    model, eos = _tiny_model_and_eos()

    final, _ = _run_scheduler(model, eos, [_request("default")])
    out = final["default"]
    assert out.finish_reason == "stop"
    assert out.completion_tokens == 1

    final, _ = _run_scheduler(model, eos, [_request("ignored", ignore_eos=True)])
    out = final["ignored"]
    assert out.finish_reason == "length"
    assert out.completion_tokens == 8
    assert out.output_token_ids == [eos] * 8


def test_text_scheduler_ignored_eos_adds_no_text() -> None:
    model, eos = _tiny_model_and_eos()
    final, deltas = _run_scheduler(model, eos, [_request("r", ignore_eos=True)])
    assert final["r"].output_text == ""
    assert "".join(deltas["r"]) == ""


def test_text_scheduler_caller_stop_token_ids_still_stop_under_ignore_eos() -> None:
    """Caller intent survives; only the model's own terminators are ignored."""
    model, eos = _tiny_model_and_eos()
    final, _ = _run_scheduler(
        model, eos, [_request("r", ignore_eos=True, stop_token_ids=[eos])]
    )
    assert final["r"].finish_reason == "stop"
    assert final["r"].completion_tokens == 1


def test_text_scheduler_mixed_batch_each_request_keeps_its_own_setting() -> None:
    model, eos = _tiny_model_and_eos()
    final, _ = _run_scheduler(
        model,
        eos,
        [
            _request("a-default"),
            _request("b-ignored", ignore_eos=True),
            _request("c-default"),
            _request("d-ignored", ignore_eos=True),
        ],
    )
    assert {k: (v.finish_reason, v.completion_tokens) for k, v in final.items()} == {
        "a-default": ("stop", 1),
        "b-ignored": ("length", 8),
        "c-default": ("stop", 1),
        "d-ignored": ("length", 8),
    }


def _bare_scheduler(eos: int):
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    scheduler = Scheduler(
        MagicMock(), _Tokenizer(eos), SchedulerConfig(max_num_seqs=2)
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    return scheduler


def _response(token: int):
    response = MagicMock(uid=0, token=token, finish_reason=None, logprobs=None)
    del response.prompt_cache
    return response


def test_text_scheduler_ignored_eos_is_not_fed_to_the_decoder_other_tokens_are() -> (
    None
):
    from rapid_mlx.request import Request, RequestStatus, SamplingParams

    eos = 7
    scheduler = _bare_scheduler(eos)
    decoder = MagicMock()
    decoder.add_token = lambda t: f"<{t}>"
    decoder.get_full_text = lambda: ""
    req = Request("r", "p", SamplingParams(max_tokens=50, ignore_eos=True))
    req.status = RequestStatus.RUNNING
    req._decoder = decoder
    scheduler.running["r"] = req
    scheduler.uid_to_request_id[0] = "r"

    (eos_out,), _ = scheduler._process_batch_responses([_response(eos)])
    (other_out,), _ = scheduler._process_batch_responses([_response(9)])

    assert eos_out.new_text == ""
    assert eos_out.new_token_ids == [eos]
    assert other_out.new_text == "<9>"
    assert req.output_token_ids == [eos, 9]


def test_text_scheduler_repetition_abort_is_off_under_ignore_eos() -> None:
    """A fixed-length run must not be cut short by the tool-loop heuristic."""
    from rapid_mlx.request import Request, RequestStatus, SamplingParams

    for ignore_eos, expected_finish in ((False, "abort"), (True, None)):
        scheduler = _bare_scheduler(eos=99)
        pattern = list(range(12))
        req = Request(
            "r", "p", SamplingParams(max_tokens=1000, ignore_eos=ignore_eos)
        )
        req.status = RequestStatus.RUNNING
        req.has_tools = True
        req.output_token_ids = pattern * 5 + pattern[:-1]
        scheduler.running["r"] = req
        scheduler.uid_to_request_id[0] = "r"
        (out,), _ = scheduler._process_batch_responses([_response(pattern[-1])])
        assert out.finish_reason == expected_finish


# ------------------------------------------------------ multimodal scheduler


def _mllm_scheduler_with_request(*, ignore_eos: bool, eos: int):
    from rapid_mlx.mllm_scheduler import (
        MLLMRequest,
        MLLMScheduler,
        MLLMSchedulerConfig,
    )
    from rapid_mlx.request import SamplingParams

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    scheduler = MLLMScheduler(MagicMock(), processor, MLLMSchedulerConfig())
    scheduler.stop_tokens = {eos}

    class Detok:
        last_segment = ""
        text = ""

        def __init__(self) -> None:
            self.fed: list[int] = []

        def reset(self) -> None:
            pass

        def add_token(self, token: int) -> None:
            self.fed.append(token)
            self.last_segment = f"<{token}>"
            self.text += self.last_segment

        def finalize(self) -> None:
            pass

    detok = Detok()
    request = MLLMRequest(
        request_id="m",
        prompt="p",
        sampling_params=SamplingParams(max_tokens=10, ignore_eos=ignore_eos),
    )
    scheduler.running["m"] = request
    scheduler.uid_to_request_id[0] = "m"
    scheduler._detokenizer_pool["m"] = detok
    return scheduler, request, detok


def _mllm_response(token: int):
    import mlx.core as mx

    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse

    return MLLMBatchResponse(
        uid=0,
        request_id="m",
        token=token,
        logprobs=mx.array([0.1]),
        finish_reason=None,
        token_is_stop_token=False,  # what the batch generator reports under ignore_eos
    )


def test_mllm_scheduler_ignored_eos_is_counted_but_adds_no_text() -> None:
    eos = 7
    scheduler, request, detok = _mllm_scheduler_with_request(ignore_eos=True, eos=eos)

    (eos_out,), _ = scheduler._process_batch_responses([_mllm_response(eos)])
    (other_out,), _ = scheduler._process_batch_responses([_mllm_response(9)])

    assert eos_out.new_text == ""
    assert eos_out.new_token_ids == [eos]
    assert other_out.new_text == "<9>"
    assert detok.fed == [9]
    assert request.output_tokens == [eos, 9]


def test_mllm_scheduler_without_ignore_eos_decodes_every_non_control_token() -> None:
    """Unchanged default: a token the generator did not flag is plain text."""
    eos = 7
    scheduler, _request, detok = _mllm_scheduler_with_request(
        ignore_eos=False, eos=eos
    )
    (out,), _ = scheduler._process_batch_responses([_mllm_response(eos)])
    assert out.new_text == f"<{eos}>"
    assert detok.fed == [eos]


# ----------------------------------------- serial speculative servers refuse


def test_dflash_family_server_refuses_ignore_eos() -> None:
    from rapid_mlx.speculative.dflash.runtime import DFlashRuntime
    from rapid_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
    )
    client = TestClient(app, base_url="http://127.0.0.1")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "ignore_eos": True,
        },
    )
    assert response.status_code == 400
    assert "ignore_eos" in json.dumps(response.json())


def test_ddtree_refuses_ignore_eos_true_only() -> None:
    from rapid_mlx.speculative.ddtree.server import _validate_request

    _validate_request(_chat())
    _validate_request(_chat(ignore_eos=False))
    with pytest.raises(HTTPException, match="ignore_eos"):
        _validate_request(_chat(ignore_eos=True))


def test_namespace_without_the_field_is_treated_as_absent() -> None:
    """Request shapes that predate the field (Anthropic adapter, tests)."""
    assert "ignore_eos" not in build_extended_sampling_kwargs(SimpleNamespace())
