"""Model-free and AST tests for PR 9 continuous-MTP routing."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from rapid_mlx.spec_decode.config import (
    SpeculativeConfigError,
    parse_speculative_config,
)
from rapid_mlx.spec_decode.mtp import continuous_routing as routing_module
from rapid_mlx.spec_decode.mtp.batched import (
    AdmissionDecision,
    BatchedMTPRoute,
    SamplingContract,
)
from rapid_mlx.spec_decode.mtp.continuous_routing import (
    ContinuousMTPAPCHit,
    ContinuousMTPIntegrationRoute,
    ContinuousMTPRequestMetadata,
    plan_router_install,
)
from rapid_mlx.spec_decode.mtp.prepared_state import (
    PreparedStateIdentity,
    prepare_mtp_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _descriptor(family="qwen3_5"):
    return MappingProxyType(
        {
            "protocol_version": 1,
            "model_family": family,
            "batch_forward": "mtp_batch_forward",
            "recursive_draft_depth": 2,
            "fixed_membership": True,
            "dynamic_join": True,
            "quantized_cache": False,
            "windowed_cache": False,
            "xtc": False,
        }
    )


class _Model:
    batched_mtp_capability = _descriptor()
    mtp = object()

    def __call__(self, *args, **kwargs):
        return args, kwargs

    def mtp_batch_forward(self, *args, **kwargs):
        return args, kwargs

    def mtp_forward(self, *args, **kwargs):
        return args, kwargs

    def make_mtp_cache(self):
        return object()


class _SchedulerResponse:
    """Strict mlx-lm response twin used to exercise compatibility filling."""

    def __init__(
        self,
        uid,
        token,
        logprobs,
        finish_reason,
        required_bookkeeping,
    ):
        self.uid = uid
        self.token = token
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.required_bookkeeping = required_bookkeeping


class _SchedulerBatchGenerator:
    class _GenerationBatch:
        Response = _SchedulerResponse

    def __init__(self, *, raw_next=([], ()), with_response=True):
        if not with_response:
            self._generation_batch = SimpleNamespace()
        else:
            self._generation_batch = self._GenerationBatch()
        self._unprocessed_sequences = deque()
        self.raw_next = raw_next
        self.closed = False
        self.removed = []

    def next(self):
        return self.raw_next

    def close(self):
        self.closed = True
        return "closed"

    def remove(self, uids, *_args, **_kwargs):
        self.removed.append(tuple(uids))
        return {uid: (f"plain-cache-{uid}", [uid]) for uid in uids}


def _scheduler_sequence(uid, *, caches=None, segments=None, processors=None):
    return (
        uid,
        [[10 + uid, 20 + uid]] if segments is None else segments,
        16,
        [SimpleNamespace(offset=0, nbytes=0)] if caches is None else caches,
        [],
        object(),
        [] if processors is None else processors,
        None,
        None,
    )


def _scheduler_request(*, temperature=0.0, stop=None, has_tools=False):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=temperature, stop=stop),
        has_tools=has_tools,
    )


def _request(lane, uid, **changes):
    values = {
        "lane_id": lane,
        "uid": uid,
        "prompt_tokens": (10 + uid, 20 + uid),
        "max_tokens": 16,
    }
    values.update(changes)
    return ContinuousMTPRequestMetadata(**values)


def _router(**changes):
    values = {"enabled": True, "hard_reserve_bytes": 0}
    values.update(changes)
    decision = plan_router_install(_Model(), **values)
    assert decision.admitted is True
    assert decision.router is not None
    return decision.router


def _identity():
    return PreparedStateIdentity.from_config(
        model_id="Qwen/Qwen3.8-Flash-Next",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )


def _apc_hit(prefix, *, identity=None):
    expected = _identity()
    state = prepare_mtp_state(
        identity=identity or expected,
        prefix_tokens=prefix,
        target_cache="target-cache",
        target_cache_tokens=len(prefix),
        mtp_cache="mtp-cache",
        mtp_cache_pairs=len(prefix) - 1,
        seed_hidden="seed-hidden",
        captured_at=10.0,
    )
    return ContinuousMTPAPCHit(
        state=state,
        expected_identity=expected,
        target_cache_tokens=len(prefix),
        mtp_cache_pairs=len(prefix) - 1,
        now=11.0,
        max_age_seconds=60.0,
    )


def test_speculative_config_preserves_continuous_auto_and_explicit_values():
    default = parse_speculative_config('{"method":"mtp"}')
    enabled = parse_speculative_config('{"method":"mtp","continuous_batching":true}')
    disabled = parse_speculative_config('{"method":"mtp","continuous_batching":false}')
    assert default is not None and default.continuous_batching is None
    # Omitted preserves fixed cohorts; dynamic membership is explicit opt-in.
    assert default.allow_dynamic_membership is False
    pinned = parse_speculative_config(
        '{"method":"mtp","allow_dynamic_membership":false}'
    )
    assert pinned is not None and pinned.allow_dynamic_membership is False
    assert enabled is not None and enabled.continuous_batching is True
    assert disabled is not None and disabled.continuous_batching is False

    dynamic = parse_speculative_config(
        '{"method":"mtp","continuous_batching":true,"allow_dynamic_membership":true}'
    )
    assert dynamic is not None and dynamic.allow_dynamic_membership is True

    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","continuous_batching":1}')
    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":1}')
    with pytest.raises(SpeculativeConfigError, match="requires continuous_batching"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":true}')
    with pytest.raises(SpeculativeConfigError, match="unsupported speculative-config"):
        parse_speculative_config('{"method":"suffix","continuous_batching":true}')


def test_install_plan_is_default_off_and_fails_closed_without_mutation():
    model = _Model()
    before = dict(vars(model))

    disabled = plan_router_install(model, enabled=False, hard_reserve_bytes=0)
    quantized = plan_router_install(
        model,
        enabled=True,
        cache_quantized=True,
        hard_reserve_bytes=0,
    )

    assert disabled.admitted is False
    assert disabled.fallback is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert "disabled" in " ".join(disabled.reasons)
    assert quantized.admitted is False
    assert "quantized cache" in " ".join(quantized.reasons)
    assert vars(model) == before


def test_single_lane_deployment_refuses_instead_of_raising():
    """``--max-num-seqs 1`` is a deployment choice, not a programming error.

    The planner runs lazily, from the first request's batch-generator build, so
    letting ``BatchedMTPConfig`` raise on ``min_batch_lanes > max_lanes`` there
    aborted that request's generation step and left it waiting for tokens that
    never came. One lane has to refuse like any other missing capability and
    keep the singleton verifier.
    """
    model = _Model()
    before = dict(vars(model))

    single = plan_router_install(model, enabled=True, max_lanes=1, hard_reserve_bytes=0)
    at_minimum = plan_router_install(
        model, enabled=True, max_lanes=2, hard_reserve_bytes=0
    )

    assert single.admitted is False
    assert single.router is None
    assert single.fallback is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert "at least 2 completion lanes" in " ".join(single.reasons)
    # The boundary itself still admits: two lanes are enough to amortize.
    assert at_minimum.admitted is True
    assert vars(model) == before


def test_single_lane_without_legacy_mtp_falls_back_to_plain_decode():
    class _PlainOnly:
        batched_mtp_capability = _descriptor()

        def __call__(self, *args, **kwargs):
            return args, kwargs

        def mtp_batch_forward(self, *args, **kwargs):
            return args, kwargs

    decision = plan_router_install(
        _PlainOnly(), enabled=True, max_lanes=1, hard_reserve_bytes=0
    )

    assert decision.admitted is False
    assert decision.fallback is ContinuousMTPIntegrationRoute.PLAIN_DECODE
    assert "this deployment has 1" in " ".join(decision.reasons)


def test_install_plan_threads_attested_dynamic_membership():
    admitted = plan_router_install(
        _Model(),
        enabled=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert admitted.admitted is True
    assert admitted.router is not None
    assert admitted.router.config.allow_dynamic_membership is True
    assert admitted.router.runtime.capabilities.dynamic_membership is True


def test_supported_requests_build_an_immutable_fixed_cohort_plan():
    router = _router()

    decision = router.plan([_request("a", 1), _request("b", 2)], free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.live_token_delivery is False
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert [lane.spec.uid for lane in decision.cohort] == [1, 2]
    assert [lane.spec.num_draft for lane in decision.cohort] == [2, 2]
    assert all(lane.prepared_state is None for lane in decision.cohort)


def test_exact_apc_sidecar_is_validated_and_carried_as_a_resume_plan():
    router = _router()
    prefix = tuple(range(64))
    hit = _apc_hit(prefix)
    requests = [
        _request(
            "a",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=hit,
        ),
        _request("b", 2),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    restored = decision.cohort[0]
    assert restored.resume_at == 64
    assert restored.spec.prompt == (999,)
    assert restored.spec.prompt_cache == "target-cache"
    assert restored.spec.mtp_cache == "mtp-cache"
    assert restored.prepared_state is hit.state
    assert decision.live_token_delivery is False


def test_bad_apc_sidecar_routes_plain_while_other_lanes_form_cohort():
    router = _router()
    prefix = tuple(range(64))
    foreign = PreparedStateIdentity.from_config(
        model_id="other/model",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )
    requests = [
        _request(
            "bad-apc",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=_apc_hit(prefix, identity=foreign),
        ),
        _request("a", 2),
        _request("b", 3),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert decision.plain_lane_ids == ("bad-apc",)
    assert "model_mismatch" in " ".join(decision.reasons)


def test_unsupported_sampling_falls_back_to_legacy_without_a_cohort():
    router = _router()
    transformed = SamplingContract(greedy=False)
    requests = [
        _request("a", 1, sampling=transformed, temperature=0.8),
        _request("b", 2, sampling=transformed, temperature=0.8),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert decision.cohort == ()
    assert decision.legacy_lane_ids == ("a", "b")
    assert "transformed_distribution_verify" in " ".join(decision.reasons)


def test_scheduler_wiring_diverts_next_and_refusal_precedes_mutation():
    tree = ast.parse((ROOT / "rapid_mlx" / "scheduler.py").read_text(encoding="utf-8"))
    installer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_continuous_mtp_router"
    )
    assignments = [
        node
        for node in ast.walk(installer)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_step"
            for target in node.targets
        )
        for node in assignments
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "next"
            and isinstance(target.value, ast.Name)
            and target.value.id == "batch_gen"
            for target in node.targets
        )
        for node in assignments
    )
    source = ast.get_source_segment(
        (ROOT / "rapid_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        installer,
    )
    assert source is not None
    assert "if not decision.admitted" in source
    assert source.index("if not decision.admitted") < source.index(
        "batch_gen._continuous_mtp_router"
    )
    assert "ContinuousMTPDriver.create" in source
    assert "continuous_responses + list(fallback_responses)" in source
    assert "scheduler_owned_termination" in source
    assert "bool(params.stop) or bool(request.has_tools)" in source
    assert source.index("_remove_queued(selected)") < source.index(
        "raw = original_next()"
    )
    assert "finishing_package.target_cache" in (
        ROOT / "rapid_mlx" / "spec_decode" / "mtp" / "continuous_driver.py"
    ).read_text(encoding="utf-8")

    scheduler_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    create_generator = next(
        node
        for node in scheduler_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_batch_generator"
    )
    create_source = ast.get_source_segment(
        (ROOT / "rapid_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        create_generator,
    )
    assert create_source is not None
    router_call = create_source.index("_install_continuous_mtp_router(")
    vendored_call = create_source.index("_install_mtp_vendored(", router_call)
    assert router_call < vendored_call


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mtp_continuous_batching": 1}, "must be a boolean"),
        ({"mtp_allow_dynamic_membership": 1}, "must be a boolean"),
        (
            {"mtp_continuous_batching": True, "spec_decode": "none"},
            "requires spec_decode='mtp'",
        ),
        (
            {"mtp_allow_dynamic_membership": True},
            "requires mtp_continuous_batching",
        ),
    ],
)
def test_scheduler_config_rejects_ambiguous_continuous_mtp_policy(changes, message):
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=message):
        SchedulerConfig(**changes)


@pytest.mark.requires_mlx
def test_live_installer_fails_closed_before_mutating_unsupported_generators(
    monkeypatch,
):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime

    enabled = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert not _install_continuous_mtp_router(
        SimpleNamespace(next=lambda: ([], [])), _Model(), enabled
    )

    no_capacity = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(
        no_capacity,
        _Model(),
        SimpleNamespace(
            spec_decode="mtp",
            mtp_continuous_batching=True,
            max_num_seqs=0,
            completion_batch_size=4,
        ),
    )
    assert not hasattr(no_capacity, "_continuous_mtp_router")

    disabled = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(
        disabled, _Model(), SchedulerConfig(spec_decode="mtp")
    )
    assert not hasattr(disabled, "_continuous_mtp_router")

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("refused")),
    )
    assembly_refused = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(assembly_refused, _Model(), enabled)
    assert not hasattr(assembly_refused, "_continuous_mtp_router")

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: object(),
    )
    missing_response = _SchedulerBatchGenerator(with_response=False)
    assert not _install_continuous_mtp_router(missing_response, _Model(), enabled)
    assert not hasattr(missing_response, "_continuous_mtp_router")


@pytest.mark.requires_mlx
def test_live_installer_drives_join_detach_remove_and_compat_response(monkeypatch):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime
    from rapid_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from rapid_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=True,
    )
    runtime = SimpleNamespace(capabilities=capabilities)
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    class _Driver:
        dynamic_membership = True
        lane_uids = (1, 2)
        pending_join_uids = ()
        last_attached_uids = ()
        has_work = True
        closed = False
        has_pending_responses = False

        def __init__(self):
            self.calls = []
            self.fail_discard = False
            self.fail_join = False
            self.fail_next = False

        def next(self):
            self.calls.append("next")
            if self.fail_next:
                raise RuntimeError("driver step failed")
            return [SimpleNamespace(uid=1, token=31)]

        def take_terminal_detaches(self):
            self.calls.append("take-terminal")
            return ()

        def resume_turnover(self):
            self.calls.append("resume")
            return ()

        def queue_lanes(self, specs, *, stop_tokens):
            self.calls.append(("join", tuple(spec.uid for spec in specs), stop_tokens))
            if self.fail_join:
                raise RuntimeError("join preparation rejected")
            return tuple(spec.uid for spec in specs)

        def discard_all(self):
            self.calls.append("discard")
            if self.fail_discard:
                raise RuntimeError("already detached")

        def retract_pending_joins(self, uids):
            self.remove_uids(uids)

        def remove_uids(self, uids):
            self.calls.append(("remove", tuple(uids)))
            return (
                SimpleNamespace(
                    uid=1,
                    target_cache="target-cache",
                    token_ids=(10, 11),
                    draft_cache="draft-cache",
                    lane=SimpleNamespace(seed_hidden="seed-hidden"),
                ),
                SimpleNamespace(
                    uid=99,
                    target_cache="foreign",
                    token_ids=(),
                    draft_cache="foreign",
                    lane=SimpleNamespace(seed_hidden=None),
                ),
            )

    driver = _Driver()
    captured = {}

    def _create(_cls, specs, installed_runtime, **kwargs):
        captured["uids"] = tuple(spec.uid for spec in specs)
        captured["runtime"] = installed_runtime
        captured["response_factory"] = kwargs["response_factory"]
        return driver

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(_create))

    batch_gen = _SchedulerBatchGenerator(raw_next=[SimpleNamespace(uid=50)])
    requests = {}
    uid_to_request_id = {}
    for uid in (1, 2, 3, 4, 5, 6, 95):
        request_id = f"req-{uid}"
        requests[request_id] = _scheduler_request()
        uid_to_request_id[uid] = request_id
    uid_to_request_id[97] = "req-97"
    batch_gen._unprocessed_sequences.extend(
        [
            (0,),
            _scheduler_sequence(98),
            _scheduler_sequence(97, segments=[]),
            _scheduler_sequence(95, segments=[]),
            _scheduler_sequence(
                96,
                caches=[SimpleNamespace(caches=[SimpleNamespace(offset="bad")])],
                processors=[object()],
            ),
            _scheduler_sequence(1),
            _scheduler_sequence(2),
        ]
    )
    requests["req-96"] = _scheduler_request()
    uid_to_request_id[96] = "req-96"

    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id=uid_to_request_id,
        free_bytes_getter=lambda: 16 * 1024**3,
        lane_bytes_estimator=lambda request: 1024,
        stop_tokens={99},
    )

    prompt, fallback = batch_gen.next()
    assert prompt == []
    assert [item.uid for item in fallback] == [50]
    assert captured["uids"] == (1, 2)
    assert captured["runtime"] is runtime

    compatible = captured["response_factory"](
        uid=1,
        token=2,
        logprobs=None,
        finish_reason=None,
        from_draft=True,
    )
    assert compatible.required_bookkeeping is None
    assert compatible.from_draft is True

    batch_gen.raw_next = ([], [])
    batch_gen._unprocessed_sequences.append(
        _scheduler_sequence(3, processors=[object()])
    )
    _, responses = batch_gen.next()
    assert [item.uid for item in responses] == [1]
    assert driver.calls[:2] == ["next", "take-terminal"]
    assert not any(
        isinstance(call, tuple) and call[0] == "join" for call in driver.calls
    )
    # This fake base next does not consume plain rows. Model their departure
    # before testing a later admissible FIFO prefix; joins cannot overtake them.
    batch_gen._unprocessed_sequences.clear()

    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (4, 5, 6)]
    )
    batch_gen.next()
    join = next(
        call for call in driver.calls if isinstance(call, tuple) and call[0] == "join"
    )
    assert join[1] == (4, 5)

    driver.lane_uids = (1, 2, 4, 5)
    batch_gen.next()
    driver.lane_uids = (1, 2)
    driver.last_attached_uids = (4, 5)
    driver.fail_join = True
    queued_before_failed_join = list(batch_gen._unprocessed_sequences)
    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == queued_before_failed_join
    driver.fail_join = False
    driver.last_attached_uids = ()
    driver.dynamic_membership = False
    batch_gen.next()
    driver.closed = True
    batch_gen.next()
    assert "resume" in driver.calls

    driver.fail_next = True
    with pytest.raises(RuntimeError, match="driver step failed"):
        batch_gen.next()
    driver.fail_next = False

    removed = batch_gen.remove([1], return_prompt_caches=True)
    assert removed[1] == ("target-cache", [10, 11])
    assert batch_gen._continuous_mtp_removed_states[1] == (
        "draft-cache",
        "seed-hidden",
    )

    assert batch_gen.close() == "closed"
    driver.fail_discard = True
    assert batch_gen.close() == "closed"


@pytest.mark.requires_mlx
def test_live_installer_retains_base_queue_when_initial_prepare_fails(monkeypatch):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime
    from rapid_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from rapid_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    runtime = SimpleNamespace(
        capabilities=ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
            dynamic_membership=True,
        )
    )
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    def fail_create(_cls, *_args, **_kwargs):
        raise RuntimeError("simulated prepare failure")

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(fail_create))
    batch_gen = _SchedulerBatchGenerator(raw_next=([], [SimpleNamespace(uid=99)]))
    sequences = [_scheduler_sequence(1), _scheduler_sequence(2)]
    batch_gen._unprocessed_sequences.extend(sequences)
    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests={"req-1": _scheduler_request(), "req-2": _scheduler_request()},
        uid_to_request_id={1: "req-1", 2: "req-2"},
        free_bytes_getter=lambda: 16 * 1024**3,
        lane_bytes_estimator=lambda request: 1024,
    )

    assert batch_gen.next()[1][0].uid == 99
    assert list(batch_gen._unprocessed_sequences) == sequences
    assert batch_gen._continuous_mtp_driver is None


@pytest.mark.requires_mlx
def test_live_installer_restores_base_queue_when_deferred_join_fails(monkeypatch):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime
    from rapid_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from rapid_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=True,
    )
    runtime = SimpleNamespace(capabilities=capabilities)
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    class _Driver:
        dynamic_membership = True
        lane_uids = (1, 2)
        has_work = True
        closed = False
        has_pending_responses = False

        def __init__(self):
            self.pending = ()
            self.removed = []

        @property
        def pending_join_uids(self):
            return self.pending

        @property
        def last_attached_uids(self):
            return ()

        def next(self):
            if self.pending:
                raise RuntimeError("simulated deferred prepare failure")
            return []

        def take_terminal_detaches(self):
            return ()

        def queue_lanes(self, specs, **_kwargs):
            self.pending = tuple(spec.uid for spec in specs)
            return self.pending

        def retract_pending_joins(self, uids):
            self.remove_uids(uids)

        def remove_uids(self, uids):
            self.removed.append(tuple(uids))
            self.pending = tuple(uid for uid in self.pending if uid not in uids)
            return ()

    driver = _Driver()
    monkeypatch.setattr(
        ContinuousMTPDriver,
        "create",
        classmethod(lambda _cls, *_args, **_kwargs: driver),
    )
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests = {
        "req-1": _scheduler_request(),
        "req-2": _scheduler_request(),
        "req-3": _scheduler_request(),
    }
    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id={1: "req-1", 2: "req-2", 3: "req-3"},
        free_bytes_getter=lambda: 16 * 1024**3,
        lane_bytes_estimator=lambda request: 1024,
    )
    batch_gen.next()
    joining = _scheduler_sequence(3)
    batch_gen._unprocessed_sequences.append(joining)

    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == []
    batch_gen.next()

    assert list(batch_gen._unprocessed_sequences) == [joining]
    assert driver.removed == [(3,)]


@pytest.mark.requires_mlx
def test_live_installer_handles_empty_pressure_and_optional_base_hooks(monkeypatch):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    batch_gen = _SchedulerBatchGenerator()
    batch_gen.close = None
    batch_gen.remove = None
    batch_gen._unprocessed_sequences.append(_scheduler_sequence(1))
    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests={"req-1": _scheduler_request()},
        uid_to_request_id={1: "req-1"},
        lane_bytes_estimator=lambda request: 1024,
        free_bytes_getter=lambda: (_ for _ in ()).throw(RuntimeError("pressure")),
    )
    assert batch_gen.next() == ([], [])
    assert batch_gen.remove([1], return_prompt_caches=True) == {}
    assert batch_gen.close() is None

    no_getter = _SchedulerBatchGenerator()
    no_getter._unprocessed_sequences.append(_scheduler_sequence(2))
    assert _install_continuous_mtp_router(
        no_getter,
        _Model(),
        config,
        requests={"req-2": _scheduler_request()},
        uid_to_request_id={2: "req-2"},
        lane_bytes_estimator=lambda request: 1024,
    )
    assert no_getter.next() == ([], [])

    empty = _SchedulerBatchGenerator()
    assert _install_continuous_mtp_router(empty, _Model(), config)
    assert empty.next() == ([], [])


@pytest.mark.requires_mlx
def test_live_installer_forms_fixed_initial_cohort_without_dynamic_join(monkeypatch):
    """Exercise the real scheduler import and installer on the Apple lane.

    The scheduler intentionally imports MLX and mlx-lm at module load so its
    hardware compatibility hooks run before BatchGenerator is captured.  This
    integration contract therefore belongs on the real-MLX lane; the pure
    routing and AST contracts above remain in the hosted no-MLX matrix.
    """
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime
    from rapid_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from rapid_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    class _BatchGenerator:
        class _GenerationBatch:
            Response = SimpleNamespace

        def __init__(self):
            self._generation_batch = self._GenerationBatch()
            self._unprocessed_sequences = deque()

        def next(self):
            return [], []

        def close(self):
            return None

        def remove(self, _uids, *_args, **_kwargs):
            return {}

    batch_gen = _BatchGenerator()
    requests = {}
    uid_to_request_id = {}
    for uid in (1, 2):
        request_id = f"req-{uid}"
        requests[request_id] = SimpleNamespace(
            sampling_params=SimpleNamespace(
                temperature=0.0,
                stop=None,
            ),
            has_tools=False,
        )
        uid_to_request_id[uid] = request_id
        batch_gen._unprocessed_sequences.append(
            (
                uid,
                [[10 + uid, 20 + uid]],
                16,
                [SimpleNamespace(offset=0, nbytes=0)],
                [],
                object(),
                [],
                None,
                None,
            )
        )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=False,
    )
    runtime = SimpleNamespace(capabilities=capabilities)
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    created = {}
    driver = SimpleNamespace(dynamic_membership=False, has_work=True)

    def _create(_cls, specs, installed_runtime, **_kwargs):
        created["uids"] = tuple(spec.uid for spec in specs)
        created["runtime"] = installed_runtime
        return driver

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(_create))

    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=False,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id=uid_to_request_id,
        free_bytes_getter=lambda: 16 * 1024**3,
        lane_bytes_estimator=lambda request: 1024,
    )

    prompt_responses, completion_responses = batch_gen.next()

    assert prompt_responses == []
    assert completion_responses == []
    assert created == {"uids": (1, 2), "runtime": runtime}
    assert batch_gen._continuous_mtp_driver is driver
    assert list(batch_gen._unprocessed_sequences) == []


def test_cli_resolves_artifact_auto_policy_before_default_off_scheduler_by_ast():
    cli_source = (ROOT / "rapid_mlx" / "cli.py").read_text(encoding="utf-8")
    scheduler_source = (ROOT / "rapid_mlx" / "scheduler.py").read_text(encoding="utf-8")
    assert 'continuous_tier == "verified"' in cli_source
    assert "config.continuous_batching is None" in cli_source
    # The cohort itself is an explicit opt-in; verified artifacts select only
    # their unquantized cache contract automatically.
    assert "args.mtp_continuous_batching = config.continuous_batching is True" in (
        cli_source
    )
    assert (
        'mtp_continuous_batching=getattr(args, "mtp_continuous_batching", False)'
        in cli_source
    )
    assert "mtp_continuous_batching: bool = False" in scheduler_source
    assert "mtp_allow_dynamic_membership: bool = False" in scheduler_source
    assert (
        'if self.mtp_continuous_batching and self.spec_decode != "mtp"'
        in scheduler_source
    )
    assert "mtp_allow_dynamic_membership=getattr(" in cli_source


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"uid": True}, "uid must be an integer"),
        ({"prompt_tokens": ()}, "non-negative integers"),
        ({"prompt_tokens": (1, -1)}, "non-negative integers"),
        ({"max_tokens": True}, "max_tokens must be an integer"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"stop_tokens": frozenset({True})}, "stop_tokens must contain integers"),
        ({"temperature": 0.5}, "greedy sampling requires"),
        (
            {"sampling": SamplingContract(greedy=False), "temperature": 0.0},
            "non-greedy sampling requires",
        ),
    ],
)
def test_request_metadata_rejects_ambiguous_scheduler_facts(changes, message):
    values = {
        "lane_id": "lane",
        "uid": 1,
        "prompt_tokens": (1, 2),
        "max_tokens": 8,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        ContinuousMTPRequestMetadata(**values)


def test_router_rejects_duplicate_lane_and_uid_identity():
    router = _router()
    with pytest.raises(ValueError, match="lane_id values must be unique"):
        router.plan([_request("same", 1), _request("same", 2)], free_bytes=1)
    with pytest.raises(ValueError, match="uid values must be unique"):
        router.plan([_request("a", 1), _request("b", 1)], free_bytes=1)


def test_static_refusals_preserve_plain_fallback_without_legacy_mtp():
    class _PlainOnly:
        batched_mtp_capability = _descriptor("qwen4_exp")

    decision = plan_router_install(
        _PlainOnly(),
        enabled=True,
        cache_windowed=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert decision.admitted is False
    assert decision.fallback is ContinuousMTPIntegrationRoute.PLAIN_DECODE
    joined = " ".join(decision.reasons)
    assert "unsupported model family" in joined
    assert "windowed cache" in joined
    assert "dynamic membership" in joined


def test_descriptor_mismatch_and_plain_runtime_fallback_are_explicit():
    class _Mismatched(_Model):
        batched_mtp_capability = MappingProxyType(
            {**dict(_descriptor()), "protocol_version": 2}
        )

    refused = plan_router_install(_Mismatched(), enabled=True, hard_reserve_bytes=0)
    assert refused.admitted is False
    assert "protocol_version" in " ".join(refused.reasons)

    plain_router = _router()
    plain_router.runtime = type(plain_router.runtime)(
        model_family=plain_router.runtime.model_family,
        capability_descriptor=plain_router.runtime.capability_descriptor,
        capabilities=plain_router.runtime.capabilities,
        legacy_mtp_available=False,
    )
    transformed = SamplingContract(greedy=False)
    decision = plain_router.plan(
        [
            _request("a", 1, sampling=transformed, temperature=0.8),
            _request("b", 2, sampling=transformed, temperature=0.8),
        ],
        free_bytes=1,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.PLAIN_DECODE
    assert decision.plain_lane_ids == ("a", "b")


def test_runtime_static_refusal_routes_legacy_or_plain_per_loaded_surface():
    for legacy in (True, False):
        router = _router()
        router._static_refusals = ("runtime drift",)
        router.runtime = type(router.runtime)(
            model_family=router.runtime.model_family,
            capability_descriptor=router.runtime.capability_descriptor,
            capabilities=router.runtime.capabilities,
            legacy_mtp_available=legacy,
        )
        decision = router.plan([_request("a", 1)], free_bytes=1)
        expected = (
            ContinuousMTPIntegrationRoute.LEGACY_MTP
            if legacy
            else ContinuousMTPIntegrationRoute.PLAIN_DECODE
        )
        assert decision.route is expected
        assert decision.reasons == ("runtime drift",)


def test_request_local_cache_refusal_does_not_block_other_cohort_lanes():
    decision = _router().plan(
        [
            _request("quantized", 1, cache_quantized=True),
            _request("a", 2),
            _request("b", 3),
        ],
        free_bytes=1,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.plain_lane_ids == ("quantized",)
    assert "quantized/windowed" in " ".join(decision.reasons)


def test_queue_only_admission_remains_queue_when_every_lane_is_plain(monkeypatch):
    monkeypatch.setattr(
        routing_module,
        "plan_admission",
        lambda *_args, **_kwargs: AdmissionDecision(BatchedMTPRoute.QUEUE),
    )
    decision = _router().plan(
        [_request("quantized", 1, cache_quantized=True)],
        free_bytes=0,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.QUEUE


# ---------------------------------------------------------------------------
# Dynamic membership: resumed caches, memory-planned lanes, singleton requeue
# ---------------------------------------------------------------------------


class _RecordingDriver:
    """Minimal live driver: records joins and exposes dynamic membership."""

    def __init__(self, specs, *, dynamic=True):
        self.specs = list(specs)
        self.dynamic_membership = dynamic
        self.pending = ()
        self.joined = []
        self.has_work = True
        self.closed = False
        self.has_pending_responses = False
        self.last_attached_uids = ()

    @property
    def lane_uids(self):
        return tuple(spec.uid for spec in self.specs)

    @property
    def pending_join_uids(self):
        return self.pending

    def next(self):
        self.specs.extend(self.joined_specs if self.pending else ())
        self.pending = ()
        return []

    def take_terminal_detaches(self):
        return ()

    def queue_lanes(self, specs, **_kwargs):
        self.joined_specs = list(specs)
        self.joined.append(tuple(spec.uid for spec in specs))
        self.pending = tuple(spec.uid for spec in specs)
        return self.pending

    def retract_pending_joins(self, _uids):
        self.pending = tuple(uid for uid in self.pending if uid not in _uids)
        self.joined_specs = [
            spec for spec in getattr(self, "joined_specs", []) if spec.uid not in _uids
        ]

    def remove_uids(self, _uids):
        self.retract_pending_joins(_uids)
        return ()

    def discard_all(self):
        return ()


def _install_recording_router(
    monkeypatch,
    batch_gen,
    requests,
    uid_to_request_id,
    *,
    dynamic=True,
    free_bytes=16 * 1024**3,
    estimator=lambda request: 1024,
    max_lanes=8,
):
    from rapid_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from rapid_mlx.spec_decode.mtp import continuous_runtime
    from rapid_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from rapid_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=dynamic,
    )
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(capabilities=capabilities),
    )
    created = []

    def _create(_cls, specs, _runtime, **_kwargs):
        driver = _RecordingDriver(specs, dynamic=dynamic)
        created.append(driver)
        return driver

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(_create))
    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=dynamic,
        max_num_seqs=max_lanes,
        completion_batch_size=max_lanes,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id=uid_to_request_id,
        free_bytes_getter=free_bytes if callable(free_bytes) else (lambda: free_bytes),
        stop_tokens={99},
        lane_bytes_estimator=estimator,
    )
    return created


def _requests_for(*uids, **request_changes):
    requests = {f"req-{uid}": _scheduler_request(**request_changes) for uid in uids}
    return requests, {uid: f"req-{uid}" for uid in uids}


@pytest.mark.requires_mlx
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_resumed_prefix_cache_lane_joins_the_cohort_with_its_cache(monkeypatch, nested):
    batch_gen = _SchedulerBatchGenerator()
    leaf = SimpleNamespace(offset=12, nbytes=0)
    # A CacheList-style wrapper resumes from its leaves' offsets too.
    resumed_cache = [SimpleNamespace(caches=[leaf], nbytes=0) if nested else leaf]
    batch_gen._unprocessed_sequences.extend(
        [
            _scheduler_sequence(1),
            _scheduler_sequence(2, caches=resumed_cache, segments=[[44]]),
        ]
    )
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)

    batch_gen.next()

    assert len(created) == 1
    fresh, resumed = created[0].specs
    assert fresh.prompt_cache is None
    assert resumed.prompt_cache == resumed_cache
    # Only the unprocessed remainder is prefilled on top of the cache.
    assert tuple(resumed.prompt) == (44,)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("offsets", [(0, 12), (12, 13), (12, -1), (12, float("inf"))])
def test_inconsistent_cache_positions_preserve_ordinary_ownership(
    monkeypatch, nested, offsets
):
    leaves = [SimpleNamespace(offset=offset, nbytes=0) for offset in offsets]
    caches = [SimpleNamespace(caches=leaves)] if nested else leaves
    batch_gen = _SchedulerBatchGenerator()
    source = [_scheduler_sequence(1), _scheduler_sequence(2, caches=caches)]
    batch_gen._unprocessed_sequences.extend(source)
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    assert created == []
    assert list(batch_gen._unprocessed_sequences) == source


@pytest.mark.requires_mlx
def test_recurrent_cache_without_offset_coexists_with_agreeing_attention_positions(
    monkeypatch,
):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    state = ArraysCache(1)
    state[0] = mx.zeros((1, 2))
    caches = [
        SimpleNamespace(offset=12, nbytes=0),
        state,
        SimpleNamespace(offset=12, nbytes=0),
    ]
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2, caches=caches)]
    )
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    assert len(created) == 1
    assert created[0].lane_uids == (1, 2)
    assert isinstance(created[0].specs[1].prompt_cache[1], ArraysCache)
    assert mx.array_equal(created[0].specs[1].prompt_cache[1][0], state[0]).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_recurrent_only_cache_preserves_state_without_inventing_a_prefix(
    monkeypatch, nested, populated
):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    state = ArraysCache(1)
    if populated:
        state[0] = mx.array([[3.0, 7.0]])
    caches = [SimpleNamespace(caches=[state])] if nested else [state]
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2, caches=caches)]
    )
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    assert created[0].lane_uids == (1, 2)
    restored = created[0].specs[1].prompt_cache
    if not populated:
        assert restored is None
    else:
        leaf = restored[0].caches[0] if nested else restored[0]
        assert leaf is not state
        assert mx.array_equal(leaf[0], state[0]).item()


@pytest.mark.requires_mlx
def test_populated_recurrent_state_with_zero_attention_position_stays_queued(
    monkeypatch,
):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    state = ArraysCache(1)
    state[0] = mx.ones((1, 2))
    batch_gen = _SchedulerBatchGenerator()
    source = [
        _scheduler_sequence(1),
        _scheduler_sequence(2, caches=[SimpleNamespace(offset=0), state]),
    ]
    batch_gen._unprocessed_sequences.extend(source)
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    assert created == []
    assert list(batch_gen._unprocessed_sequences) == source


@pytest.mark.requires_mlx
def test_windowed_or_quantized_layer_anywhere_keeps_the_lane_off_the_cohort(
    monkeypatch,
):
    class RotatingKVCache:
        offset = 0
        nbytes = 0

    class QuantizedKVCache(RotatingKVCache):
        pass

    batch_gen = _SchedulerBatchGenerator()
    nested_window = [
        SimpleNamespace(offset=0, nbytes=0),
        SimpleNamespace(caches=[RotatingKVCache()]),
    ]
    batch_gen._unprocessed_sequences.extend(
        [
            _scheduler_sequence(1),
            _scheduler_sequence(2, caches=nested_window),
            _scheduler_sequence(3, caches=[QuantizedKVCache()]),
        ]
    )
    requests, mapping = _requests_for(1, 2, 3)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)

    batch_gen.next()

    # Neither refused lane may pair with uid 1, so no cohort forms at all.
    assert created == []
    assert [sequence[0] for sequence in batch_gen._unprocessed_sequences] == [1, 2, 3]


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("free_bytes", "expected"),
    [
        # Reserve (8 GiB) + room for exactly two 1 GiB lanes: FIFO prefix of 2.
        (8 * 1024**3 + int(2.5 * 1024**3), (1, 2)),
        # Unknown telemetry preserves the ordinary queue.
        (None, None),
        # Unusable telemetry is unknown telemetry, not zero headroom.
        ("n/a", None),
        # A known cap with no headroom admits nothing that costs memory.
        (0, None),
    ],
)
def test_initial_cohort_is_memory_planned_from_lane_estimates(
    monkeypatch, free_bytes, expected
):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (1, 2, 3)]
    )
    requests, mapping = _requests_for(1, 2, 3)
    estimated = []

    def estimator(request):
        estimated.append(request)
        return 1024**3

    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        free_bytes=free_bytes,
        estimator=estimator,
    )
    batch_gen.next()

    if expected is None:
        assert created == []
    else:
        assert created[0].lane_uids == expected
    assert bool(estimated) is isinstance(free_bytes, int)


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("free_bytes", "max_lanes", "expected"),
    [
        # Live headroom already includes the running lanes; only joiners are
        # charged, the hard reserve stays intact, and FIFO stops at the first
        # lane that does not fit.
        (8 * 1024**3 + int(2.5 * 1024**3), 8, (3, 4)),
        # Without telemetry the joining requests remain in the ordinary queue.
        (None, 8, None),
        # A known cap with no headroom admits no joiner.
        (0, 8, None),
        (16 * 1024**3, 3, (3,)),
        # No room left under the reserve: nothing joins.
        (8 * 1024**3, 8, None),
    ],
)
def test_dynamic_joins_charge_projected_lane_bytes_against_headroom(
    monkeypatch, free_bytes, max_lanes, expected
):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (1, 2)]
    )
    requests, mapping = _requests_for(1, 2, 3, 4, 5)
    headroom = {"free": 16 * 1024**3}
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        free_bytes=lambda: headroom["free"],
        estimator=lambda _request: 1024**3,
        max_lanes=max_lanes,
    )
    batch_gen.next()
    driver = created[0]
    assert driver.lane_uids == (1, 2)

    headroom["free"] = free_bytes
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (3, 4, 5)]
    )
    batch_gen.next()

    if expected is None:
        assert driver.joined == []
    else:
        assert driver.joined == [expected]
        assert [seq[0] for seq in batch_gen._unprocessed_sequences] == [
            uid for uid in (3, 4, 5) if uid not in expected
        ]


class _OwnerGenerationBatch:
    Response = _SchedulerResponse

    def __init__(self, uid, *, processors=None):
        self.uids = [uid]
        self.logits_processors = [[] if processors is None else processors]
        self.prompt_cache = [SimpleNamespace(offset=40, nbytes=0)]


class _SingletonBatchGenerator(_SchedulerBatchGenerator):
    """Base generator whose generation batch holds the vendored owner."""

    def __init__(self, owner_batch, *, with_hook=True):
        super().__init__()
        self._generation_batch = owner_batch
        self._mtp_vendored_admission_owner = owner_batch.uids[0]
        self.requeued = []
        if with_hook:
            self._mtp_vendored_requeue_owner = self._requeue

    def _requeue(self, *, admit=None):
        owner = self._mtp_vendored_admission_owner
        requeued = list(
            _scheduler_sequence(
                owner, caches=[SimpleNamespace(offset=40, nbytes=0)], segments=[[99]]
            )
        )
        # The verifier hands back the row's processor context with it.
        requeued[4] = list(range(40))
        requeued[6] = self._generation_batch.logits_processors[0]
        if admit is not None and not admit(tuple(requeued)):
            return False
        if getattr(self, "refuse_removal", False):
            return False
        # Production's vendored handoff removes through the installed router.
        self.remove([owner])
        self.requeued.append(owner)
        self._generation_batch.uids = []
        self._mtp_vendored_admission_owner = None
        self._unprocessed_sequences.appendleft(tuple(requeued))
        return True


def _absorb_setup(
    monkeypatch,
    *,
    owner_request=None,
    dynamic=True,
    router_changes=None,
    **batch_changes,
):
    owner_batch = _OwnerGenerationBatch(
        7, processors=batch_changes.pop("processors", None)
    )
    batch_gen = _SingletonBatchGenerator(owner_batch, **batch_changes)
    requests, mapping = _requests_for(1, 2, 3)
    requests["req-7"] = owner_request or _scheduler_request()
    mapping[7] = "req-7"
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        dynamic=dynamic,
        **(router_changes or {}),
    )
    batch_gen.next()  # the waiting pair forms a cohort beside the singleton
    assert created[0].lane_uids == (1, 2)
    return batch_gen, owner_batch, created[0]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("cancel_before_attach", [False, True])
def test_live_dynamic_cohort_absorbs_an_eligible_singleton(
    monkeypatch, cancel_before_attach
):
    batch_gen, _owner, driver = _absorb_setup(monkeypatch)

    batch_gen.next()

    assert batch_gen.requeued == [7]
    assert driver.joined == [(7,)]
    assert driver.pending_join_uids == (7,)
    assert batch_gen.removed == [(7,)]
    joined = driver.joined_specs[0]
    assert tuple(joined.prompt) == (99,)
    assert joined.prompt_cache is not None
    if cancel_before_attach:
        # A later cancellation still retracts the reservation normally.
        batch_gen.remove([7])
        assert driver.pending_join_uids == ()
    else:
        batch_gen.next()
        assert driver.pending_join_uids == ()
        assert driver.lane_uids == (1, 2, 7)


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "case", ["sampled_owner", "processor_owner", "fixed_cohort", "foreign", "no_hook"]
)
def test_singleton_stays_put_when_the_cohort_cannot_absorb_it(monkeypatch, case):
    kwargs = {}
    if case == "sampled_owner":
        kwargs["owner_request"] = _scheduler_request(temperature=0.7)
    if case == "processor_owner":
        kwargs["processors"] = [object()]
    if case == "fixed_cohort":
        kwargs["dynamic"] = False
    if case == "no_hook":
        kwargs["with_hook"] = False
    batch_gen, owner_batch, driver = _absorb_setup(monkeypatch, **kwargs)
    if case == "foreign":
        owner_batch.uids = [7, 8]

    batch_gen.next()

    assert batch_gen.requeued == []
    assert driver.joined == []
    assert batch_gen._mtp_vendored_admission_owner == 7


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("metal", "footprint", "rss", "expected"),
    [
        (None, None, None, None),
        (None, 600, None, 400),
        (None, None, 700, None),
        (800, None, None, None),
        (400, 600, None, 400),
        # A concurrently newer Metal reading must bound the kernel snapshot.
        (800, 600, None, 200),
        (1200, 600, None, 0),
        (1200, None, None, None),
        (0, None, None, None),
        (-1, None, None, None),
        (400, None, 300, 300),
        (800, None, 700, 0),
        (0, None, 0, 1000),
        (False, None, 0, None),
        (0, None, -1, None),
    ],
)
def test_continuous_headroom_requires_comprehensive_usage(
    monkeypatch, metal, footprint, rss, expected
):
    import psutil

    from rapid_mlx import scheduler as scheduler_module
    from rapid_mlx.runtime import process_memory

    def probe(value):
        if value is None:
            raise RuntimeError("usage probe unavailable")
        return value

    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler._resolve_metal_cap_bytes = lambda: 1000
    monkeypatch.setattr(scheduler_module.mx, "get_active_memory", lambda: probe(metal))
    monkeypatch.setattr(process_memory, "get_phys_footprint", lambda: probe(footprint))
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=probe(rss))),
    )

    assert scheduler._continuous_mtp_free_bytes() == expected
    # Existing admission callers keep their established best-effort fallback.
    if metal is None:
        assert scheduler._current_metal_active_bytes() == 0
    if footprint is None and rss is None:
        assert scheduler._current_process_resident_bytes() == 0

    # A failed snapshot must not poison later successful measurements.
    monkeypatch.setattr(process_memory, "get_phys_footprint", lambda: 900)
    metal_floor = metal if type(metal) is int and metal >= 0 else 0
    assert scheduler._continuous_mtp_free_bytes() == max(
        0, 1000 - max(900, metal_floor)
    )


@pytest.mark.requires_mlx
def test_lane_bytes_estimate_adds_the_mtp_head_kv_to_the_target_projection():
    from rapid_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._estimate_request_kv_bytes = lambda _request: 1_000
    attention = SimpleNamespace(num_key_value_heads=2, head_dim=256)
    head = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    scheduler.model = SimpleNamespace(language_model=SimpleNamespace(mtp=head))
    request = SimpleNamespace(
        prompt_token_ids=list(range(30)),
        sampling_params=SimpleNamespace(max_tokens=226),
    )

    # One bf16 attention layer: 2 (K,V) * 2 heads * 256 dims * 2 bytes.
    assert scheduler._mtp_draft_kv_bytes_per_token() == 2048
    assert scheduler._continuous_mtp_lane_bytes(request) == 1_000 + 2048 * 256

    bare = Scheduler.__new__(Scheduler)
    bare._estimate_request_kv_bytes = lambda _request: 7
    bare.model = SimpleNamespace()
    with pytest.raises(ValueError, match="draft cache geometry unavailable"):
        bare._continuous_mtp_lane_bytes(request)


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "layers",
    [
        [],
        object(),
        [SimpleNamespace()],
        [
            SimpleNamespace(
                self_attn=SimpleNamespace(num_key_value_heads=0, head_dim=256)
            )
        ],
        [
            SimpleNamespace(
                self_attn=SimpleNamespace(num_key_value_heads=2, head_dim=None)
            )
        ],
    ],
)
def test_unknown_draft_geometry_refuses_projection_without_caching_zero(layers):
    from rapid_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace(mtp=SimpleNamespace(layers=layers))
    with pytest.raises(ValueError, match="geometry unavailable"):
        scheduler._mtp_draft_kv_bytes_per_token()
    assert not hasattr(scheduler, "_mtp_draft_kv_per_token_cache")
    scheduler.model = SimpleNamespace(
        mtp=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(num_key_value_heads=2, head_dim=256)
                )
            ]
        )
    )
    assert scheduler._mtp_draft_kv_bytes_per_token() == 2048


@pytest.mark.requires_mlx
def test_resumed_lane_prepares_on_a_private_copy_of_its_cache(monkeypatch):
    batch_gen = _SchedulerBatchGenerator()
    resumed_cache = [SimpleNamespace(offset=12, nbytes=0, keys=[1, 2, 3])]
    resumed = _scheduler_sequence(2, caches=resumed_cache, segments=[[44]])
    batch_gen._unprocessed_sequences.extend([_scheduler_sequence(1), resumed])
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)

    batch_gen.next()

    spec = created[0].specs[1]
    # Preparation mutates its cache before the cohort is known to form; the
    # queued tuple (the fallback owner) must keep the original untouched.
    assert spec.prompt_cache == resumed_cache
    assert spec.prompt_cache[0] is not resumed_cache[0]
    assert resumed[3][0] is resumed_cache[0]


@pytest.mark.requires_mlx
def test_unplannable_lane_is_never_admitted_on_a_guess(monkeypatch):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (1, 2, 3)]
    )
    requests, mapping = _requests_for(1, 2, 3)

    def estimator(request):
        if request is requests["req-2"]:
            raise RuntimeError("no model config")
        return 1024**3

    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        free_bytes=16 * 1024**3,
        estimator=estimator,
    )
    batch_gen.next()

    # The FIFO prefix stops at the lane whose footprint is unknown, leaving a
    # single lane: below the cohort minimum, so the base path keeps them all.
    assert created == []
    assert [seq[0] for seq in batch_gen._unprocessed_sequences] == [1, 2, 3]


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "case",
    [
        "full",
        "low",
        "unknown",
        "probe_error",
        "infinite",
        "join_error",
        "join_partial_error",
        "join_refused",
        "remove_refused",
    ],
)
def test_singleton_admission_failure_preserves_owner_cache_and_progress(
    monkeypatch, case
):
    headroom = {"free": 16 * 1024**3}

    def probe():
        if headroom["free"] == "error":
            raise RuntimeError("telemetry unavailable")
        return headroom["free"]

    batch_gen, owner, driver = _absorb_setup(
        monkeypatch,
        router_changes={
            "free_bytes": probe,
            "estimator": lambda request: 1024**3,
            "max_lanes": 2 if case == "full" else 8,
        },
    )
    cache = owner.prompt_cache
    if case == "low":
        headroom["free"] = 8 * 1024**3
    if case == "unknown":
        headroom["free"] = None
    if case == "probe_error":
        headroom["free"] = "error"
    if case == "infinite":
        headroom["free"] = float("inf")
    if case == "join_refused":
        driver.queue_lanes = lambda *args, **kwargs: ()
    if case == "join_error":

        def reject(*args, **kwargs):
            assert owner.uids == [7]  # reservation precedes any departure
            raise RuntimeError("join refused")

        driver.queue_lanes = reject
    if case == "join_partial_error":
        queue_lanes = driver.queue_lanes

        def stage_then_raise(*args, **kwargs):
            queue_lanes(*args, **kwargs)
            raise RuntimeError("fault after staging singleton reservation")

        driver.queue_lanes = stage_then_raise
    if case == "remove_refused":
        batch_gen.refuse_removal = True
    batch_gen.next()
    assert owner.uids == [7]
    assert owner.prompt_cache is cache and cache[0].offset == 40
    assert batch_gen._mtp_vendored_admission_owner == 7
    assert batch_gen.requeued == []
    assert driver.pending_join_uids == ()
    assert driver.lane_uids == (1, 2)
    assert not batch_gen._unprocessed_sequences


@pytest.mark.requires_mlx
def test_singleton_reservation_is_charged_before_other_dynamic_joins(monkeypatch):
    headroom = {"free": 16 * 1024**3}
    batch_gen, owner, driver = _absorb_setup(
        monkeypatch,
        router_changes={
            "free_bytes": lambda: headroom["free"],
            "estimator": lambda request: 1024**3,
        },
    )
    headroom["free"] = 9 * 1024**3
    batch_gen._unprocessed_sequences.append(_scheduler_sequence(3))
    batch_gen.next()
    assert owner.uids == []
    assert driver.pending_join_uids == (7,)
    assert [sequence[0] for sequence in batch_gen._unprocessed_sequences] == [3]


@pytest.mark.requires_mlx
def test_infinite_lane_estimate_is_unplannable(monkeypatch):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        estimator=lambda request: float("inf"),
    )
    batch_gen.next()
    assert created == []
    assert [sequence[0] for sequence in batch_gen._unprocessed_sequences] == [1, 2]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("partial", [False, True, "raise_after_staging"])
def test_unconfirmed_dynamic_join_keeps_original_queue_ownership(monkeypatch, partial):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2, 3, 4)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    driver = created[0]
    original = driver.queue_lanes

    def refuse(specs, **kwargs):
        if partial == "raise_after_staging":
            original(specs[:1], **kwargs)
            raise RuntimeError("fault after partial dynamic staging")
        if partial:
            return original(specs[:1], **kwargs)
        return ()

    driver.queue_lanes = refuse
    source = [_scheduler_sequence(3), _scheduler_sequence(4)]
    batch_gen._unprocessed_sequences.extend(source)
    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == source
    assert driver.pending_join_uids == ()
    assert driver.lane_uids == (1, 2)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("joining", [False, True])
def test_resumed_cache_clone_failure_preserves_source_queue(monkeypatch, joining):
    class UncopyableCache:
        offset = 12
        nbytes = 0

        def __deepcopy__(self, memo):
            raise MemoryError("cache clone refused")

    batch_gen = _SchedulerBatchGenerator()
    requests, mapping = _requests_for(1, 2, 3)
    source = [_scheduler_sequence(1), _scheduler_sequence(2)]
    resumed = _scheduler_sequence(3, caches=[UncopyableCache()], segments=[[44]])
    batch_gen._unprocessed_sequences.extend(source if joining else [source[0], resumed])
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    if joining:
        driver = created[0]
        batch_gen._unprocessed_sequences.append(resumed)
        batch_gen.next()
        assert list(batch_gen._unprocessed_sequences) == [resumed]
        assert driver.pending_join_uids == () and driver.lane_uids == (1, 2)
    else:
        assert created == []
        assert list(batch_gen._unprocessed_sequences) == [source[0], resumed]
    assert resumed[3][0].offset == 12


@pytest.mark.requires_mlx
@pytest.mark.parametrize("headroom", [16 * 1024**3, 1 << 70])
@pytest.mark.parametrize("estimator", [None, lambda _request: float("inf")])
def test_missing_lane_estimator_refuses_even_empty_cache_admission(
    monkeypatch, headroom, estimator
):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        estimator=estimator,
        free_bytes=headroom,
    )
    batch_gen.next()
    assert created == []
    assert [sequence[0] for sequence in batch_gen._unprocessed_sequences] == [1, 2]


@pytest.mark.requires_mlx
def test_unknown_production_draft_geometry_keeps_ordinary_queue_ownership(monkeypatch):
    from rapid_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace(mtp=SimpleNamespace(layers=[]))
    scheduler._estimate_request_kv_bytes = lambda _request: 1024
    batch_gen = _SchedulerBatchGenerator()
    sources = [_scheduler_sequence(1), _scheduler_sequence(2)]
    batch_gen._unprocessed_sequences.extend(sources)
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        estimator=scheduler._continuous_mtp_lane_bytes,
    )
    batch_gen.next()
    assert created == []
    assert list(batch_gen._unprocessed_sequences) == sources


@pytest.mark.requires_mlx
@pytest.mark.parametrize("estimate", [-1, 0, False, True, 1.5, "1024"])
def test_invalid_lane_projection_never_becomes_a_small_reservation(
    monkeypatch, estimate
):
    batch_gen = _SchedulerBatchGenerator()
    source = [_scheduler_sequence(1), _scheduler_sequence(2)]
    batch_gen._unprocessed_sequences.extend(source)
    requests, mapping = _requests_for(1, 2)
    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        estimator=lambda _request: estimate,
    )
    batch_gen.next()
    assert created == []
    assert list(batch_gen._unprocessed_sequences) == source


@pytest.mark.requires_mlx
def test_partial_join_rollback_preserves_earlier_pending_reservation(monkeypatch):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2, 3, 4)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    driver = created[0]
    batch_gen._unprocessed_sequences.append(_scheduler_sequence(3))
    batch_gen.next()
    assert driver.pending_join_uids == (3,)
    # Keep the earlier reservation pending while the next attempt faults.
    driver.next = lambda: []

    def stage_then_raise(specs, **_kwargs):
        driver.pending += tuple(spec.uid for spec in specs)
        raise RuntimeError("fault after a later reservation")

    driver.queue_lanes = stage_then_raise
    source = _scheduler_sequence(4)
    batch_gen._unprocessed_sequences.append(source)
    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == [source]
    assert driver.pending_join_uids == (3,)
    assert driver.lane_uids == (1, 2)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("limit", ["lanes", "bytes", "no_bytes"])
def test_dynamic_join_stops_projecting_backlog_at_capacity(monkeypatch, limit):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(*range(1, 21))
    request_uids = {id(request): int(name[4:]) for name, request in requests.items()}
    calls = []
    headroom = {"free": 16 * 1024**3}

    def estimate(request):
        calls.append(request_uids[id(request)])
        return 1024**3

    created = _install_recording_router(
        monkeypatch,
        batch_gen,
        requests,
        mapping,
        estimator=estimate,
        free_bytes=lambda: headroom["free"],
        max_lanes=3 if limit == "lanes" else 8,
    )
    batch_gen.next()
    calls.clear()
    if limit in ("bytes", "no_bytes"):
        headroom["free"] = (9 if limit == "bytes" else 8) * 1024**3
    batch_gen._unprocessed_sequences.extend(
        _scheduler_sequence(uid) for uid in range(3, 21)
    )
    batch_gen.next()
    assert calls == ([] if limit == "no_bytes" else [3])
    assert created[0].pending_join_uids == (() if limit == "no_bytes" else (3,))
    first_queued = 3 if limit == "no_bytes" else 4
    assert [seq[0] for seq in batch_gen._unprocessed_sequences] == list(
        range(first_queued, 21)
    )


@pytest.mark.requires_mlx
def test_singleton_handoff_serializes_concurrent_cancellation(monkeypatch):
    import threading

    batch_gen, _owner, driver = _absorb_setup(monkeypatch)
    requeue = batch_gen._mtp_vendored_requeue_owner
    attempted = threading.Event()
    completed = threading.Event()
    workers = []
    completed_during_handoff = []

    def cancel():
        attempted.set()
        batch_gen.remove([7])
        completed.set()

    def racing_requeue(*, admit):
        def reserve(snapshot):
            accepted = admit(snapshot)
            worker = threading.Thread(target=cancel)
            workers.append(worker)
            worker.start()
            assert attempted.wait(2)
            completed_during_handoff.append(completed.wait(0.1))
            return accepted

        return requeue(admit=reserve)

    batch_gen._mtp_vendored_requeue_owner = racing_requeue
    batch_gen.next()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()
    assert completed_during_handoff == [False]
    assert completed.is_set()
    assert driver.pending_join_uids == ()


@pytest.mark.requires_mlx
def test_join_rollback_does_not_use_live_lane_removal(monkeypatch):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2, 3)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    driver = created[0]
    queue_lanes = driver.queue_lanes

    def stage_then_raise(*args, **kwargs):
        queue_lanes(*args, **kwargs)
        raise RuntimeError("fault after staging")

    def broken_live_removal(_uids):
        raise RuntimeError("live detach unavailable")

    driver.queue_lanes = stage_then_raise
    driver.remove_uids = broken_live_removal
    source = _scheduler_sequence(3)
    batch_gen._unprocessed_sequences.append(source)
    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == [source]
    assert driver.pending_join_uids == ()
    assert driver.lane_uids == (1, 2)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("blocked", ["sampled", "unknown", "processors"])
def test_dynamic_join_never_overtakes_ineligible_fifo_head(monkeypatch, blocked):
    batch_gen = _SchedulerBatchGenerator()
    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(1), _scheduler_sequence(2)]
    )
    requests, mapping = _requests_for(1, 2, 3, 4)
    created = _install_recording_router(monkeypatch, batch_gen, requests, mapping)
    batch_gen.next()
    if blocked == "sampled":
        requests["req-3"].sampling_params.temperature = 0.7
    if blocked == "unknown":
        mapping.pop(3)
    source = [_scheduler_sequence(3), _scheduler_sequence(4)]
    if blocked == "processors":
        row = list(source[0])
        row[6] = [object()]
        source[0] = tuple(row)
    batch_gen._unprocessed_sequences.extend(source)
    batch_gen.next()
    assert list(batch_gen._unprocessed_sequences) == source
    assert created[0].pending_join_uids == ()
    assert created[0].lane_uids == (1, 2)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("operation", ["remove", "close"])
def test_inference_step_does_not_block_cancellation_or_shutdown(monkeypatch, operation):
    import threading

    batch_gen, _owner, driver = _absorb_setup(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors = []

    def blocked_step():
        entered.set()
        assert release.wait(3)
        return []

    def infer():
        try:
            batch_gen.next()
        except Exception as exc:
            errors.append(exc)

    def cleanup():
        try:
            batch_gen.remove([99]) if operation == "remove" else batch_gen.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    driver.next = blocked_step
    inference = threading.Thread(target=infer)
    inference.start()
    assert entered.wait(2)
    canceller = threading.Thread(target=cleanup)
    canceller.start()
    responsive = completed.wait(0.2)
    release.set()
    inference.join(2)
    canceller.join(2)
    assert not inference.is_alive() and not canceller.is_alive()
    assert not errors
    assert responsive, "inference held the handoff ownership lock"
