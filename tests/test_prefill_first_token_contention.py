# SPDX-License-Identifier: Apache-2.0
"""First-token protection through the production prefill dispatch policy."""

from collections import deque

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from tests.test_adaptive_prefill import _scheduler


def _candidate(*, queued=0, prefill_slots=8, completion_slots=8, samples=(0.001,)):
    scheduler = _scheduler(prompt_tokens=16_000)
    bg = scheduler.batch_generator
    bg._generation_batch = []
    bg.prefill_batch_size = prefill_slots
    bg.completion_batch_size = completion_slots
    bg._unprocessed_sequences = deque([object()] * queued)
    scheduler._prefill_cost_samples = deque(samples, maxlen=4)
    return scheduler


@pytest.mark.parametrize("queued", [1, 20])
def test_admissible_peer_protects_first_token_before_decode(queued):
    scheduler = _candidate(queued=queued, prefill_slots=2)
    assert scheduler._apply_adaptive_prefill_size() == 224
    assert scheduler.batch_generator._prompt_batch.prefill_step_size == 224
    assert scheduler._last_adaptive_prefill_size == 2048


@pytest.mark.parametrize("prefill_slots,completion_slots", [(1, 8), (8, 0)])
def test_ineligible_queue_does_not_shrink_singleton(prefill_slots, completion_slots):
    scheduler = _candidate(
        queued=10, prefill_slots=prefill_slots, completion_slots=completion_slots
    )
    assert scheduler._apply_adaptive_prefill_size() == 2048


def test_completed_prompt_keeps_protection_until_promotion():
    scheduler = _candidate()
    bg = scheduler.batch_generator
    bg._currently_processing.append([[[1]], 293, 294])
    assert scheduler._apply_adaptive_prefill_size() == 224
    # Once promoted, the existing decode protection takes over.
    bg._currently_processing.pop()
    bg._generation_batch = [object()]
    assert scheduler._apply_adaptive_prefill_size() == 480
    bg._generation_batch.clear()
    assert scheduler._apply_adaptive_prefill_size() == 2048


@pytest.mark.parametrize("peer_location", ["processing", "queued"])
def test_removing_peer_releases_bound_without_memory_hysteresis(peer_location):
    scheduler = _candidate()
    bg = scheduler.batch_generator
    peers = (
        bg._currently_processing
        if peer_location == "processing"
        else bg._unprocessed_sequences
    )
    peers.append([[[1, 2]], 0, 2])
    assert scheduler._apply_adaptive_prefill_size() == 224
    peers.pop()
    assert scheduler._apply_adaptive_prefill_size() == 2048
    assert scheduler._decode_stall_chunk_size is None


def test_contention_preserves_each_cached_suffix_checkpoint_grid():
    scheduler = _candidate()
    bg = scheduler.batch_generator
    bg._currently_processing[0][1] = 2008
    bg._currently_processing.append([[[1, 2]], 100, 16_000])
    visited = [set(), set()]
    assert scheduler._apply_adaptive_prefill_size() == 40
    for _ in range(120):
        chunk = scheduler._apply_adaptive_prefill_size()
        for row, seen in zip(bg._currently_processing, visited):
            row[1] += chunk
            seen.add(row[1])
    for seen in visited:
        assert {2048, 4096, 6144} <= seen


@pytest.mark.parametrize("selected", [32, 128, 256, 512])
def test_contention_only_shrinks_memory_selection(selected):
    scheduler = _candidate(queued=1, samples=(0.1,))
    scheduler._select_adaptive_prefill_size = lambda: selected
    assert scheduler._apply_adaptive_prefill_size() == min(selected, 64)


def test_first_token_bound_uses_existing_zero_target_escape_hatch():
    scheduler = _candidate(queued=1)
    scheduler.config.decode_stall_target_ms = 0
    assert scheduler._apply_adaptive_prefill_size() == 2048


def test_first_token_bound_is_independent_of_memory_policy_toggle():
    scheduler = _candidate(queued=1, samples=())
    scheduler.config.adaptive_prefill = False
    assert scheduler._apply_adaptive_prefill_size() == 128


def test_decode_budget_counts_only_queue_entries_with_completion_capacity():
    scheduler = _candidate(queued=10, prefill_slots=8, completion_slots=2)
    scheduler.batch_generator._generation_batch = [object()]
    assert scheduler._apply_adaptive_prefill_size() == 224  # One admitted peer.


@pytest.fixture
def real_batch():
    """Exercise actual cache/batch transitions without downloading weights."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.llama import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=16,
        )
    )
    mx.eval(model.parameters())
    scheduler = _candidate()
    scheduler.batch_generator = BatchGenerator(
        model,
        prefill_step_size=2048,
        prefill_batch_size=2,
        completion_batch_size=4,
        max_tokens=4,
    )
    try:
        yield scheduler
    finally:
        scheduler.batch_generator.close()


def test_real_batch_returns_segmented_peer_before_long_prefill_finishes(real_batch):
    scheduler = real_batch
    bg = scheduler.batch_generator
    long_uid, short_uid = bg.insert_segments(
        [
            [[1] * 8191, [2]],
            [[3] * 65, [4] * 16, [5]],
        ]
    )
    short_tokens = []
    for _ in range(12):
        scheduler._apply_adaptive_prefill_size()
        _, generated = bg.next()
        short_tokens.extend(r.token for r in generated if r.uid == short_uid)
        if len(short_tokens) == 4:
            break
    assert len(short_tokens) == 4
    assert long_uid in bg._prompt_batch.uids
    assert short_uid not in bg._prompt_batch.uids
    assert short_uid not in bg._generation_batch.uids
    assert scheduler._apply_adaptive_prefill_size() == 2048


@pytest.mark.parametrize("admitted", [False, True])
def test_real_batch_removal_releases_first_token_bound(real_batch, admitted):
    scheduler = real_batch
    bg = scheduler.batch_generator
    long_uid, cancelled_uid = bg.insert([[1] * 8192, [2] * 4096])
    assert scheduler._apply_adaptive_prefill_size() == 224
    if admitted:
        bg.next()
    bg.remove([cancelled_uid])
    assert scheduler._apply_adaptive_prefill_size() == 2048
    _, generated = bg.next()
    assert all(r.uid != cancelled_uid for r in generated)
    assert long_uid in bg._prompt_batch.uids
