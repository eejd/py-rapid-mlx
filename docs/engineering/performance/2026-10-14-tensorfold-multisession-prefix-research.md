# Qwen 27B multi-session prefix selection

Date: 2026-10-14. Owner: Vector. Research status: a bounded checkpoint-retention
candidate is reproduced; no serving change or inference speedup is qualified.

## Finding

Two conversations with the same long system instructions but different final
session labels can recompute a large common prefix. The runtime has already
computed a usable intermediate chunk boundary, but the adapter's selected
checkpoint positions can omit it. Retaining one additional existing boundary
is a smaller candidate than changing the prefill chunk plan or cache allocator.

This is distinct from #4626. That correction enables message-aware chunks and
fixes repeated histories below the old 2048-token grid. This investigation uses
that correction on main and studies reuse across different conversations.

## Method and limits

The [probe and observations](fixtures/2026-10-14-tensorfold-prefix-research/)
use the actual qualified tokenizer, the adapter's `_prompt_boundaries`, and
the pinned runtime's `PrefillPlan`, `choose_checkpoints`, and `CheckpointStore`.
Cache payloads are dummy dictionaries. The probe follows the scheduler's
checkpoint selection and insertion rules; it does not execute the scheduler,
model forwards, tensor snapshot copies, memory admission, HTTP generation, or
actual cancellation. `cached_tokens` below means the stored prefix selected by
this probe, not a measured server usage counter.

- Rapid baseline: `7ac7ad58bc8cabe609c4c5ee737707b5ff8f07bc` (the complete
  revision is also recorded in the observations; that record is authoritative).
- TensorFold 0.6.6 source: `cb2ebf0540f42604e2759b2ddef497861e928248`.
- Qualified target: `Vontra/Qwen3.8-27B-MLX-4bit`, snapshot
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Python and dependency versions, tokenizer/configuration hashes, and the
  message-marker plan name are recorded in the JSON fixture.
- All tokens came from the existing standard HF cache. No model was downloaded
  or loaded. No TTFT, throughput, physical RAM, or generated-output result is
  claimed. Shared compute and the host model lock were occupied when checked;
  the investigation did not interrupt those workloads.

The runtime's own tokenizer loader matters: a raw Transformers tokenizer in
this environment returns a mapping for some template calls, whereas the
runtime wrapper returns token IDs. An exploratory raw-tokenizer run failed
marker detection and was excluded. The published probe requires nonempty
message markers and checks the runtime source revision and tracked source diff.

## Existing behavior that works

With a 930-token synthetic request, a repeat selects 911 cached tokens. The
tested follow-up and two sibling branches also select 911 tokens. This proves
reuse of the shared system prefix in these fixtures; it does not prove complete
reuse of generated replies or arbitrary branching histories.

Cache copying already shares whole MLX buffers and copies views when needed;
building another sharing mechanism is not justified by this investigation.

The current Rapid TensorFold HTTP adapter refuses nonempty tool definitions
with HTTP 400. A suspected tools/rendering mismatch is therefore outside its
supported request path, not a demonstrated production cache regression.
Enabling tool calls is separate functionality, not a prerequisite to this
retention candidate. The existing backend/HTTP suites passed: **42 tests**.

## Miss pattern and counterfactual

Each pair uses the same repeated synthetic reference text. Only the final
system label (`red` / `blue`) and short user question differ. A cold request
populates the store before the second conversation probes it.

| Prompt tokens | Common prefix tokens | Current selected cache | Counterfactual selected cache |
| ---: | ---: | ---: | ---: |
| 514 | 493 | 0 | 0 |
| 934 | 913 | 0 | 0 |
| 1,534 | 1,513 | 0 | 0 |
| 2,534 | 2,513 | 0 | 2,048 |
| 4,534 | 4,513 | 2,048 | 4,096 |
| 8,534 | 8,513 | 6,144 | 8,192 |

For the 2,534-token case, chunk starts include 2048 and 2517. The original
request stores only 2517; that checkpoint includes the differing label and
cannot match the other conversation. The common 2048-token state was computed
but not retained. Similar rounding omits a later usable grid boundary in the
4,534- and 8,534-token cases.

The counterfactual adds one checkpoint candidate at the existing system-length
estimate minus the plan's existing 256-token minimum spacing. The runtime still
rounds it to an existing chunk start. Both requests retain exactly the same
chunk-start lists as the baseline. The affected rows add one logical cache
entry. The probe verifies that every selected hit remains within the exact
common token prefix, and that an early system-instruction edit still misses.

This is an experimental selection policy, not a proposed new default. It does
not solve short shared prefixes without an interior chunk boundary. It also
does not establish that more snapshots are affordable or improve end-to-end
latency: real recurrent-state snapshots and subsequent eviction must be measured.

A diagnostic control keeps the system text identical and puts the differing
label in the user message. It selects 911 of 934 tokens, for example. Moving
instructions between roles changes semantics; this control isolates token
locality and must not become an automatic request rewrite.

## Capacity behavior

The default store has eight ordinary entry slots plus three pinned system
entries. Sixteen distinct synthetic system groups followed by a revisit to
the first produce a miss, eleven remaining entries and six evictions. This
is expected bounded capacity, not evidence of a leak. The probe uses no byte
sizer and cannot qualify physical-memory pressure or the default byte budget.

The store already favors evicting older checkpoints superseded by a later
conversation prefix. That tradeoff can make old forks cold under pressure;
changing it requires evidence from an actual branching workload rather than
assuming every old boundary should remain pinned.

## Reproduction

Use the existing dependency environment and an unmodified checkout of the
qualified runtime source. The probe reads only tokenizer/configuration files
from the standard HF cache and refuses a missing target snapshot.

```sh
git clone https://github.com/ashhart/TensorFold.git /private/tmp/rapid-mlx-prefix-runtime
git -C /private/tmp/rapid-mlx-prefix-runtime checkout cb2ebf0540f42604e2759b2ddef497861e928248
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  docs/engineering/performance/fixtures/2026-10-14-tensorfold-prefix-research/probe.py \
  --runtime-source /private/tmp/rapid-mlx-prefix-runtime > observations.json
python -m pytest tests/test_tensorfold_qwen27_backend.py \
  tests/test_tensorfold_qwen27_http.py -q
```

## Next qualification and handoff

Tracking: #4643. Vector owns the narrow retention candidate. Harbor coordinates
an exclusive model window before latency qualification; Atlas owns any decision
to change supported request/model scope. Pixel and Echo have the research FYI
in the tracking issue; no Desktop or protocol change is required here.

Before a serving PR, compare the current adapter with an implementation that
retains the omitted existing boundary. Use fresh processes, cold/repeat/sibling
and early-edit controls, forward/reverse ordering, and the pinned real model.
Verify exact greedy and seeded generated token IDs, HTTP terminal/usage behavior,
cancellation, cache byte bounds, snapshot-copy cost, and cold-request regressions.
Measure cache restores and suffix work directly in the actual scheduler. Include
pressure from unrelated sessions to detect whether the extra snapshot makes
overall reuse worse. Keep the current chunk plan and model eligibility unchanged.

No physical block-sharing rewrite, disk persistence, tool support, or new model
qualification is justified by these selection-only results.
