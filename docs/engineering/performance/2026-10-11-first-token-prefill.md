# Prefill interference before the first output token

Date: 2026-10-11. Owner: Vector. Research experiment; normal serving code and
operator defaults remain unchanged. The candidate exists only in a benchmark
child process. This follows the earlier cache/concurrency study in commit
`d842c9c59` and isolates the period before a short request starts generating.

## Question and candidate

The current cost-aware decode bound activates only when the generation batch
has a row. A short request sharing a prompt batch with a long prefill can
therefore spend multiple full-size chunks waiting before that bound activates.

The installed prompt runtime advances one segment per row per scheduler tick,
right-pads the rows to the longest input in that tick, evaluates the prompt
cache, and then returns. A message-boundary snapshot can split even a short
prompt into a boundary segment and a template tail. Finishing the prompt also
requires a promotion tick before the first generation response is returned.
Long neighbours continue their prefill during these transitions.

The candidate first applies the existing memory and decode bounds. When two
or more prompt rows can share the next tick, it further limits their per-row
chunk using the slowest recent measured seconds per prompt token and a total
500 ms budget divided across those rows. The allowance is rounded down to
32-token tiles with a 64-token floor, then clipped to preserve grid boundaries
for every current row. Without cost samples, the total allowance is 256 row
tokens. A single uncontended prompt retains its existing configured size.

A row awaiting promotion is included. After promotion, the normal decode bound
owns protection. A queued row counts only if it has an admission slot. This
prototype does not reorder requests, pause a running GPU operation, or solve
admission saturation. Higher-concurrency engine step coalescing is also outside
this two-request experiment.

## Environment and method

- Apple M3 Ultra Mac Studio, 256 GiB unified memory, macOS 26.5.2.
- Serving and harness source: `1b8f137ca`; serving library matches base
  `87a14c382`. Python 3.12.14, MLX 0.32.3, mlx-lm 0.31.3,
  Transformers 5.15.1. Exact installed versions are recorded with results.
- Existing cached `mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot
  `32f3e8ecf65426fc3306969496342d504bfa13f3`; no weight downloads.
- Text-only OpenAI streaming route, temperature zero, thinking explicitly off,
  no speculation or disk caches. Prefix caching stays enabled, with explicit
  cache clearing and idle checks between every cold wave. Model compilation,
  allocator state and online cost samples are not reset by clearing prefix data.
- Other training and test jobs remain running. This is a shared-host experiment,
  not an exclusive hardware benchmark. Absolute timings must not be compared
  directly to the previous study.

Three arms use the same requests: fixed 2048, fixed 512 and candidate 2048.
Each arm owns a fresh loopback child server and warms it before measurements.
A forward pass and a reverse pass each contain two repetitions per workload,
for four paired samples per arm/workload. Each pass measures one long request alone per workload, and a short request
alone before each paired trial. Long-solo medians therefore have two independent
process-level samples; paired metrics have four samples. A contention wave dispatches the long
request, waits until status reports a running request, and dispatches the short
request. That observation does not fix its prefill progress at arrival.

The prose workload uses approximately 16K/256-token independent code-record
prompts and a 96-token output budget. The retrieval workload inserts a four-digit
archive code halfway through each prompt and requests only that code, with a
32-token budget. The long code is 7319 and the short code is 9283, making an
accidental cross-row answer fail validation. The exact rendered lengths are in
API usage. Retrieval must terminate with `stop` and match the expected answer;
prose may terminate at its budget and is not a semantic correctness test.

Each measured stream must have visible output, positive usage, a finish reason
and `[DONE]`. Every solo and contended measurement must report zero cached
tokens. Client TTFT runs from HTTP dispatch to first nonempty content/reasoning;
role frames are excluded. Completion time includes prefill and decode. Scheduler
traces use the same machine's monotonic clock to connect client timing to
individual dispatches; all arms use identical trace instrumentation.

## Results

All 24 paired waves completed, with six clean child-server shutdowns. All 84
measured streams and six warmups completed their protocol checks. The 42 distinct
retrieval streams (solo and paired) returned the expected code and stopped
normally. Each arm/workload has four paired observations; the table reports
medians, not tail percentiles.

| Workload | Arm | Short TTFT | Long completion | Long alone completion |
| --- | --- | ---: | ---: | ---: |
| Prose, 96-token budget | Fixed 2048 | 14.472 s | 41.743 s | 23.942 s |
| Prose, 96-token budget | Fixed 512 | 3.868 s | 40.528 s | 23.609 s |
| Prose, 96-token budget | Candidate 2048 | 3.911 s | 41.520 s | 19.811 s |
| Code retrieval | Fixed 2048 | 10.947 s | 21.166 s | 17.251 s |
| Code retrieval | Fixed 512 | 4.031 s | 28.866 s | 18.523 s |
| Code retrieval | Candidate 2048 | 4.011 s | 23.952 s | 21.290 s |

Relative to fixed 2048, the candidate lowers short TTFT by 73.0% for prose and
63.4% for retrieval. It reaches approximately the fixed-512 interactive latency
in these workloads while preserving 2048 for uncontended prompts. This is an
interactive latency result, not an end-to-end acceleration claim.

Long completion changes by -0.5% for prose and +13.2% for retrieval. The latter
is a measured regression in this campaign and must not be omitted. Single-long
baselines also vary substantially across arms despite the candidate leaving
single-prompt chunk dispatch unchanged. Host contention and measurement order
remain confounders; this campaign does not isolate how much of the long-request
change comes from scheduling versus background work.

Short TTFT samples in milliseconds, in forward then reverse order:

| Workload | Fixed 2048 | Fixed 512 | Candidate 2048 |
| --- | --- | --- | --- |
| Prose | 12958, 13274, 15671, 16988 | 3102, 3115, 4622, 4817 | 4198, 3624, 4376, 3105 |
| Retrieval | 9470, 9876, 22201, 12018 | 2592, 3834, 4227, 5193 | 4037, 3986, 3898, 4398 |

The pre-first-token dispatch windows provide the causal evidence. For example,
in the first fixed-2048 prose wave, an already-dispatched chunk overlaps the
short request's arrival. The following two-row dispatches take approximately
4.16 s and 4.23 s, and the promotion dispatch takes 2.36 s, all at chunk 2048
before any row is decoding. Candidate traces show extra bounds with
`decoding == 0`, followed by the existing decode bound after promotion. The
first candidate retrieval wave uses 192, 192, 128 and 64 tokens in this
pre-decode interval; those dispatches take about 474, 530, 441 and 282 ms.
An initial unshared 2048-token dispatch still contributes approximately 1.92 s
of waiting in that wave. Full and extracted windows are archived.

Prose outputs do not match their singleton output hashes in any paired trial
for any arm (0/4 for both long and short requests in each arm). Retrieval outputs
match their expected answers and singleton hashes in all trials. The prose
variation therefore predates the candidate, but this experiment does not prove
its cause or general semantic correctness.

Verification: 55 focused tests passed, including seven harness contracts and
existing memory/admission/decode protections. Python/JSON parsing, artifact
hashes, bounded dispatch traces, cold-cache usage, stream completion, output
hashes and retrieval answers were checked. The experiment listener is closed.

## Limits and next action

Earlier short-request promotion also activates the existing decode protection
earlier in the long prompt. More of that long prompt can consequently run under
a reduced chunk budget. Lower short TTFT does not imply unchanged long-request
completion time, even without background contention.

The first chunk dispatched before a peer is observed cannot be preempted. The
500 ms compute estimate is not a first-token deadline or a hard per-chunk
latency guarantee. Context length, padded batch shape and competing GPU work
change the cost; even the minimum chunk can exceed the target. Counting a row
about to promote is conservative because the prompt batch shrinks inside the
runtime call.

The retrieval task tests limited context retention and row separation; it does
not establish general answer quality, coding correctness, logits/token equality,
or compatibility across all recurrent model families. Prose output hashes are
recorded for diagnosis, not treated as a quality gate. No tool-call, multimodal,
Anthropic-route, speculative, saturated-admission or large-MoE qualification is
claimed.

Vector's next experiment should compare cost targets and short-segment-aware
budgets on a quiet host, retaining both current fixed controls. Track long
completion explicitly and investigate the retrieval regression before proposing
a default change. A production design should integrate a separately opt-in
first-token contention bound with the existing memory/decode mechanisms only
after broader model, cache-history and cancellation coverage. Atlas owns any
later default or compatibility decision; Harbor owns a quiet-host qualification
run if stronger latency/performance claims are needed.

## Reproduction

Use the source revision and runtime versions above, the existing local model
snapshot, and an unused loopback port. No model downloads are required. The
output path must be new so previous evidence cannot be overwritten.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1 \
python scripts/benchmark_prefill_first_token.py \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
  --output /private/tmp/rapid-mlx-first-token-prefill/results.json \
  --port 18348 --repeat 2 --cycles 2
```

Focused verification:

```bash
python -m pytest tests/test_benchmark_prefill_first_token.py \
  tests/test_adaptive_prefill.py tests/test_scheduler_admission_policy.py -q
python -m ruff check scripts/benchmark_prefill_first_token.py \
  tests/test_benchmark_prefill_first_token.py
PYTHONPATH=. python docs/engineering/performance/fixtures/2026-10-11-first-token-prefill/validate.py \
  docs/engineering/performance/fixtures/2026-10-11-first-token-prefill
```

Raw results, full scheduler traces, environment and artifact hashes are in
[the evidence directory](fixtures/2026-10-11-first-token-prefill/). Absolute machine paths in archived child commands are normalized to placeholders;
the original command hashes are retained. Scheduler traces are unchanged.
