# First-token contention: production candidate, not qualified

The text scheduler can protect short prompts before any request starts decoding
by applying its existing cost-aware chunk bound whenever multiple prompt rows
can share the next dispatch. This candidate remains **unqualified for merge**:
the shared-host measurements do not satisfy the regression gates, and the
default Qwen routing experiment exposes a different executor bottleneck.

## Change and limits

The existing 500 ms target, 64-token floor, 32-token tiles, recent cost samples
and zero-target escape hatch are reused. The bound only shrinks the
memory-selected chunk, preserves each row's checkpoint grid and releases when
the peer disappears. Queued rows count only if admission capacity permits them.
A completed prompt awaiting promotion still needs first-token protection.

This changes the text scheduler only. It cannot preempt already-dispatched
work, control another scheduler on the same executor, or guarantee end-to-end
latency. Images, video and multimodal prefill are outside this implementation.

## Environment and method

- Baseline: `d127eafa060ca87504d6396b96b97a50860aa971`.
- Candidate serving source: `7fd22a74bbe2a8adf2282bf6c715e041aaf09a1f`;
  archived per-arm scheduler SHA-256 values bind the executed source.
- Studio, Apple M3 Ultra, 256 GiB, macOS 26.5.2, Python 3.12.14;
  MLX 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2, Transformers 5.15.1.
- Qwen3.5 4B, 4-bit, revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3`, existing local cache only.
- Approximately 16K-token long prompt and 294-token templated short prompt;
  different four-digit codes embedded in each prompt, temperature zero,
  thinking/speculation/disk caches off, 32-token output limit.
- Fresh owned server per arm/cycle, warmup, long singleton, then short singleton
  and overlapping long/short pair; clear caches between waves. Both arm orders
  are measured. Production dispatch is traced without overriding its policy.
- **Shared host:** another model evaluation and another test run were active.
  The archived load samples confirm this; these are exploratory measurements.

## Forced text scheduler

Four paired samples per arm, two cycles and two repeats, with `--no-mllm`.
Times below are medians in seconds. Singleton long measurements are two
distinct observations per arm, reused across that cycle's paired repetitions.

| Chunk upper bound | Version | Short TTFT | Long completion | Short singleton TTFT | Long singleton completion |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 | Baseline | 3.499 | 21.426 | 0.406 | 17.494 |
| 512 | Candidate | 2.328 | 23.617 | 0.498 | 17.649 |
| 2048 | Baseline | 13.704 | 26.886 | 0.467 | 16.969 |
| 2048 | Candidate | 3.824 | 22.421 | 0.558 | 19.269 |

At 2048, observed short TTFT decreases 72.1% and long completion decreases
16.6%. At 512, short TTFT decreases 33.5%, but long completion increases
**10.2%**. Uncontended short TTFT varies by 19.6–22.6% even though the policy
leaves that case unchanged, demonstrating substantial measurement variability.
Do not treat these numbers as quiet-host qualification or general model gains.

## Default Qwen routing

Removing `--no-mllm`, retaining an explicit 512-token chunk, gives two paired
samples per arm in opposite orders:

| Version | Short TTFT | Long completion | Long singleton completion |
| --- | ---: | ---: | ---: |
| Baseline | 23.573 | 23.389 | 21.244 |
| Candidate | 21.419 | 21.265 | 21.414 |

All four native-text traces contain only one prompt row and an unchanged
512-token dispatch. The new protection never activates in this experiment;
the timing difference is not evidence of a policy benefit.

The first sole request is pinned to the multimodal singleton path; a later
text peer uses the native text scheduler. Both share the model-owned executor.
The first path loops through its text-prefill chunks inside one call, so
memory chunking does not yield the executor between chunks. Protecting rows
inside the native scheduler cannot unblock that other call. Default routing
or cooperative text-prefill execution needs a separate scoped change.

## Correctness and negative results

- All 20 paired waves across 12 servers pass exact-answer, cold-cache,
  terminal-stream, usage, output-hash and arrival-overlap checks: 72 distinct
  measured streams, excluding 12 warmups.
- HTTP lifecycle checks pass: cold request, exact repeat and shared-prefix
  extension all return the correct code. Both warm cases reuse 5,440 tokens.
  Cancelling a long request while a peer prefills preserves the peer's answer,
  drains active work, returns 404 on a repeated cancel, and permits another
  successful request. This probe explicitly uses the text scheduler.
- Additional Llama 3.2 1B and Qwen3 1.7B arithmetic probes stop on the **baseline**:
  output `23 + 19 = 42` violates the required exact `42` format. Their failed
  runs are retained and supply no cross-model candidate qualification. The
  quality gate was not relaxed to obtain passing results.
- 230 focused tests pass, including actual batch promotion/removal using a tiny
  initialized dense model, scheduler admission, cancellation, recurrent-state
  and disk checkpoints, prefix-boundary behavior and the CLI snapshot.
  The two admitted-peer regressions fail on the clean baseline as expected.

## Evidence and reproduction

The [evidence archive](fixtures/prefill-contention/2026-10-12-production-candidate.tar.gz)
contains results, dispatch traces, shared-host load samples, both rejected
dense-model probes, the lifecycle probe, exact emitting scripts and an inner
SHA-256 manifest. The adjacent manifest hashes the archive. Generated artifacts'
local model-cache prefix is replaced by `${HF_CACHE}`; timing, output and source
hash fields are preserved. Server logs remain private.

The archived runners select two production source trees under
`/private/tmp/rapid-mlx-prefill-contention/{base,worktree}`. Check out the baseline
and candidate commits at those paths and extract the scripts into their parent.
Use an existing cached snapshot and an unused loopback port:

```sh
python benchmark_prefill_first_token.py --model "$MODEL" \
  --output qwen-production/results.json --port 18361 \
  --cycles 2 --repeat 2 --workloads retrieval
python benchmark_prefill_default_routes.py --model "$MODEL" \
  --output qwen-default-routes/results.json --port 18364 \
  --cycles 2 --repeat 1 --workloads retrieval --arms baseline512 candidate512
python validate_evidence.py qwen-production/results.json 16
python validate_evidence.py qwen-default-routes/results.json 4
python check_http_lifecycle.py --model "$MODEL" --output http-lifecycle2 --port 18362
```

Use fresh output directories; the scripts refuse overwriting campaign data.
The archived runners retain unused experimental helper functions inherited
from the benchmark, but neither production comparison invokes them.

## Disposition

Keep the production candidate in draft. Require matched quiet-host runs with
at least 30% median short-TTFT reduction, no more than 10% long-completion or
singleton regression, and passing correctness/lifecycle checks before proposing
enabled behavior. This campaign does not meet those criteria.

Vector owns the default-route executor investigation and later qualification;
Harbor owns an exclusive measurement window; Atlas owns the compatibility and
default disposition. Do not expand this text-scheduler diff into multimodal
execution or routing changes implicitly.
