# Final frequent-workload speed search

Neither candidate qualified. The combined prototype reduced measured CPU work
but changed actual request latency by approximately zero: Python was 0.06% slower
and SQL 0.38% slower. All four answers per workload were identical. The owner's
greater-than-10% admission threshold was not met; the overall search is closed
with no new serving PR or queue entry.

Owner: Vector. The owner requested at most two additional candidates and closure
of the overall search if neither qualifies. Admission remains a measured
greater-than-10% improvement in actual frequent-workload request latency, with
correctness preserved. Baseline: `5c70e238f6068d373553c48d92786a5cf333d05b`.
Branch: `vector/frequent-speed-search-final`. This is research evidence;
production source files were not changed.

## The last two candidates

1. **Exact repetition-guard rejection filter.** Before testing every possible
   primitive period in the bounded token history, reject a proposed period when
   its last copy differs from its first. Thresholds, period order, intervention
   decisions and hard-stop behavior stay intact. The CPU screen compared 11,632
   baseline/candidate decisions, including 2,581 nonempty reference results,
   with no differences. On synthetic bounded histories, the existing function
   cost up to 458 microseconds per call; the largest measured saving was 374
   microseconds. Those are component CPU measurements, not serving gains.
2. **Avoid full-prefix output decoding.** The live native text scheduler uses
   `utils.decode.IncrementalDecoder`, which decodes its growing token history
   each step. A private prototype uses the installed tokenizer's streaming
   decoder, caches a fully decoded stop surface, and retains the original
   fallback for incomplete/invalid UTF-8 and replacement-character boundaries.
   It leaves the authoritative raw stop surface distinct from emitted text.

Correction to the preceding screening round: the presence of a per-request
detokenizer pool did not establish that the native text scheduler's live path
was incremental in computational cost. Its request admission and output handling
still use the full-prefix decoder. The multimodal scheduler uses a different
detokenizer path. The second candidate was investigated rather than dismissed
from that earlier source-reading error.

## CPU compatibility and component screen

The controller-only screen used Apple M3 Ultra, Python 3.12.14, cached Qwen 27B
and Gemma E4B **tokenizers only**. It did not load or benchmark either model.
Full dependency and tokenizer identities are in the evidence. The boundary
checks and native serving measurements used different dependency versions,
listed below; they are separate evidence sets.

The direct streaming decoder matched seven ordinary text/code/Unicode/marker
cases per tokenizer, but differed from the existing emitted surface on
replacement-character boundaries. Deterministic adversarial token sequences and
both concatenated and raw decoded surfaces are retained; a changed chunk boundary
alone is not classified as a final-answer corruption. The compatibility prototype
then passed 17,500 steps across 16 case/settings, including both special-token
policies and reset, checking every delta, visible prefix and raw stop surface.
This is bounded coverage for two tokenizer families, not universal enrollment
approval or a full lifecycle qualification.

Median cumulative CPU seconds for one synthetic generated sequence, four samples
per arm with alternating order:

| Tokenizer | Tokens | Full-prefix decoder | Direct native streaming decoder |
| --- | ---: | ---: | ---: |
| Qwen 27B | 512 | 0.0196095 | 0.0004765 |
| Qwen 27B | 2048 | 0.2997895 | 0.0021820 |
| Qwen 27B | 8192 | 4.6998230 | 0.0192770 |
| Gemma E4B | 512 | 0.0408750 | 0.0007515 |
| Gemma E4B | 2048 | 0.6573695 | 0.0033915 |
| Gemma E4B | 8192 | 10.7746175 | 0.0243770 |

These timings describe the initial component screen on sequences whose deltas
matched, before adding the compatibility wrapper. They must not be presented as
model throughput or as the measured serving speed of the eventual prototype.

## Actual serving attribution and comparison

One Apple M4 Pro, 48 GiB, ran cached Qwen3.5-4B-MLX-4bit revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`. The native text route was selected
explicitly with `--no-mllm`; instrumentation confirmed its activation. Speculative
decoding, prefix caching, disk caches and model downloads were disabled in both
arms. Temperature was zero, thinking disabled. Requests declared a read-file tool,
instructed the model to answer without calling it, and supplied a stop string, so
both exact loop protection and raw stop-surface decoding were exercised.

A baseline-only attribution run generated a warm-up and 512/2048/4096-token code
responses. Instrumented call wall times, including measurement overhead:

| Output tokens | Request seconds | Guard call seconds | Decoder call seconds |
| --- | ---: | ---: | ---: |
| 512 | 6.867916 | 0.368867 | 0.068377 |
| 2048 | 25.350070 | 2.050459 | 0.988806 |
| 4096 | 50.621304 | 3.949977 | 3.496930 |

At 4096 tokens the two call groups together represented about 14.7% of elapsed
time. This justified a serving comparison of the combined prototype. That sum
is not an achievable latency reduction: host work can overlap GPU execution.

Four fresh servers ran in baseline/candidate/candidate/baseline order. Each ran
a short warm-up, then a Python async-job-runner module/tests request and a
PostgreSQL booking-service migration/functions/reporting request. Each measured
request had a 4096-token limit; natural EOS was allowed. Both arms used the same
instrumentation. This is one bounded run per workload per server, not a statistical
confidence interval.

| Workload | Tokens | Baseline median seconds | Candidate median seconds | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| python_service | 4096 | 50.986179 | 51.019248 | -0.06% |
| sql_migration | 4096 | 51.019503 | 51.212934 | -0.38% |

Negative reduction means slower. Both tasks reached the 4096-token limit in
every trial; generated-module functionality or migration correctness was not
independently tested. The correctness assertion here is unchanged output, usage
and finish behavior across the two arms, plus the separate decoder boundary
checks. Shorter requests and other machines were not performance-qualified.

All four servers exited cleanly. Eight measured comparison requests and four
comparison warm-ups completed; the attribution run adds three measured requests
and one warm-up, for 16 HTTP streams in total. Usage and answer hashes match for
all four trials of each workload. Candidate counters confirm that native streaming
was active for the generated tokens and that the guard functions were called.
No other large Python process appeared in the host samples. The worker runtime
was Python 3.12.13, MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers
5.15.1 and tokenizers 0.22.2. The controller's separate boundary/component checks
used Transformers 5.16.1 and tokenizers 0.23.1; those checks do not establish full
worker-runtime boundary coverage.

Across the two natural workloads and warm-up, baseline decoder call wall times
were about 7.8–8.0 seconds per server, versus about 0.11 seconds for the candidate;
guard call times fell from about 8.05 to 1.80–1.82 seconds. Despite those large
component reductions, overall request time stayed approximately unchanged. This
is consistent with GPU-bound execution and host/GPU overlap; no GPU trace was
collected to isolate that mechanism. Component speed is therefore not substituted
for user-visible latency evidence.

Source/harness hashes, the cached model and tokenizer file identities, and 1,616
native dependency Python files were verified after measurement. Offline evidence
checks recomputed activation, request counts, answer/usage equality and latency
reductions. Mutations that changed an answer hash or removed fast-path activation
were rejected; the original evidence was restored and passed again.

## Reproduction and disposition

The [fixture manifest](fixtures/2026-10-15-final-speed-search/manifest.json) binds
the accompanying `evidence.tar.gz`. The fixture contains executed CPU and HTTP harnesses,
raw per-request results and activation counters, host samples, source/runtime/
model identities, compatibility checks and the cleanup receipt. Raw receipts and
failed direct-decoder boundary checks are retained rather than silently omitted.

For CPU reproduction, place this baseline at
`/private/tmp/rapid-mlx-speed-search-final/worktree`, unpack the harnesses in its
parent, and use the cached tokenizer revisions recorded in `decode-screen.json`:

```sh
python3.12 guard_screen.py
python3.12 decode_screen.py
python3.12 decode_boundaries.py
python3.12 check_fast_decode.py
```

For remote reproduction, obtain a fresh authorized lease, use that same task
root with source under `src`, create `.task-owner` with the exact contents
`vector-speed-search-final-v1`, and provide the recorded dependencies and cached
4B snapshot in the standard model cache. Execute serially:

```sh
python run_profile.py
python driver_compare.py
python metadata_http.py
```

The last command verifies source and harness identities, dependency Python files,
and the expected model/tokenizer file hashes after timing. Artifact hashes bind
evidence identity; they do not independently prove physical execution. Preserve
shared runtime/cache state, clean only the task's own processes and scratch, and
release its lease immediately after collection.

The overall search is finished at the owner's requested stopping point. These
last two candidates, including their combined serving experiment, are rejected
for the requested speed objective. This is a bounded prioritization decision,
not a proof that no optimization can exist on any model or Apple generation.
There is no follow-up candidate, outstanding benchmark, serving PR or reserved
worker from this search. The owned M4 processes and scratch were removed and its
lease released before offline reporting; shared runtimes and model caches were
preserved. Raw private receipts and the reference trail are archived separately
without lease credentials.
