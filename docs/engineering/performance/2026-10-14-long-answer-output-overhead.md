# Long-answer output overhead on M4 Pro

Date: 2026-10-14. Owner: Vector. Tracking: #4645.

## Result and disposition

The ordinary default serving path already shares its cumulative token history
instead of copying it for every token. On the tested M4 Pro, its scheduler's
response processing consumed about 0.65 CPU seconds during a 4096-token answer
whose first-content-to-completion interval was about 50.35 seconds. This does
not identify a large general decoding-speed opportunity in response formatting.

A separate serial adapter still copies every cumulative token history and
retains every intermediate result. Replaying the same real tokens through that
adapter reproduces quadratic retention. An experimental terminal-only policy
reduces measured Python allocation peaks substantially. Removing intermediate
history copies also reduces replay CPU time, but the absolute difference is
milliseconds, not seconds. No end-to-end generation speedup was measured.

Do not prioritize this as a universal decoding acceleration PR. A bounded
adapter-memory cleanup may be useful separately, after its diagnostic and
error/cancellation contracts are reviewed. No product behavior changed here.

## Source, environment and hardware

- Rapid source: `6d617f51c` (full revision and measured module hashes in the
  evidence manifest and metadata).
- Apple M4 Pro, 48 GB, 12 CPU cores (8 performance / 4 efficiency), arm64,
  macOS 26.5.1, Python 3.12.13.
- MLX 0.32.3; MLX-LM 0.31.3; MLX-VLM 0.7.2; Transformers 5.15.1;
  NumPy 2.4.4; httpx 0.28.1; psutil 7.2.2.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot
  `32f3e8ecf65426fc3306969496342d504bfa13f3`. Individual model/configuration
  hashes are included. The worker's standard HF cache did not contain it;
  the official downloader fetched approximately 3.06 GB into that cache.
- Adapter replay imports TensorFold source at the qualified revision
  `cb2ebf0540f42604e2759b2ddef497861e928248`. It does not load a TensorFold
  model or execute its inference engine.

The task used an isolated environment inheriting existing read-only dependency
packages. The inherited `rapid-mlx` distribution metadata reports 0.15.7; the
measured modules were loaded from the archived source checkout through
`PYTHONPATH`, with hashes verified against that checkout. This is source-level
research, not packaged-release acceptance.

The M4 had an exclusive lease and the existing shared model lock. Across 165
one-second native-serving observations, no other large Python process was
present. Preexisting swap remained 195,493,888 bytes throughout; this was not a
zero-swap experiment. Sampled server RSS peaked at 3,981,033,472 bytes.

## Ordinary serving experiment

The server used the default text-compatible route with `--no-spec-decode`,
disk caches and model downloads disabled during serving, and thinking off.
There was no draft model or MTP. The observed scheduler lane was `mllm`, the
default hybrid text compatibility path for this checkpoint, not a forced
alternate backend.

The synthetic request asks for many explicit Python function definitions.
One 32-token warmup precedes fixed output budgets of 512, 2048, 4096, followed
by reverse order 4096, 2048, 512. All six measured responses reached their
requested budget with `finish_reason=length`, usage and DONE. Both runs of
each length produced the same visible-text hash. Prompt length was 94 tokens;
measured requests reused 79 prompt tokens, isolating sustained decode from
repeated prefill. These intentionally capped code fragments are not a semantic
or executable-code quality evaluation.

A small wrapper records thread CPU and wall time around the existing
`_process_batch_responses` method and captures native token IDs on completion.
It does not change sampling, scheduling or output. CPU timing is diagnostic:
it is not an independent uninstrumented A/B result.

| Output tokens | Median decode interval | Tokens/s | Response-processing CPU time |
| ---: | ---: | ---: | ---: |
| 512 | 6.185 s | 82.62 | 0.052 s |
| 2,048 | 24.880 s | 82.27 | 0.299 s |
| 4,096 | 50.355 s | 81.32 | 0.653 s |

Decode interval starts at the client's first nonempty content and ends when
the stream completes. Rate uses `(completion_tokens - 1) / interval`. Values
are medians of two observations per length, not population estimates.

The history-list identity remained one object per request, confirming the
existing live-reference implementation. The response CPU/wall ratio is around
1.3% for 4096 tokens. Since CPU work can overlap device work, this ratio is not
a prediction of attainable speedup. Whole-server CPU time includes model and
runtime work and must not be confused with response-processing CPU time.

## Adapter replay experiment

The captured native token IDs are replayed one token per chunk through the
actual `TensorFoldRequestProvider.stream_generate` implementation, using the
same tokenizer and real stopping/detokenization code. A fake scheduler supplies
the captured tokens; no model or speculative decoding executes in this phase.

Three policies run in fresh children:

1. **Current:** copy cumulative token IDs and retain all intermediate records.
2. **Terminal only:** retain only the final diagnostic record; intermediate
   cumulative token IDs are still copied while each event is consumed.
3. **Linear history:** terminal-only retention plus a live token-history
   reference for intermediate records, as the ordinary scheduler already uses.

The experimental variants are temporary source transformations inside the
replay child, not changes to the serving checkout. They intentionally change
the private `last_outputs` diagnostic history and intermediate object lifetime;
they are not drop-in production patches.

Two fresh children per length/policy measure CPU and wall time, with policy
and length order reversed on the second pass. A separate fresh child per cell
measures Python allocations with tracemalloc. Its timings are excluded from
the CPU comparison. There are 27 children in total.

| Output tokens | Current Python allocation peak | Terminal-only allocation peak |
| ---: | ---: | ---: |
| 512 | 2.20 MiB | 0.79 MiB |
| 2,048 | 23.04 MiB | 0.79 MiB |
| 4,096 | 88.30 MiB | 0.91 MiB |

At 4096 tokens, current behavior retains 4097 records and 8,394,752 history
items. The terminal-only variants retain one record and 4096 history items.
The traced peak measures Python allocations during replay, not model KV
memory, total application RSS, or a live TensorFold server's peak footprint.
Process lifetime RSS peaks also include imports, so they are not used to
attribute the retention improvement.

4096-token median replay CPU times were 49.6 ms (current), 36.0 ms (terminal
only), and 23.5 ms (linear history). The small sample and first-call/import
effects limit precision, especially at short lengths. These replay timings
cannot be subtracted from the ordinary serving time: they are different paths
and there was no live serving A/B of the experimental policies.

Every child reconstructed the same native token sequence and visible-text
hash as its captured ordinary response, and retained correct final IDs, text
and completion metadata. This covers the synthetic code fixture only; it does
not qualify all tokenizers, partial Unicode, stops, cancellation, failures,
concurrent requests or diagnostic consumers.

## Evidence and reproduction

The [evidence archive](fixtures/2026-10-14-output-overhead/output-overhead-evidence.tar.gz)
contains nine files: sanitized native metrics and host observations, native
token traces, replay measurements, hardware/dependency/source/model metadata,
and the five experiment scripts. The adjacent
[manifest](fixtures/2026-10-14-output-overhead/manifest.json) binds each member
and the complete archive by SHA-256.

Reproduce in `/private/tmp/rapid-mlx-decode-output-exp/` on a leased compatible
Mac. Prepare the exact Rapid checkout as `src/`, the exact optional runtime
checkout as `runtime-src/`, a matching task environment as `venv/`, and extract
the five scripts from the archive into that task root. Set `.task-owner` to
`rapid-decode-output-m4-20261014-v1`; set `model-path` to the already-cached
qualified snapshot. Preserve the standard HF cache location. Then run:

```sh
cd /private/tmp/rapid-mlx-decode-output-exp/src
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
  ../venv/bin/python scripts/large-model-run.py --working-set-gb 6 \
  --threshold-gb 0 --lock-timeout 30 -- ../venv/bin/python ../run_native.py
../venv/bin/python ../collect_metadata.py
../venv/bin/python ../run_replay.py
```

Check complete budgets, terminal/usage records, repeated text hashes, matching
source/model hashes, quiet-host observations and all replay equality assertions.
Allow startup readiness and clean server shutdown to finish. Do not overlap
the CPU replay with native serving; the experiments deliberately run in sequence.

## Handoff and cleanup

Vector's research conclusion is that output-history copying is not a new
general bottleneck on the tested ordinary path. Any future adapter cleanup
needs explicit diagnostic retention semantics and targeted lifecycle/Unicode
tests before becoming a product change. There is no basis here for a claimed
cross-machine or cross-model throughput percentage.

Atlas, Harbor, Pixel and Echo receive the completion record in #4645. The M4
server exited cleanly; task processes, source copies and the isolated worker
environment were removed, the lease released and its heartbeat stopped.
The standard HF snapshot remains cached for reuse. Preexisting environments
and caches were preserved. No production deployment or serving patch occurred.
