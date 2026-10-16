# Bolt LiLiCorr phase 1: offline acceptance on MLX targets

**Decision: GO for a bounded, opt-in, greedy phase-2 prototype at roughly 1K
context on both MLX targets. Keep speculation off by default.** Measured mean
output per round is **4.36 (4-bit)** and **4.28 (3-bit)**, including one guaranteed
correction/bonus token. Using fresh target/drafter costs and an explicitly
estimated history overhead projects **71.4 / 70.4 tok/s**, versus fresh ordinary
pipelined controls of **55.6 / 55.5 tok/s**. These are throughput projections;
end-to-end speculative serving was not implemented or measured.

**NO-GO for general default enablement or an 8K rollout on this evidence.**
Writing is below break-even on both targets, and 3-bit chat is also below it.
Long-context acceptance is unmeasured; an estimated replay fallback leaves
only a small 8K margin. Vector owns phase 2: reuse the existing speculative
framework and qualify the Mamba2 rollback work from
`vector/nemotron-super-mtp`, then measure complete serving rounds and serial
output parity before changing default behavior.

Owner: Vector. Hardware: Studio M3 Ultra (256 GB). Scope: offline native
NVFP4 drafter qualification, greedy acceptance, and latency-based decision for
phase 2. Serving integration and recurrent rollback are outside this phase.

## Implementation and semantics

`rapid_mlx/models/lilicorr.py` implements the five-layer block backbone,
noncausal sliding attention with learned sinks, two-tap grouped dynamic
convolutions, and two-layer candidate correlator. Embeddings and output head
come from the loaded target. ModelOpt E2M1 weight bytes are reinterpreted as
little-endian uint32 words; E4M3 block scales remain byte-for-byte unchanged.
Per-matrix global scales multiply the matmul output. No requantization occurs.
Checkpoint coverage and shapes are checked before inference. The checkpoint
architecture license is OpenMDW 1.1, retained in `models/LICENSE-LILICORR` and
included in wheel package data. Weights remain separately downloaded.

The six configured taps `[1,18,35,51,68,85]` mean **outputs of zero-based
blocks**, before the target's final norm. The reference adapter adds one to
these IDs for an embedding-inclusive capture interface. The local optional
backbone tap API captures the corresponding block outputs directly. Existing
calls continue to return the normalized final hidden tensor.

Context features are the projection of all six concatenated taps, followed by
the drafter's `hidden_norm` for context K/V and the correlator anchor. Query
positions begin immediately after the last confirmed context row. Each query
contains a confirmed token and seven mask tokens (ID 990); only mask-slot
outputs go to the candidate head. The correlator uses top-eight candidates,
raw full-vocabulary log-probability features, learned candidate attention
biases, normalized in/out factors, and a left-to-right conditional greedy
walk. Unary logits are features, rather than an extra term added to edge scores.
See the [checkpoint configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/blob/4c6a5613b6d887194a3a5493e3a18a5b3930bb86/config.json)
and [LiLiCorr paper](https://arxiv.org/abs/2608.20530).

## Measurement protocol

`scripts/benchmark_bolt_lilicorr.py` contains twelve fixed prompts: two each
for coding, math, chat, writing, retrieval/summarization, and tool-call-style
output. A target generates up to 160 tokens per prompt by canonical serial
greedy decoding, with thinking disabled in the chat template and termination
at EOS. Each target's own generated trajectory supplies the ground truth.
The target captures taps while generating; proposals never mutate target caches.

The harness evaluates every eligible anchor position with eight subsequent
reference tokens available (seven drafts and a bonus). It also traverses the frozen trajectory as
disjoint speculative rounds, advancing by `accepted_drafts + 1`. This is the
canonical greedy correction/bonus-token rule. Statistics must distinguish
mean accepted drafts (0–7) from mean output tokens per round (1–8). The final
eight anchor positions are excluded; truncated trajectories are not EOS
acceptance tests. Overlapping anchors are diagnostic samples, not independent
rounds for uncertainty estimates.

Draft timers synchronize the complete proposal: embedding gathers, five
backbone layers, shared vocabulary head, candidate extraction, correlator, and
greedy walk. Three complete proposals warm each prompt before timing. Context
fusion/KV append is timed separately, including appending the actual retained
round length in one block. Initial context construction and target prefill are
outside draft timers. Target timings include synchronization of logits and
all cache states. A separate block-cost control uses real target taps at
1K and 8K absolute positions (both retain at most 1K drafter K/V), with twelve
synchronized samples after two warmups.

Fresh target verification curves use the same frozen real prompt text for
both quantizations, repeated and tokenized to form 1K/8K contexts. Prefill is
chunked at 256 tokens. T1–T8 are measured from equivalent independent cache
objects at each window, advancing the common prefix by eight tokens after
each window. Report median of twelve samples after two warmups. These curves
measure ordinary target verification, **without recurrent rollback**.

A separate ordinary asynchronous greedy-decode control uses the same 1K/8K
prefix for each quantization, no hidden taps, 256 timed tokens after the first
token, one warmup run and three measured runs. It ignores EOS to hold decode
length fixed. Prefill and first-token latency are excluded. This is an offline
pipelined decode control, not a server throughput benchmark.

## Published comparison and projection boundary

NVIDIA publishes SPEED-Bench acceptance length 4.30 overall, with coding 5.26,
math 4.59, writing 3.41, RAG 5.13, and summarization 4.80, for its **24GB target**,
draft length seven, temperature 1.0 and top-p 0.95.
See the [checkpoint card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/blob/4c6a5613b6d887194a3a5493e3a18a5b3930bb86/README.md).
This phase uses different target quantizations, greedy decoding, a small fixed
prompt set, and shorter response caps. It is a local feasibility gate, not a
SPEED-Bench reproduction or a controlled estimate of quantization-only loss.
Both accepted-draft and output-length statistics are retained because the
checkpoint card does not explicitly resolve the bonus-token convention.

Projected decode throughput uses measured output tokens per round divided by
`V8 + draft_ms + context_update_ms + rollback_overhead_ms`. It is not measured
end-to-end serving throughput. The previous 4-bit state-history experiment
measured V8 = 50.74 ms at roughly 1K, compared with ordinary V8 = 49.37 ms in
that same experiment; it also showed numerical differences requiring separate
qualification. The supplied pipelined 1K baseline is 56.5 tok/s. Neither that
baseline nor 4-bit history timings should be presented as measured 3-bit
rollback costs. Phase 2 must use the existing speculative framework and the
qualified Mamba2 rollback implementation.

## Measured acceptance

Primary statistic is block-weighted output length `A = accepted_drafts + 1`
on the disjoint greedy rounds. The accepted-draft column excludes the bonus.
The published column retains NVIDIA's metric as reported, without adding one.

| Category | 4-bit rounds | 4-bit accepted / A | 3-bit rounds | 3-bit accepted / A | NVIDIA published |
| --- | ---: | ---: | ---: | ---: | --- |
| Coding | 59 | 4.24 / 5.24 | 57 | 4.39 / 5.39 | 5.26 |
| Math | 50 | 5.14 / 6.14 | 56 | 4.54 / 5.54 | 4.59 |
| Chat | 80 | 2.83 / 3.83 | 99 | 2.13 / 3.13 | No matching category |
| Writing | 112 | 1.72 / 2.72 | 101 | 2.03 / 3.03 | 3.41 |
| RAG / summarization | 32 | 4.09 / 5.09 | 29 | 4.69 / 5.69 | 5.13 RAG; 4.80 summarization |
| Tool-call style | 21 | 6.24 / 7.24 | 19 | 6.74 / 7.74 | Not reported |
| Overall | 354 | 3.36 / **4.36** | 361 | 3.28 / **4.28** | 4.30 |

Each category has two prompts. There are 1,621 / 1,615 generated reference
tokens and 1,525 / 1,519 evaluated anchors for 4-bit / 3-bit respectively.
Disjoint rounds account for 1,542 / 1,545 output tokens, including known
correction/bonus tokens. Equal-prompt macro means are 5.04 / 5.13; the primary
round-weighted means are lower because difficult prompts need more rounds.

A 5,000-resample prompt-cluster bootstrap (seed 42) gives approximate 95%
intervals of **3.47–5.49** and **3.53–5.34** for mean A. Twelve prompt clusters
cannot establish broad workload coverage or precise per-domain bounds.
Generated continuations differ between targets, so the small difference in
aggregate acceptance is not a controlled estimate of quantization-only loss.
The local acceptance is sufficient for the projection below; matching the
published overall number is not evidence of SPEED-Bench reproduction.

| Accepted drafts j | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit rounds | 55 | 61 | 46 | 47 | 18 | 25 | 17 | 85 |
| 3-bit rounds | 60 | 58 | 56 | 39 | 25 | 23 | 18 | 82 |

## Measured cost and throughput controls

| Target | Context | V1 ms | V8 ms | V8 / V1 | Full draft block ms | Ordinary pipelined tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 1,024 | 21.06 | 50.31 | 2.39 | 8.50 | 55.63 |
| 4bit | 8,192 | 21.74 | 58.76 | 2.70 | 8.57 | 54.21 |
| 3bit | 1,024 | 20.62 | 50.19 | 2.43 | 8.42 | 55.48 |
| 3bit | 8,192 | 21.00 | 59.05 | 2.81 | 8.62 | 55.04 |

Full block costs above include the shared vocabulary head, candidate scoring
and greedy selection, with real target context features. At 8K absolute
positions, the drafter still retains only its trained 1K sliding window.
Across mixed short prompts, mean proposal cost was 7.83 / 7.73 ms and mean
retained-round context fusion/KV append cost was 0.87 / 0.80 ms. Initial
prefill/context setup is excluded. The append costs come from the short
acceptance trajectories; additional long-window append overhead is unmeasured.

Fresh T1–T8 median curves, in milliseconds:

| Target / context | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit / 1,024 | 21.06 | 28.55 | 32.58 | 35.89 | 39.69 | 43.17 | 48.31 | 50.31 |
| 4bit / 8,192 | 21.74 | 29.10 | 40.08 | 44.35 | 48.32 | 51.56 | 58.83 | 58.76 |
| 3bit / 1,024 | 20.62 | 27.71 | 32.06 | 35.67 | 39.73 | 43.33 | 48.55 | 50.19 |
| 3bit / 8,192 | 21.00 | 28.47 | 40.00 | 44.19 | 48.45 | 51.88 | 57.48 | 59.05 |

## Throughput projections and go/no-go gate

Use `1000 * mean(A) / (V8 + draft_block + context_append + rollback)`.
The zero-rollback column is an optimistic cost bound. The history-informed
scenario adds the earlier 4-bit experiment's **1.37 ms** history-forward delta
(50.74 minus 49.37 ms) to each fresh V8. This delta is **estimated for these
runs**, including 4-bit at 8K and both 3-bit contexts; it was not measured by
this phase or qualified for this port.

The replay scenario instead averages the prior measured 4-bit Mamba-only
recovery costs `[5.06,9.16,9.43,10.55,11.61,12.46,14.01]` ms over j=0..6
using the measured rejection histogram; full acceptance needs no replay.
Expected recovery is 7.13 / 7.21 ms per round. Applying those prior costs to
3-bit or 8K is also an estimate. The existing history/replay experiment did
not establish canonical serial numerical parity.

| Target | Context | Zero rollback tok/s | History-informed tok/s | Replay-estimated tok/s | Fresh baseline | History speedup | Break-even A, history |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 1,024 | 73.0 | **71.4** | 65.2 | 55.6 | 1.28× | 3.40 |
| 4bit | 8,192 | 63.9 | **62.6** | 57.8 | 54.2 | 1.16× | 3.77 |
| 3bit | 1,024 | 72.0 | **70.4** | 64.2 | 55.5 | 1.27× | 3.37 |
| 3bit | 8,192 | 62.5 | **61.3** | 56.6 | 55.0 | 1.11× | 3.84 |

The 1K history-informed gains are about **28% / 27%** over the fresh offline
pipelined controls. Relative to the supplied prior **4-bit** 56.5 tok/s
reference, the 4-bit projection is about **26%** faster. No 3-bit baseline is
inferred from that reference. Phase-2 request scheduling, sampling, transaction
bookkeeping, actual rollback and hidden-feature retention remain unmeasured.
Phase 2 must measure whole-round latency including scheduling and rollback.

Acceptance was measured at short contexts (40–134 prompt tokens plus up to
160 generated tokens). **Both 1K and 8K projections hold that measured
acceptance constant**, which is an extrapolation. The 8K replay-estimated
margins are only about 7% / 3%; ordinary runtime overhead could consume them.
Writing A=2.72 / 3.03 is below the 1K break-even A=3.40 / 3.37, as is 3-bit
chat A=3.13. These results support a controlled opt-in experiment rather than
general enablement. Longer held-out trajectories and long-context acceptance
are phase-2 qualification requirements.

## Validation, provenance, and reproduction

- **32 focused tests passed**: seven drafter/format/tap/reference tests and
  25 existing Bolt backbone tests. Final run: 4.48 seconds. Strict dense
  NumPy-reference comparisons use the CPU FP32 stream; forward and native
  NVFP4 adapter tests use the default GPU stream.
- All **25 actual-export NVFP4 matrices** passed native BF16 matmul versus
  dequantized FP32-reference checks (64 output rows each, seed 42). Worst
  relative L2 error was **0.362%**. The packed-format unit test independently
  decodes E2M1 nibbles and E4M3 scales with NumPy, rather than using the same
  dequantizer as the implementation.
- Ruff, formatting, compilation, diff checks, and wheel module/license
  inclusion checks passed. The new unit test file is in the Apple CI list.
- Native modeling and tap code is in local commit `dbaeef56f`; measured
  harness with scalar/list EOS compatibility is `f0dd34d2a`. Both measurement
  JSONs record `f0dd34d2a` plus SHA-256 of the module, target adapter and
  benchmark source. No acceptance data from the initial EOS-metadata failure
  is used.
- All model execution, including tests, used
  `lockf -k /var/tmp/rapid-mlx-large-model.lock` and a Machine Control lease
  owned by `rapid-mlx/vector-bolt-lilicorr`. The final job released its lease
  after both runs; no serving processes or other jobs were stopped.

Environment: Apple M3 Ultra, 28 CPU / 60 GPU cores, 256 GiB unified memory,
macOS 26.5.2, Python 3.12.14, MLX 0.32.4, mlx-lm 0.31.3, NumPy 2.5.4,
Transformers 5.15.1. Fresh worktree `.venv`, `pip install -e '.[dev]'`.

| Artifact | HF revision | Safetensors bytes |
| --- | --- | ---: |
| NVIDIA Bolt 24GB LiLiCorr drafter | `4c6a5613b6d887194a3a5493e3a18a5b3930bb86` | 962,485,872 |
| Rapid Bolt MLX 4-bit | `cbfda6be4e9ff9e8d3acf0c0a57c54c1a496ef81` | 43,257,123,023 |
| Rapid Bolt MLX 3-bit | `7816420d25f0c1ce2495a5a8e4dc8349f112cbce` | 31,653,318,653 |

Download snapshots with `huggingface_hub.snapshot_download` using these
revisions and the default shared HF cache. Acquire a Machine Control lease
with the owner above, maintain its heartbeat, then run from this worktree:

```sh
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python -m pytest   tests/test_lilicorr.py tests/test_nemotron_h_puzzle_vendored.py -q   --basetemp /private/tmp/bolt-lilicorr-phase1/pytest-reproduce

# TARGET and DRAFTER are the cached snapshot paths for the pinned revisions.
# Repeat for each target, with distinct output paths under this scratch root.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAPID_MLX_TRUST_REMOTE_CODE=0   lockf -k /var/tmp/rapid-mlx-large-model.lock   .venv/bin/python -u -m scripts.benchmark_bolt_lilicorr   --target "$TARGET" --drafter "$DRAFTER" --tokens 160   --output /private/tmp/bolt-lilicorr-phase1/reproduce.json

.venv/bin/ruff check rapid_mlx/models/lilicorr.py   rapid_mlx/models/nemotron_h_puzzle.py scripts/benchmark_bolt_lilicorr.py   tests/test_lilicorr.py
```

Stop the owned runs and release the lease after completion. Default harness
arguments evaluate all twelve prompts and both 1K/8K cost controls. The
[committed evidence fixture](fixtures/2026-10-16-bolt-lilicorr-phase1.json)
retains exact prompts/input/generated token IDs, per-anchor acceptance and
latency arrays, disjoint round positions, raw verification/baseline samples,
source hashes and actual-export arithmetic checks. Independent post-run
validation recomputed every accepted prefix from the full proposals, checked
round advancement and counts, checked all sample counts, and matched source
hashes before distilling the fixture. Full raw proposals/logs are in
`/private/tmp/bolt-lilicorr-phase1/`; their JSON hashes are in the fixture.

Phase-2 handoff to Vector: preserve zero-based tap ordering, confirmed-only
context, target embedding/head sharing and conditional greedy path semantics.
Use the existing speculative lifecycle for accepted-prefix commit, attention
trim, Mamba2 state selection/replay, cancellation and cleanup. Qualify Bolt
serial-versus-block numerics at every acceptance boundary. Measure real
end-to-end speed with rollback and request overhead at 1K first. Keep default
off; separately qualify longer prompts/completions and 8K acceptance. Sampled
decoding additionally requires the correlator's realized conditional proposal
probabilities in the rejection sampler. No server integration, push, PR,
release, or default-policy change was performed in phase 1.
