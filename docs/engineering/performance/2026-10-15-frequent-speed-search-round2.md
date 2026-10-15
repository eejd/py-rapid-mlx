# Frequent-workload speed search, round two

The useful lead in this round is enrolling the dense Qwen3.5 4B checkpoint
in Rapid's existing fused GDN decode path. The existing operator already handles
its recurrence geometry; an input-width admission check excludes this model.
The experimental process changes that width check from 2,048 to 2,560 and retains
the numerical probe, dtype, shape, cache, padding, mask and training guards.
This is research evidence for a narrowly scoped implementation, with no
production defaults changed by this commit.

## Environment and experiment

Baseline: `5c70e238f6068d373553c48d92786a5cf333d05b`. Both workers are
Apple M4 Pro with 48 GiB unified memory. Python 3.12.13, MLX 0.32.3,
MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers 5.15.1, NumPy 2.4.4,
HTTPX 0.28.1 and psutil 7.2.2 are pinned in the receipts. The source runs
through `PYTHONPATH`; inherited Rapid distribution metadata says 0.15.7,
so this is not a packaged release qualification. Nine relevant source files
are hashed, and 1,616 dependency Python files match their official wheel
manifests on each worker.

Checkpoint: `mlx-community/Qwen3.5-4B-MLX-4bit`, revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`; its 3,034,300,695-byte weight
file has SHA-256
`5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db`.
The second worker receives the existing checkpoint in its standard cache;
no alternate model cache or new checkpoint revision is introduced.

Requests use the default HTTP serving route, temperature zero, thinking disabled,
no speculative decoding, and disabled prefix/disk caches. A repeated ` data`
prefix precedes an instruction to generate numbered Python functions. A fresh
server performs one 32-token warmup, then two 256-token measurements per context.
The long campaign traverses short/8K/16K/16K/8K/short contexts in that order.
Four fresh servers use baseline/candidate/candidate/baseline order. Both arms
contain the same instrumentation; startup probe calls are cleared before
counting serving activations.

Decode throughput is `(completion_tokens - 1) / (response_end - first_visible_delta)`.
It includes output handling and transport; it is not a kernel-only rate or a
per-token latency distribution. TTFT is recorded separately and no TTFT
improvement is claimed. This workload keeps returning the same code continuation
across input lengths, which is useful for deterministic comparison but limits
the breadth of the quality evidence.

## GDN result

| Worker | Actual input tokens | Baseline tokens/s | Fused tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| A | 856 | 79.66 | 82.49 | +3.56% |
| A | 8,024 | 72.58 | 75.18 | +3.58% |
| A | 16,216 | 65.62 | 67.94 | +3.54% |
| B | 856 | 80.47 | 83.19 | +3.37% |
| B | 8,024 | 73.32 | 75.62 | +3.14% |
| B | 16,216 | 65.89 | 68.16 | +3.45% |

The first worker's independent short-only campaign measures 79.63 to 83.20
tokens/s, a 4.48% increase. The long-context campaign is a separate comparison
with its own controls; do not mix its baseline with the short-only campaign.
Two observations per server/context are repeated observations on one process,
not independent hardware samples or a confidence interval.

All measured streams preserve the observed answer SHA-256, requested completion
length, `length` finish and SSE terminator. Each candidate server enrolls all
24 eligible real layers, and runtime counters establish that actual HTTP decode
uses the fused recurrence rather than silently falling back. Separately, on
each worker, all 24 learned layers receive independent caches warmed by a native
16-token pass and then eight native/fused decode steps with shared input vectors.
All 192 layer-step rows per worker match by storage bits for projected output,
bf16 convolution cache and FP32 recurrent state, with finite values and 192
observed fused activations. This strengthens the numerical evidence beyond
matching a generated answer, but is not a general model-quality qualification.

## Other candidates and stopping decisions

| Candidate | Experiment | Disposition |
| --- | --- | --- |
| GDN threadgroup sizes 32/16/8/4 | Two recurrence head geometries, nonzero FP32 state and output/cache equality checks | Existing default 32 is best or essentially tied; no extra tuning change. |
| Pack four GDN input projections for the default multimodal wrapper | Existing projection parity probes on all real layers, then four fresh HTTP servers comparing fused GDN with and without packing | 83.19 to 83.35 tokens/s, about 0.19%; insufficient benefit to promote. |
| Residual addition followed by RMS normalization | Widths 2,048/2,560/4,096/5,120, bf16/fp16 equality probes and a 32-operation dependency chain | At width 2,560, native 0.329 ms versus custom 0.419 ms; reject. |
| Compile that residual/normalization group | Same chain, native and custom forms | At width 2,560, 0.347/0.369 ms versus native 0.329 ms; no useful gain. |
| RMS normalization followed by residual addition | Four widths, bf16 equality probes and a 32-operation chain | At width 2,560, native 0.311 ms versus custom 0.357 ms; compile variants approximately neutral. A width-2,048 synthetic improvement is not a measured serving gain. |

The normalization component probes establish exact numerical equality for the
sampled inputs; their checks do not reinterpret storage bits. They must not be
described as universal bitwise equivalence. The learned-GDN comparison above
explicitly compares storage bits, including signed zero. One preliminary GDN
component launch failed to import the checkout; it produced no timing data and
is excluded from the successful component receipts.

A synthetic head-count-48, input-width-5,120 geometry also passes the existing
eight-step recurrence probe for all four threadgroups. It does **not** load
Qwen 27B weights, run that model, or establish its speed or quality. Gemma,
TensorFold, speculative decoding, other precisions, batching and other chip
generations are outside this positive result.

The earlier [long-context audit](https://github.com/raullenchai/Rapid-MLX/blob/2f5b6e17a/docs/engineering/performance/2026-10-15-m4-long-context-decode.md) remains
negative: compact KV copies, larger growth slabs and attention-path adjustments
did not qualify. This round does not relabel those attempts as new opportunities.
The search stops this bounded round after preserving the positive enrollment
lead and the negative dispositions; it does not claim to exhaust every possible
inference optimization.

## Evidence and implementation boundary

The [archive](fixtures/2026-10-15-speed-search-round2/evidence.tar.gz) retains
the exact experimental harnesses, JSON receipts and the license accompanying
the normalization prototype. The [manifest](fixtures/2026-10-15-speed-search-round2/manifest.json)
binds every member; the [summary](fixtures/2026-10-15-speed-search-round2/summary.json)
is recomputed without an accelerator:

```sh
python scripts/validate_speed_search_round2.py \
  docs/engineering/performance/fixtures/2026-10-15-speed-search-round2/evidence.tar.gz \
  docs/engineering/performance/fixtures/2026-10-15-speed-search-round2/manifest.json \
  --source-dir .
```

For inference reproduction, reserve an available worker, use the pinned source,
runtime and checkpoint above, and check for competing workloads. Extract the
archived harness into `/private/tmp/rapid-mlx-speed-search-round2`, create `src`
from the baseline and a `venv`, and write `.task-owner` with the exact contents
`vector-speed-search-round2-v1`. Supply `expected-models.json` and
`official-dependency-files.json`, then run `metadata.py` and
`probe_real_layers.py` with model downloads disabled. Run `run_http.py` for the
short campaign and `run_long.py` for the context campaign, passing `baseline`
or `gdn`, model argument `qwen4b`, and distinct `STUDY_TRIAL` values in the
four-process order above. Run serially. The archived `driver-B.py` reproduces
the second worker's full sequence. Packing uses `run_packed.py` and compares
`gdn` to `gdn_pack`, retaining its own controls.

The 18 fresh servers complete 86 streams, including 68 measured responses and
18 warmups. Across 979 host observations, existing swap remains fixed at
195,493,888 bytes on A and 512,032,768 bytes on B; no foreign Python process
exceeds the observer's 512 MiB threshold. This finite process sampling does not
exclude other GPU clients, short unobserved workloads or ordinary OS activity,
and does not certify hardware exclusivity. Both task-specific remote
environments and processes are removed and both leases released after collecting
receipts; standard model caches and borrowed runtimes are preserved.

The next implementation should preserve existing 2,048-width admission and
add only the qualified 2,560-width geometry. It should retain runtime numerical
qualification and all fallback guards, cover unrelated model rejection and
existing-family compatibility, and validate serving lifecycle and broader
prompts before rollout. General head-count support and Qwen 27B qualification
belong to separate evidence. No production PR or merge-queue admission is
created by this research report.
