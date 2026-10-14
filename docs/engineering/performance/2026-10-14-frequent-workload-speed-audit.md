# Frequent-workload inference speed audit

Date: 2026-10-14. Owner: Vector. Tracking: #4646. Status: completed screening.
Serving source and defaults are unchanged; the experiments use isolated process
prototypes. This report separates actual HTTP results from component probes.

## Frozen environment

Baseline: `6d617f51cea61f9fa1cdd07a4dea11f66e04c355`.
Two independently leased Apple M4 Pro workers, each with 48 GiB unified memory
and macOS 26.5.1. Python 3.12.13, MLX 0.32.3, text runtime 0.31.3,
vision runtime 0.7.2, Transformers 5.15.1, NumPy 2.4.4, HTTPX 0.28.1 and
psutil 7.2.2. Both use the same seven source-file hashes and identical target
weight/config/tokenizer hashes. Official large-file digests were checked.
Both dependency installations were also compared with their official wheels:
all 1,450 vision-runtime and 166 text-runtime Python source files match on both
workers. The output option does not rely on a locally patched dependency.
These are two physical machines of the **same chip generation**, not
cross-generation qualification.

Models in the completed campaigns:

- `mlx-community/Qwen3.5-4B-MLX-4bit@32f3e8ecf65426fc3306969496342d504bfa13f3`
- `mlx-community/gemma-4-e2b-it-4bit@238767527555cb75a05732a84dff5d6ba0dd6809`
- `mlx-community/gemma-4-e4b-it-4bit@475b9088d29754a3379866cf5aeb6b41acd313c2`

The source runs through `PYTHONPATH` with task-local environments reusing
existing dependency packages; installs went only into the task-local environment.
Inherited Rapid distribution metadata says 0.15.7; the recorded
source hashes, rather than that installation banner, identify the serving
implementation. This is not acceptance of a packaged release.

All HTTP campaigns disable speculative decoding, disk caches and model
downloads. Greedy campaigns disable thinking explicitly. Each server binds
localhost and uses the existing large-model host lock. Completed performance
campaigns exit cleanly; a failed replay-enrollment attempt is separately
dispositioned below. Models
reside in the standard Hugging Face cache. A third worker was inspected but
declined for additional downloads because of limited disk headroom; it supplies
no inference evidence. Existing environments, caches and unrelated processes
were preserved.

## Ordinary first-token work: a useful unused dependency contract

The vision-compatible forward computes a vocabulary projection for every input
position, although ordinary generation consumes only the final position.
The installed Gemma dependency already implements `logits_to_keep=1`, including
its supported final-layer work reduction. An experimental wrapper requests this
existing contract without changing weights, the tokenizer or the sampler.
This is an architecture enrollment experiment, not a generic keyword injection
for unknown models or speculative verification.

Each worker runs four fresh children in baseline/candidate/candidate/baseline
order. Every child warms each shape, then measures three rounds with alternating
case order. There are 48 measured streams and 16 warmups **per worker**. Prefix
caching is disabled, and every result confirms zero cached prompt tokens.
The text task retrieves `7319`; the image task identifies a red 224×224 image.
Each image changes one pixel to vary its PNG bytes; the normal vision-cache
policy remains enabled. No separately verified cold-vision-cache claim is made.

TTFT begins before HTTP dispatch and ends at the first nonempty content delta,
excluding role-only frames. Complete streams require final usage, a finish
reason, `[DONE]` and the correct answer. GPU and model state are warm; an
uncached prompt is not a cold process start.

| Prompt/case | Worker A baseline → candidate | Worker B baseline → candidate |
| --- | ---: | ---: |
| 220 text tokens | 225.57 → 75.63 ms | 223.67 → 75.18 ms |
| 796 text tokens | 780.15 → 226.11 ms | 772.16 → 223.77 ms |
| 2,424 text tokens | 672.31 → 672.95 ms | 667.17 → 666.55 ms |
| Image, 280 prompt tokens | 531.79 → 340.57 ms | 526.68 → 335.81 ms |

The two shorter text cases save approximately 66–71% TTFT; the image case saves
approximately 36%. The longer control already splits prefill and evaluates
only the final token's projection, so the extra flag gives no benefit there.
This boundary is consistent with the proposed cause. Observed baseline logits
shapes contain 220/796/280 rows, while candidate forwards return one row. Worker
receipts also retain reset-per-prefill active-memory peaks; these are allocator
measurements, not total application RAM or a model-weight reduction.

All 128 streams across the two workers pass the response checks. The four text
completion tokens and the image answer remain correct. These synthetic checks
do not establish broad model quality, unknown architectures, other chip
generations, output throughput gains, or improvements to already-chunked long
prompts.

A separate four-child, order-balanced E4B campaign on worker A repeats the same
48 measured streams and 16 warmups with the second immutable model above.

| E4B prompt/case | Baseline → candidate median TTFT |
| --- | ---: |
| 220 text tokens | 439.42 → 238.48 ms |
| 796 text tokens | 1,520.66 → 781.75 ms |
| 2,424 text tokens | 2,332.76 → 2,332.29 ms |
| Image, 280 prompt tokens | 804.09 → 546.07 ms |

All 64 streams pass. Gains are approximately 46–49% for the shorter text cases
and 32% for the image; the already-chunked control stays neutral. Another four
fresh servers, baseline/candidate for each size, pass all 56 arithmetic,
multi-turn, JSON, logprobs, seeded-repeatability, cancel/drain/recovery and
concurrent-media/text HTTP checks for this **forward option**. These are
separate from the routing prototype checks below. They do not establish every
serving protocol, speculative path, model family or numerical quality boundary.

## Delayed request arrival: a distinct scheduling opportunity

Current dense Qwen serving already selects a singleton-compatible path for a
sole request and a shared-weight native scheduler for concurrent text. That is
existing functionality and is not a new finding or proposed optimization.

Gemma E2B's default multimodal scheduler instead waits for the current batch to
finish before admitting a new request. A second text request is sent after the
first 16 content chunks of a 512-token answer. The prototype reuses the existing
text wrapper and shared-executor startup with the exact same language-model
object. Active allocated memory immediately before/after the extra scheduler
starts is identical: 3,550,535,066 bytes. There is no second model load.

Four fresh children run baseline/candidate/candidate/baseline. Each warms up,
measures a 512-token singleton, measures the 512-token active answer plus a
delayed 64-token request, then measures another singleton.

| Median, worker A | Existing route | Native-text prototype |
| --- | ---: | ---: |
| Delayed request TTFT | 4.0677 s | 0.0433 s |
| Singleton decode interval | 4.1306 s | 4.1342 s |
| Active answer total time with the peer | 4.1836 s | 4.2287 s |

All 16 measured streams and four warmups have identical visible-output hashes
across routes for their matching cases. Candidate and baseline cache usage are
92 and 81 tokens respectively; this is an end-to-end routing comparison,
not an identical-cache-state microbenchmark. The important effect is allowing
the peer to progress while the first answer is still running. It is not a
single-answer decoding speedup. The active answer takes about 1.1% longer.

Separate fresh baseline/candidate servers pass all 28 HTTP checks: arithmetic,
repetition, 6,336-token retrieval, multi-turn history, JSON, finite returned
logprobs, seeded repeatability, cancel/drain/duplicate-cancel/recovery,
concurrent image and text, post-image recovery, readiness and clean exit.
Returned logprob values can differ between cache paths; numerical probability
identity and comprehensive quality qualification are not claimed. This
prototype also requires broader model/hardware enrollment work before shipping.

The two opportunities are not additive. Native text prefill already evaluates
prefix state separately from the final sampled position, so a later routing
change can subsume the ordinary-text portion of the forward-option saving.
The media-capable path still needs its own efficient output contract. Any
combined implementation must be benchmarked as a whole; do not multiply the
percentages in this report. No experiment here loads a TensorFold provider or
Nemotron, and Qwen 4B host-work results are not evidence for Qwen 27B.

## Candidates that did not justify a general speed patch

| Candidate | Executed evidence | Disposition |
| --- | --- | --- |
| Less frequent Metal allocator cache clearing | Six fresh Qwen servers, order-balanced with the next row; warmup then 512/2,048/512 tokens | At 512 tokens, median 6.218 → 6.258 s; at 2,048, 25.108 → 24.889 s. Mixed sub-1% effect; no qualified gain. |
| Skip unrequested greedy probability distributions | Same campaign, isolated greedy/no-logprobs prototype | 512 tokens 6.218 → 6.223 s; 2,048 tokens 25.108 → 24.790 s. At most approximately 1.3% here; not a drop-in public logprobs patch. |
| Reuse the existing fused top-p sampler in the multimodal factory | Four fresh children per model, six measured 512-token streams per arm, temperature 0.7/top-p 0.9 | Qwen 6.414 → 6.450 s; Gemma 4.458 → 4.373 s. No substantial cross-model gain. Fusion engagement is verified; 20 existing helper tests pass. Random text equality is not the distributional contract. |
| Submit narrow decoder layers earlier | Same order-balanced two-model HTTP design, greedy | Qwen and Gemma throughput regress approximately 1.6%; outputs remain exact. Do not broaden the existing narrow enrollment. |
| Merge dense feed-forward gate/up projections | Three real layers per model, 1/8/128 input rows; six alternating timing pairs per shape | All 18 cases are byte-exact, but ratios remain near one. Not an end-to-end speed claim or a justified new default. |
| Merge attention Q/K/V projections | Real-weight component probe at the same shapes | Narrow-row gains are modest; all six 128-row cases fail byte-exactness. A broad fusion is not qualified. |
| Enroll dense native 4B in existing whole-step replay | A guarded native-path enrollment attempt, then a load-only contract probe | Zero qualified GDN numerical tags versus 24 required, and zero attention tags versus eight required. No valid candidate performance sample; do not bypass the numerical prerequisites. |

The first campaign has 18 measured streams and six warmups, all output-exact.
Its 245 one-second host observations have no foreign Python process above
512 MiB; existing swap stays at 195,493,888 bytes. The Gemma routing campaign's
68 observations show the same conditions. These observations do not prove the
absence of every possible source of contention. The unused allocator observer
in the first campaign records no samples because its public `step` wrapper is
not the production dispatch entry point; no memory claim uses that observer.

## Coverage and remaining work

The failed replay attempt asserted successful enrollment inside first-request
setup; that assertion left the synthetic request without a completion. Its
research runner timed out, and normal shutdown did not finish. The task-owned
server was explicitly killed and the runner's escalation was corrected.
No serving source was changed, and this failed attempt contributes no latency
or throughput claim. The subsequent load-only contract probe reproduces the
missing prerequisites without sending a completion request or opening a server.
The later forward-option correctness run overlapped memory retained by that
failed process before cleanup; its 56 checks are not quiet-host performance
evidence. The earlier performance campaigns preceded the failed attempt; the
host-preparation study below began after explicit cleanup.

## Growing conversations: useful CPU cleanup, small end-to-end effect

The remaining host-work hypothesis was tested on ordinary Qwen 4B with prefix
caching enabled. Its boundary probe already encodes the prompt through the
native host cache, then the singleton multimodal processor encodes it again.
The prototype passes the already-computed IDs through the existing internal
pre-tokenized contract and preserves the original all-valid attention mask.
It does not guess tokenization of a new suffix or introduce cross-turn token
splicing. Enrollment is restricted to the observed no-BOS/ByteLevel processing
contract and the existing shared-native dense bridge.

Four fresh order-balanced servers each populate three conversations, then
append three assistant/user turns to each. All 48 streams finish correctly;
all matching input-ID hashes, attention-mask hashes, usage and cached-token
counts are identical across arms. The warm rows are:

| Prompt token range | Existing → reuse median TTFT | Processor thread CPU |
| --- | ---: | ---: |
| 1,757–1,815 | 155.03 → 154.98 ms | 1.329 → 0.096 ms |
| 7,133–7,191 | 195.15 → 193.01 ms | 4.776 → 0.246 ms |
| 29,686–29,744 | 371.05 → 354.18 ms | 20.056 → 1.328 ms |

There are six warm observations per arm per size. These are instrumented
exploratory HTTP measurements, with additional equal hash-observation work and
without a separate host sampler. The largest median TTFT difference is about
4.5%, versus approximately 0–1.1% at the smaller sizes. This identifies bounded
redundant CPU work, not a large general speedup or an immediately qualified
serving patch. It ranks below the two larger opportunities above.

The ten executable hypotheses now have either positive HTTP evidence, a small
or negative effect, or a concrete failed enrollment/correctness disposition.
The source audit also checked the common mechanisms below, so existing work is
not mistaken for a new generic optimization:

| Mechanism | Existing integration boundary / disposition |
| --- | --- |
| Continuous batching and dense text routing | `engine/batched.py`, `scheduler.py`: qualified dense Qwen already has shared native text scheduling; Gemma enrollment was measured separately. |
| Prefill fairness and adaptive budgets | `mllm_batch_generator.py`, `scheduler.py`: qualified cooperative continuation and token budgets already exist; they are not automatically applicable to every wrapper. |
| KV and recurrent prefix reuse | `memory_cache.py`, `mllm_cache.py`, `prefix_cache.py`: reuse requires real state and valid checkpoint boundaries. |
| Prompt host work | `prompt_host_cache.py`, `prompt_token_reuse.py`: exact renders/encodes and boundary variants already reuse work; the remaining duplicate pass was measured above. |
| Incremental output processing | `mllm_scheduler.py`, `scheduler.py`, `service/postprocessor.py`: existing incremental detokenizers and live histories avoid the obvious repeated-history work. |
| CPU/GPU overlap and dispatch amortization | `mllm_batch_generator.py`, `engine_core.py`: compute-ahead and native step coalescing are present; broader layer submission was tested and did not help. |
| Sampling and selected kernel fusion | `_sampler_fast_path.py`, `moe_fusion.py`, `gdn_in_proj_fusion.py`, `qwen35_fused_gdn_decode.py`: existing qualified patterns were reused before testing wider enrollment. |
| Image-feature reuse | `vision_embedding_cache.py`, `mllm_batch_generator.py`: existing content-keyed processor/feature reuse remains independent of the output-row optimization. |
| KV precision choices | `quantized_batch_cache.py`, `kv_cache_dtype.py`: supported layouts already expose precision choices; changing model/cache fidelity is not a missing execution mechanism. |
| Shared prefix memory | `singleton_cache_fastpath.py`, `paged_cache.py`: copy-on-admit protects retained state. An [earlier controlled study](2026-09-13-apcv2-host-prompt-plane.md) rejected naïve broadcast; segmented attention needs a separate native implementation. |
| Graph replay / device-specific execution | `compiled_decode.py`, `compiled_precision.py`, native kernels: numerical, cache-layout and wrapper-state contracts matter. A kernel for a different device API is not a drop-in Metal implementation. |

The initial routing hypothesis came from inspecting the older conversation
worktree instead of the pinned research worktree. Rechecking main found the
existing dense routing. That no-op routing pilot was stopped and its timing
data are excluded from the evidence archive and all benefit claims. All serving
campaigns themselves used the correct archived baseline.

The direct-adoption candidate set is now screened for frequent single-worker
chat, code and image requests at the same model precision. The immediate
shortlist is the existing last-position forward contract, followed by Gemma
text scheduling. Remaining broad alternatives need new numerical/kernel
contracts, different deployment topology, or separate fidelity qualification.
This is a documented stopping point for this audit, not proof that all future
inference optimization is impossible. Neither prototype is a production patch
or a queued PR.

Implementation handoff: Vector should first turn the existing last-position
contract into a small capability-gated ordinary-prefill change, with unknown
layouts and speculative/hidden-state consumers keeping their qualified paths.
Treat native Gemma text enrollment as a separate, larger compatibility change.
Both need production regression gates and model/runtime enrollment review;
different chip generations are required before advertising cross-generation
percentages. Atlas owns sequencing and compatibility disposition; Harbor owns
the later PR validation window. Pixel needs no UI or public API change for the
first candidate.

All task-owned worker inference processes, scratch and environments were cleaned,
and all task leases released. Standard model snapshots were retained for reuse.

## Reproduction and receipts

The [evidence archive](fixtures/2026-10-14-hotpath-search/evidence.tar.gz) contains
selected raw JSON, model/source identities and standalone experiment scripts;
the [manifest](fixtures/2026-10-14-hotpath-search/manifest.json) binds every member
and the archive by SHA-256. Raw private server logs and the invalid routing
pilot are deliberately excluded.
The [independently recomputed summary](fixtures/2026-10-14-hotpath-search/summary.json)
is generated by the archived offline validator. It checks all member hashes,
seven source hashes, six official large-file digests, response/input equality,
activation shapes, peer progress, and the 38 host-monitor files (821 samples).
Existing swap is constant per worker: 195,493,888 bytes on A and 512,032,768 on B;
these are not zero-swap claims. No monitored performance sample records a foreign
Python process above 512 MiB. The host-work exploratory run has no such sampler.

Use the exact baseline/dependency versions and immutable model revisions above.
Prepare a fresh scratch workspace on a leased quiet worker, with `src` pointing
to that baseline checkout, a Python environment containing the recorded stack,
the scripts from the archive, and a `.task-owner` file containing
`rapid-hotpath-m4-20261014-v1`. The scripts default to
`/private/tmp/rapid-mlx-hotpath-search`; change that constant consistently when
using another fresh scratch directory. Resolve the pinned Gemma snapshot using
the official `snapshot_download` default cache and write its path to
`gemma-model-path`. Do not override the model-cache location.

For example, from the baseline `src` directory, after preparing that workspace:

```bash
experiment_root=/private/tmp/rapid-mlx-hotpath-search
"$experiment_root/venv/bin/python" scripts/large-model-run.py \
  --working-set-gb 9 --threshold-gb 0 --lock-timeout 30 -- \
  "$experiment_root/venv/bin/python" "$experiment_root/run_last_logits.py"
```

`run_gemma_balanced.py` reproduces the delayed-arrival study;
`check_gemma.py` runs its correctness/lifecycle checks. Other harnesses retain
their own explicit workload and activation settings. Never overlap independent
inference campaigns on the same worker or infer a live throughput gain from a
CPU/component replay alone.

`run_last_logits_e4b.py` uses `gemma-e4b-model-path`; `check_last_logits.py`
checks this option for both sizes. `run_prepared.py` reproduces the bounded
growing-conversation CPU study. To verify the published archive without model
inference, extract `scripts/validate_evidence.py` and run:

```bash
python validate_evidence.py evidence.tar.gz manifest.json --source-dir /path/to/baseline
```
