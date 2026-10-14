# Gemma optimization cross-chip validation

Status: available-worker validation complete; a quiet M5 Max window remains pending owner handoff. Owner: Vector. Tracking: #4649. This is experimental source evidence, not a production change or release qualification.

Two existing prototypes were challenged on a different chip generation and a second model size. Last-position ordinary prefill reproduced its benefit on M2 Pro E2B. Shared-native text scheduling reproduced its delayed-request benefit on M2 Pro E2B and M4 Pro E4B. The scheduling tradeoff depends on hardware: an active answer with a concurrent peer took approximately 7.1% longer on M2, versus approximately 1.1% on M4 E4B. This does not establish support for all chips or Gemma sizes.

## Environment and controls

- Frozen source: `6d617f51cea61f9fa1cdd07a4dea11f66e04c355`, identical to the [initial audit](2026-10-14-frequent-workload-speed-audit.md). Seven critical source hashes were checked.
- M2 Pro 32 GiB; M4 Pro 48 GiB. See fixture metadata for exact OS and Python versions. Both use MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers 5.15.1; NumPy is 2.5.4 on M2 and 2.4.4 on M4. NumPy was not changed in the borrowed environments; both arms on each host use that host's identical version. This is a within-host paired comparison, not an assertion of completely identical environments across chips. All 1,616 dependency Python files match the same official wheels used in the initial audit.
- E2B: `mlx-community/gemma-4-e2b-it-4bit`, snapshot `238767527555cb75a05732a84dff5d6ba0dd6809`.
- E4B: `mlx-community/gemma-4-e4b-it-4bit`, snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2`. Complete model-file hashes match the preceding audit.
- Source selected through `PYTHONPATH`; inherited installed distribution metadata is not the tested source version. No dependency, weight, tokenizer, sampler or serving source files were edited.
- Ordinary decoding, no speculative decoding. Local HTTP first nonempty content defines TTFT; role-only frames are excluded. These are warmed model/shape measurements, not cold startup timings.
- Each campaign starts four fresh servers in baseline/candidate/candidate/baseline order. Prefill: four warm requests plus three alternating-order repetitions of four cases per server, six measured observations per arm/case. Scheduling: warm 32 tokens, solo 512, active 512 with a delayed 64-token peer triggered after 16 content chunks, solo 512, per server.
- Host samples record RSS, swap and other Python processes above 512 MiB. Existing swap stayed constant throughout each performance campaign; no qualifying foreign Python process was observed. This is not a claim of zero swap or complete isolation from every system service.

## 1. Ordinary prefill, E2B on M2 Pro

The candidate passes the dependency's existing `logits_to_keep=1` option to the ordinary multimodal-model forward. It reduces unused output-position work; it does not change weights or sampling. Prefix cache is disabled, with zero reported cached tokens.

| Case | Prompt tokens | Baseline median TTFT | Candidate median TTFT | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Text | 220 | 381.8 ms | 122.8 ms | 67.8% |
| Text | 796 | 1,308.5 ms | 368.8 ms | 71.8% |
| Already chunked text | 2,424 | 1,095.9 ms | 1,094.5 ms | 0.1% |
| Small red image | 280 | 822.6 ms | 495.9 ms | 39.7% |

All 48 measured and 16 warm streams returned the expected answer, usage and finish framing. Candidate projection receipts contain only one vocabulary-output position. The neutral chunked control agrees with the mechanism: that path already avoids the redundant full-position output. Image inputs vary PNG bytes by changing one pixel; vision-cache cold misses were not independently counted.

The prior M4 E2B result was a 66–71% reduction on the same short-text cases, with neutral chunked control. Reproduction on M2 is evidence against an M4-specific explanation. It does not qualify M1/M3/M5, other quantizations, speculative/hidden-state consumers, or larger Gemma architectures.

## 2. Shared-native text scheduling

The candidate reuses the existing Gemma text wrapper and scheduler startup with the exact same language-model object and executor. Startup active-allocation receipts show no additional model-weight allocation. Each measured case has identical visible-output hashes across both routes and all four servers. Receipt timestamps verify that the delayed peer arrives during active decoding and gets its first content before the active answer ends only on the candidate route.

| Chip / model | Delayed peer median TTFT, baseline → candidate | Active answer total, baseline → candidate | Active-answer cost |
| --- | ---: | ---: | ---: |
| M2 Pro / E2B | 5.785 s → 0.067 s | 5.945 s → 6.369 s | +7.1% |
| M4 Pro / E4B | 7.162 s → 0.075 s | 7.371 s → 7.451 s | +1.1% |

Singleton decode medians changed by less than 1% in both campaigns. This is principally removal of queue blocking for arriving requests, not a general single-request generation speedup. On M2, the short peer's own decode also took longer when sharing the batch. Admit/fairness policy needs whole-stack validation before enrollment.

The two routes report different prefix-cache reuse (81 versus 92 cached tokens for these requests). This is an end-to-end routing comparison, not a microbenchmark with identical cache states. Matching visible greedy output is not proof of identical logprob distributions or comprehensive model quality.

## Correctness and disposition

The independent HTTP suites passed 84 checks: M2 E2B baseline/last-position 28, M2 E2B baseline/native 28, M4 E4B baseline/native 28. They cover arithmetic, repeat requests, long retrieval, multiturn, JSON, finite logprobs, seeded repeatability, cancellation/drain/duplicate/recovery, concurrent image/text, subsequent recovery, readiness and clean exit. They do not qualify every protocol, model family or media modality.

The performance campaigns total 104 streams: 64 prefill plus 20 scheduling on each worker, including their warmups. Host and activation receipts are archived separately from correctness results. No runtime change has been submitted or queued by this investigation.

Vector disposition: retain candidate 1 as a small capability-scoped implementation candidate; retain candidate 2 separately with explicit active-request and throughput budgets. Their improvements cannot be added together: native text prefill already eliminates unused output positions, while the ordinary media path can still benefit from candidate 1.

M5 Max remains untested because the owner GUI-trial lease is still active; a handoff request was sent to Atlas. M3 Ultra reservation was refused because another owner's stale lease remains locked. Neither machine's running work was interrupted. E4B was already cached on the M4 worker; M2 disk capacity allowed E2B only. No evidence here covers larger dense or MoE Gemma sizes. Atlas owns the pending host window; Vector owns the additional campaign once a lease can be acquired.

## Reproduce and inspect

The fixture bundle contains the exact exploratory scripts, source/model/dependency metadata, HTTP results, output/activation receipts and host observations. It excludes private connection details, lease credentials and server logs. The original audit fixture supplies the frozen source experiment's prior provenance.

Extract `fixtures/2026-10-14-gemma-cross-chip/evidence.tar.gz` into a temporary directory and run `python3 summarize.py`. It checks the expected source/model hashes, versions, file counts, answers, output hashes, causal request ordering, single-position activation, shared-weight identity, host samples, process exits and all 84 HTTP checks; it recomputes `summary.json`. Inspect `manifest.json` for member and archive SHA-256 values. Use Python 3.12 and the stated dependency versions to rerun the scripts; their explicit scratch root and cached-model path file must be prepared for the new worker. Acquire a worker lease before inference and preserve the shared Hugging Face cache afterward.
