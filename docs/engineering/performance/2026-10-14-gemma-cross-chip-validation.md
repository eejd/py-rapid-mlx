# Gemma optimization cross-chip validation

Status: experiments completed on M2 Pro, M4 Pro and M5 Max; 26B-A4B native compatibility remains unqualified because one baseline repeatability control failed. M3 did not obtain a performance window. Owner: Vector. Tracking: #4649. This is experimental source evidence, not a production change or release qualification.

Two existing prototypes were challenged across chip generations and model architectures. Last-position ordinary prefill reproduced its benefit on M2 Pro E2B and M5 Max E2B, with a much smaller benefit on M5 Max 26B-A4B. Shared-native text scheduling reproduced its delayed-request benefit on M2 Pro E2B, M4 Pro E4B, and M5 Max E2B/26B-A4B. The scheduling tradeoff depends on hardware: an active answer with a concurrent peer took approximately 7.1% longer on M2, versus approximately 1.1% on M4 E4B. This does not establish support for all chips or Gemma sizes.

## Environment and controls

- Frozen source: `6d617f51cea61f9fa1cdd07a4dea11f66e04c355`, identical to the [initial audit](2026-10-14-frequent-workload-speed-audit.md). Seven critical source hashes were checked.
- M2 Pro 32 GiB; M4 Pro 48 GiB; M5 Max 36 GiB. See fixture metadata for exact OS and Python versions. All use MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers 5.15.1; NumPy is 2.5.4 on M2, 2.4.4 on M4, and 2.5.3 on M5. NumPy was not changed in the borrowed environments; both arms on each host use that host's identical version. This is a within-host paired comparison, not an assertion of completely identical environments across chips. All 1,616 dependency Python files match the same official wheels used in the initial audit.
- E2B: `mlx-community/gemma-4-e2b-it-4bit`, snapshot `238767527555cb75a05732a84dff5d6ba0dd6809`.
- E4B: `mlx-community/gemma-4-e4b-it-4bit`, snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2`. E2B/E4B model-file hashes match the preceding audit.
- 26B-A4B MoE: `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. All three weight-file SHA-256 values match the official pinned model revision; filtered provenance is in the fixture.
- Source selected through `PYTHONPATH`; inherited installed distribution metadata is not the tested source version. No dependency, weight, tokenizer, sampler or serving source files were edited.
- Ordinary decoding, no speculative decoding. Local HTTP first nonempty content defines TTFT; role-only frames are excluded. Exact OS versions are macOS 26.5.2 (M2), 26.5.1 (M4), and 27.0.1 (M5); Python is 3.12.13. These are warmed model/shape measurements, not cold startup timings.
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

The prior M4 E2B result was a 66–71% reduction on the same short-text cases, with neutral chunked control. Reproduction on M2 is evidence against an M4-specific explanation. These tests do not qualify M1/M3, other quantizations, speculative/hidden-state consumers, the unified 12B family, or the larger 31B dense architecture.

### M5 Max replication and larger-model boundary

| Case | E2B baseline → candidate | E2B reduction | 26B-A4B baseline → candidate | 26B-A4B reduction |
| --- | ---: | ---: | ---: | ---: |
| Short text | 50.3 → 26.6 ms | 47.2% | 109.1 → 101.3 ms | 7.2% |
| Longer text | 122.7 → 47.3 ms | 61.5% | 251.2 → 223.3 ms | 11.1% |
| Already chunked | 119.0 → 117.8 ms | 1.1% | 603.0 → 602.8 ms | 0.03% |
| Small image | 115.0 → 85.1 ms | 26.0% | 304.1 → 293.3 ms | 3.6% |

E2B uses 220/796/2424/280 actual prompt tokens; the larger model's template produces 224/800/2428/284. Each size is compared against its own baseline with identical inputs; this is not an identical-token comparison between sizes. Both 64-stream campaigns returned expected answers and one-position candidate outputs.

The smaller model's large improvement does not generalize in magnitude to the larger MoE architecture. The dependency can trim unused positions before the trailing KV-sharing layers on E2B (20 of 35 layers share KV); the 26B-A4B configuration has zero KV-sharing layers. The latter primarily avoids the redundant vocabulary-output projection. This source-supported difference explains why the option has more work to remove on the smaller model; it is not a measured per-kernel attribution. The larger model's image benefit is small and should not be marketed as a substantial media speedup.

## 2. Shared-native text scheduling

The candidate reuses the existing Gemma text wrapper and scheduler startup with the exact same language-model object and executor. Startup active-allocation receipts show no additional model-weight allocation. Each measured case has identical visible-output hashes across both routes and all four servers. Receipt timestamps verify that the delayed peer arrives during active decoding and gets its first content before the active answer ends only on the candidate route.

| Chip / model | Delayed peer median TTFT, baseline → candidate | Active answer total, baseline → candidate | Active-answer cost |
| --- | ---: | ---: | ---: |
| M2 Pro / E2B | 5.785 s → 0.067 s | 5.945 s → 6.369 s | +7.1% |
| M4 Pro / E4B | 7.162 s → 0.075 s | 7.371 s → 7.451 s | +1.1% |
| M5 Max / E2B | 2.710 s → 0.030 s | 2.787 s → 2.824 s | +1.3% |
| M5 Max / 26B-A4B | 4.434 s → 0.051 s | 4.565 s → 4.726 s | +3.5% |

Singleton decode medians changed by less than 1% in all four scheduling campaigns. This is principally removal of queue blocking for arriving requests, not a general single-request generation speedup. On M2, the short peer's own decode also took longer when sharing the batch. Admit/fairness policy needs whole-stack validation before enrollment.

The two routes report different prefix-cache reuse (81 versus 92 for E2B/E4B and 81 versus 96 for 26B-A4B). This is an end-to-end routing comparison, not a microbenchmark with identical cache states. Matching visible greedy output is not proof of identical logprob distributions or comprehensive model quality.

## Correctness and disposition

The independent HTTP suites passed **195 of 196 checks**; one 26B-A4B default-baseline seeded-repeat control failed. Suite coverage: M2 E2B baseline/last-position 28, M2 E2B baseline/native 28, M4 E4B baseline/native 28, M5 E2B baseline/last-position and baseline/native 56, and M5 26B-A4B baseline/last-position 28/28 and baseline/native 27/28. They cover arithmetic, repeat requests, long retrieval, multiturn, JSON, finite logprobs, seeded repeatability, cancellation/drain/duplicate/recovery, concurrent image/text, subsequent recovery, readiness and clean exit. They do not qualify every protocol, model family or media modality.

The performance campaigns total 272 streams including warmups: 104 from M2/M4, plus 84 each from the two M5 model campaigns. The 514 host observations have constant pre-existing swap within each campaign and no foreign Python process above 512 MiB. Host and activation receipts are archived separately from correctness results. No production runtime change has been submitted or queued by this investigation. The 26B-A4B native timing row is exploratory rather than complete compatibility qualification: its ordinary baseline produced different text for two identical seeded sampled requests. The candidate route passed that check; the failure remains unresolved, not waived or rerun for a green result. Vector owns a separate reproduction on current source before claiming full model qualification.

Vector disposition: retain candidate 1 as a small capability-scoped implementation candidate, with architecture-dependent benefit; retain candidate 2 separately with explicit active-request and throughput budgets and a pending 26B-A4B baseline repeatability investigation. Their improvements cannot be added together: native text prefill already eliminates unused output positions, while the ordinary media path can still benefit from candidate 1.

M3 Ultra did not run inference: after owner-confirmed release of a stale reservation, the exclusive performance lock remained held by existing validation beyond a bounded 30-minute wait. The prepared environment was cleaned and no machine lease was held while waiting. M5 was subsequently acquired through a normal lease when its previous workload ended. E4B was already cached on M4; E2B and the pinned 15.37 GB 26B-A4B weights were fetched to M5's standard cache. M2 disk capacity allowed E2B only. All owned serving processes, temporary environments and worker leases were cleaned/released after testing; other tasks and resident services were preserved. M1/M3, the unified 12B family, 31B dense and other precisions remain unqualified.

One M5 E2B attempt had a process-observer exception when a process disappeared and its memory information became `None`. Those 84 original performance streams are excluded. After repairing only the observer, both entire ABBA campaigns were rerun. The separate E2B HTTP checks overlapped a model download and are used only for correctness, never performance. Failure dispositions are archived; no performance gate or answer requirement was relaxed.

## Reproduce and inspect

The fixture bundle contains the exact exploratory scripts, source/model/dependency metadata, HTTP results, output/activation receipts and host observations. It excludes private connection details, lease credentials and server logs. The original audit fixture supplies the frozen source experiment's prior provenance.

Extract `fixtures/2026-10-14-gemma-cross-chip/evidence.tar.gz` into a temporary directory and run `python3 summarize.py`. It checks the expected source/model hashes, versions, file counts, answers, output hashes, causal request ordering, single-position activation, shared-weight identity, host samples, process exits and 195 passing HTTP checks and the exact single failed baseline control; it recomputes `summary.json`. Inspect `manifest.json` for member and archive SHA-256 values. Use Python 3.12 and the stated dependency versions to rerun the scripts; their explicit scratch root and cached-model path file must be prepared for the new worker. Acquire a worker lease before inference and preserve the shared Hugging Face cache afterward.
