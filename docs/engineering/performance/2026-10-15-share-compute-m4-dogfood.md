# Share Compute on M4 Pro: current-main dogfood

## Outcome

The CLI and Desktop sharing lifecycle worked against the live QuickSilver pool.
Do not treat this as full launch acceptance: a fresh long prompt still takes
8.53-8.76 seconds to first text through the public API. Repeating that prompt is
fast, so the remaining delay is fresh prefill rather than a missing cache toggle.
Contributor ledger settlement and the final signed/notarized package remain
unverified.

## Build and environment

- Main at task start: `6fb35fbb24c5b8bb151d4fe2a7d7858adfc6d146`.
- Apple M4 Pro, 12 CPU / 16 GPU cores, 48 GiB, macOS 26.5.1 (25F80).
- Fresh `apps/rapid-mac/scripts/build.sh` build, Release configuration, ad-hoc
  signature, source distribution. Embedded source stamp matches the commit.
  App signature verification and bundled CLI version/help passed on the worker.
- TensorFold 0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`, MLX 0.32.3.
- Target: `Vontra/Qwen3.8-27B-MLX-4bit`, revision
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Draft: `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- The exact snapshots were copied from existing Studio caches into the worker's
  default HF cache. This does not qualify first-download UX.

## Reproduction and measurements

Start with the standard foreground command and enter the provider key at the
hidden prompt:

```sh
rapid-mlx share qwen3.8-27b --quicksilver --worker m4-release-dogfood
```

The command automatically chose `qwen3.8-27b-tensorfold`. Its actual child used
`--disable-disk-caches --max-num-seqs 1 --no-thinking`, and health reported the
pinned TensorFold target/draft pair. Tests used an isolated port and Desktop
status session for bookkeeping. No account key was placed in command arguments.

All requests were sequential, temperature 0, maximum 32 output tokens,
streaming with usage included. TTFT means the first nonempty content delta.
Each wave started with a distinct short opening asking for the number 4, then
appended an assistant answer and a new user message containing 140 repetitions
of `alpha beta gamma delta. ` plus another request for 4. Wave markers prevented
reuse of the new long segment. Repeat reused the exact request; follow-up added
an assistant answer and a short new user turn.

The long requests contained about 780 tokens locally and 762 through the public
path. These are fresh long segments after a short opening, not cold model loads.
The local timing run used the unchanged bundled sharing CLI. The public run
rejoined the cached node with an observational relay wrapper recording only
request IDs, message hashes and response status. All nine public long/repeat/
follow-up samples below matched this worker; the serving code was unchanged.

| Case | Localhost samples, seconds | Public samples, seconds | Median local / public |
| --- | --- | --- | --- |
| Short opening | 0.7298, 0.6431, 0.6438 | Not a three-sample comparison | 0.6438 / - |
| Fresh long segment | 7.8087, 7.8122, 7.8031 | 8.7601, 8.7358, 8.5311 | 7.8087 / 8.7358 |
| Identical repeat | 0.2136, 0.2143, 0.2138 | 1.0563, 1.4687, 1.0262 | 0.2138 / 1.0563 |
| Follow-up turn | 0.3496, 0.3505, 0.3507 | 1.2401, 1.0911, 1.3276 | 0.3505 / 1.2401 |

All 12 local samples returned HTTP 200, DONE and the expected answer. Eleven
public text cases, including a routing probe and non-stream case, were positively
attributed to the worker and completed successfully. Other routing probes are
excluded from attributed timing claims. Short answers do not qualify sustained
decode throughput or concurrent load.

One tools probe ended in a client AttributeError without a validated response;
it is not counted as a pass. A later explicit function request while the worker
was offline returned a valid tool call. This run does not qualify pool tool
failover; this TensorFold profile advertises tools/media/grammar unsupported.

## Real Desktop and CLI workflow

- CLI first registration, hidden key entry, model loading, Metal warmup and relay
  connection were recorded in a real SSH terminal. Ordinary CLI rejoin reused the
  node and explicitly reported that no provider key was needed.
- Desktop used a preferences-isolated copy with the same main executable and
  bundled engine. A temporary in-process diagnostic library captured only the
  app's own windows and invoked native controls. No preparation, online or
  receipt state fixture was enabled. The engine was not instrumented for GUI UI
  capture. No screen-recording/Accessibility permission or TCC database change
  was required.
- SSH-launched Desktop correctly refused key storage with a locked/nonwritable
  Keychain. The normal logged-in GUI session had an unlocked/writable Keychain;
  first setup saved the key and actually joined the pool successfully.
- A resident personal Gemma model was unloaded before the Qwen27 sharing child
  started. Sharing used the TensorFold profile exclusively. Local chat displayed
  its sharing pause and disabled the composer.
- Stop removed the sharing process and listening port, with no automatic restore
  of the personal model. Restart did not request a key or pass
  `--provider-key-stdin` / `--reregister`. Stopping during preparation also left
  no serving process or delayed join.
- Provider/consumer account tokens were absent from captured command arguments,
  private app preferences, node cache, serving log and GUI diagnostic log. Node
  credentials remained in their intended private cache during the test; prompt
  caches were disabled on disk while in-memory prefix reuse remained effective.

## External oMLX reference (checked 2026-10-15)

These are community-submitted results hosted by oMLX, not our own controlled
A/B measurements. The closest hardware match found was M4 Pro / 16 GPU cores /
48 GB. The reported model is `mlx-community--Qwen3.8-27B-4bit`, rather than our
pinned Vontra target. oMLX 0.6.3rc2 used DFlash, thinking disabled, in-memory
DFlash cache enabled, and SpecPrefill / ANE prefill disabled:

| Prompt | Reported TTFT | Prefill rate | Source |
| --- | --- | --- | --- |
| 1,024 tokens | 10.509 s | 97.4 tok/s | [oMLX result](https://omlx.ai/benchmarks/performance/kfn85rp2) |
| 4,096 tokens | 42.697 s | 95.9 tok/s | [oMLX result](https://omlx.ai/benchmarks/performance/nj17h1k1) |

These recipes do not establish an oMLX + TensorFold result. No public result
matching our exact target/draft revisions, TensorFold runtime, hardware and
prompt was identified. Do not call these an exact same-model comparison or use
them to claim a measured Rapid speedup. They are consistent with roughly
100 tok/s fresh prefill on this hardware: a rough 780 / 97.4 estimate is 8.0 s,
close to our local 7.8 s. This estimate is not another measured benchmark.

The public reference does not demonstrate subsecond fresh 1K prefill on this
configuration. It also does not qualify cached TTFT. Our working repeat cache
and the fresh-prompt launch gate remain separate facts. A definitive comparison
requires the same leased M4, exact weights, identical prompts and TTFT definition
across fresh, identical-repeat and follow-up cases.

### Latency disposition after deeper review

For this single-request M4 Pro / 16 GPU / 48 GB configuration, the measured
7.8-second first-content latency is consistent with normal fresh-prompt
processing. It is not demonstrated to be a Rapid regression. This is an
engineering assessment from the existing receipts and source audit, not a new
same-machine oMLX A/B or a claim of a hardware performance ceiling.

- All three fresh requests had 780/780/782 total prompt tokens and zero visible
  reasoning characters; model loading and warmup happened before timing.
  TTFT ranged from 7.8031 to 7.8122 seconds (9.1 ms spread). Requests were
  sequential and used localhost, removing the public relay and concurrent
  request queue from this timing. There is no archived GPU trace to exclude
  every possible device-level contention or kernel inefficiency.
- `prompt_tokens / TTFT` is 99.8-100.2 tok/s. This is an effective request rate,
  not an instrumented GPU prefill rate: the usage receipts did not record the
  actual cached-token count, and TTFT includes first-token work and text
  emission. The preceding short opening was 54-56 tokens.
- The 0.2138-second identical-repeat median and 0.3505-second follow-up median
  demonstrate working reuse and rule out a universal seven-second frontend
  or streaming wait. Short openings had a 0.6438-second median.
- The pinned adapter uses the actual TensorFold family, 4-bit drafter packing,
  one lane and the message-aware prefill plan. It submits token jobs and emits
  incoming chunks; it does not perform model loading in the per-request path.
  The inspected scheduler's idle queue wait is 20 ms, and adapter queue reads
  wake on arrival (their 50 ms timeout is not a fixed per-chunk sleep).
- Another same-hardware DFlash recipe reports 103.2 tok/s at 4K:
  [oMLX reference](https://omlx.ai/benchmarks/performance/unjqi1po).
  At the observed 95.9-103.2 tok/s reference rates, processing 780 tokens
  estimates 7.56-8.13 seconds before separately modeling cache and first-token
  overhead. Our 7.81 seconds is within that rough envelope. The recipes differ
  in prompts, model revisions and some settings; this is not a speedup claim.

Disposition: if the product accepts this hardware's fresh-prompt latency, do
not hold release solely to pursue an arbitrary subsecond fresh-prompt target.
Proceed to release acceptance for the signed/notarized artifact and contributor
settlement. This does not certify full launch readiness. Do not claim industry
best performance, negligible remaining optimization headroom, or that long
prompts / concurrent users will have the same 7.8-second latency. The repeated
synthetic prompt and short output do not qualify representative production
workloads. Future prefill tuning remains separate from this release assessment.

## Limits and handoff

Real API replies contained usage/cost fields, but no contributor read key was
available to verify account-wide earnings or settlement. The source/ad-hoc bundle
is not acceptance of the eventual Developer ID/notarized candidate.

A separate diagnostic cleanup executable with a different ad-hoc identity stalled
in the development-key deletion fallback: `SystemKeychain.delete -> write ->
SecItemUpdate -> CSSM_ChangeKeyAcl`. Do not count that auxiliary attempt as a
successful removal or generalize it to a shipping GUI hang. Cleanup inside the
original owning app succeeded, and both development slots subsequently returned
`errSecItemNotFound`. Changed-signature/denied-ACL failure bounds remain a focused
follow-up for Pixel/Atlas before making an unconditional nonblocking claim.

Vector can profile the fresh M4 TensorFold prefill path as follow-up work: the
stable local 7.8-second result separates it from public routing overhead and
from the working prefix cache. Following the latency disposition above, Atlas
should record acceptance of the expected fresh-prompt experience rather than
treating this measurement alone as a demonstrated regression. Harbor still
needs final signed-package and contributor-ledger acceptance.

Private screenshots, the real-time CLI video and sanitized measurements are
retained separately. No credentials, private machine endpoints or recordings are
committed here. After stopping all test processes, the owning-app test key, node
cache, temporary model copies, isolated preferences, LaunchAgents and remote
scratch were removed. The machine was released and live inventory showed
available with no lease. Existing models, services and the Studio model cache
were preserved.
