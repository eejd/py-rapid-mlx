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

Vector should profile the fresh M4 TensorFold prefill path: the stable local
7.8-second result separates it from public routing overhead and from the working
prefix cache. Atlas should retain the fresh-prompt latency launch gate. Harbor
still needs final signed-package and contributor-ledger acceptance.

Private screenshots, the real-time CLI video and sanitized measurements are
retained separately. No credentials, private machine endpoints or recordings are
committed here. After stopping all test processes, the owning-app test key, node
cache, temporary model copies, isolated preferences, LaunchAgents and remote
scratch were removed. The machine was released and live inventory showed
available with no lease. Existing models, services and the Studio model cache
were preserved.
