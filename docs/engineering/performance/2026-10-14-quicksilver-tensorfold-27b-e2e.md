# QuickSilver Qwen 27B TensorFold E2E

Owner: Atlas; backend follow-up: Vector; packaging/rollout follow-up: Harbor.
Test date: 2026-10-14 UTC. Scope: explicitly selected TensorFold text serving
through the production QuickSilver consumer API. Cancellation propagation is
explicitly out of scope by owner decision.

## Environment and configuration

- Borrowed Apple M4 Pro, 48 GiB, macOS 26.5.1; no resident Rapid service.
- Signed Desktop 0.16.1 sidecar code, release commit
  `df1f4db1fc0548ec97c6376739a76f6330e9b000`.
- TensorFold 0.6.6, exact git revision
  `cb2ebf0540f42604e2759b2ddef497861e928248`, installed into an external
  task-owned Python extension directory, with MLX/MLX Metal 0.32.3.
- Target: `Vontra/Qwen3.8-27B-MLX-4bit`, revision
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; runtime packs it to 4 bits.
- Existing qualified weights copied into the worker's default HF cache.
  Offline resolution; disk prompt snapshots disabled; no thinking.
- Pool capacity explicitly set to **one** slot, matching this adapter's
  serial admission. TensorFold model loading and runtime provenance checks
  succeeded. `/healthz` reported `engine: tensorfold-qwen3.8-27b`,
  `algorithm: dflash2`, `mode: single-user-serial`, both pinned revisions and
  `max_concurrency: 1`.

The signed app bundle itself remained unchanged and passed deep/strict
codesign verification after testing. This is **not** an unmodified,
out-of-box Desktop acceptance: its shipped runtime contains MLX 0.32.2 and
no TensorFold distribution, whereas the qualified adapter requires 0.32.3.
The external extension directory and wrapper are test setup dependencies.

## Reproduction

In an environment containing Rapid 0.16.1 and the exact qualified optional
runtime, use the existing alias rather than the ordinary MTP alias:

```sh
python -m pip install 'tensorfold @ git+https://github.com/ashhart/TensorFold.git@cb2ebf0540f42604e2759b2ddef497861e928248' 'mlx==0.32.3'
rapid-mlx share qwen3.8-27b-tensorfold --quicksilver \
  --quicksilver-model qwen3.8-27b --provider-key-stdin \
  -- --max-num-seqs 1
```

Supply the provider key over stdin; use a consumer key to call
`https://api.quicksilverpro.io/v1/chat/completions`, model `qwen3.8-27b`.
For streams request usage and require HTTP 200, nonempty content, terminal
usage and `[DONE]`. Temperature is zero; short calls have a 32-token cap.

Find an opening routed to the test node, then preserve that first user
message across follow-ups so pool conversation affinity is stable. Send a
three-message request asking for a numbered list of 500 animals, capped at
512 tokens. Keep reading it through DONE. At 250 ms after its first content,
send a short follow-up on the same opening asking for the number 4. Repeat
three waves. Also send three independent short calls concurrently.

For prefill, append `Reference text: `, 140 repetitions of
`alpha beta gamma delta. `, and ` Reply with only the number 4.` to that
conversation; repeat it unchanged. Gateway billable input is 744 tokens;
the target tokenizer renders 755 native input tokens.

A temporary observer on the provider's existing tunnel methods recorded
only synthetic-message SHA-256, timestamps, HTTP status and response IDs.
It did not change inference, admission or routing. Match those hashes to
consumer payloads: consumer response IDs are rewritten by the gateway and
cannot establish node identity. Requests absent from the node trace are
not attributed to this node.

## Measured results

| Workload | Public first content | Test-node evidence |
| --- | --- | --- |
| Idle short stream | 1.029 s | Matched node trace; 0.407 s inside node/tunnel boundary |
| Three long generations | 1.454 / 1.596 / 1.681 s | All matched; node first content 0.950 / 0.953 / 0.954 s |
| Short call while long call active | 1.576 / 1.430 / 1.067 s | None reached this busy single-slot node; served elsewhere |
| Three simultaneous short calls | 0.854 / 1.719 / 1.723 s | One reached test node, with 0.357 s node TTFT |
| 744 billable input tokens, successful runs | 8.131 / 8.036 / 8.230 / 8.132 s | Node first content approximately 7.3 s, even for repeats |
| Short call after automatic reconnect | 1.214 s | Successful response with DONE/usage |
| Short call after stopping test node | 1.424 s | Successful elsewhere; does not prove whole-pool cloud fallback |

Long calls completed in 17.038 / 17.408 / 17.705 s, with `length` finish
reason. Gateway-recounted completion tokens were 454 / 438 / 438, so do
not treat those as native decoder throughput measurements.

Twenty consumer calls: **18 passed, two failed**. The failures were:

1. A forced tool call reached this node and returned HTTP 400; the consumer
   also received HTTP 400. `/healthz` explicitly reports tools/media/grammar
   unsupported. The pool did not transparently provide tool compatibility.
2. One repeated longer-input stream received content but lost the connection
   at 18.192 s with no DONE or usage. Its trace overlaps a natural tunnel
   loss and reconnect, and contains another attempt for the same payload.
   This does not prove which intermediary reset the customer connection.
   Three subsequent unchanged-input probes passed, as did a short probe.
   A later successful retry does not erase the failed stream.

Successful calls reported approximately $0.00392 total usage cost; this is
not a complete balance delta or a contributor-settlement verification.

## Interpretation and release disposition

The earlier non-TensorFold overlap result is a control, not the intended
release configuration: ordinary MTP on two advertised slots produced
21.053–35.652 s short-call TTFT, whereas disabling speculation produced
1.121–1.213 s. It localizes that control's starvation to the ordinary MTP
path. No scheduler change was applied, and it is not evidence against
TensorFold.

TensorFold plus one advertised slot avoids this waiting pattern by letting
QuickSilver route concurrent work elsewhere. This does not establish
multi-request decoding inside TensorFold or behavior when all other nodes
and cloud capacity are unavailable. The long-input latency is largely Mac
prefill, not transport: approximately 7.3 of 8.1 s is inside the node.
Identical repeats did not demonstrate effective prompt reuse. Tokenizer-only
inspection found a valid 748-token history boundary; the runtime cache
behavior needs further diagnosis, rather than assuming boundary detection
failed or claiming that TensorFold eliminates prefill cost.

**Manual TensorFold text-only service is functional. The stock 0.16.1
Compute/Earn experience is not ready to be described as TensorFold-backed.**
Required integration work: package the qualified runtime/MLX combination;
select `qwen3.8-27b-tensorfold` in Desktop and CLI pool defaults; advertise
one slot; and define how tools/media/grammar requests are excluded or routed
to compatible execution. Today both `ShareComputeModels.swift` and
`CATALOG_DEFAULT_ALIAS` select `qwen3.8-27b-4bit`, and the pool otherwise
defaults to two slots. Investigate the observed incomplete stream and repeat
the acceptance after these integration changes. Contributor settlement and
Desktop GUI restore remain unverified by this CLI test.

The test node was stopped, task-owned credentials and model copies removed,
and the borrowed machine released. No production deployment or pool policy
change was performed.
