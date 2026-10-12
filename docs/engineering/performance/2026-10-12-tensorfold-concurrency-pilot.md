# TensorFold concurrent decoding pilot

Owner: Vector. Dates: 2026-10-11/12. Status: local pilot, lead review pending.
No runtime upgrade, engine scheduler change, PR, push, or release is included.

The pinned TensorFold 0.6.6 runtime at
`cb2ebf0540f42604e2759b2ddef497861e928248`, with MLX 0.32.3,
preserves greedy per-request generated token IDs for the measured Nemotron
3.5 Lightning and Qwen3.8-27B profiles at N=2, 4 and 8. Each capacity was
repeated three times against a solo reference: eight distinct code/prose
prompts, 256 generated tokens each. Two concurrent continued conversations
also restore kept prompt state with parity. Closing one in-process stream
after 16 tokens leaves its peer correct; the cancelled prefix matches solo.

Environment: Apple M3 Ultra, 256 GiB RAM, macOS 26.5.2 build 25F84,
Python 3.12.14, MLX-LM 0.31.3, Transformers 5.15.1. All model runs use
`scripts/large-model-run.py`, the shared host lock, and offline HF/Transformers.
The qualified snapshots already in the shared cache are used read-only.

| Profile | N | Rounds/generated token, repeat range | Peak streams/round | Aggregate tok/s, repeat range |
| --- | ---: | ---: | ---: | ---: |
| Nemotron | 1 | 0.3809–0.4121 | 1 | 32.42–40.51 |
| Nemotron | 2 | 0.2329–0.2549 | 2 | 42.17–66.58 |
| Nemotron | 4 | 0.1260–0.1724 | 4 | 59.65–162.51 |
| Nemotron | 8 | 0.1118–0.1172 | 8 | 88.95–189.31 |
| Qwen | 1 | 0.2700–0.2764 | 1 | 21.06–25.24 |
| Qwen | 2 | 0.1797–0.1890 | 2 | 51.62–54.04 |
| Qwen | 4 | 0.1045–0.1079 | 4 | 63.73–84.96 |
| Qwen | 8 | 0.0630–0.0645 | 8 | 75.84–79.00 |

Wall-clock variance exceeds the anticipated ±15%; Nemotron N=1/2 preceded
a host reboot and N=4/8 followed it. These ranges establish observed work
sharing, not a causal speed-up multiplier. Singleton controls preserve
parity but have broad timing spreads, including slower repeats under wider
configuration. A strict singleton no-regression performance bound is not
established. Serial stays the default with the existing HTTP execution path.

## Explicit opt-in and execution boundary

Set `RAPID_MLX_TENSORFOLD_LANES=2`, `4` or `8` before starting a qualified
TensorFold alias. Unset or `1` keeps serial behavior. Other TensorFold
profiles refuse N>1 before loading; invalid values also fail clearly.
Only `nemotron-3.5-lightning-tensorfold` and `qwen3.8-27b-tensorfold` are
qualified for this opt-in. No ordinary-backend CLI default changes.

The adapter passes `parallel=N`, `ChatApp(lanes=N)`, and the family's
`memory_fraction` for concurrency, following the pinned runtime's serve
construction. The runtime retains one model, one GPU engine thread and its
existing scheduler. Additional lanes hold separate request caches and share
forward work. Runtime admission controls memory pressure; this pilot did not
force memory exhaustion.

The HTTP builder accepts optional `BackendExecutionPolicy(capacity, executor)`
for a backend that owns its scheduler and can safely run request waits on
multiple threads. That explicit policy skips the global generation lock,
uses the backend-owned executor, and controls admission. Without it, all
existing callers retain the global lock, one `dflash-worker`, and existing
admission/refusal behavior. The existing deadline, deferred cleanup and
stream lease handle both policies. The builder disallows policy injection
without both custom generation callbacks. The capability block reports N.

A deterministic model-free test reproduced a shared-provider race: the first
request's `[1,2]` result could be replaced by the second's `[3,4,5]`. Replies
now retain request-local final token IDs and copy once; qualification-only
diagnostics are thread-local. Audit files remain disabled unless explicitly
configured.

## Real HTTP evidence and checks

On clean serving commit `bdf74b479`, the actual Rapid CLI served the Nemotron
alias on verified-free `127.0.0.1:18462`, with capacity eight and the shared
large-model lock. Three repeats compared eight solo requests against eight
concurrent non-streaming and SSE requests. Complete generated token arrays
and returned text match per request; shared forwards peak at eight streams.
Logged mean occupancy spans 5.93–6.46 for non-streaming and 5.98–6.22 for SSE.

In each repeat, a ninth request receives HTTP 503, `Retry-After: 1`,
`error.code: at_capacity`. Closing one SSE response cancels its generation
at 8, 8 and 5 tokens respectively; all seven peers finish with solo token/text
parity. Two continued conversations also pass HTTP regeneration parity after solo
continuation references, and restore
2,789–2,790 cached tokens each. The owned server shuts down normally and its
port is free afterward. No unrelated process is touched.

Validation: 297 tests passed and one pre-existing opt-in real-model DFlash
test was skipped. The affected default and concurrency contracts, timeouts,
capacity cleanup, prefix probe and GLM default contracts are covered.
`ruff check`, `ruff format --check`, and `git diff --check` pass for the
changed files. An initial prefix-probe test attempt with dirty serving sources
failed its intended provenance guard; the committed-source rerun passes.

Not qualified: sampled decoding, eight long simultaneous continuations,
maximum contexts, disk prefix persistence, restart restoration, other model
families, or Qwen through real HTTP. No GLM or DeepSeek model was loaded.
The lead engineer owns the ship decision and further performance qualification.

## Reproduction and retained evidence

The complete report, exact commands, preserved probe sources, per-request
token arrays, latency/memory measurements, runtime provenance, and HTTP audit
and round logs are retained in
`/private/tmp/tensorfold-lane-concurrency-pilot/REPORT.md` and its adjacent
artifacts. Nemotron snapshot:
`TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` at
`d9d758fb83953437f7263256b0d96157e2a348b8`. Qwen target:
`Vontra/Qwen3.8-27B-MLX-4bit` at `70ae7fac63274ff2eac54152031433374cb80f2f`,
with `z-lab/Qwen3.8-27B-DFlash2` at
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

From the repository root, with those checkpoints already cached:

```sh
PILOT=/private/tmp/tensorfold-lane-concurrency-pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
"$PILOT/venv/bin/python" scripts/large-model-run.py --working-set-gb 40 --lock-timeout 7200 -- \
  "$PILOT/venv/bin/python" "$PILOT/probe-nemotron-measured.py" --lanes 1 --output "$PILOT/n1.json"
"$PILOT/venv/bin/python" scripts/large-model-run.py --working-set-gb 40 --lock-timeout 7200 -- \
  "$PILOT/venv/bin/python" "$PILOT/probe-nemotron-measured.py" --lanes 8 \
  --baseline "$PILOT/n1.json" --output "$PILOT/n8.json"
# The HTTP harness starts its own server through the same lock wrapper,
# repeats all cases three times, then cleans up that owned process.
"$PILOT/venv/bin/python" "$PILOT/http_e2e.py" --port 18462 --lanes 8
```

The preserved in-process probes intercept only constructor arguments so they
can qualify the runtime independently of the serving opt-in gate. Some runs
occurred during adapter implementation; their recorded HEAD is not an exact
fingerprint of the transient working tree. The clean-commit real HTTP run
provides final integration evidence. Raw output transcripts are not committed.
