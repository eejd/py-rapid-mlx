# TensorFold lane concurrency pilot: final evidence

**Verdict: works with caveats.** Greedy parity and shared runtime rounds pass for Nemotron and Qwen in-process and through real HTTP; the paired lone-stream timing and memory trade-offs are reported explicitly below. Serial remains the default.

Owner: Vector. Host: shared Mac Studio M3 Ultra, 256 GiB; macOS 26.5.2 build 25F84; Python 3.12.14, TensorFold 0.6.6 revision cb2ebf0540f42604e2759b2ddef497861e928248, MLX 0.32.3, MLX-LM 0.31.3, Transformers 5.15.1. Final serving source commit `d8331b1ce74cd2dc29afed279d5b377828d2b72d`, rebased on fetched `origin/main` d32714644. The later evidence commit changes documentation only. Every performance number below is refreshed from that serving code; prior pre-reboot and pre-review data are archived separately and omitted here.

## Why Rapid was serial

Commit `249b9bd0bc39f589f3e341a067d7a73d17d8ca86` (`fix: cap qwen27 accelerated admission at one`, #3934) introduced `_MAX_CONCURRENT_REQUESTS = 1`. Its recorded reason is: "TensorFold exposes one serial lane. Keep admission aligned with that runtime capacity even when the shared CLI default is much larger." It added a second-in-flight request refusal test with HTTP 503 / Retry-After. This was alignment with Rapid's chosen constructor capacity, not recorded evidence that the pinned runtime cannot batch.

Commit `3d00b3032e7e90d829ebe0ce88cc20e05a960c51` (#4264) added `options = {"lane_kernels": "auto", "parallel": 1}` and `ChatApp(lanes=1)` for family profiles. `2026-10-06-tensorfold-family-profiles-qualification.md` and the Bonsai, Gemma, DeepSeek qualification reports all use `--parallel 1`. `2026-10-10-nemotron-prefix-mvp.md` explicitly qualifies one request at a time and excludes concurrency. No concurrency parity failure was found in these inspected commits/docs. The GLM qualification's "serial" comparison means drafts off versus drafts on, rather than N=1 versus N>1.

## What lanes and parallel mean on this pin

`ChatApp` builds **one LaneEngine and one scheduler thread**, with up to N live streams/open prefills. `lanes` does not replicate weights or create N GPU worker threads. The family Mac loaders accept `parallel` through extra kwargs and generally ignore it; the decisive Mac scheduling parameter is `ChatApp(lanes=N)`. The CLI couples them and also passes `memory_fraction` when N>1. Rapid's pre-pilot family adapter omitted `memory_fraction`; passing it in the pilot follows the pin's own CLI and enables memory-based concurrent admission plus the stream growth gate.

Each lane holds its own model cache (KV/recurrent/MTP or draft state), plus retained checkpoint copies when configured. `checkpoint_slots=None` scales to `max(3*N, 8)`, while Rapid's byte budget stays capped at 16 GiB. Shared-forward workspace is additional to persistent per-stream state. The runtime measures short/long cache sizes, growth per token, prefill workspace and widest shared-round workspace at startup. The admission formula projects current held memory, live-stream growth, a new stream's cache and the larger of prefill/shared-round work. Its budget also accounts for memory held elsewhere on the Mac. These estimates are not N copies of the model.

A new lane that does not fit beside live streams is held to wait; its admission refusal counter can increment without an HTTP refusal. Prompt memory can evict retained checkpoints, and a request that cannot fit alone receives a runtime RequestError. During growth, the existing StreamGate pauses newer streams, or ends the newest request with an explicit out-of-memory RequestError if the older streams cannot grow. Those pressure behaviors are source-inspected; intentionally forcing memory exhaustion is outside this shared-host pilot.

The shared-round sizing routine probes the model's maximum shared forward (e.g. up to 64 streams for Nemotron), not the configured N. Its printed "at N streams" description therefore should not be read as a measured N-stream live workload. Cell peaks are the actual workload measurements.

## Family support from source inspection

This branch has five **registry** profiles plus the separate Qwen and GLM profiles (seven aliases total), rather than five aliases total.

| Profile | Mac shared-forward source support on 0.6.6 | Status in pilot |
| --- | --- | --- |
| nemotron-3.5-lightning-tensorfold | `NemotronH.hidden_rows`, up to 64 streams / 128 rows; widest exact windows checked at load | Measured |
| bonsai2-27b-tensorfold | Uses Qwen35Family; pre-M5 row path up to 32 streams / 32 rows, load-time exact windows | Source only |
| qwen3.8-flash-next-tensorfold | Up to 32 streams / 64 rows; `streams_exact` / `max_streams=1` fallback if load-time shared checks fail | Source only |
| gemma-4-26b-tensorfold | Up to 16 streams / 64 rows; load-time shared exactness failure sets `max_streams=1` | Source only |
| deepseek-v4-flash-tensorfold | Inherits GLMFlash shared protocol; up to 16 rows/streams, load-time serial fallback | Not loaded per brief |
| qwen3.8-27b-tensorfold | Qwen35Family: pre-M5 row path up to 32 streams / 32 rows | Measured in-process and real HTTP |
| glm-5.3-flash-tensorfold | GLMFlash shared protocol; up to 16 rows/streams, load-time serial fallback | Not loaded per brief |

No registry family unconditionally rejects `parallel > 1` in its Mac loader. That establishes structural support, **not measured parity or performance** for the untested families. If a family's exactness self-check fails, N requests may remain live while separate forwards preserve correctness; useful shared-forward acceleration is not available in that configuration. A one-row-only exactness result also disables shared family forwards in the engine.


## HTTP execution policy and shutdown fix

The optional `BackendExecutionPolicy(capacity, executor)` injected into `_build_app` uses the backend-owned executor and admission capacity and skips the module-global generation lock. The pinned runtime still owns batching. All existing callers without the policy retain the same global lock, one worker, deadlines, admission and 503/Retry-After behavior. Characterization tests cover both modes and stream types; the existing DFlash tests pass. The builder requires both injected generation callbacks.

`RAPID_MLX_TENSORFOLD_LANES=2|4|8` explicitly opts in on Nemotron or Qwen. Unset/1 remains serial; other profiles reject >1 before loading. Loaders pass lanes=N, parallel=N, and the runtime CLI's model_fraction for >1; capabilities report N. Provider results use a local final token list; diagnostics are thread-local. The prior result race was reproduced before the fix.

The review's blocking bug was reproduced with bounded model-free streaming and non-stream tests: `close hung after scheduler stopped without terminal chunks` (2 failed before the fix). Opt-in provider jobs now register their Cancellation under the backend closing lock before scheduler submission, check it while draining, and unregister in a finally even if submission fails. Close marks closing atomically, cancels all registered jobs, stops the runtime scheduler, and joins its executor with wait=True. Jobs still queued on that executor refuse submission after closing. Tests cover scheduler-queued, running, executor-queued, submission failure and registration-versus-close races.

A second bounded regression reproduced deferred SSE cleanup failing with `cannot schedule new futures after shutdown`, leaving one reservation held. For the injected concurrent policy only, a Python generator in GEN_CLOSED or GEN_CREATED state can close without executing its body or submitting work; the lease then releases its reservation. Default callers retain the original cleanup branch. Opt-in iterators are also tracked and finalized after the owned executor joins, including streams suspended between pulls; a separate bounded regression reproduced and fixed that retained-job case. Both deferred stream and non-stream reservations reach zero after shutdown in the tests. No detached decoding thread or wait=False workaround is used. The first real HTTP attempt on the iterator-tracking change failed during cancellation with `audit had 7 requests, expected 8`: closing an outer stream had left its inner tracked iterator retained. A model-free reproduction confirmed the retained active job. Explicit concurrent-path inner-iterator closure fixes it; the successful HTTP reruns below use that committed fix. The failed run and pre-fix test are preserved under pre-client-close-measurements/ and client-close-before.log.

**Existing serial-path follow-up:** its provider cancellations are also unregistered. Its HTTP worker belongs to the module-global executor, so backend.close stops the scheduler and returns while that worker stays blocked awaiting a terminal chunk. A model-free characterization proves this existing leak and explicitly cancels the test job afterward. The serial path was not widened to fix it. ThreadPoolExecutor interpreter-exit joining could extend the leak into a process-exit hang (source-inferred); that serial process-exit behavior was not measured. Lead engineer: retain this as a separate lifecycle follow-up before treating serial timeout/shutdown as qualified.

## Nemotron 3.5 Lightning: core measurements

Eight distinct code/prose prompts produce 2,048 tokens per cold cell. Ranges cover three repeats; per-stream tok/s and token-ID TTFT span all requests. Admission refusals across these measured cells: 0. No intentional memory-pressure run.

| N | Rounds/token | Mean streams/round | Peak streams/round | Per-stream decode tok/s | Aggregate tok/s | TTFT ms | MLX peak GiB | Parity | Repeats | Rounds per repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3838–0.4189 | 1.00–1.00 | 1 | 44.74–123.54 | 55.59–68.06 | 87.51–249.33 | 18.28–18.37 | pass | 3 | 786,812,858 |
| 2 | 0.2949–0.3164 | 1.69–1.80 | 2 | 37.67–78.68 | 76.43–92.68 | 142.22–545.63 | 18.68–18.72 | pass | 3 | 625,604,648 |
| 4 | 0.1240–0.1411 | 2.98–3.18 | 4 | 23.90–55.53 | 98.67–110.49 | 173.02–1147.95 | 19.85–20.26 | pass | 3 | 263,289,254 |
| 8 | 0.1045–0.1104 | 5.61–6.35 | 8 | 15.41–36.11 | 114.57–146.83 | 183.01–2142.95 | 20.16–20.30 | pass | 3 | 214,218,226 |

### Lone request: five interleaved load pairs

A=Rapid HTTP server lanes=1, B=Rapid HTTP server lanes=8. Five fresh-server pairs run A B A B, serving the same code and prose requests one at a time over real non-streaming TCP HTTP, greedy, 256 generated tokens each. Both returned HTTP texts and original generated token IDs match. Existing prompt-cache behavior is enabled; these are two distinct new histories per fresh server. Observational wrappers around Rapid's original backend constructor and provider iterator record metrics while preserving their calls and outputs; runtime source is untouched. The core-matrix and earlier SDK comparison loads are excluded from this paired HTTP table. Each entry is A/B; all lone rounds have one stream. Rows are actual RoundStats.rows, not maximum window width. Decode ms/token is `(last-output time - first-token time)/(tokens-1)`; it includes provider drainage after the last token. Positive B−A means the opt-in is slower. Each paired token sequence and returned HTTP text matches exactly. The raw client elapsed time is also retained; the decode metric isolates original provider token timing rather than response serialization.

| Pair | Prompt | Rounds A/B | Mean rows/round A/B | Decode ms/token A/B | B−A ms/token | B−A percent | Parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | code | 78/70 | 3.97/4.56 | 9.321/14.694 | +5.373 | +57.7% | pass |
| 1 | prose | 129/113 | 3.09/3.92 | 16.257/22.148 | +5.891 | +36.2% | pass |
| 2 | code | 67/75 | 4.90/4.44 | 9.058/15.661 | +6.603 | +72.9% | pass |
| 2 | prose | 117/121 | 3.56/3.46 | 19.664/22.673 | +3.009 | +15.3% | pass |
| 3 | code | 68/80 | 4.76/3.94 | 13.279/15.533 | +2.254 | +17.0% | pass |
| 3 | prose | 120/106 | 3.28/4.93 | 20.343/14.026 | -6.317 | -31.1% | pass |
| 4 | code | 74/72 | 4.23/4.42 | 13.773/14.185 | +0.413 | +3.0% | pass |
| 4 | prose | 127/112 | 3.18/4.19 | 21.783/13.412 | -8.371 | -38.4% | pass |
| 5 | code | 70/77 | 4.71/3.90 | 14.953/8.891 | -6.062 | -40.5% | pass |
| 5 | prose | 130/117 | 3.07/3.71 | 22.301/20.837 | -1.464 | -6.6% | pass |

Across the ten paired prompt observations, median B−A is +1.333 ms/token (+9.1%); mean +0.133 ms/token; spread -8.371–6.603 ms/token and -40.5–72.9%. The positive median is an observed lone-request cost of this opt-in in the campaign. The paired spread includes both faster and slower runs, so these observations do not isolate a causal configuration penalty or establish a universal no-regression guarantee. No kernel or calibration tuning was applied.

### Kernel and row calibration

| Pair | Exact width A/B | Engine (batch rows, max streams) A/B | Calibrated 1-row ms A/B | Calibrated exact-width ms A/B |
| --- | --- | --- | --- | --- |
| 1 | 16/16 | (128, 64)/(128, 64) | 18.066/7.319 | 75.246/50.019 |
| 2 | 16/16 | (128, 64)/(128, 64) | 33.593/10.425 | 73.279/44.093 |
| 3 | 16/16 | (128, 64)/(128, 64) | 33.435/33.199 | 63.707/73.993 |
| 4 | 16/16 | (128, 64)/(128, 64) | 10.066/20.638 | 54.880/69.572 |
| 5 | 16/16 | (128, 64)/(128, 64) | 22.146/27.798 | 50.117/65.940 |

The observed kernel path and hard row limits are identical between A and B. Exact width is the per-stream verified window limit; the engine shared-row budget is a separate limit (Qwen, for example, observes exact width 128 with shared batch rows 32). The runtime freshly times exact windows and shared rows at every load; those cost tables change across both configurations on this host, and adaptive decode row choices change as the lone-request table shows. The opt-in adds runtime concurrent-admission memory probes after family construction; it does not select a different family kernel configuration. This explanation combines observed calibration/rows with source inspection; a separate causal attribution of timing differences was not measured.

### Memory cost of enabling eight lanes

All values below are GiB except the last column (MiB). A/B order again. Idle resident is MLX active+cache bytes immediately after the loader reports ready. Idle peak is the MLX API value then: the runtime internally resets peaks while probing, so it is not a process-wide loading high-water mark. Each lone request resets the MLX peak first; request resident is active+cache at completion (cache bytes were recorded separately). Host RSS and whole-host pressure are not these metrics. Raw bytes remain in each JSON.

| Pair | Idle active A/B | Idle cache A/B | Idle resident A/B | Idle API peak A/B | Request resident range A/B | Request peak range A/B | Idle resident B−A MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 17.3456/17.3456 | 0.0000/0.0000 | 17.3456/17.3456 | 20.645/20.645 | 17.3452–17.3452/17.3452–17.3452 | 17.835–18.351/17.837–17.837 | -0.00 |
| 2 | 17.3456/17.3456 | 0.0000/0.0000 | 17.3456/17.3456 | 20.645/20.645 | 17.3452–17.3452/17.3452–17.3452 | 17.814–17.837/17.814–18.351 | +0.01 |
| 3 | 17.3456/17.3456 | 0.0000/0.0000 | 17.3456/17.3456 | 20.645/20.645 | 17.3452–17.3452/17.3452–17.3452 | 17.814–17.837/17.837–18.351 | +0.00 |
| 4 | 17.3456/17.3456 | 0.0000/0.0000 | 17.3456/17.3456 | 20.645/20.645 | 17.3452–17.3452/17.3452–17.3452 | 17.814–17.835/17.835–18.306 | -0.01 |
| 5 | 17.3456/17.3456 | 0.0000/0.0000 | 17.3456/17.3456 | 20.645/20.645 | 17.3452–17.3452/17.3452–17.3452 | 17.814–18.305/17.790–17.814 | +0.00 |

### Real HTTP and shutdown

Verified-free port 18472; actual Rapid CLI, N=8. Three repeats compare every complete concurrent non-stream and SSE token array and returned text against the same solo HTTP request. Every concurrent mode observes shared rounds; cancel-one-stream verifies its prefix and all seven remaining full answers; the ninth simultaneous request gets existing 503 / Retry-After: 1 / at_capacity. Two continued conversations restore kept prompt state and match solo regeneration. HTTP references populate the existing prompt cache before regeneration, so these HTTP aggregate rates establish observed transport-path performance with warmed state, not a cold-run speed-up; the cold matrix above is the separate throughput workload. Non-streaming HTTP TTFT is not observable at the client and is marked accordingly; SSE TTFT measures first visible content, which can follow first generated token.

| HTTP case | Logged rounds per repeat | Mean streams/round | Peak streams/round | Elapsed seconds range | Aggregate tok/s | Visible TTFT ms | Parity/contract | Repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solo | 834,884,900 | 1.00–1.00 | 1 | 25.40–28.68 | 71.41–80.62 | not observed | pass | 3 |
| nonstream | 212,217,219 | 6.07–6.34 | 8 | 13.04–14.09 | 145.36–157.07 | not observed | pass | 3 |
| stream | 211,228,209 | 5.84–6.48 | 8 | 11.65–12.76 | 160.45–175.83 | 191.15–1913.62 | pass | 3 |
| cancel_and_overflow | 197,202,202 | 5.45–5.65 | 8 | 11.71–13.11 | 137.41–153.41 | not observed | pass | 3 |
| cache_continue | 136,131,132 | 1.48–1.51 | 2 | 5.44–5.94 | 86.20–94.15 | not observed | pass | 3 |

The shutdown probe returns HTTP 504 after a 0.75-second request deadline while its non-stream worker remains active; the final audit contains 85 generated tokens, proving mid-decode termination. SIGTERM to the owned wrapper yields a server shutdown in 0.327 seconds, verified port-free with no surviving owned process. Process IDs and response body are in the HTTP artifact. No unrelated process was signalled.

## Qwen3.8-27B: core measurements

Eight distinct code/prose prompts produce 2,048 tokens per cold cell. Ranges cover three repeats; per-stream tok/s and token-ID TTFT span all requests. Admission refusals across these measured cells: 0. No intentional memory-pressure run.

| N | Rounds/token | Mean streams/round | Peak streams/round | Per-stream decode tok/s | Aggregate tok/s | TTFT ms | MLX peak GiB | Parity | Repeats | Rounds per repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2661–0.2661 | 1.00–1.00 | 1 | 16.85–63.74 | 22.75–30.39 | 415.90–628.33 | 15.93–16.23 | pass | 3 | 545,545,545 |
| 2 | 0.1816–0.1885 | 1.65–1.68 | 2 | 20.16–74.36 | 47.42–93.32 | 248.75–1064.96 | 16.37–18.59 | pass | 3 | 372,374,386 |
| 4 | 0.1035–0.1055 | 3.02–3.04 | 4 | 15.27–29.58 | 59.38–63.12 | 375.82–2207.62 | 16.99–18.55 | pass | 3 | 216,213,212 |
| 8 | 0.0630–0.0640 | 5.17–5.43 | 8 | 8.15–15.13 | 58.57–64.24 | 432.42–4886.97 | 18.32–18.57 | pass | 3 | 131,130,129 |

### Lone request: five interleaved load pairs

A=Rapid HTTP server lanes=1, B=Rapid HTTP server lanes=8. Five fresh-server pairs run A B A B, serving the same code and prose requests one at a time over real non-streaming TCP HTTP, greedy, 256 generated tokens each. Both returned HTTP texts and original generated token IDs match. Existing prompt-cache behavior is enabled; these are two distinct new histories per fresh server. Observational wrappers around Rapid's original backend constructor and provider iterator record metrics while preserving their calls and outputs; runtime source is untouched. The core-matrix and earlier SDK comparison loads are excluded from this paired HTTP table. Each entry is A/B; all lone rounds have one stream. Rows are actual RoundStats.rows, not maximum window width. Decode ms/token is `(last-output time - first-token time)/(tokens-1)`; it includes provider drainage after the last token. Positive B−A means the opt-in is slower. Each paired token sequence and returned HTTP text matches exactly. The raw client elapsed time is also retained; the decode metric isolates original provider token timing rather than response serialization.

| Pair | Prompt | Rounds A/B | Mean rows/round A/B | Decode ms/token A/B | B−A ms/token | B−A percent | Parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | code | 55/49 | 6.09/8.12 | 20.828/28.467 | +7.640 | +36.7% | pass |
| 1 | prose | 94/72 | 4.68/8.10 | 35.149/41.981 | +6.832 | +19.4% | pass |
| 2 | code | 49/49 | 8.12/8.12 | 28.111/28.658 | +0.547 | +1.9% | pass |
| 2 | prose | 72/72 | 8.10/8.10 | 41.188/26.648 | -14.539 | -35.3% | pass |
| 3 | code | 50/50 | 7.08/7.04 | 21.414/20.917 | -0.497 | -2.3% | pass |
| 3 | prose | 73/73 | 6.27/6.26 | 29.943/30.374 | +0.431 | +1.4% | pass |
| 4 | code | 49/50 | 8.12/7.26 | 20.797/21.018 | +0.221 | +1.1% | pass |
| 4 | prose | 72/72 | 8.10/6.53 | 27.258/28.094 | +0.836 | +3.1% | pass |
| 5 | code | 49/51 | 8.12/7.06 | 20.342/21.411 | +1.069 | +5.3% | pass |
| 5 | prose | 72/73 | 8.08/6.07 | 24.603/26.890 | +2.286 | +9.3% | pass |

Across the ten paired prompt observations, median B−A is +0.692 ms/token (+2.5%); mean +0.483 ms/token; spread -14.539–7.640 ms/token and -35.3–36.7%. The positive median is an observed lone-request cost of this opt-in in the campaign. The paired spread includes both faster and slower runs, so these observations do not isolate a causal configuration penalty or establish a universal no-regression guarantee. No kernel or calibration tuning was applied.

### Kernel and row calibration

| Pair | Exact width A/B | Engine (batch rows, max streams) A/B | Calibrated 1-row ms A/B | Calibrated exact-width ms A/B |
| --- | --- | --- | --- | --- |
| 1 | 128/128 | (32, 32)/(32, 32) | 30.397/78.017 | 686.970/652.550 |
| 2 | 128/128 | (32, 32)/(32, 32) | 43.834/45.546 | 639.829/641.755 |
| 3 | 128/128 | (32, 32)/(32, 32) | 53.704/52.774 | 597.113/594.378 |
| 4 | 128/128 | (32, 32)/(32, 32) | 50.952/43.651 | 604.374/596.134 |
| 5 | 128/128 | (32, 32)/(32, 32) | 52.536/53.751 | 598.201/584.287 |

The observed kernel path and hard row limits are identical between A and B. Exact width is the per-stream verified window limit; the engine shared-row budget is a separate limit (Qwen, for example, observes exact width 128 with shared batch rows 32). The runtime freshly times exact windows and shared rows at every load; those cost tables change across both configurations on this host, and adaptive decode row choices change as the lone-request table shows. The opt-in adds runtime concurrent-admission memory probes after family construction; it does not select a different family kernel configuration. This explanation combines observed calibration/rows with source inspection; a separate causal attribution of timing differences was not measured.

### Memory cost of enabling eight lanes

All values below are GiB except the last column (MiB). A/B order again. Idle resident is MLX active+cache bytes immediately after the loader reports ready. Idle peak is the MLX API value then: the runtime internally resets peaks while probing, so it is not a process-wide loading high-water mark. Each lone request resets the MLX peak first; request resident is active+cache at completion (cache bytes were recorded separately). Host RSS and whole-host pressure are not these metrics. Raw bytes remain in each JSON.

| Pair | Idle active A/B | Idle cache A/B | Idle resident A/B | Idle API peak A/B | Request resident range A/B | Request peak range A/B | Idle resident B−A MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.2826/17.8479 | 9.8988/0.5101 | 26.1814/18.3580 | 18.175/18.699 | 26.5467–26.6244/19.3805–19.5587 | 15.897–16.283/18.468–18.472 | -8011.15 |
| 2 | 16.2826/17.8479 | 9.8994/0.5100 | 26.1820/18.3580 | 18.175/18.699 | 26.5371–26.6035/19.3826–19.5592 | 15.899–16.283/18.468–18.472 | -8011.81 |
| 3 | 16.2826/17.8479 | 9.8997/0.5100 | 26.1823/18.3580 | 18.175/18.699 | 26.5598–26.6516/19.4449–19.6747 | 15.896–16.283/18.469–18.470 | -8012.06 |
| 4 | 16.2827/17.8479 | 9.8939/0.5099 | 26.1766/18.3578 | 18.175/18.699 | 26.5317–26.5981/19.4657–19.6465 | 15.899–16.283/18.468–18.470 | -8006.48 |
| 5 | 16.2826/17.8479 | 9.8989/0.5100 | 26.1815/18.3580 | 18.175/18.699 | 26.5366–26.5928/19.4496–19.6574 | 15.899–16.283/18.469–18.470 | -8011.32 |

### Real HTTP and shutdown

Verified-free port 18473; actual Rapid CLI, N=8. Three repeats compare every complete concurrent non-stream and SSE token array and returned text against the same solo HTTP request. Every concurrent mode observes shared rounds; cancel-one-stream verifies its prefix and all seven remaining full answers; the ninth simultaneous request gets existing 503 / Retry-After: 1 / at_capacity. Two continued conversations restore kept prompt state and match solo regeneration. HTTP references populate the existing prompt cache before regeneration, so these HTTP aggregate rates establish observed transport-path performance with warmed state, not a cold-run speed-up; the cold matrix above is the separate throughput workload. Non-streaming HTTP TTFT is not observable at the client and is marked accordingly; SSE TTFT measures first visible content, which can follow first generated token.

| HTTP case | Logged rounds per repeat | Mean streams/round | Peak streams/round | Elapsed seconds range | Aggregate tok/s | Visible TTFT ms | Parity/contract | Repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solo | 575,554,565 | 1.00–1.00 | 1 | 56.49–65.79 | 31.13–36.25 | not observed | pass | 3 |
| nonstream | 126,127,127 | 5.40–5.53 | 8 | 27.49–41.51 | 49.33–74.51 | not observed | pass | 3 |
| stream | 126,126,130 | 5.18–5.48 | 8 | 28.08–39.66 | 51.64–72.94 | 380.49–4383.28 | pass | 3 |
| cancel_and_overflow | 131,119,139 | 4.72–5.20 | 8 | 22.54–29.92 | 60.43–80.27 | not observed | pass | 3 |
| cache_continue | 92,105,96 | 1.15–1.49 | 2 | 15.76–23.23 | 22.04–32.49 | not observed | pass | 3 |

The shutdown probe returns HTTP 504 after a 0.75-second request deadline while its non-stream worker remains active; the final audit contains 48 generated tokens, proving mid-decode termination. SIGTERM to the owned wrapper yields a server shutdown in 0.477 seconds, verified port-free with no surviving owned process. Process IDs and response body are in the HTTP artifact. No unrelated process was signalled.

## Checks, diff and coverage limits

Local commits implement the opt-in loaders/capacity, request-local results, HTTP execution seam, cancellation registry, deferred stream cleanup, focused tests and this evidence doc. Affected files: rapid_mlx/speculative/{dflash/server.py,tensorfold_concurrency.py,tensorfold_families.py,tensorfold_qwen27.py,tensorfold_qwen27_server.py}, tests/test_tensorfold_lane_concurrency.py, tests/test_tensorfold_families_contract.py, tests/test_tensorfold_qwen27_backend.py, and this evidence document. Family adapter changes remain minimal around its existing ChatApp call. No runtime source, dependency pin, aliases.json, ordinary scheduler, MTP or native DFlash generation changed.

Final clean-source checks: 309 passed, 1 skipped in 19.75 seconds across the affected tests listed in reproduction below. The skip is the existing optional real-model DFlash Qwen3.5 end-to-end test. Ruff check and ruff format --check pass on all eight changed Python files. A dirty-source attempt failed the prefix probe's intended provenance guard (26 failed, 281 passed, 1 skipped); committing the code and rerunning resolves it. The pre-fix failures are retained as evidence, not omitted.

Not covered: sampled decoding, eight simultaneous long continuations, maximum contexts, forced admission/OOM, disk prefix persistence or restart restoration, other model families, DFlash model integration, exhaustive races under adversarial I/O, physical host/RSS memory. GLM-5.3-Flash and DeepSeek-V4-Flash were not loaded. Single-request measurements establish this workload's paired observations, not a hard bound on every host or prompt. No production ship decision, PR, push, merge or external messages. Lead engineer reviews and decides what ships.

## Exact commands and artifacts

Qualified read-only snapshots are Nemotron `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` d9d758fb83953437f7263256b0d96157e2a348b8; Qwen `Vontra/Qwen3.8-27B-MLX-4bit` 70ae7fac63274ff2eac54152031433374cb80f2f and drafter `z-lab/Qwen3.8-27B-DFlash2` 50307d4c4cde6860d4eee73e2547cd786fe8e8a4. These are the branch's previously qualified pins, not downloads or substitutions made by this pilot.

```sh
PILOT=/private/tmp/tensorfold-lane-concurrency-pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export RUFF_CACHE_DIR="$PILOT/ruff-cache"
# Executes five comparison-only A/B pairs per profile, plus pair0 for the core matrix, refreshed N=1/2/4/8 core cells (3 repeats),
# and both real HTTP harnesses (3 repeats plus timeout/SIGTERM).
"$PILOT/venv/bin/python" "$PILOT/run-final-measurements.py" > "$PILOT/final-measurements.log" 2>&1 &
PILOT_CONTROLLER_PID=$!
# While that controller is live, this comparison-only harness pauses only that
# owned controller at its current model-run boundary, runs five actual HTTP
# server A/B pairs per profile under the lock, then resumes the controller.
"$PILOT/venv/bin/python" "$PILOT/http_single_pairs.py" > "$PILOT/http-single-pairs.log" 2>&1
wait "$PILOT_CONTROLLER_PID"
# Each model-loading child uses exactly:
"$PILOT/venv/bin/python" scripts/large-model-run.py --working-set-gb 40 --lock-timeout 7200 -- \
  "$PILOT/venv/bin/python" "$PILOT/probe-final-code.py" --profile nemotron --lanes 1 \
  --paired --repeats 3 --output "$PILOT/final-nemotron-pair0-n1.json"
# Qwen uses working-set-gb=64; other pair0/N runs use --baseline ...pair0-n1.json.
# Pairs1..5 use --paired --repeats 0, alternating lanes=1 then8.
"$PILOT/venv/bin/python" "$PILOT/http-final.py" --profile nemotron --port 18472 --lanes 8
"$PILOT/venv/bin/python" "$PILOT/http-final.py" --profile qwen27 --port 18473 --lanes 8
# The HTTP harness wraps its own Rapid serve command with large-model-run.py,
# verifies the port is free, and terminates only its owned wrapper/process group.
"$PILOT/venv/bin/python" -m pytest tests/test_tensorfold_lane_concurrency.py \
  tests/test_dflash_integration.py tests/test_tensorfold_families_contract.py \
  tests/test_tensorfold_qwen27_backend.py tests/test_tensorfold_qwen27_http.py \
  tests/test_tensorfold_glm53_contract.py tests/test_benchmark_nemotron_prefix.py -q \
  -o cache_dir="$PILOT/pytest-cache" --basetemp="$PILOT/final-clean-tests-tmp"
PILOT_PY_FILES=(
  rapid_mlx/speculative/dflash/server.py rapid_mlx/speculative/tensorfold_concurrency.py
  rapid_mlx/speculative/tensorfold_families.py rapid_mlx/speculative/tensorfold_qwen27.py
  rapid_mlx/speculative/tensorfold_qwen27_server.py tests/test_tensorfold_lane_concurrency.py
  tests/test_tensorfold_families_contract.py tests/test_tensorfold_qwen27_backend.py
)
"$PILOT/venv/bin/python" -m ruff check "${PILOT_PY_FILES[@]}"
"$PILOT/venv/bin/python" -m ruff format --check "${PILOT_PY_FILES[@]}"
git diff --check
```

Every subprocess's exact argv is retained in `final-*.command.json` and the HTTP artifacts' command fields. `http_single_pairs.py` starts twenty real servers on individually verified-free ports 18480..18489 and 18500..18509 through the same lock, compares five A/B pairs per profile, and verifies port/process cleanup per server. Its `observe_http_single.py` records original provider token timing and backend MLX/row/calibration metrics without changing production or runtime source. `probe-final-code.py` calls final loaders directly (no constructor patches), records source/runtime provenance and its source SHA256, all generated IDs, seconds, rounds, actual rows, per-stream timing and memory. `http-final.py` retains actual TCP text, token audits and per-forward occupancy logs under final-http-{nemotron,qwen27}/. The report generator asserts every artifact's serving commit and paired token equality. The script files, raw artifacts, command logs, failing regressions and successful checks live exclusively in the pilot scratch directory. Earlier evidence remains under pre-stream-cleanup-measurements/ and the pre-review report; none contributes numbers to these tables.
