# Same-machine oMLX versus Rapid TensorFold Qwen 27B TTFT

Owner: Vector. Date: 2026-10-14 UTC. User requirement: exactly the same target
weights on the same machine as the preceding QuickSilver TensorFold test.
Scope is measurement; no inference implementation or production settings changed.

## Result

Cold prefill is nearly identical. oMLX's enabled prefix cache is substantially
faster on repeated input and follow-up turns than the tested Rapid 0.16.1
TensorFold integration. Short uncached input is already fast in both engines.

All values below are **client-observed first nonempty content**, in seconds,
over loopback HTTP. Each cell is the median of three measured requests after
discarding model-load and kernel warmups.

| Workload | oMLX, cache disabled | oMLX, tiered cache enabled | Rapid + TensorFold, disk cache disabled |
| --- | ---: | ---: | ---: |
| Short input, 28 tokens | 0.448 | 0.676 | 0.346 |
| First long input, 755 tokens | 7.303 | 7.245 | 7.415 |
| Identical long input repeated | 7.300 | **0.362** | 7.298 |
| Follow-up after long input, 779 tokens | 7.596 | **0.636** | 7.603 |

oMLX reports 748 cached input tokens for both repeats and follow-ups. Its first
long request reuses only the 21-token opening previously primed by the short
request. The no-cache column has zero cached tokens throughout and is the
strict cold-prefill control. The enabled-cache setting writes SSD snapshots;
it is an additional product configuration, not an equal no-disk-cache control.

The repeat result is approximately 20 times faster than the tested Rapid
TensorFold path; the follow-up is approximately 12 times faster. These ratios
apply to this synthetic cached workload, not cold requests or overall throughput.

## Environment and version binding

- Same borrowed M4 Pro Mac as the prior test: 12 CPU cores, 16 GPU cores,
  48 GiB unified memory, macOS 26.5.1. It had no resident serving process.
  Each configuration ran alone and was stopped before starting the next.
- **One physical target snapshot shared by both engines**:
  `Vontra/Qwen3.8-27B-MLX-4bit` at
  `70ae7fac63274ff2eac54152031433374cb80f2f`. oMLX's model-discovery symlink
  points to the same default-HF-cache snapshot that TensorFold loads.
  No substitute repository, quantization or re-quantized target was used.
- oMLX [official macOS 26/27 release package](https://github.com/jundot/omlx/releases/tag/v0.7.1.dev1):
  **0.7.1.dev1**, build `261009233110-macos26-27`, tag commit
  `c0b1056b41ebde9422af316cf5038a423eb8f24c`. This is a development release,
  not a stable-release claim. DMG SHA-256:
  `04fbff0b54bd8656a6684e3051879d7039cb5f4514777527614c54d6d00e0ca2`.
  Bundled MLX 0.32.3, MLX-LM `0.31.4.dev132+g94cdcae13`, Transformers 5.17.0.
  `native_kernel_status()` reports all five shipped kernel modules available,
  including `qwen35_prefill`; server logs confirm Qwen patches installed.
- Rapid signed Desktop sidecar **0.16.1**, code commit
  `df1f4db1fc0548ec97c6376739a76f6330e9b000`, with a separate task-owned
  extension directory containing qualified TensorFold 0.6.6 at
  `cb2ebf0540f42604e2759b2ddef497861e928248`, MLX/MLX Metal 0.32.3.
  Target alias `qwen3.8-27b-tensorfold`; drafter
  `z-lab/Qwen3.8-27B-DFlash2` at
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, runtime-packed to 4 bits.
- Both app bundles passed deep/strict codesign verification. Extensions were
  external to the Rapid bundle; this does not certify stock Desktop packaging.
- oMLX uses its ordinary text LLM engine with `model_type_override: llm`,
  not the VLM path. Rapid uses its qualified TensorFold/DFlash2 adapter.
  Engine implementations, Python and MLX-LM versions differ as shipped;
  the target weights, device, prompts and MLX 0.32.3 match.

## Method and reproduction

The executable harness is
[`tools/benchmark_omlx_tensorfold_ttft.py`](../../../tools/benchmark_omlx_tensorfold_ttft.py).
The [36 individual samples and environment metadata](fixtures/2026-10-14-omlx-tensorfold-ttft.json)
include usage, TTFT, elapsed time, finish reason and answer checks.

Prepare a task scratch directory containing `oMLX.app`, `Rapid-MLX Desktop.app`
and `tf-extensions/` with the pinned optional runtime above. Keep target and
drafter snapshots in the standard HF cache; the script never downloads models.
Install the optional runtime with `pip --target` into `tf-extensions` rather
than editing either signed bundle. Then run on the measurement host:

```sh
RAPID_TTFT_BENCH_ROOT=/private/tmp/rapid-mlx-omlx-ttft \
  python3 tools/benchmark_omlx_tensorfold_ttft.py
```

The harness starts each engine on `127.0.0.1:18761`, with one concurrent
request, offline HF resolution and thinking disabled in the request template.
It uses a generated private loopback bearer for Rapid and binds oMLX only to
loopback. oMLX state/model links/cache live beneath the task scratch directory,
with HF auto-discovery disabled. It measures these configurations in order:

1. oMLX `--no-cache`.
2. oMLX with a dedicated `--paged-ssd-cache-dir` and a 4 GB limit;
   the hot-cache setting is left at its shipped default.
3. Rapid TensorFold `--disable-disk-caches --no-thinking --max-num-seqs 1`.

Each engine receives a discarded short warmup and a discarded longer-input
warmup. Three waves then send: a short opening; a new longer-input turn with
that opening; an identical repeat; and a follow-up including the answer.
The wave number changes near the start of the first user message, preventing
reuse of a previous wave's long prefix. The long reference is 140 repetitions
of `alpha beta gamma delta. `, followed by a request to answer only `4`.
All calls use temperature zero, cap generation at 32 tokens, request streaming
usage and require content `4`, DONE, and usage. Neither role-only SSE events
nor keepalives count as first content. All **36/36 measured calls passed**.

Both engines report the same 28 / 755 / 779 input tokens. Rapid reports two
completion tokens and oMLX one; generation accounting is not used to derive
these TTFT comparisons. Model startup is excluded. These are three samples
per cell, not p95 or a production latency SLO. No QuickSilver requests or
production credentials were needed for this comparison.

## Implication for QuickSilver

The previous approximately 8-second public-API result for roughly 750 native
input tokens cannot be judged against a cached local oMLX response. The fair
cold local comparison is approximately 7.3–7.4 seconds in both engines.
Network/routing adds to that, but does not explain most of the prefill time.

The actionable difference is prefix reuse: the tested Rapid TensorFold path
did not show reduced TTFT on an identical input or follow-up, while oMLX
explicitly restored 748 tokens. This establishes a measurement target for
Vector to diagnose in the Rapid/TensorFold boundary and runtime cache flow;
it does not identify the defective function or prove upstream TensorFold
cannot reuse prefixes. Preserve QuickSilver's no-prompt-snapshots policy:
restoring effective in-memory reuse should be investigated before adopting
the SSD persistence enabled in the oMLX cached test.

Both engines were stopped; task-owned model copies, SSD prompt snapshots,
app bundles and optional runtime were removed, and the worker lease released.
