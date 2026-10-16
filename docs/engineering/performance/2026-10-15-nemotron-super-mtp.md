# Nemotron 3.5 Super native MTP qualification

Vector, Studio, 2026-10-15. Text only. Core implementation: `9022e0b6a`.
Full-size benchmark and HTTP qualification: `3c12b585e`. Final compatibility
fix and Metal regression qualification: `ac1b99d76`. Final CLI default fix:
`3b1851cbf`. These fixes support ordinary caches created before injection and
clamp an implicit generic draft depth to the family K=1 preset. Native MTP
tensor math is unchanged.
MTP remains off by default. No artifacts have been uploaded by this task.

## Design

The existing shared MTP generator owns drafting, target verification, greedy
acceptance, sampled rejection, indexed accepted-prefix rollback, attention KV
trimming, and request termination. No separate speculative decoder was added.
The family injector supplies the target hidden-state forward, recursive draft
forward, and draft cache factory through that protocol.

The native head contains one logical module with attention and MoE sublayers.
It normalizes the next token's embedding and the target's final normalized
hidden state separately, concatenates embedding then hidden, and fuses them.
Attention and MoE residual blocks run before the head's final norm. The target
embedding and vocabulary projection are shared. K=2 recursively feeds the
head's output into that same module; there are no additional pretrained heads.

For verification, Mamba2 projections and convolution process the block once;
the existing single-row FP32 recurrence executes each position in sequence.
Each recurrent cache retains conv-window views and intermediate SSM states.
The shared generator selects the accepted boundary through `restore_rollback`
and trims attention by the rejected suffix length. Ordinary target calls retain
the original implementation. Extraction preserves the recurrent cache type;
singleton merges avoid unnecessary padding metadata. Verification refuses
padded recurrent lanes. The existing server admission keeps this MTP route
single-request and falls back before speculation for unsupported processors.
The generic cache splitter accepts an explicit target-cache count because
stateless MoE blocks have no cache entry; other families keep their existing
layer-count fallback.

## Additive sidecars and opt-in

Both target precisions use the same 4-bit/group-64 affine draft module.
Fusion, router, latent projections and norms retain checkpoint BF16 precision.
The vocabulary head is shared. Each sidecar adds
**1,718,400,868 bytes**. Target config and weight files were not changed.

Only these files are added inside each repository:

- `mtp/model.safetensors`
- `mtp/nemotron_mtp.json`

The injector validates the manifest, complete tensor names and shapes, and
packed/floating tensor dtypes before attaching the head. Missing or incompatible
sidecars fail at startup. Eligibility requires the reviewed text adapter marker
and an explicit sidecar; it does not enable other Nemotron architectures.
The alias declares its own repository as the sidecar source, K=1, and
`mtp_default_enabled=false`. K=2 requires an explicit depth. CLI depths are
bounded to 0–2; K=0 is a validation baseline.

The sidecar requires a Rapid-MLX build containing this implementation.
After uploading it, enable the existing alias preset with:

```sh
rapid-mlx serve nemotron-3.5-super-vl \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
rapid-mlx serve nemotron-3.5-super-vl-3bit \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Local validation supplies the corresponding sidecar directory as `model`
inside the speculative config. There is no download or upload requirement for
local use.

## Environment and reproduction

Apple M3 Ultra, 28 CPU / 60 GPU cores, 256 GiB unified memory, macOS 26.5.2;
Python 3.12.14, MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2.
A fresh Python 3.12 venv was installed with `pip install -e '.[dev]'`.
Machine Control lease owner: `rapid-mlx/vector-nemotron-mtp`. GPU work waits
for `lockf -k /var/tmp/rapid-mlx-large-model.lock`; existing jobs are not
preempted. Keep a lease heartbeat active for the complete run and release it
after owned inference processes have stopped.

The benchmark uses chat, Python code, arithmetic and copy/rewrite prompts.
It freezes each prompt at 1,024 or 8,192 tokens using repeated reference notes
before a chat turn, with thinking disabled. Conditions are ordinary decoding,
K=1, and K=2; two repetitions alternate their order. Each stops at EOS or 128
generated tokens. Throughput excludes time to first token and counts the
remaining generated tokens, including the same EOS convention in all lanes.
Prefill chunks are 256 tokens. Prompt IDs, generated IDs, timings and per-depth
counters are saved in JSON. Prompt lookup is disabled to isolate native MTP.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
# Claim the Studio lease with the configured Machine Control CLI, then:
python3 "$MACHINECTL" claim --machine studio-ultra \
  --owner rapid-mlx/vector-nemotron-mtp --purpose 'Native MTP validation' \
  --output /private/tmp/nemotron-super-mtp-validation/lease.json
python3 "$MACHINECTL" hold /private/tmp/nemotron-super-mtp-validation/lease.json
# Run these from another terminal while the heartbeat remains active:
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  scripts/benchmark_nemotron_mtp.py \
  --model /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-4bit-g64 \
  --sidecar /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-4bit \
  --output /private/tmp/nemotron-super-mtp-validation/4bit.json
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  scripts/benchmark_nemotron_mtp.py \
  --model /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-3bit-mixed \
  --sidecar /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-3bit \
  --output /private/tmp/nemotron-super-mtp-validation/3bit.json
```

The M5 Max FIFO queue confirmed **36 GiB** physical memory. The 3-bit target
alone previously measured 45.903 GB of MLX allocations before adding MTP.
Full-model 1K measurements cannot fit this worker; no target weights were
copied there. Tiny Metal regression tests passed: **12 passed**, Python 3.12.13, macOS
27.0.1, MLX 0.32.4 and mlx-lm 0.31.3, tested commit `3c12b585e`.
The FIFO run released its lease after the tests. The benchmark test required a
hermetic Git-provenance stub when running from a source archive; production
model code required no correction from these Metal checks.

## Validation and measurements

All **76** Metal model/converter tests pass on M3 Ultra. Final CPU qualification
passes **260 tests**: 13 focused MTP checks and 247 related portable tests.
The M5 Max tiny Metal suite passes all **12** MTP checks present at its tested
commit, including exact K=0/1/2 greedy parity and supplied-cache reuse.
Ruff check and format pass. Final changed-line coverage against
`origin/vector/nemotron-35-super-vl`: **275/275 lines, 100%**. The final CLI
change affects only depth selection and was checked by the CPU suite after
the 76-test Metal run.

Pooled throughput is total delivered tokens after the first token divided by
total decode seconds. Each cell pools four prompts with two interleaved runs.
Depth columns show accepted-prefix counts divided by rounds proposing that
depth. Depth 2 is prefix survival through both drafts. The proposal counters
include the terminal verifier work even when the output budget truncates a
round; throughput counts only delivered tokens.

| Quant | Context | Ordinary tok/s | K1 tok/s | K2 tok/s | K1 depth 1 | K2 depth 1 | K2 depth 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4bit | 1024 | 47.62 | 62.86 | 69.53 | 430/472 (91.1%) | 308/338 (91.1%) | 254/338 (75.1%) |
| 4bit | 8192 | 46.76 | 62.48 | 58.52 | 436/466 (93.6%) | 304/334 (91.0%) | 264/334 (79.0%) |
| 3bit | 1024 | 49.43 | 64.21 | 70.14 | 438/464 (94.4%) | 302/328 (92.1%) | 274/328 (83.5%) |
| 3bit | 8192 | 48.19 | 63.02 | 61.30 | 438/464 (94.4%) | 304/318 (95.6%) | 280/318 (88.1%) |

Per-prompt throughput (tok/s) and depth acceptance:

| Quant | Context | Prompt | Ordinary | K1 | K2 | K1 D1 | K2 D1 | K2 D2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4bit | 1K | chat | 48.01 | 59.63 | 59.89 | 78.9% | 80.4% | 46.4% |
| 4bit | 1K | code | 47.68 | 63.96 | 73.26 | 93.9% | 93.5% | 82.6% |
| 4bit | 1K | math | 47.41 | 64.52 | 74.76 | 96.9% | 97.7% | 93.2% |
| 4bit | 1K | rewrite | 47.18 | 64.18 | 75.10 | 100.0% | 100.0% | 95.7% |
| 4bit | 8K | chat | 46.84 | 60.29 | 54.24 | 88.2% | 84.3% | 64.7% |
| 4bit | 8K | code | 46.14 | 62.30 | 57.37 | 93.9% | 89.6% | 75.0% |
| 4bit | 8K | math | 47.09 | 63.66 | 61.82 | 95.4% | 95.6% | 91.1% |
| 4bit | 8K | rewrite | 47.16 | 64.98 | 63.97 | 100.0% | 100.0% | 95.7% |
| 3bit | 1K | chat | 49.53 | 63.14 | 69.32 | 91.0% | 87.5% | 79.2% |
| 3bit | 1K | code | 49.51 | 63.25 | 72.30 | 91.0% | 95.7% | 80.4% |
| 3bit | 1K | math | 49.29 | 65.45 | 66.01 | 98.4% | 89.4% | 85.1% |
| 3bit | 1K | rewrite | 49.35 | 65.84 | 76.50 | 100.0% | 100.0% | 95.7% |
| 3bit | 8K | chat | 48.47 | 60.86 | 57.46 | 88.2% | 87.5% | 77.1% |
| 3bit | 8K | code | 47.80 | 62.60 | 62.66 | 93.9% | 100.0% | 90.9% |
| 3bit | 8K | math | 48.17 | 64.49 | 62.78 | 98.4% | 97.7% | 93.2% |
| 3bit | 8K | rewrite | 48.46 | 65.37 | 63.88 | 100.0% | 100.0% | 95.7% |

The shared large-model lock was held throughout model execution. Other
unit-test processes were present during parts of the matrix; this is a
shared-host prototype qualification with interleaved conditions.

Greedy parity: **20/32** 4-bit speculative runs and **14/32** 3-bit runs
match their ordinary token sequence exactly. All remaining first differences
are between the serial target's top two candidates. Canonical single-row replay
measured gaps of **0–0.75 logits**, equivalent to **0–3 BF16 output ULPs**.
These are the documented numerical near-tie exceptions: multirow verification
can choose the other candidate, after which the continuation can diverge.
Tiny-model K=0/1/2 parity is exact; recurrent-state acceptance-boundary tests
pass independently. No byte-identical full-model claim is made for near ties.
Positions below are zero-based; duplicate repetitions are collapsed.

| Quant | Context | Prompt | K | First differing position | Serial logit gap | BF16 ULPs |
| --- | --- | --- | --- | --- | --- | --- |
| 4bit | 1024 | chat | 1 | 48 | 0.25 | 1 |
| 4bit | 1024 | chat | 2 | 18 | 0.50 | 1 |
| 4bit | 1024 | code | 2 | 51 | 0.00 | 0 |
| 4bit | 1024 | math | 1 | 114 | 0.00 | 0 |
| 4bit | 1024 | math | 2 | 114 | 0.00 | 0 |
| 4bit | 8192 | chat | 2 | 120 | 0.50 | 2 |
| 3bit | 1024 | chat | 1 | 49 | 0.00 | 0 |
| 3bit | 1024 | chat | 2 | 49 | 0.00 | 0 |
| 3bit | 1024 | code | 1 | 36 | 0.00 | 0 |
| 3bit | 1024 | code | 2 | 36 | 0.00 | 0 |
| 3bit | 1024 | math | 2 | 75 | 0.00 | 0 |
| 3bit | 8192 | code | 1 | 21 | 0.00 | 0 |
| 3bit | 8192 | code | 2 | 127 | 0.25 | 1 |
| 3bit | 8192 | math | 1 | 101 | 0.75 | 3 |
| 3bit | 8192 | math | 2 | 101 | 0.75 | 3 |

HTTP gates pass for **4-bit and 3-bit, each at K=1 and K=2**. Prefix caching
was enabled and complete-response caching disabled; each server recorded six
prefix-cache hits. For every configuration:

- Stream/non-stream chat passed with thinking on/off, returning 391 for 17×23.
- Automatic tool calls used MTP and returned structured `get_weather` calls
  with `city=Paris` and `finish_reason="tool_calls"` in both response modes.
- The existing forced-call and tool-result replay gate passed in both modes.
  Grammar-owned requests retain the shared runtime's ordinary-decode fallback.
- Disconnect after eight visible deltas passed; a subsequent request returned
  `CLEAN`, without the cancelled request's sentinel. Running/waiting gauges
  returned to zero.
- No native control markers appeared in visible content or reasoning.

The tool-enabled requests' MTP verify-call counts were 12/12 (4-bit K1), 8/8
(4-bit K2), 12/12 (3-bit K1), and 9/9 (3-bit K2), non-stream/stream respectively.
An explicitly selected streaming tool parser requires
`--enable-auto-tool-choice`; the qualification commands include it.

Matrix-process MLX allocator peaks, including the loaded sidecar and temporary
buffers, were **74.457 GB** for 4-bit and **47.681 GB** for 3-bit on this 256 GiB
host. These are process peaks across the matrix, not physical 64 GB validation.

**Recommendation: keep opt-in.** K=1 is the general preset. K=2 helps short
structured/copy workloads, while K=1 wins the 8K pooled comparison on both
targets. These four prompts and two repetitions establish a useful M3 Ultra
prototype. A fleet-wide default needs broader qualification. The additional
head, recurrent history, and draft KV cache require memory beyond the ordinary target; MTP
has not been qualified on a physical 64 GB Mac. Full-model M5 performance,
sampled performance, image input and contexts beyond 8K are unqualified.

Verification commands:

```sh
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python -m pytest \
  tests/test_nemotron_h_super.py tests/test_nemotron_super_vl_convert.py \
  tests/test_nemotron_h_puzzle_vendored.py tests/test_nemotron_super_metadata.py \
  -q --cov=rapid_mlx --cov=scripts.nemotron_super_vl_convert \
  --cov=scripts.benchmark_nemotron_mtp \
  --cov-report=xml:/private/tmp/nemotron-super-mtp-validation/gpu-coverage.xml
# Final CPU coverage includes the subsequent CLI-only fix:
.venv/bin/diff-cover /private/tmp/nemotron-super-mtp-validation/compat-final-coverage.xml \
  --compare-branch origin/vector/nemotron-35-super-vl --show-uncovered --fail-under 100
git diff --name-only -z origin/vector/nemotron-35-super-vl -- '*.py' | \
  xargs -0 .venv/bin/ruff check
git diff --name-only -z origin/vector/nemotron-35-super-vl -- '*.py' | \
  xargs -0 .venv/bin/ruff format --check
```

The complete HTTP harness is preserved with the measurement artifacts. Run
it from this worktree, under the same lease, for each precision/depth:

```sh
MTP_TEST_PRECISION=4bit MTP_TEST_DEPTH=1 \
  MTP_TEST_OUTPUT=/private/tmp/nemotron-super-mtp-validation/serve-repro \
  lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-validation-2026-10-15/serve_mtp.py
# Repeat with MTP_TEST_PRECISION=3bit and MTP_TEST_DEPTH=2 as required.
# After every owned model/test server has exited:
python3 "$MACHINECTL" release \
  /private/tmp/nemotron-super-mtp-validation/lease.json --cleanup-confirmed
```

All task leases have been released and all task test servers stopped.

## Extraction and upload

Source: cached BF16 snapshot `a1ccf94d69377bf82c4db39798aac28447d9c16f`.
Run `scripts/nemotron_super_vl_convert.py --source <snapshot> --output <new-dir>
--bits 4 --mtp-only` (or `--bits 3`). Extraction runs on CPU and refuses an
existing output directory. Staging is renamed only after successful export.
Standard shared-cache blob links are supported; arbitrary escaping shard paths
are refused.

Outputs:

- `/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-4bit/`
- `/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-3bit/`

Both copies have identical hashes:

| File | SHA-256 |
| --- | --- |
| `mtp/model.safetensors` | `57b0634c3f79b3216e8cfe007dc7479e6a48483d76626c3553f61457648e8141` |
| `mtp/nemotron_mtp.json` | `36eff2e72a3b08ca02101a66638d7e4d4674710e7b9afee26dc3ec7f84373900` |

Exact additive upload commands (not executed):

```sh
hf upload rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit \
  /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-4bit/mtp mtp \
  --commit-message 'Add opt-in native MTP sidecar'
hf upload rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-3bit \
  /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-sidecar-3bit/mtp mtp \
  --commit-message 'Add opt-in native MTP sidecar'
```

These upload only the new `mtp/` files, with no deletion or replacement of target
files. Keep MTP opt-in until end-to-end measurements justify a different default.

## Diff summary and durable evidence

- `nemotron_h_inject.py`: native head, strict additive manifest loading,
  Mamba2 conv/SSM history, preserved cache extraction/merge, and ordinary-cache
  compatibility after injection.
- Shared generator: optional target-cache count for backbones with stateless
  blocks. Existing families retain their previous fallback.
- Detector, dispatcher, scheduler, CLI and aliases: explicit native sidecar,
  K=1 preset including implicit CLI selection, explicit depths 0–2, default off.
- Existing converter: CPU `--mtp-only` export, safe shared-cache shard links,
  atomic new-directory publication. Existing target conversion is unchanged.
- Apple-listed tests and `benchmark_nemotron_mtp.py`: rollback boundaries,
  sidecar corruption, opt-in, cache lifecycle, greedy parity and reproducible
  measurements.

Raw results, diagnostics, HTTP traces, scripts, environment, source hashes and
an artifact checksum manifest are archived at:

`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/mtp-validation-2026-10-15/`

`provenance.json` binds source hashes to the measured commit. Lease files,
Machine Control tokens and private management configuration are excluded.
The branch was fetched and merged again against
`origin/vector/nemotron-35-super-vl` (`753280190`); it was already up to date.
No branch push, PR creation or HF upload was performed.
