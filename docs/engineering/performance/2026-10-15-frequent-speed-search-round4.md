# Frequent-workload speed search: fourth screening round

No candidate qualified for a serving PR. The admission rule remains a measured
greater-than-10% improvement in a frequent user workload, with correctness
preserved. This round measured expert-layer components and streaming parser
CPU cost; it did not measure an end-to-end improvement. The previously rejected
completed-turn cache candidate was stopped and was not investigated further.

Owner: Vector. Baseline: `5c70e238f6068d373553c48d92786a5cf333d05b`.
Research branch: `vector/frequent-speed-search-round4`. No production changes.

## Expert projection and dispatch screen

Rapid already has merged gate/up expert projections in `rapid_mlx/moe_fusion.py`,
enrolled by the language-model loader. This is an existing mechanism, not a new
optimization. The question was whether extending its enrollment or explicitly
sorting small expert batches offered enough headroom to justify further work.

Hardware: one Apple M4 Pro, 48 GiB, macOS 26.5.1. Runtime: Python 3.12.13,
MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2, NumPy 2.4.4. The metadata check
verified 1,616 dependency Python files against official wheel identities.
No model was loaded or downloaded.

The experiment uses synthetic BF16 inputs and 4-bit affine expert weights,
group size 64, top eight experts. Both shapes execute the native language-model
`SwitchGLU`; the first uses the Gemma activation. These are operator geometries,
**not measurements of actual Gemma or Qwen models**:

- `gemma_moe_shape`: width 2,816, intermediate 2,112, 128 experts, GeGLU.
- `qwen_moe_shape`: width 2,048, intermediate 512, 256 experts, SwiGLU.

Four variants share immutable quantized arrays in separate module containers:
stock, existing projection packing, forced expert sorting, and both. Each cell
has ten timed samples of four evaluated forwards, alternating forward/reverse
variant order, after warm-up. All 32 variant/shape/row output checks were finite
and storage-bit equal to stock, with zero maximum absolute difference.

Median component latency reduction relative to stock; negative means slower:

| Shape | Rows | Packing | Sorting | Both |
| --- | ---: | ---: | ---: | ---: |
| gemma_moe_shape | 1 | +3.88% | -7.94% | -8.26% |
| gemma_moe_shape | 4 | -0.61% | -0.85% | +0.73% |
| gemma_moe_shape | 16 | -0.39% | -0.12% | -0.67% |
| gemma_moe_shape | 128 | +0.56% | -0.30% | +0.23% |
| qwen_moe_shape | 1 | +0.77% | -21.48% | -19.23% |
| qwen_moe_shape | 4 | +1.98% | -5.18% | -3.80% |
| qwen_moe_shape | 16 | -1.37% | -0.47% | -1.38% |
| qwen_moe_shape | 128 | +0.40% | -0.04% | +0.67% |

The initial multimodal-family enrollment preflight rejected the operator's
extended call signature; no performance samples were accepted from that attempt.
The current multimodal operator also handles weights, shared experts, residuals,
and its own short-block helpers. The screen did not bypass the adapter's signature
guard or establish compatibility for that runtime.

Disposition: stop this extension/sorting candidate. Component results do not
support investing in a greater-than-10% serving claim. They do not prove that
every possible expert kernel or hardware generation has the same result.

## Streaming tool parser CPU screen

A separate controller-only CPU probe fed eight-character chunks into four text
parser paths, twice per size. The payload is a file-writing call containing code.
Every accepted trial reconstructs exactly one call with the original function
name and JSON arguments. Chunk size is not a measured tokenizer distribution.
The CPU host was Apple M3 Ultra, Python 3.12; no inference or GPU benchmark ran
there. Full interpreter/platform and source identities are in the metadata.

Cumulative process CPU milliseconds per complete synthetic stream:

| Payload bytes | Qwen JSON | Hermes JSON | Gemma raw | Gemma stripped |
| ---: | ---: | ---: | ---: | ---: |
| 1027 | 0.53 | 4.03 | 6.12 | 5.86 |
| 4099 | 3.87 | 33.03 | 83.53 | 84.40 |
| 16387 | 50.36 | 430.91 | 1295.32 | 1300.29 |
| 32771 | 191.39 | 1602.65 | 5134.17 | 5134.94 |

Long bodies expose repeated work in text parsers, but these numbers are not
request latency or an achievable speedup. The probe includes constructing growing
text and timing each invocation. Real chunk distribution, GPU overlap and model
generation time were not measured. Gemma's allowlisted token-channel route emits
structured calls and bypasses text extraction in `StreamingPostProcessor`;
optimizing the raw parser cannot automatically accelerate that route.

Two Qwen coder harness preflights, using raw and JSON-quoted XML parameters,
emitted no collected tool calls and failed the reconstruction assertion. That
parser was excluded, not assigned a timing or a correctness pass. These attempts
do not establish a product bug; production finalization and wire contracts need
separate reproduction before drawing that conclusion. Failed harness receipts
are preserved in the private research archive.

Disposition: do not promote parser changes from this evidence. The screen has
neither demonstrated substantial common-request cost nor validated activation in
the serving route where the largest synthetic numbers occur.

## Additional source checks and stopping point

The text and multimodal schedulers already keep per-request incremental
detokenizers. A one-time defensive historical decode and terminal finalization
are distinct from re-decoding the full answer every token. No replacement was
prototyped from this source check. Active scheduler loops yield with `sleep(0)`;
the idle event timeout is not evidence of a per-token polling delay. Existing
hybrid throttling was retained because it has correctness responsibilities.

The bounded round is complete with no new qualifying candidate, serving PR or
queue entry. These negative screens are not an exhaustive impossibility claim.
Vector owns any subsequent search; a fresh hypothesis must show plausible
greater-than-10% frequent-workload headroom before another hardware campaign.
No worker is reserved for such future work. The M4 task processes and scratch
were removed, its lease released, and the borrowed runtime and standard model
cache preserved.

## Reproduction and evidence

The [fixture manifest](fixtures/2026-10-15-speed-search-round4/manifest.json)
binds nine members of the accompanying `evidence.tar.gz`: raw samples, hardware
and source metadata, cleanup receipt, executed harnesses, and dependency file
identities. Integrity hashes establish artifact identity, not independent proof
that hardware timing or cleanup occurred.

For the component probe, prepare this baseline at
`/private/tmp/rapid-mlx-speed-search-round4/src`, put the unpacked harness and
official dependency file manifest in its parent, and create `.task-owner` there
with the exact contents `vector-speed-search-round4-v1`. Use the stated runtime:

```sh
python moe_components.py
python metadata.py
```

For the CPU probe, put the baseline at the sibling `worktree` directory and run
`python3.12 parser_profile.py`. It needs Rapid's parser dependencies, but does
not load MLX models. The harness retains an unused coder wire fixture from the
failed preflight; only its four explicitly enumerated parser paths are measured.
Use a fresh authorized worker lease for any remote reproduction, serial timing,
and immediate cleanup/release. No benchmark or correctness claim is made for
other Apple generations, Qwen 27B, TensorFold, or model parameter sizes here.
