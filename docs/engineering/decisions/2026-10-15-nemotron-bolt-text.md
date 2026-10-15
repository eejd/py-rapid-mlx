# Nemotron 3.5 Bolt text backbone

Owner: Vector. Phase A adds the local text architecture and unit contracts.
The coordinator owns conversion, real-checkpoint validation, publication, and
model catalog entries. No model weights are required for the unit contracts.

## Config and checkpoint boundary

Converted text artifacts must have `model_type: nemotron_h_puzzle` at the root.
`rapid_mlx.models.nemotron_h_puzzle.text_config(config)` extracts the original
omni checkpoint's `llm_config` and removes its remote-code `auto_map`. Write that
returned dictionary as the conversion output's `config.json`, preserving the
checkpoint's tokenizer and chat template files separately.

This conversion boundary keeps text artifacts on the existing text-loader path
without adding an omni routing exception or advertising vision support. Direct
serving of the original `nemotron_h_omni` artifact is intentionally unsupported.
Architecture registration alone does not make the standalone conversion CLI
import Rapid's modules: conversion must explicitly call
`rapid_mlx.utils.tokenizer._register_vendored_archs()` before model construction.

The sanitizer accepts `language_model.backbone.*` and `language_model.lm_head.*`,
removes the language prefix, drops `vision_model.*`, `mlp1.*`, and MTP tensors,
and also accepts the standalone `model.*` backbone naming. It transposes original
convolution weights and stacks individual experts into the local switch layout.
Already sanitized text tensors pass through unchanged.

Each MoE block supplies its own intermediate size, expert count, and top-k.
The root expert count is only a fallback for older uniform checkpoints. The
language configuration's encoded infinite time-step bound is normalized before
execution. RMS normalization and router logits use FP32; the SSM's softplus
retains the projected activation dtype before FP32 state-space computation.
Attention, convolution, and MoE forwards reuse the installed dependency's blocks.
The local SSM adapter preserves those dtype boundaries without patching global
runtime functions. Masked or length-aware decode uses the attention-form SSM path.

The quantization predicate preserves `lm_head`, routers, and both latent
projections. Four-bit and mixed three-bit quality and size are not yet verified.
The checkpoint template uses the existing XML tool-call grammar and
`enable_thinking` prefixes; template compatibility is a source inspection finding,
not evidence of real-model generation quality.

## Verification and next phase

Phase A was validated on studio-max (Apple M5 Max, 36 GB, macOS 27.0.1)
with task-local CPython 3.12.15, MLX 0.32.4, mlx-lm 0.31.3, pytest 9.1.1,
coverage 7.16.2, and diff-cover 8.0.3. No model weights were downloaded.
The focused suite passed all 68 tests (13 Bailing, 13 Bolt, 5 registration/loader,
20 parser, and 17 CI coverage-roster contracts). Ruff check and format check
passed. Changed-line coverage against `origin/main` at
`6fb35fbb24c5b8bb151d4fe2a7d7858adfc6d146` was 100%: all 297 changed
executable production lines covered, zero missing.

The Bolt contracts now live in `tests/test_nemotron_h_puzzle_vendored.py`,
explicitly listed in the Apple CI lane. A separate architecture file follows
the established vendored-model test pattern and avoids coupling Bolt's config,
registration state, and numerical contracts to the unrelated Bailing backbone.
They cover heterogeneous config parsing, omni extraction, sanitization,
float32/BF16 four-layer forwards and cached decode, single-kind schedules,
norm/router math, quantization boundaries, SSM chunking and length-aware state,
and model registration. The full-size config contract allocates no model tensors.
Registration state is restored after each test; random weights use seed 0.

The initial M5 default-precision run had three numerical failures across the
focused suite: Bailing full-prefill/cache parity, Bolt float32 full-prefill/cache
parity, and Bolt SSM full/chunk parity. The full-float32 rerun passed.

The Bolt float32 comparisons now account for TF32's 10 fraction bits without
requiring a process environment setting. They bound both RMS and peak error by
`8 * 2**-10` times the corresponding reference scale. The eight-rounding-unit
budget accommodates composed matmuls selecting GEMM versus GEMV kernels; scaling
by tensor RMS/peak avoids unstable elementwise relative errors at cancellation
zeros. Exact expert-index and per-layer weight-shape assertions remain strict,
as does router score normalization/scaling. Negative controls inject a 1% output
head scaling error, omit router scaling, and remove the SSM residual. All must
be detected. BF16 tolerances are unchanged.

Installed MLX 0.32.4 caches `MLX_ENABLE_TF32` in a function-local C++ static in
`mlx/include/mlx/utils.h`; the Python core and Metal API stubs expose no precision
switch. The existing `test_muse_glimmer_model.py` precedent selects full fp32 in
a fresh subprocess to compare cache semantics at equal precision. These Bolt
contracts instead retain the caller's precision policy to cover the default GPU
path, using the bounded comparisons and negative controls above.

The test portability follow-up was validated on mzr-001 (Apple M4 Pro, 48 GiB,
macOS 26.5.1, CPython 3.12.12, MLX 0.32.4, mlx-lm 0.31.3) after the studio-max
FIFO queue timed out while that host remained busy. Both the unset default
policy and `MLX_ENABLE_TF32=0` passed all 68 focused tests, including the negative
controls. Each run covered all 297 changed production lines against the Phase A
base and all 23 executable test lines changed by this follow-up against
`921021257`. Ruff and diff checks passed. The task directory was deleted and
the Machine Control lease released. These follow-up runs establish M4 behavior;
M5 execution of the revised comparisons remains unverified.

Acquire a Machine Control lease before running model tests. Copy the worktree
to a task-local directory under `/private/tmp/rapid-mlx-bolt-tests/`, create a
Python 3.12 venv there, and install `-e '.[dev]'` plus `diff-cover==8.0.3`.
Run the following inside that directory while holding the lease:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest \
  tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py \
  tests/test_gpt_oss_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
  tests/test_ci_apple_coverage_union.py \
  --cov=rapid_mlx --cov-report=xml:coverage.xml -q
.venv/bin/ruff check rapid_mlx/models/nemotron_h_puzzle.py \
  rapid_mlx/utils/tokenizer.py tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py
.venv/bin/ruff format --check rapid_mlx/models/nemotron_h_puzzle.py \
  rapid_mlx/utils/tokenizer.py tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py
```

Repeat the pytest command with `MLX_ENABLE_TF32=0` for full-float32 parity.

Copy `coverage.xml` back to the source worktree to compare its actual Git base
without copying Git internals to the test machine. Mirror CI's diff-cover gate:

```sh
.venv/bin/python -m diff_cover.diff_cover_tool /path/to/coverage.xml \
  --compare-branch origin/main --show-uncovered --fail-under 100
```

CI normally combines Linux and Apple coverage first; this focused Apple run
alone covers every changed executable production line. Delete the remote task
directory after processes finish, release the lease even after failure, and
verify release with `list --json` and the lease record. A queued owner may
claim the machine immediately after release.

Pending after storage capacity and machine availability are resolved:

- Download and convert the original BF16 checkpoint in the standard shared cache.
- Prove four-bit/group-64 loading and evaluate a mixed three-bit 32 GB variant;
  record actual artifact sizes and the precise conversion recipe.
- Compare reference logits/tokens, check multi-turn chat and thinking on/off,
  and exercise streaming and non-streaming tool calls through the server.
- Measure prefill and decode on the converted artifact with exact commands.
- Add aliases and model sizes only after a local artifact works and the
  coordinator supplies the publication repository ID.

Phase A makes no real-checkpoint parity, quality, memory, or throughput claim.
