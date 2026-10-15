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

At the local Phase A checkpoint, Ruff and four config/registration tests passed.
The model-forward tests and changed-line coverage gate remain unverified: the
required Studio lease was busy and no GPU run was started. The commands below
are the coordinator's next validation action, not a record of passing results.

The new contracts live in `tests/test_bailing_hybrid_model.py`, which already runs
in the Apple CI list. They cover heterogeneous config parsing, omni extraction,
sanitization, float32/BF16 four-layer forwards and cached decode, single-kind
schedules, norm/router math, quantization boundaries, SSM chunking, and model
registration. The full-size config contract allocates no model tensors.

Acquire a Machine Control lease for `studio-ultra` before running model tests.
Run the following from this worktree while holding the lease, then release it
only after test processes finish:

```sh
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/pytest \
  tests/test_bailing_hybrid_model.py \
  tests/test_gpt_oss_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
  --cov=rapid_mlx --cov-report=xml:/private/tmp/rapid-mlx-bolt/coverage.xml
.venv/bin/ruff check rapid_mlx/models/nemotron_h_puzzle.py \
  rapid_mlx/utils/tokenizer.py tests/test_bailing_hybrid_model.py
```

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
