# Nemotron-H right-padded recurrent prefill

Owner: Vector. Branch: `vector/nemotron-h-padded-ssm-state`.

## Failure and affected path

The text model loader installs a compatibility shim for the shared Nemotron-H
SSM path. In the installed dependency version 0.31.3, the mixer reads
`ArraysCache.lengths` but omits it from `ssm_update`. The shared SSD recurrence
also advances incoming state through padded positions, preserves exhausted
rows only for negative remaining lengths, and caps its state-gather position
by the configured step instead of the current chunk size.

The batched engine's text scheduler uses `PromptProcessingBatch`, whose
`prompt` method right-pads unequal prompts, calls cache `prepare(lengths=...)`,
processes chunks, and finalizes the caches. This reaches the defective path
for both fresh prompts and restored recurrent prefix caches. Prefix snapshots
can consequently retain incorrect state; restoring a correct snapshot does
not protect a subsequent padded suffix from this defect. Unpadded prefill and
ordinary single-token decode do not need padding corrections.

## Correction and boundaries

`rapid_mlx.patches.nemotron_h_ssm` installs idempotently from the canonical text
loader, following the existing model compatibility patches. It replaces the
shared recurrence in its defining module so previously imported dispatcher
aliases also see the correction. It forwards Nemotron-H cache lengths, derives
a valid-token mask when only lengths are provided, masks incoming-state decay,
preserves rows with zero or negative remaining lengths, and bounds state
gathering by each chunk's actual size. Masked single-token prefill uses this
recurrence because the decode kernel has no padding interface.

The existing timestep preprocessing, transition dtype, state accumulation,
residual connection, model parameters, convolution implementation, cache
lifecycle, and unmasked Metal decode dispatcher remain authoritative. No
site-packages files are modified. Remove the shim once the dependency ships
both the shared recurrence corrections and the Nemotron-H length forwarding.
This is a text-backend correction; the separate multimodal SSM implementation
is outside this change.

## Regression contract

`tests/test_nemotron_h_ssm.py` is included in the Apple CI roster. A tiny
random-weight hybrid model (16 hidden channels, two SSM heads, 32 state
channels, one Mamba and one attention layer) compares valid prefill logits,
convolution and SSM states, and subsequent decode logits against independent
requests. It covers lengths 0, 255, 256, 257, 300 and 333; fresh and incoming
prefix states; prefill chunks of 1, 256 and 333; and the real prompt batch path.
An independent token recurrence additionally checks grouped heads, no explicit
mask, negative lengths, and lengths extending beyond a partial final chunk.
Left padding, unpadded chunking, loader installation, and ordinary decode
dispatch have controls. Zero-token rows retain incoming state exactly.

All eight model/prompt-batch cases fail against the unchanged dependency in
the CPU reproduction. The shim passes all 23 focused tests and the 63-test
compatibility/loader/CI-roster set on CPU, with 100% changed-line coverage
(82 executable lines). Environment: Python 3.12.14, MLX 0.32.4, dependency
0.31.3, fresh worktree `.venv`, installed with `pip install -e '.[dev]'`.
No production-model quality or performance claim is made. GPU numerical
validation remains outstanding because another job held the shared lock;
only this task's waiting lock client was cancelled. The CPU run checks Metal
decode dispatch with a kernel stub, rather than executing the Metal kernel.

CPU reproduction of the fixed contracts:

```sh
.venv/bin/python -c 'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["tests/test_nemotron_h_ssm.py", "-q"]))'
```

GPU execution must hold `lockf -k /var/tmp/rapid-mlx-large-model.lock`.
