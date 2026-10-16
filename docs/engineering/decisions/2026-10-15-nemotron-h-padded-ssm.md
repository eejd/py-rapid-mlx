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
the original CPU reproduction. The prior CPU compatibility set passed 63 tests.
GPU validation on Apple M3 Ultra (256 GB), Python 3.12.14, MLX 0.32.4 and model
dependency 0.31.3 passed all 23 focused tests with default settings and all 48
focused/Apple-CI-roster/compatibility tests with `MLX_ENABLE_TF32=0`. These runs
execute Metal decode in the numerical comparisons; only the explicit dispatcher
contract uses a kernel stub.

## Cached-checkpoint GPU evidence

Checkpoint: `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`, cached
snapshot `55ac8c89261109b36c04371cd3f479a4594208c8`. No download or checkpoint
modification was needed. `scripts/validate_nemotron_h_padding.py` loads through
the canonical loader and uses the production prompt-processing batch. Two
prompts have 5 and 308 tokens. Their prefixes (4 and 307 tokens) are processed
with a 256-token prefill step; the final prompt token exposes next-token logits,
followed by eight greedy tokens. Independent runs use the same chunk size.

The numerical control promotes floating parameters/activations to float32,
retains packed 4-bit weights, and disables TF32. Base: `origin/main` at
`ed77e0b33`; candidate implementation: `bc56335f2` plus the validation script.

| Revision | Prompt tokens | First next-token max absolute logit error | Worst error over eight steps | Eight greedy tokens match |
| --- | ---: | ---: | ---: | --- |
| Base | 5 | 7.983969 | 18.153801 | No |
| Candidate | 5 | 0.0000133514 | 0.0000181198 | Yes |
| Base | 308 | 0.00196362 | 0.00196362 | Yes |
| Candidate | 308 | 0.00196362 | 0.00196362 | Yes |

The untouched long row supplies a numerical control: its residual error is
identical on both revisions. The float32 comparison uses `atol=0.003,
rtol=0.0002` for every logit at every step; the base fails and the candidate
passes. The initial tighter `atol=0.0002` bound failed only on this unchanged
long control after the fix. The short-row correction is much tighter than the
final allowance, with peak error below 0.000019.

At original checkpoint precision, the candidate also matches both eight-token
greedy continuations, while the base diverges on the short row. Logits are not
bitwise equivalent: first-step max errors are 1.578125 for the candidate short
row and 0.8125 for the long row, with the long-row errors again identical on
both revisions. The checkpoint-precision diagnostic fails its logit bound on
both revisions. This evidence verifies recurrent padding correctness, not
bitwise batch invariance at checkpoint precision or a quality/performance claim.

Reproduce on a host with this cached snapshot (all GPU commands require the
shared lock):

```sh
mkdir -p /private/tmp/nemotron-ssm-fix-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
RAPID_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit/snapshots/55ac8c89261109b36c04371cd3f479a4594208c8"
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python -m pytest tests/test_nemotron_h_ssm.py tests/test_ci_apple_coverage_union.py tests/test_mlx_compat.py -q
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python scripts/validate_nemotron_h_padding.py --float32 --model "$RAPID_MODEL" --output /private/tmp/nemotron-ssm-fix-validation/branch.json
```

For the negative control, run the same candidate script and Python environment
with `PYTHONPATH` pointing at a detached base checkout. Base exits 1 after
writing its JSON evidence; candidate exits 0. The script reports raw errors,
allclose decisions, token IDs and decoded continuations before asserting.
Changed-line coverage against `origin/main` is 135/135 executable lines (100%),
including the compatibility shim, loader installation and validation script.
