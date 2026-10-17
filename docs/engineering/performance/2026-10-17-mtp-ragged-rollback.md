# Continuous self-MTP rollback constant factors (#4589)

Date: 2026-10-17. Owner: Vector.

## Result and scope

Reuse transaction-local validation plans across target/draft rollback, read each
KV offset vector once, and select uniform recurrent snapshots by reference.
For ragged KV rollback, share one full-capacity gather index across keys and
values, preserving masked padding and the reserved tail exactly.

Retain the original row-slice/concatenate algorithm for ragged recurrent
snapshots: every full-state vectorized candidate was slower on the representative
30-layer workload. Snapshot preflight now checks metadata without constructing
device selections. Both cache groups finish validation before either is applied;
standalone `trim_ragged_cache(validate=True)` retains that atomic validation.
Custom adapter hooks retain their existing call contract.

The rollback microbenchmarks below are model-free; end-to-end serving results
are reported separately.
Routing, admission, draft depth, defaults, dependency bounds, and cache layout
are outside this change.

## Environment and workload

- Apple M5 Max, 36 GiB unified memory; macOS 27.0.1, arm64.
- Python 3.12.13; MLX 0.32.4; mlx-lm 0.31.3.
- Baseline: `76b8296cd1cf3a009de045d719dd4823974afae2`, preserved as a
  separate scratch module; both arms run in fresh processes.
- Measured product source SHA-256 (before the validation-only compatibility fix):
  `b0fb2986c9d68a1ca4eafef1f93332b2145b2262557fe7f27ddd42b921cbaef7`.
- Cached Qwen3.6-35B-A3B-4bit config snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`; no weights loaded or downloaded.
- B=8, verify width=3, 40 target layers: 30 recurrent layers with slots
  `[8, 3, 8192]` BF16 and `[8, 32, 128, 128]` FP32; 10 KV layers with
  `[8, 2, capacity, 256]` BF16 keys and values. Capacity rounds context+3 up
  to a multiple of 256; initial left padding is `[0,1,2,3,0,1,2,3]`.
- Uniform drops `[1]*8`; ragged drops `[0,1,2,0,1,2,0,1]`. Draft drops are
  zero, so the draft cache is skipped as in the existing adapter contract.
- Independent contiguous, evaluated buffers per layer/boundary. Cache setup
  occurs outside timing. Each sample includes `adapter.rollback`, evaluation
  of resulting cache states, and `mx.synchronize()`; 5 warmups, median of 30.

## Full rollback medians (ms)

| Context | Uniform before | Uniform after | Ragged before | Ragged after |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4.383 | 0.353 | 6.252 | 4.099 |
| 4,096 | 4.375 | 0.345 | 19.809 | 9.024 |
| 16,384 | 4.345 | 0.347 | 70.532 | 25.977 |
| 32,768 | 4.366 | 0.358 | 136.138 | 48.348 |

## End-to-end serving (M5 Max)

Measured on the same M5 Max (36 GiB) with Qwen3.6-35B-A3B-4bit and its
4-bit MTP model. Two boots per arm ran in order plain → cohort-main →
cohort-patched → cohort-patched → cohort-main → plain. Each boot used a
single-request calibration, then one warm and three measured rounds per point.
Values are mean aggregate completion tok/s including prefill and HTTP;
ratios compare matching boot numbers. Short prompts contain 411–416 templated
tokens; every request returns 300 completion tokens with temperature=0,
stream=false, enable_thinking=false and chat_template_kwargs.enable_thinking=false.

| c | Boot | Plain tok/s | Cohort-main tok/s | Cohort-patched tok/s | Patched÷main |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 93.3 | 150.7 | 150.8 | 1.001× |
| 1 | 2 | 93.3 | 150.7 | 150.7 | 1.000× |
| 2 | 1 | 185.1 | 182.3 | 187.6 | 1.029× |
| 2 | 2 | 184.8 | 182.2 | 187.5 | 1.029× |
| 4 | 1 | 265.7 | 220.2 | 228.2 | 1.037× |
| 4 | 2 | 265.4 | 220.3 | 228.3 | 1.036× |
| 8 | 1 | 328.2 | 254.9 | 264.1 | 1.036× |
| 8 | 2 | 328.8 | 256.1 | 270.0 | 1.055× |

At c=1 the cohort arm cannot form a multi-lane cohort. All 48 short c=2/4/8
rounds admitted depth-2 cohorts; c=8 M2 admitted seven initial lanes in all
four cohort boots, versus eight in the other c=8 rounds. Patched serving gains
at c=2/4/8 were 2.9%/3.6–3.7%/3.6–5.5% over main; c=4/8 remained below plain.

Required parity comparisons were **64/64 identical**, with no divergence in
full messages, completion-token usage, or completion text re-encoded with the
target tokenizer. The API does not expose internal generated token IDs.
All non-timing speculative counters were identical across all four cohort
boots: 21,564 attempts, 13,426 accepts, acceptance ratio 0.6226.

The long workload (8,063–8,066 templated prompt tokens, c=2/4) formed no
continuous cohort in any of its 32 warm/measured rounds on 36 GiB. The existing
prefill memory gate estimated approximately 130 GB against a 27.1 GB cap.
Its throughput therefore does not measure continuous-cohort rollback gains.
**M3 Ultra long-context serving remains to be measured.**

Exact serve commands below preserve the measured flags and environment, with
checkout, executable and model paths generalized. `<snapshot path>` is the
target snapshot; `<MTP snapshot path>` is the draft snapshot. Run each command
in its indicated checkout with the isolated venv's `rapid-mlx` on PATH.
Boot 2 repeats the patched, main and plain commands in that order, changing
only their ports to 18913, 18914 and 18915 respectively.

```bash
# plain, patched checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host 127.0.0.1 --port 18910 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 --no-spec-decode

# cohort-main, baseline checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host 127.0.0.1 --port 18911 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 \
  --speculative-config '{"method":"mtp","model":"<MTP snapshot path>","continuous_batching":true,"allow_dynamic_membership":true}' \
  --force-spec-decode

# cohort-patched, patched checkout (boot 1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  rapid-mlx serve '<snapshot path>' --host 127.0.0.1 --port 18912 \
  --served-model-name q36 --max-num-seqs 8 --max-tokens 4096 \
  --speculative-config '{"method":"mtp","model":"<MTP snapshot path>","continuous_batching":true,"allow_dynamic_membership":true}' \
  --force-spec-decode
```

Raw rounds, admission evidence, parity records and commands remain in
`/private/tmp/rapid-mlx-4589/dogfood/`, summarized in `report.md`. These serving
measurements used the product source hash recorded above, before the later
validation-only compatibility fix; serving was not rerun for that fix.

## Item decisions

For the default adapter, preflight calls fall from three to one per applied
cache; KV offset host reads fall from four to one. Recurrent snapshot selection
is absent from preflight, and uniform restores build no gather/concatenate.

One KV layer, ragged drops, synchronized 30-sample medians (ms):

| Context | Original roll | Full-index gather (retained) | Selected-row roll/scatter |
| ---: | ---: | ---: | ---: |
| 512 | 0.772 | 0.310 | 0.535 |
| 4,096 | 2.223 | 0.857 | 2.470 |
| 16,384 | 7.785 | 2.531 | 8.984 |
| 32,768 | 15.580 | 4.760 | 17.778 |

Snapshot selection across 30 independent recurrent layers, both slots,
alternating candidate order, synchronized 30-sample medians:

| Method | ms |
| --- | ---: |
| Original row slices + concatenate (retained) | 2.920 |
| Concatenate candidates + take | 10.496 |
| Stack candidates + take_along_axis | 13.225 |
| Mask/where | 7.900 |
| Compiled mask/where | 5.311 |

The compiled mask's isolated small-slot result was 0.399 ms versus 0.406 ms
for the original, while the large recurrent-state slot regressed from 2.761
to 5.154 ms. This does not justify adding a shape-dependent selection policy.
An initial full rollback candidate using compiled masks regressed the 512-token
ragged case; the final table above uses the retained original selection.

## Verification and reproduction

All real MLX work ran on the M5 Max. Final focused suite: **171 passed**;
final broad suite: **1,456 passed, 11 skipped**, 40,301 deselected (153.53 s).
Bit-exact comparisons cover random ragged/positive-ragged, uniform and zero KV
drops, left padding, reserved tails, differing K/V dimensions, BF16/FP32, and
recurrent boundary-list/legacy snapshots. Failure cases prove validation precedes
mutation; fake-cache instrumentation proves once-per-cache validation.

From the patched checkout, using the isolated scratch venv:

```bash
export HF_HUB_OFFLINE=1
PY=/private/tmp/rapid-mlx-4589/venv/bin/python
$PY -m pytest -q tests/test_mtp_ragged_cache.py \
  tests/test_continuous_self_mtp_mlx_backend.py tests/test_continuous_mtp_runtime.py \
  --cov=rapid_mlx.spec_decode.mtp.ragged_cache --cov-report=xml:coverage.xml
$PY -m pytest -q -k 'mtp or ragged or cohort or continuous'
diff-cover coverage.xml --compare-branch=origin/main \
  --include=rapid_mlx/spec_decode/mtp/ragged_cache.py --fail-under=100
```

After the validation-only compatibility fix, local fake-only verification of
`tests/test_mtp_ragged_cache.py` passes **58 tests**, including validation-only
leaves and rejection before mutation for standalone trim and adapter rollback.
MLX imports are disabled by the scratch `pure_pytest.py` runner; no GPU is used.
The real MLX results above precede this final fix.

```bash
COVERAGE_FILE=/private/tmp/rapid-mlx-4589/coverage-final-fixes.data \
  $PY -m coverage run --source=rapid_mlx.spec_decode.mtp.ragged_cache \
  /private/tmp/rapid-mlx-4589/pure_pytest.py -q tests/test_mtp_ragged_cache.py
```

Current diff coverage is **80/80 changed product lines (100%)**. Local `ruff check`,
`ruff format --check`, and `mypy --follow-imports=silent` pass on all three
changed Python files. The added MLX tests live in the existing Apple CI-listed
`tests/test_continuous_self_mtp_mlx_backend.py`.

Benchmark scripts and raw samples remain in `/private/tmp/rapid-mlx-4589/`
on the measurement host and the development host:

```bash
$PY /private/tmp/rapid-mlx-4589/bench_rollback.py --mode before --repeats 30
$PY /private/tmp/rapid-mlx-4589/bench_rollback.py --mode after --repeats 30
$PY /private/tmp/rapid-mlx-4589/bench_roll_candidates.py
$PY /private/tmp/rapid-mlx-4589/bench_snapshot_group.py
```

`max-final-before.jsonl` and `max-final-after.jsonl` contain the final full
rollback samples and source hashes; `max-candidates.jsonl` and
`max-snapshot-group.jsonl` contain candidate measurements. Runs use bounded,
foreground subprocesses (`run_final.py`); no concurrent test or benchmark load.

## O(verify-width) follow-up

The retained KV gather remains O(context). For post-uniform cursor `s`, row
residual `r`, and physical column `p`, it reads `(p-r) % s` when `p < s` and
`p` otherwise. Zero-residual rows and the reserved tail therefore retain every
bit. Starting at minimum left padding would change masked-padding bits and was
rejected under this equivalence contract.

Vector owns the layout follow-up: decouple per-row write/valid-end positions
from the shared right-aligned cursor, coordinating attention masks, append
positions, RoPE offsets, and merge/filter/extract contracts. Increasing left
padding alone leaves accepted history at incorrect physical positions. That
layout change is required for O(verify-width) rollback and is outside this task.
