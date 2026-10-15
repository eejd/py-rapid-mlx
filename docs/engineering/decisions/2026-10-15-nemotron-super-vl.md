# Nemotron 3.5 Super VL text support

Owner: Vector. Host: Studio, M3 Ultra (28 CPU / 60 GPU cores), 256 GiB
unified memory.
Branch: `vector/nemotron-35-super-vl`, stacked on the Bolt text support branch.
Local conversion and validation only; no push, PR, upload or release execution.

## Conversion and runtime boundary

The source is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision
`a1ccf94d69377bf82c4db39798aac28447d9c16f`. The coordinator restored the
original download. On resume, all **88 files** and **64 indexed weight shards**
are present in the standard Hugging Face cache. Every safetensors shard's
header-declared final tensor extent matches its physical file size.
All 88 cached files also match their Hugging Face blob content hashes
(SHA-256 for LFS blobs and Git blob SHA-1 otherwise); the complete hash pass
took 125.59 seconds. The snapshot totals 248,937,280,067 bytes; the index declares
248,906,833,960 tensor bytes. No second weight download was started by this task.

The text converter hoists `llm_config`, derives the 88-layer count, removes
remote-code selectors, and normalizes the encoded default infinite time-step
bound by omitting it. The artifact retains `model_type: nemotron_h`, the native
language tensor layout, tokenizer files and chat template. The opt-in
`rapid_mlx_nemotron_h: true` selects a local precision adapter through the
existing checked loader's architecture callback. Ordinary Nemotron artifacts
and explicit caller architecture callbacks retain their existing loading path.

The adapter reuses the existing Bolt sanitizer, native-dtype SSM softplus,
FP32 state-space math, FP32 router logits, latent MoE, attention and cache
implementations. Super adds its language normalization boundary: FP32
normalization, activation cast back to the input dtype, then weight scaling.
Its gated Mamba normalization also gates and reduces in FP32. The checkpoint's
omni source delegates language construction to the Transformers implementation;
the source's declared version is 5.12.1. This source inspection establishes the
intended fused-path boundaries, not independent full-checkpoint NVIDIA parity.

The uniform model uses 40 Mamba, 40 MoE and eight attention layers, 512 routed
experts with top-22 selection, expert width 2688 and latent width 1024. Routing
uses sigmoid scores, correction bias for expert selection, normalization of
original selected scores and scaling by five. Shared experts have width 5376.
Attention has 32 query heads and two KV heads, dimension 128, without rotary
position encoding. The declared position limit is 1,048,576; qualification is
limited to the measured workloads below.

Sanitization strips `language_model.`, drops the vision tower, `mlp1.*`, the
additional `vision_projector.*` final norm and `language_model.mtp.*`, transposes
original convolution weights and stacks routed experts. No existing Rapid
consumer supports this Nemotron-H MTP head; its configuration is also removed.
MTP speculative decoding remains disabled.

The converted metadata selects the existing `nemotron` XML tool parser and
`qwen3` implicit-thinking parser. The supplied reasoning parser uses `<think>`
and `</think>` and treats all generated text as content when the template
prefills `<think></think>` for `enable_thinking=false`. No new parser protocol
is introduced. Optional source-parser replay/force-nonempty behavior is not a
new serving API promise.

## Reproduction

Create the task-local Python 3.12 environment and install `-e '.[dev]'`.
All model operations, including conversion and model tests, use
`lockf -k /var/tmp/rapid-mlx-large-model.lock`. The complete 88-file source
snapshot has been verified; the converter additionally rejects any missing
indexed weight shard.

```sh
source_snapshot="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/snapshots/a1ccf94d69377bf82c4db39798aac28447d9c16f"
output_root=/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  scripts/nemotron_super_vl_convert.py --source "$source_snapshot" \
  --output "$output_root/text-4bit-g64" --bits 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  scripts/nemotron_super_vl_convert.py --source "$source_snapshot" \
  --output "$output_root/text-3bit-mixed" --bits 3
```

The four-bit recipe uses affine/group-64 eligible projections and eight-bit
embeddings. Mixed three-bit uses routed experts at three-bit/group-128, other
eligible projections at four-bit/group-64, and eight-bit embeddings. Both
preserve the output head, router and latent projections at source precision.
The converter writes the model card and reports every output file, including
README, excluding `.cache` directories. Catalog sizing separately uses
`rapid_mlx._download_gate._is_weight_file`, the same filename filter used by
`scripts/gen_model_sizes.py`; it excludes README, LICENSE and the Jinja template.
Model cards require Rapid-MLX PR #4679 and follow-up and recommend 96 GB+ memory;
any shorter-context 64 GB suggestion is explicitly unqualified.

Intended publication repositories, for human upload only:

- `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit`
- `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-3bit`

## Vision disposition

The installed omni implementation has compatible RADIO, pixel-shuffle and
projector concepts, but Super adds a trained final LayerNorm on RADIO features
before pixel shuffle when the language model has MTP. The two tensors are
`vision_projector.vision_final_layernorm.{weight,bias}`. The installed omni
implementation does not consume or apply this module. Language config also
requires the derived layer count and numeric time-step bound. A direct type
rename is insufficient; image support requires an adapted projector and
language precision path plus real-image validation. Text artifacts deliberately
contain no image tower or image processor assets. Full-load compatibility
and vision qualification remain pending.

## Evidence status

Stacked on Bolt head `7118aaf72`, including the current-chunk SSM gather bound
and generator-aligned catalog expectations. All 88 source files pass content
hashes. Conversion succeeds in 321.02 seconds; the text artifact loads and
completes eight greedy smoke prompts plus two 1K generator benchmarks.

Four-bit output:
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-4bit-g64`.
It contains 69,540,668,652 bytes. The generator-compatible catalog footprint is
69,540,654,086 bytes. Fourteen converted shards pass extent checks; runtime
weights total 69,523,298,304 bytes. The directory includes its OpenMDW 1.1
model card, Rapid-only requirement and memory guidance.

The compiled norm implementation uses the same free-function binding pattern
as the shared precision adapter. Directly decorating an instance method did
not bind `self` with the installed MLX API; the tiny tensor contracts caught
this before qualification. The corrected norms pass the complete 53-test
Super/shared-model/parser suite. The converter also removes unused null model
selectors and tokenizer processor selectors from saved text metadata.

The evaluation-only paged BF16 reference passes its tiny full/cached resident
comparison bit-exactly. Full-checkpoint BF16 parity and HTTP serving are still
pending the shared lock. Quantized smoke outputs include Paris, 391, Buenos
días, primary light colors red/green/blue, and Ada name recall. Direct generator
throughput is 466.12–467.59 prompt tokens/s and 48.39–48.56 decode tokens/s for
1,024 prompt / 64 generation tokens, with a 72.68 GB process peak.
[Reproduction, environment and raw evidence](../performance/2026-10-15-nemotron-super-vl-text.md).

The four-bit catalog alias `nemotron-3.5-super-vl`, exact filtered footprint,
README alias totals and model table pass **3,774 tests**. The 42 Bolt/general
metadata tests also pass. Ruff and diff checks pass. A 100% changed-line result
was obtained before the final tokenizer-asset cleanup; the fresh 95-test suite
and final coverage gate for the current code remain queued and are not claimed
complete. The three-bit artifact has not started and has no catalog entry.

The unrelated PR #4599 validator held the lock until approximately 19:50 UTC.
The task yielded its waiting positions to Bolt's short check, then converted
and sampled four-bit. A second unrelated validator, PR #4672 (`lockf` PID 57383),
acquired the lock around 20:00 UTC. The queued fresh regressions, BF16 parity
and serving gates are blocked behind it. Harbor and the coordinator received
high-priority FYIs; the exclusive window was also requested through the user
input channel. No foreign process was stopped or modified. This is not release
approval; full-model parity, serving and final coverage remain required.

Run focused tensor and conversion gates:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/pytest \
  tests/test_nemotron_h_super.py \
  tests/test_nemotron_h_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
  tests/test_nemotron_bolt_metadata.py tests/test_model_metadata.py \
  --basetemp=/private/tmp/nemotron-super-vl-tests \
  --cov=rapid_mlx --cov=scripts --cov-report=xml:coverage.xml -q
.venv/bin/python -m diff_cover.diff_cover_tool coverage.xml \
  --compare-branch origin/vector/nemotron-35-bolt --show-uncovered --fail-under 100
```

Use the documented conversion commands only after all 88 snapshot files exist.
Eight parity prompts cover factual recall, arithmetic, code, translation, light
colors, a sequence, a brief explanation and user-name recall. Compare first-token
top-1/top-10 and greedy token sequences against MLX BF16. The evaluation-only
BF16 paging implementation must first pass its tiny resident-model comparison;
it copies exact selected source experts into the native switch MLP to avoid
materializing a roughly 241 GB contiguous model. Reference prompts run together
inside the exclusive window; every invocation holds the shared lock. Serving and
1K performance scripts are prepared in the private evidence directory.

Harbor/coordinator next action: provide the already-authorized exclusive
model-lock window and defer another full validator until qualification ends.
The inherited Bolt catalog-test mismatch is resolved by the committed
dependency rebase. Vector owns the queued final regressions/coverage,
full-checkpoint BF16 parity, stream/non-stream serving, then mixed-three-bit
conversion with its corresponding gates, catalog size and README total update.
Vision remains feasibility-only; Vector owns a separate image-input follow-up.
The human retains upload and release authorization.

Private operational scratch is under `/private/tmp/nemotron-super-vl-evidence/`;
completed measurements and harnesses are archived under
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/evidence/`.
Role start FYIs were attempted through Orca; Atlas, Pixel and Echo named
mailboxes were unavailable. Harbor's live mailbox received the start FYI.
Atlas/coordinator owns release authorization and the human owns publication.
