# Nemotron 3.5 Super VL text support

Owner: Vector. Host: Studio, M3 Ultra, 256 GiB unified memory.
Branch: `vector/nemotron-35-super-vl`, stacked on the Bolt text support branch.
Local conversion and validation only; no push, PR, upload or release execution.

## Conversion and runtime boundary

The source is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision
`a1ccf94d69377bf82c4db39798aac28447d9c16f`. The coordinator restored the
original download. On resume, all **88 files** and **64 indexed weight shards**
are present in the standard Hugging Face cache. Every safetensors shard's
header-declared final tensor extent matches its physical file size. The
snapshot totals 248,937,280,067 bytes; the checkpoint index declares
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

Stacked on completed Bolt head `b652c2265`. Before this resume, metadata,
alias schema, size-manifest, CI and preflight contracts passed **3,924 tests**.
On resume the three Super configuration-only contracts pass. Ruff check,
format check and diff checks pass. Source completion is now independently
verified as described above; no model download was initiated by this task.

The first conversion, tensor/conversion regression suite and BF16 paging
self-check are queued under the shared lock. The lock holder is the unrelated
PR #4599 validator (`lockf` PID 56926), not a process owned by this task.
The human's exclusive machine allocation has been relayed to Harbor and the
coordinator. No other process has been stopped or modified.

Additional metadata checks exposed two inherited Bolt catalog-test failures:
`tests/test_nemotron_bolt_metadata.py` expects 43,274,347,300 / 31,670,622,410
bytes, while the base branch's generator-aligned manifest records
43,274,333,036 / 31,670,608,078 bytes. The coordinator owns correcting these
expectations in the Bolt branch. The other 40 metadata checks pass.

No full-checkpoint generation has run yet. Conversion, measured artifact and
catalog sizes, eight-prompt MLX BF16 parity, serving, 1K throughput and the
100% changed-line coverage gate remain pending. This is not release approval.

Run focused tensor and conversion gates:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/pytest \
  tests/test_nemotron_h_super.py \
  tests/test_nemotron_h_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
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
materializing a roughly 241 GB contiguous model. Run reference prompts under
separate lock acquisitions to keep individual windows bounded. Serving and
1K performance scripts are prepared in the private evidence directory.

Harbor/coordinator next action: provide the already-authorized exclusive
model-lock window and correct the inherited Bolt size-test expectations.
Vector owns tensor-contract fixes, conversion, parity, stream/non-stream
serving, throughput, local catalog sizing and the final coverage gate. Vision
remains a feasibility-only report in this task; Vector owns its separate
image-input follow-up. The human retains upload and release authorization.

Private operational evidence is under `/private/tmp/nemotron-super-vl-evidence/`.
Role start FYIs were attempted through Orca; Atlas, Pixel and Echo named
mailboxes were unavailable. Harbor's live mailbox received the start FYI.
Atlas/coordinator owns release authorization and the human owns publication.
