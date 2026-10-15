# Nemotron 3.5 Super VL text support

Owner: Vector. Host: Studio, M3 Ultra, 256 GiB unified memory.
Branch: `vector/nemotron-35-super-vl`, stacked on the Bolt text support branch.
Local conversion and validation only; no push, PR, upload or release execution.

## Conversion and runtime boundary

The source is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision
`a1ccf94d69377bf82c4db39798aac28447d9c16f`. Its 88-file snapshot remains
incomplete in the standard Hugging Face cache. The original downloader exited
with a consistency-check failure on shard 9 (expected 3,980,237,344 bytes,
received 3,810,009,575); 29 files, including 16 complete shards, are present.
The coordinator must restore the download; this task has not restarted it or
modified its cached files. The
checkpoint index declares 248,906,833,960 tensor bytes. No second weight
download is required.

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
`lockf -k /var/tmp/rapid-mlx-large-model.lock`. Wait for all 88 source files;
the converter additionally rejects any missing indexed weight shard.

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
The converter writes the model card and counts every output file, including
README, excluding `.cache` directories.

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

Initial platform-neutral metadata, alias-schema and CI contracts: 3,762 passed.
Model contracts are queued on the shared lock behind an unrelated validation
process; no model-test result is claimed yet. Full source download, conversion,
parity, serving transcripts, performance, artifact catalog sizes, final changed-line
coverage and final Bolt-base rebase are pending. This is not yet a release gate.

Private operational evidence is under `/private/tmp/nemotron-super-vl-evidence/`.
Role start FYIs were attempted through Orca; Atlas, Pixel and Echo named
mailboxes were unavailable. Harbor's live mailbox received the start FYI.
Atlas/coordinator owns release authorization and the human owns publication.
