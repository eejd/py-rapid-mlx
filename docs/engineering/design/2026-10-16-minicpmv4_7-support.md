# MiniCPM-V 4.7 compatibility on the pinned multimodal runtime

Owner: Vector. Dependency: `mlx-vlm==0.7.2`, unchanged.

The 1B dense and 35B-A3B expert models enter the existing native MLLM lane.
Aliases are `minicpm-v4.7-1b-8bit` and `minicpm-v4.7-35b-a3b-mxfp4`.
Both are experimental until real-checkpoint verification completes. MTP,
DSpark and expert lookahead scheduling are outside this change.

The architecture package is copied from upstream commit
`b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0`, including the vision dtype
and cached-feature corrections. The per-file inventory, upstream SHA256s,
local-byte digests and MIT notice live under
`rapid_mlx/models/mlx_vlm_vendored/models/minicpmv4_7/`. The original
Apache-2.0 notice on the canvas RoPE implementation is retained. The offline
provenance test reverses import redirects to reconstruct upstream bytes;
extracted compatibility regions have separate upstream hashes.

## Compatibility evidence

The dependency diff from the pinned tag to that commit was inspected for
`minicpmv4_6`, `qwen3_5`, `qwen3_5_moe`, `base.py`, `utils.py`, and
`prompt_utils.py`, including the processor and loader closure.

- The 4.6 config parser hardcodes its dense text class in 0.7.2. The copied
  `_config.py` includes upstream's `_parse_text_config` hook, so both initial
  parsing and the loader's nested-config reparse select the expert class.
- The pinned vision wrapper rejects the new vision type. `_vision.py` copies
  only the small upstream `VisionModel` wrapper, reusing pinned SigLIP layers.
- `_vision_compat.py` copies only the two upstream embedding methods. Vision
  pixels use the patch-convolution dtype rather than the text embedding dtype,
  and per-request downsampling reaches vision encoding. The 4.7 architecture
  itself preserves canvas positions on cached-image and decode calls.
- The processor dependencies are unchanged between the tag and the selected
  commit. `install_auto_processor_patch`, `load_chat_template`, and the
  `InputEmbeddingsFeatures` position/delta fields already exist in 0.7.2.
- The newer expert `norm_topk_prob` option is absent from both real configs;
  pinned always-normalized routing matches the default. Neither checkpoint
  needs a new decoder interface. Existing runtime kernels remain pinned.
- The two curated quant repositories' weight indexes use canonical
  `language_model.model.*` / `switch_mlp.*` tensor names and contain no MTP
  tensors. Newer fused-expert quantization helper changes are unnecessary
  for those artifacts. Raw backbone MTP weights are discarded by sanitization.

The only architecture-source deviations are license/provenance headers,
marked import redirects, and marked subset extraction. There is no global
replacement of the pinned 4.6 or Qwen modules.

## Registration and routing

`MLXMultimodalLM.load` installs the binding lazily after reading the model
config and before loading the model/processor. The helper binds the package
as `mlx_vlm.models.minicpmv4_7` through `sys.modules` and the parent module,
reusing the existing architecture binding seam. Package import installs the
composable AutoProcessor hook. The pinned image-message registry receives
upstream's `LIST_WITH_IMAGE_FIRST` entry. Native serving renders ordered
media parts directly through the checkpoint template; the existing MLLM
video-frame processing policy is unchanged.

BYOM includes the type both with and without an installed vision extra.
The API name detector recognizes MiniCPM-V, and local metadata classifies
repackaged checkpoints. Dense/MoE hybrid flags follow the corresponding
Qwen3.5 serving policies, with speculative decoding disabled. The real
checkpoint templates use parameterized XML tool calls and Qwen thinking;
the existing `qwen3_coder_xml` and `qwen3` parsers are reused.

## Verification and handoff

No model weights were downloaded. Real config/template fixtures are pinned
in `tests/fixtures/minicpmv4_7/README.md`. Download sizes were measured with
`scripts.gen_model_sizes._size_one` for only the two new aliases.

CPU contracts: **4,425 passed, 7 skipped**:

```sh
.venv/cpu/bin/python -m pytest tests/test_minicpmv4_7_contract.py tests/test_aliases_contract.py tests/test_model_sizes.py tests/test_byom_preflight.py tests/test_model_auto_config.py tests/test_alias_hybrid_classification.py tests/test_atomic_model_catalog.py -q
```

Additional detection coverage: **95 passed, 114 deselected**:

```sh
.venv/cpu/bin/python -m pytest tests/test_api_utils.py tests/test_model_metadata.py -k 'mllm or vlm or multimodal' -q
```

Apple-only runtime/provenance files were requested locally; **six modules
skipped** because the CPU environment has no MLX. Pytest returns 5 when all
requested modules skip. The host's MLX import independently fails with
`[metal::load_device] No Metal device available` in this sandbox.

```sh
.venv/cpu/bin/python -m pytest tests/test_minicpmv4_7_mlx.py tests/test_mlx_vlm_vendored_apc_support.py tests/test_mlx_vlm_vendored_inputs.py tests/test_mlx_vlm_vendored_generate.py tests/test_mlx_vlm_vendored_speculative.py tests/test_mlx_vlm_vendored_drafters.py -q
```

The new runtime test file is in the CI Apple list. It covers real-config
parsing, nested reparsing, loader binding, tiny dense/MoE image forwards,
cached features, both downsampling modes, decode, quantized-text dtype,
local processor loading, chat templates, MTP filtering, and strict synthetic
weight round trips. Vector/caller owns running these on Metal and verifying
both real checkpoints, image/video requests, cancellation and relaunch before
removing the experimental flags. No performance claims are made here.

Post-review provenance/BYOM rerun: **120 passed, 2 skipped**:

```sh
.venv/cpu/bin/python -m pytest tests/test_minicpmv4_7_contract.py tests/test_byom_preflight.py -q
```

The live installed 0.7.2 inventory check also ran without importing Metal:
**1 passed, 115 deselected**:

```sh
.venv/bin/python -m pytest tests/test_byom_preflight.py -k vlm_snapshot_matches_pinned_runtime -q
```

Lint, formatting and diff whitespace checks passed. Vendored files retain
upstream formatting under the repository's existing formatter exclusions.

```sh
.venv/bin/python -m ruff check rapid_mlx/models/minicpmv4_7.py rapid_mlx/models/mlx_vlm_vendored/models/minicpmv4_7 rapid_mlx/models/mlx_vlm_vendored/__init__.py rapid_mlx/models/mllm.py rapid_mlx/model_auto_config.py rapid_mlx/api/utils.py rapid_mlx/byom/preflight.py rapid_mlx/byom/_vlm_model_types.py tests/test_minicpmv4_7_contract.py tests/test_minicpmv4_7_mlx.py tests/test_byom_preflight.py
.venv/bin/python -m ruff format --check rapid_mlx/models/minicpmv4_7.py rapid_mlx/models/mlx_vlm_vendored/models/minicpmv4_7 rapid_mlx/models/mlx_vlm_vendored/__init__.py rapid_mlx/models/mllm.py rapid_mlx/model_auto_config.py rapid_mlx/api/utils.py rapid_mlx/byom/preflight.py rapid_mlx/byom/_vlm_model_types.py tests/test_minicpmv4_7_contract.py tests/test_minicpmv4_7_mlx.py tests/test_byom_preflight.py
git diff --check
```
