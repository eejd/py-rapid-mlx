"""Offline Super VL text conversion; invoke under the shared large-model lock."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, quantize_model, save_config, save_model

from rapid_mlx.model_load_errors import load_model_checked
from rapid_mlx.models.nemotron_h_super import text_config

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
ASSETS = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
}


def quantization_predicate(model, bits):
    preserve = model.quant_predicate

    def predicate(path, module):
        if not preserve(path, module):
            return False
        if "embedding" in path:
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        if bits == 3:
            expert = ".switch_mlp." in path
            return {
                "bits": 3 if expert else 4,
                "group_size": 128 if expert else 64,
                "mode": "affine",
            }
        return True

    return predicate


def artifact_size(path):
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file() and ".cache" not in f.relative_to(path).parts
    )


def model_card(bits, size):
    recipe = (
        "4-bit affine, group size 64"
        if bits == 4
        else "mixed: routed experts 3-bit/group-128, other eligible projections 4-bit/group-64"
    )
    return f"""---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: {BASE_MODEL}
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, rapid-mlx, nemotron, apple-silicon]
---

# NVIDIA Nemotron 3.5 Super VL 120B-A12B — MLX {bits}bit (text)

Text conversion of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}).

- Quantization: {recipe}; embeddings 8-bit/group-64. Output head,
  routers and latent projections retain checkpoint precision.
- Artifact size: {size / 1e9:.3f} GB. Recommend **96 GB+** unified memory.
  {"64 GB Macs may try short contexts with other applications closed; this is not a 64 GB hardware qualification." if size <= 50e9 else "Allow headroom beyond the weights for prefill, caches, macOS and applications."}
- Text only: the vision tower, image projector and unused MTP head are omitted.
- Declared context limit: 1,048,576 tokens; this does not promise that the full
  window fits the recommended memory. Validation scope and measured results
  are recorded in the Rapid-MLX handoff.

## Requirements

Requires Rapid-MLX PR #4679 and follow-up. The config retains
`nemotron_h` and opts into Rapid's reviewed precision adapter. Stock loaders
do not apply that adapter; only Rapid-MLX is validated for this conversion.

```sh
rapid-mlx serve rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-{bits}bit
```

Reasoning and XML tool calling use the original tokenizer/chat template.
MTP speculative decoding and image input are not enabled.

## License

Original model by NVIDIA under [OpenMDW 1.1](https://openmdw.ai/license/1-1/).
This conversion inherits the original model's license terms.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(3, 4))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = text_config(json.loads((args.source / "config.json").read_text()))
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    missing = [name for name in shards if not (args.source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint download incomplete: {sorted(missing)}")
    start = time.monotonic()
    mx.set_default_device(mx.cpu)
    scratch = Path("/private/tmp/nemotron-super-vl-convert")
    scratch.mkdir(exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        view = Path(temporary)
        (view / "config.json").write_text(json.dumps(config, indent=2))
        for name in shards:
            (view / name).symlink_to((args.source / name).resolve())
        model, config = load_model_checked(load_model, view, lazy=True)
        model, config = quantize_model(
            model,
            config,
            64,
            args.bits,
            quant_predicate=quantization_predicate(model, args.bits),
        )
        save_model(args.output, model, donate_model=True)
        save_config(config, args.output / "config.json")
    for name in ASSETS:
        if (args.source / name).is_file():
            shutil.copyfile(args.source / name, args.output / name)
    card = args.output / "README.md"
    card.write_text(model_card(args.bits, artifact_size(args.output)))
    card.write_text(model_card(args.bits, artifact_size(args.output)))
    result = {
        "bits": args.bits,
        "output": str(args.output),
        "bytes": artifact_size(args.output),
        "wall_seconds": time.monotonic() - start,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
