# Nemotron 3.5 Super VL text qualification

Owner: Vector. Local qualification only; no upload or release authorization.
Architecture and pending gates: [decision](../decisions/2026-10-15-nemotron-super-vl.md).

## Environment and workload

Studio: M3 Ultra, 28 CPU / 60 GPU cores, 256 GiB unified memory, macOS 26.5.2.
Python 3.12.14, Rapid-MLX 0.16.1 checkout `07764546c`, based on `7118aaf72`;
MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2, Transformers 5.15.1.
The BF16 source revision is `a1ccf94d69377bf82c4db39798aac28447d9c16f`.
All 88 source files pass content-hash verification; all 64 source weight
shards and all 14 converted shards pass safetensors extent checks.

The generator benchmark loads through Rapid's precision-adapted text loader.
It repeats tokenized `The quick brown fox jumps over the lazy dog. ` to exactly
1,024 raw prompt tokens, then greedily decodes 64 tokens, with prefill chunks
of 256. Two repetitions use the resident model after eight short smoke prompts.
Every model operation holds `lockf -k /var/tmp/rapid-mlx-large-model.lock`.
Each repetition creates a fresh inference cache. Artifacts reside on RTL-2T.

## Four-bit result

The conversion takes 321.02 seconds and yields 4.609 effective bits per weight.
The output is `/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-4bit-g64`.
Weights total 69,523,298,304 bytes. The complete artifact is 69,540,668,652 bytes;
the download-size generator's file filter counts **69,540,654,086 bytes**.
The latter excludes README and the Jinja chat template. Config and tokenizer
metadata contain no remote-code selectors or image-processor class.

| Repetition | Prompt tokens/s | Decode tokens/s | Process peak memory (GB) |
| --- | ---: | ---: | ---: |
| 1 | 466.12 | 48.56 | 72.680 |
| 2 | 467.59 | 48.39 | 72.680 |

Eight greedy smoke prompts cover factual recall, arithmetic, code, translation,
light colors, a sequence, explanation and name recall. Examples: `Paris`, `391`,
`Buenos días.`, primary light colors red/green/blue, and `Your name is Ada.`
Sampling is limited to 16 greedy tokens per prompt (or EOS); this is a smoke
check, not a quality benchmark. Raw EOS tokens in the direct model evaluation
are retained as token-level evidence. HTTP token filtering remains a separate gate.

The tiny evaluation-only paged BF16 reference matches the resident BF16 model
bit-exactly for full and cached logits. Full-checkpoint BF16 parity and HTTP
serving gates are pending the shared lock. Mixed three-bit has not started.
No image or long-context qualification is claimed. Recommend 96 GB+ memory;
72.68 GB is the measured model-process peak and excludes other applications.

## Reproduction and evidence

Evidence and evaluation-only harnesses are archived at
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/evidence/`.
Use this checkout with the task-local Python environment and offline source cache.

```sh
output_root=/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$output_root/text-4bit-g64" --name 4bit --tokens 16 --bench
```

The harness writes results to `/private/tmp/nemotron-super-vl-evidence/`.
`eval-4bit.json` contains both benchmark repetitions and eight token/logit traces;
`artifact-4bit.json` records the exact catalog file set. `source-hashes.json`
records all 88 validated source hashes. Performance numbers are specific to
this hardware, software, model recipe and workload.
