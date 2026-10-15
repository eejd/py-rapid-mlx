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
are retained as token-level evidence. HTTP token filtering is checked by the separate serving gates below.

## BF16 comparison and HTTP gates

The evaluation-only paged BF16 reference matches the tiny resident BF16 model
bit-exactly for full and cached logits. The full checkpoint's eight-prompt
reference completes in 640.59 seconds. It uses exact source tensors and native
expert math without materializing the 241 GB text checkpoint on the GPU.
The comparison measures first-token logits and up to 16 independently greedy
tokens per prompt, with thinking disabled.

| Prompt | Same greedy first token | First-token top-10 overlap | Matching greedy prefix |
| --- | --- | ---: | ---: |
| France capital | No | 8/10 | 0 |
| 17 × 23 | Yes | 9/10 | 4 |
| Python addition | No | 9/10 | 0 |
| Spanish greeting | Yes | 8/10 | 5 |
| Primary light colors | Yes | 10/10 | 7 |
| Sequence continuation | Yes | 10/10 | 16 |
| Sky explanation | Yes | 10/10 | 16 |
| Ada recall | Yes | 10/10 | 6 |

Strict greedy first-token agreement is **6/8**; top-10 overlap totals **74/80**
(92.5%). Both first-token mismatches select another token tied at the BF16
maximum, giving **8/8 tie-aware first-token agreement**. Actual greedy IDs are
compared rather than the first entry of an argsorted top-10 list. Five of eight
sampled sequences match exactly. Other differences are wording or formatting:
`Paris` versus `The capital of France is Paris.`, a different code preamble,
and bold emphasis around primary colors. The clipped code sample does not
establish complete-function correctness. This is not independent full-model
NVIDIA-runtime parity or a quality benchmark.

All **seven HTTP requests pass**, with automatic parser selection:

- Multi-turn: `Your name is Ada, and your favorite color is teal.`
- Thinking off: visible answers with no reasoning content or control tokens.
- Thinking on: separate nonempty reasoning and `17 × 23 = **391**` as content.
- Non-stream and stream: `get_weather({"city":"Paris"})`, with `tool_calls`
  finish reasons and valid JSON arguments.
- Both tool-result follow-ups: `The current weather in Paris is sunny with a
  temperature of 18°C.`

No added special tokens, thinking delimiters or XML tool markers leak into
visible content or reasoning. The server shuts down, clears its caches and
releases port 8479. Its SIGTERM thread dump is the normal signal-observability
hook, followed by graceful engine/application shutdown; the harness exits zero.

The final **95 focused tests** and **3,774 catalog/README tests** pass. Ruff and
format checks pass. Changed-line coverage against
`origin/vector/nemotron-35-bolt` is **141/141 lines, 100%**. The branch also
includes the committed optional-top-k validation guard from `d0ee60130`; it does
not change this model's valid top-22 configuration or the measured norm math.

Mixed three-bit is deferred after roughly 70 minutes spent waiting for unrelated
validators. Its output directory was not created and no size/alias is invented.
Vector's next action is conversion with `--bits 3`, followed by the same gates,
reusing the archived BF16 reference. No image or long-context qualification is
claimed. Recommend 96 GB+ memory; the 72.68 GB process peak excludes other apps.

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

The corresponding reference and HTTP gates use the archived harnesses:

```sh
source_snapshot="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/snapshots/a1ccf94d69377bf82c4db39798aac28447d9c16f"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$source_snapshot" --name bf16 --tokens 16 < /dev/null
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/serve.py" \
  --model "$output_root/text-4bit-g64" --name 4bit --port 8479
.venv/bin/python "$output_root/evidence/compare.py"
```

The HTTP harness starts a loopback server with `--text-only`, disabled disk
and prefix caches, prefill step size 256 and a 768-token response cap. It
checks name/color recall across turns, reasoning on/off, XML tool calls and
tool-result follow-ups in both stream modes, finish reasons, valid JSON
arguments and absence of control tokens in visible content/reasoning. It
terminates only the server process group it started. The BF16 harness rejects
checkpoint custom code and uses the local fast tokenizer fallback.
