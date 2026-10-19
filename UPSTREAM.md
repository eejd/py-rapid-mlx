# Work carried on `local` that is not in raullenchai/Rapid-MLX

`local` = upstream release tag + the commits below. `main` mirrors `raullenchai/Rapid-MLX`
`main` (fast-forward only). `local` is based on `v0.16.2`, the release the `py-rapid-mlx` port
ships; upstream has since tagged `v0.16.3`.

| What | Branch | Upstream status |
|---|---|---|
| Do not add a second BOS to chat prompts whose template already emits it (Llama 3: scheduler admission prefilled one token more than the route's own context-guard count): one BOS-aware `encode_prompt` used by scheduler admission, the prefix-boundary probe, guided generation and `count_prompt_tokens` | `topic/chat-template-no-double-bos` (based on `local`; the scheduler tokenization is unchanged on upstream `main`, read not run) | not submitted; nothing is posted upstream before the quantbench + agentbench soak (ADR-0008); eejd/py-rapid-mlx#1 |
| MiniCPM5 tool parser (`--tool-call-parser minicpm`): an array/object argument is returned as a typed list/dict (the template and the model write Python literals, `['a', 'b']`; it came back as the string, and the route's schema validation answered HTTP 400 `Expected array, got str`: 8 of 20 `tool_arg` cases of MiniCPM5-2B-4bit); a value holding `<` or `&` without CDATA is data (it was parsed with `xml.etree`, so the call stayed text: 4 of 20) | `topic/minicpm-tolerant-typed-params` (based on `local`; `minicpm_tool_parser.py` and `api/tool_calling.py` are byte-identical on upstream `main` 2026-10-19, API read) | not submitted; nothing is posted upstream before the quantbench + agentbench soak (ADR-0008); eejd/py-rapid-mlx#6 |
