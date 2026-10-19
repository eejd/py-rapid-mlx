# Work carried on `local` that is not in raullenchai/Rapid-MLX

`local` = upstream release tag + the commits below. `main` mirrors `raullenchai/Rapid-MLX`
`main` (fast-forward only). `local` is based on `v0.16.2`, the release the `py-rapid-mlx` port
ships; upstream has since tagged `v0.16.3`.

| What | Branch | Upstream status |
|---|---|---|
| Do not add a second BOS to chat prompts whose template already emits it (Llama 3: scheduler admission prefilled one token more than the route's own context-guard count): one BOS-aware `encode_prompt` used by scheduler admission, the prefix-boundary probe, guided generation and `count_prompt_tokens` | `topic/chat-template-no-double-bos` (based on `local`; the scheduler tokenization is unchanged on upstream `main`, read not run) | not submitted; nothing is posted upstream before the quantbench + agentbench soak (ADR-0008); eejd/py-rapid-mlx#1 |
