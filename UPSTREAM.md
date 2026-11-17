# Work carried on `local` that is not in raullenchai/Rapid-MLX

`local` = upstream `v1.0.4` (`52642bde6c`), re-cut on 2026-11-17. `main` mirrors
`raullenchai/Rapid-MLX` `main` (fast-forward only).

The previous `local` (v0.16.2 + six topics, last tag `v0.16.2-local05`) is kept on
`archive/local-v0.16.2`. The tags `v0.16.2-local01`..`local05` are unchanged.

No tag is cut on this base yet; the port `py-rapid-mlx-local` stays on `v0.16.2-local05`.

## Topics pending re-evaluation

| Topic | Branch | Old PR | Issue | Status |
|---|---|---|---|---|
| Do not add a second BOS to chat prompts whose template already emits it | `topic/chat-template-no-double-bos` | #3 | #1 | re-evaluation: see eejd/py-rapid-mlx#27 |
| MiniCPM5 tool parser: typed list/dict arguments, `<`/`&` in values without CDATA | `topic/minicpm-tolerant-typed-params` | #7 | #6 | re-evaluation: see eejd/py-rapid-mlx#27 |
| `--tool-use-system-suffix {on,off}` (default off) | `topic/tool-use-system-suffix-flag` | #9 | #8 | re-evaluation: see eejd/py-rapid-mlx#27 |
| `ignore_eos` request field | `topic/ignore-eos` | #11 | #10 | re-evaluation: see eejd/py-rapid-mlx#27 |
| mlx-lm 0.32 stop-attribute compatibility | `topic/mlx-lm-032-stop-attrs` | #24 | - | re-evaluation: see eejd/py-rapid-mlx#27 |
| mlx-vlm qualified version range | `topic/mlx-vlm-qualified-range` | #25 | - | re-evaluation: see eejd/py-rapid-mlx#27 |
