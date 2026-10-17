<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Desktop no longer quits when you open a generated video**: clicking a finished video's preview now plays it in the app instead of crashing Rapid-MLX Desktop.

- **Cleaner tool-call replies**: unfinished tool payloads stay out of visible replies when arguments contain literal tool markers or a forced tool call is returned.
- **XML tool-call examples stay visible**: fenced code-block examples remain content even when they name an available tool, including in streamed replies; real calls after a closed fence still execute.
- **XML tool-call examples with CR line endings stay visible**: fenced examples written with bare CR or CRLF line endings remain content instead of running as tool calls, in streamed and non-streamed replies.
- **TensorFold lane setting no longer ignored**: setting `RAPID_MLX_TENSORFOLD_LANES` above 1, or to an invalid value, on a model that is not served through a qualified TensorFold profile now stops `rapid-mlx serve` at startup with an error instead of being silently ignored.
