<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

## Fixes

- **Quick Look in Local Search no longer crashes the app**: choosing Quick Look on a search result (or pressing Space) opens the preview reliably instead of occasionally quitting the app.

- **Concurrent long prompts queue when memory is tight**: prefill runs in memory-fitting groups so bursts no longer abort with out-of-memory errors.

- **The server recovers from an out-of-memory error without a restart**: after a batch fails with insufficient memory, the memory it held is released, so later requests succeed again. Previously `/v1/cache/clear` could not free it and only a restart restored capacity.

- **Fresh installs no longer break image generation**: a new release of the imaging library (Pillow 13.0.0) fails to import on supported Python versions, so a fresh install could pull it and the image runtime would fail to load. Rapid-MLX now keeps Pillow below 13 until a compatible release is available.
- **Logprobs requests work with MTP speculative decoding**: requests for token log probabilities no longer fail with an HTTP 500 or crash the server when MTP speculative decoding is active.
- **Desktop no longer quits when you open What's New after an update**: clicking "See what's new" on the post-update banner opens the release notes and dismisses the banner instead of crashing the app. The "Star on GitHub" prompt had the same defect and is fixed too.
