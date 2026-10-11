# Verified local microfilm run

Completed on 2026-10-11. Six selected LTX-2.5 image-to-video shots were generated
through ComfyUI and Rapid-MLX on an Apple M5 Max with 36 GB unified memory.
The Qwen-Image 2.1 keyframes were generated previously on an M3 Ultra with 256 GB.

The final film is **30.267 seconds, 1920×1080, H.264, 24 fps, 726 frames**, with
stereo 48 kHz AAC audio. It contains six generated shots plus six seconds of
title/outro cards. The source clips are 704×448, 97 frames each; requests used
704×480, but the pinned runtime encoded 448 pixels in height. The 1080p output
is an editorial upscale, explicitly labeled AI generated.

| Selected shot | Seed | Server job wall seconds |
| --- | ---: | ---: |
| 01-departure | 7300 | 55 |
| 02-cloud-sea | 7301 | 47 |
| 03-moon-canal | 7302 | 48 |
| 04-deep-blue | 7303 | 47 |
| 05-forest | 7304 | 47 |
| 06-home | 7305 | 47 |

Job wall time includes runtime/model loading and encoding, measured from the
server's integer-second created/completed timestamps. These are demo-session
measurements, not a controlled hardware comparison. Seven model generations
ran in total: the initial lunar shot (47 seconds) lost its boat by the end and
was retained under local `media/rejected/`; the selected rerun strengthens the
foreground continuity prompt. Two startup/preflight problems occurred before
any generation: cache recognition in the CLI and an absent video dependency.
The README records the working server entry point and installation requirement.

All six selected clips and the final film fully decoded without FFmpeg errors.
Each shot's start, middle and end were visually reviewed, along with the final
title cards and representative film frames. Audio tracks were checked for
presence and signal levels. The editor preserves native generated audio,
normalizes toward -23 LUFS with a -2 dBTP ceiling, applies brief edge fades and
uses silent title cards. It adds no replacement soundtrack. The final audio
mean/max levels were -26.8/-8.3 dBFS. No listening-panel or subjective audio
quality claim is made.

Four video-client tests, Ruff and whitespace checks passed. The final video and
all six source hashes were verified again after copying back to the workspace.
[verified-video-run.json](verified-video-run.json) records exact runtime and
model revisions, package versions, job IDs, file hashes and actual media shape.
The renderer and storyboard are bound by SHA-256 too.

Final SHA-256: `9feecd28eda3b2eb909b07ee653a870f0357490f40a2d94a6d1e4bdb6b557ccb`.

Final MP4, selected originals, receipts, preview sheets and rejected candidate
are local ignored `media/` artifacts; they are not included in the Git push.
Blog and tweet files remain unpublished drafts. Scope is confined to this demo;
no engine, server API, Desktop, CI or release changes.
