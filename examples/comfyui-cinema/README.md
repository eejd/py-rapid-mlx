# The Last Lantern — local AI microfilm

**Completed on 2026-10-11:** six actual LTX-2.5 motion/audio shots were generated
on an M5 Max with 36 GB unified memory and edited into
[the final film](media/the-last-lantern.mp4). The six Qwen keyframes were generated
previously on an M3 Ultra with 256 GB. The video run used seven generations:
one lunar shot was rejected because the boat disappeared, then rerun with a
stronger continuity prompt. See [verified video evidence](verified-video-run.json)
and [keyframe execution evidence](verified-keyframes.json).

Each selected source clip is 704×448, 97 frames at 24 fps. Requests specified
704×480; the pinned runtime encoded 448 pixels in height. The final 1920×1080
film is an editorial upscale with title cards and native generated audio.
Generated clips and the final film were fully decoded, and sampled frames were
visually reviewed. This is demo evidence, not a controlled hardware benchmark.

The film follows a red paper boat carrying an amber lantern through six
worlds: a rainy station, a cloud ocean, a lunar ruin, an underwater bubble, a
mushroom forest and a quiet dawn. Each shot targets 97 frames at 24 fps, about
4.04 seconds. Six generated shots provide about 24 seconds before editorial
cards. Qwen-Image supplies the first frame; LTX-2.5 generates motion and audio.
The repeated object and visual description are a continuity strategy; identity
consistency across independent generations still needs human review.

## Architecture

```text
ComfyUI RapidMLXImage → SaveImage
                             ↓ saved keyframe, second pass
                  LoadImage → RapidMLXVideo → SaveVideo
                                     ↓
                    Rapid-MLX /v1/videos → LTX-2.5 MLX
```

ComfyUI runs with `--cpu`; PyTorch carries image pixels. MLX inference runs in
separate Rapid-MLX servers. The video node returns ComfyUI's native `VIDEO`
object from the downloaded MP4, retaining generated audio. No cloud video node
is used. The pinned ComfyUI source inspected for this adapter was
`0df64eb242b7c5759c3e86afd5d1846d923b1033`.

## Setup

Use the installation and image-server instructions in
[the travel demo](../comfyui-travel/README.md), and the audited LTX runtime setup
in [the video guide](../../docs/guides/video-generation.md#ltx-25).
That guide pins the runtime to
`57952288076766abe27dda3a774b2c24f7346977`; an arbitrary newer runtime is not a
substitute. Review the model's license before publishing generated content.
Check the shared Hugging Face cache before downloading weights.

Install the video extra in the Rapid environment too; the image-only setup
does not include all image-to-video dependencies:

```bash
.venv/bin/pip install -e '.[video]'
```

Link both custom-node directories into your ComfyUI checkout, using absolute
paths appropriate to your checkout:

```bash
ln -s /absolute/path/Rapid-MLX/examples/comfyui-travel/custom_nodes/rapid_mlx \
  /absolute/path/ComfyUI/custom_nodes/rapid_mlx
ln -s /absolute/path/Rapid-MLX/examples/comfyui-cinema/custom_nodes/rapid_mlx_video \
  /absolute/path/ComfyUI/custom_nodes/rapid_mlx_video
```

Restart ComfyUI with `--cpu --listen 127.0.0.1 --port 8189` to register the nodes.
Use `RAPID_MLX_API_KEY` in the ComfyUI process if the Rapid servers require it.
Keep the servers bound to localhost.

First run the image pass with Qwen-Image 2.1 on port 18427:

```bash
python3 examples/comfyui-cinema/batch.py keyframes \
  --output examples/comfyui-cinema/media --limit 1
```

Inspect that first keyframe before increasing `--limit` to 6. Before starting the
video pass, plan model residency around available unified memory; avoid other
inference jobs. Start the audited LTX runtime with durable server artifacts:

```bash
RAPID_MLX_LTX25_RUNTIME=/absolute/path/ltx-2-mlx/.venv/bin/ltx-2-mlx \
  rapid-mlx serve ltx-2.5-mlx-q8 --host 127.0.0.1 --port 18428 \
  --video-output-dir /absolute/path/Rapid-MLX/examples/comfyui-cinema/media/server-jobs
python3 examples/comfyui-cinema/batch.py shots \
  --output examples/comfyui-cinema/media --limit 1
```

The verified run reused a complete shared-cache snapshot with
`HF_HUB_OFFLINE=1`. This checkout's CLI cache probe rejected the complete LTX
component cache. After verifying all 23 files, the run used the existing server
module entry point instead of changing the cache or engine:

```bash
HF_HUB_OFFLINE=1 RAPID_MLX_LTX25_RUNTIME=/absolute/path/ltx-2-mlx/.venv/bin/ltx-2-mlx \
  python -m rapid_mlx.server --model ltx-2.5-mlx-q8 \
  --host 127.0.0.1 --port 18428 --video-output-dir /absolute/path/cinema/media/server-jobs
```

Review the first video for actual movement, boat continuity and usable sound,
then increase `--limit` to 6. The video node requests 704×480, 97 frames, 24 fps;
these fit the current server's shape and pixel-frame limits. Check
`GET /v1/videos/capabilities` before changing them. LTX-2.5's distilled pipeline
does not accept CFG guidance or negative prompts.

`workflow.json` is the canvas workflow exported from the running ComfyUI; drag it
into the UI after installing both custom nodes. `workflow-api.json` is an **API prompt graph**, submitted as the `prompt` member
of `POST /prompt`; it is not a canvas export to drag into the UI. It illustrates
the full graph. The batch script uses two passes so saved keyframes can be
reviewed before animation. Keep ComfyUI's output directory: the second pass
loads the first pass's saved artifact from that directory.

## Recovery and evidence

The batch runner records each Comfy prompt ID before polling. Rerunning with the
same output directory resumes that prompt. Changed workflows require a fresh
output directory. If ComfyUI restarts and loses its history, inspect the saved
prompt IDs and Rapid job receipts before deciding to resubmit.

The video node stores the Rapid job ID before polling, and reuses it for an
identical image and request. Receipts and downloaded MP4s live under
`ComfyUI/output/rapid-cinema-jobs/`. An interrupted ComfyUI client does not
cancel its server job; rerunning the same node resumes it. Failed jobs remain
recorded for diagnosis. If the initial POST loses its response, inspect server
jobs before retrying: the API does not provide an idempotency guarantee.

Both layers record output SHA-256 hashes. Completed downloads are reused only
when their hashes match. Do not run duplicate copies of the batch concurrently.

CPU-only checks, with no MLX import or model loading:

```bash
python3 -m unittest discover -s examples/comfyui-cinema -p 'test_*.py' -v
ruff check examples/comfyui-cinema
```

Blog and tweet files are unpublished drafts updated with this run. Record hardware,
model/runtime revisions, timings including failed attempts, artifact hashes and
editorial transformations. Label the result as AI generated and distinguish
native generated resolution from any editorial upscale.

Primary references: [converted model and runtime](https://huggingface.co/MrMofer/ltx-2.5-mlx-q8),
[ComfyUI video contracts](https://github.com/Comfy-Org/ComfyUI/blob/0df64eb242b7c5759c3e86afd5d1846d923b1033/comfy_extras/nodes_video.py).

After reviewing all six generated clips, create the film with deterministic
editorial titles, preserved shot audio, and a clearly labeled 1080p upscale:

```bash
python3 examples/comfyui-cinema/render.py \
  --input examples/comfyui-cinema/media \
  --output examples/comfyui-cinema/media/the-last-lantern.mp4
```

The renderer requires Pillow and FFmpeg. It refuses missing shots or missing
native audio instead of silently substituting stills or a silent soundtrack.
It normalizes each shot's generated audio toward -23 LUFS with a -2 dBTP ceiling
and brief edge fades; it does not add a soundtrack.
The film starts with the generated departure shot before its title card.
Its six-second title/outro cards bring the intended film length to about 30
seconds. The final edit was exercised with the six selected LTX-generated clips. Generated
audio varies by scene; normalization preserves the source rather than adding
replacement audio.
